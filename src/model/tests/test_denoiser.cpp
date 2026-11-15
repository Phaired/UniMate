// Parity test: um::Denoiser (CPU backend) vs the upstream PyTorch goldens in
// tests/golden/denoiser/*.umtf (tools/python/denoiser_golden.py).
//
//   um_model_tests [case ...]      (default: eagle shark batch2)
//
// Outputs (forward cond / force_mask, CFG velocity) must match within
// kTolOut max-abs; intermediates are reported per stage (and loosely checked)
// so a mismatch can be localised. Exit 77 (= skipped) when the converted
// weights or the goldens are missing.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "core/umtf.h"
#include "model/denoiser.h"

namespace fs = std::filesystem;
using namespace um;

namespace {

constexpr float kTolOut = 1e-4f;    // outputs (O(1) normalised features)
constexpr float kTolInter = 1e-3f;  // intermediates: relative to their RMS

struct Err {
  double max_abs = 0, ref_rms = 0, ref_max = 0;
  int64_t masked_bad = 0;
};

Err compare(const std::vector<float>& got, const std::vector<float>& ref) {
  Err e;
  double ss = 0;
  int64_t n = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float r = ref[i], g = got[i];
    if (std::isinf(r)) {  // masked (-inf upstream, large negative here)
      if (!(g < -1e29f)) ++e.masked_bad;
      continue;
    }
    e.max_abs = std::max(e.max_abs, static_cast<double>(std::fabs(g - r)));
    if (!std::isfinite(g)) e.max_abs = INFINITY;
    ss += static_cast<double>(r) * r;
    e.ref_max = std::max(e.ref_max, static_cast<double>(std::fabs(r)));
    ++n;
  }
  e.ref_rms = n ? std::sqrt(ss / n) : 0;
  return e;
}

int g_fail = 0;

bool report(const std::string& what, const Tensor& got, const HostArray& ref, double tol_abs) {
  if (got.shape() != ref.shape) {
    std::printf("  %-28s SHAPE MISMATCH got %s want %s\n", what.c_str(),
                shape_str(got.shape()).c_str(), shape_str(ref.shape).c_str());
    ++g_fail;
    return false;
  }
  Err e = compare(got.to_vector(), ref.to_f32());
  const bool ok = e.max_abs <= tol_abs && e.masked_bad == 0;
  std::printf("  %-28s max_abs %.3e  (ref rms %.3f, max %.3f)%s%s\n", what.c_str(), e.max_abs,
              e.ref_rms, e.ref_max, e.masked_bad ? "  MASK MISMATCH" : "", ok ? "" : "  <-- FAIL");
  if (!ok) ++g_fail;
  return ok;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void run_case(Backend& be, Denoiser& model, const fs::path& path) {
  std::printf("== %s\n", path.filename().string().c_str());
  UmtfFile g = UmtfFile::load(path.string());
  const auto& meta = g.meta();

  ArrayDict cond;
  for (const auto& n : g.names())
    if (n.rfind("cond.", 0) == 0) cond[n.substr(5)] = g.get(n);
  auto t0 = std::chrono::steady_clock::now();
  model.set_condition(cond);
  std::printf("  set_condition: %.1f ms, %d ops, B=%d\n", ms_since(t0), model.last_op_count(),
              model.batch());

  Tensor x = g.get("x").to_tensor(be);
  const std::vector<float> ts = g.get("t").to_f32();
  const float cfg = meta["cfg_scale"].get<float>();
  const int inter_i = meta["inter_t_index"].is_null() ? -1 : meta["inter_t_index"].get<int>();

  double t_fwd = 0, t_vel = 0;
  int n_fwd = 0, n_vel = 0, ops_fwd = 0, ops_vel = 0;
  for (size_t i = 0; i < ts.size(); ++i) {
    const std::string si = std::to_string(i);
    std::printf(" t=%.3f\n", ts[i]);
    if (static_cast<int>(i) == inter_i) {
      DenoiserTrace tr;
      Tensor out = model.forward_traced(x, ts[i], false, &tr);
      for (const auto& n : g.names()) {
        if (n.rfind("inter.", 0) != 0) continue;
        const std::string k = n.substr(6);
        if (k == "y_uncond") continue;
        auto it = tr.find(k);
        if (it == tr.end()) {
          std::printf("  %-28s (not traced)\n", ("inter." + k).c_str());
          continue;
        }
        const HostArray& ref = g.get(n);
        // tolerance relative to the stage's own RMS
        std::vector<float> rv = ref.to_f32();
        double ss = 0; int64_t cnt = 0;
        for (float v : rv) if (std::isfinite(v)) { ss += double(v) * v; ++cnt; }
        const double rms = cnt ? std::sqrt(ss / cnt) : 1.0;
        report("inter." + k, it->second, ref, k == "final" ? kTolOut : kTolInter * std::max(rms, 1.0));
      }
      (void)out;
    }
    auto a = std::chrono::steady_clock::now();
    Tensor oc = model.forward(x, ts[i], false);
    std::vector<float> sync = oc.to_vector();
    t_fwd += ms_since(a); ++n_fwd; ops_fwd = model.last_op_count();
    report("out_cond." + si, oc, g.get("out_cond." + si), kTolOut);

    Tensor ou = model.forward(x, ts[i], true);
    report("out_uncond." + si, ou, g.get("out_uncond." + si), kTolOut);

    a = std::chrono::steady_clock::now();
    Tensor v = model.velocity(x, ts[i], cfg);
    sync = v.to_vector();
    t_vel += ms_since(a); ++n_vel; ops_vel = model.last_op_count();
    report("velocity." + si + " (cfg " + std::to_string(cfg).substr(0, 4) + ")", v,
           g.get("velocity." + si), kTolOut);
  }
  std::printf("  forward : %d backend ops, %.1f ms avg (B=%d)\n", ops_fwd, t_fwd / n_fwd, model.batch());
  std::printf("  velocity: %d backend ops, %.1f ms avg (batched 2B=%d)\n", ops_vel, t_vel / n_vel,
              2 * model.batch());
}

}  // namespace

int main(int argc, char** argv) {
  const fs::path root = UM_ROOT_DIR;
  const fs::path weights = root / "models" / "converted" / "unimate_uniml3d_f60_v3.umtf";
  const fs::path gdir = root / "tests" / "golden" / "denoiser";
  std::vector<std::string> cases;
  for (int i = 1; i < argc; ++i) cases.push_back(argv[i]);
  if (cases.empty()) cases = {"eagle", "shark", "batch2"};

  if (!fs::exists(weights)) {
    std::printf("SKIP: %s missing (run tools/python/denoiser_export.py)\n", weights.string().c_str());
    return 77;
  }
  auto be = make_cpu_backend();
  auto t0 = std::chrono::steady_clock::now();
  Denoiser model(*be, weights.string());
  std::printf("loaded %s in %.0f ms (max_joints %d, max_frames %d, feature_len %d)\n",
              weights.filename().string().c_str(), ms_since(t0), model.max_joints(),
              model.max_frames(), model.feature_len());

  int ran = 0;
  for (const auto& c : cases) {
    fs::path p = gdir / (c + ".umtf");
    if (!fs::exists(p)) {
      std::printf("SKIP case %s: %s missing (run tools/python/denoiser_golden.py)\n", c.c_str(),
                  p.string().c_str());
      continue;
    }
    try {
      run_case(*be, model, p);
      ++ran;
    } catch (const std::exception& e) {
      std::printf("EXCEPTION in %s: %s\n", c.c_str(), e.what());
      ++g_fail;
    }
  }
  if (!ran && !g_fail) return 77;
  std::printf("%s (%d failing checks)\n", g_fail ? "FAILED" : "PASSED", g_fail);
  return g_fail ? 1 : 0;
}

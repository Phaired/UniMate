// End-to-end parity of the C++ port against the real upstream PyTorch run
// (tests/golden/pipeline/<asset>_{cond,sample,motion}.umtf written by
// tools/python/pipeline_golden.py: caption, exact initial noise, Euler-20 and
// dopri5 samples at cfg 3, NFE, recovered motion).
//
//   um_e2e --asset eagle|shark --solver euler|dopri5 [--backend cpu|vulkan]
//
// Stages, each compared to the golden and reported as max|diff| / RMS / max|ref|:
//   1. T5 pooled caption + joint-name embeddings (C++ T5 on the backend)
//   2. conditioning dict built from them (pipeline::build_condition)
//   3. first CFG velocity call v(noise, t=0) (denoiser)
//   4. the whole Generator::generate() with the golden noise injected:
//      final sample, NFE (+ accepted / rejected steps for dopri5)
//   5. recovered motion (dopri5: vs the upstream recovery of the upstream
//      sample; euler: vs the C++ recovery of the upstream Euler-20 sample,
//      since the golden motion was recovered from the dopri5 sample)
// Exit 0 pass, 1 fail, 77 skipped (missing files / no Vulkan device).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "app/generator.h"
#include "model/denoiser.h"

using namespace um;
namespace fs = std::filesystem;

namespace {

int g_fail = 0;

struct Err {
  double max_abs = 0, rms = 0, ref_max = 0;
  size_t n = 0;
};
Err compare(const std::vector<float>& a, const std::vector<float>& ref) {
  Err e;
  if (a.size() != ref.size()) {
    e.max_abs = e.rms = INFINITY;
    return e;
  }
  double s = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = double(a[i]) - double(ref[i]);
    e.max_abs = std::max(e.max_abs, std::abs(d));
    e.ref_max = std::max(e.ref_max, std::abs(double(ref[i])));
    s += d * d;
  }
  e.n = a.size();
  e.rms = a.empty() ? 0 : std::sqrt(s / double(a.size()));
  return e;
}
template <class T>
std::vector<float> to_f(const std::vector<T>& v) {
  return std::vector<float>(v.begin(), v.end());
}

// Quaternion arrays compared up to sign (q and -q are the same rotation).
std::vector<float> align_quats(std::vector<float> a, const std::vector<float>& ref) {
  for (size_t i = 0; i + 3 < a.size() && i + 3 < ref.size(); i += 4) {
    double dp = 0;
    for (int k = 0; k < 4; ++k) dp += double(a[i + k]) * ref[i + k];
    if (dp < 0)
      for (int k = 0; k < 4; ++k) a[i + k] = -a[i + k];
  }
  return a;
}

// tol is relative to max(1, max|ref|); tol < 0 = report only.
void row(const std::string& stage, const std::string& what, const Err& e, double tol) {
  const double scale = std::max(1.0, e.ref_max);
  const bool ok = tol < 0 || e.max_abs <= tol * scale;
  std::printf("| %-12s | %-34s | %10.3e | %10.3e | %9.3g | %9.3e | %s\n", stage.c_str(), what.c_str(), e.max_abs,
              e.rms, e.ref_max, e.max_abs / std::max(e.ref_max, 1e-30), tol < 0 ? "info" : ok ? "ok" : "FAIL");
  if (!ok) ++g_fail;
}

void check(bool c, const std::string& msg) {
  std::printf("  %s %s\n", c ? "ok  " : "FAIL", msg.c_str());
  if (!c) ++g_fail;
}

std::string golden(const std::string& f) {
  return app::repo_root() + "/tests/golden/pipeline/" + f;
}

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  std::string asset = "eagle", solver = "euler", backend = "cpu", models;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--asset") asset = next();
    else if (a == "--solver") solver = next();
    else if (a == "--backend") backend = next();
    else if (a == "--models") models = next();
    else {
      std::fprintf(stderr, "usage: um_e2e --asset eagle|shark --solver euler|dopri5 [--backend cpu|vulkan]\n");
      return 2;
    }
  }
  if (solver != "euler" && solver != "dopri5") {
    std::fprintf(stderr, "--solver must be euler or dopri5\n");
    return 2;
  }
  const std::string key = solver == "euler" ? "euler20" : "dopri5";
  for (const auto& f : {golden(asset + "_cond.umtf"), golden(asset + "_sample.umtf"), golden(asset + "_motion.umtf")})
    if (!fs::exists(f)) {
      std::printf("SKIP: missing %s (run tools/python/pipeline_golden.py)\n", f.c_str());
      return 77;
    }
  const auto kind = app::parse_backend(backend);
  if (kind == app::BackendKind::Vulkan && !app::vulkan_compute_available()) {
    std::printf("SKIP: --backend vulkan but %s\n",
                app::vulkan_compute_compiled() ? "no usable Vulkan device" : "this build has no um_vk");
    return 77;
  }

  try {
    UmtfFile gc = UmtfFile::load(golden(asset + "_cond.umtf"));
    UmtfFile gs = UmtfFile::load(golden(asset + "_sample.umtf"));
    UmtfFile gm = UmtfFile::load(golden(asset + "_motion.umtf"));
    const std::string prompt = gc.meta()["prompt"].get<std::string>();
    const float cfg = gs.meta()["cfg_scale"].get<float>();
    const auto jn_texts = gc.meta()["joint_name_texts"].get<std::vector<std::string>>();

    GeneratorOptions go;
    go.models_dir = models;
    go.backend = kind;
    Generator gen(go);
    std::printf("e2e %s / %s on %s  (prompt \"%s\", cfg %g)\n", asset.c_str(), key.c_str(),
                gen.backend_description().c_str(), prompt.c_str(), cfg);
    if (!gen.backend_notice().empty()) std::printf("note: %s\n", gen.backend_notice().c_str());
    std::printf("models loaded in %.0f ms\n", gen.load_ms());
    gen.load_asset(app::resolve_asset(asset, gen.models_dir()));
    std::printf("asset loaded, joint-name T5 %.0f ms (%zu strings)\n\n", gen.last_asset_t5_ms(), jn_texts.size());

    std::printf("| %-12s | %-34s | %-10s | %-10s | %-9s | %-9s |\n", "stage", "quantity", "max|diff|", "RMS",
                "max|ref|", "rel");
    std::printf("|--------------|------------------------------------|------------|------------|-----------|-----------|\n");

    // ---- 1. T5 ----
    double t0 = now_ms();
    HostArray cap = gen.embed({prompt});
    const double t5_caption_ms = now_ms() - t0;
    row("T5", "caption_emb [1,768]", compare(cap.to_f32(), gc.get("caption_emb").to_f32()), 1e-4);
    HostArray jn = gen.embed(jn_texts);
    row("T5", "joint names [" + std::to_string(jn_texts.size()) + ",768]",
        compare(jn.to_f32(), gc.get("joint_name_text_emb").to_f32()), 1e-4);

    // ---- 2. conditioning ----
    const auto& dims = gen.dims();
    ArrayDict cond = pipeline::build_condition(gen.asset(), gen.stats(), dims, prompt, dims.max_frames,
                                               [&](const std::vector<std::string>& t) { return gen.embed(t); });
    int cond_keys = 0, cond_int_mismatch = 0;
    Err cond_f;
    for (const auto& name : gc.names()) {
      if (name.rfind("cond/", 0) != 0) continue;
      const std::string k = name.substr(5);
      auto it = cond.find(k);
      const HostArray& ref = gc.get(name);
      if (it == cond.end()) {
        if (k != "caption_tokens" && k != "caption_mask") std::printf("  (cond key %s not produced)\n", k.c_str());
        continue;
      }
      ++cond_keys;
      if (it->second.shape != ref.shape) {
        std::printf("  FAIL cond %s shape %s vs %s\n", k.c_str(), shape_str(it->second.shape).c_str(),
                    shape_str(ref.shape).c_str());
        ++g_fail;
        continue;
      }
      if (ref.dtype == DType::F32) {
        Err e = compare(it->second.to_f32(), ref.to_f32());
        if (e.max_abs > cond_f.max_abs) cond_f.max_abs = e.max_abs;
        cond_f.ref_max = std::max(cond_f.ref_max, e.ref_max);
        cond_f.rms = std::max(cond_f.rms, e.rms);
        if (e.max_abs > 1e-4 * std::max(1.0, e.ref_max))
          std::printf("  cond %-26s max|diff| %.3e (max|ref| %.3g)\n", k.c_str(), e.max_abs, e.ref_max);
      } else if (it->second.to_i64() != ref.to_i64()) {
        ++cond_int_mismatch;
        std::printf("  FAIL cond %s integer values differ\n", k.c_str());
      }
    }
    row("condition", "float keys (worst of " + std::to_string(cond_keys) + ")", cond_f, 1e-4);
    check(cond_int_mismatch == 0, "integer cond keys identical");

    // ---- 3. first velocity call ----
    const HostArray& noise = gs.get("noise");
    auto call_t = gs.get(key + "_call_t").to_f32();
    const HostArray& call_v = gs.get(key + "_call_v");
    Denoiser& den = gen.denoiser();
    den.set_condition(cond);
    t0 = now_ms();
    Tensor v0 = den.velocity(noise.to_tensor(gen.backend()), call_t[0], cfg);
    std::vector<float> v0h = v0.to_vector();
    const double v0_ms = now_ms() - t0;
    const int64_t per = noise.numel();
    auto cv = call_v.to_f32();
    row("denoiser", "v(noise, t=" + std::to_string(call_t[0]).substr(0, 4) + ") CFG",
        compare(v0h, std::vector<float>(cv.begin(), cv.begin() + per)), 1e-3);

    // ---- 4. full generation with the golden noise ----
    gen.set_initial_noise(noise);
    GenerateRequest req;
    req.prompt = prompt;
    req.cfg_scale = cfg;
    req.num_frames = dims.max_frames;
    req.solver = solver;
    req.num_steps = solver == "euler" ? 20 : 50;
    int last_pct = -1;
    MotionClip clip = gen.generate(req, [&](float p, const std::string&) {
      int pct = int(p * 100.f);
      if (pct / 10 != last_pct / 10) {
        last_pct = pct;
        std::fprintf(stderr, "  [%3d%%]\r", pct);
      }
    });
    const auto& st = gen.last_stats();
    const auto& ss = gen.last_solver_stats();
    const auto sample = gen.last_sample().to_f32();
    const auto ref_sample = gs.get(key).to_f32();
    // Only the valid joints carry meaning; padded joints are reported separately.
    const int J = gen.asset().num_joints, F = dims.max_frames, D = dims.feature_len;
    std::vector<float> a_valid, r_valid, a_pad, r_pad;
    for (int j = 0; j < dims.max_joints; ++j)
      for (int k = 0; k < D * F; ++k) {
        const size_t idx = size_t(j) * D * F + k;
        (j < J ? a_valid : a_pad).push_back(sample[idx]);
        (j < J ? r_valid : r_pad).push_back(ref_sample[idx]);
      }
    const double sample_tol = solver == "euler" ? 2e-4 : 5e-3;
    row("sample", key + " valid joints [" + std::to_string(J) + ",12,60]", compare(a_valid, r_valid), sample_tol);
    row("sample", key + " padded joints", compare(a_pad, r_pad), -1);
    const int nfe_ref = gs.meta()[key + "_nfe"].get<int>();
    std::printf("\n  NFE c++ %d / torch %d", ss.nfe, nfe_ref);
    if (solver == "dopri5")
      std::printf("   accepted %d / %d   rejected %d / %d   first step %.9g / %.9g", ss.accepted,
                  gs.meta()["dopri5_accepted"].get<int>(), ss.rejected, gs.meta()["dopri5_rejected"].get<int>(),
                  ss.first_step, gs.meta()["dopri5_first_step"].get<double>());
    std::printf("\n");
    if (solver == "euler") {
      check(ss.nfe == nfe_ref, "Euler NFE identical");
    } else {
      // The adaptive controller sees float32 noise in its error estimate: allow
      // one or two extra / fewer attempted steps (6 NFE each).
      check(std::abs(ss.nfe - nfe_ref) <= 12, "dopri5 NFE within 2 steps of torchdiffeq");
      // step-by-step comparison
      const HostArray& raw = gs.get("dopri5_steps@f64");
      const double* ref_steps = reinterpret_cast<const double*>(raw.bytes.data());
      const size_t nref = raw.bytes.size() / (3 * sizeof(double));
      std::printf("  step  t0(c++)      dt(c++)      acc | t0(torch)    dt(torch)    acc  rel dt diff\n");
      for (size_t i = 0; i < std::max(nref, ss.steps.size()); ++i) {
        std::printf("  %3zu ", i);
        if (i < ss.steps.size())
          std::printf(" %.9f  %.9f  %d  |", ss.steps[i].t0, ss.steps[i].dt, int(ss.steps[i].accepted));
        else
          std::printf(" %-29s|", "");
        if (i < nref) {
          const double* r = ref_steps + 3 * i;
          std::printf(" %.9f  %.9f  %d", r[0], r[1], int(r[2]));
          if (i < ss.steps.size()) std::printf("  %.2e", std::abs(ss.steps[i].dt - r[1]) / std::abs(r[1]));
        }
        std::printf("\n");
      }
    }

    // ---- 5. motion ----
    const int T = clip.num_frames;
    // Golden motion = upstream recovery of the golden sample it was made from.
    if (gm.meta().value("sample_key", std::string()) == key) {
      pipeline::RecoveredMotion r = pipeline::recover_features(gen.asset(), gen.stats(), gen.last_sample(), T);
      row("motion", "de-normalised features", compare(r.features, gm.get("motion").to_f32()), 5e-3);
      row("motion", "root trajectory", compare(to_f(r.root_pos), gm.get("root_pos").to_f32()), 5e-3);
      row("motion", "FK joint positions", compare(to_f(r.positions), gm.get("anim_positions").to_f32()), 5e-3);
      row("motion", "BVH local rotations (quat)",
          compare(align_quats(to_f(r.bvh_rot), gm.get("bvh_rotations").to_f32()), gm.get("bvh_rotations").to_f32()),
          5e-3);
      // glTF node-local rotations of the exported tracks
      auto lr = gm.get("gltf_local_r").to_f32();
      std::vector<float> mine, ref;
      for (const auto& tr : clip.tracks) {
        const auto& names = gen.asset().joint_names;
        const int j = int(std::find(names.begin(), names.end(), tr.node_name) - names.begin());
        if (j >= J || tr.rotation.size() != size_t(T) * 4) continue;
        for (int t = 0; t < T; ++t)
          for (int k = 0; k < 4; ++k) {
            mine.push_back(tr.rotation[size_t(t) * 4 + k]);
            ref.push_back(lr[(size_t(t) * J + j) * 4 + k]);
          }
      }
      row("motion", "glTF local rotations (exported)", compare(align_quats(mine, ref), ref), 5e-3);
    } else {
      pipeline::RecoveredMotion r = pipeline::recover_features(gen.asset(), gen.stats(), gen.last_sample(), T);
      pipeline::RecoveredMotion g = pipeline::recover_features(gen.asset(), gen.stats(), gs.get(key), T);
      row("motion", "features vs recover(torch euler20)", compare(r.features, g.features), 1e-3);
      row("motion", "FK positions vs recover(torch)", compare(to_f(r.positions), to_f(g.positions)), 1e-3);
      row("motion", "BVH rotations vs recover(torch)",
          compare(align_quats(to_f(r.bvh_rot), to_f(g.bvh_rot)), to_f(g.bvh_rot)), 1e-3);
    }
    check(clip.num_frames == F && int(clip.tracks.size()) >= J, "clip has all frames and joint tracks");

    std::printf("\ntimings (%s): models %.0f ms | caption T5 %.0f ms | 1 CFG velocity %.0f ms | generate: "
                "sampling %.0f ms for %d NFE = %.1f ms/velocity, total %.0f ms (torch CPU golden run: %.1f s)\n",
                gen.backend_description().c_str(), gen.load_ms(), t5_caption_ms, v0_ms, st.sample_ms, st.nfe,
                st.ms_per_velocity, st.total_ms, gs.meta()[key + "_seconds"].get<double>());
  } catch (const std::exception& e) {
    std::printf("ERROR: %s\n", e.what());
    return 1;
  }
  std::printf("%s\n", g_fail ? "E2E FAILED" : "E2E OK");
  return g_fail ? 1 : 0;
}

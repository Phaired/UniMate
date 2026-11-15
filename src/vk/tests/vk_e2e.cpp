// End-to-end parity + timing of real models on the Vulkan backend vs the CPU
// backend (and the PyTorch goldens for the denoiser). Built only when the
// T5 / model modules are part of the build (see src/vk/CMakeLists.txt).
//   um_vk_e2e            exit 77 when weights / goldens are missing
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "core/umtf.h"
#include "vk/vk_backend.h"
#ifdef UM_VK_HAVE_T5
#include "t5/t5_encoder.h"
#endif
#ifdef UM_VK_HAVE_MODEL
#include "model/denoiser.h"
#endif

namespace fs = std::filesystem;
using namespace um;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

struct Diff {
  double max_abs = 0, ref_max = 0;
  bool nonfinite = false;
};
static Diff diff(const std::vector<float>& a, const std::vector<float>& b) {
  Diff d;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::isinf(a[i]) && a[i] == b[i]) continue;
    if (!std::isfinite(b[i]) || !std::isfinite(a[i])) {
      d.nonfinite = true;
      continue;
    }
    d.max_abs = std::max(d.max_abs, double(std::fabs(a[i] - b[i])));
    d.ref_max = std::max(d.ref_max, double(std::fabs(a[i])));
  }
  return d;
}

static int g_fail = 0, g_ran = 0;

static void check(const char* what, const std::vector<float>& ref, const std::vector<float>& got, double tol) {
  Diff d = diff(ref, got);
  bool ok = !d.nonfinite && d.max_abs <= tol;
  std::printf("  %-34s max_abs %.3e (ref max %.3f)%s\n", what, d.max_abs, d.ref_max,
              ok ? "" : (d.nonfinite ? "  NON-FINITE <-- FAIL" : "  <-- FAIL"));
  if (!ok) ++g_fail;
}

int main() {
  const fs::path root = UM_ROOT_DIR;
  if (!vulkan_available()) {
    std::printf("SKIP: no Vulkan device\n");
    return 77;
  }
  auto cpu = make_cpu_backend();
  auto vk = make_vulkan_backend();
  std::printf("Vulkan: %s\n", vulkan_device_name(*vk).c_str());

#ifdef UM_VK_HAVE_T5
  {
    const fs::path w = root / "models" / "converted" / "flan-t5-base.umtf";
    if (fs::exists(w)) {
      ++g_ran;
      std::printf("== T5 encoder (flan-t5-base)\n");
      T5TextEncoder ec(*cpu, w.string()), eg(*vk, w.string());
      std::vector<std::string> texts = {"a person walks forward and then jumps over an obstacle", ""};
      const char* joints[] = {"Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "LeftShoulder", "LeftArm",
                              "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm",
                              "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg",
                              "RightLeg", "RightFoot", "RightToeBase", "tail_01", "tail_02", "wing_left_01"};
      for (int i = 0; texts.size() < 72; ++i) texts.push_back(std::string(joints[i % 25]) + " " + std::to_string(i));
      auto t0 = Clock::now();
      T5Encoding rc = ec.encode(texts);
      std::vector<float> hc = rc.hidden.to_vector();
      const double tc = ms_since(t0);
      eg.encode(texts).hidden.to_vector();  // warm-up (pipelines, pool)
      t0 = Clock::now();
      T5Encoding rg = eg.encode(texts);
      std::vector<float> hg = rg.hidden.to_vector();
      const double tg = ms_since(t0);
      std::printf("  B=%zu T=%lld: cpu %.1f ms, vulkan %.1f ms\n", texts.size(),
                  (long long)rc.hidden.dim(1), tc, tg);
      check("hidden (incl. empty-string row)", hc, hg, 1e-3);
      std::vector<std::string> one = {texts[0]};
      eg.encode(one).hidden.to_vector();
      t0 = Clock::now();
      for (int i = 0; i < 10; ++i) eg.encode(one).hidden.to_vector();
      std::printf("  single prompt: vulkan %.2f ms\n", ms_since(t0) / 10);
    } else {
      std::printf("SKIP T5: %s missing\n", w.string().c_str());
    }
  }
#endif

#ifdef UM_VK_HAVE_MODEL
  {
    const fs::path w = root / "models" / "converted" / "unimate_uniml3d_f60_v3.umtf";
    const fs::path gdir = root / "tests" / "golden" / "denoiser";
    if (fs::exists(w)) {
      std::printf("== Denoiser\n");
      auto t0 = Clock::now();
      Denoiser mc(*cpu, w.string());
      Denoiser mg(*vk, w.string());
      std::printf("  loaded both in %.0f ms; device memory after load %.0f MB\n", ms_since(t0),
                  vulkan_memory_stats(*vk).device_bytes / 1048576.0);
      for (const char* name : {"eagle", "shark", "batch2"}) {
        fs::path p = gdir / (std::string(name) + ".umtf");
        if (!fs::exists(p)) continue;
        ++g_ran;
        std::printf(" -- %s\n", name);
        UmtfFile g = UmtfFile::load(p.string());
        ArrayDict cond;
        for (const auto& n : g.names())
          if (n.rfind("cond.", 0) == 0) cond[n.substr(5)] = g.get(n);
        mc.set_condition(cond);
        mg.set_condition(cond);
        const std::vector<float> ts = g.get("t").to_f32();
        const float cfg = g.meta()["cfg_scale"].get<float>();
        Tensor xc = g.get("x").to_tensor(*cpu), xg = g.get("x").to_tensor(*vk);
        double tv = 0, tcpu = 0;
        int nv = 0;
        for (size_t i = 0; i < ts.size(); ++i) {
          const std::string si = std::to_string(i);
          auto a = Clock::now();
          std::vector<float> vc = mc.velocity(xc, ts[i], cfg).to_vector();
          tcpu += ms_since(a);
          mg.velocity(xg, ts[i], cfg).to_vector();  // warm-up
          a = Clock::now();
          const int reps = 5;
          std::vector<float> vg;
          for (int r = 0; r < reps; ++r) vg = mg.velocity(xg, ts[i], cfg).to_vector();
          tv += ms_since(a) / reps;
          ++nv;
          std::vector<float> gold = g.get("velocity." + si).to_f32();
          check(("velocity." + si + " vk vs cpu").c_str(), vc, vg, 1e-4);
          check(("velocity." + si + " vk vs golden").c_str(), gold, vg, 1e-4);
          std::vector<float> oc = g.get("out_cond." + si).to_f32();
          check(("out_cond." + si + " vk vs golden").c_str(), oc, mg.forward(xg, ts[i], false).to_vector(), 1e-4);
        }
        auto cn0 = vulkan_counters(*vk);
        mg.velocity(xg, ts[0], cfg).to_vector();
        auto cn1 = vulkan_counters(*vk);
        std::printf("  velocity: cpu %.1f ms, vulkan %.2f ms per call (B=%d -> %d, %d ops, %llu dispatches, "
                    "%llu barriers, %llu submits)\n",
                    tcpu / nv, tv / nv, mg.batch(), 2 * mg.batch(), mg.last_op_count(),
                    (unsigned long long)(cn1.dispatches - cn0.dispatches),
                    (unsigned long long)(cn1.barriers - cn0.barriers), (unsigned long long)(cn1.submits - cn0.submits));
      }
      auto ms = vulkan_memory_stats(*vk);
      std::printf("  vulkan memory: device %.0f MB (peak %.0f MB), live %.0f MB\n", ms.device_bytes / 1048576.0,
                  ms.peak_device_bytes / 1048576.0, ms.live_bytes / 1048576.0);
    } else {
      std::printf("SKIP denoiser: %s missing\n", w.string().c_str());
    }
  }
#endif
  if (!g_ran) return 77;
  std::printf("%s (%d failing checks)\n", g_fail ? "FAILED" : "PASSED", g_fail);
  return g_fail ? 1 : 0;
}

// Generator cancellation: a ProgressFn that throws mid-sampling must abort
// generate(), propagate the exception unchanged, and leave the Generator /
// backend usable: the next generation must equal a clean reference run.
//   um_app_cancel [--backend cpu|vulkan]
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "app/generator.h"

namespace {
struct Cancelled : std::runtime_error {
  Cancelled() : std::runtime_error("cancelled") {}
};
}  // namespace

int main(int argc, char** argv) {
  std::string backend = "cpu";
  for (int i = 1; i + 1 < argc; ++i)
    if (std::string(argv[i]) == "--backend") backend = argv[++i];
  const auto kind = um::app::parse_backend(backend);
  if (kind == um::app::BackendKind::Vulkan && !um::app::vulkan_compute_available()) {
    std::printf("SKIP: no Vulkan compute backend\n");
    return 77;
  }
  try {
    um::GeneratorOptions go;
    go.backend = kind;
    um::Generator gen(go);
    gen.load_asset(um::app::resolve_asset("eagle", gen.models_dir()));
    um::GenerateRequest req;
    req.prompt = "An object flaps its wings.";
    req.solver = "euler";
    req.num_steps = 4;  // 3 velocity calls
    req.seed = 7;

    um::MotionClip ref = gen.generate(req);
    const auto ref_sample = gen.last_sample().to_f32();
    int fails = 0;
    if (ref.joint_positions.size() != size_t(ref.num_frames) * ref.num_joints * 3 ||
        ref.parents.size() != size_t(ref.num_joints) || ref.num_joints <= 0) {
      std::printf("FAIL: clip has no debug skeleton\n");
      ++fails;
    }

    int calls = 0, seen_nfe = 0;
    bool caught = false;
    try {
      gen.generate(req, [&](float, const std::string& s) {
        if (s.rfind("sampling (NFE", 0) == 0) ++seen_nfe;
        if (++calls == 6) throw Cancelled();
      });
    } catch (const Cancelled&) {
      caught = true;
    }
    std::printf("cancel: exception %s after %d progress calls (%d per-NFE)\n", caught ? "propagated" : "LOST",
                calls, seen_nfe);
    if (!caught) ++fails;

    um::MotionClip again = gen.generate(req);
    const auto s2 = gen.last_sample().to_f32();
    double d = s2.size() == ref_sample.size() ? 0 : INFINITY;
    for (size_t i = 0; i < s2.size() && i < ref_sample.size(); ++i)
      d = std::fmax(d, std::fabs(double(s2[i]) - ref_sample[i]));
    std::printf("after cancel: max|sample - reference| = %.3g, NFE %d\n", d, gen.last_stats().nfe);
    if (!(d == 0.0) || again.num_frames != ref.num_frames) ++fails;
    std::printf("%s\n", fails ? "CANCEL TEST FAILED" : "CANCEL TEST OK");
    return fails ? 1 : 0;
  } catch (const std::exception& e) {
    std::printf("ERROR: %s\n", e.what());
    return 1;
  }
}

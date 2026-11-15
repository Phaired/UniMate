#include "pipeline/pipeline.h"

#include <algorithm>
#include <stdexcept>

namespace um::pipeline {

um::MotionClip generate(um::Backend& be, const AssetBundle& bundle, const Stats& stats,
                        const ModelDims& dims, const um::GenerateRequest& req,
                        const GenerationCallbacks& cb, const um::ProgressFn& progress,
                        SolverStats* solver_stats) {
  if (!cb.set_condition || !cb.velocity) throw std::runtime_error("generate: missing callbacks");
  auto report = [&](float p, const std::string& s) {
    if (progress) progress(p, s);
  };
  const int frames = (req.num_frames > 0 && req.num_frames <= dims.max_frames) ? req.num_frames
                                                                               : dims.max_frames;
  report(0.f, "conditioning");
  um::ArrayDict cond = build_condition(bundle, stats, dims, req.prompt, frames, cb.embed);
  cb.set_condition(cond);
  report(0.05f, "sampling");

  um::HostArray noise =
      make_noise({1, dims.max_joints, dims.feature_len, dims.max_frames}, req.seed);
  SamplerOptions opt;
  opt.solver = req.solver.empty() ? "dopri5" : req.solver;
  if (opt.solver != "dopri5") opt.num_steps = std::max(2, req.num_steps);
  const float cfg = req.cfg_scale;
  VelocityFn v = [&](const um::Tensor& x, float t) { return cb.velocity(x, t, cfg); };
  um::Tensor x = sample_flow(
      be, v, noise.to_tensor(be), opt,
      [&](float p, const std::string& s) { report(0.05f + 0.9f * p, s); }, solver_stats);
  report(0.95f, "recovering motion");
  um::MotionClip clip = recover_motion(bundle, stats, um::HostArray::from_tensor(x), frames, 30.f);
  report(1.f, "done");
  return clip;
}

}  // namespace um::pipeline

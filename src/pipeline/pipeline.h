// um_pipeline: the non-neural part of UniMate text-to-motion inference.
//
//   AssetBundle b = load_bundle("models/converted/bundles/eagle.umtf", ".../Eagle.glb");
//   Stats s       = load_stats("models/converted/unimate_uniml3d_f60_v3_stats.umtf", b.stats_dataset);
//   ModelDims d   = load_model_dims(".../unimate_uniml3d_f60_v3/config.json");
//   ArrayDict c   = build_condition(b, s, d, prompt, d.max_frames, t5_pooled);
//   denoiser.set_condition(c);
//   Tensor x      = sample_flow(be, [&](auto& x, float t) { return denoiser.velocity(x, t, cfg); },
//                               make_noise({1, d.max_joints, 12, d.max_frames}, seed).to_tensor(be), {});
//   MotionClip m  = recover_motion(b, s, HostArray::from_tensor(x), d.max_frames, 30.f);
//   export_glb(b.glb_path, m, "out.glb");
//
// generate() below wires exactly that through callbacks, so the pipeline does
// not depend on the model / T5 libraries.
#pragma once

#include "pipeline/bundle.h"
#include "pipeline/condition.h"
#include "pipeline/glb.h"
#include "pipeline/motion.h"
#include "pipeline/sampler.h"

namespace um::pipeline {

struct GenerationCallbacks {
  TextEmbedFn embed;                                        // pooled T5, [N, text_dim]
  std::function<void(const um::ArrayDict&)> set_condition;  // e.g. Denoiser::set_condition
  // CFG velocity, e.g. Denoiser::velocity(x, t, cfg_scale)
  std::function<um::Tensor(const um::Tensor& x, float t, float cfg_scale)> velocity;
};

// build_condition -> set_condition -> make_noise(seed) -> sample_flow ->
// recover_motion. The request's solver / num_steps / cfg_scale / num_frames
// are honoured (num_frames is the motion_length given to the model and the
// clip length; the model always generates max_frames). Progress: 0..0.05
// conditioning, 0.05..0.95 sampling, 1 done.
um::MotionClip generate(um::Backend& be, const AssetBundle& bundle, const Stats& stats,
                        const ModelDims& dims, const um::GenerateRequest& req,
                        const GenerationCallbacks& cb, const um::ProgressFn& progress = {},
                        SolverStats* solver_stats = nullptr);

}  // namespace um::pipeline

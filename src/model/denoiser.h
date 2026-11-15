// um::Denoiser -- the UniMate flow-matching denoiser (inference only),
// a port of reference/UniMate/unimate/models/denoiser/graph_adaln.py
// (UniMateGraphAdaLN: attention='graph' x text_cond='adaln') on the abstract
// um::Backend. Weights + config come from a UMTF written by
// tools/python/denoiser_export.py (EMA weights, state-dict names).
//
// ---------------------------------------------------------------------------
// Conditioning interface (set_condition): an um::ArrayDict with the SAME key
// names and shapes as the upstream `cond` dict that
// unimate/dataset/conditioning.py::create_sample_condition +
// unimate/dataset/mixture/collate.py::mixture_batch_collate produce.
// Batch dim B first; J = max_joints (71) -- every joint-axis field padded with
// zeros to max_joints, exactly as the collate does. Integer keys may be i32 or
// i64 (upstream: int64). Keys not listed are ignored.
//
//   key                       dtype     shape            notes
//   caption_emb               f32       [B, 768]         pooled T5 caption; optional
//                                                         (missing -> zeros, as upstream)
//   n_joints                  int       [B]              valid joints per sample (>= 2)
//   motion_length             int       [B]              frames to attend (sampling: = F = 60)
//   tpos_first_frame          f32       [B, J, 12]       normalised T-pose features
//   tpos_first_frame_parents  f32       [B, J, 12]       parent-copied tpos (concat_parent_features)
//   joint_depths              int       [B, J]           kinematic depth (clamped to max_depth)
//   joint_names_emb           f32       [B, J, 768]      pooled T5 of clean joint names
//   spectral_feats            f32       [B, J, 8]        Laplacian eigvecs (K must equal max_freqs)
//   graph_dist                int       [B, J, J]        hop distance, values in [0, 5]
//   joint_relations           int       [B, J, J]        edge type, values in [0, 5]
//
// Masks are derived on the host from n_joints / motion_length (joint-validity
// and temporal key masks become additive -1e30 biases); padded joints still
// produce (meaningless) outputs, like upstream.
//
// Model input / output: x and the returned velocity are [B, J, 12, F]
// (upstream [batch, max_joints, nfeats, max_frames]); F <= max_frames().
// t is the flow time in [0, 1], passed to the timestep embedder UNSCALED
// (transport.get_drift calls model(x, t) with the ODE time itself).
//
// CFG: velocity() = out_uncond + cfg * (out_cond - out_uncond)
// (inference/generate.py::ClassifierFreeSampleModel), computed as one batched
// 2B forward. cfg <= 1 returns the plain conditional pass, like
// generate_samples(). NOTE sample.py at cfg_scale == 1.0 instead wraps the
// model in _UnconditionalWrapper (force_mask=True): to reproduce that mode call
// forward(x, t, /*force_mask=*/true).
// ---------------------------------------------------------------------------
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/tensor.h"
#include "core/umtf.h"

namespace um {

// Optional per-stage activations for parity tests (names match the goldens
// written by tools/python/denoiser_golden.py, without the "inter." prefix).
using DenoiserTrace = std::map<std::string, Tensor>;

class Denoiser {
 public:
  Denoiser(Backend& be, const std::string& umtf_path);
  ~Denoiser();
  Denoiser(const Denoiser&) = delete;
  Denoiser& operator=(const Denoiser&) = delete;

  const nlohmann::json& config() const;  // config.json as exported (meta.config)
  int max_joints() const;                // 71: x / cond joint axis size
  int max_frames() const;                // 60: max_motion_length (generated frames)
  int feature_len() const;               // 12
  int batch() const;                     // B of the current condition (0 if none)

  // Upload the conditioning and precompute everything that does not depend on
  // x / t (masks, per-block graph bias, spectral + temporal RoPE cos/sin,
  // joint-name / depth embeddings, tpos encoding + tpos pool, caption
  // projection, CFG-doubled copies). Throws on missing / malformed keys.
  void set_condition(const ArrayDict& cond);

  // One denoiser pass. x [B, J, 12, F]; returns [B, J, 12, F].
  Tensor forward(const Tensor& x, float t, bool force_mask);
  // out_uncond + cfg * (out_cond - out_uncond); cfg <= 1 -> conditional pass.
  Tensor velocity(const Tensor& x, float t, float cfg_scale);

  // Same as forward(), also recording intermediates (and the static
  // conditioning tensors) into *trace.
  Tensor forward_traced(const Tensor& x, float t, bool force_mask, DenoiserTrace* trace);

  // Number of backend ops issued by the last forward / velocity call
  // (reshape views excluded; counted by the model, not the backend).
  int last_op_count() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace um

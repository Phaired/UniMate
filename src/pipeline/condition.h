// The model conditioning dict (upstream unimate/dataset/conditioning.py
// create_sample_condition + mixture/collate.py mixture_batch_collate, for one
// rig_preprocess asset registered by MotionDataset.add_cond_object).
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "core/umtf.h"
#include "pipeline/bundle.h"

namespace um::pipeline {

// Pooled text embedding: one row per input string, [N, text_dim] f32.
// Upstream pools T5 token features with a plain mean over the non-padding
// tokens (unimate/utils/text_emb_cache.pool).
using TextEmbedFn = std::function<um::HostArray(const std::vector<std::string>&)>;

// Builds the upstream `cond` dict with batch size 1. Keys / shapes / dtypes
// (J = max_joints, F = max_frames, D = feature_len, n = bundle joints):
//   crop_start_ind            i64 [1]          0 (the rest-pose reference clip)
//   motion_length             i64 [1]          num_frames (upstream: always F)
//   lengths_mask              u8  [1,1,1,F]    f < motion_length
//   n_joints                  i64 [1]          n
//   joint_mask                u8  [1,1,1,J]    j < n
//   tpos_first_frame          f32 [1,J,D]      normalised grounded T-pose (pos | identity 6D | 0 vel)
//   tpos_first_frame_parents  f32 [1,J,D]      row of the parent (root: itself)
//   parents                   i64 [1,n]        (upstream: python list of arrays)
//   edge_indexs               i64 [1,2,2(n-1)] compute_edge_indexs(parents) (upstream: list)
//   joint_relations           i64 [1,J,J]
//   graph_dist                i64 [1,J,J]
//   joint_depths              i64 [1,J]
//   mean, std                 f32 [1,J,D]      std = 1 on padding joints
//   spectral_feats            f32 [1,J,K]
//   joint_names_emb           f32 [1,J,text_dim]
//   offsets                   f32 [1,J,3]      offsets_from_positions(grounded T-pose)
//   caption_emb               f32 [1,text_dim]
// Not produced: caption_tokens / caption_mask (only text_cond='cross_attn'
// reads them; the released model is 'adaln'), and the string entries
// object_type / caption.
// `embed` is called twice: once with the unique model joint names (first
// occurrence order, as upstream batches them) and once with {prompt}. An
// empty prompt yields a zero caption_emb without calling `embed`.
um::ArrayDict build_condition(const AssetBundle& bundle, const Stats& stats,
                              const ModelDims& dims, const std::string& prompt,
                              int num_frames, const TextEmbedFn& embed);

// ---- topology helpers (unimate/utils/topology_utils.py) --------------------
std::vector<int64_t> compute_edge_indexs(const std::vector<int>& parents);  // [2*E] row-major [2,E]
std::vector<int> compute_joint_depths(const std::vector<int>& parents);
// edge relation types (0 self,1 parent,2 child,3 sibling,4 none,5 end effector)
// and BFS hop distances clamped to max_path_len; both [J*J].
void compute_edge_relations_and_distances(const std::vector<int>& parents,
                                          std::vector<int>& edge_rel,
                                          std::vector<int>& topo_dist, int max_path_len = 5);
// Smallest non-trivial eigenvectors of the symmetric normalised Laplacian,
// L2-normalised, zero padded to max_freqs ([J*max_freqs]). Upstream only
// recomputes these when the stored width != max_freqs; the eigenvector SIGNS
// of numpy.linalg.eigh are LAPACK-dependent, so this fallback pins each
// column's largest-|.| entry positive and does not match upstream bit for bit.
std::vector<float> compute_laplacian_eigenvectors(const std::vector<int>& parents, int max_freqs);

// The 12-D normalisation mean/std of joint j (root uses *_root).
void joint_mean_std(const Stats& s, int joint, const double*& mean, const double*& std);

}  // namespace um::pipeline

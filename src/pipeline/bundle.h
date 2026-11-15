// Asset bundles, normalisation stats and model dimensions.
//
// A bundle is a `data_process.rig_preprocess` output converted by
// tools/python/pipeline_convert_bundle.py (cond.npy + summary.json +
// annotation.json -> UMTF; see that script for the tensor list) plus the
// asset's canonical GLB, whose node hierarchy is read here.
#pragma once

#include <string>
#include <vector>

#include "core/umtf.h"
#include "pipeline/glb.h"

namespace um::pipeline {

// Model / dataset dimensions (config.json of the released model).
struct ModelDims {
  int max_joints = 71;   // dataset.max_joints (padding of the joint axis)
  int max_frames = 60;   // dataset.max_motion_length
  int feature_len = 12;  // dataset.feature_len
  int max_depth = 20;    // dataset.max_depth
  int max_freqs = 8;     // model.max_freqs (spectral width)
  int text_dim = 768;    // T5 hidden size (flan-t5-base)
  float cfg_scale = 3.0f;  // sampling.cfg_scale
};
// Reads config.json (dataset.*, model.max_freqs, model.text_encoder_version,
// sampling.cfg_scale). Missing fields keep the defaults above.
ModelDims load_model_dims(const std::string& config_json_path);

// One normalisation pool (feature_len values each), as float64 like numpy.
struct Stats {
  std::string key;  // dataset_stats key actually used
  std::vector<double> mean_root, std_root, mean_local, std_local;
};
// Loads models/converted/<run>_stats.umtf and resolves the pool for an asset
// whose stats dataset is `dataset_type` exactly like upstream
// MotionDataset.resolve_stats_key: the key itself, else STATS_FALLBACK
// (general -> objaverse), else any key when every pool is identical.
Stats load_stats(const std::string& umtf_path, const std::string& dataset_type = "general");

struct AssetBundle {
  std::string name;           // cond key (object type)
  std::string umtf_path, glb_path;
  std::string stats_dataset;  // e.g. "general" (summary.json)
  int num_joints = 0;

  // ---- cond entry (joint order of the model) ----
  std::vector<int> parents;                 // [J], -1 = root
  std::vector<std::string> joint_names;     // raw rig names == GLB node names
  std::vector<std::string> model_joint_names;  // strings the text encoder sees
  std::vector<double> tpos_first_frame;     // [J*3] canonical rest positions (Y up)
  std::vector<double> tpos_offsets;         // [J*3] [0] = root position
  std::vector<double> tpos_local_rotations; // [J*4] w x y z
  std::vector<double> tpos_global_rotations;// [J*4] w x y z
  std::vector<int> joint_relations;         // [J*J]
  std::vector<int> joint_graph_dists;       // [J*J]
  std::vector<int> joint_depths;            // [J]
  int spectral_k = 0;
  std::vector<float> spectral_feats;        // [J*K]
  double scale_factor = 1.0;
  nlohmann::json meta;                      // the bundle's UMTF meta

  // ---- canonical GLB ----
  std::vector<GlbNode> glb_nodes;           // empty when no GLB was given
  std::vector<int> joint_node;              // [J] node index of joint j or -1
  std::vector<int> skin_joints;             // node indices of skin 0 (if any)
};

// `glb_path` may be empty: the bundle then has no mesh hierarchy, and
// recover_motion() emits tracks in the canonical frame (identity rest
// rotations, rest = cond T-pose), named by joint_names.
AssetBundle load_bundle(const std::string& umtf_path, const std::string& glb_path);

}  // namespace um::pipeline

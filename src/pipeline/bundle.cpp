#include "pipeline/bundle.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace um::pipeline {

namespace {
[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("pipeline: " + m); }

// Exact float64 copy (`<name>@f64`, raw bytes) when present, else the f32 tensor.
std::vector<double> get_f64(const UmtfFile& f, const std::string& name) {
  if (f.has(name + "@f64")) {
    const HostArray& a = f.get(name + "@f64");
    std::vector<double> v(a.bytes.size() / 8);
    std::memcpy(v.data(), a.bytes.data(), v.size() * 8);
    return v;
  }
  auto f32 = f.get(name).to_f32();
  return std::vector<double>(f32.begin(), f32.end());
}

std::vector<int> get_i32(const UmtfFile& f, const std::string& name) {
  return f.get(name).to_i32();
}
}  // namespace

ModelDims load_model_dims(const std::string& path) {
  std::ifstream in(path);
  if (!in) fail("cannot open " + path);
  nlohmann::json j = nlohmann::json::parse(in);
  ModelDims d;
  if (j.contains("dataset")) {
    const auto& ds = j["dataset"];
    d.max_joints = ds.value("max_joints", d.max_joints);
    d.max_frames = ds.value("max_motion_length", d.max_frames);
    d.feature_len = ds.value("feature_len", d.feature_len);
    d.max_depth = ds.value("max_depth", d.max_depth);
  }
  if (j.contains("model")) {
    const auto& m = j["model"];
    d.max_freqs = m.value("max_freqs", d.max_freqs);
    std::string v = m.value("text_encoder_version", std::string());
    if (v.find("small") != std::string::npos) d.text_dim = 512;
    else if (v.find("large") != std::string::npos || v.find("3b") != std::string::npos ||
             v.find("11b") != std::string::npos) d.text_dim = 1024;
    else if (v.find("-xxl") != std::string::npos) d.text_dim = 4096;
    else if (v.find("-xl") != std::string::npos) d.text_dim = 2048;
  }
  if (j.contains("sampling")) d.cfg_scale = j["sampling"].value("cfg_scale", d.cfg_scale);
  return d;
}

Stats load_stats(const std::string& path, const std::string& dataset_type) {
  UmtfFile f = UmtfFile::load(path);
  const auto& meta = f.meta();
  std::vector<std::string> keys = meta.value("keys", std::vector<std::string>{});
  if (keys.empty()) fail(path + ": no stats pools");
  auto has = [&](const std::string& k) {
    return std::find(keys.begin(), keys.end(), k) != keys.end();
  };
  std::string key;
  if (has(dataset_type)) {
    key = dataset_type;
  } else {
    std::string like;
    if (meta.contains("fallback") && meta["fallback"].contains(dataset_type))
      like = meta["fallback"][dataset_type].get<std::string>();
    else if (dataset_type == "general")
      like = "objaverse";
    if (!like.empty() && has(like)) key = like;
    else if (meta.value("all_equal", false)) key = keys.front();
    else fail("no normalization stats for '" + dataset_type + "' in " + path);
  }
  Stats s;
  s.key = key;
  s.mean_root = get_f64(f, key + "/mean_root");
  s.std_root = get_f64(f, key + "/std_root");
  s.mean_local = get_f64(f, key + "/mean_local");
  s.std_local = get_f64(f, key + "/std_local");
  return s;
}

AssetBundle load_bundle(const std::string& umtf_path, const std::string& glb_path) {
  UmtfFile f = UmtfFile::load(umtf_path);
  AssetBundle b;
  b.umtf_path = umtf_path;
  b.glb_path = glb_path;
  b.meta = f.meta();
  b.name = b.meta.value("name", std::string());
  b.stats_dataset = b.meta.value("stats_dataset", std::string("objaverse"));
  b.parents = get_i32(f, "parents");
  const int J = static_cast<int>(b.parents.size());
  b.num_joints = J;
  b.joint_names = b.meta.value("joint_names", std::vector<std::string>{});
  b.model_joint_names = b.meta.value("model_joint_names", std::vector<std::string>{});
  if (static_cast<int>(b.model_joint_names.size()) != J) b.model_joint_names = b.joint_names;
  if (static_cast<int>(b.joint_names.size()) != J) fail(umtf_path + ": joint_names size mismatch");
  b.tpos_first_frame = get_f64(f, "tpos_first_frame");
  b.tpos_offsets = get_f64(f, "tpos_offsets");
  b.tpos_local_rotations = get_f64(f, "tpos_local_rotations");
  b.tpos_global_rotations = get_f64(f, "tpos_global_rotations");
  b.joint_relations = get_i32(f, "joint_relations");
  b.joint_graph_dists = get_i32(f, "joint_graph_dists");
  b.joint_depths = get_i32(f, "joint_depths");
  if (f.has("spectral_feats")) {
    const HostArray& sp = f.get("spectral_feats");
    b.spectral_k = static_cast<int>(sp.dim(-1));
    b.spectral_feats = sp.to_f32();
  }
  if (b.meta.contains("scale_factor") && b.meta["scale_factor"].is_number())
    b.scale_factor = b.meta["scale_factor"].get<double>();
  if (static_cast<int>(b.tpos_first_frame.size()) != J * 3 ||
      static_cast<int>(b.tpos_offsets.size()) != J * 3 ||
      static_cast<int>(b.tpos_global_rotations.size()) != J * 4 ||
      static_cast<int>(b.joint_relations.size()) != J * J ||
      static_cast<int>(b.joint_graph_dists.size()) != J * J ||
      static_cast<int>(b.joint_depths.size()) != J)
    fail(umtf_path + ": inconsistent bundle tensor shapes");
  for (int j = 0; j < J; ++j)
    if (b.parents[j] >= J || b.parents[j] < -1 || b.parents[j] == j)
      fail(umtf_path + ": bad parent index");

  b.joint_node.assign(J, -1);
  if (!glb_path.empty()) {
    GlbFile g = read_glb(glb_path);
    b.glb_nodes = glb_nodes(g.json);
    for (int j = 0; j < J; ++j) b.joint_node[j] = find_node(g.json, b.glb_nodes, b.joint_names[j]);
    if (g.json.contains("skins") && !g.json["skins"].empty() &&
        g.json["skins"][0].contains("joints"))
      for (const auto& n : g.json["skins"][0]["joints"]) b.skin_joints.push_back(n.get<int>());
    int matched = 0;
    for (int n : b.joint_node) matched += n >= 0;
    // upstream drive_and_export refuses a character holding < 50% of the joints
    if (matched * 2 < J)
      fail(glb_path + ": only " + std::to_string(matched) + " of " + std::to_string(J) +
           " joints found as nodes; not the character of this bundle");
  }
  return b;
}

}  // namespace um::pipeline

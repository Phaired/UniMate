// recover_features / recover_motion vs the upstream recovery and the numpy
// re-derivation of the Blender keyframe + glTF export step
// (tests/golden/pipeline/<asset>_motion.umtf), from the golden sample.
#include <algorithm>
#include <cmath>

#include "pipeline/pipeline.h"
#include "test_common.h"

using namespace um;
using namespace um::pipeline;

template <class A, class B>
static double maxdiff(const A& a, const B& b) {
  if (a.size() != b.size()) return INFINITY;
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(double(a[i]) - double(b[i])));
  return m;
}

// quaternion sets compared up to sign; returns max component diff
static double qdiff(const float* a, const float* b, size_t n) {
  double m = 0;
  for (size_t i = 0; i < n; ++i) {
    double dp = 0, dm = 0;
    for (int k = 0; k < 4; ++k) {
      dp = std::max(dp, std::abs(double(a[i * 4 + k]) - b[i * 4 + k]));
      dm = std::max(dm, std::abs(double(a[i * 4 + k]) + b[i * 4 + k]));
    }
    m = std::max(m, std::min(dp, dm));
  }
  return m;
}

static void check_asset(const std::string& asset) {
  const std::string mpath = pt::golden(asset + "_motion.umtf"), spath = pt::golden(asset + "_sample.umtf");
  if (!pt::exists(mpath) || !pt::exists(spath) || !pt::exists(pt::bundle_umtf(asset))) {
    std::printf("[%s] skipped\n", asset.c_str());
    return;
  }
  UmtfFile gm = UmtfFile::load(mpath), gs = UmtfFile::load(spath);
  AssetBundle b = load_bundle(pt::bundle_umtf(asset), pt::glb_of(asset));
  Stats s = load_stats(pt::stats_umtf(), b.stats_dataset);
  const HostArray& sample = gs.get(gm.meta()["sample_key"].get<std::string>());
  const int T = gm.meta()["num_frames"].get<int>(), J = b.num_joints;

  RecoveredMotion r = recover_features(b, s, sample, T);
  auto feat_ref = gm.get("motion").to_f32();
  std::vector<float> rr(r.root_rot.begin(), r.root_rot.end());
  auto rr_ref = gm.get("root_rot_quat").to_f32();
  struct Row { const char* what; double diff, scale, tol; };
  std::vector<Row> rows = {
      {"de-normalised features", maxdiff(r.features, feat_ref), pt::max_abs(feat_ref), 1e-6},
      {"root facing quat", qdiff(rr.data(), rr_ref.data(), T), 1.0, 1e-5},
      {"root trajectory", maxdiff(r.root_pos, gm.get("root_pos").to_f32()), pt::max_abs(gm.get("root_pos").to_f32()), 1e-5},
      {"FK positions", maxdiff(r.positions, gm.get("anim_positions").to_f32()), pt::max_abs(gm.get("anim_positions").to_f32()), 1e-4},
      {"RIC positions", maxdiff(r.positions_ric, gm.get("positions_ric").to_f32()), pt::max_abs(gm.get("positions_ric").to_f32()), 1e-4},
  };
  std::vector<float> bvh(r.bvh_rot.begin(), r.bvh_rot.end());
  auto bvh_ref = gm.get("bvh_rotations").to_f32();
  rows.push_back({"BVH local rotations", qdiff(bvh.data(), bvh_ref.data(), size_t(T) * J), 1.0, 1e-4});

  MotionClip clip = recover_motion(b, s, sample, T, 30.f);
  auto lr = gm.get("gltf_local_r").to_f32();   // [T,J,4] xyzw, joint order
  auto lt = gm.get("gltf_local_t").to_f32();   // [T,J,3]
  double rot_d = 0, root_t_d = 0, rest_t_d = 0;
  int tracks = 0;
  for (const auto& tr : clip.tracks) {
    int j = static_cast<int>(std::find(b.joint_names.begin(), b.joint_names.end(), tr.node_name) - b.joint_names.begin());
    if (j >= J) continue;
    ++tracks;
    std::vector<float> ref(size_t(T) * 4);
    for (int t = 0; t < T; ++t)
      for (int k = 0; k < 4; ++k) ref[t * 4 + k] = lr[(size_t(t) * J + j) * 4 + k];
    rot_d = std::max(rot_d, qdiff(tr.rotation.data(), ref.data(), T));
    if (!tr.translation.empty()) {
      for (int t = 0; t < T; ++t)
        for (int k = 0; k < 3; ++k)
          root_t_d = std::max(root_t_d, std::abs(double(tr.translation[t * 3 + k]) - lt[(size_t(t) * J + j) * 3 + k]));
    } else {  // not keyed: the re-derived export must keep the rest translation
      const auto& rest = b.glb_nodes[b.joint_node[j]].translation;
      for (int t = 0; t < T; ++t)
        for (int k = 0; k < 3; ++k)
          rest_t_d = std::max(rest_t_d, std::abs(rest[k] - lt[(size_t(t) * J + j) * 3 + k]));
    }
  }
  PT_CHECK(tracks == J, "%d tracks for %d joints", tracks, J);
  rows.push_back({"glTF local rotations", rot_d, 1.0, 1e-4});
  rows.push_back({"glTF root translation", root_t_d, pt::max_abs(lt), 1e-4});
  rows.push_back({"golden non-root t vs rest", rest_t_d, pt::max_abs(lt), 1e-5});
  rows.push_back({"debug skeleton positions", maxdiff(clip.joint_positions, gm.get("anim_positions").to_f32()),
                  pt::max_abs(gm.get("anim_positions").to_f32()), 1e-4});
  for (const auto& rw : rows) {
    std::printf("[%s] %-26s max|diff| %.3g  (scale %.3g)\n", asset.c_str(), rw.what, rw.diff, rw.scale);
    PT_CHECK(rw.diff <= rw.tol * std::max(1.0, rw.scale), "%s: %s %g", asset.c_str(), rw.what, rw.diff);
  }
  PT_CHECK(clip.num_frames == T && clip.num_joints == J && clip.fps == 30.f, "clip header");

  // trimming
  MotionClip shortc = recover_motion(b, s, sample, 24, 30.f);
  PT_CHECK(shortc.num_frames == 24 && shortc.tracks[0].rotation.size() == 24 * 4, "trim");
}

int main() {
  pt::require_or_skip({pt::stats_umtf()});
  check_asset("eagle");
  check_asset("shark");
  return pt::finish("test_motion");
}

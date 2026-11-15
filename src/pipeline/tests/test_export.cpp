// export_glb round trip: re-read with cgltf (parse + load buffers + validate),
// check the animation's channels / accessors, then evaluate the exported
// tracks with cgltf's node transforms and compare the joint world positions
// with the golden upstream FK positions mapped by the asset's
// canonical->world transform.
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>

#include "pipeline/pipeline.h"
#include "test_common.h"

using namespace um;
using namespace um::pipeline;

static void check_asset(const std::string& asset) {
  const std::string mpath = pt::golden(asset + "_motion.umtf"), spath = pt::golden(asset + "_sample.umtf");
  const std::string glb = pt::glb_of(asset);
  if (!pt::exists(mpath) || !pt::exists(spath) || !pt::exists(glb) || !pt::exists(pt::bundle_umtf(asset))) {
    std::printf("[%s] skipped\n", asset.c_str());
    return;
  }
  UmtfFile gm = UmtfFile::load(mpath), gs = UmtfFile::load(spath);
  AssetBundle b = load_bundle(pt::bundle_umtf(asset), glb);
  Stats s = load_stats(pt::stats_umtf(), b.stats_dataset);
  const int T = gm.meta()["num_frames"].get<int>(), J = b.num_joints;
  MotionClip clip = recover_motion(b, s, gs.get(gm.meta()["sample_key"].get<std::string>()), T, 30.f);

  std::filesystem::path out = std::filesystem::temp_directory_path() / ("um_pipeline_" + asset + "_anim.glb");
  export_glb(glb, clip, out.string());

  cgltf_options opt{};
  cgltf_data* data = nullptr;
  cgltf_result res = cgltf_parse_file(&opt, out.string().c_str(), &data);
  PT_CHECK(res == cgltf_result_success, "cgltf_parse_file %d", int(res));
  if (res != cgltf_result_success) return;
  res = cgltf_load_buffers(&opt, data, out.string().c_str());
  PT_CHECK(res == cgltf_result_success, "cgltf_load_buffers %d", int(res));
  res = cgltf_validate(data);
  PT_CHECK(res == cgltf_result_success, "cgltf_validate %d", int(res));

  // the original asset is untouched apart from the animation
  cgltf_data* src = nullptr;
  cgltf_parse_file(&opt, glb.c_str(), &src);
  PT_CHECK(src && src->nodes_count == data->nodes_count && src->meshes_count == data->meshes_count &&
               src->skins_count == data->skins_count && src->images_count == data->images_count,
           "scene content changed");
  if (src) cgltf_free(src);

  PT_CHECK(data->animations_count == 1, "animations %zu", data->animations_count);
  if (data->animations_count != 1) {
    cgltf_free(data);
    return;
  }
  const cgltf_animation& an = data->animations[0];
  size_t expect = 0;
  for (const auto& tr : clip.tracks) expect += !tr.rotation.empty() + !tr.translation.empty() + !tr.scale.empty();
  PT_CHECK(an.channels_count == expect, "channels %zu vs %zu", an.channels_count, expect);
  int bad_acc = 0;
  for (size_t i = 0; i < an.samplers_count; ++i) {
    const cgltf_animation_sampler& sm = an.samplers[i];
    bad_acc += sm.input->count != size_t(T) || sm.output->count != size_t(T) ||
               sm.interpolation != cgltf_interpolation_type_linear || !sm.input->has_max ||
               std::abs(sm.input->max[0] - (T - 1) / 30.0) > 1e-6;
  }
  PT_CHECK(bad_acc == 0, "%d bad sampler accessors", bad_acc);
  std::printf("[%s] exported %s: %zu channels, %zu samplers, validate ok\n", asset.c_str(),
              out.filename().string().c_str(), an.channels_count, an.samplers_count);

  // evaluate the exported animation with cgltf and compare joint world positions
  auto ref = gm.get("anim_positions").to_f32();      // canonical FK positions [T,J,3]
  auto S = canonical_to_world(b);
  std::map<std::string, cgltf_node*> by_name;
  for (size_t i = 0; i < data->nodes_count; ++i)
    if (data->nodes[i].name) by_name[data->nodes[i].name] = &data->nodes[i];
  double worst = 0, worst_rest = 0;
  for (int t = 0; t < T; ++t) {
    for (size_t c = 0; c < an.channels_count; ++c) {
      const cgltf_animation_channel& ch = an.channels[c];
      cgltf_node* n = ch.target_node;
      if (ch.target_path == cgltf_animation_path_type_rotation) {
        cgltf_accessor_read_float(ch.sampler->output, t, n->rotation, 4);
        n->has_rotation = 1;
      } else if (ch.target_path == cgltf_animation_path_type_translation) {
        cgltf_accessor_read_float(ch.sampler->output, t, n->translation, 3);
        n->has_translation = 1;
      }
    }
    for (int j = 0; j < J; ++j) {
      auto it = by_name.find(b.joint_names[j]);
      if (it == by_name.end()) continue;
      float m[16];
      cgltf_node_transform_world(it->second, m);  // column-major
      const float* p = &ref[(size_t(t) * J + j) * 3];
      for (int k = 0; k < 3; ++k) {
        double w = S[k * 4 + 0] * p[0] + S[k * 4 + 1] * p[1] + S[k * 4 + 2] * p[2] + S[k * 4 + 3];
        worst = std::max(worst, std::abs(double(m[12 + k]) - w));
      }
    }
  }
  // sanity: rest pose of the GLB == cond T-pose through S
  for (int j = 0; j < J; ++j) {
    int nd = b.joint_node[j];
    if (nd < 0) continue;
    const auto& W = b.glb_nodes[nd].rest_world;
    for (int k = 0; k < 3; ++k) {
      double w = S[k * 4 + 0] * b.tpos_first_frame[j * 3] + S[k * 4 + 1] * b.tpos_first_frame[j * 3 + 1] +
                 S[k * 4 + 2] * b.tpos_first_frame[j * 3 + 2] + S[k * 4 + 3];
      worst_rest = std::max(worst_rest, std::abs(W[k * 4 + 3] - w));
    }
  }
  double scale = pt::max_abs(ref);
  std::printf("[%s] FK of exported tracks vs upstream joint positions: max|diff| %.3g (scale %.3g); "
              "GLB rest vs cond T-pose %.3g; S scale %.6f\n",
              asset.c_str(), worst, scale, worst_rest,
              std::sqrt(S[0] * S[0] + S[4] * S[4] + S[8] * S[8]));
  PT_CHECK(worst < 1e-4 * std::max(1.0, scale), "%s FK diff %g", asset.c_str(), worst);
  cgltf_free(data);
  std::filesystem::remove(out);
}

int main() {
  pt::require_or_skip({pt::stats_umtf()});
  check_asset("eagle");
  check_asset("shark");
  return pt::finish("test_export");
}

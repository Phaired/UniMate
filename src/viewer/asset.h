// CPU side of the viewer: GLB loading (cgltf), node hierarchy, skins, pose
// evaluation and MotionClip sampling. No Vulkan in here.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "core/generation.h"

namespace um::viewer {

struct Vertex {
  float pos[3];
  float normal[3];
  float uv[2];
  uint32_t joints[4];   // indices into the skin's joint list (or 0 for rigid meshes)
  float weights[4];
};
static_assert(sizeof(Vertex) == 64);

struct Material {
  glm::vec4 base_color{1.f};
  int image = -1;          // index into Asset::images, -1 = none
  float alpha_cutoff = 0;  // > 0 for alphaMode MASK
  bool blend = false;      // alphaMode BLEND: drawn after opaque geometry, sorted, no depth write
};

struct ImageRGBA8 {
  int width = 0, height = 0;
  std::vector<uint8_t> pixels;  // width*height*4, sRGB-encoded
};

// One draw: a glTF primitive instanced by one node.
struct Draw {
  uint32_t first_index = 0, index_count = 0;
  int32_t vertex_offset = 0;
  int material = -1;
  uint32_t palette_base = 0;  // offset of this draw's matrices in the palette
  // For back-to-front sorting of blended draws: centroid of the primitive in
  // vertex (bind) space and the palette entry (relative to palette_base) that
  // carries most of its skin weight. World centre = palette[base + joint] * centre.
  glm::vec3 center{0.f};
  uint32_t center_joint = 0;
};

struct Node {
  std::string name;  // glTF name, or "node_<i>" when unnamed
  int parent = -1;
  std::vector<int> children;
  glm::vec3 t{0.f};
  glm::quat r{1.f, 0.f, 0.f, 0.f};
  glm::vec3 s{1.f};
  int mesh = -1, skin = -1;
};

struct Skin {
  std::vector<int> joints;               // node indices
  std::vector<glm::mat4> inverse_bind;   // one per joint
  uint32_t palette_base = 0;
};

struct Asset {
  std::string path;
  std::vector<Node> nodes;
  std::vector<int> order;  // node indices, parents before children
  std::unordered_map<std::string, int> node_by_name;

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  std::vector<Draw> draws;
  std::vector<Material> materials;  // last one is the default material
  std::vector<ImageRGBA8> images;
  std::vector<Skin> skins;

  // Palette layout: [skin0 joints][skin1 joints]...[one slot per rigid mesh node]
  std::vector<std::pair<int, uint32_t>> rigid_slots;  // (node, palette index)
  uint32_t palette_size = 0;

  std::vector<int> skeleton;      // nodes used as joints by any skin (sorted)
  std::vector<int> skeleton_parent;  // for each skeleton entry: index into skeleton or -1

  glm::vec3 bounds_min{0.f}, bounds_max{0.f};  // bind/rest pose, world space

  std::vector<um::MotionClip> embedded_clips;
};

bool load_asset(const std::string& path, Asset& out, std::string& err, bool load_images = true,
                float clip_fps = 30.f);

struct Pose {
  std::vector<glm::vec3> t;
  std::vector<glm::quat> r;
  std::vector<glm::vec3> s;
  std::vector<glm::mat4> global;
};

void rest_pose(const Asset& a, Pose& p);  // local TRS = rest, globals computed
void compute_globals(const Asset& a, Pose& p);
void build_palette(const Asset& a, const Pose& p, std::vector<glm::mat4>& palette);

// Track index -> node index (or -1 when the name is unknown).
std::vector<int> bind_clip(const Asset& a, const um::MotionClip& clip);
// Sets the local TRS of the bound nodes at fractional frame `f` (lerp/nlerp
// between neighbouring frames; wraps to frame 0 when `loop`). Unbound nodes and
// empty channels keep their rest value. Then recomputes globals.
void sample_clip(const Asset& a, const um::MotionClip& clip, const std::vector<int>& binding,
                 float f, bool loop, Pose& p);

// True when the clip carries a usable debug skeleton (num_joints, parents and
// num_frames*num_joints*3 joint_positions).
bool has_debug_skeleton(const um::MotionClip& clip);
// Joint positions of the clip's debug skeleton at fractional frame `f`, with
// the same interpolation / wrapping rules as sample_clip. Empty if none.
void sample_debug_skeleton(const um::MotionClip& clip, float f, bool loop, std::vector<glm::vec3>& out);

}  // namespace um::viewer

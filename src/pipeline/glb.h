// Minimal GLB (binary glTF 2.0) container I/O and node-hierarchy queries, plus
// the exporter that adds a MotionClip as a glTF animation.
//
// The pipeline does its own GLB parsing (header + JSON chunk + BIN chunk with
// nlohmann_json) instead of cgltf so that um_pipeline never defines
// CGLTF_IMPLEMENTATION (the viewer already does; two definitions would clash
// at link time). Tests re-read the exported files with cgltf.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "core/generation.h"

namespace um::pipeline {

struct GlbFile {
  nlohmann::json json;
  std::vector<uint8_t> bin;  // BIN chunk (buffer 0); empty when absent
  bool has_bin = false;
};

GlbFile read_glb(const std::string& path);   // throws std::runtime_error
void write_glb(const std::string& path, const GlbFile& glb);

// Rest-pose node hierarchy of a glTF scene.
struct GlbNode {
  std::string name;
  int parent = -1;
  std::array<double, 3> translation{0, 0, 0};
  std::array<double, 4> rotation{0, 0, 0, 1};  // x y z w (glTF order)
  std::array<double, 3> scale{1, 1, 1};
  std::array<double, 16> local{};              // row-major, column-vector convention
  std::array<double, 16> rest_world{};         // product of locals from the scene root
};

std::vector<GlbNode> glb_nodes(const nlohmann::json& gltf);
// Index of the node called `name`, or -1. When several nodes share the name,
// a node used as a skin joint wins.
int find_node(const nlohmann::json& gltf, const std::vector<GlbNode>& nodes,
              const std::string& name);

struct ExportOptions {
  std::string animation_name;      // default: clip.name
  bool replace_animations = true;  // drop animations already in the file
};

// Copy of `src_glb` with `clip` added as one glTF animation (LINEAR samplers,
// one shared time accessor t_f = f / fps). Tracks are matched to nodes by
// name; tracks naming no node are skipped, empty channels are not written.
void export_glb(const std::string& src_glb, const um::MotionClip& clip,
                const std::string& out_glb, const ExportOptions& opt = {});

}  // namespace um::pipeline

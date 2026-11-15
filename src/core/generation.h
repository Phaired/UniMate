// Types shared between the generation pipeline (producer), the GLB exporter
// and the real-time viewer (consumers). Plain data, no dependencies.
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace um {

// Animation of one glTF node, in that node's *local* glTF space (the same
// space as node.translation / node.rotation / node.scale), sampled at every
// frame of the clip. An empty channel means "keep the node's rest value".
struct NodeTrack {
  std::string node_name;            // glTF node (joint) name in the asset GLB
  std::vector<float> translation;   // num_frames * 3            (x y z)
  std::vector<float> rotation;      // num_frames * 4 quaternion (x y z w)
  std::vector<float> scale;         // num_frames * 3
};

struct MotionClip {
  std::string name;
  float fps = 30.f;
  int num_frames = 0;
  std::vector<NodeTrack> tracks;

  // Optional debug skeleton in the model's canonical space, for drawing bones
  // without a mesh: joint_positions is num_frames * num_joints * 3.
  int num_joints = 0;
  std::vector<int> parents;               // -1 for the root
  std::vector<std::string> joint_names;
  std::vector<float> joint_positions;
};

struct GenerateRequest {
  std::string prompt;
  uint32_t seed = 0;
  float cfg_scale = 3.0f;
  int num_frames = 60;          // <= model max_motion_length
  std::string solver = "dopri5";  // "dopri5" | "euler" | "heun"
  int num_steps = 50;           // fixed-step solvers
};

// Progress in [0, 1] plus a short status string; may be called from a worker
// thread.
using ProgressFn = std::function<void(float, const std::string&)>;

// What the viewer calls when the user presses "Generate". Runs on a worker
// thread; throws std::exception on failure.
using GenerateFn = std::function<MotionClip(const GenerateRequest&, const ProgressFn&)>;

}  // namespace um

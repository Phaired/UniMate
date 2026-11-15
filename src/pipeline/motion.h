// From a generated sample to per-node glTF animation tracks.
//
// Upstream chain: unimate/utils/visualization.py (de-normalise with the cond
// mean/std, float32) -> data_process/mesh_animation/animate_motion.py
// build_anim_from_npy (anim_mode 'fk', canonical asset: scale 1) ->
// motion_utils.recover_unimate_anim_from_rot (root trajectory from the
// velocity channels, BVH rotations = chordal mean of the HML child copies,
// offsets = cond tpos_offsets) -> common.drive_and_export /
// blender_rig.compute_bone_keyframes (Blender pose keys conjugated by the
// T-pose global rotations) -> Blender glTF export.
//
// The Blender part is re-derived (no Blender here). In the canonical frame,
// the posed armature-space bone matrix is M_j = [G_j T_j | X_j] with G_j the
// FK global rotation of the BVH rotations, T_j the rest (T-pose) bone
// rotation and X_j the FK position, i.e. the world delta from rest is
// D_j = [G_j | X_j - G_j P_j] (P_j: rest position). motion_axis_alignment
// makes this independent of the bone axes the importer chose. On the GLB:
//   world_j(f) = S D_j(f) S^-1 W_j
// with W_j the node's rest world matrix and S the similarity canonical ->
// glTF world fitted on the joint rest positions (identity for
// rig_preprocess canonical assets). Node-local tracks are then
// parent_world^-1 world_j. As in the Blender export only the rotations of
// every joint and the translation of the root joint(s) are animated (other
// bones keep their rest translation; scale is never keyed).
#pragma once

#include <string>
#include <vector>

#include "core/generation.h"
#include "core/umtf.h"
#include "pipeline/bundle.h"

namespace um::pipeline {

// Intermediate results (float64), all in the canonical model frame.
struct RecoveredMotion {
  int num_frames = 0, num_joints = 0;
  std::vector<float> features;       // [T*J*12] de-normalised (float32 like upstream)
  std::vector<double> root_rot;      // [T*4]  w x y z  (facing quaternion, channels 3:9)
  std::vector<double> root_pos;      // [T*3]  integrated root trajectory
  std::vector<double> bvh_rot;       // [T*J*4] w x y z local (BVH-order) rotations
  std::vector<double> global_rot;    // [T*J*9] FK global rotation matrices
  std::vector<double> positions;     // [T*J*3] FK joint positions (offsets = tpos_offsets)
  std::vector<double> positions_ric; // [T*J*3] RIC-channel positions (diagnostics)
};

// sample: [1, J_max, 12, F_max] (or [J_max, 12, F_max]) normalised model output.
// num_frames <= F_max (0 = all).
RecoveredMotion recover_features(const AssetBundle& bundle, const Stats& stats,
                                 const um::HostArray& sample, int num_frames);

// Full recovery to glTF node-local TRS tracks keyed by the GLB's joint node
// names (the cond joint names). Joints the GLB lacks get no track; GLB joints
// the model does not have keep their rest pose and follow their parent
// (upstream's default 'merge' strategy instead deletes such bones and moves
// their skin weights to the nearest kept ancestor -- a mesh edit not done
// here). Consecutive rotation quaternions are kept in one hemisphere. The
// debug skeleton holds the canonical FK positions, cond parents and joint names.
um::MotionClip recover_motion(const AssetBundle& bundle, const Stats& stats,
                              const um::HostArray& sample, int num_frames, float fps = 30.f);

// The similarity S (row-major 4x4) mapping the canonical frame onto the GLB's
// world frame, fitted on the joint rest positions (Umeyama, no reflection);
// identity without a GLB.
std::vector<double> canonical_to_world(const AssetBundle& bundle);

}  // namespace um::pipeline

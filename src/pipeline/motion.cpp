#include "pipeline/motion.h"

#include <cmath>
#include <functional>
#include <stdexcept>

#include "pipeline/condition.h"
#include "pipeline/math3.h"

namespace um::pipeline {

using namespace m3;

namespace {
[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("recover_motion: " + m); }

Mat4 row4(const double* p) {
  Mat4 m;
  for (int i = 0; i < 16; ++i) m[i] = p[i];
  return m;
}
}  // namespace

std::vector<double> canonical_to_world(const AssetBundle& b) {
  Mat4 S = mat4_identity();
  if (b.glb_nodes.empty()) return std::vector<double>(S.begin(), S.end());
  std::vector<Vec3> X, Y;
  for (int j = 0; j < b.num_joints; ++j) {
    int nd = b.joint_node[j];
    if (nd < 0) continue;
    X.push_back({b.tpos_first_frame[j * 3], b.tpos_first_frame[j * 3 + 1], b.tpos_first_frame[j * 3 + 2]});
    Y.push_back(translation(b.glb_nodes[nd].rest_world));
  }
  if (X.size() < 3) return std::vector<double>(S.begin(), S.end());
  Vec3 mx{0, 0, 0}, my{0, 0, 0};
  for (size_t i = 0; i < X.size(); ++i) mx = add(mx, X[i]), my = add(my, Y[i]);
  mx = scale(mx, 1.0 / X.size());
  my = scale(my, 1.0 / X.size());
  Mat3 H{};
  double xx = 0;
  for (size_t i = 0; i < X.size(); ++i) {
    Vec3 xc = sub(X[i], mx), yc = sub(Y[i], my);
    for (int a = 0; a < 3; ++a)
      for (int c = 0; c < 3; ++c) H[a * 3 + c] += yc[a] * xc[c];
    xx += dot(xc, xc);
  }
  if (xx < 1e-30) return std::vector<double>(S.begin(), S.end());
  Mat3 R = nearest_rotation(H);
  double tr = 0;
  for (int i = 0; i < 9; ++i) tr += R[i] * H[i];  // trace(R^T H)
  double s = tr / xx;
  Vec3 t = sub(my, scale(mul(R, mx), s));
  Mat3 sR = R;
  for (double& v : sR) v *= s;
  S = compose(sR, t);
  return std::vector<double>(S.begin(), S.end());
}

RecoveredMotion recover_features(const AssetBundle& b, const Stats& st, const HostArray& sample,
                                 int num_frames) {
  const int n = b.num_joints;
  Shape sh = sample.shape;
  if (sh.size() == 4) {
    if (sh[0] < 1) fail("empty batch");
    sh.erase(sh.begin());
  }
  if (sh.size() != 3 || sh[1] != 12) fail("sample must be [1, J, 12, F], got " + shape_str(sample.shape));
  const int Jm = static_cast<int>(sh[0]), Fm = static_cast<int>(sh[2]);
  if (n > Jm) fail("sample has fewer joints than the bundle");
  const int T = (num_frames > 0 && num_frames <= Fm) ? num_frames : Fm;
  std::vector<float> x = sample.to_f32();  // first batch element is at offset 0

  RecoveredMotion r;
  r.num_frames = T;
  r.num_joints = n;
  // ---- de-normalise (float32: motion * std + mean, visualization.py) ----
  r.features.resize(static_cast<size_t>(T) * n * 12);
  for (int j = 0; j < n; ++j) {
    const double *m, *s;
    joint_mean_std(st, j, m, s);
    for (int d = 0; d < 12; ++d) {
      const float sf = static_cast<float>(s[d]), mf = static_cast<float>(m[d]);
      for (int t = 0; t < T; ++t) {
        float v = x[(static_cast<size_t>(j) * 12 + d) * Fm + t];
        float prod = v * sf;
        r.features[(static_cast<size_t>(t) * n + j) * 12 + d] = prod + mf;
      }
    }
  }
  auto feat = [&](int t, int j, int d) {
    return static_cast<double>(r.features[(static_cast<size_t>(t) * n + j) * 12 + d]);
  };

  // ---- root: facing quaternion and integrated trajectory ----
  r.root_rot.resize(static_cast<size_t>(T) * 4);
  r.root_pos.assign(static_cast<size_t>(T) * 3, 0.0);
  std::vector<Quat> rq(T);
  for (int t = 0; t < T; ++t) {
    double c6[6];
    for (int k = 0; k < 6; ++k) c6[k] = feat(t, 0, 3 + k);
    rq[t] = mat_to_quat(rot6d_to_mat(c6));
    for (int k = 0; k < 4; ++k) r.root_rot[t * 4 + k] = rq[t][k];
  }
  Vec3 acc{0, 0, 0};
  for (int t = 0; t < T; ++t) {
    Vec3 v{0, 0, 0};
    if (t > 0) v = {feat(t - 1, 0, 9), 0.0, feat(t - 1, 0, 11)};
    acc = add(acc, qrot(qconj(rq[t]), v));
    r.root_pos[t * 3 + 0] = acc[0];
    r.root_pos[t * 3 + 1] = feat(t, 0, 1);
    r.root_pos[t * 3 + 2] = acc[2];
  }

  // ---- BVH rotations: chordal mean of the HML copies held by the children ----
  std::vector<int> has_child(n, 0);
  for (int j = 1; j < n; ++j)
    if (b.parents[j] >= 0) has_child[b.parents[j]] = 1;
  r.bvh_rot.resize(static_cast<size_t>(T) * n * 4);
  r.global_rot.resize(static_cast<size_t>(T) * n * 9);
  r.positions.resize(static_cast<size_t>(T) * n * 3);
  r.positions_ric.resize(static_cast<size_t>(T) * n * 3);
  std::vector<Mat3> sum(n), loc(n), glob(n);
  std::vector<Vec3> pos(n);
  for (int t = 0; t < T; ++t) {
    for (int j = 0; j < n; ++j) sum[j] = Mat3{};
    for (int j = 1; j < n; ++j) {
      int p = b.parents[j];
      if (p < 0) continue;
      double c6[6];
      for (int k = 0; k < 6; ++k) c6[k] = feat(t, j, 3 + k);
      Mat3 m = rot6d_to_mat(c6);
      for (int i = 0; i < 9; ++i) sum[p][i] += m[i];
    }
    for (int j = 0; j < n; ++j) {
      Mat3 R = has_child[j] ? nearest_rotation(sum[j]) : mat3_identity();
      Quat q = mat_to_quat(R);
      for (int k = 0; k < 4; ++k) r.bvh_rot[(static_cast<size_t>(t) * n + j) * 4 + k] = q[k];
      loc[j] = quat_to_mat(q);  // Animation FK uses Quaternions.transforms()
    }
    // FK (Animation.positions_global): root position = trajectory, others = tpos offsets
    for (int j = 0; j < n; ++j) {
      int p = b.parents[j];
      if (p < 0) {
        glob[j] = loc[j];
        pos[j] = {r.root_pos[t * 3], r.root_pos[t * 3 + 1], r.root_pos[t * 3 + 2]};
      } else {
        Vec3 off{b.tpos_offsets[j * 3], b.tpos_offsets[j * 3 + 1], b.tpos_offsets[j * 3 + 2]};
        glob[j] = mul(glob[p], loc[j]);
        pos[j] = add(pos[p], mul(glob[p], off));
      }
      for (int i = 0; i < 9; ++i) r.global_rot[(static_cast<size_t>(t) * n + j) * 9 + i] = glob[j][i];
      for (int k = 0; k < 3; ++k) r.positions[(static_cast<size_t>(t) * n + j) * 3 + k] = pos[j][k];
    }
    // RIC positions (recover_unimate_joint_pos_from_ric)
    for (int j = 0; j < n; ++j) {
      Vec3 q;
      if (j == 0) {
        q = {r.root_pos[t * 3], r.root_pos[t * 3 + 1], r.root_pos[t * 3 + 2]};
      } else {
        q = qrot(qconj(rq[t]), Vec3{feat(t, j, 0), feat(t, j, 1), feat(t, j, 2)});
        q[0] += r.root_pos[t * 3];
        q[2] += r.root_pos[t * 3 + 2];
      }
      for (int k = 0; k < 3; ++k) r.positions_ric[(static_cast<size_t>(t) * n + j) * 3 + k] = q[k];
    }
  }
  return r;
}

MotionClip recover_motion(const AssetBundle& b, const Stats& st, const HostArray& sample,
                          int num_frames, float fps) {
  RecoveredMotion r = recover_features(b, st, sample, num_frames);
  const int T = r.num_frames, n = r.num_joints;
  MotionClip clip;
  clip.name = b.name;
  clip.fps = fps;
  clip.num_frames = T;
  clip.num_joints = n;
  clip.parents = b.parents;
  clip.joint_names = b.joint_names;
  clip.joint_positions.resize(r.positions.size());
  for (size_t i = 0; i < r.positions.size(); ++i) clip.joint_positions[i] = static_cast<float>(r.positions[i]);

  const bool has_glb = !b.glb_nodes.empty();
  std::vector<double> Sv = canonical_to_world(b);
  const Mat4 S = row4(Sv.data()), Sinv = inverse(S);

  // rest world matrix W_j of each joint, and the node graph to evaluate
  const int N = has_glb ? static_cast<int>(b.glb_nodes.size()) : n;
  std::vector<int> node_parent(N, -1), joint_of(N, -1);
  std::vector<Mat4> rest_local(N), rest_world(N);
  if (has_glb) {
    for (int i = 0; i < N; ++i) {
      node_parent[i] = b.glb_nodes[i].parent;
      rest_local[i] = row4(b.glb_nodes[i].local.data());
      rest_world[i] = row4(b.glb_nodes[i].rest_world.data());
    }
    for (int j = 0; j < n; ++j)
      if (b.joint_node[j] >= 0) joint_of[b.joint_node[j]] = j;
  } else {  // canonical skeleton, identity rest rotations
    for (int j = 0; j < n; ++j) {
      node_parent[j] = b.parents[j];
      joint_of[j] = j;
      Vec3 P{b.tpos_first_frame[j * 3], b.tpos_first_frame[j * 3 + 1], b.tpos_first_frame[j * 3 + 2]};
      rest_world[j] = compose(mat3_identity(), P);
    }
    for (int j = 0; j < n; ++j)
      rest_local[j] = b.parents[j] < 0 ? rest_world[j] : mul(inverse(rest_world[b.parents[j]]), rest_world[j]);
  }

  // which joint nodes get tracks / translation keys
  struct Out { int node, joint; bool translate; };
  std::vector<Out> outs;
  for (int j = 0; j < n; ++j) {
    int nd = has_glb ? b.joint_node[j] : j;
    if (nd < 0) continue;
    int p = b.parents[j];
    bool translate = p < 0 || (has_glb ? node_parent[nd] != b.joint_node[p] : false);
    outs.push_back({nd, j, translate});
  }
  clip.tracks.resize(outs.size());
  for (size_t k = 0; k < outs.size(); ++k) {
    clip.tracks[k].node_name = b.joint_names[outs[k].joint];
    clip.tracks[k].rotation.resize(static_cast<size_t>(T) * 4);
    if (outs[k].translate) clip.tracks[k].translation.resize(static_cast<size_t>(T) * 3);
  }

  std::vector<Mat4> posed(N);
  std::vector<int> done(N);
  for (int t = 0; t < T; ++t) {
    std::fill(done.begin(), done.end(), 0);
    std::function<const Mat4&(int)> world = [&](int i) -> const Mat4& {
      if (done[i]) return posed[i];
      int j = joint_of[i];
      if (j >= 0) {
        const double* G = &r.global_rot[(static_cast<size_t>(t) * n + j) * 9];
        Mat3 Gm;
        for (int q = 0; q < 9; ++q) Gm[q] = G[q];
        Vec3 X{r.positions[(static_cast<size_t>(t) * n + j) * 3], r.positions[(static_cast<size_t>(t) * n + j) * 3 + 1],
               r.positions[(static_cast<size_t>(t) * n + j) * 3 + 2]};
        Vec3 P{b.tpos_first_frame[j * 3], b.tpos_first_frame[j * 3 + 1], b.tpos_first_frame[j * 3 + 2]};
        Mat4 D = compose(Gm, sub(X, mul(Gm, P)));
        posed[i] = mul(mul(mul(S, D), Sinv), rest_world[i]);
      } else if (node_parent[i] >= 0) {
        posed[i] = mul(world(node_parent[i]), rest_local[i]);
      } else {
        posed[i] = rest_local[i];
      }
      done[i] = 1;
      return posed[i];
    };
    for (size_t k = 0; k < outs.size(); ++k) {
      int nd = outs[k].node;
      const Mat4& W = world(nd);
      Mat4 L = node_parent[nd] >= 0 ? mul(inverse(world(node_parent[nd])), W) : W;
      Mat3 R = linear(L);
      for (int c = 0; c < 3; ++c) {
        double sc = std::sqrt(R[c] * R[c] + R[3 + c] * R[3 + c] + R[6 + c] * R[6 + c]);
        if (sc > 0)
          for (int rr = 0; rr < 3; ++rr) R[rr * 3 + c] /= sc;
      }
      Quat q = mat_to_quat(R);
      float* dst = &clip.tracks[k].rotation[static_cast<size_t>(t) * 4];
      float qx[4] = {static_cast<float>(q[1]), static_cast<float>(q[2]), static_cast<float>(q[3]),
                     static_cast<float>(q[0])};
      if (t > 0) {
        const float* prev = dst - 4;
        if (prev[0] * qx[0] + prev[1] * qx[1] + prev[2] * qx[2] + prev[3] * qx[3] < 0)
          for (float& v : qx) v = -v;
      }
      for (int c = 0; c < 4; ++c) dst[c] = qx[c];
      if (outs[k].translate) {
        float* tr = &clip.tracks[k].translation[static_cast<size_t>(t) * 3];
        tr[0] = static_cast<float>(L[3]);
        tr[1] = static_cast<float>(L[7]);
        tr[2] = static_cast<float>(L[11]);
      }
    }
  }
  return clip;
}

}  // namespace um::pipeline

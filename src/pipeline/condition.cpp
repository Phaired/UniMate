#include "pipeline/condition.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <stdexcept>
#include <unordered_map>

namespace um::pipeline {

namespace {
[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("build_condition: " + m); }

HostArray i64(const std::vector<int64_t>& v, Shape s) { return HostArray::from_i64(v, std::move(s)); }
HostArray f32(const std::vector<float>& v, Shape s) { return HostArray::from_f32(v, std::move(s)); }
HostArray u8(const std::vector<uint8_t>& v, Shape s) {
  HostArray a(DType::U8, std::move(s));
  std::copy(v.begin(), v.end(), a.bytes.begin());
  return a;
}
}  // namespace

void joint_mean_std(const Stats& s, int joint, const double*& mean, const double*& std) {
  mean = joint == 0 ? s.mean_root.data() : s.mean_local.data();
  std = joint == 0 ? s.std_root.data() : s.std_local.data();
}

std::vector<int64_t> compute_edge_indexs(const std::vector<int>& parents) {
  std::vector<int64_t> p, c;
  for (size_t j = 0; j < parents.size(); ++j)
    if (parents[j] >= 0 && parents[j] != static_cast<int>(j)) {
      p.push_back(parents[j]);
      c.push_back(static_cast<int64_t>(j));
    }
  std::vector<int64_t> out;
  out.reserve(4 * p.size());
  out.insert(out.end(), p.begin(), p.end());
  out.insert(out.end(), c.begin(), c.end());
  out.insert(out.end(), c.begin(), c.end());
  out.insert(out.end(), p.begin(), p.end());
  return out;
}

std::vector<int> compute_joint_depths(const std::vector<int>& parents) {
  const int n = static_cast<int>(parents.size());
  std::vector<int> d(n, -1);
  for (int j = 0; j < n; ++j)
    if (parents[j] == -1 || parents[j] == j) d[j] = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (int j = 0; j < n; ++j) {
      if (d[j] >= 0) continue;
      int p = parents[j];
      if (p >= 0 && p < n && d[p] >= 0) {
        d[j] = d[p] + 1;
        changed = true;
      }
    }
  }
  return d;
}

void compute_edge_relations_and_distances(const std::vector<int>& parents,
                                          std::vector<int>& rel, std::vector<int>& dist,
                                          int max_path_len) {
  const int n = static_cast<int>(parents.size());
  std::vector<std::vector<int>> children(n), adj(n);
  for (int c = 0; c < n; ++c) {
    int p = parents[c];
    if (p == -1) continue;
    children[p].push_back(c);
    adj[c].push_back(p);
    adj[p].push_back(c);
  }
  rel.assign(static_cast<size_t>(n) * n, 4);
  for (int i = 0; i < n; ++i) {
    int pi = parents[i];
    for (int j = 0; j < n; ++j) {
      int pj = parents[j];
      int& r = rel[static_cast<size_t>(i) * n + j];
      if (i == j) r = 0;
      else if (pj == i) r = 2;
      else if (j == pi && pi != -1) r = 1;
      else if (pi != -1 && pj == pi) r = 3;
    }
    if (children[i].empty()) rel[static_cast<size_t>(i) * n + i] = 5;
  }
  dist.assign(static_cast<size_t>(n) * n, max_path_len);
  for (int s = 0; s < n; ++s) {
    std::vector<int> d(n, 32767);
    d[s] = 0;
    std::deque<int> q{s};
    while (!q.empty()) {
      int u = q.front();
      q.pop_front();
      if (d[u] >= max_path_len) continue;
      for (int v : adj[u])
        if (d[v] > d[u] + 1) {
          d[v] = d[u] + 1;
          q.push_back(v);
        }
    }
    for (int j = 0; j < n; ++j) dist[static_cast<size_t>(s) * n + j] = std::min(d[j], max_path_len);
  }
}

std::vector<float> compute_laplacian_eigenvectors(const std::vector<int>& parents, int max_freqs) {
  const int n = static_cast<int>(parents.size());
  const int k = std::min(n - 1, max_freqs);
  // L_sym = I - D^-1/2 A D^-1/2 (PyG get_laplacian 'sym')
  std::vector<double> deg(n, 0.0), A(static_cast<size_t>(n) * n, 0.0);
  for (int c = 0; c < n; ++c) {
    int p = parents[c];
    if (p < 0 || p == c) continue;
    A[static_cast<size_t>(p) * n + c] = A[static_cast<size_t>(c) * n + p] = 1.0;
  }
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) deg[i] += A[static_cast<size_t>(i) * n + j];
  std::vector<double> L(static_cast<size_t>(n) * n, 0.0), V(static_cast<size_t>(n) * n, 0.0);
  for (int i = 0; i < n; ++i) {
    V[static_cast<size_t>(i) * n + i] = 1.0;
    for (int j = 0; j < n; ++j) {
      double a = A[static_cast<size_t>(i) * n + j];
      double w = (a != 0 && deg[i] > 0 && deg[j] > 0) ? a / std::sqrt(deg[i] * deg[j]) : 0.0;
      L[static_cast<size_t>(i) * n + j] = (i == j ? (deg[i] > 0 ? 1.0 : 0.0) : 0.0) - w;
    }
  }
  // cyclic Jacobi
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) off += L[static_cast<size_t>(p) * n + q] * L[static_cast<size_t>(p) * n + q];
    if (off < 1e-26) break;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        double apq = L[static_cast<size_t>(p) * n + q];
        if (std::abs(apq) < 1e-300) continue;
        double th = (L[static_cast<size_t>(q) * n + q] - L[static_cast<size_t>(p) * n + p]) / (2 * apq);
        double t = (th >= 0 ? 1.0 : -1.0) / (std::abs(th) + std::sqrt(th * th + 1));
        double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int r = 0; r < n; ++r) {
          double lrp = L[static_cast<size_t>(r) * n + p], lrq = L[static_cast<size_t>(r) * n + q];
          L[static_cast<size_t>(r) * n + p] = c * lrp - s * lrq;
          L[static_cast<size_t>(r) * n + q] = s * lrp + c * lrq;
        }
        for (int r = 0; r < n; ++r) {
          double lpr = L[static_cast<size_t>(p) * n + r], lqr = L[static_cast<size_t>(q) * n + r];
          L[static_cast<size_t>(p) * n + r] = c * lpr - s * lqr;
          L[static_cast<size_t>(q) * n + r] = s * lpr + c * lqr;
        }
        for (int r = 0; r < n; ++r) {
          double vrp = V[static_cast<size_t>(r) * n + p], vrq = V[static_cast<size_t>(r) * n + q];
          V[static_cast<size_t>(r) * n + p] = c * vrp - s * vrq;
          V[static_cast<size_t>(r) * n + q] = s * vrp + c * vrq;
        }
      }
  }
  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return L[static_cast<size_t>(a) * n + a] < L[static_cast<size_t>(b) * n + b];
  });
  std::vector<float> out(static_cast<size_t>(n) * max_freqs, 0.f);
  for (int c = 0; c < k; ++c) {
    int col = order[1 + c];
    double nn = 0, best = 0;
    int bi = 0;
    for (int r = 0; r < n; ++r) {
      double v = V[static_cast<size_t>(r) * n + col];
      nn += v * v;
      if (std::abs(v) > best) best = std::abs(v), bi = r;
    }
    double sgn = V[static_cast<size_t>(bi) * n + col] < 0 ? -1.0 : 1.0;
    nn = std::sqrt(nn);
    for (int r = 0; r < n; ++r)
      out[static_cast<size_t>(r) * max_freqs + c] =
          static_cast<float>(sgn * V[static_cast<size_t>(r) * n + col] / (nn > 1e-12 ? nn : 1.0));
  }
  return out;
}

um::ArrayDict build_condition(const AssetBundle& b, const Stats& st, const ModelDims& dims,
                              const std::string& prompt, int num_frames,
                              const TextEmbedFn& embed) {
  const int n = b.num_joints, J = dims.max_joints, F = dims.max_frames, D = dims.feature_len;
  if (n > J) fail(b.name + " has " + std::to_string(n) + " joints; the model is padded to " +
                  std::to_string(J));
  if (n < 2) fail(b.name + ": a skeleton needs at least 2 joints");
  if (D != 12) fail("feature_len must be 12 (UniMate representation)");
  if (num_frames <= 0 || num_frames > F) num_frames = F;
  if (static_cast<int>(st.mean_root.size()) != D || static_cast<int>(st.std_local.size()) != D)
    fail("stats width != feature_len");

  // ---- T-pose: add_cond_object root-XZ check/zeroing, then grounding ------
  std::vector<double> tpos = b.tpos_first_frame;  // [n*3]
  double extent = 0;
  for (int c = 0; c < 3; ++c) {
    double lo = 1e300, hi = -1e300;
    for (int j = 0; j < n; ++j) lo = std::min(lo, tpos[j * 3 + c]), hi = std::max(hi, tpos[j * 3 + c]);
    extent = std::max(extent, hi - lo);
  }
  if (extent == 0) extent = 1.0;
  double root_xz = std::max(std::abs(tpos[0]), std::abs(tpos[2]));
  if (root_xz > 1e-6 * extent)
    fail(b.name + ": the T-pose root is not at the XZ origin (not a canonical cond entry)");
  tpos[0] = 0.0;
  tpos[2] = 0.0;
  {
    double ymin = 1e300;
    for (int j = 0; j < n; ++j) ymin = std::min(ymin, tpos[j * 3 + 1]);
    for (int j = 0; j < n; ++j) tpos[j * 3 + 1] -= ymin;
  }

  ArrayDict cond;
  // ---- normalised T-pose features ------------------------------------------
  std::vector<float> tff(static_cast<size_t>(J) * D, 0.f), tffp(static_cast<size_t>(J) * D, 0.f);
  std::vector<float> mean(static_cast<size_t>(J) * D, 0.f), stdv(static_cast<size_t>(J) * D, 1.f);
  std::vector<double> norm(static_cast<size_t>(n) * D);
  static const double kPad[12] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0};  // identity 6D, zero vel
  for (int j = 0; j < n; ++j) {
    const double *m, *s;
    joint_mean_std(st, j, m, s);
    for (int d = 0; d < D; ++d) {
      double v = d < 3 ? tpos[j * 3 + d] : kPad[d];
      double x = (v - m[d]) / s[d];
      if (std::isnan(x)) x = 0.0;
      else if (std::isinf(x)) x = x > 0 ? 1.7976931348623157e308 : -1.7976931348623157e308;
      norm[static_cast<size_t>(j) * D + d] = x;
      tff[static_cast<size_t>(j) * D + d] = static_cast<float>(x);
      mean[static_cast<size_t>(j) * D + d] = static_cast<float>(m[d]);
      stdv[static_cast<size_t>(j) * D + d] = static_cast<float>(s[d]);
    }
  }
  for (int j = 0; j < n; ++j) {
    int src = b.parents[j] != -1 ? b.parents[j] : j;
    for (int d = 0; d < D; ++d) tffp[static_cast<size_t>(j) * D + d] = tff[static_cast<size_t>(src) * D + d];
  }

  // ---- offsets_from_positions(grounded tpos) --------------------------------
  std::vector<float> offsets(static_cast<size_t>(J) * 3, 0.f);
  for (int j = 0; j < n; ++j)
    for (int c = 0; c < 3; ++c) {
      double v = j == 0 ? tpos[c] : tpos[j * 3 + c] - tpos[b.parents[j] * 3 + c];
      offsets[static_cast<size_t>(j) * 3 + c] = static_cast<float>(v);
    }

  // ---- topology -------------------------------------------------------------
  std::vector<int64_t> rel(static_cast<size_t>(J) * J, 0), gd(static_cast<size_t>(J) * J, 0),
      depth(J, 0);
  for (int i = 0; i < n; ++i) {
    depth[i] = b.joint_depths[i];
    for (int k = 0; k < n; ++k) {
      rel[static_cast<size_t>(i) * J + k] = b.joint_relations[static_cast<size_t>(i) * n + k];
      gd[static_cast<size_t>(i) * J + k] = b.joint_graph_dists[static_cast<size_t>(i) * n + k];
    }
  }
  std::vector<float> spec_src;
  int K = b.spectral_k;
  if (K == dims.max_freqs && !b.spectral_feats.empty()) {
    spec_src = b.spectral_feats;
  } else {  // upstream create_condition recomputes on a width mismatch
    K = dims.max_freqs;
    spec_src = compute_laplacian_eigenvectors(b.parents, K);
  }
  std::vector<float> spec(static_cast<size_t>(J) * K, 0.f);
  for (int j = 0; j < n; ++j)
    for (int k = 0; k < K; ++k) spec[static_cast<size_t>(j) * K + k] = spec_src[static_cast<size_t>(j) * K + k];

  // ---- text ----------------------------------------------------------------
  std::vector<std::string> uniq;
  std::unordered_map<std::string, int> index;
  for (const auto& nm : b.model_joint_names)
    if (!nm.empty() && index.emplace(nm, static_cast<int>(uniq.size())).second) uniq.push_back(nm);
  const int T = dims.text_dim;
  std::vector<float> jemb(static_cast<size_t>(J) * T, 0.f);
  if (!uniq.empty()) {
    if (!embed) fail("no text embedder");
    HostArray e = embed(uniq);
    if (e.shape.size() != 2 || e.dim(0) != static_cast<int64_t>(uniq.size()) || e.dim(1) != T)
      fail("text embedder returned " + shape_str(e.shape) + " for " +
           std::to_string(uniq.size()) + " joint names, expected [N, " + std::to_string(T) + "]");
    auto v = e.to_f32();
    for (int j = 0; j < n; ++j) {
      auto it = index.find(b.model_joint_names[j]);
      if (it == index.end()) continue;
      std::copy_n(v.begin() + static_cast<size_t>(it->second) * T, T,
                  jemb.begin() + static_cast<size_t>(j) * T);
    }
  }
  std::vector<float> cap(T, 0.f);
  if (!prompt.empty()) {
    if (!embed) fail("no text embedder");
    HostArray e = embed({prompt});
    if (e.numel() != T) fail("caption embedding has " + std::to_string(e.numel()) + " values");
    cap = e.to_f32();
  }

  // ---- masks / sizes ----------------------------------------------------------
  std::vector<uint8_t> lmask(F), jmask(J);
  for (int f = 0; f < F; ++f) lmask[f] = f < num_frames;
  for (int j = 0; j < J; ++j) jmask[j] = j < n;
  std::vector<int64_t> parents64(b.parents.begin(), b.parents.end());
  std::vector<int64_t> edges = compute_edge_indexs(b.parents);
  const int64_t E = static_cast<int64_t>(edges.size() / 2);

  cond["crop_start_ind"] = i64({0}, {1});
  cond["motion_length"] = i64({num_frames}, {1});
  cond["lengths_mask"] = u8(lmask, {1, 1, 1, F});
  cond["n_joints"] = i64({n}, {1});
  cond["joint_mask"] = u8(jmask, {1, 1, 1, J});
  cond["tpos_first_frame"] = f32(tff, {1, J, D});
  cond["tpos_first_frame_parents"] = f32(tffp, {1, J, D});
  cond["parents"] = i64(parents64, {1, n});
  cond["edge_indexs"] = i64(edges, {1, 2, E});
  cond["joint_relations"] = i64(rel, {1, J, J});
  cond["graph_dist"] = i64(gd, {1, J, J});
  cond["joint_depths"] = i64(depth, {1, J});
  cond["mean"] = f32(mean, {1, J, D});
  cond["std"] = f32(stdv, {1, J, D});
  cond["spectral_feats"] = f32(spec, {1, J, K});
  cond["joint_names_emb"] = f32(jemb, {1, J, T});
  cond["offsets"] = f32(offsets, {1, J, 3});
  cond["caption_emb"] = f32(cap, {1, T});
  return cond;
}

}  // namespace um::pipeline

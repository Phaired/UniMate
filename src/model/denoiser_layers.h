// Internal building blocks of um::Denoiser, one struct per upstream module of
// reference/UniMate/unimate/models/denoiser/ (blocks/common.py, blocks/graph.py,
// rope.py). Weight names mirror the PyTorch state dict.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/tensor.h"
#include "core/umtf.h"

namespace um::denoiser {

// Additive mask value for masked attention keys (upstream: -inf / bool masks).
inline constexpr float kMaskNeg = -1e30f;

// Thin wrapper over the Backend that counts issued ops (reshape views are free
// and not counted).
struct Ctx {
  Backend& be;
  int ops = 0;

  Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b) { ++ops; return be.linear(x, w, b); }
  Tensor rms(const Tensor& x, const Tensor& w, float eps = 1e-6f) { ++ops; return be.rms_norm(x, w, eps); }
  Tensor modulate(const Tensor& x, const Tensor& shift, const Tensor& scale) { ++ops; return be.modulate(x, shift, scale); }
  Tensor permute(const Tensor& x, const std::vector<int>& p) { ++ops; return be.permute(x, p); }
  Tensor slice(const Tensor& x, int axis, int64_t a, int64_t b) { ++ops; return be.slice(x, axis, a, b); }
  Tensor concat(const std::vector<Tensor>& xs, int axis) { ++ops; return be.concat(xs, axis); }
  Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, float scale, const Tensor& bias) {
    ++ops; return be.attention(q, k, v, scale, bias);
  }
  Tensor rope(const Tensor& x, const Tensor& c, const Tensor& s) { ++ops; return be.rope(x, c, s); }
  Tensor add(const Tensor& a, const Tensor& b) { ++ops; return be.binary(BinaryOp::Add, a, b); }
  Tensor sub(const Tensor& a, const Tensor& b) { ++ops; return be.binary(BinaryOp::Sub, a, b); }
  Tensor mul(const Tensor& a, const Tensor& b) { ++ops; return be.binary(BinaryOp::Mul, a, b); }
  Tensor div(const Tensor& a, const Tensor& b) { ++ops; return be.binary(BinaryOp::Div, a, b); }
  Tensor affine(const Tensor& x, float s, float t) { ++ops; return be.affine(x, s, t); }
  Tensor unary(UnaryOp op, const Tensor& x) { ++ops; return be.unary(op, x); }
  Tensor silu(const Tensor& x) { return unary(UnaryOp::Silu, x); }
  Tensor mean(const Tensor& x, int axis, bool keep) { ++ops; return be.reduce(ReduceOp::Mean, x, axis, keep); }
  Tensor sum(const Tensor& x, int axis, bool keep) { ++ops; return be.reduce(ReduceOp::Sum, x, axis, keep); }
  Tensor gather(const Tensor& table, const std::vector<int32_t>& idx, const Shape& s) {
    ++ops; return be.gather_rows(table, idx, s);
  }
  Tensor broadcast_to(const Tensor& x, const Shape& s) { ++ops; return be.broadcast_to(x, s); }
  Tensor upload(const std::vector<float>& v, const Shape& s) { ++ops; return be.from_vector(v, s); }
};

// Weight access with host-side surgery helpers (row splits, bias edits).
struct Weights {
  WeightStore& ws;
  Backend& be;
  Tensor get(const std::string& n) { return ws.get(n); }
  bool has(const std::string& n) const { return ws.has(n); }
  std::vector<float> host(const std::string& n) const { return ws.host(n).to_f32(); }
  const Shape& shape(const std::string& n) const { return ws.host(n).shape; }
  // Rows [r0, r1) of a 2-d [N, K] weight or a 1-d [N] bias, uploaded.
  Tensor rows(const std::string& n, int64_t r0, int64_t r1);
};

struct Linear {
  Tensor w, b;
  static Linear load(Weights& W, const std::string& prefix, bool bias = true);
  Tensor operator()(Ctx& c, const Tensor& x) const { return c.linear(x, w, b); }
};

// nn.Sequential(Linear, SiLU, Linear)
struct Mlp2 {
  Linear l0, l2;
  static Mlp2 load(Weights& W, const std::string& prefix);
  Tensor operator()(Ctx& c, const Tensor& x) const { return l2(c, c.silu(l0(c, x))); }
};

// SwiGLUFFN with w12 split at load into w1 (x1 half) and w2 (x2 half).
struct SwiGLU {
  Linear w1, w2, w3;
  static SwiGLU load(Weights& W, const std::string& prefix);
  Tensor operator()(Ctx& c, const Tensor& x) const {
    return w3(c, c.mul(c.silu(w1(c, x)), w2(c, x)));
  }
};

// nn.MultiheadAttention(batch_first) with in_proj split into q / k / v.
struct Mha {
  Linear q, k, v, out;
  int heads = 0, dim = 0;
  static Mha load(Weights& W, const std::string& prefix, int heads);
};

// TimestepEmbedder (frequency embedding on the host, MLP on the device).
struct TimestepEmbedder {
  Mlp2 mlp;
  int freq_dim = 256;
  float max_period = 10000.f;
  static TimestepEmbedder load(Weights& W, const std::string& prefix, int freq_dim);
  std::vector<float> frequency_embedding(const std::vector<float>& t) const;  // [B, freq_dim]
  Tensor operator()(Ctx& c, const std::vector<float>& t) const;
};

// InputLayer: root / non-root split, tpos (+ parent) encoding, motion encoding.
struct InputLayer {
  Mlp2 root_tpos, root_x, joint_tpos, parent_tpos, tpos_fuse, joint_x;
  bool concat_parent = false;
  static InputLayer load(Weights& W, const std::string& prefix, bool concat_parent);
  // tpos / parents [B, J, F] -> tpos_emb [B, J, D]
  Tensor encode_tpos(Ctx& c, const Tensor& tpos, const Tensor& parents) const;
  // x [B, J, Dfeat, F] -> [B, F, J, D]
  Tensor encode_motion(Ctx& c, const Tensor& x) const;
};

// TposPool / TposCrossAttentionPool.
struct TposPool {
  int num_queries = 0;
  // cross-attention pool
  Tensor queries, norm_q, norm_kv, norm_ff, norm_out;
  Mha attn;
  SwiGLU ffn;
  Linear out_proj;
  // mean pool
  Linear proj;
  static TposPool load(Weights& W, const std::string& prefix, int num_queries, int heads);
  // tpos_emb [B, J, D], key_bias [B, 1, 1, J] (0 / kMaskNeg); mean-pool mode
  // only: valid01 [B, J, 1] (0/1), count [B, 1] (valid joints, >= 1).
  Tensor operator()(Ctx& c, const Tensor& tpos_emb, const Tensor& key_bias,
                    const Tensor& valid01, const Tensor& count) const;
};

// SpectralJointRoPE (SignNet or WIRE encoder) -> cos / sin [B, J, head_dim].
struct SpectralJointRoPE {
  bool signnet = true;
  int num_eigvecs = 8;
  Linear phi0, phi2, rho0, rho2;  // SignNet
  Tensor frequencies;             // WIRE [head_dim/2, K]
  static SpectralJointRoPE load(Weights& W, const std::string& prefix, bool signnet, int K);
  std::pair<Tensor, Tensor> cos_sin(Ctx& c, const Tensor& spectral) const;
};

// Graphormer distance + edge-type bias (GraphAttnBias / per-attention tables).
struct GraphAttnBias {
  Tensor dist_table, rel_table;  // [max_path_len+1, H], [edge_types, H] (proj * scale)
  int dist_rows = 0, rel_rows = 0, heads = 0;
  static GraphAttnBias load(Ctx& c, Weights& W, const std::string& prefix);
  // gd / rel host ints [B*J*J] -> [B, H, J, J] with key_bias [B,1,1,J] added.
  Tensor operator()(Ctx& c, const std::vector<int32_t>& gd, const std::vector<int32_t>& rel,
                    int B, int J, const Tensor& key_bias) const;
};

// Self-attention with qk RMSNorm + RoPE (both SpatialGraphSelfAttention and
// TemporalSelfAttention: they differ only in the token axis / bias / rope).
struct RopeSelfAttention {
  Linear qkv, proj;
  Tensor q_norm, k_norm;
  int heads = 0, head_dim = 0;
  static RopeSelfAttention load(Weights& W, const std::string& prefix, int heads, int head_dim);
  // Per-frame attention over joints. x [B, F, J, D]; cos / sin [B,1,1,J,hd];
  // bias [B, 1, H, J, J] (graph bias incl. key mask).
  Tensor spatial(Ctx& c, const Tensor& x, const Tensor& cos, const Tensor& sin,
                 const Tensor& bias) const;
  // Per-joint attention over frames. x [B, F, J, D]; cos / sin [F, hd];
  // bias [B, 1, 1, 1, F] or invalid (all frames valid).
  Tensor temporal(Ctx& c, const Tensor& x, const Tensor& cos, const Tensor& sin,
                  const Tensor& bias) const;
};

struct SpatioTemporalBlock {
  Tensor norm_s, norm_t, norm_mlp;
  RopeSelfAttention s_attn, t_attn;
  SwiGLU mlp;
  GraphAttnBias graph_bias;  // own tables (share_graph_attn_bias = false)
  static SpatioTemporalBlock load(Ctx& c, Weights& W, const std::string& prefix, int heads,
                                  int head_dim, bool own_graph_bias);
};

struct FinalLayer {
  Tensor norm_final, norm_q, norm_kv;
  Mha root_cross;
  Mlp2 root_out, joint_out;
  static FinalLayer load(Weights& W, const std::string& prefix, int heads);
  // h [B, F, J, D] (prefix already stripped), shift / scale [B,1,1,D],
  // kv_bias [B,1,1,1,J-1] -> [B, J, Dout, F]
  Tensor operator()(Ctx& c, const Tensor& h, const Tensor& shift, const Tensor& scale,
                    const Tensor& kv_bias) const;
};

}  // namespace um::denoiser

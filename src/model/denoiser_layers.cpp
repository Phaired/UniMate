#include "model/denoiser_layers.h"

#include <cmath>
#include <stdexcept>

namespace um::denoiser {

namespace {
[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("Denoiser: " + m); }
}  // namespace

// ---------------------------------------------------------------------------
// Weights / generic layers
// ---------------------------------------------------------------------------

Tensor Weights::rows(const std::string& n, int64_t r0, int64_t r1) {
  const HostArray& a = ws.host(n);
  std::vector<float> v = a.to_f32();
  int64_t row = a.shape.size() == 1 ? 1 : a.numel() / a.shape[0];
  if (r0 < 0 || r1 > a.shape[0] || r0 >= r1) fail("bad row split of " + n);
  std::vector<float> out(v.begin() + r0 * row, v.begin() + r1 * row);
  Shape s = a.shape;
  s[0] = r1 - r0;
  return be.from_vector(out, s);
}

Linear Linear::load(Weights& W, const std::string& p, bool bias) {
  Linear l;
  l.w = W.get(p + ".weight");
  if (bias) l.b = W.get(p + ".bias");
  return l;
}

Mlp2 Mlp2::load(Weights& W, const std::string& p) {
  return {Linear::load(W, p + ".0"), Linear::load(W, p + ".2")};
}

SwiGLU SwiGLU::load(Weights& W, const std::string& p) {
  const int64_t n2 = W.shape(p + ".w12.weight")[0];
  if (n2 % 2) fail(p + ".w12 has an odd output size");
  SwiGLU s;
  s.w1 = {W.rows(p + ".w12.weight", 0, n2 / 2), W.rows(p + ".w12.bias", 0, n2 / 2)};
  s.w2 = {W.rows(p + ".w12.weight", n2 / 2, n2), W.rows(p + ".w12.bias", n2 / 2, n2)};
  s.w3 = Linear::load(W, p + ".w3");
  return s;
}

Mha Mha::load(Weights& W, const std::string& p, int heads) {
  const int64_t e3 = W.shape(p + ".in_proj_weight")[0];
  const int64_t e = e3 / 3;
  Mha m;
  m.q = {W.rows(p + ".in_proj_weight", 0, e), W.rows(p + ".in_proj_bias", 0, e)};
  m.k = {W.rows(p + ".in_proj_weight", e, 2 * e), W.rows(p + ".in_proj_bias", e, 2 * e)};
  m.v = {W.rows(p + ".in_proj_weight", 2 * e, 3 * e), W.rows(p + ".in_proj_bias", 2 * e, 3 * e)};
  m.out = Linear::load(W, p + ".out_proj");
  m.heads = heads;
  m.dim = static_cast<int>(e);
  if (e % heads) fail(p + ": embed dim not divisible by heads");
  return m;
}

// ---------------------------------------------------------------------------
// TimestepEmbedder
// ---------------------------------------------------------------------------

TimestepEmbedder TimestepEmbedder::load(Weights& W, const std::string& p, int freq_dim) {
  TimestepEmbedder t;
  t.mlp = Mlp2::load(W, p + ".mlp");
  t.freq_dim = freq_dim;
  return t;
}

// torch: freqs = exp(-log(max_period) * arange(half, f32) / half) in float32;
// args = t[:, None] * freqs; cat([cos(args), sin(args)]).
std::vector<float> TimestepEmbedder::frequency_embedding(const std::vector<float>& t) const {
  const int half = freq_dim / 2;
  const float c = static_cast<float>(-std::log(static_cast<double>(max_period)));
  std::vector<float> freqs(half);
  for (int i = 0; i < half; ++i) freqs[i] = std::exp(c * static_cast<float>(i) / static_cast<float>(half));
  std::vector<float> out(t.size() * freq_dim, 0.f);
  for (size_t b = 0; b < t.size(); ++b) {
    for (int i = 0; i < half; ++i) {
      const float a = t[b] * freqs[i];
      out[b * freq_dim + i] = std::cos(a);
      out[b * freq_dim + half + i] = std::sin(a);
    }
  }
  return out;
}

Tensor TimestepEmbedder::operator()(Ctx& c, const std::vector<float>& t) const {
  Tensor f = c.upload(frequency_embedding(t), {static_cast<int64_t>(t.size()), freq_dim});
  return mlp(c, f);
}

// ---------------------------------------------------------------------------
// InputLayer
// ---------------------------------------------------------------------------

InputLayer InputLayer::load(Weights& W, const std::string& p, bool concat_parent) {
  InputLayer l;
  l.root_tpos = Mlp2::load(W, p + ".root_tpos_embedder");
  l.root_x = Mlp2::load(W, p + ".root_x_embedder");
  l.joint_tpos = Mlp2::load(W, p + ".joint_tpos_embedder");
  l.joint_x = Mlp2::load(W, p + ".joint_x_embedder");
  l.concat_parent = concat_parent;
  if (concat_parent) {
    l.parent_tpos = Mlp2::load(W, p + ".parent_tpos_embedder");
    l.tpos_fuse = Mlp2::load(W, p + ".tpos_fuse");
  }
  return l;
}

Tensor InputLayer::encode_tpos(Ctx& c, const Tensor& tpos, const Tensor& parents) const {
  const int64_t J = tpos.dim(1);
  Tensor root = root_tpos(c, c.slice(tpos, 1, 0, 1));          // [B, 1, D]
  Tensor joints = joint_tpos(c, c.slice(tpos, 1, 1, J));       // [B, J-1, D]
  if (concat_parent) {
    Tensor par = parent_tpos(c, c.slice(parents, 1, 1, J));
    joints = tpos_fuse(c, c.concat({joints, par}, -1));
  }
  return c.concat({root, joints}, 1);                            // [B, J, D]
}

Tensor InputLayer::encode_motion(Ctx& c, const Tensor& x) const {
  const int64_t J = x.dim(1);
  Tensor xf = c.permute(x, {0, 3, 1, 2});                        // [B, F, J, Dfeat]
  Tensor root = root_x(c, c.slice(xf, 2, 0, 1));
  Tensor joints = joint_x(c, c.slice(xf, 2, 1, J));
  return c.concat({root, joints}, 2);                            // [B, F, J, D]
}

// ---------------------------------------------------------------------------
// TposPool
// ---------------------------------------------------------------------------

TposPool TposPool::load(Weights& W, const std::string& p, int num_queries, int heads) {
  TposPool t;
  t.num_queries = num_queries;
  if (num_queries > 0) {
    const std::string q = p + ".pool";
    t.queries = W.get(q + ".queries");
    t.norm_q = W.get(q + ".norm_q.weight");
    t.norm_kv = W.get(q + ".norm_kv.weight");
    t.norm_ff = W.get(q + ".norm_ff.weight");
    t.norm_out = W.get(q + ".norm_out.weight");
    t.attn = Mha::load(W, q + ".cross_attn", heads);
    t.ffn = SwiGLU::load(W, q + ".ffn");
    t.out_proj = Linear::load(W, q + ".out_proj");
  } else {
    t.proj = Linear::load(W, p + ".proj");
  }
  return t;
}

Tensor TposPool::operator()(Ctx& c, const Tensor& tpos_emb, const Tensor& key_bias,
                            const Tensor& valid01, const Tensor& count) const {
  const int64_t B = tpos_emb.dim(0), J = tpos_emb.dim(1), D = tpos_emb.dim(2);
  if (num_queries == 0) {
    Tensor s = c.sum(c.mul(tpos_emb, valid01), 1, false);          // [B, D]
    return proj(c, c.div(s, count));
  }
  const int64_t K = queries.dim(1), H = attn.heads, hd = D / H;
  Tensor kv = c.rms(tpos_emb, norm_kv);                            // [B, J, D]
  Tensor q = attn.q(c, c.rms(queries, norm_q));                    // [1, K, D]
  q = c.broadcast_to(c.permute(q.reshape({1, K, H, hd}), {0, 2, 1, 3}), {B, H, K, hd});
  Tensor k = c.permute(attn.k(c, kv).reshape({B, J, H, hd}), {0, 2, 1, 3});
  Tensor v = c.permute(attn.v(c, kv).reshape({B, J, H, hd}), {0, 2, 1, 3});
  Tensor o = c.attention(q, k, v, 1.f / std::sqrt(static_cast<float>(hd)), key_bias);  // [B,H,K,hd]
  o = attn.out(c, c.permute(o, {0, 2, 1, 3}).reshape({B, K, D}));
  Tensor x = c.add(queries, o);                                    // [B, K, D]
  x = c.add(x, ffn(c, c.rms(x, norm_ff)));
  Tensor pooled = c.mean(x, 1, false);                             // [B, D]
  return out_proj(c, c.rms(pooled, norm_out));
}

// ---------------------------------------------------------------------------
// SpectralJointRoPE
// ---------------------------------------------------------------------------

SpectralJointRoPE SpectralJointRoPE::load(Weights& W, const std::string& p, bool signnet, int K) {
  SpectralJointRoPE r;
  r.signnet = signnet;
  r.num_eigvecs = K;
  const std::string e = p + ".spectral_encoder";
  if (signnet) {
    r.phi0 = Linear::load(W, e + ".phi.0");
    r.phi2 = Linear::load(W, e + ".phi.2");
    r.rho0 = Linear::load(W, e + ".rho.0");
    r.rho2 = Linear::load(W, e + ".rho.2");
  } else {
    r.frequencies = W.get(e + ".frequencies");
  }
  return r;
}

std::pair<Tensor, Tensor> SpectralJointRoPE::cos_sin(Ctx& c, const Tensor& sp) const {
  const int64_t B = sp.dim(0), J = sp.dim(1), K = sp.dim(2);
  if (K != num_eigvecs)
    fail("spectral_feats has K=" + std::to_string(K) + ", the model expects " +
         std::to_string(num_eigvecs));
  Tensor angles;
  if (signnet) {
    Tensor v = sp.reshape({B, J, K, 1});
    auto phi = [&](const Tensor& z) { return phi2(c, c.unary(UnaryOp::Gelu, phi0(c, z))); };
    Tensor h = c.add(phi(v), phi(c.unary(UnaryOp::Neg, v)));       // [B, J, K, hid]
    h = h.reshape({B, J, -1});
    angles = rho2(c, c.unary(UnaryOp::Gelu, rho0(c, h)));          // [B, J, hd/2]
  } else {
    angles = c.linear(sp, frequencies, Tensor());
  }
  Tensor full = c.concat({angles, angles}, -1);                    // [B, J, hd]
  return {c.unary(UnaryOp::Cos, full), c.unary(UnaryOp::Sin, full)};
}

// ---------------------------------------------------------------------------
// GraphAttnBias
// ---------------------------------------------------------------------------

GraphAttnBias GraphAttnBias::load(Ctx& c, Weights& W, const std::string& p) {
  GraphAttnBias g;
  // proj(embedding(i)) * scale for every possible index i: a [rows, H] table,
  // so the per-condition work is a row gather (identical per-row arithmetic).
  auto table = [&](const std::string& emb, const std::string& proj, const std::string& scale) {
    Tensor t = c.linear(W.get(p + emb), W.get(p + proj + ".weight"), W.get(p + proj + ".bias"));
    return c.mul(t, W.get(p + scale));
  };
  g.dist_table = table(".graph_dist_embedding.weight", ".graph_dist_proj", ".graph_dist_scale");
  g.rel_table = table(".graph_rel_embedding.weight", ".graph_rel_proj", ".graph_rel_scale");
  g.dist_rows = static_cast<int>(g.dist_table.dim(0));
  g.rel_rows = static_cast<int>(g.rel_table.dim(0));
  g.heads = static_cast<int>(g.dist_table.dim(1));
  return g;
}

Tensor GraphAttnBias::operator()(Ctx& c, const std::vector<int32_t>& gd,
                                 const std::vector<int32_t>& rel, int B, int J,
                                 const Tensor& key_bias) const {
  for (int32_t v : gd)
    if (v < 0 || v >= dist_rows) fail("graph_dist value " + std::to_string(v) + " out of [0, " + std::to_string(dist_rows) + ")");
  for (int32_t v : rel)
    if (v < 0 || v >= rel_rows) fail("joint_relations value " + std::to_string(v) + " out of [0, " + std::to_string(rel_rows) + ")");
  Tensor d = c.gather(dist_table, gd, {B, J, J});                  // [B, J, J, H]
  Tensor r = c.gather(rel_table, rel, {B, J, J});
  Tensor b = c.permute(c.add(d, r), {0, 3, 1, 2});                 // [B, H, J, J]
  return c.add(b, key_bias);                                       // keys masked
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

RopeSelfAttention RopeSelfAttention::load(Weights& W, const std::string& p, int heads, int head_dim) {
  RopeSelfAttention a;
  a.qkv = Linear::load(W, p + ".qkv");
  a.proj = Linear::load(W, p + ".proj");
  a.q_norm = W.get(p + ".q_norm.weight");
  a.k_norm = W.get(p + ".k_norm.weight");
  a.heads = heads;
  a.head_dim = head_dim;
  return a;
}

Tensor RopeSelfAttention::spatial(Ctx& c, const Tensor& x, const Tensor& cos, const Tensor& sin,
                                  const Tensor& bias) const {
  const int64_t B = x.dim(0), F = x.dim(1), J = x.dim(2), D = x.dim(3);
  const int64_t H = heads, hd = head_dim;
  Tensor t = qkv(c, x).reshape({B, F, J, 3, H, hd});
  t = c.permute(t, {3, 0, 1, 4, 2, 5});                            // [3, B, F, H, J, hd]
  Tensor q = c.slice(t, 0, 0, 1).reshape({B, F, H, J, hd});
  Tensor k = c.slice(t, 0, 1, 2).reshape({B, F, H, J, hd});
  Tensor v = c.slice(t, 0, 2, 3).reshape({B, F, H, J, hd});
  q = c.rope(c.rms(q, q_norm), cos, sin);
  k = c.rope(c.rms(k, k_norm), cos, sin);
  Tensor o = c.attention(q, k, v, 1.f / std::sqrt(static_cast<float>(hd)), bias);  // [B,F,H,J,hd]
  o = c.permute(o, {0, 1, 3, 2, 4}).reshape({B, F, J, D});
  return proj(c, o);
}

Tensor RopeSelfAttention::temporal(Ctx& c, const Tensor& x, const Tensor& cos, const Tensor& sin,
                                   const Tensor& bias) const {
  const int64_t B = x.dim(0), F = x.dim(1), J = x.dim(2), D = x.dim(3);
  const int64_t H = heads, hd = head_dim;
  Tensor t = qkv(c, x).reshape({B, F, J, 3, H, hd});
  t = c.permute(t, {3, 0, 2, 4, 1, 5});                            // [3, B, J, H, F, hd]
  Tensor q = c.slice(t, 0, 0, 1).reshape({B, J, H, F, hd});
  Tensor k = c.slice(t, 0, 1, 2).reshape({B, J, H, F, hd});
  Tensor v = c.slice(t, 0, 2, 3).reshape({B, J, H, F, hd});
  q = c.rope(c.rms(q, q_norm), cos, sin);
  k = c.rope(c.rms(k, k_norm), cos, sin);
  Tensor o = c.attention(q, k, v, 1.f / std::sqrt(static_cast<float>(hd)), bias);  // [B,J,H,F,hd]
  o = c.permute(o, {0, 3, 1, 2, 4}).reshape({B, F, J, D});
  return proj(c, o);
}

SpatioTemporalBlock SpatioTemporalBlock::load(Ctx& c, Weights& W, const std::string& p, int heads,
                                              int head_dim, bool own_graph_bias) {
  SpatioTemporalBlock b;
  b.norm_s = W.get(p + ".norm_s.weight");
  b.norm_t = W.get(p + ".norm_t.weight");
  b.norm_mlp = W.get(p + ".norm_mlp.weight");
  b.s_attn = RopeSelfAttention::load(W, p + ".s_attn", heads, head_dim);
  b.t_attn = RopeSelfAttention::load(W, p + ".t_attn", heads, head_dim);
  b.mlp = SwiGLU::load(W, p + ".mlp");
  if (own_graph_bias) b.graph_bias = GraphAttnBias::load(c, W, p + ".s_attn");
  return b;
}

// ---------------------------------------------------------------------------
// FinalLayer
// ---------------------------------------------------------------------------

FinalLayer FinalLayer::load(Weights& W, const std::string& p, int heads) {
  FinalLayer f;
  f.norm_final = W.get(p + ".norm_final.weight");
  f.norm_q = W.get(p + ".root_cross_norm_q.weight");
  f.norm_kv = W.get(p + ".root_cross_norm_kv.weight");
  f.root_cross = Mha::load(W, p + ".root_cross_attn", heads);
  f.root_out = Mlp2::load(W, p + ".root_out");
  f.joint_out = Mlp2::load(W, p + ".joint_out");
  return f;
}

Tensor FinalLayer::operator()(Ctx& c, const Tensor& h, const Tensor& shift, const Tensor& scale,
                              const Tensor& kv_bias) const {
  const int64_t B = h.dim(0), F = h.dim(1), J = h.dim(2), D = h.dim(3);
  const int64_t H = root_cross.heads, hd = D / H;
  Tensor x = c.modulate(c.rms(h, norm_final), shift, scale);       // [B, F, J, D]
  Tensor root = c.slice(x, 2, 0, 1);                               // [B, F, 1, D]
  Tensor joints = c.slice(x, 2, 1, J);                             // [B, F, J-1, D]
  // Per-frame cross-attention: root (query) -> joints (keys: normed, values: raw).
  Tensor q = root_cross.q(c, c.rms(root, norm_q)).reshape({B, F, H, 1, hd});
  Tensor kn = c.rms(joints, norm_kv);
  Tensor k = c.permute(root_cross.k(c, kn).reshape({B, F, J - 1, H, hd}), {0, 1, 3, 2, 4});
  Tensor v = c.permute(root_cross.v(c, joints).reshape({B, F, J - 1, H, hd}), {0, 1, 3, 2, 4});
  Tensor o = c.attention(q, k, v, 1.f / std::sqrt(static_cast<float>(hd)), kv_bias);  // [B,F,H,1,hd]
  o = root_cross.out(c, o.reshape({B, F, 1, D}));
  root = c.add(root, o);
  Tensor out = c.concat({root_out(c, root), joint_out(c, joints)}, 2);  // [B, F, J, Dout]
  return c.permute(out, {0, 2, 3, 1});                                   // [B, J, Dout, F]
}

}  // namespace um::denoiser

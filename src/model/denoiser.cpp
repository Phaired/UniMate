// um::Denoiser: UniMateGraphAdaLN forward (reference/UniMate/unimate/models/
// denoiser/graph_adaln.py) on the abstract Backend. See denoiser.h.
#include "model/denoiser.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "model/denoiser_layers.h"

namespace um {

using namespace denoiser;

namespace {

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("Denoiser: " + m); }

template <class T>
T jget(const nlohmann::json& j, const char* key, const T& def) {
  return j.contains(key) && !j[key].is_null() ? j[key].get<T>() : def;
}

}  // namespace

namespace denoiser {

// Conditioning tensors for one batch layout (B for forward, 2B for CFG).
struct Static {
  int B = 0;
  Tensor y_cond, y_uncond;      // [B, D]  caption projection + tpos pool (x / t independent)
  Tensor prefix;                // [B, 1, J, D]  tpos frame tokens (+ token embeddings)
  Tensor tok;                   // [B, 1, J, D]  depth + joint-name embeddings (or invalid)
  Tensor rope_cos, rope_sin;    // [B, 1, 1, J, hd]
  std::vector<Tensor> graph_bias;  // per block [B, 1, H, J, J]
  Tensor final_kv_bias;         // [B, 1, 1, 1, J-1]
  std::vector<int> motion_length;
  std::map<int64_t, Tensor> temporal_bias;  // by F+1; invalid = all frames valid
};

}  // namespace denoiser

struct Denoiser::Impl {
  Backend& be;
  std::shared_ptr<UmtfFile> file;
  WeightStore ws;
  nlohmann::json meta, cfg;
  Ctx ctx;

  // config
  int D = 0, H = 0, hd = 0, L = 0, Jmax = 0, Fmax = 0, feat = 0, text_dim = 0;
  int n_prefix = 1, max_depth = 0, K = 0;
  bool text = true, depth_emb = false, name_emb = false, concat_parent = false;
  bool inject_tpos = false, graph_bias_on = true, share_graph_bias = false;
  int tpos_queries = 0;

  // modules
  Linear cond_embedder, joint_name_embedder;
  Tensor depth_table;
  TimestepEmbedder time_embedder;
  InputLayer input_layer;
  TposPool tpos_pool;
  SpectralJointRoPE rope_j;
  GraphAttnBias shared_bias;
  std::vector<SpatioTemporalBlock> blocks;
  FinalLayer final_layer;
  Linear adaln_all;  // all blocks' + final layer's adaLN_modulation.1, gate biases - 1
  std::vector<float> rope_t_cos, rope_t_sin;  // [max_frames+1, hd]
  std::map<int64_t, std::pair<Tensor, Tensor>> rope_t_cache;

  Static s1, s2;
  bool has_cond = false;

  Impl(Backend& b, const std::string& path)
      : be(b),
        file(std::make_shared<UmtfFile>(UmtfFile::load(path))),
        ws(b, file),
        ctx{b} {
    meta = file->meta();
    if (!meta.contains("config")) fail(path + ": no meta.config (not a denoiser_export.py file)");
    cfg = meta["config"];
    load();
  }

  void load() {
    const auto& m = cfg.at("model");
    const auto& d = cfg.at("dataset");
    const auto& der = meta.value("derived", nlohmann::json::object());
    const std::string attention = jget<std::string>(m, "attention", "");
    const std::string text_cond = jget<std::string>(m, "text_cond", "");
    if (attention != "graph")
      fail("unsupported model.attention='" + attention + "' (only 'graph' is ported)");
    if (text_cond != "adaln")
      fail("unsupported model.text_cond='" + text_cond + "' (only 'adaln' is ported)");
    const std::string cond_mode = jget<std::string>(m, "cond_mode", "text");
    if (cond_mode != "text") fail("unsupported model.cond_mode='" + cond_mode + "'");
    if (jget<bool>(m, "use_graph_emb", false)) fail("model.use_graph_emb (GCNConv) is not supported");
    if (!jget<bool>(m, "use_spectral_rope", false))
      fail("model.use_spectral_rope=false (integer joint RopeND) is not supported");

    D = jget<int>(m, "latent_dim", 0);
    H = jget<int>(m, "num_heads", 0);
    L = jget<int>(m, "num_layers", 0);
    hd = D / H;
    Jmax = jget<int>(d, "max_joints", 0);
    Fmax = jget<int>(d, "max_motion_length", 0);
    feat = jget<int>(d, "feature_len", 0);
    max_depth = jget<int>(d, "max_depth", 0);
    if (max_depth <= 0) max_depth = Jmax;
    K = jget<int>(m, "max_freqs", 8);
    text_dim = jget<int>(der, "text_dim", 768);
    n_prefix = jget<int>(der, "n_prefix", 1);
    depth_emb = jget<bool>(m, "use_depth_emb", false);
    name_emb = jget<bool>(m, "use_joint_name_emb", false);
    concat_parent = jget<bool>(m, "concat_parent_features", false);
    inject_tpos = jget<bool>(m, "inject_tpos_to_adaln", false);
    tpos_queries = jget<int>(m, "num_tpos_queries", 0);
    graph_bias_on = jget<bool>(m, "use_graph_attn_bias", true);
    share_graph_bias = jget<bool>(m, "share_graph_attn_bias", false);
    if (D <= 0 || H <= 0 || D % H || L <= 0 || Jmax < 2 || Fmax <= 0 || feat <= 0)
      fail("bad dimensions in config");
    if (n_prefix != 1) fail("n_prefix != 1 is not supported");

    Weights W{ws, be};
    cond_embedder = Linear::load(W, "cond_embedder");
    time_embedder = TimestepEmbedder::load(W, "time_embedder",
                                           jget<int>(der, "timestep_frequency_embedding_size", 256));
    input_layer = InputLayer::load(W, "input_layer", concat_parent);
    if (inject_tpos)
      tpos_pool = TposPool::load(W, "tpos_pool", tpos_queries, jget<int>(der, "tpos_pool_heads", 4));
    if (name_emb) joint_name_embedder = Linear::load(W, "joint_name_embedder");
    if (depth_emb) {
      depth_table = W.get("depth_embedding.weight");
      max_depth = static_cast<int>(depth_table.dim(0)) - 1;
    }
    rope_j = SpectralJointRoPE::load(W, "rope_j", jget<bool>(m, "use_signnet", true), K);
    const bool own = graph_bias_on && !share_graph_bias;
    if (graph_bias_on && share_graph_bias) shared_bias = GraphAttnBias::load(ctx, W, "graph_bias");
    for (int l = 0; l < L; ++l)
      blocks.push_back(SpatioTemporalBlock::load(ctx, W, "transformer_blocks." + std::to_string(l),
                                                 H, hd, own));
    final_layer = FinalLayer::load(W, "final_layer", jget<int>(der, "final_root_attn_heads", 4));

    // One linear for every adaLN modulation: rows [l*9D, (l+1)*9D) = block l
    // (shift_s, scale_s, gate_s, shift_t, scale_t, gate_t, shift_m, scale_m,
    // gate_m), then the final layer's (shift, scale). Gate biases get -1 so a
    // residual "x + gate * o" is one modulate(o, x, gate - 1).
    {
      std::vector<float> w, b;
      w.reserve(static_cast<size_t>(L * 9 + 2) * D * D);
      for (int l = 0; l < L; ++l) {
        const std::string p = "transformer_blocks." + std::to_string(l) + ".adaLN_modulation.1";
        auto wl = W.host(p + ".weight");
        auto bl = W.host(p + ".bias");
        if (static_cast<int64_t>(bl.size()) != 9LL * D) fail(p + ": expected 9*latent_dim outputs");
        for (int g : {2, 5, 8})
          for (int i = 0; i < D; ++i) bl[g * D + i] -= 1.f;
        w.insert(w.end(), wl.begin(), wl.end());
        b.insert(b.end(), bl.begin(), bl.end());
      }
      auto wf = W.host("final_layer.adaLN_modulation.1.weight");
      auto bf = W.host("final_layer.adaLN_modulation.1.bias");
      w.insert(w.end(), wf.begin(), wf.end());
      b.insert(b.end(), bf.begin(), bf.end());
      const int64_t rows = static_cast<int64_t>(b.size());
      adaln_all.w = be.from_vector(w, {rows, D});
      adaln_all.b = be.from_vector(b, {rows});
    }

    if (ws.has("derived.rope_t.cos")) {
      rope_t_cos = W.host("derived.rope_t.cos");
      rope_t_sin = W.host("derived.rope_t.sin");
      if (W.shape("derived.rope_t.cos")[1] != hd) fail("derived.rope_t has the wrong head_dim");
    } else {
      // RopeND(nd=1, max_lens=[F+1], auto_base): base = (int(8L/pi)//100 + 1)*100.
      const int Lr = Fmax + n_prefix;
      const double base = (static_cast<int>(8.0 * Lr / 3.14159265358979323846) / 100 + 1) * 100;
      rope_t_cos.resize(static_cast<size_t>(Lr) * hd);
      rope_t_sin.resize(rope_t_cos.size());
      for (int p = 0; p < Lr; ++p)
        for (int i = 0; i < hd / 2; ++i) {
          const float inv = 1.f / static_cast<float>(std::pow(base, static_cast<float>(2 * i) / hd));
          const float a = static_cast<float>(p) * inv;
          for (int h2 : {i, i + hd / 2}) {
            rope_t_cos[p * hd + h2] = std::cos(a);
            rope_t_sin[p * hd + h2] = std::sin(a);
          }
        }
    }
  }

  // ---- conditioning --------------------------------------------------------

  static const HostArray& need(const ArrayDict& c, const std::string& k) {
    auto it = c.find(k);
    if (it == c.end()) fail("cond is missing key '" + k + "'");
    return it->second;
  }

  void check_shape(const HostArray& a, const std::string& k, const Shape& want) const {
    if (a.shape != want)
      fail("cond['" + k + "'] has shape " + shape_str(a.shape) + ", expected " + shape_str(want));
  }

  void set_condition(const ArrayDict& cond) {
    ctx.ops = 0;
    const HostArray& nj_a = need(cond, "n_joints");
    if (nj_a.shape.size() != 1) fail("cond['n_joints'] must be [B]");
    const int B = static_cast<int>(nj_a.shape[0]);
    const int J = Jmax;
    std::vector<int32_t> nj = nj_a.to_i32();
    for (int v : nj)
      if (v < 2 || v > J) fail("n_joints " + std::to_string(v) + " out of [2, " + std::to_string(J) + "]");
    const HostArray& ml_a = need(cond, "motion_length");
    check_shape(ml_a, "motion_length", {B});

    const HostArray& tpos_a = need(cond, "tpos_first_frame");
    check_shape(tpos_a, "tpos_first_frame", {B, J, feat});
    Tensor tpos = tpos_a.to_tensor(be);
    Tensor parents;
    if (concat_parent) {
      const HostArray& p = need(cond, "tpos_first_frame_parents");
      check_shape(p, "tpos_first_frame_parents", {B, J, feat});
      parents = p.to_tensor(be);
    }

    Static s;
    s.B = B;
    s.motion_length = ml_a.to_i32();

    // Masks: joint validity -> additive key biases.
    std::vector<float> key(static_cast<size_t>(B) * J), valid(key.size()), cnt(B);
    std::vector<float> kv(static_cast<size_t>(B) * (J - 1));
    for (int b = 0; b < B; ++b) {
      for (int j = 0; j < J; ++j) {
        key[b * J + j] = j < nj[b] ? 0.f : kMaskNeg;
        valid[b * J + j] = j < nj[b] ? 1.f : 0.f;
        if (j > 0) kv[b * (J - 1) + j - 1] = key[b * J + j];
      }
      cnt[b] = static_cast<float>(std::max(nj[b], 1));
    }
    Tensor key_bias = ctx.upload(key, {B, 1, 1, J});
    s.final_kv_bias = ctx.upload(kv, {B, 1, 1, 1, J - 1});

    // Caption projection (cond) and its CFG-masked twin (zeros -> bias only).
    Tensor cap_c, cap_u;
    {
      Tensor zeros = be.zeros({B, text_dim});
      cap_u = cond_embedder(ctx, zeros);
      auto it = cond.find("caption_emb");
      if (it == cond.end()) {
        cap_c = cap_u;  // upstream: missing key -> unconditional
      } else {
        check_shape(it->second, "caption_emb", {B, text_dim});
        cap_c = cond_embedder(ctx, it->second.to_tensor(be));
      }
    }

    // InputLayer tpos encoding + TposPool.
    Tensor tpos_emb = input_layer.encode_tpos(ctx, tpos, parents);   // [B, J, D]
    trace_static_["tpos_emb"] = tpos_emb;
    trace_static_["caption_proj"] = cap_c;
    if (inject_tpos) {
      Tensor tp = tpos_pool(ctx, tpos_emb, key_bias, ctx.upload(valid, {B, J, 1}),
                            ctx.upload(cnt, {B, 1}));
      trace_static_["tpos_pool"] = tp;
      s.y_cond = ctx.add(cap_c, tp);
      s.y_uncond = ctx.add(cap_u, tp);
    } else {
      s.y_cond = cap_c;
      s.y_uncond = cap_u;
    }

    // Token embeddings (depth, joint names), shared by every frame.
    Tensor prefix = tpos_emb, tok;
    if (depth_emb) {
      const HostArray& a = need(cond, "joint_depths");
      check_shape(a, "joint_depths", {B, J});
      std::vector<int32_t> dep = a.to_i32();
      for (auto& v : dep) {
        if (v < 0) fail("negative joint depth");
        v = std::min(v, max_depth);
      }
      Tensor de = ctx.gather(depth_table, dep, {B, J});              // [B, J, D]
      prefix = ctx.add(prefix, de);
      tok = de;
    }
    if (name_emb) {
      const HostArray& a = need(cond, "joint_names_emb");
      check_shape(a, "joint_names_emb", {B, J, text_dim});
      Tensor ne = joint_name_embedder(ctx, a.to_tensor(be));         // [B, J, D]
      prefix = ctx.add(prefix, ne);
      tok = tok.valid() ? ctx.add(tok, ne) : ne;
    }
    s.prefix = prefix.reshape({B, 1, J, D});
    if (tok.valid()) s.tok = tok.reshape({B, 1, J, D});

    // Spectral RoPE on the joint axis.
    {
      const HostArray& a = need(cond, "spectral_feats");
      if (a.shape.size() != 3 || a.shape[0] != B || a.shape[1] != J)
        fail("cond['spectral_feats'] has shape " + shape_str(a.shape) + ", expected [B, J, K]");
      auto [cs, sn] = rope_j.cos_sin(ctx, a.to_tensor(be));
      trace_static_["rope_j.cos"] = cs;
      trace_static_["rope_j.sin"] = sn;
      s.rope_cos = cs.reshape({B, 1, 1, J, hd});
      s.rope_sin = sn.reshape({B, 1, 1, J, hd});
    }

    // Graph attention bias per block (keys masked).
    if (graph_bias_on) {
      const HostArray& gd_a = need(cond, "graph_dist");
      const HostArray& rel_a = need(cond, "joint_relations");
      check_shape(gd_a, "graph_dist", {B, J, J});
      check_shape(rel_a, "joint_relations", {B, J, J});
      std::vector<int32_t> gd = gd_a.to_i32(), rel = rel_a.to_i32();
      Tensor shared;
      if (share_graph_bias) shared = shared_bias(ctx, gd, rel, B, J, key_bias);
      for (int l = 0; l < L; ++l) {
        Tensor g = share_graph_bias ? shared : blocks[l].graph_bias(ctx, gd, rel, B, J, key_bias);
        trace_static_["graph_bias." + std::to_string(l)] = g;
        s.graph_bias.push_back(g.reshape({B, 1, H, J, J}));
      }
    } else {
      for (int l = 0; l < L; ++l) s.graph_bias.push_back(key_bias.reshape({B, 1, 1, 1, J}));
    }

    // CFG layout: [cond batch; uncond batch] along the batch axis.
    Static d;
    d.B = 2 * B;
    auto dup = [&](const Tensor& t) { return t.valid() ? ctx.concat({t, t}, 0) : t; };
    d.y_cond = ctx.concat({s.y_cond, s.y_uncond}, 0);
    d.prefix = dup(s.prefix);
    d.tok = dup(s.tok);
    d.rope_cos = dup(s.rope_cos);
    d.rope_sin = dup(s.rope_sin);
    for (auto& g : s.graph_bias) d.graph_bias.push_back(dup(g));
    d.final_kv_bias = dup(s.final_kv_bias);
    d.motion_length = s.motion_length;
    d.motion_length.insert(d.motion_length.end(), s.motion_length.begin(), s.motion_length.end());

    s1 = std::move(s);
    s2 = std::move(d);
    has_cond = true;
    be.flush();
  }

  // ---- forward -------------------------------------------------------------

  std::map<std::string, Tensor> trace_static_;

  std::pair<Tensor, Tensor> rope_t(int64_t Fp) {
    auto it = rope_t_cache.find(Fp);
    if (it != rope_t_cache.end()) return it->second;
    std::vector<float> c(rope_t_cos.begin(), rope_t_cos.begin() + Fp * hd);
    std::vector<float> s(rope_t_sin.begin(), rope_t_sin.begin() + Fp * hd);
    auto r = std::make_pair(be.from_vector(c, {Fp, hd}), be.from_vector(s, {Fp, hd}));
    rope_t_cache[Fp] = r;
    return r;
  }

  Tensor temporal_bias(Static& s, int64_t Fp) {
    auto it = s.temporal_bias.find(Fp);
    if (it != s.temporal_bias.end()) return it->second;
    bool all = true;
    std::vector<float> v(static_cast<size_t>(s.B) * Fp);
    for (int b = 0; b < s.B; ++b)
      for (int64_t f = 0; f < Fp; ++f) {
        const bool ok = f < s.motion_length[b] + n_prefix;
        v[b * Fp + f] = ok ? 0.f : kMaskNeg;
        all = all && ok;
      }
    Tensor t = all ? Tensor() : be.from_vector(v, {s.B, 1, 1, 1, Fp});
    s.temporal_bias[Fp] = t;
    return t;
  }

  Tensor run(Static& s, const Tensor& x, float t, const Tensor& y_static, DenoiserTrace* tr) {
    if (!has_cond) fail("forward() before set_condition()");
    const int B = s.B;
    if (x.ndim() != 4 || x.dim(0) != B || x.dim(1) != Jmax || x.dim(2) != feat)
      fail("x has shape " + shape_str(x.shape()) + ", expected [" + std::to_string(B) + ", " +
           std::to_string(Jmax) + ", " + std::to_string(feat) + ", F]");
    const int64_t F = x.dim(3), Fp = F + n_prefix, J = Jmax;
    if (F < 1 || F > Fmax) fail("frame count " + std::to_string(F) + " out of [1, " + std::to_string(Fmax) + "]");
    auto rec = [&](const char* name, const Tensor& v) { if (tr) (*tr)[name] = v; };

    // adaLN driver y = t_emb + caption_proj (+ tpos_pool) and every modulation.
    Tensor temb = time_embedder(ctx, std::vector<float>(B, t));
    Tensor y = ctx.add(temb, y_static);
    rec("t_emb", temb);
    rec("y", y);
    const int64_t R = 9LL * L + 2;
    Tensor mods = adaln_all(ctx, ctx.silu(y)).reshape({B, R, D});
    auto chunk = [&](int64_t i) { return ctx.slice(mods, 1, i, i + 1).reshape({B, 1, 1, D}); };

    // Input layer: motion tokens (+ token embeddings), tpos frame prepended.
    Tensor h = input_layer.encode_motion(ctx, x);                  // [B, F, J, D]
    if (tr) {
      Tensor tpe = (s.B == s1.B ? trace_static_["tpos_emb"] : Tensor());
      if (tpe.valid()) (*tr)["input_layer"] = be.concat({tpe.reshape({B, 1, J, D}), h}, 1);
    }
    if (s.tok.valid()) h = ctx.add(h, s.tok);
    h = ctx.concat({s.prefix, h}, 1);                              // [B, F+1, J, D]
    rec("tokens", h);

    auto [tcos, tsin] = rope_t(Fp);
    Tensor tbias = temporal_bias(s, Fp);

    for (int l = 0; l < L; ++l) {
      const SpatioTemporalBlock& blk = blocks[l];
      const int64_t c0 = 9LL * l;
      Tensor xs = ctx.modulate(ctx.rms(h, blk.norm_s), chunk(c0 + 0), chunk(c0 + 1));
      xs = blk.s_attn.spatial(ctx, xs, s.rope_cos, s.rope_sin, s.graph_bias[l]);
      h = ctx.modulate(xs, h, chunk(c0 + 2));                      // h + gate_s * xs
      if (l == 0) rec("block0.spatial", h);
      Tensor xt = ctx.modulate(ctx.rms(h, blk.norm_t), chunk(c0 + 3), chunk(c0 + 4));
      xt = blk.t_attn.temporal(ctx, xt, tcos, tsin, tbias);
      h = ctx.modulate(xt, h, chunk(c0 + 5));                      // h + gate_t * xt
      if (l == 0) rec("block0.temporal", h);
      Tensor xm = ctx.modulate(ctx.rms(h, blk.norm_mlp), chunk(c0 + 6), chunk(c0 + 7));
      xm = blk.mlp(ctx, xm);
      h = ctx.modulate(xm, h, chunk(c0 + 8));                      // h + gate_m * xm
      if (l == 0) rec("block0.ffn", h);
    }
    rec("last_block", h);

    // Final layer is per frame: drop the prefix frame first (upstream strips it
    // after the final layer; identical values, less work).
    Tensor hf = ctx.slice(h, 1, n_prefix, Fp);
    Tensor out = final_layer(ctx, hf, chunk(9LL * L), chunk(9LL * L + 1), s.final_kv_bias);
    rec("final", out);
    return out;
  }
};

// ---------------------------------------------------------------------------

Denoiser::Denoiser(Backend& be, const std::string& umtf_path)
    : impl_(std::make_unique<Impl>(be, umtf_path)) {}
Denoiser::~Denoiser() = default;

const nlohmann::json& Denoiser::config() const { return impl_->cfg; }
int Denoiser::max_joints() const { return impl_->Jmax; }
int Denoiser::max_frames() const { return impl_->Fmax; }
int Denoiser::feature_len() const { return impl_->feat; }
int Denoiser::batch() const { return impl_->has_cond ? impl_->s1.B : 0; }
int Denoiser::last_op_count() const { return impl_->ctx.ops; }

void Denoiser::set_condition(const ArrayDict& cond) {
  impl_->trace_static_.clear();
  impl_->set_condition(cond);
}

Tensor Denoiser::forward(const Tensor& x, float t, bool force_mask) {
  return forward_traced(x, t, force_mask, nullptr);
}

Tensor Denoiser::forward_traced(const Tensor& x, float t, bool force_mask, DenoiserTrace* trace) {
  Impl& m = *impl_;
  m.ctx.ops = 0;
  if (trace)
    for (auto& [k, v] : m.trace_static_) (*trace)[k] = v;
  return m.run(m.s1, x, t, force_mask ? m.s1.y_uncond : m.s1.y_cond, trace);
}

Tensor Denoiser::velocity(const Tensor& x, float t, float cfg_scale) {
  Impl& m = *impl_;
  if (cfg_scale <= 1.f) return forward(x, t, false);
  m.ctx.ops = 0;
  const int64_t B = m.s1.B;
  Tensor x2 = m.ctx.concat({x, x}, 0);
  Tensor out = m.run(m.s2, x2, t, m.s2.y_cond, nullptr);
  Tensor oc = m.ctx.slice(out, 0, 0, B);
  Tensor ou = m.ctx.slice(out, 0, B, 2 * B);
  // ClassifierFreeSampleModel: out_uncond + cfg * (out_cond - out_uncond)
  return m.ctx.add(ou, m.ctx.affine(m.ctx.sub(oc, ou), cfg_scale, 0.f));
}

}  // namespace um

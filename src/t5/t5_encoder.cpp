#include "t5/t5_encoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace um {
namespace {

std::string layer_name(int i, const char* rest) {
  return "encoder.block." + std::to_string(i) + "." + rest;
}

}  // namespace

T5TextEncoder::T5TextEncoder(Backend& be, const std::string& umtf_path)
    : be_(&be) {
  // The host copy of the file only lives during construction: weights are
  // uploaded to the backend, tokenizer tables copied out.
  auto file = std::make_shared<const UmtfFile>(UmtfFile::load(umtf_path));
  if (!file->meta().contains("config") || !file->meta().contains("tokenizer"))
    throw std::runtime_error("T5TextEncoder: " + umtf_path + " is not a converted T5 model");
  WeightStore ws(be, file);
  tok_ = T5Tokenizer(*file);
  const auto& c = file->meta().at("config");
  cfg_.d_model = c.at("d_model").get<int>();
  cfg_.d_kv = c.at("d_kv").get<int>();
  cfg_.d_ff = c.at("d_ff").get<int>();
  cfg_.num_layers = c.at("num_layers").get<int>();
  cfg_.num_heads = c.at("num_heads").get<int>();
  cfg_.num_buckets = c.at("relative_attention_num_buckets").get<int>();
  cfg_.max_distance = c.at("relative_attention_max_distance").get<int>();
  cfg_.eps = c.at("layer_norm_epsilon").get<float>();
  if (c.value("dense_act_fn", std::string("gelu_new")) != "gelu_new" || !c.value("is_gated_act", true))
    throw std::runtime_error("T5TextEncoder: only gated-gelu (gelu_new) T5 variants are supported");

  rel_bias_ = ws.host("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight").to_f32();
  if (rel_bias_.size() != static_cast<size_t>(cfg_.num_buckets) * cfg_.num_heads)
    throw std::runtime_error("T5TextEncoder: bad relative_attention_bias shape");

  embed_ = ws.get("shared.weight");
  final_ln_ = ws.get("encoder.final_layer_norm.weight");
  layers_.resize(cfg_.num_layers);
  for (int i = 0; i < cfg_.num_layers; ++i) {
    Layer& L = layers_[i];
    L.ln0 = ws.get(layer_name(i, "layer.0.layer_norm.weight"));
    L.q = ws.get(layer_name(i, "layer.0.SelfAttention.q.weight"));
    L.k = ws.get(layer_name(i, "layer.0.SelfAttention.k.weight"));
    L.v = ws.get(layer_name(i, "layer.0.SelfAttention.v.weight"));
    L.o = ws.get(layer_name(i, "layer.0.SelfAttention.o.weight"));
    L.ln1 = ws.get(layer_name(i, "layer.1.layer_norm.weight"));
    L.wi0 = ws.get(layer_name(i, "layer.1.DenseReluDense.wi_0.weight"));
    L.wi1 = ws.get(layer_name(i, "layer.1.DenseReluDense.wi_1.weight"));
    L.wo = ws.get(layer_name(i, "layer.1.DenseReluDense.wo.weight"));
  }
}

// ---- relative position bias ---------------------------------------------------------

int T5TextEncoder::relative_bucket(int64_t rel, int num_buckets, int max_distance) {
  // Bidirectional variant; float32 arithmetic in torch's order: log(n / e)
  // * (1 / log(max_distance / e)) * (nb - e), truncated toward zero.
  int bucket = 0;
  const int nb = num_buckets / 2;
  if (rel > 0) bucket += nb;
  const int64_t n = rel < 0 ? -rel : rel;
  const int max_exact = nb / 2;
  if (n < max_exact) return bucket + static_cast<int>(n);
  const float inv = static_cast<float>(1.0 / std::log(static_cast<double>(max_distance) / max_exact));
  const float x = std::log(static_cast<float>(n) / static_cast<float>(max_exact)) * inv *
                  static_cast<float>(nb - max_exact);
  const int64_t large = std::min<int64_t>(max_exact + static_cast<int64_t>(x), nb - 1);
  return bucket + static_cast<int>(large);
}

Tensor T5TextEncoder::position_bias(int64_t T) {
  auto it = pos_bias_cache_.find(T);
  if (it != pos_bias_cache_.end()) return it->second;
  const int H = cfg_.num_heads;
  std::vector<float> pb(static_cast<size_t>(H) * T * T);
  for (int64_t i = 0; i < T; ++i)
    for (int64_t j = 0; j < T; ++j) {
      const int b = relative_bucket(j - i, cfg_.num_buckets, cfg_.max_distance);
      for (int h = 0; h < H; ++h) pb[(static_cast<size_t>(h) * T + i) * T + j] = rel_bias_[b * H + h];
    }
  Tensor t = be_->from_vector(pb, {1, H, T, T});
  if (pos_bias_cache_.size() > 64) pos_bias_cache_.clear();
  pos_bias_cache_.emplace(T, t);
  return t;
}

// ---- forward ----------------------------------------------------------------------------

Tensor T5TextEncoder::forward(const std::vector<int32_t>& ids, const std::vector<int32_t>& mask,
                              int64_t B, int64_t T, T5Intermediates* dbg) {
  if (B <= 0 || T <= 0 || ids.size() != static_cast<size_t>(B * T) || mask.size() != ids.size())
    throw std::runtime_error("T5TextEncoder::forward: bad ids / mask size");
  Backend& be = *be_;
  const int H = cfg_.num_heads, Dk = cfg_.d_kv;

  // Additive key mask, HF style: 0 where attended, finfo(float32).min elsewhere.
  std::vector<float> mbias(static_cast<size_t>(B * T));
  for (size_t i = 0; i < mbias.size(); ++i) mbias[i] = mask[i] ? 0.f : std::numeric_limits<float>::lowest();
  Tensor pos = position_bias(T);                                       // [1,H,T,T]
  Tensor bias = be.binary(BinaryOp::Add, pos, be.from_vector(mbias, {B, 1, 1, T}));  // [B,H,T,T]

  Tensor x = be.gather_rows(embed_, ids, {B, T});     // [B,T,D]
  if (dbg) { dbg->embeddings = x; dbg->position_bias = pos; }

  const auto heads = [&](const Tensor& t) {  // [B,T,H*Dk] -> [B,H,T,Dk]
    return be.permute(t.reshape({B, T, H, Dk}), {0, 2, 1, 3});
  };
  for (int l = 0; l < cfg_.num_layers; ++l) {
    const Layer& L = layers_[l];
    Tensor h = be.rms_norm(x, L.ln0, cfg_.eps);
    Tensor q = heads(be.linear(h, L.q, Tensor()));
    Tensor k = heads(be.linear(h, L.k, Tensor()));
    Tensor v = heads(be.linear(h, L.v, Tensor()));
    Tensor a = be.attention(q, k, v, 1.0f, bias);                      // T5: no 1/sqrt(d)
    a = be.permute(a, {0, 2, 1, 3}).reshape({B, T, H * Dk});
    x = be.binary(BinaryOp::Add, x, be.linear(a, L.o, Tensor()));

    h = be.rms_norm(x, L.ln1, cfg_.eps);
    Tensor g = be.unary(UnaryOp::GeluNew, be.linear(h, L.wi0, Tensor()));
    Tensor f = be.binary(BinaryOp::Mul, g, be.linear(h, L.wi1, Tensor()));
    x = be.binary(BinaryOp::Add, x, be.linear(f, L.wo, Tensor()));
    if (dbg && l == 0) dbg->block0 = x;
  }
  return be.rms_norm(x, final_ln_, cfg_.eps);
}

// ---- batching ----------------------------------------------------------------------------

T5TextEncoder::Tokenised T5TextEncoder::tokenise(const std::vector<std::string>& texts, size_t begin,
                                                 size_t end, int64_t pad_to) const {
  Tokenised r;
  r.B = static_cast<int64_t>(end - begin);
  std::vector<std::vector<int32_t>> rows;
  rows.reserve(end - begin);
  int64_t T = 1;
  for (size_t i = begin; i < end; ++i) {
    rows.push_back(tok_.encode(texts[i]));
    T = std::max<int64_t>(T, static_cast<int64_t>(rows.back().size()));
  }
  if (pad_to > 0) T = std::max(T, pad_to);
  r.T = T;
  r.ids.assign(static_cast<size_t>(r.B * T), tok_.pad_id());
  r.mask.assign(static_cast<size_t>(r.B * T), 0);
  r.lengths.assign(static_cast<size_t>(r.B), 0);
  for (int64_t b = 0; b < r.B; ++b) {
    const auto& row = rows[static_cast<size_t>(b)];
    const bool empty = texts[begin + static_cast<size_t>(b)].empty();  // T5Conditioner: e == ""
    for (size_t t = 0; t < row.size(); ++t) {
      r.ids[static_cast<size_t>(b * T) + t] = row[t];
      r.mask[static_cast<size_t>(b * T) + t] = empty ? 0 : 1;
    }
    r.lengths[static_cast<size_t>(b)] = empty ? 0 : static_cast<int32_t>(row.size());
  }
  return r;
}

T5Encoding T5TextEncoder::encode(const std::vector<std::string>& texts, T5Intermediates* dbg) {
  if (texts.empty()) throw std::runtime_error("T5TextEncoder::encode: empty batch");
  T5Encoding out;
  Tokenised all = tokenise(texts, 0, texts.size(), 0);
  const int64_t B = all.B, T = all.T;
  std::vector<Tensor> parts;
  for (int64_t b0 = 0; b0 < B; b0 += max_batch_) {
    const int64_t b1 = std::min<int64_t>(B, b0 + max_batch_);
    std::vector<int32_t> ids(all.ids.begin() + b0 * T, all.ids.begin() + b1 * T);
    std::vector<int32_t> mask(all.mask.begin() + b0 * T, all.mask.begin() + b1 * T);
    T5Intermediates part_dbg;
    parts.push_back(forward(ids, mask, b1 - b0, T, dbg ? &part_dbg : nullptr));
    if (dbg) {
      if (b0 == 0) {
        *dbg = part_dbg;
      } else {
        dbg->embeddings = be_->concat({dbg->embeddings, part_dbg.embeddings}, 0);
        dbg->block0 = be_->concat({dbg->block0, part_dbg.block0}, 0);
      }
    }
  }
  out.hidden = parts.size() == 1 ? parts[0] : be_->concat(parts, 0);
  out.input_ids = HostArray::from_i32(all.ids, {B, T});
  out.attention_mask = HostArray::from_i32(all.mask, {B, T});
  out.lengths = std::move(all.lengths);
  return out;
}

HostArray T5TextEncoder::encode_pooled(const std::vector<std::string>& texts) {
  if (texts.empty()) throw std::runtime_error("T5TextEncoder::encode_pooled: empty batch");
  Backend& be = *be_;
  const int64_t D = cfg_.d_model;
  std::vector<float> out;
  out.reserve(texts.size() * static_cast<size_t>(D));
  for (size_t b0 = 0; b0 < texts.size(); b0 += static_cast<size_t>(max_batch_)) {
    const size_t b1 = std::min(texts.size(), b0 + static_cast<size_t>(max_batch_));
    Tokenised tk = tokenise(texts, b0, b1, 0);
    Tensor hid = forward(tk.ids, tk.mask, tk.B, tk.T);                    // [B,T,D]
    std::vector<float> m(tk.mask.size());
    for (size_t i = 0; i < m.size(); ++i) m[i] = tk.mask[i] ? 1.f : 0.f;
    std::vector<float> cnt(static_cast<size_t>(tk.B));
    for (int64_t b = 0; b < tk.B; ++b) cnt[b] = static_cast<float>(std::max(tk.lengths[b], 1));
    Tensor masked = be.binary(BinaryOp::Mul, hid, be.from_vector(m, {tk.B, tk.T, 1}));
    Tensor pooled = be.binary(BinaryOp::Div, be.reduce(ReduceOp::Sum, masked, 1, false),
                              be.from_vector(cnt, {tk.B, 1}));             // [B,D]
    std::vector<float> host = pooled.to_vector();
    out.insert(out.end(), host.begin(), host.end());
  }
  return HostArray::from_f32(out, {static_cast<int64_t>(texts.size()), D});
}

}  // namespace um

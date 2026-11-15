// um::T5TextEncoder -- the flan-t5-base encoder as UniMate's T5Conditioner
// uses it (reference/UniMate/unimate/models/text_encoder/t5.py), on any
// um::Backend.
//
// Semantics mirrored from upstream (normalize_text=False, word_dropout=0,
// finetune=False -- the only configuration create_text_encoder builds):
//   * texts are tokenised as-is (no lower-casing, lemmatising or stripping:
//     upstream's spaCy normaliser is off on every inference / caching path);
//   * the batch is right-padded with <pad> (id 0) to its longest row and an
//     additive mask (0 / -FLT_MAX, torch.finfo(float32).min) hides padding;
//   * an EMPTY string still tokenises to [</s>], but its attention-mask row is
//     zeroed (T5Conditioner.tokenize), so the encoder attends uniformly over
//     all T positions of that row and the pooled vector is exactly 0;
//   * pooled = sum_t(hidden * mask) / max(sum_t mask, 1)  (T5Conditioner.forward).
//
// Model: embedding -> 12 x [RMSNorm -> MHA(no scaling, + relative position
// bias of block 0, shared) -> residual -> RMSNorm -> gated-GELU(tanh, wi_0)
// * wi_1 -> wo -> residual] -> RMSNorm. The relative position bias
// (bidirectional, 32 buckets, max distance 128) is computed on the host once
// per sequence length and cached.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "core/umtf.h"
#include "t5/t5_tokenizer.h"

namespace um {

struct T5Config {
  int d_model = 768, d_kv = 64, d_ff = 2048, num_layers = 12, num_heads = 12;
  int num_buckets = 32, max_distance = 128;
  float eps = 1e-6f;
};

struct T5Encoding {
  Tensor hidden;                  // [B, T, d_model] on the backend
  HostArray input_ids;            // i32 [B, T], right-padded with pad_id
  HostArray attention_mask;       // i32 [B, T] 0/1 (all-zero row for "")
  std::vector<int32_t> lengths;   // sum of each mask row (0 for "")
};

// Optional intermediate activations of encode() (tests / debugging).
struct T5Intermediates {
  Tensor embeddings;     // [B, T, d_model]  embed_tokens output
  Tensor block0;         // [B, T, d_model]  output of encoder.block[0]
  Tensor position_bias;  // [1, H, T, T]     relative position bias (no mask)
};

class T5TextEncoder {
 public:
  // Loads weights + tokenizer from the UMTF written by tools/python/t5_export.py.
  T5TextEncoder(Backend& be, const std::string& umtf_path);

  const T5Tokenizer& tokenizer() const { return tok_; }
  const T5Config& config() const { return cfg_; }
  int dim() const { return cfg_.d_model; }

  // Rows processed per forward pass; larger batches are split into chunks
  // (each padded to the batch-wide length T, so results do not depend on it).
  void set_max_batch(int rows) { max_batch_ = rows > 0 ? rows : 1; }
  int max_batch() const { return max_batch_; }

  // Token-level states, like T5Conditioner(pool=False). B = texts.size() >= 1.
  T5Encoding encode(const std::vector<std::string>& texts, T5Intermediates* dbg = nullptr);

  // Masked mean pooling, like T5Conditioner(pool=True): f32 [B, d_model].
  // Chunks are padded to their own longest row only (cheaper; identical result).
  HostArray encode_pooled(const std::vector<std::string>& texts);

  // Lower level: run the encoder on already tokenised rows (equal length T,
  // mask 0/1, row-major [B, T]).
  Tensor forward(const std::vector<int32_t>& ids, const std::vector<int32_t>& mask,
                 int64_t B, int64_t T, T5Intermediates* dbg = nullptr);

  // Relative-position bucket (HF T5Attention._relative_position_bucket,
  // bidirectional, float32 log arithmetic like torch).
  static int relative_bucket(int64_t rel, int num_buckets, int max_distance);
  // [1, H, T, T] position bias, cached per T.
  Tensor position_bias(int64_t T);

 private:
  struct Tokenised {
    std::vector<int32_t> ids, mask, lengths;
    int64_t B = 0, T = 0;
  };
  Tokenised tokenise(const std::vector<std::string>& texts, size_t begin, size_t end,
                     int64_t pad_to) const;

  struct Layer {
    Tensor ln0, q, k, v, o, ln1, wi0, wi1, wo;
  };

  Backend* be_;
  T5Tokenizer tok_;
  Tensor embed_, final_ln_;
  std::vector<Layer> layers_;
  T5Config cfg_;
  std::vector<float> rel_bias_;  // [num_buckets, H] host copy
  std::map<int64_t, Tensor> pos_bias_cache_;
  int max_batch_ = 64;
};

}  // namespace um

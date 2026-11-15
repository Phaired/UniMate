// um::T5Tokenizer -- SentencePiece-unigram tokenizer reproducing the HF
// `T5Tokenizer(text).input_ids` of google/flan-t5-base (transformers 5.x, i.e.
// the `tokenizers` Rust pipeline), with no protobuf / sentencepiece
// dependency. All tables come from the converted UMTF file written by
// tools/python/t5_export.py.
//
// Pipeline (identical order to HF):
//   0. Invalid UTF-8 bytes are replaced by U+FFFD (a Python str cannot hold
//      them, so HF never sees them; U+FFFD is not in the vocabulary -> <unk>).
//   1. Added special tokens (<pad>, </s>, <unk>, <extra_id_0..99>) are split
//      out of the RAW text first (leftmost-longest match) and emitted as their
//      ids; only the text between them goes through steps 2-5.
//   2. Normaliser "Precompiled" = sentencepiece's nmt_nfkc charsmap (NFKC +
//      control-char removal + whitespace mapping), applied the way
//      tokenizers' Precompiled does: per extended grapheme cluster (UAX #29,
//      tables exported from the Python `regex` module), a cluster shorter than
//      6 bytes is looked up whole (shortest-prefix hit replaces the whole
//      cluster), otherwise char by char.
//   3. WhitespaceSplit on Unicode White_Space (so runs of spaces / tabs /
//      newlines, leading and trailing whitespace all vanish).
//   4. Metaspace: every word gets a U+2581 "▁" prefix unless it already starts
//      with one, and is split before each run of U+2581.
//   5. Unigram Viterbi per word (best-path DP in double precision, ties keep
//      the earliest split), unknown characters scored min_score - 10 and runs
//      of unknowns fused into a single <unk>.
//   6. </s> appended.
//
// Exactness: bit-exact with HF for ASCII (mandatory, tested) and for every
// string of the golden set; non-ASCII is tested against a randomised 4k-string
// stress set. Never throws on any byte string.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace um {

class UmtfFile;

class T5Tokenizer {
 public:
  T5Tokenizer() = default;
  explicit T5Tokenizer(const UmtfFile& file);       // tables from the converted model
  static T5Tokenizer load(const std::string& umtf_path);

  // HF T5Tokenizer(text)["input_ids"]: pieces + </s> (when add_eos).
  std::vector<int32_t> encode(std::string_view text, bool add_eos = true) const;

  // Intermediate stages, exposed for debugging / tests.
  std::string normalize(std::string_view text) const;                   // step 2
  std::vector<std::string> pre_tokenize(std::string_view normalized) const;  // steps 3-4
  std::vector<int32_t> encode_word(std::string_view word) const;        // step 5

  int vocab_size() const { return static_cast<int>(pieces_.size()); }
  int pad_id() const { return pad_id_; }
  int eos_id() const { return eos_id_; }
  int unk_id() const { return unk_id_; }
  const std::string& piece(int id) const { return pieces_.at(id); }
  int piece_type(int id) const { return types_.at(id); }  // 1 normal 2 unk 3 control 4 added
  bool valid() const { return !pieces_.empty(); }

 private:
  // darts-clone common-prefix search over the charsmap trie; returns the
  // normalised replacement of the SHORTEST matching prefix of `key`, or
  // nullptr (tokenizers' spm_precompiled semantics).
  const char* charsmap_lookup(std::string_view key) const;
  int gcb_props(uint32_t cp) const;
  void append_normalized_segment(std::string_view text, std::vector<int32_t>& ids) const;

  std::vector<std::string> pieces_;
  std::vector<double> scores_;
  std::vector<int32_t> types_;
  std::unordered_map<std::string, int32_t> piece_to_id_;
  size_t max_piece_bytes_ = 0;
  std::vector<std::pair<std::string, int32_t>> added_;  // content, id
  std::vector<uint32_t> trie_units_;
  std::string norm_blob_;
  std::vector<int32_t> gcb_first_, gcb_last_, gcb_val_;
  double unk_score_ = 0.0;
  int pad_id_ = 0, eos_id_ = 1, unk_id_ = 2;
};

}  // namespace um

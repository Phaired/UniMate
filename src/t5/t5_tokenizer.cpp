#include "t5/t5_tokenizer.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "core/umtf.h"

namespace um {
namespace {

constexpr uint32_t kMeta = 0x2581;  // "▁"

// ---- UTF-8 ------------------------------------------------------------------

// Decodes one code point at s[i]; returns its byte length, or 0 if the bytes
// there are not well-formed UTF-8 (overlong, surrogate, > U+10FFFF, truncated).
size_t utf8_decode(std::string_view s, size_t i, uint32_t& cp) {
  const auto b = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
  const unsigned char c = b(i);
  if (c < 0x80) { cp = c; return 1; }
  size_t n;
  uint32_t min;
  if ((c & 0xE0) == 0xC0) { n = 2; cp = c & 0x1F; min = 0x80; }
  else if ((c & 0xF0) == 0xE0) { n = 3; cp = c & 0x0F; min = 0x800; }
  else if ((c & 0xF8) == 0xF0) { n = 4; cp = c & 0x07; min = 0x10000; }
  else return 0;
  if (i + n > s.size()) return 0;
  for (size_t k = 1; k < n; ++k) {
    if ((b(i + k) & 0xC0) != 0x80) return 0;
    cp = (cp << 6) | (b(i + k) & 0x3F);
  }
  if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return 0;
  return n;
}

void utf8_append(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
}

// Replace every ill-formed byte by U+FFFD so later stages can assume valid UTF-8.
std::string sanitize_utf8(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    uint32_t cp;
    size_t n = utf8_decode(s, i, cp);
    if (n == 0) {
      utf8_append(out, 0xFFFD);
      i += 1;
    } else {
      out.append(s.substr(i, n));
      i += n;
    }
  }
  return out;
}

struct Cp {
  uint32_t cp;
  size_t pos, len;  // byte range in the (valid) UTF-8 string
};

std::vector<Cp> decode_all(std::string_view s) {
  std::vector<Cp> v;
  v.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    uint32_t cp = 0xFFFD;
    size_t n = utf8_decode(s, i, cp);
    if (n == 0) { cp = 0xFFFD; n = 1; }  // unreachable after sanitize_utf8
    v.push_back({cp, i, n});
    i += n;
  }
  return v;
}

// Rust char::is_whitespace (Unicode White_Space), used by WhitespaceSplit.
bool is_white_space(uint32_t c) {
  return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
         c == 0x205F || c == 0x3000;
}

// ---- grapheme clusters (UAX #29 extended) -----------------------------------

enum Gcb { kOther, kCR, kLF, kControl, kExtend, kZWJ, kRI, kPrepend, kSpacingMark,
           kL, kV, kT, kLV, kLVT };
constexpr int kExtPict = 16;
enum InCB { kInNone = 0, kInLinker = 1, kInConsonant = 2, kInExtend = 3 };

}  // namespace

// ---- construction -------------------------------------------------------------

T5Tokenizer::T5Tokenizer(const UmtfFile& f) {
  const auto& meta = f.meta().at("tokenizer");
  pad_id_ = meta.at("pad_id").get<int>();
  eos_id_ = meta.at("eos_id").get<int>();
  unk_id_ = meta.at("unk_id").get<int>();
  const double min_score = meta.at("min_score").get<double>();
  unk_score_ = min_score - meta.value("unk_penalty", 10.0);

  const HostArray& blob = f.get("tokenizer.pieces");
  const std::vector<int32_t> offs = f.get("tokenizer.piece_offsets").to_i32();
  const std::vector<float> scores = f.get("tokenizer.scores").to_f32();
  types_ = f.get("tokenizer.piece_types").to_i32();
  const size_t V = scores.size();
  if (offs.size() != V + 1 || types_.size() != V || blob.dtype != DType::U8)
    throw std::runtime_error("T5Tokenizer: inconsistent vocabulary tables");
  const char* bp = reinterpret_cast<const char*>(blob.bytes.data());
  pieces_.resize(V);
  scores_.resize(V);
  piece_to_id_.reserve(V * 2);
  for (size_t i = 0; i < V; ++i) {
    pieces_[i].assign(bp + offs[i], bp + offs[i + 1]);
    scores_[i] = scores[i];
    piece_to_id_.emplace(pieces_[i], static_cast<int32_t>(i));
    max_piece_bytes_ = std::max(max_piece_bytes_, pieces_[i].size());
  }
  for (int32_t id : f.get("tokenizer.added_ids").to_i32()) added_.emplace_back(pieces_.at(id), id);

  const HostArray& cm = f.get("tokenizer.precompiled_charsmap");
  if (cm.bytes.size() < 4) throw std::runtime_error("T5Tokenizer: bad charsmap");
  uint32_t trie_bytes;
  std::memcpy(&trie_bytes, cm.bytes.data(), 4);
  if (4 + static_cast<size_t>(trie_bytes) > cm.bytes.size() || trie_bytes % 4)
    throw std::runtime_error("T5Tokenizer: bad charsmap trie size");
  trie_units_.resize(trie_bytes / 4);
  std::memcpy(trie_units_.data(), cm.bytes.data() + 4, trie_bytes);
  norm_blob_.assign(reinterpret_cast<const char*>(cm.bytes.data()) + 4 + trie_bytes,
                    cm.bytes.size() - 4 - trie_bytes);
  norm_blob_.push_back('\0');  // guard

  const HostArray& g = f.get("tokenizer.gcb_ranges");
  const std::vector<int32_t> gv = g.to_i32();
  const size_t R = gv.size() / 3;
  gcb_first_.resize(R); gcb_last_.resize(R); gcb_val_.resize(R);
  for (size_t r = 0; r < R; ++r) {
    gcb_first_[r] = gv[3 * r]; gcb_last_[r] = gv[3 * r + 1]; gcb_val_[r] = gv[3 * r + 2];
  }
}

T5Tokenizer T5Tokenizer::load(const std::string& umtf_path) {
  return T5Tokenizer(UmtfFile::load(umtf_path));
}

// ---- normaliser -----------------------------------------------------------------

const char* T5Tokenizer::charsmap_lookup(std::string_view key) const {
  // darts-clone DoubleArray unit layout.
  const auto has_leaf = [](uint32_t u) { return ((u >> 8) & 1) != 0; };
  const auto value = [](uint32_t u) { return u & ((1u << 31) - 1); };
  const auto label = [](uint32_t u) { return u & ((1u << 31) | 0xFF); };
  const auto offset = [](uint32_t u) { return (u >> 10) << ((u & (1u << 9)) >> 6); };
  if (trie_units_.empty()) return nullptr;
  size_t node = 0;
  uint32_t unit = trie_units_[0];
  node ^= offset(unit);
  for (unsigned char c : key) {
    node ^= c;
    if (node >= trie_units_.size()) return nullptr;
    unit = trie_units_[node];
    if (label(unit) != c) return nullptr;
    node ^= offset(unit);
    if (node >= trie_units_.size()) return nullptr;
    if (has_leaf(unit)) {
      const uint32_t v = value(trie_units_[node]);
      return v < norm_blob_.size() ? norm_blob_.c_str() + v : nullptr;  // shortest hit
    }
  }
  return nullptr;
}

int T5Tokenizer::gcb_props(uint32_t cp) const {
  auto it = std::upper_bound(gcb_first_.begin(), gcb_first_.end(), static_cast<int32_t>(cp));
  if (it == gcb_first_.begin()) return 0;
  const size_t r = static_cast<size_t>(it - gcb_first_.begin()) - 1;
  return static_cast<int32_t>(cp) <= gcb_last_[r] ? gcb_val_[r] : 0;
}

std::string T5Tokenizer::normalize(std::string_view text_in) const {
  const std::string text = sanitize_utf8(text_in);
  const std::vector<Cp> cps = decode_all(text);
  std::string out;
  out.reserve(text.size());

  const auto emit_char = [&](const Cp& c) {
    std::string_view part(text.data() + c.pos, c.len);
    if (const char* n = charsmap_lookup(part)) out += n; else out.append(part);
  };

  size_t i = 0;
  while (i < cps.size()) {
    // Extend the cluster [i, j) with the UAX #29 rules GB3-GB13 (incl. GB9c).
    size_t j = i + 1;
    int ri_run = (gcb_props(cps[i].cp) & 15) == kRI ? 1 : 0;
    bool pict_seq = (gcb_props(cps[i].cp) & kExtPict) != 0;  // ExtPict Extend* so far
    bool pict_zwj = false;                                    // ... followed by ZWJ
    int incb_state = ((gcb_props(cps[i].cp) >> 5) & 3) == kInConsonant ? 1 : 0;
    while (j < cps.size()) {
      const int pp = gcb_props(cps[j - 1].cp), cp = gcb_props(cps[j].cp);
      const int a = pp & 15, b = cp & 15;
      const int incb_b = (cp >> 5) & 3;
      bool join;
      if (a == kCR && b == kLF) join = true;                                    // GB3
      else if (a == kCR || a == kLF || a == kControl) join = false;             // GB4
      else if (b == kCR || b == kLF || b == kControl) join = false;             // GB5
      else if (a == kL && (b == kL || b == kV || b == kLV || b == kLVT)) join = true;  // GB6
      else if ((a == kLV || a == kV) && (b == kV || b == kT)) join = true;      // GB7
      else if ((a == kLVT || a == kT) && b == kT) join = true;                  // GB8
      else if (b == kExtend || b == kZWJ || b == kSpacingMark) join = true;     // GB9, GB9a
      else if (a == kPrepend) join = true;                                      // GB9b
      else if (incb_state == 2 && incb_b == kInConsonant) join = true;          // GB9c
      else if (pict_zwj && (cp & kExtPict)) join = true;                        // GB11
      else if (a == kRI && b == kRI) join = (ri_run % 2) == 1;                  // GB12/13
      else join = false;                                                        // GB999
      if (!join) break;
      // state update for the char just joined
      pict_zwj = pict_seq && b == kZWJ;
      pict_seq = (cp & kExtPict) ? true : (pict_seq && b == kExtend);
      ri_run = b == kRI ? ri_run + 1 : 0;
      if (incb_b == kInConsonant) incb_state = 1;
      else if (incb_b == kInLinker && incb_state >= 1) incb_state = 2;
      else if (incb_b == kInExtend && incb_state >= 1) {}
      else incb_state = 0;
      ++j;
    }
    const size_t pos = cps[i].pos;
    const size_t len = cps[j - 1].pos + cps[j - 1].len - pos;
    const char* whole = len < 6 ? charsmap_lookup(std::string_view(text.data() + pos, len)) : nullptr;
    if (whole) {
      out += whole;
    } else {
      for (size_t k = i; k < j; ++k) emit_char(cps[k]);
    }
    i = j;
  }
  return out;
}

// ---- pre-tokeniser ----------------------------------------------------------------

std::vector<std::string> T5Tokenizer::pre_tokenize(std::string_view normalized) const {
  const std::string text = sanitize_utf8(normalized);
  const std::vector<Cp> cps = decode_all(text);
  std::vector<std::string> out;
  size_t i = 0;
  while (i < cps.size()) {
    if (is_white_space(cps[i].cp)) { ++i; continue; }
    size_t j = i;
    while (j < cps.size() && !is_white_space(cps[j].cp)) ++j;
    // Metaspace on the word [i, j): prepend "▁" unless present, then split
    // before every run of "▁" (MergedWithNext).
    std::string cur;
    if (cps[i].cp != kMeta) utf8_append(cur, kMeta);
    for (size_t k = i; k < j; ++k) {
      if (cps[k].cp == kMeta && k > i && cps[k - 1].cp != kMeta && !cur.empty()) {
        out.push_back(std::move(cur));
        cur.clear();
      }
      cur.append(text, cps[k].pos, cps[k].len);
    }
    if (!cur.empty()) out.push_back(std::move(cur));
    i = j;
  }
  return out;
}

// ---- unigram model -------------------------------------------------------------------

std::vector<int32_t> T5Tokenizer::encode_word(std::string_view s) const {
  std::vector<int32_t> ids;
  const size_t n = s.size();
  if (n == 0) return ids;
  struct Node { double score = 0; int64_t start = -1; int32_t id = -1; };
  std::vector<Node> best(n + 1);
  best[0].start = 0;
  std::string key;
  for (size_t st = 0; st < n;) {
    uint32_t cp;
    size_t mblen = utf8_decode(s, st, cp);
    if (mblen == 0) mblen = 1;
    const double base = best[st].score;
    bool has_single = false;
    const size_t maxlen = std::min(max_piece_bytes_, n - st);
    for (size_t len = 1; len <= maxlen; ++len) {
      key.assign(s.data() + st, len);
      auto it = piece_to_id_.find(key);
      if (it == piece_to_id_.end()) continue;
      const double cand = base + scores_[it->second];
      Node& t = best[st + len];
      if (t.start < 0 || cand > t.score) { t.score = cand; t.start = static_cast<int64_t>(st); t.id = it->second; }
      if (len == mblen) has_single = true;
    }
    if (!has_single) {
      const double cand = base + unk_score_;
      Node& t = best[st + mblen];
      if (t.start < 0 || cand > t.score) { t.score = cand; t.start = static_cast<int64_t>(st); t.id = unk_id_; }
    }
    st += mblen;
  }
  // Backtrack; consecutive <unk> nodes fuse into one token (looked up by its
  // text, which can only ever resolve to <unk> again).
  std::vector<std::string> toks;
  std::string fused;
  bool in_unk = false;
  for (size_t end = n; end > 0;) {
    const Node& nd = best[end];
    if (nd.start < 0) break;  // unreachable (cannot happen on valid input)
    const size_t st = static_cast<size_t>(nd.start);
    std::string piece(s.substr(st, end - st));
    if (nd.id == unk_id_) {
      fused.insert(0, piece);
      in_unk = true;
    } else {
      if (in_unk) { toks.push_back(fused); fused.clear(); in_unk = false; }
      toks.push_back(std::move(piece));
    }
    end = st;
  }
  if (in_unk) toks.push_back(fused);
  std::reverse(toks.begin(), toks.end());
  ids.reserve(toks.size());
  for (const auto& t : toks) {
    auto it = piece_to_id_.find(t);
    ids.push_back(it == piece_to_id_.end() ? unk_id_ : it->second);
  }
  return ids;
}

// ---- full pipeline --------------------------------------------------------------------

void T5Tokenizer::append_normalized_segment(std::string_view text, std::vector<int32_t>& ids) const {
  if (text.empty()) return;
  for (const std::string& w : pre_tokenize(normalize(text))) {
    std::vector<int32_t> wi = encode_word(w);
    ids.insert(ids.end(), wi.begin(), wi.end());
  }
}

std::vector<int32_t> T5Tokenizer::encode(std::string_view text_in, bool add_eos) const {
  const std::string text = sanitize_utf8(text_in);
  std::vector<int32_t> ids;
  size_t seg = 0;
  for (size_t i = 0; i < text.size();) {
    size_t best_len = 0;
    int32_t best_id = -1;
    for (const auto& [content, id] : added_) {
      if (content.size() > best_len && content[0] == text[i] &&
          text.compare(i, content.size(), content) == 0) {
        best_len = content.size();
        best_id = id;
      }
    }
    if (best_len > 0) {
      append_normalized_segment(std::string_view(text).substr(seg, i - seg), ids);
      ids.push_back(best_id);
      i += best_len;
      seg = i;
    } else {
      ++i;
    }
  }
  append_normalized_segment(std::string_view(text).substr(seg), ids);
  if (add_eos) ids.push_back(eos_id_);
  return ids;
}

}  // namespace um

// Parity tests for um_t5 against tests/golden/t5 (tools/python/t5_golden.py).
// Exit code 77 (ctest SKIP_RETURN_CODE) when the converted model or the
// goldens are missing.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "core/umtf.h"
#include "t5/t5_encoder.h"

#ifndef UM_ROOT_DIR
#error "UM_ROOT_DIR must be defined"
#endif

namespace fs = std::filesystem;
using namespace um;

static int g_fail = 0, g_checks = 0;
#define EXPECT(cond, ...)                                  \
  do {                                                     \
    ++g_checks;                                            \
    if (!(cond)) {                                         \
      ++g_fail;                                            \
      std::printf("  FAIL %s:%d: ", __FILE__, __LINE__);   \
      std::printf(__VA_ARGS__);                            \
      std::printf("\n");                                   \
    }                                                      \
  } while (0)

static std::string ids_str(const std::vector<int32_t>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
  return s + "]";
}

static std::string escaped(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    if (c >= 0x20 && c < 0x7F) o += static_cast<char>(c);
    else { char b[8]; std::snprintf(b, sizeof b, "\\x%02X", c); o += b; }
  }
  return o;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return INFINITY;
  float m = 0.f;
  for (size_t i = 0; i < a.size(); ++i) {
    const float d = std::fabs(a[i] - b[i]);
    if (!(d <= m)) m = std::isnan(d) ? INFINITY : d;
  }
  return m;
}

template <class F>
static double time_ms(F&& f, int reps) {
  f();  // warm-up
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) f();
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

static void test_tokenizer(const T5Tokenizer& tok, const UmtfFile& g) {
  const auto& strings = g.meta().at("strings");
  const size_t n_req = g.meta().at("n_required").get<size_t>();
  const std::vector<int32_t> flat = g.get("ids_flat").to_i32();
  const std::vector<int32_t> offs = g.get("ids_offsets").to_i32();
  size_t req_bad = 0, stress_bad = 0, ascii_n = 0;
  for (size_t i = 0; i < strings.size(); ++i) {
    const std::string s = strings[i].get<std::string>();
    const std::vector<int32_t> want(flat.begin() + offs[i], flat.begin() + offs[i + 1]);
    const std::vector<int32_t> got = tok.encode(s);
    const bool ascii = std::all_of(s.begin(), s.end(), [](char c) { return (unsigned char)c < 0x80; });
    ascii_n += ascii;
    if (got != want) {
      if (i < n_req) ++req_bad; else ++stress_bad;
      if (i < n_req || stress_bad <= 15)
        std::printf("  %s mismatch \"%s\"\n    got  %s\n    want %s\n", i < n_req ? "REQUIRED" : "stress",
                    escaped(s).c_str(), ids_str(got).c_str(), ids_str(want).c_str());
    }
  }
  std::printf("  tokenizer: required %zu/%zu exact, stress %zu/%zu exact (%zu strings pure ASCII)\n",
              n_req - req_bad, n_req, (strings.size() - n_req) - stress_bad, strings.size() - n_req,
              ascii_n);
  EXPECT(req_bad == 0, "%zu required strings mismatch", req_bad);
  EXPECT(stress_bad == 0, "%zu stress strings mismatch", stress_bad);

  // Fail-safe on malformed input: never throws, bad bytes act as U+FFFD.
  const std::string bad = std::string("ab\xff\xfe cd \xc3") + "\xe2\x82";
  std::vector<int32_t> ids;
  bool threw = false;
  try { ids = tok.encode(bad); } catch (...) { threw = true; }
  EXPECT(!threw && !ids.empty() && ids.back() == tok.eos_id(), "malformed UTF-8 handling");
  EXPECT(tok.encode(bad) == tok.encode("ab\xEF\xBF\xBD\xEF\xBF\xBD cd \xEF\xBF\xBD\xEF\xBF\xBD"),
         "malformed bytes should behave like U+FFFD");
  std::string big(20000, 'x');
  for (size_t i = 0; i < big.size(); i += 7) big[i] = ' ';
  EXPECT(tok.encode(big).size() > 1000, "long input");
  EXPECT(tok.encode("") == std::vector<int32_t>{tok.eos_id()}, "empty -> [</s>]");
}

static void test_buckets(const UmtfFile& g, const T5Config& cfg) {
  const int R = g.meta().at("rel_range").get<int>();
  const std::vector<int32_t> want = g.get("rel_buckets").to_i32();
  int bad = 0;
  for (int r = -R; r <= R; ++r)
    if (T5TextEncoder::relative_bucket(r, cfg.num_buckets, cfg.max_distance) != want[r + R]) {
      if (++bad < 5) std::printf("  bucket(%d) = %d want %d\n", r,
                                 T5TextEncoder::relative_bucket(r, cfg.num_buckets, cfg.max_distance), want[r + R]);
    }
  EXPECT(bad == 0, "%d relative buckets differ", bad);
}

static void test_encoder(T5TextEncoder& enc, const UmtfFile& g) {
  float worst_hidden = 0.f, worst_pooled = 0.f;
  std::printf("  %-20s %4s %4s %10s %10s %10s %10s %10s\n", "case", "B", "T", "pos_bias", "embed",
              "block0", "hidden", "pooled");
  for (const auto& c : g.meta().at("cases")) {
    const std::string name = c.at("name").get<std::string>();
    std::vector<std::string> texts;
    for (const auto& t : c.at("texts")) texts.push_back(t.get<std::string>());

    T5Intermediates dbg;
    T5Encoding e = enc.encode(texts, &dbg);
    const HostArray& gid = g.get(name + ".input_ids");
    EXPECT(e.input_ids.shape == gid.shape && e.input_ids.to_i32() == gid.to_i32(), "%s: input_ids", name.c_str());
    EXPECT(e.attention_mask.to_i32() == g.get(name + ".attention_mask").to_i32(), "%s: attention_mask",
           name.c_str());
    if (e.input_ids.shape != gid.shape) continue;

    const float pb = max_abs_diff(dbg.position_bias.to_vector(), g.get(name + ".position_bias").to_f32());
    float emb = NAN, b0 = NAN;
    if (g.has(name + ".embeddings")) {
      emb = max_abs_diff(dbg.embeddings.to_vector(), g.get(name + ".embeddings").to_f32());
      const std::vector<float> gb0 = g.get(name + ".block0").to_f32();
      b0 = max_abs_diff(dbg.block0.to_vector(), gb0);
      float scale = 0.f;
      for (float v : gb0) scale = std::max(scale, std::fabs(v));
      // flan-t5's residual stream reaches |x| ~ 1e3 after block 0: relative check.
      EXPECT(emb == 0.f, "%s: embeddings err %g", name.c_str(), emb);
      EXPECT(b0 <= 1e-6f * scale, "%s: block0 err %g (max |x| %g)", name.c_str(), b0, scale);
      if (name == "prompt0") std::printf("  (block0 max |x| = %.4g -> relative err %.3g)\n", scale, b0 / scale);
    }
    const float hid = max_abs_diff(e.hidden.to_vector(), g.get(name + ".hidden").to_f32());
    const HostArray pooled = enc.encode_pooled(texts);
    const float pool = max_abs_diff(pooled.to_f32(), g.get(name + ".pooled").to_f32());
    std::printf("  %-20s %4lld %4lld %10.3g %10.3g %10.3g %10.3g %10.3g\n", name.c_str(),
                (long long)gid.shape[0], (long long)gid.shape[1], pb, emb, b0, hid, pool);
    EXPECT(pb == 0.f, "%s: position bias err %g", name.c_str(), pb);
    EXPECT(hid <= 1e-4f, "%s: hidden err %g", name.c_str(), hid);
    EXPECT(pool <= 2e-5f, "%s: pooled err %g", name.c_str(), pool);
    worst_hidden = std::max(worst_hidden, hid);
    worst_pooled = std::max(worst_pooled, pool);

    for (size_t b = 0; b < texts.size(); ++b)
      if (texts[b].empty()) {
        const std::vector<float> p = pooled.to_f32();
        bool zero = true;
        for (int d = 0; d < enc.dim(); ++d) zero &= p[b * enc.dim() + d] == 0.f;
        EXPECT(zero && e.lengths[b] == 0, "%s: empty string must pool to exactly 0", name.c_str());
      }
  }
  std::printf("  worst: hidden %.3g, pooled %.3g\n", worst_hidden, worst_pooled);
}

static void test_chunking_and_timing(T5TextEncoder& enc, const UmtfFile& g) {
  std::vector<std::string> joints, prompt;
  for (const auto& c : g.meta().at("cases")) {
    if (c.at("name") == "joints72") for (const auto& t : c.at("texts")) joints.push_back(t.get<std::string>());
    if (c.at("name") == "prompt0") prompt.push_back(c.at("texts")[0].get<std::string>());
  }
  if (joints.empty() || prompt.empty()) return;

  const int saved = enc.max_batch();
  enc.set_max_batch(256);
  const std::vector<float> full = enc.encode(joints).hidden.to_vector();
  const std::vector<float> fullp = enc.encode_pooled(joints).to_f32();
  enc.set_max_batch(16);
  const float dh = max_abs_diff(enc.encode(joints).hidden.to_vector(), full);
  const float dp = max_abs_diff(enc.encode_pooled(joints).to_f32(), fullp);
  std::printf("  chunked(16) vs one batch(72): hidden %.3g, pooled %.3g\n", dh, dp);
  EXPECT(dh <= 1e-6f && dp <= 1e-6f, "chunking changes results");
  enc.set_max_batch(saved);

  const double t1 = time_ms([&] { enc.encode_pooled(prompt); }, 10);
  const double t1h = time_ms([&] { (void)enc.encode(prompt).hidden.to_vector(); }, 10);
  const double t72 = time_ms([&] { enc.encode_pooled(joints); }, 5);
  const double ttok = time_ms([&] { for (const auto& j : joints) (void)enc.tokenizer().encode(j); }, 20);
  std::printf("  CPU time: 1 prompt %.1f ms pooled / %.1f ms tokens, 72 joint names %.1f ms pooled "
              "(tokenising the 72 names: %.3f ms)\n", t1, t1h, t72, ttok);
}

int main() {
  const fs::path root = UM_ROOT_DIR;
  const fs::path model = root / "models" / "converted" / "flan-t5-base.umtf";
  const fs::path gtok = root / "tests" / "golden" / "t5" / "tokenizer.umtf";
  const fs::path genc = root / "tests" / "golden" / "t5" / "encoder.umtf";
  for (const auto& p : {model, gtok, genc})
    if (!fs::exists(p)) {
      std::printf("SKIP: %s not found (run tools/python/t5_export.py and t5_golden.py)\n",
                  p.string().c_str());
      return 77;
    }
  try {
    auto be = make_cpu_backend();
    auto t0 = std::chrono::steady_clock::now();
    T5TextEncoder enc(*be, model.string());
    auto t1 = std::chrono::steady_clock::now();
    std::printf("loaded %s in %.0f ms\n", model.filename().string().c_str(),
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    const UmtfFile gt = UmtfFile::load(gtok.string());
    const UmtfFile ge = UmtfFile::load(genc.string());
    std::printf("[tokenizer]\n");
    test_tokenizer(enc.tokenizer(), gt);
    std::printf("[relative buckets]\n");
    test_buckets(ge, enc.config());
    std::printf("[encoder vs golden, backend %s]\n", be->name().c_str());
    test_encoder(enc, ge);
    std::printf("[chunking / timing]\n");
    test_chunking_and_timing(enc, ge);
  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 1;
  }
  std::printf("%d checks, %d failures\n", g_checks, g_fail);
  return g_fail ? 1 : 0;
}

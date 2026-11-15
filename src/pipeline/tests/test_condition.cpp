// build_condition vs the upstream cond dict (tests/golden/pipeline/<asset>_cond.umtf),
// with the golden T5 embeddings served through a fake TextEmbedFn.
#include <algorithm>
#include <cstring>
#include <map>

#include "pipeline/pipeline.h"
#include "test_common.h"

using namespace um;
using namespace um::pipeline;

static void check_asset(const std::string& asset) {
  const std::string gpath = pt::golden(asset + "_cond.umtf");
  if (!pt::exists(gpath) || !pt::exists(pt::bundle_umtf(asset))) {
    std::printf("[%s] skipped (missing golden or bundle)\n", asset.c_str());
    return;
  }
  UmtfFile g = UmtfFile::load(gpath);
  const auto& meta = g.meta();
  AssetBundle b = load_bundle(pt::bundle_umtf(asset), pt::glb_of(asset));
  Stats s = load_stats(pt::stats_umtf(), b.stats_dataset);
  ModelDims d = pt::exists(pt::config_json()) ? load_model_dims(pt::config_json()) : ModelDims{};
  PT_CHECK(s.key == meta["stats_key"].get<std::string>(), "stats key %s", s.key.c_str());

  std::vector<std::string> jn_texts = meta["joint_name_texts"].get<std::vector<std::string>>();
  std::string prompt = meta["prompt"].get<std::string>();
  std::vector<float> jn_emb = g.get("joint_name_text_emb").to_f32();
  std::vector<float> cap_emb = g.get("caption_emb").to_f32();
  const int T = d.text_dim;
  int calls = 0;
  TextEmbedFn embed = [&](const std::vector<std::string>& texts) {
    ++calls;
    if (texts.size() == 1 && texts[0] == prompt) return HostArray::from_f32(cap_emb, {1, T});
    PT_CHECK(texts == jn_texts, "joint-name strings fed to the encoder differ from upstream");
    std::vector<float> out(texts.size() * T, 0.f);
    for (size_t i = 0; i < texts.size(); ++i) {
      auto it = std::find(jn_texts.begin(), jn_texts.end(), texts[i]);
      if (it != jn_texts.end())
        std::copy_n(jn_emb.begin() + (it - jn_texts.begin()) * T, T, out.begin() + i * T);
    }
    return HostArray::from_f32(out, {static_cast<int64_t>(texts.size()), T});
  };

  ArrayDict c = build_condition(b, s, d, prompt, d.max_frames, embed);
  PT_CHECK(calls == 2, "embed called %d times", calls);

  std::printf("[%s] %-26s %-14s %-5s %s\n", asset.c_str(), "key", "shape", "dtype", "max|diff| (max|ref|)");
  for (const auto& name : g.names()) {
    if (name.rfind("cond/", 0) != 0) continue;
    std::string key = name.substr(5);
    const HostArray& ref = g.get(name);
    if (key == "caption_tokens" || key == "caption_mask") {
      std::printf("[%s] %-26s not produced (text_cond='adaln' does not read it)\n", asset.c_str(), key.c_str());
      PT_CHECK(c.count(key) == 0, "unexpected key %s", key.c_str());
      continue;
    }
    auto it = c.find(key);
    PT_CHECK(it != c.end(), "missing key %s", key.c_str());
    if (it == c.end()) continue;
    const HostArray& got = it->second;
    PT_CHECK(got.shape == ref.shape, "%s shape %s vs %s", key.c_str(), shape_str(got.shape).c_str(),
             shape_str(ref.shape).c_str());
    PT_CHECK(got.dtype == ref.dtype, "%s dtype %s vs %s", key.c_str(), dtype_name(got.dtype), dtype_name(ref.dtype));
    if (got.shape != ref.shape) continue;
    if (ref.dtype == DType::F32) {
      auto a = got.to_f32(), r = ref.to_f32();
      double diff = pt::max_abs_diff(a, r), mx = pt::max_abs(r);
      std::printf("[%s] %-26s %-14s %-5s %.3g (%.3g)\n", asset.c_str(), key.c_str(),
                  shape_str(ref.shape).c_str(), dtype_name(ref.dtype), diff, mx);
      PT_CHECK(diff <= 1e-5 * std::max(1.0, mx), "%s max diff %g", key.c_str(), diff);
    } else {
      bool same = got.to_i64() == ref.to_i64();
      std::printf("[%s] %-26s %-14s %-5s %s\n", asset.c_str(), key.c_str(), shape_str(ref.shape).c_str(),
                  dtype_name(ref.dtype), same ? "identical" : "DIFFERENT");
      PT_CHECK(same, "%s differs", key.c_str());
    }
  }
  for (const auto& [k, v] : c)
    PT_CHECK(g.has("cond/" + k), "extra key %s not in upstream cond", k.c_str());

  // recomputed topology == stored cond entry
  std::vector<int> rel, dist;
  compute_edge_relations_and_distances(b.parents, rel, dist, 5);
  PT_CHECK(rel == b.joint_relations, "joint_relations recompute differs");
  PT_CHECK(dist == b.joint_graph_dists, "graph dist recompute differs");
  PT_CHECK(compute_joint_depths(b.parents) == b.joint_depths, "joint depths recompute differs");
  // spectral fallback vs stored (per-column sign aligned)
  if (b.spectral_k > 0) {
    std::vector<float> sp = compute_laplacian_eigenvectors(b.parents, b.spectral_k);
    double worst = 0;
    for (int k = 0; k < b.spectral_k; ++k) {
      double dp = 0, dm = 0;
      for (int j = 0; j < b.num_joints; ++j) {
        double x = sp[j * b.spectral_k + k], y = b.spectral_feats[j * b.spectral_k + k];
        dp = std::max(dp, std::abs(x - y));
        dm = std::max(dm, std::abs(x + y));
      }
      worst = std::max(worst, std::min(dp, dm));
    }
    std::printf("[%s] spectral recompute vs stored (up to column sign): %.3g\n", asset.c_str(), worst);
  }
}

// Cross-check against the denoiser port's goldens (tests/golden/denoiser/<asset>.umtf,
// keys "cond.<key>", built independently through the upstream path). Joint
// names use this module's golden T5 embeddings; the caption is theirs.
static void check_denoiser_golden(const std::string& asset) {
  const std::string dpath = pt::root() + "/tests/golden/denoiser/" + asset + ".umtf";
  const std::string gpath = pt::golden(asset + "_cond.umtf");
  if (!pt::exists(dpath) || !pt::exists(gpath) || !pt::exists(pt::bundle_umtf(asset))) {
    std::printf("[denoiser %s] skipped\n", asset.c_str());
    return;
  }
  UmtfFile dg = UmtfFile::load(dpath), g = UmtfFile::load(gpath);
  AssetBundle b = load_bundle(pt::bundle_umtf(asset), pt::glb_of(asset));
  Stats s = load_stats(pt::stats_umtf(), b.stats_dataset);
  ModelDims d = pt::exists(pt::config_json()) ? load_model_dims(pt::config_json()) : ModelDims{};
  const std::string prompt = dg.meta()["prompts"][0].get<std::string>();
  auto jn_texts = g.meta()["joint_name_texts"].get<std::vector<std::string>>();
  auto jn_emb = g.get("joint_name_text_emb").to_f32();
  auto cap = dg.get("cond.caption_emb").to_f32();
  const int T = d.text_dim;
  TextEmbedFn embed = [&](const std::vector<std::string>& texts) {
    if (texts.size() == 1 && texts[0] == prompt) return HostArray::from_f32(cap, {1, T});
    std::vector<float> out(texts.size() * T, 0.f);
    for (size_t i = 0; i < texts.size(); ++i) {
      auto it = std::find(jn_texts.begin(), jn_texts.end(), texts[i]);
      if (it != jn_texts.end())
        std::copy_n(jn_emb.begin() + (it - jn_texts.begin()) * T, T, out.begin() + i * T);
    }
    return HostArray::from_f32(out, {static_cast<int64_t>(texts.size()), T});
  };
  ArrayDict c = build_condition(b, s, d, prompt, d.max_frames, embed);
  for (const auto& name : dg.names()) {
    if (name.rfind("cond.", 0) != 0) continue;
    std::string key = name.substr(5);
    const HostArray& ref = dg.get(name);
    auto it = c.find(key);
    PT_CHECK(it != c.end() && it->second.shape == ref.shape, "denoiser key %s missing/shape", key.c_str());
    if (it == c.end() || it->second.shape != ref.shape) continue;
    if (ref.dtype == DType::F32) {
      double diff = pt::max_abs_diff(it->second.to_f32(), ref.to_f32());
      std::printf("[denoiser %s] %-26s max|diff| %.3g\n", asset.c_str(), key.c_str(), diff);
      PT_CHECK(diff <= 1e-5 * std::max(1.0, pt::max_abs(ref.to_f32())), "denoiser %s diff %g", key.c_str(), diff);
    } else {
      bool same = it->second.to_i64() == ref.to_i64();
      std::printf("[denoiser %s] %-26s %s\n", asset.c_str(), key.c_str(), same ? "identical" : "DIFFERENT");
      PT_CHECK(same, "denoiser %s differs", key.c_str());
    }
  }
}

int main() {
  pt::require_or_skip({pt::stats_umtf()});
  check_asset("eagle");
  check_asset("shark");
  check_denoiser_golden("eagle");
  check_denoiser_golden("shark");
  // eigenvalue-ordering sanity on a chain
  std::vector<int> chain{-1, 0, 1, 2, 3};
  auto e = compute_laplacian_eigenvectors(chain, 8);
  PT_CHECK(e.size() == 5 * 8, "size");
  return pt::finish("test_condition");
}

// um::Generator -- text prompt + converted asset -> MotionClip, the complete
// UniMate inference path (upstream unimate/inference/sample.py) on one
// um::Backend:
//
//   T5 (pooled prompt + joint names) -> pipeline::build_condition ->
//   Denoiser::set_condition -> noise -> pipeline::sample_flow (CFG velocity)
//   -> pipeline::recover_motion
//
//   um::Generator gen({.models_dir = "", .backend = um::app::BackendKind::Auto});
//   gen.load_asset(um::app::resolve_asset("eagle", gen.models_dir()));
//   um::MotionClip clip = gen.generate({.prompt = "An object flaps its wings."});
//   um::pipeline::export_glb(gen.asset().glb_path, clip, "out.glb");
//
// CFG follows sample.py: cfg_scale > 1 -> v_u + s (v_c - v_u); cfg_scale == 1
// -> unconditional sampling (force_mask, the prompt is ignored); < 1 -> error.
//
// Threading: all public methods take an internal mutex, so the backend is
// only ever used by one thread at a time (e.g. the viewer's worker thread).
#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "app/assets.h"
#include "app/backend_select.h"
#include "core/generation.h"
#include "core/tensor.h"
#include "core/umtf.h"
#include "pipeline/pipeline.h"

namespace um {

class T5TextEncoder;
class Denoiser;

struct GeneratorOptions {
  std::string models_dir;  // empty: resolve_models_dir("")
  app::BackendKind backend = app::BackendKind::Auto;
  bool verbose = false;    // print backend notice / load times to stderr
  bool trim_after_generate = true;  // Vulkan: release idle pooled buffers after generate()
};

struct GenerationStats {
  double t5_ms = 0;          // text encoding during this call (0 if cached)
  double condition_ms = 0;   // build_condition + set_condition (incl. t5_ms)
  double sample_ms = 0;      // ODE integration
  double recover_ms = 0;     // motion recovery
  double total_ms = 0;
  int nfe = 0;               // velocity calls (one CFG call = one 2B forward)
  int accepted = 0, rejected = 0;
  double ms_per_velocity = 0;  // sample_ms / nfe (includes solver overhead)
  bool prompt_cached = false;
};

class Generator {
 public:
  explicit Generator(const GeneratorOptions& opt = {});
  ~Generator();
  Generator(const Generator&) = delete;
  Generator& operator=(const Generator&) = delete;

  const std::string& models_dir() const { return paths_.dir; }
  const std::string& backend_description() const { return backend_desc_; }
  const std::string& backend_notice() const { return backend_notice_; }
  bool backend_is_vulkan() const { return is_vulkan_; }
  Backend& backend() { return *be_; }
  double load_ms() const { return load_ms_; }  // weights load time

  const pipeline::ModelDims& dims() const { return dims_; }

  // Loads the bundle + GLB, resolves its normalisation stats and encodes its
  // joint names once (cached). No-op if the same asset is already loaded.
  void load_asset(const std::string& bundle_umtf, const std::string& glb);
  void load_asset(const app::AssetPaths& p) { load_asset(p.bundle_umtf, p.glb); }
  bool has_asset() const { return bundle_ != nullptr; }
  const pipeline::AssetBundle& asset() const;
  const pipeline::Stats& stats() const { return stats_; }
  double last_asset_t5_ms() const { return asset_t5_ms_; }

  // Initial noise [1, max_joints, feature_len, max_frames] used by the next
  // generate() calls instead of make_noise(seed); std::nullopt restores the
  // seeded noise.
  void set_initial_noise(std::optional<HostArray> noise);

  // progress is called before every velocity evaluation and after every
  // solver step; an exception it throws (e.g. a cancel request) aborts the
  // generation, leaves the backend usable and is rethrown unchanged.
  // The clip carries the debug skeleton (joint_positions / parents / names).
  MotionClip generate(const GenerateRequest& req, const ProgressFn& progress = {});

  // Results of the last generate().
  const GenerationStats& last_stats() const { return last_; }
  const pipeline::SolverStats& last_solver_stats() const { return last_solver_; }
  const HostArray& last_sample() const { return last_sample_; }       // normalised [1,J,12,F]
  const ArrayDict& last_condition() const { return last_cond_; }

  // Pooled T5 embeddings [N, 768] through the per-string cache.
  HostArray embed(const std::vector<std::string>& texts);

  // Lower-level access for tests (not locked: single-threaded use only).
  T5TextEncoder& text_encoder() { return *t5_; }
  Denoiser& denoiser() { return *den_; }

 private:
  MotionClip generate_locked(const GenerateRequest& req, const ProgressFn& progress);
  HostArray embed_locked(const std::vector<std::string>& texts, double* ms);

  mutable std::mutex mtx_;
  app::ModelPaths paths_;
  std::unique_ptr<Backend> be_;
  bool is_vulkan_ = false;
  std::string backend_desc_, backend_notice_;
  std::unique_ptr<T5TextEncoder> t5_;
  std::unique_ptr<Denoiser> den_;
  pipeline::ModelDims dims_;
  double load_ms_ = 0;
  bool trim_after_generate_ = true;

  std::unique_ptr<pipeline::AssetBundle> bundle_;
  pipeline::Stats stats_;
  double asset_t5_ms_ = 0;

  std::unordered_map<std::string, std::vector<float>> emb_cache_;
  std::optional<HostArray> noise_override_;

  GenerationStats last_;
  pipeline::SolverStats last_solver_;
  HostArray last_sample_;
  ArrayDict last_cond_;
};

}  // namespace um

#include "app/generator.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "model/denoiser.h"
#include "t5/t5_encoder.h"

namespace um {
namespace {

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// pipeline::load_model_dims reads a config.json file; the converted model
// carries the same JSON in its UMTF meta, so apply the same rules to it.
pipeline::ModelDims dims_from_config(const nlohmann::json& j) {
  pipeline::ModelDims d;
  if (j.contains("dataset")) {
    const auto& ds = j["dataset"];
    d.max_joints = ds.value("max_joints", d.max_joints);
    d.max_frames = ds.value("max_motion_length", d.max_frames);
    d.feature_len = ds.value("feature_len", d.feature_len);
    d.max_depth = ds.value("max_depth", d.max_depth);
  }
  if (j.contains("model")) d.max_freqs = j["model"].value("max_freqs", d.max_freqs);
  if (j.contains("sampling") && j["sampling"].contains("cfg_scale") && j["sampling"]["cfg_scale"].is_number())
    d.cfg_scale = j["sampling"]["cfg_scale"].get<float>();
  return d;
}

}  // namespace

Generator::Generator(const GeneratorOptions& opt) {
  const auto t0 = Clock::now();
  paths_ = app::model_paths(app::resolve_models_dir(opt.models_dir));
  app::CreatedBackend cb = app::create_backend(opt.backend);
  be_ = std::move(cb.backend);
  is_vulkan_ = cb.is_vulkan;
  backend_desc_ = cb.description;
  backend_notice_ = cb.notice;
  if (opt.verbose && !backend_notice_.empty()) std::fprintf(stderr, "note: %s\n", backend_notice_.c_str());

  t5_ = std::make_unique<T5TextEncoder>(*be_, paths_.t5);
  den_ = std::make_unique<Denoiser>(*be_, paths_.denoiser);
  dims_ = dims_from_config(den_->config());
  dims_.text_dim = t5_->dim();
  if (dims_.max_joints != den_->max_joints() || dims_.max_frames != den_->max_frames() ||
      dims_.feature_len != den_->feature_len())
    throw std::runtime_error("model config / denoiser dimension mismatch");
  trim_after_generate_ = opt.trim_after_generate;
  load_ms_ = ms_since(t0);
  if (opt.verbose)
    std::fprintf(stderr, "loaded models from %s on %s in %.0f ms\n", paths_.dir.c_str(), backend_desc_.c_str(),
                 load_ms_);
}

Generator::~Generator() = default;

const pipeline::AssetBundle& Generator::asset() const {
  if (!bundle_) throw std::runtime_error("Generator: no asset loaded");
  return *bundle_;
}

HostArray Generator::embed_locked(const std::vector<std::string>& texts, double* ms) {
  const int D = t5_->dim();
  std::vector<std::string> missing;
  for (const auto& s : texts)
    if (!emb_cache_.count(s) && std::find(missing.begin(), missing.end(), s) == missing.end())
      missing.push_back(s);
  if (!missing.empty()) {
    const auto t0 = Clock::now();
    HostArray e = t5_->encode_pooled(missing);
    std::vector<float> v = e.to_f32();
    for (size_t i = 0; i < missing.size(); ++i)
      emb_cache_[missing[i]] = std::vector<float>(v.begin() + i * D, v.begin() + (i + 1) * D);
    if (ms) *ms += ms_since(t0);
  }
  std::vector<float> out(texts.size() * size_t(D));
  for (size_t i = 0; i < texts.size(); ++i) std::copy_n(emb_cache_[texts[i]].begin(), D, out.begin() + i * D);
  return HostArray::from_f32(out, {int64_t(texts.size()), D});
}

HostArray Generator::embed(const std::vector<std::string>& texts) {
  std::lock_guard<std::mutex> lk(mtx_);
  return embed_locked(texts, nullptr);
}

void Generator::load_asset(const std::string& bundle_umtf, const std::string& glb) {
  std::lock_guard<std::mutex> lk(mtx_);
  if (bundle_ && bundle_->umtf_path == bundle_umtf && bundle_->glb_path == glb) return;
  auto b = std::make_unique<pipeline::AssetBundle>(pipeline::load_bundle(bundle_umtf, glb));
  pipeline::Stats s = pipeline::load_stats(paths_.stats, b->stats_dataset);
  // Encode the joint names now, with exactly the batch build_condition will
  // ask for (unique model joint names in first-occurrence order).
  std::vector<std::string> uniq;
  for (const auto& n : b->model_joint_names)
    if (std::find(uniq.begin(), uniq.end(), n) == uniq.end()) uniq.push_back(n);
  double ms = 0;
  embed_locked(uniq, &ms);
  asset_t5_ms_ = ms;
  bundle_ = std::move(b);
  stats_ = std::move(s);
}

void Generator::set_initial_noise(std::optional<HostArray> noise) {
  std::lock_guard<std::mutex> lk(mtx_);
  noise_override_ = std::move(noise);
}

MotionClip Generator::generate(const GenerateRequest& req, const ProgressFn& progress) {
  std::lock_guard<std::mutex> lk(mtx_);
  try {
    return generate_locked(req, progress);
  } catch (...) {
    // Failure or cancellation (the viewer's ProgressFn throws
    // um::viewer::GenerationCancelled mid-sampling): partial tensors are
    // already released by unwinding; drain the queued GPU work and the pool so
    // the backend is clean for the next call, then rethrow unchanged.
    try {
      app::backend_finish(*be_);
      if (is_vulkan_) app::backend_trim(*be_);
    } catch (...) {
    }
    last_sample_ = HostArray{};
    throw;
  }
}

MotionClip Generator::generate_locked(const GenerateRequest& req, const ProgressFn& progress) {
  if (!bundle_) throw std::runtime_error("Generator: no asset loaded");
  if (!(req.cfg_scale >= 1.f))
    throw std::runtime_error("cfg_scale must be >= 1 (got " + std::to_string(req.cfg_scale) + ")");
  const auto t_total = Clock::now();
  auto report = [&](float p, const std::string& s) {
    if (progress) progress(p, s);
  };
  last_ = GenerationStats{};
  last_solver_ = pipeline::SolverStats{};
  const int frames =
      (req.num_frames > 0 && req.num_frames <= dims_.max_frames) ? req.num_frames : dims_.max_frames;
  const bool unconditional = req.cfg_scale == 1.f;  // sample.py: _UnconditionalWrapper

  // ---- conditioning ----
  report(0.f, "encoding text");
  const auto t_cond = Clock::now();
  const std::string prompt = unconditional ? std::string() : req.prompt;
  last_.prompt_cached = prompt.empty() || emb_cache_.count(prompt) > 0;
  double t5_ms = 0;
  pipeline::TextEmbedFn embed_fn = [&](const std::vector<std::string>& t) { return embed_locked(t, &t5_ms); };
  last_cond_ = pipeline::build_condition(*bundle_, stats_, dims_, prompt, frames, embed_fn);
  den_->set_condition(last_cond_);
  last_.t5_ms = t5_ms;
  last_.condition_ms = ms_since(t_cond);
  report(0.05f, "sampling");

  // ---- noise ----
  const Shape shape = {1, dims_.max_joints, dims_.feature_len, dims_.max_frames};
  HostArray noise = noise_override_ ? *noise_override_ : pipeline::make_noise(shape, req.seed);
  if (noise.shape != shape)
    throw std::runtime_error("initial noise has shape " + shape_str(noise.shape) + ", expected " + shape_str(shape));

  // ---- ODE ----
  pipeline::SamplerOptions opt;
  opt.solver = req.solver.empty() ? "dopri5" : req.solver;
  if (opt.solver != "dopri5" && opt.solver != "euler" && opt.solver != "heun")
    throw std::runtime_error("unknown solver '" + opt.solver + "' (dopri5, euler, heun)");
  if (opt.solver != "dopri5") opt.num_steps = std::max(2, req.num_steps);
  const float cfg = req.cfg_scale;
  // Progress (and thus a cancellation point) before every velocity call, on
  // top of the solver's per-step callbacks.
  float last_p = 0.05f;
  int calls = 0;
  pipeline::VelocityFn v = [&](const Tensor& x, float t) {
    report(last_p, "sampling (NFE " + std::to_string(++calls) + ")");
    return unconditional ? den_->forward(x, t, /*force_mask=*/true) : den_->velocity(x, t, cfg);
  };
  const auto t_sample = Clock::now();
  Tensor x = pipeline::sample_flow(
      *be_, v, noise.to_tensor(*be_), opt,
      [&](float p, const std::string& s) {
        last_p = 0.05f + 0.9f * p;
        report(last_p, s);
      },
      &last_solver_);
  last_sample_ = HostArray::from_tensor(x);  // forces completion
  last_.sample_ms = ms_since(t_sample);
  last_.nfe = last_solver_.nfe;
  last_.accepted = last_solver_.accepted;
  last_.rejected = last_solver_.rejected;
  last_.ms_per_velocity = last_.nfe ? last_.sample_ms / last_.nfe : 0;

  // ---- motion ----
  report(0.95f, "recovering motion");
  const auto t_rec = Clock::now();
  MotionClip clip = pipeline::recover_motion(*bundle_, stats_, last_sample_, frames, 30.f);
  clip.name = unconditional ? "unconditional" : req.prompt;
  last_.recover_ms = ms_since(t_rec);
  // The Vulkan buffer pool never shrinks by itself (~2.3 GB after a run).
  if (is_vulkan_ && trim_after_generate_) app::backend_trim(*be_);
  last_.total_ms = ms_since(t_total);
  report(1.f, "done");
  return clip;
}

}  // namespace um

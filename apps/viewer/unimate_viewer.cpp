// unimate_viewer: the real-time viewer driven by the real UniMate generator.
//
//   unimate_viewer [eagle | asset_dir | asset.glb | bundle.umtf] [--prompt TEXT]
//                  [--backend auto|cpu|vulkan] [--models DIR] [--seed N] [--cfg S]
//                  [--solver dopri5|euler|heun] [--steps N]
//                  [--smoke-test SECONDS [--capture out.png]]   (auto-generate, then exit)
//                  [--demo]   (fake procedural generator: run unimate_viewer_demo instead)
//
// Threading: the viewer calls GenerateFn on a worker thread while it renders
// with its own Vulkan device on the main thread. Every use of the compute
// backend (T5, denoiser, sampler) happens inside Generator methods, which are
// serialised by the Generator's mutex; asset changes coming from the UI thread
// only record the new GLB path, and the worker loads it lazily before the next
// generation, so the UI thread never touches the compute backend.
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

#include "app/generator.h"
#include "pipeline/glb.h"
#include "viewer/viewer.h"

namespace {

const char* need(int& i, int argc, char** argv) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "missing value for %s\n", argv[i]);
    std::exit(2);
  }
  return argv[++i];
}

struct Shared {
  std::mutex mtx;
  std::string ui_glb;   // last asset loaded by the viewer (UI thread)
  std::string clip_glb; // asset of the last generated clip
};

}  // namespace

int main(int argc, char** argv) {
  std::string asset_spec = "eagle", models, backend = "auto", capture;
  double smoke = 0;
  bool cfg_given = false;
  um::GenerateRequest req;
  req.prompt = "An object flaps its wings.";
  req.solver = "dopri5";
  req.num_steps = 50;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--prompt") req.prompt = need(i, argc, argv);
    else if (a == "--backend") backend = need(i, argc, argv);
    else if (a == "--models") models = need(i, argc, argv);
    else if (a == "--seed") req.seed = uint32_t(std::strtoul(need(i, argc, argv), nullptr, 10));
    else if (a == "--cfg") req.cfg_scale = float(std::atof(need(i, argc, argv))), cfg_given = true;
    else if (a == "--solver") req.solver = need(i, argc, argv);
    else if (a == "--steps") req.num_steps = std::atoi(need(i, argc, argv));
    else if (a == "--frames") req.num_frames = std::atoi(need(i, argc, argv));
    else if (a == "--smoke-test") smoke = std::atof(need(i, argc, argv));
    else if (a == "--capture") capture = need(i, argc, argv);
    else if (a == "--demo") {
      std::fprintf(stderr, "the procedural demo is the separate executable unimate_viewer_demo\n");
      return 2;
    } else if (a == "-h" || a == "--help") {
      std::printf(
          "usage: %s [eagle|asset_dir|asset.glb|bundle.umtf] [--prompt TEXT] [--backend auto|cpu|vulkan]\n"
          "          [--models DIR] [--seed N] [--cfg S] [--solver dopri5|euler|heun] [--steps N] [--frames N]\n"
          "          [--smoke-test SECONDS [--capture out.png]]\n",
          argv[0]);
      return 0;
    } else if (!a.empty() && a[0] != '-') asset_spec = a;
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }

  std::unique_ptr<um::Generator> gen;
  um::app::AssetPaths initial;
  std::string models_dir;
  try {
    models_dir = um::app::resolve_models_dir(models);
    initial = um::app::resolve_asset(asset_spec, models_dir);
    um::GeneratorOptions go;
    go.models_dir = models_dir;
    go.backend = um::app::parse_backend(backend);
    std::printf("loading models from %s ...\n", models_dir.c_str());
    gen = std::make_unique<um::Generator>(go);
    if (!gen->backend_notice().empty()) std::printf("note: %s\n", gen->backend_notice().c_str());
    std::printf("compute backend: %s (models loaded in %.0f ms)\n", gen->backend_description().c_str(),
                gen->load_ms());
    gen->load_asset(initial);
    if (!cfg_given) req.cfg_scale = gen->dims().cfg_scale;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    if (!gen) return 1;
  }

  auto shared = std::make_shared<Shared>();
  shared->ui_glb = initial.glb;
  um::Generator* g = gen.get();

  um::viewer::ViewerOptions vo;
  vo.asset_glb = initial.glb;
  vo.title = "UniMate - " + gen->backend_description();
  vo.initial_request = req;
  vo.on_asset_loaded = [shared](const std::string& glb) {  // UI thread: record only
    std::lock_guard<std::mutex> lk(shared->mtx);
    shared->ui_glb = glb;
  };
  vo.generate = [shared, g, models_dir](const um::GenerateRequest& r, const um::ProgressFn& progress) {
    std::string glb;
    {
      std::lock_guard<std::mutex> lk(shared->mtx);
      glb = shared->ui_glb;
    }
    if (!g->has_asset() || g->asset().glb_path != glb) {
      if (progress) progress(0.f, "loading asset");
      g->load_asset(um::app::resolve_asset(glb, models_dir));  // throws with the conversion hint
    }
    um::MotionClip clip = g->generate(r, progress);
    {
      std::lock_guard<std::mutex> lk(shared->mtx);
      shared->clip_glb = glb;
    }
    const auto& st = g->last_stats();
    std::printf("generated \"%s\": %d NFE, sampling %.0f ms (%.1f ms/velocity), total %.0f ms\n",
                r.prompt.c_str(), st.nfe, st.sample_ms, st.ms_per_velocity, st.total_ms);
    std::fflush(stdout);
    return clip;
  };
  vo.export_glb = [shared](const um::MotionClip& clip, const std::string& out) {
    std::string glb;
    {
      std::lock_guard<std::mutex> lk(shared->mtx);
      glb = shared->clip_glb.empty() ? shared->ui_glb : shared->clip_glb;
    }
    um::pipeline::export_glb(glb, clip, out);
    std::printf("exported %s\n", out.c_str());
  };
  if (smoke > 0) {
    vo.auto_exit_seconds = smoke;
    vo.auto_generate = true;
    vo.capture_png = capture;
  }
  const int rc = um::viewer::run_viewer(vo);
  return rc;
}

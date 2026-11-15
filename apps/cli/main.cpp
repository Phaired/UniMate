// unimate_cli: text prompt + rigged asset -> animated GLB (+ optional PNG frames).
//
//   unimate_cli --asset eagle --prompt "An object flaps its wings." --out eagle.glb
//               [--seed N] [--cfg 3] [--frames 60] [--solver dopri5|euler|heun] [--steps N]
//               [--backend auto|cpu|vulkan] [--models DIR]
//               [--png preview.png [--png-frame N[,N...]] [--width W --height H --yaw D --pitch D --distance S --no-skeleton]]
//               [--noise file.umtf[:tensor]]  [--list-assets]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "app/generator.h"
#include "pipeline/glb.h"

#ifdef UM_CLI_HAVE_VIEWER
#include "viewer/viewer.h"
#endif

namespace {

void usage(const char* exe) {
  std::printf(
      "usage: %s --asset NAME|DIR|GLB|BUNDLE.umtf --prompt TEXT [options]\n"
      "  --out FILE.glb        animated GLB to write (default: <asset>.glb)\n"
      "  --seed N              noise seed (default 0)\n"
      "  --cfg S               classifier-free guidance scale (default: model config, 3); 1 = unconditional\n"
      "  --frames N            clip length, <= 60 (default 60)\n"
      "  --solver NAME         dopri5 (default, adaptive) | euler | heun\n"
      "  --steps N             fixed-step grid size, torchdiffeq semantics: N-1 steps (default 50)\n"
      "  --backend B           auto (default) | vulkan | cpu\n"
      "  --models DIR          converted models (default: $UNIMATE_MODELS or <repo>/models/converted)\n"
      "  --png FILE.png        render frame(s) offscreen; several frames -> FILE_fNN.png\n"
      "  --png-frame N[,N..]   frame(s) to render (default 30); 'rest' renders the bind pose\n"
      "  --width W --height H  PNG size (default 960x720)\n"
      "  --yaw D --pitch D     camera orbit angles in degrees (default 30, 20)\n"
      "  --distance S          camera distance multiplier (default 1 = framed on the rest pose)\n"
      "  --no-skeleton         do not draw bones over the mesh\n"
      "  --noise F.umtf[:T]    initial noise from tensor T (default 'noise') of a UMTF file\n"
      "  --list-assets         list the converted bundles and exit\n",
      exe);
}

const char* need(int& i, int argc, char** argv) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "missing value for %s\n", argv[i]);
    std::exit(2);
  }
  return argv[++i];
}

std::string frame_png_name(const std::string& base, const std::string& tag, bool multi) {
  if (!multi) return base;
  std::filesystem::path p(base);
  return (p.parent_path() / (p.stem().string() + "_" + tag + p.extension().string())).string();
}

}  // namespace

int main(int argc, char** argv) {
  std::string asset_spec, models, out, png, noise_spec, backend = "auto";
  std::vector<std::string> png_frames;
  int width = 960, height = 720;
  float yaw = 30.f, pitch = 20.f, distance = 1.f;
  bool skeleton = true, list = false, cfg_given = false;
  um::GenerateRequest req;
  req.num_steps = 50;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--asset") asset_spec = need(i, argc, argv);
    else if (a == "--prompt") req.prompt = need(i, argc, argv);
    else if (a == "--seed") req.seed = uint32_t(std::strtoul(need(i, argc, argv), nullptr, 10));
    else if (a == "--cfg") req.cfg_scale = float(std::atof(need(i, argc, argv))), cfg_given = true;
    else if (a == "--frames") req.num_frames = std::atoi(need(i, argc, argv));
    else if (a == "--solver") req.solver = need(i, argc, argv);
    else if (a == "--steps") req.num_steps = std::atoi(need(i, argc, argv));
    else if (a == "--backend") backend = need(i, argc, argv);
    else if (a == "--models") models = need(i, argc, argv);
    else if (a == "--out") out = need(i, argc, argv);
    else if (a == "--png") png = need(i, argc, argv);
    else if (a == "--png-frame") {
      std::stringstream ss(need(i, argc, argv));
      for (std::string t; std::getline(ss, t, ',');)
        if (!t.empty()) png_frames.push_back(t);
    } else if (a == "--width") width = std::atoi(need(i, argc, argv));
    else if (a == "--height") height = std::atoi(need(i, argc, argv));
    else if (a == "--yaw") yaw = float(std::atof(need(i, argc, argv)));
    else if (a == "--pitch") pitch = float(std::atof(need(i, argc, argv)));
    else if (a == "--distance") distance = float(std::atof(need(i, argc, argv)));
    else if (a == "--no-skeleton") skeleton = false;
    else if (a == "--noise") noise_spec = need(i, argc, argv);
    else if (a == "--list-assets") list = true;
    else if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument %s (try --help)\n", a.c_str());
      return 2;
    }
  }

  try {
    if (list) {
      const std::string dir = um::app::resolve_models_dir(models);
      std::printf("converted bundles in %s/bundles:\n", dir.c_str());
      for (const auto& n : um::app::list_bundles(dir)) {
        auto p = um::app::resolve_asset(n, dir);
        std::printf("  %-14s %s\n", n.c_str(), p.glb.c_str());
      }
      return 0;
    }
    if (asset_spec.empty()) {
      usage(argv[0]);
      return 2;
    }
    const auto kind = um::app::parse_backend(backend);
    const std::string models_dir = um::app::resolve_models_dir(models);
    um::app::AssetPaths ap = um::app::resolve_asset(asset_spec, models_dir);
    if (out.empty()) out = ap.name + ".glb";

    um::GeneratorOptions go;
    go.models_dir = models_dir;
    go.backend = kind;
    um::Generator gen(go);
    if (!gen.backend_notice().empty()) std::printf("note: %s\n", gen.backend_notice().c_str());
    std::printf("backend   : %s\n", gen.backend_description().c_str());
    std::printf("models    : %s (loaded in %.0f ms)\n", gen.models_dir().c_str(), gen.load_ms());
    if (!cfg_given) req.cfg_scale = gen.dims().cfg_scale;

    const auto t_asset = std::chrono::steady_clock::now();
    gen.load_asset(ap);
    const double asset_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_asset).count();
    std::printf("asset     : %s (%d joints)  bundle %s\n            glb %s\n", ap.name.c_str(),
                gen.asset().num_joints, ap.bundle_umtf.c_str(), ap.glb.c_str());
    std::printf("            loaded in %.0f ms (joint-name T5 %.0f ms), stats '%s'\n", asset_ms,
                gen.last_asset_t5_ms(), gen.stats().key.c_str());

    if (!noise_spec.empty()) {
      std::string file = noise_spec, tensor = "noise";
      const auto c = noise_spec.find_last_of(':');
      if (c != std::string::npos && c > 1) file = noise_spec.substr(0, c), tensor = noise_spec.substr(c + 1);
      um::UmtfFile f = um::UmtfFile::load(file);
      gen.set_initial_noise(f.get(tensor));
      std::printf("noise     : %s:%s\n", file.c_str(), tensor.c_str());
    }

    std::printf("prompt    : \"%s\"  seed %u  cfg %g  frames %d  solver %s", req.prompt.c_str(), req.seed,
                req.cfg_scale, req.num_frames, req.solver.c_str());
    if (req.solver != "dopri5") std::printf(" (%d steps)", std::max(2, req.num_steps) - 1);
    std::printf("\n");
    if (req.cfg_scale == 1.f) std::printf("note: cfg 1 = unconditional sampling, the prompt is ignored\n");

    int last_pct = -1;
    um::MotionClip clip = gen.generate(req, [&](float p, const std::string& s) {
      int pct = int(p * 100.f);
      if (pct / 5 != last_pct / 5) {
        last_pct = pct;
        std::printf("\r  [%3d%%] %-40s", pct, s.c_str());
        std::fflush(stdout);
      }
    });
    std::printf("\n");
    const auto& st = gen.last_stats();
    const auto& ss = gen.last_solver_stats();
    std::printf("timings   : T5 %.0f ms%s | conditioning %.0f ms | sampling %.0f ms (%d NFE, %.1f ms/velocity",
                st.t5_ms, st.prompt_cached ? " (cached)" : "", st.condition_ms, st.sample_ms, st.nfe,
                st.ms_per_velocity);
    if (req.solver == "dopri5") std::printf(", %d accepted / %d rejected steps", ss.accepted, ss.rejected);
    std::printf(") | recovery %.0f ms | total %.0f ms\n", st.recover_ms, st.total_ms);

    um::pipeline::export_glb(ap.glb, clip, out);
    std::printf("wrote     : %s (%d frames @ %g fps, %zu tracks)\n", out.c_str(), clip.num_frames, clip.fps,
                clip.tracks.size());

    if (!png.empty()) {
#ifdef UM_CLI_HAVE_VIEWER
      if (png_frames.empty()) png_frames.push_back(std::to_string(std::min(30, clip.num_frames - 1)));
      um::viewer::OffscreenOptions off;
      off.yaw_deg = yaw;
      off.pitch_deg = pitch;
      off.distance_scale = distance;
      off.draw_skeleton = skeleton;
      std::string err;
      off.error = &err;
      const bool multi = png_frames.size() > 1;
      for (const auto& f : png_frames) {
        const bool rest = f == "rest";
        const int fi = rest ? 0 : std::atoi(f.c_str());
        char tag[32];
        std::snprintf(tag, sizeof tag, rest ? "rest" : "f%02d", fi);
        const std::string path = frame_png_name(png, tag, multi);
        if (!um::viewer::render_offscreen_png(ap.glb, rest ? nullptr : &clip, fi, width, height, path, off)) {
          std::fprintf(stderr, "offscreen render failed: %s\n", err.c_str());
          return 1;
        }
        std::printf("wrote     : %s\n", path.c_str());
      }
#else
      std::fprintf(stderr, "--png: this build has no viewer (um_viewer); PNG not written\n");
      return 1;
#endif
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

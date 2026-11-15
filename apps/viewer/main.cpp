// unimate_viewer_demo: the UniMate viewer with a fake, procedural generator.
//
//   unimate_viewer_demo <asset.glb>                       interactive window
//   unimate_viewer_demo <asset.glb> --screenshot out.png  [--frame N] [--width W] [--height H]
//                       [--yaw DEG] [--pitch DEG] [--no-skeleton] [--no-grid] [--embedded K]
//   unimate_viewer_demo <asset.glb> --smoke-test SECONDS [--capture out.png]
//
// --frame N renders frame N of the procedural clip (omit it for the rest pose).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "viewer/viewer.h"

namespace {

void quat_mul(const float* a, const float* b, float* o) {  // xyzw
  const float ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const float bx = b[0], by = b[1], bz = b[2], bw = b[3];
  o[0] = aw * bx + ax * bw + ay * bz - az * by;
  o[1] = aw * by - ax * bz + ay * bw + az * bx;
  o[2] = aw * bz + ax * by - ay * bx + az * bw;
  o[3] = aw * bw - ax * bx - ay * by - az * bz;
}

// Sinusoidal rotation on every skin joint (by node name), on top of its rest
// rotation. Deterministic in (asset, seed, num_frames).
um::MotionClip procedural_clip(const std::string& asset, const um::GenerateRequest& req,
                               const um::ProgressFn& progress, bool simulate_work) {
  std::string err;
  auto joints = um::viewer::load_skeleton(asset, &err);
  if (joints.empty()) throw std::runtime_error(err.empty() ? "asset has no skin joints" : err);
  um::MotionClip clip;
  clip.name = "procedural: " + req.prompt;
  clip.fps = 30.f;
  clip.num_frames = std::max(1, req.num_frames);
  const float pi = 3.14159265f;
  const float period = 2.0f;  // seconds
  const float amp = 0.12f + 0.05f * std::min(req.cfg_scale, 6.f);  // radians
  const int steps = simulate_work ? std::max(1, req.solver == "dopri5" ? 40 : req.num_steps) : 1;
  for (int s = 0; s < steps; ++s) {
    if (simulate_work) std::this_thread::sleep_for(std::chrono::milliseconds(25));
    if (progress) progress(float(s + 1) / float(steps), "step " + std::to_string(s + 1) + "/" + std::to_string(steps));
  }
  for (size_t k = 0; k < joints.size(); ++k) {
    const auto& j = joints[k];
    um::NodeTrack t;
    t.node_name = j.name;
    t.rotation.resize(size_t(clip.num_frames) * 4);
    const int axis = int((k + req.seed) % 3);
    const float phase = 0.7f * float(k) + 0.37f * float(req.seed % 97);
    const float a = j.parent < 0 ? amp * 0.3f : amp;
    for (int f = 0; f < clip.num_frames; ++f) {
      const float ang = a * std::sin(2.f * pi * (f / clip.fps) / period + phase);
      float d[4] = {0, 0, 0, std::cos(0.5f * ang)};
      d[axis] = std::sin(0.5f * ang);
      quat_mul(j.rotation, d, &t.rotation[size_t(f) * 4]);
    }
    if (j.parent < 0) {  // root: small vertical bob, rest x/z
      t.translation.resize(size_t(clip.num_frames) * 3);
      for (int f = 0; f < clip.num_frames; ++f) {
        float* p = &t.translation[size_t(f) * 3];
        p[0] = j.translation[0];
        p[1] = j.translation[1] * (1.f + 0.04f * std::sin(4.f * pi * (f / clip.fps) / period));
        p[2] = j.translation[2];
      }
    }
    clip.tracks.push_back(std::move(t));
  }
  return clip;
}

const char* arg_value(int& i, int argc, char** argv) {
  if (i + 1 >= argc) {
    std::fprintf(stderr, "missing value for %s\n", argv[i]);
    std::exit(2);
  }
  return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
  std::string asset, screenshot, capture, prompt = "a creature moving rhythmically";
  int frame = -1, width = 1280, height = 720, embedded = -1;
  double smoke = 0;
  um::viewer::OffscreenOptions off;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--screenshot") screenshot = arg_value(i, argc, argv);
    else if (a == "--frame") frame = std::atoi(arg_value(i, argc, argv));
    else if (a == "--width") width = std::atoi(arg_value(i, argc, argv));
    else if (a == "--height") height = std::atoi(arg_value(i, argc, argv));
    else if (a == "--yaw") off.yaw_deg = float(std::atof(arg_value(i, argc, argv)));
    else if (a == "--pitch") off.pitch_deg = float(std::atof(arg_value(i, argc, argv)));
    else if (a == "--no-skeleton") off.draw_skeleton = false;
    else if (a == "--no-grid") off.draw_grid = false;
    else if (a == "--embedded") embedded = std::atoi(arg_value(i, argc, argv));
    else if (a == "--smoke-test") smoke = std::atof(arg_value(i, argc, argv));
    else if (a == "--capture") capture = arg_value(i, argc, argv);
    else if (a == "--prompt") prompt = arg_value(i, argc, argv);
    else if (a == "-h" || a == "--help") {
      std::printf("usage: %s <asset.glb> [--screenshot out.png [--frame N] [--width W] [--height H] [--yaw D]\n"
                  "       [--pitch D] [--no-skeleton] [--no-grid] [--embedded K]] [--smoke-test SEC [--capture out.png]]\n",
                  argv[0]);
      return 0;
    } else if (!a.empty() && a[0] != '-') asset = a;
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }

  if (!screenshot.empty()) {
    if (asset.empty()) {
      std::fprintf(stderr, "--screenshot needs an asset\n");
      return 2;
    }
    std::unique_ptr<um::MotionClip> clip;
    if (embedded >= 0) {
      auto clips = um::viewer::load_embedded_clips(asset);
      if (embedded >= int(clips.size())) {
        std::fprintf(stderr, "asset has %zu embedded animations\n", clips.size());
        return 1;
      }
      clip = std::make_unique<um::MotionClip>(clips[embedded]);
    } else if (frame >= 0) {
      um::GenerateRequest req;
      req.prompt = prompt;
      req.num_frames = 60;
      clip = std::make_unique<um::MotionClip>(procedural_clip(asset, req, nullptr, false));
    }
    std::string err;
    off.error = &err;
    if (!um::viewer::render_offscreen_png(asset, clip.get(), std::max(frame, 0), width, height, screenshot, off)) {
      std::fprintf(stderr, "offscreen render failed: %s\n", err.c_str());
      return 1;
    }
    std::printf("wrote %s\n", screenshot.c_str());
    return 0;
  }

  // Interactive viewer with the fake generator.
  auto current_asset = std::make_shared<std::string>(asset);
  auto mtx = std::make_shared<std::mutex>();
  um::viewer::ViewerOptions vo;
  vo.asset_glb = asset;
  vo.initial_request.prompt = prompt;
  vo.on_asset_loaded = [current_asset, mtx](const std::string& p) {
    std::lock_guard<std::mutex> lk(*mtx);
    *current_asset = p;
  };
  vo.generate = [current_asset, mtx](const um::GenerateRequest& req, const um::ProgressFn& progress) {
    std::string a;
    {
      std::lock_guard<std::mutex> lk(*mtx);
      a = *current_asset;
    }
    return procedural_clip(a, req, progress, true);
  };
  if (smoke > 0) {
    vo.auto_exit_seconds = smoke;
    vo.auto_generate = true;
    vo.auto_resize = true;
    vo.capture_png = capture;
  }
  return um::viewer::run_viewer(vo);
}

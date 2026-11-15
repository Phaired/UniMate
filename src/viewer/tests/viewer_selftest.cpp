// viewer_selftest: drives um_viewer through its automation hooks with a fake,
// slow generator (sleeps + regular progress callbacks), for testing frame
// pacing, cancellation, shutdown and overlays without the real model.
//
//   viewer_selftest <asset.glb> [--seconds S] [--capture out.png] [--gen-seconds G]
//                   [--no-generate] [--cancel-after S] [--close-during-gen] [--no-vsync]
//                   [--no-model-skeleton] [--model-offset X] [--resize] [--width W --height H]
//                   [--panel W] [--max-fps F] [--gen-fps F]
//   viewer_selftest --blend-test out_dir     writes a GLB with alphaMode BLEND quads and renders it
//   viewer_selftest <asset.glb> --offscreen out.png [--frame N]   offscreen render with a debug-skeleton clip
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "viewer/asset.h"  // internal: exact FK of the asset to build a reference debug skeleton
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

// Procedural clip on every skin joint. Its debug skeleton (joint_positions)
// is the asset's posed joint world positions at every frame, shifted by
// `offset` (a stand-in for a canonical space that differs from glTF world).
um::MotionClip fake_clip(const std::string& asset, const um::GenerateRequest& req, float offset) {
  std::string err;
  auto joints = um::viewer::load_skeleton(asset, &err);
  if (joints.empty()) throw std::runtime_error(err.empty() ? "asset has no skin joints" : err);
  um::MotionClip clip;
  clip.name = "selftest: " + req.prompt;
  clip.fps = 30.f;
  clip.num_frames = std::max(1, req.num_frames);
  const float pi = 3.14159265f;
  for (size_t k = 0; k < joints.size(); ++k) {
    const auto& j = joints[k];
    um::NodeTrack t;
    t.node_name = j.name;
    t.rotation.resize(size_t(clip.num_frames) * 4);
    const int axis = int((k + req.seed) % 3);
    const float phase = 0.7f * float(k);
    for (int f = 0; f < clip.num_frames; ++f) {
      const float ang = (j.parent < 0 ? 0.1f : 0.3f) * std::sin(2.f * pi * (f / clip.fps) / 2.f + phase);
      float d[4] = {0, 0, 0, std::cos(0.5f * ang)};
      d[axis] = std::sin(0.5f * ang);
      quat_mul(j.rotation, d, &t.rotation[size_t(f) * 4]);
    }
    clip.tracks.push_back(std::move(t));
  }
  um::viewer::Asset a;
  if (!um::viewer::load_asset(asset, a, err, false)) throw std::runtime_error(err);
  const auto binding = um::viewer::bind_clip(a, clip);
  um::viewer::Pose p;
  um::viewer::rest_pose(a, p);
  const int nj = int(a.skeleton.size());
  clip.num_joints = nj;
  clip.parents = a.skeleton_parent;
  for (int n : a.skeleton) clip.joint_names.push_back(a.nodes[n].name);
  clip.joint_positions.resize(size_t(clip.num_frames) * nj * 3);
  for (int f = 0; f < clip.num_frames; ++f) {
    um::viewer::sample_clip(a, clip, binding, float(f), false, p);
    for (int k = 0; k < nj; ++k) {
      const glm::vec4 w = p.global[a.skeleton[k]][3];
      float* o = &clip.joint_positions[(size_t(f) * nj + k) * 3];
      o[0] = w.x + offset;
      o[1] = w.y;
      o[2] = w.z;
    }
  }
  return clip;
}

// ---- tiny GLB writer for the alpha-blend test ----
bool write_blend_glb(const std::string& path) {
  // three unit quads facing +Z: opaque red (z=0), blended blue 50% (z=0.4),
  // blended green 50% (z=0.8); each offset in x so overlaps are partial.
  std::vector<float> pos;
  std::vector<uint16_t> idx;
  const float xs[3] = {0.f, 0.5f, 1.0f}, zs[3] = {0.f, 0.4f, 0.8f};
  for (int q = 0; q < 3; ++q) {
    const float x = xs[q], z = zs[q];
    const float v[12] = {x - 1, -1, z, x + 1, -1, z, x + 1, 1, z, x - 1, 1, z};
    pos.insert(pos.end(), v, v + 12);
    const uint16_t b = 0;  // each quad has its own POSITION accessor
    const uint16_t i[6] = {b, uint16_t(b + 1), uint16_t(b + 2), b, uint16_t(b + 2), uint16_t(b + 3)};
    idx.insert(idx.end(), i, i + 6);
  }
  std::string bin(reinterpret_cast<const char*>(idx.data()), idx.size() * 2);
  const size_t pos_off = (bin.size() + 3) & ~size_t(3);
  bin.resize(pos_off, '\0');
  bin.append(reinterpret_cast<const char*>(pos.data()), pos.size() * 4);
  while (bin.size() % 4) bin.push_back('\0');
  std::string json = R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1,2]}],)"
                     R"("nodes":[{"mesh":0,"name":"red"},{"mesh":1,"name":"blue"},{"mesh":2,"name":"green"}],"meshes":[)";
  for (int q = 0; q < 3; ++q) {
    if (q) json += ",";
    json += R"({"primitives":[{"attributes":{"POSITION":)" + std::to_string(3 + q) + R"(},"indices":)" +
            std::to_string(q) + R"(,"material":)" + std::to_string(q) + "}]}";
  }
  json += R"(],"materials":[)"
          R"({"pbrMetallicRoughness":{"baseColorFactor":[1,0.1,0.1,1]}},)"
          R"({"pbrMetallicRoughness":{"baseColorFactor":[0.1,0.2,1,0.5]},"alphaMode":"BLEND","doubleSided":true},)"
          R"({"pbrMetallicRoughness":{"baseColorFactor":[0.1,1,0.2,0.5]},"alphaMode":"BLEND","doubleSided":true}],)";
  json += R"("accessors":[)";
  for (int q = 0; q < 3; ++q)
    json += R"({"bufferView":0,"byteOffset":)" + std::to_string(q * 12) + R"(,"componentType":5123,"count":6,"type":"SCALAR"},)";
  for (int q = 0; q < 3; ++q) {
    const float x = xs[q], z = zs[q];
    char mm[160];
    std::snprintf(mm, sizeof(mm), R"("min":[%g,-1,%g],"max":[%g,1,%g])", x - 1, z, x + 1, z);
    json += R"({"bufferView":1,"byteOffset":)" + std::to_string(q * 48) + R"(,"componentType":5126,"count":4,"type":"VEC3",)" +
            mm + "}" + (q < 2 ? "," : "");
  }
  json += R"(],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":)" + std::to_string(idx.size() * 2) +
          R"(},{"buffer":0,"byteOffset":)" + std::to_string(pos_off) + R"(,"byteLength":)" +
          std::to_string(pos.size() * 4) + R"(}],"buffers":[{"byteLength":)" + std::to_string(bin.size()) + "}]}";
  while (json.size() % 4) json.push_back(' ');
  std::ofstream f(path, std::ios::binary);
  auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  u32(0x46546C67);
  u32(2);
  u32(uint32_t(12 + 8 + json.size() + 8 + bin.size()));
  u32(uint32_t(json.size()));
  u32(0x4E4F534A);
  f.write(json.data(), json.size());
  u32(uint32_t(bin.size()));
  u32(0x004E4942);
  f.write(bin.data(), bin.size());
  return bool(f);
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
  std::string asset, capture, blend_dir, offscreen;
  double seconds = 6.0, gen_seconds = 3.0;
  float model_offset = 0.f;
  int frame = 10;
  bool generate = true, compensate = true;
  um::viewer::ViewerOptions vo;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--seconds") seconds = std::atof(arg_value(i, argc, argv));
    else if (a == "--capture") capture = arg_value(i, argc, argv);
    else if (a == "--gen-seconds") gen_seconds = std::atof(arg_value(i, argc, argv));
    else if (a == "--no-generate") generate = false;
    else if (a == "--no-vsync") vo.vsync = false;
    else if (a == "--resize") vo.auto_resize = true;
    else if (a == "--width") vo.width = std::atoi(arg_value(i, argc, argv));
    else if (a == "--height") vo.height = std::atoi(arg_value(i, argc, argv));
    else if (a == "--panel") vo.panel_width = std::atoi(arg_value(i, argc, argv));
    else if (a == "--max-fps") vo.max_fps = float(std::atof(arg_value(i, argc, argv)));
    else if (a == "--gen-fps") vo.max_fps_while_generating = float(std::atof(arg_value(i, argc, argv)));
    else if (a == "--cancel-after") vo.auto_cancel_after_seconds = std::atof(arg_value(i, argc, argv));
    else if (a == "--close-during-gen") vo.auto_close_while_generating = true;
    else if (a == "--no-model-skeleton") vo.show_model_skeleton = false;
    else if (a == "--model-offset") model_offset = float(std::atof(arg_value(i, argc, argv)));
    else if (a == "--no-compensate") compensate = false;
    else if (a == "--blend-test") blend_dir = arg_value(i, argc, argv);
    else if (a == "--offscreen") offscreen = arg_value(i, argc, argv);
    else if (a == "--frame") frame = std::atoi(arg_value(i, argc, argv));
    else if (!a.empty() && a[0] != '-') asset = a;
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }

  if (!blend_dir.empty()) {
    const std::string glb = blend_dir + "/blend_test.glb";
    if (!write_blend_glb(glb)) return 1;
    {
      um::viewer::Asset a;
      std::string err;
      const bool ok = um::viewer::load_asset(glb, a, err);
      std::printf("blend asset: ok=%d %s draws=%zu materials=%zu\n", int(ok), err.c_str(), a.draws.size(),
                  a.materials.size());
      std::fflush(stdout);
      if (!ok) return 1;
    }
    um::viewer::OffscreenOptions o;
    o.draw_grid = false;
    bool ok = true;
    for (float yaw : {0.f, 180.f, 35.f}) {
      o.yaw_deg = yaw;
      o.pitch_deg = 10.f;
      const std::string out = blend_dir + "/blend_yaw" + std::to_string(int(yaw)) + ".png";
      std::string err;
      o.error = &err;
      if (!um::viewer::render_offscreen_png(glb, nullptr, 0, 640, 400, out, o)) {
        std::fprintf(stderr, "blend render failed: %s\n", err.c_str());
        ok = false;
      } else {
        std::printf("wrote %s\n", out.c_str());
      }
    }
    return ok ? 0 : 1;
  }

  // the model skeleton is shifted by +offset in x; supply the inverse transform
  float* xf = vo.model_skeleton_transform;
  xf[12] = compensate ? -model_offset : 0.f;  // column-major: translation in elements 12..14

  if (!offscreen.empty()) {
    um::GenerateRequest req;
    req.num_frames = 60;
    um::MotionClip clip = fake_clip(asset, req, model_offset);
    um::viewer::OffscreenOptions o;
    o.model_skeleton_transform[12] = compensate ? -model_offset : 0.f;
    std::string err;
    o.error = &err;
    if (!um::viewer::render_offscreen_png(asset, &clip, frame, 1280, 720, offscreen, o)) {
      std::fprintf(stderr, "offscreen failed: %s\n", err.c_str());
      return 1;
    }
    std::printf("wrote %s\n", offscreen.c_str());
    return 0;
  }

  vo.asset_glb = asset;
  vo.auto_exit_seconds = seconds;
  vo.auto_generate = generate;
  vo.capture_png = capture;
  vo.initial_request.prompt = "selftest motion";
  vo.initial_request.num_frames = 60;
  const auto t_start = std::chrono::steady_clock::now();
  vo.generate = [asset, gen_seconds, model_offset](const um::GenerateRequest& req, const um::ProgressFn& progress) {
    const int steps = 50;
    std::printf("[selftest] generator start (%.1f s, %d steps)\n", gen_seconds, steps);
    std::fflush(stdout);
    int s = 0;
    try {
      for (; s < steps; ++s) {
        std::this_thread::sleep_for(std::chrono::duration<double>(gen_seconds / steps));
        progress(float(s + 1) / steps, "step " + std::to_string(s + 1) + "/" + std::to_string(steps));
      }
    } catch (const um::viewer::GenerationCancelled&) {
      std::printf("[selftest] generator unwinding: GenerationCancelled at step %d/%d\n", s + 1, steps);
      std::fflush(stdout);
      throw;
    }
    std::printf("[selftest] generator done\n");
    std::fflush(stdout);
    return fake_clip(asset, req, model_offset);
  };
  const int rc = um::viewer::run_viewer(vo);
  // the generator lambda (and anything it captured) is destroyed after this
  // point; the viewer must already have joined the worker
  const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
  std::printf("[selftest] run_viewer returned %d after %.2f s\n", rc, t);
  return rc;
}

// Headless rendering (no window, no surface) + small asset query helpers.
#include <algorithm>
#include <cstring>

#include <stb_image_write.h>

#include "camera.h"
#include "renderer.h"
#include "viewer.h"

namespace um::viewer {

bool render_offscreen_png(const std::string& asset_glb, const um::MotionClip* clip, int frame, int w, int h,
                          const std::string& out_png) {
  return render_offscreen_png(asset_glb, clip, frame, w, h, out_png, OffscreenOptions{});
}

bool render_offscreen_png(const std::string& asset_glb, const um::MotionClip* clip, int frame, int w, int h,
                          const std::string& out_png, const OffscreenOptions& opt) {
  std::string err;
  auto fail = [&](const std::string& e) {
    if (opt.error) *opt.error = e;
    return false;
  };
  if (w <= 0 || h <= 0) return fail("invalid size");
  Asset asset;
  if (!load_asset(asset_glb, asset, err)) return fail(err);

  VkContext ctx;
  if (!ctx.create_instance({}, err) || !ctx.create_device(VK_NULL_HANDLE, err)) return fail(err);
  const VkFormat fmt = VK_FORMAT_R8G8B8A8_SRGB;
  bool ok = false;
  {
    Renderer r;
    if (!r.init(&ctx, fmt, ctx.clamp_samples(opt.msaa_samples), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1, err))
      return fail(err);
    if (!r.set_asset(asset, err)) return fail(err);
    GpuImage target;
    VkExtent2D ext{uint32_t(w), uint32_t(h)};
    if (!ctx.create_image(ext, fmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                          VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_ASPECT_COLOR_BIT, 1, target))
      return fail("offscreen image allocation failed");
    if (!r.set_targets(ext, {target.view}, err)) return fail(err);
    GpuBuffer readback;
    ctx.create_buffer(VkDeviceSize(w) * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true, readback);

    // pose
    Pose pose;
    rest_pose(asset, pose);
    if (clip && clip->num_frames > 0) {
      std::vector<int> binding = bind_clip(asset, *clip);
      sample_clip(asset, *clip, binding, float(std::clamp(frame, 0, clip->num_frames - 1)), false, pose);
    }
    std::vector<glm::mat4> palette;
    build_palette(asset, pose, palette);
    std::vector<LineVertex> bones, joints, mbones, mjoints;
    build_skeleton_lines(asset, pose, bones, joints);
    if (clip && opt.draw_model_skeleton && has_debug_skeleton(*clip)) {
      std::vector<glm::vec3> mp;
      sample_debug_skeleton(*clip, float(std::clamp(frame, 0, clip->num_frames - 1)), false, mp);
      glm::mat4 xf;
      std::memcpy(&xf[0][0], opt.model_skeleton_transform, sizeof(float) * 16);
      build_model_skeleton_lines(mp, clip->parents, xf, mbones, mjoints);
    }

    OrbitCamera cam;
    cam.yaw = glm::radians(opt.yaw_deg);
    cam.pitch = glm::radians(opt.pitch_deg);
    cam.frame(asset.bounds_min, asset.bounds_max, float(w) / float(h));
    cam.distance *= opt.distance_scale;

    FrameInput in;
    in.view = cam.view();
    in.proj = cam.proj(float(w) / float(h));
    in.cam_pos = cam.eye();
    in.palette = &palette;
    in.bones = &bones;
    in.joints = &joints;
    in.model_bones = &mbones;
    in.model_joints = &mjoints;
    in.draw_skeleton = opt.draw_skeleton;
    in.draw_model_skeleton = opt.draw_model_skeleton;
    in.draw_grid = opt.draw_grid;
    in.point_size = std::max(4.f, h / 120.f);
    grid_for_bounds(asset.bounds_min, asset.bounds_max, in);
    r.update(0, in);

    ctx.submit_now([&](VkCommandBuffer cmd) {
      r.begin_pass(cmd, 0, glm::vec4(0.02f, 0.025f, 0.035f, 1.f));
      r.draw(cmd, 0);
      r.end_pass(cmd);
      VkBufferImageCopy c{};
      c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      c.imageExtent = {ext.width, ext.height, 1};
      vkCmdCopyImageToBuffer(cmd, target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1, &c);
      VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0,
                           nullptr);
    });
    vmaInvalidateAllocation(ctx.vma, readback.alloc, 0, VK_WHOLE_SIZE);
    std::vector<uint8_t> px(size_t(w) * h * 4);
    std::memcpy(px.data(), readback.mapped, px.size());
    for (size_t i = 3; i < px.size(); i += 4) px[i] = 255;
    ok = stbi_write_png(out_png.c_str(), w, h, 4, px.data(), w * 4) != 0;
    if (!ok) err = "cannot write " + out_png;
    ctx.destroy_buffer(readback);
    r.shutdown();
    ctx.destroy_image(target);
  }
  ctx.destroy();
  if (!ok) return fail(err);
  return true;
}

std::vector<JointInfo> load_skeleton(const std::string& asset_glb, std::string* error) {
  Asset a;
  std::string err;
  std::vector<JointInfo> out;
  if (!load_asset(asset_glb, a, err, /*load_images=*/false)) {
    if (error) *error = err;
    return out;
  }
  for (size_t k = 0; k < a.skeleton.size(); ++k) {
    const Node& n = a.nodes[a.skeleton[k]];
    JointInfo j;
    j.name = n.name;
    j.parent = a.skeleton_parent[k];
    for (int i = 0; i < 3; ++i) {
      j.translation[i] = n.t[i];
      j.scale[i] = n.s[i];
    }
    j.rotation[0] = n.r.x;
    j.rotation[1] = n.r.y;
    j.rotation[2] = n.r.z;
    j.rotation[3] = n.r.w;
    out.push_back(std::move(j));
  }
  return out;
}

std::vector<um::MotionClip> load_embedded_clips(const std::string& asset_glb, float fps) {
  Asset a;
  std::string err;
  if (!load_asset(asset_glb, a, err, false, fps)) return {};
  return a.embedded_clips;
}

}  // namespace um::viewer

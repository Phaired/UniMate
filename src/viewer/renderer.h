// Forward renderer: skinned meshes (GPU skinning), ground grid, skeleton
// overlay. Renders into a render pass whose final colour attachments are
// supplied by the caller (swapchain images or an offscreen image).
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "asset.h"
#include "vk_context.h"

namespace um::viewer {

struct LineVertex {
  glm::vec3 pos;
  uint32_t color;  // RGBA8, R in the low byte
};

inline uint32_t rgba8(float r, float g, float b, float a = 1.f) {
  auto c = [](float x) { return uint32_t(glm::clamp(x, 0.f, 1.f) * 255.f + 0.5f); };
  return c(r) | (c(g) << 8) | (c(b) << 16) | (c(a) << 24);
}

struct FrameInput {
  glm::mat4 view{1.f}, proj{1.f};
  glm::vec3 cam_pos{0.f};
  const std::vector<glm::mat4>* palette = nullptr;
  const std::vector<LineVertex>* bones = nullptr;   // line list
  const std::vector<LineVertex>* joints = nullptr;  // point list
  // Second overlay (the clip's own debug skeleton), drawn after the first.
  const std::vector<LineVertex>* model_bones = nullptr;   // line list
  const std::vector<LineVertex>* model_joints = nullptr;  // point list
  bool draw_mesh = true, draw_skeleton = true, draw_grid = true, draw_model_skeleton = true;
  // 3D viewport inside the render target (x, y, w, h in pixels); w or h == 0
  // means the whole target. The whole target is still cleared.
  glm::ivec4 view_rect{0};
  glm::vec3 grid_center{0.f};  // y = ground height
  float grid_half = 10.f, grid_cell = 1.f;
  float point_size = 7.f;
};

// Ground grid under the bind-pose bounds: cell = power of ten ~ size / 10.
inline void grid_for_bounds(const glm::vec3& lo, const glm::vec3& hi, FrameInput& in) {
  glm::vec3 c = 0.5f * (lo + hi);
  float size = std::max(glm::length(hi - lo), 1e-3f);
  in.grid_center = glm::vec3(c.x, lo.y, c.z);
  in.grid_half = size * 3.f;
  in.grid_cell = std::pow(10.f, std::floor(std::log10(size / 4.f)));
}

// Builds the skeleton overlay geometry for a pose.
void build_skeleton_lines(const Asset& a, const Pose& p, std::vector<LineVertex>& bones,
                          std::vector<LineVertex>& joints);

// Overlay geometry for a clip's debug skeleton (positions from
// sample_debug_skeleton, `parents` from the clip), transformed by `xf`.
void build_model_skeleton_lines(const std::vector<glm::vec3>& pos, const std::vector<int>& parents,
                                const glm::mat4& xf, std::vector<LineVertex>& bones,
                                std::vector<LineVertex>& joints);

class Renderer {
 public:
  ~Renderer() { shutdown(); }

  bool init(VkContext* ctx, VkFormat color_format, VkSampleCountFlagBits samples, VkImageLayout final_layout,
            uint32_t frames_in_flight, std::string& err);
  void shutdown();

  // Uploads the asset. The GPU must be idle (or the previous asset unused).
  bool set_asset(const Asset& a, std::string& err);
  void clear_asset();
  bool has_asset() const { return has_asset_; }

  // (Re)creates the MSAA/depth attachments and one framebuffer per output view.
  bool set_targets(VkExtent2D extent, const std::vector<VkImageView>& outputs, std::string& err);

  void update(uint32_t frame, const FrameInput& in);
  void begin_pass(VkCommandBuffer cmd, uint32_t output_index, const glm::vec4& clear_linear);
  void draw(VkCommandBuffer cmd, uint32_t frame);
  void end_pass(VkCommandBuffer cmd) { vkCmdEndRenderPass(cmd); }

  VkRenderPass render_pass() const { return render_pass_; }
  VkSampleCountFlagBits samples() const { return samples_; }
  VkExtent2D extent() const { return extent_; }

 private:
  struct Frame {
    GpuBuffer ubo, palette, lines;
    VkDescriptorSet set = VK_NULL_HANDLE;
    uint32_t bone_count = 0, joint_count = 0, model_bone_count = 0, model_joint_count = 0;
    uint32_t line_cap = 0;               // capacity of `lines`, in vertices
    std::vector<uint32_t> blend_order;   // blended draws, back to front
    FrameInput in;
  };
  struct Texture {
    GpuImage img;
  };

  bool create_pipelines(std::string& err);
  bool upload_texture(const ImageRGBA8& img, GpuImage& out);
  void destroy_targets();
  void write_frame_set(Frame& f);
  void record_mesh_draw(VkCommandBuffer cmd, const Draw& d, int& bound_material);

  VkContext* ctx_ = nullptr;
  VkFormat color_format_ = VK_FORMAT_UNDEFINED;
  VkSampleCountFlagBits samples_ = VK_SAMPLE_COUNT_1_BIT;
  VkImageLayout final_layout_ = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
  VkFormat depth_format_ = VK_FORMAT_D32_SFLOAT;

  VkRenderPass render_pass_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout set0_layout_ = VK_NULL_HANDLE, set1_layout_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_ = VK_NULL_HANDLE;
  VkPipeline mesh_pipe_ = VK_NULL_HANDLE, mesh_blend_pipe_ = VK_NULL_HANDLE, grid_pipe_ = VK_NULL_HANDLE;
  VkPipeline line_pipe_ = VK_NULL_HANDLE, point_pipe_ = VK_NULL_HANDLE;
  VkSampler sampler_ = VK_NULL_HANDLE;
  VkDescriptorPool frame_pool_ = VK_NULL_HANDLE;
  std::vector<Frame> frames_;

  // targets
  VkExtent2D extent_{};
  GpuImage msaa_color_, depth_;
  std::vector<VkFramebuffer> framebuffers_;

  // asset
  bool has_asset_ = false;
  std::vector<Draw> draws_;
  std::vector<Material> materials_;
  GpuBuffer vbuf_, ibuf_;
  std::vector<Texture> textures_;   // asset images + 1 white texture (last)
  VkDescriptorPool material_pool_ = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> material_sets_;
  uint32_t palette_capacity_ = 0, line_capacity_ = 0;  // line_capacity_: minimum per-frame capacity
};

}  // namespace um::viewer

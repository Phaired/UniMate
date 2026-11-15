#include "renderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "grid.frag.h"
#include "grid.vert.h"
#include "line.frag.h"
#include "line.vert.h"
#include "mesh.frag.h"
#include "mesh.vert.h"

namespace um::viewer {
namespace {

struct Globals {
  glm::mat4 view_proj;
  glm::vec4 cam_pos, light_dir, light_color, sky_color, ground_color;
};

struct Push {
  glm::vec4 v0, v1;
  glm::uvec4 u0;
};
static_assert(sizeof(Push) == 48);

VkShaderModule make_module(VkDevice dev, const uint32_t* code, size_t bytes) {
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = bytes;
  ci.pCode = code;
  VkShaderModule m = VK_NULL_HANDLE;
  vkCreateShaderModule(dev, &ci, nullptr, &m);
  return m;
}

struct PipeDesc {
  VkShaderModule vs, fs;
  const VkPipelineVertexInputStateCreateInfo* vertex_input;
  VkPrimitiveTopology topology;
  bool depth_test, depth_write, blend;
  float line_width = 1.f;
};

VkPipeline make_pipeline(VkDevice dev, VkRenderPass rp, VkPipelineLayout layout, VkSampleCountFlagBits samples,
                         const PipeDesc& d) {
  VkPipelineShaderStageCreateInfo st[2]{};
  st[0].sType = st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  st[0].module = d.vs;
  st[0].pName = "main";
  st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  st[1].module = d.fs;
  st[1].pName = "main";
  VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  ia.topology = d.topology;
  VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  vp.viewportCount = 1;
  vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE;  // glTF assets are often double-sided / inconsistently wound
  rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rs.lineWidth = d.line_width;
  VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
  ms.rasterizationSamples = samples;
  VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
  ds.depthTestEnable = d.depth_test;
  ds.depthWriteEnable = d.depth_write;
  ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  VkPipelineColorBlendAttachmentState cba{};
  cba.colorWriteMask = 0xF;
  if (d.blend) {
    cba.blendEnable = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
  }
  VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
  cb.attachmentCount = 1;
  cb.pAttachments = &cba;
  VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
  VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
  dy.dynamicStateCount = 2;
  dy.pDynamicStates = dyn;
  VkPipelineVertexInputStateCreateInfo empty_vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  ci.stageCount = 2;
  ci.pStages = st;
  ci.pVertexInputState = d.vertex_input ? d.vertex_input : &empty_vi;
  ci.pInputAssemblyState = &ia;
  ci.pViewportState = &vp;
  ci.pRasterizationState = &rs;
  ci.pMultisampleState = &ms;
  ci.pDepthStencilState = &ds;
  ci.pColorBlendState = &cb;
  ci.pDynamicState = &dy;
  ci.layout = layout;
  ci.renderPass = rp;
  ci.subpass = 0;
  VkPipeline p = VK_NULL_HANDLE;
  vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &p);
  return p;
}

}  // namespace

void build_skeleton_lines(const Asset& a, const Pose& p, std::vector<LineVertex>& bones,
                          std::vector<LineVertex>& joints) {
  bones.clear();
  joints.clear();
  const uint32_t bone_col = rgba8(1.0f, 0.75f, 0.2f);
  const uint32_t joint_col = rgba8(0.2f, 0.85f, 1.0f);
  const uint32_t root_col = rgba8(1.0f, 0.2f, 0.2f);
  for (size_t k = 0; k < a.skeleton.size(); ++k) {
    glm::vec3 pos = glm::vec3(p.global[a.skeleton[k]][3]);
    int par = a.skeleton_parent[k];
    joints.push_back({pos, par < 0 ? root_col : joint_col});
    if (par >= 0) {
      bones.push_back({glm::vec3(p.global[a.skeleton[par]][3]), bone_col});
      bones.push_back({pos, bone_col});
    }
  }
}

void build_model_skeleton_lines(const std::vector<glm::vec3>& pos, const std::vector<int>& parents,
                                const glm::mat4& xf, std::vector<LineVertex>& bones,
                                std::vector<LineVertex>& joints) {
  bones.clear();
  joints.clear();
  const uint32_t bone_col = rgba8(0.35f, 1.0f, 0.35f);   // green: distinct from the skin overlay
  const uint32_t joint_col = rgba8(0.95f, 0.3f, 1.0f);   // magenta
  const size_t n = std::min(pos.size(), parents.size());
  std::vector<glm::vec3> w(n);
  for (size_t k = 0; k < n; ++k) w[k] = glm::vec3(xf * glm::vec4(pos[k], 1.f));
  for (size_t k = 0; k < n; ++k) {
    joints.push_back({w[k], joint_col});
    const int par = parents[k];
    if (par >= 0 && size_t(par) < n && size_t(par) != k) {
      bones.push_back({w[par], bone_col});
      bones.push_back({w[k], bone_col});
    }
  }
}

bool Renderer::init(VkContext* ctx, VkFormat color_format, VkSampleCountFlagBits samples, VkImageLayout final_layout,
                    uint32_t frames_in_flight, std::string& err) {
  ctx_ = ctx;
  color_format_ = color_format;
  samples_ = samples;
  final_layout_ = final_layout;
  VkDevice dev = ctx->device;

  // ---- render pass ----
  const bool msaa = samples_ != VK_SAMPLE_COUNT_1_BIT;
  std::vector<VkAttachmentDescription> att;
  VkAttachmentDescription color{};
  color.format = color_format_;
  color.samples = samples_;
  color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = msaa ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
  color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  color.finalLayout = msaa ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : final_layout_;
  att.push_back(color);
  VkAttachmentDescription depth = color;
  depth.format = depth_format_;
  depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  att.push_back(depth);
  if (msaa) {
    VkAttachmentDescription resolve = color;
    resolve.samples = VK_SAMPLE_COUNT_1_BIT;
    resolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    resolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    resolve.finalLayout = final_layout_;
    att.push_back(resolve);
  }
  VkAttachmentReference cref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkAttachmentReference dref{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
  VkAttachmentReference rref{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
  VkSubpassDescription sub{};
  sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  sub.colorAttachmentCount = 1;
  sub.pColorAttachments = &cref;
  sub.pDepthStencilAttachment = &dref;
  sub.pResolveAttachments = msaa ? &rref : nullptr;
  VkSubpassDependency deps[2]{};
  deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
  deps[0].dstSubpass = 0;
  deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
  deps[0].dstStageMask = deps[0].srcStageMask;
  deps[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
  deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
  deps[1].srcSubpass = 0;
  deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  deps[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
  deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  deps[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
  rpci.attachmentCount = uint32_t(att.size());
  rpci.pAttachments = att.data();
  rpci.subpassCount = 1;
  rpci.pSubpasses = &sub;
  rpci.dependencyCount = 2;
  rpci.pDependencies = deps;
  if (vkCreateRenderPass(dev, &rpci, nullptr, &render_pass_) != VK_SUCCESS) {
    err = "vkCreateRenderPass failed";
    return false;
  }

  // ---- layouts ----
  VkDescriptorSetLayoutBinding b0[2]{};
  b0[0].binding = 0;
  b0[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  b0[0].descriptorCount = 1;
  b0[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  b0[1].binding = 1;
  b0[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  b0[1].descriptorCount = 1;
  b0[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  lci.bindingCount = 2;
  lci.pBindings = b0;
  vkCreateDescriptorSetLayout(dev, &lci, nullptr, &set0_layout_);
  VkDescriptorSetLayoutBinding b1{};
  b1.binding = 0;
  b1.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  b1.descriptorCount = 1;
  b1.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  lci.bindingCount = 1;
  lci.pBindings = &b1;
  vkCreateDescriptorSetLayout(dev, &lci, nullptr, &set1_layout_);
  VkDescriptorSetLayout sets[2] = {set0_layout_, set1_layout_};
  VkPushConstantRange pcr{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Push)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 2;
  plci.pSetLayouts = sets;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pcr;
  vkCreatePipelineLayout(dev, &plci, nullptr, &layout_);

  if (!create_pipelines(err)) return false;

  VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
  sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  sci.maxLod = VK_LOD_CLAMP_NONE;
  if (ctx_->enabled.samplerAnisotropy) {
    sci.anisotropyEnable = VK_TRUE;
    sci.maxAnisotropy = std::min(8.f, ctx_->props.limits.maxSamplerAnisotropy);
  }
  vkCreateSampler(dev, &sci, nullptr, &sampler_);

  // ---- per-frame resources ----
  VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, frames_in_flight},
                                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, frames_in_flight}};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = frames_in_flight;
  dpci.poolSizeCount = 2;
  dpci.pPoolSizes = ps;
  vkCreateDescriptorPool(dev, &dpci, nullptr, &frame_pool_);
  frames_.resize(frames_in_flight);
  palette_capacity_ = 256;
  line_capacity_ = 4096;
  for (Frame& f : frames_) {
    if (!ctx_->create_buffer(sizeof(Globals), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true, f.ubo) ||
        !ctx_->create_buffer(palette_capacity_ * sizeof(glm::mat4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, f.palette) ||
        !ctx_->create_buffer(line_capacity_ * sizeof(LineVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, f.lines)) {
      err = "buffer allocation failed";
      return false;
    }
    f.line_cap = line_capacity_;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = frame_pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &set0_layout_;
    vkAllocateDescriptorSets(dev, &ai, &f.set);
    write_frame_set(f);
  }
  return true;
}

void Renderer::write_frame_set(Frame& f) {
  VkDescriptorBufferInfo ub{f.ubo.buffer, 0, sizeof(Globals)};
  VkDescriptorBufferInfo sb{f.palette.buffer, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSet w[2]{};
  w[0].sType = w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  w[0].dstSet = w[1].dstSet = f.set;
  w[0].dstBinding = 0;
  w[0].descriptorCount = 1;
  w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  w[0].pBufferInfo = &ub;
  w[1].dstBinding = 1;
  w[1].descriptorCount = 1;
  w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  w[1].pBufferInfo = &sb;
  vkUpdateDescriptorSets(ctx_->device, 2, w, 0, nullptr);
}

bool Renderer::create_pipelines(std::string& err) {
  VkDevice dev = ctx_->device;
  VkShaderModule mv = make_module(dev, spv_mesh_vert, sizeof(spv_mesh_vert));
  VkShaderModule mf = make_module(dev, spv_mesh_frag, sizeof(spv_mesh_frag));
  VkShaderModule lv = make_module(dev, spv_line_vert, sizeof(spv_line_vert));
  VkShaderModule lf = make_module(dev, spv_line_frag, sizeof(spv_line_frag));
  VkShaderModule gv = make_module(dev, spv_grid_vert, sizeof(spv_grid_vert));
  VkShaderModule gf = make_module(dev, spv_grid_frag, sizeof(spv_grid_frag));

  VkVertexInputBindingDescription mb{0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription ma[5] = {
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
      {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
      {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
      {3, 0, VK_FORMAT_R32G32B32A32_UINT, offsetof(Vertex, joints)},
      {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, weights)},
  };
  VkPipelineVertexInputStateCreateInfo mvi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  mvi.vertexBindingDescriptionCount = 1;
  mvi.pVertexBindingDescriptions = &mb;
  mvi.vertexAttributeDescriptionCount = 5;
  mvi.pVertexAttributeDescriptions = ma;

  VkVertexInputBindingDescription lb{0, sizeof(LineVertex), VK_VERTEX_INPUT_RATE_VERTEX};
  VkVertexInputAttributeDescription la[2] = {
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LineVertex, pos)},
      {1, 0, VK_FORMAT_R8G8B8A8_UNORM, offsetof(LineVertex, color)},
  };
  VkPipelineVertexInputStateCreateInfo lvi = mvi;
  lvi.pVertexBindingDescriptions = &lb;
  lvi.vertexAttributeDescriptionCount = 2;
  lvi.pVertexAttributeDescriptions = la;

  const float lw = ctx_->enabled.wideLines ? std::min(2.f, ctx_->props.limits.lineWidthRange[1]) : 1.f;
  mesh_pipe_ = make_pipeline(dev, render_pass_, layout_, samples_,
                             {mv, mf, &mvi, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true, true, false});
  mesh_blend_pipe_ = make_pipeline(dev, render_pass_, layout_, samples_,
                                   {mv, mf, &mvi, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true, false, true});
  grid_pipe_ = make_pipeline(dev, render_pass_, layout_, samples_,
                             {gv, gf, nullptr, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true, false, true});
  line_pipe_ = make_pipeline(dev, render_pass_, layout_, samples_,
                             {lv, lf, &lvi, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, false, false, true, lw});
  point_pipe_ = make_pipeline(dev, render_pass_, layout_, samples_,
                              {lv, lf, &lvi, VK_PRIMITIVE_TOPOLOGY_POINT_LIST, false, false, true});
  for (VkShaderModule m : {mv, mf, lv, lf, gv, gf}) vkDestroyShaderModule(dev, m, nullptr);
  if (!mesh_pipe_ || !mesh_blend_pipe_ || !grid_pipe_ || !line_pipe_ || !point_pipe_) {
    err = "pipeline creation failed";
    return false;
  }
  return true;
}

void Renderer::shutdown() {
  if (!ctx_ || !ctx_->device) return;
  VkDevice dev = ctx_->device;
  vkDeviceWaitIdle(dev);
  clear_asset();
  destroy_targets();
  for (Frame& f : frames_) {
    ctx_->destroy_buffer(f.ubo);
    ctx_->destroy_buffer(f.palette);
    ctx_->destroy_buffer(f.lines);
  }
  frames_.clear();
  if (frame_pool_) vkDestroyDescriptorPool(dev, frame_pool_, nullptr);
  if (sampler_) vkDestroySampler(dev, sampler_, nullptr);
  for (VkPipeline p : {mesh_pipe_, mesh_blend_pipe_, grid_pipe_, line_pipe_, point_pipe_})
    if (p) vkDestroyPipeline(dev, p, nullptr);
  if (layout_) vkDestroyPipelineLayout(dev, layout_, nullptr);
  if (set0_layout_) vkDestroyDescriptorSetLayout(dev, set0_layout_, nullptr);
  if (set1_layout_) vkDestroyDescriptorSetLayout(dev, set1_layout_, nullptr);
  if (render_pass_) vkDestroyRenderPass(dev, render_pass_, nullptr);
  frame_pool_ = VK_NULL_HANDLE;
  sampler_ = VK_NULL_HANDLE;
  mesh_pipe_ = mesh_blend_pipe_ = grid_pipe_ = line_pipe_ = point_pipe_ = VK_NULL_HANDLE;
  layout_ = VK_NULL_HANDLE;
  set0_layout_ = set1_layout_ = VK_NULL_HANDLE;
  render_pass_ = VK_NULL_HANDLE;
  ctx_ = nullptr;
}

bool Renderer::upload_texture(const ImageRGBA8& img, GpuImage& out) {
  const VkFormat fmt = VK_FORMAT_R8G8B8A8_SRGB;
  VkFormatProperties fp;
  vkGetPhysicalDeviceFormatProperties(ctx_->phys, fmt, &fp);
  const bool can_blit = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) &&
                        (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) &&
                        (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT);
  uint32_t mips = 1;
  if (can_blit)
    for (int s = std::max(img.width, img.height); s > 1; s >>= 1) ++mips;
  VkExtent2D ext{uint32_t(img.width), uint32_t(img.height)};
  if (!ctx_->create_image(ext, fmt,
                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                          VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_ASPECT_COLOR_BIT, mips, out))
    return false;
  GpuBuffer staging;
  if (!ctx_->create_buffer(img.pixels.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, staging)) return false;
  std::memcpy(staging.mapped, img.pixels.data(), img.pixels.size());
  vmaFlushAllocation(ctx_->vma, staging.alloc, 0, VK_WHOLE_SIZE);
  ctx_->submit_now([&](VkCommandBuffer cmd) {
    image_barrier(cmd, out.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {ext.width, ext.height, 1};
    vkCmdCopyBufferToImage(cmd, staging.buffer, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    int w = img.width, h = img.height;
    for (uint32_t m = 1; m < mips; ++m) {
      image_barrier(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, m - 1, 1);
      int nw = std::max(1, w / 2), nh = std::max(1, h / 2);
      VkImageBlit b{};
      b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m - 1, 0, 1};
      b.srcOffsets[1] = {w, h, 1};
      b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 1};
      b.dstOffsets[1] = {nw, nh, 1};
      vkCmdBlitImage(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.image,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
      image_barrier(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, m - 1, 1);
      w = nw;
      h = nh;
    }
    image_barrier(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                  VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT, mips - 1, 1);
  });
  ctx_->destroy_buffer(staging);
  return true;
}

void Renderer::clear_asset() {
  if (!ctx_) return;
  VkDevice dev = ctx_->device;
  for (Texture& t : textures_) ctx_->destroy_image(t.img);
  textures_.clear();
  if (material_pool_) vkDestroyDescriptorPool(dev, material_pool_, nullptr);
  material_pool_ = VK_NULL_HANDLE;
  material_sets_.clear();
  ctx_->destroy_buffer(vbuf_);
  ctx_->destroy_buffer(ibuf_);
  draws_.clear();
  materials_.clear();
  has_asset_ = false;
}

bool Renderer::set_asset(const Asset& a, std::string& err) {
  vkDeviceWaitIdle(ctx_->device);
  clear_asset();
  VkDevice dev = ctx_->device;

  // geometry (host-visible is plenty fast for a viewer and avoids staging)
  auto upload = [&](const void* data, size_t bytes, VkBufferUsageFlags usage, GpuBuffer& b) {
    GpuBuffer staging;
    if (!ctx_->create_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true, staging)) return false;
    std::memcpy(staging.mapped, data, bytes);
    vmaFlushAllocation(ctx_->vma, staging.alloc, 0, VK_WHOLE_SIZE);
    if (!ctx_->create_buffer(bytes, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false, b)) return false;
    ctx_->submit_now([&](VkCommandBuffer cmd) {
      VkBufferCopy c{0, 0, bytes};
      vkCmdCopyBuffer(cmd, staging.buffer, b.buffer, 1, &c);
    });
    ctx_->destroy_buffer(staging);
    return true;
  };
  if (!a.vertices.empty()) {
    if (!upload(a.vertices.data(), a.vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vbuf_) ||
        !upload(a.indices.data(), a.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, ibuf_)) {
      err = "geometry upload failed";
      return false;
    }
  }

  // textures (+ a 1x1 white fallback, last)
  for (const ImageRGBA8& img : a.images) {
    textures_.push_back({});
    if (!upload_texture(img, textures_.back().img)) {
      err = "texture upload failed";
      return false;
    }
  }
  ImageRGBA8 white{1, 1, {255, 255, 255, 255}};
  textures_.push_back({});
  if (!upload_texture(white, textures_.back().img)) {
    err = "texture upload failed";
    return false;
  }

  // material descriptor sets
  materials_ = a.materials;
  draws_ = a.draws;
  const uint32_t nm = uint32_t(materials_.size());
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nm};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = nm;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &ps;
  vkCreateDescriptorPool(dev, &dpci, nullptr, &material_pool_);
  material_sets_.resize(nm);
  std::vector<VkDescriptorSetLayout> layouts(nm, set1_layout_);
  VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  ai.descriptorPool = material_pool_;
  ai.descriptorSetCount = nm;
  ai.pSetLayouts = layouts.data();
  if (vkAllocateDescriptorSets(dev, &ai, material_sets_.data()) != VK_SUCCESS) {
    err = "descriptor allocation failed";
    return false;
  }
  for (uint32_t m = 0; m < nm; ++m) {
    int ti = materials_[m].image >= 0 ? materials_[m].image : int(textures_.size()) - 1;
    VkDescriptorImageInfo ii{sampler_, textures_[ti].img.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = material_sets_[m];
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
  }

  // grow per-frame buffers
  const uint32_t need_pal = std::max<uint32_t>(a.palette_size, 1);
  const uint32_t need_lines = uint32_t(a.skeleton.size() * 3 + 16);
  if (need_pal > palette_capacity_ || need_lines > line_capacity_) {
    palette_capacity_ = std::max(palette_capacity_, need_pal);
    line_capacity_ = std::max(line_capacity_, need_lines);
    for (Frame& f : frames_) {
      ctx_->destroy_buffer(f.palette);
      ctx_->destroy_buffer(f.lines);
      ctx_->create_buffer(palette_capacity_ * sizeof(glm::mat4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true, f.palette);
      ctx_->create_buffer(line_capacity_ * sizeof(LineVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, f.lines);
      f.line_cap = line_capacity_;
      write_frame_set(f);
    }
  }
  has_asset_ = true;
  return true;
}

void Renderer::destroy_targets() {
  if (!ctx_) return;
  for (VkFramebuffer fb : framebuffers_) vkDestroyFramebuffer(ctx_->device, fb, nullptr);
  framebuffers_.clear();
  ctx_->destroy_image(msaa_color_);
  ctx_->destroy_image(depth_);
}

bool Renderer::set_targets(VkExtent2D extent, const std::vector<VkImageView>& outputs, std::string& err) {
  destroy_targets();
  extent_ = extent;
  const bool msaa = samples_ != VK_SAMPLE_COUNT_1_BIT;
  if (msaa && !ctx_->create_image(extent, color_format_,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, samples_,
                                  VK_IMAGE_ASPECT_COLOR_BIT, 1, msaa_color_)) {
    err = "MSAA colour target allocation failed";
    return false;
  }
  if (!ctx_->create_image(extent, depth_format_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, samples_,
                          VK_IMAGE_ASPECT_DEPTH_BIT, 1, depth_)) {
    err = "depth target allocation failed";
    return false;
  }
  for (VkImageView out : outputs) {
    std::array<VkImageView, 3> views;
    uint32_t n;
    if (msaa) {
      views = {msaa_color_.view, depth_.view, out};
      n = 3;
    } else {
      views = {out, depth_.view, VK_NULL_HANDLE};
      n = 2;
    }
    VkFramebufferCreateInfo fci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fci.renderPass = render_pass_;
    fci.attachmentCount = n;
    fci.pAttachments = views.data();
    fci.width = extent.width;
    fci.height = extent.height;
    fci.layers = 1;
    VkFramebuffer fb;
    if (vkCreateFramebuffer(ctx_->device, &fci, nullptr, &fb) != VK_SUCCESS) {
      err = "framebuffer creation failed";
      return false;
    }
    framebuffers_.push_back(fb);
  }
  return true;
}

void Renderer::update(uint32_t frame, const FrameInput& in) {
  Frame& f = frames_[frame];
  f.in = in;
  Globals g;
  g.view_proj = in.proj * in.view;
  g.cam_pos = glm::vec4(in.cam_pos, 1.f);
  g.light_dir = glm::vec4(glm::normalize(glm::vec3(-0.35f, -1.0f, -0.55f)), 0.f);
  g.light_color = glm::vec4(1.45f, 1.38f, 1.28f, 0.f);
  g.sky_color = glm::vec4(0.36f, 0.40f, 0.48f, 0.f);
  g.ground_color = glm::vec4(0.13f, 0.11f, 0.10f, 0.f);
  std::memcpy(f.ubo.mapped, &g, sizeof(g));
  vmaFlushAllocation(ctx_->vma, f.ubo.alloc, 0, VK_WHOLE_SIZE);
  if (in.palette && !in.palette->empty()) {
    size_t n = std::min<size_t>(in.palette->size(), palette_capacity_);
    std::memcpy(f.palette.mapped, in.palette->data(), n * sizeof(glm::mat4));
    vmaFlushAllocation(ctx_->vma, f.palette.alloc, 0, VK_WHOLE_SIZE);
  }
  // overlay lines: [bones][joints][model bones][model joints]
  auto sz = [](const std::vector<LineVertex>* v) { return v ? v->size() : size_t(0); };
  const size_t need = sz(in.bones) + sz(in.joints) + sz(in.model_bones) + sz(in.model_joints);
  if (need > f.line_cap) {
    // this frame's previous use has completed (the caller waited on its fence)
    const uint32_t cap = uint32_t(std::max<size_t>(need + need / 2, line_capacity_));
    GpuBuffer nb;
    if (ctx_->create_buffer(cap * sizeof(LineVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, nb)) {
      ctx_->destroy_buffer(f.lines);
      f.lines = nb;
      f.line_cap = cap;
    }
  }
  f.bone_count = f.joint_count = f.model_bone_count = f.model_joint_count = 0;
  auto* dst = static_cast<LineVertex*>(f.lines.mapped);
  uint32_t used = 0;
  auto put = [&](const std::vector<LineVertex>* v, bool pairs) -> uint32_t {
    if (!v) return 0;
    uint32_t n = uint32_t(std::min<size_t>(v->size(), f.line_cap - used));
    if (pairs) n &= ~1u;
    if (n) std::memcpy(dst + used, v->data(), n * sizeof(LineVertex));
    used += n;
    return n;
  };
  f.bone_count = put(in.bones, true);
  f.joint_count = put(in.joints, false);
  f.model_bone_count = put(in.model_bones, true);
  f.model_joint_count = put(in.model_joints, false);
  vmaFlushAllocation(ctx_->vma, f.lines.alloc, 0, VK_WHOLE_SIZE);

  // blended draws, back to front by view depth of their (posed) centroid
  f.blend_order.clear();
  if (has_asset_) {
    std::vector<std::pair<float, uint32_t>> keyed;
    const glm::vec3 fwd = -glm::vec3(in.view[0][2], in.view[1][2], in.view[2][2]);  // camera forward (world)
    for (uint32_t i = 0; i < uint32_t(draws_.size()); ++i) {
      const Draw& d = draws_[i];
      const int m = d.material >= 0 ? d.material : int(materials_.size()) - 1;
      if (!materials_[m].blend) continue;
      glm::vec3 c = d.center;
      const size_t pi = size_t(d.palette_base) + d.center_joint;
      if (in.palette && pi < in.palette->size()) c = glm::vec3((*in.palette)[pi] * glm::vec4(c, 1.f));
      keyed.push_back({glm::dot(c - in.cam_pos, fwd), i});
    }
    std::stable_sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (auto& k : keyed) f.blend_order.push_back(k.second);
  }
}

void Renderer::record_mesh_draw(VkCommandBuffer cmd, const Draw& d, int& bound) {
  const VkShaderStageFlags pcs = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  const int m = d.material >= 0 ? d.material : int(materials_.size()) - 1;
  if (m != bound) {
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 1, 1, &material_sets_[m], 0, nullptr);
    bound = m;
  }
  Push p{};
  p.v0 = materials_[m].base_color;
  p.v1 = glm::vec4(materials_[m].alpha_cutoff, materials_[m].image >= 0 ? 1.f : 0.f, materials_[m].blend ? 1.f : 0.f, 0);
  p.u0 = glm::uvec4(d.palette_base, 0, 0, 0);
  vkCmdPushConstants(cmd, layout_, pcs, 0, sizeof(p), &p);
  vkCmdDrawIndexed(cmd, d.index_count, 1, d.first_index, d.vertex_offset, 0);
}

void Renderer::begin_pass(VkCommandBuffer cmd, uint32_t output_index, const glm::vec4& clear) {
  VkClearValue cv[3]{};
  cv[0].color = {{clear.r, clear.g, clear.b, clear.a}};
  cv[1].depthStencil = {1.f, 0};
  cv[2].color = cv[0].color;
  VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  bi.renderPass = render_pass_;
  bi.framebuffer = framebuffers_[output_index];
  bi.renderArea = {{0, 0}, extent_};
  bi.clearValueCount = samples_ != VK_SAMPLE_COUNT_1_BIT ? 3 : 2;
  bi.pClearValues = cv;
  vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
  VkViewport vp{0, 0, float(extent_.width), float(extent_.height), 0.f, 1.f};
  VkRect2D sc{{0, 0}, extent_};
  vkCmdSetViewport(cmd, 0, 1, &vp);
  vkCmdSetScissor(cmd, 0, 1, &sc);
}

void Renderer::draw(VkCommandBuffer cmd, uint32_t frame) {
  Frame& f = frames_[frame];
  const FrameInput& in = f.in;
  const VkShaderStageFlags pcs = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &f.set, 0, nullptr);
  if (in.view_rect.z > 0 && in.view_rect.w > 0) {
    VkViewport vp{float(in.view_rect.x), float(in.view_rect.y), float(in.view_rect.z), float(in.view_rect.w), 0.f, 1.f};
    VkRect2D sc{{in.view_rect.x, in.view_rect.y}, {uint32_t(in.view_rect.z), uint32_t(in.view_rect.w)}};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);
  }

  const bool meshes = has_asset_ && in.draw_mesh && vbuf_.buffer && !draws_.empty();
  if (meshes) {  // opaque (and alpha-masked) first
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_pipe_);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf_.buffer, &off);
    vkCmdBindIndexBuffer(cmd, ibuf_.buffer, 0, VK_INDEX_TYPE_UINT32);
    int bound = -1;
    for (const Draw& d : draws_) {
      const int m = d.material >= 0 ? d.material : int(materials_.size()) - 1;
      if (!materials_[m].blend) record_mesh_draw(cmd, d, bound);
    }
  }
  if (in.draw_grid) {
    if (!material_sets_.empty())
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 1, 1, &material_sets_.back(), 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, grid_pipe_);
    Push p{};
    p.v0 = glm::vec4(in.grid_cell, 5.f, 0, 0);
    p.v1 = glm::vec4(in.grid_center.x, in.grid_center.y, in.grid_center.z, in.grid_half);
    vkCmdPushConstants(cmd, layout_, pcs, 0, sizeof(p), &p);
    vkCmdDraw(cmd, 6, 1, 0, 0);
  }
  if (meshes && !f.blend_order.empty()) {  // alpha-blended, back to front, no depth write
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, mesh_blend_pipe_);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf_.buffer, &off);
    vkCmdBindIndexBuffer(cmd, ibuf_.buffer, 0, VK_INDEX_TYPE_UINT32);
    int bound = -1;
    for (uint32_t i : f.blend_order) record_mesh_draw(cmd, draws_[i], bound);
  }
  const bool skel = in.draw_skeleton && (f.bone_count || f.joint_count);
  const bool mskel = in.draw_model_skeleton && (f.model_bone_count || f.model_joint_count);
  if (skel || mskel) {
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &f.lines.buffer, &off);
    const float ps = ctx_->enabled.largePoints ? in.point_size : 1.f;
    auto lines = [&](uint32_t first, uint32_t count) {
      if (!count) return;
      Push p{};
      p.v0 = glm::vec4(ps, 0.f, 0, 0);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, line_pipe_);
      vkCmdPushConstants(cmd, layout_, pcs, 0, sizeof(p), &p);
      vkCmdDraw(cmd, count, 1, first, 0);
    };
    auto points = [&](uint32_t first, uint32_t count, float size) {
      if (!count) return;
      Push p{};
      p.v0 = glm::vec4(size, 1.f, 0, 0);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, point_pipe_);
      vkCmdPushConstants(cmd, layout_, pcs, 0, sizeof(p), &p);
      vkCmdDraw(cmd, count, 1, first, 0);
    };
    const uint32_t mb = f.bone_count + f.joint_count;
    if (skel) {
      lines(0, f.bone_count);
      points(f.bone_count, f.joint_count, ps);
    }
    if (mskel) {
      lines(mb, f.model_bone_count);
      points(mb + f.model_bone_count, f.model_joint_count, std::max(1.f, ps * 0.6f));
    }
  }
}

}  // namespace um::viewer

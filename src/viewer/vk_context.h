// Minimal Vulkan device context for the viewer (volk loader + VMA).
#pragma once

#include <functional>
#include <string>
#include <vector>

#include <volk.h>
#ifndef VMA_STATIC_VULKAN_FUNCTIONS
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#endif
#include <vk_mem_alloc.h>

namespace um::viewer {

struct GpuBuffer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation alloc = nullptr;
  void* mapped = nullptr;
  VkDeviceSize size = 0;
};

struct GpuImage {
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation alloc = nullptr;
  VkImageView view = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  VkExtent2D extent{};
  uint32_t mips = 1;
};

class VkContext {
 public:
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
  VkQueue queue = VK_NULL_HANDLE;
  VmaAllocator vma = nullptr;
  VkCommandPool cmd_pool = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties props{};
  VkPhysicalDeviceFeatures enabled{};
  uint32_t api_version = 0;

  ~VkContext() { destroy(); }

  // Step 1: loader + instance. `extensions` = required instance extensions
  // (e.g. from glfwGetRequiredInstanceExtensions; empty for headless).
  bool create_instance(const std::vector<const char*>& extensions, std::string& err);
  // Step 2: pick a device (able to present to `surface` if not null).
  bool create_device(VkSurfaceKHR surface, std::string& err);
  void destroy();

  bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool host_visible, GpuBuffer& out);
  void destroy_buffer(GpuBuffer& b);
  bool create_image(VkExtent2D extent, VkFormat fmt, VkImageUsageFlags usage, VkSampleCountFlagBits samples,
                    VkImageAspectFlags aspect, uint32_t mips, GpuImage& out);
  void destroy_image(GpuImage& img);

  // Records into a one-shot command buffer, submits and waits.
  void submit_now(const std::function<void(VkCommandBuffer)>& record);

  VkSampleCountFlagBits clamp_samples(int requested) const;
};

void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                   VkPipelineStageFlags src_stage, VkAccessFlags src_access, VkPipelineStageFlags dst_stage,
                   VkAccessFlags dst_access, VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
                   uint32_t base_mip = 0, uint32_t mip_count = VK_REMAINING_MIP_LEVELS);

}  // namespace um::viewer

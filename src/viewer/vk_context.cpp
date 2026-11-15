#include "vk_context.h"

#include <algorithm>
#include <cstring>

namespace um::viewer {

bool VkContext::create_instance(const std::vector<const char*>& extensions, std::string& err) {
  if (volkInitialize() != VK_SUCCESS) {
    err = "Vulkan loader (vulkan-1.dll) not found";
    return false;
  }
  api_version = volkGetInstanceVersion();
  if (api_version < VK_API_VERSION_1_2) {
    err = "Vulkan 1.2 instance required";
    return false;
  }
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "unimate_viewer";
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ci.pApplicationInfo = &app;
  ci.enabledExtensionCount = uint32_t(extensions.size());
  ci.ppEnabledExtensionNames = extensions.data();
  VkResult r = vkCreateInstance(&ci, nullptr, &instance);
  if (r != VK_SUCCESS) {
    err = "vkCreateInstance failed (" + std::to_string(int(r)) + ")";
    return false;
  }
  volkLoadInstance(instance);
  return true;
}

bool VkContext::create_device(VkSurfaceKHR surface, std::string& err) {
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(instance, &n, nullptr);
  std::vector<VkPhysicalDevice> devs(n);
  vkEnumeratePhysicalDevices(instance, &n, devs.data());
  int best_score = -1;
  for (VkPhysicalDevice pd : devs) {
    VkPhysicalDeviceProperties p;
    vkGetPhysicalDeviceProperties(pd, &p);
    if (p.apiVersion < VK_API_VERSION_1_2) continue;
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    for (uint32_t q = 0; q < qn; ++q) {
      if (!(qf[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
      if (surface) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(pd, q, surface, &present);
        if (!present) continue;
      }
      int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
      if (score > best_score) {
        best_score = score;
        phys = pd;
        queue_family = q;
        props = p;
      }
      break;
    }
  }
  if (!phys) {
    err = "no Vulkan 1.2 GPU with a graphics queue";
    return false;
  }
  VkPhysicalDeviceFeatures avail;
  vkGetPhysicalDeviceFeatures(phys, &avail);
  enabled = {};
  enabled.largePoints = avail.largePoints;
  enabled.wideLines = avail.wideLines;
  enabled.samplerAnisotropy = avail.samplerAnisotropy;
  enabled.fillModeNonSolid = avail.fillModeNonSolid;

  float prio = 1.f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = queue_family;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  std::vector<const char*> exts;
  if (surface) exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.enabledExtensionCount = uint32_t(exts.size());
  dci.ppEnabledExtensionNames = exts.data();
  dci.pEnabledFeatures = &enabled;
  VkResult r = vkCreateDevice(phys, &dci, nullptr, &device);
  if (r != VK_SUCCESS) {
    err = "vkCreateDevice failed (" + std::to_string(int(r)) + ")";
    return false;
  }
  volkLoadDevice(device);
  vkGetDeviceQueue(device, queue_family, 0, &queue);

  VmaVulkanFunctions fns{};
  fns.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  fns.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
  VmaAllocatorCreateInfo aci{};
  aci.physicalDevice = phys;
  aci.device = device;
  aci.instance = instance;
  aci.vulkanApiVersion = VK_API_VERSION_1_2;
  aci.pVulkanFunctions = &fns;
  if (vmaCreateAllocator(&aci, &vma) != VK_SUCCESS) {
    err = "vmaCreateAllocator failed";
    return false;
  }
  VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = queue_family;
  vkCreateCommandPool(device, &pci, nullptr, &cmd_pool);
  return true;
}

void VkContext::destroy() {
  if (device) {
    vkDeviceWaitIdle(device);
    if (cmd_pool) vkDestroyCommandPool(device, cmd_pool, nullptr);
    if (vma) vmaDestroyAllocator(vma);
    vkDestroyDevice(device, nullptr);
  }
  if (instance) vkDestroyInstance(instance, nullptr);
  cmd_pool = VK_NULL_HANDLE;
  vma = nullptr;
  device = VK_NULL_HANDLE;
  instance = VK_NULL_HANDLE;
}

bool VkContext::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool host_visible, GpuBuffer& out) {
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = std::max<VkDeviceSize>(size, 16);
  bci.usage = usage;
  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO;
  if (host_visible)
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VmaAllocationInfo info{};
  if (vmaCreateBuffer(vma, &bci, &aci, &out.buffer, &out.alloc, &info) != VK_SUCCESS) return false;
  out.mapped = info.pMappedData;
  out.size = bci.size;
  return true;
}

void VkContext::destroy_buffer(GpuBuffer& b) {
  if (b.buffer) vmaDestroyBuffer(vma, b.buffer, b.alloc);
  b = {};
}

bool VkContext::create_image(VkExtent2D extent, VkFormat fmt, VkImageUsageFlags usage, VkSampleCountFlagBits samples,
                             VkImageAspectFlags aspect, uint32_t mips, GpuImage& out) {
  VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = fmt;
  ici.extent = {extent.width, extent.height, 1};
  ici.mipLevels = mips;
  ici.arrayLayers = 1;
  ici.samples = samples;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = usage;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  if (vmaCreateImage(vma, &ici, &aci, &out.image, &out.alloc, nullptr) != VK_SUCCESS) return false;
  VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vci.image = out.image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vci.format = fmt;
  vci.subresourceRange = {aspect, 0, mips, 0, 1};
  if (vkCreateImageView(device, &vci, nullptr, &out.view) != VK_SUCCESS) return false;
  out.format = fmt;
  out.extent = extent;
  out.mips = mips;
  return true;
}

void VkContext::destroy_image(GpuImage& img) {
  if (img.view) vkDestroyImageView(device, img.view, nullptr);
  if (img.image) vmaDestroyImage(vma, img.image, img.alloc);
  img = {};
}

void VkContext::submit_now(const std::function<void(VkCommandBuffer)>& record) {
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = cmd_pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers(device, &ai, &cmd);
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &bi);
  record(cmd);
  vkEndCommandBuffer(cmd);
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence;
  vkCreateFence(device, &fci, nullptr, &fence);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  vkQueueSubmit(queue, 1, &si, fence);
  vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
  vkDestroyFence(device, fence, nullptr);
  vkFreeCommandBuffers(device, cmd_pool, 1, &cmd);
}

VkSampleCountFlagBits VkContext::clamp_samples(int requested) const {
  VkSampleCountFlags ok = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
  for (int s = 8; s > 1; s >>= 1)
    if (s <= requested && (ok & s)) return VkSampleCountFlagBits(s);
  return VK_SAMPLE_COUNT_1_BIT;
}

void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                   VkPipelineStageFlags src_stage, VkAccessFlags src_access, VkPipelineStageFlags dst_stage,
                   VkAccessFlags dst_access, VkImageAspectFlags aspect, uint32_t base_mip, uint32_t mip_count) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = from;
  b.newLayout = to;
  b.srcAccessMask = src_access;
  b.dstAccessMask = dst_access;
  b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = image;
  b.subresourceRange = {aspect, base_mip, mip_count, 0, 1};
  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

}  // namespace um::viewer

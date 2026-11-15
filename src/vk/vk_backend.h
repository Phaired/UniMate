// Vulkan compute implementation of um::Backend.
//
//   #include "vk/vk_backend.h"
//   if (um::vulkan_available()) be = um::make_vulkan_backend();
//   else                        be = um::make_cpu_backend();
//
// Execution is deferred: ops record compute dispatches into command buffers
// that are submitted in batches; only to_host() / flush() (and running out of
// staging space) make the host wait. Intermediate buffers come from a
// size-class pool and are recycled as soon as the last Tensor referencing them
// dies (GPU-side reuse is ordered by pipeline barriers, so this is safe even
// while earlier commands are still in flight).
//
// Like every um::Backend: not thread-safe, use from one thread at a time.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/tensor.h"

namespace um {

struct VulkanBackendOptions {
  // Physical device index as listed by vulkan_devices(); -1 = automatic
  // (first discrete GPU if prefer_discrete, otherwise the first usable one).
  int device_index = -1;
  bool prefer_discrete = true;
  // Experimental: matmul/linear (M > 8) on tensor cores via
  // VK_KHR_cooperative_matrix -- operands rounded to fp16, fp32 accumulation
  // (~3e-4 normwise error instead of ~1e-6). OFF by default; silently falls
  // back to fp32 when unsupported. Not faster than the fp32 kernel on GeForce
  // Ada (fp32-accumulate tensor rate == fp32 FMA rate), so leave it off.
  bool fp16 = false;
  // Print device selection / pipeline creation / pool growth to stderr.
  bool verbose = false;
  // Enable VK_LAYER_KHRONOS_validation if it is installed (also enabled by
  // the environment variable UM_VK_VALIDATION=1). Silently ignored otherwise.
  bool validation = false;
  // Submit the current command buffer after this many recorded dispatches
  // (lets the GPU start while the host keeps recording).
  int dispatches_per_submit = 96;
  // Per-in-flight-slot staging arena for host->device uploads (bytes).
  uint64_t staging_bytes = 32ull << 20;
  // Debug: do not use VK_KHR_push_descriptor even if available.
  bool disable_push_descriptors = false;
};

struct VulkanDeviceInfo {
  int index = -1;
  std::string name;
  uint32_t vendor_id = 0, device_id = 0;
  bool discrete = false;
  bool usable = false;           // Vulkan >= 1.2 + compute queue + subgroup ops
  uint64_t device_local_bytes = 0;
};

struct VulkanMemoryStats {
  uint64_t device_bytes = 0;       // all device buffers currently allocated (pool incl.)
  uint64_t peak_device_bytes = 0;  // high-water mark of device_bytes
  uint64_t live_bytes = 0;         // bytes held by live tensors
  uint64_t peak_live_bytes = 0;
  uint64_t pooled_free_bytes = 0;  // allocated but idle in the recycler
  uint64_t staging_bytes = 0;      // host-visible staging + readback
};

struct VulkanCounters {
  uint64_t dispatches = 0, submits = 0, barriers = 0, waits = 0;
  uint64_t uploads_bytes = 0, downloads_bytes = 0;
  uint64_t pipelines = 0;
};

// True if at least one usable Vulkan device exists (never throws).
bool vulkan_available();
std::vector<VulkanDeviceInfo> vulkan_devices();

// Throws std::runtime_error if no usable device exists.
std::unique_ptr<Backend> make_vulkan_backend(const VulkanBackendOptions& opts = {});

// Introspection of a backend made by make_vulkan_backend(). For any other
// backend: is_vulkan_backend() is false, the others return empty values.
bool is_vulkan_backend(const Backend& be);
std::string vulkan_device_name(const Backend& be);
VulkanMemoryStats vulkan_memory_stats(const Backend& be);
VulkanCounters vulkan_counters(const Backend& be);
// Wait for the GPU and release all idle pooled buffers back to the driver.
void vulkan_trim(Backend& be);
// Submit and wait for all recorded work (like flush(), but blocking).
void vulkan_finish(Backend& be);

}  // namespace um

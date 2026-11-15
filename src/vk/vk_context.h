// Private: Vulkan device, deferred command recording, buffer pool, pipelines.
#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include "vk/vk_backend.h"

namespace um::vkx {

// One entry per shader in shaders/. Order must match kKernelInfo in vk_context.cpp.
enum class K : uint32_t {
  Unary, Binary, Copy, GatherRows, Softmax, Norm, ReduceRows, ReduceCols,
  Gemm, Gemv, Ternary, Attention, GemmCoop, Count
};

struct Buffer {
  VkBuffer buf = VK_NULL_HANDLE;
  VmaAllocation alloc = nullptr;
  uint64_t size = 0;  // allocated (size-class) bytes
  void* mapped = nullptr;
};

// A storage-buffer binding (offset/range in bytes); `write` drives hazard
// tracking, which is per VkBuffer (conservative for sub-ranges).
struct Bind {
  VkBuffer buf;
  bool write;
  VkDeviceSize offset = 0;
  VkDeviceSize range = VK_WHOLE_SIZE;
};

using Spec = std::vector<uint32_t>;  // specialization constants, ids 0..n-1

struct Limits {
  uint32_t max_groups[3] = {65535, 65535, 65535};
  uint32_t max_wg_invocations = 1024;
  uint32_t max_shared_bytes = 49152;
  uint32_t subgroup_size = 32;
  uint64_t max_storage_range = 1ull << 31;
  uint64_t min_storage_align = 256;  // minStorageBufferOffsetAlignment
  uint32_t sm_count_hint = 64;  // rough number of compute units (for tile choice)
};

class Context {
 public:
  explicit Context(const VulkanBackendOptions& opts);
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  const std::string& device_name() const { return device_name_; }
  const Limits& limits() const { return limits_; }
  bool verbose() const { return opts_.verbose; }
  // fp16 tensor-core GEMM usable (options.fp16 and device support).
  bool coop() const { return coop_; }

  // ---- memory -----------------------------------------------------------
  Buffer acquire(uint64_t bytes);      // device-local, from the pool
  void release(const Buffer& b);       // back to the pool (safe immediately)
  const Buffer& dummy() const { return dummy_; }

  // ---- transfers (recorded in-stream) ------------------------------------
  void upload(const Buffer& dst, const void* src, uint64_t bytes);
  void download(VkBuffer src, uint64_t src_offset, void* dst, uint64_t bytes);  // waits

  // ---- compute -----------------------------------------------------------
  void dispatch(K k, const Spec& spec, std::initializer_list<Bind> binds,
                const void* pc, uint32_t pc_bytes, uint32_t gx, uint32_t gy = 1,
                uint32_t gz = 1);

  // The next dispatch writes `b` in a region disjoint from every write to `b`
  // since the last barrier (concat pieces): no WAW barrier needed for it.
  void allow_disjoint_write(VkBuffer b) { disjoint_ = b; }

  void submit();  // submit what is recorded (non-blocking)
  void finish();  // submit + wait for everything
  void trim();    // finish + free idle pooled buffers

  VulkanMemoryStats memory_stats() const;
  VulkanCounters counters() const { return counters_; }

 private:
  struct Pipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
  };
  struct KernelObjs {
    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    uint32_t nbind = 0;
  };
  struct Slot {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;  // fallback path only
    bool in_flight = false;
    Buffer staging;
    uint64_t staging_off = 0;
  };

  void create_instance();
  void pick_device();
  void create_device();
  void create_slots();
  void begin_if_needed();
  void hazard(std::initializer_list<Bind> binds);
  void hazard_one(VkBuffer b, bool write);
  void barrier();
  VkPipeline pipeline(K k, const Spec& spec);
  KernelObjs& kernel(K k);
  Buffer create_buffer(uint64_t bytes, VkBufferUsageFlags usage, bool host, bool host_random);
  void destroy_buffer(Buffer& b);
  void free_pool();

  VulkanBackendOptions opts_;
  std::string device_name_;
  Limits limits_;

  VkInstance instance_ = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
  VkPhysicalDevice phys_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VolkDeviceTable d_{};  // per-device function table (other modules use volk too)
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t queue_family_ = 0;
  VmaAllocator vma_ = nullptr;
  bool push_desc_ = false;
  bool coop_ = false;
  bool validation_ = false;

  std::array<KernelObjs, static_cast<size_t>(K::Count)> kernels_{};
  std::map<std::pair<uint32_t, Spec>, Pipeline> pipelines_;

  std::vector<Slot> slots_;
  uint32_t cur_ = 0;
  bool recording_ = false;
  int cb_dispatches_ = 0;
  std::unordered_set<VkBuffer> rd_, wr_;
  VkBuffer disjoint_ = VK_NULL_HANDLE;

  std::map<uint64_t, std::vector<Buffer>> free_;
  Buffer dummy_;
  Buffer readback_;

  uint64_t device_bytes_ = 0, peak_device_bytes_ = 0;
  uint64_t live_bytes_ = 0, peak_live_bytes_ = 0;
  uint64_t pooled_bytes_ = 0, staging_total_ = 0;
  VulkanCounters counters_;
};

// Probe without creating a context.
std::vector<VulkanDeviceInfo> enumerate_devices();

}  // namespace um::vkx

#include "app/backend_select.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#ifdef UM_APP_HAVE_VULKAN
#include "vk/vk_backend.h"
#endif

namespace um::app {

BackendKind parse_backend(const std::string& s0) {
  std::string s = s0;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  if (s.empty() || s == "auto") return BackendKind::Auto;
  if (s == "cpu") return BackendKind::Cpu;
  if (s == "vulkan" || s == "vk" || s == "gpu") return BackendKind::Vulkan;
  throw std::runtime_error("unknown backend '" + s0 + "' (expected auto, cpu or vulkan)");
}

const char* backend_kind_name(BackendKind k) {
  switch (k) {
    case BackendKind::Cpu: return "cpu";
    case BackendKind::Vulkan: return "vulkan";
    default: return "auto";
  }
}

bool vulkan_compute_compiled() {
#ifdef UM_APP_HAVE_VULKAN
  return true;
#else
  return false;
#endif
}

bool vulkan_compute_available() {
#ifdef UM_APP_HAVE_VULKAN
  try {
    return um::vulkan_available();
  } catch (...) {
    return false;
  }
#else
  return false;
#endif
}

CreatedBackend create_backend(BackendKind kind) {
  CreatedBackend out;
  if (kind != BackendKind::Cpu) {
#ifdef UM_APP_HAVE_VULKAN
    if (kind == BackendKind::Vulkan || um::vulkan_available()) {
      try {
        out.backend = um::make_vulkan_backend();
        out.is_vulkan = true;
        out.description = "vulkan (" + um::vulkan_device_name(*out.backend) + ")";
        return out;
      } catch (const std::exception& e) {
        if (kind == BackendKind::Vulkan)
          throw std::runtime_error(std::string("Vulkan compute backend unavailable: ") + e.what());
        out.notice = std::string("Vulkan compute backend failed (") + e.what() + "), using the CPU backend";
      }
    } else {
      out.notice = "no usable Vulkan device found, using the CPU backend";
    }
#else
    if (kind == BackendKind::Vulkan)
      throw std::runtime_error(
          "this build has no Vulkan compute backend (um_vk was not built); use --backend cpu");
    out.notice = "built without the Vulkan compute backend (um_vk), using the CPU backend";
#endif
  }
  out.backend = um::make_cpu_backend();
  out.description = "cpu";
  return out;
}

void backend_finish(Backend& be) {
#ifdef UM_APP_HAVE_VULKAN
  if (um::is_vulkan_backend(be)) {
    um::vulkan_finish(be);
    return;
  }
#endif
  be.flush();
}

}  // namespace um::app

namespace um::app {
void backend_trim(Backend& be) {
#ifdef UM_APP_HAVE_VULKAN
  if (um::is_vulkan_backend(be)) um::vulkan_trim(be);
#else
  (void)be;
#endif
}
}  // namespace um::app

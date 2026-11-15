// Run-time choice of the compute backend: --backend vulkan|cpu|auto.
//   auto   = Vulkan compute if um_vk was built and a usable device exists,
//            otherwise the CPU reference backend (a notice says why).
//   vulkan = Vulkan or an error (never a silent fallback).
#pragma once

#include <memory>
#include <string>

#include "core/tensor.h"

namespace um::app {

enum class BackendKind { Auto, Cpu, Vulkan };

// "auto" | "cpu" | "vulkan" (case-insensitive, "vk"/"gpu" accepted); throws.
BackendKind parse_backend(const std::string& s);
const char* backend_kind_name(BackendKind k);

// True if this binary was linked with the Vulkan compute backend.
bool vulkan_compute_compiled();
// True if it was compiled in AND a usable device exists (never throws).
bool vulkan_compute_available();

struct CreatedBackend {
  std::unique_ptr<Backend> backend;
  bool is_vulkan = false;
  std::string description;  // "cpu" or "vulkan (<device name>)"
  std::string notice;       // non-empty when auto fell back to CPU
};

// Throws std::runtime_error if Vulkan was requested explicitly and cannot be
// created.
CreatedBackend create_backend(BackendKind kind);

// Releases idle pooled device memory (Vulkan; no-op on CPU).
void backend_trim(Backend& be);
// Blocks until all work queued on `be` has finished (no-op on CPU).
void backend_finish(Backend& be);

}  // namespace um::app

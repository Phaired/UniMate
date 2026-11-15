# um_viewer: GLB loading (cgltf), Vulkan forward renderer with GPU skinning,
# MotionClip playback, orbit camera and a Dear ImGui panel.
um_require(volk VulkanMemoryAllocator glfw glm imgui cgltf stb)

if(NOT TARGET um_imgui_vulkan)
  add_library(um_imgui_vulkan STATIC
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp)
  target_link_libraries(um_imgui_vulkan PUBLIC um_imgui volk glfw)
  target_compile_definitions(um_imgui_vulkan PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)
  set_target_properties(um_imgui_vulkan PROPERTIES FOLDER third_party)
endif()

add_library(um_viewer STATIC
  viewer.h
  asset.h asset.cpp
  vk_context.h vk_context.cpp
  renderer.h renderer.cpp
  camera.h
  viewer.cpp
  offscreen.cpp
  third_party_impl.cpp)

target_include_directories(um_viewer PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..          # "viewer/viewer.h", "core/generation.h"
)
target_link_libraries(um_viewer
  PUBLIC  volk
  PRIVATE GPUOpen::VulkanMemoryAllocator glfw glm::glm um_imgui um_imgui_vulkan um_cgltf um_stb)
target_compile_definitions(um_viewer PRIVATE
  GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_FORCE_RADIANS GLM_ENABLE_EXPERIMENTAL
  GLFW_INCLUDE_NONE)

option(UM_VIEWER_SELFTEST "Build src/viewer/tests/viewer_selftest (viewer automation test)" OFF)
if(UM_VIEWER_SELFTEST)
  add_executable(viewer_selftest tests/viewer_selftest.cpp)
  target_link_libraries(viewer_selftest PRIVATE um_viewer glm::glm)
  target_compile_definitions(viewer_selftest PRIVATE
    GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_FORCE_RADIANS GLM_ENABLE_EXPERIMENTAL)
endif()

um_embed_shaders(um_viewer INCLUDES shaders/common.glsl SOURCES
  shaders/mesh.vert shaders/mesh.frag
  shaders/line.vert shaders/line.frag
  shaders/grid.vert shaders/grid.frag)

# Third-party dependencies. Everything is declared here (declaring is free) and
# a module pulls what it needs with um_require(<name> ...), which downloads and
# builds on first use only.
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

set(UM_VULKAN_SDK_TAG "vulkan-sdk-1.3.290.0")

FetchContent_Declare(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3 GIT_SHALLOW TRUE)

FetchContent_Declare(VulkanHeaders
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
  GIT_TAG ${UM_VULKAN_SDK_TAG} GIT_SHALLOW TRUE)

FetchContent_Declare(volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG ${UM_VULKAN_SDK_TAG} GIT_SHALLOW TRUE)

FetchContent_Declare(VulkanMemoryAllocator
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
  GIT_TAG v3.1.0 GIT_SHALLOW TRUE)

FetchContent_Declare(glslang
  GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
  GIT_TAG ${UM_VULKAN_SDK_TAG} GIT_SHALLOW TRUE)

FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG 3.4 GIT_SHALLOW TRUE)

FetchContent_Declare(glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG 1.0.1 GIT_SHALLOW TRUE)

FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.91.5 GIT_SHALLOW TRUE)

FetchContent_Declare(cgltf
  GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
  GIT_TAG v1.14 GIT_SHALLOW TRUE)

FetchContent_Declare(stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG master GIT_SHALLOW TRUE)

# um_require(dep1 dep2 ...) -- make the listed dependencies available.
# Header-only / source-drop libraries get an INTERFACE target:
#   nlohmann_json::nlohmann_json, Vulkan::Headers, volk (static, no loader
#   import lib needed: call volkInitialize()), GPUOpen::VulkanMemoryAllocator,
#   glslang-standalone (build-time shader compiler, see cmake/shaders.cmake),
#   glfw, glm::glm, um_imgui (core only; add the backends you need from
#   ${imgui_SOURCE_DIR}/backends), um_cgltf, um_stb.
macro(um_require)
  foreach(_dep ${ARGN})
    if(_dep STREQUAL "nlohmann_json")
      set(JSON_BuildTests OFF CACHE INTERNAL "")
      FetchContent_MakeAvailable(nlohmann_json)
    elseif(_dep STREQUAL "VulkanHeaders")
      FetchContent_MakeAvailable(VulkanHeaders)
    elseif(_dep STREQUAL "volk")
      FetchContent_MakeAvailable(VulkanHeaders)
      if(WIN32)
        set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR CACHE INTERNAL "")
      endif()
      set(VOLK_PULL_IN_VULKAN OFF CACHE INTERNAL "")
      FetchContent_MakeAvailable(volk)
      if(TARGET volk AND TARGET Vulkan::Headers)
        target_link_libraries(volk PUBLIC Vulkan::Headers)
      endif()
    elseif(_dep STREQUAL "VulkanMemoryAllocator")
      FetchContent_MakeAvailable(VulkanHeaders)
      FetchContent_MakeAvailable(VulkanMemoryAllocator)
    elseif(_dep STREQUAL "glslang")
      set(ENABLE_OPT OFF CACHE INTERNAL "")
      set(ENABLE_HLSL OFF CACHE INTERNAL "")
      set(GLSLANG_TESTS OFF CACHE INTERNAL "")
      set(GLSLANG_ENABLE_INSTALL OFF CACHE INTERNAL "")
      set(ENABLE_GLSLANG_BINARIES ON CACHE INTERNAL "")
      set(BUILD_EXTERNAL OFF CACHE INTERNAL "")
      FetchContent_MakeAvailable(glslang)
    elseif(_dep STREQUAL "glfw")
      set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "")
      set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
      set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")
      set(GLFW_INSTALL OFF CACHE INTERNAL "")
      FetchContent_MakeAvailable(glfw)
    elseif(_dep STREQUAL "glm")
      FetchContent_MakeAvailable(glm)
    elseif(_dep STREQUAL "imgui")
      FetchContent_MakeAvailable(imgui)
      if(NOT TARGET um_imgui)
        add_library(um_imgui STATIC
          ${imgui_SOURCE_DIR}/imgui.cpp
          ${imgui_SOURCE_DIR}/imgui_draw.cpp
          ${imgui_SOURCE_DIR}/imgui_tables.cpp
          ${imgui_SOURCE_DIR}/imgui_widgets.cpp
          ${imgui_SOURCE_DIR}/imgui_demo.cpp
          ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp)
        target_include_directories(um_imgui PUBLIC
          ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends ${imgui_SOURCE_DIR}/misc/cpp)
      endif()
    elseif(_dep STREQUAL "cgltf")
      FetchContent_MakeAvailable(cgltf)
      if(NOT TARGET um_cgltf)
        add_library(um_cgltf INTERFACE)
        target_include_directories(um_cgltf INTERFACE ${cgltf_SOURCE_DIR})
      endif()
    elseif(_dep STREQUAL "stb")
      FetchContent_MakeAvailable(stb)
      if(NOT TARGET um_stb)
        add_library(um_stb INTERFACE)
        target_include_directories(um_stb INTERFACE ${stb_SOURCE_DIR})
      endif()
    else()
      message(FATAL_ERROR "um_require: unknown dependency '${_dep}'")
    endif()
  endforeach()
endmacro()

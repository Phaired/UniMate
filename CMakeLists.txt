cmake_minimum_required(VERSION 3.24)
project(unimate_cpp LANGUAGES C CXX)

# UniMate C++/Vulkan port. Each module lives in its own directory with its own
# CMakeLists.txt and is only added when its option is ON *and* the directory
# exists, so modules can be developed and built independently:
#
#   cmake -S . -B build/<name> -G "Visual Studio 17 2022" -A x64 -DUM_BUILD_VK=OFF ...
#   cmake --build build/<name> --config Release
#
# Module dependency order: core <- vk, t5, model, pipeline <- app, viewer <- apps.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(MSVC)
  add_compile_options(/W3 /MP /utf-8 /permissive- /bigobj)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

option(UM_BUILD_VK       "Vulkan compute tensor backend (src/vk)"            ON)
option(UM_BUILD_T5       "SentencePiece tokenizer + T5 encoder (src/t5)"     ON)
option(UM_BUILD_MODEL    "UniMate denoiser (src/model)"                      ON)
option(UM_BUILD_PIPELINE "Assets, conditioning, sampler, motion (src/pipeline)" ON)
option(UM_BUILD_APP      "Generator library: T5 + denoiser + pipeline (src/app)" ON)
option(UM_BUILD_VIEWER   "Vulkan real-time viewer (src/viewer)"              ON)
option(UM_BUILD_APPS     "Executables (apps/)"                               ON)
option(UM_BUILD_TESTS    "Unit / parity tests"                               ON)

# Root of the repository, for tests that read models/ and tests/golden/.
set(UM_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

include(cmake/deps.cmake)
include(cmake/shaders.cmake)
enable_testing()

function(um_add_module option dir)
  if(${option} AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt")
    add_subdirectory(${dir})
  endif()
endfunction()

option(UM_BUILD_CORE "Tensor API, CPU backend, UMTF (src/core)" ON)
um_add_module(UM_BUILD_CORE     src/core)       # um_core
um_add_module(UM_BUILD_VK       src/vk)         # um_vk
um_add_module(UM_BUILD_T5       src/t5)         # um_t5
um_add_module(UM_BUILD_MODEL    src/model)      # um_model
um_add_module(UM_BUILD_PIPELINE src/pipeline)   # um_pipeline
um_add_module(UM_BUILD_APP      src/app)        # um_app
um_add_module(UM_BUILD_VIEWER   src/viewer)     # um_viewer
um_add_module(UM_BUILD_APPS     apps)

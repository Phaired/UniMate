# um_app: the whole text-to-motion generator (T5 + denoiser + pipeline) behind
# one class, um::Generator, plus backend selection and asset resolution.
#   target_link_libraries(my_target PRIVATE um_app)
#   #include "app/generator.h"
# The Vulkan compute backend (um_vk) is optional: without it (or when it does
# not build) the library is CPU-only and --backend auto falls back to CPU.
foreach(_dep um_core um_t5 um_model um_pipeline)
  if(NOT TARGET ${_dep})
    message(WARNING "src/app needs ${_dep}; um_app not built")
    return()
  endif()
endforeach()

add_library(um_app STATIC
  generator.h generator.cpp
  backend_select.h backend_select.cpp
  assets.h assets.cpp)
target_include_directories(um_app PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(um_app PUBLIC um_core um_t5 um_model um_pipeline)
target_compile_definitions(um_app PRIVATE UM_ROOT_DIR="${UM_ROOT_DIR}")
target_compile_features(um_app PUBLIC cxx_std_20)
set_target_properties(um_app PROPERTIES FOLDER "app")

if(TARGET um_vk)
  target_link_libraries(um_app PRIVATE um_vk)
  target_compile_definitions(um_app PRIVATE UM_APP_HAVE_VULKAN=1)
  message(STATUS "um_app: Vulkan compute backend enabled (um_vk)")
else()
  message(STATUS "um_app: um_vk not available, CPU backend only")
endif()

if(UM_BUILD_TESTS AND EXISTS "${UM_ROOT_DIR}/tests/e2e/CMakeLists.txt")
  add_subdirectory("${UM_ROOT_DIR}/tests/e2e" "${CMAKE_CURRENT_BINARY_DIR}/e2e")
endif()

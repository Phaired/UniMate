# um_vk -- Vulkan compute implementation of um::Backend.
#
# Public header: vk/vk_backend.h  (include root is src/).
# Shaders in shaders/*.comp are compiled to SPIR-V at build time and embedded.
um_require(volk VulkanMemoryAllocator)

set(UM_VK_SHADERS
  shaders/unary.comp
  shaders/binary.comp
  shaders/copy.comp
  shaders/gather_rows.comp
  shaders/softmax.comp
  shaders/norm.comp
  shaders/reduce_rows.comp
  shaders/reduce_cols.comp
  shaders/gemm.comp
  shaders/gemm_coop.comp
  shaders/gemv.comp
  shaders/ternary.comp
  shaders/attention.comp)

add_library(um_vk STATIC
  vk_backend.h
  vk_context.h
  vk_context.cpp
  vk_backend.cpp
  vk_vma.cpp
  ${UM_VK_SHADERS}
  shaders/common.glsl)
target_include_directories(um_vk PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(um_vk PRIVATE volk GPUOpen::VulkanMemoryAllocator)
if(TARGET um_core)
  target_link_libraries(um_vk PUBLIC um_core)
endif()
set_target_properties(um_vk PROPERTIES FOLDER "unimate")
source_group("shaders" FILES ${UM_VK_SHADERS} shaders/common.glsl)
um_embed_shaders(um_vk SOURCES ${UM_VK_SHADERS} INCLUDES shaders/common.glsl)

if(UM_BUILD_TESTS AND TARGET um_core)
  add_executable(um_vk_parity tests/vk_parity.cpp)
  target_link_libraries(um_vk_parity PRIVATE um_vk um_core)
  set_target_properties(um_vk_parity PROPERTIES FOLDER "unimate/tests")
  add_test(NAME um_vk_parity COMMAND um_vk_parity)

  add_executable(um_vk_bench tests/vk_bench.cpp)
  target_link_libraries(um_vk_bench PRIVATE um_vk um_core)
  set_target_properties(um_vk_bench PROPERTIES FOLDER "unimate/tests")

  # End-to-end check on the real T5 encoder / denoiser. src/t5 and src/model
  # are added after src/vk, so decide at the end of the top-level directory.
  set(UM_VK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "")
  function(um_vk_add_e2e)
    if(NOT TARGET um_t5 AND NOT TARGET um_model)
      return()
    endif()
    add_executable(um_vk_e2e "${UM_VK_SOURCE_DIR}/tests/vk_e2e.cpp")
    target_link_libraries(um_vk_e2e PRIVATE um_vk um_core)
    target_compile_definitions(um_vk_e2e PRIVATE UM_ROOT_DIR="${UM_ROOT_DIR}")
    if(TARGET um_t5)
      target_link_libraries(um_vk_e2e PRIVATE um_t5)
      target_compile_definitions(um_vk_e2e PRIVATE UM_VK_HAVE_T5)
    endif()
    if(TARGET um_model)
      target_link_libraries(um_vk_e2e PRIVATE um_model)
      target_compile_definitions(um_vk_e2e PRIVATE UM_VK_HAVE_MODEL)
    endif()
    set_target_properties(um_vk_e2e PROPERTIES FOLDER "unimate/tests")
    add_test(NAME um_vk_e2e COMMAND um_vk_e2e)
    set_tests_properties(um_vk_e2e PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
  endfunction()
  cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL um_vk_add_e2e)
endif()

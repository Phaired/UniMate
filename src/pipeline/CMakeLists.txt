# um_pipeline: asset bundles, conditioning, flow-matching ODE sampler, motion
# recovery and GLB export (the non-neural part of UniMate inference).
#   target_link_libraries(my_target PRIVATE um_pipeline)
#   #include "pipeline/pipeline.h"
if(NOT TARGET nlohmann_json::nlohmann_json)
  um_require(nlohmann_json)
endif()

add_library(um_pipeline STATIC
  pipeline.h pipeline.cpp
  bundle.h bundle.cpp
  condition.h condition.cpp
  sampler.h sampler.cpp
  motion.h motion.cpp
  glb.h glb.cpp
  math3.h)
target_include_directories(um_pipeline PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(um_pipeline PUBLIC um_core nlohmann_json::nlohmann_json)
target_compile_features(um_pipeline PUBLIC cxx_std_20)
set_target_properties(um_pipeline PROPERTIES FOLDER "pipeline")

if(UM_BUILD_TESTS)
  # cgltf is used by the export round-trip test only (the library parses GLB
  # itself so it never defines CGLTF_IMPLEMENTATION next to the viewer's).
  um_require(cgltf)
  foreach(_t condition sampler motion export)
    add_executable(um_pipeline_test_${_t} tests/test_${_t}.cpp tests/test_common.h)
    target_link_libraries(um_pipeline_test_${_t} PRIVATE um_pipeline)
    target_compile_definitions(um_pipeline_test_${_t} PRIVATE UM_ROOT_DIR="${UM_ROOT_DIR}")
    set_target_properties(um_pipeline_test_${_t} PROPERTIES FOLDER "pipeline")
    add_test(NAME um_pipeline_${_t} COMMAND um_pipeline_test_${_t})
    set_tests_properties(um_pipeline_${_t} PROPERTIES SKIP_RETURN_CODE 77)
  endforeach()
  target_link_libraries(um_pipeline_test_export PRIVATE um_cgltf)
endif()

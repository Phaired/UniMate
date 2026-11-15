# um_model: UniMate denoiser (inference) on the um::Backend tensor API.
#   target_link_libraries(my_target PUBLIC um_model)
#   #include "model/denoiser.h"
add_library(um_model STATIC
  denoiser.cpp
  denoiser_layers.cpp
  denoiser.h
  denoiser_layers.h)
target_include_directories(um_model PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(um_model PUBLIC um_core)
target_compile_features(um_model PUBLIC cxx_std_20)
set_target_properties(um_model PROPERTIES FOLDER "model")

if(UM_BUILD_TESTS)
  add_executable(um_model_tests tests/test_denoiser.cpp)
  target_link_libraries(um_model_tests PRIVATE um_model)
  target_compile_definitions(um_model_tests PRIVATE UM_ROOT_DIR="${UM_ROOT_DIR}")
  set_target_properties(um_model_tests PROPERTIES FOLDER "model")
  # Exit code 77 = converted weights / goldens missing -> reported as skipped.
  add_test(NAME um_model_denoiser_golden COMMAND um_model_tests)
  set_tests_properties(um_model_denoiser_golden PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 1800)
endif()

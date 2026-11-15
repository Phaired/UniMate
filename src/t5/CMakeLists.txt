# um_t5: SentencePiece-unigram tokenizer + flan-t5 encoder on um::Backend.
#   target_link_libraries(my_target PUBLIC um_t5)
#   #include "t5/t5_encoder.h"
# Model file: models/converted/flan-t5-base.umtf (tools/python/t5_export.py).
add_library(um_t5 STATIC
  t5_tokenizer.cpp
  t5_encoder.cpp
  t5_tokenizer.h
  t5_encoder.h)
target_include_directories(um_t5 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(um_t5 PUBLIC um_core)
target_compile_features(um_t5 PUBLIC cxx_std_20)
set_target_properties(um_t5 PROPERTIES FOLDER "t5")

if(UM_BUILD_TESTS)
  add_executable(um_t5_tests tests/test_t5.cpp)
  target_link_libraries(um_t5_tests PRIVATE um_t5)
  target_compile_definitions(um_t5_tests PRIVATE UM_ROOT_DIR="${UM_ROOT_DIR}")
  set_target_properties(um_t5_tests PROPERTIES FOLDER "t5")
  add_test(NAME um_t5_tests COMMAND um_t5_tests)
  # A missing model / golden file is reported and skipped, not failed.
  set_tests_properties(um_t5_tests PROPERTIES SKIP_RETURN_CODE 77)
endif()

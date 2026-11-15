# um_core: Tensor/Backend API, CPU reference backend, UMTF I/O.
# Use from another module:
#   target_link_libraries(my_target PUBLIC um_core)
#   #include "core/tensor.h"  /  #include "core/umtf.h"
if(NOT TARGET nlohmann_json::nlohmann_json)
  um_require(nlohmann_json)
endif()

add_library(um_core STATIC
  tensor.cpp
  cpu_backend.cpp
  umtf.cpp
  tensor.h
  umtf.h
  generation.h)
target_include_directories(um_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(um_core PUBLIC nlohmann_json::nlohmann_json)
target_compile_features(um_core PUBLIC cxx_std_20)

if(MSVC)
  # AVX2+FMA for the GEMM kernels (IEEE semantics kept: no /fp:fast).
  set_source_files_properties(cpu_backend.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/O2")
endif()
set_target_properties(um_core PROPERTIES FOLDER "core")

if(UM_BUILD_TESTS)
  add_executable(um_core_tests tests/test_core.cpp)
  target_link_libraries(um_core_tests PRIVATE um_core)
  add_test(NAME um_core_tests COMMAND um_core_tests)

  add_executable(um_core_xlang tests/test_umtf_xlang.cpp)
  target_link_libraries(um_core_xlang PRIVATE um_core)
  set_target_properties(um_core_tests um_core_xlang PROPERTIES FOLDER "core")

  # Python runner that has numpy: prefer repo venv, then `uv run --with numpy`, then python.
  set(_um_venv_py "${UM_ROOT_DIR}/.venv/Scripts/python.exe")
  find_program(_um_uv uv)
  set(UM_PY_RUNNER "" CACHE STRING "Command used to run python with numpy for cross-language tests")
  if(NOT UM_PY_RUNNER)
    if(_um_uv)
      set(UM_PY_RUNNER "${_um_uv} run --with numpy python")
    elseif(EXISTS "${_um_venv_py}")
      set(UM_PY_RUNNER "${_um_venv_py}")
    else()
      set(UM_PY_RUNNER "python")
    endif()
  endif()
  add_test(NAME um_core_xlang
    COMMAND um_core_xlang "${CMAKE_CURRENT_BINARY_DIR}/xlang_work" "${UM_PY_RUNNER}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/xlang.py")
endif()

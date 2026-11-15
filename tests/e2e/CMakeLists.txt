# End-to-end parity tests: C++ T5 + denoiser + sampler + recovery vs the real
# upstream PyTorch run (tests/golden/pipeline). Added from src/app.
#   ctest -L e2e            all of them (CPU: ~1 min per Euler test, ~6 min per dopri5 test)
#   ctest -L vulkan         the same on the Vulkan compute backend (skipped without um_vk / device)
add_executable(um_e2e test_e2e.cpp)
target_link_libraries(um_e2e PRIVATE um_app um_model um_pipeline)
set_target_properties(um_e2e PROPERTIES FOLDER "tests")

set(_um_e2e_backends cpu)
if(TARGET um_vk)
  list(APPEND _um_e2e_backends vulkan)
endif()
foreach(_be ${_um_e2e_backends})
  foreach(_asset eagle shark)
    foreach(_solver euler dopri5)
      set(_name um_e2e_${_asset}_${_solver}_${_be})
      add_test(NAME ${_name} COMMAND um_e2e --asset ${_asset} --solver ${_solver} --backend ${_be})
      set(_labels e2e ${_be})
      if(_solver STREQUAL "dopri5" AND _be STREQUAL "cpu")
        list(APPEND _labels long)
      endif()
      set_tests_properties(${_name} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600 LABELS "${_labels}")
    endforeach()
  endforeach()
endforeach()

# unitree_go2: Euler-20 golden only (pipeline_golden.py --assets unitree_go2 --euler-only --skip-ode);
# skipped (77) when the golden is absent.
foreach(_be ${_um_e2e_backends})
  add_test(NAME um_e2e_unitree_go2_euler_${_be} COMMAND um_e2e --asset unitree_go2 --solver euler --backend ${_be})
  set_tests_properties(um_e2e_unitree_go2_euler_${_be} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600 LABELS "e2e;${_be}")
endforeach()

# Cancellation / exception safety of Generator::generate (short: 3 + 2 x <=3 velocity calls).
add_executable(um_app_cancel test_cancel.cpp)
target_link_libraries(um_app_cancel PRIVATE um_app)
set_target_properties(um_app_cancel PROPERTIES FOLDER "tests")
foreach(_be ${_um_e2e_backends})
  add_test(NAME um_app_cancel_${_be} COMMAND um_app_cancel --backend ${_be})
  set_tests_properties(um_app_cancel_${_be} PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900 LABELS "app;${_be}")
endforeach()

add_executable(unimate_cli main.cpp)
target_link_libraries(unimate_cli PRIVATE um_app)
if(TARGET um_viewer)
  target_link_libraries(unimate_cli PRIVATE um_viewer)
  target_compile_definitions(unimate_cli PRIVATE UM_CLI_HAVE_VIEWER=1)
endif()
set_target_properties(unimate_cli PROPERTIES FOLDER "apps"
  VS_DEBUGGER_COMMAND_ARGUMENTS "--asset eagle --prompt \"An object flaps its wings.\" --solver euler --steps 20 --out eagle.glb --png eagle.png")

if(UM_BUILD_TESTS)
  # Quick smoke test: 2 Euler steps on eagle, export round trip (+ PNG when the viewer is built).
  set(_um_cli_smoke "${CMAKE_CURRENT_BINARY_DIR}/smoke")
  file(MAKE_DIRECTORY "${_um_cli_smoke}")
  set(_um_png_args "")
  if(TARGET um_viewer)
    set(_um_png_args --png "${_um_cli_smoke}/eagle.png" --png-frame 0,29)
  endif()
  add_test(NAME unimate_cli_smoke
    COMMAND unimate_cli --asset eagle --prompt "An object flaps its wings." --solver euler --steps 3
            --frames 30 --backend cpu --out "${_um_cli_smoke}/eagle.glb"
            ${_um_png_args})
  set_tests_properties(unimate_cli_smoke PROPERTIES LABELS "app" TIMEOUT 600)
endif()

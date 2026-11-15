add_executable(unimate_viewer_demo main.cpp)
target_link_libraries(unimate_viewer_demo PRIVATE um_viewer)
set_target_properties(unimate_viewer_demo PROPERTIES FOLDER "apps"
  VS_DEBUGGER_COMMAND_ARGUMENTS "${UM_ROOT_DIR}/reference/UniMate/assets/examples/eagle/Eagle.glb")

if(TARGET um_app)
  add_executable(unimate_viewer unimate_viewer.cpp)
  target_link_libraries(unimate_viewer PRIVATE um_app um_viewer)
  set_target_properties(unimate_viewer PROPERTIES FOLDER "apps" VS_DEBUGGER_COMMAND_ARGUMENTS "eagle")
endif()

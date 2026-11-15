# Executables. Each app is only added when the libraries it needs were built.
#   unimate_cli          prompt + asset -> animated GLB (+ PNG frames)  [um_app, um_viewer optional]
#   unimate_viewer       real-time viewer + real generator             [um_app + um_viewer]
#   unimate_viewer_demo  viewer with a fake procedural generator        [um_viewer]
if(TARGET um_app)
  add_subdirectory(cli)
endif()
if(TARGET um_viewer)
  add_subdirectory(viewer)
endif()

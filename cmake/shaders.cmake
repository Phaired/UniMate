# um_embed_shaders(<target> [INCLUDES <glsl files only used via #include>...]
#                  SOURCES a.comp b.vert ...)
#
# Compiles each GLSL source to SPIR-V at build time with the glslang binary
# built from source (no Vulkan SDK needed) and embeds it as a C header:
#
#   foo.comp  ->  #include "foo.comp.h"   // const uint32_t spv_foo_comp[] = {...};
#
# The headers land in a per-target directory that is added to the target's
# private include path.
function(um_embed_shaders target)
  cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDES" ${ARGN})
  um_require(glslang)
  set(_out_dir "${CMAKE_CURRENT_BINARY_DIR}/${target}_spv")
  set(_outs)
  set(_incs)
  foreach(_inc ${ARG_INCLUDES})
    list(APPEND _incs "${CMAKE_CURRENT_SOURCE_DIR}/${_inc}")
  endforeach()
  foreach(_src ${ARG_SOURCES})
    get_filename_component(_name "${_src}" NAME)
    get_filename_component(_dir "${CMAKE_CURRENT_SOURCE_DIR}/${_src}" DIRECTORY)
    string(MAKE_C_IDENTIFIER "${_name}" _ident)
    set(_out "${_out_dir}/${_name}.h")
    add_custom_command(
      OUTPUT "${_out}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${_out_dir}"
      COMMAND $<TARGET_FILE:glslang-standalone> -V --target-env vulkan1.2
              "-I${_dir}" --vn spv_${_ident} -o "${_out}"
              "${CMAKE_CURRENT_SOURCE_DIR}/${_src}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_src}" ${_incs} glslang-standalone
      COMMENT "GLSL -> SPIR-V ${_name}"
      VERBATIM)
    list(APPEND _outs "${_out}")
  endforeach()
  add_custom_target(${target}_shaders DEPENDS ${_outs})
  add_dependencies(${target} ${target}_shaders)
  target_include_directories(${target} PRIVATE "${_out_dir}")
endfunction()

#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 out_color;

void main() {
  if (pc.v0.y > 0.5) {  // round points
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    if (dot(d, d) > 1.0) discard;
  }
  out_color = v_color;
}

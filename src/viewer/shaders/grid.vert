#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) out vec3 v_world;

void main() {
  const vec2 corners[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
  vec2 c = corners[gl_VertexIndex];
  vec3 p = vec3(pc.v1.x + c.x * pc.v1.w, pc.v1.y, pc.v1.z + c.y * pc.v1.w);
  v_world = p;
  gl_Position = g.view_proj * vec4(p, 1.0);
}

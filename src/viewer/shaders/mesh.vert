#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(std430, set = 0, binding = 1) readonly buffer Palette { mat4 m[]; } pal;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in uvec4 in_joints;
layout(location = 4) in vec4 in_weights;

layout(location = 0) out vec3 v_world;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;

void main() {
  uint base = pc.u0.x;
  mat4 skin = in_weights.x * pal.m[base + in_joints.x] +
              in_weights.y * pal.m[base + in_joints.y] +
              in_weights.z * pal.m[base + in_joints.z] +
              in_weights.w * pal.m[base + in_joints.w];
  vec4 world = skin * vec4(in_pos, 1.0);
  v_world = world.xyz;
  v_normal = transpose(inverse(mat3(skin))) * in_normal;
  v_uv = in_uv;
  gl_Position = g.view_proj * world;
}

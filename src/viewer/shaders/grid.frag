#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(location = 0) in vec3 v_world;
layout(location = 0) out vec4 out_color;

float grid_line(vec2 coord) {
  vec2 d = fwidth(coord);
  vec2 g2 = abs(fract(coord - 0.5) - 0.5) / max(d, vec2(1e-6));
  return 1.0 - min(min(g2.x, g2.y), 1.0);
}

void main() {
  float cell = pc.v0.x;
  vec2 xz = v_world.xz;
  float minor = grid_line(xz / cell);
  float major = grid_line(xz / (cell * pc.v0.y));
  vec3 col = vec3(0.35);
  float a = max(minor * 0.35, major * 0.7);
  // world axes: X red, Z blue
  vec2 dw = fwidth(xz);
  float ax = 1.0 - min(abs(xz.y) / max(dw.y, 1e-6), 1.0);
  float az = 1.0 - min(abs(xz.x) / max(dw.x, 1e-6), 1.0);
  if (ax > 0.0) { col = mix(col, vec3(0.8, 0.15, 0.15), ax); a = max(a, ax); }
  if (az > 0.0) { col = mix(col, vec3(0.15, 0.25, 0.85), az); a = max(a, az); }
  float r = length(xz - pc.v1.xz) / pc.v1.w;
  a *= 1.0 - smoothstep(0.6, 1.0, r);
  if (a <= 0.003) discard;
  out_color = vec4(col, a);
}

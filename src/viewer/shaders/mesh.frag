#version 450
#extension GL_GOOGLE_include_directive : require
#include "common.glsl"

layout(set = 1, binding = 0) uniform sampler2D base_tex;  // sRGB format: sampled as linear

layout(location = 0) in vec3 v_world;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 0) out vec4 out_color;

void main() {
  vec4 base = pc.v0 * texture(base_tex, v_uv);
  if (pc.v1.x > 0.0 && base.a < pc.v1.x) discard;
  vec3 n = normalize(v_normal);
  if (!gl_FrontFacing) n = -n;  // double-sided lighting
  vec3 l = normalize(-g.light_dir.xyz);
  vec3 v = normalize(g.cam_pos.xyz - v_world);
  vec3 h = normalize(l + v);
  vec3 hemi = mix(g.ground_color.rgb, g.sky_color.rgb, n.y * 0.5 + 0.5);
  float ndl = max(dot(n, l), 0.0);
  float spec = pow(max(dot(n, h), 0.0), 48.0) * ndl;
  vec3 c = base.rgb * (hemi + g.light_color.rgb * ndl) + g.light_color.rgb * spec * 0.08;
  // gentle Reinhard-style shoulder so bright lights do not clip harshly
  c = c / (1.0 + max(max(c.r, c.g), c.b) * 0.15);
  // linear; the sRGB attachment encodes. Alpha only matters for BLEND materials (v1.z).
  out_color = vec4(c, pc.v1.z > 0.5 ? clamp(base.a, 0.0, 1.0) : 1.0);
}

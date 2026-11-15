// Shared declarations for the viewer shaders.
layout(set = 0, binding = 0) uniform Globals {
  mat4 view_proj;
  vec4 cam_pos;      // xyz
  vec4 light_dir;    // xyz: direction the light travels (world)
  vec4 light_color;  // rgb (linear)
  vec4 sky_color;    // rgb (linear)
  vec4 ground_color; // rgb (linear)
} g;

// Same 48-byte push block for every pipeline; each shader interprets it.
layout(push_constant) uniform Push {
  vec4 v0;   // mesh: base colour factor | grid: (cell, major_every, -, -) | lines: (point size, is_points, -, -)
  vec4 v1;   // mesh: (alpha cutoff, has_texture, is_blend, -) | grid: (center.x, ground y, center.z, half size)
  uvec4 u0;  // mesh: (palette base, -, -, -)
} pc;

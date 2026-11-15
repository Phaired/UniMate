#include "asset.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>

#include <cgltf.h>
#include <stb_image.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>

namespace um::viewer {
namespace {

int node_index(const cgltf_data* d, const cgltf_node* n) {
  return n ? int(n - d->nodes) : -1;
}

std::vector<uint8_t> decode_base64(const char* s) {
  auto val = [](char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+' || c == '-') return 62;
    if (c == '/' || c == '_') return 63;
    return -1;
  };
  std::vector<uint8_t> out;
  uint32_t acc = 0;
  int bits = 0;
  for (; *s; ++s) {
    int v = val(*s);
    if (v < 0) continue;
    acc = (acc << 6) | uint32_t(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(uint8_t((acc >> bits) & 0xFF));
    }
  }
  return out;
}

bool decode_image(const cgltf_image* img, const std::string& gltf_path, ImageRGBA8& out) {
  std::vector<uint8_t> bytes;
  const uint8_t* data = nullptr;
  size_t size = 0;
  if (img->buffer_view) {
    data = static_cast<const uint8_t*>(cgltf_buffer_view_data(img->buffer_view));
    size = img->buffer_view->size;
  } else if (img->uri) {
    std::string uri = img->uri;
    if (uri.rfind("data:", 0) == 0) {
      auto p = uri.find("base64,");
      if (p == std::string::npos) return false;
      bytes = decode_base64(uri.c_str() + p + 7);
    } else {
      std::string dec = uri;
      cgltf_decode_uri(dec.data());
      dec.resize(std::strlen(dec.c_str()));
      auto u8 = [](const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); };
      auto file = u8(gltf_path).parent_path() / u8(dec);
      std::ifstream f(file, std::ios::binary);
      if (!f) return false;
      bytes.assign(std::istreambuf_iterator<char>(f), {});
    }
    data = bytes.data();
    size = bytes.size();
  }
  if (!data || !size) return false;
  int w, h, c;
  stbi_uc* px = stbi_load_from_memory(data, int(size), &w, &h, &c, 4);
  if (!px) return false;
  out.width = w;
  out.height = h;
  out.pixels.assign(px, px + size_t(w) * h * 4);
  stbi_image_free(px);
  return true;
}

// Evaluates a glTF animation sampler at time t (component count `nc`).
void eval_sampler(const cgltf_animation_sampler* s, float t, int nc, bool is_rot, float* out) {
  const cgltf_accessor* in = s->input;
  const cgltf_accessor* o = s->output;
  size_t n = in->count;
  if (n == 0) return;
  const bool cubic = s->interpolation == cgltf_interpolation_type_cubic_spline;
  auto key_time = [&](size_t i) {
    float v = 0;
    cgltf_accessor_read_float(in, i, &v, 1);
    return v;
  };
  auto value = [&](size_t key, int which, float* v) {  // which: 0 in-tangent, 1 value, 2 out-tangent
    size_t idx = cubic ? key * 3 + which : key;
    cgltf_accessor_read_float(o, idx, v, nc);
  };
  if (t <= key_time(0) || n == 1) { value(0, 1, out); return; }
  if (t >= key_time(n - 1)) { value(n - 1, 1, out); return; }
  size_t lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    size_t mid = (lo + hi) / 2;
    (key_time(mid) <= t ? lo : hi) = mid;
  }
  float t0 = key_time(lo), t1 = key_time(hi);
  float dt = std::max(t1 - t0, 1e-9f);
  float u = (t - t0) / dt;
  float a[4] = {}, b[4] = {};
  if (s->interpolation == cgltf_interpolation_type_step) { value(lo, 1, out); return; }
  if (cubic) {
    float m0[4] = {}, m1[4] = {};
    value(lo, 1, a); value(lo, 2, m0);
    value(hi, 1, b); value(hi, 0, m1);
    float u2 = u * u, u3 = u2 * u;
    float h00 = 2 * u3 - 3 * u2 + 1, h10 = u3 - 2 * u2 + u, h01 = -2 * u3 + 3 * u2, h11 = u3 - u2;
    for (int k = 0; k < nc; ++k) out[k] = h00 * a[k] + h10 * dt * m0[k] + h01 * b[k] + h11 * dt * m1[k];
  } else {
    value(lo, 1, a);
    value(hi, 1, b);
    if (is_rot) {
      glm::quat qa(a[3], a[0], a[1], a[2]), qb(b[3], b[0], b[1], b[2]);
      glm::quat q = glm::slerp(qa, qb, u);
      out[0] = q.x; out[1] = q.y; out[2] = q.z; out[3] = q.w;
      return;
    }
    for (int k = 0; k < nc; ++k) out[k] = a[k] + (b[k] - a[k]) * u;
  }
  if (is_rot) {
    float l = std::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]);
    if (l > 0) for (int k = 0; k < 4; ++k) out[k] /= l;
  }
}

void convert_animations(const cgltf_data* d, const Asset& a, float fps, std::vector<um::MotionClip>& clips) {
  for (size_t ai = 0; ai < d->animations_count; ++ai) {
    const cgltf_animation& an = d->animations[ai];
    float t0 = 1e30f, t1 = -1e30f;
    for (size_t c = 0; c < an.channels_count; ++c) {
      const cgltf_accessor* in = an.channels[c].sampler->input;
      if (!in || in->count == 0) continue;
      float lo = 0, hi = 0;
      cgltf_accessor_read_float(in, 0, &lo, 1);
      cgltf_accessor_read_float(in, in->count - 1, &hi, 1);
      t0 = std::min(t0, lo);
      t1 = std::max(t1, hi);
    }
    if (t1 < t0) continue;
    um::MotionClip clip;
    clip.name = an.name && *an.name ? std::string(an.name) : "animation_" + std::to_string(ai);
    clip.fps = fps;
    clip.num_frames = std::max(1, int(std::floor((t1 - t0) * fps + 1e-3f)) + 1);
    std::unordered_map<int, size_t> track_of;
    for (size_t c = 0; c < an.channels_count; ++c) {
      const cgltf_animation_channel& ch = an.channels[c];
      int ni = node_index(d, ch.target_node);
      if (ni < 0) continue;
      int nc = 0;
      bool rot = false;
      switch (ch.target_path) {
        case cgltf_animation_path_type_translation: nc = 3; break;
        case cgltf_animation_path_type_rotation: nc = 4; rot = true; break;
        case cgltf_animation_path_type_scale: nc = 3; break;
        default: continue;  // morph weights unsupported
      }
      auto it = track_of.find(ni);
      if (it == track_of.end()) {
        it = track_of.emplace(ni, clip.tracks.size()).first;
        clip.tracks.push_back({});
        clip.tracks.back().node_name = a.nodes[ni].name;
      }
      um::NodeTrack& tr = clip.tracks[it->second];
      std::vector<float>& dst = ch.target_path == cgltf_animation_path_type_translation ? tr.translation
                                : rot ? tr.rotation : tr.scale;
      dst.resize(size_t(clip.num_frames) * nc);
      for (int f = 0; f < clip.num_frames; ++f)
        eval_sampler(ch.sampler, t0 + f / fps, nc, rot, &dst[size_t(f) * nc]);
    }
    clips.push_back(std::move(clip));
  }
}

void compute_normals(std::vector<Vertex>& v, size_t v0, const std::vector<uint32_t>& idx, size_t i0) {
  for (size_t i = v0; i < v.size(); ++i) v[i].normal[0] = v[i].normal[1] = v[i].normal[2] = 0;
  for (size_t i = i0; i + 2 < idx.size(); i += 3) {
    Vertex* t[3] = {&v[v0 + idx[i]], &v[v0 + idx[i + 1]], &v[v0 + idx[i + 2]]};
    glm::vec3 p0 = glm::make_vec3(t[0]->pos), p1 = glm::make_vec3(t[1]->pos), p2 = glm::make_vec3(t[2]->pos);
    glm::vec3 n = glm::cross(p1 - p0, p2 - p0);
    for (auto* x : t) for (int k = 0; k < 3; ++k) x->normal[k] += n[k];
  }
  for (size_t i = v0; i < v.size(); ++i) {
    glm::vec3 n = glm::make_vec3(v[i].normal);
    float l = glm::length(n);
    n = l > 0 ? n / l : glm::vec3(0, 1, 0);
    for (int k = 0; k < 3; ++k) v[i].normal[k] = n[k];
  }
}

}  // namespace

bool load_asset(const std::string& path, Asset& a, std::string& err, bool load_images, float clip_fps) {
  a = Asset{};
  a.path = path;
  cgltf_options opt{};
  cgltf_data* d = nullptr;
  if (cgltf_parse_file(&opt, path.c_str(), &d) != cgltf_result_success) {
    err = "cannot parse glTF: " + path;
    return false;
  }
  struct Guard { cgltf_data* d; ~Guard() { cgltf_free(d); } } guard{d};
  if (cgltf_load_buffers(&opt, d, path.c_str()) != cgltf_result_success) {
    err = "cannot load glTF buffers: " + path;
    return false;
  }

  // ---- nodes ----
  a.nodes.resize(d->nodes_count);
  for (size_t i = 0; i < d->nodes_count; ++i) {
    const cgltf_node& n = d->nodes[i];
    Node& o = a.nodes[i];
    o.name = n.name && *n.name ? std::string(n.name) : "node_" + std::to_string(i);
    o.parent = node_index(d, n.parent);
    for (size_t c = 0; c < n.children_count; ++c) o.children.push_back(node_index(d, n.children[c]));
    if (n.has_matrix) {
      glm::mat4 m = glm::make_mat4(n.matrix);
      glm::vec3 skew;
      glm::vec4 persp;
      glm::decompose(m, o.s, o.r, o.t, skew, persp);
      o.r = glm::normalize(o.r);
    } else {
      if (n.has_translation) o.t = glm::make_vec3(n.translation);
      if (n.has_rotation) o.r = glm::normalize(glm::quat(n.rotation[3], n.rotation[0], n.rotation[1], n.rotation[2]));
      if (n.has_scale) o.s = glm::make_vec3(n.scale);
    }
    o.mesh = n.mesh ? int(n.mesh - d->meshes) : -1;
    o.skin = n.skin ? int(n.skin - d->skins) : -1;
    a.node_by_name.emplace(o.name, int(i));
  }
  // parents-before-children order
  std::vector<int> stack;
  for (int i = int(a.nodes.size()) - 1; i >= 0; --i)
    if (a.nodes[i].parent < 0) stack.push_back(i);
  while (!stack.empty()) {
    int n = stack.back();
    stack.pop_back();
    a.order.push_back(n);
    for (auto it = a.nodes[n].children.rbegin(); it != a.nodes[n].children.rend(); ++it) stack.push_back(*it);
  }

  // ---- skins ----
  uint32_t palette = 0;
  std::vector<char> is_joint(a.nodes.size(), 0);
  for (size_t si = 0; si < d->skins_count; ++si) {
    const cgltf_skin& s = d->skins[si];
    Skin sk;
    sk.palette_base = palette;
    for (size_t j = 0; j < s.joints_count; ++j) {
      int ni = node_index(d, s.joints[j]);
      sk.joints.push_back(ni);
      is_joint[ni] = 1;
      glm::mat4 ibm(1.f);
      if (s.inverse_bind_matrices) cgltf_accessor_read_float(s.inverse_bind_matrices, j, glm::value_ptr(ibm), 16);
      sk.inverse_bind.push_back(ibm);
    }
    palette += uint32_t(sk.joints.size());
    a.skins.push_back(std::move(sk));
  }
  for (size_t i = 0; i < a.nodes.size(); ++i)
    if (is_joint[i]) a.skeleton.push_back(int(i));
  {
    std::unordered_map<int, int> sk_index;
    for (size_t k = 0; k < a.skeleton.size(); ++k) sk_index[a.skeleton[k]] = int(k);
    for (int n : a.skeleton) {
      int p = a.nodes[n].parent;
      while (p >= 0 && !sk_index.count(p)) p = a.nodes[p].parent;  // skip non-joint intermediates
      a.skeleton_parent.push_back(p >= 0 ? sk_index[p] : -1);
    }
  }

  // ---- images & materials ----
  std::vector<int> image_of(d->images_count, -1);
  if (load_images) {
    for (size_t i = 0; i < d->images_count; ++i) {
      ImageRGBA8 img;
      if (decode_image(&d->images[i], path, img)) {
        image_of[i] = int(a.images.size());
        a.images.push_back(std::move(img));
      }
    }
  }
  for (size_t i = 0; i < d->materials_count; ++i) {
    const cgltf_material& m = d->materials[i];
    Material o;
    if (m.has_pbr_metallic_roughness) {
      o.base_color = glm::make_vec4(m.pbr_metallic_roughness.base_color_factor);
      const cgltf_texture* tex = m.pbr_metallic_roughness.base_color_texture.texture;
      if (tex && tex->image) o.image = image_of[tex->image - d->images];
    }
    if (m.alpha_mode == cgltf_alpha_mode_mask) o.alpha_cutoff = m.alpha_cutoff > 0 ? m.alpha_cutoff : 0.5f;
    o.blend = m.alpha_mode == cgltf_alpha_mode_blend;
    a.materials.push_back(o);
  }
  a.materials.push_back(Material{glm::vec4(0.8f, 0.8f, 0.8f, 1.f), -1, 0.f});  // default
  const int default_material = int(a.materials.size()) - 1;

  // ---- meshes (one draw per node-instanced primitive) ----
  for (int ni : a.order) {
    const cgltf_node& n = d->nodes[ni];
    if (!n.mesh) continue;
    const bool skinned = n.skin != nullptr;
    uint32_t base;
    if (skinned) {
      base = a.skins[n.skin - d->skins].palette_base;
    } else {
      base = palette++;
      a.rigid_slots.push_back({ni, base});
    }
    for (size_t pi = 0; pi < n.mesh->primitives_count; ++pi) {
      const cgltf_primitive& p = n.mesh->primitives[pi];
      if (p.type != cgltf_primitive_type_triangles) continue;
      const cgltf_accessor *pos = nullptr, *nrm = nullptr, *jnt = nullptr, *wgt = nullptr;
      const cgltf_accessor* uv[4] = {};
      for (size_t k = 0; k < p.attributes_count; ++k) {
        const cgltf_attribute& at = p.attributes[k];
        switch (at.type) {
          case cgltf_attribute_type_position: pos = at.data; break;
          case cgltf_attribute_type_normal: nrm = at.data; break;
          case cgltf_attribute_type_texcoord: if (at.index < 4) uv[at.index] = at.data; break;
          case cgltf_attribute_type_joints: if (at.index == 0) jnt = at.data; break;
          case cgltf_attribute_type_weights: if (at.index == 0) wgt = at.data; break;
          default: break;
        }
      }
      if (!pos || pos->count == 0) continue;
      int mat = p.material ? int(p.material - d->materials) : default_material;
      int uv_set = 0;
      if (p.material && p.material->has_pbr_metallic_roughness)
        uv_set = std::clamp(p.material->pbr_metallic_roughness.base_color_texture.texcoord, 0, 3);
      const cgltf_accessor* uva = uv[uv_set] ? uv[uv_set] : uv[0];

      const size_t v0 = a.vertices.size(), i0 = a.indices.size();
      a.vertices.resize(v0 + pos->count);
      for (size_t v = 0; v < pos->count; ++v) {
        Vertex& o = a.vertices[v0 + v];
        std::memset(&o, 0, sizeof(o));
        cgltf_accessor_read_float(pos, v, o.pos, 3);
        if (nrm) cgltf_accessor_read_float(nrm, v, o.normal, 3);
        if (uva) cgltf_accessor_read_float(uva, v, o.uv, 2);
        if (skinned && jnt && wgt) {
          cgltf_accessor_read_uint(jnt, v, o.joints, 4);
          cgltf_accessor_read_float(wgt, v, o.weights, 4);
          const uint32_t nj = uint32_t(n.skin->joints_count);
          float sum = 0;
          for (int k = 0; k < 4; ++k) {
            if (o.joints[k] >= nj) { o.joints[k] = 0; o.weights[k] = 0; }
            sum += o.weights[k];
          }
          if (sum > 1e-6f) for (int k = 0; k < 4; ++k) o.weights[k] /= sum;
          else { o.weights[0] = 1; o.joints[0] = 0; }
        } else {
          o.weights[0] = 1.f;  // rigid: palette slot of the node (or skin joint 0)
        }
      }
      if (p.indices) {
        for (size_t k = 0; k < p.indices->count; ++k) a.indices.push_back(uint32_t(cgltf_accessor_read_index(p.indices, k)));
      } else {
        for (size_t k = 0; k < pos->count; ++k) a.indices.push_back(uint32_t(k));
      }
      a.indices.resize(i0 + (a.indices.size() - i0) / 3 * 3);
      if (!nrm) {
        std::vector<uint32_t> local(a.indices.begin() + i0, a.indices.end());
        compute_normals(a.vertices, v0, local, 0);
      }
      Draw dr;
      dr.first_index = uint32_t(i0);
      dr.index_count = uint32_t(a.indices.size() - i0);
      dr.vertex_offset = int32_t(v0);
      dr.material = mat;
      dr.palette_base = base;
      {
        glm::vec3 c(0.f);
        std::vector<float> wsum(skinned && n.skin ? n.skin->joints_count : 1, 0.f);
        for (size_t v = 0; v < pos->count; ++v) {
          const Vertex& o = a.vertices[v0 + v];
          c += glm::make_vec3(o.pos);
          for (int k = 0; k < 4; ++k)
            if (o.joints[k] < wsum.size()) wsum[o.joints[k]] += o.weights[k];
        }
        dr.center = c / float(pos->count);
        dr.center_joint = uint32_t(std::max_element(wsum.begin(), wsum.end()) - wsum.begin());
      }
      if (dr.index_count) a.draws.push_back(dr);
    }
  }
  a.palette_size = std::max(palette, 1u);

  // ---- bind-pose bounds (CPU skinning of the rest pose) ----
  {
    Pose p;
    rest_pose(a, p);
    std::vector<glm::mat4> pal;
    build_palette(a, p, pal);
    glm::vec3 lo(1e30f), hi(-1e30f);
    for (const Draw& dr : a.draws) {
      for (uint32_t k = 0; k < dr.index_count; ++k) {
        const Vertex& v = a.vertices[dr.vertex_offset + a.indices[dr.first_index + k]];
        glm::mat4 m(0.f);
        for (int j = 0; j < 4; ++j) m += v.weights[j] * pal[dr.palette_base + v.joints[j]];
        glm::vec3 w = glm::vec3(m * glm::vec4(glm::make_vec3(v.pos), 1.f));
        lo = glm::min(lo, w);
        hi = glm::max(hi, w);
      }
    }
    for (int n : a.skeleton) {
      glm::vec3 w = glm::vec3(p.global[n][3]);
      lo = glm::min(lo, w);
      hi = glm::max(hi, w);
    }
    if (lo.x > hi.x) { lo = glm::vec3(-1); hi = glm::vec3(1); }
    a.bounds_min = lo;
    a.bounds_max = hi;
  }

  convert_animations(d, a, clip_fps, a.embedded_clips);
  return true;
}

void compute_globals(const Asset& a, Pose& p) {
  p.global.resize(a.nodes.size());
  for (int n : a.order) {
    glm::mat4 local = glm::translate(glm::mat4(1.f), p.t[n]) * glm::mat4_cast(p.r[n]) * glm::scale(glm::mat4(1.f), p.s[n]);
    int par = a.nodes[n].parent;
    p.global[n] = par >= 0 ? p.global[par] * local : local;
  }
}

void rest_pose(const Asset& a, Pose& p) {
  const size_t n = a.nodes.size();
  p.t.resize(n); p.r.resize(n); p.s.resize(n);
  for (size_t i = 0; i < n; ++i) {
    p.t[i] = a.nodes[i].t;
    p.r[i] = a.nodes[i].r;
    p.s[i] = a.nodes[i].s;
  }
  compute_globals(a, p);
}

void build_palette(const Asset& a, const Pose& p, std::vector<glm::mat4>& pal) {
  pal.assign(a.palette_size, glm::mat4(1.f));
  for (const Skin& s : a.skins)
    for (size_t j = 0; j < s.joints.size(); ++j)
      pal[s.palette_base + j] = p.global[s.joints[j]] * s.inverse_bind[j];
  for (auto& [node, slot] : a.rigid_slots) pal[slot] = p.global[node];
}

std::vector<int> bind_clip(const Asset& a, const um::MotionClip& clip) {
  std::vector<int> b(clip.tracks.size(), -1);
  for (size_t i = 0; i < clip.tracks.size(); ++i) {
    auto it = a.node_by_name.find(clip.tracks[i].node_name);
    if (it != a.node_by_name.end()) b[i] = it->second;
  }
  return b;
}

void sample_clip(const Asset& a, const um::MotionClip& clip, const std::vector<int>& binding, float f,
                 bool loop, Pose& p) {
  // start from rest so nodes not in the clip keep their rest value
  for (size_t i = 0; i < a.nodes.size(); ++i) {
    p.t[i] = a.nodes[i].t;
    p.r[i] = a.nodes[i].r;
    p.s[i] = a.nodes[i].s;
  }
  const int nf = clip.num_frames;
  if (nf > 0) {
    int f0, f1;
    float u;
    if (loop) {
      f = std::fmod(f, float(nf));
      if (f < 0) f += float(nf);
      f0 = int(std::floor(f));
      u = f - float(f0);
      f0 = std::min(f0, nf - 1);
      f1 = (f0 + 1) % nf;
    } else {
      f = std::clamp(f, 0.f, float(nf - 1));
      f0 = int(std::floor(f));
      f1 = std::min(f0 + 1, nf - 1);
      u = f - float(f0);
    }
    for (size_t ti = 0; ti < clip.tracks.size() && ti < binding.size(); ++ti) {
      int n = binding[ti];
      if (n < 0) continue;
      const um::NodeTrack& tr = clip.tracks[ti];
      if (tr.translation.size() >= size_t(nf) * 3) {
        glm::vec3 x0 = glm::make_vec3(&tr.translation[size_t(f0) * 3]);
        glm::vec3 x1 = glm::make_vec3(&tr.translation[size_t(f1) * 3]);
        p.t[n] = glm::mix(x0, x1, u);
      }
      if (tr.scale.size() >= size_t(nf) * 3) {
        glm::vec3 x0 = glm::make_vec3(&tr.scale[size_t(f0) * 3]);
        glm::vec3 x1 = glm::make_vec3(&tr.scale[size_t(f1) * 3]);
        p.s[n] = glm::mix(x0, x1, u);
      }
      if (tr.rotation.size() >= size_t(nf) * 4) {
        const float* q0 = &tr.rotation[size_t(f0) * 4];
        const float* q1 = &tr.rotation[size_t(f1) * 4];
        glm::vec4 a0(q0[0], q0[1], q0[2], q0[3]), a1(q1[0], q1[1], q1[2], q1[3]);
        if (glm::dot(a0, a1) < 0) a1 = -a1;  // shortest arc
        glm::vec4 q = glm::mix(a0, a1, u);
        float l = glm::length(q);
        if (l > 1e-8f) {
          q /= l;
          p.r[n] = glm::quat(q.w, q.x, q.y, q.z);
        }
      }
    }
  }
  compute_globals(a, p);
}

bool has_debug_skeleton(const um::MotionClip& c) {
  return c.num_joints > 0 && c.num_frames > 0 && c.parents.size() == size_t(c.num_joints) &&
         c.joint_positions.size() >= size_t(c.num_frames) * size_t(c.num_joints) * 3;
}

void sample_debug_skeleton(const um::MotionClip& c, float f, bool loop, std::vector<glm::vec3>& out) {
  out.clear();
  if (!has_debug_skeleton(c)) return;
  const int nf = c.num_frames, nj = c.num_joints;
  int f0, f1;
  float u;
  if (loop) {  // same rules as sample_clip
    f = std::fmod(f, float(nf));
    if (f < 0) f += float(nf);
    f0 = std::min(int(std::floor(f)), nf - 1);
    u = f - float(f0);
    f1 = (f0 + 1) % nf;
  } else {
    f = std::clamp(f, 0.f, float(nf - 1));
    f0 = int(std::floor(f));
    f1 = std::min(f0 + 1, nf - 1);
    u = f - float(f0);
  }
  out.resize(size_t(nj));
  const float* p0 = &c.joint_positions[size_t(f0) * nj * 3];
  const float* p1 = &c.joint_positions[size_t(f1) * nj * 3];
  for (int j = 0; j < nj; ++j)
    out[j] = glm::mix(glm::make_vec3(p0 + 3 * j), glm::make_vec3(p1 + 3 * j), u);
}

}  // namespace um::viewer

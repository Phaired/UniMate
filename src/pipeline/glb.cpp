#include "pipeline/glb.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>

#include "pipeline/math3.h"

namespace um::pipeline {

namespace {
constexpr uint32_t kMagic = 0x46546C67;  // "glTF"
constexpr uint32_t kJson = 0x4E4F534A;   // "JSON"
constexpr uint32_t kBin = 0x004E4942;    // "BIN\0"

[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("glb: " + m); }

uint32_t rd32(const uint8_t* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);
  return v;
}
void wr32(std::vector<uint8_t>& out, uint32_t v) {
  uint8_t b[4];
  std::memcpy(b, &v, 4);
  out.insert(out.end(), b, b + 4);
}
void pad4(std::vector<uint8_t>& v, uint8_t fill) {
  while (v.size() % 4) v.push_back(fill);
}
}  // namespace

GlbFile read_glb(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fail("cannot open " + path);
  std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (b.size() < 20 || rd32(b.data()) != kMagic) fail(path + ": not a GLB file");
  if (rd32(b.data() + 4) != 2) fail(path + ": unsupported GLB version");
  size_t total = std::min<size_t>(rd32(b.data() + 8), b.size());
  GlbFile g;
  size_t off = 12;
  bool have_json = false;
  while (off + 8 <= total) {
    uint32_t len = rd32(b.data() + off), type = rd32(b.data() + off + 4);
    off += 8;
    if (off + len > total) fail(path + ": truncated chunk");
    if (type == kJson && !have_json) {
      g.json = nlohmann::json::parse(b.begin() + off, b.begin() + off + len);
      have_json = true;
    } else if (type == kBin && !g.has_bin) {
      g.bin.assign(b.begin() + off, b.begin() + off + len);
      g.has_bin = true;
    }
    off += len;
  }
  if (!have_json) fail(path + ": no JSON chunk");
  return g;
}

void write_glb(const std::string& path, const GlbFile& g) {
  std::string js = g.json.dump();
  std::vector<uint8_t> jchunk(js.begin(), js.end());
  pad4(jchunk, ' ');
  std::vector<uint8_t> bchunk = g.bin;
  pad4(bchunk, 0);
  std::vector<uint8_t> out;
  size_t total = 12 + 8 + jchunk.size() + (g.has_bin ? 8 + bchunk.size() : 0);
  wr32(out, kMagic);
  wr32(out, 2);
  wr32(out, static_cast<uint32_t>(total));
  wr32(out, static_cast<uint32_t>(jchunk.size()));
  wr32(out, kJson);
  out.insert(out.end(), jchunk.begin(), jchunk.end());
  if (g.has_bin) {
    wr32(out, static_cast<uint32_t>(bchunk.size()));
    wr32(out, kBin);
    out.insert(out.end(), bchunk.begin(), bchunk.end());
  }
  std::ofstream f(path, std::ios::binary);
  if (!f) fail("cannot write " + path);
  f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
  if (!f) fail("write failed: " + path);
}

std::vector<GlbNode> glb_nodes(const nlohmann::json& gltf) {
  std::vector<GlbNode> nodes;
  if (!gltf.contains("nodes")) return nodes;
  const auto& jn = gltf["nodes"];
  nodes.resize(jn.size());
  for (size_t i = 0; i < jn.size(); ++i) {
    const auto& n = jn[i];
    GlbNode& g = nodes[i];
    g.name = n.value("name", std::string());
    if (n.contains("children"))
      for (const auto& c : n["children"]) {
        size_t ci = c.get<size_t>();
        if (ci < nodes.size()) nodes[ci].parent = static_cast<int>(i);
      }
    if (n.contains("matrix")) {
      const auto& m = n["matrix"];  // column-major
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) g.local[r * 4 + c] = m[c * 4 + r].get<double>();
      // decompose for completeness
      m3::Mat3 L = m3::linear(g.local);
      for (int c = 0; c < 3; ++c)
        g.scale[c] = std::sqrt(L[c] * L[c] + L[3 + c] * L[3 + c] + L[6 + c] * L[6 + c]);
      m3::Mat3 R = L;
      for (int c = 0; c < 3; ++c)
        for (int r = 0; r < 3; ++r) R[r * 3 + c] /= (g.scale[c] != 0 ? g.scale[c] : 1);
      m3::Quat q = m3::mat_to_quat(R);
      g.rotation = {q[1], q[2], q[3], q[0]};
      g.translation = {g.local[3], g.local[7], g.local[11]};
    } else {
      if (n.contains("translation"))
        for (int k = 0; k < 3; ++k) g.translation[k] = n["translation"][k].get<double>();
      if (n.contains("rotation"))
        for (int k = 0; k < 4; ++k) g.rotation[k] = n["rotation"][k].get<double>();
      if (n.contains("scale"))
        for (int k = 0; k < 3; ++k) g.scale[k] = n["scale"][k].get<double>();
      m3::Quat q{g.rotation[3], g.rotation[0], g.rotation[1], g.rotation[2]};
      double qn = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
      if (qn > 0)
        for (double& v : q) v /= qn;
      m3::Mat3 R = m3::quat_to_mat(q);
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) R[r * 3 + c] *= g.scale[c];
      g.local = m3::compose(R, {g.translation[0], g.translation[1], g.translation[2]});
    }
  }
  std::vector<int> state(nodes.size(), 0);
  std::function<void(size_t)> world = [&](size_t i) {
    if (state[i] == 2) return;
    if (state[i] == 1) fail("node hierarchy has a cycle");
    state[i] = 1;
    if (nodes[i].parent >= 0) {
      world(static_cast<size_t>(nodes[i].parent));
      nodes[i].rest_world = m3::mul(nodes[nodes[i].parent].rest_world, nodes[i].local);
    } else {
      nodes[i].rest_world = nodes[i].local;
    }
    state[i] = 2;
  };
  for (size_t i = 0; i < nodes.size(); ++i) world(i);
  return nodes;
}

int find_node(const nlohmann::json& gltf, const std::vector<GlbNode>& nodes,
              const std::string& name) {
  std::vector<int> hits;
  for (size_t i = 0; i < nodes.size(); ++i)
    if (nodes[i].name == name) hits.push_back(static_cast<int>(i));
  if (hits.empty()) return -1;
  if (hits.size() > 1 && gltf.contains("skins")) {
    for (int h : hits)
      for (const auto& s : gltf["skins"])
        if (s.contains("joints"))
          for (const auto& j : s["joints"])
            if (j.get<int>() == h) return h;
  }
  return hits.front();
}

void export_glb(const std::string& src_glb, const um::MotionClip& clip,
                const std::string& out_glb, const ExportOptions& opt) {
  if (clip.num_frames <= 0) fail("clip has no frames");
  if (!(clip.fps > 0)) fail("clip fps must be > 0");
  GlbFile g = read_glb(src_glb);
  nlohmann::json& j = g.json;
  auto nodes = glb_nodes(j);

  if (!j.contains("buffers") || j["buffers"].empty()) {
    j["buffers"] = nlohmann::json::array({nlohmann::json{{"byteLength", 0}}});
    g.has_bin = true;
    g.bin.clear();
  } else if (j["buffers"][0].contains("uri") || !g.has_bin) {
    fail(src_glb + ": buffer 0 is not the GLB BIN chunk (external/data-URI buffers are not supported)");
  }
  if (!j.contains("bufferViews")) j["bufferViews"] = nlohmann::json::array();
  if (!j.contains("accessors")) j["accessors"] = nlohmann::json::array();
  if (opt.replace_animations || !j.contains("animations")) j["animations"] = nlohmann::json::array();

  const int T = clip.num_frames;
  auto add_accessor = [&](const std::vector<float>& data, int comps, bool minmax) {
    pad4(g.bin, 0);
    size_t offset = g.bin.size();
    const uint8_t* p = reinterpret_cast<const uint8_t*>(data.data());
    g.bin.insert(g.bin.end(), p, p + data.size() * sizeof(float));
    nlohmann::json bv{{"buffer", 0},
                      {"byteOffset", offset},
                      {"byteLength", data.size() * sizeof(float)}};
    j["bufferViews"].push_back(bv);
    int bv_index = static_cast<int>(j["bufferViews"].size()) - 1;
    static const char* kTypes[] = {"", "SCALAR", "VEC2", "VEC3", "VEC4"};
    nlohmann::json acc{{"bufferView", bv_index},
                       {"componentType", 5126},
                       {"count", data.size() / comps},
                       {"type", kTypes[comps]}};
    if (minmax) {
      std::vector<float> mn(comps, INFINITY), mx(comps, -INFINITY);
      for (size_t i = 0; i < data.size(); ++i) {
        mn[i % comps] = std::min(mn[i % comps], data[i]);
        mx[i % comps] = std::max(mx[i % comps], data[i]);
      }
      acc["min"] = mn;
      acc["max"] = mx;
    }
    j["accessors"].push_back(acc);
    return static_cast<int>(j["accessors"].size()) - 1;
  };

  std::vector<float> times(T);
  for (int f = 0; f < T; ++f) times[f] = static_cast<float>(f / static_cast<double>(clip.fps));
  int time_acc = -1;

  nlohmann::json anim;
  anim["name"] = opt.animation_name.empty() ? (clip.name.empty() ? "UniMate" : clip.name)
                                            : opt.animation_name;
  anim["channels"] = nlohmann::json::array();
  anim["samplers"] = nlohmann::json::array();
  auto add_channel = [&](int node, const char* path, const std::vector<float>& v, int comps) {
    if (v.empty()) return;
    if (static_cast<int>(v.size()) != T * comps)
      fail(std::string("track channel '") + path + "' has " + std::to_string(v.size()) +
           " values, expected " + std::to_string(T * comps));
    if (time_acc < 0) time_acc = add_accessor(times, 1, true);
    int out = add_accessor(v, comps, false);
    anim["samplers"].push_back({{"input", time_acc}, {"output", out}, {"interpolation", "LINEAR"}});
    int s = static_cast<int>(anim["samplers"].size()) - 1;
    anim["channels"].push_back({{"sampler", s}, {"target", {{"node", node}, {"path", path}}}});
  };
  for (const auto& tr : clip.tracks) {
    int node = find_node(j, nodes, tr.node_name);
    if (node < 0) continue;
    add_channel(node, "translation", tr.translation, 3);
    add_channel(node, "rotation", tr.rotation, 4);
    add_channel(node, "scale", tr.scale, 3);
  }
  if (!anim["channels"].empty()) j["animations"].push_back(anim);
  if (j["animations"].empty()) j.erase("animations");
  pad4(g.bin, 0);
  j["buffers"][0]["byteLength"] = g.bin.size();
  write_glb(out_glb, g);
}

}  // namespace um::pipeline

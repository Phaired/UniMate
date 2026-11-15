#include "app/assets.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#ifndef UM_ROOT_DIR
#define UM_ROOT_DIR "."
#endif

namespace fs = std::filesystem;

namespace um::app {
namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return s;
}

std::string norm(const fs::path& p) {
  std::error_code ec;
  fs::path a = fs::weakly_canonical(p, ec);
  if (ec) a = p;
  return a.make_preferred().string();
}

// Reads only the JSON header of a UMTF file and returns its "meta".
nlohmann::json umtf_meta(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  char magic[4];
  uint32_t version = 0;
  uint64_t n = 0;
  in.read(magic, 4);
  in.read(reinterpret_cast<char*>(&version), 4);
  in.read(reinterpret_cast<char*>(&n), 8);
  if (!in || std::string(magic, 4) != "UMTF" || n > (64u << 20))
    throw std::runtime_error(path.string() + " is not a UMTF file");
  std::string js(n, '\0');
  in.read(js.data(), std::streamsize(n));
  auto j = nlohmann::json::parse(js);
  return j.value("meta", nlohmann::json::object());
}

std::string first_glb_in(const fs::path& dir) {
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return {};
  for (auto& e : fs::directory_iterator(dir, ec))
    if (lower(e.path().extension().string()) == ".glb") return e.path().string();
  return {};
}

// GLB of a bundle: meta.source (relative to the repo root, or absolute) +
// meta.glb; then <repo>/reference/UniMate/assets/examples/<name>/.
std::string glb_of_bundle(const fs::path& bundle, const nlohmann::json& meta) {
  const std::string name = bundle.stem().string();
  std::vector<fs::path> dirs;
  if (meta.contains("source") && meta["source"].is_string()) {
    fs::path src = meta["source"].get<std::string>();
    if (src.is_absolute()) dirs.push_back(src);
    else {
      dirs.push_back(fs::path(repo_root()) / src);
      dirs.push_back(bundle.parent_path() / src);
    }
  }
  dirs.push_back(fs::path(repo_root()) / "reference" / "UniMate" / "assets" / "examples" / name);
  const std::string glb_name = meta.contains("glb") && meta["glb"].is_string() ? meta["glb"].get<std::string>() : "";
  for (const auto& d : dirs) {
    if (!glb_name.empty() && fs::exists(d / glb_name)) return norm(d / glb_name);
    std::string g = first_glb_in(d);
    if (!g.empty()) return norm(g);
  }
  return {};
}

std::string convert_hint(const std::string& what) {
  return "  convert it first:  .venv\\Scripts\\python.exe tools/python/pipeline_convert_bundle.py " + what +
         "\n  (the directory must hold the rig_preprocess output: cond.npy, summary.json, "
         "annotation.json and the .glb)";
}

}  // namespace

std::string repo_root() { return UM_ROOT_DIR; }

std::string default_models_dir() { return norm(fs::path(repo_root()) / "models" / "converted"); }

std::string resolve_models_dir(const std::string& cli_value) {
  if (!cli_value.empty()) return norm(cli_value);
  if (const char* e = std::getenv("UNIMATE_MODELS"); e && *e) return norm(e);
  return default_models_dir();
}

ModelPaths model_paths(const std::string& dir) {
  ModelPaths p;
  p.dir = dir;
  p.t5 = norm(fs::path(dir) / "flan-t5-base.umtf");
  p.denoiser = norm(fs::path(dir) / "unimate_uniml3d_f60_v3.umtf");
  p.stats = norm(fs::path(dir) / "unimate_uniml3d_f60_v3_stats.umtf");
  struct Need { const std::string* path; const char* how; };
  for (Need n : {Need{&p.t5, "tools/python/t5_export.py"},
                 Need{&p.denoiser, "tools/python/denoiser_export.py"},
                 Need{&p.stats, "tools/python/pipeline_convert_stats.py"}}) {
    if (!fs::exists(*n.path))
      throw std::runtime_error("missing model file " + *n.path + "\n  create it with: .venv\\Scripts\\python.exe " +
                               n.how + "  (see README.md), or point --models / UNIMATE_MODELS elsewhere");
  }
  return p;
}

std::vector<std::string> list_bundles(const std::string& models_dir) {
  std::vector<std::string> out;
  std::error_code ec;
  fs::path d = fs::path(models_dir) / "bundles";
  if (!fs::is_directory(d, ec)) return out;
  for (auto& e : fs::directory_iterator(d, ec))
    if (lower(e.path().extension().string()) == ".umtf") out.push_back(e.path().stem().string());
  std::sort(out.begin(), out.end());
  return out;
}

AssetPaths resolve_asset(const std::string& spec, const std::string& models_dir) {
  if (spec.empty()) throw std::runtime_error("no asset given");
  const fs::path bundles = fs::path(models_dir) / "bundles";
  const fs::path p(spec);
  std::error_code ec;
  AssetPaths out;

  auto from_bundle = [&](const fs::path& umtf, const std::string& glb_hint) {
    out.bundle_umtf = norm(umtf);
    out.name = umtf.stem().string();
    out.glb = glb_hint.empty() ? glb_of_bundle(umtf, umtf_meta(umtf)) : norm(glb_hint);
    if (out.glb.empty())
      throw std::runtime_error("bundle " + out.bundle_umtf +
                               ": cannot find its GLB (meta 'source' / 'glb'); pass the .glb path instead");
    return out;
  };

  // 1. explicit bundle file
  if (lower(p.extension().string()) == ".umtf" && fs::is_regular_file(p, ec)) return from_bundle(p, "");

  // 2. GLB file or asset directory -> bundle by directory name / stem / meta
  const bool is_glb = lower(p.extension().string()) == ".glb" && fs::is_regular_file(p, ec);
  const bool is_dir = fs::is_directory(p, ec);
  if (is_glb || is_dir) {
    const fs::path glb = is_glb ? p : fs::path(first_glb_in(p));
    if (glb.empty()) throw std::runtime_error("no .glb file in directory " + spec);
    const fs::path dir = is_glb ? glb.parent_path() : p;
    std::vector<std::string> candidates = {dir.filename().string(), lower(dir.filename().string()),
                                           glb.stem().string(), lower(glb.stem().string())};
    for (const auto& c : candidates) {
      if (c.empty()) continue;
      fs::path b = bundles / (c + ".umtf");
      if (fs::is_regular_file(b, ec)) return from_bundle(b, glb.string());
    }
    // any bundle whose meta points at this GLB
    const std::string want = lower(norm(glb));
    for (const auto& name : list_bundles(models_dir)) {
      fs::path b = bundles / (name + ".umtf");
      try {
        if (lower(glb_of_bundle(b, umtf_meta(b))) == want) return from_bundle(b, glb.string());
      } catch (...) {
      }
    }
    throw std::runtime_error("no converted bundle for " + norm(glb) + " in " + norm(bundles) + "\n" +
                             convert_hint(norm(dir)));
  }

  // 3. bundle name
  fs::path b = bundles / (spec + ".umtf");
  if (fs::is_regular_file(b, ec)) return from_bundle(b, "");
  b = bundles / (lower(spec) + ".umtf");
  if (fs::is_regular_file(b, ec)) return from_bundle(b, "");

  std::string avail;
  for (const auto& n : list_bundles(models_dir)) avail += (avail.empty() ? "" : ", ") + n;
  throw std::runtime_error("unknown asset '" + spec + "': not a file, directory or bundle name in " + norm(bundles) +
                           " (available: " + (avail.empty() ? "none" : avail) + ")\n" +
                           convert_hint("<asset_dir>   or   --examples"));
}

}  // namespace um::app

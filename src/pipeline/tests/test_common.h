// Shared helpers for the um_pipeline tests (golden files from
// tools/python/pipeline_golden.py, converted bundles from
// tools/python/pipeline_convert_*.py). A test whose inputs are missing exits
// with 77 (CTest SKIP_RETURN_CODE).
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "core/umtf.h"

namespace pt {

inline std::string root() { return UM_ROOT_DIR; }
inline std::string golden(const std::string& f) { return root() + "/tests/golden/pipeline/" + f; }
inline std::string bundle_umtf(const std::string& a) { return root() + "/models/converted/bundles/" + a + ".umtf"; }
inline std::string stats_umtf() { return root() + "/models/converted/unimate_uniml3d_f60_v3_stats.umtf"; }
inline std::string config_json() {
  return root() + "/models/unimate/unimate_uniml3d_f60_v3/config.json";
}
inline std::string glb_of(const std::string& asset) {
  namespace fs = std::filesystem;
  fs::path dir = fs::path(root()) / "reference" / "UniMate" / "assets" / "examples" / asset;
  if (!fs::is_directory(dir)) return {};
  for (auto& e : fs::directory_iterator(dir))
    if (e.path().extension() == ".glb") return e.path().string();
  return {};
}
inline bool exists(const std::string& p) { return !p.empty() && std::filesystem::exists(p); }

inline int g_failures = 0;
#define PT_CHECK(cond, ...)                                       \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond); \
      std::printf(__VA_ARGS__);                                   \
      std::printf("\n");                                          \
      ++pt::g_failures;                                           \
    }                                                             \
  } while (0)

inline void require_or_skip(const std::vector<std::string>& files) {
  for (const auto& f : files)
    if (!exists(f)) {
      std::printf("SKIP: missing %s\n", f.c_str());
      std::exit(77);
    }
}

inline int finish(const char* name) {
  if (g_failures) {
    std::printf("%s: %d failure(s)\n", name, g_failures);
    return 1;
  }
  std::printf("%s: OK\n", name);
  return 0;
}

inline double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return INFINITY;
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(double(a[i]) - double(b[i])));
  return m;
}
inline double max_abs(const std::vector<float>& a) {
  double m = 0;
  for (float v : a) m = std::max(m, std::abs(double(v)));
  return m;
}

}  // namespace pt

// Tiny assert harness for um_core tests.
#pragma once
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

inline int g_failures = 0;
inline int g_checks = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    ++g_checks;                                                                  \
    if (!(cond)) {                                                               \
      ++g_failures;                                                              \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);              \
    }                                                                            \
  } while (0)

#define CHECK_THROWS(expr)                                                       \
  do {                                                                           \
    ++g_checks;                                                                  \
    bool threw_ = false;                                                         \
    try { (void)(expr); } catch (const std::exception&) { threw_ = true; }       \
    if (!threw_) {                                                               \
      ++g_failures;                                                              \
      std::printf("  FAIL %s:%d: expected throw: %s\n", __FILE__, __LINE__, #expr); \
    }                                                                            \
  } while (0)

inline bool near_vec(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-5f) {
  if (a.size() != b.size()) {
    std::printf("  size mismatch %zu vs %zu\n", a.size(), b.size());
    return false;
  }
  for (size_t i = 0; i < a.size(); ++i) {
    float d = std::fabs(a[i] - b[i]);
    if (!(d <= tol * (1.f + std::fabs(b[i])))) {
      std::printf("  mismatch at %zu: got %.7g want %.7g\n", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}
#define CHECK_NEAR(...) CHECK(near_vec(__VA_ARGS__))

#define RUN(fn)                                                                  \
  do {                                                                           \
    int before_ = g_failures;                                                    \
    try { fn(); } catch (const std::exception& e) {                              \
      ++g_failures; std::printf("  EXCEPTION in %s: %s\n", #fn, e.what());       \
    }                                                                            \
    std::printf("[%s] %s\n", g_failures == before_ ? " ok " : "FAIL", #fn);      \
  } while (0)

inline int finish() {
  std::printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}

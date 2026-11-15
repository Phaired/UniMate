// Parity test: every um::Backend op on the Vulkan backend vs the CPU reference,
// on random inputs over many shapes (broadcast corner cases, rank-0, rank 6,
// size-1 dims, large tensors, negative axes). Prints per-op max abs / rel error.
//
// rel error of an element = |vk - cpu| / (|cpu| + 1e-3 * rms(cpu)), i.e. relative
// except for values far below the tensor's typical magnitude.
//
// Exit code 0 = all ops within tolerance (or no Vulkan device: skipped).
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "vk/vk_backend.h"

using namespace um;

namespace {

struct Stat {
  int cases = 0;
  double max_abs = 0, max_rel = 0, max_norm = 0;
  double tol = 1e-5, ntol = 0;
  std::string worst;
  int failures = 0;
};
std::map<std::string, Stat> g_stats;
std::vector<std::string> g_order;
int g_errors = 0;

std::mt19937 rng(12345);

std::vector<float> rnd(int64_t n, float lo = -2.f, float hi = 2.f) {
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(static_cast<size_t>(n));
  for (auto& x : v) x = d(rng);
  return v;
}

struct Pair {
  Tensor c, g;
};

struct Ctx {
  Backend* cpu;
  Backend* vk;
  Pair make(const Shape& s, float lo = -2.f, float hi = 2.f) {
    auto v = rnd(shape_numel(s), lo, hi);
    return {cpu->from_host(v.data(), s), vk->from_host(v.data(), s)};
  }
  Pair from(const std::vector<float>& v, const Shape& s) {
    return {cpu->from_host(v.data(), s), vk->from_host(v.data(), s)};
  }
};

// Normwise tolerance (fraction of max|ref|) for ops that accumulate: their
// outputs can cancel to ~0, where an elementwise relative bound is meaningless.
double norm_tol(const std::string& op) {
  if (op == "matmul" || op == "linear") return 1e-5;
  if (op.rfind("fp16", 0) == 0) return 3e-3;
  if (op == "attention") return 2e-5;
  if (op == "softmax" || op == "layer_norm" || op == "rms_norm" || op == "r:sum" || op == "r:mean") return 1e-6;
  if (op.rfind("u:", 0) == 0) return 1e-7;
  return 0;
}

// Pass if every element has |d| <= tol*|ref| + ntol*max|ref|.
void compare(const std::string& op, const std::string& desc, const Tensor& ref, const Tensor& got,
             double tol) {
  Stat& st = g_stats[op];
  if (st.cases == 0) g_order.push_back(op);
  st.tol = tol;
  const double ntol = norm_tol(op);
  st.ntol = ntol;
  ++st.cases;
  if (ref.shape() != got.shape()) {
    std::printf("  FAIL %s %s: shape %s vs %s\n", op.c_str(), desc.c_str(), shape_str(ref.shape()).c_str(),
                shape_str(got.shape()).c_str());
    ++st.failures;
    return;
  }
  auto a = ref.to_vector(), b = got.to_vector();
  double ss = 0;
  for (float x : a)
    if (std::isfinite(x)) ss += double(x) * x;
  const double rms = a.empty() ? 0 : std::sqrt(ss / a.size());
  double mabs = 0, mrel = 0, maxref = 0;
  bool bad = false, over = false;
  for (float x : a)
    if (std::isfinite(x)) maxref = std::max(maxref, double(std::fabs(x)));
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = a[i], y = b[i];
    if (std::isnan(x) || std::isnan(y)) {
      if (std::isnan(x) != std::isnan(y)) bad = true;
      continue;
    }
    if (std::isinf(x) || std::isinf(y)) {
      if (x != y) bad = true;
      continue;
    }
    const double d = std::fabs(x - y);
    const double r = d / (std::fabs(x) + 1e-3 * rms + 1e-30);
    mabs = std::max(mabs, d);
    mrel = std::max(mrel, r);
    if (d > tol * std::fabs(x) + ntol * maxref) over = true;
  }
  const double nrel = maxref > 0 ? mabs / maxref : mabs;
  if (nrel >= st.max_norm) st.worst = desc;
  st.max_abs = std::max(st.max_abs, mabs);
  st.max_rel = std::max(st.max_rel, mrel);
  st.max_norm = std::max(st.max_norm, nrel);
  if (bad || over) {
    ++st.failures;
    std::printf("  FAIL %-12s %-44s abs %.3g rel %.3g norm %.3g%s\n", op.c_str(), desc.c_str(), mabs, mrel,
                nrel, bad ? " (nan/inf mismatch)" : "");
  }
}

std::string S(const Shape& s) { return shape_str(s); }

}  // namespace

// Runs fc on CPU tensors and fg on VK tensors, compares.
#define CHECK(op, desc, tol, expr_cpu, expr_vk)                                             \
  do {                                                                                      \
    try {                                                                                   \
      Tensor _r = (expr_cpu);                                                               \
      Tensor _g = (expr_vk);                                                                \
      compare(op, desc, _r, _g, tol);                                                       \
    } catch (const std::exception& e) {                                                     \
      std::printf("  ERROR %s %s: %s\n", std::string(op).c_str(), std::string(desc).c_str(), e.what()); \
      ++g_errors;                                                                           \
    }                                                                                       \
  } while (0)

static void suite(Ctx& C, bool quick) {
  Backend& cpu = *C.cpu;
  Backend& vk = *C.vk;

  const std::vector<Shape> ushapes = quick ? std::vector<Shape>{{}, {7}, {3, 1, 5}}
                                           : std::vector<Shape>{{}, {1}, {7}, {3, 1, 5}, {2, 3, 1, 4, 5, 2},
                                                                {1000003}, {64, 4096}};
  // ---- unary --------------------------------------------------------------
  const char* unames[] = {"neg", "abs", "square", "sqrt", "rsqrt", "reciprocal", "exp", "log", "sin",
                          "cos", "tanh", "sigmoid", "relu", "silu", "gelu", "gelu_tanh", "gelu_new"};
  for (int op = 0; op <= static_cast<int>(UnaryOp::GeluNew); ++op) {
    UnaryOp u = static_cast<UnaryOp>(op);
    float lo = -6.f, hi = 6.f;
    if (u == UnaryOp::Sqrt || u == UnaryOp::Rsqrt || u == UnaryOp::Log) lo = 1e-3f, hi = 100.f;
    if (u == UnaryOp::Exp) lo = -80.f, hi = 80.f;
    if (u == UnaryOp::Sin || u == UnaryOp::Cos) lo = -1000.f, hi = 1000.f;
    if (u == UnaryOp::Reciprocal) lo = 0.01f, hi = 10.f;
    for (const Shape& s : ushapes) {
      Pair x = C.make(s, lo, hi);
      CHECK(std::string("u:") + unames[op], S(s), 2e-6, cpu.unary(u, x.c), vk.unary(u, x.g));
    }
  }
  for (const Shape& s : ushapes) {
    Pair x = C.make(s);
    CHECK("affine", S(s), 1e-6, cpu.affine(x.c, 1.7f, -0.3f), vk.affine(x.g, 1.7f, -0.3f));
  }

  // ---- binary -------------------------------------------------------------
  std::vector<std::pair<Shape, Shape>> bpairs = {
      {{}, {}}, {{3, 4}, {3, 4}}, {{}, {3, 4}}, {{3, 4}, {}}, {{3, 1, 5}, {4, 1}}, {{5}, {2, 3, 5}},
      {{1}, {1}}, {{4, 1}, {1, 4}}, {{2, 1, 3, 1, 5, 1}, {1, 4, 1, 6, 1, 7}}, {{2, 3, 4, 5}, {3, 1, 5}},
      {{1, 1, 7}, {6, 1}}, {{0, 3}, {1, 3}}};
  if (!quick) bpairs.push_back({{8662, 512}, {512}}), bpairs.push_back({{2, 61, 71, 512}, {2, 1, 1, 512}});
  const char* bnames[] = {"add", "sub", "mul", "div", "max", "min", "pow"};
  for (int op = 0; op <= static_cast<int>(BinaryOp::Pow); ++op) {
    BinaryOp b = static_cast<BinaryOp>(op);
    for (auto& [sa, sb] : bpairs) {
      Pair a = C.make(sa, b == BinaryOp::Pow ? 0.05f : -3.f, 3.f);
      Pair c = C.make(sb, b == BinaryOp::Pow ? -3.f : -3.f, 3.f);
      if (b == BinaryOp::Div) c = C.make(sb, 0.5f, 3.f);
      CHECK(std::string("b:") + bnames[op], S(sa) + "," + S(sb), b == BinaryOp::Pow ? 4e-6 : 1e-6,
            cpu.binary(b, a.c, c.c), vk.binary(b, a.g, c.g));
    }
  }
  {  // pow with negative bases and integer exponents
    std::vector<float> av = {-2.f, -1.5f, -3.f, 0.f, 0.f, 2.f, -0.5f, 4.f};
    std::vector<float> bv = {2.f, 3.f, -1.f, 2.f, 0.f, 0.5f, 4.f, -2.f};
    Pair a = C.from(av, {8}), c = C.from(bv, {8});
    CHECK("b:pow", "special values", 4e-6, cpu.binary(BinaryOp::Pow, a.c, c.c), vk.binary(BinaryOp::Pow, a.g, c.g));
  }

  // ---- matmul / linear ------------------------------------------------------
  std::vector<std::pair<Shape, Shape>> mm = {
      {{3, 4}, {4, 5}}, {{1, 1}, {1, 1}}, {{2, 3, 4}, {4, 5}}, {{4, 5}, {2, 5, 3}},
      {{2, 1, 3, 4}, {1, 5, 4, 6}}, {{2, 1, 3, 1, 4, 5}, {1, 2, 1, 3, 5, 2}}, {{65, 129}, {129, 67}},
      {{7, 0}, {0, 3}}, {{3, 2, 5}, {3, 5, 0}}};
  if (!quick) mm.push_back({{976, 71, 64}, {976, 64, 71}}), mm.push_back({{1000, 300}, {300, 513}});
  for (auto& [sa, sb] : mm) {
    Pair a = C.make(sa), b = C.make(sb);
    CHECK("matmul", S(sa) + "@" + S(sb), 0, cpu.matmul(a.c, b.c), vk.matmul(a.g, b.g));
  }
  struct LCase {
    Shape x;
    int64_t out;
    bool bias;
  };
  std::vector<LCase> lc = {{{5}, 7, true}, {{2, 3, 16}, 9, false}, {{1, 512}, 3072, true}, {{8, 33}, 17, true},
                           {{9, 33}, 17, true}, {{3, 4, 0}, 5, true}, {{130, 70}, 130, true}};
  if (!quick) lc.push_back({{8662, 512}, 2048, true}), lc.push_back({{72, 64, 768}, 768, false});
  for (auto& c : lc) {
    Pair x = C.make(c.x);
    Pair w = C.make({c.out, c.x.back()}, -0.1f, 0.1f);
    Pair b = C.make({c.out});
    CHECK("linear", S(c.x) + "x" + std::to_string(c.out) + (c.bias ? "+b" : ""), 0,
          cpu.linear(x.c, w.c, c.bias ? b.c : Tensor()), vk.linear(x.g, w.g, c.bias ? b.g : Tensor()));
  }

  // ---- softmax / norms --------------------------------------------------------
  std::vector<Shape> rshapes = {{1}, {7}, {3, 1}, {2, 5, 7}, {4, 40000}, {976, 71, 71}, {0, 4}};
  if (quick) rshapes = {{7}, {3, 1}, {2, 5, 7}};
  for (const Shape& s : rshapes) {
    Pair x = C.make(s, -8.f, 8.f);
    if (!s.empty() && s.back() > 4096) {
      // Long rows: the CPU reference sums 40k floats sequentially (error grows
      // ~len*eps); compare both backends against a float64 softmax instead.
      auto xv = x.c.to_vector(), rc = cpu.softmax(x.c).to_vector(), rg = vk.softmax(x.g).to_vector();
      const int64_t L = s.back();
      double ec = 0, eg = 0;
      for (int64_t r = 0; r < shape_numel(s) / L; ++r) {
        double mx = -1e300, sum = 0;
        for (int64_t j = 0; j < L; ++j) mx = std::max(mx, double(xv[r * L + j]));
        for (int64_t j = 0; j < L; ++j) sum += std::exp(double(xv[r * L + j]) - mx);
        for (int64_t j = 0; j < L; ++j) {
          double ref = std::exp(double(xv[r * L + j]) - mx) / sum;
          ec = std::max(ec, std::fabs(rc[r * L + j] - ref) / ref);
          eg = std::max(eg, std::fabs(rg[r * L + j] - ref) / ref);
        }
      }
      std::printf("  softmax %s vs float64: cpu max rel %.3g, vulkan max rel %.3g\n", S(s).c_str(), ec, eg);
      Stat& st = g_stats["softmax"];
      ++st.cases;
      if (eg > 2e-6) ++st.failures;
      continue;
    }
    CHECK("softmax", S(s), 2e-6, cpu.softmax(x.c), vk.softmax(x.g));
  }
  {  // T5-style -FLT_MAX additive mask, including fully-masked rows
    const int64_t B = 3, H = 2, T = 9;
    std::vector<float> m(B * T, 0.f);
    for (int64_t t = 5; t < T; ++t) m[0 * T + t] = -FLT_MAX;  // padding
    for (int64_t t = 0; t < T; ++t) m[1 * T + t] = -FLT_MAX;  // empty prompt: everything masked
    Pair mask = C.from(m, {B, 1, 1, T});
    Pair s = C.make({B, H, T, T}, -5.f, 5.f);
    CHECK("softmax", "scores + -FLT_MAX mask", 2e-6, cpu.softmax(cpu.binary(BinaryOp::Add, s.c, mask.c)),
          vk.softmax(vk.binary(BinaryOp::Add, s.g, mask.g)));
    std::vector<float> all(T, -FLT_MAX);
    Pair r = C.from(all, {1, T});
    CHECK("softmax", "row of -FLT_MAX", 1e-6, cpu.softmax(r.c), vk.softmax(r.g));
  }
  std::vector<Shape> nshapes = {{1}, {3, 768}, {2, 5, 7}, {8662, 512}, {5, 1}};
  if (quick) nshapes = {{3, 768}, {2, 5, 7}};
  for (const Shape& s : nshapes) {
    Pair x = C.make(s, -3.f, 5.f);
    Pair w = C.make({s.back()}), b = C.make({s.back()});
    CHECK("layer_norm", S(s) + " w,b", 2e-5, cpu.layer_norm(x.c, w.c, b.c, 1e-5f), vk.layer_norm(x.g, w.g, b.g, 1e-5f));
    CHECK("layer_norm", S(s) + " plain", 2e-5, cpu.layer_norm(x.c, Tensor(), Tensor(), 1e-6f),
          vk.layer_norm(x.g, Tensor(), Tensor(), 1e-6f));
    CHECK("rms_norm", S(s) + " w", 1e-5, cpu.rms_norm(x.c, w.c, 1e-6f), vk.rms_norm(x.g, w.g, 1e-6f));
    CHECK("rms_norm", S(s) + " plain", 1e-5, cpu.rms_norm(x.c, Tensor(), 1e-6f), vk.rms_norm(x.g, Tensor(), 1e-6f));
  }

  // ---- reduce ---------------------------------------------------------------------
  struct RCase {
    Shape s;
    int axis;
  };
  std::vector<RCase> rc = {{{}, 0}, {{}, -1}, {{7}, 0}, {{3, 4}, 0}, {{3, 4}, -1}, {{2, 3, 1, 4, 5, 2}, 2},
                           {{2, 3, 1, 4, 5, 2}, -3}, {{2, 3, 1, 4, 5, 2}, 0}, {{5, 0, 3}, 1}, {{1, 1}, 1}};
  if (!quick) rc.push_back({{4096, 1000}, 0}), rc.push_back({{1000, 4096}, 1}), rc.push_back({{3, 100000}, -1});
  const char* rnames[] = {"sum", "mean", "max", "min"};
  for (int op = 0; op < 4; ++op)
    for (auto& r : rc)
      for (bool keep : {false, true}) {
        if (op >= 2 && shape_numel(r.s) == 0) continue;
        Pair x = C.make(r.s, -1.f, 3.f);
        CHECK(std::string("r:") + rnames[op], S(r.s) + " ax" + std::to_string(r.axis) + (keep ? " keep" : ""),
              op < 2 ? 1e-5 : 0, cpu.reduce(static_cast<ReduceOp>(op), x.c, r.axis, keep),
              vk.reduce(static_cast<ReduceOp>(op), x.g, r.axis, keep));
      }

  // ---- data movement ---------------------------------------------------------------
  struct PCase {
    Shape s;
    std::vector<int> p;
  };
  std::vector<PCase> pc = {{{3, 4}, {1, 0}}, {{2, 3, 4}, {2, 0, 1}}, {{2, 3, 4}, {0, 1, 2}},
                           {{2, 1, 3, 1, 4, 5}, {5, 3, 1, 0, 4, 2}}, {{2, 3, 4}, {-1, 0, -2}},
                           {{2, 61, 71, 8, 64}, {0, 2, 3, 1, 4}}, {{1}, {0}}};
  for (auto& p : pc) {
    Pair x = C.make(p.s);
    CHECK("permute", S(p.s), 0, cpu.permute(x.c, p.p), vk.permute(x.g, p.p));
  }
  struct SCase {
    Shape s;
    int ax;
    int64_t a, b;
  };
  std::vector<SCase> sc = {{{10}, 0, 2, 7}, {{10}, 0, -3, -1}, {{4, 6, 5}, 1, 1, 4}, {{4, 6, 5}, -1, 0, 5},
                           {{4, 6, 5}, -1, 3, 2}, {{4, 6, 5}, 0, -100, 100}, {{2, 3, 1, 4, 5, 2}, 3, 1, 3},
                           {{2, 61, 71, 512}, -1, 256, 512}};
  for (auto& s : sc) {
    Pair x = C.make(s.s);
    CHECK("slice", S(s.s) + " ax" + std::to_string(s.ax) + " [" + std::to_string(s.a) + "," + std::to_string(s.b) + ")",
          0, cpu.slice(x.c, s.ax, s.a, s.b), vk.slice(x.g, s.ax, s.a, s.b));
  }
  {
    struct CCase {
      std::vector<Shape> parts;
      int ax;
    };
    std::vector<CCase> cc = {{{{2, 3}, {4, 3}}, 0}, {{{2, 3}, {2, 5}, {2, 0}, {2, 1}}, -1},
                             {{{2, 3, 4}, {2, 1, 4}, {2, 7, 4}}, 1}, {{{1}, {1}}, 0},
                             {{{2, 1, 3, 1, 4, 5}, {2, 1, 3, 2, 4, 5}}, 3}};
    for (auto& c : cc) {
      std::vector<Tensor> pcs, pgs;
      std::string d;
      for (auto& s : c.parts) {
        Pair p = C.make(s);
        pcs.push_back(p.c);
        pgs.push_back(p.g);
        d += S(s);
      }
      CHECK("concat", d + " ax" + std::to_string(c.ax), 0, cpu.concat(pcs, c.ax), vk.concat(pgs, c.ax));
    }
  }
  {
    struct GCase {
      Shape table;
      Shape idx;
    };
    std::vector<GCase> gc = {{{100, 3, 4}, {2, 5}}, {{10}, {7}}, {{10, 2}, {}}, {{5, 3}, {0}}};
    if (!quick) gc.push_back({{32128, 768}, {72, 64}});
    for (auto& g : gc) {
      Pair t = C.make(g.table);
      std::uniform_int_distribution<int> d(0, static_cast<int>(g.table[0]) - 1);
      std::vector<int32_t> idx(static_cast<size_t>(shape_numel(g.idx)));
      for (auto& i : idx) i = d(rng);
      CHECK("gather_rows", S(g.table) + "[" + S(g.idx) + "]", 0, cpu.gather_rows(t.c, idx, g.idx),
            vk.gather_rows(t.g, idx, g.idx));
    }
  }
  {
    std::vector<std::pair<Shape, Shape>> bc = {{{3, 1}, {2, 3, 4}}, {{}, {5}}, {{4}, {4}}, {{1, 1}, {3, 1, 2}},
                                               {{2, 1, 8, 71, 71}, {2, 61, 8, 71, 71}}};
    for (auto& [s, t] : bc) {
      Pair x = C.make(s);
      CHECK("broadcast_to", S(s) + "->" + S(t), 0, cpu.broadcast_to(x.c, t), vk.broadcast_to(x.g, t));
    }
  }

  // ---- fused composites (vs the CPU default implementation) -------------------------
  {
    std::vector<std::vector<Shape>> rp = {{{2, 8, 61, 64}, {61, 64}, {61, 64}},
                                          {{2, 61, 8, 71, 64}, {1, 1, 1, 71, 64}, {1, 1, 1, 71, 64}},
                                          {{4}, {4}, {4}}, {{3, 2}, {1}, {3, 1}}, {{2, 3, 6}, {2, 1, 6}, {6}}};
    for (auto& r : rp) {
      Pair x = C.make(r[0]), c = C.make(r[1]), s = C.make(r[2]);
      CHECK("rope", S(r[0]) + " cs" + S(r[1]), 1e-6, cpu.rope(x.c, c.c, s.c), vk.rope(x.g, c.g, s.g));
    }
    std::vector<std::vector<Shape>> mp = {{{2, 4331, 512}, {2, 1, 512}, {2, 1, 512}},
                                          {{3, 4}, {4}, {}}, {{}, {}, {}}, {{2, 3, 1, 4, 5, 2}, {3, 1, 1, 1, 2}, {1, 4, 5, 1}}};
    for (auto& m : mp) {
      Pair x = C.make(m[0]), sh = C.make(m[1]), sc2 = C.make(m[2]);
      CHECK("modulate", S(m[0]), 1e-6, cpu.modulate(x.c, sh.c, sc2.c), vk.modulate(x.g, sh.g, sc2.g));
    }
  }
  {
    struct ACase {
      Shape q, k, v, bias;
      float scale;
      std::string name;
    };
    std::vector<ACase> ac = {
        {{2, 3, 5, 16}, {2, 3, 7, 16}, {2, 3, 7, 16}, {}, 0.25f, "small no-bias"},
        {{1, 5}, {9, 5}, {9, 3}, {1, 9}, 0.7f, "2-d, bias row"},
        {{2, 4, 40, 32}, {2, 4, 100, 32}, {2, 4, 100, 48}, {4, 40, 1}, 0.3f, "Lk>tile, D!=DV"},
        {{2, 1, 33, 128}, {2, 1, 33, 128}, {2, 1, 33, 128}, {}, 0.088f, "D=128"},
        {{1, 2, 3, 160}, {1, 2, 4, 160}, {1, 2, 4, 160}, {}, 0.08f, "D=160 (fallback)"},
        {{3, 2, 1, 8}, {1, 2, 6, 8}, {3, 1, 6, 8}, {3, 2, 1, 6}, 1.f, "batch broadcast"},
    };
    if (!quick) {
      ac.push_back({{2, 61, 8, 71, 64}, {2, 61, 8, 71, 64}, {2, 61, 8, 71, 64}, {2, 1, 8, 71, 71}, 0.125f, "spatial"});
      ac.push_back({{2, 71, 8, 61, 64}, {2, 71, 8, 61, 64}, {2, 71, 8, 61, 64}, {}, 0.125f, "temporal"});
      ac.push_back({{72, 12, 64, 64}, {72, 12, 64, 64}, {72, 12, 64, 64}, {1, 12, 64, 64}, 1.f, "t5"});
    }
    for (auto& a : ac) {
      Pair q = C.make(a.q), k = C.make(a.k), v = C.make(a.v);
      Pair b = a.bias.empty() ? Pair{} : C.make(a.bias, -3.f, 3.f);
      CHECK("attention", a.name, 0, cpu.attention(q.c, k.c, v.c, a.scale, b.c),
            vk.attention(q.g, k.g, v.g, a.scale, b.g));
    }
    {  // T5 mask with -FLT_MAX: padded keys and a fully masked batch row, plus position bias
      const int64_t B = 4, H = 12, T = 40, d = 64;
      std::vector<float> m(B * T, 0.f);
      for (int64_t t = 7; t < T; ++t) m[0 * T + t] = -FLT_MAX;
      for (int64_t t = 0; t < T; ++t) m[2 * T + t] = -FLT_MAX;
      Pair mask = C.from(m, {B, 1, 1, T});
      Pair pos = C.make({1, H, T, T}, -4.f, 4.f);
      Pair q = C.make({B, H, T, d}), k = C.make({B, H, T, d}), v = C.make({B, H, T, d});
      Tensor bc = cpu.binary(BinaryOp::Add, pos.c, mask.c);
      Tensor bg = vk.binary(BinaryOp::Add, pos.g, mask.g);
      CHECK("attention", "t5 -FLT_MAX mask (fully masked rows)", 0, cpu.attention(q.c, k.c, v.c, 1.f, bc),
            vk.attention(q.g, k.g, v.g, 1.f, bg));
    }
  }

  // ---- exact usage patterns of the denoiser hot loop (src/model) ------------------
  {
    const int64_t B = 2, Fp = 62, J = 71, H = 8, hd = 64, D = 512;
    const float kMaskNeg = -1e30f;
    // (1) modulate: full-shape shift (residual), broadcast scale [B,1,1,D]
    Pair h = C.make({B, Fp, J, D}), x = C.make({B, Fp, J, D}), g = C.make({B, 1, 1, D});
    CHECK("modulate", "residual: shift full, scale [B,1,1,D]", 1e-6, cpu.modulate(x.c, h.c, g.c),
          vk.modulate(x.g, h.g, g.g));
    CHECK("modulate", "adaLN: shift/scale [B,1,1,D]", 1e-6, cpu.modulate(x.c, g.c, g.c), vk.modulate(x.g, g.g, g.g));
    // qkv -> permute -> slice(axis 0) views -> rms -> rope -> attention (spatial)
    Pair qkv = C.make({B, Fp, J, 3 * D});
    Pair qn = C.make({hd}), cs = C.make({B, 1, 1, J, hd}), sn = C.make({B, 1, 1, J, hd});
    std::vector<float> gb = rnd(B * H * J * J, -2.f, 2.f);
    for (int64_t b = 0; b < B; ++b)  // masked keys (padding joints) with -1e30
      for (int64_t hh = 0; hh < H; ++hh)
        for (int64_t i = 0; i < J; ++i)
          for (int64_t j = 60; j < J; ++j) gb[((b * H + hh) * J + i) * J + j] = kMaskNeg;
    Pair bias = C.from(gb, {B, 1, H, J, J});
    auto spatial = [&](Backend& be, const Tensor& t6, const Tensor& qnw, const Tensor& c, const Tensor& s,
                       const Tensor& bs) {
      Tensor t = be.permute(t6.reshape({B, Fp, J, 3, H, hd}), {3, 0, 1, 4, 2, 5});
      Tensor q = be.slice(t, 0, 0, 1).reshape({B, Fp, H, J, hd});
      Tensor k = be.slice(t, 0, 1, 2).reshape({B, Fp, H, J, hd});
      Tensor v = be.slice(t, 0, 2, 3).reshape({B, Fp, H, J, hd});
      q = be.rope(be.rms_norm(q, qnw, 1e-6f), c, s);
      k = be.rope(be.rms_norm(k, qnw, 1e-6f), c, s);
      return be.attention(q, k, v, 0.125f, bs);
    };
    CHECK("attention", "denoiser spatial (views, rope, -1e30 bias)", 0, spatial(cpu, qkv.c, qn.c, cs.c, sn.c, bias.c),
          spatial(vk, qkv.g, qn.g, cs.g, sn.g, bias.g));
    // temporal: cos/sin [Fp, hd], bias [B,1,1,1,Fp] with masked frames
    Pair tc = C.make({Fp, hd}), ts = C.make({Fp, hd});
    std::vector<float> tb(B * Fp, 0.f);
    for (int64_t f = 40; f < Fp; ++f) tb[1 * Fp + f] = kMaskNeg;
    Pair tbias = C.from(tb, {B, 1, 1, 1, Fp});
    auto temporal = [&](Backend& be, const Tensor& t6, const Tensor& qnw, const Tensor& c, const Tensor& s,
                        const Tensor& bs) {
      Tensor t = be.permute(t6.reshape({B, Fp, J, 3, H, hd}), {3, 0, 2, 4, 1, 5});
      Tensor q = be.slice(t, 0, 0, 1).reshape({B, J, H, Fp, hd});
      Tensor k = be.slice(t, 0, 1, 2).reshape({B, J, H, Fp, hd});
      Tensor v = be.slice(t, 0, 2, 3).reshape({B, J, H, Fp, hd});
      q = be.rope(be.rms_norm(q, qnw, 1e-6f), c, s);
      k = be.rope(be.rms_norm(k, qnw, 1e-6f), c, s);
      return be.attention(q, k, v, 0.125f, bs);
    };
    CHECK("attention", "denoiser temporal (rope [F,hd], frame mask)", 0,
          temporal(cpu, qkv.c, qn.c, tc.c, ts.c, tbias.c), temporal(vk, qkv.g, qn.g, tc.g, ts.g, tbias.g));
    // final root cross-attention: Lq = 1, permuted k/v, key bias
    const int64_t F = Fp - 1;
    Pair q1 = C.make({B, F, H, 1, hd}), kj = C.make({B, F, J - 1, H, hd}), vj = C.make({B, F, J - 1, H, hd});
    std::vector<float> kb(B * (J - 1), 0.f);
    for (int64_t j = 55; j < J - 1; ++j) kb[j] = kMaskNeg;
    Pair kvb = C.from(kb, {B, 1, 1, 1, J - 1});
    CHECK("attention", "denoiser root cross (Lq=1)", 0,
          cpu.attention(q1.c, cpu.permute(kj.c, {0, 1, 3, 2, 4}), cpu.permute(vj.c, {0, 1, 3, 2, 4}), 0.125f, kvb.c),
          vk.attention(q1.g, vk.permute(kj.g, {0, 1, 3, 2, 4}), vk.permute(vj.g, {0, 1, 3, 2, 4}), 0.125f, kvb.g));
    // (4) tiny gather, (5) huge adaLN linear on [B, 512]
    Pair tiny = C.make({6, 8});
    std::vector<int32_t> ti = {5, 0, 3, 3, 1, 2, 4};
    CHECK("gather_rows", "[6, 8][[7]]", 0, cpu.gather_rows(tiny.c, ti, {7}), vk.gather_rows(tiny.g, ti, {7}));
    Pair y = C.make({B, D}), wbig = C.make({47104, D}, -0.05f, 0.05f), bbig = C.make({47104});
    CHECK("linear", "adaLN [2,512]x[47104,512]^T+b", 0, cpu.linear(y.c, wbig.c, bbig.c), vk.linear(y.g, wbig.g, bbig.g));
    // (6) slice views: chunk(i) of the modulation tensor, view of a view, to_host of a view
    Pair mods = C.make({B, 92, D});
    CHECK("slice", "mods chunk [B,1,D]", 0, cpu.slice(mods.c, 1, 7, 8).reshape({B, 1, 1, D}),
          vk.slice(mods.g, 1, 7, 8).reshape({B, 1, 1, D}));
    Pair big = C.make({16, 4, 5});
    CHECK("slice", "view of view", 0, cpu.slice(cpu.slice(big.c, 0, 4, 12), 0, 2, 5),
          vk.slice(vk.slice(big.g, 0, 4, 12), 0, 2, 5));
    CHECK("slice", "misaligned (copy path)", 0, cpu.slice(big.c, 0, 1, 2), vk.slice(big.g, 0, 1, 2));
    {
      Tensor vv = vk.slice(big.g, 0, 8, 16);
      Tensor cc = cpu.slice(big.c, 0, 8, 16);
      Tensor sum_g = vk.binary(BinaryOp::Add, vv, vk.slice(big.g, 0, 0, 8));
      Tensor sum_c = cpu.binary(BinaryOp::Add, cc, cpu.slice(big.c, 0, 0, 8));
      big = Pair{};  // drop the parents: the views must keep the buffer alive
      CHECK("slice", "view outlives parent + ops on views", 0, sum_c, sum_g);
      CHECK("slice", "to_host of a view", 0, cc, vv);
    }
  }

  // ---- buffer recycling / hazards: long chain of dependent temporaries ------------
  {
    Pair x = C.make({4, 257});
    Tensor xc = x.c, xg = x.g;
    for (int i = 0; i < (quick ? 50 : 400); ++i) {
      Tensor tc = cpu.unary(UnaryOp::Sin, xc), tg = vk.unary(UnaryOp::Sin, xg);
      Tensor sc = cpu.softmax(tc), sg = vk.softmax(tg);
      xc = cpu.affine(cpu.binary(BinaryOp::Add, xc, cpu.affine(sc, 3.f, 0.f)), 0.9f, 0.01f);
      xg = vk.affine(vk.binary(BinaryOp::Add, xg, vk.affine(sg, 3.f, 0.f)), 0.9f, 0.01f);
      if (i % 37 == 0) {  // mixed sizes + concat/slice reuse
        Tensor cc = cpu.slice(cpu.concat({xc, xc}, 0), 0, 2, 6), cg = vk.slice(vk.concat({xg, xg}, 0), 0, 2, 6);
        xc = cpu.binary(BinaryOp::Mul, xc, cpu.affine(cc, 0.f, 1.f));
        xg = vk.binary(BinaryOp::Mul, xg, vk.affine(cg, 0.f, 1.f));
      }
    }
    CHECK("chain", "400 steps of recycled temporaries", 1e-4, xc, xg);
  }
}

int main(int argc, char** argv) {
  bool verbose = argc > 1 && std::string(argv[1]) == "-v";
  if (!vulkan_available()) {
    std::printf("No usable Vulkan device: skipping.\n");
    return 0;
  }
  auto cpu = make_cpu_backend();
  for (auto& d : vulkan_devices())
    std::printf("device %d: %s%s%s\n", d.index, d.name.c_str(), d.discrete ? " [discrete]" : "",
                d.usable ? "" : " [unusable]");

  auto t0 = std::chrono::steady_clock::now();
  {
    VulkanBackendOptions o;
    o.verbose = verbose;
    auto vk = make_vulkan_backend(o);
    std::printf("Vulkan backend on %s\n", vulkan_device_name(*vk).c_str());
    Ctx C{cpu.get(), vk.get()};
    suite(C, false);
    auto ms = vulkan_memory_stats(*vk);
    auto cn = vulkan_counters(*vk);
    std::printf("memory: device %.1f MB (peak %.1f), live %.1f MB, pooled %.1f MB; dispatches %llu, submits %llu, "
                "barriers %llu, pipelines %llu\n",
                ms.device_bytes / 1048576.0, ms.peak_device_bytes / 1048576.0, ms.live_bytes / 1048576.0,
                ms.pooled_free_bytes / 1048576.0, (unsigned long long)cn.dispatches,
                (unsigned long long)cn.submits, (unsigned long long)cn.barriers, (unsigned long long)cn.pipelines);
  }
  {
    // Stress configuration: descriptor-set fallback (no push descriptors),
    // a submit after every 3 dispatches, 1 MB staging (chunked uploads).
    VulkanBackendOptions o;
    o.disable_push_descriptors = true;
    o.dispatches_per_submit = 3;
    o.staging_bytes = 1 << 20;
    auto vk = make_vulkan_backend(o);
    Ctx C{cpu.get(), vk.get()};
    std::printf("stress configuration (descriptor sets, submit every 3, 1 MB staging)...\n");
    suite(C, true);
    Pair big = C.make({3, 700000});  // > staging: chunked upload
    CHECK("upload", "chunked 8.4 MB", 0, big.c, big.g);
  }
  {
    // Opt-in fp16 tensor-core GEMM: inputs rounded to fp16 (2^-11 relative),
    // fp32 accumulation -> expect ~1e-3 normwise; checked against 3e-3.
    VulkanBackendOptions o;
    o.fp16 = true;
    auto vk = make_vulkan_backend(o);
    Ctx C{cpu.get(), vk.get()};
    std::printf("fp16 tensor-core GEMM (opt-in) ...\n");
    std::vector<std::pair<Shape, Shape>> mm = {{{65, 129}, {129, 67}}, {{976, 71, 64}, {976, 64, 71}},
                                               {{2, 1, 33, 40}, {1, 3, 40, 17}}, {{300, 512}, {512, 100}}};
    for (auto& [sa, sb] : mm) {
      Pair a = C.make(sa), b = C.make(sb);
      CHECK("fp16 matmul", S(sa) + "@" + S(sb), 0, cpu->matmul(a.c, b.c), vk->matmul(a.g, b.g));
    }
    for (auto [M, K, N] : std::vector<std::array<int64_t, 3>>{{8662, 512, 1536}, {100, 70, 130}, {17, 33, 5}}) {
      Pair x = C.make({M, K}), w = C.make({N, K}, -0.1f, 0.1f), b = C.make({N});
      CHECK("fp16 linear", S({M, K}) + "x" + std::to_string(N), 0, cpu->linear(x.c, w.c, b.c),
            vk->linear(x.g, w.g, b.g));
    }
  }
  auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::printf("\n%-14s %5s %10s %10s %10s %7s %7s  %s\n", "op", "cases", "max abs", "max rel", "normwise", "rtol",
              "ntol", "worst (normwise) case");
  int fails = 0;
  for (auto& name : g_order) {
    Stat& s = g_stats[name];
    fails += s.failures;
    std::printf("%-14s %5d %10.3g %10.3g %10.3g %7.0g %7.0g  %s%s\n", name.c_str(), s.cases, s.max_abs, s.max_rel,
                s.max_norm, s.tol, s.ntol, s.worst.c_str(), s.failures ? "  <-- FAIL" : "");
  }
  std::printf("\n%d failures, %d errors (%.1f s)\n", fails, g_errors, secs);
  return (fails || g_errors) ? 1 : 0;
}

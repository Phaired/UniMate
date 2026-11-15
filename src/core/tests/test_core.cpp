#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <random>

#include "check.h"
#include "core/tensor.h"
#include "core/umtf.h"

using namespace um;
using V = std::vector<float>;

static std::unique_ptr<Backend> g_be;
static Backend& be() { return *g_be; }

static V rnd(size_t n, unsigned seed = 1, float lo = -1.f, float hi = 1.f) {
  std::mt19937 g(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  V v(n);
  for (auto& x : v) x = d(g);
  return v;
}
static Tensor T(const V& v, Shape s) { return be().from_vector(v, s); }

static void test_shapes() {
  CHECK(shape_numel({}) == 1);
  CHECK(shape_numel({2, 3, 4}) == 24);
  CHECK(shape_str({2, 3}) == "[2, 3]");
  CHECK(broadcast_shapes({2, 1, 4}, {3, 1}) == (Shape{2, 3, 4}));
  CHECK(broadcast_shapes({}, {5}) == (Shape{5}));
  CHECK_THROWS(broadcast_shapes({2, 3}, {4}));
}

static void test_views() {
  Tensor t = T(rnd(24), {2, 3, 4});
  CHECK(t.reshape({6, -1}).shape() == (Shape{6, 4}));
  CHECK(t.reshape({-1}).shape() == (Shape{24}));
  CHECK(t.reshape({4, 6}).storage() == t.storage());  // zero-copy
  CHECK_THROWS(t.reshape({5, -1}));
  CHECK_THROWS(t.reshape({-1, -1}));
  CHECK_THROWS(t.reshape({5, 5}));
  CHECK(t.unsqueeze(0).shape() == (Shape{1, 2, 3, 4}));
  CHECK(t.unsqueeze(-1).shape() == (Shape{2, 3, 4, 1}));
  CHECK(t.unsqueeze(3).shape() == (Shape{2, 3, 4, 1}));
  CHECK(t.unsqueeze(-1).squeeze(-1).shape() == (Shape{2, 3, 4}));
  CHECK(t.squeeze(0).shape() == (Shape{2, 3, 4}));  // non-1 dim: no-op
  CHECK(t.flatten(0, 1).shape() == (Shape{6, 4}));
  CHECK(t.flatten(1, -1).shape() == (Shape{2, 12}));
  CHECK(t.dim(-1) == 4);
  CHECK_THROWS(t.dim(3));
  CHECK_NEAR(t.reshape({4, 6}).to_vector(), t.to_vector());
  Tensor s = be().scalar(2.5f);
  CHECK(s.shape().empty() && s.numel() == 1 && s.to_vector()[0] == 2.5f);
}

static void test_unary() {
  V x = {-2.f, -0.5f, 0.f, 0.5f, 3.f};
  Tensor t = T(x, {5});
  auto ap = [&](UnaryOp op, float (*f)(float)) {
    V want;
    for (float v : x) want.push_back(f(v));
    return near_vec(be().unary(op, t).to_vector(), want, 1e-6f);
  };
  CHECK(ap(UnaryOp::Neg, [](float v) { return -v; }));
  CHECK(ap(UnaryOp::Abs, [](float v) { return std::fabs(v); }));
  CHECK(ap(UnaryOp::Square, [](float v) { return v * v; }));
  CHECK(ap(UnaryOp::Exp, [](float v) { return std::exp(v); }));
  CHECK(ap(UnaryOp::Sin, [](float v) { return std::sin(v); }));
  CHECK(ap(UnaryOp::Cos, [](float v) { return std::cos(v); }));
  CHECK(ap(UnaryOp::Tanh, [](float v) { return std::tanh(v); }));
  CHECK(ap(UnaryOp::Sigmoid, [](float v) { return 1.f / (1.f + std::exp(-v)); }));
  CHECK(ap(UnaryOp::Relu, [](float v) { return v > 0 ? v : 0.f; }));
  CHECK(ap(UnaryOp::Silu, [](float v) { return v / (1.f + std::exp(-v)); }));
  V p = {0.25f, 1.f, 4.f};
  Tensor tp = T(p, {3});
  CHECK_NEAR(be().unary(UnaryOp::Sqrt, tp).to_vector(), (V{0.5f, 1.f, 2.f}));
  CHECK_NEAR(be().unary(UnaryOp::Rsqrt, tp).to_vector(), (V{2.f, 1.f, 0.5f}));
  CHECK_NEAR(be().unary(UnaryOp::Reciprocal, tp).to_vector(), (V{4.f, 1.f, 0.25f}));
  CHECK_NEAR(be().unary(UnaryOp::Log, tp).to_vector(), (V{std::log(0.25f), 0.f, std::log(4.f)}));
  // exact GELU reference values (PyTorch): gelu(1)=0.8413447, gelu(-1)=-0.1586553, gelu(0.5)=0.3457312
  CHECK_NEAR(be().unary(UnaryOp::Gelu, T({1.f, -1.f, 0.5f, 0.f}, {4})).to_vector(),
             (V{0.8413447f, -0.1586553f, 0.3457312f, 0.f}), 1e-6f);
  // tanh gelu: gelu_tanh(1)=0.8411920
  CHECK_NEAR(be().unary(UnaryOp::GeluTanh, T({1.f}, {1})).to_vector(), (V{0.8411920f}), 1e-6f);
  CHECK_NEAR(be().unary(UnaryOp::GeluNew, T({1.f}, {1})).to_vector(), (V{0.8411920f}), 1e-6f);
  CHECK_NEAR(be().affine(t, 2.f, 1.f).to_vector(), (V{-3.f, 0.f, 1.f, 2.f, 7.f}));
  CHECK_NEAR((t * 2.f + 1.f).to_vector(), (V{-3.f, 0.f, 1.f, 2.f, 7.f}));
  CHECK_NEAR((-t).to_vector(), (V{2.f, 0.5f, 0.f, -0.5f, -3.f}));
}

static void test_binary() {
  Tensor a = T({1, 2, 3, 4, 5, 6}, {2, 3});
  Tensor row = T({10, 20, 30}, {3});
  Tensor col = T({100, 200}, {2, 1});
  Tensor sc = be().scalar(2.f);
  CHECK_NEAR((a + row).to_vector(), (V{11, 22, 33, 14, 25, 36}));
  CHECK_NEAR((a + col).to_vector(), (V{101, 102, 103, 204, 205, 206}));
  CHECK_NEAR((row + col).to_vector(), (V{110, 120, 130, 210, 220, 230}));
  CHECK((row + col).shape() == (Shape{2, 3}));
  CHECK_NEAR((a * sc).to_vector(), (V{2, 4, 6, 8, 10, 12}));
  CHECK_NEAR((sc * a).to_vector(), (V{2, 4, 6, 8, 10, 12}));
  CHECK_NEAR((a - sc).to_vector(), (V{-1, 0, 1, 2, 3, 4}));
  CHECK_NEAR((sc - a).to_vector(), (V{1, 0, -1, -2, -3, -4}));
  CHECK_NEAR((a / sc).to_vector(), (V{.5f, 1, 1.5f, 2, 2.5f, 3}));
  CHECK_NEAR(be().binary(BinaryOp::Max, a, T({3.f}, {1})).to_vector(), (V{3, 3, 3, 4, 5, 6}));
  CHECK_NEAR(be().binary(BinaryOp::Min, a, T({3.f}, {1})).to_vector(), (V{1, 2, 3, 3, 3, 3}));
  CHECK_NEAR(be().binary(BinaryOp::Pow, a, sc).to_vector(), (V{1, 4, 9, 16, 25, 36}));
  // 3-way broadcast [2,1,4] + [3,1]
  Tensor x = T({1, 2, 3, 4, 5, 6, 7, 8}, {2, 1, 4}), y = T({10, 20, 30}, {3, 1});
  Tensor z = x + y;
  CHECK(z.shape() == (Shape{2, 3, 4}));
  V want;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 4; ++k) want.push_back((i * 4 + k + 1) + (j + 1) * 10.f);
  CHECK_NEAR(z.to_vector(), want);
  // 0-d with 0-d, and size-0 tensors
  CHECK((be().scalar(1.f) + be().scalar(2.f)).to_vector()[0] == 3.f);
  CHECK((T({}, {0, 3}) + row).shape() == (Shape{0, 3}));
  CHECK_THROWS(a + T({1, 2}, {2}));
  // large, parallel, broadcast-with-chunking
  V big = rnd(1 << 20, 3);
  Tensor tb = T(big, {1024, 1024}), tr = T(rnd(1024, 4), {1024});
  V got = (tb + tr).to_vector(), r = tr.to_vector();
  bool ok = true;
  for (size_t i = 0; i < big.size(); ++i) ok &= got[i] == big[i] + r[i % 1024];
  CHECK(ok);
  // column broadcast on a big tensor [1024,1024] * [1024,1]
  Tensor tc = T(rnd(1024, 5), {1024, 1});
  V gotc = (tb * tc).to_vector(), c = tc.to_vector();
  ok = true;
  for (size_t i = 0; i < big.size(); ++i) ok &= gotc[i] == big[i] * c[i / 1024];
  CHECK(ok);
}

static V naive_matmul(const V& a, const V& b, int64_t n, int64_t k, int64_t m) {
  V c(n * m);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < m; ++j) {
      double s = 0;
      for (int64_t l = 0; l < k; ++l) s += (double)a[i * k + l] * b[l * m + j];
      c[i * m + j] = (float)s;
    }
  return c;
}

static void test_matmul_linear() {
  Tensor a = T({1, 2, 3, 4, 5, 6}, {2, 3}), b = T({1, 0, 0, 1, 1, 1}, {3, 2});
  CHECK_NEAR(matmul(a, b).to_vector(), (V{4, 5, 10, 11}));
  for (auto [n, k, m] : std::vector<std::array<int64_t, 3>>{{1, 1, 1}, {5, 7, 3}, {17, 33, 19}, {64, 64, 64}, {3, 130, 9}}) {
    V av = rnd(n * k, 5), bv = rnd(k * m, 6);
    CHECK_NEAR(matmul(T(av, {n, k}), T(bv, {k, m})).to_vector(), naive_matmul(av, bv, n, k, m), 1e-4f);
  }
  // batch broadcast: a[2,1,3,4] @ b[5,4,2] -> [2,5,3,2]
  V av = rnd(2 * 3 * 4, 7), bv = rnd(5 * 4 * 2, 8);
  Tensor r = matmul(T(av, {2, 1, 3, 4}), T(bv, {5, 4, 2}));
  CHECK(r.shape() == (Shape{2, 5, 3, 2}));
  V got = r.to_vector(), want;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 5; ++j) {
      V ai(av.begin() + i * 12, av.begin() + (i + 1) * 12), bj(bv.begin() + j * 8, bv.begin() + (j + 1) * 8);
      V c = naive_matmul(ai, bj, 3, 4, 2);
      want.insert(want.end(), c.begin(), c.end());
    }
  CHECK_NEAR(got, want, 1e-5f);
  CHECK_THROWS(matmul(T(rnd(6), {2, 3}), T(rnd(8), {4, 2})));

  // linear: x[M,K] w[N,K] b[N]; odd sizes to hit tile edges.
  for (auto [M, K, N] : std::vector<std::array<int64_t, 3>>{{1, 1, 1}, {3, 5, 7}, {6, 17, 5}, {33, 70, 65}, {1, 300, 130}}) {
    V xv = rnd(M * K, 9), wv = rnd(N * K, 10), bi = rnd(N, 11);
    V wt(K * N);
    for (int64_t n = 0; n < N; ++n)
      for (int64_t k = 0; k < K; ++k) wt[k * N + n] = wv[n * K + k];
    V want2 = naive_matmul(xv, wt, M, K, N);
    V nb = want2;
    for (int64_t i = 0; i < M; ++i)
      for (int64_t j = 0; j < N; ++j) want2[i * N + j] += bi[j];
    Tensor y = linear(T(xv, {M, K}), T(wv, {N, K}), T(bi, {N}));
    CHECK(y.shape() == (Shape{M, N}));
    CHECK_NEAR(y.to_vector(), want2, 1e-4f);
    Tensor y2 = linear(T(xv, {1, M, K}), T(wv, {N, K}), Tensor());
    CHECK(y2.shape() == (Shape{1, M, N}));
    CHECK_NEAR(y2.to_vector(), nb, 1e-4f);
  }
  CHECK_THROWS(linear(T(rnd(6), {2, 3}), T(rnd(8), {2, 4}), Tensor()));
}

static void test_softmax_norms() {
  Tensor x = T({1, 2, 3, 1000, 1000, 1000}, {2, 3});
  V s = softmax(x).to_vector();
  float e1 = std::exp(-2.f), e2 = std::exp(-1.f), z = 1 + e1 + e2;
  CHECK_NEAR(s, (V{e1 / z, e2 / z, 1 / z, 1.f / 3, 1.f / 3, 1.f / 3}), 1e-6f);
  CHECK(std::isfinite(s[3]));
  // layer norm vs definition
  V xv = rnd(4 * 8, 12), w = rnd(8, 13), b = rnd(8, 14);
  V want, want_na;
  for (int r = 0; r < 4; ++r) {
    double m = 0, v = 0;
    for (int j = 0; j < 8; ++j) m += xv[r * 8 + j];
    m /= 8;
    for (int j = 0; j < 8; ++j) v += (xv[r * 8 + j] - m) * (xv[r * 8 + j] - m);
    v /= 8;
    for (int j = 0; j < 8; ++j) {
      want.push_back((float)((xv[r * 8 + j] - m) / std::sqrt(v + 1e-5) * w[j] + b[j]));
      want_na.push_back((float)((xv[r * 8 + j] - m) / std::sqrt(v + 1e-6)));
    }
  }
  CHECK_NEAR(be().layer_norm(T(xv, {4, 8}), T(w, {8}), T(b, {8}), 1e-5f).to_vector(), want, 1e-5f);
  CHECK_NEAR(be().layer_norm(T(xv, {4, 8}), Tensor(), Tensor(), 1e-6f).to_vector(), want_na, 1e-5f);
  // rms norm
  V wr;
  for (int r = 0; r < 4; ++r) {
    double ms = 0;
    for (int j = 0; j < 8; ++j) ms += (double)xv[r * 8 + j] * xv[r * 8 + j];
    ms /= 8;
    for (int j = 0; j < 8; ++j) wr.push_back((float)(xv[r * 8 + j] / std::sqrt(ms + 1e-6) * w[j]));
  }
  CHECK_NEAR(be().rms_norm(T(xv, {4, 8}), T(w, {8}), 1e-6f).to_vector(), wr, 1e-5f);
}

static void test_reduce() {
  Tensor x = T({1, 2, 3, 4, 5, 6}, {2, 3});
  CHECK_NEAR(sum(x, 1).to_vector(), (V{6, 15}));
  CHECK_NEAR(sum(x, 0).to_vector(), (V{5, 7, 9}));
  CHECK_NEAR(mean(x, -1, true).to_vector(), (V{2, 5}));
  CHECK(mean(x, -1, true).shape() == (Shape{2, 1}));
  CHECK(sum(x, 0, false).shape() == (Shape{3}));
  CHECK_NEAR(be().reduce(ReduceOp::Max, x, 1, false).to_vector(), (V{3, 6}));
  CHECK_NEAR(be().reduce(ReduceOp::Min, x, 0, true).to_vector(), (V{1, 2, 3}));
  CHECK(be().reduce(ReduceOp::Min, x, 0, true).shape() == (Shape{1, 3}));
  Tensor y = T(rnd(3 * 700 * 5, 15), {3, 700, 5});
  V yv = y.to_vector(), want(3 * 5);
  for (int o = 0; o < 3; ++o)
    for (int i = 0; i < 5; ++i) {
      double s = 0;
      for (int r = 0; r < 700; ++r) s += yv[(o * 700 + r) * 5 + i];
      want[o * 5 + i] = (float)s;
    }
  CHECK_NEAR(sum(y, 1).to_vector(), want, 1e-5f);
  Tensor w = T(rnd(2 * 3 * 1500, 16), {2, 3, 1500});  // inner > chunk
  V wv = w.to_vector(), wm(2 * 1500);
  for (int o = 0; o < 2; ++o)
    for (int i = 0; i < 1500; ++i) {
      float m = -1e30f;
      for (int r = 0; r < 3; ++r) m = std::max(m, wv[(o * 3 + r) * 1500 + i]);
      wm[o * 1500 + i] = m;
    }
  CHECK_NEAR(be().reduce(ReduceOp::Max, w, 1, false).to_vector(), wm);
}

static void test_layout_ops() {
  V xv(24);
  for (int i = 0; i < 24; ++i) xv[i] = (float)i;
  Tensor x = T(xv, {2, 3, 4});
  Tensor p = x.permute({2, 0, 1});
  CHECK(p.shape() == (Shape{4, 2, 3}));
  V pv = p.to_vector();
  bool ok = true;
  for (int a = 0; a < 4; ++a)
    for (int b = 0; b < 2; ++b)
      for (int c = 0; c < 3; ++c) ok &= pv[(a * 2 + b) * 3 + c] == xv[(b * 3 + c) * 4 + a];
  CHECK(ok);
  CHECK(x.transpose(-1, -2).shape() == (Shape{2, 4, 3}));
  CHECK_NEAR(x.transpose(0, 2).transpose(0, 2).to_vector(), xv);
  CHECK_THROWS(x.permute({0, 0, 1}));
  // slice with negative indices
  CHECK_NEAR(x.slice(2, 1, 3).slice(0, 1, 2).to_vector(), (V{13, 14, 17, 18, 21, 22}));
  CHECK(x.slice(2, -2, 4).shape() == (Shape{2, 3, 2}));
  CHECK_NEAR(x.slice(2, -2, 4).to_vector(), (V{2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23}));
  CHECK_NEAR(x.slice(-1, 0, -3).to_vector(), (V{0, 4, 8, 12, 16, 20}));
  CHECK_NEAR(x.slice(1, 1, 2).to_vector(), (V{4, 5, 6, 7, 16, 17, 18, 19}));
  CHECK(x.slice(1, 2, 2).numel() == 0);
  // concat
  Tensor a = T({1, 2, 3, 4}, {2, 2}), b = T({5, 6}, {2, 1});
  CHECK_NEAR(concat({a, b}, 1).to_vector(), (V{1, 2, 5, 3, 4, 6}));
  CHECK_NEAR(concat({a, a}, 0).to_vector(), (V{1, 2, 3, 4, 1, 2, 3, 4}));
  CHECK_NEAR(concat({b, a, b}, -1).to_vector(), (V{5, 1, 2, 5, 6, 3, 4, 6}));
  CHECK_THROWS(concat({a, b}, 0));
  // gather_rows
  Tensor tab = T({0, 1, 10, 11, 20, 21}, {3, 2});
  Tensor g = be().gather_rows(tab, {2, 0, 2, 1}, {2, 2});
  CHECK(g.shape() == (Shape{2, 2, 2}));
  CHECK_NEAR(g.to_vector(), (V{20, 21, 0, 1, 20, 21, 10, 11}));
  CHECK_THROWS(be().gather_rows(tab, {3}, {1}));
  // broadcast_to
  Tensor r = T({1, 2, 3}, {3});
  CHECK_NEAR(be().broadcast_to(r, {2, 3}).to_vector(), (V{1, 2, 3, 1, 2, 3}));
  CHECK_NEAR(be().broadcast_to(T({1, 2}, {2, 1}), {2, 3}).to_vector(), (V{1, 1, 1, 2, 2, 2}));
  CHECK_NEAR(be().broadcast_to(be().scalar(7.f), {2, 2}).to_vector(), (V{7, 7, 7, 7}));
  CHECK_THROWS(be().broadcast_to(r, {2, 4}));
  // map_host, full
  Tensor m = be().map_host(r, {3}, [](const float* in, float* out) { for (int i = 0; i < 3; ++i) out[i] = in[i] * in[i]; });
  CHECK_NEAR(m.to_vector(), (V{1, 4, 9}));
  CHECK_NEAR(be().full({2, 2}, 3.f).to_vector(), (V{3, 3, 3, 3}));
  CHECK_NEAR(be().zeros({3}).to_vector(), (V{0, 0, 0}));
  CHECK_THROWS(be().from_vector({1, 2, 3}, {2, 2}));
}

// ---- composites vs their definition (computed on host) -------------------------
static void test_composites() {
  const int B = 2, L = 5, D = 8;
  V xv = rnd(B * L * D, 20), cv = rnd(L * D, 21), sv = rnd(L * D, 22);
  Tensor x = T(xv, {B, L, D}), cs = T(cv, {L, D}), sn = T(sv, {L, D});
  V want(xv.size());
  for (int b = 0; b < B; ++b)
    for (int l = 0; l < L; ++l)
      for (int d = 0; d < D; ++d) {
        float xx = xv[(b * L + l) * D + d];
        float rot = d < D / 2 ? -xv[(b * L + l) * D + d + D / 2] : xv[(b * L + l) * D + d - D / 2];
        want[(b * L + l) * D + d] = xx * cv[l * D + d] + rot * sv[l * D + d];
      }
  CHECK_NEAR(be().rope(x, cs, sn).to_vector(), want, 1e-6f);

  // attention: q[B,H,Lq,d] k[B,H,Lk,d] v[B,H,Lk,dv], bias [B,1,1,Lk]
  const int H = 3, Lq = 4, Lk = 6, d = 5, dv = 7;
  V qv = rnd(B * H * Lq * d, 30), kv = rnd(B * H * Lk * d, 31), vv = rnd(B * H * Lk * dv, 32);
  V bv(B * Lk, 0.f);
  bv[Lk - 1] = -1e9f;
  bv[Lk + 2] = -1e9f;
  float scale = 0.37f;
  V wa(B * H * Lq * dv);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int i = 0; i < Lq; ++i) {
        std::vector<double> sc(Lk);
        double mx = -1e300;
        for (int j = 0; j < Lk; ++j) {
          double s = 0;
          for (int t = 0; t < d; ++t)
            s += (double)qv[((b * H + h) * Lq + i) * d + t] * kv[((b * H + h) * Lk + j) * d + t];
          sc[j] = s * scale + bv[b * Lk + j];
          mx = std::max(mx, sc[j]);
        }
        double z = 0;
        for (auto& s : sc) { s = std::exp(s - mx); z += s; }
        for (int t = 0; t < dv; ++t) {
          double o = 0;
          for (int j = 0; j < Lk; ++j) o += sc[j] / z * vv[((b * H + h) * Lk + j) * dv + t];
          wa[((b * H + h) * Lq + i) * dv + t] = (float)o;
        }
      }
  Tensor q = T(qv, {B, H, Lq, d}), k = T(kv, {B, H, Lk, d}), v = T(vv, {B, H, Lk, dv});
  Tensor o = be().attention(q, k, v, scale, T(bv, {B, 1, 1, Lk}));
  CHECK(o.shape() == (Shape{B, H, Lq, dv}));
  CHECK_NEAR(o.to_vector(), wa, 1e-5f);
  CHECK_NEAR(be().attention(q, k, v, scale, Tensor()).to_vector(),
             be().attention(q, k, v, scale, be().zeros({1})).to_vector(), 1e-6f);

  // modulate: x[B,L,D] shift/scale [B,1,D]
  V shv = rnd(B * D, 40), scv = rnd(B * D, 41), wm(xv.size());
  for (int b = 0; b < B; ++b)
    for (int l = 0; l < L; ++l)
      for (int dd = 0; dd < D; ++dd)
        wm[(b * L + l) * D + dd] = xv[(b * L + l) * D + dd] * (1 + scv[b * D + dd]) + shv[b * D + dd];
  CHECK_NEAR(be().modulate(x, T(shv, {B, 1, D}), T(scv, {B, 1, D})).to_vector(), wm, 1e-6f);
}

static void test_weightstore_and_umtf() {
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "um_core_test";
  fs::create_directories(dir);
  std::string path = (dir / "rt.umtf").string();
  ArrayDict arrs;
  arrs["a.b.w"] = HostArray::from_f32({1, 2, 3, 4, 5, 6}, {2, 3});
  arrs["a.b.bias"] = HostArray::from_f32({0.5f, -0.5f}, {2});
  arrs["a.ids"] = HostArray::from_i32({7, -8, 9}, {3});
  arrs["z.big"] = HostArray::from_i64({1LL << 40, -5}, {2});
  HostArray u8(DType::U8, {5});
  u8.data<uint8_t>()[2] = 255;
  arrs["z.mask"] = u8;
  arrs["scalar"] = HostArray::from_f32({3.25f}, {});
  arrs["empty"] = HostArray(DType::F32, {0, 4});
  nlohmann::json meta = {{"k", "v"}, {"n", 12}, {"text", "h\xC3\xA9llo"}};
  UmtfFile::save(path, meta, arrs);
  UmtfFile f = UmtfFile::load(path);
  CHECK(f.meta() == meta);
  CHECK(f.names().size() == arrs.size());
  CHECK(f.has("a.b.w") && !f.has("nope"));
  CHECK(f.get("a.b.w").shape == (Shape{2, 3}));
  CHECK(f.get("a.b.w").to_f32() == (V{1, 2, 3, 4, 5, 6}));
  CHECK(f.get("a.ids").dtype == DType::I32 && f.get("a.ids").to_i32() == (std::vector<int32_t>{7, -8, 9}));
  CHECK(f.get("z.big").to_i64()[0] == (1LL << 40));
  CHECK(f.get("z.mask").data<uint8_t>()[2] == 255 && f.get("z.mask").data<uint8_t>()[0] == 0);
  CHECK(f.get("scalar").shape.empty() && f.get("scalar").to_f32()[0] == 3.25f);
  CHECK(f.get("empty").numel() == 0);
  CHECK(f.get("a.ids").to_f32() == (V{7, -8, 9}));
  CHECK_THROWS(f.get("missing"));
  CHECK(fs::file_size(path) % 64 == 0);
  {
    std::string bad = (dir / "bad.umtf").string();
    FILE* fp = std::fopen(bad.c_str(), "wb");
    std::fputs("NOPE", fp);
    std::fclose(fp);
    CHECK_THROWS(UmtfFile::load(bad));
    CHECK_THROWS(UmtfFile::load((dir / "does_not_exist.umtf").string()));
  }

  // WeightStore
  auto file = std::make_shared<const UmtfFile>(std::move(f));
  WeightStore ws(be(), file);
  CHECK(ws.has("a.b.w"));
  Tensor w = ws.get("a.b.w");
  CHECK(w.shape() == (Shape{2, 3}));
  CHECK(ws.get("a.b.w").storage() == w.storage());  // cached
  WeightStore sa = ws.sub("a");
  WeightStore sab = sa.sub("b");
  CHECK(sab.has("w") && sab.has("bias") && !sab.has("ids"));
  CHECK(sab.get("w").storage() == w.storage());  // shared cache
  CHECK(sa.get("ids").to_vector() == (V{7, -8, 9}));
  CHECK(!ws.sub("a.b").get_or_empty("nope").valid());
  CHECK(ws.sub("a.b").get_or_empty("bias").valid());
  CHECK(ws.sub("").has("scalar"));
  CHECK(sab.host("w").shape == (Shape{2, 3}));
  CHECK(ws.meta()["n"] == 12);
  CHECK_THROWS(sab.get("ids"));
  Tensor t = HostArray::from_tensor(w).to_tensor(be());
  CHECK_NEAR(t.to_vector(), (V{1, 2, 3, 4, 5, 6}));
}

static void bench() {
  const int64_t M = 61 * 71, K = 512, N = 2048;
  Tensor x = T(rnd(M * K, 50), {M, K}), w = T(rnd(N * K, 51), {N, K}), b = T(rnd(N, 52), {N});
  linear(x, w, b).to_vector();  // warm up
  auto t0 = std::chrono::steady_clock::now();
  const int iters = 5;
  for (int i = 0; i < iters; ++i) linear(x, w, b).to_vector();
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
  std::printf("BENCH linear [%lld,%lld]x[%lld,%lld]: %.1f ms  (%.1f GFLOP/s)\n", (long long)M, (long long)K,
              (long long)N, (long long)K, ms, 2.0 * M * K * N / (ms * 1e6));
  Tensor a = T(rnd(12 * 77 * 64, 53), {12, 77, 64}), bb = T(rnd(12 * 64 * 77, 54), {12, 64, 77});
  matmul(a, bb).to_vector();
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < 20; ++i) matmul(a, bb).to_vector();
  ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20;
  std::printf("BENCH batched matmul [12,77,64]x[12,64,77]: %.2f ms\n", ms);
}

int main() {
  g_be = make_cpu_backend();
  std::printf("backend: %s\n", be().name().c_str());
  RUN(test_shapes);
  RUN(test_views);
  RUN(test_unary);
  RUN(test_binary);
  RUN(test_matmul_linear);
  RUN(test_softmax_norms);
  RUN(test_reduce);
  RUN(test_layout_ops);
  RUN(test_composites);
  RUN(test_weightstore_and_umtf);
  bench();
  return finish();
}

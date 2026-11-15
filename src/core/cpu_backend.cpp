// CPU reference backend. PyTorch semantics, IEEE float (no fast-math), AVX2/FMA
// register-blocked GEMM for linear(), persistent thread pool for parallelism.
#include <atomic>
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

#if defined(__AVX2__) && defined(__FMA__) || (defined(_MSC_VER) && defined(__AVX2__))
#include <immintrin.h>
#define UM_AVX2 1
#endif

#include "tensor.h"

namespace um {
namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("um/cpu: " + msg); }

// ---- thread pool -------------------------------------------------------------

class Pool {
 public:
  Pool() {
    int n = static_cast<int>(std::thread::hardware_concurrency());
    if (const char* e = std::getenv("UM_THREADS")) n = std::max(1, std::atoi(e));
    n = std::max(1, std::min(n, 64));
    nthreads_ = n;
    for (int i = 1; i < n; ++i) workers_.emplace_back([this] { worker(); });
  }
  ~Pool() {
    {
      std::lock_guard<std::mutex> l(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }
  int threads() const { return nthreads_; }

  // f(begin, end) over [0, total) in chunks of at least `grain` items.
  void run(int64_t total, int64_t grain, const std::function<void(int64_t, int64_t)>& f) {
    if (total <= 0) return;
    grain = std::max<int64_t>(1, grain);
    if (total <= grain || nthreads_ == 1 || in_parallel_) {
      f(0, total);
      return;
    }
    std::lock_guard<std::mutex> submit(submit_m_);
    int64_t chunk = std::max<int64_t>(grain, (total + nthreads_ * 4 - 1) / (nthreads_ * 4));
    {
      std::lock_guard<std::mutex> l(m_);
      job_ = &f;
      next_.store(0);
      total_ = total;
      chunk_ = chunk;
      active_ = static_cast<int>(workers_.size());
      ++gen_;
      in_parallel_ = true;
    }
    cv_.notify_all();
    work();
    std::unique_lock<std::mutex> l(m_);
    done_cv_.wait(l, [this] { return active_ == 0; });
    in_parallel_ = false;
  }

 private:
  void work() {
    for (;;) {
      int64_t b = next_.fetch_add(chunk_);
      if (b >= total_) break;
      (*job_)(b, std::min(b + chunk_, total_));
    }
  }
  void worker() {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return stop_ || gen_ != seen; });
        if (stop_) return;
        seen = gen_;
      }
      work();
      {
        std::lock_guard<std::mutex> l(m_);
        if (--active_ == 0) done_cv_.notify_one();
      }
    }
  }

  int nthreads_ = 1;
  std::vector<std::thread> workers_;
  std::mutex m_, submit_m_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int64_t, int64_t)>* job_ = nullptr;
  std::atomic<int64_t> next_{0};
  int64_t total_ = 0, chunk_ = 1;
  int active_ = 0;
  uint64_t gen_ = 0;
  bool stop_ = false;
  bool in_parallel_ = false;
};

Pool& pool() {
  static Pool p;
  return p;
}

template <class F>
void parallel_for(int64_t total, int64_t grain, F&& f) {
  std::function<void(int64_t, int64_t)> fn = std::ref(f);
  pool().run(total, grain, fn);
}

// ---- storage -----------------------------------------------------------------

struct CpuStorage : TensorStorage {
  std::vector<float> v;
};

std::vector<int64_t> contig_strides(const Shape& s) {
  std::vector<int64_t> st(s.size());
  int64_t acc = 1;
  for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i) {
    st[i] = acc;
    acc *= s[i];
  }
  return st;
}

int norm_axis(int axis, int ndim, const char* what) {
  int a = axis < 0 ? axis + ndim : axis;
  if (a < 0 || a >= ndim)
    fail(std::string(what) + ": axis " + std::to_string(axis) + " out of range for ndim " +
         std::to_string(ndim));
  return a;
}

// out[i] = x[sum idx_d * st[d]] for row-major idx over os.
void strided_copy(const float* x, const Shape& os, const std::vector<int64_t>& st, float* o) {
  const int r = static_cast<int>(os.size());
  const int64_t n = shape_numel(os);
  if (n == 0) return;
  if (r == 0) {
    o[0] = x[0];
    return;
  }
  const int64_t inner = os[r - 1];
  const int64_t sin = st[r - 1];
  const int64_t rows = n / inner;
  parallel_for(rows, std::max<int64_t>(1, 16384 / inner), [&](int64_t b, int64_t e) {
    std::vector<int64_t> idx(r, 0);
    int64_t rem = b, off = 0;
    for (int d = r - 2; d >= 0; --d) {
      idx[d] = rem % os[d];
      rem /= os[d];
      off += idx[d] * st[d];
    }
    for (int64_t row = b; row < e; ++row) {
      float* po = o + row * inner;
      const float* px = x + off;
      if (sin == 1) std::memcpy(po, px, sizeof(float) * inner);
      else if (sin == 0) std::fill(po, po + inner, px[0]);
      else for (int64_t i = 0; i < inner; ++i) po[i] = px[i * sin];
      for (int d = r - 2; d >= 0; --d) {
        ++idx[d];
        off += st[d];
        if (idx[d] < os[d]) break;
        off -= st[d] * os[d];
        idx[d] = 0;
      }
    }
  });
}

// ---- elementwise kernels -------------------------------------------------------

float gelu_exact(float x) { return 0.5f * x * (1.f + std::erf(x * 0.70710678118654752440f)); }
float gelu_tanh(float x) {
  const float k = 0.79788456080286535588f;  // sqrt(2/pi)
  return 0.5f * x * (1.f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

template <class F>
Tensor unary_apply(const float* x, int64_t n, float* o, F f) {
  parallel_for(n, 16384, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) o[i] = f(x[i]);
  });
  return {};
}

template <class F>
void binary_apply(F f, const float* a, const Shape& as, const float* b, const Shape& bs,
                  const Shape& os, float* o) {
  const int64_t n = shape_numel(os);
  if (n == 0) return;
  const int64_t na = shape_numel(as), nb = shape_numel(bs);
  if (as == bs) {
    parallel_for(n, 16384, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) o[i] = f(a[i], b[i]);
    });
    return;
  }
  if (nb == 1 && as == os) {
    const float y = b[0];
    parallel_for(n, 16384, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) o[i] = f(a[i], y);
    });
    return;
  }
  if (na == 1 && bs == os) {
    const float x = a[0];
    parallel_for(n, 16384, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) o[i] = f(x, b[i]);
    });
    return;
  }
  // General broadcast.
  const int r = static_cast<int>(os.size());
  std::vector<int64_t> sa(r, 0), sb(r, 0);
  {
    auto ca = contig_strides(as), cb = contig_strides(bs);
    for (int i = 0; i < r; ++i) {
      int ia = i - (r - static_cast<int>(as.size()));
      int ib = i - (r - static_cast<int>(bs.size()));
      if (ia >= 0 && as[ia] != 1) sa[i] = ca[ia];
      if (ib >= 0 && bs[ib] != 1) sb[i] = cb[ib];
    }
  }
  const int64_t inner = os[r - 1];
  const int64_t rows = n / inner;
  const int64_t ia_s = sa[r - 1], ib_s = sb[r - 1];
  const int64_t ichunk = 16384;
  const int64_t nch = (inner + ichunk - 1) / ichunk;
  parallel_for(rows * nch, std::max<int64_t>(1, 16384 / std::min(inner, ichunk)),
               [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      int64_t row = t / nch, c = t % nch;
      int64_t i0 = c * ichunk, i1 = std::min(inner, i0 + ichunk);
      int64_t offa = 0, offb = 0, rem = row;
      for (int d = r - 2; d >= 0; --d) {
        int64_t id = rem % os[d];
        rem /= os[d];
        offa += id * sa[d];
        offb += id * sb[d];
      }
      const float* pa = a + offa;
      const float* pb = b + offb;
      float* po = o + row * inner;
      if (ia_s && ib_s) for (int64_t i = i0; i < i1; ++i) po[i] = f(pa[i], pb[i]);
      else if (ia_s) { float y = pb[0]; for (int64_t i = i0; i < i1; ++i) po[i] = f(pa[i], y); }
      else if (ib_s) { float x = pa[0]; for (int64_t i = i0; i < i1; ++i) po[i] = f(x, pb[i]); }
      else { float v = f(pa[0], pb[0]); for (int64_t i = i0; i < i1; ++i) po[i] = v; }
    }
  });
}

// ---- GEMM (linear): out[m,n] = sum_k x[m,k] * w[n,k] ---------------------------

#ifdef UM_AVX2
inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}

inline float dot1(const float* a, const float* b, int64_t K) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  int64_t k = 0;
  for (; k + 16 <= K; k += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8), acc1);
  }
  for (; k + 8 <= K; k += 8)
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc0);
  float s = hsum256(_mm256_add_ps(acc0, acc1));
  for (; k < K; ++k) s += a[k] * b[k];
  return s;
}

// 4x4 tile: out[i*ldo + j] = dot(x_i, w_j).
inline void tile4x4(const float* x, const float* w, float* out, int64_t K, int64_t ldo) {
  const float *x0 = x, *x1 = x + K, *x2 = x + 2 * K, *x3 = x + 3 * K;
  const float *w0 = w, *w1 = w + K, *w2 = w + 2 * K, *w3 = w + 3 * K;
  __m256 c[4][4];
  for (auto& r : c) for (auto& v : r) v = _mm256_setzero_ps();
  int64_t k = 0;
  for (; k + 8 <= K; k += 8) {
    __m256 a0 = _mm256_loadu_ps(x0 + k), a1 = _mm256_loadu_ps(x1 + k);
    __m256 a2 = _mm256_loadu_ps(x2 + k), a3 = _mm256_loadu_ps(x3 + k);
    __m256 b = _mm256_loadu_ps(w0 + k);
    c[0][0] = _mm256_fmadd_ps(a0, b, c[0][0]); c[1][0] = _mm256_fmadd_ps(a1, b, c[1][0]);
    c[2][0] = _mm256_fmadd_ps(a2, b, c[2][0]); c[3][0] = _mm256_fmadd_ps(a3, b, c[3][0]);
    b = _mm256_loadu_ps(w1 + k);
    c[0][1] = _mm256_fmadd_ps(a0, b, c[0][1]); c[1][1] = _mm256_fmadd_ps(a1, b, c[1][1]);
    c[2][1] = _mm256_fmadd_ps(a2, b, c[2][1]); c[3][1] = _mm256_fmadd_ps(a3, b, c[3][1]);
    b = _mm256_loadu_ps(w2 + k);
    c[0][2] = _mm256_fmadd_ps(a0, b, c[0][2]); c[1][2] = _mm256_fmadd_ps(a1, b, c[1][2]);
    c[2][2] = _mm256_fmadd_ps(a2, b, c[2][2]); c[3][2] = _mm256_fmadd_ps(a3, b, c[3][2]);
    b = _mm256_loadu_ps(w3 + k);
    c[0][3] = _mm256_fmadd_ps(a0, b, c[0][3]); c[1][3] = _mm256_fmadd_ps(a1, b, c[1][3]);
    c[2][3] = _mm256_fmadd_ps(a2, b, c[2][3]); c[3][3] = _mm256_fmadd_ps(a3, b, c[3][3]);
  }
  const float* xs[4] = {x0, x1, x2, x3};
  const float* ws[4] = {w0, w1, w2, w3};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      float s = hsum256(c[i][j]);
      for (int64_t kk = k; kk < K; ++kk) s += xs[i][kk] * ws[j][kk];
      out[i * ldo + j] = s;
    }
}
#else
inline float dot1(const float* a, const float* b, int64_t K) {
  float s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t k = 0;
  for (; k + 8 <= K; k += 8)
    for (int l = 0; l < 8; ++l) s[l] += a[k + l] * b[k + l];
  float r = ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]));
  for (; k < K; ++k) r += a[k] * b[k];
  return r;
}
inline void tile4x4(const float* x, const float* w, float* out, int64_t K, int64_t ldo) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) out[i * ldo + j] = dot1(x + i * K, w + j * K, K);
}
#endif

void gemm_nt(const float* x, const float* w, const float* bias, float* out, int64_t M, int64_t N,
             int64_t K) {
  const int64_t MB = 32, NB = 64;
  const int64_t mblocks = (M + MB - 1) / MB, nblocks = (N + NB - 1) / NB;
  const int64_t tasks = mblocks * nblocks;
  const int64_t flops_per_task = std::min(MB, M) * std::min(NB, N) * std::max<int64_t>(K, 1);
  parallel_for(tasks, std::max<int64_t>(1, 200000 / std::max<int64_t>(1, flops_per_task)),
               [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      int64_t mb = t / nblocks, nb = t % nblocks;
      int64_t m0 = mb * MB, m1 = std::min(M, m0 + MB);
      int64_t n0 = nb * NB, n1 = std::min(N, n0 + NB);
      for (int64_t n = n0; n < n1; n += 4) {
        int64_t nj = std::min<int64_t>(4, n1 - n);
        for (int64_t m = m0; m < m1; m += 4) {
          int64_t mi = std::min<int64_t>(4, m1 - m);
          float* o = out + m * N + n;
          if (mi == 4 && nj == 4) {
            tile4x4(x + m * K, w + n * K, o, K, N);
          } else {
            for (int64_t i = 0; i < mi; ++i)
              for (int64_t j = 0; j < nj; ++j)
                o[i * N + j] = dot1(x + (m + i) * K, w + (n + j) * K, K);
          }
        }
        if (bias)
          for (int64_t m = m0; m < m1; ++m)
            for (int64_t j = 0; j < std::min<int64_t>(4, n1 - n); ++j) out[m * N + n + j] += bias[n + j];
      }
    }
  });
}

// ---- backend -------------------------------------------------------------------

class CpuBackend final : public Backend {
 public:
  std::string name() const override { return "cpu"; }

  Tensor from_host(const float* data, const Shape& shape) override {
    auto st = std::make_shared<CpuStorage>();
    int64_t n = shape_numel(shape);
    for (int64_t d : shape) if (d < 0) fail("negative dimension in " + shape_str(shape));
    st->v.assign(data, data + n);
    return Tensor(this, std::move(st), shape);
  }

  void to_host(const Tensor& t, float* out) override {
    std::memcpy(out, P(t), sizeof(float) * static_cast<size_t>(t.numel()));
  }

  Tensor unary(UnaryOp op, const Tensor& x) override {
    float* o;
    Tensor r = alloc(x.shape(), o);
    const float* p = P(x);
    const int64_t n = x.numel();
    switch (op) {
      case UnaryOp::Neg: unary_apply(p, n, o, [](float v) { return -v; }); break;
      case UnaryOp::Abs: unary_apply(p, n, o, [](float v) { return std::fabs(v); }); break;
      case UnaryOp::Square: unary_apply(p, n, o, [](float v) { return v * v; }); break;
      case UnaryOp::Sqrt: unary_apply(p, n, o, [](float v) { return std::sqrt(v); }); break;
      case UnaryOp::Rsqrt: unary_apply(p, n, o, [](float v) { return 1.f / std::sqrt(v); }); break;
      case UnaryOp::Reciprocal: unary_apply(p, n, o, [](float v) { return 1.f / v; }); break;
      case UnaryOp::Exp: unary_apply(p, n, o, [](float v) { return std::exp(v); }); break;
      case UnaryOp::Log: unary_apply(p, n, o, [](float v) { return std::log(v); }); break;
      case UnaryOp::Sin: unary_apply(p, n, o, [](float v) { return std::sin(v); }); break;
      case UnaryOp::Cos: unary_apply(p, n, o, [](float v) { return std::cos(v); }); break;
      case UnaryOp::Tanh: unary_apply(p, n, o, [](float v) { return std::tanh(v); }); break;
      case UnaryOp::Sigmoid: unary_apply(p, n, o, [](float v) { return 1.f / (1.f + std::exp(-v)); }); break;
      case UnaryOp::Relu: unary_apply(p, n, o, [](float v) { return v > 0.f ? v : 0.f; }); break;
      case UnaryOp::Silu: unary_apply(p, n, o, [](float v) { return v / (1.f + std::exp(-v)); }); break;
      case UnaryOp::Gelu: unary_apply(p, n, o, gelu_exact); break;
      case UnaryOp::GeluTanh:
      case UnaryOp::GeluNew: unary_apply(p, n, o, gelu_tanh); break;
    }
    return r;
  }

  Tensor affine(const Tensor& x, float scale, float shift) override {
    float* o;
    Tensor r = alloc(x.shape(), o);
    const float* p = P(x);
    unary_apply(p, x.numel(), o, [=](float v) { return v * scale + shift; });
    return r;
  }

  Tensor binary(BinaryOp op, const Tensor& a, const Tensor& b) override {
    Shape os = broadcast_shapes(a.shape(), b.shape());
    float* o;
    Tensor r = alloc(os, o);
    const float *pa = P(a), *pb = P(b);
    const Shape &as = a.shape(), &bs = b.shape();
    switch (op) {
      case BinaryOp::Add: binary_apply([](float x, float y) { return x + y; }, pa, as, pb, bs, os, o); break;
      case BinaryOp::Sub: binary_apply([](float x, float y) { return x - y; }, pa, as, pb, bs, os, o); break;
      case BinaryOp::Mul: binary_apply([](float x, float y) { return x * y; }, pa, as, pb, bs, os, o); break;
      case BinaryOp::Div: binary_apply([](float x, float y) { return x / y; }, pa, as, pb, bs, os, o); break;
      case BinaryOp::Max: binary_apply([](float x, float y) { return (x != x || y != y) ? x + y : (x > y ? x : y); }, pa, as, pb, bs, os, o); break;
      case BinaryOp::Min: binary_apply([](float x, float y) { return (x != x || y != y) ? x + y : (x < y ? x : y); }, pa, as, pb, bs, os, o); break;
      case BinaryOp::Pow: binary_apply([](float x, float y) { return std::pow(x, y); }, pa, as, pb, bs, os, o); break;
    }
    return r;
  }

  Tensor matmul(const Tensor& a, const Tensor& b) override {
    if (a.ndim() < 2 || b.ndim() < 2) fail("matmul needs >= 2-d operands");
    const int ra = a.ndim(), rb = b.ndim();
    const int64_t n = a.dim(-2), k = a.dim(-1), m = b.dim(-1);
    if (b.dim(-2) != k)
      fail("matmul: inner dims mismatch " + shape_str(a.shape()) + " @ " + shape_str(b.shape()));
    Shape ba(a.shape().begin(), a.shape().end() - 2), bb(b.shape().begin(), b.shape().end() - 2);
    Shape bo = broadcast_shapes(ba, bb);
    Shape os = bo;
    os.push_back(n);
    os.push_back(m);
    float* o;
    Tensor r = alloc(os, o);
    const float *pa = P(a), *pb = P(b);
    const int rbatch = static_cast<int>(bo.size());
    // batch strides (in matrices) for a and b
    std::vector<int64_t> sa(rbatch, 0), sb(rbatch, 0);
    {
      int64_t acc = 1;
      for (int i = ra - 3; i >= 0; --i) {
        int pos = i + (rbatch - (ra - 2));
        sa[pos] = a.shape()[i] == 1 ? 0 : acc;
        acc *= a.shape()[i];
      }
      acc = 1;
      for (int i = rb - 3; i >= 0; --i) {
        int pos = i + (rbatch - (rb - 2));
        sb[pos] = b.shape()[i] == 1 ? 0 : acc;
        acc *= b.shape()[i];
      }
    }
    const int64_t nbatch = shape_numel(bo);
    if (nbatch == 0 || n == 0 || m == 0) return r;
    if (k == 0) {
      std::fill(o, o + r.numel(), 0.f);
      return r;
    }
    const int64_t RB = 8;
    const int64_t rblocks = (n + RB - 1) / RB;
    const int64_t tasks = nbatch * rblocks;
    parallel_for(tasks, std::max<int64_t>(1, 65536 / std::max<int64_t>(1, RB * k * m)),
                 [&](int64_t t0, int64_t t1) {
      for (int64_t t = t0; t < t1; ++t) {
        int64_t bi = t / rblocks, rb_i = t % rblocks;
        int64_t offa = 0, offb = 0, rem = bi;
        for (int d = rbatch - 1; d >= 0; --d) {
          int64_t id = rem % bo[d];
          rem /= bo[d];
          offa += id * sa[d];
          offb += id * sb[d];
        }
        const float* A = pa + offa * n * k;
        const float* B = pb + offb * k * m;
        float* C = o + bi * n * m;
        int64_t i0 = rb_i * RB, i1 = std::min(n, i0 + RB);
        for (int64_t i = i0; i < i1; ++i) {
          float* c = C + i * m;
          const float* ar = A + i * k;
          std::fill(c, c + m, 0.f);
          int64_t kk = 0;
          for (; kk + 4 <= k; kk += 4) {
            const float a0 = ar[kk], a1 = ar[kk + 1], a2 = ar[kk + 2], a3 = ar[kk + 3];
            const float *b0 = B + kk * m, *b1 = b0 + m, *b2 = b1 + m, *b3 = b2 + m;
            for (int64_t j = 0; j < m; ++j) c[j] += a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j];
          }
          for (; kk < k; ++kk) {
            const float a0 = ar[kk];
            const float* b0 = B + kk * m;
            for (int64_t j = 0; j < m; ++j) c[j] += a0 * b0[j];
          }
        }
      }
    });
    return r;
  }

  Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b) override {
    if (x.ndim() < 1 || w.ndim() != 2) fail("linear: bad shapes " + shape_str(x.shape()) + ", " + shape_str(w.shape()));
    const int64_t K = x.dim(-1), N = w.dim(0);
    if (w.dim(1) != K)
      fail("linear: in_features mismatch " + shape_str(x.shape()) + " vs weight " + shape_str(w.shape()));
    if (b.valid() && (b.ndim() != 1 || b.dim(0) != N)) fail("linear: bad bias shape " + shape_str(b.shape()));
    Shape os = x.shape();
    os.back() = N;
    float* o;
    Tensor r = alloc(os, o);
    const int64_t M = K == 0 ? 0 : x.numel() / K;
    if (K == 0) {
      int64_t rows = shape_numel(Shape(os.begin(), os.end() - 1));
      for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < N; ++j) o[i * N + j] = b.valid() ? P(b)[j] : 0.f;
      return r;
    }
    gemm_nt(P(x), P(w), b.valid() ? P(b) : nullptr, o, M, N, K);
    return r;
  }

  Tensor softmax(const Tensor& x) override {
    if (x.ndim() == 0) fail("softmax on 0-d tensor");
    float* o;
    Tensor r = alloc(x.shape(), o);
    const float* p = P(x);
    const int64_t C = x.dim(-1);
    if (C == 0) return r;
    const int64_t rows = x.numel() / C;
    parallel_for(rows, std::max<int64_t>(1, 8192 / C), [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        const float* xi = p + i * C;
        float* oi = o + i * C;
        float mx = xi[0];
        for (int64_t j = 1; j < C; ++j) mx = std::max(mx, xi[j]);
        float s = 0.f;
        for (int64_t j = 0; j < C; ++j) {
          oi[j] = std::exp(xi[j] - mx);
          s += oi[j];
        }
        const float inv = 1.f / s;
        for (int64_t j = 0; j < C; ++j) oi[j] *= inv;
      }
    });
    return r;
  }

  Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps) override {
    if (x.ndim() == 0) fail("layer_norm on 0-d tensor");
    const int64_t C = x.dim(-1);
    if (weight.valid() && weight.numel() != C) fail("layer_norm: weight size mismatch");
    if (bias.valid() && bias.numel() != C) fail("layer_norm: bias size mismatch");
    float* o;
    Tensor r = alloc(x.shape(), o);
    if (C == 0) return r;
    const float* p = P(x);
    const float* pw = weight.valid() ? P(weight) : nullptr;
    const float* pb = bias.valid() ? P(bias) : nullptr;
    const int64_t rows = x.numel() / C;
    parallel_for(rows, std::max<int64_t>(1, 8192 / C), [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        const float* xi = p + i * C;
        float* oi = o + i * C;
        double s = 0;
        for (int64_t j = 0; j < C; ++j) s += xi[j];
        const float mu = static_cast<float>(s / C);
        double v = 0;
        for (int64_t j = 0; j < C; ++j) {
          double d = xi[j] - mu;
          v += d * d;
        }
        const float inv = 1.f / std::sqrt(static_cast<float>(v / C) + eps);
        for (int64_t j = 0; j < C; ++j) {
          float y = (xi[j] - mu) * inv;
          if (pw) y *= pw[j];
          if (pb) y += pb[j];
          oi[j] = y;
        }
      }
    });
    return r;
  }

  Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) override {
    if (x.ndim() == 0) fail("rms_norm on 0-d tensor");
    const int64_t C = x.dim(-1);
    if (weight.valid() && weight.numel() != C) fail("rms_norm: weight size mismatch");
    float* o;
    Tensor r = alloc(x.shape(), o);
    if (C == 0) return r;
    const float* p = P(x);
    const float* pw = weight.valid() ? P(weight) : nullptr;
    const int64_t rows = x.numel() / C;
    parallel_for(rows, std::max<int64_t>(1, 8192 / C), [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) {
        const float* xi = p + i * C;
        float* oi = o + i * C;
        double s = 0;
        for (int64_t j = 0; j < C; ++j) s += static_cast<double>(xi[j]) * xi[j];
        const float inv = 1.f / std::sqrt(static_cast<float>(s / C) + eps);
        for (int64_t j = 0; j < C; ++j) {
          float y = xi[j] * inv;
          if (pw) y *= pw[j];
          oi[j] = y;
        }
      }
    });
    return r;
  }

  Tensor reduce(ReduceOp op, const Tensor& x, int axis, bool keepdim) override {
    if (x.ndim() == 0) {
      if (axis != 0 && axis != -1) fail("reduce: bad axis for 0-d tensor");
      return x;
    }
    const int ax = norm_axis(axis, x.ndim(), "reduce");
    const Shape& xs = x.shape();
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < ax; ++i) outer *= xs[i];
    for (int i = ax + 1; i < x.ndim(); ++i) inner *= xs[i];
    const int64_t R = xs[ax];
    Shape os;
    for (int i = 0; i < x.ndim(); ++i) {
      if (i == ax) { if (keepdim) os.push_back(1); }
      else os.push_back(xs[i]);
    }
    if (R == 0 && (op == ReduceOp::Max || op == ReduceOp::Min)) fail("reduce: max/min over empty axis");
    float* o;
    Tensor r = alloc(os, o);
    const float* p = P(x);
    const int64_t ichunk = 512;
    const int64_t nch = (inner + ichunk - 1) / ichunk;
    parallel_for(outer * nch, std::max<int64_t>(1, 16384 / std::max<int64_t>(1, R * std::min(inner, ichunk))),
                 [&](int64_t t0, int64_t t1) {
      for (int64_t t = t0; t < t1; ++t) {
        const int64_t oi = t / nch, c = t % nch;
        const int64_t i0 = c * ichunk, i1 = std::min(inner, i0 + ichunk), w = i1 - i0;
        const float* base = p + oi * R * inner + i0;
        float* ob = o + oi * inner + i0;
        if (op == ReduceOp::Sum || op == ReduceOp::Mean) {
          double acc[512];
          for (int64_t i = 0; i < w; ++i) acc[i] = 0;
          for (int64_t rr = 0; rr < R; ++rr) {
            const float* row = base + rr * inner;
            for (int64_t i = 0; i < w; ++i) acc[i] += row[i];
          }
          if (op == ReduceOp::Mean) for (int64_t i = 0; i < w; ++i) ob[i] = static_cast<float>(acc[i] / R);
          else for (int64_t i = 0; i < w; ++i) ob[i] = static_cast<float>(acc[i]);
        } else {
          for (int64_t i = 0; i < w; ++i) ob[i] = base[i];
          for (int64_t rr = 1; rr < R; ++rr) {
            const float* row = base + rr * inner;
            if (op == ReduceOp::Max) for (int64_t i = 0; i < w; ++i) ob[i] = (row[i] > ob[i] || row[i] != row[i]) ? row[i] : ob[i];
            else for (int64_t i = 0; i < w; ++i) ob[i] = (row[i] < ob[i] || row[i] != row[i]) ? row[i] : ob[i];
          }
        }
      }
    });
    return r;
  }

  Tensor permute(const Tensor& x, const std::vector<int>& perm) override {
    const int r = x.ndim();
    if (static_cast<int>(perm.size()) != r) fail("permute: wrong number of axes");
    std::vector<int> pn(r);
    std::vector<bool> seen(r, false);
    for (int i = 0; i < r; ++i) {
      pn[i] = norm_axis(perm[i], r, "permute");
      if (seen[pn[i]]) fail("permute: repeated axis");
      seen[pn[i]] = true;
    }
    auto cs = contig_strides(x.shape());
    Shape os(r);
    std::vector<int64_t> st(r);
    for (int i = 0; i < r; ++i) {
      os[i] = x.shape()[pn[i]];
      st[i] = cs[pn[i]];
    }
    float* o;
    Tensor t = alloc(os, o);
    strided_copy(P(x), os, st, o);
    return t;
  }

  Tensor slice(const Tensor& x, int axis, int64_t start, int64_t stop) override {
    const int ax = norm_axis(axis, x.ndim(), "slice");
    const int64_t D = x.shape()[ax];
    if (start < 0) start += D;
    if (stop < 0) stop += D;
    start = std::clamp<int64_t>(start, 0, D);
    stop = std::clamp<int64_t>(stop, 0, D);
    if (stop < start) stop = start;
    Shape os = x.shape();
    os[ax] = stop - start;
    float* o;
    Tensor r = alloc(os, o);
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < ax; ++i) outer *= os[i];
    for (int i = ax + 1; i < x.ndim(); ++i) inner *= os[i];
    const float* p = P(x);
    const int64_t len = (stop - start) * inner;
    if (len == 0) return r;
    parallel_for(outer, std::max<int64_t>(1, 16384 / len), [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i)
        std::memcpy(o + i * len, p + (i * D + start) * inner, sizeof(float) * len);
    });
    return r;
  }

  Tensor concat(const std::vector<Tensor>& xs, int axis) override {
    if (xs.empty()) fail("concat: empty list");
    const int nd = xs[0].ndim();
    if (nd == 0) fail("concat: 0-d tensors");
    const int ax = norm_axis(axis, nd, "concat");
    Shape os = xs[0].shape();
    int64_t total = 0;
    for (const Tensor& t : xs) {
      if (t.ndim() != nd) fail("concat: rank mismatch");
      for (int i = 0; i < nd; ++i)
        if (i != ax && t.shape()[i] != os[i])
          fail("concat: shape mismatch " + shape_str(t.shape()) + " vs " + shape_str(os));
      total += t.shape()[ax];
    }
    os[ax] = total;
    float* o;
    Tensor r = alloc(os, o);
    int64_t outer = 1, inner = 1;
    for (int i = 0; i < ax; ++i) outer *= os[i];
    for (int i = ax + 1; i < nd; ++i) inner *= os[i];
    int64_t off = 0;
    for (const Tensor& t : xs) {
      const int64_t len = t.shape()[ax] * inner;
      const float* p = P(t);
      if (len > 0)
        for (int64_t i = 0; i < outer; ++i)
          std::memcpy(o + i * total * inner + off, p + i * len, sizeof(float) * len);
      off += len;
    }
    return r;
  }

  Tensor gather_rows(const Tensor& table, const std::vector<int32_t>& indices,
                     const Shape& index_shape) override {
    if (table.ndim() < 1) fail("gather_rows: table must be >= 1-d");
    if (static_cast<int64_t>(indices.size()) != shape_numel(index_shape))
      fail("gather_rows: indices / index_shape size mismatch");
    const int64_t V = table.dim(0);
    const int64_t row = V == 0 ? 0 : table.numel() / V;
    Shape os = index_shape;
    os.insert(os.end(), table.shape().begin() + 1, table.shape().end());
    float* o;
    Tensor r = alloc(os, o);
    const float* p = P(table);
    for (size_t i = 0; i < indices.size(); ++i) {
      int64_t id = indices[i];
      if (id < 0 || id >= V)
        fail("gather_rows: index " + std::to_string(id) + " out of range [0, " + std::to_string(V) + ")");
      std::memcpy(o + i * row, p + id * row, sizeof(float) * row);
    }
    return r;
  }

  Tensor broadcast_to(const Tensor& x, const Shape& shape) override {
    if (x.shape() == shape) return x;
    const int r = static_cast<int>(shape.size());
    const int xr = x.ndim();
    if (xr > r) fail("broadcast_to: " + shape_str(x.shape()) + " -> " + shape_str(shape));
    auto cs = contig_strides(x.shape());
    std::vector<int64_t> st(r, 0);
    for (int i = 0; i < r; ++i) {
      int ix = i - (r - xr);
      if (ix < 0) continue;
      int64_t d = x.shape()[ix];
      if (d == shape[i]) st[i] = cs[ix];
      else if (d != 1) fail("broadcast_to: " + shape_str(x.shape()) + " -> " + shape_str(shape));
    }
    float* o;
    Tensor t = alloc(shape, o);
    strided_copy(P(x), shape, st, o);
    return t;
  }

 private:
  Tensor alloc(const Shape& shape, float*& out) {
    auto st = std::make_shared<CpuStorage>();
    st->v.resize(static_cast<size_t>(shape_numel(shape)));
    out = st->v.data();
    return Tensor(this, std::move(st), shape);
  }

  const float* P(const Tensor& t) const {
    if (!t.valid()) fail("invalid (empty) tensor passed to op");
    if (t.backend() != this) fail("tensor belongs to a different backend");
    return static_cast<const CpuStorage*>(t.storage().get())->v.data();
  }
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }

}  // namespace um

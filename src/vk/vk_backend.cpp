// um::Backend on Vulkan compute. See vk_backend.h for the execution model and
// vk_context.cpp for command recording / pooling / synchronisation.
//
// Every op validates shapes exactly like the CPU reference (same failure
// cases), computes the output shape on the host, collapses broadcast/strided
// index spaces to the smallest rank, and records one (or a few) dispatches.
#include "vk/vk_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vk_context.h"

namespace um {
namespace {

using vkx::Bind;
using vkx::Buffer;
using vkx::Context;
using vkx::K;
using vkx::Spec;

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("um/vk: " + msg); }

// Owns a pooled buffer, or (parent != null) is a zero-copy view at byte
// offset `off` into the parent's buffer (contiguous slices). Tensors are
// immutable, so sharing is safe; the parent stays alive through `parent`.
struct VkStorage final : TensorStorage {
  std::shared_ptr<Context> ctx;
  Buffer buf;
  uint64_t off = 0;
  std::shared_ptr<TensorStorage> parent;
  ~VkStorage() override {
    if (ctx && !parent) ctx->release(buf);
  }
};

// A tensor's bytes inside a VkBuffer.
struct View {
  VkBuffer buf = VK_NULL_HANDLE;
  uint64_t off = 0;
  uint64_t range = VK_WHOLE_SIZE;
  View() = default;
  View(VkBuffer b, uint64_t o, uint64_t r) : buf(b), off(o), range(r) {}
  View(const Buffer& b) : buf(b.buf), off(0), range(VK_WHOLE_SIZE) {}  // NOLINT
  bool valid() const { return buf != VK_NULL_HANDLE; }
};


int norm_axis(int axis, int ndim, const char* what) {
  int a = axis < 0 ? axis + ndim : axis;
  if (a < 0 || a >= ndim)
    fail(std::string(what) + ": axis " + std::to_string(axis) + " out of range for ndim " +
         std::to_string(ndim));
  return a;
}

std::vector<int64_t> contig_strides(const Shape& s) {
  std::vector<int64_t> st(s.size());
  int64_t acc = 1;
  for (int i = static_cast<int>(s.size()) - 1; i >= 0; --i) {
    st[i] = acc;
    acc *= s[i];
  }
  return st;
}

// Strides of a contiguous tensor of shape `in` broadcast (right-aligned) to `out`.
std::vector<int64_t> bcast_strides(const Shape& in, const Shape& out) {
  const int r = static_cast<int>(out.size()), ri = static_cast<int>(in.size());
  auto cs = contig_strides(in);
  std::vector<int64_t> st(r, 0);
  for (int i = 0; i < r; ++i) {
    int ii = i - (r - ri);
    if (ii >= 0 && in[ii] != 1) st[i] = cs[ii];
  }
  return st;
}

// Index-space collapse: drop size-1 dims and merge adjacent dims that are
// jointly contiguous for every operand. keep_last keeps the innermost dim
// separate (RoPE needs its extent and stride).
struct Collapsed {
  std::vector<int64_t> shape;
  std::vector<std::vector<int64_t>> strides;
  int rank() const { return static_cast<int>(shape.size()); }
};

Collapsed collapse(const Shape& shape, const std::vector<std::vector<int64_t>>& strides,
                   bool keep_last = false) {
  Collapsed c;
  c.strides.resize(strides.size());
  const int r = static_cast<int>(shape.size());
  for (int i = 0; i < r; ++i) {
    const bool last = keep_last && i == r - 1;
    if (shape[i] == 1 && !last) continue;
    bool merge = !c.shape.empty() && !last;
    if (merge)
      for (size_t o = 0; o < strides.size(); ++o)
        if (c.strides[o].back() != strides[o][i] * shape[i]) {
          merge = false;
          break;
        }
    if (merge) {
      c.shape.back() *= shape[i];
      for (size_t o = 0; o < strides.size(); ++o) c.strides[o].back() = strides[o][i];
    } else {
      c.shape.push_back(shape[i]);
      for (size_t o = 0; o < strides.size(); ++o) c.strides[o].push_back(strides[o][i]);
    }
  }
  if (c.shape.empty()) {
    c.shape.push_back(1);
    for (auto& s : c.strides) s.push_back(0);
  }
  return c;
}

uint32_t u32(int64_t v) {
  if (v < 0 || v > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    fail("index value out of 32-bit range: " + std::to_string(v));
  return static_cast<uint32_t>(v);
}

}  // namespace

class VulkanBackend final : public Backend {
 public:
  explicit VulkanBackend(const VulkanBackendOptions& o) : ctx_(std::make_shared<Context>(o)) {
    if (const char* e = std::getenv("UM_VK_GEMM")) gemm_force_ = std::atoi(e);
    if (const char* e = std::getenv("UM_VK_COOP")) coop_force_ = std::atoi(e);
    if (const char* e = std::getenv("UM_VK_ATTN_TK")) attn_tk_force_ = std::atoi(e);
  }
  ~VulkanBackend() override {
    try {
      ctx_->finish();
    } catch (...) {
    }
  }

  Context& ctx() const { return *ctx_; }

  std::string name() const override { return "vulkan"; }

  // ---- host <-> device -------------------------------------------------------
  Tensor from_host(const float* data, const Shape& shape) override {
    Tensor t = alloc(shape);
    const int64_t n = t.numel();
    if (n > 0) ctx_->upload(storage(t)->buf, data, static_cast<uint64_t>(n) * 4);
    return t;
  }

  void to_host(const Tensor& t, float* out) override {
    const int64_t n = t.numel();
    const View b = buf(t);
    if (n > 0) ctx_->download(b.buf, b.off, out, static_cast<uint64_t>(n) * 4);
  }

  void flush() override { ctx_->finish(); }

  // ---- elementwise -----------------------------------------------------------
  Tensor unary(UnaryOp op, const Tensor& x) override {
    return unary_impl(static_cast<uint32_t>(op), x, 1.f, 0.f);
  }

  Tensor affine(const Tensor& x, float scale, float shift) override {
    return unary_impl(100u, x, scale, shift);
  }

  Tensor binary(BinaryOp op, const Tensor& a, const Tensor& b) override {
    const View ba = buf(a);
    const View bb = buf(b);
    Shape os = broadcast_shapes(a.shape(), b.shape());
    Tensor out = alloc(os);
    const int64_t n = out.numel();
    if (n == 0) return out;
    struct {
      uint32_t n, op, rank, mode;
      uint32_t shape[8], sa[8], sb[8];
    } pc{};
    pc.n = u32(n);
    pc.op = static_cast<uint32_t>(op);
    if (a.shape() == b.shape()) {
      pc.mode = 0;
    } else {
      Collapsed c = collapse(os, {bcast_strides(a.shape(), os), bcast_strides(b.shape(), os)});
      if (c.rank() > 8) fail("binary: rank > 8 after collapsing " + shape_str(os));
      pc.mode = 1;
      pc.rank = static_cast<uint32_t>(c.rank());
      for (int i = 0; i < c.rank(); ++i) {
        pc.shape[i] = u32(c.shape[i]);
        pc.sa[i] = u32(c.strides[0][i]);
        pc.sb[i] = u32(c.strides[1][i]);
      }
    }
    elementwise(K::Binary, {R(ba), R(bb), W(out)}, &pc,
                sizeof(pc), n);
    return out;
  }

  // ---- matmul / linear -------------------------------------------------------
  Tensor matmul(const Tensor& a, const Tensor& b) override {
    if (a.ndim() < 2 || b.ndim() < 2) fail("matmul needs >= 2-d operands");
    const int64_t n = a.dim(-2), k = a.dim(-1), m = b.dim(-1);
    if (b.dim(-2) != k)
      fail("matmul: inner dims mismatch " + shape_str(a.shape()) + " @ " + shape_str(b.shape()));
    Shape ba(a.shape().begin(), a.shape().end() - 2), bb(b.shape().begin(), b.shape().end() - 2);
    Shape bo = broadcast_shapes(ba, bb);
    Shape os = bo;
    os.push_back(n);
    os.push_back(m);
    const int64_t nbatch = shape_numel(bo);
    if (nbatch == 0 || n == 0 || m == 0) return alloc(os);
    if (k == 0) return zeros(os);
    // Batch strides in elements.
    auto sa = bcast_strides(ba, bo), sb = bcast_strides(bb, bo);
    for (auto& s : sa) s *= n * k;
    for (auto& s : sb) s *= k * m;
    Collapsed c = collapse(bo, {sa, sb});
    if (c.rank() > 4) {  // exotic broadcast pattern: materialise the operands
      Shape fa = bo, fb = bo;
      fa.push_back(n), fa.push_back(k);
      fb.push_back(k), fb.push_back(m);
      return matmul(broadcast_to(a, fa), broadcast_to(b, fb));
    }
    Tensor out = alloc(os);
    gemm(buf(a), buf(b), View(), buf(out), n, m, k, nbatch, c, false);
    return out;
  }

  Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b) override {
    if (x.ndim() < 1 || w.ndim() != 2)
      fail("linear: bad shapes " + shape_str(x.shape()) + ", " + shape_str(w.shape()));
    const int64_t K = x.dim(-1), N = w.dim(0);
    if (w.dim(1) != K)
      fail("linear: in_features mismatch " + shape_str(x.shape()) + " vs weight " + shape_str(w.shape()));
    if (b.valid() && (b.ndim() != 1 || b.dim(0) != N)) fail("linear: bad bias shape " + shape_str(b.shape()));
    Shape os = x.shape();
    os.back() = N;
    if (shape_numel(os) == 0) return alloc(os);
    if (K == 0) return b.valid() ? broadcast_to(b, os) : zeros(os);
    const int64_t M = x.numel() / K;
    Tensor out = alloc(os);
    const View bias = b.valid() ? buf(b) : View();
    if (M <= 8) {
      struct {
        uint32_t M, N, K, has_bias, cols_per_wg;
      } pc{u32(M), u32(N), u32(K), bias.valid() ? 1u : 0u, rows_per_wg()};
      const uint64_t groups = (static_cast<uint64_t>(N) + pc.cols_per_wg - 1) / pc.cols_per_wg;
      dispatch1d(K::Gemv, {},
                 {R(buf(x)), R(buf(w)), R(bias.valid() ? bias : View(ctx_->dummy())),
                  W(out)},
                 &pc, sizeof(pc), groups);
      return out;
    }
    Collapsed c;
    c.shape = {1};
    c.strides = {{0}, {0}};
    gemm(buf(x), buf(w), bias, buf(out), M, N, K, 1, c, true);
    return out;
  }

  // ---- row ops -----------------------------------------------------------------
  Tensor softmax(const Tensor& x) override {
    if (x.ndim() == 0) fail("softmax on 0-d tensor");
    const View bx = buf(x);
    Tensor out = alloc(x.shape());
    const int64_t C = x.dim(-1);
    if (C == 0 || x.numel() == 0) return out;
    const int64_t rows = x.numel() / C;
    struct {
      uint32_t rows, len, rows_per_wg;
    } pc{u32(rows), u32(C), rows_per_wg()};
    dispatch1d(K::Softmax, {}, {R(bx), W(buf(out))}, &pc, sizeof(pc),
               (static_cast<uint64_t>(rows) + pc.rows_per_wg - 1) / pc.rows_per_wg);
    return out;
  }

  Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps) override {
    if (x.ndim() == 0) fail("layer_norm on 0-d tensor");
    const int64_t C = x.dim(-1);
    if (weight.valid() && weight.numel() != C) fail("layer_norm: weight size mismatch");
    if (bias.valid() && bias.numel() != C) fail("layer_norm: bias size mismatch");
    return norm_impl(0, x, weight, bias, eps);
  }

  Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) override {
    if (x.ndim() == 0) fail("rms_norm on 0-d tensor");
    const int64_t C = x.dim(-1);
    if (weight.valid() && weight.numel() != C) fail("rms_norm: weight size mismatch");
    return norm_impl(1, x, weight, Tensor(), eps);
  }

  Tensor reduce(ReduceOp op, const Tensor& x, int axis, bool keepdim) override {
    const View bx = buf(x);
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
      if (i == ax) {
        if (keepdim) os.push_back(1);
      } else {
        os.push_back(xs[i]);
      }
    }
    if (R == 0 && (op == ReduceOp::Max || op == ReduceOp::Min)) fail("reduce: max/min over empty axis");
    if (shape_numel(os) == 0) return alloc(os);
    if (R == 0) return full(os, op == ReduceOp::Sum ? 0.f : std::numeric_limits<float>::quiet_NaN());
    Tensor out = alloc(os);
    const uint32_t opc = static_cast<uint32_t>(op);
    if (inner == 1) {
      struct {
        uint32_t rows, len, rows_per_wg, op;
      } pc{u32(outer), u32(R), rows_per_wg(), opc};
      dispatch1d(K::ReduceRows, {}, {this->R(bx), this->W(buf(out))}, &pc, sizeof(pc),
                 (static_cast<uint64_t>(outer) + pc.rows_per_wg - 1) / pc.rows_per_wg);
    } else {
      struct {
        uint32_t n, inner, R, op;
      } pc{u32(outer * inner), u32(inner), u32(R), opc};
      elementwise(K::ReduceCols, {this->R(bx), this->W(buf(out))}, &pc, sizeof(pc), outer * inner);
    }
    return out;
  }

  // ---- data movement -------------------------------------------------------------
  Tensor permute(const Tensor& x, const std::vector<int>& perm) override {
    const int r = x.ndim();
    if (static_cast<int>(perm.size()) != r) fail("permute: wrong number of axes");
    std::vector<int> pn(r);
    std::vector<bool> seen(r, false);
    bool identity = true;
    for (int i = 0; i < r; ++i) {
      pn[i] = norm_axis(perm[i], r, "permute");
      if (seen[pn[i]]) fail("permute: repeated axis");
      seen[pn[i]] = true;
      identity = identity && pn[i] == i;
    }
    buf(x);
    if (identity) return x;  // tensors are immutable: sharing storage is safe
    auto cs = contig_strides(x.shape());
    Shape os(r);
    std::vector<int64_t> st(r);
    for (int i = 0; i < r; ++i) {
      os[i] = x.shape()[pn[i]];
      st[i] = cs[pn[i]];
    }
    Tensor out = alloc(os);
    copy(x, out, os, st, 0, contig_strides(os), 0, false);
    return out;
  }

  Tensor slice(const Tensor& x, int axis, int64_t start, int64_t stop) override {
    const int ax = norm_axis(axis, x.ndim(), "slice");
    const int64_t D = x.shape()[ax];
    if (start < 0) start += D;
    if (stop < 0) stop += D;
    start = std::clamp<int64_t>(start, 0, D);
    stop = std::clamp<int64_t>(stop, 0, D);
    if (stop < start) stop = start;
    buf(x);
    if (start == 0 && stop == D) return x;
    Shape os = x.shape();
    os[ax] = stop - start;
    auto cs = contig_strides(x.shape());
    int64_t outer = 1;
    for (int i = 0; i < ax; ++i) outer *= os[i];
    if (outer == 1 && stop > start) {
      // Contiguous chunk: zero-copy view if the offset meets the descriptor
      // offset alignment (16 B on NVIDIA).
      VkStorage* ps = storage(x);
      const uint64_t off = ps->off + static_cast<uint64_t>(start * cs[ax]) * 4;
      if (off % ctx_->limits().min_storage_align == 0) {
        auto st = std::make_shared<VkStorage>();
        st->ctx = ctx_;
        st->buf = ps->buf;
        st->off = off;
        st->parent = ps->parent ? ps->parent : x.storage();
        return Tensor(this, std::move(st), os);
      }
    }
    Tensor out = alloc(os);
    if (out.numel() == 0) return out;
    copy(x, out, os, cs, start * cs[ax], contig_strides(os), 0, false);
    return out;
  }

  Tensor concat(const std::vector<Tensor>& xs, int axis) override {
    if (xs.empty()) fail("concat: empty list");
    const int nd = xs[0].ndim();
    if (nd == 0) fail("concat: 0-d tensors");
    const int ax = norm_axis(axis, nd, "concat");
    Shape os = xs[0].shape();
    int64_t total = 0;
    for (const Tensor& t : xs) {
      buf(t);
      if (t.ndim() != nd) fail("concat: rank mismatch");
      for (int i = 0; i < nd; ++i)
        if (i != ax && t.shape()[i] != os[i])
          fail("concat: shape mismatch " + shape_str(t.shape()) + " vs " + shape_str(os));
      total += t.shape()[ax];
    }
    os[ax] = total;
    Tensor out = alloc(os);
    if (out.numel() == 0) return out;
    auto ost = contig_strides(os);
    int64_t off = 0;
    bool first = true;
    for (const Tensor& t : xs) {
      if (t.numel() > 0) {
        copy(t, out, t.shape(), contig_strides(t.shape()), 0, ost, off * ost[ax], !first);
        first = false;
      }
      off += t.shape()[ax];
    }
    return out;
  }

  Tensor gather_rows(const Tensor& table, const std::vector<int32_t>& indices,
                     const Shape& index_shape) override {
    const View bt = buf(table);
    if (table.ndim() < 1) fail("gather_rows: table must be >= 1-d");
    if (static_cast<int64_t>(indices.size()) != shape_numel(index_shape))
      fail("gather_rows: indices / index_shape size mismatch");
    const int64_t V = table.dim(0);
    const int64_t row = V == 0 ? 0 : table.numel() / V;
    Shape os = index_shape;
    os.insert(os.end(), table.shape().begin() + 1, table.shape().end());
    for (int32_t id : indices)
      if (id < 0 || id >= V)
        fail("gather_rows: index " + std::to_string(id) + " out of range [0, " + std::to_string(V) + ")");
    Tensor out = alloc(os);
    const int64_t n = out.numel();
    if (n == 0) return out;
    Buffer idx = ctx_->acquire(indices.size() * 4);
    ctx_->upload(idx, indices.data(), indices.size() * 4);
    struct {
      uint32_t n, rowlen;
    } pc{u32(n), u32(row)};
    elementwise(K::GatherRows, {R(bt), R(idx), W(out)}, &pc,
                sizeof(pc), n);
    ctx_->release(idx);  // later reuse is ordered after this dispatch by hazard tracking
    return out;
  }

  Tensor broadcast_to(const Tensor& x, const Shape& shape) override {
    buf(x);
    if (x.shape() == shape) return x;
    const int r = static_cast<int>(shape.size());
    const int xr = x.ndim();
    if (xr > r) fail("broadcast_to: " + shape_str(x.shape()) + " -> " + shape_str(shape));
    for (int i = 0; i < r; ++i) {
      int ix = i - (r - xr);
      if (ix < 0) continue;
      int64_t d = x.shape()[ix];
      if (d != shape[i] && d != 1) fail("broadcast_to: " + shape_str(x.shape()) + " -> " + shape_str(shape));
    }
    Tensor out = alloc(shape);
    if (out.numel() == 0) return out;
    copy(x, out, shape, bcast_strides(x.shape(), shape), 0, contig_strides(shape), 0, false);
    return out;
  }

  // ---- fused composites ------------------------------------------------------------
  Tensor rope(const Tensor& x, const Tensor& cos, const Tensor& sin) override {
    const int64_t d = x.dim(-1);
    if (d % 2) fail("rope: last dim must be even");
    Shape os = broadcast_shapes(broadcast_shapes(x.shape(), cos.shape()), sin.shape());
    if (os.empty() || os.back() != d) return Backend::rope(x, cos, sin);
    return ternary(0, x, cos, sin, os);
  }

  Tensor modulate(const Tensor& x, const Tensor& shift, const Tensor& scale) override {
    Shape os = broadcast_shapes(broadcast_shapes(x.shape(), scale.shape()), shift.shape());
    return ternary(1, x, shift, scale, os);
  }

  Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, float scale,
                   const Tensor& bias) override {
    if (q.ndim() < 2 || k.ndim() < 2 || v.ndim() < 2) return Backend::attention(q, k, v, scale, bias);
    const int64_t Lq = q.dim(-2), D = q.dim(-1), Lk = k.dim(-2), DV = v.dim(-1);
    if (k.dim(-1) != D || v.dim(-2) != Lk || D > 128 || DV > 128 || D % 4 || DV % 4 || D < 4 ||
        DV < 4 || Lq == 0 || Lk == 0)
      return Backend::attention(q, k, v, scale, bias);
    Shape bq(q.shape().begin(), q.shape().end() - 2), bk(k.shape().begin(), k.shape().end() - 2),
        bv(v.shape().begin(), v.shape().end() - 2);
    Shape ss = broadcast_shapes(bq, bk);
    ss.push_back(Lq);
    ss.push_back(Lk);
    if (bias.valid()) {
      ss = broadcast_shapes(ss, bias.shape());
      if (ss[ss.size() - 2] != Lq || ss.back() != Lk) return Backend::attention(q, k, v, scale, bias);
    }
    Shape bs(ss.begin(), ss.end() - 2);
    bs = broadcast_shapes(bs, bv);
    Shape os = bs;
    os.push_back(Lq);
    os.push_back(DV);
    const int64_t nbatch = shape_numel(bs);
    if (nbatch == 0) return alloc(os);
    auto mul = [](std::vector<int64_t> s, int64_t f) {
      for (auto& x : s) x *= f;
      return s;
    };
    std::vector<int64_t> sq = mul(bcast_strides(bq, bs), Lq * D);
    std::vector<int64_t> sk = mul(bcast_strides(bk, bs), Lk * D);
    std::vector<int64_t> sv = mul(bcast_strides(bv, bs), Lk * DV);
    std::vector<int64_t> sb(bs.size(), 0);
    int64_t bias_sq = 0, bias_sk = 0;
    if (bias.valid()) {
      Shape full = bs;
      full.push_back(Lq);
      full.push_back(Lk);
      auto st = bcast_strides(bias.shape(), full);
      std::copy(st.begin(), st.end() - 2, sb.begin());
      bias_sq = st[st.size() - 2];
      bias_sk = st.back();
    }
    std::vector<int64_t> so = mul(contig_strides(bs), 1);
    Collapsed c = collapse(bs, {sq, sk, sv, sb, so});
    if (c.rank() > 4) return Backend::attention(q, k, v, scale, bias);

    struct {
      uint32_t Lq, Lk, nbd, has_bias;
      uint32_t bshape[4], sq[4], sk[4], sv[4], sbias[4];
      uint32_t bias_sq, bias_sk;
      float scale;
      uint32_t nbatch;
    } pc{};
    static_assert(sizeof(pc) <= 128);
    pc.Lq = u32(Lq);
    pc.Lk = u32(Lk);
    pc.nbd = static_cast<uint32_t>(c.rank());
    pc.has_bias = bias.valid() ? 1u : 0u;
    for (int i = 0; i < c.rank(); ++i) {
      pc.bshape[i] = u32(c.shape[i]);
      pc.sq[i] = u32(c.strides[0][i]);
      pc.sk[i] = u32(c.strides[1][i]);
      pc.sv[i] = u32(c.strides[2][i]);
      pc.sbias[i] = u32(c.strides[3][i]);
    }
    pc.bias_sq = u32(bias_sq);
    pc.bias_sk = u32(bias_sk);
    pc.scale = scale;
    pc.nbatch = u32(nbatch);

    Tensor out = alloc(os);
    // Key tile of 16 rows: measured fastest on RTX 4070 Ti S (less shared memory
    // per workgroup -> higher occupancy; 8 is equal, 32 is ~1.5x slower).
    uint32_t tk = 16u;
    if (attn_tk_force_ == 8 || attn_tk_force_ == 16 || attn_tk_force_ == 32) tk = attn_tk_force_;
    const uint32_t gx = static_cast<uint32_t>((Lq + 31) / 32);
    const uint32_t gy = static_cast<uint32_t>(std::min<int64_t>(nbatch, 65535));
    const uint32_t gz = static_cast<uint32_t>((nbatch + gy - 1) / gy);
    const View bb = bias.valid() ? buf(bias) : View(ctx_->dummy());
    ctx_->dispatch(K::Attention, {u32(D), u32(DV), tk},
                   {R(q), R(k), R(v), R(bb),
                    W(out)},
                   &pc, sizeof(pc), gx, gy, gz);
    return out;
  }

 private:
  Tensor alloc(const Shape& shape) {
    for (int64_t d : shape)
      if (d < 0) fail("negative dimension in " + shape_str(shape));
    const int64_t n = shape_numel(shape);
    const uint64_t bytes = static_cast<uint64_t>(std::max<int64_t>(n, 1)) * 4;
    if (bytes > ctx_->limits().max_storage_range || n > static_cast<int64_t>(UINT32_MAX))
      fail("tensor too large for a storage buffer: " + shape_str(shape));
    auto st = std::make_shared<VkStorage>();
    st->buf = ctx_->acquire(bytes);
    st->ctx = ctx_;
    return Tensor(this, std::move(st), shape);
  }

  VkStorage* storage(const Tensor& t) const {
    if (!t.valid()) fail("invalid (empty) tensor passed to op");
    if (t.backend() != this) fail("tensor belongs to a different backend");
    return static_cast<VkStorage*>(t.storage().get());
  }
  View buf(const Tensor& t) const {
    VkStorage* s = storage(t);
    return View(s->buf.buf, s->off, static_cast<uint64_t>(std::max<int64_t>(t.numel(), 1)) * 4);
  }
  static Bind R(const View& v) { return Bind{v.buf, false, v.off, v.range}; }
  static Bind W(const View& v) { return Bind{v.buf, true, v.off, v.range}; }
  Bind R(const Tensor& t) const { return R(buf(t)); }
  Bind W(const Tensor& t) const { return W(buf(t)); }

  uint32_t rows_per_wg() const { return std::max(1u, 256u / ctx_->limits().subgroup_size); }

  void dispatch1d(K k, const Spec& spec, std::initializer_list<Bind> binds, const void* pc,
                  uint32_t pcb, uint64_t groups) {
    if (groups == 0) return;
    const uint64_t gx = std::min<uint64_t>(groups, ctx_->limits().max_groups[0]);
    const uint64_t gy = (groups + gx - 1) / gx;
    if (gy > ctx_->limits().max_groups[1]) fail("dispatch too large");
    ctx_->dispatch(k, spec, binds, pc, pcb, static_cast<uint32_t>(gx), static_cast<uint32_t>(gy), 1);
  }

  void elementwise(K k, std::initializer_list<Bind> binds, const void* pc, uint32_t pcb, int64_t n) {
    dispatch1d(k, {}, binds, pc, pcb, (static_cast<uint64_t>(n) + 255) / 256);
  }

  Tensor unary_impl(uint32_t op, const Tensor& x, float scale, float shift) {
    const View bx = buf(x);
    Tensor out = alloc(x.shape());
    const int64_t n = x.numel();
    if (n == 0) return out;
    struct {
      uint32_t n, op;
      float scale, shift;
    } pc{u32(n), op, scale, shift};
    elementwise(K::Unary, {R(bx), W(buf(out))}, &pc, sizeof(pc), n);
    return out;
  }

  Tensor norm_impl(uint32_t mode, const Tensor& x, const Tensor& w, const Tensor& b, float eps) {
    const View bx = buf(x);
    Tensor out = alloc(x.shape());
    const int64_t C = x.dim(-1);
    if (C == 0 || x.numel() == 0) return out;
    const int64_t rows = x.numel() / C;
    struct {
      uint32_t rows, len, rows_per_wg, mode, has_w, has_b;
      float eps;
    } pc{u32(rows), u32(C), rows_per_wg(), mode, w.valid() ? 1u : 0u, b.valid() ? 1u : 0u, eps};
    const View bw = w.valid() ? buf(w) : View(ctx_->dummy());
    const View bb = b.valid() ? buf(b) : View(ctx_->dummy());
    dispatch1d(K::Norm, {}, {R(bx), R(bw), R(bb), W(out)},
               &pc, sizeof(pc), (static_cast<uint64_t>(rows) + pc.rows_per_wg - 1) / pc.rows_per_wg);
    return out;
  }

  // dst[dst_off + idx.ds] = src[src_off + idx.ss] over `shape`.
  void copy(const Tensor& src, const Tensor& dst, const Shape& shape, const std::vector<int64_t>& ss,
            int64_t src_off, const std::vector<int64_t>& ds, int64_t dst_off, bool disjoint_write) {
    const int64_t n = shape_numel(shape);
    if (n == 0) return;
    Collapsed c = collapse(shape, {ss, ds});
    if (c.rank() > 8) fail("copy: rank > 8 after collapsing " + shape_str(shape));
    struct {
      uint32_t n, rank, src_off, dst_off;
      uint32_t shape[8], ss[8], ds[8];
    } pc{};
    pc.n = u32(n);
    pc.rank = static_cast<uint32_t>(c.rank());
    pc.src_off = u32(src_off);
    pc.dst_off = u32(dst_off);
    for (int i = 0; i < c.rank(); ++i) {
      pc.shape[i] = u32(c.shape[i]);
      pc.ss[i] = u32(c.strides[0][i]);
      pc.ds[i] = u32(c.strides[1][i]);
    }
    if (disjoint_write) ctx_->allow_disjoint_write(buf(dst).buf);
    elementwise(K::Copy, {R(src), W(dst)}, &pc, sizeof(pc), n);
  }

  Tensor ternary(uint32_t op, const Tensor& x, const Tensor& a, const Tensor& b, const Shape& os) {
    const View bx = buf(x);
    const View ba = buf(a);
    const View bb = buf(b);
    Collapsed c = collapse(os,
                           {bcast_strides(x.shape(), os), bcast_strides(a.shape(), os),
                            bcast_strides(b.shape(), os)},
                           op == 0);
    if (c.rank() > 6)
      return op == 0 ? Backend::rope(x, a, b) : Backend::modulate(x, a, b);
    Tensor out = alloc(os);
    const int64_t n = out.numel();
    if (n == 0) return out;
    struct {
      uint32_t n, rank, op, pad;
      uint32_t shape[6], sx[6], sa[6], sb[6];
    } pc{};
    pc.n = u32(n);
    pc.rank = static_cast<uint32_t>(c.rank());
    pc.op = op;
    for (int i = 0; i < c.rank(); ++i) {
      pc.shape[i] = u32(c.shape[i]);
      pc.sx[i] = u32(c.strides[0][i]);
      pc.sa[i] = u32(c.strides[1][i]);
      pc.sb[i] = u32(c.strides[2][i]);
    }
    elementwise(K::Ternary, {R(bx), R(ba), R(bb), W(out)},
                &pc, sizeof(pc), n);
    return out;
  }

  // Tile configurations for gemm.comp: {BM, BN, BK, TM, TN}.
  struct TileCfg {
    uint32_t bm, bn, bk, tm, tn;
  };
  static constexpr TileCfg kTiles[] = {
      {128, 128, 8, 8, 8},  // 0: large
      {64, 64, 16, 4, 4},   // 1: medium
      {32, 32, 16, 4, 4},   // 2: small
      {128, 64, 8, 8, 4},   // 3: tall
      {64, 128, 8, 4, 8},   // 4: wide
      {128, 128, 16, 8, 8}, // 5: large, deep k-step
      {128, 64, 16, 8, 4},  // 6: tall, deep k-step
  };

  int pick_tiles(int64_t M, int64_t N, int64_t batch) const {
    if (gemm_force_ >= 0 && gemm_force_ < 7) return gemm_force_;
    const int64_t sms = ctx_->limits().sm_count_hint;
    auto groups = [&](const TileCfg& t) { return ((M + t.bm - 1) / t.bm) * ((N + t.bn - 1) / t.bn) * batch; };
    if (groups(kTiles[0]) >= 2 * sms) return 0;
    if (groups(kTiles[1]) >= 2 * sms) return 1;
    return 2;
  }

  void gemm(const View A, const View B, const View bias, const View C, int64_t M,
            int64_t N, int64_t K, int64_t nbatch, const Collapsed& bc, bool trans_b) {
    struct {
      uint32_t M, N, K, has_bias;
      uint32_t nbatch, nbd, pad0, pad1;
      uint32_t bshape[4], sa[4], sb[4];
    } pc{};
    pc.M = u32(M);
    pc.N = u32(N);
    pc.K = u32(K);
    pc.has_bias = bias.valid() ? 1u : 0u;
    pc.nbatch = u32(nbatch);
    pc.nbd = static_cast<uint32_t>(bc.rank());
    for (int i = 0; i < bc.rank(); ++i) {
      pc.bshape[i] = u32(bc.shape[i]);
      pc.sa[i] = u32(bc.strides[0][i]);
      pc.sb[i] = u32(bc.strides[1][i]);
    }
    const Bind bias_b = R(bias.valid() ? bias : View(ctx_->dummy()));
    if (ctx_->coop() && M >= 16) {
      // fp16 tensor cores: {WM, WN, FM, FN} -> tile (WM*FM*16) x (WN*FN*16).
      static constexpr uint32_t kCoop[3][4] = {{2, 2, 2, 2}, {2, 2, 4, 4}, {4, 2, 2, 4}};
      int ci = coop_force_ >= 0 && coop_force_ < 3 ? coop_force_ : 2;
      const int64_t sms = ctx_->limits().sm_count_hint;
      auto tiles = [&](int i) {
        return ((M + kCoop[i][0] * kCoop[i][2] * 16 - 1) / (kCoop[i][0] * kCoop[i][2] * 16)) *
               ((N + kCoop[i][1] * kCoop[i][3] * 16 - 1) / (kCoop[i][1] * kCoop[i][3] * 16)) * nbatch;
      };
      if (coop_force_ < 0 && tiles(2) < 2 * sms) ci = 0;
      const uint32_t* c = kCoop[ci];
      const uint32_t bm = c[0] * c[2] * 16, bn = c[1] * c[3] * 16, sg = ctx_->limits().subgroup_size;
      const uint32_t gz = static_cast<uint32_t>(std::min<int64_t>(nbatch, ctx_->limits().max_groups[2]));
      ctx_->dispatch(K::GemmCoop, {c[0], c[1], c[2], c[3], trans_b ? 1u : 0u, sg, c[0] * c[1] * sg},
                     {R(A), R(B), bias_b, W(C)}, &pc, sizeof(pc), static_cast<uint32_t>((N + bn - 1) / bn),
                     static_cast<uint32_t>((M + bm - 1) / bm), gz);
      return;
    }
    const TileCfg& t = kTiles[pick_tiles(M, N, nbatch)];
    const uint64_t gx = (N + t.bn - 1) / t.bn, gy = (M + t.bm - 1) / t.bm;
    if (gx > ctx_->limits().max_groups[0] || gy > ctx_->limits().max_groups[1])
      fail("matmul too large for one dispatch");
    const uint32_t gz = static_cast<uint32_t>(std::min<int64_t>(nbatch, ctx_->limits().max_groups[2]));
    const uint32_t threads = (t.bm / t.tm) * (t.bn / t.tn);
    ctx_->dispatch(K::Gemm, {t.bm, t.bn, t.bk, t.tm, t.tn, trans_b ? 1u : 0u, threads},
                   {R(A), R(B), bias_b, W(C)},
                   &pc, sizeof(pc), static_cast<uint32_t>(gx), static_cast<uint32_t>(gy), gz);
  }

  std::shared_ptr<Context> ctx_;
  int gemm_force_ = -1;
  int coop_force_ = -1;
  int attn_tk_force_ = 0;
};

// ---- public API -------------------------------------------------------------------

std::vector<VulkanDeviceInfo> vulkan_devices() {
  try {
    return vkx::enumerate_devices();
  } catch (...) {
    return {};
  }
}

bool vulkan_available() {
  for (const auto& d : vulkan_devices())
    if (d.usable) return true;
  return false;
}

std::unique_ptr<Backend> make_vulkan_backend(const VulkanBackendOptions& opts) {
  return std::make_unique<VulkanBackend>(opts);
}

bool is_vulkan_backend(const Backend& be) { return dynamic_cast<const VulkanBackend*>(&be) != nullptr; }

std::string vulkan_device_name(const Backend& be) {
  auto* v = dynamic_cast<const VulkanBackend*>(&be);
  return v ? v->ctx().device_name() : std::string();
}

VulkanMemoryStats vulkan_memory_stats(const Backend& be) {
  auto* v = dynamic_cast<const VulkanBackend*>(&be);
  return v ? v->ctx().memory_stats() : VulkanMemoryStats{};
}

VulkanCounters vulkan_counters(const Backend& be) {
  auto* v = dynamic_cast<const VulkanBackend*>(&be);
  return v ? v->ctx().counters() : VulkanCounters{};
}

void vulkan_trim(Backend& be) {
  if (auto* v = dynamic_cast<VulkanBackend*>(&be)) v->ctx().trim();
}

void vulkan_finish(Backend& be) {
  if (auto* v = dynamic_cast<VulkanBackend*>(&be)) v->ctx().finish();
}

}  // namespace um

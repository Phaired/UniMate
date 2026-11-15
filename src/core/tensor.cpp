// Shape helpers, Tensor methods, Backend non-virtual helpers, default composite
// ops (built only from backend primitives) and free-function sugar.
#include "tensor.h"

#include <algorithm>
#include <stdexcept>

namespace um {

namespace {
[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("um: " + msg); }

int norm_axis(int axis, int ndim, const char* what) {
  int a = axis < 0 ? axis + ndim : axis;
  if (a < 0 || a >= ndim)
    fail(std::string(what) + ": axis " + std::to_string(axis) + " out of range for ndim " +
         std::to_string(ndim));
  return a;
}

Backend& be_of(const Tensor& t) {
  if (!t.valid() || !t.backend()) fail("operation on an invalid (empty) tensor");
  return *t.backend();
}
}  // namespace

// ---- shape helpers ---------------------------------------------------------

int64_t shape_numel(const Shape& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

std::string shape_str(const Shape& s) {
  std::string r = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) r += ", ";
    r += std::to_string(s[i]);
  }
  return r + "]";
}

Shape broadcast_shapes(const Shape& a, const Shape& b) {
  size_t n = std::max(a.size(), b.size());
  Shape out(n);
  for (size_t i = 0; i < n; ++i) {
    int64_t da = i < n - a.size() ? 1 : a[i - (n - a.size())];
    int64_t db = i < n - b.size() ? 1 : b[i - (n - b.size())];
    if (da == db || db == 1) out[i] = da;
    else if (da == 1) out[i] = db;
    else fail("cannot broadcast shapes " + shape_str(a) + " and " + shape_str(b));
  }
  return out;
}

// ---- Tensor ----------------------------------------------------------------

int64_t Tensor::dim(int axis) const { return shape_[norm_axis(axis, ndim(), "dim")]; }

Tensor Tensor::reshape(Shape ns) const {
  if (!valid()) fail("reshape on invalid tensor");
  int infer = -1;
  int64_t known = 1;
  for (size_t i = 0; i < ns.size(); ++i) {
    if (ns[i] == -1) {
      if (infer >= 0) fail("reshape: more than one -1 in " + shape_str(ns));
      infer = static_cast<int>(i);
    } else {
      if (ns[i] < 0) fail("reshape: negative dimension in " + shape_str(ns));
      known *= ns[i];
    }
  }
  int64_t n = numel();
  if (infer >= 0) {
    if (known == 0 || n % known != 0)
      fail("reshape: cannot infer -1 for " + shape_str(shape_) + " -> " + shape_str(ns));
    ns[infer] = n / known;
  } else if (known != n) {
    fail("reshape: numel mismatch " + shape_str(shape_) + " -> " + shape_str(ns));
  }
  return Tensor(be_, storage_, std::move(ns));
}

Tensor Tensor::unsqueeze(int axis) const {
  int a = norm_axis(axis, ndim() + 1, "unsqueeze");
  Shape s = shape_;
  s.insert(s.begin() + a, 1);
  return reshape(std::move(s));
}

Tensor Tensor::squeeze(int axis) const {
  int a = norm_axis(axis, ndim(), "squeeze");
  if (shape_[a] != 1) return *this;  // torch semantics: no-op
  Shape s = shape_;
  s.erase(s.begin() + a);
  return reshape(std::move(s));
}

// Inclusive end_axis, like torch.flatten.
Tensor Tensor::flatten(int start_axis, int end_axis) const {
  int s = norm_axis(start_axis, ndim(), "flatten");
  int e = norm_axis(end_axis, ndim(), "flatten");
  if (s > e) fail("flatten: start_axis > end_axis");
  Shape ns(shape_.begin(), shape_.begin() + s);
  int64_t m = 1;
  for (int i = s; i <= e; ++i) m *= shape_[i];
  ns.push_back(m);
  ns.insert(ns.end(), shape_.begin() + e + 1, shape_.end());
  return reshape(std::move(ns));
}

std::vector<float> Tensor::to_vector() const {
  Backend& be = be_of(*this);
  std::vector<float> v(static_cast<size_t>(numel()));
  be.to_host(*this, v.data());
  return v;
}

Tensor Tensor::permute(const std::vector<int>& perm) const { return be_of(*this).permute(*this, perm); }

Tensor Tensor::transpose(int a, int b) const {
  int x = norm_axis(a, ndim(), "transpose");
  int y = norm_axis(b, ndim(), "transpose");
  std::vector<int> perm(ndim());
  for (int i = 0; i < ndim(); ++i) perm[i] = i;
  std::swap(perm[x], perm[y]);
  return permute(perm);
}

Tensor Tensor::slice(int axis, int64_t start, int64_t stop) const {
  return be_of(*this).slice(*this, axis, start, stop);
}

// ---- Backend helpers -------------------------------------------------------

Tensor Backend::from_vector(const std::vector<float>& v, const Shape& shape) {
  if (static_cast<int64_t>(v.size()) != shape_numel(shape))
    fail("from_vector: " + std::to_string(v.size()) + " values for shape " + shape_str(shape));
  return from_host(v.data(), shape);
}

Tensor Backend::full(const Shape& shape, float value) {
  std::vector<float> v(static_cast<size_t>(shape_numel(shape)), value);
  return from_host(v.data(), shape);
}

Tensor Backend::map_host(const Tensor& x, const Shape& out_shape,
                         const std::function<void(const float*, float*)>& fn) {
  std::vector<float> in = x.to_vector();
  std::vector<float> out(static_cast<size_t>(shape_numel(out_shape)));
  fn(in.data(), out.data());
  return from_host(out.data(), out_shape);
}

// ---- default composite ops -------------------------------------------------

Tensor Backend::rope(const Tensor& x, const Tensor& cos, const Tensor& sin) {
  int64_t d = x.dim(-1);
  if (d % 2) fail("rope: last dim must be even");
  Tensor x1 = slice(x, -1, 0, d / 2);
  Tensor x2 = slice(x, -1, d / 2, d);
  Tensor rot = concat({unary(UnaryOp::Neg, x2), x1}, -1);
  return binary(BinaryOp::Add, binary(BinaryOp::Mul, x, cos), binary(BinaryOp::Mul, rot, sin));
}

Tensor Backend::attention(const Tensor& q, const Tensor& k, const Tensor& v, float scale,
                          const Tensor& bias) {
  Tensor kt = k.transpose(-1, -2);
  Tensor s = affine(matmul(q, kt), scale, 0.f);
  if (bias.valid()) s = binary(BinaryOp::Add, s, bias);
  return matmul(softmax(s), v);
}

Tensor Backend::modulate(const Tensor& x, const Tensor& shift, const Tensor& scale) {
  Tensor one_plus = affine(scale, 1.f, 1.f);
  return binary(BinaryOp::Add, binary(BinaryOp::Mul, x, one_plus), shift);
}

// ---- free-function sugar ---------------------------------------------------

Tensor operator+(const Tensor& a, const Tensor& b) { return be_of(a).binary(BinaryOp::Add, a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return be_of(a).binary(BinaryOp::Sub, a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return be_of(a).binary(BinaryOp::Mul, a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return be_of(a).binary(BinaryOp::Div, a, b); }
Tensor operator+(const Tensor& a, float b) { return be_of(a).affine(a, 1.f, b); }
Tensor operator*(const Tensor& a, float b) { return be_of(a).affine(a, b, 0.f); }
Tensor operator-(const Tensor& a) { return be_of(a).unary(UnaryOp::Neg, a); }

Tensor silu(const Tensor& x) { return be_of(x).unary(UnaryOp::Silu, x); }
Tensor gelu(const Tensor& x) { return be_of(x).unary(UnaryOp::Gelu, x); }
Tensor relu(const Tensor& x) { return be_of(x).unary(UnaryOp::Relu, x); }
Tensor matmul(const Tensor& a, const Tensor& b) { return be_of(a).matmul(a, b); }
Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b) { return be_of(x).linear(x, w, b); }
Tensor softmax(const Tensor& x) { return be_of(x).softmax(x); }
Tensor concat(const std::vector<Tensor>& xs, int axis) {
  if (xs.empty()) fail("concat: empty list");
  return be_of(xs[0]).concat(xs, axis);
}
Tensor sum(const Tensor& x, int axis, bool keepdim) {
  return be_of(x).reduce(ReduceOp::Sum, x, axis, keepdim);
}
Tensor mean(const Tensor& x, int axis, bool keepdim) {
  return be_of(x).reduce(ReduceOp::Mean, x, axis, keepdim);
}

}  // namespace um

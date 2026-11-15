// um::Tensor / um::Backend -- the contract every module is written against.
//
// A Tensor is a cheap value handle to an immutable, contiguous, row-major
// float32 array that lives on a Backend (CPU reference or Vulkan compute).
// Model code (T5, denoiser) only talks to the abstract Backend, so the same
// code runs on both and the CPU backend serves as the numerical reference for
// the Vulkan one.
//
// Conventions
//   * float32 only. Integer data (token ids, joint depths, lengths, masks)
//     stays on the host; masks reach the device as additive float biases
//     (0 / large negative) or 0/1 multipliers.
//   * Ops are functional: they return new tensors and never mutate inputs.
//   * Shapes follow PyTorch semantics; negative axes count from the end.
//   * Binary ops use NumPy broadcasting.
//   * reshape() is free (shares storage) because tensors are always contiguous.
//   * Backends may defer execution; to_host() / flush() force it.
//   * Not thread-safe: use a Backend from one thread at a time.
#pragma once

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace um {

using Shape = std::vector<int64_t>;

int64_t shape_numel(const Shape& s);
std::string shape_str(const Shape& s);
// NumPy broadcast of two shapes; throws std::runtime_error if incompatible.
Shape broadcast_shapes(const Shape& a, const Shape& b);

class Backend;

// Backend-private storage (host vector, VkBuffer, ...).
struct TensorStorage {
  virtual ~TensorStorage() = default;
};

class Tensor {
 public:
  Tensor() = default;
  Tensor(Backend* be, std::shared_ptr<TensorStorage> storage, Shape shape)
      : be_(be), storage_(std::move(storage)), shape_(std::move(shape)) {}

  bool valid() const { return storage_ != nullptr; }
  explicit operator bool() const { return valid(); }

  Backend* backend() const { return be_; }
  const std::shared_ptr<TensorStorage>& storage() const { return storage_; }
  const Shape& shape() const { return shape_; }
  int ndim() const { return static_cast<int>(shape_.size()); }
  int64_t dim(int axis) const;  // negative axis allowed
  int64_t numel() const { return shape_numel(shape_); }

  // Zero-copy views of the same storage. One dimension may be -1.
  Tensor reshape(Shape new_shape) const;
  Tensor unsqueeze(int axis) const;
  Tensor squeeze(int axis) const;
  Tensor flatten(int start_axis, int end_axis) const;

  // Convenience wrappers around the backend (defined in tensor.cpp).
  std::vector<float> to_vector() const;
  Tensor permute(const std::vector<int>& perm) const;
  Tensor transpose(int a, int b) const;
  Tensor slice(int axis, int64_t start, int64_t stop) const;

 private:
  Backend* be_ = nullptr;
  std::shared_ptr<TensorStorage> storage_;
  Shape shape_;
};

enum class UnaryOp {
  Neg, Abs, Square, Sqrt, Rsqrt, Reciprocal, Exp, Log, Sin, Cos, Tanh,
  Sigmoid, Relu, Silu,
  Gelu,      // exact: 0.5 x (1 + erf(x / sqrt 2))   (torch default)
  GeluTanh,  // tanh approximation                     (torch approximate='tanh')
  GeluNew,   // HF "gelu_new" (same formula as GeluTanh; kept for name parity)
};

enum class BinaryOp { Add, Sub, Mul, Div, Max, Min, Pow };

enum class ReduceOp { Sum, Mean, Max, Min };

class Backend {
 public:
  virtual ~Backend() = default;

  virtual std::string name() const = 0;

  // ---- host <-> device ----------------------------------------------------
  virtual Tensor from_host(const float* data, const Shape& shape) = 0;
  virtual void to_host(const Tensor& t, float* out) = 0;  // forces execution
  virtual void flush() {}                                 // run deferred work

  Tensor from_vector(const std::vector<float>& v, const Shape& shape);
  Tensor full(const Shape& shape, float value);
  Tensor zeros(const Shape& shape) { return full(shape, 0.f); }
  Tensor scalar(float v) { return full({}, v); }

  // ---- primitive ops (every backend must implement these) -----------------
  virtual Tensor unary(UnaryOp op, const Tensor& x) = 0;
  // y = x * scale + shift (scalars).
  virtual Tensor affine(const Tensor& x, float scale, float shift) = 0;
  virtual Tensor binary(BinaryOp op, const Tensor& a, const Tensor& b) = 0;
  // a[..., n, k] @ b[..., k, m] -> [..., n, m]; leading dims broadcast.
  virtual Tensor matmul(const Tensor& a, const Tensor& b) = 0;
  // x[..., in] -> [..., out]; w is [out, in] (torch nn.Linear layout),
  // b is [out] or an invalid Tensor for "no bias".
  virtual Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b) = 0;
  // Softmax over the last axis.
  virtual Tensor softmax(const Tensor& x) = 0;
  // Over the last axis: (x - mean) / sqrt(var + eps) * weight + bias.
  // weight / bias may be invalid (elementwise_affine=False).
  virtual Tensor layer_norm(const Tensor& x, const Tensor& weight,
                            const Tensor& bias, float eps) = 0;
  // Over the last axis: x * rsqrt(mean(x^2) + eps) * weight
  // (LlamaRMSNorm and T5LayerNorm). weight may be invalid.
  virtual Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) = 0;
  virtual Tensor reduce(ReduceOp op, const Tensor& x, int axis, bool keepdim) = 0;
  virtual Tensor permute(const Tensor& x, const std::vector<int>& perm) = 0;
  // Elements [start, stop) along axis (step 1; negative indices allowed).
  virtual Tensor slice(const Tensor& x, int axis, int64_t start, int64_t stop) = 0;
  virtual Tensor concat(const std::vector<Tensor>& xs, int axis) = 0;
  // table[V, ...rest] indexed by `indices` (host ints) -> index_shape + rest.
  // nn.Embedding and generic row gather.
  virtual Tensor gather_rows(const Tensor& table,
                             const std::vector<int32_t>& indices,
                             const Shape& index_shape) = 0;
  // Materialised broadcast of x to `shape`.
  virtual Tensor broadcast_to(const Tensor& x, const Shape& shape) = 0;

  // ---- composite ops -------------------------------------------------------
  // Default implementations (tensor.cpp) are built from the primitives above;
  // a backend may override any of them with a fused kernel. Semantics must be
  // identical.
  //
  // RoPE: x * cos + rotate_half(x) * sin over the last axis, where
  // rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2]). cos/sin broadcast to x.
  virtual Tensor rope(const Tensor& x, const Tensor& cos, const Tensor& sin);
  // softmax(q @ k^T * scale + bias) @ v.
  //   q [..., Lq, d], k [..., Lk, d], v [..., Lk, dv]; `bias` is additive,
  //   broadcastable to [..., Lq, Lk], or invalid.
  virtual Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v,
                           float scale, const Tensor& bias);
  // x * (1 + scale) + shift   (adaLN "modulate").
  virtual Tensor modulate(const Tensor& x, const Tensor& shift, const Tensor& scale);

  // ---- escape hatch --------------------------------------------------------
  // Round-trips through host memory. Correct on every backend but slow: use it
  // for one-off conditioning math, never inside the per-layer hot loop.
  Tensor map_host(const Tensor& x, const Shape& out_shape,
                  const std::function<void(const float* in, float* out)>& fn);
};

// Reference backend: plain C++ loops (may use std::thread / OpenMP internally).
std::unique_ptr<Backend> make_cpu_backend();

// ---- free-function sugar ----------------------------------------------------
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);
Tensor operator+(const Tensor& a, float b);
Tensor operator*(const Tensor& a, float b);
Tensor operator-(const Tensor& a);

Tensor silu(const Tensor& x);
Tensor gelu(const Tensor& x);
Tensor relu(const Tensor& x);
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor linear(const Tensor& x, const Tensor& w, const Tensor& b);
Tensor softmax(const Tensor& x);
Tensor concat(const std::vector<Tensor>& xs, int axis);
Tensor sum(const Tensor& x, int axis, bool keepdim = false);
Tensor mean(const Tensor& x, int axis, bool keepdim = false);

}  // namespace um

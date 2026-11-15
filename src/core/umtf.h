// UMTF -- "UniMate Tensor File": the one container used for everything that
// crosses the Python <-> C++ boundary (model weights, tokenizer tables,
// converted asset bundles, normalisation stats, golden test tensors).
//
// Layout (little endian):
//   char[4]  magic   = "UMTF"
//   u32      version = 1
//   u64      json_len
//   char[json_len]   UTF-8 JSON header:
//       { "meta": { ...free-form... },
//         "tensors": [ { "name": str, "dtype": "f32"|"i32"|"i64"|"u8",
//                        "shape": [int...], "offset": int, "nbytes": int }, ... ] }
//   zero padding up to the next multiple of 64 bytes  -> start of data section
//   raw tensor bytes; each tensor's `offset` is relative to the start of the
//   data section and 64-byte aligned. Row-major (C order).
//
// The Python side is tools/python/umtf.py (write_umtf / read_umtf).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "tensor.h"

namespace um {

enum class DType { F32, I32, I64, U8 };
size_t dtype_size(DType t);
const char* dtype_name(DType t);

// A host-side n-d array with a dtype. Owns its bytes.
struct HostArray {
  DType dtype = DType::F32;
  Shape shape;
  std::vector<uint8_t> bytes;

  HostArray() = default;
  HostArray(DType dt, Shape s);                       // zero-initialised
  static HostArray from_f32(const std::vector<float>& v, Shape s);
  static HostArray from_i32(const std::vector<int32_t>& v, Shape s);
  static HostArray from_i64(const std::vector<int64_t>& v, Shape s);

  int64_t numel() const { return shape_numel(shape); }
  int64_t dim(int axis) const;
  template <class T> T* data() { return reinterpret_cast<T*>(bytes.data()); }
  template <class T> const T* data() const { return reinterpret_cast<const T*>(bytes.data()); }

  // Converting copies (any dtype -> requested type).
  std::vector<float> to_f32() const;
  std::vector<int32_t> to_i32() const;
  std::vector<int64_t> to_i64() const;

  // Upload as float32 (converting if needed).
  Tensor to_tensor(Backend& be) const;
  static HostArray from_tensor(const Tensor& t);
};

// Conditioning dictionaries mirror the upstream Python `cond` dict: same key
// names, same shapes (see reference/UniMate/unimate/dataset/conditioning.py).
using ArrayDict = std::map<std::string, HostArray>;

class UmtfFile {
 public:
  // Reads the header and the whole data section. Throws std::runtime_error.
  static UmtfFile load(const std::string& path);
  static void save(const std::string& path, const nlohmann::json& meta,
                   const ArrayDict& arrays);

  const nlohmann::json& meta() const { return meta_; }
  bool has(const std::string& name) const;
  const HostArray& get(const std::string& name) const;  // throws if missing
  std::vector<std::string> names() const;               // file order
  const ArrayDict& arrays() const { return arrays_; }

 private:
  nlohmann::json meta_;
  ArrayDict arrays_;
  std::vector<std::string> order_;
};

// Lazily uploads named float tensors from a UmtfFile to a backend and caches
// them. Model code does `w.get("blocks.0.attn.qkv.weight")`.
class WeightStore {
 public:
  WeightStore(Backend& be, std::shared_ptr<const UmtfFile> file, std::string prefix = "");
  bool has(const std::string& name) const;
  Tensor get(const std::string& name);             // throws if missing
  Tensor get_or_empty(const std::string& name);    // invalid Tensor if missing
  const HostArray& host(const std::string& name) const;
  const nlohmann::json& meta() const;
  WeightStore sub(const std::string& prefix) const;  // shares file + cache
  Backend& backend() const { return *be_; }

 private:
  Backend* be_;
  std::shared_ptr<const UmtfFile> file_;
  std::string prefix_;
  std::shared_ptr<std::unordered_map<std::string, Tensor>> cache_;
};

}  // namespace um

// UMTF container I/O, HostArray and WeightStore. Byte layout: see umtf.h.
#include "umtf.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace um {

namespace {
[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("umtf: " + msg); }

constexpr uint64_t kAlign = 64;
uint64_t pad_of(uint64_t n) { return (kAlign - n % kAlign) % kAlign; }

DType dtype_from_name(const std::string& s) {
  if (s == "f32") return DType::F32;
  if (s == "i32") return DType::I32;
  if (s == "i64") return DType::I64;
  if (s == "u8") return DType::U8;
  fail("unknown dtype '" + s + "'");
}

template <class T>
std::vector<T> convert_to(const HostArray& a) {
  const int64_t n = a.numel();
  std::vector<T> out(static_cast<size_t>(n));
  switch (a.dtype) {
    case DType::F32: { auto* p = a.data<float>();    for (int64_t i = 0; i < n; ++i) out[i] = static_cast<T>(p[i]); break; }
    case DType::I32: { auto* p = a.data<int32_t>();  for (int64_t i = 0; i < n; ++i) out[i] = static_cast<T>(p[i]); break; }
    case DType::I64: { auto* p = a.data<int64_t>();  for (int64_t i = 0; i < n; ++i) out[i] = static_cast<T>(p[i]); break; }
    case DType::U8:  { auto* p = a.data<uint8_t>();  for (int64_t i = 0; i < n; ++i) out[i] = static_cast<T>(p[i]); break; }
  }
  return out;
}

template <class T>
HostArray from_typed(DType dt, const std::vector<T>& v, Shape s) {
  if (static_cast<int64_t>(v.size()) != shape_numel(s))
    fail("HostArray: " + std::to_string(v.size()) + " values for shape " + shape_str(s));
  HostArray a(dt, std::move(s));
  if (!v.empty()) std::memcpy(a.bytes.data(), v.data(), v.size() * sizeof(T));
  return a;
}
}  // namespace

size_t dtype_size(DType t) {
  switch (t) {
    case DType::F32: return 4;
    case DType::I32: return 4;
    case DType::I64: return 8;
    case DType::U8: return 1;
  }
  return 0;
}

const char* dtype_name(DType t) {
  switch (t) {
    case DType::F32: return "f32";
    case DType::I32: return "i32";
    case DType::I64: return "i64";
    case DType::U8: return "u8";
  }
  return "?";
}

// ---- HostArray -----------------------------------------------------------------

HostArray::HostArray(DType dt, Shape s) : dtype(dt), shape(std::move(s)) {
  bytes.assign(static_cast<size_t>(shape_numel(shape)) * dtype_size(dt), 0);
}

HostArray HostArray::from_f32(const std::vector<float>& v, Shape s) { return from_typed(DType::F32, v, std::move(s)); }
HostArray HostArray::from_i32(const std::vector<int32_t>& v, Shape s) { return from_typed(DType::I32, v, std::move(s)); }
HostArray HostArray::from_i64(const std::vector<int64_t>& v, Shape s) { return from_typed(DType::I64, v, std::move(s)); }

int64_t HostArray::dim(int axis) const {
  int a = axis < 0 ? axis + static_cast<int>(shape.size()) : axis;
  if (a < 0 || a >= static_cast<int>(shape.size())) fail("HostArray::dim: axis out of range");
  return shape[a];
}

std::vector<float> HostArray::to_f32() const { return convert_to<float>(*this); }
std::vector<int32_t> HostArray::to_i32() const { return convert_to<int32_t>(*this); }
std::vector<int64_t> HostArray::to_i64() const { return convert_to<int64_t>(*this); }

Tensor HostArray::to_tensor(Backend& be) const {
  if (dtype == DType::F32) return be.from_host(data<float>(), shape);
  std::vector<float> v = to_f32();
  return be.from_host(v.data(), shape);
}

HostArray HostArray::from_tensor(const Tensor& t) { return from_f32(t.to_vector(), t.shape()); }

// ---- UmtfFile ------------------------------------------------------------------

UmtfFile UmtfFile::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) fail("cannot open '" + path + "'");
  const uint64_t fsize = static_cast<uint64_t>(f.tellg());
  f.seekg(0);
  if (fsize < 16) fail(path + ": file too small");
  char head[16];
  f.read(head, 16);
  if (std::memcmp(head, "UMTF", 4) != 0) fail(path + ": not a UMTF file");
  uint32_t version;
  uint64_t json_len;
  std::memcpy(&version, head + 4, 4);
  std::memcpy(&json_len, head + 8, 8);
  if (version != 1) fail(path + ": unsupported version " + std::to_string(version));
  if (json_len > fsize - 16) fail(path + ": truncated header");
  std::string js(static_cast<size_t>(json_len), '\0');
  f.read(js.data(), static_cast<std::streamsize>(json_len));
  nlohmann::json header;
  try {
    header = nlohmann::json::parse(js);
  } catch (const std::exception& e) {
    fail(path + ": bad JSON header: " + e.what());
  }
  const uint64_t data_start = 16 + json_len + pad_of(16 + json_len);

  UmtfFile out;
  out.meta_ = header.value("meta", nlohmann::json::object());
  if (!header.contains("tensors")) fail(path + ": header has no 'tensors'");
  for (const auto& e : header["tensors"]) {
    std::string name = e.at("name").get<std::string>();
    DType dt = dtype_from_name(e.at("dtype").get<std::string>());
    Shape shape = e.at("shape").get<Shape>();
    uint64_t off = e.at("offset").get<uint64_t>();
    uint64_t nbytes = e.at("nbytes").get<uint64_t>();
    for (int64_t d : shape) if (d < 0) fail(path + ": negative dim in '" + name + "'");
    if (nbytes != static_cast<uint64_t>(shape_numel(shape)) * dtype_size(dt))
      fail(path + ": nbytes/shape mismatch for '" + name + "'");
    if (data_start + off + nbytes > fsize) fail(path + ": tensor '" + name + "' beyond end of file");
    HostArray a(dt, shape);
    f.seekg(static_cast<std::streamoff>(data_start + off));
    if (nbytes) f.read(reinterpret_cast<char*>(a.bytes.data()), static_cast<std::streamsize>(nbytes));
    if (!f) fail(path + ": read error on '" + name + "'");
    if (out.arrays_.count(name)) fail(path + ": duplicate tensor '" + name + "'");
    out.arrays_.emplace(name, std::move(a));
    out.order_.push_back(name);
  }
  return out;
}

void UmtfFile::save(const std::string& path, const nlohmann::json& meta, const ArrayDict& arrays) {
  nlohmann::json entries = nlohmann::json::array();
  uint64_t offset = 0;
  for (const auto& [name, a] : arrays) {
    entries.push_back({{"name", name},
                       {"dtype", dtype_name(a.dtype)},
                       {"shape", a.shape},
                       {"offset", offset},
                       {"nbytes", static_cast<uint64_t>(a.bytes.size())}});
    offset += a.bytes.size() + pad_of(a.bytes.size());
  }
  nlohmann::json header = {{"meta", meta.is_null() ? nlohmann::json::object() : meta},
                           {"tensors", entries}};
  const std::string js = header.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) fail("cannot write '" + path + "'");
  const uint32_t version = 1;
  const uint64_t json_len = js.size();
  f.write("UMTF", 4);
  f.write(reinterpret_cast<const char*>(&version), 4);
  f.write(reinterpret_cast<const char*>(&json_len), 8);
  f.write(js.data(), static_cast<std::streamsize>(js.size()));
  static const char zeros[kAlign] = {0};
  f.write(zeros, static_cast<std::streamsize>(pad_of(16 + json_len)));
  for (const auto& [name, a] : arrays) {
    if (!a.bytes.empty()) f.write(reinterpret_cast<const char*>(a.bytes.data()), static_cast<std::streamsize>(a.bytes.size()));
    f.write(zeros, static_cast<std::streamsize>(pad_of(a.bytes.size())));
  }
  if (!f) fail("write error on '" + path + "'");
}

bool UmtfFile::has(const std::string& name) const { return arrays_.count(name) != 0; }

const HostArray& UmtfFile::get(const std::string& name) const {
  auto it = arrays_.find(name);
  if (it == arrays_.end()) fail("missing tensor '" + name + "'");
  return it->second;
}

std::vector<std::string> UmtfFile::names() const { return order_; }

// ---- WeightStore ---------------------------------------------------------------
// Name resolution: full = prefix.empty() ? name : prefix + "." + name.
// sub(p) appends p to the prefix with the same rule (sub("") is a no-op copy).

WeightStore::WeightStore(Backend& be, std::shared_ptr<const UmtfFile> file, std::string prefix)
    : be_(&be), file_(std::move(file)), prefix_(std::move(prefix)),
      cache_(std::make_shared<std::unordered_map<std::string, Tensor>>()) {
  if (!file_) fail("WeightStore: null file");
}

static std::string join_name(const std::string& prefix, const std::string& name) {
  if (prefix.empty()) return name;
  if (name.empty()) return prefix;
  return prefix + "." + name;
}

bool WeightStore::has(const std::string& name) const { return file_->has(join_name(prefix_, name)); }

Tensor WeightStore::get(const std::string& name) {
  const std::string full = join_name(prefix_, name);
  auto it = cache_->find(full);
  if (it != cache_->end()) return it->second;
  Tensor t = file_->get(full).to_tensor(*be_);  // throws if missing
  cache_->emplace(full, t);
  return t;
}

Tensor WeightStore::get_or_empty(const std::string& name) {
  return has(name) ? get(name) : Tensor();
}

const HostArray& WeightStore::host(const std::string& name) const {
  return file_->get(join_name(prefix_, name));
}

const nlohmann::json& WeightStore::meta() const { return file_->meta(); }

WeightStore WeightStore::sub(const std::string& prefix) const {
  WeightStore s = *this;
  s.prefix_ = join_name(prefix_, prefix);
  return s;
}

}  // namespace um

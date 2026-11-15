// Cross-language UMTF round trip. argv: <work_dir> <python-command-prefix> <xlang.py>
#include <cstdlib>
#include <filesystem>

#include "check.h"
#include "core/umtf.h"

using namespace um;

static int run(const std::string& cmd) {
  std::string full = "\"" + cmd + "\"";  // cmd.exe strips the outer quotes
  std::printf("$ %s\n", cmd.c_str());
  std::fflush(stdout);
  return std::system(full.c_str());
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::printf("usage: %s <work_dir> <python cmd> <xlang.py>\n", argv[0]);
    return 2;
  }
  namespace fs = std::filesystem;
  fs::path dir = argv[1];
  fs::create_directories(dir);
  std::string py = argv[2], script = argv[3];
  std::string from_py = (dir / "from_py.umtf").string(), from_cpp = (dir / "from_cpp.umtf").string();

  // Python -> C++
  CHECK(run(py + " \"" + script + "\" write \"" + from_py + "\"") == 0);
  UmtfFile f = UmtfFile::load(from_py);
  CHECK(f.meta()["text"] == "h\xC3\xA9llo \xE4\xB8\x96\xE7\x95\x8C");
  CHECK(f.meta()["nested"]["a"].size() == 3 && f.meta()["nested"]["a"][2].is_null());
  CHECK(f.names().size() == 7);
  const HostArray& w = f.get("w");
  CHECK(w.dtype == DType::F32 && w.shape == (Shape{2, 3, 4}));
  bool ok = true;
  for (int i = 0; i < 24; ++i) ok &= w.data<float>()[i] == i * 0.5f - 3.0f;
  CHECK(ok);
  CHECK(f.get("ids").dtype == DType::I32);
  CHECK(f.get("ids").to_i32() == (std::vector<int32_t>{1, -2, 3, 400000, 5, -6}));
  CHECK(f.get("big").dtype == DType::I64);
  CHECK(f.get("big").to_i64() == (std::vector<int64_t>{1LL << 40, -(1LL << 41), 7}));
  CHECK(f.get("mask").dtype == DType::U8);
  CHECK(f.get("mask").to_i32() == (std::vector<int32_t>{1, 0, 1, 1, 0}));
  CHECK(f.get("scalar").shape.empty() && f.get("scalar").to_f32()[0] == 3.25f);
  CHECK(f.get("empty").shape == (Shape{0, 5}));
  CHECK(f.get("odd").to_f32().size() == 7 && f.get("odd").to_f32()[3] == 0.f);

  // C++ -> Python: re-save the same content; python checks it equals its sample.
  UmtfFile::save(from_cpp, f.meta(), f.arrays());
  CHECK(run(py + " \"" + script + "\" check \"" + from_cpp + "\"") == 0);
  return finish();
}

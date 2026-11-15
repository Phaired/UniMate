// Micro-benchmark of the Vulkan backend at the UniMate / T5 workload shapes.
//   um_vk_bench [--quick]
// Times are wall-clock per op, averaged over many back-to-back ops with a
// single flush at the end (i.e. the deferred-execution steady state).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/tensor.h"
#include "vk/vk_backend.h"

using namespace um;
using Clock = std::chrono::steady_clock;

static std::vector<float> rnd(int64_t n, float a = 1.f) {
  std::mt19937 r(7);
  std::uniform_real_distribution<float> d(-a, a);
  std::vector<float> v(static_cast<size_t>(n));
  for (auto& x : v) x = d(r);
  return v;
}
static Tensor make(Backend& be, const Shape& s, float a = 1.f) {
  auto v = rnd(shape_numel(s), a);
  return be.from_host(v.data(), s);
}

// Returns ms per call.
static double timeit(Backend& be, int iters, const std::function<Tensor()>& f) {
  for (int i = 0; i < 3; ++i) f();
  be.flush();
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) f();
  be.flush();
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
}

static std::unique_ptr<Backend> make_with_env(const char* var, const char* val) {
#ifdef _WIN32
  _putenv_s(var, val);
#else
  setenv(var, val, 1);
#endif
  return make_vulkan_backend();
}

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
  if (!vulkan_available()) {
    std::printf("No usable Vulkan device.\n");
    return 0;
  }
  auto vk = make_vulkan_backend();
  Backend& be = *vk;
  std::printf("device: %s\n\n", vulkan_device_name(be).c_str());

  // ---- per-dispatch overhead ----------------------------------------------------
  {
    Tensor x = make(be, {1});
    const int N = quick ? 5000 : 20000;
    be.flush();
    auto c0 = vulkan_counters(be);
    auto t0 = Clock::now();
    Tensor y = x;
    for (int i = 0; i < N; ++i) y = be.affine(y, 1.0001f, 0.f);  // dependent chain
    auto t1 = Clock::now();
    be.flush();
    auto t2 = Clock::now();
    auto c1 = vulkan_counters(be);
    double rec = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
    double tot = std::chrono::duration<double, std::micro>(t2 - t0).count() / N;
    std::printf("dispatch overhead, dependent chain of %d tiny ops: record %.2f us/op, end-to-end %.2f us/op "
                "(%llu submits, %llu barriers)\n",
                N, rec, tot, (unsigned long long)(c1.submits - c0.submits),
                (unsigned long long)(c1.barriers - c0.barriers));
    t0 = Clock::now();
    for (int i = 0; i < N; ++i) be.affine(x, 1.0001f, 0.f);  // independent ops
    be.flush();
    t2 = Clock::now();
    std::printf("dispatch overhead, %d independent tiny ops: end-to-end %.2f us/op\n", N,
                std::chrono::duration<double, std::micro>(t2 - t0).count() / N);
    Tensor s = make(be, {1});
    auto h0 = Clock::now();
    float out;
    for (int i = 0; i < 200; ++i) be.to_host(be.affine(s, 1.f, 1.f), &out);
    std::printf("round trip (1 op + to_host of 1 float): %.1f us\n\n",
                std::chrono::duration<double, std::micro>(Clock::now() - h0).count() / 200);
  }

  // ---- elementwise bandwidth ----------------------------------------------------
  {
    Shape s = {2, 61, 71, 512};
    Tensor a = make(be, s), b = make(be, s), bias = make(be, {512});
    const double mb = shape_numel(s) * 4.0 / 1e6;
    double t = timeit(be, 200, [&] { return be.binary(BinaryOp::Add, a, b); });
    std::printf("add      %-22s %7.3f ms  %6.1f GB/s\n", shape_str(s).c_str(), t, 3 * mb / t);
    t = timeit(be, 200, [&] { return be.binary(BinaryOp::Add, a, bias); });
    std::printf("add bias %-22s %7.3f ms  %6.1f GB/s\n", shape_str(s).c_str(), t, 2 * mb / t);
    t = timeit(be, 200, [&] { return be.unary(UnaryOp::Silu, a); });
    std::printf("silu     %-22s %7.3f ms  %6.1f GB/s\n", shape_str(s).c_str(), t, 2 * mb / t);
    t = timeit(be, 200, [&] { return be.rms_norm(a, bias, 1e-6f); });
    std::printf("rms_norm %-22s %7.3f ms  %6.1f GB/s\n", shape_str(s).c_str(), t, 2 * mb / t);
    t = timeit(be, 200, [&] { return be.layer_norm(a, bias, bias, 1e-6f); });
    std::printf("layernrm %-22s %7.3f ms  %6.1f GB/s\n", shape_str(s).c_str(), t, 2 * mb / t);
    Tensor sh = make(be, {2, 1, 1, 512});
    t = timeit(be, 200, [&] { return be.modulate(a, sh, sh); });
    std::printf("modulate %-22s %7.3f ms  %6.1f GB/s\n", shape_str(s).c_str(), t, 2 * mb / t);
    Tensor a5 = a.reshape({2, 61, 71, 8, 64});
    t = timeit(be, 200, [&] { return be.permute(a5, {0, 2, 3, 1, 4}); });
    std::printf("permute  %-22s %7.3f ms  %6.1f GB/s\n", "[2,61,71,8,64]->F/J", t, 2 * mb / t);
    Tensor c = make(be, {2, 1, 71, 1, 64}), sn = make(be, {2, 1, 71, 1, 64});
    t = timeit(be, 200, [&] { return be.rope(a5, c, sn); });
    std::printf("rope     %-22s %7.3f ms  %6.1f GB/s\n\n", "[2,61,71,8,64]", t, 2 * mb / t);
  }

  // ---- linear / matmul ------------------------------------------------------------
  struct L {
    const char* name;
    int64_t M, K, N;
  };
  std::vector<L> lin = {
      {"denoiser qkv  ", 2 * 61 * 71, 512, 1536},  {"denoiser out  ", 2 * 61 * 71, 512, 512},
      {"denoiser ffn up", 2 * 61 * 71, 512, 2048}, {"denoiser ffn dn", 2 * 61 * 71, 2048, 512},
      {"t5 qkv(72x64) ", 72 * 64, 768, 2304},      {"t5 ffn up      ", 72 * 64, 768, 2048},
      {"t5 ffn down    ", 72 * 64, 2048, 768},     {"t5 1 prompt q  ", 12, 768, 768},
      {"adaLN (M=2)    ", 2, 512, 3072},            {"t5 emb-size     ", 1, 768, 32128},
  };
  const char* vnames[] = {"128x128", "64x64", "32x32", "128x64", "64x128", "128x128k16", "128x64k16"};
  std::printf("linear (ms per call, GFLOP/s); columns: auto, then forced tile variants\n");
  std::printf("%-16s %18s", "shape", "auto");
  for (auto* v : vnames) std::printf(" %16s", v);
  std::printf("\n");
  std::vector<std::unique_ptr<Backend>> variants;
  for (int v = 0; v < 7; ++v) variants.push_back(make_with_env("UM_VK_GEMM", std::to_string(v).c_str()));
  make_with_env("UM_VK_GEMM", "-1");
  for (auto& l : lin) {
    std::printf("%-16s", l.name);
    const double flop = 2.0 * l.M * l.K * l.N;
    const int iters = flop > 5e9 ? 20 : (flop > 5e8 ? 50 : 200);
    {
      Tensor x = make(be, {l.M, l.K}), w = make(be, {l.N, l.K}, 0.05f), b = make(be, {l.N});
      double t = timeit(be, iters, [&] { return be.linear(x, w, b); });
      std::printf(" %8.3f %7.0f  ", t, flop / t / 1e6);
    }
    for (auto& vb : variants) {
      if (l.M <= 8) {
        std::printf(" %16s", "(gemv)");
        continue;
      }
      Tensor x = make(*vb, {l.M, l.K}), w = make(*vb, {l.N, l.K}, 0.05f), b = make(*vb, {l.N});
      double t = timeit(*vb, iters, [&] { return vb->linear(x, w, b); });
      std::printf(" %8.3f %7.0f", t, flop / t / 1e6);
    }
    std::printf("\n");
  }
  {
    // Opt-in fp16 tensor-core path (VulkanBackendOptions::fp16).
    std::vector<std::unique_ptr<Backend>> hb;
    for (int c = 0; c < 3; ++c) {
#ifdef _WIN32
      _putenv_s("UM_VK_COOP", std::to_string(c).c_str());
#endif
      VulkanBackendOptions o;
      o.fp16 = true;
      hb.push_back(make_vulkan_backend(o));
    }
#ifdef _WIN32
    _putenv_s("UM_VK_COOP", "-1");
#endif
    std::printf("\nlinear, fp16 tensor cores (opt-in): tiles 64x64(4 sg) | 128x128(4 sg) | 128x128(8 sg)\n");
    for (auto& l : lin) {
      if (l.M <= 8) continue;
      const double flop = 2.0 * l.M * l.K * l.N;
      const int iters = flop > 5e9 ? 20 : (flop > 5e8 ? 50 : 200);
      std::printf("%-16s", l.name);
      for (auto& b : hb) {
        Tensor x = make(*b, {l.M, l.K}), w = make(*b, {l.N, l.K}, 0.05f), bb = make(*b, {l.N});
        double t = timeit(*b, iters, [&] { return b->linear(x, w, bb); });
        std::printf(" %8.3f %7.0f", t, flop / t / 1e6);
      }
      std::printf("\n");
    }
  }
  {
    struct MM {
      const char* name;
      Shape a, b;
    };
    std::vector<MM> mms = {{"q@kT spatial", {976, 71, 64}, {976, 64, 71}},
                           {"p@v spatial ", {976, 71, 71}, {976, 71, 64}},
                           {"q@kT t5      ", {864, 64, 64}, {864, 64, 64}},
                           {"square 2048  ", {2048, 2048}, {2048, 2048}}};
    std::printf("\nmatmul (auto tiles)\n");
    for (auto& m : mms) {
      Tensor a = make(be, m.a), b = make(be, m.b);
      const double flop = 2.0 * shape_numel(m.a) * m.b.back();
      double t = timeit(be, 50, [&] { return be.matmul(a, b); });
      std::printf("%-14s %-30s %8.3f ms %7.0f GFLOP/s\n", m.name, (shape_str(m.a) + "@" + shape_str(m.b)).c_str(),
                  t, flop / t / 1e6);
    }
  }

  // ---- attention --------------------------------------------------------------------
  {
    struct A {
      const char* name;
      Shape q, k, bias;
      float scale;
    };
    std::vector<A> as = {
        {"spatial  [2,61,8,71,64] +bias[2,1,8,71,71]", {2, 61, 8, 71, 64}, {2, 61, 8, 71, 64}, {2, 1, 8, 71, 71}, 0.125f},
        {"temporal [2,71,8,61,64]", {2, 71, 8, 61, 64}, {2, 71, 8, 61, 64}, {}, 0.125f},
        {"t5       [72,12,64,64] +bias[72,12,64,64]", {72, 12, 64, 64}, {72, 12, 64, 64}, {72, 12, 64, 64}, 1.f},
        {"t5 1 prompt [1,12,12,64] +bias", {1, 12, 12, 64}, {1, 12, 12, 64}, {1, 12, 12, 12}, 1.f},
    };
    auto wg32 = make_with_env("UM_VK_ATTN_TK", "8");
    auto wg64 = make_with_env("UM_VK_ATTN_TK", "32");
    make_with_env("UM_VK_ATTN_TK", "0");
    std::printf("\nattention (ms): fused auto(TK16) | fused TK8 | fused TK32 | unfused composite (matmul+softmax+matmul)\n");
    for (auto& a : as) {
      auto run = [&](Backend& b, bool fused) {
        Tensor q = make(b, a.q), k = make(b, a.k), v = make(b, a.k);
        Tensor bias = a.bias.empty() ? Tensor() : make(b, a.bias);
        return timeit(b, 50, [&] {
          return fused ? b.attention(q, k, v, a.scale, bias) : b.Backend::attention(q, k, v, a.scale, bias);
        });
      };
      std::printf("%-46s %8.3f %8.3f %8.3f %8.3f\n", a.name, run(be, true), run(*wg32, true), run(*wg64, true),
                  run(be, false));
    }
  }

  // ---- CPU reference for scale ----------------------------------------------------------
  if (!quick) {
    auto cpu = make_cpu_backend();
    Tensor x = make(*cpu, {2 * 61 * 71, 512}), w = make(*cpu, {2048, 512}), b = make(*cpu, {2048});
    double t = timeit(*cpu, 3, [&] { return cpu->linear(x, w, b); });
    std::printf("\nCPU reference: linear [8662,512]x[2048,512]^T %.2f ms\n", t);
  }
  auto ms = vulkan_memory_stats(be);
  std::printf("\npeak device memory %.1f MB\n", ms.peak_device_bytes / 1048576.0);
  return 0;
}

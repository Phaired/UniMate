#include "pipeline/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace um::pipeline {

namespace {
[[noreturn]] void fail(const std::string& m) { throw std::runtime_error("sample_flow: " + m); }

// Dormand-Prince-Shampine tableau (torchdiffeq dopri5.py), float64 constants;
// torchdiffeq casts them to the state dtype (float32) before use.
const double kAlpha[6] = {1 / 5., 3 / 10., 4 / 5., 8 / 9., 1., 1.};
const double kBeta[6][6] = {
    {1 / 5., 0, 0, 0, 0, 0},
    {3 / 40., 9 / 40., 0, 0, 0, 0},
    {44 / 45., -56 / 15., 32 / 9., 0, 0, 0},
    {19372 / 6561., -25360 / 2187., 64448 / 6561., -212 / 729., 0, 0},
    {9017 / 3168., -355 / 33., 46732 / 5247., 49 / 176., -5103 / 18656., 0},
    {35 / 384., 0, 500 / 1113., 125 / 192., -2187 / 6784., 11 / 84.},
};
const double kCErr[7] = {35 / 384. - 1951 / 21600., 0, 500 / 1113. - 22642 / 50085.,
                         125 / 192. - 451 / 720., -2187 / 6784. - -12231 / 42400.,
                         11 / 84. - 649 / 6300., -1. / 60.};
const double kMid[7] = {6025192743. / 30085553152. / 2, 0, 51252292925. / 65400821598. / 2,
                        -2691868925. / 45128329728. / 2, 187940372067. / 1594534317056. / 2,
                        -1776094331. / 19743644256. / 2, 11237099. / 235043384. / 2};

struct Counter {
  const VelocityFn& v;
  int nfe = 0;
  Tensor operator()(const Tensor& x, float t) {
    ++nfe;
    Tensor r = v(x, t);
    if (!r.valid() || r.shape() != x.shape())
      fail("velocity function returned shape " + shape_str(r.shape()) + " for x " +
           shape_str(x.shape()));
    return r;
  }
};

// torchdiffeq's norms are float64 (float32 state / float64 tolerance). The
// operands are read back and the RMS is taken in double on the host: these
// are the only readbacks of the sampler.
//   rms(num / (atol + rtol * max(|a|, |b|)))   (b may be invalid)
double rms_scaled(const Tensor& num, const Tensor& a, const Tensor& b, double atol, double rtol) {
  std::vector<float> n = num.to_vector(), av = a.to_vector(), bv;
  if (b.valid()) bv = b.to_vector();
  double s = 0;
  for (size_t i = 0; i < n.size(); ++i) {
    double m = std::abs(static_cast<double>(av[i]));
    if (b.valid()) m = std::max(m, std::abs(static_cast<double>(bv[i])));
    double q = static_cast<double>(n[i]) / (atol + rtol * m);
    s += q * q;
  }
  return std::sqrt(s / static_cast<double>(n.size()));
}

// torch.sum(k * c, dim=-1) over the small stage axis, in float32 with the
// summation structure of ATen's CPU inner reduction (4 accumulators,
// combined ((a0 + a1) + a2) + a3), which matches it far more often than a
// sequential sum.
Tensor lincomb(const std::vector<Tensor>& k, const float* c, int n) {
  Tensor lane[4];
  for (int j = 0; j < n; ++j) {
    Tensor term = k[j] * c[j];
    lane[j % 4] = lane[j % 4].valid() ? lane[j % 4] + term : term;
  }
  Tensor acc = lane[0];
  for (int l = 1; l < 4; ++l)
    if (lane[l].valid()) acc = acc + lane[l];
  return acc;
}

Tensor fixed_grid(Backend& be, Counter& f, const Tensor& y0, const SamplerOptions& opt,
                  bool heun, const ProgressFn& progress) {
  (void)be;
  if (opt.num_steps < 2) fail("num_steps must be >= 2 (grid points, upstream semantics)");
  std::vector<float> grid = linspace_f32(static_cast<float>(opt.t0), static_cast<float>(opt.t1),
                                         opt.num_steps);
  Tensor y = y0;
  const int n = static_cast<int>(grid.size()) - 1;
  for (int i = 0; i < n; ++i) {
    float t0 = grid[i], t1 = grid[i + 1];
    float dt = t1 - t0;
    Tensor k1 = f(y, t0);
    if (!heun) {
      y = y + k1 * dt;  // y0 + dt * f0
    } else {           // torchdiffeq heun2 (rk2_step_func)
      Tensor k2 = f(y + (k1 * dt) * 1.0f, t0 + dt * 1.0f);
      y = y + (k1 * 0.5f + k2 * 0.5f) * dt;
    }
    if (progress) progress(static_cast<float>(i + 1) / n, "sampling");
  }
  return y;
}

Tensor dopri5(Backend& be, Counter& f, const Tensor& y0in, const SamplerOptions& opt,
              const ProgressFn& progress, SolverStats* st) {
  const double atol = opt.atol, rtol = opt.rtol;
  // time grid: torch.linspace float32 -> float64 (AdaptiveStepsizeODESolver.integrate)
  const double t_start = static_cast<double>(static_cast<float>(opt.t0));
  const double t_end = static_cast<double>(static_cast<float>(opt.t1));
  if (!(t_end > t_start)) fail("dopri5 requires t1 > t0");
  float alpha[6], beta[6][6], cerr[7], mid[7];
  for (int i = 0; i < 6; ++i) {
    alpha[i] = static_cast<float>(kAlpha[i]);
    for (int j = 0; j < 6; ++j) beta[i][j] = static_cast<float>(kBeta[i][j]);
  }
  for (int j = 0; j < 7; ++j) cerr[j] = static_cast<float>(kCErr[j]), mid[j] = static_cast<float>(kMid[j]);

  Tensor y = y0in;
  Tensor fy = f(y, static_cast<float>(t_start));
  double dt;
  if (opt.first_step > 0) {
    dt = opt.first_step;
  } else {  // _select_initial_step(order = 5 - 1); scale = atol + |y0| rtol
    double d0 = rms_scaled(y, y, Tensor(), atol, rtol);
    double d1 = rms_scaled(fy, y, Tensor(), atol, rtol);
    double h0 = (d0 < 1e-5 || d1 < 1e-5) ? static_cast<double>(1e-6f) : 0.01 * d0 / d1;
    h0 = std::abs(h0);
    Tensor y1 = y + fy * static_cast<float>(h0);
    Tensor f1 = f(y1, static_cast<float>(t_start + h0));
    double d2 = std::abs(rms_scaled(f1 - fy, y, Tensor(), atol, rtol) / h0);
    double h1;
    if (d1 <= 1e-15 && d2 <= 1e-15) h1 = std::max(1e-6, h0 * 1e-3);
    else h1 = std::pow(0.01 / std::max(d1, d2), 1.0 / 5.0);
    dt = std::min(100 * h0, std::abs(h1));
  }
  if (st) st->first_step = dt;

  double t = t_start;
  int n_steps = 0;
  while (t_end > t) {
    if (++n_steps > opt.max_num_steps) fail("max_num_steps exceeded");
    if (!std::isfinite(dt)) dt = 0.0;
    dt = std::max(dt, 0.0);
    double t1 = t + dt;
    if (!(t1 > t)) fail("underflow in dt");

    // ---- _runge_kutta_step (float32 stage times and coefficients) ----
    const float t0f = static_cast<float>(t), dtf = static_cast<float>(dt),
                t1f = static_cast<float>(t1);
    std::vector<Tensor> k(7);
    k[0] = fy;
    Tensor yi;
    for (int i = 0; i < 6; ++i) {
      float ti = alpha[i] == 1.f ? std::nextafter(t1f, -std::numeric_limits<float>::infinity())
                                 : t0f + alpha[i] * dtf;
      float c[6];
      for (int j = 0; j <= i; ++j) c[j] = beta[i][j] * dtf;
      yi = y + lincomb(k, c, i + 1);
      k[i + 1] = f(yi, ti);
    }
    Tensor y1 = yi, f1 = k[6];
    float ce[7];
    for (int j = 0; j < 7; ++j) ce[j] = dtf * cerr[j];
    Tensor err = lincomb(k, ce, 7);

    // ---- error ratio ----
    double ratio = rms_scaled(err, y, y1, atol, rtol);
    bool accept = ratio <= 1.0;
    if (st) st->steps.push_back({t, dt, accept, ratio});

    if (accept) {
      if (st) ++st->accepted;
      if (t1 >= t_end) {
        // final step: 4th-order dense output at t_end (_interp_fit / _interp_evaluate)
        float cm[7];
        for (int j = 0; j < 7; ++j) cm[j] = dtf * mid[j];
        Tensor ymid = y + lincomb(k, cm, 7);
        const Tensor& F0 = k[0];
        const Tensor& F1 = k[6];
        Tensor a = ((F1 - F0) * (2.f * dtf) - (y1 + y) * 8.f) + ymid * 16.f;
        Tensor b = (((F0 * 5.f - F1 * 3.f) * dtf + y * 18.f) + y1 * 14.f) - ymid * 32.f;
        Tensor cc = (((F1 - F0 * 4.f) * dtf - y * 11.f) - y1 * 5.f) + ymid * 16.f;
        Tensor d = F0 * dtf;
        float x = static_cast<float>((t_end - t) / (t1 - t));
        Tensor total = y + d * x;
        float xp = x;
        xp = xp * x;
        total = total + cc * xp;
        xp = xp * x;
        total = total + b * xp;
        xp = xp * x;
        total = total + a * xp;
        y = total;
        t = t1;
        if (progress) progress(1.f, "sampling");
        break;
      }
      t = t1;
      y = y1;
      fy = f1;
    } else if (st) {
      ++st->rejected;
    }
    // ---- _optimal_step_size (order 5) ----
    double factor;
    if (ratio == 0) factor = opt.ifactor;
    else {
      double dfac = ratio < 1 ? 1.0 : opt.dfactor;
      factor = std::min(opt.ifactor, std::max(opt.safety / std::pow(ratio, 1.0 / 5.0), dfac));
    }
    dt = std::max(dt * factor, 0.0);
    if (progress) progress(static_cast<float>((t - t_start) / (t_end - t_start)), "sampling");
  }
  return y;
}
}  // namespace

std::vector<float> linspace_f32(float start, float end, int steps) {
  std::vector<float> v(std::max(steps, 0));
  if (steps <= 0) return v;
  if (steps == 1) {
    v[0] = start;
    return v;
  }
  const float step = (end - start) / static_cast<float>(steps - 1);
  const int halfway = steps / 2;
  for (int i = 0; i < steps; ++i)
    v[i] = i < halfway ? start + step * static_cast<float>(i)
                       : end - step * static_cast<float>(steps - i - 1);
  return v;
}

Tensor sample_flow(Backend& be, const VelocityFn& v, const Tensor& noise, const SamplerOptions& opt,
                   const ProgressFn& progress, SolverStats* stats) {
  if (!v) fail("no velocity function");
  if (!noise.valid()) fail("invalid noise tensor");
  Counter f{v};
  if (stats) *stats = SolverStats{};
  Tensor out;
  if (opt.solver == "dopri5") out = dopri5(be, f, noise, opt, progress, stats);
  else if (opt.solver == "euler") out = fixed_grid(be, f, noise, opt, false, progress);
  else if (opt.solver == "heun" || opt.solver == "heun2") out = fixed_grid(be, f, noise, opt, true, progress);
  else fail("unknown solver '" + opt.solver + "' (dopri5 | euler | heun)");
  if (stats) stats->nfe = f.nfe;
  return out;
}

HostArray make_noise(const Shape& shape, uint64_t seed) {
  uint64_t sm = seed;
  auto splitmix = [&]() {
    uint64_t z = (sm += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  };
  uint64_t s[4] = {splitmix(), splitmix(), splitmix(), splitmix()};
  auto rotl = [](uint64_t x, int k) { return (x << k) | (x >> (64 - k)); };
  auto next = [&]() {
    uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return r;
  };
  auto uniform = [&]() { return (static_cast<double>(next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); };
  const double two_pi = 6.283185307179586476925286766559;
  HostArray a(DType::F32, shape);
  float* p = a.data<float>();
  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; i += 2) {
    double u1 = uniform(), u2 = uniform();
    double r = std::sqrt(-2.0 * std::log(u1));
    p[i] = static_cast<float>(r * std::cos(two_pi * u2));
    if (i + 1 < n) p[i + 1] = static_cast<float>(r * std::sin(two_pi * u2));
  }
  return a;
}

}  // namespace um::pipeline

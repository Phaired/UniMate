// Flow-matching ODE sampling (upstream unimate/models/flow/transport.py
// Sampler.sample_ode + integrators.ode -> torchdiffeq.odeint).
//
// Upstream defaults (Sampler.sample_ode): sampling_method='dopri5',
// num_steps=50 (the output grid t = linspace(0, 1, 50); for an adaptive solver
// it only selects where the dense output is evaluated -- generate_samples
// keeps the last point, t = 1), atol=1e-6, rtol=1e-3, interval [0, 1]
// (velocity model + linear path: Transport.check_interval gives t0=0, t1=1),
// noise at t=0, data at t=1. CFG is applied inside the velocity function
// (ClassifierFreeSampleModel: v_u + s (v_c - v_u), s = 3).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core/generation.h"
#include "core/tensor.h"
#include "core/umtf.h"

namespace um::pipeline {

// Velocity field of the (CFG-wrapped) denoiser: x is [B,J,D,F] normalised.
using VelocityFn = std::function<um::Tensor(const um::Tensor& x, float t)>;

struct SamplerOptions {
  std::string solver = "dopri5";  // "dopri5" | "euler" | "heun" (torchdiffeq heun2)
  // torchdiffeq grid size, upstream semantics: fixed-step solvers integrate on
  // linspace(t0, t1, num_steps), i.e. num_steps - 1 steps (num_steps - 1 NFE
  // for euler, 2 (num_steps - 1) for heun); ignored by dopri5's stepping.
  int num_steps = 50;
  double atol = 1e-6;
  double rtol = 1e-3;
  double t0 = 0.0, t1 = 1.0;
  // dopri5 controller (torchdiffeq RKAdaptiveStepsizeODESolver defaults)
  double safety = 0.9, ifactor = 10.0, dfactor = 0.2;
  double first_step = 0.0;   // 0: _select_initial_step
  int max_num_steps = 100000;
};

struct SolverStep {
  double t0 = 0, dt = 0;  // step start / size (float64, like torchdiffeq's time)
  bool accepted = false;
  double error_ratio = 0;
};

struct SolverStats {
  int nfe = 0;
  int accepted = 0, rejected = 0;
  double first_step = 0;
  std::vector<SolverStep> steps;
};

// Integrates dx/dt = v(x, t) from t0 to t1 starting at `noise` and returns
// x(t1). dopri5 replicates torchdiffeq 0.2.5: Dormand-Prince tableau with
// FSAL, stage times computed in float32 (stages at alpha == 1 use
// nextafter(t1, -inf), torchdiffeq's Perturb.PREV), initial step selection
// (Hairer II.4), RMS error norm over error / (atol + rtol max(|y0|, |y1|)),
// step factor min(10, max(0.9 err^-1/5, err < 1 ? 1 : 0.2)), and the final
// value by 4th-order dense interpolation at t1. State math runs on the
// backend; the only readbacks are the operands of the RMS norms (error,
// y0, y1 per step; y0/f0/f1 for the initial step), reduced in float64 on the
// host like torchdiffeq. Stage sums use ATen's 4-accumulator order. Accepted
// steps / NFE match torchdiffeq; step sizes agree to ~1e-5 relative on the
// real model (the error estimate is float32 cancellation noise, so smooth
// fields can differ up to ~1e-3 relative in dt without changing the pattern).
um::Tensor sample_flow(um::Backend& be, const VelocityFn& v, const um::Tensor& noise,
                       const SamplerOptions& opt, const um::ProgressFn& progress = {},
                       SolverStats* stats = nullptr);

// Standard-normal noise from the pipeline's own generator (xoshiro256**
// seeded through splitmix64, Box-Muller). Deterministic across platforms for
// a given seed; it does NOT reproduce torch.randn / torch.manual_seed.
um::HostArray make_noise(const um::Shape& shape, uint64_t seed);

// torch.linspace(start, end, steps) in float32, bit-exact (ATen's two-sided
// formula), used for the fixed-step grids.
std::vector<float> linspace_f32(float start, float end, int steps);

}  // namespace um::pipeline

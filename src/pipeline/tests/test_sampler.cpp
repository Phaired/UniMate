// sample_flow vs torchdiffeq: analytic vector fields (ode_analytic.umtf) and a
// replay of the real CFG denoiser trajectory (<asset>_sample.umtf: every
// velocity call recorded upstream is fed back by call index).
#include <cmath>
#include <cstring>
#include <map>

#include "pipeline/pipeline.h"
#include "test_common.h"

using namespace um;
using namespace um::pipeline;

static std::vector<double> steps_f64(const UmtfFile& f, const std::string& name) {
  const HostArray& a = f.get(name + "@f64");
  std::vector<double> v(a.bytes.size() / 8);
  std::memcpy(v.data(), a.bytes.data(), v.size() * 8);
  return v;  // rows of (t0, dt, accepted)
}

static void compare_steps(const char* tag, const SolverStats& st, const std::vector<double>& ref) {
  const size_t n = ref.size() / 3;
  PT_CHECK(st.steps.size() == n, "%s: %zu steps vs %zu", tag, st.steps.size(), n);
  double worst_t = 0, worst_dt = 0;
  int acc_mismatch = 0;
  for (size_t i = 0; i < std::min(n, st.steps.size()); ++i) {
    worst_t = std::max(worst_t, std::abs(st.steps[i].t0 - ref[i * 3]));
    worst_dt = std::max(worst_dt, std::abs(st.steps[i].dt - ref[i * 3 + 1]) / std::abs(ref[i * 3 + 1]));
    acc_mismatch += (st.steps[i].accepted ? 1.0 : 0.0) != ref[i * 3 + 2];
  }
  std::printf("  %-22s steps %zu/%zu  accept-pattern mismatches %d  max|dt0| %.3g  max rel|ddt| %.3g\n",
              tag, st.steps.size(), n, acc_mismatch, worst_t, worst_dt);
  PT_CHECK(acc_mismatch == 0, "%s accept pattern differs", tag);
  // error estimates are float32 cancellation noise on smooth fields, so dt
  // matches to ~1e-3 relative at worst (identical accept/reject pattern is the requirement)
  PT_CHECK(worst_dt < 2e-3, "%s step sizes differ: %g", tag, worst_dt);
}

static void analytic(Backend& be) {
  const std::string path = pt::golden("ode_analytic.umtf");
  if (!pt::exists(path)) {
    std::printf("[analytic] skipped: no %s\n", path.c_str());
    return;
  }
  UmtfFile g = UmtfFile::load(path);
  Tensor x0 = g.get("x0").to_tensor(be), mu = g.get("mu").to_tensor(be),
         a = g.get("a").to_tensor(be), b = g.get("b").to_tensor(be);
  auto sinT = [&](const Tensor& t) { return be.unary(UnaryOp::Sin, t); };
  std::map<std::string, VelocityFn> fields;
  fields["rf"] = [&](const Tensor& x, float t) {
    return be.binary(BinaryOp::Div, mu - x, be.scalar(1.1f - t));
  };
  fields["osc"] = [&](const Tensor& x, float t) {
    return (a * std::cos(4.0f * t)) * x + b * sinT(x);
  };
  fields["tanh"] = [&](const Tensor& x, float t) {
    return ((be.unary(UnaryOp::Tanh, x) * (1.0f - t)) * 3.0f - x * 0.5f) + mu * t;
  };
  fields["stiff"] = [&](const Tensor& x, float t) {
    return ((a * (8.0f * std::cos(11.0f * t))) - (x - mu * std::sin(7.0f * t)) * 12.0f) +
           sinT(x * 3.0f) * 4.0f;
  };
  std::printf("[analytic] field  solver   nfe(c++/torch)  max|x_c++ - x_torch|  (max|x|)\n");
  for (auto& [name, fn] : fields) {
    const auto& fm = g.meta()["fields"][name];
    SamplerOptions o;
    SolverStats st;
    Tensor out = sample_flow(be, fn, x0, o, {}, &st);
    auto ref = g.get(name + "/dopri5").to_f32();
    double d = pt::max_abs_diff(out.to_vector(), ref);
    std::printf("[analytic] %-6s dopri5   %3d/%-3d  acc %d/%d rej %d/%d  %.3g (%.3g)  h0 %.9g/%.9g\n",
                name.c_str(), st.nfe, fm["dopri5_nfe"].get<int>(), st.accepted,
                fm["dopri5_accepted"].get<int>(), st.rejected, fm["dopri5_rejected"].get<int>(), d,
                pt::max_abs(ref), st.first_step, fm["dopri5_first_step"].get<double>());
    PT_CHECK(st.nfe == fm["dopri5_nfe"].get<int>(), "%s dopri5 nfe", name.c_str());
    PT_CHECK(std::abs(st.first_step - fm["dopri5_first_step"].get<double>()) <
                 1e-5 * fm["dopri5_first_step"].get<double>(), "%s first step", name.c_str());
    compare_steps(name.c_str(), st, steps_f64(g, name + "/dopri5_steps"));
    PT_CHECK(d < 1e-4, "%s dopri5 final diff %g", name.c_str(), d);
    for (const char* solver : {"euler", "heun"}) {
      SamplerOptions fo;
      fo.solver = solver;
      fo.num_steps = 20;
      SolverStats fs;
      Tensor fout = sample_flow(be, fn, x0, fo, {}, &fs);
      std::string key = name + (std::string(solver) == "euler" ? "/euler20" : "/heun20");
      auto fref = g.get(key).to_f32();
      double fd = pt::max_abs_diff(fout.to_vector(), fref);
      int nref = fm[std::string(solver) == "euler" ? "euler20_nfe" : "heun20_nfe"].get<int>();
      std::printf("[analytic] %-6s %-6s   %3d/%-3d  %.3g (%.3g)\n", name.c_str(), solver, fs.nfe, nref, fd,
                  pt::max_abs(fref));
      PT_CHECK(fs.nfe == nref, "%s %s nfe", name.c_str(), solver);
      PT_CHECK(fd < 1e-5, "%s %s final diff %g", name.c_str(), solver, fd);
    }
  }
}

// Replays the recorded upstream velocity calls by index.
static void replay(Backend& be, const std::string& asset) {
  const std::string path = pt::golden(asset + "_sample.umtf");
  if (!pt::exists(path)) {
    std::printf("[replay %s] skipped\n", asset.c_str());
    return;
  }
  UmtfFile g = UmtfFile::load(path);
  Tensor noise = g.get("noise").to_tensor(be);
  for (const std::string key : {"euler20", "dopri5"}) {
    if (!g.has(key)) continue;
    const HostArray& V = g.get(key + "_call_v");
    std::vector<float> ts = g.get(key + "_call_t").to_f32();
    const int64_t per = V.numel() / V.dim(0);
    const float* vp = V.data<float>();
    int idx = 0;
    double worst_t = 0;
    VelocityFn fn = [&](const Tensor& x, float t) {
      if (idx >= V.dim(0)) {
        ++idx;
        return be.zeros(x.shape());
      }
      worst_t = std::max(worst_t, std::abs(double(t) - double(ts[idx])));
      std::vector<float> v(vp + per * idx, vp + per * (idx + 1));
      ++idx;
      return be.from_vector(v, x.shape());
    };
    SamplerOptions o;
    o.solver = key == "dopri5" ? "dopri5" : "euler";
    o.num_steps = key == "dopri5" ? 50 : 20;
    SolverStats st;
    Tensor out = sample_flow(be, fn, noise, o, {}, &st);
    auto ref = g.get(key).to_f32();
    double d = pt::max_abs_diff(out.to_vector(), ref);
    int nref = g.meta()[key + "_nfe"].get<int>();
    std::printf("[replay %s] %-7s nfe %d/%d  max|t - t_torch| %.3g  max|x - x_torch| %.3g (max|x| %.3g)\n",
                asset.c_str(), key.c_str(), st.nfe, nref, worst_t, d, pt::max_abs(ref));
    PT_CHECK(st.nfe == nref, "%s %s nfe", asset.c_str(), key.c_str());
    PT_CHECK(worst_t < (key == "euler20" ? 0.0 : 1e-5) + 1e-12, "%s %s call times differ (%g)", asset.c_str(), key.c_str(), worst_t);
    PT_CHECK(d < 1e-4, "%s %s final diff %g", asset.c_str(), key.c_str(), d);
    if (key == "dopri5") {
      PT_CHECK(st.accepted == g.meta()["dopri5_accepted"].get<int>(), "accepted");
      PT_CHECK(st.rejected == g.meta()["dopri5_rejected"].get<int>(), "rejected");
      compare_steps(("replay " + asset).c_str(), st, steps_f64(g, "dopri5_steps"));
    }
  }
}

static void noise_and_grid() {
  HostArray a = make_noise({1, 71, 12, 60}, 42), b = make_noise({1, 71, 12, 60}, 42),
            c = make_noise({1, 71, 12, 60}, 43);
  PT_CHECK(a.bytes == b.bytes, "noise not deterministic");
  PT_CHECK(a.bytes != c.bytes, "seeds collide");
  HostArray big = make_noise({1 << 20}, 7);
  double m = 0, v = 0;
  for (float x : big.to_f32()) m += x, v += double(x) * x;
  m /= big.numel();
  v = v / big.numel() - m * m;
  std::printf("[noise] mean %.4f var %.4f over 2^20 samples\n", m, v);
  PT_CHECK(std::abs(m) < 5e-3 && std::abs(v - 1) < 5e-3, "noise moments");
  auto g = linspace_f32(0.f, 1.f, 20);
  PT_CHECK(g.front() == 0.f && g.back() == 1.f && g.size() == 20, "linspace");
}

int main() {
  auto be = make_cpu_backend();
  noise_and_grid();
  analytic(*be);
  replay(*be, "eagle");
  replay(*be, "shark");
  return pt::finish("test_sampler");
}

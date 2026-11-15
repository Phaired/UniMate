// Shared helpers for um_vk compute shaders.
//
// GPU builtins (exp/log/sin/tanh) are only specified to a few ULP *of the
// exp2/log2 path* and drift badly near zeros or for large arguments, so the
// transcendental functions the reference backend gets from libm are
// re-implemented here with Cody-Waite range reduction + polynomials; they are
// within ~1-2 ULP of a correctly rounded float result over the useful range.

#extension GL_EXT_control_flow_attributes : enable

#define UM_INF uintBitsToFloat(0x7f800000u)
#define UM_NAN uintBitsToFloat(0x7fc00000u)

// Linear invocation index for 1-D work split over a 2-D grid (> 65535 groups).
uint um_linear_group() { return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x; }

float um_ldexp(float m, int e) {
  // Two steps so that e in [-276, 254] neither overflows the scale factor nor
  // flushes it to zero.
  int e1 = e / 2;
  return ldexp(ldexp(m, e1), e - e1);
}

float um_exp(float x) {
  if (isnan(x)) return x;
  if (x > 88.72284f) return UM_INF;
  if (x < -103.972084f) return 0.0;
  float k = roundEven(x * 1.44269504088896341f);
  precise float r = x - k * 0.693359375f;   // ln2 hi (exact product for |k| < 2^15)
  r = r - k * -2.12194440e-4f;               // ln2 lo
  // exp(r), |r| <= 0.3466: Taylor to degree 8 (truncation < 1e-9 relative).
  float p = 2.48015873e-5f;
  p = p * r + 1.98412698e-4f;
  p = p * r + 1.38888889e-3f;
  p = p * r + 8.33333333e-3f;
  p = p * r + 4.16666667e-2f;
  p = p * r + 1.66666667e-1f;
  p = p * r + 0.5f;
  p = p * r * r + r;
  precise float e = 1.0f + p;
  return um_ldexp(e, int(k));
}

float um_log(float x) {
  if (isnan(x) || x < 0.0) return UM_NAN;
  if (x == 0.0) return -UM_INF;
  if (isinf(x)) return UM_INF;
  int e;
  float m = frexp(x, e);  // m in [0.5, 1)
  if (m < 0.70710678f) {
    m *= 2.0;
    e -= 1;
  }
  precise float f = m - 1.0;
  float s = f / (2.0 + f);
  float z = s * s;
  // 2*atanh(s) = 2s(1 + z/3 + z^2/5 + ...), |s| <= 0.1716
  float poly = z * (0.33333333f + z * (0.2f + z * (0.14285714f + z * (0.11111111f + z * 0.09090909f))));
  // log(m) = 2s + 2s*poly
  precise float lm2 = 2.0 * s + 2.0 * s * poly;
  float fe = float(e);
  precise float res = fe * 0.693359375f + (lm2 + fe * -2.12194440e-4f);
  return res;
}

// Cody-Waite reduction by pi/2: x = k*pi/2 + r, |r| <= pi/4.
float um_reduce_pio2(float x, out int q) {
  float k = roundEven(x * 0.636619772367581343f);
  precise float r = fma(k, -1.5703125f, x);
  r = fma(k, -4.837512969970703125e-4f, r);
  r = fma(k, -7.54978995489188216e-8f, r);
  q = int(k) & 3;
  return r;
}
float um_sin_poly(float r) {
  float z = r * r;
  float p = (-1.9515295891e-4f * z + 8.3321608736e-3f) * z - 1.6666654611e-1f;
  return r + r * z * p;
}
float um_cos_poly(float r) {
  float z = r * r;
  float p = (2.443315711809948e-5f * z - 1.388731625493765e-3f) * z + 4.166664568298827e-2f;
  return 1.0 - 0.5 * z + z * z * p;
}
float um_sin(float x) {
  if (isinf(x) || isnan(x)) return UM_NAN;
  int q;
  float r = um_reduce_pio2(x, q);
  float v = (q & 1) == 0 ? um_sin_poly(r) : um_cos_poly(r);
  return (q & 2) == 0 ? v : -v;
}
float um_cos(float x) {
  if (isinf(x) || isnan(x)) return UM_NAN;
  int q;
  float r = um_reduce_pio2(x, q);
  float v = (q & 1) == 0 ? um_cos_poly(r) : um_sin_poly(r);
  return (q == 1 || q == 2) ? -v : v;
}

float um_tanh(float x) {
  float a = abs(x);
  float r;
  if (a < 0.625f) {
    float z = x * x;
    float p = ((((-5.70498872745e-3f * z + 2.06390887954e-2f) * z - 5.37397155531e-2f) * z +
                1.33314422036e-1f) * z - 3.33332819422e-1f);
    return x + x * z * p;
  } else if (a > 9.01f) {
    r = 1.0;
  } else {
    float e = um_exp(2.0 * a);
    r = 1.0 - 2.0 / (e + 1.0);
  }
  return x < 0.0 ? -r : r;
}

float um_sigmoid(float x) { return 1.0 / (1.0 + um_exp(-x)); }

// erfc with < 1.2e-7 fractional error (Numerical Recipes erfcc, Chebyshev fit).
float um_erfc(float x) {
  float z = abs(x);
  float t = 1.0 / (1.0 + 0.5 * z);
  float p = -0.82215223f + t * 0.17087277f;
  p = 1.48851587f + t * p;
  p = -1.13520398f + t * p;
  p = 0.27886807f + t * p;
  p = -0.18628806f + t * p;
  p = 0.09678418f + t * p;
  p = 0.37409196f + t * p;
  p = 1.00002368f + t * p;
  float ans = t * um_exp(-z * z - 1.26551223f + t * p);
  return x >= 0.0 ? ans : 2.0 - ans;
}

// Same rounding structure as the reference 0.5*x*(1 + erf(x/sqrt2)).
float um_erf(float z) { return z >= 0.0 ? 1.0 - um_erfc(z) : um_erfc(-z) - 1.0; }
float um_gelu(float x) { return 0.5 * x * (1.0 + um_erf(x * 0.70710678118654752f)); }

float um_gelu_tanh(float x) {
  precise float inner = 0.79788456080286536f * (x + 0.044715f * x * x * x);
  return 0.5 * x * (1.0 + um_tanh(inner));
}

float um_pow(float a, float b) {
  if (b == 0.0) return 1.0;
  if (isnan(a) || isnan(b)) return UM_NAN;
  if (a == 1.0) return 1.0;
  if (b == 1.0) return a;
  if (b == 2.0) return a * a;
  if (b == 0.5 && a >= 0.0) return sqrt(a);
  bool bint = (b == floor(b)) && abs(b) < 16777216.0;
  bool odd = bint && (mod(b, 2.0) == 1.0);
  if (a == 0.0) {
    if (b > 0.0) return odd ? a : 0.0;
    return odd ? (1.0 / a) : UM_INF;
  }
  if (a < 0.0 && !bint) return UM_NAN;
  float r = um_exp(b * um_log(abs(a)));
  return (a < 0.0 && odd) ? -r : r;
}

// UnaryOp codes (match um::UnaryOp order) + 100 = affine.
float um_unary(uint op, float v, float scale, float shift) {
  switch (op) {
    case 0u: return -v;
    case 1u: return abs(v);
    case 2u: return v * v;
    case 3u: return sqrt(v);
    case 4u: return 1.0 / sqrt(v);
    case 5u: return 1.0 / v;
    case 6u: return um_exp(v);
    case 7u: return um_log(v);
    case 8u: return um_sin(v);
    case 9u: return um_cos(v);
    case 10u: return um_tanh(v);
    case 11u: return um_sigmoid(v);
    case 12u: return v > 0.0 ? v : 0.0;
    case 13u: return v / (1.0 + um_exp(-v));
    case 14u: return um_gelu(v);
    case 15u:
    case 16u: return um_gelu_tanh(v);
    default: {
      precise float r = v * scale + shift;
      return r;
    }
  }
}

// BinaryOp codes (match um::BinaryOp order).
float um_binary(uint op, float a, float b) {
  switch (op) {
    case 0u: return a + b;
    case 1u: return a - b;
    case 2u: return a * b;
    case 3u: return a / b;
    case 4u: return (isnan(a) || isnan(b)) ? a + b : (a > b ? a : b);
    case 5u: return (isnan(a) || isnan(b)) ? a + b : (a < b ? a : b);
    default: return um_pow(a, b);
  }
}

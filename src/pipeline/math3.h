// Small double-precision 3D math used by the pipeline (internal).
// Quaternions are stored w,x,y,z (upstream / Motion-library order) unless a
// function name says xyzw (glTF order). Matrices are row-major 3x3 / 4x4
// acting on column vectors (p' = M p), like numpy / Blender.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace um::pipeline::m3 {

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;   // w x y z
using Mat3 = std::array<double, 9>;   // row-major
using Mat4 = std::array<double, 16>;  // row-major

inline Mat3 mat3_identity() { return {1, 0, 0, 0, 1, 0, 0, 0, 1}; }
inline Mat4 mat4_identity() { return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; }

inline Vec3 add(const Vec3& a, const Vec3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline Vec3 sub(const Vec3& a, const Vec3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline Vec3 scale(const Vec3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }

inline Mat3 mul(const Mat3& a, const Mat3& b) {
  Mat3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r[i * 3 + j] = a[i * 3 + 0] * b[0 * 3 + j] + a[i * 3 + 1] * b[1 * 3 + j] +
                     a[i * 3 + 2] * b[2 * 3 + j];
  return r;
}
inline Vec3 mul(const Mat3& a, const Vec3& v) {
  return {a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
          a[6] * v[0] + a[7] * v[1] + a[8] * v[2]};
}
inline Mat3 transpose(const Mat3& a) {
  return {a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]};
}
inline double det(const Mat3& a) {
  return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
         a[2] * (a[3] * a[7] - a[4] * a[6]);
}

inline Mat4 mul(const Mat4& a, const Mat4& b) {
  Mat4 r{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      double s = 0;
      for (int k = 0; k < 4; ++k) s += a[i * 4 + k] * b[k * 4 + j];
      r[i * 4 + j] = s;
    }
  return r;
}
inline Mat4 compose(const Mat3& r, const Vec3& t) {
  return {r[0], r[1], r[2], t[0], r[3], r[4], r[5], t[1], r[6], r[7], r[8], t[2], 0, 0, 0, 1};
}
inline Mat3 linear(const Mat4& m) { return {m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]}; }
inline Vec3 translation(const Mat4& m) { return {m[3], m[7], m[11]}; }

// General 4x4 inverse (cofactor expansion).
inline Mat4 inverse(const Mat4& m) {
  Mat4 inv{};
  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
  double d = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  double id = d != 0 ? 1.0 / d : 0.0;
  for (double& v : inv) v *= id;
  return inv;
}

// ---- quaternions (w x y z), Hamilton product, q*v = q v q^-1 ---------------
inline Quat qmul(const Quat& q, const Quat& r) {
  return {r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
          r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2],
          r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1],
          r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]};
}
inline Quat qconj(const Quat& q) { return {q[0], -q[1], -q[2], -q[3]}; }
inline Vec3 qrot(const Quat& q, const Vec3& v) {
  Quat p = qmul(qmul(q, Quat{0, v[0], v[1], v[2]}), qconj(q));
  return {p[1], p[2], p[3]};
}
// Holden Quaternions.transforms(): standard rotation matrix (no normalisation).
inline Mat3 quat_to_mat(const Quat& q) {
  double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
  double x2 = qx + qx, y2 = qy + qy, z2 = qz + qz;
  double xx = qx * x2, yy = qy * y2, wx = qw * x2, xy = qx * y2, yz = qy * z2, wy = qw * y2,
         xz = qx * z2, zz = qz * z2, wz = qw * z2;
  return {1.0 - (yy + zz), xy - wz, xz + wy, xy + wz, 1.0 - (xx + zz), yz - wx,
          xz - wy, yz + wx, 1.0 - (xx + yy)};
}
// upstream rotation_conversions.matrix_to_quaternion_np: Shepperd, one branch
// chosen by the largest candidate, normalised, w >= 0.
inline Quat mat_to_quat(const Mat3& m) {
  double d0 = m[0], d1 = m[4], d2 = m[8];
  double cand[4] = {1 + d0 + d1 + d2, 1 + d0 - d1 - d2, 1 - d0 + d1 - d2, 1 - d0 - d1 + d2};
  int b = 0;
  for (int i = 1; i < 4; ++i)
    if (cand[i] > cand[b]) b = i;
  double s = 2.0 * std::sqrt(std::max(cand[b], 1e-12));
  double m21_m12 = m[7] - m[5], m02_m20 = m[2] - m[6], m10_m01 = m[3] - m[1];
  double m21p12 = m[7] + m[5], m02p20 = m[2] + m[6], m10p01 = m[3] + m[1];
  Quat q;
  if (b == 0) q = {s / 4, m21_m12 / s, m02_m20 / s, m10_m01 / s};
  else if (b == 1) q = {m21_m12 / s, s / 4, m10p01 / s, m02p20 / s};
  else if (b == 2) q = {m02_m20 / s, m10p01 / s, s / 4, m21p12 / s};
  else q = {m10_m01 / s, m02p20 / s, m21p12 / s, s / 4};
  double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  for (double& v : q) v /= n;
  if (q[0] < 0)
    for (double& v : q) v = -v;
  return q;
}

// upstream rotation_6d_to_matrix_np: x = c6[0:3], y = c6[3:6] are the first two
// COLUMNS after Gram-Schmidt; a degenerate vector decodes to the identity.
inline Mat3 rot6d_to_mat(const double* c6) {
  const double AXIS_TOL = 1e-6, PARALLEL_TOL = 1e-6;
  Vec3 xr{c6[0], c6[1], c6[2]}, yr{c6[3], c6[4], c6[5]};
  double nx = norm(xr), ny = norm(yr);
  auto gs = [](const Vec3& xa, const Vec3& ya, bool& finite) {
    Vec3 x = scale(xa, 1.0 / norm(xa));
    Vec3 z = cross(x, ya);
    z = scale(z, 1.0 / norm(z));
    Vec3 y = cross(z, x);
    Mat3 m{x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]};
    finite = true;
    for (double v : m) finite = finite && std::isfinite(v);
    return m;
  };
  bool finite = false;
  Mat3 m = gs(xr, yr, finite);
  bool ok = finite;
  for (int i = 0; i < 6; ++i) ok = ok && std::isfinite(c6[i]);
  if (ok) {
    double sine = norm(cross(scale(xr, 1.0 / nx), scale(yr, 1.0 / ny)));
    ok = nx > AXIS_TOL && ny > AXIS_TOL && sine > PARALLEL_TOL;
  }
  if (ok) return m;
  return mat3_identity();
}

// Symmetric 4x4 Jacobi eigen-solver; returns eigenvector of the largest eigenvalue.
inline std::array<double, 4> sym4_top_eigenvector(std::array<double, 16> a) {
  std::array<double, 16> v{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  for (int sweep = 0; sweep < 64; ++sweep) {
    double off = 0;
    for (int p = 0; p < 4; ++p)
      for (int q = p + 1; q < 4; ++q) off += a[p * 4 + q] * a[p * 4 + q];
    if (off < 1e-30) break;
    for (int p = 0; p < 4; ++p)
      for (int q = p + 1; q < 4; ++q) {
        double apq = a[p * 4 + q];
        if (std::abs(apq) < 1e-300) continue;
        double theta = (a[q * 4 + q] - a[p * 4 + p]) / (2 * apq);
        double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1));
        double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < 4; ++k) {  // A = J^T A J
          double akp = a[k * 4 + p], akq = a[k * 4 + q];
          a[k * 4 + p] = c * akp - s * akq;
          a[k * 4 + q] = s * akp + c * akq;
        }
        for (int k = 0; k < 4; ++k) {
          double apk = a[p * 4 + k], aqk = a[q * 4 + k];
          a[p * 4 + k] = c * apk - s * aqk;
          a[q * 4 + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 4; ++k) {
          double vkp = v[k * 4 + p], vkq = v[k * 4 + q];
          v[k * 4 + p] = c * vkp - s * vkq;
          v[k * 4 + q] = s * vkp + c * vkq;
        }
      }
  }
  int best = 0;
  for (int i = 1; i < 4; ++i)
    if (a[i * 4 + i] > a[best * 4 + best]) best = i;
  return {v[0 * 4 + best], v[1 * 4 + best], v[2 * 4 + best], v[3 * 4 + best]};
}

// The rotation R maximising trace(R^T M) (= U diag(1,1,det(UV^T)) V^T of the
// SVD M = U S V^T, i.e. upstream's det-corrected projection), via Horn's
// quaternion method.
inline Mat3 nearest_rotation(const Mat3& M) {
  // K such that q^T K q = trace(R(q)^T M)
  double Sxx = M[0], Sxy = M[1], Sxz = M[2], Syx = M[3], Syy = M[4], Syz = M[5], Szx = M[6],
         Szy = M[7], Szz = M[8];
  std::array<double, 16> K{Sxx + Syy + Szz, Szy - Syz, Sxz - Szx, Syx - Sxy,
                           Szy - Syz, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz,
                           Sxz - Szx, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy,
                           Syx - Sxy, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz};
  auto q = sym4_top_eigenvector(K);
  double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  Quat qq{q[0] / n, q[1] / n, q[2] / n, q[3] / n};
  return quat_to_mat(qq);
}

}  // namespace um::pipeline::m3

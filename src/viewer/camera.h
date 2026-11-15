// Orbit camera (Y up, right-handed, Vulkan clip space).
#pragma once

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace um::viewer {

struct OrbitCamera {
  glm::vec3 target{0.f};
  float distance = 3.f;
  float yaw = glm::radians(30.f);    // 0 = looking from +Z towards -Z
  float pitch = glm::radians(20.f);  // > 0 = above the target
  float fov_y = glm::radians(40.f);
  float radius = 1.f;                // scene radius, for clip planes

  glm::vec3 eye() const {
    return target + distance * glm::vec3(std::cos(pitch) * std::sin(yaw), std::sin(pitch),
                                         std::cos(pitch) * std::cos(yaw));
  }
  glm::mat4 view() const { return glm::lookAt(eye(), target, glm::vec3(0, 1, 0)); }
  glm::mat4 proj(float aspect) const {
    float n = std::max(distance - 4.f * radius, distance * 0.01f);
    n = std::max(n, 1e-4f);
    float f = distance + 50.f * radius;
    glm::mat4 p = glm::perspective(fov_y, aspect, n, f);
    p[1][1] *= -1.f;  // Vulkan: +Y down in clip space
    return p;
  }

  void frame(const glm::vec3& lo, const glm::vec3& hi, float aspect = 16.f / 9.f) {
    target = 0.5f * (lo + hi);
    radius = std::max(0.5f * glm::length(hi - lo), 1e-3f);
    // Smallest distance at which all 8 box corners project inside the view
    // (with a margin) for the current yaw/pitch; the box centre stays at the
    // view centre.
    const float margin = 0.92f;
    const float ty = std::tan(0.5f * fov_y) * margin;
    const float tx = std::tan(0.5f * fov_y) * std::max(aspect, 1e-3f) * margin;
    const glm::vec3 back = glm::vec3(std::cos(pitch) * std::sin(yaw), std::sin(pitch), std::cos(pitch) * std::cos(yaw));
    const glm::vec3 fwd = -back;
    glm::vec3 right = glm::cross(fwd, glm::vec3(0, 1, 0));
    right = glm::length(right) > 1e-6f ? glm::normalize(right) : glm::vec3(1, 0, 0);
    const glm::vec3 up = glm::cross(right, fwd);
    float d = 1e-3f;
    for (int i = 0; i < 8; ++i) {
      const glm::vec3 c = glm::vec3(i & 1 ? hi.x : lo.x, i & 2 ? hi.y : lo.y, i & 4 ? hi.z : lo.z) - target;
      // corner depth in front of the eye is d + dot(c, fwd); require |x| <= tx * depth, |y| <= ty * depth
      d = std::max(d, std::abs(glm::dot(c, right)) / tx - glm::dot(c, fwd));
      d = std::max(d, std::abs(glm::dot(c, up)) / ty - glm::dot(c, fwd));
    }
    distance = std::max(d, radius * 0.05f);
  }

  void orbit(float dx, float dy) {  // pixels
    yaw -= dx * 0.008f;
    pitch = std::clamp(pitch + dy * 0.008f, glm::radians(-89.f), glm::radians(89.f));
  }
  void pan(float dx, float dy, float viewport_h) {
    glm::mat4 v = view();
    glm::vec3 right(v[0][0], v[1][0], v[2][0]);
    glm::vec3 up(v[0][1], v[1][1], v[2][1]);
    float k = 2.f * distance * std::tan(0.5f * fov_y) / std::max(viewport_h, 1.f);
    target += (-dx * right + dy * up) * k;
  }
  void zoom(float wheel) { distance = std::clamp(distance * std::pow(0.88f, wheel), radius * 0.05f, radius * 200.f); }
};

}  // namespace um::viewer

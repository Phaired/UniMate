// Windowed viewer: GLFW window + swapchain, Dear ImGui panel, playback,
// generation on a worker thread.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

// Vulkan (volk) must come before GLFW so glfwCreateWindowSurface is declared.
#include "camera.h"
#include "renderer.h"
#include "viewer.h"

#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <imgui_stdlib.h>
#include <stb_image_write.h>

namespace um::viewer {
namespace {

constexpr uint32_t kFramesInFlight = 2;

struct ClipEntry {
  um::MotionClip clip;
  bool embedded = false;
  std::string label;
  std::vector<int> binding;
  int matched = 0;
};

// Shared with the worker thread. The worker is always joined by the UI thread
// (poll_job, stop_job or shutdown), never detached.
struct GenJob {
  std::atomic<bool> running{false};
  std::atomic<bool> done{false};
  std::atomic<bool> cancel{false};  // set by the UI; the ProgressFn then throws GenerationCancelled
  std::mutex m;
  float progress = 0.f;
  std::string status;
  bool ok = false, cancelled = false;
  um::MotionClip result;
  std::string error;
  um::GenerateRequest request;
  double start_time = 0.0;
  std::thread thread;
};

float srgb_to_linear(float c) { return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f); }

class App {
 public:
  explicit App(const ViewerOptions& o) : opt_(o) {
    req_ = o.initial_request;
    if (req_.prompt.empty()) req_.prompt = "walks forward slowly";
    asset_path_ = o.asset_glb;
    show_skeleton_ = o.show_skeleton;
    show_model_skeleton_ = o.show_model_skeleton;
    std::memcpy(&model_xf_[0][0], o.model_skeleton_transform, sizeof(float) * 16);
    print_stats_ = o.print_stats || o.auto_exit_seconds > 0;
  }
  ~App() { stop_job(); }  // safety net: run() already joins on every normal path
  int run();

 private:
  bool init_window();
  bool create_swapchain(std::string& err);
  void destroy_swapchain(bool keep_handle);
  void load_asset_ui(const std::string& path);
  void rebind_clips();
  bool start_generate();
  void cancel_job();
  void stop_job();  // cancel + join (blocking)
  bool job_running() const { return job_ && job_->running; }
  void poll_job();
  glm::ivec4 view_rect() const;  // 3D area (x, y, w, h) in framebuffer pixels
  float view_aspect() const {
    glm::ivec4 r = view_rect();
    return r.w > 0 ? float(r.z) / float(r.w) : 16.f / 9.f;
  }
  void frame_camera();
  void build_ui(float dt);
  void handle_input();
  void update_pose();
  bool save_capture(VkBuffer, VmaAllocation, void* mapped);

  ViewerOptions opt_;
  GLFWwindow* window_ = nullptr;
  VkContext ctx_;
  VkSurfaceKHR surface_ = VK_NULL_HANDLE;
  Renderer renderer_;

  // swapchain
  VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
  VkFormat sc_format_ = VK_FORMAT_UNDEFINED;
  VkExtent2D sc_extent_{};
  std::vector<VkImage> sc_images_;
  std::vector<VkImageView> sc_views_;
  std::vector<VkSemaphore> render_done_;  // per swapchain image
  uint32_t min_image_count_ = 2;
  bool sc_transfer_src_ = false;
  bool sc_dirty_ = false;

  // per frame in flight
  VkCommandBuffer cmd_[kFramesInFlight]{};
  VkFence fence_[kFramesInFlight]{};
  VkSemaphore acquired_[kFramesInFlight]{};
  VkDescriptorPool imgui_pool_ = VK_NULL_HANDLE;

  // scene
  std::unique_ptr<Asset> asset_;
  std::string asset_path_, asset_error_;
  OrbitCamera cam_;
  Pose pose_;
  std::vector<glm::mat4> palette_;
  std::vector<LineVertex> bones_, joints_, model_bones_, model_joints_;
  std::vector<glm::vec3> model_pos_;
  bool show_skeleton_ = true, show_mesh_ = true, show_grid_ = true, show_model_skeleton_ = true;
  glm::mat4 model_xf_{1.f};
  bool cam_user_moved_ = false;  // re-frame on resize until the user moves the camera
  float panel_px_ = 0.f;         // docked panel width in framebuffer pixels (from the last UI frame)

  // clips & playback
  std::vector<ClipEntry> clips_;
  int current_ = -1;  // -1 = rest pose
  double play_time_ = 0.0;
  bool playing_ = true, loop_ = true;
  float speed_ = 1.f;

  // generation
  um::GenerateRequest req_;
  std::shared_ptr<GenJob> job_;
  std::string gen_error_, gen_info_;
  int gen_counter_ = 0;
  std::string export_path_ = "generated.glb";
  std::string export_msg_;

  double start_time_ = 0.0;
  // frame rate readout: frames counted over ~0.5 s windows (not an average of 1/dt,
  // which is dominated by the near-zero dt of frames that do not block in present)
  float fps_avg_ = 0.f, frame_ms_ = 0.f;
  double fps_window_start_ = 0.0;
  int fps_window_frames_ = 0;
  bool print_stats_ = false;
  uint64_t stat_frames_ = 0, stat_gen_frames_ = 0, stat_min_waits_ = 0;
  double stat_gen_time_ = 0.0;
};

bool App::init_window() {
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  window_ = glfwCreateWindow(opt_.width, opt_.height, opt_.title.c_str(), nullptr, nullptr);
  if (!window_) return false;
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) {
    static_cast<App*>(glfwGetWindowUserPointer(w))->sc_dirty_ = true;
  });
  return true;
}

bool App::create_swapchain(std::string& err) {
  VkSurfaceCapabilitiesKHR caps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx_.phys, surface_, &caps);
  int fw = 0, fh = 0;
  glfwGetFramebufferSize(window_, &fw, &fh);
  VkExtent2D ext = caps.currentExtent;
  if (ext.width == UINT32_MAX) {
    ext.width = std::clamp<uint32_t>(uint32_t(fw), caps.minImageExtent.width, caps.maxImageExtent.width);
    ext.height = std::clamp<uint32_t>(uint32_t(fh), caps.minImageExtent.height, caps.maxImageExtent.height);
  }
  if (ext.width == 0 || ext.height == 0) return true;  // minimised; retry later

  if (sc_format_ == VK_FORMAT_UNDEFINED) {
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.phys, surface_, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.phys, surface_, &n, fmts.data());
    for (VkFormat want : {VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB})
      for (auto& f : fmts)
        if (sc_format_ == VK_FORMAT_UNDEFINED && f.format == want && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
          sc_format_ = want;
    if (sc_format_ == VK_FORMAT_UNDEFINED) {
      err = "surface has no sRGB format";
      return false;
    }
  }
  uint32_t pn = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(ctx_.phys, surface_, &pn, nullptr);
  std::vector<VkPresentModeKHR> modes(pn);
  vkGetPhysicalDeviceSurfacePresentModesKHR(ctx_.phys, surface_, &pn, modes.data());
  VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
  if (!opt_.vsync) {
    for (VkPresentModeKHR m : {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR})
      if (mode == VK_PRESENT_MODE_FIFO_KHR && std::find(modes.begin(), modes.end(), m) != modes.end()) mode = m;
  }
  if (print_stats_) {
    std::string avail;
    for (VkPresentModeKHR m : modes) avail += " " + std::to_string(int(m));
    std::printf("[viewer] present mode %d (vsync=%d), available:%s\n", int(mode), int(opt_.vsync), avail.c_str());
  }
  uint32_t count = std::max(caps.minImageCount + 1, 3u);
  if (caps.maxImageCount) count = std::min(count, caps.maxImageCount);
  min_image_count_ = std::max(caps.minImageCount, 2u);
  sc_transfer_src_ = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;

  VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  ci.surface = surface_;
  ci.minImageCount = count;
  ci.imageFormat = sc_format_;
  ci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  ci.imageExtent = ext;
  ci.imageArrayLayers = 1;
  ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | (sc_transfer_src_ ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
  ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.preTransform = caps.currentTransform;
  ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  ci.presentMode = mode;
  ci.clipped = VK_TRUE;
  ci.oldSwapchain = swapchain_;
  VkSwapchainKHR sc;
  if (vkCreateSwapchainKHR(ctx_.device, &ci, nullptr, &sc) != VK_SUCCESS) {
    err = "vkCreateSwapchainKHR failed";
    return false;
  }
  destroy_swapchain(false);
  swapchain_ = sc;
  sc_extent_ = ext;
  uint32_t n = 0;
  vkGetSwapchainImagesKHR(ctx_.device, sc, &n, nullptr);
  sc_images_.resize(n);
  vkGetSwapchainImagesKHR(ctx_.device, sc, &n, sc_images_.data());
  for (VkImage img : sc_images_) {
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = sc_format_;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageView v;
    vkCreateImageView(ctx_.device, &vci, nullptr, &v);
    sc_views_.push_back(v);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore s;
    vkCreateSemaphore(ctx_.device, &sci, nullptr, &s);
    render_done_.push_back(s);
  }
  if (!renderer_.set_targets(ext, sc_views_, err)) return false;
  return true;
}

void App::destroy_swapchain(bool destroy_handle) {
  for (VkImageView v : sc_views_) vkDestroyImageView(ctx_.device, v, nullptr);
  for (VkSemaphore s : render_done_) vkDestroySemaphore(ctx_.device, s, nullptr);
  sc_views_.clear();
  render_done_.clear();
  sc_images_.clear();
  if (swapchain_) vkDestroySwapchainKHR(ctx_.device, swapchain_, nullptr);
  swapchain_ = VK_NULL_HANDLE;
  (void)destroy_handle;
}

void App::rebind_clips() {
  for (ClipEntry& c : clips_) {
    c.binding = asset_ ? bind_clip(*asset_, c.clip) : std::vector<int>(c.clip.tracks.size(), -1);
    c.matched = int(std::count_if(c.binding.begin(), c.binding.end(), [](int b) { return b >= 0; }));
  }
}

void App::load_asset_ui(const std::string& path) {
  // A running job animates the previous asset (and the generator may read the
  // asset path set by on_asset_loaded): cancel it before switching. The UI
  // disables "Load" while a job runs; this covers programmatic calls.
  if (job_running()) {
    stop_job();
    poll_job();
  }
  auto a = std::make_unique<Asset>();
  std::string err;
  if (!load_asset(path, *a, err)) {
    asset_error_ = err;
    return;
  }
  if (!renderer_.set_asset(*a, err)) {
    asset_error_ = err;
    renderer_.clear_asset();
    asset_.reset();
    return;
  }
  asset_error_.clear();
  asset_ = std::move(a);
  asset_path_ = path;
  // drop the previous asset's embedded clips, keep generated ones
  clips_.erase(std::remove_if(clips_.begin(), clips_.end(), [](const ClipEntry& c) { return c.embedded; }),
               clips_.end());
  std::vector<ClipEntry> emb;
  for (auto& c : asset_->embedded_clips) {
    ClipEntry e;
    e.clip = c;
    e.embedded = true;
    e.label = "[glb] " + c.name;
    emb.push_back(std::move(e));
  }
  clips_.insert(clips_.begin(), emb.begin(), emb.end());
  rebind_clips();
  current_ = -1;
  play_time_ = 0;
  rest_pose(*asset_, pose_);
  cam_user_moved_ = false;
  frame_camera();
  if (opt_.on_asset_loaded) opt_.on_asset_loaded(path);
}

glm::ivec4 App::view_rect() const {
  const int w = int(sc_extent_.width), h = int(sc_extent_.height);
  int px = std::clamp(int(std::lround(panel_px_)), 0, std::max(0, w - 64));
  return glm::ivec4(px, 0, std::max(1, w - px), std::max(1, h));
}

void App::frame_camera() {
  if (asset_) cam_.frame(asset_->bounds_min, asset_->bounds_max, view_aspect());
}

bool App::start_generate() {
  // one job at a time: a finished-but-unpolled job is collected first
  if (!opt_.generate || !asset_ || job_running()) return false;
  if (job_) poll_job();
  if (job_) return false;
  auto job = std::make_shared<GenJob>();
  job->running = true;
  job->request = req_;
  job->status = "starting";
  job->start_time = glfwGetTime();
  gen_error_.clear();
  gen_info_.clear();
  um::GenerateFn fn = opt_.generate;
  job->thread = std::thread([job, fn]() {
    um::ProgressFn progress = [job](float p, const std::string& s) {
      if (job->cancel.load()) throw GenerationCancelled();
      std::lock_guard<std::mutex> lk(job->m);
      job->progress = p;
      job->status = s;
    };
    try {
      um::MotionClip clip = fn(job->request, progress);
      std::lock_guard<std::mutex> lk(job->m);
      if (job->cancel.load()) {
        job->cancelled = true;  // finished anyway, but the user no longer wants it
      } else {
        job->result = std::move(clip);
        job->ok = true;
      }
    } catch (const GenerationCancelled&) {
      std::lock_guard<std::mutex> lk(job->m);
      job->cancelled = true;
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lk(job->m);
      if (job->cancel.load()) job->cancelled = true;  // generator rewrapped our exception
      else job->error = e.what();
    } catch (...) {
      std::lock_guard<std::mutex> lk(job->m);
      if (job->cancel.load()) job->cancelled = true;
      else job->error = "unknown exception";
    }
    job->running = false;
    job->done = true;
  });
  job_ = job;
  return true;
}

void App::cancel_job() {
  if (!job_running() || job_->cancel.load()) return;
  job_->cancel = true;
  std::lock_guard<std::mutex> lk(job_->m);
  job_->status = "cancelling...";
}

void App::stop_job() {
  if (!job_) return;
  if (job_->running) {
    cancel_job();
    if (print_stats_) std::printf("[viewer] waiting for the generation worker to stop\n");
  }
  if (job_->thread.joinable()) job_->thread.join();
}

void App::poll_job() {
  if (!job_ || !job_->done) return;
  if (job_->thread.joinable()) job_->thread.join();
  std::shared_ptr<GenJob> keep = job_;  // job_.reset() below must not destroy the locked mutex
  std::lock_guard<std::mutex> lk(keep->m);
  if (job_->cancelled) {
    gen_info_ = "cancelled";
    if (print_stats_) std::printf("[viewer] generation cancelled after %.2f s\n", glfwGetTime() - job_->start_time);
  } else if (job_->ok) {
    if (print_stats_) std::printf("[viewer] generation finished after %.2f s\n", glfwGetTime() - job_->start_time);
    ClipEntry e;
    e.clip = std::move(job_->result);
    if (e.clip.fps <= 0) e.clip.fps = 30.f;
    char buf[64];
    std::snprintf(buf, sizeof(buf), " (seed %u)", job_->request.seed);
    std::string p = job_->request.prompt.substr(0, 40);
    e.label = "#" + std::to_string(++gen_counter_) + " " + p + buf;
    if (e.clip.name.empty()) e.clip.name = job_->request.prompt;
    if (asset_) e.binding = bind_clip(*asset_, e.clip);
    e.matched = int(std::count_if(e.binding.begin(), e.binding.end(), [](int b) { return b >= 0; }));
    clips_.push_back(std::move(e));
    current_ = int(clips_.size()) - 1;
    play_time_ = 0;
    playing_ = true;
  } else {
    gen_error_ = job_->error.empty() ? "generation failed" : job_->error;
  }
  job_->done = false;
  job_.reset();
}

void App::handle_input() {
  ImGuiIO& io = ImGui::GetIO();
  if (!io.WantCaptureMouse) {
    ImVec2 d = io.MouseDelta;
    const bool moved = (d.x != 0.f || d.y != 0.f);
    if (io.MouseDown[0] && moved) {
      cam_.orbit(d.x, d.y);
      cam_user_moved_ = true;
    }
    if ((io.MouseDown[1] || io.MouseDown[2]) && moved) {
      cam_.pan(d.x * io.DisplayFramebufferScale.x, d.y * io.DisplayFramebufferScale.y, float(view_rect().w));
      cam_user_moved_ = true;
    }
    if (io.MouseWheel != 0.f) {
      cam_.zoom(io.MouseWheel);
      cam_user_moved_ = true;
    }
  }
  if (!io.WantCaptureKeyboard) {
    if (ImGui::IsKeyPressed(ImGuiKey_Space)) playing_ = !playing_;
    if (ImGui::IsKeyPressed(ImGuiKey_F)) {
      frame_camera();
      cam_user_moved_ = false;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_S)) show_skeleton_ = !show_skeleton_;
    if (ImGui::IsKeyPressed(ImGuiKey_M)) show_model_skeleton_ = !show_model_skeleton_;
  }
}

void App::build_ui(float dt) {
  (void)dt;
  // Panel docked in a full-height strip on the left; horizontally resizable.
  // The 3D view uses the rest of the window (see view_rect()).
  const ImGuiIO& io = ImGui::GetIO();
  const float disp_w = io.DisplaySize.x, disp_h = io.DisplaySize.y;
  const float max_w = std::max(160.f, disp_w * 0.6f);
  ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(std::clamp(float(opt_.panel_width), 160.f, max_w), disp_h), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSizeConstraints(ImVec2(160.f, disp_h), ImVec2(max_w, disp_h));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
  ImGui::Begin("UniMate", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
  ImGui::PopStyleVar();
  panel_px_ = ImGui::GetWindowWidth() * io.DisplayFramebufferScale.x;

  const bool busy = job_running();
  // ---- asset ----
  if (ImGui::CollapsingHeader("Asset", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::SetNextItemWidth(-60);
    ImGui::InputText("##asset", &asset_path_);
    ImGui::SameLine();
    ImGui::BeginDisabled(busy);  // the running job animates the current asset
    if (ImGui::Button("Load")) load_asset_ui(asset_path_);
    ImGui::EndDisabled();
    if (busy) ImGui::TextDisabled("cancel the generation to load another asset");
    if (!asset_error_.empty()) ImGui::TextColored(ImVec4(1, 0.35f, 0.3f, 1), "%s", asset_error_.c_str());
    if (asset_)
      ImGui::TextDisabled("%zu nodes, %zu joints, %zu skins, %zu verts, %zu draws", asset_->nodes.size(),
                          asset_->skeleton.size(), asset_->skins.size(), asset_->vertices.size(), asset_->draws.size());
  }

  // ---- generation ----
  if (ImGui::CollapsingHeader("Generate", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::InputTextMultiline("##prompt", &req_.prompt, ImVec2(-1, ImGui::GetTextLineHeight() * 3.5f));
    ImGui::SetNextItemWidth(140);
    ImGui::InputScalar("seed", ImGuiDataType_U32, &req_.seed);
    ImGui::SameLine();
    if (ImGui::Button("random")) req_.seed = std::random_device{}();
    ImGui::SliderFloat("cfg scale", &req_.cfg_scale, 0.f, 10.f, "%.2f");
    ImGui::SliderInt("frames", &req_.num_frames, 1, 300);
    const char* solvers[] = {"dopri5", "euler", "heun"};
    int si = 0;
    for (int i = 0; i < 3; ++i)
      if (req_.solver == solvers[i]) si = i;
    if (ImGui::Combo("solver", &si, solvers, 3)) req_.solver = solvers[si];
    ImGui::BeginDisabled(req_.solver == "dopri5");
    ImGui::SliderInt("steps", &req_.num_steps, 1, 200);
    ImGui::EndDisabled();
    if (busy) {
      const bool cancelling = job_->cancel.load();
      ImGui::BeginDisabled(true);
      ImGui::Button("Generating...", ImVec2(ImGui::GetContentRegionAvail().x * 0.6f, 0));
      ImGui::EndDisabled();
      ImGui::SameLine();
      ImGui::BeginDisabled(cancelling);
      if (ImGui::Button(cancelling ? "Cancelling..." : "Cancel", ImVec2(-1, 0))) cancel_job();
      ImGui::EndDisabled();
    } else {
      ImGui::BeginDisabled(!opt_.generate || !asset_);
      if (ImGui::Button("Generate##run", ImVec2(-1, 0))) start_generate();
      ImGui::EndDisabled();
    }
    if (!opt_.generate) ImGui::TextDisabled("no generator attached");
    if (!gen_info_.empty() && !busy) ImGui::TextDisabled("last job: %s", gen_info_.c_str());
    if (job_) {
      float p;
      std::string s;
      {
        std::lock_guard<std::mutex> lk(job_->m);
        p = job_->progress;
        s = job_->status;
      }
      ImGui::ProgressBar(std::clamp(p, 0.f, 1.f), ImVec2(-1, 0), s.c_str());
    }
    if (!gen_error_.empty()) ImGui::TextColored(ImVec4(1, 0.35f, 0.3f, 1), "error: %s", gen_error_.c_str());
  }

  // ---- clips ----
  if (ImGui::CollapsingHeader("Clips", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (ImGui::BeginListBox("##clips", ImVec2(-1, ImGui::GetTextLineHeightWithSpacing() * 7))) {
      if (ImGui::Selectable("[rest pose]", current_ < 0)) current_ = -1;
      for (int i = 0; i < int(clips_.size()); ++i) {
        const ClipEntry& c = clips_[i];
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s  (%d fr, %d/%zu tracks)##%d", c.label.c_str(), c.clip.num_frames, c.matched,
                      c.clip.tracks.size(), i);
        if (ImGui::Selectable(buf, current_ == i)) {
          current_ = i;
          play_time_ = 0;
          playing_ = true;
        }
      }
      ImGui::EndListBox();
    }
    if (opt_.export_glb && current_ >= 0) {
      ImGui::SetNextItemWidth(-70);
      ImGui::InputText("##export", &export_path_);
      ImGui::SameLine();
      if (ImGui::Button("Export")) {
        try {
          opt_.export_glb(clips_[current_].clip, export_path_);
          export_msg_ = "exported " + export_path_;
        } catch (const std::exception& e) {
          export_msg_ = std::string("export failed: ") + e.what();
        }
      }
      if (!export_msg_.empty()) ImGui::TextDisabled("%s", export_msg_.c_str());
    }
  }

  // ---- playback ----
  if (ImGui::CollapsingHeader("Playback", ImGuiTreeNodeFlags_DefaultOpen)) {
    if (current_ >= 0) {
      const um::MotionClip& c = clips_[current_].clip;
      const float fps = c.fps > 0 ? c.fps : 30.f;
      const int last = std::max(c.num_frames - 1, 0);
      if (ImGui::Button("|<")) play_time_ = 0;
      ImGui::SameLine();
      if (ImGui::Button("<")) { playing_ = false; play_time_ = std::max(0.0, std::floor(play_time_ * fps - 0.5) / fps); }
      ImGui::SameLine();
      if (ImGui::Button(playing_ ? "Pause" : "Play", ImVec2(60, 0))) playing_ = !playing_;
      ImGui::SameLine();
      if (ImGui::Button(">")) { playing_ = false; play_time_ = std::min<double>(last, std::floor(play_time_ * fps + 1.5)) / fps; }
      ImGui::SameLine();
      if (ImGui::Button(">|")) { playing_ = false; play_time_ = last / fps; }
      ImGui::SameLine();
      ImGui::Checkbox("loop", &loop_);
      float frame = float(play_time_ * fps);
      if (loop_ && c.num_frames > 0) frame = std::fmod(frame, float(c.num_frames));
      ImGui::SetNextItemWidth(-1);
      if (ImGui::SliderFloat("##frame", &frame, 0.f, float(loop_ ? c.num_frames : last), "frame %.1f")) {
        play_time_ = frame / fps;
      }
      ImGui::SliderFloat("speed", &speed_, 0.05f, 3.f, "%.2fx");
      ImGui::TextDisabled("%d frames @ %.1f fps (%.2f s)", c.num_frames, fps, c.num_frames / fps);
    } else {
      ImGui::TextDisabled("rest pose (select a clip)");
    }
  }

  if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("mesh", &show_mesh_);
    ImGui::SameLine();
    ImGui::Checkbox("skeleton (S)", &show_skeleton_);
    ImGui::SameLine();
    ImGui::Checkbox("grid", &show_grid_);
    const bool has_model = current_ >= 0 && current_ < int(clips_.size()) && has_debug_skeleton(clips_[current_].clip);
    ImGui::Checkbox("model skeleton (M)", &show_model_skeleton_);
    ImGui::SameLine();
    if (has_model) ImGui::TextColored(ImVec4(0.12f, 1.f, 0.12f, 1.f), "%d joints", clips_[current_].clip.num_joints);
    else ImGui::TextDisabled("(clip has none)");
    if (ImGui::Button("Frame (F)")) {
      frame_camera();
      cam_user_moved_ = false;
    }
    ImGui::TextDisabled("%.0f fps (%.2f ms), %ux%u, MSAA x%d", fps_avg_, frame_ms_, sc_extent_.width,
                        sc_extent_.height, int(renderer_.samples()));
    if (busy && opt_.max_fps_while_generating > 0)
      ImGui::TextDisabled("rendering capped at %.0f fps while generating", opt_.max_fps_while_generating);
    ImGui::TextDisabled("LMB orbit, RMB/MMB pan, wheel zoom, Space play");
  }
  ImGui::End();
}

void App::update_pose() {
  if (!asset_) return;
  if (current_ >= 0 && current_ < int(clips_.size()) && clips_[current_].clip.num_frames > 0) {
    const ClipEntry& c = clips_[current_];
    const float fps = c.clip.fps > 0 ? c.clip.fps : 30.f;
    float f = float(play_time_ * fps);
    if (!loop_ && f >= float(c.clip.num_frames - 1)) {
      f = float(c.clip.num_frames - 1);
      playing_ = false;
    }
    sample_clip(*asset_, c.clip, c.binding, f, loop_, pose_);
    if (show_model_skeleton_ && has_debug_skeleton(c.clip)) {
      sample_debug_skeleton(c.clip, f, loop_, model_pos_);
      build_model_skeleton_lines(model_pos_, c.clip.parents, model_xf_, model_bones_, model_joints_);
    } else {
      model_bones_.clear();
      model_joints_.clear();
    }
  } else {
    rest_pose(*asset_, pose_);
    model_bones_.clear();
    model_joints_.clear();
  }
  build_palette(*asset_, pose_, palette_);
  build_skeleton_lines(*asset_, pose_, bones_, joints_);
}

bool App::save_capture(VkBuffer, VmaAllocation alloc, void* mapped) {
  vmaInvalidateAllocation(ctx_.vma, alloc, 0, VK_WHOLE_SIZE);
  const uint32_t w = sc_extent_.width, h = sc_extent_.height;
  std::vector<uint8_t> px(size_t(w) * h * 4);
  std::memcpy(px.data(), mapped, px.size());
  const bool bgra = sc_format_ == VK_FORMAT_B8G8R8A8_SRGB;
  for (size_t i = 0; i < px.size(); i += 4) {
    if (bgra) std::swap(px[i], px[i + 2]);
    px[i + 3] = 255;
  }
  return stbi_write_png(opt_.capture_png.c_str(), int(w), int(h), 4, px.data(), int(w) * 4) != 0;
}

int App::run() {
  if (!glfwInit()) {
    std::fprintf(stderr, "glfwInit failed\n");
    return 1;
  }
  std::string err;
  if (!glfwVulkanSupported()) {
    std::fprintf(stderr, "GLFW: Vulkan not supported\n");
    glfwTerminate();
    return 1;
  }
  uint32_t ne = 0;
  const char** exts = glfwGetRequiredInstanceExtensions(&ne);
  std::vector<const char*> inst_exts(exts, exts + ne);
  if (!ctx_.create_instance(inst_exts, err) || !init_window() ||
      glfwCreateWindowSurface(ctx_.instance, window_, nullptr, &surface_) != VK_SUCCESS ||
      !ctx_.create_device(surface_, err)) {
    std::fprintf(stderr, "viewer init failed: %s\n", err.empty() ? "window/surface creation" : err.c_str());
    glfwTerminate();
    return 1;
  }

  // Determine the swapchain format first (renderer pipelines depend on it).
  {
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.phys, surface_, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx_.phys, surface_, &n, fmts.data());
    for (VkFormat want : {VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB})
      for (auto& f : fmts)
        if (sc_format_ == VK_FORMAT_UNDEFINED && f.format == want && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
          sc_format_ = want;
    if (sc_format_ == VK_FORMAT_UNDEFINED) {
      std::fprintf(stderr, "surface has no sRGB format\n");
      return 1;
    }
  }
  if (!renderer_.init(&ctx_, sc_format_, ctx_.clamp_samples(opt_.msaa_samples), VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                      kFramesInFlight, err) ||
      !create_swapchain(err)) {
    std::fprintf(stderr, "renderer init failed: %s\n", err.c_str());
    return 1;
  }

  VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = ctx_.cmd_pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = kFramesInFlight;
  vkAllocateCommandBuffers(ctx_.device, &cai, cmd_);
  for (uint32_t i = 0; i < kFramesInFlight; ++i) {
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(ctx_.device, &fci, nullptr, &fence_[i]);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(ctx_.device, &sci, nullptr, &acquired_[i]);
  }

  // ---- ImGui ----
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr;
  ImGui::StyleColorsDark();
  {  // the swapchain is sRGB: convert the (sRGB-authored) style colours to linear
    ImGuiStyle& st = ImGui::GetStyle();
    for (int i = 0; i < ImGuiCol_COUNT; ++i) {
      ImVec4& c = st.Colors[i];
      c = ImVec4(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z), c.w);
    }
  }
  ImGui_ImplGlfw_InitForVulkan(window_, true);
  {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 16;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    vkCreateDescriptorPool(ctx_.device, &dpci, nullptr, &imgui_pool_);
    ImGui_ImplVulkan_InitInfo ii{};
    ii.Instance = ctx_.instance;
    ii.PhysicalDevice = ctx_.phys;
    ii.Device = ctx_.device;
    ii.QueueFamily = ctx_.queue_family;
    ii.Queue = ctx_.queue;
    ii.DescriptorPool = imgui_pool_;
    ii.RenderPass = renderer_.render_pass();
    ii.MinImageCount = min_image_count_;
    ii.ImageCount = std::max<uint32_t>(uint32_t(sc_images_.size()), min_image_count_);
    ii.MSAASamples = renderer_.samples();
    ImGui_ImplVulkan_Init(&ii);
  }

  panel_px_ = float(std::max(opt_.panel_width, 0));  // refined by the first UI frame
  if (!asset_path_.empty()) load_asset_ui(asset_path_);

  // capture readback buffer (allocated lazily)
  GpuBuffer capture_buf;
  bool capture_requested = false, captured = false;

  start_time_ = glfwGetTime();
  double last = start_time_;
  fps_window_start_ = start_time_;
  double pace_deadline = start_time_;  // frame-rate cap: next frame not before this time
  float framed_aspect = view_aspect();
  uint32_t frame = 0;
  int auto_step = 0;
  bool auto_generated = false, auto_closed = false;
  std::shared_ptr<GenJob> auto_cancelled_job;
  while (!glfwWindowShouldClose(window_)) {
    glfwPollEvents();
    const double now = glfwGetTime();
    const double elapsed = now - start_time_;

    // ---- automation ----
    if (opt_.auto_resize) {
      static const struct { double t; int w, h, iconify; } steps[] = {
          {1.0, 1100, 650, 0}, {1.8, 1700, 950, 0}, {2.6, 0, 0, 1}, {3.2, 0, 0, 2}, {3.8, opt_.width, opt_.height, 0}};
      if (auto_step < 5 && elapsed >= steps[auto_step].t) {
        auto& s = steps[auto_step++];
        if (s.iconify == 1) glfwIconifyWindow(window_);
        else if (s.iconify == 2) glfwRestoreWindow(window_);
        else glfwSetWindowSize(window_, s.w, s.h);
      }
    }
    if (opt_.auto_generate && !auto_generated && asset_ && elapsed > 0.3) {
      auto_generated = true;
      const bool first = start_generate();
      const bool second = start_generate();  // must be refused while the first runs
      if (print_stats_)
        std::printf("[viewer] auto-generate: started=%d, second concurrent start refused=%d\n", int(first),
                    int(!second));
    }
    if (job_running() && opt_.auto_cancel_after_seconds > 0 && auto_cancelled_job != job_ &&
        now - job_->start_time >= opt_.auto_cancel_after_seconds) {
      auto_cancelled_job = job_;
      if (print_stats_) std::printf("[viewer] auto-cancel after %.2f s\n", now - job_->start_time);
      cancel_job();
    }
    if (job_running() && opt_.auto_close_while_generating && !auto_closed && now - job_->start_time >= 0.5) {
      auto_closed = true;
      if (print_stats_) std::printf("[viewer] auto-close while generating\n");
      glfwSetWindowShouldClose(window_, GLFW_TRUE);
      continue;
    }
    bool exit_now = false;
    if (opt_.auto_exit_seconds > 0 && elapsed >= opt_.auto_exit_seconds) {
      if (opt_.capture_png.empty() || captured || !sc_transfer_src_) exit_now = true;
      else capture_requested = true;
    }
    if (exit_now) break;

    int fw = 0, fh = 0;
    glfwGetFramebufferSize(window_, &fw, &fh);
    if (fw == 0 || fh == 0 || glfwGetWindowAttrib(window_, GLFW_ICONIFIED)) {
      // minimised: nothing to present; sleep until an event (or a short timeout
      // for automation and job polling) instead of spinning
      glfwWaitEventsTimeout(0.1);
      ++stat_min_waits_;
      last = glfwGetTime();
      fps_window_start_ = last;
      fps_window_frames_ = 0;
      pace_deadline = last;
      continue;
    }
    if (sc_dirty_ || !swapchain_ || uint32_t(fw) != sc_extent_.width || uint32_t(fh) != sc_extent_.height) {
      vkDeviceWaitIdle(ctx_.device);
      if (!create_swapchain(err)) {
        std::fprintf(stderr, "swapchain: %s\n", err.c_str());
        break;
      }
      sc_dirty_ = false;
      ImGui_ImplVulkan_SetMinImageCount(min_image_count_);
      if (!swapchain_) continue;
    }

    vkWaitForFences(ctx_.device, 1, &fence_[frame], VK_TRUE, UINT64_MAX);
    uint32_t image = 0;
    VkResult ar = vkAcquireNextImageKHR(ctx_.device, swapchain_, UINT64_MAX, acquired_[frame], VK_NULL_HANDLE, &image);
    if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
      sc_dirty_ = true;
      continue;
    }
    if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) {
      std::fprintf(stderr, "vkAcquireNextImageKHR failed (%d)\n", int(ar));
      break;
    }
    vkResetFences(ctx_.device, 1, &fence_[frame]);

    const float dt = float(now - last);
    last = now;
    const bool was_generating = job_running();
    poll_job();
    if (playing_ && current_ >= 0) play_time_ += double(dt) * speed_;

    // frame-rate readout: frames / wall time over ~0.5 s windows
    ++fps_window_frames_;
    if (now - fps_window_start_ >= 0.5) {
      const double span = now - fps_window_start_;
      fps_avg_ = float(double(fps_window_frames_) / span);
      frame_ms_ = float(1000.0 * span / double(fps_window_frames_));
      fps_window_start_ = now;
      fps_window_frames_ = 0;
    }
    if (was_generating) {
      ++stat_gen_frames_;
      stat_gen_time_ += dt;
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    build_ui(dt);
    handle_input();
    ImGui::Render();
    update_pose();
    {  // keep the asset framed in the free area (window or panel resized) until the user moves the camera
      const float a = view_aspect();
      if (!cam_user_moved_ && std::abs(a - framed_aspect) > 1e-3f) frame_camera();
      framed_aspect = a;
    }

    FrameInput in;
    const glm::ivec4 vr = view_rect();
    in.view_rect = vr;
    in.view = cam_.view();
    in.proj = cam_.proj(float(vr.z) / float(vr.w));
    in.cam_pos = cam_.eye();
    in.palette = &palette_;
    in.bones = &bones_;
    in.joints = &joints_;
    in.model_bones = &model_bones_;
    in.model_joints = &model_joints_;
    in.draw_mesh = show_mesh_;
    in.draw_skeleton = show_skeleton_;
    in.draw_model_skeleton = show_model_skeleton_;
    in.draw_grid = show_grid_ && asset_;
    if (asset_) grid_for_bounds(asset_->bounds_min, asset_->bounds_max, in);
    renderer_.update(frame, in);

    VkCommandBuffer cmd = cmd_[frame];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    renderer_.begin_pass(cmd, image, glm::vec4(0.02f, 0.025f, 0.035f, 1.f));
    renderer_.draw(cmd, frame);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    renderer_.end_pass(cmd);
    const bool do_capture = capture_requested && !captured;
    if (do_capture) {
      ctx_.destroy_buffer(capture_buf);
      ctx_.create_buffer(VkDeviceSize(sc_extent_.width) * sc_extent_.height * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true,
                         capture_buf);
      VkImage img = sc_images_[image];
      image_barrier(cmd, img, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
      VkBufferImageCopy c{};
      c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      c.imageExtent = {sc_extent_.width, sc_extent_.height, 1};
      vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, capture_buf.buffer, 1, &c);
      image_barrier(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    }
    vkEndCommandBuffer(cmd);

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &acquired_[frame];
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &render_done_[image];
    vkQueueSubmit(ctx_.queue, 1, &si, fence_[frame]);

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &render_done_[image];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &image;
    VkResult pr = vkQueuePresentKHR(ctx_.queue, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) sc_dirty_ = true;
    else if (pr != VK_SUCCESS) {
      std::fprintf(stderr, "vkQueuePresentKHR failed (%d)\n", int(pr));
      break;
    }
    if (do_capture) {
      vkWaitForFences(ctx_.device, 1, &fence_[frame], VK_TRUE, UINT64_MAX);
      captured = save_capture(capture_buf.buffer, capture_buf.alloc, capture_buf.mapped);
      if (captured) std::printf("captured window to %s\n", opt_.capture_png.c_str());
      else captured = true;  // do not retry forever
    }
    frame = (frame + 1) % kFramesInFlight;
    ++stat_frames_;

    // Frame-rate cap on top of vsync (always while a job runs, so the render
    // loop leaves the GPU to the generator). Deadline based, so coarse OS timer
    // wake-ups average out; events received meanwhile are queued for ImGui.
    const float cap = job_running() && opt_.max_fps_while_generating > 0 ? opt_.max_fps_while_generating : opt_.max_fps;
    if (cap > 0) {
      const double period = 1.0 / double(cap);
      pace_deadline += period;
      double t = glfwGetTime();
      if (pace_deadline < t - period) pace_deadline = t;  // fell behind (or cap just enabled): restart
      while (t < pace_deadline && !glfwWindowShouldClose(window_)) {
        glfwWaitEventsTimeout(pace_deadline - t);
        t = glfwGetTime();
      }
    } else {
      pace_deadline = glfwGetTime();
    }
  }

  if (print_stats_) {
    const double t = glfwGetTime() - start_time_;
    std::printf("[viewer] stats: %llu frames in %.2f s (%.1f fps avg), panel readout %.1f fps\n",
                (unsigned long long)stat_frames_, t, t > 0 ? double(stat_frames_) / t : 0.0, fps_avg_);
    if (stat_min_waits_) std::printf("[viewer] minimised: %llu idle waits, nothing rendered\n", (unsigned long long)stat_min_waits_);
    if (stat_gen_time_ > 0)
      std::printf("[viewer] while generating: %llu frames in %.2f s (%.1f fps)\n",
                  (unsigned long long)stat_gen_frames_, stat_gen_time_, double(stat_gen_frames_) / stat_gen_time_);
  }
  // Never detach the worker: the generator object it uses belongs to the caller
  // and may be destroyed as soon as run_viewer returns. Cancel and join it.
  if (job_running()) glfwHideWindow(window_);  // the join may take a solver step or two
  stop_job();
  job_.reset();
  if (print_stats_) std::printf("[viewer] worker joined\n");
  vkDeviceWaitIdle(ctx_.device);
  ctx_.destroy_buffer(capture_buf);
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  vkDestroyDescriptorPool(ctx_.device, imgui_pool_, nullptr);
  for (uint32_t i = 0; i < kFramesInFlight; ++i) {
    vkDestroyFence(ctx_.device, fence_[i], nullptr);
    vkDestroySemaphore(ctx_.device, acquired_[i], nullptr);
  }
  renderer_.shutdown();
  destroy_swapchain(true);
  vkDestroySurfaceKHR(ctx_.instance, surface_, nullptr);
  ctx_.destroy();
  glfwDestroyWindow(window_);
  glfwTerminate();
  return 0;
}

}  // namespace

int run_viewer(const ViewerOptions& options) {
  App app(options);
  return app.run();
}

}  // namespace um::viewer

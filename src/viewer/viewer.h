// Public API of the UniMate real-time viewer (um_viewer).
//
//   #include "viewer/viewer.h"      // include dir: <repo>/src
//
// Depends only on core/generation.h (plain structs). Vulkan 1.2+, loaded at
// runtime through volk: no Vulkan SDK or validation layers required.
#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/generation.h"

namespace um::viewer {

// Thrown by the ProgressFn the viewer passes to GenerateFn once the job has
// been cancelled (Cancel button, window closed, asset reload). Generators must
// let it propagate (or at least rethrow): the viewer treats it, and any other
// exception raised after a cancel request, as "cancelled", not as an error.
struct GenerationCancelled : std::runtime_error {
  GenerationCancelled() : std::runtime_error("generation cancelled") {}
};

struct ViewerOptions {
  std::string asset_glb;  // initial asset (may be empty; can be loaded from the UI)

  // Called on a worker thread when the user presses "Generate". Null disables
  // the button. Exceptions are caught and shown in the UI. Cancellation is
  // cooperative: the ProgressFn it receives throws GenerationCancelled when the
  // job is cancelled, so call it regularly (e.g. every solver step). run_viewer
  // always joins the worker before returning (it never detaches it), so objects
  // captured by `generate` may safely be destroyed after run_viewer returns.
  um::GenerateFn generate;

  // Called (on the UI thread) every time an asset is (re)loaded, including the
  // initial one, so the generator knows which GLB it must animate.
  std::function<void(const std::string& asset_glb)> on_asset_loaded;

  // Optional "Export GLB" button (called on the UI thread). Null hides it.
  std::function<void(const um::MotionClip&, const std::string& out_path)> export_glb;

  // Initial values of the generation panel.
  um::GenerateRequest initial_request;

  int width = 1600, height = 900;
  std::string title = "UniMate viewer";
  bool vsync = true;     // FIFO present mode (paced by the display refresh)
  int msaa_samples = 4;  // 1, 2, 4, 8 (clamped to what the device supports)

  // Frame pacing on top of vsync. 0 = no extra cap. While a generation job is
  // running the render loop is capped at max_fps_while_generating so it leaves
  // the GPU to the generator's compute work. A minimised window never renders.
  float max_fps = 0.f;
  float max_fps_while_generating = 30.f;

  // Width in pixels of the control panel, docked on the left. The 3D view
  // (projection centre, auto-framing, mouse input) uses the remaining area.
  int panel_width = 400;

  // ---- skeleton overlays ----
  bool show_skeleton = true;        // the asset's posed skin joints
  // The current clip's own debug skeleton (MotionClip::joint_positions /
  // parents, in the model's canonical space), drawn in green/magenta when the
  // clip has one. It should coincide with the skin-joint overlay when the
  // motion mapping is right.
  bool show_model_skeleton = true;
  // Column-major 4x4 (glTF/glm convention) applied to the model skeleton's
  // positions before drawing: canonical space -> glTF world. Identity default.
  float model_skeleton_transform[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

  // ---- automation / smoke testing (no user input needed) ----
  double auto_exit_seconds = 0.0;   // > 0: close the window after this time (cancels and joins a running job)
  bool auto_generate = false;       // press "Generate" once at start-up
  bool auto_resize = false;         // resize the window a few times while running
  std::string capture_png;          // if set, save the last presented frame (incl. UI) here
  double auto_cancel_after_seconds = 0.0;  // > 0: press "Cancel" this long after a job started
  bool auto_close_while_generating = false;  // close the window ~0.5 s into the first job
  bool print_stats = false;         // print present mode and frame statistics to stdout
                                    // (also on when auto_exit_seconds > 0)
};

// Opens the window and runs until it is closed. Returns 0 on success.
int run_viewer(const ViewerOptions& options);

struct OffscreenOptions {
  bool draw_skeleton = true;
  bool draw_grid = true;
  // Orbit camera around the auto-framed bind-pose bounds. yaw = 0 looks at the
  // asset from +Z (its front, glTF forward), towards -Z. Degrees.
  float yaw_deg = 30.f;
  float pitch_deg = 20.f;
  float distance_scale = 1.0f;  // multiplies the auto-framing distance
  int msaa_samples = 4;
  std::string* error = nullptr;  // optional: receives the failure reason
  // Clip debug skeleton overlay (see ViewerOptions::show_model_skeleton).
  bool draw_model_skeleton = true;
  float model_skeleton_transform[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};

// Renders one frame of `clip` (or the bind/rest pose when clip == nullptr) to
// a w x h PNG without creating a window. `frame` is clamped to the clip.
bool render_offscreen_png(const std::string& asset_glb, const um::MotionClip* clip,
                          int frame, int w, int h, const std::string& out_png);
bool render_offscreen_png(const std::string& asset_glb, const um::MotionClip* clip,
                          int frame, int w, int h, const std::string& out_png,
                          const OffscreenOptions& options);

// Skeleton description of an asset (nodes used as joints by its skins, in
// glTF node order), with their rest local TRS. Handy for procedural clips.
struct JointInfo {
  std::string name;
  int parent = -1;           // index into the returned vector, -1 for roots
  float translation[3] = {0, 0, 0};
  float rotation[4] = {0, 0, 0, 1};  // x y z w
  float scale[3] = {1, 1, 1};
};
std::vector<JointInfo> load_skeleton(const std::string& asset_glb, std::string* error = nullptr);

// The asset's embedded glTF animations, resampled at `fps` as MotionClips.
std::vector<um::MotionClip> load_embedded_clips(const std::string& asset_glb, float fps = 30.f);

}  // namespace um::viewer

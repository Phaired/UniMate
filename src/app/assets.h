// Locating models and converted asset bundles.
//
// Model directory (UMTF files written by tools/python/*_export.py /
// pipeline_convert_*.py):  --models DIR  >  $UNIMATE_MODELS  >
// <repo>/models/converted.
//
// An asset is a (bundle UMTF, GLB) pair. resolve_asset() accepts
//   * a bundle name:        "eagle"  -> <models>/bundles/eagle.umtf
//   * a bundle UMTF path:   ".../bundles/eagle.umtf"
//   * an asset directory:   "reference/UniMate/assets/examples/eagle"
//   * a GLB path:           ".../eagle/Eagle.glb"
// The GLB of a bundle comes from its meta (source dir + glb name); the bundle
// of a GLB / directory is <models>/bundles/<dir name>.umtf, or any bundle whose
// meta points at that GLB. A GLB without a converted bundle is an error that
// names the conversion command.
#pragma once

#include <string>
#include <vector>

namespace um::app {

std::string repo_root();  // compile-time source root
std::string default_models_dir();
// `cli_value` wins if non-empty, then $UNIMATE_MODELS, then the default.
std::string resolve_models_dir(const std::string& cli_value);

struct ModelPaths {
  std::string dir, t5, denoiser, stats;
};
// Throws (with the conversion commands) when a file is missing.
ModelPaths model_paths(const std::string& models_dir);

struct AssetPaths {
  std::string name;         // bundle name (file stem)
  std::string bundle_umtf;
  std::string glb;
};
AssetPaths resolve_asset(const std::string& spec, const std::string& models_dir);

// Names of the converted bundles in <models>/bundles.
std::vector<std::string> list_bundles(const std::string& models_dir);

}  // namespace um::app

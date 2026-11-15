// Single translation unit holding the implementations of the header-only
// third-party libraries used by the viewer.
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define VMA_IMPLEMENTATION
#include "vk_context.h"

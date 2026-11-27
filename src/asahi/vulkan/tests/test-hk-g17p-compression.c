/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT
 *
 * Pure checks of the G17P lossless-compression admission policy
 * (hk_g17p_compression.h). The generic hk_can_compress() checks are applied
 * separately by the driver.
 */
#include "asahi/vulkan/hk_g17p_compression.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned checks;
#define CHECK(x)                                                              \
   do {                                                                       \
      checks++;                                                               \
      if (!(x)) {                                                             \
         fprintf(stderr, "line %u: %s\n", __LINE__, #x);                      \
         exit(1);                                                             \
      }                                                                       \
   } while (0)

#define ELIGIBLE(i) hk_g17p_compression_eligible(AGX_CHIP_G17P, 0, &(i))

static VkImageCreateInfo
image(VkFormat format, uint32_t w, uint32_t h, uint32_t mips,
      VkImageUsageFlags usage)
{
   return (VkImageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {w, h, 1},
      .mipLevels = mips,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
}

static void
check_formats(void)
{
   const VkImageUsageFlags color_usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

   /* Every core format below VK_FORMAT_ASTC_12x12_SRGB_BLOCK. */
   unsigned color = 0, color16 = 0, depth = 0;
   for (unsigned f = 1; f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; ++f) {
      const bool c = hk_g17p_compression_color_format((VkFormat)f);
      const bool c16 = hk_g17p_compression_color_format_16b((VkFormat)f);
      const bool d = hk_g17p_compression_depth_format((VkFormat)f);
      CHECK(!(c && d));
      CHECK(!(c && c16));
      CHECK(!(c16 && d));
      color += c;
      color16 += c16;
      depth += d;

      VkImageCreateInfo i = image((VkFormat)f, 64, 64, 1, color_usage);
      CHECK(ELIGIBLE(i) == (c || c16 || d));
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G17P, AGX_CHIP_ID_A18_PRO,
                                           0, &i) == (c || c16 || d));
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G17P, AGX_CHIP_ID_M4_PRO,
                                           0, &i) == (c || c16 || d));
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G17P, AGX_CHIP_ID_M4_MAX,
                                           0, &i) == (c || c16 || d));
   }
   CHECK(color == 46);
   CHECK(color16 == 3);
   CHECK(depth == 3);

   /* No packed/padded RGB, no stencil or combined depth/stencil, no
    * block-compressed or YCbCr formats. */
   static const VkFormat rejected[] = {
      VK_FORMAT_R32G32B32_SFLOAT,
      VK_FORMAT_R8G8B8_UNORM,        VK_FORMAT_R16G16B16_SFLOAT,
      VK_FORMAT_B10G11R11_UFLOAT_PACK32,
      VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, VK_FORMAT_R4G4B4A4_UNORM_PACK16,
      VK_FORMAT_A1R5G5B5_UNORM_PACK16, VK_FORMAT_S8_UINT,
      VK_FORMAT_D24_UNORM_S8_UINT,   VK_FORMAT_D16_UNORM_S8_UINT,
      VK_FORMAT_X8_D24_UNORM_PACK32,
      VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
      VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, VK_FORMAT_G8B8G8R8_422_UNORM,
      VK_FORMAT_R64_UINT,
   };
   for (unsigned r = 0; r < sizeof(rejected) / sizeof(rejected[0]); ++r) {
      VkImageCreateInfo i = image(rejected[r], 64, 64, 1, color_usage);
      CHECK(!ELIGIBLE(i));
   }
}

static void
check_shape_for_chip(VkFormat format, bool is_depth, uint32_t chip_id)
{
   const bool is_color_16b = hk_g17p_compression_color_format_16b(format);
   const VkImageUsageFlags usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT |
      (is_depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
   const VkImageCreateInfo base = image(format, 2408, 1508, 1, usage);
#define CHIP_ELIGIBLE(i)                                                       \
   hk_hal200_compression_eligible(AGX_CHIP_G17P, chip_id, 0, &(i))
   CHECK(CHIP_ELIGIBLE(base));

   /* Chip and plane. */
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
      CHECK(hk_hal200_compression_eligible((enum agx_chip)chip, chip_id, 0,
                                           &base) ==
            agx_chip_has_cap((enum agx_chip)chip,
                             AGX_CAP_HAL200_LOSSLESS_COMPRESSION));
   /* Plane 1 is the stencil plane of D32_SFLOAT_S8_UINT alone. */
   CHECK(hk_hal200_compression_eligible(AGX_CHIP_G17P, chip_id, 1, &base) ==
         (format == VK_FORMAT_D32_SFLOAT_S8_UINT));
   CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G17P, chip_id, 2, &base));
   CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G17P, chip_id, 0, NULL));
   CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G17P, chip_id, 1, NULL));

   /* Usage: only texture/PBE/ZLS usages are admitted. */
   VkImageUsageFlags allowed =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
   if (is_depth)
      allowed |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
   for (unsigned bit = 0; bit < 32; ++bit) {
      VkImageCreateInfo i = base;
      i.usage |= 1u << bit;
      CHECK(CHIP_ELIGIBLE(i) == !!((1u << bit) & allowed));
   }
   for (unsigned usage_bits = 0; usage_bits < 256; ++usage_bits) {
      VkImageCreateInfo i = base;
      i.usage = usage_bits & allowed;
      CHECK(CHIP_ELIGIBLE(i));
   }
   static const VkImageUsageFlags refused[] = {
      VK_IMAGE_USAGE_STORAGE_BIT, VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT,
      VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
      VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR,
      VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR,
      VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
      VK_IMAGE_USAGE_FRAGMENT_DENSITY_MAP_BIT_EXT,
   };
   for (unsigned r = 0; r < sizeof(refused) / sizeof(refused[0]); ++r) {
      VkImageCreateInfo i = base;
      i.usage |= refused[r];
      CHECK(CHIP_ELIGIBLE(i) ==
            (is_depth &&
             refused[r] == VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT));
   }

   /* Sample-location tracking is also admitted for single-sample depth.
    * Aliasing is admitted for every shape; storage stays refused with it. */
   for (unsigned bit = 0; bit < 32; ++bit) {
      VkImageCreateInfo i = base;
      i.flags = 1u << bit;
      CHECK(CHIP_ELIGIBLE(i) ==
            ((1u << bit) == VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT ||
             (1u << bit) == VK_IMAGE_CREATE_EXTENDED_USAGE_BIT ||
             (1u << bit) == VK_IMAGE_CREATE_ALIAS_BIT ||
             (is_depth && (1u << bit) ==
                          VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT)));
   }
   {
      VkImageCreateInfo i = base;
      i.flags = VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
      CHECK(CHIP_ELIGIBLE(i));
      i.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
      CHECK(!CHIP_ELIGIBLE(i));
   }

   /* Geometry. */
   VkImageCreateInfo i = base;
   i.imageType = VK_IMAGE_TYPE_1D;
   CHECK(!CHIP_ELIGIBLE(i));
   i = base; i.imageType = VK_IMAGE_TYPE_3D;
   CHECK(CHIP_ELIGIBLE(i) == (!is_depth && !is_color_16b));
   i = base; i.extent.depth = 2;
   CHECK(!CHIP_ELIGIBLE(i));
   i = base; i.arrayLayers = 2;
   CHECK(CHIP_ELIGIBLE(i));
   i = base; i.arrayLayers = 6;
   CHECK(CHIP_ELIGIBLE(i));
   /* Mipmapped 2D arrays and mipmapped cubemaps compress for 1-8B and 16B
    * colour; depth and mipmapped 3D stay plain. */
   i = base; i.arrayLayers = 4; i.mipLevels = 4;
   CHECK(CHIP_ELIGIBLE(i) == !is_depth);
   i = base; i.extent.width = 256; i.extent.height = 256;
   i.arrayLayers = 6; i.mipLevels = 4;
   i.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
   CHECK(CHIP_ELIGIBLE(i) == !is_depth);
   i = base; i.imageType = VK_IMAGE_TYPE_3D; i.extent.depth = 4; i.mipLevels = 2;
   CHECK(!CHIP_ELIGIBLE(i));
   for (unsigned s = 2; s <= 64; s <<= 1) {
      i = base; i.samples = (VkSampleCountFlagBits)s;
      CHECK(CHIP_ELIGIBLE(i) == (!is_color_16b && (s == 2 || s == 4)));
      i.arrayLayers = 4;
      CHECK(CHIP_ELIGIBLE(i) == (!is_color_16b && (s == 2 || s == 4)));
   }
   i = base; i.tiling = VK_IMAGE_TILING_LINEAR;
   CHECK(!CHIP_ELIGIBLE(i));
   i = base; i.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
   CHECK(!CHIP_ELIGIBLE(i));
   i = base; i.mipLevels = 0;
   CHECK(!CHIP_ELIGIBLE(i));
   i = base; i.mipLevels = 12;
   CHECK(CHIP_ELIGIBLE(i) == !is_depth);
   i = base; i.extent.width = 15;
   CHECK(!CHIP_ELIGIBLE(i));
   i = base; i.extent.height = 15;
   CHECK(!CHIP_ELIGIBLE(i));
   i = base; i.extent.width = 16; i.extent.height = 16;
   CHECK(CHIP_ELIGIBLE(i));
   i = base; i.extent.width = 17; i.extent.height = 31;
   CHECK(CHIP_ELIGIBLE(i));
   i = base; i.extent.width = 16384; i.extent.height = 16384;
   CHECK(CHIP_ELIGIBLE(i));
   i = base; i.sharingMode = VK_SHARING_MODE_CONCURRENT;
   CHECK(CHIP_ELIGIBLE(i));

   /* Shared images stay plain; an empty external handle set does not. */
   VkExternalMemoryImageCreateInfo external = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   VkImageFormatListCreateInfo list = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
      .pNext = &external,
   };
   i = base; i.pNext = &external;
   CHECK(!CHIP_ELIGIBLE(i));
   i = base; i.pNext = &list;
   CHECK(!CHIP_ELIGIBLE(i));
   external.handleTypes = 0;
   CHECK(CHIP_ELIGIBLE(i));
   list.pNext = NULL;
   CHECK(CHIP_ELIGIBLE(i));
   VkBaseInStructure wsi = {.sType = HK_G17P_WSI_IMAGE_CREATE_INFO_MESA};
   i = base; i.pNext = &wsi;
   CHECK(!CHIP_ELIGIBLE(i));
#undef CHIP_ELIGIBLE
}

static void
check_shape(VkFormat format, bool is_depth)
{
   check_shape_for_chip(format, is_depth, 0);
}

static void
check_minecraft(void)
{
   /* The live Minecraft 26.3 images: two 2408x1508 RGBA8 colour targets
    * (usage 0x17), the 2048^2 five-level blocks atlas (0x17) and the
    * 2408x1508 D32 depth target (0x27). All take the compressed path. */
   const VkImageCreateInfo color =
      image(VK_FORMAT_R8G8B8A8_UNORM, 2408, 1508, 1, 0x17);
   const VkImageCreateInfo atlas =
      image(VK_FORMAT_R8G8B8A8_UNORM, 2048, 2048, 5, 0x17);
   const VkImageCreateInfo depth = image(VK_FORMAT_D32_SFLOAT, 2408, 1508, 1,
                                         0x27);
   CHECK(ELIGIBLE(color));
   CHECK(ELIGIBLE(atlas));
   CHECK(ELIGIBLE(depth));
   /* Small colour images the application never renders to stay plain:
    * each upload into a compressed image is a render pass, and a small
    * blit-generated RGBA8/BGRA8 UNORM mip chain stays on the raw compute
    * path. From HK_G17P_MIP_PBE_MIN_TEXELS texels, and attachments at any
    * size, compress. */
   const VkImageUsageFlags texture_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
   const VkFormat compute_blit[] = {VK_FORMAT_R8G8B8A8_UNORM,
                                    VK_FORMAT_B8G8R8A8_UNORM};
   for (unsigned f = 0; f < 2; ++f) {
      VkImageCreateInfo t = image(compute_blit[f], 256, 256, 9, texture_usage);
      CHECK(!ELIGIBLE(t));
      t.mipLevels = 1;
      CHECK(!ELIGIBLE(t));
      VkImageCreateInfo single = image(compute_blit[f], 512, 256, 1, texture_usage);
      CHECK(ELIGIBLE(single));
      t.mipLevels = 9;
      t.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      CHECK(ELIGIBLE(t));

      /* 362x362 is 131,044 texels, just below 2^17; 512x256 is 2^17. */
      VkImageCreateInfo big = image(compute_blit[f], 362, 362, 9, texture_usage);
      CHECK(!ELIGIBLE(big));
      big = image(compute_blit[f], 512, 256, 10, texture_usage);
      CHECK(ELIGIBLE(big));
      big = image(compute_blit[f], 4096, 4096, 13, texture_usage);
      CHECK(ELIGIBLE(big));
   }

   /* The same holds for every compressed colour format and mip count: a
    * small glyph-cache style R8 texture taking tile uploads stays plain
    * unless it is also rendered to. */
   VkImageCreateInfo r8 = image(VK_FORMAT_R8_UNORM, 130, 130, 1, texture_usage);
   CHECK(!ELIGIBLE(r8));
   r8.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   CHECK(ELIGIBLE(r8));
   r8 = image(VK_FORMAT_R8_UNORM, 512, 512, 1, texture_usage);
   CHECK(ELIGIBLE(r8));
   VkImageCreateInfo srgb = image(VK_FORMAT_R8G8B8A8_SRGB, 256, 256, 9,
                                  texture_usage);
   CHECK(!ELIGIBLE(srgb));
   srgb.extent.width = 512;
   CHECK(ELIGIBLE(srgb));
   VkImageCreateInfo half = image(VK_FORMAT_R16G16B16A16_SFLOAT, 256, 256, 9,
                                  texture_usage);
   CHECK(!ELIGIBLE(half));
   half.extent.width = 512;
   CHECK(ELIGIBLE(half));

   /* The swapchain is external/WSI memory and stays plain. */
   VkExternalMemoryImageCreateInfo external = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   VkImageCreateInfo i = image(VK_FORMAT_B8G8R8A8_UNORM, 2408, 1508, 1, 0x13);
   CHECK(ELIGIBLE(i));
   i.pNext = &external;
   CHECK(!ELIGIBLE(i));
}

static void
check_zink(void)
{
   /* Portal 2 under zink (t-1079): GL render targets declare
    * feedback-loop usage and keep the uncompressed layout. The GL Z24S8
    * depth buffer is D32S8 on Honeykrisp; both its depth and its stencil
    * plane compress. */
   const VkImageUsageFlags rt = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
   const VkImageUsageFlags ds = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
   VkImageCreateInfo color = image(VK_FORMAT_R8G8B8A8_SRGB, 2408, 1506, 1, rt);
   color.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
   const VkFormat views[] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB};
   VkImageFormatListCreateInfo list = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
      .viewFormatCount = 2,
      .pViewFormats = views,
   };
   color.pNext = &list;
   CHECK(!ELIGIBLE(color));
   VkImageCreateInfo depth =
      image(VK_FORMAT_D32_SFLOAT_S8_UINT, 2408, 1506, 1, ds);
   depth.flags = VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;
   CHECK(ELIGIBLE(depth));
   CHECK(hk_g17p_compression_eligible(AGX_CHIP_G17P, 1, &depth));
   /* This is Portal's actual shape. A device-copy performance query must
    * find its compressed layout when host transfer is removed; otherwise
    * zink retains HOST_TRANSFER and never uses depth compression. */
   depth.usage |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
   CHECK(!ELIGIBLE(depth));
   depth.usage &= ~VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
   CHECK(ELIGIBLE(depth));
   /* The same depth buffer at 2x/4x MSAA, with zink's feedback-loop usage
    * and sample-location flag: its depth plane compresses, the S8 plane and
    * input-attachment use stay plain. */
   for (unsigned s = 2; s <= 4; s <<= 1) {
      VkImageCreateInfo ms = depth;
      ms.samples = (VkSampleCountFlagBits)s;
      CHECK(ELIGIBLE(ms));
      CHECK(!hk_g17p_compression_eligible(AGX_CHIP_G17P, 1, &ms));
      ms.usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
      CHECK(!ELIGIBLE(ms));
   }
   depth.samples = VK_SAMPLE_COUNT_8_BIT;
   CHECK(!ELIGIBLE(depth));
   const VkImageCreateInfo d32 = image(VK_FORMAT_D32_SFLOAT, 2408, 1506, 1, ds);
   CHECK(ELIGIBLE(d32));
   /* A host-transfer image keeps the plain layout host copies require. */
   VkImageCreateInfo hic = color;
   hic.usage |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
   CHECK(!ELIGIBLE(hic));
}

#define SHARED(i) hk_g17p_shared_compression_eligible(AGX_CHIP_G17P, 0, &(i))

static VkImageCreateInfo
swapchain_image(VkFormat format, uint32_t w, uint32_t h,
                VkImageUsageFlags usage)
{
   VkImageCreateInfo i = image(format, w, h, 1, usage);
   i.flags = VK_IMAGE_CREATE_ALIAS_BIT;
   i.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
   return i;
}

static void
check_shared(void)
{
   /* vkmark's swapchain (headless and Wayland): one colour attachment. The
    * WSI image carries ALIAS, the Mesa WSI marker and external memory, all
    * of which the shared contract admits. */
   VkExternalMemoryImageCreateInfo external = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   VkBaseInStructure wsi = {
      .sType = HK_G17P_WSI_IMAGE_CREATE_INFO_MESA,
      .pNext = (const VkBaseInStructure *)&external,
   };
   VkImageCreateInfo vkmark = swapchain_image(
      VK_FORMAT_B8G8R8A8_SRGB, 1920, 1080, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
   vkmark.pNext = &wsi;
   CHECK(SHARED(vkmark));
   /* The private policy keeps refusing it: sharing goes through the
    * modifier contract only. */
   CHECK(!ELIGIBLE(vkmark));
   vkmark.tiling = VK_IMAGE_TILING_OPTIMAL;
   CHECK(!SHARED(vkmark));
   vkmark.tiling = VK_IMAGE_TILING_LINEAR;
   CHECK(!SHARED(vkmark));

   /* Every private colour format, and nothing else (depth, stencil,
    * 16-byte texels, packed RGB, YCbCr, block compression). */
   unsigned shared = 0;
   for (unsigned f = 1; f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; ++f) {
      VkImageCreateInfo i = swapchain_image((VkFormat)f, 64, 64,
                                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
      CHECK(SHARED(i) == hk_g17p_compression_color_format((VkFormat)f));
      shared += SHARED(i);
   }
   CHECK(shared == 46);
   static const VkFormat refused_formats[] = {
      VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT_S8_UINT,
      VK_FORMAT_S8_UINT, VK_FORMAT_R32G32B32A32_SFLOAT,
      VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
      VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
   };
   for (unsigned r = 0;
        r < sizeof(refused_formats) / sizeof(refused_formats[0]); ++r) {
      VkImageCreateInfo i = swapchain_image(refused_formats[r], 64, 64,
                                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
      CHECK(!SHARED(i));
   }

   const VkImageCreateInfo base = swapchain_image(
      VK_FORMAT_R8G8B8A8_UNORM, 2408, 1506,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
   CHECK(SHARED(base));

   /* Native HAL200 (G16G, G17P) and M5 only, plane 0 only. */
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
      CHECK(hk_g17p_shared_compression_eligible((enum agx_chip)chip, 0,
                                                &base) ==
            hk_chip_shared_compression_supported((enum agx_chip)chip));
   CHECK(!hk_g17p_shared_compression_eligible(AGX_CHIP_G17P, 1, &base));
   CHECK(!hk_g17p_shared_compression_eligible(AGX_CHIP_G17P, 0, NULL));

   /* Usage: texture/PBE usages plus import-side feedback loops. Storage and
    * host transfer access the image raw. */
   const VkImageUsageFlags allowed_usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
   for (unsigned bit = 0; bit < 32; ++bit) {
      VkImageCreateInfo i = base;
      i.usage |= 1u << bit;
      CHECK(SHARED(i) == !!((1u << bit) & allowed_usage));
   }
   /* No colour-attachment requirement: a blit-only swapchain is a transfer
    * destination, which hk_image_init() makes renderable. */
   VkImageCreateInfo blit_only = base;
   blit_only.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   CHECK(SHARED(blit_only));

   /* Flags: mutable/extended views and WSI's ALIAS only. */
   for (unsigned bit = 0; bit < 32; ++bit) {
      VkImageCreateInfo i = base;
      i.flags = 1u << bit;
      CHECK(SHARED(i) == ((1u << bit) == VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT ||
                          (1u << bit) == VK_IMAGE_CREATE_EXTENDED_USAGE_BIT ||
                          (1u << bit) == VK_IMAGE_CREATE_ALIAS_BIT));
   }

   /* One level, one layer, one sample, 2D, at least 16x16. */
   VkImageCreateInfo i = base;
   i.mipLevels = 2;
   CHECK(!SHARED(i));
   i = base; i.arrayLayers = 2;
   CHECK(!SHARED(i));
   i = base; i.extent.depth = 2;
   CHECK(!SHARED(i));
   i = base; i.imageType = VK_IMAGE_TYPE_3D;
   CHECK(!SHARED(i));
   i = base; i.imageType = VK_IMAGE_TYPE_1D;
   CHECK(!SHARED(i));
   for (unsigned s = 2; s <= 64; s <<= 1) {
      i = base; i.samples = (VkSampleCountFlagBits)s;
      CHECK(!SHARED(i));
   }
   i = base; i.extent.width = 15;
   CHECK(!SHARED(i));
   i = base; i.extent.height = 15;
   CHECK(!SHARED(i));
   i = base; i.extent.width = 16; i.extent.height = 16;
   CHECK(SHARED(i));
   i = base; i.extent.width = 16384; i.extent.height = 16384;
   CHECK(SHARED(i));

   /* The format-query shape (no extent) agrees with the create shape. */
   CHECK(hk_g17p_shared_compression_shape(AGX_CHIP_G17P, base.format,
                                          base.imageType, base.flags,
                                          base.usage));
   CHECK(!hk_g17p_shared_compression_shape(AGX_CHIP_G17P, base.format,
                                           base.imageType, base.flags,
                                           VK_IMAGE_USAGE_STORAGE_BIT));
   CHECK(!hk_g17p_shared_compression_shape(AGX_CHIP_G14G, base.format,
                                           base.imageType, base.flags,
                                           base.usage));

   /* zink's kopper swapchain and imports carry feedback-loop usage. The
    * shape admits it so an import of a compressed buffer succeeds;
    * hk_can_compress() still refuses it when allocating. */
   VkImageCreateInfo kopper = base;
   kopper.usage |= VK_IMAGE_USAGE_SAMPLED_BIT |
                   VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
                   VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
   CHECK(SHARED(kopper));
}

static void
check_m5_depth(void)
{
   /* Minecraft under zink: a D32 depth buffer with zink's usages and the
    * sample-location flag zink sets on depth attachments. */
   const VkImageUsageFlags ds = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
   static const VkFormat formats[] = {
      VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
   };
   for (unsigned f = 0; f < sizeof(formats) / sizeof(formats[0]); ++f) {
      VkImageCreateInfo i = image(formats[f], 2976, 1872, 1, ds);
      i.flags = VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;
      for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
         CHECK(hk_m5_depth_compression_eligible((enum agx_chip)chip, 0, &i) ==
               agx_chip_g17_m5((enum agx_chip)chip));
      /* Of the second planes, only the single-sample stencil plane of
       * D32_SFLOAT_S8_UINT is admitted. */
      CHECK(hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 1, &i) ==
            (formats[f] == VK_FORMAT_D32_SFLOAT_S8_UINT));
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, NULL));
      /* Neither policy admits a depth image as colour, nor colour as depth. */
      CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &i));

      VkImageCreateInfo c = i;
      c.usage |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c = i;
      c.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c = i;
      c.usage &= ~VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c = i;
      c.samples = VK_SAMPLE_COUNT_4_BIT;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c = i;
      c.mipLevels = 2;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      /* Layered depth compresses on every M5 part. */
      c = i;
      c.arrayLayers = 2;
      CHECK(hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      CHECK(hk_m5_depth_compression_eligible(AGX_CHIP_G17G, 0, &c));
      CHECK(hk_m5_depth_compression_eligible(AGX_CHIP_G17C, 0, &c));
      /* So do cube-compatible depth arrays (vkd3d-proton's square six-layer
       * placed depth, both planes), but only in a cube's shape. */
      c = image(formats[f], 1024, 1024, 1, ds);
      c.arrayLayers = 6;
      c.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT | VK_IMAGE_CREATE_ALIAS_BIT;
      for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip) {
         CHECK(hk_m5_depth_compression_eligible((enum agx_chip)chip, 0, &c) ==
               agx_chip_g17_m5((enum agx_chip)chip));
         CHECK(hk_m5_depth_compression_eligible((enum agx_chip)chip, 1, &c) ==
               (agx_chip_g17_m5((enum agx_chip)chip) &&
                formats[f] == VK_FORMAT_D32_SFLOAT_S8_UINT));
      }
      c.arrayLayers = 12;
      CHECK(hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c.arrayLayers = 8;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c.arrayLayers = 6;
      c.extent.height = 512;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c.extent.height = 1024;
      c.mipLevels = 2;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c = i;
      c.flags |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      /* Placed aliases of identical parameters, both planes. */
      c = i;
      c.flags |= VK_IMAGE_CREATE_ALIAS_BIT;
      for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip) {
         CHECK(hk_m5_depth_compression_eligible((enum agx_chip)chip, 0, &c) ==
               agx_chip_g17_m5((enum agx_chip)chip));
         CHECK(hk_m5_depth_compression_eligible((enum agx_chip)chip, 1, &c) ==
               (agx_chip_g17_m5((enum agx_chip)chip) &&
                formats[f] == VK_FORMAT_D32_SFLOAT_S8_UINT));
      }
      c = i;
      c.extent.width = 15;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c = i;
      VkExternalMemoryImageCreateInfo external = {
         .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
         .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
      };
      c.pNext = &external;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));

      /* Later links are read too: external memory or the WSI marker behind
       * another structure still refuses compression. */
      VkImageFormatListCreateInfo list = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
      };
      VkBaseInStructure wsi = {.sType = HK_G17P_WSI_IMAGE_CREATE_INFO_MESA};
      c = i;
      c.pNext = &list;
      CHECK(hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      list.pNext = &external;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      list.pNext = &wsi;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
   }

   /* No stencil-only or other depth/stencil formats. */
   static const VkFormat rejected[] = {
      VK_FORMAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT,
      VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_X8_D24_UNORM_PACK32,
      VK_FORMAT_R32_SFLOAT,
   };
   for (unsigned r = 0; r < sizeof(rejected) / sizeof(rejected[0]); ++r) {
      VkImageCreateInfo i = image(rejected[r], 64, 64, 1, ds);
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &i));
   }
}

static void
check_m5_color(void)
{
   const VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   /* Minecraft's blocks atlas: RGBA8, 2048x2048, five mips; zink gives its
    * sampled textures colour-attachment usage. */
   VkImageCreateInfo i = image(VK_FORMAT_R8G8B8A8_UNORM, 2048, 2048, 5, usage);
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
      CHECK(hk_m5_color_compression_eligible((enum agx_chip)chip, 0, &i) ==
            agx_chip_has_cap((enum agx_chip)chip,
                             AGX_CAP_GEN10_2D_COMPRESSION));
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G18G, 0, &i));
   i.mipLevels = 1;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &i));
   i.mipLevels = 12;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &i));
   /* 2D arrays (single-level and mipmapped), cubes, and single-level 3D are
    * admitted for 1-8 byte colour formats, on every chip with Gen10 private
    * compression and on no other (G18G included). */
   VkImageCreateInfo c = i;
   c.arrayLayers = 2;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
      CHECK(hk_m5_color_compression_eligible((enum agx_chip)chip, 0, &c) ==
            agx_chip_has_cap((enum agx_chip)chip,
                             AGX_CAP_GEN10_2D_COMPRESSION));
   c.mipLevels = 1;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G18G, 0, &c));
   /* 2D_ARRAY_COMPATIBLE belongs to 3D images only. */
   c.flags = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i;
   c.mipLevels = 1;
   c.extent.width = 128;
   c.extent.height = 128;
   c.arrayLayers = 6;
   c.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17C, 0, &c));
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G18G, 0, &c));
   /* Cube arrays and mipmapped cubes too; a layer count that is not a whole
    * number of faces, or non-square faces, are not cubes. */
   c.arrayLayers = 12;
   c.mipLevels = 8;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c.arrayLayers = 7;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c.arrayLayers = 6;
   c.extent.width = 129;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i;
   c.imageType = VK_IMAGE_TYPE_3D;
   c.extent.width = 65;
   c.extent.height = 63;
   c.extent.depth = 4;
   c.arrayLayers = 1;
   c.mipLevels = 1;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G18G, 0, &c));
   /* Slices rendered through 2D array or 2D views keep the 3D layout. */
   c.flags = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT |
             VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c.flags = 0;
   c.mipLevels = 2;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c.mipLevels = 1;
   c.extent.depth = 0;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   /* A 2D image keeps one depth slice. */
   c = i;
   c.extent.depth = 2;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   /* A transfer destination is made renderable like an attachment; an
    * image that is neither is not compressed. */
   c = i;
   c.usage &= ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c.usage &= ~VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   /* 16-byte formats compress as single-sample, single-layer 2D only (a
    * spilled one is decompressed in place by libagx_agx3_decompress). */
   static const VkFormat sixteen_byte[] = {
      VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT,
      VK_FORMAT_R32G32B32A32_SINT,
   };
   for (unsigned f = 0; f < sizeof(sixteen_byte) / sizeof(sixteen_byte[0]);
        ++f) {
      c = i;
      c.format = sixteen_byte[f];
      CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
      for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
         CHECK(hk_m5_color_compression_eligible((enum agx_chip)chip, 0, &c) ==
               agx_chip_has_cap((enum agx_chip)chip,
                                AGX_CAP_GEN10_2D_COMPRESSION));
      c.mipLevels = 1;
      CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c.arrayLayers = 2;
      CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c.arrayLayers = 6;
      c.extent.height = c.extent.width;
      c.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c.flags = 0;
      c.arrayLayers = 1;
      c.imageType = VK_IMAGE_TYPE_3D;
      c.extent.depth = 4;
      CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c = i;
      c.format = sixteen_byte[f];
      c.mipLevels = 1;
      c.samples = VK_SAMPLE_COUNT_2_BIT;
      CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   }
   /* Twelve-byte RGB32 texels are not a compression format. */
   c = i;
   c.format = VK_FORMAT_R32G32B32_SFLOAT;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i;
   c.samples = VK_SAMPLE_COUNT_4_BIT;
   c.mipLevels = 1;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   /* One-byte texels compress both single-level and mipmapped under M5's
    * 2 KiB compressed-level layout alignment. */
   static const VkFormat one_byte[] = {
      VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT,
   };
   for (unsigned f = 0; f < sizeof(one_byte) / sizeof(one_byte[0]); ++f) {
      c = i;
      c.format = one_byte[f];
      CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
      CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G18G, 0, &c));
      c.arrayLayers = 4;
      CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c.arrayLayers = 1;
      c.mipLevels = 1;
      CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   }
   c = i;
   c.format = VK_FORMAT_R8G8_UNORM;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   /* B10G11R11_UFLOAT_PACK32 stays plain on M5 (compressed tiles lose NaN). */
   c = i;
   c.mipLevels = 1;
   c.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));

   /* A placed alias of identical parameters gets the same layout on every M5
    * part. A single-plane alias of a multi-planar plane needs DISJOINT, which
    * stays refused, as do sparse bindings. */
   c = i;
   c.flags = VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
      CHECK(hk_m5_color_compression_eligible((enum agx_chip)chip, 0, &c) ==
            agx_chip_has_cap((enum agx_chip)chip,
                             AGX_CAP_GEN10_2D_COMPRESSION));
   c.format = VK_FORMAT_R8_UNORM;
   c.mipLevels = 1;
   c.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17G, 0, &c));
   c.flags |= VK_IMAGE_CREATE_DISJOINT_BIT;
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
      CHECK(!hk_m5_color_compression_eligible((enum agx_chip)chip, 0, &c));
   c.flags = VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_SPARSE_BINDING_BIT;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17C, 0, &c));
}

static void
check_m5_depth_msaa(void)
{
   _Static_assert(DRM_ASAHI_FEATURE_M5_DEPTH_MSAA_COMPRESSION ==
                     (UINT64_C(1) << 23), "depth capability allocation");
   const uint64_t capability = DRM_ASAHI_FEATURE_M5_DEPTH_MSAA_COMPRESSION;
   const uint64_t feature_sets[] = {0, UINT64_C(1) << 15,
                                  UINT64_C(1) << 16, capability, UINT64_MAX};
   const VkFormat formats[] = {VK_FORMAT_D32_SFLOAT,
                               VK_FORMAT_D32_SFLOAT_S8_UINT};
   const VkImageUsageFlags uses[] = {
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
         VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
   };
   for (unsigned f = 0; f < 2; ++f) {
      for (unsigned n = 2; n <= 4; n *= 2) {
         for (unsigned u = 0; u < 3; ++u) {
            VkImageCreateInfo i = image(formats[f], 3024, 1964, 1, uses[u]);
            i.samples = n;
            for (unsigned c = 0; c < 5; ++c) {
               const bool capable = (feature_sets[c] & capability) != 0;
               CHECK(hk_m5_depth_msaa_compression_eligible(
                        AGX_CHIP_G17S, 0, &i, feature_sets[c]) ==
                     (capable && u < 2));
               CHECK(hk_m5_depth_msaa_transfer_eligible(
                        AGX_CHIP_G17S, 0, &i, feature_sets[c]) ==
                     (capable && u == 2));
            }
         }
      }
   }

   VkImageCreateInfo base = image(VK_FORMAT_D32_SFLOAT_S8_UINT, 17, 19, 1,
                                 uses[2]);
   base.samples = VK_SAMPLE_COUNT_4_BIT;
   for (unsigned bad = 0; bad < 15; ++bad) {
      VkImageCreateInfo i = base;
      unsigned plane = 0;
      switch (bad) {
      case 0: i.samples = VK_SAMPLE_COUNT_1_BIT; break;
      case 1: i.samples = VK_SAMPLE_COUNT_8_BIT; break;
      case 2: i.format = VK_FORMAT_D16_UNORM; break;
      case 3: i.extent.width = 15; break;
      case 4: i.extent.height = 15; break;
      case 5: i.extent.depth = 2; break;
      case 6: i.arrayLayers = 2; break;
      case 7: i.mipLevels = 2; break;
      case 8: i.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT; break;
      case 9: i.flags = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT; break;
      case 10: i.usage |= VK_IMAGE_USAGE_STORAGE_BIT; break;
      case 11: i.usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT; break;
      case 12: i.tiling = VK_IMAGE_TILING_LINEAR; break;
      case 13: i.imageType = VK_IMAGE_TYPE_3D; break;
      case 14: plane = 1; break;
      }
      CHECK(!hk_m5_depth_msaa_compression_eligible(AGX_CHIP_G17S, plane,
                                                 &i, capability));
      CHECK(!hk_m5_depth_msaa_transfer_eligible(AGX_CHIP_G17S, plane, &i,
                                              capability));
   }
   VkImageStencilUsageCreateInfo stencil = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO,
      .stencilUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT,
   };
   base.pNext = &stencil;
   CHECK(hk_m5_depth_msaa_transfer_eligible(AGX_CHIP_G17S, 0, &base,
                                           capability));
   stencil.stencilUsage |= VK_IMAGE_USAGE_SAMPLED_BIT;
   CHECK(!hk_m5_depth_msaa_transfer_eligible(AGX_CHIP_G17S, 0, &base,
                                            capability));
   CHECK(!hk_m5_depth_msaa_transfer_eligible(AGX_CHIP_G17S, 0, NULL,
                                            capability));
}

static void
check_m5_msaa_color(void)
{
   const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   VkImageCreateInfo i = image(VK_FORMAT_R16G16B16A16_SFLOAT, 3024, 1964, 1,
                               usage);
   static const enum agx_chip m5_chips[] = {
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };
   for (unsigned m = 0; m < sizeof(m5_chips) / sizeof(m5_chips[0]); ++m) {
      for (unsigned samples = 1; samples <= 8; samples *= 2) {
         i.samples = (VkSampleCountFlagBits)samples;
         CHECK(hk_m5_msaa_color_compression_eligible(m5_chips[m], 0, &i) ==
               (samples == 2 || samples == 4));
      }
   }
   i.samples = VK_SAMPLE_COUNT_4_BIT;
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; chip++)
      CHECK(hk_m5_msaa_color_compression_eligible(
               (enum agx_chip)chip, 0, &i) ==
            agx_chip_g17_m5((enum agx_chip)chip));
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 1, &i));
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, NULL));
   VkImageCreateInfo c = i;
   c.pNext = &i;
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   /* A mutable format is admitted here; hk_can_compress() then requires a
    * format list of compatible views. Aliasing is not. */
   c = i; c.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
   CHECK(hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i; c.flags = VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
   CHECK(hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i; c.usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
   CHECK(hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i; c.flags = VK_IMAGE_CREATE_ALIAS_BIT;
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i; c.arrayLayers = 2;
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i; c.mipLevels = 2;
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i; c.format = VK_FORMAT_R8G8B8A8_UNORM;
   CHECK(hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i; c.format = VK_FORMAT_B8G8R8A8_SRGB;
   CHECK(hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i; c.format = VK_FORMAT_R32G32B32A32_SFLOAT;
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i; c.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i; c.usage &= ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i; c.extent.width = 15;
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i; c.imageType = VK_IMAGE_TYPE_3D;
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
}

static void
check_g16g_hal200_compression(void)
{
   const VkImageUsageFlags depth_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   const VkImageUsageFlags color_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   const VkImageUsageFlags texture_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
   static const VkFormat depth_formats[] = {
      VK_FORMAT_D32_SFLOAT,
      VK_FORMAT_D16_UNORM,
      VK_FORMAT_D32_SFLOAT_S8_UINT,
   };
   static const VkFormat color_16b_formats[] = {
      VK_FORMAT_R32G32B32A32_SFLOAT,
      VK_FORMAT_R32G32B32A32_UINT,
      VK_FORMAT_R32G32B32A32_SINT,
   };
   static const VkFormat mrt4_formats[] = {
      VK_FORMAT_R8G8B8A8_UNORM,
      VK_FORMAT_R8G8B8A8_SRGB,
      VK_FORMAT_R16G16B16A16_SFLOAT,
      VK_FORMAT_A2B10G10R10_UNORM_PACK32,
      VK_FORMAT_R32_SFLOAT,
      VK_FORMAT_R8_UNORM,
   };

   /* Full format sweep on G16G (M4) matches G17P: all 46 1-8B colour formats,
    * all 3 16B colour formats, and all 3 depth formats compress; R11G11B10F
    * and standalone S8 stay plain. */
   for (unsigned f = 1; f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; ++f) {
      const VkFormat fmt = (VkFormat)f;
      const bool expect = hk_g17p_compression_color_format(fmt) ||
                          hk_g17p_compression_color_format_16b(fmt) ||
                          hk_g17p_compression_depth_format(fmt);
      VkImageCreateInfo i = image(fmt, 64, 64, 1, color_usage);
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &i) ==
            expect);
      CHECK(hk_g17p_compression_eligible(AGX_CHIP_G16G, 0, &i) == expect);
      /* Non-HAL200 native AGX3 (M3 G15*), M5 (HAL300), G18G, and AGX2 never
       * enter the HAL200 policy. */
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G15G, 0, 0, &i));
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G15S, 0, 0, &i));
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G15C, 0, 0, &i));
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G17G, 0, 0, &i));
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G17S, 0, 0, &i));
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G17C, 0, 0, &i));
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G18G, 0, 0, &i));
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G14G, 0, 0, &i));
   }

   /* Depth (1x and 2x/4x MSAA, single-layer and single-level arrays). */
   for (unsigned f = 0; f < sizeof(depth_formats) / sizeof(depth_formats[0]); ++f) {
      VkImageCreateInfo d = image(depth_formats[f], 1920, 1080, 1, depth_usage);
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &d));
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 1, &d) ==
            (depth_formats[f] == VK_FORMAT_D32_SFLOAT_S8_UINT));
      /* Single-level depth arrays (shadow maps / cascades). */
      VkImageCreateInfo arr = d;
      arr.arrayLayers = 4;
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &arr));
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 1, &arr) ==
            (depth_formats[f] == VK_FORMAT_D32_SFLOAT_S8_UINT));
      arr.mipLevels = 2;
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &arr));

      for (unsigned s = 2; s <= 4; s <<= 1) {
         d.samples = (VkSampleCountFlagBits)s;
         CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &d));
         /* Non-HAL200 native AGX3 (G15) stays plain. */
         CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G15G, 0, 0, &d));
         /* Stencil plane stays plain at 2x/4x MSAA. */
         CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 1, &d));

         VkImageCreateInfo x = d;
         x.usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
         CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &x));
         /* Zink's feedback-loop usage and sample-location flag on every
          * depth attachment leave the depth plane's storage unchanged. */
         x = d;
         x.usage |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
         CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &x));
         x.flags |= VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;
         CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &x));
         CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 1, &x));
         x = d;
         x.arrayLayers = 4;
         CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &x));
         CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 1, &x));
         x = d;
         x.mipLevels = 2;
         CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &x));
      }

      d.samples = VK_SAMPLE_COUNT_8_BIT;
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &d));
   }

   /* 16-byte colour targets (rgba32f / rgba32u / rgba32i): single-sample 2D,
    * mip chains, 2D arrays, and cubemaps compress on G16G; MSAA and 3D stay
    * plain (matching G17P). */
   for (unsigned f = 0;
        f < sizeof(color_16b_formats) / sizeof(color_16b_formats[0]); ++f) {
      VkImageCreateInfo c16 =
         image(color_16b_formats[f], 1920, 1080, 1, color_usage);
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &c16));
      c16.mipLevels = 8;
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &c16));
      c16.arrayLayers = 4;
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &c16));
      VkImageCreateInfo cube =
         image(color_16b_formats[f], 256, 256, 4, color_usage);
      cube.arrayLayers = 6;
      cube.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &cube));

      VkImageCreateInfo bad =
         image(color_16b_formats[f], 1920, 1080, 1, color_usage);
      bad.samples = VK_SAMPLE_COUNT_2_BIT;
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &bad));
      bad.samples = VK_SAMPLE_COUNT_4_BIT;
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &bad));
      bad = image(color_16b_formats[f], 64, 64, 1, color_usage);
      bad.imageType = VK_IMAGE_TYPE_3D;
      bad.extent.depth = 4;
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &bad));
   }

   /* 1-8 byte MRT/colour targets: 1x, 2x/4x MSAA, layered, cube, 3D, and input
    * attachments compress across native HAL200 (G16G and G17P); R11G11B10F and
    * storage stay plain on both. */
   for (unsigned f = 0; f < sizeof(mrt4_formats) / sizeof(mrt4_formats[0]); ++f) {
      VkImageCreateInfo c = image(mrt4_formats[f], 1920, 1080, 1, color_usage);
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &c));
      c.usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &c));
      c.samples = VK_SAMPLE_COUNT_2_BIT;
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &c));
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G17P, AGX_CHIP_ID_A18_PRO,
                                           0, &c));
      c.samples = VK_SAMPLE_COUNT_4_BIT;
      c.arrayLayers = 4;
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &c));
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G17P, AGX_CHIP_ID_A18_PRO,
                                           0, &c));
      c.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &c));
      CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G17P, AGX_CHIP_ID_A18_PRO,
                                            0, &c));

      VkImageCreateInfo vol = image(mrt4_formats[f], 64, 64, 1, color_usage);
      vol.imageType = VK_IMAGE_TYPE_3D;
      vol.extent.depth = 8;
      CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &vol));
   }
   VkImageCreateInfo r11 =
      image(VK_FORMAT_B10G11R11_UFLOAT_PACK32, 1920, 1080, 1, color_usage);
   CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &r11));

   /* Classic-submission non-attachment crossover on G16G: textures never used as
    * attachments stay plain below HK_CLASSIC_MIP_PBE_MIN_TEXELS (1 << 20 =
    * 1024x1024) and compress at or above it; attachments compress at any size
    * >= 16x16. */
   VkImageCreateInfo tex = image(VK_FORMAT_R8G8B8A8_UNORM, 512, 256, 1, texture_usage);
   CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &tex));
   CHECK(hk_hal200_compression_eligible(AGX_CHIP_G17P, AGX_CHIP_ID_A18_PRO, 0, &tex));
   tex.extent.width = 768;
   tex.extent.height = 768;
   CHECK(!hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &tex));
   tex.extent.width = 1024;
   tex.extent.height = 1024;
   CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &tex));
   tex = image(VK_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, color_usage);
   CHECK(hk_hal200_compression_eligible(AGX_CHIP_G16G, 0x8132, 0, &tex));

   /* Shared DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED contract on G16G. */
   VkImageCreateInfo shared =
      swapchain_image(VK_FORMAT_B8G8R8A8_SRGB, 1920, 1080,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
   CHECK(hk_g17p_shared_compression_eligible(AGX_CHIP_G16G, 0, &shared));
   shared.format = VK_FORMAT_R32G32B32A32_SFLOAT;
   CHECK(!hk_g17p_shared_compression_eligible(AGX_CHIP_G16G, 0, &shared));
}

int
main(void)
{
   check_formats();
   check_shape(VK_FORMAT_R8G8B8A8_UNORM, false);
   check_shape(VK_FORMAT_B8G8R8A8_SRGB, false);
   check_shape(VK_FORMAT_R16G16B16A16_SFLOAT, false);
   check_shape_for_chip(VK_FORMAT_R32G32B32A32_SFLOAT, false,
                        AGX_CHIP_ID_A18_PRO);
   check_shape_for_chip(VK_FORMAT_R32G32B32A32_UINT, false,
                        AGX_CHIP_ID_A18_PRO);
   check_shape_for_chip(VK_FORMAT_R32G32B32A32_SINT, false,
                        AGX_CHIP_ID_A18_PRO);
   check_shape(VK_FORMAT_R8_UNORM, false);
   check_shape(VK_FORMAT_D32_SFLOAT, true);
   check_shape(VK_FORMAT_D16_UNORM, true);
   check_shape(VK_FORMAT_D32_SFLOAT_S8_UINT, true);
   check_minecraft();
   check_zink();
   check_shared();
   check_g16g_hal200_compression();
   check_m5_depth();
   check_m5_color();
   check_m5_msaa_color();
   check_m5_depth_msaa();
   printf("HAL200/M5 compression policy: %u checks passed\n", checks);
   return 0;
}

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
   unsigned color = 0, depth = 0;
   for (unsigned f = 1; f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; ++f) {
      const bool c = hk_g17p_compression_color_format((VkFormat)f);
      const bool d = hk_g17p_compression_depth_format((VkFormat)f);
      CHECK(!(c && d));
      color += c;
      depth += d;

      VkImageCreateInfo i = image((VkFormat)f, 64, 64, 1, color_usage);
      CHECK(ELIGIBLE(i) == (c || d));
   }
   CHECK(color == 46);
   CHECK(depth == 3);

   /* No 16-byte texels (tilebuffer spill), no packed/padded RGB, no stencil
    * or combined depth/stencil, no block-compressed or YCbCr formats. */
   static const VkFormat rejected[] = {
      VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT,
      VK_FORMAT_R32G32B32A32_SINT,   VK_FORMAT_R32G32B32_SFLOAT,
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
check_shape(VkFormat format, bool is_depth)
{
   const VkImageUsageFlags usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT |
      (is_depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
   const VkImageCreateInfo base = image(format, 2408, 1508, 1, usage);
   CHECK(ELIGIBLE(base));

   /* Chip and plane. */
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
      CHECK(hk_g17p_compression_eligible((enum agx_chip)chip, 0, &base) ==
            (chip == AGX_CHIP_G17P));
   /* Plane 1 is the stencil plane of D32_SFLOAT_S8_UINT alone. */
   CHECK(hk_g17p_compression_eligible(AGX_CHIP_G17P, 1, &base) ==
         (format == VK_FORMAT_D32_SFLOAT_S8_UINT));
   CHECK(!hk_g17p_compression_eligible(AGX_CHIP_G17P, 2, &base));
   CHECK(!hk_g17p_compression_eligible(AGX_CHIP_G17P, 0, NULL));
   CHECK(!hk_g17p_compression_eligible(AGX_CHIP_G17P, 1, NULL));

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
      CHECK(ELIGIBLE(i) == !!((1u << bit) & allowed));
   }
   for (unsigned usage_bits = 0; usage_bits < 256; ++usage_bits) {
      VkImageCreateInfo i = base;
      i.usage = usage_bits & allowed;
      CHECK(ELIGIBLE(i));
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
      CHECK(ELIGIBLE(i) == (is_depth && refused[r] ==
                           VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT));
   }

   /* Sample-location tracking is also admitted for single-sample depth.
    * Aliasing is admitted for every shape; storage stays refused with it. */
   for (unsigned bit = 0; bit < 32; ++bit) {
      VkImageCreateInfo i = base;
      i.flags = 1u << bit;
      CHECK(ELIGIBLE(i) ==
            ((1u << bit) == VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT ||
             (1u << bit) == VK_IMAGE_CREATE_EXTENDED_USAGE_BIT ||
             (1u << bit) == VK_IMAGE_CREATE_ALIAS_BIT ||
             (is_depth && (1u << bit) ==
                          VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT)));
   }
   {
      VkImageCreateInfo i = base;
      i.flags = VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
      CHECK(ELIGIBLE(i));
      i.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
      CHECK(!ELIGIBLE(i));
   }

   /* Geometry. */
   VkImageCreateInfo i = base;
   i.imageType = VK_IMAGE_TYPE_1D;
   CHECK(!ELIGIBLE(i));
   i = base; i.imageType = VK_IMAGE_TYPE_3D;
   CHECK(!ELIGIBLE(i));
   i = base; i.extent.depth = 2;
   CHECK(!ELIGIBLE(i));
   i = base; i.arrayLayers = 2;
   CHECK(!ELIGIBLE(i));
   i = base; i.arrayLayers = 6;
   CHECK(!ELIGIBLE(i));
   for (unsigned s = 2; s <= 64; s <<= 1) {
      i = base; i.samples = (VkSampleCountFlagBits)s;
      CHECK(!ELIGIBLE(i));
   }
   i = base; i.tiling = VK_IMAGE_TILING_LINEAR;
   CHECK(!ELIGIBLE(i));
   i = base; i.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
   CHECK(!ELIGIBLE(i));
   i = base; i.mipLevels = 0;
   CHECK(!ELIGIBLE(i));
   i = base; i.mipLevels = 12;
   CHECK(ELIGIBLE(i) == !is_depth);
   i = base; i.extent.width = 15;
   CHECK(!ELIGIBLE(i));
   i = base; i.extent.height = 15;
   CHECK(!ELIGIBLE(i));
   i = base; i.extent.width = 16; i.extent.height = 16;
   CHECK(ELIGIBLE(i));
   i = base; i.extent.width = 17; i.extent.height = 31;
   CHECK(ELIGIBLE(i));
   i = base; i.extent.width = 16384; i.extent.height = 16384;
   CHECK(ELIGIBLE(i));
   i = base; i.sharingMode = VK_SHARING_MODE_CONCURRENT;
   CHECK(ELIGIBLE(i));

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
   CHECK(!ELIGIBLE(i));
   i = base; i.pNext = &list;
   CHECK(!ELIGIBLE(i));
   external.handleTypes = 0;
   CHECK(ELIGIBLE(i));
   list.pNext = NULL;
   CHECK(ELIGIBLE(i));
   VkBaseInStructure wsi = {.sType = HK_G17P_WSI_IMAGE_CREATE_INFO_MESA};
   i = base; i.pNext = &wsi;
   CHECK(!ELIGIBLE(i));
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
   depth.samples = VK_SAMPLE_COUNT_4_BIT;
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

   /* G17P and M5 only, plane 0 only. */
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
      CHECK(hk_g17p_shared_compression_eligible((enum agx_chip)chip, 0,
                                                &base) ==
            (chip == AGX_CHIP_G17P || chip == AGX_CHIP_G17G ||
             chip == AGX_CHIP_G17S || chip == AGX_CHIP_G17C));
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
      c = i;
      c.arrayLayers = 2;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c = i;
      c.flags |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT;
      CHECK(!hk_m5_depth_compression_eligible(AGX_CHIP_G17S, 0, &c));
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
            agx_chip_g17_m5((enum agx_chip)chip));
   i.mipLevels = 1;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &i));
   i.mipLevels = 12;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &i));
   VkImageCreateInfo c = i;
   c.arrayLayers = 2;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   /* A transfer destination is made renderable like an attachment; an
    * image that is neither is not compressed. */
   c = i;
   c.usage &= ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c.usage &= ~VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i;
   c.format = VK_FORMAT_R32G32B32A32_SFLOAT;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   c = i;
   c.samples = VK_SAMPLE_COUNT_4_BIT;
   c.mipLevels = 1;
   CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   /* One-byte texels compress only as a single level. */
   static const VkFormat one_byte[] = {
      VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT,
   };
   for (unsigned f = 0; f < sizeof(one_byte) / sizeof(one_byte[0]); ++f) {
      c = i;
      c.format = one_byte[f];
      CHECK(!hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
      c.mipLevels = 1;
      CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   }
   c = i;
   c.format = VK_FORMAT_R8G8_UNORM;
   CHECK(hk_m5_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
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
   for (unsigned samples = 1; samples <= 8; samples *= 2) {
      i.samples = (VkSampleCountFlagBits)samples;
      CHECK(hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &i) ==
            (samples == 2 || samples == 4));
   }
   i.samples = VK_SAMPLE_COUNT_4_BIT;
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; chip++)
      CHECK(hk_m5_msaa_color_compression_eligible(
               (enum agx_chip)chip, 0, &i) == (chip == AGX_CHIP_G17S));
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 1, &i));
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, NULL));
   VkImageCreateInfo c = i;
   c.pNext = &i;
   CHECK(!hk_m5_msaa_color_compression_eligible(AGX_CHIP_G17S, 0, &c));
   /* A mutable format is admitted here; hk_can_compress() then requires a
    * format list of compatible views. Aliasing is not. */
   c = i; c.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
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

int
main(void)
{
   check_formats();
   check_shape(VK_FORMAT_R8G8B8A8_UNORM, false);
   check_shape(VK_FORMAT_B8G8R8A8_SRGB, false);
   check_shape(VK_FORMAT_R16G16B16A16_SFLOAT, false);
   check_shape(VK_FORMAT_R8_UNORM, false);
   check_shape(VK_FORMAT_D32_SFLOAT, true);
   check_shape(VK_FORMAT_D16_UNORM, true);
   check_shape(VK_FORMAT_D32_SFLOAT_S8_UINT, true);
   check_minecraft();
   check_zink();
   check_shared();
   check_m5_depth();
   check_m5_color();
   check_m5_msaa_color();
   check_m5_depth_msaa();
   printf("G17P compression policy: %u checks passed\n", checks);
   return 0;
}

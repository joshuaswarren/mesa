/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT
 *
 * Pure checks of the G17P lossless-compression admission policy
 * (hk_g17p_compression.h). The generic hk_can_compress() checks are applied
 * separately by the driver.
 */
#include "asahi/vulkan/hk_g17p_compression.h"
#include <stdio.h>
#include <stdlib.h>

/* The physical T6040 selector in hk_image.c supplies this capability.
 * The pre-existing test-hk-g17p-compression separately pins false/Neo. */
static bool
g16s_eligible(enum agx_chip chip, unsigned plane, const VkImageCreateInfo *info)
{
   return hk_hal200_compression_eligible(chip, 0x6040, plane, info, true);
}

static unsigned checks;
#define CHECK(x)                                                              \
   do {                                                                       \
      checks++;                                                               \
      if (!(x)) {                                                             \
         fprintf(stderr, "line %u: %s\n", __LINE__, #x);                      \
         exit(1);                                                             \
      }                                                                       \
   } while (0)

/* Exercise mutable extension chains through a runtime call, as the driver
 * does, rather than letting the compiler fold the whole stack fixture. */
static bool (*volatile runtime_eligible)(enum agx_chip, unsigned,
                                         const VkImageCreateInfo *) =
   g16s_eligible;
#define ELIGIBLE(i) runtime_eligible(AGX_CHIP_G17P, 0, &(i))

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
      const bool d = hk_g17p_compression_depth_format((VkFormat)f) ||
                     f == VK_FORMAT_S8_UINT;
      CHECK(!(c && d));
      color += c;
      depth += d;

      VkImageCreateInfo i = image((VkFormat)f, 64, 64, 1, color_usage);
      CHECK(ELIGIBLE(i) == (c || hk_g17p_compression_depth_format((VkFormat)f)));
   }
   CHECK(color == 46);
   CHECK(depth == 4);

   /* No 16-byte texels (tilebuffer spill), no packed/padded RGB, no packed
    * depth/stencil, no block-compressed or YCbCr formats. */
   static const VkFormat rejected[] = {
      VK_FORMAT_R32G32B32A32_SFLOAT,
      VK_FORMAT_R32G32B32A32_UINT,
      VK_FORMAT_R32G32B32A32_SINT,
      VK_FORMAT_R32G32B32_SFLOAT,
      VK_FORMAT_R8G8B8_UNORM,
      VK_FORMAT_R16G16B16_SFLOAT,
      VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
      VK_FORMAT_R4G4B4A4_UNORM_PACK16,
      VK_FORMAT_A1R5G5B5_UNORM_PACK16,
      VK_FORMAT_D24_UNORM_S8_UINT,
      VK_FORMAT_D16_UNORM_S8_UINT,
      VK_FORMAT_X8_D24_UNORM_PACK32,
      VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
      VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
      VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
      VK_FORMAT_G8B8G8R8_422_UNORM,
      VK_FORMAT_R64_UINT,
   };
   for (unsigned r = 0; r < sizeof(rejected) / sizeof(rejected[0]); ++r) {
      VkImageCreateInfo i = image(rejected[r], 64, 64, 1, color_usage);
      CHECK(!ELIGIBLE(i));
   }
}

static void
check_shape(VkFormat format, bool is_zs)
{
   const bool is_depth = hk_g17p_compression_depth_format(format);
   const VkImageUsageFlags usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT |
      (is_zs ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
             : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
   const VkImageCreateInfo base = image(format, 2408, 1508, 1, usage);
   CHECK(ELIGIBLE(base));

   /* Chip and plane. */
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
      CHECK(g16s_eligible((enum agx_chip)chip, 0, &base) ==
            (chip == AGX_CHIP_G17P));
   /* Plane 1 is the stencil plane of D32_SFLOAT_S8_UINT alone. */
   CHECK(g16s_eligible(AGX_CHIP_G17P, 1, &base) ==
         (format == VK_FORMAT_D32_SFLOAT_S8_UINT));
   CHECK(!g16s_eligible(AGX_CHIP_G17P, 2, &base));
   CHECK(!g16s_eligible(AGX_CHIP_G17P, 0, NULL));
   CHECK(!g16s_eligible(AGX_CHIP_G17P, 1, NULL));

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
   CHECK(ELIGIBLE(i) == !is_zs);
   i = base; i.extent.depth = 2;
   CHECK(!ELIGIBLE(i));
   i = base; i.arrayLayers = 2;
   CHECK(ELIGIBLE(i));
   i = base; i.arrayLayers = 6;
   CHECK(ELIGIBLE(i));
   /* Of the multisample colour attachments, only 2x (one layer) on M4 Pro. */
   for (unsigned s = 2; s <= 64; s <<= 1) {
      i = base; i.samples = (VkSampleCountFlagBits)s;
      CHECK(ELIGIBLE(i) == (s == 2));
      i.arrayLayers = 2;
      CHECK(!ELIGIBLE(i));
   }
   i = base; i.tiling = VK_IMAGE_TILING_LINEAR;
   CHECK(!ELIGIBLE(i));
   i = base; i.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
   CHECK(!ELIGIBLE(i));
   i = base; i.mipLevels = 0;
   CHECK(!ELIGIBLE(i));
   i = base; i.mipLevels = 12;
   CHECK(ELIGIBLE(i));
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

/* Compression is chosen for the image's whole lifetime. Its first render
 * pass, attachment slot and neighbouring formats are unknown at allocation.
 * Cover every admitted colour format, including the smallest ones that can
 * spill next to wide targets, and input/transient attachment combinations.
 * A 2x single-layer attachment is admitted: allocated first, every target
 * of at most eight bytes fits the 64-byte 2x sample (hk_resident_attachments).
 */
static void
check_colour_msaa_spill_policy(void)
{
   for (unsigned f = 1; f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; ++f) {
      if (!hk_g17p_compression_color_format((VkFormat)f))
         continue;

      for (unsigned samples = 1; samples <= 4; samples *= 2) {
         for (unsigned layers = 1; layers <= 4; layers *= 4) {
            /* 512x512: above the size below which a never-attachment image
             * stays plain (HK_G17P_MIP_PBE_MIN_TEXELS). */
            VkImageCreateInfo i = image((VkFormat)f, 512, 512, 1,
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            i.samples = (VkSampleCountFlagBits)samples;
            i.arrayLayers = layers;
            CHECK(ELIGIBLE(i));
            for (unsigned mask = 0; mask < 4; ++mask) {
               VkImageCreateInfo attachment = i;
               attachment.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
               if (mask & 1)
                  attachment.usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
               if (mask & 2)
                  attachment.usage |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
               const bool expected =
                  samples == 1 || (samples == 2 && layers == 1 && !(mask & 1));
               CHECK(ELIGIBLE(attachment) == expected);
               attachment.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
                                  VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
               CHECK(ELIGIBLE(attachment) == expected);
            }
         }
      }
   }
}

static void
check_layered_and_msaa(void)
{
   /* Large enough that the small never-attachment rule does not apply. */
   VkImageCreateInfo i =
      image(VK_FORMAT_R8G8B8A8_SRGB, 512, 512, 1,
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT);
   i.arrayLayers = 12;
   i.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
   CHECK(ELIGIBLE(i));
   VkImageCreateInfo small = i;
   small.extent.width = small.extent.height = 256;
   CHECK(!ELIGIBLE(small));
   i.mipLevels = 5;
   CHECK(!ELIGIBLE(i));
   i.mipLevels = 1;
   i.flags = 0;
   for (unsigned samples = 1; samples <= 64; samples <<= 1) {
      i.samples = (VkSampleCountFlagBits)samples;
      CHECK(ELIGIBLE(i) == (samples <= 4));
      i.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      CHECK(ELIGIBLE(i) == (samples == 1));
      i.usage &= ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      i.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
      CHECK(!ELIGIBLE(i));
      i.usage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
   }
   i.samples = VK_SAMPLE_COUNT_1_BIT;
   i.arrayLayers = 1;
   i.imageType = VK_IMAGE_TYPE_3D;
   i.extent.depth = 9;
   i.flags = VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT |
             VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT;
   CHECK(ELIGIBLE(i));
   i.mipLevels = 5;
   CHECK(!ELIGIBLE(i));
   i.mipLevels = 1;
   i.samples = VK_SAMPLE_COUNT_2_BIT;
   CHECK(!ELIGIBLE(i));
   i.samples = VK_SAMPLE_COUNT_1_BIT;
   i.arrayLayers = 2;
   CHECK(!ELIGIBLE(i));
   i.arrayLayers = 1;
   i.format = VK_FORMAT_D32_SFLOAT;
   CHECK(!ELIGIBLE(i));


}

/* Preserve the pre-port depth/stencil policy across the shapes observed in
 * depth sampling, feedback loops, host copies and transfer clears. Allocation
 * tests alone did not catch the unsupported colour-alias command paths.
 * The exceptions are the depth plane at 2x (one layer), without
 * feedback-loop use, and single-level single-sample arrays. */
static void
check_legacy_depth_policy(void)
{
   const VkFormat formats[] = { VK_FORMAT_D16_UNORM, VK_FORMAT_D32_SFLOAT,
      VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_S8_UINT };
   const VkImageUsageFlags usages[] = {
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
         VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT,
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT,
   };
   for (unsigned f = 0; f < 4; ++f) {
      for (unsigned u = 0; u < 4; ++u) {
         for (unsigned levels = 1; levels <= 8; levels *= 2) {
            for (unsigned layers = 1; layers <= 6; ++layers) {
               for (unsigned samples = 1; samples <= 4; samples *= 2) {
                  VkImageCreateInfo i = image(formats[f], 128, 128, levels,
                                               usages[u]);
                  i.arrayLayers = layers;
                  i.samples = (VkSampleCountFlagBits)samples;
                  const bool expected =
                     f < 3 && u < 3 && levels == 1 &&
                     (samples == 1 ||
                      (samples == 2 && layers == 1 && u != 1));
                  CHECK(ELIGIBLE(i) == expected);
                  CHECK(g16s_eligible(AGX_CHIP_G17P, 1, &i) ==
                        (expected && f == 2 && samples == 1));
               }
            }
         }
      }
   }
}

/* Kyty's guest depth targets: shadow-map cascades rendered one layer per
 * pass, with transfer, sampled and (when enabled) feedback-loop usage. */
static void
check_depth_arrays(void)
{
   const VkImageUsageFlags kyty = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   const VkFormat formats[] = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM,
                               VK_FORMAT_D32_SFLOAT_S8_UINT};
   for (unsigned f = 0; f < 3; ++f) {
      VkImageCreateInfo i = image(formats[f], 1536, 1536, 1, kyty);
      i.arrayLayers = 4;
      CHECK(ELIGIBLE(i));
      CHECK(g16s_eligible(AGX_CHIP_G17P, 1, &i) == (f == 2));
      /* The Neo policy keeps one layer. */
      CHECK(!hk_g17p_compression_eligible(AGX_CHIP_G17P, 0, &i));
      i.usage |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
      CHECK(ELIGIBLE(i));
      i.flags = VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;
      CHECK(ELIGIBLE(i));
      i.arrayLayers = 2048;
      CHECK(ELIGIBLE(i));

      /* Not cube maps, mip chains, multisampled arrays or volumes. */
      VkImageCreateInfo c = i;
      c.arrayLayers = 6;
      c.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      CHECK(!ELIGIBLE(c));
      c = i; c.mipLevels = 2;
      CHECK(!ELIGIBLE(c));
      c = i; c.samples = VK_SAMPLE_COUNT_2_BIT;
      c.usage &= ~VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
      c.flags = 0;
      CHECK(!ELIGIBLE(c));
      c.arrayLayers = 1;
      CHECK(ELIGIBLE(c)); /* the 2x depth plane, one layer */
      c = i; c.arrayLayers = 0;
      CHECK(!ELIGIBLE(c));
      c = i; c.extent.depth = 2;
      CHECK(!ELIGIBLE(c));
      c = i; c.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
      CHECK(!ELIGIBLE(c));
   }

   /* Standalone stencil stays plain. */
   VkImageCreateInfo s = image(VK_FORMAT_S8_UINT, 1536, 1536, 1, kyty);
   s.arrayLayers = 4;
   CHECK(!ELIGIBLE(s));
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
check_transport_fever_3(void)
{
   /* Transport Fever 3 at 2x anti-aliasing: its 3456x2234 G-buffer is
    * R11G11B10F, A2R10G10B10 and RGBA8 (usage 0x13) and D32S8 (0x23),
    * reloaded and stored about six times a frame. All but R11G11B10F (which
    * would lose NaN) and the stencil plane compress on M4 Pro; nothing
    * multisampled does on the other G17P chips. */
   static const struct { VkFormat format; VkImageUsageFlags usage; bool m4_pro; }
      gbuffer[] = {
         {VK_FORMAT_B10G11R11_UFLOAT_PACK32, 0x13, false},
         {VK_FORMAT_A2R10G10B10_UNORM_PACK32, 0x13, true},
         {VK_FORMAT_R8G8B8A8_UNORM, 0x13, true},
         {VK_FORMAT_D32_SFLOAT_S8_UINT, 0x23, true},
      };
   for (unsigned g = 0; g < sizeof(gbuffer) / sizeof(gbuffer[0]); ++g) {
      VkImageCreateInfo i =
         image(gbuffer[g].format, 3456, 2234, 1, gbuffer[g].usage);
      i.samples = VK_SAMPLE_COUNT_2_BIT;
      CHECK(ELIGIBLE(i) == gbuffer[g].m4_pro);
      CHECK(!g16s_eligible(AGX_CHIP_G17P, 1, &i));
      CHECK(!hk_g17p_compression_eligible(AGX_CHIP_G17P, 0, &i));
      i.samples = VK_SAMPLE_COUNT_4_BIT;
      CHECK(!ELIGIBLE(i));
   }
}

static void
check_r11g11b10f(void)
{
   const VkImageUsageFlags attachment = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

   /* A compressed tile of the R11G11B10F class reads NaN back as zero, and
    * a render or copy into the attachment must keep it: never compressed. */
   VkImageCreateInfo i =
      image(VK_FORMAT_B10G11R11_UFLOAT_PACK32, 3456, 2234, 12, attachment);
   CHECK(!ELIGIBLE(i));
   i.mipLevels = 1;
   CHECK(!ELIGIBLE(i));
   i.samples = VK_SAMPLE_COUNT_2_BIT;
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
   CHECK(g16s_eligible(AGX_CHIP_G17P, 1, &depth));
   /* This is Portal's actual shape. A device-copy performance query must
    * find its compressed layout when host transfer is removed; otherwise
    * zink retains HOST_TRANSFER and never uses depth compression. */
   depth.usage |= VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
   CHECK(!ELIGIBLE(depth));
   depth.usage &= ~VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;
   CHECK(ELIGIBLE(depth));
   depth.samples = VK_SAMPLE_COUNT_4_BIT;
   CHECK(!ELIGIBLE(depth));
   /* At 2x neither its feedback-loop usage nor its sample-location flag. */
   depth.samples = VK_SAMPLE_COUNT_2_BIT;
   CHECK(!ELIGIBLE(depth));
   depth.usage &= ~VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
   CHECK(!ELIGIBLE(depth));
   depth.flags = 0;
   CHECK(ELIGIBLE(depth));
   CHECK(!g16s_eligible(AGX_CHIP_G17P, 1, &depth));
   depth.usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
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

   /* G17P only, plane 0 only. */
   for (unsigned chip = 0; chip <= AGX_CHIP_UNKNOWN; ++chip)
      CHECK(hk_g17p_shared_compression_eligible((enum agx_chip)chip, 0,
                                                &base) ==
            (chip == AGX_CHIP_G17P || agx_chip_g17_m5((enum agx_chip)chip)));
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

/* The M4 Max (T6041) keeps a reloaded 2x sample whole
 * (agx_chip_uncovered_tile_ms_reload_defect()), so it takes the 2x colour
 * attachments and 2x depth, nothing more of the M4 Pro's shapes. The A18 Pro
 * (T8140, "Neo") does not, and keeps its 2x images plain. */
static bool
m4_max_eligible(enum agx_chip chip, unsigned plane,
                const VkImageCreateInfo *info)
{
   return hk_hal200_compression_eligible(chip, 0x6041, plane, info, false);
}

static bool
neo_eligible(enum agx_chip chip, unsigned plane, const VkImageCreateInfo *info)
{
   return hk_hal200_compression_eligible(chip, 0x8140, plane, info, false);
}

static bool (*volatile runtime_m4_max)(enum agx_chip, unsigned,
                                       const VkImageCreateInfo *) =
   m4_max_eligible;
static bool (*volatile runtime_neo)(enum agx_chip, unsigned,
                                    const VkImageCreateInfo *) = neo_eligible;
#define M4_MAX(i) runtime_m4_max(AGX_CHIP_G17P, 0, &(i))
#define NEO(i)    runtime_neo(AGX_CHIP_G17P, 0, &(i))

static void
check_m4_max(void)
{
   const VkImageUsageFlags rt =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   const VkImageUsageFlags ds = VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

   VkImageCreateInfo c = image(VK_FORMAT_R8G8B8A8_UNORM, 1920, 1080, 1, rt);
   CHECK(M4_MAX(c) && NEO(c));

   c.samples = VK_SAMPLE_COUNT_2_BIT;
   CHECK(M4_MAX(c) && !NEO(c));
   VkImageCreateInfo x = c;
   x.samples = VK_SAMPLE_COUNT_4_BIT;
   CHECK(!M4_MAX(x));
   x = c;
   x.usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
   CHECK(!M4_MAX(x));
   x = c;
   x.arrayLayers = 2;
   CHECK(!M4_MAX(x));
   x = c;
   x.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   CHECK(!M4_MAX(x));
   x = c;
   x.format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
   CHECK(!M4_MAX(x));

   /* Layered colour stays on the M4 Pro. */
   x = image(VK_FORMAT_R8G8B8A8_UNORM, 256, 256, 1, rt);
   x.arrayLayers = 6;
   CHECK(!M4_MAX(x) && g16s_eligible(AGX_CHIP_G17P, 0, &x));

   VkImageCreateInfo d = image(VK_FORMAT_D32_SFLOAT, 1920, 1080, 1, ds);
   CHECK(M4_MAX(d));
   d.samples = VK_SAMPLE_COUNT_2_BIT;
   CHECK(M4_MAX(d) && !NEO(d));
   d.samples = VK_SAMPLE_COUNT_4_BIT;
   CHECK(!M4_MAX(d));

   /* Depth arrays: M4 Pro only. */
   d = image(VK_FORMAT_D32_SFLOAT, 1024, 1024, 1, ds);
   d.arrayLayers = 4;
   CHECK(!M4_MAX(d) && g16s_eligible(AGX_CHIP_G17P, 0, &d));

   /* Other chips never. */
   CHECK(!runtime_m4_max(AGX_CHIP_G14G, 0, &c));
}

int
main(void)
{
   check_formats();
   check_transport_fever_3();
   check_r11g11b10f();
   check_shape(VK_FORMAT_R8G8B8A8_UNORM, false);
   check_shape(VK_FORMAT_B8G8R8A8_SRGB, false);
   check_shape(VK_FORMAT_R16G16B16A16_SFLOAT, false);
   check_shape(VK_FORMAT_R8_UNORM, false);
   check_legacy_depth_policy();
   check_depth_arrays();
   check_layered_and_msaa();
   check_colour_msaa_spill_policy();
   check_minecraft();
   check_zink();
   check_shared();
   check_m4_max();
   printf("G17P compression policy: %u checks passed\n", checks);
   return 0;
}

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "asahi/lib/agx_chip.h"
#include "hk_g17p_meta_copy_core.h"
#include "drm-uapi/asahi_drm.h"
#include "vulkan/vulkan_core.h"
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/* HAL200 lossless compression policy (G17P full policy, plus 2x/4x MSAA depth
 * across native HAL200).
 *
 * The compressed Gen9 PBE (write) packet matches Apple's HAL200 StateFields
 * encoder byte-for-byte for GPU-tiled one-layer 2D images, the read
 * descriptor carries the same compression bit and metadata address, and the
 * G17P ZLS control word Apple's Framebuffer builds uses the same load/store
 * compression bits (2/4/6/8), 16-bit depth format bit and metadata stride
 * encoding as Mesa's agx_pack_zls_control. The shapes below are the ones
 * whose every write path in hk goes through one of those three units or the
 * libagx_fast_clear metadata kernel:
 *
 *  - Color: single-plane, 1-8 byte texels, one layer, one sample, plus on
 *    A18 Pro (AGX_CHIP_ID_A18_PRO, measured) 16-byte texels
 *    (R32G32B32A32_{UINT,SINT,SFLOAT}), one layer, one sample. Layered and
 *    multisampled compressed PBE packets are refused by agx_pbe_agx3_core.c
 *    (no witness), so arrays/MSAA stay plain. 1-8 byte targets never spill
 *    (8 x 8 B <= 120 B/sample); a spilled single-sample 16-byte target is
 *    decompressed in-place before the render pass via
 *    libagx_g17p_decompress_16b / libagx_reset_compression_metadata.
 *  - Colour at 2x where a 2x reload keeps the whole sample
 *    (!agx_chip_uncovered_tile_ms_reload_defect(chip, chip_id, 2): M4 Pro
 *    and M4 Max): also colour attachments of one level and one layer. G17P
 *    has no MSAA/1-8 B decompression program; at 2x the tile holds 64 bytes a
 *    sample, so the eight bytes-at-most targets, allocated first
 *    (hk_resident_attachments()), always stay in it, and a reload does not
 *    make them spill.
 *    A multiview pass has an eight-byte budget, so on a device with the
 *    multiview feature 2x colour stays plain (hk_image.c). Input
 *    attachments stay plain.
 *  - Not R11G11B10F: attachments store through the format's own class
 *    (hk_eot_pbe_format()), and a compressed tile of that class reads a NaN
 *    field back as zero, however it was written. Rendering NaN into an
 *    R11G11B10F attachment has to keep it, so they stay plain.
 *  - Depth: D16_UNORM, D32_SFLOAT and both planes of D32_SFLOAT_S8_UINT,
 *    one level, one sample (single-layer or arrays) on G17P; across HAL200
 *    (G16G and G17P) also the depth plane at 2x and 4x (one level, one layer,
 *    plane 0 only). Multisampled depth goes through ZLS rather than the colour
 *    tilebuffer's background reload program, so it is unaffected by
 *    AGX_QUIRK_UNCOVERED_TILE_MS_DEFECT. The 2x/4x depth plane excludes
 *    input-attachment, feedback-loop and sample-location use (its S8 plane
 *    stays plain). The ZLS unit compresses the S8 stencil plane with the same
 *    metadata layout as depth (stencil
 *    load/store compression bits 4/8), it is read through its R8_UINT
 *    alias, and its clears are metadata fast clears. Zink stores GL Z24S8
 *    as D32S8, so these planes carry every GL depth/stencil buffer's ZLS
 *    traffic; buffer uploads into them take the graphics copy (R32_UINT
 *    and R8_UINT PBEs).
 *
 * Images that may be shared with another process or device (external
 * memory, WSI, DRM modifiers) stay plain under this private policy. Colour
 * images are shared compressed only through the GPU_TILED_COMPRESSED
 * modifier contract, see hk_g17p_shared_compression_eligible().
 *
 * Aliasing (VK_IMAGE_CREATE_ALIAS_BIT, which vkd3d-proton sets on every
 * placed resource) is admitted. The layout is a function of the creation
 * parameters, so images created alike and bound to the same memory share
 * body and metadata, as the flag requires. A newly bound image starts with
 * whatever its memory held, metadata included. That is not specific to
 * ALIAS: app memory taken from the BO cache is not zeroed either. Each write
 * path above writes whole 16x16 tiles with their metadata words, so stale
 * words only decide how tiles read that were not written since the contents
 * became undefined, and those texels are undefined anyway. The body has its
 * full uncompressed size, with the metadata after it, so no mode reaches
 * outside the image. Storage images are not admitted: shader stores leave
 * the metadata as it was (hk_image.c).
 *
 * This is the only HAL200 policy: every eligible image is compressed, with no
 * runtime opt-out.
 */

/* Colour formats admitted on G17P: plain single-plane formats of 1-8 bytes
 * whose physical G17P storage equals the API format (no RGB padding) and
 * whose channel class has a bitwise copy view. hk_can_compress() additionally
 * requires ail_can_compress() (renderable, at least 16x16). */
static inline bool
hk_g17p_compression_color_format(VkFormat format)
{
   switch (format) {
   /* 1 byte */
   case VK_FORMAT_R8_UNORM:
   case VK_FORMAT_R8_SNORM:
   case VK_FORMAT_R8_UINT:
   case VK_FORMAT_R8_SINT:
   /* 2 bytes */
   case VK_FORMAT_R8G8_UNORM:
   case VK_FORMAT_R8G8_SNORM:
   case VK_FORMAT_R8G8_UINT:
   case VK_FORMAT_R8G8_SINT:
   case VK_FORMAT_R16_UNORM:
   case VK_FORMAT_R16_SNORM:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R16_SINT:
   case VK_FORMAT_R16_SFLOAT:
   case VK_FORMAT_R5G6B5_UNORM_PACK16:
   case VK_FORMAT_B5G6R5_UNORM_PACK16:
   /* 4 bytes */
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_R8G8B8A8_SINT:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_SNORM_PACK32:
   case VK_FORMAT_A8B8G8R8_UINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SINT_PACK32:
   case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
   case VK_FORMAT_A2B10G10R10_UINT_PACK32:
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
   /* Not B10G11R11_UFLOAT: G17P packs it into one tile word, and its
    * end-of-tile programs store that word through the format's own class
    * (hk_eot_pbe_format()), which loses NaN when compressed. */
   case VK_FORMAT_R16G16_UNORM:
   case VK_FORMAT_R16G16_SNORM:
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R16G16_SINT:
   case VK_FORMAT_R16G16_SFLOAT:
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32_SINT:
   case VK_FORMAT_R32_SFLOAT:
   /* 8 bytes */
   case VK_FORMAT_R16G16B16A16_UNORM:
   case VK_FORMAT_R16G16B16A16_SNORM:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R16G16B16A16_SINT:
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32_SINT:
   case VK_FORMAT_R32G32_SFLOAT:
      return true;
   default:
      return false;
   }
}

/* 16-byte colour formats admitted on G17P for single-sample, single-layer 2D
 * private images. Kept separate from hk_g17p_compression_color_format() so
 * 2x MSAA, layered/3D, shared-modifier and M5 policies keep their 1-8 byte
 * format set unchanged. */
static inline bool
hk_g17p_compression_color_format_16b(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R32G32B32A32_UINT:
   case VK_FORMAT_R32G32B32A32_SINT:
   case VK_FORMAT_R32G32B32A32_SFLOAT:
      return true;
   default:
      return false;
   }
}

static inline bool
hk_g17p_compression_depth_format(VkFormat format)
{
   return format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D32_SFLOAT ||
          format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

/* Mesa-private WSI image marker (src/vulkan/wsi/wsi_common_private.h). */
#define HK_G17P_WSI_IMAGE_CREATE_INFO_MESA ((VkStructureType)1000001002)

/* Whether plane `plane` of an image created with `info` is compressed on
 * `chip`. There is no opt-out: every eligible HAL200 image takes the
 * compressed path. The generic hk_can_compress() checks (sparse,
 * mutable-format compatibility, ail_can_compress, the AGX-wide nocompress
 * debug flag) still apply. */
static inline bool
hk_hal200_compression_eligible(enum agx_chip chip, uint32_t chip_id,
                               unsigned plane, const VkImageCreateInfo *info)
{
   if (!agx_chip_native_hal200(chip) || !info ||
       (plane != 0 && !(plane == 1 &&
                        info->format == VK_FORMAT_D32_SFLOAT_S8_UINT)))
      return false;

   /* Keep depth/stencil off the extended colour shapes: their mip chains and
    * standalone S8 reach colour-alias transfer paths that refuse them, which
    * makes legal commands fail at EndCommandBuffer. Single-level depth arrays
    * are admitted below. Colour's layered/MSAA packet support is independent
    * of this restriction.
    *
    * All G17P SoCs share the Gen9 PBE, ZLS and texture encodings for layered,
    * cube and 3D colour, non-attachment 2x/4x MSAA colour, and single-level
    * depth arrays. 2x colour attachments are additionally gated by the
    * hardware 2x uncovered-tile reload defect below (ms2x). 4x colour
    * attachments stay plain on every G17P SoC: a 4x reload loses the upper
    * bytes of a wide sample (agx_chip_uncovered_tile_ms_reload_defect(chip,
    * chip_id, 4)), so a pass that reloads them spills them, and G17P has no
    * MSAA/1-8 B decompression program for a spilled compressed target. */
   const bool ms2x =
      !agx_chip_uncovered_tile_ms_reload_defect(chip, chip_id, 2);
   /* 16-byte colour (single-sample, single-layer 2D): measured on A18 Pro
    * (T8140). M4 Pro and M4 Max share the same HAL200 PBE, texture and
    * compression metadata encoding; missing there is a hardware run of the
    * 16-byte formats and their 8-MRT spill decompression path. */
   const bool color_16b =
      chip_id == AGX_CHIP_ID_A18_PRO &&
      hk_g17p_compression_color_format_16b(info->format);
   /* Not B10G11R11_UFLOAT: a compressed tile of its class reads NaN back as
    * zero, and a render or copy into the attachment must keep NaN. */
   const bool color_1_8b = hk_g17p_compression_color_format(info->format);
   const bool color = color_1_8b || color_16b;
   const bool depth = hk_g17p_compression_depth_format(info->format);

   /* Shared images keep their plain layout contract. Read each header field
    * by copy: the chain's structures have other types, and Mesa builds with
    * strict aliasing (gcc -O2 dropped the stores of a stack chain). Copy pNext
    * straight into next: through a copied VkBaseInStructure, GCC 16's
    * points-to analysis loses the chain and drops its stores again. */
   for (const void *next = info->pNext; next;) {
      VkStructureType type;
      const void *following;
      memcpy(&type, next, sizeof(type));
      memcpy(&following, (const char *)next + offsetof(VkBaseInStructure, pNext),
             sizeof(following));
      if (type == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO) {
         VkExternalMemoryImageCreateInfo external;
         memcpy(&external, next, sizeof(external));
         if (external.handleTypes)
            return false;
      }
      if (type == HK_G17P_WSI_IMAGE_CREATE_INFO_MESA)
         return false;
      next = following;
   }

   VkImageCreateFlags allowed_flags =
      VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT |
      VK_IMAGE_CREATE_ALIAS_BIT;
   if (color_1_8b && info->imageType == VK_IMAGE_TYPE_2D)
      allowed_flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
   if (color_1_8b && info->imageType == VK_IMAGE_TYPE_3D)
      allowed_flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT |
                       VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT;
   /* Zink marks every depth attachment sample-location compatible, including
    * single-sample images. This changes sample-location tracking, not the
    * depth plane's storage class. The one-sample restriction below remains.
    * Refusing the flag also makes the host-copy performance query advertise
    * a plain device layout, so zink adds HOST_TRANSFER and keeps depth plain.
    */
   if (depth)
      allowed_flags |= VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;
   /* Only usages whose every access goes through the texture unit, a PBE or
    * ZLS. Storage and host-transfer access is raw; feedback loops read what
    * is being written; video, shading-rate and density-map units have their
    * own (unknown) compression contracts. */
   VkImageUsageFlags allowed_usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;

   /* Zink depth feedback reads use the depth plane's ZLS layout. Color
    * feedback keeps the plain layout required by texture barriers. */
   if (depth)
      allowed_usage |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;

   const bool multisampled = info->samples != VK_SAMPLE_COUNT_1_BIT;
   const bool three_d = info->imageType == VK_IMAGE_TYPE_3D;
   if (!color) {
      /* 2x/4x depth goes through ZLS (agx_pack_zls_control), sampled texture
       * reads/resolves, and graphics copies (hk_copy_requires_gfx), never the
       * colour tilebuffer's background reload program, so it is unaffected by
       * AGX_QUIRK_UNCOVERED_TILE_MS_DEFECT across native HAL200. Plane 0 is
       * verified lossless on M4 (G16G) and A18 Pro (G17P); 2x depth was also
       * measured on M4 Pro / M4 Max (G17P). The S8 plane stays plain. */
      const bool depth_ms =
         plane == 0 &&
         (info->samples == VK_SAMPLE_COUNT_2_BIT ||
          info->samples == VK_SAMPLE_COUNT_4_BIT) &&
         depth &&
         !(info->usage & (VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT)) &&
         !(info->flags &
           VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT);
      /* Non-G17P HAL200 parts (G16G) only take 2x/4x MSAA depth here;
       * single-sample depth stays plain until verified separately on G16G. */
      if (chip != AGX_CHIP_G17P && !depth_ms)
         return false;
      /* Single-sample, single-level depth arrays across G17P. Every access to
       * one layer addresses its metadata at compression_layer_stride_B: ZLS
       * (with the metadata layer stride for layered rendering), the metadata
       * fast clear, the texture and PBE descriptors. The graphics copies,
       * uploads and blits split into single-layer passes (hk_cmd_meta.c). */
      const bool depth_array =
         info->samples == VK_SAMPLE_COUNT_1_BIT && info->mipLevels == 1 &&
         depth;
      if (info->imageType != VK_IMAGE_TYPE_2D || info->extent.depth != 1 ||
          !info->arrayLayers || (info->arrayLayers != 1 && !depth_array) ||
          (info->samples != VK_SAMPLE_COUNT_1_BIT && !depth_ms) ||
          info->tiling != VK_IMAGE_TILING_OPTIMAL || info->mipLevels < 1 ||
          (info->flags & ~allowed_flags) || (info->usage & ~allowed_usage) ||
          info->extent.width < 16 || info->extent.height < 16)
         return false;

      return depth && info->mipLevels == 1;
   }

   /* Non-G17P HAL200 parts (G16G) keep colour plain here; G16G single-sample
    * colour compression is gated separately (#638). */
   if (chip != AGX_CHIP_G17P)
      return false;

   /* 16-byte texels stay on the single-sample, single-layer 2D shape. */
   if (color_16b && (three_d || multisampled || info->arrayLayers != 1))
      return false;

   if ((info->imageType != VK_IMAGE_TYPE_2D && !three_d) ||
       !info->extent.depth || !info->arrayLayers ||
       (three_d ? (info->arrayLayers != 1 || multisampled)
                : info->extent.depth != 1) ||
       (multisampled &&
        ((info->samples != VK_SAMPLE_COUNT_2_BIT &&
          info->samples != VK_SAMPLE_COUNT_4_BIT) ||
         info->mipLevels != 1)) ||
       ((info->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
        (multisampled || info->extent.width != info->extent.height ||
         info->arrayLayers % 6)) ||
       info->tiling != VK_IMAGE_TILING_OPTIMAL || info->mipLevels < 1 ||
       (info->flags & ~allowed_flags) || (info->usage & ~allowed_usage) ||
       info->extent.width < 16 || info->extent.height < 16) {
      return false;
   }

   if ((three_d || info->arrayLayers > 1) && info->mipLevels != 1)
      return false;
   /* #66: a legal MRT mix can spill any MSAA colour attachment, and G17P
    * has no MSAA/1-8 B decompression program for a spilled compressed one.
    * Only a 2x single-layer attachment on a SoC without the 2x uncovered-tile
    * reload defect always stays in the tilebuffer (see the policy above); 4x,
    * arrays, and chips with the 2x reload defect stay plain. */
   if (multisampled &&
       (info->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)) &&
       (!ms2x || info->samples != VK_SAMPLE_COUNT_2_BIT ||
        info->arrayLayers != 1 ||
        (info->usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)))
      return false;

   /* An RGBA8/BGRA8 UNORM mip chain the application never renders to is
    * written only by uploads and mip blits. A plain image takes each
    * upload as one raw compute copy and each mip level as a raw compute
    * dispatch (hk_g17p_compute_blit). A compressed one takes each upload
    * as a render pass, since G17P has no compute store that writes
    * compressed tiles, and its mips through hk_g17p_compressed_blit.
    * For small images the render passes dominate: 200 128x128 textures
    * uploaded and mipmapped in one command buffer took 2.6 times as
    * long compressed. From the size where a render pass also beats
    * compute for a mip level (HK_G17P_MIP_PBE_MIN_TEXELS), compression
    * wins: on the A18 Pro a 2048x2048 texture uploads in 0.55-0.64 of
    * the time, generates its mips in 0.40-0.57 and samples in
    * 0.69-0.84. Keep only the small ones plain. Images that are also
    * attachments are compressed at any size.
    *
    * The same holds without mips and for every compressed colour format:
    * each upload into a compressed image is a render pass. On the M4 Max
    * (T6041) an R8 130x130 texture taking 64 tile uploads per command
    * buffer (aurora-tilesubmit, a glyph-cache pattern) spent 1.37 ms per
    * submit compressed against 34 us plain. */
   if (!(info->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)) &&
       !hk_g17p_mip_level_through_pbe(info->extent.width,
                                      info->extent.height))
      return false;
   return true;
}

/* G17P compression policy without a specific chip id: admits the architectural
 * G17P shapes, while keeping 2x reload-dependent shapes off (chip_id 0 keeps
 * the 2x uncovered-tile reload defect active). */
static inline bool
hk_g17p_compression_eligible(enum agx_chip chip, unsigned plane,
                             const VkImageCreateInfo *info)
{
   return hk_hal200_compression_eligible(chip, 0, plane, info);
}

/* G17P and M5 colour images shared through
 * DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED.
 *
 * The modifier is the sharing contract. With one level, one layer and one
 * sample, the G17P (legacy AIL policy) compressed allocation is a function of
 * the format's block size and the extent alone: GPU tiling, the image, then
 * its metadata, in one memory plane. The M5 allocation (AIL_LAYOUT_POLICY_M5
 * and its metadata, one memory plane) is too. Exporter and importer therefore derive
 * the same layout, and import already follows the modifier through
 * hk_map_compression(). The packets that read and write it are the ones the
 * private policy above uses for single-level colour images: the compressed
 * 2D PBE (end-of-tile stores, meta copies and uploads) and the compressed
 * texture descriptor, with clears through the metadata fast clear.
 *
 * The shape is the private colour policy's restricted to one level, without
 * its exclusion of external memory and WSI images and without a
 * colour-attachment requirement: hk_image_init() makes every transfer
 * destination renderable, so a swapchain image written only by blits or
 * copies also takes the compressed PBE. WSI creates every swapchain image
 * with ALIAS; aliasing images of identical parameters derive the same
 * layout. Storage and host-transfer usages stay excluded. Feedback-loop
 * usage is accepted on import only (zink adds it to every imported
 * render-target dma-buf) and never chosen at allocation (hk_can_compress()
 * refuses it). hk_image.c adds the PBE store-class check.
 */
static inline bool
hk_g17p_shared_compression_shape(enum agx_chip chip, VkFormat format,
                                 VkImageType type, VkImageCreateFlags flags,
                                 VkImageUsageFlags usage)
{
   const VkImageCreateFlags allowed_flags =
      VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT |
      VK_IMAGE_CREATE_ALIAS_BIT;
   const VkImageUsageFlags allowed_usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;

   return agx_chip_g17p_or_m5(chip) &&
          type == VK_IMAGE_TYPE_2D &&
          !(flags & ~allowed_flags) && !(usage & ~allowed_usage) &&
          hk_g17p_compression_color_format(format);
}

static inline bool
hk_g17p_shared_compression_eligible(enum agx_chip chip, unsigned plane,
                                    const VkImageCreateInfo *info)
{
   return plane == 0 && info &&
          info->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT &&
          hk_g17p_shared_compression_shape(chip, info->format,
                                           info->imageType, info->flags,
                                           info->usage) &&
          info->extent.depth == 1 && info->mipLevels == 1 &&
          info->arrayLayers == 1 && info->samples == VK_SAMPLE_COUNT_1_BIT &&
          info->extent.width >= 16 && info->extent.height >= 16;
}

/* M5 lossless compression of colour render targets.
 *
 * The Gen10 compressed PBE (write) packet and texture (read) descriptor are
 * native-verified for GPU-tiled one-layer 2D images (byte-identical to
 * the native M5 PBE and texture descriptor encoders), and the allocation is
 * the measured M5 layout with its metadata (AIL_LAYOUT_POLICY_M5). Every
 * write of such an image then goes through the PBE: end-of-tile stores,
 * and the graphics meta paths hk_xfer_copyimage2() and the buffer uploads
 * take for a compressed destination (the raw tiled copiers refuse one).
 *
 * Admitted: the G17P colour formats (1-8 byte texels, so a single-sample
 * tilebuffer never spills a compressed target), one layer, any mip count
 * (one level for one-byte texels, see hk_m5_compression_one_byte()),
 * one sample, a colour attachment or a transfer destination (which
 * hk_image_init() makes renderable, so its uploads take the PBE path like
 * an attachment's: sampled textures, as native compresses them), and
 * otherwise only sampled, input-attachment and transfer usages. Shared
 * images (external memory, WSI) stay plain.
 *
 * VK_IMAGE_CREATE_ALIAS_BIT is admitted here and by the depth policy below.
 * Vulkan defines aliased contents only between images of identical creation
 * parameters, which this policy maps to the same layout, metadata included,
 * so placed aliases agree on every M5 part. The other form of ALIAS, a
 * single-plane image sharing a plane of a multi-planar image, requires
 * VK_IMAGE_CREATE_DISJOINT_BIT on both images; that flag stays refused, so
 * such an image keeps the plain layout of the plane it aliases. Sparse
 * flags stay refused.
 *
 * Every level of a compressed mip chain is described by the level-0 surface
 * and metadata addresses plus the level fields, in the PBE and
 * texture packets alike, as native does; the hardware finds the
 * level, so AIL_LAYOUT_POLICY_M5's per-level metadata placement must be the
 * hardware's.
 */
/* M5 lossless compression of colour render targets and sampled textures.
 *
 * The Gen10 compressed PBE (write) packet and texture (read) descriptor
 * support GPU-tiled 2D (single-level and mipmapped, including 1-byte mip
 * chains aligned on 2 KiB compressed-level boundaries in
 * AIL_LAYOUT_POLICY_M5), 2D arrays (single-level and mipmapped), cube maps,
 * single-level 3D volumes (1-8 byte + B10G11R11 formats), and single-sample
 * single-layer 16-byte formats (R32G32B32A32_{UINT,SINT,SFLOAT}, with spilled
 * MRT attachments decompressed in place via libagx_g17p_decompress_16b using
 * M5's 16x8 metadata tile geometry). Shared images (external memory, WSI) stay
 * plain under this private policy.
 */
static inline bool
hk_m5_color_compression_eligible(enum agx_chip chip, unsigned plane,
                                 const VkImageCreateInfo *info)
{
   if (!agx_chip_g17_m5(chip) || plane != 0 || !info)
      return false;

   /* Read the chain as hk_g17p_compression_eligible() does. */
   for (const void *next = info->pNext; next;) {
      VkStructureType type;
      const void *following;
      memcpy(&type, next, sizeof(type));
      memcpy(&following, (const char *)next + offsetof(VkBaseInStructure, pNext),
             sizeof(following));
      if (type == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO) {
         VkExternalMemoryImageCreateInfo external;
         memcpy(&external, next, sizeof(external));
         if (external.handleTypes)
            return false;
      }
      if (type == HK_G17P_WSI_IMAGE_CREATE_INFO_MESA)
         return false;
      next = following;
   }

   const bool color_1_8b =
      hk_g17p_compression_color_format(info->format) ||
      /* M5 stores it in its own class (agx_rt0_pbe_format). */
      info->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32;
   const bool color_16b = hk_g17p_compression_color_format_16b(info->format);
   if (!color_1_8b && !color_16b)
      return false;

   const bool three_d = info->imageType == VK_IMAGE_TYPE_3D;
   if (color_16b && (three_d || info->arrayLayers != 1))
      return false;

   VkImageCreateFlags allowed_flags =
      VK_IMAGE_CREATE_ALIAS_BIT |
      VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
   if (color_1_8b && info->imageType == VK_IMAGE_TYPE_2D)
      allowed_flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
   if (color_1_8b && three_d)
      allowed_flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT |
                       VK_IMAGE_CREATE_2D_VIEW_COMPATIBLE_BIT_EXT;
   const VkImageUsageFlags allowed_usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;

   if ((info->imageType != VK_IMAGE_TYPE_2D && !three_d) ||
       !info->extent.depth || !info->arrayLayers || !info->mipLevels ||
       (three_d ? (info->arrayLayers != 1 || info->mipLevels != 1)
                : info->extent.depth != 1) ||
       ((info->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
        (info->extent.width != info->extent.height ||
         info->arrayLayers % 6)))
      return false;

   return info->samples == VK_SAMPLE_COUNT_1_BIT &&
          info->tiling == VK_IMAGE_TILING_OPTIMAL &&
          !(info->flags & ~allowed_flags) &&
          (info->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT)) &&
          !(info->usage & ~allowed_usage) && info->extent.width >= 16 &&
          info->extent.height >= 16;
}

/* Multisample colour formats with a qualified compressed metadata shape:
 * RGBA16F, and the 4-byte RGBA8/BGRA8 UNORM and sRGB formats, which share the
 * R32_UINT texel size of the witnessed raw copy packet. */
static inline bool
hk_m5_msaa_color_compression_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R16G16B16A16_SFLOAT:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
      return true;
   default:
      return false;
   }
}

/* Qualified multisample metadata shapes. Sharing and aliasing flags are not
 * admitted; a mutable format is, and hk_can_compress() then requires a format
 * list of compression-compatible views.
 */
static inline bool
hk_m5_msaa_color_compression_eligible(enum agx_chip chip, unsigned plane,
                                     const VkImageCreateInfo *info)
{
   const VkImageUsageFlags allowed_usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

   if (chip != AGX_CHIP_G17S || plane != 0 || !info)
      return false;

   for (const void *next = info->pNext; next;) {
      VkStructureType type;
      memcpy(&type, next, sizeof(type));
      if (type != VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO)
         return false;
      memcpy(&next, (const char *)next + offsetof(VkBaseInStructure, pNext),
             sizeof(next));
   }

   return !(info->flags & ~VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) &&
          info->imageType == VK_IMAGE_TYPE_2D &&
          info->extent.depth == 1 && info->extent.width >= 16 &&
          info->extent.height >= 16 && info->arrayLayers == 1 &&
          info->mipLevels == 1 &&
          hk_m5_msaa_color_compression_format(info->format) &&
          (info->samples == VK_SAMPLE_COUNT_2_BIT ||
           info->samples == VK_SAMPLE_COUNT_4_BIT) &&
          info->tiling == VK_IMAGE_TILING_OPTIMAL &&
          (info->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &&
          !(info->usage & ~allowed_usage);
}

/* M5 lossless compression of depth attachments.
 *
 * The native HAL300 framebuffer setup sets ZLS control bits 2 and
 * 6 exactly when the attached depth level has compression metadata, bits 4
 * and 8 for stencil metadata and bit 26 for Depth16Unorm, and programs the
 * metadata address and its cache-line layer stride beside the surface: the
 * encoding agx_pack_zls_control() and hk's render setup already produce.
 * Depth is read through the same Gen10 texture descriptor as colour, and the
 * copy paths write it through its R16/R32_UINT colour alias with the
 * compressed PBE.
 *
 * Admitted: D16_UNORM, D32_SFLOAT and both planes of D32_SFLOAT_S8_UINT
 * (stencil is read through its R8_UINT alias and copied through the same
 * graphics path), one mip, one sample (single-layer and layered 2D arrays),
 * a depth attachment, and otherwise only sampled, input-attachment,
 * feedback-loop and transfer usages. Zink marks its depth attachments
 * sample-location compatible and feedback-loop capable, which changes
 * neither. Shared images (external memory, WSI) stay plain.
 */
static inline bool
hk_m5_depth_compression_eligible(enum agx_chip chip, unsigned plane,
                                 const VkImageCreateInfo *info)
{
   if (!agx_chip_g17_m5(chip) || !info ||
       (plane != 0 && !(plane == 1 &&
                        info->format == VK_FORMAT_D32_SFLOAT_S8_UINT)))
      return false;

   /* Read the chain as hk_g17p_compression_eligible() does. */
   for (const void *next = info->pNext; next;) {
      VkStructureType type;
      const void *following;
      memcpy(&type, next, sizeof(type));
      memcpy(&following, (const char *)next + offsetof(VkBaseInStructure, pNext),
             sizeof(following));
      if (type == VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO) {
         VkExternalMemoryImageCreateInfo external;
         memcpy(&external, next, sizeof(external));
         if (external.handleTypes)
            return false;
      }
      if (type == HK_G17P_WSI_IMAGE_CREATE_INFO_MESA)
         return false;
      next = following;
   }

   const VkImageCreateFlags allowed_flags =
      VK_IMAGE_CREATE_ALIAS_BIT |
      VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT |
      VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;
   const VkImageUsageFlags allowed_usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;

   return info->imageType == VK_IMAGE_TYPE_2D && info->extent.depth == 1 &&
          info->mipLevels == 1 &&
          info->samples == VK_SAMPLE_COUNT_1_BIT &&
          info->tiling == VK_IMAGE_TILING_OPTIMAL &&
          !(info->flags & ~allowed_flags) &&
          (info->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) &&
          !(info->usage & ~allowed_usage) && info->extent.width >= 16 &&
          info->extent.height >= 16 &&
          hk_g17p_compression_depth_format(info->format);
}

/* Stage-one experiment: native ZLS plus texture reads, no PBE destination.
 * The capability is advertised only by an explicitly enabled private kernel.
 * Transfer/image-copy and additional image semantics require separate gates.
 */
static inline bool
hk_m5_depth_msaa_compression_eligible(enum agx_chip chip, unsigned plane,
                                     const VkImageCreateInfo *info,
                                     uint64_t kernel_features)
{
   const uint64_t capability = DRM_ASAHI_FEATURE_M5_DEPTH_MSAA_COMPRESSION;
   const VkImageUsageFlags allowed_usage =
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
   /* The stencil plane of D32_SFLOAT_S8_UINT compresses with depth where the
    * kernel admits compressed multisample stencil. */
   const bool plane_ok =
      plane == 0 ||
      (plane == 1 && info && info->format == VK_FORMAT_D32_SFLOAT_S8_UINT &&
       (kernel_features & DRM_ASAHI_FEATURE_M5_STENCIL_MSAA_COMPRESSION));
   return (kernel_features & capability) &&
          agx_chip_g17_m5(chip) && plane_ok && info && !info->pNext &&
          info->imageType == VK_IMAGE_TYPE_2D && info->extent.depth == 1 &&
          info->arrayLayers == 1 && info->mipLevels == 1 &&
          (info->samples == VK_SAMPLE_COUNT_2_BIT ||
           info->samples == VK_SAMPLE_COUNT_4_BIT) &&
          info->tiling == VK_IMAGE_TILING_OPTIMAL && info->flags == 0 &&
          (info->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) &&
          !(info->usage & ~allowed_usage) && info->extent.width >= 16 &&
          info->extent.height >= 16 &&
          (info->format == VK_FORMAT_D32_SFLOAT ||
           info->format == VK_FORMAT_D32_SFLOAT_S8_UINT);
}

/* Private transfer qualification: preserve the stage-one shape and capability
 * checks while admitting only the complete DS/sampled/transfer usage set.
 * Stencil stays plain; its sole permitted extension excludes sampled usage.
 */
static inline bool
hk_m5_depth_msaa_transfer_eligible(enum agx_chip chip, unsigned plane,
                                  const VkImageCreateInfo *info,
                                  uint64_t kernel_features)
{
   const VkImageUsageFlags usage =
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   if (!info || info->usage != usage)
      return false;

   if (info->pNext) {
      VkBaseInStructure extension;
      memcpy(&extension, info->pNext, sizeof(extension));
      if (info->format != VK_FORMAT_D32_SFLOAT_S8_UINT ||
          extension.sType != VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO ||
          extension.pNext)
         return false;
      VkImageStencilUsageCreateInfo stencil;
      memcpy(&stencil, info->pNext, sizeof(stencil));
      if (stencil.stencilUsage != (usage & ~VK_IMAGE_USAGE_SAMPLED_BIT))
         return false;
   }

   VkImageCreateInfo base = *info;
   base.pNext = NULL;
   base.usage &= ~(VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT);
   return hk_m5_depth_msaa_compression_eligible(chip, plane, &base,
                                                kernel_features);
}

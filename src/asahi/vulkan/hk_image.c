/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_image.h"
#include "hk_frame_anatomy.h"
#include "hk_instance.h"
#include "hk_g17p_compression.h"
#include "agx_bg_eot_key.h"
#include "hk_native_descriptor.h"
#include "asahi/layout/layout.h"
#include "drm-uapi/drm_fourcc.h"
#include "util/bitscan.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/macros.h"
#include "util/u_math.h"
#include "util/u_debug.h"
#include "vulkan/vulkan_core.h"

#include "agx_bo.h"
#include "hk_buffer.h"
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_entrypoints.h"
#include "hk_g17p_meta_copy_core.h"
#include "hk_image_layout_core.h"
#include "hk_physical_device.h"

#include "vk_format.h"

#include <inttypes.h>

/* However, exposing the standard sparse block sizes requires using the standard
 * alignment 65k.
 */
#define HK_SPARSE_ALIGN_B 65536

DEBUG_GET_ONCE_BOOL_OPTION(m5_depth_msaa_transfer,
                          "ASAHI_M5_DEPTH_MSAA_TRANSFER", true)

DEBUG_GET_ONCE_BOOL_OPTION(m5_depth_msaa_compression,
                          "ASAHI_M5_DEPTH_MSAA_COMPRESSION", true)

static VkFormatFeatureFlags2
hk_get_image_plane_format_features(struct hk_physical_device *pdev,
                                   VkFormat vk_format, VkImageTiling tiling)
{
   VkFormatFeatureFlags2 features = 0;

   /* G17P reconstructs single-plane interleaved 4:2:2 in the shader rather
    * than in the texture unit (hk_image_view.c programs these views as plain
    * four-component words), so it can honour chroma-reconstruction features
    * that the generic path below cannot, and it used to advertise all of
    * them.  Explicit reconstruction is withdrawn.
    *
    * Advertising CHROMA_RECONSTRUCTION_EXPLICIT is what makes a conformance
    * run evaluate this driver against the explicit reference, and the
    * explicit 4:2:2 reference in VK-GL-CTS is wrong for COSITED_EVEN: at even
    * luma index it weights the chroma sample at subI + 1, one chroma texel
    * right of the texel COSITED_EVEN co-sites with that luma texel.  We do
    * not carry a capability whose only effect is to select a broken
    * reference.  MIDPOINT_CHROMA_SAMPLES and SEPARATE_RECONSTRUCTION_FILTER
    * go with it: both only describe how chroma is reconstructed, and the
    * remaining COSITED_EVEN path is reconstructed implicitly as far as the
    * API is concerned.
    *
    * Note this is NOT bit-for-bit what the generic path below computes for
    * these formats.  There n_planes == 1 and the single plane's
    * denominator_scales are {1, 1} (vk_format.c, y_plane(...)), so
    * cosited_chroma stays false and neither chroma-location bit is added at
    * all.  That reading is not reconciled with M1/M2 conformance and has not
    * been measured on M1/M2 hardware, so do not copy it here on the strength
    * of code inspection alone.  COSITED_EVEN is kept because it is what this
    * driver actually implements and passes.
    */
   if (agx_native_graphics(&pdev->dev) && hk_packed_ycbcr_bits(vk_format)) {
      if (tiling != VK_IMAGE_TILING_OPTIMAL && tiling != VK_IMAGE_TILING_LINEAR)
         return 0;
      return VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
             VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
             VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
             VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
             VK_FORMAT_FEATURE_2_COSITED_CHROMA_SAMPLES_BIT |
             VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT;
   }

   /* This optional format needs hacks for opaque black, so hide for
    * performance. We might specially enable this for Proton / behind a driconf.
    */
   if (vk_format == VK_FORMAT_A8_UNORM_KHR)
      return 0;

   enum pipe_format p_format = hk_format_to_pipe_format(vk_format);
   /* D16S8 uses the existing independent D16 and S8 planes, as D32S8 does
    * on G17P and M5. */
   if (vk_format == VK_FORMAT_D16_UNORM_S8_UINT) {
      if (pdev->dev.chip != AGX_CHIP_G17P && !agx_chip_g17_m5(pdev->dev.chip))
         return 0;
      p_format = PIPE_FORMAT_Z16_UNORM;
   }
   if (p_format == PIPE_FORMAT_NONE)
      return 0;

   enum pipe_format padded = hk_padded_image_format(p_format);
   if ((agx_native_graphics(&pdev->dev) || agx_chip_g17_m5(pdev->dev.chip)) &&
       padded != p_format) {
      /* Linear and external modifier layouts must retain the API's packed
       * texel stride. This path is restricted to driver-owned optimal images. */
      if (tiling != VK_IMAGE_TILING_OPTIMAL)
         return 0;
      VkFormatFeatureFlags2 rgb = VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
         VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
      const enum pipe_format integer = hk_scaled_texture_format(padded);

      /* A scaled format reaches this block only because it is three-component
       * and therefore padded. Everything this block would grant it rests on
       * the integer-fetch-plus-convert emulation that the withdrawal below
       * removed for every other component count, so withhold it here too and
       * leave these formats uniformly unsupported as images, as they are on
       * G13 and G14. */
      if (integer != PIPE_FORMAT_NONE)
         return 0;

      const enum pipe_format storage =
         integer != PIPE_FORMAT_NONE ? integer : padded;
      /* vkCmdBlitImage between a scaled and a non-scaled format has no
       * agreed-upon result: the scaled side carries integer values and the
       * other carries normalized ones, so no conversion satisfies both. Other
       * drivers withhold every image feature from scaled formats for this
       * reason. Keep sampling and transfers, which are well-defined, and
       * withhold blit. */
      const bool blittable = integer == PIPE_FORMAT_NONE;
      if (ail_pixel_format[storage].texturable) {
         rgb |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT;
         if (blittable)
            rgb |= VK_FORMAT_FEATURE_2_BLIT_SRC_BIT;
         if (!util_format_is_pure_integer(storage) ||
             hk_uscaled_normalized_format(padded) != PIPE_FORMAT_NONE)
            rgb |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
      }
      if (ail_pixel_format[storage].renderable && blittable)
         rgb |= VK_FORMAT_FEATURE_2_BLIT_DST_BIT;
      if (ail_pixel_format[padded].renderable) {
         rgb |= VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT;
         if (!util_format_is_pure_integer(padded))
            rgb |= VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT;
      }
      if ((rgb & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT) &&
          agx_chip_sampler_generation(pdev->dev.chip) ==
             AGX_SAMPLER_GENERATION_GEN5)
         rgb |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT;
      return rgb;
   }

   /* Keep compressed-family reporting atomic with the corresponding Vulkan
    * feature. The complete BC, ETC2/EAC and ASTC LDR format matrices are
    * hardware-qualified through the native G17P sampled/upload path. */
   if (agx_native_graphics(&pdev->dev) &&
       util_format_is_compressed(p_format) &&
       !hk_g17p_bc_format_supported(vk_format)) {
      const enum util_format_layout layout =
         util_format_description(p_format)->layout;
      const bool enabled = layout == UTIL_FORMAT_LAYOUT_ETC ||
                           layout == UTIL_FORMAT_LAYOUT_ASTC;
      if (!enabled)
         return 0;
   }

   /* NPOT formats only supported for texel buffers */
   if (!util_is_power_of_two_nonzero(util_format_get_blocksize(p_format)))
      return 0;

   /* Linear block-compressed images are all sorts of problematic, not sure
    * if AGX even supports them. Don't try.
    */
   if (util_format_is_compressed(p_format) && tiling != VK_IMAGE_TILING_OPTIMAL)
      return 0;

   /* No scaled format has an ail_pixel_format[] entry on any generation: the
    * hardware does not sample them.  hk_scaled_texture_format() maps SSCALED
    * to SINT and USCALED to UINT, and the image features above were granted on
    * the strength of that emulation -- an integer fetch plus an int-to-float
    * conversion in the shader.  Gate them on descriptor generation 9 and G13
    * and G14 never saw them, so the whole scaled image-format surface existed
    * on one generation only, which is not a hardware difference.
    *
    * Withdraw it.  radv and anv withhold every image feature from scaled
    * formats; matching that makes the optional VK_FORMAT_*_[SU]SCALED_* image
    * formats uniformly unsupported here, as they already are on G13 and G14.
    * Vertex-buffer use of the same formats is unaffected: it does not come
    * through this function.
    */

   if (ail_pixel_format[p_format].texturable) {
      features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT;
      features |= VK_FORMAT_FEATURE_2_BLIT_SRC_BIT;

      /* We can sample integer formats but it doesn't make sense to linearly
       * filter them.
       */
      if (!util_format_is_pure_integer(p_format)) {
         features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
      }

      if (vk_format_has_depth(vk_format)) {
         features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
      }
   }

   if (ail_pixel_format[p_format].renderable) {
      features |= VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT |
                  VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT |
                  VK_FORMAT_FEATURE_2_BLIT_DST_BIT |
                  VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
                  VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT |
                  VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT;
   }

   if (vk_format_is_depth_or_stencil(vk_format)) {
      if (!(p_format == PIPE_FORMAT_Z32_FLOAT ||
            p_format == PIPE_FORMAT_S8_UINT ||
            p_format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT ||
            p_format == PIPE_FORMAT_Z16_UNORM) ||
          tiling == VK_IMAGE_TILING_LINEAR)
         return 0;

      /* Depth/stencil blits use the common graphics meta path, so the
       * destination capability follows attachment support rather than the
       * colour-PBE `renderable` table above. */
      features |= VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT |
                  VK_FORMAT_FEATURE_2_BLIT_DST_BIT;
   }

   /* Our image atomic lowering doesn't bother to handle linear */
   if ((p_format == PIPE_FORMAT_R32_UINT || p_format == PIPE_FORMAT_R32_SINT ||
        (p_format == PIPE_FORMAT_R32_FLOAT &&
         (pdev->dev.chip == AGX_CHIP_G17P ||
          agx_chip_g17_m5(pdev->dev.chip)))) &&
       tiling == VK_IMAGE_TILING_OPTIMAL) {

      features |= VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT;
   }

   if ((features & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT) &&
       agx_chip_sampler_generation(pdev->dev.chip) == AGX_SAMPLER_GENERATION_GEN5)
      features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT;

   if (features != 0) {
      features |= VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT;
      features |= VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
      features |= VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT;

      if (vk_format_has_depth(vk_format)) {
         features |= VK_FORMAT_FEATURE_2_DEPTH_COPY_ON_COMPUTE_QUEUE_BIT_KHR |
                     VK_FORMAT_FEATURE_2_DEPTH_COPY_ON_TRANSFER_QUEUE_BIT_KHR;
      }
      if (vk_format_has_stencil(vk_format)) {
         features |= VK_FORMAT_FEATURE_2_STENCIL_COPY_ON_COMPUTE_QUEUE_BIT_KHR |
                     VK_FORMAT_FEATURE_2_STENCIL_COPY_ON_TRANSFER_QUEUE_BIT_KHR;
      }
   }

   return features;
}

VkFormatFeatureFlags2
hk_get_image_format_features(struct hk_physical_device *pdev,
                             VkFormat vk_format, VkImageTiling tiling)
{
   const struct vk_format_ycbcr_info *ycbcr_info =
      vk_format_get_ycbcr_info(vk_format);
   if (ycbcr_info == NULL)
      return hk_get_image_plane_format_features(pdev, vk_format, tiling);

   /* For multi-plane, we get the feature flags of each plane separately,
    * then take their intersection as the overall format feature flags
    */
   VkFormatFeatureFlags2 features = ~0ull;
   bool cosited_chroma = false;
   for (uint8_t plane = 0; plane < ycbcr_info->n_planes; plane++) {
      const struct vk_format_ycbcr_plane *plane_info =
         &ycbcr_info->planes[plane];
      features &=
         hk_get_image_plane_format_features(pdev, plane_info->format, tiling);
      if (plane_info->denominator_scales[0] > 1 ||
          plane_info->denominator_scales[1] > 1)
         cosited_chroma = true;
   }
   if (features == 0)
      return 0;

   /* Uh... We really should be able to sample from YCbCr */
   assert(features & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT);
   assert(features & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT);

   /* These aren't allowed for YCbCr formats */
   features &=
      ~(VK_FORMAT_FEATURE_2_BLIT_SRC_BIT | VK_FORMAT_FEATURE_2_BLIT_DST_BIT |
        VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT |
        VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT);

   /* This is supported on all YCbCr formats */
   features |=
      VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT;

   if (ycbcr_info->n_planes > 1) {
      /* DISJOINT_BIT implies that each plane has its own separate binding,
       * while SEPARATE_RECONSTRUCTION_FILTER_BIT implies that luma and chroma
       * each have their own, separate filters, so these two bits make sense
       * for multi-planar formats only.
       *
       * For MIDPOINT_CHROMA_SAMPLES_BIT, NVIDIA HW on single-plane interleaved
       * YCbCr defaults to COSITED_EVEN, which is inaccurate and fails tests.
       * This can be fixed with a NIR tweak but for now, we only enable this bit
       * for multi-plane formats. See Issue #9525 on the mesa/main tracker.
       */
      features |=
         VK_FORMAT_FEATURE_DISJOINT_BIT |
         VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT |
         VK_FORMAT_FEATURE_2_MIDPOINT_CHROMA_SAMPLES_BIT;
   }

   /* G17P used to add CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE here, and only
    * G17P: agx_native_graphics() is G16G/G17P, so M1/M2 reach these formats
    * without it and every explicit_* conformance case is NotSupported there.
    *
    * That bit is the sole route into VK-GL-CTS's explicit reconstruction
    * reference, and both of its routines are wrong.  reconstructLinearX-
    * ChromaSample() treats its constant as the weight of the first chroma
    * sample; reconstructLinearXYChromaSample() hands the same constants to
    * linearInterpolate(), which weights the second.  So the X-only routine is
    * wrong at COSITED_EVEN and the XY routine is wrong at MIDPOINT in either
    * axis -- which is exactly the failing set: 4:2:2 cosited, and 4:2:0
    * cosited_midpoint / midpoint_cosited / midpoint_midpoint, with
    * cosited_cosited passing.
    *
    * We are not giving this up because we cannot do it.  G17P's explicit
    * multi-plane reconstruction is correct: over 171 logged failing sample
    * points the driver matched the Vulkan spec reconstruction 171/171 and the
    * as-written CTS value 0/171, while the reference's own bounds contained
    * the as-written value 171/171 and the spec value 0/171.  The same
    * instrument scored 248/248 on packed 4:2:2 before it was extended here.
    * We are giving up claiming a capability whose only effect is to select a
    * proven-wrong reference, and matching what M1/M2 report.
    *
    * The cost is real and deliberate: 512 conformance cases that pass today
    * become NotSupported, alongside the 512 that fail.  Nothing outside
    * ycbcr.conversion.*.chroma_reconstruction moves -- of the 53,768
    * multi-plane YCbCr cases, exactly 1,024 request explicit reconstruction.
    *
    * Nothing is needed on the sampler side.  Unlike packed single-plane,
    * whose views are programmed as plain four-component word textures so
    * hk_sampler.c must force the shader lowering on, multi-plane views are
    * real YCbCr views: hk_sampler.c's forcing is gated on
    * hk_packed_ycbcr_bits(), which is 0 here, so multi-plane only ever
    * reconstructed explicitly when an application asked.  Withdrawing the bit
    * makes that request illegal, lower_ycbcr_explicit() stops firing, and the
    * common implicit path takes over -- the path the 1,536 default_* cases
    * already exercise and pass.
    */

   if (cosited_chroma)
      features |= VK_FORMAT_FEATURE_COSITED_CHROMA_SAMPLES_BIT;

   return features;
}

static VkFormatFeatureFlags2
vk_image_usage_to_format_features(VkImageUsageFlagBits usage_flag)
{
   assert(util_bitcount(usage_flag) == 1);
   switch (usage_flag) {
   case VK_IMAGE_USAGE_TRANSFER_SRC_BIT:
      return VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
             VK_FORMAT_FEATURE_BLIT_SRC_BIT;
   case VK_IMAGE_USAGE_TRANSFER_DST_BIT:
      return VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
             VK_FORMAT_FEATURE_BLIT_DST_BIT;
   case VK_IMAGE_USAGE_SAMPLED_BIT:
      return VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT;
   case VK_IMAGE_USAGE_STORAGE_BIT:
      return VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT;
   case VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT:
      return VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT;
   case VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT:
      return VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT;
   default:
      return 0;
   }
}

/* Enable the validated multisample color layouts on G17S. */
DEBUG_GET_ONCE_NUM_OPTION(m5_msaa_color_compression,
                         "ASAHI_M5_MSAA_COLOR_COMPRESSION", 6)

static bool
hk_can_compress(const struct agx_device *dev, VkFormat format, unsigned plane,
                unsigned width, unsigned height, unsigned samples,
                VkImageCreateFlagBits flags, VkImageUsageFlagBits usage,
                const void *pNext, bool allow_agx3, bool srgb_pair_views)
{
   const struct vk_format_ycbcr_info *ycbcr_info =
      vk_format_get_ycbcr_info(format);
   const VkFormat api_format = format;

   if (ycbcr_info) {
      format = ycbcr_info->planes[plane].format;
      width /= ycbcr_info->planes[plane].denominator_scales[0];
      height /= ycbcr_info->planes[plane].denominator_scales[0];
   } else if (hk_split_depth_stencil(format)) {
      format = hk_depth_stencil_plane_format(format, plane);
   }

   /* Allow disabling compression for debugging */
   if (dev->debug & AGX_DBG_NOCOMPRESS)
      return false;

   /* Copies must preserve the packed word, including non-canonical shared
    * exponents and NaN payloads. The R32_UINT copy view cannot decode these
    * formats' compression classes. Apply this to modifier queries too, so
    * imported images cannot reintroduce a compressed packed-float surface.
    */
   if (hk_agx2_packed_float_copy(dev->chip, format))
      return false;

   /* AGX3 compresses only what its policies in hk_g17p_compression.h
    * admitted (allow_agx3). The private policy excludes images shared through
    * external memory; G17P and M5 share colour images compressed only through the
    * GPU_TILED_COMPRESSED modifier (hk_g17p_shared_compression_eligible()),
    * which is all hk_can_compress_format() advertises on AGX3. */
   if (hk_descriptor_generation_is_agx3(
          agx_chip_descriptor_generation(dev->chip)) &&
       !allow_agx3) {
      perf_debug_dev(dev, "No compression: AGX3 image outside G17P policy");
      return false;
   }

   /* TODO: Handle compression with sparse. This should be doable but it's a bit
    * subtle. Correctness first.
    */
   if (flags & (VK_IMAGE_CREATE_SPARSE_ALIASED_BIT |
                VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT)) {
      perf_debug_dev(dev, "No compression: sparse");
      return false;
   }

   /* Image compression is not (yet?) supported with host image copies,
    * although the vendor driver does support something similar if I recall.
    * Compression is not supported in hardware for storage images or mutable
    * formats in general.
    *
    * Feedback loops are problematic with compression. The GL driver bans them.
    * Interestingly, the relevant CTS tests pass on G13G and G14C, but not on
    * G13D. Keep that restriction except for depth images already admitted by
    * the G17P or M5 policy. Rejecting those here contradicts the depth policy and
    * makes the host-copy performance query report a plain device layout;
    * Zink then retains HOST_TRANSFER and never compresses its depth images.
    */
   /* Decided on the API format: the stencil plane of D32_SFLOAT_S8_UINT is
    * S8_UINT here, and zink gives its depth/stencil images feedback-loop
    * usage. */
   const bool depth_feedback =
      (dev->chip == AGX_CHIP_G17P || agx_chip_g17_m5(dev->chip)) &&
      allow_agx3 && hk_g17p_compression_depth_format(api_format);
   if ((usage & (VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT |
                 VK_IMAGE_USAGE_STORAGE_BIT)) ||
       ((usage & VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT) &&
        !depth_feedback)) {

      perf_debug_dev(
         dev, "No compression: incompatible usage -%s%s%s",
         (usage & VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT) ? " host-transfer" : "",
         (usage & VK_IMAGE_USAGE_STORAGE_BIT) ? " storage" : "",
         (usage & VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT)
            ? " feedback-loop"
            : "");
      return false;
   }

   enum pipe_format p_format = hk_format_to_pipe_format(format);

   /* Check for format compatibility if mutability is enabled. */
   if (flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) {
      const struct VkImageFormatListCreateInfo *format_list =
         (void *)vk_find_struct_const(pNext, IMAGE_FORMAT_LIST_CREATE_INFO);

      if (!format_list || format_list->viewFormatCount == 0) {
         /* Without a list any compatible format may view the image, and
          * compressed data only decodes through its own channel layout. The
          * hk_mutable_format_srgb_pair application option asserts that only
          * the format's sRGB/UNORM twin (same channels) will. */
         const bool has_twin = util_format_srgb(p_format) != PIPE_FORMAT_NONE ||
                               util_format_is_srgb(p_format);
         if (!srgb_pair_views || !has_twin)
            return false;
      }

      for (unsigned i = 0; format_list && i < format_list->viewFormatCount;
           ++i) {
         if (format_list->pViewFormats[i] == VK_FORMAT_UNDEFINED)
            continue;

         enum pipe_format view_format =
            hk_format_to_pipe_format(format_list->pViewFormats[i]);

         if (!ail_formats_compatible(p_format, view_format)) {
            perf_debug_dev(dev, "No compression: incompatible image view");
            return false;
         }
      }
   }

   if (!ail_can_compress(p_format, width, height, samples)) {
      perf_debug_dev(dev, "No compression: invalid layout %s %ux%ux%u",
                     util_format_short_name(p_format), width, height, samples);
      return false;
   }

   if (allow_agx3)
      perf_debug_dev(dev, "G17P compression enabled: %s %ux%u",
                     util_format_short_name(p_format), width, height);

   return true;
}

/* M5 lossless storage compression of formats without image atomics: the
 * half-float formats, RG32F, RGBA32F and the 8- and 16-bit-channel integer
 * and normalized formats have none, and R32F has them only with the float32
 * image atomic features, so it is compressed only when the device did not
 * enable them (and its views cannot reinterpret it). The single-layer/level
 * native read/write packets use the allocator's recovered Gen10 metadata
 * layout, the same layout colour attachments use, so a storage image may also
 * be a colour or input attachment and a transfer destination: end-of-tile
 * stores, shader image stores, clears and uploads all write it through
 * compressed PBE descriptors, and an input attachment reads it as a sampled
 * colour attachment does. Bits 0-10 select RGBA16F, RG16F, R16F, R32F, RG32F, RGBA32F,
 * RGBA8 UNORM, the other RGBA16 formats, the other RG16 formats, the other
 * RGBA8 formats and the other R16 formats, respectively; bit 12 admits
 * mipmapped images, whose levels compress as those of colour attachments
 * do. Zero retains the uncompressed diagnostic control. */
DEBUG_GET_ONCE_NUM_OPTION(m5_storage_compression,
                         "ASAHI_M5_STORAGE_COMPRESSION", 6143)

static bool
hk_m5_storage_compression_eligible(const struct agx_device *dev, unsigned plane,
                                 const VkImageCreateInfo *info,
                                 bool float_image_atomics)
{
   /* Typeless resources of D3D translation layers are mutable and list
    * their view formats: the formats of one channel layout (UNORM, SNORM,
    * UINT, SINT, FLOAT). None of these formats has image atomics, so such an
    * image compresses when hk_can_compress() finds only formats of the
    * image's channel layout in the list. */
   const VkImageCreateFlags allowed_flags =
      VK_IMAGE_CREATE_EXTENDED_USAGE_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
   const VkBaseInStructure *ext = info->pNext;
   const bool format_list_only =
      (info->flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) && ext &&
      ext->sType == VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO &&
      !ext->pNext;

   if (dev->chip != AGX_CHIP_G17S || plane ||
       (info->pNext && !format_list_only) ||
       (info->flags & ~allowed_flags) ||
       info->imageType != VK_IMAGE_TYPE_2D || info->extent.depth != 1 ||
       info->extent.width < 16 || info->extent.height < 16 ||
       (info->mipLevels != 1 &&
        !(debug_get_option_m5_storage_compression() & 4096)) ||
       info->arrayLayers != 1 ||
       info->samples != VK_SAMPLE_COUNT_1_BIT ||
       info->tiling != VK_IMAGE_TILING_OPTIMAL ||
       !(info->usage & VK_IMAGE_USAGE_STORAGE_BIT) ||
       (info->usage & ~(VK_IMAGE_USAGE_STORAGE_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)))
      return false;

   unsigned bit;
   switch (info->format) {
   case VK_FORMAT_R16G16B16A16_SFLOAT: bit = 1; break;
   case VK_FORMAT_R16G16_SFLOAT: bit = 2; break;
   case VK_FORMAT_R16_SFLOAT: bit = 4; break;
   case VK_FORMAT_R32_SFLOAT:
      if (float_image_atomics)
         return false;
      bit = 8;
      break;
   case VK_FORMAT_R32G32_SFLOAT: bit = 16; break;
   case VK_FORMAT_R32G32B32A32_SFLOAT: bit = 32; break;
   case VK_FORMAT_R8G8B8A8_UNORM: bit = 64; break;
   case VK_FORMAT_R16G16B16A16_UNORM:
   case VK_FORMAT_R16G16B16A16_SNORM:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R16G16B16A16_SINT: bit = 128; break;
   case VK_FORMAT_R16G16_UNORM:
   case VK_FORMAT_R16G16_SNORM:
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R16G16_SINT: bit = 256; break;
   case VK_FORMAT_R8G8B8A8_SNORM:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_R8G8B8A8_SINT: bit = 512; break;
   case VK_FORMAT_R16_UNORM:
   case VK_FORMAT_R16_SNORM:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R16_SINT: bit = 1024; break;
   default: return false;
   }
   /* A mutable image's views may be FLOAT when the image is not: the list
    * check in hk_can_compress() keeps them to the image's channel layout. */
   if ((info->flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) && !format_list_only)
      return false;
   return debug_get_option_m5_storage_compression() & bit;
}

static bool
hk_can_compress_create_info(const struct agx_device *dev, unsigned plane,
                            const VkImageCreateInfo *info,
                            bool float_image_atomics, bool srgb_pair_views);

/* Whether shaders of this device may use atomics on R32_SFLOAT images. */
static bool
hk_float_image_atomics(const struct hk_device *dev)
{
   const struct vk_features *f = &dev->vk.enabled_features;
   return f->shaderImageFloat32Atomics || f->shaderImageFloat32AtomicAdd ||
          f->shaderImageFloat32AtomicMinMax || f->sparseImageFloat32Atomics ||
          f->sparseImageFloat32AtomicAdd || f->sparseImageFloat32AtomicMinMax;
}

/* Compressed data is only decodable through its own channel class. Every
 * G17P writer of a colour image is a PBE; the end-of-tile PBE may store a
 * different format from the image (agx_g17p_rt0_pbe_format packs
 * R11G11B10/RGB9E5 into R32_UINT words). Never compress an image whose store
 * class would differ from the class its textures read. */
static bool
hk_g17p_store_class_matches(const struct agx_device *dev, VkFormat format)
{
   const enum pipe_format p_format = hk_format_to_pipe_format(format);
   const enum pipe_format store = agx_g17p_rt0_pbe_format(
      util_format_linear(hk_padded_image_format(p_format)));
   if (ail_formats_compatible(p_format, store))
      return true;

   perf_debug_dev(dev, "No compression: G17P stores %s through class of %s",
                  util_format_short_name(p_format),
                  util_format_short_name(store));
   return false;
}

/* DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED on G17P: the shape
 * hk_g17p_shared_compression_shape() admits, a matching PBE store class, and
 * views only of compatible formats. pNext may be NULL (format queries). */
static bool
hk_g17p_shared_compression_supported(const struct agx_device *dev,
                                     VkFormat format, VkImageType type,
                                     VkImageCreateFlags flags,
                                     VkImageUsageFlags usage,
                                     const void *pNext)
{
   if ((dev->debug & AGX_DBG_NOCOMPRESS) ||
       !hk_g17p_shared_compression_shape(dev->chip, format, type, flags,
                                         usage) ||
       !hk_g17p_store_class_matches(dev, format))
      return false;

   if (flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) {
      const struct VkImageFormatListCreateInfo *list =
         pNext ? vk_find_struct_const(pNext, IMAGE_FORMAT_LIST_CREATE_INFO)
               : NULL;
      if (!list || !list->viewFormatCount)
         return false;
      const enum pipe_format p_format = hk_format_to_pipe_format(format);
      for (unsigned i = 0; i < list->viewFormatCount; ++i) {
         if (list->pViewFormats[i] != VK_FORMAT_UNDEFINED &&
             !ail_formats_compatible(
                p_format, hk_format_to_pipe_format(list->pViewFormats[i])))
            return false;
      }
   }
   return true;
}

bool
hk_can_compress_format(const struct agx_device *dev, VkFormat format)
{
   /* Check compressability of a sufficiently large image of the same
    * format, since we don't have dimensions here. This is lossy for
    * small images, but that's ok.
    *
    * Likewise, we do not set flags as flags only disable compression.
    *
    * On AGX3 only G17P and M5 advertise the modifier, for the colour formats
    * they share compressed (hk_g17p_shared_compression_supported()).
    */
   const bool g17p_shared = hk_g17p_shared_compression_supported(
      dev, format, VK_IMAGE_TYPE_2D, 0, 0, NULL);
   return hk_can_compress(dev, format, 0, 64, 64, 1, 0, 0, NULL, g17p_shared,
                          false);
}

bool
hk_interchange_format(const struct agx_device *dev, VkFormat format)
{
   if (dev->chip != AGX_CHIP_G17P || (dev->debug & AGX_DBG_NOCOMPRESS))
      return false;

   switch (format) {
   case VK_FORMAT_B8G8R8A8_UNORM:
   case VK_FORMAT_B8G8R8A8_SRGB:
   case VK_FORMAT_R8G8B8A8_UNORM:
   case VK_FORMAT_R8G8B8A8_SRGB:
      return true;
   default:
      return false;
   }
}

static bool
hk_interchange_supported(const struct agx_device *dev, VkFormat format,
                         VkImageType type, VkImageCreateFlags flags,
                         VkImageUsageFlags usage, const void *pNext)
{
   if (!hk_interchange_format(dev, format) || type != VK_IMAGE_TYPE_2D)
      return false;

   /* Same restrictions as any compressed image: no storage, host copies or
    * sparse, and views only of compatible formats.  Feedback-loop usage is
    * accepted here exactly as for an explicit GPU_TILED_COMPRESSED modifier
    * (zink adds it to every imported render-target dmabuf); allocation never
    * chooses interchange for it, see choose_drm_format_mod(). */
   if (usage & (VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT | VK_IMAGE_USAGE_STORAGE_BIT))
      return false;
   if (flags & (VK_IMAGE_CREATE_SPARSE_ALIASED_BIT |
                VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
                VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT |
                VK_IMAGE_CREATE_DISJOINT_BIT))
      return false;

   if (flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) {
      const struct VkImageFormatListCreateInfo *list =
         pNext ? vk_find_struct_const(pNext, IMAGE_FORMAT_LIST_CREATE_INFO)
               : NULL;
      if (!list || !list->viewFormatCount)
         return false;
      const enum pipe_format p_format = hk_format_to_pipe_format(format);
      for (unsigned i = 0; i < list->viewFormatCount; ++i) {
         if (!ail_formats_compatible(
                p_format, hk_format_to_pipe_format(list->pViewFormats[i])))
            return false;
      }
   }
   return true;
}

static bool
hk_interchange_create_info(struct hk_device *dev, const VkImageCreateInfo *info)
{
   return hk_interchange_supported(&dev->dev, info->format, info->imageType,
                                   info->flags, info->usage, info->pNext) &&
          info->mipLevels == 1 && info->arrayLayers == 1 &&
          info->samples == VK_SAMPLE_COUNT_1_BIT && info->extent.width >= 16 &&
          info->extent.height >= 16 && info->extent.width <= 16384 &&
          info->extent.height <= 16384;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetPhysicalDeviceImageFormatProperties2(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceImageFormatInfo2 *pImageFormatInfo,
   VkImageFormatProperties2 *pImageFormatProperties)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);

   const VkPhysicalDeviceExternalImageFormatInfo *external_info =
      vk_find_struct_const(pImageFormatInfo->pNext,
                           PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);

   /* Initialize to zero in case we return VK_ERROR_FORMAT_NOT_SUPPORTED */
   memset(&pImageFormatProperties->imageFormatProperties, 0,
          sizeof(pImageFormatProperties->imageFormatProperties));

   const struct vk_format_ycbcr_info *ycbcr_info =
      vk_format_get_ycbcr_info(pImageFormatInfo->format);

   /* For the purposes of these checks, we don't care about all the extra
    * YCbCr features and we just want the accumulation of features available
    * to all planes of the given format.
    */
   VkFormatFeatureFlags2 features;
   if (ycbcr_info == NULL) {
      features = hk_get_image_plane_format_features(
         pdev, pImageFormatInfo->format, pImageFormatInfo->tiling);
   } else {
      features = ~0ull;
      assert(ycbcr_info->n_planes > 0);
      for (uint8_t plane = 0; plane < ycbcr_info->n_planes; plane++) {
         const VkFormat plane_format = ycbcr_info->planes[plane].format;
         features &= hk_get_image_plane_format_features(
            pdev, plane_format, pImageFormatInfo->tiling);
      }
   }

   /* Sparse + host-image-copy doesn't make sense. Forbid it. */
   if (pImageFormatInfo->flags & (VK_IMAGE_CREATE_SPARSE_ALIASED_BIT |
                                  VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                                  VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT)) {
      /* Sparse format enumeration also uses this query. Do not report a
       * supported sparse image when the device cannot bind sparse memory.
       */
      if (!pdev->vk.supported_features.sparseBinding)
         return VK_ERROR_FORMAT_NOT_SUPPORTED;

      if (pdev->dev.chip == AGX_CHIP_G17P &&
          (pImageFormatInfo->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT)) {
         const struct vk_features *supported = &pdev->vk.supported_features;
         if (pImageFormatInfo->tiling != VK_IMAGE_TILING_OPTIMAL ||
             (pImageFormatInfo->type == VK_IMAGE_TYPE_2D &&
              !supported->sparseResidencyImage2D) ||
             (pImageFormatInfo->type == VK_IMAGE_TYPE_3D &&
              !supported->sparseResidencyImage3D))
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
      }

      features &= ~VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT;

      /* We also do not support sparse RESIDENCY on a format whose texel block
       * is not a power of two, as we have no residency granularity to report
       * for that interaction and there are no known use cases.
       *
       * Sparse BINDING and ALIASED have no such interaction: they bind and
       * alias opaque memory ranges and never ask about texel blocks.
       * Refusing them there made
       * vkGetPhysicalDeviceImageFormatProperties fail a combination the spec
       * requires. CTS excludes a non-power-of-two pixel size only under
       * VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT (vktApiFeatureInfo.cpp,
       * isRequiredImageParameterCombination), and the combination it then
       * demands and we refused is
       * SPARSE_ALIASED_BIT|SPARSE_BINDING_BIT with SAMPLED usage --
       * captured on Neo. Every three-channel format failed
       * dEQP-VK.api.info.image_format_properties as soon as sparseBinding was
       * advertised. G17P stores a three-channel image padded to four, so it
       * is the generation that reaches this at all.
       */
      if ((pImageFormatInfo->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) &&
          !util_is_power_of_two_nonzero(
             vk_format_get_blocksize(pImageFormatInfo->format))) {

         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      }
   }

   if (pImageFormatInfo->flags & (VK_IMAGE_CREATE_SPARSE_ALIASED_BIT |
                                  VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                                  VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) &&
       (pImageFormatInfo->usage & VK_IMAGE_USAGE_HOST_TRANSFER_BIT))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   if (features == 0)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   if (pImageFormatInfo->tiling == VK_IMAGE_TILING_LINEAR &&
       pImageFormatInfo->type != VK_IMAGE_TYPE_2D)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   if (pImageFormatInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT &&
       pImageFormatInfo->type != VK_IMAGE_TYPE_2D)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   if (ycbcr_info && pImageFormatInfo->type != VK_IMAGE_TYPE_2D)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   /* From the Vulkan 1.3.279 spec:
    *
    *    VUID-VkImageCreateInfo-tiling-04121
    *
    *    "If tiling is VK_IMAGE_TILING_LINEAR, flags must not contain
    *    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT"
    *
    *    VUID-VkImageCreateInfo-imageType-00970
    *
    *    "If imageType is VK_IMAGE_TYPE_1D, flags must not contain
    *    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT"
    */
   if (pImageFormatInfo->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT &&
       (pImageFormatInfo->type == VK_IMAGE_TYPE_1D ||
        pImageFormatInfo->tiling == VK_IMAGE_TILING_LINEAR))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   /* From the Vulkan 1.3.279 spec:
    *
    *    VUID-VkImageCreateInfo-flags-09403
    *
    *    "If flags contains VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT, flags
    *    must not include VK_IMAGE_CREATE_SPARSE_ALIASED_BIT,
    *    VK_IMAGE_CREATE_SPARSE_BINDING_BIT, or
    *    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT"
    */
   if ((pImageFormatInfo->flags & VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT) &&
       (pImageFormatInfo->flags & (VK_IMAGE_CREATE_SPARSE_ALIASED_BIT |
                                   VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                                   VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   /* Multiplane formats are not supported with sparse residency. This has no
    * known use cases and is forbidden in other APIs.
    *
    * Neither is depth/stencil: this is a hardware limitation on G13. Hardware
    * support is added with G14, but that's not implemented yet. We could
    * emulate on G13 but it'd be fiddly. Fortunately, vkd3d-proton doesn't need
    * sparse depth, as RADV has the same limitation!
    */
   if ((ycbcr_info ||
        vk_format_is_depth_or_stencil(pImageFormatInfo->format)) &&
       (pImageFormatInfo->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT)) {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }

   const VkPhysicalDeviceImageDrmFormatModifierInfoEXT *drm_mod_info =
      vk_find_struct_const(pImageFormatInfo->pNext,
                           PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT);
   const bool interchange =
      pImageFormatInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT &&
      drm_mod_info &&
      drm_mod_info->drmFormatModifier == DRM_FORMAT_MOD_APPLE_INTERCHANGE;
   if (interchange &&
       !hk_interchange_supported(&pdev->dev, pImageFormatInfo->format,
                                 pImageFormatInfo->type, pImageFormatInfo->flags,
                                 pImageFormatInfo->usage, pImageFormatInfo->pNext))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   const bool g17p_compressed =
      (pdev->dev.chip == AGX_CHIP_G17P || agx_chip_g17_m5(pdev->dev.chip)) &&
      pImageFormatInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT &&
      drm_mod_info &&
      drm_mod_info->drmFormatModifier ==
         DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED;
   if (g17p_compressed &&
       !hk_g17p_shared_compression_supported(
          &pdev->dev, pImageFormatInfo->format, pImageFormatInfo->type,
          pImageFormatInfo->flags, pImageFormatInfo->usage,
          pImageFormatInfo->pNext))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   const uint32_t max_dim = pImageFormatInfo->type == VK_IMAGE_TYPE_3D
                               ? pdev->vk.properties.maxImageDimension3D
                               : 16384;
   VkExtent3D maxExtent;
   uint32_t maxArraySize;
   switch (pImageFormatInfo->type) {
   case VK_IMAGE_TYPE_1D:
      maxExtent = (VkExtent3D){max_dim, 1, 1};
      maxArraySize = 2048;
      break;
   case VK_IMAGE_TYPE_2D:
      maxExtent = (VkExtent3D){max_dim, max_dim, 1};
      maxArraySize = 2048;
      break;
   case VK_IMAGE_TYPE_3D:
      maxExtent = (VkExtent3D){max_dim, max_dim, max_dim};
      maxArraySize = 1;
      break;
   default:
      UNREACHABLE("Invalid image type");
   }
   if (pImageFormatInfo->tiling == VK_IMAGE_TILING_LINEAR)
      maxArraySize = 1;

   assert(util_is_power_of_two_nonzero(max_dim));
   uint32_t maxMipLevels = util_logbase2(max_dim) + 1;
   if (ycbcr_info != NULL || pImageFormatInfo->tiling == VK_IMAGE_TILING_LINEAR)
      maxMipLevels = 1;
   if (interchange || g17p_compressed) {
      maxMipLevels = 1;
      maxArraySize = 1;
   }

   VkSampleCountFlags sampleCounts = VK_SAMPLE_COUNT_1_BIT;
   if (pImageFormatInfo->tiling == VK_IMAGE_TILING_OPTIMAL &&
       pImageFormatInfo->type == VK_IMAGE_TYPE_2D && ycbcr_info == NULL &&
       (features & (VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT |
                    VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
       !(pImageFormatInfo->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)) {

      sampleCounts =
         VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT |
         (agx_chip_g17_m5(pdev->dev.chip) ? VK_SAMPLE_COUNT_8_BIT : 0);

      VkSampleCountFlags attachmentSampleCounts = UINT32_MAX;
      if (features & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT)
         attachmentSampleCounts &=
            pdev->vk.properties.framebufferColorSampleCounts;
      if (vk_format_has_depth(pImageFormatInfo->format))
         attachmentSampleCounts &=
            pdev->vk.properties.framebufferDepthSampleCounts;
      if (vk_format_has_stencil(pImageFormatInfo->format))
         attachmentSampleCounts &=
            pdev->vk.properties.framebufferStencilSampleCounts;
      sampleCounts &= attachmentSampleCounts;
   }

   /* Residency sample counts must agree with the feature bits. In
    * particular, exposing ordinary 2x/4x attachments does not enable sparse
    * multisampling. The sparse format query delegates to this query too.
    */
   if (pdev->dev.chip == AGX_CHIP_G17P &&
       (pImageFormatInfo->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT))
      sampleCounts &= VK_SAMPLE_COUNT_1_BIT;

   /* From the Vulkan 1.2.199 spec:
    *
    *    "VK_IMAGE_CREATE_EXTENDED_USAGE_BIT specifies that the image can be
    *    created with usage flags that are not supported for the format the
    *    image is created with but are supported for at least one format a
    *    VkImageView created from the image can have."
    *
    * If VK_IMAGE_CREATE_EXTENDED_USAGE_BIT is set, views can be created with
    * different usage than the image so we can't always filter on usage.
    * There is one exception to this below for storage.
    */
   const VkImageUsageFlags image_usage = pImageFormatInfo->usage;
   VkImageUsageFlags view_usage = image_usage;
   if (pImageFormatInfo->flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT)
      view_usage = 0;

   if (view_usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) {
      if (!(features & (VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))) {
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      }
   }

   u_foreach_bit(b, view_usage) {
      VkFormatFeatureFlags2 usage_features =
         vk_image_usage_to_format_features(1 << b);
      if (usage_features && !(features & usage_features))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }

   const VkExternalMemoryProperties *ext_mem_props = NULL;
   if (external_info != NULL && external_info->handleType != 0) {
      bool tiling_has_explicit_layout;
      switch (pImageFormatInfo->tiling) {
      case VK_IMAGE_TILING_LINEAR:
      case VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT:
         tiling_has_explicit_layout = true;
         break;
      case VK_IMAGE_TILING_OPTIMAL:
         tiling_has_explicit_layout = false;
         break;
      default:
         UNREACHABLE("Unsupported VkImageTiling");
      }

      switch (external_info->handleType) {
      case VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT:
         /* No special restrictions */
         if (tiling_has_explicit_layout) {
            /* With an explicit memory layout, we don't care which type of
             * fd the image belongs too. Both OPAQUE_FD and DMA_BUF are
             * interchangeable here.
             */
            ext_mem_props = &hk_dma_buf_mem_props;
         } else {
            ext_mem_props = &hk_opaque_fd_mem_props;
         }
         break;

      case VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT:
         if (!tiling_has_explicit_layout) {
            return vk_errorf(pdev, VK_ERROR_FORMAT_NOT_SUPPORTED,
                             "VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT "
                             "requires VK_IMAGE_TILING_LINEAR or "
                             "VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT");
         }
         ext_mem_props = &hk_dma_buf_mem_props;
         break;

      default:
         /* From the Vulkan 1.3.256 spec:
          *
          *    "If handleType is not compatible with the [parameters] in
          *    VkPhysicalDeviceImageFormatInfo2, then
          *    vkGetPhysicalDeviceImageFormatProperties2 returns
          *    VK_ERROR_FORMAT_NOT_SUPPORTED."
          */
         return vk_errorf(pdev, VK_ERROR_FORMAT_NOT_SUPPORTED,
                          "unsupported VkExternalMemoryTypeFlagBits 0x%x",
                          external_info->handleType);
      }
   }

   const unsigned plane_count =
      vk_format_get_plane_count(pImageFormatInfo->format);

   /* From the Vulkan 1.3.259 spec, VkImageCreateInfo:
    *
    *    VUID-VkImageCreateInfo-imageCreateFormatFeatures-02260
    *
    *    "If format is a multi-planar format, and if imageCreateFormatFeatures
    *    (as defined in Image Creation Limits) does not contain
    *    VK_FORMAT_FEATURE_DISJOINT_BIT, then flags must not contain
    *    VK_IMAGE_CREATE_DISJOINT_BIT"
    *
    * This is satisfied trivially because we support DISJOINT on all
    * multi-plane formats.  Also,
    *
    *    VUID-VkImageCreateInfo-format-01577
    *
    *    "If format is not a multi-planar format, and flags does not include
    *    VK_IMAGE_CREATE_ALIAS_BIT, flags must not contain
    *    VK_IMAGE_CREATE_DISJOINT_BIT"
    */
   if (plane_count == 1 &&
       !(pImageFormatInfo->flags & VK_IMAGE_CREATE_ALIAS_BIT) &&
       (pImageFormatInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   if (ycbcr_info &&
       (pImageFormatInfo->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   pImageFormatProperties->imageFormatProperties = (VkImageFormatProperties){
      .maxExtent = maxExtent,
      .maxMipLevels = maxMipLevels,
      .maxArrayLayers = maxArraySize,
      .sampleCounts = sampleCounts,
      .maxResourceSize = UINT32_MAX, /* TODO */
   };

   vk_foreach_struct(sType, s, pImageFormatProperties->pNext) {
      switch (sType) {
      case VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES: {
         VkExternalImageFormatProperties *p = s;
         /* From the Vulkan 1.3.256 spec:
          *
          *    "If handleType is 0, vkGetPhysicalDeviceImageFormatProperties2
          *    will behave as if VkPhysicalDeviceExternalImageFormatInfo was
          *    not present, and VkExternalImageFormatProperties will be
          *    ignored."
          *
          * This is true if and only if ext_mem_props == NULL
          */
         if (ext_mem_props != NULL)
            p->externalMemoryProperties = *ext_mem_props;
         break;
      }
      case VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES: {
         VkSamplerYcbcrConversionImageFormatProperties *ycbcr_props = s;
         ycbcr_props->combinedImageSamplerDescriptorCount = plane_count;
         break;
      }
      case VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY_EXT: {
         VkHostImageCopyDevicePerformanceQueryEXT *hic_props = s;

         /* Compare against the device-copy image, including private G17P
          * compression. hk_can_compress_format() is for DRM modifiers and
          * deliberately excludes private AGX3 compression.
          *
          * The query omits dimensions, levels, layers and samples. Choose a
          * shape that permits compression: if any such image can compress,
          * host-transfer images cannot promise the same layout.
          */
         const VkExternalMemoryImageCreateInfo external_create_info = {
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            .pNext = pImageFormatInfo->pNext,
            .handleTypes = external_info ? external_info->handleType : 0,
         };
         const VkImageCreateInfo device_copy_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &external_create_info,
            .flags = pImageFormatInfo->flags,
            .imageType = pImageFormatInfo->type,
            .format = pImageFormatInfo->format,
            .extent = {64, 64, 1},
            .mipLevels = 1,
            .arrayLayers = agx_is_m4_pro(&pdev->dev) &&
                                 (pImageFormatInfo->flags &
                                  VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)
                              ? 6 : 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = pImageFormatInfo->tiling,
            .usage = (pImageFormatInfo->usage &
                      ~VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT) |
                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
         };
         bool can_compress = false;
         if (pImageFormatInfo->tiling == VK_IMAGE_TILING_OPTIMAL) {
            for (unsigned plane = 0; plane < plane_count; ++plane) {
               can_compress |= hk_can_compress_create_info(
                  &pdev->dev, plane, &device_copy_info, false, false);
            }
         }

         hic_props->optimalDeviceAccess = hic_props->identicalMemoryLayout =
            !can_compress;
         break;
      }
      default:
         vk_debug_ignored_stype(sType);
         break;
      }
   }

   return VK_SUCCESS;
}

static VkSparseImageFormatProperties
hk_fill_sparse_image_fmt_props(enum pipe_format format, unsigned samples,
                               VkImageAspectFlags aspects, bool sliced_3d)
{
   /* Apple tile sizes are exactly 16KiB. The Vulkan standard block sizes are
    * sized to be exactly 64KiB. Fortunately, they correspond directly to the
    * Apple sizes (except for MSAA 2x), just doubled in each dimensions. Our
    * sparse binding code gangs together 4 hardware tiles into an API tile. We
    * just need to derive the correct size here.
    */
   unsigned blocksize_B = util_format_get_blocksize(format) * samples;
   struct ail_tile ail_size = ail_get_max_tile_size(blocksize_B);

   VkExtent3D granularity = {
      ail_size.width_el * 2 * util_format_get_blockwidth(format),
      ail_size.height_el * 2 * util_format_get_blockheight(format),
      1,
   };

   return (VkSparseImageFormatProperties){
      .aspectMask = aspects,
      .flags = VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT |
               (sliced_3d ? VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT : 0),
      .imageGranularity = granularity,
   };
}

VKAPI_ATTR void VKAPI_CALL
hk_GetPhysicalDeviceSparseImageFormatProperties2(
   VkPhysicalDevice physicalDevice,
   const VkPhysicalDeviceSparseImageFormatInfo2 *pFormatInfo,
   uint32_t *pPropertyCount, VkSparseImageFormatProperties2 *pProperties)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   VkResult result;

   /* Check if the given format info is valid first before returning sparse
    * props.  The easiest way to do this is to just call
    * hk_GetPhysicalDeviceImageFormatProperties2()
    */
   const VkPhysicalDeviceImageFormatInfo2 img_fmt_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = pFormatInfo->format,
      .type = pFormatInfo->type,
      .tiling = pFormatInfo->tiling,
      .usage = pFormatInfo->usage,
      .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
               VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
   };

   VkImageFormatProperties2 img_fmt_props2 = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      .pNext = NULL,
   };

   result = hk_GetPhysicalDeviceImageFormatProperties2(
      physicalDevice, &img_fmt_info, &img_fmt_props2);
   if (result != VK_SUCCESS) {
      *pPropertyCount = 0;
      return;
   }

   const VkImageFormatProperties *props = &img_fmt_props2.imageFormatProperties;
   if (!(pFormatInfo->samples & props->sampleCounts)) {
      *pPropertyCount = 0;
      return;
   }

   VK_OUTARRAY_MAKE_TYPED(VkSparseImageFormatProperties2, out, pProperties,
                          pPropertyCount);

   VkImageAspectFlags aspects = vk_format_aspects(pFormatInfo->format);

   vk_outarray_append_typed(VkSparseImageFormatProperties2, &out, props)
   {
      props->properties = hk_fill_sparse_image_fmt_props(
         vk_format_to_pipe_format(pFormatInfo->format), pFormatInfo->samples,
         aspects, pdev->dev.chip == AGX_CHIP_G17P &&
                     pFormatInfo->type == VK_IMAGE_TYPE_3D);
   }
}

static bool
hk_can_compress_create_info(const struct agx_device *dev, unsigned plane,
                            const VkImageCreateInfo *info,
                            bool float_image_atomics, bool srgb_pair_views)
{
   /* AGX3: G17P compresses every image hk_g17p_compression.h admits. Other
    * AGX3 chips stay plain until their packets are recovered. */
   const bool m5_storage =
      hk_m5_storage_compression_eligible(dev, plane, info, float_image_atomics);
   bool allow_agx3 =
      hk_m5_msaa_color_compression_eligible(
         dev->chip, plane, info,
         debug_get_option_m5_msaa_color_compression()) ||
      hk_m5_depth_msaa_compression_eligible(
         dev->chip, plane, info, dev->params.features,
         debug_get_option_m5_depth_msaa_compression()) ||
      hk_m5_depth_msaa_transfer_eligible(
         dev->chip, plane, info, dev->params.features,
         debug_get_option_m5_depth_msaa_compression(),
         debug_get_option_m5_depth_msaa_transfer()) ||
      m5_storage || hk_hal200_compression_eligible(dev->chip, plane, info,
                                                  agx_is_m4_pro(dev)) ||
      hk_g17p_shared_compression_eligible(dev->chip, plane, info) ||
      hk_m5_color_compression_eligible(dev->chip, plane, info) ||
      hk_m5_depth_compression_eligible(dev->chip, plane, info);

   if (allow_agx3 && !vk_format_is_depth_or_stencil(info->format) &&
       !hk_g17p_store_class_matches(dev, info->format))
      allow_agx3 = false;

   return hk_can_compress(dev, info->format, plane, info->extent.width,
                          info->extent.height, info->samples, info->flags,
                          m5_storage ? info->usage & ~VK_IMAGE_USAGE_STORAGE_BIT
                                     : info->usage,
                          info->pNext, allow_agx3, srgb_pair_views);
}

static enum ail_tiling
hk_map_tiling(struct hk_device *dev, const VkImageCreateInfo *info,
              uint64_t modifier)
{
   switch (info->tiling) {
   case VK_IMAGE_TILING_LINEAR:
      return AIL_TILING_LINEAR;

   case VK_IMAGE_TILING_OPTIMAL:
      return AIL_TILING_GPU;

   case VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT:
      return ail_drm_modifier_to_tiling(modifier);

   default:
      UNREACHABLE("invalid tiling");
   }
}

static enum ail_tiling
hk_map_compression(struct hk_device *dev, const VkImageCreateInfo *info,
                   unsigned plane, uint64_t modifier)
{
   switch (info->tiling) {
   case VK_IMAGE_TILING_LINEAR:
      return false;

   case VK_IMAGE_TILING_OPTIMAL:
      return hk_can_compress_create_info(
         &dev->dev, plane, info, hk_float_image_atomics(dev),
         hk_physical_device_instance(hk_device_physical(dev))
            ->drirc.misc.mutable_format_srgb_pair);

   case VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT:
      return ail_is_drm_modifier_compressed(modifier);

   default:
      UNREACHABLE("invalid tiling");
   }
}

static uint32_t
modifier_get_score(uint64_t mod)
{
   switch (mod) {
   case DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED:
      return 10;

   /* Display-native compressed layout, scanned out by DCP directly.  Below
    * GPU_TILED too, so a compositor's main tranche built from this driver's
    * modifier list keeps choosing a GPU-tiled layout: GPU_TILED_COMPRESSED
    * where hk_can_compress_create_info() admits the image (G17P colour
    * images, hk_g17p_shared_compression_eligible()), GPU_TILED otherwise.
    * Interchange then wins only against LINEAR, i.e. in a display (scanout)
    * list that cannot take GPU tiling.  Such a list must come from the
    * display plane (IN_FORMATS) or the compositor, never from the GPU
    * capability list.  A linear-only G17P plane thus stays linear. */
   case DRM_FORMAT_MOD_APPLE_INTERCHANGE:
      return 3;

   case DRM_FORMAT_MOD_APPLE_GPU_TILED:
      return 5;

   case DRM_FORMAT_MOD_LINEAR:
      return 1;

   default:
      return 0;
   }
}

static uint64_t
choose_drm_format_mod(struct hk_device *dev, uint8_t plane_count,
                      const VkImageCreateInfo *info, uint32_t modifier_count,
                      const uint64_t *modifiers)
{
   uint64_t best_mod = UINT64_MAX;
   uint32_t best_score = 0;
   bool can_compress = true;

   for (uint8_t plane = 0; plane < plane_count; plane++) {
      if (!hk_can_compress_create_info(&dev->dev, plane, info,
                                       hk_float_image_atomics(dev), false))
         can_compress = false;
   }

   const bool interchange =
      plane_count == 1 &&
      !(info->usage & VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT) &&
      hk_interchange_create_info(dev, info);

   for (uint32_t i = 0; i < modifier_count; ++i) {
      if (!can_compress &&
          modifiers[i] == DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED)
         continue;
      if (!interchange &&
          modifiers[i] == DRM_FORMAT_MOD_APPLE_INTERCHANGE)
         continue;

      uint32_t score = modifier_get_score(modifiers[i]);
      if (score > best_score) {
         best_mod = modifiers[i];
         best_score = score;
      }
   }

   if (best_score > 0)
      return best_mod;
   else
      return DRM_FORMAT_MOD_INVALID;
}

static VkResult
hk_image_init(struct hk_device *dev, struct hk_image *image,
              const VkImageCreateInfo *pCreateInfo)
{
   vk_image_init(&dev->vk, &image->vk, pCreateInfo);

   if ((image->vk.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
       image->vk.samples > 1) {
      image->vk.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
      image->vk.stencil_usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
   }

   if (image->vk.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
      image->vk.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
   if (image->vk.usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
      image->vk.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

   image->plane_count = vk_format_get_plane_count(pCreateInfo->format);
   image->disjoint = image->plane_count > 1 &&
                     (pCreateInfo->flags & VK_IMAGE_CREATE_DISJOINT_BIT);

   /* We do not support interleaved depth/stencil. Instead, we decompose to
    * a depth plane and a stencil plane.
    */
   if (hk_split_depth_stencil(image->vk.format)) {
      image->plane_count = 2;
   }

   if (image->vk.create_flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) {
      /* Sparse multiplane is not supported. Sparse depth/stencil not supported
       * on G13 so we're fine there too.
       */
      assert(image->plane_count == 1);
   }

   const struct VkImageDrmFormatModifierExplicitCreateInfoEXT
      *mod_explicit_info = NULL;

   if (pCreateInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT) {
      assert(!image->vk.wsi_legacy_scanout);
      mod_explicit_info = vk_find_struct_const(
         pCreateInfo->pNext,
         IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT);

      uint64_t modifier = DRM_FORMAT_MOD_INVALID;

      if (mod_explicit_info) {
         modifier = mod_explicit_info->drmFormatModifier;
      } else {
         const struct VkImageDrmFormatModifierListCreateInfoEXT *mod_list_info =
            vk_find_struct_const(
               pCreateInfo->pNext,
               IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT);

         modifier = choose_drm_format_mod(dev, image->plane_count, pCreateInfo,
                                          mod_list_info->drmFormatModifierCount,
                                          mod_list_info->pDrmFormatModifiers);

      }

      assert(modifier != DRM_FORMAT_MOD_INVALID);
      assert(image->vk.drm_format_mod == DRM_FORMAT_MOD_INVALID);
      image->vk.drm_format_mod = modifier;
   }

   if (mod_explicit_info &&
       (mod_explicit_info->drmFormatModifierPlaneCount != image->plane_count ||
        !mod_explicit_info->pPlaneLayouts)) {
      return vk_errorf(
         dev, VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
         "DRM modifier provides %u planes for a %u-plane image",
         mod_explicit_info->drmFormatModifierPlaneCount, image->plane_count);
   }

   if (image->vk.drm_format_mod == DRM_FORMAT_MOD_APPLE_INTERCHANGE) {
      if (!hk_interchange_create_info(dev, pCreateInfo))
         return vk_errorf(dev, VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
                          "interchange modifier on an ineligible image");

      /* The row pitch is part of the modifier's contract with appledrm. */
      const uint64_t pitch_B = DIV_ROUND_UP(pCreateInfo->extent.width, 16) *
                               256 * vk_format_get_blocksize(pCreateInfo->format);
      if (mod_explicit_info &&
          mod_explicit_info->pPlaneLayouts[0].rowPitch != pitch_B)
         return vk_errorf(dev, VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
                          "interchange plane row pitch 0x%" PRIx64 " != 0x%" PRIx64,
                          mod_explicit_info->pPlaneLayouts[0].rowPitch, pitch_B);
   }

   /* G17P and M5 share compressed colour images only in the shape both sides
    * allocate identically (hk_g17p_shared_compression_eligible()). */
   if (image->vk.drm_format_mod == DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED &&
       (dev->dev.chip == AGX_CHIP_G17P || agx_chip_g17_m5(dev->dev.chip)) &&
       !(hk_g17p_shared_compression_supported(
            &dev->dev, pCreateInfo->format, pCreateInfo->imageType,
            pCreateInfo->flags, pCreateInfo->usage, pCreateInfo->pNext) &&
         hk_g17p_shared_compression_eligible(dev->dev.chip, 0, pCreateInfo)))
      return vk_errorf(dev, VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
                       "compressed modifier on an ineligible image");

   const struct vk_format_ycbcr_info *ycbcr_info =
      vk_format_get_ycbcr_info(pCreateInfo->format);
   uint64_t implicit_plane_cursor_B = 0;

   for (uint8_t plane = 0; plane < image->plane_count; plane++) {
      VkFormat format =
         ycbcr_info ? ycbcr_info->planes[plane].format : pCreateInfo->format;

      format = hk_depth_stencil_plane_format(format, plane);

      const uint8_t width_scale =
         ycbcr_info ? ycbcr_info->planes[plane].denominator_scales[0] : 1;
      const uint8_t height_scale =
         ycbcr_info ? ycbcr_info->planes[plane].denominator_scales[1] : 1;

      const enum pipe_format p_format = hk_format_to_pipe_format(format);
      const enum pipe_format layout_format =
         (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
               pCreateInfo->tiling == VK_IMAGE_TILING_OPTIMAL
            ? hk_padded_image_format(p_format)
            : p_format;
      uint32_t linear_stride_B = 0;
      if (mod_explicit_info &&
          image->vk.drm_format_mod == DRM_FORMAT_MOD_LINEAR) {
         const uint64_t minimum_stride_B = util_format_get_stride(
            p_format, DIV_ROUND_UP(pCreateInfo->extent.width, width_scale));
         if (!hk_image_explicit_linear_stride(
                mod_explicit_info->pPlaneLayouts[plane].rowPitch,
                minimum_stride_B, &linear_stride_B)) {
            return vk_errorf(
               dev, VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
               "invalid linear DRM plane %u row pitch 0x%" PRIx64, plane,
               mod_explicit_info->pPlaneLayouts[plane].rowPitch);
         }
      }

      bool compressed =
         hk_map_compression(dev, pCreateInfo, plane, image->vk.drm_format_mod);

      image->planes[plane].layout = (struct ail_layout){
         .policy = ail_layout_policy_for_descriptor_generation(
            agx_chip_descriptor_generation(dev->dev.chip)),
         .tiling = hk_map_tiling(dev, pCreateInfo, image->vk.drm_format_mod),
         .compressed = compressed,
         .mipmapped_z = pCreateInfo->imageType == VK_IMAGE_TYPE_3D,
         .sparse_residency =
            pCreateInfo->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
         .format = layout_format,

         .linear_stride_B = linear_stride_B,

         .width_px = pCreateInfo->extent.width / width_scale,
         .height_px = pCreateInfo->extent.height / height_scale,
         .depth_px = MAX2(pCreateInfo->extent.depth, pCreateInfo->arrayLayers),

         .levels = pCreateInfo->mipLevels,
         .sample_count_sa = pCreateInfo->samples,
         .writeable_image = !compressed,

         /* TODO: Maybe optimize this, our GL driver doesn't bother though */
         .renderable = true,
      };

      ail_make_miptree(&image->planes[plane].layout);

      uint64_t plane_end_B;
      if (mod_explicit_info) {
         const uint64_t plane_offset_B =
            mod_explicit_info->pPlaneLayouts[plane].offset;
         if (!hk_image_explicit_plane_range(plane_offset_B,
                                            image->planes[plane].layout.size_B,
                                            &plane_end_B)) {
            return vk_errorf(
               dev, VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
               "invalid DRM plane %u range at 0x%" PRIx64, plane,
               plane_offset_B);
         }
         image->planes[plane].offset_B = plane_offset_B;

         /* The offset is relative to the image's memory binding, or for a
          * disjoint image to the plane's own binding, where planes cannot
          * overlap one another.
          */
         for (uint8_t prior = 0; !image->disjoint && prior < plane; prior++) {
            if (hk_image_plane_ranges_overlap(
                   plane_offset_B, image->planes[plane].layout.size_B,
                   image->planes[prior].offset_B,
                   image->planes[prior].layout.size_B)) {
               return vk_errorf(
                  dev, VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
                  "DRM plane %u overlaps plane %u", plane, prior);
            }
         }
      } else if (image->disjoint) {
         image->planes[plane].offset_B = 0;
      } else if (!hk_image_implicit_plane_range(
                    /* Sparse binds map each plane in whole pages. */
                    (pCreateInfo->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT)
                       ? align64(implicit_plane_cursor_B, HK_SPARSE_ALIGN_B)
                       : implicit_plane_cursor_B,
                    image->planes[plane].layout.size_B,
                    &image->planes[plane].offset_B, &plane_end_B)) {
         return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "image plane layout exceeds the address space");
      }

      if (!image->disjoint)
         implicit_plane_cursor_B = MAX2(implicit_plane_cursor_B, plane_end_B);
   }

   return VK_SUCCESS;
}

static VkResult
hk_image_plane_alloc_vma(struct hk_device *dev, struct hk_image_plane *plane,
                         VkImageCreateFlags create_flags)
{
   const bool sparse_bound = create_flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT;
   const bool sparse_resident =
      create_flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
   assert(sparse_bound || !sparse_resident);

   if (sparse_bound) {
      size_t size = align(plane->layout.size_B, HK_SPARSE_ALIGN_B);
      plane->va = agx_va_alloc(&dev->dev, size, AIL_PAGESIZE, 0, 0);
      if (!plane->va) {
         return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "Sparse VMA allocation failed");
      }
      plane->addr = plane->va->addr;

      /* Bind scratch pages to discard writes, including from lowered software
       * texture atomics. Reads will use the hardware texture unit sparse
       * handling to properly handle residency queries.
       *
       * In the future we could optimize this out using the PBE sparse support
       * but that needs more reverse-engineering.
       */
      VkResult result = hk_reserve_sparse_scratch(dev, size);
      if (result != VK_SUCCESS) {
         agx_va_free(&dev->dev, plane->va, true);
         plane->va = NULL;
         plane->addr = 0;
         return result;
      }
      plane->sparse_scratch_B = agx_chip_g17_m5(dev->dev.chip) ? size : 0;
      result = hk_bind_scratch(dev, plane->va, 0, size);
      if (result != VK_SUCCESS) {
         hk_release_sparse_scratch(dev, plane->sparse_scratch_B);
         plane->sparse_scratch_B = 0;
         agx_va_free(&dev->dev, plane->va, true);
         plane->va = NULL;
         plane->addr = 0;
         return result;
      }
      /* Bound memory is reached through this plane's own address as well
       * (hk_cdm_alias_scope()); hk_image_plane_finish() drops the count. */
      p_atomic_inc(&dev->va_alias_resources);
   }

   if (sparse_resident) {
      plane->sparse_map =
         agx_bo_create(&dev->dev, plane->layout.sparse_table_size_B,
                       AIL_PAGESIZE, 0, "Sparse map");
      if (!plane->sparse_map)
         return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "Sparse residency map allocation failed");

      /* Zero-initialize the sparse map. This ensures all tiles are disabled,
       * which provides correct behaviour for unmapped tiles.
       */
      void *map = agx_bo_map(plane->sparse_map);
      if (!map) {
         agx_bo_unreference(&dev->dev, plane->sparse_map);
         plane->sparse_map = NULL;
         return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "Sparse residency map mapping failed");
      }
      memset(map, 0, plane->layout.sparse_table_size_B);
   }

   return VK_SUCCESS;
}

static void
hk_image_plane_finish(struct hk_device *dev, struct hk_image_plane *plane,
                      VkImageCreateFlags create_flags,
                      const VkAllocationCallbacks *pAllocator)
{
   if (plane->va) {
      agx_va_free(&dev->dev, plane->va, true);
      plane->va = NULL;
      p_atomic_dec(&dev->va_alias_resources);
   }
   hk_release_sparse_scratch(dev, plane->sparse_scratch_B);
   plane->sparse_scratch_B = 0;

   agx_bo_unreference(&dev->dev, plane->sparse_map);
}

static void
hk_image_finish(struct hk_device *dev, struct hk_image *image,
                const VkAllocationCallbacks *pAllocator)
{
   for (uint8_t plane = 0; plane < image->plane_count; plane++) {
      hk_image_plane_finish(dev, &image->planes[plane], image->vk.create_flags,
                            pAllocator);
   }

   vk_image_finish(&image->vk);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateImage(VkDevice _device, const VkImageCreateInfo *pCreateInfo,
               const VkAllocationCallbacks *pAllocator, VkImage *pImage)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct hk_image *image;
   VkResult result;

   if (wsi_common_is_swapchain_image(pCreateInfo))
      return wsi_common_create_swapchain_image(&pdev->wsi_device, pCreateInfo,
                                               pImage);

   image = vk_zalloc2(&dev->vk.alloc, pAllocator, sizeof(*image), 8,
                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!image)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = hk_image_init(dev, image, pCreateInfo);
   if (result != VK_SUCCESS) {
      hk_image_finish(dev, image, pAllocator);
      vk_free2(&dev->vk.alloc, pAllocator, image);
      return result;
   }

   /* Frame anatomy profiler: the compression decision for every image large
    * enough to matter for bandwidth. */
   if (unlikely(dev->frame_anatomy) && pCreateInfo->extent.width >= 128 &&
       pCreateInfo->extent.height >= 128) {
      /* The view formats a mutable image lists, which the compression
       * decision depends on. */
      char views[256] = "";
      const VkImageFormatListCreateInfo *list =
         vk_find_struct_const(pCreateInfo->pNext, IMAGE_FORMAT_LIST_CREATE_INFO);
      for (unsigned i = 0; list && i < list->viewFormatCount; ++i) {
         const size_t len = strlen(views);
         snprintf(views + len, sizeof(views) - len, "%s%s", i ? "," : " views=",
                  util_format_short_name(
                     hk_format_to_pipe_format(list->pViewFormats[i])));
      }
      hk_frame_anatomy_note(
         dev,
         "image %ux%ux%u fmt=%s samples=%u mips=%u layers=%u usage=0x%x "
         "flags=0x%x tiling=%u pnext=%u compressed=%u%s",
         pCreateInfo->extent.width, pCreateInfo->extent.height,
         pCreateInfo->extent.depth,
         util_format_short_name(hk_format_to_pipe_format(pCreateInfo->format)),
         pCreateInfo->samples, pCreateInfo->mipLevels,
         pCreateInfo->arrayLayers, pCreateInfo->usage, pCreateInfo->flags,
         pCreateInfo->tiling, pCreateInfo->pNext != NULL,
         image->planes[0].layout.compressed, views);
   }

   for (uint8_t plane = 0; plane < image->plane_count; plane++) {
      result = hk_image_plane_alloc_vma(dev, &image->planes[plane],
                                        image->vk.create_flags);
      if (result != VK_SUCCESS) {
         hk_image_finish(dev, image, pAllocator);
         vk_free2(&dev->vk.alloc, pAllocator, image);
         return result;
      }
   }

   *pImage = hk_image_to_handle(image);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyImage(VkDevice device, VkImage _image,
                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_image, image, _image);

   if (!image)
      return;

   hk_image_finish(dev, image, pAllocator);
   vk_free2(&dev->vk.alloc, pAllocator, image);
}

static void
hk_image_plane_add_req(struct hk_image_plane *plane, bool sparse,
                       uint64_t *size_B, uint32_t *align_B)
{
   unsigned plane_align_B = sparse ? HK_SPARSE_ALIGN_B : HK_IMAGE_PLANE_ALIGN_B;
   assert(util_is_power_of_two_or_zero64(*align_B));
   assert(util_is_power_of_two_or_zero64(plane_align_B));

   *align_B = MAX2(*align_B, plane_align_B);
   assert(plane->offset_B <= UINT64_MAX - plane->layout.size_B);
   *size_B = MAX2(*size_B, plane->offset_B + plane->layout.size_B);
}

/* All of G17P's memory is one unified heap, and DXVK suballocates every
 * resource the driver does not prefer to have dedicated into chunks of up to
 * 256 MiB, packing even large render targets one to three per chunk; the
 * Subnautica menu leaves at least 31% of those chunks unused. Prefer (never
 * require) a dedicated allocation for attachments and for large images so
 * they get exactly their size. Sparse and disjoint images cannot take one,
 * and images that may be exported or imported keep their answer. */
#define HK_DEDICATED_IMAGE_MIN_B (4ull << 20)

static bool
hk_image_prefers_dedicated(const struct hk_image *image, bool sparse,
                           uint64_t size_B)
{
   if (sparse || image->disjoint || image->vk.external_handle_types ||
       image->vk.tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT)
      return false;

   const VkImageUsageFlags attachment =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   return (image->vk.usage & attachment) || size_B >= HK_DEDICATED_IMAGE_MIN_B;
}

static void
hk_get_image_memory_requirements(struct hk_device *dev, struct hk_image *image,
                                 VkImageAspectFlags aspects,
                                 VkMemoryRequirements2 *pMemoryRequirements)
{
   struct hk_physical_device *pdev = hk_device_physical(dev);
   uint32_t memory_types = (1 << pdev->mem_type_count) - 1;
   bool sparse =
      image->vk.create_flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                                VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT);

   uint64_t size_B = 0;
   uint32_t align_B = 0;
   if (image->disjoint) {
      uint8_t plane = hk_image_aspects_to_plane(image, aspects);
      hk_image_plane_add_req(&image->planes[plane], sparse, &size_B, &align_B);
   } else {
      for (unsigned plane = 0; plane < image->plane_count; plane++)
         hk_image_plane_add_req(&image->planes[plane], sparse, &size_B,
                                &align_B);
   }

   /* For sparse binding, we need to pad to the standard alignment so we don't
    * clobber over things when we bind memory.
    */
   if (sparse) {
      size_B = align64(size_B, align_B);
   }

   pMemoryRequirements->memoryRequirements.memoryTypeBits = memory_types;
   pMemoryRequirements->memoryRequirements.alignment = align_B;
   pMemoryRequirements->memoryRequirements.size = size_B;

   vk_foreach_struct(sType, ext, pMemoryRequirements->pNext) {
      switch (sType) {
      case VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS: {
         VkMemoryDedicatedRequirements *dedicated = (void *)ext;
         dedicated->prefersDedicatedAllocation =
            hk_image_prefers_dedicated(image, sparse, size_B);
         dedicated->requiresDedicatedAllocation = false;
         break;
      }
      default:
         vk_debug_ignored_stype(sType);
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_GetImageMemoryRequirements2(VkDevice device,
                               const VkImageMemoryRequirementsInfo2 *pInfo,
                               VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_image, image, pInfo->image);

   const VkImagePlaneMemoryRequirementsInfo *plane_info =
      vk_find_struct_const(pInfo->pNext, IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO);
   const VkImageAspectFlags aspects =
      image->disjoint ? plane_info->planeAspect : image->vk.aspects;

   hk_get_image_memory_requirements(dev, image, aspects, pMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDeviceImageMemoryRequirements(VkDevice device,
                                    const VkDeviceImageMemoryRequirements *pInfo,
                                    VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   ASSERTED VkResult result;
   struct hk_image image = {0};

   result = hk_image_init(dev, &image, pInfo->pCreateInfo);
   assert(result == VK_SUCCESS);

   const VkImageAspectFlags aspects =
      image.disjoint ? pInfo->planeAspect : image.vk.aspects;

   hk_get_image_memory_requirements(dev, &image, aspects, pMemoryRequirements);

   hk_image_finish(dev, &image, NULL);
}

static VkSparseImageMemoryRequirements
hk_fill_sparse_image_memory_reqs(const struct ail_layout *layout,
                                 VkImageAspectFlags aspects, bool sliced_3d)
{
   VkSparseImageFormatProperties sparse_format_props =
      hk_fill_sparse_image_fmt_props(layout->format, layout->sample_count_sa,
                                     aspects, sliced_3d);

   unsigned tail_level = layout->mip_tail_first_lod;
   assert(tail_level <= layout->levels);
   VkSparseImageMemoryRequirements sparse_memory_reqs = {
      .formatProperties = sparse_format_props,
      .imageMipTailFirstLod = layout->mip_tail_first_lod,
      .imageMipTailStride = 0,
   };

   /* imageMipTailSize must be aligned to the sparse block size (65k). This
    * requires us to manage the miptail manually, because 16k is the actual
    * hardware alignment here so we need to give the illusion of extra
    * padding. Annoying!
    */
   if (tail_level == 0) {
      sparse_memory_reqs.imageMipTailSize =
         align(layout->size_B, HK_SPARSE_ALIGN_B);

      sparse_memory_reqs.imageMipTailOffset = 0;
   } else if (tail_level < layout->levels) {
      sparse_memory_reqs.imageMipTailSize =
         align(layout->mip_tail_stride * layout->depth_px, HK_SPARSE_ALIGN_B);

      /* TODO: sparse metadata */
      sparse_memory_reqs.imageMipTailOffset = HK_MIP_TAIL_START_OFFSET;
   } else {
      sparse_memory_reqs.imageMipTailSize = 0;
      sparse_memory_reqs.imageMipTailOffset = HK_MIP_TAIL_START_OFFSET;
   }

   return sparse_memory_reqs;
}

static void
hk_get_image_sparse_memory_requirements(
   struct hk_device *dev, struct hk_image *image, VkImageAspectFlags aspects,
   uint32_t *pSparseMemoryRequirementCount,
   VkSparseImageMemoryRequirements2 *pSparseMemoryRequirements)
{
   VK_OUTARRAY_MAKE_TYPED(VkSparseImageMemoryRequirements2, out,
                          pSparseMemoryRequirements,
                          pSparseMemoryRequirementCount);

   /* From the Vulkan 1.3.279 spec:
    *
    *    "The sparse image must have been created using the
    *    VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT flag to retrieve valid sparse
    *    image memory requirements."
    */
   if (!(image->vk.create_flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT))
      return;

   /* We don't support multiplane sparse for now */
   if (image->plane_count > 1)
      return;

   vk_outarray_append_typed(VkSparseImageMemoryRequirements2, &out, reqs)
   {
      reqs->memoryRequirements =
         hk_fill_sparse_image_memory_reqs(
            &image->planes[0].layout, aspects,
            dev->dev.chip == AGX_CHIP_G17P &&
               image->vk.image_type == VK_IMAGE_TYPE_3D);
   };
}

VKAPI_ATTR void VKAPI_CALL
hk_GetImageSparseMemoryRequirements2(
   VkDevice device, const VkImageSparseMemoryRequirementsInfo2 *pInfo,
   uint32_t *pSparseMemoryRequirementCount,
   VkSparseImageMemoryRequirements2 *pSparseMemoryRequirements)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_image, image, pInfo->image);

   const VkImageAspectFlags aspects = image->vk.aspects;

   hk_get_image_sparse_memory_requirements(dev, image, aspects,
                                           pSparseMemoryRequirementCount,
                                           pSparseMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDeviceImageSparseMemoryRequirements(
   VkDevice device, const VkDeviceImageMemoryRequirements *pInfo,
   uint32_t *pSparseMemoryRequirementCount,
   VkSparseImageMemoryRequirements2 *pSparseMemoryRequirements)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   ASSERTED VkResult result;
   struct hk_image image = {0};

   result = hk_image_init(dev, &image, pInfo->pCreateInfo);
   assert(result == VK_SUCCESS);

   const VkImageAspectFlags aspects =
      image.disjoint ? pInfo->planeAspect : image.vk.aspects;

   hk_get_image_sparse_memory_requirements(dev, &image, aspects,
                                           pSparseMemoryRequirementCount,
                                           pSparseMemoryRequirements);

   hk_image_finish(dev, &image, NULL);
}

static void
hk_get_image_subresource_layout(UNUSED struct hk_device *dev,
                                struct hk_image *image,
                                const VkImageSubresource2KHR *pSubresource,
                                VkSubresourceLayout2KHR *pLayout)
{
   const VkImageSubresource *isr = &pSubresource->imageSubresource;

   const uint8_t p = hk_image_aspects_to_plane(image, isr->aspectMask);
   const struct hk_image_plane *plane = &image->planes[p];

   uint64_t offset_B = plane->offset_B;
   offset_B +=
      ail_get_layer_level_B(&plane->layout, isr->arrayLayer, isr->mipLevel);

   bool is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D;

   pLayout->subresourceLayout = (VkSubresourceLayout){
      .offset = offset_B,
      .size = ail_get_level_size_B(&plane->layout, isr->mipLevel),

      /* From the spec:
       *
       *     It is legal to call vkGetImageSubresourceLayout2KHR with a image
       *     created with tiling equal to VK_IMAGE_TILING_OPTIMAL, but the
       * members of VkSubresourceLayout2KHR::subresourceLayout will have
       * undefined values in this case.
       *
       * So don't collapse with mips.
       */
      .rowPitch = isr->mipLevel
                     ? 0
                     : ail_get_wsi_stride_B(&plane->layout, isr->mipLevel),
      .arrayPitch = is_3d ? 0 : plane->layout.layer_stride_B,
      .depthPitch = is_3d ? plane->layout.layer_stride_B : 0,
   };

   VkSubresourceHostMemcpySizeEXT *memcpy_size =
      vk_find_struct(pLayout, SUBRESOURCE_HOST_MEMCPY_SIZE_EXT);
   if (memcpy_size) {
      memcpy_size->size = pLayout->subresourceLayout.size;
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_GetImageSubresourceLayout2KHR(VkDevice device, VkImage _image,
                                 const VkImageSubresource2KHR *pSubresource,
                                 VkSubresourceLayout2KHR *pLayout)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_image, image, _image);

   hk_get_image_subresource_layout(dev, image, pSubresource, pLayout);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDeviceImageSubresourceLayoutKHR(
   VkDevice device, const VkDeviceImageSubresourceInfoKHR *pInfo,
   VkSubresourceLayout2KHR *pLayout)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   ASSERTED VkResult result;
   struct hk_image image = {0};

   result = hk_image_init(dev, &image, pInfo->pCreateInfo);
   assert(result == VK_SUCCESS);

   hk_get_image_subresource_layout(dev, &image, pInfo->pSubresource, pLayout);

   hk_image_finish(dev, &image, NULL);
}

static bool
hk_image_plane_bind(struct hk_device *dev, struct hk_image_plane *plane,
                    struct hk_device_memory *mem, uint64_t memory_offset_B)
{
   if (plane->va) {
#if 0
      agx_bo_bind_vma(dev->ws_dev,
                             mem->bo,
                             plane->addr,
                             plane->va,
                             *offset_B,
                             plane->nil.pte_kind);
#endif
      UNREACHABLE("todo");
   } else {
      /* The plane must lie within the memory object, which may be smaller
       * than the BO backing it (BOs come page-rounded and from a cache) */
      const uint64_t mem_size_B =
         mem->vk.size ? MIN2(mem->bo->size, mem->vk.size) : mem->bo->size;
      uint64_t bo_offset_B;
      if (!hk_image_plane_binding(mem->bo->va->addr, mem_size_B,
                                  memory_offset_B, plane->offset_B,
                                  plane->layout.size_B, &plane->addr,
                                  &bo_offset_B))
         return false;

      plane->map = agx_bo_map(mem->bo) + bo_offset_B;
      plane->rem = mem_size_B - bo_offset_B;
   }

   return true;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_BindImageMemory2(VkDevice device, uint32_t bindInfoCount,
                    const VkBindImageMemoryInfo *pBindInfos)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   for (uint32_t i = 0; i < bindInfoCount; ++i) {
      VK_FROM_HANDLE(hk_device_memory, mem, pBindInfos[i].memory);
      VK_FROM_HANDLE(hk_image, image, pBindInfos[i].image);

      /* Ignore this struct on Android, we cannot access swapchain structures
       * there. */
#ifdef HK_USE_WSI_PLATFORM
      if (!mem) {
         const VkBindImageMemorySwapchainInfoKHR *swapchain_info =
            vk_find_struct_const(pBindInfos[i].pNext,
                                 BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR);
         assert(swapchain_info && swapchain_info->swapchain != VK_NULL_HANDLE);
         mem = hk_device_memory_from_handle(wsi_common_get_memory(
            swapchain_info->swapchain, swapchain_info->imageIndex));
      }
#endif

      assert(mem);
      const uint64_t offset_B = pBindInfos[i].memoryOffset;
      if (image->disjoint) {
         const VkBindImagePlaneMemoryInfo *plane_info = vk_find_struct_const(
            pBindInfos[i].pNext, BIND_IMAGE_PLANE_MEMORY_INFO);
         uint8_t plane =
            hk_image_aspects_to_plane(image, plane_info->planeAspect);
         if (!hk_image_plane_bind(dev, &image->planes[plane], mem, offset_B))
            return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                             "image plane %u exceeds bound memory", plane);
      } else {
         for (unsigned plane = 0; plane < image->plane_count; plane++) {
            if (!hk_image_plane_bind(dev, &image->planes[plane], mem, offset_B))
               return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                                "image plane %u exceeds bound memory", plane);
         }
      }

      const VkBindMemoryStatusKHR *status =
         vk_find_struct_const(pBindInfos[i].pNext, BIND_MEMORY_STATUS_KHR);
      if (status != NULL && status->pResult != NULL)
         *status->pResult = VK_SUCCESS;
   }

   return VK_SUCCESS;
}

static uint32_t
hk_plane_index(VkFormat format, VkImageAspectFlags aspect_mask)
{
   switch (aspect_mask) {
   default:
      assert(aspect_mask != VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT);
      return 0;
   case VK_IMAGE_ASPECT_PLANE_1_BIT:
   case VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT:
      return 1;
   case VK_IMAGE_ASPECT_PLANE_2_BIT:
   case VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT:
      return 2;
   case VK_IMAGE_ASPECT_STENCIL_BIT:
      return hk_split_depth_stencil(format);
   }
}

static void
hk_copy_memory_to_image(struct hk_device *device, struct hk_image *dst_image,
                        const VkMemoryToImageCopyEXT *info, bool copy_memcpy)
{
   unsigned plane =
      hk_plane_index(dst_image->vk.format, info->imageSubresource.aspectMask);
   const struct ail_layout *layout = &dst_image->planes[plane].layout;

   VkOffset3D offset = info->imageOffset;
   VkExtent3D extent = info->imageExtent;
   uint32_t src_width = info->memoryRowLength ?: extent.width;
   uint32_t src_height = info->memoryImageHeight ?: extent.height;

   uint32_t blocksize_B = util_format_get_blocksize(layout->format);

   /* Align width and height to block */
   src_width =
      DIV_ROUND_UP(src_width, util_format_get_blockwidth(layout->format));
   src_height =
      DIV_ROUND_UP(src_height, util_format_get_blockheight(layout->format));

   uint32_t src_pitch = src_width * blocksize_B;

   unsigned start_layer = (dst_image->vk.image_type == VK_IMAGE_TYPE_3D)
                             ? offset.z
                             : info->imageSubresource.baseArrayLayer;
   uint32_t layers =
      MAX2(extent.depth, vk_image_subresource_layer_count(
                            &dst_image->vk, &info->imageSubresource));

   unsigned level = info->imageSubresource.mipLevel;
   uint32_t image_offset = ail_get_layer_level_B(layout, start_layer, level);
   uint32_t dst_layer_stride = layout->layer_stride_B;
   uint32_t src_layer_stride = copy_memcpy
                                  ? ail_get_level_size_B(layout, level)
                                  : (src_width * src_height * blocksize_B);
   bool tiled = ail_is_level_twiddled_uncompressed(
      layout, info->imageSubresource.mipLevel);

   const char *src = (const char *)info->pHostPointer;
   char *dst = (char *)dst_image->planes[plane].map + image_offset;
   for (unsigned layer = 0; layer < layers;
        layer++, src += src_layer_stride, dst += dst_layer_stride) {
      if (copy_memcpy) {
         memcpy(dst, src, ail_get_level_size_B(layout, level));
      } else if (!tiled) {
         uint32_t dst_pitch = ail_get_linear_stride_B(layout, level);
         /*TODO:comp*/
         for (unsigned y = 0; y < extent.height; y++) {
            memcpy(dst + dst_pitch * (y + offset.y) + offset.x * blocksize_B,
                   src + src_pitch * y, extent.width * blocksize_B);
         }
      } else {
         ail_tile(dst, (void *)src, layout, level, src_pitch, offset.x,
                  offset.y, extent.width, extent.height);
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyMemoryToImageEXT(VkDevice _device,
                        const VkCopyMemoryToImageInfoEXT *info)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   VK_FROM_HANDLE(hk_image, dst_image, info->dstImage);

   for (unsigned i = 0; i < info->regionCount; i++) {
      hk_copy_memory_to_image(device, dst_image, &info->pRegions[i],
                              info->flags & VK_HOST_IMAGE_COPY_MEMCPY_EXT);
   }

   return VK_SUCCESS;
}

static void
hk_copy_image_to_memory(struct hk_device *device, struct hk_image *src_image,
                        const VkImageToMemoryCopyEXT *info, bool copy_memcpy)
{
   unsigned plane =
      hk_plane_index(src_image->vk.format, info->imageSubresource.aspectMask);
   const struct ail_layout *layout = &src_image->planes[plane].layout;

   VkOffset3D offset = info->imageOffset;
   VkExtent3D extent = info->imageExtent;
   uint32_t dst_width = info->memoryRowLength ?: extent.width;
   uint32_t dst_height = info->memoryImageHeight ?: extent.height;

#if 0
   copy_compressed(src_image->vk.format, &offset, &extent, &dst_width,
                   &dst_height);
#endif

   uint32_t blocksize_B = util_format_get_blocksize(layout->format);

   /* Align width and height to block */
   dst_width =
      DIV_ROUND_UP(dst_width, util_format_get_blockwidth(layout->format));
   dst_height =
      DIV_ROUND_UP(dst_height, util_format_get_blockheight(layout->format));

   uint32_t dst_pitch = dst_width * blocksize_B;

   unsigned start_layer = (src_image->vk.image_type == VK_IMAGE_TYPE_3D)
                             ? offset.z
                             : info->imageSubresource.baseArrayLayer;
   uint32_t layers =
      MAX2(extent.depth, vk_image_subresource_layer_count(
                            &src_image->vk, &info->imageSubresource));
   unsigned level = info->imageSubresource.mipLevel;

   uint32_t image_offset = ail_get_layer_level_B(layout, start_layer, level);
   uint32_t src_layer_stride = layout->layer_stride_B;
   uint32_t dst_layer_stride = copy_memcpy
                                  ? ail_get_level_size_B(layout, level)
                                  : (dst_width * dst_height * blocksize_B);

   bool tiled = ail_is_level_twiddled_uncompressed(
      layout, info->imageSubresource.mipLevel);

   const char *src = (const char *)src_image->planes[plane].map + image_offset;
   char *dst = (char *)info->pHostPointer;
   for (unsigned layer = 0; layer < layers;
        layer++, src += src_layer_stride, dst += dst_layer_stride) {

      if (copy_memcpy) {
         memcpy(dst, src, dst_layer_stride);
      } else if (!tiled) {
         /* TODO: comp */
         uint32_t src_pitch = ail_get_linear_stride_B(layout, level);
         for (unsigned y = 0; y < extent.height; y++) {
            memcpy(dst + dst_pitch * y,
                   src + src_pitch * (y + offset.y) + offset.x * blocksize_B,
                   extent.width * blocksize_B);
         }
      } else {
         ail_detile((void *)src, dst, layout, info->imageSubresource.mipLevel,
                    dst_pitch, offset.x, offset.y, extent.width, extent.height);
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyImageToMemoryEXT(VkDevice _device,
                        const VkCopyImageToMemoryInfoEXT *info)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   VK_FROM_HANDLE(hk_image, image, info->srcImage);

   for (unsigned i = 0; i < info->regionCount; i++) {
      hk_copy_image_to_memory(device, image, &info->pRegions[i],
                              info->flags & VK_HOST_IMAGE_COPY_MEMCPY_EXT);
   }

   return VK_SUCCESS;
}

static void
hk_copy_image_to_image_cpu(struct hk_device *device, struct hk_image *src_image,
                           struct hk_image *dst_image, const VkImageCopy2 *info,
                           bool copy_memcpy)
{
   unsigned src_plane =
      hk_plane_index(src_image->vk.format, info->srcSubresource.aspectMask);
   unsigned dst_plane =
      hk_plane_index(dst_image->vk.format, info->dstSubresource.aspectMask);

   const struct ail_layout *src_layout = &src_image->planes[src_plane].layout;
   const struct ail_layout *dst_layout = &dst_image->planes[dst_plane].layout;

   VkOffset3D src_offset = info->srcOffset;
   VkOffset3D dst_offset = info->dstOffset;
   VkExtent3D extent = info->extent;
   uint32_t layers_to_copy = MAX2(
      info->extent.depth,
      vk_image_subresource_layer_count(&src_image->vk, &info->srcSubresource));

   /* See comment above. */
#if 0
   copy_compressed(src_image->vk.format, &src_offset, &extent, NULL, NULL);
   copy_compressed(dst_image->vk.format, &dst_offset, NULL, NULL, NULL);
#endif

   unsigned src_start_layer = (src_image->vk.image_type == VK_IMAGE_TYPE_3D)
                                 ? src_offset.z
                                 : info->srcSubresource.baseArrayLayer;
   unsigned dst_start_layer = (dst_image->vk.image_type == VK_IMAGE_TYPE_3D)
                                 ? dst_offset.z
                                 : info->dstSubresource.baseArrayLayer;

   uint32_t src_layer_stride = src_layout->layer_stride_B;
   uint32_t dst_layer_stride = dst_layout->layer_stride_B;

   uint32_t dst_block_B = util_format_get_blocksize(dst_layout->format);
   uint32_t src_block_B = util_format_get_blocksize(src_layout->format);

   uint32_t src_image_offset = ail_get_layer_level_B(
      src_layout, src_start_layer, info->srcSubresource.mipLevel);
   uint32_t dst_image_offset = ail_get_layer_level_B(
      dst_layout, dst_start_layer, info->dstSubresource.mipLevel);

   bool src_tiled = ail_is_level_twiddled_uncompressed(
      src_layout, info->srcSubresource.mipLevel);
   bool dst_tiled = ail_is_level_twiddled_uncompressed(
      dst_layout, info->dstSubresource.mipLevel);

   const char *src =
      (const char *)src_image->planes[src_plane].map + src_image_offset;
   char *dst = (char *)dst_image->planes[dst_plane].map + dst_image_offset;
   for (unsigned layer = 0; layer < layers_to_copy;
        layer++, src += src_layer_stride, dst += dst_layer_stride) {

      if (copy_memcpy) {
         uint32_t src_size =
            ail_get_level_size_B(src_layout, info->srcSubresource.mipLevel);
         uint32_t dst_size =
            ail_get_level_size_B(dst_layout, info->dstSubresource.mipLevel);

         assert(src_size == dst_size);
         memcpy(dst, src, src_size);
      } else if (!src_tiled && !dst_tiled) {
         /* TODO comp */
         uint32_t src_pitch =
            ail_get_linear_stride_B(src_layout, info->srcSubresource.mipLevel);

         uint32_t dst_pitch =
            ail_get_linear_stride_B(dst_layout, info->dstSubresource.mipLevel);

         for (unsigned y = 0; y < extent.height; y++) {
            memcpy(dst + dst_pitch * (y + dst_offset.y) +
                      dst_offset.x * dst_block_B,
                   src + src_pitch * (y + src_offset.y) +
                      src_offset.x * src_block_B,
                   extent.width * src_block_B);
         }
      } else if (!src_tiled) {
         UNREACHABLE("todo");
#if 0
         fdl6_memcpy_linear_to_tiled(
            dst_offset.x, dst_offset.y, extent.width, extent.height, dst,
            src + src_pitch * src_offset.y + src_offset.x * src_layout->cpp,
            dst_layout, info->dstSubresource.mipLevel, src_pitch,
            &device->physical_device->ubwc_config);
#endif
      } else if (!dst_tiled) {
         UNREACHABLE("todo");
#if 0
         fdl6_memcpy_tiled_to_linear(
            src_offset.x, src_offset.y, extent.width, extent.height,
            dst + dst_pitch * dst_offset.y + dst_offset.x * dst_layout->cpp,
            src, src_layout, info->dstSubresource.mipLevel, dst_pitch,
            &device->physical_device->ubwc_config);
#endif
      } else {
         unsigned src_level = info->srcSubresource.mipLevel;
         unsigned dst_level = info->dstSubresource.mipLevel;
         uint32_t block_width = src_layout->tilesize_el[src_level].width_el;
         uint32_t block_height = src_layout->tilesize_el[src_level].height_el;

         /* Twiddled images have a single "tile" sized to the entire image, so
          * break it up so we'll fit.
          */
         if (src_layout->tiling == AIL_TILING_TWIDDLED) {
            block_width = block_height = MIN2(block_width, 32);
         }

         uint32_t temp_pitch = block_width * src_block_B;
         size_t temp_tile_size = temp_pitch * (src_offset.y + extent.height);

         /* Work tile-by-tile, holding the unswizzled tile in a temporary
          * buffer.
          */
         char *temp_tile = malloc(temp_tile_size);

         for (unsigned by = src_offset.y / block_height;
              by * block_height < src_offset.y + extent.height; by++) {
            uint32_t src_y_start = MAX2(src_offset.y, by * block_height);
            uint32_t dst_y_start = src_y_start - src_offset.y + dst_offset.y;
            uint32_t height =
               MIN2((by + 1) * block_height, src_offset.y + extent.height) -
               src_y_start;
            for (unsigned bx = src_offset.x / block_width;
                 bx * block_width < src_offset.x + extent.width; bx++) {
               uint32_t src_x_start = MAX2(src_offset.x, bx * block_width);
               uint32_t dst_x_start = src_x_start - src_offset.x + dst_offset.x;
               uint32_t width =
                  MIN2((bx + 1) * block_width, src_offset.x + extent.width) -
                  src_x_start;

               ail_detile((void *)src, temp_tile, src_layout, src_level,
                          temp_pitch, src_x_start, src_y_start, width, height);
               ail_tile(dst, temp_tile, dst_layout, dst_level, temp_pitch,
                        dst_x_start, dst_y_start, width, height);
            }
         }

         free(temp_tile);
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyImageToImageEXT(VkDevice _device,
                       const VkCopyImageToImageInfoEXT *pCopyImageToImageInfo)
{
   VK_FROM_HANDLE(hk_device, device, _device);
   VK_FROM_HANDLE(hk_image, src_image, pCopyImageToImageInfo->srcImage);
   VK_FROM_HANDLE(hk_image, dst_image, pCopyImageToImageInfo->dstImage);
   bool copy_memcpy =
      pCopyImageToImageInfo->flags & VK_HOST_IMAGE_COPY_MEMCPY_EXT;

   for (uint32_t i = 0; i < pCopyImageToImageInfo->regionCount; ++i) {
      if (hk_split_depth_stencil(src_image->vk.format)) {
         VkImageCopy2 info = pCopyImageToImageInfo->pRegions[i];
         u_foreach_bit(b, info.dstSubresource.aspectMask) {
            info.srcSubresource.aspectMask = BITFIELD_BIT(b);
            info.dstSubresource.aspectMask = BITFIELD_BIT(b);
            hk_copy_image_to_image_cpu(device, src_image, dst_image, &info,
                                       copy_memcpy);
         }
         continue;
      }

      hk_copy_image_to_image_cpu(device, src_image, dst_image,
                                 pCopyImageToImageInfo->pRegions + i,
                                 copy_memcpy);
   }

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_TransitionImageLayoutEXT(
   VkDevice device, uint32_t transitionCount,
   const VkHostImageLayoutTransitionInfoEXT *transitions)
{
   /* We don't do anything with layouts so this should be a no-op */
   return VK_SUCCESS;
}

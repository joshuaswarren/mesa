/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_native_descriptor_pack.h"

#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_pbe_agx3_core.h"
#include "util/macros.h"
#include <assert.h>
#include "util/u_atomic.h"

#include <string.h>

bool
hk_translate_native_texture_sample_count(unsigned generation, unsigned samples,
                                         enum agx_sample_count *out)
{
   if (!out || samples == 1 ||
       !hk_native_texture_sample_count_representable(generation, samples))
      return false;

   switch (samples) {
   case 2:
      *out = AGX_SAMPLE_COUNT_2;
      return true;
   case 4:
      *out = AGX_SAMPLE_COUNT_4;
      return true;
   case 8:
      *out = AGX_SAMPLE_COUNT_8;
      return true;
   default:
      return false;
   }
}

bool
hk_pack_native_texture(unsigned generation,
                       struct hk_native_texture_descriptor *out,
                       const struct AGX_TEXTURE *legacy)
{
   if (!out || !legacy || !hk_texture_descriptor_size(generation))
      return false;

   memset(out, 0, sizeof(*out));

   if (hk_descriptor_generation_is_agx2(generation)) {
      AGX_TEXTURE_pack(out->opaque, legacy);
      return true;
   }

   const bool multisampled =
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;
   if (!hk_native_texture_supported(generation, multisampled,
                                    legacy->buffer_offset_sw))
      return false;

   /* The Apple Gen9/10 encoders accept the three recovered two-bit selector
    * values only.  Reject malformed semantic input instead of truncating it.
    */
   if ((multisampled &&
        (unsigned)legacy->samples > (unsigned)AGX_SAMPLE_COUNT_8) ||
       (!multisampled && legacy->samples != AGX_SAMPLE_COUNT_2))
      return false;

   const bool has_levels =
      legacy->first_level || legacy->last_level || legacy->unk_mipmapped;
   const bool extended = legacy->extended || has_levels;
   if ((legacy->address & 0xf) || legacy->address >> 48 ||
       (legacy->compressed &&
        ((legacy->acceleration_buffer & 0xf) ||
         legacy->acceleration_buffer >> 48)))
      return false;

   if (generation == 9) {
      if (legacy->compressed &&
          (!extended || !legacy->acceleration_buffer ||
           (multisampled && legacy->samples == AGX_SAMPLE_COUNT_8)))
         return false;

      if (!legacy->width || legacy->width > 16384 || !legacy->height ||
          legacy->height > 16384)
         return false;

      /* Stride and Depth share word1[61:46], sixteen bits, and the genxml
       * modifiers put stride >> 4 and depth - 1 there.  util_bitpack_uint()
       * is `v << start`, so a value that does not fit runs past the field and
       * into "Page-aligned layers" (126) and "Extended" (127) -- with only a
       * debug-build assert to catch it.  Refuse instead.  Nothing hk builds
       * reaches either bound today (the widest stride is a buffer texture's
       * AGX_TEXTURE_BUFFER_WIDTH * 16 - 16, and maxImageArrayLayers is far
       * below 65536), so this is a guard on a latent overrun rather than a
       * behaviour change. */
      /* Interchange takes the row-stride field like Linear (HAL200
       * encodeImageReadState handles layouts 0 and 3 on one path). */
      const bool strided = legacy->layout == AGX_LAYOUT_LINEAR ||
                           legacy->layout == AGX_LAYOUT_INTERCHANGE;
      if (strided ? ((legacy->stride >> 4) > 0xffffu)
                  : (legacy->depth > 0x10000u))
         return false;
      /* Interchange images are single-layer, single-level 2D; the layered
       * background descriptor every colour view also packs is a one-layer
       * 2D array. */
      if (legacy->layout == AGX_LAYOUT_INTERCHANGE &&
          ((legacy->dimension != AGX_TEXTURE_DIMENSION_2D &&
            legacy->dimension != AGX_TEXTURE_DIMENSION_2D_ARRAY) ||
           legacy->depth != 1 || has_levels))
         return false;

      /* HAL200 encodeImageReadState clears word1[62] for ordinary non-array
       * 2D and 2DMS textures. Its conditional layer-mode writer is not the
       * inherited AIL page-aligned-allocation flag. Keep this correction
       * bounded to fully represented, single-layer, one-level images;
       * array/compression policies need their own evidence.
       */
      const bool plain_2d =
         (legacy->dimension == AGX_TEXTURE_DIMENSION_2D ||
          legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED) &&
         legacy->layout == AGX_LAYOUT_GPU && legacy->depth == 1 &&
         !legacy->compressed && !extended &&
         legacy->mode == AGX_IMAGE_MODE_NORMAL;

      struct AGX_TEXTURE_GEN9 cfg = {
         .dimension = legacy->dimension,
         .layout = legacy->layout,
         .channels = legacy->channels,
         .type = legacy->type,
         .swizzle_r = legacy->swizzle_r,
         .swizzle_g = legacy->swizzle_g,
         .swizzle_b = legacy->swizzle_b,
         .swizzle_a = legacy->swizzle_a,
         .width = legacy->width,
         .height = legacy->height,
         .samples = legacy->samples,
         .has_levels = has_levels,
         .compressed = legacy->compressed,
         .state_control = legacy->mode | (legacy->compression << 2),
         .address = legacy->address,
         .srgb = legacy->srgb,
         .srgb_2_channel = legacy->srgb_2_channel,
         .stride = strided ? legacy->stride : 0,
         .depth = strided ? 1 : legacy->depth,
         .page_aligned_layers = plain_2d ? false : legacy->page_aligned_layers,
         .extended = extended,
         .acceleration_buffer =
            legacy->compressed && legacy->extended
               ? legacy->acceleration_buffer
               : 0,
         .depth_linear = legacy->layout == AGX_LAYOUT_LINEAR &&
                               legacy->extended
                            ? legacy->depth_linear
                            : 1,
         .layer_stride_linear = legacy->layout == AGX_LAYOUT_LINEAR &&
                                       legacy->extended
                                    ? legacy->layer_stride_linear
                                    : 0,
         .first_level = legacy->first_level,
         .last_level = legacy->last_level,
      };

      AGX_TEXTURE_GEN9_pack(out->opaque, &cfg);
      return true;
   }

   if (generation == 10) {
      if (!legacy->width || legacy->width > 32768 || !legacy->height ||
          legacy->height > 32768)
         return false;

      struct AGX_TEXTURE_GEN10 cfg = {
         .dimension = legacy->dimension,
         .layout = legacy->layout,
         .channels = legacy->channels,
         .type = legacy->type,
         .swizzle_r = legacy->swizzle_r,
         .swizzle_g = legacy->swizzle_g,
         .swizzle_b = legacy->swizzle_b,
         .swizzle_a = legacy->swizzle_a,
         .width = legacy->width,
         .height = legacy->height,
         .samples = legacy->samples,
         .unk_mipmapped = legacy->unk_mipmapped,
         .compressed = legacy->compressed,
         .image_mode_enable = legacy->mode != AGX_IMAGE_MODE_NORMAL,
         /* The HAL300 texture encoder writes 1 for 16 KiB, 2 for 64 KiB and 3
          * for 256 KiB sparse pages. AIL uses 16 KiB pages. */
         .image_mode = legacy->mode == AGX_IMAGE_MODE_SPARSE ? 1 : legacy->mode,
         .address = legacy->address,
         .srgb = legacy->srgb,
         .srgb_2_channel = legacy->srgb_2_channel,
         .stride = legacy->layout == AGX_LAYOUT_LINEAR ? legacy->stride : 0,
         .depth = legacy->layout == AGX_LAYOUT_LINEAR ? 1 : legacy->depth,
         .page_aligned_layers = legacy->page_aligned_layers,
         .extended = extended,
         .acceleration_buffer =
            legacy->compressed && legacy->extended
               ? legacy->acceleration_buffer
               : 0,
         .depth_linear = legacy->layout == AGX_LAYOUT_LINEAR &&
                               legacy->extended
                            ? legacy->depth_linear
                            : 1,
         .layer_stride_linear = legacy->layout == AGX_LAYOUT_LINEAR &&
                                       legacy->extended
                                    ? legacy->layer_stride_linear
                                    : 0,
         .first_level = legacy->first_level,
         .last_level = legacy->last_level,
      };

      AGX_TEXTURE_GEN10_pack(out->opaque, &cfg);
      return true;
   }

   return false;
}

void
hk_native_texture_set_level_zero_extent(
   struct hk_native_texture_descriptor *out, uint32_t width, uint32_t height)
{
   /* Width and height are each at most 32768 (hk_pack_native_texture). */
   assert(width <= 0xffff && height <= 0xffff);
   out->opaque[AGX_GEN10_IMAGE_EXTENT_OFFSET / 4] = width | (height << 16);
}

bool
hk_pack_native_pbe(unsigned generation,
                   struct hk_native_pbe_descriptor *out,
                   const struct AGX_PBE *legacy)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));
   if (!legacy || !hk_pbe_descriptor_size(generation))
      return false;

   if (hk_descriptor_generation_is_agx2(generation)) {
      AGX_PBE_pack(out->opaque, legacy);
      return true;
   }

   if (!hk_descriptor_generation_is_agx3(generation))
      return false;

   /* The recovered subset excludes fields whose Gen9/10 semantic placement
    * remains incomplete: sparse mode and the software-addressing sideband. The host-only HAL200 PBE encoder oracle matches the complete
    * GPU-tiled 2D compressed packet, including the metadata GPU address and
    * mip extension. Other compressed shapes remain closed. */
   const bool array = (generation == 9 || generation == 10) &&
                      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY;
   const bool array_multisample =
      (generation == 9 || generation == 10) &&
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;
   const bool three_d = (generation == 9 || generation == 10) &&
                        legacy->dimension == AGX_TEXTURE_DIMENSION_3D;
   const bool layered = array || array_multisample || three_d;
   /* Both generations have native dimension-4 single-layer MS stores.
    * Gen9 additionally has the qualified dimension-8 array packet. */
   const bool multisample =
      (generation == 9 || generation == 10) &&
      (legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
       array_multisample);
   const bool interchange = legacy->layout == AGX_LAYOUT_INTERCHANGE;
   const bool compressed_gen9 = legacy->compressed && generation == 9 &&
      (legacy->layout == AGX_LAYOUT_GPU ||
       (interchange && !layered && !multisample));
   const bool compressed_2d = legacy->compressed &&
      (generation == 9 || generation == 10) &&
      (legacy->dimension == AGX_TEXTURE_DIMENSION_2D ||
       (generation == 10 && array && legacy->layers == 1 &&
        legacy->level == 0 && legacy->levels == 1)) &&
      (legacy->layout == AGX_LAYOUT_GPU || (generation == 9 && interchange)) &&
      !multisample;
   /* Private destination capability for raw D32 copy views. The independently
    * witnessed packet combines only Gen10 R32_UINT with compressed 2x/4x.
    * Image allocation policy remains separate and unchanged. */
   const bool compressed_msaa_r32_uint =
      generation == 10 &&
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED &&
      legacy->layout == AGX_LAYOUT_GPU &&
      legacy->channels == AGX_CHANNELS_R32 &&
      legacy->type == AGX_TEXTURE_TYPE_UINT && !legacy->srgb &&
      legacy->layers == 1 && legacy->level == 0 && legacy->levels == 1 &&
      !legacy->unk_mipmapped &&
      (legacy->samples == AGX_SAMPLE_COUNT_2 ||
       legacy->samples == AGX_SAMPLE_COUNT_4);
   const bool compressed_msaa_rgba16 =
      generation == 10 &&
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED &&
      legacy->layout == AGX_LAYOUT_GPU &&
      legacy->channels == AGX_CHANNELS_R16G16B16A16 &&
      (legacy->type == AGX_TEXTURE_TYPE_FLOAT ||
       legacy->type == AGX_TEXTURE_TYPE_UINT) && !legacy->srgb &&
      legacy->layers == 1 && legacy->level == 0 && legacy->levels == 1 &&
      !legacy->unk_mipmapped &&
      (legacy->samples == AGX_SAMPLE_COUNT_2 ||
       legacy->samples == AGX_SAMPLE_COUNT_4);
   /* RGBA8/BGRA8 (UNORM and sRGB) share the R32_UINT texel size and so its
    * compressed multisample metadata layout; see
    * agx_pbe_agx3_2d_multisample_pack(). */
   const bool compressed_msaa_rgba8 =
      generation == 10 &&
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED &&
      legacy->layout == AGX_LAYOUT_GPU &&
      legacy->channels == AGX_CHANNELS_R8G8B8A8 &&
      legacy->type == AGX_TEXTURE_TYPE_UNORM &&
      legacy->layers == 1 && legacy->level == 0 && legacy->levels == 1 &&
      !legacy->unk_mipmapped &&
      (legacy->samples == AGX_SAMPLE_COUNT_2 ||
       legacy->samples == AGX_SAMPLE_COUNT_4);
   if ((!layered && !multisample &&
        legacy->dimension != AGX_TEXTURE_DIMENSION_2D) ||
       (legacy->layout == AGX_LAYOUT_LINEAR &&
        (legacy->level != 0 || legacy->levels != 1 ||
         legacy->unk_mipmapped)) ||
       (legacy->layout != AGX_LAYOUT_LINEAR &&
        legacy->layout != AGX_LAYOUT_TWIDDLED &&
        legacy->layout != AGX_LAYOUT_GPU &&
        !(interchange && generation == 9 && !layered && !multisample &&
          legacy->level == 0 && legacy->levels == 1)) ||
       legacy->unk_52 || legacy->rotate_90 || legacy->flip_vertical ||
       (multisample ? (legacy->samples != AGX_SAMPLE_COUNT_2 &&
                       legacy->samples != AGX_SAMPLE_COUNT_4 &&
                       !(generation == 10 && legacy->samples == AGX_SAMPLE_COUNT_8))
                    : legacy->samples != AGX_SAMPLE_COUNT_2) ||
       (legacy->compressed && !compressed_gen9 && !compressed_2d &&
                          !compressed_msaa_r32_uint &&
                          !compressed_msaa_rgba16 && !compressed_msaa_rgba8) ||
       legacy->mode != AGX_IMAGE_MODE_NORMAL ||
       (!layered && legacy->layers != 1) ||
       (multisample && (legacy->layout == AGX_LAYOUT_LINEAR ||
                        legacy->level != 0 || legacy->levels != 1 ||
                        legacy->unk_mipmapped ||
                        (legacy->compressed && !compressed_gen9 &&
                          !compressed_msaa_r32_uint &&
                          !compressed_msaa_rgba16 &&
                          !compressed_msaa_rgba8))) ||
       (legacy->extended != legacy->compressed) ||
       (legacy->layout == AGX_LAYOUT_LINEAR || interchange
           ? legacy->stride < 12 || legacy->stride > UINT32_MAX - 4
                                             : legacy->stride != 0) ||
       (!legacy->compressed && legacy->acceleration_buffer) ||
       legacy->depth_linear != 1 || legacy->layer_stride_linear ||
       legacy->level_offset_sw || legacy->aligned_width_msaa_sw ||
       legacy->sample_count_log2_sw || legacy->tile_width_sw != 1 ||
       legacy->tile_height_sw != 1 || legacy->layer_stride_sw ||
       legacy->buffer_offset_sw)
      return false;

   const struct agx_pbe_agx3_2d_fields fields = {
      .layout = legacy->layout,
      .channels = legacy->channels,
      .type = legacy->type,
      .swizzle_r = legacy->swizzle_r,
      .swizzle_g = legacy->swizzle_g,
      .swizzle_b = legacy->swizzle_b,
      .swizzle_a = legacy->swizzle_a,
      .width = legacy->width,
      .height = legacy->height,
      .buffer = legacy->buffer,
      .compressed = legacy->compressed,
      .acceleration_buffer = legacy->acceleration_buffer,
      .stride_B = legacy->layout == AGX_LAYOUT_LINEAR || interchange
                     ? legacy->stride + 4
                     : 0,
      .page_aligned_layers = legacy->page_aligned_layers,
      .srgb = legacy->srgb,
   };
   struct agx_pbe_agx3_descriptor packed;
   const bool valid = multisample
      ? (array_multisample
            ? agx_pbe_agx3_2d_array_multisample_pack(
                 generation, &fields, legacy->layers,
                 legacy->samples == AGX_SAMPLE_COUNT_8 ? 8u :
                 legacy->samples == AGX_SAMPLE_COUNT_4 ? 4u : 2u, &packed)
            : agx_pbe_agx3_2d_multisample_pack(
                 generation, &fields,
                 legacy->samples == AGX_SAMPLE_COUNT_8 ? 8u :
                 legacy->samples == AGX_SAMPLE_COUNT_4 ? 4u : 2u, &packed))
      : layered
      ? (array ? agx_pbe_agx3_2d_array_mip_pack(
                    generation, &fields, legacy->layers, legacy->level,
                    legacy->levels, &packed)
               : agx_pbe_agx3_3d_mip_pack(generation, &fields, legacy->layers,
                                          legacy->level, legacy->levels,
                                          &packed))
      : agx_pbe_agx3_2d_mip_pack(generation, &fields, legacy->level,
                                 legacy->levels, &packed);
   if (!valid)
      return false;

   static_assert(sizeof(out->opaque) == sizeof(packed.bytes),
                 "opaque descriptor matches the packed size");
   memcpy(out->opaque, packed.bytes, sizeof(packed.bytes));
   return true;
}

static bool
hk_pack_null_native_texture(unsigned generation,
                            struct hk_native_texture_descriptor *out)
{
   const struct AGX_TEXTURE semantic = {
      AGX_TEXTURE_header,
      .layout = AGX_LAYOUT_TWIDDLED,
      .channels = AGX_CHANNELS_R8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_0,
      .swizzle_g = AGX_CHANNEL_0,
      .swizzle_b = AGX_CHANNEL_0,
      .swizzle_a = AGX_CHANNEL_0,
      .address = AGX_ZERO_PAGE_ADDRESS,
   };

   return hk_pack_native_texture(generation, out, &semantic);
}

/* The null descriptor is a pure function of the generation, and every
 * sampled-image descriptor write starts from it (three planes each), so pack
 * it once per generation.  A refused generation is not cached: it takes the
 * packer, and its exact output, every time. */
struct hk_native_texture_descriptor
   hk_null_native_texture_packed[HK_NULL_NATIVE_TEXTURE_GENERATIONS];
uint32_t hk_null_native_texture_state[HK_NULL_NATIVE_TEXTURE_GENERATIONS];

bool
hk_set_null_native_texture_slow(unsigned generation,
                                struct hk_native_texture_descriptor *out)
{
   struct hk_native_texture_descriptor *packed = hk_null_native_texture_packed;
   uint32_t *state = hk_null_native_texture_state;

   if (!out)
      return false;

   if (generation < HK_NULL_NATIVE_TEXTURE_GENERATIONS &&
       p_atomic_read(&state[generation]) == 2) {
      *out = packed[generation];
      return true;
   }

   if (!hk_pack_null_native_texture(generation, out))
      return false;

   if (generation < HK_NULL_NATIVE_TEXTURE_GENERATIONS &&
       p_atomic_cmpxchg(&state[generation], 0, 1) == 0) {
      packed[generation] = *out;
      p_atomic_set(&state[generation], 2);
   }
   return true;
}

bool
hk_set_null_native_pbe(unsigned generation,
                       struct hk_native_pbe_descriptor *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));
   if (!hk_native_pbe_hardware_supported(generation))
      return false;

   const struct AGX_PBE semantic = {
      AGX_PBE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_TWIDDLED,
      .channels = AGX_CHANNELS_R8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_R,
      .swizzle_b = AGX_CHANNEL_R,
      .swizzle_a = AGX_CHANNEL_R,
      .buffer = AGX_SCRATCH_PAGE_ADDRESS,
   };
   return hk_pack_native_pbe(generation, out, &semantic);
}

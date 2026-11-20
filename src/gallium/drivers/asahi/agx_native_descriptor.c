/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_native_descriptor.h"

#include "asahi/genxml/agx_pack.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_pbe_agx3_core.h"

#include <string.h>

bool
agx_gallium_classify_standard_border(const uint32_t color[4], bool integer,
                                     enum agx_border_colour *out)
{
   if (!color || !out)
      return false;

   bool rgb_zero = true, rgba_one = true;
   for (unsigned i = 0; i < 4; ++i) {
      union {
         uint32_t u;
         float f;
      } value = {.u = color[i]};
      const bool zero = integer ? color[i] == 0 : value.f == 0.0f;
      const bool one = integer ? color[i] == 1 : value.f == 1.0f;
      rgb_zero &= i == 3 || zero;
      rgba_one &= one;
   }

   union {
      uint32_t u;
      float f;
   } alpha = {.u = color[3]};
   const bool alpha_zero = integer ? color[3] == 0 : alpha.f == 0.0f;
   const bool alpha_one = integer ? color[3] == 1 : alpha.f == 1.0f;

   if (rgb_zero && alpha_zero) {
      *out = AGX_BORDER_COLOUR_TRANSPARENT_BLACK;
   } else if (rgb_zero && alpha_one) {
      *out = AGX_BORDER_COLOUR_OPAQUE_BLACK;
   } else if (rgba_one) {
      *out = AGX_BORDER_COLOUR_OPAQUE_WHITE;
   } else {
      return false;
   }

   return true;
}

bool
agx_gallium_pack_native_sampler(
   enum agx_chip chip, struct agx_gallium_sampler_descriptor *out,
   const struct agx_gallium_native_sampler *state)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   if (!state || !agx_chip_native_agx3(chip) ||
       agx_chip_sampler_heap_element_size(chip) != sizeof(*out))
      return false;

   /* The packed word is eight bytes at the start of the 32-byte heap slot;
    * the rest stays zero. */
   switch (agx_chip_sampler_generation(chip)) {
   case AGX_SAMPLER_GENERATION_GEN4_HAL200:
      agx_pack(out->opaque, SAMPLER_HAL200, cfg) {
         cfg.minimum_lod = state->minimum_lod;
         cfg.maximum_lod = state->maximum_lod;
         cfg.maximum_anisotropy = state->maximum_anisotropy;
         cfg.magnify = state->magnify;
         cfg.minify = state->minify;
         cfg.mip_filter = state->mip_filter;
         cfg.wrap_s = state->wrap_s;
         cfg.wrap_t = state->wrap_t;
         cfg.wrap_r = state->wrap_r;
         cfg.pixel_coordinates = state->pixel_coordinates;
         cfg.compare_func = state->compare_func;
         cfg.compare_enable = state->compare_enable;
         cfg.lod_bias = state->lod_bias;
         cfg.seamful_cube_maps = state->seamful_cube_maps;
         cfg.border_colour = state->border_colour;
      }
      return true;

   case AGX_SAMPLER_GENERATION_GEN4:
      /* No bias field: the shader applies every sampler bias. */
      agx_pack(out->opaque, SAMPLER, cfg) {
         cfg.minimum_lod = state->minimum_lod;
         cfg.maximum_lod = state->maximum_lod;
         cfg.maximum_anisotropy = state->maximum_anisotropy;
         cfg.magnify = state->magnify;
         cfg.minify = state->minify;
         cfg.mip_filter = state->mip_filter;
         cfg.wrap_s = state->wrap_s;
         cfg.wrap_t = state->wrap_t;
         cfg.wrap_r = state->wrap_r;
         cfg.pixel_coordinates = state->pixel_coordinates;
         cfg.compare_func = state->compare_func;
         cfg.compare_enable = state->compare_enable;
         cfg.seamful_cube_maps = state->seamful_cube_maps;
         cfg.border_colour = state->border_colour;
      }
      return true;

   default:
      return false;
   }
}

bool
agx_gallium_pack_g17p_txf_sampler(
   enum agx_chip chip, struct agx_gallium_sampler_descriptor *out)
{
   if (!out)
      return false;

   memset(out, 0, sizeof(*out));

   if (agx_chip_hal300_path(chip)) {
      /* The same texel-fetch state in G17G's Gen5 record, as
       * hk_pack_native_txf_sampler() and the G17G tile reload use it: the
       * compare-mode selector is set even without a comparison. */
      agx_pack(out->opaque, SAMPLER_GEN5, cfg) {
         cfg.minimum_lod = 0.0f;
         cfg.maximum_lod = INFINITY;
         cfg.maximum_anisotropy = 1;
         cfg.mip_filter = AGX_MIP_FILTER_NEAREST;
         cfg.wrap_s = cfg.wrap_t = cfg.wrap_r = AGX_WRAP_CLAMP_TO_BORDER;
         cfg.compare_mode = true;
         cfg.compare_func = AGX_COMPARE_FUNC_LEQUAL;
         cfg.border_colour = AGX_BORDER_COLOUR_TRANSPARENT_BLACK;
      }
      return true;
   }

   /* Honeykrisp's passing G17P texture.read() path reserves sampler zero with
    * these exact HAL200 semantics. In particular, texel fetch does not use
    * the legacy background path's pixel-coordinate/linear/edge-clamp state.
    * Keep the remaining 24 bytes in the 32-byte heap slot zero. */
   const struct agx_gallium_native_sampler txf = {
      .minimum_lod = 0.0f,
      .maximum_lod = INFINITY,
      .maximum_anisotropy = 1,
      .mip_filter = AGX_MIP_FILTER_NEAREST,
      .wrap_s = AGX_WRAP_CLAMP_TO_BORDER,
      .wrap_t = AGX_WRAP_CLAMP_TO_BORDER,
      .wrap_r = AGX_WRAP_CLAMP_TO_BORDER,
      .compare_func = AGX_COMPARE_FUNC_LEQUAL,
      .border_colour = AGX_BORDER_COLOUR_TRANSPARENT_BLACK,
   };
   return agx_gallium_pack_native_sampler(chip, out, &txf);
}

bool
agx_gallium_g17p_sampler_handle(enum agx_chip chip, unsigned api_sampler,
                                uint16_t *native_handle)
{
   if (!agx_chip_agx3_path(chip) ||
       api_sampler >= AGX_GALLIUM_G17P_API_SAMPLERS || !native_handle)
      return false;

   *native_handle = api_sampler + 1;
   return true;
}

bool
agx_gallium_g17p_custom_border_handle(enum agx_chip chip,
                                      unsigned api_sampler,
                                      uint16_t *native_handle)
{
   if (!agx_chip_agx3_path(chip) ||
       api_sampler >= AGX_GALLIUM_G17P_API_SAMPLERS || !native_handle)
      return false;

   *native_handle = AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE + api_sampler;
   return true;
}

bool
agx_gallium_pack_native_texture(
   enum agx_descriptor_generation generation,
   struct agx_gallium_texture_descriptor *out,
   const struct AGX_TEXTURE *semantic)
{
   const unsigned descriptor_size =
      agx_gallium_texture_descriptor_size(generation);

   if (!out || !semantic || !descriptor_size)
      return false;

   memset(out, 0, descriptor_size);

   if (agx_gallium_descriptor_generation_is_agx2(generation)) {
      AGX_TEXTURE_pack(out->opaque, semantic);
      return true;
   }

   const bool multisampled =
      semantic->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
      semantic->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;

   if (!agx_gallium_native_texture_supported(
          generation, multisampled, semantic->buffer_offset_sw))
      return false;

   if (multisampled) {
      unsigned samples;
      switch (semantic->samples) {
      case AGX_SAMPLE_COUNT_2:
         samples = 2;
         break;
      case AGX_SAMPLE_COUNT_4:
         samples = 4;
         break;
      case AGX_SAMPLE_COUNT_8:
         samples = 8;
         break;
      default:
         return false;
      }

      if (!agx_gallium_native_texture_sample_count_representable(
             generation, samples))
         return false;
   } else if (semantic->samples != AGX_SAMPLE_COUNT_2) {
      /* Sample Count has no 1x value, so zero/2x is the inert default for a
       * non-multisampled dimension.  Do not silently accept other selectors
       * where the hardware field has no defined meaning.
       */
      return false;
   }

   const bool has_levels = semantic->first_level || semantic->last_level ||
                           semantic->unk_mipmapped;
   const bool extended = semantic->extended || has_levels;

   if ((semantic->address & 0xf) || semantic->address >> 48 ||
       (semantic->compressed &&
        ((semantic->acceleration_buffer & 0xf) ||
         semantic->acceleration_buffer >> 48)))
      return false;

   if (generation == AGX_DESCRIPTOR_GENERATION_9) {
      if (!semantic->width || semantic->width > 16384 || !semantic->height ||
          semantic->height > 16384)
         return false;

      /* Every recovered nonlinear extended Gen9 descriptor carries a second
       * VA immediately after the selected image range.  Compression metadata
       * aliases that word, while an ordinary array/mip/cube uses the end VA. */
      if (extended && semantic->layout != AGX_LAYOUT_LINEAR &&
          !semantic->acceleration_buffer)
         return false;

      /* The captured cube form is extended and includes a secondary address.
       * Preserve those requirements for a complete payload. */
      if (semantic->dimension == AGX_TEXTURE_DIMENSION_CUBE &&
          (!extended || !semantic->acceleration_buffer))
         return false;

      /* HAL200 encodeImageReadState clears word1[62] for an ordinary
       * non-array 2D texture.  Its conditional layer-mode writer is not the
       * inherited AIL page-aligned-allocation flag.  Keep this correction
       * bounded to the fully represented plain one-level case; arrays,
       * compression, and image modes retain their existing policies.
       */
      const bool plain_2d =
         semantic->dimension == AGX_TEXTURE_DIMENSION_2D &&
         semantic->layout == AGX_LAYOUT_GPU && semantic->depth == 1 &&
         !semantic->compressed && !extended &&
         semantic->mode == AGX_IMAGE_MODE_NORMAL;

      struct AGX_TEXTURE_GEN9 cfg = {
         .dimension = semantic->dimension,
         .layout = semantic->layout,
         .channels = semantic->channels,
         .type = semantic->type,
         .swizzle_r = semantic->swizzle_r,
         .swizzle_g = semantic->swizzle_g,
         .swizzle_b = semantic->swizzle_b,
         .swizzle_a = semantic->swizzle_a,
         .width = semantic->width,
         .height = semantic->height,
         .samples = semantic->samples,
         .has_levels = has_levels,
         .compressed = semantic->compressed,
         .state_control = semantic->mode | (semantic->compression << 2),
         .address = semantic->address,
         .srgb = semantic->srgb,
         .srgb_2_channel = semantic->srgb_2_channel,
         .stride = semantic->layout == AGX_LAYOUT_LINEAR ? semantic->stride : 0,
         .depth = semantic->layout == AGX_LAYOUT_LINEAR ? 1 : semantic->depth,
         .page_aligned_layers =
            plain_2d ? false : semantic->page_aligned_layers,
         .extended = extended,
         /* Gen9 reuses this wire address as the exclusive range end for an
          * ordinary extended texture, or compression metadata otherwise. */
         .acceleration_buffer = extended ? semantic->acceleration_buffer : 0,
         .depth_linear = semantic->layout == AGX_LAYOUT_LINEAR &&
                               semantic->extended
                            ? semantic->depth_linear
                            : 1,
         .layer_stride_linear = semantic->layout == AGX_LAYOUT_LINEAR &&
                                       semantic->extended
                                    ? semantic->layer_stride_linear
                                    : 0,
         .first_level = semantic->first_level,
         .last_level = semantic->last_level,
      };

      AGX_TEXTURE_GEN9_pack(out->opaque, &cfg);

      return true;
   }

   if (generation == AGX_DESCRIPTOR_GENERATION_10) {
      if (!semantic->width || semantic->width > 32768 || !semantic->height ||
          semantic->height > 32768)
         return false;

      struct AGX_TEXTURE_GEN10 cfg = {
         .dimension = semantic->dimension,
         .layout = semantic->layout,
         .channels = semantic->channels,
         .type = semantic->type,
         .swizzle_r = semantic->swizzle_r,
         .swizzle_g = semantic->swizzle_g,
         .swizzle_b = semantic->swizzle_b,
         .swizzle_a = semantic->swizzle_a,
         .width = semantic->width,
         .height = semantic->height,
         .samples = semantic->samples,
         .unk_mipmapped = semantic->unk_mipmapped,
         .compressed = semantic->compressed,
         .image_mode_enable = semantic->mode != AGX_IMAGE_MODE_NORMAL,
         .image_mode = semantic->mode,
         .address = semantic->address,
         .srgb = semantic->srgb,
         .srgb_2_channel = semantic->srgb_2_channel,
         .stride = semantic->layout == AGX_LAYOUT_LINEAR ? semantic->stride : 0,
         .depth = semantic->layout == AGX_LAYOUT_LINEAR ? 1 : semantic->depth,
         .page_aligned_layers = semantic->page_aligned_layers,
         .extended = extended,
         /* Gen10 keeps this word for compression metadata only. A captured
          * *uncompressed* mipmapped array leaves it zero, so the Gen9 habit
          * of publishing an end VA here does not carry over: on a compressed
          * image the metadata simply follows the image data, which is why
          * its value happens to equal address + layer_stride * layers. */
         .acceleration_buffer =
            semantic->compressed && semantic->extended
               ? semantic->acceleration_buffer
               : 0,
         .depth_linear = semantic->layout == AGX_LAYOUT_LINEAR &&
                               semantic->extended
                            ? semantic->depth_linear
                            : 1,
         .layer_stride_linear = semantic->layout == AGX_LAYOUT_LINEAR &&
                                       semantic->extended
                                    ? semantic->layer_stride_linear
                                    : 0,
         .first_level = semantic->first_level,
         .last_level = semantic->last_level,
      };

      AGX_TEXTURE_GEN10_pack(out->opaque, &cfg);
      return true;
   }

   return false;
}

bool
agx_gallium_set_null_native_texture(
   enum agx_descriptor_generation generation,
   struct agx_gallium_texture_descriptor *out)
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

   return agx_gallium_pack_native_texture(generation, out, &semantic);
}

/* Map the descriptor-generation enum onto the shared codec's numeric
 * generation. Zero means "no recovered AGX3 encoding". */
static uint32_t
agx_gallium_pbe_agx3_generation(enum agx_descriptor_generation generation)
{
   switch (generation) {
   case AGX_DESCRIPTOR_GENERATION_9:
      return AGX_PBE_AGX3_GENERATION_9;
   case AGX_DESCRIPTOR_GENERATION_10:
      return AGX_PBE_AGX3_GENERATION_10;
   default:
      return 0;
   }
}

static bool
agx_gallium_pack_native_pbe_impl(
   enum agx_descriptor_generation generation,
   struct agx_gallium_pbe_descriptor *out,
   const struct AGX_PBE *legacy, bool g18_mips)
{
   const unsigned descriptor_size =
      agx_gallium_pbe_descriptor_size(generation);

   if (!out || !legacy || !descriptor_size)
      return false;

   memset(out, 0, descriptor_size);
   if (!agx_gallium_native_pbe_supported(generation))
      return false;

   if (agx_gallium_descriptor_generation_is_agx2(generation)) {
      AGX_PBE_pack(out->opaque, legacy);
      return true;
   }

   const uint32_t agx3_generation =
      agx_gallium_pbe_agx3_generation(generation);
   if (!agx3_generation)
      return false;

   /* The same gate hk_pack_native_pbe() applies field for field -- the one
    * that already runs on this hardware through Honeykrisp -- so a shape is
    * admitted here exactly when Honeykrisp admits it: 2D, and on Gen9 2D
    * arrays and 3D with mips, one- and multi-layer multisample, and the
    * GPU-tiled compressed 2D packet. Gen10 adds its layered packets (2D
    * array and 3D, storage images included), which take the same native
    * mip-chain fields. Fields with no proven Gen9/Gen10 placement (sparse
    * mode, Gen10 gamma, the software-addressing sideband Gallium uses for
    * AGX2 image atomics, EMRT and spilled render targets) are refused rather
    * than dropped.
    */
   const bool gen10_multisample =
      g18_mips && agx3_generation == AGX_PBE_AGX3_GENERATION_10 &&
      (legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
       legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED);
   const bool g18_layered =
      g18_mips &&
      (legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY ||
       legacy->dimension == AGX_TEXTURE_DIMENSION_3D);
   const bool gen9 = agx3_generation == AGX_PBE_AGX3_GENERATION_9;
   const bool gen10_layered =
      agx3_generation == AGX_PBE_AGX3_GENERATION_10 &&
      (legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY ||
       legacy->dimension == AGX_TEXTURE_DIMENSION_3D);
   const bool array =
      (gen9 && legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY) ||
      (gen10_layered &&
       legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY);
   const bool array_multisample =
      (gen9 || gen10_multisample) &&
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;
   const bool three_d =
      (gen9 && legacy->dimension == AGX_TEXTURE_DIMENSION_3D) ||
      (gen10_layered && legacy->dimension == AGX_TEXTURE_DIMENSION_3D);
   const bool layered = array || array_multisample || three_d;
   const bool multisample =
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
      array_multisample;
   const bool compressed_2d =
      legacy->compressed && gen9 &&
      legacy->dimension == AGX_TEXTURE_DIMENSION_2D &&
      legacy->layout == AGX_LAYOUT_GPU && !multisample;
   if ((!layered && !multisample && !g18_layered &&
        legacy->dimension != AGX_TEXTURE_DIMENSION_2D) ||
       /* Gen10 linear 2D is agx_pbe_agx3_2d_pack()'s layout exactly. The
        * macOS 26.6 G17G driver's write descriptor for a buffer-backed
        * RGBA8Unorm texture, read back from its descriptor heap
        * (gpu/g17g-witness/argdump.m in the m5air tree):
        *
        *   256x64,    stride 1024:  00001f80ffe40a02 0003f01000001800
        *   2560x1600, stride 10240: 00031f89ffe40a02 0027f01000001800
        *
        * address >> 4 in word1[43:0] and stride - 16 from bit 40.
        *
        * On G18G it is also the J873 bgra8/gears/colour end-of-tile PBE
        * (ffc60a02 00007f80 00005800 0003f010 for BGRA8 256x256, pitch 1024),
        * which the codec reproduces; on G18G the PBE holds pitch/16-1 at
        * word1[59:44]. hk already admits it (8295c0d67c5). */
       (legacy->layout == AGX_LAYOUT_LINEAR &&
        (legacy->level != 0 || legacy->levels != 1 ||
         legacy->unk_mipmapped)) ||
       (legacy->layout != AGX_LAYOUT_LINEAR &&
        legacy->layout != AGX_LAYOUT_TWIDDLED &&
        legacy->layout != AGX_LAYOUT_GPU) ||
       legacy->unk_52 || legacy->rotate_90 || legacy->flip_vertical ||
       (multisample ? (legacy->samples != AGX_SAMPLE_COUNT_2 &&
                       legacy->samples != AGX_SAMPLE_COUNT_4)
                    : legacy->samples != AGX_SAMPLE_COUNT_2) ||
       (legacy->compressed && !compressed_2d && !gen10_multisample) ||
       legacy->mode != AGX_IMAGE_MODE_NORMAL ||
       (!layered && !g18_layered && legacy->layers != 1) ||
       (g18_layered && legacy->layout == AGX_LAYOUT_LINEAR) ||
       (multisample && (legacy->layout == AGX_LAYOUT_LINEAR ||
                        legacy->level != 0 || legacy->levels != 1 ||
                        legacy->unk_mipmapped || (gen9 && legacy->compressed))) ||
       (!gen9 && !g18_mips && (legacy->level != 0 || legacy->levels != 1 ||
                  legacy->unk_mipmapped)) ||
       (!gen9 && legacy->srgb) ||
       (legacy->extended != legacy->compressed) ||
       (legacy->layout == AGX_LAYOUT_LINEAR
           ? legacy->stride < 12 || legacy->stride > UINT32_MAX - 4
           : legacy->stride != 0) ||
       (!legacy->compressed && !gen10_multisample && legacy->acceleration_buffer) ||
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
      .stride_B =
         legacy->layout == AGX_LAYOUT_LINEAR ? legacy->stride + 4 : 0,
      .page_aligned_layers = legacy->page_aligned_layers,
      .srgb = legacy->srgb,
   };

   struct agx_pbe_agx3_descriptor packed;
   const unsigned samples = legacy->samples == AGX_SAMPLE_COUNT_4 ? 4u : 2u;
   const bool valid =
      g18_layered
         ? agx_pbe_g18_layered_mip_pack(&fields, legacy->dimension, legacy->layers,
                                      legacy->level, legacy->levels, &packed)
      /* G18G's Gen10 multisample PBE has its own encoder. */
      : multisample && gen10_multisample
         ? (array_multisample
               ? agx_pbe_g18_2d_array_multisample_pack(
                    &fields, legacy->layers, samples, &packed)
               : agx_pbe_g18_2d_multisample_pack(&fields, samples, &packed))
      : multisample
         ? (array_multisample
               ? agx_pbe_agx3_2d_array_multisample_pack(
                    agx3_generation, &fields, legacy->layers, samples,
                    &packed)
               : agx_pbe_agx3_2d_multisample_pack(agx3_generation, &fields,
                                                  samples, &packed))
      : layered
         ? (array ? agx_pbe_agx3_2d_array_mip_pack(
                       agx3_generation, &fields, legacy->layers,
                       legacy->level, legacy->levels, &packed)
                  : agx_pbe_agx3_3d_mip_pack(agx3_generation, &fields,
                                             legacy->layers, legacy->level,
                                             legacy->levels, &packed))
      : gen9 ? agx_pbe_agx3_2d_mip_pack(agx3_generation, &fields,
                                        legacy->level, legacy->levels, &packed)
      : g18_mips
         ? agx_pbe_g18_2d_mip_pack(&fields, legacy->level, legacy->levels, &packed)
         : agx_pbe_agx3_2d_pack(agx3_generation, &fields, &packed);
   if (!valid)
      return false;

   static_assert(sizeof(packed.bytes) == AGX_GALLIUM_DESCRIPTOR_AGX3_SIZE,
                 "the AGX3 PBE descriptor is the AGX3 descriptor stride");
   memcpy(out->opaque, packed.bytes, sizeof(packed.bytes));
   return true;
}

bool
agx_gallium_pack_native_pbe(
   enum agx_descriptor_generation generation,
   struct agx_gallium_pbe_descriptor *out, const struct AGX_PBE *legacy)
{
   return agx_gallium_pack_native_pbe_impl(generation, out, legacy, false);
}

bool
agx_gallium_pack_g18_native_pbe(
   struct agx_gallium_pbe_descriptor *out, const struct AGX_PBE *legacy)
{
   return agx_gallium_pack_native_pbe_impl(
      AGX_DESCRIPTOR_GENERATION_10, out, legacy, true);
}

bool
agx_gallium_set_null_native_pbe(
   enum agx_descriptor_generation generation,
   struct agx_gallium_pbe_descriptor *out)
{
   /* AGX_PBE_header already supplies the 1s for width, height, levels,
    * layers, depth_linear and the two software tile dimensions. */
   struct AGX_PBE legacy = {
      AGX_PBE_header,
      .layout = AGX_LAYOUT_TWIDDLED,
      .channels = AGX_CHANNELS_R8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_R,
      .swizzle_b = AGX_CHANNEL_R,
      .swizzle_a = AGX_CHANNEL_R,
      .buffer = AGX_SCRATCH_PAGE_ADDRESS,
   };

   /* AGX3's recovered PBE geometry is 2D only, and the null descriptor
    * Honeykrisp already runs on this hardware is 2D. The AGX2 null
    * descriptor keeps the dimension it has always had.
    */
   if (agx_gallium_descriptor_generation_is_agx3(generation))
      legacy.dimension = AGX_TEXTURE_DIMENSION_2D;

   return agx_gallium_pack_native_pbe(generation, out, &legacy);
}

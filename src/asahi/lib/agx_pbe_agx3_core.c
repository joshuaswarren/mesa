/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_pbe_agx3_core.h"
#include "agx_bytes.h"

#include <string.h>

#define AGX_PBE_DIMENSION_2D 2u
#define AGX_PBE_DIMENSION_2D_ARRAY 3u
#define AGX_PBE_DIMENSION_2D_MULTISAMPLE 4u
#define AGX_PBE_DIMENSION_3D 5u
#define AGX_PBE_DIMENSION_2D_ARRAY_MULTISAMPLE 8u
#define AGX_PBE_SAMPLE_SELECTOR_SHIFT 56u
#define AGX_PBE_LAYOUT_LINEAR 0u
#define AGX_PBE_LAYOUT_TWIDDLED 1u
#define AGX_PBE_LAYOUT_GPU 2u
/* G17P Interchange, Gen9 2D only: the Linear row-stride field, in macroblocks
 * per row (the HAL200 Gen9 PBE treats layouts 0 and 3 the same way; the
 * Interchange stride is counted in macroblocks). */
#define AGX_PBE_LAYOUT_INTERCHANGE 3u

static void
store_le64(uint8_t *dst, uint64_t value)
{
   for (unsigned i = 0; i < 8; ++i)
      dst[i] = (uint8_t)(value >> (i * 8));
}

static uint64_t
load_le64(const uint8_t *src)
{
   uint64_t value = 0;
   for (unsigned i = 0; i < 8; ++i)
      value |= (uint64_t)src[i] << (i * 8);
   return value;
}

static bool
channel_is_uncompressed(uint8_t channels)
{
   switch (channels) {
   case 0x00: /* R8 */
   case 0x09: /* R16 */
   case 0x0a: /* R8G8 */
   case 0x0b: /* R5G6B5 */
   case 0x0c: /* R4G4B4A4 */
   case 0x0d: /* A1R5G5B5 */
   case 0x0e: /* R5G5B5A1 */
   case 0x21: /* R32 */
   case 0x23: /* R16G16 */
   case 0x25: /* R11G11B10 */
   case 0x26: /* R10G10B10A2 */
   case 0x27: /* R9G9B9E5 */
   case 0x28: /* R8G8B8A8 */
   case 0x31: /* R32G32 */
   case 0x32: /* R16G16B16A16 */
   case 0x38: /* R32G32B32A32 */
      return true;
   default:
      return false;
   }
}

static bool
generation_geometry(uint32_t generation, unsigned *height_shift,
                    unsigned *dimension_bits)
{
   switch (generation) {
   case AGX_PBE_AGX3_GENERATION_9:
      *height_shift = 38;
      *dimension_bits = 14;
      return true;
   case AGX_PBE_AGX3_GENERATION_10:
      *height_shift = 39;
      *dimension_bits = 15;
      return true;
   default:
      *height_shift = 0;
      *dimension_bits = 0;
      return false;
   }
}

static bool
fields_valid(uint32_t generation,
             const struct agx_pbe_agx3_2d_fields *fields)
{
   unsigned height_shift, dimension_bits;
   if (!fields ||
       !generation_geometry(generation, &height_shift, &dimension_bits))
      return false;

   const uint32_t maximum_dimension = UINT32_C(1) << dimension_bits;
   const bool linear = fields->layout == AGX_PBE_LAYOUT_LINEAR;
   const bool interchange = fields->layout == AGX_PBE_LAYOUT_INTERCHANGE;
   const bool strided = linear || interchange;
   const bool stride_valid =
      strided ? fields->stride_B >= 16 && !(fields->stride_B & 0xf) &&
                   fields->stride_B - 16 <
                      (UINT32_C(1) <<
                       (generation == AGX_PBE_AGX3_GENERATION_10 ? 20 : 21))
              : fields->stride_B == 0;
   const bool address_valid =
      fields->buffer && !(fields->buffer & 0xf) &&
      !(fields->buffer >> (strided ? 44 : 48));
   const bool compression_valid = fields->compressed
      ? (generation == AGX_PBE_AGX3_GENERATION_9 ||
         generation == AGX_PBE_AGX3_GENERATION_10) &&
           (fields->layout == AGX_PBE_LAYOUT_GPU ||
            (interchange && generation == AGX_PBE_AGX3_GENERATION_9)) &&
           (generation != AGX_PBE_AGX3_GENERATION_10 ||
            (!(fields->buffer & 0x3f) && !fields->page_aligned_layers)) &&
           fields->acceleration_buffer &&
           !(fields->acceleration_buffer & 0xf) &&
           !(fields->acceleration_buffer >> 48)
      : !fields->acceleration_buffer;
   return (!interchange || generation == AGX_PBE_AGX3_GENERATION_9) &&
          (strided || fields->layout == AGX_PBE_LAYOUT_TWIDDLED ||
           fields->layout == AGX_PBE_LAYOUT_GPU) &&
          channel_is_uncompressed(fields->channels) && fields->type <= 5 &&
          fields->swizzle_r <= 3 && fields->swizzle_g <= 3 &&
          fields->swizzle_b <= 3 && fields->swizzle_a <= 3 && fields->width &&
          fields->width <= maximum_dimension && fields->height &&
          fields->height <= maximum_dimension && address_valid && stride_valid &&
          compression_valid;
}

bool
agx_pbe_agx3_2d_pack(uint32_t generation,
                     const struct agx_pbe_agx3_2d_fields *fields,
                     struct agx_pbe_agx3_descriptor *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields_valid(generation, fields))
      return false;

   unsigned height_shift, dimension_bits;
   if (!generation_geometry(generation, &height_shift, &dimension_bits))
      return false;

   uint64_t word0 = AGX_PBE_DIMENSION_2D;
   word0 |= (uint64_t)fields->layout << 4;
   word0 |= (uint64_t)fields->channels << 6;
   word0 |= (uint64_t)fields->type << 13;
   word0 |= (uint64_t)fields->swizzle_r << 16;
   word0 |= (uint64_t)fields->swizzle_g << 18;
   word0 |= (uint64_t)fields->swizzle_b << 20;
   word0 |= (uint64_t)fields->swizzle_a << 22;
   word0 |= (uint64_t)(fields->width - 1u) << 24;
   word0 |= (uint64_t)(fields->height - 1u) << height_shift;
   /* The native HAL300 PBE encoder puts compression at bit61,
    * whereas HAL200 uses bit59. Both extend word1 and write aux>>4 in
    * word2. Full single-level packets are checked against both oracles. */
   if (fields->compressed)
      word0 |= UINT64_C(1) <<
               (generation == AGX_PBE_AGX3_GENERATION_10 ? 61 : 59);

   /* The Gen9 PBE descriptor only sets word1 bit60 for array/type-flagged or
    * kind3/4 images. This codec admits ordinary kind2, one layer only: its
    * implicit extent and bit60 remain zero, even when the allocator padded
    * that layer to a page. The legacy allocator flag is not that Gen9 field.
    * Preserve the existing Gen10 encoding.
    */
   const bool layer_flag = generation == AGX_PBE_AGX3_GENERATION_10 &&
                           fields->page_aligned_layers;
   uint64_t word1 = fields->buffer >> 4;
   /* The HAL300 PBE descriptor holds stride-in-16-byte-units minus one at
    * word1[59:44], the byte stride divided by 16. This shares the shifted-byte
    * expression with Gen9, but retains four more address bits and one fewer
    * stride bit. */
   if (fields->layout == AGX_PBE_LAYOUT_LINEAR ||
       fields->layout == AGX_PBE_LAYOUT_INTERCHANGE)
      word1 |= (uint64_t)(fields->stride_B - 16) << 40;
   else
      word1 |= (uint64_t)layer_flag << 60;

   /* The HAL200 Gen9 PBE descriptor sets word1 bit 61 in every layout whenever
    * the texture format is sRGB-encoded, except for a few extended pixel
    * formats that select bit 62 as well. The texture descriptor carries the
    * same property as its sRGB bits, and AGX2's PBE keeps its sRGB store
    * conversion at this bit. */
   /* Gen10 (M5) converts stores to sRGB at the same word1 bit 61: an
    * R8G8B8A8_SRGB storage image and storage texel buffer stored
    * vec4(0.5, 0.2, 0.735, 0.5) as bc 7c df 80 with it and as the linear
    * 80 33 bb 80 without it (port/srgbsweep, vmfix6c). */
   word1 |= (uint64_t)fields->srgb << 61;
   if (fields->compressed)
      word1 |= UINT64_C(1) << 63;
   agx_store_le64(&out->bytes[0], word0);
   agx_store_le64(&out->bytes[8], word1);
   if (fields->compressed)
      agx_store_le64(&out->bytes[16], fields->acceleration_buffer >> 4);
   out->length = AGX_PBE_AGX3_DESCRIPTOR_LENGTH;
   return true;
}

/* Gen9 PBE mip chains: level | (levels - 1) selects the extended descriptor.
 * The level is word2[47:44], levels-1 is word2[51:48], with word0 bit58 and
 * word1 bit63 marking the extension. Using the native chain fields preserves
 * the allocation's mip tail addressing rather than pretending each level is a
 * standalone image.
 *
 * On G17P these fields depend only on the level arguments, and a tiled
 * 2D-array or 3D target keeps word2 as is (its layer count lands in word1), so
 * layered targets take the same extension.
 */
static void
mark_mip_chain(uint32_t generation, struct agx_pbe_agx3_descriptor *out, uint32_t level,
               uint32_t levels)
{
   if (!level && levels == 1)
      return;

   agx_store_le64(&out->bytes[0], agx_load_le64(&out->bytes[0]) |
                  (UINT64_C(1) << (generation == AGX_PBE_AGX3_GENERATION_10 ? 60 : 58)));
   agx_store_le64(&out->bytes[8], agx_load_le64(&out->bytes[8]) | (UINT64_C(1) << 63));
   agx_store_le64(&out->bytes[16],
              (agx_load_le64(&out->bytes[16]) & ((UINT64_C(1) << 44) - 1)) |
              ((uint64_t)level << 44) | ((uint64_t)(levels - 1) << 48));
}

/* The HAL200 Gen9 PBE descriptor (G17P) holds a layer count at word1[59:44]
 * and page-aligned layers at bit60 for every layered write target, keeping the
 * 2D geometry. A texture's own write descriptor (write type 1) takes
 * arrayLength - 1 for an array dimension whatever the length, so a one-layer
 * array is still dimension 3, and depth - 1 for a 3D texture.
 */
bool
agx_pbe_g18_layered_mip_pack(const struct agx_pbe_agx3_2d_fields *fields,
                             uint32_t dimension, uint32_t layers,
                             uint32_t level, uint32_t levels,
                             struct agx_pbe_agx3_descriptor *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields ||
       (dimension != AGX_PBE_DIMENSION_2D_ARRAY &&
        dimension != AGX_PBE_DIMENSION_3D) ||
       fields->layout == AGX_PBE_LAYOUT_LINEAR || fields->compressed ||
       layers < 1 || layers > UINT16_MAX + 1u ||
       !agx_pbe_g18_2d_mip_pack(fields, level, levels, out))
      return false;

   /* G18 storage targets carry the dimension and the layer/depth count the
    * same way. Native replay528 covers both tiled layouts, one-layer arrays
    * and nonzero depth planes. Native612 adds selected levels in mip chains:
    * the dimension/count and page flag compose with the ordinary 2D mip fields
    * without rebasing the address or dimensions. The base codec preserves the
    * layer flag. */
   uint64_t word0 = load_le64(&out->bytes[0]);
   uint64_t word1 = load_le64(&out->bytes[8]);
   word0 = (word0 & ~UINT64_C(0xf)) | dimension;
   word1 |= (uint64_t)(layers - 1) << 44;
   store_le64(&out->bytes[0], word0);
   store_le64(&out->bytes[8], word1);
   return true;
}

static bool
pack_layered(uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
             unsigned dimension, uint32_t layers, uint32_t level,
             uint32_t levels, struct agx_pbe_agx3_descriptor *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   /* The pinned native G17X Gen10 PBE encoder selects array
    * dimension 3, deposits layers-1 at word1[59:44], and
    * controls page alignment at word1[60]. Full 2/4/16-layer
    * packets agree over Gen10's own 2D width/height geometry.
    * The 3D branch selects dimension 5, then takes its
    * depth from w2 and deposits depth-1 in the same field.
    * Gen10 mip chains retain the shared level fields but move the extension
    * flag from word0 bit58 to bit60. */
   const bool m5_layered = generation == AGX_PBE_AGX3_GENERATION_10 &&
      (dimension == AGX_PBE_DIMENSION_2D_ARRAY ||
       dimension == AGX_PBE_DIMENSION_2D_ARRAY_MULTISAMPLE ||
       dimension == AGX_PBE_DIMENSION_3D);
   if ((generation != AGX_PBE_AGX3_GENERATION_9 && !m5_layered) || !fields ||
       (fields->compressed &&
        !((generation == AGX_PBE_AGX3_GENERATION_9 &&
           fields->layout == AGX_PBE_LAYOUT_GPU &&
           (dimension == AGX_PBE_DIMENSION_2D_ARRAY ||
            (level == 0 && levels == 1))) ||
          (generation == AGX_PBE_AGX3_GENERATION_10 &&
           dimension == AGX_PBE_DIMENSION_2D_ARRAY && layers == 1 &&
           level == 0 && levels == 1))) ||
       fields->layout == AGX_PBE_LAYOUT_LINEAR || layers < 1 ||
       layers > UINT16_MAX + 1u || !levels || levels > 16 ||
       level >= levels ||
       !agx_pbe_agx3_2d_pack(generation, fields, out))
      return false;

   uint64_t word0 = agx_load_le64(&out->bytes[0]);
   uint64_t word1 = agx_load_le64(&out->bytes[8]);
   word0 = (word0 & ~UINT64_C(0xf)) | dimension;
   word1 |= (uint64_t)(layers - 1) << 44;
   word1 |= (uint64_t)fields->page_aligned_layers << 60;
   agx_store_le64(&out->bytes[0], word0);
   agx_store_le64(&out->bytes[8], word1);
   mark_mip_chain(generation, out, level, levels);
   return true;
}

bool
agx_pbe_agx3_2d_array_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t layers, struct agx_pbe_agx3_descriptor *out)
{
   return pack_layered(generation, fields, AGX_PBE_DIMENSION_2D_ARRAY, layers,
                       0, 1, out);
}

bool
agx_pbe_agx3_3d_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t depth, struct agx_pbe_agx3_descriptor *out)
{
   return pack_layered(generation, fields, AGX_PBE_DIMENSION_3D, depth, 0, 1,
                       out);
}

bool
agx_pbe_agx3_2d_array_mip_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t layers, uint32_t level, uint32_t levels,
   struct agx_pbe_agx3_descriptor *out)
{
   return pack_layered(generation, fields, AGX_PBE_DIMENSION_2D_ARRAY, layers,
                       level, levels, out);
}

bool
agx_pbe_agx3_3d_mip_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t depth, uint32_t level, uint32_t levels,
   struct agx_pbe_agx3_descriptor *out)
{
   return pack_layered(generation, fields, AGX_PBE_DIMENSION_3D, depth, level,
                       levels, out);
}

/* The G18 PBE descriptor keeps Gen10's base geometry but places sample
 * selection at word0[59:58] and compression at bit61. Scalar multisample
 * targets do not inherit the page-aligned-layer flag; arrays retain both that
 * flag and their dimension even with one layer.
 */
static bool
pack_gen10_multisample(const struct agx_pbe_agx3_2d_fields *fields,
                      bool array, uint32_t layers, uint32_t samples,
                      struct agx_pbe_agx3_descriptor *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !fields || (samples != 2 && samples != 4) ||
       !layers || layers > UINT16_MAX + 1u ||
       /* G18 StateFields masks the shifted image address with
        * 0xffffffffffc. Reject unrepresentable low bits instead of dropping
        * them; metadata retains its independent 16-byte alignment. */
       (fields->buffer & 0x3f) ||
       (fields->layout != AGX_PBE_LAYOUT_TWIDDLED &&
        fields->layout != AGX_PBE_LAYOUT_GPU) ||
       (fields->compressed
           ? fields->layout != AGX_PBE_LAYOUT_GPU ||
                !fields->acceleration_buffer ||
                (fields->acceleration_buffer & 0xf) ||
                (fields->acceleration_buffer >> 48)
           : fields->acceleration_buffer != 0))
      return false;

   /* Keep the single-sample codec's admission and bit layout unchanged. */
   struct agx_pbe_agx3_2d_fields base = *fields;
   base.compressed = false;
   base.acceleration_buffer = 0;
   base.page_aligned_layers = false;
   if (!agx_pbe_agx3_2d_pack(AGX_PBE_AGX3_GENERATION_10, &base, out))
      return false;

   uint64_t word0 = load_le64(&out->bytes[0]);
   word0 = (word0 & ~UINT64_C(0xf)) |
           (array ? AGX_PBE_DIMENSION_2D_ARRAY_MULTISAMPLE
                  : AGX_PBE_DIMENSION_2D_MULTISAMPLE);
   word0 |= (uint64_t)(samples >> 2) << 58;
   word0 |= (uint64_t)fields->compressed << 61;

   uint64_t word1 = load_le64(&out->bytes[8]);
   if (array) {
      word1 |= (uint64_t)(layers - 1) << 44;
      word1 |= (uint64_t)fields->page_aligned_layers << 60;
   }
   word1 |= (uint64_t)fields->compressed << 63;
   store_le64(&out->bytes[0], word0);
   store_le64(&out->bytes[8], word1);
   if (fields->compressed)
      store_le64(&out->bytes[16], fields->acceleration_buffer >> 4);
   return true;
}

bool
agx_pbe_g18_2d_multisample_pack(const struct agx_pbe_agx3_2d_fields *fields,
                                uint32_t samples,
                                struct agx_pbe_agx3_descriptor *out)
{
   return pack_gen10_multisample(fields, false, 1, samples, out);
}

bool
agx_pbe_g18_2d_array_multisample_pack(
   const struct agx_pbe_agx3_2d_fields *fields, uint32_t layers,
   uint32_t samples, struct agx_pbe_agx3_descriptor *out)
{
   return pack_gen10_multisample(fields, true, layers, samples, out);
}

bool
agx_pbe_agx3_2d_multisample_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t samples, struct agx_pbe_agx3_descriptor *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   /* A multisample colour target is never linear and never mipmapped: the
    * end-of-tile store addresses it through the sample-expanded twiddled
    * layout. The pinned native PBE encoder deposits its sample selector at
    * word0[59:58], two bits above Gen9's. Native 2x/4x
    * complete packets ground both generations here, and the native 8x packet
    * grounds Gen10's selector 2. Tile profiles are qualified independently.
    */
   /* Preserve raw D32 copies alongside the private RGBA16 and RGBA8 color
    * cases. A compressed multisample packet is the compressed single-sample
    * packet with the multisample dimension and selector; the metadata follows
    * the sample-expanded surface, so it depends only on the texel size, which
    * RGBA8 shares with the witnessed R32_UINT shape. Gen10 keeps the sRGB
    * store conversion at word1 bit 61, as in the single-sample packet. Image
    * allocation policy remains separate from packet encoding. */
   const bool compressed_r32_uint = fields &&
      generation == AGX_PBE_AGX3_GENERATION_10 &&
      fields->layout == AGX_PBE_LAYOUT_GPU &&
      fields->channels == 0x21 /* R32 */ && fields->type == 2 /* UINT */ &&
      !fields->srgb && (samples == 2 || samples == 4);
   const bool compressed_rgba16 = fields &&
      generation == AGX_PBE_AGX3_GENERATION_10 &&
      fields->layout == AGX_PBE_LAYOUT_GPU &&
      fields->channels == 0x32 /* R16G16B16A16 */ &&
      (fields->type == 4 /* FLOAT */ || fields->type == 2 /* UINT */) &&
      !fields->srgb && (samples == 2 || samples == 4);
   const bool compressed_rgba8 = fields &&
      generation == AGX_PBE_AGX3_GENERATION_10 &&
      fields->layout == AGX_PBE_LAYOUT_GPU &&
      fields->channels == 0x28 /* R8G8B8A8 */ && fields->type == 0 /* UNORM */ &&
      (samples == 2 || samples == 4);
   if ((generation != AGX_PBE_AGX3_GENERATION_9 &&
        generation != AGX_PBE_AGX3_GENERATION_10) || !fields ||
       (fields->compressed && !compressed_r32_uint && !compressed_rgba16 &&
        !compressed_rgba8 &&
        !(generation == AGX_PBE_AGX3_GENERATION_9 &&
          fields->layout == AGX_PBE_LAYOUT_GPU)) ||
       (samples != 2 && samples != 4 &&
        !(generation == AGX_PBE_AGX3_GENERATION_10 && samples == 8)) ||
       fields->layout == AGX_PBE_LAYOUT_LINEAR ||
       !agx_pbe_agx3_2d_pack(generation, fields, out))
      return false;

   /* The Gen9 PBE leaves word0[59:53] clear apart from the selector,
    * so the two selector bits are the only ones this adds on top of the
    * single-sample packet.
    */
   const uint64_t selector = (uint64_t)((samples >> 2) & 3u);
   uint64_t word0 = agx_load_le64(&out->bytes[0]);
   word0 = (word0 & ~UINT64_C(0xf)) | AGX_PBE_DIMENSION_2D_MULTISAMPLE;
   word0 |= selector << (generation == AGX_PBE_AGX3_GENERATION_10
                            ? 58 : AGX_PBE_SAMPLE_SELECTOR_SHIFT);
   agx_store_le64(&out->bytes[0], word0);
   return true;
}

bool
agx_pbe_agx3_2d_array_multisample_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t layers, uint32_t samples,
   struct agx_pbe_agx3_descriptor *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   /* HAL200's array+multisample path selects dimension 8 while retaining the
    * layered word1 fields and the ordinary multisample selector. */
   if ((samples != 2 && samples != 4 &&
        !(generation == AGX_PBE_AGX3_GENERATION_10 && samples == 8)) ||
       !pack_layered(generation, fields,
                     AGX_PBE_DIMENSION_2D_ARRAY_MULTISAMPLE, layers, 0, 1,
                     out))
      return false;

   uint64_t word0 = agx_load_le64(&out->bytes[0]);
   word0 |= (uint64_t)((samples >> 2) & 3u)
            << (generation == AGX_PBE_AGX3_GENERATION_10
                   ? 58 : AGX_PBE_SAMPLE_SELECTOR_SHIFT);
   agx_store_le64(&out->bytes[0], word0);
   return true;
}

bool
agx_pbe_agx3_2d_mip_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t level, uint32_t levels, struct agx_pbe_agx3_descriptor *out)
{
   if (out)
      memset(out, 0, sizeof(*out));
   /* A compressed Gen10 level keeps the compressed single-level packet
    * (word0 bit61, word1 bit63, word2[43:0] = metadata >> 4) and takes the
    * same chain fields as an uncompressed one: the pinned native PBE encoder
    * (level and level count passed as arguments) gives
    * 0x30001f803fe40a22 / 0x8000000001234500 / 0x0004200004567800 for an
    * RGBA8 UNORM 64x64 level 2 of 5 with metadata 0x45678000. The metadata
    * address stays the level-0 one: the native framebuffer PBE setup asks
    * for the GPU address of level 0 of both the surface and the metadata
    * (two calls on the same path). */
   if ((generation != AGX_PBE_AGX3_GENERATION_9 &&
        generation != AGX_PBE_AGX3_GENERATION_10) || !levels || levels > 16 ||
       (fields && fields->layout == AGX_PBE_LAYOUT_LINEAR &&
        (level || levels != 1)) ||
       level >= levels || !agx_pbe_agx3_2d_pack(generation, fields, out))
      return false;

   mark_mip_chain(generation, out, level, levels);
   return true;
}

bool
agx_pbe_g18_2d_mip_pack(
   const struct agx_pbe_agx3_2d_fields *fields, uint32_t level,
   uint32_t levels, struct agx_pbe_agx3_descriptor *out)
{
   if (out)
      memset(out, 0, sizeof(*out));
   /* Uncompressed only, as when this profile was grounded: word2 below
    * carries the level fields, not compression metadata. */
   if (!fields || fields->compressed || !levels || levels > 16 ||
       level >= levels ||
       (fields->layout == AGX_PBE_LAYOUT_LINEAR && (level || levels != 1)) ||
       !agx_pbe_agx3_2d_pack(AGX_PBE_AGX3_GENERATION_10, fields, out))
      return false;

   /* G18 PBE, native375: unlike Gen9's bit58, G18 marks the chain at word0
    * bit60. The level and levels-minus-one nibbles are word2[47:44] and
    * [51:48]. Preserve the base address and dimensions so the hardware
    * performs native mip-tail addressing. */
   if (level || levels != 1) {
      store_le64(&out->bytes[0], load_le64(&out->bytes[0]) | (UINT64_C(1) << 60));
      store_le64(&out->bytes[8], load_le64(&out->bytes[8]) | (UINT64_C(1) << 63));
      store_le64(&out->bytes[16],
                 ((uint64_t)level | ((uint64_t)(levels - 1) << 4)) << 44);
   }
   return true;
}

bool
agx_pbe_agx3_2d_unpack(uint32_t generation, const void *bytes, size_t size,
                       struct agx_pbe_agx3_2d_fields *fields)
{
   if (fields)
      memset(fields, 0, sizeof(*fields));

   unsigned height_shift, dimension_bits;
   if (!bytes || !fields || size != AGX_PBE_AGX3_DESCRIPTOR_LENGTH ||
       !generation_geometry(generation, &height_shift, &dimension_bits))
      return false;

   const uint8_t *raw = bytes;
   const uint64_t word0 = agx_load_le64(&raw[0]);
   const uint64_t word1 = agx_load_le64(&raw[8]);
   const uint64_t word2 = agx_load_le64(&raw[16]);
   const uint64_t word3 = agx_load_le64(&raw[24]);
   const uint64_t word0_mask =
      generation == AGX_PBE_AGX3_GENERATION_9
         ? (UINT64_C(1) << 52) - 1
         : (UINT64_C(1) << 54) - 1;

   const uint8_t layout = (word0 >> 4) & 0x3;
   const unsigned compression_bit =
      generation == AGX_PBE_AGX3_GENERATION_10 ? 61 : 59;
   const bool compressed = (word0 & (UINT64_C(1) << compression_bit)) != 0;
   const uint64_t srgb_bit = UINT64_C(1) << 61;
   const uint64_t word1_mask = layout == AGX_PBE_LAYOUT_LINEAR
      ? ((UINT64_C(1) << 61) - 1) | srgb_bit
      : layout == AGX_PBE_LAYOUT_INTERCHANGE
      ? ((UINT64_C(1) << 61) - 1) | srgb_bit |
           (compressed ? UINT64_C(1) << 63 : 0)
      : ((UINT64_C(1) << 44) - 1) | srgb_bit |
           (compressed ? UINT64_C(1) << 63 : 0) |
           (generation == AGX_PBE_AGX3_GENERATION_10 ? UINT64_C(1) << 60 : 0);
   const uint64_t permitted_word0 =
      word0_mask | (compressed ? UINT64_C(1) << compression_bit : 0);
   if ((word0 & 0xf) != AGX_PBE_DIMENSION_2D ||
       (word0 & ~permitted_word0) || (word1 & ~word1_mask) ||
       (compressed ? (word2 & ~((UINT64_C(1) << 44) - 1)) : word2) ||
       word3)
      return false;

   fields->layout = layout;
   fields->channels = (word0 >> 6) & 0x7f;
   fields->type = (word0 >> 13) & 0x7;
   fields->swizzle_r = (word0 >> 16) & 0x3;
   fields->swizzle_g = (word0 >> 18) & 0x3;
   fields->swizzle_b = (word0 >> 20) & 0x3;
   fields->swizzle_a = (word0 >> 22) & 0x3;
   fields->width = ((word0 >> 24) & ((UINT64_C(1) << dimension_bits) - 1)) + 1;
   fields->height =
      ((word0 >> height_shift) & ((UINT64_C(1) << dimension_bits) - 1)) + 1;
   fields->srgb = (word1 & srgb_bit) != 0;
   fields->compressed = compressed;
   if (compressed)
      fields->acceleration_buffer = word2 << 4;
   if (layout == AGX_PBE_LAYOUT_LINEAR || layout == AGX_PBE_LAYOUT_INTERCHANGE) {
      fields->buffer = (word1 & ((UINT64_C(1) << 40) - 1)) << 4;
      fields->stride_B = ((word1 >> 40) & ((UINT64_C(1) << 21) - 1)) + 16;
   } else {
      fields->buffer = (word1 & ((UINT64_C(1) << 44) - 1)) << 4;
      fields->page_aligned_layers = (word1 >> 60) & 1;
   }

   if (!fields_valid(generation, fields)) {
      memset(fields, 0, sizeof(*fields));
      return false;
   }

   return true;
}

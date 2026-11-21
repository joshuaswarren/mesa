/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Every check in this test is an assert(); keep them in NDEBUG builds. */
#undef NDEBUG

#include "hk_native_descriptor_pack.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx_pbe_agx3_core.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct legacy_sampled_image_descriptor {
   uint8_t texture[24];
   uint8_t sampler[8];
   uint16_t sampler_index;
   int16_t clamp_0_sampler_index;
   uint16_t lod_bias;
   uint16_t min_lod;
   uint16_t min_lod_uint;
   uint16_t pad;
   uint32_t border[4];
   uint8_t pad2[4];
};

static void
check_layout(unsigned generation, unsigned texture_size,
             unsigned sampled_stride, unsigned storage_stride)
{
   const struct hk_sampled_image_descriptor_layout sampled =
      hk_sampled_image_descriptor_layout(generation);
   const struct hk_storage_image_descriptor_layout storage =
      hk_storage_image_descriptor_layout(generation);

   assert(hk_texture_descriptor_size(generation) == texture_size);
   assert(sampled.texture == 0);
   assert(sampled.sampler == texture_size);
   assert(sampled.sampler_index == texture_size + 8);
   assert(sampled.clamp_0_sampler_index == texture_size + 10);
   assert(sampled.lod_bias == texture_size + 12);
   assert(sampled.min_lod == texture_size + 14);
   assert(sampled.min_lod_uint == texture_size + 16);
   assert(sampled.border == texture_size + 20);
   assert(sampled.stride == sampled_stride);
   assert(storage.texture == 0);
   assert(storage.pbe == texture_size);
   assert(storage.stride == storage_stride);
}

static void
check_g17p_storage_binding_alignment(void)
{
   const struct hk_storage_image_descriptor_layout storage =
      hk_storage_image_descriptor_layout(9);

   assert(storage.stride + HK_G17P_STORAGE_SW_SIZE == 80);
   assert(HK_G17P_STORAGE_ALIGNMENT == 32);
   assert(HK_G17P_STORAGE_STRIDE == 96);

   /* CTS compute.pipeline.basic.copy_ssbo_to_image_small binds a 16-byte
    * SSBO descriptor first. The following storage-image record must advance
    * to 32 before adding the PBE half at +32. Every array element must retain
    * that alignment too. */
   const unsigned binding_offset = HK_G17P_STORAGE_ALIGNMENT;
   assert(((binding_offset + storage.pbe) & 31) == 0);
   for (unsigned element = 0; element < 8; ++element)
      assert(((binding_offset + element * HK_G17P_STORAGE_STRIDE +
               storage.pbe) & 31) == 0);
}

static void
check_gen9_layer_mode_preserved(struct AGX_TEXTURE semantic)
{
   struct hk_native_texture_descriptor unaligned, aligned;
   semantic.page_aligned_layers = false;
   assert(hk_pack_native_texture(9, &unaligned, &semantic));
   semantic.page_aligned_layers = true;
   assert(hk_pack_native_texture(9, &aligned, &semantic));

   /* Outside the corrected contract, retain exactly the inherited bit and
    * leave every other byte unchanged. This is a scope check, not native
    * qualification of the excluded descriptor kinds.
    */
   assert((unaligned.opaque[3] & (1u << 30)) == 0);
   aligned.opaque[3] ^= 1u << 30;
   assert(memcmp(&aligned, &unaligned, sizeof(aligned)) == 0);
}

static void
check_gen9_bc1_rgba_texture(void)
{
   /* Native G17P descriptor capture, EXP-0028 fmt_bc1_rgba. It pins the
    * BC1 codec selector (channels0x74 -> physical fmt bytes 0x1d22), ordinary
    * 2D layout, RGBA swizzle, 32x32 extent and the uncompressed-block VA. */
   static const uint8_t expected[32] = {
      0x22, 0x1d, 0x88, 0xf6, 0x01, 0x7c, 0x00, 0x00,
      0x50, 0x1c, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const struct AGX_TEXTURE semantic = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_BC1,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 32,
      .height = 32,
      .depth = 1,
      .address = UINT64_C(0x1000001c500),
   };
   struct hk_native_texture_descriptor packed;
   assert(hk_pack_native_texture(9, &packed, &semantic));
   assert(memcmp(&packed, expected, sizeof(expected)) == 0);
}

static void
check_gen9_etc2_rgba8_texture(void)
{
   /* Native G17P descriptor capture, EXP-0028 fmt_eac_rgba8. It pins the
    * ETC2/EAC RGBA8 codec selector (channels0x59 -> physical fmt bytes
    * 0x1662), ordinary 2D layout, RGBA swizzle, 32x32 extent and VA. */
   static const uint8_t expected[32] = {
      0x62, 0x16, 0x88, 0xf6, 0x01, 0x7c, 0x00, 0x00,
      0x50, 0x1c, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const struct AGX_TEXTURE semantic = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_ETC2_RGBA8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 32,
      .height = 32,
      .depth = 1,
      .address = UINT64_C(0x1000001c500),
   };
   struct hk_native_texture_descriptor packed;
   assert(hk_pack_native_texture(9, &packed, &semantic));
   assert(memcmp(&packed, expected, sizeof(expected)) == 0);
}

static void
check_gen9_astc_4x4_texture(void)
{
   /* Native G17P descriptor capture, EXP-0028 fmt_astc_4x4_ldr. It pins
    * the ASTC 4x4 codec selector (channels0x60 -> physical fmt bytes
    * 0x1822), ordinary 2D layout, RGBA swizzle, 32x32 extent and VA. */
   static const uint8_t expected[32] = {
      0x22, 0x18, 0x88, 0xf6, 0x01, 0x7c, 0x00, 0x00,
      0x50, 0x1c, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const struct AGX_TEXTURE semantic = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_ASTC_4X4,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 32,
      .height = 32,
      .depth = 1,
      .address = UINT64_C(0x1000001c500),
   };
   struct hk_native_texture_descriptor packed;
   assert(hk_pack_native_texture(9, &packed, &semantic));
   assert(memcmp(&packed, expected, sizeof(expected)) == 0);
}

static void
check_gen9_cube_textures(void)
{
   /* EXP-0017/EXP-M4-07 prove that an uncompressed cube is six consecutive
    * 2D Morton planes. The Vulkan probe's exact 64x64 shape has neither a
    * compression selector nor an auxiliary VA. */
   static const uint8_t uncompressed_expected[32] = {
      0x26,0x0a,0x88,0xf6,0x03,0xfc,0x00,0x00,
      0x50,0x1c,0x00,0x00,0x10,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
   };
   /* A18 Pro EXP-0015 type_cube is shader-read-only and therefore losslessly
    * compressed. Its auxiliary metadata begins after all six 16x16 faces. */
   static const uint8_t compressed_expected[32] = {
      0x26,0x0a,0x88,0xf6,0x00,0x3c,0x00,0x08,
      0x50,0x1c,0x00,0x00,0x10,0x00,0x00,0x80,
      0xd0,0x1d,0x00,0x00,0x10,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
   };
   struct AGX_TEXTURE semantic = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_CUBE,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R8G8B8A8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 64,
      .height = 64,
      .depth = 1,
      .address = UINT64_C(0x1000001c500),
   };
   struct hk_native_texture_descriptor packed;
   assert(hk_pack_native_texture(9, &packed, &semantic));
   assert(memcmp(&packed, uncompressed_expected,
                 sizeof(uncompressed_expected)) == 0);

   semantic.width = 16;
   semantic.height = 16;
   semantic.compressed = true;
   semantic.extended = true;
   semantic.acceleration_buffer = UINT64_C(0x1000001dd00);
   assert(hk_pack_native_texture(9, &packed, &semantic));
   assert(memcmp(&packed, compressed_expected,
                 sizeof(compressed_expected)) == 0);
}

static void
check_gen9_cube_array_texture(void)
{
   /* The shader-read-only capture is compressed: depth counts cubes, while
    * the compression metadata follows all twelve face planes. */
   static const uint8_t expected[32] = {
      0x27,0x0a,0x88,0xf6,0x00,0x3c,0x00,0x08,
      0x50,0x1c,0x00,0x00,0x10,0x40,0x00,0x80,
      0x50,0x1f,0x00,0x00,0x10,0x00,0x00,0x00,
      0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
   };
   const struct AGX_TEXTURE semantic = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_CUBE_ARRAY,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R8G8B8A8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 16,
      .height = 16,
      .depth = 2,
      .address = UINT64_C(0x1000001c500),
      .compressed = true,
      .extended = true,
      .acceleration_buffer = UINT64_C(0x1000001f500),
   };
   struct hk_native_texture_descriptor packed;
   assert(hk_pack_native_texture(9, &packed, &semantic));
   assert(memcmp(&packed, expected, sizeof(expected)) == 0);
}

static void
check_gen9_bc_textures(void)
{
   static const uint8_t bc1_rgba_capture[32] = {
      0x22, 0x1d, 0x88, 0xf6, 0x01, 0x7c, 0x00, 0x00,
      0x50, 0x1c, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };

   /* EXP-M4-08 captured every BC codec on A18 Pro. These are the complete
    * first descriptor words for a 32x32 ordinary 2D texture, including codec,
    * numeric type, component routing and dimensions. */
   static const struct {
      enum agx_channels channels;
      enum agx_texture_type type;
      enum agx_channel g, b, a;
      uint64_t word0;
   } cases[] = {
      {AGX_CHANNELS_BC1, AGX_TEXTURE_TYPE_UNORM, AGX_CHANNEL_G,
       AGX_CHANNEL_B, AGX_CHANNEL_A, UINT64_C(0x00007c01f6881d22)},
      {AGX_CHANNELS_BC2, AGX_TEXTURE_TYPE_UNORM, AGX_CHANNEL_G,
       AGX_CHANNEL_B, AGX_CHANNEL_A, UINT64_C(0x00007c01f6881d62)},
      {AGX_CHANNELS_BC3, AGX_TEXTURE_TYPE_UNORM, AGX_CHANNEL_G,
       AGX_CHANNEL_B, AGX_CHANNEL_A, UINT64_C(0x00007c01f6881da2)},
      {AGX_CHANNELS_BC4, AGX_TEXTURE_TYPE_UNORM, AGX_CHANNEL_0,
       AGX_CHANNEL_0, AGX_CHANNEL_1, UINT64_C(0x00007c01f9681de2)},
      {AGX_CHANNELS_BC4, AGX_TEXTURE_TYPE_SNORM, AGX_CHANNEL_0,
       AGX_CHANNEL_0, AGX_CHANNEL_1, UINT64_C(0x00007c01f9683de2)},
      {AGX_CHANNELS_BC5, AGX_TEXTURE_TYPE_UNORM, AGX_CHANNEL_G,
       AGX_CHANNEL_0, AGX_CHANNEL_1, UINT64_C(0x00007c01f9481e22)},
      {AGX_CHANNELS_BC5, AGX_TEXTURE_TYPE_SNORM, AGX_CHANNEL_G,
       AGX_CHANNEL_0, AGX_CHANNEL_1, UINT64_C(0x00007c01f9483e22)},
      {AGX_CHANNELS_BC6H, AGX_TEXTURE_TYPE_FLOAT, AGX_CHANNEL_G,
       AGX_CHANNEL_B, AGX_CHANNEL_1, UINT64_C(0x00007c01f8889e62)},
      {AGX_CHANNELS_BC6H_UFLOAT, AGX_TEXTURE_TYPE_FLOAT, AGX_CHANNEL_G,
       AGX_CHANNEL_B, AGX_CHANNEL_1, UINT64_C(0x00007c01f8889ea2)},
      {AGX_CHANNELS_BC7, AGX_TEXTURE_TYPE_UNORM, AGX_CHANNEL_G,
       AGX_CHANNEL_B, AGX_CHANNEL_A, UINT64_C(0x00007c01f6881ee2)},
   };

   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      const struct AGX_TEXTURE semantic = {
         AGX_TEXTURE_header,
         .dimension = AGX_TEXTURE_DIMENSION_2D,
         .layout = AGX_LAYOUT_GPU,
         .channels = cases[i].channels,
         .type = cases[i].type,
         .swizzle_r = AGX_CHANNEL_R,
         .swizzle_g = cases[i].g,
         .swizzle_b = cases[i].b,
         .swizzle_a = cases[i].a,
         .width = 32,
         .height = 32,
         .depth = 1,
         .address = UINT64_C(0x1000001c500),
      };
      struct hk_native_texture_descriptor packed;
      uint64_t word0;
      assert(hk_pack_native_texture(9, &packed, &semantic));
      memcpy(&word0, packed.opaque, sizeof(word0));
      assert(word0 == cases[i].word0);
      if (i == 0)
         assert(memcmp(&packed, bc1_rgba_capture,
                       sizeof(bc1_rgba_capture)) == 0);
   }
}

static void
check_buffer_metadata(void)
{
   struct AGX_TEXTURE semantic = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_LINEAR,
      .channels = AGX_CHANNELS_R32,
      .type = AGX_TEXTURE_TYPE_UINT,
      .width = 16384, .height = 1, .depth = 1,
      .stride = 16384 * 4 - 16,
      .address = UINT64_C(0x123000000000),
      .buffer_size_sw = 64,
   };
   struct hk_native_texture_descriptor native;
   assert(hk_pack_native_texture(9, &native, &semantic));
   struct AGX_TEXTURE_GEN9 hardware;
   assert(AGX_TEXTURE_GEN9_unpack(NULL, (const uint8_t *)&native, &hardware));
   assert(hardware.address == semantic.address && hardware.width == 16384);

   /* The old buffer crawl decoded these Gen9 bytes as AGX2. Its software
    * element count is absent, so every valid coordinate was rejected.
    */
   struct AGX_TEXTURE old_crawl;
   AGX_TEXTURE_unpack(NULL, (const uint8_t *)&native, &old_crawl);
   assert(old_crawl.buffer_size_sw == 0);

   const struct hk_g17p_buffer_texture_metadata metadata = {
      .elements = 64, .offset_el = 3,
      .address = semantic.address,
   };
   assert(sizeof(metadata) == 24);
   assert(AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET + offsetof(
      struct hk_g17p_buffer_texture_metadata, elements) ==
      AGX_G17P_BUFFER_TEXTURE_SIZE_OFFSET);
   assert(AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET + offsetof(
      struct hk_g17p_buffer_texture_metadata, offset_el) ==
      AGX_G17P_BUFFER_TEXTURE_TAIL_OFFSET);
   assert(AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET + offsetof(
      struct hk_g17p_buffer_texture_metadata, address) ==
      AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET);
   assert(AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET + offsetof(
      struct hk_g17p_buffer_texture_metadata, rgb32) ==
      AGX_G17P_BUFFER_TEXTURE_RGB32_OFFSET);
   assert(AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET + offsetof(
      struct hk_g17p_buffer_texture_metadata, sparse) ==
      AGX_G17P_BUFFER_TEXTURE_SPARSE_OFFSET);
   assert(AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET + sizeof(metadata) ==
          AGX_G17P_BUFFER_TEXTURE_DESCRIPTOR_SIZE);
   uint8_t descriptor[AGX_G17P_BUFFER_TEXTURE_DESCRIPTOR_SIZE] = {0};
   memcpy(descriptor, &native, sizeof(native));
   memcpy(descriptor + AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET,
          &metadata, sizeof(metadata));
   uint32_t count, tail;
   uint64_t address;
   memcpy(&count, descriptor + AGX_G17P_BUFFER_TEXTURE_SIZE_OFFSET, sizeof(count));
   memcpy(&tail, descriptor + AGX_G17P_BUFFER_TEXTURE_TAIL_OFFSET, sizeof(tail));
   memcpy(&address, descriptor + AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET, sizeof(address));
   assert(count == 64 && tail == 3 && address == semantic.address);
   assert(memcmp(descriptor, &native, sizeof(native)) == 0);

   /* A storage texel buffer is written through the Gen9 linear PBE: width
    * 16384, a row pitch of one row of texels and no software tail. The store
    * handle addresses that PBE, so its tail metadata sits PBE_OFFSET bytes
    * before the texture-relative offset.
    */
   const struct AGX_PBE pbe_semantic = {
      AGX_PBE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_LINEAR,
      .channels = AGX_CHANNELS_R32,
      .type = AGX_TEXTURE_TYPE_UINT,
      .width = 16384, .height = 1,
      .stride = 16384 * 4 - 4,
      .buffer = UINT64_C(0x4000012340),
   };
   struct hk_native_pbe_descriptor pbe;
   assert(hk_pack_native_pbe(9, &pbe, &pbe_semantic));
   struct agx_pbe_agx3_2d_fields pbe_fields;
   assert(agx_pbe_agx3_2d_unpack(9, pbe.opaque, sizeof(pbe.opaque),
                                 &pbe_fields));
   assert(pbe_fields.layout == AGX_LAYOUT_LINEAR &&
          pbe_fields.width == 16384 && pbe_fields.height == 1 &&
          pbe_fields.stride_B == 16384 * 4 &&
          pbe_fields.buffer == pbe_semantic.buffer);
   assert(hk_storage_image_descriptor_layout(9).pbe ==
          AGX_G17P_BUFFER_TEXTURE_PBE_OFFSET);

   struct AGX_PBE pbe_with_tail = pbe_semantic;
   pbe_with_tail.buffer_offset_sw = 3;
   assert(!hk_pack_native_pbe(9, &pbe, &pbe_with_tail));
}

static void
check_layered_multisample_pbe(void)
{
   struct AGX_PBE semantic = {
      AGX_PBE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R8G8B8A8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 200,
      .height = 180,
      .buffer = UINT64_C(0x123456789ab0),
      .layers = 12,
      .samples = AGX_SAMPLE_COUNT_4,
   };
   struct hk_native_pbe_descriptor packed;
   assert(hk_pack_native_pbe(9, &packed, &semantic));

   uint64_t word0, word1;
   memcpy(&word0, (const uint8_t *)packed.opaque, sizeof(word0));
   memcpy(&word1, (const uint8_t *)packed.opaque + 8, sizeof(word1));
   assert((word0 & 0xf) == 8);
   assert(((word0 >> 56) & 3) == 1);
   assert(((word1 >> 44) & 0xffff) == 11);

   semantic.samples = AGX_SAMPLE_COUNT_2;
   assert(hk_pack_native_pbe(9, &packed, &semantic));
   memcpy(&word0, (const uint8_t *)packed.opaque, sizeof(word0));
   assert(((word0 >> 56) & 3) == 0);

   semantic.layers = 0;
   assert(!hk_pack_native_pbe(9, &packed, &semantic));
   semantic.layers = 12;
   assert(hk_pack_native_pbe(10, &packed, &semantic));
   memcpy(&word0, (const uint8_t *)packed.opaque, sizeof(word0));
   memcpy(&word1, (const uint8_t *)packed.opaque + 8, sizeof(word1));
   assert((word0 & 0xf) == 8);
   assert(((word0 >> 58) & 3) == 0);
   assert(((word1 >> 44) & 0xffff) == 11);
}

static void
check_compressed_msaa_pbe(void)
{
   /* Independent offline constructor outputs, including the metadata
    * extension, for a 3024x1964 RGBA16F surface. */
   const uint64_t expected[2][4] = {
      {UINT64_C(0x2003d58bcfe48ca4), UINT64_C(0x8000000001234500),
       UINT64_C(0x4567800), 0},
      {UINT64_C(0x2403d58bcfe48ca4), UINT64_C(0x8000000001234500),
       UINT64_C(0x4567800), 0},
   };
   const uint64_t expected_read[2][4] = {
      {UINT64_C(0x203d58bcf6888ca4), UINT64_C(0x8000000001234500),
       UINT64_C(0x4567800), 0},
      {UINT64_C(0x243d58bcf6888ca4), UINT64_C(0x8000000001234500),
       UINT64_C(0x4567800), 0},
   };
   struct AGX_PBE semantic = {
      AGX_PBE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R16G16B16A16,
      .type = AGX_TEXTURE_TYPE_FLOAT,
      .swizzle_r = AGX_CHANNEL_R, .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B, .swizzle_a = AGX_CHANNEL_A,
      .width = 3024, .height = 1964,
      .buffer = UINT64_C(0x12345000),
      .compressed = true, .extended = true,
      .acceleration_buffer = UINT64_C(0x45678000),
   };
   struct hk_native_pbe_descriptor packed;
   for (unsigned i = 0; i < 2; i++) {
      semantic.samples = i ? AGX_SAMPLE_COUNT_4 : AGX_SAMPLE_COUNT_2;
      assert(hk_pack_native_pbe(10, &packed, &semantic));
      assert(!memcmp(&packed, expected[i], sizeof(expected[i])));
      assert(hk_pack_native_pbe(9, &packed, &semantic));
      const struct AGX_TEXTURE read = {
         AGX_TEXTURE_header,
         .dimension = AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED,
         .layout = AGX_LAYOUT_GPU,
         .channels = AGX_CHANNELS_R16G16B16A16,
         .type = AGX_TEXTURE_TYPE_FLOAT,
         .swizzle_r = AGX_CHANNEL_R, .swizzle_g = AGX_CHANNEL_G,
         .swizzle_b = AGX_CHANNEL_B, .swizzle_a = AGX_CHANNEL_A,
         .width = 3024, .height = 1964, .depth = 1,
         .address = UINT64_C(0x12345000),
         .compressed = true, .extended = true,
         .acceleration_buffer = UINT64_C(0x45678000),
         .samples = i ? AGX_SAMPLE_COUNT_4 : AGX_SAMPLE_COUNT_2,
      };
      struct hk_native_texture_descriptor read_packed;
      assert(hk_pack_native_texture(10, &read_packed, &read));
      assert(!memcmp(&read_packed, expected_read[i], sizeof(expected_read[i])));
   }
   semantic.samples = AGX_SAMPLE_COUNT_8;
   assert(!hk_pack_native_pbe(10, &packed, &semantic));
   semantic.samples = AGX_SAMPLE_COUNT_4;
   semantic.acceleration_buffer = 0;
   assert(!hk_pack_native_pbe(10, &packed, &semantic));
   semantic.acceleration_buffer = UINT64_C(0x45678000);
   semantic.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;
   assert(!hk_pack_native_pbe(10, &packed, &semantic));
}

static void
check_captured_rgba8_lossless_pbe(void)
{
   struct AGX_PBE p = {AGX_PBE_header};
   p.dimension = AGX_TEXTURE_DIMENSION_2D;
   p.layout = AGX_LAYOUT_GPU;
   p.channels = AGX_CHANNELS_R8G8B8A8;
   p.type = AGX_TEXTURE_TYPE_UNORM;
   p.swizzle_r = AGX_CHANNEL_R; p.swizzle_g = AGX_CHANNEL_G;
   p.swizzle_b = AGX_CHANNEL_B; p.swizzle_a = AGX_CHANNEL_A;
   p.width = p.height = 1024;
   p.buffer = UINT64_C(0x10000080000);
   p.acceleration_buffer = UINT64_C(0x10000480000);
   p.compressed = p.extended = true;
   /* Live native StateFields, compressed default texture, first record. */
   const uint64_t expected[4] = {
      UINT64_C(0x0800ffc3ffe40a22), UINT64_C(0x8000001000008000),
      UINT64_C(0x0000001000048000), 0,
   };
   struct hk_native_pbe_descriptor actual;
   assert(hk_pack_native_pbe(9, &actual, &p));
   assert(memcmp(&actual, expected, sizeof(expected)) == 0);
   const uint64_t expected10[4] = {
      UINT64_C(0x2001ff83ffe40a22), UINT64_C(0x8000001000008000),
      UINT64_C(0x0000001000048000), 0,
   };
   assert(hk_pack_native_pbe(10, &actual, &p));
   assert(memcmp(&actual, expected10, sizeof(expected10)) == 0);
   /* Pinned native Gen10 PBE encoder (offline execution, compression-oracle
    * native_pbe.py): RGBA8 UNORM 64x64, level 2 of 5, compressed. */
   {
      struct AGX_PBE m = p;
      m.width = m.height = 64;
      m.buffer = UINT64_C(0x12345000);
      m.acceleration_buffer = UINT64_C(0x45678000);
      m.level = 2;
      m.levels = 5;
      const uint64_t native_mip10[4] = {
         UINT64_C(0x30001f803fe40a22), UINT64_C(0x8000000001234500),
         UINT64_C(0x0004200004567800), 0,
      };
      assert(hk_pack_native_pbe(10, &actual, &m));
      assert(memcmp(&actual, native_mip10, sizeof(native_mip10)) == 0);
   }
   const struct AGX_PBE original = p;
#define REJECT_COMPRESSED(change) do { p = original; change; \
   assert(!hk_pack_native_pbe(9, &actual, &p)); } while (0)
   /* Compressed mip chains and the other compressed colour classes are
    * admitted by the G17P compression policy; only shapes outside the
    * compressed packet stay refused. */
   REJECT_COMPRESSED(p.acceleration_buffer = 0);
   REJECT_COMPRESSED(p.acceleration_buffer++);
   REJECT_COMPRESSED(p.extended = false);
   REJECT_COMPRESSED(p.layers = 2);
   REJECT_COMPRESSED(p.layout = AGX_LAYOUT_LINEAR);
   REJECT_COMPRESSED(p.mode = AGX_IMAGE_MODE_SPARSE);
#undef REJECT_COMPRESSED
}

/* A sampled or storage view's level-zero extent lives in the Gen10 software
 * word; a validating decode must accept it and give it back. */
static void
check_level_zero_extent_round_trip(void)
{
   const struct AGX_TEXTURE texture = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R8G8B8A8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .width = 1920,
      .height = 1080,
      .address = 0x80000,
      .first_level = 1,
      .last_level = 3,
      .unk_mipmapped = true,
   };
   struct hk_native_texture_descriptor packed;
   assert(hk_pack_native_texture(10, &packed, &texture));
   hk_native_texture_set_level_zero_extent(&packed, 960, 540);
   assert(packed.opaque[AGX_GEN10_IMAGE_EXTENT_OFFSET / 4] ==
          (960u | (540u << 16)));

   char *log = NULL;
   size_t log_size = 0;
   FILE *fp = open_memstream(&log, &log_size);
   assert(fp);
   struct AGX_TEXTURE_GEN10 gen10;
   const bool valid =
      AGX_TEXTURE_GEN10_unpack(fp, (const uint8_t *)packed.opaque, &gen10);
   fclose(fp);
   assert(valid && log_size == 0);
   free(log);
   assert(gen10.software_word == (960u | (540u << 16)));
   assert(gen10.width == 1920 && gen10.height == 1080);
   assert(gen10.first_level == 1 && gen10.last_level == 3);

   /* Repacking the decoded fields reproduces the descriptor, word 7
    * included. */
   uint32_t repacked[8];
   AGX_TEXTURE_GEN10_pack(repacked, &gen10);
   assert(!memcmp(repacked, packed.opaque, sizeof(repacked)));
}

static void
check_sparse_translation(void)
{
   const struct AGX_TEXTURE texture = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R32,
      .type = AGX_TEXTURE_TYPE_UINT,
      .width = 128,
      .height = 64,
      .address = 0x80000,
      .mode = AGX_IMAGE_MODE_SPARSE,
   };
   struct hk_native_texture_descriptor packed;
   assert(hk_pack_native_texture(10, &packed, &texture));
   struct AGX_TEXTURE_GEN10 gen10;
   assert(AGX_TEXTURE_GEN10_unpack(NULL, (const uint8_t *)packed.opaque,
                                   &gen10));
   /* The native HAL300 texture descriptor encoder: enabled with
    * page size 0x4000 maps to selector1, not the legacy sparse mode3. */
   assert(gen10.image_mode_enable);
   assert(gen10.image_mode == 1);
   assert((gen10.address & ~UINT64_C(0x30)) == texture.address);

   assert(hk_pack_native_texture(9, &packed, &texture));
   struct AGX_TEXTURE_GEN9 gen9;
   assert(AGX_TEXTURE_GEN9_unpack(NULL, (const uint8_t *)packed.opaque,
                                  &gen9));
   assert((gen9.state_control & 3) == AGX_IMAGE_MODE_SPARSE);

   /* Both native HALs emit this exact sparse-table maintenance command. */
   const uint8_t expected[] = {0x02, 0x41, 0x02, 0x60};
   const enum agx_chip chips[] = {
      AGX_CHIP_G17P, AGX_CHIP_G17G, AGX_CHIP_G17S,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(chips); ++i) {
      uint8_t stream[AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE];
      assert(agx3_cdm_sstt_flush_invalidate_pack(chips[i], stream));
      assert(memcmp(stream, expected, sizeof(expected)) == 0);
      assert(agx3_cdm_sstt_flush_invalidate_validate(chips[i], stream,
                                                   sizeof(stream)));
      assert(!agx3_cdm_sstt_flush_invalidate_validate(chips[i], stream, 3));
   }
}

/* Check each field independently of the packer's dimension dispatch. The
 * colour codec and footprint remain zero for every lossless shape. */
static void
check_lossless_layered_packets(void)
{
   static const struct {
      enum agx_texture_dimension dimension;
      unsigned encoded_dimension;
      unsigned layers;
      unsigned samples;
   } cases[] = {
      {AGX_TEXTURE_DIMENSION_2D, 2, 1, 1},
      {AGX_TEXTURE_DIMENSION_2D_ARRAY, 3, 1, 1},
      {AGX_TEXTURE_DIMENSION_2D_ARRAY, 3, 12, 1},
      {AGX_TEXTURE_DIMENSION_2D_ARRAY, 3, 65536, 1},
      {AGX_TEXTURE_DIMENSION_3D, 5, 9, 1},
      {AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED, 4, 1, 2},
      {AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED, 4, 1, 4},
      {AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED, 8, 12, 2},
      {AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED, 8, 12, 4},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      const bool layered = cases[i].encoded_dimension == 3 ||
                           cases[i].encoded_dimension == 5 ||
                           cases[i].encoded_dimension == 8;
      struct AGX_PBE p = {
         AGX_PBE_header,
         .dimension = cases[i].dimension,
         .layout = AGX_LAYOUT_GPU,
         .channels = AGX_CHANNELS_R8G8B8A8,
         .type = AGX_TEXTURE_TYPE_UNORM,
         .swizzle_r = AGX_CHANNEL_R,
         .swizzle_g = AGX_CHANNEL_G,
         .swizzle_b = AGX_CHANNEL_B,
         .swizzle_a = AGX_CHANNEL_A,
         .width = 16384,
         .height = 127,
         .buffer = UINT64_C(0x123456789ab0),
         .acceleration_buffer = UINT64_C(0x23456789abc0),
         .layers = cases[i].layers,
         .samples =
            cases[i].samples == 4 ? AGX_SAMPLE_COUNT_4 : AGX_SAMPLE_COUNT_2,
         .compressed = true,
         .extended = true,
         .page_aligned_layers = true,
      };
      struct hk_native_pbe_descriptor packed;
      assert(hk_pack_native_pbe(9, &packed, &p));
      uint64_t w[4];
      memcpy(w, packed.opaque, sizeof(w));
      assert((w[0] & 15) == cases[i].encoded_dimension);
      assert(((w[0] >> 4) & 3) == 2);
      assert(((w[0] >> 24) & 0x3fff) == p.width - 1);
      assert(((w[0] >> 38) & 0x3fff) == p.height - 1);
      assert(((w[0] >> 56) & 3) == (cases[i].samples == 4));
      assert((w[0] >> 59) == 1);
      assert((w[1] & ((UINT64_C(1) << 44) - 1)) == p.buffer >> 4);
      assert(((w[1] >> 44) & 0xffff) == p.layers - 1);
      assert(((w[1] >> 60) & 1) == layered);
      assert((w[1] >> 61) == 4);
      /* GPU-tiled compression has no explicit metadata layer stride and
       * does not combine the secondary stride and mip-range unions. */
      assert(w[2] == p.acceleration_buffer >> 4);
      assert(w[3] == 0);
      assert(!hk_pack_native_pbe(10, &packed, &p));

      const struct AGX_PBE original = p;
#define REFUSE(change)                                                         \
   do {                                                                        \
      p = original;                                                            \
      change;                                                                  \
      memset(&packed, 0xff, sizeof(packed));                                   \
      assert(!hk_pack_native_pbe(9, &packed, &p));                             \
      assert(memcmp(&packed, &(struct hk_native_pbe_descriptor){0},            \
                    sizeof(packed)) == 0);                                     \
   } while (0)
      REFUSE(p.acceleration_buffer = 0);
      REFUSE(p.acceleration_buffer++);
      REFUSE(p.acceleration_buffer = UINT64_C(1) << 48);
      REFUSE(p.buffer++);
      REFUSE(p.buffer = UINT64_C(1) << 48);
      REFUSE(p.extended = false);
      REFUSE(p.samples = AGX_SAMPLE_COUNT_8);
      REFUSE(p.layout = AGX_LAYOUT_TWIDDLED);
      if (layered) {
         REFUSE(p.layers = 0);
         REFUSE(p.layers = 65537);
         REFUSE(p.levels = 2);
      }
#undef REFUSE
   }

   static const enum agx_texture_dimension dimensions[] = {
      AGX_TEXTURE_DIMENSION_2D_ARRAY,
      AGX_TEXTURE_DIMENSION_CUBE,
      AGX_TEXTURE_DIMENSION_CUBE_ARRAY,
      AGX_TEXTURE_DIMENSION_3D,
      AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED,
      AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED,
   };
   for (unsigned i = 0; i < sizeof(dimensions) / sizeof(dimensions[0]); ++i) {
      const bool ms =
         dimensions[i] == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ||
         dimensions[i] == AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED;
      struct AGX_TEXTURE t = {
         AGX_TEXTURE_header,
         .dimension = dimensions[i],
         .layout = AGX_LAYOUT_GPU,
         .channels = AGX_CHANNELS_R8G8B8A8,
         .type = AGX_TEXTURE_TYPE_UINT,
         .width = 256,
         .height = 128,
         /* Cubes count cubes here; the allocation counts physical faces. */
         .depth =
            dimensions[i] == AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED ? 1 : 2,
         .address = UINT64_C(0x123456789ab0),
         .acceleration_buffer = UINT64_C(0x23456789abc0),
         .compressed = true,
         .extended = true,
         .page_aligned_layers = true,
         .samples = ms ? AGX_SAMPLE_COUNT_4 : AGX_SAMPLE_COUNT_2,
      };
      struct hk_native_texture_descriptor packed;
      assert(hk_pack_native_texture(9, &packed, &t));
      struct AGX_TEXTURE_GEN9 read;
      assert(
         AGX_TEXTURE_GEN9_unpack(NULL, (const uint8_t *)packed.opaque, &read));
      assert(read.dimension == t.dimension);
      assert(read.width == t.width && read.height == t.height);
      assert(read.depth == t.depth);
      assert(read.samples == t.samples);
      assert(read.compressed && read.extended);
      assert(read.address == t.address);
      assert(read.acceleration_buffer == t.acceleration_buffer);
      assert(read.page_aligned_layers);
      uint64_t w[4];
      memcpy(w, packed.opaque, sizeof(w));
      assert(((w[2] >> 60) & 3) == 0 && w[3] == 0);
      if (ms) {
         t.samples = AGX_SAMPLE_COUNT_8;
         assert(!hk_pack_native_texture(9, &packed, &t));
      } else {
         t.first_level = 1;
         t.last_level = 3;
         t.unk_mipmapped = true;
         assert(hk_pack_native_texture(9, &packed, &t));
         memcpy(w, packed.opaque, sizeof(w));
         assert(((w[2] >> 44) & 0xffff) == 0x31);
         assert((w[2] & ((UINT64_C(1) << 44) - 1)) ==
                t.acceleration_buffer >> 4);
      }
   }
}

int
main(void)
{
   check_sparse_translation();
   check_level_zero_extent_round_trip();
   check_lossless_layered_packets();
   check_buffer_metadata();
   check_layered_multisample_pbe();
   uint64_t atomic_sw = 0;
   assert(hk_pack_g17p_image_atomic_sw(0x180, 64, 32, 0x4000, 4,
                                       &atomic_sw));
   assert((atomic_sw & ((UINT64_C(1) << 27) - 1)) == 3);
   assert(((atomic_sw >> 27) & 0xf) == 6);
   assert(((atomic_sw >> 31) & 0xf) == 5);
   assert(((atomic_sw >> 35) & ((UINT64_C(1) << 27) - 1)) == 0x80);
   assert((atomic_sw >> 62) == 2);
   assert(!hk_pack_g17p_image_atomic_sw(1, 64, 64, 0x4000, 1,
                                        &atomic_sw));
   assert(!hk_pack_g17p_image_atomic_sw(0, 48, 64, 0x4000, 1,
                                        &atomic_sw));
   assert(!hk_pack_g17p_image_atomic_sw(0, 64, 64, 0x4000, 3,
                                        &atomic_sw));
   check_g17p_storage_binding_alignment();

   check_gen9_bc1_rgba_texture();
   check_gen9_etc2_rgba8_texture();
   check_gen9_astc_4x4_texture();
   check_gen9_cube_textures();
   check_gen9_cube_array_texture();
   check_gen9_bc_textures();

   /* Ordinary HAL200 2D has no layer-mode bit, even when its backing BO
    * allocation is page aligned. Compare the entire packet, not only bit62.
    */
   struct AGX_TEXTURE plain = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R8G8B8A8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R, .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B, .swizzle_a = AGX_CHANNEL_A,
      .width = 2, .height = 2, .address = 0x87ffefc000ull,
   };
   struct hk_native_texture_descriptor plain_unaligned, plain_aligned;
   for (unsigned generation = 8; generation <= 10; ++generation) {
      plain.page_aligned_layers = false;
      assert(hk_pack_native_texture(generation, &plain_unaligned, &plain));
      plain.page_aligned_layers = true;
      assert(hk_pack_native_texture(generation, &plain_aligned, &plain));
      if (generation == 9) {
         static const uint8_t expected[32] = {
            0x22,0x0a,0x88,0x16,0x00,0x04,0x00,0x00,
            0x00,0xfc,0xfe,0x7f,0x08,0x00,0x00,0x00,
         };
         assert(memcmp(&plain_aligned, expected, sizeof(expected)) == 0);
         assert(memcmp(&plain_aligned, &plain_unaligned, sizeof(plain_aligned)) == 0);
      } else {
         assert(memcmp(&plain_aligned, &plain_unaligned, sizeof(plain_aligned)) != 0);
      }
   }

   /* Each case independently crosses one plain-2D predicate boundary. In
    * particular, a one-layer array must still preserve the old array policy.
    */
   struct AGX_TEXTURE excluded = plain;
   excluded.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.dimension = AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED;
   excluded.samples = AGX_SAMPLE_COUNT_4;
   excluded.width = 64;
   excluded.height = 64;
   excluded.address = 0x5ffdea0000ull;
   excluded.page_aligned_layers = true;
   assert(hk_pack_native_texture(9, &plain_aligned, &excluded));
   static const uint8_t native_msaa4[32] = {
      0x24,0x0a,0x88,0xf6,0x03,0xfc,0x00,0x01,
      0x00,0xa0,0xde,0xff,0x05,0x00,0x00,0x00,
   };
   assert(memcmp(&plain_aligned, native_msaa4, sizeof(native_msaa4)) == 0);
   excluded.page_aligned_layers = false;
   assert(hk_pack_native_texture(9, &plain_unaligned, &excluded));
   assert(memcmp(&plain_aligned, &plain_unaligned,
                 sizeof(plain_aligned)) == 0);

   excluded = plain;
   excluded.depth = 2;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.layout = AGX_LAYOUT_LINEAR;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.layout = AGX_LAYOUT_TWIDDLED;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.first_level = 1;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.last_level = 1;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.unk_mipmapped = true;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.extended = true;
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.compressed = true;
   excluded.extended = true;
   excluded.acceleration_buffer = UINT64_C(0x100000a0000);
   check_gen9_layer_mode_preserved(excluded);

   excluded = plain;
   excluded.mode = AGX_IMAGE_MODE_SPARSE;
   check_gen9_layer_mode_preserved(excluded);

   static const struct {
      const char *chip;
      unsigned generation;
      unsigned texture_size;
      unsigned sampled_stride;
      unsigned storage_stride;
   } matrix[] = {
      {"G13 (M1/M2 class)", 7, 24, 64, 48},
      {"G14 (M2 class)", 8, 24, 64, 48},
      {"G15G (M3)", 9, 32, 72, 64},
      {"G15S (M3 Pro)", 9, 32, 72, 64},
      {"G15C/G15D (M3 Max/Ultra)", 9, 32, 72, 64},
      {"G16P (phone mapping ungrounded)", 9, 32, 72, 64},
      {"G16G (M4)", 9, 32, 72, 64},
      {"G16S (M4 Pro)", 9, 32, 72, 64},
      {"G16C (M4 Max)", 9, 32, 72, 64},
      {"G17P (A18 Pro) / G17A", 9, 32, 72, 64},
      {"G17G (M5)", 10, 32, 72, 64},
      {"G17S (M5 Pro)", 10, 32, 72, 64},
      {"G17C (M5 Max)", 10, 32, 72, 64},
   };

   for (unsigned i = 0; i < sizeof(matrix) / sizeof(matrix[0]); ++i) {
      assert(matrix[i].chip != NULL);
      check_layout(matrix[i].generation, matrix[i].texture_size,
                   matrix[i].sampled_stride, matrix[i].storage_stride);
   }

   /* The A16/G15P descriptor ABI is absent from the measured corpus. */
   assert(hk_texture_descriptor_size(0) == 0);
   assert(hk_sampled_image_descriptor_layout(0).stride == 0);
   assert(!hk_native_texture_supported(0, false, 0));
   assert(!hk_native_pbe_supported(0));

   /* Every byte of the existing AGX2 ABI remains at its old offset. */
   const struct hk_sampled_image_descriptor_layout legacy =
      hk_sampled_image_descriptor_layout(8);
   assert(sizeof(struct legacy_sampled_image_descriptor) == legacy.stride);
   assert(offsetof(struct legacy_sampled_image_descriptor, sampler) ==
          legacy.sampler);
   assert(offsetof(struct legacy_sampled_image_descriptor, sampler_index) ==
          legacy.sampler_index);
   assert(offsetof(struct legacy_sampled_image_descriptor,
                   clamp_0_sampler_index) == legacy.clamp_0_sampler_index);
   assert(offsetof(struct legacy_sampled_image_descriptor, lod_bias) ==
          legacy.lod_bias);
   assert(offsetof(struct legacy_sampled_image_descriptor, min_lod) ==
          legacy.min_lod);
   assert(offsetof(struct legacy_sampled_image_descriptor, min_lod_uint) ==
          legacy.min_lod_uint);
   assert(offsetof(struct legacy_sampled_image_descriptor, border) ==
          legacy.border);

   assert(hk_native_texture_supported(7, true, 12));
   assert(hk_native_texture_supported(9, true, 0));
   assert(hk_native_texture_supported(9, false, 0));
   assert(!hk_native_texture_supported(9, false, 4));
   assert(hk_native_texture_supported(10, true, 0));
   assert(!hk_native_texture_supported(10, false, 12));
   assert(hk_native_texture_sample_count_representable(7, 2));
   assert(hk_native_texture_sample_count_representable(8, 4));
   assert(hk_native_texture_sample_count_representable(9, 2));
   assert(hk_native_texture_sample_count_representable(9, 4));
   assert(hk_native_texture_sample_count_representable(9, 8));
   assert(!hk_native_texture_sample_count_representable(9, 16));
   assert(hk_native_texture_sample_count_representable(10, 8));
   assert(!hk_native_texture_sample_count_representable(10, 16));
   assert(!hk_native_texture_sample_count_representable(0, 1));
   assert(hk_native_pbe_supported(7));
   assert(hk_native_pbe_supported(8));
   assert(!hk_native_pbe_supported(9));
   assert(!hk_native_pbe_supported(10));

   assert(hk_native_pbe_hardware_supported(7));
   assert(hk_native_pbe_hardware_supported(8));
   assert(hk_native_pbe_hardware_supported(9));
   assert(hk_native_pbe_hardware_supported(10));
   assert(!hk_native_pbe_hardware_supported(0));

   struct hk_native_texture_descriptor null9, null10;
   assert(hk_set_null_native_texture(9, &null9));
   assert(hk_set_null_native_texture(10, &null10));
   assert(memcmp(&null9, &(struct hk_native_texture_descriptor){0},
                 sizeof(null9)) != 0);
   assert(memcmp(&null10, &(struct hk_native_texture_descriptor){0},
                 sizeof(null10)) != 0);

   struct AGX_TEXTURE semantic = {
      AGX_TEXTURE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R8G8B8A8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 256,
      .height = 128,
      .address = 0x80000,
      .compressed = true,
      .extended = true,
      .acceleration_buffer = 0xa0000,
   };
   struct agx_texture_packed legacy_packed;
   struct hk_native_texture_descriptor native_packed;
   AGX_TEXTURE_pack((uint32_t *)&legacy_packed, &semantic);
   assert(hk_pack_native_texture(8, &native_packed, &semantic));
   assert(memcmp(&legacy_packed, &native_packed, sizeof(legacy_packed)) == 0);
   for (unsigned i = sizeof(legacy_packed); i < sizeof(native_packed); ++i)
      assert(((const uint8_t *)&native_packed)[i] == 0);

   /* M5 native bytes are grounded in a live G17X texture descriptor readout. */
   static const uint8_t m5_texture[32] = {
      0x22, 0x0a, 0x88, 0xf6, 0x0f, 0xf8, 0x03, 0x20,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x80,
      0x00, 0xa0, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   semantic.address = 0x10000080000ull;
   semantic.acceleration_buffer = 0x100000a0000ull;
   assert(hk_pack_native_texture(10, &native_packed, &semantic));
   assert(memcmp(&native_packed, m5_texture, sizeof(m5_texture)) == 0);

   /* Pinned native texture encoder (offline execution, compression-oracle
    * native_read.py): a compressed 64x64 RGBA8 mip chain viewed from level 2
    * with 5 levels takes the level-0 data and metadata addresses (the
    * metadata address query passes level 0), word0 bit60 and
    * first/last level 2/6 in word2. */
   {
      struct AGX_TEXTURE mip = semantic;
      mip.width = mip.height = 64;
      mip.address = UINT64_C(0x12345000);
      mip.acceleration_buffer = UINT64_C(0x45678000);
      mip.unk_mipmapped = true;
      mip.first_level = 2;
      mip.last_level = 6;
      const uint64_t native_mip_texture[4] = {
         UINT64_C(0x3001f803f6880a22), UINT64_C(0x8000000001234500),
         UINT64_C(0x0006200004567800), 0,
      };
      assert(hk_pack_native_texture(10, &native_packed, &mip));
      assert(memcmp(&native_packed, native_mip_texture,
                    sizeof(native_mip_texture)) == 0);
   }

   enum agx_sample_count samples;
   assert(hk_translate_native_texture_sample_count(9, 2, &samples));
   assert(samples == AGX_SAMPLE_COUNT_2);
   assert(hk_translate_native_texture_sample_count(9, 4, &samples));
   assert(samples == AGX_SAMPLE_COUNT_4);
   assert(hk_translate_native_texture_sample_count(9, 8, &samples));
   assert(samples == AGX_SAMPLE_COUNT_8);
   assert(hk_translate_native_texture_sample_count(10, 8, &samples));
   assert(samples == AGX_SAMPLE_COUNT_8);
   assert(!hk_translate_native_texture_sample_count(10, 1, &samples));

   semantic.dimension = AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED;
   semantic.compressed = semantic.extended = false;
   semantic.acceleration_buffer = 0;
   static const struct {
      enum agx_sample_count samples;
      unsigned selector;
   } gen9_msaa[] = {
      {AGX_SAMPLE_COUNT_2, 0},
      {AGX_SAMPLE_COUNT_4, 1},
      {AGX_SAMPLE_COUNT_8, 2},
   };
   for (unsigned i = 0; i < sizeof(gen9_msaa) / sizeof(gen9_msaa[0]); ++i) {
      semantic.samples = gen9_msaa[i].samples;
      assert(hk_pack_native_texture(9, &native_packed, &semantic));

      uint64_t gen9_msaa_word0;
      memcpy(&gen9_msaa_word0, native_packed.opaque,
             sizeof(gen9_msaa_word0));
      assert(((gen9_msaa_word0 >> 56) & 3) == gen9_msaa[i].selector);

      struct AGX_TEXTURE_GEN9 unpacked = {0};
      assert(AGX_TEXTURE_GEN9_unpack(NULL,
                                     (const uint8_t *)native_packed.opaque,
                                     &unpacked));
      assert(unpacked.samples == gen9_msaa[i].samples);
   }

   memset(&native_packed, 0xff, sizeof(native_packed));
   semantic.samples = (enum agx_sample_count)3;
   assert(!hk_pack_native_texture(9, &native_packed, &semantic));
   assert(memcmp(&native_packed, &(struct hk_native_texture_descriptor){0},
                 sizeof(native_packed)) == 0);
   memset(&native_packed, 0xff, sizeof(native_packed));
   assert(!hk_pack_native_texture(10, &native_packed, &semantic));
   assert(memcmp(&native_packed, &(struct hk_native_texture_descriptor){0},
                 sizeof(native_packed)) == 0);

   semantic.samples = AGX_SAMPLE_COUNT_8;
   assert(hk_pack_native_texture(10, &native_packed, &semantic));
   uint64_t m5_msaa_word0;
   memcpy(&m5_msaa_word0, native_packed.opaque, sizeof(m5_msaa_word0));
   assert(((m5_msaa_word0 >> 58) & 3) == AGX_SAMPLE_COUNT_8);

   semantic.dimension = AGX_TEXTURE_DIMENSION_2D;
   memset(&native_packed, 0xff, sizeof(native_packed));
   assert(!hk_pack_native_texture(9, &native_packed, &semantic));
   assert(memcmp(&native_packed, &(struct hk_native_texture_descriptor){0},
                 sizeof(native_packed)) == 0);
   memset(&native_packed, 0xff, sizeof(native_packed));
   assert(!hk_pack_native_texture(10, &native_packed, &semantic));
   assert(memcmp(&native_packed, &(struct hk_native_texture_descriptor){0},
                 sizeof(native_packed)) == 0);

   semantic.samples = AGX_SAMPLE_COUNT_2;
   semantic.buffer_offset_sw = 1;
   assert(!hk_pack_native_texture(10, &native_packed, &semantic));
   for (unsigned i = 0; i < sizeof(native_packed); ++i)
      assert(((const uint8_t *)&native_packed)[i] == 0);

   struct AGX_PBE pbe = {
      AGX_PBE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_GPU,
      .channels = AGX_CHANNELS_R8G8B8A8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_R,
      .swizzle_g = AGX_CHANNEL_G,
      .swizzle_b = AGX_CHANNEL_B,
      .swizzle_a = AGX_CHANNEL_A,
      .width = 256,
      .height = 128,
      .buffer = 0x80000,
      .compressed = true,
      .extended = true,
      .acceleration_buffer = 0xa0000,
   };
   struct hk_native_pbe_descriptor native_pbe;

   struct agx_pbe_packed legacy_pbe;
   AGX_PBE_pack(legacy_pbe.opaque, &pbe);
   assert(hk_pack_native_pbe(8, &native_pbe, &pbe));
   assert(memcmp(&legacy_pbe, &native_pbe, sizeof(legacy_pbe)) == 0);
   for (unsigned i = sizeof(legacy_pbe); i < sizeof(native_pbe); ++i)
      assert(((const uint8_t *)&native_pbe)[i] == 0);

   /* Honeykrisp's ordinary color-store subset uses the exact shared Gen9/10
    * word geometry and leaves all unresolved sideband fields zero.
    */
   static const uint8_t gen9_pbe[32] = {
      0x22, 0x0a, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   static const uint8_t gen10_pbe[32] = {
      0x22, 0x0a, 0xe4, 0xff, 0x80, 0x3f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x10,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.buffer = 0x10000080000ull;
   pbe.compressed = false;
   pbe.extended = false;
   pbe.acceleration_buffer = 0;
   pbe.page_aligned_layers = true;
   /* Allocator padding is not Gen9's single-layer PBE word1 bit60.
    * Gen10 and the legacy-generation bridge retain their original bytes.
    */
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_pbe, sizeof(gen9_pbe)) == 0);

   /* Pure HAL200 StateFields witness for the shape of Minecraft's blocks
    * atlas: RGBA8 UNORM, GPU-tiled 2048x2048, one layer, five mips. The
    * addresses are controlled inputs, not addresses captured from the game.
    */
   static const uint8_t gen9_compressed_atlas_pbe[32] = {
      0x22, 0x0a, 0xe4, 0xff, 0xc7, 0xff, 0x01, 0x0c,
      0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x80,
      0x00, 0xa0, 0x00, 0x00, 0x10, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct AGX_PBE compressed_atlas = pbe;
   compressed_atlas.width = 2048;
   compressed_atlas.height = 2048;
   compressed_atlas.buffer = 0x10000000000ull;
   compressed_atlas.compressed = true;
   compressed_atlas.extended = true;
   compressed_atlas.acceleration_buffer = 0x100000a0000ull;
   compressed_atlas.levels = 5;
   compressed_atlas.page_aligned_layers = false;
   assert(hk_pack_native_pbe(9, &native_pbe, &compressed_atlas));
   assert(memcmp(&native_pbe, gen9_compressed_atlas_pbe,
                 sizeof(gen9_compressed_atlas_pbe)) == 0);
   for (unsigned level = 0; level < 5; ++level) {
      uint8_t mip_expected[32];
      memcpy(mip_expected, gen9_compressed_atlas_pbe,
             sizeof(mip_expected));
      mip_expected[21] = level << 4;
      compressed_atlas.level = level;
      assert(hk_pack_native_pbe(9, &native_pbe, &compressed_atlas));
      assert(memcmp(&native_pbe, mip_expected, sizeof(mip_expected)) == 0);
   }
   /* Independent pure HAL200 StateFields packets for one-mip RGBA8 color
    * targets. A single-level image has no word0 mip-chain bit; compressed
    * metadata still requires the word1 extension bit. */
   static const struct {
      unsigned width, height;
      uint64_t words[4];
   } compressed_color_targets[] = {
      {1920, 1080, {0x08010dc77fe40a22ull, 0x8000001000000000ull,
                    0x000000100000a000ull, 0}},
      {2408, 1506, {0x0801784967e40a22ull, 0x8000001000000000ull,
                    0x000000100000a000ull, 0}},
      {2560, 1536, {0x08017fc9ffe40a22ull, 0x8000001000000000ull,
                    0x000000100000a000ull, 0}},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(compressed_color_targets); ++i) {
      struct AGX_PBE color_target = compressed_atlas;
      color_target.width = compressed_color_targets[i].width;
      color_target.height = compressed_color_targets[i].height;
      color_target.level = 0;
      color_target.levels = 1;
      assert(hk_pack_native_pbe(9, &native_pbe, &color_target));
      assert(memcmp(&native_pbe, compressed_color_targets[i].words,
                    sizeof(native_pbe)) == 0);
   }
   /* The same atlas on Gen10, from the pinned native PBE encoder (offline
    * execution, compression-oracle native_pbe.py, levels = 5 at stack +0x48,
    * level at +0x44): the compressed single-level words plus the chain. */
   for (unsigned level = 0; level < 5; ++level) {
      const uint64_t gen10_atlas[4] = {
         0x3003ff87ffe40a22ull, 0x8000001000000000ull,
         0x000400100000a000ull | ((uint64_t)level << 44), 0,
      };
      compressed_atlas.level = level;
      assert(hk_pack_native_pbe(10, &native_pbe, &compressed_atlas));
      assert(memcmp(&native_pbe, gen10_atlas, sizeof(gen10_atlas)) == 0);
   }
   compressed_atlas.level = 0;
   compressed_atlas.acceleration_buffer = 0;
   assert(!hk_pack_native_pbe(9, &native_pbe, &compressed_atlas));
   compressed_atlas.acceleration_buffer = 0x100000a0001ull;
   assert(!hk_pack_native_pbe(9, &native_pbe, &compressed_atlas));

   /* Actual G17P UMD entry for RG16_UINT: channels0x23, UINT type2, 4B. */
   static const uint8_t gen9_rg16_uint_pbe[32] = {
      0xe2, 0x48, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R16G16;
   pbe.type = AGX_TEXTURE_TYPE_UINT;
   pbe.swizzle_b = 0;
   pbe.swizzle_a = 0;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rg16_uint_pbe,
                 sizeof(gen9_rg16_uint_pbe)) == 0);
   static const uint8_t gen9_rg16_sint_pbe[32] = {
      0xe2, 0x68, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.type = AGX_TEXTURE_TYPE_SINT;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rg16_sint_pbe,
                 sizeof(gen9_rg16_sint_pbe)) == 0);

   /* Actual G17P MTL10 entry is R8 channels0, unsigned-normalized type0 and
    * one byte per pixel. */
   static const uint8_t gen9_r8_unorm_pbe[32] = {
      0x22, 0x00, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R8;
   pbe.type = AGX_TEXTURE_TYPE_UNORM;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_r8_unorm_pbe,
                 sizeof(gen9_r8_unorm_pbe)) == 0);

   pbe.channels = AGX_CHANNELS_R8G8B8A8;
   pbe.type = AGX_TEXTURE_TYPE_UNORM;
   pbe.swizzle_b = AGX_CHANNEL_B;
   pbe.swizzle_a = AGX_CHANNEL_A;

   /* Native R32_UINT storage/PBE capture pins channels0x21 and UINT type2:
    * its 64-wide word starts 0x3f004862. Preserve that exact format field
    * through the Gen9 RT0 descriptor path at this test's 256x128 extent. */
   static const uint8_t gen9_r32_uint_pbe[32] = {
      0x62, 0x48, 0x00, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R32;
   pbe.type = AGX_TEXTURE_TYPE_UINT;
   pbe.swizzle_g = 0;
   pbe.swizzle_b = 0;
   pbe.swizzle_a = 0;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_r32_uint_pbe,
                 sizeof(gen9_r32_uint_pbe)) == 0);

   /* Gen9 uses the same scalar R32 channels with signed-integer type3. */
   static const uint8_t gen9_r32_sint_pbe[32] = {
      0x62, 0x68, 0x00, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.type = AGX_TEXTURE_TYPE_SINT;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_r32_sint_pbe,
                 sizeof(gen9_r32_sint_pbe)) == 0);

   /* Fresh G17P EOT key 0x08021100 identifies a two-word scalar-float pair.
    * The recovered Gen9 PBE fields carry that pair as channels0x31/FLOAT4,
    * including the G swizzle, yielding this exact RT0 descriptor. */
   static const uint8_t gen9_rg32_float_pbe[32] = {
      0x62, 0x8c, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R32G32;
   pbe.type = AGX_TEXTURE_TYPE_FLOAT;
   pbe.swizzle_g = AGX_CHANNEL_G;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rg32_float_pbe,
                 sizeof(gen9_rg32_float_pbe)) == 0);

   /* G17P EOT key0x08021300 identifies the two-word unsigned pair. */
   static const uint8_t gen9_rg32_uint_pbe[32] = {
      0x62, 0x4c, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R32G32;
   pbe.type = AGX_TEXTURE_TYPE_UINT;
   pbe.swizzle_g = AGX_CHANNEL_G;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rg32_uint_pbe,
                 sizeof(gen9_rg32_uint_pbe)) == 0);

   /* Actual G17P RGBA32_UINT entry: channels0x38, UINT type2, internal EOT
    * format11 and sixteen bytes per pixel. */
   static const uint8_t gen9_rgba32_uint_pbe[32] = {
      0x22, 0x4e, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R32G32B32A32;
   pbe.type = AGX_TEXTURE_TYPE_UINT;
   pbe.swizzle_g = AGX_CHANNEL_G;
   pbe.swizzle_b = AGX_CHANNEL_B;
   pbe.swizzle_a = AGX_CHANNEL_A;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rgba32_uint_pbe,
                 sizeof(gen9_rgba32_uint_pbe)) == 0);

   /* Actual G17P RGBA32_SFLOAT entry: channels0x38, FLOAT type4, internal EOT
    * format17 and sixteen bytes per pixel. */
   static const uint8_t gen9_rgba32_float_pbe[32] = {
      0x22, 0x8e, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.type = AGX_TEXTURE_TYPE_FLOAT;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rgba32_float_pbe,
                 sizeof(gen9_rgba32_float_pbe)) == 0);

   /* Adjacent actual G17P RGBA32_SINT entry: signed type3 and EOT format10. */
   static const uint8_t gen9_rgba32_sint_pbe[32] = {
      0x22, 0x6e, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.type = AGX_TEXTURE_TYPE_SINT;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rgba32_sint_pbe,
                 sizeof(gen9_rgba32_sint_pbe)) == 0);
   pbe.type = AGX_TEXTURE_TYPE_UINT;

   /* Actual G17P packed UINT entry: channels0x26, UINT type2, internal EOT
    * format18 and four bytes per pixel. */
   static const uint8_t gen9_rgb10a2_uint_pbe[32] = {
      0xa2, 0x49, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R10G10B10A2;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rgb10a2_uint_pbe,
                 sizeof(gen9_rgb10a2_uint_pbe)) == 0);

   /* Actual G17P MTL R16G16B16A16_UINT entry: channels0x32, UINT type2,
    * internal EOT format9 and eight bytes per pixel. */
   static const uint8_t gen9_rgba16_uint_pbe[32] = {
      0xa2, 0x4c, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R16G16B16A16;
   pbe.type = AGX_TEXTURE_TYPE_UINT;
   pbe.swizzle_g = AGX_CHANNEL_G;
   pbe.swizzle_b = AGX_CHANNEL_B;
   pbe.swizzle_a = AGX_CHANNEL_A;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rgba16_uint_pbe,
                 sizeof(gen9_rgba16_uint_pbe)) == 0);

   /* Adjacent actual G17P entry: the same four U16 lanes use SINT type3 and
    * internal EOT format8. Its native EOT program is independently pinned. */
   static const uint8_t gen9_rgba16_sint_pbe[32] = {
      0xa2, 0x6c, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.type = AGX_TEXTURE_TYPE_SINT;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rgba16_sint_pbe,
                 sizeof(gen9_rgba16_sint_pbe)) == 0);
   pbe.type = AGX_TEXTURE_TYPE_UINT;
   pbe.swizzle_b = 0;
   pbe.swizzle_a = 0;

   /* Actual G17P UMD table entry for MTL format33 (RG8_UINT) resolves to
    * channels0x0a, UINT type2 and two bytes per pixel. */
   static const uint8_t gen9_rg8_uint_pbe[32] = {
      0xa2, 0x42, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R8G8;
   pbe.type = AGX_TEXTURE_TYPE_UINT;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rg8_uint_pbe,
                 sizeof(gen9_rg8_uint_pbe)) == 0);

   /* Actual G17P MTL format13 entry: scalar channels0x00, UINT type2,
    * internal EOT format5 and one byte per pixel. */
   static const uint8_t gen9_r8_uint_pbe[32] = {
      0x22, 0x40, 0x00, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R8;
   pbe.swizzle_g = 0;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_r8_uint_pbe,
                 sizeof(gen9_r8_uint_pbe)) == 0);

   /* Adjacent actual MTL format14 uses scalar channels0x00, SINT type3,
    * internal EOT format4 and one byte per pixel. */
   static const uint8_t gen9_r8_sint_pbe[32] = {
      0x22, 0x60, 0x00, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.type = AGX_TEXTURE_TYPE_SINT;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_r8_sint_pbe,
                 sizeof(gen9_r8_sint_pbe)) == 0);

   pbe.channels = AGX_CHANNELS_R8G8;
   pbe.type = AGX_TEXTURE_TYPE_UINT;
   pbe.swizzle_g = AGX_CHANNEL_G;

   /* Adjacent actual MTL format34 resolves to the same R8G8 channels0x0a
    * with signed-integer type3 and two bytes per pixel. */
   static const uint8_t gen9_rg8_sint_pbe[32] = {
      0xa2, 0x62, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.type = AGX_TEXTURE_TYPE_SINT;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rg8_sint_pbe,
                 sizeof(gen9_rg8_sint_pbe)) == 0);

   /* Actual MTL format32 resolves to RG8 channels0x0a, signed-normalized
    * type1 and two bytes per pixel. */
   static const uint8_t gen9_rg8_snorm_pbe[32] = {
      0xa2, 0x22, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.type = AGX_TEXTURE_TYPE_SNORM;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rg8_snorm_pbe,
                 sizeof(gen9_rg8_snorm_pbe)) == 0);

   /* Actual MTL format12 resolves to R8 channels0, signed-normalized type1
   * and one byte per pixel. */
   static const uint8_t gen9_r8_snorm_pbe[32] = {
      0x22, 0x20, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R8;
   pbe.type = AGX_TEXTURE_TYPE_SNORM;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_r8_snorm_pbe,
                 sizeof(gen9_r8_snorm_pbe)) == 0);

   /* Actual G17P MTL94 is BGR10A2 UNORM: channels0x26, type0, four bytes
    * per pixel and the exact B/G/R/A PBE composition. */
   static const uint8_t gen9_bgr10a2_unorm_pbe[32] = {
      0xa2, 0x09, 0xc6, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R10G10B10A2;
   pbe.type = AGX_TEXTURE_TYPE_UNORM;
   pbe.swizzle_r = AGX_CHANNEL_B;
   pbe.swizzle_g = AGX_CHANNEL_G;
   pbe.swizzle_b = AGX_CHANNEL_R;
   pbe.swizzle_a = AGX_CHANNEL_A;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_bgr10a2_unorm_pbe,
                 sizeof(gen9_bgr10a2_unorm_pbe)) == 0);
   pbe.swizzle_r = AGX_CHANNEL_R;
   pbe.swizzle_b = AGX_CHANNEL_B;

   /* Actual G17P UMD table entry for MTL format73 (RGBA8_UINT) resolves to
    * channels0x28, UINT type2 and four bytes per pixel. */
   static const uint8_t gen9_rgba8_uint_pbe[32] = {
      0x22, 0x4a, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R8G8B8A8;
   pbe.type = AGX_TEXTURE_TYPE_UINT;
   pbe.swizzle_b = AGX_CHANNEL_B;
   pbe.swizzle_a = AGX_CHANNEL_A;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rgba8_uint_pbe,
                 sizeof(gen9_rgba8_uint_pbe)) == 0);

   /* Adjacent actual MTL format74 resolves to the same RGBA8 channels0x28
    * with signed-integer type3 and four bytes per pixel. */
   static const uint8_t gen9_rgba8_sint_pbe[32] = {
      0x22, 0x6a, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.type = AGX_TEXTURE_TYPE_SINT;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rgba8_sint_pbe,
                 sizeof(gen9_rgba8_sint_pbe)) == 0);

   /* Actual MTL format72 resolves to RGBA8 channels0x28, signed-normalized
    * type1 and four bytes per pixel. */
   static const uint8_t gen9_rgba8_snorm_pbe[32] = {
      0x22, 0x2a, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.type = AGX_TEXTURE_TYPE_SNORM;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rgba8_snorm_pbe,
                 sizeof(gen9_rgba8_snorm_pbe)) == 0);

   /* B8G8R8A8_SNORM shares MTL72's native channels/type and EOT program.
    * MTL80 independently grounds the BGR memory ordering; the PBE permutation
    * is the only difference from RGBA8_SNORM. */
   static const uint8_t gen9_bgra8_snorm_pbe[32] = {
      0x22, 0x2a, 0xc6, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.swizzle_r = AGX_CHANNEL_B;
   pbe.swizzle_b = AGX_CHANNEL_R;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_bgra8_snorm_pbe,
                 sizeof(gen9_bgra8_snorm_pbe)) == 0);
   pbe.swizzle_r = AGX_CHANNEL_R;
   pbe.swizzle_b = AGX_CHANNEL_B;

   /* Actual MTL format112 resolves to RGBA16 channels0x32,
    * signed-normalized type1 and eight bytes per pixel. */
   static const uint8_t gen9_rgba16_snorm_pbe[32] = {
      0xa2, 0x2c, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R16G16B16A16;
   pbe.type = AGX_TEXTURE_TYPE_SNORM;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rgba16_snorm_pbe,
                 sizeof(gen9_rgba16_snorm_pbe)) == 0);

   /* Actual MTL format22 resolves to R16 channels0x09, signed-normalized
    * type1 and two bytes per pixel. */
   static const uint8_t gen9_r16_snorm_pbe[32] = {
      0x62, 0x22, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R16;
   pbe.type = AGX_TEXTURE_TYPE_SNORM;
   pbe.swizzle_b = 0;
   pbe.swizzle_a = 0;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_r16_snorm_pbe,
                 sizeof(gen9_r16_snorm_pbe)) == 0);

   /* Actual MTL format20 resolves to R16 channels0x09, unsigned-normalized
    * type0 and two bytes per pixel. */
   static const uint8_t gen9_r16_unorm_pbe[32] = {
      0x62, 0x02, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R16;
   pbe.type = AGX_TEXTURE_TYPE_UNORM;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_r16_unorm_pbe,
                 sizeof(gen9_r16_unorm_pbe)) == 0);

   /* Actual MTL format23 resolves to R16 channels0x09, unsigned-integer
    * type2 and two bytes per pixel. */
   static const uint8_t gen9_r16_uint_pbe[32] = {
      0x62, 0x42, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R16;
   pbe.type = AGX_TEXTURE_TYPE_UINT;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_r16_uint_pbe,
                 sizeof(gen9_r16_uint_pbe)) == 0);

   /* Actual MTL format24 resolves to R16 channels0x09, signed-integer
    * type3 and two bytes per pixel. */
   static const uint8_t gen9_r16_sint_pbe[32] = {
      0x62, 0x62, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R16;
   pbe.type = AGX_TEXTURE_TYPE_SINT;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_r16_sint_pbe,
                 sizeof(gen9_r16_sint_pbe)) == 0);

   /* Actual MTL format25 resolves to R16 channels0x09, floating-point
    * type4 and two bytes per pixel. */
   static const uint8_t gen9_r16_float_pbe[32] = {
      0x62, 0x82, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R16;
   pbe.type = AGX_TEXTURE_TYPE_FLOAT;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_r16_float_pbe,
                 sizeof(gen9_r16_float_pbe)) == 0);

   /* Actual MTL format65 resolves to R16G16 channels0x23, floating-point
    * type4 and four bytes per pixel. */
   static const uint8_t gen9_rg16_float_pbe[32] = {
      0xe2, 0x88, 0x04, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R16G16;
   pbe.type = AGX_TEXTURE_TYPE_FLOAT;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rg16_float_pbe,
                 sizeof(gen9_rg16_float_pbe)) == 0);

   /* Actual MTL format115 resolves to RGBA16 channels0x32, floating-point
    * type4 and eight bytes per pixel. */
   static const uint8_t gen9_rgba16_float_pbe[32] = {
      0xa2, 0x8c, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   pbe.channels = AGX_CHANNELS_R16G16B16A16;
   pbe.type = AGX_TEXTURE_TYPE_FLOAT;
   pbe.swizzle_b = AGX_CHANNEL_B;
   pbe.swizzle_a = AGX_CHANNEL_A;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_rgba16_float_pbe,
                 sizeof(gen9_rgba16_float_pbe)) == 0);

   pbe.channels = AGX_CHANNELS_R8G8B8A8;
   pbe.type = AGX_TEXTURE_TYPE_UNORM;
   pbe.swizzle_g = AGX_CHANNEL_G;
   pbe.swizzle_b = AGX_CHANNEL_B;
   pbe.swizzle_a = AGX_CHANNEL_A;

   assert(hk_pack_native_pbe(10, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen10_pbe, sizeof(gen10_pbe)) == 0);
   assert(!hk_native_pbe_supported(10));
   pbe.unk_mipmapped = true;
   for (unsigned levels = 2; levels <= 16; ++levels) {
      pbe.levels = levels;
      for (unsigned level = 0; level < levels; ++level) {
         pbe.level = level;
         assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
         uint8_t expected[32];
         memcpy(expected, gen9_pbe, sizeof(expected));
         expected[7] |= 4;
         expected[15] |= 128;
         expected[21] = level << 4;
         expected[22] = levels - 1;
         assert(memcmp(&native_pbe, expected, sizeof(expected)) == 0);
      }
   }
   pbe.unk_mipmapped = false;
   pbe.level = 0;
   pbe.levels = 1;


#define CHECK_PBE_REJECT(statement_)                                           \
   do {                                                                        \
      statement_;                                                              \
      memset(&native_pbe, 0xff, sizeof(native_pbe));                           \
      assert(!hk_pack_native_pbe(9, &native_pbe, &pbe));                       \
      assert(memcmp(&native_pbe, &(struct hk_native_pbe_descriptor){0},        \
                    sizeof(native_pbe)) == 0);                                 \
   } while (0)

   CHECK_PBE_REJECT(pbe.buffer |= 1);
   pbe.buffer = 0x10000080000ull;
   pbe.layout = AGX_LAYOUT_LINEAR;
   pbe.stride = 1020; /* 1024-byte row pitch in the legacy minus-4 ABI. */
   static const uint8_t linear_pbe[32] = {
      0x02, 0x0a, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0xf0, 0x03, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, linear_pbe, sizeof(linear_pbe)) == 0);
   uint8_t gen10_linear_pbe[32];
   memcpy(gen10_linear_pbe, linear_pbe, sizeof(gen10_linear_pbe));
   /* Height expands to 15 bits, starting at descriptor bit39. */
   gen10_linear_pbe[4] = 0x80;
   gen10_linear_pbe[5] = 0x3f;
   assert(hk_pack_native_pbe(10, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen10_linear_pbe, sizeof(gen10_linear_pbe)) == 0);
   pbe.stride = 0;
   pbe.layout = AGX_LAYOUT_GPU;
   pbe.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY;
   pbe.layers = 2;
   static const uint8_t gen9_array_pbe[32] = {
      0x23, 0x0a, 0xe4, 0xff, 0xc0, 0x1f, 0x00, 0x00,
      0x00, 0x80, 0x00, 0x00, 0x10, 0x10, 0x00, 0x10,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, gen9_array_pbe,
                 sizeof(gen9_array_pbe)) == 0);
   assert(hk_pack_native_pbe(10, &native_pbe, &pbe));
   /* A one-layer array keeps dimension 3 with a zero layer field. */
   pbe.layers = 1;
   uint8_t expected_layered[32];
   memcpy(expected_layered, gen9_array_pbe, sizeof(expected_layered));
   expected_layered[13] = 0x00;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, expected_layered,
                 sizeof(expected_layered)) == 0);
   /* A 3D target is the same packet with dimension 5 and depth - 1. */
   pbe.dimension = AGX_TEXTURE_DIMENSION_3D;
   pbe.layers = 4;
   expected_layered[0] = 0x25;
   expected_layered[13] = 0x30;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, expected_layered,
                 sizeof(expected_layered)) == 0);
   assert(hk_pack_native_pbe(10, &native_pbe, &pbe));
   /* A level of a mipmapped 3D or array texture keeps the whole chain's
    * geometry and adds the native mip extension: word0 bit58, word1 bit63,
    * level at word2[47:44] and levels-1 at word2[51:48]. */
   pbe.level = 1;
   pbe.levels = 3;
   pbe.unk_mipmapped = true;
   uint8_t expected_mip[32];
   memcpy(expected_mip, expected_layered, sizeof(expected_mip));
   expected_mip[7] |= 0x04;
   expected_mip[15] |= 0x80;
   expected_mip[21] = 0x10;
   expected_mip[22] = 0x02;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, expected_mip, sizeof(expected_mip)) == 0);
   pbe.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY;
   pbe.layers = 1;
   expected_mip[0] = 0x23;
   expected_mip[13] = 0x00;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(memcmp(&native_pbe, expected_mip, sizeof(expected_mip)) == 0);
   CHECK_PBE_REJECT(pbe.level = 3);
   pbe.dimension = AGX_TEXTURE_DIMENSION_3D;
   pbe.layers = 4;
   pbe.levels = 1;
   pbe.unk_mipmapped = false;
   CHECK_PBE_REJECT(pbe.level = 1);
   pbe.level = 0;
   CHECK_PBE_REJECT(pbe.layout = AGX_LAYOUT_LINEAR);
   pbe.layout = AGX_LAYOUT_GPU;
   pbe.layers = 1;
   pbe.dimension = AGX_TEXTURE_DIMENSION_CUBE;
   CHECK_PBE_REJECT((void)0);
   pbe.dimension = AGX_TEXTURE_DIMENSION_2D;
   CHECK_PBE_REJECT(pbe.compressed = true);
   pbe.compressed = false;
   CHECK_PBE_REJECT(pbe.level = 1);
   pbe.level = 0;
   CHECK_PBE_REJECT(pbe.levels = 17);
   pbe.levels = 1;
   /* Gen9 and Gen10 convert stores to sRGB with word1 bit 61 (Gen10:
    * hardware sweep, port/srgbsweep). */
   pbe.srgb = true;
   assert(hk_pack_native_pbe(9, &native_pbe, &pbe));
   assert(((const uint8_t *)&native_pbe)[15] & 0x20);
   assert(hk_pack_native_pbe(10, &native_pbe, &pbe));
   assert(((const uint8_t *)&native_pbe)[15] & 0x20);
   pbe.srgb = false;
   CHECK_PBE_REJECT(pbe.mode = AGX_IMAGE_MODE_SPARSE);
   pbe.mode = AGX_IMAGE_MODE_NORMAL;
   CHECK_PBE_REJECT(pbe.level_offset_sw = 0x80);
   pbe.level_offset_sw = 0;
   CHECK_PBE_REJECT(pbe.aligned_width_msaa_sw = 64);
   pbe.aligned_width_msaa_sw = 0;
   CHECK_PBE_REJECT(pbe.sample_count_log2_sw = 1);
   pbe.sample_count_log2_sw = 0;
   CHECK_PBE_REJECT(pbe.tile_width_sw = 16);
   pbe.tile_width_sw = 1;
   CHECK_PBE_REJECT(pbe.tile_height_sw = 16);
   pbe.tile_height_sw = 1;
   CHECK_PBE_REJECT(pbe.layer_stride_sw = 0x80);
   pbe.layer_stride_sw = 0;
   CHECK_PBE_REJECT(pbe.buffer_offset_sw = 1);

#undef CHECK_PBE_REJECT

   struct hk_native_pbe_descriptor null_pbe9, null_pbe10;
   assert(hk_set_null_native_pbe(9, &null_pbe9));
   assert(hk_set_null_native_pbe(10, &null_pbe10));
   assert(memcmp(&null_pbe9, &(struct hk_native_pbe_descriptor){0},
                 sizeof(null_pbe9)) != 0);
   assert(memcmp(&null_pbe10, &(struct hk_native_pbe_descriptor){0},
                 sizeof(null_pbe10)) != 0);

   {
      const struct AGX_PBE linear = {
         AGX_PBE_header,
         .dimension = AGX_TEXTURE_DIMENSION_2D,
         .layout = AGX_LAYOUT_LINEAR,
         .channels = AGX_CHANNELS_R8G8B8A8,
         .type = AGX_TEXTURE_TYPE_UNORM,
         .swizzle_r = AGX_CHANNEL_R, .swizzle_g = AGX_CHANNEL_G,
         .swizzle_b = AGX_CHANNEL_B, .swizzle_a = AGX_CHANNEL_A,
         .width = 1920, .height = 1080,
         .buffer = UINT64_C(0x5ffdf6c000), .stride = 7680 - 4,
      };
      const uint8_t expected[32] = {
         0x02,0x0a,0xe4,0x7f,0x87,0x1b,0x02,0x00,
         0x00,0x6c,0xdf,0xff,0x05,0xf0,0x1d,0x00,
      };
      struct hk_native_pbe_descriptor packed;
      assert(hk_pack_native_pbe(10, &packed, &linear));
      assert(!memcmp(&packed, expected, sizeof(expected)));
      struct AGX_PBE oversized = linear;
      oversized.stride = (1u << 20) + 16 - 4;
      assert(!hk_pack_native_pbe(10, &packed, &oversized));
   }
   {
      /* Pinned native Gen10 PBE and texture encoder offline oracle, 64x64
       * RGBA8, one sample/layer/mip, data=0x12345000, aux=0x45678000.
       * Gen10 moves the PBE compression flag from Gen9 bit59 to bit61. */
      const uint64_t expected_pbe[4] = {
         UINT64_C(0x20001f803fe40a22), UINT64_C(0x8000000001234500),
         UINT64_C(0x4567800), 0,
      };
      struct AGX_PBE semantic = {
         AGX_PBE_header,
         .dimension = AGX_TEXTURE_DIMENSION_2D,
         .layout = AGX_LAYOUT_GPU,
         .channels = AGX_CHANNELS_R8G8B8A8,
         .type = AGX_TEXTURE_TYPE_UNORM,
         .swizzle_r = AGX_CHANNEL_R, .swizzle_g = AGX_CHANNEL_G,
         .swizzle_b = AGX_CHANNEL_B, .swizzle_a = AGX_CHANNEL_A,
         .width = 64, .height = 64, .buffer = UINT64_C(0x12345000),
         .compressed = true, .extended = true,
         .acceleration_buffer = UINT64_C(0x45678000),
      };
      struct hk_native_pbe_descriptor packed;
      assert(hk_pack_native_pbe(10, &packed, &semantic));
      assert(!memcmp(&packed, expected_pbe, sizeof(expected_pbe)));
      struct agx_pbe_agx3_2d_fields decoded;
      assert(agx_pbe_agx3_2d_unpack(10, &packed, sizeof(packed), &decoded));
      assert(decoded.compressed &&
             decoded.acceleration_buffer == semantic.acceleration_buffer);
      semantic.buffer += 16;
      assert(!hk_pack_native_pbe(10, &packed, &semantic));
      semantic.buffer -= 16;
      semantic.page_aligned_layers = true;
      assert(!hk_pack_native_pbe(10, &packed, &semantic));
      semantic.page_aligned_layers = false;
      semantic.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY;
      assert(hk_pack_native_pbe(10, &packed, &semantic));
      uint64_t array_expected[4];
      memcpy(array_expected, expected_pbe, sizeof(array_expected));
      array_expected[0] |= 1;
      assert(!memcmp(&packed, array_expected, sizeof(array_expected)));
      semantic.layers = 2;
      assert(!hk_pack_native_pbe(10, &packed, &semantic));
      semantic.layers = 1;
      semantic.levels = 2;
      assert(!hk_pack_native_pbe(10, &packed, &semantic));
   }
   check_compressed_msaa_pbe();
   check_captured_rgba8_lossless_pbe();
   puts("honeykrisp native descriptor geometry/policy: passed");
   return 0;
}

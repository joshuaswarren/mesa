/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact bounded Gen9/Gen10 PBE descriptor codec.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_PBE_AGX3_DESCRIPTOR_LENGTH 32u
#define AGX_PBE_AGX3_GENERATION_9      9u
#define AGX_PBE_AGX3_GENERATION_10     10u

/* Non-sparse 2D/layered PBE descriptor fields for Gen9/Gen10 (linear,
 * twiddled/GPU tiled, and GPU-tiled compressed). Gen9 scalar 2D has no
 * page-aligned-layers bit (ignored on pack, false on unpack); Gen10 retains
 * it. Unrepresented bits must be zero on unpack. */
struct agx_pbe_agx3_2d_fields {
   uint8_t layout;
   uint8_t channels;
   uint8_t type;
   uint8_t swizzle_r;
   uint8_t swizzle_g;
   uint8_t swizzle_b;
   uint8_t swizzle_a;
   uint16_t width;
   uint16_t height;
   uint64_t buffer;
   /* Gen9/10 GPU-tiled 2D lossless compression metadata base. Exactly zero when
    * compressed is false; 16-byte aligned and below bit 48 otherwise. */
   uint64_t acceleration_buffer;
   bool compressed;
   /* Linear row pitch in bytes. Zero for tiled layouts. */
   uint32_t stride_B;
   /* Legacy layout metadata: encoded only for Gen10, not ordinary Gen9 2D. */
   bool page_aligned_layers;
   /* Gen9 only: convert stores to sRGB, word1 bit 61. */
   bool srgb;
};

struct agx_pbe_agx3_descriptor {
   uint8_t bytes[AGX_PBE_AGX3_DESCRIPTOR_LENGTH];
   uint8_t length;
};

bool agx_pbe_agx3_2d_pack(
   uint32_t generation,
   const struct agx_pbe_agx3_2d_fields *fields,
   struct agx_pbe_agx3_descriptor *out);

/* Gen9 HAL200 2D-array extension. The Gen9 PBE keeps the 2D field
 * geometry, selects dimension3, and places layers-1 at word1[59:44] with
 * page-aligned-layers at bit60. A one-layer array keeps dimension3. */
bool agx_pbe_agx3_2d_array_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t layers, struct agx_pbe_agx3_descriptor *out);

bool agx_pbe_agx3_3d_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t depth, struct agx_pbe_agx3_descriptor *out);

/* The layered packets with the native mip extension of
 * agx_pbe_agx3_2d_mip_pack(). Layers or depth, width, height and buffer
 * describe the whole chain; level selects the addressed level. */
bool agx_pbe_agx3_2d_array_mip_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t layers, uint32_t level, uint32_t levels,
   struct agx_pbe_agx3_descriptor *out);

bool agx_pbe_agx3_3d_mip_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t depth, uint32_t level, uint32_t levels,
   struct agx_pbe_agx3_descriptor *out);

/* Tiled multisample colour target (dimension 4, logical width/height).
 * Sample selector `(samples >> 2) & 3` sits at word0[57:56] (Gen9: 2x/4x)
 * or word0[59:58] (Gen10: 2x/4x/8x). */
bool agx_pbe_agx3_2d_multisample_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t samples, struct agx_pbe_agx3_descriptor *out);

/* Layered multisample colour target. The Gen9 encoder combines the layered
 * dimension-8 selector and layer-count fields with the same 2x/4x sample
 * selector as the dimension-4 packet. */
bool agx_pbe_agx3_2d_array_multisample_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t layers, uint32_t samples,
   struct agx_pbe_agx3_descriptor *out);

/* G18G multisample colour targets, Gen10 descriptors. The native G18
 * StateFields geometry: sample selector at word0[59:58], compression at
 * word0 bit61 and extension at word1 bit63. The scalar packet has no
 * page-aligned-layer flag. A supplied compression metadata address is
 * encoded, not derived; storage/layout qualification remains the caller's
 * responsibility. Gen10 tiled image addresses require 64-byte alignment;
 * compression metadata requires 16-byte alignment. 2x and 4x only. G18G
 * callers use these in place of the generic entry points above, whose Gen10
 * form is the G17G-family (agx3c) one. */
bool agx_pbe_g18_2d_multisample_pack(
   const struct agx_pbe_agx3_2d_fields *fields, uint32_t samples,
   struct agx_pbe_agx3_descriptor *out);

/* The array form keeps the layered geometry with its own sample/compression
 * placement, including a one-layer array. */
bool agx_pbe_g18_2d_array_multisample_pack(
   const struct agx_pbe_agx3_2d_fields *fields, uint32_t layers,
   uint32_t samples, struct agx_pbe_agx3_descriptor *out);

/* Gen9 native mip extension: the existing descriptor is unchanged for
 * level=0, levels=1. Width/height and buffer describe the complete mip chain. */
bool agx_pbe_agx3_2d_mip_pack(
   uint32_t generation, const struct agx_pbe_agx3_2d_fields *fields,
   uint32_t level, uint32_t levels, struct agx_pbe_agx3_descriptor *out);

/* G18 PBE ordinary tiled 2D mip chain. G17G is not admitted by this entry
 * point's evidence. Width/height/buffer describe the base allocation, not the
 * selected level. */
bool agx_pbe_g18_2d_mip_pack(
   const struct agx_pbe_agx3_2d_fields *fields, uint32_t level,
   uint32_t levels, struct agx_pbe_agx3_descriptor *out);

/* G18 PBE, storage-write mode: uncompressed tiled array (dimension3) or 3D
 * (dimension5) with native mip selection. Buffer is already rebased to the
 * first addressed slice; layers is the remaining layer/depth count. Base
 * allocation dimensions/address and the independently supplied page flag are
 * preserved. */
bool agx_pbe_g18_layered_mip_pack(
   const struct agx_pbe_agx3_2d_fields *fields, uint32_t dimension,
   uint32_t layers, uint32_t level, uint32_t levels,
   struct agx_pbe_agx3_descriptor *out);

bool agx_pbe_agx3_2d_unpack(
   uint32_t generation, const void *bytes, size_t size,
   struct agx_pbe_agx3_2d_fields *fields);

#ifdef __cplusplus
}
#endif

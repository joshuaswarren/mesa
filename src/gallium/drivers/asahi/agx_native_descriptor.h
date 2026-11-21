/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/genxml/agx_pack.h"
#include "asahi/lib/agx_chip.h"

#include <stdbool.h>
#include <stdint.h>

/* Gallium used struct agx_texture_packed as both the descriptor and its array
 * geometry.  That happens to be correct for Gen7/8, where texture and PBE
 * descriptors are 24 bytes, but it aliases adjacent entries once Gen9 grows
 * both descriptors to 32 bytes.  Keep the native allocation opaque so a C
 * type can never silently select the AGX2 stride again.
 */
#define AGX_GALLIUM_DESCRIPTOR_AGX2_SIZE 24u
#define AGX_GALLIUM_DESCRIPTOR_AGX3_SIZE 32u
#define AGX_GALLIUM_DESCRIPTOR_MAX_SIZE  AGX_GALLIUM_DESCRIPTOR_AGX3_SIZE
#define AGX_GALLIUM_SAMPLER_AGX3_SIZE    32u
#define AGX_GALLIUM_G17P_API_SAMPLERS    32u

/* Custom borders are implemented like Honeykrisp: the ordinary API sampler
 * bank clamps to one and a second bank clamps to zero.  The shader combines
 * those samples with the exact API border colour. */
#define AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE \
   (1u + AGX_GALLIUM_G17P_API_SAMPLERS)

/* LOD queries cannot carry the software bias used by ordinary samples.
 * Their private sampler bank keeps the API bias in the native descriptor. */
#define AGX_GALLIUM_G17P_QUERY_LOD_BASE \
   (AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE + AGX_GALLIUM_G17P_API_SAMPLERS)

/* Shadow gathers need an unfiltered border mask, independent of the depth
 * comparison. The zero companion is already non-comparing; this is its
 * clamp-to-one partner. Keep the ordinary comparison sampler unchanged. */
#define AGX_GALLIUM_G17P_BORDER_NONCOMPARE_BASE \
   (AGX_GALLIUM_G17P_QUERY_LOD_BASE + AGX_GALLIUM_G17P_API_SAMPLERS)

/* HAL200's immediate sampler operand names native slots 0..15. Gallium keeps
 * slot zero for texel fetch, leaving API samplers 0..14 directly encodable.
 * Later API samplers use an indirect native-table index instead. */
#define AGX_GALLIUM_G17P_IMMEDIATE_API_SAMPLERS 15u

struct agx_gallium_texture_descriptor {
   uint32_t opaque[AGX_GALLIUM_DESCRIPTOR_MAX_SIZE / 4];
};

struct agx_gallium_pbe_descriptor {
   uint32_t opaque[AGX_GALLIUM_DESCRIPTOR_MAX_SIZE / 4];
};

struct agx_gallium_sampler_descriptor {
   uint32_t opaque[AGX_GALLIUM_SAMPLER_AGX3_SIZE / 4];
};

struct agx_gallium_storage_descriptor_layout {
   uint8_t texture;
   uint8_t pbe;
   uint8_t stride;
};

static inline bool
agx_gallium_descriptor_generation_is_agx2(
   enum agx_descriptor_generation generation)
{
   return generation == AGX_DESCRIPTOR_GENERATION_7 ||
          generation == AGX_DESCRIPTOR_GENERATION_8;
}

static inline bool
agx_gallium_descriptor_generation_is_agx3(
   enum agx_descriptor_generation generation)
{
   return generation == AGX_DESCRIPTOR_GENERATION_9 ||
          generation == AGX_DESCRIPTOR_GENERATION_10;
}

static inline unsigned
agx_gallium_texture_descriptor_size(
   enum agx_descriptor_generation generation)
{
   return agx_gallium_descriptor_generation_is_agx2(generation)
             ? AGX_GALLIUM_DESCRIPTOR_AGX2_SIZE
          : agx_gallium_descriptor_generation_is_agx3(generation)
             ? AGX_GALLIUM_DESCRIPTOR_AGX3_SIZE
             : 0;
}

static inline unsigned
agx_gallium_pbe_descriptor_size(enum agx_descriptor_generation generation)
{
   return agx_gallium_texture_descriptor_size(generation);
}

static inline struct agx_gallium_storage_descriptor_layout
agx_gallium_storage_descriptor_layout(
   enum agx_descriptor_generation generation)
{
   const unsigned texture = agx_gallium_texture_descriptor_size(generation);
   const unsigned pbe = agx_gallium_pbe_descriptor_size(generation);

   if (!texture || !pbe)
      return (struct agx_gallium_storage_descriptor_layout){0};

   return (struct agx_gallium_storage_descriptor_layout){
      .texture = 0,
      .pbe = texture,
      .stride = texture + pbe,
   };
}

/* AGX3 buffer descriptors have no recovered equivalent for AGX2's software
 * tail offset.  Keep those unavailable rather than aliasing unknown bits in a
 * native descriptor.  Multisample descriptor representation is recovered on
 * both AGX3 generations; this is not by itself an execution/capability gate.
 */
static inline bool
agx_gallium_native_texture_supported(
   enum agx_descriptor_generation generation, bool multisampled,
   unsigned buffer_tail_offset)
{
   if (agx_gallium_descriptor_generation_is_agx2(generation))
      return true;

   if (!agx_gallium_descriptor_generation_is_agx3(generation))
      return false;

   return buffer_tail_offset == 0;
}

/* Return whether the native sampled-texture descriptor has a recovered
 * representation for a numeric sample count. This is not a Gallium/API
 * capability predicate: the live G17S corpus has descriptor bytes but no
 * executed 8x sampling command, and the complete AGX3 path remains gated.
 */
static inline bool
agx_gallium_native_texture_sample_count_representable(
   enum agx_descriptor_generation generation, unsigned samples)
{
   if (samples == 1)
      return agx_gallium_texture_descriptor_size(generation) != 0;

   if (agx_gallium_descriptor_generation_is_agx2(generation))
      return samples == 2 || samples == 4;

   if (generation == AGX_DESCRIPTOR_GENERATION_9 ||
       generation == AGX_DESCRIPTOR_GENERATION_10)
      return samples == 2 || samples == 4 || samples == 8;

   return false;
}

/* Whether the generation has any recovered PBE descriptor encoding at all.
 *
 * This is a generation predicate, not a per-image one.  AGX3's recovered PBE
 * geometry is a bounded subset -- non-sparse 2D (on Gen9 also arrays, 3D,
 * mips and multisample, and the GPU-tiled compressed single-layer 2D packet)
 * with no software addressing sideband -- and that subset is enforced per
 * descriptor by agx_gallium_pack_native_pbe(), which fails closed for
 * anything outside it.  Gallium still feeds AGX2 sideband state (image
 * atomics, EMRT, spilled render targets) through the same packet, so those
 * images are refused rather than silently encoded with the bits dropped.
 */
static inline bool
agx_gallium_native_pbe_supported(
   enum agx_descriptor_generation generation)
{
   return agx_gallium_descriptor_generation_is_agx2(generation) ||
          agx_gallium_descriptor_generation_is_agx3(generation);
}

/* Whether Gallium may write any compressed image, shared ones included.
 *
 * Gallium decompresses a compressed render target in place when its
 * tilebuffer spills it, and a shared one when a feedback loop samples it
 * (agx_decompress_inplace()). That kernel takes the AGX2 24+24-byte
 * descriptor pair and has no AGX3 build, so only AGX2 qualifies. On AGX3
 * the compressed PBE itself is encoded (Gen9 GPU-tiled 2D, see
 * agx_gallium_pack_native_pbe()); G17P compresses its private images within
 * agx_g17p_compression_color_format(), which never spill, and imported
 * compressed images stay sample-only (external) there.
 */
static inline bool
agx_gallium_shared_compression_supported(
   enum agx_descriptor_generation generation)
{
   return agx_gallium_descriptor_generation_is_agx2(generation);
}

struct AGX_TEXTURE;
struct AGX_PBE;

/* Classify the three border colours represented directly by every native AGX
 * sampler generation. Returns false for colours that require shader-backed
 * custom-border emulation. */
bool agx_gallium_classify_standard_border(
   const uint32_t color[4], bool integer,
   enum agx_border_colour *out);

bool agx_gallium_pack_native_texture(
   enum agx_descriptor_generation generation,
   struct agx_gallium_texture_descriptor *out,
   const struct AGX_TEXTURE *semantic);

/* Sampler state of one native per-stage table entry. */
struct agx_gallium_native_sampler {
   float minimum_lod;
   float maximum_lod;
   /* Applied only where the sampler word has a bias field
    * (agx_chip_sampler_has_lod_bias()); elsewhere the shader applies it. */
   float lod_bias;
   unsigned maximum_anisotropy;
   enum agx_filter magnify;
   enum agx_filter minify;
   enum agx_mip_filter mip_filter;
   enum agx_wrap wrap_s;
   enum agx_wrap wrap_t;
   enum agx_wrap wrap_r;
   bool pixel_coordinates;
   bool compare_enable;
   enum agx_compare_func compare_func;
   enum agx_border_colour border_colour;
   bool seamful_cube_maps;
};

/* Pack a native sampler table entry in the chip's sampler layout: HAL200 on
 * G17P, Gen4 on G15. */
bool agx_gallium_pack_native_sampler(
   enum agx_chip chip, struct agx_gallium_sampler_descriptor *out,
   const struct agx_gallium_native_sampler *state);

/* Exact sampler zero used by native G17P texture.read()/texel-fetch paths. */
bool agx_gallium_pack_g17p_txf_sampler(
   enum agx_chip chip, struct agx_gallium_sampler_descriptor *out);

/* Convert a Gallium API sampler slot to the index in the per-stage HAL200
 * table. This is deliberately not the legacy global-heap 28+index handle. */
bool agx_gallium_g17p_sampler_handle(enum agx_chip chip,
                                    unsigned api_sampler,
                                    uint16_t *native_handle);

bool agx_gallium_g17p_custom_border_handle(enum agx_chip chip,
                                           unsigned api_sampler,
                                           uint16_t *native_handle);

bool agx_gallium_set_null_native_texture(
   enum agx_descriptor_generation generation,
   struct agx_gallium_texture_descriptor *out);

/* Takes the SEMANTIC PBE, not the packed AGX2 bytes.  The AGX2 packet is a
 * union -- stride aliases levels/layers, and the whole software addressing
 * sideband aliases the extended word -- so only the producer knows which arm
 * is live.  Unpacking the bytes back would read the wrong arm and either
 * refuse a legal descriptor or accept an illegal one.
 */
bool agx_gallium_pack_native_pbe(
   enum agx_descriptor_generation generation,
   struct agx_gallium_pbe_descriptor *out,
   const struct AGX_PBE *semantic);

/* G18-only mip-chain extension; retains all semantic sideband checks. */
bool agx_gallium_pack_g18_native_pbe(
   struct agx_gallium_pbe_descriptor *out, const struct AGX_PBE *semantic);

bool agx_gallium_set_null_native_pbe(
   enum agx_descriptor_generation generation,
   struct agx_gallium_pbe_descriptor *out);

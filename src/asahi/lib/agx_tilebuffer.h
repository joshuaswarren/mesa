/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include "drm-uapi/asahi_drm.h"
#include "agx_abi.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "util/format/u_formats.h"
#include "agx_chip.h"
#include "agx_pack.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum render targets per framebuffer. This is NOT architectural, but it
 * is the ~universal API limit so there's no point in allowing more.
 */
#define AGX_MAX_RENDER_TARGETS (8)

/* Forward declarations to keep the header lean */
struct nir_shader;
struct nir_def;
struct nir_builder;

struct agx_tilebuffer_layout {
   /* Logical format of each render target. Use agx_tilebuffer_physical_format
    * to get the physical format.
    */
   enum pipe_format logical_format[AGX_MAX_RENDER_TARGETS];

   /* Which render targets are spilled. */
   bool spilled[AGX_MAX_RENDER_TARGETS];

   /* Offset into the sample of each render target. If a render target is
    * spilled, its offset is UNDEFINED. Use agx_tilebuffer_offset_B to access.
    */
   uint8_t _offset_B[AGX_MAX_RENDER_TARGETS];

   /* Total bytes per sample, rounded up as needed. Spilled render targets do
    * not count against this.
    */
   uint8_t sample_size_B;

   /* Number of samples per pixel */
   uint8_t nr_samples;

   /* If layered rendering is used */
   bool layered : 1;

   /* Native imageblock state replaces the AGX2 USC shared record. */
   bool native_imageblock : 1;

   /* Most bytes a sample may hold before render targets spill, or zero for
    * the 64-byte AGX2 bound. The tile size bounds it separately. */
   uint8_t max_sample_size_B;

   /* Independent storage and raster-sample budgets. Zero retains the legacy
    * byte capacity or imposes no additional sample budget, respectively. */
   uint32_t max_bytes_per_tile;
   uint16_t max_samples_per_tile;

   /* Selected tile size in pixels. Either 32*32, 32*16, or 16*16. */
   uint16_t tile_size;

   /* Render targets allocated ahead of the others because the descending
    * alignment order would spill one of them; zero for that order. See
    * agx_build_tilebuffer_layout_resident(). */
   uint8_t resident_mask;

   /* Named so that layouts hashed or compared byte-wise have no
    * indeterminate padding. Always zero. */
   uint8_t reserved[3];

   /* USC word corresponding to this configuration of the tilebuffer. Zero
    * for native imageblock layouts and for tiles above its 64-KiB limit
    * (G18G): such layouts use the native Ioto imageblock path and must not
    * emit this word. */
   struct agx_usc_shared_packed usc;
};

static inline void
agx_tilebuffer_set_drm_cmd(struct drm_asahi_cmd_render *c,
                           const struct agx_tilebuffer_layout *layout)
{
   c->utile_width_px = layout->tile_size >= 32 * 16 ? 32 : 16;
   c->utile_height_px = layout->tile_size <= 32 * 16 ? 16 : 32;
}

/*
 * _offset_B is undefined for non-spilled render targets. This safe accessor
 * asserts that render targets are not spilled rather than returning garbage.
 */
static inline uint8_t
agx_tilebuffer_offset_B(const struct agx_tilebuffer_layout *layout, unsigned rt)
{
   assert(rt < AGX_MAX_RENDER_TARGETS);
   assert(!layout->spilled[rt] && "precondition");

   return layout->_offset_B[rt];
}

static inline bool
agx_tilebuffer_spills(const struct agx_tilebuffer_layout *layout)
{
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (layout->spilled[rt])
         return true;
   }

   return false;
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout(const enum pipe_format *formats, uint8_t nr_cbufs,
                            uint8_t nr_samples, bool layered);

/* As agx_build_tilebuffer_layout(), spilling render targets only once a
 * sample would exceed max_sample_size_B bytes (zero for the AGX2 bound). */
struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_bounded(const enum pipe_format *formats,
                                    uint8_t nr_cbufs, uint8_t nr_samples,
                                    bool layered, uint8_t max_sample_size_B);

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_with_limits(const enum pipe_format *formats,
                                      uint8_t nr_cbufs, uint8_t nr_samples,
                                      bool layered, uint8_t max_sample_size_B,
                                      uint32_t max_bytes_per_tile,
                                      uint16_t max_samples_per_tile);

/* agx_build_tilebuffer_layout_with_limits() with resident_mask as in
 * agx_build_tilebuffer_layout_resident(). */
struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_with_limits_resident(
   const enum pipe_format *formats, uint8_t nr_cbufs, uint8_t nr_samples,
   bool layered, uint8_t max_sample_size_B, uint32_t max_bytes_per_tile,
   uint16_t max_samples_per_tile, uint8_t resident_mask);

/* Attachment 0 alone in the tilebuffer, every other attachment spilled to
 * memory. */
struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_rt0_resident(const enum pipe_format *formats,
                                         uint8_t nr_cbufs, uint8_t nr_samples,
                                         uint32_t max_bytes_per_tile,
                                         uint16_t max_samples_per_tile);

/* Native imageblock strides are not limited by the AGX2 USC bitfield. */
struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_for_imageblock(
   const enum pipe_format *formats, uint8_t nr_cbufs, uint8_t nr_samples,
   bool layered, uint8_t max_sample_size_B, uint32_t max_bytes_per_tile,
   uint16_t max_samples_per_tile);

/* As agx_build_tilebuffer_layout_for_imageblock(), but when that layout
 * spills one of the render targets in resident_mask, those are allocated
 * first so that they stay in the tilebuffer and the others spill instead.
 * The applied mask is recorded in the layout; building again with it
 * reproduces the layout. */
struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_resident(
   const enum pipe_format *formats, uint8_t nr_cbufs, uint8_t nr_samples,
   bool layered, uint8_t max_sample_size_B, uint32_t max_bytes_per_tile,
   uint16_t max_samples_per_tile, uint8_t resident_mask);

/* Compact attachment slots for internal programs without moving tile data.
 * map[dense slot] identifies the original logical attachment. */
unsigned agx_tilebuffer_compact(const struct agx_tilebuffer_layout *src,
                                struct agx_tilebuffer_layout *dst,
                                uint8_t map[AGX_MAX_RENDER_TARGETS]);

void agx_tilebuffer_set_samples(struct agx_tilebuffer_layout *tib,
                                unsigned nr_samples);

bool agx_nir_lower_tilebuffer(struct nir_shader *shader,
                              struct agx_tilebuffer_layout *tib,
                              uint8_t *colormasks, unsigned *bindless_base,
                              struct nir_def *write_samples, bool *translucent,
                              bool preserve_zero_mask,
                              bool lower_g17p_formats);

/* As agx_nir_lower_tilebuffer, plus explicit software sRGB and R11G11B10
 * conversions of tile values and, with x_alpha_one, 1.0 stored into the X
 * channel of RGBX/BGRX targets (all used on G18G). ms_whole_sample: the
 * multisampled tile accesses move whole samples, taking F16 but not I16
 * (G18G). */
bool agx_nir_lower_tilebuffer_with_conversions(
   struct nir_shader *shader, struct agx_tilebuffer_layout *tib,
   uint8_t *colormasks, unsigned *bindless_base, struct nir_def *write_samples,
   bool *translucent, bool preserve_zero_mask, bool lower_srgb,
   bool lower_r11g11b10, bool ms_whole_sample, bool lower_g17p_formats,
   bool x_alpha_one);
/* Put spilled render target accesses back on immediate texture state slots
 * after descriptor crawling has run (G17G's bindful image accesses). */
bool agx_nir_rebind_spilled_rts(struct nir_shader *shader);

bool agx_nir_lower_to_per_sample(struct nir_shader *shader);
bool agx_nir_apply_api_mask_to_force_early(struct nir_shader *shader,
                                          enum agx_chip chip);

bool agx_nir_lower_monolithic_msaa(struct nir_shader *shader,
                                   uint8_t nr_samples);
bool agx_nir_lower_monolithic_msaa_m5(struct nir_shader *shader,
                                      uint8_t nr_samples);
bool agx_nir_lower_m5_zs_sample_loop(struct nir_shader *shader,
                                     unsigned nr_samples);

/* Export registers of one sample's colours in a G17P fragment unrolled per
 * sample: eight halves for each location up to the last one written. */
static inline unsigned
agx_g17p_sample_colour_stride(uint8_t loc_written)
{
   return 8 * util_last_bit(loc_written);
}

/* Export base of sample `sample`'s depth (stencil two halves later) in such
 * a fragment, for every sample after the first: four halves per sample after
 * every sample's colours. */
static inline unsigned
agx_g17p_sample_zs_base(unsigned nr_samples, uint8_t loc_written,
                        unsigned sample)
{
   return AGX_ABI_FOUT_COLOUR(0) +
          nr_samples * agx_g17p_sample_colour_stride(loc_written) +
          4 * (sample - 1);
}

/* Where each colour location's components sit in the export registers of one
 * sample of a G17P fragment unrolled per sample.
 *
 * The fragment ABI gives every location a four-component slot, which is what
 * a location that writes fewer components still costs once the slot is
 * repeated for each sample. Eight two-component 32-bit targets at four
 * samples need 276 halves that way, and G17P's register file is 192 -- the
 * draw is refused for want of registers the shader never writes. The packed
 * layout gives each location only the components it exports, which brings the
 * same fragment to 148 halves. Both layouts describe the same thing, so the
 * unroll and the epilog share this one description of it.
 */
struct agx_g17p_colour_layout {
   /* Halves from AGX_ABI_FOUT_COLOUR(0) to this location's first component */
   uint8_t offset[8];

   /* Components this location occupies, zero if it exports none */
   uint8_t comps[8];

   /* Halves one sample's colours occupy in total */
   uint16_t stride;
};

/* `colour_comps` holds the components each location exports, four bits per
 * location, as agx_fs_epilog_link_info records them. It is only consulted for
 * the packed layout; the ABI layout is fixed by the loc_written mask alone.
 */
static inline void
agx_g17p_colour_layout_init(struct agx_g17p_colour_layout *l,
                            uint8_t loc_written, uint8_t size_32,
                            uint32_t colour_comps, bool packed)
{
   unsigned offset = 0;

   for (unsigned loc = 0; loc < 8; ++loc) {
      const bool written = loc_written & BITFIELD_BIT(loc);

      if (packed) {
         /* Each component is preloaded and exported on its own, so a 32-bit
          * location has to start on an even half like any register pair. */
         const unsigned halves_per_comp =
            (size_32 & BITFIELD_BIT(loc)) ? 2 : 1;
         const unsigned comps =
            written ? util_last_bit((colour_comps >> (4 * loc)) & 0xf) : 0;

         offset = ALIGN_POT(offset, halves_per_comp);
         l->offset[loc] = offset;
         l->comps[loc] = comps;
         offset += comps * halves_per_comp;
      } else {
         l->offset[loc] = AGX_ABI_FOUT_COLOUR(loc) - AGX_ABI_FOUT_COLOUR(0);
         l->comps[loc] = 4;
      }
   }

   /* Every sample repeats the layout, and AGX_ABI_FOUT_COLOUR(0) is even, so
    * an even stride is what keeps each sample's pairs aligned too. */
   l->stride =
      packed ? ALIGN_POT(offset, 2) : agx_g17p_sample_colour_stride(loc_written);
}

/* Export base of location `loc`'s first component for sample `sample`. */
static inline unsigned
agx_g17p_sample_colour_base(const struct agx_g17p_colour_layout *l,
                            unsigned sample, unsigned loc)
{
   return AGX_ABI_FOUT_COLOUR(0) + (sample * l->stride) + l->offset[loc];
}

/* As agx_g17p_sample_zs_base, for a laid-out fragment. */
static inline unsigned
agx_g17p_sample_zs_base_layout(const struct agx_g17p_colour_layout *l,
                               unsigned nr_samples, unsigned sample)
{
   return AGX_ABI_FOUT_COLOUR(0) + (nr_samples * l->stride) +
          4 * (sample - 1);
}

/* Halves of export registers the whole unrolled fragment needs. Passing the
 * sample count as the sample index names the half one past the last sample's
 * stencil, which is exactly the count of halves used. */
static inline unsigned
agx_g17p_sample_exports_end(const struct agx_g17p_colour_layout *l,
                            unsigned nr_samples)
{
   return agx_g17p_sample_zs_base_layout(l, nr_samples, nr_samples);
}

void agx_nir_lower_sample_exports_to_end(struct nir_shader *shader);
bool agx_nir_needs_sample_count(const struct nir_shader *shader);
bool agx_nir_specialize_sample_count(struct nir_shader *shader, unsigned samples);
bool agx_nir_unroll_g17p_samples(struct nir_shader *shader,
                                 unsigned nr_samples, uint8_t loc_written);
bool agx_nir_unroll_g17p_samples_layout(
   struct nir_shader *shader, unsigned nr_samples,
   const struct agx_g17p_colour_layout *layout);

bool agx_nir_hoist_g17p_zs_writes(struct nir_shader *shader);

bool agx_nir_lower_sample_intrinsics(struct nir_shader *shader,
                                     bool ignore_sample_mask_without_msaa);

bool agx_nir_lower_sample_intrinsics_wide(struct nir_shader *shader,
                                          bool ignore_sample_mask_without_msaa);

uint32_t agx_tilebuffer_total_size(struct agx_tilebuffer_layout *tib);

enum pipe_format
agx_tilebuffer_physical_format(const struct agx_tilebuffer_layout *tib,
                               unsigned rt);

bool agx_tilebuffer_supports_mask(struct agx_tilebuffer_layout *tib,
                                  unsigned rt);

void agx_tilebuffer_pack_usc(struct agx_tilebuffer_layout *tib);

bool agx_g17p_rt0_tile_layout_supported(
   const struct agx_tilebuffer_layout *tib);

/* Logical active attachment mask for compact RGBA8 storage in partition zero. */
unsigned agx_g17p_rgba8_mrt_mask(const struct agx_tilebuffer_layout *tib);

/* Compact G18 four-byte slots: RGBA8 or a scalar R32 integer/float word. */
unsigned agx_g18g_word_mrt_mask(const struct agx_tilebuffer_layout *tib);

/* Resident RGBA32 or scalar R32I/U multisample attachments on G18. */
unsigned agx_g18g_raw32_mrt_mask(const struct agx_tilebuffer_layout *tib);
unsigned agx_g18g_ms_mrt_mask(const struct agx_tilebuffer_layout *tib);

/* Native-1130 general single-sample mixed-format G18 layouts (run 1139).
 * Apple packs the attachments in index order, each at the next offset aligned
 * to its tile lane width (holes take no space), in a 32x32 tile of an
 * eight-byte-aligned sample. agx_g18g_native_mixed_tile_bytes() returns that
 * tile's bytes when the format list takes this layout -- two to eight
 * attachments of witnessed formats, not owned by the RGBA8/R32 word or
 * RGBA32/RG32 paths, and a tile the kernel admits -- and zero otherwise.
 * agx_build_tilebuffer_layout_g18_mixed() builds it; the gallium layout and the
 * fragment epilog both use it, so both compute the same offsets.
 * kernel_native_samples: the kernel advertises
 * DRM_ASAHI_FEATURE_G18_NATIVE_TILE_SAMPLES, tested in
 * agx_device::params.features. */
unsigned
agx_g18g_native_mixed_tile_bytes(const enum pipe_format *formats,
                                 unsigned nr_cbufs, unsigned nr_samples,
                                 bool layered, bool kernel_native_samples);
struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_g18_mixed(const enum pipe_format *formats,
                                      uint8_t nr_cbufs);
/* The attachment mask of a layout built that way (zero otherwise). */
unsigned
agx_g18g_native_mixed_mask(const struct agx_tilebuffer_layout *tib);

/* The IMGWBLK f29 tile class Apple's end of tile uses for a witnessed format:
 * 0 8-bit integer lanes, 1 16-bit lanes, 2 32-bit lanes, 3 tile bytes in the
 * attachment's own encoding (8-bit normalized, RGB10A2, R11G11B10F). */
int agx_g18g_native_mixed_eot_class(enum pipe_format format);

enum agx_g17p_mrt_layout {
   AGX_G17P_MRT_NONE,
   AGX_G17P_MRT_WBOIT,
   AGX_G17P_MRT_RGBA8,
   AGX_G17P_MRT_R11G11B10,
   AGX_G17P_MRT_RGBA32,
   AGX_G17P_MRT_R32_QUAD,
   AGX_G17P_MRT_MIXED,
};

/* Classify the byte-grounded G17P multiple-render-target layouts. The
 * background/EOT frontends use this common result to select their matching
 * shader, clear-uniform, and PBE-table contracts. */
enum agx_g17p_mrt_layout
agx_g17p_classify_mrt(const struct agx_tilebuffer_layout *tib);

#ifdef __cplusplus
} /* extern C */
#endif

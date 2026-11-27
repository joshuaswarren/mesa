/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_tilebuffer.h"
#include <assert.h>
#include <string.h>
#include "util/bitset.h"
#include "util/format/u_format.h"
#include "util/u_debug.h"
#include "agx_bg_eot_key.h"
#include "layout.h"

/* Maximum number of bytes per tile on G13G. This may change in future versions
 * of the architecture.
 */
#define MAX_BYTES_PER_TILE (32768 - 1)

/* Maximum bytes per sample in the tilebuffer. Greater allocations require
 * spilling render targets to memory.
 */
#define MAX_BYTES_PER_SAMPLE (64)

/* USC Shared::sample_stride_in_8_bytes is a four-bit field (cmdbuf.xml), so
 * the record cannot name a sample wider than 15 * 8 bytes however large the
 * imageblock is. G17P's shaders do address 128 -- Apple's own eight-target
 * fragments store at tile offsets 0..112 -- but there is no witness for the
 * record at that size, and eight 16-byte targets pack 0x10 into a field that
 * holds 0xf. Keep the layout inside what the record can hold and spill the
 * remainder, which is the same mechanism every other over-large layout uses.
 */
#define MAX_ENCODABLE_BYTES_PER_SAMPLE (15 * 8)

/* Minimum tile size in pixels, architectural. */
#define MIN_TILE_SIZE_PX (16 * 16)

/* Select the largest tile size satisfying both independent budgets. */
static uint16_t
agx_select_tile_size(unsigned sample_size_B, unsigned nr_samples,
                     unsigned max_bytes_per_tile, unsigned max_samples_per_tile)
{
   const unsigned px_size_B = sample_size_B * nr_samples;
   const unsigned sizes[] = {32 * 32, 32 * 16, 16 * 16};
   for (unsigned i = 0; i < ARRAY_SIZE(sizes); ++i) {
      if (px_size_B * sizes[i] <= max_bytes_per_tile &&
          (!max_samples_per_tile ||
           nr_samples * sizes[i] <= max_samples_per_tile))
         return sizes[i];
   }

   UNREACHABLE("No tile fits the device allocation limits");
}

static inline unsigned
format_align_B(enum pipe_format format)
{
   /* For some reason util_format_get_blocksize(NONE) = 1 */
   enum pipe_format phys = ail_pixel_format[format].renderable;
   return (format != PIPE_FORMAT_NONE) ? util_format_get_blocksize(phys) : 0;
}

static unsigned
format_size_B(enum pipe_format format)
{
   unsigned size_B = format_align_B(format);
   enum pipe_format phys = ail_pixel_format[format].renderable;
   if (util_format_get_nr_components(phys) == 1) {
      size_B *= util_format_get_nr_components(format);
   }

   return size_B;
}

unsigned
agx_tilebuffer_format_size_B(enum pipe_format format)
{
   return format_size_B(format);
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout(const enum pipe_format *formats, uint8_t nr_cbufs,
                            uint8_t nr_samples, bool layered)
{
   return agx_build_tilebuffer_layout_bounded(formats, nr_cbufs, nr_samples,
                                              layered, 0);
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_bounded(const enum pipe_format *formats,
                                    uint8_t nr_cbufs, uint8_t nr_samples,
                                    bool layered, uint8_t max_sample_size_B)
{
   return agx_build_tilebuffer_layout_with_limits(
      formats, nr_cbufs, nr_samples, layered, max_sample_size_B, 0, 0);
}

/* Whether render target a is allocated before b: the targets in first_mask
 * ahead of the others, and within each group in descending order of
 * alignment, which leaves no padding inside a group. */
static bool
agx_tilebuffer_allocate_before(const enum pipe_format *formats,
                               uint8_t first_mask, unsigned a, unsigned b)
{
   const bool first_a = first_mask & BITFIELD_BIT(a);
   const bool first_b = first_mask & BITFIELD_BIT(b);
   if (first_a != first_b)
      return first_a;
   return format_align_B(formats[a]) > format_align_B(formats[b]);
}

/* Allocate each render target a tilebuffer offset in the order above, or
 * spill it once it no longer fits. Returns the bytes used per sample. */
static uint32_t
agx_tilebuffer_allocate(struct agx_tilebuffer_layout *tib,
                        const enum pipe_format *formats, uint8_t nr_cbufs,
                        uint8_t nr_samples, unsigned max_bytes_per_sample,
                        unsigned tile_capacity, uint8_t first_mask)
{
   uint32_t offset_B = 0;
   uint8_t order[] = {0, 1, 2, 3, 4, 5, 6, 7};

   /* Insertion sort: simple, stable, fast for small n, and free for n=1. */
   for (int i = 1; i < nr_cbufs; ++i) {
      for (int j = i; j > 0 && agx_tilebuffer_allocate_before(
                                  formats, first_mask, order[j], order[j - 1]);
           --j) {
         SWAP(order[j], order[j - 1]);
      }
   }

   for (unsigned i = 0; i < nr_cbufs; ++i) {
      unsigned rt = order[i];
      enum pipe_format format = formats[rt];
      tib->logical_format[rt] = format;
      tib->spilled[rt] = false;
      tib->_offset_B[rt] = 0;

      /* Padding only arises between the two groups. */
      const uint32_t aligned_B =
         align(offset_B, MAX2(format_align_B(format), 1));
      assert((first_mask || aligned_B == offset_B) &&
             "loop invariant ensured by the sort");
      offset_B = aligned_B;

      unsigned size_B = format_size_B(format);

      /* If allocating this render target would exceed any tilebuffer limits, we
       * need to spill it to memory. Otherwise, allocate it to the tilebuffer.
       */
      unsigned new_offset_B = offset_B + size_B;
      bool fits = (new_offset_B <= max_bytes_per_sample) &&
                  (ALIGN_POT(new_offset_B, 8) * MIN_TILE_SIZE_PX *
                   nr_samples) <= tile_capacity;
      if (fits) {
         tib->_offset_B[rt] = offset_B;
         offset_B = new_offset_B;
      } else {
         tib->spilled[rt] = true;
      }
   }

   return offset_B;
}

static struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_internal(const enum pipe_format *formats,
                                      uint8_t nr_cbufs, uint8_t nr_samples,
                                      bool layered, uint8_t max_sample_size_B,
                                      uint32_t max_bytes_per_tile,
                                      uint16_t max_samples_per_tile,
                                      bool native_imageblock,
                                      uint8_t resident_mask)
{
   struct agx_tilebuffer_layout tib = {
      .nr_samples = nr_samples,
      .layered = layered,
      .native_imageblock = native_imageblock,
      .max_sample_size_B = max_sample_size_B,
      .max_bytes_per_tile = max_bytes_per_tile,
      .max_samples_per_tile = max_samples_per_tile,
   };
   const unsigned max_bytes_per_sample =
      native_imageblock ? max_sample_size_B :
      MIN2(max_sample_size_B ? max_sample_size_B : MAX_BYTES_PER_SAMPLE,
           /* Larger Ioto allocations (G18G's 96-KiB multisample tiles) do
            * not use USC_SHARED's stride. */
           max_bytes_per_tile > 65536 ? 248 : MAX_ENCODABLE_BYTES_PER_SAMPLE);
   const unsigned tile_capacity =
      max_bytes_per_tile ? max_bytes_per_tile : MAX_BYTES_PER_TILE;

   uint32_t offset_B =
      agx_tilebuffer_allocate(&tib, formats, nr_cbufs, nr_samples,
                              max_bytes_per_sample, tile_capacity, 0);

   /* Keep the requested render targets in the tilebuffer if the plain order
    * would spill one of them (M5 compressed multisample attachments: their
    * in-pass memory reads do not observe the pass's own stores). */
   uint8_t spilled_mask = 0;
   for (unsigned rt = 0; rt < nr_cbufs; ++rt) {
      if (tib.spilled[rt])
         spilled_mask |= BITFIELD_BIT(rt);
   }
   resident_mask &= BITFIELD_MASK(nr_cbufs);
   if (resident_mask & spilled_mask) {
      offset_B = agx_tilebuffer_allocate(&tib, formats, nr_cbufs, nr_samples,
                                         max_bytes_per_sample, tile_capacity,
                                         resident_mask);
      tib.resident_mask = resident_mask;
   }

   assert(offset_B <= max_bytes_per_sample && "loop invariant");

   /* Multisampling needs a nonempty allocation.
    * XXX: Check this against hw
    */
   if (nr_samples > 1)
      offset_B = MAX2(offset_B, 1);

   /* M5 likewise reserves one eight-byte sample for a layered pass with no
    * colour attachment: native layered attachmentless and depth-only passes
    * carry imageblock sample size 8 (command +0x1fb0) at one and four samples
    * (native-window/item1-layered). */
   if (native_imageblock && layered)
      offset_B = MAX2(offset_B, 1);

   tib.sample_size_B = align(offset_B, 8);
   tib.tile_size = agx_select_tile_size(
      tib.sample_size_B, nr_samples,
      max_bytes_per_tile ? max_bytes_per_tile : MAX_BYTES_PER_TILE,
      max_samples_per_tile);

   agx_tilebuffer_pack_usc(&tib);
   return tib;
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_with_limits(
   const enum pipe_format *formats, uint8_t nr_cbufs, uint8_t nr_samples,
   bool layered, uint8_t max_sample_size_B, uint32_t max_bytes_per_tile,
   uint16_t max_samples_per_tile)
{
   return agx_build_tilebuffer_layout_with_limits_resident(
      formats, nr_cbufs, nr_samples, layered, max_sample_size_B,
      max_bytes_per_tile, max_samples_per_tile, 0);
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_with_limits_resident(
   const enum pipe_format *formats, uint8_t nr_cbufs, uint8_t nr_samples,
   bool layered, uint8_t max_sample_size_B, uint32_t max_bytes_per_tile,
   uint16_t max_samples_per_tile, uint8_t resident_mask)
{
   return agx_build_tilebuffer_layout_internal(
      formats, nr_cbufs, nr_samples, layered, max_sample_size_B,
      max_bytes_per_tile, max_samples_per_tile, false, resident_mask);
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_rt0_resident(const enum pipe_format *formats,
                                         uint8_t nr_cbufs, uint8_t nr_samples,
                                         uint32_t max_bytes_per_tile,
                                         uint16_t max_samples_per_tile)
{
   struct agx_tilebuffer_layout tib = agx_build_tilebuffer_layout_with_limits(
      formats, MIN2(nr_cbufs, 1), nr_samples, false, 0, max_bytes_per_tile,
      max_samples_per_tile);

   for (unsigned rt = 1; rt < nr_cbufs; ++rt) {
      tib.logical_format[rt] = formats[rt];
      tib.spilled[rt] = formats[rt] != PIPE_FORMAT_NONE;
   }

   return tib;
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_for_imageblock(
   const enum pipe_format *formats, uint8_t nr_cbufs, uint8_t nr_samples,
   bool layered, uint8_t max_sample_size_B, uint32_t max_bytes_per_tile,
   uint16_t max_samples_per_tile)
{
   return agx_build_tilebuffer_layout_resident(
      formats, nr_cbufs, nr_samples, layered, max_sample_size_B,
      max_bytes_per_tile, max_samples_per_tile, 0);
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_resident(
   const enum pipe_format *formats, uint8_t nr_cbufs, uint8_t nr_samples,
   bool layered, uint8_t max_sample_size_B, uint32_t max_bytes_per_tile,
   uint16_t max_samples_per_tile, uint8_t resident_mask)
{
   assert(max_sample_size_B && !(max_sample_size_B & 7));
   assert(max_bytes_per_tile);
   return agx_build_tilebuffer_layout_internal(
      formats, nr_cbufs, nr_samples, layered, max_sample_size_B,
      max_bytes_per_tile, max_samples_per_tile, true, resident_mask);
}

void
agx_tilebuffer_apply_first_mask(struct agx_tilebuffer_layout *tib,
                                uint8_t first_mask)
{
   assert(tib->resident_mask == 0 && "first_mask conflicts with resident_mask");

   enum pipe_format formats[AGX_MAX_RENDER_TARGETS];
   unsigned nr_cbufs = 0;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      formats[rt] = tib->logical_format[rt];
      if (formats[rt] != PIPE_FORMAT_NONE)
         nr_cbufs = rt + 1;
   }
   first_mask &= BITFIELD_MASK(nr_cbufs);

   const unsigned max_bytes_per_sample =
      tib->native_imageblock ? tib->max_sample_size_B :
      MIN2(tib->max_sample_size_B ? tib->max_sample_size_B : MAX_BYTES_PER_SAMPLE,
           tib->max_bytes_per_tile > 65536 ? 248 : MAX_ENCODABLE_BYTES_PER_SAMPLE);
   const unsigned tile_capacity =
      tib->max_bytes_per_tile ? tib->max_bytes_per_tile : MAX_BYTES_PER_TILE;

   uint32_t offset_B =
      agx_tilebuffer_allocate(tib, formats, nr_cbufs, tib->nr_samples,
                              max_bytes_per_sample, tile_capacity, first_mask);

   assert(offset_B <= max_bytes_per_sample && "loop invariant");

   if (tib->nr_samples > 1)
      offset_B = MAX2(offset_B, 1);

   if (tib->native_imageblock && tib->layered)
      offset_B = MAX2(offset_B, 1);

   tib->sample_size_B = align(offset_B, 8);
   tib->tile_size = agx_select_tile_size(
      tib->sample_size_B, tib->nr_samples, tile_capacity,
      tib->max_samples_per_tile);
   tib->first_mask = first_mask;

   agx_tilebuffer_pack_usc(tib);
}

bool
agx_tilebuffer_pack_first_if_fits(struct agx_tilebuffer_layout *tib,
                                  uint8_t first_mask,
                                  uint32_t max_first_end_B)
{
   if (tib->resident_mask != 0)
      return false;

   uint8_t active_first = 0;
   bool already_fits = true;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (!(first_mask & BITFIELD_BIT(rt)) ||
          tib->logical_format[rt] == PIPE_FORMAT_NONE)
         continue;
      active_first |= BITFIELD_BIT(rt);
      if (tib->spilled[rt] ||
          tib->_offset_B[rt] + format_size_B(tib->logical_format[rt]) >
             max_first_end_B)
         already_fits = false;
   }
   if (!active_first || already_fits)
      return false;

   struct agx_tilebuffer_layout cand = *tib;
   agx_tilebuffer_apply_first_mask(&cand, active_first);

   if (cand.tile_size < tib->tile_size)
      return false;

   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (!tib->spilled[rt] && cand.spilled[rt])
         return false;
      if (active_first & BITFIELD_BIT(rt)) {
         if (cand.spilled[rt] ||
             cand._offset_B[rt] + format_size_B(cand.logical_format[rt]) >
                max_first_end_B)
            return false;
      }
   }

   *tib = cand;
   return true;
}

/*
 * With attachmentless rendering in Vulkan, the sample count may not known until
 * draw-time. It's convenient to construct an agx_tilebuffer_layout anyway when
 * beginning rendering, updating the sample count later. This helper allows the
 * driver to set the sample count in a partial agx_tilebuffer_layout.
 *
 * When doing so, we need to rebuild entirely since e.g. tile size might change.
 */
void
agx_tilebuffer_set_samples(struct agx_tilebuffer_layout *tib,
                           unsigned nr_samples)
{
   assert(tib->nr_samples == 0 && "must not be initialized");

   *tib = (tib->native_imageblock ? agx_build_tilebuffer_layout_for_imageblock
                                  : agx_build_tilebuffer_layout_with_limits)(
      tib->logical_format, ARRAY_SIZE(tib->logical_format), nr_samples,
      tib->layered, tib->max_sample_size_B, tib->max_bytes_per_tile,
      tib->max_samples_per_tile);
}

enum pipe_format
agx_tilebuffer_physical_format(const struct agx_tilebuffer_layout *tib,
                               unsigned rt)
{
   return ail_pixel_format[tib->logical_format[rt]].renderable;
}

bool
agx_tilebuffer_supports_mask(struct agx_tilebuffer_layout *tib, unsigned rt)
{
   /* We don't bother support masking with spilled render targets. This might be
    * optimized in the future but spilling is so rare anyway it's not worth it.
    */
   if (tib->spilled[rt])
      return false;

   enum pipe_format fmt = agx_tilebuffer_physical_format(tib, rt);
   return ail_isa_format_supports_mask((enum ail_isa_format)fmt);
}

uint32_t
agx_tilebuffer_total_size(struct agx_tilebuffer_layout *tib)
{
   return tib->sample_size_B * tib->nr_samples * tib->tile_size;
}

void
agx_tilebuffer_pack_usc(struct agx_tilebuffer_layout *tib)
{
   /* USC_SHARED belongs to the legacy pipeline and can encode at most
    * 64 KiB. Native imageblock layouts, and G18's larger resident tiles, use
    * Ioto imageblock records built from the full layout below, not this
    * packed USC record. Leave it empty rather than truncating the allocation
    * or overflowing groups(256). */
   if (tib->native_imageblock || agx_tilebuffer_total_size(tib) > 65536) {
      memset(&tib->usc, 0, sizeof(tib->usc));
      return;
   }

   agx_pack(&tib->usc, USC_SHARED, cfg) {
      if (tib->nr_samples > 0) {
         cfg.uses_shared_memory = true;
         assert(tib->sample_size_B <= MAX_ENCODABLE_BYTES_PER_SAMPLE &&
                "the layout must stay inside the record's four-bit field");
         cfg.sample_stride_in_8_bytes = tib->sample_size_B / 8;
         cfg.sample_count = tib->nr_samples;
         cfg.bytes_per_threadgroup = agx_tilebuffer_total_size(tib);

         if (tib->tile_size == 32 * 32)
            cfg.layout = AGX_SHARED_LAYOUT_32X32;
         else if (tib->tile_size == 32 * 16)
            cfg.layout = AGX_SHARED_LAYOUT_32X16;
         else if (tib->tile_size == 16 * 16)
            cfg.layout = AGX_SHARED_LAYOUT_16X16;
         else
            UNREACHABLE("Invalid tile size");
      } else {
         cfg.layout = AGX_SHARED_LAYOUT_VERTEX_COMPUTE;
         cfg.bytes_per_threadgroup = 65536;
      }
   }
}

static unsigned
agx_word_mrt_mask(const struct agx_tilebuffer_layout *tib,
                  unsigned max_bytes_per_tile, bool raw32, bool layered, bool srgb)
{
   if (!tib || (tib->layered && !layered) || tib->nr_samples != 1 ||
       (tib->tile_size != 32 * 32 && tib->tile_size != 32 * 16))
      return 0;

   unsigned count = 0, mask = 0;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      const enum pipe_format format = tib->logical_format[rt];
      if (format == PIPE_FORMAT_NONE)
         continue;
      if ((format != PIPE_FORMAT_R8G8B8A8_UNORM &&
           format != PIPE_FORMAT_B8G8R8A8_UNORM &&
           format != PIPE_FORMAT_R8G8B8X8_UNORM &&
           format != PIPE_FORMAT_B8G8R8X8_UNORM &&
           !(srgb && !tib->layered &&
             (format == PIPE_FORMAT_R8G8B8A8_SRGB ||
              format == PIPE_FORMAT_B8G8R8A8_SRGB ||
              format == PIPE_FORMAT_R8G8B8X8_SRGB ||
              format == PIPE_FORMAT_B8G8R8X8_SRGB)) &&
           !(raw32 && !tib->layered && (format == PIPE_FORMAT_R32_UINT ||
                       format == PIPE_FORMAT_R32_SINT ||
                       format == PIPE_FORMAT_R32_FLOAT))) ||
          tib->spilled[rt] || tib->_offset_B[rt] != 4 * count)
         return 0;
      mask |= BITFIELD_BIT(rt);
      ++count;
   }
   return count && tib->sample_size_B == ALIGN_POT(4 * count, 8) &&
          tib->sample_size_B * tib->tile_size <= max_bytes_per_tile ? mask : 0;
}

unsigned
agx_g17p_rgba8_mrt_mask(const struct agx_tilebuffer_layout *tib)
{
   return agx_word_mrt_mask(tib, 32767, false, false, false);
}

unsigned
agx_g18g_word_mrt_mask(const struct agx_tilebuffer_layout *tib)
{
   /* Native1068's two-target sRGB clear/load/mixed renders use the same
    * four-byte packed slots and sample state. Keep logical sRGB conversion
    * in each fragment/background shader and linear byte stores in the EOT.
    * R32 occupies the same four-byte tile slot as RGBA8. Native426/427
    * ground scalar word tile operations and the EOT argument/store ABI.
    * Logical conversion remains in the per-attachment shader/descriptor.
    * Layered G18 layouts admit only the RGBA8 family: native array MRT EOT
    * programs cover two through eight targets, with the same compact offsets. */
   return tib && tib->tile_size == 32 * 32
             ? agx_word_mrt_mask(tib, 32768, true, true, true) : 0;
}

static bool
agx_g18g_rg32_format(enum pipe_format f)
{
   return f == PIPE_FORMAT_R32G32_FLOAT || f == PIPE_FORMAT_R32G32_SINT ||
          f == PIPE_FORMAT_R32G32_UINT;
}

/* Native1094 rg32 (Apple M6, verified capture): two RG32 Uint/Sint/Float
 * targets in slots 0 and 1, homogeneous or Float+Sint, sit at compact
 * eight-byte offsets 0/8 of a sixteen-byte single-sample sample. Their
 * fragment stores are lstcol W32 mask 3 (f10=2, f26=3, f57=0/8) with the
 * RGBA32 pixel classes, the load background stores the same form, and the
 * end-of-tile text is byte-identical to the two-target R32-word program
 * (native427) already emitted for two RGBA32 targets. The fragment, load
 * background and end-of-tile bytes do not depend on the RG32 type. The
 * sparse {0,2} capture used Apple's generic clear programs only, so it is
 * not admitted. */
static unsigned
agx_g18g_rg32_pair_mask(const struct agx_tilebuffer_layout *tib)
{
   unsigned mask = 0, count = 0;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      const enum pipe_format f = tib->logical_format[rt];
      if (f == PIPE_FORMAT_NONE)
         continue;
      if (!agx_g18g_rg32_format(f) || tib->_offset_B[rt] != 8 * count)
         return 0;
      mask |= BITFIELD_BIT(rt);
      count++;
   }
   return count == 2 && mask == 0x3 && tib->sample_size_B == 16 ? mask : 0;
}

unsigned
agx_g18g_raw32_mrt_mask(const struct agx_tilebuffer_layout *tib)
{
   if (!tib || tib->nr_samples != 1 || tib->tile_size != 32 * 32 ||
       tib->layered || agx_tilebuffer_spills(tib))
      return 0;

   const unsigned rg32 = agx_g18g_rg32_pair_mask(tib);
   if (rg32)
      return rg32;

   /* Native855 four-target clear/load/mixed/sparse renders keep the same
    * compact sixteen-byte slots, fragment stores and BG/EOT programs for
    * Uint, Sint and Float: the type lives only in each target's descriptor.
    * Every target must still be a resident RGBA32 slot in compact order. */
   unsigned mask = 0, count = 0;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      enum pipe_format f = tib->logical_format[rt];
      if (f == PIPE_FORMAT_NONE)
         continue;
      if ((f != PIPE_FORMAT_R32G32B32A32_FLOAT &&
           f != PIPE_FORMAT_R32G32B32A32_SINT &&
           f != PIPE_FORMAT_R32G32B32A32_UINT) ||
          tib->_offset_B[rt] != 16 * count)
         return 0;
      mask |= BITFIELD_BIT(rt);
      count++;
   }
   return count >= 2 && count <= 8 && tib->sample_size_B == 16 * count
             ? mask : 0;
}

/* The kernel admits a single-sample G18 render whose 32x32 utile holds at most
 * 32 KiB, or a 48..128-byte sample in steps of 16 (module 1120,
 * g17_uapi.rs translate_g18_render_command `wide_single`); a larger tile fails
 * the submission (TilebufferSize). Module 1134 also admits exactly the native
 * 40-, 56- and 72-byte samples of capture 1121 and advertises that with
 * DRM_ASAHI_FEATURE_G18_NATIVE_TILE_SAMPLES (kernel_native_samples). Never emit
 * a tile the kernel would reject. */
static bool
agx_g18g_kernel_admits_tile(unsigned sample_B, bool kernel_native_samples)
{
   return sample_B * 32 * 32 <= 32768 ||
          (sample_B >= 48 && sample_B <= 128 && !(sample_B & 15)) ||
          (kernel_native_samples &&
           (sample_B == 40 || sample_B == 56 || sample_B == 72));
}

/* Native1130 (Apple M6, verified capture 1130; run 1139): the formats of the
 * CTS draw-buffers random batches, all 39 workloads of capture 1130 and all 60
 * mixed renders of capture 1121 (the same 32 formats in both). For each, the
 * fragment, clear and reload background stores and the end-of-tile IMGWBLK use
 * the lane width and lane count of Mesa's physical tile format
 * (ail_pixel_format[].renderable) and nothing else of the format:
 *
 *  - 8-bit lanes (lstcol f10 = 0): R8/RG8/RGBA8/RGBX8 Unorm, R8/RG8/RGBA8
 *    Uint/Sint;
 *  - 16-bit lanes (f10 = 1): R16/RG16/RGBA16 Uint/Sint, RGBA16 Float,
 *    RGB10A2 Uint (four lanes), B5G6R5 (three lanes), B5G5R5A1 and B4G4R4A4
 *    (four lanes each);
 *  - 32-bit lanes (f10 = 2): R32/RG32/RGBA32 Uint/Sint/Float, and one packed
 *    word for RGB10A2 Unorm and R11G11B10 Float.
 *
 * Any other format keeps the existing refusal until a capture shows it. */
static bool
agx_g18g_native_mixed_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8_SINT:
   case PIPE_FORMAT_R8G8_SINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   case PIPE_FORMAT_R16_UINT:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16_SINT:
   case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R16G16B16A16_SINT:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_B5G6R5_UNORM:
   case PIPE_FORMAT_B5G5R5A1_UNORM:
   case PIPE_FORMAT_B4G4R4A4_UNORM:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return true;
   default:
      return false;
   }
}

int
agx_g18g_native_mixed_eot_class(enum pipe_format format)
{
   if (!agx_g18g_native_mixed_format(format))
      return -1;

   /* Native1130: every end-of-tile IMGWBLK of the 39 workloads (and 1121's 60)
    * carries f29 by this class and f30 = 1. */
   switch ((enum ail_isa_format)ail_pixel_format[format].renderable) {
   case AIL_ISA_FORMAT_I8:
      return 0;
   case AIL_ISA_FORMAT_I16:
   case AIL_ISA_FORMAT_F16:
      return 1;
   case AIL_ISA_FORMAT_I32:
      return 2;
   case AIL_ISA_FORMAT_U8NORM:
   case AIL_ISA_FORMAT_RGB10A2:
   case AIL_ISA_FORMAT_RG11B10F:
      return 3;
   default:
      return -1;
   }
}

/* Index order, each attachment at the next offset aligned to its lane width.
 * Returns the end of the last attachment (zero if none). */
static unsigned
agx_g18g_index_order_offsets(const enum pipe_format *formats,
                             unsigned nr_cbufs,
                             uint8_t offset_B[AGX_MAX_RENDER_TARGETS],
                             unsigned *active)
{
   unsigned end = 0;
   *active = 0;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      const enum pipe_format format =
         rt < nr_cbufs ? formats[rt] : PIPE_FORMAT_NONE;
      offset_B[rt] = 0;
      if (format == PIPE_FORMAT_NONE)
         continue;
      const unsigned align = MAX2(format_align_B(format), 1);
      end = ALIGN_POT(end, align);
      offset_B[rt] = end;
      end += format_size_B(format);
      (*active)++;
   }
   return end;
}

struct agx_tilebuffer_layout
agx_build_tilebuffer_layout_g18_mixed(const enum pipe_format *formats,
                                      uint8_t nr_cbufs)
{
   uint8_t offset_B[AGX_MAX_RENDER_TARGETS];
   unsigned active;
   const unsigned end =
      agx_g18g_index_order_offsets(formats, nr_cbufs, offset_B, &active);
   const unsigned sample_B = ALIGN_POT(end, 8);

   struct agx_tilebuffer_layout tib = {
      .nr_samples = 1,
      .layered = false,
      .sample_size_B = sample_B,
      .max_sample_size_B = sample_B,
      .max_bytes_per_tile = MAX2(sample_B * 32 * 32, 32768),
      .max_samples_per_tile = 1024,
      /* Native1130: specLM f19 = f20 = 5 (32x32), f21 = 0, f22 = the sample
       * size, f25 = 1 in the fragment, background and end-of-tile setup of all
       * 39 workloads, whatever the sample size (16..56; 1121 also 72). */
      .tile_size = 32 * 32,
   };
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      tib.logical_format[rt] = rt < nr_cbufs ? formats[rt] : PIPE_FORMAT_NONE;
      tib._offset_B[rt] = offset_B[rt];
   }
   agx_tilebuffer_pack_usc(&tib);
   return tib;
}

/* The layout and allocation an a42c45c7-era G18 single-sample render builds
 * for these formats, to keep the RGBA8/R32 word and RGBA32/RG32 paths in sole
 * charge of what they already admit. */
static bool
agx_g18g_owned_by_existing_path(const enum pipe_format *formats,
                                unsigned nr_cbufs)
{
   /* agx_gallium_tilebuffer_layout's raw32 budget: two to eight resident
    * RGBA32 attachments of any type. */
   unsigned active = 0;
   bool raw32 = true;
   for (unsigned rt = 0; rt < nr_cbufs && rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      const enum pipe_format f = formats[rt];
      if (f == PIPE_FORMAT_NONE)
         continue;
      raw32 &= f == PIPE_FORMAT_R32G32B32A32_FLOAT ||
               f == PIPE_FORMAT_R32G32B32A32_SINT ||
               f == PIPE_FORMAT_R32G32B32A32_UINT;
      active++;
   }
   if (raw32 && active >= 2 && active <= AGX_MAX_RENDER_TARGETS)
      return true;

   const struct agx_tilebuffer_layout sorted =
      agx_build_tilebuffer_layout_with_limits(formats, nr_cbufs, 1, false, 0,
                                              32768, 1024);
   return agx_g18g_word_mrt_mask(&sorted) || agx_g18g_raw32_mrt_mask(&sorted);
}

unsigned
agx_g18g_native_mixed_tile_bytes(const enum pipe_format *formats,
                                 unsigned nr_cbufs, unsigned nr_samples,
                                 bool layered, bool kernel_native_samples)
{
   if (nr_samples != 1 || layered)
      return 0;

   for (unsigned rt = 0; rt < nr_cbufs && rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (formats[rt] != PIPE_FORMAT_NONE &&
          !agx_g18g_native_mixed_format(formats[rt]))
         return 0;
   }

   uint8_t offset_B[AGX_MAX_RENDER_TARGETS];
   unsigned active;
   const unsigned end =
      agx_g18g_index_order_offsets(formats, nr_cbufs, offset_B, &active);
   const unsigned sample_B = ALIGN_POT(end, 8);
   if (active < 2 || active > AGX_MAX_RENDER_TARGETS || sample_B > 128 ||
       !agx_g18g_kernel_admits_tile(sample_B, kernel_native_samples) ||
       agx_g18g_owned_by_existing_path(formats, nr_cbufs))
      return 0;

   return sample_B * 32 * 32;
}

static unsigned
agx_g18g_native_mixed_layout_mask(const struct agx_tilebuffer_layout *tib)
{
   if (!tib || tib->nr_samples != 1 || tib->tile_size != 32 * 32 ||
       tib->layered || agx_tilebuffer_spills(tib))
      return 0;

   unsigned mask = 0;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      const enum pipe_format format = tib->logical_format[rt];
      if (format == PIPE_FORMAT_NONE)
         continue;
      if (!agx_g18g_native_mixed_format(format))
         return 0;
      mask |= BITFIELD_BIT(rt);
   }

   uint8_t offset_B[AGX_MAX_RENDER_TARGETS];
   unsigned active;
   const unsigned end = agx_g18g_index_order_offsets(
      tib->logical_format, AGX_MAX_RENDER_TARGETS, offset_B, &active);
   if (active < 2 || tib->sample_size_B != ALIGN_POT(end, 8))
      return 0;

   u_foreach_bit(rt, mask) {
      if (tib->_offset_B[rt] != offset_B[rt])
         return 0;
   }
   return mask;
}

unsigned
agx_g18g_native_mixed_mask(const struct agx_tilebuffer_layout *tib)
{
   const unsigned mask = agx_g18g_native_mixed_layout_mask(tib);
   if (!mask || agx_g18g_owned_by_existing_path(tib->logical_format,
                                                AGX_MAX_RENDER_TARGETS))
      return 0;
   return mask;
}

unsigned
agx_g18g_native_mixed_epilog_mask(const struct agx_tilebuffer_layout *tib)
{
   return agx_g18g_native_mixed_layout_mask(tib);
}

unsigned
agx_g18g_ms_mrt_mask(const struct agx_tilebuffer_layout *tib)
{
   /* Native G18 retains three RGBA32 attachments in a 96-KiB tile.
    * Native513 also grounds scalar R32Sint/R32Uint MS partitions. Keep
    * each layout homogeneous: every target occupies either sixteen or
    * four consecutive bytes per sample, with eight-byte sample padding. */
   if (!tib || (tib->nr_samples != 2 && tib->nr_samples != 4) ||
       tib->tile_size != (tib->nr_samples == 2 ? 32 * 32 : 32 * 16) ||
       tib->layered || agx_tilebuffer_spills(tib))
      return 0;

   unsigned mask = 0, count = 0, stride = 0;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      const enum pipe_format format = tib->logical_format[rt];
      if (format == PIPE_FORMAT_NONE)
         continue;
      const unsigned bytes =
         (format == PIPE_FORMAT_R32_SINT || format == PIPE_FORMAT_R32_UINT)
            ? 4
            : (format == PIPE_FORMAT_R32G32B32A32_FLOAT ||
               format == PIPE_FORMAT_R32G32B32A32_SINT ||
               format == PIPE_FORMAT_R32G32B32A32_UINT) ? 16 : 0;
      if (!bytes || (stride && stride != bytes) ||
          tib->_offset_B[rt] != bytes * count)
         return 0;
      stride = bytes;
      mask |= BITFIELD_BIT(rt);
      count++;
   }
   return count >= 2 && count <= 3 &&
          tib->sample_size_B == ALIGN_POT(stride * count, 8)
             ? mask : 0;
}

unsigned
agx_tilebuffer_compact(const struct agx_tilebuffer_layout *src,
                       struct agx_tilebuffer_layout *dst,
                       uint8_t map[AGX_MAX_RENDER_TARGETS])
{
   if (!src || !dst)
      return 0;

   /* Fast path for already-compact layouts (e.g. RT0 only or RT0..k-1
    * contiguous): avoid copying the struct twice and zeroing arrays. */
   unsigned count = 0;
   bool compact = true;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (src->logical_format[rt] != PIPE_FORMAT_NONE) {
         if (rt != count) {
            compact = false;
            break;
         }
         if (map)
            map[count] = rt;
         ++count;
      }
   }
   if (compact) {
      if (dst != src)
         *dst = *src;
      return count;
   }

   const struct agx_tilebuffer_layout original = *src;
   *dst = original;
   memset(dst->logical_format, 0, sizeof(dst->logical_format));
   memset(dst->spilled, 0, sizeof(dst->spilled));
   memset(dst->_offset_B, 0, sizeof(dst->_offset_B));
   count = 0;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (original.logical_format[rt] == PIPE_FORMAT_NONE)
         continue;
      if (map)
         map[count] = rt;
      dst->logical_format[count] = original.logical_format[rt];
      dst->spilled[count] = original.spilled[rt];
      dst->_offset_B[count++] = original._offset_B[rt];
   }
   return count;
}

bool
agx_g17p_rt0_tile_layout_supported(const struct agx_tilebuffer_layout *tib)
{
   if (!tib)
      return false;

   /* Compaction only carries the spill flag of an occupied target, so read
    * the flags before it: a spilled slot with no format is inconsistent
    * metadata and must not be silently dropped. */
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (tib->spilled[rt])
         return false;
   }

   struct agx_tilebuffer_layout compact;
   const unsigned count = agx_tilebuffer_compact(tib, &compact, NULL);
   tib = &compact;
   /* No attachment supplies a sample count. Draw-time state completes this
    * layout; the empty BG/EOT programs use the single-sample form. */
   if (count == 0 && compact.nr_samples == 0)
      compact.nr_samples = 1;
   if ((tib->nr_samples != 1 && tib->nr_samples != 2 &&
        tib->nr_samples != 4) ||
       tib->layered || tib->spilled[0])
      return false;

   /* Four 8-byte samples fill exactly 32 KiB at 32x32, but the hardware
    * limit is strictly smaller. Keep the generic allocator's valid 32x16
    * layout; BG/EOT use pixel coordinates and the render UAPI publishes the
    * utile geometry to firmware. */
   const enum agx_g17p_mrt_layout mrt = agx_g17p_classify_mrt(tib);
   const bool raw32 = mrt == AGX_G17P_MRT_RGBA32 ||
                     mrt == AGX_G17P_MRT_R32_QUAD;
   /* A multisample tile of a 16-byte sample fits only 16x16. */
   if (tib->tile_size != 32 * 32 && tib->tile_size != 32 * 16 &&
       !((mrt == AGX_G17P_MRT_RGBA32 || mrt == AGX_G17P_MRT_MIXED ||
          tib->nr_samples > 1) &&
         tib->tile_size == 16 * 16))
      return false;

   /* Multisample tiles repeat one sample's allocation, so what matters is the
    * physical tilebuffer format rather than the API format in front of it.
    * A hardware sweep at four samples, reading every sample back
    * individually before resolving, covers each physical family below:
    * rgba8, rgb565, rgba4, rgb5_a1, rgb10_a2, r8, rg8, r16f and rgba16f all
    * reach every sample and resolve to the drawn colour. Naming the physical
    * format also admits the BGRA and RGBX spellings of the same tiles.
    */
   if (tib->nr_samples > 1 && tib->logical_format[0] != PIPE_FORMAT_NONE) {
      switch (agx_tilebuffer_physical_format(tib, 0)) {
      case PIPE_FORMAT_R8_UNORM:
      case PIPE_FORMAT_R8G8_UNORM:
      case PIPE_FORMAT_R8G8B8A8_UNORM:
      case PIPE_FORMAT_R16_UNORM:
      case PIPE_FORMAT_R16_FLOAT:
      case PIPE_FORMAT_R10G10B10A2_UNORM:
      /* The signed normalized families, which GL 3.3 reaches where GL 3.0 did
       * not: KHR-GL33.texture_size_promotion renders to them. The same sweep
       * covers them -- r8_snorm, rg8_snorm, rgba8_snorm, r16_snorm, rg16_snorm
       * and rgba16_snorm all reach every sample and resolve. */
      case PIPE_FORMAT_R8_SNORM:
      case PIPE_FORMAT_R16_SNORM:
      /* The integer families the same case renders to. The sweep draws these
       * through an integer fragment output and reads them back with an
       * integer sampler: r8i and rgba8i share the eight-bit tile, rgb10a2ui
       * the sixteen-bit one, and all three reach every sample. */
      case PIPE_FORMAT_R8_UINT:
      case PIPE_FORMAT_R16_UINT:
      /* Vulkan's qualified raw-copy/resolve path preserves individual
       * R32 words, including packed RGB9E5, until the separate resolve. */
      case PIPE_FORMAT_R32_UINT:
      case PIPE_FORMAT_R9G9B9E5_FLOAT:
      /* The remaining two colour formats the conformance blit walks. Their
       * tiles are the same eight bytes per sample as the families above. */
      case PIPE_FORMAT_R8G8B8A8_SRGB:
      case PIPE_FORMAT_R11G11B10_FLOAT:
         break;
      default:
         return false;
      }
   }

   if (agx_g17p_rgba8_mrt_mask(tib))
      return true;

   for (unsigned rt = mrt == AGX_G17P_MRT_MIXED ? AGX_MAX_RENDER_TARGETS
                      : raw32 || mrt == AGX_G17P_MRT_RGBA8 ? 4
                      : mrt != AGX_G17P_MRT_NONE ? 2 : 1;
        rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (tib->logical_format[rt] != PIPE_FORMAT_NONE)
         return false;
   }

   /* A multisample layout reserves one empty byte-aligned sample even with no
    * colour attachment. */
   if (tib->logical_format[0] == PIPE_FORMAT_NONE)
      return tib->sample_size_B == (tib->nr_samples > 1 ? 8 : 0);

   const enum pipe_format format = tib->logical_format[0];
   const bool scalar_r32f = format == PIPE_FORMAT_R32_FLOAT;
   const bool scalar_r32i = format == PIPE_FORMAT_R32_SINT;
   const bool pair_r32f = format == PIPE_FORMAT_R32G32_FLOAT;
   const bool pair_r32i = format == PIPE_FORMAT_R32G32_SINT;
   const bool pair_r32u = format == PIPE_FORMAT_R32G32_UINT;
   const bool scalar_r8u = format == PIPE_FORMAT_R8_UINT;
   const bool scalar_r8i = format == PIPE_FORMAT_R8_SINT;
   const bool rgba16u_or_x = format == PIPE_FORMAT_R16G16B16A16_UINT ||
                             format == PIPE_FORMAT_R16G16B16X16_UINT;
   const bool rgba16i_or_x = format == PIPE_FORMAT_R16G16B16A16_SINT ||
                             format == PIPE_FORMAT_R16G16B16X16_SINT;
   /* API RGB32 surfaces use the four-lane RGBX physical family. The unused
    * lane has the same tile footprint and EOT protocol as RGBA32. */
   const bool rgbx32u = tib->sample_size_B == 16 &&
      (tib->logical_format[0] == PIPE_FORMAT_R32G32B32A32_UINT ||
       tib->logical_format[0] == PIPE_FORMAT_R32G32B32X32_UINT) &&
      agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R32_UINT;
   const bool rgbx32i = tib->sample_size_B == 16 &&
      (tib->logical_format[0] == PIPE_FORMAT_R32G32B32A32_SINT ||
       tib->logical_format[0] == PIPE_FORMAT_R32G32B32X32_SINT) &&
      agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R32_UINT;
   const bool rgbx32f = tib->sample_size_B == 16 &&
      (tib->logical_format[0] == PIPE_FORMAT_R32G32B32A32_FLOAT ||
       tib->logical_format[0] == PIPE_FORMAT_R32G32B32X32_FLOAT) &&
      agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R32_UINT;
   /* At any sample count: the sweep draws this one through an integer
    * fragment output and reads every sample back with an integer sampler, and
    * all four are the drawn value. KHR-GL33.texture_size_promotion renders to
    * it multisampled. */
   const bool rgb10a2u = tib->sample_size_B == 8 &&
      (tib->logical_format[0] == PIPE_FORMAT_R10G10B10A2_UINT ||
       tib->logical_format[0] == PIPE_FORMAT_B10G10R10A2_UINT) &&
      agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R16_UINT;
   /* The packed float tile is eight bytes per sample at any sample count, and
    * KHR-GL30.framebuffer_blit.multisampled_to_singlesampled_blit_color_config
    * exercises it at four samples through GL_R11F_G11F_B10F. Whether such a
    * tile can be resolved by averaging the raw word is a separate question
    * from whether the layout is admissible; restricting admission to one
    * sample refuses that conformance case outright. */
   const bool r11g11b10f = tib->sample_size_B == 8 &&
      tib->logical_format[0] == PIPE_FORMAT_R11G11B10_FLOAT &&
      agx_tilebuffer_physical_format(tib, 0) ==
         PIPE_FORMAT_R11G11B10_FLOAT;
   /* RGB9E5 keeps multisample words too; its resolve runs after the pass
    * (hk_agx3_g17p_direct_color_resolve_supported). */
   const bool r9g9b9e5f = tib->sample_size_B == 8 &&
      tib->logical_format[0] == PIPE_FORMAT_R9G9B9E5_FLOAT &&
      agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R9G9B9E5_FLOAT;

   return mrt != AGX_G17P_MRT_NONE || rgbx32u || rgbx32i || rgbx32f ||
          rgb10a2u || r11g11b10f || r9g9b9e5f ||
          (tib->sample_size_B == 8 &&
           (agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R8_UNORM ||
            agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R16_FLOAT ||
            agx_tilebuffer_physical_format(tib, 0) ==
               PIPE_FORMAT_R10G10B10A2_UNORM ||
            /* Every spelling of the signed normalized tiles at once. Naming
             * the physical format admits the BGRA and RGBX forms the logical
             * list had to enumerate and still missed: GL 3.3's
             * texture_size_promotion renders to r8g8b8x8_snorm. */
            agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R8_SNORM ||
            agx_tilebuffer_physical_format(tib, 0) == PIPE_FORMAT_R16_SNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16_UNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R32_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8X8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8_SRGB ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8_SRGB ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_SRGB ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8A8_SRGB ||
            /* GLES3's GL_SRGB8 colour buffer: the same 8-byte sRGB tile
             * with the alpha channel ignored. */
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8X8_SRGB ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8X8_SRGB ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8X8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8A8_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8X8_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8X8_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_B8G8R8A8_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16_UNORM ||
            tib->logical_format[0] == PIPE_FORMAT_R16_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16_FLOAT ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16_FLOAT ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_FLOAT ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16_UINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16_SINT ||
            tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_UNORM ||
            scalar_r32f || scalar_r32i || pair_r32f || pair_r32i ||
            pair_r32u || scalar_r8u || scalar_r8i || rgba16u_or_x ||
            rgba16i_or_x));
}

/* Other layouts of two to eight targets share Apple's multi-target EOT
 * builtin, whose targets differ only in a per-format IMGWBLK selector, and a
 * background program without pixel transactions. Each target keeps its own
 * tile offset, so admit any unspilled layout whose formats all have a grounded
 * selector.
 *
 * Layered layouts are admitted on the same terms. The earlier `!tib->layered`
 * clause was the scope of what had been captured -- only the 2D builtin family
 * was on hand -- and not a hardware limit: Apple's factory emits the same
 * program for a 2D-ARRAY attachment, differing only in that it also publishes
 * the layer wire, and its per-target selector is still the only
 * format-dependent byte. Those texts are now recorded for two through eight
 * targets in agx3_g17p_array_eot.h, which is the full range this function
 * admits. */
static enum agx_g17p_mrt_layout
agx_g17p_classify_mixed_mrt(const struct agx_tilebuffer_layout *tib,
                            unsigned active_count)
{
   bool mixed = active_count >= 2 &&
                active_count <= AGX_MAX_RENDER_TARGETS;
   BITSET_DECLARE(occupied_B, 256) = {0};
   for (unsigned rt = 0; rt < active_count; ++rt) {
      const unsigned offset_B = tib->_offset_B[rt];
      const unsigned size_B = format_size_B(tib->logical_format[rt]);
      mixed &= !tib->spilled[rt] &&
               agx_g17p_mrt_eot_selector(tib->logical_format[rt]) != 0 &&
               size_B && offset_B + size_B <= tib->sample_size_B;
      if (!mixed)
         break;
      mixed &= !BITSET_TEST_RANGE(occupied_B, offset_B, offset_B + size_B - 1);
      BITSET_SET_RANGE(occupied_B, offset_B, offset_B + size_B - 1);
   }

   return mixed ? AGX_G17P_MRT_MIXED : AGX_G17P_MRT_NONE;
}

enum agx_g17p_mrt_layout
agx_g17p_classify_mrt(const struct agx_tilebuffer_layout *tib)
{
   if (!tib)
      return AGX_G17P_MRT_NONE;
   struct agx_tilebuffer_layout compact;
   const unsigned active_count = agx_tilebuffer_compact(tib, &compact, NULL);
   tib = &compact;

   /* Multisample multi-target layouts take the general mixed-format
    * programs, whose end-of-tile builtin has a multisample form. */
   if (tib->nr_samples == 2 || tib->nr_samples == 4 || tib->nr_samples == 8)
      return agx_g17p_classify_mixed_mrt(tib, active_count);
   if (tib->nr_samples != 1)
      return AGX_G17P_MRT_NONE;

   if (tib->spilled[0] || tib->spilled[1] || tib->_offset_B[0] != 0)
      return agx_g17p_classify_mixed_mrt(tib, active_count);

   /* Native G17P constructors cover two through four identical RGBA32
    * targets in one imageblock. Keep the existing four-target R32 path. */
   const enum pipe_format format = tib->logical_format[0];
   const bool rgba32 = format == PIPE_FORMAT_R32G32B32A32_FLOAT ||
                       format == PIPE_FORMAT_R32G32B32A32_UINT ||
                       format == PIPE_FORMAT_R32G32B32A32_SINT;
   if (rgba32 || format == PIPE_FORMAT_R32_FLOAT) {
      const unsigned stride = rgba32 ? 16 : 4;
      unsigned count = 0;
      while (count < AGX_MAX_RENDER_TARGETS &&
             tib->logical_format[count] == format)
         count++;
      bool supported = !tib->layered && count >= 2 && count <= 4 &&
                       (rgba32 || count == 4) &&
                       tib->sample_size_B == count * stride;
      for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
         supported &= rt < count ? tib->logical_format[rt] == format &&
                             !tib->spilled[rt] && tib->_offset_B[rt] == rt * stride
                       : tib->logical_format[rt] == PIPE_FORMAT_NONE;
      }
      if (supported)
         return rgba32 ? AGX_G17P_MRT_RGBA32 : AGX_G17P_MRT_R32_QUAD;
   }

   bool rgba8 = !tib->layered && active_count >= 2 && active_count <= 4 &&
                tib->sample_size_B == ALIGN_POT(4 * active_count, 8);
   for (unsigned rt = 0; rt < active_count; ++rt) {
      const enum pipe_format f = tib->logical_format[rt];
      rgba8 &= (f == PIPE_FORMAT_R8G8B8A8_UNORM ||
                f == PIPE_FORMAT_B8G8R8A8_UNORM ||
                f == PIPE_FORMAT_R8G8B8X8_UNORM ||
                f == PIPE_FORMAT_B8G8R8X8_UNORM) &&
               !tib->spilled[rt] && tib->_offset_B[rt] == 4 * rt;
   }
   if (rgba8)
      return AGX_G17P_MRT_RGBA8;

   if (active_count > 2)
      return agx_g17p_classify_mixed_mrt(tib, active_count);

   if (tib->logical_format[0] == PIPE_FORMAT_R16G16B16A16_FLOAT &&
       tib->logical_format[1] == PIPE_FORMAT_R8_UNORM &&
       tib->_offset_B[1] == 8 && tib->sample_size_B == 16)
      return AGX_G17P_MRT_WBOIT;

   const bool rt0_rgba8 =
      tib->logical_format[0] == PIPE_FORMAT_R8G8B8A8_UNORM ||
      tib->logical_format[0] == PIPE_FORMAT_B8G8R8A8_UNORM ||
      tib->logical_format[0] == PIPE_FORMAT_R8G8B8X8_UNORM ||
      tib->logical_format[0] == PIPE_FORMAT_B8G8R8X8_UNORM;
   const bool rt1_rgba8 =
      tib->logical_format[1] == PIPE_FORMAT_R8G8B8A8_UNORM ||
      tib->logical_format[1] == PIPE_FORMAT_B8G8R8A8_UNORM ||
      tib->logical_format[1] == PIPE_FORMAT_R8G8B8X8_UNORM ||
      tib->logical_format[1] == PIPE_FORMAT_B8G8R8X8_UNORM;

   /* These logical channel orders and alpha-ignored variants all use the
    * same four-lane U8 tile representation. Swizzle and X-channel discard
    * live in the per-attachment PBE descriptor, outside the shared MRT EOT
    * program. */
   if (rt0_rgba8 && rt1_rgba8 &&
       tib->_offset_B[1] == 4 && tib->sample_size_B == 8)
      return AGX_G17P_MRT_RGBA8;

   if (tib->logical_format[0] == PIPE_FORMAT_R11G11B10_FLOAT &&
       tib->logical_format[1] == PIPE_FORMAT_R11G11B10_FLOAT &&
       tib->_offset_B[1] == 4 && tib->sample_size_B == 8)
      return AGX_G17P_MRT_R11G11B10;

   return agx_g17p_classify_mixed_mrt(tib, active_count);
}

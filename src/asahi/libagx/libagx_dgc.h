/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "compiler/libcl/libcl.h"
#include "compiler/shader_enums.h"
#include "lib/agx_chip.h"
#include "lib/agx_vdm_barrier_core.h"
#include "lib/agx_vdm_index_list_core.h"
#include "agx_pack.h"

#define agx_push(ptr, T, cfg)                                                  \
   for (unsigned _loop = 0; _loop < 1;                                         \
        ++_loop, ptr = (GLOBAL void *)(((uintptr_t)ptr) + AGX_##T##_LENGTH))   \
      agx_pack(ptr, T, cfg)

#define agx_push_packed(ptr, src, T)                                           \
   static_assert(sizeof(src) == AGX_##T##_LENGTH,                             \
                 "packed AGX command size must match the hardware ABI");     \
   memcpy(ptr, &src, sizeof(src));                                             \
   ptr = (GLOBAL void *)(((uintptr_t)ptr) + sizeof(src));

static inline enum agx_index_size
agx_translate_index_size(uint8_t size_B)
{
   /* Index sizes are encoded logarithmically */
   static_assert(__builtin_ctz(1) == AGX_INDEX_SIZE_U8,
                 "8-bit index size encoding must match the hardware ABI");
   static_assert(__builtin_ctz(2) == AGX_INDEX_SIZE_U16,
                 "16-bit index size encoding must match the hardware ABI");
   static_assert(__builtin_ctz(4) == AGX_INDEX_SIZE_U32,
                 "32-bit index size encoding must match the hardware ABI");

   assert((size_B == 1) || (size_B == 2) || (size_B == 4));
   return __builtin_ctz(size_B);
}

static inline unsigned
agx_indices_to_B(unsigned x, enum agx_index_size size)
{
   return x << size;
}

static inline uint8_t
agx_index_size_to_B(enum agx_index_size size)
{
   return agx_indices_to_B(1, size);
}

struct agx_workgroup {
   uint32_t x, y, z;
};

static inline struct agx_workgroup
agx_workgroup(uint32_t x, uint32_t y, uint32_t z)
{
   return (struct agx_workgroup){.x = x, .y = y, .z = z};
}

static inline unsigned
agx_workgroup_threads(struct agx_workgroup wg)
{
   return wg.x * wg.y * wg.z;
}

/* VK_EXT_transform_feedback converts the byte counter written by streamout
 * into a non-indexed indirect draw count.  Keep the arithmetic shared by the
 * GPU helper and host oracle, including the saturating subtraction required
 * when counterOffset is larger than the captured byte count. */
static inline uint32_t
agx_xfb_vertex_count(uint32_t byte_count, uint32_t counter_offset,
                     uint32_t vertex_stride)
{
   return vertex_stride && byte_count > counter_offset
             ? (byte_count - counter_offset) / vertex_stride
             : 0;
}

struct agx_grid {
   enum agx_cdm_mode mode;
   union {
      uint32_t count[3];
      uint64_t ptr;
   };
};

static struct agx_grid
agx_3d(uint32_t x, uint32_t y, uint32_t z)
{
   return (struct agx_grid){.mode = AGX_CDM_MODE_DIRECT, .count = {x, y, z}};
}

static struct agx_grid
agx_1d(uint32_t x)
{
   return agx_3d(x, 1, 1);
}

static struct agx_grid
agx_2d(uint32_t x, uint32_t y)
{
   return agx_3d(x, y, 1);
}

static struct agx_grid
agx_grid_indirect(uint64_t ptr)
{
   return (struct agx_grid){.mode = AGX_CDM_MODE_INDIRECT_GLOBAL, .ptr = ptr};
}

static struct agx_grid
agx_grid_indirect_local(uint64_t ptr)
{
   return (struct agx_grid){.mode = AGX_CDM_MODE_INDIRECT_LOCAL, .ptr = ptr};
}

static inline bool
agx_is_indirect(struct agx_grid grid)
{
   return grid.mode != AGX_CDM_MODE_DIRECT;
}

enum agx_barrier {
   /* No barrier/cache operations needed */
   AGX_BARRIER_NONE = 0,

   /* Catch-all for all defined barriers. Because we have not yet
    * reverse-engineered the finer details here, this is the only barrier we
    * have....
    */
   AGX_BARRIER_ALL = (1 << 0),

   /* Specifies ordering requirements for the compute job to happen before/after
    * the concurrent enqueued graphics on a graphical command buffer.
    */
   AGX_PREGFX = (1 << 1),
   AGX_POSTGFX = (1 << 2),
};

struct agx_draw {
   struct agx_grid b;
   uint64_t index_buffer;
   uint64_t index_buffer_range_B;
   uint32_t start;
   uint32_t index_bias;
   uint32_t start_instance;

   /* Primitive restart enabled. If true, implies indexed */
   bool restart;
   enum agx_index_size index_size;

   /* TODO: Optimize this boolean. We can't just check if index_buffer != 0
    * because that breaks with null index buffers.
    */
   bool indexed;
};

static inline struct agx_draw
agx_draw_indirect(uint64_t ptr)
{
   return (struct agx_draw){.b = agx_grid_indirect(ptr)};
}

static inline struct agx_draw
agx_draw_indexed(uint32_t index_count, uint32_t instance_count,
                 uint32_t first_index, uint32_t index_bias,
                 uint32_t first_instance, uint64_t buf, uint64_t range_B,
                 enum agx_index_size index_size, bool restart)
{
   return (struct agx_draw){
      .b = agx_3d(index_count, instance_count, 1),
      .index_buffer = buf,
      .index_buffer_range_B = range_B,
      .start = first_index,
      .index_bias = index_bias,
      .start_instance = first_instance,
      .index_size = index_size,
      .restart = restart,
      .indexed = true,
   };
}

static inline struct agx_draw
agx_draw_indexed_indirect(uint64_t ptr, uint64_t buf, uint64_t range_B,
                          enum agx_index_size index_size, bool restart)
{
   return (struct agx_draw){
      .b = agx_grid_indirect(ptr),
      .index_buffer = buf,
      .index_buffer_range_B = range_B,
      .index_size = index_size,
      .restart = restart,
      .indexed = true,
   };
}

static inline uint64_t
agx_draw_index_range_B(struct agx_draw d)
{
   uint64_t range_B = d.index_buffer_range_B;
   if (!agx_is_indirect(d.b)) {
      const uint64_t start_B = (uint64_t)d.start << d.index_size;
      range_B = start_B < range_B ? range_B - start_B : 0;
   }

   return range_B;
}

/* Bytes of the hardware index fetch window for `span_B` bytes of indices
 * starting at `ib`. The window is whole dwords from the dword that holds the
 * first index. The bound range is exact and may end inside a dword, so count
 * to the dword holding its last byte. With a dword-aligned binding this is
 * the old rounding of the range itself. */
static inline uint64_t
agx_index_fetch_window_B(uint64_t ib, uint64_t span_B)
{
   return (span_B + (ib & 3) + 3) & ~(uint64_t)3;
}

static inline unsigned
agx_draw_index_range_el(struct agx_draw d)
{
   assert(d.indexed);
   const uint64_t el = agx_draw_index_range_B(d) >> d.index_size;
   return el > 0xffffffffu ? 0xffffffffu : (unsigned)el;
}

static inline uint64_t
agx_draw_index_buffer(struct agx_draw d)
{
   assert(d.indexed);

   uint64_t ib = d.index_buffer;
   if (!agx_is_indirect(d.b))
      ib += (uint64_t)d.start << d.index_size;

   return ib;
}

static inline uint32_t
agx_draw_direct_index_buffer_size_dwords(struct agx_draw d, uint64_t ib)
{
   const uint64_t range_B = agx_draw_index_range_B(d);
   const uint64_t max_dwords_B = (uint64_t)0xffffffffu * 4;
   if (range_B >= max_dwords_B)
      return 0xffffffffu;

   const uint64_t dwords = agx_index_fetch_window_B(ib, range_B) >> 2;
   return dwords > 0xffffffffu ? 0xffffffffu : (uint32_t)dwords;
}

static inline uint64_t
agx_draw_legacy_index_buffer_size_B(struct agx_draw d, uint64_t ib)
{
   return (uint64_t)agx_draw_direct_index_buffer_size_dwords(d, ib) << 2;
}

/* Apple's AGX3 indirect indexed record stores the available index-buffer
 * range in dwords minus one.  Use a widened intermediate so a range near the
 * u32 ceiling cannot wrap while rounding up. */
static inline bool
agx_draw_indirect_index_buffer_size_word(struct agx_draw d,
                                         uint32_t *size_word)
{
   assert(d.indexed && agx_is_indirect(d.b));

   const uint64_t range_B = agx_draw_index_range_B(d);
   const uint64_t max_dwords_B = (uint64_t)0xffffffffu * 4;
   if (range_B == 0 || range_B > max_dwords_B)
      return false;

   const uint64_t dwords = agx_index_fetch_window_B(
                              agx_draw_index_buffer(d), range_B) >> 2;
   if (dwords == 0 || dwords > (uint64_t)0xffffffffu)
      return false;

   *size_word = (uint32_t)(dwords - 1);
   return true;
}

static inline uint32_t
agx_primitive_restart_index(enum agx_index_size index_size)
{
   return BITFIELD_MASK(8 * agx_index_size_to_B(index_size));
}

static bool
agx_direct_draw_overreads_indices(struct agx_draw d)
{
   uint64_t range_B = ((uint64_t)d.start + d.b.count[0]) << d.index_size;
   return range_B > d.index_buffer_range_B;
}

static inline bool
agx_vdm_draw_supported(enum agx_chip chip, struct agx_draw draw,
                       enum agx_primitive topology)
{
   switch (chip) {
   case AGX_CHIP_G13G:
   case AGX_CHIP_G13X:
   case AGX_CHIP_G14G:
   case AGX_CHIP_G14X:
   case AGX_CHIP_G15P:
      return true;
   default:
      break;
   }

   if (!agx_vdm_index_list_direct_target_supported_inline(chip))
      return false;

   const uint64_t ib = draw.indexed ? agx_draw_index_buffer(draw) : 0;
   if (agx_is_indirect(draw.b)) {
      uint32_t size_word = 0;
      if (draw.indexed &&
          !agx_draw_indirect_index_buffer_size_word(draw, &size_word))
         return false;

      struct agx_vdm_index_list_indirect_words words;
      return agx_vdm_index_list_indirect_pack_words(
         chip, (uint8_t)topology, draw.indexed, draw.restart,
         (uint8_t)draw.index_size, ib, draw.b.ptr,
         draw.indexed ? agx_primitive_restart_index(draw.index_size) : 0,
         size_word, &words);
   }

   struct agx_vdm_index_list_direct_words words;
   return agx_vdm_index_list_direct_pack_words(
      chip, (uint8_t)topology, draw.indexed, draw.restart,
      (uint8_t)draw.index_size, ib, draw.b.count[0], draw.b.count[1],
      draw.indexed ? draw.index_bias : draw.start,
      draw.indexed ? agx_draw_direct_index_buffer_size_dwords(draw, ib) : 0,
      &words);
}

static inline GLOBAL uint32_t *
agx_cdm_launch(GLOBAL uint32_t *out, enum agx_chip chip, struct agx_grid grid,
               struct agx_workgroup wg,
               struct agx_cdm_launch_word_0_packed launch, uint32_t usc)
{
#ifndef __OPENCL_VERSION__
   struct agx_cdm_launch_word_0_packed mode;
   agx_pack(&mode, CDM_LAUNCH_WORD_0, cfg) {
      cfg.mode = grid.mode;
   }

   agx_merge(launch, mode, CDM_LAUNCH_WORD_0);
#endif

   agx_push_packed(out, launch, CDM_LAUNCH_WORD_0);

   agx_push(out, CDM_LAUNCH_WORD_1, cfg) {
      cfg.pipeline = usc;
   }

   if (chip == AGX_CHIP_G14X) {
      agx_push(out, CDM_UNK_G14X, cfg)
         ;
   }

   if (agx_is_indirect(grid)) {
      agx_push(out, CDM_INDIRECT, cfg) {
         cfg.address_hi = grid.ptr >> 32;
         cfg.address_lo = grid.ptr;
      }
   } else {
      agx_push(out, CDM_GLOBAL_SIZE, cfg) {
         cfg.x = grid.count[0];
         cfg.y = grid.count[1];
         cfg.z = grid.count[2];
      }
   }

   if (grid.mode != AGX_CDM_MODE_INDIRECT_LOCAL) {
      agx_push(out, CDM_LOCAL_SIZE, cfg) {
         cfg.x = wg.x;
         cfg.y = wg.y;
         cfg.z = wg.z;
      }
   }

   return out;
}

/* `restart_index` is the value an indexed draw's indices are compared with
 * when draw.restart is set. Index List records that carry their own restart
 * State Update write it there; older records take it from the separate
 * VDM_STATE_RESTART_INDEX the caller emits. */
static inline GLOBAL uint32_t *
agx_vdm_draw(GLOBAL uint32_t *out, enum agx_chip chip, struct agx_draw draw,
             enum agx_primitive topology, uint32_t restart_index)
{
   uint64_t ib = draw.indexed ? agx_draw_index_buffer(draw) : 0;

   if (agx_vdm_index_list_direct_target_supported_inline(chip)) {
      if (agx_is_indirect(draw.b)) {
         uint32_t size_word = 0;
         const bool size_valid =
            !draw.indexed ||
            agx_draw_indirect_index_buffer_size_word(draw, &size_word);
         struct agx_vdm_index_list_indirect_words words;
         const bool valid =
            size_valid && agx_vdm_index_list_indirect_pack_words(
                             chip, (uint8_t)topology, draw.indexed,
                             draw.restart, (uint8_t)draw.index_size, ib,
                             draw.b.ptr, draw.indexed ? restart_index : 0,
                             size_word, &words);
         assert(valid);
         if (!valid)
            return out;

         for (unsigned i = 0; i < words.count; ++i)
            *out++ = words.words[i];
         return out;
      } else {
         struct agx_vdm_index_list_direct_words words;
         const bool valid = agx_vdm_index_list_direct_pack_words(
            chip, (uint8_t)topology, draw.indexed, draw.restart,
            (uint8_t)draw.index_size, ib, draw.b.count[0], draw.b.count[1],
            draw.indexed ? draw.index_bias : draw.start,
            draw.indexed ? agx_draw_direct_index_buffer_size_dwords(draw, ib)
                         : 0,
            &words);
         assert(valid);
         if (!valid)
            return out;

         for (unsigned i = 0; i < words.count; ++i)
            *out++ = words.words[i];
         return out;
      }
   }

   assert(agx_vdm_draw_supported(chip, draw, topology));
   if (!agx_vdm_draw_supported(chip, draw, topology))
      return out;

   agx_push(out, INDEX_LIST, cfg) {
      cfg.primitive = topology;

      if (agx_is_indirect(draw.b)) {
         cfg.indirect_buffer_present = true;
      } else {
         cfg.instance_count_present = true;
         cfg.index_count_present = true;
         cfg.start_present = true;
      }

      if (draw.indexed) {
         cfg.restart_enable = draw.restart;
         cfg.index_buffer_hi = ib >> 32;
         cfg.index_size = draw.index_size;

         cfg.index_buffer_present = true;
         cfg.index_buffer_size_present = true;
      }
   }

   if (draw.indexed) {
      agx_push(out, INDEX_LIST_BUFFER_LO, cfg) {
         cfg.buffer_lo = ib;
      }
   }

   if (agx_is_indirect(draw.b)) {
      agx_push(out, INDEX_LIST_INDIRECT_BUFFER, cfg) {
         cfg.address_hi = draw.b.ptr >> 32;
         cfg.address_lo = draw.b.ptr & BITFIELD_MASK(32);
      }
   } else {
      agx_push(out, INDEX_LIST_COUNT, cfg) {
         cfg.count = draw.b.count[0];
      }

      agx_push(out, INDEX_LIST_INSTANCES, cfg) {
         cfg.count = draw.b.count[1];
      }

      agx_push(out, INDEX_LIST_START, cfg) {
         cfg.start = draw.indexed ? draw.index_bias : draw.start;
      }
   }

   if (draw.indexed) {
      agx_push(out, INDEX_LIST_BUFFER_SIZE, cfg) {
         cfg.size = agx_draw_legacy_index_buffer_size_B(draw, ib);
      }
   }

   return out;
}

/* agx_vdm_draw for G18G's callers, which also accepts AGX_CHIP_G18G (packed
 * as G17G's records). `unk2` is the HAL300 Index List word the pipeline
 * selects (agx_vdm_index_list_hal300_unk2); other generations ignore it. */
static inline GLOBAL uint32_t *
agx_vdm_draw_unk2(GLOBAL uint32_t *out, enum agx_chip chip,
                  struct agx_draw draw, enum agx_primitive topology,
                  uint32_t restart_index, uint32_t unk2)
{
   uint64_t ib = draw.indexed ? agx_draw_index_buffer(draw) : 0;

   if (agx_vdm_index_list_indexed_or_indirect_supported_inline(chip)) {
      if (agx_is_indirect(draw.b)) {
         uint32_t size_word = 0;
         const bool size_valid =
            !draw.indexed ||
            agx_draw_indirect_index_buffer_size_word(draw, &size_word);
         struct agx_vdm_index_list_indirect_words words;
         const bool valid =
            size_valid && agx_vdm_index_list_indirect_pack_words_unk2(
                             chip, (uint8_t)topology, draw.indexed,
                             draw.restart, (uint8_t)draw.index_size, ib,
                             draw.b.ptr, draw.indexed ? restart_index : 0,
                             size_word, unk2, &words);
         assert(valid);
         if (!valid)
            return out;

         for (unsigned i = 0; i < words.count; ++i)
            *out++ = words.words[i];
         return out;
      } else {
         struct agx_vdm_index_list_direct_words words;
         const bool valid = agx_vdm_index_list_direct_pack_words_unk2(
            chip, (uint8_t)topology, draw.indexed, draw.restart,
            (uint8_t)draw.index_size, ib, draw.b.count[0], draw.b.count[1],
            draw.indexed ? draw.index_bias : draw.start,
            draw.indexed ? agx_draw_direct_index_buffer_size_dwords(draw, ib)
                         : 0,
            unk2, &words);
         assert(valid);
         if (!valid)
            return out;

         for (unsigned i = 0; i < words.count; ++i)
            *out++ = words.words[i];
         return out;
      }
   }

   assert(agx_vdm_draw_supported(chip, draw, topology));
   if (!agx_vdm_draw_supported(chip, draw, topology))
      return out;

   agx_push(out, INDEX_LIST, cfg) {
      cfg.primitive = topology;

      if (agx_is_indirect(draw.b)) {
         cfg.indirect_buffer_present = true;
      } else {
         cfg.instance_count_present = true;
         cfg.index_count_present = true;
         cfg.start_present = true;
      }

      if (draw.indexed) {
         cfg.restart_enable = draw.restart;
         cfg.index_buffer_hi = ib >> 32;
         cfg.index_size = draw.index_size;

         cfg.index_buffer_present = true;
         cfg.index_buffer_size_present = true;
      }
   }

   if (draw.indexed) {
      agx_push(out, INDEX_LIST_BUFFER_LO, cfg) {
         cfg.buffer_lo = ib;
      }
   }

   if (agx_is_indirect(draw.b)) {
      agx_push(out, INDEX_LIST_INDIRECT_BUFFER, cfg) {
         cfg.address_hi = draw.b.ptr >> 32;
         cfg.address_lo = draw.b.ptr & BITFIELD_MASK(32);
      }
   } else {
      agx_push(out, INDEX_LIST_COUNT, cfg) {
         cfg.count = draw.b.count[0];
      }

      agx_push(out, INDEX_LIST_INSTANCES, cfg) {
         cfg.count = draw.b.count[1];
      }

      agx_push(out, INDEX_LIST_START, cfg) {
         cfg.start = draw.indexed ? draw.index_bias : draw.start;
      }
   }

   if (draw.indexed) {
      agx_push(out, INDEX_LIST_BUFFER_SIZE, cfg) {
         cfg.size = agx_draw_legacy_index_buffer_size_B(draw, ib);
      }
   }

   return out;
}

static inline uint32_t
agx_vdm_draw_size(enum agx_chip chip, struct agx_draw draw)
{
   if (agx_vdm_index_list_direct_target_supported_inline(chip)) {
      const uint64_t ib = draw.indexed ? agx_draw_index_buffer(draw) : 0;
      if (agx_is_indirect(draw.b)) {
         uint32_t size_word = 0;
         if (draw.indexed &&
             !agx_draw_indirect_index_buffer_size_word(draw, &size_word))
            return 0;

         struct agx_vdm_index_list_indirect_words words;
         if (!agx_vdm_index_list_indirect_pack_words(
                chip, 0, draw.indexed, draw.restart,
                (uint8_t)draw.index_size, ib, draw.b.ptr,
                draw.indexed
                   ? agx_primitive_restart_index(draw.index_size)
                   : 0,
                size_word, &words))
            return 0;

         return words.count * 4;
      } else {
         struct agx_vdm_index_list_direct_words words;
         if (!agx_vdm_index_list_direct_pack_words(
                chip, 0, draw.indexed, draw.restart,
                (uint8_t)draw.index_size, ib, draw.b.count[0], draw.b.count[1],
                draw.indexed ? draw.index_bias : draw.start,
                draw.indexed
                   ? agx_draw_direct_index_buffer_size_dwords(draw, ib)
                   : 0,
                &words))
            return 0;

         return words.count * 4;
      }
   }

   switch (chip) {
   case AGX_CHIP_G15G:
   case AGX_CHIP_G15S:
   case AGX_CHIP_G15C:
   case AGX_CHIP_G15D:
   case AGX_CHIP_G16P:
   case AGX_CHIP_G16G:
   case AGX_CHIP_G16S:
   case AGX_CHIP_G16C:
   case AGX_CHIP_G17P:
   case AGX_CHIP_G17A:
   case AGX_CHIP_G17G:
   case AGX_CHIP_G17S:
   case AGX_CHIP_G17C:
      return 0;
   default:
      break;
   }

   uint32_t size = AGX_INDEX_LIST_LENGTH;

   if (agx_is_indirect(draw.b)) {
      size += AGX_INDEX_LIST_INDIRECT_BUFFER_LENGTH;
   } else {
      size += AGX_INDEX_LIST_COUNT_LENGTH;
      size += AGX_INDEX_LIST_INSTANCES_LENGTH;
      size += AGX_INDEX_LIST_START_LENGTH;
   }

   if (draw.indexed) {
      size += AGX_INDEX_LIST_BUFFER_LO_LENGTH;
      size += AGX_INDEX_LIST_BUFFER_SIZE_LENGTH;
   }

   return size;
}

static inline GLOBAL uint32_t *
agx_vdm_barrier(GLOBAL uint32_t *out, enum agx_chip chip,
                bool kernel_admits_inval)
{
   /* The AGX3 encoding uses this exact type-1 memory-barrier word on every
    * byte-witnessed G15--G17 identity.  Do not feed those chips through the
    * legacy genxml packet below: that packet is AGX2 and has a different
    * opcode/field layout.  Keep the unwitnessed G16P/G17A identities closed
    * instead of guessing from their neighbouring families.  G18G memory
    * barriers use the same word. */
   if (agx_chip_has_record(chip, AGX_RECORD_VDM_BARRIER)) {
      /* With the USC cache invalidate where the kernel admits it
       * (agx_vdm_barrier_core.h). */
      *out++ = agx_vdm_barrier_core_word(chip, kernel_admits_inval);
      return out;
   }

   switch (chip) {
   case AGX_CHIP_G16P:
   case AGX_CHIP_G17A:
   case AGX_CHIP_UNKNOWN:
      return out;

   default:
      break;
   }

   agx_push(out, VDM_BARRIER, cfg) {
      cfg.unk_4 = (chip == AGX_CHIP_G13X);
      cfg.unk_5 = true;
      cfg.unk_6 = true;
      cfg.unk_8 = true;
      cfg.unk_11 = true;
      cfg.unk_20 = true;
      cfg.unk_24 = (chip == AGX_CHIP_G13X) || (chip == AGX_CHIP_G14X);
      cfg.unk_26 = (chip == AGX_CHIP_G13X);
   }

   return out;
}

static inline GLOBAL uint32_t *
agx_cdm_barrier(GLOBAL uint32_t *out, enum agx_chip chip)
{
   agx_push(out, CDM_BARRIER, cfg) {
      cfg.unk_5 = true;
      cfg.unk_6 = true;
      cfg.unk_8 = true;
      // cfg.unk_11 = true;
      // cfg.unk_20 = true;
      // cfg.unk_24 = true; if clustered?
      if (chip == AGX_CHIP_G13X) {
         cfg.unk_4 = true;
         // cfg.unk_26 = true;
      }

      /* With multiple launches in the same CDM stream, we can get cache
       * coherency (? or sync?) issues. We hit this with blits, which need - in
       * between dispatches - need the PBE cache to be flushed and the texture
       * cache to be invalidated. Until we know what bits mean what exactly,
       * let's just set these after every launch to be safe. We can revisit in
       * the future when we figure out what the bits mean.
       */
      cfg.unk_0 = true;
      cfg.unk_1 = true;
      cfg.unk_2 = true;
      cfg.usc_cache_inval = true;
      cfg.unk_4 = true;
      cfg.unk_5 = true;
      cfg.unk_6 = true;
      cfg.unk_7 = true;
      cfg.unk_8 = true;
      cfg.unk_9 = true;
      cfg.unk_10 = true;
      cfg.unk_11 = true;
      cfg.unk_12 = true;
      cfg.unk_13 = true;
      cfg.unk_14 = true;
      cfg.unk_15 = true;
      cfg.unk_16 = true;
      cfg.unk_17 = true;
      cfg.unk_18 = true;
      cfg.unk_19 = true;
   }

   return out;
}

static inline GLOBAL uint32_t *
agx_vdm_return(GLOBAL uint32_t *out)
{
   agx_push(out, VDM_BARRIER, cfg) {
      cfg.returns = true;
   }

   return out;
}

static inline GLOBAL uint32_t *
agx_cdm_return(GLOBAL uint32_t *out)
{
   agx_push(out, CDM_STREAM_RETURN, cfg)
      ;

   return out;
}

static inline GLOBAL uint32_t *
agx_cdm_terminate(GLOBAL uint32_t *out)
{
   agx_push(out, CDM_STREAM_TERMINATE, _)
      ;

   return out;
}

static inline GLOBAL uint32_t *
agx_vdm_terminate(GLOBAL uint32_t *out)
{
   agx_push(out, VDM_STREAM_TERMINATE, _)
      ;

   return out;
}

static inline GLOBAL uint32_t *
agx_cdm_jump(GLOBAL uint32_t *out, uint64_t target)
{
   agx_push(out, CDM_STREAM_LINK, cfg) {
      cfg.target_lo = target & BITFIELD_MASK(32);
      cfg.target_hi = target >> 32;
   }

   return out;
}

static inline GLOBAL uint32_t *
agx_vdm_jump(GLOBAL uint32_t *out, uint64_t target)
{
   agx_push(out, VDM_STREAM_LINK, cfg) {
      cfg.target_lo = target & BITFIELD_MASK(32);
      cfg.target_hi = target >> 32;
   }

   return out;
}

static inline GLOBAL uint32_t *
agx_cs_jump(GLOBAL uint32_t *out, uint64_t target, bool vdm)
{
   return vdm ? agx_vdm_jump(out, target) : agx_cdm_jump(out, target);
}

static inline GLOBAL uint32_t *
agx_cdm_call(GLOBAL uint32_t *out, uint64_t target)
{
   agx_push(out, CDM_STREAM_LINK, cfg) {
      cfg.target_lo = target & BITFIELD_MASK(32);
      cfg.target_hi = target >> 32;
      cfg.with_return = true;
   }

   return out;
}

static inline GLOBAL uint32_t *
agx_vdm_call(GLOBAL uint32_t *out, uint64_t target)
{
   agx_push(out, VDM_STREAM_LINK, cfg) {
      cfg.target_lo = target & BITFIELD_MASK(32);
      cfg.target_hi = target >> 32;
      cfg.with_return = true;
   }

   return out;
}

#define AGX_MAX_LINKED_USC_SIZE                                                \
   (AGX_USC_PRESHADER_LENGTH + AGX_USC_FRAGMENT_PROPERTIES_LENGTH +            \
    AGX_USC_REGISTERS_LENGTH + AGX_USC_SHADER_LENGTH + AGX_USC_SHARED_LENGTH + \
    AGX_USC_SAMPLER_LENGTH + (AGX_USC_UNIFORM_LENGTH * 9))

/*
 * This data structure contains everything needed to dispatch a compute shader
 * (and hopefully eventually graphics?).
 *
 * It is purely flat, no CPU pointers. That makes it suitable for sharing
 * between CPU and GPU. The intention is that it is packed on the CPU side and
 * then consumed on either host or device for dispatching work.
 */
struct agx_shader {
   struct agx_cdm_launch_word_0_packed launch;
   struct agx_workgroup workgroup;

   struct {
      uint32_t size;
      uint8_t data[AGX_MAX_LINKED_USC_SIZE];
   } usc;
};

/* Opaque structure representing a USC program being constructed */
struct agx_usc_builder {
   GLOBAL uint8_t *head;

#ifndef NDEBUG
   uint8_t *begin;
   size_t size;
#endif
} PACKED;

static struct agx_usc_builder
agx_usc_builder(GLOBAL void *out, ASSERTED size_t size)
{
   return (struct agx_usc_builder){
      .head = out,

#ifndef NDEBUG
      .begin = out,
      .size = size,
#endif
   };
}

static bool
agx_usc_builder_validate(struct agx_usc_builder *b, size_t size)
{
#ifndef NDEBUG
   assert(((b->head - b->begin) + size) <= b->size);
#endif

   return true;
}

#define agx_usc_pack(b, struct_name, template)                                 \
   for (bool it =                                                              \
           agx_usc_builder_validate((b), AGX_USC_##struct_name##_LENGTH);      \
        it; it = false, (b)->head += AGX_USC_##struct_name##_LENGTH)           \
      agx_pack((b)->head, USC_##struct_name, template)

#define agx_usc_push_blob(b, blob, length)                                     \
   for (bool it = agx_usc_builder_validate((b), length); it;                   \
        it = false, (b)->head += length)                                       \
      memcpy((b)->head, blob, length);

#define agx_usc_push_packed(b, struct_name, packed)                            \
   agx_usc_push_blob(b, packed.opaque, AGX_USC_##struct_name##_LENGTH);

static void
agx_usc_uniform(struct agx_usc_builder *b, unsigned start_halfs,
                unsigned size_halfs, uint64_t buffer)
{
   assert((start_halfs + size_halfs) <= (1 << 9) && "uniform file overflow");
   assert(size_halfs <= 64 && "caller's responsibility to split");
   assert(size_halfs > 0 && "no empty uniforms");

   if (start_halfs & BITFIELD_BIT(8)) {
      agx_usc_pack(b, UNIFORM_HIGH, cfg) {
         cfg.start_halfs = start_halfs & BITFIELD_MASK(8);
         cfg.size_halfs = size_halfs;
         cfg.buffer = buffer;
      }
   } else {
      agx_usc_pack(b, UNIFORM, cfg) {
         cfg.start_halfs = start_halfs;
         cfg.size_halfs = size_halfs;
         cfg.buffer = buffer;
      }
   }
}

static inline void
agx_usc_words_precomp(GLOBAL uint32_t *out, CONST struct agx_shader *s,
                      uint64_t data, unsigned data_size)
{
   /* Map the data directly as uniforms starting at u0 */
   struct agx_usc_builder b = agx_usc_builder(out, sizeof(s->usc.data));
   agx_usc_uniform(&b, 0, DIV_ROUND_UP(data_size, 2), data);
   agx_usc_push_blob(&b, s->usc.data, s->usc.size);
}

static inline uint32_t
agx_robust_index_available(uint32_t first, uint64_t range_B,
                           enum agx_index_size index_size)
{
   const uint64_t start_B = (uint64_t)first << index_size;
   if (start_B >= range_B)
      return 0;
   const uint64_t avail = (range_B - start_B) >> index_size;
   return avail > 0xffffffffu ? 0xffffffffu : (uint32_t)avail;
}

/* Also used by host tests: copy exactly the accessible bytes, including a
 * short or unaligned source, and zero every byte of the extension. A uint32
 * index can require 128 bytes of padding, more than one byte per lane. */
static inline void
agx_robust_index_copy(GLOBAL uint8_t *dst, CONST uint8_t *src,
                      uint32_t available_B, uint32_t output_B, uint32_t lane)
{
   for (uint64_t i = lane; i < output_B; i += 32)
      dst[i] = i < available_B ? src[i] : 0;
}

static inline uint32_t
agx_robust_index_clamped_count(uint32_t requested, uint32_t available,
                                enum agx_primitive topology)
{
   /* Repeated zero indices are visible points, not degenerate primitives. */
   if (topology == AGX_PRIMITIVE_POINTS)
      return requested;

   const uint32_t limit =
      available > 0xffffffffu - 32
         ? 0xffffffffu
         : available + 32;
   return requested < limit ? requested : limit;
}

static inline bool
agx_robust_index_shadow_sizes(uint32_t count, enum agx_index_size index_size,
                              uint32_t *payload_B, uint32_t *allocation_B)
{
   if (index_size > AGX_INDEX_SIZE_U32 ||
       count > (0xffffffffu - 15) >> index_size)
      return false;

   *payload_B = agx_indices_to_B(count, index_size);
   *allocation_B = align(*payload_B, 16);
   return true;
}

/*
 * Robust indexed draws are generated on the GPU after the indirect argument
 * record is visible.  G17P's HAL200 direct indexed body is not just the
 * generic Index List: it always starts with the native restart State Update,
 * and a freshly emitted render state additionally requires its context-
 * relative draw-state selector immediately before that body.
 *
 * Keep the exact construction shared between the OpenCL producer and host
 * tests.  Older command-stream generations retain their existing generic
 * body unchanged.
 */
static inline GLOBAL uint32_t *
agx_vdm_draw_robust_index(GLOBAL uint32_t *out, enum agx_chip chip,
                          struct agx_draw draw,
                          enum agx_primitive topology,
                          uint32_t restart_index,
                          uint32_t draw_state_context_offset,
                          bool render_state_emitted,
                          uint64_t empty_draw_jump)
{
   /* `chip` names the record family: every native AGX3 chip encodes these
    * records as G17P does, and passes agx_chip_record_family() so that one
    * precompiled kernel serves them all. */
   /* G17P's native direct indexed packet requires both counts to be nonzero.
    * Indirect-count predication deliberately zeros records above the runtime
    * count, so those slots cannot contain a direct packet. Jump over the
    * fixed-size GPU-generated slot instead. The host supplies the compact
    * context-relative address immediately after this reservation, matching
    * the same non-returning Stream Link used when a CPU-authored VDM chunk
    * fills. A freshly emitted bind tail still requires its selector before
    * any other VDM block. Emit that selector before the link so the link
    * header is not decoded as a context-relative state address. No draw
    * executes and the following CPU-authored command remains the only parser
    * target.
    */
   if ((agx_chip_native_hal200(chip) ||
        agx_vdm_index_list_direct_is_hal300_inline(chip)) &&
       (draw.b.count[0] == 0 || draw.b.count[1] == 0)) {
      if (agx_chip_native_hal200(chip) && render_state_emitted)
         *out++ = draw_state_context_offset;

      *out++ = 0x80000000u | (uint32_t)(empty_draw_jump >> 32);
      *out++ = (uint32_t)empty_draw_jump;
      return out;
   }

   const uint64_t start_B = (uint64_t)draw.start << draw.index_size;
   const uint64_t ib = draw.index_buffer + start_B;
   const uint32_t range_dwords =
      agx_draw_direct_index_buffer_size_dwords(draw, ib);

   if (agx_vdm_index_list_direct_target_supported_inline(chip)) {
      struct agx_vdm_index_list_direct_words words;
      if (!agx_vdm_index_list_direct_pack_words(
             chip, (uint8_t)topology, true, draw.restart,
             (uint8_t)draw.index_size, ib, draw.b.count[0], draw.b.count[1],
             draw.index_bias, range_dwords, &words))
         return out;

      if (agx_chip_native_hal200(chip)) {
         if (render_state_emitted)
            *out++ = draw_state_context_offset;

         *out++ = 0x40000001u;
         *out++ = restart_index;
      }

      for (unsigned i = 0; i < words.count; ++i)
         *out++ = words.words[i];
      return out;
   }

   /* This producer receives a host-validated direct indexed draw. Keep its
    * failure path free of the printf/abort ABI: G17P precompiled machine code
    * is cacheable and cannot embed a process-local diagnostic-buffer address. */
   *out++ = 0x61f00000u | ((uint32_t)(ib >> 32) & 0xffu) |
            ((uint32_t)topology << 8) |
            ((uint32_t)draw.index_size << 17) |
            (draw.restart ? 1u << 16 : 0);
   *out++ = (uint32_t)ib;
   *out++ = draw.b.count[0];
   *out++ = draw.b.count[1];
   *out++ = draw.index_bias;
   *out++ = range_dwords;
   return out;
}

static inline unsigned
libagx_draw_robust_index_vdm_size(enum agx_chip chip,
                                  bool render_state_emitted)
{
   struct agx_draw draw =
      agx_draw_indexed(0, 1, 0, 0, 0, 0, 4, AGX_INDEX_SIZE_U16, false);
   unsigned size_B = agx_vdm_draw_size(chip, draw);

   if (agx_chip_native_hal200(chip))
      size_B += 8 + (render_state_emitted ? 4 : 0);

   return size_B;
}

/* A batch shares its index binding, topology and heap. Every record keeps its
 * own source arguments, VDM destination and native state/empty-draw link. */
#define LIBAGX_ROBUST_INDEX_COALESCE_BLOCK 1024u

struct libagx_robust_index_draw {
   uint64_t vdm;
   uint64_t command;
   uint64_t empty_draw_jump;
   uint32_t draw_state_offset;
   uint32_t render_state_emitted;
};
static_assert(sizeof(struct libagx_robust_index_draw) == 32,
              "robust index batch host/device ABI");

static inline bool
libagx_robust_index_cmd_is_empty(uint64_t command)
{
   CONST uint32_t *words = (CONST uint32_t *)(uintptr_t)command;
   return words[0] == 0u || words[1] == 0u;
}

/* On AGX3 (G15/G16/G17P) and HAL300 (M5/G18G) hardware,
 * agx_vdm_draw_robust_index() emits a VDM_STREAM_LINK over each empty slot
 * because direct indexed packets require non-zero counts. Chasing tens of
 * thousands of skipped-draw links without executing a draw hits the VDM
 * timeout observed on M5 and triggers firmware recovery.
 *
 * Coalesce each contiguous empty run into jumps of up to
 * LIBAGX_ROBUST_INDEX_COALESCE_BLOCK slots (1024 bounds one lane's scan time
 * and is not derived from the hardware limit): only the head of an empty run
 * or a block-aligned empty slot scans forward to the next non-empty draw (or
 * the block boundary), keeping total scan work O(N) while bounding skipped
 * hops to at most ceil(N / block) + 1 (plus any chunk-roll links landed on).
 * Runs identically on lane 0 on GPU and in host unit tests. */
static inline uint64_t
libagx_robust_index_coalesce_empty_jump(
   CONST struct libagx_robust_index_draw *draws, uint32_t g,
   uint32_t batch_count, enum agx_chip chip)
{
   uint64_t empty_draw_jump = draws[g].empty_draw_jump;
   if (!(agx_chip_native_hal200(chip) ||
         agx_vdm_index_list_direct_is_hal300_inline(chip)) ||
       batch_count <= 1u || g >= batch_count ||
       !libagx_robust_index_cmd_is_empty(draws[g].command))
      return empty_draw_jump;

   const uint32_t block = LIBAGX_ROBUST_INDEX_COALESCE_BLOCK;
   bool is_head = (g & (block - 1u)) == 0u;
   if (!is_head)
      is_head = !libagx_robust_index_cmd_is_empty(draws[g - 1u].command);
   if (!is_head)
      return empty_draw_jump;

   const uint32_t block_rem = block - (g & (block - 1u));
   const uint32_t batch_rem = batch_count - g;
   const uint32_t end = g + (block_rem < batch_rem ? block_rem : batch_rem);
   uint32_t first_nonempty = end;

   for (uint32_t j = g + 1u; j < end; ++j) {
      if (!libagx_robust_index_cmd_is_empty(draws[j].command)) {
         first_nonempty = j;
         break;
      }
   }

   return draws[first_nonempty - 1u].empty_draw_jump;
}

#define LIBAGX_INDIRECT_VDM_MAX_RELOCS 8u
#define LIBAGX_INDIRECT_VDM_LANES_PER_DRAW 4u

struct libagx_indirect_ioto_reloc {
   uint64_t dest_lo;
   uint64_t dest_hi;
   uint32_t delta;
   uint16_t offset;
   /* 0 = DRAW_PARAMS, 1 = DRAW_ID */
   uint8_t kind;
   /* bit 0: low LDIMM is 2-byte form, bit 1: high LDIMM is 2-byte form */
   uint8_t short_mask;
};
static_assert(sizeof(struct libagx_indirect_ioto_reloc) == 24,
              "indirect ioto reloc host/device ABI");

struct libagx_indirect_vdm_desc {
   uint64_t sub_vdm;
   uint64_t vdm_template;
   uint64_t in_buffer;
   /* 0 for fixed drawCount (vkCmdDraw[Indexed]Indirect). */
   uint64_t count_buffer;
   uint64_t return_link_compact;
   /* 0 when the draw does not republish vertex state per record. */
   uint64_t ioto_out;
   uint64_t ioto_template;
   /* 0 when the vertex shader does not read gl_DrawID. */
   uint64_t draw_ids_out;
   uint32_t max_draw_count;
   uint32_t in_stride_B;
   uint32_t vdm_record_dwords;
   uint32_t vdm_indirect_word_off;
   /* 0xffffffffu when no VDM State Update pipeline pair is present. */
   uint32_t vdm_pipeline_word_off;
   uint32_t vdm_pipeline_raw_template;
   uint32_t param_offset_B;
   uint32_t ioto_copy_dwords;
   uint32_t ioto_stride_B;
   uint32_t reloc_count;
   struct libagx_indirect_ioto_reloc relocs[LIBAGX_INDIRECT_VDM_MAX_RELOCS];
};
static_assert(sizeof(struct libagx_indirect_vdm_desc) == 296,
              "indirect VDM builder host/device ABI");

static inline uint64_t
libagx_ioto_ldimm_long_immediate(uint32_t immediate)
{
   return ((uint64_t)(immediate & 0x7fu) << 8) |
          ((uint64_t)((immediate >> 25) & 0x7fu) << 25) |
          ((uint64_t)((((immediate >> 7) & 0xfu) << 1) |
                      (((immediate >> 11) & 3u) << 10)) << 32) |
          ((uint64_t)(((immediate >> 13) & 0xfu) |
                      (((immediate >> 17) & 0xffu) << 4)) << 48);
}

static inline uint32_t
libagx_ioto_patch_dword(uint32_t w, uint32_t byte_off, uint32_t patch_off,
                        bool is_short, uint64_t word, uint32_t immediate)
{
   if (is_short) {
      if (patch_off == byte_off)
         return (w & 0xffff00ffu) | ((immediate & 0xffu) << 8);
      if (patch_off == byte_off + 2u)
         return (w & 0x00ffffffu) | ((immediate & 0xffu) << 24);
      return w;
   }

   if (byte_off + 4u <= patch_off || byte_off >= patch_off + 8u)
      return w;

   if (patch_off == byte_off)
      return (uint32_t)word;
   if (patch_off + 4u == byte_off)
      return (uint32_t)(word >> 32);
   if (patch_off == byte_off + 2u)
      return (w & 0x0000ffffu) | (((uint32_t)word & 0xffffu) << 16);
   if (patch_off + 2u == byte_off)
      return (uint32_t)(word >> 16);
   /* patch_off + 6u == byte_off */
   return (w & 0xffff0000u) | (uint32_t)(word >> 48);
}

static inline void
libagx_load_u32x4(CONST uint32_t *src, uint32_t w[4])
{
#ifdef __OPENCL_VERSION__
   uint4 v = *(CONST uint4 *)src;
   w[0] = v.x;
   w[1] = v.y;
   w[2] = v.z;
   w[3] = v.w;
#else
   w[0] = src[0];
   w[1] = src[1];
   w[2] = src[2];
   w[3] = src[3];
#endif
}

static inline void
libagx_store_u32x4(GLOBAL uint32_t *dst, const uint32_t w[4])
{
#ifdef __OPENCL_VERSION__
   *(GLOBAL uint4 *)dst = (uint4)(w[0], w[1], w[2], w[3]);
#else
   dst[0] = w[0];
   dst[1] = w[1];
   dst[2] = w[2];
   dst[3] = w[3];
#endif
}

static inline void
libagx_build_indirect_vdm_sublane(CONST struct libagx_indirect_vdm_desc *desc,
                                  uint32_t tid)
{
   uint32_t count = desc->max_draw_count;
   if (desc->count_buffer) {
      uint32_t raw_count =
         *(const GLOBAL uint32_t *)(uintptr_t)desc->count_buffer;
      if (raw_count < count)
         count = raw_count;
   }

   const uint32_t draw = tid / LIBAGX_INDIRECT_VDM_LANES_PER_DRAW;
   const uint32_t sublane = tid % LIBAGX_INDIRECT_VDM_LANES_PER_DRAW;

   if (draw > count)
      return;

   GLOBAL uint32_t *vdm_out = (GLOBAL uint32_t *)(uintptr_t)(
      desc->sub_vdm + (uint64_t)draw * desc->vdm_record_dwords * 4u);

   if (draw == count) {
      if (sublane == 0) {
         uint64_t target = desc->return_link_compact;
         vdm_out[0] = 0x80000000u | (uint32_t)(target >> 32);
         vdm_out[1] = (uint32_t)target;
      }
      return;
   }

   const uint64_t draw_va = desc->in_buffer + (uint64_t)draw * desc->in_stride_B;
   uint64_t prog_va = 0;

   if (desc->ioto_out) {
      if (desc->draw_ids_out && sublane == 0) {
         GLOBAL uint32_t *draw_ids =
            (GLOBAL uint32_t *)(uintptr_t)desc->draw_ids_out;
         draw_ids[draw] = draw;
      }

      prog_va = desc->ioto_out + (uint64_t)draw * desc->ioto_stride_B;
      GLOBAL uint32_t *ioto_dst = (GLOBAL uint32_t *)(uintptr_t)prog_va;
      CONST uint32_t *ioto_src =
         (CONST uint32_t *)(uintptr_t)desc->ioto_template;
      const uint64_t params_va = draw_va + desc->param_offset_B;
      const uint64_t draw_id_va = desc->draw_ids_out + (uint64_t)draw * 4u;
      const uint32_t ioto_quads = desc->ioto_copy_dwords >> 2;

      for (uint32_t q = sublane; q < ioto_quads;
           q += LIBAGX_INDIRECT_VDM_LANES_PER_DRAW) {
         uint32_t w[4];
         libagx_load_u32x4(ioto_src + q * 4u, w);
         const uint32_t quad_off = q * 16u;

         for (uint32_t r = 0; r < desc->reloc_count; ++r) {
            const struct libagx_indirect_ioto_reloc rel = desc->relocs[r];
            const bool low_short = (rel.short_mask & 1u) != 0;
            const bool hi_short = (rel.short_mask & 2u) != 0;
            const uint32_t lo_off = rel.offset;
            const uint32_t hi_off = lo_off + (low_short ? 2u : 8u);
            const uint32_t hi_end = hi_off + (hi_short ? 2u : 8u);

            if (quad_off + 16u > lo_off && quad_off < hi_end) {
               const uint64_t base =
                  (rel.kind == 0 ? params_va : draw_id_va) + rel.delta;
               const uint32_t lo_imm = (uint32_t)base;
               const uint32_t hi_imm = (uint32_t)(base >> 32);
               const uint64_t lo_word =
                  low_short
                     ? 0
                     : (rel.dest_lo | libagx_ioto_ldimm_long_immediate(lo_imm));
               const uint64_t hi_word =
                  hi_short
                     ? 0
                     : (rel.dest_hi | libagx_ioto_ldimm_long_immediate(hi_imm));

               for (uint32_t lane = 0; lane < 4u; ++lane) {
                  const uint32_t byte_off = quad_off + lane * 4u;
                  w[lane] = libagx_ioto_patch_dword(w[lane], byte_off, lo_off,
                                                    low_short, lo_word, lo_imm);
                  w[lane] = libagx_ioto_patch_dword(w[lane], byte_off, hi_off,
                                                    hi_short, hi_word, hi_imm);
               }
            }
         }

         libagx_store_u32x4(ioto_dst + q * 4u, w);
      }

      for (uint32_t k = (ioto_quads << 2) + sublane; k < desc->ioto_copy_dwords;
           k += LIBAGX_INDIRECT_VDM_LANES_PER_DRAW) {
         uint32_t w = ioto_src[k];
         const uint32_t byte_off = k * 4u;

         for (uint32_t r = 0; r < desc->reloc_count; ++r) {
            const struct libagx_indirect_ioto_reloc rel = desc->relocs[r];
            const bool low_short = (rel.short_mask & 1u) != 0;
            const bool hi_short = (rel.short_mask & 2u) != 0;
            const uint32_t lo_off = rel.offset;
            const uint32_t hi_off = lo_off + (low_short ? 2u : 8u);
            const uint32_t hi_end = hi_off + (hi_short ? 2u : 8u);

            if (byte_off + 4u > lo_off && byte_off < hi_end) {
               const uint64_t base =
                  (rel.kind == 0 ? params_va : draw_id_va) + rel.delta;
               const uint32_t lo_imm = (uint32_t)base;
               const uint32_t hi_imm = (uint32_t)(base >> 32);
               const uint64_t lo_word =
                  low_short
                     ? 0
                     : (rel.dest_lo | libagx_ioto_ldimm_long_immediate(lo_imm));
               const uint64_t hi_word =
                  hi_short
                     ? 0
                     : (rel.dest_hi | libagx_ioto_ldimm_long_immediate(hi_imm));
               w = libagx_ioto_patch_dword(w, byte_off, lo_off, low_short,
                                           lo_word, lo_imm);
               w = libagx_ioto_patch_dword(w, byte_off, hi_off, hi_short,
                                           hi_word, hi_imm);
            }
         }

         ioto_dst[k] = w;
      }
   }

   CONST uint32_t *vdm_src = (CONST uint32_t *)(uintptr_t)desc->vdm_template;
   const uint32_t pipe_off = desc->vdm_pipeline_word_off;
   const uint32_t pipe_w0 =
      desc->vdm_pipeline_raw_template |
      (uint32_t)(((prog_va >> 38) & 0x3ffu) << 22);
   const uint32_t pipe_w1 = (uint32_t)(prog_va >> 6);
   const uint32_t ind_off = desc->vdm_indirect_word_off;
   const uint32_t ind_w0 = (uint32_t)((draw_va >> 32) & 0xffffu);
   const uint32_t ind_w1 = (uint32_t)draw_va & 0xfffffffcu;

   if ((desc->vdm_record_dwords & 3u) == 0u) {
      const uint32_t vdm_quads = desc->vdm_record_dwords >> 2;
      for (uint32_t q = sublane; q < vdm_quads;
           q += LIBAGX_INDIRECT_VDM_LANES_PER_DRAW) {
         uint32_t w[4];
         libagx_load_u32x4(vdm_src + q * 4u, w);
         const uint32_t base_k = q * 4u;
         for (uint32_t lane = 0; lane < 4u; ++lane) {
            const uint32_t k = base_k + lane;
            if (pipe_off != 0xffffffffu) {
               if (k == pipe_off)
                  w[lane] = pipe_w0;
               else if (k == pipe_off + 1u)
                  w[lane] = pipe_w1;
            }
            if (k == ind_off)
               w[lane] = ind_w0;
            else if (k == ind_off + 1u)
               w[lane] = ind_w1;
         }
         libagx_store_u32x4(vdm_out + q * 4u, w);
      }
   } else {
      for (uint32_t k = sublane; k < desc->vdm_record_dwords;
           k += LIBAGX_INDIRECT_VDM_LANES_PER_DRAW) {
         uint32_t w = vdm_src[k];
         if (pipe_off != 0xffffffffu) {
            if (k == pipe_off)
               w = pipe_w0;
            else if (k == pipe_off + 1u)
               w = pipe_w1;
         }
         if (k == ind_off)
            w = ind_w0;
         else if (k == ind_off + 1u)
            w = ind_w1;
         vdm_out[k] = w;
      }
   }
}

/* Software task stage (G17P): one record per task draw. The task kernel
 * stores each workgroup's child grid and payload in its entry, and raises
 * max_children to its child count; the mesh object and mesh shader read them
 * back. An entry is the child grid (16 B) followed by the payload.
 */
struct libagx_task_record {
   uint32_t max_children;
   uint32_t task_count;
   uint32_t stride_B;
   /* Invocations per mesh workgroup, for the mesh invocation statistic. */
   uint32_t mesh_threads;
   uint64_t entries;
   /* Task kernel grid in workgroups, for indirect launches. */
   uint32_t dispatch[3];
   uint32_t pad2;
};
static_assert(sizeof(struct libagx_task_record) == 40,
              "task record host/device ABI");

#define LIBAGX_TASK_ENTRY_PAYLOAD_B 16

static inline unsigned
libagx_remap_adj_count(unsigned count, enum mesa_prim prim)
{
   if (prim == MESA_PRIM_TRIANGLE_STRIP_ADJACENCY) {
      /* Spec gives formula for # of primitives in a tri strip adj */
      unsigned c4 = count >= 4 ? count - 4 : 0;
      return 3 * (c4 / 2);
   } else if (prim == MESA_PRIM_LINE_STRIP_ADJACENCY) {
      return 2 * (count >= 3 ? count - 3 : 0);
   } else {
      /* Adjacency lists just drop half the vertices. */
      return count / 2;
   }
}

/*
 * Contiguous run of indirect draw VDM records with constant byte stride inside
 * one VDM chunk. Draw `i` in `[first, first + count)` begins at
 * `base_addr + (uint64_t)(i - first) * stride` (always a top-level VDM record
 * boundary with at least 8 bytes written for that draw).
 */
struct libagx_vdm_draw_span {
   uint64_t base_addr;
   uint32_t first;
   uint32_t count;
   uint32_t stride;
   uint32_t _pad[3];
};
static_assert(sizeof(struct libagx_vdm_draw_span) == 32,
              "VDM draw span host/device ABI");

/*
 * Per-command-buffer mutable state for reusable command buffers: records the
 * previously patched VDM address and its original 8 bytes so a subsequent
 * submission restores them before patching at the new runtime draw count.
 */
struct libagx_vdm_truncate_state {
   uint64_t patched_addr;
   uint32_t orig_w0;
   uint32_t orig_w1;
};
static_assert(sizeof(struct libagx_vdm_truncate_state) == 16,
              "VDM truncate state host/device ABI");

static inline bool
libagx_vdm_span_append(GLOBAL struct libagx_vdm_draw_span *spans,
                       uint32_t *num_spans, uint32_t capacity,
                       uint32_t draw_idx, uint64_t addr)
{
   if (*num_spans > 0) {
      GLOBAL struct libagx_vdm_draw_span *last = &spans[*num_spans - 1];
      if (last->first + last->count == draw_idx) {
         if (last->count == 1) {
            if (addr > last->base_addr &&
                (addr - last->base_addr) <= UINT32_MAX) {
               last->stride = (uint32_t)(addr - last->base_addr);
               last->count = 2;
               return true;
            }
         } else if (addr == last->base_addr +
                               (uint64_t)last->count * last->stride) {
            last->count++;
            return true;
         }
      }
   }

   if (*num_spans >= capacity)
      return false;

   spans[(*num_spans)++] = (struct libagx_vdm_draw_span){
      .base_addr = addr,
      .first = draw_idx,
      .count = 1,
      .stride = 0,
   };
   return true;
}

static inline uint64_t
libagx_vdm_span_find_addr(CONST struct libagx_vdm_draw_span *spans,
                          uint32_t num_spans, uint32_t draw_idx)
{
   uint32_t lo = 0;
   uint32_t hi = num_spans;

   while (lo < hi) {
      uint32_t mid = lo + ((hi - lo) >> 1);
      struct libagx_vdm_draw_span s = spans[mid];
      if (draw_idx < s.first) {
         hi = mid;
      } else if (draw_idx >= s.first + s.count) {
         lo = mid + 1;
      } else {
         return s.base_addr + (uint64_t)(draw_idx - s.first) * s.stride;
      }
   }

   return 0;
}

/*
 * Shared host/device implementation of runtime VDM stream truncation for
 * DrawIndirectCount. Restores any previously patched 8-byte VDM site first
 * (supporting reusable command buffers whose draw count changes across
 * submissions), then patches draw `*draw_count` with the non-returning
 * VDM_STREAM_LINK `(link_w0, link_w1)` whenever `*draw_count < max_draw_count`.
 */
static inline void
libagx_vdm_patch_indirect_count_impl(
   GLOBAL struct libagx_vdm_truncate_state *state,
   CONST struct libagx_vdm_draw_span *spans, CONST uint32_t *draw_count,
   uint32_t num_spans, uint32_t max_draw_count, uint32_t link_w0,
   uint32_t link_w1)
{
   uint64_t prev = state->patched_addr;
   if (prev != 0) {
      GLOBAL uint32_t *prev_dst = (GLOBAL uint32_t *)(uintptr_t)prev;
      prev_dst[0] = state->orig_w0;
      prev_dst[1] = state->orig_w1;
      state->patched_addr = 0;
   }

   uint32_t n = *draw_count;
   if (n >= max_draw_count)
      return;

   uint64_t addr = libagx_vdm_span_find_addr(spans, num_spans, n);
   if (addr == 0)
      return;

   GLOBAL uint32_t *dst = (GLOBAL uint32_t *)(uintptr_t)addr;
   state->orig_w0 = dst[0];
   state->orig_w1 = dst[1];
   state->patched_addr = addr;

   dst[0] = link_w0;
   dst[1] = link_w1;
}

#ifndef __OPENCL_VERSION__
/*
 * Runtime invariant check for VDM stream-truncation recording sites:
 * every recorded draw must remain in the same graphics control stream (`cs`)
 * and advance the stream by at least 8 bytes (AGX_VDM_STREAM_LINK_LENGTH)
 * unless a fresh chunk was opened for the draw.
 */
static inline bool
libagx_vdm_draw_site_valid(const void *cs_before, const void *cs_after,
                           uint64_t chunk_before, uint64_t chunk_after,
                           const void *cur_before, const void *cur_after)
{
   if (cs_before != cs_after)
      return false;

   if (chunk_before != chunk_after)
      return true;

   const uintptr_t before = (uintptr_t)cur_before;
   const uintptr_t after = (uintptr_t)cur_after;
   return after >= before && (after - before) >= 8u;
}
#endif

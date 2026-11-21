/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 direct VDM Index List representations.
 */

#pragma once

#ifndef __OPENCL_VERSION__
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_INDEX_LIST_DIRECT_NONINDEXED_MAX_LENGTH 20u
#define AGX_VDM_INDEX_LIST_DIRECT_INDEXED_MAX_LENGTH 32u

/* Indirect draws. The VDM record is CPU-written exactly like a direct draw;
 * only the count/instances/start live in a GPU-side argument buffer the VDM
 * dereferences at execution via the Index List "Indirect buffer" pointer.
 * G15/G16/G17P are 12 B (non-indexed) / 32 B (indexed); HAL300 (M5) appends an
 * Unk2=7 word to each, making them 16 B / 36 B. AGX2 is not a target here. */
#define AGX_VDM_INDEX_LIST_INDIRECT_NONINDEXED_MAX_LENGTH 16u
#define AGX_VDM_INDEX_LIST_INDIRECT_INDEXED_MAX_LENGTH 36u

enum agx_vdm_index_list_index_size {
   AGX_VDM_INDEX_LIST_INDEX_SIZE_U8 = 0,
   AGX_VDM_INDEX_LIST_INDEX_SIZE_U16 = 1,
   AGX_VDM_INDEX_LIST_INDEX_SIZE_U32 = 2,
};

struct agx_vdm_index_list_direct_nonindexed_info {
   uint8_t primitive;
   uint32_t count;
   uint32_t instances;
   uint32_t start;
};

struct agx_vdm_index_list_direct_indexed_info {
   uint8_t primitive;
   bool restart_enable;
   enum agx_vdm_index_list_index_size index_size;
   uint64_t index_buffer;
   uint32_t count;
   uint32_t instances;
   uint32_t start;

   /*
    * Preserved exactly as emitted after Apple's range computation.  Although
    * this word is dword-scaled, its complete byte-range semantics are not
    * independently grounded for every API edge case, so this helper does not
    * invent a byte-size conversion.
    */
   uint32_t index_buffer_size_word;
};

struct agx_vdm_index_list_direct_nonindexed {
   uint8_t bytes[AGX_VDM_INDEX_LIST_DIRECT_NONINDEXED_MAX_LENGTH];
   uint8_t length;
};

struct agx_vdm_index_list_direct_indexed {
   uint8_t bytes[AGX_VDM_INDEX_LIST_DIRECT_INDEXED_MAX_LENGTH];
   uint8_t length;
};

struct agx_vdm_index_list_indirect_nonindexed_info {
   uint8_t primitive;
   /* Native absolute argument-buffer VA. */
   uint64_t indirect_buffer;
};

struct agx_vdm_index_list_indirect_indexed_info {
   uint8_t primitive;
   bool restart_enable;
   enum agx_vdm_index_list_index_size index_size;
   uint64_t index_buffer;    /* index-buffer VA */
   /* Native absolute argument-buffer VA. */
   uint64_t indirect_buffer;
   uint32_t restart_index;   /* State Update restart value */
   uint32_t index_buffer_size_word; /* index-buffer size in dwords, minus 1 */
};

struct agx_vdm_index_list_indirect_nonindexed {
   uint8_t bytes[AGX_VDM_INDEX_LIST_INDIRECT_NONINDEXED_MAX_LENGTH];
   uint8_t length;
};

struct agx_vdm_index_list_indirect_indexed {
   uint8_t bytes[AGX_VDM_INDEX_LIST_INDIRECT_INDEXED_MAX_LENGTH];
   uint8_t length;
};

/*
 * Native command recording and the GPU-side DGC kernels must construct the
 * same words.  Keep the endian-neutral word representation inline so both
 * address spaces share the exact target gate and field layout; the host
 * wrappers below only serialize/deserialize those words.
 */
struct agx_vdm_index_list_direct_words {
   uint32_t words[AGX_VDM_INDEX_LIST_DIRECT_INDEXED_MAX_LENGTH / 4];
   uint8_t count;
};

/* Indirect records reach 9 words (36 B) on HAL300 indexed draws, one more than
 * the direct maximum, so they need their own wider carrier. */
struct agx_vdm_index_list_indirect_words {
   uint32_t words[AGX_VDM_INDEX_LIST_INDIRECT_INDEXED_MAX_LENGTH / 4];
   uint8_t count;
};

static inline bool
agx_vdm_index_list_direct_target_supported_inline(enum agx_chip chip)
{
   return agx_chip_has_record(chip, AGX_RECORD_VDM_INDEX_LIST);
}

static inline bool
agx_vdm_index_list_direct_is_hal300_inline(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
          chip == AGX_CHIP_G17C;
}

static inline bool
agx_vdm_index_list_index_size_supported_inline(enum agx_chip chip,
                                                uint8_t index_size)
{
   (void)chip;
   return index_size == AGX_VDM_INDEX_LIST_INDEX_SIZE_U8 ||
          index_size == AGX_VDM_INDEX_LIST_INDEX_SIZE_U16 ||
          index_size == AGX_VDM_INDEX_LIST_INDEX_SIZE_U32;
}

static inline bool
agx_vdm_index_list_direct_primitive_supported_inline(uint8_t primitive)
{
   switch (primitive) {
   case 0:
   case 1:
   case 3:
   case 5:
   case 6:
   case 9:
   case 11:
   case 13:
   case 14:
   case 15:
      return true;
   default:
      return false;
   }
}

static inline bool
agx_vdm_index_list_direct_pack_words(
   enum agx_chip chip, uint8_t primitive, bool indexed, bool restart_enable,
   uint8_t index_size, uint64_t index_buffer, uint32_t count,
   uint32_t instances, uint32_t start, uint32_t index_buffer_size_word,
   struct agx_vdm_index_list_direct_words *out)
{
   if (!out)
      return false;

   for (unsigned i = 0; i < AGX_VDM_INDEX_LIST_DIRECT_INDEXED_MAX_LENGTH / 4;
        ++i)
      out->words[i] = 0;
   out->count = 0;

   if (!agx_vdm_index_list_direct_target_supported_inline(chip) ||
       !agx_vdm_index_list_direct_primitive_supported_inline(primitive) ||
       instances == 0 ||
       (indexed && (!agx_vdm_index_list_index_size_supported_inline(
                       chip, index_size) ||
                    index_buffer >= ((uint64_t)1 << 48))))
      return false;

   const bool hal300 = agx_vdm_index_list_direct_is_hal300_inline(chip);
   if (!indexed) {
      out->words[0] =
         (hal300 ? 0x69c40000u : 0x61c40000u) | ((uint32_t)primitive << 8);
      out->words[1] = count;
      out->words[2] = instances;
      out->words[3] = start;
      if (hal300)
         out->words[4] = 7;
      out->count = hal300 ? 5 : 4;
      return true;
   }

   out->words[0] = (hal300 ? 0x69f00000u : 0x61f00000u) |
                   (uint32_t)((index_buffer >> 32) & (uint64_t)0xff) |
                   ((uint32_t)primitive << 8) | ((uint32_t)index_size << 17) |
                   (restart_enable ? 1u << 16 : 0);
   out->words[1] = (uint32_t)index_buffer;
   out->words[2] = count;
   out->words[3] = instances;
   out->words[4] = start;
   unsigned cursor = 5;
   if (hal300)
      out->words[cursor++] = 7;
   out->words[cursor++] = index_buffer_size_word;
   out->words[cursor++] = (uint32_t)((index_buffer >> 40) & (uint64_t)0xff);
   out->count = (uint8_t)cursor;
   return true;
}

/*
 * Indirect Index List words, mirroring agx_vdm_index_list_direct_pack_words.
 * `indirect_buffer` (A) is the argument-buffer VA; for the indexed form,
 * `index_buffer` (IB) is the index-buffer VA, `restart_index` seeds the
 * prepended 8-byte State Update, and `index_buffer_size_word` is the
 * index-buffer size in dwords minus one. The indexed record is laid out with
 * the State Update first, so word 0 here is 0x40000001.
 *
 * The indexed header is byte-identical to Apple's observed
 *   (table[prim] + (apple_index_size << 17)) & 0xffeeff00 | IB[39:32]
 *   | (restart ? bit16) | bit20
 * with table = 0x6422/0x6c22; that reduces to the additive field build below
 * because the encoded index-size field equals this enum's value (U16=1, U32=2),
 * exactly as the already-validated direct indexed packer does it.
 */
static inline bool
agx_vdm_index_list_indirect_pack_words(
   enum agx_chip chip, uint8_t primitive, bool indexed, bool restart_enable,
   uint8_t index_size, uint64_t index_buffer, uint64_t indirect_buffer,
   uint32_t restart_index, uint32_t index_buffer_size_word,
   struct agx_vdm_index_list_indirect_words *out)
{
   if (!out)
      return false;

   for (unsigned i = 0; i < AGX_VDM_INDEX_LIST_INDIRECT_INDEXED_MAX_LENGTH / 4;
        ++i)
      out->words[i] = 0;
   out->count = 0;

   if (!agx_vdm_index_list_direct_target_supported_inline(chip) ||
       !agx_vdm_index_list_direct_primitive_supported_inline(primitive) ||
       indirect_buffer >= ((uint64_t)1 << 48) || (indirect_buffer & 3u) != 0 ||
       (indexed && (!agx_vdm_index_list_index_size_supported_inline(
                       chip, index_size) ||
                    index_buffer >= ((uint64_t)1 << 48))))
      return false;

   const bool hal300 = agx_vdm_index_list_direct_is_hal300_inline(chip);
   const uint32_t addr_hi = (uint32_t)((indirect_buffer >> 32) & 0xffffu);
   const uint32_t addr_lo = (uint32_t)indirect_buffer & 0xfffffffcu;

   if (!indexed) {
      out->words[0] =
         (hal300 ? 0x6c040000u : 0x64040000u) | ((uint32_t)primitive << 8);
      out->words[1] = addr_hi;
      out->words[2] = addr_lo;
      if (hal300)
         out->words[3] = 7u;
      out->count = hal300 ? 4u : 3u;
      return true;
   }

   out->words[0] = 0x40000001u;
   out->words[1] = restart_index;
   out->words[2] = (uint32_t)(3u << 29) | (hal300 ? (1u << 27) : 0u) |
                   (1u << 26) | (1u << 21) | (1u << 20) |
                   ((uint32_t)(index_size & 7u) << 17) |
                   (restart_enable ? (1u << 16) : 0u) |
                   ((uint32_t)primitive << 8) |
                   (uint32_t)((index_buffer >> 32) & 0xffu);
   out->words[3] = (uint32_t)index_buffer;
   out->words[4] = addr_hi;
   out->words[5] = addr_lo;
   unsigned cursor = 6;
   if (hal300)
      out->words[cursor++] = 7u;
   out->words[cursor++] = index_buffer_size_word;
   out->words[cursor++] = (uint32_t)((index_buffer >> 40) & 0xffu);
   out->count = (uint8_t)cursor;
   return true;
}

/*
 * G18G (M6) Index Lists. G18G writes the HAL300 records
 * (agx_chip_hal300_records): its Index List words are G17G's above, byte for
 * byte, except the HAL300 Unk 2 word, which G18G takes from the pipeline
 * (agx_vdm_index_list_hal300_unk2) instead of the constant 7. The helpers
 * above stay as they are for every other chip; G18G callers use these.
 */

/* HAL300's extra Index List word ("Unk 2", gated by header bit 27). It is a
 * per-pipeline value taken from word 4 of the UVS record {0x0c280000, ...}:
 * n = byte 0 + byte 2, the vertex output count (the same on G18G and G17G).
 * Every J873 capture has n <= 9 and 7. Only G18G takes the derived value; the
 * other HAL300 chips keep the constant their captures show. */
static inline uint32_t
agx_vdm_index_list_hal300_unk2(enum agx_chip chip, uint32_t uvs_word4)
{
   const uint32_t n = (uvs_word4 & 0xffu) + ((uvs_word4 >> 16) & 0xffu);

   if (chip != AGX_CHIP_G18G)
      return 7u;

   return n <= 9    ? 7u
          : n <= 11 ? 6u
          : n == 12 ? 5u
          : n <= 15 ? 4u
          : n <= 19 ? 3u
          : n <= 23 ? 2u
                    : 0u;
}

/* The chip whose Index List words a chip writes: G18G writes G17G's. */
static inline enum agx_chip
agx_vdm_index_list_record_chip(enum agx_chip chip)
{
   return chip == AGX_CHIP_G18G ? AGX_CHIP_G17G : chip;
}

/* Direct, indexed and indirect Index Lists, G18G included.
 *
 * Sessions 20260925-a and -b of the J873 native captures (26A428) contain
 * only direct non-indexed draws; sessions -g1 and -g2 add direct indexed ones
 * (indexed-u16, indexed-strip, quad-strip, gears). Those are Mesa's HAL300
 * direct indexed words byte for byte, e.g. 69f20600 00018300 3 1 0 7 0000000f
 * 1 for three u16 triangles and 69f30900 ... for a restarted u16 strip.
 *
 * No indirect draw is captured; the indirect forms below, non-indexed and
 * indexed with its State Update, follow the same header layout as the
 * direct ones, on G18G and G17G alike. */
static inline bool
agx_vdm_index_list_indexed_or_indirect_supported_inline(enum agx_chip chip)
{
   return agx_vdm_index_list_direct_target_supported_inline(
      agx_vdm_index_list_record_chip(chip));
}

static inline bool
agx_vdm_index_list_nonindexed_has_tail(enum agx_chip chip)
{
   /* HAL300 non-indexed draws carry the fifth (Unk 2) word, the G17G family
    * included.
    *
    * G18G is witnessed with the fifth word: every J873 native render draws
    * with 0x69c40000 | primitive << 8, count, instances, start and 7, e.g.
    * 69c40600 3 1 0 7 for the colour triangle and 69c40100 6 1 6 7 for
    * lines. */
   return agx_vdm_index_list_direct_is_hal300_inline(
      agx_vdm_index_list_record_chip(chip));
}

/* agx_vdm_index_list_direct_pack_words, with the HAL300 Unk 2 word given. */
static inline bool
agx_vdm_index_list_direct_pack_words_unk2(
   enum agx_chip chip, uint8_t primitive, bool indexed, bool restart_enable,
   uint8_t index_size, uint64_t index_buffer, uint32_t count,
   uint32_t instances, uint32_t start, uint32_t index_buffer_size_word,
   uint32_t unk2, struct agx_vdm_index_list_direct_words *out)
{
   const enum agx_chip record_chip = agx_vdm_index_list_record_chip(chip);
   if (!agx_vdm_index_list_direct_pack_words(
          record_chip, primitive, indexed, restart_enable, index_size,
          index_buffer, count, instances, start, index_buffer_size_word, out))
      return false;

   if (agx_vdm_index_list_direct_is_hal300_inline(record_chip))
      out->words[indexed ? 5 : 4] = unk2;
   return true;
}

/* agx_vdm_index_list_indirect_pack_words, with the HAL300 Unk 2 word given. */
static inline bool
agx_vdm_index_list_indirect_pack_words_unk2(
   enum agx_chip chip, uint8_t primitive, bool indexed, bool restart_enable,
   uint8_t index_size, uint64_t index_buffer, uint64_t indirect_buffer,
   uint32_t restart_index, uint32_t index_buffer_size_word, uint32_t unk2,
   struct agx_vdm_index_list_indirect_words *out)
{
   const enum agx_chip record_chip = agx_vdm_index_list_record_chip(chip);
   if (!agx_vdm_index_list_indirect_pack_words(
          record_chip, primitive, indexed, restart_enable, index_size,
          index_buffer, indirect_buffer, restart_index,
          index_buffer_size_word, out))
      return false;

   if (agx_vdm_index_list_direct_is_hal300_inline(record_chip))
      out->words[indexed ? 6 : 3] = unk2;
   return true;
}

bool agx_vdm_index_list_direct_nonindexed_pack(
   enum agx_chip chip,
   const struct agx_vdm_index_list_direct_nonindexed_info *info,
   struct agx_vdm_index_list_direct_nonindexed *out);

bool agx_vdm_index_list_direct_nonindexed_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_index_list_direct_nonindexed_info *info);

bool agx_vdm_index_list_direct_indexed_pack(
   enum agx_chip chip,
   const struct agx_vdm_index_list_direct_indexed_info *info,
   struct agx_vdm_index_list_direct_indexed *out);

bool agx_vdm_index_list_direct_indexed_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_index_list_direct_indexed_info *info);

bool agx_vdm_index_list_indirect_nonindexed_pack(
   enum agx_chip chip,
   const struct agx_vdm_index_list_indirect_nonindexed_info *info,
   struct agx_vdm_index_list_indirect_nonindexed *out);

bool agx_vdm_index_list_indirect_nonindexed_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_index_list_indirect_nonindexed_info *info);

bool agx_vdm_index_list_indirect_indexed_pack(
   enum agx_chip chip,
   const struct agx_vdm_index_list_indirect_indexed_info *info,
   struct agx_vdm_index_list_indirect_indexed *out);

bool agx_vdm_index_list_indirect_indexed_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_index_list_indirect_indexed_info *info);

#ifdef __cplusplus
}
#endif

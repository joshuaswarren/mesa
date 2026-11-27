/*
 * Copyright 2026 Asahi Linux contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "asahi/compiler/agx3_ioto_uniform_program.h"
#include "asahi/lib/agx_ioto_load_core.h"
#include "util/macros.h"
#include "util/u_math.h"

/*
 * A G17P loader program is composed for one set of uniform range addresses.
 * Each range is loaded through a base materialized by an LDIMM pair (a
 * relocation); a range within INT16_MAX bytes of a lower one may instead
 * reuse that range's base with an offset. The program stays valid for other
 * addresses as long as every range sharing a base moves with it and each
 * base's LDIMMs keep their size: then only the base immediates change.
 *
 * Ranges are grouped into clusters of addresses less than INT16_MAX apart
 * (with a margin for load chunking); any two ranges that may share a base are
 * in one cluster. A plan fits new addresses when every range moves by the
 * same amount as its cluster and no 2-byte LDIMM needs a value of 128 or
 * more. A program composed with independent ranges (each range its own base,
 * every LDIMM 8 bytes) fits any addresses.
 */

/* HK_AGX3_G17P_GRAPHICS_MAX_RANGES: a graphics stage with all its launch
 * copies (hk_g17p_graphics_uniform_ranges()). */
#define HK_G17P_PATCH_MAX_RANGES  19
#define HK_G17P_PATCH_MAX_RELOCS  32
#define HK_G17P_PATCH_CLUSTER_GAP ((uint64_t)INT16_MAX + 64)

struct hk_g17p_patch_reloc {
   /* agx_ioto_load_ldimm_long_dest() of the low and high halves */
   uint64_t dest[2];
   uint32_t delta;
   uint16_t offset;
   uint8_t range;
   /* bit 0: the low half's LDIMM is the 2-byte form, bit 1: the high half's */
   uint8_t short_mask;
};

struct hk_g17p_patch_plan {
   uint64_t address[HK_G17P_PATCH_MAX_RANGES];
   uint8_t leader[HK_G17P_PATCH_MAX_RANGES];
   uint8_t range_count;
   uint8_t reloc_count;
   /* Whether any range shares a cluster with another, and the relocations
    * with a 2-byte LDIMM: what hk_g17p_patch_plan_fits() has to check. */
   bool shared;
   uint8_t short_count;
   uint8_t short_relocs[HK_G17P_PATCH_MAX_RELOCS];
   struct hk_g17p_patch_reloc relocs[HK_G17P_PATCH_MAX_RELOCS];
};

static inline bool
hk_g17p_patch_plan_init(struct hk_g17p_patch_plan *plan,
                        const struct agx3_ioto_uniform_range *ranges,
                        unsigned range_count,
                        const struct agx_ioto_load_relocation *relocs,
                        unsigned reloc_count, size_t length, bool independent)
{
   if (range_count > HK_G17P_PATCH_MAX_RANGES ||
       reloc_count > HK_G17P_PATCH_MAX_RELOCS)
      return false;

   plan->range_count = range_count;
   plan->reloc_count = reloc_count;

   unsigned order[HK_G17P_PATCH_MAX_RANGES];
   for (unsigned i = 0; i < range_count; ++i) {
      plan->address[i] = ranges[i].address;
      unsigned j = i;
      while (j && ranges[order[j - 1]].address > ranges[i].address) {
         order[j] = order[j - 1];
         --j;
      }
      order[j] = i;
   }

   uint64_t end = 0;
   unsigned leader = 0;
   plan->shared = false;
   for (unsigned k = 0; k < range_count; ++k) {
      const unsigned i = order[k];
      const uint64_t start = ranges[i].address;
      if (k == 0 || independent || start > end + HK_G17P_PATCH_CLUSTER_GAP) {
         leader = i;
         end = 0;
      }
      plan->leader[i] = leader;
      plan->shared |= leader != i;
      end = MAX2(end, start + (uint64_t)ranges[i].size_halfs * 2);
   }

   plan->short_count = 0;
   for (unsigned r = 0; r < reloc_count; ++r) {
      const struct agx_ioto_load_relocation *rel = &relocs[r];
      unsigned i = 0;
      while (i < range_count &&
             (rel->address < ranges[i].address ||
              rel->address - ranges[i].address >=
                 (uint64_t)ranges[i].size_halfs * 2))
         ++i;
      if (i == range_count ||
          (rel->low_length != 2 && rel->low_length != 8) ||
          (rel->high_length != 2 && rel->high_length != 8) ||
          rel->offset + rel->low_length + rel->high_length > length)
         return false;

      plan->relocs[r] = (struct hk_g17p_patch_reloc){
         .dest = {agx_ioto_load_ldimm_long_dest(rel->dest_half),
                  agx_ioto_load_ldimm_long_dest(rel->dest_half + 2)},
         .delta = rel->address - ranges[i].address,
         .offset = rel->offset,
         .range = i,
         .short_mask = (rel->low_length == 2 ? 1 : 0) |
                       (rel->high_length == 2 ? 2 : 0),
      };
      if (plan->relocs[r].short_mask)
         plan->short_relocs[plan->short_count++] = r;
   }
   return true;
}

static inline bool
hk_g17p_patch_plan_fits(const struct hk_g17p_patch_plan *plan,
                        const struct agx3_ioto_uniform_range *ranges)
{
   for (unsigned i = 0; plan->shared && i < plan->range_count; ++i) {
      const unsigned l = plan->leader[i];
      if (ranges[i].address - plan->address[i] !=
          ranges[l].address - plan->address[l])
         return false;
   }

   for (unsigned k = 0; k < plan->short_count; ++k) {
      const struct hk_g17p_patch_reloc *rel =
         &plan->relocs[plan->short_relocs[k]];
      const uint64_t base = ranges[rel->range].address + rel->delta;
      if (((rel->short_mask & 1) && (uint32_t)base >= 128) ||
          ((rel->short_mask & 2) && (base >> 32) >= 128))
         return false;
   }
   return true;
}

static inline void
hk_g17p_patch_ldimm(uint8_t *out, bool is_short, uint64_t dest,
                    uint32_t immediate)
{
   if (is_short) {
      out[1] = (uint8_t)immediate;
   } else {
      const uint64_t word =
         util_cpu_to_le64(dest | agx_ioto_load_ldimm_long_immediate(immediate));
      memcpy(out, &word, sizeof(word));
   }
}

static inline void
hk_g17p_patch_plan_apply(const struct hk_g17p_patch_plan *plan,
                         const struct agx3_ioto_uniform_range *ranges,
                         uint8_t *bytes)
{
   for (unsigned r = 0; r < plan->reloc_count; ++r) {
      const struct hk_g17p_patch_reloc *rel = &plan->relocs[r];
      const uint64_t base = ranges[rel->range].address + rel->delta;
      uint8_t *out = bytes + rel->offset;
      const bool low_short = rel->short_mask & 1;
      hk_g17p_patch_ldimm(out, low_short, rel->dest[0], (uint32_t)base);
      hk_g17p_patch_ldimm(out + (low_short ? 2 : 8), rel->short_mask & 2,
                          rel->dest[1], (uint32_t)(base >> 32));
   }
}

/* The LDIMMs plan patches, a low then a high half for each relocation: their
 * offsets in the program it was made from, and the immediates they take for
 * ranges (either may be NULL). Returns their count. */
static inline unsigned
hk_g17p_patch_plan_ldimms(const struct hk_g17p_patch_plan *plan,
                          const struct agx3_ioto_uniform_range *ranges,
                          unsigned *offsets, uint32_t *immediates)
{
   for (unsigned r = 0; r < plan->reloc_count; ++r) {
      const struct hk_g17p_patch_reloc *rel = &plan->relocs[r];
      if (offsets) {
         offsets[2 * r] = rel->offset;
         offsets[2 * r + 1] = rel->offset + ((rel->short_mask & 1) ? 2 : 8);
      }
      if (immediates) {
         const uint64_t base = ranges[rel->range].address + rel->delta;
         immediates[2 * r] = (uint32_t)base;
         immediates[2 * r + 1] = (uint32_t)(base >> 32);
      }
   }
   return 2 * plan->reloc_count;
}

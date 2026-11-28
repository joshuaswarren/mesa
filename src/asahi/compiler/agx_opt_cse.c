/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2021 Collabora, Ltd.
 * Copyright 2014 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "util/compiler.h"
#include "agx_builder.h"

/*
 * This pass handles CSE'ing repeated expressions created in the process of
 * translating from NIR. Also, currently this is intra-block only, to make it
 * work over multiple block we'd need to bring forward dominance calculation.
 */

static inline uint64_t
hash_mix64(uint64_t h, uint64_t x)
{
   h ^= x * 0x9e3779b97f4a7c15ull;
   return (h ^ (h >> 27)) * 0xbf58476d1ce4e5b9ull;
}

static inline uint32_t
hash_finalize(uint64_t h)
{
   h ^= h >> 31;
   h *= 0x94d049bb133111ebull;
   return (uint32_t)(h ^ (h >> 32));
}

/* Hash an instruction using inline 64-bit multiply-mix. */
static uint32_t
hash_instr(const void *data)
{
   const agx_instr *I = data;
   uint64_t h = ((uint64_t)I->nr_dests << 8) | I->nr_srcs;

   /* Explicitly skip destinations, except for size and type */
   agx_foreach_dest(I, d) {
      h = hash_mix64(h, ((uint64_t)I->dest[d].type) |
                        (((uint64_t)I->dest[d].size) << 16));
   }

   /* Hash the source array as-is */
   STATIC_ASSERT(sizeof(agx_index) == 8);
   for (unsigned s = 0; s < I->nr_srcs; ++s) {
      uint64_t w;
      memcpy(&w, &I->src[s], sizeof(w));
      h = hash_mix64(h, w);
   }

   /* Hash everything else in the instruction starting from the opcode */
   const uint8_t *p = (const uint8_t *)&I->op;
   const size_t tail_bytes = sizeof(agx_instr) - offsetof(agx_instr, op);
   size_t off = 0;
   for (; off + 8 <= tail_bytes; off += 8) {
      uint64_t w;
      memcpy(&w, p + off, 8);
      h = hash_mix64(h, w);
   }
   if (off < tail_bytes) {
      uint64_t w = 0;
      memcpy(&w, p + off, tail_bytes - off);
      h = hash_mix64(h, w);
   }

   return hash_finalize(h);
}

static bool
instrs_equal(const void *_i1, const void *_i2)
{
   const agx_instr *i1 = _i1, *i2 = _i2;

   if (i1->op != i2->op)
      return false;
   if (i1->nr_srcs != i2->nr_srcs)
      return false;
   if (i1->nr_dests != i2->nr_dests)
      return false;

   /* Explicitly skip everything but size, channels, and type */
   agx_foreach_dest(i1, d) {
      if (i1->dest[d].type != i2->dest[d].type)
         return false;
      if (i1->dest[d].size != i2->dest[d].size)
         return false;
      if (i1->dest[d].channels_m1 != i2->dest[d].channels_m1)
         return false;
   }

   agx_foreach_src(i1, s) {
      agx_index s1 = i1->src[s], s2 = i2->src[s];

      if (memcmp(&s1, &s2, sizeof(s1)) != 0)
         return false;
   }

   if (i1->imm != i2->imm)
      return false;
   if (i1->invert_cond != i2->invert_cond)
      return false;
   if (i1->dim != i2->dim)
      return false;
   if (i1->agx3_cube_scalar_coords != i2->agx3_cube_scalar_coords)
      return false;
   if (i1->agx3_split_coords != i2->agx3_split_coords)
      return false;
   if (i1->offset != i2->offset)
      return false;
   /* Two samples that differ only in their coordinate offset are different
    * samples. This field is not in the union `imm` covers, so it needs its
    * own line here. */
   if (i1->offset_xyz != i2->offset_xyz)
      return false;
   if (i1->pixel_offset != i2->pixel_offset)
      return false;
   if (i1->colour_target != i2->colour_target)
      return false;
   if (i1->sample_groups != i2->sample_groups)
      return false;
   if (i1->shadow != i2->shadow)
      return false;
   if (i1->interpolation != i2->interpolation)
      return false;
   if (i1->perspective != i2->perspective)
      return false;
   if (i1->shift != i2->shift)
      return false;
   if (i1->memory_is_packed != i2->memory_is_packed)
      return false;
   if (i1->memory_index_is_bytes != i2->memory_index_is_bytes)
      return false;
   if (i1->no_signed_zero != i2->no_signed_zero)
      return false;
   if (i1->fcmpsel_minifloat_select != i2->fcmpsel_minifloat_select)
      return false;
   if (i1->memory_byte_offset != i2->memory_byte_offset)
      return false;
   if (i1->saturate != i2->saturate)
      return false;
   if (i1->mask != i2->mask)
      return false;
   if (i1->format != i2->format)
      return false;

   return true;
}

/* Determines what instructions the above routines have to handle */
static bool
instr_can_cse(const agx_instr *I)
{
   return agx_opcodes_info[I->op].can_eliminate &&
          agx_opcodes_info[I->op].can_reorder;
}

enum cse_mode {
   /* Merge every repeated expression of a block */
   CSE_ALL,

   /* Merge only while the earlier result is still live: no range grows */
   CSE_OVERLAP,

   /* Also extend the earlier result's range where the register pressure
    * stays within the shader's existing peak */
   CSE_BOUNDED,
};

/*
 * Register pressure at every instruction of a block, in the halves RA's
 * demand charges (vectors round up to a power of two): what is live into the
 * instruction together with what it defines. Positions count from 1 like the
 * CSE walk. Returns the block's peak.
 */
static unsigned
cse_block_pressure(agx_context *ctx, agx_block *block, unsigned position,
                   const uint8_t *sizes, BITSET_WORD *live, unsigned *pressure)
{
   unsigned peak = 0, count = 0;

   memset(live, 0, BITSET_BYTES(ctx->alloc));
   U_SPARSE_BITSET_FOREACH_SET(&block->live_out, v) {
      BITSET_SET(live, v);
      count += sizes[v];
   }

   agx_foreach_instr_in_block_rev(block, I) {
      unsigned here = count;

      agx_foreach_ssa_dest(I, d) {
         const unsigned v = I->dest[d].value;
         if (BITSET_TEST(live, v)) {
            BITSET_CLEAR(live, v);
            count -= sizes[v];
         } else {
            here += sizes[v];
         }
      }

      if (I->op != AGX_OPCODE_PHI) {
         agx_foreach_ssa_src(I, s) {
            const unsigned v = I->src[s].value;
            if (!BITSET_TEST(live, v)) {
               BITSET_SET(live, v);
               count += sizes[v];
               here += sizes[v];
            }
         }
      }

      pressure[position--] = here;
      peak = MAX2(peak, here);
   }

   return peak;
}

static void
agx_opt_cse_impl(agx_context *ctx, enum cse_mode mode)
{
   const bool require_overlap = mode != CSE_ALL;
   struct set *instr_set = _mesa_set_create(NULL, hash_instr, instrs_equal);
   agx_index *replacement = calloc(ctx->alloc, sizeof(*replacement));
   unsigned replacement_start = ctx->alloc;
   unsigned replacement_end = 0;
   unsigned *last = NULL;
   unsigned last_start = ctx->alloc;
   unsigned last_end = 0;
   if (require_overlap) {
      last = calloc(ctx->alloc, sizeof(*last));
      agx_compute_liveness(ctx);
      agx_foreach_instr_global(ctx, I) {
         agx_foreach_ssa_src(I, s)
            I->src[s].kill = false;
      }
   }

   uint8_t *sizes = NULL;
   unsigned *all_pressure = NULL;
   unsigned *block_offsets = NULL;
   BITSET_DECLARE(stack_live, 4096);
   BITSET_WORD *live = NULL;
   unsigned peak = 0;
   if (mode == CSE_BOUNDED) {
      sizes = calloc(ctx->alloc, sizeof(*sizes));
      live = ctx->alloc <= 4096 ? stack_live : BITSET_CALLOC(ctx->alloc);
      block_offsets = malloc((ctx->num_blocks + 1) * sizeof(*block_offsets));
      unsigned total = 0;
      unsigned b = 0;
      agx_foreach_block(ctx, block) {
         unsigned n = 0;
         agx_foreach_instr_in_block(block, I) {
            n++;
            agx_foreach_ssa_dest(I, d) {
               sizes[I->dest[d].value] =
                  util_next_power_of_two(agx_index_size_16(I->dest[d]));
            }
         }
         block_offsets[b++] = total;
         total += n + 1;
      }
      block_offsets[b] = total;

      all_pressure = malloc(MAX2(total, 1u) * sizeof(*all_pressure));
      b = 0;
      agx_foreach_block(ctx, block) {
         const unsigned off = block_offsets[b];
         const unsigned len = block_offsets[b + 1] - off - 1;
         all_pressure[off] = 0;
         peak = MAX2(peak, cse_block_pressure(ctx, block, len, sizes, live,
                                              all_pressure + off));
         b++;
      }
   }

   unsigned block_idx = 0;
   agx_foreach_block(ctx, block) {
      if (replacement_start < replacement_end) {
         memset(replacement + replacement_start, 0,
                sizeof(*replacement) * (replacement_end - replacement_start));
      }
      replacement_start = ctx->alloc;
      replacement_end = 0;
      _mesa_set_clear(instr_set, NULL);
      unsigned *pressure =
         mode == CSE_BOUNDED ? all_pressure + block_offsets[block_idx] : NULL;
      block_idx++;

      bool has_live_out = false;
      if (require_overlap) {
         if (last_start < last_end) {
            memset(last + last_start, 0,
                   sizeof(*last) * (last_end - last_start));
         }
         last_start = ctx->alloc;
         last_end = 0;
         unsigned position = 0;
         agx_foreach_instr_in_block(block, I) {
            ++position;
            if (I->op != AGX_OPCODE_PHI) {
               agx_foreach_ssa_src(I, s) {
                  const unsigned v = I->src[s].value;
                  last[v] = position;
                  last_start = MIN2(last_start, v);
                  last_end = MAX2(last_end, v + 1);
               }
            }
         }
         U_SPARSE_BITSET_FOREACH_SET(&block->live_out, v) {
            has_live_out = true;
            last[v] = position;
            last_start = MIN2(last_start, v);
            last_end = MAX2(last_end, v + 1);
         }
      }

      unsigned position = 0;
      agx_foreach_instr_in_block(block, instr) {
         ++position;
         agx_foreach_ssa_src(instr, s) {
            agx_index repl = replacement[instr->src[s].value];
            if (!agx_is_null(repl))
               agx_replace_src(instr, s, repl);
         }
         if (!instr_can_cse(instr))
            continue;

         bool found;
         struct set_entry *entry =
            _mesa_set_search_or_add(instr_set, instr, &found);
         if (found) {
            const agx_instr *match = entry->key;
            if (require_overlap) {
               bool safe = true;
               agx_foreach_ssa_dest(instr, d) {
                  const unsigned a = match->dest[d].value;
                  const unsigned b = instr->dest[d].value;
                  if (has_live_out)
                     safe &= !u_sparse_bitset_test(&block->live_out, b);

                  /* Without overlap, the earlier result becomes live from its
                   * last use up to here as well. */
                  if (last[b] >= position && last[a] < position) {
                     for (unsigned p = last[a] + 1;
                          safe && mode == CSE_BOUNDED && p < position; ++p)
                        safe = pressure[p] + sizes[a] <= peak;

                     safe &= mode == CSE_BOUNDED;
                  }
               }
               if (!safe) {
                  entry->key = instr;
                  continue;
               }
               agx_foreach_ssa_dest(instr, d) {
                  const unsigned a = match->dest[d].value;
                  const unsigned b = instr->dest[d].value;
                  if (mode == CSE_BOUNDED && last[b] >= position) {
                     for (unsigned p = last[a] + 1; p < position; ++p)
                        pressure[p] += sizes[a];
                  }
                  if (last[b] > last[a]) {
                     last[a] = last[b];
                     last_start = MIN2(last_start, a);
                     last_end = MAX2(last_end, a + 1);
                  }
               }
            }
            agx_foreach_dest(instr, d) {
               const unsigned dest = instr->dest[d].value;
               replacement[dest] = match->dest[d];
               replacement_start = MIN2(replacement_start, dest);
               replacement_end = MAX2(replacement_end, dest + 1);
            }
         }
      }
   }

   free(last);
   free(sizes);
   free(all_pressure);
   free(block_offsets);
   if (live && live != stack_live)
      free(live);
   free(replacement);
   _mesa_set_destroy(instr_set, NULL);
}

void
agx_opt_cse(agx_context *ctx)
{
   agx_opt_cse_impl(ctx, CSE_ALL);
}

/* Like agx_opt_cse_pressure, but a merge may also extend the earlier result's
 * live range as long as no instruction it spans then exceeds the register
 * pressure the shader already peaks at. */
void
agx_opt_cse_bounded(agx_context *ctx)
{
   agx_opt_cse_impl(ctx, CSE_BOUNDED);
}

void
agx_opt_cse_pressure(agx_context *ctx)
{
   agx_opt_cse_impl(ctx, CSE_OVERLAP);
}

/*
 * Exposed for testing. instrs_equal() enumerates the instruction's immediates
 * one at a time, so a field added to agx_instr without a matching line here
 * lets two otherwise-identical instructions collapse into one --
 * test-agx3-pack-binary.cpp guards I->offset_xyz against exactly that.
 */
bool
agx_instrs_equal_for_cse(const agx_instr *i1, const agx_instr *i2)
{
   return instrs_equal(i1, i2);
}

/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "asahi/lib/agx_chip_caps.h"

#define AGX_MAX_PENDING (8)

/*
 * Returns whether an instruction is asynchronous and needs a scoreboard slot
 */
static bool
instr_is_async(agx_instr *I)
{
   return agx_opcodes_info[I->op].immediates & AGX_IMMEDIATE_SCOREBOARD;
}

struct slot {
   /* Set of registers this slot is currently writing */
   BITSET_DECLARE(writes, AGX_NUM_REGS);

   /* Number of pending messages on this slot. Must not exceed
    * AGX_MAX_PENDING for correct results.
    */
   uint8_t nr_pending;

   /* Whether a threadgroup barrier must wait for this slot. Read-only
    * non-aliasing device loads write only private registers and cannot race
    * with threadgroup memory, so they may stay in flight across a
    * threadgroup_barrier until their destination registers are consumed (or a
    * memory_barrier/block exit is reached).
    */
   bool drain_on_tg_barrier;
};

/*
 * Insert waits within a block to stall after every async instruction. Useful
 * for debugging.
 */
static void
agx_insert_waits_trivial(agx_context *ctx, agx_block *block)
{
   agx_foreach_instr_in_block_safe(block, I) {
      if (instr_is_async(I)) {
         agx_builder b = agx_init_builder(ctx, agx_after_instr(I));
         agx_wait(&b, I->scoreboard);
      }
   }
}

/*
 * Insert waits within a block, assuming scoreboard slots have already been
 * assigned. This waits for everything at the end of the block, rather than
 * doing something more intelligent/global. This should be optimized.
 *
 * XXX: Do any instructions read their sources asynchronously?
 */
/* Precompute the first-use (or next-draining-barrier) instruction index for
 * every instruction in the block in a single O(N) backward pass so smart-wait
 * slot assignment and its 16-instruction lookahead run in linear time even on
 * long blocks with many independent async loads.
 */
static unsigned *
compute_first_use_positions(agx_block *block, unsigned *count_out)
{
   unsigned n = 0;
   agx_foreach_instr_in_block(block, I)
      n++;

   *count_out = n;
   if (!n)
      return NULL;

   unsigned *first_use = malloc(n * sizeof(unsigned));
   if (!first_use)
      return NULL;

   unsigned next_use[AGX_NUM_REGS];
   for (unsigned r = 0; r < AGX_NUM_REGS; ++r)
      next_use[r] = n;
   unsigned next_mem_barrier = n;
   unsigned next_tg_barrier = n;

   unsigned pos = n;
   agx_foreach_instr_in_block_rev(block, I) {
      pos--;
      if (I->op == AGX_OPCODE_MEMORY_BARRIER) {
         first_use[pos] = pos;
         next_mem_barrier = pos;
         next_tg_barrier = pos;
         for (unsigned r = 0; r < AGX_NUM_REGS; ++r)
            next_use[r] = pos;
         continue;
      }
      if (I->op == AGX_OPCODE_THREADGROUP_BARRIER) {
         first_use[pos] = pos;
         next_tg_barrier = pos;
         continue;
      }

      if (instr_is_async(I)) {
         bool crosses_tg =
            (I->op == AGX_OPCODE_DEVICE_LOAD && I->memory_can_reorder);
         unsigned u = crosses_tg ? next_mem_barrier
                                 : MIN2(next_mem_barrier, next_tg_barrier);
         agx_foreach_dest(I, d) {
            if (agx_is_null(I->dest[d]) ||
                I->dest[d].type != AGX_INDEX_REGISTER)
               continue;
            unsigned end = I->dest[d].value + agx_index_size_16(I->dest[d]);
            for (unsigned r = I->dest[d].value; r < end && r < AGX_NUM_REGS; ++r) {
               if (next_use[r] < u)
                  u = next_use[r];
            }
         }
         first_use[pos] = u;
      } else {
         first_use[pos] = n;
      }

      agx_foreach_src(I, s) {
         if (I->src[s].type != AGX_INDEX_REGISTER)
            continue;
         unsigned end = I->src[s].value + agx_index_size_16(I->src[s]);
         for (unsigned r = I->src[s].value; r < end && r < AGX_NUM_REGS; ++r)
            next_use[r] = pos;
      }
      agx_foreach_dest(I, d) {
         if (I->dest[d].type != AGX_INDEX_REGISTER)
            continue;
         unsigned end = I->dest[d].value + agx_index_size_16(I->dest[d]);
         for (unsigned r = I->dest[d].value; r < end && r < AGX_NUM_REGS; ++r)
            next_use[r] = pos;
      }
   }

   return first_use;
}

static void
agx_insert_waits_local(agx_context *ctx, agx_block *block)
{
   struct slot slots[2] = {0};
   unsigned slot_min_use[2] = {UINT_MAX, UINT_MAX};
   const bool smart_waits =
      ctx->stage == MESA_SHADER_COMPUTE && ctx->key &&
      agx_chip_has_cap(ctx->key->dev.chip, AGX_CAP_COMPUTE_SMART_WAITS);
   unsigned block_len = 0;
   unsigned *first_use =
      smart_waits ? compute_first_use_positions(block, &block_len) : NULL;
   unsigned pos = 0;

   agx_foreach_instr_in_block_safe(block, I) {
      uint8_t wait_mask = 0;

      /* Check for read-after-write */
      agx_foreach_src(I, s) {
         if (I->src[s].type != AGX_INDEX_REGISTER)
            continue;

         unsigned nr_read = agx_index_size_16(I->src[s]);
         for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
            if (BITSET_TEST_COUNT(slots[slot].writes, I->src[s].value, nr_read))
               wait_mask |= BITSET_BIT(slot);
         }
      }

      /* Check for write-after-write */
      agx_foreach_dest(I, d) {
         if (I->dest[d].type != AGX_INDEX_REGISTER)
            continue;

         unsigned nr_writes = agx_index_size_16(I->dest[d]);
         for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
            if (BITSET_TEST_COUNT(slots[slot].writes, I->dest[d].value,
                                  nr_writes))
               wait_mask |= BITSET_BIT(slot);
         }
      }

      /* Check for barriers */
      if (I->op == AGX_OPCODE_THREADGROUP_BARRIER) {
         for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
            if (slots[slot].nr_pending && slots[slot].drain_on_tg_barrier)
               wait_mask |= BITSET_BIT(slot);
         }
      } else if (I->op == AGX_OPCODE_MEMORY_BARRIER) {
         for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
            if (slots[slot].nr_pending)
               wait_mask |= BITSET_BIT(slot);
         }
      }

      unsigned my_use = UINT_MAX;
      /* Try to assign a slot */
      if (instr_is_async(I)) {
         if (!smart_waits) {
            for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
               if (slots[slot].nr_pending == 0) {
                  I->scoreboard = slot;
                  break;
               }
            }
         } else {
            my_use = (first_use && pos < block_len) ? first_use[pos] : block_len;
            unsigned p0 = (wait_mask & 1u) ? 0u : slots[0].nr_pending;
            unsigned p1 = (wait_mask & 2u) ? 0u : slots[1].nr_pending;
            unsigned u0 = p0 ? slot_min_use[0] : UINT_MAX;
            unsigned u1 = p1 ? slot_min_use[1] : UINT_MAX;

            if (p0 == 0 && p1 == 0) {
               I->scoreboard = 0;
            } else if (p0 == 0 || p1 == 0) {
               unsigned free_s = (p0 == 0) ? 0u : 1u;
               unsigned busy_s = 1u - free_s;
               unsigned bp = busy_s ? p1 : p0;
               unsigned bu = busy_s ? u1 : u0;
               unsigned diff = (my_use > bu) ? (my_use - bu) : (bu - my_use);
               bool upcoming_far = false;
               unsigned look = 0;
               for (agx_instr *K = list_entry(I->link.next, agx_instr, link);
                    &K->link != &block->instructions && look < 16;
                    K = list_entry(K->link.next, agx_instr, link), ++look) {
                  if (instr_is_async(K)) {
                     unsigned kpos = pos + 1 + look;
                     unsigned ku = (first_use && kpos < block_len)
                                      ? first_use[kpos]
                                      : block_len;
                     unsigned kdiff = (ku > bu) ? (ku - bu) : (bu - ku);
                     if (kdiff > 64) {
                        upcoming_far = true;
                        break;
                     }
                  }
               }
               if (bp < AGX_MAX_PENDING && diff <= 48 && upcoming_far)
                  I->scoreboard = busy_s;
               else
                  I->scoreboard = free_s;
            } else {
               if (p0 >= AGX_MAX_PENDING && p1 < AGX_MAX_PENDING) {
                  I->scoreboard = 1;
               } else if (p1 >= AGX_MAX_PENDING && p0 < AGX_MAX_PENDING) {
                  I->scoreboard = 0;
               } else {
                  unsigned d0 = (my_use > u0) ? (my_use - u0) : (u0 - my_use);
                  unsigned d1 = (my_use > u1) ? (my_use - u1) : (u1 - my_use);
                  if (d0 < d1)
                     I->scoreboard = 0;
                  else if (d1 < d0)
                     I->scoreboard = 1;
                  else
                     I->scoreboard = (p0 <= p1) ? 0 : 1;
               }
            }
         }
      }

      /* Check for slot overflow */
      if (instr_is_async(I) &&
          !(wait_mask & BITSET_BIT(I->scoreboard)) &&
          slots[I->scoreboard].nr_pending >= AGX_MAX_PENDING)
         wait_mask |= BITSET_BIT(I->scoreboard);

      /* Insert the appropriate waits, clearing the slots */
      u_foreach_bit(slot, wait_mask) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_wait(&b, slot);

         BITSET_ZERO(slots[slot].writes);
         slots[slot].nr_pending = 0;
         slots[slot].drain_on_tg_barrier = false;
         slot_min_use[slot] = UINT_MAX;
      }

      /* Record access */
      if (instr_is_async(I)) {
         agx_foreach_dest(I, d) {
            if (agx_is_null(I->dest[d]))
               continue;

            assert(I->dest[d].type == AGX_INDEX_REGISTER);
            BITSET_SET_COUNT(slots[I->scoreboard].writes, I->dest[d].value,
                             agx_index_size_16(I->dest[d]));
         }

         if (I->op != AGX_OPCODE_DEVICE_LOAD || !I->memory_can_reorder)
            slots[I->scoreboard].drain_on_tg_barrier = true;

         slots[I->scoreboard].nr_pending++;
         if (smart_waits && my_use < slot_min_use[I->scoreboard])
            slot_min_use[I->scoreboard] = my_use;
      }

      pos++;
   }

   free(first_use);

   /* If there are outstanding messages, wait for them. An ordinary shader's
    * exit block does not need a wait because nothing executes afterward.
    * Preamble state stores are different: the main shader consumes their
    * architectural texture/sampler state after the preamble stops.
    */
   if (block != agx_exit_block(ctx) || ctx->is_preamble) {
      agx_builder b = agx_init_builder(ctx, agx_after_block_logical(block));

      for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
         if (slots[slot].nr_pending)
            agx_wait(&b, slot);
      }
   }
}

/*
 * Assign scoreboard slots to asynchronous instructions and insert waits for the
 * appropriate hazard tracking.
 */
void
agx_insert_waits(agx_context *ctx)
{
   agx_foreach_block(ctx, block) {
      if (agx_compiler_debug & AGX_DBG_WAIT)
         agx_insert_waits_trivial(ctx, block);
      else
         agx_insert_waits_local(ctx, block);
   }
}

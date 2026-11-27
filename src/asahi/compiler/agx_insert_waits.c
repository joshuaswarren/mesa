/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_debug.h"

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
static void
agx_insert_waits_local(agx_context *ctx, agx_block *block)
{
   struct slot slots[2] = {0};

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

      /* Try to assign a free slot */
      if (instr_is_async(I)) {
         for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
            if (slots[slot].nr_pending == 0) {
               I->scoreboard = slot;
               break;
            }
         }
      }

      /* Check for slot overflow */
      if (instr_is_async(I) &&
          slots[I->scoreboard].nr_pending >= AGX_MAX_PENDING)
         wait_mask |= BITSET_BIT(I->scoreboard);

      /* Insert the appropriate waits, clearing the slots */
      u_foreach_bit(slot, wait_mask) {
         agx_builder b = agx_init_builder(ctx, agx_before_instr(I));
         agx_wait(&b, slot);

         BITSET_ZERO(slots[slot].writes);
         slots[slot].nr_pending = 0;
         slots[slot].drain_on_tg_barrier = false;
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
      }
   }

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

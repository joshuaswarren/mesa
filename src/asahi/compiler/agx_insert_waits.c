/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_debug.h"

#define AGX_MAX_PENDING (8)

/*
 * The scoreboarding immediate is packed into 2 bits of every asynchronous
 * instruction, so at most 4 slots exist. Divergent if/else regions lower to
 * exec-mask fallthrough rather than real branches, so pending messages may be
 * carried across those block boundaries and waited at first use instead of
 * being drained at every block exit.
 */
#define AGX_NUM_SLOTS (4)

/* Maximum nesting of divergent if/else regions tracked for cross-block
 * scoreboard state. Beyond this we fall back to conservative per-block
 * draining.
 */
#define AGX_MAX_FRAMES (64)

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
 * Open divergent if/else region. Divergent control flow lowers to exec-mask
 * push/pop with fallthrough, so a load issued in the then arm is still in
 * flight at the merge; the merge seeds the union of both arms' pending state.
 */
struct frame {
   /* Pending state on entry to the if, inherited by both arms */
   struct slot entry[AGX_NUM_SLOTS];

   /* Pending state left by the last block of the then arm */
   struct slot then_exit[AGX_NUM_SLOTS];

   /* Index of the else block (target of the if) */
   unsigned else_index;

   /* Index of the block following the if/else region */
   unsigned merge_index;

   /* Whether the else arm has been entered */
   bool else_seen;
};

static void
slots_copy(struct slot *dst, const struct slot *src)
{
   for (unsigned s = 0; s < AGX_NUM_SLOTS; ++s)
      dst[s] = src[s];
}

static void
slots_zero(struct slot *dst)
{
   for (unsigned s = 0; s < AGX_NUM_SLOTS; ++s) {
      BITSET_ZERO(dst[s].writes);
      dst[s].nr_pending = 0;
   }
}

/*
 * Merge two arms' pending state at a divergent merge. Scoreboard queues are
 * per-lane hardware state (the same accounting the straight-line path relies
 * on), and the two arms of a divergent if/else are mutually exclusive per
 * lane, so a lane has at most max(then, else) outstanding messages per slot,
 * never the sum. The write sets still union: a post-merge reader can be fed
 * by either arm depending on the lane's path, so pending writes from both
 * arms must trigger the wait.
 */
static void
slots_union(struct slot *dst, const struct slot *src)
{
   for (unsigned s = 0; s < AGX_NUM_SLOTS; ++s) {
      BITSET_OR(dst[s].writes, dst[s].writes, src[s].writes);
      dst[s].nr_pending = MAX2(dst[s].nr_pending, src[s].nr_pending);
   }
}

/* Terminator (last flow instruction) of a block, or NULL */
static agx_instr *
block_terminator(agx_block *block)
{
   agx_foreach_instr_in_block_rev(block, I) {
      if (!instr_after_logical_end(I))
         return NULL;

      return I;
   }

   return NULL;
}

static bool
is_mask_branch(agx_instr *I)
{
   return I && (I->op == AGX_OPCODE_IF_ICMP || I->op == AGX_OPCODE_IF_FCMP);
}

/*
 * Assign scoreboard slots to asynchronous instructions and insert waits for
 * the appropriate hazard tracking, carrying pending state across the
 * exec-mask boundaries of divergent if/else regions. Only real branches,
 * loop back-edges and barriers force pending messages to complete.
 */
static void
agx_insert_waits_regions(agx_context *ctx)
{
   struct slot slots[AGX_NUM_SLOTS] = {0};
   struct frame frames[AGX_MAX_FRAMES];
   unsigned nr_frames = 0;

   /* Blocks are walked (and indexed) in source order, so the i'th block
    * visited has index i.
    */
   agx_block **order = calloc(ctx->num_blocks, sizeof(agx_block *));
   fprintf(stderr, "AGXWAITS: enter shader=%s blocks=%u\n",
           ctx->nir->info.name ? ctx->nir->info.name : "(unnamed)",
           ctx->num_blocks);

   agx_foreach_block(ctx, block) {
      order[block->index] = block;

      /* Resolve region boundaries keyed by block index */
      while (nr_frames > 0) {
         struct frame *f = &frames[nr_frames - 1];

         if (!f->else_seen && block->index == f->else_index) {
            /* The then arm's pending state is saved, not dropped: the else
             * arm runs with the state from before the if (SSA dominance
             * means it cannot read a then-only definition), and the merge
             * seeds the union of both arms' state below.
             */
            slots_copy(f->then_exit, slots);
            slots_copy(slots, f->entry);
            f->else_seen = true;

            /* The then arm is complete, so its last block is now known
             * (blocks are visited in index order). Resolve the merge from
             * its fallthrough successor; on any unexpected shape, fold the
             * then arm's pending state in now and retire the frame.
             */
            agx_block *end_then = order[f->else_index - 1];

            if (!end_then->unconditional_jumps &&
                end_then->successors[0] && !end_then->successors[1] &&
                end_then->successors[0]->index > f->else_index) {
               f->merge_index = end_then->successors[0]->index;
               fprintf(stderr,
                       "AGXWAITS: blk%u else-arrival frame=%u merge=%u\n",
                       block->index, nr_frames - 1, f->merge_index);
            } else {
               slots_union(slots, f->then_exit);
               fprintf(stderr,
                       "AGXWAITS: blk%u else-arrival frame=%u FOLD shape\n",
                       block->index, nr_frames - 1);
               nr_frames--;
            }
            continue;
         } else if (f->merge_index != UINT_MAX &&
                    block->index == f->merge_index) {
            slots_union(slots, f->then_exit);
            fprintf(stderr, "AGXWAITS: blk%u merge-arrival frame=%u\n",
                    block->index, nr_frames - 1);
            nr_frames--;
            continue;
         }

         break;
      }

      agx_foreach_instr_in_block_safe(block, I) {
         uint8_t wait_mask = 0;

         /* Check for read-after-write */
         agx_foreach_src(I, s) {
            if (I->src[s].type != AGX_INDEX_REGISTER)
               continue;

            unsigned nr_read = agx_index_size_16(I->src[s]);
            for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
               if (BITSET_TEST_COUNT(slots[slot].writes, I->src[s].value,
                                     nr_read))
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
         if (I->op == AGX_OPCODE_THREADGROUP_BARRIER ||
             I->op == AGX_OPCODE_MEMORY_BARRIER) {

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
         }

         /* Record access */
         if (instr_is_async(I)) {
            agx_foreach_dest(I, d) {
               if (agx_is_null(I->dest[d]))
                  continue;

               assert(I->dest[d].type == AGX_INDEX_REGISTER);
               BITSET_SET_COUNT(slots[I->scoreboard].writes,
                                I->dest[d].value,
                                agx_index_size_16(I->dest[d]));
            }

            slots[I->scoreboard].nr_pending++;
         }
      }

      /* Decide whether pending messages must complete before leaving the
       * block. Inside a tracked divergent region they may carry across the
       * exec-mask fallthrough; everything else drains as before.
       */
      bool carry = false;

      if (block != agx_exit_block(ctx)) {
         agx_instr *term = block_terminator(block);

         if (is_mask_branch(term) && !term->target)
            fprintf(stderr, "AGXWAITS: blk%u reject target=NULL\n",
                    block->index);
         else if (is_mask_branch(term) && nr_frames >= AGX_MAX_FRAMES)
            fprintf(stderr, "AGXWAITS: blk%u reject frames-full\n",
                    block->index);

         if (is_mask_branch(term) && term->target &&
             nr_frames < AGX_MAX_FRAMES) {
            agx_block *then_blk = block->successors[0];
            agx_block *else_blk = block->successors[1];

            /* Shape checks on the local structure: the then arm starts at
             * the layout successor and the else block is the branch target.
             * The merge is resolved when the else arm is entered, since
             * blocks after the current one are not yet known here.
             */
            if (then_blk && else_blk && term->target &&
                term->target->index == else_blk->index &&
                then_blk->index == block->index + 1) {

               struct frame *f = &frames[nr_frames++];
               slots_copy(f->entry, slots);
               slots_zero(f->then_exit);
               f->else_index = else_blk->index;
               f->merge_index = UINT_MAX;
               f->else_seen = false;
               carry = true;
               fprintf(stderr, "AGXWAITS: blk%u ACCEPT frame=%u else=%u\n",
                       block->index, nr_frames - 1, f->else_index);
            } else {
               fprintf(stderr,
                       "AGXWAITS: blk%u reject shape tgt=%d s0=%d s1=%d\n",
                       block->index,
                       term->target ? (int)term->target->index : -1,
                       then_blk ? (int)then_blk->index : -1,
                       else_blk ? (int)else_blk->index : -1);
            }
         }

         /* Last block of the then arm carries into the else arm */
         if (!carry && nr_frames > 0) {
            struct frame *f = &frames[nr_frames - 1];

            if (!f->else_seen && block->index == f->else_index - 1)
               carry = true;
            else if (f->else_seen && f->merge_index != UINT_MAX &&
                     block->index == f->merge_index - 1)
               carry = true;
         }
      }

      if (!carry) {
         unsigned drained = 0;
         agx_builder b =
            agx_init_builder(ctx, agx_after_block_logical(block));

         for (unsigned slot = 0; slot < ARRAY_SIZE(slots); ++slot) {
            if (slots[slot].nr_pending) {
               agx_wait(&b, slot);
               BITSET_ZERO(slots[slot].writes);
               slots[slot].nr_pending = 0;
               drained++;
            }
         }

         if (drained)
            fprintf(stderr, "AGXWAITS: blk%u drain slots=%u\n",
                    block->index, drained);
      } else {
         fprintf(stderr, "AGXWAITS: blk%u carry\n", block->index);
      }
   }

   free(order);
}

/*
 * Assign scoreboard slots to asynchronous instructions and insert waits for the
 * appropriate hazard tracking.
 */
void
agx_insert_waits(agx_context *ctx)
{
   if (agx_compiler_debug & AGX_DBG_WAIT) {
      agx_foreach_block(ctx, block)
         agx_insert_waits_trivial(ctx, block);
   } else {
      agx_insert_waits_regions(ctx);
   }
}

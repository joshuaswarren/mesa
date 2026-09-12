/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_debug.h"

#define AGX_MAX_PENDING (8)

/*
 * The scoreboard index is one bit in the IR ("Scoreboard index, 0 or 1",
 * agx_instr::scoreboard in agx_compiler.h) and at most two bits in any
 * encoder, with texture sample packing a single bit (agx_pack.c). Two
 * slots is the documented hardware configuration; it still batches the
 * guarded loads of the target kernels across both slots.
 */
#define AGX_NUM_SLOTS (2)

/* Widening AGX_NUM_SLOTS past the IR field width silently truncates every
 * slot assignment above 1; re-prove the hardware contract before changing.
 */
_Static_assert(AGX_NUM_SLOTS <= 2, "agx_instr::scoreboard is a 1-bit field");
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
 * push/pop with fallthrough (emit_if in agx_compile.c: the else arm begins
 * with an unconditional else_fcmp, and a pop_exec is appended after the
 * logical end of the last block of the else arm for every region), so a load
 * issued in the then arm is still in flight inside the else arm and at the
 * merge.
 */
struct frame {
   /* Pending state on entry to the if, unioned into the else arm */
   struct slot entry[AGX_NUM_SLOTS];

   /* Index of the else block (target of the if) */
   unsigned else_index;

   /* Whether the else arm has been entered */
   bool else_seen;
};

static void
slots_copy(struct slot *dst, const struct slot *src)
{
   for (unsigned s = 0; s < AGX_NUM_SLOTS; ++s)
      dst[s] = src[s];
}

/*
 * Merge carried state at an else arrival. The write sets union so any later
 * reader waits for whichever arm's pending write it must order against:
 * register allocation runs before this pass, so a physical register written
 * by the then arm may legally be rewritten by the else arm, and an in-flight
 * writeback races that rewrite unless the else arm sees the pending state.
 *
 * Pending counts sum rather than max: the pass assumes nothing about whether
 * scoreboard queues are per-lane or shared per subgroup, so carried state
 * must assume both contributions are outstanding. Sums are clamped, and any
 * slot driven past AGX_MAX_PENDING by a merge is force-drained at the merge,
 * so the bound holds under either accounting.
 */
static void
slots_union(struct slot *dst, const struct slot *src)
{
   for (unsigned s = 0; s < AGX_NUM_SLOTS; ++s) {
      BITSET_OR(dst[s].writes, dst[s].writes, src[s].writes);
      dst[s].nr_pending =
         MIN2(dst[s].nr_pending + src[s].nr_pending, 4 * AGX_MAX_PENDING);
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
 *
 * Regions close lazily: every if/else region ends with a pop_exec appended
 * after the logical end of the last else-arm block, so a frame is closed
 * when a block is entered whose predecessor ends with pop_exec and which
 * lies past the region's else block. No merge index is resolved up front,
 * so every CFG shape a region can have is handled by the same rule. When a
 * frame fails to close, carried state only over-approximates the hardware's
 * pending messages, which costs extra waits at first use and nothing else.
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
           (ctx->nir && ctx->nir->info.name) ? ctx->nir->info.name
                                             : "(unnamed)",
           ctx->num_blocks);

   agx_foreach_block(ctx, block) {
      order[block->index] = block;

      while (nr_frames > 0) {
         struct frame *f = &frames[nr_frames - 1];

         if (!f->else_seen && block->index == f->else_index) {
            /* Else arrival: the then arm's pending writes stay in flight
             * (masked lanes skipped issuing them, the hardware did not
             * complete them), so the else arm must see the union of the
             * entry and then-exit state.
             */
            fprintf(stderr, "AGXWAITS: blk%u else-arrival frame=%u\n",
                    block->index, nr_frames - 1);
            slots_union(slots, f->entry);
            f->else_seen = true;

            uint8_t overflow = 0;

            for (unsigned s = 0; s < AGX_NUM_SLOTS; ++s) {
               if (slots[s].nr_pending > AGX_MAX_PENDING)
                  overflow |= BITSET_BIT(s);
            }

            u_foreach_bit(slot, overflow) {
               agx_builder b =
                  agx_init_builder(ctx, agx_before_block(block));
               agx_wait(&b, slot);

               BITSET_ZERO(slots[slot].writes);
               slots[slot].nr_pending = 0;
            }

            if (overflow) {
               fprintf(stderr, "AGXWAITS: blk%u overflow-drain mask=%x\n",
                       block->index, overflow);
            }

            continue;
         } else if (f->else_seen && block->index > f->else_index) {
            agx_instr *prev_term = block_terminator(order[block->index - 1]);

            if (prev_term && prev_term->op == AGX_OPCODE_POP_EXEC) {
               fprintf(stderr, "AGXWAITS: blk%u close frame=%u\n",
                       block->index, nr_frames - 1);
               nr_frames--;
               continue;
            }
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
       * block. Inside a tracked divergent region, a fallthrough edge stays
       * in the same masked execution, so state carries; real control
       * transfers drain as before. An else arm's last block ends with
       * pop_exec and falls through into the merge, which the close rule has
       * not seen yet, so it carries explicitly.
       */
      bool carry = false;

      if (block != agx_exit_block(ctx)) {
         agx_instr *term = block_terminator(block);

         if (is_mask_branch(term) && term->target &&
             nr_frames < AGX_MAX_FRAMES) {
            agx_block *then_blk = block->successors[0];
            agx_block *else_blk = block->successors[1];

            /* Shape checks on the local structure: the then arm starts at
             * the layout successor and the else block is the branch target.
             */
            if (then_blk && else_blk &&
                term->target->index == else_blk->index &&
                then_blk->index == block->index + 1) {

               struct frame *f = &frames[nr_frames++];
               slots_copy(f->entry, slots);
               f->else_index = else_blk->index;
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
         } else if (is_mask_branch(term) && !term->target) {
            fprintf(stderr, "AGXWAITS: blk%u reject target=NULL\n",
                    block->index);
         } else if (is_mask_branch(term) && nr_frames >= AGX_MAX_FRAMES) {
            fprintf(stderr, "AGXWAITS: blk%u reject frames-full\n",
                    block->index);
         } else if (nr_frames > 0 && !block->unconditional_jumps &&
                    (!term || term->op == AGX_OPCODE_POP_EXEC)) {
            carry = true;
         }
      }

      if (!carry && block != agx_exit_block(ctx)) {
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

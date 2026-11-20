/*
 * Copyright 2025 Valve Corporation
 * Copyright 2019-2022 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include "util/bitset.h"
#include "util/lut.h"
#include "util/u_debug.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "shader_enums.h"


/* Remove bit-identical overwrites only after allocation, without extending
 * allocation live ranges. The cache pass below recomputes all use hints. */
static void
agx3_remove_redundant_physical_copies(agx_context *ctx)
{
   if (ctx->key->dev.encoding != AGX_ISA_ENCODING_AGX3C ||
       ctx->stage != MESA_SHADER_FRAGMENT || ctx->is_preamble)
      return;

   uint32_t serial = 0;
   agx_foreach_block(ctx, block) {
      uint32_t value[AGX_NUM_REGS] = {0};
      agx_foreach_instr_in_block_safe(block, I) {
         const struct agx_opcode_info *info = &agx_opcodes_info[I->op];
         if (!info->can_eliminate || !info->can_reorder ||
             info->schedule_class != AGX_SCHEDULE_CLASS_NONE ||
             !I->nr_dests) {
            memset(value, 0, sizeof(value));
            continue;
         }

         uint32_t copied[4] = {0}, old_value[4] = {0};
         bool copy = false;
         const agx_index dst = I->nr_dests == 1 ? I->dest[0] : agx_null();
         const unsigned count = agx_is_null(dst) ? 0 : agx_index_size_16(dst);
         const bool plain_dst = dst.type == AGX_INDEX_REGISTER && !dst.memory &&
            !dst.abs && !dst.neg && !I->saturate && !I->shift &&
            !I->invert_cond && !I->memory_is_packed &&
            count >= 1 && count <= 4 && dst.value + count <= AGX_NUM_REGS;
         if (plain_dst) {
            for (unsigned c = 0; c < count; ++c)
               old_value[c] = value[dst.value + c];
         }
         if (plain_dst && I->op == AGX_OPCODE_MOV_IMM && count <= 2) {
            for (unsigned c = 0; c < count; ++c)
               copied[c] = UINT32_C(0x80000000) |
                           ((I->imm >> (16 * c)) & 0xffff);
            copy = true;
         } else if (plain_dst && I->op == AGX_OPCODE_BITOP &&
                    I->truth_table == UTIL_LUT2(a) && I->nr_srcs == 2) {
            const agx_index src = I->src[0];
            if (!src.memory && !src.abs && !src.neg && src.size == dst.size &&
                agx_index_size_16(src) == count) {
               if (src.type == AGX_INDEX_REGISTER &&
                   src.value + count <= AGX_NUM_REGS) {
                  for (unsigned c = 0; c < count; ++c) {
                     if (!value[src.value + c])
                        value[src.value + c] = ++serial;
                     copied[c] = value[src.value + c];
                  }
                  copy = true;
               } else if (src.type == AGX_INDEX_UNIFORM &&
                          src.value + count <= 512) {
                  for (unsigned c = 0; c < count; ++c)
                     copied[c] = UINT32_C(0x40000000) | (src.value + c);
                  copy = true;
               }
            }
         }
         if (copy) {
            bool redundant = true;
            for (unsigned c = 0; c < count; ++c)
               redundant &= old_value[c] == copied[c];
            if (redundant) {
               agx_remove_instruction(I);
               continue;
            }
            for (unsigned c = 0; c < count; ++c)
               value[dst.value + c] = copied[c];
         } else {
            agx_foreach_dest(I, d) {
               const agx_index dest = I->dest[d];
               const unsigned n = agx_index_size_16(dest);
               if (dest.type != AGX_INDEX_REGISTER || dest.memory ||
                   dest.value + n > AGX_NUM_REGS) {
                  memset(value, 0, sizeof(value));
                  break;
               }
               for (unsigned c = 0; c < n; ++c)
                  value[dest.value + c] = ++serial;
            }
         }
      }
   }
}

/* agx_opt_break_if moves a conditional break into the block after its if,
 * but the edge to the break target stays on the emptied then block. The
 * target's live-in is live at the break itself. */
static agx_block *
fused_break_target(const agx_instr *I)
{
   return (I->op == AGX_OPCODE_BREAK_IF_ICMP ||
           I->op == AGX_OPCODE_BREAK_IF_FCMP)
             ? I->target
             : NULL;
}

static void
postra_liveness_ins(BITSET_WORD *live, agx_instr *I)
{
   agx_foreach_reg_dest(I, d) {
      unsigned reg = I->dest[d].value;
      BITSET_CLEAR_COUNT(live, reg, agx_index_size_16(I->dest[d]));
   }

   agx_foreach_reg_src(I, s) {
      unsigned reg = I->src[s].value;
      BITSET_SET_COUNT(live, reg, agx_index_size_16(I->src[s]));
   }
}

/*
 * Globally, liveness analysis uses a fixed-point algorithm based on a
 * worklist. We initialize a work list with the exit block. We iterate the work
 * list to compute live_in from live_out for each block on the work list,
 * adding the predecessors of the block to the work list if we made progress.
 */
static void
postra_liveness(agx_context *ctx)
{
   u_worklist worklist;
   agx_worklist_init(ctx, &worklist);

   bool fused_breaks = false;
   agx_foreach_block(ctx, block) {
      BITSET_ZERO(block->reg_live_in);
      BITSET_ZERO(block->reg_live_out);

      agx_foreach_instr_in_block(block, I)
         fused_breaks |= fused_break_target(I) != NULL;
   }

   /* A fused break's block is not a predecessor of its target, so with
    * fused breaks the whole analysis repeats until nothing changes. */
   bool changed;
   do {
      changed = false;
      agx_foreach_block(ctx, block)
         agx_worklist_push_tail(&worklist, block);

      while (!u_worklist_is_empty(&worklist)) {
         /* Pop off in reverse order since liveness is backwards */
         agx_block *blk = agx_worklist_pop_tail(&worklist);

         /* Calculate liveness locally */
         agx_foreach_successor(blk, succ) {
            BITSET_OR(blk->reg_live_out, blk->reg_live_out, succ->reg_live_in);
         }

         BITSET_DECLARE(live, AGX_NUM_REGS);
         memcpy(live, blk->reg_live_out, sizeof(live));

         agx_foreach_instr_in_block_rev(blk, ins) {
            agx_block *target = fused_break_target(ins);
            if (target)
               BITSET_OR(live, live, target->reg_live_in);

            postra_liveness_ins(live, ins);
         }

         if (BITSET_EQUAL(blk->reg_live_in, live))
            continue;

         /* We made progress, so we need to reprocess the predecessors */
         memcpy(blk->reg_live_in, live, sizeof(live));
         changed = true;
         agx_foreach_predecessor(blk, pred) {
            agx_worklist_push_head(&worklist, *pred);
         }
      }
   } while (fused_breaks && changed);

   u_worklist_fini(&worklist);
}

static bool
writes_reg(const agx_instr *I, unsigned reg)
{
   agx_foreach_reg_dest(I, d) {
      unsigned count = agx_index_size_16(I->dest[d]);

      if (reg >= I->dest[d].value && (reg - I->dest[d].value) < count)
         return true;
   }

   return false;
}

/*
 * Mark last-use sources to allow the hardware to discard from the register
 * cache. Last use information follows immediately from (post-RA) liveness
 * analysis: a register is dead immediately after its last use.
 *
 * Mark cache hints on sources/destinations to encourage the hardware to make
 * better use of the register cache. This is a simple local analysis.
 */
void
agx_opt_register_cache(agx_context *ctx)
{
   agx3_remove_redundant_physical_copies(ctx);

   /* Analyze the shader globally */
   postra_liveness(ctx);

   agx_foreach_block(ctx, block) {
      /* Live-set at each point in the program */
      BITSET_DECLARE(live, AGX_NUM_REGS);
      memcpy(live, block->reg_live_out, sizeof(live));

      /* Set of registers read "soon" by an ALU instruction. These are
       * candidates for the .cache bit.
       */
      BITSET_DECLARE(alu_reads, AGX_NUM_REGS) = {0};

      agx_foreach_instr_in_block_rev(block, I) {
         agx_block *target = fused_break_target(I);
         if (target)
            BITSET_OR(live, live, target->reg_live_in);

         agx_foreach_reg_dest(I, d) {
            uint64_t reg = I->dest[d].value;
            unsigned nr = agx_index_size_16(I->dest[d]);

            I->dest[d].cache = BITSET_TEST(alu_reads, I->dest[d].value);
            BITSET_CLEAR_COUNT(alu_reads, reg, nr);
         }

         agx_foreach_reg_src(I, s) {
            I->src[s].cache = BITSET_TEST(alu_reads, I->src[s].value);
         }

         agx_foreach_reg_src(I, s) {
            uint64_t reg = I->src[s].value;
            unsigned nr = agx_index_size_16(I->src[s]);

            /* If the register dead after this instruction, it's the last use.
             * That includes if the register is overwritten this cycle, but that
             * won't show up in the liveness analysis.
             */
            bool lu = !BITSET_TEST_COUNT(live, reg, nr) || writes_reg(I, reg);

            /* Handling divergent blocks would require physical CFG awareness.
             * Just bail for now, skipping this pass won't affect correctness.
             * Subgroup-uniform ifs/loops have no physical divergence.
             */
            I->src[s].discard = lu && !block->cf_divergent;

            /* Mark any source read by an ALU instruction in the same block as
             * wanting a .cache hint. This is better than just marking
             * everything, although it overly hints for very long blocks and
             * underhints for registers used across block boundaries. It's
             * probably good enough, though, and it's not clear how to do much
             * better given our limited understanding of the hardware.
             */
            if (agx_is_alu(I))
               BITSET_SET_COUNT(alu_reads, reg, nr);

            assert(!(I->src[s].discard && I->src[s].cache));
         }

         /* M5 (AGX3C): a register read by more than one source of the same
          * instruction is not discarded by any of them. With both sources
          * discarding, allocate_uvb with the primitive and vertex counts in
          * one register (point meshes) read the wrong vertex count; the
          * discard appears to take effect before the later read. Not
          * setting a discard is always safe: it only gives up a register
          * cache saving. */
         if (ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C) {
            agx_foreach_reg_src(I, s) {
               if (!I->src[s].discard)
                  continue;
               const unsigned a = I->src[s].value;
               const unsigned an = agx_index_size_16(I->src[s]);
               agx_foreach_reg_src(I, t) {
                  if (t == s)
                     continue;
                  const unsigned b = I->src[t].value;
                  const unsigned bn = agx_index_size_16(I->src[t]);
                  if (a < b + bn && b < a + an) {
                     I->src[s].discard = false;
                     break;
                  }
               }
            }
         }

         postra_liveness_ins(live, I);
      }
   }

   /* Apple's G17P fragment loop test keeps
    * its counter operand in the register cache (the f11 hint on the icmp's
    * counter source). The block-local analysis above cannot: the counter's next
    * reads are at the top of the loop, across the back edge. Cache a loop
    * compare's register source when the loop header reads it with an ALU
    * instruction before writing it, unless this read is its last use. */
   if (!agx_isa_is_agx3(ctx->key->dev.encoding))
      return;

   agx_foreach_block(ctx, block) {
      agx_block *header = NULL;
      agx_foreach_successor(block, succ) {
         if (succ->loop_header && succ->index <= block->index)
            header = succ;
      }
      if (!header)
         continue;

      agx_foreach_instr_in_block(block, I) {
         if (I->op != AGX_OPCODE_IF_ICMP && I->op != AGX_OPCODE_WHILE_ICMP &&
             I->op != AGX_OPCODE_BREAK_IF_ICMP)
            continue;
         agx_foreach_reg_src(I, s) {
            if (I->src[s].discard || I->src[s].cache ||
                I->src[s].type != AGX_INDEX_REGISTER)
               continue;
            const unsigned reg = I->src[s].value;
            const unsigned nr = agx_index_size_16(I->src[s]);
            bool read = false;
            agx_foreach_instr_in_block(header, H) {
               bool reads = false;
               agx_foreach_reg_src(H, hs) {
                  const unsigned r = H->src[hs].value;
                  const unsigned rn = agx_index_size_16(H->src[hs]);
                  reads |= r < reg + nr && reg < r + rn;
               }
               if (reads) {
                  read = agx_is_alu(H);
                  break;
               }
               if (writes_reg(H, reg))
                  break;
            }
            I->src[s].cache = read;
         }
      }
   }
}

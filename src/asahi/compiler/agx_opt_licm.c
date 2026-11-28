/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "util/bitset.h"
#include "util/sparse_bitset.h"
#include "util/u_dynarray.h"
#include "agx_compiler.h"

/*
 * Loop-invariant motion of late-materialized operands.
 *
 * Backend lowerings after NIR's loop-invariant code motion create operands
 * per use: immediates the consuming instruction cannot encode become mov_imm
 * (agx_lower_agx3_immediates) and uniforms it cannot read in place become a
 * mov from the uniform (agx_lower_uniform_sources), both right before the
 * consumer. Inside a loop they then run every iteration.
 *
 * Register allocation calls this when the program fits its register budget
 * without spilling. It hoists the operands to the loop's preheader and merges
 * equal ones. A hoisted value is live through the whole loop, so a loop keeps
 * its hoisting only if register demand, measured as allocation measures it,
 * stays within max_demand, the demand allocation measured before: it then
 * assigns registers within the same occupancy step and has nothing to spill
 * or rematerialize back into the loop. Demand counts registers, not where
 * they are, so the loop must also leave room in the file_regs registers
 * allocation assigns from for its widest value: in a full file, a value live
 * through the loop sits where a vector needs to go, and allocation copies it
 * out of the way and back in every iteration.
 */

struct licm_loop {
   unsigned header, latch;
};

/* Where an instruction was: after `prev` in `block`, or first if NULL. */
struct licm_move {
   agx_instr *I;
   agx_block *block;
   agx_instr *prev;
};

struct licm_rewrite {
   agx_instr *I;
   unsigned s;
   agx_index old;
};

/* Whether I reads its sources as ordinary ALU operands. Vector construction
 * and phis want a value in a particular register, as do memory, texture and
 * export operands: defined right before such a use, allocation places the
 * value there directly, while a value live through the loop would need a copy
 * there every iteration. */
static bool
plain_use(const agx_instr *I)
{
   switch (I->op) {
   case AGX_OPCODE_COLLECT:
   case AGX_OPCODE_SPLIT:
   case AGX_OPCODE_PHI:
   case AGX_OPCODE_EXPORT:
      return false;
   case AGX_OPCODE_IF_FCMP:
   case AGX_OPCODE_IF_ICMP:
   case AGX_OPCODE_ELSE_FCMP:
   case AGX_OPCODE_ELSE_ICMP:
   case AGX_OPCODE_WHILE_FCMP:
   case AGX_OPCODE_WHILE_ICMP:
   case AGX_OPCODE_BREAK_IF_FCMP:
   case AGX_OPCODE_BREAK_IF_ICMP:
      return true;
   default:
      return agx_opcodes_info[I->op].schedule_class == AGX_SCHEDULE_CLASS_NONE;
   }
}

/* M5 unpacked 2D/3D scalar LOD/bias samples require their half-float source to
 * be discarded at the sample (agx3_pack_binary.c). Hoisting a loop-local copy
 * into the preheader makes it live across the back-edge and strips discard. */
static bool
m5_unpacked_scalar_lod(const agx_context *ctx, const agx_instr *I)
{
   return ctx->key && ctx->key->dev.encoding == AGX_ISA_ENCODING_AGX3C &&
          !agx_is_g18g(&ctx->key->dev) &&
          I->op == AGX_OPCODE_TEXTURE_SAMPLE && !I->memory_is_packed &&
          (I->dim == AGX_DIM_2D || I->dim == AGX_DIM_3D) &&
          (I->lod_mode == AGX_LOD_MODE_LOD_MIN ||
           I->lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS) &&
          I->src[1].type == AGX_INDEX_NORMAL &&
          I->src[1].size == AGX_SIZE_16;
}

static bool
hoistable(const agx_context *ctx, const agx_instr *I, const BITSET_WORD *pinned)
{
   if (I->nr_dests != 1 || I->dest[0].type != AGX_INDEX_NORMAL ||
       I->dest[0].memory || BITSET_TEST(pinned, I->dest[0].value))
      return false;

   if (I->op == AGX_OPCODE_MOV_IMM)
      return true;

   /* A uniform is constant through the program unless the program writes
    * uniforms itself. */
   return I->op == AGX_OPCODE_MOV && I->nr_srcs == 1 &&
          I->src[0].type == AGX_INDEX_UNIFORM && !I->src[0].memory &&
          !ctx->uniforms_written;
}

static int
cmp_loop_size(const void *a, const void *b)
{
   const struct licm_loop *x = a, *y = b;
   const unsigned sx = x->latch - x->header, sy = y->latch - y->header;
   return sx < sy ? -1 : sx > sy ? 1 : (int)x->header - (int)y->header;
}

static unsigned
current_demand(agx_context *ctx)
{
   agx_compute_liveness(ctx);
   return agx_calc_register_demand(ctx, false);
}

static void
revert(struct util_dynarray *moves, struct util_dynarray *rewrites)
{
   util_dynarray_foreach(rewrites, struct licm_rewrite, r)
      r->I->src[r->s] = r->old;

   /* Each recorded predecessor is one the pass never moved, since any moved
    * before it was already gone: restoring in reverse removal order puts
    * instructions that were adjacent back in their order. */
   util_dynarray_foreach_reverse(moves, struct licm_move, m) {
      /* Hoisted instructions sit in the preheader; merged ones were only
       * removed. */
      if (m->I->link.next)
         list_del(&m->I->link);
      list_add(&m->I->link,
               m->prev ? &m->prev->link : &m->block->instructions);
   }
}

bool
agx_opt_licm(agx_context *ctx, unsigned max_demand, unsigned file_regs)
{
   /* Passes that delete blocks leave gaps in the indices. */
   unsigned nr_blocks = 0;
   agx_foreach_block(ctx, block)
      nr_blocks = MAX2(nr_blocks, block->index + 1);

   agx_block **blocks = calloc(MAX2(nr_blocks, 1), sizeof(*blocks));
   struct licm_loop *loops = calloc(MAX2(nr_blocks, 1), sizeof(*loops));
   unsigned nr_loops = 0;

   agx_foreach_block(ctx, block)
      blocks[block->index] = block;

   /* In structured control flow a loop spans the source-order interval from
    * its header to its last back edge. */
   agx_foreach_block(ctx, block) {
      agx_foreach_successor(block, succ) {
         if (!succ->loop_header || succ->index > block->index)
            continue;

         unsigned i;
         for (i = 0; i < nr_loops && loops[i].header != succ->index; ++i)
            ;
         if (i == nr_loops)
            loops[nr_loops++] = (struct licm_loop){succ->index, block->index};
         loops[i].latch = MAX2(loops[i].latch, block->index);
      }
   }

   if (!nr_loops) {
      free(blocks);
      free(loops);
      return false;
   }

   /* Innermost first: an operand hoisted out of an inner loop lands in the
    * outer loop's body and is considered again there. */
   qsort(loops, nr_loops, sizeof(*loops), cmp_loop_size);

   agx_index *replace = calloc(MAX2(ctx->alloc, 1), sizeof(*replace));
   BITSET_WORD *pinned = BITSET_CALLOC(MAX2(ctx->alloc, 1));
   agx_foreach_instr_global(ctx, I) {
      if (!plain_use(I)) {
         agx_foreach_ssa_src(I, s)
            BITSET_SET(pinned, I->src[s].value);
      } else if (m5_unpacked_scalar_lod(ctx, I)) {
         BITSET_SET(pinned, I->src[1].value);
      }
   }
   struct util_dynarray hoisted, moves, rewrites, merged;
   util_dynarray_init(&hoisted, NULL);
   util_dynarray_init(&moves, NULL);
   util_dynarray_init(&rewrites, NULL);
   util_dynarray_init(&merged, NULL);
   bool progress = false;

   for (unsigned l = 0; l < nr_loops; ++l) {
      const struct licm_loop loop = loops[l];
      agx_block *header = blocks[loop.header];
      if (!header)
         continue;

      /* The one predecessor outside the loop, entering only the loop. */
      agx_block *pre = NULL;
      unsigned outside = 0;
      agx_foreach_predecessor(header, p) {
         if ((*p)->index < loop.header) {
            pre = *p;
            outside++;
         }
      }
      if (outside != 1 || agx_num_successors(pre) != 1)
         continue;

      util_dynarray_clear(&hoisted);
      util_dynarray_clear(&moves);
      util_dynarray_clear(&rewrites);
      util_dynarray_clear(&merged);
      agx_cursor cursor = agx_after_block_logical(pre);
      unsigned width = 0;

      for (unsigned b = loop.header; b <= loop.latch; ++b) {
         if (!blocks[b])
            continue;

         /* Avoid hoisting outer-loop immediates/uniforms across an enclosed
          * inner loop that does not execute `blocks[b]`: doing so keeps the
          * constant live across every inner-loop iteration while saving at
          * most one execution per outer iteration. */
         bool outside_inner_loop = false;
         for (unsigned k = 0; k < l; ++k) {
            if (loop.header < loops[k].header && loops[k].latch <= loop.latch &&
                (b < loops[k].header || b > loops[k].latch)) {
               outside_inner_loop = true;
               break;
            }
         }

         agx_foreach_instr_in_block_safe(blocks[b], I) {
            agx_foreach_ssa_dest(I, d) {
               width = MAX2(width, util_next_power_of_two(
                                      agx_index_size_16(I->dest[d])));
            }

            if (outside_inner_loop || !hoistable(ctx, I, pinned))
               continue;

            agx_instr *same = NULL;
            util_dynarray_foreach(&hoisted, agx_instr *, H) {
               if (agx_instrs_equal_for_cse(*H, I)) {
                  same = *H;
                  break;
               }
            }

            struct licm_move m = {
               .I = I,
               .block = blocks[b],
               .prev = I->link.prev == &blocks[b]->instructions
                          ? NULL
                          : list_entry(I->link.prev, agx_instr, link),
            };
            util_dynarray_append(&moves, m);
            agx_remove_instruction(I);

            if (same) {
               replace[I->dest[0].value] = same->dest[0];
               util_dynarray_append(&merged, I->dest[0].value);
            } else {
               agx_builder_insert(&cursor, I);
               util_dynarray_append(&hoisted, I);
            }
         }
      }

      if (!util_dynarray_num_elements(&moves, struct licm_move))
         continue;

      if (util_dynarray_num_elements(&merged, unsigned)) {
         agx_foreach_instr_global(ctx, I) {
            agx_foreach_ssa_src(I, s) {
               agx_index r = replace[I->src[s].value];
               if (r.type != AGX_INDEX_NORMAL)
                  continue;

               struct licm_rewrite rw = {I, s, I->src[s]};
               util_dynarray_append(&rewrites, rw);
               agx_replace_src(I, s, r);
            }
         }
      }

      const unsigned demand = current_demand(ctx);
      if (demand <= max_demand && demand + width <= file_regs) {
         progress = true;
      } else {
         revert(&moves, &rewrites);
      }

      util_dynarray_foreach(&merged, unsigned, v)
         replace[*v] = agx_null();
   }

   /* Leave liveness describing the final program. */
   agx_compute_liveness(ctx);

   util_dynarray_fini(&merged);
   util_dynarray_fini(&rewrites);
   util_dynarray_fini(&moves);
   util_dynarray_fini(&hoisted);
   free(pinned);
   free(replace);
   free(loops);
   free(blocks);
   return progress;
}

/*
 * Sink single-block-used MOV_IMM instructions closer to their sole consumer
 * block whenever doing so reduces cross-block live range without increasing
 * inner-loop execution frequency:
 *   1. use_depth <= def_depth (never executes more often), OR
 *   2. the immediate is currently live-in across a strictly deeper inner loop
 *      (depth > use_depth), such as outer-loop or cold-branch immediates
 *      defined in block0 and kept live across a hot inner loop.
 */
bool
agx_opt_sink_immediates(agx_context *ctx)
{
   unsigned nr_blocks = 0;
   agx_foreach_block(ctx, block)
      nr_blocks = MAX2(nr_blocks, block->index + 1);
   if (nr_blocks <= 1 || !ctx->alloc)
      return false;

   agx_block **blocks = calloc(nr_blocks, sizeof(*blocks));
   struct licm_loop *loops = calloc(nr_blocks, sizeof(*loops));
   unsigned *loop_depth = calloc(nr_blocks, sizeof(*loop_depth));
   unsigned nr_loops = 0;

   agx_foreach_block(ctx, block)
      blocks[block->index] = block;

   agx_foreach_block(ctx, block) {
      agx_foreach_successor(block, succ) {
         if (!succ->loop_header || succ->index > block->index)
            continue;

         unsigned i;
         for (i = 0; i < nr_loops && loops[i].header != succ->index; ++i)
            ;
         if (i == nr_loops)
            loops[nr_loops++] = (struct licm_loop){succ->index, block->index};
         loops[i].latch = MAX2(loops[i].latch, block->index);
      }
   }

   for (unsigned l = 0; l < nr_loops; ++l) {
      for (unsigned b = loops[l].header; b <= loops[l].latch; ++b) {
         if (blocks[b])
            loop_depth[b]++;
      }
   }

   agx_instr **def_instr = calloc(ctx->alloc, sizeof(*def_instr));
   agx_block **def_block = calloc(ctx->alloc, sizeof(*def_block));
   agx_block **use_block = calloc(ctx->alloc, sizeof(*use_block));
   agx_instr **first_use = calloc(ctx->alloc, sizeof(*first_use));
   agx_block *const MULTI_USE = (agx_block *)(uintptr_t)1;

   agx_foreach_block(ctx, block) {
      bool past_cf = false;
      agx_foreach_instr_in_block(block, I) {
         if (instr_after_logical_end(I))
            past_cf = true;

         if (I->op == AGX_OPCODE_MOV_IMM && I->nr_dests == 1 &&
             I->dest[0].type == AGX_INDEX_NORMAL && !I->dest[0].memory &&
             (I->dest[0].size == AGX_SIZE_16 ||
              I->dest[0].size == AGX_SIZE_32)) {
            unsigned v = I->dest[0].value;
            if (v < ctx->alloc) {
               def_instr[v] = I;
               def_block[v] = block;
            }
         }

         agx_foreach_ssa_src(I, s) {
            unsigned v = I->src[s].value;
            if (v >= ctx->alloc)
               continue;
            if (I->op == AGX_OPCODE_PHI || past_cf) {
               use_block[v] = MULTI_USE;
            } else if (!use_block[v]) {
               use_block[v] = block;
               first_use[v] = I;
            } else if (use_block[v] != block) {
               use_block[v] = MULTI_USE;
            }
         }
      }
   }

   bool have_liveness = false;
   bool progress = false;

   for (unsigned v = 0; v < ctx->alloc; ++v) {
      if (!def_instr[v] || !use_block[v] || use_block[v] == MULTI_USE ||
          use_block[v] == def_block[v])
         continue;

      const unsigned def_d = loop_depth[def_block[v]->index];
      const unsigned use_d = loop_depth[use_block[v]->index];
      bool should_sink = (use_d <= def_d);

      if (!should_sink && nr_loops) {
         if (!have_liveness) {
            agx_compute_liveness(ctx);
            have_liveness = true;
         }
         for (unsigned l = 0; l < nr_loops; ++l) {
            agx_block *hdr = blocks[loops[l].header];
            if (hdr && loop_depth[hdr->index] > use_d &&
                u_sparse_bitset_test(&hdr->live_in, v)) {
               should_sink = true;
               break;
            }
         }
      }

      if (!should_sink)
         continue;

      agx_instr *I = def_instr[v];
      agx_remove_instruction(I);
      agx_cursor cursor = agx_before_instr(first_use[v]);
      agx_builder_insert(&cursor, I);
      progress = true;
   }

   free(first_use);
   free(use_block);
   free(def_block);
   free(def_instr);
   free(loop_depth);
   free(loops);
   free(blocks);
   return progress;
}

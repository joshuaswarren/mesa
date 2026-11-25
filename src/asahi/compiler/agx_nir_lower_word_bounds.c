/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"
#include "agx_nir.h"
#include "nir.h"

/*
 * Check a constant program's constant-offset buffer reads in words.
 *
 * A robust read at a constant offset keeps its value only when its exclusive
 * end lies within the descriptor's byte size:
 *
 *    bounds_agx(data, end, size) = end <= size ? data : 0
 *
 * which is one compare-select taking `end` as an 8-bit immediate. Every check
 * with an end past 255 bytes therefore first materializes its end with a
 * constant move. Large constant buffers (a G-buffer pass reading a 2 KiB D3D
 * constant buffer) pay that move for most of their reads, and a constant
 * program reruns for every tile its draw touches.
 *
 * For a word-aligned end, end <= size exactly when end/4 <= size/4 (rounded
 * down): end/4 is an integer, and an integer is at most a real exactly when it
 * is at most its floor. A size is a 32-bit unsigned value, so size/4 < 2^30,
 * and every end/4 here is below 2^14. Subtracting a common base C < 2^14 from
 * both sides of a signed comparison is then exact too, as no term leaves
 * (-2^31, 2^31):
 *
 *    end <= size  <=>  end/4 - C <= size/4 - C        (signed)
 *
 * Choose C per 256-word window so that end/4 - C is an 8-bit immediate. The
 * checks of one descriptor then share one shift (size/4) and one subtraction
 * per window past the first, instead of a constant move per check. Each check
 * keeps its own exact predicate: no value is read, kept or zeroed
 * differently. A descriptor's checks are rewritten only when that is cheaper.
 *
 * Main programs keep their byte checks: there, the shared word sizes would be
 * more live uniform values competing for registers.
 */

/* Constant-buffer offsets stay well below 64 KiB: 64 windows of 1 KiB. */
#define WORD_BOUND_WINDOWS 64

struct word_bound {
   nir_alu_instr *alu;
   uint32_t end;
};

struct word_bound_group {
   nir_scalar size;
   struct util_dynarray checks;
};

static nir_cursor
after_scalar_def(nir_scalar s)
{
   nir_instr *instr = nir_def_instr(s.def);
   return instr->type == nir_instr_type_phi ? nir_after_phis(instr->block)
                                            : nir_after_instr(instr);
}

static void
collect_checks(nir_function_impl *impl, struct util_dynarray *groups,
               void *mem)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op != nir_op_bounds_agx || alu->def.bit_size != 32 ||
             alu->def.num_components != 1 ||
             !nir_src_is_const(alu->src[1].src))
            continue;

         const uint32_t end = nir_alu_src_as_uint(alu->src[1]);
         if (end == 0 || (end % 4) != 0 ||
             (end / 4) >= (WORD_BOUND_WINDOWS << 8))
            continue;

         nir_scalar size =
            nir_get_scalar(alu->src[2].src.ssa, alu->src[2].swizzle[0]);
         struct word_bound_group *group = NULL;
         util_dynarray_foreach(groups, struct word_bound_group, g) {
            if (nir_scalar_equal(g->size, size)) {
               group = g;
               break;
            }
         }
         if (!group) {
            group = util_dynarray_grow(groups, struct word_bound_group, 1);
            *group = (struct word_bound_group){.size = size};
            util_dynarray_init(&group->checks, mem);
         }
         struct word_bound check = {.alu = alu, .end = end};
         util_dynarray_append(&group->checks, check);
      }
   }
}

static bool
words_cheaper(const struct word_bound_group *g)
{
   /* A byte check costs a constant move once its end is past 255. In words,
    * the descriptor costs a shift, plus a constant move and a subtraction for
    * each 256-word window past the first. */
   unsigned byte_cost = 0, word_cost = 1;
   bool windows[WORD_BOUND_WINDOWS] = {false};
   util_dynarray_foreach(&g->checks, struct word_bound, c) {
      byte_cost += c->end > 255;
      const unsigned window = (c->end / 4) >> 8;
      if (window && !windows[window]) {
         windows[window] = true;
         word_cost += 2;
      }
   }
   return word_cost < byte_cost;
}

static void
rewrite_group(const struct word_bound_group *g)
{
   nir_builder b = nir_builder_at(after_scalar_def(g->size));
   nir_def *window_words[WORD_BOUND_WINDOWS] = {NULL};
   window_words[0] = nir_ushr_imm(&b, nir_mov_scalar(&b, g->size), 2);

   util_dynarray_foreach(&g->checks, struct word_bound, c) {
      const unsigned window = (c->end / 4) >> 8;
      if (!window_words[window]) {
         window_words[window] =
            nir_iadd_imm(&b, window_words[0], -(int64_t)(window << 8));
      }
   }

   util_dynarray_foreach(&g->checks, struct word_bound, c) {
      const uint32_t end_words = c->end / 4;
      const unsigned window = end_words >> 8;
      b.cursor = nir_before_instr(&c->alu->instr);
      nir_def *data = nir_ssa_for_alu_src(&b, c->alu, 0);
      nir_def *checked;
      if (window == 0) {
         /* Unsigned is exact here: size/4 is never negative. */
         checked = nir_bounds_agx(&b, data, nir_imm_int(&b, end_words),
                                  window_words[0]);
      } else {
         nir_def *outside = nir_ilt_imm(&b, window_words[window],
                                        end_words - (window << 8));
         checked = nir_bcsel(&b, outside, nir_imm_int(&b, 0), data);
      }
      nir_def_replace(&c->alu->def, checked);
   }
}

bool
agx_nir_lower_word_bounds(nir_shader *shader)
{
   /* nir_shader_get_preamble() would create an empty one. */
   nir_function_impl *entrypoint = nir_shader_get_entrypoint(shader);
   if (!entrypoint->preamble)
      return false;
   nir_function_impl *impl = entrypoint->preamble->impl;

   void *mem = ralloc_context(NULL);
   struct util_dynarray groups;
   util_dynarray_init(&groups, mem);
   collect_checks(impl, &groups, mem);

   bool progress = false;
   util_dynarray_foreach(&groups, struct word_bound_group, g) {
      if (words_cheaper(g)) {
         rewrite_group(g);
         progress = true;
      }
   }

   ralloc_free(mem);
   nir_foreach_function_impl(other, shader) {
      if (other != impl)
         nir_no_progress(other);
   }
   return nir_progress(progress, impl, nir_metadata_control_flow);
}

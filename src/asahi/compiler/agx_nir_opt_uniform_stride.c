/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Rows of a 2D array read at row + c (an unrolled loop's rows, a stencil's
 * neighbours, a GEMM tile's rows) address (x + c) * stride + y; with
 * iadd(imul(iadd(x, #c), s), y) after agx_nir_lower_address, which later
 * fuses into an add and a multiply-add per row. When the stride s is
 * uniform, rewrite it as
 *
 *    (x * s + y) + c * s
 *
 * The first term is common to every row (CSE keeps one), c * s is uniform
 * and moves into the preamble, so each row costs one add. Exact: both sides
 * are the same 32-bit modular sum. Run before the preamble is formed.
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"

struct row {
   nir_alu_instr *add;    /* iadd(imul(iadd(x, #c), s), y) */
   nir_scalar x, s, y;
   uint32_t c;
};

static bool
single_use(nir_def *def)
{
   return list_is_singular(&def->uses);
}

/* imul(iadd(x, #c), s) with s uniform, x divergent, c != 0, each used only
 * here (so the row's own add and multiply go away). */
static bool
match_row(nir_scalar mul, nir_scalar *x, nir_scalar *stride, uint32_t *c)
{
   if (!nir_scalar_is_alu(mul) || nir_scalar_alu_op(mul) != nir_op_imul ||
       mul.def->bit_size != 32 || mul.def->num_components != 1 ||
       !single_use(mul.def))
      return false;
   nir_alu_instr *mul_alu = nir_def_as_alu(mul.def);
   for (unsigned m = 0; m < 2; ++m) {
      nir_scalar sum = nir_scalar_chase_alu_src(mul, m);
      nir_scalar s = nir_scalar_chase_alu_src(mul, 1 - m);
      /* Divergence where the row uses it (the shader is not in LCSSA). */
      if (!nir_scalar_is_alu(sum) || nir_scalar_alu_op(sum) != nir_op_iadd ||
          sum.def->num_components != 1 || !single_use(sum.def) ||
          nir_src_is_divergent(&mul_alu->src[1 - m].src))
         continue;
      nir_alu_instr *sum_alu = nir_def_as_alu(sum.def);
      for (unsigned k = 0; k < 2; ++k) {
         nir_scalar cc = nir_scalar_chase_alu_src(sum, k);
         nir_scalar xx = nir_scalar_chase_alu_src(sum, 1 - k);
         if (nir_scalar_is_const(cc) && nir_scalar_as_uint(cc) != 0 &&
             nir_src_is_divergent(&sum_alu->src[1 - k].src)) {
            *x = xx;
            *stride = s;
            *c = (uint32_t)nir_scalar_as_uint(cc);
            return true;
         }
      }
   }
   return false;
}

/* The row this add is, from the IR as it is now. */
static bool
match_add(nir_alu_instr *alu, struct row *r)
{
   if (alu->op != nir_op_iadd || alu->def.bit_size != 32 ||
       alu->def.num_components != 1)
      return false;
   nir_scalar def = nir_get_scalar(&alu->def, 0);
   for (unsigned m = 0; m < 2; ++m) {
      if (match_row(nir_scalar_chase_alu_src(def, m), &r->x, &r->s, &r->c)) {
         r->add = alu;
         r->y = nir_scalar_chase_alu_src(def, 1 - m);
         return true;
      }
   }
   return false;
}

static bool
same_scalar(nir_scalar a, nir_scalar b)
{
   return a.def == b.def && a.comp == b.comp;
}

/* Rows of one array in one block: iadd(imul(iadd(x, #c), s), y) with the same
 * x, s and y. Two or more become (x * s + y) once, plus a uniform c * s each:
 * the common part fuses into one multiply-add; c * s goes to the preamble. */
static bool
opt_block(nir_builder *b, nir_block *block, struct util_dynarray *rows)
{
   util_dynarray_clear(rows);
   nir_foreach_instr(instr, block) {
      struct row r;
      /* At most 64 rows per block: the grouping below is quadratic. */
      if (instr->type == nir_instr_type_alu &&
          util_dynarray_num_elements(rows, struct row) < 64 &&
          match_add(nir_instr_as_alu(instr), &r))
         util_dynarray_append(rows, r);
   }

   const unsigned n = util_dynarray_num_elements(rows, struct row);
   struct row *r = rows->data;
   bool progress = false;
   for (unsigned i = 0; i < n; ++i) {
      if (!r[i].add)
         continue;
      /* A rewritten group replaced its rows' results, which a later row may
       * use as its x or y: read every remaining row from the IR again. */
      for (unsigned j = i; j < n; ++j) {
         if (r[j].add && !match_add(r[j].add, &r[j]))
            r[j].add = NULL;
      }
      if (!r[i].add)
         continue;
      unsigned group = 0;
      for (unsigned j = i; j < n; ++j)
         group += r[j].add && same_scalar(r[i].x, r[j].x) &&
                  same_scalar(r[i].s, r[j].s) && same_scalar(r[i].y, r[j].y);
      if (group < 2)
         continue;

      /* The common part right before the first row of the group. */
      b->cursor = nir_before_instr(&r[i].add->instr);
      nir_def *s = nir_mov_scalar(b, r[i].s);
      nir_def *base = nir_iadd(b, nir_imul(b, nir_mov_scalar(b, r[i].x), s),
                               nir_mov_scalar(b, r[i].y));
      for (unsigned j = n; j-- > i;) {
         if (!r[j].add || !same_scalar(r[i].x, r[j].x) ||
             !same_scalar(r[i].s, r[j].s) || !same_scalar(r[i].y, r[j].y))
            continue;
         b->cursor = nir_before_instr(&r[j].add->instr);
         nir_def *offset = nir_imul(b, nir_imm_int(b, r[j].c), s);
         nir_def_replace(&r[j].add->def, nir_iadd(b, base, offset));
         r[j].add = NULL;
      }
      progress = true;
   }
   return progress;
}

bool
agx_nir_opt_uniform_stride(nir_shader *shader)
{
   nir_divergence_analysis(shader);
   bool any = false;
   struct util_dynarray rows;
   util_dynarray_init(&rows, NULL);
   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      bool progress = false;
      nir_foreach_block(block, impl)
         progress |= opt_block(&b, block, &rows);
      any |= nir_progress(progress, impl, nir_metadata_control_flow);
   }
   util_dynarray_fini(&rows);
   return any;
}

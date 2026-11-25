/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "util/bitset.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"
#include "agx_nir.h"
#include "nir.h"

/*
 * Share the end computation of robust buffer reads at constant distances.
 *
 * A robust read keeps its value only when its exclusive end lies within the
 * descriptor's byte size:
 *
 *    bounds_agx(data, end, size) = end <= size ? data : 0
 *
 * The reads of one structure in a loop (a light record, a constant buffer
 * row) differ only by a constant field offset, and each computes its own end
 * as a constant > 255, an iadd.nuw chain, or (when the caller guarantees
 * size < UINT32_MAX) a uadd_sat chain:
 *
 *    end = uadd_sat(uadd_sat(X, c), n)   or   iadd.nuw(iadd.nuw(X, c), n)
 *
 * Plain wrapping iadd expressions are left intact so wrap-around semantics
 * are never altered. For the reads of one block with the same X and size,
 * partition the sorted offsets into 255-byte windows, take each window's
 * smallest c, and compute
 *
 *    base = uadd_sat(X, c_min)   (or iadd.nuw(X, c_min) when non-saturating)
 *    R = usub_sat(size, base)
 *
 * once, after which each read in the window checks the immediate
 * c - c_min + n <= 255 against R:
 *
 *    X + c + n <= size  <=>  c - c_min + n <= size - (X + c_min)
 *
 * when size >= base, and both sides are false otherwise (R = 0 while the
 * immediate is at least n > 0). Because saturating ends are matched only when
 * size < UINT32_MAX and wrapping inner adds are never peeled, this rewrite is
 * exact for every defined input.
 */

/* bounds_agx takes its end as an 8-bit immediate. */
#define SHARE_BOUNDS_MAX_IMM 255

struct bound_read {
   nir_alu_instr *alu;
   uint32_t c, n;
   bool saturating;
};

struct bound_group {
   nir_scalar x, size;
   struct util_dynarray reads;
};

static bool
scalar_const(nir_scalar s, uint32_t *value)
{
   if (!nir_scalar_is_const(s))
      return false;
   *value = nir_scalar_as_uint(s);
   return true;
}

/* Match end = const > 255, iadd.nuw(X, k), iadd.nuw(iadd.nuw(X, c), n), or
 * (when size_below_u32_max guarantees size < UINT32_MAX so a saturated
 * UINT32_MAX end is always out of bounds) the uadd_sat equivalents. Inner
 * iadd is peeled only when it carries no_unsigned_wrap so wrapping additions
 * are never turned into saturating bases. */
static bool
match_end(nir_scalar end, bool size_below_u32_max, nir_scalar *x, uint32_t *c,
          uint32_t *n, bool *saturating)
{
   uint32_t k = 0;
   if (scalar_const(end, &k)) {
      if (k <= SHARE_BOUNDS_MAX_IMM)
         return false;
      *x = (nir_scalar){0};
      *c = k - 1;
      *n = 1;
      *saturating = false;
      return true;
   }

   if (!nir_scalar_is_alu(end) || end.def->bit_size != 32)
      return false;

   nir_op op = nir_scalar_alu_op(end);
   bool outer_sat = op == nir_op_uadd_sat;
   bool outer_nuw =
      op == nir_op_iadd &&
      nir_instr_as_alu(nir_def_instr(end.def))->no_unsigned_wrap;
   if (!(outer_nuw || (outer_sat && size_below_u32_max)))
      return false;

   nir_scalar a = nir_scalar_chase_alu_src(end, 0);
   nir_scalar b = nir_scalar_chase_alu_src(end, 1);
   nir_scalar inner;
   if (scalar_const(b, n))
      inner = a;
   else if (scalar_const(a, n))
      inner = b;
   else
      return false;

   if (*n == 0)
      return false;

   *c = 0;
   *x = inner;
   bool inner_sat = false;
   if (nir_scalar_is_alu(inner)) {
      nir_op iop = nir_scalar_alu_op(inner);
      bool can_peel =
         (iop == nir_op_uadd_sat && size_below_u32_max) ||
         (iop == nir_op_iadd &&
          nir_instr_as_alu(nir_def_instr(inner.def))->no_unsigned_wrap);
      if (can_peel) {
         nir_scalar p = nir_scalar_chase_alu_src(inner, 0);
         nir_scalar q = nir_scalar_chase_alu_src(inner, 1);
         uint32_t c0 = 0;
         if (scalar_const(q, &c0)) {
            *x = p;
            *c = c0;
            inner_sat = iop == nir_op_uadd_sat;
         } else if (scalar_const(p, &c0)) {
            *x = q;
            *c = c0;
            inner_sat = iop == nir_op_uadd_sat;
         }
      }
   }

   if (*n > SHARE_BOUNDS_MAX_IMM) {
      if (*c == 0) {
         *c = *n - 1;
         *n = 1;
      } else {
         return false;
      }
   }

   *saturating = outer_sat || inner_sat;
   return !nir_scalar_is_const(*x);
}

static uint32_t
group_hash(const void *key)
{
   const struct bound_group *g = key;
   uint32_t h = _mesa_hash_pointer(g->x.def) ^ (g->x.comp * 31);
   return h ^ (_mesa_hash_pointer(g->size.def) * 7) ^ g->size.comp;
}

static bool
group_equal(const void *a, const void *b)
{
   const struct bound_group *x = a, *y = b;
   return nir_scalar_equal(x->x, y->x) && nir_scalar_equal(x->size, y->size);
}

static int
bound_read_cmp(const void *pa, const void *pb)
{
   const struct bound_read *a = pa, *b = pb;
   if (a->c != b->c)
      return a->c < b->c ? -1 : 1;
   if (a->n != b->n)
      return a->n < b->n ? -1 : 1;
   return (a->alu->instr.index > b->alu->instr.index) -
          (a->alu->instr.index < b->alu->instr.index);
}

static bool
share_group(nir_builder *b, struct bound_group *g, void *mem_ctx)
{
   unsigned count = util_dynarray_num_elements(&g->reads, struct bound_read);
   if (count < 2)
      return false;

   struct bound_read *reads = g->reads.data;
   qsort(reads, count, sizeof(*reads), bound_read_cmp);

   BITSET_WORD *done = BITSET_RZALLOC(mem_ctx, count);
   unsigned *win = ralloc_array(mem_ctx, unsigned, count);
   bool progress = false;

   for (unsigned i = 0; i < count; i++) {
      if (BITSET_TEST(done, i))
         continue;

      uint32_t c_min = reads[i].c;
      unsigned win_count = 0;
      bool any_sat = false;
      for (unsigned j = i; j < count; j++) {
         if (BITSET_TEST(done, j))
            continue;
         if ((uint64_t)reads[j].c - c_min >= SHARE_BOUNDS_MAX_IMM)
            break;
         if ((uint64_t)reads[j].c - c_min + reads[j].n <= SHARE_BOUNDS_MAX_IMM) {
            win[win_count++] = j;
            any_sat |= reads[j].saturating;
         }
      }

      if (win_count < 2)
         continue;

      unsigned earliest = win[0];
      for (unsigned k = 1; k < win_count; k++) {
         if (reads[win[k]].alu->instr.index < reads[earliest].alu->instr.index)
            earliest = win[k];
      }

      b->cursor = nir_before_instr(&reads[earliest].alu->instr);

      nir_def *base;
      if (g->x.def) {
         nir_def *x = nir_mov_scalar(b, g->x);
         if (!c_min) {
            base = x;
         } else if (any_sat) {
            base = nir_uadd_sat(b, x, nir_imm_int(b, c_min));
         } else {
            base = nir_iadd_imm(b, x, c_min);
            nir_instr_as_alu(nir_def_instr(base))->no_unsigned_wrap = true;
         }
      } else {
         base = nir_imm_int(b, c_min);
      }
      nir_def *rest = nir_usub_sat(b, nir_mov_scalar(b, g->size), base);

      for (unsigned k = 0; k < win_count; k++) {
         unsigned idx = win[k];
         uint64_t imm = (uint64_t)reads[idx].c - c_min + reads[idx].n;
         b->cursor = nir_before_instr(&reads[idx].alu->instr);
         nir_src_rewrite(&reads[idx].alu->src[1].src, nir_imm_int(b, imm));
         reads[idx].alu->src[1].swizzle[0] = 0;
         nir_src_rewrite(&reads[idx].alu->src[2].src, rest);
         reads[idx].alu->src[2].swizzle[0] = 0;
         BITSET_SET(done, idx);
      }

      progress = true;
   }

   return progress;
}

static bool
share_block(nir_builder *b, nir_block *block, bool size_below_u32_max,
            void *mem_ctx)
{
   struct set *groups = _mesa_set_create(mem_ctx, group_hash, group_equal);
   struct util_dynarray order;
   util_dynarray_init(&order, mem_ctx);

   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op != nir_op_bounds_agx || alu->def.bit_size != 32 ||
          alu->def.num_components != 1)
         continue;

      nir_scalar end = nir_scalar_resolved(alu->src[1].src.ssa,
                                           alu->src[1].swizzle[0]);
      nir_scalar size = nir_scalar_resolved(alu->src[2].src.ssa,
                                            alu->src[2].swizzle[0]);
      if (size.def->bit_size != 32)
         continue;

      struct bound_read read = {.alu = alu};
      nir_scalar x;
      if (!match_end(end, size_below_u32_max, &x, &read.c, &read.n,
                     &read.saturating))
         continue;

      struct bound_group key = {.x = x, .size = size};
      bool found = false;
      struct set_entry *e = _mesa_set_search_or_add(groups, &key, &found);
      struct bound_group *g;
      if (!found) {
         g = ralloc(mem_ctx, struct bound_group);
         *g = key;
         util_dynarray_init(&g->reads, mem_ctx);
         e->key = g;
         util_dynarray_append(&order, g);
      } else {
         g = (struct bound_group *)e->key;
      }
      util_dynarray_append(&g->reads, read);
   }

   bool progress = false;
   util_dynarray_foreach(&order, struct bound_group *, g)
      progress |= share_group(b, *g, mem_ctx);

   return progress;
}

bool
agx_nir_share_bounds(nir_shader *s, bool size_below_u32_max)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   if (!impl)
      return false;

   void *mem_ctx = ralloc_context(NULL);
   nir_builder b = nir_builder_create(impl);
   bool progress = false;

   nir_index_instrs(impl);

   nir_foreach_block(block, impl)
      progress |= share_block(&b, block, size_below_u32_max, mem_ctx);

   ralloc_free(mem_ctx);
   nir_foreach_function_impl(other, s) {
      if (other != impl)
         nir_no_progress(other);
   }
   return nir_progress(progress, impl, nir_metadata_control_flow);
}

/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"
#include "agx_compiler.h"
#include "agx_nir.h"
#include <stdlib.h>

static nir_def *divide_exact(nir_builder *b, nir_scalar s, unsigned shift,
                             unsigned depth);

/* Local memory instructions require 16-bit offsets, so we add conversions. */
static bool
pass(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_src *offset = nir_get_io_offset_src(intr);
   if (!nir_is_shared_access(intr))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *address = offset->ssa;

   /* AGX3 ldtg/sttg register indices count scalar elements, while AGX2's
    * local-memory base remains byte addressed. Keep atomics on their separate
    * byte-addressed encoding and convert only ordinary loads/stores. */
   const bool element_indices = *(const bool *)data;
   if (element_indices &&
       (intr->intrinsic == nir_intrinsic_load_shared ||
        intr->intrinsic == nir_intrinsic_store_shared)) {
      const unsigned bit_size =
         intr->intrinsic == nir_intrinsic_load_shared
            ? intr->def.bit_size
            : nir_src_bit_size(intr->src[0]);
      const unsigned shift = util_logbase2(bit_size / 8);
      /* Divide the byte arithmetic itself where that is exact, so a scaled
       * thread index becomes an element index without a shift per access. */
      nir_def *elements =
         shift ? divide_exact(b, nir_get_scalar(address, 0), shift, 4) : NULL;
      address = elements ? elements : nir_ushr_imm(b, address, shift);
   }

   if (address->bit_size != 16)
      address = nir_u2u16(b, address);

   if (address == offset->ssa)
      return false;

   nir_src_rewrite(offset, address);
   return true;
}

bool
agx_nir_lower_shared_bitsize(nir_shader *shader, bool element_indices)
{
   return nir_shader_intrinsics_pass(shader, pass, nir_metadata_control_flow,
                                     &element_indices);
}

/*
 * Divide `s` by 2^shift exactly, or return NULL when that cannot be done
 * without a shift instruction.  Every rewrite here is exact for every value of
 * the input: a constant divides only when it has no low bits, a left shift only
 * when it shifts at least as far, a multiply only when its constant factor is a
 * multiple of the divisor.  This is the fold nir_opt_algebraic would perform
 * had the conversion run before it -- doing it here keeps the pass usable after
 * the optimizer, where `ushr(ishl(a, 2), 2)` is instead rewritten into a mixed
 * width `iand` that the AGX3 encoder has no form for.
 */
static nir_def *
divide_exact(nir_builder *b, nir_scalar s, unsigned shift, unsigned depth)
{
   const uint64_t mask = (UINT64_C(1) << shift) - 1;

   if (nir_scalar_is_const(s)) {
      const uint64_t c = nir_scalar_as_uint(s);
      if (c & mask)
         return NULL;
      return nir_imm_intN_t(b, c >> shift, s.def->bit_size);
   }

   if (depth == 0 || !nir_scalar_is_alu(s))
      return NULL;

   switch (nir_scalar_alu_op(s)) {
   case nir_op_ishl: {
      const nir_scalar amount = nir_scalar_chase_alu_src(s, 1);
      if (!nir_scalar_is_const(amount))
         return NULL;
      const uint64_t k = nir_scalar_as_uint(amount);
      if (k < shift)
         return NULL;
      const nir_scalar value = nir_scalar_chase_alu_src(s, 0);
      nir_def *kept = nir_channel(b, value.def, value.comp);
      return k == shift ? kept : nir_ishl_imm(b, kept, k - shift);
   }

   case nir_op_amul:
   case nir_op_imul: {
      for (unsigned i = 0; i < 2; ++i) {
         const nir_scalar factor = nir_scalar_chase_alu_src(s, i);
         if (!nir_scalar_is_const(factor))
            continue;
         const uint64_t f = nir_scalar_as_uint(factor);
         if (f & mask)
            return NULL;
         const nir_scalar value = nir_scalar_chase_alu_src(s, 1 - i);
         return nir_imul_imm(b, nir_channel(b, value.def, value.comp),
                             f >> shift);
      }
      return NULL;
   }

   case nir_op_imadshl_agx:
   case nir_op_imsubshl_agx: {
      /* (src0 * src1) +/- (src2 << src3): the shifted term divides when it
       * shifts at least as far, the product when either factor is a constant
       * multiple of the divisor (or zero). */
      const nir_scalar amount = nir_scalar_chase_alu_src(s, 3);
      if (!nir_scalar_is_const(amount))
         return NULL;
      const uint64_t k = nir_scalar_as_uint(amount);
      if (k < shift)
         return NULL;
      nir_scalar factor[2] = {nir_scalar_chase_alu_src(s, 0),
                              nir_scalar_chase_alu_src(s, 1)};
      nir_def *f[2] = {nir_channel(b, factor[0].def, factor[0].comp),
                       nir_channel(b, factor[1].def, factor[1].comp)};
      bool divided = false;
      for (unsigned i = 0; i < 2 && !divided; ++i) {
         if (!nir_scalar_is_const(factor[i]))
            continue;
         const uint64_t c = nir_scalar_as_uint(factor[i]);
         if (c & mask)
            continue;
         f[i] = nir_imm_intN_t(b, c >> shift, s.def->bit_size);
         divided = true;
      }
      if (!divided)
         return NULL;
      const nir_scalar value = nir_scalar_chase_alu_src(s, 2);
      nir_def *v = nir_channel(b, value.def, value.comp);
      nir_def *k_s = nir_imm_int(b, k - shift);
      return nir_scalar_alu_op(s) == nir_op_imadshl_agx
                ? nir_imadshl_agx(b, f[0], f[1], v, k_s)
                : nir_imsubshl_agx(b, f[0], f[1], v, k_s);
   }

   case nir_op_iadd: {
      nir_def *x =
         divide_exact(b, nir_scalar_chase_alu_src(s, 0), shift, depth - 1);
      if (!x)
         return NULL;
      nir_def *y =
         divide_exact(b, nir_scalar_chase_alu_src(s, 1), shift, depth - 1);
      return y ? nir_iadd(b, x, y) : NULL;
   }

   default:
      return NULL;
   }
}

/*
 * AGX3 ldtg/sttg address threadgroup memory as
 *
 *    base + (index << shift) + byte_offset
 *
 * where `shift` is the access width -- agx3_translate_local() sets
 * shift = width (agx3_pack_binary.c), grounded against Apple's own output.
 * The index register
 * therefore counts *scalar elements*, while NIR's shared offsets count bytes.
 * AGX2's local-memory base stays byte addressed and AGX3 threadgroup atomics
 * keep their own byte-addressed immediate field (agx_local_atomic_base), so
 * only ordinary AGX3 loads and stores convert.
 *
 * Getting this wrong is silent: the instruction encodes either way and the
 * hardware simply strides four times too far for a 32-bit element, so a kernel
 * can only use the first quarter of its threadgroup block.  That is what every
 * internal (libagx) kernel did on G17P, because the conversion used to live
 * behind agx_preprocess_nir_g17p() and asahi_clc -- which preprocesses one
 * shader for g13g, g13x and g17p at once -- cannot call it.
 */
static bool
element_index(struct nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_shared &&
       intr->intrinsic != nir_intrinsic_store_shared)
      return false;

   const unsigned bit_size = intr->intrinsic == nir_intrinsic_load_shared
                                ? intr->def.bit_size
                                : nir_src_bit_size(intr->src[0]);
   const unsigned shift = util_logbase2(bit_size / 8);
   if (shift == 0)
      return false;

   nir_src *offset = nir_get_io_offset_src(intr);
   nir_def *address = offset->ssa;

   b->cursor = nir_before_instr(&intr->instr);

   /* agx_nir_lower_shared_bitsize() may already have narrowed the byte offset
    * to 16 bits.  Divide the value it was narrowed from, so the arithmetic
    * that produced the byte offset stays visible and the narrowing is applied
    * to the element index, exactly as when this conversion runs first. */
   nir_scalar narrow = nir_scalar_resolved(address, 0);
   if (nir_scalar_is_alu(narrow) && nir_scalar_alu_op(narrow) == nir_op_u2u16) {
      const nir_scalar wide = nir_scalar_chase_alu_src(narrow, 0);
      nir_def *elements = divide_exact(b, wide, shift, 4);
      if (!elements) {
         elements =
            nir_ushr_imm(b, nir_channel(b, wide.def, wide.comp), shift);
      }
      nir_src_rewrite(offset, nir_u2u16(b, elements));
      return true;
   }

   nir_def *elements = divide_exact(b, narrow, shift, 4);
   nir_src_rewrite(offset, elements ? elements
                                    : nir_ushr_imm(b, address, shift));
   return true;
}

bool
agx_nir_lower_shared_element_index(nir_shader *shader)
{
   /* Records that the conversion is now owned, so the backstop in
    * agx_compile_shader_nir() does not apply it a second time. */
   shader->info.shared_element_index_agx = true;
   return nir_shader_intrinsics_pass(shader, element_index,
                                     nir_metadata_control_flow, NULL);
}

/* Flatten a 32-bit iadd tree into at most four non-constant terms and the sum
 * of its constants. */
static bool
gather_sum(nir_scalar s, unsigned depth, nir_scalar *terms, unsigned *count,
           uint64_t *constant, bool limit_u16)
{
   if (nir_scalar_is_const(s)) {
      *constant += nir_scalar_as_uint(s);
      return !limit_u16 || *constant <= UINT16_MAX;
   }
   if (depth && nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd) {
      return gather_sum(nir_scalar_chase_alu_src(s, 0), depth - 1, terms, count,
                        constant, limit_u16) &&
             gather_sum(nir_scalar_chase_alu_src(s, 1), depth - 1, terms, count,
                        constant, limit_u16);
   }
   if (*count == 4)
      return false;
   terms[(*count)++] = s;
   return true;
}

/* nir_unsigned_upper_bound() with the AGX address multiply-add understood:
 * (src0 * src1) + (src2 << src3), bounded termwise. */
static uint64_t
term_upper_bound(nir_shader *shader, struct hash_table *ranges, nir_scalar s,
                 unsigned depth)
{
   if (depth && nir_scalar_is_alu(s) &&
       nir_scalar_alu_op(s) == nir_op_imadshl_agx) {
      const nir_scalar amount = nir_scalar_chase_alu_src(s, 3);
      if (!nir_scalar_is_const(amount) || nir_scalar_as_uint(amount) > 31)
         return UINT32_MAX;
      const uint64_t x =
         term_upper_bound(shader, ranges, nir_scalar_chase_alu_src(s, 0), depth - 1);
      const uint64_t y =
         term_upper_bound(shader, ranges, nir_scalar_chase_alu_src(s, 1), depth - 1);
      const uint64_t z =
         term_upper_bound(shader, ranges, nir_scalar_chase_alu_src(s, 2), depth - 1);
      const unsigned k = nir_scalar_as_uint(amount);
      if ((x && y > UINT32_MAX / x) || z > (UINT32_MAX >> k))
         return UINT32_MAX;
      return MIN2(x * y + (z << k), UINT32_MAX);
   }
   return nir_unsigned_upper_bound(shader, ranges, s);
}

static int
compare_scalars(const void *a, const void *b)
{
   const nir_scalar *x = a, *y = b;
   if (x->def->index != y->def->index)
      return x->def->index < y->def->index ? -1 : 1;
   return (int)x->comp - (int)y->comp;
}

struct elem_candidate {
   nir_intrinsic_instr *intr;
   nir_block *block;
   unsigned segment;
   nir_scalar sum;
   nir_scalar terms[4];
   unsigned count;
   uint64_t elements;
   uint64_t e_hi;
   unsigned e_lo;
   unsigned shift;
   unsigned base;
   bool bounded;
   bool rewritten;
};

static bool
same_terms(const struct elem_candidate *a, const struct elem_candidate *b)
{
   if (a->count != b->count)
      return false;
   for (unsigned t = 0; t < a->count; ++t) {
      if (a->terms[t].def != b->terms[t].def ||
          a->terms[t].comp != b->terms[t].comp)
         return false;
   }
   return true;
}

static bool
sum_matches_canonical(nir_scalar sum, const nir_scalar *terms, unsigned count)
{
   if (count != 2 || !nir_scalar_is_alu(sum) ||
       nir_scalar_alu_op(sum) != nir_op_iadd)
      return false;
   nir_scalar s0 = nir_scalar_chase_alu_src(sum, 0);
   nir_scalar s1 = nir_scalar_chase_alu_src(sum, 1);
   return s0.def == terms[0].def && s0.comp == terms[0].comp &&
          s1.def == terms[1].def && s1.comp == terms[1].comp;
}

static bool
analyze_elem_candidate(nir_shader *shader, struct hash_table *ranges,
                       nir_block *block, unsigned segment,
                       nir_intrinsic_instr *intr, struct elem_candidate *out)
{
   if (intr->intrinsic != nir_intrinsic_load_shared &&
       intr->intrinsic != nir_intrinsic_store_shared)
      return false;

   const unsigned bits = intr->intrinsic == nir_intrinsic_load_shared
                            ? intr->def.bit_size
                            : nir_src_bit_size(intr->src[0]);
   if (bits != 16 && bits != 32)
      return false;

   const unsigned shift = util_logbase2(bits / 8);
   nir_src *offset = nir_get_io_offset_src(intr);
   nir_scalar s = nir_scalar_resolved(offset->ssa, 0);
   if (s.def->bit_size != 16 || !nir_scalar_is_alu(s) ||
       nir_scalar_alu_op(s) != nir_op_u2u16)
      return false;

   s = nir_scalar_chase_alu_src(s, 0);
   const unsigned base = nir_intrinsic_base(intr);
   const bool small_shared =
      shader->info.shared_size > 0 && shader->info.shared_size <= 32768;
   if (base > 255 || s.def->bit_size != 32)
      return false;

   const bool is_iadd =
      nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_iadd;
   if (!is_iadd && !small_shared)
      return false;

   unsigned count = 0;
   uint64_t elements = 0;
   if (!gather_sum(s, 5, out->terms, &count, &elements, !small_shared) || !count)
      return false;

   const unsigned max_lo = (255u - base) >> shift;
   uint64_t e_hi = 0;
   unsigned e_lo = 0;
   bool bounded = false;
   if (is_iadd && elements <= UINT16_MAX) {
      bool split_ok = true;
      if (elements <= max_lo) {
         e_lo = elements;
      } else {
         const unsigned chunk = 256u >> shift;
         e_hi = (elements / chunk) * chunk;
         e_lo = elements - e_hi;
         if (e_lo > max_lo)
            split_ok = false;
      }
      if (split_ok) {
         uint64_t bound = 0;
         for (unsigned t = 0; t < count; ++t)
            bound += term_upper_bound(shader, ranges, out->terms[t], 2);
         bounded = bound <= UINT16_MAX - elements;
      }
   }

   if (!bounded && !small_shared)
      return false;

   qsort(out->terms, count, sizeof(out->terms[0]), compare_scalars);
   out->intr = intr;
   out->block = block;
   out->segment = segment;
   out->sum = s;
   out->count = count;
   out->elements = elements;
   out->e_hi = e_hi;
   out->e_lo = e_lo;
   out->shift = shift;
   out->base = base;
   out->bounded = bounded;
   out->rewritten = false;
   return true;
}

/* Element-index lowering leaves byte additions beneath the shift/narrowing:
 * u2u16((byte_address + constant) >> log2(element_size)).  Extract a small,
 * aligned constant into LDTG/STTG's byte offset.  Both the source addition and
 * the narrowed element index must be proven not to wrap: moving an add across
 * either wrap would change the accessed address. */
static bool
fold_shifted_byte_offset(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_shared &&
       intr->intrinsic != nir_intrinsic_store_shared)
      return false;

   const unsigned bits = intr->intrinsic == nir_intrinsic_load_shared
                            ? intr->def.bit_size
                            : nir_src_bit_size(intr->src[0]);
   if (bits != 16 && bits != 32)
      return false;
   const unsigned shift = util_logbase2(bits / 8);
   nir_src *offset = nir_get_io_offset_src(intr);
   nir_scalar s = nir_scalar_resolved(offset->ssa, 0);
   if (s.def->bit_size != 16 || !nir_scalar_is_alu(s) ||
       nir_scalar_alu_op(s) != nir_op_u2u16)
      return false;
   s = nir_scalar_chase_alu_src(s, 0);
   const unsigned base = nir_intrinsic_base(intr);
   if (base > 255)
      return false;

   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_ushr)
      return false;
   nir_scalar amount = nir_scalar_chase_alu_src(s, 1);
   if (!nir_scalar_is_const(amount) || nir_scalar_as_uint(amount) != shift)
      return false;
   s = nir_scalar_chase_alu_src(s, 0);
   if (s.def->bit_size != 32 || !nir_scalar_is_alu(s) ||
       nir_scalar_alu_op(s) != nir_op_iadd)
      return false;

   for (unsigned c = 0; c < 2; ++c) {
      nir_scalar constant = nir_scalar_chase_alu_src(s, c);
      if (!nir_scalar_is_const(constant))
         continue;
      const uint32_t add = nir_scalar_as_uint(constant);
      if (!add || add > 255 - base || (add & ((1u << shift) - 1)))
         continue;
      nir_scalar variable = nir_scalar_chase_alu_src(s, 1 - c);
      const uint32_t limit = ((UINT32_C(1) << (16 + shift)) - 1) - add;
      if (nir_unsigned_upper_bound(b->shader, data, variable) > limit)
         continue;

      b->cursor = nir_before_instr(&intr->instr);
      nir_def *index = nir_ushr_imm(b, nir_mov_scalar(b, variable), shift);
      nir_src_rewrite(offset, nir_u2u16(b, index));
      nir_intrinsic_set_base(intr, base + add);
      return true;
   }
   return false;
}

/* Fold exact-divided element index sums into LDTG/STTG's 8-bit byte offset.
 * For clusters of accesses sharing the same variable terms across multiple
 * 256-byte windows (such as strided vertical stencil taps), split
 * elements = e_hi + e_lo into a shared chunk base (V + e_hi) plus immediate
 * byte offset (e_lo << shift) so up to 256 / elem_size taps share one add. */
static bool
fold_elem_offsets_impl(nir_function_impl *impl, struct hash_table *ranges)
{
   struct util_dynarray cands;
   util_dynarray_init(&cands, NULL);

   nir_shader *shader = impl->function->shader;
   const bool small_shared =
      shader->info.shared_size > 0 && shader->info.shared_size <= 32768;

   nir_foreach_block(block, impl) {
      unsigned segment = 0;
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_terminate ||
             intr->intrinsic == nir_intrinsic_terminate_if ||
             intr->intrinsic == nir_intrinsic_demote ||
             intr->intrinsic == nir_intrinsic_demote_if) {
            segment++;
            continue;
         }
         struct elem_candidate cand;
         if (analyze_elem_candidate(shader, ranges, block, segment, intr, &cand))
            util_dynarray_append(&cands, cand);
      }
   }

   const unsigned num_cands =
      util_dynarray_num_elements(&cands, struct elem_candidate);
   if (!num_cands) {
      util_dynarray_fini(&cands);
      return false;
   }

   struct elem_candidate *arr = cands.data;
   nir_builder b = nir_builder_create(impl);
   bool progress = false;

   /* When shared memory fits in 32 KiB, every in-bounds 16/32-bit element
    * index is strictly below 2^14 (shift >= 1), so signed 16-bit deltas
    * between same-block accesses sharing the same variable terms are exact
    * and anchor + delta cannot wrap 16 bits. Greedily anchor each 256-byte
    * window at its lowest executed member so offsets fold into LDTG/STTG.base
    * even when variable terms include preamble uniforms or subgroup IDs. */
   if (small_shared) {
      bool *visited = calloc(num_cands, sizeof(bool));
      for (unsigned i = 0; i < num_cands; ++i) {
         if (visited[i])
            continue;
         unsigned members[512];
         int rel[512];
         unsigned m = 0;
         for (unsigned j = i; j < num_cands && m < ARRAY_SIZE(members); ++j) {
            if (visited[j])
               continue;
            if (arr[j].block != arr[i].block ||
                arr[j].segment != arr[i].segment ||
                arr[j].shift != arr[i].shift ||
                !same_terms(&arr[i], &arr[j]))
               continue;
            visited[j] = true;
            rel[m] = (int16_t)(uint16_t)(arr[j].elements - arr[i].elements);
            members[m++] = j;
         }
         if (m < 2)
            continue;

         /* Insertion sort by relative offset ascending. */
         for (unsigned a = 1; a < m; ++a) {
            int r = rel[a];
            unsigned idx = members[a];
            unsigned k = a;
            while (k > 0 && rel[k - 1] > r) {
               rel[k] = rel[k - 1];
               members[k] = members[k - 1];
               k--;
            }
            rel[k] = r;
            members[k] = idx;
         }

         const int max_span = (int)(32768u >> arr[i].shift);
         if (rel[m - 1] - rel[0] >= max_span)
            continue;

         unsigned start = 0;
         while (start < m) {
            struct elem_candidate *anchor = &arr[members[start]];
            const int anchor_rel = rel[start];
            unsigned end = start + 1;
            while (end < m) {
               struct elem_candidate *cand = &arr[members[end]];
               const unsigned delta = (unsigned)(rel[end] - anchor_rel);
               if (cand->base + (delta << cand->shift) > 255u)
                  break;
               end++;
            }
            if (end - start >= 2) {
               const uint32_t anchor_c = (uint16_t)anchor->elements;
               for (unsigned k = start; k < end; ++k) {
                  struct elem_candidate *cand = &arr[members[k]];
                  const unsigned delta = (unsigned)(rel[k] - anchor_rel);
                  cand->rewritten = true;
                  if (k == start && cand->count == 1 && anchor_c == 0)
                     continue;
                  b.cursor = nir_before_instr(&cand->intr->instr);
                  nir_def *variable = nir_mov_scalar(&b, cand->terms[0]);
                  for (unsigned t = 1; t < cand->count; ++t)
                     variable =
                        nir_iadd(&b, variable, nir_mov_scalar(&b, cand->terms[t]));
                  if (anchor_c > 0)
                     variable = nir_iadd_imm(&b, variable, anchor_c);
                  nir_src_rewrite(nir_get_io_offset_src(cand->intr),
                                  nir_u2u16(&b, variable));
                  if (delta > 0)
                     nir_intrinsic_set_base(cand->intr,
                                            cand->base + (delta << cand->shift));
                  progress = true;
               }
            }
            start = end;
         }
      }
      free(visited);
   }

   for (unsigned i = 0; i < num_cands; ++i) {
      const struct elem_candidate *c = &arr[i];
      if (!c->bounded || c->rewritten)
         continue;
      unsigned var_uses = 0;
      unsigned bucket_uses = 0;
      for (unsigned j = 0; j < num_cands; ++j) {
         if (!arr[j].bounded || arr[j].rewritten || !same_terms(c, &arr[j]))
            continue;
         var_uses++;
         if (c->e_hi == arr[j].e_hi)
            bucket_uses++;
      }

      bool rewrite = false;
      if (c->e_hi == 0) {
         if (c->e_lo > 0)
            rewrite = true;
         else if (c->count > 1 && var_uses >= 2 &&
                  !sum_matches_canonical(c->sum, c->terms, c->count))
            rewrite = true;
      } else if (bucket_uses >= 2) {
         rewrite = true;
      }

      if (!rewrite)
         continue;

      b.cursor = nir_before_instr(&c->intr->instr);
      nir_def *variable = nir_mov_scalar(&b, c->terms[0]);
      for (unsigned t = 1; t < c->count; ++t)
         variable = nir_iadd(&b, variable, nir_mov_scalar(&b, c->terms[t]));
      if (c->e_hi > 0)
         variable = nir_iadd_imm(&b, variable, c->e_hi);
      nir_src_rewrite(nir_get_io_offset_src(c->intr), nir_u2u16(&b, variable));
      if (c->e_lo > 0)
         nir_intrinsic_set_base(c->intr, c->base + (c->e_lo << c->shift));
      progress = true;
   }

   util_dynarray_fini(&cands);
   return nir_progress(progress, impl, nir_metadata_control_flow);
}

bool
agx_nir_opt_shared_byte_offsets(nir_shader *shader)
{
   struct hash_table *ranges = _mesa_pointer_hash_table_create(NULL);
   bool progress = false;
   nir_foreach_function_impl(impl, shader)
      progress |= fold_elem_offsets_impl(impl, ranges);
   progress |= nir_shader_intrinsics_pass(shader, fold_shifted_byte_offset,
                                          nir_metadata_control_flow, ranges);
   _mesa_hash_table_destroy(ranges, NULL);
   return progress;
}

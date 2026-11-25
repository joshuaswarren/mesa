/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Merge reads of consecutive 32-bit words of one buffer into vector loads.
 *
 * Buffer reads address base + u2u64(X + c) with a 32-bit offset X + c. Two
 * such reads of words c and c + 4 are adjacent in memory unless X + c wraps
 * 2^32 between them; such an offset lies past the end of any buffer, so a
 * shader that executes both reads has undefined behaviour whenever they are
 * not adjacent. nir_opt_load_store_vectorize cannot use that argument, since
 * it does not know that every read it merges executes: it asks for a proof
 * that X + c does not wrap, which a loop's induction variable never has.
 *
 * This pass merges only reads that all execute together: plain (unbounded,
 * robustness-free, neither volatile, coherent nor speculatable, so never
 * hoisted out of the code that guarded them) reads in one block, with
 * nothing between them that can end the invocation and, unless they may be
 * reordered, nothing that writes memory. The merged load sits at the earliest of the
 * reads and takes its offset from that read's own offset term, so it needs
 * no value computed later.
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"
#include "agx_nir_linear.h"

struct word_read {
   nir_intrinsic_instr *load;
   nir_def *base;
   nir_scalar offset_scalar;
   struct lin offset;
};

static bool
is_op(nir_scalar s, nir_op op)
{
   return nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == op;
}

/* base + u2u64(offset) with a 32-bit offset into a buffer binding (see
 * agx_lin_binding_base: its range is what makes a wrapping offset invalid). */
static bool
split_address(nir_def *addr, nir_def **base, nir_scalar *offset)
{
   nir_scalar s = nir_scalar_resolved(addr, 0);
   if (addr->bit_size != 64 || addr->num_components != 1 || !is_op(s, nir_op_iadd))
      return false;

   for (unsigned i = 0; i < 2; i++) {
      nir_scalar u = nir_scalar_chase_alu_src(s, i);
      nir_scalar other = nir_scalar_chase_alu_src(s, 1 - i);
      if (is_op(u, nir_op_u2u64) && !is_op(other, nir_op_u2u64) &&
          other.def->num_components == 1) {
         nir_scalar off = nir_scalar_chase_alu_src(u, 0);
         if (off.def->bit_size == 32 && agx_lin_binding_base(other.def)) {
            *base = other.def;
            *offset = off;
            return true;
         }
      }
   }
   return false;
}

static int
read_cmp(const void *pa, const void *pb)
{
   const struct word_read *a = pa, *b = pb;
#define CMP(u, v)                                                             \
   if ((u) != (v))                                                            \
      return (u) < (v) ? -1 : 1
   CMP(a->load->intrinsic, b->load->intrinsic);
   CMP(a->base->index, b->base->index);
   CMP(nir_intrinsic_access(a->load), nir_intrinsic_access(b->load));
   int t = lin_cmp_terms(&a->offset, &b->offset);
   if (t)
      return t;
   CMP(a->offset.constant, b->offset.constant);
   CMP(a->load->instr.index, b->load->instr.index);
#undef CMP
   return 0;
}

static bool
same_record(const struct word_read *a, const struct word_read *b)
{
   return a->load->intrinsic == b->load->intrinsic && a->base == b->base &&
          nir_intrinsic_access(a->load) == nir_intrinsic_access(b->load) &&
          !lin_cmp_terms(&a->offset, &b->offset);
}

/* Nothing between the reads may end the invocation early, nor, for reads
 * that may not be reordered, write memory. */
static bool
all_execute(nir_instr *first, nir_instr *last, bool reorderable)
{
   for (nir_instr *instr = nir_instr_next(first); instr != last;
        instr = nir_instr_next(instr)) {
      if (instr->type == nir_instr_type_call || instr->type == nir_instr_type_jump)
         return false;
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (!reorderable &&
          !(nir_intrinsic_infos[intr->intrinsic].flags & NIR_INTRINSIC_CAN_ELIMINATE))
         return false;
      switch (intr->intrinsic) {
      case nir_intrinsic_terminate:
      case nir_intrinsic_terminate_if:
      case nir_intrinsic_demote:
      case nir_intrinsic_demote_if:
         return false;
      default:
         break;
      }
   }
   return true;
}

static bool
merge_block(nir_block *block, struct util_dynarray *reads)
{
   util_dynarray_clear(reads);
   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
      if ((in->intrinsic != nir_intrinsic_load_global &&
           in->intrinsic != nir_intrinsic_load_global_constant) ||
          in->num_components != 1 || in->def.bit_size != 32 ||
          (nir_intrinsic_access(in) &
           (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_CAN_SPECULATE)) ||
          nir_intrinsic_align(in) < 4)
         continue;

      struct word_read r = {.load = in};
      if (!split_address(in->src[0].ssa, &r.base, &r.offset_scalar))
         continue;
      r.offset = (struct lin){.valid = true};
      lin_parse(&r.offset, r.offset_scalar, 1, 12);
      if (r.offset.valid && r.offset.n) {
         lin_normalize(&r.offset);
         util_dynarray_append(reads, r);
      }
   }

   unsigned n = util_dynarray_num_elements(reads, struct word_read);
   if (n < 2)
      return false;

   struct word_read *r = reads->data;
   qsort(r, n, sizeof(*r), read_cmp);

   bool progress = false;
   for (unsigned i = 0; i < n;) {
      unsigned count = 1;
      while (i + count < n && count < 4 && same_record(&r[i], &r[i + count]) &&
             r[i + count].offset.constant ==
                r[i + count - 1].offset.constant + 4)
         count++;

      if (count < 2) {
         i++;
         continue;
      }

      unsigned first = i, last = i;
      for (unsigned j = i + 1; j < i + count; j++) {
         if (r[j].load->instr.index < r[first].load->instr.index)
            first = j;
         if (r[j].load->instr.index > r[last].load->instr.index)
            last = j;
      }
      bool reorderable = true;
      for (unsigned j = i; j < i + count; j++)
         reorderable &= nir_intrinsic_can_reorder(r[j].load);
      if (!all_execute(&r[first].load->instr, &r[last].load->instr, reorderable)) {
         i++;
         continue;
      }

      /* An earlier merge may have replaced the value an address was built
       * from (an offset that is itself a merged read): read the base and the
       * offset from the live address, not from the list. */
      nir_def *live_base;
      nir_scalar live_offset;
      if (!split_address(r[first].load->src[0].ssa, &live_base, &live_offset)) {
         i++;
         continue;
      }
      nir_builder b = nir_builder_at(nir_before_instr(&r[first].load->instr));
      nir_def *off_val = nir_channel(&b, live_offset.def, live_offset.comp);
      nir_def *x = nir_iadd_imm(&b, off_val,
                                (uint32_t)(r[i].offset.constant -
                                           r[first].offset.constant));
      nir_def *addr = nir_iadd(&b, live_base, nir_u2u64(&b, x));
      nir_intrinsic_instr *proto = r[i].load;
      nir_def *vec;
      if (proto->intrinsic == nir_intrinsic_load_global_constant)
         vec = nir_load_global_constant(&b, count, 32, addr,
                                        .align_mul = 4,
                                        .access = nir_intrinsic_access(proto));
      else
         vec = nir_load_global(&b, count, 32, addr, .align_mul = 4,
                               .access = nir_intrinsic_access(proto));

      for (unsigned j = 0; j < count; j++) {
         nir_def_replace(&r[i + j].load->def, nir_channel(&b, vec, j));
      }
      progress = true;
      i += count;
   }
   return progress;
}

bool
agx_nir_merge_global_words(nir_shader *shader)
{
   bool any = false;
   struct util_dynarray reads;
   util_dynarray_init(&reads, NULL);

   nir_foreach_function_impl(impl, shader) {
      nir_index_ssa_defs(impl);
      nir_index_instrs(impl);
      bool progress = false;
      nir_foreach_block(block, impl)
         progress |= merge_block(block, &reads);
      any |= nir_progress(progress, impl, nir_metadata_control_flow);
   }

   util_dynarray_fini(&reads);
   return any;
}

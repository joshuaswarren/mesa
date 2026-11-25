/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * M5 loads and stores add a signed 16-bit byte offset to base + index. Move
 * constant parts of buffer offsets there, in two cases agx_nir_lower_address
 * cannot see once agx_nir_fuse_lea has turned the offset into a LEA:
 *
 * 1. An array element base + u2u64(amul(i + c, s)), as nir_lower_explicit_io
 *    emits for buf[i + c]. A valid access has 0 <= i + c < 2^31 / s (buffers
 *    are limited to INT32_MAX bytes on M5), so sext(i) * s + c * s is the
 *    same address and needs no add: the unrolled iterations of a loop over
 *    buf[i], buf[i + 1], ... all index with i.
 *
 * 2. Byte buffers addressed as base + u2u64(X + c) by the shader itself.
 *    zext(X + c) = zext(X + c0) + (c - c0) unless X + c wraps between them;
 *    then one of the two offsets is past 2^32 - 2^16, outside every M5
 *    buffer. When accesses at X + c0 and X + c run together (one block,
 *    nothing between them that ends the invocation), a shader that executes
 *    them on such offsets has undefined behaviour, so all of them may share
 *    the index X + c0 with byte offsets c - c0.
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"

#include "agx_nir_linear.h"

struct access {
   nir_intrinsic_instr *intr;
   nir_def *base, *offset;
   struct lin lin;
   /* An array element: offset = amul(index, 1 << log_stride). */
   bool array;
   nir_scalar index;
   unsigned log_stride;
};

/* base + u2u64(offset) with a 32-bit offset into a buffer binding (see
 * agx_lin_binding_base: the binding's range is what both cases below rely
 * on). */
static bool
split_address(nir_def *addr, nir_def **base, nir_def **offset)
{
   nir_scalar s = nir_scalar_resolved(addr, 0);
   if (addr->bit_size != 64 || addr->num_components != 1 ||
       !agx_lin_is_op(s, nir_op_iadd))
      return false;

   for (unsigned i = 0; i < 2; i++) {
      nir_scalar u = nir_scalar_chase_alu_src(s, i);
      nir_scalar other = nir_scalar_chase_alu_src(s, 1 - i);
      if (agx_lin_is_op(u, nir_op_u2u64) && !agx_lin_is_op(other, nir_op_u2u64) &&
          other.def->num_components == 1) {
         nir_scalar off = nir_scalar_chase_alu_src(u, 0);
         if (off.def->bit_size == 32 && off.def->num_components == 1 &&
             agx_lin_binding_base(other.def)) {
            *base = other.def;
            *offset = off.def;
            return true;
         }
      }
   }
   return false;
}

static unsigned
access_bit_size(nir_intrinsic_instr *intr)
{
   return intr->intrinsic == nir_intrinsic_store_global
             ? nir_src_bit_size(intr->src[0])
             : intr->def.bit_size;
}

static nir_src *
address_src(nir_intrinsic_instr *intr)
{
   return intr->intrinsic == nir_intrinsic_store_global ? &intr->src[1]
                                                        : &intr->src[0];
}

static enum pipe_format
format_for(unsigned bit_size)
{
   switch (bit_size) {
   case 8:
      return PIPE_FORMAT_R8_UINT;
   case 16:
      return PIPE_FORMAT_R16_UINT;
   default:
      return PIPE_FORMAT_R32_UINT;
   }
}

/* Replace an access by the native form base + index << shift + byte_offset. */
static void
emit_native(nir_builder *b, nir_intrinsic_instr *intr, nir_def *base,
            nir_def *index, unsigned shift, int32_t byte_offset, bool sext)
{
   const unsigned bit_size = access_bit_size(intr);
   const uint32_t mem_base =
      shift | ((uint32_t)(uint16_t)byte_offset << AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT);
   const enum pipe_format format = format_for(bit_size);

   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global:
      nir_def_replace(&intr->def,
                      nir_load_agx(b, intr->def.num_components, bit_size, base,
                                   index, .access = nir_intrinsic_access(intr),
                                   .base = mem_base, .format = format,
                                   .sign_extend = sext));
      break;
   case nir_intrinsic_load_global_constant:
      nir_def_replace(&intr->def,
                      nir_load_constant_agx(b, intr->def.num_components,
                                            bit_size, base, index,
                                            .access = nir_intrinsic_access(intr),
                                            .base = mem_base, .format = format,
                                            .sign_extend = sext));
      break;
   default:
      nir_store_agx(b, intr->src[0].ssa, base, index,
                    .access = nir_intrinsic_access(intr), .base = mem_base,
                    .format = format, .sign_extend = sext);
      nir_instr_remove(&intr->instr);
      break;
   }
}

/* offset = amul(index, s) with a power-of-two stride the index scale covers. */
static bool
match_array(struct access *a)
{
   nir_scalar off = nir_scalar_chase_movs(nir_get_scalar(a->offset, 0));
   if (!agx_lin_is_op(off, nir_op_amul))
      return false;
   const unsigned format_shift = util_logbase2(access_bit_size(a->intr) / 8);
   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar stride = nir_scalar_chase_alu_src(off, i);
      nir_scalar index = nir_scalar_chase_movs(nir_scalar_chase_alu_src(off, 1 - i));
      if (!nir_scalar_is_const(stride) || nir_scalar_is_const(index))
         continue;
      const uint64_t s = nir_scalar_as_uint(stride);
      if (!util_is_power_of_two_nonzero64(s))
         return false;
      const unsigned log_s = util_logbase2_64(s);
      if (log_s < format_shift || log_s > format_shift + 2)
         return false;
      a->array = true;
      a->index = index;
      a->log_stride = log_s;
      return true;
   }
   return false;
}

/* Case 1: base + u2u64(amul(iadd(i, #c), #s)). */
static bool
fold_array_element(nir_builder *b, struct access *a)
{
   nir_scalar off = nir_scalar_chase_movs(nir_get_scalar(a->offset, 0));
   if (!agx_lin_is_op(off, nir_op_amul))
      return false;

   const unsigned bit_size = access_bit_size(a->intr);
   const unsigned format_shift = util_logbase2(bit_size / 8);

   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar stride = nir_scalar_chase_alu_src(off, i);
      nir_scalar index = nir_scalar_chase_movs(nir_scalar_chase_alu_src(off, 1 - i));
      if (!nir_scalar_is_const(stride) || !agx_lin_is_op(index, nir_op_iadd))
         continue;

      const uint64_t s = nir_scalar_as_uint(stride);
      if (!util_is_power_of_two_nonzero64(s))
         return false;
      const unsigned log_s = util_logbase2_64(s);
      if (log_s < format_shift || log_s > format_shift + 2)
         return false;

      for (unsigned j = 0; j < 2; ++j) {
         nir_scalar k = nir_scalar_chase_alu_src(index, j);
         nir_scalar rest = nir_scalar_chase_alu_src(index, 1 - j);
         if (!nir_scalar_is_const(k) || nir_scalar_is_const(rest))
            continue;
         const int64_t byte_offset = nir_scalar_as_int(k) * (int64_t)s;
         if (byte_offset < INT16_MIN || byte_offset > INT16_MAX)
            return false;
         b->cursor = nir_before_instr(&a->intr->instr);
         emit_native(b, a->intr, a->base, nir_channel(b, rest.def, rest.comp),
                     log_s - format_shift, (int32_t)byte_offset, true);
         return true;
      }
      return false;
   }
   return false;
}

static bool
ends_invocation(nir_instr *instr)
{
   if (instr->type == nir_instr_type_call || instr->type == nir_instr_type_jump)
      return true;
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   switch (nir_instr_as_intrinsic(instr)->intrinsic) {
   case nir_intrinsic_terminate:
   case nir_intrinsic_terminate_if:
   case nir_intrinsic_demote:
   case nir_intrinsic_demote_if:
      return true;
   default:
      return false;
   }
}

static bool
fold_block(nir_builder *b, nir_block *block, struct util_dynarray *list)
{
   bool progress = false;
   util_dynarray_clear(list);

   /* Accesses since the last instruction that can end the invocation. */
   unsigned segment = 0;
   struct util_dynarray segs;
   util_dynarray_init(&segs, NULL);

   nir_foreach_instr_safe(instr, block) {
      if (ends_invocation(instr)) {
         segment++;
         continue;
      }
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_load_global &&
          intr->intrinsic != nir_intrinsic_load_global_constant &&
          intr->intrinsic != nir_intrinsic_store_global)
         continue;

      /* A speculatable access may have been hoisted out of the code that
       * guarded it, so its offset need not be one the program accesses. */
      if (nir_intrinsic_access(intr) & ACCESS_CAN_SPECULATE)
         continue;

      struct access a = {.intr = intr};
      if (!split_address(address_src(intr)->ssa, &a.base, &a.offset))
         continue;

      a.lin = (struct lin){.valid = true};
      if (match_array(&a)) {
         lin_parse(&a.lin, a.index, 1, 8);
      } else if (access_bit_size(intr) == 8) {
         lin_parse(&a.lin, nir_get_scalar(a.offset, 0), 1, 8);
      } else {
         continue;
      }
      if (!a.lin.valid || a.lin.n == 0)
         continue;
      lin_normalize(&a.lin);
      util_dynarray_append(list, a);
      util_dynarray_append(&segs, segment);
   }

   unsigned n = util_dynarray_num_elements(list, struct access);
   struct access *acc = list->data;
   unsigned *seg = segs.data;
   bool *done = calloc(n ? n : 1, sizeof(bool));

   for (unsigned i = 0; i < n; ++i) {
      if (done[i])
         continue;

      /* The group of i: same base and terms; array elements of one stride,
       * or bytes in one segment. */
      unsigned members[64], m = 0;
      uint32_t cmin = acc[i].lin.constant;
      for (unsigned j = i; j < n && m < ARRAY_SIZE(members); ++j) {
         if (done[j] || acc[j].base != acc[i].base ||
             acc[j].array != acc[i].array ||
             (acc[i].array ? acc[j].log_stride != acc[i].log_stride
                           : seg[j] != seg[i]) ||
             !lin_same_terms(&acc[i].lin, &acc[j].lin))
            continue;
         /* Keep the spread small enough for the byte offset. */
         int32_t d = (int32_t)(acc[j].lin.constant - acc[i].lin.constant);
         if (d < -0x3fff || d > 0x3fff)
            continue;
         members[m++] = j;
         if ((int32_t)(acc[j].lin.constant - cmin) < 0)
            cmin = acc[j].lin.constant;
      }
      for (unsigned k = 0; k < m; ++k)
         done[members[k]] = true;

      bool distinct = false;
      for (unsigned k = 1; k < m; ++k)
         distinct |= acc[members[k]].lin.constant != acc[members[0]].lin.constant;
      if (m < 2 || !distinct) {
         /* A lone array element still folds its own constant addend. */
         if (acc[i].array && fold_array_element(b, &acc[i]))
            progress = true;
         continue;
      }

      if (acc[i].array) {
         /* Array elements X + c of one array, as unrolled iterations read
          * them: one index X + cmin, sign-extended (valid indices are
          * nonnegative and below 2^31 / stride, so X + cmin, at most
          * c - cmin below a valid index, does not overflow), and byte
          * offsets (c - cmin) * stride. */
         b->cursor = nir_before_instr(&acc[i].intr->instr);
         nir_def *index =
            nir_iadd_imm(b, nir_channel(b, acc[i].index.def, acc[i].index.comp),
                         (uint32_t)(cmin - acc[i].lin.constant));
         for (unsigned k = 0; k < m; ++k) {
            struct access *a = &acc[members[k]];
            const int64_t off = (int64_t)(int32_t)(a->lin.constant - cmin)
                                << a->log_stride;
            const unsigned fs = util_logbase2(access_bit_size(a->intr) / 8);
            if (off < INT16_MIN || off > INT16_MAX)
               continue;
            emit_native(b, a->intr, a->base, index, a->log_stride - fs,
                        (int32_t)off, true);
         }
         progress = true;
         continue;
      }

      /* The shared index X + cmin, computed at the first access (i): its
       * offset value dominates every later one in the block. */
      b->cursor = nir_before_instr(&acc[i].intr->instr);
      nir_def *index = nir_iadd_imm(b, acc[i].offset,
                                    (uint32_t)(cmin - acc[i].lin.constant));
      for (unsigned k = 0; k < m; ++k) {
         struct access *a = &acc[members[k]];
         emit_native(b, a->intr, a->base, index, 0,
                     (int32_t)(a->lin.constant - cmin), false);
      }
      progress = true;
   }

   free(done);
   util_dynarray_fini(&segs);
   return progress;
}

bool
agx_nir_fold_m5_offsets(nir_shader *shader)
{
   bool any = false;
   struct util_dynarray list;
   util_dynarray_init(&list, NULL);

   nir_foreach_function_impl(impl, shader) {
      nir_index_ssa_defs(impl);
      nir_builder b = nir_builder_create(impl);
      bool progress = false;
      nir_foreach_block(block, impl)
         progress |= fold_block(&b, block, &list);
      any |= nir_progress(progress, impl, nir_metadata_control_flow);
   }

   util_dynarray_fini(&list);
   return any;
}

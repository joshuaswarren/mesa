/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
#include <string.h>

/*
 * AGX generally flushes FP32 denorms. However, the min/max instructions do not
 * as they are implemented with cmpsel. We need to flush the results of fp32
 * min/max for correctness. Doing so naively will generate redundant flushes, so
 * this pass tries to be clever and elide flushes when possible.
 *
 */
static bool
normal_bound(nir_scalar value, bool is_max)
{
   value = nir_scalar_chase_movs(value);
   if (!nir_scalar_is_const(value))
      return false;
   const uint32_t bits = nir_scalar_as_uint(value);
   const unsigned exponent = (bits >> 23) & 255;
   return exponent > 0 && exponent < 255 && (bits >> 31) == !is_max;
}

static bool
could_be_denorm(nir_scalar s, unsigned depth, unsigned *budget)
{
   if (!*budget)
      return true;
   --*budget;
   s = nir_scalar_chase_movs(s);
   /* Constants can be denorms only if they are denorms. */
   if (nir_scalar_is_const(s)) {
      /* Classify FP32 bits before promotion to double normalizes them. */
      uint32_t bits = nir_scalar_as_uint(s);
      return (bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0;
   }

   if (!depth)
      return true;
   if (nir_def_instr(s.def)->type == nir_instr_type_phi) {
      nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(s.def));
      nir_foreach_phi_src(src, phi) {
         if (could_be_denorm(nir_get_scalar(src->src.ssa, s.comp), depth - 1,
                             budget))
            return true;
      }
      return false;
   }
   if (nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == nir_op_bcsel)
      return could_be_denorm(nir_scalar_chase_alu_src(s, 1), depth - 1,
                             budget) ||
             could_be_denorm(nir_scalar_chase_alu_src(s, 2), depth - 1, budget);

   /* Floating-point instructions flush denormals, so ALU results can only be
    * denormal if they are not from a float instruction. Crucially fmin/fmax
    * flushes in NIR, so this pass handles chains of fmin/fmax properly.
    */
   if (nir_scalar_is_alu(s)) {
      nir_op op = nir_scalar_alu_op(s);
      if (op == nir_op_fmin_agx || op == nir_op_fmax_agx) {
         nir_scalar a = nir_scalar_chase_alu_src(s, 0);
         nir_scalar b = nir_scalar_chase_alu_src(s, 1);
         const bool is_max = op == nir_op_fmax_agx;
         if (normal_bound(a, is_max) || normal_bound(b, is_max))
            return false;
         return could_be_denorm(a, depth - 1, budget) ||
                could_be_denorm(b, depth - 1, budget);
      }
      nir_alu_type T = nir_op_infos[op].output_type;
      return nir_alu_type_get_base_type(T) != nir_type_float;
   }

   /* Otherwise, assume it could be denormal (say, loading from a buffer). */
   return true;
}

/* Whether the result of a min/max of scalar sources s0 and s1 needs an
 * explicit denormal flush. */
static bool
needs_flush(const nir_shader *shader, nir_scalar src0, nir_scalar src1,
            bool is_max, bool ftz_only)
{
   /* Without a flush-to-zero execution mode, whether a denormal result is
    * flushed is implementation-defined (Vulkan, "Precision and Operation of
    * SPIR-V Instructions": any denormal an instruction produces may be
    * flushed), so the min/max result may stay denormal. Only a shader that
    * asks for flushing needs it. Callers that keep flushing every min/max
    * pass false.
    */
   const unsigned mode = shader->info.float_controls_execution_mode;
   if (ftz_only && !(mode & FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32))
      return false;

   if (normal_bound(src0, is_max) || normal_bound(src1, is_max))
      return false;

   unsigned budget = 64;
   return could_be_denorm(src0, 8, &budget) || could_be_denorm(src1, 8, &budget);
}

static nir_def *
flush(nir_builder *b, nir_def *def)
{
   b->fp_math_ctrl = nir_fp_no_fast_math;
   return nir_fadd_imm(b, def, -0.0);
}

static bool
lower(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if ((alu->op != nir_op_fmin && alu->op != nir_op_fmax) ||
       (alu->def.bit_size != 32))
      return false;

   /* Lower the op, we'll fix up the denorms right after. */
   if (alu->op == nir_op_fmax)
      alu->op = nir_op_fmax_agx;
   else
      alu->op = nir_op_fmin_agx;

   const bool *ftz_only = data;
   bool canonicalize = false;
   for (unsigned c = 0; c < alu->def.num_components; ++c) {
      nir_scalar scalar = nir_get_scalar(&alu->def, c);
      canonicalize |= needs_flush(b->shader, nir_scalar_chase_alu_src(scalar, 0),
                                  nir_scalar_chase_alu_src(scalar, 1),
                                  alu->op == nir_op_fmax_agx, *ftz_only);
   }

   if (canonicalize) {
      b->cursor = nir_after_instr(&alu->instr);
      nir_def *canonicalized = flush(b, &alu->def);
      nir_def_rewrite_uses_after(&alu->def, canonicalized);
   }

   return true;
}

/* A tie between a nonzero, non-denormal constant and anything is between
 * identical bits. */
static bool
nonzero_constant(nir_scalar s)
{
   return nir_scalar_is_const(s) &&
          (nir_scalar_as_uint(s) & 0x7f800000u) != 0;
}

static bool
zero_constant(nir_scalar s)
{
   return nir_scalar_is_const(s) &&
          (nir_scalar_as_uint(s) & 0x7fffffffu) == 0;
}

/* fcmpsel returns one of its operands unchanged, so the hardware min/max is
 * already exact for signed zeros except when +0 and -0 tie: then it returns
 * its second operand. Integer min/max of the bits orders the two zeros as
 * IEEE 754-2019 does (-0 < +0) and returns either operand of any other tie,
 * as they are identical. That needs feq to compare denormals exactly when
 * the shader preserves them (AGX_CAP_SIGNED_ZERO_MINMAX_SELECT): a compare
 * that flushed its inputs would tie two different denormals of one sign and
 * imin/imax would pick by bit pattern. Where denormals may be flushed, such
 * a pick differs from the exact result only by a denormal. NaNs never
 * tie. This replaces the generic lower_fminmax_signed_zero emulation (up to
 * 9 instructions) with 3, with the bare min/max when a source is a nonzero
 * constant, or with one select against a zero constant.
 */
static bool
lower_signed_zero(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if ((alu->op != nir_op_fmin && alu->op != nir_op_fmax) ||
       alu->def.bit_size != 32 || alu->def.num_components != 1 ||
       !nir_alu_instr_is_signed_zero_preserve(alu))
      return false;

   const bool *ftz_only = data;
   const bool is_max = alu->op == nir_op_fmax;
   nir_scalar src0 = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 0);
   nir_scalar src1 = nir_scalar_chase_alu_src(nir_get_scalar(&alu->def, 0), 1);

   b->cursor = nir_before_instr(&alu->instr);
   b->fp_math_ctrl = alu->fp_math_ctrl;
   nir_def *s0 = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *s1 = nir_ssa_for_alu_src(b, alu, 1);
   nir_def *res;

   const int zero = zero_constant(src0) ? 0 : zero_constant(src1) ? 1 : -1;
   if (zero >= 0) {
      /* One select, ordered so a NaN or a tie picks the right zero:
       *    max(x, +0) = 0 < x ? x : +0      min(x, -0) = x < 0 ? x : -0
       *    max(x, -0) = x >= 0 ? x : -0     min(x, +0) = 0 >= x ? x : +0
       */
      nir_def *z = zero ? s1 : s0;
      nir_def *x = zero ? s0 : s1;
      const bool negative = nir_scalar_as_uint(zero ? src1 : src0) >> 31;
      nir_def *take_x = is_max ? (negative ? nir_fge(b, x, z) : nir_flt(b, z, x))
                               : (negative ? nir_flt(b, x, z) : nir_fge(b, z, x));
      res = nir_bcsel(b, take_x, x, z);
   } else {
      res = is_max ? nir_fmax_agx(b, s0, s1) : nir_fmin_agx(b, s0, s1);
      if (!nonzero_constant(src0) && !nonzero_constant(src1)) {
         nir_def *tie = is_max ? nir_imax(b, s0, s1) : nir_imin(b, s0, s1);
         res = nir_bcsel(b, nir_feq(b, s0, s1), tie, res);
      }
   }

   if (needs_flush(b->shader, src0, src1, is_max, *ftz_only))
      res = flush(b, res);

   nir_def_replace(&alu->def, res);
   return true;
}

bool
agx_nir_lower_fminmax_signed_zero(nir_shader *s, bool ftz_only)
{
   return nir_shader_alu_pass(s, lower_signed_zero, nir_metadata_control_flow,
                              &ftz_only);
}

bool
agx_nir_lower_fminmax(nir_shader *s, bool ftz_only)
{
   return nir_shader_alu_pass(s, lower, nir_metadata_control_flow, &ftz_only);
}

/* A shared-memory load cannot introduce a denormal if every writer stores
 * canonical FP32 values. Uninitialized shared reads have undefined values;
 * this analysis never relies on a particular value for them. Integer/partial
 * writes and atomics conservatively disable the shared-memory proof. */
struct canonical_state {
   bool shared_canonical;
   unsigned removed;
};

static bool
canonical_value(nir_scalar value, bool shared_canonical, nir_scalar *path,
                unsigned depth, unsigned *budget)
{
   if (!*budget)
      return false;
   (*budget)--;
   value = nir_scalar_chase_movs(value);
   if (value.def->bit_size != 32 || depth >= 64)
      return false;
   for (unsigned i = 0; i < depth; ++i)
      if (path[i].def == value.def && path[i].comp == value.comp)
         return false;
   path[depth++] = value;

   if (nir_scalar_is_const(value)) {
      uint32_t bits = nir_scalar_as_uint(value);
      if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0)
         return false; /* Do not remove a possible signalling-NaN quieting. */
      return (bits & 0x7f800000u) != 0 || (bits & 0x007fffffu) == 0;
   }

   if (nir_def_instr(value.def)->type == nir_instr_type_phi) {
      nir_phi_instr *phi = nir_def_as_phi(value.def);
      nir_foreach_phi_src(src, phi) {
         if (!canonical_value(nir_get_scalar(src->src.ssa, value.comp),
                              shared_canonical, path, depth, budget))
            return false;
      }
      return true;
   }

   if (nir_scalar_is_intrinsic(value)) {
      if (!shared_canonical ||
          nir_scalar_intrinsic_op(value) != nir_intrinsic_load_shared)
         return false;
      nir_intrinsic_instr *load = nir_def_as_intrinsic(value.def);
      return nir_intrinsic_align_mul(load) >= 4 &&
             (nir_intrinsic_align_offset(load) & 3) == 0;
   }

   if (!nir_scalar_is_alu(value))
      return false;

   nir_op op = nir_scalar_alu_op(value);
   switch (op) {
   case nir_op_bcsel:
   case nir_op_b32csel:
      return canonical_value(nir_scalar_chase_alu_src(value, 1),
                             shared_canonical, path, depth, budget) &&
             canonical_value(nir_scalar_chase_alu_src(value, 2),
                             shared_canonical, path, depth, budget);
   case nir_op_fmin:
   case nir_op_fmax:
   case nir_op_fmin_agx:
   case nir_op_fmax_agx:
      return canonical_value(nir_scalar_chase_alu_src(value, 0),
                             shared_canonical, path, depth, budget) &&
             canonical_value(nir_scalar_chase_alu_src(value, 1),
                             shared_canonical, path, depth, budget);
   case nir_op_fneg:
   case nir_op_fabs:
      return canonical_value(nir_scalar_chase_alu_src(value, 0),
                             shared_canonical, path, depth, budget);
   case nir_op_f2f32:
      return nir_scalar_chase_alu_src(value, 0).def->bit_size == 16;
   case nir_op_fadd:
   case nir_op_fmul:
   case nir_op_ffma:
   case nir_op_fdiv:
   case nir_op_frcp:
   case nir_op_frsq:
   case nir_op_fsqrt:
   case nir_op_fexp2:
   case nir_op_flog2:
   case nir_op_ffloor:
   case nir_op_fceil:
   case nir_op_ftrunc:
   case nir_op_fround_even:
   case nir_op_i2f32:
   case nir_op_u2f32:
      return true;
   default:
      return false;
   }
}

static bool
canonical(nir_def *def, unsigned component, bool shared_canonical)
{
   nir_scalar path[64];
   unsigned budget = 1024;
   return canonical_value(nir_get_scalar(def, component), shared_canonical,
                          path, 0, &budget);
}

static bool
remove_canonical_minmax(nir_builder *b, nir_alu_instr *alu, void *data)
{
   struct canonical_state *state = data;
   if (alu->op != nir_op_fadd || alu->def.bit_size != 32 ||
       alu->def.num_components != 1)
      return false;

   nir_scalar self = nir_get_scalar(&alu->def, 0);
   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar zero = nir_scalar_chase_alu_src(self, i);
      nir_scalar value = nir_scalar_chase_alu_src(self, 1 - i);
      if (!nir_scalar_is_const(zero) ||
          nir_scalar_as_uint(zero) != 0x80000000u || !nir_scalar_is_alu(value))
         continue;
      nir_op op = nir_scalar_alu_op(value);
      if (op != nir_op_fmin_agx && op != nir_op_fmax_agx)
         continue;
      if (!canonical(value.def, value.comp, state->shared_canonical))
         continue;
      b->cursor = nir_before_instr(&alu->instr);
      nir_def *replacement = nir_channel(b, value.def, value.comp);
      nir_def_rewrite_uses(&alu->def, replacement);
      nir_instr_remove(&alu->instr);
      state->removed++;
      return true;
   }
   return false;
}

bool
agx_nir_opt_canonical_minmax(nir_shader *s)
{
   if (nir_is_denorm_preserve(s->info.float_controls_execution_mode, 32))
      return false;

   bool seen_store = false, shared_canonical = true;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_shared) {
               seen_store = true;
               nir_def *value = intr->src[0].ssa;
               if (value->bit_size != 32 || nir_intrinsic_align_mul(intr) < 4 ||
                   (nir_intrinsic_align_offset(intr) & 3) != 0) {
                  shared_canonical = false;
                  continue;
               }
               for (unsigned c = 0; c < value->num_components; ++c)
                  if ((nir_intrinsic_write_mask(intr) & (1u << c)) &&
                      !canonical(value, c, false))
                     shared_canonical = false;
            } else {
               const char *name = nir_intrinsic_infos[intr->intrinsic].name;
               if ((strstr(name, "shared") &&
                    (strstr(name, "atomic") || strstr(name, "store"))) ||
                   (strstr(name, "deref") &&
                    (strstr(name, "atomic") || strstr(name, "store") ||
                     strstr(name, "copy"))))
                  shared_canonical = false;
            }
         }
      }
   }
   struct canonical_state state = {seen_store && shared_canonical, 0};
   return nir_shader_alu_pass(s, remove_canonical_minmax,
                              nir_metadata_control_flow, &state);
}

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"

/*
 * Lower 32-bit integer division and modulo with a divergent dividend and a
 * non-constant uniform divisor d through the exact reciprocal
 * M = floor(2^32 / d) (M = UINT32_MAX for d == 1).
 *
 * This is nir_lower_idiv's algorithm with the reciprocal made exact. M depends
 * only on d and is straight-line 32-bit ALU, so agx_nir_opt_preamble hoists it
 * into @preamble, and it costs one uniform, like nir_lower_idiv's reciprocal.
 * Because 2^32 / d - 1 < M <= 2^32 / d, the quotient estimate umul_high(n, M)
 * is the quotient or one below it for every n, so the main shader needs one
 * correction step where nir_lower_idiv needs two:
 *   - udiv/umod: umul_high, imul, isub, uge, bcsel (+ iadd or isub)
 *   - signed: the same on |n| and |d|, then nir_lower_idiv's sign fixups
 *
 * Division by zero is undefined in GLSL/SPIR-V; nothing hangs or faults:
 * udiv(n, 0) returns (n >> 1) + 1 and umod(n, 0) returns n.
 */

/* M = floor(2^32 / d) for d >= 2, UINT32_MAX for d == 1:
 *
 *  1. q0 from a float reciprocal scaled by 2^32 - 2^14, so it stays below
 *     2^32 / d even with the reciprocal 16 ULP high: 1 <= q0 and
 *     q0 * d <= 2^32, so the exact error e = 2^32 - q0 * d is -(q0 * d) in
 *     32-bit arithmetic. The hardware reciprocal is within 2.5 ULP, so the
 *     correctly rounded one is not needed.
 *  2. One integer Newton-Raphson step, q1 = q0 + umul_high(q0, e), leaves q1
 *     at M or M - 1.
 *  3. One exact step up when the remainder 2^32 - q1 * d is still >= d.
 *
 * Checked exhaustively on the CPU, mirroring these operations in float32 with
 * round-to-nearest and truncating conversions: every d in [1, 2^32) with the
 * reciprocal off by up to 16 ULP either way, 1.4e11 cases, all exact, with
 * the invariants above holding throughout.
 */
static nir_def *
uniform_reciprocal(nir_builder *b, nir_def *d)
{
   nir_def *d_safe = nir_umax(b, d, nir_imm_int(b, 2));
   nir_def *q0 = nir_f2u32(
      b, nir_fmul_imm(b, nir_frcp(b, nir_u2f32(b, d_safe)), 4294950912.0));
   q0 = nir_umax(b, q0, nir_imm_int(b, 1));

   nir_def *q1 =
      nir_iadd(b, q0, nir_umul_high(b, q0, nir_ineg(b, nir_imul(b, q0, d_safe))));
   nir_def *rem = nir_ineg(b, nir_imul(b, q1, d_safe));
   nir_def *q2 = nir_bcsel(b, nir_uge(b, rem, d_safe), nir_iadd_imm(b, q1, 1), q1);

   return nir_bcsel(b, nir_ieq_imm(b, d, 1), nir_imm_int(b, UINT32_MAX), q2);
}

/* n / d or n % d from the quotient estimate umul_high(n, M), which is exact or
 * one low. */
static nir_def *
udivmod(nir_builder *b, nir_def *n, nir_def *d, bool modulo)
{
   nir_def *q = nir_umul_high(b, n, uniform_reciprocal(b, d));
   nir_def *r = nir_isub(b, n, nir_imul(b, q, d));
   nir_def *ge = nir_uge(b, r, d);
   return modulo ? nir_bcsel(b, ge, nir_isub(b, r, d), r)
                 : nir_bcsel(b, ge, nir_iadd_imm(b, q, 1), q);
}

static nir_def *
lower_divmod_scalar(nir_builder *b, nir_op op, nir_def *n, nir_def *d)
{
   if (op == nir_op_udiv || op == nir_op_umod)
      return udivmod(b, n, d, op == nir_op_umod);

   /* Signed, as nir_lower_idiv: divide magnitudes, then fix the signs. */
   nir_def *n_neg = nir_ilt_imm(b, n, 0);
   nir_def *res = udivmod(b, nir_iabs(b, n), nir_iabs(b, d), op != nir_op_idiv);

   switch (op) {
   case nir_op_idiv:
      return nir_bcsel(b, nir_ine(b, n_neg, nir_ilt_imm(b, d, 0)),
                       nir_ineg(b, res), res);
   case nir_op_irem:
      return nir_bcsel(b, n_neg, nir_ineg(b, res), res);
   case nir_op_imod: {
      nir_def *r = nir_bcsel(b, n_neg, nir_ineg(b, res), res);
      nir_def *need_adj = nir_iand(b, nir_ine_imm(b, r, 0),
                                   nir_ilt_imm(b, nir_ixor(b, r, d), 0));
      return nir_bcsel(b, need_adj, nir_iadd(b, r, d), r);
   }
   default:
      UNREACHABLE("unexpected op");
   }
}

/* Check whether a def can be hoisted into @preamble by nir_opt_preamble.
 * Only constant/ALU chains rooted in genuinely preamble-movable uniform loads
 * (push constants, UBOs, constant globals, uniforms, num_workgroups, image
 * sizes) qualify. Subgroup-uniform values that cannot hoist to @preamble
 * (e.g. load_workgroup_id, non-readonly SSBOs, shared loads, subgroup ops,
 * loop phis) return false so we never leave the reciprocal calculation in
 * the main shader body. Checking movability directly also avoids calling
 * nir_divergence_analysis before backend/poly intrinsics are lowered.
 *
 * Before shader I/O lowering, push constants and UBOs are still deref loads.
 * They count as the load_push_constant and load_ubo they lower to, so
 * preprocessing can predict which divisions this pass will take. */
static bool
is_preamble_movable(nir_def *def, struct hash_table *cache, unsigned depth)
{
   struct hash_entry *he = _mesa_hash_table_search(cache, def);
   if (he)
      return (uintptr_t)he->data != 0;

   if (depth == 0)
      return false;

   /* Break cycles conservatively before recursing. */
   _mesa_hash_table_insert(cache, def, (void *)(uintptr_t)0);

   bool ok = false;
   nir_instr *instr = nir_def_instr(def);
   switch (instr->type) {
   case nir_instr_type_load_const:
      ok = true;
      break;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      ok = true;
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i) {
         if (!is_preamble_movable(alu->src[i].src.ssa, cache, depth - 1)) {
            ok = false;
            break;
         }
      }
      break;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_push_constant:
      case nir_intrinsic_load_preamble:
      case nir_intrinsic_load_uniform:
      case nir_intrinsic_load_ubo:
      case nir_intrinsic_load_global_constant:
      case nir_intrinsic_load_constant_agx:
      case nir_intrinsic_load_num_workgroups:
      case nir_intrinsic_load_workgroup_size:
      case nir_intrinsic_load_sysval_agx:
      case nir_intrinsic_image_size:
      case nir_intrinsic_image_deref_size:
      case nir_intrinsic_bindless_image_size: {
         const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
         ok = true;
         for (unsigned i = 0; i < info->num_srcs; ++i) {
            if (!is_preamble_movable(intr->src[i].ssa, cache, depth - 1)) {
               ok = false;
               break;
            }
         }
         break;
      }
      case nir_intrinsic_load_deref: {
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         ok = nir_deref_mode_must_be(deref, nir_var_mem_push_const |
                                               nir_var_mem_ubo);
         for (; ok && deref->deref_type != nir_deref_type_var;
              deref = nir_deref_instr_parent(deref)) {
            if (deref->deref_type == nir_deref_type_array)
               ok = is_preamble_movable(deref->arr.index.ssa, cache, depth - 1);
            else
               ok = deref->deref_type == nir_deref_type_struct;
         }
         break;
      }
      default:
         ok = false;
         break;
      }
      break;
   }
   default:
      ok = false;
      break;
   }

   _mesa_hash_table_insert(cache, def, (void *)(uintptr_t)(ok ? 1 : 0));
   return ok;
}

/* Only optimize non-preamble-movable dividend / non-constant preamble-movable
 * divisor. Constant divisors are handled by nir_opt_idiv_const, and
 * preamble-movable/preamble-movable divisions hoist directly to @preamble.
 * The cache must be private to a single shader. */
bool
agx_nir_uniform_idiv_applies(nir_alu_instr *alu, struct hash_table *cache)
{
   return !nir_src_is_const(alu->src[1].src) &&
          is_preamble_movable(alu->src[1].src.ssa, cache, 32) &&
          !is_preamble_movable(alu->src[0].src.ssa, cache, 32);
}

static bool
lower_uniform_idiv_instr(nir_builder *b, nir_alu_instr *alu, void *data)
{
   struct hash_table *cache = data;

   if (alu->def.bit_size != 32 || alu->def.num_components != 1)
      return false;

   switch (alu->op) {
   case nir_op_udiv:
   case nir_op_umod:
   case nir_op_idiv:
   case nir_op_irem:
   case nir_op_imod:
      break;
   default:
      return false;
   }

   if (!agx_nir_uniform_idiv_applies(alu, cache))
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *n = nir_mov_alu(b, alu->src[0], 1);
   nir_def *d = nir_mov_alu(b, alu->src[1], 1);
   nir_def *out = lower_divmod_scalar(b, alu->op, n, d);
   nir_def_replace(&alu->def, out);
   return true;
}

bool
agx_nir_opt_uniform_idiv(nir_shader *shader)
{
   struct hash_table *cache = _mesa_pointer_hash_table_create(NULL);
   bool progress = nir_shader_alu_pass(shader, lower_uniform_idiv_instr,
                                       nir_metadata_control_flow, cache);
   _mesa_hash_table_destroy(cache, NULL);
   return progress;
}

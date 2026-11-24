/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"

/*
 * Lower 32-bit integer division and modulo with a divergent dividend and a
 * non-constant uniform divisor using branchfree Granlund-Montgomery magic
 * multipliers.
 *
 * The magic-multiplier calculation depends only on the uniform divisor and
 * uses pure 32-bit straight-line ALU (no control flow, no 64-bit integer
 * types), so agx_nir_opt_preamble hoists it cleanly into @preamble. The
 * per-invocation hot path drops from nir_lower_idiv's ~14-30 instructions
 * (two sequential correction chains + sign fixups) to:
 *   - udiv: 5 ALU ops (umul_high, isub, ushr, iadd, ushr)
 *   - idiv: 7 ALU ops (imul_high, iadd, ishr, ushr, iadd, ixor, isub)
 *   - umod/irem: +1 imad (-d * q + n)
 *   - imod: +4 ALU ops (floored adjustment when r != 0 && (r ^ d) < 0)
 *
 * Division by zero is undefined in GLSL/SPIR-V; nothing hangs or faults, and
 * umod/irem/imod(n, 0) return n while udiv(n, 0) returns n >> 1 and
 * idiv(n, 0) returns n / 2 (differing from nir_lower_idiv's n + 1 for udiv).
 */

struct u64_pair {
   nir_def *hi;
   nir_def *lo;
};

static struct u64_pair
sub64(nir_builder *b, struct u64_pair a, struct u64_pair c)
{
   nir_def *borrow = nir_b2i32(b, nir_ult(b, a.lo, c.lo));
   return (struct u64_pair){
      .hi = nir_isub(b, nir_isub(b, a.hi, c.hi), borrow),
      .lo = nir_isub(b, a.lo, c.lo),
   };
}

static nir_def *
gt64(nir_builder *b, struct u64_pair a, struct u64_pair c)
{
   return nir_ior(b, nir_ult(b, c.hi, a.hi),
                  nir_iand(b, nir_ieq(b, a.hi, c.hi), nir_ult(b, c.lo, a.lo)));
}

static nir_def *
ge64(nir_builder *b, struct u64_pair a, struct u64_pair c)
{
   return nir_ior(b, nir_ult(b, c.hi, a.hi),
                  nir_iand(b, nir_ieq(b, a.hi, c.hi), nir_uge(b, a.lo, c.lo)));
}

static struct u64_pair
mul64(nir_builder *b, nir_def *x, nir_def *y)
{
   return (struct u64_pair){
      .hi = nir_umul_high(b, x, y),
      .lo = nir_imul(b, x, y),
   };
}

/* Compute exact floor((hi * 2^32) / d) for all 2 <= d <= UINT32_MAX and
 * 0 <= hi < d using only pure 32-bit branchfree ALU. */
static nir_def *
udiv64_hi_exact(nir_builder *b, nir_def *hi, nir_def *d)
{
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *one = nir_imm_int(b, 1);
   struct u64_pair num = { .hi = hi, .lo = zero };

   unsigned prev_ctrl = b->fp_math_ctrl;
   b->fp_math_ctrl = nir_fp_exact;

   nir_def *df = nir_u2f32(b, d);
   nir_def *r0 = nir_frcp(b, df);
   /* One Newton-Raphson step: r1 = r0 * (2 - df * r0). */
   nir_def *r1 = nir_fmul(b, r0, nir_ffma(b, nir_fneg(b, df), r0, nir_imm_float(b, 2.0f)));

   /* Slightly conservative initial estimate so f2u32 never overflows 2^32. */
   nir_def *qf = nir_fmul(b, nir_u2f32(b, hi),
                          nir_fmul(b, r1, nir_imm_float(b, 4294963200.0f)));
   nir_def *q = nir_f2u32(b, qf);

   /* Handle rare tiny overshoot when u2f32(d) rounded down. */
   struct u64_pair prod = mul64(b, q, d);
   nir_def *over = gt64(b, prod, num);
   struct u64_pair ov_diff = sub64(b, prod, num);
   nir_def *ohi = nir_bcsel(b, over, ov_diff.hi, zero);
   nir_def *olo = nir_bcsel(b, over, ov_diff.lo, zero);
   nir_def *of = nir_ffma(b, nir_u2f32(b, ohi), nir_imm_float(b, 4294967296.0f),
                          nir_u2f32(b, olo));
   nir_def *dq_down = nir_iadd_imm(b, nir_f2u32(b, nir_fmul(b, of, r1)), 1);
   q = nir_bcsel(b, over, nir_isub(b, q, dq_down), q);

   prod = mul64(b, q, d);
   nir_def *over2 = gt64(b, prod, num);
   q = nir_bcsel(b, over2, nir_isub(b, q, one), q);
   prod = mul64(b, q, d);

   /* Non-negative 64-bit remainder: refine upward once. */
   struct u64_pair rem = sub64(b, num, prod);
   nir_def *rf = nir_ffma(b, nir_u2f32(b, rem.hi), nir_imm_float(b, 4294967296.0f),
                          nir_u2f32(b, rem.lo));
   nir_def *dq_up_raw = nir_f2u32(b, nir_fmul(b, rf, nir_fmul(b, r1, nir_imm_float(b, 0.999999f))));
   nir_def *dq_up = nir_umin(b, nir_isub(b, nir_imm_int(b, 0xffffffff), q), dq_up_raw);
   q = nir_iadd(b, q, dq_up);

   b->fp_math_ctrl = prev_ctrl;

   /* Exact +/-3 conditional fixup steps guarantee zero ULP error for all d. */
   prod = mul64(b, q, d);
   struct u64_pair d64 = { .hi = zero, .lo = d };
   for (unsigned i = 0; i < 3; ++i) {
      nir_def *ov = gt64(b, prod, num);
      q = nir_bcsel(b, ov, nir_isub(b, q, one), q);
      struct u64_pair np = sub64(b, prod, d64);
      prod.hi = nir_bcsel(b, ov, np.hi, prod.hi);
      prod.lo = nir_bcsel(b, ov, np.lo, prod.lo);
   }

   rem = sub64(b, num, prod);
   for (unsigned i = 0; i < 3; ++i) {
      nir_def *un = ge64(b, rem, d64);
      q = nir_bcsel(b, un, nir_iadd(b, q, one), q);
      struct u64_pair nr = sub64(b, rem, d64);
      rem.hi = nir_bcsel(b, un, nr.hi, rem.hi);
      rem.lo = nir_bcsel(b, un, nr.lo, rem.lo);
   }

   return q;
}

static nir_def *
lower_unsigned_div_scalar(nir_builder *b, nir_def *n, nir_def *d)
{
   nir_def *is_one = nir_ieq_imm(b, d, 1);
   nir_def *d_safe = nir_umax(b, d, nir_imm_int(b, 2));
   nir_def *l = nir_iadd_imm(b, nir_ufind_msb(b, nir_iadd_imm(b, d_safe, -1)), 1);
   nir_def *pow2 = nir_bcsel(b, nir_ieq_imm(b, l, 32), nir_imm_int(b, 0),
                             nir_ishl(b, nir_imm_int(b, 1), l));
   nir_def *hi = nir_isub(b, pow2, d_safe);
   nir_def *m = nir_bcsel(b, is_one, nir_imm_int(b, 1),
                          nir_iadd_imm(b, udiv64_hi_exact(b, hi, d_safe), 1));
   nir_def *sh1 = nir_bcsel(b, is_one, nir_imm_int(b, 0), nir_imm_int(b, 1));
   nir_def *sh2 = nir_bcsel(b, is_one, nir_imm_int(b, 0), nir_iadd_imm(b, l, -1));

   nir_def *t1 = nir_umul_high(b, n, m);
   nir_def *t2 = nir_isub(b, n, t1);
   nir_def *t3 = nir_ushr(b, t2, sh1);
   nir_def *t4 = nir_iadd(b, t1, t3);
   return nir_ushr(b, t4, sh2);
}

static nir_def *
lower_signed_div_scalar(nir_builder *b, nir_def *n, nir_def *d)
{
   nir_def *ad = nir_iabs(b, d);
   nir_def *is_one = nir_ieq_imm(b, ad, 1);
   nir_def *d_safe = nir_umax(b, ad, nir_imm_int(b, 2));
   nir_def *l = nir_iadd_imm(b, nir_ufind_msb(b, nir_iadd_imm(b, d_safe, -1)), 1);
   nir_def *hi = nir_ishl(b, nir_imm_int(b, 1), nir_iadd_imm(b, l, -1));
   nir_def *m_u = nir_iadd_imm(b, udiv64_hi_exact(b, hi, d_safe), 1);

   nir_def *d_pos = nir_ige_imm(b, d, 0);
   nir_def *m = nir_bcsel(b, is_one, nir_imm_int(b, 0),
                          nir_bcsel(b, d_pos, m_u, nir_ineg(b, m_u)));
   nir_def *sh = nir_bcsel(b, is_one, nir_imm_int(b, 0), nir_iadd_imm(b, l, -1));
   nir_def *add_coeff = nir_bcsel(b, d_pos, nir_imm_int(b, 1), nir_imm_int(b, -1));
   nir_def *add_sign = nir_bcsel(b, is_one, nir_imm_int(b, 0), nir_imm_int(b, 1));

   nir_def *h = nir_imul_high(b, n, m);
   nir_def *t = nir_iadd(b, h, nir_imul(b, n, add_coeff));
   nir_def *q0 = nir_ishr(b, t, sh);
   nir_def *sg = nir_ushr_imm(b, t, 31);
   return nir_iadd(b, q0, nir_imul(b, sg, add_sign));
}

static nir_def *
lower_divmod_scalar(nir_builder *b, nir_op op, nir_def *n, nir_def *d)
{
   switch (op) {
   case nir_op_udiv:
      return lower_unsigned_div_scalar(b, n, d);
   case nir_op_umod: {
      nir_def *q = lower_unsigned_div_scalar(b, n, d);
      return nir_iadd(b, n, nir_imul(b, q, nir_ineg(b, d)));
   }
   case nir_op_idiv:
      return lower_signed_div_scalar(b, n, d);
   case nir_op_irem: {
      nir_def *q = lower_signed_div_scalar(b, n, d);
      return nir_iadd(b, n, nir_imul(b, q, nir_ineg(b, d)));
   }
   case nir_op_imod: {
      nir_def *q = lower_signed_div_scalar(b, n, d);
      nir_def *r = nir_iadd(b, n, nir_imul(b, q, nir_ineg(b, d)));
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
 * loop phis) return false so we never leave udiv64_hi_exact in the main
 * shader body. Checking movability directly also avoids calling
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

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Strength-reduce 32-bit signed integer division and modulo (idiv, imod, irem)
 * when the dividend, divisor, or both are provably nonnegative.
 *
 * Generic nir_lower_idiv lowers signed division by taking iabs of both
 * operands and applying sign fixups on the result. When an operand is known
 * to be nonnegative (via nir_unsigned_upper_bound <= INT32_MAX, bounded
 * global invocation ID + base, imax/imin/bcsel, or a dominating branch guard),
 * we can drop its iabs and simplify or eliminate the sign fixup:
 *
 * - nonneg_a && nonneg_b:
 *     idiv(a, b) -> udiv(a, b)
 *     imod(a, b) -> umod(a, b)
 *     irem(a, b) -> umod(a, b)
 * - nonneg_a && !nonneg_b:
 *     irem(a, b) -> umod(a, iabs(b))
 *     idiv(a, b) -> q = udiv(a, iabs(b)); bcsel(ilt(b, 0), ineg(q), q)
 *     imod(a, b) -> r = umod(a, iabs(b));
 *                   bcsel(ieq(r, 0), 0, iadd(r, bcsel(ilt(b, 0), b, 0)))
 * - !nonneg_a && nonneg_b (non-constant b, so nir_opt_idiv_const keeps consts):
 *     idiv(a, b) -> q = udiv(iabs(a), b); bcsel(ilt(a, 0), ineg(q), q)
 *     irem(a, b) -> r = umod(iabs(a), b); bcsel(ilt(a, 0), ineg(r), r)
 *     imod(a, b) -> r = umod(iabs(a), b);
 *                   bcsel(ilt(a, 0), bcsel(ieq(r, 0), 0, isub(b, r)), r)
 *
 * Every identity above is bit-exact over all 32-bit signed inputs with b != 0,
 * including INT32_MIN (where two's-complement iabs(INT32_MIN) == 0x80000000u).
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_range_analysis.h"
#include "util/hash_table.h"

struct nonneg_ctx {
   nir_shader *shader;
   struct hash_table *range_ht;
};

static bool
scalar_equal(nir_scalar a, nir_scalar b)
{
   a = nir_scalar_chase_movs(a);
   b = nir_scalar_chase_movs(b);
   return a.def == b.def && a.comp == b.comp;
}

static bool
is_bounded_global_id_plus_base(const nir_shader *shader, nir_scalar s)
{
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_iadd)
      return false;

   const nir_shader_compiler_options *opt = shader->options;
   if (!opt)
      return false;

   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
   if (!nir_scalar_is_intrinsic(a) || !nir_scalar_is_intrinsic(b))
      return false;

   nir_intrinsic_op opa = nir_scalar_intrinsic_op(a);
   nir_intrinsic_op opb = nir_scalar_intrinsic_op(b);
   if (opa == nir_intrinsic_load_base_global_invocation_id &&
       opb == nir_intrinsic_load_global_invocation_id) {
      nir_scalar tmp = a;
      a = b;
      b = tmp;
      opa = nir_scalar_intrinsic_op(a);
      opb = nir_scalar_intrinsic_op(b);
   }
   if (opa != nir_intrinsic_load_global_invocation_id ||
       opb != nir_intrinsic_load_base_global_invocation_id ||
       a.comp != b.comp || a.comp >= 3)
      return false;

   /* Vulkan guarantees baseGroup[c] + groupCount[c] <= maxComputeWorkGroupCount[c],
    * so the sum of local-relative global ID and base global ID stays within
    * max_workgroup_count[c] * max_wg_size[c] - 1. */
   const uint64_t max_groups = opt->max_workgroup_count[a.comp];
   if (!max_groups)
      return false;
   const uint64_t max_local =
      shader->info.workgroup_size_variable
         ? (opt->max_workgroup_invocations ? opt->max_workgroup_invocations
                                           : UINT16_MAX)
         : shader->info.workgroup_size[a.comp];
   return max_local > 0 && max_groups * max_local - 1 <= INT32_MAX;
}

static bool
scalar_is_nonneg_expr(struct nonneg_ctx *ctx, nir_scalar s, unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   if (s.def->bit_size != 32)
      return false;

   if (nir_unsigned_upper_bound(ctx->shader, ctx->range_ht, s) <= INT32_MAX)
      return true;

   if (is_bounded_global_id_plus_base(ctx->shader, s))
      return true;

   if (depth >= 6 || !nir_scalar_is_alu(s))
      return false;

   switch (nir_scalar_alu_op(s)) {
   case nir_op_imax:
      return scalar_is_nonneg_expr(ctx, nir_scalar_chase_alu_src(s, 0),
                                   depth + 1) ||
             scalar_is_nonneg_expr(ctx, nir_scalar_chase_alu_src(s, 1),
                                   depth + 1);
   case nir_op_imin:
      return scalar_is_nonneg_expr(ctx, nir_scalar_chase_alu_src(s, 0),
                                   depth + 1) &&
             scalar_is_nonneg_expr(ctx, nir_scalar_chase_alu_src(s, 1),
                                   depth + 1);
   case nir_op_bcsel:
      return scalar_is_nonneg_expr(ctx, nir_scalar_chase_alu_src(s, 1),
                                   depth + 1) &&
             scalar_is_nonneg_expr(ctx, nir_scalar_chase_alu_src(s, 2),
                                   depth + 1);
   default:
      return false;
   }
}

static bool
cond_implies_nonneg(struct nonneg_ctx *ctx, nir_scalar cond, nir_scalar target,
                    bool truth, unsigned depth)
{
   cond = nir_scalar_chase_movs(cond);
   if (depth >= 8 || !nir_scalar_is_alu(cond))
      return false;

   const nir_op op = nir_scalar_alu_op(cond);
   if (op == nir_op_inot) {
      return cond_implies_nonneg(ctx, nir_scalar_chase_alu_src(cond, 0), target,
                                 !truth, depth + 1);
   }
   if ((truth && op == nir_op_iand) || (!truth && op == nir_op_ior)) {
      return cond_implies_nonneg(ctx, nir_scalar_chase_alu_src(cond, 0), target,
                                 truth, depth + 1) ||
             cond_implies_nonneg(ctx, nir_scalar_chase_alu_src(cond, 1), target,
                                 truth, depth + 1);
   }

   if (nir_op_infos[op].num_inputs != 2)
      return false;

   nir_scalar s0 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 0));
   nir_scalar s1 = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 1));
   if (s0.def->bit_size != 32 || s1.def->bit_size != 32)
      return false;

   switch (op) {
   case nir_op_ige:
      if (truth) {
         if (scalar_equal(s0, target) && scalar_is_nonneg_expr(ctx, s1, 0))
            return true;
      } else {
         if (scalar_equal(s1, target) &&
             ((nir_scalar_is_const(s0) && nir_scalar_as_int(s0) >= -1) ||
              scalar_is_nonneg_expr(ctx, s0, 0)))
            return true;
      }
      return false;

   case nir_op_ilt:
      if (!truth) {
         if (scalar_equal(s0, target) && scalar_is_nonneg_expr(ctx, s1, 0))
            return true;
      } else {
         if (scalar_equal(s1, target) &&
             ((nir_scalar_is_const(s0) && nir_scalar_as_int(s0) >= -1) ||
              scalar_is_nonneg_expr(ctx, s0, 0)))
            return true;
      }
      return false;

   case nir_op_ult:
      if (truth) {
         if (scalar_equal(s0, target) &&
             ((nir_scalar_is_const(s1) &&
               nir_scalar_as_uint(s1) <= 0x80000000u) ||
              scalar_is_nonneg_expr(ctx, s1, 0)))
            return true;
      } else {
         if (scalar_equal(s1, target) && scalar_is_nonneg_expr(ctx, s0, 0))
            return true;
      }
      return false;

   case nir_op_uge:
      if (!truth) {
         if (scalar_equal(s0, target) &&
             ((nir_scalar_is_const(s1) &&
               nir_scalar_as_uint(s1) <= 0x80000000u) ||
              scalar_is_nonneg_expr(ctx, s1, 0)))
            return true;
      } else {
         if (scalar_equal(s1, target) && scalar_is_nonneg_expr(ctx, s0, 0))
            return true;
      }
      return false;

   case nir_op_ieq:
      if (!truth)
         return false;
      return (scalar_equal(s0, target) && scalar_is_nonneg_expr(ctx, s1, 0)) ||
             (scalar_equal(s1, target) && scalar_is_nonneg_expr(ctx, s0, 0));

   case nir_op_ine:
      if (truth)
         return false;
      return (scalar_equal(s0, target) && scalar_is_nonneg_expr(ctx, s1, 0)) ||
             (scalar_equal(s1, target) && scalar_is_nonneg_expr(ctx, s0, 0));

   default:
      return false;
   }
}

static bool
is_nonneg_at_block(struct nonneg_ctx *ctx, nir_scalar target, nir_block *block)
{
   target = nir_scalar_chase_movs(target);
   if (scalar_is_nonneg_expr(ctx, target, 0))
      return true;

   for (nir_cf_node *node = &block->cf_node; node->parent; node = node->parent) {
      if (node->parent->type != nir_cf_node_if)
         continue;

      nir_if *nif = nir_cf_node_as_if(node->parent);
      bool in_then = false;
      bool in_else = false;
      foreach_list_typed(nir_cf_node, child, node, &nif->then_list) {
         if (child == node) {
            in_then = true;
            break;
         }
      }
      if (!in_then) {
         foreach_list_typed(nir_cf_node, child, node, &nif->else_list) {
            if (child == node) {
               in_else = true;
               break;
            }
         }
      }
      if ((in_then || in_else) &&
          cond_implies_nonneg(ctx, nir_scalar_resolved(nif->condition.ssa, 0),
                              target, in_then, 0))
         return true;
   }

   return false;
}

static bool
opt_nonneg_idiv_instr(nir_builder *b, nir_alu_instr *alu, void *data)
{
   struct nonneg_ctx *ctx = data;

   if (alu->def.bit_size != 32 || alu->def.num_components != 1)
      return false;

   const nir_op op = alu->op;
   if (op != nir_op_idiv && op != nir_op_imod && op != nir_op_irem)
      return false;

   nir_scalar sa = nir_scalar_resolved(alu->src[0].src.ssa, alu->src[0].swizzle[0]);
   nir_scalar sb = nir_scalar_resolved(alu->src[1].src.ssa, alu->src[1].swizzle[0]);
   nir_block *block = alu->instr.block;

   const bool nonneg_a = is_nonneg_at_block(ctx, sa, block);
   const bool nonneg_b = is_nonneg_at_block(ctx, sb, block);
   if (!nonneg_a && !nonneg_b)
      return false;

   /* Keep constant-divisor signed division with unknown-sign dividend for
    * nir_opt_idiv_const, which emits tighter signed magic-multiply sequences. */
   if (!nonneg_a && nir_scalar_is_const(sb))
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *d = nir_ssa_for_alu_src(b, alu, 1);
   nir_def *res = NULL;

   if (nonneg_a && nonneg_b) {
      res = (op == nir_op_idiv) ? nir_udiv(b, a, d) : nir_umod(b, a, d);
   } else if (nonneg_a) {
      nir_def *abs_d = nir_iabs(b, d);
      if (op == nir_op_irem) {
         res = nir_umod(b, a, abs_d);
      } else if (op == nir_op_idiv) {
         nir_def *q = nir_udiv(b, a, abs_d);
         res = nir_bcsel(b, nir_ilt_imm(b, d, 0), nir_ineg(b, q), q);
      } else {
         nir_def *r = nir_umod(b, a, abs_d);
         nir_def *neg_adj = nir_bcsel(b, nir_ilt_imm(b, d, 0),
                                      d, nir_imm_int(b, 0));
         res = nir_bcsel(b, nir_ieq_imm(b, r, 0),
                         nir_imm_int(b, 0), nir_iadd(b, r, neg_adj));
      }
   } else {
      nir_def *abs_a = nir_iabs(b, a);
      nir_def *neg_a = nir_ilt_imm(b, a, 0);
      if (op == nir_op_idiv) {
         nir_def *q = nir_udiv(b, abs_a, d);
         res = nir_bcsel(b, neg_a, nir_ineg(b, q), q);
      } else if (op == nir_op_irem) {
         nir_def *r = nir_umod(b, abs_a, d);
         res = nir_bcsel(b, neg_a, nir_ineg(b, r), r);
      } else {
         nir_def *r = nir_umod(b, abs_a, d);
         nir_def *neg_r = nir_bcsel(b, nir_ieq_imm(b, r, 0),
                                    nir_imm_int(b, 0), nir_isub(b, d, r));
         res = nir_bcsel(b, neg_a, neg_r, r);
      }
   }

   nir_def_replace(&alu->def, res);
   return true;
}

bool
agx_nir_opt_nonneg_idiv(nir_shader *shader)
{
   struct nonneg_ctx ctx = {
      .shader = shader,
      .range_ht = _mesa_pointer_hash_table_create(NULL),
   };

   bool progress = nir_shader_alu_pass(shader, opt_nonneg_idiv_instr,
                                       nir_metadata_control_flow, &ctx);

   _mesa_hash_table_destroy(ctx.range_ht, NULL);
   return progress;
}

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

enum hk_pack_loop_param {
   HK_PACK_LOOP_PARAM_IM2COL_KH = 0,
   HK_PACK_LOOP_PARAM_DW_KH,
   HK_PACK_LOOP_PARAM_DW_KW,
   HK_PACK_LOOP_PARAM_GEMM_K_ITERS,
   HK_PACK_LOOP_PARAM_PAD_H,
   HK_PACK_LOOP_PARAM_PAD_W,
   HK_PACK_LOOP_PARAM_PAD_ZLP_ITERS,
   HK_PACK_LOOP_PARAM_GEMM_K,
   HK_PACK_LOOP_PARAM_COUNT,
};

enum hk_pack_loop_op_kind {
   HK_PACK_LOOP_OP_CONST = 0,
   HK_PACK_LOOP_OP_PARAM,
   HK_PACK_LOOP_OP_ADD,
   HK_PACK_LOOP_OP_MUL,
   HK_PACK_LOOP_OP_MAX,
};

struct hk_pack_loop_op {
   uint8_t kind;
   uint8_t lhs;
   uint8_t rhs;
   uint32_t imm;
};

#define HK_PACK_LOOP_MAX_OPS 64u
#define HK_PACK_LOOP_WORK_MAX_OPS 128u

/* Post-order expressions derived directly from the emitted NIR control-flow
 * tree:
 * - ops[0..count): worst-case (total loop back-edges + 1). valid == false means
 *   at least one loop could not be bounded statically, forcing pack_loops.
 * - work_ops[0..work_count): worst-case dynamic weighted executable NIR ops
 *   along any control-flow path, paired with static_nir_ops (sum across all
 *   blocks) to tighten per-thread instruction bounds for multi-branch shaders
 *   without underestimating loop work. */
struct hk_pack_loop_bound {
   bool valid;
   bool work_valid;
   uint8_t count;
   uint8_t work_count;
   uint32_t static_nir_ops;
   struct hk_pack_loop_op ops[HK_PACK_LOOP_MAX_OPS];
   struct hk_pack_loop_op work_ops[HK_PACK_LOOP_WORK_MAX_OPS];
};

static inline bool
hk_pack_eval_ops(const struct hk_pack_loop_op *ops, uint8_t count,
                 uint8_t max_ops,
                 const uint64_t params[HK_PACK_LOOP_PARAM_COUNT],
                 uint64_t *val_out)
{
   *val_out = 0;
   if (!ops || count == 0 || count > max_ops || max_ops > HK_PACK_LOOP_WORK_MAX_OPS)
      return false;

   uint64_t vals[HK_PACK_LOOP_WORK_MAX_OPS];
   for (uint8_t i = 0; i < count; ++i) {
      const struct hk_pack_loop_op op = ops[i];
      uint64_t v = 0;
      switch (op.kind) {
      case HK_PACK_LOOP_OP_CONST:
         v = op.imm;
         break;
      case HK_PACK_LOOP_OP_PARAM:
         if (!params || op.imm >= HK_PACK_LOOP_PARAM_COUNT)
            return false;
         v = params[op.imm];
         break;
      case HK_PACK_LOOP_OP_ADD:
         if (op.lhs >= i || op.rhs >= i ||
             __builtin_add_overflow(vals[op.lhs], vals[op.rhs], &v))
            return false;
         break;
      case HK_PACK_LOOP_OP_MUL:
         if (op.lhs >= i || op.rhs >= i ||
             __builtin_mul_overflow(vals[op.lhs], vals[op.rhs], &v))
            return false;
         break;
      case HK_PACK_LOOP_OP_MAX:
         if (op.lhs >= i || op.rhs >= i)
            return false;
         v = vals[op.lhs] >= vals[op.rhs] ? vals[op.lhs] : vals[op.rhs];
         break;
      default:
         return false;
      }
      vals[i] = v;
   }

   *val_out = vals[count - 1];
   return true;
}

static inline bool
hk_pack_eval_loop_bound(const struct hk_pack_loop_bound *lb,
                        const uint64_t params[HK_PACK_LOOP_PARAM_COUNT],
                        uint64_t *bound_out)
{
   *bound_out = 0;
   if (!lb || !lb->valid)
      return false;
   uint64_t res = 0;
   if (!hk_pack_eval_ops(lb->ops, lb->count, HK_PACK_LOOP_MAX_OPS, params, &res) ||
       res == 0)
      return false;
   *bound_out = res;
   return true;
}

static inline bool
hk_pack_eval_work_bound(const struct hk_pack_loop_bound *lb,
                        const uint64_t params[HK_PACK_LOOP_PARAM_COUNT],
                        uint64_t *dyn_nir_ops_out)
{
   *dyn_nir_ops_out = 0;
   if (!lb || !lb->valid || !lb->work_valid || lb->static_nir_ops == 0)
      return false;
   uint64_t res = 0;
   if (!hk_pack_eval_ops(lb->work_ops, lb->work_count, HK_PACK_LOOP_WORK_MAX_OPS,
                         params, &res) ||
       res == 0)
      return false;
   *dyn_nir_ops_out = res;
   return true;
}

/* Compute an upper bound on per-thread AGX instructions for a bounded looping
 * shader. Always requires a valid loop back-edge bound (`lb->valid`); when the
 * CF-weighted NIR work bound is also available, scales `agx_instrs` by
 * ceil(2 * dyn_nir_ops / static_nir_ops) (clamped to [agx_instrs, agx_instrs *
 * loop_bound]) so cold mutually-exclusive paths do not multiply loop trips. */
static inline bool
hk_pack_eval_effective_instrs(const struct hk_pack_loop_bound *lb,
                              uint64_t agx_instrs,
                              const uint64_t params[HK_PACK_LOOP_PARAM_COUNT],
                              uint64_t *eff_instrs_out)
{
   *eff_instrs_out = 0;
   if (!agx_instrs)
      return false;

   uint64_t loop_bound = 0;
   if (!hk_pack_eval_loop_bound(lb, params, &loop_bound))
      return false;

   uint64_t naive = 0;
   const bool naive_ok = !__builtin_mul_overflow(agx_instrs, loop_bound, &naive);

   uint64_t dyn_nir_ops = 0;
   if (hk_pack_eval_work_bound(lb, params, &dyn_nir_ops)) {
      const __uint128_t num = (__uint128_t)2u * (__uint128_t)agx_instrs *
                              (__uint128_t)dyn_nir_ops;
      const __uint128_t scaled =
         (num + (__uint128_t)lb->static_nir_ops - 1u) /
         (__uint128_t)lb->static_nir_ops;
      if (scaled <= UINT64_MAX) {
         uint64_t eff = (uint64_t)scaled;
         if (eff < agx_instrs)
            eff = agx_instrs;
         if (naive_ok && eff > naive)
            eff = naive;
         *eff_instrs_out = eff;
         return true;
      }
   }

   if (!naive_ok)
      return false;
   *eff_instrs_out = naive;
   return true;
}

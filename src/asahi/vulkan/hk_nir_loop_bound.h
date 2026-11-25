/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string.h>
#include "compiler/nir/nir.h"
#include "hk_pack_loop_bound.h"

struct hk_nir_loop_param_binding {
   nir_loop *loop;
   enum hk_pack_loop_param param;
};

#define HK_NIR_LOOP_MAX_BINDINGS 16u

struct hk_nir_loop_binding_list {
   struct hk_nir_loop_param_binding entries[HK_NIR_LOOP_MAX_BINDINGS];
   unsigned count;
   bool overflow;
};

static inline void
hk_nir_loop_binding_add(struct hk_nir_loop_binding_list *list,
                        nir_loop *loop, enum hk_pack_loop_param param)
{
   if (!list || !loop)
      return;
   if (list->count >= HK_NIR_LOOP_MAX_BINDINGS) {
      list->overflow = true;
      return;
   }
   list->entries[list->count++] = (struct hk_nir_loop_param_binding){
      .loop = loop,
      .param = param,
   };
}

struct hk_nir_lb_builder {
   struct hk_pack_loop_op *ops;
   uint8_t *count;
   uint8_t max_ops;
   const struct hk_nir_loop_param_binding *bindings;
   unsigned num_bindings;
   bool failed;
};

static inline int
hk_nir_lb_emit(struct hk_nir_lb_builder *b, struct hk_pack_loop_op op)
{
   if (b->failed)
      return -1;
   for (uint8_t i = 0; i < *b->count; ++i) {
      const struct hk_pack_loop_op *e = &b->ops[i];
      if (e->kind == op.kind && e->lhs == op.lhs && e->rhs == op.rhs &&
          e->imm == op.imm)
         return (int)i;
   }
   if (*b->count >= b->max_ops) {
      b->failed = true;
      return -1;
   }
   const int idx = (int)(*b->count)++;
   b->ops[idx] = op;
   return idx;
}

static inline int
hk_nir_lb_const(struct hk_nir_lb_builder *b, uint32_t imm)
{
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_CONST, .imm = imm,
   });
}

static inline int
hk_nir_lb_param(struct hk_nir_lb_builder *b, enum hk_pack_loop_param param)
{
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_PARAM, .imm = (uint32_t)param,
   });
}

static inline bool
hk_nir_lb_is_const(const struct hk_nir_lb_builder *b, int idx, uint32_t *imm_out)
{
   if (idx < 0 || b->failed)
      return false;
   const struct hk_pack_loop_op op = b->ops[idx];
   if (op.kind != HK_PACK_LOOP_OP_CONST)
      return false;
   if (imm_out)
      *imm_out = op.imm;
   return true;
}

static inline int
hk_nir_lb_add(struct hk_nir_lb_builder *b, int lhs, int rhs)
{
   if (b->failed || lhs < 0 || rhs < 0)
      return -1;
   uint32_t a = 0, c = 0;
   const bool ca = hk_nir_lb_is_const(b, lhs, &a);
   const bool cc = hk_nir_lb_is_const(b, rhs, &c);
   if (ca && a == 0)
      return rhs;
   if (cc && c == 0)
      return lhs;
   if (ca && cc && (uint64_t)a + (uint64_t)c <= UINT32_MAX)
      return hk_nir_lb_const(b, a + c);
   if (lhs > rhs) {
      const int tmp = lhs;
      lhs = rhs;
      rhs = tmp;
   }
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_ADD, .lhs = (uint8_t)lhs, .rhs = (uint8_t)rhs,
   });
}

static inline int
hk_nir_lb_mul(struct hk_nir_lb_builder *b, int lhs, int rhs)
{
   if (b->failed || lhs < 0 || rhs < 0)
      return -1;
   uint32_t a = 0, c = 0;
   const bool ca = hk_nir_lb_is_const(b, lhs, &a);
   const bool cc = hk_nir_lb_is_const(b, rhs, &c);
   if ((ca && a == 0) || (cc && c == 0))
      return hk_nir_lb_const(b, 0);
   if (ca && a == 1)
      return rhs;
   if (cc && c == 1)
      return lhs;
   if (ca && cc && (uint64_t)a * (uint64_t)c <= UINT32_MAX)
      return hk_nir_lb_const(b, a * c);
   if (lhs > rhs) {
      const int tmp = lhs;
      lhs = rhs;
      rhs = tmp;
   }
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_MUL, .lhs = (uint8_t)lhs, .rhs = (uint8_t)rhs,
   });
}

static inline int
hk_nir_lb_max(struct hk_nir_lb_builder *b, int lhs, int rhs)
{
   if (b->failed || lhs < 0 || rhs < 0)
      return -1;
   if (lhs == rhs)
      return lhs;
   uint32_t a = 0, c = 0;
   const bool ca = hk_nir_lb_is_const(b, lhs, &a);
   const bool cc = hk_nir_lb_is_const(b, rhs, &c);
   if (ca && a == 0)
      return rhs;
   if (cc && c == 0)
      return lhs;
   if (ca && cc)
      return hk_nir_lb_const(b, a >= c ? a : c);
   if (lhs > rhs) {
      const int tmp = lhs;
      lhs = rhs;
      rhs = tmp;
   }
   return hk_nir_lb_emit(b, (struct hk_pack_loop_op){
      .kind = HK_PACK_LOOP_OP_MAX, .lhs = (uint8_t)lhs, .rhs = (uint8_t)rhs,
   });
}

/* Affine symbolic value: pure constant `c` when `node < 0`, or `ops[node] + c`
 * when `node >= 0`. Keeping additive constants out-of-band avoids emitting
 * dead intermediate CONST ops during CF walks, preventing op-budget overflows
 * on multi-branch shaders. */
struct hk_nir_lb_val {
   int node;
   uint64_t c;
};

static inline struct hk_nir_lb_val
hk_nir_lb_val_const(uint64_t c)
{
   return (struct hk_nir_lb_val){ .node = -1, .c = c };
}

static inline int
hk_nir_lb_val_materialize(struct hk_nir_lb_builder *b, struct hk_nir_lb_val v)
{
   if (b->failed)
      return -1;
   if (v.c > UINT32_MAX) {
      b->failed = true;
      return -1;
   }
   if (v.node < 0)
      return hk_nir_lb_const(b, (uint32_t)v.c);
   if (v.c == 0)
      return v.node;
   return hk_nir_lb_add(b, v.node, hk_nir_lb_const(b, (uint32_t)v.c));
}

static inline struct hk_nir_lb_val
hk_nir_lb_val_add(struct hk_nir_lb_builder *b, struct hk_nir_lb_val a,
                  struct hk_nir_lb_val d)
{
   if (b->failed)
      return hk_nir_lb_val_const(0);
   uint64_t sum_c = 0;
   if (__builtin_add_overflow(a.c, d.c, &sum_c) || sum_c > UINT32_MAX) {
      b->failed = true;
      return hk_nir_lb_val_const(0);
   }
   if (a.node < 0 && d.node < 0)
      return hk_nir_lb_val_const(sum_c);
   if (a.node < 0)
      return (struct hk_nir_lb_val){ .node = d.node, .c = sum_c };
   if (d.node < 0)
      return (struct hk_nir_lb_val){ .node = a.node, .c = sum_c };
   return (struct hk_nir_lb_val){
      .node = hk_nir_lb_add(b, a.node, d.node),
      .c = sum_c,
   };
}

static inline struct hk_nir_lb_val
hk_nir_lb_val_max(struct hk_nir_lb_builder *b, struct hk_nir_lb_val a,
                  struct hk_nir_lb_val d)
{
   if (b->failed)
      return hk_nir_lb_val_const(0);
   if (a.node < 0 && d.node < 0)
      return hk_nir_lb_val_const(a.c >= d.c ? a.c : d.c);
   if (a.node == d.node)
      return (struct hk_nir_lb_val){ .node = a.node, .c = a.c >= d.c ? a.c : d.c };
   /* All symbolic ops are non-negative, so ops[node] + c >= c. */
   if (a.node >= 0 && d.node < 0 && d.c <= a.c)
      return a;
   if (d.node >= 0 && a.node < 0 && a.c <= d.c)
      return d;
   const uint64_t min_c = a.c < d.c ? a.c : d.c;
   const int la = hk_nir_lb_val_materialize(
      b, (struct hk_nir_lb_val){ .node = a.node, .c = a.c - min_c });
   const int ld = hk_nir_lb_val_materialize(
      b, (struct hk_nir_lb_val){ .node = d.node, .c = d.c - min_c });
   return (struct hk_nir_lb_val){
      .node = hk_nir_lb_max(b, la, ld),
      .c = min_c,
   };
}

static inline struct hk_nir_lb_val
hk_nir_lb_val_mul(struct hk_nir_lb_builder *b, struct hk_nir_lb_val a,
                  struct hk_nir_lb_val d)
{
   if (b->failed)
      return hk_nir_lb_val_const(0);
   if ((a.node < 0 && a.c == 0) || (d.node < 0 && d.c == 0))
      return hk_nir_lb_val_const(0);
   if (a.node < 0 && d.node < 0) {
      uint64_t prod = 0;
      if (__builtin_mul_overflow(a.c, d.c, &prod) || prod > UINT32_MAX) {
         b->failed = true;
         return hk_nir_lb_val_const(0);
      }
      return hk_nir_lb_val_const(prod);
   }
   if (a.node < 0 && a.c == 1)
      return d;
   if (d.node < 0 && d.c == 1)
      return a;
   if (a.node < 0 && a.c <= UINT32_MAX) {
      uint64_t prod_c = 0;
      if (!__builtin_mul_overflow(a.c, d.c, &prod_c) && prod_c <= UINT32_MAX) {
         return (struct hk_nir_lb_val){
            .node = hk_nir_lb_mul(b, hk_nir_lb_const(b, (uint32_t)a.c), d.node),
            .c = prod_c,
         };
      }
   }
   if (d.node < 0 && d.c <= UINT32_MAX) {
      uint64_t prod_c = 0;
      if (!__builtin_mul_overflow(d.c, a.c, &prod_c) && prod_c <= UINT32_MAX) {
         return (struct hk_nir_lb_val){
            .node = hk_nir_lb_mul(b, hk_nir_lb_const(b, (uint32_t)d.c), a.node),
            .c = prod_c,
         };
      }
   }
   const int la = hk_nir_lb_val_materialize(b, a);
   const int ld = hk_nir_lb_val_materialize(b, d);
   return (struct hk_nir_lb_val){
      .node = hk_nir_lb_mul(b, la, ld),
      .c = 0,
   };
}

static inline bool
hk_nir_lb_is_simple_break_if(nir_if *nif)
{
   if (exec_list_is_empty(&nif->then_list) ||
       exec_list_get_head(&nif->then_list) !=
          exec_list_get_tail(&nif->then_list))
      return false;
   nir_cf_node *tn =
      exec_node_data(nir_cf_node, exec_list_get_head(&nif->then_list), node);
   if (tn->type != nir_cf_node_block)
      return false;
   nir_block *tb = nir_cf_node_as_block(tn);
   if (exec_list_is_empty(&tb->instr_list) ||
       exec_list_get_head(&tb->instr_list) !=
          exec_list_get_tail(&tb->instr_list))
      return false;
   nir_instr *ti =
      exec_node_data(nir_instr, exec_list_get_head(&tb->instr_list), node);
   if (ti->type != nir_instr_type_jump ||
       nir_instr_as_jump(ti)->type != nir_jump_break)
      return false;

   if (exec_list_is_empty(&nif->else_list) ||
       exec_list_get_head(&nif->else_list) !=
          exec_list_get_tail(&nif->else_list))
      return false;
   nir_cf_node *en =
      exec_node_data(nir_cf_node, exec_list_get_head(&nif->else_list), node);
   if (en->type != nir_cf_node_block)
      return false;
   return exec_list_is_empty(&nir_cf_node_as_block(en)->instr_list);
}

static inline struct hk_nir_lb_val
hk_nir_lb_loop_trips(struct hk_nir_lb_builder *b, nir_loop *loop)
{
   nir_cf_node *prev_cf = nir_cf_node_prev(&loop->cf_node);
   if (!prev_cf || prev_cf->type != nir_cf_node_block) {
      b->failed = true;
      return hk_nir_lb_val_const(0);
   }
   nir_block *pre = nir_cf_node_as_block(prev_cf);
   if (exec_list_is_empty(&loop->body)) {
      b->failed = true;
      return hk_nir_lb_val_const(0);
   }
   nir_cf_node *first_cf =
      exec_node_data(nir_cf_node, exec_list_get_head(&loop->body), node);
   if (first_cf->type != nir_cf_node_block) {
      b->failed = true;
      return hk_nir_lb_val_const(0);
   }
   nir_block *hdr = nir_cf_node_as_block(first_cf);
   nir_cf_node *second_cf = nir_cf_node_next(first_cf);
   if (!second_cf || second_cf->type != nir_cf_node_if) {
      b->failed = true;
      return hk_nir_lb_val_const(0);
   }
   nir_if *brk_if = nir_cf_node_as_if(second_cf);
   if (!hk_nir_lb_is_simple_break_if(brk_if)) {
      b->failed = true;
      return hk_nir_lb_val_const(0);
   }

   /* If the break condition is a disjunction (A || B), breaking occurs at least
    * as soon as either disjunct becomes true, so proving a trip bound from any
    * disjunct is a sound upper bound. */
   const nir_def *cands[2] = { brk_if->condition.ssa, NULL };
   unsigned num_cands = 1;
   if (cands[0] && nir_def_instr(cands[0])->type == nir_instr_type_alu) {
      nir_alu_instr *top = nir_instr_as_alu(nir_def_instr(cands[0]));
      if (top->op == nir_op_ior) {
         cands[0] = top->src[0].src.ssa;
         cands[1] = top->src[1].src.ssa;
         num_cands = 2;
      }
   }

   bool found_canonical = false;
   bool found_const = false;
   uint64_t best_const_trips = UINT64_MAX;
   for (unsigned ci = 0; ci < num_cands; ++ci) {
      const nir_def *cond_def = cands[ci];
      if (!cond_def || nir_def_instr(cond_def)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *cmp = nir_instr_as_alu(nir_def_instr(cond_def));
      if (cmp->op != nir_op_ige && cmp->op != nir_op_uge)
         continue;

      const nir_def *iv_def = cmp->src[0].src.ssa;
      if (!iv_def || iv_def->num_components != 1 || iv_def->bit_size != 32 ||
          nir_def_instr(iv_def)->type != nir_instr_type_phi ||
          nir_def_instr(iv_def)->block != hdr)
         continue;

      nir_phi_instr *iv = nir_instr_as_phi(nir_def_instr(iv_def));
      const nir_src *init_src = NULL;
      const nir_src *back_src = NULL;
      unsigned src_count = 0;
      nir_foreach_phi_src(ps, iv) {
         src_count++;
         if (ps->pred == pre)
            init_src = &ps->src;
         else
            back_src = &ps->src;
      }
      if (src_count != 2 || !init_src || !back_src ||
          nir_def_instr(back_src->ssa)->type != nir_instr_type_alu)
         continue;

      nir_alu_instr *step_alu = nir_instr_as_alu(nir_def_instr(back_src->ssa));
      if (step_alu->op != nir_op_iadd)
         continue;
      const nir_src *step_src = NULL;
      if (step_alu->src[0].src.ssa == iv_def)
         step_src = &step_alu->src[1].src;
      else if (step_alu->src[1].src.ssa == iv_def)
         step_src = &step_alu->src[0].src;
      else
         continue;

      found_canonical = true;
      const nir_src *limit_src = &cmp->src[1].src;
      if (nir_src_is_const(*init_src) && nir_src_is_const(*step_src) &&
          nir_src_is_const(*limit_src)) {
         uint64_t trips = 0;
         bool ok = true;
         if (cmp->op == nir_op_ige) {
            const int64_t init_i = nir_src_as_int(*init_src);
            const int64_t step_i = nir_src_as_int(*step_src);
            const int64_t limit_i = nir_src_as_int(*limit_src);
            if (step_i <= 0)
               ok = false;
            else if (limit_i > init_i)
               trips = ((uint64_t)(limit_i - init_i) + (uint64_t)step_i - 1ull) / (uint64_t)step_i;
         } else {
            const uint64_t init_u = nir_src_as_uint(*init_src);
            const uint64_t step_u = nir_src_as_uint(*step_src);
            const uint64_t limit_u = nir_src_as_uint(*limit_src);
            if (step_u == 0)
               ok = false;
            else if (limit_u > init_u) {
               const uint64_t span = limit_u - init_u;
               if (span > UINT64_MAX - (step_u - 1ull))
                  ok = false;
               else
                  trips = (span + step_u - 1ull) / step_u;
            }
         }
         if (ok && trips <= UINT32_MAX && trips < best_const_trips) {
            best_const_trips = trips;
            found_const = true;
         }
      }
   }

   if (!found_canonical) {
      b->failed = true;
      return hk_nir_lb_val_const(0);
   }

   /* Check explicit dynamic parameter binding first (structure already verified). */
   for (unsigned i = 0; i < b->num_bindings; ++i) {
      if (b->bindings[i].loop == loop) {
         return (struct hk_nir_lb_val){
            .node = hk_nir_lb_param(b, b->bindings[i].param),
            .c = 0,
         };
      }
   }

   if (!found_const) {
      b->failed = true;
      return hk_nir_lb_val_const(0);
   }
   return hk_nir_lb_val_const(best_const_trips);
}

static inline struct hk_nir_lb_val
hk_nir_lb_walk_list(struct hk_nir_lb_builder *b, const struct exec_list *cf_list)
{
   struct hk_nir_lb_val total = hk_nir_lb_val_const(0);
   foreach_list_typed(nir_cf_node, cf, node, cf_list) {
      if (b->failed)
         return hk_nir_lb_val_const(0);
      if (cf->type == nir_cf_node_block) {
         continue;
      } else if (cf->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(cf);
         const struct hk_nir_lb_val t = hk_nir_lb_walk_list(b, &nif->then_list);
         const struct hk_nir_lb_val e = hk_nir_lb_walk_list(b, &nif->else_list);
         total = hk_nir_lb_val_add(b, total, hk_nir_lb_val_max(b, t, e));
      } else if (cf->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(cf);
         const struct hk_nir_lb_val trips = hk_nir_lb_loop_trips(b, loop);
         const struct hk_nir_lb_val body = hk_nir_lb_walk_list(b, &loop->body);
         const struct hk_nir_lb_val per_iter =
            hk_nir_lb_val_add(b, hk_nir_lb_val_const(1), body);
         const struct hk_nir_lb_val loop_edges =
            hk_nir_lb_val_mul(b, trips, per_iter);
         total = hk_nir_lb_val_add(b, total, loop_edges);
      } else {
         b->failed = true;
         return hk_nir_lb_val_const(0);
      }
   }
   return total;
}

static inline uint32_t
hk_nir_lb_block_weight(const nir_block *blk)
{
   uint32_t w = 0;
   nir_foreach_instr(instr, blk) {
      switch (instr->type) {
      case nir_instr_type_alu:
      case nir_instr_type_tex:
      case nir_instr_type_jump:
         w++;
         break;
      case nir_instr_type_intrinsic: {
         const nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic == nir_intrinsic_cmat_muladd_agx ||
             intrin->intrinsic == nir_intrinsic_barrier)
            w += 4;
         else
            w++;
         break;
      }
      default:
         break;
      }
   }
   return w;
}

static inline struct hk_nir_lb_val
hk_nir_lb_walk_work_list(struct hk_nir_lb_builder *b,
                         const struct exec_list *cf_list)
{
   struct hk_nir_lb_val total = hk_nir_lb_val_const(0);
   foreach_list_typed(nir_cf_node, cf, node, cf_list) {
      if (b->failed)
         return hk_nir_lb_val_const(0);
      if (cf->type == nir_cf_node_block) {
         const uint32_t w = hk_nir_lb_block_weight(nir_cf_node_as_block(cf));
         if (w > 0)
            total = hk_nir_lb_val_add(b, total, hk_nir_lb_val_const(w));
      } else if (cf->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(cf);
         const struct hk_nir_lb_val t = hk_nir_lb_walk_work_list(b, &nif->then_list);
         const struct hk_nir_lb_val e = hk_nir_lb_walk_work_list(b, &nif->else_list);
         total = hk_nir_lb_val_add(b, total, hk_nir_lb_val_max(b, t, e));
      } else if (cf->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(cf);
         const struct hk_nir_lb_val trips = hk_nir_lb_loop_trips(b, loop);
         const struct hk_nir_lb_val body = hk_nir_lb_walk_work_list(b, &loop->body);
         const nir_block *hdr = nir_loop_first_block(loop);
         const uint32_t hdr_w = hk_nir_lb_block_weight(hdr) + 1u;
         const struct hk_nir_lb_val loop_work =
            hk_nir_lb_val_add(b, hk_nir_lb_val_mul(b, trips, body),
                              hk_nir_lb_val_const(hdr_w));
         total = hk_nir_lb_val_add(b, total, loop_work);
      } else {
         b->failed = true;
         return hk_nir_lb_val_const(0);
      }
   }
   return total;
}

/* Analyze the emitted NIR control-flow tree and populate `out` with an
 * expression computing worst-case (total loop back-edges + 1), plus a
 * CF-weighted dynamic executable NIR op bound (`work_ops` / `static_nir_ops`).
 * Fails closed (`out->valid = false`) if any loop cannot be bounded statically
 * or via a verified dynamic parameter binding. */
static inline bool
hk_nir_compute_pack_loop_bound(nir_shader *nir,
                               const struct hk_nir_loop_param_binding *bindings,
                               unsigned num_bindings,
                               struct hk_pack_loop_bound *out)
{
   if (!out)
      return false;
   memset(out, 0, sizeof(*out));
   if (!nir)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint((nir_shader *)nir);
   if (!impl)
      return false;

   struct hk_nir_lb_builder b = {
      .ops = out->ops,
      .count = &out->count,
      .max_ops = HK_PACK_LOOP_MAX_OPS,
      .bindings = bindings,
      .num_bindings = num_bindings,
      .failed = false,
   };

   const struct hk_nir_lb_val edges = hk_nir_lb_walk_list(&b, &impl->body);
   const struct hk_nir_lb_val bound_v =
      hk_nir_lb_val_add(&b, edges, hk_nir_lb_val_const(1));
   const int bound = hk_nir_lb_val_materialize(&b, bound_v);
   if (b.failed || bound < 0 || out->count == 0) {
      memset(out, 0, sizeof(*out));
      return false;
   }
   /* Ensure the final expression value sits at ops[count - 1]. */
   if (bound != (int)out->count - 1) {
      if (out->count >= HK_PACK_LOOP_MAX_OPS) {
         memset(out, 0, sizeof(*out));
         return false;
      }
      out->ops[out->count] = out->ops[bound];
      out->count++;
   }
   out->valid = true;

   /* Best-effort CF-weighted executable NIR op bound (falls back cleanly to
    * back-edges + 1 if work_ops exceeds HK_PACK_LOOP_WORK_MAX_OPS). */
   uint64_t static_ops = 0;
   nir_foreach_block(blk, impl) {
      static_ops += hk_nir_lb_block_weight(blk);
   }
   if (static_ops > 0 && static_ops <= UINT32_MAX) {
      struct hk_nir_lb_builder wb = {
         .ops = out->work_ops,
         .count = &out->work_count,
         .max_ops = HK_PACK_LOOP_WORK_MAX_OPS,
         .bindings = bindings,
         .num_bindings = num_bindings,
         .failed = false,
      };
      const struct hk_nir_lb_val work_v =
         hk_nir_lb_walk_work_list(&wb, &impl->body);
      const int work = hk_nir_lb_val_materialize(&wb, work_v);
      if (!wb.failed && work >= 0 && out->work_count > 0) {
         bool ok = true;
         if (work != (int)out->work_count - 1) {
            if (out->work_count >= HK_PACK_LOOP_WORK_MAX_OPS) {
               ok = false;
            } else {
               out->work_ops[out->work_count] = out->work_ops[work];
               out->work_count++;
            }
         }
         if (ok) {
            out->static_nir_ops = (uint32_t)static_ops;
            out->work_valid = true;
         } else {
            out->work_count = 0;
         }
      } else {
         out->work_count = 0;
      }
   }
   return true;
}

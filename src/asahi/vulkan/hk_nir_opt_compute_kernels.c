/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_compute_opt.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_physical_device.h"

#include "compiler/glsl_types.h"
#include "nir_builder.h"
#include "nir_deref.h"
#include "util/u_math.h"

#include <math.h>

/* ---------------------------------------------------------------------------
 * Canonicalization & structural matching helpers
 * --------------------------------------------------------------------------- */

static nir_def *
strip_mov(nir_def *def)
{
   while (def && nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
      if (alu->op == nir_op_mov && alu->src[0].src.ssa->num_components == 1) {
         def = alu->src[0].src.ssa;
         continue;
      }
      break;
   }
   return def;
}

static bool
match_const_i32(nir_def *def, int32_t *val_out)
{
   def = strip_mov(def);
   if (!def || def->bit_size != 32 || def->num_components != 1)
      return false;
   if (nir_def_instr(def)->type != nir_instr_type_load_const)
      return false;
   *val_out = nir_src_as_int(nir_src_for_ssa(def));
   return true;
}

static bool
match_const_u32(nir_def *def, uint32_t *val_out)
{
   def = strip_mov(def);
   if (!def || def->bit_size != 32 || def->num_components != 1)
      return false;
   if (nir_def_instr(def)->type != nir_instr_type_load_const)
      return false;
   *val_out = nir_src_as_uint(nir_src_for_ssa(def));
   return true;
}

static bool
match_const_f32_bits(nir_def *def, uint32_t *bits_out)
{
   return match_const_u32(def, bits_out);
}

static bool
is_const_i32(nir_def *def, int32_t expected)
{
   int32_t v = 0;
   return match_const_i32(def, &v) && v == expected;
}

static bool
is_const_bool(nir_def *def, bool expected)
{
   def = strip_mov(def);
   if (!def || def->bit_size != 1 || def->num_components != 1)
      return false;
   if (nir_def_instr(def)->type != nir_instr_type_load_const)
      return false;
   return nir_src_as_bool(nir_src_for_ssa(def)) == expected;
}

static bool
match_const_u32_any(nir_def *def, uint32_t *val_out)
{
   def = strip_mov(def);
   if (!def || def->num_components != 1 ||
       (def->bit_size != 8 && def->bit_size != 16 && def->bit_size != 32))
      return false;
   if (nir_def_instr(def)->type != nir_instr_type_load_const)
      return false;
   *val_out = (uint32_t)nir_src_as_uint(nir_src_for_ssa(def));
   return true;
}

static bool
is_const_u_any(nir_def *def, uint32_t expected)
{
   uint32_t v = 0;
   return match_const_u32_any(def, &v) && v == expected;
}

static bool
match_push_const_u32(nir_def *def, uint16_t *offset_out)
{
   def = strip_mov(def);
   if (!def || def->bit_size != 32 || def->num_components != 1)
      return false;

   unsigned comp = 0;
   if (nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
      if (alu->op != nir_op_mov)
         return false;
      comp = alu->src[0].swizzle[0];
      def = alu->src[0].src.ssa;
   }

   if (!def || def->bit_size != 32 ||
       nir_def_instr(def)->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(def));
   if (intr->intrinsic != nir_intrinsic_load_push_constant)
      return false;
   if (!nir_src_is_const(intr->src[0]))
      return false;

   uint64_t off = (uint64_t)nir_intrinsic_base(intr) +
                  nir_src_as_uint(intr->src[0]) + (uint64_t)comp * 4u;
   if ((off & 3u) != 0 || off + 4u > HK_MAX_PUSH_SIZE)
      return false;

   *offset_out = (uint16_t)off;
   return true;
}

static bool
match_sysval_comp(nir_def *def, nir_intrinsic_op op, unsigned *comp_out)
{
   if (!def || def->bit_size != 32 || def->num_components != 1)
      return false;

   if (nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
      if (alu->op == nir_op_mov) {
         nir_def *src = alu->src[0].src.ssa;
         if (nir_def_instr(src)->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(nir_def_instr(src))->intrinsic == op) {
            *comp_out = alu->src[0].swizzle[0];
            return true;
         }
      }
   }
   return false;
}

static bool
match_sysval_alu_src(const nir_alu_src *asrc, nir_intrinsic_op op,
                     unsigned *comp_out)
{
   nir_def *src = asrc->src.ssa;
   if (!src || src->bit_size != 32)
      return false;
   if (nir_def_instr(src)->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(nir_def_instr(src))->intrinsic == op) {
      *comp_out = asrc->swizzle[0];
      return true;
   }
   return match_sysval_comp(src, op, comp_out);
}

static bool
match_scaled_sysval_comp(const nir_alu_src *asrc, nir_intrinsic_op op,
                         const uint16_t wg_size[3], unsigned comp)
{
   if (comp >= 3 || wg_size[comp] == 0)
      return false;

   unsigned sc = 0;
   if (wg_size[comp] == 1 && match_sysval_alu_src(asrc, op, &sc) && sc == comp)
      return true;

   nir_def *bm = strip_mov(asrc->src.ssa);
   if (!bm || nir_def_instr(bm)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(bm));
   if (mul->op == nir_op_imul) {
      for (unsigned ms = 0; ms < 2; ms++) {
         if (match_sysval_alu_src(&mul->src[ms], op, &sc) && sc == comp &&
             nir_src_is_const(mul->src[1 - ms].src) &&
             nir_src_comp_as_uint(mul->src[1 - ms].src,
                                  mul->src[1 - ms].swizzle[0]) ==
                wg_size[comp])
            return true;
      }
   } else if (mul->op == nir_op_ishl) {
      if (match_sysval_alu_src(&mul->src[0], op, &sc) && sc == comp &&
          nir_src_is_const(mul->src[1].src) &&
          (1u << nir_src_comp_as_uint(mul->src[1].src,
                                      mul->src[1].swizzle[0])) ==
             wg_size[comp])
         return true;
   } else if (mul->op == nir_op_iadd && wg_size[comp] == 2) {
      unsigned sc0 = 0, sc1 = 0;
      if (match_sysval_alu_src(&mul->src[0], op, &sc0) && sc0 == comp &&
          match_sysval_alu_src(&mul->src[1], op, &sc1) && sc1 == comp)
         return true;
   }
   return false;
}

static bool
match_global_id_comp(nir_def *def, const uint16_t wg_size[3],
                     unsigned *comp_out)
{
   if (match_sysval_comp(def, nir_intrinsic_load_global_invocation_id,
                         comp_out))
      return true;

   unsigned wg_c = 0;
   if (match_sysval_comp(def, nir_intrinsic_load_workgroup_id, &wg_c) &&
       wg_c < 3 && wg_size[wg_c] == 1) {
      *comp_out = wg_c;
      return true;
   }

   if (def && def->num_components == 1 &&
       nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *m = nir_instr_as_alu(nir_def_instr(def));
      if (m->op == nir_op_mov && m->src[0].src.ssa->num_components > 1 &&
          nir_def_instr(m->src[0].src.ssa)->type == nir_instr_type_alu) {
         nir_alu_instr *vadd =
            nir_instr_as_alu(nir_def_instr(m->src[0].src.ssa));
         if (vadd->op == nir_op_iadd) {
            unsigned c = m->src[0].swizzle[0];
            nir_def *s0 = vadd->src[0].src.ssa;
            nir_def *s1 = vadd->src[1].src.ssa;
            if (nir_def_instr(s0)->type == nir_instr_type_intrinsic &&
                nir_def_instr(s1)->type == nir_instr_type_intrinsic) {
               nir_intrinsic_op op0 =
                  nir_instr_as_intrinsic(nir_def_instr(s0))->intrinsic;
               nir_intrinsic_op op1 =
                  nir_instr_as_intrinsic(nir_def_instr(s1))->intrinsic;
               if ((op0 == nir_intrinsic_load_global_invocation_id &&
                    op1 == nir_intrinsic_load_base_global_invocation_id) ||
                   (op1 == nir_intrinsic_load_global_invocation_id &&
                    op0 == nir_intrinsic_load_base_global_invocation_id)) {
                  *comp_out = c;
                  return true;
               }
               if (c < 3 && wg_size[c] == 1 &&
                   ((op0 == nir_intrinsic_load_workgroup_id &&
                     op1 == nir_intrinsic_load_base_workgroup_id) ||
                    (op1 == nir_intrinsic_load_workgroup_id &&
                     op0 == nir_intrinsic_load_base_workgroup_id))) {
                  *comp_out = c;
                  return true;
               }
            }
         }
      }
   }

   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(def));
   if (add->op != nir_op_iadd)
      return false;

   for (unsigned side = 0; side < 2; side++) {
      unsigned gc = 0, bc = 0;
      if (match_sysval_alu_src(&add->src[side],
                               nir_intrinsic_load_global_invocation_id, &gc) &&
          gc < 3) {
         if ((match_sysval_alu_src(&add->src[1 - side],
                                   nir_intrinsic_load_base_global_invocation_id,
                                   &bc) &&
              bc == gc) ||
             match_scaled_sysval_comp(&add->src[1 - side],
                                      nir_intrinsic_load_base_workgroup_id,
                                      wg_size, gc)) {
            *comp_out = gc;
            return true;
         }
      }
      if (match_sysval_alu_src(&add->src[side],
                               nir_intrinsic_load_workgroup_id, &gc) &&
          gc < 3 && wg_size[gc] == 1) {
         if ((match_sysval_alu_src(&add->src[1 - side],
                                   nir_intrinsic_load_base_workgroup_id,
                                   &bc) &&
              bc == gc) ||
             (match_sysval_alu_src(&add->src[1 - side],
                                   nir_intrinsic_load_base_global_invocation_id,
                                   &bc) &&
              bc == gc)) {
            *comp_out = gc;
            return true;
         }
      }
   }

   for (unsigned side = 0; side < 2; side++) {
      unsigned loc_c = 0;
      if (!match_sysval_alu_src(&add->src[side],
                                nir_intrinsic_load_local_invocation_id, &loc_c))
         continue;
      if (loc_c >= 3)
         continue;
      if (match_scaled_sysval_comp(&add->src[1 - side],
                                   nir_intrinsic_load_workgroup_id, wg_size,
                                   loc_c)) {
         *comp_out = loc_c;
         return true;
      }
   }

   return false;
}

static bool
same_val(nir_def *a, nir_def *b, const uint16_t wg_size[3])
{
   a = strip_mov(a);
   b = strip_mov(b);
   if (a == b)
      return true;
   uint16_t oa = 0, ob = 0;
   if (match_push_const_u32(a, &oa) && match_push_const_u32(b, &ob))
      return oa == ob;
   unsigned ca = 0, cb = 0;
   if (match_global_id_comp(a, wg_size, &ca) &&
       match_global_id_comp(b, wg_size, &cb))
      return ca == cb;
   return false;
}

static bool
match_pc_minus_1(nir_def *def, nir_def *pc, const uint16_t wg_size[3])
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_isub) {
      return same_val(alu->src[0].src.ssa, pc, wg_size) &&
             is_const_i32(alu->src[1].src.ssa, 1);
   }
   if (alu->op == nir_op_iadd) {
      for (unsigned s = 0; s < 2; s++) {
         if (same_val(alu->src[s].src.ssa, pc, wg_size) &&
             is_const_i32(alu->src[1 - s].src.ssa, -1))
            return true;
      }
   }
   return false;
}

/* Match `row * stride + col` in either add/mul operand order. */
static bool
match_linear_index(nir_def *idx, nir_def *row, nir_def *col,
                   uint16_t *stride_pc_out, const uint16_t wg_size[3])
{
   idx = strip_mov(idx);
   if (!idx || nir_def_instr(idx)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(idx));
   if (add->op != nir_op_iadd)
      return false;

   for (unsigned side = 0; side < 2; side++) {
      if (!same_val(add->src[side].src.ssa, col, wg_size))
         continue;
      nir_def *prod = strip_mov(add->src[1 - side].src.ssa);
      if (!prod || nir_def_instr(prod)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(prod));
      if (mul->op != nir_op_imul && mul->op != nir_op_amul)
         continue;
      for (unsigned ms = 0; ms < 2; ms++) {
         uint16_t spc = 0;
         if (same_val(mul->src[ms].src.ssa, row, wg_size) &&
             match_push_const_u32(mul->src[1 - ms].src.ssa, &spc)) {
            *stride_pc_out = spc;
            return true;
         }
      }
   }
   return false;
}

struct bool_term {
   nir_def *def;
   bool negated;
};

static bool
collect_and_terms(nir_def *cond, bool negated, struct bool_term *out,
                  unsigned *count, unsigned max_terms)
{
   cond = strip_mov(cond);
   if (!cond || cond->bit_size != 1 || cond->num_components != 1)
      return false;

   if (nir_def_instr(cond)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
      if (alu->op == nir_op_inot)
         return collect_and_terms(alu->src[0].src.ssa, !negated, out, count,
                                  max_terms);

      if (!negated && alu->op == nir_op_iand) {
         return collect_and_terms(alu->src[0].src.ssa, false, out, count,
                                  max_terms) &&
                collect_and_terms(alu->src[1].src.ssa, false, out, count,
                                  max_terms);
      }
      if (negated && alu->op == nir_op_ior) {
         return collect_and_terms(alu->src[0].src.ssa, true, out, count,
                                  max_terms) &&
                collect_and_terms(alu->src[1].src.ssa, true, out, count,
                                  max_terms);
      }
      if (alu->op == nir_op_bcsel) {
         nir_def *c = alu->src[0].src.ssa;
         nir_def *t = alu->src[1].src.ssa;
         nir_def *e = alu->src[2].src.ssa;
         if (!negated) {
            if (is_const_bool(e, false))
               return collect_and_terms(c, false, out, count, max_terms) &&
                      collect_and_terms(t, false, out, count, max_terms);
            if (is_const_bool(t, false))
               return collect_and_terms(c, true, out, count, max_terms) &&
                      collect_and_terms(e, false, out, count, max_terms);
         } else {
            if (is_const_bool(t, true))
               return collect_and_terms(c, true, out, count, max_terms) &&
                      collect_and_terms(e, true, out, count, max_terms);
            if (is_const_bool(e, true))
               return collect_and_terms(c, false, out, count, max_terms) &&
                      collect_and_terms(t, true, out, count, max_terms);
         }
      }
   }

   if (*count >= max_terms)
      return false;
   out[(*count)++] = (struct bool_term){.def = cond, .negated = negated};
   return true;
}

static bool
decompose_cmp_term(struct bool_term t, nir_def **lhs_out, nir_def **rhs_out,
                   bool *is_lt_out, bool *is_unsigned_out)
{
   if (nir_def_instr(t.def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(t.def));
   nir_def *a = alu->src[0].src.ssa;
   nir_def *b = alu->src[1].src.ssa;

   switch (alu->op) {
   case nir_op_ilt:
      *lhs_out = a;
      *rhs_out = b;
      *is_lt_out = !t.negated;
      *is_unsigned_out = false;
      return true;
   case nir_op_ige:
      *lhs_out = a;
      *rhs_out = b;
      *is_lt_out = t.negated;
      *is_unsigned_out = false;
      return true;
   case nir_op_ult:
      *lhs_out = a;
      *rhs_out = b;
      *is_lt_out = !t.negated;
      *is_unsigned_out = true;
      return true;
   case nir_op_uge:
      *lhs_out = a;
      *rhs_out = b;
      *is_lt_out = t.negated;
      *is_unsigned_out = true;
      return true;
   default:
      return false;
   }
}

/* Match `x < width && y < height` (signed or unsigned). */
static bool
match_xy_bounds_cond(nir_def *cond, bool negated, const uint16_t wg_size[3],
                     nir_def **x_out, nir_def **y_out,
                     nir_def **w_pc_out, nir_def **h_pc_out)
{
   struct bool_term terms[4];
   unsigned n = 0;
   if (!collect_and_terms(cond, negated, terms, &n, 4) || n != 2)
      return false;

   nir_def *x = NULL, *y = NULL, *w_pc = NULL, *h_pc = NULL;
   for (unsigned i = 0; i < 2; i++) {
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(terms[i], &lhs, &rhs, &is_lt, &is_u) || !is_lt)
         return false;
      unsigned comp = 0;
      uint16_t pc = 0;
      if (!match_global_id_comp(lhs, wg_size, &comp) || comp > 1 ||
          !match_push_const_u32(rhs, &pc))
         return false;
      if (comp == 0) {
         if (x)
            return false;
         x = lhs;
         w_pc = rhs;
      } else {
         if (y)
            return false;
         y = lhs;
         h_pc = rhs;
      }
   }
   if (!x || !y)
      return false;
   *x_out = x;
   *y_out = y;
   *w_pc_out = w_pc;
   *h_pc_out = h_pc;
   return true;
}

static bool
cf_list_is_empty_block(struct exec_list *cf_list)
{
   if (!exec_list_is_singular(cf_list))
      return false;
   nir_cf_node *n =
      exec_node_data(nir_cf_node, exec_list_get_head(cf_list), node);
   return n->type == nir_cf_node_block &&
          exec_list_is_empty(&nir_cf_node_as_block(n)->instr_list);
}

static bool
cf_list_is_just_return(struct exec_list *cf_list)
{
   if (!exec_list_is_singular(cf_list))
      return false;
   nir_cf_node *n =
      exec_node_data(nir_cf_node, exec_list_get_head(cf_list), node);
   if (n->type != nir_cf_node_block)
      return false;
   nir_block *b = nir_cf_node_as_block(n);
   if (!exec_list_is_singular(&b->instr_list))
      return false;
   nir_instr *instr = nir_block_first_instr(b);
   return instr->type == nir_instr_type_jump &&
          nir_instr_as_jump(instr)->type == nir_jump_return;
}

/* Extract the top-level 2D bounds guard `if (x < W && y < H) { body }` or
 * `if (x >= W || y >= H) return; body`. */
static bool
match_top_level_xy_guard(nir_function_impl *impl, const uint16_t wg_size[3],
                         nir_if **outer_if_out,
                         struct exec_list **body_list_out,
                         bool *early_return_out,
                         nir_def **x_out, nir_def **y_out,
                         nir_def **w_pc_out, nir_def **h_pc_out)
{
   nir_if *first_if = NULL;
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type == nir_cf_node_if) {
         first_if = nir_cf_node_as_if(node);
         break;
      }
      if (node->type != nir_cf_node_block)
         return false;
   }
   if (!first_if)
      return false;

   /* Case A: `if (x < W && y < H) { body }` (with empty else and empty tail). */
   if (cf_list_is_empty_block(&first_if->else_list) &&
       match_xy_bounds_cond(first_if->condition.ssa, false, wg_size, x_out,
                            y_out, w_pc_out, h_pc_out)) {
      nir_cf_node *after = nir_cf_node_next(&first_if->cf_node);
      if (after && after->type == nir_cf_node_block &&
          !nir_cf_node_next(after) &&
          exec_list_is_empty(&nir_cf_node_as_block(after)->instr_list)) {
         *outer_if_out = first_if;
         *body_list_out = &first_if->then_list;
         *early_return_out = false;
         return true;
      }
   }

   /* Case B: `if (!(x < W && y < H)) { } else { body }` */
   if (cf_list_is_empty_block(&first_if->then_list) &&
       match_xy_bounds_cond(first_if->condition.ssa, true, wg_size, x_out,
                            y_out, w_pc_out, h_pc_out)) {
      nir_cf_node *after = nir_cf_node_next(&first_if->cf_node);
      if (after && after->type == nir_cf_node_block &&
          !nir_cf_node_next(after) &&
          exec_list_is_empty(&nir_cf_node_as_block(after)->instr_list)) {
         *outer_if_out = first_if;
         *body_list_out = &first_if->else_list;
         *early_return_out = false;
         return true;
      }
   }

   /* Case C: `if (x >= W || y >= H) return; body` */
   if (cf_list_is_just_return(&first_if->then_list) &&
       cf_list_is_empty_block(&first_if->else_list) &&
       match_xy_bounds_cond(first_if->condition.ssa, true, wg_size, x_out,
                            y_out, w_pc_out, h_pc_out)) {
      *outer_if_out = first_if;
      *body_list_out = &impl->body;
      *early_return_out = true;
      return true;
   }

   return false;
}

/* Match `x >= 1 && x < W - 1 && y >= 1 && y < H - 1` (in any order, accepting
 * `0 < x`, `x >= 1`, `x != 0`, `x < W - 1`). */
static bool
match_interior_1px_cond(nir_def *cond, bool negated, nir_def *x, nir_def *y,
                        nir_def *w_pc, nir_def *h_pc,
                        const uint16_t wg_size[3])
{
   struct bool_term terms[6];
   unsigned n = 0;
   if (!collect_and_terms(cond, negated, terms, &n, 6) || n != 4)
      return false;

   bool x_lo = false, x_hi = false, y_lo = false, y_hi = false;
   for (unsigned i = 0; i < 4; i++) {
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (decompose_cmp_term(terms[i], &lhs, &rhs, &is_lt, &is_u)) {
         if (is_lt && is_const_i32(lhs, 0) && same_val(rhs, x, wg_size)) {
            x_lo = true;
         } else if (!is_lt && same_val(lhs, x, wg_size) &&
                    is_const_i32(rhs, 1)) {
            x_lo = true;
         } else if (is_lt && is_const_i32(lhs, 0) &&
                    same_val(rhs, y, wg_size)) {
            y_lo = true;
         } else if (!is_lt && same_val(lhs, y, wg_size) &&
                    is_const_i32(rhs, 1)) {
            y_lo = true;
         } else if (is_lt && same_val(lhs, x, wg_size) &&
                    match_pc_minus_1(rhs, w_pc, wg_size)) {
            x_hi = true;
         } else if (is_lt && same_val(lhs, y, wg_size) &&
                    match_pc_minus_1(rhs, h_pc, wg_size)) {
            y_hi = true;
         } else {
            return false;
         }
      } else {
         return false;
      }
   }
   return x_lo && x_hi && y_lo && y_hi;
}

static bool
match_ssbo_array_deref(nir_deref_instr *deref, unsigned expected_stride,
                       uint32_t *set_out, uint32_t *binding_out,
                       nir_def **index_out)
{
   if (!deref || !nir_deref_mode_is(deref, nir_var_mem_ssbo))
      return false;

   nir_def *elem_index = NULL;
   unsigned elem_stride = 0;

   while (deref) {
      if (deref->deref_type == nir_deref_type_array ||
          deref->deref_type == nir_deref_type_ptr_as_array) {
         if (elem_index != NULL)
            return false;
         elem_index = deref->arr.index.ssa;
         elem_stride = nir_deref_instr_array_stride(deref);
         deref = nir_deref_instr_parent(deref);
      } else if (deref->deref_type == nir_deref_type_struct) {
         nir_deref_instr *parent = nir_deref_instr_parent(deref);
         if (!parent)
            return false;
         if (glsl_get_struct_field_offset(parent->type, deref->strct.index) !=
             0)
            return false;
         deref = parent;
      } else if (deref->deref_type == nir_deref_type_cast) {
         if (nir_def_instr(deref->parent.ssa)->type == nir_instr_type_deref) {
            deref = nir_src_as_deref(deref->parent);
            continue;
         }
         if (nir_def_instr(deref->parent.ssa)->type != nir_instr_type_intrinsic)
            return false;
         nir_intrinsic_instr *lvd =
            nir_instr_as_intrinsic(nir_def_instr(deref->parent.ssa));
         if (lvd->intrinsic != nir_intrinsic_load_vulkan_descriptor)
            return false;
         if (nir_def_instr(lvd->src[0].ssa)->type != nir_instr_type_intrinsic)
            return false;
         nir_intrinsic_instr *vri =
            nir_instr_as_intrinsic(nir_def_instr(lvd->src[0].ssa));
         if (vri->intrinsic != nir_intrinsic_vulkan_resource_index)
            return false;
         if (nir_intrinsic_desc_type(vri) != nir_descriptor_type_storage_buffer)
            return false;
         if (!nir_src_is_const(vri->src[0]) ||
             nir_src_as_uint(vri->src[0]) != 0)
            return false;

         if (!elem_index || elem_stride != expected_stride)
            return false;
         if (elem_index->bit_size != 32 || elem_index->num_components != 1)
            return false;

         *set_out = nir_intrinsic_desc_set(vri);
         *binding_out = nir_intrinsic_binding(vri);
         *index_out = elem_index;
         return true;
      } else {
         return false;
      }
   }

   return false;
}

static bool
populate_binding_info(uint32_t set_layout_count,
                      struct vk_descriptor_set_layout *const *set_layouts,
                      uint32_t set, uint32_t binding,
                      struct hk_compute_opt_binding *out)
{
   if (set >= HK_MAX_SETS || set >= set_layout_count || !set_layouts ||
       !set_layouts[set])
      return false;

   const struct hk_descriptor_set_layout *layout =
      vk_to_hk_descriptor_set_layout(set_layouts[set]);
   if (!layout || binding >= layout->binding_count)
      return false;

   const struct hk_descriptor_set_binding_layout *bl =
      &layout->binding[binding];
   if (bl->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || bl->array_size != 1)
      return false;
   if (bl->flags &
       (VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT))
      return false;

   out->set = set;
   out->binding = binding;
   out->descriptor_offset_B = bl->offset;
   memcpy(out->set_layout_hash, layout->vk.blake3, sizeof(out->set_layout_hash));
   return true;
}

static nir_def *
strip_u2u32(nir_def *def)
{
   def = strip_mov(def);
   if (def && nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
      if (alu->op == nir_op_u2u32)
         return strip_mov(alu->src[0].src.ssa);
   }
   return def;
}

static bool
is_allowed_base_intrinsic(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_load_push_constant:
   case nir_intrinsic_vulkan_resource_index:
   case nir_intrinsic_load_vulkan_descriptor:
   case nir_intrinsic_load_global_invocation_id:
   case nir_intrinsic_load_base_global_invocation_id:
   case nir_intrinsic_load_base_workgroup_id:
   case nir_intrinsic_load_local_invocation_id:
   case nir_intrinsic_load_workgroup_id:
   case nir_intrinsic_load_num_workgroups:
   case nir_intrinsic_load_workgroup_size:
      return true;
   default:
      return false;
   }
}

/* ---------------------------------------------------------------------------
 * Matcher 1: RGBA8 -> Grayscale u8 (`(r*w_r + g*w_g + b*w_b) >> shr`)
 * --------------------------------------------------------------------------- */

/* Match byte channel `chan` (0, 1, 2) extracted from 32-bit `px` (or u8x4 `px`)
 * as u32. */
static bool
match_u32_byte_channel(nir_def *def, nir_def *px, unsigned *chan_out)
{
   def = strip_mov(def);
   if (!def || def->bit_size != 32 || def->num_components != 1)
      return false;
   if (nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));

   /* u8x4 load: `u2u32(mov(px).c)` */
   if (px->bit_size == 8 && px->num_components == 4) {
      if (alu->op != nir_op_u2u32 && alu->op != nir_op_i2i32)
         return false;
      nir_def *s = alu->src[0].src.ssa;
      if (s && nir_def_instr(s)->type == nir_instr_type_alu) {
         nir_alu_instr *mv = nir_instr_as_alu(nir_def_instr(s));
         if (mv->op == nir_op_mov && mv->src[0].src.ssa == px &&
             mv->src[0].swizzle[0] < 3) {
            *chan_out = mv->src[0].swizzle[0];
            return true;
         }
      }
      return false;
   }

   /* u2u32(u2u8(px >> shift)) or u2u32(unpack_32_2x16_split_y(px) & 0xff) */
   if (alu->op == nir_op_u2u32) {
      nir_def *s = strip_mov(alu->src[0].src.ssa);
      if (s && nir_def_instr(s)->type == nir_instr_type_alu) {
         nir_alu_instr *sa = nir_instr_as_alu(nir_def_instr(s));
         if (sa->op == nir_op_iand && s->bit_size == 16) {
            for (unsigned side = 0; side < 2; side++) {
               if (!nir_src_is_const(sa->src[side].src) ||
                   nir_src_as_uint(sa->src[side].src) != 0xffu)
                  continue;
               nir_def *hi = strip_mov(sa->src[1 - side].src.ssa);
               if (hi && nir_def_instr(hi)->type == nir_instr_type_alu) {
                  nir_alu_instr *ha = nir_instr_as_alu(nir_def_instr(hi));
                  if (ha->op == nir_op_unpack_32_2x16_split_y &&
                      strip_mov(ha->src[0].src.ssa) == strip_mov(px)) {
                     *chan_out = 2;
                     return true;
                  }
                  if (ha->op == nir_op_unpack_32_2x16_split_x &&
                      strip_mov(ha->src[0].src.ssa) == strip_mov(px)) {
                     *chan_out = 0;
                     return true;
                  }
               }
            }
         }
         if (sa->op == nir_op_u2u8) {
            nir_def *in = strip_mov(sa->src[0].src.ssa);
            if (in == strip_mov(px)) {
               *chan_out = 0;
               return true;
            }
            if (in && nir_def_instr(in)->type == nir_instr_type_alu) {
               nir_alu_instr *sh = nir_instr_as_alu(nir_def_instr(in));
               uint32_t k = 0;
               if ((sh->op == nir_op_ushr || sh->op == nir_op_ishr) &&
                   strip_mov(sh->src[0].src.ssa) == strip_mov(px) &&
                   match_const_u32(sh->src[1].src.ssa, &k) &&
                   (k == 8 || k == 16)) {
                  *chan_out = k / 8;
                  return true;
               }
            }
         }
      }
      return false;
   }

   /* extract_u8(px, c) */
   if (alu->op == nir_op_extract_u8) {
      uint32_t c = 0;
      if (strip_mov(alu->src[0].src.ssa) == strip_mov(px) &&
          match_const_u32(alu->src[1].src.ssa, &c) && c < 3) {
         *chan_out = c;
         return true;
      }
      return false;
   }

   /* ubitfield_extract(px, 8*c, 8) */
   if (alu->op == nir_op_ubitfield_extract) {
      uint32_t off = 0, bits = 0;
      if (strip_mov(alu->src[0].src.ssa) == strip_mov(px) &&
          match_const_u32(alu->src[1].src.ssa, &off) &&
          match_const_u32(alu->src[2].src.ssa, &bits) &&
          bits == 8 && (off == 0 || off == 8 || off == 16)) {
         *chan_out = off / 8;
         return true;
      }
      return false;
   }

   /* (px >> (8*c)) & 0xff or (px & (0xff << 8*c)) >> (8*c) */
   if (alu->op == nir_op_iand) {
      for (unsigned side = 0; side < 2; side++) {
         if (!is_const_i32(alu->src[side].src.ssa, 0xff))
            continue;
         nir_def *other = strip_mov(alu->src[1 - side].src.ssa);
         if (other == strip_mov(px)) {
            *chan_out = 0;
            return true;
         }
         if (other && nir_def_instr(other)->type == nir_instr_type_alu) {
            nir_alu_instr *sh = nir_instr_as_alu(nir_def_instr(other));
            uint32_t k = 0;
            if ((sh->op == nir_op_ushr || sh->op == nir_op_ishr) &&
                strip_mov(sh->src[0].src.ssa) == strip_mov(px) &&
                match_const_u32(sh->src[1].src.ssa, &k) &&
                (k == 8 || k == 16)) {
               *chan_out = k / 8;
               return true;
            }
            if (sh->op == nir_op_extract_u16 &&
                strip_mov(sh->src[0].src.ssa) == strip_mov(px) &&
                match_const_u32(sh->src[1].src.ssa, &k) && k <= 1) {
               *chan_out = k * 2;
               return true;
            }
         }
      }
   }

   if (alu->op == nir_op_ushr || alu->op == nir_op_ishr) {
      uint32_t k = 0;
      if (match_const_u32(alu->src[1].src.ssa, &k) && (k == 8 || k == 16)) {
         nir_def *in = strip_mov(alu->src[0].src.ssa);
         if (in && nir_def_instr(in)->type == nir_instr_type_alu) {
            nir_alu_instr *an = nir_instr_as_alu(nir_def_instr(in));
            if (an->op == nir_op_iand) {
               uint32_t mask = 0xffu << k;
               for (unsigned side = 0; side < 2; side++) {
                  uint32_t c = 0;
                  if (match_const_u32(an->src[side].src.ssa, &c) && c == mask &&
                      strip_mov(an->src[1 - side].src.ssa) == strip_mov(px)) {
                     *chan_out = k / 8;
                     return true;
                  }
               }
            }
         }
      }
   }

   return false;
}

static bool
collect_luma_terms(nir_def *def, nir_def *px, uint32_t weights[3],
                   bool seen[3], unsigned depth)
{
   if (depth > 6)
      return false;
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));

   if (alu->op == nir_op_iadd) {
      return collect_luma_terms(alu->src[0].src.ssa, px, weights, seen,
                                depth + 1) &&
             collect_luma_terms(alu->src[1].src.ssa, px, weights, seen,
                                depth + 1);
   }

   if (alu->op == nir_op_imul || alu->op == nir_op_amul) {
      for (unsigned side = 0; side < 2; side++) {
         uint32_t w = 0;
         unsigned chan = 0;
         if (match_const_u32(alu->src[side].src.ssa, &w) && w > 0 &&
             match_u32_byte_channel(alu->src[1 - side].src.ssa, px, &chan) &&
             chan < 3 && !seen[chan]) {
            seen[chan] = true;
            weights[chan] = w;
            return true;
         }
      }
   }
   return false;
}

static bool
match_rgba8_to_gray(nir_shader *s, uint32_t set_layout_count,
                    struct vk_descriptor_set_layout *const *set_layouts,
                    struct hk_compute_opt_plan *plan)
{
   if (s->info.num_ssbos != 2)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_if *outer_if = NULL;
   struct exec_list *body_list = NULL;
   bool early_ret = false;
   nir_def *x = NULL, *y = NULL, *w_pc = NULL, *h_pc = NULL;
   if (!match_top_level_xy_guard(impl, s->info.workgroup_size, &outer_if,
                                 &body_list, &early_ret, &x, &y, &w_pc, &h_pc))
      return false;

   /* Single load (u32), single store (u8), no other CF or side effects. */
   nir_intrinsic_instr *load_intr = NULL, *store_intr = NULL;
   nir_foreach_block(blk, impl) {
      nir_foreach_instr(instr, blk) {
         if (instr->type == nir_instr_type_tex ||
             instr->type == nir_instr_type_call)
            return false;
         if (instr->type == nir_instr_type_jump) {
            nir_jump_instr *j = nir_instr_as_jump(instr);
            if (!(early_ret && j->type == nir_jump_return &&
                  blk == nir_if_first_then_block(outer_if)))
               return false;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_deref) {
            if (load_intr)
               return false;
            load_intr = intr;
         } else if (intr->intrinsic == nir_intrinsic_store_deref) {
            if (store_intr)
               return false;
            store_intr = intr;
         } else if (!is_allowed_base_intrinsic(intr->intrinsic)) {
            return false;
         }
      }
   }
   if (!load_intr || !store_intr)
      return false;

   /* Verify no extra nested `if` or `loop`. */
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type == nir_cf_node_if) {
         if (nir_cf_node_as_if(node) != outer_if)
            return false;
         if (!early_ret) {
            foreach_list_typed(nir_cf_node, inner, node, body_list) {
               if (inner->type != nir_cf_node_block)
                  return false;
            }
         }
      } else if (node->type != nir_cf_node_block) {
         return false;
      }
   }

   /* Must load u32 (or u8x4) from src[y * src_stride + x] and store u8 to dst[y * dst_stride + x]. */
   const bool load_is_u32 =
      load_intr->def.bit_size == 32 && load_intr->def.num_components == 1;
   const bool load_is_u8x4 =
      load_intr->def.bit_size == 8 && load_intr->def.num_components == 4;
   if (!load_is_u32 && !load_is_u8x4)
      return false;
   nir_def *st_val = strip_mov(store_intr->src[1].ssa);
   if (!st_val || st_val->bit_size != 8 || st_val->num_components != 1)
      return false;

   uint32_t src_set = 0, src_bind = 0, dst_set = 0, dst_bind = 0;
   nir_def *src_idx = NULL, *dst_idx = NULL;
   if (!match_ssbo_array_deref(nir_src_as_deref(load_intr->src[0]), 4,
                               &src_set, &src_bind, &src_idx) ||
       !match_ssbo_array_deref(nir_src_as_deref(store_intr->src[0]), 1,
                               &dst_set, &dst_bind, &dst_idx))
      return false;

   uint16_t src_stride_pc = 0, dst_stride_pc = 0;
   if (!match_linear_index(src_idx, y, x, &src_stride_pc,
                           s->info.workgroup_size) ||
       !match_linear_index(dst_idx, y, x, &dst_stride_pc,
                           s->info.workgroup_size))
      return false;

   /* Match st_val = u2u8(ushr/ishr/ubitfield_extract(r*w_r + g*w_g + b*w_b, shr)). */
   if (nir_def_instr(st_val)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *trunc = nir_instr_as_alu(nir_def_instr(st_val));
   if (trunc->op != nir_op_u2u8 && trunc->op != nir_op_i2i8)
      return false;
   nir_def *shifted = strip_mov(trunc->src[0].src.ssa);
   if (!shifted || nir_def_instr(shifted)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *sh = nir_instr_as_alu(nir_def_instr(shifted));
   uint32_t shr = 0;
   if (sh->op == nir_op_ushr || sh->op == nir_op_ishr) {
      if (!match_const_u32(sh->src[1].src.ssa, &shr) || shr < 1 || shr > 24)
         return false;
   } else if (sh->op == nir_op_ubitfield_extract) {
      uint32_t bits = 0;
      if (!match_const_u32(sh->src[1].src.ssa, &shr) || shr < 1 || shr > 24 ||
          !match_const_u32(sh->src[2].src.ssa, &bits) || bits != 8)
         return false;
   } else {
      return false;
   }

   uint32_t weights[3] = {0};
   bool seen[3] = {false};
   if (!collect_luma_terms(sh->src[0].src.ssa, &load_intr->def, weights, seen,
                           0))
      return false;
   if (!seen[0] || !seen[1] || !seen[2])
      return false;
   uint64_t max_sum =
      255ull * ((uint64_t)weights[0] + weights[1] + weights[2]);
   if (max_sum > (uint64_t)INT32_MAX)
      return false;

   uint16_t w_off = 0, h_off = 0;
   if (!match_push_const_u32(w_pc, &w_off) ||
       !match_push_const_u32(h_pc, &h_off))
      return false;
   if (!populate_binding_info(set_layout_count, set_layouts, src_set, src_bind,
                              &plan->bufs[0]) ||
       !populate_binding_info(set_layout_count, set_layouts, dst_set, dst_bind,
                              &plan->bufs[1]))
      return false;

   plan->kind = HK_COMPUTE_OPT_RGBA8_TO_GRAY_U8;
   plan->tile_w = 256;
   plan->tile_h = 4;
   plan->pc_off[0] = w_off;
   plan->pc_off[1] = h_off;
   plan->pc_off[2] = src_stride_pc;
   plan->pc_off[3] = dst_stride_pc;
   plan->params[0] = weights[0];
   plan->params[1] = weights[1];
   plan->params[2] = weights[2];
   plan->params[3] = shr;
   return true;
}

/* ---------------------------------------------------------------------------
 * Matcher 2: Canny Hysteresis 3x3 8-neighbor u8 propagation
 * --------------------------------------------------------------------------- */

struct hyst_lin_idx {
   int c_x;
   int c_yw;
   int c_w;
   int c_const;
};

static bool
decompose_hyst_idx(nir_def *def, nir_def *x, nir_def *y, nir_def *w_pc,
                   const uint16_t wg_size[3], struct hyst_lin_idx *out,
                   int sign, unsigned depth)
{
   if (depth > 8)
      return false;
   def = strip_mov(def);
   if (!def)
      return false;

   if (same_val(def, x, wg_size)) {
      out->c_x += sign;
      return true;
   }
   if (same_val(def, w_pc, wg_size)) {
      out->c_w += sign;
      return true;
   }
   int32_t c = 0;
   if (match_const_i32(def, &c)) {
      out->c_const += sign * c;
      return true;
   }
   if (nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));

   if (alu->op == nir_op_iadd) {
      return decompose_hyst_idx(alu->src[0].src.ssa, x, y, w_pc, wg_size, out,
                                sign, depth + 1) &&
             decompose_hyst_idx(alu->src[1].src.ssa, x, y, w_pc, wg_size, out,
                                sign, depth + 1);
   }
   if (alu->op == nir_op_isub) {
      return decompose_hyst_idx(alu->src[0].src.ssa, x, y, w_pc, wg_size, out,
                                sign, depth + 1) &&
             decompose_hyst_idx(alu->src[1].src.ssa, x, y, w_pc, wg_size, out,
                                -sign, depth + 1);
   }
   if (alu->op == nir_op_ineg) {
      return decompose_hyst_idx(alu->src[0].src.ssa, x, y, w_pc, wg_size, out,
                                -sign, depth + 1);
   }
   if (alu->op == nir_op_imul || alu->op == nir_op_amul) {
      for (unsigned side = 0; side < 2; side++) {
         if (!same_val(alu->src[side].src.ssa, w_pc, wg_size))
            continue;
         nir_def *other = strip_mov(alu->src[1 - side].src.ssa);
         if (same_val(other, y, wg_size)) {
            out->c_yw += sign;
            return true;
         }
         /* (y +/- 1) * w */
         if (other && nir_def_instr(other)->type == nir_instr_type_alu) {
            nir_alu_instr *oa = nir_instr_as_alu(nir_def_instr(other));
            if (oa->op == nir_op_iadd) {
               for (unsigned k = 0; k < 2; k++) {
                  int32_t dy = 0;
                  if (same_val(oa->src[k].src.ssa, y, wg_size) &&
                      match_const_i32(oa->src[1 - k].src.ssa, &dy) &&
                      dy >= -1 && dy <= 1) {
                     out->c_yw += sign;
                     out->c_w += sign * dy;
                     return true;
                  }
               }
            } else if (oa->op == nir_op_isub) {
               int32_t dy = 0;
               if (same_val(oa->src[0].src.ssa, y, wg_size) &&
                   match_const_i32(oa->src[1].src.ssa, &dy) &&
                   dy >= -1 && dy <= 1) {
                  out->c_yw += sign;
                  out->c_w -= sign * dy;
                  return true;
               }
            }
         }
      }
   }
   return false;
}

static int
neighbor_bit_from_dx_dy(int dx, int dy)
{
   if (dx < -1 || dx > 1 || dy < -1 || dy > 1 || (dx == 0 && dy == 0))
      return -1;
   int idx = (dy + 1) * 3 + (dx + 1);
   if (idx > 4)
      idx--;
   return idx; /* 0..7 */
}

/* Match `u2u32(load) >= 2` (or `load >= 2`, `1 < load`, `load > 1`). */
static bool
match_ge_2(nir_def *cond, nir_def **val_out)
{
   cond = strip_mov(cond);
   if (!cond || cond->bit_size != 1 || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   nir_def *a = strip_mov(alu->src[0].src.ssa);
   nir_def *b = strip_mov(alu->src[1].src.ssa);
   if ((alu->op == nir_op_ige || alu->op == nir_op_uge) &&
       is_const_u_any(b, 2)) {
      *val_out = a;
      return true;
   }
   if ((alu->op == nir_op_ilt || alu->op == nir_op_ult) &&
       is_const_u_any(a, 1)) {
      *val_out = b;
      return true;
   }
   return false;
}

/* Collect leaves of an `imax`/`umax` tree of u2u32(load_u8) values. */
static bool
collect_max_loads(nir_def *def, nir_intrinsic_instr **loads, unsigned *count,
                  unsigned max_loads)
{
   def = strip_mov(def);
   if (!def)
      return false;
   if (nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
      if (alu->op == nir_op_imax || alu->op == nir_op_umax) {
         return collect_max_loads(alu->src[0].src.ssa, loads, count,
                                  max_loads) &&
                collect_max_loads(alu->src[1].src.ssa, loads, count, max_loads);
      }
   }
   nir_def *u = strip_u2u32(def);
   if (!u || nir_def_instr(u)->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(u));
   if (intr->intrinsic != nir_intrinsic_load_deref || *count >= max_loads)
      return false;
   loads[(*count)++] = intr;
   return true;
}

/* Collect leaves of an OR / bcsel(..., true, ...) chain of `load >= 2` terms. */
static bool
collect_ge2_or_loads(nir_def *cond, nir_intrinsic_instr **loads,
                     unsigned *count, unsigned max_loads)
{
   cond = strip_mov(cond);
   if (!cond || cond->bit_size != 1 ||
       nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   if (alu->op == nir_op_ior) {
      return collect_ge2_or_loads(alu->src[0].src.ssa, loads, count,
                                  max_loads) &&
             collect_ge2_or_loads(alu->src[1].src.ssa, loads, count, max_loads);
   }
   if (alu->op == nir_op_bcsel) {
      if (is_const_bool(alu->src[1].src.ssa, true))
         return collect_ge2_or_loads(alu->src[0].src.ssa, loads, count,
                                     max_loads) &&
                collect_ge2_or_loads(alu->src[2].src.ssa, loads, count,
                                     max_loads);
      if (is_const_bool(alu->src[2].src.ssa, true))
         return false;
   }
   nir_def *v = NULL;
   if (!match_ge_2(cond, &v))
      return false;
   return collect_max_loads(v, loads, count, max_loads);
}

static bool
match_int_eq_const(nir_def *cond, nir_def *target, int32_t c, bool *is_ne_out)
{
   cond = strip_mov(cond);
   if (!cond || cond->bit_size != 1 ||
       nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   if (alu->op != nir_op_ieq && alu->op != nir_op_ine)
      return false;
   for (unsigned side = 0; side < 2; side++) {
      if (strip_u2u32(alu->src[side].src.ssa) == strip_u2u32(target) &&
          is_const_u_any(alu->src[1 - side].src.ssa, (uint32_t)c)) {
         *is_ne_out = (alu->op == nir_op_ine);
         return true;
      }
   }
   return false;
}

/* Verify that `cond` requires the exact bounds needed for neighbor `(dx, dy)`. */
static bool
match_neighbor_guard(nir_def *cond, int dx, int dy, nir_def *x, nir_def *y,
                     nir_def *w_pc, nir_def *h_pc, const uint16_t wg_size[3])
{
   struct bool_term terms[4];
   unsigned n = 0;
   if (!collect_and_terms(cond, false, terms, &n, 4))
      return false;

   bool need_x_lo = (dx == -1), need_x_hi = (dx == 1);
   bool need_y_lo = (dy == -1), need_y_hi = (dy == 1);
   unsigned expected =
      (unsigned)need_x_lo + need_x_hi + need_y_lo + need_y_hi;
   if (n != expected)
      return false;

   bool got_x_lo = false, got_x_hi = false, got_y_lo = false, got_y_hi = false;
   for (unsigned i = 0; i < n; i++) {
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(terms[i], &lhs, &rhs, &is_lt, &is_u))
         return false;
      if ((is_lt && is_const_i32(lhs, 0) && same_val(rhs, x, wg_size)) ||
          (!is_lt && same_val(lhs, x, wg_size) && is_const_i32(rhs, 1)))
         got_x_lo = true;
      else if ((is_lt && is_const_i32(lhs, 0) && same_val(rhs, y, wg_size)) ||
               (!is_lt && same_val(lhs, y, wg_size) && is_const_i32(rhs, 1)))
         got_y_lo = true;
      else if (is_lt && same_val(lhs, x, wg_size) &&
               match_pc_minus_1(rhs, w_pc, wg_size))
         got_x_hi = true;
      else if (is_lt && same_val(lhs, y, wg_size) &&
               match_pc_minus_1(rhs, h_pc, wg_size))
         got_y_hi = true;
      else
         return false;
   }
   return got_x_lo == need_x_lo && got_x_hi == need_x_hi &&
          got_y_lo == need_y_lo && got_y_hi == need_y_hi;
}

/* Recursively verify the border else-arm of Canny hysteresis: every load must
 * be a guarded 1-byte src neighbor load at a unique `(dx, dy)` whose enclosing
 * `if` implies the required `(dx, dy)` bounds, and every result update must be
 * `res = (nbr >= 2) ? 3 : prev` starting from `1`. */
struct hyst_border_ctx {
   nir_def *x, *y, *w_pc, *h_pc;
   const uint16_t *wg_size;
   uint32_t src_set, src_bind;
   uint8_t seen_mask;
   nir_intrinsic_instr *loads[8];
   unsigned load_count;
};

static bool
verify_hyst_border_cf(struct exec_list *cf_list, struct hyst_border_ctx *ctx,
                      nir_def *in_val, nir_def **out_val_ptr)
{
   nir_def *cur = in_val;
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      if (node->type == nir_cf_node_block) {
         nir_block *blk = nir_cf_node_as_block(node);
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_jump ||
                instr->type == nir_instr_type_tex ||
                instr->type == nir_instr_type_call)
               return false;
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_deref)
                  return false; /* loads in border must be inside a guard if */
               if (!is_allowed_base_intrinsic(intr->intrinsic))
                  return false;
            }
         }
         continue;
      }
      if (node->type != nir_cf_node_if)
         return false;

      nir_if *nif = nir_cf_node_as_if(node);
      if (!cf_list_is_empty_block(&nif->else_list))
         return false;

      /* Check if this `if` is an outer grouping guard (like `y > 0 || x > 0`
       * from NIR's common-subexpression factoring) or a leaf neighbor guard. */
      bool has_nested_if = false;
      foreach_list_typed(nir_cf_node, cn, node, &nif->then_list) {
         if (cn->type == nir_cf_node_if)
            has_nested_if = true;
         else if (cn->type != nir_cf_node_block)
            return false;
      }

      nir_def *then_out = NULL;
      if (has_nested_if) {
         if (!verify_hyst_border_cf(&nif->then_list, ctx, cur, &then_out))
            return false;
      } else {
         /* Leaf neighbor guard: must contain exactly 1 u8 load from src at
          * `(dx, dy)` matching `nif->condition`, and compute
          * `(nbr >= 2) ? 3 : cur`. */
         if (!exec_list_is_singular(&nif->then_list))
            return false;
         nir_block *tb = nir_if_first_then_block(nif);
         nir_intrinsic_instr *nload = NULL;
         nir_foreach_instr(instr, tb) {
            if (instr->type == nir_instr_type_jump ||
                instr->type == nir_instr_type_tex ||
                instr->type == nir_instr_type_call)
               return false;
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_deref) {
                  if (nload)
                     return false;
                  nload = intr;
               } else if (!is_allowed_base_intrinsic(intr->intrinsic)) {
                  return false;
               }
            }
         }
         if (!nload || nload->def.bit_size != 8 ||
             nload->def.num_components != 1)
            return false;
         uint32_t ls = 0, lb = 0;
         nir_def *lidx = NULL;
         if (!match_ssbo_array_deref(nir_src_as_deref(nload->src[0]), 1, &ls,
                                     &lb, &lidx) ||
             ls != ctx->src_set || lb != ctx->src_bind)
            return false;
         struct hyst_lin_idx lin = {0};
         if (!decompose_hyst_idx(lidx, ctx->x, ctx->y, ctx->w_pc, ctx->wg_size,
                                 &lin, 1, 0) ||
             lin.c_x != 1 || lin.c_yw != 1)
            return false;
         int bit = neighbor_bit_from_dx_dy(lin.c_const, lin.c_w);
         if (bit < 0 || (ctx->seen_mask & (1u << bit)))
            return false;
         if (!match_neighbor_guard(nif->condition.ssa, lin.c_const, lin.c_w,
                                   ctx->x, ctx->y, ctx->w_pc, ctx->h_pc,
                                   ctx->wg_size))
            return false;
         ctx->seen_mask |= (uint8_t)(1u << bit);
         ctx->loads[ctx->load_count++] = nload;

         /* Check the phi right after `nif` selects `(nload >= 2) ? 3 : cur`
          * on then-edge and `cur` on else-edge. */
         nir_block *merge = nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node));
         nir_phi_instr *phi = NULL;
         nir_foreach_phi(p, merge) {
            if (phi)
               return false;
            phi = p;
         }
         if (!phi)
            return false;
         nir_def *t_src = NULL, *e_src = NULL;
         nir_foreach_phi_src(ps, phi) {
            if (ps->pred == tb)
               t_src = strip_mov(ps->src.ssa);
            else if (ps->pred == nir_if_last_else_block(nif))
               e_src = strip_mov(ps->src.ssa);
         }
         if (!t_src || !e_src || e_src != cur)
            return false;
         if (nir_def_instr(t_src)->type != nir_instr_type_alu)
            return false;
         nir_alu_instr *sel = nir_instr_as_alu(nir_def_instr(t_src));
         if (sel->op != nir_op_bcsel)
            return false;
         nir_def *cmp_v = NULL;
         if (!match_ge_2(sel->src[0].src.ssa, &cmp_v) ||
             strip_u2u32(cmp_v) != &nload->def)
            return false;
         if (!is_const_i32(sel->src[1].src.ssa, 3) ||
             strip_mov(sel->src[2].src.ssa) != cur)
            return false;
         cur = &phi->def;
         continue;
      }

      /* Merge block after grouping `if`. */
      nir_block *merge = nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node));
      nir_phi_instr *phi = NULL;
      nir_foreach_phi(p, merge) {
         if (phi)
            return false;
         phi = p;
      }
      if (!phi)
         return false;
      nir_def *t_src = NULL, *e_src = NULL;
      nir_foreach_phi_src(ps, phi) {
         if (ps->pred == nir_if_last_then_block(nif))
            t_src = strip_mov(ps->src.ssa);
         else if (ps->pred == nir_if_last_else_block(nif))
            e_src = strip_mov(ps->src.ssa);
      }
      if (!t_src || !e_src || t_src != then_out || e_src != cur)
         return false;
      cur = &phi->def;
   }
   *out_val_ptr = cur;
   return true;
}

static bool
match_canny_hysteresis(nir_shader *s, uint32_t set_layout_count,
                       struct vk_descriptor_set_layout *const *set_layouts,
                       struct hk_compute_opt_plan *plan)
{
   if (s->info.num_ssbos != 2)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_if *outer_if = NULL;
   struct exec_list *body_list = NULL;
   bool early_ret = false;
   nir_def *x = NULL, *y = NULL, *w_pc = NULL, *h_pc = NULL;
   if (!match_top_level_xy_guard(impl, s->info.workgroup_size, &outer_if,
                                 &body_list, &early_ret, &x, &y, &w_pc, &h_pc))
      return false;

   /* Find the center load and single store inside body_list. */
   nir_block *entry_blk =
      early_ret ? nir_cf_node_as_block(nir_cf_node_next(&outer_if->cf_node))
                : nir_cf_node_as_block(
                     exec_node_data(nir_cf_node, exec_list_get_head(body_list),
                                    node));
   if (!entry_blk)
      return false;

   nir_intrinsic_instr *center_load = NULL;
   nir_foreach_instr(instr, entry_blk) {
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_deref) {
            if (center_load)
               return false;
            center_load = intr;
         }
      }
   }
   if (!center_load || center_load->def.bit_size != 8 ||
       center_load->def.num_components != 1)
      return false;

   uint32_t src_set = 0, src_bind = 0;
   nir_def *center_idx = NULL;
   if (!match_ssbo_array_deref(nir_src_as_deref(center_load->src[0]), 1,
                               &src_set, &src_bind, &center_idx))
      return false;
   uint16_t src_stride_pc = 0, w_off = 0, h_off = 0;
   if (!match_push_const_u32(w_pc, &w_off) ||
       !match_push_const_u32(h_pc, &h_off) ||
       !match_linear_index(center_idx, y, x, &src_stride_pc,
                           s->info.workgroup_size) ||
       src_stride_pc != w_off)
      return false;

   /* Next node after entry_blk must be `if (val != 2)` (or `if (val == 2)`). */
   nir_cf_node *n1 = nir_cf_node_next(&entry_blk->cf_node);
   if (!n1 || n1->type != nir_cf_node_if)
      return false;
   nir_if *if_eq2 = nir_cf_node_as_if(n1);
   bool is_ne2 = false;
   if (!match_int_eq_const(if_eq2->condition.ssa, &center_load->def, 2,
                           &is_ne2))
      return false;

   struct exec_list *not2_list = is_ne2 ? &if_eq2->then_list : &if_eq2->else_list;
   struct exec_list *eq2_list = is_ne2 ? &if_eq2->else_list : &if_eq2->then_list;
   if (!cf_list_is_empty_block(eq2_list))
      return false;

   /* Inside not2_list: must contain `if (val == 1)` (or `val != 1`). */
   nir_if *if_eq1 = NULL;
   foreach_list_typed(nir_cf_node, cn, node, not2_list) {
      if (cn->type == nir_cf_node_if) {
         if (if_eq1)
            return false;
         if_eq1 = nir_cf_node_as_if(cn);
      } else if (cn->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!if_eq1)
      return false;
   bool is_ne1 = false;
   if (!match_int_eq_const(if_eq1->condition.ssa, &center_load->def, 1,
                           &is_ne1))
      return false;
   struct exec_list *eq1_list = is_ne1 ? &if_eq1->else_list : &if_eq1->then_list;
   struct exec_list *not1_list = is_ne1 ? &if_eq1->then_list : &if_eq1->else_list;
   if (!cf_list_is_empty_block(not1_list))
      return false;

   /* Inside eq1_list: must contain the single interior/border `if`. */
   nir_if *if_int = NULL;
   foreach_list_typed(nir_cf_node, cn, node, eq1_list) {
      if (cn->type == nir_cf_node_if) {
         if (if_int)
            return false;
         if_int = nir_cf_node_as_if(cn);
      } else if (cn->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!if_int)
      return false;

   bool int_in_then = false;
   if (match_interior_1px_cond(if_int->condition.ssa, false, x, y, w_pc, h_pc,
                               s->info.workgroup_size))
      int_in_then = true;
   else if (match_interior_1px_cond(if_int->condition.ssa, true, x, y, w_pc,
                                    h_pc, s->info.workgroup_size))
      int_in_then = false;
   else
      return false;

   struct exec_list *int_list =
      int_in_then ? &if_int->then_list : &if_int->else_list;
   struct exec_list *bdr_list =
      int_in_then ? &if_int->else_list : &if_int->then_list;

   /* Interior arm must be a single block with 8 src loads at all 8 neighbors. */
   if (!exec_list_is_singular(int_list))
      return false;
   nir_block *int_blk =
      nir_cf_node_as_block(exec_node_data(nir_cf_node, exec_list_get_head(int_list), node));
   nir_intrinsic_instr *int_loads[8];
   unsigned int_load_cnt = 0;
   uint8_t int_mask = 0;
   nir_foreach_instr(instr, int_blk) {
      if (instr->type == nir_instr_type_jump ||
          instr->type == nir_instr_type_tex ||
          instr->type == nir_instr_type_call)
         return false;
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_load_deref) {
         if (int_load_cnt >= 8 || intr->def.bit_size != 8 ||
             intr->def.num_components != 1)
            return false;
         uint32_t ls = 0, lb = 0;
         nir_def *lidx = NULL;
         if (!match_ssbo_array_deref(nir_src_as_deref(intr->src[0]), 1, &ls,
                                     &lb, &lidx) ||
             ls != src_set || lb != src_bind)
            return false;
         struct hyst_lin_idx lin = {0};
         if (!decompose_hyst_idx(lidx, x, y, w_pc, s->info.workgroup_size, &lin,
                                 1, 0) ||
             lin.c_x != 1 || lin.c_yw != 1)
            return false;
         int bit = neighbor_bit_from_dx_dy(lin.c_const, lin.c_w);
         if (bit < 0 || (int_mask & (1u << bit)))
            return false;
         int_mask |= (uint8_t)(1u << bit);
         int_loads[int_load_cnt++] = intr;
      } else if (!is_allowed_base_intrinsic(intr->intrinsic)) {
         return false;
      }
   }
   if (int_load_cnt != 8 || int_mask != 0xffu)
      return false;

   /* Verify border arm has all 8 guarded neighbors. */
   nir_block *int_merge =
      nir_cf_node_as_block(nir_cf_node_next(&if_int->cf_node));
   nir_phi_instr *phi_int = NULL;
   nir_foreach_phi(p, int_merge) {
      if (phi_int)
         return false;
      phi_int = p;
   }
   if (!phi_int)
      return false;

   nir_def *int_res = NULL, *bdr_res = NULL;
   nir_block *int_tail =
      int_in_then ? nir_if_last_then_block(if_int) : nir_if_last_else_block(if_int);
   nir_block *bdr_tail =
      int_in_then ? nir_if_last_else_block(if_int) : nir_if_last_then_block(if_int);
   nir_foreach_phi_src(ps, phi_int) {
      if (ps->pred == int_tail)
         int_res = strip_mov(ps->src.ssa);
      else if (ps->pred == bdr_tail)
         bdr_res = strip_mov(ps->src.ssa);
   }
   if (!int_res || !bdr_res)
      return false;

   /* Verify int_res = bcsel(any_nbr >= 2, 3, 1). */
   if (nir_def_instr(int_res)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *int_sel = nir_instr_as_alu(nir_def_instr(int_res));
   if (int_sel->op != nir_op_bcsel || !is_const_i32(int_sel->src[1].src.ssa, 3) ||
       !is_const_i32(int_sel->src[2].src.ssa, 1))
      return false;
   nir_intrinsic_instr *used_loads[8];
   unsigned used_cnt = 0;
   if (!collect_ge2_or_loads(int_sel->src[0].src.ssa, used_loads, &used_cnt, 8) ||
       used_cnt != 8)
      return false;
   uint8_t used_mask = 0;
   for (unsigned i = 0; i < 8; i++) {
      for (unsigned j = 0; j < 8; j++) {
         if (used_loads[i] == int_loads[j])
            used_mask |= (uint8_t)(1u << j);
      }
   }
   if (used_mask != 0xffu)
      return false;

   /* Find the initial `1` constant used by the border chain (must be const 1). */
   nir_def *bdr_init = NULL;
   /* Check first leaf or phi in border list for initial value `1`. */
   nir_foreach_block(blk, impl) {
      nir_foreach_instr(instr, blk) {
         if (instr->type == nir_instr_type_load_const) {
            nir_load_const_instr *lc = nir_instr_as_load_const(instr);
            if (lc->def.bit_size == 32 && lc->def.num_components == 1 &&
                lc->value[0].i32 == 1) {
               /* Find the const 1 actually feeding bdr_list. */
               struct hyst_border_ctx bctx = {
                  .x = x, .y = y, .w_pc = w_pc, .h_pc = h_pc,
                  .wg_size = s->info.workgroup_size,
                  .src_set = src_set, .src_bind = src_bind,
               };
               nir_def *bdr_out = NULL;
               if (verify_hyst_border_cf(bdr_list, &bctx, &lc->def, &bdr_out) &&
                   bctx.seen_mask == 0xffu && bdr_out == bdr_res) {
                  bdr_init = &lc->def;
                  break;
               }
            }
         }
      }
      if (bdr_init)
         break;
   }
   if (!bdr_init)
      return false;

   /* Verify merge of if_eq1: eq1 -> phi_int, not1 -> u2u32(center_load). */
   nir_block *eq1_merge =
      nir_cf_node_as_block(nir_cf_node_next(&if_eq1->cf_node));
   nir_phi_instr *phi_eq1 = NULL;
   nir_foreach_phi(p, eq1_merge) {
      if (phi_eq1)
         return false;
      phi_eq1 = p;
   }
   if (!phi_eq1)
      return false;
   nir_def *p1_t = NULL, *p1_e = NULL;
   nir_block *eq1_tail =
      is_ne1 ? nir_if_last_else_block(if_eq1) : nir_if_last_then_block(if_eq1);
   nir_block *not1_tail =
      is_ne1 ? nir_if_last_then_block(if_eq1) : nir_if_last_else_block(if_eq1);
   nir_foreach_phi_src(ps, phi_eq1) {
      if (ps->pred == eq1_tail)
         p1_t = strip_mov(ps->src.ssa);
      else if (ps->pred == not1_tail)
         p1_e = strip_mov(ps->src.ssa);
   }
   if (p1_t != &phi_int->def || strip_u2u32(p1_e) != &center_load->def)
      return false;

   /* Verify merge of if_eq2: not2 -> phi_eq1, eq2 -> 3, followed by single store. */
   nir_block *eq2_merge =
      nir_cf_node_as_block(nir_cf_node_next(&if_eq2->cf_node));
   nir_phi_instr *phi_eq2 = NULL;
   nir_foreach_phi(p, eq2_merge) {
      if (phi_eq2)
         return false;
      phi_eq2 = p;
   }
   if (!phi_eq2)
      return false;
   nir_def *p2_not2 = NULL, *p2_eq2 = NULL;
   nir_block *not2_tail =
      is_ne2 ? nir_if_last_then_block(if_eq2) : nir_if_last_else_block(if_eq2);
   nir_block *eq2_tail =
      is_ne2 ? nir_if_last_else_block(if_eq2) : nir_if_last_then_block(if_eq2);
   nir_foreach_phi_src(ps, phi_eq2) {
      if (ps->pred == not2_tail)
         p2_not2 = strip_mov(ps->src.ssa);
      else if (ps->pred == eq2_tail)
         p2_eq2 = strip_mov(ps->src.ssa);
   }
   if (p2_not2 != &phi_eq1->def || !is_const_i32(p2_eq2, 3))
      return false;

   nir_intrinsic_instr *store_intr = NULL;
   nir_foreach_instr(instr, eq2_merge) {
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_deref) {
            if (store_intr)
               return false;
            store_intr = intr;
         }
      }
   }
   if (!store_intr)
      return false;

   uint32_t dst_set = 0, dst_bind = 0;
   nir_def *dst_idx = NULL;
   uint16_t dst_stride_pc = 0;
   if (!match_ssbo_array_deref(nir_src_as_deref(store_intr->src[0]), 1,
                               &dst_set, &dst_bind, &dst_idx) ||
       !match_linear_index(dst_idx, y, x, &dst_stride_pc,
                           s->info.workgroup_size) ||
       dst_stride_pc != w_off)
      return false;

   nir_def *st_val = strip_mov(store_intr->src[1].ssa);
   if (!st_val || st_val->bit_size != 8 || st_val->num_components != 1 ||
       nir_def_instr(st_val)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *st_trunc = nir_instr_as_alu(nir_def_instr(st_val));
   if ((st_trunc->op != nir_op_u2u8 && st_trunc->op != nir_op_i2i8) ||
       strip_mov(st_trunc->src[0].src.ssa) != &phi_eq2->def)
      return false;

   /* Completeness: exactly 17 loads (1 center + 8 interior + 8 border) and 1 store. */
   unsigned total_loads = 0, total_stores = 0;
   nir_foreach_block(blk, impl) {
      nir_foreach_instr(instr, blk) {
         if (instr->type == nir_instr_type_tex ||
             instr->type == nir_instr_type_call)
            return false;
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_deref)
               total_loads++;
            else if (intr->intrinsic == nir_intrinsic_store_deref)
               total_stores++;
            else if (!is_allowed_base_intrinsic(intr->intrinsic))
               return false;
         }
      }
   }
   if (total_loads != 17 || total_stores != 1)
      return false;

   if (!populate_binding_info(set_layout_count, set_layouts, src_set, src_bind,
                              &plan->bufs[0]) ||
       !populate_binding_info(set_layout_count, set_layouts, dst_set, dst_bind,
                              &plan->bufs[1]))
      return false;

   plan->kind = HK_COMPUTE_OPT_CANNY_HYSTERESIS_U8;
   plan->tile_w = 256;
   plan->tile_h = 4;
   plan->pc_off[0] = w_off;
   plan->pc_off[1] = h_off;
   return true;
}

/* ---------------------------------------------------------------------------
 * Matcher 3: Canny Non-Maximum Suppression (mag[f32] + dir[i32] -> dst[u8])
 * --------------------------------------------------------------------------- */

/* Match `((lut >> ((d << 1) [& 31])) & 3) - 1` or
 * `ubitfield_extract(lut, (d << 1) [& 31], 2) - 1`. */
static bool
match_dir_lut_delta(nir_def *def, nir_def *dir_def, uint32_t *lut_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *sub = nir_instr_as_alu(nir_def_instr(def));
   nir_def *bfe = NULL;
   if (sub->op == nir_op_iadd) {
      for (unsigned s = 0; s < 2; s++) {
         if (is_const_i32(sub->src[s].src.ssa, -1))
            bfe = strip_mov(sub->src[1 - s].src.ssa);
      }
   } else if (sub->op == nir_op_isub && is_const_i32(sub->src[1].src.ssa, 1)) {
      bfe = strip_mov(sub->src[0].src.ssa);
   }
   if (!bfe || nir_def_instr(bfe)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *ba = nir_instr_as_alu(nir_def_instr(bfe));
   uint32_t lut = 0;
   nir_def *shift = NULL;
   if (ba->op == nir_op_ubitfield_extract) {
      if (!match_const_u32(ba->src[0].src.ssa, &lut) ||
          !is_const_i32(ba->src[2].src.ssa, 2))
         return false;
      shift = strip_mov(ba->src[1].src.ssa);
   } else if (ba->op == nir_op_iand) {
      for (unsigned s = 0; s < 2; s++) {
         if (!is_const_i32(ba->src[s].src.ssa, 3))
            continue;
         nir_def *sh = strip_mov(ba->src[1 - s].src.ssa);
         if (sh && nir_def_instr(sh)->type == nir_instr_type_alu) {
            nir_alu_instr *sha = nir_instr_as_alu(nir_def_instr(sh));
            if (sha->op == nir_op_ushr &&
                match_const_u32(sha->src[0].src.ssa, &lut)) {
               shift = strip_mov(sha->src[1].src.ssa);
            }
         }
      }
   }
   if (!shift)
      return false;

   /* Strip optional `& 31`. */
   if (nir_def_instr(shift)->type == nir_instr_type_alu) {
      nir_alu_instr *ma = nir_instr_as_alu(nir_def_instr(shift));
      if (ma->op == nir_op_iand) {
         for (unsigned s = 0; s < 2; s++) {
            if (is_const_i32(ma->src[s].src.ssa, 31)) {
               shift = strip_mov(ma->src[1 - s].src.ssa);
               break;
            }
         }
      }
   }

   /* Match `dir << 1`, `dir * 2`, or `dir + dir`. */
   if (!shift || nir_def_instr(shift)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *sa = nir_instr_as_alu(nir_def_instr(shift));
   bool ok_shift = false;
   if (sa->op == nir_op_ishl &&
       strip_mov(sa->src[0].src.ssa) == strip_mov(dir_def) &&
       is_const_i32(sa->src[1].src.ssa, 1))
      ok_shift = true;
   else if (sa->op == nir_op_imul) {
      for (unsigned s = 0; s < 2; s++) {
         if (strip_mov(sa->src[s].src.ssa) == strip_mov(dir_def) &&
             is_const_i32(sa->src[1 - s].src.ssa, 2))
            ok_shift = true;
      }
   } else if (sa->op == nir_op_iadd &&
              strip_mov(sa->src[0].src.ssa) == strip_mov(dir_def) &&
              strip_mov(sa->src[1].src.ssa) == strip_mov(dir_def))
      ok_shift = true;

   if (!ok_shift)
      return false;

   /* Every 2-bit entry in `lut` must be in {0, 1, 2} so `(entry - 1)` is
    * strictly in {-1, 0, +1} for every possible 32-bit `dir`. */
   for (unsigned k = 0; k < 16; k++) {
      if (((lut >> (2 * k)) & 3u) == 3u)
         return false;
   }

   *lut_out = lut;
   return true;
}

static bool
match_add_or_sub_delta(nir_def *def, nir_def *base, nir_def *delta,
                       bool want_sub, const uint16_t wg_size[3])
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (!want_sub && alu->op == nir_op_iadd) {
      return (same_val(alu->src[0].src.ssa, base, wg_size) &&
              strip_mov(alu->src[1].src.ssa) == strip_mov(delta)) ||
             (same_val(alu->src[1].src.ssa, base, wg_size) &&
              strip_mov(alu->src[0].src.ssa) == strip_mov(delta));
   }
   if (want_sub && alu->op == nir_op_isub) {
      return same_val(alu->src[0].src.ssa, base, wg_size) &&
             strip_mov(alu->src[1].src.ssa) == strip_mov(delta);
   }
   if (want_sub && alu->op == nir_op_iadd) {
      for (unsigned s = 0; s < 2; s++) {
         if (!same_val(alu->src[s].src.ssa, base, wg_size))
            continue;
         nir_def *neg = strip_mov(alu->src[1 - s].src.ssa);
         if (neg && nir_def_instr(neg)->type == nir_instr_type_alu) {
            nir_alu_instr *na = nir_instr_as_alu(nir_def_instr(neg));
            if (na->op == nir_op_ineg &&
                strip_mov(na->src[0].src.ssa) == strip_mov(delta))
               return true;
         }
      }
   }
   return false;
}

static bool
match_flt_pair(nir_def *cond, nir_def *lo, nir_def *hi)
{
   cond = strip_mov(cond);
   if (!cond || cond->bit_size != 1 ||
       nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   return alu->op == nir_op_flt &&
          strip_mov(alu->src[0].src.ssa) == strip_mov(lo) &&
          strip_mov(alu->src[1].src.ssa) == strip_mov(hi);
}

static bool
match_canny_nms(nir_shader *s, uint32_t set_layout_count,
                struct vk_descriptor_set_layout *const *set_layouts,
                struct hk_compute_opt_plan *plan)
{
   if (s->info.num_ssbos != 3)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_if *outer_if = NULL;
   struct exec_list *body_list = NULL;
   bool early_ret = false;
   nir_def *x = NULL, *y = NULL, *w_pc = NULL, *h_pc = NULL;
   if (!match_top_level_xy_guard(impl, s->info.workgroup_size, &outer_if,
                                 &body_list, &early_ret, &x, &y, &w_pc, &h_pc))
      return false;

   uint16_t w_off = 0, h_off = 0;
   if (!match_push_const_u32(w_pc, &w_off) ||
       !match_push_const_u32(h_pc, &h_off))
      return false;

   /* Find the single interior `if` inside body_list. */
   nir_if *if_int = NULL;
   foreach_list_typed(nir_cf_node, cn, node, body_list) {
      if (cn->type == nir_cf_node_if) {
         nir_if *cand = nir_cf_node_as_if(cn);
         if (cand == outer_if)
            continue;
         if (if_int)
            return false;
         if_int = cand;
      } else if (cn->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!if_int)
      return false;

   bool int_in_then = false;
   if (match_interior_1px_cond(if_int->condition.ssa, false, x, y, w_pc, h_pc,
                               s->info.workgroup_size))
      int_in_then = true;
   else if (match_interior_1px_cond(if_int->condition.ssa, true, x, y, w_pc,
                                    h_pc, s->info.workgroup_size))
      int_in_then = false;
   else
      return false;

   struct exec_list *int_list =
      int_in_then ? &if_int->then_list : &if_int->else_list;
   struct exec_list *bdr_list =
      int_in_then ? &if_int->else_list : &if_int->then_list;
   if (!exec_list_is_singular(int_list) || !cf_list_is_empty_block(bdr_list))
      return false;

   nir_block *int_blk = nir_cf_node_as_block(
      exec_node_data(nir_cf_node, exec_list_get_head(int_list), node));

   /* Collect the 4 32-bit loads inside int_blk: mag_c, dir_c, mag_p, mag_m. */
   nir_intrinsic_instr *loads[4];
   unsigned nloads = 0;
   nir_foreach_instr(instr, int_blk) {
      if (instr->type == nir_instr_type_jump ||
          instr->type == nir_instr_type_tex ||
          instr->type == nir_instr_type_call)
         return false;
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_load_deref) {
         if (nloads >= 4 || intr->def.bit_size != 32 ||
             intr->def.num_components != 1)
            return false;
         loads[nloads++] = intr;
      } else if (!is_allowed_base_intrinsic(intr->intrinsic)) {
         return false;
      }
   }
   if (nloads != 4)
      return false;

   /* Identify mag_c and dir_c (both indexed at `y * width + x`) and the 2
    * neighbor mag loads. */
   nir_intrinsic_instr *mag_c = NULL, *dir_c = NULL;
   nir_intrinsic_instr *nbr[2] = {NULL, NULL};
   unsigned nnbr = 0;
   uint32_t l_set[4], l_bind[4];
   nir_def *l_idx[4];
   for (unsigned i = 0; i < 4; i++) {
      if (!match_ssbo_array_deref(nir_src_as_deref(loads[i]->src[0]), 4,
                                  &l_set[i], &l_bind[i], &l_idx[i]))
         return false;
   }

   /* Find which 2 loads share `(set, binding)` with a third load -> that's `mag`. */
   uint32_t mag_set = 0, mag_bind = 0, dir_set = 0, dir_bind = 0;
   bool found_mag = false;
   for (unsigned i = 0; i < 4; i++) {
      unsigned cnt = 0;
      for (unsigned j = 0; j < 4; j++) {
         if (l_set[j] == l_set[i] && l_bind[j] == l_bind[i])
            cnt++;
      }
      if (cnt == 3) {
         mag_set = l_set[i];
         mag_bind = l_bind[i];
         found_mag = true;
      } else if (cnt == 1) {
         dir_set = l_set[i];
         dir_bind = l_bind[i];
         dir_c = loads[i];
      }
   }
   if (!found_mag || !dir_c)
      return false;

   uint16_t dir_stride = 0;
   for (unsigned i = 0; i < 4; i++) {
      if (loads[i] == dir_c) {
         if (!match_linear_index(l_idx[i], y, x, &dir_stride,
                                 s->info.workgroup_size) ||
             dir_stride != w_off)
            return false;
         continue;
      }
      uint16_t spc = 0;
      if (!mag_c &&
          match_linear_index(l_idx[i], y, x, &spc, s->info.workgroup_size) &&
          spc == w_off) {
         mag_c = loads[i];
      } else if (nnbr < 2) {
         nbr[nnbr++] = loads[i];
      } else {
         return false;
      }
   }
   if (!mag_c || nnbr != 2)
      return false;

   /* Verify the 2 neighbor indices are `(y + dy)*W + (x + dx)` and
    * `(y - dy)*W + (x - dx)` where `(dx, dy)` come from LUTs on `dir_c`. */
   nir_def *nidx0 = NULL, *nidx1 = NULL;
   for (unsigned i = 0; i < 4; i++) {
      if (loads[i] == nbr[0])
         nidx0 = l_idx[i];
      if (loads[i] == nbr[1])
         nidx1 = l_idx[i];
   }
   if (!nidx0 || !nidx1)
      return false;

   /* Extract (row0, col0) from nidx0 = row0 * W + col0. */
   nir_def *row0 = NULL, *col0 = NULL, *row1 = NULL, *col1 = NULL;
   for (unsigned k = 0; k < 2; k++) {
      nir_def *ni = k == 0 ? strip_mov(nidx0) : strip_mov(nidx1);
      if (!ni || nir_def_instr(ni)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(ni));
      if (add->op != nir_op_iadd)
         return false;
      nir_def *r = NULL, *c = NULL;
      for (unsigned side = 0; side < 2; side++) {
         nir_def *prod = strip_mov(add->src[side].src.ssa);
         if (!prod || nir_def_instr(prod)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(prod));
         if (mul->op != nir_op_imul && mul->op != nir_op_amul)
            continue;
         for (unsigned ms = 0; ms < 2; ms++) {
            uint16_t spc = 0;
            if (match_push_const_u32(mul->src[ms].src.ssa, &spc) &&
                spc == w_off) {
               r = strip_mov(mul->src[1 - ms].src.ssa);
               c = strip_mov(add->src[1 - side].src.ssa);
               break;
            }
         }
         if (r)
            break;
      }
      if (!r || !c)
         return false;
      if (k == 0) {
         row0 = r;
         col0 = c;
      } else {
         row1 = r;
         col1 = c;
      }
   }

   /* One of (col0, row0) is (x + dx, y + dy) and the other is (x - dx, y - dy). */
   uint32_t dx_lut = 0, dy_lut = 0;
   bool matched_deltas = false;
   for (unsigned swap = 0; swap < 2 && !matched_deltas; swap++) {
      nir_def *c_plus = swap ? col1 : col0;
      nir_def *r_plus = swap ? row1 : row0;
      nir_def *c_minus = swap ? col0 : col1;
      nir_def *r_minus = swap ? row0 : row1;

      if (nir_def_instr(c_plus)->type != nir_instr_type_alu ||
          nir_def_instr(r_plus)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *ca = nir_instr_as_alu(nir_def_instr(c_plus));
      nir_alu_instr *ra = nir_instr_as_alu(nir_def_instr(r_plus));
      if (ca->op != nir_op_iadd || ra->op != nir_op_iadd)
         continue;

      nir_def *dx = NULL, *dy = NULL;
      for (unsigned side = 0; side < 2; side++) {
         if (same_val(ca->src[side].src.ssa, x, s->info.workgroup_size))
            dx = strip_mov(ca->src[1 - side].src.ssa);
         if (same_val(ra->src[side].src.ssa, y, s->info.workgroup_size))
            dy = strip_mov(ra->src[1 - side].src.ssa);
      }
      if (!dx || !dy)
         continue;
      if (match_dir_lut_delta(dx, &dir_c->def, &dx_lut) &&
          match_dir_lut_delta(dy, &dir_c->def, &dy_lut) &&
          match_add_or_sub_delta(c_minus, x, dx, true, s->info.workgroup_size) &&
          match_add_or_sub_delta(r_minus, y, dy, true, s->info.workgroup_size))
         matched_deltas = true;
   }
   if (!matched_deltas)
      return false;

   /* Verify merge phi and output expression:
    *   out = (m > nbr0 && m > nbr1) ? ((m > high) ? 2 : (m > low ? 1 : 0)) : 0 */
   nir_block *merge = nir_cf_node_as_block(nir_cf_node_next(&if_int->cf_node));
   nir_phi_instr *phi = NULL;
   nir_foreach_phi(p, merge) {
      if (phi)
         return false;
      phi = p;
   }
   if (!phi)
      return false;

   nir_def *int_val = NULL, *bdr_val = NULL;
   nir_block *int_tail =
      int_in_then ? nir_if_last_then_block(if_int) : nir_if_last_else_block(if_int);
   nir_block *bdr_tail =
      int_in_then ? nir_if_last_else_block(if_int) : nir_if_last_then_block(if_int);
   nir_foreach_phi_src(ps, phi) {
      if (ps->pred == int_tail)
         int_val = strip_mov(ps->src.ssa);
      else if (ps->pred == bdr_tail)
         bdr_val = strip_mov(ps->src.ssa);
   }
   if (!int_val || !is_const_i32(bdr_val, 0))
      return false;

   /* Unwrap the two NMS neighbor checks (nested bcsel or combined iand bcsel). */
   nir_def *thresh_val = NULL;
   if (nir_def_instr(int_val)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *s1 = nir_instr_as_alu(nir_def_instr(int_val));
   if (s1->op != nir_op_bcsel || !is_const_i32(s1->src[2].src.ssa, 0))
      return false;

   struct bool_term nms_terms[4];
   unsigned nnms = 0;
   bool got_n0 = false, got_n1 = false;
   if (collect_and_terms(s1->src[0].src.ssa, false, nms_terms, &nnms, 4) &&
       nnms == 2 && !nms_terms[0].negated && !nms_terms[1].negated) {
      for (unsigned i = 0; i < 2; i++) {
         if (match_flt_pair(nms_terms[i].def, &nbr[0]->def, &mag_c->def))
            got_n0 = true;
         else if (match_flt_pair(nms_terms[i].def, &nbr[1]->def, &mag_c->def))
            got_n1 = true;
      }
      if (got_n0 && got_n1)
         thresh_val = strip_mov(s1->src[1].src.ssa);
   }
   if (!thresh_val) {
      /* `flt(fmax(nbr0, nbr1), mag_c)` canonicalized by nir_opt_algebraic. */
      nir_def *c0 = strip_mov(s1->src[0].src.ssa);
      if (c0 && nir_def_instr(c0)->type == nir_instr_type_alu) {
         nir_alu_instr *ca = nir_instr_as_alu(nir_def_instr(c0));
         if (ca->op == nir_op_flt &&
             strip_mov(ca->src[1].src.ssa) == &mag_c->def) {
            nir_def *mx = strip_mov(ca->src[0].src.ssa);
            if (mx && nir_def_instr(mx)->type == nir_instr_type_alu) {
               nir_alu_instr *ma = nir_instr_as_alu(nir_def_instr(mx));
               if (ma->op == nir_op_fmax) {
                  nir_def *a = strip_mov(ma->src[0].src.ssa);
                  nir_def *b = strip_mov(ma->src[1].src.ssa);
                  if ((a == &nbr[0]->def && b == &nbr[1]->def) ||
                      (a == &nbr[1]->def && b == &nbr[0]->def))
                     thresh_val = strip_mov(s1->src[1].src.ssa);
               }
            }
         }
      }
   }
   if (!thresh_val) {
      /* Nested bcsel: `bcsel(nbrA < mag, bcsel(nbrB < mag, thresh_val, 0), 0)` */
      nir_def *inner = strip_mov(s1->src[1].src.ssa);
      if (inner && nir_def_instr(inner)->type == nir_instr_type_alu) {
         nir_alu_instr *s2 = nir_instr_as_alu(nir_def_instr(inner));
         if (s2->op == nir_op_bcsel && is_const_i32(s2->src[2].src.ssa, 0)) {
            if ((match_flt_pair(s1->src[0].src.ssa, &nbr[0]->def, &mag_c->def) &&
                 match_flt_pair(s2->src[0].src.ssa, &nbr[1]->def, &mag_c->def)) ||
                (match_flt_pair(s1->src[0].src.ssa, &nbr[1]->def, &mag_c->def) &&
                 match_flt_pair(s2->src[0].src.ssa, &nbr[0]->def, &mag_c->def)))
               thresh_val = strip_mov(s2->src[1].src.ssa);
         }
      }
   }
   if (!thresh_val || nir_def_instr(thresh_val)->type != nir_instr_type_alu)
      return false;

   /* Match `bcsel(high < mag, 2, bcsel(low < mag, 1, 0))`. */
   nir_alu_instr *th = nir_instr_as_alu(nir_def_instr(thresh_val));
   if (th->op != nir_op_bcsel || !is_const_i32(th->src[1].src.ssa, 2))
      return false;
   nir_def *hi_cmp = strip_mov(th->src[0].src.ssa);
   if (!hi_cmp || nir_def_instr(hi_cmp)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *hca = nir_instr_as_alu(nir_def_instr(hi_cmp));
   uint32_t hi_bits = 0, lo_bits = 0;
   if (hca->op != nir_op_flt ||
       !match_const_f32_bits(hca->src[0].src.ssa, &hi_bits) ||
       strip_mov(hca->src[1].src.ssa) != &mag_c->def)
      return false;

   nir_def *lo_sel = strip_mov(th->src[2].src.ssa);
   if (!lo_sel || nir_def_instr(lo_sel)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *lsel = nir_instr_as_alu(nir_def_instr(lo_sel));
   nir_def *lo_cmp = NULL;
   if (lsel->op == nir_op_bcsel && is_const_i32(lsel->src[1].src.ssa, 1) &&
       is_const_i32(lsel->src[2].src.ssa, 0)) {
      lo_cmp = strip_mov(lsel->src[0].src.ssa);
   } else if (lsel->op == nir_op_b2i32) {
      lo_cmp = strip_mov(lsel->src[0].src.ssa);
   } else {
      return false;
   }
   if (!lo_cmp || nir_def_instr(lo_cmp)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *lca = nir_instr_as_alu(nir_def_instr(lo_cmp));
   if (lca->op != nir_op_flt ||
       !match_const_f32_bits(lca->src[0].src.ssa, &lo_bits) ||
       strip_mov(lca->src[1].src.ssa) != &mag_c->def)
      return false;

   float lo_f = 0.0f, hi_f = 0.0f;
   memcpy(&lo_f, &lo_bits, sizeof(lo_f));
   memcpy(&hi_f, &hi_bits, sizeof(hi_f));
   if (!isfinite(lo_f) || !isfinite(hi_f) || lo_f > hi_f)
      return false;

   /* Single store at merge block: dst[y * width + x] = u2u8(phi). */
   nir_intrinsic_instr *store_intr = NULL;
   nir_foreach_instr(instr, merge) {
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_deref) {
            if (store_intr)
               return false;
            store_intr = intr;
         }
      }
   }
   if (!store_intr)
      return false;

   uint32_t dst_set = 0, dst_bind = 0;
   nir_def *dst_idx = NULL;
   uint16_t dst_stride = 0;
   if (!match_ssbo_array_deref(nir_src_as_deref(store_intr->src[0]), 1,
                               &dst_set, &dst_bind, &dst_idx) ||
       !match_linear_index(dst_idx, y, x, &dst_stride, s->info.workgroup_size) ||
       dst_stride != w_off)
      return false;
   nir_def *st_val = strip_mov(store_intr->src[1].ssa);
   if (!st_val || st_val->bit_size != 8 || st_val->num_components != 1 ||
       nir_def_instr(st_val)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *st_trunc = nir_instr_as_alu(nir_def_instr(st_val));
   if ((st_trunc->op != nir_op_u2u8 && st_trunc->op != nir_op_i2i8) ||
       strip_mov(st_trunc->src[0].src.ssa) != &phi->def)
      return false;

   /* Completeness: exactly 4 loads and 1 store across the entire shader. */
   unsigned total_loads = 0, total_stores = 0;
   nir_foreach_block(blk, impl) {
      nir_foreach_instr(instr, blk) {
         if (instr->type == nir_instr_type_tex ||
             instr->type == nir_instr_type_call)
            return false;
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_deref)
               total_loads++;
            else if (intr->intrinsic == nir_intrinsic_store_deref)
               total_stores++;
            else if (!is_allowed_base_intrinsic(intr->intrinsic))
               return false;
         }
      }
   }
   if (total_loads != 4 || total_stores != 1)
      return false;

   if (!populate_binding_info(set_layout_count, set_layouts, dir_set, dir_bind,
                              &plan->bufs[0]) ||
       !populate_binding_info(set_layout_count, set_layouts, mag_set, mag_bind,
                              &plan->bufs[1]) ||
       !populate_binding_info(set_layout_count, set_layouts, dst_set, dst_bind,
                              &plan->bufs[2]))
      return false;

   plan->kind = HK_COMPUTE_OPT_CANNY_NMS_U8;
   plan->tile_w = 256;
   plan->tile_h = 4;
   plan->pc_off[0] = w_off;
   plan->pc_off[1] = h_off;
   plan->params[0] = dx_lut;
   plan->params[1] = dy_lut;
   plan->params[2] = lo_bits;
   plan->params[3] = hi_bits;
   return true;
}

/* ---------------------------------------------------------------------------
 * Matcher 4: Canny Finalize (`state >= 3 ? u8(clamp(mag, 0, 255)) : 0`)
 * --------------------------------------------------------------------------- */

static bool
match_canny_finalize(nir_shader *s, uint32_t set_layout_count,
                     struct vk_descriptor_set_layout *const *set_layouts,
                     struct hk_compute_opt_plan *plan)
{
   if (s->info.num_ssbos != 3)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_if *outer_if = NULL;
   struct exec_list *body_list = NULL;
   bool early_ret = false;
   nir_def *x = NULL, *y = NULL, *w_pc = NULL, *h_pc = NULL;
   if (!match_top_level_xy_guard(impl, s->info.workgroup_size, &outer_if,
                                 &body_list, &early_ret, &x, &y, &w_pc, &h_pc))
      return false;

   uint16_t w_off = 0, h_off = 0;
   if (!match_push_const_u32(w_pc, &w_off) ||
       !match_push_const_u32(h_pc, &h_off))
      return false;

   nir_block *entry_blk =
      early_ret ? nir_cf_node_as_block(nir_cf_node_next(&outer_if->cf_node))
                : nir_cf_node_as_block(
                     exec_node_data(nir_cf_node, exec_list_get_head(body_list),
                                    node));
   if (!entry_blk)
      return false;

   nir_intrinsic_instr *state_load = NULL;
   nir_foreach_instr(instr, entry_blk) {
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_deref) {
            if (state_load)
               return false;
            state_load = intr;
         }
      }
   }
   if (!state_load || state_load->def.bit_size != 8 ||
       state_load->def.num_components != 1)
      return false;

   uint32_t st_set = 0, st_bind = 0;
   nir_def *st_idx = NULL;
   uint16_t st_stride = 0;
   if (!match_ssbo_array_deref(nir_src_as_deref(state_load->src[0]), 1,
                               &st_set, &st_bind, &st_idx) ||
       !match_linear_index(st_idx, y, x, &st_stride, s->info.workgroup_size) ||
       st_stride != w_off)
      return false;

   nir_cf_node *n1 = nir_cf_node_next(&entry_blk->cf_node);
   if (!n1 || n1->type != nir_cf_node_if)
      return false;
   nir_if *if_edge = nir_cf_node_as_if(n1);

   /* Match `state >= thresh` (in either arm). */
   nir_def *cond = strip_mov(if_edge->condition.ssa);
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *ca = nir_instr_as_alu(nir_def_instr(cond));
   uint32_t thresh = 0;
   bool edge_in_then = false;
   if ((ca->op == nir_op_ige || ca->op == nir_op_uge) &&
       strip_u2u32(ca->src[0].src.ssa) == &state_load->def &&
       match_const_u32_any(ca->src[1].src.ssa, &thresh) &&
       thresh >= 1 && thresh <= 255) {
      edge_in_then = true;
   } else if ((ca->op == nir_op_ilt || ca->op == nir_op_ult) &&
              strip_u2u32(ca->src[0].src.ssa) == &state_load->def &&
              match_const_u32_any(ca->src[1].src.ssa, &thresh) &&
              thresh >= 1 && thresh <= 255) {
      edge_in_then = false;
   } else {
      return false;
   }

   struct exec_list *edge_list =
      edge_in_then ? &if_edge->then_list : &if_edge->else_list;
   struct exec_list *zero_list =
      edge_in_then ? &if_edge->else_list : &if_edge->then_list;
   if (!exec_list_is_singular(edge_list) || !cf_list_is_empty_block(zero_list))
      return false;

   nir_block *edge_blk = nir_cf_node_as_block(
      exec_node_data(nir_cf_node, exec_list_get_head(edge_list), node));
   nir_intrinsic_instr *mag_load = NULL;
   nir_foreach_instr(instr, edge_blk) {
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_deref) {
            if (mag_load)
               return false;
            mag_load = intr;
         }
      }
   }
   if (!mag_load || mag_load->def.bit_size != 32 ||
       mag_load->def.num_components != 1)
      return false;

   uint32_t mag_set = 0, mag_bind = 0;
   nir_def *mag_idx = NULL;
   uint16_t mag_stride = 0;
   if (!match_ssbo_array_deref(nir_src_as_deref(mag_load->src[0]), 4,
                               &mag_set, &mag_bind, &mag_idx) ||
       !match_linear_index(mag_idx, y, x, &mag_stride, s->info.workgroup_size) ||
       mag_stride != w_off)
      return false;

   nir_block *merge =
      nir_cf_node_as_block(nir_cf_node_next(&if_edge->cf_node));
   nir_phi_instr *phi = NULL;
   nir_foreach_phi(p, merge) {
      if (phi)
         return false;
      phi = p;
   }
   if (!phi)
      return false;

   nir_def *edge_val = NULL, *zero_val = NULL;
   nir_block *edge_tail = edge_in_then ? nir_if_last_then_block(if_edge)
                                       : nir_if_last_else_block(if_edge);
   nir_block *zero_tail = edge_in_then ? nir_if_last_else_block(if_edge)
                                       : nir_if_last_then_block(if_edge);
   nir_foreach_phi_src(ps, phi) {
      if (ps->pred == edge_tail)
         edge_val = strip_mov(ps->src.ssa);
      else if (ps->pred == zero_tail)
         zero_val = strip_mov(ps->src.ssa);
   }
   if (!edge_val || !is_const_i32(zero_val, 0))
      return false;

   /* Verify edge_val = f2i32/f2u32(fmin(fmax(mag_load, 0.0f), 255.0f)). */
   if (nir_def_instr(edge_val)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *cvt = nir_instr_as_alu(nir_def_instr(edge_val));
   if (cvt->op != nir_op_f2i32 && cvt->op != nir_op_f2u32)
      return false;
   bool is_f2u32 = (cvt->op == nir_op_f2u32);

   nir_def *clamped = strip_mov(cvt->src[0].src.ssa);
   if (!clamped || nir_def_instr(clamped)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *mn = nir_instr_as_alu(nir_def_instr(clamped));
   if (mn->op != nir_op_fmin)
      return false;
   nir_def *mx_def = NULL;
   for (unsigned side = 0; side < 2; side++) {
      uint32_t bits = 0;
      if (match_const_f32_bits(mn->src[side].src.ssa, &bits) &&
          bits == 0x437f0000u) /* 255.0f */
         mx_def = strip_mov(mn->src[1 - side].src.ssa);
   }
   if (!mx_def || nir_def_instr(mx_def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *mx = nir_instr_as_alu(nir_def_instr(mx_def));
   if (mx->op != nir_op_fmax)
      return false;
   bool matched_mag = false;
   for (unsigned side = 0; side < 2; side++) {
      uint32_t bits = 0;
      if (match_const_f32_bits(mx->src[side].src.ssa, &bits) && bits == 0u &&
          strip_mov(mx->src[1 - side].src.ssa) == &mag_load->def)
         matched_mag = true;
   }
   if (!matched_mag)
      return false;

   /* Single store at merge: dst[y * dst_stride + x] = u2u8(phi). */
   nir_intrinsic_instr *store_intr = NULL;
   nir_foreach_instr(instr, merge) {
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_deref) {
            if (store_intr)
               return false;
            store_intr = intr;
         }
      }
   }
   if (!store_intr)
      return false;

   uint32_t dst_set = 0, dst_bind = 0;
   nir_def *dst_idx = NULL;
   uint16_t dst_stride = 0;
   if (!match_ssbo_array_deref(nir_src_as_deref(store_intr->src[0]), 1,
                               &dst_set, &dst_bind, &dst_idx) ||
       !match_linear_index(dst_idx, y, x, &dst_stride, s->info.workgroup_size))
      return false;
   nir_def *st_val = strip_mov(store_intr->src[1].ssa);
   if (!st_val || st_val->bit_size != 8 || st_val->num_components != 1 ||
       nir_def_instr(st_val)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *st_trunc = nir_instr_as_alu(nir_def_instr(st_val));
   if ((st_trunc->op != nir_op_u2u8 && st_trunc->op != nir_op_i2i8) ||
       strip_mov(st_trunc->src[0].src.ssa) != &phi->def)
      return false;

   unsigned total_loads = 0, total_stores = 0;
   nir_foreach_block(blk, impl) {
      nir_foreach_instr(instr, blk) {
         if (instr->type == nir_instr_type_tex ||
             instr->type == nir_instr_type_call)
            return false;
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_deref)
               total_loads++;
            else if (intr->intrinsic == nir_intrinsic_store_deref)
               total_stores++;
            else if (!is_allowed_base_intrinsic(intr->intrinsic))
               return false;
         }
      }
   }
   if (total_loads != 2 || total_stores != 1)
      return false;

   if (!populate_binding_info(set_layout_count, set_layouts, st_set, st_bind,
                              &plan->bufs[0]) ||
       !populate_binding_info(set_layout_count, set_layouts, mag_set, mag_bind,
                              &plan->bufs[1]) ||
       !populate_binding_info(set_layout_count, set_layouts, dst_set, dst_bind,
                              &plan->bufs[2]))
      return false;

   plan->kind = HK_COMPUTE_OPT_CANNY_FINALIZE_U8;
   plan->flags = is_f2u32 ? 1u : 0u;
   plan->tile_w = 256;
   plan->tile_h = 4;
   plan->pc_off[0] = w_off;
   plan->pc_off[1] = h_off;
   plan->pc_off[2] = dst_stride;
   plan->params[0] = thresh;
   return true;
}

/* ---------------------------------------------------------------------------
 * Matcher 5: Grid-stride bilinear u8 resize / copy with mirrored border pad
 * --------------------------------------------------------------------------- */

static bool
cf_list_is_single_break_co(struct exec_list *list)
{
   if (!exec_list_is_singular(list))
      return false;
   nir_cf_node *n = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   if (n->type != nir_cf_node_block)
      return false;
   nir_block *blk = nir_cf_node_as_block(n);
   if (exec_list_is_empty(&blk->instr_list))
      return false;
   nir_instr *last = nir_block_last_instr(blk);
   if (last->type != nir_instr_type_jump ||
       nir_instr_as_jump(last)->type != nir_jump_break)
      return false;
   nir_foreach_instr(instr, blk) {
      if (instr != last)
         return false;
   }
   return true;
}

static bool
match_grid_stride_loop_co(nir_loop *loop, unsigned comp,
                          const uint16_t wg[3], nir_def **coord_out,
                          uint16_t *limit_pc_out, nir_if **exit_if_out)
{
   nir_block *hdr = nir_loop_first_block(loop);
   nir_phi_instr *phi = NULL;
   nir_foreach_phi(p, hdr) {
      if (p->def.bit_size != 32 || p->def.num_components != 1)
         continue;
      nir_def *init = NULL, *latch = NULL;
      bool bad = false;
      nir_foreach_phi_src(ps, p) {
         if (ps->pred->index < hdr->index) {
            if (init)
               bad = true;
            init = ps->src.ssa;
         } else {
            if (latch)
               bad = true;
            latch = ps->src.ssa;
         }
      }
      if (bad || !init || !latch)
         continue;
      unsigned init_c = 0;
      if (!match_global_id_comp(init, wg, &init_c) || init_c != comp)
         continue;
      nir_def *ls = strip_mov(latch);
      if (!ls || nir_def_instr(ls)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *ladd = nir_instr_as_alu(nir_def_instr(ls));
      if (ladd->op != nir_op_iadd)
         continue;
      bool step_ok = false;
      for (unsigned side = 0; side < 2; side++) {
         if (strip_mov(ladd->src[side].src.ssa) == &p->def &&
             match_scaled_sysval_comp(&ladd->src[1 - side],
                                      nir_intrinsic_load_num_workgroups, wg, comp)) {
            step_ok = true;
            break;
         }
      }
      if (!step_ok)
         continue;
      if (phi)
         return false;
      phi = p;
   }
   if (!phi)
      return false;

   nir_if *exit_if = NULL;
   foreach_list_typed(nir_cf_node, n, node, &loop->body) {
      if (n->type == nir_cf_node_if) {
         exit_if = nir_cf_node_as_if(n);
         break;
      }
      if (n->type != nir_cf_node_block)
         return false;
   }
   if (!exit_if)
      return false;

   bool exit_on_true = false;
   if (cf_list_is_single_break_co(&exit_if->then_list) &&
       cf_list_is_empty_block(&exit_if->else_list)) {
      exit_on_true = true;
   } else if (cf_list_is_single_break_co(&exit_if->else_list) &&
              cf_list_is_empty_block(&exit_if->then_list)) {
      exit_on_true = false;
   } else {
      return false;
   }

   struct bool_term terms[2];
   unsigned nt = 0;
   if (!collect_and_terms(exit_if->condition.ssa, !exit_on_true, terms, &nt, 2) ||
       nt != 1)
      return false;

   nir_def *lhs = NULL, *rhs = NULL;
   bool is_lt = false, is_u = false;
   if (!decompose_cmp_term(terms[0], &lhs, &rhs, &is_lt, &is_u))
      return false;

   uint16_t lim_off = 0;
   if (is_lt || strip_mov(lhs) != &phi->def ||
       !match_push_const_u32(rhs, &lim_off))
      return false;

   *coord_out = &phi->def;
   *limit_pc_out = lim_off;
   *exit_if_out = exit_if;
   return true;
}

struct resize_lin {
   int32_t ax, ay, adw, adh, ab, a1;
};

static bool
eval_resize_lin(nir_def *def, nir_def *x, nir_def *y, uint16_t dw_pc,
                uint16_t dh_pc, uint16_t b_pc, struct resize_lin *out,
                unsigned depth)
{
   if (depth > 16 || !def || def->bit_size != 32 || def->num_components != 1)
      return false;
   def = strip_mov(def);
   if (!def)
      return false;

   memset(out, 0, sizeof(*out));
   if (def == x) {
      out->ax = 1;
      return true;
   }
   if (def == y) {
      out->ay = 1;
      return true;
   }

   int32_t c = 0;
   if (match_const_i32(def, &c)) {
      out->a1 = c;
      return true;
   }

   uint16_t pc = 0;
   if (match_push_const_u32(def, &pc)) {
      if (pc == dw_pc) {
         out->adw = 1;
         return true;
      }
      if (pc == dh_pc) {
         out->adh = 1;
         return true;
      }
      if (pc == b_pc) {
         out->ab = 1;
         return true;
      }
      return false;
   }

   if (nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));

   if (alu->op == nir_op_ineg) {
      struct resize_lin a;
      if (!eval_resize_lin(alu->src[0].src.ssa, x, y, dw_pc, dh_pc, b_pc, &a,
                           depth + 1))
         return false;
      out->ax = -a.ax;
      out->ay = -a.ay;
      out->adw = -a.adw;
      out->adh = -a.adh;
      out->ab = -a.ab;
      out->a1 = -a.a1;
      return true;
   }
   if (alu->op == nir_op_iadd || alu->op == nir_op_isub) {
      struct resize_lin a, b;
      if (!eval_resize_lin(alu->src[0].src.ssa, x, y, dw_pc, dh_pc, b_pc, &a,
                           depth + 1) ||
          !eval_resize_lin(alu->src[1].src.ssa, x, y, dw_pc, dh_pc, b_pc, &b,
                           depth + 1))
         return false;
      int32_t sgn = (alu->op == nir_op_iadd) ? 1 : -1;
      out->ax = a.ax + sgn * b.ax;
      out->ay = a.ay + sgn * b.ay;
      out->adw = a.adw + sgn * b.adw;
      out->adh = a.adh + sgn * b.adh;
      out->ab = a.ab + sgn * b.ab;
      out->a1 = a.a1 + sgn * b.a1;
      return true;
   }
   if (alu->op == nir_op_ishl) {
      int32_t sh = 0;
      struct resize_lin a;
      if (!match_const_i32(alu->src[1].src.ssa, &sh) || sh < 0 || sh > 4 ||
          !eval_resize_lin(alu->src[0].src.ssa, x, y, dw_pc, dh_pc, b_pc, &a,
                           depth + 1))
         return false;
      int32_t k = 1 << sh;
      out->ax = a.ax * k;
      out->ay = a.ay * k;
      out->adw = a.adw * k;
      out->adh = a.adh * k;
      out->ab = a.ab * k;
      out->a1 = a.a1 * k;
      return true;
   }
   if (alu->op == nir_op_imul || alu->op == nir_op_amul) {
      int32_t k = 0;
      nir_def *other = NULL;
      if (match_const_i32(alu->src[0].src.ssa, &k))
         other = alu->src[1].src.ssa;
      else if (match_const_i32(alu->src[1].src.ssa, &k))
         other = alu->src[0].src.ssa;
      else
         return false;
      struct resize_lin a;
      if (!eval_resize_lin(other, x, y, dw_pc, dh_pc, b_pc, &a, depth + 1))
         return false;
      out->ax = a.ax * k;
      out->ay = a.ay * k;
      out->adw = a.adw * k;
      out->adh = a.adh * k;
      out->ab = a.ab * k;
      out->a1 = a.a1 * k;
      return true;
   }
   return false;
}

static bool
match_dst_row_col_idx(nir_def *idx, uint16_t ds_pc, nir_def **row_out,
                      nir_def **col_out)
{
   idx = strip_mov(idx);
   if (!idx || nir_def_instr(idx)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(idx));
   if (add->op != nir_op_iadd)
      return false;
   for (unsigned side = 0; side < 2; side++) {
      nir_def *prod = strip_mov(add->src[side].src.ssa);
      if (!prod || nir_def_instr(prod)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(prod));
      if (mul->op != nir_op_imul && mul->op != nir_op_amul)
         continue;
      for (unsigned ms = 0; ms < 2; ms++) {
         uint16_t pc = 0;
         if (match_push_const_u32(mul->src[ms].src.ssa, &pc) && pc == ds_pc) {
            *row_out = mul->src[1 - ms].src.ssa;
            *col_out = add->src[1 - side].src.ssa;
            return true;
         }
      }
   }
   return false;
}

/* Classify a coordinate expression relative to x/y, dst_w/dst_h, border:
 *   0 = primary (`coord + border`), 1 = low mirror (`border - coord`),
 *   2 = high mirror (`2 * dim + border - 2 - coord`), -1 = none. */
static int
classify_mirror_coord(nir_def *def, bool is_y, nir_def *x, nir_def *y,
                      uint16_t dw_pc, uint16_t dh_pc, uint16_t b_pc)
{
   struct resize_lin l;
   if (!eval_resize_lin(def, x, y, dw_pc, dh_pc, b_pc, &l, 0))
      return -1;
   int32_t c_main = is_y ? l.ay : l.ax;
   int32_t c_other = is_y ? l.ax : l.ay;
   int32_t d_main = is_y ? l.adh : l.adw;
   int32_t d_other = is_y ? l.adw : l.adh;
   if (c_other != 0 || d_other != 0 || l.ab != 1)
      return -1;
   if (c_main == 1 && d_main == 0 && l.a1 == 0)
      return 0;
   if (c_main == -1 && d_main == 0 && l.a1 == 0)
      return 1;
   if (c_main == -1 && d_main == 2 && l.a1 == -2)
      return 2;
   return -1;
}

/* Match border guard predicates:
 *   0x01 = x_lft_nz (`x != 0 && x <= border`)
 *   0x02 = x_rgt    (`dst_w - 2 - border < x && x < dst_w - 1`)
 *   0x04 = y_top_nz (`y != 0 && y <= border`)
 *   0x08 = y_bot    (`dst_h - 2 - border < y && y < dst_h - 1`)
 *   0x10 = y_any    (`y <= border || (dst_h - 2 - border < y && y < dst_h - 1)`)
 *   0x20 = any_bdr  (`x_lft || x_rgt || y_top || y_bot`)
 */
static bool
match_coord_le_border(struct bool_term t, nir_def *coord, uint16_t b_pc)
{
   nir_def *lhs = NULL, *rhs = NULL;
   bool is_lt = false, is_u = false;
   if (!decompose_cmp_term(t, &lhs, &rhs, &is_lt, &is_u))
      return false;
   uint16_t pc = 0;
   return !is_lt && match_push_const_u32(lhs, &pc) && pc == b_pc &&
          strip_mov(rhs) == coord;
}

static bool
match_coord_ne_zero(struct bool_term t, nir_def *coord)
{
   if (nir_def_instr(t.def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *a = nir_instr_as_alu(nir_def_instr(t.def));
   bool is_ne = (a->op == nir_op_ine && !t.negated) ||
                (a->op == nir_op_ieq && t.negated);
   if (is_ne) {
      return (strip_mov(a->src[0].src.ssa) == coord &&
              is_const_i32(a->src[1].src.ssa, 0)) ||
             (strip_mov(a->src[1].src.ssa) == coord &&
              is_const_i32(a->src[0].src.ssa, 0));
   }
   nir_def *lhs = NULL, *rhs = NULL;
   bool is_lt = false, is_u = false;
   if (decompose_cmp_term(t, &lhs, &rhs, &is_lt, &is_u)) {
      if (is_lt && is_const_i32(lhs, 0) && strip_mov(rhs) == coord)
         return true;
      if (!is_lt && strip_mov(lhs) == coord && is_const_i32(rhs, 1))
         return true;
   }
   return false;
}

static bool
match_coord_hi_band(nir_def *cond, bool is_y, nir_def *x, nir_def *y,
                    uint16_t dw_pc, uint16_t dh_pc, uint16_t b_pc)
{
   struct bool_term terms[4];
   unsigned nt = 0;
   if (!collect_and_terms(cond, false, terms, &nt, 4) || nt != 2)
      return false;
   nir_def *coord = is_y ? y : x;
   bool has_lo = false, has_hi = false;
   for (unsigned i = 0; i < 2; i++) {
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      if (!decompose_cmp_term(terms[i], &lhs, &rhs, &is_lt, &is_u) || !is_lt)
         return false;
      if (strip_mov(lhs) == coord) {
         /* `coord < dim - 1` */
         struct resize_lin l;
         if (!eval_resize_lin(rhs, x, y, dw_pc, dh_pc, b_pc, &l, 0))
            return false;
         int32_t dm = is_y ? l.adh : l.adw;
         int32_t do_ = is_y ? l.adw : l.adh;
         if (l.ax == 0 && l.ay == 0 && l.ab == 0 && dm == 1 && do_ == 0 &&
             l.a1 == -1)
            has_hi = true;
      } else if (strip_mov(rhs) == coord) {
         /* `dim - 2 - border < coord` */
         struct resize_lin l;
         if (!eval_resize_lin(lhs, x, y, dw_pc, dh_pc, b_pc, &l, 0))
            return false;
         int32_t dm = is_y ? l.adh : l.adw;
         int32_t do_ = is_y ? l.adw : l.adh;
         if (l.ax == 0 && l.ay == 0 && l.ab == -1 && dm == 1 && do_ == 0 &&
             l.a1 == -2)
            has_lo = true;
      }
   }
   return has_lo && has_hi;
}

static bool
match_coord_lo_nz(nir_def *cond, nir_def *coord, uint16_t b_pc)
{
   struct bool_term terms[4];
   unsigned nt = 0;
   if (!collect_and_terms(cond, false, terms, &nt, 4) || nt != 2)
      return false;
   bool has_le = false, has_nz = false;
   for (unsigned i = 0; i < 2; i++) {
      if (match_coord_le_border(terms[i], coord, b_pc))
         has_le = true;
      else if (match_coord_ne_zero(terms[i], coord))
         has_nz = true;
   }
   return has_le && has_nz;
}

static bool
match_y_any_cond(nir_def *cond, nir_def *x, nir_def *y, uint16_t dw_pc,
                 uint16_t dh_pc, uint16_t b_pc)
{
   /* `y <= border || y_bot` => negated is conjunction of `!(y <= border)` and `!y_bot`. */
   struct bool_term terms[4];
   unsigned nt = 0;
   if (!collect_and_terms(cond, true, terms, &nt, 4) || nt != 2)
      return false;
   bool has_top = false, has_bot = false;
   for (unsigned i = 0; i < 2; i++) {
      struct bool_term flipped = {terms[i].def, !terms[i].negated};
      if (match_coord_le_border(flipped, y, b_pc))
         has_top = true;
      else if (!flipped.negated &&
               match_coord_hi_band(flipped.def, true, x, y, dw_pc, dh_pc, b_pc))
         has_bot = true;
   }
   return has_top && has_bot;
}

static bool
match_umin_xy(nir_def *def, nir_def *x, nir_def *y)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != nir_op_umin && alu->op != nir_op_imin)
      return false;
   nir_def *a = strip_mov(alu->src[0].src.ssa);
   nir_def *b = strip_mov(alu->src[1].src.ssa);
   return (a == x && b == y) || (a == y && b == x);
}

static bool
match_any_border_cond(nir_def *cond, nir_def *x, nir_def *y, uint16_t dw_pc,
                      uint16_t dh_pc, uint16_t b_pc)
{
   struct bool_term terms[8];
   unsigned nt = 0;
   if (!collect_and_terms(cond, true, terms, &nt, 8) || (nt != 3 && nt != 4))
      return false;
   bool xl = false, xr = false, yt = false, yb = false;
   for (unsigned i = 0; i < nt; i++) {
      struct bool_term flipped = {terms[i].def, !terms[i].negated};
      nir_def *lhs = NULL, *rhs = NULL;
      bool is_lt = false, is_u = false;
      uint16_t pc = 0;
      if (decompose_cmp_term(flipped, &lhs, &rhs, &is_lt, &is_u) && !is_lt &&
          match_push_const_u32(lhs, &pc) && pc == b_pc &&
          match_umin_xy(rhs, x, y)) {
         xl = true;
         yt = true;
      } else if (match_coord_le_border(flipped, x, b_pc))
         xl = true;
      else if (match_coord_le_border(flipped, y, b_pc))
         yt = true;
      else if (!flipped.negated &&
               match_coord_hi_band(flipped.def, false, x, y, dw_pc, dh_pc, b_pc))
         xr = true;
      else if (!flipped.negated &&
               match_coord_hi_band(flipped.def, true, x, y, dw_pc, dh_pc, b_pc))
         yb = true;
      else
         return false;
   }
   return xl && xr && yt && yb;
}

struct mirror_walk_ctx {
   nir_def *x, *y, *val_u8;
   uint16_t dw_pc, dh_pc, ds_pc, b_pc;
   uint32_t dst_set, dst_bind;
   uint8_t seen_mask;
   nir_intrinsic_instr *stores[8];
};

static bool
match_mirror_guard_cond(nir_def *c, struct mirror_walk_ctx *ctx,
                        unsigned *guard_out)
{
   if (match_coord_lo_nz(c, ctx->x, ctx->b_pc))
      *guard_out = 0x1;
   else if (match_coord_hi_band(c, false, ctx->x, ctx->y, ctx->dw_pc,
                                ctx->dh_pc, ctx->b_pc))
      *guard_out = 0x2;
   else if (match_coord_lo_nz(c, ctx->y, ctx->b_pc))
      *guard_out = 0x4;
   else if (match_coord_hi_band(c, true, ctx->x, ctx->y, ctx->dw_pc,
                                ctx->dh_pc, ctx->b_pc))
      *guard_out = 0x8;
   else if (match_y_any_cond(c, ctx->x, ctx->y, ctx->dw_pc, ctx->dh_pc,
                             ctx->b_pc))
      *guard_out = 0x10;
   else
      return false;
   return true;
}

static bool
walk_mirror_border_list(struct exec_list *list, unsigned active_guards,
                        struct mirror_walk_ctx *ctx, unsigned depth)
{
   if (depth > 8)
      return false;

   foreach_list_typed(nir_cf_node, n, node, list) {
      if (n->type == nir_cf_node_block) {
         nir_block *blk = nir_cf_node_as_block(n);
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_deref ||
                intr->intrinsic == nir_intrinsic_deref_atomic)
               return false;
            if (intr->intrinsic != nir_intrinsic_store_deref)
               continue;
            if (nir_intrinsic_write_mask(intr) != 1 ||
                strip_mov(intr->src[1].ssa) != ctx->val_u8 ||
                (nir_intrinsic_access(intr) &
                 (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL)))
               return false;
            uint32_t set = 0, bind = 0;
            nir_def *idx = NULL;
            if (!match_ssbo_array_deref(nir_src_as_deref(intr->src[0]), 1, &set,
                                        &bind, &idx) ||
                set != ctx->dst_set || bind != ctx->dst_bind)
               return false;
            nir_def *row = NULL, *col = NULL;
            if (!match_dst_row_col_idx(idx, ctx->ds_pc, &row, &col))
               return false;
            int ry = classify_mirror_coord(row, true, ctx->x, ctx->y,
                                           ctx->dw_pc, ctx->dh_pc, ctx->b_pc);
            int rx = classify_mirror_coord(col, false, ctx->x, ctx->y,
                                           ctx->dw_pc, ctx->dh_pc, ctx->b_pc);
            if (ry < 0 || rx < 0 || (ry == 0 && rx == 0))
               return false;

            /* Map (ry, rx) to slot 0..7 and verify exact active guard bits:
             *   bit 0: x_lft_nz, bit 1: x_rgt, bit 2: y_top_nz, bit 3: y_bot */
            unsigned want_guard = 0;
            if (rx == 1)
               want_guard |= 0x1;
            else if (rx == 2)
               want_guard |= 0x2;
            if (ry == 1)
               want_guard |= 0x4;
            else if (ry == 2)
               want_guard |= 0x8;
            if ((active_guards & 0xf) != want_guard)
               return false;

            unsigned slot = (unsigned)(ry * 3 + rx) - 1;
            if (slot >= 8 || (ctx->seen_mask & (1u << slot)))
               return false;
            ctx->seen_mask |= (uint8_t)(1u << slot);
            ctx->stores[slot] = intr;
         }
      } else if (n->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(n);
         struct exec_list *body = NULL;
         if (cf_list_is_empty_block(&nif->else_list))
            body = &nif->then_list;
         else
            return false;

         unsigned add_guard = 0;
         if (!match_mirror_guard_cond(nif->condition.ssa, ctx, &add_guard))
            return false;

         if (!walk_mirror_border_list(body, active_guards | add_guard, ctx,
                                      depth + 1))
            return false;
      } else {
         return false;
      }
   }
   return true;
}

/* Match `float(src_dim) / float(dst_dim)` or `float(src_dim) * frcp(float(dst_dim))`
 * strictly without non-default fp_math_ctrl. For non-internal shaders, AGX lowers
 * relaxed `nir_op_fdiv` to `fmul(num, frcp(den))`, matching the expression emitted
 * by build_bilinear_resize_border_u8_kernel. */
static bool
match_pc_f32_ratio(nir_def *def, uint16_t *num_pc_out, uint16_t *den_pc_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->fp_math_ctrl != 0)
      return false;

   nir_def *num = NULL, *den = NULL;
   if (alu->op == nir_op_fdiv) {
      num = strip_mov(alu->src[0].src.ssa);
      den = strip_mov(alu->src[1].src.ssa);
   } else if (alu->op == nir_op_fmul) {
      for (unsigned side = 0; side < 2; side++) {
         nir_def *r = strip_mov(alu->src[side].src.ssa);
         if (r && nir_def_instr(r)->type == nir_instr_type_alu) {
            nir_alu_instr *ra = nir_instr_as_alu(nir_def_instr(r));
            if (ra->op == nir_op_frcp && ra->fp_math_ctrl == 0) {
               den = strip_mov(ra->src[0].src.ssa);
               num = strip_mov(alu->src[1 - side].src.ssa);
               break;
            }
         }
      }
   } else {
      return false;
   }
   if (!num || !den || nir_def_instr(num)->type != nir_instr_type_alu ||
       nir_def_instr(den)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *na = nir_instr_as_alu(nir_def_instr(num));
   nir_alu_instr *da = nir_instr_as_alu(nir_def_instr(den));
   if ((na->op != nir_op_i2f32 && na->op != nir_op_u2f32) ||
       (da->op != nir_op_i2f32 && da->op != nir_op_u2f32))
      return false;
   return match_push_const_u32(na->src[0].src.ssa, num_pc_out) &&
          match_push_const_u32(da->src[0].src.ssa, den_pc_out);
}

/* Match `s = float(coord) * (float(src_dim) * frcp(float(dst_dim)))` strictly
 * so the emitted scale expression reproduces the matched float32 rounding. */
static bool
match_scaled_coord_f32(nir_def *def, nir_def *coord, uint16_t want_dst_pc,
                       uint16_t *src_dim_pc_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != nir_op_fmul || alu->fp_math_ctrl != 0)
      return false;

   for (unsigned side = 0; side < 2; side++) {
      nir_def *cf = strip_mov(alu->src[side].src.ssa);
      if (!cf || nir_def_instr(cf)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *ca = nir_instr_as_alu(nir_def_instr(cf));
      if ((ca->op == nir_op_u2f32 || ca->op == nir_op_i2f32) &&
          strip_mov(ca->src[0].src.ssa) == coord) {
         uint16_t s_pc = 0, d_pc = 0;
         if (match_pc_f32_ratio(alu->src[1 - side].src.ssa, &s_pc, &d_pc) &&
             d_pc == want_dst_pc) {
            *src_dim_pc_out = s_pc;
            return true;
         }
      }
   }
   return false;
}

static bool
match_int_part_f32(nir_def *tr, nir_def *s_def)
{
   tr = strip_mov(tr);
   if (!tr || nir_def_instr(tr)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *ta = nir_instr_as_alu(nir_def_instr(tr));
   if ((ta->op == nir_op_ftrunc || ta->op == nir_op_ffloor) &&
       strip_mov(ta->src[0].src.ssa) == s_def)
      return true;
   if (ta->op == nir_op_i2f32 || ta->op == nir_op_u2f32) {
      nir_def *iv = strip_mov(ta->src[0].src.ssa);
      if (iv && nir_def_instr(iv)->type == nir_instr_type_alu) {
         nir_alu_instr *ia = nir_instr_as_alu(nir_def_instr(iv));
         if ((ia->op == nir_op_f2i32 || ia->op == nir_op_f2u32) &&
             strip_mov(ia->src[0].src.ssa) == s_def)
            return true;
      }
   }
   return false;
}

static bool
match_ftrunc_frac(nir_def *def, nir_def *s_def)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->fp_math_ctrl != 0)
      return false;
   if (alu->op == nir_op_ffract && strip_mov(alu->src[0].src.ssa) == s_def)
      return true;
   if (alu->op == nir_op_fsub && strip_mov(alu->src[0].src.ssa) == s_def)
      return match_int_part_f32(alu->src[1].src.ssa, s_def);
   if (alu->op == nir_op_fadd) {
      for (unsigned side = 0; side < 2; side++) {
         if (strip_mov(alu->src[side].src.ssa) != s_def)
            continue;
         nir_def *neg = strip_mov(alu->src[1 - side].src.ssa);
         if (!neg || nir_def_instr(neg)->type != nir_instr_type_alu ||
             nir_instr_as_alu(nir_def_instr(neg))->op != nir_op_fneg)
            continue;
         if (match_int_part_f32(
                nir_instr_as_alu(nir_def_instr(neg))->src[0].src.ssa, s_def))
            return true;
      }
   }
   return false;
}

/* Match `clamp(f2i32(s) + add_imm, 0, src_dim - 1)`. */
static bool
match_clamped_src_idx(nir_def *def, nir_def *s_def, int32_t add_imm,
                      uint16_t src_dim_pc)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *outer = nir_instr_as_alu(nir_def_instr(def));
   nir_def *mid = NULL;
   bool saw_min = false, saw_max = false;

   for (unsigned step = 0; step < 2; step++) {
      nir_alu_instr *a = step == 0 ? outer
                                   : (mid && nir_def_instr(mid)->type == nir_instr_type_alu
                                         ? nir_instr_as_alu(nir_def_instr(mid))
                                         : NULL);
      if (!a)
         return false;
      if (a->op == nir_op_imin && !saw_min) {
         bool ok = false;
         for (unsigned side = 0; side < 2; side++) {
            nir_def *lim = strip_mov(a->src[side].src.ssa);
            if (!lim || nir_def_instr(lim)->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *la = nir_instr_as_alu(nir_def_instr(lim));
            uint16_t pc = 0;
            bool m1 =
               (la->op == nir_op_iadd &&
                ((match_push_const_u32(la->src[0].src.ssa, &pc) &&
                  pc == src_dim_pc && is_const_i32(la->src[1].src.ssa, -1)) ||
                 (match_push_const_u32(la->src[1].src.ssa, &pc) &&
                  pc == src_dim_pc && is_const_i32(la->src[0].src.ssa, -1)))) ||
               (la->op == nir_op_isub &&
                match_push_const_u32(la->src[0].src.ssa, &pc) &&
                pc == src_dim_pc && is_const_i32(la->src[1].src.ssa, 1));
            if (m1) {
               mid = strip_mov(a->src[1 - side].src.ssa);
               saw_min = true;
               ok = true;
               break;
            }
         }
         if (!ok)
            return false;
      } else if (a->op == nir_op_imax && !saw_max) {
         if (is_const_i32(a->src[0].src.ssa, 0))
            mid = strip_mov(a->src[1].src.ssa);
         else if (is_const_i32(a->src[1].src.ssa, 0))
            mid = strip_mov(a->src[0].src.ssa);
         else
            return false;
         saw_max = true;
      } else {
         return false;
      }
   }

   if (!saw_min || !saw_max || !mid ||
       nir_def_instr(mid)->type != nir_instr_type_alu)
      return false;
   if (add_imm != 0) {
      nir_alu_instr *ad = nir_instr_as_alu(nir_def_instr(mid));
      if (ad->op != nir_op_iadd)
         return false;
      if (is_const_i32(ad->src[1].src.ssa, add_imm))
         mid = strip_mov(ad->src[0].src.ssa);
      else if (is_const_i32(ad->src[0].src.ssa, add_imm))
         mid = strip_mov(ad->src[1].src.ssa);
      else
         return false;
      if (!mid || nir_def_instr(mid)->type != nir_instr_type_alu)
         return false;
   }
   nir_alu_instr *fi = nir_instr_as_alu(nir_def_instr(mid));
   return (fi->op == nir_op_f2i32 || fi->op == nir_op_f2u32) &&
          strip_mov(fi->src[0].src.ssa) == s_def;
}

static bool
match_one_minus_f32(nir_def *def, nir_def **t_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *a = nir_instr_as_alu(nir_def_instr(def));
   if (a->fp_math_ctrl != 0)
      return false;
   uint32_t bits = 0;
   if (a->op == nir_op_fsub && match_const_f32_bits(a->src[0].src.ssa, &bits) &&
       bits == 0x3f800000u) {
      *t_out = strip_mov(a->src[1].src.ssa);
      return true;
   }
   if (a->op == nir_op_fadd) {
      for (unsigned side = 0; side < 2; side++) {
         if (!match_const_f32_bits(a->src[side].src.ssa, &bits) ||
             bits != 0x3f800000u)
            continue;
         nir_def *neg = strip_mov(a->src[1 - side].src.ssa);
         if (neg && nir_def_instr(neg)->type == nir_instr_type_alu &&
             nir_instr_as_alu(nir_def_instr(neg))->op == nir_op_fneg) {
            *t_out = strip_mov(
               nir_instr_as_alu(nir_def_instr(neg))->src[0].src.ssa);
            return true;
         }
      }
   }
   return false;
}

static bool
match_lerp_f32(nir_def *def, nir_def **a_out, nir_def **b_out, nir_def **t_out,
               enum hk_resize_lerp_mode *mode_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->fp_math_ctrl != 0)
      return false;

   /* Match `a * (1.0 - t) + b * t` (as `fadd(fmul, fmul)` or `ffma`),
    * recording the exact fusion form so the emitter reproduces rounding. */
   if (alu->op == nir_op_fadd || alu->op == nir_op_ffma) {
      nir_def *p0a = NULL, *p0b = NULL, *p1a = NULL, *p1b = NULL;
      if (alu->op == nir_op_ffma) {
         p0a = strip_mov(alu->src[0].src.ssa);
         p0b = strip_mov(alu->src[1].src.ssa);
         nir_def *c = strip_mov(alu->src[2].src.ssa);
         if (c && nir_def_instr(c)->type == nir_instr_type_alu) {
            nir_alu_instr *ca = nir_instr_as_alu(nir_def_instr(c));
            if (ca->op == nir_op_fmul && ca->fp_math_ctrl == 0) {
               p1a = strip_mov(ca->src[0].src.ssa);
               p1b = strip_mov(ca->src[1].src.ssa);
            }
         }
      } else {
         nir_def *s0 = strip_mov(alu->src[0].src.ssa);
         nir_def *s1 = strip_mov(alu->src[1].src.ssa);
         if (s0 && s1 && nir_def_instr(s0)->type == nir_instr_type_alu &&
             nir_def_instr(s1)->type == nir_instr_type_alu) {
            nir_alu_instr *m0 = nir_instr_as_alu(nir_def_instr(s0));
            nir_alu_instr *m1 = nir_instr_as_alu(nir_def_instr(s1));
            if (m0->op == nir_op_fmul && m1->op == nir_op_fmul &&
                m0->fp_math_ctrl == 0 && m1->fp_math_ctrl == 0) {
               p0a = strip_mov(m0->src[0].src.ssa);
               p0b = strip_mov(m0->src[1].src.ssa);
               p1a = strip_mov(m1->src[0].src.ssa);
               p1b = strip_mov(m1->src[1].src.ssa);
            }
         }
      }
      if (p0a && p0b && p1a && p1b) {
         nir_def *pairs[2][2] = {{p0a, p0b}, {p1a, p1b}};
         for (unsigned which = 0; which < 2; which++) {
            for (unsigned side = 0; side < 2; side++) {
               nir_def *t = NULL;
               if (match_one_minus_f32(pairs[which][side], &t)) {
                  nir_def *a_cand = pairs[which][1 - side];
                  enum hk_resize_lerp_mode m =
                     alu->op == nir_op_fadd
                        ? (which == 0 ? HK_RESIZE_LERP_ONE_MINUS_ADD_A
                                      : HK_RESIZE_LERP_ONE_MINUS_ADD_B)
                        : (which == 0 ? HK_RESIZE_LERP_ONE_MINUS_FFMA_A
                                      : HK_RESIZE_LERP_ONE_MINUS_FFMA_B);
                  if (pairs[1 - which][0] == t) {
                     *a_out = a_cand;
                     *b_out = pairs[1 - which][1];
                     *t_out = t;
                     *mode_out = m;
                     return true;
                  }
                  if (pairs[1 - which][1] == t) {
                     *a_out = a_cand;
                     *b_out = pairs[1 - which][0];
                     *t_out = t;
                     *mode_out = m;
                     return true;
                  }
               }
            }
         }
      }
   }

   nir_def *mul_a = NULL, *mul_b = NULL, *base = NULL;
   if (alu->op == nir_op_ffma) {
      mul_a = strip_mov(alu->src[0].src.ssa);
      mul_b = strip_mov(alu->src[1].src.ssa);
      base = strip_mov(alu->src[2].src.ssa);
   } else if (alu->op == nir_op_fadd) {
      for (unsigned side = 0; side < 2; side++) {
         nir_def *m = strip_mov(alu->src[side].src.ssa);
         if (m && nir_def_instr(m)->type == nir_instr_type_alu) {
            nir_alu_instr *ma = nir_instr_as_alu(nir_def_instr(m));
            if (ma->op == nir_op_fmul && ma->fp_math_ctrl == 0) {
               mul_a = strip_mov(ma->src[0].src.ssa);
               mul_b = strip_mov(ma->src[1].src.ssa);
               base = strip_mov(alu->src[1 - side].src.ssa);
               break;
            }
         }
      }
   }
   if (!mul_a || !mul_b || !base)
      return false;

   /* One of mul_a/mul_b must be `b - base` (`fsub(b, base)` or `fadd(b, fneg(base))`). */
   const enum hk_resize_lerp_mode diff_mode =
      alu->op == nir_op_ffma ? HK_RESIZE_LERP_DIFF_FFMA : HK_RESIZE_LERP_DIFF_ADD;
   nir_def *mands[2] = {mul_a, mul_b};
   for (unsigned side = 0; side < 2; side++) {
      nir_def *d = mands[side];
      if (!d || nir_def_instr(d)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *da = nir_instr_as_alu(nir_def_instr(d));
      if (da->fp_math_ctrl != 0)
         continue;
      if (da->op == nir_op_fsub && strip_mov(da->src[1].src.ssa) == base) {
         *a_out = base;
         *b_out = strip_mov(da->src[0].src.ssa);
         *t_out = mands[1 - side];
         *mode_out = diff_mode;
         return true;
      }
      if (da->op == nir_op_fadd) {
         for (unsigned ds = 0; ds < 2; ds++) {
            nir_def *neg = strip_mov(da->src[ds].src.ssa);
            if (neg && nir_def_instr(neg)->type == nir_instr_type_alu &&
                nir_instr_as_alu(nir_def_instr(neg))->op == nir_op_fneg &&
                strip_mov(nir_instr_as_alu(nir_def_instr(neg))->src[0].src.ssa) ==
                   base) {
               *a_out = base;
               *b_out = strip_mov(da->src[1 - ds].src.ssa);
               *t_out = mands[1 - side];
               *mode_out = diff_mode;
               return true;
            }
         }
      }
   }
   return false;
}

static bool
match_u2f32_load_deref(nir_def *def, nir_intrinsic_instr **load_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != nir_op_u2f32 && alu->op != nir_op_i2f32)
      return false;
   nir_def *s = strip_mov(alu->src[0].src.ssa);
   if (s && nir_def_instr(s)->type == nir_instr_type_alu) {
      nir_alu_instr *ext = nir_instr_as_alu(nir_def_instr(s));
      if (ext->op == nir_op_u2u32 || ext->op == nir_op_u2u16)
         s = strip_mov(ext->src[0].src.ssa);
   }
   if (!s || s->bit_size != 8 || nir_def_instr(s)->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(s));
   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;
   *load_out = intr;
   return true;
}

static bool
match_resize_top_cond(nir_def *cond, bool *resize_on_true_out,
                      uint16_t dw_pc, uint16_t dh_pc, uint16_t sw_pc,
                      uint16_t sh_pc)
{
   for (unsigned try_true = 0; try_true < 2; try_true++) {
      /* Resize holds when `sw != dw || sh != dh`, so negated is `sw == dw && sh == dh`. */
      struct bool_term terms[4];
      unsigned nt = 0;
      if (!collect_and_terms(cond, try_true == 0, terms, &nt, 4) || nt != 2)
         continue;
      bool got_w = false, got_h = false;
      for (unsigned i = 0; i < 2; i++) {
         if (nir_def_instr(terms[i].def)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *a = nir_instr_as_alu(nir_def_instr(terms[i].def));
         bool is_eq = (a->op == nir_op_ieq && !terms[i].negated) ||
                      (a->op == nir_op_ine && terms[i].negated);
         if (!is_eq)
            continue;
         uint16_t p0 = 0, p1 = 0;
         if (!match_push_const_u32(a->src[0].src.ssa, &p0) ||
             !match_push_const_u32(a->src[1].src.ssa, &p1))
            continue;
         if ((p0 == sw_pc && p1 == dw_pc) || (p0 == dw_pc && p1 == sw_pc))
            got_w = true;
         else if ((p0 == sh_pc && p1 == dh_pc) || (p0 == dh_pc && p1 == sh_pc))
            got_h = true;
      }
      if (got_w && got_h) {
         *resize_on_true_out = (try_true == 0);
         return true;
      }
   }
   return false;
}

/* Match un-unswitched form: `y_loop -> x_loop -> if (do_resize) -> phi ->
 * primary store + mirrored border subtree` (before backend loop unswitching). */
static bool
match_bilinear_resize_border_inloop_u8(
   nir_shader *s, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts,
   struct hk_compute_opt_plan *plan)
{
   const uint16_t wg[3] = {
      (uint16_t)plan->orig_wg_size[0],
      (uint16_t)plan->orig_wg_size[1],
      (uint16_t)plan->orig_wg_size[2],
   };
   nir_function_impl *impl = nir_shader_get_entrypoint(s);

   nir_loop *y_loop = NULL;
   foreach_list_typed(nir_cf_node, n, node, &impl->body) {
      if (n->type == nir_cf_node_loop) {
         if (y_loop)
            return false;
         y_loop = nir_cf_node_as_loop(n);
      } else if (n->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!y_loop)
      return false;

   nir_def *y = NULL, *x = NULL;
   uint16_t dh_pc = 0, dw_pc = 0;
   nir_if *y_ex = NULL, *x_ex = NULL;
   if (!match_grid_stride_loop_co(y_loop, 1, wg, &y, &dh_pc, &y_ex))
      return false;

   nir_loop *x_loop = NULL;
   foreach_list_typed(nir_cf_node, n, node, &y_loop->body) {
      if (n->type == nir_cf_node_if) {
         if (nir_cf_node_as_if(n) != y_ex)
            return false;
      } else if (n->type == nir_cf_node_loop) {
         if (x_loop)
            return false;
         x_loop = nir_cf_node_as_loop(n);
      } else if (n->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!x_loop || !match_grid_stride_loop_co(x_loop, 0, wg, &x, &dw_pc, &x_ex))
      return false;

   /* Inside `x_loop`: `x_ex`, then `mode_if` (resize vs copy), then either:
    * - Form A: merge phi -> single `prim_st` -> mirrored border `if`(s), or
    * - Form B: in-arm primary stores in `mode_if` -> dst reload (`reload_ld`)
    *   at primary index -> mirrored border `if`(s) (either wrapped in a single
    *   `any_border` `if` or as direct sibling border `if`s). */
   nir_if *mode_if = NULL;
   nir_if *bdr_ifs[5] = {NULL};
   unsigned bdr_if_cnt = 0;
   nir_intrinsic_instr *prim_st = NULL;
   nir_intrinsic_instr *reload_ld = NULL;
   foreach_list_typed(nir_cf_node, n, node, &x_loop->body) {
      if (n->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(n);
         if (nif == x_ex)
            continue;
         if (!mode_if) {
            mode_if = nif;
         } else {
            if (bdr_if_cnt >= 5 || !cf_list_is_empty_block(&nif->else_list))
               return false;
            bdr_ifs[bdr_if_cnt++] = nif;
         }
      } else if (n->type == nir_cf_node_block) {
         nir_block *blk = nir_cf_node_as_block(n);
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_deref) {
               if (prim_st || reload_ld || !mode_if || bdr_if_cnt > 0)
                  return false;
               prim_st = intr;
            } else if (intr->intrinsic == nir_intrinsic_load_deref) {
               if (prim_st || reload_ld || !mode_if || bdr_if_cnt > 0)
                  return false;
               reload_ld = intr;
            } else if (intr->intrinsic == nir_intrinsic_deref_atomic) {
               return false;
            }
         }
      } else {
         return false;
      }
   }
   if (!mode_if || bdr_if_cnt == 0 || (!prim_st && !reload_ld))
      return false;

   /* Count loads & stores in each arm of `mode_if` (4 = resize, 1 = copy). */
   struct exec_list *arms[2] = {&mode_if->then_list, &mode_if->else_list};
   unsigned arm_loads[2] = {0, 0};
   unsigned arm_stores[2] = {0, 0};
   nir_intrinsic_instr *arm_store[2] = {NULL, NULL};
   nir_intrinsic_instr *cpy_load = NULL;
   for (unsigned a = 0; a < 2; a++) {
      foreach_list_typed(nir_cf_node, n, node, arms[a]) {
         if (n->type != nir_cf_node_block)
            return false;
         nir_block *blk = nir_cf_node_as_block(n);
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_deref) {
               arm_loads[a]++;
            } else if (intr->intrinsic == nir_intrinsic_store_deref) {
               if (arm_stores[a] >= 1)
                  return false;
               arm_store[a] = intr;
               arm_stores[a]++;
            } else if (intr->intrinsic == nir_intrinsic_deref_atomic) {
               return false;
            }
         }
      }
   }

   unsigned res_arm = 0, cpy_arm = 1;
   if (arm_loads[0] == 4 && arm_loads[1] == 1) {
      res_arm = 0;
      cpy_arm = 1;
   } else if (arm_loads[0] == 1 && arm_loads[1] == 4) {
      res_arm = 1;
      cpy_arm = 0;
   } else {
      return false;
   }

   const bool inarm_store_form = (reload_ld != NULL);
   nir_intrinsic_instr *res_prim_st = NULL, *cpy_prim_st = NULL;
   nir_def *val_u8 = NULL, *res_val_u8 = NULL, *cpy_val_u8 = NULL;
   bool phi_is_u32 = false;

   if (inarm_store_form) {
      if (arm_stores[0] != 1 || arm_stores[1] != 1)
         return false;
      res_prim_st = arm_store[res_arm];
      cpy_prim_st = arm_store[cpy_arm];
      if (nir_intrinsic_write_mask(res_prim_st) != 1 ||
          nir_intrinsic_write_mask(cpy_prim_st) != 1)
         return false;
      res_val_u8 = strip_mov(res_prim_st->src[1].ssa);
      cpy_val_u8 = strip_mov(cpy_prim_st->src[1].ssa);
      val_u8 = &reload_ld->def;
      if (val_u8->bit_size != 8 || val_u8->num_components != 1)
         return false;
   } else {
      if (arm_stores[0] != 0 || arm_stores[1] != 0 ||
          nir_intrinsic_write_mask(prim_st) != 1)
         return false;
      /* Merge phi of `mode_if` produces `val_u8` (directly or via `u2u8(phi32)`). */
      val_u8 = strip_mov(prim_st->src[1].ssa);
      nir_def *phi_def = val_u8;
      if (phi_def && nir_def_instr(phi_def)->type == nir_instr_type_alu) {
         nir_alu_instr *pa = nir_instr_as_alu(nir_def_instr(phi_def));
         if (pa->op == nir_op_u2u8 || pa->op == nir_op_i2i8) {
            phi_def = strip_mov(pa->src[0].src.ssa);
            phi_is_u32 = true;
         }
      }
      if (!phi_def || nir_def_instr(phi_def)->type != nir_instr_type_phi)
         return false;
      nir_phi_instr *vphi = nir_instr_as_phi(nir_def_instr(phi_def));
      if (nir_cf_node_prev(&vphi->instr.block->cf_node) != &mode_if->cf_node)
         return false;

      nir_block *res_blk = res_arm == 0 ? nir_if_last_then_block(mode_if)
                                        : nir_if_last_else_block(mode_if);
      nir_block *cpy_blk = cpy_arm == 0 ? nir_if_last_then_block(mode_if)
                                        : nir_if_last_else_block(mode_if);
      nir_foreach_phi_src(ps, vphi) {
         if (ps->pred == res_blk)
            res_val_u8 = strip_mov(ps->src.ssa);
         else if (ps->pred == cpy_blk)
            cpy_val_u8 = strip_mov(ps->src.ssa);
      }
      if (phi_is_u32 && cpy_val_u8 &&
          nir_def_instr(cpy_val_u8)->type == nir_instr_type_alu) {
         nir_alu_instr *ca = nir_instr_as_alu(nir_def_instr(cpy_val_u8));
         if (ca->op == nir_op_u2u32 || ca->op == nir_op_u2u16)
            cpy_val_u8 = strip_mov(ca->src[0].src.ssa);
      }
   }
   if (!res_val_u8 || !cpy_val_u8 ||
       nir_def_instr(cpy_val_u8)->type != nir_instr_type_intrinsic)
      return false;
   cpy_load = nir_instr_as_intrinsic(nir_def_instr(cpy_val_u8));
   if (cpy_load->intrinsic != nir_intrinsic_load_deref)
      return false;

   nir_intrinsic_instr *dst_ref_st = inarm_store_form ? cpy_prim_st : prim_st;
   uint32_t src_set = 0, src_bind = 0, dst_set = 0, dst_bind = 0;
   nir_def *cl_idx = NULL, *ps_idx = NULL;
   if (!match_ssbo_array_deref(nir_src_as_deref(cpy_load->src[0]), 1, &src_set,
                               &src_bind, &cl_idx) ||
       !match_ssbo_array_deref(nir_src_as_deref(dst_ref_st->src[0]), 1,
                               &dst_set, &dst_bind, &ps_idx))
      return false;
   if (src_set == dst_set && src_bind == dst_bind)
      return false;
   if ((nir_intrinsic_access(cpy_load) | nir_intrinsic_access(dst_ref_st)) &
       (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL))
      return false;

   uint16_t ss_pc = 0;
   if (!match_linear_index(cl_idx, y, x, &ss_pc, wg))
      return false;

   /* Primary store index: `(y + border) * dst_stride + (x + border)`. */
   uint16_t ds_pc = 0, b_pc = 0;
   {
      nir_def *idx = strip_mov(ps_idx);
      if (!idx || nir_def_instr(idx)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(idx));
      if (add->op != nir_op_iadd)
         return false;
      bool found = false;
      for (unsigned side = 0; side < 2 && !found; side++) {
         nir_def *prod = strip_mov(add->src[side].src.ssa);
         nir_def *col = strip_mov(add->src[1 - side].src.ssa);
         if (!prod || !col || nir_def_instr(prod)->type != nir_instr_type_alu ||
             nir_def_instr(col)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(prod));
         nir_alu_instr *cadd = nir_instr_as_alu(nir_def_instr(col));
         if ((mul->op != nir_op_imul && mul->op != nir_op_amul) ||
             cadd->op != nir_op_iadd)
            continue;
         uint16_t bp = 0;
         if (strip_mov(cadd->src[0].src.ssa) == x &&
             match_push_const_u32(cadd->src[1].src.ssa, &bp)) {
         } else if (strip_mov(cadd->src[1].src.ssa) == x &&
                    match_push_const_u32(cadd->src[0].src.ssa, &bp)) {
         } else {
            continue;
         }
         for (unsigned ms = 0; ms < 2; ms++) {
            uint16_t sp = 0;
            nir_def *row = strip_mov(mul->src[1 - ms].src.ssa);
            if (!match_push_const_u32(mul->src[ms].src.ssa, &sp) || !row ||
                nir_def_instr(row)->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *radd = nir_instr_as_alu(nir_def_instr(row));
            uint16_t bp2 = 0;
            if (radd->op == nir_op_iadd &&
                ((strip_mov(radd->src[0].src.ssa) == y &&
                  match_push_const_u32(radd->src[1].src.ssa, &bp2) &&
                  bp2 == bp) ||
                 (strip_mov(radd->src[1].src.ssa) == y &&
                  match_push_const_u32(radd->src[0].src.ssa, &bp2) &&
                  bp2 == bp))) {
               ds_pc = sp;
               b_pc = bp;
               found = true;
               break;
            }
         }
      }
      if (!found)
         return false;
   }

   if (inarm_store_form) {
      if ((nir_intrinsic_access(res_prim_st) |
           nir_intrinsic_access(reload_ld)) &
          (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL))
         return false;
      nir_intrinsic_instr *chk[2] = {res_prim_st, reload_ld};
      for (unsigned i = 0; i < 2; i++) {
         uint32_t s_set = 0, s_bind = 0;
         nir_def *s_idx = NULL, *s_row = NULL, *s_col = NULL;
         if (!match_ssbo_array_deref(nir_src_as_deref(chk[i]->src[0]), 1,
                                     &s_set, &s_bind, &s_idx) ||
             s_set != dst_set || s_bind != dst_bind ||
             !match_dst_row_col_idx(s_idx, ds_pc, &s_row, &s_col) ||
             classify_mirror_coord(s_row, true, x, y, dw_pc, dh_pc, b_pc) != 0 ||
             classify_mirror_coord(s_col, false, x, y, dw_pc, dh_pc, b_pc) != 0)
            return false;
      }
   }

   /* Unwrap `res_val_u8 = [u2u8(]f2i32(fmin(fmax(bilerp, 0.0f), 255.0f))[)]`. */
   nir_def *fi_d = res_val_u8;
   if (!phi_is_u32) {
      if (nir_def_instr(res_val_u8)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *u8a = nir_instr_as_alu(nir_def_instr(res_val_u8));
      if (u8a->op != nir_op_u2u8 && u8a->op != nir_op_i2i8)
         return false;
      fi_d = strip_mov(u8a->src[0].src.ssa);
   }
   if (!fi_d || nir_def_instr(fi_d)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *fia = nir_instr_as_alu(nir_def_instr(fi_d));
   if ((fia->op != nir_op_f2i32 && fia->op != nir_op_f2u32) ||
       fia->fp_math_ctrl != 0)
      return false;
   nir_def *cl_hi = strip_mov(fia->src[0].src.ssa);
   if (!cl_hi || nir_def_instr(cl_hi)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *mina = nir_instr_as_alu(nir_def_instr(cl_hi));
   if ((mina->op != nir_op_fmin && mina->op != nir_op_fmin_agx) ||
       (mina->fp_math_ctrl & ~nir_fp_preserve_inf) != 0)
      return false;
   uint32_t bits_hi = 0, bits_lo = 0;
   nir_def *cl_lo = NULL;
   if (match_const_f32_bits(mina->src[1].src.ssa, &bits_hi) &&
       bits_hi == 0x437f0000u)
      cl_lo = strip_mov(mina->src[0].src.ssa);
   else if (match_const_f32_bits(mina->src[0].src.ssa, &bits_hi) &&
            bits_hi == 0x437f0000u)
      cl_lo = strip_mov(mina->src[1].src.ssa);
   else
      return false;
   if (!cl_lo || nir_def_instr(cl_lo)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *maxa = nir_instr_as_alu(nir_def_instr(cl_lo));
   if ((maxa->op != nir_op_fmax && maxa->op != nir_op_fmax_agx) ||
       (maxa->fp_math_ctrl & ~nir_fp_preserve_inf) != 0)
      return false;
   nir_def *bilerp_val = NULL;
   if (match_const_f32_bits(maxa->src[1].src.ssa, &bits_lo) && bits_lo == 0)
      bilerp_val = strip_mov(maxa->src[0].src.ssa);
   else if (match_const_f32_bits(maxa->src[0].src.ssa, &bits_lo) &&
            bits_lo == 0)
      bilerp_val = strip_mov(maxa->src[1].src.ssa);
   else
      return false;

   nir_def *top_f = NULL, *bot_f = NULL, *fy_d = NULL;
   nir_def *p00_f = NULL, *p01_f = NULL, *fx0_d = NULL;
   nir_def *p10_f = NULL, *p11_f = NULL, *fx1_d = NULL;
   enum hk_resize_lerp_mode lerp_val, lerp_top, lerp_bot;
   if (!match_lerp_f32(bilerp_val, &top_f, &bot_f, &fy_d, &lerp_val) ||
       !match_lerp_f32(top_f, &p00_f, &p01_f, &fx0_d, &lerp_top) ||
       !match_lerp_f32(bot_f, &p10_f, &p11_f, &fx1_d, &lerp_bot) ||
       fx0_d != fx1_d || lerp_val != lerp_top || lerp_val != lerp_bot)
      return false;

   nir_intrinsic_instr *p[4] = {NULL};
   if (!match_u2f32_load_deref(p00_f, &p[0]) ||
       !match_u2f32_load_deref(p01_f, &p[1]) ||
       !match_u2f32_load_deref(p10_f, &p[2]) ||
       !match_u2f32_load_deref(p11_f, &p[3]))
      return false;

   uint16_t sw_pc = 0, sh_pc = 0;
   nir_def *sx_f = NULL, *sy_f = NULL;
   {
      nir_def *fdefs[2] = {fx0_d, fy_d};
      nir_def *coords[2] = {x, y};
      uint16_t d_pcs[2] = {dw_pc, dh_pc};
      uint16_t *s_pcs[2] = {&sw_pc, &sh_pc};
      nir_def **s_fs[2] = {&sx_f, &sy_f};
      for (unsigned k = 0; k < 2; k++) {
         nir_def *fd = strip_mov(fdefs[k]);
         if (!fd || nir_def_instr(fd)->type != nir_instr_type_alu)
            return false;
         nir_alu_instr *fa = nir_instr_as_alu(nir_def_instr(fd));
         nir_def *cand = NULL;
         if (fa->op == nir_op_ffract || fa->op == nir_op_fsub)
            cand = strip_mov(fa->src[0].src.ssa);
         else if (fa->op == nir_op_fadd) {
            nir_def *s0 = strip_mov(fa->src[0].src.ssa);
            nir_def *s1 = strip_mov(fa->src[1].src.ssa);
            if (match_scaled_coord_f32(s0, coords[k], d_pcs[k], s_pcs[k]))
               cand = s0;
            else if (match_scaled_coord_f32(s1, coords[k], d_pcs[k], s_pcs[k]))
               cand = s1;
         }
         if (!cand ||
             !match_scaled_coord_f32(cand, coords[k], d_pcs[k], s_pcs[k]) ||
             !match_ftrunc_frac(fd, cand))
            return false;
         *s_fs[k] = cand;
      }
   }

   for (unsigned k = 0; k < 4; k++) {
      if (nir_intrinsic_access(p[k]) &
          (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL))
         return false;
      uint32_t lset = 0, lbind = 0;
      nir_def *lidx = NULL;
      if (!match_ssbo_array_deref(nir_src_as_deref(p[k]->src[0]), 1, &lset,
                                  &lbind, &lidx) ||
          lset != src_set || lbind != src_bind)
         return false;
      nir_def *lrow = NULL, *lcol = NULL;
      if (!match_dst_row_col_idx(lidx, ss_pc, &lrow, &lcol))
         return false;
      if (!match_clamped_src_idx(lrow, sy_f, (k >= 2) ? 1 : 0, sh_pc) ||
          !match_clamped_src_idx(lcol, sx_f, (k & 1) ? 1 : 0, sw_pc))
         return false;
   }

   bool res_on_true = false;
   if (!match_resize_top_cond(mode_if->condition.ssa, &res_on_true, dw_pc,
                              dh_pc, sw_pc, sh_pc) ||
       (res_on_true ? 0u : 1u) != res_arm)
      return false;

   struct mirror_walk_ctx m = {
      .x = x, .y = y, .val_u8 = val_u8,
      .dw_pc = dw_pc, .dh_pc = dh_pc, .ds_pc = ds_pc, .b_pc = b_pc,
      .dst_set = dst_set, .dst_bind = dst_bind,
   };
   if (bdr_if_cnt == 1 &&
       match_any_border_cond(bdr_ifs[0]->condition.ssa, x, y, dw_pc, dh_pc,
                             b_pc)) {
      if (!walk_mirror_border_list(&bdr_ifs[0]->then_list, 0, &m, 0))
         return false;
   } else {
      for (unsigned i = 0; i < bdr_if_cnt; i++) {
         unsigned add_guard = 0;
         if (!match_mirror_guard_cond(bdr_ifs[i]->condition.ssa, &m, &add_guard) ||
             !walk_mirror_border_list(&bdr_ifs[i]->then_list, add_guard, &m, 1))
            return false;
      }
   }
   if (m.seen_mask != 0xff)
      return false;

   unsigned seen_loads = 0, seen_stores = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex ||
             instr->type == nir_instr_type_call)
            return false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_deref) {
            bool ok = (intr == cpy_load || intr == reload_ld);
            for (unsigned k = 0; k < 4 && !ok; k++)
               ok = (intr == p[k]);
            if (!ok)
               return false;
            seen_loads++;
         } else if (intr->intrinsic == nir_intrinsic_store_deref) {
            bool ok = (intr == prim_st || intr == res_prim_st ||
                       intr == cpy_prim_st);
            for (unsigned k = 0; k < 8 && !ok; k++)
               ok = (intr == m.stores[k]);
            if (!ok)
               return false;
            seen_stores++;
         } else if (!is_allowed_base_intrinsic(intr->intrinsic)) {
            return false;
         }
      }
   }
   const unsigned expect_loads = inarm_store_form ? 6u : 5u;
   const unsigned expect_stores = inarm_store_form ? 10u : 9u;
   if (seen_loads != expect_loads || seen_stores != expect_stores)
      return false;

   if (!populate_binding_info(set_layout_count, set_layouts, src_set, src_bind,
                              &plan->bufs[0]) ||
       !populate_binding_info(set_layout_count, set_layouts, dst_set, dst_bind,
                              &plan->bufs[1]))
      return false;

   plan->kind = HK_COMPUTE_OPT_BILINEAR_RESIZE_BORDER_U8;
   plan->flags = (uint8_t)lerp_val;
   plan->tile_w = 128;
   plan->tile_h = 8;
   plan->params[0] = maxa->fp_math_ctrl;
   plan->params[1] = mina->fp_math_ctrl;
   plan->pc_off[0] = dw_pc;
   plan->pc_off[1] = dh_pc;
   plan->pc_off[2] = ds_pc;
   plan->pc_off[3] = b_pc;
   plan->pc_off[4] = sw_pc;
   plan->pc_off[5] = sh_pc;
   plan->pc_off[6] = ss_pc;
   return true;
}

static bool
match_bilinear_resize_border_u8(
   nir_shader *s, uint32_t set_layout_count,
   struct vk_descriptor_set_layout *const *set_layouts,
   struct hk_compute_opt_plan *plan)
{
   const uint16_t wg[3] = {
      (uint16_t)plan->orig_wg_size[0],
      (uint16_t)plan->orig_wg_size[1],
      (uint16_t)plan->orig_wg_size[2],
   };
   nir_function_impl *impl = nir_shader_get_entrypoint(s);

   /* Top-level body: either single `if` selecting resize vs copy branch
    * (loop-unswitched) or outer `y` loop containing inner `x` loop with
    * `if (do_resize)` inside. */
   nir_if *top_if = NULL;
   foreach_list_typed(nir_cf_node, n, node, &impl->body) {
      if (n->type == nir_cf_node_if) {
         if (top_if)
            return false;
         top_if = nir_cf_node_as_if(n);
      } else if (n->type == nir_cf_node_loop) {
         return match_bilinear_resize_border_inloop_u8(
            s, set_layout_count, set_layouts, plan);
      } else if (n->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!top_if)
      return false;

   /* Both arms contain a single outer `y` loop. Distinguish resize (4 loads)
    * from copy (1 load). */
   nir_loop *loops[2] = {NULL, NULL};
   struct exec_list *arms[2] = {&top_if->then_list, &top_if->else_list};
   unsigned arm_loads[2] = {0, 0};

   for (unsigned a = 0; a < 2; a++) {
      foreach_list_typed(nir_cf_node, n, node, arms[a]) {
         if (n->type == nir_cf_node_loop) {
            if (loops[a])
               return false;
            loops[a] = nir_cf_node_as_loop(n);
         } else if (n->type != nir_cf_node_block) {
            return false;
         }
      }
      if (!loops[a])
         return false;
      nir_foreach_block_in_cf_node(blk, &loops[a]->cf_node) {
         nir_foreach_instr(instr, blk) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_load_deref)
               arm_loads[a]++;
         }
      }
   }

   unsigned res_arm = 0, cpy_arm = 1;
   if (arm_loads[0] == 4 && arm_loads[1] == 1) {
      res_arm = 0;
      cpy_arm = 1;
   } else if (arm_loads[0] == 1 && arm_loads[1] == 4) {
      res_arm = 1;
      cpy_arm = 0;
   } else {
      return false;
   }

   /* Match grid-stride loops in both arms. */
   nir_def *ry = NULL, *rx = NULL, *cy = NULL, *cx = NULL;
   uint16_t dw_pc = 0, dh_pc = 0, cdw_pc = 0, cdh_pc = 0;
   nir_if *ry_ex = NULL, *rx_ex = NULL, *cy_ex = NULL, *cx_ex = NULL;
   if (!match_grid_stride_loop_co(loops[res_arm], 1, wg, &ry, &dh_pc, &ry_ex) ||
       !match_grid_stride_loop_co(loops[cpy_arm], 1, wg, &cy, &cdh_pc, &cy_ex) ||
       dh_pc != cdh_pc)
      return false;

   nir_loop *rx_loop = NULL, *cx_loop = NULL;
   foreach_list_typed(nir_cf_node, n, node, &loops[res_arm]->body) {
      if (n->type == nir_cf_node_if) {
         if (nir_cf_node_as_if(n) != ry_ex)
            return false;
      } else if (n->type == nir_cf_node_loop) {
         if (rx_loop)
            return false;
         rx_loop = nir_cf_node_as_loop(n);
      } else if (n->type != nir_cf_node_block) {
         return false;
      }
   }
   foreach_list_typed(nir_cf_node, n, node, &loops[cpy_arm]->body) {
      if (n->type == nir_cf_node_if) {
         if (nir_cf_node_as_if(n) != cy_ex)
            return false;
      } else if (n->type == nir_cf_node_loop) {
         if (cx_loop)
            return false;
         cx_loop = nir_cf_node_as_loop(n);
      } else if (n->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!rx_loop || !cx_loop)
      return false;

   if (!match_grid_stride_loop_co(rx_loop, 0, wg, &rx, &dw_pc, &rx_ex) ||
       !match_grid_stride_loop_co(cx_loop, 0, wg, &cx, &cdw_pc, &cx_ex) ||
       dw_pc != cdw_pc)
      return false;

   /* Inside each inner `x` loop: `x_exit` followed by primary store + border_if. */
   nir_if *res_bdr_if = NULL, *cpy_bdr_if = NULL;
   nir_intrinsic_instr *res_prim_st = NULL, *cpy_prim_st = NULL;
   nir_intrinsic_instr *cpy_load = NULL;
   unsigned res_load_cnt = 0;

   foreach_list_typed(nir_cf_node, n, node, &rx_loop->body) {
      if (n->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(n);
         if (nif == rx_ex)
            continue;
         if (res_bdr_if)
            return false;
         res_bdr_if = nif;
      } else if (n->type == nir_cf_node_block) {
         nir_block *blk = nir_cf_node_as_block(n);
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_deref) {
               if (res_load_cnt >= 4)
                  return false;
               res_load_cnt++;
            } else if (intr->intrinsic == nir_intrinsic_store_deref) {
               if (res_prim_st)
                  return false;
               res_prim_st = intr;
            }
         }
      } else {
         return false;
      }
   }
   foreach_list_typed(nir_cf_node, n, node, &cx_loop->body) {
      if (n->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(n);
         if (nif == cx_ex)
            continue;
         if (cpy_bdr_if)
            return false;
         cpy_bdr_if = nif;
      } else if (n->type == nir_cf_node_block) {
         nir_block *blk = nir_cf_node_as_block(n);
         nir_foreach_instr(instr, blk) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_deref) {
               if (cpy_load)
                  return false;
               cpy_load = intr;
            } else if (intr->intrinsic == nir_intrinsic_store_deref) {
               if (cpy_prim_st)
                  return false;
               cpy_prim_st = intr;
            }
         }
      } else {
         return false;
      }
   }
   if (!res_bdr_if || !cpy_bdr_if || !res_prim_st || !cpy_prim_st ||
       !cpy_load || res_load_cnt != 4)
      return false;
   if (!cf_list_is_empty_block(&res_bdr_if->else_list) ||
       !cf_list_is_empty_block(&cpy_bdr_if->else_list))
      return false;

   /* Extract `src` and `dst` bindings, `src_stride`, `dst_stride`, `border`
    * from the copy branch primary load & store. */
   uint32_t src_set = 0, src_bind = 0, dst_set = 0, dst_bind = 0;
   nir_def *cl_idx = NULL, *cs_idx = NULL, *rs_idx = NULL;
   if (!match_ssbo_array_deref(nir_src_as_deref(cpy_load->src[0]), 1, &src_set,
                               &src_bind, &cl_idx) ||
       !match_ssbo_array_deref(nir_src_as_deref(cpy_prim_st->src[0]), 1,
                               &dst_set, &dst_bind, &cs_idx))
      return false;
   if (src_set == dst_set && src_bind == dst_bind)
      return false;
   if ((nir_intrinsic_access(cpy_load) | nir_intrinsic_access(cpy_prim_st) |
        nir_intrinsic_access(res_prim_st)) &
       (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL))
      return false;
   if (strip_mov(cpy_prim_st->src[1].ssa) != &cpy_load->def)
      return false;

   /* Copy load index: `cy * src_stride + cx`. */
   uint16_t ss_pc = 0;
   if (!match_linear_index(cl_idx, cy, cx, &ss_pc, wg))
      return false;

   /* Copy primary store index: `(cy + border) * dst_stride + (cx + border)`. */
   uint16_t ds_pc = 0, b_pc = 0;
   {
      nir_def *idx = strip_mov(cs_idx);
      if (!idx || nir_def_instr(idx)->type != nir_instr_type_alu)
         return false;
      nir_alu_instr *add = nir_instr_as_alu(nir_def_instr(idx));
      if (add->op != nir_op_iadd)
         return false;
      bool found = false;
      for (unsigned side = 0; side < 2 && !found; side++) {
         nir_def *prod = strip_mov(add->src[side].src.ssa);
         nir_def *col = strip_mov(add->src[1 - side].src.ssa);
         if (!prod || !col || nir_def_instr(prod)->type != nir_instr_type_alu ||
             nir_def_instr(col)->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *mul = nir_instr_as_alu(nir_def_instr(prod));
         nir_alu_instr *cadd = nir_instr_as_alu(nir_def_instr(col));
         if ((mul->op != nir_op_imul && mul->op != nir_op_amul) ||
             cadd->op != nir_op_iadd)
            continue;
         uint16_t bp = 0;
         if (strip_mov(cadd->src[0].src.ssa) == cx &&
             match_push_const_u32(cadd->src[1].src.ssa, &bp)) {
         } else if (strip_mov(cadd->src[1].src.ssa) == cx &&
                    match_push_const_u32(cadd->src[0].src.ssa, &bp)) {
         } else {
            continue;
         }
         for (unsigned ms = 0; ms < 2; ms++) {
            uint16_t sp = 0;
            nir_def *row = strip_mov(mul->src[1 - ms].src.ssa);
            if (!match_push_const_u32(mul->src[ms].src.ssa, &sp) || !row ||
                nir_def_instr(row)->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *radd = nir_instr_as_alu(nir_def_instr(row));
            uint16_t bp2 = 0;
            if (radd->op == nir_op_iadd &&
                ((strip_mov(radd->src[0].src.ssa) == cy &&
                  match_push_const_u32(radd->src[1].src.ssa, &bp2) &&
                  bp2 == bp) ||
                 (strip_mov(radd->src[1].src.ssa) == cy &&
                  match_push_const_u32(radd->src[0].src.ssa, &bp2) &&
                  bp2 == bp))) {
               ds_pc = sp;
               b_pc = bp;
               found = true;
               break;
            }
         }
      }
      if (!found)
         return false;
   }

   /* Verify resize primary store target & bilinear value expression. */
   uint32_t rs_set = 0, rs_bind = 0;
   if (!match_ssbo_array_deref(nir_src_as_deref(res_prim_st->src[0]), 1,
                               &rs_set, &rs_bind, &rs_idx) ||
       rs_set != dst_set || rs_bind != dst_bind)
      return false;
   {
      nir_def *rrow = NULL, *rcol = NULL;
      if (!match_dst_row_col_idx(rs_idx, ds_pc, &rrow, &rcol) ||
          classify_mirror_coord(rrow, true, rx, ry, dw_pc, dh_pc, b_pc) != 0 ||
          classify_mirror_coord(rcol, false, rx, ry, dw_pc, dh_pc, b_pc) != 0)
         return false;
   }

   /* Unwrap `u2u8(f2i32(fmin(fmax(val, 0.0f), 255.0f)))`. */
   nir_def *res_val_u8 = strip_mov(res_prim_st->src[1].ssa);
   if (!res_val_u8 || nir_def_instr(res_val_u8)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *u8a = nir_instr_as_alu(nir_def_instr(res_val_u8));
   if (u8a->op != nir_op_u2u8)
      return false;
   nir_def *fi_d = strip_mov(u8a->src[0].src.ssa);
   if (!fi_d || nir_def_instr(fi_d)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *fia = nir_instr_as_alu(nir_def_instr(fi_d));
   if ((fia->op != nir_op_f2i32 && fia->op != nir_op_f2u32) ||
       fia->fp_math_ctrl != 0)
      return false;
   nir_def *cl_hi = strip_mov(fia->src[0].src.ssa);
   if (!cl_hi || nir_def_instr(cl_hi)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *mina = nir_instr_as_alu(nir_def_instr(cl_hi));
   if ((mina->op != nir_op_fmin && mina->op != nir_op_fmin_agx) ||
       (mina->fp_math_ctrl & ~nir_fp_preserve_inf) != 0)
      return false;
   uint32_t bits_hi = 0, bits_lo = 0;
   nir_def *cl_lo = NULL;
   if (match_const_f32_bits(mina->src[1].src.ssa, &bits_hi) &&
       bits_hi == 0x437f0000u)
      cl_lo = strip_mov(mina->src[0].src.ssa);
   else if (match_const_f32_bits(mina->src[0].src.ssa, &bits_hi) &&
            bits_hi == 0x437f0000u)
      cl_lo = strip_mov(mina->src[1].src.ssa);
   else
      return false;
   if (!cl_lo || nir_def_instr(cl_lo)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *maxa = nir_instr_as_alu(nir_def_instr(cl_lo));
   if ((maxa->op != nir_op_fmax && maxa->op != nir_op_fmax_agx) ||
       (maxa->fp_math_ctrl & ~nir_fp_preserve_inf) != 0)
      return false;
   nir_def *bilerp_val = NULL;
   if (match_const_f32_bits(maxa->src[1].src.ssa, &bits_lo) && bits_lo == 0)
      bilerp_val = strip_mov(maxa->src[0].src.ssa);
   else if (match_const_f32_bits(maxa->src[0].src.ssa, &bits_lo) &&
            bits_lo == 0)
      bilerp_val = strip_mov(maxa->src[1].src.ssa);
   else
      return false;

   /* Match `bilerp_val = lerp(lerp(p00, p01, fx), lerp(p10, p11, fx), fy)`. */
   nir_def *top_f = NULL, *bot_f = NULL, *fy_d = NULL;
   nir_def *p00_f = NULL, *p01_f = NULL, *fx0_d = NULL;
   nir_def *p10_f = NULL, *p11_f = NULL, *fx1_d = NULL;
   enum hk_resize_lerp_mode lerp_val, lerp_top, lerp_bot;
   if (!match_lerp_f32(bilerp_val, &top_f, &bot_f, &fy_d, &lerp_val) ||
       !match_lerp_f32(top_f, &p00_f, &p01_f, &fx0_d, &lerp_top) ||
       !match_lerp_f32(bot_f, &p10_f, &p11_f, &fx1_d, &lerp_bot) ||
       fx0_d != fx1_d || lerp_val != lerp_top || lerp_val != lerp_bot)
      return false;

   nir_intrinsic_instr *p[4] = {NULL};
   if (!match_u2f32_load_deref(p00_f, &p[0]) ||
       !match_u2f32_load_deref(p01_f, &p[1]) ||
       !match_u2f32_load_deref(p10_f, &p[2]) ||
       !match_u2f32_load_deref(p11_f, &p[3]))
      return false;

   /* Extract `sx_f`, `sy_f` from `fx0_d`, `fy_d` and verify `src_w`, `src_h`. */
   uint16_t sw_pc = 0, sh_pc = 0;
   nir_def *sx_f = NULL, *sy_f = NULL;
   {
      /* Extract candidate `s` from `fx = s - ftrunc(s)` (or `ffract(s)`). */
      nir_def *fdefs[2] = {fx0_d, fy_d};
      nir_def *coords[2] = {rx, ry};
      uint16_t d_pcs[2] = {dw_pc, dh_pc};
      uint16_t *s_pcs[2] = {&sw_pc, &sh_pc};
      nir_def **s_fs[2] = {&sx_f, &sy_f};
      for (unsigned k = 0; k < 2; k++) {
         nir_def *fd = strip_mov(fdefs[k]);
         if (!fd || nir_def_instr(fd)->type != nir_instr_type_alu)
            return false;
         nir_alu_instr *fa = nir_instr_as_alu(nir_def_instr(fd));
         nir_def *cand = NULL;
         if (fa->op == nir_op_ffract || fa->op == nir_op_fsub)
            cand = strip_mov(fa->src[0].src.ssa);
         else if (fa->op == nir_op_fadd) {
            nir_def *s0 = strip_mov(fa->src[0].src.ssa);
            nir_def *s1 = strip_mov(fa->src[1].src.ssa);
            if (match_scaled_coord_f32(s0, coords[k], d_pcs[k], s_pcs[k]))
               cand = s0;
            else if (match_scaled_coord_f32(s1, coords[k], d_pcs[k], s_pcs[k]))
               cand = s1;
         }
         if (!cand ||
             !match_scaled_coord_f32(cand, coords[k], d_pcs[k], s_pcs[k]) ||
             !match_ftrunc_frac(fd, cand))
            return false;
         *s_fs[k] = cand;
      }
   }

   /* Verify the 4 bilinear source loads hit exact `(y0/y1)*src_stride + (x0/x1)`. */
   for (unsigned k = 0; k < 4; k++) {
      if (nir_intrinsic_access(p[k]) &
          (ACCESS_VOLATILE | ACCESS_COHERENT | ACCESS_NON_TEMPORAL))
         return false;
      uint32_t lset = 0, lbind = 0;
      nir_def *lidx = NULL;
      if (!match_ssbo_array_deref(nir_src_as_deref(p[k]->src[0]), 1, &lset,
                                  &lbind, &lidx) ||
          lset != src_set || lbind != src_bind)
         return false;
      nir_def *lrow = NULL, *lcol = NULL;
      if (!match_dst_row_col_idx(lidx, ss_pc, &lrow, &lcol))
         return false;
      if (!match_clamped_src_idx(lrow, sy_f, (k >= 2) ? 1 : 0, sh_pc) ||
          !match_clamped_src_idx(lcol, sx_f, (k & 1) ? 1 : 0, sw_pc))
         return false;
   }

   /* Verify top-level condition matches `(src_w != dst_w) || (src_h != dst_h)`. */
   bool res_on_true = false;
   if (!match_resize_top_cond(top_if->condition.ssa, &res_on_true, dw_pc, dh_pc,
                              sw_pc, sh_pc))
      return false;
   if ((res_on_true ? 0u : 1u) != res_arm)
      return false;

   /* Verify mirrored border reflection subtrees in both arms. */
   if (!match_any_border_cond(res_bdr_if->condition.ssa, rx, ry, dw_pc, dh_pc,
                              b_pc) ||
       !match_any_border_cond(cpy_bdr_if->condition.ssa, cx, cy, dw_pc, dh_pc,
                              b_pc))
      return false;

   struct mirror_walk_ctx res_m = {
      .x = rx, .y = ry, .val_u8 = res_val_u8,
      .dw_pc = dw_pc, .dh_pc = dh_pc, .ds_pc = ds_pc, .b_pc = b_pc,
      .dst_set = dst_set, .dst_bind = dst_bind,
   };
   struct mirror_walk_ctx cpy_m = {
      .x = cx, .y = cy, .val_u8 = &cpy_load->def,
      .dw_pc = dw_pc, .dh_pc = dh_pc, .ds_pc = ds_pc, .b_pc = b_pc,
      .dst_set = dst_set, .dst_bind = dst_bind,
   };
   if (!walk_mirror_border_list(&res_bdr_if->then_list, 0, &res_m, 0) ||
       res_m.seen_mask != 0xff ||
       !walk_mirror_border_list(&cpy_bdr_if->then_list, 0, &cpy_m, 0) ||
       cpy_m.seen_mask != 0xff)
      return false;

   /* Whole-shader completeness check: exactly 5 loads, 18 stores, 0 atomics/tex/calls. */
   unsigned seen_loads = 0, seen_stores = 0;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex ||
             instr->type == nir_instr_type_call)
            return false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_deref) {
            bool ok = (intr == cpy_load);
            for (unsigned k = 0; k < 4 && !ok; k++)
               ok = (intr == p[k]);
            if (!ok)
               return false;
            seen_loads++;
         } else if (intr->intrinsic == nir_intrinsic_store_deref) {
            bool ok = (intr == res_prim_st || intr == cpy_prim_st);
            for (unsigned k = 0; k < 8 && !ok; k++)
               ok = (intr == res_m.stores[k] || intr == cpy_m.stores[k]);
            if (!ok)
               return false;
            seen_stores++;
         } else if (!is_allowed_base_intrinsic(intr->intrinsic)) {
            return false;
         }
      }
   }
   if (seen_loads != 5 || seen_stores != 18)
      return false;

   if (!populate_binding_info(set_layout_count, set_layouts, src_set, src_bind,
                              &plan->bufs[0]) ||
       !populate_binding_info(set_layout_count, set_layouts, dst_set, dst_bind,
                              &plan->bufs[1]))
      return false;

   plan->kind = HK_COMPUTE_OPT_BILINEAR_RESIZE_BORDER_U8;
   plan->flags = (uint8_t)lerp_val;
   plan->tile_w = 128;
   plan->tile_h = 8;
   plan->params[0] = maxa->fp_math_ctrl;
   plan->params[1] = mina->fp_math_ctrl;
   plan->pc_off[0] = dw_pc;
   plan->pc_off[1] = dh_pc;
   plan->pc_off[2] = ds_pc;
   plan->pc_off[3] = b_pc;
   plan->pc_off[4] = sw_pc;
   plan->pc_off[5] = sh_pc;
   plan->pc_off[6] = ss_pc;
   return true;
}

/* ---------------------------------------------------------------------------
 * Top-level Matcher Entrypoint
 * --------------------------------------------------------------------------- */

/* Expand AGX-specific fused integer ops on the scratch matcher clone. */
static bool
expand_agx_ops_instr(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   b->cursor = nir_before_instr(&alu->instr);
   nir_def *rep = NULL;

   switch (alu->op) {
   case nir_op_imadshl_agx: {
      nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *m = nir_ssa_for_alu_src(b, alu, 1);
      nir_def *c = nir_ssa_for_alu_src(b, alu, 2);
      nir_def *sh = nir_ssa_for_alu_src(b, alu, 3);
      rep = nir_iadd(b, nir_imul(b, a, m), nir_ishl(b, c, sh));
      break;
   }
   case nir_op_imsubshl_agx: {
      nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *m = nir_ssa_for_alu_src(b, alu, 1);
      nir_def *c = nir_ssa_for_alu_src(b, alu, 2);
      nir_def *sh = nir_ssa_for_alu_src(b, alu, 3);
      rep = nir_isub(b, nir_imul(b, a, m), nir_ishl(b, c, sh));
      break;
   }
   case nir_op_amul: {
      nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *m = nir_ssa_for_alu_src(b, alu, 1);
      rep = nir_imul(b, a, m);
      break;
   }
   default:
      return false;
   }

   nir_def_replace(&alu->def, rep);
   return true;
}

/* Hoist pure constant/push-constant/sysval loads into the entry block so
 * glslang's short-circuit boolean control flow flattens via peephole_select. */
static bool
hoist_matcher_invariants(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_block *start = nir_start_block(impl);
   bool progress = false;

   nir_foreach_block(block, impl) {
      if (block == start)
         continue;
      nir_foreach_instr_safe(instr, block) {
         bool hoist = false;
         if (instr->type == nir_instr_type_load_const) {
            hoist = true;
         } else if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_load_push_constant:
               hoist = nir_src_is_const(intr->src[0]);
               break;
            case nir_intrinsic_load_global_invocation_id:
            case nir_intrinsic_load_base_global_invocation_id:
            case nir_intrinsic_load_base_workgroup_id:
            case nir_intrinsic_load_local_invocation_id:
            case nir_intrinsic_load_workgroup_id:
            case nir_intrinsic_load_num_workgroups:
            case nir_intrinsic_load_workgroup_size:
               hoist = true;
               break;
            default:
               break;
            }
         }
         if (hoist) {
            nir_instr_remove(instr);
            nir_instr_insert_after_block(start, instr);
            progress = true;
         }
      }
   }

   if (progress)
      nir_progress(true, impl, nir_metadata_control_flow);
   return progress;
}

static void
materialize_vector_alu_srcs(nir_shader *s)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_create(impl);
   nir_foreach_block(blk, impl) {
      nir_foreach_instr(instr, blk) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         if (alu->op == nir_op_mov || alu->def.num_components != 1)
            continue;
         unsigned num_srcs = nir_op_infos[alu->op].num_inputs;
         for (unsigned i = 0; i < num_srcs; i++) {
            nir_def *src = alu->src[i].src.ssa;
            if (src->num_components <= 1)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_def *ext = nir_channel(&b, src, alu->src[i].swizzle[0]);
            nir_src_rewrite(&alu->src[i].src, ext);
            alu->src[i].swizzle[0] = 0;
         }
      }
   }
   nir_opt_cse(s);
}

bool
hk_nir_match_compute_opt(const nir_shader *orig_nir, uint32_t set_layout_count,
                         struct vk_descriptor_set_layout *const *set_layouts,
                         struct hk_compute_opt_plan *plan_out)
{
   if (!orig_nir || !plan_out || orig_nir->info.stage != MESA_SHADER_COMPUTE ||
       orig_nir->info.internal || orig_nir->info.workgroup_size_variable ||
       orig_nir->info.num_images > 0 || orig_nir->info.num_textures > 0 ||
       orig_nir->info.num_ubos > 0)
      return false;

   if (orig_nir->info.workgroup_size[0] == 0 ||
       orig_nir->info.workgroup_size[1] == 0 ||
       orig_nir->info.workgroup_size[2] == 0)
      return false;

   if (!exec_list_is_singular(&orig_nir->functions))
      return false;

   if (orig_nir->info.num_ssbos != 2 && orig_nir->info.num_ssbos != 3)
      return false;

   nir_foreach_variable_in_shader(var, orig_nir) {
      if (var->data.mode &
          (nir_var_mem_shared | nir_var_image | nir_var_uniform))
         return false;
   }

   nir_shader *s = nir_shader_clone(NULL, orig_nir);
   if (!s)
      return false;

   nir_lower_vars_to_ssa(s);
   nir_lower_system_values(s);
   nir_lower_load_const_to_scalar(s);
   nir_lower_explicit_io(s, nir_var_mem_push_const,
                         nir_address_format_32bit_offset);
   nir_lower_alu_to_scalar(s, NULL, NULL);
   nir_shader_alu_pass(s, expand_agx_ops_instr, nir_metadata_control_flow, NULL);

   bool progress;
   unsigned iter = 0;
   do {
      progress = false;
      progress |= hoist_matcher_invariants(s);
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_cse);
      NIR_PASS(progress, s, nir_opt_dce);
      NIR_PASS(progress, s, nir_opt_undef);
      NIR_PASS(progress, s, nir_opt_dead_cf);
      const nir_opt_peephole_select_options peep_opts = {
         .limit = 8,
         .indirect_load_ok = true,
         .expensive_alu_ok = true,
      };
      NIR_PASS(progress, s, nir_opt_peephole_select, &peep_opts);
      NIR_PASS(progress, s, nir_opt_if, nir_opt_if_optimize_phi_true_false);
   } while (progress && ++iter < 8);

   materialize_vector_alu_srcs(s);
   nir_metadata_require(nir_shader_get_entrypoint(s), nir_metadata_block_index);

   struct hk_compute_opt_plan plan = {0};
   plan.orig_wg_size[0] = orig_nir->info.workgroup_size[0];
   plan.orig_wg_size[1] = orig_nir->info.workgroup_size[1];
   plan.orig_wg_size[2] = orig_nir->info.workgroup_size[2];

   bool matched = false;
   if (s->info.num_ssbos == 2) {
      matched = match_rgba8_to_gray(s, set_layout_count, set_layouts, &plan) ||
                match_canny_hysteresis(s, set_layout_count, set_layouts, &plan) ||
                match_bilinear_resize_border_u8(s, set_layout_count,
                                                set_layouts, &plan);
   } else if (s->info.num_ssbos == 3) {
      matched = match_canny_nms(s, set_layout_count, set_layouts, &plan) ||
                match_canny_finalize(s, set_layout_count, set_layouts, &plan);
   }

   ralloc_free(s);
   if (matched)
      *plan_out = plan;
   return matched;
}

/* ---------------------------------------------------------------------------
 * NIR Builders
 * --------------------------------------------------------------------------- */

static nir_deref_instr *
build_ssbo_base(nir_builder *b, const struct hk_compute_opt_binding *bind,
                nir_address_format ssbo_addr_format,
                const struct glsl_type *elem_type, unsigned elem_stride,
                unsigned align_mul)
{
   unsigned comps = nir_address_format_num_components(ssbo_addr_format);
   unsigned bits = nir_address_format_bit_size(ssbo_addr_format);
   nir_def *ridx = nir_vulkan_resource_index(
      b, comps, bits, nir_imm_int(b, 0), .desc_set = bind->set,
      .binding = bind->binding,
      .desc_type = nir_descriptor_type_storage_buffer);
   nir_def *desc = nir_load_vulkan_descriptor(
      b, comps, bits, ridx, .desc_type = nir_descriptor_type_storage_buffer);
   return nir_build_deref_cast_with_alignment(
      b, desc, nir_var_mem_ssbo, elem_type, elem_stride, align_mul, 0);
}

static nir_def *
load_pc_i32(nir_builder *b, uint16_t offset_B)
{
   return nir_load_push_constant(b, 1, 32, nir_imm_int(b, 0), .base = offset_B,
                                 .range = 4);
}

static nir_def *
load_ssbo_ro(nir_builder *b, nir_deref_instr *base, nir_def *idx)
{
   return nir_load_deref_with_access(b, nir_build_deref_ptr_as_array(b, base, idx),
                                     ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static void
store_ssbo_wo(nir_builder *b, nir_deref_instr *base, nir_def *idx, nir_def *val,
              unsigned wrmask)
{
   nir_store_deref_with_access(b, nir_build_deref_ptr_as_array(b, base, idx),
                               val, wrmask, ACCESS_NON_READABLE);
}

/* Builder 1: 4-pixel-per-thread vectorized RGBA8 -> Grayscale u8.
 * Workgroup (64, 4, 1) = 256 threads covering 256x4 pixels; coalesced 128-bit
 * u32x4 loads and 32-bit u8x4 stores. */
static nir_def *
compute_gray_u8(nir_builder *b, nir_def *px, uint32_t wr, uint32_t wg,
                uint32_t wb, uint32_t shr)
{
   nir_def *r = nir_iand_imm(b, px, 0xff);
   nir_def *g = nir_ubitfield_extract_imm(b, px, 8, 8);
   nir_def *bl = nir_ubitfield_extract_imm(b, px, 16, 8);
   nir_def *sum = nir_iadd(
      b, nir_iadd(b, nir_imul_imm(b, r, wr), nir_imul_imm(b, g, wg)),
      nir_imul_imm(b, bl, wb));
   return nir_u2u8(b, nir_ushr_imm(b, sum, shr));
}

static void
build_rgba8_to_gray_kernel(nir_builder *b,
                           const struct hk_compute_opt_plan *plan,
                           nir_address_format ssbo_addr_format)
{
   b->shader->info.workgroup_size[0] = 64;
   b->shader->info.workgroup_size[1] = 4;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.workgroup_size_variable = false;
   b->shader->info.shared_size = 0;

   nir_deref_instr *src_u32 = build_ssbo_base(
      b, &plan->bufs[0], ssbo_addr_format, glsl_uint_type(), 4, 4);
   nir_deref_instr *src_u32x4 = build_ssbo_base(
      b, &plan->bufs[0], ssbo_addr_format, glsl_uvec4_type(), 4, 4);
   nir_deref_instr *dst_u8 = build_ssbo_base(
      b, &plan->bufs[1], ssbo_addr_format, glsl_uint8_t_type(), 1, 1);
   nir_deref_instr *dst_u8x4 = build_ssbo_base(
      b, &plan->bufs[1], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1, 1);

   nir_def *width = load_pc_i32(b, plan->pc_off[0]);
   nir_def *height = load_pc_i32(b, plan->pc_off[1]);
   nir_def *src_stride = load_pc_i32(b, plan->pc_off[2]);
   nir_def *dst_stride = load_pc_i32(b, plan->pc_off[3]);

   nir_def *lid = nir_load_local_invocation_id(b);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *x0 = nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wgid, 0), 8),
                          nir_ishl_imm(b, nir_channel(b, lid, 0), 2));
   nir_def *y = nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wgid, 1), 2),
                         nir_channel(b, lid, 1));

   uint32_t wr = plan->params[0], wg = plan->params[1], wb = plan->params[2],
            shr = plan->params[3];

   nir_push_if(b, nir_ilt(b, y, height));
   {
      nir_def *src_row = nir_imul(b, y, src_stride);
      nir_def *dst_row = nir_imul(b, y, dst_stride);

      nir_push_if(b, nir_ige(b, width, nir_iadd_imm(b, x0, 4)));
      {
         nir_def *px4 = load_ssbo_ro(b, src_u32x4, nir_iadd(b, src_row, x0));
         nir_def *g[4];
         for (unsigned k = 0; k < 4; k++)
            g[k] = compute_gray_u8(b, nir_channel(b, px4, k), wr, wg, wb, shr);
         store_ssbo_wo(b, dst_u8x4, nir_iadd(b, dst_row, x0),
                       nir_vec(b, g, 4), 0xf);
      }
      nir_push_else(b, NULL);
      {
         for (unsigned k = 0; k < 4; k++) {
            nir_def *xk = nir_iadd_imm(b, x0, k);
            nir_push_if(b, nir_ilt(b, xk, width));
            {
               nir_def *px = load_ssbo_ro(b, src_u32, nir_iadd(b, src_row, xk));
               nir_def *g = compute_gray_u8(b, px, wr, wg, wb, shr);
               store_ssbo_wo(b, dst_u8, nir_iadd(b, dst_row, xk), g, 0x1);
            }
            nir_pop_if(b, NULL);
         }
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

/* Builder 2: Barrier-free 4-pixel-per-thread Canny Hysteresis u8 with
 * non-weak fast-path skip.
 * Workgroup (64, 4, 1) = 256 threads covering 256x4 pixels (0 B shared memory,
 * 0 barriers). Interior quads (99.8%) issue a single coalesced u8x4 load; when
 * no pixel in the quad equals 1 (~95-98% of quads), all 8-neighbor reads are
 * skipped completely and only a single coalesced u8x4 store is issued. */
static nir_def *
eval_hyst_border_pixel(nir_builder *b, nir_deref_instr *src_u8, nir_def *x,
                       nir_def *y, nir_def *width, nir_def *height)
{
   nir_def *c = load_ssbo_ro(b, src_u8, nir_iadd(b, nir_imul(b, y, width), x));
   nir_def *one8 = nir_imm_intN_t(b, 1, 8);
   nir_def *two8 = nir_imm_intN_t(b, 2, 8);
   nir_def *three8 = nir_imm_intN_t(b, 3, 8);

   nir_variable *res =
      nir_local_variable_create(b->impl, glsl_uint8_t_type(), "hb");
   nir_store_var(b, res, nir_bcsel(b, nir_ieq(b, c, two8), three8, c), 0x1);

   nir_push_if(b, nir_ieq(b, c, one8));
   {
      nir_variable *mx =
         nir_local_variable_create(b->impl, glsl_uint8_t_type(), "hmx");
      nir_store_var(b, mx, nir_imm_intN_t(b, 0, 8), 0x1);
      nir_variable *dyv =
         nir_local_variable_create(b->impl, glsl_int_type(), "hdy");
      nir_store_var(b, dyv, nir_imm_int(b, -1), 0x1);
      nir_push_loop(b);
      {
         nir_def *dy = nir_load_var(b, dyv);
         nir_push_if(b, nir_ilt_imm(b, dy, 2));
         {
            nir_def *ny = nir_iadd(b, y, dy);
            nir_push_if(b, nir_iand(b, nir_ige_imm(b, ny, 0),
                                    nir_ilt(b, ny, height)));
            {
               nir_def *nrow = nir_imul(b, ny, width);
               nir_variable *dxv =
                  nir_local_variable_create(b->impl, glsl_int_type(), "hdx");
               nir_store_var(b, dxv, nir_imm_int(b, -1), 0x1);
               nir_push_loop(b);
               {
                  nir_def *dx = nir_load_var(b, dxv);
                  nir_push_if(b, nir_ilt_imm(b, dx, 2));
                  {
                     nir_def *nx = nir_iadd(b, x, dx);
                     nir_def *not_self =
                        nir_ior(b, nir_ine_imm(b, dx, 0), nir_ine_imm(b, dy, 0));
                     nir_def *in_x = nir_iand(b, nir_ige_imm(b, nx, 0),
                                              nir_ilt(b, nx, width));
                     nir_push_if(b, nir_iand(b, not_self, in_x));
                     {
                        nir_def *nv =
                           load_ssbo_ro(b, src_u8, nir_iadd(b, nrow, nx));
                        nir_store_var(b, mx,
                                      nir_umax(b, nir_load_var(b, mx), nv), 0x1);
                     }
                     nir_pop_if(b, NULL);
                     nir_store_var(b, dxv, nir_iadd_imm(b, dx, 1), 0x1);
                  }
                  nir_push_else(b, NULL);
                  {
                     nir_jump(b, nir_jump_break);
                  }
                  nir_pop_if(b, NULL);
               }
               nir_pop_loop(b, NULL);
            }
            nir_pop_if(b, NULL);
            nir_store_var(b, dyv, nir_iadd_imm(b, dy, 1), 0x1);
         }
         nir_push_else(b, NULL);
         {
            nir_jump(b, nir_jump_break);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_loop(b, NULL);
      nir_store_var(
         b, res,
         nir_bcsel(b, nir_uge(b, nir_load_var(b, mx), two8), three8, one8),
         0x1);
   }
   nir_pop_if(b, NULL);
   return nir_load_var(b, res);
}

static void
build_canny_hysteresis_kernel(nir_builder *b,
                              const struct hk_compute_opt_plan *plan,
                              nir_address_format ssbo_addr_format)
{
   b->shader->info.workgroup_size[0] = 64;
   b->shader->info.workgroup_size[1] = 4;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.workgroup_size_variable = false;
   b->shader->info.shared_size = 0;

   nir_deref_instr *src_u8 = build_ssbo_base(
      b, &plan->bufs[0], ssbo_addr_format, glsl_uint8_t_type(), 1, 1);
   nir_deref_instr *src_u8x2 = build_ssbo_base(
      b, &plan->bufs[0], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 2), 1, 1);
   nir_deref_instr *src_u8x4 = build_ssbo_base(
      b, &plan->bufs[0], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1, 1);
   nir_deref_instr *dst_u8 = build_ssbo_base(
      b, &plan->bufs[1], ssbo_addr_format, glsl_uint8_t_type(), 1, 1);
   nir_deref_instr *dst_u8x4 = build_ssbo_base(
      b, &plan->bufs[1], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1, 1);

   nir_def *width = load_pc_i32(b, plan->pc_off[0]);
   nir_def *height = load_pc_i32(b, plan->pc_off[1]);

   nir_def *lid = nir_load_local_invocation_id(b);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *x0 = nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wgid, 0), 8),
                          nir_ishl_imm(b, nir_channel(b, lid, 0), 2));
   nir_def *gy = nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wgid, 1), 2),
                          nir_channel(b, lid, 1));

   nir_push_if(b, nir_iand(b, nir_ilt(b, gy, height), nir_ilt(b, x0, width)));
   {
      nir_def *row = nir_imul(b, gy, width);
      nir_def *wm1 = nir_iadd_imm(b, width, -1);
      nir_def *hm1 = nir_iadd_imm(b, height, -1);
      nir_def *quad_interior = nir_iand(
         b, nir_iand(b, nir_ige_imm(b, x0, 1),
                     nir_ige(b, wm1, nir_iadd_imm(b, x0, 4))),
         nir_iand(b, nir_ige_imm(b, gy, 1), nir_ilt(b, gy, hm1)));

      nir_push_if(b, quad_interior);
      {
         nir_def *idx = nir_iadd(b, row, x0);
         nir_def *c = load_ssbo_ro(b, src_u8x4, idx);
         nir_def *cv[4] = {
            nir_channel(b, c, 0),
            nir_channel(b, c, 1),
            nir_channel(b, c, 2),
            nir_channel(b, c, 3),
         };
         nir_def *one8 = nir_imm_intN_t(b, 1, 8);
         nir_def *two8 = nir_imm_intN_t(b, 2, 8);
         nir_def *three8 = nir_imm_intN_t(b, 3, 8);

         nir_def *def_o[4];
         for (unsigned k = 0; k < 4; k++)
            def_o[k] = nir_bcsel(b, nir_ieq(b, cv[k], two8), three8, cv[k]);

         nir_variable *out_var = nir_local_variable_create(
            b->impl, glsl_vector_type(GLSL_TYPE_UINT8, 4), "out");
         nir_store_var(b, out_var, nir_vec(b, def_o, 4), 0xf);

         nir_def *has_weak = nir_ior(
            b, nir_ior(b, nir_ieq(b, cv[0], one8), nir_ieq(b, cv[1], one8)),
            nir_ior(b, nir_ieq(b, cv[2], one8), nir_ieq(b, cv[3], one8)));

         nir_push_if(b, has_weak);
         {
            nir_def *idx_up = nir_isub(b, idx, width);
            nir_def *idx_dn = nir_iadd(b, idx, width);
            nir_def *t0 = load_ssbo_ro(b, src_u8x4, nir_iadd_imm(b, idx_up, -1));
            nir_def *t1 = load_ssbo_ro(b, src_u8x2, nir_iadd_imm(b, idx_up, 3));
            nir_def *b0 = load_ssbo_ro(b, src_u8x4, nir_iadd_imm(b, idx_dn, -1));
            nir_def *b1 = load_ssbo_ro(b, src_u8x2, nir_iadd_imm(b, idx_dn, 3));
            nir_def *ml = load_ssbo_ro(b, src_u8, nir_iadd_imm(b, idx, -1));
            nir_def *mr = load_ssbo_ro(b, src_u8, nir_iadd_imm(b, idx, 4));

            nir_def *top[6] = {
               nir_channel(b, t0, 0), nir_channel(b, t0, 1),
               nir_channel(b, t0, 2), nir_channel(b, t0, 3),
               nir_channel(b, t1, 0), nir_channel(b, t1, 1),
            };
            nir_def *mid[6] = {ml, cv[0], cv[1], cv[2], cv[3], mr};
            nir_def *bot[6] = {
               nir_channel(b, b0, 0), nir_channel(b, b0, 1),
               nir_channel(b, b0, 2), nir_channel(b, b0, 3),
               nir_channel(b, b1, 0), nir_channel(b, b1, 1),
            };

            nir_def *col_tb[6], *col_all[6];
            for (unsigned j = 0; j < 6; j++) {
               col_tb[j] = nir_umax(b, top[j], bot[j]);
               col_all[j] = nir_umax(b, col_tb[j], mid[j]);
            }
            nir_def *o[4];
            for (unsigned k = 0; k < 4; k++) {
               nir_def *m8 = nir_umax(
                  b, nir_umax(b, col_all[k], col_all[k + 2]), col_tb[k + 1]);
               nir_def *promoted =
                  nir_bcsel(b, nir_uge(b, m8, two8), three8, one8);
               o[k] = nir_bcsel(b, nir_ieq(b, cv[k], one8), promoted, def_o[k]);
            }
            nir_store_var(b, out_var, nir_vec(b, o, 4), 0xf);
         }
         nir_pop_if(b, NULL);

         store_ssbo_wo(b, dst_u8x4, idx, nir_load_var(b, out_var), 0xf);
      }
      nir_push_else(b, NULL);
      {
         nir_variable *kv =
            nir_local_variable_create(b->impl, glsl_int_type(), "hk");
         nir_store_var(b, kv, nir_imm_int(b, 0), 0x1);
         nir_push_loop(b);
         {
            nir_def *k = nir_load_var(b, kv);
            nir_def *xk = nir_iadd(b, x0, k);
            nir_push_if(b, nir_iand(b, nir_ilt_imm(b, k, 4),
                                    nir_ilt(b, xk, width)));
            {
               nir_def *ov =
                  eval_hyst_border_pixel(b, src_u8, xk, gy, width, height);
               store_ssbo_wo(b, dst_u8, nir_iadd(b, row, xk), ov, 0x1);
               nir_store_var(b, kv, nir_iadd_imm(b, k, 1), 0x1);
            }
            nir_push_else(b, NULL);
            {
               nir_jump(b, nir_jump_break);
            }
            nir_pop_if(b, NULL);
         }
         nir_pop_loop(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

/* Builder 3: 4-pixel-per-thread vectorized Canny NMS with early-out skip.
 * Workgroup (64, 4, 1) covering 256x4 pixels; coalesced 128-bit f32x4 mag loads
 * and 32-bit u8x4 stores, skipping all dir & neighbor mag loads whenever all 4
 * pixels in the quad satisfy mag <= low_thresh. */
static nir_def *
eval_nms_pixel(nir_builder *b, nir_deref_instr *dir_i32,
               nir_deref_instr *mag_f32, nir_def *x, nir_def *y, nir_def *width,
               nir_def *idx, nir_def *m, uint32_t dx_lut, uint32_t dy_lut,
               nir_def *lo_f, nir_def *hi_f)
{
   nir_variable *res = nir_local_variable_create(b->impl, glsl_uint8_t_type(), "nms");
   nir_store_var(b, res, nir_imm_intN_t(b, 0, 8), 0x1);

   nir_push_if(b, nir_flt(b, lo_f, m));
   {
      nir_def *d = load_ssbo_ro(b, dir_i32, idx);
      nir_def *sh = nir_iand_imm(b, nir_ishl_imm(b, d, 1), 31);
      nir_def *dx = nir_iadd_imm(
         b, nir_ubitfield_extract(b, nir_imm_int(b, (int32_t)dx_lut), sh,
                                  nir_imm_int(b, 2)),
         -1);
      nir_def *dy = nir_iadd_imm(
         b, nir_ubitfield_extract(b, nir_imm_int(b, (int32_t)dy_lut), sh,
                                  nir_imm_int(b, 2)),
         -1);

      nir_def *i1 = nir_iadd(b, nir_imul(b, nir_iadd(b, y, dy), width),
                             nir_iadd(b, x, dx));
      nir_def *i2 = nir_iadd(b, nir_imul(b, nir_isub(b, y, dy), width),
                             nir_isub(b, x, dx));
      nir_def *m1 = load_ssbo_ro(b, mag_f32, i1);
      nir_def *m2 = load_ssbo_ro(b, mag_f32, i2);

      nir_def *is_max = nir_iand(b, nir_flt(b, m1, m), nir_flt(b, m2, m));
      nir_def *cls = nir_bcsel(b, nir_flt(b, hi_f, m), nir_imm_intN_t(b, 2, 8),
                               nir_imm_intN_t(b, 1, 8));
      nir_store_var(b, res, nir_bcsel(b, is_max, cls, nir_imm_intN_t(b, 0, 8)),
                    0x1);
   }
   nir_pop_if(b, NULL);
   return nir_load_var(b, res);
}

static void
build_canny_nms_kernel(nir_builder *b, const struct hk_compute_opt_plan *plan,
                       nir_address_format ssbo_addr_format)
{
   b->shader->info.workgroup_size[0] = 64;
   b->shader->info.workgroup_size[1] = 4;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.workgroup_size_variable = false;
   b->shader->info.shared_size = 0;

   nir_deref_instr *dir_i32 = build_ssbo_base(
      b, &plan->bufs[0], ssbo_addr_format, glsl_int_type(), 4, 4);
   nir_deref_instr *mag_f32 = build_ssbo_base(
      b, &plan->bufs[1], ssbo_addr_format, glsl_float_type(), 4, 4);
   nir_deref_instr *mag_f32x4 = build_ssbo_base(
      b, &plan->bufs[1], ssbo_addr_format, glsl_vec4_type(), 4, 4);
   nir_deref_instr *dst_u8 = build_ssbo_base(
      b, &plan->bufs[2], ssbo_addr_format, glsl_uint8_t_type(), 1, 1);
   nir_deref_instr *dst_u8x4 = build_ssbo_base(
      b, &plan->bufs[2], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1, 1);

   nir_def *width = load_pc_i32(b, plan->pc_off[0]);
   nir_def *height = load_pc_i32(b, plan->pc_off[1]);
   nir_def *wm1 = nir_iadd_imm(b, width, -1);
   nir_def *hm1 = nir_iadd_imm(b, height, -1);

   uint32_t dx_lut = plan->params[0], dy_lut = plan->params[1];
   nir_def *lo_f = nir_imm_int(b, (int32_t)plan->params[2]);
   nir_def *hi_f = nir_imm_int(b, (int32_t)plan->params[3]);
   nir_def *z8 = nir_imm_intN_t(b, 0, 8);
   nir_def *z8x4 = nir_vec4(b, z8, z8, z8, z8);

   nir_def *lid = nir_load_local_invocation_id(b);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *x0 = nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wgid, 0), 8),
                          nir_ishl_imm(b, nir_channel(b, lid, 0), 2));
   nir_def *y = nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wgid, 1), 2),
                         nir_channel(b, lid, 1));

   nir_push_if(b, nir_ilt(b, y, height));
   {
      nir_def *row = nir_imul(b, y, width);
      nir_def *y_interior = nir_iand(b, nir_ige_imm(b, y, 1), nir_ilt(b, y, hm1));

      nir_push_if(b, y_interior);
      {
         nir_def *x_interior = nir_iand(
            b, nir_ige_imm(b, x0, 1), nir_ige(b, wm1, nir_iadd_imm(b, x0, 4)));

         nir_push_if(b, x_interior);
         {
            nir_def *base_idx = nir_iadd(b, row, x0);
            nir_def *m4 = load_ssbo_ro(b, mag_f32x4, base_idx);
            nir_def *m0 = nir_channel(b, m4, 0);
            nir_def *m1 = nir_channel(b, m4, 1);
            nir_def *m2 = nir_channel(b, m4, 2);
            nir_def *m3 = nir_channel(b, m4, 3);
            nir_def *any = nir_ior(
               b, nir_ior(b, nir_flt(b, lo_f, m0), nir_flt(b, lo_f, m1)),
               nir_ior(b, nir_flt(b, lo_f, m2), nir_flt(b, lo_f, m3)));

            nir_variable *ov = nir_local_variable_create(
               b->impl, glsl_vector_type(GLSL_TYPE_UINT8, 4), "ov");
            nir_store_var(b, ov, z8x4, 0xf);

            nir_push_if(b, any);
            {
               nir_def *mv[4] = {m0, m1, m2, m3};
               nir_def *o[4];
               for (unsigned k = 0; k < 4; k++) {
                  o[k] = eval_nms_pixel(
                     b, dir_i32, mag_f32, nir_iadd_imm(b, x0, k), y, width,
                     nir_iadd_imm(b, base_idx, k), mv[k], dx_lut, dy_lut, lo_f,
                     hi_f);
               }
               nir_store_var(b, ov, nir_vec(b, o, 4), 0xf);
            }
            nir_pop_if(b, NULL);

            store_ssbo_wo(b, dst_u8x4, base_idx, nir_load_var(b, ov), 0xf);
         }
         nir_push_else(b, NULL);
         {
            for (unsigned k = 0; k < 4; k++) {
               nir_def *xk = nir_iadd_imm(b, x0, k);
               nir_push_if(b, nir_ilt(b, xk, width));
               {
                  nir_def *ik = nir_iadd(b, row, xk);
                  nir_variable *ok = nir_local_variable_create(
                     b->impl, glsl_uint8_t_type(), "ok");
                  nir_store_var(b, ok, z8, 0x1);
                  nir_push_if(b, nir_iand(b, nir_ige_imm(b, xk, 1),
                                          nir_ilt(b, xk, wm1)));
                  {
                     nir_def *m = load_ssbo_ro(b, mag_f32, ik);
                     nir_store_var(
                        b, ok,
                        eval_nms_pixel(b, dir_i32, mag_f32, xk, y, width, ik, m,
                                       dx_lut, dy_lut, lo_f, hi_f),
                        0x1);
                  }
                  nir_pop_if(b, NULL);
                  store_ssbo_wo(b, dst_u8, ik, nir_load_var(b, ok), 0x1);
               }
               nir_pop_if(b, NULL);
            }
         }
         nir_pop_if(b, NULL);
      }
      nir_push_else(b, NULL);
      {
         /* Top/bottom border rows: output 0 without touching mag or dir. */
         nir_push_if(b, nir_ige(b, width, nir_iadd_imm(b, x0, 4)));
         {
            store_ssbo_wo(b, dst_u8x4, nir_iadd(b, row, x0), z8x4, 0xf);
         }
         nir_push_else(b, NULL);
         {
            for (unsigned k = 0; k < 4; k++) {
               nir_def *xk = nir_iadd_imm(b, x0, k);
               nir_push_if(b, nir_ilt(b, xk, width));
               {
                  store_ssbo_wo(b, dst_u8, nir_iadd(b, row, xk), z8, 0x1);
               }
               nir_pop_if(b, NULL);
            }
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

/* Builder 4: 4-pixel-per-thread vectorized Canny Finalize. */
static nir_def *
eval_finalize_pixel(nir_builder *b, nir_deref_instr *mag_f32, nir_def *idx,
                    nir_def *st, nir_def *thresh8, bool is_f2u32)
{
   nir_variable *res = nir_local_variable_create(b->impl, glsl_uint8_t_type(), "fn");
   nir_store_var(b, res, nir_imm_intN_t(b, 0, 8), 0x1);
   nir_push_if(b, nir_uge(b, st, thresh8));
   {
      nir_def *m = load_ssbo_ro(b, mag_f32, idx);
      nir_def *c = nir_fmin(b, nir_fmax(b, m, nir_imm_float(b, 0.0f)),
                            nir_imm_float(b, 255.0f));
      nir_def *iv = is_f2u32 ? nir_f2u32(b, c) : nir_f2i32(b, c);
      nir_store_var(b, res, nir_u2u8(b, iv), 0x1);
   }
   nir_pop_if(b, NULL);
   return nir_load_var(b, res);
}

static void
build_canny_finalize_kernel(nir_builder *b,
                            const struct hk_compute_opt_plan *plan,
                            nir_address_format ssbo_addr_format)
{
   b->shader->info.workgroup_size[0] = 64;
   b->shader->info.workgroup_size[1] = 4;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.workgroup_size_variable = false;
   b->shader->info.shared_size = 0;

   nir_deref_instr *st_u8 = build_ssbo_base(
      b, &plan->bufs[0], ssbo_addr_format, glsl_uint8_t_type(), 1, 1);
   nir_deref_instr *st_u8x4 = build_ssbo_base(
      b, &plan->bufs[0], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1, 1);
   nir_deref_instr *mag_f32 = build_ssbo_base(
      b, &plan->bufs[1], ssbo_addr_format, glsl_float_type(), 4, 4);
   nir_deref_instr *dst_u8 = build_ssbo_base(
      b, &plan->bufs[2], ssbo_addr_format, glsl_uint8_t_type(), 1, 1);
   nir_deref_instr *dst_u8x4 = build_ssbo_base(
      b, &plan->bufs[2], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1, 1);

   nir_def *width = load_pc_i32(b, plan->pc_off[0]);
   nir_def *height = load_pc_i32(b, plan->pc_off[1]);
   nir_def *dst_stride = load_pc_i32(b, plan->pc_off[2]);
   nir_def *thresh8 = nir_imm_intN_t(b, (uint8_t)plan->params[0], 8);
   bool is_f2u32 = (plan->flags & 1u) != 0;
   nir_def *z8 = nir_imm_intN_t(b, 0, 8);
   nir_def *z8x4 = nir_vec4(b, z8, z8, z8, z8);

   nir_def *lid = nir_load_local_invocation_id(b);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *x0 = nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wgid, 0), 8),
                          nir_ishl_imm(b, nir_channel(b, lid, 0), 2));
   nir_def *y = nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wgid, 1), 2),
                         nir_channel(b, lid, 1));

   nir_push_if(b, nir_ilt(b, y, height));
   {
      nir_def *src_row = nir_imul(b, y, width);
      nir_def *dst_row = nir_imul(b, y, dst_stride);

      nir_push_if(b, nir_ige(b, width, nir_iadd_imm(b, x0, 4)));
      {
         nir_def *sidx = nir_iadd(b, src_row, x0);
         nir_def *s4 = load_ssbo_ro(b, st_u8x4, sidx);
         nir_def *s0 = nir_channel(b, s4, 0);
         nir_def *s1 = nir_channel(b, s4, 1);
         nir_def *s2 = nir_channel(b, s4, 2);
         nir_def *s3 = nir_channel(b, s4, 3);
         nir_def *any = nir_ior(
            b, nir_ior(b, nir_uge(b, s0, thresh8), nir_uge(b, s1, thresh8)),
            nir_ior(b, nir_uge(b, s2, thresh8), nir_uge(b, s3, thresh8)));

         nir_variable *ov = nir_local_variable_create(
            b->impl, glsl_vector_type(GLSL_TYPE_UINT8, 4), "ov");
         nir_store_var(b, ov, z8x4, 0xf);

         nir_push_if(b, any);
         {
            nir_def *sv[4] = {s0, s1, s2, s3};
            nir_def *o[4];
            for (unsigned k = 0; k < 4; k++) {
               o[k] = eval_finalize_pixel(b, mag_f32, nir_iadd_imm(b, sidx, k),
                                          sv[k], thresh8, is_f2u32);
            }
            nir_store_var(b, ov, nir_vec(b, o, 4), 0xf);
         }
         nir_pop_if(b, NULL);

         store_ssbo_wo(b, dst_u8x4, nir_iadd(b, dst_row, x0),
                       nir_load_var(b, ov), 0xf);
      }
      nir_push_else(b, NULL);
      {
         for (unsigned k = 0; k < 4; k++) {
            nir_def *xk = nir_iadd_imm(b, x0, k);
            nir_push_if(b, nir_ilt(b, xk, width));
            {
               nir_def *si = nir_iadd(b, src_row, xk);
               nir_def *sv = load_ssbo_ro(b, st_u8, si);
               nir_def *ov =
                  eval_finalize_pixel(b, mag_f32, si, sv, thresh8, is_f2u32);
               store_ssbo_wo(b, dst_u8, nir_iadd(b, dst_row, xk), ov, 0x1);
            }
            nir_pop_if(b, NULL);
         }
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

/* Builder 5: Vectorized 4-pixel-per-thread bilinear u8 resize / copy with
 * mirrored border padding.
 * Workgroup (32, 8, 1) = 256 threads covering 128x8 pixels:
 * - Interior quads (>95%) skip all 20 border branches and issue coalesced
 *   u8x4 stores (plus u8x4 loads on copy and u8x2 loads on bilinear resize).
 * - Border quads (~5%) run exact scalar primary + mirrored border stores. */
static void
emit_mirror_border_writes(nir_builder *b, nir_deref_instr *dst_u8,
                          nir_def *x, nir_def *y, nir_def *val_u8,
                          nir_def *dst_w, nir_def *dst_h, nir_def *dst_stride,
                          nir_def *border)
{
   nir_def *y_row = nir_imul(b, nir_iadd(b, y, border), dst_stride);
   nir_def *px = nir_iadd(b, x, border);
   store_ssbo_wo(b, dst_u8, nir_iadd(b, y_row, px), val_u8, 0x1);

   nir_def *x_lim_lo = nir_isub(b, nir_iadd_imm(b, dst_w, -2), border);
   nir_def *x_lim_hi = nir_iadd_imm(b, dst_w, -1);
   nir_def *y_lim_lo = nir_isub(b, nir_iadd_imm(b, dst_h, -2), border);
   nir_def *y_lim_hi = nir_iadd_imm(b, dst_h, -1);

   nir_def *x_lft = nir_uge(b, border, x);
   nir_def *x_rgt = nir_iand(b, nir_ult(b, x_lim_lo, x), nir_ult(b, x, x_lim_hi));
   nir_def *y_top = nir_uge(b, border, y);
   nir_def *y_bot = nir_iand(b, nir_ult(b, y_lim_lo, y), nir_ult(b, y, y_lim_hi));
   nir_def *y_any = nir_ior(b, y_top, y_bot);

   nir_push_if(b, nir_ior(b, nir_ior(b, x_lft, x_rgt), y_any));
   {
      nir_def *x_lft_nz = nir_iand(b, nir_ine_imm(b, x, 0), x_lft);
      nir_def *y_top_nz = nir_iand(b, nir_ine_imm(b, y, 0), y_top);
      nir_def *ry_top = nir_imul(b, nir_isub(b, border, y), dst_stride);
      nir_def *ry_bot = nir_imul(
         b,
         nir_isub(b, nir_iadd(b, nir_ishl_imm(b, dst_h, 1),
                              nir_iadd_imm(b, border, -2)),
                  y),
         dst_stride);

      nir_push_if(b, x_lft_nz);
      {
         nir_def *rx_lft = nir_isub(b, border, x);
         store_ssbo_wo(b, dst_u8, nir_iadd(b, y_row, rx_lft), val_u8, 0x1);
         nir_push_if(b, y_top_nz);
         store_ssbo_wo(b, dst_u8, nir_iadd(b, ry_top, rx_lft), val_u8, 0x1);
         nir_pop_if(b, NULL);
         nir_push_if(b, y_bot);
         store_ssbo_wo(b, dst_u8, nir_iadd(b, ry_bot, rx_lft), val_u8, 0x1);
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);

      nir_push_if(b, y_top_nz);
      store_ssbo_wo(b, dst_u8, nir_iadd(b, ry_top, px), val_u8, 0x1);
      nir_pop_if(b, NULL);

      nir_push_if(b, x_rgt);
      {
         nir_def *rx_rgt = nir_isub(
            b,
            nir_iadd(b, nir_ishl_imm(b, dst_w, 1), nir_iadd_imm(b, border, -2)),
            x);
         store_ssbo_wo(b, dst_u8, nir_iadd(b, y_row, rx_rgt), val_u8, 0x1);
         nir_push_if(b, y_top_nz);
         store_ssbo_wo(b, dst_u8, nir_iadd(b, ry_top, rx_rgt), val_u8, 0x1);
         nir_pop_if(b, NULL);
         nir_push_if(b, y_bot);
         store_ssbo_wo(b, dst_u8, nir_iadd(b, ry_bot, rx_rgt), val_u8, 0x1);
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);

      nir_push_if(b, y_bot);
      store_ssbo_wo(b, dst_u8, nir_iadd(b, ry_bot, px), val_u8, 0x1);
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

static nir_def *
eval_lerp_f32(nir_builder *b, nir_def *a, nir_def *bv, nir_def *t,
              enum hk_resize_lerp_mode mode)
{
   switch (mode) {
   case HK_RESIZE_LERP_ONE_MINUS_ADD_A: {
      nir_def *omt = nir_fsub(b, nir_imm_float(b, 1.0f), t);
      return nir_fadd(b, nir_fmul(b, omt, a), nir_fmul(b, t, bv));
   }
   case HK_RESIZE_LERP_ONE_MINUS_ADD_B: {
      nir_def *omt = nir_fsub(b, nir_imm_float(b, 1.0f), t);
      return nir_fadd(b, nir_fmul(b, t, bv), nir_fmul(b, omt, a));
   }
   case HK_RESIZE_LERP_ONE_MINUS_FFMA_A: {
      nir_def *omt = nir_fsub(b, nir_imm_float(b, 1.0f), t);
      return nir_ffma(b, omt, a, nir_fmul(b, t, bv));
   }
   case HK_RESIZE_LERP_ONE_MINUS_FFMA_B: {
      nir_def *omt = nir_fsub(b, nir_imm_float(b, 1.0f), t);
      return nir_ffma(b, t, bv, nir_fmul(b, omt, a));
   }
   case HK_RESIZE_LERP_DIFF_FFMA:
      return nir_ffma(b, t, nir_fsub(b, bv, a), a);
   case HK_RESIZE_LERP_DIFF_ADD:
      return nir_fadd(b, nir_fmul(b, t, nir_fsub(b, bv, a)), a);
   }
   UNREACHABLE("invalid hk_resize_lerp_mode");
}

static nir_def *
eval_bilerp_u8_pixel(nir_builder *b, nir_def *p00, nir_def *p01, nir_def *p10,
                     nir_def *p11, nir_def *fx, nir_def *fy,
                     enum hk_resize_lerp_mode lerp_mode, uint32_t fmax_fp_ctrl,
                     uint32_t fmin_fp_ctrl)
{
   nir_def *f00 = nir_u2f32(b, p00);
   nir_def *f01 = nir_u2f32(b, p01);
   nir_def *f10 = nir_u2f32(b, p10);
   nir_def *f11 = nir_u2f32(b, p11);
   nir_def *top = eval_lerp_f32(b, f00, f01, fx, lerp_mode);
   nir_def *bot = eval_lerp_f32(b, f10, f11, fx, lerp_mode);
   nir_def *val = eval_lerp_f32(b, top, bot, fy, lerp_mode);
   nir_def *lo = nir_fmax(b, val, nir_imm_float(b, 0.0f));
   if (nir_def_instr(lo)->type == nir_instr_type_alu)
      nir_instr_as_alu(nir_def_instr(lo))->fp_math_ctrl = fmax_fp_ctrl;
   nir_def *cl = nir_fmin(b, lo, nir_imm_float(b, 255.0f));
   if (nir_def_instr(cl)->type == nir_instr_type_alu)
      nir_instr_as_alu(nir_def_instr(cl))->fp_math_ctrl = fmin_fp_ctrl;
   return nir_u2u8(b, nir_f2i32(b, cl));
}

static void
emit_resize_border_loop(nir_builder *b, nir_deref_instr *src_u8,
                        nir_deref_instr *dst_u8, nir_def *x0, nir_def *py,
                        nir_def *dst_w, nir_def *dst_h, nir_def *dst_stride,
                        nir_def *border, nir_def *swm1, nir_def *sx_scale,
                        nir_def *fy, nir_def *row0, nir_def *row1,
                        enum hk_resize_lerp_mode lerp_mode,
                        uint32_t fmax_fp_ctrl, uint32_t fmin_fp_ctrl)
{
   nir_variable *kv = nir_local_variable_create(b->impl, glsl_uint_type(), "k");
   nir_store_var(b, kv, nir_imm_int(b, 0), 0x1);
   nir_loop *loop = nir_push_loop(b);
   loop->control = nir_loop_control_dont_unroll;
   {
      nir_def *k = nir_load_var(b, kv);
      nir_push_if(b, nir_uge_imm(b, k, 4));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      nir_def *xk = nir_iadd(b, x0, k);
      nir_push_if(b, nir_ult(b, xk, dst_w));
      {
         nir_def *sx = nir_fmul(b, nir_u2f32(b, xk), sx_scale);
         nir_def *fx = nir_fsub(b, sx, nir_ftrunc(b, sx));
         nir_def *ix = nir_f2i32(b, sx);
         nir_def *ix0 = nir_imin(b, nir_imax(b, ix, nir_imm_int(b, 0)), swm1);
         nir_def *ix1 =
            nir_imin(b, nir_imax(b, nir_iadd_imm(b, ix, 1), nir_imm_int(b, 0)),
                     swm1);
         nir_def *p00 = load_ssbo_ro(b, src_u8, nir_iadd(b, row0, ix0));
         nir_def *p01 = load_ssbo_ro(b, src_u8, nir_iadd(b, row0, ix1));
         nir_def *p10 = load_ssbo_ro(b, src_u8, nir_iadd(b, row1, ix0));
         nir_def *p11 = load_ssbo_ro(b, src_u8, nir_iadd(b, row1, ix1));
         nir_def *val = eval_bilerp_u8_pixel(b, p00, p01, p10, p11, fx, fy,
                                             lerp_mode, fmax_fp_ctrl,
                                             fmin_fp_ctrl);
         emit_mirror_border_writes(b, dst_u8, xk, py, val, dst_w, dst_h,
                                   dst_stride, border);
      }
      nir_pop_if(b, NULL);
      nir_store_var(b, kv, nir_iadd_imm(b, k, 1), 0x1);
   }
   nir_pop_loop(b, NULL);
}

static void
emit_copy_border_loop(nir_builder *b, nir_deref_instr *src_u8,
                      nir_deref_instr *dst_u8, nir_def *x0, nir_def *py,
                      nir_def *dst_w, nir_def *dst_h, nir_def *dst_stride,
                      nir_def *border, nir_def *srow)
{
   nir_variable *kv = nir_local_variable_create(b->impl, glsl_uint_type(), "kc");
   nir_store_var(b, kv, nir_imm_int(b, 0), 0x1);
   nir_loop *loop = nir_push_loop(b);
   loop->control = nir_loop_control_dont_unroll;
   {
      nir_def *k = nir_load_var(b, kv);
      nir_push_if(b, nir_uge_imm(b, k, 4));
      nir_jump(b, nir_jump_break);
      nir_pop_if(b, NULL);

      nir_def *xk = nir_iadd(b, x0, k);
      nir_push_if(b, nir_ult(b, xk, dst_w));
      {
         nir_def *val = load_ssbo_ro(b, src_u8, nir_iadd(b, srow, xk));
         emit_mirror_border_writes(b, dst_u8, xk, py, val, dst_w, dst_h,
                                   dst_stride, border);
      }
      nir_pop_if(b, NULL);
      nir_store_var(b, kv, nir_iadd_imm(b, k, 1), 0x1);
   }
   nir_pop_loop(b, NULL);
}

static void
build_bilinear_resize_border_u8_kernel(nir_builder *b,
                                       const struct hk_compute_opt_plan *plan,
                                       nir_address_format ssbo_addr_format)
{
   b->shader->info.workgroup_size[0] = 32;
   b->shader->info.workgroup_size[1] = 8;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.workgroup_size_variable = false;
   b->shader->info.shared_size = 0;

   nir_deref_instr *src_u8 = build_ssbo_base(
      b, &plan->bufs[0], ssbo_addr_format, glsl_uint8_t_type(), 1, 1);
   nir_deref_instr *src_u8x2 = build_ssbo_base(
      b, &plan->bufs[0], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 2), 1, 1);
   nir_deref_instr *src_u8x4 = build_ssbo_base(
      b, &plan->bufs[0], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1, 1);
   nir_deref_instr *dst_u8 = build_ssbo_base(
      b, &plan->bufs[1], ssbo_addr_format, glsl_uint8_t_type(), 1, 1);
   nir_deref_instr *dst_u8x4 = build_ssbo_base(
      b, &plan->bufs[1], ssbo_addr_format,
      glsl_vector_type(GLSL_TYPE_UINT8, 4), 1, 1);

   nir_def *dst_w = load_pc_i32(b, plan->pc_off[0]);
   nir_def *dst_h = load_pc_i32(b, plan->pc_off[1]);
   nir_def *dst_stride = load_pc_i32(b, plan->pc_off[2]);
   nir_def *border = load_pc_i32(b, plan->pc_off[3]);
   nir_def *src_w = load_pc_i32(b, plan->pc_off[4]);
   nir_def *src_h = load_pc_i32(b, plan->pc_off[5]);
   nir_def *src_stride = load_pc_i32(b, plan->pc_off[6]);

   nir_def *lid = nir_load_local_invocation_id(b);
   nir_def *wgid = nir_load_workgroup_id(b);
   nir_def *x0 = nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wgid, 0), 7),
                          nir_ishl_imm(b, nir_channel(b, lid, 0), 2));
   nir_def *py = nir_iadd(b, nir_ishl_imm(b, nir_channel(b, wgid, 1), 3),
                          nir_channel(b, lid, 1));

   const enum hk_resize_lerp_mode lerp_mode =
      (enum hk_resize_lerp_mode)plan->flags;
   const uint32_t fmax_fp_ctrl = plan->params[0];
   const uint32_t fmin_fp_ctrl = plan->params[1];
   nir_def *do_resize =
      nir_ior(b, nir_ine(b, src_w, dst_w), nir_ine(b, src_h, dst_h));

   nir_push_if(b, do_resize);
   {
      nir_push_if(b, nir_ult(b, py, dst_h));
      {
         nir_def *x_lim_lo = nir_isub(b, nir_iadd_imm(b, dst_w, -2), border);
         nir_def *y_lim_lo = nir_isub(b, nir_iadd_imm(b, dst_h, -2), border);
         nir_def *quad_int = nir_iand(
            b,
            nir_iand(b, nir_ult(b, border, py), nir_uge(b, y_lim_lo, py)),
            nir_iand(b, nir_ult(b, border, x0),
                     nir_uge(b, x_lim_lo, nir_iadd_imm(b, x0, 3))));

         nir_def *sx_scale =
            nir_fmul(b, nir_i2f32(b, src_w), nir_frcp(b, nir_i2f32(b, dst_w)));
         nir_def *sy_scale =
            nir_fmul(b, nir_i2f32(b, src_h), nir_frcp(b, nir_i2f32(b, dst_h)));

         nir_def *sy = nir_fmul(b, nir_u2f32(b, py), sy_scale);
         nir_def *fy = nir_fsub(b, sy, nir_ftrunc(b, sy));
         nir_def *iy = nir_f2i32(b, sy);
         nir_def *shm1 = nir_iadd_imm(b, src_h, -1);
         nir_def *swm1 = nir_iadd_imm(b, src_w, -1);
         nir_def *y0 = nir_imin(b, nir_imax(b, iy, nir_imm_int(b, 0)), shm1);
         nir_def *y1 =
            nir_imin(b, nir_imax(b, nir_iadd_imm(b, iy, 1), nir_imm_int(b, 0)),
                     shm1);
         nir_def *row0 = nir_imul(b, y0, src_stride);
         nir_def *row1 = nir_imul(b, y1, src_stride);

         nir_push_if(b, quad_int);
         {
            nir_def *ix3 =
               nir_f2i32(b, nir_fmul(b, nir_u2f32(b, nir_iadd_imm(b, x0, 3)),
                                     sx_scale));
            nir_def *fast_u8x2 = nir_ult(b, ix3, swm1);
            nir_variable *out_v[4];
            for (unsigned k = 0; k < 4; k++)
               out_v[k] =
                  nir_local_variable_create(b->impl, glsl_uint8_t_type(), "ro");

            nir_push_if(b, fast_u8x2);
            {
               for (unsigned k = 0; k < 4; k++) {
                  nir_def *sx =
                     nir_fmul(b, nir_u2f32(b, nir_iadd_imm(b, x0, k)), sx_scale);
                  nir_def *fx = nir_fsub(b, sx, nir_ftrunc(b, sx));
                  nir_def *ix = nir_f2i32(b, sx);
                  nir_def *r0 =
                     load_ssbo_ro(b, src_u8x2, nir_iadd(b, row0, ix));
                  nir_def *r1 =
                     load_ssbo_ro(b, src_u8x2, nir_iadd(b, row1, ix));
                  nir_store_var(
                     b, out_v[k],
                     eval_bilerp_u8_pixel(b, nir_channel(b, r0, 0),
                                          nir_channel(b, r0, 1),
                                          nir_channel(b, r1, 0),
                                          nir_channel(b, r1, 1), fx, fy,
                                          lerp_mode, fmax_fp_ctrl,
                                          fmin_fp_ctrl),
                     0x1);
               }
            }
            nir_push_else(b, NULL);
            {
               for (unsigned k = 0; k < 4; k++) {
                  nir_def *sx =
                     nir_fmul(b, nir_u2f32(b, nir_iadd_imm(b, x0, k)), sx_scale);
                  nir_def *fx = nir_fsub(b, sx, nir_ftrunc(b, sx));
                  nir_def *ix = nir_f2i32(b, sx);
                  nir_def *ix0 =
                     nir_imin(b, nir_imax(b, ix, nir_imm_int(b, 0)), swm1);
                  nir_def *ix1 = nir_imin(
                     b, nir_imax(b, nir_iadd_imm(b, ix, 1), nir_imm_int(b, 0)),
                     swm1);
                  nir_def *p00 =
                     load_ssbo_ro(b, src_u8, nir_iadd(b, row0, ix0));
                  nir_def *p01 =
                     load_ssbo_ro(b, src_u8, nir_iadd(b, row0, ix1));
                  nir_def *p10 =
                     load_ssbo_ro(b, src_u8, nir_iadd(b, row1, ix0));
                  nir_def *p11 =
                     load_ssbo_ro(b, src_u8, nir_iadd(b, row1, ix1));
                  nir_store_var(
                     b, out_v[k],
                     eval_bilerp_u8_pixel(b, p00, p01, p10, p11, fx, fy,
                                          lerp_mode, fmax_fp_ctrl,
                                          fmin_fp_ctrl),
                     0x1);
               }
            }
            nir_pop_if(b, NULL);

            nir_def *didx = nir_iadd(
               b, nir_imul(b, nir_iadd(b, py, border), dst_stride),
               nir_iadd(b, x0, border));
            store_ssbo_wo(b, dst_u8x4, didx,
                          nir_vec4(b, nir_load_var(b, out_v[0]),
                                   nir_load_var(b, out_v[1]),
                                   nir_load_var(b, out_v[2]),
                                   nir_load_var(b, out_v[3])),
                          0xf);
         }
         nir_push_else(b, NULL);
         {
            emit_resize_border_loop(b, src_u8, dst_u8, x0, py, dst_w, dst_h,
                                    dst_stride, border, swm1, sx_scale, fy,
                                    row0, row1, lerp_mode, fmax_fp_ctrl,
                                    fmin_fp_ctrl);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   {
      nir_push_if(b, nir_ult(b, py, dst_h));
      {
         /* Copy branch (`src_w == dst_w && src_h == dst_h`). */
         nir_def *x_lim_lo = nir_isub(b, nir_iadd_imm(b, dst_w, -2), border);
         nir_def *y_lim_lo = nir_isub(b, nir_iadd_imm(b, dst_h, -2), border);
         nir_def *quad_int = nir_iand(
            b,
            nir_iand(b, nir_ult(b, border, py), nir_uge(b, y_lim_lo, py)),
            nir_iand(b, nir_ult(b, border, x0),
                     nir_uge(b, x_lim_lo, nir_iadd_imm(b, x0, 3))));
         nir_def *srow = nir_imul(b, py, src_stride);
         nir_push_if(b, quad_int);
         {
            nir_def *v4 = load_ssbo_ro(b, src_u8x4, nir_iadd(b, srow, x0));
            nir_def *didx = nir_iadd(
               b, nir_imul(b, nir_iadd(b, py, border), dst_stride),
               nir_iadd(b, x0, border));
            store_ssbo_wo(b, dst_u8x4, didx, v4, 0xf);
         }
         nir_push_else(b, NULL);
         {
            emit_copy_border_loop(b, src_u8, dst_u8, x0, py, dst_w, dst_h,
                                  dst_stride, border, srow);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

nir_shader *
hk_nir_build_compute_opt(void *mem_ctx, const nir_shader *orig_nir,
                         const struct hk_compute_opt_plan *plan,
                         nir_address_format ssbo_addr_format)
{
   if (!orig_nir || !plan)
      return NULL;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, orig_nir->options, "%s_compute_opt",
      orig_nir->info.name ? orig_nir->info.name : "compute_opt");
   b.shader->info.internal = false;
   ralloc_steal(mem_ctx, b.shader);
   b.shader->info.zero_initialize_shared_memory =
      orig_nir->info.zero_initialize_shared_memory;
   b.shader->info.float_controls_execution_mode =
      orig_nir->info.float_controls_execution_mode;

   switch ((enum hk_compute_opt_kind)plan->kind) {
   case HK_COMPUTE_OPT_RGBA8_TO_GRAY_U8:
      build_rgba8_to_gray_kernel(&b, plan, ssbo_addr_format);
      break;
   case HK_COMPUTE_OPT_CANNY_HYSTERESIS_U8:
      build_canny_hysteresis_kernel(&b, plan, ssbo_addr_format);
      break;
   case HK_COMPUTE_OPT_CANNY_NMS_U8:
      build_canny_nms_kernel(&b, plan, ssbo_addr_format);
      break;
   case HK_COMPUTE_OPT_CANNY_FINALIZE_U8:
      build_canny_finalize_kernel(&b, plan, ssbo_addr_format);
      break;
   case HK_COMPUTE_OPT_BILINEAR_RESIZE_BORDER_U8:
      build_bilinear_resize_border_u8_kernel(&b, plan, ssbo_addr_format);
      break;
   default:
      ralloc_free(b.shader);
      return NULL;
   }

   nir_lower_vars_to_ssa(b.shader);
   nir_validate_shader(b.shader, "after hk_nir_build_compute_opt");
   return b.shader;
}

/* ---------------------------------------------------------------------------
 * Dispatch-time Validation
 * --------------------------------------------------------------------------- */

static bool
resolve_ssbo_binding(const struct hk_descriptor_state *desc,
                     const struct hk_compute_opt_binding *b,
                     struct hk_buffer_address *addr_out)
{
   if (b->set >= HK_MAX_SETS)
      return false;
   if (desc->buffer_active & BITFIELD_BIT(b->set))
      return false;

   if (desc->push_active & BITFIELD_BIT(b->set)) {
      const struct hk_push_descriptor_set *push = desc->push[b->set];
      if (!push)
         return false;
      if (memcmp(push->layout_hash, b->set_layout_hash,
                 sizeof(push->layout_hash)) != 0)
         return false;
      if ((uint64_t)b->descriptor_offset_B + sizeof(*addr_out) > push->size)
         return false;
      memcpy(addr_out, push->data + b->descriptor_offset_B, sizeof(*addr_out));
   } else {
      const struct hk_descriptor_set *ds = desc->sets[b->set];
      if (!ds || !ds->layout || !ds->mapped_ptr)
         return false;
      if (b->binding >= ds->layout->binding_count)
         return false;
      const struct hk_descriptor_set_binding_layout *bl =
         &ds->layout->binding[b->binding];
      if (bl->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
          bl->offset != b->descriptor_offset_B)
         return false;
      if (bl->flags &
          (VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
           VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT))
         return false;
      if ((uint64_t)b->descriptor_offset_B + sizeof(*addr_out) > ds->size)
         return false;
      memcpy(addr_out, (const uint8_t *)ds->mapped_ptr + b->descriptor_offset_B,
             sizeof(*addr_out));
   }

   if (addr_out->base_addr == 0 || addr_out->sparse_map_page != 0 ||
       addr_out->size == 0)
      return false;
   if (addr_out->base_addr > UINT64_MAX - addr_out->size)
      return false;
   return true;
}

static bool
va_ranges_disjoint(uint64_t a_base, uint64_t a_len, uint64_t b_base,
                   uint64_t b_len)
{
   if (a_base > UINT64_MAX - a_len || b_base > UINT64_MAX - b_len)
      return false;
   return (a_base + a_len <= b_base) || (b_base + b_len <= a_base);
}

static bool
read_written_push_i32(const struct hk_descriptor_state *desc, uint16_t offset_B,
                      int32_t *val_out)
{
   if ((offset_B & 3u) != 0 || offset_B + 4u > HK_MAX_PUSH_SIZE)
      return false;
   unsigned dword = offset_B >> 2;
   if (dword >= 64 || !(desc->push_written_dwords & (UINT64_C(1) << dword)))
      return false;
   memcpy(val_out, desc->root.push + offset_B, sizeof(*val_out));
   return true;
}

bool
hk_compute_opt_validate_dispatch_core(
   bool alias_free, const struct hk_descriptor_state *desc,
   const struct hk_compute_opt_plan *plan, uint32_t base_x, uint32_t base_y,
   uint32_t base_z, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
   uint32_t max_groups_x, uint32_t max_groups_y, uint32_t *out_groups_x,
   uint32_t *out_groups_y)
{
   if (!alias_free || !desc || !plan || !out_groups_x || !out_groups_y)
      return false;
   if (base_x != 0 || base_y != 0 || base_z != 0)
      return false;
   if (groups_x == 0 || groups_y == 0 || groups_z != 1)
      return false;
   if (plan->tile_w == 0 || plan->tile_h == 0)
      return false;

   int32_t width = 0, height = 0;
   if (!read_written_push_i32(desc, plan->pc_off[0], &width) ||
       !read_written_push_i32(desc, plan->pc_off[1], &height))
      return false;
   if (width < 1 || height < 1)
      return false;
   if ((uint32_t)width > (uint32_t)INT32_MAX - (plan->tile_w + 4u) ||
       (uint32_t)height > (uint32_t)INT32_MAX - (plan->tile_h + 4u))
      return false;
   if ((enum hk_compute_opt_kind)plan->kind !=
       HK_COMPUTE_OPT_BILINEAR_RESIZE_BORDER_U8) {
      if ((uint64_t)groups_x * plan->orig_wg_size[0] < (uint64_t)width ||
          (uint64_t)groups_y * plan->orig_wg_size[1] < (uint64_t)height)
         return false;
   }

   switch ((enum hk_compute_opt_kind)plan->kind) {
   case HK_COMPUTE_OPT_RGBA8_TO_GRAY_U8: {
      int32_t src_stride = 0, dst_stride = 0;
      if (!read_written_push_i32(desc, plan->pc_off[2], &src_stride) ||
          !read_written_push_i32(desc, plan->pc_off[3], &dst_stride))
         return false;
      if (src_stride < width || dst_stride < width)
         return false;
      uint64_t src_elems =
         (uint64_t)(height - 1) * (uint64_t)src_stride + (uint64_t)width;
      uint64_t dst_elems =
         (uint64_t)(height - 1) * (uint64_t)dst_stride + (uint64_t)width;
      if (src_elems > (uint64_t)INT32_MAX || dst_elems > (uint64_t)INT32_MAX)
         return false;
      uint64_t src_bytes = src_elems * 4u;
      uint64_t dst_bytes = dst_elems;
      struct hk_buffer_address src_addr = {0}, dst_addr = {0};
      if (!resolve_ssbo_binding(desc, &plan->bufs[0], &src_addr) ||
          !resolve_ssbo_binding(desc, &plan->bufs[1], &dst_addr))
         return false;
      if (src_addr.size < src_bytes || dst_addr.size < dst_bytes ||
          (src_addr.base_addr & 3u) != 0)
         return false;
      if (!va_ranges_disjoint(dst_addr.base_addr, dst_bytes, src_addr.base_addr,
                              src_bytes))
         return false;
      break;
   }

   case HK_COMPUTE_OPT_CANNY_HYSTERESIS_U8: {
      uint64_t elems = (uint64_t)width * (uint64_t)height;
      if (elems > (uint64_t)INT32_MAX)
         return false;
      struct hk_buffer_address src_addr = {0}, dst_addr = {0};
      if (!resolve_ssbo_binding(desc, &plan->bufs[0], &src_addr) ||
          !resolve_ssbo_binding(desc, &plan->bufs[1], &dst_addr))
         return false;
      if (src_addr.size < elems || dst_addr.size < elems)
         return false;
      if (!va_ranges_disjoint(dst_addr.base_addr, elems, src_addr.base_addr,
                              elems))
         return false;
      break;
   }

   case HK_COMPUTE_OPT_CANNY_NMS_U8: {
      uint64_t elems = (uint64_t)width * (uint64_t)height;
      if (elems > (uint64_t)INT32_MAX)
         return false;
      uint64_t bytes4 = elems * 4u;
      struct hk_buffer_address dir_addr = {0}, mag_addr = {0}, dst_addr = {0};
      if (!resolve_ssbo_binding(desc, &plan->bufs[0], &dir_addr) ||
          !resolve_ssbo_binding(desc, &plan->bufs[1], &mag_addr) ||
          !resolve_ssbo_binding(desc, &plan->bufs[2], &dst_addr))
         return false;
      if (dir_addr.size < bytes4 || mag_addr.size < bytes4 ||
          dst_addr.size < elems ||
          ((dir_addr.base_addr | mag_addr.base_addr) & 3u) != 0)
         return false;
      if (!va_ranges_disjoint(dst_addr.base_addr, elems, dir_addr.base_addr,
                              bytes4) ||
          !va_ranges_disjoint(dst_addr.base_addr, elems, mag_addr.base_addr,
                              bytes4))
         return false;
      break;
   }

   case HK_COMPUTE_OPT_CANNY_FINALIZE_U8: {
      int32_t dst_stride = 0;
      if (!read_written_push_i32(desc, plan->pc_off[2], &dst_stride) ||
          dst_stride < width)
         return false;
      uint64_t src_elems = (uint64_t)width * (uint64_t)height;
      uint64_t dst_elems =
         (uint64_t)(height - 1) * (uint64_t)dst_stride + (uint64_t)width;
      if (src_elems > (uint64_t)INT32_MAX || dst_elems > (uint64_t)INT32_MAX)
         return false;
      uint64_t mag_bytes = src_elems * 4u;
      struct hk_buffer_address st_addr = {0}, mag_addr = {0}, dst_addr = {0};
      if (!resolve_ssbo_binding(desc, &plan->bufs[0], &st_addr) ||
          !resolve_ssbo_binding(desc, &plan->bufs[1], &mag_addr) ||
          !resolve_ssbo_binding(desc, &plan->bufs[2], &dst_addr))
         return false;
      if (st_addr.size < src_elems || mag_addr.size < mag_bytes ||
          dst_addr.size < dst_elems || (mag_addr.base_addr & 3u) != 0)
         return false;
      if (!va_ranges_disjoint(dst_addr.base_addr, dst_elems, st_addr.base_addr,
                              src_elems) ||
          !va_ranges_disjoint(dst_addr.base_addr, dst_elems, mag_addr.base_addr,
                              mag_bytes))
         return false;
      break;
   }

   case HK_COMPUTE_OPT_BILINEAR_RESIZE_BORDER_U8: {
      int32_t dst_stride = 0, border = 0, src_w = 0, src_h = 0, src_stride = 0;
      if (!read_written_push_i32(desc, plan->pc_off[2], &dst_stride) ||
          !read_written_push_i32(desc, plan->pc_off[3], &border) ||
          !read_written_push_i32(desc, plan->pc_off[4], &src_w) ||
          !read_written_push_i32(desc, plan->pc_off[5], &src_h) ||
          !read_written_push_i32(desc, plan->pc_off[6], &src_stride))
         return false;
      if (src_w < 1 || src_h < 1 || src_stride < src_w || border < 0 ||
          border > 1024)
         return false;
      if (width < border + 2 || height < border + 2)
         return false;
      const uint64_t dst_w_pad = (uint64_t)width + 2u * (uint64_t)border;
      const uint64_t dst_h_pad = (uint64_t)height + 2u * (uint64_t)border;
      if ((uint64_t)dst_stride < dst_w_pad ||
          dst_w_pad > (uint64_t)INT32_MAX - 16u ||
          dst_h_pad > (uint64_t)INT32_MAX - 16u)
         return false;
      const uint64_t src_bytes =
         (uint64_t)(src_h - 1) * (uint64_t)src_stride + (uint64_t)src_w;
      const uint64_t dst_bytes =
         (dst_h_pad - 1u) * (uint64_t)dst_stride + dst_w_pad;
      if (src_bytes > (uint64_t)INT32_MAX || dst_bytes > (uint64_t)INT32_MAX)
         return false;

      struct hk_buffer_address src_addr = {0}, dst_addr = {0};
      if (!resolve_ssbo_binding(desc, &plan->bufs[0], &src_addr) ||
          !resolve_ssbo_binding(desc, &plan->bufs[1], &dst_addr))
         return false;
      if (src_addr.size < src_bytes || dst_addr.size < dst_bytes)
         return false;
      if (!va_ranges_disjoint(dst_addr.base_addr, dst_bytes, src_addr.base_addr,
                              src_bytes))
         return false;
      break;
   }

   default:
      return false;
   }

   const uint32_t new_groups_x =
      DIV_ROUND_UP((uint32_t)width, (uint32_t)plan->tile_w);
   const uint32_t new_groups_y =
      DIV_ROUND_UP((uint32_t)height, (uint32_t)plan->tile_h);
   if (new_groups_x == 0 || new_groups_y == 0 ||
       new_groups_x > max_groups_x || new_groups_y > max_groups_y)
      return false;

   *out_groups_x = new_groups_x;
   *out_groups_y = new_groups_y;
   return true;
}

bool
hk_compute_opt_validate_dispatch(const struct hk_device *dev,
                                 const struct hk_descriptor_state *desc,
                                 const struct hk_compute_opt_plan *plan,
                                 uint32_t base_x, uint32_t base_y,
                                 uint32_t base_z, uint32_t groups_x,
                                 uint32_t groups_y, uint32_t groups_z,
                                 uint32_t *out_groups_x, uint32_t *out_groups_y)
{
   if (!dev || !dev->vk.physical)
      return false;

   const uint32_t *max_wg =
      dev->vk.physical->properties.maxComputeWorkGroupCount;

   const bool alias_free = p_atomic_read(&dev->va_alias_resources) == 0 &&
                           p_atomic_read(&dev->alias_live) == 0 &&
                           !hk_cdm_alias_scope(dev);

   return hk_compute_opt_validate_dispatch_core(
      alias_free, desc, plan, base_x, base_y, base_z, groups_x, groups_y,
      groups_z, max_wg[0], max_wg[1], out_groups_x, out_groups_y);
}

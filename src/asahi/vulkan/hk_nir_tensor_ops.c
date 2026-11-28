/* SPDX-License-Identifier: MIT */
#include "hk_nir_tensor_ops.h"
#include "hk_identity_copy4.h"
#include "hk_nir_wgid.h"
#include "compiler/nir/nir_builder.h"

struct tensor_buffer {
   uint8_t set;
   uint16_t binding;
   uint8_t desc_comps;
   uint32_t resource;
};

static nir_scalar
scalar(nir_def *def)
{
   return nir_get_scalar(def, 0);
}

static bool
imm_u32(nir_scalar s, uint32_t expected)
{
   s = nir_scalar_chase_movs(s);
   return s.def && nir_scalar_is_const(s) && nir_scalar_as_uint(s) == expected;
}

static nir_alu_instr *
alu(nir_scalar s, nir_op op)
{
   s = nir_scalar_chase_movs(s);
   if (!s.def)
      return NULL;
   nir_instr *instr = nir_def_instr(s.def);
   if (instr->type != nir_instr_type_alu)
      return NULL;
   nir_alu_instr *a = nir_instr_as_alu(instr);
   return a->op == op ? a : NULL;
}

static nir_scalar
arg(nir_scalar s, unsigned i)
{
   s = nir_scalar_chase_movs(s);
   return nir_scalar_chase_alu_src(s, i);
}

static bool
same(nir_scalar a, nir_scalar b)
{
   a = nir_scalar_chase_movs(a);
   b = nir_scalar_chase_movs(b);
   return a.def && b.def && nir_scalar_equal(a, b);
}

static nir_scalar
strip_zero_add(nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   for (unsigned i = 0; i < 3 && alu(s, nir_op_iadd); ++i) {
      if (imm_u32(arg(s, 0), 0))
         s = nir_scalar_chase_movs(arg(s, 1));
      else if (imm_u32(arg(s, 1), 0))
         s = nir_scalar_chase_movs(arg(s, 0));
      else
         break;
   }
   return s;
}

static nir_intrinsic_instr *
intr(nir_def *def, nir_intrinsic_op op)
{
   if (!def || nir_def_instr(def)->type != nir_instr_type_intrinsic)
      return NULL;
   nir_intrinsic_instr *i = nir_instr_as_intrinsic(nir_def_instr(def));
   return i->intrinsic == op ? i : NULL;
}

static nir_phi_instr *
phi_of(nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   return s.def && !s.comp && s.def->num_components == 1 &&
          nir_def_instr(s.def)->type == nir_instr_type_phi
             ? nir_instr_as_phi(nir_def_instr(s.def))
             : NULL;
}

static nir_scalar
phi_in(nir_phi_instr *p, nir_block *block)
{
   nir_phi_src *edge = p ? nir_phi_get_src_from_block(p, block) : NULL;
   return edge ? scalar(edge->src.ssa) : (nir_scalar){0};
}

static nir_if *
as_if(nir_cf_node *node)
{
   return node && node->type == nir_cf_node_if ? nir_cf_node_as_if(node) : NULL;
}

static bool
inside_arm(nir_cf_node *node, nir_if *branch, bool then)
{
   struct exec_list *arm = then ? &branch->then_list : &branch->else_list;
   for (nir_cf_node *cur = node; cur && cur != &branch->cf_node; cur = cur->parent) {
      if (cur->parent == &branch->cf_node) {
         foreach_list_typed(nir_cf_node, item, node, arm) {
            if (item == cur)
               return true;
         }
         return false;
      }
   }
   return false;
}

static bool
acyclic(const struct exec_list *list, unsigned depth)
{
   if (!depth)
      return false;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop)
         return false;
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (!acyclic(&nif->then_list, depth - 1) ||
             !acyclic(&nif->else_list, depth - 1))
            return false;
      }
   }
   return true;
}

static bool
is_coord_id(nir_scalar s, unsigned axis, unsigned local_size)
{
   s = nir_scalar_chase_movs(s);
   if (!s.def || s.comp != axis)
      return false;
   return intr(s.def, nir_intrinsic_load_global_invocation_id) ||
          (local_size == 1 && intr(s.def, nir_intrinsic_load_workgroup_id));
}

static bool
match_coord(nir_scalar s, unsigned axis, unsigned local_size)
{
   s = strip_zero_add(s);
   if (is_coord_id(s, axis, local_size))
      return true;
   if (!alu(s, nir_op_iadd))
      return false;
   nir_scalar a = strip_zero_add(arg(s, 0));
   nir_scalar b = strip_zero_add(arg(s, 1));
   if (is_coord_id(a, axis, local_size) &&
       hk_nir_match_base_wg_scaled(b, axis, local_size))
      return true;
   if (is_coord_id(b, axis, local_size) &&
       hk_nir_match_base_wg_scaled(a, axis, local_size))
      return true;
   return false;
}

static bool
match_push_any(nir_scalar s, uint16_t *off_out, uint16_t *range_out)
{
   s = nir_scalar_chase_movs(s);
   nir_intrinsic_instr *op = intr(s.def, nir_intrinsic_load_push_constant);
   if (!op || op->def.bit_size != 32 || s.comp >= op->def.num_components ||
       nir_intrinsic_access(op) != 0)
      return false;
   nir_scalar off_s = nir_scalar_chase_movs(scalar(op->src[0].ssa));
   if (!off_s.def || !nir_scalar_is_const(off_s))
      return false;
   const uint64_t off = (uint64_t)nir_intrinsic_base(op) +
                        nir_scalar_as_uint(off_s) + (uint64_t)s.comp * 4u;
   if ((off & 3) || off + 4 > HK_COMPUTE_OPT_PUSH_BYTES)
      return false;
   if (off_out)
      *off_out = (uint16_t)off;
   if (range_out) {
      uint64_t rng = MAX2((uint64_t)nir_intrinsic_base(op) + nir_intrinsic_range(op),
                          off + 4);
      if (rng > HK_COMPUTE_OPT_PUSH_BYTES)
         rng = off + 4;
      *range_out = (uint16_t)rng;
   }
   return true;
}

static bool
match_push_eq(nir_scalar s, uint16_t expected_off)
{
   uint16_t off = 0;
   return match_push_any(s, &off, NULL) && off == expected_off;
}

static bool
match_div(nir_scalar div, nir_scalar *num, nir_scalar *den)
{
   div = nir_scalar_chase_movs(div);
   if (alu(div, nir_op_idiv) || alu(div, nir_op_udiv)) {
      *num = arg(div, 0);
      *den = arg(div, 1);
      return true;
   }
   return false;
}

static bool
match_div_mul(nir_scalar prod, nir_scalar num, nir_scalar *den)
{
   if (!alu(prod, nir_op_imul))
      return false;
   nir_scalar a = arg(prod, 0), b = arg(prod, 1);
   nir_scalar qn, qd;
   if (match_div(a, &qn, &qd) && same(qn, num) && same(qd, b)) {
      *den = qd;
      return true;
   }
   if (match_div(b, &qn, &qd) && same(qn, num) && same(qd, a)) {
      *den = qd;
      return true;
   }
   return false;
}

static bool
match_mod(nir_scalar rem, nir_scalar num, nir_scalar *den)
{
   rem = nir_scalar_chase_movs(rem);
   if (alu(rem, nir_op_irem) || alu(rem, nir_op_imod) || alu(rem, nir_op_umod)) {
      if (!same(arg(rem, 0), num))
         return false;
      *den = arg(rem, 1);
      return true;
   }
   if (alu(rem, nir_op_isub))
      return same(arg(rem, 0), num) && match_div_mul(arg(rem, 1), num, den);
   if (alu(rem, nir_op_iadd)) {
      nir_scalar a = arg(rem, 0), b = arg(rem, 1);
      if (same(a, num) && alu(b, nir_op_ineg) && match_div_mul(arg(b, 0), num, den))
         return true;
      if (same(b, num) && alu(a, nir_op_ineg) && match_div_mul(arg(a, 0), num, den))
         return true;
   }
   return false;
}

static bool
match_short_or(nir_scalar s, nir_scalar *lhs, nir_scalar *rhs)
{
   s = nir_scalar_chase_movs(s);
   if (alu(s, nir_op_ior)) {
      *lhs = arg(s, 0);
      *rhs = arg(s, 1);
      return true;
   }
   if (alu(s, nir_op_bcsel)) {
      nir_scalar c = arg(s, 0), t = arg(s, 1), f = arg(s, 2);
      if (imm_u32(t, 1)) {
         *lhs = c;
         *rhs = f;
         return true;
      }
      if (imm_u32(f, 1)) {
         if (alu(c, nir_op_inot)) {
            *lhs = arg(c, 0);
            *rhs = t;
            return true;
         }
         /* bcsel(ilt(a, b), rhs, true) == !(a < b) || rhs == (a >= b) || rhs */
         if (alu(c, nir_op_ilt) || alu(c, nir_op_ult)) {
            *lhs = c; /* caller handles negated sense via bcsel helper */
            return false;
         }
      }
   }
   nir_phi_instr *p = phi_of(s);
   if (!p)
      return false;
   nir_cf_node *prev = nir_cf_node_prev(&p->instr.block->cf_node);
   nir_if *nif = as_if(prev);
   if (!nif)
      return false;
   nir_scalar c = scalar(nif->condition.ssa);
   nir_scalar tv = phi_in(p, nir_if_last_then_block(nif));
   nir_scalar fv = phi_in(p, nir_if_last_else_block(nif));
   if (alu(c, nir_op_inot) && same(fv, arg(c, 0))) {
      *lhs = arg(c, 0);
      *rhs = tv;
      return true;
   }
   if (imm_u32(tv, 1)) {
      *lhs = c;
      *rhs = fv;
      return true;
   }
   if (imm_u32(fv, 1) && alu(c, nir_op_inot)) {
      *lhs = arg(c, 0);
      *rhs = tv;
      return true;
   }
   return false;
}

static bool
match_short_and(nir_scalar s, nir_scalar *lhs, nir_scalar *rhs)
{
   s = nir_scalar_chase_movs(s);
   if (alu(s, nir_op_iand)) {
      *lhs = arg(s, 0);
      *rhs = arg(s, 1);
      return true;
   }
   if (alu(s, nir_op_bcsel)) {
      nir_scalar c = arg(s, 0), t = arg(s, 1), f = arg(s, 2);
      if (imm_u32(f, 0)) {
         *lhs = c;
         *rhs = t;
         return true;
      }
      if (imm_u32(t, 0) && alu(c, nir_op_inot)) {
         *lhs = arg(c, 0);
         *rhs = f;
         return true;
      }
   }
   nir_phi_instr *p = phi_of(s);
   if (!p)
      return false;
   nir_cf_node *prev = nir_cf_node_prev(&p->instr.block->cf_node);
   nir_if *nif = as_if(prev);
   if (!nif)
      return false;
   nir_scalar c = scalar(nif->condition.ssa);
   nir_scalar tv = phi_in(p, nir_if_last_then_block(nif));
   nir_scalar fv = phi_in(p, nir_if_last_else_block(nif));
   if (same(fv, c) || imm_u32(fv, 0)) {
      *lhs = c;
      *rhs = tv;
      return true;
   }
   return false;
}

/* Match `coord < lim` (want_in_bounds=true) or `coord >= lim` (false),
 * including signed/unsigned variants and flipped operand order. */
static bool
match_bound_cmp(nir_scalar s, bool want_in_bounds, const uint16_t ls[3],
                unsigned *axis_out, nir_scalar *coord_out, nir_scalar *lim_out)
{
   s = nir_scalar_chase_movs(s);
   while (alu(s, nir_op_inot)) {
      want_in_bounds = !want_in_bounds;
      s = nir_scalar_chase_movs(arg(s, 0));
   }
   nir_scalar c = {0}, lim = {0};
   if (want_in_bounds) {
      if (alu(s, nir_op_ilt) || alu(s, nir_op_ult)) {
         c = arg(s, 0);
         lim = arg(s, 1);
      } else {
         return false;
      }
   } else {
      if (alu(s, nir_op_ige) || alu(s, nir_op_uge)) {
         c = arg(s, 0);
         lim = arg(s, 1);
      } else {
         return false;
      }
   }
   if (match_coord(c, 0, ls[0])) {
      *axis_out = 0;
      *coord_out = strip_zero_add(c);
      *lim_out = lim;
      return true;
   }
   if (match_coord(c, 1, ls[1])) {
      *axis_out = 1;
      *coord_out = strip_zero_add(c);
      *lim_out = lim;
      return true;
   }
   return false;
}

static bool
is_bool_negation(nir_scalar a, nir_scalar b)
{
   a = nir_scalar_chase_movs(a);
   b = nir_scalar_chase_movs(b);
   if (!a.def || !b.def)
      return false;
   if (alu(a, nir_op_inot) && same(arg(a, 0), b))
      return true;
   if (alu(b, nir_op_inot) && same(arg(b, 0), a))
      return true;
   if ((alu(a, nir_op_ilt) && alu(b, nir_op_ige)) ||
       (alu(a, nir_op_ige) && alu(b, nir_op_ilt)) ||
       (alu(a, nir_op_ult) && alu(b, nir_op_uge)) ||
       (alu(a, nir_op_uge) && alu(b, nir_op_ult))) {
      return same(arg(a, 0), arg(b, 0)) && same(arg(a, 1), arg(b, 1));
   }
   return false;
}

static bool
match_2d_guard(nir_if *branch, bool body_in_then, const uint16_t ls[3],
               nir_scalar *gx_out, nir_scalar *gy_out,
               nir_scalar *xlim_out, nir_scalar *ylim_out)
{
   nir_scalar cond = nir_scalar_chase_movs(scalar(branch->condition.ssa));
   bool in_bounds = body_in_then;
   while (alu(cond, nir_op_inot)) {
      in_bounds = !in_bounds;
      cond = nir_scalar_chase_movs(arg(cond, 0));
   }

   nir_scalar a = {0}, b = {0};
   bool a_in = in_bounds, b_in = in_bounds;
   bool matched = false;
   if (in_bounds) {
      if (match_short_and(cond, &a, &b)) {
         matched = true;
      } else if (alu(cond, nir_op_bcsel) && imm_u32(arg(cond, 1), 0)) {
         /* bcsel(oob_0, false, in_1) == (!oob_0 && in_1) */
         a = arg(cond, 0);
         b = arg(cond, 2);
         a_in = false;
         b_in = true;
         matched = true;
      } else {
         nir_phi_instr *p = phi_of(cond);
         nir_if *nif = p ? as_if(nir_cf_node_prev(&p->instr.block->cf_node)) : NULL;
         if (nif) {
            nir_scalar c = scalar(nif->condition.ssa);
            nir_scalar tv = phi_in(p, nir_if_last_then_block(nif));
            nir_scalar fv = phi_in(p, nir_if_last_else_block(nif));
            if (imm_u32(tv, 0) || is_bool_negation(c, tv)) {
               a = c;
               b = fv;
               a_in = false;
               b_in = true;
               matched = true;
            }
         }
      }
   } else {
      if (match_short_or(cond, &a, &b)) {
         matched = true;
      } else if (alu(cond, nir_op_bcsel) && imm_u32(arg(cond, 2), 1)) {
         /* bcsel(in_bounds_0, oob_1, true) == (!in_bounds_0 || oob_1) */
         a = arg(cond, 0);
         b = arg(cond, 1);
         a_in = true;
         b_in = false;
         matched = true;
      } else {
         nir_phi_instr *p = phi_of(cond);
         nir_if *nif = p ? as_if(nir_cf_node_prev(&p->instr.block->cf_node)) : NULL;
         if (nif) {
            nir_scalar c = scalar(nif->condition.ssa);
            nir_scalar tv = phi_in(p, nir_if_last_then_block(nif));
            nir_scalar fv = phi_in(p, nir_if_last_else_block(nif));
            if (imm_u32(fv, 1) || is_bool_negation(c, fv)) {
               /* if (in_bounds_0) { y_oob } else { true / !in_bounds_0 } */
               a = c;
               b = tv;
               a_in = true;
               b_in = false;
               matched = true;
            } else if (imm_u32(tv, 1) || same(tv, c)) {
               a = c;
               b = fv;
               a_in = false;
               b_in = false;
               matched = true;
            }
         }
      }
   }
   if (!matched)
      return false;

   unsigned ax0 = 99, ax1 = 99;
   nir_scalar c0, l0, c1, l1;
   if (!match_bound_cmp(a, a_in, ls, &ax0, &c0, &l0) ||
       !match_bound_cmp(b, b_in, ls, &ax1, &c1, &l1) ||
       ax0 == ax1)
      return false;

   if (ax0 == 0) {
      *gx_out = c0;
      *xlim_out = l0;
      *gy_out = c1;
      *ylim_out = l1;
   } else {
      *gx_out = c1;
      *xlim_out = l1;
      *gy_out = c0;
      *ylim_out = l0;
   }
   return true;
}

static bool
match_buffer_32(nir_intrinsic_instr *op, struct tensor_buffer *out,
                nir_scalar *index)
{
   nir_deref_instr *d = nir_src_as_deref(op->src[0]);
   if (!d || d->deref_type != nir_deref_type_array ||
       !nir_deref_mode_is(d, nir_var_mem_ssbo))
      return false;
   const struct glsl_type *elem = d->type;
   if (elem != glsl_float_type() && elem != glsl_int_type() &&
       elem != glsl_uint_type())
      return false;
   *index = strip_zero_add(scalar(d->arr.index.ssa));

   d = nir_deref_instr_parent(d);
   if (!d || d->deref_type != nir_deref_type_struct || d->strct.index != 0)
      return false;
   const struct glsl_type *parent_t = nir_deref_instr_parent(d)
                                         ? nir_deref_instr_parent(d)->type
                                         : NULL;
   if (!parent_t || !glsl_type_is_struct_or_ifc(parent_t) ||
       glsl_get_struct_field_offset(parent_t, 0) > 0)
      return false;
   if (glsl_get_explicit_stride(d->type) != 4)
      return false;

   d = nir_deref_instr_parent(d);
   if (!d || d->deref_type != nir_deref_type_cast || d->cast.ptr_stride != 0)
      return false;
   nir_intrinsic_instr *desc = intr(d->parent.ssa, nir_intrinsic_load_vulkan_descriptor);
   if (!desc || nir_intrinsic_desc_type(desc) != nir_descriptor_type_storage_buffer)
      return false;
   nir_intrinsic_instr *res = intr(desc->src[0].ssa, nir_intrinsic_vulkan_resource_index);
   if (!res || nir_intrinsic_desc_type(res) != nir_descriptor_type_storage_buffer ||
       !imm_u32(scalar(res->src[0].ssa), 0) ||
       res->def.num_components < 3 || res->def.num_components > 5)
      return false;
   const unsigned set = nir_intrinsic_desc_set(res);
   const unsigned binding = nir_intrinsic_binding(res);
   if (set > UINT8_MAX || binding > UINT16_MAX)
      return false;

   *out = (struct tensor_buffer){
      .set = (uint8_t)set,
      .binding = (uint16_t)binding,
      .desc_comps = (uint8_t)res->def.num_components,
      .resource = nir_intrinsic_resource_type(res),
   };
   return true;
}

static bool
same_buf(struct tensor_buffer a, struct tensor_buffer b)
{
   return a.set == b.set && a.binding == b.binding;
}

static bool
flatten_iadd(nir_scalar s, nir_scalar stop, nir_scalar *terms, unsigned *count,
             unsigned max_terms, unsigned depth)
{
   s = strip_zero_add(s);
   if (depth < 8 && (!stop.def || !same(s, stop)) &&
       s.def && s.def->num_components == 1 && alu(s, nir_op_iadd)) {
      return flatten_iadd(arg(s, 0), stop, terms, count, max_terms, depth + 1) &&
             flatten_iadd(arg(s, 1), stop, terms, count, max_terms, depth + 1);
   }
   if (*count >= max_terms)
      return false;
   terms[(*count)++] = s;
   return true;
}

static bool
match_mul_push(nir_scalar s, nir_scalar var, uint16_t *push_off, uint16_t *pb)
{
   s = nir_scalar_chase_movs(s);
   if (!alu(s, nir_op_imul))
      return false;
   nir_scalar a = strip_zero_add(arg(s, 0));
   nir_scalar b = strip_zero_add(arg(s, 1));
   uint16_t off = 0, rng = 0;
   if (same(a, var) && match_push_any(b, &off, &rng)) {
      *push_off = off;
      *pb = MAX2(*pb, rng);
      return true;
   }
   if (same(b, var) && match_push_any(a, &off, &rng)) {
      *push_off = off;
      *pb = MAX2(*pb, rng);
      return true;
   }
   return false;
}

static bool
match_push_or_mul2(nir_scalar s, uint16_t *off0, uint16_t *off1, uint16_t *pb)
{
   s = nir_scalar_chase_movs(s);
   uint16_t r0 = 0, r1 = 0;
   if (match_push_any(s, off0, &r0)) {
      *off1 = UINT16_MAX;
      *pb = MAX2(*pb, r0);
      return true;
   }
   if (alu(s, nir_op_imul) &&
       match_push_any(arg(s, 0), off0, &r0) &&
       match_push_any(arg(s, 1), off1, &r1)) {
      *pb = MAX2(*pb, MAX2(r0, r1));
      return true;
   }
   return false;
}

static bool
strip_mul_push_eq(nir_scalar s, uint16_t push_off, nir_scalar *other_out)
{
   s = nir_scalar_chase_movs(s);
   if (!alu(s, nir_op_imul))
      return false;
   nir_scalar a = strip_zero_add(arg(s, 0));
   nir_scalar b = strip_zero_add(arg(s, 1));
   if (match_push_eq(b, push_off)) {
      *other_out = a;
      return true;
   }
   if (match_push_eq(a, push_off)) {
      *other_out = b;
      return true;
   }
   return false;
}

/* Extract `row` from `row * xlim` where `xlim` is either `push[d0]` or
 * `push[d0] * push[d1]` (accepting right-associated `row * (d0 * d1)` and
 * left-associated `(row * d0) * d1` / `(row * d1) * d0`). */
static bool
extract_mul_dim(nir_scalar s, nir_scalar xlim, uint16_t d0, uint16_t d1,
                nir_scalar *row_out)
{
   s = nir_scalar_chase_movs(s);
   if (!alu(s, nir_op_imul))
      return false;
   nir_scalar a = strip_zero_add(arg(s, 0));
   nir_scalar b = strip_zero_add(arg(s, 1));
   if (same(a, xlim)) {
      *row_out = b;
      return true;
   }
   if (same(b, xlim)) {
      *row_out = a;
      return true;
   }
   if (d1 == UINT16_MAX)
      return strip_mul_push_eq(s, d0, row_out);

   nir_scalar inner = {0};
   if (strip_mul_push_eq(s, d1, &inner) && strip_mul_push_eq(inner, d0, row_out))
      return true;
   if (strip_mul_push_eq(s, d0, &inner) && strip_mul_push_eq(inner, d1, row_out))
      return true;
   return false;
}

static bool
match_mul_dim(nir_scalar s, nir_scalar row, nir_scalar xlim,
              uint16_t d0, uint16_t d1)
{
   nir_scalar extracted = {0};
   return extract_mul_dim(s, xlim, d0, d1, &extracted) && same(extracted, row);
}

static bool
match_src_2d_linear(nir_scalar src_idx, nir_scalar gy, nir_scalar gx,
                    nir_scalar xlim, nir_scalar q, nir_scalar r,
                    uint16_t x0_off, uint16_t x1_off, uint16_t dim_c_off)
{
   nir_scalar terms[4];
   unsigned nt = 0;
   if (!flatten_iadd(src_idx, gx, terms, &nt, 4, 0))
      return false;
   if (nt == 2) {
      if ((match_mul_dim(terms[0], gy, xlim, x0_off, x1_off) && same(terms[1], gx)) ||
          (match_mul_dim(terms[1], gy, xlim, x0_off, x1_off) && same(terms[0], gx)))
         return true;
   } else if (nt == 3 && q.def && r.def && dim_c_off != UINT16_MAX) {
      bool has_gy = false, has_q = false, has_r = false;
      for (unsigned i = 0; i < 3; ++i) {
         if (!has_gy && match_mul_dim(terms[i], gy, xlim, x0_off, x1_off))
            has_gy = true;
         else if (!has_r && same(terms[i], r))
            has_r = true;
         else if (!has_q && alu(terms[i], nir_op_imul)) {
            nir_scalar a = strip_zero_add(arg(terms[i], 0));
            nir_scalar b = strip_zero_add(arg(terms[i], 1));
            if ((same(a, q) && match_push_eq(b, dim_c_off)) ||
                (same(b, q) && match_push_eq(a, dim_c_off)))
               has_q = true;
         }
      }
      if (has_gy && has_q && has_r)
         return true;
   }
   return false;
}

/* Scan a simple 1-load, 1-store acyclic 2D-guarded compute shader. */
static bool
scan_simple_2d_copy(nir_shader *nir,
                    struct tensor_buffer *src_out, struct tensor_buffer *dst_out,
                    nir_scalar *src_idx_out, nir_scalar *dst_idx_out,
                    nir_scalar *gx_out, nir_scalar *gy_out,
                    nir_scalar *xlim_out, nir_scalar *ylim_out)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE || nir->info.workgroup_size_variable ||
       !hk_identity_copy4_supported_local(nir->info.workgroup_size) ||
       nir->info.shared_size != 0 || nir->info.num_ssbos != 2 ||
       nir->num_uniforms > HK_COMPUTE_OPT_PUSH_BYTES)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_function_impl(other, nir) {
      if (other != impl)
         return false;
   }
   if (!acyclic(&impl->body, 8))
      return false;

   nir_intrinsic_instr *load = NULL, *store = NULL;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_deref:
         case nir_instr_type_phi:
         case nir_instr_type_load_const:
         case nir_instr_type_alu:
            continue;
         case nir_instr_type_intrinsic:
            break;
         default:
            return false;
         }
         nir_intrinsic_instr *op = nir_instr_as_intrinsic(instr);
         switch (op->intrinsic) {
         case nir_intrinsic_load_global_invocation_id:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_base_workgroup_id:
         case nir_intrinsic_load_push_constant:
         case nir_intrinsic_vulkan_resource_index:
         case nir_intrinsic_load_vulkan_descriptor:
            break;
         case nir_intrinsic_load_deref:
            if (load || op->def.bit_size != 32 || op->def.num_components != 1 ||
                (nir_intrinsic_access(op) &
                 ~(ACCESS_CAN_REORDER | ACCESS_RESTRICT | ACCESS_IN_BOUNDS |
                   ACCESS_CAN_SPECULATE)) != ACCESS_NON_WRITEABLE)
               return false;
            load = op;
            break;
         case nir_intrinsic_store_deref:
            if (store || nir_intrinsic_write_mask(op) != 1 ||
                op->src[1].ssa->bit_size != 32 ||
                op->src[1].ssa->num_components != 1 ||
                (nir_intrinsic_access(op) &
                 ~(ACCESS_NON_READABLE | ACCESS_RESTRICT | ACCESS_IN_BOUNDS |
                   ACCESS_CAN_SPECULATE)))
               return false;
            store = op;
            break;
         default:
            return false;
         }
      }
   }

   if (!load || !store)
      return false;
   nir_scalar val = nir_scalar_chase_movs(scalar(store->src[1].ssa));
   if (val.def != &load->def || val.comp != 0)
      return false;

   nir_block *wb = store->instr.block;
   nir_if *guard = as_if(wb->cf_node.parent);
   if (!guard)
      return false;
   const bool in_then = inside_arm(&wb->cf_node, guard, true);
   const bool in_else = inside_arm(&wb->cf_node, guard, false);
   if (in_then == in_else || !inside_arm(&load->instr.block->cf_node, guard, in_then))
      return false;

   if (guard->cf_node.parent == &impl->cf_node) {
      if (!match_2d_guard(guard, in_then, nir->info.workgroup_size,
                          gx_out, gy_out, xlim_out, ylim_out))
         return false;
   } else {
      nir_if *outer = as_if(guard->cf_node.parent);
      if (!outer || outer->cf_node.parent != &impl->cf_node)
         return false;
      const bool out_then = inside_arm(&guard->cf_node, outer, true);
      const bool out_else = inside_arm(&guard->cf_node, outer, false);
      if (out_then == out_else)
         return false;
      unsigned ax0 = 99, ax1 = 99;
      nir_scalar c0, l0, c1, l1;
      if (!match_bound_cmp(scalar(outer->condition.ssa), out_then,
                           nir->info.workgroup_size, &ax0, &c0, &l0) ||
          !match_bound_cmp(scalar(guard->condition.ssa), in_then,
                           nir->info.workgroup_size, &ax1, &c1, &l1) ||
          ax0 == ax1)
         return false;
      if (ax0 == 0) {
         *gx_out = c0;
         *xlim_out = l0;
         *gy_out = c1;
         *ylim_out = l1;
      } else {
         *gx_out = c1;
         *xlim_out = l1;
         *gy_out = c0;
         *ylim_out = l0;
      }
   }

   if (!match_buffer_32(load, src_out, src_idx_out) ||
       !match_buffer_32(store, dst_out, dst_idx_out) ||
       dst_out->resource != nir_resource_type_read_write_storage_buffer ||
       same_buf(*src_out, *dst_out))
      return false;

   return true;
}

static bool
match_concat_x(nir_scalar gx, nir_scalar gy, nir_scalar xlim, nir_scalar ylim,
               nir_scalar src_idx, nir_scalar dst_idx,
               struct hk_tensor_op_info *info)
{
   uint16_t pb = 0;
   uint16_t x0 = UINT16_MAX, x1 = UINT16_MAX;
   uint16_t y0 = UINT16_MAX, y1 = UINT16_MAX;
   if (!match_push_or_mul2(xlim, &x0, &x1, &pb) || x1 == UINT16_MAX)
      return false;
   if (!match_push_or_mul2(ylim, &y0, &y1, &pb))
      return false;

   /* dst_idx = q * dst_stride_q + gy * dst_stride_y + [dst_off_x] + r */
   nir_scalar terms[4];
   unsigned nt = 0;
   if (!flatten_iadd(dst_idx, (nir_scalar){0}, terms, &nt, 4, 0) || nt < 3 || nt > 4)
      return false;

   nir_scalar q_s = {0}, r_s = {0};
   uint16_t dim_c_off = UINT16_MAX, dim_q_off = UINT16_MAX;
   uint16_t dst_stride_q = UINT16_MAX, dst_stride_y = UINT16_MAX;
   uint16_t dst_off_x = UINT16_MAX;

   /* Identify r = mod(gx, dim_c) first. */
   int r_idx = -1;
   for (unsigned i = 0; i < nt; ++i) {
      nir_scalar den = {0};
      uint16_t off = 0, rng = 0;
      if (match_mod(terms[i], gx, &den) && match_push_any(den, &off, &rng)) {
         if (r_idx >= 0)
            return false;
         r_idx = (int)i;
         r_s = terms[i];
         dim_c_off = off;
         pb = MAX2(pb, rng);
      }
   }
   if (r_idx < 0)
      return false;

   if (dim_c_off == x1)
      dim_q_off = x0;
   else if (dim_c_off == x0)
      dim_q_off = x1;
   else
      return false;

   for (unsigned i = 0; i < nt; ++i) {
      if ((int)i == r_idx)
         continue;
      nir_scalar t = terms[i];
      if (dst_stride_y == UINT16_MAX && match_mul_push(t, gy, &dst_stride_y, &pb))
         continue;
      if (dst_stride_q == UINT16_MAX && alu(t, nir_op_imul)) {
         nir_scalar a = strip_zero_add(arg(t, 0));
         nir_scalar b = strip_zero_add(arg(t, 1));
         nir_scalar qn = {0}, qd = {0};
         uint16_t off = 0, rng = 0;
         if (match_div(a, &qn, &qd) && same(qn, gx) &&
             match_push_eq(qd, dim_c_off) && match_push_any(b, &off, &rng)) {
            q_s = a;
            dst_stride_q = off;
            pb = MAX2(pb, rng);
            continue;
         }
         if (match_div(b, &qn, &qd) && same(qn, gx) &&
             match_push_eq(qd, dim_c_off) && match_push_any(a, &off, &rng)) {
            q_s = b;
            dst_stride_q = off;
            pb = MAX2(pb, rng);
            continue;
         }
      }
      uint16_t off = 0, rng = 0;
      if (dst_off_x == UINT16_MAX && match_push_any(t, &off, &rng)) {
         dst_off_x = off;
         pb = MAX2(pb, rng);
         continue;
      }
      return false;
   }

   if (dst_stride_q == UINT16_MAX || dst_stride_y == UINT16_MAX)
      return false;

   if (!match_src_2d_linear(src_idx, gy, gx, xlim, q_s, r_s, x0, x1, dim_c_off))
      return false;

   info->kind = HK_TENSOR_OP_CONCAT_X;
   info->push_bytes = pb;
   info->dim_n_off = (y1 != UINT16_MAX) ? y0 : UINT16_MAX;
   info->dim_h_off = (y1 != UINT16_MAX) ? y1 : y0;
   info->dim_q_off = dim_q_off;
   info->dim_c_off = dim_c_off;
   info->dst_stride_y_off = dst_stride_y;
   info->dst_off_x_off = dst_off_x;
   info->dst_stride_q_off = dst_stride_q;
   info->scale_y_off = UINT16_MAX;
   return true;
}

static bool
match_concat_y(nir_scalar gx, nir_scalar gy, nir_scalar xlim, nir_scalar ylim,
               nir_scalar src_idx, nir_scalar dst_idx,
               struct hk_tensor_op_info *info)
{
   uint16_t pb = 0;
   uint16_t x0 = UINT16_MAX, x1 = UINT16_MAX;
   uint16_t y0 = UINT16_MAX, y1 = UINT16_MAX;
   if (!match_push_or_mul2(xlim, &x0, &x1, &pb))
      return false;
   if (!match_push_or_mul2(ylim, &y0, &y1, &pb) || y1 == UINT16_MAX)
      return false;

   if (!match_src_2d_linear(src_idx, gy, gx, xlim, (nir_scalar){0}, (nir_scalar){0},
                            x0, x1, UINT16_MAX))
      return false;

   /* dst_idx = dst_row * xlim + gx */
   nir_scalar top[2];
   unsigned ntop = 0;
   if (!flatten_iadd(dst_idx, gx, top, &ntop, 2, 0) || ntop != 2)
      return false;

   nir_scalar mul_term = {0};
   if (same(top[0], gx))
      mul_term = top[1];
   else if (same(top[1], gx))
      mul_term = top[0];
   else
      return false;

   nir_scalar dst_row = {0};
   if (!extract_mul_dim(mul_term, xlim, x0, x1, &dst_row))
      return false;

   /* dst_row = n * dst_stride_n + [dst_y_off] + h, where n = gy / dim_h, h = gy % dim_h */
   nir_scalar rterms[3];
   unsigned nr = 0;
   if (!flatten_iadd(dst_row, (nir_scalar){0}, rterms, &nr, 3, 0) || nr < 2 || nr > 3)
      return false;

   uint16_t dim_h_off = UINT16_MAX, dim_n_off = UINT16_MAX;
   uint16_t dst_stride_n = UINT16_MAX, dst_y_off = UINT16_MAX;
   int h_idx = -1;
   for (unsigned i = 0; i < nr; ++i) {
      nir_scalar den = {0};
      uint16_t off = 0, rng = 0;
      if (match_mod(rterms[i], gy, &den) && match_push_any(den, &off, &rng)) {
         if (h_idx >= 0)
            return false;
         h_idx = (int)i;
         dim_h_off = off;
         pb = MAX2(pb, rng);
      }
   }
   if (h_idx < 0)
      return false;

   if (dim_h_off == y1)
      dim_n_off = y0;
   else if (dim_h_off == y0)
      dim_n_off = y1;
   else
      return false;

   for (unsigned i = 0; i < nr; ++i) {
      if ((int)i == h_idx)
         continue;
      nir_scalar t = rterms[i];
      if (dst_stride_n == UINT16_MAX && alu(t, nir_op_imul)) {
         nir_scalar a = strip_zero_add(arg(t, 0));
         nir_scalar b = strip_zero_add(arg(t, 1));
         nir_scalar qn = {0}, qd = {0};
         uint16_t off = 0, rng = 0;
         if (match_div(a, &qn, &qd) && same(qn, gy) &&
             match_push_eq(qd, dim_h_off) && match_push_any(b, &off, &rng)) {
            dst_stride_n = off;
            pb = MAX2(pb, rng);
            continue;
         }
         if (match_div(b, &qn, &qd) && same(qn, gy) &&
             match_push_eq(qd, dim_h_off) && match_push_any(a, &off, &rng)) {
            dst_stride_n = off;
            pb = MAX2(pb, rng);
            continue;
         }
      }
      uint16_t off = 0, rng = 0;
      if (dst_y_off == UINT16_MAX && match_push_any(t, &off, &rng)) {
         dst_y_off = off;
         pb = MAX2(pb, rng);
         continue;
      }
      return false;
   }

   if (dst_stride_n == UINT16_MAX)
      return false;

   info->kind = HK_TENSOR_OP_CONCAT_Y;
   info->push_bytes = pb;
   info->dim_n_off = dim_n_off;
   info->dim_h_off = dim_h_off;
   info->dim_q_off = (x1 != UINT16_MAX) ? x0 : UINT16_MAX;
   info->dim_c_off = (x1 != UINT16_MAX) ? x1 : x0;
   info->dst_stride_y_off = dst_stride_n;
   info->dst_off_x_off = dst_y_off;
   info->dst_stride_q_off = UINT16_MAX;
   info->scale_y_off = UINT16_MAX;
   return true;
}

static bool
match_floor_scale(nir_scalar s, nir_scalar coord, uint16_t *scale_off,
                  uint16_t *pb, unsigned *fmul_ctrl_out)
{
   s = nir_scalar_chase_movs(s);
   if (!alu(s, nir_op_f2i32))
      return false;
   nir_scalar fl = nir_scalar_chase_movs(arg(s, 0));
   if (!alu(fl, nir_op_ffloor))
      return false;
   nir_scalar mul_s = nir_scalar_chase_movs(arg(fl, 0));
   nir_alu_instr *mul = alu(mul_s, nir_op_fmul);
   if (!mul)
      return false;
   nir_scalar a = nir_scalar_chase_movs(arg(mul_s, 0));
   nir_scalar b = nir_scalar_chase_movs(arg(mul_s, 1));
   uint16_t off = 0, rng = 0;
   if (alu(a, nir_op_i2f32) && same(arg(a, 0), coord) &&
       match_push_any(b, &off, &rng)) {
      *scale_off = off;
      *pb = MAX2(*pb, rng);
      *fmul_ctrl_out = mul->fp_math_ctrl;
      return true;
   }
   if (alu(b, nir_op_i2f32) && same(arg(b, 0), coord) &&
       match_push_any(a, &off, &rng)) {
      *scale_off = off;
      *pb = MAX2(*pb, rng);
      *fmul_ctrl_out = mul->fp_math_ctrl;
      return true;
   }
   return false;
}

static bool
match_nhwc_idx(nir_scalar idx, nir_scalar *y_out, nir_scalar *x_out,
               nir_scalar c_var, uint16_t ch_off, uint16_t *w_off, uint16_t *pb)
{
   /* ((y * W) + x) * ch + c  OR  (y * W * ch) + (x * ch) + c */
   nir_scalar top[3];
   unsigned ntop = 0;
   if (!flatten_iadd(idx, (nir_scalar){0}, top, &ntop, 3, 0))
      return false;

   if (ntop == 2) {
      nir_scalar mul_ch = {0};
      if (same(top[0], c_var))
         mul_ch = top[1];
      else if (same(top[1], c_var))
         mul_ch = top[0];
      else
         return false;

      if (!alu(mul_ch, nir_op_imul))
         return false;
      nir_scalar a = strip_zero_add(arg(mul_ch, 0));
      nir_scalar b = strip_zero_add(arg(mul_ch, 1));
      nir_scalar xy = {0};
      if (match_push_eq(a, ch_off))
         xy = b;
      else if (match_push_eq(b, ch_off))
         xy = a;
      else
         return false;

      nir_scalar xyt[2];
      unsigned nxy = 0;
      if (!flatten_iadd(xy, (nir_scalar){0}, xyt, &nxy, 2, 0) || nxy != 2)
         return false;

      for (unsigned k = 0; k < 2; ++k) {
         nir_scalar prod = xyt[k], other = xyt[1 - k];
         if (!alu(prod, nir_op_imul))
            continue;
         nir_scalar pa = strip_zero_add(arg(prod, 0));
         nir_scalar pb_s = strip_zero_add(arg(prod, 1));
         uint16_t off = 0, rng = 0;
         if (match_push_any(pb_s, &off, &rng)) {
            *y_out = pa;
            *x_out = other;
            *w_off = off;
            *pb = MAX2(*pb, rng);
            return true;
         }
         if (match_push_any(pa, &off, &rng)) {
            *y_out = pb_s;
            *x_out = other;
            *w_off = off;
            *pb = MAX2(*pb, rng);
            return true;
         }
      }
   } else if (ntop == 3) {
      int c_idx = -1;
      for (unsigned i = 0; i < 3; ++i) {
         if (same(top[i], c_var)) {
            c_idx = (int)i;
            break;
         }
      }
      if (c_idx < 0)
         return false;

      nir_scalar rem[2];
      unsigned nr = 0;
      for (unsigned i = 0; i < 3; ++i) {
         if ((int)i != c_idx)
            rem[nr++] = top[i];
      }

      /* Strip `* ch` from each of the two spatial terms; one becomes `y * W`
       * (or `(y * (W * ch))` right-associated) and the other becomes `x`. */
      for (unsigned k = 0; k < 2; ++k) {
         nir_scalar x_s = {0};
         if (!strip_mul_push_eq(rem[1 - k], ch_off, &x_s))
            continue;

         nir_scalar yw = {0};
         if (strip_mul_push_eq(rem[k], ch_off, &yw) && alu(yw, nir_op_imul)) {
            nir_scalar pa = strip_zero_add(arg(yw, 0));
            nir_scalar pb_s = strip_zero_add(arg(yw, 1));
            uint16_t off = 0, rng = 0;
            if (match_push_any(pb_s, &off, &rng)) {
               *y_out = pa;
               *x_out = x_s;
               *w_off = off;
               *pb = MAX2(*pb, rng);
               return true;
            }
            if (match_push_any(pa, &off, &rng)) {
               *y_out = pb_s;
               *x_out = x_s;
               *w_off = off;
               *pb = MAX2(*pb, rng);
               return true;
            }
         }
         if (alu(rem[k], nir_op_imul)) {
            nir_scalar a = strip_zero_add(arg(rem[k], 0));
            nir_scalar b = strip_zero_add(arg(rem[k], 1));
            nir_scalar w_s = {0};
            uint16_t off = 0, rng = 0;
            if (strip_mul_push_eq(b, ch_off, &w_s) && match_push_any(w_s, &off, &rng)) {
               *y_out = a;
               *x_out = x_s;
               *w_off = off;
               *pb = MAX2(*pb, rng);
               return true;
            }
            if (strip_mul_push_eq(a, ch_off, &w_s) && match_push_any(w_s, &off, &rng)) {
               *y_out = b;
               *x_out = x_s;
               *w_off = off;
               *pb = MAX2(*pb, rng);
               return true;
            }
         }
      }
   }
   return false;
}

static bool
match_resample_nearest(nir_scalar gx, nir_scalar gy, nir_scalar xlim,
                       nir_scalar ylim, nir_scalar src_idx, nir_scalar dst_idx,
                       struct hk_tensor_op_info *info,
                       unsigned *fmul_x_ctrl, unsigned *fmul_y_ctrl)
{
   uint16_t pb = 0;
   uint16_t x0 = UINT16_MAX, x1 = UINT16_MAX;
   uint16_t y0 = UINT16_MAX, y1 = UINT16_MAX;
   if (!match_push_or_mul2(xlim, &x0, &x1, &pb) || x1 == UINT16_MAX)
      return false;
   if (!match_push_or_mul2(ylim, &y0, &y1, &pb) || y1 != UINT16_MAX)
      return false;

   /* dst_idx = (gy * out_w + x) * ch + c  OR  (gy * out_w) * ch + x * ch + c */
   nir_scalar top[3];
   unsigned ntop = 0;
   if (!flatten_iadd(dst_idx, (nir_scalar){0}, top, &ntop, 3, 0) || ntop < 2 || ntop > 3)
      return false;

   nir_scalar c_var = {0}, den = {0};
   uint16_t ch_off = UINT16_MAX, out_w_off = UINT16_MAX;
   uint16_t rng = 0;
   for (unsigned i = 0; i < ntop; ++i) {
      if (match_mod(top[i], gx, &den) && match_push_any(den, &ch_off, &rng)) {
         c_var = top[i];
         break;
      }
   }
   if (!c_var.def)
      return false;
   pb = MAX2(pb, rng);

   if (ch_off == x1)
      out_w_off = x0;
   else if (ch_off == x0)
      out_w_off = x1;
   else
      return false;

   nir_scalar dst_y = {0}, dst_x = {0};
   uint16_t dst_w_off = UINT16_MAX;
   if (!match_nhwc_idx(dst_idx, &dst_y, &dst_x, c_var, ch_off, &dst_w_off, &pb))
      return false;
   if (dst_w_off != out_w_off || !same(dst_y, gy))
      return false;
   nir_scalar qn = {0}, qd = {0};
   if (!match_div(dst_x, &qn, &qd) || !same(qn, gx) || !match_push_eq(qd, ch_off))
      return false;

   nir_scalar src_y = {0}, src_x = {0};
   uint16_t in_w_off = UINT16_MAX;
   if (!match_nhwc_idx(src_idx, &src_y, &src_x, c_var, ch_off, &in_w_off, &pb))
      return false;

   uint16_t sx_off = UINT16_MAX, sy_off = UINT16_MAX;
   if (!match_floor_scale(src_x, dst_x, &sx_off, &pb, fmul_x_ctrl) ||
       !match_floor_scale(src_y, gy, &sy_off, &pb, fmul_y_ctrl))
      return false;

   info->kind = HK_TENSOR_OP_RESAMPLE_NEAREST;
   info->push_bytes = pb;
   info->dim_n_off = UINT16_MAX;
   info->dim_h_off = y0;
   info->dim_q_off = out_w_off;
   info->dim_c_off = ch_off;
   info->dst_stride_y_off = in_w_off;
   info->dst_off_x_off = UINT16_MAX;
   info->dst_stride_q_off = sx_off;
   info->scale_y_off = sy_off;
   return true;
}

static nir_def *
push_u32(nir_builder *b, uint16_t off, uint16_t range)
{
   return nir_load_push_constant(b, 1, 32, nir_imm_int(b, off),
                                 .base = 0, .range = range, .align_mul = 4);
}

static nir_def *
buf_addr(nir_builder *b, struct tensor_buffer loc, nir_def *idx,
         unsigned comps, enum gl_access_qualifier access)
{
   nir_def *res = nir_vulkan_resource_index(
      b, loc.desc_comps, 32, nir_imm_int(b, 0),
      .desc_type = nir_descriptor_type_storage_buffer, .resource_type = loc.resource,
      .desc_set = loc.set, .binding = loc.binding);
   nir_def *desc = nir_load_vulkan_descriptor(
      b, loc.desc_comps, 32, res, .desc_type = nir_descriptor_type_storage_buffer);
   const struct glsl_type *elem =
      comps == 4 ? glsl_vec4_type() :
      comps == 2 ? glsl_vector_type(GLSL_TYPE_FLOAT, 2) :
                   glsl_float_type();
   const unsigned align = comps * 4u;
   const struct glsl_type *arr_t = glsl_array_type(elem, 0, align);
   nir_deref_instr *arr =
      nir_build_deref_cast_with_alignment(b, desc, nir_var_mem_ssbo, arr_t, 0, align, 0);
   nir_deref_instr *el = nir_build_deref_array(b, arr, idx);
   el->arr.in_bounds = true;
   return nir_deref_buffer_address(b, 1, 64, &el->def, .access = access);
}

static void
emit_copy_n(nir_builder *b, struct tensor_buffer src, struct tensor_buffer dst,
            nir_def *src_idx, nir_def *dst_idx, unsigned comps)
{
   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   const unsigned align = comps * 4u;
   nir_def *sa = buf_addr(b, src, src_idx, comps, read);
   nir_def *v = nir_load_global(b, comps, 32, sa, .align_mul = align, .access = read);
   nir_def *da = buf_addr(b, dst, dst_idx, comps, ACCESS_NON_READABLE);
   nir_store_global(b, v, da, .align_mul = align,
                    .write_mask = (uint32_t)BITFIELD_MASK(comps),
                    .access = ACCESS_NON_READABLE);
}

static nir_shader *
create_empty_cs(const nir_shader *orig, const char *suffix,
                uint16_t lx, uint16_t ly, uint16_t lz, uint16_t push_bytes)
{
   nir_shader *nir = nir_shader_create(NULL, MESA_SHADER_COMPUTE, orig->options);
   if (!nir)
      return NULL;
   nir->info.name = ralloc_asprintf(nir, "%s%s", orig->info.name ? orig->info.name : "cs", suffix);
   nir->info.workgroup_size[0] = lx;
   nir->info.workgroup_size[1] = ly;
   nir->info.workgroup_size[2] = lz;
   nir->info.workgroup_size_variable = false;
   nir->info.num_ssbos = orig->info.num_ssbos;
   nir->info.float_controls_execution_mode = orig->info.float_controls_execution_mode;
   nir->num_uniforms = push_bytes;
   nir_function *fn = nir_function_create(nir, "main");
   fn->is_entrypoint = true;
   nir_function_impl_create(fn);
   return nir;
}

/* 2D fast path for CONCAT_X when dst_stride_q == dim_c (contiguous row) OR
 * dim_c in {1, 2, 4} (one vector per q). Zero integer division. */
static nir_shader *
build_concat_x_2d(const nir_shader *orig, struct tensor_buffer src,
                  struct tensor_buffer dst, const struct hk_tensor_op_info *info)
{
   nir_shader *nir = create_empty_cs(orig, "_concatx_2d", 64, 2, 1, info->push_bytes);
   if (!nir)
      return NULL;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   const uint16_t pb = info->push_bytes;
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gx = nir_channel(&b, gid, 0);
   nir_def *gy = nir_channel(&b, gid, 1);

   nir_def *dim_h = push_u32(&b, info->dim_h_off, pb);
   nir_def *y_lim = info->dim_n_off != UINT16_MAX
                       ? nir_imul(&b, push_u32(&b, info->dim_n_off, pb), dim_h)
                       : dim_h;
   nir_def *dim_q = push_u32(&b, info->dim_q_off, pb);
   nir_def *dim_c = push_u32(&b, info->dim_c_off, pb);
   nir_def *x_lim = nir_imul(&b, dim_q, dim_c);
   nir_def *dst_sy = push_u32(&b, info->dst_stride_y_off, pb);
   nir_def *dst_ox = info->dst_off_x_off != UINT16_MAX
                        ? push_u32(&b, info->dst_off_x_off, pb)
                        : nir_imm_int(&b, 0);
   nir_def *dst_sq = push_u32(&b, info->dst_stride_q_off, pb);

   nir_def *contig = nir_ieq(&b, dst_sq, dim_c);
   nir_def *or_row = nir_ior(&b, nir_ior(&b, x_lim, dst_ox), dst_sy);
   nir_def *or_str = nir_ior(&b, nir_ior(&b, dst_ox, dst_sy), dst_sq);
   nir_def *is_v4 = nir_bcsel(
      &b, contig, nir_ieq_imm(&b, nir_iand_imm(&b, or_row, 3), 0),
      nir_iand(&b, nir_ieq_imm(&b, dim_c, 4),
               nir_ieq_imm(&b, nir_iand_imm(&b, or_str, 3), 0)));
   nir_def *is_v2 = nir_bcsel(
      &b, contig, nir_ieq_imm(&b, nir_iand_imm(&b, or_row, 1), 0),
      nir_iand(&b, nir_ieq_imm(&b, dim_c, 2),
               nir_ieq_imm(&b, nir_iand_imm(&b, or_str, 1), 0)));

   nir_if *v4_if = nir_push_if(&b, is_v4);
   {
      nir_def *x_cnt = nir_bcsel(&b, contig, nir_ushr_imm(&b, x_lim, 2), dim_q);
      nir_if *ok = nir_push_if(&b, nir_iand(&b, nir_ult(&b, gx, x_cnt),
                                                nir_ult(&b, gy, y_lim)));
      {
         nir_def *q_st = nir_bcsel(&b, contig, nir_imm_int(&b, 1),
                                   nir_ushr_imm(&b, dst_sq, 2));
         nir_def *s_idx = nir_iadd(&b, nir_imul(&b, gy, x_cnt), gx);
         nir_def *d_idx = nir_iadd(
            &b,
            nir_iadd(&b, nir_imul(&b, gx, q_st),
                     nir_imul(&b, gy, nir_ushr_imm(&b, dst_sy, 2))),
            nir_ushr_imm(&b, dst_ox, 2));
         emit_copy_n(&b, src, dst, s_idx, d_idx, 4);
      }
      nir_pop_if(&b, ok);
   }
   nir_push_else(&b, v4_if);
   {
      nir_if *v2_if = nir_push_if(&b, is_v2);
      {
         nir_def *x_cnt = nir_bcsel(&b, contig, nir_ushr_imm(&b, x_lim, 1), dim_q);
         nir_if *ok = nir_push_if(&b, nir_iand(&b, nir_ult(&b, gx, x_cnt),
                                                   nir_ult(&b, gy, y_lim)));
         {
            nir_def *q_st = nir_bcsel(&b, contig, nir_imm_int(&b, 1),
                                      nir_ushr_imm(&b, dst_sq, 1));
            nir_def *s_idx = nir_iadd(&b, nir_imul(&b, gy, x_cnt), gx);
            nir_def *d_idx = nir_iadd(
               &b,
               nir_iadd(&b, nir_imul(&b, gx, q_st),
                        nir_imul(&b, gy, nir_ushr_imm(&b, dst_sy, 1))),
               nir_ushr_imm(&b, dst_ox, 1));
            emit_copy_n(&b, src, dst, s_idx, d_idx, 2);
         }
         nir_pop_if(&b, ok);
      }
      nir_push_else(&b, v2_if);
      {
         nir_def *x_cnt = nir_bcsel(&b, contig, x_lim, dim_q);
         nir_if *ok = nir_push_if(&b, nir_iand(&b, nir_ult(&b, gx, x_cnt),
                                                   nir_ult(&b, gy, y_lim)));
         {
            nir_def *q_st = nir_bcsel(&b, contig, nir_imm_int(&b, 1), dst_sq);
            nir_def *s_idx = nir_iadd(&b, nir_imul(&b, gy, x_cnt), gx);
            nir_def *d_idx = nir_iadd(
               &b, nir_iadd(&b, nir_imul(&b, gx, q_st), nir_imul(&b, gy, dst_sy)),
               dst_ox);
            emit_copy_n(&b, src, dst, s_idx, d_idx, 1);
         }
         nir_pop_if(&b, ok);
      }
      nir_pop_if(&b, v2_if);
   }
   nir_pop_if(&b, v4_if);

   nir_opt_dce(nir);
   nir_shader_gather_info(nir, impl);
   return nir;
}

/* 3D general path for CONCAT_X: grid = (c_units, dim_q, y_lim), zero division. */
static nir_shader *
build_concat_x_3d(const nir_shader *orig, struct tensor_buffer src,
                  struct tensor_buffer dst, const struct hk_tensor_op_info *info)
{
   nir_shader *nir = create_empty_cs(orig, "_concatx_3d", 4, 16, 2, info->push_bytes);
   if (!nir)
      return NULL;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   const uint16_t pb = info->push_bytes;
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *rc = nir_channel(&b, gid, 0);
   nir_def *q = nir_channel(&b, gid, 1);
   nir_def *y = nir_channel(&b, gid, 2);

   nir_def *dim_h = push_u32(&b, info->dim_h_off, pb);
   nir_def *y_lim = info->dim_n_off != UINT16_MAX
                       ? nir_imul(&b, push_u32(&b, info->dim_n_off, pb), dim_h)
                       : dim_h;
   nir_def *dim_q = push_u32(&b, info->dim_q_off, pb);
   nir_def *dim_c = push_u32(&b, info->dim_c_off, pb);
   nir_def *dst_sy = push_u32(&b, info->dst_stride_y_off, pb);
   nir_def *dst_ox = info->dst_off_x_off != UINT16_MAX
                        ? push_u32(&b, info->dst_off_x_off, pb)
                        : nir_imm_int(&b, 0);
   nir_def *dst_sq = push_u32(&b, info->dst_stride_q_off, pb);

   nir_def *or_all = nir_ior(&b, nir_ior(&b, dim_c, dst_ox),
                             nir_ior(&b, dst_sy, dst_sq));
   nir_def *is_v4 = nir_ieq_imm(&b, nir_iand_imm(&b, or_all, 3), 0);
   nir_def *is_v2 = nir_ieq_imm(&b, nir_iand_imm(&b, or_all, 1), 0);
   nir_def *qy_ok = nir_iand(&b, nir_ult(&b, q, dim_q), nir_ult(&b, y, y_lim));

   nir_if *v4_if = nir_push_if(&b, is_v4);
   {
      nir_def *c4 = nir_ushr_imm(&b, dim_c, 2);
      nir_if *ok = nir_push_if(&b, nir_iand(&b, nir_ult(&b, rc, c4), qy_ok));
      {
         nir_def *s_idx = nir_iadd(&b, nir_imul(&b, nir_iadd(&b, nir_imul(&b, y, dim_q), q), c4), rc);
         nir_def *d_idx = nir_iadd(
            &b,
            nir_iadd(&b, nir_imul(&b, q, nir_ushr_imm(&b, dst_sq, 2)),
                     nir_imul(&b, y, nir_ushr_imm(&b, dst_sy, 2))),
            nir_iadd(&b, nir_ushr_imm(&b, dst_ox, 2), rc));
         emit_copy_n(&b, src, dst, s_idx, d_idx, 4);
      }
      nir_pop_if(&b, ok);
   }
   nir_push_else(&b, v4_if);
   {
      nir_if *v2_if = nir_push_if(&b, is_v2);
      {
         nir_def *c2 = nir_ushr_imm(&b, dim_c, 1);
         nir_if *ok = nir_push_if(&b, nir_iand(&b, nir_ult(&b, rc, c2), qy_ok));
         {
            nir_def *s_idx = nir_iadd(&b, nir_imul(&b, nir_iadd(&b, nir_imul(&b, y, dim_q), q), c2), rc);
            nir_def *d_idx = nir_iadd(
               &b,
               nir_iadd(&b, nir_imul(&b, q, nir_ushr_imm(&b, dst_sq, 1)),
                        nir_imul(&b, y, nir_ushr_imm(&b, dst_sy, 1))),
               nir_iadd(&b, nir_ushr_imm(&b, dst_ox, 1), rc));
            emit_copy_n(&b, src, dst, s_idx, d_idx, 2);
         }
         nir_pop_if(&b, ok);
      }
      nir_push_else(&b, v2_if);
      {
         nir_if *ok = nir_push_if(&b, nir_iand(&b, nir_ult(&b, rc, dim_c), qy_ok));
         {
            nir_def *s_idx = nir_iadd(&b, nir_imul(&b, nir_iadd(&b, nir_imul(&b, y, dim_q), q), dim_c), rc);
            nir_def *d_idx = nir_iadd(
               &b, nir_iadd(&b, nir_imul(&b, q, dst_sq), nir_imul(&b, y, dst_sy)),
               nir_iadd(&b, dst_ox, rc));
            emit_copy_n(&b, src, dst, s_idx, d_idx, 1);
         }
         nir_pop_if(&b, ok);
      }
      nir_pop_if(&b, v2_if);
   }
   nir_pop_if(&b, v4_if);

   nir_opt_dce(nir);
   nir_shader_gather_info(nir, impl);
   return nir;
}

/* Flattened 1D 4x-vec4 copy for CONCAT_Y when dim_n == 1 (16 floats/thread). */
static nir_shader *
build_concat_y_1d_vec4(const nir_shader *orig, struct tensor_buffer src,
                       struct tensor_buffer dst, const struct hk_tensor_op_info *info)
{
   const uint16_t wg_x = 128;
   const uint32_t wg_chunk = (uint32_t)wg_x * 4u;
   nir_shader *nir = create_empty_cs(orig, "_concaty_1d4", wg_x, 1, 1, info->push_bytes);
   if (!nir)
      return NULL;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   const uint16_t pb = info->push_bytes;
   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   nir_def *wg_id = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *wg_base = nir_imul_imm(&b, wg_id, wg_chunk);
   nir_def *base = nir_iadd(&b, wg_base, lid);

   nir_def *dim_h = push_u32(&b, info->dim_h_off, pb);
   nir_def *dim_c = push_u32(&b, info->dim_c_off, pb);
   nir_def *x_lim = info->dim_q_off != UINT16_MAX
                       ? nir_imul(&b, push_u32(&b, info->dim_q_off, pb), dim_c)
                       : dim_c;
   nir_def *dst_y_off = info->dst_off_x_off != UINT16_MAX
                           ? push_u32(&b, info->dst_off_x_off, pb)
                           : nir_imm_int(&b, 0);
   nir_def *total4 = nir_ushr_imm(&b, nir_imul(&b, dim_h, x_lim), 2);
   nir_def *dst_off4 = nir_ushr_imm(&b, nir_imul(&b, dst_y_off, x_lim), 2);

   nir_def *full_wg = nir_iand(
      &b, nir_uge_imm(&b, total4, wg_chunk),
      nir_uge(&b, nir_iadd_imm(&b, total4, -(int32_t)wg_chunk), wg_base));
   nir_if *fast = nir_push_if(&b, full_wg);
   {
      nir_def *vals[4];
      nir_def *d_idxs[4];
      for (unsigned k = 0; k < 4; ++k) {
         nir_def *idx = k ? nir_iadd_imm(&b, base, (int32_t)(k * wg_x)) : base;
         d_idxs[k] = nir_iadd(&b, idx, dst_off4);
         vals[k] = nir_load_global(&b, 4, 32, buf_addr(&b, src, idx, 4, read),
                                   .align_mul = 16, .access = read);
      }
      for (unsigned k = 0; k < 4; ++k) {
         nir_store_global(&b, vals[k],
                          buf_addr(&b, dst, d_idxs[k], 4, ACCESS_NON_READABLE),
                          .align_mul = 16, .write_mask = 0xf,
                          .access = ACCESS_NON_READABLE);
      }
   }
   nir_push_else(&b, fast);
   {
      for (unsigned k = 0; k < 4; ++k) {
         nir_def *idx = k ? nir_iadd_imm(&b, base, (int32_t)(k * wg_x)) : base;
         nir_if *ok = nir_push_if(&b, nir_ult(&b, idx, total4));
         {
            nir_def *v = nir_load_global(&b, 4, 32, buf_addr(&b, src, idx, 4, read),
                                         .align_mul = 16, .access = read);
            nir_def *d_idx = nir_iadd(&b, idx, dst_off4);
            nir_store_global(&b, v,
                             buf_addr(&b, dst, d_idx, 4, ACCESS_NON_READABLE),
                             .align_mul = 16, .write_mask = 0xf,
                             .access = ACCESS_NON_READABLE);
         }
         nir_pop_if(&b, ok);
      }
   }
   nir_pop_if(&b, fast);

   nir_opt_dce(nir);
   nir_shader_gather_info(nir, impl);
   return nir;
}

/* 3D general path for CONCAT_Y: grid = (x_units, dim_h, dim_n), zero division. */
static nir_shader *
build_concat_y_3d(const nir_shader *orig, struct tensor_buffer src,
                  struct tensor_buffer dst, const struct hk_tensor_op_info *info)
{
   nir_shader *nir = create_empty_cs(orig, "_concaty_3d", 32, 4, 1, info->push_bytes);
   if (!nir)
      return NULL;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   const uint16_t pb = info->push_bytes;
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gx = nir_channel(&b, gid, 0);
   nir_def *h = nir_channel(&b, gid, 1);
   nir_def *n = nir_channel(&b, gid, 2);

   nir_def *dim_n = push_u32(&b, info->dim_n_off, pb);
   nir_def *dim_h = push_u32(&b, info->dim_h_off, pb);
   nir_def *dim_c = push_u32(&b, info->dim_c_off, pb);
   nir_def *x_lim = info->dim_q_off != UINT16_MAX
                       ? nir_imul(&b, push_u32(&b, info->dim_q_off, pb), dim_c)
                       : dim_c;
   nir_def *dst_sn = push_u32(&b, info->dst_stride_y_off, pb);
   nir_def *dst_yo = info->dst_off_x_off != UINT16_MAX
                        ? push_u32(&b, info->dst_off_x_off, pb)
                        : nir_imm_int(&b, 0);

   nir_def *is_v4 = nir_ieq_imm(&b, nir_iand_imm(&b, x_lim, 3), 0);
   nir_def *is_v2 = nir_ieq_imm(&b, nir_iand_imm(&b, x_lim, 1), 0);
   nir_def *hn_ok = nir_iand(&b, nir_ult(&b, h, dim_h), nir_ult(&b, n, dim_n));
   nir_def *src_row = nir_iadd(&b, nir_imul(&b, n, dim_h), h);
   nir_def *dst_row = nir_iadd(&b, nir_imul(&b, n, dst_sn), nir_iadd(&b, dst_yo, h));

   nir_if *v4_if = nir_push_if(&b, is_v4);
   {
      nir_def *x4 = nir_ushr_imm(&b, x_lim, 2);
      nir_if *ok = nir_push_if(&b, nir_iand(&b, nir_ult(&b, gx, x4), hn_ok));
      {
         nir_def *s_idx = nir_iadd(&b, nir_imul(&b, src_row, x4), gx);
         nir_def *d_idx = nir_iadd(&b, nir_imul(&b, dst_row, x4), gx);
         emit_copy_n(&b, src, dst, s_idx, d_idx, 4);
      }
      nir_pop_if(&b, ok);
   }
   nir_push_else(&b, v4_if);
   {
      nir_if *v2_if = nir_push_if(&b, is_v2);
      {
         nir_def *x2 = nir_ushr_imm(&b, x_lim, 1);
         nir_if *ok = nir_push_if(&b, nir_iand(&b, nir_ult(&b, gx, x2), hn_ok));
         {
            nir_def *s_idx = nir_iadd(&b, nir_imul(&b, src_row, x2), gx);
            nir_def *d_idx = nir_iadd(&b, nir_imul(&b, dst_row, x2), gx);
            emit_copy_n(&b, src, dst, s_idx, d_idx, 2);
         }
         nir_pop_if(&b, ok);
      }
      nir_push_else(&b, v2_if);
      {
         nir_if *ok = nir_push_if(&b, nir_iand(&b, nir_ult(&b, gx, x_lim), hn_ok));
         {
            nir_def *s_idx = nir_iadd(&b, nir_imul(&b, src_row, x_lim), gx);
            nir_def *d_idx = nir_iadd(&b, nir_imul(&b, dst_row, x_lim), gx);
            emit_copy_n(&b, src, dst, s_idx, d_idx, 1);
         }
         nir_pop_if(&b, ok);
      }
      nir_pop_if(&b, v2_if);
   }
   nir_pop_if(&b, v4_if);

   nir_opt_dce(nir);
   nir_shader_gather_info(nir, impl);
   return nir;
}

/* 3D vec4/vec2/scalar nearest-neighbor resample: grid = (c_units, out_w, out_h). */
static nir_shader *
build_resample_nearest_3d(const nir_shader *orig, struct tensor_buffer src,
                          struct tensor_buffer dst,
                          const struct hk_tensor_op_info *info,
                          unsigned fmul_x_ctrl, unsigned fmul_y_ctrl)
{
   nir_shader *nir = create_empty_cs(orig, "_resample_3d", 8, 8, 2, info->push_bytes);
   if (!nir)
      return NULL;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   const uint16_t pb = info->push_bytes;
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gc = nir_channel(&b, gid, 0);
   nir_def *ox = nir_channel(&b, gid, 1);
   nir_def *oy = nir_channel(&b, gid, 2);

   nir_def *out_w = push_u32(&b, info->dim_q_off, pb);
   nir_def *out_h = push_u32(&b, info->dim_h_off, pb);
   nir_def *ch = push_u32(&b, info->dim_c_off, pb);
   nir_def *in_w = push_u32(&b, info->dst_stride_y_off, pb);
   nir_def *scale_x = push_u32(&b, info->dst_stride_q_off, pb);
   nir_def *scale_y = push_u32(&b, info->scale_y_off, pb);

   nir_def *is_v4 = nir_ieq_imm(&b, nir_iand_imm(&b, ch, 3), 0);
   nir_def *is_v2 = nir_ieq_imm(&b, nir_iand_imm(&b, ch, 1), 0);
   nir_def *xy_ok = nir_iand(&b, nir_ult(&b, ox, out_w), nir_ult(&b, oy, out_h));

   b.fp_math_ctrl = fmul_x_ctrl;
   nir_def *fx = nir_fmul(&b, nir_i2f32(&b, ox), scale_x);
   b.fp_math_ctrl = 0;
   nir_def *sx = nir_f2i32(&b, nir_ffloor(&b, fx));

   b.fp_math_ctrl = fmul_y_ctrl;
   nir_def *fy = nir_fmul(&b, nir_i2f32(&b, oy), scale_y);
   b.fp_math_ctrl = 0;
   nir_def *sy = nir_f2i32(&b, nir_ffloor(&b, fy));

   nir_def *src_px = nir_iadd(&b, nir_imul(&b, sy, in_w), sx);
   nir_def *dst_px = nir_iadd(&b, nir_imul(&b, oy, out_w), ox);

   nir_if *v4_if = nir_push_if(&b, is_v4);
   {
      nir_def *c4 = nir_ushr_imm(&b, ch, 2);
      nir_if *ok = nir_push_if(&b, nir_iand(&b, nir_ult(&b, gc, c4), xy_ok));
      {
         nir_def *s_idx = nir_iadd(&b, nir_imul(&b, src_px, c4), gc);
         nir_def *d_idx = nir_iadd(&b, nir_imul(&b, dst_px, c4), gc);
         emit_copy_n(&b, src, dst, s_idx, d_idx, 4);
      }
      nir_pop_if(&b, ok);
   }
   nir_push_else(&b, v4_if);
   {
      nir_if *v2_if = nir_push_if(&b, is_v2);
      {
         nir_def *c2 = nir_ushr_imm(&b, ch, 1);
         nir_if *ok = nir_push_if(&b, nir_iand(&b, nir_ult(&b, gc, c2), xy_ok));
         {
            nir_def *s_idx = nir_iadd(&b, nir_imul(&b, src_px, c2), gc);
            nir_def *d_idx = nir_iadd(&b, nir_imul(&b, dst_px, c2), gc);
            emit_copy_n(&b, src, dst, s_idx, d_idx, 2);
         }
         nir_pop_if(&b, ok);
      }
      nir_push_else(&b, v2_if);
      {
         nir_if *ok = nir_push_if(&b, nir_iand(&b, nir_ult(&b, gc, ch), xy_ok));
         {
            nir_def *s_idx = nir_iadd(&b, nir_imul(&b, src_px, ch), gc);
            nir_def *d_idx = nir_iadd(&b, nir_imul(&b, dst_px, ch), gc);
            emit_copy_n(&b, src, dst, s_idx, d_idx, 1);
         }
         nir_pop_if(&b, ok);
      }
      nir_pop_if(&b, v2_if);
   }
   nir_pop_if(&b, v4_if);

   nir_opt_dce(nir);
   nir_shader_gather_info(nir, impl);
   return nir;
}

bool
hk_nir_tensor_op_may_match(const struct nir_shader *orig)
{
   if (!orig || orig->info.stage != MESA_SHADER_COMPUTE ||
       orig->info.workgroup_size_variable || orig->info.shared_size != 0 ||
       !hk_identity_copy4_supported_local(orig->info.workgroup_size) ||
       orig->info.num_ssbos != 2 || orig->info.num_images > 0 ||
       orig->info.num_textures > 0 || orig->info.num_ubos > 0 ||
       orig->num_uniforms > HK_COMPUTE_OPT_PUSH_BYTES)
      return false;

   nir_foreach_variable_in_shader(var, orig) {
      if (var->data.mode &
          (nir_var_mem_shared | nir_var_image | nir_var_uniform))
         return false;
   }

   if (!exec_list_is_singular(&orig->functions))
      return false;

   const nir_function_impl *impl = nir_shader_get_entrypoint(orig);
   if (!impl || !acyclic(&impl->body, 8))
      return false;

   return true;
}

bool
hk_nir_match_and_build_tensor_op(const struct nir_shader *orig,
                                 struct hk_tensor_op_info *info_out,
                                 struct nir_shader **alt_primary_out,
                                 struct nir_shader **alt_secondary_out)
{
   *alt_primary_out = NULL;
   *alt_secondary_out = NULL;
   memset(info_out, 0, sizeof(*info_out));

   if (!hk_nir_tensor_op_may_match(orig))
      return false;

   nir_shader *nir = nir_shader_clone(NULL, orig);
   if (!nir)
      return false;
   nir_opt_constant_folding(nir);
   nir_opt_cse(nir);

   struct tensor_buffer src = {0}, dst = {0};
   nir_scalar src_idx = {0}, dst_idx = {0};
   nir_scalar gx = {0}, gy = {0}, xlim = {0}, ylim = {0};

   if (!scan_simple_2d_copy(nir, &src, &dst, &src_idx, &dst_idx,
                            &gx, &gy, &xlim, &ylim)) {
      ralloc_free(nir);
      return false;
   }

   struct hk_tensor_op_info info = {
      .src_set = src.set,
      .dst_set = dst.set,
      .src_binding = src.binding,
      .dst_binding = dst.binding,
      .local = {
         orig->info.workgroup_size[0],
         orig->info.workgroup_size[1],
         orig->info.workgroup_size[2],
      },
   };

   unsigned fmul_x_ctrl = 0, fmul_y_ctrl = 0;
   nir_shader *p = NULL, *s = NULL;
   if (match_concat_x(gx, gy, xlim, ylim, src_idx, dst_idx, &info)) {
      p = build_concat_x_2d(orig, src, dst, &info);
      s = build_concat_x_3d(orig, src, dst, &info);
   } else if (match_concat_y(gx, gy, xlim, ylim, src_idx, dst_idx, &info)) {
      p = build_concat_y_1d_vec4(orig, src, dst, &info);
      s = build_concat_y_3d(orig, src, dst, &info);
   } else if (match_resample_nearest(gx, gy, xlim, ylim, src_idx, dst_idx,
                                     &info, &fmul_x_ctrl, &fmul_y_ctrl)) {
      p = build_resample_nearest_3d(orig, src, dst, &info, fmul_x_ctrl, fmul_y_ctrl);
   }

   ralloc_free(nir);
   if (!p) {
      ralloc_free(s);
      return false;
   }

   *info_out = info;
   *alt_primary_out = p;
   *alt_secondary_out = s;
   return true;
}

/* The element deref an output store addresses: deref_buffer_address of
 * cast(load_vulkan_descriptor(output))[index], as the cooperative-matrix GEMM
 * builder writes it (gemm_f32_addr(), gemm_vec4_addr()). NULL otherwise. */
static nir_deref_instr *
output_store_element(nir_def *addr, unsigned set, unsigned binding)
{
   nir_intrinsic_instr *ba = intr(addr, nir_intrinsic_deref_buffer_address);
   if (!ba)
      return NULL;
   nir_deref_instr *el = nir_src_as_deref(ba->src[0]);
   if (!el || el->deref_type != nir_deref_type_array ||
       el->arr.index.ssa->bit_size != 32)
      return NULL;
   nir_deref_instr *cast = nir_deref_instr_parent(el);
   if (!cast || cast->deref_type != nir_deref_type_cast)
      return NULL;
   nir_intrinsic_instr *desc = intr(cast->parent.ssa, nir_intrinsic_load_vulkan_descriptor);
   if (!desc || nir_intrinsic_desc_type(desc) != nir_descriptor_type_storage_buffer)
      return NULL;
   nir_intrinsic_instr *res = intr(desc->src[0].ssa, nir_intrinsic_vulkan_resource_index);
   if (!res || nir_intrinsic_desc_type(res) != nir_descriptor_type_storage_buffer ||
       nir_intrinsic_desc_set(res) != set || nir_intrinsic_binding(res) != binding)
      return NULL;
   const unsigned stride = nir_deref_instr_array_stride(el);
   return stride == 4 || stride == 8 || stride == 16 ? el : NULL;
}

static nir_def *
push_u64(nir_builder *b, uint16_t off, uint16_t range)
{
   nir_def *lo = push_u32(b, off, range);
   nir_def *hi = push_u32(b, off + 4u, range);
   return nir_pack_64_2x32_split(b, lo, hi);
}

bool
hk_nir_apply_gemm_fadd(struct nir_shader *nir,
                       const struct hk_nir_gemm_output *info,
                       enum hk_compute_pair_dual_write dual_write)
{
   if (!nir || !info || !hk_compute_pair_dual_is_fadd(dual_write))
      return false;

   const bool inplace = dual_write == HK_COMPUTE_PAIR_DUAL_FADD_INPLACE ||
                        dual_write == HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE;
   const bool has_copy = dual_write == HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE ||
                         dual_write == HK_COMPUTE_PAIR_DUAL_FADD_COPY_OUT;
   const uint32_t dual_off = hk_dual_write_push_offset(info->push_bytes);
   const uint32_t extra = has_copy ? sizeof(struct hk_dual_fadd_push)
                                   : (inplace ? 8u : 16u);
   if (dual_off + extra > 256u)
      return false;

   const uint16_t pb = (uint16_t)MAX2((uint32_t)nir->num_uniforms, dual_off + extra);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   /* The residual and destination addresses are push constants, as in every
    * other dual write, plus the store's byte offset from the output's first
    * element. None of them depends on the output descriptor's address: a
    * difference of addresses added to it makes every epilogue access only as
    * right as the output base the compiled shader forms, and a wrong base
    * sends the access outside every buffer, which faults the GPU on G16G
    * (no soft faults). */
   nir_def *res_va = push_u64(
      &b, (uint16_t)(dual_off + offsetof(struct hk_dual_fadd_push, res_va_lo)), pb);
   nir_def *dst1_va =
      !inplace
         ? push_u64(&b, (uint16_t)(dual_off + offsetof(struct hk_dual_fadd_push, dst1_va_lo)), pb)
         : NULL;
   nir_def *dst2_va =
      has_copy
         ? push_u64(&b, (uint16_t)(dual_off + offsetof(struct hk_dual_fadd_push, dst2_va_lo)), pb)
         : NULL;

   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   unsigned transformed = 0;

   nir_foreach_block(block, impl) {
      nir_intrinsic_instr *stores[16];
      unsigned nstores = 0;
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *st = nir_instr_as_intrinsic(instr);
         if (st->intrinsic != nir_intrinsic_store_global)
            continue;
         if (nstores >= ARRAY_SIZE(stores))
            return false;
         if (!output_store_element(st->src[1].ssa, info->output.set,
                                   info->output.binding))
            return false;
         nir_def *val = st->src[0].ssa;
         const unsigned comps = val->num_components;
         if (val->bit_size != 32 || (comps != 1 && comps != 2 && comps != 4) ||
             nir_intrinsic_write_mask(st) != BITFIELD_MASK(comps))
            return false;
         stores[nstores++] = st;
      }
      if (!nstores)
         continue;

      /* Phase 1: Batch all residual loads upfront before any store in the block
       * so DRAM load latency overlaps across tiles. */
      nir_def *res_vals[16];
      nir_def *sums[16];
      nir_def *out_addrs[16];
      nir_def *elem_offs[16];
      unsigned comps_arr[16];
      unsigned align_arr[16];
      for (unsigned i = 0; i < nstores; ++i) {
         nir_intrinsic_instr *st = stores[i];
         nir_def *val = st->src[0].ssa;
         comps_arr[i] = val->num_components;
         align_arr[i] = MAX2(nir_intrinsic_align_mul(st), 4u);
         out_addrs[i] = st->src[1].ssa;
         b.cursor = nir_before_instr(&stores[0]->instr);
         /* Make sure out_addrs[i] dominates stores[0]; if not, insert right
          * before st itself. */
         if (i > 0 && nir_def_instr(out_addrs[i])->block == block) {
            bool after_first = false;
            nir_foreach_instr(scan, block) {
               if (scan == &stores[0]->instr) {
                  after_first = true;
               } else if (scan == nir_def_instr(out_addrs[i])) {
                  if (after_first)
                     b.cursor = nir_before_instr(&st->instr);
                  break;
               }
            }
         }
         /* Byte offset of this store from the output's first element: the
          * builder's index already counts output.offset_B. */
         nir_deref_instr *el = output_store_element(
            out_addrs[i], info->output.set, info->output.binding);
         elem_offs[i] = nir_u2u64(
            &b, nir_iadd_imm(&b,
                             nir_imul_imm(&b, el->arr.index.ssa,
                                          nir_deref_instr_array_stride(el)),
                             -(int64_t)info->output.offset_B));
         nir_def *res_addr = nir_iadd(&b, res_va, elem_offs[i]);
         res_vals[i] = nir_load_global(&b, comps_arr[i], 32, res_addr,
                                       .align_mul = align_arr[i], .access = read);
      }

      /* Phase 2: Compute sums after all loads have been issued, and emit stores
       * de-interleaved by destination buffer (all out1, then dst1, then dst2). */
      nir_intrinsic_instr *last_st = stores[nstores - 1];
      if (inplace) {
         for (unsigned i = 0; i < nstores; ++i) {
            b.cursor = nir_before_instr(&stores[i]->instr);
            sums[i] = nir_fadd(&b, stores[i]->src[0].ssa, res_vals[i]);
            nir_src_rewrite(&stores[i]->src[0], sums[i]);
         }
         b.cursor = nir_after_instr(&last_st->instr);
      } else {
         b.cursor = nir_after_instr(&last_st->instr);
         for (unsigned i = 0; i < nstores; ++i)
            sums[i] = nir_fadd(&b, stores[i]->src[0].ssa, res_vals[i]);
         for (unsigned i = 0; i < nstores; ++i) {
            nir_def *d1_addr = nir_iadd(&b, dst1_va, elem_offs[i]);
            nir_store_global(&b, sums[i], d1_addr, .align_mul = align_arr[i],
                             .write_mask = (uint32_t)BITFIELD_MASK(comps_arr[i]),
                             .access = ACCESS_NON_READABLE);
         }
      }

      if (has_copy) {
         for (unsigned i = 0; i < nstores; ++i) {
            nir_def *d2_addr = nir_iadd(&b, dst2_va, elem_offs[i]);
            nir_store_global(&b, sums[i], d2_addr, .align_mul = align_arr[i],
                             .write_mask = (uint32_t)BITFIELD_MASK(comps_arr[i]),
                             .access = ACCESS_NON_READABLE);
         }
      }

      transformed += nstores;
   }

   if (!transformed)
      return false;

   nir->num_uniforms = pb;
   const char *suffix = has_copy ? (inplace ? "_fadd_in_dcopy" : "_fadd_out_dcopy")
                                 : (inplace ? "_fadd_in" : "_fadd_out");
   nir->info.name = ralloc_asprintf(nir, "%s%s", nir->info.name ? nir->info.name : "gemm", suffix);
   nir_opt_dce(nir);
   nir_shader_gather_info(nir, impl);
   return true;
}

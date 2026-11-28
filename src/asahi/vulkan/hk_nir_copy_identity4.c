/* SPDX-License-Identifier: MIT */
#include "hk_identity_copy4.h"
#include "hk_nir_loop_bound.h"
#include "hk_nir_fast_udiv.h"
#include "hk_nir_match_pointwise_clamp.h"
#include "hk_nir_wgid.h"
#include "compiler/nir/nir_builder.h"

static nir_scalar
scalar(nir_def *def)
{
   return nir_get_scalar(def, 0);
}

static bool
imm(nir_scalar s, uint32_t n)
{
   s = nir_scalar_chase_movs(s);
   return s.def && nir_scalar_is_const(s) && nir_scalar_as_uint(s) == n;
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
      if (imm(arg(s, 0), 0))
         s = nir_scalar_chase_movs(arg(s, 1));
      else if (imm(arg(s, 1), 0))
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
phi(nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   return s.def && !s.comp && s.def->num_components == 1 &&
      nir_def_instr(s.def)->type == nir_instr_type_phi ?
      nir_instr_as_phi(nir_def_instr(s.def)) : NULL;
}

static nir_scalar
phi_in(nir_phi_instr *p, nir_block *block)
{
   nir_phi_src *edge = p ? nir_phi_get_src_from_block(p, block) : NULL;
   return edge ? scalar(edge->src.ssa) : (nir_scalar){0};
}

static bool
const_arg(nir_scalar s, nir_op op, unsigned pos, uint32_t bits)
{
   return alu(s, op) && imm(arg(s, pos), bits);
}

static bool
unary(nir_scalar s, nir_op op, nir_scalar v)
{
   return alu(s, op) && same(arg(s, 0), v);
}

/* Match un-lowered integer division (idiv/udiv). Non-internal Vulkan compute
 * shaders on native AGX3 defer nir_lower_idiv until hk_lower_nir(), so
 * matchers inspect exact NIR division semantics rather than lowered sequences.
 */
static bool
match_div(nir_scalar div, nir_scalar *numerator, nir_scalar *denominator)
{
   div = nir_scalar_chase_movs(div);
   if (alu(div, nir_op_idiv) || alu(div, nir_op_udiv)) {
      *numerator = arg(div, 0);
      *denominator = arg(div, 1);
      return true;
   }
   return false;
}

static bool
match_mul_pair(nir_scalar s, nir_scalar a, nir_scalar b)
{
   if (!alu(s, nir_op_imul))
      return false;
   nir_scalar x = arg(s, 0), y = arg(s, 1);
   return (same(x, a) && same(y, b)) || (same(x, b) && same(y, a));
}

static bool
match_div_mul(nir_scalar prod, nir_scalar numerator, nir_scalar *denominator)
{
   if (!alu(prod, nir_op_imul))
      return false;
   nir_scalar a = arg(prod, 0), b = arg(prod, 1);
   nir_scalar qn, qd;
   if (match_div(a, &qn, &qd) && same(qn, numerator) && same(qd, b)) {
      *denominator = qd;
      return true;
   }
   if (match_div(b, &qn, &qd) && same(qn, numerator) && same(qd, a)) {
      *denominator = qd;
      return true;
   }
   return false;
}

static bool
match_mod(nir_scalar rem, nir_scalar numerator, nir_scalar *denominator)
{
   rem = nir_scalar_chase_movs(rem);
   if (alu(rem, nir_op_irem) || alu(rem, nir_op_imod) || alu(rem, nir_op_umod)) {
      if (!same(arg(rem, 0), numerator))
         return false;
      *denominator = arg(rem, 1);
      return true;
   }
   if (alu(rem, nir_op_isub)) {
      return same(arg(rem, 0), numerator) &&
             match_div_mul(arg(rem, 1), numerator, denominator);
   }
   if (alu(rem, nir_op_iadd)) {
      nir_scalar a = arg(rem, 0), b = arg(rem, 1);
      if (same(a, numerator) && alu(b, nir_op_ineg) &&
          match_div_mul(arg(b, 0), numerator, denominator))
         return true;
      if (same(b, numerator) && alu(a, nir_op_ineg) &&
          match_div_mul(arg(a, 0), numerator, denominator))
         return true;
   }
   return false;
}

static bool
match_push_any(nir_scalar s, uint16_t *off_out, uint16_t *range_out)
{
   s = nir_scalar_chase_movs(s);
   nir_intrinsic_instr *op = intr(s.def, nir_intrinsic_load_push_constant);
   if (!op || s.comp || op->def.bit_size != 32 || op->def.num_components != 1 ||
       nir_intrinsic_access(op) != 0)
      return false;
   nir_scalar off = scalar(op->src[0].ssa);
   if (!nir_scalar_is_const(off))
      return false;
   uint64_t byte_off = nir_intrinsic_base(op) + nir_scalar_as_uint(off);
   if ((byte_off & 3) || byte_off + 4 > HK_COMPUTE_OPT_PUSH_BYTES)
      return false;
   *off_out = (uint16_t)byte_off;
   if (range_out) {
      uint32_t r = MAX2(nir_intrinsic_base(op) + nir_intrinsic_range(op),
                        (uint32_t)(byte_off + 4));
      if (r > HK_COMPUTE_OPT_PUSH_BYTES)
         r = (uint32_t)(byte_off + 4);
      *range_out = (uint16_t)r;
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
match_mul_push_eq(nir_scalar s, uint16_t expected_off, nir_scalar *other_out)
{
   s = nir_scalar_chase_movs(s);
   if (!alu(s, nir_op_imul))
      return false;
   nir_scalar a = arg(s, 0), b = arg(s, 1);
   if (match_push_eq(b, expected_off)) {
      *other_out = a;
      return true;
   }
   if (match_push_eq(a, expected_off)) {
      *other_out = b;
      return true;
   }
   return false;
}

enum im2col_param_id {
   IM2COL_BORDER_WIDTH,
   IM2COL_BORDER_HEIGHT,
   IM2COL_INPUT_WIDTH,
   IM2COL_INPUT_HEIGHT,
   IM2COL_KERNEL_WIDTH,
   IM2COL_PATCHES_WIDTH,
   IM2COL_PATCHES_HEIGHT,
   IM2COL_STRIDE_WIDTH,
   IM2COL_STRIDE_HEIGHT,
   IM2COL_CHANNELS,
   IM2COL_PARAM_COUNT,
};

struct im2col_params {
   uint16_t off[IM2COL_PARAM_COUNT];
   bool bound[IM2COL_PARAM_COUNT];
   uint16_t push_bytes;
};

static bool
bind_im2col_param(struct im2col_params *p, enum im2col_param_id id, nir_scalar s)
{
   uint16_t off = 0, rng = 0;
   if (!match_push_any(s, &off, &rng))
      return false;
   if (p->bound[id]) {
      if (p->off[id] != off)
         return false;
   } else {
      p->bound[id] = true;
      p->off[id] = off;
   }
   p->push_bytes = MAX2(p->push_bytes, MAX2(rng, (uint16_t)(off + 4)));
   return true;
}

static bool
bind_im2col_mul(struct im2col_params *p, enum im2col_param_id id,
                nir_scalar s, nir_scalar *other_out)
{
   if (!alu(s, nir_op_imul))
      return false;
   nir_scalar a = arg(s, 0), b = arg(s, 1);
   if (p->bound[id]) {
      if (match_push_eq(b, p->off[id])) {
         *other_out = a;
         return true;
      }
      if (match_push_eq(a, p->off[id])) {
         *other_out = b;
         return true;
      }
      return false;
   }
   if (bind_im2col_param(p, id, b)) {
      *other_out = a;
      return true;
   }
   if (bind_im2col_param(p, id, a)) {
      *other_out = b;
      return true;
   }
   return false;
}

/* Accept load_workgroup_id[axis] as GlobalInvocationID[axis] when
 * local_size[axis] == 1 (so LocalInvocationID[axis] == 0).
 *
 * NOTE: The bare load_workgroup_id[axis] form (without + load_base_workgroup_id)
 * equals GlobalInvocationID[axis] ONLY when base_group[axis] == 0. This matcher
 * is sound only together with the dispatch-time nonzero-base refusal gate
 * (hk_nir_dispatch_zero_base() in hk_cmd_dispatch.c:hk_select_identity_copy4()),
 * which falls back to the original shader whenever any vkCmdDispatchBase offset
 * is nonzero. */
static bool
is_coord_id(nir_scalar s, unsigned axis, unsigned local_size)
{
   s = nir_scalar_chase_movs(s);
   return s.def && s.def->bit_size == 32 && s.comp == axis &&
          (intr(s.def, nir_intrinsic_load_global_invocation_id) ||
           (local_size == 1 && intr(s.def, nir_intrinsic_load_workgroup_id)));
}

static bool
coord(nir_scalar s, unsigned axis, unsigned local_size)
{
   s = strip_zero_add(s);
   if (is_coord_id(s, axis, local_size))
      return true;
   if (!alu(s, nir_op_iadd))
      return false;

   nir_scalar g = arg(s, 0), base = arg(s, 1);
   if (is_coord_id(g, axis, local_size) &&
       hk_nir_match_base_wg_scaled(base, axis, local_size))
      return true;
   if (is_coord_id(base, axis, local_size) &&
       hk_nir_match_base_wg_scaled(g, axis, local_size))
      return true;
   return false;
}

static nir_if *
as_if(nir_cf_node *node)
{
   return node && node->type == nir_cf_node_if ? nir_cf_node_as_if(node) : NULL;
}

static nir_if *
value_branch(nir_scalar s)
{
   nir_phi_instr *p = phi(s);
   if (!p || !p->instr.block || exec_list_length(&p->srcs) != 2)
      return NULL;
   return as_if(nir_cf_node_prev(&p->instr.block->cf_node));
}

/* A meet value selects exactly the branch's last THEN or ELSE block.
 * Every matching branch is also verified to be acyclic without jumps.
 */
static bool
branch_value(nir_scalar s, nir_scalar *cond, nir_scalar *yes, nir_scalar *no)
{
   nir_if *branch = value_branch(s);
   if (!branch)
      return false;

   nir_phi_instr *p = phi(s);
   *yes = phi_in(p, nir_if_last_then_block(branch));
   *no = phi_in(p, nir_if_last_else_block(branch));
   *cond = scalar(branch->condition.ssa);
   return yes->def && no->def && cond->def;
}

static bool
match_short_or(nir_scalar s, nir_scalar *lhs, nir_scalar *rhs)
{
   if (alu(s, nir_op_ior)) {
      *lhs = arg(s, 0);
      *rhs = arg(s, 1);
      return true;
   }
   nir_scalar cond, yes, no;
   if (branch_value(s, &cond, &yes, &no) && unary(cond, nir_op_inot, no)) {
      *lhs = no;
      *rhs = yes;
      return true;
   }
   return false;
}

static bool
inside_arm(nir_cf_node *node, nir_if *branch, bool then)
{
   if (!branch)
      return false;
   while (node->parent && node->parent != &branch->cf_node)
      node = node->parent;
   if (!node->parent)
      return false;

   struct exec_list *list = then ? &branch->then_list : &branch->else_list;
   foreach_list_typed(nir_cf_node, member, node, list) {
      if (member == node)
         return true;
   }
   return false;
}

static bool
match_eq_one_push(nir_scalar s, uint16_t *off_out, uint16_t *rng_out)
{
   if (!alu(s, nir_op_ieq))
      return false;
   nir_scalar a = arg(s, 0), b = arg(s, 1);
   if (imm(b, 1) && match_push_any(a, off_out, rng_out))
      return true;
   if (imm(a, 1) && match_push_any(b, off_out, rng_out))
      return true;
   return false;
}

static bool
match_unit_strides(nir_scalar value, uint16_t *s0_out, uint16_t *s1_out,
                   uint16_t *pb)
{
   nir_scalar a, b;
   if (alu(value, nir_op_iand)) {
      a = arg(value, 0);
      b = arg(value, 1);
   } else {
      nir_scalar test, w, h;
      if (!branch_value(value, &test, &w, &h) || !same(test, h))
         return false;
      a = h;
      b = w;
   }
   uint16_t r0 = 0, r1 = 0;
   if (!match_eq_one_push(a, s0_out, &r0) || !match_eq_one_push(b, s1_out, &r1))
      return false;
   if (*s0_out == *s1_out)
      return false;
   *pb = MAX2(*pb, MAX2(r0, r1));
   return true;
}

static bool
check_unit_stride_pair(const struct im2col_params *p, uint16_t s0, uint16_t s1)
{
   if (!p->bound[IM2COL_STRIDE_WIDTH] || !p->bound[IM2COL_STRIDE_HEIGHT])
      return false;
   const uint16_t sw = p->off[IM2COL_STRIDE_WIDTH];
   const uint16_t sh = p->off[IM2COL_STRIDE_HEIGHT];
   return (s0 == sw && s1 == sh) || (s0 == sh && s1 == sw);
}

static bool
padding_x(struct im2col_params *p, nir_scalar s)
{
   if (!p->bound[IM2COL_CHANNELS] || !alu(s, nir_op_imul))
      return false;
   nir_scalar a = arg(s, 0), b = arg(s, 1);
   if (match_push_eq(b, p->off[IM2COL_CHANNELS]))
      return bind_im2col_param(p, IM2COL_BORDER_WIDTH, a);
   if (match_push_eq(a, p->off[IM2COL_CHANNELS]))
      return bind_im2col_param(p, IM2COL_BORDER_WIDTH, b);
   return false;
}

static bool
bordered_width(struct im2col_params *p, nir_scalar s)
{
   if (!alu(s, nir_op_iadd))
      return false;
   nir_scalar a = arg(s, 0), b = arg(s, 1);
   if (padding_x(p, b) && bind_im2col_param(p, IM2COL_INPUT_WIDTH, a))
      return true;
   if (padding_x(p, a) && bind_im2col_param(p, IM2COL_INPUT_WIDTH, b))
      return true;
   return false;
}

static bool
match_sub_expr(nir_scalar s, nir_scalar lhs, nir_scalar *rhs_out)
{
   if (alu(s, nir_op_isub) && same(arg(s, 0), lhs)) {
      *rhs_out = arg(s, 1);
      return true;
   }
   if (alu(s, nir_op_iadd)) {
      nir_scalar a = arg(s, 0), b = arg(s, 1);
      if (same(a, lhs) && alu(b, nir_op_ineg)) {
         *rhs_out = arg(b, 0);
         return true;
      }
      if (same(b, lhs) && alu(a, nir_op_ineg)) {
         *rhs_out = arg(a, 0);
         return true;
      }
   }
   return false;
}

/* The exact signed output-column count becomes IW/C when pads are zero,
 * both strides are one, and KW equals C.
 */
static bool
columns(struct im2col_params *p, nir_scalar s)
{
   nir_scalar body;
   if (const_arg(s, nir_op_iadd, 1, 1))
      body = arg(s, 0);
   else if (const_arg(s, nir_op_iadd, 0, 1))
      body = arg(s, 1);
   else
      return false;

   nir_scalar diff, stride, iw, ic, kw, kc;
   if (!match_div(body, &diff, &stride) ||
       !bind_im2col_param(p, IM2COL_STRIDE_WIDTH, stride))
      return false;

   nir_scalar term0, term1;
   if (alu(diff, nir_op_isub)) {
      term0 = arg(diff, 0);
      term1 = arg(diff, 1);
   } else if (alu(diff, nir_op_iadd) && alu(arg(diff, 1), nir_op_ineg)) {
      term0 = arg(diff, 0);
      term1 = arg(arg(diff, 1), 0);
   } else if (alu(diff, nir_op_iadd) && alu(arg(diff, 0), nir_op_ineg)) {
      term0 = arg(diff, 1);
      term1 = arg(arg(diff, 0), 0);
   } else {
      return false;
   }

   if (!match_div(term0, &iw, &ic) ||
       !bind_im2col_param(p, IM2COL_CHANNELS, ic) ||
       !match_div(term1, &kw, &kc) ||
       !bind_im2col_param(p, IM2COL_KERNEL_WIDTH, kw) ||
       !bind_im2col_param(p, IM2COL_CHANNELS, kc))
      return false;

   /* Match `iw_pad = is_unit ? (iw + 2*pad_x) : (iw + pad_x)` across
    * phi/bcsel and algebraic hoisting of common addends (`one_pad`, `pad_x`,
    * or `iw`). */
   if (alu(iw, nir_op_iadd)) {
      nir_scalar a = arg(iw, 0), b = arg(iw, 1);
      nir_scalar base = {0}, sel = {0};
      if (alu(b, nir_op_bcsel)) {
         base = a;
         sel = b;
      } else if (alu(a, nir_op_bcsel)) {
         base = b;
         sel = a;
      }
      if (sel.def) {
         uint16_t s0 = 0, s1 = 0;
         nir_scalar is_unit = arg(sel, 0), tval = arg(sel, 1), fval = arg(sel, 2);
         if (match_unit_strides(is_unit, &s0, &s1, &p->push_bytes) &&
             check_unit_stride_pair(p, s0, s1)) {
            /* Form A: one_pad + bcsel(is_unit, pad_x, 0) */
            if (bordered_width(p, base) && padding_x(p, tval) && imm(fval, 0))
               return true;
            /* Form B: pad_x + bcsel(is_unit, one_pad, iw) */
            if (padding_x(p, base) && bordered_width(p, tval) &&
                match_push_eq(fval, p->off[IM2COL_INPUT_WIDTH]))
               return true;
            /* Form C: iw + bcsel(is_unit, 2*pad_x, pad_x) */
            if (match_push_eq(base, p->off[IM2COL_INPUT_WIDTH]) && padding_x(p, fval)) {
               if (const_arg(tval, nir_op_ishl, 1, 1) && padding_x(p, arg(tval, 0)))
                  return true;
               if (alu(tval, nir_op_imul)) {
                  nir_scalar m0 = arg(tval, 0), m1 = arg(tval, 1);
                  if ((imm(m0, 2) && padding_x(p, m1)) ||
                      (imm(m1, 2) && padding_x(p, m0)))
                     return true;
               }
            }
         }
      }
   }

   nir_scalar is_unit, both_pads, one_pad;
   uint16_t s0 = 0, s1 = 0;
   if (alu(iw, nir_op_bcsel)) {
      is_unit = arg(iw, 0);
      both_pads = arg(iw, 1);
      one_pad = arg(iw, 2);
   } else if (!branch_value(iw, &is_unit, &both_pads, &one_pad)) {
      return false;
   }
   if (!match_unit_strides(is_unit, &s0, &s1, &p->push_bytes) ||
       !check_unit_stride_pair(p, s0, s1) ||
       !bordered_width(p, one_pad) ||
       !alu(both_pads, nir_op_iadd))
      return false;

   nir_scalar ba = arg(both_pads, 0), bb = arg(both_pads, 1);
   if ((same(ba, one_pad) && padding_x(p, bb)) ||
       (same(bb, one_pad) && padding_x(p, ba)))
      return true;

   /* nir_opt_algebraic may fold (iw + pad_x) + pad_x into iw + (pad_x << 1)
    * or iw + (pad_x * 2) or iw + ((bw << 1) * ch). */
   nir_scalar two_pad = {0};
   if (match_push_eq(ba, p->off[IM2COL_INPUT_WIDTH]))
      two_pad = bb;
   else if (match_push_eq(bb, p->off[IM2COL_INPUT_WIDTH]))
      two_pad = ba;
   else
      return false;

   if (const_arg(two_pad, nir_op_ishl, 1, 1) && padding_x(p, arg(two_pad, 0)))
      return true;
   if (alu(two_pad, nir_op_imul)) {
      nir_scalar m0 = arg(two_pad, 0), m1 = arg(two_pad, 1);
      if ((imm(m0, 2) && padding_x(p, m1)) || (imm(m1, 2) && padding_x(p, m0)))
         return true;
      if (const_arg(m0, nir_op_ishl, 1, 1) &&
          ((match_push_eq(arg(m0, 0), p->off[IM2COL_BORDER_WIDTH]) &&
            match_push_eq(m1, p->off[IM2COL_CHANNELS])) ||
           (match_push_eq(arg(m0, 0), p->off[IM2COL_CHANNELS]) &&
            match_push_eq(m1, p->off[IM2COL_BORDER_WIDTH]))))
         return true;
      if (const_arg(m1, nir_op_ishl, 1, 1) &&
          ((match_push_eq(arg(m1, 0), p->off[IM2COL_BORDER_WIDTH]) &&
            match_push_eq(m0, p->off[IM2COL_CHANNELS])) ||
           (match_push_eq(arg(m1, 0), p->off[IM2COL_CHANNELS]) &&
            match_push_eq(m0, p->off[IM2COL_BORDER_WIDTH]))))
         return true;
   }
   return false;
}

static bool
columns_borderless(struct im2col_params *p, nir_scalar s)
{
   nir_scalar body;
   if (const_arg(s, nir_op_iadd, 1, 1))
      body = arg(s, 0);
   else if (const_arg(s, nir_op_iadd, 0, 1))
      body = arg(s, 1);
   else
      return false;

   nir_scalar diff, stride, iw, ic, kw, kc;
   if (!match_div(body, &diff, &stride) ||
       !bind_im2col_param(p, IM2COL_STRIDE_WIDTH, stride))
      return false;

   nir_scalar term0, term1;
   if (alu(diff, nir_op_isub)) {
      term0 = arg(diff, 0);
      term1 = arg(diff, 1);
   } else if (alu(diff, nir_op_iadd) && alu(arg(diff, 1), nir_op_ineg)) {
      term0 = arg(diff, 0);
      term1 = arg(arg(diff, 1), 0);
   } else if (alu(diff, nir_op_iadd) && alu(arg(diff, 0), nir_op_ineg)) {
      term0 = arg(diff, 1);
      term1 = arg(arg(diff, 0), 0);
   } else {
      return false;
   }

   return match_div(term0, &iw, &ic) &&
          bind_im2col_param(p, IM2COL_INPUT_WIDTH, iw) &&
          bind_im2col_param(p, IM2COL_CHANNELS, ic) &&
          match_div(term1, &kw, &kc) &&
          bind_im2col_param(p, IM2COL_KERNEL_WIDTH, kw) &&
          bind_im2col_param(p, IM2COL_CHANNELS, kc);
}

static bool
original_coords(struct im2col_params *p, nir_scalar tap, nir_scalar row,
                nir_scalar x, nir_scalar y, nir_scalar *cols_out)
{
   if (!alu(y, nir_op_iadd) || !alu(x, nir_op_iadd))
      return false;

   /* y = (row / cols) * stride_height + (tap / kernel_width) */
   nir_scalar ya = arg(y, 0), yb = arg(y, 1);
   nir_scalar y_row_mul = ya, y_tap_div = yb;
   nir_scalar tap_n, tap_d;
   if (!match_div(y_tap_div, &tap_n, &tap_d) || !same(tap_n, tap)) {
      y_row_mul = yb;
      y_tap_div = ya;
      if (!match_div(y_tap_div, &tap_n, &tap_d) || !same(tap_n, tap))
         return false;
   }
   if (!bind_im2col_param(p, IM2COL_KERNEL_WIDTH, tap_d))
      return false;

   nir_scalar row_div;
   if (!bind_im2col_mul(p, IM2COL_STRIDE_HEIGHT, y_row_mul, &row_div))
      return false;
   nir_scalar row_n, row_divisor;
   if (!match_div(row_div, &row_n, &row_divisor) || !same(row_n, row))
      return false;

   /* x = ((row % cols) * stride_width) * channels + (tap % kernel_width) */
   nir_scalar xa = arg(x, 0), xb = arg(x, 1);
   nir_scalar x_row_term = xa, x_tap_mod = xb;
   nir_scalar mod_d;
   if (!match_mod(x_tap_mod, tap, &mod_d) ||
       !bind_im2col_param(p, IM2COL_KERNEL_WIDTH, mod_d)) {
      x_row_term = xb;
      x_tap_mod = xa;
      if (!match_mod(x_tap_mod, tap, &mod_d) ||
          !bind_im2col_param(p, IM2COL_KERNEL_WIDTH, mod_d))
         return false;
   }

   /* Accept either ((row % cols) * sw) * ch or (row % cols) * (sw * ch). */
   nir_scalar row_mod = {0}, sw_s = {0}, ch_s = {0};
   if (!alu(x_row_term, nir_op_imul))
      return false;
   nir_scalar m0 = arg(x_row_term, 0), m1 = arg(x_row_term, 1);
   if (alu(m0, nir_op_imul)) {
      ch_s = m1;
      nir_scalar i0 = arg(m0, 0), i1 = arg(m0, 1);
      if (match_mod(i0, row, &mod_d) && same(mod_d, row_divisor)) {
         row_mod = i0;
         sw_s = i1;
      } else if (match_mod(i1, row, &mod_d) && same(mod_d, row_divisor)) {
         row_mod = i1;
         sw_s = i0;
      } else {
         return false;
      }
   } else if (alu(m1, nir_op_imul)) {
      ch_s = m0;
      nir_scalar i0 = arg(m1, 0), i1 = arg(m1, 1);
      if (match_mod(i0, row, &mod_d) && same(mod_d, row_divisor)) {
         row_mod = i0;
         sw_s = i1;
      } else if (match_mod(i1, row, &mod_d) && same(mod_d, row_divisor)) {
         row_mod = i1;
         sw_s = i0;
      } else {
         return false;
      }
   } else {
      return false;
   }
   (void)row_mod;

   if (!bind_im2col_param(p, IM2COL_STRIDE_WIDTH, sw_s) ||
       !bind_im2col_param(p, IM2COL_CHANNELS, ch_s))
      return false;

   *cols_out = row_divisor;
   return true;
}

struct copy_buffer {
   uint8_t set;
   uint8_t desc_comps;
   uint16_t binding;
   unsigned resource;
};

static bool
match_buffer_typed(nir_intrinsic_instr *op, struct copy_buffer *out,
                   nir_scalar *index, bool allow_int)
{
   nir_def *ptr = op->src[0].ssa;
   if (nir_def_instr(ptr)->type != nir_instr_type_deref)
      return false;

   nir_deref_instr *element = nir_instr_as_deref(nir_def_instr(ptr));
   nir_deref_instr *array = nir_deref_instr_parent(element);
   nir_deref_instr *root = array ? nir_deref_instr_parent(array) : NULL;
   if (element->deref_type != nir_deref_type_array ||
       element->modes != nir_var_mem_ssbo || !element->arr.in_bounds ||
       !(element->type == glsl_float_type() ||
         (allow_int && (element->type == glsl_int_type() ||
                        element->type == glsl_uint_type()))) ||
       !array || array->deref_type != nir_deref_type_struct ||
       array->modes != nir_var_mem_ssbo || array->strct.index != 0 ||
       !glsl_type_is_unsized_array(array->type) ||
       glsl_get_explicit_stride(array->type) != 4 ||
       !root || root->deref_type != nir_deref_type_cast ||
       root->modes != nir_var_mem_ssbo || root->cast.ptr_stride != 0 ||
       root->cast.align_offset != 0 || !glsl_type_is_struct_or_ifc(root->type) ||
       glsl_get_length(root->type) != 1 || glsl_get_struct_field_offset(root->type, 0) != 0)
      return false;

   nir_intrinsic_instr *load = intr(root->parent.ssa, nir_intrinsic_load_vulkan_descriptor);
   nir_intrinsic_instr *resource = load ?
      intr(load->src[0].ssa, nir_intrinsic_vulkan_resource_index) : NULL;
   const unsigned comps = root->def.num_components;
   if (comps < 3 || comps > 5 || root->def.bit_size != 32 ||
       root->cast.align_mul != 4 ||
       !resource || resource->def.num_components != comps ||
       load->def.num_components != comps || resource->def.bit_size != 32 ||
       load->def.bit_size != 32 || !imm(scalar(resource->src[0].ssa), 0) ||
       nir_intrinsic_desc_type(resource) != nir_descriptor_type_storage_buffer ||
       nir_intrinsic_desc_type(load) != nir_descriptor_type_storage_buffer ||
       nir_intrinsic_desc_set(resource) > UINT8_MAX ||
       nir_intrinsic_binding(resource) > UINT16_MAX)
      return false;

   const unsigned type = nir_intrinsic_resource_type(resource);
   if (type != nir_resource_type_read_only_storage_buffer &&
       type != nir_resource_type_read_write_storage_buffer)
      return false;

   nir_def *offset = element->arr.index.ssa;
   if (offset->bit_size != 32 || offset->num_components != 1)
      return false;

   *index = nir_scalar_resolved(offset, 0);
   *out = (struct copy_buffer) {
      .set = nir_intrinsic_desc_set(resource),
      .desc_comps = comps,
      .binding = nir_intrinsic_binding(resource),
      .resource = type,
   };
   return true;
}

static bool
match_buffer(nir_intrinsic_instr *op, struct copy_buffer *out, nir_scalar *index)
{
   return match_buffer_typed(op, out, index, false);
}

static bool
same_binding(struct copy_buffer a, struct copy_buffer b)
{
   return a.set == b.set && a.binding == b.binding;
}

static bool
zero_then(nir_scalar val, nir_scalar *cond, nir_scalar *next)
{
   nir_scalar zero;
   return branch_value(val, cond, &zero, next) && imm(zero, 0);
}

static bool
out_of_bounds(struct im2col_params *p, nir_scalar test,
              nir_scalar tap, nir_scalar row)
{
   nir_scalar oob_x, oob_y;
   if (!match_short_or(test, &oob_x, &oob_y))
      return false;
   if (alu(oob_x, nir_op_ige) && alu(oob_y, nir_op_ige)) {
      if (same(arg(oob_x, 0), tap) && same(arg(oob_y, 0), row)) {
         return bind_im2col_param(p, IM2COL_PATCHES_WIDTH, arg(oob_x, 1)) &&
                bind_im2col_param(p, IM2COL_PATCHES_HEIGHT, arg(oob_y, 1));
      }
      if (same(arg(oob_x, 0), row) && same(arg(oob_y, 0), tap)) {
         return bind_im2col_param(p, IM2COL_PATCHES_HEIGHT, arg(oob_x, 1)) &&
                bind_im2col_param(p, IM2COL_PATCHES_WIDTH, arg(oob_y, 1));
      }
   }
   return false;
}

static bool
safe_unit_read(struct im2col_params *p, nir_scalar value, nir_scalar tap,
               nir_scalar row, nir_scalar index, nir_intrinsic_instr *copy)
{
   nir_scalar cutoff[4];
   for (unsigned i = 0; i < ARRAY_SIZE(cutoff); ++i) {
      if (!inside_arm(&copy->instr.block->cf_node, value_branch(value), false) ||
          !zero_then(value, &cutoff[i], &value))
         return false;
   }
   nir_scalar xlo = cutoff[0], ylo = cutoff[1], xhi = cutoff[2], yhi = cutoff[3];
   if (value.def != &copy->def ||
       !alu(xlo, nir_op_ilt) || !alu(ylo, nir_op_ilt))
      return false;

   nir_scalar x = arg(xlo, 0), y = arg(ylo, 0);
   nir_scalar cols_s = {0};
   /* Bind STRIDE_WIDTH, STRIDE_HEIGHT, CHANNELS, KERNEL_WIDTH first so
    * padding_x and bordered_width unambiguously identify BORDER_WIDTH and
    * INPUT_WIDTH.
    */
   if (!original_coords(p, tap, row, x, y, &cols_s))
      return false;

   if (!padding_x(p, arg(xlo, 1)) ||
       !bind_im2col_param(p, IM2COL_BORDER_HEIGHT, arg(ylo, 1)))
      return false;

   if (!alu(xhi, nir_op_ige) || !same(arg(xhi, 0), x) ||
       !bordered_width(p, arg(xhi, 1)))
      return false;

   if (!alu(yhi, nir_op_ige) || !same(arg(yhi, 0), y) ||
       !alu(arg(yhi, 1), nir_op_iadd))
      return false;
   nir_scalar yhi_a = arg(arg(yhi, 1), 0), yhi_b = arg(arg(yhi, 1), 1);
   if (match_push_eq(yhi_b, p->off[IM2COL_BORDER_HEIGHT])) {
      if (!bind_im2col_param(p, IM2COL_INPUT_HEIGHT, yhi_a))
         return false;
   } else if (match_push_eq(yhi_a, p->off[IM2COL_BORDER_HEIGHT])) {
      if (!bind_im2col_param(p, IM2COL_INPUT_HEIGHT, yhi_b))
         return false;
   } else {
      return false;
   }

   if (!columns(p, cols_s))
      return false;

   /* index = (y - BORDER_HEIGHT) * INPUT_WIDTH + (x - padding_x) */
   if (!alu(index, nir_op_iadd))
      return false;
   nir_scalar ia = arg(index, 0), ib = arg(index, 1);
   nir_scalar src_y = {0}, src_x = {0};
   if (bind_im2col_mul(p, IM2COL_INPUT_WIDTH, ia, &src_y)) {
      src_x = ib;
   } else if (bind_im2col_mul(p, IM2COL_INPUT_WIDTH, ib, &src_y)) {
      src_x = ia;
   } else {
      return false;
   }

   nir_scalar bh_sub = {0}, px_sub = {0};
   return match_sub_expr(src_y, y, &bh_sub) &&
          match_push_eq(bh_sub, p->off[IM2COL_BORDER_HEIGHT]) &&
          match_sub_expr(src_x, x, &px_sub) &&
          padding_x(p, px_sub);
}

static bool
acyclic(const struct exec_list *list, unsigned depth)
{
   if (!depth)
      return false;

   foreach_list_typed(nir_cf_node, cf, node, list) {
      if (cf->type == nir_cf_node_block)
         continue;
      nir_if *branch = as_if(cf);
      if (!branch || !acyclic(&branch->then_list, depth - 1) ||
          !acyclic(&branch->else_list, depth - 1))
         return false;
   }
   return true;
}

struct access_set {
   nir_intrinsic_instr *store;
   nir_intrinsic_instr *load[2];
   unsigned load_count;
};

static bool
collect_accesses(nir_function_impl *impl, struct access_set *accesses)
{
   unsigned count = 0;
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
            if (count == ARRAY_SIZE(accesses->load) || op->def.bit_size != 32 ||
                op->def.num_components != 1 ||
                (nir_intrinsic_access(op) & ~ACCESS_CAN_REORDER) != ACCESS_NON_WRITEABLE)
               return false;
            accesses->load[count++] = op;
            break;
         case nir_intrinsic_store_deref:
            if (accesses->store || nir_intrinsic_write_mask(op) != 1 ||
                nir_intrinsic_access(op) != ACCESS_NON_READABLE ||
                op->src[1].ssa->bit_size != 32 || op->src[1].ssa->num_components != 1)
               return false;
            accesses->store = op;
            break;
         default:
            return false;
         }
      }
   }
   accesses->load_count = count;
   return accesses->store && count >= 1 && count <= ARRAY_SIZE(accesses->load);
}

static bool
safe_nonunit_read(struct im2col_params *p, nir_scalar value, nir_scalar tap,
                  nir_scalar row, nir_scalar index, nir_intrinsic_instr *read_op)
{
   nir_scalar cond, inner;
   if (!inside_arm(&read_op->instr.block->cf_node, value_branch(value), false) ||
       !zero_then(value, &cond, &inner) || inner.def != &read_op->def)
      return false;

   /* cond is (x >= input_width || y >= input_height). */
   nir_scalar oob_x, oob_y;
   if (!match_short_or(cond, &oob_x, &oob_y) ||
       !alu(oob_x, nir_op_ige) || !alu(oob_y, nir_op_ige))
      return false;

   nir_scalar x, y;
   if (match_push_eq(arg(oob_x, 1), p->off[IM2COL_INPUT_WIDTH]) &&
       match_push_eq(arg(oob_y, 1), p->off[IM2COL_INPUT_HEIGHT])) {
      x = arg(oob_x, 0);
      y = arg(oob_y, 0);
   } else if (match_push_eq(arg(oob_y, 1), p->off[IM2COL_INPUT_WIDTH]) &&
              match_push_eq(arg(oob_x, 1), p->off[IM2COL_INPUT_HEIGHT])) {
      x = arg(oob_y, 0);
      y = arg(oob_x, 0);
   } else {
      return false;
   }

   nir_scalar cols_s = {0};
   if (!original_coords(p, tap, row, x, y, &cols_s) || !columns(p, cols_s))
      return false;

   if (!alu(index, nir_op_iadd))
      return false;
   nir_scalar ia = arg(index, 0), ib = arg(index, 1), y_chk = {0};
   if (bind_im2col_mul(p, IM2COL_INPUT_WIDTH, ia, &y_chk) &&
       same(y_chk, y) && same(ib, x))
      return true;
   if (bind_im2col_mul(p, IM2COL_INPUT_WIDTH, ib, &y_chk) &&
       same(y_chk, y) && same(ia, x))
      return true;
   return false;
}

static bool
match_im2col(nir_shader *nir, struct copy_buffer *src, struct copy_buffer *dst,
             struct im2col_params *params, bool *im2col_vec4_ok)
{
   memset(params, 0, sizeof(*params));
   if (nir->info.stage != MESA_SHADER_COMPUTE || nir->info.workgroup_size_variable ||
       !hk_identity_copy4_supported_local(nir->info.workgroup_size) ||
       nir->info.num_ssbos != 2 ||
       nir->num_uniforms > HK_COMPUTE_OPT_PUSH_BYTES)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_function_impl(other, nir) {
      if (other != impl)
         return false;
   }
   if (!acyclic(&impl->body, 16))
      return false;

   struct access_set ops = {0};
   if (!collect_accesses(impl, &ops))
      return false;

   nir_scalar dst_index;
   if (!match_buffer(ops.store, dst, &dst_index) ||
       dst->resource != nir_resource_type_read_write_storage_buffer ||
       !alu(dst_index, nir_op_iadd))
      return false;

   nir_scalar tap = {0}, row = {0};
   nir_scalar da = arg(dst_index, 0), db = arg(dst_index, 1);
   if (coord(db, 0, nir->info.workgroup_size[0]) &&
       bind_im2col_mul(params, IM2COL_PATCHES_WIDTH, da, &row) &&
       coord(row, 1, nir->info.workgroup_size[1])) {
      tap = db;
   } else if (coord(da, 0, nir->info.workgroup_size[0]) &&
              bind_im2col_mul(params, IM2COL_PATCHES_WIDTH, db, &row) &&
              coord(row, 1, nir->info.workgroup_size[1])) {
      tap = da;
   } else {
      return false;
   }

   nir_block *write_block = ops.store->instr.block;
   nir_if *bounds = as_if(write_block->cf_node.parent);
   if (!bounds || bounds->cf_node.parent != &impl->cf_node ||
       write_block != nir_if_last_else_block(bounds) ||
       !out_of_bounds(params, scalar(bounds->condition.ssa), tap, row))
      return false;

   nir_scalar payload = scalar(ops.store->src[1].ssa);
   if (ops.load_count == 1) {
      nir_intrinsic_instr *ld = ops.load[0];
      nir_scalar src_index = {0};
      if (payload.def != &ld->def ||
          !inside_arm(&ld->instr.block->cf_node, bounds, false) ||
          !match_buffer(ld, src, &src_index) ||
          same_binding(*src, *dst) ||
          !alu(src_index, nir_op_iadd))
         return false;

      nir_scalar ia = arg(src_index, 0), ib = arg(src_index, 1);
      nir_scalar x = {0}, y = {0};
      if (bind_im2col_mul(params, IM2COL_INPUT_WIDTH, ia, &y)) {
         x = ib;
      } else if (bind_im2col_mul(params, IM2COL_INPUT_WIDTH, ib, &y)) {
         x = ia;
      } else {
         return false;
      }

      nir_scalar cols_s = {0};
      if (!original_coords(params, tap, row, x, y, &cols_s) ||
          !columns_borderless(params, cols_s))
         return false;

      for (unsigned i = 0; i < IM2COL_PARAM_COUNT; ++i) {
         if (i == IM2COL_BORDER_WIDTH || i == IM2COL_BORDER_HEIGHT ||
             i == IM2COL_INPUT_HEIGHT) {
            if (params->bound[i])
               return false;
            params->off[i] = UINT16_MAX;
         } else if (!params->bound[i]) {
            return false;
         }
      }
      if (im2col_vec4_ok)
         *im2col_vec4_ok = true;
      return true;
   }

   nir_if *stride_branch = value_branch(payload);
   nir_scalar is_unit, unit_value, other_value;
   uint16_t s0 = 0, s1 = 0;
   if (!stride_branch || !inside_arm(&stride_branch->cf_node, bounds, false) ||
       !branch_value(payload, &is_unit, &unit_value, &other_value) ||
       !match_unit_strides(is_unit, &s0, &s1, &params->push_bytes))
      return false;

   nir_scalar discarded_cond, other_read;
   if (!zero_then(other_value, &discarded_cond, &other_read))
      return false;
   nir_intrinsic_instr *unused = intr(other_read.def, nir_intrinsic_load_deref);
   if (unused != ops.load[0] && unused != ops.load[1])
      return false;
   nir_intrinsic_instr *copy = unused == ops.load[0] ? ops.load[1] : ops.load[0];
   if (!inside_arm(&unused->instr.block->cf_node, stride_branch, false) ||
       !inside_arm(&copy->instr.block->cf_node, stride_branch, true))
      return false;

   struct copy_buffer other;
   nir_scalar src_index, unused_index;
   if (!match_buffer(copy, src, &src_index) ||
       !match_buffer(unused, &other, &unused_index) || !same_binding(*src, other) ||
       same_binding(*src, *dst) ||
       !safe_unit_read(params, unit_value, tap, row, src_index, copy) ||
       !check_unit_stride_pair(params, s0, s1))
      return false;

   for (unsigned i = 0; i < IM2COL_PARAM_COUNT; ++i) {
      if (!params->bound[i])
         return false;
   }

   if (im2col_vec4_ok)
      *im2col_vec4_ok = safe_nonunit_read(params, other_value, tap, row, unused_index, unused);
   return true;
}

static nir_def *
address(nir_builder *b, struct copy_buffer loc, nir_def *index,
        enum gl_access_qualifier access)
{
   nir_def *resource = nir_vulkan_resource_index(
      b, loc.desc_comps, 32, nir_imm_int(b, 0),
      .desc_type = nir_descriptor_type_storage_buffer, .resource_type = loc.resource,
      .desc_set = loc.set, .binding = loc.binding);
   nir_def *descriptor = nir_load_vulkan_descriptor(
      b, loc.desc_comps, 32, resource, .desc_type = nir_descriptor_type_storage_buffer);
   const glsl_type *array_type = glsl_array_type(glsl_vec4_type(), 0, 16);
   nir_deref_instr *array = nir_build_deref_cast_with_alignment(
      b, descriptor, nir_var_mem_ssbo, array_type, 0, 16, 0);
   nir_deref_instr *element = nir_build_deref_array(b, array, index);
   element->arr.in_bounds = true;
   return nir_deref_buffer_address(b, 1, 64, &element->def, .access = access);
}

static nir_def *
address_scalar(nir_builder *b, struct copy_buffer loc, nir_def *index,
               enum gl_access_qualifier access, bool is_int)
{
   nir_def *resource = nir_vulkan_resource_index(
      b, loc.desc_comps, 32, nir_imm_int(b, 0),
      .desc_type = nir_descriptor_type_storage_buffer, .resource_type = loc.resource,
      .desc_set = loc.set, .binding = loc.binding);
   nir_def *descriptor = nir_load_vulkan_descriptor(
      b, loc.desc_comps, 32, resource, .desc_type = nir_descriptor_type_storage_buffer);
   const glsl_type *elem = is_int ? glsl_int_type() : glsl_float_type();
   const glsl_type *array_type = glsl_array_type(elem, 0, 4);
   nir_deref_instr *array = nir_build_deref_cast_with_alignment(
      b, descriptor, nir_var_mem_ssbo, array_type, 0, 4, 0);
   nir_deref_instr *element = nir_build_deref_array(b, array, index);
   element->arr.in_bounds = true;
   return nir_deref_buffer_address(b, 1, 64, &element->def, .access = access);
}

static nir_def *
push_at(nir_builder *b, uint16_t offset, uint16_t range)
{
   return nir_load_push_constant(b, 1, 32, nir_imm_int(b, offset),
                                 .base = 0, .range = range, .align_mul = 4);
}

static bool
match_linear_add_offset(nir_scalar idx, nir_scalar base_coord,
                        uint16_t *off_out, uint16_t *range_out)
{
   idx = strip_zero_add(idx);
   if (!alu(idx, nir_op_iadd))
      return false;
   nir_scalar a = strip_zero_add(arg(idx, 0));
   nir_scalar b = strip_zero_add(arg(idx, 1));
   if (same(a, base_coord) && match_push_any(b, off_out, range_out))
      return true;
   if (same(b, base_coord) && match_push_any(a, off_out, range_out))
      return true;
   return false;
}

static bool
match_linear_1d(nir_shader *nir, enum hk_identity_copy4_kind *kind_out,
                struct copy_buffer srcs[2], struct copy_buffer *dst,
                uint16_t *count_off, uint16_t *src_off, uint16_t *dst_off,
                uint16_t *push_bytes, unsigned *fadd_fp_ctrl)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE || nir->info.workgroup_size_variable ||
       nir->info.workgroup_size[0] == 0 || nir->info.workgroup_size[0] > 1024 ||
       nir->info.workgroup_size[1] != 1 || nir->info.workgroup_size[2] != 1 ||
       nir->info.shared_size != 0 ||
       (nir->info.num_ssbos != 2 && nir->info.num_ssbos != 3))
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_function_impl(other, nir) {
      if (other != impl)
         return false;
   }
   if (!acyclic(&impl->body, 4))
      return false;

   nir_intrinsic_instr *store = NULL;
   nir_intrinsic_instr *loads[2] = {NULL, NULL};
   unsigned load_count = 0;

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
            if (load_count == 2 || op->def.bit_size != 32 ||
                op->def.num_components != 1 ||
                (nir_intrinsic_access(op) & ~ACCESS_CAN_REORDER) != ACCESS_NON_WRITEABLE)
               return false;
            loads[load_count++] = op;
            break;
         case nir_intrinsic_store_deref:
            if (store || nir_intrinsic_write_mask(op) != 1 ||
                nir_intrinsic_access(op) != ACCESS_NON_READABLE ||
                op->src[1].ssa->bit_size != 32 || op->src[1].ssa->num_components != 1)
               return false;
            store = op;
            break;
         default:
            return false;
         }
      }
   }

   if (!store || load_count == 0)
      return false;

   nir_block *write_block = store->instr.block;
   nir_if *bounds = as_if(write_block->cf_node.parent);
   if (!bounds || bounds->cf_node.parent != &impl->cf_node)
      return false;

   const bool in_then = inside_arm(&write_block->cf_node, bounds, true);
   const bool in_else = inside_arm(&write_block->cf_node, bounds, false);
   if (in_then == in_else)
      return false;
   for (unsigned i = 0; i < load_count; ++i) {
      if (!inside_arm(&loads[i]->instr.block->cf_node, bounds, in_then))
         return false;
   }

   /* Match guard: `if (idx < count) { ... }` or `if (idx >= count) return;`. */
   nir_scalar cond = scalar(bounds->condition.ssa);
   nir_scalar idx = {0}, cnt = {0};
   if (in_then && alu(cond, nir_op_ilt)) {
      idx = arg(cond, 0);
      cnt = arg(cond, 1);
   } else if (in_else && alu(cond, nir_op_ige)) {
      idx = arg(cond, 0);
      cnt = arg(cond, 1);
   } else {
      return false;
   }
   idx = strip_zero_add(idx);
   if (!coord(idx, 0, nir->info.workgroup_size[0]))
      return false;

   uint16_t cnt_off = 0, cnt_range = 0;
   if (!match_push_any(cnt, &cnt_off, &cnt_range))
      return false;

   const bool allow_int = (load_count == 1);
   nir_scalar dst_idx;
   if (!match_buffer_typed(store, dst, &dst_idx, allow_int) ||
       dst->resource != nir_resource_type_read_write_storage_buffer)
      return false;
   dst_idx = strip_zero_add(dst_idx);

   nir_scalar val = scalar(store->src[1].ssa);
   if (load_count == 1) {
      if (val.def != &loads[0]->def || val.comp != 0)
         return false;
      nir_scalar src_idx;
      if (!match_buffer_typed(loads[0], &srcs[0], &src_idx, allow_int))
         return false;
      src_idx = strip_zero_add(src_idx);
      if (same(src_idx, idx) && same(dst_idx, idx)) {
         *kind_out = HK_IDENTITY_COPY4_LINEAR_COPY;
         *count_off = cnt_off;
         *src_off = 0;
         *dst_off = 0;
         *push_bytes = cnt_range;
         *fadd_fp_ctrl = 0;
         return true;
      }
      uint16_t s_off = 0, s_rng = 0, d_off = 0, d_rng = 0;
      if (match_linear_add_offset(src_idx, idx, &s_off, &s_rng) &&
          match_linear_add_offset(dst_idx, idx, &d_off, &d_rng)) {
         *kind_out = HK_IDENTITY_COPY4_LINEAR_OFFSET;
         *count_off = cnt_off;
         *src_off = s_off;
         *dst_off = d_off;
         *push_bytes = MAX2(cnt_range, MAX2(s_rng, d_rng));
         *fadd_fp_ctrl = 0;
         return true;
      }
      return false;
   }

   /* load_count == 2: dst[idx] = src0[idx] + src1[idx] (FP32 fadd). */
   if (!same(dst_idx, idx))
      return false;
   nir_alu_instr *add = alu(val, nir_op_fadd);
   if (!add || add->def.bit_size != 32 || add->def.num_components != 1)
      return false;
   nir_scalar a0 = arg(val, 0), a1 = arg(val, 1);
   nir_intrinsic_instr *lhs_load = NULL, *rhs_load = NULL;
   if (a0.def == &loads[0]->def && a1.def == &loads[1]->def) {
      lhs_load = loads[0];
      rhs_load = loads[1];
   } else if (a0.def == &loads[1]->def && a1.def == &loads[0]->def) {
      lhs_load = loads[1];
      rhs_load = loads[0];
   } else {
      return false;
   }

   nir_scalar s0_idx, s1_idx;
   if (!match_buffer_typed(lhs_load, &srcs[0], &s0_idx, false) ||
       !match_buffer_typed(rhs_load, &srcs[1], &s1_idx, false))
      return false;
   if (!same(strip_zero_add(s0_idx), idx) || !same(strip_zero_add(s1_idx), idx))
      return false;

   *kind_out = HK_IDENTITY_COPY4_LINEAR_FADD;
   *count_off = cnt_off;
   *src_off = 0;
   *dst_off = 0;
   *push_bytes = cnt_range;
   *fadd_fp_ctrl = add->fp_math_ctrl;
   return true;
}

struct pad4d_match {
   struct copy_buffer in_buf;
   struct copy_buffer out_buf;
   struct copy_buffer pad_buf;
   uint16_t in_shape_off[4];
   uint16_t out_shape_off[4];
   uint32_t pad_idx[4];
   uint16_t push_bytes;
};

static bool
match_pad_load(nir_scalar s, struct copy_buffer *pad_buf, bool *have_pad_buf,
               uint32_t *idx_out)
{
   s = nir_scalar_chase_movs(s);
   nir_intrinsic_instr *op = intr(s.def, nir_intrinsic_load_deref);
   if (!op || s.comp || op->def.bit_size != 32 || op->def.num_components != 1)
      return false;
   struct copy_buffer buf = {0};
   nir_scalar idx;
   if (!match_buffer_typed(op, &buf, &idx, true) || !nir_scalar_is_const(idx))
      return false;
   uint32_t ci = (uint32_t)nir_scalar_as_uint(idx);
   if (ci > 64)
      return false;
   if (*have_pad_buf) {
      if (!same_binding(*pad_buf, buf))
         return false;
   } else {
      *pad_buf = buf;
      *have_pad_buf = true;
   }
   *idx_out = ci;
   return true;
}

static bool
match_pad_sub(nir_scalar s, nir_scalar coord_v, struct copy_buffer *pad_buf,
              bool *have_pad_buf, uint32_t expected_idx)
{
   nir_scalar sub = {0};
   uint32_t ci = 0;
   return match_sub_expr(s, coord_v, &sub) &&
          match_pad_load(sub, pad_buf, have_pad_buf, &ci) &&
          ci == expected_idx;
}

struct rm4d_term {
   nir_scalar leaf;
   uint16_t factors[4];
   unsigned nfactors;
};

static bool
match_push_factors(nir_scalar s, uint16_t factors[4], unsigned *nfactors,
                   unsigned depth)
{
   if (depth > 4)
      return false;
   s = nir_scalar_chase_movs(s);
   uint16_t off = 0;
   if (match_push_any(s, &off, NULL)) {
      if (*nfactors >= 4)
         return false;
      factors[(*nfactors)++] = off;
      return true;
   }
   if (alu(s, nir_op_imul)) {
      unsigned saved = *nfactors;
      if (match_push_factors(arg(s, 0), factors, nfactors, depth + 1) &&
          match_push_factors(arg(s, 1), factors, nfactors, depth + 1))
         return true;
      *nfactors = saved;
   }
   return false;
}

static bool
expand_rm4d(nir_scalar s, struct rm4d_term *terms, unsigned *count,
            unsigned max_terms, unsigned depth)
{
   if (depth > 12)
      return false;
   s = nir_scalar_chase_movs(s);
   if (alu(s, nir_op_iadd)) {
      /* Stop at (coord - pad) leaves so subtractions remain intact. */
      nir_scalar a = arg(s, 0), b = arg(s, 1);
      if (alu(a, nir_op_ineg) || alu(b, nir_op_ineg)) {
         if (*count >= max_terms)
            return false;
         terms[(*count)++] = (struct rm4d_term){ .leaf = s };
         return true;
      }
      return expand_rm4d(a, terms, count, max_terms, depth + 1) &&
             expand_rm4d(b, terms, count, max_terms, depth + 1);
   }
   if (alu(s, nir_op_imul)) {
      uint16_t f[4];
      unsigned nf = 0;
      nir_scalar other;
      if (match_push_factors(arg(s, 1), f, &nf, 0)) {
         other = arg(s, 0);
      } else if (match_push_factors(arg(s, 0), f, &nf, 0)) {
         other = arg(s, 1);
      } else {
         return false;
      }
      unsigned start = *count;
      if (!expand_rm4d(other, terms, count, max_terms, depth + 1))
         return false;
      for (unsigned i = start; i < *count; ++i) {
         if (terms[i].nfactors + nf > 4)
            return false;
         for (unsigned k = 0; k < nf; ++k)
            terms[i].factors[terms[i].nfactors++] = f[k];
      }
      return true;
   }
   if (*count >= max_terms)
      return false;
   terms[(*count)++] = (struct rm4d_term){ .leaf = s };
   return true;
}

static bool
factors_match_multiset(const uint16_t *got, unsigned ngot,
                       const uint16_t *want, unsigned nwant)
{
   if (ngot != nwant)
      return false;
   bool used[4] = {false};
   for (unsigned i = 0; i < nwant; ++i) {
      bool found = false;
      for (unsigned j = 0; j < ngot; ++j) {
         if (!used[j] && got[j] == want[i]) {
            used[j] = true;
            found = true;
            break;
         }
      }
      if (!found)
         return false;
   }
   return true;
}

static bool
match_pad4d_out_idx(nir_scalar expr, const nir_scalar coords[4],
                    nir_scalar gx, nir_scalar gy,
                    uint16_t oh_off, uint16_t ow_off, uint16_t oc_off)
{
   struct rm4d_term terms[4];
   unsigned count = 0;
   if (!expand_rm4d(expr, terms, &count, 4, 0))
      return false;

   const uint16_t f0[3] = {oh_off, ow_off, oc_off};
   const uint16_t f1[2] = {ow_off, oc_off};
   const uint16_t f2[1] = {oc_off};
   bool have[4] = {false};

   for (unsigned i = 0; i < count; ++i) {
      if (same(terms[i].leaf, coords[0]) &&
          factors_match_multiset(terms[i].factors, terms[i].nfactors, f0, 3)) {
         if (have[0]) return false;
         have[0] = true;
      } else if (same(terms[i].leaf, coords[1]) &&
                 factors_match_multiset(terms[i].factors, terms[i].nfactors, f1, 2)) {
         if (have[1]) return false;
         have[1] = true;
      } else if (same(terms[i].leaf, coords[2]) &&
                 factors_match_multiset(terms[i].factors, terms[i].nfactors, f2, 1)) {
         if (have[2]) return false;
         have[2] = true;
      } else if (same(terms[i].leaf, coords[3]) && terms[i].nfactors == 0) {
         if (have[3]) return false;
         have[3] = true;
      } else if (same(terms[i].leaf, gy) &&
                 factors_match_multiset(terms[i].factors, terms[i].nfactors, f1, 2)) {
         if (have[0] || have[1]) return false;
         have[0] = have[1] = true;
      } else if (same(terms[i].leaf, gx) && terms[i].nfactors == 0) {
         if (have[2] || have[3]) return false;
         have[2] = have[3] = true;
      } else {
         return false;
      }
   }
   return have[0] && have[1] && have[2] && have[3];
}

static bool
match_pad4d_in_idx(nir_scalar expr, const nir_scalar coords[4],
                   struct copy_buffer *pad_buf, bool *have_pad,
                   const uint32_t pad_idx[4], const uint16_t in_off[4])
{
   struct rm4d_term terms[4];
   unsigned count = 0;
   if (!expand_rm4d(expr, terms, &count, 4, 0) || count != 4)
      return false;

   const uint16_t f0[3] = {in_off[1], in_off[2], in_off[3]};
   const uint16_t f1[2] = {in_off[2], in_off[3]};
   const uint16_t f2[1] = {in_off[3]};
   bool have[4] = {false};

   for (unsigned i = 0; i < count; ++i) {
      if (match_pad_sub(terms[i].leaf, coords[0], pad_buf, have_pad, pad_idx[0]) &&
          factors_match_multiset(terms[i].factors, terms[i].nfactors, f0, 3)) {
         if (have[0]) return false;
         have[0] = true;
      } else if (match_pad_sub(terms[i].leaf, coords[1], pad_buf, have_pad, pad_idx[1]) &&
                 factors_match_multiset(terms[i].factors, terms[i].nfactors, f1, 2)) {
         if (have[1]) return false;
         have[1] = true;
      } else if (match_pad_sub(terms[i].leaf, coords[2], pad_buf, have_pad, pad_idx[2]) &&
                 factors_match_multiset(terms[i].factors, terms[i].nfactors, f2, 1)) {
         if (have[2]) return false;
         have[2] = true;
      } else if (match_pad_sub(terms[i].leaf, coords[3], pad_buf, have_pad, pad_idx[3]) &&
                 terms[i].nfactors == 0) {
         if (have[3]) return false;
         have[3] = true;
      } else {
         return false;
      }
   }
   return have[0] && have[1] && have[2] && have[3];
}

static bool
match_pad4d(nir_shader *nir, struct pad4d_match *out)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE || nir->info.workgroup_size_variable ||
       !hk_identity_copy4_supported_local(nir->info.workgroup_size) ||
       nir->info.num_ssbos != 3 || nir->num_uniforms > HK_COMPUTE_OPT_PUSH_BYTES)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_function_impl(other, nir) {
      if (other != impl)
         return false;
   }
   if (!acyclic(&impl->body, 16))
      return false;

   nir_intrinsic_instr *stores[2] = {NULL, NULL};
   unsigned nstores = 0;
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
            if (op->def.bit_size != 32 || op->def.num_components != 1 ||
                (nir_intrinsic_access(op) & ~(ACCESS_CAN_REORDER | ACCESS_RESTRICT | ACCESS_IN_BOUNDS)) !=
                   ACCESS_NON_WRITEABLE)
               return false;
            break;
         case nir_intrinsic_store_deref:
            if (nstores >= 2 || op->num_components != 1 ||
                op->src[1].ssa->bit_size != 32 || nir_intrinsic_write_mask(op) != 1 ||
                (nir_intrinsic_access(op) & ~(ACCESS_NON_READABLE | ACCESS_RESTRICT | ACCESS_IN_BOUNDS)))
               return false;
            stores[nstores++] = op;
            break;
         default:
            return false;
         }
      }
   }
   if (nstores != 2)
      return false;

   nir_intrinsic_instr *zero_st = NULL, *copy_st = NULL;
   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar v = scalar(stores[i]->src[1].ssa);
      if (imm(v, 0))
         zero_st = stores[i];
      else if (intr(v.def, nir_intrinsic_load_deref))
         copy_st = stores[i];
   }
   if (!zero_st || !copy_st)
      return false;

   nir_if *pad_if = as_if(zero_st->instr.block->cf_node.parent);
   if (!pad_if || pad_if != as_if(copy_st->instr.block->cf_node.parent) ||
       !inside_arm(&zero_st->instr.block->cf_node, pad_if, true) ||
       !inside_arm(&copy_st->instr.block->cf_node, pad_if, false))
      return false;

   nir_if *outer_if = as_if(pad_if->cf_node.parent);
   if (!outer_if || outer_if->cf_node.parent != &impl->cf_node ||
       !inside_arm(&pad_if->cf_node, outer_if, false))
      return false;

   nir_scalar o_x, o_y;
   if (!match_short_or(scalar(outer_if->condition.ssa), &o_x, &o_y) ||
       !alu(o_x, nir_op_ige) || !alu(o_y, nir_op_ige))
      return false;

   nir_scalar gx = arg(o_x, 0), gy = arg(o_y, 0);
   if (!coord(gx, 0, nir->info.workgroup_size[0]) ||
       !coord(gy, 1, nir->info.workgroup_size[1]))
      return false;

   nir_scalar x_lim = arg(o_x, 1), y_lim = arg(o_y, 1);
   if (!alu(x_lim, nir_op_imul) || !alu(y_lim, nir_op_imul))
      return false;
   uint16_t ow_off = 0, oc_off = 0, on_off = 0, oh_off = 0, pb = 0, rtmp = 0;
   if (!match_push_any(arg(x_lim, 0), &ow_off, &rtmp)) return false;
   pb = MAX2(pb, rtmp);
   if (!match_push_any(arg(x_lim, 1), &oc_off, &rtmp)) return false;
   pb = MAX2(pb, rtmp);
   if (!match_push_any(arg(y_lim, 0), &on_off, &rtmp)) return false;
   pb = MAX2(pb, rtmp);
   if (!match_push_any(arg(y_lim, 1), &oh_off, &rtmp)) return false;
   pb = MAX2(pb, rtmp);

   struct copy_buffer out_b0 = {0}, out_b1 = {0};
   nir_scalar out_idx0, out_idx1;
   if (!match_buffer(zero_st, &out_b0, &out_idx0) ||
       !match_buffer(copy_st, &out_b1, &out_idx1) ||
       !same_binding(out_b0, out_b1) || !same(out_idx0, out_idx1))
      return false;

   nir_scalar terms[8];
   nir_scalar cur = scalar(pad_if->condition.ssa);
   for (int k = 7; k >= 1; --k) {
      nir_scalar prev, rhs;
      if (!match_short_or(cur, &prev, &rhs))
         return false;
      terms[k] = rhs;
      cur = prev;
   }
   terms[0] = cur;

   struct copy_buffer pad_buf = {0};
   bool have_pad = false;
   nir_scalar coords[4];
   uint32_t pad_idx[4] = {0};
   uint16_t in_off[4] = {0};
   for (unsigned d = 0; d < 4; ++d) {
      if (!alu(terms[d], nir_op_ilt))
         return false;
      coords[d] = arg(terms[d], 0);
      if (!match_pad_load(arg(terms[d], 1), &pad_buf, &have_pad, &pad_idx[d]))
         return false;
   }
   for (unsigned d = 0; d < 4; ++d) {
      if (!alu(terms[4 + d], nir_op_ige))
         return false;
      if (!match_pad_sub(arg(terms[4 + d], 0), coords[d], &pad_buf, &have_pad, pad_idx[d]))
         return false;
      if (!match_push_any(arg(terms[4 + d], 1), &in_off[d], &rtmp))
         return false;
      pb = MAX2(pb, rtmp);
   }

   nir_scalar dn, dd;
   if (!match_div(coords[0], &dn, &dd) || !same(dn, gy))
      return false;
   uint16_t div_oh = 0;
   if (!match_push_any(dd, &div_oh, NULL))
      return false;
   if (div_oh == on_off) {
      uint16_t tmp = on_off; on_off = oh_off; oh_off = tmp;
   } else if (div_oh != oh_off) {
      return false;
   }
   nir_scalar md;
   if (!match_mod(coords[1], gy, &md) || !match_push_eq(md, oh_off))
      return false;

   if (!match_div(coords[2], &dn, &dd) || !same(dn, gx))
      return false;
   uint16_t div_oc = 0;
   if (!match_push_any(dd, &div_oc, NULL))
      return false;
   if (div_oc == ow_off) {
      uint16_t tmp = ow_off; ow_off = oc_off; oc_off = tmp;
   } else if (div_oc != oc_off) {
      return false;
   }
   if (!match_mod(coords[3], gx, &md) || !match_push_eq(md, oc_off))
      return false;

   if (!match_pad4d_out_idx(out_idx0, coords, gx, gy, oh_off, ow_off, oc_off))
      return false;

   nir_intrinsic_instr *in_ld = intr(copy_st->src[1].ssa, nir_intrinsic_load_deref);
   struct copy_buffer in_buf = {0};
   nir_scalar in_idx;
   if (!in_ld || !match_buffer(in_ld, &in_buf, &in_idx) ||
       same_binding(in_buf, out_b0) || same_binding(pad_buf, out_b0) ||
       same_binding(in_buf, pad_buf))
      return false;

   if (!match_pad4d_in_idx(in_idx, coords, &pad_buf, &have_pad, pad_idx, in_off))
      return false;

   *out = (struct pad4d_match) {
      .in_buf = in_buf,
      .out_buf = out_b0,
      .pad_buf = pad_buf,
      .in_shape_off = {in_off[0], in_off[1], in_off[2], in_off[3]},
      .out_shape_off = {on_off, oh_off, ow_off, oc_off},
      .pad_idx = {pad_idx[0], pad_idx[1], pad_idx[2], pad_idx[3]},
      .push_bytes = pb,
   };
   return true;
}

static nir_shader *
build_pad4d_vec4(nir_shader *nir, const struct pad4d_match *m,
                 const uint16_t orig_local[3], struct hk_identity_copy4 *info)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list old;
   nir_cf_extract(&old, nir_before_impl(impl), nir_after_impl(impl));
   nir_cf_delete(&old);
   nir_builder b = nir_builder_at(nir_after_block(nir_start_block(impl)));

   nir->info.workgroup_size[0] = 64;
   nir->info.workgroup_size[1] = 2;
   nir->info.workgroup_size[2] = 1;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gx = nir_channel(&b, gid, 0);
   nir_def *gy = nir_channel(&b, gid, 1);

   nir_def *in_n = push_at(&b, m->in_shape_off[0], m->push_bytes);
   nir_def *in_h = push_at(&b, m->in_shape_off[1], m->push_bytes);
   nir_def *in_w = push_at(&b, m->in_shape_off[2], m->push_bytes);
   nir_def *in_c = push_at(&b, m->in_shape_off[3], m->push_bytes);
   nir_def *out_n = push_at(&b, m->out_shape_off[0], m->push_bytes);
   nir_def *out_h = push_at(&b, m->out_shape_off[1], m->push_bytes);
   nir_def *out_w = push_at(&b, m->out_shape_off[2], m->push_bytes);
   nir_def *out_c = push_at(&b, m->out_shape_off[3], m->push_bytes);

   nir_def *in_c4 = nir_ushr_imm(&b, in_c, 2);
   nir_def *out_c4 = nir_ushr_imm(&b, out_c, 2);
   nir_def *out_x_lim = nir_imul(&b, out_w, out_c4);
   nir_def *out_y_lim = nir_imul(&b, out_n, out_h);

   nir_if *active = nir_push_if(&b, nir_iand(&b, nir_ult(&b, gx, out_x_lim),
                                                 nir_ult(&b, gy, out_y_lim)));
   {
      const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
      nir_def *p[4];
      for (unsigned d = 0; d < 4; ++d) {
         nir_def *pa = address_scalar(&b, m->pad_buf, nir_imm_int(&b, m->pad_idx[d]), read, true);
         p[d] = nir_load_global(&b, 1, 32, pa, .align_mul = 4, .access = read);
      }

      nir_def *zero_n = nir_imm_int(&b, 0);
      nir_def *n_div = NULL, *h_div = NULL;
      nir_if *n1_if = nir_push_if(&b, nir_ieq_imm(&b, out_n, 1));
      {
         /* Uniform out_n == 1: avoid udiv/umod on gy. */
      }
      nir_push_else(&b, n1_if);
      {
         struct hk_nir_fast_udiv h_fd = hk_nir_fast_udiv_prepare(&b, out_h);
         hk_nir_fast_udivmod(&b, &h_fd, gy, &n_div, &h_div);
      }
      nir_pop_if(&b, n1_if);
      nir_def *n = nir_if_phi(&b, zero_n, n_div);
      nir_def *h = nir_if_phi(&b, gy, h_div);
      struct hk_nir_fast_udiv c4_fd = hk_nir_fast_udiv_prepare(&b, out_c4);
      nir_def *w = NULL, *c4 = NULL;
      hk_nir_fast_udivmod(&b, &c4_fd, gx, &w, &c4);

      nir_def *in_n_i = nir_isub(&b, n, p[0]);
      nir_def *in_h_i = nir_isub(&b, h, p[1]);
      nir_def *in_w_i = nir_isub(&b, w, p[2]);
      nir_def *n_ok = nir_iand(&b, nir_ige(&b, n, p[0]), nir_ilt(&b, in_n_i, in_n));
      nir_def *h_ok = nir_iand(&b, nir_ige(&b, h, p[1]), nir_ilt(&b, in_h_i, in_h));
      nir_def *w_ok = nir_iand(&b, nir_ige(&b, w, p[2]), nir_ilt(&b, in_w_i, in_w));
      nir_def *spatial_ok = nir_iand(&b, nir_iand(&b, n_ok, h_ok), w_ok);

      nir_def *zero_f = nir_imm_float(&b, 0.0f);
      nir_def *zero_v4 = nir_vec4(&b, zero_f, zero_f, zero_f, zero_f);

      nir_def *val_spatial = NULL;
      nir_if *sp_if = nir_push_if(&b, spatial_ok);
      {
         nir_def *in_sp4 =
            nir_imul(&b, nir_iadd(&b, nir_imul(&b, nir_iadd(&b, nir_imul(&b, in_n_i, in_h), in_h_i), in_w), in_w_i), in_c4);
         nir_def *v_fast = NULL, *v_slow = NULL;
         nir_if *p3_zero = nir_push_if(&b, nir_ieq_imm(&b, p[3], 0));
         {
            nir_def *v_ld = NULL;
            nir_if *c_ok = nir_push_if(&b, nir_ult(&b, c4, in_c4));
            {
               nir_def *in_addr = address(&b, m->in_buf, nir_iadd(&b, in_sp4, c4), read);
               v_ld = nir_load_global(&b, 4, 32, in_addr, .align_mul = 16, .access = read);
            }
            nir_push_else(&b, c_ok);
            nir_pop_if(&b, c_ok);
            v_fast = nir_if_phi(&b, v_ld, zero_v4);
         }
         nir_push_else(&b, p3_zero);
         {
            nir_def *in_sp = nir_ishl_imm(&b, in_sp4, 2);
            nir_def *c_base = nir_ishl_imm(&b, c4, 2);
            nir_def *ch_vals[4];
            for (unsigned k = 0; k < 4; ++k) {
               nir_def *c_k = nir_iadd_imm(&b, c_base, k);
               nir_def *ci = nir_isub(&b, c_k, p[3]);
               nir_def *v_k = NULL;
               nir_def *ci_ok = nir_iand(&b, nir_ige(&b, c_k, p[3]), nir_ilt(&b, ci, in_c));
               nir_if *ck = nir_push_if(&b, ci_ok);
               {
                  nir_def *sa = address_scalar(&b, m->in_buf, nir_iadd(&b, in_sp, ci), read, false);
                  v_k = nir_load_global(&b, 1, 32, sa, .align_mul = 4, .access = read);
               }
               nir_push_else(&b, ck);
               nir_pop_if(&b, ck);
               ch_vals[k] = nir_if_phi(&b, v_k, zero_f);
            }
            v_slow = nir_vec4(&b, ch_vals[0], ch_vals[1], ch_vals[2], ch_vals[3]);
         }
         nir_pop_if(&b, p3_zero);
         val_spatial = nir_if_phi(&b, v_fast, v_slow);
      }
      nir_push_else(&b, sp_if);
      nir_pop_if(&b, sp_if);
      nir_def *val = nir_if_phi(&b, val_spatial, zero_v4);

      nir_def *out_idx4 = nir_iadd(&b, nir_imul(&b, gy, out_x_lim), gx);
      nir_def *out_addr = address(&b, m->out_buf, out_idx4, ACCESS_NON_READABLE);
      nir_store_global(&b, val, out_addr, .align_mul = 16, .write_mask = 0xf,
                       .access = ACCESS_NON_READABLE);
   }
   nir_pop_if(&b, active);

   nir_opt_dce(nir);
   nir_shader_gather_info(nir, impl);
   memset(info, 0, sizeof(*info));
   info->kind = HK_IDENTITY_COPY4_PAD4;
   info->src_count = 1;
   info->src_set = m->in_buf.set;
   info->dst_set = m->out_buf.set;
   info->pad_set = m->pad_buf.set;
   info->src_binding = m->in_buf.binding;
   info->dst_binding = m->out_buf.binding;
   info->pad_binding = m->pad_buf.binding;
   info->push_bytes = m->push_bytes;
   for (unsigned d = 0; d < 4; ++d) {
      info->in_shape_off[d] = m->in_shape_off[d];
      info->out_shape_off[d] = m->out_shape_off[d];
      info->pad_idx[d] = m->pad_idx[d];
   }
   for (unsigned d = 0; d < 3; ++d)
      info->local[d] = orig_local[d];
   return nir;
}

static void
emit_im2col_kh3_s1_4y(nir_builder *b, struct copy_buffer src, struct copy_buffer dst,
                      bool vec4_indexed, unsigned comps, unsigned align_mul,
                      nir_def *s0, nir_def *in_row_stride,
                      nir_def *d0, nir_def *dst_row_stride, nir_def *kw_stride)
{
   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   const unsigned wmask = (1u << comps) - 1u;
   nir_def *s1 = nir_iadd(b, s0, in_row_stride);
   nir_def *s2 = nir_iadd(b, s1, in_row_stride);
   nir_def *v0 = nir_load_global(
      b, comps, 32,
      vec4_indexed ? address(b, src, s0, read) : address_scalar(b, src, s0, read, false),
      .align_mul = align_mul, .access = read);
   nir_def *v1 = nir_load_global(
      b, comps, 32,
      vec4_indexed ? address(b, src, s1, read) : address_scalar(b, src, s1, read, false),
      .align_mul = align_mul, .access = read);
   nir_def *v2 = nir_load_global(
      b, comps, 32,
      vec4_indexed ? address(b, src, s2, read) : address_scalar(b, src, s2, read, false),
      .align_mul = align_mul, .access = read);

   nir_def *s_cur = s2;
   nir_def *d_cur = d0;
   nir_def *win[3] = { v0, v1, v2 };
   for (unsigned r = 0; r < 4; ++r) {
      nir_def *dc = d_cur;
      for (unsigned k = 0; k < 3; ++k) {
         nir_store_global(
            b, win[k],
            vec4_indexed ? address(b, dst, dc, ACCESS_NON_READABLE)
                         : address_scalar(b, dst, dc, ACCESS_NON_READABLE, false),
            .align_mul = align_mul, .write_mask = wmask, .access = ACCESS_NON_READABLE);
         if (k < 2)
            dc = nir_iadd(b, dc, kw_stride);
      }
      if (r < 3) {
         s_cur = nir_iadd(b, s_cur, in_row_stride);
         d_cur = nir_iadd(b, d_cur, dst_row_stride);
         nir_def *vn = nir_load_global(
            b, comps, 32,
            vec4_indexed ? address(b, src, s_cur, read) : address_scalar(b, src, s_cur, read, false),
            .align_mul = align_mul, .access = read);
         win[0] = win[1];
         win[1] = win[2];
         win[2] = vn;
      }
   }
}

static void
emit_im2col_kh3_s2_4y(nir_builder *b, struct copy_buffer src, struct copy_buffer dst,
                      bool vec4_indexed, unsigned comps, unsigned align_mul,
                      nir_def *s0, nir_def *in_row_stride,
                      nir_def *d0, nir_def *dst_row_stride, nir_def *kw_stride,
                      nir_def *zero_i)
{
   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   const unsigned wmask = (1u << comps) - 1u;
   nir_def *v_init = nir_load_global(
      b, comps, 32,
      vec4_indexed ? address(b, src, s0, read) : address_scalar(b, src, s0, read, false),
      .align_mul = align_mul, .access = read);

   nir_loop *lp = nir_push_loop(b);
   lp->control = nir_loop_control_dont_unroll;
   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&lp->cf_node));
   nir_block *hdr = nir_loop_first_block(lp);

   nir_phi_instr *r_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&r_phi->instr, &r_phi->def, 1, 32);
   nir_phi_instr_add_src(r_phi, pre, zero_i);

   nir_phi_instr *s_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&s_phi->instr, &s_phi->def, 1, 32);
   nir_phi_instr_add_src(s_phi, pre, s0);

   nir_phi_instr *d_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&d_phi->instr, &d_phi->def, 1, 32);
   nir_phi_instr_add_src(d_phi, pre, d0);

   nir_phi_instr *v_phi = nir_phi_instr_create(b->shader);
   nir_def_init(&v_phi->instr, &v_phi->def, comps, 32);
   nir_phi_instr_add_src(v_phi, pre, v_init);

   nir_def *rv = &r_phi->def;
   nir_if *done = nir_push_if(b, nir_ige_imm(b, rv, 4));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, done);

   nir_def *s_cur = &s_phi->def;
   nir_def *s1 = nir_iadd(b, s_cur, in_row_stride);
   nir_def *s2 = nir_iadd(b, s1, in_row_stride);
   nir_def *v1 = nir_load_global(
      b, comps, 32,
      vec4_indexed ? address(b, src, s1, read) : address_scalar(b, src, s1, read, false),
      .align_mul = align_mul, .access = read);
   nir_def *v2 = nir_load_global(
      b, comps, 32,
      vec4_indexed ? address(b, src, s2, read) : address_scalar(b, src, s2, read, false),
      .align_mul = align_mul, .access = read);

   nir_def *d_cur = &d_phi->def;
   nir_def *d1 = nir_iadd(b, d_cur, kw_stride);
   nir_def *d2 = nir_iadd(b, d1, kw_stride);
   nir_store_global(
      b, &v_phi->def,
      vec4_indexed ? address(b, dst, d_cur, ACCESS_NON_READABLE)
                   : address_scalar(b, dst, d_cur, ACCESS_NON_READABLE, false),
      .align_mul = align_mul, .write_mask = wmask, .access = ACCESS_NON_READABLE);
   nir_store_global(
      b, v1,
      vec4_indexed ? address(b, dst, d1, ACCESS_NON_READABLE)
                   : address_scalar(b, dst, d1, ACCESS_NON_READABLE, false),
      .align_mul = align_mul, .write_mask = wmask, .access = ACCESS_NON_READABLE);
   nir_store_global(
      b, v2,
      vec4_indexed ? address(b, dst, d2, ACCESS_NON_READABLE)
                   : address_scalar(b, dst, d2, ACCESS_NON_READABLE, false),
      .align_mul = align_mul, .write_mask = wmask, .access = ACCESS_NON_READABLE);

   nir_block *back = nir_loop_last_block(lp);
   nir_phi_instr_add_src(r_phi, back, nir_iadd_imm(b, rv, 1));
   nir_phi_instr_add_src(s_phi, back, s2);
   nir_phi_instr_add_src(d_phi, back, nir_iadd(b, d_cur, dst_row_stride));
   nir_phi_instr_add_src(v_phi, back, v2);
   nir_instr_insert(nir_before_block(hdr), &r_phi->instr);
   nir_instr_insert(nir_before_block(hdr), &s_phi->instr);
   nir_instr_insert(nir_before_block(hdr), &d_phi->instr);
   nir_instr_insert(nir_before_block(hdr), &v_phi->instr);
   nir_pop_loop(b, lp);
}

nir_shader *
hk_nir_im2col_vec4_with_bound(const nir_shader *orig,
                              struct hk_pack_loop_bound *loop_bound_out)
{
   struct hk_nir_loop_binding_list lb_list = {0};
   nir_shader *nir = nir_shader_clone(NULL, orig);
   nir_opt_constant_folding(nir);

   struct copy_buffer src = {0}, dst = {0};
   struct im2col_params p = {0};
   bool vec4_ok = false;
   if (!match_im2col(nir, &src, &dst, &p, &vec4_ok) || !vec4_ok) {
      ralloc_free(nir);
      return NULL;
   }

   const uint32_t im2_off = hk_im2col_push_offset(p.push_bytes);
   const uint32_t total_push_bytes = im2_off + sizeof(struct hk_im2col_push);
   if (total_push_bytes > 256u) {
      ralloc_free(nir);
      return NULL;
   }
   nir->num_uniforms = total_push_bytes;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list old;
   nir_cf_extract(&old, nir_before_impl(impl), nir_after_impl(impl));
   nir_cf_delete(&old);
   nir_builder b = nir_builder_at(nir_after_block(nir_start_block(impl)));

   /* 3D grid: X = ceil(kw / 4), Y = out_x, Z = ceil(rows_y / 4), processing 4
    * vertical output rows per thread so kh == 3 interior pixels reuse shared
    * input rows (sh == 1: 6 loads / 12 stores, -50%; sh == 2: 9 loads / 12
    * stores, -25%) across both vec4 and rem == 1 scalar tail lanes. */
   nir->info.workgroup_size[0] = 4;
   nir->info.workgroup_size[1] = 8;
   nir->info.workgroup_size[2] = 2;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *gx = nir_channel(&b, gid, 0);
   nir_def *out_x = nir_channel(&b, gid, 1);
   nir_def *oy_quad = nir_channel(&b, gid, 2);
   nir_def *out_y0 = nir_ishl_imm(&b, oy_quad, 2);

   const bool has_border = p.bound[IM2COL_BORDER_WIDTH];
   nir_def *bw = has_border ? push_at(&b, p.off[IM2COL_BORDER_WIDTH], total_push_bytes) : NULL;
   nir_def *bh = has_border ? push_at(&b, p.off[IM2COL_BORDER_HEIGHT], total_push_bytes) : NULL;
   nir_def *iw = push_at(&b, p.off[IM2COL_INPUT_WIDTH], total_push_bytes);
   nir_def *ih = has_border ? push_at(&b, p.off[IM2COL_INPUT_HEIGHT], total_push_bytes) : NULL;
   nir_def *kw = push_at(&b, p.off[IM2COL_KERNEL_WIDTH], total_push_bytes);
   nir_def *pw = push_at(&b, p.off[IM2COL_PATCHES_WIDTH], total_push_bytes);
   nir_def *sw = push_at(&b, p.off[IM2COL_STRIDE_WIDTH], total_push_bytes);
   nir_def *sh = push_at(&b, p.off[IM2COL_STRIDE_HEIGHT], total_push_bytes);
   nir_def *ch = push_at(&b, p.off[IM2COL_CHANNELS], total_push_bytes);

   nir_def *cols = push_at(&b, (uint16_t)(im2_off + offsetof(struct hk_im2col_push, cols)),
                           total_push_bytes);
   nir_def *rows_y = push_at(&b, (uint16_t)(im2_off + offsetof(struct hk_im2col_push, rows_y)),
                             total_push_bytes);
   nir_def *kh = push_at(&b, (uint16_t)(im2_off + offsetof(struct hk_im2col_push, kh)),
                         total_push_bytes);

   nir_def *unit = has_border ? nir_iand(&b, nir_ieq_imm(&b, sw, 1), nir_ieq_imm(&b, sh, 1)) : NULL;
   nir_def *or_align = nir_ior(&b, nir_ior(&b, ch, iw), nir_ior(&b, kw, pw));
   nir_def *is_vec4 = nir_ieq_imm(&b, nir_iand_imm(&b, or_align, 3), 0);

   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   nir_def *zero_i = nir_imm_int(&b, 0);
   nir_def *zero_f = nir_imm_float(&b, 0.0f);
   nir_def *zero_v4 = nir_vec4(&b, zero_f, zero_f, zero_f, zero_f);

   nir_if *v4_if = nir_push_if(&b, is_vec4);
   {
      nir_def *kw4 = nir_ushr_imm(&b, kw, 2);
      nir_def *in_b = nir_iand(&b, nir_iand(&b, nir_ult(&b, gx, kw4),
                                                nir_ult(&b, out_x, cols)),
                                   nir_ult(&b, out_y0, rows_y));
      nir_if *active = nir_push_if(&b, in_b);
      {
         nir_def *pw4 = nir_ushr_imm(&b, pw, 2);
         nir_def *c4 = nir_ushr_imm(&b, ch, 2);
         nir_def *iw4 = nir_ushr_imm(&b, iw, 2);
         nir_def *row0 = nir_iadd(&b, nir_imul(&b, out_y0, cols), out_x);
         nir_def *x4 = nir_iadd(&b, nir_imul(&b, out_x, nir_imul(&b, sw, c4)), gx);
         nir_def *y_base0 = nir_imul(&b, out_y0, sh);
         nir_def *dst_base4_0 = nir_iadd(&b, nir_imul(&b, row0, pw4), gx);
         nir_def *dst_row_stride4 = nir_imul(&b, cols, pw4);
         nir_def *has_y3 = nir_ult(&b, nir_iadd_imm(&b, out_y0, 3), rows_y);

         nir_def *x_eff4 = x4;
         nir_def *y_eff0 = y_base0;
         nir_def *x_ok = NULL;
         if (has_border) {
            nir_def *pad_x4 = nir_imul(&b, bw, c4);
            x_eff4 = nir_bcsel(&b, unit, nir_isub(&b, x4, pad_x4), x4);
            y_eff0 = nir_bcsel(&b, unit, nir_isub(&b, y_base0, bh), y_base0);
            x_ok = nir_ult(&b, x_eff4, iw4);
         }

         nir_def *kh3_4y = nir_iand(&b, nir_ieq_imm(&b, kh, 3), has_y3);
         nir_def *kh3_s1_4y = nir_iand(&b, kh3_4y, nir_ieq_imm(&b, sh, 1));
         nir_def *kh3_s2_4y = nir_iand(&b, kh3_4y, nir_ieq_imm(&b, sh, 2));
         if (has_border) {
            kh3_s1_4y = nir_iand(&b, kh3_s1_4y,
               nir_iand(&b, x_ok, nir_iand(&b, nir_uge_imm(&b, ih, 6),
                                               nir_uge(&b, nir_iadd_imm(&b, ih, -6), y_eff0))));
            kh3_s2_4y = nir_iand(&b, kh3_s2_4y,
               nir_iand(&b, x_ok, nir_iand(&b, nir_uge_imm(&b, ih, 9),
                                               nir_uge(&b, nir_iadd_imm(&b, ih, -9), y_eff0))));
         }

         nir_if *k3_s1_if = nir_push_if(&b, kh3_s1_4y);
         {
            /* kh == 3, sh == 1, 4 interior rows: 6 loads, 12 stores (-50%). */
            nir_def *s0 = nir_iadd(&b, nir_imul(&b, y_eff0, iw4), x_eff4);
            emit_im2col_kh3_s1_4y(&b, src, dst, true, 4, 16, s0, iw4,
                                  dst_base4_0, dst_row_stride4, kw4);
         }
         nir_push_else(&b, k3_s1_if);
         {
            nir_if *k3_s2_if = nir_push_if(&b, kh3_s2_4y);
            {
               /* kh == 3, sh == 2, 4 interior rows: 9 loads, 12 stores (-25%). */
               nir_def *s0 = nir_iadd(&b, nir_imul(&b, y_eff0, iw4), x_eff4);
               emit_im2col_kh3_s2_4y(&b, src, dst, true, 4, 16, s0, iw4,
                                     dst_base4_0, dst_row_stride4, kw4, zero_i);
            }
            nir_push_else(&b, k3_s2_if);
            {
               nir_loop *dy_lp = nir_push_loop(&b);
               dy_lp->control = nir_loop_control_dont_unroll;
               nir_block *dy_pre = nir_cf_node_as_block(nir_cf_node_prev(&dy_lp->cf_node));
               nir_block *dy_hdr = nir_loop_first_block(dy_lp);
               nir_phi_instr *dy_phi = nir_phi_instr_create(b.shader);
               nir_def_init(&dy_phi->instr, &dy_phi->def, 1, 32);
               nir_phi_instr_add_src(dy_phi, dy_pre, zero_i);
               nir_def *dy_v = &dy_phi->def;
               nir_if *dy_done = nir_push_if(&b, nir_ige_imm(&b, dy_v, 4));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, dy_done);

               nir_def *oy = nir_iadd(&b, out_y0, dy_v);
               nir_if *oy_ok = nir_push_if(&b, nir_ult(&b, oy, rows_y));
               {
                  nir_def *y_e = nir_iadd(&b, y_eff0, nir_imul(&b, dy_v, sh));
                  nir_def *dst_b4 = nir_iadd(&b, dst_base4_0, nir_imul(&b, dy_v, dst_row_stride4));
                  nir_loop *lp = nir_push_loop(&b);
                  hk_nir_loop_binding_add(&lb_list, lp, HK_PACK_LOOP_PARAM_IM2COL_KH);
                  lp->control = nir_loop_control_dont_unroll;
                  nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&lp->cf_node));
                  nir_block *hdr = nir_loop_first_block(lp);
                  nir_phi_instr *ky_phi = nir_phi_instr_create(b.shader);
                  nir_def_init(&ky_phi->instr, &ky_phi->def, 1, 32);
                  nir_phi_instr_add_src(ky_phi, pre, zero_i);
                  nir_phi_instr *di_phi = nir_phi_instr_create(b.shader);
                  nir_def_init(&di_phi->instr, &di_phi->def, 1, 32);
                  nir_phi_instr_add_src(di_phi, pre, dst_b4);

                  nir_def *ky = &ky_phi->def;
                  nir_def *dst_idx4 = &di_phi->def;
                  nir_if *done = nir_push_if(&b, nir_ige(&b, ky, kh));
                  nir_jump(&b, nir_jump_break);
                  nir_pop_if(&b, done);

                  nir_def *y_cur = nir_iadd(&b, y_e, ky);
                  nir_def *val = NULL;
                  if (has_border) {
                     nir_def *v_ld = NULL;
                     nir_if *ok_if = nir_push_if(&b, nir_iand(&b, x_ok, nir_ult(&b, y_cur, ih)));
                     {
                        nir_def *s_idx4 = nir_iadd(&b, nir_imul(&b, y_cur, iw4), x_eff4);
                        nir_def *sa = address(&b, src, s_idx4, read);
                        v_ld = nir_load_global(&b, 4, 32, sa, .align_mul = 16, .access = read);
                     }
                     nir_push_else(&b, ok_if);
                     nir_pop_if(&b, ok_if);
                     val = nir_if_phi(&b, v_ld, zero_v4);
                  } else {
                     nir_def *s_idx4 = nir_iadd(&b, nir_imul(&b, y_cur, iw4), x_eff4);
                     nir_def *sa = address(&b, src, s_idx4, read);
                     val = nir_load_global(&b, 4, 32, sa, .align_mul = 16, .access = read);
                  }

                  nir_def *da = address(&b, dst, dst_idx4, ACCESS_NON_READABLE);
                  nir_store_global(&b, val, da, .align_mul = 16, .write_mask = 0xf,
                                   .access = ACCESS_NON_READABLE);

                  nir_block *back = nir_loop_last_block(lp);
                  nir_phi_instr_add_src(ky_phi, back, nir_iadd_imm(&b, ky, 1));
                  nir_phi_instr_add_src(di_phi, back, nir_iadd(&b, dst_idx4, kw4));
                  nir_instr_insert(nir_before_block(hdr), &ky_phi->instr);
                  nir_instr_insert(nir_before_block(hdr), &di_phi->instr);
                  nir_pop_loop(&b, lp);
               }
               nir_pop_if(&b, oy_ok);

               nir_block *dy_back = nir_cursor_current_block(b.cursor);
               nir_phi_instr_add_src(dy_phi, dy_back, nir_iadd_imm(&b, dy_v, 1));
               nir_instr_insert(nir_before_block(dy_hdr), &dy_phi->instr);
               nir_pop_loop(&b, dy_lp);
            }
            nir_pop_if(&b, k3_s2_if);
         }
         nir_pop_if(&b, k3_s1_if);
      }
      nir_pop_if(&b, active);
   }
   nir_push_else(&b, v4_if);
   {
      /* Non-vec4-aligned path (e.g. RGB ch=3, kw=9): process 4 floats per X
       * thread x 4 vertical output rows (align_mul = 4 vec4 loads/stores for
       * full 4-float chunks, plus compact rem == 1 scalar streaming fast path
       * and general 1..3 scalar tail). */
      nir_def *kx_base = nir_ishl_imm(&b, gx, 2);
      nir_def *in_b = nir_iand(&b, nir_iand(&b, nir_ult(&b, kx_base, kw),
                                                nir_ult(&b, out_x, cols)),
                                   nir_ult(&b, out_y0, rows_y));
      nir_if *active = nir_push_if(&b, in_b);
      {
         nir_def *row0 = nir_iadd(&b, nir_imul(&b, out_y0, cols), out_x);
         nir_def *x = nir_iadd(&b, nir_imul(&b, out_x, nir_imul(&b, sw, ch)), kx_base);
         nir_def *y_base0 = nir_imul(&b, out_y0, sh);
         nir_def *dst_base0 = nir_iadd(&b, nir_imul(&b, row0, pw), kx_base);
         nir_def *dst_row_stride = nir_imul(&b, cols, pw);
         nir_def *has_y3 = nir_ult(&b, nir_iadd_imm(&b, out_y0, 3), rows_y);

         nir_def *x_eff = x;
         nir_def *y_eff0 = y_base0;
         if (has_border) {
            nir_def *pad_x = nir_imul(&b, bw, ch);
            x_eff = nir_bcsel(&b, unit, nir_isub(&b, x, pad_x), x);
            y_eff0 = nir_bcsel(&b, unit, nir_isub(&b, y_base0, bh), y_base0);
         }

         nir_def *kh3_4y = nir_iand(&b, nir_ieq_imm(&b, kh, 3), has_y3);
         nir_def *full4 = nir_iand(&b, nir_uge_imm(&b, kw, 4),
                                       nir_uge(&b, nir_iadd_imm(&b, kw, -4), kx_base));
         nir_if *f4_if = nir_push_if(&b, full4);
         {
            nir_def *x_all_ok = has_border
               ? nir_iand(&b, nir_uge_imm(&b, iw, 4),
                              nir_uge(&b, nir_iadd_imm(&b, iw, -4), x_eff))
               : NULL;
            nir_def *kh3_s1_4y = nir_iand(&b, kh3_4y, nir_ieq_imm(&b, sh, 1));
            nir_def *kh3_s2_4y = nir_iand(&b, kh3_4y, nir_ieq_imm(&b, sh, 2));
            if (has_border) {
               kh3_s1_4y = nir_iand(&b, kh3_s1_4y,
                  nir_iand(&b, x_all_ok, nir_iand(&b, nir_uge_imm(&b, ih, 6),
                                                      nir_uge(&b, nir_iadd_imm(&b, ih, -6), y_eff0))));
               kh3_s2_4y = nir_iand(&b, kh3_s2_4y,
                  nir_iand(&b, x_all_ok, nir_iand(&b, nir_uge_imm(&b, ih, 9),
                                                      nir_uge(&b, nir_iadd_imm(&b, ih, -9), y_eff0))));
            }
            nir_if *k3_s1_if = nir_push_if(&b, kh3_s1_4y);
            {
               nir_def *s0 = nir_iadd(&b, nir_imul(&b, y_eff0, iw), x_eff);
               emit_im2col_kh3_s1_4y(&b, src, dst, false, 4, 4, s0, iw,
                                     dst_base0, dst_row_stride, kw);
            }
            nir_push_else(&b, k3_s1_if);
            {
               nir_if *k3_s2_if = nir_push_if(&b, kh3_s2_4y);
               {
                  nir_def *s0 = nir_iadd(&b, nir_imul(&b, y_eff0, iw), x_eff);
                  emit_im2col_kh3_s2_4y(&b, src, dst, false, 4, 4, s0, iw,
                                        dst_base0, dst_row_stride, kw, zero_i);
               }
               nir_push_else(&b, k3_s2_if);
               {
                  nir_loop *dy_lp = nir_push_loop(&b);
                  dy_lp->control = nir_loop_control_dont_unroll;
                  nir_block *dy_pre = nir_cf_node_as_block(nir_cf_node_prev(&dy_lp->cf_node));
                  nir_block *dy_hdr = nir_loop_first_block(dy_lp);
                  nir_phi_instr *dy_phi = nir_phi_instr_create(b.shader);
                  nir_def_init(&dy_phi->instr, &dy_phi->def, 1, 32);
                  nir_phi_instr_add_src(dy_phi, dy_pre, zero_i);
                  nir_def *dy_v = &dy_phi->def;
                  nir_if *dy_done = nir_push_if(&b, nir_ige_imm(&b, dy_v, 4));
                  nir_jump(&b, nir_jump_break);
                  nir_pop_if(&b, dy_done);

                  nir_def *oy = nir_iadd(&b, out_y0, dy_v);
                  nir_if *oy_ok = nir_push_if(&b, nir_ult(&b, oy, rows_y));
                  {
                     nir_def *y_e = nir_iadd(&b, y_eff0, nir_imul(&b, dy_v, sh));
                     nir_def *dst_b = nir_iadd(&b, dst_base0, nir_imul(&b, dy_v, dst_row_stride));
                     nir_loop *lp = nir_push_loop(&b);
                     hk_nir_loop_binding_add(&lb_list, lp, HK_PACK_LOOP_PARAM_IM2COL_KH);
                     lp->control = nir_loop_control_dont_unroll;
                     nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&lp->cf_node));
                     nir_block *hdr = nir_loop_first_block(lp);
                     nir_phi_instr *ky_phi = nir_phi_instr_create(b.shader);
                     nir_def_init(&ky_phi->instr, &ky_phi->def, 1, 32);
                     nir_phi_instr_add_src(ky_phi, pre, zero_i);
                     nir_phi_instr *di_phi = nir_phi_instr_create(b.shader);
                     nir_def_init(&di_phi->instr, &di_phi->def, 1, 32);
                     nir_phi_instr_add_src(di_phi, pre, dst_b);

                     nir_def *ky = &ky_phi->def;
                     nir_def *dst_idx = &di_phi->def;
                     nir_if *done = nir_push_if(&b, nir_ige(&b, ky, kh));
                     nir_jump(&b, nir_jump_break);
                     nir_pop_if(&b, done);

                     nir_def *y_cur = nir_iadd(&b, y_e, ky);
                     nir_def *val = NULL;
                     if (has_border) {
                        nir_def *y_ok = nir_ult(&b, y_cur, ih);
                        nir_def *v_y = NULL;
                        nir_if *y_if = nir_push_if(&b, y_ok);
                        {
                           nir_def *row_s = nir_imul(&b, y_cur, iw);
                           nir_def *v_fast = NULL, *v_part = NULL;
                           nir_if *xa_if = nir_push_if(&b, x_all_ok);
                           {
                              nir_def *sa = address_scalar(&b, src, nir_iadd(&b, row_s, x_eff), read, false);
                              v_fast = nir_load_global(&b, 4, 32, sa, .align_mul = 4, .access = read);
                           }
                           nir_push_else(&b, xa_if);
                           {
                              nir_def *lanes[4];
                              for (unsigned c = 0; c < 4; ++c) {
                                 nir_def *xc = c ? nir_iadd_imm(&b, x_eff, c) : x_eff;
                                 nir_def *v_c = NULL;
                                 nir_if *cx = nir_push_if(&b, nir_ult(&b, xc, iw));
                                 {
                                    nir_def *sa = address_scalar(&b, src, nir_iadd(&b, row_s, xc), read, false);
                                    v_c = nir_load_global(&b, 1, 32, sa, .align_mul = 4, .access = read);
                                 }
                                 nir_push_else(&b, cx);
                                 nir_pop_if(&b, cx);
                                 lanes[c] = nir_if_phi(&b, v_c, zero_f);
                              }
                              v_part = nir_vec4(&b, lanes[0], lanes[1], lanes[2], lanes[3]);
                           }
                           nir_pop_if(&b, xa_if);
                           v_y = nir_if_phi(&b, v_fast, v_part);
                        }
                        nir_push_else(&b, y_if);
                        nir_pop_if(&b, y_if);
                        val = nir_if_phi(&b, v_y, zero_v4);
                     } else {
                        nir_def *s_idx = nir_iadd(&b, nir_imul(&b, y_cur, iw), x_eff);
                        nir_def *sa = address_scalar(&b, src, s_idx, read, false);
                        val = nir_load_global(&b, 4, 32, sa, .align_mul = 4, .access = read);
                     }

                     nir_def *da = address_scalar(&b, dst, dst_idx, ACCESS_NON_READABLE, false);
                     nir_store_global(&b, val, da, .align_mul = 4, .write_mask = 0xf,
                                      .access = ACCESS_NON_READABLE);

                     nir_block *back = nir_loop_last_block(lp);
                     nir_phi_instr_add_src(ky_phi, back, nir_iadd_imm(&b, ky, 1));
                     nir_phi_instr_add_src(di_phi, back, nir_iadd(&b, dst_idx, kw));
                     nir_instr_insert(nir_before_block(hdr), &ky_phi->instr);
                     nir_instr_insert(nir_before_block(hdr), &di_phi->instr);
                     nir_pop_loop(&b, lp);
                  }
                  nir_pop_if(&b, oy_ok);

                  nir_block *dy_back = nir_cursor_current_block(b.cursor);
                  nir_phi_instr_add_src(dy_phi, dy_back, nir_iadd_imm(&b, dy_v, 1));
                  nir_instr_insert(nir_before_block(dy_hdr), &dy_phi->instr);
                  nir_pop_loop(&b, dy_lp);
               }
               nir_pop_if(&b, k3_s2_if);
            }
            nir_pop_if(&b, k3_s1_if);
         }
         nir_push_else(&b, f4_if);
         {
            /* Trailing 1..3 scalar floats when kw % 4 != 0. Specialize rem == 1
             * (e.g. RGB kw=9, gx=2 sharing a subgroup with full4 gx=0,1) for
             * kh == 3, sh in {1,2} to avoid nested-loop subgroup divergence. */
            nir_def *rem1 = nir_iand(&b, kh3_4y,
                                     nir_ieq_imm(&b, nir_isub(&b, kw, kx_base), 1));
            nir_def *r1_s2 = nir_iand(&b, rem1, nir_ieq_imm(&b, sh, 2));
            nir_def *r1_s1 = nir_iand(&b, rem1, nir_ieq_imm(&b, sh, 1));
            if (has_border) {
               nir_def *x1_ok = nir_ult(&b, x_eff, iw);
               r1_s2 = nir_iand(&b, r1_s2,
                  nir_iand(&b, x1_ok, nir_iand(&b, nir_uge_imm(&b, ih, 9),
                                                   nir_uge(&b, nir_iadd_imm(&b, ih, -9), y_eff0))));
               r1_s1 = nir_iand(&b, r1_s1,
                  nir_iand(&b, x1_ok, nir_iand(&b, nir_uge_imm(&b, ih, 6),
                                                   nir_uge(&b, nir_iadd_imm(&b, ih, -6), y_eff0))));
            }
            nir_if *r1_s2_if = nir_push_if(&b, r1_s2);
            {
               nir_def *s0 = nir_iadd(&b, nir_imul(&b, y_eff0, iw), x_eff);
               emit_im2col_kh3_s2_4y(&b, src, dst, false, 1, 4, s0, iw,
                                     dst_base0, dst_row_stride, kw, zero_i);
            }
            nir_push_else(&b, r1_s2_if);
            {
               nir_if *r1_s1_if = nir_push_if(&b, r1_s1);
               {
                  nir_def *s0 = nir_iadd(&b, nir_imul(&b, y_eff0, iw), x_eff);
                  emit_im2col_kh3_s1_4y(&b, src, dst, false, 1, 4, s0, iw,
                                        dst_base0, dst_row_stride, kw);
               }
               nir_push_else(&b, r1_s1_if);
               {
                  nir_loop *dy_lp = nir_push_loop(&b);
                  dy_lp->control = nir_loop_control_dont_unroll;
                  nir_block *dy_pre = nir_cf_node_as_block(nir_cf_node_prev(&dy_lp->cf_node));
                  nir_block *dy_hdr = nir_loop_first_block(dy_lp);
                  nir_phi_instr *dy_phi = nir_phi_instr_create(b.shader);
                  nir_def_init(&dy_phi->instr, &dy_phi->def, 1, 32);
                  nir_phi_instr_add_src(dy_phi, dy_pre, zero_i);
                  nir_def *dy_v = &dy_phi->def;
                  nir_if *dy_done = nir_push_if(&b, nir_ige_imm(&b, dy_v, 4));
                  nir_jump(&b, nir_jump_break);
                  nir_pop_if(&b, dy_done);

                  nir_def *oy = nir_iadd(&b, out_y0, dy_v);
                  nir_if *oy_ok = nir_push_if(&b, nir_ult(&b, oy, rows_y));
                  {
                     nir_def *y_e = nir_iadd(&b, y_eff0, nir_imul(&b, dy_v, sh));
                     nir_def *dst_b = nir_iadd(&b, dst_base0, nir_imul(&b, dy_v, dst_row_stride));
                     nir_loop *lp = nir_push_loop(&b);
                     hk_nir_loop_binding_add(&lb_list, lp, HK_PACK_LOOP_PARAM_IM2COL_KH);
                     lp->control = nir_loop_control_dont_unroll;
                     nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&lp->cf_node));
                     nir_block *hdr = nir_loop_first_block(lp);
                     nir_phi_instr *ky_phi = nir_phi_instr_create(b.shader);
                     nir_def_init(&ky_phi->instr, &ky_phi->def, 1, 32);
                     nir_phi_instr_add_src(ky_phi, pre, zero_i);
                     nir_phi_instr *di_phi = nir_phi_instr_create(b.shader);
                     nir_def_init(&di_phi->instr, &di_phi->def, 1, 32);
                     nir_phi_instr_add_src(di_phi, pre, dst_b);

                     nir_def *ky = &ky_phi->def;
                     nir_def *dst_idx = &di_phi->def;
                     nir_if *done = nir_push_if(&b, nir_ige(&b, ky, kh));
                     nir_jump(&b, nir_jump_break);
                     nir_pop_if(&b, done);

                     nir_def *y_cur = nir_iadd(&b, y_e, ky);
                     nir_def *row_s = nir_imul(&b, y_cur, iw);
                     nir_def *y_ok = has_border ? nir_ult(&b, y_cur, ih) : NULL;
                     for (unsigned c = 0; c < 3; ++c) {
                        nir_if *c_if = c ? nir_push_if(&b, nir_ult(&b, nir_iadd_imm(&b, kx_base, c), kw)) : NULL;
                        {
                           nir_def *xc = c ? nir_iadd_imm(&b, x_eff, c) : x_eff;
                           nir_def *val = NULL;
                           if (has_border) {
                              nir_def *v_ld = NULL;
                              nir_if *ok_if = nir_push_if(&b, nir_iand(&b, y_ok, nir_ult(&b, xc, iw)));
                              {
                                 nir_def *sa = address_scalar(&b, src, nir_iadd(&b, row_s, xc), read, false);
                                 v_ld = nir_load_global(&b, 1, 32, sa, .align_mul = 4, .access = read);
                              }
                              nir_push_else(&b, ok_if);
                              nir_pop_if(&b, ok_if);
                              val = nir_if_phi(&b, v_ld, zero_f);
                           } else {
                              nir_def *sa = address_scalar(&b, src, nir_iadd(&b, row_s, xc), read, false);
                              val = nir_load_global(&b, 1, 32, sa, .align_mul = 4, .access = read);
                           }
                           nir_def *dc = c ? nir_iadd_imm(&b, dst_idx, c) : dst_idx;
                           nir_def *da = address_scalar(&b, dst, dc, ACCESS_NON_READABLE, false);
                           nir_store_global(&b, val, da, .align_mul = 4, .write_mask = 1,
                                            .access = ACCESS_NON_READABLE);
                        }
                        if (c_if)
                           nir_pop_if(&b, c_if);
                     }

                     nir_block *back = nir_loop_last_block(lp);
                     nir_phi_instr_add_src(ky_phi, back, nir_iadd_imm(&b, ky, 1));
                     nir_phi_instr_add_src(di_phi, back, nir_iadd(&b, dst_idx, kw));
                     nir_instr_insert(nir_before_block(hdr), &ky_phi->instr);
                     nir_instr_insert(nir_before_block(hdr), &di_phi->instr);
                     nir_pop_loop(&b, lp);
                  }
                  nir_pop_if(&b, oy_ok);

                  nir_block *dy_back = nir_cursor_current_block(b.cursor);
                  nir_phi_instr_add_src(dy_phi, dy_back, nir_iadd_imm(&b, dy_v, 1));
                  nir_instr_insert(nir_before_block(dy_hdr), &dy_phi->instr);
                  nir_pop_loop(&b, dy_lp);
               }
               nir_pop_if(&b, r1_s1_if);
            }
            nir_pop_if(&b, r1_s2_if);
         }
         nir_pop_if(&b, f4_if);
      }
      nir_pop_if(&b, active);
   }
   nir_pop_if(&b, v4_if);

   nir_opt_dce(nir);
   nir_shader_gather_info(nir, impl);
   if (loop_bound_out) {
      if (lb_list.overflow)
         memset(loop_bound_out, 0, sizeof(*loop_bound_out));
      else
         hk_nir_compute_pack_loop_bound(nir, lb_list.entries, lb_list.count,
                                        loop_bound_out);
   }
   return nir;
}

nir_shader *
hk_nir_im2col_vec4(const nir_shader *orig)
{
   return hk_nir_im2col_vec4_with_bound(orig, NULL);
}

bool
hk_nir_copy_identity4_may_match(const nir_shader *orig)
{
   if (!orig || orig->info.stage != MESA_SHADER_COMPUTE ||
       orig->info.workgroup_size_variable || orig->info.shared_size != 0 ||
       !hk_identity_copy4_supported_local(orig->info.workgroup_size) ||
       (orig->info.num_ssbos != 2 && orig->info.num_ssbos != 3) ||
       orig->info.num_images > 0 || orig->info.num_textures > 0 ||
       orig->info.num_ubos > 0)
      return false;

   nir_foreach_variable_in_shader(var, orig) {
      if (var->data.mode &
          (nir_var_mem_shared | nir_var_image | nir_var_uniform))
         return false;
   }

   if (!exec_list_is_singular(&orig->functions))
      return false;

   const nir_function_impl *impl = nir_shader_get_entrypoint(orig);
   if (!impl || !acyclic(&impl->body, 16))
      return false;

   return true;
}

static nir_shader *
hk_nir_copy_identity4_build(const nir_shader *orig,
                            struct hk_identity_copy4 *info, bool dual_copy)
{
   if (!hk_nir_copy_identity4_may_match(orig))
      return NULL;

   nir_shader *nir = nir_shader_clone(NULL, orig);
   nir_opt_constant_folding(nir);

   const uint16_t orig_local[3] = {
      nir->info.workgroup_size[0],
      nir->info.workgroup_size[1],
      nir->info.workgroup_size[2],
   };

   struct copy_buffer srcs[2] = {{0}};
   struct copy_buffer dst = {0};
   struct im2col_params im2col_p = {0};
   enum hk_identity_copy4_kind kind = HK_IDENTITY_COPY4_IM2COL_1X1;
   uint16_t count_off = 0, src_off = 0, dst_off = 0, push_bytes = 0;
   unsigned fadd_fp_ctrl = 0;
   bool im2col_vec4_ok = false;

   if (match_im2col(nir, &srcs[0], &dst, &im2col_p, &im2col_vec4_ok)) {
      kind = HK_IDENTITY_COPY4_IM2COL_1X1;
      push_bytes = im2col_p.push_bytes;
   } else {
      struct pad4d_match pad4 = {0};
      if (match_pad4d(nir, &pad4)) {
         if (dual_copy) {
            ralloc_free(nir);
            return NULL;
         }
         return build_pad4d_vec4(nir, &pad4, orig_local, info);
      }
      if (!match_linear_1d(nir, &kind, srcs, &dst, &count_off, &src_off,
                           &dst_off, &push_bytes, &fadd_fp_ctrl)) {
         ralloc_free(nir);
         return NULL;
      }
   }

   if (dual_copy &&
       (kind != HK_IDENTITY_COPY4_LINEAR_FADD ||
        hk_dual_write_push_offset(push_bytes) +
           sizeof(struct hk_dual_copy_push) > UINT16_MAX)) {
      ralloc_free(nir);
      return NULL;
   }

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list old;
   nir_cf_extract(&old, nir_before_impl(impl), nir_after_impl(impl));
   nir_cf_delete(&old);
   nir_builder b = nir_builder_at(nir_after_block(nir_start_block(impl)));

   /* Flatten to a 1D packed vec4 workgroup, transferring 4 coalesced vec4s
    * (64B) per lane to expose memory-level parallelism and cut WG count 4x. */
   const uint16_t wg_x = 128;
   const uint32_t wg_chunk = (uint32_t)wg_x * 4;
   nir->info.workgroup_size[0] = wg_x;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;

   nir_def *wg_id = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *wg_base = nir_imul_imm(&b, wg_id, wg_chunk);
   nir_def *base = nir_iadd(&b, wg_base, lid);
   nir_def *total4 = NULL;
   if (kind == HK_IDENTITY_COPY4_IM2COL_1X1) {
      nir_def *width4 =
         nir_ushr_imm(&b, push_at(&b, im2col_p.off[IM2COL_PATCHES_WIDTH], push_bytes), 2);
      total4 = nir_imul(&b, push_at(&b, im2col_p.off[IM2COL_PATCHES_HEIGHT], push_bytes), width4);
   } else {
      total4 = nir_ushr_imm(&b, push_at(&b, count_off, push_bytes), 2);
   }

   const enum gl_access_qualifier read = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   nir_def *src_off4 = NULL, *dst_off4 = NULL;
   if (kind == HK_IDENTITY_COPY4_LINEAR_OFFSET) {
      src_off4 = nir_ushr_imm(&b, push_at(&b, src_off, push_bytes), 2);
      dst_off4 = nir_ushr_imm(&b, push_at(&b, dst_off, push_bytes), 2);
   }

   nir_def *dst2_base = NULL;
   nir_def *write_c = NULL;
   if (dual_copy) {
      const uint32_t dual_off = hk_dual_write_push_offset(push_bytes);
      const uint16_t dual_end =
         (uint16_t)(dual_off + sizeof(struct hk_dual_copy_push));
      nir_def *lo = push_at(
         &b, (uint16_t)(dual_off + offsetof(struct hk_dual_copy_push, dst2_va_lo)),
         dual_end);
      nir_def *hi = push_at(
         &b, (uint16_t)(dual_off + offsetof(struct hk_dual_copy_push, dst2_va_hi)),
         dual_end);
      nir_def *flags = push_at(
         &b, (uint16_t)(dual_off + offsetof(struct hk_dual_copy_push, flags)),
         dual_end);
      dst2_base = nir_pack_64_2x32_split(&b, lo, hi);
      write_c = nir_ieq_imm(&b, nir_iand_imm(&b, flags, HK_DUAL_WRITE_FLAG_SKIP_C), 0);
   }

   /* Workgroup-uniform interior check without uint32 underflow. */
   nir_def *full_wg = nir_iand(
      &b, nir_uge(&b, total4, nir_imm_int(&b, wg_chunk)),
      nir_uge(&b, nir_iadd_imm(&b, total4, -(int32_t)wg_chunk), wg_base));
   nir_if *fast = nir_push_if(&b, full_wg);
   {
      nir_def *vals[4];
      nir_def *idxs[4];
      nir_def *dst_idxs[4];
      for (unsigned k = 0; k < 4; ++k) {
         nir_def *idx = k ? nir_iadd_imm(&b, base, (int32_t)(k * wg_x)) : base;
         idxs[k] = idx;
         nir_def *s_idx = src_off4 ? nir_iadd(&b, idx, src_off4) : idx;
         dst_idxs[k] = dst_off4 ? nir_iadd(&b, idx, dst_off4) : idx;
         nir_def *in0 = address(&b, srcs[0], s_idx, read);
         nir_def *p0 = nir_load_global(&b, 4, 32, in0, .align_mul = 16, .access = read);
         if (kind == HK_IDENTITY_COPY4_LINEAR_FADD) {
            nir_def *in1 = address(&b, srcs[1], s_idx, read);
            nir_def *p1 = nir_load_global(&b, 4, 32, in1, .align_mul = 16, .access = read);
            b.fp_math_ctrl = fadd_fp_ctrl;
            p0 = nir_fadd(&b, p0, p1);
            b.fp_math_ctrl = 0;
         }
         vals[k] = p0;
      }
      nir_if *wc_if = write_c ? nir_push_if(&b, write_c) : NULL;
      {
         for (unsigned k = 0; k < 4; ++k) {
            nir_def *out = address(&b, dst, dst_idxs[k], ACCESS_NON_READABLE);
            nir_store_global(&b, vals[k], out, .align_mul = 16, .write_mask = 0xf,
                             .access = ACCESS_NON_READABLE);
         }
      }
      if (wc_if)
         nir_pop_if(&b, wc_if);
      if (dst2_base) {
         for (unsigned k = 0; k < 4; ++k) {
            nir_def *out2 =
               nir_iadd(&b, dst2_base, nir_ishl_imm(&b, nir_u2u64(&b, idxs[k]), 4));
            nir_store_global(&b, vals[k], out2, .align_mul = 16,
                             .write_mask = 0xf, .access = ACCESS_NON_READABLE);
         }
      }
   }
   nir_push_else(&b, fast);
   {
      for (unsigned k = 0; k < 4; ++k) {
         nir_def *idx = k ? nir_iadd_imm(&b, base, (int32_t)(k * wg_x)) : base;
         nir_if *ok = nir_push_if(&b, nir_ult(&b, idx, total4));
         {
            nir_def *s_idx = src_off4 ? nir_iadd(&b, idx, src_off4) : idx;
            nir_def *d_idx = dst_off4 ? nir_iadd(&b, idx, dst_off4) : idx;
            nir_def *in0 = address(&b, srcs[0], s_idx, read);
            nir_def *p0 = nir_load_global(&b, 4, 32, in0, .align_mul = 16, .access = read);
            if (kind == HK_IDENTITY_COPY4_LINEAR_FADD) {
               nir_def *in1 = address(&b, srcs[1], s_idx, read);
               nir_def *p1 = nir_load_global(&b, 4, 32, in1, .align_mul = 16, .access = read);
               b.fp_math_ctrl = fadd_fp_ctrl;
               p0 = nir_fadd(&b, p0, p1);
               b.fp_math_ctrl = 0;
            }
            nir_if *wc_if = write_c ? nir_push_if(&b, write_c) : NULL;
            {
               nir_def *out = address(&b, dst, d_idx, ACCESS_NON_READABLE);
               nir_store_global(&b, p0, out, .align_mul = 16, .write_mask = 0xf,
                                .access = ACCESS_NON_READABLE);
            }
            if (wc_if)
               nir_pop_if(&b, wc_if);
            if (dst2_base) {
               nir_def *out2 =
                  nir_iadd(&b, dst2_base, nir_ishl_imm(&b, nir_u2u64(&b, idx), 4));
               nir_store_global(&b, p0, out2, .align_mul = 16,
                                .write_mask = 0xf, .access = ACCESS_NON_READABLE);
            }
         }
         nir_pop_if(&b, ok);
      }
   }
   nir_pop_if(&b, fast);

   nir_opt_dce(nir);
   nir_shader_gather_info(nir, impl);
   memset(info, 0, sizeof(*info));
   info->kind = (uint8_t)kind;
   info->src_count = (kind == HK_IDENTITY_COPY4_LINEAR_FADD) ? 2 : 1;
   info->src_set = srcs[0].set;
   info->src1_set = srcs[1].set;
   info->dst_set = dst.set;
   info->src_binding = srcs[0].binding;
   info->src1_binding = srcs[1].binding;
   info->dst_binding = dst.binding;
   if (kind == HK_IDENTITY_COPY4_IM2COL_1X1) {
      info->border_width_offset = im2col_p.off[IM2COL_BORDER_WIDTH];
      info->border_height_offset = im2col_p.off[IM2COL_BORDER_HEIGHT];
      info->input_width_offset = im2col_p.off[IM2COL_INPUT_WIDTH];
      info->input_height_offset = im2col_p.off[IM2COL_INPUT_HEIGHT];
      info->kernel_width_offset = im2col_p.off[IM2COL_KERNEL_WIDTH];
      info->patches_width_offset = im2col_p.off[IM2COL_PATCHES_WIDTH];
      info->patches_height_offset = im2col_p.off[IM2COL_PATCHES_HEIGHT];
      info->stride_width_offset = im2col_p.off[IM2COL_STRIDE_WIDTH];
      info->stride_height_offset = im2col_p.off[IM2COL_STRIDE_HEIGHT];
      info->channels_offset = im2col_p.off[IM2COL_CHANNELS];
      info->im2col_vec4_ok = im2col_vec4_ok;
   }
   info->count_offset = count_off;
   info->src_offset_offset = src_off;
   info->dst_offset_offset = dst_off;
   info->push_bytes = push_bytes;
   if (kind == HK_IDENTITY_COPY4_LINEAR_FADD) {
      info->fadd_fp_math_ctrl = fadd_fp_ctrl;
      info->fadd_float_controls_execution_mode =
         orig->info.float_controls_execution_mode;
      info->fadd_meta_valid = true;
   }
   for (unsigned d = 0; d < 3; ++d)
      info->local[d] = orig_local[d];
   return nir;
}

nir_shader *
hk_nir_copy_identity4(const nir_shader *orig, struct hk_identity_copy4 *info)
{
   return hk_nir_copy_identity4_build(orig, info, false);
}

nir_shader *
hk_nir_copy_identity4_fadd_dual(const nir_shader *orig,
                                struct hk_identity_copy4 *info)
{
   return hk_nir_copy_identity4_build(orig, info, true);
}

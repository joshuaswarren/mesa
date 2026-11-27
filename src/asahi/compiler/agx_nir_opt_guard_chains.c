/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Optimize chains of guarded conditional blocks in AGX3 shaders:
 *
 * 1. Outer-guard fusion:
 *
 *       if (c1) {
 *          prefix;        // pure ALU and fault-safe / redirectable global loads
 *          if (c2) {
 *             body;
 *          }
 *          suffix;        // phis and identity-preserving reductions
 *       }
 *
 *    becomes
 *
 *       prefix_safe;
 *       if (c1 & c2) {
 *          body;
 *          suffix_then;
 *       }
 *
 *    Global loads in `prefix` keep their original address when already marked
 *    ACCESS_CAN_SPECULATE or when `c1` is `a != b` and a dominating
 *    unconditional global load reads the identical address expression with `b`
 *    substituted for `a` (so `a == b` reads an address already proven mapped
 *    and aligned). Otherwise their address is redirected to
 *    AGX_ZERO_PAGE_ADDRESS when `!c1`, as in agx_nir_opt_flatten_loads().
 *
 * 2. Independent prefix hoisting across pure `if`s:
 *
 *    In an unrolled loop body such as
 *
 *       p0 = load(a0); c0 = test(p0); if (c0) { body0 }
 *       p1 = load(a1); c1 = test(p1); if (c1) { body1 }
 *       ...
 *
 *    every control-flow boundary on AGX3 forces a wait for all loads in flight,
 *    serializing `p0`, `p1`, ... one after another. When `if (c0)` has no
 *    side effects or early exits, `p1` and `c1` post-dominate `if (c0)` and do
 *    not depend on its phis, so hoisting them before `if (c0)` lets all loads
 *    issue in flight together with zero extra speculation.
 *
 * 3. Coalescing adjacent expensive one-sided divergent `if`s:
 *
 *       if (c0) { body0 }
 *       if (c1) { body1 }
 *       ...
 *
 *    becomes
 *
 *       if (c0 | c1 | ...) {
 *          if (c0) { body0 }
 *          if (c1) { body1 }
 *          ...
 *       }
 *
 *    When all lanes in a SIMD-group fail all `k` tests, one combined
 *    `if_icmp + jmp_exec_none + pop_exec` skips all `k` bodies at once.
 */

#include "asahi/lib/agx_abi.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "util/set.h"
#include "util/u_dynarray.h"
#include "agx_nir.h"

#define MAX_PREFIX_INSTRS        24u
#define MAX_PREFIX_LOADS         4u
#define MAX_GROUP_IFS            8u
#define MAX_LIVE_HOISTED_DWORDS  48u
#define MIN_COALESCE_BODY_INSTRS 12u

static unsigned
load_bytes(const nir_intrinsic_instr *intr)
{
   return intr->def.num_components * DIV_ROUND_UP(intr->def.bit_size, 8);
}

static uint64_t
aligned_from(const nir_intrinsic_instr *intr, uint64_t from)
{
   const uint64_t mul = MAX2(nir_intrinsic_align_mul(intr), 1);
   const uint64_t offset = nir_intrinsic_align_offset(intr);
   return from + ((offset - from) & (mul - 1));
}

static bool
is_redirectable_global_load(const nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global:
   case nir_intrinsic_load_global_constant:
      if (intr->src[0].ssa->bit_size != 64 ||
          intr->src[0].ssa->num_components != 1)
         return false;
      return !(nir_intrinsic_access(intr) & (ACCESS_VOLATILE | ACCESS_ATOMIC));
   default:
      return false;
   }
}

static bool
fits_zero_page(const nir_intrinsic_instr *intr, uint64_t offset)
{
   return offset <= AGX_ZERO_PAGE_SIZE - load_bytes(intr) &&
          offset == aligned_from(intr, offset);
}

static bool
is_scalar_offset(const nir_def *def)
{
   if (def->num_components != 1 || def->bit_size != 64)
      return false;
   if (nir_def_is_const(def))
      return true;
   const nir_alu_instr *alu = nir_def_as_alu_or_null(def);
   if (!alu)
      return false;

   switch (alu->op) {
   case nir_op_u2u64:
   case nir_op_i2i64:
      return true;
   case nir_op_ishl:
   case nir_op_imul:
   case nir_op_amul:
      return nir_src_is_const(alu->src[1].src) &&
             is_scalar_offset(alu->src[0].src.ssa);
   case nir_op_iadd:
      return is_scalar_offset(alu->src[0].src.ssa) &&
             is_scalar_offset(alu->src[1].src.ssa);
   default:
      return false;
   }
}

static nir_def *
offset_unless(nir_builder *b, nir_def *def, nir_def *taken)
{
   if (nir_def_is_const(def))
      return def;

   nir_alu_instr *alu = nir_def_as_alu(def);
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
   switch (alu->op) {
   case nir_op_u2u64:
   case nir_op_i2i64:
      return nir_build_alu1(
         b, alu->op, nir_bcsel(b, taken, x, nir_imm_zero(b, 1, x->bit_size)));
   case nir_op_ishl:
   case nir_op_imul:
   case nir_op_amul:
      return nir_build_alu2(b, alu->op, offset_unless(b, x, taken),
                            nir_ssa_for_alu_src(b, alu, 1));
   default:
      return nir_iadd(b, offset_unless(b, x, taken),
                      offset_unless(b, nir_ssa_for_alu_src(b, alu, 1), taken));
   }
}

static uint64_t
offset_residual(const nir_def *def)
{
   if (nir_def_is_const(def))
      return nir_scalar_as_uint(nir_get_scalar((nir_def *)def, 0));

   const nir_alu_instr *alu = nir_def_as_alu(def);
   switch (alu->op) {
   case nir_op_u2u64:
   case nir_op_i2i64:
      return 0;
   case nir_op_ishl:
      return offset_residual(alu->src[0].src.ssa)
             << (nir_alu_src_as_uint(alu->src[1]) & 63);
   case nir_op_imul:
   case nir_op_amul:
      return offset_residual(alu->src[0].src.ssa) *
             nir_alu_src_as_uint(alu->src[1]);
   default:
      return offset_residual(alu->src[0].src.ssa) +
             offset_residual(alu->src[1].src.ssa);
   }
}

static int
offset_source(const nir_alu_instr *alu)
{
   if (alu->op != nir_op_iadd || alu->def.num_components != 1 ||
       alu->src[0].src.ssa->num_components != 1 ||
       alu->src[1].src.ssa->num_components != 1)
      return -1;

   for (unsigned i = 0; i < 2; ++i) {
      if (is_scalar_offset(alu->src[i].src.ssa))
         return i;
   }
   return -1;
}

static uint64_t
address_residual(const nir_def *addr)
{
   const nir_alu_instr *alu = nir_def_as_alu_or_null(addr);
   int i = alu ? offset_source(alu) : -1;
   if (i < 0)
      return 0;

   return address_residual(alu->src[1 - i].src.ssa) +
          offset_residual(alu->src[i].src.ssa);
}

static nir_def *
address_unless(nir_builder *b, nir_def *addr, nir_def *taken)
{
   nir_alu_instr *alu = nir_def_as_alu_or_null(addr);
   int i = alu ? offset_source(alu) : -1;
   if (i < 0)
      return nir_bcsel(b, taken, addr, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));

   return nir_iadd(b, address_unless(b, alu->src[1 - i].src.ssa, taken),
                   offset_unless(b, alu->src[i].src.ssa, taken));
}

static bool
list_is_single_empty_block(const struct exec_list *list)
{
   if (!exec_list_is_singular(list))
      return false;
   const nir_cf_node *node =
      exec_node_data(nir_cf_node, exec_list_get_head_const(list), node);
   if (node->type != nir_cf_node_block)
      return false;
   const nir_block *blk = nir_cf_node_as_block(node);
   return exec_list_is_empty(&blk->instr_list);
}

static bool
block_is_phis_only(const nir_block *block)
{
   nir_foreach_instr(instr, block) {
      if (instr->type != nir_instr_type_phi)
         return false;
   }
   return true;
}

static bool
instr_is_lane_dependent(const nir_intrinsic_instr *intr)
{
   if (nir_intrinsic_infos[intr->intrinsic].flags &
       (NIR_INTRINSIC_SUBGROUP | NIR_INTRINSIC_QUADGROUP))
      return true;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_active_subgroup_invocation_agx:
   case nir_intrinsic_load_active_subgroup_count_agx:
   case nir_intrinsic_quad_ballot_agx:
   case nir_intrinsic_is_helper_invocation:
   case nir_intrinsic_load_helper_invocation:
   case nir_intrinsic_demote:
   case nir_intrinsic_demote_if:
   case nir_intrinsic_demote_samples:
   case nir_intrinsic_terminate:
   case nir_intrinsic_terminate_if:
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy_coarse:
   case nir_intrinsic_barrier:
      return true;
   default:
      return false;
   }
}

/* Whether a CF node has no side effects, early exits, synchronization, or
 * lane-mask-dependent operations, so code after it post-dominates it with the
 * same memory state and active execution mask. */
static bool
cf_node_is_pure_read_only(nir_cf_node *node)
{
   nir_foreach_block_in_cf_node(block, node) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_alu:
         case nir_instr_type_load_const:
         case nir_instr_type_undef:
         case nir_instr_type_phi:
            break;
         case nir_instr_type_tex:
            if (nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr)))
               return false;
            break;
         case nir_instr_type_intrinsic: {
            const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (!(nir_intrinsic_infos[intr->intrinsic].flags &
                  NIR_INTRINSIC_CAN_ELIMINATE))
               return false;
            if (instr_is_lane_dependent(intr))
               return false;
            if (nir_intrinsic_has_access(intr) &&
                (nir_intrinsic_access(intr) &
                 (ACCESS_VOLATILE | ACCESS_ATOMIC)))
               return false;
            break;
         }
         default:
            return false;
         }
      }
   }
   return true;
}

/* Match `cond` as `a != b` (or `!(a == b)`). */
static bool
match_ine(nir_def *cond, nir_def **a, nir_def **b)
{
   nir_alu_instr *alu = nir_def_as_alu_or_null(cond);
   if (!alu)
      return false;

   if (alu->op == nir_op_inot && alu->def.num_components == 1) {
      nir_alu_instr *inner = nir_def_as_alu_or_null(alu->src[0].src.ssa);
      if (!inner || inner->op != nir_op_ieq ||
          inner->src[0].src.ssa->num_components != 1 ||
          inner->src[1].src.ssa->num_components != 1)
         return false;
      *a = inner->src[0].src.ssa;
      *b = inner->src[1].src.ssa;
      return true;
   }

   if (alu->op == nir_op_ine && alu->def.num_components == 1 &&
       alu->src[0].src.ssa->num_components == 1 &&
       alu->src[1].src.ssa->num_components == 1) {
      *a = alu->src[0].src.ssa;
      *b = alu->src[1].src.ssa;
      return true;
   }

   return false;
}

static bool expr_equal_with_subst(const nir_def *u, const nir_def *v,
                                  const nir_def *from, const nir_def *to,
                                  unsigned depth);

/* Whether source `i` of `au` and source `j` of `av` read the same channels of
 * equal expressions. Every channel the ALU reads is compared, which can be more
 * than the destination has (pack_64_2x32 reads two channels for one). */
static bool
alu_src_equal_with_subst(const nir_alu_instr *au, unsigned i,
                         const nir_alu_instr *av, unsigned j,
                         const nir_def *from, const nir_def *to, unsigned depth)
{
   const unsigned comps = nir_ssa_alu_instr_src_components(au, i);
   if (comps != nir_ssa_alu_instr_src_components(av, j))
      return false;

   for (unsigned c = 0; c < comps; ++c) {
      if (au->src[i].swizzle[c] != av->src[j].swizzle[c])
         return false;
   }

   return expr_equal_with_subst(au->src[i].src.ssa, av->src[j].src.ssa, from,
                                to, depth);
}

static bool
expr_equal_with_subst(const nir_def *u, const nir_def *v, const nir_def *from,
                      const nir_def *to, unsigned depth)
{
   if (u == from && v == to)
      return true;
   if (u == v)
      return true;
   if (depth == 0 || u->bit_size != v->bit_size ||
       u->num_components != v->num_components)
      return false;

   if (nir_def_is_const(u) && nir_def_is_const(v)) {
      for (unsigned c = 0; c < u->num_components; ++c) {
         if (nir_scalar_as_uint(nir_get_scalar((nir_def *)u, c)) !=
             nir_scalar_as_uint(nir_get_scalar((nir_def *)v, c)))
            return false;
      }
      return true;
   }

   const nir_alu_instr *au = nir_def_as_alu_or_null(u);
   const nir_alu_instr *av = nir_def_as_alu_or_null(v);
   if (!au || !av || au->op != av->op)
      return false;

   const unsigned n = nir_op_infos[au->op].num_inputs;
   bool direct = true;
   for (unsigned i = 0; i < n; ++i) {
      if (!alu_src_equal_with_subst(au, i, av, i, from, to, depth - 1)) {
         direct = false;
         break;
      }
   }
   if (direct)
      return true;

   return n == 2 &&
          (nir_op_infos[au->op].algebraic_properties &
           NIR_OP_IS_2SRC_COMMUTATIVE) &&
          alu_src_equal_with_subst(au, 0, av, 1, from, to, depth - 1) &&
          alu_src_equal_with_subst(au, 1, av, 0, from, to, depth - 1);
}

/* True if `intr` (guarded by `cond`) is proven mapped and aligned even when
 * `!cond`, because `cond` is `a != b` and a dominating unconditional global
 * load already dereferences the exact same address expression with `b` in
 * place of `a`. */
static bool
dominating_load_proves_safe(nir_block *prev_block, nir_def *cond,
                            const nir_intrinsic_instr *intr)
{
   nir_def *a = NULL, *b = NULL;
   if (!match_ine(cond, &a, &b))
      return false;

   const unsigned need_bytes = load_bytes(intr);
   const unsigned need_mul = MAX2(nir_intrinsic_align_mul(intr), 1);
   const unsigned need_off = nir_intrinsic_align_offset(intr);
   const nir_def *addr = intr->src[0].ssa;

   for (nir_block *blk = prev_block; blk; blk = blk->imm_dom) {
      nir_foreach_instr(i, blk) {
         if (i->type != nir_instr_type_intrinsic)
            continue;
         const nir_intrinsic_instr *dom = nir_instr_as_intrinsic(i);
         if (dom->intrinsic != nir_intrinsic_load_global &&
             dom->intrinsic != nir_intrinsic_load_global_constant)
            continue;
         if (load_bytes(dom) < need_bytes)
            continue;
         const unsigned dom_mul = MAX2(nir_intrinsic_align_mul(dom), 1);
         if (dom_mul < need_mul ||
             ((nir_intrinsic_align_offset(dom) - need_off) & (need_mul - 1)))
            continue;
         const nir_def *dom_addr = dom->src[0].ssa;
         if (expr_equal_with_subst(addr, dom_addr, a, b, 10) ||
             expr_equal_with_subst(addr, dom_addr, b, a, 10))
            return true;
      }
   }

   return false;
}

static bool
phi_src_for_pred(const nir_phi_instr *phi, const nir_block *pred, nir_def **out)
{
   nir_foreach_phi_src(src, phi) {
      if (src->pred == pred) {
         *out = src->src.ssa;
         return true;
      }
   }
   return false;
}

/* Check whether `alu` in `suffix_block` is `op(else_val, inner_phi)` (or
 * commutative) where the scalar `inner_phi` evaluates to the identity element
 * of `op` on `inner_else`, so `alu` evaluates to `else_val` whenever `inner_if`
 * is not taken. If so, returns the `inner_then` input of `inner_phi` in
 * `*then_rhs`, which has the same single component as `alu`. */
static bool
match_suffix_identity_alu(const nir_alu_instr *alu, const nir_block *suffix_block,
                          const nir_block *inner_then, const nir_block *inner_else,
                          const nir_def *else_val, nir_def **then_rhs,
                          unsigned *phi_src_idx)
{
   if (alu->instr.block != suffix_block || alu->def.num_components != 1)
      return false;

   uint64_t identity = 0;
   switch (alu->op) {
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_iadd:
      identity = 0;
      break;
   case nir_op_iand:
      identity = BITFIELD64_MASK(alu->def.bit_size);
      break;
   default:
      return false;
   }

   for (unsigned side = 0; side < 2; ++side) {
      if (alu->src[1 - side].src.ssa != else_val ||
          alu->src[1 - side].swizzle[0] != 0 ||
          alu->src[side].swizzle[0] != 0)
         continue;
      /* The rewrite rebuilds `alu` from the phi's `inner_then` input, so a
       * vector phi read through a swizzle would yield a vector result. */
      nir_def *other = alu->src[side].src.ssa;
      if (other->num_components != 1 ||
          nir_def_instr(other)->type != nir_instr_type_phi ||
          nir_def_block(other) != suffix_block)
         continue;
      const nir_phi_instr *iphi = nir_def_as_phi(other);
      nir_def *ithen = NULL, *ielse = NULL;
      if (!phi_src_for_pred(iphi, inner_then, &ithen) ||
          !phi_src_for_pred(iphi, inner_else, &ielse))
         continue;
      if (!nir_def_is_const(ielse))
         continue;
      if (nir_scalar_as_uint(nir_get_scalar(ielse, 0)) != identity)
         continue;
      *then_rhs = ithen;
      *phi_src_idx = side;
      return true;
   }
   return false;
}

static bool
try_fuse_nested_guard(nir_builder *b, nir_if *outer)
{
   if (outer->control == nir_selection_control_dont_flatten ||
       !list_is_single_empty_block(&outer->else_list))
      return false;

   /* Expect then_list == [prefix_block, inner_if, suffix_block]. */
   nir_cf_node *n0 = nir_cf_node_next(&nir_if_first_then_block(outer)->cf_node);
   if (!n0 || n0->type != nir_cf_node_if)
      return false;
   nir_if *inner = nir_cf_node_as_if(n0);
   nir_cf_node *n1 = nir_cf_node_next(&inner->cf_node);
   if (!n1 || n1->type != nir_cf_node_block || nir_cf_node_next(n1) != NULL)
      return false;
   if (!list_is_single_empty_block(&inner->else_list))
      return false;

   nir_block *prev_block =
      nir_cf_node_as_block(nir_cf_node_prev(&outer->cf_node));
   if (nir_block_ends_in_jump(prev_block))
      return false;

   nir_block *prefix_block = nir_if_first_then_block(outer);
   nir_block *suffix_block = nir_cf_node_as_block(n1);
   nir_block *outer_else = nir_if_first_else_block(outer);
   nir_block *merge_block =
      nir_cf_node_as_block(nir_cf_node_next(&outer->cf_node));
   nir_block *inner_last_then = nir_if_last_then_block(inner);
   nir_block *inner_else = nir_if_first_else_block(inner);

   if (nir_block_ends_in_jump(prefix_block) ||
       nir_block_ends_in_jump(suffix_block) ||
       nir_block_ends_in_jump(inner_last_then))
      return false;

   /* Validate prefix_block: small, pure ALU + fault-safe/redirectable loads. */
   unsigned prefix_count = 0, prefix_loads = 0;
   nir_foreach_instr(instr, prefix_block) {
      switch (instr->type) {
      case nir_instr_type_load_const:
      case nir_instr_type_undef:
         break;
      case nir_instr_type_alu:
         prefix_count++;
         break;
      case nir_instr_type_intrinsic: {
         const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (!(nir_intrinsic_infos[intr->intrinsic].flags &
               NIR_INTRINSIC_CAN_ELIMINATE) ||
             instr_is_lane_dependent(intr))
            return false;
         if (nir_intrinsic_has_access(intr) &&
             (nir_intrinsic_access(intr) & (ACCESS_VOLATILE | ACCESS_ATOMIC)))
            return false;

         if (is_redirectable_global_load(intr)) {
            const bool can_spec =
               (nir_intrinsic_access(intr) & ACCESS_CAN_SPECULATE) ||
               dominating_load_proves_safe(prev_block, outer->condition.ssa,
                                           intr);
            if (!can_spec && !fits_zero_page(intr, aligned_from(intr, 0)))
               return false;
            prefix_loads++;
            prefix_count++;
         } else if (nir_instr_can_speculate(instr)) {
            if (nir_intrinsic_has_access(intr)) {
               prefix_loads++;
               prefix_count++;
            }
         } else {
            return false;
         }
         break;
      }
      default:
         return false;
      }
   }

   if (prefix_count == 0 || prefix_count > MAX_PREFIX_INSTRS ||
       prefix_loads > MAX_PREFIX_LOADS)
      return false;

   /* Validate every non-phi instruction in suffix_block and every phi in
    * merge_block: each merge_block phi must reduce cleanly to inner_then or
    * else_val. */
   struct set *matched_suffix_alus = _mesa_pointer_set_create(NULL);
   nir_foreach_phi(phi, merge_block) {
      nir_def *s_val = NULL, *e_val = NULL;
      if (!phi_src_for_pred(phi, suffix_block, &s_val) ||
          !phi_src_for_pred(phi, outer_else, &e_val)) {
         _mesa_set_destroy(matched_suffix_alus, NULL);
         return false;
      }

      if (s_val == e_val)
         continue;

      nir_instr *si = nir_def_instr(s_val);
      if (si->block == suffix_block && si->type == nir_instr_type_phi) {
         nir_def *ithen = NULL, *ielse = NULL;
         if (phi_src_for_pred(nir_instr_as_phi(si), inner_last_then, &ithen) &&
             phi_src_for_pred(nir_instr_as_phi(si), inner_else, &ielse) &&
             ielse == e_val)
            continue;
      }

      if (si->block == suffix_block && si->type == nir_instr_type_alu) {
         nir_def *then_rhs = NULL;
         unsigned phi_src_idx = 0;
         if (match_suffix_identity_alu(nir_instr_as_alu(si), suffix_block,
                                       inner_last_then, inner_else, e_val,
                                       &then_rhs, &phi_src_idx)) {
            _mesa_set_add(matched_suffix_alus, si);
            continue;
         }
      }

      _mesa_set_destroy(matched_suffix_alus, NULL);
      return false;
   }

   /* Ensure no other non-phi instruction lives in suffix_block, and that no
    * instruction or phi in suffix_block has a use outside suffix_block and
    * merge_block's phis (all suffix_block instructions will be removed). */
   nir_foreach_instr(instr, suffix_block) {
      if (instr->type != nir_instr_type_phi &&
          !_mesa_set_search(matched_suffix_alus, instr)) {
         _mesa_set_destroy(matched_suffix_alus, NULL);
         return false;
      }
      nir_def *def = nir_instr_def(instr);
      if (!def)
         continue;
      nir_foreach_use_including_if(use, def) {
         if (nir_src_is_if(use)) {
            _mesa_set_destroy(matched_suffix_alus, NULL);
            return false;
         }
         nir_instr *ui = nir_src_use_instr(use);
         if (ui->block == suffix_block)
            continue;
         if (ui->block == merge_block && ui->type == nir_instr_type_phi)
            continue;
         _mesa_set_destroy(matched_suffix_alus, NULL);
         return false;
      }
   }
   _mesa_set_destroy(matched_suffix_alus, NULL);

   /* Rewrite: redirect unsafe loads in prefix_block, move prefix_block to
    * prev_block, combine conditions into outer->condition, and flatten inner
    * into outer. */
   nir_def *outer_cond = outer->condition.ssa;
   nir_foreach_instr_safe(instr, prefix_block) {
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (is_redirectable_global_load(intr) &&
             !(nir_intrinsic_access(intr) & ACCESS_CAN_SPECULATE) &&
             !dominating_load_proves_safe(prev_block, outer_cond, intr)) {
            b->cursor = nir_before_instr(instr);
            nir_def *addr = intr->src[0].ssa;
            if (fits_zero_page(intr, address_residual(addr))) {
               addr = address_unless(b, addr, outer_cond);
            } else {
               addr = nir_bcsel(
                  b, outer_cond, addr,
                  nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS +
                                      aligned_from(intr, 0)));
            }
            nir_src_rewrite(&intr->src[0], addr);
         }
      }
   }

   nir_foreach_instr_safe(instr, prefix_block) {
      exec_node_remove(&instr->node);
      instr->block = prev_block;
      exec_list_push_tail(&prev_block->instr_list, &instr->node);
   }

   b->cursor = nir_after_block(prev_block);
   nir_def *combined = nir_iand(b, outer_cond, inner->condition.ssa);
   nir_src_rewrite(&outer->condition, combined);

   /* Rewire merge_block phis to read directly from inner_last_then. */
   nir_foreach_phi(phi, merge_block) {
      nir_def *s_val = NULL, *e_val = NULL;
      phi_src_for_pred(phi, suffix_block, &s_val);
      phi_src_for_pred(phi, outer_else, &e_val);

      nir_def *new_then_val = e_val;
      if (s_val != e_val) {
         nir_instr *si = nir_def_instr(s_val);
         if (si->type == nir_instr_type_phi) {
            phi_src_for_pred(nir_instr_as_phi(si), inner_last_then,
                             &new_then_val);
         } else {
            nir_alu_instr *alu = nir_instr_as_alu(si);
            nir_def *then_rhs = NULL;
            unsigned phi_src_idx = 0;
            ASSERTED bool ok = match_suffix_identity_alu(
               alu, suffix_block, inner_last_then, inner_else, e_val, &then_rhs,
               &phi_src_idx);
            assert(ok);
            b->cursor = nir_after_block(inner_last_then);
            nir_def *lhs = (phi_src_idx == 0) ? then_rhs : e_val;
            nir_def *rhs = (phi_src_idx == 0) ? e_val : then_rhs;
            new_then_val = nir_build_alu2(b, alu->op, lhs, rhs);
         }
      }

      nir_foreach_phi_src(src, phi) {
         if (src->pred == suffix_block) {
            nir_src_rewrite(&src->src, new_then_val);
            break;
         }
      }
   }

   /* Suffix instructions now have no remaining uses. */
   nir_foreach_instr_reverse_safe(instr, suffix_block) {
      nir_instr_remove(instr);
   }

   /* Splice inner->then_list before inner and drop the now-empty inner if. */
   nir_cf_list then_cf;
   nir_cf_extract(&then_cf, nir_before_cf_list(&inner->then_list),
                  nir_after_cf_list(&inner->then_list));
   nir_cf_reinsert(&then_cf, nir_before_cf_node(&inner->cf_node));
   nir_cf_node_remove(&inner->cf_node);

   return true;
}

static bool
fuse_nested_guards_in_list(nir_builder *b, struct exec_list *list)
{
   bool progress = false;
   foreach_list_typed_safe(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= fuse_nested_guards_in_list(b, &nif->then_list);
         progress |= fuse_nested_guards_in_list(b, &nif->else_list);
         while (try_fuse_nested_guard(b, nif)) {
            nir_progress(true, b->impl, nir_metadata_none);
            nir_metadata_require(b->impl, nir_metadata_dominance);
            progress = true;
         }
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= fuse_nested_guards_in_list(b, &loop->body);
         break;
      }
      default:
         break;
      }
   }
   return progress;
}

static bool
instr_is_hoistable_across_pure_if(const nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_tex:
      return !nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr));
   case nir_instr_type_intrinsic: {
      const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (!(nir_intrinsic_infos[intr->intrinsic].flags &
            NIR_INTRINSIC_CAN_ELIMINATE) ||
          instr_is_lane_dependent(intr))
         return false;
      if (nir_intrinsic_has_access(intr) &&
          (nir_intrinsic_access(intr) & (ACCESS_VOLATILE | ACCESS_ATOMIC)))
         return false;
      return true;
   }
   default:
      return false;
   }
}

struct dep_check {
   const struct set *blocked;
   bool depends;
};

static bool
check_src_blocked(nir_src *src, void *state)
{
   struct dep_check *dc = state;
   if (_mesa_set_search(dc->blocked, src->ssa)) {
      dc->depends = true;
      return false;
   }
   return true;
}

static unsigned
def_dwords(const nir_def *def)
{
   return def->num_components * DIV_ROUND_UP(def->bit_size, 32);
}

/* Hoist pure, phi-independent load/test blocks across preceding pure read-only
 * `if`s in the same CF list so consecutive unrolled iterations issue their
 * loads together before any control-flow wait barrier. */
static bool
hoist_independent_blocks_in_list(struct exec_list *list)
{
   bool progress = false;

   /* Recurse into nested CF first. */
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= hoist_independent_blocks_in_list(&nif->then_list);
         progress |= hoist_independent_blocks_in_list(&nif->else_list);
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= hoist_independent_blocks_in_list(&loop->body);
      }
   }

   for (nir_cf_node *node = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
        node != NULL;) {
      if (node->type != nir_cf_node_if) {
         node = nir_cf_node_next(node);
         continue;
      }

      nir_if *first_if = nir_cf_node_as_if(node);
      nir_block *prev_block =
         nir_cf_node_as_block(nir_cf_node_prev(&first_if->cf_node));
      if (nir_block_ends_in_jump(prev_block) ||
          !cf_node_is_pure_read_only(&first_if->cf_node)) {
         node = nir_cf_node_next(node);
         continue;
      }

      /* Walk up to MAX_GROUP_IFS consecutive pure ifs:
       *   prev_block -> if_0 -> mid_1 -> if_1 -> ... -> mid_{k} -> if_{k} */
      struct set *blocked = _mesa_pointer_set_create(NULL);
      struct util_dynarray hoist_list = UTIL_DYNARRAY_INIT;
      unsigned group_ifs = 1;
      unsigned live_dwords = 0;

      nir_cf_node *cur = nir_cf_node_next(&first_if->cf_node);
      while (cur && cur->type == nir_cf_node_block &&
             group_ifs < MAX_GROUP_IFS) {
         nir_block *mid = nir_cf_node_as_block(cur);
         nir_cf_node *next_if_node = nir_cf_node_next(&mid->cf_node);
         if (!next_if_node || next_if_node->type != nir_cf_node_if)
            break;
         nir_if *next_if = nir_cf_node_as_if(next_if_node);
         if (!cf_node_is_pure_read_only(&next_if->cf_node))
            break;

         /* Tentatively scan `mid` for hoistable instructions. Stop scanning
          * `mid` (and terminate the hoist group) at the first side-effecting
          * or otherwise non-reorderable instruction (e.g. a store, atomic, or
          * barrier) so no subsequent load or condition can ever move above it. */
         struct util_dynarray cand = UTIL_DYNARRAY_INIT;
         struct set *cand_set = _mesa_pointer_set_create(NULL);
         bool moves_load_or_cond = false;
         bool hit_barrier = false;
         nir_def *next_cond = next_if->condition.ssa;

         nir_foreach_instr(instr, mid) {
            nir_def *def = nir_instr_def(instr);
            if (instr->type == nir_instr_type_phi) {
               if (def)
                  _mesa_set_add(blocked, def);
               continue;
            }
            if (!instr_is_hoistable_across_pure_if(instr)) {
               hit_barrier = true;
               break;
            }
            struct dep_check dc = {blocked, false};
            nir_foreach_src(instr, check_src_blocked, &dc);
            if (dc.depends) {
               if (def)
                  _mesa_set_add(blocked, def);
               continue;
            }
            util_dynarray_append(&cand, instr);
            if (def) {
               _mesa_set_add(cand_set, def);
               if (def == next_cond)
                  moves_load_or_cond = true;
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_intrinsic_has_access(nir_instr_as_intrinsic(instr)))
                  moves_load_or_cond = true;
            }
         }

         if (moves_load_or_cond) {
            /* Count how many 32-bit values from `cand` remain live across
             * `first_if` (i.e. have any use outside `cand`). */
            unsigned add_live = 0;
            util_dynarray_foreach(&cand, nir_instr *, ip) {
               nir_def *def = nir_instr_def(*ip);
               if (!def || (*ip)->type == nir_instr_type_load_const ||
                   (*ip)->type == nir_instr_type_undef)
                  continue;
               bool escapes = false;
               nir_foreach_use_including_if(use, def) {
                  if (nir_src_is_if(use)) {
                     escapes = true;
                     break;
                  }
                  nir_instr *parent = nir_src_use_instr(use);
                  nir_def *pdef = nir_instr_def(parent);
                  if (!pdef || !_mesa_set_search(cand_set, pdef)) {
                     escapes = true;
                     break;
                  }
               }
               if (escapes)
                  add_live += def_dwords(def);
            }

            if (live_dwords + add_live <= MAX_LIVE_HOISTED_DWORDS) {
               live_dwords += add_live;
               util_dynarray_foreach(&cand, nir_instr *, ip) {
                  util_dynarray_append(&hoist_list, *ip);
               }
            } else {
               /* Budget exceeded: stop growing this hoist group. */
               _mesa_set_destroy(cand_set, NULL);
               util_dynarray_fini(&cand);
               break;
            }
         } else {
            /* `cand` stays in `mid`: mark its defs blocked so later `mid`
             * blocks in this group cannot hoist a use above its definition. */
            util_dynarray_foreach(&cand, nir_instr *, ip) {
               nir_def *def = nir_instr_def(*ip);
               if (def)
                  _mesa_set_add(blocked, def);
            }
         }

         _mesa_set_destroy(cand_set, NULL);
         util_dynarray_fini(&cand);

         if (hit_barrier)
            break;

         group_ifs++;
         cur = nir_cf_node_next(&next_if->cf_node);
      }

      if (util_dynarray_num_elements(&hoist_list, nir_instr *) > 0) {
         util_dynarray_foreach(&hoist_list, nir_instr *, ip) {
            nir_instr *instr = *ip;
            exec_node_remove(&instr->node);
            instr->block = prev_block;
            exec_list_push_tail(&prev_block->instr_list, &instr->node);
         }
         progress = true;
      }

      _mesa_set_destroy(blocked, NULL);
      util_dynarray_fini(&hoist_list);

      /* Advance past the group we just processed. */
      node = cur;
   }

   return progress;
}

static bool
cf_list_has_no_loops(const struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop)
         return false;
      if (node->type == nir_cf_node_if) {
         const nir_if *nif = nir_cf_node_as_if(node);
         if (!cf_list_has_no_loops(&nif->then_list) ||
             !cf_list_has_no_loops(&nif->else_list))
            return false;
      }
   }
   return true;
}

static bool
if_body_is_coalesce_candidate(nir_if *nif)
{
   if (nif->control == nir_selection_control_dont_flatten ||
       list_is_single_empty_block(&nif->then_list) ||
       !list_is_single_empty_block(&nif->else_list) ||
       !cf_list_has_no_loops(&nif->then_list) ||
       !nir_src_is_divergent(&nif->condition))
      return false;

   unsigned count = 0;
   bool expensive = false;

   nir_foreach_block_in_cf_node(block, &nif->cf_node) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_load_const:
         case nir_instr_type_undef:
         case nir_instr_type_phi:
            break;
         case nir_instr_type_alu: {
            const nir_op op = nir_instr_as_alu(instr)->op;
            count++;
            switch (op) {
            case nir_op_frsq:
            case nir_op_frcp:
            case nir_op_fsqrt:
            case nir_op_flog2:
            case nir_op_fexp2:
            case nir_op_fsin:
            case nir_op_fcos:
               expensive = true;
               break;
            default:
               break;
            }
            break;
         }
         case nir_instr_type_tex:
            if (nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr)))
               return false;
            expensive = true;
            count++;
            break;
         case nir_instr_type_intrinsic: {
            const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (instr_is_lane_dependent(intr))
               return false;
            if (nir_intrinsic_has_access(intr))
               expensive = true;
            count++;
            break;
         }
         default:
            /* Reject jumps, calls, etc. */
            return false;
         }
      }
   }

   return expensive || count >= MIN_COALESCE_BODY_INSTRS;
}

static bool
block_inside_cf_node(const nir_block *block, const nir_cf_node *ancestor)
{
   for (const nir_cf_node *n = &block->cf_node; n; n = n->parent) {
      if (n == ancestor)
         return true;
   }
   return false;
}

/* Trace the value `def` takes if every `if` in `ifs[0..k)` takes its empty
 * else branch. Because each `mid_m` is phis-only and each `if_m` has an empty
 * else block, walking the else-source terminates at a def dominating `ifs[0]`. */
static nir_def *
trace_all_false_value(nir_def *def, nir_if *const *ifs,
                      nir_block *const *mids, unsigned k)
{
   for (unsigned step = 0; step <= k; ++step) {
      nir_instr *instr = nir_def_instr(def);
      if (instr->type != nir_instr_type_phi)
         return def;
      int idx = -1;
      for (unsigned m = 0; m < k; ++m) {
         if (instr->block == mids[m]) {
            idx = (int)m;
            break;
         }
      }
      if (idx < 0)
         return def;
      nir_block *else_blk = nir_if_first_else_block(ifs[idx]);
      nir_def *e_val = NULL;
      if (!phi_src_for_pred(nir_instr_as_phi(instr), else_blk, &e_val))
         return NULL;
      def = e_val;
   }
   return NULL;
}

static bool
instr_is_pure_mid_cond_prep(const nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;
   case nir_instr_type_alu: {
      const nir_op op = nir_instr_as_alu(instr)->op;
      switch (op) {
      case nir_op_idiv:
      case nir_op_udiv:
      case nir_op_imod:
      case nir_op_umod:
      case nir_op_irem:
         return false;
      default:
         return true;
      }
   }
   case nir_instr_type_intrinsic: {
      const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      return intr->intrinsic == nir_intrinsic_load_preamble ||
             intr->intrinsic == nir_intrinsic_load_push_constant;
   }
   default:
      return false;
   }
}

struct mid_src_check {
   nir_block *prev_block;
   struct set *hoisted_defs;
   bool ok;
};

static bool
check_mid_src_dominates(nir_src *src, void *data)
{
   struct mid_src_check *st = data;
   nir_def *def = src->ssa;
   if (_mesa_set_search(st->hoisted_defs, def))
      return true;
   if (nir_block_dominates(nir_def_block(def), st->prev_block))
      return true;
   st->ok = false;
   return false;
}

/* Check whether all non-phi instructions in `mid` are pure condition-prep ops
 * whose sources dominate `prev_block` (or earlier hoisted mid instructions),
 * appending them to `hoist_instrs` and their defs to `hoisted_defs`. */
static bool
collect_pure_mid_instrs(nir_block *mid, nir_block *prev_block,
                        struct set *hoisted_defs,
                        struct util_dynarray *hoist_instrs)
{
   unsigned non_phi = 0;
   nir_foreach_instr(instr, mid) {
      if (instr->type == nir_instr_type_phi)
         continue;
      if (++non_phi > 16 || !instr_is_pure_mid_cond_prep(instr))
         return false;
      struct mid_src_check sc = {prev_block, hoisted_defs, true};
      nir_foreach_src(instr, check_mid_src_dominates, &sc);
      if (!sc.ok)
         return false;
      nir_def *def = nir_instr_def(instr);
      if (!def)
         return false;
      _mesa_set_add(hoisted_defs, def);
      util_dynarray_append(hoist_instrs, instr);
   }
   return true;
}

/* Match a scalar 1-bit conjunction `a && b` where exactly one side is an
 * integer inequality `u != v`. For the outer fast-path guard, dropping the
 * rare-false `u != v` term is sound because `cond => other`, while each inner
 * `if (cond)` still checks the exact conjunction. */
static nir_def *
relax_ine_conjunction(nir_def *cond, nir_block *prev_block)
{
   nir_alu_instr *alu = nir_def_as_alu_or_null(cond);
   if (!alu || alu->op != nir_op_iand || alu->def.num_components != 1 ||
       alu->def.bit_size != 1 || alu->src[0].swizzle[0] != 0 ||
       alu->src[1].swizzle[0] != 0)
      return cond;

   nir_def *s0 = alu->src[0].src.ssa;
   nir_def *s1 = alu->src[1].src.ssa;
   if (s0->num_components != 1 || s1->num_components != 1)
      return cond;

   nir_def *u = NULL, *v = NULL;
   const bool s0_ine = match_ine(s0, &u, &v);
   const bool s1_ine = match_ine(s1, &u, &v);
   if (s0_ine == s1_ine)
      return cond;

   nir_def *kept = s0_ine ? s1 : s0;
   return nir_block_dominates(nir_def_block(kept), prev_block) ? kept : cond;
}

struct float_thresh_cmp {
   nir_op base_op;
   bool negated;
   nir_def *src[2];
   uint32_t fp_math_ctrl;
};

static bool
match_float_thresh_cmp(nir_def *cond, struct float_thresh_cmp *out)
{
   if (cond->num_components != 1 || cond->bit_size != 1)
      return false;

   bool negated = false;
   nir_alu_instr *alu = nir_def_as_alu_or_null(cond);
   if (!alu)
      return false;
   if (alu->op == nir_op_inot) {
      if (alu->src[0].swizzle[0] != 0)
         return false;
      negated = true;
      alu = nir_def_as_alu_or_null(alu->src[0].src.ssa);
      if (!alu)
         return false;
   }

   if ((alu->op != nir_op_fge && alu->op != nir_op_flt) ||
       alu->def.num_components != 1 ||
       alu->src[0].swizzle[0] != 0 || alu->src[1].swizzle[0] != 0)
      return false;

   nir_def *s0 = alu->src[0].src.ssa;
   nir_def *s1 = alu->src[1].src.ssa;
   if (s0->bit_size != 32 || s1->bit_size != 32 ||
       s0->num_components != 1 || s1->num_components != 1)
      return false;

   out->base_op = alu->op;
   out->negated = negated;
   out->src[0] = s0;
   out->src[1] = s1;
   out->fp_math_ctrl = alu->fp_math_ctrl;
   return true;
}

static bool
same_float_thresh_def(const nir_def *a, const nir_def *b)
{
   if (a == b)
      return true;
   return nir_def_is_const(a) && nir_def_is_const(b) &&
          nir_scalar_as_uint(nir_get_scalar((nir_def *)a, 0)) ==
             nir_scalar_as_uint(nir_get_scalar((nir_def *)b, 0));
}

static bool
def_is_float_zero(const nir_def *def)
{
   return def->num_components == 1 && def->bit_size == 32 &&
          nir_def_is_const(def) &&
          nir_scalar_as_float(nir_get_scalar((nir_def *)def, 0)) == 0.0f;
}

/* When all `k` compare inputs are single-use `fadd(x_m, shared_bias)` compared
 * against `0.0`, unwrap to `x_m` against `-shared_bias` (matching
 * `nir_opt_algebraic_late`'s `fge/flt(fadd(a, b), 0.0) -> fge/flt(a, -b)`
 * rule before our selection tree adds a second use to the `fadd`). */
static bool
try_unwrap_zero_fadd_bias(nir_builder *b, nir_block *prev_block,
                          nir_op base_op, nir_def **vals, unsigned k,
                          nir_def **thresh_inout)
{
   if (!def_is_float_zero(*thresh_inout))
      return false;

   nir_alu_instr *fadds[MAX_GROUP_IFS];
   for (unsigned m = 0; m < k; ++m) {
      nir_alu_instr *fa = nir_def_as_alu_or_null(vals[m]);
      if (!fa || fa->op != nir_op_fadd || fa->def.num_components != 1 ||
          fa->def.bit_size != 32 || !list_is_singular(&fa->def.uses) ||
          fa->src[0].swizzle[0] != 0 || fa->src[1].swizzle[0] != 0)
         return false;
      if (base_op == nir_op_fge && (fa->fp_math_ctrl & nir_fp_preserve_inf))
         return false;
      fadds[m] = fa;
   }

   /* Find an additive operand shared across all `k` fadds. */
   nir_def *bias = NULL;
   nir_def *unwrapped[MAX_GROUP_IFS];
   for (unsigned s0 = 0; s0 < 2 && !bias; ++s0) {
      nir_def *cand = fadds[0]->src[s0].src.ssa;
      if (cand->num_components != 1 || cand->bit_size != 32 ||
          !nir_block_dominates(nir_def_block(cand), prev_block))
         continue;
      unwrapped[0] = fadds[0]->src[1 - s0].src.ssa;
      bool all_match = true;
      for (unsigned m = 1; m < k; ++m) {
         if (same_float_thresh_def(fadds[m]->src[0].src.ssa, cand)) {
            unwrapped[m] = fadds[m]->src[1].src.ssa;
         } else if (same_float_thresh_def(fadds[m]->src[1].src.ssa, cand)) {
            unwrapped[m] = fadds[m]->src[0].src.ssa;
         } else {
            all_match = false;
            break;
         }
      }
      if (!all_match)
         continue;
      for (unsigned m = 0; m < k; ++m) {
         if (unwrapped[m]->num_components != 1 ||
             unwrapped[m]->bit_size != 32 ||
             !nir_block_dominates(nir_def_block(unwrapped[m]), prev_block)) {
            all_match = false;
            break;
         }
      }
      if (all_match)
         bias = cand;
   }

   if (!bias)
      return false;

   for (unsigned m = 0; m < k; ++m)
      vals[m] = unwrapped[m];

   /* Unwrap fneg(fneg(x)) -> x (IEEE-754 bit-exact including signed zeros,
    * infinities, and NaN payloads) so downstream passes see the original
    * uniform/SSA value instead of a redundant double negation. */
   nir_alu_instr *bias_alu = nir_def_as_alu_or_null(bias);
   if (bias_alu && bias_alu->op == nir_op_fneg &&
       bias_alu->def.num_components == 1 && bias_alu->def.bit_size == 32 &&
       bias_alu->src[0].swizzle[0] == 0 &&
       bias_alu->src[0].src.ssa->num_components == 1 &&
       nir_block_dominates(nir_def_block(bias_alu->src[0].src.ssa), prev_block)) {
      *thresh_inout = bias_alu->src[0].src.ssa;
   } else {
      *thresh_inout = nir_fneg(b, bias);
   }
   return true;
}

/* When `k >= 3` outer terms test `k` 32-bit floats against the same threshold
 * `T` with the same ordered or negated-ordered comparison `P(x)`, reduce
 * `P(x_0) | ... | P(x_{k-1})` via a balanced binary selection tree:
 *
 *   For negated `P(x) = !cmp(x, T)`:   pick(a, b) = bcsel(cmp(a, T), b, a)
 *   For ordered `P(x) =  cmp(x, T)`:   pick(a, b) = bcsel(cmp(a, T), a, b)
 *
 * This is IEEE-754 bit-exact including NaN (unlike fmin/fmax, which drop NaN):
 * whenever `P(a)` holds (including NaN for `!cmp`), `pick(a, b)` returns `a`;
 * otherwise it returns `b`, so `P(pick(a, b)) == P(a) | P(b)`.
 *
 * The backend fuses each `cmp + bcsel` into a single `FCMPSEL` and the final
 * compare into `IF_FCMP`, replacing `k` boolean compares plus `k - 1` ORs with
 * `k - 1` `FCMPSEL`s plus one fused branch compare. */
static nir_def *
try_build_float_thresh_tree(nir_builder *b, nir_block *prev_block,
                            nir_def *const *terms, unsigned k)
{
   if (k < 3)
      return NULL;

   struct float_thresh_cmp cmps[MAX_GROUP_IFS];
   for (unsigned m = 0; m < k; ++m) {
      if (!match_float_thresh_cmp(terms[m], &cmps[m]))
         return NULL;
      if (m > 0 && (cmps[m].base_op != cmps[0].base_op ||
                    cmps[m].negated != cmps[0].negated))
         return NULL;
   }

   int thresh_side = -1;
   for (unsigned side = 2; side-- > 0;) {
      bool shared = true;
      for (unsigned m = 1; m < k; ++m) {
         if (!same_float_thresh_def(cmps[m].src[side], cmps[0].src[side])) {
            shared = false;
            break;
         }
      }
      if (shared) {
         thresh_side = (int)side;
         break;
      }
   }
   if (thresh_side < 0)
      return NULL;

   nir_def *thresh = cmps[0].src[thresh_side];
   if (!nir_block_dominates(nir_def_block(thresh), prev_block))
      return NULL;

   nir_def *vals[MAX_GROUP_IFS];
   uint32_t fp_ctrl = 0;
   for (unsigned m = 0; m < k; ++m) {
      vals[m] = cmps[m].src[1 - thresh_side];
      if (!nir_block_dominates(nir_def_block(vals[m]), prev_block))
         return NULL;
      fp_ctrl |= cmps[m].fp_math_ctrl;
   }

   const nir_op base_op = cmps[0].base_op;
   const bool negated = cmps[0].negated;
   try_unwrap_zero_fadd_bias(b, prev_block, base_op, vals, k, &thresh);

   const uint32_t saved_fp_ctrl = b->fp_math_ctrl;
   b->fp_math_ctrl = fp_ctrl;

   unsigned n = k;
   while (n > 1) {
      unsigned out = 0;
      for (unsigned i = 0; i + 1 < n; i += 2) {
         nir_def *va = vals[i];
         nir_def *vb = vals[i + 1];
         nir_def *c = thresh_side == 1 ? nir_build_alu2(b, base_op, va, thresh)
                                       : nir_build_alu2(b, base_op, thresh, va);
         vals[out++] = negated ? nir_bcsel(b, c, vb, va)
                               : nir_bcsel(b, c, va, vb);
      }
      if (n & 1)
         vals[out++] = vals[n - 1];
      n = out;
   }

   nir_def *final_cmp =
      thresh_side == 1 ? nir_build_alu2(b, base_op, vals[0], thresh)
                       : nir_build_alu2(b, base_op, thresh, vals[0]);
   b->fp_math_ctrl = saved_fp_ctrl;

   return negated ? nir_inot(b, final_cmp) : final_cmp;
}

static bool
alu_is_sinkable_cond_op(nir_op op)
{
   switch (op) {
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_inot:
   case nir_op_bcsel:
   case nir_op_ine:
   case nir_op_ieq:
   case nir_op_ilt:
   case nir_op_ige:
   case nir_op_ult:
   case nir_op_uge:
   case nir_op_fge:
   case nir_op_flt:
   case nir_op_feq:
   case nir_op_fneu:
   case nir_op_fadd:
   case nir_op_iadd:
   case nir_op_isub:
      return true;
   default:
      return false;
   }
}

static void
collect_prev_cond_dag(nir_def *def, nir_block *prev_block,
                      struct set *cond_instrs)
{
   nir_instr *instr = nir_def_instr(def);
   if (instr->block != prev_block || instr->type != nir_instr_type_alu)
      return;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (!alu_is_sinkable_cond_op(alu->op) ||
       _mesa_set_search(cond_instrs, instr))
      return;

   _mesa_set_add(cond_instrs, instr);
   const unsigned n = nir_op_infos[alu->op].num_inputs;
   for (unsigned i = 0; i < n; ++i)
      collect_prev_cond_dag(alu->src[i].src.ssa, prev_block, cond_instrs);
}

static bool
cf_node_strictly_inside(const nir_cf_node *node, const nir_cf_node *ancestor)
{
   for (const nir_cf_node *n = node->parent; n; n = n->parent) {
      if (n == ancestor)
         return true;
   }
   return false;
}

/* Sink condition-DAG ALU instructions in `prev_block` whose every use now lies
 * strictly inside `outer` (because the outer fast-path guard relaxed or
 * replaced the individual `if` conditions) to the entry of `outer->then_list`. */
static void
sink_superseded_cond_alu(nir_block *prev_block, nir_if *outer,
                         nir_if *const *ifs, unsigned k)
{
   struct set *cond_instrs = _mesa_pointer_set_create(NULL);
   for (unsigned m = 0; m < k; ++m)
      collect_prev_cond_dag(ifs[m]->condition.ssa, prev_block, cond_instrs);

   nir_block *then_first = nir_if_first_then_block(outer);
   nir_foreach_instr_reverse_safe(instr, prev_block) {
      if (!_mesa_set_search(cond_instrs, instr))
         continue;
      nir_def *def = nir_instr_def(instr);
      if (!def)
         continue;

      bool any_use = false;
      bool all_inside = true;
      nir_foreach_use_including_if(use, def) {
         any_use = true;
         if (nir_src_is_if(use)) {
            if (!cf_node_strictly_inside(&nir_src_use_if(use)->cf_node,
                                         &outer->cf_node)) {
               all_inside = false;
               break;
            }
         } else {
            nir_block *ub = nir_src_use_instr(use)->block;
            if (!block_inside_cf_node(ub, &outer->cf_node)) {
               all_inside = false;
               break;
            }
         }
      }

      if (any_use && all_inside) {
         exec_node_remove(&instr->node);
         instr->block = then_first;
         exec_list_push_head(&then_first->instr_list, &instr->node);
      }
   }

   _mesa_set_destroy(cond_instrs, NULL);
}

/* Wrap adjacent expensive one-sided divergent `if`s (with only phis and pure
 * condition-prep ops between them and all conditions computable before the
 * first `if`) in a combined guard `if (c0 | c1 | ... | c_{k-1})`. */
static bool
coalesce_adjacent_ifs_in_list(nir_function_impl *impl, nir_builder *b,
                              struct exec_list *list)
{
   bool progress = false;

   /* Recurse into nested CF first. */
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= coalesce_adjacent_ifs_in_list(impl, b, &nif->then_list);
         progress |= coalesce_adjacent_ifs_in_list(impl, b, &nif->else_list);
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         progress |= coalesce_adjacent_ifs_in_list(impl, b, &loop->body);
      }
   }

   for (nir_cf_node *node = exec_node_data(nir_cf_node, exec_list_get_head(list), node);
        node != NULL;) {
      if (node->type != nir_cf_node_if) {
         node = nir_cf_node_next(node);
         continue;
      }

      nir_if *ifs[MAX_GROUP_IFS];
      nir_block *mids[MAX_GROUP_IFS]; /* mids[m] is the block right after ifs[m] */
      unsigned k = 0;

      nir_if *first_if = nir_cf_node_as_if(node);
      nir_block *prev_block =
         nir_cf_node_as_block(nir_cf_node_prev(&first_if->cf_node));
      if (nir_block_ends_in_jump(prev_block) ||
          !if_body_is_coalesce_candidate(first_if)) {
         node = nir_cf_node_next(node);
         continue;
      }

      ifs[0] = first_if;
      mids[0] = nir_cf_node_as_block(nir_cf_node_next(&first_if->cf_node));
      k = 1;

      struct set *hoisted_defs = _mesa_pointer_set_create(NULL);
      struct util_dynarray hoist_instrs = UTIL_DYNARRAY_INIT;

      while (k < MAX_GROUP_IFS) {
         nir_block *mid = mids[k - 1];
         nir_cf_node *next_node = nir_cf_node_next(&mid->cf_node);
         if (!next_node || next_node->type != nir_cf_node_if)
            break;
         nir_if *next_if = nir_cf_node_as_if(next_node);
         if (!if_body_is_coalesce_candidate(next_if))
            break;

         const unsigned saved_hoist =
            util_dynarray_num_elements(&hoist_instrs, nir_instr *);
         if (!block_is_phis_only(mid) &&
             !collect_pure_mid_instrs(mid, prev_block, hoisted_defs,
                                      &hoist_instrs)) {
            nir_instr **arr = util_dynarray_begin(&hoist_instrs);
            const unsigned cur_n =
               util_dynarray_num_elements(&hoist_instrs, nir_instr *);
            for (unsigned j = saved_hoist; j < cur_n; ++j)
               _mesa_set_remove_key(hoisted_defs, nir_instr_def(arr[j]));
            hoist_instrs.size = saved_hoist * sizeof(nir_instr *);
            break;
         }

         /* Condition must dominate `prev_block` or be hoisted into it. */
         nir_def *cond_def = next_if->condition.ssa;
         if (!_mesa_set_search(hoisted_defs, cond_def) &&
             !nir_block_dominates(nir_def_block(cond_def), prev_block)) {
            nir_instr **arr = util_dynarray_begin(&hoist_instrs);
            const unsigned cur_n =
               util_dynarray_num_elements(&hoist_instrs, nir_instr *);
            for (unsigned j = saved_hoist; j < cur_n; ++j)
               _mesa_set_remove_key(hoisted_defs, nir_instr_def(arr[j]));
            hoist_instrs.size = saved_hoist * sizeof(nir_instr *);
            break;
         }

         ifs[k] = next_if;
         mids[k] = nir_cf_node_as_block(nir_cf_node_next(&next_if->cf_node));
         k++;
      }

      if (k < 2) {
         _mesa_set_destroy(hoisted_defs, NULL);
         util_dynarray_fini(&hoist_instrs);
         node = nir_cf_node_next(node);
         continue;
      }

      /* Collect every phi in mids[0..k-1] whose value escapes the chain,
       * along with its all-false value dominating first_if. */
      struct util_dynarray esc_phis = UTIL_DYNARRAY_INIT;
      struct util_dynarray esc_inits = UTIL_DYNARRAY_INIT;
      bool ok = true;

      for (unsigned m = 0; m < k && ok; ++m) {
         nir_foreach_phi(phi, mids[m]) {
            bool escapes = false;
            nir_foreach_use_including_if(use, &phi->def) {
               if (nir_src_is_if(use)) {
                  nir_if *uif = nir_src_use_if(use);
                  bool in_chain = false;
                  for (unsigned j = 0; j < k; ++j) {
                     if (uif == ifs[j]) {
                        in_chain = true;
                        break;
                     }
                  }
                  if (!in_chain) {
                     escapes = true;
                     break;
                  }
               } else {
                  nir_instr *ui = nir_src_use_instr(use);
                  nir_block *ub = ui->block;
                  bool in_chain = false;
                  for (unsigned j = 0; j < k; ++j) {
                     if (block_inside_cf_node(ub, &ifs[j]->cf_node) ||
                         (j < k - 1 && ub == mids[j]) ||
                         (j == k - 1 && ub == mids[j] &&
                          ui->type == nir_instr_type_phi)) {
                        in_chain = true;
                        break;
                     }
                  }
                  if (!in_chain) {
                     escapes = true;
                     break;
                  }
               }
            }
            if (!escapes)
               continue;

            nir_def *init = trace_all_false_value(&phi->def, ifs, mids, k);
            if (!init) {
               ok = false;
               break;
            }
            util_dynarray_append(&esc_phis, phi);
            util_dynarray_append(&esc_inits, init);
         }
      }

      if (!ok) {
         _mesa_set_destroy(hoisted_defs, NULL);
         util_dynarray_fini(&hoist_instrs);
         util_dynarray_fini(&esc_phis);
         util_dynarray_fini(&esc_inits);
         node = &mids[k - 1]->cf_node;
         continue;
      }

      /* Hoist pure condition-prep instructions from mids[0..k-2] into
       * prev_block so all `ifs[0..k-1]` conditions are ready before `outer`
       * and every intermediate block is phis-only. */
      util_dynarray_foreach(&hoist_instrs, nir_instr *, ip) {
         nir_instr *instr = *ip;
         exec_node_remove(&instr->node);
         instr->block = prev_block;
         exec_list_push_tail(&prev_block->instr_list, &instr->node);
      }
      _mesa_set_destroy(hoisted_defs, NULL);
      util_dynarray_fini(&hoist_instrs);

      /* Build combined condition at end of prev_block. Relax conjunctive
       * `a != b && pred` terms for the outer fast-path guard and, when `k >= 3`
       * terms compare floats against the same threshold, reduce them via a
       * NaN-exact FCMPSEL selection tree instead of materializing `k` booleans. */
      b->cursor = nir_after_block(prev_block);
      nir_def *outer_terms[MAX_GROUP_IFS];
      bool simplified_outer = false;
      for (unsigned m = 0; m < k; ++m) {
         outer_terms[m] =
            relax_ine_conjunction(ifs[m]->condition.ssa, prev_block);
         if (outer_terms[m] != ifs[m]->condition.ssa)
            simplified_outer = true;
      }

      nir_def *any = try_build_float_thresh_tree(b, prev_block, outer_terms, k);
      if (any) {
         simplified_outer = true;
      } else {
         any = outer_terms[0];
         for (unsigned m = 1; m < k; ++m)
            any = nir_ior(b, any, outer_terms[m]);
      }

      /* Insert outer `if (any)` right before `ifs[0]`, then extract
       * `[ifs[0] .. mids[k-1] phis]` and move it into `outer->then_list`. */
      b->cursor = nir_before_cf_node(&ifs[0]->cf_node);
      nir_if *outer = nir_push_if(b, any);
      nir_pop_if(b, outer);

      nir_cf_list chain;
      nir_cf_extract(&chain, nir_before_cf_node(&ifs[0]->cf_node),
                     nir_after_phis(mids[k - 1]));
      nir_cf_reinsert(&chain, nir_before_cf_list(&outer->then_list));

      if (simplified_outer)
         sink_superseded_cond_alu(prev_block, outer, ifs, k);

      nir_block *outer_then_last = nir_if_last_then_block(outer);
      nir_block *outer_else = nir_if_first_else_block(outer);
      nir_block *outer_merge =
         nir_cf_node_as_block(nir_cf_node_next(&outer->cf_node));

      const unsigned n_esc =
         util_dynarray_num_elements(&esc_phis, nir_phi_instr *);
      nir_phi_instr **phis_arr = util_dynarray_begin(&esc_phis);
      nir_def **inits_arr = util_dynarray_begin(&esc_inits);

      for (unsigned e = 0; e < n_esc; ++e) {
         nir_phi_instr *old_phi = phis_arr[e];
         nir_phi_instr *new_phi = nir_phi_instr_create(b->shader);
         nir_def_init(&new_phi->instr, &new_phi->def,
                      old_phi->def.num_components, old_phi->def.bit_size);
         nir_phi_instr_add_src(new_phi, outer_then_last, &old_phi->def);
         nir_phi_instr_add_src(new_phi, outer_else, inits_arr[e]);
         nir_instr_insert(nir_before_block(outer_merge), &new_phi->instr);

         nir_foreach_use_including_if_safe(use, &old_phi->def) {
            if (!nir_src_is_if(use) &&
                nir_src_use_instr(use) == &new_phi->instr)
               continue;
            nir_block *ub = nir_src_get_block(use);
            if (block_inside_cf_node(ub, &outer->cf_node))
               continue;
            nir_src_rewrite(use, &new_phi->def);
         }
      }

      util_dynarray_fini(&esc_phis);
      util_dynarray_fini(&esc_inits);

      progress = true;
      nir_progress(true, impl, nir_metadata_none);
      nir_metadata_require(impl, nir_metadata_dominance |
                                    nir_metadata_divergence);
      node = &outer_merge->cf_node;
   }

   return progress;
}

bool
agx_nir_opt_guard_chains(nir_shader *shader)
{
   bool pre_progress = false;

   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);
      bool impl_progress = false;

      nir_metadata_require(impl, nir_metadata_dominance);
      if (fuse_nested_guards_in_list(&b, &impl->body)) {
         impl_progress = true;
         nir_progress(true, impl, nir_metadata_none);
      }

      if (hoist_independent_blocks_in_list(&impl->body)) {
         impl_progress = true;
         nir_progress(true, impl, nir_metadata_none);
      }

      if (!impl_progress)
         nir_no_progress(impl);
      pre_progress |= impl_progress;
   }

   /* CSE values that fusion/hoisting just pulled out of separate `if`s into a
    * common block so coalescing sees shared float thresholds and biases. */
   if (pre_progress)
      nir_opt_cse(shader);

   bool coalesce_progress = false;
   nir_foreach_function_impl(impl, shader) {
      nir_builder b = nir_builder_create(impl);

      nir_metadata_require(impl, nir_metadata_dominance |
                                    nir_metadata_divergence);
      if (coalesce_adjacent_ifs_in_list(impl, &b, &impl->body)) {
         coalesce_progress = true;
         nir_progress(true, impl, nir_metadata_none);
      } else {
         nir_no_progress(impl);
      }
   }

   return pre_progress || coalesce_progress;
}

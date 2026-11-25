/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_math.h"

/*
 * Optimize single-workgroup shared-memory indirect bitonic sort kernels:
 *
 * When a compute shader executes only on workgroup 0, initializes a shared
 * uint32 index array `idx[0..N-1] = i` and `idx[N..P-1] = sentinel` (with
 * power-of-two P <= capacity), performs a bitonic sort over `idx[]` using a
 * read-only global key `key(a) = (a != sentinel) ? load_global(addr(a)) : def`
 * with zero global stores prior to sort completion, and `2 * shared_size <= 32K`:
 *
 *  1. Allocate a shadow shared array `key[]` right after `idx[]`.
 *  2. Memoize `key[i] = load_global(addr(i))` once during the init loop and
 *     `key[i] = def` during the pad loop (eliminating O(N log^2 N) random
 *     global gathers inside the bitonic sort).
 *  3. Execute the intra-subgroup stages (`k <= 32` and the `j <= 16` tails of
 *     every `k >= 64`) in registers using native 1-cycle AGX3 `shuffle_xor`
 *     with zero shared-memory traffic and zero workgroup barriers.
 *  4. For `j >= 32` stages, iterate directly over the `P / 2` active pairs
 *     (`p = lid; p < P/2; p += WG`) so 100% of lanes perform a valid compare-
 *     exchange with zero `if (i < ixj)` divergence and 2x fewer iterations.
 *
 * All floating-point/integer comparisons (including NaN, -0.0, and ties) are
 * evaluated identically to the original shader (bit-exact IEEE-754).
 */

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
is_const_u32(nir_def *def, uint32_t expected)
{
   def = strip_mov(def);
   return def && nir_src_is_const(nir_src_for_ssa(def)) &&
          nir_src_as_uint(nir_src_for_ssa(def)) == expected;
}

static bool
match_sysval_comp(nir_def *def, nir_intrinsic_op op, unsigned comp)
{
   def = strip_mov(def);
   if (!def)
      return false;

   if (nir_def_instr(def)->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(def));
      return intr->intrinsic == op && comp == 0 &&
             intr->def.num_components == 1;
   }

   if (nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
      if (alu->op == nir_op_mov) {
         nir_def *src = alu->src[0].src.ssa;
         if (nir_def_instr(src)->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(nir_def_instr(src))->intrinsic == op &&
             alu->src[0].swizzle[0] == comp)
            return true;
      }
   }

   return false;
}

static bool
match_sysval_alu_src(const nir_alu_src *asrc, nir_intrinsic_op op, unsigned comp)
{
   nir_def *src = asrc->src.ssa;
   if (!src)
      return false;
   if (nir_def_instr(src)->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(nir_def_instr(src))->intrinsic == op &&
       asrc->swizzle[0] == comp)
      return true;
   return match_sysval_comp(src, op, comp);
}

static bool
match_lid_x(nir_def *def)
{
   return match_sysval_comp(def, nir_intrinsic_load_local_invocation_id, 0) ||
          match_sysval_comp(def, nir_intrinsic_load_local_invocation_index, 0);
}

static bool
is_dispatch_uniform_offset(nir_def *def, unsigned depth)
{
   if (depth > 8)
      return false;
   def = strip_mov(def);
   if (!def)
      return false;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const)
      return true;
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_base_workgroup_id:
      case nir_intrinsic_load_preamble:
      case nir_intrinsic_load_root_agx:
      case nir_intrinsic_load_push_constant:
         return true;
      case nir_intrinsic_load_global_constant:
         return is_dispatch_uniform_offset(intr->src[0].ssa, depth + 1);
      default:
         return false;
      }
   }
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      const nir_op_info *info = &nir_op_infos[alu->op];
      for (unsigned s = 0; s < info->num_inputs; s++) {
         if (!is_dispatch_uniform_offset(alu->src[s].src.ssa, depth + 1))
            return false;
      }
      return true;
   }
   return false;
}

static bool
match_wg_id_x_src(const nir_alu_src *asrc)
{
   if (match_sysval_alu_src(asrc, nir_intrinsic_load_workgroup_id, 0))
      return true;

   nir_def *def = strip_mov(asrc->src.ssa);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_iadd) {
      for (unsigned s = 0; s < 2; s++) {
         if (match_sysval_alu_src(&alu->src[s], nir_intrinsic_load_workgroup_id,
                                  0) &&
             is_dispatch_uniform_offset(alu->src[1 - s].src.ssa, 0))
            return true;
      }
   }
   return false;
}

static bool
cf_list_is_empty(struct exec_list *list)
{
   if (!exec_list_is_singular(list))
      return false;
   nir_cf_node *n =
      exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   return n->type == nir_cf_node_block &&
          exec_list_is_empty(&nir_cf_node_as_block(n)->instr_list);
}

/* Match `val = iadd(base, step)` or `iadd(step, base)`. */
static bool
match_iadd_const(nir_def *def, nir_def *base, uint32_t step)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != nir_op_iadd)
      return false;
   for (unsigned s = 0; s < 2; s++) {
      if (strip_mov(alu->src[s].src.ssa) == strip_mov(base) &&
          is_const_u32(alu->src[1 - s].src.ssa, step))
         return true;
   }
   return false;
}

/* Match `off = idx << 2` (or `idx * 4` / `amul(idx, 4)`). */
static bool
match_u32_byte_offset(nir_def *off, nir_def **idx_out)
{
   off = strip_mov(off);
   if (!off || nir_def_instr(off)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(off));
   if (alu->op == nir_op_ishl && is_const_u32(alu->src[1].src.ssa, 2)) {
      *idx_out = strip_mov(alu->src[0].src.ssa);
      return true;
   }
   if (alu->op == nir_op_imul || alu->op == nir_op_amul) {
      for (unsigned s = 0; s < 2; s++) {
         if (is_const_u32(alu->src[s].src.ssa, 4)) {
            *idx_out = strip_mov(alu->src[1 - s].src.ssa);
            return true;
         }
      }
   }
   return false;
}

/* Match `loop` header single induction variable phi. */
static bool
match_single_loop_phi(nir_loop *loop, nir_phi_instr **phi_out,
                      nir_def **init_out, nir_def **next_out)
{
   nir_block *hdr = nir_loop_first_block(loop);
   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   nir_block *last = nir_loop_last_block(loop);
   nir_phi_instr *found = NULL;

   nir_foreach_phi(phi, hdr) {
      if (found)
         return false;
      found = phi;
   }
   if (!found || found->def.bit_size != 32 || found->def.num_components != 1)
      return false;

   nir_def *init = NULL, *next = NULL;
   unsigned cnt = 0;
   nir_foreach_phi_src(ps, found) {
      cnt++;
      if (ps->pred == pre)
         init = ps->src.ssa;
      else if (ps->pred->index >= hdr->index && ps->pred->index <= last->index) {
         if (next && next != ps->src.ssa)
            return false;
         next = ps->src.ssa;
      } else {
         return false;
      }
   }
   if (!init || !next || cnt < 2)
      return false;
   *phi_out = found;
   *init_out = strip_mov(init);
   *next_out = strip_mov(next);
   return true;
}

/* Match `if (cond) break;` (or `if (!cond) {} else { break; }`) and return
 * `cond` that triggers the break. */
static bool
match_break_if(nir_if *nif, nir_def **break_cond_out, bool *negated_out)
{
   nir_block *tb = nir_if_first_then_block(nif);
   nir_block *eb = nir_if_first_else_block(nif);
   bool t_brk = tb == nir_if_last_then_block(nif) &&
                nir_block_ends_in_break(tb) &&
                exec_list_is_singular(&tb->instr_list);
   bool e_brk = eb == nir_if_last_else_block(nif) &&
                nir_block_ends_in_break(eb) &&
                exec_list_is_singular(&eb->instr_list);
   bool t_emp = cf_list_is_empty(&nif->then_list);
   bool e_emp = cf_list_is_empty(&nif->else_list);

   if (t_brk && e_emp) {
      *break_cond_out = strip_mov(nif->condition.ssa);
      *negated_out = false;
      return true;
   }
   if (e_brk && t_emp) {
      *break_cond_out = strip_mov(nif->condition.ssa);
      *negated_out = true;
      return true;
   }
   return false;
}

/* Match condition equivalent to `iv >= bound` (signed or unsigned). */
static bool
match_iv_ge_bound(nir_def *cond, bool neg, nir_def *iv, nir_def **bound_out)
{
   cond = strip_mov(cond);
   while (cond && nir_def_instr(cond)->type == nir_instr_type_alu &&
          nir_instr_as_alu(nir_def_instr(cond))->op == nir_op_inot) {
      neg = !neg;
      cond = strip_mov(nir_instr_as_alu(nir_def_instr(cond))->src[0].src.ssa);
   }
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   nir_def *s0 = strip_mov(alu->src[0].src.ssa);
   nir_def *s1 = strip_mov(alu->src[1].src.ssa);

   if (!neg && (alu->op == nir_op_ige || alu->op == nir_op_uge) && s0 == iv) {
      *bound_out = s1;
      return true;
   }
   if (neg && (alu->op == nir_op_ilt || alu->op == nir_op_ult) && s0 == iv) {
      *bound_out = s1;
      return true;
   }
   return false;
}

/* Match condition equivalent to `iv > bound` (i.e. `bound < iv`). */
static bool
match_iv_gt_bound(nir_def *cond, bool neg, nir_def *iv, nir_def *bound)
{
   cond = strip_mov(cond);
   while (cond && nir_def_instr(cond)->type == nir_instr_type_alu &&
          nir_instr_as_alu(nir_def_instr(cond))->op == nir_op_inot) {
      neg = !neg;
      cond = strip_mov(nir_instr_as_alu(nir_def_instr(cond))->src[0].src.ssa);
   }
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   nir_def *s0 = strip_mov(alu->src[0].src.ssa);
   nir_def *s1 = strip_mov(alu->src[1].src.ssa);

   if (!neg && (alu->op == nir_op_ult || alu->op == nir_op_ilt) &&
       s0 == bound && s1 == iv)
      return true;
   if (neg && (alu->op == nir_op_uge || alu->op == nir_op_ige) &&
       s0 == bound && s1 == iv)
      return true;
   return false;
}

/* Match `for (i = init; i < bound; i += wg_x) store_shared(val, i << 2)`. */
static bool
match_simple_shared_fill_loop(nir_loop *loop, uint32_t wg_x,
                              nir_phi_instr **iv_phi_out,
                              nir_def **init_out, nir_def **bound_out,
                              nir_intrinsic_instr **store_out)
{
   nir_phi_instr *iv = NULL;
   nir_def *init = NULL, *next = NULL;
   if (!match_single_loop_phi(loop, &iv, &init, &next))
      return false;
   if (!match_iadd_const(next, &iv->def, wg_x))
      return false;

   nir_block *hdr = nir_loop_first_block(loop);
   nir_cf_node *n1 = nir_cf_node_next(&hdr->cf_node);
   if (!n1 || n1->type != nir_cf_node_if)
      return false;
   nir_def *bcond = NULL;
   bool bneg = false;
   if (!match_break_if(nir_cf_node_as_if(n1), &bcond, &bneg))
      return false;
   nir_def *bound = NULL;
   if (!match_iv_ge_bound(bcond, bneg, &iv->def, &bound))
      return false;

   nir_cf_node *n2 = nir_cf_node_next(n1);
   if (!n2 || n2->type != nir_cf_node_block || nir_cf_node_next(n2) != NULL)
      return false;

   nir_block *body = nir_cf_node_as_block(n2);
   nir_intrinsic_instr *store = NULL;
   nir_foreach_instr(instr, body) {
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_shared) {
            if (store)
               return false;
            store = intr;
         } else {
            return false;
         }
      } else if (instr->type != nir_instr_type_alu &&
                 instr->type != nir_instr_type_load_const) {
         return false;
      }
   }
   if (!store || nir_intrinsic_base(store) != 0 ||
       nir_src_bit_size(store->src[0]) != 32 ||
       nir_src_num_components(store->src[0]) != 1 ||
       (nir_intrinsic_access(store) & ACCESS_VOLATILE))
      return false;

   nir_def *store_idx = NULL;
   if (!match_u32_byte_offset(store->src[1].ssa, &store_idx) ||
       store_idx != &iv->def)
      return false;

   *iv_phi_out = iv;
   *init_out = init;
   *bound_out = bound;
   *store_out = store;
   return true;
}

/* Match `val == N + lid` (or `lid + N`). */
static bool
match_add_lid(nir_def *def, nir_def *base)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != nir_op_iadd)
      return false;
   for (unsigned s = 0; s < 2; s++) {
      if (strip_mov(alu->src[s].src.ssa) == strip_mov(base) &&
          (match_sysval_alu_src(&alu->src[1 - s],
                                nir_intrinsic_load_local_invocation_id, 0) ||
           match_sysval_alu_src(&alu->src[1 - s],
                                nir_intrinsic_load_local_invocation_index, 0)))
         return true;
   }
   return false;
}

/* Match `next_pow2(N)` computed via bit-smear on `N - 1` plus `1`. */
static bool
match_bit_smear_step(nir_def *def, unsigned shift, nir_def **in_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *ior = nir_instr_as_alu(nir_def_instr(def));
   if (ior->op != nir_op_ior)
      return false;
   for (unsigned s = 0; s < 2; s++) {
      nir_def *x = strip_mov(ior->src[s].src.ssa);
      nir_def *sh = strip_mov(ior->src[1 - s].src.ssa);
      if (!sh || nir_def_instr(sh)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *sa = nir_instr_as_alu(nir_def_instr(sh));
      if ((sa->op == nir_op_ishr || sa->op == nir_op_ushr) &&
          strip_mov(sa->src[0].src.ssa) == x &&
          is_const_u32(sa->src[1].src.ssa, shift)) {
         *in_out = x;
         return true;
      }
      /* nir_opt_algebraic may lower ishr/ushr(x, 16) to extract_i16/u16(x, 1)
       * or i2i32/u2u32(unpack_32_2x16_split_y(x)). */
      if (shift == 16 &&
          (sa->op == nir_op_extract_i16 || sa->op == nir_op_extract_u16) &&
          strip_mov(sa->src[0].src.ssa) == x &&
          is_const_u32(sa->src[1].src.ssa, 1)) {
         *in_out = x;
         return true;
      }
      if (shift == 16 && (sa->op == nir_op_i2i32 || sa->op == nir_op_u2u32)) {
         nir_def *unp = strip_mov(sa->src[0].src.ssa);
         if (unp && nir_def_instr(unp)->type == nir_instr_type_alu &&
             nir_instr_as_alu(nir_def_instr(unp))->op ==
                nir_op_unpack_32_2x16_split_y &&
             strip_mov(nir_instr_as_alu(nir_def_instr(unp))->src[0].src.ssa) == x) {
            *in_out = x;
            return true;
         }
      }
   }
   return false;
}

static bool
match_next_pow2(nir_def *p, nir_def *n)
{
   p = strip_mov(p);
   if (!p || nir_def_instr(p)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *add1 = nir_instr_as_alu(nir_def_instr(p));
   if (add1->op != nir_op_iadd)
      return false;
   nir_def *cur = NULL;
   for (unsigned s = 0; s < 2; s++) {
      if (is_const_u32(add1->src[s].src.ssa, 1)) {
         cur = strip_mov(add1->src[1 - s].src.ssa);
         break;
      }
   }
   if (!cur)
      return false;

   static const unsigned shifts[5] = {16, 8, 4, 2, 1};
   for (unsigned i = 0; i < 5; i++) {
      nir_def *prev = NULL;
      if (!match_bit_smear_step(cur, shifts[i], &prev))
         return false;
      cur = prev;
   }

   /* cur must be `n - 1` (iadd(n, -1) or isub(n, 1)). */
   if (!cur || nir_def_instr(cur)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *sub1 = nir_instr_as_alu(nir_def_instr(cur));
   if (sub1->op == nir_op_iadd)
      return match_iadd_const(cur, n, 0xffffffffu);
   if (sub1->op == nir_op_isub)
      return strip_mov(sub1->src[0].src.ssa) == strip_mov(n) &&
             is_const_u32(sub1->src[1].src.ssa, 1);
   return false;
}

/* Match condition `idx != sentinel` (returns `neg=true` if `idx == sentinel`). */
static bool
match_ne_const(nir_def *cond, nir_def *idx, uint32_t cval, bool *is_eq_out)
{
   bool neg = false;
   cond = strip_mov(cond);
   while (cond && nir_def_instr(cond)->type == nir_instr_type_alu &&
          nir_instr_as_alu(nir_def_instr(cond))->op == nir_op_inot) {
      neg = !neg;
      cond = strip_mov(nir_instr_as_alu(nir_def_instr(cond))->src[0].src.ssa);
   }
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(cond));
   if (alu->op != nir_op_ine && alu->op != nir_op_ieq)
      return false;
   for (unsigned s = 0; s < 2; s++) {
      if (strip_mov(alu->src[s].src.ssa) == strip_mov(idx) &&
          is_const_u32(alu->src[1 - s].src.ssa, cval)) {
         *is_eq_out = (alu->op == nir_op_ieq) != neg;
         return true;
      }
   }
   return false;
}

/* Verify that `def_a` and `def_b` are structurally isomorphic pure DAGs of
 * loop-invariant leaves (dominating `init_block`) parameterized solely by
 * `idx_a` vs `idx_b`. */
static bool
exprs_isomorphic_in_idx(nir_def *da, nir_def *db, nir_def *idx_a,
                        nir_def *idx_b, nir_block *init_block, unsigned depth)
{
   if (depth > 16)
      return false;
   da = strip_mov(da);
   db = strip_mov(db);
   if (!da || !db)
      return false;
   if (da == idx_a && db == idx_b)
      return true;

   nir_instr *ia = nir_def_instr(da);
   nir_instr *ib = nir_def_instr(db);
   if (ia->type != ib->type)
      return false;

   if (ia->type == nir_instr_type_load_const) {
      if (da == db)
         return true;
      nir_load_const_instr *ca = nir_instr_as_load_const(ia);
      nir_load_const_instr *cb = nir_instr_as_load_const(ib);
      if (ca->def.bit_size != cb->def.bit_size ||
          ca->def.num_components != cb->def.num_components)
         return false;
      for (unsigned c = 0; c < ca->def.num_components; c++) {
         if (ca->value[c].u64 != cb->value[c].u64)
            return false;
      }
      return true;
   }

   if (da == db) {
      /* Must be invariant and dominate `init_block` so cloning is valid. */
      return nir_block_dominates(ia->block, init_block);
   }

   if (ia->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *ina = nir_instr_as_intrinsic(ia);
      nir_intrinsic_instr *inb = nir_instr_as_intrinsic(ib);
      if (ina->intrinsic == nir_intrinsic_load_global_constant &&
          inb->intrinsic == nir_intrinsic_load_global_constant &&
          ina->def.num_components == inb->def.num_components &&
          ina->def.bit_size == inb->def.bit_size &&
          nir_intrinsic_align_mul(ina) == nir_intrinsic_align_mul(inb) &&
          nir_intrinsic_align_offset(ina) == nir_intrinsic_align_offset(inb) &&
          nir_intrinsic_access(ina) == nir_intrinsic_access(inb)) {
         return exprs_isomorphic_in_idx(ina->src[0].ssa, inb->src[0].ssa,
                                        idx_a, idx_b, init_block, depth + 1);
      }
      return false;
   }

   if (ia->type == nir_instr_type_alu) {
      nir_alu_instr *aa = nir_instr_as_alu(ia);
      nir_alu_instr *ab = nir_instr_as_alu(ib);
      if (aa->op != ab->op || aa->def.bit_size != ab->def.bit_size ||
          aa->def.num_components != ab->def.num_components)
         return false;
      const nir_op_info *info = &nir_op_infos[aa->op];
      bool direct = true;
      for (unsigned s = 0; s < info->num_inputs; s++) {
         if (memcmp(aa->src[s].swizzle, ab->src[s].swizzle,
                    aa->def.num_components) != 0 ||
             !exprs_isomorphic_in_idx(aa->src[s].src.ssa, ab->src[s].src.ssa,
                                      idx_a, idx_b, init_block, depth + 1)) {
            direct = false;
            break;
         }
      }
      if (direct)
         return true;
      if ((info->algebraic_properties & NIR_OP_IS_2SRC_COMMUTATIVE) &&
          info->num_inputs == 2 &&
          memcmp(aa->src[0].swizzle, ab->src[1].swizzle,
                 aa->def.num_components) == 0 &&
          memcmp(aa->src[1].swizzle, ab->src[0].swizzle,
                 aa->def.num_components) == 0) {
         return exprs_isomorphic_in_idx(aa->src[0].src.ssa, ab->src[1].src.ssa,
                                        idx_a, idx_b, init_block, depth + 1) &&
                exprs_isomorphic_in_idx(aa->src[1].src.ssa, ab->src[0].src.ssa,
                                        idx_a, idx_b, init_block, depth + 1);
      }
   }

   return false;
}

/* Clone pure address DAG `da` replacing `idx_a` with `new_idx`. */
static nir_def *
clone_expr_subst_idx(nir_builder *b, nir_def *da, nir_def *idx_a,
                     nir_def *new_idx, nir_block *init_block)
{
   da = strip_mov(da);
   if (da == idx_a)
      return new_idx;

   nir_instr *ia = nir_def_instr(da);
   if (ia->type == nir_instr_type_load_const) {
      nir_load_const_instr *c = nir_instr_as_load_const(ia);
      return nir_build_imm(b, c->def.num_components, c->def.bit_size, c->value);
   }
   if (nir_block_dominates(ia->block, init_block))
      return da;

   if (ia->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *ina = nir_instr_as_intrinsic(ia);
      assert(ina->intrinsic == nir_intrinsic_load_global_constant);
      nir_def *naddr = clone_expr_subst_idx(b, ina->src[0].ssa, idx_a, new_idx,
                                            init_block);
      return nir_load_global_constant(
         b, ina->def.num_components, ina->def.bit_size, naddr,
         .align_mul = nir_intrinsic_align_mul(ina),
         .align_offset = nir_intrinsic_align_offset(ina),
         .access = nir_intrinsic_access(ina));
   }

   assert(ia->type == nir_instr_type_alu);
   nir_alu_instr *aa = nir_instr_as_alu(ia);
   const nir_op_info *info = &nir_op_infos[aa->op];
   nir_alu_instr *na = nir_alu_instr_create(b->shader, aa->op);
   nir_def_init(&na->instr, &na->def, aa->def.num_components, aa->def.bit_size);
   na->fp_math_ctrl = aa->fp_math_ctrl;
   for (unsigned s = 0; s < info->num_inputs; s++) {
      nir_def *ns = clone_expr_subst_idx(b, aa->src[s].src.ssa, idx_a, new_idx,
                                         init_block);
      na->src[s].src = nir_src_for_ssa(ns);
      memcpy(na->src[s].swizzle, aa->src[s].swizzle, sizeof(aa->src[s].swizzle));
   }
   nir_builder_instr_insert(b, &na->instr);
   return &na->def;
}

/* Match `val = (idx != sentinel) ? load_global(addr(idx)) : default_const`
 * represented as a phi after an `if` (or nested inside an outer `if (a!=s || b!=s)`). */
static bool
match_guarded_key_phi(nir_def *key_def, nir_def *idx, uint32_t sentinel,
                      nir_intrinsic_instr **load_out, uint32_t *default_val_out)
{
   key_def = strip_mov(key_def);
   if (!key_def || nir_def_instr(key_def)->type != nir_instr_type_phi)
      return false;

   nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(key_def));
   nir_cf_node *prev_cf = nir_cf_node_prev(&phi->instr.block->cf_node);
   if (!prev_cf || prev_cf->type != nir_cf_node_if)
      return false;
   nir_if *nif = nir_cf_node_as_if(prev_cf);

   nir_def *then_v = NULL, *else_v = NULL;
   unsigned pcnt = 0;
   nir_foreach_phi_src(ps, phi) {
      pcnt++;
      if (ps->pred == nir_if_last_then_block(nif))
         then_v = strip_mov(ps->src.ssa);
      else if (ps->pred == nir_if_last_else_block(nif))
         else_v = strip_mov(ps->src.ssa);
   }
   if (pcnt != 2 || !then_v || !else_v)
      return false;

   /* Case 1: direct `if (idx != sentinel) { load } else { const }`. */
   bool is_eq = false;
   if (match_ne_const(nif->condition.ssa, idx, sentinel, &is_eq)) {
      nir_def *load_v = is_eq ? else_v : then_v;
      nir_def *def_v = is_eq ? then_v : else_v;
      if (!nir_src_is_const(nir_src_for_ssa(def_v)) ||
          nir_def_instr(load_v)->type != nir_instr_type_intrinsic)
         return false;
      nir_intrinsic_instr *ld = nir_instr_as_intrinsic(nir_def_instr(load_v));
      if ((ld->intrinsic != nir_intrinsic_load_global &&
           ld->intrinsic != nir_intrinsic_load_global_constant) ||
          ld->def.bit_size != 32 || ld->def.num_components != 1 ||
          ld->src[0].ssa->bit_size != 64 ||
          (nir_intrinsic_access(ld) & (ACCESS_VOLATILE | ACCESS_ATOMIC)))
         return false;
      if (ld->intrinsic == nir_intrinsic_load_global &&
          !(nir_intrinsic_access(ld) & ACCESS_NON_WRITEABLE))
         return false;
      *load_out = ld;
      *default_val_out = nir_src_as_uint(nir_src_for_ssa(def_v));
      return true;
   }

   /* Case 2: outer `if (a != s || b != s)` wrapping an inner `if (idx != s)`
    * (with the inactive arm yielding `default_const`). */
   nir_def *inner_v = NULL, *outer_const = NULL;
   if (nir_src_is_const(nir_src_for_ssa(else_v))) {
      inner_v = then_v;
      outer_const = else_v;
   } else if (nir_src_is_const(nir_src_for_ssa(then_v))) {
      inner_v = else_v;
      outer_const = then_v;
   } else {
      return false;
   }

   uint32_t inner_def = 0;
   if (!match_guarded_key_phi(inner_v, idx, sentinel, load_out, &inner_def))
      return false;
   if (nir_src_as_uint(nir_src_for_ssa(outer_const)) != inner_def)
      return false;
   *default_val_out = inner_def;
   return true;
}

/* Match `((i & k) == 0)` (`eq_zero=true`) or `((i & k) != 0)` / `((i & k) == k)`
 * (`eq_zero=false`). */
static bool
match_bit_test(nir_def *cond, nir_def *i, nir_def *k, bool *eq_zero_out)
{
   bool neg = false;
   cond = strip_mov(cond);
   while (cond && nir_def_instr(cond)->type == nir_instr_type_alu &&
          nir_instr_as_alu(nir_def_instr(cond))->op == nir_op_inot) {
      neg = !neg;
      cond = strip_mov(nir_instr_as_alu(nir_def_instr(cond))->src[0].src.ssa);
   }
   if (!cond || nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *cmp = nir_instr_as_alu(nir_def_instr(cond));
   if (cmp->op != nir_op_ieq && cmp->op != nir_op_ine)
      return false;

   for (unsigned s = 0; s < 2; s++) {
      nir_def *and_d = strip_mov(cmp->src[s].src.ssa);
      nir_def *rhs = strip_mov(cmp->src[1 - s].src.ssa);
      if (!and_d || nir_def_instr(and_d)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *a = nir_instr_as_alu(nir_def_instr(and_d));
      if (a->op != nir_op_iand)
         continue;
      nir_def *a0 = strip_mov(a->src[0].src.ssa);
      nir_def *a1 = strip_mov(a->src[1].src.ssa);
      if (!((a0 == i && a1 == k) || (a0 == k && a1 == i)))
         continue;
      if (is_const_u32(rhs, 0)) {
         *eq_zero_out = (cmp->op == nir_op_ieq) != neg;
         return true;
      }
      if (rhs == k) {
         *eq_zero_out = (cmp->op == nir_op_ine) != neg;
         return true;
      }
   }
   return false;
}

/* Match `and(a, b)` represented as `iand(a, b)` or `bcsel(a, b, false)`. */
static bool
match_bool_and(nir_def *def, nir_def **a_out, nir_def **b_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_iand) {
      *a_out = strip_mov(alu->src[0].src.ssa);
      *b_out = strip_mov(alu->src[1].src.ssa);
      return true;
   }
   if (alu->op == nir_op_bcsel && nir_src_is_const(alu->src[2].src) &&
       !nir_src_as_bool(alu->src[2].src)) {
      *a_out = strip_mov(alu->src[0].src.ssa);
      *b_out = strip_mov(alu->src[1].src.ssa);
      return true;
   }
   return false;
}

/* Match `or(a, b)` represented as `ior(a, b)` or `bcsel(a, true, b)`. */
static bool
match_bool_or(nir_def *def, nir_def **a_out, nir_def **b_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_ior) {
      *a_out = strip_mov(alu->src[0].src.ssa);
      *b_out = strip_mov(alu->src[1].src.ssa);
      return true;
   }
   if (alu->op == nir_op_bcsel && nir_src_is_const(alu->src[1].src) &&
       nir_src_as_bool(alu->src[1].src)) {
      *a_out = strip_mov(alu->src[0].src.ssa);
      *b_out = strip_mov(alu->src[2].src.ssa);
      return true;
   }
   return false;
}

/* Match `cmp(x, y)` for strict comparison `<` (`flt`/`ilt`/`ult`) and return
 * whether `x == ka && y == kb` (`ka_lt_kb = true`) or `x == kb && y == ka`
 * (`ka_lt_kb = false`). */
static bool
match_key_cmp(nir_def *def, nir_def *ka, nir_def *kb, nir_op *cmp_op_out,
              bool *ka_lt_kb_out)
{
   def = strip_mov(def);
   if (!def || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op != nir_op_flt && alu->op != nir_op_ilt && alu->op != nir_op_ult)
      return false;
   nir_def *s0 = strip_mov(alu->src[0].src.ssa);
   nir_def *s1 = strip_mov(alu->src[1].src.ssa);
   if (s0 == ka && s1 == kb) {
      *cmp_op_out = alu->op;
      *ka_lt_kb_out = true;
      return true;
   }
   if (s0 == kb && s1 == ka) {
      *cmp_op_out = alu->op;
      *ka_lt_kb_out = false;
      return true;
   }
   return false;
}

/* Match bitonic swap predicate:
 * `((i & k) == 0 && cmp(ka, kb)) || ((i & k) != 0 && cmp(kb, ka))`
 * (or the opposite polarity when ascending-when-zero). Sets `swap_if_ka_lt_kb_at_zero`
 * = true when `(i & k) == 0` swaps on `cmp(ka, kb)` (descending), false when
 * `(i & k) == 0` swaps on `cmp(kb, ka)` (ascending). */
static bool
match_bitonic_swap_cond(nir_def *cond, nir_def *i, nir_def *k,
                        nir_def *ka, nir_def *kb, nir_op *cmp_op_out,
                        bool *swap_lt_at_zero_out)
{
   nir_def *t0 = NULL, *t1 = NULL;
   if (!match_bool_or(cond, &t0, &t1))
      return false;

   nir_op op[2] = {nir_num_opcodes, nir_num_opcodes};
   bool eq_zero[2] = {false, false};
   bool ka_lt[2] = {false, false};

   nir_def *ts[2] = {t0, t1};
   for (unsigned idx = 0; idx < 2; idx++) {
      nir_def *a = NULL, *b = NULL;
      if (!match_bool_and(ts[idx], &a, &b))
         return false;
      bool matched = false;
      for (unsigned swap = 0; swap < 2 && !matched; swap++) {
         nir_def *x = swap ? b : a;
         nir_def *y = swap ? a : b;
         if (match_bit_test(x, i, k, &eq_zero[idx]) &&
             match_key_cmp(y, ka, kb, &op[idx], &ka_lt[idx]))
            matched = true;
      }
      if (!matched)
         return false;
   }

   if (op[0] != op[1] || eq_zero[0] == eq_zero[1] || ka_lt[0] == ka_lt[1])
      return false;

   *cmp_op_out = op[0];
   *swap_lt_at_zero_out = eq_zero[0] ? ka_lt[0] : ka_lt[1];
   return true;
}

static bool
is_shared_barrier(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_barrier)
      return false;
   return (nir_intrinsic_memory_modes(intr) & nir_var_mem_shared) &&
          nir_intrinsic_execution_scope(intr) >= SCOPE_SUBGROUP;
}

static bool
block_has_shared_barrier(nir_block *blk)
{
   nir_foreach_instr(instr, blk) {
      if (is_shared_barrier(instr))
         return true;
   }
   return false;
}

static bool
is_global_or_external_write(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_store_global:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_image_store:
   case nir_intrinsic_image_atomic:
   case nir_intrinsic_image_atomic_swap:
   case nir_intrinsic_bindless_image_store:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap:
      return true;
   default:
      return false;
   }
}

/* Emit one in-register subgroup bitonic compare-exchange stage for constant
 * `step` in {1, 2, 4, 8, 16}. Both lanes of every `(lane, lane ^ step)` pair
 * evaluate the exact same comparison direction (`want_lt = (asc ^ is_hi)` when
 * `swap_lt_at_zero`, or `!(asc ^ is_hi)` otherwise) so NaNs and ties behave
 * identically to the original single-lane compare-and-swap. */
static void
emit_subgroup_bitonic_stage(nir_builder *b, nir_def **idx_io, nir_def **key_io,
                            nir_def *is_hi, nir_def *asc_cond,
                            unsigned step, nir_op cmp_op, bool swap_lt_at_zero)
{
   nir_def *my_idx = *idx_io;
   nir_def *my_key = *key_io;
   nir_def *o_idx = nir_shuffle_xor(b, my_idx, nir_imm_int(b, step));
   nir_def *o_key = nir_shuffle_xor(b, my_key, nir_imm_int(b, step));

   nir_def *want_lt = swap_lt_at_zero ? nir_ixor(b, asc_cond, is_hi)
                                      : nir_ieq(b, asc_cond, is_hi);
   nir_def *cmp_mo = nir_build_alu2(b, cmp_op, my_key, o_key);
   nir_def *cmp_om = nir_build_alu2(b, cmp_op, o_key, my_key);
   nir_def *swap = nir_bcsel(b, want_lt, cmp_mo, cmp_om);

   *idx_io = nir_bcsel(b, swap, o_idx, my_idx);
   *key_io = nir_bcsel(b, swap, o_key, my_key);
}

/* Emit one in-register compare-exchange between lower element `a` and higher
 * element `b` (same lane, zero shuffles), bit-exact on ties and NaNs. */
static void
emit_reg_bitonic_pair(nir_builder *b, nir_def **a_idx, nir_def **a_key,
                      nir_def **b_idx, nir_def **b_key, nir_def *asc_cond,
                      nir_op cmp_op, bool swap_lt_at_zero)
{
   nir_def *ia = *a_idx, *ka = *a_key;
   nir_def *ib = *b_idx, *kb = *b_key;
   nir_def *c_ab = nir_build_alu2(b, cmp_op, ka, kb);
   nir_def *c_ba = nir_build_alu2(b, cmp_op, kb, ka);
   nir_def *swap = swap_lt_at_zero ? nir_bcsel(b, asc_cond, c_ab, c_ba)
                                    : nir_bcsel(b, asc_cond, c_ba, c_ab);
   *a_idx = nir_bcsel(b, swap, ib, ia);
   *b_idx = nir_bcsel(b, swap, ia, ib);
   *a_key = nir_bcsel(b, swap, kb, ka);
   *b_key = nir_bcsel(b, swap, ka, kb);
}

static void
emit_wg_shared_barrier(nir_builder *b, mesa_scope scope)
{
   nir_barrier(b, .execution_scope = scope, .memory_scope = scope,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);
}

static void
emit_subgroup_tail_16(nir_builder *b, nir_def **idx_io, nir_def **key_io,
                      nir_def *const is_hi_bits[5], nir_def *asc_cond,
                      nir_op cmp_op, bool swap_lt_at_zero)
{
   for (int bit = 4; bit >= 0; bit--) {
      emit_subgroup_bitonic_stage(b, idx_io, key_io, is_hi_bits[bit], asc_cond,
                                  1u << bit, cmp_op, swap_lt_at_zero);
   }
}

bool
agx_nir_opt_shared_index_sort(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE || nir->info.internal ||
       nir->info.workgroup_size_variable)
      return false;

   const uint32_t wg_x = nir->info.workgroup_size[0];
   if (wg_x < 32 || !util_is_power_of_two_nonzero(wg_x) ||
       nir->info.workgroup_size[1] != 1 || nir->info.workgroup_size[2] != 1)
      return false;

   /* Need room to double shared memory within Apple GPU's 32 KiB limit. */
   const uint32_t orig_shared = nir->info.shared_size;
   if (orig_shared < 8 || orig_shared > 16384 || (orig_shared & 3u) != 0)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_metadata_require(impl, nir_metadata_block_index | nir_metadata_dominance);

   /* Top-level must be a single `if (wg_id.x == 0)` with empty inactive arm
    * and zero side effects outside it, guaranteeing only 1 workgroup runs. */
   nir_if *wg0_if = NULL;
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (node->type == nir_cf_node_loop)
         return false;
      if (node->type == nir_cf_node_if) {
         if (wg0_if)
            return false;
         wg0_if = nir_cf_node_as_if(node);
      } else if (node->type == nir_cf_node_block) {
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            if (is_global_or_external_write(instr))
               return false;
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
               if (op == nir_intrinsic_store_shared ||
                   op == nir_intrinsic_shared_atomic ||
                   op == nir_intrinsic_shared_atomic_swap)
                  return false;
            }
         }
      }
   }
   if (!wg0_if)
      return false;

   bool wg_is_ne = false;
   nir_def *wg_cond = strip_mov(wg0_if->condition.ssa);
   while (wg_cond && nir_def_instr(wg_cond)->type == nir_instr_type_alu &&
          nir_instr_as_alu(nir_def_instr(wg_cond))->op == nir_op_inot) {
      wg_is_ne = !wg_is_ne;
      wg_cond =
         strip_mov(nir_instr_as_alu(nir_def_instr(wg_cond))->src[0].src.ssa);
   }
   if (!wg_cond || nir_def_instr(wg_cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *wg_cmp = nir_instr_as_alu(nir_def_instr(wg_cond));
   if (wg_cmp->op != nir_op_ieq && wg_cmp->op != nir_op_ine)
      return false;
   bool matched_wg0 = false;
   for (unsigned s = 0; s < 2; s++) {
      if (match_wg_id_x_src(&wg_cmp->src[s]) &&
          is_const_u32(wg_cmp->src[1 - s].src.ssa, 0)) {
         matched_wg0 = true;
         break;
      }
   }
   if (!matched_wg0)
      return false;
   const bool active_in_then = (wg_cmp->op == nir_op_ieq) != wg_is_ne;
   struct exec_list *active_list =
      active_in_then ? &wg0_if->then_list : &wg0_if->else_list;
   struct exec_list *inactive_list =
      active_in_then ? &wg0_if->else_list : &wg0_if->then_list;
   if (!cf_list_is_empty(inactive_list))
      return false;

   /* Walk top-level CF nodes inside `if (wg_id.x == 0)` to find:
    *   init_loop -> barrier -> pad_loop -> barrier -> sort_node (if or loop) */
   nir_loop *init_loop = NULL, *pad_loop = NULL;
   nir_cf_node *sort_outer_node = NULL;
   nir_loop *k_loop = NULL;
   bool seen_init_bar = false, seen_pad_bar = false;

   foreach_list_typed(nir_cf_node, node, node, active_list) {
      if (!init_loop) {
         if (node->type == nir_cf_node_loop)
            init_loop = nir_cf_node_as_loop(node);
         else if (node->type != nir_cf_node_block)
            return false;
      } else if (!pad_loop) {
         if (node->type == nir_cf_node_block) {
            if (block_has_shared_barrier(nir_cf_node_as_block(node)))
               seen_init_bar = true;
         } else if (node->type == nir_cf_node_loop && seen_init_bar) {
            pad_loop = nir_cf_node_as_loop(node);
         } else {
            return false;
         }
      } else if (!sort_outer_node) {
         if (node->type == nir_cf_node_block) {
            if (block_has_shared_barrier(nir_cf_node_as_block(node)))
               seen_pad_bar = true;
         } else if (seen_pad_bar && node->type == nir_cf_node_if) {
            sort_outer_node = node;
         } else if (seen_pad_bar && node->type == nir_cf_node_loop) {
            sort_outer_node = node;
            k_loop = nir_cf_node_as_loop(node);
         } else {
            return false;
         }
      } else {
         break;
      }
   }
   if (!init_loop || !pad_loop || !sort_outer_node)
      return false;

   /* Match init_loop: `for (i = lid; i < N; i += WG) shared[i] = i;` */
   nir_phi_instr *init_iv = NULL;
   nir_def *init_start = NULL, *n_def = NULL;
   nir_intrinsic_instr *init_store = NULL;
   if (!match_simple_shared_fill_loop(init_loop, wg_x, &init_iv, &init_start,
                                      &n_def, &init_store))
      return false;
   if (!match_lid_x(init_start) ||
       strip_mov(init_store->src[0].ssa) != &init_iv->def)
      return false;

   /* Match pad_loop: `for (i = N + lid; i < P; i += WG) shared[i] = sentinel;` */
   nir_phi_instr *pad_iv = NULL;
   nir_def *pad_start = NULL, *p_def = NULL;
   nir_intrinsic_instr *pad_store = NULL;
   if (!match_simple_shared_fill_loop(pad_loop, wg_x, &pad_iv, &pad_start,
                                      &p_def, &pad_store))
      return false;
   if (!match_add_lid(pad_start, n_def) ||
       !nir_src_is_const(pad_store->src[0]))
      return false;
   const uint32_t sentinel = nir_src_as_uint(pad_store->src[0]);

   /* Prove P is a power of two (next_pow2(N)). */
   if (!match_next_pow2(p_def, n_def))
      return false;

   /* Unwrap optional `if (P >= 2)` guard around `k_loop`. */
   nir_if *p_ge2_if = NULL;
   if (!k_loop) {
      p_ge2_if = nir_cf_node_as_if(sort_outer_node);
      bool gneg = false;
      nir_def *gbound = NULL;
      if (!cf_list_is_empty(&p_ge2_if->else_list) ||
          !match_iv_ge_bound(p_ge2_if->condition.ssa, gneg, p_def, &gbound) ||
          !is_const_u32(gbound, 2))
         return false;
      foreach_list_typed(nir_cf_node, n, node, &p_ge2_if->then_list) {
         if (n->type == nir_cf_node_loop) {
            if (k_loop)
               return false;
            k_loop = nir_cf_node_as_loop(n);
         } else if (n->type != nir_cf_node_block) {
            return false;
         }
      }
      if (!k_loop)
         return false;
   }

   /* Match outer `k` loop: `k = 2; while (k <= P) { ... k <<= 1; }`. */
   nir_phi_instr *k_phi = NULL;
   nir_def *k_init = NULL, *k_next = NULL;
   if (!match_single_loop_phi(k_loop, &k_phi, &k_init, &k_next))
      return false;
   if (!is_const_u32(k_init, 2))
      return false;
   /* k_next must be `k << 1` or `k + k`. */
   if (!k_next || nir_def_instr(k_next)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *k_alu = nir_instr_as_alu(nir_def_instr(k_next));
   bool k_doubles =
      (k_alu->op == nir_op_ishl &&
       strip_mov(k_alu->src[0].src.ssa) == &k_phi->def &&
       is_const_u32(k_alu->src[1].src.ssa, 1)) ||
      (k_alu->op == nir_op_iadd &&
       strip_mov(k_alu->src[0].src.ssa) == &k_phi->def &&
       strip_mov(k_alu->src[1].src.ssa) == &k_phi->def);
   if (!k_doubles)
      return false;

   /* Find `j_loop` and `k` break inside `k_loop` (allowing optional `if (k>>1 != 0)`). */
   nir_loop *j_loop = NULL;
   bool k_break_ok = false;
   foreach_list_typed(nir_cf_node, n, node, &k_loop->body) {
      if (n->type == nir_cf_node_loop) {
         if (j_loop)
            return false;
         j_loop = nir_cf_node_as_loop(n);
      } else if (n->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(n);
         nir_def *bc = NULL;
         bool bn = false;
         if (match_break_if(nif, &bc, &bn)) {
            if (match_iv_gt_bound(bc, bn, &k_phi->def, p_def) ||
                match_iv_gt_bound(bc, bn, k_next, p_def)) {
               k_break_ok = true;
               continue;
            }
            return false;
         }
         /* Optional guard `if ((k >> 1) != 0) { j_loop }`. */
         if (!j_loop && cf_list_is_empty(&nif->else_list)) {
            foreach_list_typed(nir_cf_node, inner, node, &nif->then_list) {
               if (inner->type == nir_cf_node_loop) {
                  if (j_loop)
                     return false;
                  j_loop = nir_cf_node_as_loop(inner);
               } else if (inner->type != nir_cf_node_block) {
                  return false;
               }
            }
            if (!j_loop)
               return false;
         } else {
            return false;
         }
      }
   }
   if (!j_loop || !k_break_ok)
      return false;

   /* Match middle `j` loop: `j = k >> 1; while (j != 0) { i_loop; barrier; j >>= 1; }`. */
   nir_phi_instr *j_phi = NULL;
   nir_def *j_init = NULL, *j_next = NULL;
   if (!match_single_loop_phi(j_loop, &j_phi, &j_init, &j_next))
      return false;
   if (!j_init || nir_def_instr(j_init)->type != nir_instr_type_alu ||
       !j_next || nir_def_instr(j_next)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *ji = nir_instr_as_alu(nir_def_instr(j_init));
   nir_alu_instr *jn = nir_instr_as_alu(nir_def_instr(j_next));
   if ((ji->op != nir_op_ushr && ji->op != nir_op_ishr) ||
       strip_mov(ji->src[0].src.ssa) != &k_phi->def ||
       !is_const_u32(ji->src[1].src.ssa, 1))
      return false;
   if ((jn->op != nir_op_ushr && jn->op != nir_op_ishr) ||
       strip_mov(jn->src[0].src.ssa) != &j_phi->def ||
       !is_const_u32(jn->src[1].src.ssa, 1))
      return false;

   nir_loop *i_loop = NULL;
   bool j_break_ok = false, j_barrier_ok = false;
   foreach_list_typed(nir_cf_node, n, node, &j_loop->body) {
      if (n->type == nir_cf_node_loop) {
         if (i_loop)
            return false;
         i_loop = nir_cf_node_as_loop(n);
      } else if (n->type == nir_cf_node_block) {
         if (i_loop && block_has_shared_barrier(nir_cf_node_as_block(n)))
            j_barrier_ok = true;
      } else if (n->type == nir_cf_node_if) {
         nir_def *bc = NULL;
         bool bn = false;
         if (!match_break_if(nir_cf_node_as_if(n), &bc, &bn))
            return false;
         bool is_eq0 = false;
         if ((match_ne_const(bc, &j_phi->def, 0, &is_eq0) && (is_eq0 != bn)) ||
             (match_ne_const(bc, j_next, 0, &is_eq0) && (is_eq0 != bn))) {
            j_break_ok = true;
         } else {
            return false;
         }
      }
   }
   if (!i_loop || !j_break_ok || !j_barrier_ok)
      return false;

   /* Match inner `i` loop: `for (i = lid; i < P; i += WG)`. */
   nir_phi_instr *i_phi = NULL;
   nir_def *i_init = NULL, *i_next = NULL;
   if (!match_single_loop_phi(i_loop, &i_phi, &i_init, &i_next))
      return false;
   if (!match_lid_x(i_init) || !match_iadd_const(i_next, &i_phi->def, wg_x))
      return false;

   nir_block *i_hdr = nir_loop_first_block(i_loop);
   nir_cf_node *i_n1 = nir_cf_node_next(&i_hdr->cf_node);
   if (!i_n1 || i_n1->type != nir_cf_node_if)
      return false;
   nir_def *ibc = NULL;
   bool ibn = false;
   nir_def *i_bound = NULL;
   if (!match_break_if(nir_cf_node_as_if(i_n1), &ibc, &ibn) ||
       !match_iv_ge_bound(ibc, ibn, &i_phi->def, &i_bound) ||
       i_bound != p_def)
      return false;

   /* Find the single `if (i < (i ^ j))` inside `i_loop`. */
   nir_if *pair_if = NULL;
   for (nir_cf_node *n = nir_cf_node_next(i_n1); n; n = nir_cf_node_next(n)) {
      if (n->type == nir_cf_node_if) {
         if (pair_if)
            return false;
         pair_if = nir_cf_node_as_if(n);
      } else if (n->type != nir_cf_node_block) {
         return false;
      }
   }
   if (!pair_if || !cf_list_is_empty(&pair_if->else_list))
      return false;

   /* Verify pair_if condition is `ult(i, ixj)` or `ilt(i, ixj)` where `ixj = i ^ j`. */
   nir_def *pcond = strip_mov(pair_if->condition.ssa);
   if (!pcond || nir_def_instr(pcond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *pcmp = nir_instr_as_alu(nir_def_instr(pcond));
   if (pcmp->op != nir_op_ult && pcmp->op != nir_op_ilt)
      return false;
   if (strip_mov(pcmp->src[0].src.ssa) != &i_phi->def)
      return false;
   nir_def *ixj_def = strip_mov(pcmp->src[1].src.ssa);
   if (!ixj_def || nir_def_instr(ixj_def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *xalu = nir_instr_as_alu(nir_def_instr(ixj_def));
   if (xalu->op != nir_op_ixor)
      return false;
   nir_def *x0 = strip_mov(xalu->src[0].src.ssa);
   nir_def *x1 = strip_mov(xalu->src[1].src.ssa);
   if (!((x0 == &i_phi->def && x1 == &j_phi->def) ||
         (x0 == &j_phi->def && x1 == &i_phi->def)))
      return false;

   /* Inside `pair_if->then_list`: find `a = load_shared(i<<2)`,
    * `b = load_shared(ixj<<2)`, and the final swap `if (swap_cond) { store b, store a }`. */
   nir_def *a_def = NULL, *b_def = NULL;
   nir_block *pair_entry = nir_if_first_then_block(pair_if);
   nir_foreach_instr(instr, pair_entry) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_load_shared ||
          nir_intrinsic_base(intr) != 0 || intr->def.bit_size != 32 ||
          intr->def.num_components != 1)
         return false;
      nir_def *lidx = NULL;
      if (!match_u32_byte_offset(intr->src[0].ssa, &lidx))
         return false;
      if (lidx == &i_phi->def)
         a_def = &intr->def;
      else if (lidx == ixj_def)
         b_def = &intr->def;
      else
         return false;
   }
   if (!a_def || !b_def)
      return false;

   /* Find the last `if` inside `pair_if->then_list` (the swap `if`). */
   nir_if *swap_if = NULL;
   foreach_list_typed(nir_cf_node, n, node, &pair_if->then_list) {
      if (n->type == nir_cf_node_if)
         swap_if = nir_cf_node_as_if(n);
      else if (n->type == nir_cf_node_loop)
         return false;
   }
   if (!swap_if || !cf_list_is_empty(&swap_if->else_list) ||
       !exec_list_is_singular(&swap_if->then_list))
      return false;

   nir_intrinsic_instr *swap_st_i = NULL, *swap_st_ixj = NULL;
   nir_foreach_instr(instr, nir_if_first_then_block(swap_if)) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_store_shared ||
          nir_intrinsic_base(intr) != 0 ||
          nir_src_bit_size(intr->src[0]) != 32 ||
          nir_src_num_components(intr->src[0]) != 1)
         return false;
      nir_def *sidx = NULL;
      if (!match_u32_byte_offset(intr->src[1].ssa, &sidx))
         return false;
      if (sidx == &i_phi->def && strip_mov(intr->src[0].ssa) == b_def)
         swap_st_i = intr;
      else if (sidx == ixj_def && strip_mov(intr->src[0].ssa) == a_def)
         swap_st_ixj = intr;
      else
         return false;
   }
   if (!swap_st_i || !swap_st_ixj)
      return false;

   /* Extract `ka` and `kb` from `swap_if->condition`! */
   nir_def *ka_def = NULL, *kb_def = NULL;
   nir_def *or0 = NULL, *or1 = NULL;
   if (!match_bool_or(swap_if->condition.ssa, &or0, &or1))
      return false;
   nir_def *and0 = NULL, *and1 = NULL;
   if (!match_bool_and(or0, &and0, &and1))
      return false;
   nir_def *cmp_d = NULL;
   bool dummy_ez = false;
   if (match_bit_test(and0, &i_phi->def, &k_phi->def, &dummy_ez))
      cmp_d = and1;
   else if (match_bit_test(and1, &i_phi->def, &k_phi->def, &dummy_ez))
      cmp_d = and0;
   else
      return false;
   cmp_d = strip_mov(cmp_d);
   if (!cmp_d || nir_def_instr(cmp_d)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *first_cmp = nir_instr_as_alu(nir_def_instr(cmp_d));
   nir_def *cand0 = strip_mov(first_cmp->src[0].src.ssa);
   nir_def *cand1 = strip_mov(first_cmp->src[1].src.ssa);

   nir_intrinsic_instr *load_a = NULL, *load_b = NULL;
   uint32_t def_a = 0, def_b = 0;
   if (match_guarded_key_phi(cand0, a_def, sentinel, &load_a, &def_a) &&
       match_guarded_key_phi(cand1, b_def, sentinel, &load_b, &def_b)) {
      ka_def = cand0;
      kb_def = cand1;
   } else if (match_guarded_key_phi(cand1, a_def, sentinel, &load_a, &def_a) &&
              match_guarded_key_phi(cand0, b_def, sentinel, &load_b, &def_b)) {
      ka_def = cand1;
      kb_def = cand0;
   } else {
      return false;
   }

   if (def_a != def_b || load_a->intrinsic != load_b->intrinsic)
      return false;

   nir_op cmp_op = nir_num_opcodes;
   bool swap_lt_at_zero = false;
   if (!match_bitonic_swap_cond(swap_if->condition.ssa, &i_phi->def,
                                &k_phi->def, ka_def, kb_def, &cmp_op,
                                &swap_lt_at_zero))
      return false;

   /* Verify both global load addresses are isomorphic pure DAGs of `a_def` vs
    * `b_def` with all invariant leaves dominating `init_loop`! */
   nir_block *init_body =
      nir_cf_node_as_block(nir_cf_node_next(nir_cf_node_next(&nir_loop_first_block(init_loop)->cf_node)));
   if (!exprs_isomorphic_in_idx(load_a->src[0].ssa, load_b->src[0].ssa,
                                a_def, b_def, init_body, 0))
      return false;

   /* Global soundness audit across the ENTIRE shader:
    * 1. No global/external writes before `sort_outer_node` completes.
    * 2. No other `store_shared` or `shared_atomic` anywhere in the shader.
    * 3. No unexpected side-effecting intrinsics inside `sort_outer_node`.
    * 4. Every `load_shared` outside `sort_outer_node` must be a scalar u32
    *    indexed load (`idx * 4`, `base == 0`) so interleaved `u32x2` layout
    *    can remap it cleanly to `idx << 3`. */
   nir_block *after_sort_blk =
      nir_cf_node_as_block(nir_cf_node_next(sort_outer_node));
   const unsigned sort_end_idx = after_sort_blk->index;
   const unsigned sort_start_idx =
      nir_cf_node_as_block(nir_cf_node_prev(sort_outer_node))->index;

   nir_intrinsic_instr *ext_loads[32];
   nir_def *ext_load_idxs[32];
   unsigned num_ext_loads = 0;

   nir_foreach_block(blk, impl) {
      nir_foreach_instr(instr, blk) {
         if (is_global_or_external_write(instr) && blk->index < sort_end_idx)
            return false;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_shared_atomic ||
             intr->intrinsic == nir_intrinsic_shared_atomic_swap)
            return false;
         if (intr->intrinsic == nir_intrinsic_store_shared) {
            if (intr != init_store && intr != pad_store &&
                intr != swap_st_i && intr != swap_st_ixj)
               return false;
         }
         if (blk->index > sort_start_idx && blk->index < sort_end_idx) {
            switch (intr->intrinsic) {
            case nir_intrinsic_load_shared:
            case nir_intrinsic_store_shared:
            case nir_intrinsic_load_global:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_barrier:
            case nir_intrinsic_load_local_invocation_id:
            case nir_intrinsic_load_local_invocation_index:
            case nir_intrinsic_load_workgroup_id:
            case nir_intrinsic_load_base_workgroup_id:
            case nir_intrinsic_load_preamble:
               break;
            default:
               return false;
            }
         } else if (intr->intrinsic == nir_intrinsic_load_shared) {
            nir_def *lidx = NULL;
            if (nir_intrinsic_base(intr) != 0 || intr->def.bit_size != 32 ||
                intr->def.num_components != 1 ||
                !match_u32_byte_offset(intr->src[0].ssa, &lidx) ||
                num_ext_loads >= ARRAY_SIZE(ext_loads))
               return false;
            ext_loads[num_ext_loads] = intr;
            ext_load_idxs[num_ext_loads] = lidx;
            num_ext_loads++;
         }
      }
   }

   /* All checks passed! Apply interleaved `u32x2 (idx, key)` shadow-cache +
    * 4-element-per-lane (`k <= 128`, `j <= 64`) register bitonic sort. */
   nir->info.shared_size = orig_shared + orig_shared;
   const mesa_scope wg_scope = wg_x > 32 ? SCOPE_WORKGROUP : SCOPE_SUBGROUP;

   /* 1. Rewrite `init_loop`: store interleaved `u32x2(i, key)` at `i << 3`. */
   nir_builder b = nir_builder_at(nir_after_instr(&init_store->instr));
   nir_variable *ik_var =
      nir_local_variable_create(impl, glsl_uint_type(), "bs_ik");
   nir_store_var(&b, ik_var, nir_imm_int(&b, 0), 0x1);
   nir_push_if(&b, nir_ige_imm(&b, n_def, 2));
   {
      nir_def *init_addr =
         clone_expr_subst_idx(&b, load_a->src[0].ssa, a_def, &init_iv->def,
                              init_body);
      nir_def *init_key;
      if (load_a->intrinsic == nir_intrinsic_load_global_constant) {
         init_key = nir_load_global_constant(
            &b, 1, 32, init_addr,
            .align_mul = nir_intrinsic_align_mul(load_a),
            .align_offset = nir_intrinsic_align_offset(load_a));
      } else {
         init_key = nir_load_global(
            &b, 1, 32, init_addr,
            .access = nir_intrinsic_access(load_a),
            .align_mul = nir_intrinsic_align_mul(load_a),
            .align_offset = nir_intrinsic_align_offset(load_a));
      }
      if ((int32_t)sentinel >= 0) {
         init_key = nir_bcsel(&b, nir_ieq_imm(&b, &init_iv->def, sentinel),
                              nir_imm_int(&b, def_a), init_key);
      }
      nir_store_var(&b, ik_var, init_key, 0x1);
   }
   nir_pop_if(&b, NULL);
   nir_def *init_pair = nir_vec2(&b, &init_iv->def, nir_load_var(&b, ik_var));
   nir_store_shared(&b, init_pair, nir_ishl_imm(&b, &init_iv->def, 3),
                    .base = 0, .write_mask = 0x3, .align_mul = 8);
   nir_instr_remove(&init_store->instr);

   /* 2. Rewrite `pad_loop`: store interleaved `u32x2(sentinel, def_a)` at `i << 3`. */
   b.cursor = nir_after_instr(&pad_store->instr);
   nir_def *pad_pair =
      nir_vec2(&b, nir_imm_int(&b, sentinel), nir_imm_int(&b, def_a));
   nir_store_shared(&b, pad_pair, nir_ishl_imm(&b, &pad_iv->def, 3),
                    .base = 0, .write_mask = 0x3, .align_mul = 8);
   nir_instr_remove(&pad_store->instr);

   /* 3. Remap post-sort `load_shared(idx * 4)` to `load_shared(idx << 3)`. */
   for (unsigned u = 0; u < num_ext_loads; u++) {
      b.cursor = nir_before_instr(&ext_loads[u]->instr);
      nir_src_rewrite(&ext_loads[u]->src[0],
                      nir_ishl_imm(&b, ext_load_idxs[u], 3));
   }

   /* 4. Replace `sort_outer_node` with interleaved `u32x2` 4-elem/lane
    * register bitonic sort (`P >= 128`) plus small-`P` (`2..64`) path. */
   b.cursor = nir_before_cf_node(sort_outer_node);
   nir_def *lid = init_start;
   nir_def *lane = nir_iand_imm(&b, lid, 31);
   nir_def *sg_id = nir_ushr_imm(&b, lid, 5);
   nir_def *is_hi_bits[5];
   for (unsigned bit = 0; bit < 5; bit++)
      is_hi_bits[bit] = nir_ine_imm(&b, nir_iand_imm(&b, lane, 1u << bit), 0);
   nir_def *asc_bits[4];
   for (unsigned bit = 1; bit <= 4; bit++)
      asc_bits[bit - 1] = nir_ieq_imm(&b, nir_iand_imm(&b, lane, 1u << bit), 0);

   nir_push_if(&b, nir_uge_imm(&b, p_def, 128));
   {
      /* Fast path: P >= 128 (always a multiple of 128). Each subgroup sorts
       * 128 contiguous elements (4 per lane) in registers for k = 2..128 and
       * for every j = 64..1 tail of k = 256..P! */
      const uint32_t blk_stride = wg_x << 2; /* (wg_x / 32) * 128 */
      nir_def *sg_base0 = nir_ishl_imm(&b, sg_id, 7);

      /* Phase 1: k = 2..128 per 128-element subgroup block, 1 workgroup barrier. */
      nir_variable *blk_var =
         nir_local_variable_create(impl, glsl_uint_type(), "bs_blk");
      nir_store_var(&b, blk_var, sg_base0, 0x1);
      nir_loop *p1_blk_loop = nir_push_loop(&b);
      p1_blk_loop->control = nir_loop_control_dont_unroll;
      {
         nir_def *blk = nir_load_var(&b, blk_var);
         nir_push_if(&b, nir_uge(&b, blk, p_def));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, NULL);

         /* 1a. Sort the 4 32-element chunks (k = 2, 4, 8, 16, 32). Each lane
          * writes back to its own private 4 slots; no barrier needed before 1b. */
         nir_variable *m_var =
            nir_local_variable_create(impl, glsl_uint_type(), "bs_m");
         nir_store_var(&b, m_var, nir_imm_int(&b, 0), 0x1);
         nir_loop *m_loop = nir_push_loop(&b);
         m_loop->control = nir_loop_control_dont_unroll;
         {
            nir_def *mv = nir_load_var(&b, m_var);
            nir_push_if(&b, nir_uge_imm(&b, mv, 4));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, NULL);

            nir_def *idx =
               nir_iadd(&b, nir_iadd(&b, blk, nir_ishl_imm(&b, mv, 5)), lane);
            nir_def *off = nir_ishl_imm(&b, idx, 3);
            nir_def *v =
               nir_load_shared(&b, 2, 32, off, .base = 0, .align_mul = 8);
            nir_def *ci = nir_channel(&b, v, 0);
            nir_def *ck = nir_channel(&b, v, 1);

            /* k = 2, 4, 8, 16 */
            for (unsigned k_bit = 1; k_bit <= 4; k_bit++) {
               for (int j_bit = (int)k_bit - 1; j_bit >= 0; j_bit--) {
                  emit_subgroup_bitonic_stage(&b, &ci, &ck, is_hi_bits[j_bit],
                                              asc_bits[k_bit - 1], 1u << j_bit,
                                              cmp_op, swap_lt_at_zero);
               }
            }
            /* k = 32: asc = ((m & 1) == 0) */
            nir_def *asc32 = nir_ieq_imm(&b, nir_iand_imm(&b, mv, 1), 0);
            emit_subgroup_tail_16(&b, &ci, &ck, is_hi_bits, asc32, cmp_op,
                                  swap_lt_at_zero);

            nir_store_shared(&b, nir_vec2(&b, ci, ck), off, .base = 0,
                             .write_mask = 0x3, .align_mul = 8);
            nir_store_var(&b, m_var, nir_iadd_imm(&b, mv, 1), 0x1);
         }
         nir_pop_loop(&b, m_loop);

         /* 1b. Merge the 4 32-element chunks into a sorted 128-element block
          * (k = 64 and k = 128) purely in registers! */
         nir_def *base_lane = nir_iadd(&b, blk, lane);
         nir_def *offs[4], *ei[4], *ek[4];
         for (unsigned m = 0; m < 4; m++) {
            offs[m] = nir_ishl_imm(&b, nir_iadd_imm(&b, base_lane, m * 32), 3);
            nir_def *v =
               nir_load_shared(&b, 2, 32, offs[m], .base = 0, .align_mul = 8);
            ei[m] = nir_channel(&b, v, 0);
            ek[m] = nir_channel(&b, v, 1);
         }

         /* k = 64: (e0,e1) asc=true, (e2,e3) asc=false */
         nir_def *t_true = nir_imm_true(&b);
         nir_def *t_false = nir_imm_false(&b);
         emit_reg_bitonic_pair(&b, &ei[0], &ek[0], &ei[1], &ek[1], t_true,
                               cmp_op, swap_lt_at_zero);
         emit_reg_bitonic_pair(&b, &ei[2], &ek[2], &ei[3], &ek[3], t_false,
                               cmp_op, swap_lt_at_zero);
         emit_subgroup_tail_16(&b, &ei[0], &ek[0], is_hi_bits, t_true, cmp_op,
                               swap_lt_at_zero);
         emit_subgroup_tail_16(&b, &ei[1], &ek[1], is_hi_bits, t_true, cmp_op,
                               swap_lt_at_zero);
         emit_subgroup_tail_16(&b, &ei[2], &ek[2], is_hi_bits, t_false, cmp_op,
                               swap_lt_at_zero);
         emit_subgroup_tail_16(&b, &ei[3], &ek[3], is_hi_bits, t_false, cmp_op,
                               swap_lt_at_zero);

         /* k = 128: asc128 = ((blk & 128) == 0) */
         nir_def *asc128 = nir_ieq_imm(&b, nir_iand_imm(&b, blk, 128), 0);
         emit_reg_bitonic_pair(&b, &ei[0], &ek[0], &ei[2], &ek[2], asc128,
                               cmp_op, swap_lt_at_zero);
         emit_reg_bitonic_pair(&b, &ei[1], &ek[1], &ei[3], &ek[3], asc128,
                               cmp_op, swap_lt_at_zero);
         emit_reg_bitonic_pair(&b, &ei[0], &ek[0], &ei[1], &ek[1], asc128,
                               cmp_op, swap_lt_at_zero);
         emit_reg_bitonic_pair(&b, &ei[2], &ek[2], &ei[3], &ek[3], asc128,
                               cmp_op, swap_lt_at_zero);
         for (unsigned m = 0; m < 4; m++) {
            emit_subgroup_tail_16(&b, &ei[m], &ek[m], is_hi_bits, asc128,
                                  cmp_op, swap_lt_at_zero);
            nir_store_shared(&b, nir_vec2(&b, ei[m], ek[m]), offs[m], .base = 0,
                             .write_mask = 0x3, .align_mul = 8);
         }

         nir_store_var(&b, blk_var, nir_iadd_imm(&b, blk, blk_stride), 0x1);
      }
      nir_pop_loop(&b, p1_blk_loop);

      emit_wg_shared_barrier(&b, wg_scope);

      /* Phase 2: Inter-subgroup stages for k = 256, 512, ..., P. */
      nir_push_if(&b, nir_uge_imm(&b, p_def, 256));
      {
         nir_def *half_p = nir_ushr_imm(&b, p_def, 1);
         nir_variable *k_var =
            nir_local_variable_create(impl, glsl_uint_type(), "bs_k");
         nir_store_var(&b, k_var, nir_imm_int(&b, 256), 0x1);

         nir_loop *new_k = nir_push_loop(&b);
         {
            nir_def *kv = nir_load_var(&b, k_var);
            nir_push_if(&b, nir_ult(&b, p_def, kv));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, NULL);

            /* 2a. Shared-memory stages j = k>>1 .. 128 (interleaved u32x2,
             * dense pair index p in [0, P/2) = 100% active lanes). */
            nir_variable *j_var =
               nir_local_variable_create(impl, glsl_uint_type(), "bs_j");
            nir_store_var(&b, j_var, nir_ushr_imm(&b, kv, 1), 0x1);

            nir_loop *new_j = nir_push_loop(&b);
            {
               nir_def *jv = nir_load_var(&b, j_var);
               nir_push_if(&b, nir_ult_imm(&b, jv, 128));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, NULL);

               nir_def *j_mask = nir_iadd_imm(&b, jv, -1);
               nir_variable *p_var =
                  nir_local_variable_create(impl, glsl_uint_type(), "bs_p");
               nir_store_var(&b, p_var, lid, 0x1);

               nir_loop *p_loop = nir_push_loop(&b);
               {
                  nir_def *pv = nir_load_var(&b, p_var);
                  nir_push_if(&b, nir_uge(&b, pv, half_p));
                  nir_jump(&b, nir_jump_break);
                  nir_pop_if(&b, NULL);

                  nir_def *lo = nir_iand(&b, pv, j_mask);
                  nir_def *hi = nir_ishl_imm(&b, nir_isub(&b, pv, lo), 1);
                  nir_def *i0 = nir_ior(&b, hi, lo);
                  nir_def *i1 = nir_ior(&b, i0, jv);
                  nir_def *off0 = nir_ishl_imm(&b, i0, 3);
                  nir_def *off1 = nir_ishl_imm(&b, i1, 3);

                  nir_def *va = nir_load_shared(&b, 2, 32, off0, .base = 0,
                                                .align_mul = 8);
                  nir_def *vb = nir_load_shared(&b, 2, 32, off1, .base = 0,
                                                .align_mul = 8);
                  nir_def *ka = nir_channel(&b, va, 1);
                  nir_def *kb = nir_channel(&b, vb, 1);
                  nir_def *asc = nir_ieq_imm(&b, nir_iand(&b, i0, kv), 0);
                  nir_def *c_ab = nir_build_alu2(&b, cmp_op, ka, kb);
                  nir_def *c_ba = nir_build_alu2(&b, cmp_op, kb, ka);
                  nir_def *do_swap = swap_lt_at_zero
                     ? nir_bcsel(&b, asc, c_ab, c_ba)
                     : nir_bcsel(&b, asc, c_ba, c_ab);

                  nir_push_if(&b, do_swap);
                  {
                     nir_store_shared(&b, vb, off0, .base = 0,
                                      .write_mask = 0x3, .align_mul = 8);
                     nir_store_shared(&b, va, off1, .base = 0,
                                      .write_mask = 0x3, .align_mul = 8);
                  }
                  nir_pop_if(&b, NULL);

                  nir_store_var(&b, p_var, nir_iadd_imm(&b, pv, wg_x), 0x1);
               }
               nir_pop_loop(&b, p_loop);

               emit_wg_shared_barrier(&b, wg_scope);
               nir_store_var(&b, j_var, nir_ushr_imm(&b, jv, 1), 0x1);
            }
            nir_pop_loop(&b, new_j);

            /* 2b. Tail stages j = 64, 32, 16, 8, 4, 2, 1 in registers
             * (4 elements/lane = 128 elements/subgroup, 1 barrier total!). */
            nir_variable *tblk_var =
               nir_local_variable_create(impl, glsl_uint_type(), "bs_tblk");
            nir_store_var(&b, tblk_var, sg_base0, 0x1);

            nir_loop *tail_loop = nir_push_loop(&b);
            tail_loop->control = nir_loop_control_dont_unroll;
            {
               nir_def *blk = nir_load_var(&b, tblk_var);
               nir_push_if(&b, nir_uge(&b, blk, p_def));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, NULL);

               nir_def *base_lane = nir_iadd(&b, blk, lane);
               nir_def *offs[4], *ei[4], *ek[4];
               for (unsigned m = 0; m < 4; m++) {
                  offs[m] =
                     nir_ishl_imm(&b, nir_iadd_imm(&b, base_lane, m * 32), 3);
                  nir_def *v = nir_load_shared(&b, 2, 32, offs[m], .base = 0,
                                               .align_mul = 8);
                  ei[m] = nir_channel(&b, v, 0);
                  ek[m] = nir_channel(&b, v, 1);
               }
               nir_def *asc = nir_ieq_imm(&b, nir_iand(&b, blk, kv), 0);

               /* j = 64, j = 32 in registers */
               emit_reg_bitonic_pair(&b, &ei[0], &ek[0], &ei[2], &ek[2], asc,
                                     cmp_op, swap_lt_at_zero);
               emit_reg_bitonic_pair(&b, &ei[1], &ek[1], &ei[3], &ek[3], asc,
                                     cmp_op, swap_lt_at_zero);
               emit_reg_bitonic_pair(&b, &ei[0], &ek[0], &ei[1], &ek[1], asc,
                                     cmp_op, swap_lt_at_zero);
               emit_reg_bitonic_pair(&b, &ei[2], &ek[2], &ei[3], &ek[3], asc,
                                     cmp_op, swap_lt_at_zero);

               /* j = 16, 8, 4, 2, 1 via 1-cycle subgroup shuffle_xor */
               for (unsigned m = 0; m < 4; m++) {
                  emit_subgroup_tail_16(&b, &ei[m], &ek[m], is_hi_bits, asc,
                                        cmp_op, swap_lt_at_zero);
                  nir_store_shared(&b, nir_vec2(&b, ei[m], ek[m]), offs[m],
                                   .base = 0, .write_mask = 0x3,
                                   .align_mul = 8);
               }

               nir_store_var(&b, tblk_var, nir_iadd_imm(&b, blk, blk_stride),
                             0x1);
            }
            nir_pop_loop(&b, tail_loop);

            emit_wg_shared_barrier(&b, wg_scope);
            nir_store_var(&b, k_var, nir_ishl_imm(&b, kv, 1), 0x1);
         }
         nir_pop_loop(&b, new_k);
      }
      nir_pop_if(&b, NULL);
   }
   nir_push_else(&b, NULL);
   {
      /* Small-P fallback path for P in {2, 4, 8, 16, 32, 64}. */
      nir_push_if(&b, nir_uge_imm(&b, p_def, 2));
      {
         nir_variable *si_var =
            nir_local_variable_create(impl, glsl_uint_type(), "bs_si");
         nir_store_var(&b, si_var, lid, 0x1);
         nir_loop *s_loop = nir_push_loop(&b);
         s_loop->control = nir_loop_control_dont_unroll;
         {
            nir_def *i = nir_load_var(&b, si_var);
            nir_push_if(&b, nir_uge(&b, i, p_def));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, NULL);

            nir_def *off = nir_ishl_imm(&b, i, 3);
            nir_def *v =
               nir_load_shared(&b, 2, 32, off, .base = 0, .align_mul = 8);
            nir_def *ci = nir_channel(&b, v, 0);
            nir_def *ck = nir_channel(&b, v, 1);

            emit_subgroup_bitonic_stage(&b, &ci, &ck, is_hi_bits[0],
                                        asc_bits[0], 1, cmp_op,
                                        swap_lt_at_zero);
            nir_variable *iv_s =
               nir_local_variable_create(impl, glsl_uint_type(), "bs_s_i");
            nir_variable *kv_s =
               nir_local_variable_create(impl, glsl_uint_type(), "bs_s_k");
            nir_store_var(&b, iv_s, ci, 0x1);
            nir_store_var(&b, kv_s, ck, 0x1);

            for (unsigned k_bit = 2; k_bit <= 5; k_bit++) {
               nir_push_if(&b, nir_uge_imm(&b, p_def, 1u << k_bit));
               {
                  nir_def *li = nir_load_var(&b, iv_s);
                  nir_def *lk = nir_load_var(&b, kv_s);
                  nir_def *asc =
                     k_bit < 5
                        ? asc_bits[k_bit - 1]
                        : nir_ieq_imm(&b, nir_iand_imm(&b, i, 32), 0);
                  for (int j_bit = (int)k_bit - 1; j_bit >= 0; j_bit--) {
                     emit_subgroup_bitonic_stage(&b, &li, &lk,
                                                 is_hi_bits[j_bit], asc,
                                                 1u << j_bit, cmp_op,
                                                 swap_lt_at_zero);
                  }
                  nir_store_var(&b, iv_s, li, 0x1);
                  nir_store_var(&b, kv_s, lk, 0x1);
               }
               nir_pop_if(&b, NULL);
            }

            nir_store_shared(&b,
                             nir_vec2(&b, nir_load_var(&b, iv_s),
                                      nir_load_var(&b, kv_s)),
                             off, .base = 0, .write_mask = 0x3,
                             .align_mul = 8);
            nir_store_var(&b, si_var, nir_iadd_imm(&b, i, wg_x), 0x1);
         }
         nir_pop_loop(&b, s_loop);
         emit_wg_shared_barrier(&b, wg_scope);

         /* If P == 64: single j = 32 pair pass + single j = 16..1 tail pass. */
         nir_push_if(&b, nir_ieq_imm(&b, p_def, 64));
         {
            nir_push_if(&b, nir_ult_imm(&b, lid, 32));
            {
               nir_def *off0 = nir_ishl_imm(&b, lid, 3);
               nir_def *off1 = nir_ishl_imm(&b, nir_iadd_imm(&b, lid, 32), 3);
               nir_def *va =
                  nir_load_shared(&b, 2, 32, off0, .base = 0, .align_mul = 8);
               nir_def *vb =
                  nir_load_shared(&b, 2, 32, off1, .base = 0, .align_mul = 8);
               nir_def *ia = nir_channel(&b, va, 0);
               nir_def *ka = nir_channel(&b, va, 1);
               nir_def *ib = nir_channel(&b, vb, 0);
               nir_def *kb = nir_channel(&b, vb, 1);
               emit_reg_bitonic_pair(&b, &ia, &ka, &ib, &kb, nir_imm_true(&b),
                                     cmp_op, swap_lt_at_zero);
               nir_store_shared(&b, nir_vec2(&b, ia, ka), off0, .base = 0,
                                .write_mask = 0x3, .align_mul = 8);
               nir_store_shared(&b, nir_vec2(&b, ib, kb), off1, .base = 0,
                                .write_mask = 0x3, .align_mul = 8);
            }
            nir_pop_if(&b, NULL);
            emit_wg_shared_barrier(&b, wg_scope);

            nir_variable *t64 =
               nir_local_variable_create(impl, glsl_uint_type(), "bs_t64");
            nir_store_var(&b, t64, lid, 0x1);
            nir_loop *l64 = nir_push_loop(&b);
            l64->control = nir_loop_control_dont_unroll;
            {
               nir_def *i = nir_load_var(&b, t64);
               nir_push_if(&b, nir_uge_imm(&b, i, 64));
               nir_jump(&b, nir_jump_break);
               nir_pop_if(&b, NULL);

               nir_def *off = nir_ishl_imm(&b, i, 3);
               nir_def *v =
                  nir_load_shared(&b, 2, 32, off, .base = 0, .align_mul = 8);
               nir_def *ci = nir_channel(&b, v, 0);
               nir_def *ck = nir_channel(&b, v, 1);
               emit_subgroup_tail_16(&b, &ci, &ck, is_hi_bits,
                                     nir_imm_true(&b), cmp_op,
                                     swap_lt_at_zero);
               nir_store_shared(&b, nir_vec2(&b, ci, ck), off, .base = 0,
                                .write_mask = 0x3, .align_mul = 8);
               nir_store_var(&b, t64, nir_iadd_imm(&b, i, wg_x), 0x1);
            }
            nir_pop_loop(&b, l64);
            emit_wg_shared_barrier(&b, wg_scope);
         }
         nir_pop_if(&b, NULL);
      }
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);

   /* Remove original sort CF node and convert our new locals to SSA. */
   nir_cf_node_remove(sort_outer_node);
   nir_lower_vars_to_ssa(nir);
   nir_progress(true, impl, nir_metadata_none);
   return true;
}

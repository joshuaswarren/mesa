/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "util/hash_table.h"

/*
 * Uniform 3x3 specialization and straight-line unrolling for nested 2D
 * pure-read depthwise convolution loops on AGX3 (shared AGX3B + AGX3C).
 *
 * Matches a nested 2D uniform-bound counted loop (even when marked
 * nir_loop_control_dont_unroll):
 *   for (kh = 0; kh < KH; kh += 1) {
 *      [optional uniform if (inner_guard) {]
 *         [optional row-prologue loads / moves]
 *         for (kw = 0; kw < KW; kw += 1) {
 *            // pure readonly loads + float FMAs + address steps
 *         }
 *      [}]
 *      // row-epilogue address steps
 *   }
 *
 * When KH and KW are convergent, loop-invariant, non-constant 32-bit bounds
 * and the loop body fits within the register-pressure and code-size budget,
 * emits a convergent runtime guard `if (KH == 3 && KW == 3 [&& inner_guard])`:
 *   - then: directly unrolls all 3x3 = 9 taps straight-line, eliminating all
 *     12 loop branches, header phis, and sliding-window copy moves while
 *     keeping per-row input live ranges tight.
 *   - else: preserves the untouched original nested loop for all other
 *     kernel dimensions.
 */

#define DW3X3_MAX_OUTER_INSTRS 64
#define DW3X3_MAX_OUTER_PHIS   16
#define DW3X3_MAX_INNER_PHIS   20
#define DW3X3_MAX_LIVE_OUTS    16

static bool
block_in_cf_node(nir_block *block, nir_cf_node *node)
{
   for (nir_cf_node *cur = &block->cf_node; cur != NULL; cur = cur->parent) {
      if (cur == node)
         return true;
   }
   return false;
}

static bool
def_dominates_loop(nir_def *def, nir_loop *loop)
{
   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   return nir_block_dominates(nir_def_block(def), pre);
}

static bool
is_inv_bound_def(nir_def *def, nir_loop *loop, unsigned depth)
{
   if (depth > 6 || !def)
      return false;
   if (def_dominates_loop(def, loop))
      return true;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const)
      return true;
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_load_preamble &&
          intr->intrinsic != nir_intrinsic_load_constant_agx &&
          intr->intrinsic != nir_intrinsic_load_global_constant)
         return false;
      unsigned n = nir_intrinsic_infos[intr->intrinsic].num_srcs;
      for (unsigned i = 0; i < n; ++i) {
         if (!is_inv_bound_def(intr->src[i].ssa, loop, depth + 1))
            return false;
      }
      return true;
   }
   return false;
}

static nir_def *
materialize_inv_bound(nir_builder *b, nir_def *def, nir_loop *loop,
                      unsigned depth)
{
   assert(depth <= 6);
   if (def_dominates_loop(def, loop))
      return def;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const) {
      nir_load_const_instr *lc = nir_instr_as_load_const(instr);
      return nir_build_imm(b, def->num_components, def->bit_size, lc->value);
   }
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   unsigned n = nir_intrinsic_infos[intr->intrinsic].num_srcs;
   nir_def *mat[4] = {NULL};
   for (unsigned i = 0; i < n; ++i)
      mat[i] = materialize_inv_bound(b, intr->src[i].ssa, loop, depth + 1);
   nir_intrinsic_instr *clone =
      nir_instr_as_intrinsic(nir_instr_clone(b->shader, instr));
   nir_builder_instr_insert(b, &clone->instr);
   for (unsigned i = 0; i < n; ++i)
      nir_src_rewrite(&clone->src[i], mat[i]);
   return &clone->def;
}

static nir_def *
chase_movs(nir_def *def)
{
   while (def && nir_def_instr(def)->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_def_as_alu(def);
      if (alu->op == nir_op_mov && def->num_components == 1 &&
          alu->src[0].src.ssa->num_components == 1) {
         def = alu->src[0].src.ssa;
         continue;
      }
      break;
   }
   return def;
}

static nir_def *
phi_src_for_block(nir_phi_instr *phi, nir_block *pred)
{
   nir_foreach_phi_src(src, phi) {
      if (src->pred == pred)
         return src->src.ssa;
   }
   return NULL;
}

static bool
instr_is_pure_read(nir_instr *instr, bool *saw_load, bool *saw_fma)
{
   switch (instr->type) {
   case nir_instr_type_phi:
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
   case nir_instr_type_jump:
      return true;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op == nir_op_ffma || alu->op == nir_op_fmul)
         *saw_fma = true;
      return true;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_preamble:
      case nir_intrinsic_load_constant_agx:
      case nir_intrinsic_load_ubo:
         return true;
      case nir_intrinsic_load_agx:
      case nir_intrinsic_load_global:
      case nir_intrinsic_load_ssbo: {
         unsigned access = nir_intrinsic_access(intr);
         if (!(access & ACCESS_NON_WRITEABLE) ||
             (access & (ACCESS_COHERENT | ACCESS_VOLATILE)))
            return false;
         *saw_load = true;
         return true;
      }
      default:
         return false;
      }
   }
   default:
      return false;
   }
}

struct counted_loop_info {
   nir_loop *loop;
   nir_block *pre;
   nir_block *hdr;
   nir_block *latch;
   nir_if *exit_if;
   nir_phi_instr *iv_phi;
   nir_scalar bound;
   bool bottom_tested;
};

/* Match a counted loop `iv = 0; iv < bound; iv += 1` (either top-tested
 * `if (iv >= bound) break;` or bottom-tested `if (iv + 1 >= bound) break;`)
 * with convergent non-constant bound dominating `dom_loop`. */
static bool
match_counted_unit_loop(nir_loop *loop, nir_loop *dom_loop,
                        nir_if *exit_if, struct counted_loop_info *info)
{
   memset(info, 0, sizeof(*info));
   if (nir_loop_has_continue_construct(loop))
      return false;

   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&loop->cf_node));
   nir_block *hdr = nir_loop_first_block(loop);
   nir_block *latch = nir_loop_last_block(loop);
   if (util_dynarray_num_elements(&hdr->predecessors, nir_block *) != 2)
      return false;

   /* Verify exit_if is `if (cond) { break; } else {}`. */
   nir_block *etb = nir_if_first_then_block(exit_if);
   nir_block *eeb = nir_if_first_else_block(exit_if);
   if (etb != nir_if_last_then_block(exit_if) ||
       eeb != nir_if_last_else_block(exit_if) ||
       !exec_list_is_empty(&eeb->instr_list))
      return false;
   nir_instr *brk = nir_block_last_instr(etb);
   if (!brk || brk->type != nir_instr_type_jump ||
       nir_instr_as_jump(brk)->type != nir_jump_break ||
       nir_block_first_instr(etb) != brk)
      return false;

   nir_def *cond = chase_movs(exit_if->condition.ssa);
   if (nir_def_instr(cond)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *cmp = nir_def_as_alu(cond);
   if (cmp->op != nir_op_ige && cmp->op != nir_op_uge)
      return false;

   nir_def *lhs = chase_movs(cmp->src[0].src.ssa);
   nir_scalar bound = nir_scalar_chase_movs(
      nir_get_scalar(cmp->src[1].src.ssa, cmp->src[1].swizzle[0]));
   if (bound.def->bit_size != 32 || bound.def->divergent ||
       !is_inv_bound_def(bound.def, dom_loop, 0))
      return false;
   /* Already-constant bounds are left to generic unrolling. */
   if (nir_def_instr(bound.def)->type == nir_instr_type_load_const)
      return false;

   /* Find matching iv_phi: init == 0, step == iv + 1, lhs == iv or iv + 1. */
   nir_phi_instr *iv_phi = NULL;
   bool bottom_tested = false;
   nir_foreach_phi(phi, hdr) {
      if (phi->def.num_components != 1 || phi->def.bit_size != 32 ||
          phi->def.divergent)
         continue;
      nir_def *init = phi_src_for_block(phi, pre);
      nir_def *next = phi_src_for_block(phi, latch);
      if (!init || !next)
         continue;
      if (nir_def_instr(init)->type != nir_instr_type_load_const ||
          nir_instr_as_load_const(nir_def_instr(init))->value[0].u32 != 0)
         continue;
      next = chase_movs(next);
      if (nir_def_instr(next)->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *step = nir_def_as_alu(next);
      if (step->op != nir_op_iadd)
         continue;
      bool step_ok = false;
      for (unsigned i = 0; i < 2; ++i) {
         if (chase_movs(step->src[i].src.ssa) == &phi->def &&
             nir_src_is_const(step->src[1 - i].src) &&
             nir_src_as_uint(step->src[1 - i].src) == 1)
            step_ok = true;
      }
      if (!step_ok)
         continue;

      if (lhs == &phi->def) {
         iv_phi = phi;
         bottom_tested = false;
         break;
      } else if (lhs == next) {
         iv_phi = phi;
         bottom_tested = true;
         break;
      }
   }

   if (!iv_phi)
      return false;

   /* Top-tested loops must have exit_if right after hdr; bottom-tested loops
    * must have exit_if after the body block and only a single latch block
    * following it. */
   if (!bottom_tested) {
      if (nir_cf_node_next(&hdr->cf_node) != &exit_if->cf_node)
         return false;
   } else {
      if (nir_cf_node_next(&exit_if->cf_node) != &latch->cf_node)
         return false;
   }

   info->loop = loop;
   info->pre = pre;
   info->hdr = hdr;
   info->latch = latch;
   info->exit_if = exit_if;
   info->iv_phi = iv_phi;
   info->bound = bound;
   info->bottom_tested = bottom_tested;
   return true;
}

struct dw3x3_match {
   struct counted_loop_info outer;
   struct counted_loop_info inner;

   /* Optional `if (guard) { [pre_inner] inner } else {}` around inner loop. */
   nir_if *guard_if;
   bool guard_if_is_outer_inv;
   nir_block *guard_merge_blk;

   /* Optional single pure-read `if (tap_ok) { blk } else {}` inside inner loop. */
   nir_if *inner_tap_if;
   nir_block *inner_tap_merge_blk;

   nir_block *outer_post_exit_blk;
   nir_block *outer_pre_exit_blk;
   nir_block *inner_pre_blk;

   nir_def *live_outs[DW3X3_MAX_LIVE_OUTS];
   unsigned num_live_outs;
};

static bool
match_dw3x3_nest(nir_loop *outer_loop, struct dw3x3_match *m)
{
   memset(m, 0, sizeof(*m));

   /* Outer loop may be either:
    *   - top-tested: outer_hdr -> outer_exit -> post_exit -> (inner|guard) -> latch
    *   - bottom-tested: outer_hdr -> (inner|guard) -> pre_exit -> outer_exit -> latch
    */
   nir_block *outer_hdr = nir_loop_first_block(outer_loop);
   nir_block *outer_latch = nir_loop_last_block(outer_loop);
   nir_cf_node *after_hdr = nir_cf_node_next(&outer_hdr->cf_node);
   nir_cf_node *before_latch = nir_cf_node_prev(&outer_latch->cf_node);
   if (!after_hdr || !before_latch)
      return false;

   bool matched_outer = false;
   if (after_hdr->type == nir_cf_node_if) {
      nir_if *cand = nir_cf_node_as_if(after_hdr);
      if (match_counted_unit_loop(outer_loop, outer_loop, cand, &m->outer) &&
          !m->outer.bottom_tested)
         matched_outer = true;
   }
   if (!matched_outer && before_latch->type == nir_cf_node_if) {
      nir_if *cand = nir_cf_node_as_if(before_latch);
      if (match_counted_unit_loop(outer_loop, outer_loop, cand, &m->outer) &&
          m->outer.bottom_tested)
         matched_outer = true;
   }
   if (!matched_outer)
      return false;

   nir_cf_node *mid = NULL;
   nir_block *after_mid_expected = NULL;
   if (!m->outer.bottom_tested) {
      nir_cf_node *post_exit_node = nir_cf_node_next(&m->outer.exit_if->cf_node);
      if (!post_exit_node || post_exit_node->type != nir_cf_node_block)
         return false;
      m->outer_post_exit_blk = nir_cf_node_as_block(post_exit_node);
      mid = nir_cf_node_next(post_exit_node);
      after_mid_expected = m->outer.latch;
   } else {
      m->outer_post_exit_blk = outer_hdr;
      mid = after_hdr;
      nir_cf_node *pre_exit_node = nir_cf_node_prev(&m->outer.exit_if->cf_node);
      if (!pre_exit_node || pre_exit_node->type != nir_cf_node_block)
         return false;
      m->outer_pre_exit_blk = nir_cf_node_as_block(pre_exit_node);
      after_mid_expected = m->outer_pre_exit_blk;
   }
   if (!mid)
      return false;

   nir_loop *inner_loop = NULL;
   if (mid->type == nir_cf_node_loop) {
      if (nir_cf_node_next(mid) != &after_mid_expected->cf_node)
         return false;
      inner_loop = nir_cf_node_as_loop(mid);
      m->inner_pre_blk = m->outer_post_exit_blk;
   } else if (mid->type == nir_cf_node_if) {
      if (nir_cf_node_next(mid) != &after_mid_expected->cf_node)
         return false;
      nir_if *gif = nir_cf_node_as_if(mid);
      nir_block *geb = nir_if_first_else_block(gif);
      if (geb != nir_if_last_else_block(gif) ||
          !exec_list_is_empty(&geb->instr_list))
         return false;

      /* Then-list must be: [pre_block], inner_loop, [empty post_block]. */
      nir_block *pre_blk = nir_if_first_then_block(gif);
      nir_cf_node *in_node = nir_cf_node_next(&pre_blk->cf_node);
      if (!in_node || in_node->type != nir_cf_node_loop)
         return false;
      nir_cf_node *post_in = nir_cf_node_next(in_node);
      if (!post_in || post_in->type != nir_cf_node_block ||
          nir_cf_node_next(post_in) != NULL)
         return false;
      if (!exec_list_is_empty(&nir_cf_node_as_block(post_in)->instr_list))
         return false;

      m->guard_if = gif;
      m->guard_if_is_outer_inv =
         !gif->condition.ssa->divergent &&
         def_dominates_loop(gif->condition.ssa, outer_loop);
      m->guard_merge_blk = after_mid_expected;
      m->inner_pre_blk = pre_blk;
      inner_loop = nir_cf_node_as_loop(in_node);
   } else {
      return false;
   }

   /* Inner loop must have 1 exit_if and at most 1 simple pure-read tap_if
    * (`if (c) { single_block } else {}`), with no nested loops. */
   nir_if *inner_ifs_arr[2] = {NULL, NULL};
   unsigned inner_ifs = 0;
   foreach_list_typed(nir_cf_node, node, node, &inner_loop->body) {
      if (node->type == nir_cf_node_loop)
         return false;
      if (node->type == nir_cf_node_if) {
         if (inner_ifs >= 2)
            return false;
         inner_ifs_arr[inner_ifs++] = nir_cf_node_as_if(node);
      }
   }
   if (inner_ifs == 0)
      return false;

   nir_if *inner_exit = NULL;
   nir_if *tap_if = NULL;
   for (unsigned i = 0; i < inner_ifs; ++i) {
      struct counted_loop_info tmp;
      if (!inner_exit &&
          match_counted_unit_loop(inner_loop, outer_loop, inner_ifs_arr[i], &tmp)) {
         inner_exit = inner_ifs_arr[i];
         m->inner = tmp;
      } else if (!tap_if) {
         tap_if = inner_ifs_arr[i];
      } else {
         return false;
      }
   }
   if (!inner_exit)
      return false;

   if (tap_if) {
      nir_block *ttb = nir_if_first_then_block(tap_if);
      nir_block *teb = nir_if_first_else_block(tap_if);
      if (ttb != nir_if_last_then_block(tap_if) ||
          teb != nir_if_last_else_block(tap_if) ||
          !exec_list_is_empty(&teb->instr_list))
         return false;
      nir_cf_node *after_tap = nir_cf_node_next(&tap_if->cf_node);
      if (!after_tap || after_tap->type != nir_cf_node_block)
         return false;
      /* Verify tap_if sits inside the body (before exit_if when bottom-tested,
       * or after exit_if when top-tested). */
      if (m->inner.bottom_tested) {
         if (nir_cf_node_next(&m->inner.hdr->cf_node) != &tap_if->cf_node ||
             nir_cf_node_next(after_tap) != &m->inner.exit_if->cf_node)
            return false;
      } else {
         nir_cf_node *post_ex = nir_cf_node_next(&m->inner.exit_if->cf_node);
         if (!post_ex || nir_cf_node_next(post_ex) != &tap_if->cf_node ||
             after_tap != &m->inner.latch->cf_node)
            return false;
      }
      m->inner_tap_if = tap_if;
      m->inner_tap_merge_blk = nir_cf_node_as_block(after_tap);
   }

   /* Count phis and verify pure-read body + instruction budget. */
   unsigned n_outer_phis = 0;
   nir_foreach_phi(phi, m->outer.hdr) {
      (void)phi;
      if (++n_outer_phis > DW3X3_MAX_OUTER_PHIS)
         return false;
   }
   unsigned n_inner_phis = 0;
   nir_foreach_phi(phi, m->inner.hdr) {
      (void)phi;
      if (++n_inner_phis > DW3X3_MAX_INNER_PHIS)
         return false;
   }

   unsigned total_instrs = 0;
   bool saw_load = false;
   bool saw_fma = false;
   nir_foreach_block_in_cf_node(blk, &outer_loop->cf_node) {
      nir_foreach_instr(instr, blk) {
         bool in_inner = block_in_cf_node(blk, &inner_loop->cf_node);
         bool blk_load = false, blk_fma = false;
         if (!instr_is_pure_read(instr, &blk_load, &blk_fma))
            return false;
         if (in_inner) {
            saw_load |= blk_load;
            saw_fma |= blk_fma;
         }
         if (instr->type != nir_instr_type_phi &&
             instr->type != nir_instr_type_jump &&
             instr->type != nir_instr_type_load_const) {
            if (++total_instrs > DW3X3_MAX_OUTER_INSTRS)
               return false;
         }
      }
   }
   if (!saw_load || !saw_fma)
      return false;

   /* Collect live-out SSA values defined in outer_loop and used outside it. */
   nir_foreach_block_in_cf_node(blk, &outer_loop->cf_node) {
      nir_foreach_instr(instr, blk) {
         nir_def *def = nir_instr_def(instr);
         if (!def)
            continue;
         bool used_outside = false;
         nir_foreach_use_including_if(src, def) {
            nir_block *ublk = nir_src_is_if(src)
               ? nir_cf_node_as_block(
                    nir_cf_node_prev(&nir_src_use_if(src)->cf_node))
               : nir_src_use_instr(src)->block;
            if (!block_in_cf_node(ublk, &outer_loop->cf_node)) {
               used_outside = true;
               break;
            }
         }
         if (!used_outside)
            continue;
         /* Top-tested exits through outer_hdr (only header phis live out);
          * bottom-tested exits through outer_pre_exit_blk (must dominate it). */
         if (!m->outer.bottom_tested) {
            if (instr->type != nir_instr_type_phi || blk != m->outer.hdr)
               return false;
         } else {
            if (!nir_block_dominates(blk, m->outer_pre_exit_blk))
               return false;
         }
         if (m->num_live_outs >= DW3X3_MAX_LIVE_OUTS)
            return false;
         m->live_outs[m->num_live_outs++] = def;
      }
   }

   return m->num_live_outs > 0;
}

static bool
remap_src_cb(nir_src *src, void *state)
{
   struct hash_table *remap = state;
   struct hash_entry *e = _mesa_hash_table_search(remap, src->ssa);
   if (e)
      nir_src_rewrite(src, e->data);
   return true;
}

static nir_def *
lookup_remap(struct hash_table *remap, nir_def *def)
{
   struct hash_entry *e = _mesa_hash_table_search(remap, def);
   return e ? (nir_def *)e->data : def;
}

static void
clone_block_non_phi_instrs(nir_builder *b, nir_block *src_blk,
                           struct hash_table *remap)
{
   nir_foreach_instr(instr, src_blk) {
      if (instr->type == nir_instr_type_phi ||
          instr->type == nir_instr_type_jump)
         continue;
      nir_instr *clone = nir_instr_clone(b->shader, instr);
      nir_builder_instr_insert(b, clone);
      nir_foreach_src(clone, remap_src_cb, remap);
      nir_def *orig_def = nir_instr_def(instr);
      nir_def *clone_def = nir_instr_def(clone);
      if (orig_def && clone_def)
         _mesa_hash_table_insert(remap, orig_def, clone_def);
   }
}

static void
clone_simple_tap_if(nir_builder *b, nir_if *tap_if, nir_block *merge_blk,
                    struct hash_table *remap)
{
   nir_def *cond = lookup_remap(remap, tap_if->condition.ssa);
   nir_if *cif = nir_push_if(b, cond);
   struct hash_table *then_map = _mesa_pointer_hash_table_create(NULL);
   hash_table_foreach(remap, entry)
      _mesa_hash_table_insert(then_map, entry->key, entry->data);
   clone_block_non_phi_instrs(b, nir_if_first_then_block(tap_if), then_map);
   nir_pop_if(b, cif);

   nir_block *orig_then = nir_if_last_then_block(tap_if);
   nir_block *orig_else = nir_if_last_else_block(tap_if);
   nir_block *new_then = nir_if_last_then_block(cif);
   nir_block *new_else = nir_if_last_else_block(cif);

   nir_foreach_phi(phi, merge_blk) {
      nir_def *tval = lookup_remap(then_map, phi_src_for_block(phi, orig_then));
      nir_def *eval = lookup_remap(remap, phi_src_for_block(phi, orig_else));
      nir_phi_instr *nphi = nir_phi_instr_create(b->shader);
      nir_def_init(&nphi->instr, &nphi->def, phi->def.num_components,
                   phi->def.bit_size);
      nir_phi_instr_add_src(nphi, new_then, tval);
      nir_phi_instr_add_src(nphi, new_else, eval);
      nir_builder_instr_insert(b, &nphi->instr);
      _mesa_hash_table_insert(remap, &phi->def, &nphi->def);
   }

   _mesa_hash_table_destroy(then_map, NULL);
}

static void
emit_unrolled_inner_3(nir_builder *b, const struct dw3x3_match *m,
                      struct hash_table *outer_map,
                      struct hash_table *inner_out_map)
{
   if (m->inner_pre_blk != m->outer.hdr)
      clone_block_non_phi_instrs(b, m->inner_pre_blk, inner_out_map);

   nir_foreach_phi(phi, m->inner.hdr) {
      nir_def *init = phi_src_for_block(phi, m->inner.pre);
      _mesa_hash_table_insert(inner_out_map, &phi->def,
                              lookup_remap(inner_out_map, init));
   }

   for (unsigned kx = 0; kx < 3; ++kx) {
      clone_block_non_phi_instrs(b, m->inner.hdr, inner_out_map);
      if (!m->inner.bottom_tested) {
         nir_cf_node *n = nir_cf_node_next(&m->inner.exit_if->cf_node);
         if (m->inner_tap_if) {
            clone_block_non_phi_instrs(b, nir_cf_node_as_block(n),
                                       inner_out_map);
            clone_simple_tap_if(b, m->inner_tap_if, m->inner_tap_merge_blk,
                                inner_out_map);
            clone_block_non_phi_instrs(b, m->inner_tap_merge_blk,
                                       inner_out_map);
         } else {
            while (n) {
               assert(n->type == nir_cf_node_block);
               clone_block_non_phi_instrs(b, nir_cf_node_as_block(n),
                                          inner_out_map);
               n = nir_cf_node_next(n);
            }
         }
         nir_def *next_vals[DW3X3_MAX_INNER_PHIS];
         unsigned pi = 0;
         nir_foreach_phi(phi, m->inner.hdr)
            next_vals[pi++] = lookup_remap(
               inner_out_map, phi_src_for_block(phi, m->inner.latch));
         pi = 0;
         nir_foreach_phi(phi, m->inner.hdr)
            _mesa_hash_table_insert(inner_out_map, &phi->def, next_vals[pi++]);
      } else {
         if (m->inner_tap_if) {
            clone_simple_tap_if(b, m->inner_tap_if, m->inner_tap_merge_blk,
                                inner_out_map);
            clone_block_non_phi_instrs(b, m->inner_tap_merge_blk,
                                       inner_out_map);
         }
         if (kx < 2) {
            clone_block_non_phi_instrs(b, m->inner.latch, inner_out_map);
            nir_def *next_vals[DW3X3_MAX_INNER_PHIS];
            unsigned pi = 0;
            nir_foreach_phi(phi, m->inner.hdr)
               next_vals[pi++] = lookup_remap(
                  inner_out_map, phi_src_for_block(phi, m->inner.latch));
            pi = 0;
            nir_foreach_phi(phi, m->inner.hdr)
               _mesa_hash_table_insert(inner_out_map, &phi->def,
                                       next_vals[pi++]);
         }
      }
   }
   (void)outer_map;
}

static void
emit_unrolled_3x3(nir_builder *b, const struct dw3x3_match *m,
                  nir_def **then_live_outs)
{
   struct hash_table *outer_map = _mesa_pointer_hash_table_create(NULL);

   /* Re-query outer_pre because nir_push_if() inserted the guard if before
    * outer.loop and updated outer.hdr's predecessor to the post-if block. */
   nir_block *outer_pre =
      nir_cf_node_as_block(nir_cf_node_prev(&m->outer.loop->cf_node));
   nir_foreach_phi(phi, m->outer.hdr) {
      nir_def *init = phi_src_for_block(phi, outer_pre);
      assert(init != NULL);
      _mesa_hash_table_insert(outer_map, &phi->def, init);
   }

   for (unsigned ky = 0; ky < 3; ++ky) {
      clone_block_non_phi_instrs(b, m->outer.hdr, outer_map);
      if (m->outer_post_exit_blk != m->outer.hdr &&
          m->outer_post_exit_blk != m->inner_pre_blk)
         clone_block_non_phi_instrs(b, m->outer_post_exit_blk, outer_map);

      if (m->guard_if && !m->guard_if_is_outer_inv) {
         nir_def *gcond = lookup_remap(outer_map, m->guard_if->condition.ssa);
         nir_if *rif = nir_push_if(b, gcond);
         struct hash_table *inner_map = _mesa_pointer_hash_table_create(NULL);
         hash_table_foreach(outer_map, entry)
            _mesa_hash_table_insert(inner_map, entry->key, entry->data);
         emit_unrolled_inner_3(b, m, outer_map, inner_map);
         nir_pop_if(b, rif);

         nir_block *orig_then = nir_if_last_then_block(m->guard_if);
         nir_block *orig_else = nir_if_last_else_block(m->guard_if);
         nir_block *new_then = nir_if_last_then_block(rif);
         nir_block *new_else = nir_if_last_else_block(rif);
         nir_foreach_phi(phi, m->guard_merge_blk) {
            nir_def *tval =
               lookup_remap(inner_map, phi_src_for_block(phi, orig_then));
            nir_def *eval =
               lookup_remap(outer_map, phi_src_for_block(phi, orig_else));
            nir_phi_instr *nphi = nir_phi_instr_create(b->shader);
            nir_def_init(&nphi->instr, &nphi->def, phi->def.num_components,
                         phi->def.bit_size);
            nir_phi_instr_add_src(nphi, new_then, tval);
            nir_phi_instr_add_src(nphi, new_else, eval);
            nir_builder_instr_insert(b, &nphi->instr);
            _mesa_hash_table_insert(outer_map, &phi->def, &nphi->def);
         }
         _mesa_hash_table_destroy(inner_map, NULL);
      } else {
         struct hash_table *inner_map = _mesa_pointer_hash_table_create(NULL);
         hash_table_foreach(outer_map, entry)
            _mesa_hash_table_insert(inner_map, entry->key, entry->data);
         emit_unrolled_inner_3(b, m, outer_map, inner_map);
         hash_table_foreach(inner_map, entry)
            _mesa_hash_table_insert(outer_map, entry->key, entry->data);
         _mesa_hash_table_destroy(inner_map, NULL);

         if (m->guard_if) {
            nir_block *then_end = nir_if_last_then_block(m->guard_if);
            nir_foreach_phi(phi, m->guard_merge_blk) {
               nir_def *tval = phi_src_for_block(phi, then_end);
               assert(tval != NULL);
               _mesa_hash_table_insert(outer_map, &phi->def,
                                       lookup_remap(outer_map, tval));
            }
         }
      }

      if (m->outer.bottom_tested) {
         clone_block_non_phi_instrs(b, m->outer_pre_exit_blk, outer_map);
         if (ky < 2) {
            clone_block_non_phi_instrs(b, m->outer.latch, outer_map);
            nir_def *outer_next[DW3X3_MAX_OUTER_PHIS];
            unsigned pi = 0;
            nir_foreach_phi(phi, m->outer.hdr)
               outer_next[pi++] = lookup_remap(
                  outer_map, phi_src_for_block(phi, m->outer.latch));
            pi = 0;
            nir_foreach_phi(phi, m->outer.hdr)
               _mesa_hash_table_insert(outer_map, &phi->def, outer_next[pi++]);
         }
      } else {
         clone_block_non_phi_instrs(b, m->outer.latch, outer_map);
         nir_def *outer_next[DW3X3_MAX_OUTER_PHIS];
         unsigned pi = 0;
         nir_foreach_phi(phi, m->outer.hdr)
            outer_next[pi++] = lookup_remap(
               outer_map, phi_src_for_block(phi, m->outer.latch));
         pi = 0;
         nir_foreach_phi(phi, m->outer.hdr)
            _mesa_hash_table_insert(outer_map, &phi->def, outer_next[pi++]);
      }
   }

   for (unsigned i = 0; i < m->num_live_outs; ++i)
      then_live_outs[i] = lookup_remap(outer_map, m->live_outs[i]);

   _mesa_hash_table_destroy(outer_map, NULL);
}

static void
transform_dw3x3_nest(const struct dw3x3_match *m)
{
   nir_builder b = nir_builder_at(nir_before_cf_node(&m->outer.loop->cf_node));

   nir_def *ob_def =
      materialize_inv_bound(&b, m->outer.bound.def, m->outer.loop, 0);
   nir_def *outer_bound = ob_def->num_components == 1
      ? ob_def
      : nir_channel(&b, ob_def, m->outer.bound.comp);
   nir_def *ib_def =
      materialize_inv_bound(&b, m->inner.bound.def, m->outer.loop, 0);
   nir_def *inner_bound = ib_def->num_components == 1
      ? ib_def
      : nir_channel(&b, ib_def, m->inner.bound.comp);
   nir_def *cond = nir_iand(&b, nir_ieq_imm(&b, outer_bound, 3),
                            nir_ieq_imm(&b, inner_bound, 3));
   if (m->guard_if && m->guard_if_is_outer_inv)
      cond = nir_iand(&b, cond, m->guard_if->condition.ssa);

   nir_if *nif = nir_push_if(&b, cond);
   nir_def *then_outs[DW3X3_MAX_LIVE_OUTS];
   emit_unrolled_3x3(&b, m, then_outs);
   nir_pop_if(&b, nif);

   /* Move untouched original outer loop into nif->else_list. */
   nir_cf_list orig;
   nir_cf_extract(&orig, nir_before_cf_node(&m->outer.loop->cf_node),
                  nir_after_cf_node(&m->outer.loop->cf_node));
   nir_cf_reinsert(&orig, nir_after_cf_list(&nif->else_list));

   /* Merge live-out values after nif. */
   nir_block *then_end = nir_if_last_then_block(nif);
   nir_block *else_end = nir_if_last_else_block(nif);
   nir_block *after_blk = nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node));
   b.cursor = nir_before_block(after_blk);

   for (unsigned i = 0; i < m->num_live_outs; ++i) {
      nir_def *orig_def = m->live_outs[i];
      nir_phi_instr *phi = nir_phi_instr_create(b.shader);
      nir_def_init(&phi->instr, &phi->def, orig_def->num_components,
                   orig_def->bit_size);
      nir_phi_instr_add_src(phi, then_end, then_outs[i]);
      nir_phi_instr_add_src(phi, else_end, orig_def);
      nir_builder_instr_insert(&b, &phi->instr);

      nir_foreach_use_including_if_safe(src, orig_def) {
         if (!nir_src_is_if(src) && nir_src_use_instr(src) == &phi->instr)
            continue;
         nir_block *ublk = nir_src_is_if(src)
            ? nir_cf_node_as_block(
                 nir_cf_node_prev(&nir_src_use_if(src)->cf_node))
            : nir_src_use_instr(src)->block;
         if (!block_in_cf_node(ublk, &nif->cf_node))
            nir_src_rewrite(src, &phi->def);
      }
   }
}

static bool
scan_cf_list(struct exec_list *cf_list, struct hash_table *visited)
{
   foreach_list_typed_safe(nir_cf_node, node, node, cf_list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (scan_cf_list(&nif->then_list, visited))
            return true;
         if (scan_cf_list(&nif->else_list, visited))
            return true;
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         if (_mesa_hash_table_search(visited, loop))
            continue;
         struct dw3x3_match m;
         if (match_dw3x3_nest(loop, &m)) {
            _mesa_hash_table_insert(visited, loop, loop);
            transform_dw3x3_nest(&m);
            return true;
         }
         if (scan_cf_list(&loop->body, visited))
            return true;
      }
   }
   return false;
}

bool
agx_nir_opt_dw_conv_3x3(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_COMPUTE)
      return false;

   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      if (!impl->function->is_entrypoint) {
         nir_no_progress(impl);
         continue;
      }

      struct hash_table *visited = _mesa_pointer_hash_table_create(NULL);
      bool func_progress = false;

      while (true) {
         nir_metadata_require(impl, nir_metadata_dominance |
                                    nir_metadata_block_index);
         nir_divergence_analysis(nir);
         if (!scan_cf_list(&impl->body, visited))
            break;
         func_progress = true;
         nir_progress(true, impl, nir_metadata_none);
      }

      _mesa_hash_table_destroy(visited, NULL);
      progress |= nir_progress(func_progress, impl, nir_metadata_none);
   }

   return progress;
}

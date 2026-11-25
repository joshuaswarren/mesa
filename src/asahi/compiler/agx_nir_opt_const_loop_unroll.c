/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * Exact-divisor partial unrolling for compile-time constant-trip counted loops
 * that full loop unrolling left intact due to body size, plus monotonic 32-bit
 * bit-pack accumulator strength reduction.
 *
 * When a loop's exact trip count N is a compile-time multiple of F (F in
 * {4, 8}), replicating the loop body F times in place requires neither a
 * runtime guard nor a remainder tail loop.
 *
 * For a [0, 32) unit-stepped loop accumulating bits into a zero-initialized
 * 32-bit word via `d = cond ? (d | (1u << iv)) : d`, each F-bit chunk is
 * packed with small constant masks `1u << k` (0 <= k < F) and merged into `d`
 * with a single funnel shift `(d >> F) | (chunk << (32 - F))` (which lowers to
 * one native `extr_agx` on AGX/AGX3). After 32 / F iterations every bit lands
 * at its exact original position.
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_control_flow.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"

enum const_loop_rel {
   CL_REL_LT,
   CL_REL_LE,
   CL_REL_GT,
   CL_REL_GE,
};

struct const_counted_loop {
   nir_loop *loop;
   nir_if *nif;
   bool break_in_then;
   bool bottom;
   nir_phi_instr *iv;
   int64_t init_val;
   int64_t bound_val;
   int64_t step;
   enum const_loop_rel rel;
   bool is_unsigned;
   uint64_t trips;
};

static bool
block_is_lone_break(nir_block *block)
{
   nir_instr *last = nir_block_last_instr(block);
   return last && last == nir_block_first_instr(block) &&
          last->type == nir_instr_type_jump &&
          nir_instr_as_jump(last)->type == nir_jump_break;
}

static bool
branch_is_lone_break(struct exec_list *list)
{
   if (!exec_list_is_singular(list))
      return false;
   nir_cf_node *node =
      exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   return node->type == nir_cf_node_block &&
          block_is_lone_break(nir_cf_node_as_block(node));
}

static bool
branch_is_empty(struct exec_list *list)
{
   if (!exec_list_is_singular(list))
      return false;
   nir_cf_node *node =
      exec_node_data(nir_cf_node, exec_list_get_head(list), node);
   return node->type == nir_cf_node_block &&
          exec_list_is_empty(&nir_cf_node_as_block(node)->instr_list);
}

static bool
block_in_loop(nir_block *block, nir_loop *loop)
{
   for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent) {
      if (n == &loop->cf_node)
         return true;
   }
   return false;
}

static nir_block *
src_use_block(nir_src *src)
{
   if (nir_src_is_if(src))
      return nir_cf_node_as_block(
         nir_cf_node_prev(&nir_src_use_if(src)->cf_node));
   return nir_src_use_instr(src)->block;
}

static bool
lane_dependent_intr(const nir_intrinsic_instr *intr)
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
   case nir_intrinsic_demote_samples:
   case nir_intrinsic_intersect_ray_agx:
   case nir_intrinsic_continue_ray_agx:
   case nir_intrinsic_load_scs_uniform_agx:
   case nir_intrinsic_store_scs_uniform_agx:
   case nir_intrinsic_mesh_payload_entry_barrier_agx:
   case nir_intrinsic_object_payload_exit_barrier_agx:
   case nir_intrinsic_fence_mesh_uvb_agx:
   case nir_intrinsic_cmat_muladd_agx:
   case nir_intrinsic_barrier:
   case nir_intrinsic_terminate:
   case nir_intrinsic_terminate_if:
   case nir_intrinsic_demote:
   case nir_intrinsic_demote_if:
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy_coarse:
      return true;
   default:
      return false;
   }
}

static int
count_unrollable_body(nir_loop *loop, nir_if *terminator, bool *has_load_out)
{
   int count = 0;
   bool has_load = false;

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_jump:
            if (block->cf_node.parent != &terminator->cf_node)
               return -1;
            break;
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (lane_dependent_intr(intr))
               return -1;
            switch (intr->intrinsic) {
            case nir_intrinsic_load_global:
            case nir_intrinsic_load_global_constant:
            case nir_intrinsic_load_ssbo:
            case nir_intrinsic_load_shared:
            case nir_intrinsic_load_agx:
               has_load = true;
               break;
            default:
               break;
            }
            break;
         }
         case nir_instr_type_tex:
            if (nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr)))
               return -1;
            has_load = true;
            break;
         case nir_instr_type_call:
            return -1;
         default:
            break;
         }
         if (instr->type != nir_instr_type_phi &&
             instr->type != nir_instr_type_load_const &&
             !(instr->type == nir_instr_type_intrinsic &&
               nir_instr_as_intrinsic(instr)->intrinsic ==
                  nir_intrinsic_load_preamble))
            count++;
      }
   }

   *has_load_out = has_load;
   return count;
}

static bool
compute_top_trips(int64_t init, int64_t bound, int64_t step,
                  enum const_loop_rel rel, bool is_unsigned, uint64_t *trips_out)
{
   if (step == 0)
      return false;

   if (is_unsigned) {
      uint64_t uinit = (uint32_t)init;
      uint64_t ubound = (uint32_t)bound;
      if (step > 0) {
         uint64_t ustep = (uint64_t)step;
         if (rel == CL_REL_LT) {
            if (uinit >= ubound)
               return false;
            uint64_t diff = ubound - uinit;
            uint64_t n = (diff - 1) / ustep + 1;
            if (uinit + n * ustep > UINT32_MAX)
               return false;
            *trips_out = n;
            return true;
         } else if (rel == CL_REL_LE) {
            if (uinit > ubound)
               return false;
            uint64_t diff = ubound - uinit;
            uint64_t n = diff / ustep + 1;
            if (uinit + n * ustep > UINT32_MAX)
               return false;
            *trips_out = n;
            return true;
         }
         return false;
      } else {
         uint64_t ustep = (uint64_t)(-step);
         if (rel == CL_REL_GT) {
            if (uinit <= ubound)
               return false;
            uint64_t diff = uinit - ubound;
            uint64_t n = (diff - 1) / ustep + 1;
            if (n * ustep > uinit)
               return false;
            *trips_out = n;
            return true;
         } else if (rel == CL_REL_GE) {
            if (uinit < ubound)
               return false;
            uint64_t diff = uinit - ubound;
            uint64_t n = diff / ustep + 1;
            if (n * ustep > uinit)
               return false;
            *trips_out = n;
            return true;
         }
         return false;
      }
   } else {
      if (step > 0) {
         if (rel == CL_REL_LT) {
            if (init >= bound)
               return false;
            uint64_t diff = (uint64_t)(bound - init);
            uint64_t n = (diff - 1) / (uint64_t)step + 1;
            if (init + (int64_t)(n * (uint64_t)step) > INT32_MAX)
               return false;
            *trips_out = n;
            return true;
         } else if (rel == CL_REL_LE) {
            if (init > bound)
               return false;
            uint64_t diff = (uint64_t)(bound - init);
            uint64_t n = diff / (uint64_t)step + 1;
            if (init + (int64_t)(n * (uint64_t)step) > INT32_MAX)
               return false;
            *trips_out = n;
            return true;
         }
         return false;
      } else {
         uint64_t ustep = (uint64_t)(-step);
         if (rel == CL_REL_GT) {
            if (init <= bound)
               return false;
            uint64_t diff = (uint64_t)(init - bound);
            uint64_t n = (diff - 1) / ustep + 1;
            if (init - (int64_t)(n * ustep) < INT32_MIN)
               return false;
            *trips_out = n;
            return true;
         } else if (rel == CL_REL_GE) {
            if (init < bound)
               return false;
            uint64_t diff = (uint64_t)(init - bound);
            uint64_t n = diff / ustep + 1;
            if (init - (int64_t)(n * ustep) < INT32_MIN)
               return false;
            *trips_out = n;
            return true;
         }
         return false;
      }
   }
}

static bool
analyze_const_counted_loop(nir_loop *loop, struct const_counted_loop *out)
{
   if (!nir_loop_has_back_edge(loop) || nir_loop_has_continue_construct(loop) ||
       loop->partially_unrolled ||
       loop->control == nir_loop_control_dont_unroll)
      return false;

   nir_block *preheader = nir_loop_predecessor_block(loop);
   nir_block *header = nir_loop_first_block(loop);
   nir_block *latch = nir_loop_last_block(loop);
   if (nir_block_num_preds(header) != 2 || nir_block_ends_in_jump(latch))
      return false;

   /* Innermost loops only. */
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      for (nir_cf_node *p = block->cf_node.parent; p != &loop->cf_node;
           p = p->parent) {
         if (p->type == nir_cf_node_loop)
            return false;
      }
   }

   /* Find single exit at bottom or top. */
   nir_cf_node *first = nir_cf_node_next(&header->cf_node);
   nir_cf_node *last_if = nir_cf_node_prev(&latch->cf_node);
   nir_if *nif = NULL;
   bool bottom = false;
   bool break_in_then = false;

   for (unsigned pass = 0; pass < 2 && !nif; ++pass) {
      nir_cf_node *node = pass == 0 ? last_if : first;
      if (!node || node->type != nir_cf_node_if)
         continue;
      if (pass == 0 && !exec_list_is_empty(&latch->instr_list))
         continue;
      nir_if *cand = nir_cf_node_as_if(node);
      if (branch_is_lone_break(&cand->then_list) &&
          branch_is_empty(&cand->else_list))
         break_in_then = true;
      else if (branch_is_lone_break(&cand->else_list) &&
               branch_is_empty(&cand->then_list))
         break_in_then = false;
      else
         continue;
      nif = cand;
      bottom = (pass == 0);
   }
   if (!nif)
      return false;

   /* Top-tested loops cannot have side effects in header before the exit. */
   if (!bottom) {
      nir_foreach_instr(instr, header) {
         if (instr->type == nir_instr_type_intrinsic &&
             (!(nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic].flags &
                NIR_INTRINSIC_CAN_ELIMINATE) ||
              (nir_intrinsic_has_access(nir_instr_as_intrinsic(instr)) &&
               (nir_intrinsic_access(nir_instr_as_intrinsic(instr)) &
                ACCESS_VOLATILE))))
            return false;
         if (instr->type == nir_instr_type_call)
            return false;
      }

      /* For top-tested loops, keep only loops whose live-outs are header phis
       * so in-place unrolling preserves dominance of every outside use. */
      nir_foreach_block_in_cf_node(block, &loop->cf_node) {
         nir_foreach_instr(instr, block) {
            nir_def *def = nir_instr_def(instr);
            if (!def)
               continue;
            if (instr->type == nir_instr_type_phi && block == header)
               continue;
            nir_foreach_use_including_if(src, def) {
               if (!block_in_loop(src_use_block(src), loop))
                  return false;
            }
         }
      }
   }

   nir_scalar cond = nir_scalar_resolved(nif->condition.ssa, 0);
   bool negate = false;
   while (nir_scalar_is_alu(cond) && nir_scalar_alu_op(cond) == nir_op_inot) {
      negate = !negate;
      cond = nir_scalar_chase_alu_src(cond, 0);
   }
   if (!nir_scalar_is_alu(cond))
      return false;

   nir_op op = nir_scalar_alu_op(cond);
   if (op != nir_op_ilt && op != nir_op_ige &&
       op != nir_op_ult && op != nir_op_uge)
      return false;

   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 0));
   nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 1));
   if (a.def->bit_size != 32 || a.def->num_components != 1 ||
       b.def->bit_size != 32 || b.def->num_components != 1)
      return false;

   bool iv_left = false;
   nir_scalar ivs = {0}, bound = {0};
   int64_t offset = 0;

   for (unsigned side = 0; side < 2 && !ivs.def; ++side) {
      nir_scalar x = side == 0 ? a : b;
      nir_scalar y = side == 0 ? b : a;
      if (!nir_scalar_is_const(y))
         continue;
      int64_t c = 0;
      if (nir_scalar_is_alu(x) && nir_scalar_alu_op(x) == nir_op_iadd) {
         for (unsigned k = 0; k < 2; ++k) {
            nir_scalar kk = nir_scalar_chase_movs(nir_scalar_chase_alu_src(x, k));
            nir_scalar other =
               nir_scalar_chase_movs(nir_scalar_chase_alu_src(x, 1 - k));
            if (nir_scalar_is_const(kk) && other.comp == 0 &&
                nir_def_instr(other.def)->type == nir_instr_type_phi) {
               c = nir_scalar_as_int(kk);
               x = other;
               break;
            }
         }
      }
      if (x.comp == 0 && nir_def_instr(x.def)->type == nir_instr_type_phi &&
          nir_def_instr(x.def)->block == header && x.def->bit_size == 32 &&
          c >= -0x10000 && c <= 0x10000) {
         ivs = x;
         bound = y;
         offset = c;
         iv_left = (side == 0);
      }
   }
   if (!ivs.def)
      return false;

   nir_phi_instr *phi = nir_instr_as_phi(nir_def_instr(ivs.def));
   nir_phi_src *init_src = nir_phi_get_src_from_block(phi, preheader);
   nir_phi_src *back_src = nir_phi_get_src_from_block(phi, latch);
   if (!init_src || !back_src)
      return false;

   nir_scalar init_s = nir_scalar_resolved(init_src->src.ssa, 0);
   if (!nir_scalar_is_const(init_s))
      return false;

   nir_scalar update = nir_scalar_resolved(back_src->src.ssa, 0);
   if (!nir_scalar_is_alu(update) || nir_scalar_alu_op(update) != nir_op_iadd)
      return false;

   int64_t step = 0;
   bool found_step = false;
   for (unsigned s = 0; s < 2; ++s) {
      nir_scalar src = nir_scalar_chase_movs(nir_scalar_chase_alu_src(update, s));
      nir_scalar other =
         nir_scalar_chase_movs(nir_scalar_chase_alu_src(update, 1 - s));
      if (other.def == &phi->def && other.comp == 0 && nir_scalar_is_const(src)) {
         step = nir_scalar_as_int(src);
         found_step = true;
      }
   }
   if (!found_step || step == 0 || step > 0x10000 || step < -0x10000)
      return false;

   /* Require test offset == 0 for top-tested, offset == step for bottom-tested. */
   if (!bottom && offset != 0)
      return false;
   if (bottom && offset != step)
      return false;

   bool break_on_true = (break_in_then != negate);
   bool less = (op == nir_op_ilt || op == nir_op_ult);
   bool cont_less = break_on_true ? !less : less;
   enum const_loop_rel rel;
   if (iv_left)
      rel = cont_less ? CL_REL_LT : CL_REL_GE;
   else
      rel = cont_less ? CL_REL_GT : CL_REL_LE;

   if ((rel == CL_REL_LT || rel == CL_REL_LE) != (step > 0))
      return false;

   const bool is_unsigned = (op == nir_op_ult || op == nir_op_uge);
   const int64_t init_val =
      is_unsigned ? (int64_t)nir_scalar_as_uint(init_s)
                  : nir_scalar_as_int(init_s);
   const int64_t bound_val =
      is_unsigned ? (int64_t)nir_scalar_as_uint(bound)
                  : nir_scalar_as_int(bound);

   uint64_t trips = 0;
   if (!bottom) {
      if (!compute_top_trips(init_val, bound_val, step, rel, is_unsigned, &trips))
         return false;
   } else {
      int64_t next_init = init_val + step;
      if (is_unsigned) {
         if (next_init < 0 || next_init > (int64_t)UINT32_MAX)
            return false;
      } else {
         if (next_init < INT32_MIN || next_init > INT32_MAX)
            return false;
      }
      uint64_t more = 0;
      if (!compute_top_trips(next_init, bound_val, step, rel, is_unsigned, &more))
         return false;
      trips = 1 + more;
   }

   if (trips < 4 || trips > 64)
      return false;

   *out = (struct const_counted_loop){
      .loop = loop,
      .nif = nif,
      .break_in_then = break_in_then,
      .bottom = bottom,
      .iv = phi,
      .init_val = init_val,
      .bound_val = bound_val,
      .step = step,
      .rel = rel,
      .is_unsigned = is_unsigned,
      .trips = trips,
   };
   return true;
}

static unsigned
choose_exact_factor(const struct const_counted_loop *cl, unsigned max_factor)
{
   bool has_load = false;
   int size = count_unrollable_body(cl->loop, cl->nif, &has_load);
   if (size < 20 || !has_load)
      return 0;

   static const unsigned factors[] = {8, 4};
   for (unsigned i = 0; i < ARRAY_SIZE(factors); ++i) {
      unsigned f = factors[i];
      if (f <= max_factor && cl->trips >= f && (cl->trips % f) == 0 &&
          (unsigned)size * f <= 720)
         return f;
   }
   return 0;
}

static void
unroll_const_loop_in_place(const struct const_counted_loop *cl, unsigned factor)
{
   nir_loop *loop = cl->loop;
   nir_block *header = nir_loop_first_block(loop);

   nir_lower_phis_to_regs_block(header, true);
   loop->partially_unrolled = true;

   if (cl->bottom) {
      /* Structure is: [body_part] -> bottom_if { break } -> [latch_stores].
       * Insert (factor - 1) clones of [body_part, latch_stores] BEFORE the
       * original [body_part], keeping original [body_part, bottom_if,
       * latch_stores] as the final unrolled iteration (k = factor - 1).
       * Any outside-of-loop use of a body def still reads copy (factor - 1). */
      nir_cf_list body_part, latch_stores;
      nir_cf_extract(&body_part, nir_before_cf_list(&loop->body),
                     nir_before_cf_node(&cl->nif->cf_node));
      nir_cf_extract(&latch_stores, nir_after_cf_node(&cl->nif->cf_node),
                     nir_after_cf_list(&loop->body));

      nir_cf_list copies[2 * 16];
      assert(factor <= 16);
      for (unsigned i = 0; i + 1 < factor; ++i) {
         struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
         nir_cf_list_clone(&copies[2 * i], &body_part, &loop->cf_node, remap);
         nir_cf_list_clone(&copies[2 * i + 1], &latch_stores, &loop->cf_node,
                           remap);
         _mesa_hash_table_destroy(remap, NULL);
      }

      for (unsigned i = 0; i + 1 < factor; ++i) {
         nir_cf_reinsert(&copies[2 * i], nir_before_cf_node(&cl->nif->cf_node));
         nir_cf_reinsert(&copies[2 * i + 1],
                         nir_before_cf_node(&cl->nif->cf_node));
      }
      nir_cf_reinsert(&body_part, nir_before_cf_node(&cl->nif->cf_node));
      nir_cf_reinsert(&latch_stores, nir_after_cf_node(&cl->nif->cf_node));
   } else {
      /* Structure is: [head] -> top_if { break } -> [rest].
       * Keep head and top_if once at the top, followed by rest and
       * (factor - 1) clones of [head, rest]. */
      nir_cf_list head, rest;
      nir_cf_extract(&head, nir_before_cf_list(&loop->body),
                     nir_before_cf_node(&cl->nif->cf_node));
      nir_cf_extract(&rest, nir_after_cf_node(&cl->nif->cf_node),
                     nir_after_cf_list(&loop->body));

      nir_cf_list copies[2 * 16];
      assert(factor <= 16);
      for (unsigned i = 1; i < factor; ++i) {
         struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
         nir_cf_list_clone(&copies[2 * i], &head, &loop->cf_node, remap);
         nir_cf_list_clone(&copies[2 * i + 1], &rest, &loop->cf_node, remap);
         _mesa_hash_table_destroy(remap, NULL);
      }

      nir_cf_reinsert(&head, nir_before_cf_node(&cl->nif->cf_node));
      nir_cf_reinsert(&rest, nir_after_cf_list(&loop->body));
      for (unsigned i = 1; i < factor; ++i) {
         nir_cf_reinsert(&copies[2 * i], nir_after_cf_list(&loop->body));
         nir_cf_reinsert(&copies[2 * i + 1], nir_after_cf_list(&loop->body));
      }
   }
}

/* Match `s == iv_def + const_off` through a chain of scalar iadd-by-const ops. */
static bool
match_iv_plus_const(nir_scalar s, nir_def *iv_def, int64_t *out_off,
                    unsigned depth)
{
   s = nir_scalar_chase_movs(s);
   if (s.def == iv_def && s.comp == 0) {
      *out_off = 0;
      return true;
   }
   if (!depth || !nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_iadd)
      return false;

   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar c = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, i));
      nir_scalar o = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1 - i));
      if (nir_scalar_is_const(c)) {
         int64_t sub = 0;
         if (match_iv_plus_const(o, iv_def, &sub, depth - 1)) {
            *out_off = sub + nir_scalar_as_int(c);
            return true;
         }
      }
   }
   return false;
}

static bool
match_bit_mask_for_offset(nir_scalar s, nir_def *iv_def, int64_t expected_off)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_ishl)
      return false;

   nir_scalar base = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   nir_scalar sh = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
   if (!nir_scalar_is_const(base) || nir_scalar_as_uint(base) != 1)
      return false;

   if (nir_scalar_is_alu(sh) && nir_scalar_alu_op(sh) == nir_op_iand) {
      nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(sh, 0));
      nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(sh, 1));
      if (nir_scalar_is_const(b) && nir_scalar_as_uint(b) == 31)
         sh = a;
      else if (nir_scalar_is_const(a) && nir_scalar_as_uint(a) == 31)
         sh = b;
   }

   int64_t off = 0;
   return match_iv_plus_const(sh, iv_def, &off, 16) && off == expected_off;
}

/* Match `ior(prev, 1u << (iv + expected_off))` (either operand order). */
static bool
match_ior_bit(nir_scalar s, nir_def *iv_def, int64_t expected_off,
              nir_def **prev_out, nir_alu_instr **ior_out)
{
   s = nir_scalar_chase_movs(s);
   if (!nir_scalar_is_alu(s) || nir_scalar_alu_op(s) != nir_op_ior)
      return false;
   nir_alu_instr *ior = nir_def_as_alu(s.def);
   if (ior->def.num_components != 1 || ior->def.bit_size != 32)
      return false;

   nir_scalar a = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 0));
   nir_scalar b = nir_scalar_chase_movs(nir_scalar_chase_alu_src(s, 1));
   if (match_bit_mask_for_offset(b, iv_def, expected_off) && a.comp == 0) {
      *prev_out = a.def;
      *ior_out = ior;
      return true;
   }
   if (match_bit_mask_for_offset(a, iv_def, expected_off) && b.comp == 0) {
      *prev_out = b.def;
      *ior_out = ior;
      return true;
   }
   return false;
}

/* After unrolling a [0, 32) unit-stepped loop by an exact divisor F of 32,
 * rewrite any 0-initialized 32-bit bit-pack accumulator phi `d` whose F steps
 * each conditionally OR `1u << (iv + k)` into an F-bit constant-mask chunk
 * merged with one funnel shift `(d >> F) | (chunk << (32 - F))`. */
static bool
strength_reduce_unrolled_bitpack(nir_function_impl *impl, nir_loop *loop,
                                 unsigned factor)
{
   if (factor < 2 || factor > 16 || (32u % factor) != 0)
      return false;

   nir_block *preheader = nir_loop_predecessor_block(loop);
   nir_block *header = nir_loop_first_block(loop);
   nir_block *latch = nir_loop_last_block(loop);

   /* Locate the unrolled IV phi (init == 0, back == iv + factor). */
   nir_phi_instr *iv_phi = NULL;
   nir_foreach_phi(phi, header) {
      if (phi->def.num_components != 1 || phi->def.bit_size != 32)
         continue;
      nir_phi_src *init_s = nir_phi_get_src_from_block(phi, preheader);
      nir_phi_src *back_s = nir_phi_get_src_from_block(phi, latch);
      if (!init_s || !back_s)
         continue;
      nir_scalar is = nir_scalar_resolved(init_s->src.ssa, 0);
      if (!nir_scalar_is_const(is) || nir_scalar_as_uint(is) != 0)
         continue;
      int64_t step = 0;
      if (match_iv_plus_const(nir_scalar_resolved(back_s->src.ssa, 0),
                              &phi->def, &step, 16) &&
          step == (int64_t)factor) {
         iv_phi = phi;
         break;
      }
   }
   if (!iv_phi)
      return false;

   bool progress = false;
   nir_builder b = nir_builder_create(impl);

   nir_foreach_phi(d_phi, header) {
      if (d_phi == iv_phi || d_phi->def.num_components != 1 ||
          d_phi->def.bit_size != 32)
         continue;

      nir_phi_src *init_s = nir_phi_get_src_from_block(d_phi, preheader);
      nir_phi_src *back_s = nir_phi_get_src_from_block(d_phi, latch);
      if (!init_s || !back_s)
         continue;
      nir_scalar is = nir_scalar_resolved(init_s->src.ssa, 0);
      if (!nir_scalar_is_const(is) || nir_scalar_as_uint(is) != 0)
         continue;

      nir_def *d_back = back_s->src.ssa;
      nir_block *body_blk = nir_def_block(d_back);
      if (!block_in_loop(body_blk, loop))
         continue;

      nir_def *conds[16];
      bool invert[16];
      nir_alu_instr *sels[16];
      nir_alu_instr *iors[16];
      nir_def *cur = d_back;
      bool ok = true;

      for (int k = (int)factor - 1; k >= 0; --k) {
         nir_instr *ci = nir_def_instr(cur);
         if (ci->block != body_blk || ci->type != nir_instr_type_alu) {
            ok = false;
            break;
         }
         nir_alu_instr *sel = nir_instr_as_alu(ci);
         if (sel->op != nir_op_bcsel || sel->def.num_components != 1 ||
             sel->def.bit_size != 32) {
            ok = false;
            break;
         }

         nir_scalar c = nir_scalar_chase_movs(
            nir_get_scalar(sel->src[0].src.ssa, sel->src[0].swizzle[0]));
         nir_scalar t = nir_scalar_chase_movs(
            nir_get_scalar(sel->src[1].src.ssa, sel->src[1].swizzle[0]));
         nir_scalar f = nir_scalar_chase_movs(
            nir_get_scalar(sel->src[2].src.ssa, sel->src[2].swizzle[0]));
         if (c.def->bit_size != 1 || c.comp != 0) {
            ok = false;
            break;
         }

         nir_def *prev = NULL;
         nir_alu_instr *ior = NULL;
         bool inv = false;
         if (match_ior_bit(t, &iv_phi->def, k, &prev, &ior) &&
             f.def == prev && f.comp == 0) {
            inv = false;
         } else if (match_ior_bit(f, &iv_phi->def, k, &prev, &ior) &&
                    t.def == prev && t.comp == 0) {
            inv = true;
         } else {
            ok = false;
            break;
         }

         /* Intermediate results must not escape the chain. */
         if (k < (int)factor - 1) {
            nir_foreach_use_including_if(u, cur) {
               if (nir_src_is_if(u)) {
                  ok = false;
                  break;
               }
               nir_instr *ui = nir_src_use_instr(u);
               if (ui != &sels[k + 1]->instr && ui != &iors[k + 1]->instr) {
                  ok = false;
                  break;
               }
            }
            if (!ok)
               break;
         }

         conds[k] = c.def;
         invert[k] = inv;
         sels[k] = sel;
         iors[k] = ior;
         cur = prev;
      }

      if (!ok || cur != &d_phi->def)
         continue;

      /* `d_phi` itself must not be read inside the loop outside step 0. */
      nir_foreach_use_including_if(u, &d_phi->def) {
         if (nir_src_is_if(u)) {
            if (block_in_loop(src_use_block(u), loop)) {
               ok = false;
               break;
            }
            continue;
         }
         nir_instr *ui = nir_src_use_instr(u);
         if (block_in_loop(ui->block, loop) &&
             ui != &sels[0]->instr && ui != &iors[0]->instr) {
            ok = false;
            break;
         }
      }
      if (!ok)
         continue;

      /* Also check `d_back` has no in-loop uses other than `d_phi`. */
      nir_foreach_use_including_if(u, d_back) {
         if (nir_src_is_if(u)) {
            if (block_in_loop(src_use_block(u), loop)) {
               ok = false;
               break;
            }
            continue;
         }
         nir_instr *ui = nir_src_use_instr(u);
         if (block_in_loop(ui->block, loop) && ui != &d_phi->instr) {
            ok = false;
            break;
         }
      }
      if (!ok)
         continue;

      b.cursor = nir_before_instr(&sels[0]->instr);
      nir_def *chunk = invert[0]
                          ? nir_bcsel(&b, conds[0], nir_imm_int(&b, 0),
                                      nir_imm_int(&b, 1))
                          : nir_bcsel(&b, conds[0], nir_imm_int(&b, 1),
                                      nir_imm_int(&b, 0));
      for (unsigned k = 1; k < factor; ++k) {
         b.cursor = nir_before_instr(&sels[k]->instr);
         nir_def *bit = invert[k]
                           ? nir_bcsel(&b, conds[k], nir_imm_int(&b, 0),
                                       nir_imm_int(&b, 1u << k))
                           : nir_bcsel(&b, conds[k], nir_imm_int(&b, 1u << k),
                                       nir_imm_int(&b, 0));
         chunk = nir_ior(&b, chunk, bit);
      }

      b.cursor = nir_before_instr(&sels[factor - 1]->instr);
      nir_def *d_new = nir_ior(&b, nir_ushr_imm(&b, &d_phi->def, factor),
                               nir_ishl_imm(&b, chunk, 32 - factor));
      nir_def_replace(d_back, d_new);
      progress = true;
   }

   return progress;
}

static void
collect_loops_rec(struct exec_list *list, struct util_dynarray *loops)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         collect_loops_rec(&nif->then_list, loops);
         collect_loops_rec(&nif->else_list, loops);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         collect_loops_rec(&loop->body, loops);
         util_dynarray_append(loops, loop);
         break;
      }
      default:
         break;
      }
   }
}

bool
agx_nir_opt_const_loop_unroll(nir_shader *shader, unsigned max_factor,
                              unsigned *widest)
{
   if (max_factor < 4 || shader->info.cs.has_cooperative_matrix)
      return false;

   bool any = false;

   nir_foreach_function_impl(impl, shader) {
      nir_metadata_require(impl, nir_metadata_block_index);
      nir_index_ssa_defs(impl);

      unsigned n = 0;
      while (n < 32) {
         struct util_dynarray loops;
         util_dynarray_init(&loops, NULL);
         collect_loops_rec(&impl->body, &loops);

         struct const_counted_loop cl;
         unsigned f = 0;
         util_dynarray_foreach(&loops, nir_loop *, lp) {
            if (analyze_const_counted_loop(*lp, &cl) &&
                (f = choose_exact_factor(&cl, max_factor)))
               break;
         }
         util_dynarray_fini(&loops);
         if (!f)
            break;

         if (widest)
            *widest = MAX2(*widest, f);

         const bool maybe_bitpack =
            (cl.init_val == 0 && cl.step == 1 && cl.trips == 32);

         unroll_const_loop_in_place(&cl, f);
         nir_progress(true, impl, nir_metadata_none);
         nir_lower_reg_intrinsics_to_ssa_impl(impl);
         nir_metadata_require(impl, nir_metadata_block_index);
         nir_index_ssa_defs(impl);

         if (maybe_bitpack)
            strength_reduce_unrolled_bitpack(impl, cl.loop, f);

         n++;
      }

      if (n) {
         any = true;
         nir_progress(true, impl, nir_metadata_none);
      } else {
         nir_no_progress(impl);
      }
   }

   return any;
}

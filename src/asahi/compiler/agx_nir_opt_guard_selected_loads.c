/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
#include "util/u_dynarray.h"

/*
 * Two general control-flow / memory-access optimizations for AGX:
 *
 * 1. agx_nir_opt_guard_selected_loads:
 *    When a chain of scalar selects yields a fallback value whenever a base
 *    value already equals that fallback:
 *
 *       r = bcsel(c1, bcsel(c2, base_val, fallback), fallback)
 *
 *    then whenever bitwise (base_val == fallback), r is unconditionally equal
 *    to fallback regardless of c1, c2, and any instructions used exclusively
 *    to compute them. If that exclusive backward slice inside the same basic
 *    block contains at least two readonly, reorderable device loads (and no
 *    side effects, barriers, subgroup ops, or derivatives), wrap the slice in
 *    `if (base_val != fallback)` so invocations whose base value already
 *    matches the fallback skip those loads and address calculations.
 *
 * 2. agx_nir_opt_forward_merge_loads:
 *    Forward non-volatile, non-coherent raw-format store_agx values to
 *    matching load_agx instructions instead of round-tripping through global
 *    memory:
 *    - Within a single basic block, track the latest store_agx in a linear
 *      forward walk and forward it to matching load_agx instructions with no
 *      intervening aliasing store or synchronization instruction, and into
 *      the unaliased prefix of an immediately following `if` arm.
 *    - Across an if/else merge, when both arms end with matching store_agx
 *      instructions to the exact same base pointer, format, and canonical
 *      index expression, and the merge block re-reads that location before
 *      any intervening memory write or barrier, forward via a phi.
 */

static bool
is_readonly_reorderable_load(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic == nir_intrinsic_load_constant_agx)
      return true;

   if (intr->intrinsic == nir_intrinsic_load_agx) {
      enum gl_access_qualifier access = nir_intrinsic_access(intr);
      return (access & ACCESS_CAN_REORDER) && (access & ACCESS_NON_WRITEABLE);
   }

   return false;
}

static bool
instr_safe_for_guarded_slice(const nir_instr *instr)
{
   if (instr->type == nir_instr_type_alu ||
       instr->type == nir_instr_type_load_const)
      return true;

   return is_readonly_reorderable_load(instr);
}

static bool
all_uses_in_slice(const nir_def *def, const BITSET_WORD *in_slice,
                  unsigned max_instr_idx)
{
   nir_foreach_use_including_if(use, def) {
      if (nir_src_is_if(use))
         return false;

      nir_instr *user = nir_src_use_instr(use);
      if (user->index >= max_instr_idx || !BITSET_TEST(in_slice, user->index))
         return false;
   }
   return true;
}

static bool
try_guard_select_in_block(nir_builder *b, nir_block *block,
                          unsigned max_instr_idx, BITSET_WORD *in_slice,
                          struct set *guarded_roots)
{
   /* Walk backward so we catch the outermost select of a chain first. */
   nir_foreach_instr_reverse(instr, block) {
      if (instr->type != nir_instr_type_alu)
         continue;

      nir_alu_instr *root = nir_instr_as_alu(instr);
      if (root->op != nir_op_bcsel || root->def.num_components != 1 ||
          root->def.bit_size > 32 || _mesa_set_search(guarded_roots, root))
         continue;

      /* Identify a chain of selects sharing one fallback arm:
       *   bcsel(c1, bcsel(c2, ..., base_val, fallback), fallback)
       * or with the fallback in src[1] at any level. */
      for (unsigned fb_side = 2; fb_side >= 1; --fb_side) {
         nir_scalar fb_sc =
            nir_get_scalar(root->src[fb_side].src.ssa, root->src[fb_side].swizzle[0]);
         nir_scalar cur = nir_get_scalar(root->src[3 - fb_side].src.ssa,
                                         root->src[3 - fb_side].swizzle[0]);

         /* Collect intermediate selects in the chain. */
         nir_alu_instr *chain[8];
         unsigned chain_len = 0;
         chain[chain_len++] = root;

         while (chain_len < ARRAY_SIZE(chain) &&
                cur.def->num_components == 1 &&
                nir_def_block(cur.def) == block &&
                nir_scalar_is_alu(cur) &&
                nir_scalar_alu_op(cur) == nir_op_bcsel &&
                list_is_singular(&cur.def->uses)) {
            nir_alu_instr *inner = nir_def_as_alu(cur.def);
            nir_scalar s1 =
               nir_get_scalar(inner->src[1].src.ssa, inner->src[1].swizzle[0]);
            nir_scalar s2 =
               nir_get_scalar(inner->src[2].src.ssa, inner->src[2].swizzle[0]);

            if (nir_scalar_equal(s2, fb_sc)) {
               chain[chain_len++] = inner;
               cur = s1;
            } else if (nir_scalar_equal(s1, fb_sc)) {
               chain[chain_len++] = inner;
               cur = s2;
            } else {
               break;
            }
         }

         /* A lone select has no shared fallback. Guarding it can turn a
          * predicate such as "set this bit" into an extra divergent branch
          * around the loads that produce the condition. */
         if (chain_len < 2)
            continue;

         nir_def *base_val = cur.def;
         nir_def *fallback = fb_sc.def;
         if (base_val->num_components != 1 || fallback->num_components != 1 ||
             base_val->bit_size != root->def.bit_size ||
             fallback->bit_size != root->def.bit_size ||
             base_val == fallback)
            continue;

         /* Avoid guarding when base_val is already a constant (nothing to skip
          * dynamically unless base_val varies per invocation). */
         if (nir_def_instr(base_val)->type == nir_instr_type_load_const)
            continue;

         /* Close the exclusive backward slice in this block. */
         memset(in_slice, 0, BITSET_BYTES(max_instr_idx));
         for (unsigned i = 0; i < chain_len; ++i)
            BITSET_SET(in_slice, chain[i]->instr.index);

         unsigned slice_loads = 0;
         unsigned slice_count = chain_len;
         nir_foreach_instr_reverse(cand, block) {
            if (cand->index >= max_instr_idx)
               continue;
            if (BITSET_TEST(in_slice, cand->index))
               continue;
            if (!instr_safe_for_guarded_slice(cand))
               continue;

            nir_def *def = nir_instr_def(cand);
            if (!def || def == base_val || def == fallback)
               continue;

            if (!all_uses_in_slice(def, in_slice, max_instr_idx))
               continue;

            BITSET_SET(in_slice, cand->index);
            slice_count++;
            if (is_readonly_reorderable_load(cand))
               slice_loads++;
         }

         if (slice_loads < 2 || slice_count > 128)
            continue;

         /* Collect slice instructions in forward program order. */
         nir_instr *ordered[128];
         unsigned n_ordered = 0;
         nir_foreach_instr(it, block) {
            if (it->index < max_instr_idx && BITSET_TEST(in_slice, it->index))
               ordered[n_ordered++] = it;
         }
         assert(n_ordered == slice_count);

         /* Insert `if (base_val != fallback)` right before `root`, move the
          * exclusive slice into the `then` branch, and merge via a phi. */
         b->cursor = nir_before_instr(&root->instr);
         nir_def *guard = nir_ine(b, base_val, fallback);
         nir_if *nif = nir_push_if(b, guard);
         nir_block *then_block = nir_if_first_then_block(nif);
         for (unsigned i = 0; i < n_ordered; ++i)
            nir_instr_move(nir_after_block(then_block), ordered[i]);
         nir_pop_if(b, nif);

         nir_def *merged = nir_if_phi(b, &root->def, fallback);
         nir_instr *phi_instr = nir_def_instr(merged);
         nir_foreach_use_including_if_safe(use, &root->def) {
            if (!nir_src_is_if(use) && nir_src_use_instr(use) == phi_instr)
               continue;
            nir_src_rewrite(use, merged);
         }
         _mesa_set_add(guarded_roots, root);
         return true;
      }
   }

   return false;
}

bool
agx_nir_opt_guard_selected_loads(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      bool impl_progress = false;
      struct set *guarded_roots = _mesa_pointer_set_create(NULL);
      bool changed;
      do {
         changed = false;
         nir_index_instrs(impl);
         unsigned max_idx = 0;
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block)
               max_idx = MAX2(max_idx, instr->index + 1);
         }
         if (!max_idx)
            break;

         BITSET_WORD *in_slice = rzalloc_array(NULL, BITSET_WORD,
                                               BITSET_WORDS(max_idx));
         nir_builder b = nir_builder_create(impl);
         nir_foreach_block_safe(block, impl) {
            if (try_guard_select_in_block(&b, block, max_idx, in_slice,
                                          guarded_roots)) {
               changed = true;
               impl_progress = true;
               break;
            }
         }
         ralloc_free(in_slice);
      } while (changed);
      _mesa_set_destroy(guarded_roots, NULL);

      progress |= nir_progress(impl_progress, impl, nir_metadata_none);
   }

   return progress;
}

/*
 * Canonical representation of a 32-bit integer address/index expression for
 * store-to-load forwarding across an if/else merge.
 *
 * A leaf is either:
 *   - a uniform preamble word identified by its half-register offset, or
 *   - an SSA scalar (def->index, comp).
 * A product term is either a single leaf (a, 0) or a sorted product of two
 * leaves (a, b), plus an additive constant.
 */
struct idx_atom {
   uint8_t kind; /* 0 = none, 1 = preamble_base, 2 = ssa, 3 = sum_hash */
   uint32_t a;
   uint32_t b;
};

struct idx_term {
   int32_t sign; /* +1 or -1 */
   struct idx_atom x;
   struct idx_atom y; /* kind=0 for linear leaf, else product x*y */
};

struct idx_expr {
   struct idx_term terms[16];
   unsigned n;
   uint32_t constant;
   bool valid;
};

static int
atom_cmp(struct idx_atom u, struct idx_atom v)
{
   if (u.kind != v.kind)
      return u.kind < v.kind ? -1 : 1;
   if (u.a != v.a)
      return u.a < v.a ? -1 : 1;
   if (u.b != v.b)
      return u.b < v.b ? -1 : 1;
   return 0;
}

static int
term_cmp(const void *pa, const void *pb)
{
   const struct idx_term *a = pa, *b = pb;
   int c = atom_cmp(a->x, b->x);
   if (c)
      return c;
   c = atom_cmp(a->y, b->y);
   if (c)
      return c;
   if (a->sign != b->sign)
      return a->sign < b->sign ? -1 : 1;
   return 0;
}

static struct idx_atom
scalar_atom(nir_scalar s)
{
   s = nir_scalar_chase_movs(s);
   nir_instr *instr = nir_def_instr(s.def);
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_load_preamble) {
         unsigned half_stride = DIV_ROUND_UP(intr->def.bit_size, 16);
         unsigned base = nir_intrinsic_base(intr) + s.comp * half_stride;
         return (struct idx_atom){.kind = 1, .a = base, .b = intr->def.bit_size};
      }
   }
   return (struct idx_atom){.kind = 2, .a = s.def->index, .b = s.comp};
}

static bool
expr_add_term(struct idx_expr *e, int32_t sign, struct idx_atom x,
              struct idx_atom y)
{
   if (y.kind && atom_cmp(x, y) > 0) {
      struct idx_atom tmp = x;
      x = y;
      y = tmp;
   }
   if (e->n >= ARRAY_SIZE(e->terms)) {
      e->valid = false;
      return false;
   }
   e->terms[e->n++] = (struct idx_term){.sign = sign, .x = x, .y = y};
   return true;
}

static void
parse_idx_expr(struct idx_expr *e, nir_scalar s, int32_t sign, unsigned depth);

static bool
parse_product(struct idx_expr *e, nir_scalar a, nir_scalar b, int32_t sign,
              unsigned depth)
{
   if (depth == 0)
      return false;

   /* Distribute over a small sum on either side: (p + q) * r = p*r + q*r. */
   struct idx_expr ea = {.valid = true};
   struct idx_expr eb = {.valid = true};
   parse_idx_expr(&ea, a, 1, depth - 1);
   parse_idx_expr(&eb, b, 1, depth - 1);
   if (!ea.valid || !eb.valid)
      return false;

   /* Reject quadratic+ nested products to keep expansion bounded. */
   for (unsigned i = 0; i < ea.n; ++i)
      if (ea.terms[i].y.kind)
         return false;
   for (unsigned j = 0; j < eb.n; ++j)
      if (eb.terms[j].y.kind)
         return false;

   struct idx_expr saved = *e;
   e->constant += (uint32_t)(sign * (int64_t)(ea.constant * eb.constant));

   if (eb.constant != 0) {
      /* Only handle constant 1/-1 trivially; otherwise bail to avoid scale factor. */
      if (eb.constant == 1) {
         for (unsigned i = 0; i < ea.n; ++i)
            expr_add_term(e, sign * ea.terms[i].sign, ea.terms[i].x,
                          (struct idx_atom){0});
      } else if (eb.constant == (uint32_t)-1) {
         for (unsigned i = 0; i < ea.n; ++i)
            expr_add_term(e, -sign * ea.terms[i].sign, ea.terms[i].x,
                          (struct idx_atom){0});
      } else if (ea.n > 0) {
         *e = saved;
         return false;
      }
   }
   if (ea.constant != 0) {
      if (ea.constant == 1) {
         for (unsigned j = 0; j < eb.n; ++j)
            expr_add_term(e, sign * eb.terms[j].sign, eb.terms[j].x,
                          (struct idx_atom){0});
      } else if (ea.constant == (uint32_t)-1) {
         for (unsigned j = 0; j < eb.n; ++j)
            expr_add_term(e, -sign * eb.terms[j].sign, eb.terms[j].x,
                          (struct idx_atom){0});
      } else if (eb.n > 0) {
         *e = saved;
         return false;
      }
   }

   for (unsigned i = 0; i < ea.n; ++i) {
      for (unsigned j = 0; j < eb.n; ++j) {
         if (!expr_add_term(e, sign * ea.terms[i].sign * eb.terms[j].sign,
                            ea.terms[i].x, eb.terms[j].x)) {
            *e = saved;
            return false;
         }
      }
   }
   if (!e->valid) {
      *e = saved;
      return false;
   }
   return true;
}

static void
parse_idx_expr(struct idx_expr *e, nir_scalar s, int32_t sign, unsigned depth)
{
   if (!e->valid)
      return;

   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_const(s)) {
      uint32_t c = nir_scalar_as_uint(s);
      e->constant += sign > 0 ? c : -c;
      return;
   }

   if (depth > 0 && nir_scalar_is_alu(s)) {
      nir_op op = nir_scalar_alu_op(s);
      if (op == nir_op_iadd) {
         parse_idx_expr(e, nir_scalar_chase_alu_src(s, 0), sign, depth - 1);
         parse_idx_expr(e, nir_scalar_chase_alu_src(s, 1), sign, depth - 1);
         return;
      } else if (op == nir_op_isub) {
         parse_idx_expr(e, nir_scalar_chase_alu_src(s, 0), sign, depth - 1);
         parse_idx_expr(e, nir_scalar_chase_alu_src(s, 1), -sign, depth - 1);
         return;
      } else if (op == nir_op_imul) {
         if (parse_product(e, nir_scalar_chase_alu_src(s, 0),
                           nir_scalar_chase_alu_src(s, 1), sign, depth - 1))
            return;
      } else if (op == nir_op_imadshl_agx) {
         nir_scalar sh = nir_scalar_chase_alu_src(s, 3);
         if (nir_scalar_is_const(sh) && nir_scalar_as_uint(sh) == 0) {
            if (parse_product(e, nir_scalar_chase_alu_src(s, 0),
                              nir_scalar_chase_alu_src(s, 1), sign, depth - 1)) {
               parse_idx_expr(e, nir_scalar_chase_alu_src(s, 2), sign,
                              depth - 1);
               return;
            }
         }
      }
   }

   expr_add_term(e, sign, scalar_atom(s), (struct idx_atom){0});
}

static bool
same_index_expr(nir_def *a, nir_def *b)
{
   if (a == b)
      return true;
   if (a->num_components != 1 || b->num_components != 1 ||
       a->bit_size != 32 || b->bit_size != 32)
      return false;

   struct idx_expr ea = {.valid = true};
   struct idx_expr eb = {.valid = true};
   parse_idx_expr(&ea, nir_get_scalar(a, 0), 1, 6);
   parse_idx_expr(&eb, nir_get_scalar(b, 0), 1, 6);
   if (!ea.valid || !eb.valid || ea.constant != eb.constant || ea.n != eb.n)
      return false;

   qsort(ea.terms, ea.n, sizeof(ea.terms[0]), term_cmp);
   qsort(eb.terms, eb.n, sizeof(eb.terms[0]), term_cmp);
   for (unsigned i = 0; i < ea.n; ++i) {
      if (term_cmp(&ea.terms[i], &eb.terms[i]) != 0)
         return false;
   }
   return true;
}

static bool
same_base_ptr(nir_def *a, nir_def *b)
{
   if (a == b)
      return true;
   if (a->bit_size != b->bit_size || a->num_components != b->num_components)
      return false;

   nir_instr *ia = nir_def_instr(a);
   nir_instr *ib = nir_def_instr(b);
   if (ia->type == nir_instr_type_intrinsic &&
       ib->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *ina = nir_instr_as_intrinsic(ia);
      nir_intrinsic_instr *inb = nir_instr_as_intrinsic(ib);
      if (ina->intrinsic == nir_intrinsic_load_preamble &&
          inb->intrinsic == nir_intrinsic_load_preamble &&
          nir_intrinsic_base(ina) == nir_intrinsic_base(inb))
         return true;
   }
   return false;
}

static bool
instr_may_alias_or_sync(const nir_instr *instr)
{
   if (instr->type == nir_instr_type_call)
      return true;
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_agx:
   case nir_intrinsic_load_constant_agx:
   case nir_intrinsic_load_preamble:
   case nir_intrinsic_load_global_invocation_id:
   case nir_intrinsic_load_local_invocation_id:
   case nir_intrinsic_load_local_invocation_index:
   case nir_intrinsic_load_workgroup_id:
   case nir_intrinsic_load_num_workgroups:
   case nir_intrinsic_load_subgroup_invocation:
      return false;
   default:
      return !(nir_intrinsic_infos[intr->intrinsic].flags &
               NIR_INTRINSIC_CAN_ELIMINATE);
   }
}

static bool
is_raw_forwarding_format(enum pipe_format fmt, unsigned bit_size,
                         unsigned num_components)
{
   return util_format_is_pure_integer(fmt) &&
          util_format_get_blocksizebits(fmt) == bit_size * num_components;
}

static bool
store_matches_load(const nir_intrinsic_instr *store,
                   const nir_intrinsic_instr *load)
{
   enum pipe_format fmt = nir_intrinsic_format(load);
   return (nir_intrinsic_access(store) &
           (ACCESS_VOLATILE | ACCESS_COHERENT)) == 0 &&
          (nir_intrinsic_access(load) &
           (ACCESS_VOLATILE | ACCESS_COHERENT)) == 0 &&
          is_raw_forwarding_format(fmt, load->def.bit_size,
                                   load->def.num_components) &&
          nir_intrinsic_base(store) == nir_intrinsic_base(load) &&
          nir_intrinsic_format(store) == fmt &&
          nir_intrinsic_sign_extend(store) == nir_intrinsic_sign_extend(load) &&
          store->src[0].ssa->bit_size == load->def.bit_size &&
          store->src[0].ssa->num_components == load->def.num_components &&
          same_base_ptr(store->src[1].ssa, load->src[0].ssa) &&
          same_index_expr(store->src[2].ssa, load->src[1].ssa);
}

static bool
forward_store_into_block_prefix(nir_intrinsic_instr *store, nir_block *block)
{
   bool progress = false;
   nir_foreach_instr_safe(instr, block) {
      if (instr->type == nir_instr_type_phi)
         continue;
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_agx &&
             store_matches_load(store, intr)) {
            nir_def_replace(&intr->def, store->src[0].ssa);
            progress = true;
            continue;
         }
      }
      if (instr_may_alias_or_sync(instr))
         break;
   }
   return progress;
}

static nir_intrinsic_instr *
find_trailing_matching_store(nir_block *block, nir_intrinsic_instr *load)
{
   if (nir_block_ends_in_jump(block))
      return NULL;

   nir_foreach_instr_reverse(instr, block) {
      if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_store_agx) {
            if (store_matches_load(intr, load))
               return intr;
            return NULL;
         }
      }
      if (instr_may_alias_or_sync(instr))
         return NULL;
   }
   return NULL;
}

bool
agx_nir_opt_forward_merge_loads(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      bool impl_progress = false;
      nir_index_ssa_defs(impl);

      nir_foreach_block(block, impl) {
         nir_cf_node *prev = nir_cf_node_prev(&block->cf_node);
         if (prev && prev->type == nir_cf_node_if &&
             util_dynarray_num_elements(&block->predecessors, nir_block *) == 2) {
            nir_if *nif = nir_cf_node_as_if(prev);
            nir_block *then_blk = nir_if_last_then_block(nif);
            nir_block *else_blk = nir_if_last_else_block(nif);
            bool has_then = false, has_else = false;
            util_dynarray_foreach(&block->predecessors, nir_block *, pred) {
               has_then |= (*pred == then_blk);
               has_else |= (*pred == else_blk);
            }
            if (has_then && has_else) {
               nir_foreach_instr_safe(instr, block) {
                  if (instr->type == nir_instr_type_phi)
                     continue;

                  if (instr->type == nir_instr_type_intrinsic) {
                     nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);
                     if (load->intrinsic == nir_intrinsic_load_agx &&
                         !(nir_intrinsic_access(load) &
                           (ACCESS_VOLATILE | ACCESS_COHERENT))) {
                        if (is_raw_forwarding_format(
                               nir_intrinsic_format(load), load->def.bit_size,
                               load->def.num_components)) {
                           nir_intrinsic_instr *ts =
                              find_trailing_matching_store(then_blk, load);
                           nir_intrinsic_instr *es =
                              ts ? find_trailing_matching_store(else_blk, load)
                                 : NULL;
                           if (ts && es) {
                              nir_phi_instr *phi = nir_phi_instr_create(shader);
                              nir_def_init(&phi->instr, &phi->def,
                                           load->def.num_components,
                                           load->def.bit_size);
                              nir_phi_instr_add_src(phi, then_blk,
                                                    ts->src[0].ssa);
                              nir_phi_instr_add_src(phi, else_blk,
                                                    es->src[0].ssa);
                              nir_instr_insert_before_block(block, &phi->instr);
                              nir_def_replace(&load->def, &phi->def);
                              impl_progress = true;
                              continue;
                           }
                        }
                     }
                  }

                  if (instr_may_alias_or_sync(instr))
                     break;
               }
            }
         }

         /* Also forward same-block straight-line non-coherent store_agx to
          * subsequent load_agx of the identical address and format (for
          * example after unswitching a diamond outside a loop), and into the
          * unaliased prefix of an immediately following `if` arm. */
         nir_intrinsic_instr *cur_store = NULL;
         nir_foreach_instr_safe(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_store_agx) {
                  cur_store = NULL;
                  enum pipe_format fmt = nir_intrinsic_format(intr);
                  if (!(nir_intrinsic_access(intr) &
                        (ACCESS_VOLATILE | ACCESS_COHERENT)) &&
                      is_raw_forwarding_format(
                         fmt, intr->src[0].ssa->bit_size,
                         intr->src[0].ssa->num_components))
                     cur_store = intr;
                  continue;
               }
               if (cur_store && intr->intrinsic == nir_intrinsic_load_agx &&
                   store_matches_load(cur_store, intr)) {
                  nir_def_replace(&intr->def, cur_store->src[0].ssa);
                  impl_progress = true;
                  continue;
               }
            }
            if (instr_may_alias_or_sync(instr))
               cur_store = NULL;
         }

         if (cur_store && !nir_block_ends_in_jump(block)) {
            nir_cf_node *next = nir_cf_node_next(&block->cf_node);
            if (next && next->type == nir_cf_node_if) {
               nir_if *nif = nir_cf_node_as_if(next);
               impl_progress |= forward_store_into_block_prefix(
                  cur_store, nir_if_first_then_block(nif));
               impl_progress |= forward_store_into_block_prefix(
                  cur_store, nir_if_first_else_block(nif));
            }
         }
      }

      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }

   return progress;
}

/*
 * 3. agx_nir_fold_const_select_alu:
 *    When a unary or constant-operand binary integer ALU op consumes a
 *    single-use tree of scalar bcsels whose leaves are all compile-time
 *    constants, push the operation into the constant leaves and fold it away.
 */
static bool
is_const_select_tree(nir_def *def, unsigned bit_size, unsigned depth)
{
   if (def->num_components != 1 || def->bit_size != bit_size)
      return false;

   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const)
      return true;

   if (depth == 0 || instr->type != nir_instr_type_alu ||
       !list_is_singular(&def->uses))
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_bcsel)
      return false;

   return is_const_select_tree(alu->src[1].src.ssa, bit_size, depth - 1) &&
          is_const_select_tree(alu->src[2].src.ssa, bit_size, depth - 1);
}

static uint64_t
eval_const_int_op(nir_op op, uint64_t x, uint64_t k, bool k_on_lhs,
                  unsigned bit_size)
{
   uint64_t mask = BITFIELD64_MASK(bit_size);
   uint64_t lhs = k_on_lhs ? k : x;
   uint64_t rhs = k_on_lhs ? x : k;
   uint64_t res = 0;

   switch (op) {
   case nir_op_iadd:
      res = lhs + rhs;
      break;
   case nir_op_isub:
      res = lhs - rhs;
      break;
   case nir_op_iand:
      res = lhs & rhs;
      break;
   case nir_op_ior:
      res = lhs | rhs;
      break;
   case nir_op_ixor:
      res = lhs ^ rhs;
      break;
   case nir_op_ishl:
      res = lhs << (rhs & (bit_size - 1));
      break;
   case nir_op_ushr:
      res = (lhs & mask) >> (rhs & (bit_size - 1));
      break;
   default:
      UNREACHABLE("unsupported const-select op");
   }
   return res & mask;
}

static nir_def *
rewrite_const_select_tree(nir_builder *b, nir_def *def, nir_op op, uint64_t k,
                          bool k_on_lhs, unsigned bit_size)
{
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_load_const) {
      uint64_t v = nir_src_as_uint(nir_src_for_ssa(def));
      uint64_t folded = eval_const_int_op(op, v, k, k_on_lhs, bit_size);
      return nir_imm_intN_t(b, folded, bit_size);
   }

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   assert(alu->op == nir_op_bcsel);
   b->cursor = nir_before_instr(instr);
   nir_def *t = rewrite_const_select_tree(b, alu->src[1].src.ssa, op, k,
                                          k_on_lhs, bit_size);
   nir_def *f = rewrite_const_select_tree(b, alu->src[2].src.ssa, op, k,
                                          k_on_lhs, bit_size);
   b->cursor = nir_before_instr(instr);
   nir_src_rewrite(&alu->src[1].src, t);
   alu->src[1].swizzle[0] = 0;
   nir_src_rewrite(&alu->src[2].src, f);
   alu->src[2].swizzle[0] = 0;
   return def;
}

bool
agx_nir_fold_const_select_alu(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      bool impl_progress = false;
      nir_builder b = nir_builder_create(impl);

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;

            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->def.num_components != 1 || alu->def.bit_size > 32)
               continue;

            switch (alu->op) {
            case nir_op_iadd:
            case nir_op_isub:
            case nir_op_iand:
            case nir_op_ior:
            case nir_op_ixor:
            case nir_op_ishl:
            case nir_op_ushr:
               break;
            default:
               continue;
            }

            nir_scalar s0 = nir_get_scalar(alu->src[0].src.ssa, alu->src[0].swizzle[0]);
            nir_scalar s1 = nir_get_scalar(alu->src[1].src.ssa, alu->src[1].swizzle[0]);
            bool k_on_lhs = false;
            nir_def *tree = NULL;
            uint64_t k = 0;

            if (nir_scalar_is_const(s1) && !nir_scalar_is_const(s0) &&
                s0.def->num_components == 1) {
               tree = s0.def;
               k = nir_scalar_as_uint(s1);
               k_on_lhs = false;
            } else if (nir_scalar_is_const(s0) && !nir_scalar_is_const(s1) &&
                       s1.def->num_components == 1) {
               tree = s1.def;
               k = nir_scalar_as_uint(s0);
               k_on_lhs = true;
            } else {
               continue;
            }

            if (nir_def_instr(tree)->type != nir_instr_type_alu ||
                nir_def_as_alu(tree)->op != nir_op_bcsel)
               continue;

            if (!is_const_select_tree(tree, alu->def.bit_size, 8))
               continue;

            nir_def *res = rewrite_const_select_tree(&b, tree, alu->op, k,
                                                     k_on_lhs, alu->def.bit_size);
            nir_def_replace(&alu->def, res);
            impl_progress = true;
         }
      }

      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }

   return progress;
}

/*
 * 4. agx_nir_opt_narrow_u8_fadd_tree:
 *    Narrow exact-integer FP32 linear combinations of u2f32(u8) leaves
 *    (fadd, fsub, fadd(a, fneg(b)), ffma(±2.0, a, b), fmul(2.0, a)) into
 *    16-bit integer arithmetic with i2f32 at the root exits when doing so
 *    strictly eliminates u2f32 conversions.
 *
 *    In IEEE-754 round-to-nearest-even (standard FP32 mode), every node in the
 *    matched DAG is an exact integer in [-16384, +16384] and never -0.0, so
 *    the integer evaluation is bit-for-bit identical including signed zero.
 */
struct u8_int_node {
   bool valid;
   bool is_leaf;
   int32_t min_v;
   int32_t max_v;
   uint8_t op; /* 0=leaf, 1=add, 2=sub, 3=add_shl1, 4=sub_shl1, 5=shl1_sub, 6=shl1 */
   uint32_t lhs_idx;
   uint32_t rhs_idx;
   nir_def *leaf_u8;
   nir_def *i16_def;
};

static bool
is_exact_fconst(nir_scalar s, float val)
{
   s = nir_scalar_chase_movs(s);
   return nir_scalar_is_const(s) && nir_scalar_as_float(s) == val;
}

static bool
chase_nonneg_u8_fneg(nir_scalar s, const struct u8_int_node *nodes,
                     unsigned ssa_alloc, uint32_t *out_idx)
{
   s = nir_scalar_chase_movs(s);
   if (s.def->num_components != 1 || !nir_scalar_is_alu(s))
      return false;
   nir_alu_instr *alu = nir_def_as_alu(s.def);
   if (alu->op != nir_op_fneg)
      return false;
   nir_scalar inner = nir_scalar_chase_alu_src(s, 0);
   inner = nir_scalar_chase_movs(inner);
   if (inner.def->num_components != 1 || inner.def->index >= ssa_alloc)
      return false;
   if (!nodes[inner.def->index].valid)
      return false;
   *out_idx = inner.def->index;
   return true;
}

static bool
range_fits_i16(int32_t lo, int32_t hi)
{
   return lo >= -16384 && hi <= 16384;
}

static nir_def *
materialize_u8_int_node(nir_builder *b, struct u8_int_node *nodes,
                        nir_def **defs_by_idx, uint32_t idx)
{
   struct u8_int_node *n = &nodes[idx];
   if (n->i16_def)
      return n->i16_def;

   nir_instr *orig = nir_def_instr(defs_by_idx[idx]);
   if (n->is_leaf) {
      b->cursor = nir_before_instr(orig);
      n->i16_def = nir_u2u32(b, n->leaf_u8);
      return n->i16_def;
   }

   nir_def *lhs = materialize_u8_int_node(b, nodes, defs_by_idx, n->lhs_idx);
   nir_def *rhs =
      n->op == 6 ? NULL : materialize_u8_int_node(b, nodes, defs_by_idx, n->rhs_idx);

   b->cursor = nir_before_instr(orig);
   switch (n->op) {
   case 1: /* lhs + rhs */
      n->i16_def = nir_iadd(b, lhs, rhs);
      break;
   case 2: /* lhs - rhs */
      n->i16_def = nir_isub(b, lhs, rhs);
      break;
   case 3: /* lhs + (rhs << 1) */
      n->i16_def = nir_iadd(b, lhs, nir_ishl_imm(b, rhs, 1));
      break;
   case 4: /* lhs - (rhs << 1) */
      n->i16_def = nir_isub(b, lhs, nir_ishl_imm(b, rhs, 1));
      break;
   case 5: /* (lhs << 1) - rhs */
      n->i16_def = nir_isub(b, nir_ishl_imm(b, lhs, 1), rhs);
      break;
   case 6: /* lhs << 1 */
      n->i16_def = nir_ishl_imm(b, lhs, 1);
      break;
   default:
      UNREACHABLE("invalid u8_int_node op");
   }
   return n->i16_def;
}

bool
agx_nir_opt_narrow_u8_fadd_tree(nir_shader *shader)
{
   /* Require standard RNE so (+0.0) + (-0.0) == +0.0. */
   if (shader->info.float_controls_execution_mode &
       FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP32)
      return false;

   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      nir_index_ssa_defs(impl);
      unsigned n_defs = impl->ssa_alloc;
      if (!n_defs) {
         nir_no_progress(impl);
         continue;
      }

      struct u8_int_node *nodes = rzalloc_array(NULL, struct u8_int_node, n_defs);
      nir_def **defs_by_idx = rzalloc_array(NULL, nir_def *, n_defs);
      bool any_nonleaf = false;

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            nir_def *def = nir_instr_def(instr);
            if (!def || def->index >= n_defs)
               continue;
            defs_by_idx[def->index] = def;

            if (instr->type != nir_instr_type_alu)
               continue;

            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->def.num_components != 1 ||
                alu->def.bit_size != 32)
               continue;

            struct u8_int_node *cur = &nodes[def->index];
            if (alu->op == nir_op_u2f32) {
               nir_scalar s = nir_get_scalar(alu->src[0].src.ssa, alu->src[0].swizzle[0]);
               if (s.def->num_components == 1 && s.def->bit_size == 8) {
                  cur->valid = true;
                  cur->is_leaf = true;
                  cur->min_v = 0;
                  cur->max_v = 255;
                  cur->leaf_u8 = s.def;
               }
               continue;
            }

            if (alu->op == nir_op_fadd) {
               nir_scalar s0 = nir_scalar_chase_movs(
                  nir_get_scalar(alu->src[0].src.ssa, alu->src[0].swizzle[0]));
               nir_scalar s1 = nir_scalar_chase_movs(
                  nir_get_scalar(alu->src[1].src.ssa, alu->src[1].swizzle[0]));
               bool v0 = s0.def->num_components == 1 && s0.def->index < n_defs &&
                         nodes[s0.def->index].valid;
               bool v1 = s1.def->num_components == 1 && s1.def->index < n_defs &&
                         nodes[s1.def->index].valid;
               uint32_t neg_idx = 0;
               if (v0 && v1) {
                  int32_t lo = nodes[s0.def->index].min_v + nodes[s1.def->index].min_v;
                  int32_t hi = nodes[s0.def->index].max_v + nodes[s1.def->index].max_v;
                  if (range_fits_i16(lo, hi)) {
                     cur->valid = true;
                     cur->op = 1;
                     cur->lhs_idx = s0.def->index;
                     cur->rhs_idx = s1.def->index;
                     cur->min_v = lo;
                     cur->max_v = hi;
                     any_nonleaf = true;
                  }
               } else if (v0 && chase_nonneg_u8_fneg(s1, nodes, n_defs, &neg_idx)) {
                  int32_t lo = nodes[s0.def->index].min_v - nodes[neg_idx].max_v;
                  int32_t hi = nodes[s0.def->index].max_v - nodes[neg_idx].min_v;
                  if (range_fits_i16(lo, hi)) {
                     cur->valid = true;
                     cur->op = 2;
                     cur->lhs_idx = s0.def->index;
                     cur->rhs_idx = neg_idx;
                     cur->min_v = lo;
                     cur->max_v = hi;
                     any_nonleaf = true;
                  }
               } else if (v1 && chase_nonneg_u8_fneg(s0, nodes, n_defs, &neg_idx)) {
                  int32_t lo = nodes[s1.def->index].min_v - nodes[neg_idx].max_v;
                  int32_t hi = nodes[s1.def->index].max_v - nodes[neg_idx].min_v;
                  if (range_fits_i16(lo, hi)) {
                     cur->valid = true;
                     cur->op = 2;
                     cur->lhs_idx = s1.def->index;
                     cur->rhs_idx = neg_idx;
                     cur->min_v = lo;
                     cur->max_v = hi;
                     any_nonleaf = true;
                  }
               }
            } else if (alu->op == nir_op_fsub) {
               nir_scalar s0 = nir_scalar_chase_movs(
                  nir_get_scalar(alu->src[0].src.ssa, alu->src[0].swizzle[0]));
               nir_scalar s1 = nir_scalar_chase_movs(
                  nir_get_scalar(alu->src[1].src.ssa, alu->src[1].swizzle[0]));
               bool v0 = s0.def->num_components == 1 && s0.def->index < n_defs &&
                         nodes[s0.def->index].valid;
               bool v1 = s1.def->num_components == 1 && s1.def->index < n_defs &&
                         nodes[s1.def->index].valid;
               if (v0 && v1) {
                  int32_t lo = nodes[s0.def->index].min_v - nodes[s1.def->index].max_v;
                  int32_t hi = nodes[s0.def->index].max_v - nodes[s1.def->index].min_v;
                  if (range_fits_i16(lo, hi)) {
                     cur->valid = true;
                     cur->op = 2;
                     cur->lhs_idx = s0.def->index;
                     cur->rhs_idx = s1.def->index;
                     cur->min_v = lo;
                     cur->max_v = hi;
                     any_nonleaf = true;
                  }
               }
            } else if (alu->op == nir_op_fmul) {
               nir_scalar s0 = nir_scalar_chase_movs(
                  nir_get_scalar(alu->src[0].src.ssa, alu->src[0].swizzle[0]));
               nir_scalar s1 = nir_scalar_chase_movs(
                  nir_get_scalar(alu->src[1].src.ssa, alu->src[1].swizzle[0]));
               nir_scalar v = {0};
               if (is_exact_fconst(s0, 2.0f))
                  v = s1;
               else if (is_exact_fconst(s1, 2.0f))
                  v = s0;
               if (v.def && v.def->num_components == 1 && v.def->index < n_defs &&
                   nodes[v.def->index].valid) {
                  int32_t lo = 2 * nodes[v.def->index].min_v;
                  int32_t hi = 2 * nodes[v.def->index].max_v;
                  if (range_fits_i16(lo, hi)) {
                     cur->valid = true;
                     cur->op = 6;
                     cur->lhs_idx = v.def->index;
                     cur->min_v = lo;
                     cur->max_v = hi;
                     any_nonleaf = true;
                  }
               }
            } else if (alu->op == nir_op_ffma) {
               nir_scalar s0 = nir_scalar_chase_movs(
                  nir_get_scalar(alu->src[0].src.ssa, alu->src[0].swizzle[0]));
               nir_scalar s1 = nir_scalar_chase_movs(
                  nir_get_scalar(alu->src[1].src.ssa, alu->src[1].swizzle[0]));
               nir_scalar s2 = nir_scalar_chase_movs(
                  nir_get_scalar(alu->src[2].src.ssa, alu->src[2].swizzle[0]));

               int factor = 0;
               nir_scalar mul_v = {0};
               if (is_exact_fconst(s0, 2.0f)) {
                  factor = 2;
                  mul_v = s1;
               } else if (is_exact_fconst(s1, 2.0f)) {
                  factor = 2;
                  mul_v = s0;
               } else if (is_exact_fconst(s0, -2.0f)) {
                  factor = -2;
                  mul_v = s1;
               } else if (is_exact_fconst(s1, -2.0f)) {
                  factor = -2;
                  mul_v = s0;
               }

               if (factor != 0 && mul_v.def->num_components == 1 &&
                   mul_v.def->index < n_defs && nodes[mul_v.def->index].valid) {
                  uint32_t m_idx = mul_v.def->index;
                  bool v2 = s2.def->num_components == 1 && s2.def->index < n_defs &&
                            nodes[s2.def->index].valid;
                  uint32_t neg_idx = 0;
                  if (factor == 2 && v2) {
                     int32_t lo = nodes[s2.def->index].min_v + 2 * nodes[m_idx].min_v;
                     int32_t hi = nodes[s2.def->index].max_v + 2 * nodes[m_idx].max_v;
                     if (range_fits_i16(lo, hi)) {
                        cur->valid = true;
                        cur->op = 3; /* s2 + (m << 1) */
                        cur->lhs_idx = s2.def->index;
                        cur->rhs_idx = m_idx;
                        cur->min_v = lo;
                        cur->max_v = hi;
                        any_nonleaf = true;
                     }
                  } else if (factor == 2 &&
                             chase_nonneg_u8_fneg(s2, nodes, n_defs, &neg_idx)) {
                     int32_t lo = 2 * nodes[m_idx].min_v - nodes[neg_idx].max_v;
                     int32_t hi = 2 * nodes[m_idx].max_v - nodes[neg_idx].min_v;
                     if (range_fits_i16(lo, hi)) {
                        cur->valid = true;
                        cur->op = 5; /* (m << 1) - neg_idx */
                        cur->lhs_idx = m_idx;
                        cur->rhs_idx = neg_idx;
                        cur->min_v = lo;
                        cur->max_v = hi;
                        any_nonleaf = true;
                     }
                  } else if (factor == -2 && v2) {
                     /* (-2.0 * m) + s2: since s2 is never -0.0, (-0.0)+(+0.0)=+0.0 */
                     int32_t lo = nodes[s2.def->index].min_v - 2 * nodes[m_idx].max_v;
                     int32_t hi = nodes[s2.def->index].max_v - 2 * nodes[m_idx].min_v;
                     if (range_fits_i16(lo, hi)) {
                        cur->valid = true;
                        cur->op = 4; /* s2 - (m << 1) */
                        cur->lhs_idx = s2.def->index;
                        cur->rhs_idx = m_idx;
                        cur->min_v = lo;
                        cur->max_v = hi;
                        any_nonleaf = true;
                     }
                  }
               }
            }
         }
      }

      bool impl_progress = false;
      if (any_nonleaf) {
         /* Check profitability: count exclusive u2f32 leaves vs root exits. */
         bool *is_root = rzalloc_array(NULL, bool, n_defs);
         unsigned root_exits = 0;
         unsigned excl_leaves = 0;

         for (unsigned i = 0; i < n_defs; ++i) {
            if (!nodes[i].valid)
               continue;

            bool used_outside = false;
            bool used_inside = false;
            nir_foreach_use_including_if(use, defs_by_idx[i]) {
               if (nir_src_is_if(use)) {
                  used_outside = true;
                  continue;
               }
               nir_instr *u = nir_src_use_instr(use);
               if (u->type == nir_instr_type_alu &&
                   nir_instr_as_alu(u)->op == nir_op_fneg) {
                  /* An fneg is inside if all of its uses are valid nodes. */
                  nir_def *fdef = &nir_instr_as_alu(u)->def;
                  bool fneg_in = false, fneg_out = false;
                  nir_foreach_use_including_if(fuse, fdef) {
                     if (nir_src_is_if(fuse)) {
                        fneg_out = true;
                     } else {
                        nir_def *fud = nir_instr_def(nir_src_use_instr(fuse));
                        if (fud && fud->index < n_defs && nodes[fud->index].valid)
                           fneg_in = true;
                        else
                           fneg_out = true;
                     }
                  }
                  if (fneg_in)
                     used_inside = true;
                  if (fneg_out)
                     used_outside = true;
                  continue;
               }
               nir_def *ud = nir_instr_def(u);
               if (ud && ud->index < n_defs && nodes[ud->index].valid)
                  used_inside = true;
               else
                  used_outside = true;
            }

            if (nodes[i].is_leaf) {
               if (used_inside && !used_outside)
                  excl_leaves++;
            } else if (used_outside) {
               is_root[i] = true;
               root_exits++;
            }
         }

         if (excl_leaves > root_exits && root_exits > 0) {
            nir_builder b = nir_builder_create(impl);
            for (unsigned i = 0; i < n_defs; ++i) {
               if (!is_root[i])
                  continue;
               nir_def *val_i16 = materialize_u8_int_node(&b, nodes, defs_by_idx, i);
               b.cursor = nir_before_instr(nir_def_instr(defs_by_idx[i]));
               nir_def *fval = nir_i2f32(&b, val_i16);
               nir_def_replace(defs_by_idx[i], fval);
               impl_progress = true;
            }
         }
         ralloc_free(is_root);
      }

      ralloc_free(nodes);
      ralloc_free(defs_by_idx);
      progress |= nir_progress(impl_progress, impl, nir_metadata_control_flow);
   }

   return progress;
}

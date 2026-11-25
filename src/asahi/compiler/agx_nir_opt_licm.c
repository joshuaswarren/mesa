/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "util/sparse_bitset.h"
#include "util/u_dynarray.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_nir.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_worklist.h"

/*
 * Loop-invariant code motion that keeps register demand in check.
 *
 * nir_opt_licm hoists every invariant instruction of a loop, and each hoisted
 * result that the loop still reads stays live across the whole loop. Shaders
 * that address hundreds of memory locations from one loop-invariant base, as
 * DXVK compute shaders do for their shared-memory taps, then carry hundreds of
 * hoisted addresses through a loop that already fills the register file, and
 * fill them from spill slots on every iteration, which costs more than the
 * additions hoisting saved. Simulate nir_opt_licm's instruction motion and
 * re-run liveness so each loop and the function as a whole only hoist if
 * estimated register demand stays within budget and within the same occupancy
 * tier.
 *
 * The estimate counts the 16-bit halves of SSA values live at each point,
 * leaving out constants and preamble values, which immediates and uniforms
 * hold rather than registers.
 */

struct sim_block {
   nir_block *block;
   struct util_dynarray instrs;
   struct u_sparse_bitset live_in;
   struct u_sparse_bitset live_out;
};

struct licm_estimate {
   uint8_t *halves;
   BITSET_DECLARE(non_load_uniform_pair, AGX_NUM_UNIFORMS / 4);
   unsigned reserve;
   unsigned align;
   void *mem_ctx;

   struct u_sparse_bitset live;
   unsigned live_halves;
};

static bool
pack_preamble_pair_base(nir_def *def, unsigned *lo_base_out)
{
   nir_instr *instr = nir_def_instr(def);
   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_pack_64_2x32_split)
      return false;

   nir_instr *lo = nir_def_instr(alu->src[0].src.ssa);
   nir_instr *hi = nir_def_instr(alu->src[1].src.ssa);
   if (lo->type != nir_instr_type_intrinsic ||
       hi->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *li = nir_instr_as_intrinsic(lo);
   nir_intrinsic_instr *hi_i = nir_instr_as_intrinsic(hi);
   if (li->intrinsic != nir_intrinsic_load_preamble ||
       hi_i->intrinsic != nir_intrinsic_load_preamble ||
       li->def.bit_size != 32 || hi_i->def.bit_size != 32)
      return false;

   unsigned lo_base = nir_intrinsic_base(li) + 2 * alu->src[0].swizzle[0];
   unsigned hi_base = nir_intrinsic_base(hi_i) + 2 * alu->src[1].swizzle[0];
   if ((lo_base & 3) != 0 || hi_base != lo_base + 2 ||
       lo_base >= AGX_NUM_UNIFORMS ||
       !agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3C, AGX_OPCODE_DEVICE_LOAD,
                                  0, lo_base, AGX_SIZE_64))
      return false;

   *lo_base_out = lo_base;
   return true;
}

static bool
pack_uses_are_load_bases(nir_def *def)
{
   nir_foreach_use_including_if(use, def) {
      if (nir_src_is_if(use) ||
          nir_src_use_instr(use)->type != nir_instr_type_intrinsic)
         return false;

      nir_intrinsic_instr *ui = nir_instr_as_intrinsic(nir_src_use_instr(use));
      if ((ui->intrinsic != nir_intrinsic_load_agx &&
           ui->intrinsic != nir_intrinsic_load_constant_agx) ||
          use != &ui->src[0])
         return false;
   }

   return true;
}

static bool
def_is_zero_half_uniform(const struct licm_estimate *est, nir_def *def)
{
   nir_instr *instr = nir_def_instr(def);

   if (instr->type == nir_instr_type_load_const ||
       instr->type == nir_instr_type_undef ||
       (instr->type == nir_instr_type_intrinsic &&
        nir_instr_as_intrinsic(instr)->intrinsic ==
           nir_intrinsic_load_preamble))
      return true;

   /* Packing two 4-half-aligned adjacent 32-bit preamble words into a 64-bit
    * device-load base stays in uniform registers on AGX3C rather than occupying
    * GPRs, provided no pack of that same uniform pair in the function has a
    * non-load use that post-LICM CSE could merge into it. */
   unsigned lo_base;
   if (pack_preamble_pair_base(def, &lo_base) &&
       !BITSET_TEST(est->non_load_uniform_pair, lo_base / 4))
      return true;

   return false;
}

static unsigned
def_halves(const struct licm_estimate *est, nir_def *def)
{
   if (def_is_zero_half_uniform(est, def))
      return 0;

   /* Match RA (agx_calc_register_demand), which rounds live vector widths up
    * to powers of two for live-range splitting. */
   unsigned halves = def->num_components * DIV_ROUND_UP(def->bit_size, 16);
   return halves ? util_next_power_of_two(halves) : 0;
}

static bool
record_halves(nir_def *def, void *data)
{
   struct licm_estimate *est = data;
   unsigned halves = def_halves(est, def);
   est->halves[def->index] = MIN2(halves, UINT8_MAX);
   if (halves > 8)
      est->align = 16;
   return true;
}

static void
scan_shader_ra_props(struct licm_estimate *est, nir_function_impl *impl,
                     unsigned *file_limit)
{
   /* Any function with a loop sets any_cf in RA, reserving at least 1 half. */
   est->reserve = 1;
   est->align = 8;
   const bool check_ms_array =
      impl->function->shader->info.stage == MESA_SHADER_FRAGMENT &&
      !impl->function->is_preamble;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         unsigned lo_base;
         if (def && pack_preamble_pair_base(def, &lo_base) &&
             !pack_uses_are_load_bases(def))
            BITSET_SET(est->non_load_uniform_pair, lo_base / 4);

         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_read_invocation:
            case nir_intrinsic_quad_broadcast:
               est->reserve = MAX2(est->reserve, 4);
               break;
            case nir_intrinsic_bindless_image_sparse_load:
            case nir_intrinsic_image_sparse_load:
               est->align = 16;
               FALLTHROUGH;
            case nir_intrinsic_bindless_image_load:
            case nir_intrinsic_image_load:
               if (check_ms_array &&
                   nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_MS &&
                   nir_intrinsic_image_array(intr))
                  *file_limit = MIN2(*file_limit, 184u);
               break;
            default:
               break;
            }
         } else if (instr->type == nir_instr_type_tex) {
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE &&
                tex->op != nir_texop_txs && tex->op != nir_texop_query_levels &&
                tex->op != nir_texop_texture_samples)
               est->reserve = MAX2(est->reserve, AGX3C_CEM_LOW_SCRATCH_END);

            if (check_ms_array && tex->sampler_dim == GLSL_SAMPLER_DIM_MS &&
                tex->is_array && tex->op != nir_texop_txs &&
                tex->op != nir_texop_query_levels &&
                tex->op != nir_texop_texture_samples)
               *file_limit = MIN2(*file_limit, 184u);

            if (tex->is_sparse)
               est->align = 16;

            if (tex->op == nir_texop_txd) {
               int ddx_idx = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
               bool has_min =
                  nir_tex_instr_src_index(tex, nir_tex_src_min_lod) >= 0;
               if (ddx_idx >= 0 &&
                   2 * nir_tex_instr_src_size(tex, ddx_idx) +
                         (has_min ? 1 : 0) >
                      4)
                  est->align = 16;
            }
         }
      }
   }

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block)
         nir_foreach_def(instr, record_halves, est);
   }
}

static void
set_live(struct licm_estimate *est, unsigned index)
{
   if (!u_sparse_bitset_test(&est->live, index)) {
      u_sparse_bitset_set(&est->live, index);
      est->live_halves += est->halves[index];
   }
}

static bool
kill_def(nir_def *def, void *data)
{
   struct licm_estimate *est = data;

   if (u_sparse_bitset_test(&est->live, def->index)) {
      u_sparse_bitset_clear(&est->live, def->index);
      est->live_halves -= est->halves[def->index];
   }

   return true;
}

static bool
gen_src(nir_src *src, void *data)
{
   set_live(data, src->ssa->index);
   return true;
}

/* AGX3C RT-completion SCS loads keep a physical destination in agx_dce.c even
 * when unread, and RA keeps such unread destinations allocated from their
 * write through block exit. */
static bool
instr_is_scs_rt_completion(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_scs_register_agx:
      return (nir_intrinsic_flags(intr) & 8) != 0;
   case nir_intrinsic_load_scs_base_agx:
      return !(nir_intrinsic_flags(intr) & AGX_NIR_SCS_TOUCH) &&
             (nir_intrinsic_flags(intr) & 8) != 0;
   default:
      return false;
   }
}

/* Unread enter_query_agx writes a physical destination in RA that is freed
 * immediately after the instruction (agx_register_allocate.c). */
static bool
instr_is_enter_query(const nir_instr *instr)
{
   return instr->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(instr)->intrinsic ==
             nir_intrinsic_enter_query_agx;
}

static unsigned
block_peak_instrs(struct licm_estimate *est, nir_block *block,
                  struct u_sparse_bitset *live_out, nir_instr *const *instrs,
                  unsigned nr_instrs)
{
   u_sparse_bitset_dup_with_ctx(&est->live, live_out, est->mem_ctx);
   est->live_halves = 0;
   U_SPARSE_BITSET_FOREACH_SET(&est->live, i)
      est->live_halves += est->halves[i];

   /* Unread RT-completion SCS loads stay live from their write to block exit;
    * seeding them at exit keeps them live through the reverse walk until
    * kill_def clears them at their defining instruction. */
   for (unsigned i = 0; i < nr_instrs; ++i) {
      nir_def *def = nir_instr_def(instrs[i]);
      if (def && nir_def_is_unused(def) && instr_is_scs_rt_completion(instrs[i]))
         set_live(est, def->index);
   }

   nir_if *following_if = nir_block_get_following_if(block);
   if (following_if)
      gen_src(&following_if->condition, est);

   unsigned peak = est->live_halves;

   for (unsigned i = nr_instrs; i-- > 0;) {
      nir_instr *instr = instrs[i];
      /* Results are written while the sources are still read. Also account for
       * unread enter_query_agx, whose destination is freed right after write. */
      unsigned dead_write = 0;
      nir_def *def = nir_instr_def(instr);
      if (def && !u_sparse_bitset_test(&est->live, def->index) &&
          instr_is_enter_query(instr))
         dead_write = est->halves[def->index];
      nir_foreach_src(instr, gen_src, est);
      peak = MAX2(peak, est->live_halves + dead_write);
      nir_foreach_def(instr, kill_def, est);
   }

   u_sparse_bitset_free(&est->live);
   return peak;
}

struct licm_sim {
   struct sim_block *blocks;
   unsigned *def_block;
   nir_loop *loop;
};

static bool
sim_defined_before_loop(nir_src *src, void *data)
{
   struct licm_sim *sim = data;
   return sim->def_block[src->ssa->index] <=
          nir_loop_predecessor_block(sim->loop)->index;
}

/* Mirror nir_opt_licm's invariance predicate when filter == NULL */
static bool
sim_instr_invariant(nir_instr *instr, struct licm_sim *sim)
{
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;

   case nir_instr_type_intrinsic:
      if (!nir_intrinsic_can_reorder(nir_instr_as_intrinsic(instr)))
         return false;
      FALLTHROUGH;

   case nir_instr_type_alu:
   case nir_instr_type_tex:
   case nir_instr_type_deref:
      return nir_foreach_src(instr, sim_defined_before_loop, sim);

   default:
      return false;
   }
}

static bool
loop_hoists(nir_loop *loop)
{
   if (!nir_loop_has_back_edge(loop))
      return false;

   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      if (nir_block_ends_in_return_or_halt(block))
         return false;

      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             (nir_instr_as_intrinsic(instr)->intrinsic ==
                 nir_intrinsic_terminate ||
              nir_instr_as_intrinsic(instr)->intrinsic ==
                 nir_intrinsic_terminate_if))
            return false;
      }
   }

   return true;
}

static bool
sim_visit_block(nir_block *block, struct licm_sim *sim)
{
   struct sim_block *src = &sim->blocks[block->index];
   struct sim_block *dst =
      &sim->blocks[nir_loop_predecessor_block(sim->loop)->index];
   unsigned nr = util_dynarray_num_elements(&src->instrs, nir_instr *);
   nir_instr **data = util_dynarray_begin(&src->instrs);
   unsigned kept = 0;
   bool progress = false;

   for (unsigned i = 0; i < nr; ++i) {
      nir_instr *instr = data[i];
      if (sim_instr_invariant(instr, sim)) {
         util_dynarray_append(&dst->instrs, instr);
         nir_def *def = nir_instr_def(instr);
         if (def)
            sim->def_block[def->index] = dst->block->index;
         progress = true;
      } else {
         data[kept++] = instr;
      }
   }

   src->instrs.size = kept * sizeof(nir_instr *);
   return progress;
}

/* Mirror nir_opt_licm's bottom-up CF traversal verbatim */
static bool
sim_visit_cf_list(struct exec_list *list, struct licm_sim *sim)
{
   bool progress = false;

   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block: {
         nir_cf_node *next = nir_cf_node_next(node);
         bool optimize_loop = false;

         if (next && next->type == nir_cf_node_loop) {
            nir_loop *inner_loop = nir_cf_node_as_loop(next);
            optimize_loop = loop_hoists(inner_loop);
            if (optimize_loop) {
               nir_loop *outer_loop = sim->loop;
               sim->loop = inner_loop;
               progress |= sim_visit_cf_list(&inner_loop->body, sim);
               progress |= sim_visit_cf_list(&inner_loop->continue_list, sim);
               sim->loop = outer_loop;
            }
         }

         nir_block *block = nir_cf_node_as_block(node);
         if (sim->loop &&
             nir_block_dominates(block, nir_loop_successor_block(sim->loop)))
            progress |= sim_visit_block(block, sim);

         if (next && next->type == nir_cf_node_loop && !optimize_loop) {
            nir_loop *loop = nir_cf_node_as_loop(next);
            progress |= sim_visit_cf_list(&loop->body, sim);
            progress |= sim_visit_cf_list(&loop->continue_list, sim);
         }
         break;
      }
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= sim_visit_cf_list(&nif->then_list, sim);
         progress |= sim_visit_cf_list(&nif->else_list, sim);
         break;
      }
      case nir_cf_node_loop:
         break;
      case nir_cf_node_function:
         UNREACHABLE("Unsupported cf_node type");
      }
   }

   return progress;
}

static bool
sim_set_def_dead(nir_def *def, void *live)
{
   u_sparse_bitset_clear(live, def->index);
   return true;
}

static bool
sim_set_src_live(nir_src *src, void *live)
{
   u_sparse_bitset_set(live, src->ssa->index);
   return true;
}

static void
compute_sim_liveness(nir_function_impl *impl, struct sim_block *blocks,
                     void *mem_ctx)
{
   nir_block_worklist worklist;
   nir_block_worklist_init(&worklist, impl->num_blocks, NULL);

   nir_foreach_block(block, impl) {
      u_sparse_bitset_init(&blocks[block->index].live_in, impl->ssa_alloc,
                           mem_ctx);
      u_sparse_bitset_init(&blocks[block->index].live_out, impl->ssa_alloc,
                           mem_ctx);
      nir_block_worklist_push_head(&worklist, block);
   }

   while (!nir_block_worklist_is_empty(&worklist)) {
      nir_block *block = nir_block_worklist_pop_head(&worklist);
      struct sim_block *sb = &blocks[block->index];

      u_sparse_bitset_free(&sb->live_in);
      u_sparse_bitset_dup_with_ctx(&sb->live_in, &sb->live_out, mem_ctx);

      nir_if *following_if = nir_block_get_following_if(block);
      if (following_if)
         sim_set_src_live(&following_if->condition, &sb->live_in);

      unsigned nr = util_dynarray_num_elements(&sb->instrs, nir_instr *);
      nir_instr *const *instrs = util_dynarray_begin(&sb->instrs);
      for (unsigned i = nr; i-- > 0;) {
         nir_foreach_def(instrs[i], sim_set_def_dead, &sb->live_in);
         nir_foreach_src(instrs[i], sim_set_src_live, &sb->live_in);
      }

      nir_foreach_pred(pred, block) {
         struct u_sparse_bitset tmp;
         u_sparse_bitset_dup_with_ctx(&tmp, &sb->live_in, mem_ctx);

         nir_foreach_phi(phi, block)
            u_sparse_bitset_clear(&tmp, phi->def.index);

         nir_foreach_phi(phi, block) {
            nir_foreach_phi_src(src, phi) {
               if (src->pred == pred) {
                  sim_set_src_live(&src->src, &tmp);
                  break;
               }
            }
         }

         bool changed =
            u_sparse_bitset_merge(&blocks[pred->index].live_out, &tmp);
         u_sparse_bitset_free(&tmp);
         if (changed)
            nir_block_worklist_push_tail(&worklist, pred);
      }
   }

   nir_block_worklist_fini(&worklist);
}

static bool
demand_fits(const struct licm_estimate *est, unsigned base_peak,
            unsigned post_peak, unsigned file_limit)
{
   if (post_peak <= base_peak)
      return true;

   unsigned limit = ROUND_DOWN_TO(MIN2(file_limit, 256u), est->align);
   unsigned base_demand = ALIGN_POT(base_peak + est->reserve, est->align);
   unsigned post_demand = ALIGN_POT(post_peak + est->reserve, est->align);
   if (base_demand > limit || post_demand > limit)
      return false;

   unsigned occ_max =
      ROUND_DOWN_TO(agx_occupancy_for_register_count(base_demand).max_registers,
                    est->align);
   return post_demand <= occ_max;
}

static bool
loops_fit(const struct licm_estimate *est, const unsigned *base_peaks,
          const unsigned *post_peaks, struct exec_list *list, unsigned budget)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         if (!loops_fit(est, base_peaks, post_peaks, &nif->then_list, budget) ||
             !loops_fit(est, base_peaks, post_peaks, &nif->else_list, budget))
            return false;
         break;
      }

      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         unsigned loop_base = 0, loop_post = 0;
         nir_foreach_block_in_cf_node(block, &loop->cf_node) {
            loop_base = MAX2(loop_base, base_peaks[block->index]);
            loop_post = MAX2(loop_post, post_peaks[block->index]);
         }

         if (!demand_fits(est, loop_base, loop_post, budget))
            return false;

         if (!loops_fit(est, base_peaks, post_peaks, &loop->body, budget) ||
             !loops_fit(est, base_peaks, post_peaks, &loop->continue_list,
                        budget))
            return false;
         break;
      }

      default:
         break;
      }
   }

   return true;
}

static bool
cf_list_has_loop(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop)
         return true;
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (cf_list_has_loop(&nif->then_list) ||
             cf_list_has_loop(&nif->else_list))
            return true;
      }
   }
   return false;
}

static bool
def_outside_loop(const nir_def *def, const nir_loop *loop)
{
   for (const nir_cf_node *node = nir_def_block(def)->cf_node.parent; node;
        node = node->parent) {
      if (node == &loop->cf_node)
         return false;
   }
   return true;
}

#define AGX_MAX_REASSOC_EQ_HOISTS_PER_LOOP 4

static bool
reassociate_eq_in_block(nir_builder *b, nir_block *block,
                        nir_loop *loop, unsigned *hoisted_in_loop)
{
   bool progress = false;
   nir_block *pre = nir_loop_predecessor_block(loop);
   const bool top_level = block->cf_node.parent == &loop->cf_node;

   nir_foreach_instr_safe(instr, block) {
      if (*hoisted_in_loop >= AGX_MAX_REASSOC_EQ_HOISTS_PER_LOOP)
         break;
      if (instr->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if ((alu->op != nir_op_ieq && alu->op != nir_op_ine) ||
          alu->def.num_components != 1)
         continue;

      const unsigned bit_size = alu->src[0].src.ssa->bit_size;
      if (bit_size != 16 && bit_size != 32)
         continue;

      for (unsigned s = 0; s < 2; ++s) {
         nir_def *lhs = alu->src[s].src.ssa;
         nir_def *inv = alu->src[1 - s].src.ssa;
         if (lhs->num_components != 1 || alu->src[s].swizzle[0] != 0 ||
             inv->num_components != 1 || alu->src[1 - s].swizzle[0] != 0 ||
             nir_def_is_const(inv) || !def_outside_loop(inv, loop) ||
             def_outside_loop(lhs, loop) || !list_is_singular(&lhs->uses))
            continue;

         nir_instr *lhs_instr = nir_def_instr(lhs);
         if (lhs_instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *add = nir_instr_as_alu(lhs_instr);
         if (add->op != nir_op_iadd || add->def.num_components != 1 ||
             add->def.bit_size != bit_size)
            continue;

         int const_idx = -1;
         if (nir_src_is_const(add->src[0].src) &&
             !nir_src_is_const(add->src[1].src))
            const_idx = 0;
         else if (nir_src_is_const(add->src[1].src) &&
                  !nir_src_is_const(add->src[0].src))
            const_idx = 1;
         if (const_idx < 0)
            continue;

         const uint64_t c = nir_src_comp_as_uint(add->src[const_idx].src,
                                                 add->src[const_idx].swizzle[0]);
         if (c == 0)
            continue;

         const uint64_t neg_c =
            (uint64_t)(-(int64_t)c) & BITFIELD64_MASK(bit_size);
         /* Top-tested loops place their body after `if (exit) break`, so the
          * body block does not dominate the loop successor and nir_opt_licm
          * will not visit it. Integer addition cannot trap and `inv` is
          * defined outside the loop, so materialize `inv - C` directly in the
          * preheader for unconditional loop-body blocks (bounded per loop to
          * keep live-through register demand small). */
         if (top_level && pre) {
            b->cursor = nir_after_block_before_jump(pre);
            (*hoisted_in_loop)++;
         } else {
            b->cursor = nir_before_instr(&alu->instr);
         }
         nir_def *adj = nir_iadd_imm(b, inv, neg_c);

         nir_src_rewrite(&alu->src[s].src, add->src[1 - const_idx].src.ssa);
         alu->src[s].swizzle[0] = add->src[1 - const_idx].swizzle[0];
         nir_src_rewrite(&alu->src[1 - s].src, adj);
         alu->src[1 - s].swizzle[0] = 0;
         nir_instr_remove(&add->instr);
         progress = true;
         break;
      }
   }

   return progress;
}

static bool
reassociate_eq_in_cf_list(nir_builder *b, struct exec_list *list,
                          nir_loop *loop, unsigned *hoisted_in_loop)
{
   bool progress = false;

   foreach_list_typed(nir_cf_node, node, node, list) {
      switch (node->type) {
      case nir_cf_node_block:
         if (loop)
            progress |= reassociate_eq_in_block(b, nir_cf_node_as_block(node),
                                                loop, hoisted_in_loop);
         break;
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         progress |= reassociate_eq_in_cf_list(b, &nif->then_list, loop,
                                               hoisted_in_loop);
         progress |= reassociate_eq_in_cf_list(b, &nif->else_list, loop,
                                               hoisted_in_loop);
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *inner = nir_cf_node_as_loop(node);
         unsigned inner_hoists = 0;
         progress |=
            reassociate_eq_in_cf_list(b, &inner->body, inner, &inner_hoists);
         progress |= reassociate_eq_in_cf_list(b, &inner->continue_list, inner,
                                               &inner_hoists);
         break;
      }
      default:
         break;
      }
   }

   return progress;
}

/*
 * Reassociate loop-variant constant-offset equality comparisons against a
 * loop-invariant value so the constant adjustment becomes loop-invariant:
 *
 *    (x + C) == inv  <==>  x == (inv - C)
 *    (x + C) != inv  <==>  x != (inv - C)
 *
 * Modular two's-complement equality holds for all bit patterns with no
 * overflow preconditions. For unconditional loop-body blocks the pure integer
 * adjustment `inv - C` is placed directly in the loop preheader (bounded per
 * loop), since top-tested loop bodies sit after `if (exit) break` and do not
 * dominate the loop successor visited by nir_opt_licm.
 */
bool
agx_nir_reassociate_loop_invariant_eq(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function_impl(impl, shader) {
      if (impl->function->is_preamble || !cf_list_has_loop(&impl->body))
         continue;

      nir_builder b = nir_builder_create(impl);
      unsigned top_hoists = 0;
      if (reassociate_eq_in_cf_list(&b, &impl->body, NULL, &top_hoists)) {
         nir_progress(true, impl, nir_metadata_control_flow);
         progress = true;
      } else {
         nir_no_progress(impl);
      }
   }

   return progress;
}

/*
 * Run nir_opt_licm if no loop or shader peak's estimated register demand would
 * exceed budget halves or cross an occupancy step with the invariants hoisted.
 */
bool
agx_nir_opt_licm(nir_shader *shader, unsigned budget, bool cap_rt_compute)
{
   bool any_hoist = false;

   nir_foreach_function_impl(impl, shader) {
      if (!cf_list_has_loop(&impl->body))
         continue;

      nir_metadata_require(impl, nir_metadata_block_index |
                                    nir_metadata_dominance |
                                    nir_metadata_live_defs);

      void *mem_ctx = ralloc_context(NULL);
      struct licm_estimate est = {
         .halves = rzalloc_array(mem_ctx, uint8_t, impl->ssa_alloc),
         .mem_ctx = mem_ctx,
      };

      unsigned impl_limit = budget;
      if (cap_rt_compute && !impl->function->is_preamble)
         impl_limit = MIN2(impl_limit, AGX_M5_RT_REGISTER_BUDGET);
      scan_shader_ra_props(&est, impl, &impl_limit);

      struct sim_block *blocks =
         rzalloc_array(mem_ctx, struct sim_block, impl->num_blocks);
      unsigned *def_block = rzalloc_array(mem_ctx, unsigned, impl->ssa_alloc);
      unsigned *base_peaks = rzalloc_array(mem_ctx, unsigned, impl->num_blocks);
      unsigned *post_peaks = rzalloc_array(mem_ctx, unsigned, impl->num_blocks);

      nir_foreach_block(block, impl) {
         struct sim_block *sb = &blocks[block->index];
         sb->block = block;
         util_dynarray_init(&sb->instrs, mem_ctx);

         nir_foreach_instr(instr, block) {
            nir_def *def = nir_instr_def(instr);
            if (def)
               def_block[def->index] = block->index;
            if (instr->type != nir_instr_type_phi)
               util_dynarray_append(&sb->instrs, instr);
         }

         base_peaks[block->index] = block_peak_instrs(
            &est, block, &block->live_out, util_dynarray_begin(&sb->instrs),
            util_dynarray_num_elements(&sb->instrs, nir_instr *));
      }

      struct licm_sim sim = {
         .blocks = blocks,
         .def_block = def_block,
      };
      bool hoisted = sim_visit_cf_list(&impl->body, &sim);
      bool fits = true;

      if (hoisted) {
         any_hoist = true;
         compute_sim_liveness(impl, blocks, mem_ctx);

         unsigned func_base = 0, func_post = 0;
         nir_foreach_block(block, impl) {
            struct sim_block *sb = &blocks[block->index];
            post_peaks[block->index] = block_peak_instrs(
               &est, block, &sb->live_out, util_dynarray_begin(&sb->instrs),
               util_dynarray_num_elements(&sb->instrs, nir_instr *));
            func_base = MAX2(func_base, base_peaks[block->index]);
            func_post = MAX2(func_post, post_peaks[block->index]);
         }

         fits = demand_fits(&est, func_base, func_post, impl_limit) &&
                loops_fit(&est, base_peaks, post_peaks, &impl->body, impl_limit);
      }

      ralloc_free(mem_ctx);

      if (!fits)
         return false;
   }

   if (!any_hoist)
      return false;

   return nir_opt_licm(shader, NULL);
}

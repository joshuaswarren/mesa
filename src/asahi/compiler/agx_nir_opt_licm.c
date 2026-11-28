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
   enum agx_isa_encoding encoding;
   uint8_t *halves;
   BITSET_DECLARE(non_load_uniform_pair, AGX_NUM_UNIFORMS / 4);
   unsigned reserve;
   unsigned align;
   void *mem_ctx;

   struct u_sparse_bitset live;
   unsigned live_halves;
};

static bool
pack_preamble_pair_base(nir_def *def, unsigned *lo_base_out,
                        enum agx_isa_encoding encoding)
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
       !agx_instr_accepts_uniform(encoding, AGX_OPCODE_DEVICE_LOAD,
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
   if (pack_preamble_pair_base(def, &lo_base, est->encoding) &&
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
         if (def && pack_preamble_pair_base(def, &lo_base, est->encoding) &&
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

enum licm_spec_mode {
   LICM_SPEC_NONE = 0,
   LICM_SPEC_ALL,
   LICM_SPEC_SFU_CHAIN,
   LICM_SPEC_CONST_LOAD_CHAIN,
   LICM_SPEC_AGX2_INTEGER,
};

struct licm_sim {
   struct sim_block *blocks;
   unsigned *def_block;
   nir_loop *loop;
   enum licm_spec_mode spec_mode;
   bool hoisted_non_const;
   bool hoisted_sfu;
   bool hoisted_const_load;
};

static bool
sim_defined_before_loop(nir_src *src, void *data)
{
   struct licm_sim *sim = data;
   return sim->def_block[src->ssa->index] <=
          nir_loop_predecessor_block(sim->loop)->index;
}

/* Pure, non-faulting instructions safe to hoist out of unconditional direct
 * loop-body blocks that do not dominate the loop exit (such as the body of a
 * top-tested loop after `if (exit) break`). Never hoists regular memory loads,
 * textures, derefs, integer divisions, or subgroup/lane-dependent ops. */
static bool
agx_licm_can_speculate(nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      return true;

   case nir_instr_type_alu: {
      const nir_alu_instr *alu = nir_instr_as_alu(instr);
      switch (alu->op) {
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
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      const nir_intrinsic_info *info = &nir_intrinsic_infos[intr->intrinsic];
      if (!info->has_dest || !nir_intrinsic_can_reorder(intr))
         return false;
      switch (intr->intrinsic) {
      case nir_intrinsic_load_preamble:
      case nir_intrinsic_load_push_constant:
      case nir_intrinsic_load_global_invocation_id:
      case nir_intrinsic_load_local_invocation_id:
      case nir_intrinsic_load_local_invocation_index:
      case nir_intrinsic_load_subgroup_invocation:
      case nir_intrinsic_load_subgroup_id:
      case nir_intrinsic_load_workgroup_id:
      case nir_intrinsic_load_num_workgroups:
         return true;
      default:
         if (info->flags & (NIR_INTRINSIC_SUBGROUP | NIR_INTRINSIC_QUADGROUP))
            return false;
         return nir_intrinsic_has_access(intr) &&
                (nir_intrinsic_access(intr) & ACCESS_CAN_SPECULATE);
      }
   }

   default:
      return false;
   }
}

static bool
block_is_direct_loop_body(const nir_block *block, const nir_loop *loop)
{
   return block->cf_node.parent == &loop->cf_node;
}

/* These instructions are safe to execute even when the loop exits before
 * its first iteration. Keep conditional arithmetic in its original branch.
 */
static bool
agx2_integer_invariant(nir_instr *instr)
{
   if (instr->type == nir_instr_type_intrinsic) {
      switch (nir_instr_as_intrinsic(instr)->intrinsic) {
      case nir_intrinsic_load_preamble:
      case nir_intrinsic_load_subgroup_invocation:
      case nir_intrinsic_load_subgroup_id:
      case nir_intrinsic_load_local_invocation_index:
      case nir_intrinsic_load_local_invocation_id:
      case nir_intrinsic_load_workgroup_id:
         return true;
      default:
         return false;
      }
   }
   if (instr->type != nir_instr_type_alu)
      return false;

   switch (nir_instr_as_alu(instr)->op) {
   case nir_op_mov:
   case nir_op_iadd:
   case nir_op_isub:
   case nir_op_imul:
   case nir_op_imin:
   case nir_op_imax:
   case nir_op_umin:
   case nir_op_umax:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_ixor:
   case nir_op_ishl:
   case nir_op_ushr:
   case nir_op_ishr:
   case nir_op_ubitfield_extract:
   case nir_op_ibitfield_extract:
   case nir_op_bitfield_insert:
   case nir_op_u2u16:
   case nir_op_u2u32:
   case nir_op_i2i16:
   case nir_op_i2i32:
   case nir_op_imadshl_agx:
   case nir_op_extr_agx:
      return true;
   default:
      return false;
   }
}

static bool
agx2_integer_licm_filter(nir_instr *instr, nir_loop *loop, bool dominates_exit)
{
   return block_is_direct_loop_body(instr->block, loop) &&
          agx2_integer_invariant(instr);
}

static nir_block *
instr_current_block(nir_instr *instr, const struct licm_sim *sim)
{
   if (!sim)
      return instr->block;
   nir_def *def = nir_instr_def(instr);
   return def ? sim->blocks[sim->def_block[def->index]].block : instr->block;
}

static bool
loop_has_terminate(nir_loop *loop)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             (nir_instr_as_intrinsic(instr)->intrinsic ==
                 nir_intrinsic_terminate ||
              nir_instr_as_intrinsic(instr)->intrinsic ==
                 nir_intrinsic_terminate_if))
            return true;
      }
   }
   return false;
}

static bool
alu_is_expensive_sfu(nir_op op)
{
   switch (op) {
   case nir_op_fsinc_agx3:
   case nir_op_fsin_agx:
   case nir_op_fsin:
   case nir_op_fcos:
   case nir_op_frcp:
   case nir_op_frsq:
   case nir_op_fsqrt:
   case nir_op_fexp2:
   case nir_op_flog2:
   case nir_op_fpow:
      return true;
   default:
      return false;
   }
}

/* Scalar FP / range-reduction / select ops that participate in finishing a
 * lowered transcendental (such as sin/cos range reduction and small-angle
 * clamp). Excluding integer address math prevents unrelated offset chains from
 * being pulled into the SFU subgraph. */
static bool
alu_is_sfu_consumer_op(nir_op op)
{
   if (alu_is_expensive_sfu(op))
      return true;
   switch (op) {
   case nir_op_fmul:
   case nir_op_fadd:
   case nir_op_fsub:
   case nir_op_ffma:
   case nir_op_fneg:
   case nir_op_fabs:
   case nir_op_fmin:
   case nir_op_fmax:
   case nir_op_flt:
   case nir_op_fge:
   case nir_op_feq:
   case nir_op_fneu:
   case nir_op_bcsel:
   case nir_op_ixor:
   case nir_op_iand:
   case nir_op_ior:
   case nir_op_inot:
   case nir_op_ieq:
   case nir_op_ine:
      return true;
   default:
      return false;
   }
}

static bool
instr_fed_by_sfu(nir_instr *instr, nir_loop *loop, nir_block *pre,
                 const struct licm_sim *sim, unsigned depth)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu_is_expensive_sfu(alu->op))
      return true;
   if (depth == 0 || !alu_is_sfu_consumer_op(alu->op))
      return false;

   for (unsigned s = 0; s < nir_op_infos[alu->op].num_inputs; ++s) {
      nir_instr *si = nir_def_instr(alu->src[s].src.ssa);
      nir_block *sb = instr_current_block(si, sim);
      if ((block_is_direct_loop_body(sb, loop) || sb == pre) &&
          instr_fed_by_sfu(si, loop, pre, sim, depth - 1))
         return true;
   }
   return false;
}

static bool
def_feeds_sfu_chain(nir_def *def, nir_loop *loop, nir_block *pre,
                    const struct licm_sim *sim, unsigned depth)
{
   if (depth == 0)
      return false;

   nir_foreach_use(u, def) {
      nir_instr *ui = nir_src_use_instr(u);
      nir_block *ub = instr_current_block(ui, sim);
      if (!block_is_direct_loop_body(ub, loop) || ui->type != nir_instr_type_alu)
         continue;
      nir_alu_instr *ualu = nir_instr_as_alu(ui);
      if (alu_is_expensive_sfu(ualu->op))
         return true;
      if (alu_is_sfu_consumer_op(ualu->op) &&
          instr_fed_by_sfu(ui, loop, pre, sim, 6))
         return true;
      if (agx_licm_can_speculate(ui) &&
          def_feeds_sfu_chain(&ualu->def, loop, pre, sim, depth - 1))
         return true;
   }
   return false;
}

/* True if `instr` belongs to a loop-invariant transcendental (SFU) subgraph in
 * a direct body block of `loop` (either an SFU op, its backward operand slice,
 * or its scalar FP / range-reduction consumers and their sibling predicates). */
static bool
instr_in_direct_sfu_chain(nir_instr *instr, nir_loop *loop,
                          const struct licm_sim *sim)
{
   nir_block *blk = instr_current_block(instr, sim);
   if (!block_is_direct_loop_body(blk, loop) ||
       !agx_licm_can_speculate(instr) || loop_has_terminate(loop))
      return false;

   if (instr->type == nir_instr_type_alu &&
       alu_is_expensive_sfu(nir_instr_as_alu(instr)->op))
      return true;

   nir_block *pre = nir_loop_predecessor_block(loop);
   if (instr_fed_by_sfu(instr, loop, pre, sim, 6))
      return true;

   nir_def *def = nir_instr_def(instr);
   return def && def_feeds_sfu_chain(def, loop, pre, sim, 16);
}

static bool
loop_has_direct_body_sfu(nir_loop *loop)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      if (block->cf_node.parent->type != nir_cf_node_loop)
         continue;
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_alu &&
             alu_is_expensive_sfu(nir_instr_as_alu(instr)->op))
            return true;
      }
   }
   return false;
}

static bool
agx_licm_filter_all(nir_instr *instr, nir_loop *loop, bool dominates_exit)
{
   if (dominates_exit)
      return true;
   return block_is_direct_loop_body(instr->block, loop) &&
          agx_licm_can_speculate(instr) && !loop_has_terminate(loop);
}

static bool
instr_is_speculatable_const_load(nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_global_constant ||
       !nir_intrinsic_can_reorder(intr))
      return false;
   enum gl_access_qualifier access = nir_intrinsic_access(intr);
   return (access & ACCESS_CAN_SPECULATE) &&
          !(access & (ACCESS_BUFFER_DESCRIPTOR_AGX | ACCESS_BUFFER_BINDING_AGX));
}

static bool
def_feeds_const_load_chain(nir_def *def, nir_loop *loop,
                           const struct licm_sim *sim, unsigned depth)
{
   if (depth == 0)
      return false;

   nir_foreach_use(u, def) {
      nir_instr *ui = nir_src_use_instr(u);
      nir_block *ub = instr_current_block(ui, sim);
      if (!block_is_direct_loop_body(ub, loop))
         continue;
      if (instr_is_speculatable_const_load(ui))
         return true;
      nir_def *udef = nir_instr_def(ui);
      if (udef && agx_licm_can_speculate(ui) &&
          def_feeds_const_load_chain(udef, loop, sim, depth - 1))
         return true;
   }
   return false;
}

static bool
instr_in_direct_const_load_chain(nir_instr *instr, nir_loop *loop,
                                 const struct licm_sim *sim)
{
   nir_block *blk = instr_current_block(instr, sim);
   if (!block_is_direct_loop_body(blk, loop) ||
       !agx_licm_can_speculate(instr) || loop_has_terminate(loop))
      return false;

   if (instr_is_speculatable_const_load(instr))
      return true;

   nir_def *def = nir_instr_def(instr);
   return def && def_feeds_const_load_chain(def, loop, sim, 16);
}

static bool
loop_has_direct_body_const_load(nir_loop *loop)
{
   nir_foreach_block_in_cf_node(block, &loop->cf_node) {
      if (block->cf_node.parent->type != nir_cf_node_loop)
         continue;
      nir_foreach_instr(instr, block) {
         if (instr_is_speculatable_const_load(instr))
            return true;
      }
   }
   return false;
}

static bool
cf_list_has_direct_body_const_load(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop) {
         if (loop_has_direct_body_const_load(nir_cf_node_as_loop(node)))
            return true;
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (cf_list_has_direct_body_const_load(&nif->then_list) ||
             cf_list_has_direct_body_const_load(&nif->else_list))
            return true;
      }
   }
   return false;
}

static bool
shader_has_direct_body_const_load(nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      if (cf_list_has_direct_body_const_load(&impl->body))
         return true;
   }
   return false;
}

static bool
agx_licm_filter_const_load(nir_instr *instr, nir_loop *loop,
                           bool dominates_exit)
{
   if (dominates_exit)
      return true;
   return instr_in_direct_const_load_chain(instr, loop, NULL);
}

static bool
agx_licm_filter_sfu(nir_instr *instr, nir_loop *loop, bool dominates_exit)
{
   if (dominates_exit)
      return true;
   return instr_in_direct_sfu_chain(instr, loop, NULL);
}

/* Mirror nir_opt_licm's invariance predicate for the selected spec_mode */
static bool
sim_instr_invariant(nir_instr *instr, struct licm_sim *sim, bool dom_exit,
                    bool direct_body)
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
      if (!nir_foreach_src(instr, sim_defined_before_loop, sim))
         return false;
      if (sim->spec_mode == LICM_SPEC_AGX2_INTEGER)
         return direct_body && agx2_integer_invariant(instr);
      if (dom_exit)
         return true;
      if (sim->spec_mode == LICM_SPEC_ALL)
         return direct_body && agx_licm_can_speculate(instr);
      if (sim->spec_mode == LICM_SPEC_SFU_CHAIN)
         return direct_body && instr_in_direct_sfu_chain(instr, sim->loop, sim);
      if (sim->spec_mode == LICM_SPEC_CONST_LOAD_CHAIN)
         return direct_body &&
                instr_in_direct_const_load_chain(instr, sim->loop, sim);
      return false;

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
sim_visit_block(nir_block *block, struct licm_sim *sim, bool dom_exit,
                bool direct_body)
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
      if (sim_instr_invariant(instr, sim, dom_exit, direct_body)) {
         util_dynarray_append(&dst->instrs, instr);
         nir_def *def = nir_instr_def(instr);
         if (def)
            sim->def_block[def->index] = dst->block->index;
         if (instr->type != nir_instr_type_load_const &&
             instr->type != nir_instr_type_undef) {
            sim->hoisted_non_const = true;
            if (instr->type == nir_instr_type_alu &&
                alu_is_expensive_sfu(nir_instr_as_alu(instr)->op))
               sim->hoisted_sfu = true;
            if (instr_is_speculatable_const_load(instr))
               sim->hoisted_const_load = true;
         }
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
         if (sim->loop) {
            const bool dom_exit =
               nir_block_dominates(block, nir_loop_successor_block(sim->loop));
            if (dom_exit || sim->spec_mode != LICM_SPEC_NONE) {
               const bool direct_body =
                  block_is_direct_loop_body(block, sim->loop);
               progress |=
                  sim_visit_block(block, sim, dom_exit, direct_body);
            }
         }

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
def_outside_loop(nir_def *def, const nir_loop *loop)
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
      if (impl->function->is_preamble || !cf_list_has_loop(&impl->body)) {
         nir_no_progress(impl);
         continue;
      }

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

static bool
simulate_shader_licm(nir_shader *shader, unsigned budget, bool cap_rt_compute,
                     enum licm_spec_mode spec_mode, bool *any_hoist_out)
{
   bool any_hoist = false;
   bool any_useful = false;

   nir_foreach_function_impl(impl, shader) {
      if (!cf_list_has_loop(&impl->body))
         continue;

      nir_metadata_require(impl, nir_metadata_block_index |
                                    nir_metadata_dominance |
                                    nir_metadata_live_defs);

      void *mem_ctx = ralloc_context(NULL);
      struct licm_estimate est = {
         .encoding = spec_mode == LICM_SPEC_AGX2_INTEGER
                        ? AGX_ISA_ENCODING_AGX2 : AGX_ISA_ENCODING_AGX3C,
         .halves = rzalloc_array(mem_ctx, uint8_t, impl->ssa_alloc),
         .mem_ctx = mem_ctx,
      };

      unsigned impl_limit = budget;
      if (cap_rt_compute && !impl->function->is_preamble)
         impl_limit = MIN2(impl_limit, AGX_M5_RT_REGISTER_BUDGET);
      scan_shader_ra_props(&est, impl, &impl_limit);

      /* Speculative direct-body hoisting leaves explicit backend RA headroom
       * below the 64-half max-occupancy cliff: <=48 NIR halves for general
       * pure ALU hoisting and <=56 NIR halves for targeted SFU-chain hoisting. */
      const unsigned effective_limit =
         spec_mode == LICM_SPEC_ALL
            ? MIN2(impl_limit, 48u)
            : (spec_mode == LICM_SPEC_SFU_CHAIN ? MIN2(impl_limit, 56u)
                                                : impl_limit);

      struct sim_block *blocks =
         rzalloc_array(mem_ctx, struct sim_block, impl->num_blocks);
      unsigned *def_block = rzalloc_array(mem_ctx, unsigned, impl->ssa_alloc);
      unsigned *base_peaks = rzalloc_array(mem_ctx, unsigned, impl->num_blocks);
      unsigned *post_peaks = rzalloc_array(mem_ctx, unsigned, impl->num_blocks);

      unsigned func_base_pre = 0;
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
         func_base_pre = MAX2(func_base_pre, base_peaks[block->index]);
      }

      if (spec_mode != LICM_SPEC_NONE &&
          ALIGN_POT(func_base_pre + est.reserve, est.align) >
             ROUND_DOWN_TO(MIN2(effective_limit, 256u), est.align)) {
         ralloc_free(mem_ctx);
         *any_hoist_out = false;
         return false;
      }

      struct licm_sim sim = {
         .blocks = blocks,
         .def_block = def_block,
         .spec_mode = spec_mode,
      };
      bool hoisted = sim_visit_cf_list(&impl->body, &sim);
      bool fits = true;

      if (hoisted) {
         any_hoist = true;
         if (spec_mode == LICM_SPEC_ALL || spec_mode == LICM_SPEC_AGX2_INTEGER)
            any_useful |= sim.hoisted_non_const;
         else if (spec_mode == LICM_SPEC_SFU_CHAIN)
            any_useful |= sim.hoisted_sfu;
         else if (spec_mode == LICM_SPEC_CONST_LOAD_CHAIN)
            any_useful |= sim.hoisted_const_load;
         else
            any_useful = true;
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

         /* Speculative modes must keep even unchanged functions strictly
          * inside effective_limit so high-pressure loops never incur
          * non-NULL-filter load_const hoisting. */
         if (spec_mode != LICM_SPEC_NONE &&
             ALIGN_POT(func_post + est.reserve, est.align) >
                ROUND_DOWN_TO(MIN2(effective_limit, 256u), est.align)) {
            fits = false;
         } else {
            fits = demand_fits(&est, func_base, func_post, effective_limit) &&
                   loops_fit(&est, base_peaks, post_peaks, &impl->body,
                             effective_limit);
         }
      }

      ralloc_free(mem_ctx);

      if (!fits) {
         *any_hoist_out = false;
         return false;
      }
   }

   /* Non-NULL nir_opt_licm filters bypass the filter for load_const. Do not
    * enter a speculative mode when only constants would be hoisted (e.g. a
    * loop-varying SFU op in a direct loop body), or loop constants get pulled
    * into outer-block registers for zero ALU savings. */
   if (spec_mode != LICM_SPEC_NONE && !any_useful) {
      *any_hoist_out = false;
      return false;
   }

   *any_hoist_out = any_hoist;
   return true;
}

static bool
cf_list_has_direct_body_sfu(struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_loop) {
         if (loop_has_direct_body_sfu(nir_cf_node_as_loop(node)))
            return true;
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         if (cf_list_has_direct_body_sfu(&nif->then_list) ||
             cf_list_has_direct_body_sfu(&nif->else_list))
            return true;
      }
   }
   return false;
}

static bool
shader_has_direct_body_sfu(nir_shader *shader)
{
   nir_foreach_function_impl(impl, shader) {
      if (cf_list_has_direct_body_sfu(&impl->body))
         return true;
   }
   return false;
}

/*
 * Run nir_opt_licm if no loop or shader peak's estimated register demand would
 * exceed budget halves or cross an occupancy step with the invariants hoisted.
 * Low-pressure loops (<=48 NIR halves post-LICM) allow general speculative
 * hoisting of pure non-trapping ALU from direct loop bodies; medium-pressure
 * loops containing direct-body transcendentals (<=56 NIR halves post-LICM)
 * hoist only the invariant SFU chain so unrelated address math stays unhoisted
 * and full 1024-thread occupancy is preserved. All higher-pressure shaders
 * use exit-dominating blocks only (NULL filter).
 */
bool
agx_nir_opt_licm(nir_shader *shader, unsigned budget, bool cap_rt_compute)
{
   bool any_hoist = false;
   if (simulate_shader_licm(shader, budget, cap_rt_compute, LICM_SPEC_ALL,
                            &any_hoist)) {
      if (!any_hoist)
         return false;
      return nir_opt_licm(shader, agx_licm_filter_all);
   }

   if (shader_has_direct_body_sfu(shader) &&
       simulate_shader_licm(shader, budget, cap_rt_compute, LICM_SPEC_SFU_CHAIN,
                            &any_hoist)) {
      if (!any_hoist)
         return false;
      return nir_opt_licm(shader, agx_licm_filter_sfu);
   }

   if (simulate_shader_licm(shader, budget, cap_rt_compute, LICM_SPEC_NONE,
                            &any_hoist)) {
      if (!any_hoist)
         return false;
      return nir_opt_licm(shader, NULL);
   }

   return false;
}

bool
agx_nir_opt_licm_direct_sfu(nir_shader *shader, unsigned budget)
{
   if (!shader_has_direct_body_sfu(shader))
      return false;

   bool any_hoist = false;
   if (!simulate_shader_licm(shader, budget, false, LICM_SPEC_SFU_CHAIN,
                             &any_hoist) ||
       !any_hoist)
      return false;

   return nir_opt_licm(shader, agx_licm_filter_sfu);
}

bool
agx_nir_opt_licm_direct_const_loads(nir_shader *shader, unsigned budget)
{
   if (!shader_has_direct_body_const_load(shader))
      return false;

   bool any_hoist = false;
   if (!simulate_shader_licm(shader, budget, false, LICM_SPEC_CONST_LOAD_CHAIN,
                             &any_hoist) ||
       !any_hoist)
      return false;

   return nir_opt_licm(shader, agx_licm_filter_const_load);
}

bool
agx_nir_opt_licm_agx2_integer(nir_shader *shader, unsigned budget)
{
   bool any_hoist = false;
   if (!simulate_shader_licm(shader, budget, false, LICM_SPEC_AGX2_INTEGER,
                             &any_hoist) || !any_hoist)
      return false;

   return nir_opt_licm(shader, agx2_integer_licm_filter);
}

/*
 * Copyright 2023 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/list.h"
#include "agx_nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "nir_opcodes.h"

/* XXX: cribbed from nak, move to common */
static nir_def *
nir_udiv_round_up(nir_builder *b, nir_def *n, nir_def *d)
{
   return nir_udiv(b, nir_iadd(b, n, nir_iadd_imm(b, d, -1)), d);
}

struct agx_subgroup_lower_options {
   bool optimize_ballot_count;
   bool g17p_clustered_float;
   bool native_shuffle_xor;
   bool m5_float_minmax;
};

/* AGX3B (G17P) and AGX3C (M5) shuffle a 32-bit value across the subgroup by a
 * constant lane XOR or up/down shift in one instruction (simd_shuffle_{xor,up,down}
 * with an immediate 1..31), where the general shuffle is four broadcasts and
 * three selects. The mask/delta is usually a constant only once loops are
 * unrolled, so such shuffles are kept through the early lowering and
 * agx_nir_lower_dynamic_shuffle_xor() lowers the ones that are left with any
 * other operand.
 */
static bool
is_native_relative_shuffle(const nir_intrinsic_instr *intr)
{
   return (intr->intrinsic == nir_intrinsic_shuffle_xor ||
           intr->intrinsic == nir_intrinsic_shuffle_up ||
           intr->intrinsic == nir_intrinsic_shuffle_down) &&
          intr->def.num_components == 1 && intr->def.bit_size == 32;
}

static bool
has_native_relative_shuffle_delta(const nir_intrinsic_instr *intr)
{
   return nir_src_is_const(intr->src[1]) &&
          nir_src_as_uint(intr->src[1]) >= 1 &&
          nir_src_as_uint(intr->src[1]) <= 31;
}

/* Single-source dispatch for AGX's custom subgroup lowerings.
 * When `b == NULL`, reports whether `lower()` would rewrite `intr` as a
 * required legalization (excluding the optional popcount(ballot(true))
 * peephole, which leaves a legal native 1x32 ballot if unlowered).
 */
static bool
lower_custom_subgroup_intrinsic(
   nir_builder *b, nir_intrinsic_instr *intr,
   const struct agx_subgroup_lower_options *agx_options)
{
   if (b)
      b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_quad_vote_any: {
      if (!b)
         return true;
      nir_def *ballot = nir_quad_ballot_agx(b, 16, intr->src[0].ssa);
      nir_def_rewrite_uses(&intr->def, nir_ine_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_quad_vote_all: {
      if (!b)
         return true;
      nir_def *ballot =
         nir_quad_ballot_agx(b, 16, nir_inot(b, intr->src[0].ssa));
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_elect: {
      if (!b)
         return true;
      nir_def *active_id = nir_load_active_subgroup_invocation_agx(b, 16);
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, active_id, 0));
      return true;
   }

   case nir_intrinsic_first_invocation: {
      if (!b)
         return true;
      nir_def *active_id = nir_load_active_subgroup_invocation_agx(b, 16);
      nir_def *is_first = nir_ieq_imm(b, active_id, 0);
      nir_def *first_bit = nir_ballot(b, 1, 32, is_first);
      nir_def_rewrite_uses(&intr->def, nir_ufind_msb(b, first_bit));
      return true;
   }

   case nir_intrinsic_last_invocation: {
      if (!b)
         return true;
      nir_def *active_mask = nir_ballot(b, 1, 32, nir_imm_true(b));
      nir_def_rewrite_uses(&intr->def, nir_ufind_msb(b, active_mask));
      return true;
   }

   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_vote_feq: {
      if (!b)
         return true;
      /* The common lowering does:
       *
       *    vote_all(x == read_first(x))
       *
       * This is not optimal for AGX, since we have ufind_msb but not ctz, so
       * it's cheaper to read the last invocation than the first. So we do:
       *
       *    vote_all(x == read_last(x))
       *
       * implemented with lowered instructions as
       *
       *    ballot(x != broadcast(x, ffs(ballot(true)))) == 0
       */
      nir_def *active_mask = nir_ballot(b, 1, 32, nir_imm_true(b));
      nir_def *active_bit = nir_ufind_msb(b, active_mask);
      nir_def *other = nir_read_invocation(b, intr->src[0].ssa, active_bit);
      nir_def *is_ne;

      if (intr->intrinsic == nir_intrinsic_vote_feq) {
         is_ne = nir_fneu(b, other, intr->src[0].ssa);
      } else {
         is_ne = nir_ine(b, other, intr->src[0].ssa);
      }

      nir_def *ballot = nir_ballot(b, 1, 32, is_ne);
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_load_num_subgroups: {
      if (!b)
         return true;
      nir_def *workgroup_size = nir_load_workgroup_size(b);
      workgroup_size = nir_imul(b,
                                nir_imul(b, nir_channel(b, workgroup_size, 0),
                                         nir_channel(b, workgroup_size, 1)),
                                nir_channel(b, workgroup_size, 2));
      nir_def *subgroup_size = nir_imm_int(b, 32);
      nir_def *num_subgroups =
         nir_udiv_round_up(b, workgroup_size, subgroup_size);
      nir_def_rewrite_uses(&intr->def, num_subgroups);
      return true;
   }

   case nir_intrinsic_shuffle: {
      if (!b)
         return true;
      nir_def *data = intr->src[0].ssa;
      nir_def *target = intr->src[1].ssa;

      /* The hardware shuffle instruction chooses a single index within the
       * target quad to shuffle each source quad with. Consequently, the low
       * 2-bits of shuffle indices should not be quad divergent.  To implement
       * arbitrary shuffle, pull each low 2-bits index in the quad separately.
       */
      nir_def *quad_start = nir_iand_imm(b, target, 0x1c);
      nir_def *result = NULL;

      for (unsigned i = 0; i < 4; ++i) {
         nir_def *target_i = nir_iadd_imm(b, quad_start, i);
         nir_def *shuf = nir_read_invocation(b, data, target_i);

         if (result)
            result = nir_bcsel(b, nir_ieq(b, target, target_i), shuf, result);
         else
            result = shuf;
      }

      nir_def_rewrite_uses(&intr->def, result);
      return true;
   }

   case nir_intrinsic_inclusive_scan: {
      if (!b)
         return true;
      /* If we got here, we support the corresponding exclusive scan in
       * hardware, so just handle the last element.
       */
      nir_op red_op = nir_intrinsic_reduction_op(intr);
      nir_def *data = intr->src[0].ssa;

      b->cursor = nir_after_instr(&intr->instr);
      intr->intrinsic = nir_intrinsic_exclusive_scan;
      nir_def *accum = nir_build_alu2(b, red_op, data, &intr->def);
      nir_def_rewrite_uses_after(&intr->def, accum);
      return true;
   }

   case nir_intrinsic_ballot: {
      if (!b)
         return false;
      /* Optimize popcount(ballot(true)) to load_active_subgroup_count_agx().
       *
       * Preserve ballot membership on G17P and in M5 fragment shaders. M5
       * lowers the active count to a reduction, which also counts helper
       * invocations after fragment termination.
       */
      if (!agx_options->optimize_ballot_count || !nir_src_is_const(intr->src[0]) ||
          !nir_src_as_bool(intr->src[0]) ||
          !list_is_singular(&intr->def.uses))
         return false;

      nir_src *use = list_first_entry(&intr->def.uses, nir_src, use_link);
      nir_instr *parent = nir_src_use_instr(use);
      if (parent->type != nir_instr_type_alu)
         return false;

      nir_alu_instr *alu = nir_instr_as_alu(parent);
      if (alu->op != nir_op_bit_count)
         return false;

      nir_def_rewrite_uses(&alu->def,
                           nir_load_active_subgroup_count_agx(b, 32));
      return true;
   }

   default:
      return false;
   }
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   return lower_custom_subgroup_intrinsic(b, intr, data);
}



static bool
lower_subgroup_filter(const nir_intrinsic_instr *intr, const void *data)
{
   const struct agx_subgroup_lower_options *agx_options = data;
   unsigned cluster = 0;

   if (agx_options->native_shuffle_xor && is_native_relative_shuffle(intr))
      return false;

   /* Use default behaviour for everything but scans */
   if (intr->intrinsic != nir_intrinsic_exclusive_scan &&
       intr->intrinsic != nir_intrinsic_inclusive_scan &&
       intr->intrinsic != nir_intrinsic_reduce)
      return true;

   if (intr->def.num_components > 1 || intr->def.bit_size == 1)
      return true;

   /* Hardware supports quad ops but no other support clustered reductions. */
   if (nir_intrinsic_has_cluster_size(intr)) {
      cluster = nir_intrinsic_cluster_size(intr);
      if (cluster && cluster != 4 && cluster < 32)
         return true;
   }

   switch (nir_intrinsic_reduction_op(intr)) {
   case nir_op_imul:
      return true;

   case nir_op_fmin:
   case nir_op_fmax:
      /* Fresh G17P witnesses pin the native qRedfmin/qRedfmax forms. Keep
       * every other target and non-quad shape on the established common
       * lowering path. */
      if (agx_options->g17p_clustered_float && cluster == 4 &&
          intr->def.bit_size == 32)
         return false;
      /* M5's subgroup reduce takes float minimum and maximum: keep a
       * reduction over whole subgroups (no cluster, or one of 32 or more)
       * for it, in place of a shuffle tree of five shuffles and compares.
       * It reduces the active lanes only. The tree it replaces is built
       * from fmin/fmax with the builder's default float controls, which do
       * not preserve signed zeros or NaNs, and the instruction matches
       * it: NaN lanes are ignored unless every active lane is NaN, either
       * zero may come back for a group with zeros of both signs, and
       * denormals compare as zeros of their sign (the result may be a
       * denormal operand). */
      if (agx_options->m5_float_minmax && (cluster == 0 || cluster >= 32) &&
          intr->intrinsic == nir_intrinsic_reduce && intr->def.bit_size == 32)
         return false;
      return true;

   case nir_op_iadd:
   case nir_op_iand:
   case nir_op_ixor:
   case nir_op_ior:
      /* these have dedicated 64-bit lowering paths that use the 32-bit hardware
       * instructions so are likely better than the full lowering.
       */
      return false;

   default:
      /* otherwise, lower 64-bit, since the hw ops are at most 32-bit. */
      return intr->def.bit_size == 64;
   }
}

/* AGX float min/max return one of their operands' bits, so
 * agx_nir_lower_fminmax() flushes their results when the shader asks for
 * 32-bit denormals to be flushed. A subgroup min/max can return a lane's own
 * value without any min/max, as a reduction over one active lane or in
 * clusters of one does, so flush the value it reduces first. A value already
 * flushed is left alone, so every subgroup lowering can run this first.
 */
static bool
is_flushed(nir_def *def)
{
   if (!nir_def_is_alu(def))
      return false;
   nir_alu_instr *alu = nir_def_as_alu(def);
   if (alu->op == nir_op_fcanonicalize)
      return true;
   return alu->op == nir_op_fadd && nir_src_is_const(alu->src[1].src) &&
          nir_src_comp_as_uint(alu->src[1].src, alu->src[1].swizzle[0]) ==
             0x80000000u;
}

static bool
flush_reduced_value(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if ((intr->intrinsic != nir_intrinsic_reduce &&
        intr->intrinsic != nir_intrinsic_inclusive_scan &&
        intr->intrinsic != nir_intrinsic_exclusive_scan) ||
       intr->def.bit_size != 32)
      return false;

   const nir_op op = nir_intrinsic_reduction_op(intr);
   if ((op != nir_op_fmin && op != nir_op_fmax) || is_flushed(intr->src[0].ssa))
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   b->fp_math_ctrl = nir_fp_no_fast_math;
   nir_src_rewrite(&intr->src[0], nir_fadd_imm(b, intr->src[0].ssa, -0.0));
   return true;
}

bool
agx_nir_flush_subgroup_minmax(nir_shader *s)
{
   if (!nir_is_denorm_flush_to_zero(s->info.float_controls_execution_mode, 32))
      return false;

   return nir_shader_intrinsics_pass(s, flush_reduced_value,
                                     nir_metadata_control_flow, NULL);
}

static struct agx_subgroup_lower_options
agx_subgroup_options_for_target(const nir_shader *s, bool g17p, bool m5,
                                bool g18g)
{
   const bool compute = s->info.stage == MESA_SHADER_COMPUTE;

   /* The native float reduce treats denormals as zeros of their sign and does
    * not preserve them. Under DenormFlushToZero 32,
    * agx_nir_flush_subgroup_minmax() flushes every lane's FP32 input to zero
    * before the reduction, so M5's 1-instruction native simd_reduce produces
    * exact flushed results while DenormPreserve 32 keeps the shuffle tree.
    * Both native_shuffle_xor and m5_float_minmax encode identically across
    * compute and graphics stages on AGX3C. */
   const bool preserve32 =
      nir_is_denorm_preserve(s->info.float_controls_execution_mode, 32);
   const bool denorm32_exact =
      nir_is_denorm_flush_to_zero(s->info.float_controls_execution_mode, 32) ||
      preserve32;

   if (g17p) {
      return (struct agx_subgroup_lower_options){
         .optimize_ballot_count = false,
         .g17p_clustered_float = true,
         .native_shuffle_xor = compute,
         .m5_float_minmax = compute && !denorm32_exact,
      };
   } else if (m5) {
      return (struct agx_subgroup_lower_options){
         .optimize_ballot_count = s->info.stage != MESA_SHADER_FRAGMENT,
         .g17p_clustered_float = false,
         .native_shuffle_xor = !g18g,
         .m5_float_minmax = !g18g && !preserve32,
      };
   } else {
      return (struct agx_subgroup_lower_options){
         .optimize_ballot_count = true,
         .g17p_clustered_float = false,
         .native_shuffle_xor = false,
         .m5_float_minmax = false,
      };
   }
}

static bool
agx_nir_lower_subgroups_internal(nir_shader *s,
                                 struct agx_subgroup_lower_options agx_options)
{
   /* First, do as much common lowering as we can */
   nir_lower_subgroups_options opts = {
      .filter = lower_subgroup_filter,
      .filter_data = &agx_options,
      .lower_read_first_invocation = true,
      .lower_inverse_ballot = true,
      .lower_to_scalar = true,
      .lower_relative_shuffle = true,
      .lower_rotate_to_shuffle = true,
      .lower_subgroup_masks = true,
      .lower_reduce = true,
      .lower_vote = true,
      .ballot_components = 1,
      .ballot_bit_size = 32,
      .subgroup_size = 32,
   };

   bool progress = agx_nir_flush_subgroup_minmax(s);

   progress |= nir_lower_subgroups(s, &opts);

   /* Then do AGX-only lowerings on top */
   progress |= nir_shader_intrinsics_pass(s, lower, nir_metadata_control_flow,
                                          (void *)&agx_options);

   return progress;
}

bool
agx_nir_lower_subgroups_for_target(nir_shader *s, bool g17p, bool m5, bool g18g)
{
   return agx_nir_lower_subgroups_internal(
      s, agx_subgroup_options_for_target(s, g17p, m5, g18g));
}

bool
agx_nir_lower_subgroups(nir_shader *s)
{
   return agx_nir_lower_subgroups_for_target(s, false, false, false);
}

bool
agx_nir_lower_subgroups_g17p(nir_shader *s)
{
   return agx_nir_lower_subgroups_for_target(s, true, false, false);
}

bool
agx_nir_lower_subgroups_m5(nir_shader *s, bool native_shuffle_xor,
                           bool native_float_minmax)
{
   struct agx_subgroup_lower_options agx_options =
      agx_subgroup_options_for_target(s, false, true, false);
   agx_options.native_shuffle_xor &= native_shuffle_xor;
   agx_options.m5_float_minmax &= native_float_minmax;
   return agx_nir_lower_subgroups_internal(s, agx_options);
}

static bool
dynamic_shuffle_xor_filter(const nir_intrinsic_instr *intr, const void *data)
{
   return is_native_relative_shuffle(intr) &&
          !has_native_relative_shuffle_delta(intr);
}

/*
 * Lower the shuffles agx_nir_lower_subgroups_m5() kept whose mask did not
 * become a constant the instruction can hold.
 */
bool
agx_nir_lower_dynamic_shuffle_xor(nir_shader *s)
{
   const struct agx_subgroup_lower_options agx_options = {0};
   const nir_lower_subgroups_options opts = {
      .filter = dynamic_shuffle_xor_filter,
      .lower_relative_shuffle = true,
      .ballot_components = 1,
      .ballot_bit_size = 32,
      .subgroup_size = 32,
   };

   if (!nir_lower_subgroups(s, &opts))
      return false;

   nir_shader_intrinsics_pass(s, lower, nir_metadata_control_flow,
                              (void *)&agx_options);
   return true;
}

#ifndef NDEBUG
/* Report whether agx_nir_lower_subgroups_for_target() would lower `intr` as a
 * required legalization (excluding the optional popcount(ballot(true))
 * peephole). Used by agx_assert_post_lower_invariants() so the post-lower fast
 * path asserts against the lowering's own options and filter instead of a
 * hand-written intrinsic list that falsely rejects native 1x32 ballot / scalar
 * quad ops.
 */
bool
agx_nir_subgroup_intrinsic_needs_lowering(const nir_shader *s,
                                          const nir_intrinsic_instr *intr,
                                          bool g17p, bool m5, bool g18g)
{
   const struct agx_subgroup_lower_options agx_options =
      agx_subgroup_options_for_target(s, g17p, m5, g18g);

   /* Intrinsics lowered by AGX's own lower() pass (shared single switch). */
   if (lower_custom_subgroup_intrinsic(NULL, (nir_intrinsic_instr *)intr,
                                       &agx_options))
      return true;

   /* Common nir_lower_subgroups() only touches intrinsics accepted by
    * lower_subgroup_filter(). */
   if (!lower_subgroup_filter(intr, &agx_options))
      return false;

   /* Keep in sync with nir_lower_subgroups_options in
    * agx_nir_lower_subgroups_internal() above (verified by the debug
    * self-check test in test-agx3-pso-compile-fastpath.c). */
   switch (intr->intrinsic) {
   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all:
   case nir_intrinsic_load_subgroup_size:
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_load_subgroup_eq_mask:
   case nir_intrinsic_load_subgroup_ge_mask:
   case nir_intrinsic_load_subgroup_gt_mask:
   case nir_intrinsic_load_subgroup_le_mask:
   case nir_intrinsic_load_subgroup_lt_mask:
   case nir_intrinsic_inverse_ballot:
   case nir_intrinsic_ballot_bitfield_extract:
   case nir_intrinsic_ballot_bit_count_reduce:
   case nir_intrinsic_ballot_find_lsb:
   case nir_intrinsic_ballot_find_msb:
   case nir_intrinsic_ballot_bit_count_exclusive:
   case nir_intrinsic_ballot_bit_count_inclusive:
   case nir_intrinsic_shuffle_xor:
   case nir_intrinsic_shuffle_up:
   case nir_intrinsic_shuffle_down:
   case nir_intrinsic_shuffle_up_intel:
   case nir_intrinsic_shuffle_down_intel:
   case nir_intrinsic_reduce:
   case nir_intrinsic_exclusive_scan:
   case nir_intrinsic_rotate:
      return true;

   case nir_intrinsic_ballot:
      return intr->def.num_components != 1 || intr->def.bit_size != 32;

   case nir_intrinsic_read_invocation:
   case nir_intrinsic_quad_broadcast:
   case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical:
   case nir_intrinsic_quad_swap_diagonal:
      return intr->num_components > 1;

   default:
      return false;
   }
}
#endif

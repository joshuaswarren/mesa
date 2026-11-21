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

/* M5 shuffles a 32-bit value across the subgroup by a constant lane XOR in
 * one instruction (simd_shuffle_xor with an immediate), where the general
 * shuffle is four broadcasts and three selects. The mask is usually a
 * constant only once loops are unrolled, so such shuffles are kept through
 * the early lowering and agx_nir_lower_dynamic_shuffle_xor() lowers the ones
 * that are left with any other mask.
 */
static bool
is_native_shuffle_xor(const nir_intrinsic_instr *intr)
{
   return intr->intrinsic == nir_intrinsic_shuffle_xor &&
          intr->def.num_components == 1 && intr->def.bit_size == 32;
}

static bool
has_native_shuffle_xor_mask(const nir_intrinsic_instr *intr)
{
   return nir_src_is_const(intr->src[1]) &&
          nir_src_as_uint(intr->src[1]) >= 1 &&
          nir_src_as_uint(intr->src[1]) <= 31;
}

static bool
lower(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_quad_vote_any: {
      nir_def *ballot = nir_quad_ballot_agx(b, 16, intr->src[0].ssa);
      nir_def_rewrite_uses(&intr->def, nir_ine_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_quad_vote_all: {
      nir_def *ballot =
         nir_quad_ballot_agx(b, 16, nir_inot(b, intr->src[0].ssa));
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, ballot, 0));
      return true;
   }

   case nir_intrinsic_elect: {
      nir_def *active_id = nir_load_active_subgroup_invocation_agx(b, 16);
      nir_def_rewrite_uses(&intr->def, nir_ieq_imm(b, active_id, 0));
      return true;
   }

   case nir_intrinsic_first_invocation: {
      nir_def *active_id = nir_load_active_subgroup_invocation_agx(b, 16);
      nir_def *is_first = nir_ieq_imm(b, active_id, 0);
      nir_def *first_bit = nir_ballot(b, 1, 32, is_first);
      nir_def_rewrite_uses(&intr->def, nir_ufind_msb(b, first_bit));
      return true;
   }

   case nir_intrinsic_last_invocation: {
      nir_def *active_mask = nir_ballot(b, 1, 32, nir_imm_true(b));
      nir_def_rewrite_uses(&intr->def, nir_ufind_msb(b, active_mask));
      return true;
   }

   case nir_intrinsic_vote_ieq:
   case nir_intrinsic_vote_feq: {
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
      /* Optimize popcount(ballot(true)) to load_active_subgroup_count_agx().
       *
       * Preserve ballot membership on G17P and in M5 fragment shaders. M5
       * lowers the active count to a reduction, which also counts helper
       * invocations after fragment termination.
       */
      const struct agx_subgroup_lower_options *agx_options = data;
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
lower_subgroup_filter(const nir_intrinsic_instr *intr, const void *data)
{
   const struct agx_subgroup_lower_options *agx_options = data;
   unsigned cluster = 0;

   if (agx_options->native_shuffle_xor && is_native_shuffle_xor(intr))
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

static bool
agx_nir_lower_subgroups_internal(nir_shader *s, bool g17p_clustered_float,
                                 bool optimize_ballot_count,
                                 bool native_shuffle_xor, bool m5_float_minmax)
{
   const struct agx_subgroup_lower_options agx_options = {
      .optimize_ballot_count = optimize_ballot_count,
      .g17p_clustered_float = g17p_clustered_float,
      .native_shuffle_xor = native_shuffle_xor,
      .m5_float_minmax = m5_float_minmax,
   };

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
agx_nir_lower_subgroups(nir_shader *s)
{
   return agx_nir_lower_subgroups_internal(s, false, true, false, false);
}

bool
agx_nir_lower_subgroups_g17p(nir_shader *s)
{
   return agx_nir_lower_subgroups_internal(s, true, false, false, false);
}

bool
agx_nir_lower_subgroups_m5(nir_shader *s, bool native_shuffle_xor,
                           bool native_float_minmax)
{
   const bool compute = s->info.stage == MESA_SHADER_COMPUTE;

   /* The native float reduce does not flush denormal results, so it is not
    * used when the shader asks for 32-bit denormals to be flushed: those
    * reductions keep the tree, of values agx_nir_flush_subgroup_minmax()
    * has flushed.
    */
   const bool ftz32 =
      nir_is_denorm_flush_to_zero(s->info.float_controls_execution_mode, 32);

   return agx_nir_lower_subgroups_internal(
      s, false, s->info.stage != MESA_SHADER_FRAGMENT,
      native_shuffle_xor && compute, native_float_minmax && compute && !ftz32);
}

static bool
dynamic_shuffle_xor_filter(const nir_intrinsic_instr *intr, const void *data)
{
   return is_native_shuffle_xor(intr) && !has_native_shuffle_xor_mask(intr);
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

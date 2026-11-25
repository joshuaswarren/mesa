/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "compiler/list.h"
#include "compiler/nir/nir_builder.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "nir.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"

/*
 * sample_mask takes two bitmasks as arguments, TARGET and LIVE. Each bit refers
 * to an indexed sample. Roughly, the instruction does:
 *
 *    foreach sample in TARGET {
 *       if sample in LIVE {
 *          run depth/stencil/occlusion test/update
 *       } else {
 *          kill sample
 *       }
 *    }
 *
 * As a special case, TARGET may be set to all-1s (~0) to refer to all samples
 * regardless of the framebuffer sample count.
 *
 * For example, to discard an entire pixel unconditionally, we could run:
 *
 *    sample_mask ~0, 0
 *
 * sample_mask must follow these rules:
 *
 * 1. All sample_mask instructions affecting a sample must execute before a
 *    local_store_pixel instruction targeting that sample. This ensures that
 *    nothing is written for discarded samples (whether discarded in shader or
 *    due to a failed depth/stencil test).
 *
 * 2. If sample_mask is used anywhere in a shader, then on every execution path,
 *    every sample must be killed or else run depth/stencil tests exactly ONCE.
 *
 * 3. If a sample is killed, future sample_mask instructions have
 *    no effect on that sample. The following code sequence correctly implements
 *    a conditional discard (if there are no other sample_mask instructions in
 *    the shader):
 *
 *       sample_mask discarded, 0
 *       sample_mask ~0, ~0
 *
 *    but this sequence is incorrect:
 *
 *       sample_mask ~0, ~discarded
 *       sample_mask ~0, ~0         <-- incorrect: depth/stencil tests run twice
 *
 * 4. Conversely, if a sample is tested, future sample_mask instructions may not
 *    discard that sample. The following code is invalid:
 *
 *      sample_mask ~0, ~0
 *      sample_mask discarded, 0
 *      st_tile
 *
 *    To implement the semantic of "force early tests with discard", manual
 *    colour masking must be used. It's a weird case, but CTS does it.
 *
 * 5. zs_emit may be used in the shader exactly once to trigger tests.
 * sample_mask with 0 may be used to discard early.
 *
 * 6. On G17P a SIMD group may run only one testing sample_mask, even if two
 *    cover disjoint lanes. Testing in both arms of an if loses a few percent
 *    of the depth writes of groups that take both arms, differently every
 *    frame, and can hang the 3D stage. Kill-only updates are not limited.
 *
 * This pass lowers demote_samples to sample_mask instructions satisfying these
 * rules. Other passes should not generate sample_mask instructions, as there
 * are too many footguns.
 */

#define ALL_SAMPLES (0xFF)
#define BASE_Z      1
#define BASE_S      2

static bool
lower_discard_to_sample_mask_0(nir_builder *b, nir_intrinsic_instr *intr,
                               UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_demote_samples)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_sample_mask_agx(b, nir_u2u16(b, intr->src[0].ssa),
                       nir_imm_intN_t(b, 0, 16));
   nir_instr_remove(&intr->instr);
   return true;
}

/* M5 (G17G/G17S) form of the final coverage update. Native M5 fragments
 * (native G17G compiler: discard + alpha-to-coverage at 4x) issue exactly
 * one testing MOVMSK, unconditionally, after their kills; a lane whose live
 * set is empty still executes it. Whether this shader part triggers the
 * tests is the uniform tests_zs (another linked part may own them), so pick
 * the testing or the pure-kill form with a uniform branch instead of
 * splitting per lane (agx_nir_split_g17p_sample_tests): a SIMD group never
 * reaches the testing update with only some of its live lanes. */
static void
final_coverage_update(nir_builder *b, nir_def *killed, bool uniform_tests)
{
   nir_def *all_samples = nir_imm_intN_t(b, ALL_SAMPLES, 16);
   nir_def *live = nir_ixor(b, killed, all_samples);
   nir_def *testing = nir_load_shader_part_tests_zs_agx(b);

   if (!uniform_tests) {
      nir_sample_mask_agx(b, nir_ior(b, testing, killed), live);
      return;
   }

   nir_push_if(b, nir_ine_imm(b, testing, 0));
   nir_sample_mask_agx(b, all_samples, live);
   nir_push_else(b, NULL);
   nir_sample_mask_agx(b, killed, nir_imm_intN_t(b, 0, 16));
   nir_pop_if(b, NULL);
}

static nir_intrinsic_instr *
last_discard_in_block(nir_block *block)
{
   nir_foreach_instr_reverse(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_demote_samples)
         return intr;
   }

   return NULL;
}

static bool
cf_node_contains_discard(nir_cf_node *node)
{
   nir_foreach_block_in_cf_node(block, node) {
      if (last_discard_in_block(block))
         return true;
   }

   return false;
}

/*
 * We want to run depth/stencil tests as early as possible, but we have to
 * wait until after the last discard. We find the last discard and
 * execute depth/stencil tests in the first unconditional block after (if
 * in conditional control flow), or fuse depth/stencil tests into the
 * sample instruction (if in unconditional control flow).
 *
 * To do so, we walk the root control flow list backwards, looking for the
 * earliest unconditionally executed instruction after all discard.
 */
static void
run_tests_after_last_discard(nir_builder *b, bool uniform_tests)
{
   foreach_list_typed_reverse(nir_cf_node, node, node, &b->impl->body) {
      if (node->type == nir_cf_node_block) {
         /* Unconditionally executed block */
         nir_block *block = nir_cf_node_as_block(node);
         nir_intrinsic_instr *intr = last_discard_in_block(block);

         if (intr) {
            /* Last discard is executed unconditionally, so fuse tests:
             *
             *    sample_mask (testing | killed), ~killed
             *
             * When testing, this is `sample_mask ~0, ~killed` which kills the
             * kill set and triggers tests on the rest.
             *
             * When not testing, this is `sample_mask killed, ~killed` which is
             * equivalent to `sample_mask killed, 0`, killing without testing.
             */
            b->cursor = nir_before_instr(&intr->instr);
            final_coverage_update(b, nir_u2u16(b, intr->src[0].ssa),
                                  uniform_tests);
            nir_instr_remove(&intr->instr);
            return;
         } else {
            /* Set cursor for insertion due to a preceding conditionally
             * executed discard.
             */
            b->cursor = nir_before_block_after_phis(block);
         }
      } else if (cf_node_contains_discard(node)) {
         /* Conditionally executed block contains the last discard. Test
          * depth/stencil for remaining samples in unconditional code after.
          *
          * If we're not testing, this turns into sample_mask(0, ~0) which is a
          * no-op.
          */
         if (uniform_tests) {
            nir_push_if(b, nir_ine_imm(b, nir_load_shader_part_tests_zs_agx(b), 0));
            nir_sample_mask_agx(b, nir_imm_intN_t(b, ALL_SAMPLES, 16),
                                nir_imm_intN_t(b, ALL_SAMPLES, 16));
            nir_pop_if(b, NULL);
         } else {
            nir_sample_mask_agx(b, nir_load_shader_part_tests_zs_agx(b),
                                nir_imm_intN_t(b, ALL_SAMPLES, 16));
         }
         return;
      }
   }
}

static void
run_tests_at_start(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   nir_sample_mask_agx(&b, nir_imm_intN_t(&b, ALL_SAMPLES, 16),
                       nir_imm_intN_t(&b, ALL_SAMPLES, 16));
}

static bool
lower_sample_mask(nir_shader *shader, bool uniform_tests)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   bool writes_zs =
      shader->info.outputs_written &
      (BITFIELD64_BIT(FRAG_RESULT_STENCIL) | BITFIELD64_BIT(FRAG_RESULT_DEPTH));

   if (shader->info.fs.early_fragment_tests) {
      /* run tests early, if we need testing */
      if (shader->info.fs.uses_discard || writes_zs ||
          shader->info.writes_memory) {

         run_tests_at_start(shader);
      }
   } else if (shader->info.fs.uses_discard) {
      /* If we have zs_emit, the tests will be triggered by zs_emit, otherwise
       * we need to trigger tests explicitly. Allow sample_mask with zs_emit.
       */
      if (!writes_zs) {
         nir_builder b = nir_builder_create(impl);

         /* run tests late */
         run_tests_after_last_discard(&b, uniform_tests);
      }
   } else {
      /* regular shaders that don't use discard have nothing to lower */
      return nir_no_progress(impl);
   }

   nir_progress(true, impl,
                uniform_tests ? nir_metadata_none : nir_metadata_control_flow);

   nir_shader_intrinsics_pass(shader, lower_discard_to_sample_mask_0,
                              nir_metadata_control_flow, NULL);

   return true;
}


/* M5 shader parts: one uniformly selected final coverage update. */
bool
agx_nir_lower_sample_mask_m5(nir_shader *shader)
{
   return lower_sample_mask(shader, true);
}

/* The split shader's testing flag is a whole-sample boolean (~0/0), not an
 * arbitrary sample set. Preserve that provenance before sysval lowering and
 * preamble promotion. G18 has witnessed full testing updates and non-testing
 * kills, so mark the fused update for selection between those exact words:
 *
 *   sample_mask(testing | killed, killed ^ 0xff)
 *     testing: sample_mask(0xff, killed ^ 0xff)
 *     otherwise: sample_mask(killed, 0)
 *
 * The latter has disjoint TARGET/LIVE in the original and therefore performs
 * no tests. Keep TARGET/LIVE and control flow unchanged: branching around a
 * coverage operation could skip the pixel transaction entry on one path.
 * Do not apply this to an arbitrary partial-target testing update.
 */
static bool
mark_g18_sample_mask_tests(UNUSED nir_builder *b, nir_intrinsic_instr *intr,
                            UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_sample_mask_agx ||
       nir_intrinsic_flags(intr) || intr->src[0].ssa->bit_size != 16 ||
       intr->src[1].ssa->bit_size != 16)
      return false;

   nir_alu_instr *target = nir_src_as_alu(intr->src[0]);
   nir_alu_instr *live = nir_src_as_alu(intr->src[1]);
   bool subset = false;
   if (target && target->op == nir_op_iand) {
      unsigned constant = nir_src_is_const(target->src[0].src) ? 0 : 1;
      if (!nir_src_is_const(target->src[constant].src))
         return false;
      unsigned bits = nir_src_as_uint(target->src[constant].src);
      if (!bits || (bits & ~ALL_SAMPLES))
         return false;
      target = nir_src_as_alu(target->src[constant ^ 1].src);
      subset = true;
   }
   if (!target || target->op != nir_op_ior || !live ||
       live->op != nir_op_ixor)
      return false;

   for (unsigned t = 0; t < 2; ++t) {
      nir_intrinsic_instr *testing =
         nir_src_as_intrinsic(target->src[t].src);
      if (!testing || testing->intrinsic !=
                         nir_intrinsic_load_shader_part_tests_zs_agx)
         continue;

      nir_src killed = target->src[1 - t].src;
      for (unsigned k = 0; k < 2; ++k) {
         if (!nir_srcs_equal(live->src[k].src, killed) ||
             !nir_src_is_const(live->src[1 - k].src) ||
             nir_src_as_uint(live->src[1 - k].src) != ALL_SAMPLES)
            continue;

         nir_intrinsic_set_flags(intr, subset ? AGX_SAMPLE_MASK_SUBSET_FULL_OR_KILL
                                               : AGX_SAMPLE_MASK_FULL_OR_KILL);
         return true;
      }
   }

   return false;
}

bool
agx_nir_mark_g18_sample_mask_tests(nir_shader *shader)
{
   return nir_shader_intrinsics_pass(shader, mark_g18_sample_mask_tests,
                                     nir_metadata_all, NULL);
}

struct deferred_ctx {
   nir_variable *killed;
};

static bool
accumulate_demote(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct deferred_ctx *ctx = data;
   if (intr->intrinsic != nir_intrinsic_demote_samples)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *killed = nir_u2u16(b, intr->src[0].ssa);
   nir_store_var(b, ctx->killed,
                 nir_ior(b, nir_load_var(b, ctx->killed), killed),
                 nir_component_mask(1));
   nir_instr_remove(&intr->instr);
   return true;
}

/*
 * Deferring every demote to one end-of-shader coverage update keeps demoted
 * lanes alive as quad helpers, but it also moves the depth/stencil test to
 * the end of the shader, so fragments that fail the test still run the whole
 * shader. When demoted lanes provably never execute an operation that reads
 * other lanes of the quad or subgroup after their demote, keeping them alive
 * has no observable effect and the immediate lowering is equivalent. It
 * kills and tests right after the last discard, as Apple's G17P code does
 * with MOVMSK (e.g. HLSL/OpKill "if (c) discard; if (!c) { ... }").
 *
 * The proof is structural and conservative: demotes and the guarded code
 * must sit in top-level control flow, and every later quad-dependent
 * instruction must be inside a top-level if whose branch excludes the
 * demoted lanes by an exact boolean negation of the demote's guard. A G17P
 * texture result carried across a later loop is also lane-dependent: hardware
 * drops intermittent fragment groups if a coverage kill precedes that shape,
 * even when the loop executes zero iterations. That did not reproduce on G17P
 * with one testing update per group (see update_coverage_after_last_demote),
 * but the other chips here have not been retested.
 */
static bool
derivative_instr(const nir_instr *instr)
{
   if (instr->type == nir_instr_type_tex)
      return nir_tex_instr_has_implicit_derivative(nir_instr_as_tex(instr));
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   switch (nir_instr_as_intrinsic(instr)->intrinsic) {
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddy_coarse:
      return true;
   default:
      return false;
   }
}

static bool
lane_state_instr(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic || derivative_instr(instr))
      return false;

   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (nir_intrinsic_infos[intr->intrinsic].flags &
       (NIR_INTRINSIC_SUBGROUP | NIR_INTRINSIC_QUADGROUP))
      return true;

   switch (intr->intrinsic) {
   case nir_intrinsic_is_helper_invocation:
   case nir_intrinsic_load_helper_invocation:
   case nir_intrinsic_load_active_subgroup_invocation_agx:
   case nir_intrinsic_load_active_subgroup_count_agx:
   case nir_intrinsic_quad_ballot_agx:
      return true;
   default:
      return false;
   }
}

static bool
quad_dependent_instr(const nir_instr *instr)
{
   return derivative_instr(instr) || lane_state_instr(instr);
}

static bool
is_demote(const nir_instr *instr)
{
   return instr->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(instr)->intrinsic ==
             nir_intrinsic_demote_samples;
}

static bool
is_texture(const nir_instr *instr)
{
   return instr->type == nir_instr_type_tex;
}

static bool
cf_list_has(struct exec_list *list, bool (*pred)(const nir_instr *))
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      nir_foreach_block_in_cf_node(block, node) {
         nir_foreach_instr(instr, block) {
            if (pred(instr))
               return true;
         }
      }
   }
   return false;
}

static bool
cf_node_has(nir_cf_node *node, bool (*pred)(const nir_instr *))
{
   nir_foreach_block_in_cf_node(block, node) {
      nir_foreach_instr(instr, block) {
         if (pred(instr))
            return true;
      }
   }
   return false;
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
cf_node_has_loop(nir_cf_node *node)
{
   if (node->type == nir_cf_node_loop)
      return true;

   if (node->type == nir_cf_node_if) {
      nir_if *nif = nir_cf_node_as_if(node);
      return cf_list_has_loop(&nif->then_list) ||
             cf_list_has_loop(&nif->else_list);
   }

   return false;
}

static nir_alu_instr *
bool_alu(nir_def *def)
{
   return def->bit_size == 1 && def->num_components == 1
             ? nir_def_as_alu_or_null(def)
             : NULL;
}

static bool
same_srcs(const nir_alu_instr *a, const nir_alu_instr *b, bool swapped)
{
   return nir_op_infos[a->op].num_inputs == 2 &&
          nir_op_infos[b->op].num_inputs == 2 &&
          nir_alu_srcs_equal(a, b, 0, swapped ? 1 : 0) &&
          nir_alu_srcs_equal(a, b, 1, swapped ? 0 : 1);
}

/* a == !b, exactly (1-bit booleans only). */
static bool
bool_negation(nir_def *a, nir_def *b)
{
   nir_alu_instr *x = bool_alu(a), *y = bool_alu(b);
   if (x && x->op == nir_op_inot && x->src[0].src.ssa == b &&
       x->src[0].swizzle[0] == 0)
      return true;
   if (y && y->op == nir_op_inot && y->src[0].src.ssa == a &&
       y->src[0].swizzle[0] == 0)
      return true;
   if (!x || !y)
      return false;

   static const nir_op pairs[][2] = {
      {nir_op_ieq, nir_op_ine},
      {nir_op_feq, nir_op_fneu},
      {nir_op_ilt, nir_op_ige},
      {nir_op_ult, nir_op_uge},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(pairs); ++i) {
      for (unsigned o = 0; o < 2; ++o) {
         if (x->op == pairs[i][o] && y->op == pairs[i][!o]) {
            const bool commutative = i < 2;
            if (same_srcs(x, y, false) ||
                (commutative && same_srcs(x, y, true)))
               return true;
         }
      }
   }
   return false;
}

static bool
bool_equal(nir_def *a, nir_def *b)
{
   if (a == b)
      return true;
   nir_alu_instr *x = bool_alu(a);
   return x && x->op == nir_op_inot && x->src[0].swizzle[0] == 0 &&
          bool_negation(x->src[0].src.ssa, b);
}

static bool
demoted_lanes_reach_sensitive_work(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   const bool has_texture = cf_list_has(&impl->body, is_texture);
   bool seen = false, all = false;
   nir_def *guard = NULL;
   bool guard_negated = false;

   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      if (cf_node_has(node, is_demote)) {
         if (has_texture && cf_node_has_loop(node))
            return true;

         if (node->type == nir_cf_node_block) {
            /* Unconditionally executed demote. demote_if(c) arrives here as
             * demote_samples(bcsel(c, samples, 0)), which demotes only lanes
             * satisfying c. No quad operation may follow it in the same
             * block. */
            /* A previous top-level node may already have demoted lanes.
             * They can reach quad operations before this block's first
             * demote as well as those after it. */
            bool after = seen;
            nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
               if (is_demote(instr)) {
                  nir_def *mask = nir_instr_as_intrinsic(instr)->src[0].ssa;
                  nir_alu_instr *sel = nir_def_as_alu_or_null(mask);
                  nir_def *cond = NULL;
                  bool negated = false;
                  if (sel && sel->op == nir_op_bcsel &&
                      sel->src[0].swizzle[0] == 0) {
                     if (nir_src_is_const(sel->src[2].src) &&
                         nir_src_as_uint(sel->src[2].src) == 0) {
                        cond = sel->src[0].src.ssa;
                     } else if (nir_src_is_const(sel->src[1].src) &&
                                nir_src_as_uint(sel->src[1].src) == 0) {
                        cond = sel->src[0].src.ssa;
                        negated = true;
                     }
                  }
                  if (!cond)
                     all = true;
                  else if (!seen && !guard) {
                     guard = cond;
                     guard_negated = negated;
                  } else if (!(guard == cond && guard_negated == negated))
                     all = true;
                  after = true;
               } else if (after && quad_dependent_instr(instr)) {
                  return true;
               }
            }
         } else if (node->type == nir_cf_node_if) {
            nir_if *nif = nir_cf_node_as_if(node);
            const bool in_then = cf_list_has(&nif->then_list, is_demote);
            const bool in_else = cf_list_has(&nif->else_list, is_demote);
            if (cf_node_has(node, quad_dependent_instr))
               return true;
            nir_def *cond = nif->condition.ssa;
            if (in_then && in_else) {
               all = true;
            } else if (!seen && !guard) {
               guard = cond;
               guard_negated = in_else;
            } else if (!(guard && cond == guard &&
                         guard_negated == in_else)) {
               all = true;
            }
         } else {
            return true;
         }
         seen = true;
         continue;
      }

      if (!seen)
         continue;

      if (has_texture && cf_node_has_loop(node))
         return true;

      if (!cf_node_has(node, quad_dependent_instr))
         continue;
      if (all || node->type != nir_cf_node_if)
         return true;

      /* Demoted lanes satisfy `guard` (or its negation). A branch excludes
       * them if its condition contradicts that. */
      nir_if *nif = nir_cf_node_as_if(node);
      nir_def *cond = nif->condition.ssa;
      const bool then_excluded = guard_negated ? bool_equal(cond, guard)
                                               : bool_negation(cond, guard);
      const bool else_excluded = guard_negated ? bool_negation(cond, guard)
                                               : bool_equal(cond, guard);
      if ((cf_list_has(&nif->then_list, quad_dependent_instr) &&
           !then_excluded) ||
          (cf_list_has(&nif->else_list, quad_dependent_instr) &&
           !else_excluded))
         return true;
   }
   return false;
}

/* Whether a sampling shader runs a loop in or after its first demoting
 * top-level node. A coverage kill ahead of that loop is the shape that drops
 * fragment groups on G17P (see demoted_lanes_reach_sensitive_work), even a
 * kill-only update for quads with no live lane left. */
static bool
demote_precedes_sampled_loop(nir_function_impl *impl)
{
   if (!cf_list_has(&impl->body, is_texture))
      return false;

   bool seen = false;
   foreach_list_typed(nir_cf_node, node, node, &impl->body) {
      seen |= cf_node_has(node, is_demote);
      if (seen && cf_node_has_loop(node))
         return true;
   }
   return false;
}

/* Non-quad subgroup operations, whose active set must not lose demoted
 * lanes earlier than the deferred lowering would. */
static bool
subgroup_instr(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   const nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
   if (op == nir_intrinsic_load_active_subgroup_invocation_agx ||
       op == nir_intrinsic_load_active_subgroup_count_agx)
      return true;
   /* Quad operations are also flagged SUBGROUP; they only read their own
    * quad, which a dead-quad kill never leaves partially alive. */
   const unsigned flags =
      nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic].flags;
   return (flags & NIR_INTRINSIC_SUBGROUP) &&
          !(flags & NIR_INTRINSIC_QUADGROUP);
}

/* Whether every active lane of the quad has all samples in `killed`.
 * Spelled with the AGX quad ballot directly (subgroup lowering has already
 * run when hk lowers discards) and as a compare against zero, the quad
 * ballot form G17P encodes. */
static nir_def *
quad_fully_killed(nir_builder *b, nir_def *killed)
{
   nir_def *all = nir_imm_intN_t(b, ALL_SAMPLES, 16);
   nir_def *missing = nir_ixor(b, nir_iand(b, killed, all), all);
   return nir_ieq_imm(b, nir_quad_ballot_agx(b, 16, nir_ine_imm(b, missing, 0)),
                      0);
}

/*
 * A fully demoted quad has no live lane left to read its helper values, so
 * nothing observes whether it keeps running. Once the accumulated kill mask
 * covers every sample of every active lane of a quad, kill those samples
 * right away (kill only: demoted samples are never tested) instead of
 * running the rest of the shader for them. Partially demoted quads keep
 * their demoted lanes as helpers until the deferred coverage update; a
 * rasterizer helper lane counts as dead only if it demoted too. Shaders with
 * non-quad subgroup operations keep every demoted lane, since those lanes
 * would leave the active set early. The vote runs in top-level control
 * flow, after the node that demoted, where every non-killed lane of the quad
 * is active.
 */
static void
kill_dead_quads(nir_function_impl *impl, nir_variable *killed)
{
   foreach_list_typed_safe(nir_cf_node, node, node, &impl->body) {
      if (!cf_node_has(node, is_demote) || node->type == nir_cf_node_block)
         continue;
      nir_builder b = nir_builder_at(nir_after_cf_node(node));
      nir_def *k = nir_load_var(&b, killed);
      nir_def *dead = quad_fully_killed(&b, k);
      nir_sample_mask_agx(&b, nir_bcsel(&b, dead, k, nir_imm_intN_t(&b, 0, 16)),
                          nir_imm_intN_t(&b, 0, 16));
   }
}

/* The most instructions a lane can run after its last demote in this list. */
static unsigned
instrs_after_last_demote(struct exec_list *list)
{
   unsigned instrs = 0;

   foreach_list_typed_reverse(nir_cf_node, node, node, list) {
      if (!cf_node_has(node, is_demote)) {
         nir_foreach_block_in_cf_node(block, node)
            instrs += exec_list_length(&block->instr_list);
      } else if (node->type == nir_cf_node_block) {
         nir_foreach_instr_reverse(instr, nir_cf_node_as_block(node)) {
            if (is_demote(instr))
               break;
            instrs++;
         }
         return instrs;
      } else if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         return instrs + MAX2(instrs_after_last_demote(&nif->then_list),
                              instrs_after_last_demote(&nif->else_list));
      } else {
         return instrs;
      }
   }

   return instrs;
}

static nir_cf_node *
last_demote_node(struct exec_list *list)
{
   foreach_list_typed_reverse(nir_cf_node, node, node, list) {
      if (cf_node_has(node, is_demote))
         return node;
   }

   return NULL;
}

/* Whether lanes have enough left to run after their last demote for moving
 * it below the coverage update to pay. */
static bool
worth_moving(struct exec_list *list)
{
   return instrs_after_last_demote(list) >= 64;
}

static bool
worth_splitting(nir_cf_node *node)
{
   if (!node || node->type != nir_cf_node_if)
      return false;

   nir_if *nif = nir_cf_node_as_if(node);
   return worth_moving(&nif->then_list) || worth_moving(&nif->else_list);
}

static nir_if *split_if_after_last_demote(nir_builder *b, nir_if *nif);

/* Where the work after a demoting node starts. */
static nir_cursor
after_demoting_node(nir_cf_node *node)
{
   if (node->type == nir_cf_node_block) {
      return nir_after_instr(
         &last_discard_in_block(nir_cf_node_as_block(node))->instr);
   }

   return nir_after_cf_node_and_phis(node);
}

/* Move what follows the last demote of the list to the builder's cursor. */
static void
move_after_last_demote(nir_builder *b, struct exec_list *list)
{
   nir_cf_node *node = last_demote_node(list);

   if (worth_splitting(node))
      split_if_after_last_demote(b, nir_cf_node_as_if(node));

   nir_cursor begin = node ? after_demoting_node(node) : nir_before_cf_list(list);
   nir_cf_list rest;
   nir_cf_extract(&rest, begin, nir_after_cf_list(list));
   b->cursor = nir_cf_reinsert(&rest, b->cursor);
}

/* Give what follows the demotes of an if a second if of its own on the same
 * condition, built at the cursor. An arm with little left stays above the
 * update, or the values the other arm carries across it would stay live
 * through that arm too. The merge moves, so its phis go through registers;
 * the caller puts SSA back afterwards. */
static nir_if *
split_if_after_last_demote(nir_builder *b, nir_if *nif)
{
   nir_lower_phis_to_regs_block(
      nir_cf_node_as_block(nir_cf_node_next(&nif->cf_node)), true);

   nir_if *rest = nir_push_if(b, nif->condition.ssa);
   if (worth_moving(&nif->then_list))
      move_after_last_demote(b, &nif->then_list);
   nir_push_else(b, rest);
   if (worth_moving(&nif->else_list))
      move_after_last_demote(b, &nif->else_list);
   nir_pop_if(b, rest);
   return rest;
}

/*
 * Issue the one final coverage update right after the last demote instead of
 * at the end of the shader, where every lane of a SIMD group reaches it
 * together: two testing updates in one group, even on disjoint lanes, lose
 * depth writes and fragments and can hang the 3D stage. That rules out an
 * update per arm of an if, so a demote inside an if with work left below it
 * splits the if:
 *
 *    if (c) { A; demote; B } else { C }
 *
 * becomes, when C is small,
 *
 *    if (c) { A; demote } else { C }
 *    update
 *    if (c) { B }
 *
 * Returns whether it split anything.
 */
static bool
update_coverage_after_last_demote(nir_builder *b, nir_function_impl *impl,
                                  nir_variable *killed, nir_def **a2c_slot)
{
   nir_cf_node *node = last_demote_node(&impl->body);
   const bool split = worth_splitting(node);

   if (split) {
      b->cursor = nir_after_cf_node(node);
      nir_if *rest = split_if_after_last_demote(b, nir_cf_node_as_if(node));
      b->cursor = nir_before_cf_node(&rest->cf_node);
   } else {
      b->cursor = node ? after_demoting_node(node) : nir_before_impl(impl);
   }

   if (a2c_slot) {
      *a2c_slot = nir_imm_intN_t(b, 0, 16);
      nir_demote_samples(b, *a2c_slot);
   }

   final_coverage_update(b, nir_load_var(b, killed), false);
   return split;
}

/* The most instructions alpha-to-coverage may repeat at the coverage update. */
#define MAX_COVERAGE_CLONE 512

struct coverage_clone {
   nir_builder *b;
   nir_instr *at;
   struct hash_table *remap;
   unsigned instrs;
};

static bool
available_at(nir_def *def, nir_instr *at)
{
   nir_instr *instr = nir_def_instr(def);

   return instr->block == at->block
             ? instr->index < at->index
             : nir_block_dominates(instr->block, at->block);
}

static nir_def *clone_for_coverage(struct coverage_clone *c, nir_def *def);

/* The condition for a block's load to run at the cursor as it would in place:
 * a load from a branch not taken may read through an invalid pointer. */
static bool
guard_for_coverage(struct coverage_clone *c, nir_block *block, nir_def **guard)
{
   for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
      if (node->type == nir_cf_node_loop)
         return false;
      if (node->type != nir_cf_node_if)
         continue;

      nir_if *nif = nir_cf_node_as_if(node);
      nir_def *cond = clone_for_coverage(c, nif->condition.ssa);
      if (!cond)
         return false;
      if (!nir_block_dominates(nir_if_first_then_block(nif), block))
         cond = nir_inot(c->b, cond);
      *guard = *guard ? nir_iand(c->b, *guard, cond) : cond;
   }

   return true;
}

/*
 * Copy what a value needs to the cursor, turning the phis of ifs below it into
 * selects, or return NULL for anything that cannot run ahead of its control
 * flow or passes MAX_COVERAGE_CLONE. What was copied is dead then.
 */
static nir_def *
clone_for_coverage(struct coverage_clone *c, nir_def *def)
{
   if (available_at(def, c->at))
      return def;

   struct hash_entry *he = _mesa_hash_table_search(c->remap, def);
   if (he)
      return he->data;

   if (++c->instrs > MAX_COVERAGE_CLONE)
      return NULL;

   nir_builder *b = c->b;
   nir_instr *instr = nir_def_instr(def);
   nir_def *guard = NULL, *copy;

   switch (instr->type) {
   case nir_instr_type_load_const:
   case nir_instr_type_undef:
      break;
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      /* A channel of a vector needs that channel, not the whole vector */
      if (alu->op == nir_op_mov && def->num_components == 1) {
         nir_scalar s = nir_scalar_resolved(def, 0);
         nir_def *src = clone_for_coverage(c, s.def);
         if (!src)
            return NULL;

         copy = nir_channel(b, src, s.comp);
         _mesa_hash_table_insert(c->remap, def, copy);
         return copy;
      }
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i) {
         if (!clone_for_coverage(c, alu->src[i].src.ssa))
            return NULL;
      }
      break;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (derivative_instr(instr) || !nir_intrinsic_can_reorder(intr))
         return NULL;

      if (!nir_instr_can_speculate(instr) &&
          !guard_for_coverage(c, instr->block, &guard))
         return NULL;

      for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs;
           ++i) {
         if (!clone_for_coverage(c, intr->src[i].ssa))
            return NULL;
      }
      break;
   }
   case nir_instr_type_phi: {
      nir_cf_node *prev = nir_cf_node_prev(&instr->block->cf_node);
      if (!prev || prev->type != nir_cf_node_if)
         return NULL;

      nir_if *nif = nir_cf_node_as_if(prev);
      nir_def *values[2] = {NULL, NULL};
      nir_foreach_phi_src(src, nir_instr_as_phi(instr)) {
         const unsigned arm = src->pred == nir_if_last_then_block(nif) ? 0
                              : src->pred == nir_if_last_else_block(nif) ? 1
                                                                         : 2;
         if (arm == 2 || !(values[arm] = clone_for_coverage(c, src->src.ssa)))
            return NULL;
      }

      nir_def *cond = clone_for_coverage(c, nif->condition.ssa);
      if (!cond || !values[0] || !values[1])
         return NULL;

      copy = nir_bcsel(b, cond, values[0], values[1]);
      _mesa_hash_table_insert(c->remap, def, copy);
      return copy;
   }
   default:
      return NULL;
   }

   nir_if *nif = guard ? nir_push_if(b, guard) : NULL;
   nir_instr *clone = nir_instr_clone_deep(b->shader, instr, c->remap);
   nir_builder_instr_insert(b, clone);
   copy = nir_instr_def(clone);
   if (nif) {
      nir_pop_if(b, nif);
      copy = nir_if_phi(b, copy,
                        nir_undef(b, def->num_components, def->bit_size));
      nir_progress(true, b->impl, nir_metadata_none);
      nir_metadata_require(b->impl,
                           nir_metadata_dominance | nir_metadata_instr_index);
   }
   _mesa_hash_table_insert(c->remap, def, copy);
   return copy;
}

/*
 * Alpha-to-coverage is known long before the end of most shaders that discard,
 * and until its kill is known nothing can test: a fragment hidden by another of
 * the same pass would run to the end. Run the kill at the coverage update, so
 * that the update tests. Returns false to leave it to the epilog, where it was.
 */
static bool
kill_alpha_coverage_at_update(nir_function_impl *impl, nir_def *slot,
                              nir_def *mask)
{
   nir_metadata_require(impl, nir_metadata_dominance | nir_metadata_instr_index);

   nir_builder b = nir_builder_at(nir_after_instr(nir_def_instr(slot)));
   struct coverage_clone clone = {
      .b = &b,
      .at = nir_def_instr(slot),
      .remap = _mesa_pointer_hash_table_create(NULL),
   };
   nir_def *kill = clone_for_coverage(&clone, mask);
   _mesa_hash_table_destroy(clone.remap, NULL);

   if (kill)
      nir_def_rewrite_uses(slot, nir_u2u16(&b, kill));
   nir_progress(true, impl, nir_metadata_none);
   return kill != NULL;
}

/*
 * Native G16G variant. A MOVMSK coverage update takes the killed lanes out of
 * the quad: derivatives and quad operations after a demote then read stale
 * neighbours and helperInvocationEXT() lanes stop contributing
 * (dEQP-VK.glsl.demote.*_deriv, draw.renderpass.shader_invocation.helper_*).
 * Demote only needs the fragment's coverage gone by the time its colour is
 * stored, so accumulate every demote into one killed mask and issue a single
 * coverage update at the end of the shader. With early tests the tests still
 * run at the start; otherwise the final update also runs the tests, which is
 * the fused form run_tests_after_last_discard() builds for an unconditional
 * last discard.
 */
static bool
lower_sample_mask_deferred(nir_shader *shader, bool uniform_tests,
                           bool early_update, bool *a2c_in_main,
                           bool keep_demoted, bool one_update)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);

   /* The caller lowered alpha-to-coverage to a demote at the end of the shader
    * for us to place. Take it out: what the shader discards decides the rest. */
   nir_def *a2c_mask = NULL;
   if (a2c_in_main) {
      nir_intrinsic_instr *a2c =
         last_discard_in_block(nir_impl_last_block(impl));
      /* The caller only moves alpha-to-coverage here when nothing folds the
       * demote away first. If it is gone anyway, leave alpha-to-coverage to
       * the epilog rather than guess. */
      assert(a2c != NULL);
      if (a2c) {
         a2c_mask = a2c->src[0].ssa;
         nir_instr_remove(&a2c->instr);
      }
      *a2c_in_main = false;
   }

   const bool lanes_reach_sensitive_work =
      demoted_lanes_reach_sensitive_work(shader);
   const bool sampled_loop_after_demote = demote_precedes_sampled_loop(impl);

   /* Only shaders whose derivatives, quad operations or sampled loops need
    * every lane defer their kills. Depth-writing (z_export) and forced-early
    * shaders rely on the kill preceding the test-triggering operation or the
    * colour export: pipeline.*.multisample.z_export.*_static_atc and
    * samplemask_early_fragment_tests_depth lost their depth writes with a
    * kill at the end of the shader. */
   if (!keep_demoted &&
       !shader->info.fs.needs_coarse_quad_helper_invocations &&
       !shader->info.fs.needs_full_quad_helper_invocations &&
       !lanes_reach_sensitive_work)
      return lower_sample_mask(shader, uniform_tests);

   bool writes_zs =
      shader->info.outputs_written &
      (BITFIELD64_BIT(FRAG_RESULT_STENCIL) | BITFIELD64_BIT(FRAG_RESULT_DEPTH));

   const bool early = shader->info.fs.early_fragment_tests;
   if (!keep_demoted && !early && !writes_zs &&
       shader->info.fs.uses_discard && !lanes_reach_sensitive_work)
      return lower_sample_mask(shader, uniform_tests);

   if (early) {
      if (shader->info.fs.uses_discard || writes_zs ||
          shader->info.writes_memory)
         run_tests_at_start(shader);
   }

   if (!shader->info.fs.uses_discard)
      return nir_progress(early, impl, nir_metadata_control_flow);

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   struct deferred_ctx ctx = {
      .killed = nir_local_variable_create(impl, glsl_uint16_t_type(),
                                          "killed_samples"),
   };
   nir_store_var(&b, ctx.killed, nir_imm_intN_t(&b, 0, 16),
                 nir_component_mask(1));

   /* The update ends a SIMD group once its lanes are killed or have failed
    * the tests the update runs, and until it runs a discarding shader has no
    * early tests at all, however rarely it discards. On G17P the lanes it
    * kills keep feeding their quad's derivatives while the quad lives. It
    * would drop the side effects of fragments that fail the tests, though,
    * and change the lanes that helper state and subgroup operations see.
    * Depth-writing and forced-early shaders stay on the update at the end. */
   early_update &= !early && !writes_zs && !shader->info.writes_memory &&
                   !shader->info.fs.uses_sample_shading &&
                   !cf_list_has(&impl->body, lane_state_instr) &&
                   !(agx_get_compiler_debug() & AGX_DBG_LATECOV);
   nir_def *a2c_slot = NULL;
   const bool split =
      early_update &&
      update_coverage_after_last_demote(&b, impl, ctx.killed,
                                        a2c_mask ? &a2c_slot : NULL);

   nir_shader_intrinsics_pass(shader, accumulate_demote,
                              nir_metadata_none, &ctx);

   /* Unconditional top-level demotes are handled where they occur: their
    * block continues in uniform control flow. No dead-quad kill may precede
    * a sampled loop in this late form: every update stays after it. */
   if (!early_update && !one_update &&
       !cf_list_has(&impl->body, subgroup_instr) &&
       !sampled_loop_after_demote) {
      nir_foreach_block(block, impl) {
         if (block->cf_node.parent != &impl->cf_node)
            continue;
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic !=
                   nir_intrinsic_store_deref)
               continue;
            nir_intrinsic_instr *st = nir_instr_as_intrinsic(instr);
            if (nir_intrinsic_get_var(st, 0) != ctx.killed ||
                nir_src_is_const(st->src[1]))
               continue;
            nir_builder qb = nir_builder_at(nir_after_instr(instr));
            nir_def *k = st->src[1].ssa;
            nir_def *dead = quad_fully_killed(&qb, k);
            nir_sample_mask_agx(&qb,
                                nir_bcsel(&qb, dead, k,
                                          nir_imm_intN_t(&qb, 0, 16)),
                                nir_imm_intN_t(&qb, 0, 16));
         }
      }
      kill_dead_quads(impl, ctx.killed);
   }

   if (!early_update) {
      b.cursor = nir_after_impl(impl);
      nir_def *killed = nir_load_var(&b, ctx.killed);
      if (early || writes_zs) {
         /* Tests already ran (or zs_emit runs them): kill without testing. */
         nir_sample_mask_agx(&b, killed, nir_imm_intN_t(&b, 0, 16));
      } else {
         final_coverage_update(&b, killed, uniform_tests);
      }
   }

   nir_progress(true, impl, nir_metadata_none);
   if (split) {
      nir_repair_ssa_impl(impl);
      nir_lower_reg_intrinsics_to_ssa_impl(impl);
   }
   /* Not NIR_PASS: a2c_slot and a2c_mask point into this impl, and the
    * NIR_DEBUG clone/serialize checks would replace it under them. */
   nir_lower_vars_to_ssa(shader);
   if (a2c_slot)
      *a2c_in_main = kill_alpha_coverage_at_update(impl, a2c_slot, a2c_mask);
   return true;
}

/*
 * A fragment shader unrolled per sample (G17P): every copy demotes only its
 * own sample. A kill-only coverage update that removes some of a multisampled
 * pixel's samples leaves those samples without their previous contents: they
 * resolve as zero instead of keeping what the attachment held. Samples
 * removed by the testing update keep it. So collect every copy's demotes and
 * apply them in the single testing update at the end, the form a per-pixel
 * shader with one unconditional discard gets. No kill-only update is issued,
 * dead quads included.
 *
 * Only for shaders the single final update suits: no forced-early tests, no
 * depth/stencil export, no memory writes (the callers check).
 */
bool
agx_nir_lower_sample_mask_one_update(nir_shader *shader)
{
   assert(!shader->info.fs.early_fragment_tests);
   return lower_sample_mask_deferred(shader, false, false, NULL, true, true);
}

/* M5: the same, with M5's uniformly selected final update. */
bool
agx_nir_lower_sample_mask_one_update_m5(nir_shader *shader)
{
   assert(!shader->info.fs.early_fragment_tests);
   return lower_sample_mask_deferred(shader, true, false, NULL, true, true);
}

bool
agx_nir_lower_sample_mask_deferred(nir_shader *shader)
{
   return lower_sample_mask_deferred(shader, false, false, NULL, false, false);
}

/* G17P: the final coverage update right after the last demote on each path.
 * With a2c_in_main the shader's last demote is alpha-to-coverage's, and
 * *a2c_in_main returns whether it stayed in the shader. */
bool
agx_nir_lower_sample_mask_deferred_g17p(nir_shader *shader, bool *a2c_in_main)
{
   return lower_sample_mask_deferred(shader, false, true, a2c_in_main, false,
                                     false);
}

/*
 * G13/G14 and other parts without the deferred lowering kill demoted samples
 * where the demote happens. The kill ends a quad once none of its non-helper
 * lanes is left, and its rasterizer helper lanes go with it although they
 * never demoted. Non-quad subgroup operations of the lanes that keep running
 * would then miss those helpers (a ballot taken after another quad's lanes
 * were terminated counts fewer lanes than a ballot of the same lanes taken
 * before). Such shaders apply every kill at the end instead, after the last
 * subgroup operation.
 */
bool
agx_nir_lower_sample_mask(nir_shader *shader)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   if (shader->info.fs.uses_discard &&
       cf_list_has(&impl->body, subgroup_instr))
      return lower_sample_mask_deferred(shader, false, false, NULL, true, false);

   return lower_sample_mask(shader, false);
}

/* M5: one uniformly selected final coverage update (final_coverage_update). */
bool
agx_nir_lower_sample_mask_deferred_m5(nir_shader *shader)
{
   return lower_sample_mask_deferred(shader, true, false, NULL, false, false);
}

bool
agx_nir_coalesce_sample_mask_updates(nir_shader *shader, unsigned nr_samples)
{
   if ((nr_samples != 2 && nr_samples != 4) || shader->info.writes_memory ||
       shader->info.fs.early_fragment_tests ||
       shader->info.fs.needs_coarse_quad_helper_invocations ||
       shader->info.fs.needs_full_quad_helper_invocations)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_intrinsic_instr *updates[4];
   unsigned update_bits[4];
   unsigned count = 0, covered = 0;
   const unsigned all = BITFIELD_MASK(nr_samples);
   nir_block *update_block = NULL;
   nir_def *target = NULL, *live = NULL;
   unsigned flags = 0;

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_sample_mask_agx)
            continue;

         /* Keep kill-only operations in place. Below, moving a testing
          * update past one requires disjoint affected sample sets. */
         if (nir_src_is_const(intr->src[1]) &&
             nir_src_as_uint(intr->src[1]) == 0)
            continue;

         if (count == ARRAY_SIZE(updates) ||
             (update_block && update_block != block))
            return false;

         nir_alu_instr *mask = nir_src_as_alu(intr->src[0]);
         if (!mask || mask->op != nir_op_iand ||
             intr->src[0].ssa->bit_size != 16 ||
             intr->src[1].ssa->bit_size != 16)
            return false;

         unsigned constant = nir_src_is_const(mask->src[0].src) ? 0 : 1;
         if (!nir_src_is_const(mask->src[constant].src))
            return false;
         unsigned bits = nir_src_as_uint(mask->src[constant].src);
         nir_def *base = mask->src[constant ^ 1].src.ssa;
         if (base->num_components != 1 ||
             mask->src[constant].src.ssa->num_components != 1 ||
             mask->src[constant ^ 1].swizzle[0] != 0 ||
             mask->src[constant].swizzle[0] != 0)
            return false;
         if (!bits || (bits & ~all) || (covered & bits) ||
             (target && target != base) ||
             (live && live != intr->src[1].ssa) ||
             (count && flags != nir_intrinsic_flags(intr)))
            return false;

         target = base;
         live = intr->src[1].ssa;
         flags = nir_intrinsic_flags(intr);
         update_block = block;
         covered |= bits;
         update_bits[count] = bits;
         updates[count++] = intr;
      }
   }

   if (count < 2 || covered != all)
      return false;

   bool between = false;
   unsigned next_update = 0, moved_samples = 0;
   nir_foreach_instr(instr, update_block) {
      if (instr == &updates[0]->instr)
         between = true;
      if (!between)
         continue;
      if (instr == &updates[count - 1]->instr)
         break;
      if (instr == &updates[next_update]->instr) {
         moved_samples |= update_bits[next_update++];
         continue;
      }
      if (instr->type == nir_instr_type_alu ||
          instr->type == nir_instr_type_load_const)
         continue;
      if (instr->type != nir_instr_type_intrinsic)
         return false;
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_sample_mask_agx) {
         if (!nir_src_is_const(intr->src[1]) ||
             nir_src_as_uint(intr->src[1]) != 0)
            return false;
         nir_alu_instr *mask = nir_src_as_alu(intr->src[0]);
         if (!mask || mask->op != nir_op_iand ||
             intr->src[0].ssa->bit_size != 16 ||
             intr->src[0].ssa->num_components != 1)
            return false;
         unsigned constant = nir_src_is_const(mask->src[0].src) ? 0 : 1;
         if (!nir_src_is_const(mask->src[constant].src) ||
             mask->src[constant].src.ssa->num_components != 1 ||
             mask->src[constant ^ 1].src.ssa->num_components != 1 ||
             mask->src[constant].swizzle[0] != 0 ||
             mask->src[constant ^ 1].swizzle[0] != 0)
            return false;
         unsigned killed = nir_src_as_uint(mask->src[constant].src);
         if (killed & moved_samples)
            return false;
      } else if (intr->intrinsic != nir_intrinsic_export_agx &&
          !nir_intrinsic_can_reorder(intr))
         return false;
   }

   nir_builder b = nir_builder_at(nir_before_instr(&updates[count - 1]->instr));
   nir_intrinsic_instr *combined = nir_sample_mask_agx(&b, target, live);
   if (flags == AGX_SAMPLE_MASK_SUBSET_FULL_OR_KILL)
      flags = AGX_SAMPLE_MASK_FULL_OR_KILL;
   nir_intrinsic_set_flags(combined, flags);
   for (unsigned i = 0; i < count; ++i)
      nir_instr_remove(&updates[i]->instr);

   nir_foreach_function_impl(other, shader) {
      nir_progress(other == impl, other, nir_metadata_control_flow);
   }
   return true;
}

/*
 * Give a G17P tile access the sample-mask source its hardware form actually
 * takes.
 *
 * At one sample the operand is the whole-pixel immediate, which is what Mesa
 * already spells as ALL_SAMPLES, so nothing has to happen.
 *
 * At two or four samples there is no immediate form at all. Apple's compiler
 * reads the live coverage word out of a special register and passes that
 * register as the mask -- raw for a per-pixel shader ("every sample this
 * fragment covers"), ANDed with a one-hot bit for each iteration of an
 * unrolled per-sample shader. Reproduce exactly those two shapes.
 *
 * A non-constant mask is already an expression in a register, so it is left
 * alone and reaches the packer as the register form by itself. An epilog
 * unrolled per sample (agx_nir_unroll_g17p_samples) has folded each sample's
 * `mask & active_samples` to that sample's one-hot constant beforehand, so it
 * too becomes coverage ANDed with the sample's bit.
 */
struct g17p_tile_sample_state {
   unsigned samples;
   nir_def *coverage;
   nir_block *block;
   /* msaa163: G18G reads the coverage as the 16-bit active-sample word (W16
    * SR39); the agx3c encoder G18G compiles through has no load_sample_mask
    * form. */
   bool g18;
};

static bool
lower_g17p_tile_sample(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct g17p_tile_sample_state *state = data;
   unsigned src;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_local_pixel_agx:
      src = 0;
      break;
   case nir_intrinsic_store_local_pixel_agx:
      src = 1;
      break;
   default:
      /* Coverage changes at depth/stencil writes, demotes and interlocks.
       * Conservatively end the local reuse window at any side effect. */
      if (!(nir_intrinsic_infos[intr->intrinsic].flags &
            NIR_INTRINSIC_CAN_ELIMINATE))
         state->coverage = NULL;
      return false;
   }

   const unsigned samples = state->samples;
   if (samples == 1) {
      /* Per-sample epilogs still carry a dynamic active-sample mask, but
       * there is only sample zero. The native 1x tile instruction takes the
       * whole-pixel immediate; preserve its form for these epilogs. */
      if (nir_src_is_const(intr->src[src]) &&
          nir_src_as_uint(intr->src[src]) == ALL_SAMPLES)
         return false;

      b->cursor = nir_before_instr(&intr->instr);
      nir_src_rewrite(&intr->src[src],
                      nir_imm_intN_t(b, ALL_SAMPLES, 16));
      return true;
   }

   /* Already a live expression: the per-sample lowerings have ANDed the mask
    * with an active-sample register, which is the register form. */
   if (!nir_src_is_const(intr->src[src]))
      return false;

   const unsigned mask = nir_src_as_uint(intr->src[src]);
   if (mask == 0) {
      /* Static unrolling can prove that the API mask excludes this sample.
       * A zero-coverage tile access is dead, and keeping it would turn one
       * surviving sample into an unsupported mixed zero/nonzero protocol. */
      b->cursor = nir_before_instr(&intr->instr);
      if (intr->intrinsic == nir_intrinsic_load_local_pixel_agx) {
         nir_def_rewrite_uses(
            &intr->def,
            nir_undef(b, intr->def.num_components, intr->def.bit_size));
      }
      nir_instr_remove(&intr->instr);
      return true;
   }

   if (mask != ALL_SAMPLES &&
       mask >= (1u << samples))
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   /* active_samples_agx is a loop system value: NIR may CSE it across
    * depth/stencil writes. At this point we need the hardware's *current*
    * coverage, including the tests for this particular unrolled sample.
    * load_sample_mask is ordered and lowers to the same G17P SR39 read. */
   if (!state->coverage || state->block != intr->instr.block) {
      state->coverage = state->g18 ? nir_load_active_samples_agx(b)
                                   : nir_u2u16(b, nir_load_sample_mask(b));
      state->block = intr->instr.block;
   }
   nir_def *coverage = state->coverage;
   nir_def *sel = (mask == ALL_SAMPLES)
                     ? coverage
                     : nir_iand(b, coverage, nir_imm_intN_t(b, mask, 16));

   nir_src_rewrite(&intr->src[src], sel);
   return true;
}

bool
agx_nir_lower_g17p_tile_samples(nir_shader *s, unsigned nr_samples)
{
   if (nr_samples != 1 && nr_samples != 2 && nr_samples != 4 && nr_samples != 8)
      return false;

   struct g17p_tile_sample_state state = {.samples = nr_samples};
   return nir_shader_intrinsics_pass(s, lower_g17p_tile_sample,
                                     nir_metadata_control_flow, &state);
}

bool
agx_nir_lower_g18_tile_samples(nir_shader *s, unsigned nr_samples)
{
   /* Native326 adds two samples to the existing four-sample SR39 protocol. */
   if (nr_samples != 2 && nr_samples != 4)
      return false;

   struct g17p_tile_sample_state state = {.samples = nr_samples, .g18 = true};
   return nir_shader_intrinsics_pass(s, lower_g17p_tile_sample,
                                     nir_metadata_control_flow, &state);
}

bool
agx_nir_lower_g17p_1x_tile_samples(nir_shader *s)
{
   /* Honeykrisp advertises only one sample on G17P.  Per-sample epilog
    * lowering still materializes `1 << sample_id`, but sample_id can only be
    * zero.  Native 1x fragments encode the ordinary whole-pixel LSTCOL/LLDCOL
    * forms, which carry no sample operand. */
   return agx_nir_lower_g17p_tile_samples(s, 1);
}

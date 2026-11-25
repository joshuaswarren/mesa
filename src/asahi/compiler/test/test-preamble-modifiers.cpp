/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}

static void
check_modifier(bool g17p, bool widen)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "uniform modifiers");
   nir_def *value = agx_test_load_preamble(&b, 1, widen ? 16 : 32, 4);
   value = widen ? nir_f2f32(&b, value) : nir_fabs(&b, value);
   nir_def *x = nir_channel(&b, nir_load_frag_coord(&b), 0);
   nir_def *result = nir_fmul(&b, x, value);
   agx_test_store_output(&b, nir_vec4(&b, result, result, result, result),
                         nir_imm_int(&b, 0), nir_type_float32,
                         FRAG_RESULT_DATA0);
   unsigned sizes[3] = {16, 0, 0};
   bool progress = agx_nir_opt_preamble(b.shader, sizes, g17p, false, NULL);
   const bool hoisted = g17p && !widen;
   if (hoisted) {
      EXPECT_TRUE(progress);
   }
   unsigned modifiers = 0, exported = 0;
   nir_function_impl *main = nir_shader_get_entrypoint(b.shader);
   nir_foreach_block(block, main) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_alu) {
            nir_op op = nir_instr_as_alu(instr)->op;
            modifiers += op == (widen ? nir_op_f2f32 : nir_op_fabs);
         } else if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            exported += intr->intrinsic == nir_intrinsic_load_preamble &&
                        nir_intrinsic_base(intr) >= 16;
         }
      }
   }
   EXPECT_EQ(modifiers, hoisted ? 0u : 1u);
   EXPECT_EQ(exported, hoisted ? 1u : 0u);
   if (hoisted) {
      EXPECT_EQ(sizes[0], 18u);
   }
   ralloc_free(b.shader);
}

TEST(PreambleModifiers, HoistsUniformAbsoluteValueOnAGX3B)
{
   check_modifier(true, false);
   check_modifier(false, false);
}

TEST(PreambleModifiers, PreservesFreeUniformHalfWidening)
{
   check_modifier(true, true);
   check_modifier(false, true);
}

TEST(PreambleModifiers, CancelsDoubleFnegAcrossPreamble)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "cancel preamble fneg");

   nir_function *pre_fn = nir_function_create(b.shader, "preamble");
   pre_fn->is_preamble = true;
   nir_function_impl *pre_impl = nir_function_impl_create(pre_fn);
   b.impl->preamble = pre_fn;

   nir_builder pb = nir_builder_at(nir_before_impl(pre_impl));
   nir_def *raw = agx_test_load_preamble(&pb, 1, 32, 4);
   nir_def *neg = nir_fneg(&pb, raw);
   agx_test_store_preamble(&pb, neg, 12);

   b.cursor = nir_before_impl(b.impl);
   nir_def *u = agx_test_load_preamble(&b, 1, 32, 12);
   nir_def *uneg = nir_fneg(&b, u);
   nir_def *uabs = nir_fabs(&b, u);
   nir_def *sum = nir_fadd(&b, uneg, uabs);
   nir_intrinsic_instr *gst =
      nir_store_global(&b, sum, nir_imm_int64(&b, 0x1000));
   nir_intrinsic_set_align(gst, 4, 0);
   nir_intrinsic_set_write_mask(gst, 1);

   EXPECT_TRUE(agx_nir_cancel_preamble_fneg(b.shader, 8));
   nir_validate_shader(b.shader, "after agx_nir_cancel_preamble_fneg");

   unsigned main_fnegs = 0, main_fabs = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_alu) {
            nir_op op = nir_instr_as_alu(instr)->op;
            main_fnegs += op == nir_op_fneg;
            main_fabs += op == nir_op_fabs;
         }
      }
   }
   EXPECT_EQ(main_fnegs, 0u);
   EXPECT_EQ(main_fabs, 1u);

   /* Preamble store now writes raw directly instead of fneg(raw). */
   nir_intrinsic_instr *st = NULL;
   nir_foreach_instr(instr, nir_start_block(pre_impl)) {
      if (instr->type == nir_instr_type_intrinsic &&
          nir_instr_as_intrinsic(instr)->intrinsic ==
             nir_intrinsic_store_preamble)
         st = nir_instr_as_intrinsic(instr);
   }
   ASSERT_NE(st, nullptr);
   EXPECT_EQ(st->src[0].ssa, raw);

   ralloc_free(b.shader);
}

TEST(PreambleModifiers, CancelPreambleFnegRejectsDirectUse)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "keep preamble fneg");

   nir_function *pre_fn = nir_function_create(b.shader, "preamble");
   pre_fn->is_preamble = true;
   nir_function_impl *pre_impl = nir_function_impl_create(pre_fn);
   b.impl->preamble = pre_fn;

   nir_builder pb = nir_builder_at(nir_before_impl(pre_impl));
   nir_def *raw = agx_test_load_preamble(&pb, 1, 32, 4);
   agx_test_store_preamble(&pb, nir_fneg(&pb, raw), 12);

   b.cursor = nir_before_impl(b.impl);
   nir_def *u = agx_test_load_preamble(&b, 1, 32, 12);
   nir_def *sum = nir_fadd(&b, nir_fneg(&b, u), u);
   nir_intrinsic_instr *gst =
      nir_store_global(&b, sum, nir_imm_int64(&b, 0x1000));
   nir_intrinsic_set_align(gst, 4, 0);
   nir_intrinsic_set_write_mask(gst, 1);

   EXPECT_FALSE(agx_nir_cancel_preamble_fneg(b.shader, 8));
   ralloc_free(b.shader);
}

TEST(PreambleModifiers, ReassociatesLoopInvariantEquality)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "reassociate loop eq");

   nir_def *inv = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *init = nir_channel(&b, nir_load_local_invocation_id(&b), 0);

   nir_block *pre = nir_cursor_current_block(b.cursor);
   nir_loop *loop = nir_push_loop(&b);
   nir_block *hdr = nir_loop_first_block(loop);

   nir_phi_instr *j_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&j_phi->instr, &j_phi->def, 1, 32);
   nir_phi_instr_add_src(j_phi, pre, init);

   nir_def *j = &j_phi->def;
   /* Top-tested loop: body sits after `if (j >= 4096) break`, so the body
    * block does not dominate the loop successor. Reassociation must hoist
    * `inv - 32` directly into the preheader without relying on nir_opt_licm. */
   nir_push_if(&b, nir_uge_imm(&b, j, 4096));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_def *j32 = nir_iadd_imm(&b, j, 32);
   nir_def *ne = nir_ine(&b, j32, inv);
   nir_intrinsic_instr *gst =
      nir_store_global(&b, nir_b2i32(&b, ne), nir_imm_int64(&b, 0x1000));
   nir_intrinsic_set_align(gst, 4, 0);
   nir_intrinsic_set_write_mask(gst, 1);

   nir_def *j_next = nir_iadd_imm(&b, j, 128);
   nir_phi_instr_add_src(j_phi, nir_loop_last_block(loop), j_next);
   nir_instr_insert(nir_before_block(nir_loop_first_block(loop)),
                    &j_phi->instr);
   nir_pop_loop(&b, loop);

   EXPECT_TRUE(agx_nir_reassociate_loop_invariant_eq(b.shader));
   nir_validate_shader(b.shader, "after agx_nir_reassociate_loop_invariant_eq");

   /* Only the loop increment iadd(+128) remains inside the loop; inv - 32 is
    * hoisted directly into the preheader. */
   unsigned pre_iadds = 0, loop_iadds = 0;
   nir_foreach_instr(instr, pre) {
      if (instr->type == nir_instr_type_alu &&
          nir_instr_as_alu(instr)->op == nir_op_iadd)
         pre_iadds++;
   }
   nir_foreach_block_in_cf_node(blk, &loop->cf_node) {
      nir_foreach_instr(instr, blk) {
         if (instr->type == nir_instr_type_alu &&
             nir_instr_as_alu(instr)->op == nir_op_iadd)
            loop_iadds++;
      }
   }
   EXPECT_EQ(pre_iadds, 1u);
   EXPECT_EQ(loop_iadds, 1u);

   ralloc_free(b.shader);
}

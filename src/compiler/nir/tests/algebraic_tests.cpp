/*
 * Copyright © 2018 Intel Corporation
 * Copyright © 2021 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "nir_test.h"
#include "nir_constant_expressions.h"
extern "C" {
#include "nir_search.h"
}
#include <vector>
#include <utility>

namespace {

class algebraic_test_base : public nir_test {
protected:
   algebraic_test_base();

   virtual void run_pass()=0;

   void test_op(nir_op op, nir_def *src0, nir_def *src1, nir_def *src2,
                nir_def *src3, const char *desc);

   void test_2src_op(nir_op op, int64_t src0, int64_t src1);

   void require_one_alu(nir_op op);

   nir_variable *res_var;
};

algebraic_test_base::algebraic_test_base()
   : nir_test::nir_test("nir_opt_algebraic_test")
{
   res_var = nir_local_variable_create(b->impl, glsl_int_type(), "res");
}

void algebraic_test_base::test_op(nir_op op, nir_def *src0, nir_def *src1,
                                     nir_def *src2, nir_def *src3, const char *desc)
{
   nir_def *res_deref = &nir_build_deref_var(b, res_var)->def;

   /* create optimized expression */
   nir_intrinsic_instr *optimized_instr = nir_build_store_deref(
      b, res_deref, nir_build_alu(b, op, src0, src1, src2, src3), 0x1);

   run_pass();
   b->cursor = nir_after_cf_list(&b->impl->body);

   /* create reference expression */
   nir_intrinsic_instr *ref_instr = nir_build_store_deref(
      b, res_deref, nir_build_alu(b, op, src0, src1, src2, src3), 0x1);

   /* test equality */
   nir_opt_constant_folding(b->shader);

   ASSERT_TRUE(nir_src_is_const(ref_instr->src[1]));
   ASSERT_TRUE(nir_src_is_const(optimized_instr->src[1]));

   int32_t ref = nir_src_as_int(ref_instr->src[1]);
   int32_t optimized = nir_src_as_int(optimized_instr->src[1]);

   EXPECT_EQ(ref, optimized) << "Test input: " << desc;

   /* reset shader */
   exec_list_make_empty(&nir_start_block(b->impl)->instr_list);
   b->cursor = nir_after_cf_list(&b->impl->body);
}

void algebraic_test_base::test_2src_op(nir_op op, int64_t src0, int64_t src1)
{
   char desc[128];
   snprintf(desc, sizeof(desc), "%s(%" PRId64 ", %" PRId64 ")", nir_op_infos[op].name, src0, src1);
   test_op(op, nir_imm_int(b, src0), nir_imm_int(b, src1), NULL, NULL, desc);
}

void algebraic_test_base::require_one_alu(nir_op op)
{
   unsigned count = 0;
   nir_foreach_instr(instr, nir_start_block(b->impl)) {
      if (instr->type == nir_instr_type_alu) {
         ASSERT_TRUE(nir_instr_as_alu(instr)->op == op);
         ASSERT_EQ(count, 0);
         count++;
      }
   }
}

class nir_opt_algebraic_test : public algebraic_test_base {
protected:
   virtual void run_pass() {
      nir_opt_algebraic(b->shader);
   }
};

class nir_opt_idiv_const_test : public algebraic_test_base {
protected:
   virtual void run_pass() {
      nir_opt_idiv_const(b->shader, 8);
   }
};

class nir_opt_mqsad_test : public algebraic_test_base {
protected:
   virtual void run_pass() {
      nir_opt_mqsad(b->shader);
   }
};

static std::vector<unsigned> unary_match_condition_calls;

static bool
record_unary_match_condition(const nir_alu_instr *instr)
{
   unary_match_condition_calls.push_back(instr->def.index);
   return true;
}

TEST_F(nir_opt_algebraic_test, matched_variable_preserves_def_bits_and_swizzles)
{
   for (unsigned bits : {8u, 16u, 32u, 64u}) {
      for (unsigned components : {1u, 2u, 4u, 8u, 16u}) {
         for (unsigned depth : {2u, 4u, 8u}) {
            SCOPED_TRACE(bits);
            SCOPED_TRACE(components);
            SCOPED_TRACE(depth);
            b->cursor = nir_after_cf_list(&b->impl->body);
            nir_load_const_instr *input =
               nir_load_const_instr_create(b->shader, components, bits);
            for (unsigned c = 0; c < components; ++c)
               input->value[c] = nir_const_value_for_uint(17 + c, bits);
            nir_builder_instr_insert(b, &input->instr);
            nir_def *value = &input->def;
            std::vector<unsigned> indices;
            std::vector<std::vector<unsigned>> swizzles;
            for (unsigned level = 0; level < depth; ++level) {
               value = nir_inot(b, value);
               nir_alu_instr *alu = nir_def_as_alu(value);
               indices.push_back(value->index);
               swizzles.emplace_back();
               for (unsigned c = 0; c < components; ++c) {
                  unsigned swizzle =
                     (components - 1 - c + level) % components;
                  alu->src[0].swizzle[c] = swizzle;
                  swizzles.back().push_back(swizzle);
               }
            }
            nir_intrinsic_instr *store =
               nir_intrinsic_instr_create(b->shader, nir_intrinsic_store_ssbo);
            store->num_components = components;
            store->src[0] = nir_src_for_ssa(value);
            store->src[1] = nir_src_for_ssa(nir_imm_int(b, 0));
            store->src[2] = nir_src_for_ssa(nir_imm_int(b, 0));
            nir_intrinsic_set_write_mask(store, BITFIELD_MASK(components));
            nir_intrinsic_set_align_mul(store, 1);
            nir_builder_instr_insert(b, &store->instr);

            const unsigned variable = NIR_SEARCH_MAX_VARIABLES - 1;
            std::vector<nir_search_value_union> values(depth + 3);
            values[0].variable = {};
            values[0].variable.value.type = nir_search_value_variable;
            values[0].variable.variable = variable;
            values[0].variable.cond_index = -1;
            for (unsigned c = 0; c < NIR_MAX_VEC_COMPONENTS; ++c)
               values[0].variable.swizzle[c] = c;
            for (unsigned level = 1; level <= depth; ++level) {
               nir_search_expression &expr = values[level].expression;
               expr = {};
               expr.value.type = nir_search_value_expression;
               expr.opcode = nir_op_inot;
               expr.swizzle = -1;
               expr.comm_expr_mask = 0;
               expr.cond_index = 0;
               expr.srcs[0] = level - 1;
            }
            nir_search_constant &zero = values[depth + 1].constant;
            zero = {};
            zero.data.i = 0;
            zero.value.type = nir_search_value_constant;
            zero.value.bit_size = -int(variable) - 1;
            zero.type = nir_type_int;
            nir_search_expression &replacement = values[depth + 2].expression;
            replacement = {};
            replacement.value.type = nir_search_value_expression;
            replacement.value.bit_size = -int(variable) - 1;
            replacement.opcode = nir_op_iadd;
            replacement.swizzle = -1;
            replacement.comm_expr_mask = 0;
            replacement.cond_index = -1;
            replacement.srcs[0] = 0;
            replacement.srcs[1] = depth + 1;
            const uint16_t transition[] = {2};
            std::vector<per_op_table> operations(nir_num_search_ops);
            for (per_op_table &op : operations) {
               op.num_filtered_states = 1;
               op.table = transition;
            }
            const struct transform transforms[] = {
               {static_cast<uint16_t>(depth), static_cast<uint16_t>(depth + 2), 0}, {0, 0, ~0u},
            };
            const uint16_t offsets[] = {1, 1, 0};
            const nir_search_expression_cond conditions[] = {
               record_unary_match_condition,
            };
            nir_algebraic_table table = {};
            table.transforms = transforms;
            table.transform_offsets = offsets;
            table.pass_op_table = operations.data();
            table.values = values.data();
            table.expression_cond = conditions;
            const bool enabled[] = {true};
            unary_match_condition_calls.clear();
            ASSERT_TRUE(nir_algebraic_impl(b->impl, enabled, &table));
            ASSERT_GE(unary_match_condition_calls.size(), depth);
            for (unsigned level = 0; level < depth; ++level)
               EXPECT_EQ(unary_match_condition_calls[level],
                         indices[depth - 1 - level]);
            ASSERT_EQ(nir_src_bit_size(store->src[0]), bits);
            ASSERT_EQ(nir_src_num_components(store->src[0]), components);
            nir_validate_shader(b->shader, "matched variable representation");
            nir_opt_constant_folding(b->shader);
            ASSERT_TRUE(nir_src_is_const(store->src[0]));
            for (unsigned c = 0; c < components; ++c) {
               unsigned input_component = c;
               for (unsigned level = depth; level-- > 0;)
                  input_component = swizzles[level][input_component];
               EXPECT_EQ(nir_src_comp_as_uint(store->src[0], c),
                         17u + input_component);
            }
         }
      }
   }
}

static std::vector<const nir_alu_instr *> initial_worklist_visits;

static bool
record_initial_worklist_visit(const nir_alu_instr *instr)
{
   initial_worklist_visits.push_back(instr);
   return false;
}

TEST_F(nir_opt_algebraic_test, initial_worklist_order_and_flags)
{
   for (unsigned count : {0u, 1u, 7u, 8u, 9u, 15u, 16u, 17u,
                          31u, 32u, 33u, 127u, 128u, 129u}) {
      for (unsigned layout = 0; layout < 3; ++layout) {
         SCOPED_TRACE(count);
         SCOPED_TRACE(layout);
         nir_shader_compiler_options options = {};
         nir_builder current = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &options, "initial worklist order");
         ralloc_steal(b->shader, current.shader);
         nir_builder *build = &current;
         auto append_alu = [&](unsigned amount) {
            for (unsigned i = 0; i < amount; ++i)
               nir_inot(build, nir_imm_int(build, i + 3));
         };
         nir_load_local_invocation_index(build);
         if (layout == 0) {
            append_alu(count);
         } else if (layout == 1) {
            append_alu(count / 3);
            nir_if *branch = nir_push_if(build, nir_imm_true(build));
            append_alu(count / 3);
            nir_push_else(build, branch);
            append_alu(count - 2 * (count / 3));
            nir_pop_if(build, branch);
         } else {
            nir_loop *loop = nir_push_loop(build);
            nir_loop_add_continue_construct(loop);
            append_alu(count / 4);
            nir_if *branch = nir_push_if(build, nir_imm_true(build));
            append_alu(count / 4);
            nir_jump(build, nir_jump_break);
            nir_push_else(build, branch);
            append_alu(count / 4);
            nir_jump(build, nir_jump_continue);
            nir_pop_if(build, branch);
            nir_push_continue(build, loop);
            append_alu(count - 3 * (count / 4));
            nir_pop_loop(build, loop);
         }
         nir_validate_shader(current.shader, "before initial worklist");
         std::vector<const nir_alu_instr *> expected;
         nir_foreach_block_reverse(block, build->impl) {
            nir_foreach_instr_reverse(instr, block) {
               if (instr->type == nir_instr_type_alu) {
                  ASSERT_EQ(nir_instr_as_alu(instr)->op, nir_op_inot);
                  expected.push_back(nir_instr_as_alu(instr));
               }
            }
         }
         ASSERT_EQ(expected.size(), count);
         std::vector<nir_instr *> all;
         nir_foreach_block(block, build->impl) {
            nir_foreach_instr(instr, block) {
               instr->pass_flags = 0x5a;
               all.push_back(instr);
            }
         }
         const uint16_t transition[] = {2};
         std::vector<per_op_table> operations(nir_num_search_ops);
         for (per_op_table &op : operations) {
            op.num_filtered_states = 1;
            op.table = transition;
         }
         nir_search_value_union values[2] = {};
         values[0].variable = {};
         values[0].variable.value.type = nir_search_value_variable;
         values[0].variable.cond_index = -1;
         values[1].expression = {};
         values[1].expression.value.type = nir_search_value_expression;
         values[1].expression.opcode = nir_op_inot;
         values[1].expression.swizzle = -1;
         values[1].expression.comm_expr_mask = 0;
         values[1].expression.cond_index = 0;
         const struct transform transforms[] = {{1, 0, 0}, {0, 0, ~0u}};
         const uint16_t offsets[] = {1, 1, 0};
         const nir_search_expression_cond conditions[] = {
            record_initial_worklist_visit,
         };
         nir_algebraic_table table = {};
         table.transforms = transforms;
         table.transform_offsets = offsets;
         table.pass_op_table = operations.data();
         table.values = values;
         table.expression_cond = conditions;
         const bool enabled[] = {true};
         initial_worklist_visits.clear();
         EXPECT_FALSE(nir_algebraic_impl(build->impl, enabled, &table));
         EXPECT_EQ(initial_worklist_visits, expected);
         for (nir_instr *instr : all)
            EXPECT_EQ(instr->pass_flags, 0u);
         nir_validate_shader(current.shader, "after initial worklist");
         ralloc_free(current.shader);
      }
   }
}

static void
filter_variable(nir_search_value_union &value, unsigned bits = 0)
{
   value.variable = {};
   value.variable.value.type = nir_search_value_variable;
   value.variable.value.bit_size = bits;
   value.variable.cond_index = -1;
   for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; ++i)
      value.variable.swizzle[i] = i;
}

static void
filter_expression(nir_search_value_union &value, uint16_t op, unsigned src,
                  unsigned bits = 0, int condition = -1)
{
   value.expression = {};
   value.expression.value.type = nir_search_value_expression;
   value.expression.value.bit_size = bits;
   value.expression.opcode = op;
   value.expression.srcs[0] = src;
   value.expression.swizzle = -1;
   value.expression.comm_expr_mask = 0;
   value.expression.cond_index = condition;
}

static nir_intrinsic_instr *
filter_store(nir_builder *build, nir_def *value)
{
   nir_intrinsic_instr *store =
      nir_intrinsic_instr_create(build->shader, nir_intrinsic_store_ssbo);
   store->num_components = value->num_components;
   store->src[0] = nir_src_for_ssa(value);
   store->src[1] = nir_src_for_ssa(nir_imm_int(build, 0));
   store->src[2] = nir_src_for_ssa(nir_imm_int(build, 0));
   nir_intrinsic_set_write_mask(store, BITFIELD_MASK(value->num_components));
   nir_intrinsic_set_align_mul(store, 1);
   nir_builder_instr_insert(build, &store->instr);
   return store;
}

TEST_F(nir_opt_algebraic_test, absent_opcodes_keep_state_zero_rules)
{
   for (bool supported : {false, true}) {
      SCOPED_TRACE(supported);
      b->cursor = nir_after_cf_list(&b->impl->body);
      nir_def *input = nir_imm_int(b, 37);
      nir_def *value = nir_inot(b, nir_inot(b, input));
      nir_intrinsic_instr *store = filter_store(b, value);
      nir_search_value_union values[3] = {};
      filter_variable(values[0]);
      filter_expression(values[1], nir_op_inot, 0);
      filter_expression(values[2], nir_op_inot, 1);
      const uint16_t transition[] = {0};
      std::vector<per_op_table> operations(nir_num_search_ops);
      if (supported) {
         operations[nir_op_inot].num_filtered_states = 1;
         operations[nir_op_inot].table = transition;
      }
      const struct transform transforms[] = {{2, 0, 0}, {0, 0, ~0u}};
      const uint16_t offsets[] = {0, 1, 1};
      nir_algebraic_table table = {};
      table.transforms = transforms;
      table.transform_offsets = offsets;
      table.pass_op_table = operations.data();
      table.values = values;
      const bool enabled[] = {true};
      ASSERT_TRUE(nir_algebraic_impl(b->impl, enabled, &table));
      EXPECT_EQ(store->src[0].ssa, input);
      nir_validate_shader(b->shader, "state zero rule retained");
      nir_opt_constant_folding(b->shader);
      EXPECT_EQ(nir_src_as_uint(store->src[0]), 37u);
   }
}

static std::vector<unsigned> filter_eligible_visits;

static bool
record_filter_eligible(const nir_alu_instr *instr)
{
   filter_eligible_visits.push_back(instr->def.index);
   return true;
}

TEST_F(nir_opt_algebraic_test, newly_eligible_users_remain_queued)
{
   nir_def *input = nir_load_local_invocation_index(b);
   nir_def *opaque = nir_iadd(b, input, nir_imm_int(b, 1));
   nir_def *single_not = nir_inot(b, opaque);
   nir_def *inner_neg = nir_ineg(b, single_not);
   nir_def *outer_neg = nir_ineg(b, inner_neg);
   nir_def *parent = nir_inot(b, outer_neg);
   nir_intrinsic_instr *store = filter_store(b, parent);
   const unsigned neg_index = outer_neg->index;
   const unsigned parent_index = parent->index;
   nir_search_value_union values[5] = {};
   filter_variable(values[0]);
   filter_expression(values[1], nir_op_inot, 0);
   filter_expression(values[2], nir_op_inot, 1, 0, 0);
   filter_expression(values[3], nir_op_ineg, 0);
   filter_expression(values[4], nir_op_ineg, 3, 0, 0);
   const uint16_t not_filter[] = {0, 0, 1, 1, 0, 0};
   const uint16_t neg_filter[] = {0, 0, 0, 0, 1, 1};
   const uint16_t not_transition[] = {2, 3};
   const uint16_t neg_transition[] = {4, 5};
   std::vector<per_op_table> operations(nir_num_search_ops);
   operations[nir_op_inot] = {not_filter, 2, not_transition};
   operations[nir_op_ineg] = {neg_filter, 2, neg_transition};
   const struct transform transforms[] = {
      {0, 0, ~0u}, {2, 0, 0}, {0, 0, ~0u}, {4, 0, 0}, {0, 0, ~0u},
   };
   const uint16_t offsets[] = {0, 0, 0, 1, 0, 3};
   const nir_search_expression_cond conditions[] = {record_filter_eligible};
   nir_algebraic_table table = {};
   table.transforms = transforms;
   table.transform_offsets = offsets;
   table.pass_op_table = operations.data();
   table.values = values;
   table.expression_cond = conditions;
   const bool enabled[] = {true};
   filter_eligible_visits.clear();
   ASSERT_TRUE(nir_algebraic_impl(b->impl, enabled, &table));
   EXPECT_EQ(store->src[0].ssa, opaque);
   EXPECT_EQ(filter_eligible_visits,
             (std::vector<unsigned>{neg_index, parent_index}));
   nir_validate_shader(b->shader, "newly eligible user");
   filter_eligible_visits.clear();
   EXPECT_FALSE(nir_algebraic_impl(b->impl, enabled, &table));
   EXPECT_TRUE(filter_eligible_visits.empty());
}

TEST_F(nir_opt_algebraic_test, filtered_opcodes_use_conversion_search_class)
{
   nir_def *input = nir_u2u16(b, nir_load_local_invocation_index(b));
   nir_def *middle = nir_build_alu(b, nir_op_i2i32, input, nullptr, nullptr, nullptr);
   nir_def *root = nir_build_alu(b, nir_op_i2i64, middle, nullptr, nullptr, nullptr);
   nir_intrinsic_instr *store = filter_store(b, root);
   nir_search_value_union values[4] = {};
   filter_variable(values[0], 16);
   filter_expression(values[1], nir_search_op_i2i, 0, 32);
   filter_expression(values[2], nir_search_op_i2i, 1, 64);
   filter_expression(values[3], nir_search_op_i2i, 0, 64);
   const uint16_t transition[] = {2};
   std::vector<per_op_table> operations(nir_num_search_ops);
   operations[nir_search_op_i2i] = {nullptr, 1, transition};
   const struct transform transforms[] = {{2, 3, 0}, {0, 0, ~0u}};
   const uint16_t offsets[] = {1, 1, 0};
   nir_algebraic_table table = {};
   table.transforms = transforms;
   table.transform_offsets = offsets;
   table.pass_op_table = operations.data();
   table.values = values;
   const bool enabled[] = {true};
   ASSERT_TRUE(nir_algebraic_impl(b->impl, enabled, &table));
   nir_alu_instr *result = nir_src_as_alu(store->src[0]);
   ASSERT_NE(result, nullptr);
   EXPECT_EQ(result->op, nir_op_i2i64);
   EXPECT_EQ(result->src[0].src.ssa, input);
   EXPECT_EQ(result->def.bit_size, 64u);
   nir_validate_shader(b->shader, "conversion search class");
}

TEST_F(nir_opt_algebraic_test, skipped_opcodes_have_flags_cleared)
{
   nir_def *input = nir_load_local_invocation_index(b);
   nir_def *opaque = nir_iadd(b, input, nir_imm_int(b, 1));
   nir_intrinsic_instr *store = filter_store(b, opaque);
   std::vector<nir_instr *> visited;
   nir_foreach_block(block, b->impl) {
      nir_foreach_instr(instr, block) {
         instr->pass_flags = 0x5a;
         visited.push_back(instr);
      }
   }
   std::vector<per_op_table> operations(nir_num_search_ops);
   const struct transform transforms[] = {{0, 0, ~0u}};
   const uint16_t offsets[] = {0, 0};
   nir_search_value_union values[1] = {};
   nir_algebraic_table table = {};
   table.transforms = transforms;
   table.transform_offsets = offsets;
   table.pass_op_table = operations.data();
   table.values = values;
   const bool enabled[] = {true};
   EXPECT_FALSE(nir_algebraic_impl(b->impl, enabled, &table));
   EXPECT_EQ(store->src[0].ssa, opaque);
   EXPECT_GT(visited.size(), 1u);
   for (nir_instr *instr : visited)
      EXPECT_EQ(instr->pass_flags, 0u);
}


static std::vector<std::pair<unsigned, unsigned>> root_source_visits;
static unsigned root_condition_calls;

static bool
record_root_source(const nir_search_state *, const nir_alu_instr *,
                   unsigned src, unsigned components, const uint8_t *)
{
   root_source_visits.emplace_back(src, components);
   return true;
}

static bool
root_condition_second_try(const nir_alu_instr *)
{
   return ++root_condition_calls == 2;
}

static nir_def *
root_header_input(nir_builder *build, unsigned components, const char *name)
{
   nir_variable *variable = nir_local_variable_create(
      build->impl, glsl_vector_type(GLSL_TYPE_FLOAT, components), name);
   return nir_load_var(build, variable);
}

TEST_F(nir_opt_algebraic_test, root_header_guards_preserve_source_visits)
{
   for (unsigned components : {1u, 2u, 4u, 8u, 16u}) {
      for (unsigned mode = 0; mode < 6; ++mode) {
         SCOPED_TRACE(components);
         SCOPED_TRACE(mode);
         nir_shader_compiler_options options = {};
         nir_builder current = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &options, "root header guards");
         ralloc_steal(b->shader, current.shader);
         nir_def *lhs = root_header_input(&current, components, "lhs");
         nir_def *rhs = root_header_input(&current, components, "rhs");
         nir_def *root = nir_fadd(&current, lhs, rhs);
         nir_def_as_alu(root)->fp_math_ctrl = nir_fp_exact;
         nir_intrinsic_instr *store = filter_store(&current, root);
         nir_search_value_union values[3] = {};
         filter_variable(values[0]);
         filter_variable(values[1]);
         values[1].variable.variable = 1;
         values[0].variable.cond_index = 0;
         values[1].variable.cond_index = 0;
         filter_expression(values[2], mode == 2 ? nir_op_fsub : nir_op_fadd,
                           0, mode == 1 ? 64 : 32);
         values[2].expression.srcs[1] = 1;
         values[2].expression.comm_exprs = 1;
         values[2].expression.comm_expr_mask = 1;
         if (mode == 3)
            values[2].expression.fp_math_ctrl_exclude = nir_fp_exact;
         if (mode >= 4)
            values[2].expression.swizzle = mode == 4 ? 1 : 0;
         const uint16_t transition[] = {2};
         std::vector<per_op_table> operations(nir_num_search_ops);
         operations[nir_op_fadd] = {nullptr, 1, transition};
         const struct transform transforms[] = {
            {0, 0, ~0u}, {2, 0, 0}, {0, 0, ~0u},
         };
         const uint16_t offsets[] = {0, 0, 1};
         const nir_search_variable_cond conditions[] = {record_root_source};
         nir_algebraic_table table = {};
         table.transforms = transforms;
         table.transform_offsets = offsets;
         table.pass_op_table = operations.data();
         table.values = values;
         table.variable_cond = conditions;
         const bool enabled[] = {true};
         const bool expected = mode == 0 || (mode == 5 && components == 1);
         root_source_visits.clear();
         EXPECT_EQ(nir_algebraic_impl(current.impl, enabled, &table), expected);
         EXPECT_EQ(store->src[0].ssa, expected ? lhs : root);
         if (expected)
            EXPECT_EQ(root_source_visits,
                      (std::vector<std::pair<unsigned, unsigned>>{{0, components}, {1, components}}));
         else
            EXPECT_TRUE(root_source_visits.empty());
         nir_validate_shader(current.shader, "root header guards");
      }
   }
}

TEST_F(nir_opt_algebraic_test, root_expression_conditions_keep_direction_order)
{
   for (unsigned components : {1u, 4u}) {
      for (bool reject_header : {false, true}) {
         SCOPED_TRACE(components);
         SCOPED_TRACE(reject_header);
         nir_shader_compiler_options options = {};
         nir_builder current = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &options, "root condition order");
         ralloc_steal(b->shader, current.shader);
         nir_def *lhs = root_header_input(&current, components, "lhs");
         nir_def *rhs = root_header_input(&current, components, "rhs");
         nir_def *root = nir_fadd(&current, lhs, rhs);
         nir_intrinsic_instr *store = filter_store(&current, root);
         nir_search_value_union values[3] = {};
         filter_variable(values[0]);
         filter_variable(values[1]);
         values[1].variable.variable = 1;
         values[0].variable.cond_index = 0;
         values[1].variable.cond_index = 0;
         filter_expression(values[2], nir_op_fadd, 0,
                           reject_header ? 64 : 32, 0);
         values[2].expression.srcs[1] = 1;
         values[2].expression.comm_exprs = 1;
         values[2].expression.comm_expr_mask = 1;
         const uint16_t transition[] = {2};
         std::vector<per_op_table> operations(nir_num_search_ops);
         operations[nir_op_fadd] = {nullptr, 1, transition};
         const struct transform transforms[] = {
            {0, 0, ~0u}, {2, 0, 0}, {0, 0, ~0u},
         };
         const uint16_t offsets[] = {0, 0, 1};
         const nir_search_expression_cond expression_conditions[] = {root_condition_second_try};
         const nir_search_variable_cond variable_conditions[] = {record_root_source};
         nir_algebraic_table table = {};
         table.transforms = transforms;
         table.transform_offsets = offsets;
         table.pass_op_table = operations.data();
         table.values = values;
         table.expression_cond = expression_conditions;
         table.variable_cond = variable_conditions;
         const bool enabled[] = {true};
         root_condition_calls = 0;
         root_source_visits.clear();
         EXPECT_EQ(nir_algebraic_impl(current.impl, enabled, &table), !reject_header);
         EXPECT_EQ(root_condition_calls, 2u);
         EXPECT_EQ(store->src[0].ssa, reject_header ? root : rhs);
         if (reject_header)
            EXPECT_TRUE(root_source_visits.empty());
         else
            EXPECT_EQ(root_source_visits,
                      (std::vector<std::pair<unsigned, unsigned>>{{1, components}, {0, components}}));
         nir_validate_shader(current.shader, "root condition order");
      }
   }
}

TEST_F(nir_opt_algebraic_test, root_fp_controls_reach_replacement)
{
   for (unsigned components : {1u, 4u}) {
      for (unsigned control = 0; control <= nir_fp_no_fast_math; ++control) {
         if ((control & (nir_fp_no_contract | nir_fp_no_reassoc)) &&
             !(control & nir_fp_no_transform))
            continue;
         SCOPED_TRACE(components);
         SCOPED_TRACE(control);
         nir_shader_compiler_options options = {};
         nir_builder current = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &options, "root FP controls");
         ralloc_steal(b->shader, current.shader);
         nir_def *lhs = root_header_input(&current, components, "lhs");
         nir_def *rhs = root_header_input(&current, components, "rhs");
         nir_def *root = nir_fadd(&current, lhs, rhs);
         nir_def_as_alu(root)->fp_math_ctrl = control;
         nir_intrinsic_instr *store = filter_store(&current, root);
         nir_search_value_union values[4] = {};
         filter_variable(values[0]);
         filter_variable(values[1]);
         values[1].variable.variable = 1;
         filter_expression(values[2], nir_op_fadd, 0, 32);
         values[2].expression.srcs[1] = 1;
         values[2].expression.comm_exprs = 1;
         values[2].expression.comm_expr_mask = 1;
         filter_expression(values[3], nir_op_fmul, 0, 32);
         values[3].expression.srcs[1] = 1;
         const uint16_t transition[] = {2};
         std::vector<per_op_table> operations(nir_num_search_ops);
         operations[nir_op_fadd] = {nullptr, 1, transition};
         const struct transform transforms[] = {
            {0, 0, ~0u}, {2, 3, 0}, {0, 0, ~0u},
         };
         const uint16_t offsets[] = {0, 0, 1};
         nir_algebraic_table table = {};
         table.transforms = transforms;
         table.transform_offsets = offsets;
         table.pass_op_table = operations.data();
         table.values = values;
         const bool enabled[] = {true};
         ASSERT_TRUE(nir_algebraic_impl(current.impl, enabled, &table));
         nir_alu_instr *replacement = nir_src_as_alu(store->src[0]);
         ASSERT_NE(replacement, nullptr);
         EXPECT_EQ(replacement->op, nir_op_fmul);
         EXPECT_EQ(replacement->fp_math_ctrl,
                   nir_op_valid_fp_math_ctrl(nir_op_fmul, control));
         nir_validate_shader(current.shader, "root FP controls");
      }
   }
}

TEST_F(nir_opt_algebraic_test, fixed_output_root_preserves_input_components)
{
   for (int swizzle : {-1, 0, 1}) {
      SCOPED_TRACE(swizzle);
      nir_shader_compiler_options options = {};
      nir_builder current = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &options, "fixed output root");
      ralloc_steal(b->shader, current.shader);
      nir_def *lhs = root_header_input(&current, 4, "lhs");
      nir_def *rhs = root_header_input(&current, 4, "rhs");
      nir_def *root = nir_fdot4(&current, lhs, rhs);
      nir_intrinsic_instr *store = filter_store(&current, root);
      nir_search_value_union values[3] = {};
      filter_variable(values[0]);
      filter_variable(values[1]);
      values[1].variable.variable = 1;
      values[0].variable.cond_index = 0;
      values[1].variable.cond_index = 0;
      filter_expression(values[2], nir_op_fdot4, 0, 32);
      values[2].expression.srcs[1] = 1;
      values[2].expression.comm_exprs = 1;
      values[2].expression.comm_expr_mask = 1;
      values[2].expression.swizzle = swizzle;
      const uint16_t transition[] = {2};
      std::vector<per_op_table> operations(nir_num_search_ops);
      operations[nir_op_fdot4] = {nullptr, 1, transition};
      const struct transform transforms[] = {
         {0, 0, ~0u}, {2, 0, 0}, {0, 0, ~0u},
      };
      const uint16_t offsets[] = {0, 0, 1};
      const nir_search_variable_cond conditions[] = {record_root_source};
      nir_algebraic_table table = {};
      table.transforms = transforms;
      table.transform_offsets = offsets;
      table.pass_op_table = operations.data();
      table.values = values;
      table.variable_cond = conditions;
      const bool enabled[] = {true};
      root_source_visits.clear();
      EXPECT_EQ(nir_algebraic_impl(current.impl, enabled, &table), swizzle <= 0);
      if (swizzle <= 0)
         EXPECT_EQ(root_source_visits,
                   (std::vector<std::pair<unsigned, unsigned>>{{0, 4}, {1, 4}}));
      else {
         EXPECT_TRUE(root_source_visits.empty());
         EXPECT_EQ(store->src[0].ssa, root);
      }
      nir_validate_shader(current.shader, "fixed output root");
   }
}


static nir_def *
binding_input(nir_builder *build, unsigned components, const char *name)
{
   nir_variable *variable = nir_local_variable_create(
      build->impl, glsl_vector_type(GLSL_TYPE_UINT, components), name);
   return nir_load_var(build, variable);
}

TEST_F(nir_opt_algebraic_test, repeated_binding_keeps_first_active_swizzle)
{
   for (unsigned components : {1u, 2u, 4u, 8u, 16u}) {
      for (bool same : {false, true}) {
         SCOPED_TRACE(components);
         SCOPED_TRACE(same);
         nir_shader_compiler_options options = {};
         nir_builder current = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &options, "repeated variable swizzle");
         ralloc_steal(b->shader, current.shader);
         nir_def *input = binding_input(&current, 16, "input");
         nir_alu_instr *add = nir_alu_instr_create(current.shader, nir_op_iadd);
         nir_def_init(&add->instr, &add->def, components, 32);
         for (unsigned src = 0; src < 2; ++src) {
            add->src[src].src = nir_src_for_ssa(input);
            for (unsigned i = 0; i < components; ++i)
               add->src[src].swizzle[i] = (i + 3 + (src && !same)) % 16;
         }
         nir_builder_instr_insert(&current, &add->instr);
         nir_intrinsic_instr *store = filter_store(&current, &add->def);
         nir_search_value_union values[2] = {};
         filter_variable(values[0]);
         filter_expression(values[1], nir_op_iadd, 0);
         values[1].expression.srcs[1] = 0;
         const uint16_t transition[] = {2};
         std::vector<per_op_table> operations(nir_num_search_ops);
         operations[nir_op_iadd] = {nullptr, 1, transition};
         const struct transform transforms[] = {
            {0, 0, ~0u}, {1, 0, 0}, {0, 0, ~0u},
         };
         const uint16_t offsets[] = {0, 0, 1};
         nir_algebraic_table table = {};
         table.transforms = transforms;
         table.transform_offsets = offsets;
         table.pass_op_table = operations.data();
         table.values = values;
         const bool enabled[] = {true};
         EXPECT_EQ(nir_algebraic_impl(current.impl, enabled, &table), same);
         if (same) {
            nir_alu_instr *result = nir_src_as_alu(store->src[0]);
            ASSERT_NE(result, nullptr);
            EXPECT_EQ(result->op, nir_op_mov);
            EXPECT_EQ(result->src[0].src.ssa, input);
            for (unsigned i = 0; i < components; ++i)
               EXPECT_EQ(result->src[0].swizzle[i], (i + 3) % 16);
         } else {
            EXPECT_EQ(store->src[0].ssa, &add->def);
         }
         nir_validate_shader(current.shader, "repeated variable swizzle");
      }
   }
}

static std::vector<unsigned> binding_condition_visits;

static bool
binding_accepts_flipped_source(const nir_search_state *, const nir_alu_instr *,
                               unsigned src, unsigned, const uint8_t *)
{
   binding_condition_visits.push_back(src);
   return src == 1;
}

TEST_F(nir_opt_algebraic_test, conditioned_and_direct_bindings_keep_direction_state)
{
   nir_def *lhs = binding_input(b, 4, "lhs");
   nir_def *rhs = binding_input(b, 4, "rhs");
   nir_def *root = nir_iadd(b, lhs, rhs);
   nir_intrinsic_instr *store = filter_store(b, root);
   nir_search_value_union values[3] = {};
   filter_variable(values[0]);
   values[0].variable.cond_index = 0;
   filter_variable(values[1]);
   values[1].variable.variable = 1;
   values[1].variable.swizzle[0] = NIR_MAX_VEC_COMPONENTS - 1;
   filter_expression(values[2], nir_op_iadd, 0);
   values[2].expression.srcs[1] = 1;
   values[2].expression.comm_exprs = 1;
   values[2].expression.comm_expr_mask = 1;
   const uint16_t transition[] = {2};
   std::vector<per_op_table> operations(nir_num_search_ops);
   operations[nir_op_iadd] = {nullptr, 1, transition};
   const struct transform transforms[] = {
      {0, 0, ~0u}, {2, 1, 0}, {0, 0, ~0u},
   };
   const uint16_t offsets[] = {0, 0, 1};
   const nir_search_variable_cond conditions[] = {binding_accepts_flipped_source};
   nir_algebraic_table table = {};
   table.transforms = transforms;
   table.transform_offsets = offsets;
   table.pass_op_table = operations.data();
   table.values = values;
   table.variable_cond = conditions;
   const bool enabled[] = {true};
   binding_condition_visits.clear();
   ASSERT_TRUE(nir_algebraic_impl(b->impl, enabled, &table));
   EXPECT_EQ(binding_condition_visits, (std::vector<unsigned>{0, 1}));
   EXPECT_EQ(store->src[0].ssa, lhs);
   nir_validate_shader(b->shader, "conditioned and direct bindings");
}

TEST_F(nir_opt_algebraic_test, rejected_constant_binding_does_not_commit_state)
{
   for (unsigned components : {1u, 4u, 16u}) {
      SCOPED_TRACE(components);
      nir_shader_compiler_options options = {};
      nir_builder current = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &options, "rejected constant binding");
      ralloc_steal(b->shader, current.shader);
      nir_def *lhs = binding_input(&current, components, "lhs");
      nir_def *rhs = binding_input(&current, components, "rhs");
      nir_def *root = nir_iadd(&current, lhs, rhs);
      nir_intrinsic_instr *store = filter_store(&current, root);
      nir_search_value_union values[5] = {};
      filter_variable(values[0]);
      values[0].variable.is_constant = true;
      filter_variable(values[1]);
      filter_variable(values[2]);
      values[2].variable.variable = 1;
      filter_expression(values[3], nir_op_iadd, 0);
      values[3].expression.srcs[1] = 2;
      values[3].expression.comm_exprs = 1;
      values[3].expression.comm_expr_mask = 1;
      filter_expression(values[4], nir_op_iadd, 1);
      values[4].expression.srcs[1] = 2;
      values[4].expression.comm_exprs = 1;
      values[4].expression.comm_expr_mask = 1;
      const uint16_t transition[] = {2};
      std::vector<per_op_table> operations(nir_num_search_ops);
      operations[nir_op_iadd] = {nullptr, 1, transition};
      const struct transform transforms[] = {
         {0, 0, ~0u}, {3, 1, 0}, {4, 1, 0}, {0, 0, ~0u},
      };
      const uint16_t offsets[] = {0, 0, 1};
      nir_algebraic_table table = {};
      table.transforms = transforms;
      table.transform_offsets = offsets;
      table.pass_op_table = operations.data();
      table.values = values;
      const bool enabled[] = {true};
      ASSERT_TRUE(nir_algebraic_impl(current.impl, enabled, &table));
      EXPECT_EQ(store->src[0].ssa, lhs);
      nir_validate_shader(current.shader, "rejected constant binding");
   }
}


static std::vector<unsigned> seen_binding_callback_visits;

static bool
record_seen_binding_callback(const nir_search_state *, const nir_alu_instr *,
                             unsigned src, unsigned, const uint8_t *)
{
   seen_binding_callback_visits.push_back(src);
   return true;
}

TEST_F(nir_opt_algebraic_test, seen_binding_conditions_precede_SSA_rejection)
{
   for (bool wrong_width : {false, true}) {
      SCOPED_TRACE(wrong_width);
      nir_shader_compiler_options options = {};
      nir_builder current = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &options, "seen binding callback");
      ralloc_steal(b->shader, current.shader);
      nir_def *lhs = binding_input(&current, 4, "lhs");
      nir_def *rhs = binding_input(&current, 4, "rhs");
      nir_def *root = nir_iadd(&current, lhs, rhs);
      nir_intrinsic_instr *store = filter_store(&current, root);
      nir_search_value_union values[3] = {};
      filter_variable(values[0]);
      filter_variable(values[1], wrong_width ? 64 : 32);
      values[1].variable.cond_index = 0;
      filter_expression(values[2], nir_op_iadd, 0);
      values[2].expression.srcs[1] = 1;
      values[2].expression.comm_exprs = 1;
      values[2].expression.comm_expr_mask = 1;
      const uint16_t transition[] = {2};
      std::vector<per_op_table> operations(nir_num_search_ops);
      operations[nir_op_iadd] = {nullptr, 1, transition};
      const struct transform transforms[] = {
         {0, 0, ~0u}, {2, 0, 0}, {0, 0, ~0u},
      };
      const uint16_t offsets[] = {0, 0, 1};
      const nir_search_variable_cond conditions[] = {record_seen_binding_callback};
      nir_algebraic_table table = {};
      table.transforms = transforms;
      table.transform_offsets = offsets;
      table.pass_op_table = operations.data();
      table.values = values;
      table.variable_cond = conditions;
      const bool enabled[] = {true};
      seen_binding_callback_visits.clear();
      EXPECT_FALSE(nir_algebraic_impl(current.impl, enabled, &table));
      EXPECT_EQ(store->src[0].ssa, root);
      if (wrong_width)
         EXPECT_TRUE(seen_binding_callback_visits.empty());
      else
         EXPECT_EQ(seen_binding_callback_visits, (std::vector<unsigned>{1, 0}));
      nir_validate_shader(current.shader, "seen binding callback");
   }
}

TEST_F(nir_opt_algebraic_test, same_variable_loads_keep_distinct_SSA_bindings)
{
   for (unsigned components : {1u, 4u, 16u}) {
      SCOPED_TRACE(components);
      nir_shader_compiler_options options = {};
      nir_builder current = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &options, "distinct binding identity");
      ralloc_steal(b->shader, current.shader);
      nir_variable *variable = nir_local_variable_create(
         current.impl, glsl_vector_type(GLSL_TYPE_UINT, components), "value");
      nir_def *lhs = nir_load_var(&current, variable);
      nir_def *rhs = nir_load_var(&current, variable);
      ASSERT_NE(lhs, rhs);
      nir_def *root = nir_iadd(&current, lhs, rhs);
      nir_intrinsic_instr *store = filter_store(&current, root);
      nir_search_value_union values[2] = {};
      filter_variable(values[0]);
      filter_expression(values[1], nir_op_iadd, 0);
      values[1].expression.srcs[1] = 0;
      const uint16_t transition[] = {2};
      std::vector<per_op_table> operations(nir_num_search_ops);
      operations[nir_op_iadd] = {nullptr, 1, transition};
      const struct transform transforms[] = {
         {0, 0, ~0u}, {1, 0, 0}, {0, 0, ~0u},
      };
      const uint16_t offsets[] = {0, 0, 1};
      nir_algebraic_table table = {};
      table.transforms = transforms;
      table.transform_offsets = offsets;
      table.pass_op_table = operations.data();
      table.values = values;
      const bool enabled[] = {true};
      EXPECT_FALSE(nir_algebraic_impl(current.impl, enabled, &table));
      EXPECT_EQ(store->src[0].ssa, root);
      nir_validate_shader(current.shader, "distinct binding identity");
   }
}

TEST_F(nir_opt_algebraic_test, bounds_agx_unsigned_end_and_bound)
{
   for (unsigned bits : {8u, 16u, 32u, 64u}) {
      uint64_t mask = UINT64_MAX >> (64 - bits);
      uint64_t sign = UINT64_C(1) << (bits - 1);
      const uint64_t edge[] = {0, 1, 63, 64, 65, sign - 1, sign,
                              sign + 1, mask - 1, mask};
      const uint64_t data[] = {0, sign, mask, UINT64_C(0x7fc12345) & mask};
      for (uint64_t value : data) {
         for (uint64_t end : edge) {
            for (uint64_t bound : edge) {
               nir_const_value v[] = {
                  nir_const_value_for_uint(value, bits),
                  nir_const_value_for_uint(end, bits),
                  nir_const_value_for_uint(bound, bits),
               };
               nir_const_value *src[] = {&v[0], &v[1], &v[2]};
               nir_const_value out = {};
               uint16_t poison = 0;
               nir_eval_const_opcode(nir_op_bounds_agx, &out, &poison,
                                     1, bits, src, 0);
               EXPECT_EQ(poison, 0u);
               EXPECT_EQ(nir_const_value_as_uint(out, bits),
                         end <= bound ? value : 0)
                  << "bits=" << bits << " end=" << end << " bound=" << bound;
            }
         }
      }
   }
}

TEST_F(nir_opt_algebraic_test, bounds_agx_constant_fold_matches_unsigned_backend)
{
   nir_def *checked = nir_bounds_agx(b, nir_imm_int(b, 0xdeadbeef),
                                      nir_imm_int(b, UINT32_MAX),
                                      nir_imm_int(b, 64));
   nir_deref_instr *deref = nir_build_deref_var(b, res_var);
   nir_intrinsic_instr *store =
      nir_build_store_deref(b, &deref->def, checked, 1, (gl_access_qualifier)0);
   ASSERT_TRUE(nir_opt_constant_folding(b->shader));
   ASSERT_TRUE(nir_src_is_const(store->src[1]));
   EXPECT_EQ(nir_src_as_uint(store->src[1]), 0u);
}

TEST_F(nir_opt_algebraic_test, umod_pow2_src2)
{
   for (int i = 0; i <= 9; i++)
      test_2src_op(nir_op_umod, i, 4);
   test_2src_op(nir_op_umod, UINT32_MAX, 4);
}

TEST_F(nir_opt_algebraic_test, imod_pow2_src2)
{
   for (int i = -9; i <= 9; i++) {
      test_2src_op(nir_op_imod, i, 4);
      test_2src_op(nir_op_imod, i, -4);
      test_2src_op(nir_op_imod, i, INT32_MIN);
   }
   test_2src_op(nir_op_imod, INT32_MAX, 4);
   test_2src_op(nir_op_imod, INT32_MAX, -4);
   test_2src_op(nir_op_imod, INT32_MIN, 4);
   test_2src_op(nir_op_imod, INT32_MIN, -4);
   test_2src_op(nir_op_imod, INT32_MIN, INT32_MIN);
}

TEST_F(nir_opt_algebraic_test, irem_pow2_src2)
{
   for (int i = -9; i <= 9; i++) {
      test_2src_op(nir_op_irem, i, 4);
      test_2src_op(nir_op_irem, i, -4);
   }
   test_2src_op(nir_op_irem, INT32_MAX, 4);
   test_2src_op(nir_op_irem, INT32_MAX, -4);
   test_2src_op(nir_op_irem, INT32_MIN, 4);
   test_2src_op(nir_op_irem, INT32_MIN, -4);
}

TEST_F(nir_opt_algebraic_test, msad)
{
   options.lower_bitfield_extract = true;
   options.has_bfe = true;
   options.has_msad = true;

   nir_def *src0 = nir_load_var(b, nir_local_variable_create(b->impl, glsl_int_type(), "src0"));
   nir_def *src1 = nir_load_var(b, nir_local_variable_create(b->impl, glsl_int_type(), "src1"));

   /* This mimics the sequence created by vkd3d-proton. */
   nir_def *res = NULL;
   for (unsigned i = 0; i < 4; i++) {
      nir_def *ref = nir_ubitfield_extract(b, src0, nir_imm_int(b, i * 8), nir_imm_int(b, 8));
      nir_def *src = nir_ubitfield_extract(b, src1, nir_imm_int(b, i * 8), nir_imm_int(b, 8));
      nir_def *is_ref_zero = nir_ieq_imm(b, ref, 0);
      nir_def *abs_diff = nir_iabs(b, nir_isub(b, ref, src));
      nir_def *masked_diff = nir_bcsel(b, is_ref_zero, nir_imm_int(b, 0), abs_diff);
      if (res)
         res = nir_iadd(b, res, masked_diff);
      else
         res = masked_diff;
   }

   nir_store_var(b, res_var, res, 0x1);

   while (nir_opt_algebraic(b->shader)) {
      nir_opt_constant_folding(b->shader);
      nir_opt_dce(b->shader);
   }

   require_one_alu(nir_op_msad_4x8);
}

TEST_F(nir_opt_algebraic_test, commutative_match_does_not_leak_fp_math_ctrl)
{
   nir_def *src = nir_load_var(
      b, nir_local_variable_create(b->impl, glsl_float_type(), "src"));
   nir_def *inner = nir_fneg(b, src);
   nir_def_as_alu(inner)->fp_math_ctrl = nir_fp_exact;
   nir_def *outer = nir_fneg(b, inner);
   nir_def *result = nir_fmin(b, outer, inner);
   nir_variable *result_var =
      nir_local_variable_create(b->impl, glsl_float_type(), "result");
   nir_intrinsic_instr *store = nir_build_store_deref(
      b, &nir_build_deref_var(b, result_var)->def, result, 0x1);

   ASSERT_TRUE(nir_opt_algebraic(b->shader));
   nir_alu_instr *optimized = nir_src_as_alu(store->src[1]);
   ASSERT_NE(optimized, nullptr);
   EXPECT_EQ(optimized->op, nir_op_fneg);
   EXPECT_EQ(optimized->fp_math_ctrl & nir_fp_exact, 0);
}

TEST_F(nir_opt_mqsad_test, mqsad)
{
   options.lower_bitfield_extract = true;
   options.has_bfe = true;
   options.has_msad = true;
   options.has_shfr32 = true;

   nir_def *ref = nir_load_var(b, nir_local_variable_create(b->impl, glsl_int_type(), "ref"));
   nir_def *src = nir_load_var(b, nir_local_variable_create(b->impl, glsl_ivec_type(2), "src"));
   nir_def *accum = nir_load_var(b, nir_local_variable_create(b->impl, glsl_ivec_type(4), "accum"));

   nir_def *srcx = nir_channel(b, src, 0);
   nir_def *srcy = nir_channel(b, src, 1);

   nir_def *res[4];
   for (unsigned i = 0; i < 4; i++) {
      nir_def *src1 = srcx;
      switch (i) {
      case 0:
         break;
      case 1:
         src1 = nir_bitfield_select(b, nir_imm_int(b, 0xff000000), nir_ishl_imm(b, srcy, 24),
                                    nir_ushr_imm(b, srcx, 8));
         break;
      case 2:
         src1 = nir_bitfield_select(b, nir_imm_int(b, 0xffff0000), nir_ishl_imm(b, srcy, 16),
                                    nir_extract_u16(b, srcx, nir_imm_int(b, 1)));
         break;
      case 3:
         src1 = nir_bitfield_select(b, nir_imm_int(b, 0xffffff00), nir_ishl_imm(b, srcy, 8),
                                    nir_extract_u8_imm(b, srcx, 3));
         break;
      }

      res[i] = nir_msad_4x8(b, ref, src1, nir_channel(b, accum, i));
   }

   nir_store_var(b, nir_local_variable_create(b->impl, glsl_ivec_type(4), "res"), nir_vec(b, res, 4), 0xf);

   while (nir_opt_algebraic(b->shader)) {
      nir_opt_constant_folding(b->shader);
      nir_opt_dce(b->shader);
   }

   ASSERT_TRUE(nir_opt_mqsad(b->shader));
   nir_opt_copy_prop(b->shader);
   nir_opt_dce(b->shader);

   require_one_alu(nir_op_mqsad_4x8);
}

TEST_F(nir_opt_idiv_const_test, umod)
{
   for (uint32_t d : {16u, 17u, 0u, UINT32_MAX}) {
      for (int i = 0; i <= 40; i++)
         test_2src_op(nir_op_umod, i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_umod, UINT32_MAX - i, d);
   }
}

TEST_F(nir_opt_idiv_const_test, imod)
{
   for (int32_t d : {16, -16, 17, -17, 0, INT32_MIN, INT32_MAX}) {
      for (int i = -40; i <= 40; i++)
         test_2src_op(nir_op_imod, i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_imod, INT32_MIN + i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_imod, INT32_MAX - i, d);
   }
}

TEST_F(nir_opt_idiv_const_test, irem)
{
   for (int32_t d : {16, -16, 17, -17, 0, INT32_MIN, INT32_MAX}) {
      for (int i = -40; i <= 40; i++)
         test_2src_op(nir_op_irem, i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_irem, INT32_MIN + i, d);
      for (int i = 0; i < 20; i++)
         test_2src_op(nir_op_irem, INT32_MAX - i, d);
   }
}

static const nir_alu_instr *commutation_test_root;
static unsigned commutation_root_attempts;

static bool
record_commutation_attempt(const nir_alu_instr *instr)
{
   if (instr == commutation_test_root)
      ++commutation_root_attempts;
   return true;
}

TEST_F(nir_opt_algebraic_test, commutation_masks_preserve_direction_order)
{
   static const nir_shader_compiler_options options = {};
   for (int bit : {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8}) {
      for (bool swapped : {false, true}) {
         for (unsigned components : {1u, 2u, 4u, 8u, 16u}) {
            for (unsigned bits : {8u, 16u, 32u, 64u}) {
               SCOPED_TRACE(bit);
               SCOPED_TRACE(swapped);
               SCOPED_TRACE(components);
               SCOPED_TRACE(bits);
               nir_builder current = nir_builder_init_simple_shader(
                  MESA_SHADER_COMPUTE, &options, "commutation direction");
               nir_def *input = nir_undef(&current, components, bits);
               nir_def *zero = nir_imm_intN_t(&current, 0, bits);
               nir_def *value = swapped ? nir_iadd(&current, input, zero) :
                                         nir_iadd(&current, zero, input);
               nir_intrinsic_instr *store = filter_store(&current, value);
               nir_search_value_union values[3] = {};
               values[0].constant.value.type = nir_search_value_constant;
               values[0].constant.type = nir_type_uint;
               values[0].constant.data.u = 0;
               filter_variable(values[1], bits);
               filter_expression(values[2], nir_op_iadd, 0, bits, 0);
               values[2].expression.srcs[1] = 1;
               const bool tracked = bit >= 0 && bit < 8;
               values[2].expression.comm_expr_mask = tracked ?
                  static_cast<uint8_t>(1u << bit) : 0;
               values[2].expression.comm_exprs = static_cast<uint8_t>(bit + 1);
               std::vector<per_op_table> operations(nir_num_search_ops);
               const struct transform transforms[] = {{2, 1, 0}, {0, 0, ~0u}};
               const uint16_t offsets[] = {0, 1, 1};
               const nir_search_expression_cond predicates[] = {
                  record_commutation_attempt,
               };
               nir_algebraic_table table = {};
               table.transforms = transforms;
               table.transform_offsets = offsets;
               table.pass_op_table = operations.data();
               table.values = values;
               table.expression_cond = predicates;
               const bool enabled[] = {true};
               commutation_test_root = nir_def_as_alu(value);
               commutation_root_attempts = 0;
               nir_validate_shader(current.shader, "before commutation search");
               bool progress = nir_algebraic_impl(current.impl, enabled, &table);
               commutation_test_root = nullptr;
               const bool expected_match = !swapped || tracked;
               const unsigned combinations = 1u << MIN2(bit + 1, 8);
               const unsigned expected_attempts = expected_match ?
                  (swapped ? (1u << bit) + 1 : 1u) : combinations;
               EXPECT_EQ(progress, expected_match);
               EXPECT_EQ(commutation_root_attempts, expected_attempts);
               EXPECT_EQ(store->src[0].ssa, expected_match ? input : value);
               nir_validate_shader(current.shader, "after commutation search");
               ralloc_free(current.shader);
            }
         }
      }
   }
}

static unsigned
count_narrowing_i2i(nir_shader *shader)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;

            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if ((alu->op == nir_op_i2i8 || alu->op == nir_op_i2i16 ||
                 alu->op == nir_op_i2i32) &&
                nir_src_bit_size(alu->src[0].src) > alu->def.bit_size)
               count++;
         }
      }
   }
   return count;
}

TEST_F(nir_opt_algebraic_test, downcast_of_sign_extended_value_is_u2u)
{
   nir_def *base = nir_imm_int64(b, 0x100000);
   nir_def *value16 = nir_load_global(b, 1, 16, base, (gl_access_qualifier)0, 2, 0);
   nir_def *value32 = nir_i2i32(b, value16);
   nir_def *byte0 = nir_u2u8(b, value32);
   nir_store_global(b, nir_i2f32(b, byte0), nir_imm_int64(b, 0x200000), 1,
                    (gl_access_qualifier)0, 4, 0);

   nir_validate_shader(b->shader, "before the pass");
   run_pass();

   /* An i2i narrowing would fail validation; the downcast has to stay u2u. */
   EXPECT_EQ(count_narrowing_i2i(b->shader), 0u);
   bool u2u8_of_16 = false;
   nir_foreach_function_impl(impl, b->shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_u2u8 &&
                nir_src_bit_size(alu->src[0].src) == 16)
               u2u8_of_16 = true;
         }
      }
   }
   EXPECT_TRUE(u2u8_of_16);
   nir_validate_shader(b->shader, "after the pass");
}

TEST_F(nir_opt_algebraic_test, downcast_sign_extended_inner_narrow)
{
   nir_def *base = nir_imm_int64(b, 0x100000);
   nir_def *value8 = nir_load_global(b, 1, 8, base, (gl_access_qualifier)0, 1, 0);
   nir_def *value32 = nir_i2i32(b, value8);
   nir_def *word = nir_u2u16(b, value32);
   nir_store_global(b, nir_i2f32(b, word), nir_imm_int64(b, 0x200000), 1,
                    (gl_access_qualifier)0, 4, 0);

   nir_validate_shader(b->shader, "before the pass");
   run_pass();

   EXPECT_EQ(count_narrowing_i2i(b->shader), 0u);
   bool i2i16_of_8 = false;
   nir_foreach_function_impl(impl, b->shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_i2i16 &&
                nir_src_bit_size(alu->src[0].src) == 8)
               i2i16_of_8 = true;
         }
      }
   }
   EXPECT_TRUE(i2i16_of_8);
   nir_validate_shader(b->shader, "after the pass");
}

}

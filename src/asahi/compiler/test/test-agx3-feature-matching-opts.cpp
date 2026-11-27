/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_constant_expressions.h"
extern "C" {
#include "asahi/lib/agx_abi.h"
#include "agx_nir.h"
}

namespace {

nir_builder
make_compute(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

nir_def *
load_preamble_base(nir_builder *b, unsigned comps, unsigned bits, unsigned base)
{
   nir_def *d = nir_load_preamble(b, comps, bits);
   nir_intrinsic_set_base(nir_def_as_intrinsic(d), base);
   return d;
}

nir_def *
load_push_u32(nir_builder *b, unsigned offset, unsigned range = 16)
{
   nir_def *d = nir_load_push_constant(b, 1, 32, nir_imm_int(b, offset));
   nir_intrinsic_set_range(nir_def_as_intrinsic(d), range);
   return d;
}

nir_def *
load_global_ex(nir_builder *b, unsigned comps, unsigned bits, nir_def *addr,
               unsigned align_mul, gl_access_qualifier access = (gl_access_qualifier)0)
{
   nir_def *d = nir_load_global(b, comps, bits, addr);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(d);
   nir_intrinsic_set_align(intr, align_mul, 0);
   nir_intrinsic_set_access(intr, access);
   return d;
}

void
store_global_ex(nir_builder *b, nir_def *val, nir_def *addr, unsigned align_mul)
{
   nir_intrinsic_instr *intr = nir_store_global(b, val, addr);
   nir_intrinsic_set_align(intr, align_mul, 0);
}

nir_def *
load_agx_r8(nir_builder *b, nir_def *base, nir_def *idx)
{
   nir_def *d = nir_load_agx(b, 1, 8, base, idx);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(d);
   nir_intrinsic_set_base(intr, 0);
   nir_intrinsic_set_format(intr, PIPE_FORMAT_R8_UINT);
   return d;
}

void
store_agx_r8(nir_builder *b, nir_def *val, nir_def *base, nir_def *idx)
{
   nir_intrinsic_instr *intr = nir_store_agx(b, val, base, idx);
   nir_intrinsic_set_base(intr, 0);
   nir_intrinsic_set_format(intr, PIPE_FORMAT_R8_UINT);
}

nir_def *
buffer_addr(nir_builder *b, nir_def *byte_offset, unsigned preamble_base = 4)
{
   nir_def *set = load_preamble_base(b, 1, 64, preamble_base);
   nir_def *desc = nir_load_global_constant(b, 4, 32, set);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(desc);
   nir_intrinsic_set_access(
      intr, (gl_access_qualifier)(ACCESS_CAN_SPECULATE |
                                  ACCESS_BUFFER_BINDING_AGX));
   nir_intrinsic_set_align(intr, 16, 0);
   nir_def *base = nir_pack_64_2x32_split(b, nir_channel(b, desc, 0),
                                          nir_channel(b, desc, 1));
   return nir_iadd(b, base, nir_u2u64(b, byte_offset));
}

unsigned
count_alu_op(nir_shader *s, nir_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               n++;
         }
      }
   }
   return n;
}

unsigned
count_in_loops(nir_shader *s, nir_intrinsic_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         bool in_loop = false;
         for (nir_cf_node *p = block->cf_node.parent; p; p = p->parent) {
            if (p->type == nir_cf_node_loop) {
               in_loop = true;
               break;
            }
         }
         if (!in_loop)
            continue;
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               n++;
         }
      }
   }
   return n;
}

unsigned
count_alu_in_loops(nir_shader *s, nir_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         bool in_loop = false;
         for (nir_cf_node *p = block->cf_node.parent; p; p = p->parent) {
            if (p->type == nir_cf_node_loop) {
               in_loop = true;
               break;
            }
         }
         if (!in_loop)
            continue;
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               n++;
         }
      }
   }
   return n;
}

/* Build a balanced 16-entry select tree over byte leaves indexed by `idx`. */
nir_def *
build_byte_lut_tree(nir_builder *b, nir_def *idx, nir_def *const leaves[16],
                    unsigned lo, unsigned hi)
{
   if (lo == hi)
      return nir_u2u32(b, leaves[lo]);
   unsigned mid = (lo + hi + 1) / 2;
   nir_def *left = build_byte_lut_tree(b, idx, leaves, lo, mid - 1);
   nir_def *right = build_byte_lut_tree(b, idx, leaves, mid, hi);
   return nir_bcsel(b, nir_ult_imm(b, idx, mid), left, right);
}

uint32_t
eval_lut_expr(nir_scalar s, const uint8_t bytes[16], nir_def *const loads[16],
              nir_def *idx_def, uint32_t idx_val)
{
   if (s.def == idx_def)
      return idx_val;
   for (unsigned i = 0; i < 16; ++i) {
      if (s.def == loads[i])
         return bytes[i];
   }
   if (nir_scalar_is_const(s))
      return (uint32_t)nir_scalar_as_uint(s);
   EXPECT_TRUE(nir_scalar_is_alu(s));
   nir_alu_instr *alu = nir_def_as_alu(s.def);
   const nir_op_info *info = &nir_op_infos[alu->op];
   nir_const_value src[4][NIR_MAX_VEC_COMPONENTS] = {};
   nir_const_value *srcs[4];
   for (unsigned i = 0; i < info->num_inputs; ++i) {
      nir_scalar in = nir_scalar_chase_alu_src(s, i);
      uint32_t v = eval_lut_expr(in, bytes, loads, idx_def, idx_val);
      src[i][0] = nir_const_value_for_uint(v, in.def->bit_size);
      srcs[i] = src[i];
   }
   nir_const_value dest[NIR_MAX_VEC_COMPONENTS] = {};
   uint16_t poison = 0;
   nir_eval_const_opcode(alu->op, dest, &poison, 1, alu->def.bit_size, srcs, 0);
   return (uint32_t)nir_const_value_as_uint(dest[0], alu->def.bit_size);
}

} // namespace

/* 16-entry loop-invariant byte LUT select tree packs into 4 words in preheader
 * and evaluates identically for every index 0..15. */
TEST(FeatureMatchingOpts, PackByteLutSelect16Exact)
{
   nir_builder b = make_compute("pack-byte-lut-16");
   nir_def *leaves[16];
   for (unsigned i = 0; i < 16; ++i)
      leaves[i] = load_global_ex(&b, 1, 8, buffer_addr(&b, nir_imm_int(&b, i)), 1);

   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);

   nir_def *idx_def = NULL;
   nir_def *lut_user = NULL;
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_uge_imm(&b, i, 24));
      nir_def *idx = nir_iand_imm(&b, i, 15);
      idx_def = idx;
      nir_def *v = build_byte_lut_tree(&b, idx, leaves, 0, 15);
      nir_def *sum = nir_iadd(&b, nir_load_var(&b, acc), v);
      lut_user = sum;
      nir_store_var(&b, acc, sum, 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   store_global_ex(&b, nir_load_var(&b, acc),
                   buffer_addr(&b, nir_imm_int(&b, 64)), 4);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);
   ASSERT_EQ(count_alu_op(b.shader, nir_op_bcsel), 15u);
   /* The loop position i (its phi, past the copy nir_lower_vars_to_ssa
    * leaves), from which idx = i & 15. */
   nir_def *i_def =
      nir_scalar_chase_movs(nir_get_scalar(nir_def_as_alu(idx_def)->src[0].src.ssa, 0)).def;

   EXPECT_TRUE(agx_nir_opt_pack_byte_lut_select(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "pack-byte-lut-16");

   /* 15 selects collapse to 3 word selects inside the loop. */
   EXPECT_EQ(count_alu_op(b.shader, nir_op_bcsel), 3u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_pack_32_2x16_split), 4u);

   /* Exhaustively check every loop position (the index is i & 15, so 16..23
    * wrap) against distinct byte values, evaluating from i itself: the packed
    * read may take its byte lane straight from i. */
   const uint8_t sample[16] = {
      0x13, 0xa7, 0x00, 0xff, 0x42, 0x89, 0x5c, 0xde,
      0x01, 0x7f, 0x80, 0x3b, 0xc4, 0x61, 0x9e, 0x28,
   };
   nir_alu_instr *sum_alu = nir_def_as_alu(lut_user);
   nir_scalar val_s = nir_scalar_resolved(sum_alu->src[1].src.ssa, 0);
   for (unsigned k = 0; k < 24; ++k) {
      EXPECT_EQ(eval_lut_expr(val_s, sample, leaves, i_def, k),
                sample[k & 15])
         << "position " << k;
   }

   /* One bitfield extract reads the byte: no variable shift and mask. */
   EXPECT_EQ(count_alu_in_loops(b.shader, nir_op_ubitfield_extract), 1u);
   EXPECT_EQ(count_alu_in_loops(b.shader, nir_op_ushr), 0u);
   ralloc_free(b.shader);
}

/* Unbounded index must NOT be packed (out-of-range index would read wrong byte). */
TEST(FeatureMatchingOpts, PackByteLutRefusesUnboundedIndex)
{
   nir_builder b = make_compute("pack-byte-lut-unbounded");
   nir_def *leaves[16];
   for (unsigned i = 0; i < 16; ++i)
      leaves[i] = load_global_ex(&b, 1, 8, buffer_addr(&b, nir_imm_int(&b, i)), 1);

   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_uge_imm(&b, i, 24));
      nir_def *v = build_byte_lut_tree(&b, i, leaves, 0, 15);
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   store_global_ex(&b, nir_load_var(&b, acc),
                   buffer_addr(&b, nir_imm_int(&b, 64)), 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);

   EXPECT_FALSE(agx_nir_opt_pack_byte_lut_select(b.shader));
   ralloc_free(b.shader);
}

/* Exact-bound specialization flattens inner descriptor loop and hoists
 * outer-invariant desc1 loads and loop-header count2 loads out of outer loop. */
TEST(FeatureMatchingOpts, SpecializeExactNestedAndHoistReadonlyLoads)
{
   nir_builder b = make_compute("knn-specialize-hoist");
   nir_def *stride = load_push_u32(&b, 0);
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *base1 = nir_imul(&b, gid, stride);

   nir_variable *jv = nir_local_variable_create(b.impl, glsl_uint_type(), "j");
   nir_variable *best = nir_local_variable_create(b.impl, glsl_uint_type(), "best");
   nir_store_var(&b, jv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, best, nir_imm_int(&b, 0xffffffff), 1);

   nir_loop *outer = nir_push_loop(&b);
   {
      nir_def *j = nir_load_var(&b, jv);
      nir_def *count2 =
         load_global_ex(&b, 1, 32, buffer_addr(&b, nir_imm_int(&b, 0), 4), 4);
      nir_break_if(&b, nir_ige(&b, j, count2));

      nir_def *base2 = nir_imul(&b, j, stride);
      nir_variable *kv = nir_local_variable_create(b.impl, glsl_uint_type(), "k");
      nir_variable *dist = nir_local_variable_create(b.impl, glsl_uint_type(), "d");
      nir_store_var(&b, kv, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, dist, nir_imm_int(&b, 0), 1);

      nir_loop *inner = nir_push_loop(&b);
      {
         nir_def *k = nir_load_var(&b, kv);
         nir_break_if(&b, nir_ige(&b, k, stride));
         nir_def *d1 = load_global_ex(
            &b, 1, 32, buffer_addr(&b, nir_ishl_imm(&b, nir_iadd(&b, base1, k), 2), 8),
            4);
         nir_def *d2 = load_global_ex(
            &b, 1, 32, buffer_addr(&b, nir_ishl_imm(&b, nir_iadd(&b, base2, k), 2), 12),
            4);
         nir_def *x = nir_bit_count(&b, nir_ixor(&b, d1, d2));
         nir_store_var(&b, dist, nir_iadd(&b, nir_load_var(&b, dist), x), 1);
         nir_store_var(&b, kv, nir_iadd_imm(&b, k, 1), 1);
      }
      nir_pop_loop(&b, inner);

      nir_store_var(&b, best, nir_umin(&b, nir_load_var(&b, best), nir_load_var(&b, dist)), 1);
      nir_store_var(&b, jv, nir_iadd_imm(&b, j, 1), 1);
   }
   nir_pop_loop(&b, outer);
   store_global_ex(&b, nir_load_var(&b, best),
                   buffer_addr(&b, nir_ishl_imm(&b, gid, 2), 16), 4);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(_, b.shader, nir_opt_dce);

   unsigned widest = 0;
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 320, &widest));
   EXPECT_EQ(widest, 8u);
   for (unsigned iter = 0; iter < 3; ++iter) {
      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_remove_phis);
      NIR_PASS(_, b.shader, nir_opt_dce);
      NIR_PASS(_, b.shader, nir_opt_dead_cf);
      NIR_PASS(_, b.shader, nir_opt_cse);
      NIR_PASS(_, b.shader, nir_opt_algebraic);
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      NIR_PASS(_, b.shader, nir_opt_loop_unroll);
   }
   nir_validate_shader(b.shader, "after exact specialize + unroll");

   /* Fallback else-arm inner loop must NOT be runtime-unrolled (at most 2
    * scalar loads inside the fallback inner loop, never 16). */
   unsigned fallback_loads = 0;
   foreach_list_typed(nir_cf_node, node, node, &b.impl->body) {
      if (node->type != nir_cf_node_if)
         continue;
      nir_if *nif = nir_cf_node_as_if(node);
      foreach_list_typed(nir_cf_node, en, node, &nif->else_list) {
         nir_foreach_block_in_cf_node(block, en) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic ==
                      nir_intrinsic_load_global)
                  fallback_loads++;
            }
         }
      }
   }
   EXPECT_LE(fallback_loads, 3u);

   EXPECT_TRUE(agx_nir_merge_global_words(b.shader));
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(_, b.shader, nir_opt_dce);
   EXPECT_TRUE(agx_nir_hoist_readonly_loop_loads(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "after hoist readonly loop loads");

   /* Count hoisted loads outside loops: count2 + 2x u32x4 desc1 = >= 3 loads
    * outside loops (plus the fallback arm's hoisted count2). */
   unsigned outside_loads = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         bool in_loop = false;
         for (nir_cf_node *p = block->cf_node.parent; p; p = p->parent) {
            if (p->type == nir_cf_node_loop) {
               in_loop = true;
               break;
            }
         }
         if (in_loop)
            continue;
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_global)
               outside_loads++;
         }
      }
   }
   EXPECT_GE(outside_loads, 3u);
   ralloc_free(b.shader);
}

/* Loops with a memory store or volatile/coherent load must NOT hoist loads. */
TEST(FeatureMatchingOpts, HoistReadonlyLoopLoadsRefusesWritesAndVolatile)
{
   for (unsigned mode = 0; mode < 2; ++mode) {
      nir_builder b = make_compute("hoist-refuse");
      nir_def *n = load_push_u32(&b, 0);
      nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
      nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);

      nir_loop *loop = nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, iv);
         nir_break_if(&b, nir_ige(&b, i, n));
         gl_access_qualifier access =
            mode == 1 ? ACCESS_COHERENT : (gl_access_qualifier)0;
         nir_def *v = load_global_ex(
            &b, 1, 32, buffer_addr(&b, nir_imm_int(&b, 0)), 4, access);
         if (mode == 0)
            store_global_ex(&b, i, buffer_addr(&b, nir_ishl_imm(&b, i, 2)), 4);
         nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
         nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
      }
      nir_pop_loop(&b, loop);
      store_global_ex(&b, nir_load_var(&b, acc),
                      buffer_addr(&b, nir_imm_int(&b, 64)), 4);
      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);

      EXPECT_FALSE(agx_nir_hoist_readonly_loop_loads(b.shader)) << "mode " << mode;
      EXPECT_EQ(count_in_loops(b.shader, nir_intrinsic_load_global), 1u);
      ralloc_free(b.shader);
   }
}

/* Direct-body pure ALU (such as lowered trig inside a top-tested loop) hoists
 * cleanly while respecting occupancy budgets and refusing integer division. */
TEST(FeatureMatchingOpts, LicmDirectBodyPureAlu)
{
   nir_builder b = make_compute("licm-direct-body");
   nir_def *n = load_push_u32(&b, 0);
   nir_def *angle = load_push_u32(&b, 4);
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_float(&b, 0.0f), 1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_ige(&b, i, n));
      nir_def *s = nir_fsinc_agx3(&b, angle);
      nir_def *v = nir_fmul(&b, s, nir_u2f32(&b, i));
      nir_store_var(&b, acc, nir_fadd(&b, nir_load_var(&b, acc), v), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   store_global_ex(&b, nir_load_var(&b, acc),
                   buffer_addr(&b, nir_imm_int(&b, 0)), 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);
   ASSERT_EQ(count_alu_in_loops(b.shader, nir_op_fsinc_agx3), 1u);

   EXPECT_TRUE(agx_nir_opt_licm(b.shader, 64, false));
   nir_validate_shader(b.shader, "licm-direct-body");
   EXPECT_EQ(count_alu_in_loops(b.shader, nir_op_fsinc_agx3), 0u);
   ralloc_free(b.shader);
}

/* Direct-body integer division must NOT be speculatively hoisted out of a
 * top-tested loop (could trap on zero divisor when loop runs 0 iterations). */
TEST(FeatureMatchingOpts, LicmDirectBodyRefusesIdiv)
{
   nir_builder b = make_compute("licm-refuse-idiv");
   nir_def *n = load_push_u32(&b, 0);
   nir_def *num = load_push_u32(&b, 4);
   nir_def *den = load_push_u32(&b, 8);
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_ige(&b, i, n));
      nir_def *q = nir_idiv(&b, num, den);
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), q), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   store_global_ex(&b, nir_load_var(&b, acc),
                   buffer_addr(&b, nir_imm_int(&b, 0)), 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);

   agx_nir_opt_licm(b.shader, 64, false);
   EXPECT_EQ(count_alu_in_loops(b.shader, nir_op_idiv), 1u);
   ralloc_free(b.shader);
}

/* AGX3C compute uniform-address atomic compiles cleanly with subgroup
 * aggregation enabled. */
TEST(FeatureMatchingOpts, Agx3cComputeUniformAtomicCompiles)
{
   nir_builder b = make_compute("agx3c-uniform-atomic");
   nir_def *old = nir_global_atomic(&b, 32, nir_imm_int64(&b, 0x1000),
                                    nir_imm_int(&b, 1));
   nir_intrinsic_set_atomic_op(nir_def_as_intrinsic(old), nir_atomic_op_iadd);
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   store_global_ex(&b, old,
                   nir_iadd(&b, nir_imm_int64(&b, 0x2000),
                            nir_u2u64(&b, nir_ishl_imm(&b, gid, 2))),
                   4);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   struct agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   EXPECT_NE(part.binary, nullptr);
   EXPECT_GT(part.info.main_size, 0u);
   free(part.binary);
   ralloc_free(b.shader);
}

/* specialize_exact_nested() clones and folds pre-existing constant-derived
 * predicates of `bound` (such as `0 < bound` guarding the inner loop) inside
 * the fast arm so dead-CF removes the guard and exposes outer-invariant loads
 * directly in the outer loop body for agx_nir_hoist_readonly_loop_loads(). */
TEST(FeatureMatchingOpts, SpecializeExactNestedFoldsOuterBoundPredicate)
{
   nir_builder b = make_compute("spec-exact-outer-pred");
   nir_def *n_outer = load_push_u32(&b, 0);
   nir_def *stride = load_push_u32(&b, 4);
   nir_def *has_stride = nir_ilt(&b, nir_imm_int(&b, 0), stride);
   nir_def *tid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *d1_base = nir_imul(&b, tid, stride);

   nir_variable *j_var = nir_local_variable_create(b.impl, glsl_uint_type(), "j");
   nir_variable *best_var = nir_local_variable_create(b.impl, glsl_uint_type(), "best");
   nir_store_var(&b, j_var, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, best_var, nir_imm_int(&b, 9999), 1);

   nir_loop *outer = nir_push_loop(&b);
   {
      nir_def *j = nir_load_var(&b, j_var);
      nir_break_if(&b, nir_ige(&b, j, n_outer));

      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

      /* Guarded by pre-computed `0 < stride` defined outside `outer`. */
      nir_push_if(&b, has_stride);
      {
         nir_def *d2_base = nir_imul(&b, j, stride);
         nir_variable *k_var = nir_local_variable_create(b.impl, glsl_uint_type(), "k");
         nir_store_var(&b, k_var, nir_imm_int(&b, 0), 1);
         nir_loop *inner = nir_push_loop(&b);
         {
            nir_def *k = nir_load_var(&b, k_var);
            nir_break_if(&b, nir_ige(&b, k, stride));
            nir_def *a1 = buffer_addr(&b, nir_ishl_imm(&b, nir_iadd(&b, d1_base, k), 2), 8);
            nir_def *a2 = buffer_addr(&b, nir_ishl_imm(&b, nir_iadd(&b, d2_base, k), 2), 12);
            nir_def *v1 = load_global_ex(&b, 1, 32, a1, 4);
            nir_def *v2 = load_global_ex(&b, 1, 32, a2, 4);
            nir_def *pop = nir_bit_count(&b, nir_ixor(&b, v1, v2));
            nir_store_var(&b, acc_var, nir_iadd(&b, nir_load_var(&b, acc_var), pop), 1);
            nir_store_var(&b, k_var, nir_iadd_imm(&b, k, 1), 1);
         }
         nir_pop_loop(&b, inner);
      }
      nir_pop_if(&b, NULL);

      nir_store_var(&b, best_var,
                    nir_imin(&b, nir_load_var(&b, best_var), nir_load_var(&b, acc_var)), 1);
      nir_store_var(&b, j_var, nir_iadd_imm(&b, j, 1), 1);
   }
   nir_pop_loop(&b, outer);
   store_global_ex(&b, nir_load_var(&b, best_var),
                   buffer_addr(&b, nir_ishl_imm(&b, tid, 2), 16), 4);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(_, b.shader, nir_opt_dce);

   unsigned widest = 0;
   ASSERT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 320, &widest));
   EXPECT_EQ(widest, 8u);
   for (unsigned i = 0; i < 4; ++i) {
      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_remove_phis);
      NIR_PASS(_, b.shader, nir_opt_dce);
      NIR_PASS(_, b.shader, nir_opt_dead_cf);
      NIR_PASS(_, b.shader, nir_opt_cse);
      NIR_PASS(_, b.shader, nir_opt_algebraic);
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      NIR_PASS(_, b.shader, nir_opt_loop_unroll);
   }
   EXPECT_TRUE(agx_nir_merge_global_words(b.shader));
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(_, b.shader, nir_opt_dce);
   EXPECT_TRUE(agx_nir_hoist_readonly_loop_loads(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "spec-exact-outer-pred");

   /* Both u32x4 loads of desc1 must now be hoisted outside the specialized
    * outer loop (not trapped inside `if (0 < stride)`). */
   unsigned outside_vec4 = 0;
   nir_foreach_block(block, b.impl) {
      bool in_loop = false;
      for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent) {
         if (n->type == nir_cf_node_loop)
            in_loop = true;
      }
      if (in_loop)
         continue;
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if ((intr->intrinsic == nir_intrinsic_load_global ||
                 intr->intrinsic == nir_intrinsic_load_global_constant) &&
                intr->def.num_components == 4)
               outside_vec4++;
         }
      }
   }
   EXPECT_GE(outside_vec4, 2u);
   ralloc_free(b.shader);
}

/* Guard-chain coalescing wraps adjacent multi-block divergent border ifs with
 * pure condition-prep ops between them under one combined fast-path guard. */
TEST(FeatureMatchingOpts, GuardChainsCoalesceMultiBlockWithMidCondPrep)
{
   nir_builder b = make_compute("guard-chains-mid-prep");
   nir_def *tid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *w = load_push_u32(&b, 0);
   nir_def *border = load_push_u32(&b, 4);
   nir_def *val = nir_u2u8(&b, tid);

   nir_def *c0 = nir_ult(&b, tid, border);
   nir_push_if(&b, c0);
   {
      store_global_ex(&b, val, buffer_addr(&b, tid), 1);
      nir_push_if(&b, nir_ine_imm(&b, tid, 0));
      {
         store_global_ex(&b, val, buffer_addr(&b, nir_iadd_imm(&b, tid, 1)), 1);
      }
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);

   /* Pure condition-prep ALU in the intermediate block between the two ifs. */
   nir_def *hi_start = nir_isub(&b, w, border);
   nir_def *c1 = nir_uge(&b, tid, hi_start);
   nir_push_if(&b, c1);
   {
      store_global_ex(&b, val, buffer_addr(&b, nir_iadd_imm(&b, tid, 2)), 1);
   }
   nir_pop_if(&b, NULL);

   EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
   nir_validate_shader(b.shader, "guard-chains-mid-prep");

   /* Top-level CF list should now contain a single combined outer `if`. */
   unsigned top_ifs = 0;
   foreach_list_typed(nir_cf_node, node, node, &b.impl->body) {
      if (node->type == nir_cf_node_if)
         top_ifs++;
   }
   EXPECT_EQ(top_ifs, 1u);
   ralloc_free(b.shader);
}

/* Same-block straight-line store_agx -> load_agx forwarding eliminates a
 * redundant global reload of the identical address. */
TEST(FeatureMatchingOpts, ForwardMergeLoadsSameBlockStraightLine)
{
   nir_builder b = make_compute("forward-straight-line");
   nir_def *base = load_preamble_base(&b, 1, 64, 0);
   nir_def *idx = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *val = nir_u2u8(&b, idx);

   store_agx_r8(&b, val, base, idx);
   nir_def *reloaded = load_agx_r8(&b, base, idx);
   store_agx_r8(&b, reloaded, base, nir_iadd_imm(&b, idx, 1));

   EXPECT_TRUE(agx_nir_opt_forward_merge_loads(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "forward-straight-line");

   unsigned loads = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_agx)
            loads++;
      }
   }
   EXPECT_EQ(loads, 0u);
   ralloc_free(b.shader);
}

/* SFU-chain-only direct-body LICM hoists the invariant trig chain without
 * hoisting unrelated integer loop-invariant ALU. */
TEST(FeatureMatchingOpts, LicmDirectSfuChainKeepsUnrelatedIntAlu)
{
   nir_builder b = make_compute("licm-sfu-chain-only");
   nir_def *n = load_push_u32(&b, 0);
   nir_def *angle = load_push_u32(&b, 4);
   nir_def *stride = load_push_u32(&b, 8);
   nir_def *tid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_float(&b, 0.0f), 1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_ige(&b, i, n));
      nir_def *rad = nir_fmul_imm(&b, angle, 0.017453292f);
      nir_def *s = nir_fmul(&b, nir_fsinc_agx3(&b, rad), rad);
      /* Unrelated loop-invariant integer math that should stay unhoisted. */
      nir_def *row_off = nir_imul(&b, tid, stride);
      nir_def *addr = buffer_addr(&b, nir_ishl_imm(&b, nir_iadd(&b, row_off, i), 2));
      nir_def *w = load_global_ex(&b, 1, 32, addr, 4);
      nir_store_var(&b, acc,
                    nir_fadd(&b, nir_load_var(&b, acc), nir_fmul(&b, s, w)), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   store_global_ex(&b, nir_load_var(&b, acc),
                   buffer_addr(&b, nir_imm_int(&b, 0)), 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);

   EXPECT_TRUE(agx_nir_opt_licm_direct_sfu(b.shader, 56));
   nir_validate_shader(b.shader, "licm-sfu-chain-only");
   EXPECT_EQ(count_alu_in_loops(b.shader, nir_op_fsinc_agx3), 0u);
   EXPECT_EQ(count_alu_in_loops(b.shader, nir_op_imul), 1u);
   ralloc_free(b.shader);
}

/* Loop-varying SFU ops must NOT trigger speculative SFU-chain LICM or pull
 * unrelated loop constants out into registers. */
TEST(FeatureMatchingOpts, LicmDirectSfuIgnoresLoopVaryingSfu)
{
   nir_builder b = make_compute("licm-sfu-varying");
   nir_def *n = load_push_u32(&b, 0);
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_float(&b, 0.0f), 1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_ige(&b, i, n));
      nir_def *v = nir_frcp(&b, nir_fadd_imm(&b, nir_u2f32(&b, i), 1.0f));
      nir_store_var(&b, acc, nir_fadd(&b, nir_load_var(&b, acc), v), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   store_global_ex(&b, nir_load_var(&b, acc),
                   buffer_addr(&b, nir_imm_int(&b, 0)), 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);

   EXPECT_FALSE(agx_nir_opt_licm_direct_sfu(b.shader, 56));
   EXPECT_EQ(count_alu_in_loops(b.shader, nir_op_frcp), 1u);
   ralloc_free(b.shader);
}

/* Uniform 2-arm if inside nested loops unswitches out of the outer loop and
 * exposes loop-invariant arithmetic inside the surviving arm. */
TEST(FeatureMatchingOpts, UnswitchUniformIfOutOfNestedLoops)
{
   nir_builder b = make_compute("unswitch-uniform-if");
   nir_def *h = load_push_u32(&b, 4);
   nir_def *w = load_push_u32(&b, 8);
   nir_def *scale = load_push_u32(&b, 12);
   nir_def *gid = nir_load_global_invocation_id(&b, 32);

   nir_variable *yv = nir_local_variable_create(b.impl, glsl_uint_type(), "y");
   nir_store_var(&b, yv, nir_channel(&b, gid, 1), 1);
   nir_loop *outer = nir_push_loop(&b);
   {
      nir_def *y = nir_load_var(&b, yv);
      nir_break_if(&b, nir_uge(&b, y, h));
      nir_variable *xv = nir_local_variable_create(b.impl, glsl_uint_type(), "x");
      nir_store_var(&b, xv, nir_channel(&b, gid, 0), 1);
      nir_loop *inner = nir_push_loop(&b);
      {
         nir_def *x = nir_load_var(&b, xv);
         nir_break_if(&b, nir_uge(&b, x, w));
         /* Compute uniform mode condition inside inner loop to exercise
          * rematerialization before outer-loop unswitching. */
         nir_def *mode = nir_ine_imm(&b, load_push_u32(&b, 0), 0);
         nir_if *nif = nir_push_if(&b, mode);
         {
            nir_def *yf = nir_fmul(&b, nir_u2f32(&b, y), scale);
            nir_def *xf = nir_fmul(&b, nir_u2f32(&b, x), scale);
            nir_def *v = nir_f2u32(&b, nir_fadd(&b, yf, xf));
            store_global_ex(
               &b, v,
               buffer_addr(&b, nir_ishl_imm(&b, nir_iadd(&b, nir_imul(&b, y, w), x), 2), 4),
               4);
         }
         nir_push_else(&b, nif);
         {
            nir_def *idx = nir_iadd(&b, nir_imul(&b, y, w), x);
            nir_def *v = load_global_ex(
               &b, 1, 32, buffer_addr(&b, nir_ishl_imm(&b, idx, 2), 8), 4);
            store_global_ex(&b, v,
                            buffer_addr(&b, nir_ishl_imm(&b, idx, 2), 4), 4);
         }
         nir_pop_if(&b, nif);
         nir_store_var(&b, xv, nir_iadd_imm(&b, x, 32), 1);
      }
      nir_pop_loop(&b, inner);
      nir_store_var(&b, yv, nir_iadd_imm(&b, y, 32), 1);
   }
   nir_pop_loop(&b, outer);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);

   EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 320, NULL));
   NIR_PASS(_, b.shader, nir_opt_dead_cf);
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "unswitch-uniform-if");

   /* The top-level body should now begin with the unswitched `if (mode)`. */
   unsigned top_ifs = 0;
   foreach_list_typed(nir_cf_node, node, node, &b.impl->body) {
      if (node->type == nir_cf_node_if)
         top_ifs++;
   }
   EXPECT_EQ(top_ifs, 1u);
   ralloc_free(b.shader);
}

/* Store-to-load forwarding propagates an unclobbered trailing store_agx into
 * the prefix of an immediately following `if` arm. */
TEST(FeatureMatchingOpts, ForwardMergeLoadsIntoFollowingIfArm)
{
   nir_builder b = make_compute("forward-into-if");
   nir_def *base = load_preamble_base(&b, 1, 64, 0);
   nir_def *pvec = load_preamble_base(&b, 4, 32, 58);
   nir_def *border = load_preamble_base(&b, 1, 32, 64);
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0);
   nir_def *y = nir_channel(&b, gid, 1);
   nir_def *val = nir_u2u8(&b, x);

   nir_def *row_s = nir_imadshl_agx(&b, nir_iadd(&b, y, nir_channel(&b, pvec, 3)),
                                    nir_channel(&b, pvec, 2),
                                    nir_channel(&b, pvec, 3), nir_imm_int(&b, 0));
   nir_def *idx_store = nir_iadd(&b, x, row_s);
   store_agx_r8(&b, val, base, idx_store);

   nir_def *idx_load = nir_iadd(&b, nir_iadd(&b, x, border),
                                nir_imul(&b, nir_iadd(&b, y, border),
                                         nir_channel(&b, pvec, 2)));
   nir_if *nif = nir_push_if(&b, nir_ult_imm(&b, x, 4));
   {
      nir_def *reloaded = load_agx_r8(&b, base, idx_load);
      store_agx_r8(&b, reloaded, base, nir_iadd_imm(&b, idx_load, 1));
   }
   nir_pop_if(&b, nif);

   EXPECT_TRUE(agx_nir_opt_forward_merge_loads(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "forward-into-if");

   unsigned loads = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_agx)
            loads++;
      }
   }
   EXPECT_EQ(loads, 0u);
   ralloc_free(b.shader);
}

/* Narrowed 8-bit LUT trees consumed directly by u2u32 bypass u2u8->u2u32. */
TEST(FeatureMatchingOpts, PackByteLutBypassesU2u32RoundTrip)
{
   nir_builder b = make_compute("pack-byte-lut-u8");
   nir_def *leaves[8];
   for (unsigned i = 0; i < 8; ++i)
      leaves[i] = load_global_ex(&b, 1, 8, buffer_addr(&b, nir_imm_int(&b, i)), 1);

   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_uge_imm(&b, i, 16));
      nir_def *idx = nir_iand_imm(&b, i, 7);
      nir_def *s01 = nir_bcsel(&b, nir_ult_imm(&b, idx, 1), leaves[0], leaves[1]);
      nir_def *s23 = nir_bcsel(&b, nir_ult_imm(&b, idx, 3), leaves[2], leaves[3]);
      nir_def *s45 = nir_bcsel(&b, nir_ult_imm(&b, idx, 5), leaves[4], leaves[5]);
      nir_def *s67 = nir_bcsel(&b, nir_ult_imm(&b, idx, 7), leaves[6], leaves[7]);
      nir_def *s03 = nir_bcsel(&b, nir_ult_imm(&b, idx, 2), s01, s23);
      nir_def *s47 = nir_bcsel(&b, nir_ult_imm(&b, idx, 6), s45, s67);
      nir_def *v8 = nir_bcsel(&b, nir_ult_imm(&b, idx, 4), s03, s47);
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), nir_u2u32(&b, v8)), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   store_global_ex(&b, nir_load_var(&b, acc),
                   buffer_addr(&b, nir_imm_int(&b, 64)), 4);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);
   EXPECT_TRUE(agx_nir_opt_pack_byte_lut_select(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "pack-byte-lut-u8");

   EXPECT_EQ(count_alu_in_loops(b.shader, nir_op_u2u8), 0u);
   EXPECT_EQ(count_alu_in_loops(b.shader, nir_op_u2u32), 0u);
   ralloc_free(b.shader);
}

/* Backend MOV_IMM sinking moves a preheader immediate into its single outer-loop
 * consumer block when it would otherwise stay live across an enclosed inner loop. */
TEST(FeatureMatchingOpts, SinkImmediatesAcrossInnerLoop)
{
   void *memctx = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(memctx);
   agx_context *ctx = b->shader;
   agx_block *b0 = agx_start_block(ctx);
   agx_block *b_outer = agx_test_block(ctx);
   agx_block *b_inner = agx_test_block(ctx);
   agx_block *b_after = agx_test_block(ctx);
   agx_block *b_exit = agx_test_block(ctx);

   b_outer->loop_header = true;
   b_inner->loop_header = true;
   agx_block_add_successor(b0, b_outer);
   agx_block_add_successor(b_outer, b_inner);
   agx_block_add_successor(b_inner, b_inner);
   agx_block_add_successor(b_inner, b_after);
   agx_block_add_successor(b_after, b_outer);
   agx_block_add_successor(b_after, b_exit);

   b->cursor = agx_after_block(b0);
   agx_index c = agx_mov_imm(b, 32, 0xffffffffu);
   agx_index x = agx_mov_imm(b, 32, 1);

   b->cursor = agx_after_block(b_inner);
   agx_index y = agx_iadd(b, x, agx_immediate(1), 0);
   (void)y;

   b->cursor = agx_after_block(b_after);
   agx_index z = agx_and(b, x, c);
   (void)z;

   b->cursor = agx_after_block(b_exit);
   agx_stop(b);

   EXPECT_TRUE(agx_opt_sink_immediates(ctx));

   bool found_in_after = false;
   agx_foreach_instr_in_block(b_after, I) {
      if (I->op == AGX_OPCODE_MOV_IMM && I->imm == 0xffffffffu)
         found_in_after = true;
   }
   EXPECT_TRUE(found_in_after);
   ralloc_free(memctx);
}

/* Pre-unroll direct-body constant-load LICM hoists speculatable bounded UBO
 * loads and their descriptor/bounds chains out of loops while leaving
 * unrelated loop-invariant integer math unhoisted. */
TEST(FeatureMatchingOpts, LicmDirectConstLoadsHoistsSpeculatableBoundedUbo)
{
   nir_builder b = make_compute("licm-const-load-chain");
   nir_def *n = load_push_u32(&b, 0);
   nir_def *stride = load_push_u32(&b, 4);
   nir_def *root = load_preamble_base(&b, 1, 64, 0);
   nir_def *tid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_float(&b, 0.0f), 1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_ige(&b, i, n));

      nir_def *desc = nir_load_global_constant(
         &b, 4, 32, nir_iadd_imm(&b, root, 64),
         .access = (gl_access_qualifier)(ACCESS_CAN_SPECULATE |
                                         ACCESS_BUFFER_DESCRIPTOR_AGX),
         .align_mul = 16);
      nir_def *bound = nir_usub_sat(&b, nir_channel(&b, desc, 2), nir_imm_int(&b, 3));
      nir_def *valid = nir_ult(&b, nir_imm_int(&b, 4), bound);
      nir_def *base = nir_bcsel(
         &b, valid,
         nir_pack_64_2x32_split(&b, nir_channel(&b, desc, 0), nir_channel(&b, desc, 1)),
         nir_imm_int64(&b, AGX_ZERO_PAGE_ADDRESS));
      nir_def *k = nir_load_global_constant(
         &b, 1, 32, nir_iadd_imm(&b, base, 4),
         .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE |
                                         ACCESS_CAN_REORDER |
                                         ACCESS_CAN_SPECULATE),
         .align_mul = 4);

      /* Unrelated loop-invariant integer math stays inside the loop. */
      nir_def *row_off = nir_imul(&b, tid, stride);
      nir_def *addr = buffer_addr(&b, nir_ishl_imm(&b, nir_iadd(&b, row_off, i), 2));
      nir_def *w = load_global_ex(&b, 1, 32, addr, 4);
      nir_store_var(&b, acc,
                    nir_fadd(&b, nir_load_var(&b, acc), nir_fmul(&b, k, w)), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   store_global_ex(&b, nir_load_var(&b, acc),
                   buffer_addr(&b, nir_imm_int(&b, 0)), 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);

   EXPECT_TRUE(agx_nir_opt_licm_direct_const_loads(b.shader, 56));
   nir_validate_shader(b.shader, "licm-const-load-chain");
   EXPECT_EQ(count_alu_in_loops(b.shader, nir_op_usub_sat), 0u);
   EXPECT_EQ(count_alu_in_loops(b.shader, nir_op_imul), 1u);
   ralloc_free(b.shader);
}

/* Non-speculatable loads and descriptor-only loads (feeding a non-speculatable
 * read) must NOT trigger direct constant-load LICM. */
TEST(FeatureMatchingOpts, LicmDirectConstLoadsRejectsUnspeculatable)
{
   nir_builder b = make_compute("licm-const-load-reject");
   nir_def *n = load_push_u32(&b, 0);
   nir_def *root = load_preamble_base(&b, 1, 64, 0);
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_float_type(), "acc");
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_float(&b, 0.0f), 1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_ige(&b, i, n));
      nir_def *desc = nir_load_global_constant(
         &b, 4, 32, nir_iadd_imm(&b, root, 64),
         .access = (gl_access_qualifier)(ACCESS_CAN_SPECULATE |
                                         ACCESS_BUFFER_DESCRIPTOR_AGX),
         .align_mul = 16);
      nir_def *base =
         nir_pack_64_2x32_split(&b, nir_channel(&b, desc, 0), nir_channel(&b, desc, 1));
      nir_def *k = nir_load_global_constant(
         &b, 1, 32, nir_iadd_imm(&b, base, 4),
         .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
         .align_mul = 4);
      nir_store_var(&b, acc, nir_fadd(&b, nir_load_var(&b, acc), k), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   store_global_ex(&b, nir_load_var(&b, acc),
                   buffer_addr(&b, nir_imm_int(&b, 0)), 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);

   EXPECT_FALSE(agx_nir_opt_licm_direct_const_loads(b.shader, 56));
   ralloc_free(b.shader);
}

/* G17P bounded dynamic-UBO loads (bcsel onto AGX_ZERO_PAGE_ADDRESS) promote to
 * AGX3_UNIFORM_DMA_BUFFER in compute and vertex stages; mismatched bounds stay
 * in the software preamble. */
TEST(FeatureMatchingOpts, BoundedDynamicUboPromotesToUniformDma)
{
   auto make_bounded_read = [](nir_builder *b, nir_def *desc, uint32_t offset_B,
                               uint32_t sub_val) {
      nir_def *adj = nir_usub_sat(b, nir_channel(b, desc, 2), nir_imm_int(b, sub_val));
      nir_def *valid = nir_ult(b, nir_imm_int(b, offset_B), adj);
      nir_def *base = nir_bcsel(
         b, valid,
         nir_pack_64_2x32_split(b, nir_channel(b, desc, 0), nir_channel(b, desc, 1)),
         nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
      return nir_load_global_constant(
         b, 1, 32, nir_iadd_imm(b, base, offset_B),
         .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE |
                                         ACCESS_CAN_REORDER |
                                         ACCESS_CAN_SPECULATE),
         .align_mul = 4);
   };

   /* 1. Vertex shader with bounded dynamic UBO load + 1 cheap ALU op drops its
    * entire software preamble once uniform DMA replaces the bounded read. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "agx3-bounded-ubo-vs");
      b.shader->info.io_lowered = true;
      b.shader->info.outputs_written = VARYING_BIT_POS;

      nir_def *root = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *desc = nir_load_global_constant(
         &b, 4, 32, nir_iadd_imm(&b, root, 64),
         .access = (gl_access_qualifier)(ACCESS_CAN_SPECULATE |
                                         ACCESS_BUFFER_DESCRIPTOR_AGX),
         .align_mul = 16);
      nir_def *v0 = make_bounded_read(&b, desc, 0, 3);
      nir_def *v1 = make_bounded_read(&b, desc, 4, 3);
      nir_def *vid = nir_load_vertex_id(&b);
      nir_def *x = nir_fadd(&b, nir_u2f32(&b, nir_iadd(&b, vid, nir_ineg(&b, v0))), v1);
      nir_def *pos[4] = {x, v1, nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f)};
      for (unsigned c = 0; c < 4; ++c)
         nir_store_uvs_agx(&b, pos[c], nir_imm_intN_t(&b, c, 16));
      agx_preprocess_nir(b.shader);

      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.reserved_preamble = 4;
      key.agx3_native_cprog = true;
      key.g17p_native_uniforms = true;
      key.g17p_uniform_dma = true;
      key.g17p_dma_root = 0;
      key.g17p_dma_buffers_B = 64;
      key.g17p_dma_buffer_count = 4;

      struct agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_FALSE(part.info.has_preamble);
      ASSERT_EQ(part.info.agx3_uniform_dma_count, 1u);
      EXPECT_EQ(part.info.agx3_uniform_dma[0].source, AGX3_UNIFORM_DMA_BUFFER);
      EXPECT_EQ(part.info.agx3_uniform_dma[0].descriptor_B, 64u);
      EXPECT_EQ(part.info.agx3_uniform_dma[0].offset_B, 0u);
      EXPECT_EQ(part.info.agx3_uniform_dma[0].halfs, 4u);
      free(part.binary);
      ralloc_free(b.shader);
   }

   /* 2. Negative control: wrong bounds subtraction (sub=7 instead of 3 for a
    * 4-byte load) must NOT be promoted to AGX3_UNIFORM_DMA_BUFFER. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "agx3-bounded-ubo-bad-bound");
      b.shader->info.io_lowered = true;
      b.shader->info.outputs_written = VARYING_BIT_POS;

      nir_def *root = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *desc = nir_load_global_constant(
         &b, 4, 32, nir_iadd_imm(&b, root, 64),
         .access = (gl_access_qualifier)(ACCESS_CAN_SPECULATE |
                                         ACCESS_BUFFER_DESCRIPTOR_AGX),
         .align_mul = 16);
      nir_def *v = make_bounded_read(&b, desc, 4, 7);
      nir_def *vid = nir_load_vertex_id(&b);
      nir_def *x = nir_fadd(&b, nir_u2f32(&b, vid), v);
      nir_def *pos[4] = {x, v, nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f)};
      for (unsigned c = 0; c < 4; ++c)
         nir_store_uvs_agx(&b, pos[c], nir_imm_intN_t(&b, c, 16));
      agx_preprocess_nir(b.shader);

      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.reserved_preamble = 4;
      key.agx3_native_cprog = true;
      key.g17p_native_uniforms = true;
      key.g17p_uniform_dma = true;
      key.g17p_dma_root = 0;
      key.g17p_dma_buffers_B = 64;
      key.g17p_dma_buffer_count = 4;

      struct agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      bool has_buf_dma = false;
      for (unsigned i = 0; i < part.info.agx3_uniform_dma_count; ++i)
         has_buf_dma |= part.info.agx3_uniform_dma[i].source == AGX3_UNIFORM_DMA_BUFFER;
      EXPECT_FALSE(has_buf_dma);
      free(part.binary);
      ralloc_free(b.shader);
   }
}

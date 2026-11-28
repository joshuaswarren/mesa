/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "asahi/lib/agx_abi.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_compile.h"
#include "agx_nir.h"
}

namespace {

static struct agx_shader_key
g17p_key(void)
{
   struct agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G17P;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.reserved_preamble = 16;
   return key;
}

static struct agx_shader_key
agx3c_key(void)
{
   struct agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G17S;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.reserved_preamble = 16;
   return key;
}

static nir_def *
make_load_global(nir_builder *b, unsigned comps, unsigned bits, nir_def *addr,
                 unsigned align_mul, gl_access_qualifier access)
{
   nir_def *def = nir_load_global(b, comps, bits, addr);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(def);
   nir_intrinsic_set_align(intr, align_mul, 0);
   nir_intrinsic_set_access(intr, access);
   return def;
}

static void
make_store_global(nir_builder *b, nir_def *val, nir_def *addr,
                  unsigned align_mul)
{
   nir_intrinsic_instr *intr = nir_store_global(b, val, addr);
   nir_intrinsic_set_align(intr, align_mul, 0);
   nir_intrinsic_set_write_mask(intr, nir_component_mask(val->num_components));
}

/* `scalar | vec.chan`, reading the channel through the ALU source swizzle (as
 * copy propagation leaves it) rather than through a separate mov. */
static nir_def *
ior_channel(nir_builder *b, nir_def *scalar, nir_def *vec, unsigned chan)
{
   nir_alu_instr *alu = nir_alu_instr_create(b->shader, nir_op_ior);
   alu->src[0].src = nir_src_for_ssa(scalar);
   alu->src[1].src = nir_src_for_ssa(vec);
   alu->src[1].swizzle[0] = chan;
   nir_def_init(&alu->instr, &alu->def, 1, scalar->bit_size);
   nir_builder_instr_insert(b, &alu->instr);
   return &alu->def;
}

/* pack_64_2x32 of channels `lo` and `hi` of `vec`, through the source
 * swizzle. */
static nir_def *
pack_64_2x32_channels(nir_builder *b, nir_def *vec, unsigned lo, unsigned hi)
{
   nir_alu_instr *alu = nir_alu_instr_create(b->shader, nir_op_pack_64_2x32);
   alu->src[0].src = nir_src_for_ssa(vec);
   alu->src[0].swizzle[0] = lo;
   alu->src[0].swizzle[1] = hi;
   return nir_builder_alu_instr_finish_and_insert(b, alu);
}

static unsigned
count_ifs(const nir_shader *nir)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_cf_node *next = nir_cf_node_next(&block->cf_node);
         if (next && next->type == nir_cf_node_if)
            n++;
      }
   }
   return n;
}

static unsigned
count_if_depth_in_list(const struct exec_list *list, unsigned depth,
                       unsigned *max_depth, unsigned *top_level_ifs)
{
   unsigned n = 0;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         const nir_if *nif = nir_cf_node_as_if(node);
         n++;
         if (depth == 0)
            (*top_level_ifs)++;
         if (depth + 1 > *max_depth)
            *max_depth = depth + 1;
         n += count_if_depth_in_list(&nif->then_list, depth + 1, max_depth,
                                     top_level_ifs);
         n += count_if_depth_in_list(&nif->else_list, depth + 1, max_depth,
                                     top_level_ifs);
      } else if (node->type == nir_cf_node_loop) {
         const nir_loop *loop = nir_cf_node_as_loop(node);
         n += count_if_depth_in_list(&loop->body, depth, max_depth,
                                     top_level_ifs);
      }
   }
   return n;
}

static bool
uses_zero_page_const(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_load_const)
               continue;
            const nir_load_const_instr *lc = nir_instr_as_load_const(instr);
            if (lc->def.bit_size == 64 &&
                lc->value[0].u64 == AGX_ZERO_PAGE_ADDRESS)
               return true;
         }
      }
   }
   return false;
}

TEST(GuardChains, FuseNestedGuardWithDominatingLoadNoRedirect)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "fuse-dom");
   b.shader->info.workgroup_size[0] = 32;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *wid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *vel_base = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 8);

   /* Dominating unconditional load of pos[wid]. */
   nir_def *self_addr =
      nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, wid), 3));
   nir_def *self_pos =
      make_load_global(&b, 2, 32, self_addr, 8, (gl_access_qualifier)0);

   nir_def *zero = nir_imm_float(&b, 0.0f);
   nir_def *izero = nir_imm_int(&b, 0);
   nir_def *acc_in = nir_imm_int(&b, 1);

   nir_push_if(&b, nir_ine(&b, tid, wid));
   nir_def *j_addr =
      nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 3));
   nir_def *j_pos =
      make_load_global(&b, 2, 32, j_addr, 8, (gl_access_qualifier)0);
   nir_def *dx = nir_fsub(&b, nir_channel(&b, self_pos, 0),
                          nir_channel(&b, j_pos, 0));
   nir_def *cond = nir_flt(&b, nir_fmul(&b, dx, dx), nir_imm_float(&b, 4.0f));
   nir_push_if(&b, cond);
   nir_def *v_addr =
      nir_iadd(&b, vel_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 3));
   nir_def *vel =
      make_load_global(&b, 2, 32, v_addr, 8, (gl_access_qualifier)0);
   nir_def *fx = nir_fmul(&b, nir_frsq(&b, nir_fmul(&b, dx, dx)),
                          nir_channel(&b, vel, 0));
   nir_def *flag = nir_imm_int(&b, 8);
   nir_pop_if(&b, NULL);
   nir_def *inner_fx = nir_if_phi(&b, fx, zero);
   nir_def *inner_flag = nir_if_phi(&b, flag, izero);
   nir_def *ored = nir_ior(&b, acc_in, inner_flag);
   nir_pop_if(&b, NULL);

   nir_def *out_fx = nir_if_phi(&b, inner_fx, zero);
   nir_def *out_flag = nir_if_phi(&b, ored, acc_in);
   make_store_global(&b, nir_vec2(&b, out_fx, out_flag),
                     nir_iadd(&b, out_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 3)),
                     8);

   EXPECT_EQ(count_ifs(b.shader), 2u);
   EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
   nir_validate_shader(b.shader, "after fuse-dom");
   EXPECT_EQ(count_ifs(b.shader), 1u);
   /* Dominating load proved pos[tid] safe when tid == wid: no zero-page redirect. */
   EXPECT_FALSE(uses_zero_page_const(b.shader));
   ralloc_free(b.shader);
}

TEST(GuardChains, FuseNestedGuardWithoutDominatingLoadRedirectsToZeroPage)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "fuse-redir");
   b.shader->info.workgroup_size[0] = 32;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *zero = nir_imm_float(&b, 0.0f);

   nir_push_if(&b, nir_ult(&b, tid, nir_imm_int(&b, 16)));
   nir_def *addr =
      nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2));
   nir_def *val =
      make_load_global(&b, 1, 32, addr, 4, (gl_access_qualifier)0);
   nir_push_if(&b, nir_flt(&b, val, nir_imm_float(&b, 1.0f)));
   nir_def *r = nir_frsq(&b, val);
   nir_pop_if(&b, NULL);
   nir_def *inner_r = nir_if_phi(&b, r, zero);
   nir_pop_if(&b, NULL);

   nir_def *res = nir_if_phi(&b, inner_r, zero);
   make_store_global(&b, res,
                     nir_iadd(&b, out_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2)),
                     4);

   EXPECT_EQ(count_ifs(b.shader), 2u);
   EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
   nir_validate_shader(b.shader, "after fuse-redir");
   EXPECT_EQ(count_ifs(b.shader), 1u);
   EXPECT_TRUE(uses_zero_page_const(b.shader));
   ralloc_free(b.shader);
}

TEST(GuardChains, DominatingLoadProofComparesEveryPackedAddressChannel)
{
   /* pack_64_2x32 has one destination channel but reads two. With a high
    * word from a different channel (v.xz against v.xy), the dominating load
    * covers another address, so the guarded load must still be redirected.
    * With the same channels the proof holds and nothing is redirected. */
   for (unsigned mode = 0; mode < 2; ++mode) {
      const bool same_hi = mode == 1;
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "fuse-pack-channels");
      b.shader->info.workgroup_size[0] = 32;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *wid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
      nir_def *hi_a = agx_test_load_preamble(&b, 1, 32, 0);
      nir_def *hi_b = agx_test_load_preamble(&b, 1, 32, 2);
      nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 4);
      nir_def *zero = nir_imm_float(&b, 0.0f);

      nir_def *v_dom = nir_vec3(&b, nir_ishl_imm(&b, wid, 2), hi_a, hi_b);
      nir_def *self =
         make_load_global(&b, 1, 32, pack_64_2x32_channels(&b, v_dom, 0, 1), 4,
                          (gl_access_qualifier)0);

      nir_push_if(&b, nir_ine(&b, tid, wid));
      nir_def *v_pre = nir_vec3(&b, nir_ishl_imm(&b, tid, 2), hi_a, hi_b);
      nir_def *other = make_load_global(
         &b, 1, 32, pack_64_2x32_channels(&b, v_pre, 0, same_hi ? 1 : 2), 4,
         (gl_access_qualifier)0);
      nir_push_if(&b, nir_flt(&b, other, self));
      nir_def *r = nir_frsq(&b, other);
      nir_pop_if(&b, NULL);
      nir_def *inner_r = nir_if_phi(&b, r, zero);
      nir_pop_if(&b, NULL);

      nir_def *res = nir_if_phi(&b, inner_r, zero);
      make_store_global(
         &b, res,
         nir_iadd(&b, out_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2)), 4);

      EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after fuse-pack-channels");
      EXPECT_EQ(count_ifs(b.shader), 1u);
      EXPECT_EQ(uses_zero_page_const(b.shader), !same_hi) << "mode " << mode;
      ralloc_free(b.shader);
   }
}

TEST(GuardChains, FuseRejectsVectorInnerPhiReadByScalarSuffix)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "fuse-vector-phi");
   b.shader->info.workgroup_size[0] = 32;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *acc_in = nir_imm_int(&b, 1);
   nir_def *zero2 = nir_imm_ivec2(&b, 0, 0);

   nir_push_if(&b, nir_ult(&b, tid, nir_imm_int(&b, 16)));
   nir_def *addr =
      nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2));
   nir_def *val = make_load_global(&b, 1, 32, addr, 4, (gl_access_qualifier)0);
   nir_push_if(&b, nir_flt(&b, val, nir_imm_float(&b, 1.0f)));
   nir_def *flags =
      nir_vec2(&b, nir_f2i32(&b, nir_frsq(&b, val)), nir_imm_int(&b, 4));
   nir_pop_if(&b, NULL);
   /* 32x2 inner phi whose .x feeds the scalar 32x1 suffix reduction. */
   nir_def *inner_flags = nir_if_phi(&b, flags, zero2);
   nir_def *ored = ior_channel(&b, acc_in, inner_flags, 0);
   nir_pop_if(&b, NULL);

   nir_def *res = nir_if_phi(&b, ored, acc_in);
   make_store_global(
      &b, res, nir_iadd(&b, out_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2)),
      4);

   /* Fusing would rebuild the reduction from the 32x2 then-value and feed a
    * vector into the scalar merge phi: the guards must stay as they are. */
   EXPECT_EQ(count_ifs(b.shader), 2u);
   EXPECT_FALSE(agx_nir_opt_guard_chains(b.shader));
   nir_validate_shader(b.shader, "after fuse-vector-phi");
   EXPECT_EQ(count_ifs(b.shader), 2u);
   ralloc_free(b.shader);
}

TEST(GuardChains, NegativeControlRejectsSideEffectsAndVolatile)
{
   for (unsigned mode = 0; mode < 2; ++mode) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "neg-fuse");
      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *addr =
         nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2));
      nir_def *zero = nir_imm_float(&b, 0.0f);

      nir_push_if(&b, nir_ult(&b, tid, nir_imm_int(&b, 16)));
      nir_def *val = make_load_global(
         &b, 1, 32, addr, 4,
         mode == 0 ? ACCESS_VOLATILE : (gl_access_qualifier)0);
      if (mode == 1)
         make_store_global(&b, val, addr, 4);
      nir_push_if(&b, nir_flt(&b, val, nir_imm_float(&b, 1.0f)));
      nir_def *r = nir_frsq(&b, val);
      nir_pop_if(&b, NULL);
      nir_def *ir = nir_if_phi(&b, r, zero);
      nir_pop_if(&b, NULL);

      make_store_global(&b, nir_if_phi(&b, ir, zero), addr, 4);

      EXPECT_FALSE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after neg-fuse");
      ralloc_free(b.shader);
   }
}

TEST(GuardChains, HoistAndCoalesceFourIterNBodyChain)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "nbody-chain");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *wid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *vel_base = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 8);
   nir_def *r2 = agx_test_load_preamble(&b, 1, 32, 12);

   nir_def *self_pos = make_load_global(
      &b, 2, 32,
      nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, wid), 3)), 8,
      (gl_access_qualifier)0);
   nir_def *self_vel = make_load_global(
      &b, 2, 32,
      nir_iadd(&b, vel_base, nir_ishl_imm(&b, nir_u2u64(&b, wid), 3)), 8,
      (gl_access_qualifier)0);

   nir_def *acc_x = nir_imm_float(&b, 0.0f);
   nir_def *acc_y = nir_imm_float(&b, 0.0f);
   nir_def *acc_f = nir_imm_int(&b, 0);
   nir_def *zero_i = nir_imm_int(&b, 0);

   for (unsigned k = 0; k < 4; ++k) {
      nir_def *j = k ? nir_iadd_imm(&b, tid, 32 * k) : tid;
      nir_push_if(&b, nir_ine(&b, j, wid));
      nir_def *j_pos = make_load_global(
         &b, 2, 32,
         nir_iadd(&b, pos_base, nir_ishl_imm(&b, nir_u2u64(&b, j), 3)), 8,
         (gl_access_qualifier)0);
      nir_def *dx = nir_fsub(&b, nir_channel(&b, self_pos, 0),
                             nir_channel(&b, j_pos, 0));
      nir_def *dy = nir_fsub(&b, nir_channel(&b, self_pos, 1),
                             nir_channel(&b, j_pos, 1));
      nir_def *d2 =
         nir_ffma(&b, dx, dx, nir_fmul(&b, dy, dy));
      nir_push_if(&b, nir_flt(&b, d2, r2));
      nir_def *j_vel = make_load_global(
         &b, 2, 32,
         nir_iadd(&b, vel_base, nir_ishl_imm(&b, nir_u2u64(&b, j), 3)), 8,
         (gl_access_qualifier)0);
      nir_def *inv = nir_frsq(&b, d2);
      nir_def *nx = nir_fmul(&b, dx, inv);
      nir_def *ny = nir_fmul(&b, dy, inv);
      nir_def *dvx = nir_fsub(&b, nir_channel(&b, self_vel, 0),
                              nir_channel(&b, j_vel, 0));
      nir_def *dvy = nir_fsub(&b, nir_channel(&b, self_vel, 1),
                              nir_channel(&b, j_vel, 1));
      nir_def *dot = nir_ffma(&b, dvx, nx, nir_fmul(&b, dvy, ny));
      nir_def *imp = nir_fmul(&b, dot, nir_imm_float(&b, -0.312f));
      nir_def *nx_acc = nir_ffma(&b, imp, nx, acc_x);
      nir_def *ny_acc = nir_ffma(&b, imp, ny, acc_y);
      nir_def *approaching = nir_flt(&b, dot, nir_imm_float(&b, 0.0f));
      nir_def *sx = nir_bcsel(&b, approaching, nx_acc, acc_x);
      nir_def *sy = nir_bcsel(&b, approaching, ny_acc, acc_y);
      nir_def *sf =
         nir_bcsel(&b, approaching, nir_imm_int(&b, 8), nir_imm_int(&b, 0));
      nir_pop_if(&b, NULL);
      nir_def *ix = nir_if_phi(&b, sx, acc_x);
      nir_def *iy = nir_if_phi(&b, sy, acc_y);
      nir_def *iflg = nir_if_phi(&b, sf, zero_i);
      nir_def *of = nir_ior(&b, acc_f, iflg);
      nir_pop_if(&b, NULL);

      acc_x = nir_if_phi(&b, ix, acc_x);
      acc_y = nir_if_phi(&b, iy, acc_y);
      acc_f = nir_if_phi(&b, of, acc_f);
   }

   make_store_global(
      &b, nir_vec3(&b, acc_x, acc_y, acc_f),
      nir_iadd(&b, out_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 4)), 16);

   /* Before: 8 ifs (4 top-level nested pairs). */
   EXPECT_EQ(count_ifs(b.shader), 8u);
   EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
   nir_validate_shader(b.shader, "after nbody-chain");

   /* After: 1 top-level coalesced fast-path guard containing 4 inner ifs = 5 ifs total. */
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   unsigned max_depth = 0, top_ifs = 0;
   unsigned total_ifs = count_if_depth_in_list(&impl->body, 0, &max_depth, &top_ifs);
   EXPECT_EQ(top_ifs, 1u);
   EXPECT_EQ(total_ifs, 5u);
   EXPECT_EQ(max_depth, 2u);
   EXPECT_FALSE(uses_zero_page_const(b.shader));

   /* End-to-end compile on both G17P (AGX3B) and G17S (AGX3C) with zero spills. */
   nir_shader *copy = nir_shader_clone(NULL, b.shader);
   agx_preprocess_nir_g17p(b.shader);
   struct agx_shader_key k17p = g17p_key();
   struct agx_shader_part out17p = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &k17p, &out17p));
   EXPECT_EQ(out17p.info.stats.spills, 0u);
   free(out17p.binary);

   agx_preprocess_nir_m5(copy);
   struct agx_shader_key k3c = agx3c_key();
   struct agx_shader_part out3c = {};
   ASSERT_TRUE(agx_compile_shader_nir(copy, &k3c, &out3c));
   EXPECT_EQ(out3c.info.stats.spills, 0u);
   free(out3c.binary);

   ralloc_free(copy);
   ralloc_free(b.shader);
}

TEST(GuardChains, NegativeControlDoesNotHoistAcrossStoreOrSubgroup)
{
   for (unsigned mode = 0; mode < 2; ++mode) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "neg-hoist");
      b.shader->info.workgroup_size[0] = 32;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *a0 = nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2));
      nir_def *a1 =
         nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, nir_iadd_imm(&b, tid, 32)), 2));

      nir_def *v0 = make_load_global(&b, 1, 32, a0, 4, (gl_access_qualifier)0);
      nir_push_if(&b, nir_flt(&b, v0, nir_imm_float(&b, 1.0f)));
      nir_def *r0;
      if (mode == 0) {
         make_store_global(&b, v0, a0, 4);
         r0 = nir_frsq(&b, v0);
      } else {
         r0 = nir_shuffle_xor(&b, v0, nir_imm_int(&b, 1));
      }
      nir_pop_if(&b, NULL);
      nir_def *p0 = nir_if_phi(&b, r0, v0);

      /* Must NOT hoist v1 above the first if (mode 0 has store, mode 1 has subgroup). */
      nir_def *v1 = make_load_global(&b, 1, 32, a1, 4, (gl_access_qualifier)0);
      nir_push_if(&b, nir_flt(&b, v1, nir_imm_float(&b, 1.0f)));
      nir_def *r1 = nir_frsq(&b, v1);
      nir_pop_if(&b, NULL);
      nir_def *p1 = nir_if_phi(&b, r1, v1);

      make_store_global(&b, nir_fadd(&b, p0, p1), a0, 4);

      EXPECT_FALSE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after neg-hoist");
      ralloc_free(b.shader);
   }
}

TEST(GuardChains, NegativeControlDoesNotHoistAcrossStoreOrBarrierBetweenIfs)
{
   for (unsigned mode = 0; mode < 2; ++mode) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "neg-mid-fence");
      b.shader->info.workgroup_size[0] = 32;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *a0 = nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2));
      nir_def *a1 =
         nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, nir_iadd_imm(&b, tid, 32)), 2));

      nir_def *v0 = make_load_global(&b, 1, 32, a0, 4, (gl_access_qualifier)0);
      nir_push_if(&b, nir_flt(&b, v0, nir_imm_float(&b, 1.0f)));
      nir_def *r0 = nir_frsq(&b, v0);
      nir_pop_if(&b, NULL);
      nir_def *p0 = nir_if_phi(&b, r0, v0);

      /* Side-effecting / non-reorderable instruction in `mid` between two pure
       * ifs: mode 0 has a global store to the same address loaded below; mode 1
       * has a workgroup memory+execution barrier before a shared load. Neither
       * subsequent load may hoist above the fence or the preceding if. */
      nir_def *v1;
      if (mode == 0) {
         make_store_global(&b, p0, a1, 4);
         v1 = make_load_global(&b, 1, 32, a1, 4, (gl_access_qualifier)0);
      } else {
         nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
                     .memory_scope = SCOPE_WORKGROUP,
                     .memory_semantics = NIR_MEMORY_ACQ_REL,
                     .memory_modes = nir_var_mem_shared);
         v1 = nir_load_shared(&b, 1, 32, nir_ishl_imm(&b, tid, 2), .align_mul = 4);
      }

      nir_push_if(&b, nir_flt(&b, v1, nir_imm_float(&b, 1.0f)));
      nir_def *r1 = nir_frsq(&b, v1);
      nir_pop_if(&b, NULL);
      nir_def *p1 = nir_if_phi(&b, r1, v1);

      make_store_global(&b, nir_fadd(&b, p0, p1), a0, 4);

      EXPECT_FALSE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after neg-mid-fence");
      ralloc_free(b.shader);
   }
}

TEST(GuardChains, HoistPreservesDominanceAcrossUnhoistedMidBlock)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "hoist-unhoisted-mid");
   b.shader->info.workgroup_size[0] = 32;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *a0 = nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2));
   nir_def *a2 =
      nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, nir_iadd_imm(&b, tid, 64)), 2));

   nir_def *v0 = make_load_global(&b, 1, 32, a0, 4, (gl_access_qualifier)0);
   nir_def *c0 = nir_flt(&b, v0, nir_imm_float(&b, 1.0f));
   nir_def *c1 = nir_flt(&b, v0, nir_imm_float(&b, 2.0f));

   nir_push_if(&b, c0);
   nir_def *r0 = nir_frsq(&b, v0);
   nir_pop_if(&b, NULL);
   nir_def *p0 = nir_if_phi(&b, r0, v0);

   /* mid_1 has only a pure constant/ALU def (no load, not if_1's condition),
    * so mid_1 is not hoisted. Its def must still block dependent instructions
    * in mid_2 from hoisting above if_0. */
   nir_def *scale = nir_fmul(&b, v0, nir_imm_float(&b, 0.9f));

   nir_push_if(&b, c1);
   nir_def *r1 = nir_frsq(&b, p0);
   nir_pop_if(&b, NULL);
   nir_def *p1 = nir_if_phi(&b, r1, p0);

   /* mid_2 loads v2 and computes a scaled value + condition depending on
    * `scale` (defined in mid_1). v2 itself is independent and may hoist, but
    * `scaled` and `c2` must remain dominated by `scale`. */
   nir_def *v2 = make_load_global(&b, 1, 32, a2, 4, (gl_access_qualifier)0);
   nir_def *scaled = nir_fmul(&b, v2, scale);
   nir_def *c2 = nir_flt(&b, scaled, nir_imm_float(&b, 1.0f));

   nir_push_if(&b, c2);
   nir_def *r2 = nir_frsq(&b, scaled);
   nir_pop_if(&b, NULL);
   nir_def *p2 = nir_if_phi(&b, r2, p1);

   make_store_global(&b, p2, a0, 4);

   agx_nir_opt_guard_chains(b.shader);
   nir_validate_shader(b.shader, "after hoist-unhoisted-mid");
   ralloc_free(b.shader);
}

static unsigned
count_alu_op_in_block(const nir_block *block, nir_op op)
{
   unsigned n = 0;
   nir_foreach_instr(instr, block) {
      if (instr->type == nir_instr_type_alu && nir_instr_as_alu(instr)->op == op)
         n++;
   }
   return n;
}

TEST(GuardChains, CoalesceRelaxesIneAndBuildsFloatThreshTree)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "coalesce-fthresh");
   b.shader->info.workgroup_size[0] = 32;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *wid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *thr = agx_test_load_preamble(&b, 1, 32, 8);

   nir_def *acc = nir_imm_float(&b, 0.0f);
   nir_def *vals[4];
   nir_def *conds[4];
   for (unsigned k = 0; k < 4; ++k) {
      nir_def *idx = k == 0 ? tid : nir_iadd_imm(&b, tid, 32 * k);
      nir_def *addr = nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, idx), 2));
      vals[k] = make_load_global(&b, 1, 32, addr, 4, (gl_access_qualifier)0);
      nir_def *neq = nir_ine(&b, idx, wid);
      nir_def *lt = nir_inot(&b, nir_fge(&b, vals[k], thr));
      conds[k] = nir_iand(&b, neq, lt);
   }

   for (unsigned k = 0; k < 4; ++k) {
      nir_push_if(&b, conds[k]);
      nir_def *r = nir_fadd(&b, acc, nir_frsq(&b, vals[k]));
      nir_pop_if(&b, NULL);
      acc = nir_if_phi(&b, r, acc);
   }

   make_store_global(
      &b, acc, nir_iadd(&b, out_base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2)), 4);

   EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
   nir_validate_shader(b.shader, "after coalesce-fthresh");

   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   nir_block *entry = nir_start_block(impl);
   /* Outer guard uses 3 bcsels (balanced selection tree over 4 floats), and all
    * 4 ine/iand instructions were sunk out of the hot entry block. */
   EXPECT_EQ(count_alu_op_in_block(entry, nir_op_bcsel), 3u);
   EXPECT_EQ(count_alu_op_in_block(entry, nir_op_ine), 0u);
   EXPECT_EQ(count_alu_op_in_block(entry, nir_op_iand), 0u);
   EXPECT_EQ(count_alu_op_in_block(entry, nir_op_ior), 0u);

   nir_cf_node *next = nir_cf_node_next(&entry->cf_node);
   ASSERT_NE(next, nullptr);
   ASSERT_EQ(next->type, nir_cf_node_if);
   nir_block *then_first = nir_if_first_then_block(nir_cf_node_as_if(next));
   EXPECT_EQ(count_alu_op_in_block(then_first, nir_op_ine), 4u);
   EXPECT_EQ(count_alu_op_in_block(then_first, nir_op_iand), 4u);

   /* End-to-end compile on G17P (AGX3B) and G17S (AGX3C) with zero spills. */
   nir_shader *copy = nir_shader_clone(NULL, b.shader);
   agx_preprocess_nir_g17p(b.shader);
   struct agx_shader_key k17p = g17p_key();
   struct agx_shader_part out17p = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &k17p, &out17p));
   EXPECT_EQ(out17p.info.stats.spills, 0u);
   free(out17p.binary);

   agx_preprocess_nir_m5(copy);
   struct agx_shader_key k3c = agx3c_key();
   struct agx_shader_part out3c = {};
   ASSERT_TRUE(agx_compile_shader_nir(copy, &k3c, &out3c));
   EXPECT_EQ(out3c.info.stats.spills, 0u);
   free(out3c.binary);

   ralloc_free(copy);
   ralloc_free(b.shader);
}

TEST(GuardChains, CoalesceFloatThreshTreeVariantsAndRefusals)
{
   /* Positive variants: flt(x, T) with k=3, and fge(T, x) (T on LHS) with k=4. */
   for (unsigned mode = 0; mode < 2; ++mode) {
      const unsigned k_ifs = mode == 0 ? 3u : 4u;
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "fthresh-variant");
      b.shader->info.workgroup_size[0] = 32;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *thr = agx_test_load_preamble(&b, 1, 32, 4);

      nir_def *vals[4];
      nir_def *conds[4];
      for (unsigned k = 0; k < k_ifs; ++k) {
         nir_def *idx = nir_iadd_imm(&b, tid, 32 * k);
         nir_def *addr = nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, idx), 2));
         vals[k] = make_load_global(&b, 1, 32, addr, 4, (gl_access_qualifier)0);
         conds[k] = mode == 0 ? nir_flt(&b, vals[k], thr)
                              : nir_fge(&b, thr, vals[k]);
      }

      nir_def *acc = nir_imm_float(&b, 0.0f);
      for (unsigned k = 0; k < k_ifs; ++k) {
         nir_push_if(&b, conds[k]);
         nir_def *r = nir_fadd(&b, acc, nir_frsq(&b, vals[k]));
         nir_pop_if(&b, NULL);
         acc = nir_if_phi(&b, r, acc);
      }
      make_store_global(
         &b, acc, nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2)), 4);

      EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after fthresh-variant");
      nir_block *entry = nir_start_block(nir_shader_get_entrypoint(b.shader));
      EXPECT_EQ(count_alu_op_in_block(entry, nir_op_bcsel), k_ifs - 1u);
      EXPECT_EQ(count_alu_op_in_block(entry, nir_op_ior), 0u);
      ralloc_free(b.shader);
   }

   /* Negative controls: (0) different thresholds, (1) mixed cmp ops, (2) k == 2.
    * Coalescing still runs, but must fall back to ior without emitting bcsel. */
   for (unsigned neg = 0; neg < 3; ++neg) {
      const unsigned k_ifs = neg == 2 ? 2u : 4u;
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "fthresh-refusal");
      b.shader->info.workgroup_size[0] = 32;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *thr0 = agx_test_load_preamble(&b, 1, 32, 4);
      nir_def *thr1 = agx_test_load_preamble(&b, 1, 32, 8);

      nir_def *vals[4];
      nir_def *conds[4];
      for (unsigned k = 0; k < k_ifs; ++k) {
         nir_def *idx = nir_iadd_imm(&b, tid, 32 * k);
         nir_def *addr = nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, idx), 2));
         vals[k] = make_load_global(&b, 1, 32, addr, 4, (gl_access_qualifier)0);
         nir_def *t = (neg == 0 && k == 3) ? thr1 : thr0;
         conds[k] = (neg == 1 && k == 3) ? nir_fge(&b, vals[k], t)
                                         : nir_flt(&b, vals[k], t);
      }

      nir_def *acc = nir_imm_float(&b, 0.0f);
      for (unsigned k = 0; k < k_ifs; ++k) {
         nir_push_if(&b, conds[k]);
         nir_def *r = nir_fadd(&b, acc, nir_frsq(&b, vals[k]));
         nir_pop_if(&b, NULL);
         acc = nir_if_phi(&b, r, acc);
      }
      make_store_global(
         &b, acc, nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2)), 4);

      EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after fthresh-refusal");
      nir_block *entry = nir_start_block(nir_shader_get_entrypoint(b.shader));
      EXPECT_EQ(count_alu_op_in_block(entry, nir_op_bcsel), 0u);
      EXPECT_EQ(count_alu_op_in_block(entry, nir_op_ior), k_ifs - 1u);
      ralloc_free(b.shader);
   }

   /* Negative control for sinking: an `ine` term with a use after `outer` must
    * stay in `entry` rather than sinking into `outer->then_list`. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "fthresh-liveout-cond");
      b.shader->info.workgroup_size[0] = 32;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *wid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *thr = agx_test_load_preamble(&b, 1, 32, 4);

      nir_def *vals[4];
      nir_def *neqs[4];
      nir_def *conds[4];
      for (unsigned k = 0; k < 4; ++k) {
         nir_def *idx = nir_iadd_imm(&b, tid, 32 * k);
         nir_def *addr = nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, idx), 2));
         vals[k] = make_load_global(&b, 1, 32, addr, 4, (gl_access_qualifier)0);
         neqs[k] = nir_ine(&b, idx, wid);
         conds[k] = nir_iand(&b, neqs[k], nir_flt(&b, vals[k], thr));
      }

      nir_def *acc = nir_imm_float(&b, 0.0f);
      for (unsigned k = 0; k < 4; ++k) {
         nir_push_if(&b, conds[k]);
         nir_def *r = nir_fadd(&b, acc, nir_frsq(&b, vals[k]));
         nir_pop_if(&b, NULL);
         acc = nir_if_phi(&b, r, acc);
      }
      nir_def *final_v = nir_bcsel(&b, neqs[0], acc, nir_imm_float(&b, 1.0f));
      make_store_global(
         &b, final_v, nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2)), 4);

      EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after fthresh-liveout-cond");
      nir_block *entry = nir_start_block(nir_shader_get_entrypoint(b.shader));
      EXPECT_EQ(count_alu_op_in_block(entry, nir_op_ine), 1u);
      ralloc_free(b.shader);
   }

   /* Unwrap fge(fadd(x_m, bias), 0.0) -> fge(x_m, -bias) when fadd allows ninf
    * and is single-use, sinking the fadds out of the hot entry block. When
    * preserve_inf is set on fadd, keep the fadds in entry instead. */
   for (bool preserve_inf : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "fthresh-zero-fadd");
      b.shader->info.workgroup_size[0] = 32;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *neg_thr = agx_test_load_preamble(&b, 1, 32, 4);
      nir_def *zero = nir_imm_float(&b, 0.0f);

      nir_def *vals[4];
      nir_def *conds[4];
      for (unsigned k = 0; k < 4; ++k) {
         nir_def *idx = nir_iadd_imm(&b, tid, 32 * k);
         nir_def *addr = nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, idx), 2));
         vals[k] = make_load_global(&b, 1, 32, addr, 4, (gl_access_qualifier)0);
         b.fp_math_ctrl = preserve_inf ? nir_fp_preserve_inf : nir_fp_fast_math;
         nir_def *diff = nir_fadd(&b, vals[k], neg_thr);
         b.fp_math_ctrl = nir_fp_preserve_inf | nir_fp_preserve_nan;
         conds[k] = nir_inot(&b, nir_fge(&b, diff, zero));
      }
      b.fp_math_ctrl = nir_fp_fast_math;

      nir_def *acc = nir_imm_float(&b, 0.0f);
      for (unsigned k = 0; k < 4; ++k) {
         nir_push_if(&b, conds[k]);
         nir_def *r = nir_fadd(&b, acc, nir_frsq(&b, vals[k]));
         nir_pop_if(&b, NULL);
         acc = nir_if_phi(&b, r, acc);
      }
      make_store_global(
         &b, acc, nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2)), 4);

      EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after fthresh-zero-fadd");
      nir_block *entry = nir_start_block(nir_shader_get_entrypoint(b.shader));
      EXPECT_EQ(count_alu_op_in_block(entry, nir_op_bcsel), 3u);
      EXPECT_EQ(count_alu_op_in_block(entry, nir_op_fadd),
                preserve_inf ? 4u : 0u);
      EXPECT_EQ(count_alu_op_in_block(entry, nir_op_fneg),
                preserve_inf ? 0u : 1u);
      ralloc_free(b.shader);
   }

   /* Un-CSE'd per-if threshold expressions pulled out of separate nested ifs
    * by fusion+hoisting must be CSE'd before coalescing so try_unwrap_zero_fadd_bias
    * still matches the shared bias across all 4 guards. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "fthresh-hoisted-cse-fadd");
      b.shader->info.workgroup_size[0] = 32;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *wid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *radius = agx_test_load_preamble(&b, 1, 32, 4);
      nir_def *zero = nir_imm_float(&b, 0.0f);

      /* Dominating unconditional load of pos[wid] so nested loads fuse without
       * zero-page redirect. */
      nir_def *self_addr =
         nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, wid), 2));
      nir_def *self_v =
         make_load_global(&b, 1, 32, self_addr, 4, (gl_access_qualifier)0);

      nir_def *acc = self_v;
      for (unsigned k = 0; k < 4; ++k) {
         nir_def *idx = k == 0 ? tid : nir_iadd_imm(&b, tid, 32 * k);
         nir_push_if(&b, nir_ine(&b, idx, wid));
         nir_def *addr =
            nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, idx), 2));
         nir_def *v =
            make_load_global(&b, 1, 32, addr, 4, (gl_access_qualifier)0);
         nir_def *neg_thr = nir_fneg(&b, nir_fmul(&b, radius, radius));
         nir_def *diff = nir_fadd(&b, v, neg_thr);
         nir_push_if(&b, nir_inot(&b, nir_fge(&b, diff, zero)));
         nir_def *r = nir_fadd(&b, acc, nir_frsq(&b, v));
         nir_pop_if(&b, NULL);
         nir_def *inner_phi = nir_if_phi(&b, r, acc);
         nir_pop_if(&b, NULL);
         acc = nir_if_phi(&b, inner_phi, acc);
      }
      make_store_global(
         &b, acc, nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2)), 4);

      EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after fthresh-hoisted-cse-fadd");
      nir_block *entry = nir_start_block(nir_shader_get_entrypoint(b.shader));
      EXPECT_EQ(count_alu_op_in_block(entry, nir_op_bcsel), 3u);
      EXPECT_EQ(count_alu_op_in_block(entry, nir_op_fadd), 0u);
      /* Only the original hoisted neg_thr fneg remains; no double-fneg emitted. */
      EXPECT_EQ(count_alu_op_in_block(entry, nir_op_fneg), 1u);
      ralloc_free(b.shader);
   }
}

/* Eight adjacent guarded ifs coalesce into one outer fast-path guard with a
 * 7-bcsel balanced float-threshold tree, and a compact-prefix guarded loop
 * unrolls by 8 (while >2 unconditional dwords/iter or unguarded bodies stay 4). */
TEST(GuardChains, CoalescesEightGuardedIfsAndCompactPrefixLoopUnrollsByEight)
{
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "coalesce-8");
      b.shader->info.workgroup_size[0] = 32;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *thr = agx_test_load_preamble(&b, 1, 32, 4);
      nir_def *neg_thr = nir_fneg(&b, thr);
      nir_def *zero = nir_imm_float(&b, 0.0f);

      nir_def *vals[8];
      nir_def *conds[8];
      for (unsigned k = 0; k < 8; ++k) {
         nir_def *idx = nir_iadd_imm(&b, tid, 32 * k);
         nir_def *addr =
            nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, idx), 2));
         vals[k] = make_load_global(&b, 1, 32, addr, 4, (gl_access_qualifier)0);
         nir_def *diff = nir_fadd(&b, vals[k], neg_thr);
         conds[k] = nir_inot(&b, nir_fge(&b, diff, zero));
      }

      nir_def *acc = nir_imm_float(&b, 0.0f);
      for (unsigned k = 0; k < 8; ++k) {
         nir_push_if(&b, conds[k]);
         nir_def *r = nir_fadd(&b, acc, nir_frsq(&b, vals[k]));
         nir_pop_if(&b, NULL);
         acc = nir_if_phi(&b, r, acc);
      }
      make_store_global(
         &b, acc, nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2)),
         4);

      EXPECT_TRUE(agx_nir_opt_guard_chains(b.shader));
      nir_validate_shader(b.shader, "after coalesce-8");
      EXPECT_EQ(count_ifs(b.shader), 9u);
      nir_block *entry = nir_start_block(nir_shader_get_entrypoint(b.shader));
      EXPECT_EQ(count_alu_op_in_block(entry, nir_op_bcsel), 7u);
      EXPECT_EQ(count_alu_op_in_block(entry, nir_op_fadd), 0u);
      ralloc_free(b.shader);
   }

   auto run_loop_unroll = [](bool nested_if, unsigned top_load_comps) -> unsigned {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "guarded-unroll8");
      b.shader->info.workgroup_size[0] = 32;
      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *base2 = agx_test_load_preamble(&b, 1, 64, 4);
      nir_def *n = agx_test_load_preamble(&b, 1, 32, 8);
      nir_def *thr = agx_test_load_preamble(&b, 1, 32, 10);
      nir_def *one = nir_imm_float(&b, 1.0f);

      nir_variable *iv_var =
         nir_local_variable_create(b.impl, glsl_uint_type(), "iv");
      nir_variable *acc_var =
         nir_local_variable_create(b.impl, glsl_float_type(), "acc");
      nir_store_var(&b, iv_var, tid, 1);
      nir_store_var(&b, acc_var, nir_imm_float(&b, 0.0f), 1);

      nir_loop *loop = nir_push_loop(&b);
      nir_def *iv = nir_load_var(&b, iv_var);
      nir_break_if(&b, nir_uge(&b, iv, n));
      nir_def *addr =
         nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, iv), 4));
      nir_def *v = make_load_global(&b, top_load_comps, 32, addr, 4,
                                    (gl_access_qualifier)0);
      nir_def *x = nir_channel(&b, v, 0);
      if (nested_if)
         nir_push_if(&b, nir_flt(&b, x, thr));
      nir_def *addr2 =
         nir_iadd(&b, base2, nir_ishl_imm(&b, nir_u2u64(&b, iv), 3));
      nir_def *v2 =
         make_load_global(&b, 2, 32, addr2, 8, (gl_access_qualifier)0);
      nir_def *v2x = nir_channel(&b, v2, 0);
      nir_def *v2y = nir_channel(&b, v2, 1);
      nir_def *t = nir_frsq(&b, x);
      for (unsigned i = 0; i < 32; ++i)
         t = nir_ffma(&b, t, (i & 1) ? v2y : v2x, one);
      nir_store_var(&b, acc_var, nir_fadd(&b, nir_load_var(&b, acc_var), t), 1);
      if (nested_if)
         nir_pop_if(&b, NULL);
      nir_store_var(&b, iv_var, nir_iadd_imm(&b, iv, 32), 1);
      nir_pop_loop(&b, loop);

      make_store_global(
         &b, nir_load_var(&b, acc_var),
         nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, tid), 2)), 4);
      nir_lower_vars_to_ssa(b.shader);
      unsigned widest = 0;
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 320, &widest));
      nir_validate_shader(b.shader, "after guarded-unroll8");
      ralloc_free(b.shader);
      return widest;
   };

   /* Positive: compact guarded prefix (2 top-level dwords, ~48 total instrs) -> 8x. */
   EXPECT_EQ(run_loop_unroll(true, 2), 8u);
   /* Negative control: >2 unconditional top-level loaded dwords/iter stays at 4x. */
   EXPECT_EQ(run_loop_unroll(true, 3), 4u);
   /* Negative control: same size without a nested if stays at 4x. */
   EXPECT_EQ(run_loop_unroll(false, 2), 4u);
}

} // namespace

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
extern "C" {
#include "agx_compiler.h"
#include "agx_nir.h"
}

static nir_builder
make_compute_shader(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

TEST(G17PMod3RA, BankTiebreakAvoidsCoReadConflicts)
{
   /* Two loop-invariant 32-bit values k and m separated by 2 intermediate live
    * registers would normally land on r1 and r4 (both 1 mod 3), causing a
    * ~30% G17P mod-3 bank conflict on every ALU op reading (k, m). With G17P
    * mod-3 tie-breaking, RA places m on a non-conflicting bank strictly within
    * the 64-half occupancy tier. */
   nir_builder nb = make_compute_shader("g17p-mod3-ra-test");
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->nir = nb.shader;
   b->shader->stage = MESA_SHADER_COMPUTE;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *loop = agx_test_block(b->shader);
   agx_block *latch = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   loop->loop_header = true;
   agx_block_add_successor(entry, loop);
   agx_block_add_successor(loop, latch);
   agx_block_add_successor(latch, loop);
   agx_block_add_successor(loop, exit);

   b->cursor = agx_after_block(entry);
   agx_index k = agx_mov_imm(b, 32, 0x3f800000);
   agx_index pad0 = agx_mov_imm(b, 32, 0x40000000);
   agx_index pad1 = agx_mov_imm(b, 32, 0x40400000);
   agx_index m = agx_mov_imm(b, 32, 0x40800000);

   b->cursor = agx_after_block(loop);
   agx_index x0 = agx_ffma(b, k, m, pad0);
   agx_index x1 = agx_ffma(b, k, m, pad1);
   agx_index x2 = agx_ffma(b, k, m, x0);
   agx_index x3 = agx_ffma(b, k, m, x1);

   b->cursor = agx_after_block(exit);
   agx_unit_test(b, x2);
   agx_unit_test(b, x3);

   agx_compute_liveness(b->shader);
   agx_ra(b->shader);

   unsigned mod3_conflicts = 0;
   agx_foreach_instr_in_block(loop, I) {
      if (I->op != AGX_OPCODE_FFMA)
         continue;
      if (I->src[0].type == AGX_INDEX_REGISTER &&
          I->src[1].type == AGX_INDEX_REGISTER) {
         unsigned r0 = I->src[0].value >> 1;
         unsigned r1 = I->src[1].value >> 1;
         if (r0 != r1 && (r0 % 3u) == (r1 % 3u))
            mod3_conflicts++;
      }
   }
   EXPECT_EQ(mod3_conflicts, 0u);
   EXPECT_LE(b->shader->max_reg + 1u, 64u);
   EXPECT_EQ(b->shader->scratch_size_B, 0u);
   ralloc_free(mem);
   ralloc_free(nb.shader);

   /* End-to-end AGX3B compile of a multi-accumulator FMA compute loop packs
    * cleanly with zero spills and stays inside the 64-half tier. */
   nir_builder eb = make_compute_shader("g17p-mod3-e2e");
   nir_def *buf = nir_load_preamble(&eb, 1, 64, .base = 0);
   nir_def *out = nir_load_preamble(&eb, 1, 64, .base = 4);
   nir_def *n = nir_load_preamble(&eb, 1, 32, .base = 8);
   nir_def *gid = nir_load_global_invocation_id(&eb, 32);
   nir_def *tid = nir_channel(&eb, gid, 0);
   nir_def *k_val = nir_load_agx(&eb, 1, 32, buf, tid,
                                 .access = ACCESS_NON_WRITEABLE,
                                 .format = PIPE_FORMAT_R32_UINT);
   nir_def *m_val = nir_load_agx(&eb, 1, 32, buf, nir_iadd_imm(&eb, tid, 1),
                                 .access = ACCESS_NON_WRITEABLE,
                                 .format = PIPE_FORMAT_R32_UINT);
   nir_variable *vi = nir_local_variable_create(eb.impl, glsl_uint_type(), "i");
   nir_variable *va0 = nir_local_variable_create(eb.impl, glsl_float_type(), "a0");
   nir_variable *va1 = nir_local_variable_create(eb.impl, glsl_float_type(), "a1");
   nir_store_var(&eb, vi, nir_imm_int(&eb, 0), 1);
   nir_store_var(&eb, va0, nir_imm_float(&eb, 0.0f), 1);
   nir_store_var(&eb, va1, nir_imm_float(&eb, 1.0f), 1);
   nir_loop *lp = nir_push_loop(&eb);
   {
      nir_def *i = nir_load_var(&eb, vi);
      nir_break_if(&eb, nir_ige(&eb, i, n));
      nir_def *a0 = nir_load_var(&eb, va0);
      nir_def *a1 = nir_load_var(&eb, va1);
      nir_store_var(&eb, va0, nir_ffma(&eb, k_val, m_val, a0), 1);
      nir_store_var(&eb, va1, nir_ffma(&eb, m_val, a0, a1), 1);
      nir_store_var(&eb, vi, nir_iadd_imm(&eb, i, 1), 1);
   }
   nir_pop_loop(&eb, lp);
   nir_store_agx(&eb, nir_fadd(&eb, nir_load_var(&eb, va0), nir_load_var(&eb, va1)),
                 out, tid, .format = PIPE_FORMAT_R32_UINT);
   NIR_PASS(_, eb.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, eb.shader, nir_opt_dce);
   agx_preprocess_nir(eb.shader);
   struct agx_shader_key ekey = {};
   ekey.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   ekey.dev.chip = AGX_CHIP_G17P;
   ekey.no_preamble = true;
   ekey.reserved_preamble = 16;
   struct agx_shader_part part = {};
   EXPECT_TRUE(agx_compile_shader_nir(eb.shader, &ekey, &part));
   EXPECT_EQ(part.info.scratch_size, 0u);
   EXPECT_LE(part.info.nr_gprs, 64u);
   free(part.binary);
   ralloc_free(eb.shader);
}

TEST(G17PMod3RA, BankTiebreakRespectsDynamic96HalfTier)
{
   /* When conservative pre-RA demand exceeds 96 halves (putting g17p_tier_cap
    * at 128) yet actual allocation fits within 96 halves, G17P mod-3 bank
    * tie-breaking must never step a candidate above half 96. */
   nir_builder nb = make_compute_shader("g17p-mod3-tier-guard");
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->nir = nb.shader;
   b->shader->stage = MESA_SHADER_COMPUTE;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *loop = agx_test_block(b->shader);
   agx_block *latch = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   loop->loop_header = true;
   agx_block_add_successor(entry, loop);
   agx_block_add_successor(loop, latch);
   agx_block_add_successor(latch, loop);
   agx_block_add_successor(loop, exit);

   /* Keep 44 32-bit values live across the loop (r1..r44 -> halves 2..89,
    * plus r0 reserved -> 90 halves) and allocate one 32-bit vec2 (align 4 ->
    * r46..r47, halves 92..95) plus scalar acc (r45, halves 90..91) whose
    * pair conflicts mod-3 with earlier live regs. Conservative pre-RA demand
    * rounds past 96, but actual RA fits in <= 96 halves. */
   b->cursor = agx_after_block(entry);
   b->shader->any_cf = true;
   agx_index live[44];
   for (unsigned i = 0; i < 44; ++i)
      live[i] = agx_mov_imm(b, 32, 0x3f800000 + i);

   b->cursor = agx_after_block(loop);
   agx_index pair = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_device_load_to(b, pair, agx_uniform(0, AGX_SIZE_64),
                      live[0], AGX_FORMAT_I32, 0x3, 0, false);
   agx_instr *sp = agx_split(b, 2, pair);
   agx_index p0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_index p1 = agx_temp(b->shader, AGX_SIZE_32);
   sp->dest[0] = p0;
   sp->dest[1] = p1;
   /* Co-read p0 and p1 against multiple banks so r46/r47 has nonzero cost. */
   agx_index acc = agx_ffma(b, p0, live[5], live[8]);
   acc = agx_ffma(b, p1, live[6], acc);
   acc = agx_ffma(b, p0, live[11], acc);
   acc = agx_ffma(b, p1, live[12], acc);

   b->cursor = agx_after_block(exit);
   for (unsigned i = 0; i < 44; ++i)
      agx_unit_test(b, live[i]);
   agx_unit_test(b, acc);

   agx_compute_liveness(b->shader);
   agx_ra(b->shader);

   EXPECT_LE(b->shader->max_reg + 1u, 96u);
   EXPECT_EQ(b->shader->scratch_size_B, 0u);
   ralloc_free(mem);
   ralloc_free(nb.shader);
}

TEST(G17PMod3RA, LoopRmwChainAvoidsBackedgeXorSwaps)
{
   /* Four loop-carried scalar accumulators updated by multi-step FMA chains
    * should coalesce cleanly on the loop back edge without emitting XOR-swap
    * cycles in the latch block. */
   nir_builder nb = make_compute_shader("g17p-rmw-chain");
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->nir = nb.shader;
   b->shader->stage = MESA_SHADER_COMPUTE;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *loop = agx_test_block(b->shader);
   agx_block *latch = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   loop->loop_header = true;
   agx_block_add_successor(entry, loop);
   agx_block_add_successor(loop, latch);
   agx_block_add_successor(latch, loop);
   agx_block_add_successor(loop, exit);

   b->cursor = agx_after_block(entry);
   b->shader->any_cf = true;
   agx_index init[4];
   for (unsigned i = 0; i < 4; ++i)
      init[i] = agx_mov_imm(b, 32, 0);
   agx_index w[4];
   for (unsigned i = 0; i < 4; ++i)
      w[i] = agx_mov_imm(b, 32, 0x3f800000 + i);

   b->cursor = agx_after_block(loop);
   agx_instr *phis[4];
   agx_index acc[4];
   for (unsigned i = 0; i < 4; ++i) {
      acc[i] = agx_temp(b->shader, AGX_SIZE_32);
      phis[i] = agx_phi_to(b, acc[i], 2);
      phis[i]->src[0] = init[i];
   }

   for (unsigned step = 0; step < 4; ++step) {
      agx_index s = agx_fadd(b, w[step], w[(step + 1) & 3]);
      for (unsigned i = 0; i < 4; ++i)
         acc[i] = agx_ffma(b, s, w[step], acc[i]);
   }
   for (unsigned i = 0; i < 4; ++i)
      phis[i]->src[1] = acc[i];

   b->cursor = agx_after_block(exit);
   for (unsigned i = 0; i < 4; ++i)
      agx_unit_test(b, acc[i]);

   agx_compute_liveness(b->shader);
   agx_ra(b->shader);

   unsigned latch_xors = 0;
   agx_foreach_instr_in_block(latch, I) {
      if (I->op == AGX_OPCODE_BITOP)
         latch_xors++;
   }
   EXPECT_EQ(latch_xors, 0u);
   EXPECT_EQ(b->shader->scratch_size_B, 0u);
   ralloc_free(mem);
   ralloc_free(nb.shader);
}

TEST(G17PMod3RA, NonG17PAndPreambleGating)
{
   /* Verify mod-3 tie-breaking, loop_phi_targets, and rmw_web stay strictly
    * gated on AGX_CAP_MOD3_REG_BANK_RA and non-preamble shaders: disabled on
    * AGX3D (G18G), on non-capable AGX2 chips (G13G), and on G17P/M5
    * preambles. Use the same k/pad0/pad1/m loop that AGX3B and M5 resolve to 0
    * conflicts and confirm it stays untouched (k=r1, m=r4) when gated off. */
   for (unsigned mode = 0; mode < 4; ++mode) {
      nir_builder nb = make_compute_shader("g17p-mod3-gating");
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      struct agx_shader_info out = {};
      if (mode == 0) {
         key.dev.chip = AGX_CHIP_G18G;
         key.dev.encoding = AGX_ISA_ENCODING_AGX3D;
      } else if (mode == 1) {
         key.dev.chip = AGX_CHIP_G13G;
         key.dev.encoding = AGX_ISA_ENCODING_AGX2;
      } else if (mode == 2) {
         key.dev.chip = AGX_CHIP_G17P;
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         b->shader->is_preamble = true;
      } else {
         key.dev.chip = AGX_CHIP_G17C;
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
         b->shader->is_preamble = true;
      }
      b->shader->key = &key;
      b->shader->out = &out;
      b->shader->nir = nb.shader;
      b->shader->stage = MESA_SHADER_COMPUTE;

      agx_block *entry = agx_start_block(b->shader);
      agx_block *loop = agx_test_block(b->shader);
      agx_block *latch = agx_test_block(b->shader);
      agx_block *exit = agx_test_block(b->shader);
      loop->loop_header = true;
      agx_block_add_successor(entry, loop);
      agx_block_add_successor(loop, latch);
      agx_block_add_successor(latch, loop);
      agx_block_add_successor(loop, exit);

      b->cursor = agx_after_block(entry);
      agx_index k = agx_mov_imm(b, 32, 0x3f800000);
      agx_index pad0 = agx_mov_imm(b, 32, 0x40000000);
      agx_index pad1 = agx_mov_imm(b, 32, 0x40400000);
      agx_index m = agx_mov_imm(b, 32, 0x40800000);

      b->cursor = agx_after_block(loop);
      agx_index x0 = agx_ffma(b, k, m, pad0);
      agx_index x1 = agx_ffma(b, k, m, pad1);
      agx_index x2 = agx_ffma(b, k, m, x0);
      agx_index x3 = agx_ffma(b, k, m, x1);

      b->cursor = agx_after_block(exit);
      agx_unit_test(b, x2);
      agx_unit_test(b, x3);

      agx_compute_liveness(b->shader);
      agx_ra(b->shader);

      unsigned mod3_conflicts = 0;
      agx_foreach_instr_in_block(loop, I) {
         if (I->op != AGX_OPCODE_FFMA)
            continue;
         if (I->src[0].type == AGX_INDEX_REGISTER &&
             I->src[1].type == AGX_INDEX_REGISTER) {
            unsigned r0 = I->src[0].value >> 1;
            unsigned r1 = I->src[1].value >> 1;
            if (r0 != r1 && (r0 % 3u) == (r1 % 3u))
               mod3_conflicts++;
         }
      }
      EXPECT_GT(mod3_conflicts, 0u);
      EXPECT_EQ(b->shader->scratch_size_B, 0u);
      ralloc_free(mem);
      ralloc_free(nb.shader);
   }
}

TEST(G17PMod3RA, M5ResolvesMod3Conflicts)
{
   /* Verify native AGX3 (G15, G16G) and M5 (G17G, G17S, G17C) carry
    * AGX_CAP_MOD3_REG_BANK_RA and resolve the same co-read loop to zero
    * mod-3 bank conflicts without spilling. */
   const enum agx_chip cap_chips[] = {
      AGX_CHIP_G15G,
      AGX_CHIP_G15S,
      AGX_CHIP_G15C,
      AGX_CHIP_G16G,
      AGX_CHIP_G17G,
      AGX_CHIP_G17S,
      AGX_CHIP_G17C,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(cap_chips); ++i) {
      nir_builder nb = make_compute_shader("m5-mod3-loop");
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      struct agx_shader_info out = {};
      key.dev.chip = cap_chips[i];
      key.dev.encoding = agx_chip_isa_encoding(cap_chips[i]);
      b->shader->key = &key;
      b->shader->out = &out;
      b->shader->nir = nb.shader;
      b->shader->stage = MESA_SHADER_COMPUTE;

      agx_block *entry = agx_start_block(b->shader);
      agx_block *loop = agx_test_block(b->shader);
      agx_block *latch = agx_test_block(b->shader);
      agx_block *exit = agx_test_block(b->shader);
      loop->loop_header = true;
      agx_block_add_successor(entry, loop);
      agx_block_add_successor(loop, latch);
      agx_block_add_successor(latch, loop);
      agx_block_add_successor(loop, exit);

      b->cursor = agx_after_block(entry);
      agx_index k = agx_mov_imm(b, 32, 0x3f800000);
      agx_index pad0 = agx_mov_imm(b, 32, 0x40000000);
      agx_index pad1 = agx_mov_imm(b, 32, 0x40400000);
      agx_index m = agx_mov_imm(b, 32, 0x40800000);

      b->cursor = agx_after_block(loop);
      agx_index x0 = agx_ffma(b, k, m, pad0);
      agx_index x1 = agx_ffma(b, k, m, pad1);
      agx_index x2 = agx_ffma(b, k, m, x0);
      agx_index x3 = agx_ffma(b, k, m, x1);

      b->cursor = agx_after_block(exit);
      agx_unit_test(b, x2);
      agx_unit_test(b, x3);

      agx_compute_liveness(b->shader);
      agx_ra(b->shader);

      unsigned mod3_conflicts = 0;
      agx_foreach_instr_in_block(loop, I) {
         if (I->op != AGX_OPCODE_FFMA)
            continue;
         if (I->src[0].type == AGX_INDEX_REGISTER &&
             I->src[1].type == AGX_INDEX_REGISTER) {
            unsigned r0 = I->src[0].value >> 1;
            unsigned r1 = I->src[1].value >> 1;
            if (r0 != r1 && (r0 % 3u) == (r1 % 3u))
               mod3_conflicts++;
         }
      }
      EXPECT_EQ(mod3_conflicts, 0u);
      EXPECT_EQ(b->shader->scratch_size_B, 0u);
      ralloc_free(mem);
      ralloc_free(nb.shader);
   }
}

TEST(G17PMod3RA, SsaZeroAndDeadPartnerDoNotCausePhantomConflicts)
{
   /* Case 1: SSA 0 is a co-read partner of `v` that has NOT been visited yet
    * when `v` is allocated in `entry`, while `any_cf` reserves r0l
    * (used_regs[0] = 1). Unvisited SSA 0 must not look live at r0 (bank 0) and
    * push `v` off its natural slot r3 (halves 6..7, bank 0). */
   {
      nir_builder nb = make_compute_shader("g17p-mod3-ssa0-unvisited");
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->nir = nb.shader;
      b->shader->stage = MESA_SHADER_COMPUTE;

      agx_block *entry = agx_start_block(b->shader);
      agx_block *loop = agx_test_block(b->shader);
      agx_block *latch = agx_test_block(b->shader);
      agx_block *exit = agx_test_block(b->shader);
      loop->loop_header = true;
      agx_block_add_successor(entry, loop);
      agx_block_add_successor(loop, latch);
      agx_block_add_successor(latch, loop);
      agx_block_add_successor(loop, exit);

      b->shader->any_cf = true;
      /* Reserve SSA index 0 before emitting entry instructions. */
      agx_index p_ssa0 = agx_temp(b->shader, AGX_SIZE_32);
      ASSERT_EQ(p_ssa0.value, 0u);

      b->cursor = agx_after_block(entry);
      agx_index pad0 = agx_mov_imm(b, 32, 0x3f800000); /* r1 */
      agx_index pad1 = agx_mov_imm(b, 32, 0x40000000); /* r2 */
      agx_index v = agx_mov_imm(b, 32, 0x40400000);    /* natural r3 (bank 0) */
      agx_instr *v_def = agx_last_instr(entry);

      b->cursor = agx_after_block(loop);
      agx_mov_imm_to(b, p_ssa0, 0x40800000);
      agx_index a = agx_ffma(b, v, p_ssa0, pad0);
      agx_index c = agx_ffma(b, v, p_ssa0, pad1);

      b->cursor = agx_after_block(exit);
      agx_unit_test(b, a);
      agx_unit_test(b, c);

      agx_compute_liveness(b->shader);
      agx_ra(b->shader);

      /* `v` must stay at r3 (half 6) — no phantom bank-0 penalty from SSA 0. */
      EXPECT_EQ(v_def->dest[0].value, 6u);
      EXPECT_EQ(b->shader->scratch_size_B, 0u);
      ralloc_free(mem);
      ralloc_free(nb.shader);
   }

   /* Case 2: A partner killed before a sibling value is allocated must have
    * its reg_to_ssa invalidated so it does not falsely penalize allocations. */
   {
      nir_builder nb = make_compute_shader("g17p-mod3-dead-partner");
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->nir = nb.shader;
      b->shader->stage = MESA_SHADER_COMPUTE;

      agx_block *entry = agx_start_block(b->shader);
      agx_block *loop = agx_test_block(b->shader);
      agx_block *latch = agx_test_block(b->shader);
      agx_block *exit = agx_test_block(b->shader);
      loop->loop_header = true;
      agx_block_add_successor(entry, loop);
      agx_block_add_successor(loop, latch);
      agx_block_add_successor(latch, loop);
      agx_block_add_successor(loop, exit);

      b->cursor = agx_after_block(entry);
      b->shader->any_cf = true;
      /* dead_tmp lands at r1 (bank 1), dies immediately at keep0, freeing r1.
       * Then live values occupy r1..r3 and loop co-reads stay conflict-free
       * without spurious inflation. */
      agx_index dead_tmp = agx_mov_imm(b, 32, 0x3f800000);
      agx_index live0 = agx_fadd(b, dead_tmp, agx_immediate(0));
      agx_index live1 = agx_mov_imm(b, 32, 0x40000000);
      agx_index live2 = agx_mov_imm(b, 32, 0x40400000);

      b->cursor = agx_after_block(loop);
      agx_index s0 = agx_ffma(b, live0, live1, live2);
      agx_index s1 = agx_ffma(b, live0, live2, s0);

      b->cursor = agx_after_block(exit);
      agx_unit_test(b, s1);

      agx_compute_liveness(b->shader);
      agx_ra(b->shader);

      EXPECT_LE(b->shader->max_reg + 1u, 16u);
      EXPECT_EQ(b->shader->scratch_size_B, 0u);
      ralloc_free(mem);
      ralloc_free(nb.shader);
   }
}

TEST(G17PMod3RA, AboveHwmHoleAndUnalignedBoundaryResolution)
{
   /* Case 1: `first_reg` lands in a low-index hole (r1, bank 1) while HWM is
    * far above it (r0..r7 = 16 halves), and its loop co-read partner `k` sits
    * at r4 (bank 1). With tier_limit capped at cur_hwm + step_room (not
    * first_reg + count + step_room), the above-HWM loop checks r8 (bank 2) and
    * resolves the bank conflict to 0. */
   {
      nir_builder nb = make_compute_shader("g17p-mod3-hole-above-hwm");
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->nir = nb.shader;
      b->shader->stage = MESA_SHADER_COMPUTE;

      agx_block *entry = agx_start_block(b->shader);
      agx_block *loop = agx_test_block(b->shader);
      agx_block *latch = agx_test_block(b->shader);
      agx_block *exit = agx_test_block(b->shader);
      loop->loop_header = true;
      agx_block_add_successor(entry, loop);
      agx_block_add_successor(loop, latch);
      agx_block_add_successor(latch, loop);
      agx_block_add_successor(loop, exit);

      b->cursor = agx_after_block(entry);
      b->shader->any_cf = true;
      agx_index hole = agx_mov_imm(b, 32, 0x3f000000);  /* r1 (killed later) */
      agx_index live0 = agx_mov_imm(b, 32, 0x3f800000); /* r2 */
      agx_index live1 = agx_mov_imm(b, 32, 0x40000000); /* r3 */
      agx_index k = agx_mov_imm(b, 32, 0x40400000);     /* r4 (bank 1) */
      agx_index live2 = agx_mov_imm(b, 32, 0x40800000); /* r5 */
      agx_index live3 = agx_mov_imm(b, 32, 0x40a00000); /* r6 */
      agx_index live4 = agx_mov_imm(b, 32, 0x40c00000); /* r7 (bank 1) */
      /* Kill `hole` at r1 after HWM has reached r7 (16 halves). */
      agx_unit_test(b, hole);
      /* `m` sees first_reg = r1 (bank 1 hole), conflicting with k (r4) and
       * live4 (r7). No other slot below HWM is free; must step above HWM to
       * r8 (bank 2) via cur_hwm + step_room. */
      agx_index m = agx_mov_imm(b, 32, 0x40e00000);

      b->cursor = agx_after_block(loop);
      agx_index a0 = agx_ffma(b, k, m, live0);
      agx_index a1 = agx_ffma(b, live4, m, live1);
      agx_index a2 = agx_ffma(b, k, m, live2);
      agx_index a3 = agx_ffma(b, live4, m, live3);

      b->cursor = agx_after_block(exit);
      agx_unit_test(b, a0);
      agx_unit_test(b, a1);
      agx_unit_test(b, a2);
      agx_unit_test(b, a3);

      agx_compute_liveness(b->shader);
      agx_ra(b->shader);

      unsigned mod3_conflicts = 0;
      agx_foreach_instr_in_block(loop, I) {
         if (I->op != AGX_OPCODE_FFMA)
            continue;
         if (I->src[0].type == AGX_INDEX_REGISTER &&
             I->src[1].type == AGX_INDEX_REGISTER) {
            unsigned r0 = I->src[0].value >> 1;
            unsigned r1 = I->src[1].value >> 1;
            if (r0 != r1 && (r0 % 3u) == (r1 % 3u))
               mod3_conflicts++;
         }
      }
      EXPECT_EQ(mod3_conflicts, 0u);
      EXPECT_EQ(b->shader->scratch_size_B, 0u);
      ralloc_free(mem);
      ralloc_free(nb.shader);
   }

   /* Case 2: Unaligned HWM boundary for a 4-half aligned vec2.
    * Keep r1 (2..3), r4 (8..9), and a bank-2 partner at r5 (10..11) live while
    * freeing r2..r3 (4..7, banks 2 & 0) and r6 (12..13) so HWM = 14 halves.
    * The vec2 sees first_reg = r2..r3 (4..7, conflicting with r5 on bank 2)
    * while r6..r7 (12..15, banks 0 & 1) straddles HWM = 14. The old
    * ALIGN_POT(14, 4) = 16 skipped slot 12..15; above_start =
    * ALIGN_POT(14 - 4 + 1, 4) = 12 picks r6..r7 (max_reg + 1 <= 16) with zero
    * bank conflicts. */
   {
      nir_builder nb = make_compute_shader("g17p-mod3-unaligned-hwm");
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->nir = nb.shader;
      b->shader->stage = MESA_SHADER_COMPUTE;

      agx_block *entry = agx_start_block(b->shader);
      agx_block *loop = agx_test_block(b->shader);
      agx_block *latch = agx_test_block(b->shader);
      agx_block *exit = agx_test_block(b->shader);
      loop->loop_header = true;
      agx_block_add_successor(entry, loop);
      agx_block_add_successor(loop, latch);
      agx_block_add_successor(latch, loop);
      agx_block_add_successor(loop, exit);

      b->cursor = agx_after_block(entry);
      b->shader->any_cf = true;
      agx_index live1 = agx_mov_imm(b, 32, 0x3f800000); /* r1 (2..3) */
      agx_index tmp2 = agx_mov_imm(b, 32, 0x3f900000);  /* r2 (4..5, freed) */
      agx_index tmp3 = agx_mov_imm(b, 32, 0x3fa00000);  /* r3 (6..7, freed) */
      agx_index live4 = agx_mov_imm(b, 32, 0x40000000); /* r4 (8..9) */
      agx_index bank2 = agx_mov_imm(b, 32, 0x40400000); /* r5 (10..11, bank 2) */
      agx_index tmp6 = agx_mov_imm(b, 32, 0x40800000);  /* r6 (12..13, freed) */
      agx_unit_test(b, tmp2);
      agx_unit_test(b, tmp3);
      agx_unit_test(b, tmp6);

      /* Now HWM = 14 halves; free slots are r2..r3 (4..7, bank 2 on lane 0)
       * and r6.. (12.., straddling HWM=14, banks 0 & 1). */
      agx_index pair = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_instr *ld = agx_device_load_to(b, pair, agx_uniform(0, AGX_SIZE_64),
                                         live1, AGX_FORMAT_I32, 0x3, 0, false);
      agx_instr *sp = agx_split(b, 2, pair);
      agx_index p0 = agx_temp(b->shader, AGX_SIZE_32);
      agx_index p1 = agx_temp(b->shader, AGX_SIZE_32);
      sp->dest[0] = p0;
      sp->dest[1] = p1;

      b->cursor = agx_after_block(loop);
      agx_index a0 = agx_ffma(b, p0, bank2, live4);
      agx_index a1 = agx_ffma(b, p1, bank2, a0);
      agx_index a2 = agx_ffma(b, p0, bank2, a1);

      b->cursor = agx_after_block(exit);
      agx_unit_test(b, live1);
      agx_unit_test(b, live4);
      agx_unit_test(b, bank2);
      agx_unit_test(b, a2);

      agx_compute_liveness(b->shader);
      agx_ra(b->shader);

      /* Must pick straddling slot r6..r7 (half 12), not r2..r3 (half 4,
       * bank-2 conflict with r5) and not r8..r9 (half 16, skipping half 12). */
      EXPECT_EQ(ld->dest[0].value, 12u);
      EXPECT_LE(b->shader->max_reg + 1u, 16u);
      EXPECT_EQ(b->shader->scratch_size_B, 0u);
      ralloc_free(mem);
      ralloc_free(nb.shader);
   }
}

TEST(G17PMod3RA, MultiBlockLoopAndLivePreheaderKeepPhiCoalescing)
{
   /* Case 1: Multi-block loop (header -> body -> latch -> header) on M5 where
    * the body block is distinct from header. Enclosing loop-header phi targets
    * must stay protected in body and RMW accumulator chains must stay in their
    * header phi registers so latch emits zero back-edge mov/swap copies. */
   {
      nir_builder nb = make_compute_shader("m5-mod3-multiblock-loop");
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      struct agx_shader_info out = {};
      key.dev.chip = AGX_CHIP_G17C;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      b->shader->key = &key;
      b->shader->out = &out;
      b->shader->nir = nb.shader;
      b->shader->stage = MESA_SHADER_COMPUTE;

      agx_block *entry = agx_start_block(b->shader);
      agx_block *header = agx_test_block(b->shader);
      agx_block *body = agx_test_block(b->shader);
      agx_block *latch = agx_test_block(b->shader);
      agx_block *exit = agx_test_block(b->shader);
      header->loop_header = true;
      agx_block_add_successor(entry, header);
      agx_block_add_successor(header, body);
      agx_block_add_successor(header, exit);
      agx_block_add_successor(body, latch);
      agx_block_add_successor(latch, header);

      b->cursor = agx_after_block(entry);
      b->shader->any_cf = true;
      agx_index init[8];
      for (unsigned i = 0; i < 8; ++i)
         init[i] = agx_mov_imm(b, 32, 0);
      agx_index w[4];
      for (unsigned i = 0; i < 4; ++i)
         w[i] = agx_mov_imm(b, 32, 0x3f800000 + i);

      b->cursor = agx_after_block(header);
      agx_instr *phis[8];
      agx_index acc[8];
      for (unsigned i = 0; i < 8; ++i) {
         acc[i] = agx_temp(b->shader, AGX_SIZE_32);
         phis[i] = agx_phi_to(b, acc[i], 2);
         phis[i]->src[0] = init[i];
      }

      b->cursor = agx_after_block(body);
      for (unsigned step = 0; step < 4; ++step) {
         agx_index s = agx_fadd(b, w[step], w[(step + 1) & 3]);
         for (unsigned i = 0; i < 8; ++i)
            acc[i] = agx_ffma(b, s, w[step], acc[i]);
      }
      for (unsigned i = 0; i < 8; ++i)
         phis[i]->src[1] = acc[i];

      b->cursor = agx_after_block(exit);
      for (unsigned i = 0; i < 8; ++i)
         agx_unit_test(b, phis[i]->dest[0]);

      agx_compute_liveness(b->shader);
      agx_ra(b->shader);

      unsigned latch_copies = 0;
      agx_foreach_instr_in_block(latch, I) {
         if (I->op == AGX_OPCODE_MOV || I->op == AGX_OPCODE_BITOP)
            latch_copies++;
      }
      EXPECT_EQ(latch_copies, 0u);
      EXPECT_EQ(b->shader->scratch_size_B, 0u);
      ralloc_free(mem);
      ralloc_free(nb.shader);
   }

   /* Case 2: Sliding-window loop whose pre-header phi sources remain live
    * after the loop (so phi_web[root].reg stays pinned to the pre-header
    * registers while the header phis take distinct registers) and whose
    * loop-carried updates also feed a post-loop merge phi. Direct back-edge
    * tracking must keep all updates in their loop-header phi registers with
    * zero back-edge copies. */
   {
      nir_builder nb = make_compute_shader("m5-mod3-live-preheader-phi");
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      struct agx_shader_info out = {};
      key.dev.chip = AGX_CHIP_G17C;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      b->shader->key = &key;
      b->shader->out = &out;
      b->shader->nir = nb.shader;
      b->shader->stage = MESA_SHADER_COMPUTE;

      agx_block *entry = agx_start_block(b->shader);
      agx_block *preheader = agx_test_block(b->shader);
      agx_block *loop = agx_test_block(b->shader);
      agx_block *latch = agx_test_block(b->shader);
      agx_block *after_loop = agx_test_block(b->shader);
      agx_block *bypass = agx_test_block(b->shader);
      agx_block *exit = agx_test_block(b->shader);
      loop->loop_header = true;
      agx_block_add_successor(entry, preheader);
      agx_block_add_successor(entry, bypass);
      agx_block_add_successor(preheader, loop);
      agx_block_add_successor(loop, latch);
      agx_block_add_successor(latch, loop);
      agx_block_add_successor(loop, after_loop);
      agx_block_add_successor(bypass, exit);
      agx_block_add_successor(after_loop, exit);

      b->cursor = agx_after_block(entry);
      b->shader->any_cf = true;
      agx_index base[4];
      for (unsigned i = 0; i < 4; ++i)
         base[i] = agx_mov_imm(b, 16, 10 + i);
      agx_index step_v = agx_mov_imm(b, 16, 3);

      b->cursor = agx_after_block(loop);
      agx_instr *phis[4];
      agx_index cur[4];
      for (unsigned i = 0; i < 4; ++i) {
         cur[i] = agx_temp(b->shader, AGX_SIZE_16);
         phis[i] = agx_phi_to(b, cur[i], 2);
         phis[i]->src[0] = base[i];
      }
      agx_index nxt[4];
      for (unsigned i = 0; i < 4; ++i) {
         agx_index t = agx_iadd(b, step_v, agx_immediate(i + 1), 0);
         nxt[i] = agx_iadd(b, cur[i], t, 0);
         phis[i]->src[1] = nxt[i];
      }

      b->cursor = agx_after_block(exit);
      agx_index merged[4];
      for (unsigned i = 0; i < 4; ++i) {
         merged[i] = agx_temp(b->shader, AGX_SIZE_16);
         agx_instr *ephi = agx_phi_to(b, merged[i], 2);
         ephi->src[0] = base[i];
         ephi->src[1] = nxt[i];
      }
      for (unsigned i = 0; i < 4; ++i) {
         /* Keep pre-header `base[i]` live across the loop. */
         agx_unit_test(b, agx_iadd(b, merged[i], agx_neg(base[i]), 0));
      }

      agx_compute_liveness(b->shader);
      agx_ra(b->shader);

      unsigned latch_copies = 0;
      agx_foreach_instr_in_block(latch, I) {
         if (I->op == AGX_OPCODE_MOV || I->op == AGX_OPCODE_BITOP)
            latch_copies++;
      }
      EXPECT_EQ(latch_copies, 0u);
      EXPECT_EQ(b->shader->scratch_size_B, 0u);
      ralloc_free(mem);
      ralloc_free(nb.shader);
   }
}

TEST(G17PMod3RA, AdjacentFmaAccumulatorsAvoidSameBankRuns)
{
   /* Consecutive independent 3-source FFMA accumulator updates inside a hot
    * loop must avoid clustering adjacent destinations onto the same mod-3
    * register bank (which serializes back-to-back bank ports on G17/M5). */
   nir_builder nb = make_compute_shader("m5-mod3-adj-fma");
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   struct agx_shader_info out = {};
   key.dev.chip = AGX_CHIP_G17C;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   b->shader->key = &key;
   b->shader->out = &out;
   b->shader->nir = nb.shader;
   b->shader->stage = MESA_SHADER_COMPUTE;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *loop = agx_test_block(b->shader);
   agx_block *latch = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   loop->loop_header = true;
   agx_block_add_successor(entry, loop);
   agx_block_add_successor(loop, latch);
   agx_block_add_successor(latch, loop);
   agx_block_add_successor(loop, exit);

   b->cursor = agx_after_block(entry);
   b->shader->any_cf = true;
   agx_index a = agx_mov_imm(b, 32, 0x3f800000);
   agx_index w = agx_mov_imm(b, 32, 0x40000000);
   agx_index init[6];
   for (unsigned i = 0; i < 6; ++i)
      init[i] = agx_ffma(b, a, w, agx_mov_imm(b, 32, 0x3f000000 + i));

   b->cursor = agx_after_block(loop);
   agx_instr *phis[6];
   agx_index acc[6];
   for (unsigned i = 0; i < 6; ++i) {
      acc[i] = agx_temp(b->shader, AGX_SIZE_32);
      phis[i] = agx_phi_to(b, acc[i], 2);
      phis[i]->src[0] = init[i];
   }
   for (unsigned i = 0; i < 6; ++i) {
      acc[i] = agx_ffma(b, a, w, acc[i]);
      phis[i]->src[1] = acc[i];
   }

   b->cursor = agx_after_block(exit);
   for (unsigned i = 0; i < 6; ++i)
      agx_unit_test(b, acc[i]);

   agx_compute_liveness(b->shader);
   agx_ra(b->shader);

   unsigned adj_same_bank = 0;
   int prev_bank = -1;
   agx_foreach_instr_in_block(loop, I) {
      if (I->op != AGX_OPCODE_FFMA)
         continue;
      int bank = (int)((I->dest[0].value >> 1) % 3u);
      if (prev_bank == bank)
         adj_same_bank++;
      prev_bank = bank;
   }
   unsigned latch_copies = 0;
   agx_foreach_instr_in_block(latch, I) {
      if (I->op == AGX_OPCODE_MOV || I->op == AGX_OPCODE_BITOP)
         latch_copies++;
   }
   EXPECT_EQ(adj_same_bank, 0u);
   EXPECT_EQ(latch_copies, 0u);
   EXPECT_EQ(b->shader->scratch_size_B, 0u);
   ralloc_free(mem);
   ralloc_free(nb.shader);
}

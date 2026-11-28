/*
 * Copyright 2023 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx3_compute_program.h"
#include "agx3_static_spill_core.h"
#include "agx_builder.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_test.h"
extern "C" {
#include "asahi/isa/disasm.h"
}
#include "compiler/nir/nir_builder.h"

#include "util/macros.h"
#include "util/u_dynarray.h"
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

#define CASE(instr, expected)                                                  \
   do {                                                                        \
      agx_builder *A = agx_test_builder(mem_ctx);                              \
      agx_builder *B = agx_test_builder(mem_ctx);                              \
      {                                                                        \
         agx_builder *b = A;                                                   \
         instr;                                                                \
      }                                                                        \
      {                                                                        \
         agx_builder *b = B;                                                   \
         expected;                                                             \
      }                                                                        \
      A->shader->scratch_size_B = 1024;                                        \
      agx_lower_spill(A->shader);                                              \
      ASSERT_SHADER_EQUAL(A->shader, B->shader);                               \
   } while (0)

class LowerSpill : public testing::Test {
 protected:
   LowerSpill()
   {
      mem_ctx = ralloc_context(NULL);

      wx = agx_register(0, AGX_SIZE_32);
      hy = agx_register(2, AGX_SIZE_16);

      mw4 = agx_memory_register(0, AGX_SIZE_32);
      mh4 = agx_memory_register(0, AGX_SIZE_16);
      mw4.channels_m1 = 4 - 1;
      mh4.channels_m1 = 4 - 1;

      wx4 = wx;
      wx4.channels_m1 = 4 - 1;

      hy4 = hy;
      hy4.channels_m1 = 4 - 1;
   }

   ~LowerSpill()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;
   agx_index wx, hy, wx4, hy4;
   agx_index mw4, mh4;

   unsigned scalar = BITFIELD_MASK(1);
   unsigned vec4 = BITFIELD_MASK(4);

   enum agx_format i16 = AGX_FORMAT_I16;
   enum agx_format i32 = AGX_FORMAT_I32;
};

TEST_F(LowerSpill, ScalarSpills)
{
   CASE(agx_mov_to(b, agx_memory_register(11, AGX_SIZE_16), hy),
        agx_stack_store(b, hy, agx_immediate(22), i16, scalar));

   CASE(agx_mov_to(b, agx_memory_register(18, AGX_SIZE_32), wx),
        agx_stack_store(b, wx, agx_immediate(36), i32, scalar));
}

TEST_F(LowerSpill, ScalarFills)
{
   CASE(agx_mov_to(b, hy, agx_memory_register(11, AGX_SIZE_16)),
        agx_stack_load_to(b, hy, agx_immediate(22), i16, scalar));

   CASE(agx_mov_to(b, wx, agx_memory_register(18, AGX_SIZE_32)),
        agx_stack_load_to(b, wx, agx_immediate(36), i32, scalar));
}

TEST_F(LowerSpill, VectorSpills)
{
   CASE(agx_mov_to(b, mh4, hy4),
        agx_stack_store(b, hy4, agx_immediate(0), i16, vec4));

   CASE(agx_mov_to(b, mw4, wx4),
        agx_stack_store(b, wx4, agx_immediate(0), i32, vec4));
}

TEST_F(LowerSpill, VectorFills)
{
   CASE(agx_mov_to(b, hy4, mh4),
        agx_stack_load_to(b, hy4, agx_immediate(0), i16, vec4));

   CASE(agx_mov_to(b, wx4, mw4),
        agx_stack_load_to(b, wx4, agx_immediate(0), i32, vec4));
}

TEST_F(LowerSpill, ScalarSpill64)
{
   CASE(agx_mov_to(b, agx_memory_register(16, AGX_SIZE_64),
                   agx_register(8, AGX_SIZE_64)),
        agx_stack_store(b, agx_register(8, AGX_SIZE_64), agx_immediate(32), i32,
                        BITFIELD_MASK(2)));
}

TEST_F(LowerSpill, ScalarFill64)
{
   CASE(agx_mov_to(b, agx_register(16, AGX_SIZE_64),
                   agx_memory_register(8, AGX_SIZE_64)),
        agx_stack_load_to(b, agx_register(16, AGX_SIZE_64), agx_immediate(16),
                          i32, BITFIELD_MASK(2)));
}

TEST_F(LowerSpill, Vec6Spill)
{
   CASE(
      {
         agx_index mvec6 = agx_memory_register(16, AGX_SIZE_32);
         agx_index vec6 = agx_register(8, AGX_SIZE_32);
         vec6.channels_m1 = 6 - 1;
         mvec6.channels_m1 = 6 - 1;

         agx_mov_to(b, mvec6, vec6);
      },
      {
         agx_index vec4 = agx_register(8, AGX_SIZE_32);
         agx_index vec2 = agx_register(8 + (4 * 2), AGX_SIZE_32);
         vec4.channels_m1 = 4 - 1;
         vec2.channels_m1 = 2 - 1;

         agx_stack_store(b, vec4, agx_immediate(32), i32, BITFIELD_MASK(4));
         agx_stack_store(b, vec2, agx_immediate(32 + 4 * 4), i32,
                         BITFIELD_MASK(2));
      });
}

TEST_F(LowerSpill, Vec6Fill)
{
   CASE(
      {
         agx_index mvec6 = agx_memory_register(16, AGX_SIZE_32);
         agx_index vec6 = agx_register(8, AGX_SIZE_32);
         vec6.channels_m1 = 6 - 1;
         mvec6.channels_m1 = 6 - 1;

         agx_mov_to(b, vec6, mvec6);
      },
      {
         agx_index vec4 = agx_register(8, AGX_SIZE_32);
         agx_index vec2 = agx_register(8 + (4 * 2), AGX_SIZE_32);
         vec4.channels_m1 = 4 - 1;
         vec2.channels_m1 = 2 - 1;

         agx_stack_load_to(b, vec4, agx_immediate(32), i32, BITFIELD_MASK(4));
         agx_stack_load_to(b, vec2, agx_immediate(32 + 4 * 4), i32,
                           BITFIELD_MASK(2));
      });
}

TEST_F(LowerSpill, AGX3SplitsVectorStackAccess)
{
   agx_builder *A = agx_test_builder(mem_ctx);
   agx_builder *B = agx_test_builder(mem_ctx);

   {
      agx_builder *b = A;
      agx_index h3 = agx_register(6, AGX_SIZE_16);
      h3.channels_m1 = 3 - 1;
      agx_index w2 = agx_register(4, AGX_SIZE_32);
      w2.channels_m1 = 2 - 1;

      agx_stack_store(b, h3, agx_immediate(12), i16, BITFIELD_MASK(3));
      agx_stack_load_to(b, w2, agx_immediate(8), i32, BITFIELD_MASK(2));
   }
   {
      agx_builder *b = B;
      for (unsigned c = 0; c < 3; ++c) {
         agx_stack_store(b, agx_register(6 + c, AGX_SIZE_16),
                         agx_immediate(12 + 2 * c), i16, 1);
      }
      for (unsigned c = 0; c < 2; ++c) {
         agx_stack_load_to(b, agx_register(4 + 2 * c, AGX_SIZE_32),
                           agx_immediate(8 + 4 * c), i32, 1);
      }
   }

   agx3_split_vector_stack_access(A->shader);
   ASSERT_SHADER_EQUAL(A->shader, B->shader);
}

static void
build_live_rcp_chain(agx_builder *b, unsigned count)
{
   std::vector<agx_index> vals(count);
   agx_index seed = agx_uniform(0, AGX_SIZE_32);
   for (unsigned i = 0; i < count; ++i) {
      vals[i] = agx_rcp(b, seed);
      seed = vals[i];
   }
   for (unsigned i = 0; i < count; ++i)
      agx_unit_test(b, vals[i]);
}

TEST(SpillSlots, LargeSpillFrameHonorsLaunchCeiling)
{
   struct Case {
      enum agx_isa_encoding enc;
      enum agx_chip chip;
      mesa_shader_stage stage;
      uint32_t max_scratch_B;
      bool expect_success;
   };

   const Case cases[] = {
      /* AGX2 defaults to AGX2_MAX_SPILL_SLOTS (4096 B) when max_scratch_B is
       * unset or explicitly capped at 4 KiB, and admits up to 16 KiB when the
       * driver sets max_scratch_B (e.g. Vulkan GS raster replays). */
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, MESA_SHADER_COMPUTE, 0, false},
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, MESA_SHADER_VERTEX,
       AGX3_G17P_MAX_SPILL_B, false},
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, MESA_SHADER_VERTEX,
       2u * AGX_NUM_MODELED_REGS, true},
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, MESA_SHADER_COMPUTE,
       2u * AGX_NUM_MODELED_REGS, true},
      /* Graphics with the default or legacy launch limits is capped at 4 KiB. */
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, MESA_SHADER_VERTEX, 0, false},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, MESA_SHADER_FRAGMENT,
       AGX3_G17P_MAX_SPILL_B, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, MESA_SHADER_VERTEX, 0, false},
      /* HAL300 hardware vertex launches can publish the full linked TPS
       * frame through render resources, including for a GS raster replay. */
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, MESA_SHADER_VERTEX,
       2u * AGX_NUM_MODELED_REGS, true},
      /* AGX3 CDM stages (compute / GS) admit > 4 KiB up to 16 KiB in RA. */
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, MESA_SHADER_GEOMETRY,
       AGX3_G17P_MAX_COMPUTE_SPILL_B, true},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, MESA_SHADER_COMPUTE,
       AGX3_G17P_MAX_COMPUTE_SPILL_B, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, MESA_SHADER_COMPUTE, 1u << 16,
       true},
   };

   /* Keep 2400 16-bit halves (> 4096 B) simultaneously live across non-remat
    * instructions so RA must allocate spill slots above the 2048-half
    * (4096 B) boundary. */
   constexpr unsigned N = 1200;

   for (const Case &c : cases) {
      SCOPED_TRACE(c.chip);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = c.enc;
      key.dev.chip = c.chip;
      key.has_scratch = true;
      key.max_scratch_B = c.max_scratch_B;
      struct agx_shader_info out = {};
      b->shader->key = &key;
      b->shader->out = &out;
      b->shader->stage = c.stage;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = c.stage;

      build_live_rcp_chain(b, N);

      agx_ra(b->shader);
      if (c.expect_success) {
         EXPECT_FALSE(b->shader->ra_failed);
         EXPECT_FALSE(b->shader->ra_mem_exhausted);
         EXPECT_GT(b->shader->scratch_size_B, AGX3_G17P_MAX_SPILL_B);
         EXPECT_LE(b->shader->scratch_size_B, AGX_NUM_MODELED_REGS * 2u);
      } else {
         EXPECT_TRUE(b->shader->ra_failed);
         EXPECT_TRUE(b->shader->ra_mem_exhausted);
      }
      ralloc_free(mem);
   }
}

TEST(SpillSlots, SpillSlotExhaustionFailsClosedWithEarlyKill)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.has_scratch = true;
   key.max_scratch_B = AGX3_G17P_MAX_COMPUTE_SPILL_B;
   struct agx_shader_info out = {};
   b->shader->key = &key;
   b->shader->out = &out;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   /* Exceed AGX_NUM_MODELED_REGS in RA_MEM while spilling killed GPR sources:
    * must fail closed with ra_failed + ra_mem_exhausted instead of tripping
    * early-kill asserts or falling through into GPR live-range splitting. */
   constexpr unsigned N = AGX_NUM_MODELED_REGS / 2 + 128;
   build_live_rcp_chain(b, N);

   agx_ra(b->shader);
   EXPECT_TRUE(b->shader->ra_failed);
   EXPECT_TRUE(b->shader->ra_mem_exhausted);
   ralloc_free(mem);
}

TEST(SpillSlots, HighSlotEncodeRoundTripAGX3AndAGX2)
{
   const struct {
      enum agx_isa_encoding enc;
      enum agx_chip chip;
   } agx3_targets[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S},
   };
   const unsigned tps_halves[] = {2048u, 4096u, 8190u};
   const enum agx_size sizes[] = {AGX_SIZE_16, AGX_SIZE_32};

   for (const auto &target : agx3_targets) {
      for (unsigned tps_half : tps_halves) {
         for (enum agx_size size : sizes) {
            SCOPED_TRACE(tps_half);
            void *mem = ralloc_context(NULL);
            agx_builder *b = agx_test_builder(mem);
            agx_shader_key key = {};
            key.dev.encoding = target.enc;
            key.dev.chip = target.chip;
            key.has_scratch = true;
            key.max_scratch_B = AGX3_G17P_MAX_COMPUTE_SPILL_B;
            struct agx_shader_info out = {};
            b->shader->key = &key;
            b->shader->out = &out;
            b->shader->stage = MESA_SHADER_COMPUTE;
            b->shader->scratch_size_B = AGX_NUM_MODELED_REGS * 2u;

            agx_index src_gpr = agx_register(4, size);
            agx_index dst_gpr = agx_register(8, size);
            agx_index slot = agx_memory_register(tps_half, size);
            agx_mov_to(b, slot, src_gpr);
            agx_mov_to(b, dst_gpr, slot);
            agx_stop(b);

            agx_lower_spill(b->shader);
            agx3_split_vector_stack_access(b->shader);

            struct util_dynarray binary = UTIL_DYNARRAY_INIT;
            ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
            ASSERT_GE(binary.size, 12u);

            const uint8_t *code = (const uint8_t *)binary.data;
            agx3_static_spill_store_desc store = {};
            size_t store_len = 0;
            for (size_t len : {10u, 8u, 6u, 4u}) {
               if (len <= binary.size &&
                   agx3_static_spill_decode_store(target.enc, code, len,
                                                  &store)) {
                  store_len = len;
                  break;
               }
            }
            ASSERT_GT(store_len, 0u);
            EXPECT_EQ(store.tps_half, tps_half);
            EXPECT_EQ(store.src, 4u);
            EXPECT_EQ(store.width,
                      size == AGX_SIZE_16 ? AGX3_MEM_W16 : AGX3_MEM_W32);

            agx3_static_spill_load_desc load = {};
            size_t load_len = 0;
            for (size_t len : {10u, 8u, 6u, 4u}) {
               if (store_len + len <= binary.size &&
                   agx3_static_spill_decode_load(target.enc, code + store_len,
                                                 len, &load)) {
                  load_len = len;
                  break;
               }
            }
            ASSERT_GT(load_len, 0u);
            EXPECT_EQ(load.tps_half, tps_half);
            EXPECT_EQ(load.dest, 8u);
            EXPECT_EQ(load.width,
                      size == AGX_SIZE_16 ? AGX3_MEM_W16 : AGX3_MEM_W32);

            util_dynarray_fini(&binary);
            ralloc_free(mem);
         }
      }
   }

   /* Verify AGX2 stack_store/stack_load immediate packing handles offsets
    * >= 4096 bytes cleanly (16-bit immediate field) up to the modeled
    * spill-slot ceiling. */
   for (unsigned tps_half : {2048u, 8190u}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX2;
      key.dev.chip = AGX_CHIP_G13G;
      struct agx_shader_info out = {};
      b->shader->key = &key;
      b->shader->out = &out;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->scratch_size_B = AGX_NUM_MODELED_REGS * 2u;

      agx_index slot = agx_memory_register(tps_half, AGX_SIZE_32);
      agx_mov_to(b, slot, agx_register(4, AGX_SIZE_32));
      agx_mov_to(b, agx_register(8, AGX_SIZE_32), slot);
      agx_stop(b);

      agx_lower_spill(b->shader);

      struct util_dynarray binary = UTIL_DYNARRAY_INIT;
      agx_pack_binary(b->shader, &binary);
      ASSERT_GE(binary.size, 16u);

      char *disasm = NULL;
      size_t disasm_len = 0;
      FILE *fp = open_memstream(&disasm, &disasm_len);
      ASSERT_NE(fp, nullptr);
      signed store_bytes = agx2_disassemble_instr(
         (BITSET_WORD *)binary.data, fp, 0, binary.size, false);
      ASSERT_GT(store_bytes, 0);
      signed load_bytes = agx2_disassemble_instr(
         (BITSET_WORD *)((uint8_t *)binary.data + store_bytes), fp, 0,
         binary.size - store_bytes, false);
      ASSERT_GT(load_bytes, 0);
      fclose(fp);

      char expected_offs[32];
      snprintf(expected_offs, sizeof(expected_offs), "%u", tps_half * 2u);
      ASSERT_NE(disasm, nullptr);
      EXPECT_NE(strstr(disasm, "stack_store"), nullptr);
      EXPECT_NE(strstr(disasm, "stack_load"), nullptr);
      EXPECT_NE(strstr(disasm, expected_offs), nullptr);

      free(disasm);
      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}

TEST(SpillSlots, FullCompileRefusesOnRaMemExhaustion)
{
   nir_builder nb = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "ra_mem_exhaust");
   nb.shader->info.internal = false;
   nir_def *addr = nir_load_preamble(&nb, 1, 64, .base = 0);
   nir_def *seed = nir_load_global(&nb, 1, 32, addr, .align_mul = 4);

   /* 2600 simultaneously-live 32-bit values exceed a 4096-byte spill ceiling
    * and force RA_MEM exhaustion; agx_compile_shader_nir must refuse cleanly
    * without retrying GPR budget cuts or hitting early-kill asserts. */
   constexpr unsigned N = 2600;
   std::vector<nir_def *> vals(N);
   nir_def *cur = seed;
   for (unsigned i = 0; i < N; ++i) {
      cur = nir_fadd(&nb, nir_frcp(&nb, cur), seed);
      vals[i] = cur;
   }
   nir_def *acc = vals[N - 1];
   for (int i = N - 2; i >= 0; --i)
      acc = nir_fadd(&nb, acc, vals[i]);
   nir_store_global(&nb, acc, addr, .align_mul = 4);
   agx_preprocess_nir(nb.shader);

   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.has_scratch = true;
   key.max_scratch_B = AGX3_G17P_MAX_SPILL_B;
   key.no_preamble = true;

   struct agx_shader_part part = {};
   EXPECT_FALSE(agx_compile_shader_nir(nb.shader, &key, &part));
   free(part.binary);
   ralloc_free(nb.shader);
}

TEST(SpillSlots, FullCompileAtomicReturnRespectsTargetCeiling)
{
   glsl_type_singleton_init_or_ref();
   for (bool allow_cdm : {false, true}) {
      SCOPED_TRACE(allow_cdm);
      nir_builder nb = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "atomic_return_ceiling");
      nb.shader->info.internal = false;

      nir_def *addr = nir_load_preamble(&nb, 1, 64, .base = 0);
      nir_def *val = nir_load_preamble(&nb, 1, 32, .base = 4);
      nir_def *old = nir_global_atomic(&nb, 32, addr, val,
                                       .atomic_op = nir_atomic_op_iadd);
      nir_def *loaded = nir_load_scratch(&nb, 1, 32, nir_imm_int(&nb, 0),
                                         .align_mul = 4);
      nir_store_global(&nb, nir_iadd(&nb, old, loaded), addr, .align_mul = 4);
      agx_preprocess_nir(nb.shader);
      nb.shader->scratch_size = AGX3_G17P_MAX_SPILL_B;

      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.has_scratch = true;
      key.max_scratch_B = allow_cdm ? AGX3_G17P_MAX_COMPUTE_SPILL_B : 0;
      key.no_preamble = true;

      struct agx_shader_part part = {};
      bool ok = agx_compile_shader_nir(nb.shader, &key, &part);
      if (allow_cdm) {
         EXPECT_TRUE(ok);
         EXPECT_EQ(part.info.scratch_size, AGX3_G17P_MAX_SPILL_B);
         EXPECT_EQ(part.info.agx3_spectpr_scratch_size,
                   AGX3_G17P_MAX_SPILL_B + 16u);
      } else {
         /* 4096 B NIR scratch + 16 B returned-atomic reservation = 4112 B,
          * which exceeds the 4096 B graphics/default ceiling. */
         EXPECT_FALSE(ok);
      }

      free(part.binary);
      ralloc_free(nb.shader);
   }
   glsl_type_singleton_decref();
}

TEST(SpillSlots, RaMemBoundReservesAtomicReturnAndG18GAlignment)
{
   /* AGX3B returned atomic reserves 16 bytes above align(scratch_size_B, 4).
    * With spill_base_B = 4080 and max_scratch_B = 4096, zero spill slots
    * remain so any spill must fail closed with ra_mem_exhausted. */
   {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.has_scratch = true;
      key.max_scratch_B = AGX3_G17P_MAX_SPILL_B;
      struct agx_shader_info out = {};
      b->shader->key = &key;
      b->shader->out = &out;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
      b->shader->spill_base_B = AGX3_G17P_MAX_SPILL_B - 16u;

      agx_index addr = agx_mov(b, agx_uniform(0, AGX_SIZE_64));
      agx_index val = agx_mov(b, agx_uniform(4, AGX_SIZE_32));
      agx_index old = agx_atomic(b, val, addr, agx_immediate(0),
                                 AGX_ATOMIC_OPC_ADD);
      agx_unit_test(b, old);
      build_live_rcp_chain(b, 200);

      agx_ra(b->shader);
      EXPECT_TRUE(b->shader->ra_failed);
      EXPECT_TRUE(b->shader->ra_mem_exhausted);
      ralloc_free(mem);
   }

   /* G18G aligns its published TPS frame to 16 bytes: under a 4088-byte limit,
    * max aligned frame is 4080, so spill_base_B = 4080 leaves 0 spill slots. */
   {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = AGX_CHIP_G18G;
      key.dev.emit_agx3d = true;
      key.has_scratch = true;
      key.max_scratch_B = 4088u;
      struct agx_shader_info out = {};
      b->shader->key = &key;
      b->shader->out = &out;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
      b->shader->spill_base_B = 4080u;

      build_live_rcp_chain(b, 200);

      agx_ra(b->shader);
      EXPECT_TRUE(b->shader->ra_failed);
      EXPECT_TRUE(b->shader->ra_mem_exhausted);
      ralloc_free(mem);
   }
}

class Agx3SpillConstantPhi : public testing::Test {
protected:
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   Agx3SpillConstantPhi() {
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      b->shader->key = &key;
   }
   ~Agx3SpillConstantPhi() { ralloc_free(mem); }
   agx_index literal(uint64_t bits, enum agx_size size = AGX_SIZE_32) {
      agx_index x = agx_temp(b->shader, size);
      agx_mov_imm_to(b, x, bits);
      return x;
   }
   void join() { b->cursor = agx_after_block(agx_test_block(b->shader)); }
   agx_instr *phi(agx_index x, agx_index y, agx_index dest) {
      agx_instr *I = agx_phi_to(b, dest, 2);
      I->src[0] = x; I->src[1] = y;
      return I;
   }
   agx_instr *definition(agx_index x) {
      agx_foreach_instr_global(b->shader, I) {
         if (I->nr_dests == 1 && agx_is_equiv(I->dest[0], x)) return I;
      }
      return nullptr;
   }
};

TEST_F(Agx3SpillConstantPhi, SameLiteralCopiesJoin)
{
   agx_index x = literal(0x3f800000), y = literal(0x3f800000);
   agx_index copy = agx_mov(b, x);
   join();
   agx_index out = agx_temp(b->shader, AGX_SIZE_32);
   phi(copy, y, out);
   ASSERT_TRUE(agx_fold_agx3_spill_constant_phis(b->shader));
   ASSERT_EQ(definition(out)->op, AGX_OPCODE_MOV_IMM);
   EXPECT_EQ(definition(out)->imm, 0x3f800000u);
}

TEST_F(Agx3SpillConstantPhi, GroundedLoopWebPreservesLiteralPayload)
{
   agx_index root = literal(0x7fc01234);
   join();
   agx_index a = agx_temp(b->shader, AGX_SIZE_32);
   agx_index c = agx_temp(b->shader, AGX_SIZE_32);
   phi(root, c, a); phi(root, a, c);
   ASSERT_TRUE(agx_fold_agx3_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(a)->op, AGX_OPCODE_MOV_IMM);
   EXPECT_EQ(definition(c)->op, AGX_OPCODE_MOV_IMM);
   EXPECT_EQ(definition(a)->imm, 0x7fc01234u);
   EXPECT_EQ(definition(c)->imm, 0x7fc01234u);
}

TEST_F(Agx3SpillConstantPhi, ChangingLoopInputRemainsPhi)
{
   agx_index zero = literal(0), one = literal(1);
   join();
   agx_index a = agx_temp(b->shader, AGX_SIZE_32);
   agx_index c = agx_temp(b->shader, AGX_SIZE_32);
   phi(zero, c, a); phi(one, a, c);
   EXPECT_FALSE(agx_fold_agx3_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(a)->op, AGX_OPCODE_PHI);
   EXPECT_EQ(definition(c)->op, AGX_OPCODE_PHI);
}

TEST_F(Agx3SpillConstantPhi, RootFreeCycleCannotBorrowAnotherRootsValue)
{
   agx_index root = literal(1);
   join();
   agx_index a = agx_temp(b->shader, AGX_SIZE_32);
   agx_index c = agx_temp(b->shader, AGX_SIZE_32);
   agx_index out = agx_temp(b->shader, AGX_SIZE_32);
   phi(c, c, a); phi(a, a, c); phi(root, a, out);
   EXPECT_FALSE(agx_fold_agx3_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(out)->op, AGX_OPCODE_PHI);
}

TEST_F(Agx3SpillConstantPhi, RejectsUnprovenOrModifiedSources)
{
   for (unsigned mode = 0; mode < 9; ++mode) {
      agx_builder *old = b;
      b = agx_test_builder(mem); b->shader->key = &key;
      agx_index root = literal(1), other = root;
      switch (mode) {
      case 0: other.abs = true; break;
      case 1: other.neg = true; break;
      case 2: other.memory = true; break;
      case 3: other.size = AGX_SIZE_16; break;
      case 4: other.channels_m1 = 1; break;
      case 5: other = agx_uniform(2, AGX_SIZE_32); break;
      case 6: other = agx_register(2, AGX_SIZE_32); break;
      case 7: other = agx_immediate(1); break;
      case 8: other = agx_undef(AGX_SIZE_32); break;
      }
      join();
      agx_index out = agx_temp(b->shader, AGX_SIZE_32);
      phi(root, other, out);
      EXPECT_FALSE(agx_fold_agx3_spill_constant_phis(b->shader));
      EXPECT_EQ(definition(out)->op, AGX_OPCODE_PHI);
      b = old;
   }
}

TEST_F(Agx3SpillConstantPhi, LiteralWidthsAndSignedZeroBits)
{
   agx_index zero = literal(0), negzero = literal(0x80000000);
   agx_index short1 = literal(0x10001, AGX_SIZE_16);
   agx_index short2 = literal(1, AGX_SIZE_16);
   join();
   agx_index wide = agx_temp(b->shader, AGX_SIZE_32);
   agx_index narrow = agx_temp(b->shader, AGX_SIZE_16);
   phi(zero, negzero, wide); phi(short1, short2, narrow);
   ASSERT_TRUE(agx_fold_agx3_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(wide)->op, AGX_OPCODE_PHI);
   EXPECT_EQ(definition(narrow)->op, AGX_OPCODE_MOV_IMM);
   EXPECT_EQ(definition(narrow)->imm, 1u);
}

TEST_F(Agx3SpillConstantPhi, PreservesParallelPhiPrefix)
{
   agx_index zero = literal(0), one = literal(1);
   join();
   agx_index folded = agx_temp(b->shader, AGX_SIZE_32);
   agx_index varying = agx_temp(b->shader, AGX_SIZE_32);
   phi(zero, zero, folded); phi(zero, one, varying);
   agx_index result = agx_iadd(b, folded, varying, 0);
   ASSERT_TRUE(agx_fold_agx3_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(varying)->op, AGX_OPCODE_PHI);
   EXPECT_EQ(definition(folded)->op, AGX_OPCODE_MOV_IMM);
   EXPECT_EQ(definition(result)->op, AGX_OPCODE_IADD);
   bool body = false;
   agx_foreach_block(b->shader, block) {
      body = false;
      agx_foreach_instr_in_block(block, I) {
         if (I->op != AGX_OPCODE_PHI) body = true;
         else EXPECT_FALSE(body);
      }
   }
}

TEST_F(Agx3SpillConstantPhi, Agx3bFoldsAndAgx2OrPreambleUnchanged)
{
   agx_index root = literal(1); join();
   agx_index out = agx_temp(b->shader, AGX_SIZE_32);
   phi(root, root, out);
   key.dev.encoding = AGX_ISA_ENCODING_AGX2;
   EXPECT_FALSE(agx_fold_agx3_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(out)->op, AGX_OPCODE_PHI);

   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   b->shader->is_preamble = true;
   EXPECT_FALSE(agx_fold_agx3_spill_constant_phis(b->shader));
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   EXPECT_FALSE(agx_fold_agx3_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(out)->op, AGX_OPCODE_PHI);

   b->shader->is_preamble = false;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   ASSERT_TRUE(agx_fold_agx3_spill_constant_phis(b->shader));
   EXPECT_EQ(definition(out)->op, AGX_OPCODE_MOV_IMM);
   EXPECT_EQ(definition(out)->imm, 1u);
}

TEST(SpillSlots, Agx2SpillBoundHonorsImmediateOffsetCeiling)
{
   /* AGX2 stack_load/stack_store immediate byte offsets are 16-bit (< 64 KiB).
    * When NIR scratch (spill_base_B) already consumes >= 64 KiB, RA_MEM must
    * fail closed instead of emitting an out-of-range stack immediate. */
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX2;
   key.dev.chip = AGX_CHIP_G13G;
   key.has_scratch = true;
   key.max_scratch_B = 2u * AGX_NUM_MODELED_REGS;
   struct agx_shader_info out = {};
   b->shader->key = &key;
   b->shader->out = &out;
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;
   b->shader->spill_base_B = 0x10000u;

   build_live_rcp_chain(b, 200);

   agx_ra(b->shader);
   EXPECT_TRUE(b->shader->ra_failed);
   EXPECT_TRUE(b->shader->ra_mem_exhausted);
   ralloc_free(mem);
}

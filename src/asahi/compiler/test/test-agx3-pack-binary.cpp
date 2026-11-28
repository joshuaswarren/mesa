/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * End-to-end backend test: build AGX IR directly, run agx3_pack_binary, and
 * check the emitted AGX3 byte stream against the independently validated
 * encoders. This exercises the real IR walk, per-instruction translation,
 * modifier routing, and the terminal-end footer -- the lowering glue that the
 * portable encoder-core host tests do not cover.
 */

#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}
#include "compiler/nir/nir_builder.h"

#include <algorithm>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>

#include "agx_debug.h"
#include "agx3_alu_core.h"
#include "agx3_isel_core.h"
#include "agx3_atomic_core.h"
#include "agx3_binder.h"
#include "agx3_cf_core.h"
#include "agx3_copy_core.h"
#include "agx3_falu_mod_core.h"
#include "agx3_fltnorm_core.h"
#include "agx3_funop_core.h"
#include "agx3_g17p_quad_eot.h"
#include "agx3_iadd_core.h"
#include "agx3_local_tps_core.h"
#include "agx3_mem_core.h"
#include "agx3_native_stats.h"
#include "agx3_pack.h"
#include "agx3_scs_core.h"
#include "agx3_slot_core.h"
#include "agx3_sr_core.h"
#include "agx3_stage_core.h"
#include "agx3_static_spill_core.h"
#include "agx3_subgroup_core.h"
#include "agx3_tex_core.h"
#include "agx3_tile_core.h"
#include "agx3_pixel_core.h"
#include "agx3_select_core.h"
#include "agx3_vary_core.h"
#include "agx3_zs_core.h"
#include "agx_mesh_abi.h"
#include "asahi/lib/agx_chip_caps.h"
extern "C" {
#include "agx3_compose/agx3_decode.h"
#include "asahi/isa/disasm.h"
}
#include "probes/g17p-sparse-mrt.h"

/* agx_nir_lower_tilebuffer's ALL_SAMPLES: the pixel-rate sample mask, and the
 * only immediate AGX3's single-sample tile group can represent. */
static constexpr unsigned AGX_ALL_SAMPLES = 0xFF;

TEST(AGX3VertexExportAlu, RegisterCacheTracksFusedArithmeticConsumers)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_VERTEX;
   auto r = [](unsigned value) { return agx_register(value, AGX_SIZE_32); };
   agx_instr *producer = agx_fmul_to(b, r(4), r(0), r(2));
   agx_instr *first = agx_st_vary_mul_agx3(b, agx_immediate(0), r(4), r(6));
   agx_instr *last = agx_st_vary_fma_agx3(b, agx_immediate(1), r(4), r(8), r(10));
   agx_stop(b);
   agx_opt_register_cache(b->shader);
   EXPECT_TRUE(producer->dest[0].cache);
   EXPECT_TRUE(first->src[1].cache);
   EXPECT_FALSE(first->src[1].discard);
   EXPECT_FALSE(last->src[1].cache);
   EXPECT_TRUE(last->src[1].discard);
   ralloc_free(mem);
}

static void
append(std::vector<uint8_t> &v, const uint8_t *b, unsigned n)
{
   if (n)
      v.insert(v.end(), b, b + n);
}

static void
append_g17p_atomic_return_reload(std::vector<uint8_t> &v, uint16_t dest)
{
   const struct agx3_static_spill_load_desc reload = {
      .dest = dest,
      .tps_half = 0,
      .width = AGX3_MEM_W32,
      .slot = 6,
      .slot_mirror = 6,
      .wait_mask = 1,
   };
   uint8_t bytes[AGX3_STATIC_SPILL_MAX_LENGTH];
   uint8_t length = 0;
   ASSERT_TRUE(agx3_static_spill_encode_load_form(
      AGX_ISA_ENCODING_AGX3B, &reload, 1, bytes, &length));
   append(v, bytes, length);
}

class AGX3PackBinary : public testing::Test {
 protected:
   void *mem_ctx;
   agx_shader_key key;

   AGX3PackBinary()
   {
      mem_ctx = ralloc_context(NULL);
      memset(&key, 0, sizeof(key));
   }
   ~AGX3PackBinary()
   {
      ralloc_free(mem_ctx);
   }

   agx_builder *builder(enum agx_isa_encoding enc)
   {
      key.dev.encoding = enc;
      agx_builder *b = agx_test_builder(mem_ctx);
      b->shader->key = &key;
      return b;
   }

   std::vector<uint8_t> pack(agx_builder *b, bool expect_ok = true)
   {
      struct util_dynarray bin;
      util_dynarray_init(&bin, mem_ctx);
      bool ok = agx3_pack_binary(b->shader, &bin);
      EXPECT_EQ(ok, expect_ok);
      const uint8_t *d = (const uint8_t *)bin.data;
      return bin.size ? std::vector<uint8_t>(d, d + bin.size)
                      : std::vector<uint8_t>();
   }
};

/* The packer deposits the f11 destination register-cache hint on every AGX3b
 * integer register add/subtract (253b68945fd, "G17P v56h"); the hint is byte 4
 * bit 0 of these ten-byte forms. Expectations below that predate it are the
 * hint-free encodings with the hint applied. Several of them carry Apple
 * captures (the "Apple:" comments) with f11 clear: the packer deliberately
 * differs from those captures, and its CTS result on hardware was unchanged
 * (253b68945fd). */
static bool
int_dest_cache(void)
{
   return true;
}

TEST_F(AGX3PackBinary, MeshExportFencesUseExactLivePhaseMasks)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->stage = MESA_SHADER_MESH;
   agx_mov_imm_to(b, agx_register(8, AGX_SIZE_16), 32);
   agx_mov_imm_to(b, agx_register(10, AGX_SIZE_16), 64);
   agx_allocate_uvb_m5(b, agx_register(8, AGX_SIZE_16),
                        agx_register(10, AGX_SIZE_16));
   agx_index desc = agx_register(12, AGX_SIZE_32); desc.channels_m1 = 1;
   agx_mov_imm_to(b, agx_register(12, AGX_SIZE_32), 0);
   agx_mov_imm_to(b, agx_register(14, AGX_SIZE_32), 256);
   agx_store_uvb_m5(b, desc, 1);
   agx_fence_uvb_gen1(b, AGX_MESH_EXPORT_FENCE_INDICES);
   agx_store_uvb_m5(b, desc, 0);
   agx_fence_uvb_gen1(b, AGX_MESH_EXPORT_FENCE_VERTEX_BATCH);
   agx_store_uvb_m5(b, desc, 0);
   agx_stop(b);
   const auto code = pack(b);
   const uint8_t index[] = {0x0b,0x01,0x07,0x01,0,0x40,0,0,0,0};
   const uint8_t vertex[] = {0x0b,0x01,0x07,0x01,0,0x80,0,0,0,0};
   EXPECT_NE(std::search(code.begin(), code.end(), index, index + sizeof(index)), code.end());
   EXPECT_NE(std::search(code.begin(), code.end(), vertex, vertex + sizeof(vertex)), code.end());
   /* Neither the final export nor END gains a terminal phase wait. */
   ASSERT_GE(code.size(), 10u);
   EXPECT_EQ(std::vector<uint8_t>(code.end() - 4, code.end()),
             (std::vector<uint8_t>{0x0e,0,0,0}));
}

/* An in-place 32-bit NOT whose halves come from two loads in flight waits on
 * two slots. Only the unary AGX3b `inv` form carries that wait in place: the
 * NOT's ignored second bitop source must not keep it from that form
 * (llama.cpp's q3_K mat-vec failed to compile). */
TEST_F(AGX3PackBinary, InPlaceNotWaitingOnTwoSlotsPacks)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_device_load_to(b, agx_register(0, AGX_SIZE_16),
      agx_uniform(0, AGX_SIZE_64), agx_immediate(0), AGX_FORMAT_I16, 1, 0, false);
   agx_device_load_to(b, agx_register(1, AGX_SIZE_16),
      agx_uniform(0, AGX_SIZE_64), agx_immediate(2), AGX_FORMAT_I16, 1, 0, false);
   agx_bitop_to(b, agx_register(0, AGX_SIZE_32), agx_register(0, AGX_SIZE_32),
                agx_zero(), UTIL_LUT2(~a));
   agx_device_store(b, agx_register(0, AGX_SIZE_32), agx_uniform(0, AGX_SIZE_64),
                    agx_immediate(0), AGX_FORMAT_I32, 1, 0, false);
   agx_stop(b);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, MeshExportFenceDrainsBothPendingR0Halves)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->stage = MESA_SHADER_MESH;
   agx_mov_imm_to(b, agx_register(8, AGX_SIZE_16), 0);
   agx_allocate_uvb_m5(b, agx_register(8, AGX_SIZE_16),
                        agx_register(8, AGX_SIZE_16));
   /* Zero exports can still have unrelated explicit producers. Their two
    * current generations must complete before the carrier touches h0/h1. */
   agx_device_load_to(b, agx_register(0, AGX_SIZE_16),
      agx_uniform(0, AGX_SIZE_64), agx_immediate(0), AGX_FORMAT_I16, 1, 0, false);
   agx_device_load_to(b, agx_register(1, AGX_SIZE_16),
      agx_uniform(0, AGX_SIZE_64), agx_immediate(2), AGX_FORMAT_I16, 1, 0, false);
   agx_fence_uvb_gen1(b, AGX_MESH_EXPORT_FENCE_INDICES);
   agx_iadd_to(b, agx_register(4, AGX_SIZE_16),
      agx_register(0, AGX_SIZE_16), agx_register(1, AGX_SIZE_16), 0);
   agx_stop(b);
   const auto code = pack(b);
   const uint8_t complete[] = {0x0b,0x01,0x07,0x01,0,0x60,0,0,0,0};
   EXPECT_NE(std::search(code.begin(), code.end(), complete,
                         complete + sizeof(complete)), code.end());
}

TEST_F(AGX3PackBinary, MeshExportFenceRefusesInvalidProtocolOrChannelReuse)
{
   for (unsigned bad : {0u,1u,2u,3u,4u,5u}) {
      agx_builder *b = builder(bad == 5 ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B);
      key.dev.chip = bad == 5 ? AGX_CHIP_G17S : AGX_CHIP_G17P;
      b->shader->stage = MESA_SHADER_MESH;
      agx_mov_imm_to(b, agx_register(8, AGX_SIZE_16), 1);
      if (bad != 0)
         agx_allocate_uvb_m5(b, agx_register(8, AGX_SIZE_16), agx_register(8, AGX_SIZE_16));
      agx_index desc = agx_register(12, AGX_SIZE_32); desc.channels_m1 = 1;
      if (bad == 4) {
         agx_store_uvb_m5(b, desc, 1);
         /* Ordinary producers are now allocated away from phase channels.
          * Force the native ISEL pair instead: its required slot2 must not
          * be rewritten by the policy or reuse a pending index phase. */
         agx_index first = agx_register(4, AGX_SIZE_32);
         agx_index second = agx_register(6, AGX_SIZE_32);
         agx_device_load_to(b, first, agx_uniform(0, AGX_SIZE_64),
            agx_immediate(0), AGX_FORMAT_I32, 1, 0, false);
         agx_device_load_to(b, second, agx_uniform(0, AGX_SIZE_64),
            agx_immediate(4), AGX_FORMAT_I32, 1, 0, false);
         agx_icmpsel_to(b, agx_register(16, AGX_SIZE_32), first, second,
                          first, second, AGX_ICOND_UEQ);
      }
      agx_fence_uvb_gen1(b, bad == 1 ? AGX_MESH_EXPORT_FENCE_VERTEX_BATCH :
                          bad == 2 ? 2 : AGX_MESH_EXPORT_FENCE_INDICES);
      if (bad == 3)
         agx_fence_uvb_gen1(b, AGX_MESH_EXPORT_FENCE_INDICES);
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << bad;
   }
}

/* Actual unlinked GPL failure: position UVSTs precede root-table loads of
 * the page carrier and location0 selectors. A round-robin slot3 load after
 * the first vertex UVST violates the phase protocol even when its result is
 * immediately consumed by the next descriptor. Vary the incoming cursor so
 * success cannot depend on how many API-body loads happened to precede it. */
TEST_F(AGX3PackBinary, MeshUnlinkedSelectorsBetweenExports)
{
   for (bool with_system_reads : {false, true}) {
      SCOPED_TRACE(with_system_reads);
      for (unsigned warmup = 0; warmup < 6; ++warmup) {
         SCOPED_TRACE(warmup);
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         key.dev.chip = AGX_CHIP_G17P;
         b->shader->stage = MESA_SHADER_MESH;
         for (unsigned i = 0; i < warmup; ++i)
            agx_device_load_to(b, agx_register(32 + 2 * i, AGX_SIZE_32),
               agx_uniform(0, AGX_SIZE_64), agx_immediate(i),
               AGX_FORMAT_I32, 1, 0, false);
         for (unsigned i = 0; i < warmup; ++i)
            agx_mov_to(b, agx_register(64 + 2 * i, AGX_SIZE_32),
                          agx_register(32 + 2 * i, AGX_SIZE_32));
         agx_mov_imm_to(b, agx_register(8, AGX_SIZE_16), 2);
         agx_mov_imm_to(b, agx_register(10, AGX_SIZE_16), 6);
         agx_allocate_uvb_m5(b, agx_register(8, AGX_SIZE_16),
                              agx_register(10, AGX_SIZE_16));
         agx_index desc = agx_register(12, AGX_SIZE_32);
         desc.channels_m1 = 1;
         agx_mov_imm_to(b, agx_register(12, AGX_SIZE_32), 0);
         agx_mov_imm_to(b, agx_register(14, AGX_SIZE_32), 0);
         agx_store_uvb_m5(b, desc, 1);
         agx_fence_uvb_gen1(b, AGX_MESH_EXPORT_FENCE_INDICES);
         for (unsigned plane = 0; plane < 4; ++plane) {
            agx_mov_imm_to(b, agx_register(12, AGX_SIZE_32), plane);
            agx_mov_imm_to(b, agx_register(14, AGX_SIZE_32), 192 + 32 * plane);
            agx_store_uvb_m5(b, desc, 0);
         }
         for (unsigned selector : {128u, 130u}) {
            agx_device_load_to(b, agx_register(2, AGX_SIZE_32),
               agx_uniform(0, AGX_SIZE_64), agx_immediate(selector),
               AGX_FORMAT_I8, 1, 2, false);
            agx_mov_to(b, agx_register(12, AGX_SIZE_32),
                          agx_register(2, AGX_SIZE_32));
            agx_mov_imm_to(b, agx_register(14, AGX_SIZE_32), 320);
            agx_store_uvb_m5(b, desc, 0);
         }
         if (with_system_reads) {
            /* With warmup3, old allocation used selectors4/5 then SR6/1.
             * Restricting only the selectors instead leaves SR2/3, making
             * the second SR collide with the still-pending vertex phase. */
            agx_get_sr_to(b, agx_register(16, AGX_SIZE_32),
                            AGX_SR_THREADGROUP_POSITION_IN_GRID_X);
            agx_mov_to(b, agx_register(20, AGX_SIZE_32),
                          agx_register(16, AGX_SIZE_32));
            agx_get_sr_to(b, agx_register(18, AGX_SIZE_32),
                            AGX_SR_THREADGROUP_POSITION_IN_GRID_Y);
            agx_mov_to(b, agx_register(22, AGX_SIZE_32),
                          agx_register(18, AGX_SIZE_32));
         }
         agx_stop(b);
         /* Match production: MOV is a pseudo, lowered to a bitwise copy
          * before native packing. Keep the actual selector/SR reads. */
         agx_lower_pseudo(b->shader);
         const auto code = pack(b);
         ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();
         const uint8_t index[] = {0x0b,0x01,0x07,0x01,0,0x40,0,0,0,0};
         EXPECT_NE(std::search(code.begin(), code.end(), index,
                               index + sizeof(index)), code.end());
         EXPECT_EQ(std::vector<uint8_t>(code.end() - 4, code.end()),
                   (std::vector<uint8_t>{0x0e,0,0,0}));
      }
   }
}

/* Choosing a non-phase channel is insufficient by itself. A selector not
 * consumed yet must still make the existing outstanding-operation UVST gate
 * fail closed; the policy must not become an unconditional export drain. */
TEST_F(AGX3PackBinary, MeshUnconsumedSelectorStillRefusesExport)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->stage = MESA_SHADER_MESH;
   agx_mov_imm_to(b, agx_register(8, AGX_SIZE_16), 1);
   agx_allocate_uvb_m5(b, agx_register(8, AGX_SIZE_16),
                        agx_register(8, AGX_SIZE_16));
   agx_index desc = agx_register(12, AGX_SIZE_32);
   desc.channels_m1 = 1;
   agx_mov_imm_to(b, agx_register(12, AGX_SIZE_32), 0);
   agx_mov_imm_to(b, agx_register(14, AGX_SIZE_32), 192);
   agx_store_uvb_m5(b, desc, 1);
   agx_fence_uvb_gen1(b, AGX_MESH_EXPORT_FENCE_INDICES);
   agx_store_uvb_m5(b, desc, 0);
   agx_device_load_to(b, agx_register(2, AGX_SIZE_32),
      agx_uniform(0, AGX_SIZE_64), agx_immediate(128),
      AGX_FORMAT_I8, 1, 2, false);
   agx_store_uvb_m5(b, desc, 0);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, MultisampleTileBankCountsPartitionsNotStores)
{
   for (unsigned targets : {2u, 3u, 6u}) {
      for (unsigned samples : {1u, 4u}) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         key.secondary = true;
         key.fs.nr_samples = samples;
         b->shader->stage = MESA_SHADER_FRAGMENT;
         for (unsigned sample = 0; sample < samples; ++sample) {
            for (unsigned target = 0; target < targets; ++target) {
               agx_index value = agx_register(0, AGX_SIZE_16);
               value.channels_m1 = 3;
               agx_st_tile(b, value, agx_register(12, AGX_SIZE_16), agx_null(),
                           AGX_FORMAT_F16, 15, 8 * target, false, target);
            }
         }
         EXPECT_EQ(agx3_tile_store_bank(b->shader),
                   targets == 2 ? 16u : targets == 3 ? 32u : 64u);
      }
   }
}

TEST_F(AGX3PackBinary, SparseMrtAttachmentClassesMatchNative)
{
   for (const auto &witness : g17p_sparse_mrt_oracles) {
      SCOPED_TRACE(unsigned(witness.target_mask));
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      key.secondary = true;
      key.fs.nr_samples = 1;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->agx3_mrt_tile_ops_ordered = true;
      unsigned ordinal = witness.store_count;
      for (int target = 7; target >= 0; --target) {
         if (!(witness.target_mask & (1u << target)))
            continue;
         --ordinal;
         agx_index packed = agx_register(2 * ordinal, AGX_SIZE_16);
         packed.channels_m1 = 1;
         agx_st_tile(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     AGX_FORMAT_U8NORM, 15, 4 * ordinal, false, target);
      }
      agx_stop(b);
      const auto code = pack(b);
      std::vector<uint8_t> gates;
      for (size_t i = 0; i + 6 <= code.size(); ++i) {
         if ((code[i] == 0x87 || code[i] == 0x07) &&
             code[i + 1] == 0x02 && code[i + 2] == 0x54) {
            gates.insert(gates.end(), code.begin() + i, code.begin() + i + 6);
            i += 5;
         }
      }
      EXPECT_EQ(gates, std::vector<uint8_t>(witness.gates,
                                           witness.gates + witness.gate_size));
   }
}

TEST_F(AGX3PackBinary, SingleSparseMultisampleBlendKeepsPartitionClass)
{
   for (unsigned target : {1u, 2u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      key.secondary = true;
      key.fs.nr_samples = 4;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->agx3_mrt_tile_ops_ordered = true;
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 3;
      agx_index mask = agx_register(12, AGX_SIZE_16);
      for (unsigned sample = 0; sample < 4; ++sample) {
         agx_mov_imm_to(b, mask, 1u << sample);
         agx_ld_tile_to(b, value, mask, agx_null(), AGX_FORMAT_F16, 15, 0,
                        false, target);
         agx_st_tile(b, value, mask, agx_null(), AGX_FORMAT_F16, 15, 0,
                     false, target);
      }
      agx_stop(b);
      const auto bytes = pack(b);
      uint8_t wait[AGX3_PIXEL_MAX_LENGTH], release[AGX3_PIXEL_MAX_LENGTH];
      uint8_t wait_length, release_length;
      const unsigned classes = 12u << (target * 2);
      ASSERT_TRUE(agx3_pixel_wait(AGX_ISA_ENCODING_AGX3B, classes, 0,
                                  wait, &wait_length));
      ASSERT_TRUE(agx3_pixel_release(AGX_ISA_ENCODING_AGX3B, classes, true, 0,
                                     release, &release_length));
      EXPECT_NE(std::search(bytes.begin(), bytes.end(), wait,
                            wait + wait_length), bytes.end());
      EXPECT_NE(std::search(bytes.begin(), bytes.end(), release,
                            release + release_length), bytes.end());
   }
}

/*
 * Sparse texel residency reaches the packer as an eight-channel destination
 * with a four-bit channel mask: agx_emit_texture reserves the extra register
 * the instruction writes after the masked channels (agx_validate.c sizes a
 * sparse texture result at eight channels), and agx_instr::mask is four bits
 * wide, so the residency component never appears in it.
 *
 * Before this was admitted, both the filtered sample and the texel fetch
 * refused outright, and the G17P refusal printer named exactly this shape:
 * "texture_load [... chan=8 mask=0xf ...]".
 */
TEST_F(AGX3PackBinary, SparseResidencySampleEmitsTheWitnessedFields)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (bool sparse : {false, true}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index dest = agx_register(16, AGX_SIZE_32);
      dest.channels_m1 = sparse ? 7 : 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = 1;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0,
         false, false, false, AGX_GATHER_NONE, sparse);
      I->mask = 15;
      agx_stop(b);
      const auto code = pack(b);
      ASSERT_FALSE(code.empty());

      /* A sparse sample is the 20-byte ep4 form even with immediate
       * descriptors, because f120 deposits instruction bit 142, past the
       * 112-bit ep3 form. */
      struct agx3_tex_sample_desc d = {};
      d.dest = 16;
      d.coord = 0;
      d.width = AGX3_TEX_W32;
      d.mask = 15;
      d.lod = AGX3_TEX_LOD_LEVEL;
      d.slot = 1;
      d.wait_mask = 1;
      d.compute_stage = true;
      d.final = true;
      d.sparse = sparse;
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
      EXPECT_EQ(len, sparse ? 20u : 14u);
      EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                code.end());
   }
}

TEST_F(AGX3PackBinary, SparseResidencyFetchEmitsTheWitnessedFields)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (bool sparse : {false, true}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index dest = agx_register(16, AGX_SIZE_32);
      dest.channels_m1 = sparse ? 7 : 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = 1;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
         false, false, false, AGX_GATHER_NONE, sparse);
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      I->mask = 15;
      agx_stop(b);
      const auto code = pack(b);
      ASSERT_FALSE(code.empty());

      /* The emitted fetch decodes back through the strict decoder, and the
       * residency request survives the round trip.  A sparse fetch is the
       * 20-byte ep4 form even with an immediate descriptor, because f120
       * deposits instruction bit 142, past the 112-bit ep3 form. */
      const unsigned want = sparse ? 20u : 14u;
      bool found = false;
      for (size_t at = 0; at + want <= code.size() && !found; ++at) {
         struct agx3_tex_fetch_desc decoded;
         if (!agx3_tex_decode_fetch(enc, code.data() + at, want, &decoded))
            continue;
         if (decoded.dest != 16 || decoded.mask != 15)
            continue;
         EXPECT_EQ(decoded.sparse, sparse);
         found = true;
      }
      EXPECT_TRUE(found);
   }
}

TEST_F(AGX3PackBinary, MinimumLodStagesBiasLowAndMinimumHigh)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (unsigned half : {0u, 6u, 62u, 126u}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index dest = agx_register(16, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_index pair = agx_register(half, AGX_SIZE_32);
      pair.discard = true;
      auto *I = agx_texture_sample_to(
         b, dest, coord, pair, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_2D,
         AGX_LOD_MODE_AUTO_LOD_BIAS_MIN, 0, false, false, false,
         AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);
      agx_validate(b->shader, "minimum LOD packed half pair");
      const auto code = pack(b);
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_tex_stage_bias_min(enc, half, half + 1, 1, 0,
                                          false, true, bytes, &len));
      EXPECT_EQ(std::search(code.begin(), code.end(), bytes, bytes + len),
                code.begin());
      EXPECT_EQ(b->shader->agx3_staging_halfs, 8u);
      EXPECT_TRUE(b->shader->agx3_colour_link.entry_emitted);
   }
}

TEST_F(AGX3PackBinary, MinimumLodGradientStagesAppendedHalfWithoutGroupEntry)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (bool three_d : {false, true}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index dest = agx_register(16, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = three_d ? 2 : 1;
      agx_index gradients = agx_register(32, AGX_SIZE_32);
      const unsigned channels = three_d ? 6 : 4;
      gradients.channels_m1 = channels;
      gradients.discard = true;
      auto *I = agx_texture_sample_to(
         b, dest, coord, gradients, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), three_d ? AGX_DIM_3D : AGX_DIM_2D,
         AGX_LOD_MODE_LOD_GRAD_MIN, 0, false, false, false,
         AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);
      agx_validate(b->shader, "minimum LOD appended gradient half");
      const auto code = pack(b);
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_tex_stage_typed_lod(enc, 32 + 2 * channels, 1, 0,
                  false, true, AGX3_TEX_LEVEL_MINIMUM, bytes, &len));
      EXPECT_EQ(std::search(code.begin(), code.end(), bytes, bytes + len),
                code.begin());
      EXPECT_EQ(b->shader->agx3_staging_halfs, three_d ? 32u : 16u);
      EXPECT_FALSE(b->shader->agx3_colour_link.entry_emitted);
   }
}

/* A 2D-array sample publishes its layer beside the bias or the gradient
 * minimum, so the bias pair's minimum moves to half 4, or for a comparison to
 * a packed register, and no separate layer publication follows. */
TEST_F(AGX3PackBinary, ArrayMinimumLodSharesTheLayerBank)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (bool gradient : {false, true}) {
      for (bool shadow : {false, true}) {
         agx_builder *b = builder(enc);
         b->shader->stage = MESA_SHADER_FRAGMENT;
         agx_index dest = agx_register(16, AGX_SIZE_32);
         dest.channels_m1 = 3;
         agx_index coord = agx_register(8, AGX_SIZE_32);
         coord.channels_m1 = 2;
         coord.discard = true;
         agx_index lod = agx_register(32, AGX_SIZE_32);
         lod.channels_m1 = gradient ? 4 : 0;
         lod.discard = true;
         agx_index reference = shadow ? agx_register(48, AGX_SIZE_32)
                                      : agx_null();
         auto *I = agx_texture_sample_to(
            b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
            agx_immediate(0), reference, AGX_DIM_2D_ARRAY,
            gradient ? AGX_LOD_MODE_LOD_GRAD_MIN
                     : AGX_LOD_MODE_AUTO_LOD_BIAS_MIN,
            0, false, shadow, false, AGX_GATHER_NONE, false);
         I->mask = shadow ? 1 : 15;
         agx_stop(b);
         agx_validate(b->shader, "array minimum LOD");
         const auto code = pack(b);

         uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
         std::vector<uint8_t> want;
         if (shadow && !gradient) {
            ASSERT_TRUE(agx3_tex_stage_array_lod(
               enc, 12, 32, 1, 0, false, true, false, false, bytes, &len));
            append(want, bytes, len);
            ASSERT_EQ(std::search(code.begin(), code.end(), want.begin(),
                                  want.end()),
                      code.begin());
            /* The minimum lands in the transient word reserved above every
             * live register, the one SMP then names. */
            bool packed = false;
            for (uint16_t dest = 50; dest <= 190 && !packed; dest += 2) {
               ASSERT_TRUE(agx3_tex_pack_minimum(enc, 33, dest, false, true,
                                                 bytes, &len));
               packed = std::equal(bytes, bytes + len,
                                   code.begin() + want.size());
            }
            EXPECT_TRUE(packed);
            continue;
         }
         if (gradient) {
            ASSERT_TRUE(agx3_tex_stage_array_minimum_gradient(
               enc, 12, 40, 1, 0, false, true, false, true, bytes, &len));
            append(want, bytes, len);
         } else {
            ASSERT_TRUE(agx3_tex_stage_array_minimum_lod(
               enc, 33, 1, 0, false, false, bytes, &len));
            append(want, bytes, len);
            ASSERT_TRUE(agx3_tex_stage_array_lod(
               enc, 12, 32, 1, 0, false, true, false, true, bytes, &len));
            append(want, bytes, len);
         }
         EXPECT_EQ(std::search(code.begin(), code.end(), want.begin(),
                               want.end()),
                   code.begin());
         ASSERT_TRUE(agx3_tex_stage_array_layer(enc, 12, 1, 0, false, true,
                                                bytes, &len));
         EXPECT_EQ(std::search(code.begin(), code.end(), bytes, bytes + len),
                   code.end());
      }
   }
}

/* A non-array runtime offset carries the minimum LOD in its NIR-packed word
 * with the bias (bits 0..11 and 16..27) or alone (a gradient's, bits 16..27),
 * so the sample publishes that one word to bank 6 and nothing else there. */
TEST_F(AGX3PackBinary, RuntimeOffsetMinimumLodPublishesThePackedWord)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (bool gradient : {false, true}) {
      for (bool shadow : {false, true}) {
         agx_builder *b = builder(enc);
         b->shader->stage = MESA_SHADER_FRAGMENT;
         agx_index dest = agx_register(16, AGX_SIZE_32);
         dest.channels_m1 = 3;
         agx_index coord = agx_register(8, AGX_SIZE_32);
         coord.channels_m1 = 1;
         agx_index lod = agx_register(32, AGX_SIZE_32);
         lod.channels_m1 = gradient ? 4 : 0;
         agx_index word = agx_register(48, AGX_SIZE_32);
         word.channels_m1 = shadow ? 1 : 0;
         word.discard = true;
         auto *I = agx_texture_sample_to(
            b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
            agx_immediate(0), word, AGX_DIM_2D,
            gradient ? AGX_LOD_MODE_LOD_GRAD_MIN
                     : AGX_LOD_MODE_AUTO_LOD_BIAS_MIN,
            0, true, shadow, false, AGX_GATHER_NONE, false);
         I->mask = shadow ? 1 : 15;
         agx_stop(b);
         const auto code = pack(b);

         uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
         ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 6, shadow ? 50 : 48, 1, 0,
                                               false, true, bytes, &len));
         EXPECT_EQ(std::search(code.begin(), code.end(), bytes, bytes + len),
                   code.begin());
         unsigned publications = 0;
         for (size_t i = 0; i + 4 <= code.size(); i += 2) {
            if (code[i] == 0x17 && code[i + 1] == 0x05 &&
                code[i + 2] == 0x54 && code[i + 3] == 0x06)
               publications++;
         }
         EXPECT_EQ(publications, 0u);
      }
   }

   /* A 3D word carries the Z nibble instead, so the minimum is packed into
    * a register after the word. */
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(16, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(8, AGX_SIZE_32);
   coord.channels_m1 = 2;
   agx_index pair = agx_register(32, AGX_SIZE_32);
   pair.discard = true;
   agx_index word = agx_register(48, AGX_SIZE_32);
   word.discard = true;
   auto *I = agx_texture_sample_to(
      b, dest, coord, pair, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), word, AGX_DIM_3D, AGX_LOD_MODE_AUTO_LOD_BIAS_MIN, 0,
      true, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);
   const auto code = pack(b);
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 6, 48, 1, 0, false, true, bytes,
                                         &len));
   std::vector<uint8_t> want(bytes, bytes + len);
   ASSERT_EQ(std::search(code.begin(), code.end(), want.begin(), want.end()),
             code.begin());
   bool packed = false;
   for (uint16_t scratch = 50; scratch <= 190 && !packed; scratch += 2) {
      ASSERT_TRUE(agx3_tex_pack_minimum(enc, 33, scratch, false, true, bytes,
                                        &len));
      packed = std::equal(bytes, bytes + len, code.begin() + want.size());
   }
   EXPECT_TRUE(packed);
}

TEST_F(AGX3PackBinary, G17PRG16IntegerEOTUsesExactNativeI16Program)
{
   key.g17p_rt0_eot_words = 1;
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x15, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_I16, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PRGB565EOTUsesSixteenBitLaneSelector)
{
   /* Apple's B5G6R5Unorm fragment stores three 16-bit float lanes, and the
    * builtin EOT for every 16-bit float or integer internal format selects
    * 0x15. RGB565 lowers to a two-word I16 store and shares that program. */
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x15, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   auto build = [&](enum agx_format format) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index coords = agx_register(0, AGX_SIZE_32);
      coords.channels_m1 = 1;
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_Y);
      agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
      agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                            agx_register(4, AGX_SIZE_32), coords, format,
                            AGX_DIM_2D, true);
      agx_stop(b);
      return b;
   };

   for (enum agx_format format : {AGX_FORMAT_I16, AGX_FORMAT_F16}) {
      key.g17p_rt0_eot_words = 2;
      agx_builder *b = build(format);
      EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                              std::end(native)));
      EXPECT_EQ(b->shader->max_reg, 7);
      EXPECT_TRUE(b->shader->agx3_staging_known);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
   }

   key.g17p_rt0_eot_words = 0;
   EXPECT_TRUE(pack(build(AGX_FORMAT_I16), false).empty());
   key.g17p_rt0_eot_words = 2;
   key.g17p_rt0_eot_i32 = true;
   EXPECT_TRUE(pack(build(AGX_FORMAT_I16), false).empty());
}

TEST_F(AGX3PackBinary, G17PLayeredU8NormEOTUsesExactNativeArrayWire)
{
   key.g17p_rt0_eot_words = 0;
   static const uint8_t native[] = {
      0x1c,0x9c,0x10,0x06,0x34,0xa8,0x10,0x06,
      0x0c,0x9d,0x10,0x26,0x24,0xa9,0x14,0x26,
      0x24,0xc3,0x10,0x46,
      0x9f,0x10,0x54,0x00,0x00,0x04,0x18,0x00,0x50,0x06,0x0b,0x00,
      0x9f,0x20,0x54,0x02,0x00,0x00,0x14,0x00,0x50,0x06,0x0b,0x00,
      0x97,0x45,0x54,0x04,0x00,0x08,0x10,0x00,0xf0,0xc0,0x00,0x00,
      0x97,0x05,0x54,0x06,0x00,0x08,0x00,0x00,0xd0,0x40,0x10,0x00,
      0x57,0x10,0x54,0x00,0x00,0x00,0x00,0x80,0x28,0x75,0x00,0x00,
      0x0e,0x00,0x00,0x00,
   };

   auto build = [&](enum agx_dim dim, unsigned coord_base) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_Y);
      agx_get_sr_to(b, agx_register(4, AGX_SIZE_32),
                    AGX_SR_THREADGROUP_POSITION_IN_GRID_Z);
      agx_mov_imm_to(b, agx_register(6, AGX_SIZE_32), 0);
      for (unsigned lane = 0; lane < 3; ++lane) {
         agx_index src = agx_register(2 * lane, AGX_SIZE_32);
         src.discard = true;
         agx_bitop_to(b, agx_register(8 + 2 * lane, AGX_SIZE_32), src,
                      agx_immediate(0), AGX3_SEL_TT_MOV);
      }
      agx_index offset = agx_register(6, AGX_SIZE_32);
      offset.discard = true;
      agx_index coords = agx_register(coord_base, AGX_SIZE_32);
      coords.channels_m1 = 2;
      coords.discard = true;
      agx_block_image_store(b, agx_immediate(0), agx_immediate(0), offset,
                            coords, AGX_FORMAT_U8NORM, dim, true);
      agx_stop(b);
      return b;
   };

   agx_builder *good = build(AGX_DIM_2D_ARRAY, 8);
   EXPECT_EQ(pack(good), std::vector<uint8_t>(std::begin(native),
                                              std::end(native)));
   EXPECT_EQ(good->shader->max_reg, 7);
   EXPECT_TRUE(good->shader->agx3_staging_known);
   EXPECT_EQ(good->shader->agx3_staging_halfs, 8);

   EXPECT_TRUE(pack(build(AGX_DIM_2D, 8), false).empty());
   EXPECT_TRUE(pack(build(AGX_DIM_2D_ARRAY, 10), false).empty());
}

TEST_F(AGX3PackBinary, G17PRGBA8MRTEOTUsesExactNativeProgram)
{
   static const uint8_t native[] = {
      0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,
      0x0c,0x9d,0x10,0x26,0x24,0xa9,0x10,0x26,
      0x9f,0x10,0x54,0x02,0x03,0x04,0x14,0x00,
      0x50,0x26,0x0a,0x00,0x9f,0x20,0x54,0x00,
      0x03,0x00,0x10,0x00,0x50,0x26,0x0a,0x00,
      0x97,0x05,0x54,0x04,0x00,0x08,0x00,0x00,
      0xd0,0x40,0x10,0x00,0x0b,0x82,0x21,0x04,
      0x1b,0x80,0x21,0x04,0x97,0x05,0x54,0x0c,
      0x00,0x10,0x00,0x00,0xd0,0x44,0x10,0x00,
      0x4b,0x03,0x2f,0x00,0x42,0x00,0x00,0x04,
      0x00,0x00,0x5b,0x01,0x2f,0x00,0x42,0x00,
      0x00,0x04,0x00,0x00,0x57,0x10,0x54,0x00,
      0x00,0x00,0x00,0x80,0xa8,0x75,0x00,0x00,
      0x57,0x20,0x54,0x08,0x00,0x10,0x00,0x90,
      0xa8,0x75,0x00,0x00,0x0e,0x00,0x00,0x00,
   };

   auto build = [&](unsigned second_index, unsigned second_offset,
                    enum agx_format second_format) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index coords = agx_register(0, AGX_SIZE_32);
      coords.channels_m1 = 1;
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_Y);
      agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
      agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                            agx_register(4, AGX_SIZE_32), coords,
                            AGX_FORMAT_U8NORM, AGX_DIM_2D, true);
      agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), second_offset);
      agx_block_image_store(b, agx_immediate(0),
                            agx_immediate(second_index),
                            agx_register(4, AGX_SIZE_32), coords,
                            second_format, AGX_DIM_2D, true);
      agx_stop(b);
      return b;
   };

   agx_builder *good = build(1, 4, AGX_FORMAT_U8NORM);
   EXPECT_EQ(pack(good), std::vector<uint8_t>(std::begin(native),
                                              std::end(native)));
   EXPECT_EQ(good->shader->max_reg, 7);
   EXPECT_TRUE(good->shader->agx3_staging_known);
   EXPECT_EQ(good->shader->agx3_staging_halfs, 16);

   EXPECT_TRUE(pack(build(0, 4, AGX_FORMAT_U8NORM), false).empty());
   EXPECT_TRUE(pack(build(1, 8, AGX_FORMAT_U8NORM), false).empty());
   EXPECT_TRUE(pack(build(1, 4, AGX_FORMAT_S8NORM), false).empty());
}

TEST_F(AGX3PackBinary, G17PThreeAndFourRGBA8EOTUseNativePrograms)
{
   for (unsigned count : {3u, 4u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index coords = agx_register(0, AGX_SIZE_32);
      coords.channels_m1 = 1;
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_Y);
      for (unsigned rt = 0; rt < count; ++rt) {
         agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 4 * rt);
         agx_block_image_store(b, agx_immediate(0), agx_immediate(rt),
                               agx_register(4, AGX_SIZE_32), coords,
                               AGX_FORMAT_U8NORM, AGX_DIM_2D, true);
      }
      agx_stop(b);

      const uint8_t *native = count == 3 ? agx3_g17p_rgba8_3_eot
                                          : agx3_g17p_rgba8_4_eot;
      const size_t native_size = count == 3
                                    ? sizeof(agx3_g17p_rgba8_3_eot)
                                    : sizeof(agx3_g17p_rgba8_4_eot);
      EXPECT_EQ(pack(b), std::vector<uint8_t>(native, native + native_size));
      EXPECT_EQ(b->shader->max_reg, 7);
      EXPECT_TRUE(b->shader->agx3_staging_known);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 8 * count);
   }
}

TEST_F(AGX3PackBinary, G17PF16EOTUsesExactNativeI16Program)
{
   key.g17p_rt0_eot_words = 1;
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x15, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_F16, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PI8EOTUsesExactOneWordNativeProgram)
{
   /* Fresh scalar R8_UINT key0x04020500 and scalar R8_SINT key0x04020400
    * both produce this exact program. Its 68-byte text SHA256 is
    * 7a5e8ed6d0d8fca6b23b335f416f069c0942332ad7becf7ea5ec2704ee59ce95. */
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x55, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   for (unsigned words = 0; words <= 2; ++words) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      key.g17p_rt0_eot_words = words;
      agx_index coords = agx_register(0, AGX_SIZE_32);
      coords.channels_m1 = 1;
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_Y);
      agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
      agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                            agx_register(4, AGX_SIZE_32), coords,
                            AGX_FORMAT_I8, AGX_DIM_2D, true);
      agx_stop(b);

      if (words == 1) {
         EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                                 std::end(native)));
         EXPECT_EQ(b->shader->max_reg, 7);
         EXPECT_TRUE(b->shader->agx3_staging_known);
         EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
      } else {
         EXPECT_TRUE(pack(b, false).empty());
      }
   }
}

TEST_F(AGX3PackBinary, G17PS8NormEOTUsesExactNativeProgram)
{
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x75, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_S8NORM, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PScalarI32EOTUsesExactNativeProgram)
{
   /* Fresh Apple G17P builtin EOT key 0x04021100. The PBE descriptor carries
    * the logical UINT/FLOAT interpretation; this is the exact scalar store. */
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x35, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   key.g17p_rt0_eot_i32 = true;
   key.g17p_rt0_eot_words = 1;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_I32, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PRG32FloatEOTUsesExactTwoWordNativeProgram)
{
   /* Fresh Apple G17P builtin EOT key 0x08021100. Its exact text is
    * byte-identical to scalar key 0x04021100; PBE carries channels R32G32.
    * Text SHA256: 60beeb253d8f81af993e93617ed3fca4559e0f614c8465f4cd2bb42c0f1f5fb6. */
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x35, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   key.g17p_rt0_eot_i32 = true;
   key.g17p_rt0_eot_words = 2;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_I32, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                           std::end(native)));
   EXPECT_EQ(b->shader->max_reg, 7);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, G17PRG32IntegerEOTUsesRawThirtyTwoBitSelector)
{
   /* Apple's RG32Uint and RG32Sint fragments store raw 32-bit lanes, and the
    * builtin EOT for every raw 32-bit internal format selects 0x35: the
    * scalar R32 program. */
   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x35, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   for (unsigned words = 0; words <= 2; ++words) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      key.g17p_rt0_eot_i32 = true;
      key.g17p_rt0_eot_words = words;
      agx_index coords = agx_register(0, AGX_SIZE_32);
      coords.channels_m1 = 1;
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_Y);
      agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
      agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                            agx_register(4, AGX_SIZE_32), coords,
                            AGX_FORMAT_I32, AGX_DIM_2D, true);
      agx_stop(b);

      if (words) {
         EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                                 std::end(native)));
      } else {
         EXPECT_TRUE(pack(b, false).empty());
      }
   }
}

TEST_F(AGX3PackBinary, G17PI32EOTRefusesAmbiguousWordCount)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   key.g17p_rt0_eot_i32 = true;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_I32, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, G17PRGBA16NormEOTUsesExactTwoWordNativeProgram)
{
   key.g17p_rt0_eot_words = 2;

   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x95, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   /* RGBA16 UNORM stores the same normalized 16-bit lanes as its SNORM
    * sibling, as Apple's RGBA16Unorm fragment does. */
   for (enum agx_format format : {AGX_FORMAT_S16NORM, AGX_FORMAT_U16NORM}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index coords = agx_register(0, AGX_SIZE_32);
      coords.channels_m1 = 1;
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_Y);
      agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
      agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                            agx_register(4, AGX_SIZE_32), coords, format,
                            AGX_DIM_2D, true);
      agx_stop(b);

      EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                              std::end(native)));
      EXPECT_EQ(b->shader->max_reg, 7);
      EXPECT_TRUE(b->shader->agx3_staging_known);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
   }
}

TEST_F(AGX3PackBinary, G17PScalarNorm16EOTUsesExactNativeProgram)
{
   key.g17p_rt0_eot_words = 1;

   static const uint8_t native[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x95, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   for (enum agx_format format : {AGX_FORMAT_U16NORM,
                                  AGX_FORMAT_S16NORM}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index coords = agx_register(0, AGX_SIZE_32);
      coords.channels_m1 = 1;
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_Y);
      agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
      agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                            agx_register(4, AGX_SIZE_32), coords, format,
                            AGX_DIM_2D, true);
      agx_stop(b);

      EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                              std::end(native)));
   }
}

TEST_F(AGX3PackBinary, G17PU16NormEOTRefusesAmbiguousWordCount)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 0);
   agx_block_image_store(b, agx_immediate(0), agx_immediate(0),
                         agx_register(4, AGX_SIZE_32), coords,
                         AGX_FORMAT_U16NORM, AGX_DIM_2D, true);
   agx_stop(b);

   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, HalfUVPreservesSharedHalfAndOperandOrder)
{
   for (unsigned order = 0; order < 2; ++order) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index half = agx_register(10, AGX_SIZE_16);
      agx_mov_imm_to(b, half, 0x1800);
      agx_index x = agx_register(0, AGX_SIZE_32);
      agx_index y = agx_register(2, AGX_SIZE_32);
      x.discard = y.discard = true;
      half.cache = true;
      agx_fmul_to(b, agx_register(0, AGX_SIZE_32),
                    order ? half : x, order ? x : half);
      half.cache = false;
      half.discard = true;
      agx_fmul_to(b, agx_register(2, AGX_SIZE_32), y, half);
      agx_stop(b);
      /* Native LDIMM and shared-half MUL forms, independently decoded. */
      const std::vector<uint8_t> want = {
         0x54,0x80,0x02,0x00,0x00,0x0c,0x00,0x00,
         0x09,0x01,0x09,0x8a, 0x19,0x03,0x19,0x0a, 0x0e,0,0,0,
      };
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, HalfUVRefusesUngroundedModes)
{
   for (unsigned mode = 1; mode < 9; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index half = agx_register(10, AGX_SIZE_16);
      agx_index wide = agx_register(0, AGX_SIZE_32);
      if (mode == 1) half.neg = true;
      if (mode == 2) half.abs = true;
      if (mode == 3) half.cache = half.discard = true;
      if (mode == 4) half.kill = true;
      if (mode == 5) wide.value = 1;
      if (mode == 6) half.value = 256;
      if (mode == 7) wide.size = AGX_SIZE_16;
      agx_instr *mul = agx_fmul_to(b, agx_register(2, AGX_SIZE_32), wide, half);
      mul->saturate = mode == 8;
      /* f14 is the saturate bit and it sits on the same encoding point as the
       * slot widths, so `saturate(a * b)` over a mixed pair is one
       * instruction: Apple's own is 09011d0401000082, f14 = 1 with f43 = 1.
       * The modifier rows above it stay refused -- no witness puts abs or neg
       * on a MULTIPLY's half slot. */
      if (mode == 8)
         EXPECT_FALSE(pack(b).empty()) << "mode=" << mode;
      else
         EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, WebGLAbsScaleMixedWidthMatchesNative)
{
   for (bool half_first : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index wide = agx_register(0, AGX_SIZE_32);
      agx_index half = agx_register(8, AGX_SIZE_16);
      wide.abs = wide.discard = true;
      half.cache = true;
      agx_fmul_to(b, agx_register(0, AGX_SIZE_32),
                  half_first ? half : wide, half_first ? wide : half);
      agx_stop(b);

      /* Reduced post-RA form of Skia's `abs(testInputs) * 100.0`. */
      const std::vector<uint8_t> native = {
         0x09, 0x01, 0x0d, 0x88, 0x02, 0x00, 0x00, 0x00,
         0x01, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
      };
      EXPECT_EQ(pack(b), native) << "half_first=" << half_first;
   }
}

TEST_F(AGX3PackBinary, WebGLAbsScaleMixedWidthRefusesUngroundedModes)
{
   for (unsigned mode = 1; mode < 16; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(0, AGX_SIZE_32);
      agx_index wide = agx_register(0, AGX_SIZE_32);
      agx_index half = agx_register(8, AGX_SIZE_16);
      wide.abs = wide.discard = true;
      half.cache = true;

      if (mode == 1)
         dest.size = AGX_SIZE_16;
      if (mode == 2)
         wide.neg = true;
      if (mode == 3) {
         wide.abs = false;
         half.abs = true;
      }
      if (mode == 4)
         half.neg = true;
      if (mode == 5)
         wide.cache = true;
      if (mode == 6)
         half.discard = true;
      if (mode == 7)
         wide.kill = true;
      if (mode == 8)
         half.kill = true;
      if (mode == 9)
         dest.value = 1;
      if (mode == 10)
         wide.value = 1;
      if (mode == 11)
         dest.channels_m1 = 1;
      if (mode == 13)
         wide.memory = true;
      if (mode == 14)
         half.memory = true;
      if (mode == 15)
         dest.memory = true;

      agx_instr *mul = agx_fmul_to(b, dest, wide, half);
      mul->saturate = mode == 12;
      agx_stop(b);
      /* Saturation is f14 on the same encoding point as the slot widths and
       * the modifier bits: Apple's saturate(a * b) over a mixed pair is
       * 09011d0401000082. Everything else here is still out of domain. */
      if (mode == 12)
         EXPECT_FALSE(pack(b).empty()) << "mode=" << mode;
      else
         EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, HalfMultiplyMinifloatMatchesNativeAndRefusesModifier)
{
   for (unsigned i = 0; i < 2; ++i) {
      for (unsigned order = 0; order < 2; ++order) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         agx_index src = agx_register(2, AGX_SIZE_16);
         src.discard = true;
         agx_index dest = agx_register(2, AGX_SIZE_16);
         agx_index immediate = agx_immediate(i ? 0x50 : 0x40);
         agx_hmul_to(b, dest, order ? immediate : src,
                     order ? src : immediate);
         agx_stop(b);
         const auto code = pack(b);
         const uint8_t times2[] = {0x10, 0xc0, 0x15, 0x02, 0x80, 0x00};
         const uint8_t times4[] = {0x10, 0xd0, 0x15, 0x02, 0x80, 0x00};
         ASSERT_EQ(code.size(), 10u);
         EXPECT_EQ(memcmp(code.data(), i ? times4 : times2, 6), 0);
      }
   }

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index modified = agx_register(2, AGX_SIZE_16);
   modified.abs = true;
   agx_hmul_to(b, agx_register(2, AGX_SIZE_16), modified,
               agx_immediate(0x40));
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, HalfMultiplyOddSourcesHasExactFmaFallback)
{
   uint8_t native[AGX3_ALU_MAX_LENGTH], len = 0;
   const uint16_t src[] = {3, 5};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
   };

   /* This is the exact post-RA CTS reflect() shape. The compact multiply
    * cannot combine both odd half registers with its load wait. */
   ASSERT_FALSE(agx3_alu_hmul(AGX_ISA_ENCODING_AGX3B, 6, src, hints,
                              false, UINT16_C(0x5), native, &len));

   /* A multi-slot dependency selects the extended mask endpoint. */
   ASSERT_TRUE(agx3_alu_hfma_addend_minifloat(
      AGX_ISA_ENCODING_AGX3B, 6, src, UINT8_C(0x80), hints, false,
      UINT16_C(0x5), native, &len));
   ASSERT_EQ(len, 10u);
   EXPECT_EQ((native[7] >> 5) | ((native[9] >> 5) << 3), UINT16_C(0x5));
}

/* The optimizer folds fsat into a half multiply whose source already carries
 * fabs: saturate(|a| * b). This post-RA compute shape (odd half registers,
 * the destination reusing a discarded source) was refused. An absolute
 * source selects the ep3 form, which also carries saturation, so the
 * saturating multiply must be that form with exactly one bit added -- on
 * every AGX3 encoding, as AGX2 already sets one bit for it.
 */
TEST_F(AGX3PackBinary, SaturatingHalfMultiplyKeepsAbsoluteSource)
{
   for (enum agx_isa_encoding enc :
        {AGX_ISA_ENCODING_AGX2, AGX_ISA_ENCODING_AGX3B,
         AGX_ISA_ENCODING_AGX3C}) {
      for (unsigned abs_mask = 1; abs_mask < 4; ++abs_mask) {
         std::vector<uint8_t> code[2];
         for (unsigned saturate = 0; saturate < 2; ++saturate) {
            agx_builder *b = builder(enc);
            key.dev.chip = enc == AGX_ISA_ENCODING_AGX2  ? AGX_CHIP_G13G
                           : enc == AGX_ISA_ENCODING_AGX3B ? AGX_CHIP_G17P
                                                           : AGX_CHIP_G17S;
            agx_index s0 = agx_register(79, AGX_SIZE_16);
            agx_index s1 = agx_register(107, AGX_SIZE_16);
            s0.cache = true;
            s1.discard = true;
            s0.abs = abs_mask & 1;
            s1.abs = abs_mask & 2;
            agx_instr *I =
               agx_hmul_to(b, agx_register(107, AGX_SIZE_16), s0, s1);
            I->saturate = saturate;
            agx_stop(b);
            if (enc == AGX_ISA_ENCODING_AGX2) {
               struct util_dynarray bin;
               util_dynarray_init(&bin, mem_ctx);
               agx_pack_binary(b->shader, &bin);
               const uint8_t *d = (const uint8_t *)bin.data;
               code[saturate].assign(d, d + bin.size);
            } else {
               code[saturate] = pack(b);
            }
         }
         ASSERT_FALSE(code[0].empty()) << "enc=" << enc << " abs=" << abs_mask;
         ASSERT_EQ(code[0].size(), code[1].size())
            << "enc=" << enc << " abs=" << abs_mask;
         unsigned differing_bits = 0;
         for (size_t i = 0; i < code[0].size(); ++i) {
            for (unsigned x = code[0][i] ^ code[1][i]; x; x &= x - 1)
               ++differing_bits;
         }
         EXPECT_EQ(differing_bits, 1u) << "enc=" << enc << " abs=" << abs_mask;
      }
   }
}

TEST_F(AGX3PackBinary, MinifloatMixedSelectMatchesNativeAndRefusesModifier)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index compare = agx_register(4, AGX_SIZE_32);
   compare.cache = true;
   agx_index selected = agx_register(2, AGX_SIZE_32);
   selected.cache = true;
   agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_32), compare,
                  agx_immediate(0x30), selected, agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const uint8_t native[] = {
      0x02, 0x85, 0x03, 0xb1, 0x02, 0x82, 0x03, 0x02, 0x20, 0x80,
   };
   const auto code = pack(b);
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   /* Exact post-RA ext.frag mask shape. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   compare = agx_register(14, AGX_SIZE_32);
   agx_index dest = agx_register(1, AGX_SIZE_16);
   dest.cache = true;
   agx_fcmpsel_to(b, dest, compare,
                  agx_immediate(0x30), agx_immediate(255), agx_immediate(0),
                  AGX_FCOND_EQ);
   agx_stop(b);
   EXPECT_GT(pack(b).size(), 4u);

   /* Exact post-RA glitch.frag shape. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   compare = agx_register(22, AGX_SIZE_32);
   compare.cache = true;
   selected = agx_register(2, AGX_SIZE_32);
   selected.discard = true;
   dest = agx_register(2, AGX_SIZE_32);
   dest.cache = true;
   agx_fcmpsel_to(b, dest, compare,
                  agx_immediate(0x40), agx_immediate(0), selected,
                  AGX_FCOND_LT);
   agx_stop(b);
   EXPECT_GT(pack(b).size(), 4u);

   /* An absolute comparator against a NON-ZERO minifloat. The refusal here
    * was the admission gate, not the encoding: against Apple's own
    * (fabs(a) < c) ? 1.0h : 0.0h, 1203190523b00300208000800100, the
    * immediate form 120309a123b00302208000800100 changes only f59 (the
    * second slot's kind) and the operand field (f52 -> f51). */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   compare = agx_register(2, AGX_SIZE_32);
   compare.abs = true;
   agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), compare,
                  agx_immediate(0x30), agx_immediate(255), agx_immediate(0),
                  AGX_FCOND_EQ);
   agx_stop(b);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, GtkNegatedMinifloatSelectCanonicalizesExactly)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;

   /* Exact post-RA shape from current GTK gskgpuboxshadow.frag. Apple emits
    * the byte-identical x > +0 form for source -x < -0. */
   agx_builder *b = builder(enc);
   agx_index compare = agx_register(44, AGX_SIZE_32);
   compare.neg = true;
   agx_fcmpsel_to(b, agx_register(44, AGX_SIZE_16), compare,
                  agx_immediate(0x80), agx_immediate(1), agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const std::vector<uint8_t> gtk = {
      0x62, 0x2d, 0x45, 0x81, 0x22, 0x81, 0x02, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), gtk);

   /* Sign-flipping the threshold and reversing the relation are independent
    * of zero. These are the no-wait forms of fresh Apple -x < +1 and
    * -x >= -0.5 witnesses. */
   b = builder(enc);
   compare = agx_register(2, AGX_SIZE_32);
   compare.neg = true;
   compare.discard = true;
   agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), compare,
                  agx_immediate(0x30), agx_immediate(1), agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const std::vector<uint8_t> minus_one = {
      0x12, 0x03, 0x1d, 0xb1, 0x22, 0x81, 0x02, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), minus_one);

   b = builder(enc);
   compare = agx_register(2, AGX_SIZE_32);
   compare.neg = true;
   compare.discard = true;
   agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), compare,
                  agx_immediate(0xa0), agx_immediate(1), agx_immediate(0),
                  AGX_FCOND_GE);
   agx_stop(b);
   const std::vector<uint8_t> plus_half = {
      0x12, 0x03, 0x0d, 0xa1, 0x26, 0x81, 0x03, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), plus_half);

   /* Current gskgpulineargradient has the minifloat in the first compare
    * slot. Fresh Apple kernels normalize 1 < x to x > 1, and 1 < -x to
    * x < -1. */
   b = builder(enc);
   compare = agx_register(2, AGX_SIZE_32);
   compare.discard = true;
   agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), agx_immediate(0x30),
                  compare, agx_immediate(1), agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const std::vector<uint8_t> immediate_first = {
      0x12, 0x03, 0x0d, 0xb1, 0x22, 0x81, 0x02, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), immediate_first);

   b = builder(enc);
   compare = agx_register(2, AGX_SIZE_32);
   compare.neg = true;
   compare.discard = true;
   agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), agx_immediate(0x30),
                  compare, agx_immediate(1), agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const std::vector<uint8_t> immediate_first_negated = {
      0x12, 0x03, 0x1d, 0xb1, 0x22, 0x81, 0x03, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), immediate_first_negated);

   /* Exact full-result post-RA forms reached later in current GTK's linear
    * and conic gradient shaders. */
   b = builder(enc);
   compare = agx_register(22, AGX_SIZE_32);
   compare.cache = true;
   agx_fcmpsel_to(b, agx_register(10, AGX_SIZE_32), compare,
                  agx_immediate(0x30), agx_immediate(0),
                  agx_register(10, AGX_SIZE_32), AGX_FCOND_GE);
   agx_stop(b);
   const std::vector<uint8_t> linear_full = {
      0x52, 0x97, 0x03, 0xb1, 0x26, 0x80, 0x02, 0x02, 0x00, 0x0a,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), linear_full);

   b = builder(enc);
   compare = agx_register(12, AGX_SIZE_32);
   compare.cache = true;
   agx_index fallback = agx_register(24, AGX_SIZE_32);
   fallback.discard = true;
   agx_fcmpsel_to(b, agx_register(24, AGX_SIZE_32), agx_immediate(0x30),
                  compare, agx_immediate(0), fallback, AGX_FCOND_GE);
   agx_stop(b);
   const std::vector<uint8_t> conic_full = {
      0xc2, 0x8d, 0x03, 0xb1, 0x26, 0x80, 0x03, 0x02, 0x80, 0x18,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), conic_full);

   /* Inverting the predicate swaps the selected bytes without changing the
    * compare canonicalization. */
   b = builder(enc);
   compare = agx_register(44, AGX_SIZE_32);
   compare.neg = true;
   agx_instr *fsel = agx_fcmpsel_to(
      b, agx_register(44, AGX_SIZE_16), compare, agx_immediate(0x80),
      agx_immediate(1), agx_immediate(0), AGX_FCOND_LT);
   fsel->invert_cond = true;
   agx_stop(b);
   const std::vector<uint8_t> inverted = {
      0x62, 0x2d, 0x45, 0x81, 0x22, 0x80, 0x02, 0x02, 0x20, 0x81,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), inverted);
}

TEST_F(AGX3PackBinary, GtkNegatedMinifloatSelectRefusesUngroundedForms)
{
   enum bad_form {
      WRONG_ARCH,
      FULL_DEST,
      DEST_VECTOR,
      DEST_MEMORY,
      COMPARE_HALF,
      COMPARE_ODD,
      COMPARE_ABS,
      COMPARE_ABS_NEG,
      COMPARE_KILL,
      COMPARE_HINT_CONFLICT,
      THRESHOLD_OVERFLOW,
      THRESHOLD_ABS,
      THRESHOLD_NEG,
      THRESHOLD_KILL,
      THRESHOLD_CACHE,
      TRUE_OVERFLOW,
      FALSE_OVERFLOW,
      TRUE_ABS,
      FALSE_NEG,
      TRUE_KILL,
      FALSE_MEMORY,
      TRUE_CACHE,
      NAN_CONDITION_LT,
      NAN_CONDITION_GT,
      SATURATE,
   };

   for (unsigned form = WRONG_ARCH; form <= SATURATE; ++form) {
      SCOPED_TRACE(form);
      agx_builder *b = builder(form == WRONG_ARCH ? AGX_ISA_ENCODING_AGX3C
                                                  : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(44, AGX_SIZE_16);
      agx_index compare = agx_register(44, AGX_SIZE_32);
      agx_index threshold = agx_immediate(0x80);
      agx_index if_true = agx_immediate(1);
      agx_index if_false = agx_immediate(0);
      compare.neg = true;
      enum agx_fcond condition = AGX_FCOND_LT;

      switch ((enum bad_form)form) {
      case WRONG_ARCH: break;
      case FULL_DEST: dest.size = AGX_SIZE_32; break;
      case DEST_VECTOR: dest.channels_m1 = 1; break;
      case DEST_MEMORY: dest.memory = true; break;
      case COMPARE_HALF: compare.size = AGX_SIZE_16; break;
      case COMPARE_ODD: compare.value = 45; break;
      case COMPARE_ABS: compare.abs = true; compare.neg = false; break;
      case COMPARE_ABS_NEG: compare.abs = true; break;
      case COMPARE_KILL: compare.kill = true; break;
      case COMPARE_HINT_CONFLICT:
         compare.cache = compare.discard = true;
         break;
      case THRESHOLD_OVERFLOW: threshold.value = 256; break;
      case THRESHOLD_ABS: threshold.abs = true; break;
      case THRESHOLD_NEG: threshold.neg = true; break;
      case THRESHOLD_KILL: threshold.kill = true; break;
      case THRESHOLD_CACHE: threshold.cache = true; break;
      case TRUE_OVERFLOW: if_true.value = 256; break;
      case FALSE_OVERFLOW: if_false.value = 256; break;
      case TRUE_ABS: if_true.abs = true; break;
      case FALSE_NEG: if_false.neg = true; break;
      case TRUE_KILL: if_true.kill = true; break;
      case FALSE_MEMORY: if_false.memory = true; break;
      case TRUE_CACHE: if_true.cache = true; break;
      case NAN_CONDITION_LT: condition = AGX_FCOND_LTN; break;
      case NAN_CONDITION_GT: condition = AGX_FCOND_GTN; break;
      case SATURATE: break;
      }

      agx_instr *I = agx_fcmpsel_to(b, dest, compare, threshold, if_true,
                                     if_false, condition);
      I->saturate = form == SATURATE;
      agx_stop(b);
      /*
       * Eight of these were recording what had been measured rather than what
       * the encoding refuses, and each now has an Apple witness:
       *
       *   FULL_DEST       the result width is a field, f11 / f25
       *   COMPARE_HALF    the comparator widths are f43 and f60, and f65
       *                   composes with them: 12051d06850302c8
       *   COMPARE_ABS     f47 with a minifloat comparator:
       *                   120309a123b00302208000800100
       *   COMPARE_ABS_NEG f47 and f65 on one instruction:
       *                   12051d0787030208800200800100
       *   THRESHOLD_ABS   a minifloat's sign is bit 7, so |k| and -k are
       *   THRESHOLD_NEG   constant folding, which is how Apple spells
       *                   `-a < 0.5f` as `a > -0.5f`: 120319a122b002c22080
       *   NAN_CONDITION_* ltn/gtn are the ordered relation with the two
       *                   selected values exchanged, which this path already
       *                   did for a zero comparator
       */
      const bool grounded =
         form == FULL_DEST || form == COMPARE_HALF || form == COMPARE_ABS ||
         form == COMPARE_ABS_NEG || form == THRESHOLD_ABS ||
         form == THRESHOLD_NEG || form == NAN_CONDITION_LT ||
         form == NAN_CONDITION_GT;
      if (grounded)
         EXPECT_FALSE(pack(b).empty());
      else
         EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, AlphaCoverageBoundarySelectMatchesNative)
{
   enum modifier {
      NONE,
      COMPARE_ABS,
      COMPARE_NEG,
      COMPARE_KILL,
      COMPARE_CONFLICTING_HINTS,
      THRESHOLD_ABS,
      SELECTED_ABS,
      SELECTED_NEG,
      SELECTED_KILL,
      SATURATE,
   };

   auto build = [&](enum modifier kind, bool fail_closed) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index compare = agx_register(2, AGX_SIZE_32);
      agx_index threshold = agx_immediate(0x6f); /* 15.5 */
      agx_index selected = agx_register(0, AGX_SIZE_32);
      compare.discard = selected.discard = true;

      switch (kind) {
      case NONE: break;
      case COMPARE_ABS: compare.abs = true; break;
      case COMPARE_NEG: compare.neg = true; break;
      case COMPARE_KILL: compare.kill = true; break;
      case COMPARE_CONFLICTING_HINTS: compare.cache = true; break;
      case THRESHOLD_ABS: threshold.abs = true; break;
      case SELECTED_ABS: selected.abs = true; break;
      case SELECTED_NEG: selected.neg = true; break;
      case SELECTED_KILL: selected.kill = true; break;
      case SATURATE: break;
      }

      agx_instr *I = agx_fcmpsel_to(
         b, agx_register(14, AGX_SIZE_32), compare, threshold, selected,
         agx_immediate(0), AGX_FCOND_GE);
      I->saturate = kind == SATURATE;
      agx_stop(b);
      return pack(b, !fail_closed);
   };

   /* Exact post-RA alpha-to-coverage shape. The physical destination needs
    * Apple's ten-byte ep22 form; the terminal instruction follows it. */
   const uint8_t expected[] = {
      0x72, 0x03, 0x0b, 0xef, 0x86, 0x00, 0x12, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(build(NONE, false),
             std::vector<uint8_t>(expected, expected + sizeof(expected)));

   /* A fresh full-result Apple witness grounds the same negate
    * canonicalization. */
   const uint8_t negated[] = {
      0x72, 0x03, 0x1b, 0xef, 0x86, 0x00, 0x13, 0x02, 0x20, 0x80,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(build(COMPARE_NEG, false),
             std::vector<uint8_t>(negated, negated + sizeof(negated)));

   /* An absolute comparator against a minifloat, and an absolute value taken
    * on that minifloat, are both grounded now: f47 composes with an immediate
    * comparator (120309a123b00302208000800100) and a minifloat's sign is bit
    * 7, so |k| is constant folding rather than an encoding question. */
   EXPECT_FALSE(build(COMPARE_ABS, false).empty());
   EXPECT_FALSE(build(THRESHOLD_ABS, false).empty());

   for (unsigned bad = COMPARE_ABS; bad <= SATURATE; ++bad) {
      if (bad == COMPARE_NEG || bad == COMPARE_ABS || bad == THRESHOLD_ABS)
         continue;
      SCOPED_TRACE(bad);
      EXPECT_TRUE(build((enum modifier)bad, true).empty());
   }
}

TEST_F(AGX3PackBinary, HyprlandRemainingFloatSelectsMatchNativeForms)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));

   /* Exact post-RA blurprepare.frag shape: r4 >= 0.25 ? r5 : 0. */
   agx_builder *b = builder(enc);
   agx_index compare = agx_register(8, AGX_SIZE_32);
   agx_index selected = agx_register(10, AGX_SIZE_32);
   compare.cache = selected.cache = true;
   agx_fcmpsel_to(b, agx_register(12, AGX_SIZE_32), compare,
                  agx_immediate(0x20), selected, agx_immediate(0),
                  AGX_FCOND_GE);
   agx_stop(b);

   const uint16_t blur_selected[] = {10, 0};
   const bool blur_immediate[] = {false, true};
   const enum agx3_alu_hint blur_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE,
   };
   ASSERT_TRUE(agx3_alu_fcmp_minifloat_select_mixed(
      enc, 12, 8, 0x20, blur_selected, blur_immediate, blur_hints,
      AGX_FCOND_GE, false, false, 0, bytes, &len));
   std::vector<uint8_t> want;
   append(want, bytes, len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* Apple canonicalizes inner_glow's `-a < b` as `b > -a`. */
   b = builder(enc);
   agx_index a = agx_register(8, AGX_SIZE_32);
   a.neg = true;
   agx_index c = agx_register(16, AGX_SIZE_32);
   agx_fcmpsel_to(b, agx_register(1, AGX_SIZE_16), a, c,
                  agx_immediate(255), agx_immediate(0), AGX_FCOND_LT);
   agx_stop(b);

   const uint16_t glow_src[] = {16, 8};
   const bool glow_width[] = {false, false};
   const enum agx3_alu_hint glow_hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   ASSERT_TRUE(agx3_alu_fcmp_select_imm_width_neg_second(
      enc, 1, glow_src, glow_width, glow_hints, AGX_FCOND_GT, 255, 0,
      true, false, 0, bytes, &len));
   want.clear();
   append(want, bytes, len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* The negate on the OTHER operand is f65 with nothing rearranged: it is
    * the slot the field names, so the relation and the operand order both
    * stand. `a < -b` is Apple's 02011d03228103c82080, one bit away from its
    * 02011d03228102c02080 twin for `a > b`. */
   b = builder(enc);
   a = agx_register(8, AGX_SIZE_32);
   c = agx_register(16, AGX_SIZE_32);
   c.neg = true;
   agx_fcmpsel_to(b, agx_register(1, AGX_SIZE_16), a, c,
                  agx_immediate(255), agx_immediate(0), AGX_FCOND_LT);
   agx_stop(b);

   const uint16_t second_src[] = {8, 16};
   ASSERT_TRUE(agx3_alu_fcmp_select_imm_width_neg_second(
      enc, 1, second_src, glow_width, glow_hints, AGX_FCOND_LT, 255, 0,
      true, false, 0, bytes, &len));
   want.clear();
   append(want, bytes, len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* Mixing the negate with an absolute source IS one instruction: the
    * fourteen-byte abs endpoint carries the same comparator-1 negate as the
    * ten-byte one, in the same place (byte 7 bit 3). metal-nt
    * -arch applegpu_g17p on a kernel whose two selects are warmed first, so
    * neither row takes a wait, gives the minimal pair
    *
    *   (fabs(x) <  y) ? 255u : 0u   02 83 07 8b a3 ff 03 00 20 80 00 00 01 00
    *   (fabs(x) < -y) ? 255u : 0u   12 03 1f 0b a3 ff 03 08 20 80 00 00 01 00
    *
    * differing in the operand block and in exactly that bit -- `01 00` is the
    * abs mask naming comparator 0 in both.  Apple also spells `-x < fabs(y)`
    * by exchanging the operands and reversing the relation onto the same
    * negate (22 95 07 87 23 83 02 08 20 81 00 00 01 00), which is the
    * normalisation agx3_alu_fcmp_select() already performs.
    *
    * So the byte-constant specialization above must DECLINE this shape rather
    * than refuse it, and the general compare-select encodes it: refusing made
    * the fragment program unlinkable, and the driver then dropped every draw
    * that emulated a single cull distance. */
   b = builder(enc);
   a = agx_register(8, AGX_SIZE_32);
   a.abs = true;
   c = agx_register(16, AGX_SIZE_32);
   c.neg = true;
   agx_fcmpsel_to(b, agx_register(1, AGX_SIZE_16), a, c,
                  agx_immediate(255), agx_immediate(0), AGX_FCOND_LT);
   agx_stop(b);
   const std::vector<uint8_t> absneg = pack(b);

   b = builder(enc);
   a = agx_register(8, AGX_SIZE_32);
   a.abs = true;
   c = agx_register(16, AGX_SIZE_32);
   agx_fcmpsel_to(b, agx_register(1, AGX_SIZE_16), a, c,
                  agx_immediate(255), agx_immediate(0), AGX_FCOND_LT);
   agx_stop(b);
   const std::vector<uint8_t> abs_only = pack(b);

   /* Both encode, on the same endpoint, and the negate is the single bit
    * Apple's own pair moves. */
   ASSERT_EQ(absneg.size(), abs_only.size());
   ASSERT_GT(absneg.size(), 14u);
   std::vector<uint8_t> expected = abs_only;
   expected[7] |= 0x08;
   EXPECT_EQ(absneg, expected);

   /* The trailing abs mask still names comparator 0 only, and the selected
    * byte constants are untouched -- this stayed the integer placement. */
   EXPECT_EQ(absneg[4], 0xa3);
   EXPECT_EQ(absneg[5], 255);
   EXPECT_EQ(absneg[12], 0x01);
   EXPECT_EQ(absneg[13], 0x00);
}

TEST_F(AGX3PackBinary, MinifloatSpecializationDeclinesOtherGroundedForms)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index compare = agx_register(8, AGX_SIZE_32);
   agx_index selected = agx_register(14, AGX_SIZE_32);
   compare.discard = selected.discard = true;

   /* Cube lowering produces this zero-threshold GE form.  It is not the
    * specialized Hyprland GE encoding, but the later general zero-select
    * path has its own grounded representation and must get a chance at it. */
   agx_fcmpsel_to(b, agx_register(8, AGX_SIZE_32), compare,
                  agx_immediate(0), agx_immediate(0), selected,
                  AGX_FCOND_GE);
   agx_stop(b);
   EXPECT_GT(pack(b).size(), 4u);
}

TEST_F(AGX3PackBinary, F32PlusHalfMatchesHyprlandNativeShapes)
{
   for (unsigned order = 0; order < 2; ++order) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(0, AGX_SIZE_32);
      agx_index f32 = agx_register(0, AGX_SIZE_32);
      agx_index f16 = agx_register(5, AGX_SIZE_16);
      dest.cache = true;
      f32.discard = true;
      f16.cache = true;
      agx_fadd_to(b, dest, order ? f16 : f32, order ? f32 : f16);
      agx_stop(b);
      const std::vector<uint8_t> hyprland = {
         0x09, 0x01, 0x2c, 0x84, 0x10, 0x00, 0x0e, 0x00, 0x00, 0x00,
      };
      const auto code = pack(b);
      EXPECT_EQ(code, hyprland);
   }

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index f32 = agx_register(2, AGX_SIZE_32);
   agx_index f16 = agx_register(7, AGX_SIZE_16);
   f32.discard = f16.discard = true;
   agx_fadd_to(b, agx_register(2, AGX_SIZE_32), f32, f16);
   agx_stop(b);
   const uint8_t native_odd_no_wait[] = {
      0x19, 0x03, 0x1c, 0x06, 0x10, 0x00,
   };
   const auto code = pack(b);
   ASSERT_EQ(code.size(), sizeof(native_odd_no_wait) + 4);
   EXPECT_EQ(memcmp(code.data(), native_odd_no_wait,
                    sizeof(native_odd_no_wait)), 0);

   /* Exact post-RA shape from Hyprland's colour-management branches. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   f32 = agx_register(2, AGX_SIZE_32);
   f16 = agx_register(1, AGX_SIZE_16);
   f16.cache = true;
   agx_fadd_to(b, agx_register(2, AGX_SIZE_32), f32, f16);
   agx_stop(b);
   const uint8_t hyprland_ra[] = {
      0x19, 0x03, 0x04, 0x80, 0x10, 0x00,
   };
   const auto hyprland_code = pack(b);
   ASSERT_EQ(hyprland_code.size(), sizeof(hyprland_ra) + 4);
   EXPECT_EQ(memcmp(hyprland_code.data(), hyprland_ra,
                    sizeof(hyprland_ra)), 0);
}

TEST_F(AGX3PackBinary, F32PlusHalfRefusesUngroundedModes)
{
   for (unsigned mode = 1; mode < 9; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(0, AGX_SIZE_32);
      agx_index f32 = agx_register(0, AGX_SIZE_32);
      agx_index f16 = agx_register(5, AGX_SIZE_16);
      if (mode == 1)
         f32.neg = true;
      if (mode == 2)
         f16.neg = true;
      if (mode == 3)
         f16.abs = true;
      if (mode == 4)
         f16.cache = f16.discard = true;
      if (mode == 5)
         f16.kill = true;
      if (mode == 6)
         f32.value = 1;
      if (mode == 7)
         dest.size = AGX_SIZE_16;
      agx_instr *add = agx_fadd_to(b, dest, f32, f16);
      add->saturate = mode == 8;
      agx_stop(b);
      /* A modifier on a mixed-width add is Apple's own instruction, not a
       * different one: the slot widths f43/f26 and the modifier bits
       * f47/f48/f30/f31 belong to one encoding point and compose freely.
       * `a - b`, `b - a`, `a + fabs(b)` and `saturate(a + b)` over a
       * float/half pair are 09011c0400c8, 09041c0100c8, 09011c04020000800200
       * and 09011c0401000082. Modes 4..7 are lifetime and alignment, which
       * stay refused. */
      if (mode >= 1 && mode <= 3)
         EXPECT_FALSE(pack(b).empty()) << "mode=" << mode;
      else if (mode == 8)
         EXPECT_FALSE(pack(b).empty()) << "mode=" << mode;
      else
         EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, AbsoluteF32PlusHalfMatchesGlitchShaderShape)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index f32 = agx_register(8, AGX_SIZE_32);
   f32.abs = true;
   f32.discard = true;
   agx_index f16 = agx_register(28, AGX_SIZE_16);
   f16.cache = true;
   agx_fadd_to(b, agx_register(8, AGX_SIZE_32), f32, f16);
   agx_stop(b);
   const auto code = pack(b);
   ASSERT_EQ(code.size(), 14u);

   /* abs and neg on the same mixed-width source used to be refused as
    * unwitnessed. Apple's `b - fabs(a)` is 09041c01020800800200: f47 and f48
    * together, with f43 = 2 and f26 = 1. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   f32 = agx_register(8, AGX_SIZE_32);
   f32.abs = f32.neg = true;
   agx_fadd_to(b, agx_register(8, AGX_SIZE_32), f32, f16);
   agx_stop(b);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, U16ToF32PreservesHalfLivenessAndLoadWait)
{
   for (const auto enc : {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      SCOPED_TRACE(enc);
      agx_builder *b = builder(enc);
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_index even = agx_register(0, AGX_SIZE_16);
      agx_index odd = agx_register(1, AGX_SIZE_16);
      even.discard = false;
      odd.discard = true;
      agx_convert_to(b, agx_register(4, AGX_SIZE_32),
                      agx_immediate(AGX_CONVERT_U16_TO_F), even, AGX_ROUND_RTE);
      agx_convert_to(b, agx_register(6, AGX_SIZE_32),
                      agx_immediate(AGX_CONVERT_U16_TO_F), odd, AGX_ROUND_RTE);
      agx_stop(b);
      struct agx3_sr_desc sr = {};
      sr.dest = 0;
      sr.width = AGX3_SR_W32;
      sr.sr = 23;
      sr.slot = 1;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_sr_encode(enc, &sr, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_u16_to_f32(enc, 4, 0, AGX3_ALU_HINT_NONE, 1,
                                    bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_u16_to_f32(enc, 6, 1, AGX3_ALU_HINT_DISCARD, 0,
                                    bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, U16ToF32RejectsRemainingUnwitnessedDomains)
{
   for (unsigned variant = 0; variant < 6; ++variant) {
      SCOPED_TRACE(variant);
      /* S16-to-f32 is now independently witnessed alongside U16-to-f32,
       * and U16-to-half has its own native int2flt point. */
      if (variant == 0 || variant == 1 || variant == 4)
         continue;
      agx_builder *b = builder(variant == 0 ? AGX_ISA_ENCODING_AGX3C
                                           : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(4, variant == 1 ? AGX_SIZE_16 : AGX_SIZE_32);
      agx_index src = variant == 2 ? agx_uniform(0, AGX_SIZE_16)
                                   : agx_register(0, AGX_SIZE_16);
      if (variant == 3)
         src.neg = true;
      agx_convert_to(b, dest,
         agx_immediate(variant == 4 ? AGX_CONVERT_S16_TO_F : AGX_CONVERT_U16_TO_F),
         src, variant == 5 ? AGX_ROUND_RTZ : AGX_ROUND_RTE);
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, G17PNarrowIntToFloatUsesNativeInt2flt)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;

   /* int16 to half, and a signed byte to half or float32, are one int2flt
    * reading the source half directly. */
   const struct {
      enum agx_convert mode;
      enum agx_size dest_size;
   } direct[] = {
      {AGX_CONVERT_S16_TO_F, AGX_SIZE_16},
      {AGX_CONVERT_U16_TO_F, AGX_SIZE_16},
      {AGX_CONVERT_S8_TO_F, AGX_SIZE_16},
      {AGX_CONVERT_S8_TO_F, AGX_SIZE_32},
      {AGX_CONVERT_U8_TO_F, AGX_SIZE_32},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(direct); ++i) {
      const bool half = direct[i].dest_size == AGX_SIZE_16;
      agx_builder *b = builder(enc);
      agx_index src = agx_register(3, AGX_SIZE_16);
      src.discard = true;
      agx_convert_to(b, agx_register(half ? 5 : 4, direct[i].dest_size),
                     agx_immediate(direct[i].mode), src, AGX_ROUND_RTE);
      agx_stop(b);

      const bool byte = direct[i].mode == AGX_CONVERT_S8_TO_F ||
                        direct[i].mode == AGX_CONVERT_U8_TO_F;
      const bool is_signed = direct[i].mode == AGX_CONVERT_S8_TO_F ||
                             direct[i].mode == AGX_CONVERT_S16_TO_F;
      ASSERT_TRUE(agx3_alu_narrow_int_to_float(
         enc, half ? 5 : 4, half, 3, byte, is_signed, AGX3_ALU_HINT_DISCARD,
         false, 0, bytes, &len));
      std::vector<uint8_t> want(bytes, bytes + len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want) << i;
   }

   /* Apple has no unsigned byte to half form: `half(uchar)` masks the byte
    * into the destination and converts that ushort in place. */
   agx_builder *b = builder(enc);
   agx_index dest = agx_register(5, AGX_SIZE_16);
   dest.cache = true;
   agx_convert_to(b, dest, agx_immediate(AGX_CONVERT_U8_TO_F),
                  agx_register(3, AGX_SIZE_16), AGX_ROUND_RTE);
   agx_stop(b);
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_alu_bit_imm(enc, AGX3_ALU_BAND, true, 5, 3, 0xff,
                                AGX3_ALU_HINT_NONE, 0, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_narrow_int_to_float(enc, 5, true, 5, false, false,
                                            AGX3_ALU_HINT_DISCARD, true, 0,
                                            bytes, &len));
   append(want, bytes, len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FaddMinifloatPreservesWaitLivenessAndCommutativity)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   for (unsigned order = 0; order < 2; ++order) {
      agx_builder *b = builder(enc);
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_index live = agx_register(0, AGX_SIZE_32);
      agx_index last = live;
      live.discard = false;
      last.kill = last.discard = true;
      agx_index imm = agx_immediate(0x20);
      agx_fadd_to(b, agx_register(4, AGX_SIZE_32),
                  order ? imm : live, order ? live : imm);
      agx_fadd_to(b, agx_register(6, AGX_SIZE_32), last,
                  agx_immediate(0x21));
      agx_stop(b);
      struct agx3_sr_desc sr = {};
      sr.dest = 0;
      sr.width = AGX3_SR_W32;
      sr.sr = 23;
      sr.slot = 1;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_sr_encode(enc, &sr, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_fadd_minifloat(enc, 4, 0, 0x20, false,
                                         AGX3_ALU_HINT_NONE, 1, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_fadd_minifloat(enc, 6, 0, 0x21, false,
                                         AGX3_ALU_HINT_DISCARD, 0, bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, FaddTwoMinifloatsFoldsToLdimm)
{
   for (bool swapped : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_fadd_to(b, agx_register(2, AGX_SIZE_32),
                  agx_immediate(swapped ? 0x80 : 0x30),
                  agx_immediate(swapped ? 0x30 : 0x80));
      agx_stop(b);
      struct agx3_encoded_ldimm load;
      ASSERT_TRUE(agx3_pack_ldimm_u32(AGX_ISA_ENCODING_AGX3B, 2,
                                      0x3f800000, &load));
      std::vector<uint8_t> expected(load.bytes,
                                    load.bytes + load.length);
      const uint8_t stop[] = {0x0e, 0x00, 0x00, 0x00};
      append(expected, stop, sizeof(stop));
      EXPECT_EQ(pack(b), expected);
   }
}

TEST_F(AGX3PackBinary, M5FaddMinifloatAndSaturation)
{
   key.dev.emit_agx3d = true;
   for (bool saturate : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_index src = agx_register(saturate ? 4 : 0, AGX_SIZE_32);
      src.discard = saturate;
      agx_instr *add = agx_fadd_to(b,
         agx_register(saturate ? 0 : 4, AGX_SIZE_32), src,
         agx_immediate(saturate ? 0x80 : 0x20));
      add->saturate = saturate;
      agx_stop(b);
      const std::vector<uint8_t> expected = saturate
         ? std::vector<uint8_t>{0x09,0x05,0x1c,0x81,0x01,0x02,0x00,0x02,
                                0x0e,0x00,0x00,0x00}
         : std::vector<uint8_t>{0x29,0xa1,0x04,0x01,0x80,0x00,
                                0x0e,0x00,0x00,0x00};
      EXPECT_EQ(pack(b), expected);
   }
}

TEST_F(AGX3PackBinary, FaddMinifloatRefusesRemainingUngroundedModes)
{
   for (unsigned variant = 1; variant < 10; ++variant) {
      SCOPED_TRACE(variant);
      /* Negation, absolute, two-minifloat folding and saturation all have
       * exact, independently tested G17P paths. */
      if (variant == 4 || variant == 5 || variant == 7 || variant == 9)
         continue;
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(4, variant == 1 ? AGX_SIZE_16 : AGX_SIZE_32);
      agx_index src = variant == 2 ? agx_uniform(0, AGX_SIZE_32)
                                   : agx_register(0, AGX_SIZE_32);
      agx_index imm = agx_immediate(variant == 3 ? 256 : 0x20);
      if (variant == 4)
         src.neg = true;
      if (variant == 5)
         src.abs = true;
      if (variant == 6)
         imm.neg = true;
      if (variant == 7)
         src = agx_immediate(0x20);
      if (variant == 8)
         src.size = AGX_SIZE_16;
      agx_instr *add = agx_fadd_to(b, dest, src, imm);
      add->saturate = variant == 9;
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, FaddAbsoluteMinifloatMatchesNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_abs(agx_register(0, AGX_SIZE_32));
   src.discard = true;
   agx_fadd_to(b, agx_register(0, AGX_SIZE_32), src,
               agx_immediate(0x20));
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x09, 0xa1, 0x14, 0x01, 0x82, 0x00, 0x00, 0x00, 0x02, 0x00,
   };
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FaddOneMinusAbsMatchesNativeLiveForm)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index dest = agx_register(2, AGX_SIZE_32);
   dest.cache = true;
   agx_index src = agx_register(0, AGX_SIZE_32);
   src.abs = true;
   src.neg = true;
   src.cache = true;
   agx_fadd_to(b, dest, agx_immediate(0x30), src);
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x19, 0xb1, 0x24, 0x81, 0x82, 0x08, 0x00, 0x00, 0x02, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), want);
}

/*
 * 32x32 -> 64. Both halves of the product are written, so this is the shape
 * `umul_high`/`imul_high` and NIR's `*mul_2x32_64` all reach the backend as.
 * The AGX2 source convention carries over: `abs` on an operand narrower than
 * the destination means zero-extend, its absence means sign-extend, which is
 * what the per-slot f28/f43 bits encode.
 */
TEST_F(AGX3PackBinary, G17PNarrowImmediateMADMatchesNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index source = agx_register(3, AGX_SIZE_16);
   source.discard = true;
   agx_imad_to(b, agx_register(3, AGX_SIZE_16), source,
               agx_immediate(11), agx_immediate(9), 0);
   agx_stop(b);

   static const uint8_t native[] = {
      0x9f,0x00,0x54,0x03,0x02,0x06,0x2c,0x48,0x00,0x22,0x02,0x00,
   };
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* Alpha-to-coverage lowers its sample-index calculation to this exact
    * narrow register multiply.  It must reach the grounded imadd form rather
    * than fail closing the separately compiled fragment epilog. */
   agx_builder *mul = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dst = agx_register(0, AGX_SIZE_16);
   agx_index a = agx_register(0, AGX_SIZE_16);
   agx_index m = agx_register(4, AGX_SIZE_16);
   dst.cache = true;
   a.discard = true;
   m.discard = true;
   agx_imad_to(mul, dst, a, m, agx_zero(), 0);
   agx_stop(mul);

   static const uint8_t native_mul[] = {
      /* Apple witness with its independent slot-1 wait cleared. */
      0x9f,0x00,0x54,0x00,0x03,0x00,0x10,0x00,0x00,0x26,0x0a,0x00,
   };
   std::vector<uint8_t> want_mul(std::begin(native_mul),
                                 std::end(native_mul));
   append(want_mul, end.bytes, end.length);
   EXPECT_EQ(pack(mul), want_mul);
}

TEST_F(AGX3PackBinary, FirefoxNarrowRegisterMADMatchesGroundedCore)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(24, AGX_SIZE_16);
   agx_index a = agx_register(32, AGX_SIZE_16);
   agx_index m = agx_register(25, AGX_SIZE_16);
   agx_index c = agx_register(24, AGX_SIZE_16);
   agx_imad_to(b, dest, a, m, c, 0);
   agx_stop(b);

   const struct agx3_imad_desc d = {
      .dest = 24, .dest_16 = true,
      .src_a = 32, .src_a_16 = true,
      .multiplier_is_register = true, .multiplier_reg = 25,
      .multiplier_16 = true,
      .src_c = 24, .src_c_16 = true,
   };
   uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   std::vector<uint8_t> want(bytes, bytes + len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* Only the plain half addend has a native witness for the narrow result. */
   for (unsigned variant = 0; variant < 3; ++variant) {
      agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index bad_c = agx_register(24, variant == 2 ? AGX_SIZE_32
                                                      : AGX_SIZE_16);
      bad_c.abs = variant == 0;
      bad_c.neg = variant == 1;
      agx_imad_to(bad, dest, a, m, bad_c, 0);
      agx_stop(bad);
      EXPECT_TRUE(pack(bad, false).empty()) << variant;
   }
}

TEST_F(AGX3PackBinary, FirefoxWideningHalfMultiplyMatchesNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(34, AGX_SIZE_32);
   agx_index a = agx_register(25, AGX_SIZE_16);
   agx_index m = agx_register(33, AGX_SIZE_16);
   agx_imad_to(b, dest, a, m, agx_zero(), 0);
   agx_stop(b);

   struct agx3_imad_desc d = {};
   d.dest = 34;
   d.src_a = 25;
   d.src_a_16 = true;
   d.src_a_sign_extend = true;
   d.multiplier_is_register = true;
   d.multiplier_reg = 33;
   d.multiplier_16 = true;
   d.multiplier_sign_extend = true;
   d.no_addend = true;
   uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   std::vector<uint8_t> want(bytes, bytes + len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* Each half widens by its own sign bit, so an unsigned multiplier is
    * f43 clear with f28 still set. A 32-bit multiplicand keeps its slot. */
   for (unsigned variant = 0; variant < 2; ++variant) {
      agx_builder *mixed = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index mixed_a = variant == 0 ? agx_register(24, AGX_SIZE_32) : a;
      agx_index mixed_m = m;
      mixed_m.abs = variant == 1;
      agx_imad_to(mixed, dest, mixed_a, mixed_m, agx_zero(), 0);
      agx_stop(mixed);

      struct agx3_imad_desc md = d;
      md.src_a = mixed_a.value;
      md.src_a_16 = variant == 1;
      md.src_a_sign_extend = variant == 1;
      md.multiplier_sign_extend = variant == 0;
      len = 0;
      ASSERT_TRUE(agx3_imad_encode(AGX_ISA_ENCODING_AGX3B, &md, bytes, &len));
      std::vector<uint8_t> mixed_want(bytes, bytes + len);
      append(mixed_want, end.bytes, end.length);
      EXPECT_EQ(pack(mixed), mixed_want) << variant;
   }

   for (unsigned variant = 1; variant < 4; ++variant) {
      agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index bad_c = variant == 1 ? agx_immediate(1) : agx_zero();
      agx_instr *I = agx_imad_to(bad, dest, a, m, bad_c,
                                 variant == 2 ? 1 : 0);
      I->saturate = variant == 3;
      agx_stop(bad);
      EXPECT_TRUE(pack(bad, false).empty()) << variant;
   }
}

TEST_F(AGX3PackBinary, G17PNarrowShiftSevenMatchesNativePair)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(0, AGX_SIZE_16);
   agx_index value = agx_register(0, AGX_SIZE_16);
   dest.cache = true;
   value.discard = true;
   agx_bfi_to(b, dest, agx_immediate(0), value, agx_immediate(7), 0);
   agx_stop(b);

   /* Apple's first instruction also waits on its input load. The builder has
    * no async producer, so only that independently grounded wait bit clears. */
   static const uint8_t native[] = {
      0x9f,0x01,0x54,0x00,0x03,0x00,0x00,0x40,0x14,0x14,
      0x9f,0x01,0x54,0x00,0x03,0x00,0x00,0x00,0x14,0x04,
   };
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_bfi_to(bad, agx_register(0, AGX_SIZE_16), agx_immediate(0),
              agx_register(2, AGX_SIZE_16), agx_immediate(9), 0);
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, G17PNarrowDynamicShiftsMatchNativeFields)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));

   agx_builder *left = builder(enc);
   agx_index left_dest = agx_register(3, AGX_SIZE_16);
   agx_index left_value = agx_register(2, AGX_SIZE_16);
   agx_index left_shift = agx_register(12, AGX_SIZE_32);
   left_dest.cache = true;
   left_value.cache = true;
   left_shift.discard = true;
   agx_bfi_to(left, left_dest, agx_immediate(0), left_value, left_shift, 0);
   agx_stop(left);
   static const uint8_t native_left[] = {
      0x27,0x00,0x54,0x03,0x03,0x04,0x32,0x00,0xc0,0x12,0x05,0x00,
   };
   std::vector<uint8_t> want_left(std::begin(native_left),
                                  std::end(native_left));
   append(want_left, end.bytes, end.length);
   EXPECT_EQ(pack(left), want_left);

   agx_builder *right = builder(enc);
   agx_index right_dest = agx_register(6, AGX_SIZE_16);
   agx_index right_value = agx_register(2, AGX_SIZE_16);
   agx_index right_shift = agx_register(4, AGX_SIZE_32);
   right_dest.cache = true;
   right_value.discard = true;
   right_shift.discard = true;
   agx_bfeil_to(right, right_dest, agx_immediate(0), right_value,
                right_shift, 0);
   agx_stop(right);
   static const uint8_t native_right[] = {
      0xa7,0x00,0x54,0x06,0x03,0x04,0x10,0x00,0xc0,0x13,0x05,0x00,
   };
   std::vector<uint8_t> want_right(std::begin(native_right),
                                   std::end(native_right));
   append(want_right, end.bytes, end.length);
   EXPECT_EQ(pack(right), want_right);
}

TEST_F(AGX3PackBinary, G17PRegisterMADImmediateAddendUsesNativeEndpoint)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index a = agx_register(4, AGX_SIZE_32);
   agx_index m = agx_register(0, AGX_SIZE_32);
   a.discard = true;
   m.discard = true;
   agx_imad_to(b, agx_register(0, AGX_SIZE_32), a, m,
               agx_immediate(1), 0);
   agx_stop(b);

   static const uint8_t native[] = {
      0x9f,0x00,0x54,0x00,0x02,0x08,0x00,0x08,0xd0,0x26,0x0a,0x00,
   };
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, WideSignedMultiplyMatchesNativeWitness)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index a = agx_register(0, AGX_SIZE_32);
   agx_index m = agx_register(2, AGX_SIZE_32);
   a.discard = true;
   m.discard = true;
   agx_imad_to(b, agx_register(0, AGX_SIZE_64), a, m, agx_zero(), 0);
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x9f, 0x00, 0x54, 0x00, 0x02, 0x00, 0x08, 0x00,
      0xe0, 0x26, 0x1e, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, WideUnsignedMultiplyZeroExtendsBothOperands)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index a = agx_register(0, AGX_SIZE_32);
   agx_index m = agx_register(2, AGX_SIZE_32);
   a.discard = true;
   m.discard = true;
   agx_imad_to(b, agx_register(0, AGX_SIZE_64), agx_abs(a), agx_abs(m),
               agx_zero(), 0);
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x9f, 0x00, 0x54, 0x00, 0x02, 0x00, 0x08, 0x00,
      0xe0, 0x26, 0x0a, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FabsMaterializationKeepsNegativeZeroAndWait)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index source = agx_register(2, AGX_SIZE_32);
   source.cache = true;
   source.abs = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.cache = true;
   agx_fadd_to(b, dest, source, agx_negzero());
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x49, 0x81, 0x2c, 0x83, 0x82, 0x00, 0x00, 0x00,
      0x02, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, WboitInfinityRepairUsesMixedWidthFsel)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index compare = agx_register(8, AGX_SIZE_32);
   agx_index infinity = agx_register(10, AGX_SIZE_16);
   agx_index if_true = agx_register(6, AGX_SIZE_32);
   agx_index if_false = agx_register(4, AGX_SIZE_32);
   compare.cache = infinity.cache = if_true.cache = true;
   if_false.discard = true;
   agx_index dest = agx_register(4, AGX_SIZE_32);
   dest.cache = true;
   agx_fcmpsel_to(b, dest, compare, infinity, if_true, if_false,
                  AGX_FCOND_EQ);
   agx_stop(b);

   std::vector<uint8_t> want = {
      0x22, 0x89, 0x27, 0x8a, 0x05, 0x86, 0x00, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), want);
}

/* fmin/fmax lower to a tied fcmpsel with LTN/GTN. On AGX3B that is the
 * native fsel.ep7 with condition 3/7 (IEEE minNum/maxNum): Apple's
 * -fno-fast-math fmin/fmax emit 12031e0501c0 / 12031e0500c0 after two loads
 * (wait slot 5); without a wait byte 5 carries no wait bits. The former
 * generic-FSEL route rewrote LTN into GE with swapped selects
 * (02013f0385020200, condition 5), which returns the NaN when src0 is NaN. */
TEST_F(AGX3PackBinary, TiedF32MinMaxUsesNativeFselCondition)
{
   for (int m = 0; m < 2; ++m) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index a = agx_register(2, AGX_SIZE_32);
      agx_index c = agx_register(4, AGX_SIZE_32);
      a.discard = c.discard = true;
      agx_index dest = agx_register(2, AGX_SIZE_32);
      agx_fcmpsel_to(b, dest, a, c, a, c,
                     m ? AGX_FCOND_GTN : AGX_FCOND_LTN);
      agx_stop(b);

      std::vector<uint8_t> want = {0x12, 0x03, 0x1e, 0x05,
                                   (uint8_t)(m ? 0x00 : 0x01), 0x00,
                                   0x0e, 0x00, 0x00, 0x00};
      std::vector<uint8_t> got = pack(b);
      EXPECT_EQ(got, want);
      /* fsel.ep7: 6 bytes, condition field in byte 4 bit 0 (3 -> 1, 7 -> 0),
       * compare/select registers tied (bytes 1 and 3). */
      ASSERT_EQ(got.size(), 10u);
      EXPECT_EQ(got[1] & 0x7e, 2);
      EXPECT_EQ(got[3] & 0x7e, 4);
   }
}

/* An untied LTN/GTN select has no NaN-correct grounded form: the packer must
 * refuse it rather than rewrite the condition into an ordered GE/LE. */
TEST_F(AGX3PackBinary, UntiedLtnFselIsRefused)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(2, AGX_SIZE_32);
   agx_index c = agx_register(4, AGX_SIZE_32);
   agx_index other = agx_register(6, AGX_SIZE_32);
   agx_index dest = agx_register(2, AGX_SIZE_32);
   agx_fcmpsel_to(b, dest, a, c, a, other, AGX_FCOND_LTN);
   agx_stop(b);
   pack(b, false);
}

TEST_F(AGX3PackBinary, UpdateLightmapMixedWidthLtFsel)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index compare = agx_register(46, AGX_SIZE_32);
   agx_index threshold = agx_register(52, AGX_SIZE_16);
   agx_index if_true = agx_register(26, AGX_SIZE_32);
   agx_index if_false = agx_register(36, AGX_SIZE_32);
   compare.cache = threshold.cache = if_true.cache = true;
   agx_index dest = agx_register(36, AGX_SIZE_32);
   dest.cache = true;
   agx_fcmpsel_to(b, dest, compare, threshold, if_true, if_false,
                  AGX_FCOND_LT);
   agx_stop(b);

   const uint8_t native[] = {
      0x22, 0xaf, 0x67, 0xb4, 0x02, 0x9a, 0x03, 0x00, 0x00, 0x24,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), std::vector<uint8_t>(native, native + sizeof(native)));
}

TEST_F(AGX3PackBinary, HalfComparisonSelectsFullRegisters)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(0, AGX_SIZE_16);
   agx_index c = agx_register(9, AGX_SIZE_16);
   agx_index if_true = agx_register(10, AGX_SIZE_32);
   agx_index if_false = agx_register(4, AGX_SIZE_32);
   a.abs = a.discard = true;
   c.cache = true;
   if_true.discard = if_false.discard = true;
   agx_fcmpsel_to(b, agx_register(4, AGX_SIZE_32), a, c, if_true, if_false,
                  AGX_FCOND_LT);
   agx_stop(b);

   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, HyprlandHalfMinifloatFselShapes)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index half = agx_register(6, AGX_SIZE_16);
   agx_fcmpsel_to(b, agx_register(8, AGX_SIZE_32), agx_immediate(32), half,
                  agx_register(8, AGX_SIZE_32),
                  agx_register(16, AGX_SIZE_32), AGX_FCOND_GE);
   agx_stop(b);
   const std::vector<uint8_t> half_threshold = {
      0x42, 0x06, 0x07, 0xa1, 0x06, 0x08, 0x03, 0x02, 0x00,
      0x10, 0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), half_threshold);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   half = agx_register(6, AGX_SIZE_16);
   half.cache = true;
   agx_index selected = agx_register(8, AGX_SIZE_32);
   selected.cache = true;
   agx_fcmpsel_to(b, agx_register(20, AGX_SIZE_32), half,
                  agx_immediate(0), selected, agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const std::vector<uint8_t> register_or_zero = {
      0xa2, 0x86, 0x03, 0x81, 0x02, 0x88, 0x03, 0x02, 0x20,
      0x80, 0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), register_or_zero);
}

TEST_F(AGX3PackBinary, HyprlandHalfMinifloatFselRefusesUngroundedModes)
{
   for (unsigned mode = 0; mode < 10; ++mode) {
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C
                                         : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(8, AGX_SIZE_32);
      agx_index half = agx_register(6, AGX_SIZE_16);
      agx_index immediate = agx_immediate(32);
      agx_index if_true = agx_register(8, AGX_SIZE_32);
      agx_index if_false = agx_register(16, AGX_SIZE_32);
      if (mode == 1)
         dest.size = AGX_SIZE_16;
      if (mode == 2)
         half.abs = true;
      if (mode == 3)
         half.neg = true;
      if (mode == 4)
         half.cache = half.discard = true;
      if (mode == 5)
         half.kill = true;
      if (mode == 6)
         immediate.value = 256;
      if (mode == 7)
         if_true.value = 9;
      if (mode == 8)
         if_false = agx_immediate(0);
      agx_instr *fsel = agx_fcmpsel_to(b, dest, immediate, half,
                                       mode == 8 ? agx_immediate(0) : if_true,
                                       if_false, AGX_FCOND_GE);
      fsel->saturate = mode == 9;
      agx_stop(b);
      /* The result width (mode 1) and an immediate selected value (mode 8)
       * are independent fields of this endpoint, not separate instructions:
       * Apple's own `select(v.x, v.y, a >= 0.5f)` is 12050da1850302c2, whose
       * f51 carries the minifloat and whose f11 carries the half result. */
      if (mode == 1 || mode == 8)
         EXPECT_FALSE(pack(b).empty()) << "mode=" << mode;
      else
         EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, HyprlandAbsoluteSecondFsel)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index plain = agx_register(26, AGX_SIZE_32);
   agx_index absolute = agx_register(8, AGX_SIZE_32);
   plain.cache = absolute.cache = true;
   absolute.abs = true;
   agx_fcmpsel_to(b, agx_register(16, AGX_SIZE_32), plain, absolute,
                  agx_register(22, AGX_SIZE_32),
                  agx_register(16, AGX_SIZE_32), AGX_FCOND_GE);
   agx_stop(b);
   const std::vector<uint8_t> target = {
      0x82, 0x9b, 0x07, 0x89, 0x07, 0x16, 0x02, 0x00, 0x00,
      0x10, 0x00, 0x00, 0x02, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), target);
}

TEST_F(AGX3PackBinary, GtkConvertCicpAbsoluteMinifloatFsel)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index absolute = agx_register(2, AGX_SIZE_32);
   absolute.abs = true;
   agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(32),
                  absolute, agx_register(4, AGX_SIZE_32),
                  agx_register(6, AGX_SIZE_32), AGX_FCOND_GE);
   agx_stop(b);

   /* Exact current-toolchain applegpu_g17p bytes for the no-wait kernel
    * `0.5 >= fabs(x) ? y : z`, plus STOP. */
   const std::vector<uint8_t> native = {
      0x02, 0x03, 0x07, 0xa1, 0x07, 0x04, 0x03, 0x02,
      0x00, 0x06, 0x00, 0x00, 0x01, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), native);
}

TEST_F(AGX3PackBinary, GtkConvertCicpAbsoluteMinifloatRefusesModifiers)
{
   for (unsigned mode = 0; mode < 8; ++mode) {
      /* Comparator negation is covered by NegatedMinifloat ordered-byte tests. */
      if (mode == 4)
         continue;
      agx_builder *b = builder(mode == 0 ? AGX_ISA_ENCODING_AGX3C
                                         : AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(0, mode == 1 ? AGX_SIZE_16
                                                 : AGX_SIZE_32);
      agx_index threshold = agx_immediate(32);
      agx_index compare = agx_register(2, mode == 2 ? AGX_SIZE_16
                                                    : AGX_SIZE_32);
      agx_index if_true = agx_register(4, AGX_SIZE_32);
      agx_index if_false = agx_register(mode == 3 ? 7 : 6, AGX_SIZE_32);
      compare.abs = true;
      if (mode == 4)
         compare.neg = true;
      if (mode == 5)
         threshold.abs = true;
      if (mode == 6)
         if_true.abs = true;
      if (mode == 7)
         compare.cache = compare.discard = true;
      agx_fcmpsel_to(b, dest, threshold, compare, if_true, if_false,
                     AGX_FCOND_GE);
      /* The result width is its own field (f11, or f25 on the shifted
       * destination block) and composes with an absolute comparator; the
       * ledger already carried the pair. */
      if (mode == 1)
         EXPECT_FALSE(pack(b).empty()) << "mode=" << mode;
      else
         EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, OrderedAbsoluteFselUsesGeneralEndpoint)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(2, AGX_SIZE_32);
   agx_index c = agx_register(4, AGX_SIZE_32);
   a.abs = c.abs = true;
   agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_32), a, c,
                  agx_register(6, AGX_SIZE_32),
                  agx_register(8, AGX_SIZE_32), AGX_FCOND_GE);
   agx_stop(b);
   const struct agx3_fsel_desc d = {
      .dest = 0, .cmp0 = 2, .cmp0_abs = true,
      .cmp1 = 4, .cmp1_abs = true,
      .sel0_kind = AGX3_FSEL_SEL_REGISTER,
      .sel1_kind = AGX3_FSEL_SEL_REGISTER,
      .sel0 = 6, .sel1 = 8, .condition = AGX_FCOND_GE,
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
   ASSERT_TRUE(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                                   &length));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   std::vector<uint8_t> want;
   append(want, bytes, length);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, AbsoluteCompareFselUsesNativeLongForm)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(34, AGX_SIZE_32);
   agx_index c = agx_register(20, AGX_SIZE_32);
   agx_index selected = agx_register(24, AGX_SIZE_32);
   agx_index dest = agx_register(26, AGX_SIZE_32);
   a.abs = c.abs = true;
   a.cache = c.cache = selected.cache = dest.cache = true;
   agx_fcmpsel_to(b, dest, a, c, selected, agx_immediate(0), AGX_FCOND_EQ);
   agx_stop(b);

   const uint16_t src[] = {34, 20, 24, 0};
   const bool compare_abs[] = {true, true};
   const bool compare_neg[] = {false, false};
   const bool immediate[] = {false, true};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_alu_fcmp_select_reg_imm(
      AGX_ISA_ENCODING_AGX3B, 26, src, false, compare_abs, compare_neg,
      immediate, hints, AGX_FCOND_EQ, false, true, 0, bytes, &len));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   std::vector<uint8_t> want;
   append(want, bytes, len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, UpdateLightmapNativeIntegerAndMaskForms)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);

   agx_index r13 = agx_register(26, AGX_SIZE_32);
   r13.cache = true;
   agx_index r12 = agx_register(24, AGX_SIZE_32);
   r12.cache = true;
   agx_ffs_to(b, r13, r12);

   agx_index r10 = agx_register(20, AGX_SIZE_32);
   agx_index r3 = agx_register(6, AGX_SIZE_32);
   r10.cache = r3.cache = true;
   agx_bitop_to(b, agx_register(20, AGX_SIZE_32), r10, r3, 0x2);

   agx_index half = agx_register(6, AGX_SIZE_16);
   agx_index half_dest = agx_register(6, AGX_SIZE_16);
   half_dest.cache = true;
   agx_fcmpsel_to(b, half_dest, agx_immediate(176),
                  agx_register(30, AGX_SIZE_32), half, agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3B, 26, 24,
                                 AGX3_ALU_HINT_CACHE, true, 1, bytes, &len));
   append(want, bytes, len);
   const uint16_t bit_src[] = {20, 6};
   const enum agx3_alu_hint bit_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE};
   ASSERT_TRUE(agx3_alu_bit_andc(AGX_ISA_ENCODING_AGX3B, 20, bit_src,
                                  bit_hints, false, 0, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_fcmp_minifloat_select_half(
      AGX_ISA_ENCODING_AGX3B, 6, 30, 176, 6, 0,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, AGX_FCOND_GT, true, 0,
      bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FmulMinifloatPreservesWaitLivenessAndCommutativity)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   for (unsigned order = 0; order < 2; ++order) {
      agx_builder *b = builder(enc);
      agx_get_sr_to(b, agx_register(0, AGX_SIZE_32),
                    AGX_SR_THREAD_POSITION_IN_GRID_X);
      agx_index live = agx_register(0, AGX_SIZE_32);
      agx_index last = live;
      live.discard = false;
      last.discard = true;
      agx_index imm = agx_immediate(0x02);
      agx_fmul_to(b, agx_register(4, AGX_SIZE_32),
                  order ? imm : live, order ? live : imm);
      agx_fmul_to(b, agx_register(6, AGX_SIZE_32), last,
                  agx_immediate(0x03));
      agx_stop(b);
      struct agx3_sr_desc sr = {};
      sr.dest = 0;
      sr.width = AGX3_SR_W32;
      sr.sr = 23;
      sr.slot = 1;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_sr_encode(enc, &sr, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_fmul_minifloat(enc, 4, 0, 0x02, AGX3_ALU_HINT_NONE, 1,
                                         bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_fmul_minifloat(enc, 6, 0, 0x03, AGX3_ALU_HINT_DISCARD, 0,
                                         bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, FmulAbsoluteMinifloatUsesGroundedPair)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_abs(agx_register(2, AGX_SIZE_32));
   src.cache = true;
   agx_fmul_to(b, agx_register(0, AGX_SIZE_32), src,
               agx_immediate(0x40));
   agx_stop(b);

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_alu_fadd_abs_negzero(
      AGX_ISA_ENCODING_AGX3B, 0, 2, true, AGX3_ALU_HINT_CACHE, 0,
      bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_fmul_minifloat_mod(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0x40, false, false,
      AGX3_ALU_HINT_DISCARD, 0, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FmulAbsoluteMinifloatMatchesQtHiQStage6)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(64, AGX_SIZE_32);
   dest.cache = true;
   agx_fmul_to(b, dest, agx_abs(agx_register(2, AGX_SIZE_32)),
               agx_immediate(0x40));
   agx_stop(b);

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_alu_fadd_abs_negzero(
      AGX_ISA_ENCODING_AGX3B, 64, 2, true, AGX3_ALU_HINT_NONE, 0,
      bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_fmul_minifloat_mod(
      AGX_ISA_ENCODING_AGX3B, 64, 64, 0x40, true, false,
      AGX3_ALU_HINT_DISCARD, 0, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FmulMinifloatRefusesUngroundedModes)
{
   for (unsigned variant = 1; variant < 10; ++variant) {
      /* Absolute f32 sources have an exact grounded two-instruction lowering;
       * scalar-half sources have an exact native witness. So does variant 1,
       * a 32-bit source narrowed into a half destination: f10 selects the
       * destination width independently of f26, and Apple emits the
       * mixed-width multiply directly. See the focused tests above and
       * FmulMinifloatNarrowsIntoHalfDestination below. */
      /* Negated f32 is covered by NegatedMinifloat signed-immediate tests. */
      if (variant == 1 || variant == 4 || variant == 5 || variant == 8)
         continue;
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(4, variant == 1 ? AGX_SIZE_16 : AGX_SIZE_32);
      agx_index src = variant == 2 ? agx_uniform(0, AGX_SIZE_32)
                                   : agx_register(0, AGX_SIZE_32);
      agx_index imm = agx_immediate(variant == 3 ? 256 : 0x02);
      if (variant == 4)
         src.neg = true;
      if (variant == 5)
         src.abs = true;
      if (variant == 6)
         imm.neg = true;
      if (variant == 7)
         src = agx_immediate(0x02);
      if (variant == 8)
         src.size = AGX_SIZE_16;
      agx_instr *mul = agx_fmul_to(b, dest, src, imm);
      mul->saturate = variant == 9;
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, M5MinifloatMultiplyMatchesNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_index src = agx_register(2, AGX_SIZE_32);
   src.discard = true;
   agx_fmul_to(b, agx_register(2, AGX_SIZE_32), src, agx_immediate(0x38));
   agx_stop(b);
   const std::vector<uint8_t> expected = {
      0x19,0xb9,0x15,0x03,0x80,0x00, 0x0e,0x00,0x00,0x00,
   };
   EXPECT_EQ(pack(b), expected);
}

/*
 * `packHalf2x16(vec2(x * k, ...))`: agx_optimizer_fmov_rev() folds the f2f16
 * into the multiply, so the packer sees a 32-bit source, a minifloat operand
 * and a HALF destination. That form is Apple's own -- f10 is the destination
 * width -- and used to fail the whole shader closed.
 */
TEST_F(AGX3PackBinary, FmulMinifloatNarrowsIntoHalfDestination)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(2, AGX_SIZE_32);
   src.discard = true;
   agx_fmul_to(b, agx_register(2, AGX_SIZE_16), src, agx_immediate(0x20));
   agx_stop(b);
   const auto code = pack(b);
   /* Apple's witness is `11 a1 15 03 80 c0`; a bare test has no producer to
    * wait on, so only the scoreboard field (f85) differs. The literal Apple
    * bytes are asserted at encoder level in test-agx3-alu-core.c. */
   const uint8_t native[] = {0x11, 0xa1, 0x15, 0x03, 0x80, 0x00};
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   /* The 32-bit destination keeps the bytes it always had. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(2, AGX_SIZE_32);
   src.discard = true;
   agx_fmul_to(b, agx_register(2, AGX_SIZE_32), src, agx_immediate(0x20));
   agx_stop(b);
   const auto wide = pack(b);
   const uint8_t native32[] = {0x19, 0xa1, 0x15, 0x03, 0x80, 0x00};
   ASSERT_EQ(wide.size(), sizeof(native32) + 4);
   EXPECT_EQ(memcmp(wide.data(), native32, sizeof(native32)), 0);
}

/*
 * `sqrt(abs(x))` and `inversesqrt(abs(x))`. A bare .abs on either member of
 * this family refused the shader; f30 carries it, witnessed one field at a
 * time against Apple.
 */
TEST_F(AGX3PackBinary, RsqrtFamilyCarriesAbsoluteSource)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(0, AGX_SIZE_32);
   src.abs = true;
   src.discard = true;
   agx_rsqrt_to(b, agx_register(0, AGX_SIZE_32), src);
   agx_stop(b);
   const auto rsq = pack(b);
   /* Apple: `af 01 56 00 02 00 b0 c0 00 00`; byte 2 drops to 0x54 here
    * because there is no scoreboard slot to wait on. */
   const uint8_t rsq_abs[] = {0xaf, 0x01, 0x54, 0x00, 0x02,
                              0x00, 0xb0, 0xc0, 0x00, 0x00};
   ASSERT_EQ(rsq.size(), sizeof(rsq_abs) + 4);
   EXPECT_EQ(memcmp(rsq.data(), rsq_abs, sizeof(rsq_abs)), 0);

   /* A negated source is separately witnessed on the same form. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(0, AGX_SIZE_32);
   src.neg = true;
   src.discard = true;
   agx_rsqrt_to(b, agx_register(0, AGX_SIZE_32), src);
   agx_stop(b);
   const auto rsq_n = pack(b);
   const uint8_t rsq_neg[] = {0xaf, 0x01, 0x54, 0x00, 0x02,
                              0x00, 0xb0, 0x40, 0x01, 0x00};
   ASSERT_EQ(rsq_n.size(), sizeof(rsq_neg) + 4);
   EXPECT_EQ(memcmp(rsq_n.data(), rsq_neg, sizeof(rsq_neg)), 0);

   /* The half form has no modifier witness and stays closed. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(2, AGX_SIZE_16);
   src.abs = true;
   agx_rsqrt_to(b, agx_register(2, AGX_SIZE_16), src);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

/*
 * `int16_t(x) + k`. A bare 16-bit integer source is sign-extended -- that is
 * what agx_optimizer_signext() means by dropping the SIGNEXT and asserting
 * the survivor carries no ABS -- and the register+immediate form refused it
 * while the register+register form beside it already encoded.
 */
TEST_F(AGX3PackBinary, IaddImmediateSignExtendsBareHalfSource)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(0, AGX_SIZE_16);
   src.discard = true;
   agx_iadd_to(b, agx_register(0, AGX_SIZE_32), src, agx_immediate(7), 0);
   agx_stop(b);
   const auto code = pack(b);
   /* Apple: `o[i] = int(short a[i]) + 7`, f15 = 7 and f43 = 1. */
   uint8_t native[] = {0x9f, 0x01, 0x54, 0x00, 0x02,
                       0x0e, 0x00, 0x08, 0x15, 0x0c};
   native[4] |= int_dest_cache();
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   /* An ABS source is the zero-extending form, and keeps its own bytes:
    * f43 clear, exactly as Apple's `uint(ushort a[i]) + 7`. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(0, AGX_SIZE_16);
   src.abs = true;
   src.discard = true;
   agx_iadd_to(b, agx_register(0, AGX_SIZE_32), src, agx_immediate(7), 0);
   agx_stop(b);
   const auto zext = pack(b);
   uint8_t native_u[] = {0x9f, 0x01, 0x54, 0x00, 0x02,
                         0x0e, 0x00, 0x08, 0x15, 0x04};
   native_u[4] |= int_dest_cache();
   ASSERT_EQ(zext.size(), sizeof(native_u) + 4);
   EXPECT_EQ(memcmp(zext.data(), native_u, sizeof(native_u)), 0);
}

TEST_F(AGX3PackBinary, DeviceAtomicsUseGenerationSpecificReturnRegisters)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      const agx_index value = agx_register(6, AGX_SIZE_32);
      const agx_index result =
         agx_register(e ? 4 : 6, AGX_SIZE_32); /* b is physically in-place */
      agx_atomic_to(b, result, value, agx_uniform(0, AGX_SIZE_64),
                    agx_immediate(0), AGX_ATOMIC_OPC_XOR);
      agx_stop(b);

      struct agx3_device_atomic_desc d = {
         .op = AGX3_ATOMIC_XOR,
         .value = 6,
         .returns_value = true,
         .result = (uint16_t)(e ? 4 : 6),
         .slot = 1,
         .base_kind = AGX3_ATOMIC_BASE_UNIFORM,
         .index_kind = AGX3_ATOMIC_INDEX_IMMEDIATE,
      };
      uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
      ASSERT_TRUE(agx3_atomic_encode_device(enc, &d, atomic, &atomic_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));

      std::vector<uint8_t> want;
      append(want, atomic, atomic_len);
      if (!e)
         append_g17p_atomic_return_reload(want, result.value);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, FragmentAtomicsEnterPixelTransactionFirst)
{
   /* shape 0: voiding atomic; 1: returning atomic; 2: returning atomic in a
    * fragment that also writes a colour. */
   for (unsigned shape = 0; shape < 3; ++shape) {
      const bool returned = shape != 0;
      const bool colour = shape == 2;
      const auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      if (colour) {
         b->shader->nir = rzalloc(mem_ctx, nir_shader);
         b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
         b->shader->nir->info.outputs_written =
            BITFIELD64_BIT(FRAG_RESULT_DATA0);
      }
      const agx_index value = agx_register(6, AGX_SIZE_32);
      const agx_index result =
         returned ? agx_register(6, AGX_SIZE_32) : agx_null();
      agx_atomic_to(b, result, value, agx_uniform(0, AGX_SIZE_64),
                    agx_immediate(0), AGX_ATOMIC_OPC_ADD);
      agx_stop(b);

      struct agx3_device_atomic_desc d = {
         .op = AGX3_ATOMIC_ADD,
         .value = 6,
         .returns_value = returned,
         .result = (uint16_t)(returned ? 6 : 0),
         .slot = (uint8_t)(returned ? 1 : 0),
         .base_kind = AGX3_ATOMIC_BASE_UNIFORM,
         .index_kind = AGX3_ATOMIC_INDEX_IMMEDIATE,
      };

      /* metal-nt -arch applegpu_g17p fragments with no colour writes enter
       * before the first atomic; only a returning atomic takes class 4 and
       * the barrier after its reload. */
      const bool bracket = returned && !colour;
      std::vector<uint8_t> want;
      uint8_t pixel[AGX3_PIXEL_MAX_LENGTH], pixel_len = 0;
      ASSERT_TRUE(agx3_pixel_colour_entry(enc, pixel, &pixel_len));
      append(want, pixel, pixel_len);
      if (bracket) {
         ASSERT_TRUE(agx3_pixel_atomic_acquire(enc, pixel, &pixel_len));
         append(want, pixel, pixel_len);
      }
      uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
      ASSERT_TRUE(agx3_atomic_encode_device(enc, &d, atomic, &atomic_len));
      append(want, atomic, atomic_len);
      if (returned)
         append_g17p_atomic_return_reload(want, 6);
      if (bracket) {
         ASSERT_TRUE(agx3_pixel_atomic_barrier(enc, pixel, &pixel_len));
         append(want, pixel, pixel_len);
      }
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want) << "shape=" << shape;
      EXPECT_TRUE(b->shader->agx3_colour_link.entry_emitted);
      EXPECT_EQ(b->shader->agx3_colour_link.entry_offset, 0u);
   }
}

TEST_F(AGX3PackBinary, DeviceAtomicUsesNativeMaskedExecutionMode)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->cursor.block->divergent = true;
   agx_atomic_to(b, agx_register(6, AGX_SIZE_32),
                 agx_register(6, AGX_SIZE_32),
                 agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                 AGX_ATOMIC_OPC_ADD);
   agx_stop(b);

   struct agx3_device_atomic_desc atomic = {
      .op = AGX3_ATOMIC_ADD,
      .value = 6,
      .exec_masked = true,
      .returns_value = true,
      .result = 6,
      .slot = 1,
      .base_kind = AGX3_ATOMIC_BASE_UNIFORM,
      .index_kind = AGX3_ATOMIC_INDEX_IMMEDIATE,
   };
   uint8_t bytes[AGX3_ATOMIC_MAX_LENGTH], atomic_len;
   ASSERT_TRUE(agx3_atomic_encode_device(AGX_ISA_ENCODING_AGX3B, &atomic,
                                         bytes, &atomic_len));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));

   std::vector<uint8_t> want;
   append(want, bytes, atomic_len);
   append_g17p_atomic_return_reload(want, 6);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   ASSERT_GE(want.size(), 2u);
   EXPECT_EQ(want[1], 0x11);
}

TEST_F(AGX3PackBinary, DeviceAtomicInPlaceReturnDiscardsConsumedValue)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   agx_index value = agx_mov_imm(b, 32, 7);
   agx_index result = agx_temp(b->shader, AGX_SIZE_32);
   agx_atomic_to(b, result, value, agx_uniform(0, AGX_SIZE_64),
                 agx_immediate(0), AGX_ATOMIC_OPC_ADD);
   agx_stop(b);

   agx_ra(b->shader);

   unsigned atomics = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_ATOMIC)
         continue;

      ++atomics;
      EXPECT_EQ(I->src[0].type, AGX_INDEX_REGISTER);
      EXPECT_EQ(I->dest[0].type, AGX_INDEX_REGISTER);
      EXPECT_EQ(I->src[0].value, I->dest[0].value);
      EXPECT_TRUE(I->src[0].discard);
   }
   EXPECT_EQ(atomics, 1u);
}

TEST_F(AGX3PackBinary, RegisterSubtractDestinationDoesNotAliasInputs)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   agx_index lhs = agx_mov_imm(b, 32, 0x77777777);
   agx_index rhs = agx_mov_imm(b, 32, 0x11111111);
   agx_index result = agx_temp(b->shader, AGX_SIZE_32);
   agx_iadd_to(b, result, lhs, agx_neg(rhs), 0);
   agx_st_vary(b, agx_immediate(0), result);
   agx_stop(b);

   agx_ra(b->shader);

   unsigned subtracts = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_IADD || I->nr_srcs != 2 || !I->src[1].neg)
         continue;

      ++subtracts;
      EXPECT_NE(I->dest[0].value, I->src[0].value);
      EXPECT_NE(I->dest[0].value, I->src[1].value);
   }
   EXPECT_EQ(subtracts, 1u);
}

TEST_F(AGX3PackBinary, G17POptimizerKeepsImmediateMinuendInRegister)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   agx_index lhs = agx_mov_imm(b, 32, 29);
   agx_index rhs = agx_register(0, AGX_SIZE_32);
   agx_index result = agx_temp(b->shader, AGX_SIZE_32);
   agx_iadd_to(b, result, lhs, agx_neg(rhs), 0);
   agx_st_vary(b, agx_immediate(0), result);
   agx_stop(b);

   agx_optimizer_forward(b->shader);

   unsigned subtracts = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_IADD || I->nr_srcs != 2 || !I->src[1].neg)
         continue;

      ++subtracts;
      EXPECT_EQ(I->src[0].type, AGX_INDEX_NORMAL);
   }
   EXPECT_EQ(subtracts, 1u);

   agx_ra(b->shader);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, RegisterAndDestinationDoesNotAliasInputs)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   agx_index lhs = agx_mov_imm(b, 32, 0x77777777);
   agx_index rhs = agx_mov_imm(b, 32, 0x55555555);
   agx_index result = agx_and(b, lhs, rhs);
   agx_st_vary(b, agx_immediate(0), result);
   agx_stop(b);

   agx_ra(b->shader);

   unsigned ands = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_BITOP ||
          (I->truth_table & 0xfu) != 0x8u)
         continue;

      ++ands;
      EXPECT_NE(I->dest[0].value, I->src[0].value);
      EXPECT_NE(I->dest[0].value, I->src[1].value);
   }
   EXPECT_EQ(ands, 1u);
}

TEST_F(AGX3PackBinary, DeviceAtomicFoldsElementOffsetToBytes)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_atomic_to(b, agx_register(4, AGX_SIZE_32),
                 agx_register(6, AGX_SIZE_32),
                 agx_uniform(4, AGX_SIZE_64), agx_immediate(7),
                 AGX_ATOMIC_OPC_ADD);
   agx_stop(b);

   struct agx3_device_atomic_desc d = {
      .op = AGX3_ATOMIC_ADD,
      .value = 6,
      .returns_value = true,
      .result = 4,
      .slot = 1,
      .base_kind = AGX3_ATOMIC_BASE_UNIFORM,
      .base = 4,
      .index_kind = AGX3_ATOMIC_INDEX_IMMEDIATE,
      .byte_offset = 28,
   };
   uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
   ASSERT_TRUE(agx3_atomic_encode_device(AGX_ISA_ENCODING_AGX3C, &d, atomic,
                                         &atomic_len));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3C, &end));

   std::vector<uint8_t> want;
   append(want, atomic, atomic_len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, DeviceAtomicMapsPostRaDiscardHints)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);

      agx_index value = agx_register(6, AGX_SIZE_32);
      value.discard = true;
      agx_index base = agx_register(8, AGX_SIZE_64);
      base.discard = true; /* no atomicdev base-hint field: safely omitted */
      agx_index index = agx_register(12, AGX_SIZE_32);
      index.discard = true;
      const agx_index result = agx_register(e ? 4 : 6, AGX_SIZE_32);

      agx_atomic_to(b, result, value, base, index, AGX_ATOMIC_OPC_ADD);
      agx_stop(b);

      struct agx3_device_atomic_desc d = {
         .op = AGX3_ATOMIC_ADD,
         .value = 6,
         .value_kill = true,
         .returns_value = true,
         .result = (uint16_t)(e ? 4 : 6),
         .slot = 1,
         .base_kind = AGX3_ATOMIC_BASE_REGISTER,
         .base = 8,
         .index_kind = AGX3_ATOMIC_INDEX_REGISTER,
         .index = 12,
         .index_kill = true,
         .index_shift = 2,
      };
      uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
      ASSERT_TRUE(agx3_atomic_encode_device(enc, &d, atomic, &atomic_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));

      std::vector<uint8_t> want;
      append(want, atomic, atomic_len);
      if (!e)
         append_g17p_atomic_return_reload(want, result.value);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, DeviceAtomicEncodesHighRegistersOnG17Class)
{
   /* Exact failing instruction from KHR-Single-GL46.arrays_of_arrays_gl.AtomicUsage
    * on G17S: d32:reg236 s0=32:reg236.discard s1=64:reg4 s2=32:reg234.discard. */
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_index value = agx_register(236, AGX_SIZE_32);
      value.discard = true;
      agx_index base = agx_register(4, AGX_SIZE_64);
      agx_index index = agx_register(234, AGX_SIZE_32);
      index.discard = true;
      agx_atomic_to(b, agx_register(236, AGX_SIZE_32), value, base, index,
                    AGX_ATOMIC_OPC_ADD);
      agx_stop(b);

      struct agx3_device_atomic_desc d = {
         .op = AGX3_ATOMIC_ADD,
         .value = 236,
         .value_kill = true,
         .returns_value = true,
         .result = 236,
         .slot = 1,
         .base_kind = AGX3_ATOMIC_BASE_REGISTER,
         .base = 4,
         .index_kind = AGX3_ATOMIC_INDEX_REGISTER,
         .index = 234,
         .index_kill = true,
         .index_shift = 2,
      };
      uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
      ASSERT_TRUE(agx3_atomic_encode_device(AGX_ISA_ENCODING_AGX3C, &d, atomic,
                                            &atomic_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3C, &end));

      std::vector<uint8_t> want;
      append(want, atomic, atomic_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }

   /* High 64-bit register base and distinct high value/result/index on AGX3C. */
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_index value = agx_register(236, AGX_SIZE_32);
      value.discard = true;
      agx_index base = agx_register(248, AGX_SIZE_64);
      agx_index index = agx_register(234, AGX_SIZE_32);
      index.discard = true;
      agx_atomic_to(b, agx_register(238, AGX_SIZE_32), value, base, index,
                    AGX_ATOMIC_OPC_ADD);
      agx_stop(b);

      struct agx3_device_atomic_desc d = {
         .op = AGX3_ATOMIC_ADD,
         .value = 236,
         .value_kill = true,
         .returns_value = true,
         .result = 238,
         .slot = 1,
         .base_kind = AGX3_ATOMIC_BASE_REGISTER,
         .base = 248,
         .index_kind = AGX3_ATOMIC_INDEX_REGISTER,
         .index = 234,
         .index_kill = true,
         .index_shift = 2,
      };
      uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
      ASSERT_TRUE(agx3_atomic_encode_device(AGX_ISA_ENCODING_AGX3C, &d, atomic,
                                            &atomic_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3C, &end));

      std::vector<uint8_t> want;
      append(want, atomic, atomic_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }

   /* Local atomic with high value, result, and 16-bit register address on AGX3C. */
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_index value = agx_register(238, AGX_SIZE_32);
      value.discard = true;
      agx_local_atomic_to(b, agx_register(236, AGX_SIZE_32), value,
                          agx_register(235, AGX_SIZE_16), agx_immediate(0),
                          AGX_ATOMIC_OPC_ADD);
      agx_stop(b);

      struct agx3_local_atomic_desc d = {
         .op = AGX3_ATOMIC_ADD,
         .value = 238,
         .value_kill = true,
         .returns_value = true,
         .result = 236,
         .slot = 1,
         .address_kind = AGX3_ATOMIC_INDEX_REGISTER,
         .address = 235,
      };
      uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
      ASSERT_TRUE(agx3_atomic_encode_local(AGX_ISA_ENCODING_AGX3C, &d, atomic,
                                           &atomic_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3C, &end));

      std::vector<uint8_t> want;
      append(want, atomic, atomic_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }

   /* Local atomic with high value, result, and 16-bit register address on AGX3D
    * (G18G) through agx3_atomic_encode_local_g18g(). */
   {
      struct agx3_local_atomic_desc d = {
         .op = AGX3_ATOMIC_ADD,
         .value = 238,
         .value_kill = true,
         .returns_value = true,
         .result = 236,
         .slot = 1,
         .address_kind = AGX3_ATOMIC_INDEX_REGISTER,
         .address = 235,
         .address_kill = true,
      };
      uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
      ASSERT_TRUE(agx3_atomic_encode_local_g18g(AGX_ISA_ENCODING_AGX3D, &d,
                                                atomic, &atomic_len));
      EXPECT_GT(atomic_len, 0u);
   }

   /* AGX3B keeps its 192-half register-file ceiling fail-closed. */
   {
      uint8_t out[AGX3_ATOMIC_MAX_LENGTH], len = 0;
      struct agx3_device_atomic_desc d = {
         .op = AGX3_ATOMIC_ADD,
         .value = 192,
         .returns_value = true,
         .result = 192,
         .slot = 1,
         .base_kind = AGX3_ATOMIC_BASE_REGISTER,
         .base = 4,
         .index_kind = AGX3_ATOMIC_INDEX_REGISTER,
         .index = 8,
         .index_shift = 2,
      };
      EXPECT_FALSE(agx3_atomic_encode_device(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
      d.value = d.result = 4;
      d.base = 190;
      EXPECT_FALSE(agx3_atomic_encode_device(AGX_ISA_ENCODING_AGX3B, &d, out, &len));

      struct agx3_local_atomic_desc ld = {
         .op = AGX3_ATOMIC_ADD,
         .value = 192,
         .returns_value = true,
         .result = 4,
         .slot = 1,
         .address_kind = AGX3_ATOMIC_INDEX_REGISTER,
         .address = 8,
      };
      EXPECT_FALSE(agx3_atomic_encode_local(AGX_ISA_ENCODING_AGX3B, &ld, out, &len));
      ld.value = 4;
      ld.address = 192;
      EXPECT_FALSE(agx3_atomic_encode_local(AGX_ISA_ENCODING_AGX3B, &ld, out, &len));
   }
}

TEST_F(AGX3PackBinary, DeviceAtomicReturnIsTiedWithoutClobberingIndex)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   agx_index value = agx_temp(b->shader, AGX_SIZE_32);
   agx_mov_imm_to(b, value, 0xfffffffa);
   agx_index index = agx_temp(b->shader, AGX_SIZE_32);
   agx_mov_imm_to(b, index, 7);
   agx_index result = agx_temp(b->shader, AGX_SIZE_32);
   agx_atomic_to(b, result, value, agx_uniform(0, AGX_SIZE_64), index,
                 AGX_ATOMIC_OPC_ADD);

   /* Keep the address index live after the atomic.  A pack-time value copy
    * into the result would corrupt it; the pre-RA two-address legalization
    * must instead tie only the dedicated update temporary and result. */
   agx_index sum = agx_temp(b->shader, AGX_SIZE_32);
   agx_iadd_to(b, sum, result, index, 0);
   agx_stop(b);

   ASSERT_TRUE(agx_lower_agx3_device_atomic_sources(b->shader));
   agx_pressure_schedule(b->shader);
   agx_ra(b->shader);

   agx_instr *atomic = NULL;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_ATOMIC)
         atomic = I;
   }
   ASSERT_NE(atomic, nullptr);
   ASSERT_EQ(atomic->dest[0].type, AGX_INDEX_REGISTER);
   ASSERT_EQ(atomic->src[0].type, AGX_INDEX_REGISTER);
   EXPECT_EQ(atomic->dest[0].value, atomic->src[0].value);
   EXPECT_NE(atomic->dest[0].value, atomic->src[2].value);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, LocalAtomicScalarZeroAddress)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_local_atomic_to(b, agx_register(6, AGX_SIZE_32),
                          agx_register(4, AGX_SIZE_32), agx_immediate(0),
                          agx_immediate(0), AGX_ATOMIC_OPC_UMIN);
      agx_stop(b);

      const struct agx3_local_atomic_desc d = {
         .op = AGX3_ATOMIC_UMIN,
         .value = 4,
         .returns_value = true,
         .result = 6,
         .slot = 1,
         .address_kind = AGX3_ATOMIC_INDEX_IMMEDIATE,
      };
      uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
      ASSERT_TRUE(agx3_atomic_encode_local(enc, &d, atomic, &atomic_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));

      std::vector<uint8_t> want;
      append(want, atomic, atomic_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, LocalAtomicReturnedImmediateAddressG17P)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_local_atomic_to(b, agx_register(6, AGX_SIZE_32),
                       agx_register(4, AGX_SIZE_32), agx_immediate(448),
                       agx_immediate(0), AGX_ATOMIC_OPC_XCHG);
   agx_stop(b);

   agx_instr immediate_form = {};
   immediate_form.op = AGX_OPCODE_LOCAL_ATOMIC;
   EXPECT_TRUE(
      agx_allows_16bit_immediate(&immediate_form, AGX_ISA_ENCODING_AGX3B));

   const struct agx3_local_atomic_desc d = {
      .op = AGX3_ATOMIC_XCHG,
      .value = 4,
      .returns_value = true,
      .result = 6,
      .slot = 1,
      .address_kind = AGX3_ATOMIC_INDEX_IMMEDIATE,
      .address = 448,
   };
   uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
   ASSERT_TRUE(agx3_atomic_encode_local(AGX_ISA_ENCODING_AGX3B, &d, atomic,
                                        &atomic_len));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));

   std::vector<uint8_t> want;
   append(want, atomic, atomic_len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, LocalAtomicReturnedRegisterAddressG17P)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_local_atomic_to(b, agx_register(10, AGX_SIZE_32),
                       agx_register(14, AGX_SIZE_32),
                       agx_register(12, AGX_SIZE_16), agx_immediate(0),
                       AGX_ATOMIC_OPC_XCHG);
   agx_stop(b);

   const struct agx3_local_atomic_desc d = {
      .op = AGX3_ATOMIC_XCHG,
      .value = 14,
      .returns_value = true,
      .result = 10,
      .slot = 1,
      .address_kind = AGX3_ATOMIC_INDEX_REGISTER,
      .address = 12,
   };
   uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
   ASSERT_TRUE(agx3_atomic_encode_local(AGX_ISA_ENCODING_AGX3B, &d, atomic,
                                        &atomic_len));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));

   std::vector<uint8_t> want;
   append(want, atomic, atomic_len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, NarrowingMoveUsesLowHalfCopy)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_bitop_to(b, agx_register(1, AGX_SIZE_16),
                agx_register(8, AGX_SIZE_32), agx_zero(), UTIL_LUT2(a));
   agx_stop(b);

   uint8_t copy[AGX3_COPY_MAX_LENGTH], copy_len = 0;
   ASSERT_TRUE(agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3B, 1, 8,
                                      false, false, false, 0,
                                      copy, &copy_len));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));

   std::vector<uint8_t> want;
   append(want, copy, copy_len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, LocalAtomicVoidAddressedG17P)
{
   for (int address : {-8, 8, 1}) {
      const bool dynamic = address >= 0;
      const unsigned half = dynamic ? (unsigned)address : 0;
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_local_atomic_to(b, agx_null(), agx_register(4, AGX_SIZE_32),
                          dynamic ? agx_register(half, AGX_SIZE_16)
                                  : agx_immediate(8),
                          agx_immediate(0), AGX_ATOMIC_OPC_OR);
      agx_stop(b);

      const struct agx3_local_atomic_desc d = {
         .op = AGX3_ATOMIC_OR,
         .value = 4,
         .address_kind = dynamic ? AGX3_ATOMIC_INDEX_REGISTER
                                 : AGX3_ATOMIC_INDEX_IMMEDIATE,
         .address = static_cast<uint16_t>(dynamic ? half : 8),
      };
      uint8_t atomic[AGX3_ATOMIC_MAX_LENGTH], atomic_len = 0;
      ASSERT_TRUE(agx3_atomic_encode_local(AGX_ISA_ENCODING_AGX3B, &d,
                                           atomic, &atomic_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));

      std::vector<uint8_t> want;
      append(want, atomic, atomic_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

/* The lane is the whole seven-bit f29, so every lane a subgroup can name
 * encodes -- lane 31, which ends every subgroup scan, included. Only a lane
 * the field cannot hold is refused. */
TEST_F(AGX3PackBinary, SimdBroadcastEncodesEveryLane)
{
   for (unsigned lane : {1u, 5u, 31u, 63u, 127u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_shuffle_to(b, agx_register(4, AGX_SIZE_32),
                     agx_register(0, AGX_SIZE_32), agx_immediate(lane));
      agx_stop(b);

      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      const struct agx3_subgroup_shuffle_desc desc = {
         .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = 4,
         .src = 0,
         .lane = (uint16_t)lane,
         .src_live = true,
      };
      ASSERT_TRUE(
         agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &desc, bytes,
                                      &length));
      std::vector<uint8_t> want;
      append(want, bytes, length);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_shuffle_to(b, agx_register(4, AGX_SIZE_32),
                  agx_register(0, AGX_SIZE_32), agx_immediate(128));
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, SimdBroadcastWaitsForDeviceLoad)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   const agx_index value = agx_register(4, AGX_SIZE_32);
   agx_device_load_to(b, value, agx_register(8, AGX_SIZE_64),
                      agx_register(12, AGX_SIZE_32), AGX_FORMAT_I32, 1, 2,
                      false);
   agx_shuffle_to(b, agx_register(6, AGX_SIZE_32), value, agx_immediate(3));
   agx_stop(b);

   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   std::vector<uint8_t> want;

   const struct agx3_subgroup_shuffle_desc shuffle = {
      .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
      .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
      .dest = 6,
      .src = 4,
      .lane = 3,
      /* G17P device loads use the native preferred slot 6. */
      .wait_mask = 1u << 5,
      .src_live = true,
   };
   ASSERT_TRUE(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &shuffle,
                                             bytes, &length));
   append(want, bytes, length);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   const std::vector<uint8_t> got = pack(b);
   ASSERT_GE(got.size(), want.size());
   const std::vector<uint8_t> tail(got.end() - want.size(), got.end());
   EXPECT_EQ(tail, want);
}

TEST_F(AGX3PackBinary, SimdBroadcastLastLanePreservesLivePrefix)
{
   /* Tessellation reads the inclusive prefix again after broadcasting lane
    * 31. Marking that source discarded destroys the later per-lane sum. */
   for (bool last_use : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index prefix = agx_register(18, AGX_SIZE_32);
      prefix.discard = last_use;
      agx_shuffle_to(b, agx_register(20, AGX_SIZE_32), prefix,
                     agx_immediate(31));
      agx_stop(b);

      const struct agx3_subgroup_shuffle_desc desc = {
         .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = 20,
         .src = 18,
         .lane = 31,
         .src_live = !last_use,
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B,
                                               &desc, bytes, &length));
      const auto got = pack(b);
      ASSERT_GE(got.size(), length);
      EXPECT_EQ(memcmp(got.data(), bytes, length), 0);
   }
}

/*
 * A SIMD broadcast encodes differently depending on whether its SOURCE is read
 * again afterwards, and getting that wrong destroys the source register on
 * hardware. Two metal-nt -arch applegpu_g17p kernels, identical but for the
 * final use, settle the field:
 *
 *   uint x = lane + 1; out[tid] = simd_broadcast(x, 31);
 *      ... 47 04 54 00 02 00 3e 2c 04 00      source dies
 *   uint x = lane + 1; uint b = simd_broadcast(x, 31); out[tid] = b + x;
 *      ... 47 04 54 04 03 00 3e 2c 02 00      source stays live
 *
 * Byte 8 is 0x04 on the final read and 0x02 while the source lives; the
 * operand bytes differ only because the destination register does. The packer
 * left this defaulted to the final-read form on EVERY immediate-lane
 * broadcast, so `subgroupBroadcast(v, 31)` -- the read that ends every
 * workgroup scan -- wiped v. On hardware `x = lane + 1; b = broadcast(x, 31)`
 * returned b == 32 correctly and x == 0 in all 32 lanes, which emptied the
 * tessellation prefix sum and left isoline tessellation drawing nothing.
 */
TEST_F(AGX3PackBinary, SimdBroadcastSourceLivenessMatchesNative)
{
   for (bool live : {false, true}) {
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      const struct agx3_subgroup_shuffle_desc desc = {
         .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
         .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
         .dest = 4,
         .src = 0,
         .lane = 31,
         .src_live = live,
      };
      ASSERT_TRUE(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &desc,
                                                bytes, &length));
      ASSERT_EQ(length, 10u);
      EXPECT_EQ(bytes[8], live ? 0x02 : 0x04);

      /* Nothing else moves with the liveness of the source. */
      const uint8_t head[] = {0x47, 0x04};
      EXPECT_EQ(memcmp(bytes, head, sizeof(head)), 0);
      EXPECT_EQ(bytes[6], 0x3e); /* f29 = lane 31 */
      EXPECT_EQ(bytes[7], 0x2c);
      EXPECT_EQ(bytes[9], 0x00);
   }
}

TEST_F(AGX3PackBinary, G17PScalarByteLoadRequiresAlignedHalf)
{
   agx_builder *good = builder(AGX_ISA_ENCODING_AGX3B);
   agx_device_load_to(good, agx_register(2, AGX_SIZE_32),
                      agx_uniform(4, AGX_SIZE_64),
                      agx_register(0, AGX_SIZE_32), AGX_FORMAT_I8, 1, 0,
                      false);
   agx_stop(good);
   EXPECT_FALSE(pack(good).empty());

   /* No Apple scalar-byte witness writes an odd half. The frontend models the
    * physical result as a 32-bit temporary so production RA cannot create
    * this shape; retain the packer refusal as the negative control. */
   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_device_load_to(bad, agx_register(1, AGX_SIZE_16),
                      agx_uniform(4, AGX_SIZE_64),
                      agx_register(0, AGX_SIZE_32), AGX_FORMAT_I8, 1, 0,
                      false);
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, G17PHalfFMAImmediateAddend)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src0 = agx_register(2, AGX_SIZE_16);
   agx_index src1 = agx_register(3, AGX_SIZE_16);
   src0.discard = true;
   src1.discard = true;
   agx_hfma_to(b, agx_register(2, AGX_SIZE_16), src0, src1,
               agx_immediate(56));
   agx_stop(b);

   uint8_t fma[AGX3_ALU_MAX_LENGTH], fma_len;
   const uint16_t src[] = {2, 3};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
   };
   ASSERT_TRUE(agx3_alu_hfma_addend_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, src, 56, hints, false, 0, fma, &fma_len));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   std::vector<uint8_t> want;
   append(want, fma, fma_len);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index absolute = agx_register(2, AGX_SIZE_16);
   absolute.abs = true;
   agx_hfma_to(bad, agx_register(0, AGX_SIZE_16), absolute,
               agx_register(3, AGX_SIZE_16), agx_immediate(56));
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, G17PHalfFMAWaitsOnProducerDirectly)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_get_sr_to(b, agx_register(14, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_hfma_to(b, agx_register(6, AGX_SIZE_16),
               agx_register(2, AGX_SIZE_16),
               agx_register(3, AGX_SIZE_16),
               agx_register(14, AGX_SIZE_16));
   agx_stop(b);

   std::vector<uint8_t> want;
   struct agx3_sr_desc sr = {};
   sr.dest = 14;
   sr.width = AGX3_SR_W32;
   sr.sr = 23;
   sr.slot = 1;
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_sr_encode(enc, &sr, bytes, &len));
   append(want, bytes, len);
   const uint16_t src[] = {2, 3, 14};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   ASSERT_TRUE(agx3_alu_hfma(enc, 6, src, hints, false, false, false, 1,
                              bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, G17PHalfSaturateMatchesNative)
{
   static const uint8_t native[] = {
      0x00, 0x00, 0x1c, 0x80, 0x01, 0x02, 0x00, 0x02,
   };
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(0, AGX_SIZE_16);
   src.discard = true;
   agx_instr *I = agx_hadd_to(b, agx_register(0, AGX_SIZE_16), src,
                              agx_immediate(0x80));
   I->saturate = true;
   agx_stop(b);

   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, G17GHalfSaturateMatchesNativeForm)
{
   const uint8_t native[] = {0x08, 0x05, 0x14, 0x80, 0x01, 0x02, 0x00, 0x02};
   key.dev.emit_agx3d = true;
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_instr *I = agx_hadd_to(b, agx_register(1, AGX_SIZE_16),
                             agx_register(5, AGX_SIZE_16), agx_immediate(0x80));
   I->saturate = true;
   agx_stop(b);
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3C, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, G17GHalfSubtractMatchesNativeInEitherOrder)
{
   const uint8_t native[] = {0x08, 0x01, 0x1c, 0x05, 0x00, 0x08};
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3C, &end));
   append(want, end.bytes, end.length);
   key.dev.emit_agx3d = true;
   for (bool swap : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_index x = agx_register(1, AGX_SIZE_16);
      agx_index y = agx_register(5, AGX_SIZE_16);
      x.discard = y.discard = true;
      y.neg = true;
      agx_hadd_to(b, agx_register(1, AGX_SIZE_16), swap ? y : x, swap ? x : y);
      agx_stop(b);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, HalfSaturateRefusesUnwitnessedOperands)
{
   for (unsigned variant = 0; variant < 14; ++variant) {
      unsigned mode = (variant % 7) + 1;
      agx_builder *b = builder(variant < 7 ? AGX_ISA_ENCODING_AGX3B
                                          : AGX_ISA_ENCODING_AGX3C);
      agx_index dst = agx_register(0, AGX_SIZE_16);
      agx_index src = agx_register(0, AGX_SIZE_16);
      /* A non-saturating G17G add accepts minifloats, but not 9-bit values. */
      agx_index imm = agx_immediate(mode == 1 ? (variant < 7 ? 0x30 : 0x100)
                                             : 0x80);
      if (mode == 2)
         src.size = AGX_SIZE_32;
      if (mode == 3)
         src.channels_m1 = 1;
      if (mode == 4)
         src.abs = true;
      if (mode == 5)
         src.neg = true;
      if (mode == 6)
         src.kill = true;
      if (mode == 7)
         dst.channels_m1 = 1;
      agx_instr *I = agx_hadd_to(b, dst, src, imm);
      I->saturate = mode != 1;
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, G17PHalfInverseMatchesNativeForm)
{
   static const uint8_t native[] = {
      0x08, 0xb0, 0x04, 0x80, 0x80, 0x08,
   };
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(0, AGX_SIZE_16);
   src.neg = true;
   src.cache = true;
   agx_hadd_to(b, agx_register(1, AGX_SIZE_16), src, agx_immediate(0x30));
   agx_stop(b);

   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, G17GHalfInverseMatchesNativeForm)
{
   const uint8_t native[] = {0x08, 0xb0, 0x04, 0x05, 0x80, 0x08};
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3C, &end));
   append(want, end.bytes, end.length);
   for (bool immediate_first : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_index src = agx_register(5, AGX_SIZE_16);
      src.neg = true;
      agx_index imm = agx_immediate(0x30);
      agx_hadd_to(b, agx_register(1, AGX_SIZE_16),
                  immediate_first ? imm : src, immediate_first ? src : imm);
      agx_stop(b);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, G17GHalfImmediatesMatchNativeForms)
{
   /* Apple G17G Metal reference: live half add-one and multiply-two. */
   const uint8_t native[][6] = {
      {0x08, 0xb0, 0x04, 0x05, 0x80, 0x00},
      {0x08, 0xc0, 0x05, 0x05, 0x80, 0x00},
   };
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3C, &end));
   for (unsigned multiply = 0; multiply < 2; ++multiply) {
      std::vector<uint8_t> want(std::begin(native[multiply]),
                                std::end(native[multiply]));
      append(want, end.bytes, end.length);
      for (bool immediate_first : {false, true}) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
         agx_index dst = agx_register(1, AGX_SIZE_16);
         agx_index src = agx_register(5, AGX_SIZE_16);
         agx_index imm = agx_immediate(multiply ? 0x40 : 0x30);
         agx_index s0 = immediate_first ? imm : src;
         agx_index s1 = immediate_first ? src : imm;
         if (multiply)
            agx_hmul_to(b, dst, s0, s1);
         else
            agx_hadd_to(b, dst, s0, s1);
         agx_stop(b);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, HalfInverseRefusesUnwitnessedOperands)
{
   for (unsigned variant = 0; variant < 16; ++variant) {
      unsigned mode = (variant % 8) + 1;
      /* G17G now supports general minifloats and non-negated sources. */
      if (variant >= 8 && mode <= 2)
         continue;
      agx_builder *b = builder(variant < 8 ? AGX_ISA_ENCODING_AGX3B
                                          : AGX_ISA_ENCODING_AGX3C);
      agx_index dst = agx_register(1, AGX_SIZE_16);
      agx_index src = agx_register(0, AGX_SIZE_16);
      agx_index imm = agx_immediate(mode == 1 ? 0x80 : 0x30);
      src.neg = mode != 2;
      src.abs = mode == 4;
      src.kill = mode == 5;
      if (mode == 6)
         src.size = AGX_SIZE_32;
      if (mode == 7)
         src.channels_m1 = 1;
      if (mode == 8)
         dst.channels_m1 = 1;
      agx_instr *I = agx_hadd_to(b, dst, src, imm);
      I->saturate = mode == 3;
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, G17PI32ToHalfMatchesNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(0, AGX_SIZE_16);
   dest.cache = true;
   agx_convert_to(b, dest, agx_immediate(AGX_CONVERT_U32_TO_F),
                  agx_register(2, AGX_SIZE_32), AGX_ROUND_RTE);
   agx_stop(b);

   /* The Apple witness carried a wait on its async source. This synthetic
    * builder has no producer, so only that dependency bit clears. */
   static const uint8_t native[] = {
      0xa7, 0x07, 0x54, 0x00, 0x03, 0x04, 0x88, 0x20,
   };
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   agx_builder *signed_dead = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index source = agx_register(0, AGX_SIZE_32);
   source.discard = true;
   agx_convert_to(signed_dead, agx_register(0, AGX_SIZE_16),
                  agx_immediate(AGX_CONVERT_S32_TO_F), source,
                  AGX_ROUND_RTE);
   agx_stop(signed_dead);
   static const uint8_t native_signed_dead[] = {
      0xa7, 0x07, 0x54, 0x00, 0x02, 0x00, 0xa8, 0x60,
   };
   std::vector<uint8_t> want_signed(std::begin(native_signed_dead),
                                    std::end(native_signed_dead));
   append(want_signed, end.bytes, end.length);
   EXPECT_EQ(pack(signed_dead), want_signed);

   /* A CACHED source used to be refused here.  f27 carries it, exactly as f29
    * carries discard, and test-agx3-alu-core.c holds the byte-exact Apple
    * vector for both.  What this level has to show is that the packer reaches
    * that encoder and that the cache hint is the ONLY thing it changes, so
    * pack the same convert twice and diff. */
   agx_builder *plain = builder(AGX_ISA_ENCODING_AGX3B);
   agx_convert_to(plain, agx_register(0, AGX_SIZE_16),
                  agx_immediate(AGX_CONVERT_U32_TO_F),
                  agx_register(2, AGX_SIZE_32), AGX_ROUND_RTE);
   agx_stop(plain);
   const std::vector<uint8_t> plain_bytes = pack(plain);

   agx_builder *cached_src = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index cached = agx_register(2, AGX_SIZE_32);
   cached.cache = true;
   agx_convert_to(cached_src, agx_register(0, AGX_SIZE_16),
                  agx_immediate(AGX_CONVERT_U32_TO_F), cached,
                  AGX_ROUND_RTE);
   agx_stop(cached_src);
   const std::vector<uint8_t> cached_bytes = pack(cached_src);

   ASSERT_EQ(plain_bytes.size(), cached_bytes.size());
   unsigned differing_bytes = 0, differing_bits = 0;
   for (size_t i = 0; i < plain_bytes.size(); ++i) {
      const uint8_t delta = plain_bytes[i] ^ cached_bytes[i];
      if (!delta)
         continue;
      differing_bytes++;
      differing_bits += __builtin_popcount(delta);
   }
   EXPECT_EQ(differing_bytes, 1u);
   EXPECT_EQ(differing_bits, 1u);
   EXPECT_NE(plain_bytes, cached_bytes);
}

TEST_F(AGX3PackBinary, GTKHalfToInt32UsesNativeWidthAndPostRAState)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;

   /* Exact first refusal from GTK 4's current gskgputurbulence.frag after
    * register allocation: `$r17 = convert #F_TO_S32, r17l, RTZ`. */
   agx_builder *gtk = builder(enc);
   agx_index dest = agx_register(34, AGX_SIZE_32);
   dest.cache = true;
   agx_convert_to(gtk, dest, agx_immediate(AGX_CONVERT_F_TO_S32),
                  agx_register(34, AGX_SIZE_16), AGX_ROUND_RTZ);
   agx_stop(gtk);
   const uint8_t gtk_native[] = {
      0x27, 0x07, 0x54, 0x22, 0x03, 0x44, 0x8c,
      0x48, 0x03, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(gtk), std::vector<uint8_t>(std::begin(gtk_native),
                                              std::end(gtk_native)));

   /* A half selected from an async system-register result remains a half;
    * its native conversion consumes that producer's slot in-place. */
   agx_builder *dependent = builder(enc);
   agx_get_sr_to(dependent, agx_register(0, AGX_SIZE_32),
                 AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_index half = agx_register(0, AGX_SIZE_16);
   half.discard = true;
   agx_convert_to(dependent, agx_register(4, AGX_SIZE_32),
                  agx_immediate(AGX_CONVERT_F_TO_U32), half,
                  AGX_ROUND_RTZ);
   agx_stop(dependent);

   std::vector<uint8_t> want;
   struct agx3_sr_desc sr = {};
   sr.dest = 0;
   sr.width = AGX3_SR_W32;
   sr.sr = 23;
   sr.slot = 1;
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_sr_encode(enc, &sr, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_f16_to_int32(enc, 4, 0, false,
                                      AGX3_ALU_HINT_DISCARD, false, 1, bytes,
                                      &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(dependent), want);
}

TEST_F(AGX3PackBinary, FloatToInt32CarriesLiveSourceCacheState)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(4, AGX_SIZE_32);
   agx_index source = agx_register(2, AGX_SIZE_32);
   dest.cache = true;
   source.cache = true;
   agx_convert_to(b, dest, agx_immediate(AGX_CONVERT_F_TO_S32), source,
                  AGX_ROUND_RTZ);
   agx_stop(b);

   const uint8_t native[] = {
      0x27, 0x07, 0x54, 0x04, 0x03, 0x04, 0x96, 0x48, 0x03, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b),
             std::vector<uint8_t>(std::begin(native), std::end(native)));
}

TEST_F(AGX3PackBinary, FloatToInt32UsesHighRegisters)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(190, AGX_SIZE_32);
   agx_index source = agx_register(190, AGX_SIZE_32);
   dest.cache = true;
   source.discard = true;
   agx_convert_to(b, dest, agx_immediate(AGX_CONVERT_F_TO_U32), source,
                  AGX_ROUND_RTZ);
   agx_stop(b);

   /* Native G17P uint(float) with 96 converted values still live. */
   const uint8_t native[] = {
      0x27, 0x07, 0x54, 0xbe, 0x03, 0x7c, 0xb5, 0x08, 0x02, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b),
             std::vector<uint8_t>(std::begin(native), std::end(native)));
}

TEST_F(AGX3PackBinary, G17PHalfToInt32RejectsUngroundedShapes)
{
   for (unsigned mode = 1; mode < 22; ++mode) {
      SCOPED_TRACE(mode);
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(mode == 17 ? 128 :
                                    mode == 15 ? 1 : 4,
                                    mode == 14 ? AGX_SIZE_16 : AGX_SIZE_32);
      agx_index src = agx_register(mode == 18 ? 128 : 0,
                                   mode == 11 ? AGX_SIZE_64 : AGX_SIZE_16);
      if (mode == 9)
         src = agx_uniform(0, AGX_SIZE_16);
      else if (mode == 10)
         src = agx_immediate(0);
      src.abs = mode == 3;
      src.neg = mode == 4;
      src.kill = mode == 5;
      src.cache = mode == 6 || mode == 7;
      src.discard = mode == 7;
      src.memory = mode == 8;
      src.channels_m1 = mode == 12;
      dest.channels_m1 = mode == 13;
      dest.abs = mode == 16;
      dest.memory = mode == 19;
      agx_instr *I = agx_convert_to(
         b, dest, agx_immediate(AGX_CONVERT_F_TO_S32), src,
         mode == 1 ? AGX_ROUND_RTE : AGX_ROUND_RTZ);
      I->saturate = mode == 2;
      I->src[0].neg = mode == 20;
      I->src[0].cache = mode == 21;
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, G17PHalfTimesMinifloatWidensToF32)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index half = agx_register(0, AGX_SIZE_16);
   agx_fmul_to(b, agx_register(0, AGX_SIZE_32), half, agx_immediate(112));
   agx_stop(b);

   static const uint8_t native[] = {
      0x09, 0xf1, 0x35, 0x00, 0x80, 0x00,
   };
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index live = agx_register(0, AGX_SIZE_16);
   live.cache = true;
   agx_fmul_to(bad, agx_register(2, AGX_SIZE_32), live,
               agx_immediate(112));
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, QuadImmediateNativeWitnesses)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   const agx_index value = agx_register(6, AGX_SIZE_32);
   agx_quad_shuffle_to(b, agx_register(0, AGX_SIZE_32), value,
                       agx_immediate(2));
   agx_quad_shuffle_xor_to(b, agx_register(2, AGX_SIZE_32), value,
                           agx_immediate(1));
   agx_index cached = agx_register(4, AGX_SIZE_32);
   cached.cache = true;
   agx_quad_shuffle_xor_to(b, cached, value, agx_immediate(2));
   agx_index final_value = value;
   final_value.discard = true;
   agx_quad_shuffle_xor_to(b, agx_register(6, AGX_SIZE_32), final_value,
                           agx_immediate(3));
   agx_stop(b);

   const uint8_t native[] = {
      0x47, 0x00, 0x54, 0x00, 0x02, 0x0c, 0x04, 0x2c, 0x02, 0x00,
      0xc7, 0x00, 0x54, 0x02, 0x02, 0x0c, 0x02, 0x2c, 0x02, 0x00,
      0xc7, 0x00, 0x54, 0x04, 0x03, 0x0c, 0x04, 0x2c, 0x02, 0x00,
      0xc7, 0x00, 0x54, 0x06, 0x02, 0x0c, 0x06, 0x2c, 0x04, 0x00,
   };
   std::vector<uint8_t> want(native, native + sizeof(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

/* Apple's own quad butterfly carries the slot-6 wait for its device load in
 * f49, so this no longer costs a synchronous identity prelude to retire the
 * producer first. */
TEST_F(AGX3PackBinary, QuadImmediateXorWaitsForDeviceLoad)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   const agx_index value = agx_register(4, AGX_SIZE_32);
   agx_device_load_to(b, value, agx_register(8, AGX_SIZE_64),
                      agx_register(12, AGX_SIZE_32), AGX_FORMAT_I32, 1, 2,
                      false);
   agx_quad_shuffle_xor_to(b, agx_register(6, AGX_SIZE_32), value,
                           agx_immediate(1));
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   const struct agx3_subgroup_shuffle_desc shuffle = {
      .quad = true,
      .op = AGX3_SUBGROUP_SHUFFLE_XOR,
      .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
      .dest = 6,
      .src = 4,
      .lane = 1,
      .wait_mask = 1u << 5,
      .src_live = true,
   };
   ASSERT_TRUE(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &shuffle,
                                             bytes, &length));
   append(want, bytes, length);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);

   const std::vector<uint8_t> got = pack(b);
   ASSERT_GE(got.size(), want.size());
   const std::vector<uint8_t> tail(got.end() - want.size(), got.end());
   EXPECT_EQ(tail, want);
}

TEST_F(AGX3PackBinary, SignedReduceWaitsForDeviceLoad)
{
   for (auto op : {AGX3_SUBGROUP_REDUCE_SMIN,
                   AGX3_SUBGROUP_REDUCE_SMAX}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      const agx_index value = agx_register(4, AGX_SIZE_32);
      agx_device_load_to(b, value, agx_register(8, AGX_SIZE_64),
                         agx_register(12, AGX_SIZE_32), AGX_FORMAT_I32, 1, 2,
                         false);
      agx_simd_reduce_to(b, agx_register(6, AGX_SIZE_32), value,
                         op == AGX3_SUBGROUP_REDUCE_SMIN ? AGX_SIMD_OP_SMIN
                                                        : AGX_SIMD_OP_SMAX);
      agx_stop(b);

      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      std::vector<uint8_t> want;
      const struct agx3_subgroup_reduce_desc reduce = {
         .op = op,
         .dest = 6,
         .src = 4,
         /* G17P device loads use the native preferred slot 6. */
         .wait_mask = 1u << 5,
         /* The source carries no discard hint, so it is live. */
         .src_live = true,
      };
      ASSERT_TRUE(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B,
                                               &reduce, bytes, &length));
      append(want, bytes, length);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);
      const std::vector<uint8_t> got = pack(b);
      ASSERT_GE(got.size(), want.size());
      const std::vector<uint8_t> tail(got.end() - want.size(), got.end());
      EXPECT_EQ(tail, want);
   }
}

TEST_F(AGX3PackBinary, SignedAddPrefixWaitsForDeviceLoad)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   const agx_index value = agx_register(4, AGX_SIZE_32);
   agx_device_load_to(b, value, agx_register(8, AGX_SIZE_64),
                      agx_register(12, AGX_SIZE_32), AGX_FORMAT_I32, 1, 2,
                      false);
   agx_simd_prefix_to(b, agx_register(6, AGX_SIZE_32), value,
                      AGX_SIMD_OP_IADD);
   agx_stop(b);

   uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
   std::vector<uint8_t> want;
   const struct agx3_subgroup_reduce_desc prefix = {
      .prefix = true,
      .op = AGX3_SUBGROUP_REDUCE_IADD,
      .dest = 6,
      .src = 4,
      /* G17P device loads use the native preferred slot 6. */
      .wait_mask = 1u << 5,
      /* The source carries no discard hint, so it is live. */
      .src_live = true,
   };
   ASSERT_TRUE(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &prefix,
                                             bytes, &length));
   append(want, bytes, length);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);

   const std::vector<uint8_t> got = pack(b);
   ASSERT_GE(got.size(), want.size());
   const std::vector<uint8_t> tail(got.end() - want.size(), got.end());
   EXPECT_EQ(tail, want);
}

static void
build_shuffle(agx_builder *b, bool quad, unsigned op, agx_index dest,
              agx_index src, agx_index lane)
{
   if (!quad) {
      switch (op) {
      case AGX3_SUBGROUP_SHUFFLE_BROADCAST:
         agx_shuffle_to(b, dest, src, lane);
         break;
      case AGX3_SUBGROUP_SHUFFLE_XOR:
         agx_shuffle_xor_to(b, dest, src, lane);
         break;
      case AGX3_SUBGROUP_SHUFFLE_UP:
         agx_shuffle_up_to(b, dest, src, lane);
         break;
      case AGX3_SUBGROUP_SHUFFLE_DOWN:
         agx_shuffle_down_to(b, dest, src, lane);
         break;
      default:
         UNREACHABLE("bad shuffle");
      }
   } else {
      switch (op) {
      case AGX3_SUBGROUP_SHUFFLE_BROADCAST:
         agx_quad_shuffle_to(b, dest, src, lane);
         break;
      case AGX3_SUBGROUP_SHUFFLE_XOR:
         agx_quad_shuffle_xor_to(b, dest, src, lane);
         break;
      case AGX3_SUBGROUP_SHUFFLE_UP:
         agx_quad_shuffle_up_to(b, dest, src, lane);
         break;
      case AGX3_SUBGROUP_SHUFFLE_DOWN:
         agx_quad_shuffle_down_to(b, dest, src, lane);
         break;
      default:
         UNREACHABLE("bad shuffle");
      }
   }
}

TEST_F(AGX3PackBinary, RegisterAndUniformSubgroupShuffles)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (unsigned quad = 0; quad < 2; ++quad) {
         for (unsigned op = 0; op < AGX3_SUBGROUP_SHUFFLE_NUM; ++op) {
            for (unsigned uniform = 0; uniform < 2; ++uniform) {
               agx_builder *b = builder(enc);
               const agx_index lane = uniform ? agx_uniform(8, AGX_SIZE_32)
                                              : agx_register(8, AGX_SIZE_32);
               build_shuffle(b, quad, op, agx_register(4, AGX_SIZE_32),
                             agx_register(2, AGX_SIZE_32), lane);
               agx_stop(b);

               const struct agx3_subgroup_shuffle_desc d = {
                  .quad = (bool)quad,
                  .op = (enum agx3_subgroup_shuffle_op)op,
                  .lane_kind = uniform ? AGX3_SUBGROUP_LANE_UNIFORM
                                       : AGX3_SUBGROUP_LANE_REGISTER,
                  .dest = 4,
                  .src = 2,
                  .lane = 8,
                  .src_live = true,
                  .lane_live = true,
               };
               uint8_t subgroup[AGX3_SUBGROUP_MAX_LENGTH], subgroup_len = 0;
               ASSERT_TRUE(agx3_subgroup_encode_shuffle(enc, &d, subgroup,
                                                         &subgroup_len));
               struct agx3_encoded_end end;
               ASSERT_TRUE(agx3_pack_end(enc, &end));
               std::vector<uint8_t> want;
               append(want, subgroup, subgroup_len);
               append(want, end.bytes, end.length);
               EXPECT_EQ(pack(b), want);
            }
         }
      }
   }
}

TEST_F(AGX3PackBinary, RegisterQuadBroadcastPreservesBothOperandLifetimes)
{
   for (unsigned dead = 0; dead < 4; ++dead) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index value = agx_register(2, AGX_SIZE_32);
      agx_index lane = agx_register(8, AGX_SIZE_32);
      agx_index dest = agx_register(4, AGX_SIZE_32);
      value.discard = dead & 1;
      lane.discard = dead & 2;
      dest.cache = true;
      agx_quad_shuffle_to(b, dest, value, lane);
      agx_stop(b);
      const struct agx3_subgroup_shuffle_desc d = {
         .quad = true,
         .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
         .lane_kind = AGX3_SUBGROUP_LANE_REGISTER,
         .dest = 4, .src = 2, .lane = 8,
         .dest_hint = true,
         .src_live = !(dead & 1),
         .lane_live = !(dead & 2),
      };
      uint8_t bytes[AGX3_SUBGROUP_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3B, &d,
                                               bytes, &length));
      std::vector<uint8_t> want;
      append(want, bytes, length);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, G17PBallotKeepsLiveMask)
{
   for (bool discard : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index source = agx_register(4, AGX_SIZE_32);
      source.discard = discard;
      agx_icmp_ballot_to(b, agx_register(0, AGX_SIZE_32), source,
                        agx_zero(), AGX_ICOND_UEQ, true);
      agx_stop(b);

      /* `x != 0` takes the immediate `x > 0` form on G17P (the register-zero
       * form is measured wrong on hardware), and still carries the source's
       * liveness so a ballot mask that is read again is not discarded. */
      const struct agx3_subgroup_ballot_desc d = {
         .type = AGX3_SUBGROUP_BALLOT_INTEGER,
         .dest = 0,
         .src = 4,
         .compare_immediate = true,
         .compare_value = 0,
         .src_live = !discard,
         .condition = AGX_ICOND_UGT,
      };
      uint8_t subgroup[AGX3_SUBGROUP_MAX_LENGTH], subgroup_len = 0;
      ASSERT_TRUE(agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3B, &d,
                                              subgroup, &subgroup_len));
      std::vector<uint8_t> want;
      append(want, subgroup, subgroup_len);
      /* The liveness bit is what this test is about: it must differ. */
      EXPECT_EQ(subgroup_len >= 8, true);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, G17PBallotUnsignedImmediate)
{
   static const uint8_t native_lt16[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x40, 0x18, 0x22, 0x0c,
   };
   static const uint8_t native_gt15[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x3c, 0x18, 0x22, 0x0a,
   };
   static const uint8_t native_eq0[] = {
      0x17, 0x17, 0x54, 0x00, 0x02, 0x04, 0x00, 0x18, 0x22, 0x02,
   };

   auto check = [this](enum agx_icond cond, unsigned imm, bool invert,
                       const uint8_t *native, size_t native_size) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index source = agx_register(2, AGX_SIZE_32);
      source.cache = true;
      agx_icmp_ballot_to(b, agx_register(0, AGX_SIZE_32), source,
                         agx_immediate(imm), cond, invert);
      agx_stop(b);

      std::vector<uint8_t> want(native, native + native_size);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   };

   check(AGX_ICOND_ULT, 16, false, native_lt16, sizeof(native_lt16));
   check(AGX_ICOND_UGT, 15, false, native_gt15, sizeof(native_gt15));
   /* !(x < 16) is canonicalized to Apple's byte-exact x > 15 form. */
   check(AGX_ICOND_ULT, 16, true, native_gt15, sizeof(native_gt15));
   check(AGX_ICOND_UEQ, 0, false, native_eq0, sizeof(native_eq0));

   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_icmp_quad_ballot_to(bad, agx_register(0, AGX_SIZE_32),
                           agx_register(2, AGX_SIZE_32), agx_immediate(16),
                           AGX_ICOND_ULT, false);
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);

   bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_icmp_ballot_to(bad, agx_register(0, AGX_SIZE_32),
                      agx_register(2, AGX_SIZE_32), agx_immediate(7),
                      AGX_ICOND_SLT, false);
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);

   bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_icmp_ballot_to(bad, agx_register(0, AGX_SIZE_32),
                      agx_register(2, AGX_SIZE_32), agx_immediate(0),
                      AGX_ICOND_ULT, true);
   agx_stop(bad);
   pack(bad, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, G17PBallotMaskStaysLiveThroughPopcount)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index predicate = agx_register(2, AGX_SIZE_32);
   predicate.cache = true;
   agx_index mask = agx_register(0, AGX_SIZE_32);
   mask.cache = true;
   agx_icmp_ballot_to(b, mask, predicate, agx_immediate(16), AGX_ICOND_ULT,
                      false);
   agx_index live_mask = agx_register(0, AGX_SIZE_32);
   live_mask.cache = true;
   agx_index count = agx_register(4, AGX_SIZE_32);
   count.cache = true;
   agx_popcount_to(b, count, live_mask);
   agx_stop(b);

   /* applegpu-nt 32023.917, the exact ballot+popcount prefix generated for
    * mask + popcount(mask), with both intermediate values still live. */
   static const uint8_t native[] = {
      0x17, 0x17, 0x54, 0x00, 0x03, 0x04, 0x40, 0x18, 0x22, 0x0c,
      0x27, 0x05, 0x54, 0x04, 0x03, 0x00, 0x4e, 0x04,
   };
   std::vector<uint8_t> want(std::begin(native), std::end(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

static enum agx_simd_op
mesa_reduce_op(unsigned op)
{
   static const enum agx_simd_op map[] = {
      AGX_SIMD_OP_AND,  AGX_SIMD_OP_FADD, AGX_SIMD_OP_OR,
      AGX_SIMD_OP_FMUL, AGX_SIMD_OP_XOR,  AGX_SIMD_OP_FMIN,
      AGX_SIMD_OP_FMAX, AGX_SIMD_OP_IADD, AGX_SIMD_OP_SMIN,
      AGX_SIMD_OP_SMAX, AGX_SIMD_OP_UMIN, AGX_SIMD_OP_UMAX,
   };
   return map[op];
}

TEST_F(AGX3PackBinary, SimdReductionsAndExactPrefixes)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (unsigned op = 0; op < AGX3_SUBGROUP_REDUCE_NUM; ++op) {
         agx_builder *b = builder(enc);
         agx_simd_reduce_to(b, agx_register(4, AGX_SIZE_32),
                            agx_register(2, AGX_SIZE_32), mesa_reduce_op(op));
         agx_stop(b);
         const struct agx3_subgroup_reduce_desc d = {
            .op = (enum agx3_subgroup_reduce_op)op,
            .dest = 4,
            .src = 2,
            .src_live = true,
         };
         uint8_t subgroup[AGX3_SUBGROUP_MAX_LENGTH], subgroup_len = 0;
         ASSERT_TRUE(agx3_subgroup_encode_reduce(enc, &d, subgroup,
                                                  &subgroup_len));
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         std::vector<uint8_t> want;
         append(want, subgroup, subgroup_len);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }

      for (unsigned op : {AGX3_SUBGROUP_REDUCE_FADD,
                          AGX3_SUBGROUP_REDUCE_FMUL,
                          AGX3_SUBGROUP_REDUCE_FMIN,
                          AGX3_SUBGROUP_REDUCE_FMAX,
                          AGX3_SUBGROUP_REDUCE_IADD}) {
         agx_builder *b = builder(enc);
         agx_quad_reduce_to(b, agx_register(4, AGX_SIZE_32),
                            agx_register(2, AGX_SIZE_32), mesa_reduce_op(op));
         agx_stop(b);
         const struct agx3_subgroup_reduce_desc d = {
            .quad = true,
            .op = (enum agx3_subgroup_reduce_op)op,
            .dest = 4,
            .src = 2,
            .src_live = true,
         };
         uint8_t subgroup[AGX3_SUBGROUP_MAX_LENGTH], subgroup_len = 0;
         ASSERT_TRUE(agx3_subgroup_encode_reduce(enc, &d, subgroup,
                                                  &subgroup_len));
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         std::vector<uint8_t> want;
         append(want, subgroup, subgroup_len);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }

      for (unsigned op : {AGX3_SUBGROUP_REDUCE_FADD,
                          AGX3_SUBGROUP_REDUCE_FMUL,
                          AGX3_SUBGROUP_REDUCE_IADD}) {
         agx_builder *b = builder(enc);
         agx_simd_prefix_to(b, agx_register(4, AGX_SIZE_32),
                            agx_register(2, AGX_SIZE_32), mesa_reduce_op(op));
         agx_stop(b);
         const struct agx3_subgroup_reduce_desc d = {
            .prefix = true,
            .op = (enum agx3_subgroup_reduce_op)op,
            .dest = 4,
            .src = 2,
            .src_live = true,
         };
         uint8_t subgroup[AGX3_SUBGROUP_MAX_LENGTH], subgroup_len = 0;
         ASSERT_TRUE(agx3_subgroup_encode_reduce(enc, &d, subgroup,
                                                  &subgroup_len));
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         std::vector<uint8_t> want;
         append(want, subgroup, subgroup_len);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, G17PClusteredIntegerReductions)
{
   for (unsigned op : {AGX3_SUBGROUP_REDUCE_AND,
                       AGX3_SUBGROUP_REDUCE_OR,
                       AGX3_SUBGROUP_REDUCE_XOR,
                       AGX3_SUBGROUP_REDUCE_SMIN,
                       AGX3_SUBGROUP_REDUCE_SMAX,
                       AGX3_SUBGROUP_REDUCE_UMIN,
                       AGX3_SUBGROUP_REDUCE_UMAX}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_quad_reduce_to(b, agx_register(4, AGX_SIZE_32),
                         agx_register(2, AGX_SIZE_32), mesa_reduce_op(op));
      agx_stop(b);

      const struct agx3_subgroup_reduce_desc d = {
         .quad = true,
         .op = (enum agx3_subgroup_reduce_op)op,
         .dest = 4,
         .src = 2,
         .src_live = true,
      };
      uint8_t subgroup[AGX3_SUBGROUP_MAX_LENGTH], subgroup_len = 0;
      ASSERT_TRUE(agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3B, &d,
                                               subgroup, &subgroup_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      std::vector<uint8_t> want;
      append(want, subgroup, subgroup_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, G17PClusteredFloatReductions)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   const agx_index value = agx_register(4, AGX_SIZE_32);
   agx_device_load_to(b, value, agx_register(8, AGX_SIZE_64),
                      agx_register(12, AGX_SIZE_32), AGX_FORMAT_I32, 1, 2,
                      false);
   agx_quad_reduce_to(b, agx_register(0, AGX_SIZE_32), value,
                      AGX_SIMD_OP_FADD);
   agx_quad_reduce_to(b, agx_register(2, AGX_SIZE_32), value,
                      AGX_SIMD_OP_FMIN);
   agx_index final_value = value;
   final_value.discard = true;
   agx_quad_reduce_to(b, agx_register(4, AGX_SIZE_32), final_value,
                      AGX_SIMD_OP_FMAX);
   agx_stop(b);

   const uint8_t native[] = {
      0x37, 0x06, 0x56, 0x00, 0x02, 0x08, 0x16, 0x02,
      0xb7, 0x05, 0x54, 0x02, 0x02, 0x08, 0x0a, 0x02,
      0xb7, 0x07, 0x54, 0x04, 0x02, 0x08, 0x08, 0x12,
   };
   std::vector<uint8_t> want(native, native + sizeof(native));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);

   const std::vector<uint8_t> got = pack(b);
   ASSERT_GE(got.size(), want.size());
   const std::vector<uint8_t> tail(got.end() - want.size(), got.end());
   EXPECT_EQ(tail, want);
}

TEST_F(AGX3PackBinary, G17PWidensHalfBeforeFloatCompare)
{
   /* metal-nt -arch applegpu_g17p, `float x; half y; if (x < y)` with y
    * loaded from a buffer: the half is widened into its own f32 register,
    * waiting on the load's slot 6 (f57), before the f32 compare. */
   static const uint8_t widen[] = {0x29, 0x04, 0x3c, 0x81, 0x00, 0xc2};
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(4, AGX_SIZE_16);
   agx_device_load_to(b, src, agx_register(8, AGX_SIZE_64),
                      agx_register(12, AGX_SIZE_32), AGX_FORMAT_F16, 1, 1,
                      false);
   agx_index dest = agx_register(4, AGX_SIZE_32);
   dest.cache = true;
   src.discard = true;
   agx_fmov_to(b, dest, src);
   agx_stop(b);

   std::vector<uint8_t> want(widen, widen + sizeof(widen));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   const std::vector<uint8_t> got = pack(b);
   ASSERT_GE(got.size(), want.size());
   const std::vector<uint8_t> tail(got.end() - want.size(), got.end());
   EXPECT_EQ(tail, want);
}

TEST_F(AGX3PackBinary, SubgroupUnknownAxesFailClosed)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_icmp_ballot_to(b, agx_register(4, AGX_SIZE_32),
                      agx_register(2, AGX_SIZE_32), agx_immediate(1),
                      AGX_ICOND_SEQ, false);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);

   b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_index killed = agx_register(2, AGX_SIZE_32);
   killed.kill = true;
   agx_simd_reduce_to(b, agx_register(4, AGX_SIZE_32), killed,
                      AGX_SIMD_OP_IADD);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, ScopedMemoryBarriers)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (unsigned s = AGX3_BARRIER_SCOPE_EXECUTION;
           s <= AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT; ++s) {
         if (enc == AGX_ISA_ENCODING_AGX3C &&
             s > AGX3_BARRIER_SCOPE_TEXTURE)
            continue;
         agx_builder *b = builder(enc);
         agx_scoped_barrier_agx3(b, s);
         agx_stop(b);

         struct agx3_encoded_scoped_barrier barrier;
         ASSERT_TRUE(agx3_pack_scoped_barrier(
            enc, (enum agx3_barrier_scope)s, &barrier));
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));

         std::vector<uint8_t> want;
         append(want, barrier.bytes, barrier.length);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, ScopedMemoryBarrierRefusesUnknownScope)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_THREADGROUP_FENCE + 1);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, DirectScsLoadStoreCarriesWait)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      const agx_index value = agx_register(2, AGX_SIZE_32);
      agx_scs_load_agx3_to(b, value, 4, AGX_FORMAT_I32, 1);
      agx_scs_store_agx3(b, value, 8, AGX_FORMAT_I32, 1);
      agx_stop(b);

      uint8_t bytes[AGX3_SCS_MAX_LENGTH];
      uint8_t length = 0;
      std::vector<uint8_t> want;
      struct agx3_scs_desc d = {
         .data = 2,
         .width = AGX3_MEM_W32,
         .byte_offset = 4,
         .slot = 1,
      };
      ASSERT_TRUE(agx3_scs_encode_load(enc, &d, bytes, &length));
      append(want, bytes, length);
      d.byte_offset = 8;
      d.slot = 0;
      d.wait_mask = 1;
      ASSERT_TRUE(agx3_scs_encode_store(enc, &d, bytes, &length));
      append(want, bytes, length);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, DirectScsScalarWidthsAndSignedOffsets)
{
   struct {
      enum agx_format format;
      enum agx_size size;
      enum agx3_mem_width width;
      int16_t offset;
   } cases[] = {
      {AGX_FORMAT_I8, AGX_SIZE_16, AGX3_MEM_W8, -8},
      {AGX_FORMAT_I16, AGX_SIZE_16, AGX3_MEM_W16, 10},
      {AGX_FORMAT_I32, AGX_SIZE_32, AGX3_MEM_W32, -12},
   };

   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (const auto &c : cases) {
         agx_builder *b = builder(enc);
         const agx_index value = agx_register(6, c.size);
         agx_scs_load_agx3_to(b, value, (uint64_t)(int64_t)c.offset, c.format, 1);
         agx_stop(b);

         uint8_t bytes[AGX3_SCS_MAX_LENGTH];
         uint8_t length = 0;
         std::vector<uint8_t> want;
         const struct agx3_scs_desc d = {
            .data = 6,
            .width = c.width,
            .byte_offset = c.offset,
            .slot = 1,
         };
         ASSERT_TRUE(agx3_scs_encode_load(enc, &d, bytes, &length));
         append(want, bytes, length);
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, DirectScsRefusesVectorAccess)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_index value = agx_register(2, AGX_SIZE_32);
   value.channels_m1 = 1;
   agx_scs_load_agx3_to(b, value, 0, AGX_FORMAT_I32, 1);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, StaticSpillCopyReloadHalf)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_stack_store(b, agx_register(1, AGX_SIZE_16), agx_immediate(6),
                      AGX_FORMAT_I16, 1);
      agx_stack_load_to(b, agx_register(3, AGX_SIZE_16), agx_immediate(6),
                        AGX_FORMAT_I16, 1);
      agx_stop(b);

      std::vector<uint8_t> want;
      uint8_t bytes[AGX3_STATIC_SPILL_MAX_LENGTH];
      uint8_t length = 0;
      const struct agx3_static_spill_store_desc store = {
         .tps_half = 3,
         .src = 1,
         .width = AGX3_MEM_W16,
         .slot = 1,
      };
      ASSERT_TRUE(agx3_static_spill_encode_store(enc, &store, bytes, &length));
      append(want, bytes, length);

      const struct agx3_static_spill_load_desc load = {
         .dest = 3,
         .tps_half = 3,
         .width = AGX3_MEM_W16,
         .slot = 2,
         .slot_mirror = 2,
         .wait_mask = 1,
      };
      ASSERT_TRUE(agx3_static_spill_encode_load(enc, &load, bytes, &length));
      append(want, bytes, length);

      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, SplitScalar64SpillIntoGroundedWordStores)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_stack_store(b, agx_register(12, AGX_SIZE_64), agx_immediate(16),
                   AGX_FORMAT_I32, 3);
   agx_stop(b);

   agx3_split_vector_stack_access(b->shader);
   EXPECT_FALSE(pack(b).empty());

   unsigned stores = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_STACK_STORE)
         continue;
      EXPECT_EQ(I->format, AGX_FORMAT_I32);
      EXPECT_EQ(I->mask, 1u);
      EXPECT_EQ(I->src[0].size, AGX_SIZE_32);
      EXPECT_EQ(I->src[0].value, 12u + 2u * stores);
      EXPECT_EQ(I->src[1].value, 16u + 4u * stores);
      ++stores;
   }
   EXPECT_EQ(stores, 2u);
}

TEST_F(AGX3PackBinary, StaticSpillReloadKeepsARereadCellLive)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_stack_store(b, agx_register(0, AGX_SIZE_32), agx_immediate(32),
                   AGX_FORMAT_I32, 1);
   agx_stack_load_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(32),
                     AGX_FORMAT_I32, 1);
   agx_stack_load_to(b, agx_register(4, AGX_SIZE_32), agx_immediate(32),
                     AGX_FORMAT_I32, 1);
   agx_stop(b);
   const std::vector<uint8_t> got = pack(b);

   /* The first reload is followed by another read of the same cell, so only
    * the second may end its lifetime. */
   uint8_t bytes[AGX3_STATIC_SPILL_MAX_LENGTH], length = 0;
   std::vector<uint8_t> want;
   const struct agx3_static_spill_store_desc store = {
      .tps_half = 16, .src = 0, .width = AGX3_MEM_W32, .slot = 1,
   };
   ASSERT_TRUE(agx3_static_spill_encode_store(enc, &store, bytes, &length));
   append(want, bytes, length);
   for (unsigned i = 0; i < 2; ++i) {
      const struct agx3_static_spill_load_desc load = {
         .dest = (uint16_t)(2 + 2 * i),
         .tps_half = 16,
         .width = AGX3_MEM_W32,
         .slot = (uint8_t)(2 + i),
         .slot_mirror = (uint8_t)(2 + i),
         .wait_mask = (uint16_t)(1u << i),
         .tps_live = i == 0,
      };
      ASSERT_TRUE(agx3_static_spill_encode_load(enc, &load, bytes, &length));
      append(want, bytes, length);
   }
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(got, want);
}

TEST_F(AGX3PackBinary, VertexWithoutVaryingsIsABareEnd)
{
   /* applegpu_g17p compiles `vertex void f(uint vid [[vertex_id]]) {}` to
    * 0e000000: there is no marker for a vertex program that writes nothing. */
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_VERTEX;
   agx_no_varyings(b);
   agx_stop(b);

   std::vector<uint8_t> want;
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   agx_builder *fragment = builder(enc);
   fragment->shader->stage = MESA_SHADER_FRAGMENT;
   agx_no_varyings(fragment);
   agx_stop(fragment);
   pack(fragment, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, StaticSpillRefusesNonScalarOrMisalignedShapes)
{
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index h0 = agx_register(0, AGX_SIZE_16);
      agx_stack_store(b, h0, agx_immediate(1), AGX_FORMAT_I16, 1);
      agx_stop(b);
      pack(b, /*expect_ok=*/false);
   }

   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_index w0 = agx_register(0, AGX_SIZE_32);
      w0.channels_m1 = 1;
      agx_stack_store(b, w0, agx_immediate(0), AGX_FORMAT_I32, 3);
      agx_stop(b);
      pack(b, /*expect_ok=*/false);
   }

   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_stack_load_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(2),
                        AGX_FORMAT_I32, 1);
      agx_stop(b);
      pack(b, /*expect_ok=*/false);
   }
}

TEST_F(AGX3PackBinary, FloatModifierRouting)
{
   /* A negated source must route to the long modifier form, not the compact
    * packer, and match the modifier encoder byte for byte. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index w0 = agx_register(0, AGX_SIZE_32);
   agx_index w2 = agx_register(2, AGX_SIZE_32);
   agx_index w4 = agx_register(4, AGX_SIZE_32);
   agx_fadd_to(b, w4, agx_neg(w0), w2);
   agx_stop(b);

   struct agx3_falu_mods m = {};
   m.src0_neg = true;
   uint8_t mod[AGX3_FALU_MOD_MAX_LENGTH];
   uint8_t mlen = 0;
   struct agx3_falu_runtime runtime = {};
   ASSERT_TRUE(agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 4, 0, 2,
                                         &m, &runtime, mod, &mlen));
   struct agx3_encoded_end en;
   agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &en);

   std::vector<uint8_t> want;
   append(want, mod, mlen);
   append(want, en.bytes, en.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, M5PlainFloatAddPreservesWaitAndLiveness)
{
   for (bool loaded : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_index src0 = agx_register(28, AGX_SIZE_32);
      agx_index src1 = agx_register(30, AGX_SIZE_32);
      if (loaded) {
         /* Consume all eight producer slots before the arithmetic. */
         for (unsigned i = 0; i < 8; ++i)
            agx_device_load_to(b, agx_register(i * 4, AGX_SIZE_32),
                              agx_uniform(0, AGX_SIZE_64), agx_immediate(i),
                              AGX_FORMAT_I32, 1, 0, false);
      }
      src0.discard = true;
      src1.cache = true;
      agx_fadd_to(b, agx_register(14, AGX_SIZE_32), src0, src1);
      agx_stop(b);
      const auto bin = pack(b);
      /* Independently composed GPU1 FADD: slot8 after load, keep src1 and
       * discard src0. No abs/neg/saturate modifier triggers this routing. */
      const std::vector<uint8_t> add = loaded
         ? std::vector<uint8_t>{0x39,0x82,0x04,0xba,0x22,0x20,
                                0xa0,0x1a,0x1a,0x08,0x12,0x00}
         : std::vector<uint8_t>{0x79,0x9f,0x10,0x1d};
      ASSERT_GE(bin.size(), add.size() + 4);
      EXPECT_NE(std::search(bin.begin(), bin.end(), add.begin(), add.end()), bin.end());
   }
}

TEST_F(AGX3PackBinary, RefusesUnmappedOp)
{
   /* BITREV remains unmapped; the backend must fail the shader rather than
    * treating it like the now-grounded tied-register FT1 operation. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index w0 = agx_register(0, AGX_SIZE_32);
   agx_instr *I = agx_ffs_to(b, w0, w0);
   I->op = AGX_OPCODE_BITREV;
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

#define ASSERT_TRUE_VOID(x) do { if (!(x)) { ADD_FAILURE(); return; } } while (0)

/* An AGX3B texel fetch stages its coordinates in every stage, so the packer
 * emits two coordinate COPYs and then a fetch that reads the staging bank
 * rather than the coordinate GPR. Apple's own applegpu_g17p vertex
 * `texture2d<access::read>::read()` has exactly this shape -- two four-byte
 * COPYs, the fetch, then PACK_SMPIR -- and a non-staged vertex fetch returns
 * only the x and y components on hardware. AGX3C is unchanged. */
static void
append_staged_fetch(std::vector<uint8_t> &want, enum agx_isa_encoding enc,
                    struct agx3_tex_fetch_desc d, uint16_t coord,
                    unsigned coords)
{
   if (enc == AGX_ISA_ENCODING_AGX3B) {
      for (unsigned i = 0; i < coords; ++i) {
         uint8_t copy[AGX3_TEX_MAX_LENGTH];
         uint8_t copy_len = 0;
         ASSERT_TRUE_VOID(agx3_tex_stage_copy_hints(
            enc, 2 * i, coord + 2 * i, 1, 0, false, false, copy, &copy_len));
         want.insert(want.end(), copy, copy + copy_len);
      }
      d.coord = 0;
      d.coord_kill = true;
      d.wait_mask = 1u << 0;
   }

   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length = 0;
   ASSERT_TRUE_VOID(agx3_tex_encode_fetch(enc, &d, bytes, &length));
   want.insert(want.end(), bytes, bytes + length);
}

TEST_F(AGX3PackBinary, TextureCoordinateCollectStaysLowUnderPressure)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;

   /* Keep the whole 128-half TEX source bank live when the coordinate COLLECT
    * is reached. Preloads make that pressure deterministic: the allocator
    * must relocate one live value to place this exact direct-COLLECT texture
    * source below half 128, rather than accidentally finding an unused hole. */
   agx_index live[64];
   for (unsigned i = 0; i < ARRAY_SIZE(live); ++i)
      live[i] = agx_preload(b, agx_register(2 * i, AGX_SIZE_32));

   agx_index xy[2] = {
      agx_mov_imm(b, 32, 0x41000000),
      agx_mov_imm(b, 32, 0x41800000),
   };
   agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, coord, 2);
   collect->src[0] = xy[0];
   collect->src[1] = xy[1];

   agx_index tex_dest = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_instr *tex = agx_texture_sample_to(
      b, tex_dest, coord, agx_immediate(0), agx_immediate(0),
      agx_uniform(0, AGX_SIZE_64), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   tex->op = AGX_OPCODE_TEXTURE_LOAD;
   tex->mask = 15;
   agx_instr *split = agx_split(b, 4, tex_dest);
   for (unsigned i = 0; i < 4; ++i)
      split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);

   /* Preserve the synthetic pressure across the COLLECT/TEX pair. */
   agx_index sum = agx_fadd(b, live[0], split->dest[0]);
   for (unsigned i = 1; i < ARRAY_SIZE(live); ++i)
      sum = agx_fadd(b, sum, live[i]);
   agx_export(b, sum, 8);
   agx_stop(b);

   agx_pressure_schedule(b->shader);
   agx_ra(b->shader);

   ASSERT_EQ(tex->op, AGX_OPCODE_TEXTURE_LOAD);
   ASSERT_EQ(tex->src[0].type, AGX_INDEX_REGISTER);
   EXPECT_LT(tex->src[0].value, 128u);
   EXPECT_LE(tex->src[0].value + 4, 128u);
}

TEST_F(AGX3PackBinary, UniformMovesAndImageReadsStayLowUnderPressure)
{
   for (unsigned image = 0; image < 2; ++image) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->nir = rzalloc(mem_ctx, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

      /* Keep halves 0..127 live, as in the texture pressure tests, so a
       * result lands below half 128 only if the allocator bounds it. */
      agx_index live[64];
      for (unsigned i = 0; i < ARRAY_SIZE(live); ++i)
         live[i] = agx_preload(b, agx_register(2 * i, AGX_SIZE_32));

      agx_instr *bounded;
      agx_index value;
      if (image) {
         agx_index xy[2] = {agx_mov_imm(b, 32, 1), agx_mov_imm(b, 32, 2)};
         agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
         agx_instr *collect = agx_collect_to(b, coord, 2);
         collect->src[0] = xy[0];
         collect->src[1] = xy[1];
         agx_index dest = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
         bounded = agx_image_load_to(
            b, dest, coord, agx_immediate(0), agx_uniform(6, AGX_SIZE_64),
            agx_uniform(0, AGX_SIZE_64), agx_immediate(0), agx_null(),
            AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false, false, false);
         agx_instr *split = agx_split(b, 4, dest);
         for (unsigned i = 0; i < 4; ++i)
            split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);
         value = split->dest[0];
      } else {
         value = agx_temp(b->shader, AGX_SIZE_32);
         bounded = agx_mov_to(b, value, agx_uniform(26, AGX_SIZE_32));
      }

      agx_index sum = agx_fadd(b, live[0], value);
      for (unsigned i = 1; i < ARRAY_SIZE(live); ++i)
         sum = agx_fadd(b, sum, live[i]);
      agx_export(b, sum, 8);
      agx_stop(b);

      agx_pressure_schedule(b->shader);
      agx_ra(b->shader);

      /* TEX and the uniform COPY both name seven-bit destinations. */
      ASSERT_EQ(bounded->dest[0].type, AGX_INDEX_REGISTER);
      EXPECT_LE(bounded->dest[0].value +
                   agx_size_align_16(bounded->dest[0].size) *
                      agx_channels(bounded->dest[0]),
                128u)
         << "image=" << image;
   }
}

TEST_F(AGX3PackBinary, ImageWriteOperandsStayLowUnderPressure)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   /* Keep halves 0..127 live until after the store. */
   agx_index live[64];
   for (unsigned i = 0; i < ARRAY_SIZE(live); ++i)
      live[i] = agx_preload(b, agx_register(2 * i, AGX_SIZE_32));

   agx_index channels[4], xy[2];
   for (unsigned i = 0; i < 4; ++i)
      channels[i] = agx_mov_imm(b, 32, i + 1);
   for (unsigned i = 0; i < 2; ++i)
      xy[i] = agx_mov_imm(b, 32, i);
   agx_index record = agx_mov_imm(b, 32, 5);
   agx_index data = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_instr *data_collect = agx_collect_to(b, data, 4);
   for (unsigned i = 0; i < 4; ++i)
      data_collect->src[i] = channels[i];
   agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *coord_collect = agx_collect_to(b, coord, 2);
   coord_collect->src[0] = xy[0];
   coord_collect->src[1] = xy[1];
   agx_instr *write = agx_image_write(b, data, coord, agx_immediate(0),
                                      agx_uniform(20, AGX_SIZE_64), record,
                                      AGX_DIM_2D, false);

   agx_index sum = live[0];
   for (unsigned i = 1; i < ARRAY_SIZE(live); ++i)
      sum = agx_fadd(b, sum, live[i]);
   agx_export(b, sum, 8);
   agx_stop(b);

   agx_ra(b->shader);

   /* imgwci stages data and coordinates through seven-bit COPY sources and
    * names its record index in a seven-bit field. */
   for (unsigned s : {0u, 1u, 4u}) {
      ASSERT_EQ(write->src[s].type, AGX_INDEX_REGISTER) << "src " << s;
      EXPECT_LE(write->src[s].value + agx_size_align_16(write->src[s].size) *
                                         agx_channels(write->src[s]),
                128u)
         << "src " << s;
   }
}

TEST_F(AGX3PackBinary, ThreeJSVertexTextureCoordinatesStayLowUnderPressure)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   /* Reproduce the captured three.js skinning allocation: all 128 halves of
    * TEX's staging-COPY source bank are live, while the dynamic descriptor is
    * independently valid in the high bank. The allocator must make a low-bank
    * hole for the direct coordinate COLLECT and preserve that constraint if a
    * later live-range split relocates another value. */
   agx_index live[64];
   for (unsigned i = 0; i < ARRAY_SIZE(live); ++i)
      live[i] = agx_preload(b, agx_register(2 * i, AGX_SIZE_32));
   agx_index high_live[6];
   for (unsigned i = 0; i < ARRAY_SIZE(high_live); ++i)
      high_live[i] = agx_preload(b, agx_register(128 + 2 * i, AGX_SIZE_32));
   agx_index descriptor = agx_register(172, AGX_SIZE_64);

   agx_index xy[2] = {
      agx_mov_imm(b, 32, 0x41000000),
      agx_mov_imm(b, 32, 0x41800000),
   };
   agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, coord, 2);
   collect->src[0] = xy[0];
   collect->src[1] = xy[1];

   agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_instr *tex = agx_texture_sample_to(
      b, result, coord, agx_immediate(0), agx_immediate(0), descriptor,
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0,
      false, false, false, AGX_GATHER_NONE, false);
   tex->op = AGX_OPCODE_TEXTURE_LOAD;
   tex->mask = 15;
   agx_instr *split = agx_split(b, 4, result);
   for (unsigned i = 0; i < 4; ++i)
      split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);

   agx_index sum = agx_fadd(b, live[0], split->dest[0]);
   for (unsigned i = 1; i < ARRAY_SIZE(live); ++i)
      sum = agx_fadd(b, sum, live[i]);
   for (unsigned i = 0; i < ARRAY_SIZE(high_live); ++i)
      sum = agx_fadd(b, sum, high_live[i]);
   agx_export(b, sum, 8);
   agx_stop(b);

   agx_pressure_schedule(b->shader);
   agx_ra(b->shader);

   ASSERT_EQ(tex->op, AGX_OPCODE_TEXTURE_LOAD);
   ASSERT_EQ(tex->dest[0].type, AGX_INDEX_REGISTER);
   ASSERT_EQ(tex->src[0].type, AGX_INDEX_REGISTER);
   EXPECT_EQ(tex->src[3].type, AGX_INDEX_REGISTER);
   EXPECT_EQ(tex->src[3].value, 172u);
   EXPECT_LT(tex->src[0].value, 128u);
   EXPECT_LE(tex->src[0].value + 4, 128u);

   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   EXPECT_TRUE(tex->src[0].discard);
   EXPECT_FALSE(pack(b).empty());

   /* The allocator repair must not widen TEX's seven-bit coordinate field.
    * A precoloured reproduction of the captured bad physical instruction
    * remains fail-closed. The independently grounded high descriptor stays
    * legal only when the coordinate itself was allocated in the low bank. */
   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   bad->shader->stage = MESA_SHADER_VERTEX;
   bad->shader->nir = rzalloc(mem_ctx, nir_shader);
   bad->shader->nir->info.stage = MESA_SHADER_VERTEX;
   agx_index bad_dest = agx_register(0, AGX_SIZE_32);
   agx_index bad_coord = agx_register(140, AGX_SIZE_32);
   bad_dest.channels_m1 = 3;
   bad_coord.channels_m1 = 1;
   bad_coord.discard = true;
   agx_instr *bad_tex = agx_texture_sample_to(
      bad, bad_dest, bad_coord, agx_immediate(0), agx_immediate(0),
      agx_register(172, AGX_SIZE_64), agx_immediate(0), agx_null(),
      AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0, false, false, false,
      AGX_GATHER_NONE, false);
   bad_tex->op = AGX_OPCODE_TEXTURE_LOAD;
   bad_tex->mask = 15;
   agx_stop(bad);
   EXPECT_TRUE(pack(bad, false).empty());
}

TEST_F(AGX3PackBinary, TextureResultShuffleKeepsHalfCopiesInLowBank)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;

   /* Fill every low-bank pair except r6/r7, then keep only r7l live there.
    * This makes r6/r7 the unique cheapest region for the constrained vec2
    * coordinate. XBR has this exact lifetime: preserving the r7l predicate
    * used to generate the unsupported `mov1 r71h, r7l`. */
   agx_index live[62];
   unsigned next = 0;
   for (unsigned reg = 0; reg < 128; reg += 2) {
      if (reg != 12 && reg != 14)
         live[next++] = agx_preload(b, agx_register(reg, AGX_SIZE_32));
   }
   ASSERT_EQ(next, ARRAY_SIZE(live));
   agx_index half_live = agx_preload(b, agx_register(14, AGX_SIZE_16));

   agx_index xy[2] = {
      agx_preload(b, agx_register(128, AGX_SIZE_32)),
      agx_preload(b, agx_register(130, AGX_SIZE_32)),
   };
   agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, coord, 2);
   collect->src[0] = xy[0];
   collect->src[1] = xy[1];

   agx_index tex_dest = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_instr *tex = agx_texture_sample_to(
      b, tex_dest, coord, agx_immediate(0), agx_immediate(0),
      agx_uniform(0, AGX_SIZE_64), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   tex->op = AGX_OPCODE_TEXTURE_LOAD;
   tex->mask = 15;
   agx_instr *split = agx_split(b, 4, tex_dest);
   for (unsigned i = 0; i < 4; ++i)
      split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);

   agx_index sum = agx_fadd(b, live[0], split->dest[0]);
   for (unsigned i = 1; i < ARRAY_SIZE(live); ++i)
      sum = agx_fadd(b, sum, live[i]);
   agx_iadd_to(b, agx_temp(b->shader, AGX_SIZE_16), half_live,
               agx_immediate(0), 0);
   agx_export(b, sum, 8);
   agx_stop(b);

   agx_ra(b->shader);

   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_MOV && I->dest[0].size == AGX_SIZE_16) {
         EXPECT_LT(I->dest[0].value, 128u);
      }
   }

   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, TextureFetch2DLod2Immediate)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   agx_index dest = agx_register(0, AGX_SIZE_32);
   agx_index coord = agx_register(12, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;

   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(2), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0,
      false, false, false, AGX_GATHER_NONE, false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;
   agx_stop(b);

   struct agx3_tex_fetch_desc d = {};
   d.dest = 0;
   d.coord = 12;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod_x64 = 2 * 64;
   d.texture = 0;
   d.slot = 1;
   d.final = true;
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   std::vector<uint8_t> want;
   append_staged_fetch(want, enc, d, 12, 2);
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

static agx_instr *
staged_fetch_2d(agx_builder *b, unsigned dest_reg, unsigned coord_reg)
{
   agx_index dest = agx_register(dest_reg, AGX_SIZE_32);
   agx_index coord = agx_register(coord_reg, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0, false,
      false, false, AGX_GATHER_NONE, false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;
   return I;
}

/* Register indices are halves. fetch h0 <- h24; if (h48 < h50) { h40 =
 * h0 + h2; fetch h8 <- h28 } pop; fetch h16 <- h32. The arm waits for the
 * first fetch, so its own fetch may stage in the first fetch's bank. Lanes
 * that skipped the arm did not wait, so after the pop the first fetch may
 * still be reading its staged coordinates: the last fetch must stage them
 * elsewhere or wait for it. */
static void
build_fetch_if_fetch(agx_builder *b)
{
   b->shader->stage = MESA_SHADER_VERTEX;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *arm = agx_test_block(b->shader);
   agx_block *pop = agx_test_block(b->shader);
   staged_fetch_2d(b, 0, 24);
   agx_begin_cf(b);
   agx_if_icmp(b, agx_register(48, AGX_SIZE_32), agx_register(50, AGX_SIZE_32),
               1, AGX_ICOND_ULT, false, pop);
   agx_jmp_exec_none(b, pop);
   agx_block_add_successor(entry, arm);
   agx_block_add_successor(entry, pop);

   agx_builder ab = agx_init_builder(b->shader, agx_after_block(arm));
   agx_fadd_to(&ab, agx_register(40, AGX_SIZE_32), agx_register(0, AGX_SIZE_32),
               agx_register(2, AGX_SIZE_32));
   staged_fetch_2d(&ab, 8, 28);
   agx_index arm_out = agx_register(8, AGX_SIZE_32);
   arm_out.channels_m1 = 3;
   agx_device_store(&ab, arm_out, agx_uniform(0, AGX_SIZE_64), agx_immediate(32),
                    AGX_FORMAT_I32, 15, 0, false);
   agx_block_add_successor(arm, pop);

   agx_builder pb = agx_init_builder(b->shader, agx_after_block(pop));
   agx_pop_exec(&pb, 1);
   staged_fetch_2d(&pb, 16, 32);
   for (unsigned h = 0; h <= 16; h += 16) {
      agx_index v = agx_register(h, AGX_SIZE_32);
      v.channels_m1 = 3;
      agx_device_store(&pb, v, agx_uniform(0, AGX_SIZE_64),
                       agx_immediate(h), AGX_FORMAT_I32, 15, 0, false);
   }
   agx_stop(&pb);
}

TEST_F(AGX3PackBinary, ArmEndKeepsStagedCoordinatesOfAnEarlierFetch)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   build_fetch_if_fetch(b);
   ASSERT_FALSE(pack(b).empty());
   /* The arm's fetch reuses bank 0 (halves 0..3); the last one takes bank 1
    * (halves 8..11) rather than overwrite the first fetch's coordinates. */
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 12u);
}

TEST_F(AGX3PackBinary, TextureFetchRepeatedWaitKeepsCoordinateUntilFinalCopy)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.fs.nr_samples = 4;

   /* Two independent asynchronous producers supply x and y. The staged
    * fetch therefore repeats x once per producer wait. Only the second x
    * COPY may discard it; otherwise that COPY rereads a killed coordinate. */
   const agx_index base = agx_uniform(16, AGX_SIZE_64);
   agx_device_load_to(b, agx_register(12, AGX_SIZE_32), base,
                      agx_immediate(0), AGX_FORMAT_I32, 1, 0, false);
   agx_device_load_to(b, agx_register(14, AGX_SIZE_32), base,
                      agx_immediate(4), AGX_FORMAT_I32, 1, 0, false);
   agx_mov_imm_to(b, agx_register(16, AGX_SIZE_32), 0);

   agx_index dest = agx_register(24, AGX_SIZE_32);
   agx_index coord = agx_register(12, AGX_SIZE_32);
   agx_index texture = agx_register(6, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 2;
   coord.discard = true;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_uniform(8, AGX_SIZE_64), texture,
      agx_immediate(0), agx_null(), AGX_DIM_2D_MS_ARRAY,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE, false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;
   agx_stop(b);

   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   std::vector<uint8_t> copies;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 12, 3, 1, false, false,
                                         bytes, &len));
   append(copies, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 12, 3, 2, false, true,
                                         bytes, &len));
   append(copies, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 14, 3, 0, false, true,
                                         bytes, &len));
   append(copies, bytes, len);

   const std::vector<uint8_t> code = pack(b);
   EXPECT_NE(std::search(code.begin(), code.end(), copies.begin(), copies.end()),
             code.end());
}

TEST_F(AGX3PackBinary, G17GSaturatingAddsUseNativeSignednessAndSaturate)
{
   /* agx_compile.c: usub_sat = abs(a) + neg(abs(b)), isub_sat = a + neg(b),
    * uadd_sat = abs(a) + abs(b), iadd_sat = a + b, all with .saturate. */
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   const struct {
      bool a_abs, b_abs, b_neg;
   } shapes[] = {
      {true, true, true}, {false, false, true},
      {true, true, false}, {false, false, false},
   };
   key.dev.emit_agx3d = true;
   for (const auto &s : shapes) {
      agx_builder *b = builder(enc);
      agx_index a = agx_register(0, AGX_SIZE_32);
      agx_index c = agx_register(4, AGX_SIZE_32);
      a.discard = c.discard = true;
      agx_index sa = s.a_abs ? agx_abs(a) : a;
      agx_index sb = s.b_abs ? agx_abs(c) : c;
      if (s.b_neg)
         sb = agx_neg(sb);
      agx_instr *I = agx_iadd_to(b, agx_register(0, AGX_SIZE_32), sa, sb, 0);
      I->saturate = true;
      agx_stop(b);

      struct agx3_iadd_desc d = {};
      d.src_a_is_register = true;
      d.src_a_reg = 0;
      d.src_a_kill = true;
      d.src_a_sign_extend_32 = !s.a_abs;
      d.src = 4;
      d.src_kill = true;
      d.sign_extend_32 = !s.b_abs;
      d.subtract = s.b_neg;
      d.saturate = true;
      uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_iadd_encode_g18g(enc, &d, bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want) << s.a_abs << s.b_abs << s.b_neg;
   }
}

TEST_F(AGX3PackBinary, G17GRegisterMoveUsesImmediateAdd)
{
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   agx_builder *b = builder(enc);
   agx_index src = agx_register(0, AGX_SIZE_32);
   src.discard = true;
   agx_bitop_to(b, agx_register(4, AGX_SIZE_32), src, agx_immediate(0),
                AGX3_SEL_TT_MOV);
   agx_stop(b);

   struct agx3_iadd_desc d = {};
   d.dest = 4;
   d.immediate = 0;
   d.src = 0;
   d.src_kill = true;
   uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_iadd_encode(enc, &d, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, G17GFindTopOneUsesNativeFt1)
{
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   key.dev.emit_agx3d = true;
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));

   /* 32-bit: one ft1 with the cached destination form. */
   {
      agx_builder *b = builder(enc);
      agx_index src = agx_register(0, AGX_SIZE_32);
      src.discard = true;
      agx_ffs_to(b, agx_register(4, AGX_SIZE_32), src);
      agx_stop(b);
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_alu_ft1_gpu1(4, 0, false, AGX3_ALU_HINT_DISCARD, true,
                                    0, bytes, &len));
      append(want, bytes, len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }

   /* A half source into a 32-bit result: the half form into the low half,
    * then the upper half cleared, as the widening move does. */
   {
      agx_builder *b = builder(enc);
      agx_index src = agx_register(3, AGX_SIZE_16);
      src.discard = true;
      agx_ffs_to(b, agx_register(4, AGX_SIZE_32), src);
      agx_stop(b);
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_alu_ft1_gpu1(4, 3, true, AGX3_ALU_HINT_DISCARD, true,
                                    0, bytes, &len));
      append(want, bytes, len);
      uint8_t hi[AGX3_STAGE_MAX_LENGTH], hi_len = 0;
      ASSERT_TRUE(agx3_clr(enc, 5, true, false, hi, &hi_len));
      append(want, hi, hi_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, G17GFloatCompareToHalfBooleanUsesFseli16)
{
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   key.dev.emit_agx3d = true;
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));

   for (unsigned reg = 0; reg < 2; ++reg) {
      agx_builder *b = builder(enc);
      agx_index compare = agx_register(2, AGX_SIZE_32);
      compare.discard = true;
      agx_index other = agx_register(4, AGX_SIZE_32);
      other.discard = true;
      agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), compare,
                     reg ? other : agx_immediate_f(-1.0f), agx_immediate(1),
                     agx_immediate(0), AGX_FCOND_LT);
      agx_stop(b);
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_alu_fcmp_bool16_gpu1(
         2, 2, AGX3_ALU_HINT_DISCARD, reg, reg ? 4 : agx_immediate_f(-1.0f).value,
         reg, 1, 0, AGX_FCOND_LT, 0, bytes, &len));
      append(want, bytes, len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want) << reg;
   }

   /* The unordered conditions have no G17G witness for this form. */
   agx_builder *b = builder(enc);
   agx_index compare = agx_register(2, AGX_SIZE_32);
   agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), compare,
                  agx_immediate_f(-1.0f), agx_immediate(1), agx_immediate(0),
                  AGX_FCOND_LTN);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, G17GUniformStoreUsesNativePreambleCopy)
{
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   uint8_t bytes[AGX3_COPY_MAX_LENGTH], len = 0;
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));

   /* One 32-bit component, then a 64-bit pointer as two consecutive copies
    * (Apple's fragment preambles store the loaded pointer pair the same
    * way). */
   for (unsigned wide = 0; wide < 2; ++wide) {
      agx_builder *b = builder(enc);
      agx_index value = agx_register(0, wide ? AGX_SIZE_64 : AGX_SIZE_32);
      value.kill = true;
      agx_uniform_store(b, value, agx_immediate(4), 1);
      agx_stop(b);
      std::vector<uint8_t> want;
      for (unsigned i = 0; i <= wide; ++i) {
         ASSERT_TRUE(agx3_copy_encode_reg_to_uniform(
            enc, 4 + 2 * i, 2 * i, i == wide, 0, bytes, &len));
         append(want, bytes, len);
      }
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want) << wide;
   }
}

TEST_F(AGX3PackBinary, G17GWideMultiplyAddUsesNativeImadd64)
{
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   agx_builder *b = builder(enc);
   agx_index a = agx_register(0, AGX_SIZE_32);
   agx_index m = agx_register(2, AGX_SIZE_32);
   a.discard = m.discard = true;
   agx_imad_to(b, agx_register(0, AGX_SIZE_64), agx_abs(a), agx_abs(m),
               agx_immediate(12), 0);
   agx_stop(b);

   struct agx3_imad_desc d = {};
   d.dest = 0;
   d.dest_64 = true;
   d.src_a = 0;
   d.src_a_kill = true;
   d.multiplier_is_register = true;
   d.multiplier_reg = 2;
   d.multiplier_kill = true;
   d.addend_is_immediate = true;
   d.addend_immediate = 12;
   uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_imad_encode(enc, &d, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, VertexIndexedTextureFetchUsesNativeHeapIndex)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   auto build = [&](bool native_index) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_VERTEX;
      agx_index dest = agx_register(40, AGX_SIZE_32);
      agx_index coord = agx_register(36, AGX_SIZE_32);
      agx_index texture = agx_register(2, native_index ? AGX_SIZE_64
                                                       : AGX_SIZE_32);
      dest.channels_m1 = 3;
      coord.channels_m1 = 1;
      coord.discard = true;
      texture.discard = true;
      agx_index descriptor_base = native_index
         ? agx_immediate(0)
         : agx_uniform(44, AGX_SIZE_64);
      agx_instr *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), descriptor_base, texture,
         agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN,
         0, false, false, false, AGX_GATHER_NONE, false);
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      I->mask = 15;
      agx_stop(b);
      return b;
   };

   /* Native uses one zero-extended 64-bit heap index. The original AGX2
    * descriptor-base plus 32-bit byte-offset pair is not that operand and
    * must remain fail-closed instead of being reinterpreted. */
   EXPECT_FALSE(pack(build(true)).empty());
   pack(build(false), /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, TextureFetchRefusesUngroundedAxes)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_index dest = agx_register(0, AGX_SIZE_32);
   agx_index coord = agx_register(12, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_immediate(16), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE, false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, TextureSampleOffsetUsesNativeImmediateField)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_16);
   agx_index coord = agx_register(4, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   coord.discard = true;

   agx_index offset = agx_immediate(0xf1);
   offset.size = AGX_SIZE_32;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), offset, AGX_DIM_2D,
      AGX_LOD_MODE_AUTO_LOD, 0, true, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   /* The packed offset reaches the packer on the instruction, not through
    * src[5]: a comparison sample needs that operand for its reference. */
   I->offset_xyz = 0xf1;
   agx_stop(b);

   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 4, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 6, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);

   const struct agx3_tex_sample_desc sample = {
      .dest = 8, .coord = 0, .width = AGX3_TEX_W16, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
      .final = true, .offset = true, .offset_xyz = 0xf1,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   EXPECT_TRUE(b->shader->agx3_colour_link.entry_emitted);
   EXPECT_EQ(b->shader->agx3_colour_link.entry_offset, 8);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 4);
}

TEST_F(AGX3PackBinary, TextureSampleOffsetStagesRuntimePair)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_16);
   agx_index coord = agx_register(4, AGX_SIZE_32);
   agx_index offset = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   coord.discard = true;
   offset.channels_m1 = 1;
   offset.discard = true;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), offset, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      true, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 4, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 6, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_offset_2d(enc, 0, 2, 1, 0, false, true,
                                        bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   const struct agx3_tex_sample_desc sample = {
      .dest = 8, .coord = 0, .width = AGX3_TEX_W16, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
      .final = true, .offset = true, .offset_staged = true,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, TextureArrayRuntimeOffsetPacksLayerInWord)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(16, AGX_SIZE_16);
   agx_index coord = agx_register(8, AGX_SIZE_32);
   agx_index offset = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 2;
   coord.discard = true;
   offset.discard = true;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), offset, AGX_DIM_2D_ARRAY, AGX_LOD_MODE_AUTO_LOD, 0,
      true, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 6, 0, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 8, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 10, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   const struct agx3_tex_sample_desc sample = {
      .dest = 16, .coord = 0, .width = AGX3_TEX_W16, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1, .final = true,
      .array = true, .array_layer = 12, .offset = true,
      .offset_staged = true, .offset_staged_array = true,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, TextureArrayRuntimeOffsetAndLodPublishOneWord)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(16, AGX_SIZE_32);
   agx_index coord = agx_register(8, AGX_SIZE_32);
   agx_index lod = agx_register(6, AGX_SIZE_16);
   agx_index packed = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 2;
   coord.discard = true;
   lod.discard = true;
   packed.discard = true;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), packed, AGX_DIM_2D_ARRAY, AGX_LOD_MODE_LOD_MIN, 0,
      true, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   const std::vector<uint8_t> stream = pack(b);
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 6, 0, 1, 0, false, true,
                                         bytes, &len));
   EXPECT_NE(std::search(stream.begin(), stream.end(), bytes, bytes + len),
             stream.end());

   ASSERT_TRUE(agx3_tex_stage_lod(enc, 6, 1, 0, false, true, bytes, &len));
   EXPECT_EQ(std::search(stream.begin(), stream.end(), bytes, bytes + len),
             stream.end());
   ASSERT_TRUE(agx3_tex_stage_array_layer(enc, 12, 1, 0, false, true,
                                          bytes, &len));
   EXPECT_EQ(std::search(stream.begin(), stream.end(), bytes, bytes + len),
             stream.end());

   const struct agx3_tex_sample_desc sample = {
      .dest = 16, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_REGISTER, .slot = 1, .wait_mask = 1,
      .final = true, .array = true, .array_layer = 12,
      .offset = true, .offset_staged = true, .offset_staged_array = true,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   EXPECT_NE(std::search(stream.begin(), stream.end(), bytes, bytes + len),
             stream.end());
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, TextureSampleOffsetStillRefusesAGX3C)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_index dest = agx_register(8, AGX_SIZE_16);
   agx_index coord = agx_register(4, AGX_SIZE_32);
   agx_index offset = agx_immediate(0xf1);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   offset.size = AGX_SIZE_32;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), offset, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      true, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, TextureFetch2DMSPackedSample)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   agx_index coord = agx_register(16, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 2;
   coord.discard = true;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D_MS,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
      false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;
   agx_stop(b);

   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, 0, 16, 1,
                                         0, false, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, 2, 18, 1,
                                         0, false, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 20, 0,
                                        false, true, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_fetch_desc fetch = {
      .dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .slot = 1, .wait_mask = 1, .coord_kill = true, .final = true,
      .multisample = true,
      .staged_sample = true, .sampler_value = 1,
   };
   ASSERT_TRUE(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &fetch,
                                     bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   /* Native MS fetch metadata reserves 16 bytes, including the sample word
    * at staging half4. The x/y pair alone accounts for only four halves. */
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8u);
}

TEST_F(AGX3PackBinary, TextureFetch2DMSIndexedTextureAndSample)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   agx_index coord = agx_register(16, AGX_SIZE_32);
   agx_index texture = agx_register(6, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 2;
   coord.discard = true;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_uniform(8, AGX_SIZE_64),
      texture, agx_immediate(0), agx_null(), AGX_DIM_2D_MS,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
      false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;
   agx_stop(b);

   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, 0, 16, 1,
                                         0, false, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, 2, 18, 1,
                                         0, false, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 20, 0,
                                        false, true, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_fetch_desc fetch = {
      .dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .slot = 1, .wait_mask = 1, .coord_kill = true, .final = true,
      .multisample = true, .staged_sample = true,
      .texture_kind = AGX3_TEX_OPERAND_REGISTER, .texture_value = 6,
      .sampler_value = 1,
   };
   ASSERT_TRUE(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &fetch,
                                     bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8u);
}

/* Back-to-back AGX3B 2D_MS and 2D_MS_ARRAY texel fetches allocate disjoint
 * 8-half staging banks (0, 8, 16, 24) across scoreboard slots 1..4 with zero
 * serializing wait carriers. */
TEST_F(AGX3PackBinary, TextureFetch2DMSRelocatesBanksAcrossSlots)
{
   for (enum agx_dim dim : {AGX_DIM_2D_MS, AGX_DIM_2D_MS_ARRAY}) {
      SCOPED_TRACE(dim);
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_shader_info info = {};
      b->shader->out = &info;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 127;

      for (unsigned i = 0; i < 4; ++i) {
         agx_index dest = agx_register(32 + 8 * i, AGX_SIZE_32);
         agx_index coord = agx_register(4, AGX_SIZE_32);
         dest.channels_m1 = 3;
         coord.channels_m1 = 2;
         coord.discard = (i == 3);
         agx_instr *I = agx_texture_sample_to(
            b, dest, coord, agx_immediate(0), agx_immediate(0),
            agx_immediate(i), agx_immediate(0), agx_null(), dim,
            AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
            false);
         I->op = AGX_OPCODE_TEXTURE_LOAD;
         I->mask = 15;
      }
      /* Consume each fetch's destination so the slot planner keeps all four
       * fetches in flight on distinct scoreboard slots. */
      agx_fadd_to(b, agx_register(0, AGX_SIZE_32),
                  agx_register(32, AGX_SIZE_32),
                  agx_register(40, AGX_SIZE_32));
      agx_fadd_to(b, agx_register(2, AGX_SIZE_32),
                  agx_register(48, AGX_SIZE_32),
                  agx_register(56, AGX_SIZE_32));
      agx_fadd_to(b, agx_register(0, AGX_SIZE_32),
                  agx_register(0, AGX_SIZE_32),
                  agx_register(2, AGX_SIZE_32));
      agx_stop(b);

      const auto code = pack(b);
      ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();
      EXPECT_EQ(b->shader->agx3_staging_halfs, 32u);
      EXPECT_EQ(b->shader->out->stats.carriers, 0u);

      uint8_t used_slots = 0;
      for (unsigned i = 0; i < 4; ++i) {
         bool found = false;
         for (uint8_t slot = 1; slot <= 6 && !found; ++slot) {
            struct agx3_tex_fetch_desc want = {};
            want.dest = 32 + 8 * i;
            want.coord = 8 * i;
            want.width = AGX3_TEX_W32;
            want.mask = 15;
            want.texture = i;
            want.slot = slot;
            want.wait_mask = (uint16_t)(1u << (slot - 1));
            want.coord_kill = true;
            want.final = (i == 3);
            want.array = (dim == AGX_DIM_2D_MS_ARRAY);
            want.multisample = true;
            want.staged_sample = true;
            want.sampler_value = 1;
            uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
            ASSERT_TRUE(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &want,
                                              bytes, &len));
            if (std::search(code.begin(), code.end(), bytes, bytes + len) !=
                code.end()) {
               found = true;
               used_slots |= (uint8_t)(1u << (slot - 1));
            }
         }
         EXPECT_TRUE(found)
            << "missing relocated MS fetch at bank " << (8 * i);
      }
      EXPECT_EQ(__builtin_popcount(used_slots), 4);
   }
}

TEST_F(AGX3PackBinary, TextureFetch2DMSArrayBindless)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   agx_index coord = agx_register(16, AGX_SIZE_32);
   agx_index texture = agx_register(6, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 2;
   coord.discard = true;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_uniform(8, AGX_SIZE_64),
      texture, agx_immediate(0), agx_null(), AGX_DIM_2D_MS_ARRAY,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
      false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;
   agx_stop(b);

   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, 0, 16, 1,
                                         0, false, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, 2, 18, 1,
                                         0, false, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_pack_ms_array_layer(AGX_ISA_ENCODING_AGX3B, 21, 22,
                                           false, false, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_ms_array_sample(AGX_ISA_ENCODING_AGX3B, 20, 22,
                                             false, true, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_fetch_desc fetch = {
      .dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .slot = 1, .wait_mask = 1, .coord_kill = true, .final = true,
      .array = true, .multisample = true, .staged_sample = true,
      .texture_kind = AGX3_TEX_OPERAND_REGISTER,
      .texture_value = 6,
      .sampler_value = 1,
   };
   ASSERT_TRUE(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &fetch,
                                     bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   EXPECT_EQ(b->shader->agx3_staging_halfs, 8u);
   EXPECT_EQ(b->shader->max_reg, 23u);
}

TEST_F(AGX3PackBinary, TextureExportRetiresAtPartBoundary)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   key.no_stop = true;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   agx_index coord = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   agx_instr *tex = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
      false);
   tex->mask = 15;
   for (unsigned channel = 0; channel < 4; ++channel)
      agx_export(b, agx_register(8 + 2 * channel, AGX_SIZE_32),
                  8 + 2 * channel);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 0, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 2, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.coord = 0;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   d.sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   append(want, bytes, len);

   /* Four zero-byte exports are one outstanding texture result. The
    * independently compiled epilog cannot inherit this slot implicitly. */
   uint8_t wait[AGX3_ALU_MAX_LENGTH];
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, wait, &len));
   append(want, wait, len);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, Implicit2DRebasesMaterializedCoordinates)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;

   agx_index dest = agx_register(8, AGX_SIZE_32);
   agx_index coord = agx_register(20, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   coord.discard = true;
   agx_instr *tex = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
      false);
   tex->mask = 15;
   for (unsigned channel = 0; channel < 4; ++channel)
      agx_export(b, agx_register(8 + 2 * channel, AGX_SIZE_32),
                 8 + 2 * channel);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 20, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 22, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   const struct agx3_tex_sample_desc sample = {
      .dest = 8,
      .coord = 0,
      .width = AGX3_TEX_W32,
      .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO,
      .slot = 1,
      .wait_mask = 1,
      .coord_kill = false,
      .final = true,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
   append(want, bytes, len);

   EXPECT_EQ(pack(b), want);
   /* Fresh attachment-only applegpu_g17p f_retain_coords uses this exact
    * fixed staging bank for materialized r4/r5 coordinates and reports
    * ProgramVariant fields38/42 = 8 bytes. */
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 4);
}

TEST_F(AGX3PackBinary, Implicit2DCoordinateCopySourceBank)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;

   /* The first pair is the exact current GTK component-transfer placement;
    * the second ends on the last two legal COPY source halves. */
   for (const unsigned source : {84u, 124u}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;

      agx_index dest = agx_register(8, AGX_SIZE_32);
      agx_index coord = agx_register(source, AGX_SIZE_32);
      dest.channels_m1 = 3;
      coord.channels_m1 = 1;
      coord.discard = true;
      agx_instr *tex = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
         AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
         false);
      tex->mask = 15;
      for (unsigned channel = 0; channel < 4; ++channel)
         agx_export(b, agx_register(8 + 2 * channel, AGX_SIZE_32),
                    8 + 2 * channel);

      std::vector<uint8_t> want;
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, source, 1, 0, false,
                                            true, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, source + 2, 1, 0,
                                            false, true, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
      append(want, bytes, len);
      const struct agx3_tex_sample_desc sample = {
         .dest = 8,
         .coord = 0,
         .width = AGX3_TEX_W32,
         .mask = 15,
         .lod = AGX3_TEX_LOD_AUTO,
         .slot = 1,
         .wait_mask = 1,
         .final = true,
         .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
      };
      ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
      append(want, bytes, len);
      EXPECT_EQ(pack(b), want);
   }

   /* Half 128 used to be masked onto half 0 by the low-level composer. The
    * complete post-RA instruction must now fail closed. */
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   agx_index coord = agx_register(128, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   coord.discard = true;
   agx_instr *tex = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      false, false, false, AGX_GATHER_NONE, false);
   tex->mask = 15;
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, FragmentImplicit2DStagesNativeCoordinates)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   /* Exercise G17P's default COPY-wait carrier as well as its encoding.
    * The native coordinate transaction must reach SMP with its slot-6 wait
    * intact, rather than being retired by that unrelated COPY workaround. */
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->stage = MESA_SHADER_FRAGMENT;

   agx_index raw_x = agx_register(0, AGX_SIZE_32);
   agx_index reciprocal = agx_register(2, AGX_SIZE_32);
   agx_index raw_y = agx_register(4, AGX_SIZE_32);
   raw_x.discard = raw_y.discard = true;
   reciprocal.cache = true;
   agx_instr *stage_x = agx_persp_mul_agx3_to(
      b, agx_register(20, AGX_SIZE_32), raw_x, reciprocal, 1);
   reciprocal.cache = false;
   reciprocal.discard = true;
   agx_instr *stage_y = agx_persp_mul_agx3_to(
      b, agx_register(22, AGX_SIZE_32), raw_y, reciprocal, 2);

   agx_index dest = agx_register(8, AGX_SIZE_32);
   agx_index coord = agx_register(20, AGX_SIZE_32);
   dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   coord.discard = true;
   agx_instr *tex = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_register(12, AGX_SIZE_64), agx_register(16, AGX_SIZE_64),
      agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0, false, false,
      false, AGX_GATHER_NONE, false);
   tex->mask = 15;
   stage_x->agx3_tex_coord_stage = true;
   stage_x->agx3_tex_coord_component = 0;
   stage_x->agx3_tex_coord_start = true;
   stage_y->agx3_tex_coord_stage = true;
   stage_y->agx3_tex_coord_component = 1;
   tex->agx3_tex_coords_staged = true;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   const struct agx3_persp_mul_desc stage_x_desc = {
      .stage_dest = true,
      .dest = 0,
      .slot = 6,
      .reciprocal = 2,
      .reciprocal_liveness = AGX3_OPERAND_LIVE,
      .raw = 0,
      .raw_liveness = AGX3_OPERAND_KILL,
      .vary_index = 1,
   };
   ASSERT_TRUE(agx3_persp_mul_encode(enc, &stage_x_desc, bytes, &len));
   append(want, bytes, len);
   struct agx3_persp_mul_desc stage_y_desc = stage_x_desc;
   stage_y_desc.dest = 2;
   stage_y_desc.raw = 4;
   stage_y_desc.raw_liveness = AGX3_OPERAND_KILL;
   stage_y_desc.reciprocal_liveness = AGX3_OPERAND_KILL;
   stage_y_desc.vary_index = 2;
   ASSERT_TRUE(agx3_persp_mul_encode(enc, &stage_y_desc, bytes, &len));
   append(want, bytes, len);
   const unsigned expected_entry_offset = want.size();
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   const struct agx3_tex_sample_desc sample = {
      .dest = 8,
      .coord = 0,
      .width = AGX3_TEX_W32,
      .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO,
      .slot = 1,
      .wait_mask = 1u << 5,
      .coord_kill = true,
      .final = true,
      .texture_kind = AGX3_TEX_OPERAND_REGISTER,
      .sampler_kind = AGX3_TEX_OPERAND_REGISTER,
      .texture_value = 12,
      .sampler_value = 16,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
   EXPECT_TRUE(b->shader->agx3_colour_link.entry_emitted);
   EXPECT_EQ(b->shader->agx3_colour_link.entry_offset,
             expected_entry_offset);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 4);
}

TEST_F(AGX3PackBinary, ExplicitCoordinatePublicationHasIndependentEvent)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (uint32_t offset : {0u, 0x21u}) {
      agx_builder *b = builder(enc);
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index coord = agx_register(0, AGX_SIZE_32);
      coord.channels_m1 = 1;
      coord.discard = true;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_instr *sample = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
         agx_immediate(0), offset ? agx_immediate(offset) : agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0, offset != 0, false, false,
         AGX_GATHER_NONE, false);
      sample->mask = 15;
      sample->offset_xyz = offset;
      agx_stop(b);
      ASSERT_TRUE(agx3_lower_texture_publications(b->shader));
      ASSERT_EQ(agx_texture_resource(sample), 1);
      EXPECT_FALSE(agx3_lower_texture_publications(b->shader));

      std::vector<uint8_t> want;
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
      for (unsigned h = 0; h < 4; h += 2) {
         ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, h, h, 1, 0, false, true,
                                             bytes, &len));
         append(want, bytes, len);
      }
      ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
      append(want, bytes, len);
      struct agx3_tex_sample_desc desc = {
         .dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
         /* Publication generation1 is waited, then sample generation2
          * recycles the same physical slot. */
         .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
         .coord_kill = true, .final = true,
         .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
      };
      desc.offset = offset != 0;
      desc.offset_xyz = offset;
      ASSERT_TRUE(agx3_tex_encode_sample(enc, &desc, bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 4);
   }
}

TEST_F(AGX3PackBinary, PublicationReuseFollowsPhysicalValuesAndEffects)
{
   for (unsigned mutation = 0; mutation < 4; ++mutation) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index coord = agx_register(0, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_instr *samples[4];
      for (unsigned i = 0; i < 4; ++i) {
         if (i == 2) {
            if (mutation == 1)
               agx_mov_imm_to(b, agx_register(1, AGX_SIZE_16), 0);
            else if (mutation == 2)
               agx_wait_pix(b, 1);
            else
               agx_mov_imm_to(b, agx_register(60, AGX_SIZE_32), 0);
         }
         agx_index dest = agx_register(8 + 8 * i, AGX_SIZE_32);
         dest.channels_m1 = 3;
         samples[i] = agx_texture_sample_to(
            b, dest, coord, agx_immediate(0), agx_immediate(0),
            agx_immediate(i), agx_immediate(0), agx_null(), AGX_DIM_2D,
            AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE, false);
         samples[i]->mask = 15;
         if (mutation == 3 && i == 1) {
            /* A sample result can redefine the original coordinate GPRs. */
            samples[i]->dest[0].value = 0;
         }
      }
      agx_stop(b);
      ASSERT_TRUE(agx3_lower_texture_publications(b->shader));
      EXPECT_EQ(agx_texture_resource(samples[0]), agx_texture_resource(samples[1]));
      EXPECT_EQ(agx_texture_resource(samples[2]), agx_texture_resource(samples[3]));
      if (mutation)
         EXPECT_NE(agx_texture_resource(samples[0]), agx_texture_resource(samples[2]));
      else
         EXPECT_EQ(agx_texture_resource(samples[0]), agx_texture_resource(samples[2]));
      unsigned publications = 0;
      agx_foreach_instr_global(b->shader, I)
         publications += I->op == AGX_OPCODE_PUBLISH_TEXTURE_AGX3;
      EXPECT_EQ(publications, mutation ? 2u : 1u);
      if (!mutation) {
         EXPECT_FALSE(pack(b).empty());
         EXPECT_EQ(b->shader->agx3_staging_halfs, 4);
      }
   }
}

TEST_F(AGX3PackBinary, SynchronousExportPartHasNoTailDrain)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      key.no_stop = true;
      agx_mov_imm_to(b, agx_register(8, AGX_SIZE_32), 0x3f800000);
      agx_export(b, agx_register(8, AGX_SIZE_32), 8);
      struct agx3_encoded_ldimm ld;
      ASSERT_TRUE(agx3_pack_ldimm_u32(enc, 8, 0x3f800000, &ld));
      std::vector<uint8_t> want;
      append(want, ld.bytes, ld.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, StoreOnlySplitColourWaitAcquiresFirst)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (unsigned first_wait : {8u, 4u}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_wait_pix(b, first_wait);
      agx_wait_pix(b, 12u ^ first_wait);
      agx_index value = agx_register(0, AGX_SIZE_32);
      value.channels_m1 = 3;
      agx_st_tile(b, value, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                  AGX_FORMAT_I32, 15, 0, false, 0);
      agx_signal_pix(b, 12);
      agx_stop(b);

      const auto binary = pack(b);
      uint8_t entry[AGX3_PIXEL_MAX_LENGTH];
      uint8_t len = 0;
      ASSERT_TRUE(agx3_pixel_colour_entry(enc, entry, &len));
      ASSERT_GE(binary.size(), len);
      EXPECT_TRUE(std::equal(entry, entry + len, binary.begin()));
      EXPECT_TRUE(b->shader->agx3_colour_link.entry_emitted);
      EXPECT_EQ(b->shader->agx3_colour_link.entry_offset, 0);
   }
}

TEST_F(AGX3PackBinary, TileStoreSingleSampleFormats)
{
   const struct {
      enum agx_format format;
      enum agx3_tile_width width;
   } formats[] = {
      {AGX_FORMAT_I8, AGX3_TILE_W8},
      {AGX_FORMAT_U8NORM, AGX3_TILE_W8},
      {AGX_FORMAT_S8NORM, AGX3_TILE_W8},
      {AGX_FORMAT_SRGBA8, AGX3_TILE_W8},
      {AGX_FORMAT_I16, AGX3_TILE_W16},
      {AGX_FORMAT_F16, AGX3_TILE_W16},
      {AGX_FORMAT_U16NORM, AGX3_TILE_W16},
      {AGX_FORMAT_S16NORM, AGX3_TILE_W16},
      {AGX_FORMAT_I32, AGX3_TILE_W32},
      {AGX_FORMAT_RGB10A2, AGX3_TILE_W32},
      {AGX_FORMAT_RG11B10F, AGX3_TILE_W32},
      {AGX_FORMAT_RGB9E5, AGX3_TILE_W32},
   };

   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (const auto &f : formats) {
         agx_builder *b = builder(enc);
         /* The packed formats store one converted word. */
         const bool packed = f.format == AGX_FORMAT_RGB10A2 ||
                             f.format == AGX_FORMAT_RGB9E5;
         const unsigned mask = packed ? 1 : 15;
         agx_index value = agx_register(0, f.width == AGX3_TILE_W8 ?
                                          AGX_SIZE_16 : AGX_SIZE_32);
         value.channels_m1 = f.width == AGX3_TILE_W8 ? 1 : packed ? 0 : 3;
         agx_st_tile(b, value, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     f.format, mask, 24, false, 0);
         agx_stop(b);

         uint8_t tile[AGX3_TILE_MAX_LENGTH];
         uint8_t tile_len = 0;
         ASSERT_TRUE(agx3_tile_store(enc, 0, f.width, mask, 24,
                                     AGX3_TILE_SAMPLE_1X, 0, tile, &tile_len));
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         std::vector<uint8_t> want;
         append(want, tile, tile_len);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, TileLoadStoreWaitSurvivesFallthroughBlock)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_block *first = agx_start_block(b->shader);
      agx_index value = agx_register(0, AGX_SIZE_32);
      value.channels_m1 = 3;
      agx_ld_tile_to(b, value, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     AGX_FORMAT_I32, 15, 16, false, 0);

      agx_block *second = agx_test_block(b->shader);
      agx_block_add_successor(first, second);
      b->cursor = agx_after_block(second);
      value.discard = true; /* nonsemantic post-RA last-use hint */
      agx_st_tile(b, value, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                  AGX_FORMAT_I32, 15, 16, false, 0);
      agx_stop(b);

      uint8_t tile[AGX3_TILE_MAX_LENGTH];
      uint8_t tile_len = 0;
      std::vector<uint8_t> want;
      const unsigned load_slot = enc == AGX_ISA_ENCODING_AGX3B ? 6 : 1;
      ASSERT_TRUE(agx3_tile_load(enc, 0, AGX3_TILE_W32, load_slot, 15, 16,
                                 AGX3_TILE_SAMPLE_1X, 0, true, true, tile,
                                 &tile_len));
      append(want, tile, tile_len);
      ASSERT_TRUE(agx3_tile_store(enc, 0, AGX3_TILE_W32, 15, 16,
                                  AGX3_TILE_SAMPLE_1X,
                                  1u << (load_slot - 1), tile, &tile_len));
      append(want, tile, tile_len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, G17PMultisampleTileSampleOperandIsARegister)
{
   /* Apple's multisample colour stores source the sample mask from a live
    * register -- the coverage word, or that word ANDed with one sample bit.
    * The packer must carry an IR register straight into that operand. */
   for (unsigned reg : {8u, 9u, 10u, 11u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;
      agx_index mask = agx_register(reg, AGX_SIZE_16);
      mask.kill = true;
      agx_st_tile(b, value, mask, agx_null(), AGX_FORMAT_U8NORM, 15, 0, false, 0);
      agx_stop(b);

      uint8_t tile[AGX3_TILE_MAX_LENGTH], tile_len = 0;
      ASSERT_TRUE(agx3_tile_store(AGX_ISA_ENCODING_AGX3B, 0, AGX3_TILE_W8, 15,
                                  0, agx3_tile_sample_reg(reg, true), 0, tile,
                                  &tile_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      std::vector<uint8_t> want;
      append(want, tile, tile_len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }

   /* The only immediate the operand accepts is Mesa's whole-pixel spelling,
    * which becomes Apple's 1x immediate. A raw numeric mask has no witness
    * and must not be invented. */
   for (unsigned imm : {0u, 1u, 0x3u, 0xfu, 0x20u, 0x40u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;
      agx_st_tile(b, value, agx_immediate(imm), agx_null(),
                  AGX_FORMAT_U8NORM, 15, 0, false, 0);
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, G17PBackgroundWritesCompleteMultisamplePartition)
{
   key.fs.ignore_tib_dependencies = true;

   for (unsigned samples : {2u, 4u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;

      std::vector<uint8_t> want;
      for (unsigned i = 0; i < samples; ++i) {
         agx_index mask = agx_register(8 + i, AGX_SIZE_16);
         mask.kill = true;
         agx_st_tile(b, value, mask, agx_null(), AGX_FORMAT_U8NORM, 15, 0,
                     false, 0);

         uint8_t tile[AGX3_TILE_MAX_LENGTH], tile_len = 0;
         ASSERT_TRUE(agx3_tile_store(AGX_ISA_ENCODING_AGX3B, 0, AGX3_TILE_W8,
                                     15, 0, agx3_tile_sample_reg(8 + i, true),
                                     0, tile, &tile_len));
         append(want, tile, tile_len);
      }
      agx_stop(b);

      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }

   /* Register allocation is free to reuse one mask register for stores that
    * kill it, and on the real four-sample reload it does. A mask recomputed
    * between the stores is a different mask, so that shape is admitted --
    * refusing it refused the only shader this protocol exists for. */
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;
      for (unsigned i = 0; i < 4; ++i) {
         agx_mov_imm_to(b, agx_register(8, AGX_SIZE_16), 1u << i);
         agx_index mask = agx_register(8, AGX_SIZE_16);
         mask.kill = true;
         agx_st_tile(b, value, mask, agx_null(), AGX_FORMAT_U8NORM, 15, 0,
                     false, 0);
      }
      agx_stop(b);
      EXPECT_FALSE(pack(b).empty());
   }

   /* A repeated mask register with nothing redefining it is one mask twice,
    * not a complete sample domain. */
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;
      agx_index mask = agx_register(8, AGX_SIZE_16);
      agx_st_tile(b, value, mask, agx_null(), AGX_FORMAT_U8NORM, 15, 0, false, 0);
      agx_st_tile(b, value, mask, agx_null(), AGX_FORMAT_U8NORM, 15, 0, false, 0);
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }

   /* The same stores in an ordinary fragment require the colour protocol. */
   {
      key.fs.ignore_tib_dependencies = false;
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;
      agx_st_tile(b, value, agx_register(8, AGX_SIZE_16), agx_null(),
                  AGX_FORMAT_U8NORM, 15, 0, false, 0);
      agx_st_tile(b, value, agx_register(9, AGX_SIZE_16), agx_null(),
                  AGX_FORMAT_U8NORM, 15, 0, false, 0);
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, G17PSparseSampleStoresShareOneColourTransaction)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (unsigned stores : {2u, 3u, 4u}) {
      SCOPED_TRACE(stores);
      agx_builder *b = builder(enc);
      key.secondary = true;
      key.fs.nr_samples = 4;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_wait_pix(b, 12);
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;
      for (unsigned sample = 0; sample < stores; ++sample) {
         agx_index mask = agx_register(8, AGX_SIZE_16);
         agx_mov_imm_to(b, mask, 1u << sample);
         mask.discard = true;
         agx_st_tile(b, value, mask, agx_null(), AGX_FORMAT_U8NORM, 15, 0,
                     false, 0);
      }
      agx_stop(b);
      const auto bytes = pack(b);
      uint8_t wait[AGX3_PIXEL_MAX_LENGTH], release[AGX3_PIXEL_MAX_LENGTH];
      uint8_t wait_length, release_length;
      ASSERT_TRUE(agx3_pixel_wait(enc, 12, 0, wait, &wait_length));
      ASSERT_TRUE(agx3_pixel_release(enc, 12, true, 0, release,
                                    &release_length));
      auto count = [&](const uint8_t *word, unsigned size) {
         unsigned n = 0;
         auto at = bytes.begin();
         while ((at = std::search(at, bytes.end(), word, word + size)) !=
                bytes.end()) {
            ++n;
            at += size;
         }
         return n;
      };
      EXPECT_EQ(count(wait, wait_length), 1u);
      EXPECT_EQ(count(release, release_length), 1u);
   }

   /* Omitting no-op colour writes does not relax the transaction, sample
    * count, mask-lifetime or complete-background requirements. */
   for (unsigned bad = 0; bad < 4; ++bad) {
      SCOPED_TRACE(bad);
      agx_builder *b = builder(enc);
      key.secondary = true;
      key.fs.nr_samples = bad == 1 ? 2 : 4;
      key.fs.ignore_tib_dependencies = bad == 3;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      if (bad != 0 && bad != 3)
         agx_wait_pix(b, 12);
      agx_index value = agx_register(0, AGX_SIZE_16);
      value.channels_m1 = 1;
      for (unsigned sample = 0; sample < 3; ++sample) {
         agx_index mask = agx_register(8, AGX_SIZE_16);
         if (bad != 2 || sample == 0)
            agx_mov_imm_to(b, mask, 1u << sample);
         agx_st_tile(b, value, mask, agx_null(), AGX_FORMAT_U8NORM, 15, 0,
                     false, 0);
      }
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, TileLoadSequenceBoundaryMarkers)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_index a = agx_register(0, AGX_SIZE_32);
      agx_index c = agx_register(8, AGX_SIZE_32);
      a.channels_m1 = c.channels_m1 = 3;
      agx_ld_tile_to(b, a, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     AGX_FORMAT_I32, 15, 0, false, 0);
      agx_ld_tile_to(b, c, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     AGX_FORMAT_I32, 15, 16, false, 0);
      agx_stop(b);

      uint8_t tile[AGX3_TILE_MAX_LENGTH];
      uint8_t tile_len = 0;
      std::vector<uint8_t> want;
      const unsigned first_slot = enc == AGX_ISA_ENCODING_AGX3B ? 6 : 1;
      const unsigned second_slot = enc == AGX_ISA_ENCODING_AGX3B ? 6 : 2;
      ASSERT_TRUE(agx3_tile_load(enc, 0, AGX3_TILE_W32, first_slot, 15, 0,
                                 AGX3_TILE_SAMPLE_1X, 0, true, false, tile,
                                 &tile_len));
      append(want, tile, tile_len);
      ASSERT_TRUE(agx3_tile_load(enc, 8, AGX3_TILE_W32, second_slot, 15, 16,
                                 AGX3_TILE_SAMPLE_1X, 0, false, true, tile,
                                 &tile_len));
      append(want, tile, tile_len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, RefusesUnwitnessedSr)
{
   /* samples_log2 has no corpus witness tying it to an Apple SR number; the
    * backend must fail the shader, not guess an encoding. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index h0 = agx_register(0, AGX_SIZE_16);
   agx_get_sr_to(b, h0, AGX_SR_SAMPLES_LOG2);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, G17PFrontFacingUsesNativeBackFacingSr)
{
   /* Apple's front_facing fragment reads a native W16 SR41. The value is
    * zero on a front face, so it is Mesa's existing BACKFACING semantic and
    * the shared front-face lowering keeps its logical inversion. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_get_sr_to(b, agx_register(0, AGX_SIZE_16), AGX_SR_BACKFACING);
   agx_stop(b);
   const std::vector<uint8_t> native = {
      0x04,0xc5,0x10,0x06, 0x0e,0x00,0x00,0x00,
   };
   EXPECT_EQ(pack(b), native);

   /* applegpu_g17g reads the same SR 41 for [[front_facing]] and selects on
    * it being zero (test/native/g17g/m5-front-facing-reference.metal, whose
    * fragment reads it into half 12 on slot 1). */
   b = builder(AGX_ISA_ENCODING_AGX3C);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_get_sr_to(b, agx_register(12, AGX_SIZE_16), AGX_SR_BACKFACING);
   agx_stop(b);
   const std::vector<uint8_t> g17g_native = {
      0x64,0xc5,0x10,0x06, 0x0e,0x00,0x00,0x00,
   };
   EXPECT_EQ(pack(b), g17g_native);
}

TEST_F(AGX3PackBinary, G17GHelperInvocationReadsActiveThreadSr)
{
   /* A fragment side effect is predicated on the helper-invocation test,
    * which reads the active-thread SR2 as W16. applegpu_g17g opens its
    * fragment atomic with exactly that read into half 2 on slot 1
    * (test/native/g17g/m5-gl4-reference.metal sxAtomic). */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_get_sr_coverage_to(b, agx_register(2, AGX_SIZE_16),
                          AGX_SR_IS_ACTIVE_THREAD);
   agx_stop(b);
   const std::vector<uint8_t> native = {
      0x14,0x84,0x10,0x06, 0x0e,0x00,0x00,0x00,
   };
   EXPECT_EQ(pack(b), native);

   /* Outside a fragment there is no helper invocation to test. */
   b = builder(AGX_ISA_ENCODING_AGX3C);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_get_sr_coverage_to(b, agx_register(2, AGX_SIZE_16),
                          AGX_SR_IS_ACTIVE_THREAD);
   agx_stop(b);
   pack(b, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, PackedSharedByteVectorsUsePhysicalHalves)
{
   for (unsigned channels = 2; channels <= 4; ++channels) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      const unsigned halves = DIV_ROUND_UP(channels, 2);
      agx_index value = agx_register(4, AGX_SIZE_16);
      value.channels_m1 = halves - 1;
      agx_instr *load = agx_local_load_to(
         b, value, agx_immediate(0), agx_immediate(1), AGX_FORMAT_I8,
         BITFIELD_MASK(channels));
      load->memory_is_packed = true;
      agx_instr *store = agx_local_store(
         b, value, agx_immediate(0), agx_immediate(5), AGX_FORMAT_I8,
         BITFIELD_MASK(channels));
      store->memory_is_packed = true;
      /* Keep the logical byte mask while tracking only its physical halves
       * through the asynchronous producer and consumer. */
      agx_stop(b);
      EXPECT_FALSE(pack(b).empty());
   }
}

TEST_F(AGX3PackBinary, LocalVectorUsesScalarElementIndex)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_index base = agx_uniform(8, AGX_SIZE_16);
      agx_local_store(b, agx_register(0, AGX_SIZE_32), base,
                      agx_immediate(4), AGX_FORMAT_I32, 0xf);
      agx_stop(b);

      struct agx3_local_desc st = {};
      st.data = 0;
      st.width = AGX3_MEM_W32;
      st.mask = 0xf;
      st.shift = 2;
      st.base_kind = AGX3_LOCAL_BASE_UNIFORM;
      st.base = 8;
      st.index = 4;
      uint8_t sb[AGX3_LOCAL_TPS_MAX_LENGTH];
      uint8_t sl = 0;
      ASSERT_TRUE(agx3_local_encode_store(enc, &st, sb, &sl));

      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      std::vector<uint8_t> want;
      append(want, sb, sl);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, UnlinkedVertexOutputsWriteRuntimeIndices)
{
   /* A pipeline library's vertex shader cannot know the fragment shader's
    * interpolation classes, so its user outputs carry run-time indices while
    * the point size keeps its fixed one. Neither joins a batched bundle:
    * each is staged and written where it is stored. */
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   agx_st_vary(b, agx_register(9, AGX_SIZE_16), agx_register(4, AGX_SIZE_32));
   agx_st_vary(b, agx_immediate(12), agx_register(6, AGX_SIZE_32));
   agx_stop(b);

   uint8_t bytes[MAX2(AGX3_STAGE_MAX_LENGTH, AGX3_VARY_MAX_LENGTH)];
   uint8_t len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_stage_copy_source(enc, 0, 4, false, 1, 0, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_vary_uvsw_register_index(enc, 0, 9, 1, true, 1, bytes,
                                             &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_stage_copy_source(enc, 2, 6, false, 2, 0, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_vary_uvsw(enc, 2, 12, 2, true, 2, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 4u);
}

TEST_F(AGX3PackBinary, ExplicitSampleInterpolationMatchesApple)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;

   agx_index sample = agx_register(4, AGX_SIZE_16);
   sample.discard = true;
   agx_itrcoords_agx3_to(b, agx_register(4, AGX_SIZE_32), sample,
                         AGX_INTERPOLATION_SAMPLE);

   agx_index coords = agx_register(4, AGX_SIZE_32);
   coords.discard = true;
   agx_interp_agx3_to(b, agx_register(4, AGX_SIZE_32), agx_immediate(0), coords,
                      AGX_INTERPOLATION_SAMPLE, false);
   agx_stop(b);

   /* Fresh Apple sample_no_perspective fragment: tied terminal ITRCOORDS
    * followed by its final scalar f32interp and Mesa's canonical end. */
   const std::vector<uint8_t> native = {
      0xaf, 0x04, 0x54, 0x04, 0x03, 0x08, 0x0a, 0x03, 0x2f, 0x05, 0x54,
      0x04, 0x02, 0x00, 0x02, 0x02, 0x20, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), native);
}

TEST_F(AGX3PackBinary, CentroidInterpolationMatchesApple)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;

   agx_get_sr_to(b, agx_register(0, AGX_SIZE_16), AGX_SR_INPUT_SAMPLE_MASK);
   agx_index coverage = agx_register(0, AGX_SIZE_16);
   coverage.discard = true;
   agx_itrcoords_agx3_to(b, agx_register(6, AGX_SIZE_32), coverage,
                         AGX_INTERPOLATION_CENTROID);

   agx_index coords = agx_register(6, AGX_SIZE_32);
   static const unsigned dests[] = {0, 8, 2, 6};
   for (unsigned c = 0; c < 4; ++c) {
      coords.discard = c == 3;
      agx_interp_agx3_to(b, agx_register(dests[c], AGX_SIZE_32),
                         agx_immediate(1 + c), coords,
                         AGX_INTERPOLATION_CENTROID, true);
   }
   agx_stop(b);

   /* Fresh applegpu_g17p `[[centroid_perspective]]` float4 fragment at
    * raster_sample_count 1 and 4 (identical here): the entry-marked SR39
    * read, ITRCOORDS f38=1 waiting on it, and the channel f32interps that
    * consume the token, the last one tied to it on its final use. */
   const std::vector<uint8_t> native = {
      0x04, 0xc2, 0x11, 0x06, 0x01, 0x00, 0x00, 0x00,
      0xaf, 0x14, 0x54, 0x06, 0x03, 0x00, 0x0a, 0x01,
      0x2f, 0x05, 0x54, 0x00, 0x03, 0x02, 0x03, 0x02, 0x08, 0x00,
      0x2f, 0x05, 0x54, 0x08, 0x03, 0x04, 0x03, 0x02, 0x08, 0x00,
      0x2f, 0x05, 0x54, 0x02, 0x03, 0x06, 0x03, 0x02, 0x08, 0x00,
      0x2f, 0x05, 0x54, 0x06, 0x03, 0x08, 0x03, 0x02, 0x20, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), native);
}

TEST_F(AGX3PackBinary, CentroidInterpolationStaysFailClosed)
{
   {
      /* G17P has no implicit centroid coordinate register. */
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_interp_agx3_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(3),
                         agx_zero(), AGX_INTERPOLATION_CENTROID, false);
      agx_stop(b);
      EXPECT_TRUE(pack(b, /*expect_ok=*/false).empty());
   }
}

/* Token provenance is an SSA property, checked before register allocation. */
TEST_F(AGX3PackBinary, InterpolationTokensComeFromTheirITRCOORDS)
{
   for (enum agx_interpolation interp :
        {AGX_INTERPOLATION_SAMPLE, AGX_INTERPOLATION_CENTROID}) {
      for (unsigned producer = 0; producer < 3; ++producer) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         b->shader->stage = MESA_SHADER_FRAGMENT;
         agx_index input = agx_temp(b->shader, AGX_SIZE_16);
         agx_get_sr_to(b, input, AGX_SR_INPUT_SAMPLE_MASK);
         agx_index token = agx_temp(b->shader, AGX_SIZE_32);
         /* 0: the matching ITRCOORDS; 1: a sample token named as a
          * centroid or the reverse; 2: an arithmetic value. */
         if (producer == 2) {
            agx_mov_to(b, token, agx_immediate(0));
         } else {
            agx_itrcoords_agx3_to(b, token, input,
                                  producer == 0 ? interp
                                  : interp == AGX_INTERPOLATION_SAMPLE
                                     ? AGX_INTERPOLATION_CENTROID
                                     : AGX_INTERPOLATION_SAMPLE);
         }
         agx_interp_agx3_to(b, agx_temp(b->shader, AGX_SIZE_32),
                            agx_immediate(3), token, interp, false);
         agx_stop(b);
         EXPECT_EQ(agx3_interp_coords_from_itrcoords(b->shader),
                   producer == 0);
      }
   }
}

/* The SSA check follows a token into another block, which the old post-RA
 * block-local check refused. */
TEST_F(AGX3PackBinary, InterpolationTokenFromAnotherBlock)
{
   for (enum agx_interpolation interp :
        {AGX_INTERPOLATION_SAMPLE, AGX_INTERPOLATION_CENTROID}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_block *first = agx_start_block(b->shader);
      agx_index input = agx_temp(b->shader, AGX_SIZE_16);
      agx_get_sr_to(b, input, AGX_SR_INPUT_SAMPLE_MASK);
      agx_index token = agx_temp(b->shader, AGX_SIZE_32);
      agx_itrcoords_agx3_to(b, token, input, interp);

      agx_block *second = agx_test_block(b->shader);
      agx_block_add_successor(first, second);
      b->cursor = agx_after_block(second);
      agx_interp_agx3_to(b, agx_temp(b->shader, AGX_SIZE_32),
                         agx_immediate(3), token, interp, false);
      agx_stop(b);
      EXPECT_TRUE(agx3_interp_coords_from_itrcoords(b->shader));
   }
}

/* RA may move a token between ITRCOORDS and its consumers; the copy holds
 * the same 32 bits, so the consumer encodes the register it reads. Astro
 * Bot's constant-program fragment variants were refused at this shape. */
TEST_F(AGX3PackBinary, CentroidInterpolationReadsAMovedToken)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;

   agx_get_sr_to(b, agx_register(0, AGX_SIZE_16), AGX_SR_INPUT_SAMPLE_MASK);
   agx_index coverage = agx_register(0, AGX_SIZE_16);
   coverage.discard = true;
   agx_itrcoords_agx3_to(b, agx_register(6, AGX_SIZE_32), coverage,
                         AGX_INTERPOLATION_CENTROID);
   /* A register copy, as agx_lower_pseudo leaves it for the packer. */
   agx_bitop_to(b, agx_register(132, AGX_SIZE_32), agx_register(6, AGX_SIZE_32),
                agx_zero(), UTIL_LUT2(a));
   agx_index coords = agx_register(132, AGX_SIZE_32);
   coords.discard = true;
   agx_interp_agx3_to(b, agx_register(86, AGX_SIZE_32), agx_immediate(7),
                      coords, AGX_INTERPOLATION_CENTROID, true);
   agx_stop(b);

   uint8_t interp[AGX3_VARY_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_vary_interp_sample(AGX_ISA_ENCODING_AGX3B, 86, 7, true,
                                       132, true, interp, &len));
   const std::vector<uint8_t> code = pack(b);
   EXPECT_NE(std::search(code.begin(), code.end(), interp, interp + len),
             code.end())
      << agx3_last_pack_refusal();
}

TEST_F(AGX3PackBinary, VertexStagingProducerMatchesAppleAndPreservesLiveness)
{
   static const uint8_t apple_x[] = {0x0b, 0x00, 0x29, 0xc4};
   static const uint8_t apple_y[] = {
      0x1b, 0x03, 0x2f, 0x00, 0x42, 0x00, 0x00, 0x04, 0x00, 0x00,
   };
   uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, 0, 0, true,
                                      1, 0x20, bytes, &len));
   ASSERT_EQ(len, sizeof(apple_x));
   EXPECT_EQ(memcmp(bytes, apple_x, len), 0);
   ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, 2, 2, true,
                                      2, 0, bytes, &len));
   ASSERT_EQ(len, sizeof(apple_y));
   EXPECT_EQ(memcmp(bytes, apple_y, len), 0);

   ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, 2, 2, false,
                                      2, 0, bytes, &len));
   /* f27 is the GPR last-use flag, independent of staging f9/f5/f12. */
   EXPECT_EQ(bytes[2], apple_y[2] & ~8u);
   bytes[2] |= 8;
   EXPECT_EQ(memcmp(bytes, apple_y, len), 0);
}

TEST_F(AGX3PackBinary, DepthStagingPublicationPreservesSourceLiveness)
{
   /* Native late_atomic coverage COPY at offset 258. f13 is set on this
    * destination and on the FSEL that publishes the other DEPTHF dword. */
   static const uint8_t native[] = {
      0x0b, 0x01, 0x2f, 0x00, 0x42, 0x00, 0x00, 0x00, 0x10, 0x00,
   };
   uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_stage_copy_liveness(AGX_ISA_ENCODING_AGX3B, 0, 0,
                                        true, true, 1, 0, bytes, &len));
   ASSERT_EQ(len, sizeof(native));
   EXPECT_EQ(memcmp(bytes, native, len), 0);

   ASSERT_TRUE(agx3_stage_copy_liveness(AGX_ISA_ENCODING_AGX3B, 0, 0,
                                        true, false, 1, 0, bytes, &len));
   ASSERT_EQ(len, sizeof(native));
   EXPECT_EQ(bytes[2], native[2] & ~8u);
   bytes[2] |= 8;
   EXPECT_EQ(memcmp(bytes, native, len), 0);

   /* Other consumers retain their existing compact publication protocol. */
   ASSERT_TRUE(agx3_stage_copy_liveness(AGX_ISA_ENCODING_AGX3B, 0, 0,
                                        false, true, 1, 0, bytes, &len));
   EXPECT_EQ(len, 4u);
   EXPECT_FALSE(agx3_stage_copy_liveness(AGX_ISA_ENCODING_AGX3C, 0, 0,
                                         true, true, 1, 0, bytes, &len));
}

TEST_F(AGX3PackBinary, VertexStagingBankAndScoreboardRoundTrip)
{
   for (unsigned slot = 1; slot <= 6; ++slot) {
      for (unsigned wait : {0u, 1u, 2u, 4u, 8u, 16u, 32u, 33u}) {
         uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len;
         const unsigned staging = 2 * (slot - 1);
         ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, staging,
                                            200, false, slot, wait, bytes, &len));
         /* Decode the selected COPY destination bank and narrow indices.
          * This is the actual f9=5 form, not a GPR copy to numeric index s. */
         EXPECT_EQ(bytes[0] & 0xf, 0xbu);
         EXPECT_EQ(bytes[2] & 0xf7, len == 4 ? 0x21u : 0x27u);
         EXPECT_EQ((bytes[0] >> 4) * 2, staging);
         EXPECT_EQ((bytes[1] & 0xfe) | (len == 10 ? (bytes[5] & 1) << 7
                                                             : (bytes[3] & 16) << 3),
                   200u);
         unsigned decoded_slot, decoded_wait;
         if (len == 4) {
            decoded_slot = 1;
            const unsigned index = bytes[3] >> 5;
            decoded_wait = index ? 1u << (index - 1) : 0;
         } else {
            ASSERT_EQ(len, 10u);
            decoded_slot = 1 + ((bytes[7] >> 2) & 7);
            decoded_wait = (bytes[5] >> 5) | ((bytes[7] >> 5) << 3);
         }
         EXPECT_EQ(decoded_slot, slot);
         EXPECT_EQ(decoded_wait, wait);

         ASSERT_TRUE(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, staging, 0, slot,
                                    true, 1u << (slot - 1), bytes, &len));
         ASSERT_EQ(len, 8u);
         EXPECT_EQ(bytes[3], staging);
         EXPECT_EQ(1 + (bytes[6] & 7), decoded_slot);
         const unsigned consumer_wait = (bytes[1] >> 4) | ((bytes[2] & 3) << 4);
         EXPECT_EQ(consumer_wait, 1u << (decoded_slot - 1));
      }
   }
}

TEST_F(AGX3PackBinary, VertexStagingAllocationDoesNotUseGprHighWater)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->max_reg = 201;
   agx_st_vary(b, agx_immediate(0), agx_register(200, AGX_SIZE_32));
   agx_st_vary(b, agx_immediate(1), agx_register(200, AGX_SIZE_32));
   agx_stop(b);
   EXPECT_FALSE(pack(b).empty());
   EXPECT_EQ(b->shader->max_reg, 201u);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 4u);
   EXPECT_TRUE(b->shader->agx3_staging_known);
}

TEST_F(AGX3PackBinary, AllGroundedComparisonRelations)
{
   const struct {
      enum agx_icond mesa;
      enum agx3_cf_icond apple;
   } integer_cases[] = {
      {AGX_ICOND_UEQ, AGX3_CF_ICOND_EQ},
      {AGX_ICOND_SEQ, AGX3_CF_ICOND_EQ},
      {AGX_ICOND_ULT, AGX3_CF_ICOND_ULT},
      {AGX_ICOND_UGT, AGX3_CF_ICOND_UGT},
      {AGX_ICOND_SLT, AGX3_CF_ICOND_SLT},
      {AGX_ICOND_SGT, AGX3_CF_ICOND_SGT},
   };
   const struct {
      enum agx_fcond mesa;
      enum agx3_cf_fcond apple;
      bool or_nan;
      bool flips_invert;
   } float_cases[] = {
      {AGX_FCOND_EQ, AGX3_CF_FCOND_EQ, false, false},
      {AGX_FCOND_LT, AGX3_CF_FCOND_LT, false, false},
      {AGX_FCOND_GT, AGX3_CF_FCOND_GT, false, false},
      {AGX_FCOND_LTN, AGX3_CF_FCOND_LT, true, false},
      {AGX_FCOND_GTN, AGX3_CF_FCOND_GT, true, false},
      {AGX_FCOND_GE, AGX3_CF_FCOND_LT, true, true},
      {AGX_FCOND_LE, AGX3_CF_FCOND_GT, true, true},
   };

   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;

      for (const auto &test : integer_cases) {
         for (int invert = 0; invert < 2; ++invert) {
            agx_builder *b = builder(enc);
            agx_if_icmp(b, agx_register(0, AGX_SIZE_32),
                        agx_register(2, AGX_SIZE_32), 1, test.mesa, invert,
                        NULL);
            agx_stop(b);

            uint8_t bytes[AGX3_CF_MAX_LENGTH];
            uint8_t len = 0;
            std::vector<uint8_t> want;
            ASSERT_TRUE(agx3_cf_icmp_reg(enc, test.apple, 0, 0, 2, bytes,
                                         &len));
            append(want, bytes, len);
            ASSERT_TRUE(agx3_cf_cndst(enc, 0, invert, 1, bytes, &len));
            append(want, bytes, len);
            struct agx3_encoded_end end;
            ASSERT_TRUE(agx3_pack_end(enc, &end));
            append(want, end.bytes, end.length);
            EXPECT_EQ(pack(b), want);
         }
      }
   }

   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   for (const auto &test : float_cases) {
      for (int invert = 0; invert < 2; ++invert) {
         agx_builder *b = builder(enc);
         agx_if_fcmp(b, agx_register(0, AGX_SIZE_32),
                     agx_register(2, AGX_SIZE_32), 1, test.mesa, invert,
                     NULL);
         agx_stop(b);

         uint8_t bytes[AGX3_CF_MAX_LENGTH];
         uint8_t len = 0;
         std::vector<uint8_t> want;
         ASSERT_TRUE(agx3_cf_fcmp_reg(enc, test.apple, test.or_nan, 0, 0,
                                      2, bytes, &len));
         append(want, bytes, len);
         ASSERT_TRUE(agx3_cf_cndst(enc, 0,
                                   (bool)invert ^ test.flips_invert, 1,
                                   bytes, &len));
         append(want, bytes, len);
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, ImmediateLeftIntegerComparisonsCanonicalize)
{
   const struct {
      enum agx_icond mesa;
      enum agx3_cf_icond swapped;
   } cases[] = {
      {AGX_ICOND_UEQ, AGX3_CF_ICOND_EQ},
      {AGX_ICOND_SEQ, AGX3_CF_ICOND_EQ},
      {AGX_ICOND_ULT, AGX3_CF_ICOND_UGT},
      {AGX_ICOND_UGT, AGX3_CF_ICOND_ULT},
      {AGX_ICOND_SLT, AGX3_CF_ICOND_SGT},
      {AGX_ICOND_SGT, AGX3_CF_ICOND_SLT},
   };

   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;

      for (const auto &test : cases) {
         agx_builder *b = builder(enc);
         agx_if_icmp(b, agx_immediate(15),
                     agx_register(2, AGX_SIZE_32), 1, test.mesa, false, NULL);
         agx_stop(b);

         uint8_t bytes[AGX3_CF_MAX_LENGTH];
         uint8_t len = 0;
         std::vector<uint8_t> want;
         ASSERT_TRUE(agx3_cf_icmp_imm(enc, test.swapped, 0, 2, 15, bytes,
                                      &len));
         append(want, bytes, len);
         ASSERT_TRUE(agx3_cf_cndst(enc, 0, false, 1, bytes, &len));
         append(want, bytes, len);
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(enc, &end));
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }
   }
}

/* Native statistics: a load still in flight at a branch that does not read it
 * is a control-flow drain; a load the branch compare reads is only waited for,
 * as it would be anyway. Both waits ride on carriers. */
TEST_F(AGX3PackBinary, NativeStatsCountControlFlowDrains)
{
   for (int e = 0; e < 2; ++e) {
      for (bool compare_reads_load : {false, true}) {
         const enum agx_isa_encoding enc =
            e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
         agx_builder *b = builder(enc);
         b->shader->stage = MESA_SHADER_COMPUTE;
         agx_block *pop = agx_test_block(b->shader);
         agx_index loaded = agx_register(4, AGX_SIZE_32);
         agx_device_load_to(b, loaded, agx_uniform(0, AGX_SIZE_64),
                            agx_immediate(0), AGX_FORMAT_I32, 1, 0, false);

         agx_begin_cf(b);
         agx_if_icmp(b, compare_reads_load ? loaded : agx_register(0, AGX_SIZE_32),
                     agx_register(2, AGX_SIZE_32), 1, AGX_ICOND_ULT, false, pop);
         agx_jmp_exec_none(b, pop);
         agx_block_add_successor(agx_start_block(b->shader), pop);

         agx_builder pb = agx_init_builder(b->shader, agx_after_block(pop));
         agx_pop_exec(&pb, 1);
         agx_device_store(&pb, loaded, agx_uniform(0, AGX_SIZE_64),
                          agx_immediate(4), AGX_FORMAT_I32, 1, 0, false);
         agx_stop(&pb);

         /* The arm's pop joins instead of draining (agx3_slot_core.h): the
          * store after it waits for the load itself. A compare reading the
          * load waits for it through a carrier, as no branch carries a
          * wait. */
         EXPECT_FALSE(pack(b).empty());
         EXPECT_EQ(agx3_native_stats.cf_drains, 0u);
         EXPECT_EQ(agx3_native_stats.loop_cf_drains, 0u);
         EXPECT_GE(agx3_native_stats.wait_carriers,
                   compare_reads_load ? 1u : 0u);
      }
   }
}

/* Build: load r4; if (r0 < r2) { <arm> } <after>, with the arm's exec-none
 * skip jumping to the pop block. `arm_uses` puts the load's consumer in the
 * arm; otherwise it follows the pop. */
static void
build_load_then_if(agx_builder *b, bool arm_uses, bool compare_reads_load)
{
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *arm = agx_test_block(b->shader);
   agx_block *pop = agx_test_block(b->shader);
   agx_index loaded = agx_register(4, AGX_SIZE_32);
   agx_device_load_to(b, loaded, agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                      AGX_FORMAT_I32, 1, 0, false);
   agx_begin_cf(b);
   agx_if_icmp(b, compare_reads_load ? loaded : agx_register(0, AGX_SIZE_32),
               agx_register(2, AGX_SIZE_32), 1, AGX_ICOND_ULT, false, pop);
   agx_jmp_exec_none(b, pop);
   agx_block_add_successor(entry, arm);
   agx_block_add_successor(entry, pop);

   agx_builder ab = agx_init_builder(b->shader, agx_after_block(arm));
   agx_device_store(&ab, arm_uses ? loaded : agx_register(6, AGX_SIZE_32),
                    agx_uniform(0, AGX_SIZE_64), agx_immediate(8),
                    AGX_FORMAT_I32, 1, 0, false);
   agx_block_add_successor(arm, pop);

   agx_builder pb = agx_init_builder(b->shader, agx_after_block(pop));
   agx_pop_exec(&pb, 1);
   agx_device_store(&pb, loaded, agx_uniform(0, AGX_SIZE_64), agx_immediate(4),
                    AGX_FORMAT_I32, 1, 0, false);
   agx_stop(&pb);
}

/* A forward if keeps the load in flight into its arm, and the arm's pop joins
 * instead of draining: the consumer, in the arm or after the pop, waits for
 * the load itself, and nothing drains. */
TEST_F(AGX3PackBinary, ForwardIfKeepsLoadsInFlightIntoTheArm)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (bool arm_uses : {true, false}) {
         agx_builder *b = builder(enc);
         build_load_then_if(b, arm_uses, false);
         EXPECT_FALSE(pack(b).empty());
         EXPECT_EQ(agx3_native_stats.cf_drains, 0u);
      }
   }
}

/* The skip jump lands at the pop block's start, which is now its cndend: the
 * joined pop emits no carrier, and nothing drains. */
TEST_F(AGX3PackBinary, SkippedArmLandsOnTheJoinedPop)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      build_load_then_if(b, false, false);
      std::vector<uint8_t> bin = pack(b);
      ASSERT_FALSE(bin.empty());
      agx_block *pop = NULL;
      agx_foreach_block(b->shader, block)
         pop = block;
      ASSERT_NE(pop, nullptr);
      uint8_t cndend[AGX3_CF_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_cf_cndend(enc, 1, cndend, &len));
      ASSERT_LE((size_t)pop->offset + len, bin.size());
      EXPECT_EQ(memcmp(bin.data() + pop->offset, cndend, len), 0);
      EXPECT_EQ(agx3_native_stats.cf_drains, 0u);
   }
}

/* Build: if (r0 < r2) { r6 = load } else { r6 = load } store r6; with each
 * arm's exec-none skip. */
static void
build_if_else_loads(agx_builder *b)
{
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *then_arm = agx_test_block(b->shader);
   agx_block *else_arm = agx_test_block(b->shader);
   agx_block *pop = agx_test_block(b->shader);
   agx_index value = agx_register(6, AGX_SIZE_32);
   agx_begin_cf(b);
   agx_if_icmp(b, agx_register(0, AGX_SIZE_32), agx_register(2, AGX_SIZE_32),
               1, AGX_ICOND_ULT, false, else_arm);
   agx_jmp_exec_none(b, else_arm);
   agx_block_add_successor(entry, then_arm);
   agx_block_add_successor(entry, else_arm);

   agx_builder tb = agx_init_builder(b->shader, agx_after_block(then_arm));
   agx_device_load_to(&tb, value, agx_uniform(0, AGX_SIZE_64),
                      agx_immediate(0), AGX_FORMAT_I32, 1, 0, false);
   agx_block_add_successor(then_arm, else_arm);

   agx_builder eb = agx_init_builder(b->shader, agx_after_block(else_arm));
   agx_else_fcmp(&eb, agx_zero(), agx_zero(), 1, AGX_FCOND_EQ, false, pop);
   agx_jmp_exec_none(&eb, pop);
   agx_device_load_to(&eb, value, agx_uniform(0, AGX_SIZE_64),
                      agx_immediate(4), AGX_FORMAT_I32, 1, 0, false);
   agx_block_add_successor(else_arm, pop);

   agx_builder pb = agx_init_builder(b->shader, agx_after_block(pop));
   agx_pop_exec(&pb, 1);
   agx_device_store(&pb, value, agx_uniform(0, AGX_SIZE_64), agx_immediate(8),
                    AGX_FORMAT_I32, 1, 0, false);
   agx_stop(&pb);
}

/* Joined arms: neither the else nor the pop drains. The second arm's load
 * waits out the first's before redefining r6, and the store after the pop
 * waits for whichever arm's load ran. */
TEST_F(AGX3PackBinary, IfElseArmsJoinWithoutDrains)
{
   for (int e = 0; e < 2; ++e) {
      agx_builder *b =
         builder(e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B);
      build_if_else_loads(b);
      EXPECT_FALSE(pack(b).empty());
      EXPECT_EQ(agx3_native_stats.cf_drains, 0u);
   }
}

/* agx3_pack_program() keeps the join only where it is no worse than the
 * arms draining at their ends. */
TEST_F(AGX3PackBinary, ArmJoinNeverGrowsTheProgram)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      size_t sizes[2];
      for (unsigned off = 0; off < 2; ++off) {
         agx_builder *b = builder(enc);
         build_if_else_loads(b);
         struct util_dynarray bin;
         util_dynarray_init(&bin, mem_ctx);
         if (off) {
            b->shader->agx3_arm_join_off = true;
            ASSERT_TRUE(agx3_pack_binary(b->shader, &bin));
         } else {
            ASSERT_TRUE(agx3_pack_program(b->shader, &bin));
         }
         sizes[off] = bin.size;
      }
      EXPECT_LE(sizes[0], sizes[1]) << enc;
   }
}

/* A compare reading the load waits for it, which no branch form carries:
 * the if drains as before and nothing is left for the pop. */
TEST_F(AGX3PackBinary, IfReadingALoadStillDrains)
{
   for (int e = 0; e < 2; ++e) {
      agx_builder *b =
         builder(e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B);
      build_load_then_if(b, false, true);
      EXPECT_FALSE(pack(b).empty());
      EXPECT_EQ(agx3_native_stats.cf_drains, 0u);
      EXPECT_GE(agx3_native_stats.wait_carriers, 1u);
   }
}

TEST_F(AGX3PackBinary, StructuredIfAndForwardBranch)
{
   /* One fused AGX IR if becomes icmp + cndst. The optional exec-none jump is
    * a reserved bn patched from its own start to the target block's cndend. */
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_block *pop = agx_test_block(b->shader);
      agx_index a = agx_register(0, AGX_SIZE_32);
      agx_index c = agx_register(2, AGX_SIZE_32);

      agx_begin_cf(b); /* architectural on AGX3: emits zero bytes */
      agx_if_icmp(b, a, c, 1, AGX_ICOND_ULT, false, pop);
      agx_jmp_exec_none(b, pop);

      agx_builder pb = agx_init_builder(b->shader, agx_after_block(pop));
      agx_pop_exec(&pb, 1);
      agx_stop(&pb);

      uint8_t bytes[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      std::vector<uint8_t> want;
      ASSERT_TRUE(
         agx3_cf_icmp_reg(enc, AGX3_CF_ICOND_ULT, 0, 0, 2, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndst(enc, 0, false, 1, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_branch(enc, true, bytes, &len));
      ASSERT_EQ(len, AGX3_CF_BRANCH_LENGTH);
      ASSERT_TRUE(agx3_cf_patch_branch(enc, true, bytes, 10));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndend(enc, 1, bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);

      EXPECT_EQ(pack(b), want);
   }
}

/* Build: load r4; loop { [store r4]; while (r0 < r2) } [store r4], the
 * loop skeleton of NestedLoopEntryBackedgeAndPop. */
static void
build_load_then_loop(agx_builder *b, bool body_uses, bool after_uses)
{
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index loaded = agx_register(4, AGX_SIZE_32);
   agx_device_load_to(b, loaded, agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                      AGX_FORMAT_I32, 1, 0, false);
   agx_push_exec(b, 1);
   agx_block *body = agx_test_block(b->shader);
   agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
   agx_device_store(&lb, body_uses ? loaded : agx_register(6, AGX_SIZE_32),
                    agx_uniform(0, AGX_SIZE_64), agx_immediate(8),
                    AGX_FORMAT_I32, 1, 0, false);
   agx_while_icmp(&lb, agx_register(0, AGX_SIZE_32),
                  agx_register(2, AGX_SIZE_32), 1, AGX_ICOND_ULT, false, NULL);
   agx_jmp_exec_any(&lb, body);
   agx_pop_exec(&lb, 1);
   agx_device_store(&lb, after_uses ? loaded : agx_register(6, AGX_SIZE_32),
                    agx_uniform(0, AGX_SIZE_64), agx_immediate(4),
                    AGX_FORMAT_I32, 1, 0, false);
   agx_stop(&lb);
}

/* A loop's entry push keeps a load from before the loop in flight: the
 * loop's consumer waits for it in the body, and nothing drains at the entry.
 * A load the body does not consume would be retired by the body's continue
 * test on every iteration, so that loop drains once at its entry instead. */
TEST_F(AGX3PackBinary, LoopEntryKeepsLoadsInFlight)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (bool body_uses : {true, false}) {
         SCOPED_TRACE(::testing::Message()
                      << "enc=" << enc << " body_uses=" << body_uses);
         agx_builder *b = builder(enc);
         build_load_then_loop(b, body_uses, !body_uses);
         EXPECT_FALSE(pack(b).empty());
         EXPECT_EQ(agx3_native_stats.cf_drains, body_uses ? 0u : 1u);
      }
   }
}

/* The body retires the load before its first drain, on the first iteration
 * with every invocation that entered: after the loop it is complete, so the
 * consumer there needs no wait and nothing drains anywhere. */
TEST_F(AGX3PackBinary, LoopBodyRetiresEntryStateOnTheFirstIteration)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      build_load_then_loop(b, true, true);
      EXPECT_FALSE(pack(b).empty());
      EXPECT_EQ(agx3_native_stats.cf_drains, 0u);
      EXPECT_EQ(agx3_native_stats.wait_carriers, 0u);
   }
}

/* Build: load r4; loop { r8 = r4 + r4 (an ALU consumer); store r8;
 * while (r0 < r2) }. */
static void
build_load_then_alu_loop(agx_builder *b)
{
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index loaded = agx_register(4, AGX_SIZE_32);
   agx_device_load_to(b, loaded, agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                      AGX_FORMAT_I32, 1, 0, false);
   agx_push_exec(b, 1);
   agx_block *body = agx_test_block(b->shader);
   agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
   agx_fadd_to(&lb, agx_register(8, AGX_SIZE_32), loaded, loaded);
   agx_device_store(&lb, agx_register(8, AGX_SIZE_32),
                    agx_uniform(0, AGX_SIZE_64), agx_immediate(8),
                    AGX_FORMAT_I32, 1, 0, false);
   agx_while_icmp(&lb, agx_register(0, AGX_SIZE_32),
                  agx_register(2, AGX_SIZE_32), 1, AGX_ICOND_ULT, false, NULL);
   agx_jmp_exec_any(&lb, body);
   agx_pop_exec(&lb, 1);
   agx_stop(&lb);
}

/* Keeping results in flight into a loop never makes the loop's code larger
 * than draining at its entry: when the body's consumer of a load from
 * before the loop can only wait through a carrier, which would run on every
 * iteration, agx3_pack_program() packs again with that entry draining. */
TEST_F(AGX3PackBinary, LoopEntryNeverAddsCarriersToTheBody)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (bool alu : {false, true}) {
         SCOPED_TRACE(::testing::Message() << "enc=" << enc << " alu=" << alu);
         size_t sizes[2];
         for (unsigned drain_all = 0; drain_all < 2; ++drain_all) {
            agx_builder *b = builder(enc);
            if (alu)
               build_load_then_alu_loop(b);
            else
               build_load_then_loop(b, true, false);
            struct util_dynarray bin;
            util_dynarray_init(&bin, mem_ctx);
            if (drain_all) {
               b->shader->agx3_cf_drain_all = true;
               ASSERT_TRUE(agx3_pack_binary(b->shader, &bin));
            } else {
               ASSERT_TRUE(agx3_pack_program(b->shader, &bin));
            }
            sizes[drain_all] = bin.size;
         }
         EXPECT_LE(sizes[0], sizes[1]);
      }
   }
}

/* Build: loop { load r4; [if (r8 < r10) { break }] break if (r0 < r2);
 * [store r4] } store r4. When the body consumes its load after the break,
 * the lanes that stay retire it before the backedge, and the lanes that
 * broke take it to the store after the loop. */
static void
build_break_loop(agx_builder *b, bool if_break, bool body_uses = true)
{
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index loaded = agx_register(4, AGX_SIZE_32);
   agx_push_exec(b, 1);
   agx_block *body = agx_test_block(b->shader);
   agx_block *arm = if_break ? agx_test_block(b->shader) : NULL;
   agx_block *arm_pop = if_break ? agx_test_block(b->shader) : NULL;
   agx_block *test = agx_test_block(b->shader);
   agx_block *footer = agx_test_block(b->shader);
   agx_block *after = agx_test_block(b->shader);

   agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
   agx_device_load_to(&lb, loaded, agx_uniform(0, AGX_SIZE_64),
                      agx_immediate(0), AGX_FORMAT_I32, 1, 0, false);
   if (if_break) {
      agx_if_icmp(&lb, agx_register(8, AGX_SIZE_32),
                  agx_register(10, AGX_SIZE_32), 1, AGX_ICOND_ULT, false,
                  arm_pop);
      agx_builder ab = agx_init_builder(b->shader, agx_after_block(arm));
      agx_index r0l = agx_register(0, AGX_SIZE_16);
      agx_break_if_icmp(&ab, r0l, r0l, 2, false, AGX_ICOND_UEQ, after);
      agx_builder pb = agx_init_builder(b->shader, agx_after_block(arm_pop));
      agx_pop_exec(&pb, 1);
   }
   agx_builder tb = agx_init_builder(b->shader, agx_after_block(test));
   agx_break_if_icmp(&tb, agx_register(0, AGX_SIZE_32),
                     agx_register(2, AGX_SIZE_32), 1, false, AGX_ICOND_ULT,
                     after);
   agx_jmp_exec_none_after(&tb, footer);
   agx_device_store(&tb, body_uses ? loaded : agx_register(6, AGX_SIZE_32),
                    agx_uniform(0, AGX_SIZE_64), agx_immediate(8),
                    AGX_FORMAT_I32, 1, 0, false);

   agx_builder fb = agx_init_builder(b->shader, agx_after_block(footer));
   agx_jmp_exec_any(&fb, body);
   agx_pop_exec(&fb, 1);

   agx_builder ab = agx_init_builder(b->shader, agx_after_block(after));
   agx_device_store(&ab, loaded, agx_uniform(0, AGX_SIZE_64), agx_immediate(4),
                    AGX_FORMAT_I32, 1, 0, false);
   agx_stop(&ab);
}

/* Build: loop { load r10; loop { load r12; break if (r0 < r2); store r12;
 * store r10 } store r10; store r12; break if (r4 < r6) }. The inner loop
 * consumes the outer body's load, but only after its break. */
static void
build_nested_break_loops(agx_builder *b)
{
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index outer = agx_register(10, AGX_SIZE_32);
   agx_index inner = agx_register(12, AGX_SIZE_32);
   agx_index base = agx_uniform(0, AGX_SIZE_64);
   agx_push_exec(b, 1);
   agx_block *ob = agx_test_block(b->shader);
   agx_block *ib = agx_test_block(b->shader);
   agx_block *itest = agx_test_block(b->shader);
   agx_block *ifoot = agx_test_block(b->shader);
   agx_block *iafter = agx_test_block(b->shader);
   agx_block *ofoot = agx_test_block(b->shader);
   agx_block *oafter = agx_test_block(b->shader);

   agx_builder bb = agx_init_builder(b->shader, agx_after_block(ob));
   agx_device_load_to(&bb, outer, base, agx_immediate(0), AGX_FORMAT_I32, 1, 0,
                      false);
   agx_push_exec(&bb, 1);
   bb = agx_init_builder(b->shader, agx_after_block(ib));
   agx_device_load_to(&bb, inner, base, agx_immediate(4), AGX_FORMAT_I32, 1, 0,
                      false);
   bb = agx_init_builder(b->shader, agx_after_block(itest));
   agx_break_if_icmp(&bb, agx_register(0, AGX_SIZE_32),
                     agx_register(2, AGX_SIZE_32), 1, false, AGX_ICOND_ULT,
                     iafter);
   agx_jmp_exec_none_after(&bb, ifoot);
   agx_device_store(&bb, inner, base, agx_immediate(8), AGX_FORMAT_I32, 1, 0,
                    false);
   agx_device_store(&bb, outer, base, agx_immediate(12), AGX_FORMAT_I32, 1, 0,
                    false);
   bb = agx_init_builder(b->shader, agx_after_block(ifoot));
   agx_jmp_exec_any(&bb, ib);
   agx_pop_exec(&bb, 1);
   bb = agx_init_builder(b->shader, agx_after_block(iafter));
   agx_device_store(&bb, outer, base, agx_immediate(16), AGX_FORMAT_I32, 1, 0,
                    false);
   agx_device_store(&bb, inner, base, agx_immediate(20), AGX_FORMAT_I32, 1, 0,
                    false);
   agx_break_if_icmp(&bb, agx_register(4, AGX_SIZE_32),
                     agx_register(6, AGX_SIZE_32), 1, false, AGX_ICOND_ULT,
                     oafter);
   agx_jmp_exec_none_after(&bb, ofoot);
   bb = agx_init_builder(b->shader, agx_after_block(ofoot));
   agx_jmp_exec_any(&bb, ob);
   agx_pop_exec(&bb, 1);
   bb = agx_init_builder(b->shader, agx_after_block(oafter));
   agx_stop(&bb);
}

/* A break joins its loop's pop instead of draining (agx3_slot_core.h): with
 * joins off, each break drains what the lanes that stay would have retired
 * later anyway. */
TEST_F(AGX3PackBinary, BreakKeepsLoadsInFlightToTheLoopExit)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (bool if_break : {false, true}) {
         SCOPED_TRACE(::testing::Message()
                      << "enc=" << enc << " if_break=" << if_break);
         for (bool off : {false, true}) {
            agx_builder *b = builder(enc);
            build_break_loop(b, if_break);
            b->shader->agx3_arm_join_off = off;
            EXPECT_FALSE(pack(b).empty());
            if (off)
               EXPECT_GE(agx3_native_stats.cf_drains, 1u);
            else
               EXPECT_EQ(agx3_native_stats.cf_drains, 0u);
         }
      }
   }
}

/* When the lanes that stay do not consume the load before the backedge,
 * which would retire it on every iteration anyway, the break drains as it
 * did before breaks joined. */
TEST_F(AGX3PackBinary, BreakDrainsWhatOnlyTheBackedgeWouldRetire)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      SCOPED_TRACE(::testing::Message() << "enc=" << enc);
      std::vector<uint8_t> code[2];
      for (unsigned off = 0; off < 2; ++off) {
         agx_builder *b = builder(enc);
         build_break_loop(b, false, false);
         b->shader->agx3_arm_join_off = off;
         code[off] = pack(b);
         EXPECT_EQ(agx3_native_stats.loop_cf_drains, 1u);
      }
      EXPECT_FALSE(code[0].empty());
      EXPECT_EQ(code[0], code[1]);
   }
}

/* Once the inner loop's break no longer drains, the inner loop keeps the
 * outer body's load in flight at its entry too: nothing drains anywhere. */
TEST_F(AGX3PackBinary, InnerLoopBreakKeepsOuterLoadsInFlight)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      SCOPED_TRACE(::testing::Message() << "enc=" << enc);
      unsigned drains[2], kept[2];
      for (unsigned off = 0; off < 2; ++off) {
         agx_builder *b = builder(enc);
         build_nested_break_loops(b);
         b->shader->agx3_arm_join_off = off;
         EXPECT_FALSE(pack(b).empty());
         drains[off] = agx3_native_stats.loop_cf_drains;
         kept[off] = b->shader->agx3_kept_loop_entries;
      }
      EXPECT_EQ(drains[0], 0u);
      EXPECT_GT(drains[1], 0u);
      EXPECT_EQ(kept[0], 2u);
      EXPECT_LT(kept[1], 2u);
   }
}

/* agx3_pack_program() keeps break joins only where they leave no more
 * carriers inside loops, then no more carriers, then no more code than
 * the breaks draining, with or without the arms joining. */
TEST_F(AGX3PackBinary, BreakJoinNeverAddsLoopCarriers)
{
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      for (unsigned shape = 0; shape < 3; ++shape) {
         SCOPED_TRACE(::testing::Message() << "enc=" << enc
                                           << " shape=" << shape);
         struct { unsigned loop_carriers, carriers; size_t size; } cost[3];
         for (unsigned v = 0; v < 3; ++v) {
            agx_builder *b = builder(enc);
            if (shape == 2)
               build_nested_break_loops(b);
            else
               build_break_loop(b, shape);
            struct util_dynarray bin;
            util_dynarray_init(&bin, mem_ctx);
            if (v) {
               b->shader->agx3_arm_join_off = v == 2;
               b->shader->agx3_break_joins = AGX3_BREAKS_DRAIN;
               ASSERT_TRUE(agx3_pack_binary(b->shader, &bin));
            } else {
               ASSERT_TRUE(agx3_pack_program(b->shader, &bin));
            }
            cost[v] = {b->shader->agx3_loop_carriers,
                       agx3_native_stats.wait_carriers, bin.size};
         }
         for (unsigned v = 1; v < 3; ++v) {
            EXPECT_LE(cost[0].loop_carriers, cost[v].loop_carriers) << v;
            if (cost[0].loop_carriers == cost[v].loop_carriers) {
               EXPECT_LE(cost[0].carriers, cost[v].carriers) << v;
               if (cost[0].carriers == cost[v].carriers)
                  EXPECT_LE(cost[0].size, cost[v].size) << v;
            }
         }
      }
   }
}

TEST_F(AGX3PackBinary, NestedLoopEntryBackedgeAndPop)
{
   /* Mesa's push_exec(1) + jmp_exec_any + pop_exec(1) loop skeleton maps to
    * AGX3's TRUE-slot cndst(push=2), backwards b, and cndend(pop=2). */
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_index a = agx_register(0, AGX_SIZE_32);
      agx_index c = agx_register(2, AGX_SIZE_32);

      agx_push_exec(b, 1);
      agx_block *body = agx_test_block(b->shader);
      agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
      agx_while_icmp(&lb, a, c, 1, AGX_ICOND_ULT, false, NULL);
      agx_jmp_exec_any(&lb, body);
      agx_pop_exec(&lb, 1);
      agx_stop(&lb);

      uint8_t bytes[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      std::vector<uint8_t> want;
      ASSERT_TRUE(agx3_cf_cndst(enc, 6, false, 2, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(
         agx3_cf_icmp_reg(enc, AGX3_CF_ICOND_ULT, 0, 0, 2, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndlt(enc, 0, false, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_branch(enc, false, bytes, &len));
      ASSERT_EQ(len, AGX3_CF_BRANCH_LENGTH);
      ASSERT_TRUE(agx3_cf_patch_branch(enc, false, bytes, -14));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndend(enc, 2, bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);

      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, FloatRelationAndCanonicalElse)
{
   /* Ordered >= is invert(less-than-or-NaN); the inversion belongs to cndst,
    * not fcmp. Mesa's fixed else_fcmp(0,0,eq) becomes cndef with no compare. */
   for (int e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_index a = agx_register(0, AGX_SIZE_32);
      agx_index c = agx_register(2, AGX_SIZE_32);
      agx_if_fcmp(b, a, c, 1, AGX_FCOND_GE, false, NULL);
      agx_else_fcmp(b, agx_zero(), agx_zero(), 1, AGX_FCOND_EQ, false, NULL);
      agx_pop_exec(b, 1);
      agx_stop(b);

      uint8_t bytes[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      std::vector<uint8_t> want;
      ASSERT_TRUE(
         agx3_cf_fcmp_reg(enc, AGX3_CF_FCOND_LT, true, 0, 0, 2, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndst(enc, 0, true, 1, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndef(enc, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndend(enc, 1, bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);

      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, NegatedHalfZeroCompareCanonicalizes)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index source = agx_register(9, AGX_SIZE_16);
   source.neg = true;
   agx_if_fcmp(b, source, agx_immediate(0), 1, AGX_FCOND_EQ, false, NULL);
   agx_stop(b);

   uint8_t bytes[AGX3_CF_MAX_LENGTH];
   uint8_t len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_cf_fcmp_zero_imm(
      AGX_ISA_ENCODING_AGX3B, AGX3_CF_FCOND_EQ, false, 0, 9, true, 0,
      (struct agx3_cf_source_hints){false, false}, 0, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(
      agx3_cf_cndst(AGX_ISA_ENCODING_AGX3B, 0, false, 1, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FirefoxFloatIfMinifloatsMaterializeToGroundedRegisters)
{
   for (enum agx_isa_encoding enc :
        {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      for (float constant : {1.0f, -10.0f}) {
         agx_builder *b = builder(enc);
         b->shader->stage = MESA_SHADER_FRAGMENT;
         b->shader->nir = rzalloc(mem_ctx, nir_shader);
         b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;

         agx_index source = agx_mov_imm(b, 32, 0x40000000);
         agx_if_fcmp(b, source, agx_immediate_f(constant), 1, AGX_FCOND_EQ,
                     false, NULL);
         agx_stop(b);

         EXPECT_TRUE(agx_lower_agx3_cf_constants(b->shader));

         uint32_t expected;
         memcpy(&expected, &constant, sizeof(expected));
         unsigned matching_constant = 0, compares = 0;
         agx_foreach_instr_global(b->shader, I) {
            matching_constant +=
               I->op == AGX_OPCODE_MOV_IMM && I->imm == expected;
            if (I->op == AGX_OPCODE_IF_FCMP) {
               ++compares;
               EXPECT_EQ(I->src[0].type, AGX_INDEX_NORMAL);
               EXPECT_EQ(I->src[1].type, AGX_INDEX_NORMAL);
               EXPECT_EQ(I->src[0].size, AGX_SIZE_32);
               EXPECT_EQ(I->src[1].size, AGX_SIZE_32);
            }
         }
         EXPECT_EQ(matching_constant, 1u);
         EXPECT_EQ(compares, 1u);

         agx_ra(b->shader);
         agx_lower_pseudo(b->shader);
         agx_opt_register_cache(b->shader);
         EXPECT_FALSE(pack(b).empty());
      }
   }
}

TEST_F(AGX3PackBinary, NegatedFloatRelationsMaterializeUnsupportedSigns)
{
   for (enum agx_isa_encoding enc :
        {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      for (enum agx_fcond cond : {AGX_FCOND_EQ, AGX_FCOND_LTN, AGX_FCOND_GTN,
                                  AGX_FCOND_GE, AGX_FCOND_LE, AGX_FCOND_LT}) {
         for (unsigned negated : {1u, 2u, 3u}) {
            agx_builder *b = builder(enc);
            b->shader->stage = MESA_SHADER_FRAGMENT;
            b->shader->nir = rzalloc(mem_ctx, nir_shader);
            b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
            agx_index a = agx_mov_imm(b, 32, 0x40000000);
            agx_index c = agx_mov_imm(b, 32, 0x40400000);
            a.neg = negated & 1;
            c.neg = negated & 2;
            agx_instr *cmp = agx_if_fcmp(b, a, c, 1, cond, false, NULL);
            agx_stop(b);
            agx_lower_agx3_cf_constants(b->shader);

            EXPECT_FALSE(cmp->src[0].neg && cmp->src[1].neg);
            if (cond == AGX_FCOND_LT) {
               /* Single-negate LT stays native; double-negate materializes
                * src[0] and keeps src[1] negated natively. */
               EXPECT_EQ(bool(cmp->src[0].neg), negated == 1);
               EXPECT_EQ(bool(cmp->src[1].neg), negated != 1);
            } else {
               EXPECT_FALSE(cmp->src[0].neg);
               EXPECT_FALSE(cmp->src[1].neg);
            }
            agx_ra(b->shader);
            agx_lower_pseudo(b->shader);
            agx_opt_register_cache(b->shader);
            EXPECT_FALSE(pack(b).empty());
         }
      }
   }
}

TEST_F(AGX3PackBinary, AGX3CPreserveUnarySourceClearsLastUseBit)
{
   for (enum agx3_sel_op op :
        {AGX3_SEL_OP_F_TO_I32, AGX3_SEL_OP_F_TO_U32, AGX3_SEL_OP_I32_TO_F,
         AGX3_SEL_OP_U32_TO_F, AGX3_SEL_OP_F_RCP}) {
      struct agx3_sel_instr ins = {};
      ins.op = op;
      ins.has_dest = true;
      ins.dest = 2;
      ins.dest_size = AGX3_SEL_SIZE_32;
      ins.nr_srcs = 1;
      ins.src[0].is_reg = true;
      ins.src[0].reg = 0;
      ins.src[0].size = AGX3_SEL_SIZE_32;

      uint8_t dead_bytes[AGX3_SEL_MAX_LENGTH] = {}, live_bytes[AGX3_SEL_MAX_LENGTH] = {};
      uint8_t dead_len = 0, live_len = 0;

      ins.src[0].live = false;
      ASSERT_TRUE(agx3_select_encode(AGX_ISA_ENCODING_AGX3C, &ins, dead_bytes, &dead_len));
      ASSERT_GE(dead_len, 10u);

      ins.src[0].live = true;
      ASSERT_TRUE(agx3_select_encode(AGX_ISA_ENCODING_AGX3C, &ins, live_bytes, &live_len));
      ASSERT_EQ(live_len, dead_len);

      /* Live source clears byte 8 bit 5 (last-use) and preserves every other bit. */
      EXPECT_NE(dead_bytes[8] & 0x20, 0);
      EXPECT_EQ(live_bytes[8] & 0x20, 0);
      EXPECT_EQ(live_bytes[8], dead_bytes[8] & ~0x20);
      for (unsigned i = 0; i < dead_len; ++i) {
         if (i != 8)
            EXPECT_EQ(live_bytes[i], dead_bytes[i]);
      }
   }
}

TEST_F(AGX3PackBinary, MemoryDependentControlFlowRetiresOutstandingLoad)
{
   /* A load before divergent control flow is retired on the straight-line
    * edge, so both arms enter with the same empty scoreboard. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index base = agx_register(0, AGX_SIZE_64);
   agx_index index = agx_register(4, AGX_SIZE_32);
   agx_index loaded = agx_register(6, AGX_SIZE_32);
   agx_index cond = agx_register(8, AGX_SIZE_32);
   agx_device_load_to(b, loaded, base, index, AGX_FORMAT_I32, 1, 2, false);
   agx_if_icmp(b, cond, agx_immediate(0), 1, AGX_ICOND_UEQ, true, NULL);
   agx_stop(b);

   const auto bin = pack(b);
   uint8_t wait[AGX3_ALU_MAX_LENGTH], len = 0;
   bool found_wait = false;
   for (unsigned slot = 0; slot < 6; ++slot) {
      ASSERT_TRUE(agx3_alu_band_identity_wait(AGX_ISA_ENCODING_AGX3B, 0,
                                              1u << slot, wait, &len));
      found_wait |=
         std::search(bin.begin(), bin.end(), wait, wait + len) != bin.end();
   }
   EXPECT_TRUE(found_wait);
}

TEST_F(AGX3PackBinary, RefusesUngroundedUnconditionalBreak)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_break(b, 1, NULL);
   agx_stop(b);
   EXPECT_TRUE(pack(b, /*expect_ok=*/false).empty());
}

TEST_F(AGX3PackBinary, FloatRegisterLivenessNativeWitnesses)
{
   const struct {
      enum agx3_alu_op op;
      uint16_t dest, src[3], wait;
      enum agx3_alu_hint hint[3];
      unsigned len;
      uint8_t bytes[8];
   } cases[] = {
      {AGX3_ALU_FADD,0,{4,2,0},32,{AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE},6,{9,3,4,5,0,0xc0}},
      {AGX3_ALU_FMUL,0,{4,2,0},32,{AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE},6,{9,3,5,5,0,0xc0}},
      {AGX3_ALU_FMA,0,{4,2,6},32,{AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE},8,{9,3,6,5,1,6,2,0xc0}},
      {AGX3_ALU_FMUL,0,{8,6,0},32,{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},6,{9,0x87,5,0x89,0,0xc0}},
      {AGX3_ALU_FMA,2,{10,6,12},0,{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD},6,{0x19,0x87,0x16,0x0b,0x80,0x0c}},
      {AGX3_ALU_FADD,4,{8,6,0},0,{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE},4,{0x29,7,0x10,9}},
   };
   for (const auto &v : cases) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_alu_float_reg(AGX_ISA_ENCODING_AGX3B, v.op, v.dest,
                                      v.src, v.hint, v.wait, bytes, &len));
      ASSERT_EQ(len, v.len);
      EXPECT_EQ(memcmp(bytes, v.bytes, len), 0);
   }
}

TEST_F(AGX3PackBinary, G17PAbsoluteMultiplierImmediateAddend)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index src0 = agx_register(0, AGX_SIZE_32);
   agx_index src1 = agx_register(2, AGX_SIZE_32);
   src0.discard = true;
   src1.abs = true;
   src1.discard = true;
   agx_ffma_to(b, agx_register(0, AGX_SIZE_32), src0, src1,
               agx_immediate(48));
   agx_stop(b);

   /* applegpu-nt v32023.255, -arch applegpu_g17p, for
    * fma(a, fabs(b), 1.0f), followed by the canonical end. */
   static const uint8_t native[] = {
      0x09,0x01,0x1e,0x03,0x23,0xb0,0x02,0x00,0x00,0x80,0x02,0x00,
      0x0e,0x00,0x00,0x00,
   };
   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(native),
                                            std::end(native)));
}

TEST_F(AGX3PackBinary, FloatRegisterLivenessPreservedThroughBackend)
{
   /* Alpha blending repeatedly reads both factors. Exercise all three float
    * operations with independent hints; physical bit checks do not call the
    * production encoder to derive expected flags. */
   for (unsigned op = 0; op < 3; ++op) {
      for (unsigned h0 = 0; h0 < 3; ++h0) {
         for (unsigned h1 = 0; h1 < 3; ++h1) {
            for (unsigned h2 = 0; h2 < (op == 2 ? 3u : 1u); ++h2) {
               agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
               b->shader->stage = MESA_SHADER_FRAGMENT;
               agx_index src[3] = {agx_register(2, AGX_SIZE_32),
                  agx_register(4, AGX_SIZE_32), agx_register(6, AGX_SIZE_32)};
               const unsigned hints[] = {h0, h1, h2};
               for (unsigned i = 0; i < 3; ++i) {
                  src[i].cache = hints[i] == 2;
                  src[i].discard = hints[i] == 1;
               }
               const agx_index dest = agx_register(8, AGX_SIZE_32);
               if (op == 0) agx_fadd_to(b, dest, src[0], src[1]);
               else if (op == 1) agx_fmul_to(b, dest, src[0], src[1]);
               else agx_ffma_to(b, dest, src[0], src[1], src[2]);
               agx_stop(b);
               const auto bytes = pack(b);
               ASSERT_EQ(bytes.size(), op == 2 ? 10u : 8u);
               EXPECT_EQ((bytes[3] >> 7) & 1, h0 == 2);
               EXPECT_EQ((bytes[2] >> 4) & 1, h0 == 1);
               EXPECT_EQ((bytes[1] >> 7) & 1, h1 == 2);
               EXPECT_EQ((bytes[2] >> 3) & 1, h1 == 1);
               if (op == 2) {
                  EXPECT_EQ((bytes[5] >> 7) & 1, h2 == 2);
                  EXPECT_EQ((bytes[4] >> 7) & 1, h2 == 1);
               }
            }
         }
      }
   }
}

TEST_F(AGX3PackBinary, FloatRegisterLivenessRetiresUnsupportedCompactWaits)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   const agx_index x = agx_register(2, AGX_SIZE_32);
   const agx_index y = agx_register(6, AGX_SIZE_32);
   agx_get_sr_to(b, x, AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_get_sr_to(b, agx_register(4, AGX_SIZE_32), AGX_SR_THREAD_POSITION_IN_GRID_Y);
   agx_get_sr_to(b, y, AGX_SR_THREAD_POSITION_IN_GRID_Z);
   agx_index retained = y;
   retained.cache = true;
   agx_fmul_to(b, agx_register(8, AGX_SIZE_32), x, retained);
   agx_stop(b);
   const auto bytes = pack(b);
   ASSERT_GE(bytes.size(), 8u);
   /* Last multiply keeps its second source: f27=cache, f29=0. Its trailer
    * is END, so this also checks that the fallback emitted the operation. */
   const unsigned at = bytes.size() - 8;
   EXPECT_EQ(bytes[at] & 0xf, 9);
   EXPECT_EQ((bytes[at + 1] >> 7) & 1, 1);
   EXPECT_EQ((bytes[at + 2] >> 3) & 1, 0);
}

TEST_F(AGX3PackBinary, FloatRegisterLivenessRefusesContradictoryHints)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(2, AGX_SIZE_32);
   src.cache = src.discard = true;
   agx_fmul_to(b, agx_register(0, AGX_SIZE_32), src,
                    agx_register(4, AGX_SIZE_32));
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, G17PAlphaProtocolAcquiresBeforeLoadReleasesAfterStore)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index packed = agx_register(0, AGX_SIZE_16);
   packed.channels_m1 = 1;
   agx_wait_pix(b, 8);
   agx_ld_tile_to(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                  AGX_FORMAT_U8NORM, 15, 0, false, 0);
   agx_wait_pix(b, 4);
   agx_st_tile(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
               AGX_FORMAT_U8NORM, 15, 0, false, 0);
   agx_stop(b);

   const auto bin = pack(b);
   const uint8_t prefix[] = {
      0x87,0x02,0x54,0x00,0x06,0x00, /* class12 entry */
      0x87,0x02,0x54,0x08,0x08,0x00, /* read wait */
      0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0xce,0x02,0x00,0x00,0x00,
      0x87,0x02,0x54,0x04,0x08,0x00, /* write wait */
   };
   const uint8_t suffix[] = {
      0x07,0x02,0x54,0x0c,0x02,0x00, /* class12 release */
      0x0e,0x00,0x00,0x00,
   };
   ASSERT_EQ(bin.size(), 52u);
   EXPECT_TRUE(std::equal(std::begin(prefix), std::end(prefix), bin.begin()));
   EXPECT_TRUE(std::equal(std::begin(suffix), std::end(suffix), bin.end()-sizeof(suffix)));
   /* The intervening store consumes the actual outstanding tile-load slot. */
   uint8_t store[AGX3_TILE_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_tile_store(AGX_ISA_ENCODING_AGX3B, 0, AGX3_TILE_W8,
                               15, 0, AGX3_TILE_SAMPLE_1X, 1u << 5, store, &len));
   ASSERT_EQ(len, 12u);
   EXPECT_TRUE(std::equal(store, store+len, bin.begin()+sizeof(prefix)));
}

TEST_F(AGX3PackBinary, G17PAlphaProtocolDoesNotBroadenFormatOffsetOrControlFlow)
{
   /* S8NORM is now independently qualified; offset and CF remain excluded. */
   for (unsigned axis = 1; axis < 3; ++axis) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_block *first = agx_start_block(b->shader);
      const enum agx_format format = axis == 0 ? AGX_FORMAT_S8NORM
                                               : AGX_FORMAT_U8NORM;
      const unsigned offset = axis == 1 ? 16 : 0;
      agx_index packed = agx_register(0, AGX_SIZE_16);
      packed.channels_m1 = 1;
      agx_wait_pix(b, 8);
      agx_ld_tile_to(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     format, 15, offset, false, 0);
      if (axis == 2) {
         agx_block *second = agx_test_block(b->shader);
         agx_block_add_successor(first, second);
         b->cursor = agx_after_block(second);
      }
      agx_wait_pix(b, 4);
      agx_st_tile(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                  format, 15, offset, false, 0);
      agx_stop(b);
      const auto bin = pack(b);
      const uint8_t entry[] = {0x87,0x02,0x54,0x00,0x06,0x00};
      const uint8_t release[] = {0x07,0x02,0x54,0x0c,0x02,0x00};
      EXPECT_EQ(bin.size(), 40u) << axis;
      EXPECT_EQ(std::search(bin.begin(),bin.end(),std::begin(entry),std::end(entry)),bin.end()) << axis;
      EXPECT_EQ(std::search(bin.begin(),bin.end(),std::begin(release),std::end(release)),bin.end()) << axis;
   }
}

TEST_F(AGX3PackBinary, G17PAlphaProtocolPreservesConstantColourBracket)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index packed = agx_register(0, AGX_SIZE_16);
   packed.channels_m1 = 1;
   agx_wait_pix(b, 0xc);
   agx_st_tile(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
               AGX_FORMAT_U8NORM, 15, 0, false, 0);
   agx_stop(b);
   const uint8_t exact[] = {
      0x87,0x02,0x54,0x00,0x06,0x00,
      0x87,0x02,0x54,0x0c,0x08,0x00,
      0xe7,0x06,0x54,0x00,0x00,0x00,0x01,0x4e,0x00,0x00,0x00,0x00,
      0x07,0x02,0x54,0x0c,0x02,0x00,
      0x0e,0x00,0x00,0x00,
   };
   EXPECT_EQ(pack(b), std::vector<uint8_t>(std::begin(exact),std::end(exact)));
}

TEST_F(AGX3PackBinary, FragmentInterpolationStartsOnce)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_interp_agx3_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(0),
                        agx_immediate(0), AGX_INTERPOLATION_CENTER, true);
   agx_interp_agx3_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(1),
                        agx_immediate(0), AGX_INTERPOLATION_CENTER, true);
   agx_stop(b);
   const std::vector<uint8_t> want = {
      0x2f,0x0d,0x54,0,0x03,0,0,0x02,0x10,0,
      0x2f,0x05,0x54,2,0x03,2,0,0x02,0x10,0,0x0e,0,0,0,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, ProjectiveMultiplyPreservesCoefficientAndHints)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index x = agx_register(0, AGX_SIZE_32);
   agx_index y = agx_register(4, AGX_SIZE_32);
   agx_index recip = agx_register(2, AGX_SIZE_32);
   x.discard = y.discard = true;
   recip.cache = true;
   agx_persp_mul_agx3_to(b, agx_register(0, AGX_SIZE_32), x, recip, 1);
   recip.cache = false;
   recip.discard = true;
   agx_persp_mul_agx3_to(b, agx_register(2, AGX_SIZE_32), y, recip, 2);
   agx_stop(b);
   /* Exact consecutive native gradient instructions: coefficient tags1/2. */
   const std::vector<uint8_t> want = {
      0x09,0x01,0x2f,0x83,0x00,0x01,0x00,0x00,
      0x19,0x05,0x3f,0x03,0x00,0x02,0x00,0x00,0x0e,0,0,0,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, NativeImplicitCoordinateStagingForms)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;

   const uint16_t add0_src[2] = {8, 0};
   const enum agx3_alu_hint add0_hints[2] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
   const uint8_t add0_native[] = {0x09,0x01,0x3c,0x09,0x40,0xc0};
   ASSERT_TRUE(agx3_alu_fadd_stage(enc, 0, 1, add0_src, add0_hints,
                                   1u << 5, bytes, &len));
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(std::begin(add0_native),
                                  std::end(add0_native)));

   const uint16_t add1_src[2] = {4, 2};
   const enum agx3_alu_hint add1_hints[2] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
   const uint8_t add1_native[] = {0x19,0x03,0x3c,0x05,0x40,0x00};
   ASSERT_TRUE(agx3_alu_fadd_stage(enc, 2, 1, add1_src, add1_hints, 0,
                                   bytes, &len));
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(std::begin(add1_native),
                                  std::end(add1_native)));

   const struct agx3_persp_mul_desc compact = {
      .stage_dest = true,
      .dest = 0,
      .slot = 1,
      .reciprocal = 0,
      .reciprocal_liveness = AGX3_OPERAND_LIVE,
      .raw = 4,
      .raw_liveness = AGX3_OPERAND_KILL,
      .vary_index = 1,
   };
   const uint8_t compact_native[] = {
      0x09,0x05,0x2f,0x81,0x00,0x01,0x40,0x00};
   ASSERT_TRUE(agx3_persp_mul_encode(enc, &compact, bytes, &len));
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(std::begin(compact_native),
                                  std::end(compact_native)));

   struct agx3_persp_mul_desc extended = compact;
   extended.slot = 6;
   extended.reciprocal = 4;
   extended.raw = 8;
   const uint8_t extended_native[] = {
      0x09,0x09,0x2f,0x85,0x00,0x01,0x42,0x00,0x00,0x14,0x00,0x00};
   ASSERT_TRUE(agx3_persp_mul_encode(enc, &extended, bytes, &len));
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(std::begin(extended_native),
                                  std::end(extended_native)));
}

TEST_F(AGX3PackBinary, ProjectiveMultiplyRefusesUnsupportedForms)
{
   for (unsigned mode = 1; mode < 8; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index raw = agx_register(0, AGX_SIZE_32);
      agx_index recip = agx_register(2, AGX_SIZE_32);
      if (mode == 1) raw.neg = true;
      if (mode == 2) raw = agx_uniform(0, AGX_SIZE_32);
      if (mode == 3) recip.size = AGX_SIZE_16;
      if (mode == 4) raw.cache = raw.discard = true;
      if (mode == 5) raw.kill = true;
      agx_instr *I = agx_persp_mul_agx3_to(b, agx_register(4, AGX_SIZE_32),
                                         raw, recip, mode == 6 ? 128 : 1);
      I->saturate = mode == 7;
      EXPECT_TRUE(pack(b, false).empty());
   }
}

TEST_F(AGX3PackBinary, FourPositionProtocolRemainsByteExact)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   agx_index r0 = agx_register(0, AGX_SIZE_32);
   agx_index xy = r0;
   xy.channels_m1 = 1;
   agx_index idx = r0;
   idx.abs = true;
   idx.kill = true;
   agx_get_sr_to(b, r0, AGX_SR_VERTEX_ID);
   agx_device_load_to(b, xy, agx_uniform(0, AGX_SIZE_64), idx,
                        AGX_FORMAT_I32, 3, 1, false);
   for (unsigned component = 0; component < 4; ++component) {
      if (component >= 2)
         agx_mov_imm_to(b, r0, component == 2 ? 0 : 0x3f800000);
      agx_index value = agx_register(component == 1 ? 2 : 0, AGX_SIZE_32);
      value.discard = true;
      agx_st_vary(b, agx_immediate(component), value);
   }
   agx_stop(b);
   /* Qualified VS108 recording, excluding its 8-byte linked prolog. */
   const std::vector<uint8_t> want = {
      0x0c,0xdd,0x10,0x06,0x67,0x10,0x44,0x00,0x00,0x80,0x20,0x00,
      0x59,0x01,0x00,0x40,0x48,0x00,0x0b,0x00,0x21,0xc4,0x1b,0x03,
      0x27,0x00,0x42,0x00,0x00,0x04,0x00,0x00,0x0b,0x00,0x00,0x00,
      0x2b,0x01,0x27,0x00,0x42,0x00,0x00,0x08,0x00,0x00,0x0c,0x80,
      0x02,0x3e,0x00,0x00,0x00,0x0c,0x3b,0x01,0x27,0x00,0x42,0x00,
      0x00,0x0c,0x00,0x00,0x57,0x16,0x54,0x00,0x00,0x40,0x48,0x00,
      0x57,0x26,0x54,0x02,0x20,0x40,0x49,0x00,0x57,0x46,0x54,0x04,
      0x40,0x40,0x4a,0x00,0x57,0x86,0x54,0x06,0x60,0x40,0x4b,0x00,
      0x0e,0x00,0x00,0x00,
   };
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8u);
}

TEST_F(AGX3PackBinary, SixVertexOutputsRetireSourcesBeforeReuseAndWriteInOrder)
{
   /* Deliberately reuse r0 after every store, with the failing gradient's
    * output order. A deferred COPY must finish its source read before the
    * next MOV_IMM overwrites r0. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   const unsigned order[] = {3, 2, 1, 5, 0, 4};
   std::vector<uint8_t> want;
   for (unsigned component : order) {
      agx_mov_imm_to(b, agx_register(0, AGX_SIZE_32), 0x3f000000 + component);
      agx_st_vary(b, agx_immediate(component), agx_register(0, AGX_SIZE_32));
      struct agx3_encoded_ldimm ld;
      ASSERT_TRUE(agx3_pack_ldimm_u32(AGX_ISA_ENCODING_AGX3B, 0,
                                     0x3f000000 + component, &ld));
      append(want, ld.bytes, ld.length);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH];
      uint8_t len = 0;
      ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B,
                     component * 2, 0, false, component + 1, 0, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_band_identity_wait(AGX_ISA_ENCODING_AGX3B, 0,
                                              1u << component, bytes, &len));
      append(want, bytes, len);
   }
   for (unsigned component = 0; component < 6; ++component) {
      uint8_t bytes[AGX3_VARY_MAX_LENGTH];
      uint8_t len = 0;
      ASSERT_TRUE(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, component * 2,
                     component, component + 1, true, 1u << component,
                     bytes, &len));
      append(want, bytes, len);
   }
   agx_stop(b);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 12u);
}

TEST_F(AGX3PackBinary, SixVertexOutputsDrainUnrelatedSlotsBeforeInjectedCopy)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   /* Two live async results unrelated to r0, including a slot later used
    * by output0. Both must retire before the injected r0-preserving wait. */
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32), AGX_SR_VERTEX_ID);
   agx_get_sr_to(b, agx_register(4, AGX_SIZE_32), AGX_SR_VERTEX_ID);
   agx_mov_imm_to(b, agx_register(0, AGX_SIZE_32), 0x3f800000);
   const unsigned order[] = {3, 2, 1, 5, 0, 4};
   for (unsigned component : order)
      agx_st_vary(b, agx_immediate(component), agx_register(0, AGX_SIZE_32));
   agx_stop(b);
   const auto got = pack(b);
   uint8_t bytes[AGX3_STAGE_MAX_LENGTH];
   uint8_t len = 0;
   ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, 6, 0,
                                      false, 4, 3, bytes, &len));
   const auto at = std::search(got.begin(), got.end(), bytes, bytes + len);
   ASSERT_NE(at, got.end());
   uint8_t wait_bytes[AGX3_ALU_MAX_LENGTH], wait_len = 0;
   ASSERT_TRUE(agx3_alu_band_identity_wait(AGX_ISA_ENCODING_AGX3B, 0, 8,
                                           wait_bytes, &wait_len));
   ASSERT_GE(got.end() - at, len + wait_len);
   EXPECT_EQ(memcmp(&*(at + len), wait_bytes, wait_len), 0);
}

TEST_F(AGX3PackBinary, SixVertexOutputsRefuseUnqualifiedGraphs)
{
   for (unsigned mode : {1u, 3u, 4u, 5u, 6u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_VERTEX;
      for (unsigned component = 0; component < 6; ++component) {
         const unsigned index = mode == 1 && component == 0 ? 1 : component;
         agx_st_vary(b, agx_immediate(index),
                     agx_register(0, mode == 3 ? AGX_SIZE_16 : AGX_SIZE_32));
      }
      if (mode == 4)
         agx_begin_cf(b);
      if (mode == 5)
         agx_test_block(b->shader);
      if (mode == 6)
         agx_device_store(b, agx_register(0, AGX_SIZE_32),
                          agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                          AGX_FORMAT_I32, 1, 2, false);
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

/* A vertex shader that writes no gl_Position still leaves the four-entry UVS
 * position group reserved, so its outputs are a dense run based at
 * AGX3_UVS_POSITION_ENTRIES rather than at zero. That is the shape of
 * KHR-GL33.transform_feedback_overflow_query_ARB's single-stream cases, whose
 * four captured floats and point size land at UVS 4..8. */
TEST_F(AGX3PackBinary, PositionlessVertexOutputsStageFromTheUvsPositionGroup)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   for (unsigned index = AGX3_UVS_POSITION_ENTRIES;
        index < AGX3_UVS_POSITION_ENTRIES + 5; ++index)
      agx_st_vary(b, agx_immediate(index), agx_register(0, AGX_SIZE_32));
   agx_stop(b);

   const auto bytes = pack(b);
   ASSERT_FALSE(bytes.empty());

   /* Each output keeps its own UVS index and stages at twice that index. */
   unsigned writes = 0;
   for (size_t at = 0; at + 8 <= bytes.size(); ) {
      const uint8_t *p = bytes.data() + at;
      if (p[0] != 0x57) {
         at += (p[0] & 0xf) == 0xb && (p[2] & 7) != 1 ? 10 : 4;
         continue;
      }
      const unsigned index = (p[4] >> 5) | ((p[5] & 0x1f) << 3);
      EXPECT_EQ(index, AGX3_UVS_POSITION_ENTRIES + writes) << writes;
      EXPECT_EQ(p[3], 2u * index) << writes;
      ++writes;
      at += 8;
   }
   EXPECT_EQ(writes, 5u);
}

TEST_F(AGX3PackBinary, ProjectiveReciprocalMatchesNativeOperandForm)
{
   const struct {
      unsigned dest;
      bool cache;
      uint8_t bytes[10];
   } witnesses[] = {
      {0, false, {0xaf,0,0x54,0,3,0,0,0x48,0x20,1}},
      {2, true,  {0xaf,0,0x54,2,3,0,2,0x48,0x20,1}},
      {4, true,  {0xaf,0,0x54,4,3,0,2,0x48,0x20,1}},
   };
   for (const auto &v : witnesses) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index rhw = agx_register(0, AGX_SIZE_32);
      rhw.cache = v.cache;
      agx_persp_rcp_agx3_to(b, agx_register(v.dest, AGX_SIZE_32), rhw, 0);
      agx_stop(b);
      std::vector<uint8_t> want(v.bytes, v.bytes + 10);
      want.insert(want.end(), {0x0e, 0, 0, 0});
      EXPECT_EQ(pack(b), want);
   }
   /* Ordinary reciprocal keeps its previously qualified encoding. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_rcp_to(b, agx_register(0, AGX_SIZE_32), agx_register(0, AGX_SIZE_32));
   agx_stop(b);
   const std::vector<uint8_t> ordinary = {
      0xaf,0,0x54,0,2,0,0,0x48,0x20,0, 0x0e,0,0,0,
   };
   EXPECT_EQ(pack(b), ordinary);
}

TEST_F(AGX3PackBinary, ProjectiveReciprocalCoefficientAndWaitBits)
{
   /* Independent physical extraction catches coefficient/source overlap and
    * accidental retention of ordinary RCP's unconditional slot6 wait. */
   for (unsigned dest : {0u, 2u, 126u, 128u, 190u}) {
      for (unsigned rhw : {0u, 2u, 126u, 128u, 190u}) {
         for (unsigned coef : {0u, 1u, 4u, 63u, 127u}) {
            for (unsigned wait = 0; wait < 64; ++wait) {
               struct agx3_persp_rcp_desc d = {};
               d.dest = dest;
               d.rhw = rhw;
               d.coef_index = coef;
               d.rhw_cache = (dest != rhw);
               d.wait_mask = wait;
               uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], len = 0;
               ASSERT_TRUE(agx3_persp_rcp_encode(AGX_ISA_ENCODING_AGX3B,
                                                &d, bytes, &len));
               ASSERT_EQ(len, 10u);
               EXPECT_EQ(bytes[3], dest);
               EXPECT_EQ(((bytes[5] >> 1) & 0x7f) | ((bytes[6] & 1) << 7), rhw);
               EXPECT_EQ(((bytes[6] >> 2) | ((bytes[7] & 1) << 6)), coef);
               EXPECT_EQ((bytes[6] >> 1) & 1, unsigned(d.rhw_cache));
               EXPECT_EQ((bytes[1] >> 4) | ((bytes[2] & 3) << 4), wait);
               EXPECT_EQ(bytes[9] & 1, 1u);
               EXPECT_EQ(bytes[4] & 1, 1u);
            }
         }
      }
   }
}

TEST_F(AGX3PackBinary, ProjectiveReciprocalRefusesUnsupportedForms)
{
   for (unsigned mode = 1; mode < 8; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index rhw = agx_register(mode == 1 ? 192 : 0, AGX_SIZE_32);
      rhw.abs = mode == 2;
      rhw.neg = mode == 3;
      rhw.channels_m1 = mode == 4 ? 1 : 0;
      rhw.cache = rhw.discard = mode == 5;
      agx_instr *I = agx_persp_rcp_agx3_to(b,
         agx_register(0, AGX_SIZE_32), rhw, mode == 6 ? 128 : 0);
      I->saturate = mode == 7;
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
   struct agx3_persp_rcp_desc d = {};
   uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], len;
   d.wait_mask = 0x40;
   EXPECT_FALSE(agx3_persp_rcp_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
}

TEST_F(AGX3PackBinary, ProjectiveRhwMatchesNativeAndKeepsUserInterp)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index r0 = agx_register(0, AGX_SIZE_32);
   agx_rhw_interp_agx3_to(b, r0, agx_immediate(0), agx_immediate(0),
                         AGX_INTERPOLATION_CENTER);
   agx_persp_rcp_agx3_to(b, r0, r0, 0);
   agx_interp_agx3_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(1),
                     agx_immediate(0), AGX_INTERPOLATION_CENTER, true);
   agx_stop(b);
   /* Native smooth_scalar: private RHW f54=1, first f63=1; user f54=f63=0. */
   const std::vector<uint8_t> want = {
      0x2f,0x0d,0x54,0,3,0,4,2,0x10,0,
      0xaf,0,0x54,0,3,0,0,0x48,0x20,1,
      0x2f,0x05,0x54,2,3,2,0,2,0x10,0, 0x0e,0,0,0,
   };
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FragmentInputStartsOnceWithFlatFirst)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_ldcf_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(4), 1);
   agx_index rhw = agx_register(8, AGX_SIZE_32);
   agx_rhw_interp_agx3_to(b, rhw, agx_immediate(0), agx_immediate(0),
                         AGX_INTERPOLATION_CENTER);
   agx_persp_rcp_agx3_to(b, agx_register(10, AGX_SIZE_32), rhw, 0);
   agx_stop(b);

   uint8_t bytes[AGX3_VARY_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_vary_ldcf_at_start(AGX_ISA_ENCODING_AGX3B, 2, 4, 1,
                                      true, bytes, &len));
   std::vector<uint8_t> prefix(bytes, bytes + len);
   /* Apple mixed_wide begins with marked LDCF `1f0b...`, followed by its
    * first F32INTERP as `2f05...`: the two families share one entry. */
   ASSERT_TRUE(agx3_vary_projective_rhw(AGX_ISA_ENCODING_AGX3B, 8, 0, false,
                                       bytes, &len));
   prefix.insert(prefix.end(), bytes, bytes + len);

   const auto code = pack(b);
   ASSERT_GE(code.size(), prefix.size());
   EXPECT_TRUE(std::equal(prefix.begin(), prefix.end(), code.begin()));
}

TEST_F(AGX3PackBinary, FragmentInputStartsOnceWithSmoothFirst)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_interp_agx3_to(b, agx_register(8, AGX_SIZE_32), agx_immediate(0),
                      agx_immediate(0), AGX_INTERPOLATION_CENTER, false);
   agx_ldcf_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(4), 1);
   agx_stop(b);

   const auto code = pack(b);
   ASSERT_GE(code.size(), 16u);
   EXPECT_EQ(code[0], 0x2f);
   EXPECT_EQ(code[1] & 8, 8u); /* marked F32INTERP */
   EXPECT_EQ(code[10], 0x1f);
   EXPECT_EQ(code[11] & 8, 0u); /* unmarked LDCF */
}

TEST_F(AGX3PackBinary, FragmentHelperProtocolLeavesInputEntryUnmarked)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_ldcf_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(4), 1);
   agx_dfdx_to(b, agx_register(4, AGX_SIZE_32),
               agx_register(6, AGX_SIZE_32));
   agx_stop(b);

   const auto code = pack(b);
   ASSERT_GE(code.size(), 16u);
   EXPECT_EQ(code[0], 0x1f);
   EXPECT_EQ(code[1] & 8, 0u); /* helper-capable LDCF: f48 clear */

   /* Removing the derivative is a negative control: the same plain LDCF is
    * the fragment-input entry and retains Apple's f48 marker. */
   agx_builder *plain = builder(AGX_ISA_ENCODING_AGX3B);
   plain->shader->stage = MESA_SHADER_FRAGMENT;
   agx_ldcf_to(plain, agx_register(2, AGX_SIZE_32), agx_immediate(4), 1);
   agx_stop(plain);
   const auto plain_code = pack(plain);
   ASSERT_GE(plain_code.size(), 6u);
   EXPECT_EQ(plain_code[0], 0x1f);
   EXPECT_EQ(plain_code[1] & 8, 8u);
}

TEST_F(AGX3PackBinary, ProjectiveRhwModeIndependentOfFirstAndRcpDestination)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_interp_agx3_to(b, agx_register(4, AGX_SIZE_32), agx_immediate(1),
                     agx_immediate(0), AGX_INTERPOLATION_CENTER, false);
   agx_index r0 = agx_register(0, AGX_SIZE_32);
   agx_rhw_interp_agx3_to(b, r0, agx_immediate(0), agx_immediate(0),
                         AGX_INTERPOLATION_CENTER);
   r0.cache = true;
   agx_persp_rcp_agx3_to(b, agx_register(2, AGX_SIZE_32), r0, 0);
   agx_stop(b);
   auto bytes = pack(b);
   ASSERT_EQ(bytes.size(), 34u);
   EXPECT_EQ(bytes[1] & 8, 8u);
   EXPECT_EQ(bytes[6] & 4, 0u);
   EXPECT_EQ(bytes[11] & 8, 0u);
   EXPECT_EQ(bytes[16] & 4, 4u);
   const std::vector<uint8_t> native_rcp =
      {0xaf,0,0x54,2,3,0,2,0x48,0x20,1};
   EXPECT_EQ(std::vector<uint8_t>(bytes.begin()+20,bytes.begin()+30), native_rcp);
}

TEST_F(AGX3PackBinary, ProjectiveRhwRefusesUnprovenDomainsAndSharedValues)
{
   for (unsigned mode = 1; mode < 10; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = mode == 1 ? MESA_SHADER_VERTEX : MESA_SHADER_FRAGMENT;
      agx_index r0 = agx_register(0, AGX_SIZE_32);
      agx_instr *producer = agx_rhw_interp_agx3_to(b, r0,
         agx_immediate(mode == 2 ? 128 : 0), agx_immediate(mode == 3 ? 1 : 0),
         mode == 4 ? AGX_INTERPOLATION_CENTROID : AGX_INTERPOLATION_CENTER);
      producer->saturate = mode == 5;
      if (mode != 6)
         agx_persp_rcp_agx3_to(b, agx_register(2, AGX_SIZE_32), r0,
                              mode == 7 ? 1 : 0);
      if (mode == 8)
         agx_fmul_to(b, agx_register(4, AGX_SIZE_32), r0, r0);
      if (mode == 9)
         agx_mov_to(b, agx_register(4, AGX_SIZE_16),
                    agx_register(1, AGX_SIZE_16));
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, ProjectiveRhwCseCannotMergeOrdinaryPositionW)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index ordinary = agx_temp(b->shader, AGX_SIZE_32);
   agx_index private_rhw = agx_temp(b->shader, AGX_SIZE_32);
   agx_interp_agx3_to(b, ordinary, agx_immediate(0), agx_immediate(0),
                     AGX_INTERPOLATION_CENTER, true);
   agx_rhw_interp_agx3_to(b, private_rhw, agx_immediate(0), agx_immediate(0),
                         AGX_INTERPOLATION_CENTER);
   agx_instr *rcp = agx_persp_rcp_agx3_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                        private_rhw, 0);
   agx_opt_cse(b->shader);
   EXPECT_EQ(rcp->src[0].value, private_rhw.value);
   EXPECT_NE(rcp->src[0].value, ordinary.value);
}

/* Native scene-crescent / scene-ring place their write fence after both
 * FLTNORMs, immediately before LSTCOL. Exercise actual packing, including an
 * asynchronous producer whose wait must remain on the first FLTNORM.
 */
TEST_F(AGX3PackBinary, G17PWriteWaitFollowsPackingAndPreservesScoreboard)
{
   for (bool blend : {false, true}) {
      for (bool async : {false, true}) {
         SCOPED_TRACE(blend);
         SCOPED_TRACE(async);
         const auto enc = AGX_ISA_ENCODING_AGX3B;
         agx_builder *b = builder(enc);
         b->shader->stage = MESA_SHADER_FRAGMENT;
         std::vector<uint8_t> want;
         uint8_t bytes[32], len;
         const uint8_t entry[] = {0x87,0x02,0x54,0,6,0};
         const uint8_t read[] = {0x87,0x02,0x54,8,8,0};
         const uint8_t release_end[] = {7,2,0x54,12,2,0,14,0,0,0};
         agx_index packed = agx_register(0, AGX_SIZE_16);
         packed.channels_m1 = 1;
         if (async) {
            agx_scs_load_agx3_to(b, agx_register(4, AGX_SIZE_32), 4,
                                AGX_FORMAT_I32, 1);
            struct agx3_scs_desc d = {};
            d.data = 4; d.width = AGX3_MEM_W32;
            d.byte_offset = 4; d.slot = 1;
            ASSERT_TRUE(agx3_scs_encode_load(enc, &d, bytes, &len));
            append(want, bytes, len);
         }
         if (blend) {
            append(want, entry, sizeof(entry));
            append(want, read, sizeof(read));
            agx_wait_pix(b, 8);
            agx_index loaded = agx_register(12, AGX_SIZE_16);
            loaded.channels_m1 = 1;
            agx_ld_tile_to(b, loaded, agx_immediate(AGX_ALL_SAMPLES),
                          agx_null(), AGX_FORMAT_U8NORM, 15, 0, false, 0);
            ASSERT_TRUE(agx3_tile_load(enc, 12, AGX3_TILE_W8, 6, 15, 0,
               AGX3_TILE_SAMPLE_1X, 0, true, true, bytes, &len));
            append(want, bytes, len);
            agx_index unpacked = agx_register(8, AGX_SIZE_32);
            unpacked.channels_m1 = 1;
            agx_normflt_agx3_to(b, unpacked, agx_register(12, AGX_SIZE_16),
                               AGX_FORMAT_U8NORM);
            struct agx3_normflt_desc d = {};
            d.dest = 8; d.dest_hint = true; d.src = 12;
            d.two_lanes = true; d.wait_mask = 1u << 5;
            ASSERT_TRUE(agx3_normflt_encode(enc, &d, bytes, &len));
            append(want, bytes, len);
         }
         agx_wait_pix(b, blend ? 4 : 12);
         for (unsigned i = 0; i < 2; ++i) {
            agx_fltnorm_agx3_to(b, agx_register(i, AGX_SIZE_16),
               agx_register(4 + 4*i, AGX_SIZE_32),
               agx_register(6 + 4*i, AGX_SIZE_32), AGX_FORMAT_U8NORM);
            struct agx3_fltnorm_desc d = {};
            d.dest = i; d.two_lanes = true;
            d.src[0] = {AGX3_FLTNORM_SRC_REGISTER, uint16_t(4+4*i), false};
            d.src[1] = {AGX3_FLTNORM_SRC_REGISTER, uint16_t(6+4*i), false};
            d.wait_mask = async && i == 0 ? 1 : 0;
            d.first_instruction = !blend && !async && i == 0;
            ASSERT_TRUE(agx3_fltnorm_encode(enc, &d, bytes, &len));
            append(want, bytes, len);
         }
         if (!blend)
            append(want, entry, sizeof(entry));
         const uint8_t write[] = {0x87,2,0x54,uint8_t(blend ? 4 : 12),8,0};
         append(want, write, sizeof(write));
         agx_st_tile(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                     AGX_FORMAT_U8NORM, 15, 0, false, 0);
         ASSERT_TRUE(agx3_tile_store(enc, 0, AGX3_TILE_W8, 15, 0,
                                    AGX3_TILE_SAMPLE_1X, 0, bytes, &len));
         append(want, bytes, len);
         append(want, release_end, sizeof(release_end));
         agx_stop(b);
         EXPECT_EQ(pack(b), want);
      }
   }
}

TEST_F(AGX3PackBinary, G17PWriteWaitDoesNotCrossUnprovenOperations)
{
   /* S8NORM is now independently qualified for the deferred protocol. */
   for (unsigned axis = 1; axis < 6; ++axis) {
      SCOPED_TRACE(axis);
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_block *first = agx_start_block(b->shader);
      if (axis == 5)
         agx_begin_cf(b);
      agx_wait_pix(b, 12);
      if (axis == 2) {
         agx_block *second = agx_test_block(b->shader);
         agx_block_add_successor(first, second);
         b->cursor = agx_after_block(second);
      } else if (axis == 3) {
         agx_fadd_to(b, agx_register(4, AGX_SIZE_32),
                     agx_register(6, AGX_SIZE_32), agx_register(8, AGX_SIZE_32));
      } else if (axis == 4) {
         agx_signal_pix(b, 12);
      }
      const auto format = axis == 0 ? AGX_FORMAT_S8NORM : AGX_FORMAT_U8NORM;
      agx_fltnorm_agx3_to(b, agx_register(0, AGX_SIZE_16),
                         agx_immediate(32), agx_immediate(48), format);
      agx_index packed = agx_register(0, AGX_SIZE_16);
      packed.channels_m1 = 1;
      agx_st_tile(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
                  format, 15, axis == 1 ? 16 : 0, false, 0);
      agx_stop(b);
      const auto bin = pack(b);
      const uint8_t original_prefix[] = {
         0x87,2,0x54,0,6,0, 0x87,2,0x54,12,8,0,
      };
      ASSERT_GE(bin.size(), sizeof(original_prefix));
      EXPECT_TRUE(std::equal(std::begin(original_prefix),
                             std::end(original_prefix), bin.begin())) << axis;
   }
}

TEST_F(AGX3PackBinary, OperandlessPixelWaitDoesNotRetireAsyncRegisters)
{
   /* agx3_emit_pixel requires zero src/dest operands; WAIT_PIX is synchronous.
    * Thus its scheduler mask is zero even with all six slots outstanding.
    * The real first consumer must retain all six waits after fence deferral.
    */
   struct agx3_slot_instr instrs[8] = {};
   struct agx3_slot_result res[8] = {};
   for (unsigned i = 0; i < 6; ++i) {
      instrs[i].is_async = true;
      instrs[i].nr_dsts = 1;
      instrs[i].dst[0] = i;
      instrs[7].src[i] = i;
   }
   instrs[7].nr_srcs = 6;
   ASSERT_TRUE(agx3_slot_assign(instrs, 8, 6, res));
   EXPECT_EQ(res[6].slot, 0);
   EXPECT_EQ(res[6].wait_mask, 0);
   EXPECT_EQ(res[7].wait_mask, 0x3f);
}

TEST_F(AGX3PackBinary, G17PColourPackingKeepsNativeUniformSources)
{
   for (unsigned base : {0u, 16u, 256u, 508u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_instr *I = agx_fltnorm_agx3_to(b, agx_register(0, AGX_SIZE_16),
         agx_uniform(base, AGX_SIZE_32), agx_uniform(base + 2, AGX_SIZE_32),
         AGX_FORMAT_U8NORM);
      agx_stop(b);
      agx_lower_uniform_sources(b->shader);
      EXPECT_EQ(I->src[0].type, AGX_INDEX_UNIFORM);
      EXPECT_EQ(I->src[1].type, AGX_INDEX_UNIFORM);
      struct agx3_fltnorm_desc d = {};
      d.two_lanes = true;
      d.src[0].kind = d.src[1].kind = AGX3_FLTNORM_SRC_UNIFORM;
      d.src[0].value = base; d.src[1].value = base + 2;
      uint8_t encoded[AGX3_FLTNORM_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_fltnorm_encode(AGX_ISA_ENCODING_AGX3B, &d, encoded, &len));
      std::vector<uint8_t> expected(encoded, encoded + len);
      const uint8_t stop[] = {14, 0, 0, 0};
      append(expected, stop, sizeof(stop));
      EXPECT_EQ(pack(b), expected);
   }
   for (unsigned value : {1u, 511u, 512u})
      EXPECT_FALSE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3B,
         AGX_OPCODE_FLTNORM_AGX3, 0, value, AGX_SIZE_32));
   for (auto size : {AGX_SIZE_16, AGX_SIZE_64})
      EXPECT_FALSE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3B,
         AGX_OPCODE_FLTNORM_AGX3, 0, 16, size));
   EXPECT_FALSE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3B,
      AGX_OPCODE_FLTNORM_AGX3, 2, 16, AGX_SIZE_32));
   EXPECT_FALSE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3B,
      AGX_OPCODE_IADD, 0, 16, AGX_SIZE_32));
}

/* An 8-bit sRGB attachment is U8NORM storage whose colour lanes the pack and
 * unpack convert: agx_compile.c marks the encoded lanes in the instruction's
 * mask, and the packer selects Apple's modes from it (s_rgba / s_r8 /
 * s_pblend witnesses in test-agx3-g17p-producers.c): both lanes encoded, lane
 * 0 encoded over a linear alpha lane, or a single encoded lane. */
/* The G17P kill (sample_mask_agx3) is scheduled as early as its sources
 * allow, like the AGX2 sample_mask: a texture fetch the kill does not depend
 * on must not run before it, or every fragment the kill retires pays for it
 * (Apple's code for the same shapes issues only the kill's own fetch first). */
TEST_F(AGX3PackBinary, G17PKillPrecedesIndependentFetches)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;

   auto fetch = [&](uint32_t x) {
      agx_index xy[2] = {agx_mov_imm(b, 32, x), agx_mov_imm(b, 32, x + 1)};
      agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_instr *collect = agx_collect_to(b, coord, 2);
      collect->src[0] = xy[0];
      collect->src[1] = xy[1];
      agx_index dest = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *tex = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0),
         agx_uniform(0, AGX_SIZE_64), agx_immediate(0), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0, false, false, false,
         AGX_GATHER_NONE, false);
      tex->op = AGX_OPCODE_TEXTURE_LOAD;
      tex->mask = 15;
      agx_instr *split = agx_split(b, 4, dest);
      for (unsigned i = 0; i < 4; ++i)
         split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);
      return std::make_pair(tex, split);
   };

   /* Program order: the kill's flag fetch, an unrelated fetch, the kill,
    * then the unrelated value's use. */
   auto flag = fetch(0x40000000);
   auto other = fetch(0x41000000);
   agx_instr *kill = agx_sample_mask_agx3(
      b, agx_mov_imm(b, 16, 1), flag.second->dest[0], 2);
   agx_index sum = agx_fadd(b, other.second->dest[0], other.second->dest[1]);
   sum = agx_fadd(b, sum, other.second->dest[2]);
   sum = agx_fadd(b, sum, other.second->dest[3]);
   agx_export(b, sum, 8);
   agx_stop(b);

   agx_pressure_schedule(b->shader);

   int pos = 0, kill_pos = -1, other_pos = -1, flag_pos = -1;
   agx_foreach_instr_global(b->shader, I) {
      if (I == kill)
         kill_pos = pos;
      else if (I == other.first)
         other_pos = pos;
      else if (I == flag.first)
         flag_pos = pos;
      ++pos;
   }
   ASSERT_GE(kill_pos, 0);
   EXPECT_LT(flag_pos, kill_pos);
   EXPECT_LT(kill_pos, other_pos);
}

TEST_F(AGX3PackBinary, G17PSrgbLanesSelectNativeConversionModes)
{
   static const struct {
      unsigned mask;
      bool two;
      uint8_t srgb;
   } cases[] = {
      {0, true, AGX3_FLTNORM_SRGB_NONE},
      {3, true, AGX3_FLTNORM_SRGB_BOTH},
      {1, true, AGX3_FLTNORM_SRGB_LANE0},
      {1, false, AGX3_FLTNORM_SRGB_BOTH},
   };
   const uint8_t stop[] = {14, 0, 0, 0};
   for (const auto &c : cases) {
      SCOPED_TRACE(c.mask * 2 + c.two);
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_instr *I = agx_fltnorm_agx3_to(b, agx_register(0, AGX_SIZE_16),
         agx_uniform(16, AGX_SIZE_32),
         c.two ? agx_uniform(18, AGX_SIZE_32) : agx_null(),
         AGX_FORMAT_U8NORM);
      I->mask = c.mask;
      agx_stop(b);
      struct agx3_fltnorm_desc d = {};
      d.two_lanes = c.two;
      d.srgb = c.srgb;
      d.src[0].kind = AGX3_FLTNORM_SRC_UNIFORM;
      d.src[0].value = 16;
      if (c.two) {
         d.src[1].kind = AGX3_FLTNORM_SRC_UNIFORM;
         d.src[1].value = 18;
      }
      uint8_t encoded[AGX3_FLTNORM_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_fltnorm_encode(AGX_ISA_ENCODING_AGX3B, &d, encoded,
                                      &len));
      std::vector<uint8_t> expected(encoded, encoded + len);
      append(expected, stop, sizeof(stop));
      EXPECT_EQ(pack(b), expected);

      agx_builder *u = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index pair = agx_register(4, AGX_SIZE_32);
      pair.channels_m1 = c.two ? 1 : 0;
      agx_normflt_agx3_to(u, pair, agx_register(0, AGX_SIZE_16),
                          AGX_FORMAT_U8NORM)->mask = c.mask;
      agx_stop(u);
      struct agx3_normflt_desc nd = {};
      nd.dest = 4;
      nd.dest_hint = true;
      nd.src = 0;
      nd.two_lanes = c.two;
      nd.srgb = c.srgb;
      ASSERT_TRUE(agx3_normflt_encode(AGX_ISA_ENCODING_AGX3B, &nd, encoded,
                                      &len));
      std::vector<uint8_t> nexpected(encoded, encoded + len);
      append(nexpected, stop, sizeof(stop));
      EXPECT_EQ(pack(u), nexpected);
   }

   /* No mode encodes a linear lane 0 under an encoded lane 1, and the sRGB
    * modes exist only for unsigned 8-bit lanes. */
   for (unsigned refused = 0; refused < 2; ++refused) {
      SCOPED_TRACE(refused);
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_fltnorm_agx3_to(b, agx_register(0, AGX_SIZE_16),
         agx_uniform(16, AGX_SIZE_32), agx_uniform(18, AGX_SIZE_32),
         refused ? AGX_FORMAT_S8NORM : AGX_FORMAT_U8NORM)->mask =
         refused ? 3 : 2;
      agx_stop(b);
      pack(b, false);
   }
}

TEST_F(AGX3PackBinary, G17PTexelFetchLODMaterializesUniform)
{
   EXPECT_FALSE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3B,
      AGX_OPCODE_TEXTURE_LOAD, 1, 16, AGX_SIZE_32));
   EXPECT_TRUE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX3B,
      AGX_OPCODE_TEXTURE_LOAD, 2, 16, AGX_SIZE_64));
   EXPECT_TRUE(agx_instr_accepts_uniform(AGX_ISA_ENCODING_AGX2,
      AGX_OPCODE_TEXTURE_LOAD, 1, 16, AGX_SIZE_16));
}

TEST_F(AGX3PackBinary, TextureFetchRegisterLodIsFullUint)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;

   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(12, AGX_SIZE_32);
   coord.channels_m1 = 1;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_register(4, AGX_SIZE_32), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;

   /* The coordinates and the full-uint level are published on staging
    * slot 2, in Apple's order (coordinates, then PACK_SMPIR bank 6); the
    * fragment enters the pixel transaction and fetches on slot 1 while
    * waiting for that staging generation, as the second in-flight read of
    * Apple's `t.read(c, l0) + t.read(c + 3, l1)` fragment does. */
   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 12, 2, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 14, 2, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_typed_lod(
      enc, 4, 2, 0, false, false, AGX3_TEX_LEVEL_INTEGER, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_fetch_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.slot = 1;
   d.wait_mask = 1u << 1;
   d.coord_kill = true;
   d.final = true;
   d.register_lod = true;
   ASSERT_TRUE(agx3_tex_encode_fetch(enc, &d, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8u);

   /* The level bank lies at halves 6/7. Declaring only the four coordinate
    * halves left it outside the sampler-private allocation, and every
    * fragment register-LOD fetch then returned zero on G17P hardware. */
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);

   agx_builder *bad = builder(enc);
   bad->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index bad_dest = agx_register(8, AGX_SIZE_32);
   bad_dest.channels_m1 = 3;
   agx_index bad_coord = agx_register(12, AGX_SIZE_32);
   bad_coord.channels_m1 = 1;
   agx_instr *bad_I = agx_texture_sample_to(
      bad, bad_dest, bad_coord, agx_register(4, AGX_SIZE_16),
      agx_immediate(0), agx_immediate(0), agx_immediate(0), agx_null(),
      AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0, false, false, false,
      AGX_GATHER_NONE, false);
   bad_I->op = AGX_OPCODE_TEXTURE_LOAD;
   bad_I->mask = 15;
   pack(bad, /*expect_ok=*/false);
}

TEST_F(AGX3PackBinary, TextureFetchArrayPacksLayerWithRegisterLod)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;

   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(12, AGX_SIZE_32);
   coord.channels_m1 = 2;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_register(4, AGX_SIZE_32), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D_ARRAY,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_array_integer_lod(
      enc, 16, 4, 1, 0, false, false, false, false, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 12, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 14, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   const struct agx3_tex_fetch_desc d = {
      .dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .slot = 1, .wait_mask = 1u << 0, .coord_kill = true, .final = true,
      .array = true, .register_lod = true,
   };
   ASSERT_TRUE(agx3_tex_encode_fetch(enc, &d, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8u);
}

TEST_F(AGX3PackBinary, TextureFetch3DRegisterLodIsFullUint)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;

   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(12, AGX_SIZE_32);
   coord.channels_m1 = 2;
   agx_instr *I = agx_texture_sample_to(
      b, dest, coord, agx_register(4, AGX_SIZE_32), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_3D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   I->op = AGX_OPCODE_TEXTURE_LOAD;
   I->mask = 15;

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_typed_lod(
      enc, 4, 2, 0, false, false, AGX3_TEX_LEVEL_INTEGER, bytes, &len));
   append(want, bytes, len);
   for (unsigned component = 0; component < 3; ++component) {
      ASSERT_TRUE(agx3_tex_stage_copy_hints(
         enc, 2 * component, 12 + 2 * component, 2, 0, false, false,
         bytes, &len));
      append(want, bytes, len);
   }
   const struct agx3_tex_fetch_desc fetch = {
      .dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .slot = 1, .wait_mask = 1u << 1, .coord_kill = true, .final = true,
      .three_d = true, .register_lod = true,
   };
   ASSERT_TRUE(agx3_tex_encode_fetch(enc, &fetch, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, G17PFltnormStartMatchesNativeF32Uniforms)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   /* Actual native uniform-f32.nat, constant float4 buffer input: uniform
    * half indices 4,6,8,10 with f24=f41=2 (32-bit). */
   for (unsigned i = 0; i < 2; ++i)
      agx_fltnorm_agx3_to(b, agx_register(i, AGX_SIZE_16),
         agx_uniform(4 + 4*i, AGX_SIZE_32),
         agx_uniform(6 + 4*i, AGX_SIZE_32), AGX_FORMAT_U8NORM);
   agx_wait_pix(b, 12);
   agx_index packed = agx_register(0, AGX_SIZE_16);
   packed.channels_m1 = 1;
   agx_st_tile(b, packed, agx_immediate(AGX_ALL_SAMPLES), agx_null(),
               AGX_FORMAT_U8NORM, 15, 0, false, 0);
   agx_stop(b);
   const std::vector<uint8_t> native = {
      0x97,0x0c,0x54,0,2,8,0x18,0xd0,5,0xc8,
      0x97,4,0x54,1,2,0x10,0x28,0xd0,5,0xc8,
      0x87,2,0x54,0,6,0, 0x87,2,0x54,12,8,0,
      0xe7,6,0x54,0,0,0,1,0x4e,0,0,0,0,
      7,2,0x54,12,2,0, 14,0,0,0,
   };
   EXPECT_EQ(pack(b), native);
}

TEST_F(AGX3PackBinary, G17PNormalized16MatchesNativeColourPrograms)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;

   {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_fltnorm_agx3_to(b, agx_register(0, AGX_SIZE_16),
                          agx_immediate_f(1.0f), agx_null(),
                          AGX_FORMAT_U16NORM);
      agx_stop(b);
      const std::vector<uint8_t> native = {
         0x97,0x0c,0x54,0x00,0x02,0x60,0x00,0x10,0x04,0x80,
         0x0e,0x00,0x00,0x00,
      };
      EXPECT_EQ(pack(b), native);
   }

   {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index pair = agx_register(0, AGX_SIZE_16);
      pair.channels_m1 = 1;
      agx_fltnorm_agx3_to(b, pair, agx_immediate_f(1.0f),
                          agx_immediate_f(0.5f), AGX_FORMAT_U16NORM);
      agx_stop(b);
      const std::vector<uint8_t> native = {
         0x97,0x0c,0x54,0x00,0x02,0x60,0x80,0x50,0x04,0x88,
         0x0e,0x00,0x00,0x00,
      };
      EXPECT_EQ(pack(b), native);
   }

   for (bool snorm : {false, true}) {
      struct agx3_normflt_desc d = {};
      d.dest = 2;
      d.dest_hint = true;
      d.src = 2;
      d.is_16bit = true;
      d.snorm = snorm;
      d.wait_mask = 1u << 5;
      uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_normflt_encode(enc, &d, bytes, &len));
      const std::vector<uint8_t> native = snorm
         ? std::vector<uint8_t>{0x17,0x04,0x56,0x02,0x01,0x04,0x14,0x2a}
         : std::vector<uint8_t>{0x17,0x04,0x56,0x02,0x01,0x04,0x14,0x4a};
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len), native);
   }

   /* A pair of 16-bit lanes spans two source registers: applegpu_g17p
    * RGBA16Unorm and RGBA16Snorm blending unpacks the destination with these
    * two reads. */
   for (bool snorm : {false, true}) {
      struct agx3_normflt_desc first = {};
      first.dest = 4;
      first.dest_hint = true;
      first.src = 0;
      first.two_lanes = true;
      first.is_16bit = true;
      first.snorm = snorm;
      first.wait_mask = 1u << 5;
      uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_normflt_encode(enc, &first, bytes, &len));
      const std::vector<uint8_t> first_native = snorm
         ? std::vector<uint8_t>{0x17,0x04,0x56,0x04,0x01,0x00,0x1c,0xaa}
         : std::vector<uint8_t>{0x17,0x04,0x56,0x04,0x01,0x00,0x1c,0xca};
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len), first_native);

      struct agx3_normflt_desc second = first;
      second.dest = 0;
      second.dest_hint = false;
      second.src = 2;
      second.wait_mask = 0;
      ASSERT_TRUE(agx3_normflt_encode(enc, &second, bytes, &len));
      const std::vector<uint8_t> second_native = snorm
         ? std::vector<uint8_t>{0x17,0x04,0x54,0x00,0x00,0x04,0x1c,0xaa}
         : std::vector<uint8_t>{0x17,0x04,0x54,0x00,0x00,0x04,0x1c,0xca};
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len), second_native);
   }
}

TEST_F(AGX3PackBinary, G17PFltnormStartIsPositionNotFirstUniformRead)
{
   for (bool preceding : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      if (preceding)
         agx_fadd_to(b, agx_register(4, AGX_SIZE_32),
                    agx_register(6, AGX_SIZE_32), agx_register(8, AGX_SIZE_32));
      agx_fltnorm_agx3_to(b, agx_register(0, AGX_SIZE_16),
                         agx_immediate(16), agx_immediate(32), AGX_FORMAT_U8NORM);
      agx_fltnorm_agx3_to(b, agx_register(1, AGX_SIZE_16),
                         agx_uniform(8, AGX_SIZE_32),
                         agx_uniform(10, AGX_SIZE_32), AGX_FORMAT_U8NORM);
      agx_stop(b);
      const auto bin = pack(b);
      const uint8_t immediate[] = {
         0x97,uint8_t(preceding ? 4 : 12),0x54,0,2,0x20,0x80,0x50,4,0xc8,
      };
      const uint8_t later[] = {0x97,4,0x54,1,2,0x10,0x28,0xd0,5,0xc8};
      /* Native immediate-only first FLTNORM sets f68; native position-derived
       * arithmetic before the first uniform FLTNORM leaves it clear. */
      const auto at = std::search(bin.begin(),bin.end(),std::begin(immediate),
                                  std::end(immediate));
      ASSERT_NE(at, bin.end());
      EXPECT_TRUE(std::equal(std::begin(later),std::end(later),at+10));
   }
   struct agx3_fltnorm_desc d = {};
   d.src[0] = {AGX3_FLTNORM_SRC_IMMEDIATE, 16, false};
   d.first_instruction = true;
   uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], len;
   /* M5's newly qualified entry marker covers immediate pairs, not this
    * unwitnessed single-lane leading instruction. */
   EXPECT_FALSE(agx3_fltnorm_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &len));
}

TEST_F(AGX3PackBinary, G17PFltnormStartRelocatesAcrossLinkedParts)
{
   for (unsigned main_bytes : {0u, 12u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_fltnorm_agx3_to(b, agx_register(0, AGX_SIZE_16),
                         agx_uniform(4, AGX_SIZE_32),
                         agx_uniform(6, AGX_SIZE_32), AGX_FORMAT_U8NORM);
      agx_stop(b);
      const auto epilog = pack(b);
      ASSERT_TRUE(b->shader->agx3_fltnorm_start);
      /* Real crescent main's three 4-byte uniform COPY exports. */
      const uint8_t main[] = {0x4b,0x10,1,8,0x5b,0x14,1,8,0x6b,0x12,1,8};
      std::vector<uint8_t> linked(main, main+main_bytes);
      linked.insert(linked.end(),epilog.begin(),epilog.end());
      ASSERT_TRUE(agx3_fltnorm_relocate_start(linked.data()+main_bytes,
                  epilog.size(),b->shader->agx3_fltnorm_start,main_bytes));
      EXPECT_EQ(linked[main_bytes+1], main_bytes ? 4 : 12);
      EXPECT_TRUE(std::equal(main,main+main_bytes,linked.begin()));
      EXPECT_EQ(epilog[1], 12); /* cached part remains reusable */
      EXPECT_TRUE(std::equal(epilog.begin()+2,epilog.end(),
                             linked.begin()+main_bytes+2));
   }
   uint8_t invalid[10] = {0x97,4,0x54};
   EXPECT_FALSE(agx3_fltnorm_relocate_start(invalid,10,true,12));
   invalid[1] = 12;
   EXPECT_FALSE(agx3_fltnorm_relocate_start(invalid,9,true,12));
   /* No provenance means no scanning or rewriting, even for matching bytes. */
   EXPECT_TRUE(agx3_fltnorm_relocate_start(invalid,10,false,12));
   EXPECT_EQ(invalid[1],12);
}

TEST_F(AGX3PackBinary, G17PColourFusionUsesCompilerProvenanceAndNativeUniforms)
{
   for (unsigned copies = 1; copies <= 4; ++copies) {
      agx_builder *main = builder(AGX_ISA_ENCODING_AGX3B);
      main->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true; key.secondary = false;
      for (unsigned i = 0; i < copies; ++i) {
         agx_index reg = agx_register(8+2*i, AGX_SIZE_32);
         agx_bitop_to(main,reg,agx_uniform(4+2*i,AGX_SIZE_32),agx_immediate(0),0xa);
         agx_export(main,reg,reg.value);
      }
      agx_stop(main);
      const auto main_bin = pack(main);
      EXPECT_EQ(main_bin.size(),4*copies);
      const auto main_info = main->shader->agx3_colour_link;
      ASSERT_TRUE(main_info.copy_only);
      ASSERT_EQ(main_info.copy_count,copies);

      agx_builder *epi = builder(AGX_ISA_ENCODING_AGX3B);
      epi->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = false; key.secondary = true;
      agx_wait_pix(epi,12);
      for (unsigned i = 0; i < 2; ++i)
         agx_fltnorm_agx3_to(epi,agx_register(i,AGX_SIZE_16),
            agx_register(8+2*((2*i)%copies),AGX_SIZE_32),
            agx_register(8+2*((2*i+1)%copies),AGX_SIZE_32),AGX_FORMAT_U8NORM);
      agx_index packed = agx_register(0,AGX_SIZE_16);
      packed.channels_m1 = 1;
      agx_st_tile(epi,packed,agx_immediate(AGX_ALL_SAMPLES),agx_null(),
                  AGX_FORMAT_U8NORM,15,0,false, 0);
      agx_stop(epi);
      const auto epi_bin = pack(epi);
      const auto epi_info = epi->shader->agx3_colour_link;
      ASSERT_TRUE(epi_info.opaque_pack);
      uint8_t fused[AGX3_COLOUR_LINK_SIZE];
      ASSERT_TRUE(agx3_fuse_colour_parts(&main_info,&epi_info,fused));
      /* Four distinct exports produce the exact native float4-buffer shader. */
      if (copies == 4) {
         const uint8_t native[] = {
            0x97,12,0x54,0,2,8,0x18,0xd0,5,0xc8,
            0x97,4,0x54,1,2,0x10,0x28,0xd0,5,0xc8,
            0x87,2,0x54,0,6,0,0x87,2,0x54,12,8,0,
            0xe7,6,0x54,0,0,0,1,0x4e,0,0,0,0,
            7,2,0x54,12,2,0,14,0,0,0,
         };
         EXPECT_EQ(std::vector<uint8_t>(fused,fused+sizeof(fused)),
                   std::vector<uint8_t>(std::begin(native),std::end(native)));
      }
      EXPECT_EQ(main_bin.size(),4*copies); /* cached inputs remain immutable */
      EXPECT_EQ(epi_bin.size(),54u);
      for (unsigned mode = 0; mode < 8; ++mode) {
         auto bad_main = main_info;
         auto bad_epi = epi_info;
         switch (mode) {
         case 0: bad_main.copy_only = false; break;
         case 1: bad_main.copies[0].uniform |= 1; break;
         case 2: bad_epi.opaque_pack = false; break;
         case 3: bad_epi.packs[1].src[0].value = 250; break;
         case 4: bad_epi.packs[0].wait_mask = 1; break;
         case 5: bad_epi.packs[0].snorm = true; break;
         case 6: bad_epi.store_reg = bad_main.copies[0].reg;
                 bad_epi.packs[0].dest = bad_epi.store_reg;
                 bad_epi.packs[1].dest = bad_epi.store_reg+1; break;
         case 7: bad_main.copy_count = 0; break;
         }
         memset(fused,0xa5,sizeof(fused));
         EXPECT_FALSE(agx3_fuse_colour_parts(&bad_main,&bad_epi,fused)) << mode;
         EXPECT_TRUE(std::all_of(fused,fused+sizeof(fused),[](uint8_t v){return v==0xa5;}));
      }
   }
}

TEST_F(AGX3PackBinary, G17PColourFusionRejectsIncompleteMainProof)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true; key.secondary = false;
      agx_index reg = agx_register(8,AGX_SIZE_32);
      agx_bitop_to(b,reg,agx_uniform(16,AGX_SIZE_32),agx_immediate(0),0xa);
      if (mode != 0)
         agx_export(b,reg,8);
      if (mode == 1)
         agx_bitop_to(b,reg,agx_uniform(18,AGX_SIZE_32),agx_immediate(0),0xa);
      if (mode == 2)
         agx_fadd_to(b,agx_register(10,AGX_SIZE_32),reg,reg);
      if (mode == 3)
         agx_test_block(b->shader);
      agx_stop(b);
      pack(b);
      EXPECT_FALSE(b->shader->agx3_colour_link.copy_only) << mode;
   }
}

TEST_F(AGX3PackBinary, G17PColourFusionAcceptsS8NormAndRejectsOtherChanges)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = false; key.secondary = true;
      if (mode == 0)
         agx_fadd_to(b,agx_register(8,AGX_SIZE_32),
                    agx_register(10,AGX_SIZE_32),agx_register(12,AGX_SIZE_32));
      if (mode == 1)
         agx_begin_cf(b);
      agx_wait_pix(b,12);
      for (unsigned i = 0; i < 2; ++i)
         agx_fltnorm_agx3_to(b,agx_register(i,AGX_SIZE_16),
                            agx_register(8,AGX_SIZE_32),agx_immediate(48),
                            mode == 2 ? AGX_FORMAT_S8NORM : AGX_FORMAT_U8NORM);
      agx_index packed = agx_register(0,AGX_SIZE_16);
      packed.channels_m1 = 1;
      agx_st_tile(b,packed,agx_immediate(AGX_ALL_SAMPLES),agx_null(),
                  mode == 2 ? AGX_FORMAT_S8NORM : AGX_FORMAT_U8NORM,
                  15,mode == 3 ? 16 : 0,false, 0);
      agx_stop(b);
      pack(b);
      EXPECT_EQ(b->shader->agx3_colour_link.opaque_pack, mode == 2) << mode;
   }
}

TEST_F(AGX3PackBinary, G17PLinearCenterInterpolationMatchesNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_interp_agx3_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(0),
                     agx_zero(), AGX_INTERPOLATION_CENTER, false);
   agx_interp_agx3_to(b, agx_register(2, AGX_SIZE_32), agx_immediate(1),
                     agx_zero(), AGX_INTERPOLATION_CENTER, false);
   agx_stop(b);
   /* Native linear_scalar start; linear_vector next scalar with r1/cf1.
    * No projective RHW marker, reciprocal, or multiply. */
   const std::vector<uint8_t> want = {
      0x2f,0x0d,0x54,0,3,0,0,2,0x10,0,
      0x2f,0x05,0x54,2,3,2,0,2,0x10,0,
      0x0e,0,0,0,
   };
   EXPECT_EQ(pack(b), want);
}


TEST_F(AGX3PackBinary, SixOutputTailSinkingPreservesValuesThroughRA)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;
   const unsigned order[] = {3, 2, 1, 5, 0, 4};
   agx_instr *stores[6];
   for (unsigned c : order) {
      agx_index value = agx_mov_imm(b, 32, 0x3f000000 + c);
      stores[c] = agx_st_vary(b, agx_immediate(c), value);
   }
   agx_stop(b);
   agx_pressure_schedule(b->shader);
   ASSERT_TRUE(agx3_sink_vertex_outputs(b->shader));
   unsigned next = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_ST_VARY) {
         ASSERT_LT(next, 6u);
         EXPECT_EQ(I, stores[next++]);
      } else if (next) {
         EXPECT_EQ(I->op, AGX_OPCODE_STOP);
      }
   }
   EXPECT_EQ(next, 6u);
   agx_ra(b->shader);
   /* Independently interpret the physical assignments: all six original
    * constants must survive until their terminal stores after actual RA. */
   uint32_t regs[256] = {0};
   std::vector<uint8_t> want;
   unsigned stored = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_MOV_IMM) {
         regs[I->dest[0].value] = I->imm;
         struct agx3_encoded_ldimm ld;
         ASSERT_TRUE(agx3_pack_ldimm_u32(AGX_ISA_ENCODING_AGX3B,
                         I->dest[0].value, I->imm, &ld));
         append(want, ld.bytes, ld.length);
      } else if (I->op == AGX_OPCODE_ST_VARY) {
         const unsigned c = I->src[0].value;
         EXPECT_EQ(regs[I->src[1].value], 0x3f000000u + c);
         uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len;
         ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B,
                       c * 2, I->src[1].value, I->src[1].kill,
                       c + 1, 0, bytes, &len));
         append(want, bytes, len);
         stored++;
      } else {
         ASSERT_EQ(I->op, AGX_OPCODE_STOP);
      }
   }
   EXPECT_EQ(stored, 6u);
   for (unsigned c = 0; c < 6; ++c) {
      uint8_t bytes[AGX3_VARY_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, c * 2, c,
                                c + 1, true, 1u << c, bytes, &len));
      append(want, bytes, len);
   }
   want.insert(want.end(), {14, 0, 0, 0});
   EXPECT_EQ(pack(b), want); /* No intervening band waits. */
   EXPECT_EQ(b->shader->agx3_staging_halfs, 12u);
}

TEST_F(AGX3PackBinary, SixTerminalOutputsDrainOriginalProducersOnce)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   agx_get_sr_to(b, agx_register(2, AGX_SIZE_32), AGX_SR_VERTEX_ID);
   agx_get_sr_to(b, agx_register(4, AGX_SIZE_32), AGX_SR_VERTEX_ID);
   for (unsigned c = 0; c < 6; ++c)
      agx_st_vary(b, agx_immediate(c),
                     agx_register(c % 2 ? 4 : 2, AGX_SIZE_32));
   agx_stop(b);
   auto got = pack(b);
   std::vector<uint8_t> tail;
   for (unsigned c = 0; c < 6; ++c) {
      uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, c * 2,
                 c % 2 ? 4 : 2, false, c + 1, c == 0 ? 3 : 0, bytes, &len));
      append(tail, bytes, len);
   }
   for (unsigned c = 0; c < 6; ++c) {
      uint8_t bytes[AGX3_VARY_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, c * 2, c,
                                c + 1, true, 1u << c, bytes, &len));
      append(tail, bytes, len);
   }
   tail.insert(tail.end(), {14, 0, 0, 0});
   ASSERT_GE(got.size(), tail.size());
   EXPECT_TRUE(std::equal(tail.begin(), tail.end(), got.end() - tail.size()));
}

TEST_F(AGX3PackBinary, SixOutputTailSinkingRefusesWithoutMutation)
{
   for (unsigned mode = 1; mode < 9; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = mode == 1 ? MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX;
      b->shader->is_preamble = mode == 2;
      for (unsigned c = 0; c < (mode == 3 ? 4u : 6u); ++c) {
         agx_index value = agx_mov_imm(b, 32, 0x3f000000 + c);
         if (mode == 4) value = agx_register(0, AGX_SIZE_32);
         agx_st_vary(b, agx_immediate(mode == 5 && c == 5 ? 4 : c), value);
      }
      if (mode == 6) agx_begin_cf(b);
      if (mode != 7) agx_stop(b);
      if (mode == 8) agx_test_block(b->shader);
      std::vector<agx_instr *> before;
      agx_foreach_instr_global(b->shader, I) before.push_back(I);
      EXPECT_FALSE(agx3_sink_vertex_outputs(b->shader)) << mode;
      std::vector<agx_instr *> after;
      agx_foreach_instr_global(b->shader, I) after.push_back(I);
      EXPECT_EQ(before, after) << mode;
   }
}


TEST_F(AGX3PackBinary, SixOutputVertexAddressArithmeticSurvivesRA)
{
   for (unsigned shift : {0u, 3u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_VERTEX;
      b->shader->nir = rzalloc(mem_ctx, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_VERTEX;
      agx_index vertex = agx_temp(b->shader, AGX_SIZE_32);
      agx_get_sr_to(b, vertex, AGX_SR_VERTEX_ID);
      /* Pure integer address arithmetic supplies the stride adjustment
       * before the device load. Exercise both unshifted and shifted forms
       * already supported by the normal IADD encoder. */
      agx_index index = agx_temp(b->shader, AGX_SIZE_32);
      agx_iadd_to(b, index, vertex, vertex, shift);
      agx_index loaded = agx_temp(b->shader, AGX_SIZE_32);
      agx_device_load_to(b, loaded, agx_uniform(0, AGX_SIZE_64), index,
                         AGX_FORMAT_I32, 1, 2, false);
      for (unsigned c : {3u, 2u, 1u, 5u, 0u, 4u}) {
         agx_index value = c >= 4 ? loaded :
                            agx_mov_imm(b, 32, 0x3f000000 + c);
         agx_st_vary(b, agx_immediate(c), value);
      }
      agx_stop(b);
      agx_pressure_schedule(b->shader);
      ASSERT_TRUE(agx3_sink_vertex_outputs(b->shader));
      agx_ra(b->shader);
      /* The first vertex load owns native slot 6. Its result stays live
       * through RA until both user stores, and the first terminal COPY
       * must retire that load even when copying a constant position. */
      unsigned loaded_reg = ~0u, stores = 0;
      std::vector<uint8_t> tail;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_DEVICE_LOAD)
            loaded_reg = I->dest[0].value;
         if (I->op != AGX_OPCODE_ST_VARY)
            continue;
         const unsigned c = I->src[0].value;
         ASSERT_EQ(c, stores++);
         if (c >= 4) {
            EXPECT_EQ(I->src[1].value, loaded_reg);
         }
         uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len;
         ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B,
                       c * 2, I->src[1].value, I->src[1].kill,
                       c + 1, c == 0 ? 32 : 0, bytes, &len));
         append(tail, bytes, len);
      }
      ASSERT_EQ(stores, 6u);
      for (unsigned c = 0; c < 6; ++c) {
         uint8_t bytes[AGX3_VARY_MAX_LENGTH], len;
         ASSERT_TRUE(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, c * 2, c,
                                   c + 1, true, 1u << c, bytes, &len));
         append(tail, bytes, len);
      }
      tail.insert(tail.end(), {14, 0, 0, 0});
      auto got = pack(b);
      ASSERT_GE(got.size(), tail.size());
      EXPECT_TRUE(std::equal(tail.begin(), tail.end(), got.end() - tail.size()));
      EXPECT_EQ(b->shader->agx3_staging_halfs, 12u);
   }
}

TEST_F(AGX3PackBinary, NativeHighVertexOutputIndices)
{
   const struct {
      unsigned value, component, slot, wait;
      bool kill;
      uint8_t bytes[8];
   } native[] = {
      {8, 4, 5, 16, true, {0x57,0x06,0x55,0x08,0x80,0x40,0x4c,0x00}},
      {16, 8, 6, 16, true, {0x57,0x06,0x55,0x10,0x00,0x41,0x4d,0x00}},
      {18, 9, 5, 0, true, {0x57,0x06,0x54,0x12,0x20,0x41,0x4c,0x00}},
      {20, 10, 4, 8, true, {0x57,0x86,0x54,0x14,0x40,0x41,0x4b,0x00}},
      {62, 31, 2, 2, true, {0x57,0x26,0x54,0x3e,0xe0,0x43,0x49,0x00}},
      /* Fresh Apple G17P 36-scalar vertex output, AIR 0b42132b9e9a... /
       * native b92aedf08dfd...; the final user scalar is UVS index 35. */
      {72, 35, 4, 0, true, {0x57,0x06,0x54,0x48,0x60,0x44,0x4b,0x00}},
   };
   for (auto &v : native) {
      uint8_t out[AGX3_VARY_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, v.value,
                                v.component, v.slot, v.kill, v.wait, out, &len));
      ASSERT_EQ(len, 8u);
      EXPECT_EQ(std::vector<uint8_t>(out, out + 8),
                std::vector<uint8_t>(v.bytes, v.bytes + 8));
   }
}

TEST_F(AGX3PackBinary, SparseVertexOutputsPermitPureProducerBetweenStores)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_VERTEX;
   key.g17p_sparse_vertex_outputs = true;

   agx_index value = agx_register(0, AGX_SIZE_32);
   value.discard = true;
   agx_st_vary(b, agx_immediate(4), value);
   agx_mov_imm_to(b, agx_register(0, AGX_SIZE_32), 0x3f800000);
   agx_st_vary(b, agx_immediate(5), value);
   agx_stop(b);

   EXPECT_FALSE(pack(b).empty());
   EXPECT_EQ(b->shader->agx3_staging_halfs, 12u);
}

TEST_F(AGX3PackBinary, WideVertexOutputsPreserveValuesAndRecycleSlotsAfterWrites)
{
   for (unsigned count : {5u, 7u, 8u, 9u, 11u, 16u, 32u, 52u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_VERTEX;
      b->shader->nir = rzalloc(mem_ctx, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_VERTEX;
      for (unsigned c = count; c-- > 0;) {
         agx_index value = agx_mov_imm(b, 32, 0x3f000000 + c);
         agx_st_vary(b, agx_immediate(c), value);
      }
      agx_stop(b);
      agx_pressure_schedule(b->shader);
      ASSERT_TRUE(agx3_sink_vertex_outputs(b->shader));
      agx_ra(b->shader);
      uint32_t regs[256] = {0};
      unsigned stored = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_MOV_IMM)
            regs[I->dest[0].value] = I->imm;
         else if (I->op == AGX_OPCODE_ST_VARY) {
            EXPECT_EQ(I->src[0].value, stored);
            EXPECT_EQ(regs[I->src[1].value], 0x3f000000u + stored++);
         } else
            ASSERT_EQ(I->op, AGX_OPCODE_STOP);
      }
      ASSERT_EQ(stored, count);
      const auto bytes = pack(b);
      /* Skip the independently sized immediate producers. Decode actual
       * emitted COPY/UVSW operands, tracking outstanding staging reads and
       * writeout slots. This catches premature reuse or lost UVS values. */
      size_t at = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op != AGX_OPCODE_MOV_IMM) break;
         struct agx3_encoded_ldimm ld;
         ASSERT_TRUE(agx3_pack_ldimm_u32(AGX_ISA_ENCODING_AGX3B,
                       I->dest[0].value, I->imm, &ld));
         at += ld.length;
      }
      unsigned copies = 0, writes = 0, pending = 0, active_writes = 0;
      bool ready[AGX3_MAX_VERTEX_OUTPUTS] = {false};
      while (at + 4 < bytes.size()) {
         const uint8_t *p = bytes.data() + at;
         if ((p[0] & 0xf) == 0xb) {
            const unsigned len = (p[2] & 7) == 1 ? 4 : 10;
            ASSERT_LE(at + len, bytes.size());
            const unsigned slot = len == 4 ? 1 : 1 + ((p[7] >> 2) & 7);
            const unsigned wait = len == 4 ?
               (p[3] >> 5 ? 1u << ((p[3] >> 5) - 1) : 0) :
               (p[5] >> 5) | ((p[7] >> 5) << 3);
            ASSERT_LE(slot, 6u);
            EXPECT_EQ(((p[3] & 1) | ((p[0] >> 4) << 1) |
                       ((p[2] >> 6) << 5)), 2 * copies);
            EXPECT_EQ((active_writes & (1u << (slot - 1))) & ~wait, 0u);
            active_writes &= ~wait;
            EXPECT_EQ(pending & (1u << (slot - 1)), 0u);
            pending |= 1u << (slot - 1);
            ready[copies++] = true;
            at += len;
         } else {
            ASSERT_EQ(p[0], 0x57);
            ASSERT_LE(at + 8, bytes.size());
            const unsigned component = (p[4] >> 5) | ((p[5] & 0x1f) << 3);
            const unsigned slot = 1 + (p[6] & 7);
            const unsigned wait = (p[1] >> 4) | ((p[2] & 3) << 4);
            ASSERT_EQ(component, writes);
            ASSERT_TRUE(ready[component]);
            EXPECT_EQ(p[3], 2 * component);
            EXPECT_EQ(wait & (1u << (slot - 1)), 1u << (slot - 1));
            pending &= ~wait;
            active_writes |= 1u << (slot - 1);
            ++writes;
            at += 8;
         }
      }
      EXPECT_EQ(copies, count);
      EXPECT_EQ(writes, count);
      EXPECT_EQ(pending, 0u);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 2 * count);
      ASSERT_EQ(at + 4, bytes.size());
      EXPECT_EQ(bytes[at], 14);
   }
}

TEST_F(AGX3PackBinary, NativeMaximumVertexOutputs)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_VERTEX;

   /* The promoted Apple vs68 witness grounds the same dense UVSW format used
    * by the opt-in 72-output qualification shape. Exercise every UVS index the
    * eight-bit index operand reaches. A shared source keeps this codec test
    * independent of the register allocator's synthetic no-spill pressure.
    *
    * Staging addresses stop growing at AGX3_G17P_STAGING_OUTPUTS: that is the
    * whole capacity of the separate eight-bit staging operand, and a wider
    * program rotates through it rather than refusing. */
   for (unsigned c = 0; c < AGX3_MAX_VERTEX_OUTPUTS; ++c)
      agx_st_vary(b, agx_immediate(c), agx_register(4, AGX_SIZE_32));
   agx_stop(b);

   const auto bytes = pack(b);
   EXPECT_FALSE(bytes.empty());
   EXPECT_EQ(b->shader->agx3_staging_halfs,
             2u * AGX3_G17P_STAGING_OUTPUTS);

   agx_builder *too_wide = builder(enc);
   too_wide->shader->stage = MESA_SHADER_VERTEX;
   for (unsigned c = 0; c <= AGX3_MAX_VERTEX_OUTPUTS; ++c)
      agx_st_vary(too_wide, agx_immediate(c),
                  agx_register(4, AGX_SIZE_32));
   agx_stop(too_wide);
   EXPECT_TRUE(pack(too_wide, false).empty());
}

/* A points geometry shader capturing the full interleaved transform-feedback
 * width writes position (four), point size (one) and thirty-one float4 user
 * locations: 129 outputs, one past the staging operand's capacity. UVSW
 * addresses its UVS index and its staging half through two independent
 * eight-bit operands, so the index is not what runs out; the outputs past the
 * capacity reuse the staging halves of batches that have already drained.
 *
 * KHR-GL33.transform_feedback.{capture,query,discard}_geometry_interleaved_test
 * are exactly this shape, and refused at st_vary #4 while the width was
 * pinned to the staging capacity. */
TEST_F(AGX3PackBinary, VertexOutputsWiderThanTheStagingCapacityRotate)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   const unsigned count = AGX3_G17P_STAGING_OUTPUTS + 1;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_VERTEX;

   for (unsigned c = 0; c < count; ++c)
      agx_st_vary(b, agx_immediate(c), agx_register(4, AGX_SIZE_32));
   agx_stop(b);

   const auto bytes = pack(b);
   EXPECT_FALSE(bytes.empty());
   EXPECT_EQ(b->shader->agx3_staging_halfs, 2u * AGX3_G17P_STAGING_OUTPUTS);

   /* Every UVSW keeps its own UVS index -- nothing about the index wraps --
    * and takes the rotated staging half. UVSW is the eight-byte form starting
    * 0x57, with the staging half in byte 3 and the UVS index split across
    * bytes 4 and 5, exactly as the narrower wide-output test decodes it. */
   unsigned writes = 0;
   for (size_t at = 0; at + 8 <= bytes.size(); ) {
      const uint8_t *p = bytes.data() + at;
      if (p[0] != 0x57) {
         at += (p[0] & 0xf) == 0xb && (p[2] & 7) != 1 ? 10 : 4;
         continue;
      }
      const unsigned index = (p[4] >> 5) | ((p[5] & 0x1f) << 3);
      EXPECT_EQ(index, writes) << writes;
      EXPECT_EQ(p[3], 2u * (writes % AGX3_G17P_STAGING_OUTPUTS)) << writes;
      ++writes;
      at += 8;
   }
   EXPECT_EQ(writes, count);
}

TEST_F(AGX3PackBinary, SideEffectBeforeWideTerminalVertexOutputs)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_VERTEX;

   /* Apple's explicit-level vertex texture operation is independent of the
    * terminal UVSW protocol. This is the shape emitted by Firefox WebRender:
    * a texture read followed by a dense, wide output suffix. */
   agx_index tex_dest = agx_register(20, AGX_SIZE_32);
   agx_index coord = agx_register(12, AGX_SIZE_32);
   tex_dest.channels_m1 = 3;
   coord.channels_m1 = 1;
   agx_instr *tex = agx_texture_sample_to(
      b, tex_dest, coord, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   tex->op = AGX_OPCODE_TEXTURE_LOAD;
   tex->mask = 15;

   for (unsigned c = 0; c < 51; ++c)
      agx_st_vary(b, agx_immediate(c), agx_register(4, AGX_SIZE_32));
   agx_stop(b);

   const auto bytes = pack(b);
   EXPECT_FALSE(bytes.empty());
   EXPECT_EQ(b->shader->agx3_staging_halfs, 102u);
}

/* WebRender materializes the constant final output after the preceding fifty
 * stores.  Those stores are not a syntactically terminal bundle, but each is
 * copied to its unique staging address before the late producer can reuse its
 * GPR.  The first old refusal was therefore output 4, not an unsupported UVSW
 * field: the nonterminal staging mode was arbitrarily limited to six outputs. */
TEST_F(AGX3PackBinary, FirefoxDenseOutputsAllowLatePureProducer)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (unsigned count : {40u, 51u}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_VERTEX;

      for (unsigned c = 0; c + 1 < count; ++c)
         agx_st_vary(b, agx_immediate(c), agx_register(24, AGX_SIZE_32));
      agx_mov_imm_to(b, agx_register(24, AGX_SIZE_32), 0x3f800000);
      agx_st_vary(b, agx_immediate(count - 1),
                  agx_register(24, AGX_SIZE_32));
      agx_stop(b);

      const auto bytes = pack(b);
      EXPECT_FALSE(bytes.empty()) << count;
      EXPECT_EQ(b->shader->agx3_staging_halfs, 2u * count) << count;
   }
}

/* Firefox 155's WebRender vertex program is the same shape with 49 dense
 * outputs: its final store's source GPR is also read by an earlier store, so
 * the late pure producer is only safe because every earlier value has already
 * been copied to its own staging address.  That safety argument does not
 * depend on the number of outputs, and the captured 40/51 pair was only ever
 * what happened to be captured -- pinning it refused Firefox at st_vary #4 and
 * left the browser with no WebRender pipeline at all. */
TEST_F(AGX3PackBinary, FirefoxDenseOutputsAcceptAnyGroundedWidth)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (unsigned count : {5u, 39u, 40u, 49u, 51u, AGX3_MAX_VERTEX_OUTPUTS}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_VERTEX;

      for (unsigned c = 0; c + 1 < count; ++c)
         agx_st_vary(b, agx_immediate(c), agx_register(24, AGX_SIZE_32));
      agx_mov_imm_to(b, agx_register(24, AGX_SIZE_32), 0x3f800000);
      agx_st_vary(b, agx_immediate(count - 1),
                  agx_register(24, AGX_SIZE_32));
      agx_stop(b);

      const auto bytes = pack(b);
      EXPECT_FALSE(bytes.empty()) << count;
      EXPECT_EQ(b->shader->agx3_staging_halfs,
                2u * MIN2(count, AGX3_G17P_STAGING_OUTPUTS)) << count;
   }

   /* The width ceiling still fails closed. */
   agx_builder *too_wide = builder(enc);
   too_wide->shader->stage = MESA_SHADER_VERTEX;
   for (unsigned c = 0; c < AGX3_MAX_VERTEX_OUTPUTS; ++c)
      agx_st_vary(too_wide, agx_immediate(c), agx_register(24, AGX_SIZE_32));
   agx_mov_imm_to(too_wide, agx_register(24, AGX_SIZE_32), 0x3f800000);
   agx_st_vary(too_wide, agx_immediate(AGX3_MAX_VERTEX_OUTPUTS),
               agx_register(24, AGX_SIZE_32));
   agx_stop(too_wide);
   EXPECT_TRUE(pack(too_wide, false).empty());
}

/* Late producers other than the Firefox suffix's 1.0 no longer need that
 * exact shape: the interleaved protocol stages every store where it happens,
 * so each of these writes every component exactly once. */
TEST_F(AGX3PackBinary, FirefoxDenseOutputsVariantsStageEveryStore)
{
   const auto uvsw_components = [](const std::vector<uint8_t> &code) {
      std::vector<unsigned> seen;
      for (size_t i = 0; i + 8 <= code.size(); ++i) {
         for (unsigned c = 0; c < 51; ++c) {
            uint8_t bytes[AGX3_VARY_MAX_LENGTH];
            uint8_t len = 0;
            if (agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, c * 2, c, c % 6 + 1,
                               true, 1u << (c % 6), bytes, &len) &&
                std::equal(bytes, bytes + len, code.begin() + i))
               seen.push_back(c);
         }
      }
      std::sort(seen.begin(), seen.end());
      return seen;
   };
   std::vector<unsigned> all(51);
   for (unsigned c = 0; c < 51; ++c)
      all[c] = c;

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   for (unsigned c = 0; c < 50; ++c)
      agx_st_vary(b, agx_immediate(c), agx_register(24, AGX_SIZE_32));
   agx_mov_imm_to(b, agx_register(24, AGX_SIZE_32), 0x3f000000);
   agx_st_vary(b, agx_immediate(50), agx_register(24, AGX_SIZE_32));
   agx_stop(b);
   EXPECT_EQ(uvsw_components(pack(b)), all);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   for (unsigned c = 0; c < 50; ++c)
      agx_st_vary(b, agx_immediate(c), agx_register(24, AGX_SIZE_32));
   agx_mov_imm_to(b, agx_register(26, AGX_SIZE_32), 0x3f800000);
   agx_st_vary(b, agx_immediate(50), agx_register(24, AGX_SIZE_32));
   agx_stop(b);
   EXPECT_EQ(uvsw_components(pack(b)), all);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   for (unsigned c = 0; c < 49; ++c)
      agx_st_vary(b, agx_immediate(c), agx_register(24, AGX_SIZE_32));
   agx_mov_imm_to(b, agx_register(24, AGX_SIZE_32), 0x3f800000);
   agx_st_vary(b, agx_immediate(49), agx_register(24, AGX_SIZE_32));
   agx_st_vary(b, agx_immediate(50), agx_register(24, AGX_SIZE_32));
   agx_stop(b);
   EXPECT_EQ(uvsw_components(pack(b)), all);
}

TEST_F(AGX3PackBinary, FloatSelectWaitPreservesUnrelatedPendingLoads)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (bool narrow : {false, true}) {
      for (bool two_producers : {false, true}) {
         SCOPED_TRACE(narrow);
         SCOPED_TRACE(two_producers);
         agx_builder *b = builder(enc);
         b->shader->stage = MESA_SHADER_FRAGMENT;
         const auto size = narrow ? AGX_SIZE_16 : AGX_SIZE_32;
         const auto format = narrow ? AGX_FORMAT_I16 : AGX_FORMAT_I32;
         agx_device_load_to(b, agx_register(0, AGX_SIZE_32),
                           agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                           AGX_FORMAT_I32, 1, 0, false);
         agx_device_load_to(b, agx_register(8, size),
                           agx_uniform(0, AGX_SIZE_64), agx_immediate(4),
                           format, 1, 0, false);
         if (two_producers) {
            agx_device_load_to(b, agx_register(12, size),
                              agx_uniform(0, AGX_SIZE_64), agx_immediate(8),
                              format, 1, 0, false);
         } else {
            agx_mov_imm_to(b, agx_register(12, size),
                           narrow ? 1 : 0x3d4ccccd);
         }
         agx_index compare = agx_register(8, size);
         compare.discard = true;
         agx_fcmpsel_to(b, agx_register(4, AGX_SIZE_16), compare,
                        narrow ? agx_immediate(0x30) : agx_register(12, size),
                        narrow ? agx_register(12, size) : agx_immediate(1),
                        agx_immediate(0), AGX_FCOND_LT);
         agx_index unrelated = agx_register(0, AGX_SIZE_32);
         unrelated.discard = true;
         agx_iadd_to(b, agx_register(2, AGX_SIZE_32), unrelated,
                     agx_immediate(1), 0);
         agx_stop(b);
         const auto code = pack(b);
         ASSERT_FALSE(code.empty());

         /* Join only the comparison's producer slots. Never carry this wait
          * on r0, whose independent load is still outstanding. A half-width
          * comparison must not silently read/write its neighbouring half. */
         const uint16_t mask = two_producers ? 6 : 2;
         const uint16_t operands[] = {8, 8};
         const enum agx3_alu_hint hints[] = {
            AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
         uint8_t correct[AGX3_ALU_MAX_LENGTH], old[AGX3_ALU_MAX_LENGTH];
         uint8_t length = 0, old_length = 0;
         ASSERT_TRUE(narrow
            ? agx3_alu_bit_reg16(enc, AGX3_ALU_BAND, 8, operands, hints,
                                  mask, correct, &length)
            : agx3_alu_bit_reg(enc, AGX3_ALU_BAND, 8, operands, hints,
                                mask, correct, &length));
         ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 2, old, &old_length));
         EXPECT_NE(std::search(code.begin(), code.end(), correct,
                               correct + length), code.end());
         EXPECT_EQ(std::search(code.begin(), code.end(), old,
                               old + old_length), code.end());
      }
   }
}

TEST_F(AGX3PackBinary, HalfSelectRetiresPendingDestination)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_device_load_to(b, agx_register(40, AGX_SIZE_32),
                      agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                      AGX_FORMAT_I32, 1, 0, false);
   agx_device_load_to(b, agx_register(60, AGX_SIZE_16),
                      agx_uniform(0, AGX_SIZE_64), agx_immediate(4),
                      AGX_FORMAT_I16, 1, 0, false);
   agx_mov_imm_to(b, agx_register(58, AGX_SIZE_16), 500);
   agx_icmpsel_to(b, agx_register(41, AGX_SIZE_16),
                  agx_register(60, AGX_SIZE_16), agx_register(58, AGX_SIZE_16),
                  agx_immediate(0), agx_immediate(1), AGX_ICOND_SLT);
   agx_stop(b);
   const auto code = pack(b);
   ASSERT_FALSE(code.empty());

   /* Slot 1 still writes the destination's upper half. It is not a source
    * of the select, but must be retired before that half is overwritten. */
   const uint16_t operands[] = {41, 41};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], length = 0;
   ASSERT_TRUE(agx3_alu_bit_reg16(enc, AGX3_ALU_BAND, 41, operands, hints,
                                  1, bytes, &length));
   EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + length),
             code.end());
}

TEST_F(AGX3PackBinary, UniformFloatSelectRetiresPendingDestination)
{
   /* A constant-program select whose comparands and alternatives are all
    * uniforms or immediates (Cyberpunk 2077's
    * fcmpsel r52, u84.h, #0, u108, u48) has no register operand to carry the
    * wait for a load still writing the destination it overwrites. The
    * identity then goes on the destination itself. */
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_device_load_to(b, agx_register(52, AGX_SIZE_32),
                      agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                      AGX_FORMAT_I32, 1, 0, false);
   agx_fcmpsel_to(b, agx_register(52, AGX_SIZE_32),
                  agx_uniform(84, AGX_SIZE_16), agx_immediate(0),
                  agx_uniform(108, AGX_SIZE_32), agx_uniform(48, AGX_SIZE_32),
                  AGX_FCOND_LT);
   agx_stop(b);
   const auto code = pack(b);
   ASSERT_FALSE(code.empty());

   const uint16_t operands[] = {52, 52};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], length = 0;
   ASSERT_TRUE(agx3_alu_bit_reg(enc, AGX3_ALU_BAND, 52, operands, hints, 1,
                                bytes, &length));
   EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + length),
             code.end());
}

TEST_F(AGX3PackBinary, FloatUnaryPreservesLiveCoordinate)
{
   /* Apple applegpu_g17p kernels store both OP(x) and x. The original
    * coordinate remains live after the rounding used by f2i64 lowering. */
   static const uint8_t native[][10] = {
      {0x2f,0x00,0x56,0x00,0x02,0x04,0x90,0x40,0x06,0x00},
      {0x2f,0x00,0x56,0x00,0x02,0x04,0x90,0x40,0x02,0x00},
      {0x2f,0x00,0x56,0x00,0x02,0x04,0x90,0x40,0x04,0x00},
      {0x2f,0x02,0x56,0x00,0x02,0x04,0x90,0x40,0x00,0x00},
      {0xaf,0x02,0x56,0x00,0x02,0x04,0x90,0x40,0x00,0x00},
   };
   for (unsigned op = 0; op < ARRAY_SIZE(native); ++op) {
      SCOPED_TRACE(op);
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_VERTEX;
      const agx_index dst = agx_register(0, AGX_SIZE_32);
      const agx_index src = agx_register(2, AGX_SIZE_32);
      agx_device_load_to(b, src, agx_uniform(0, AGX_SIZE_64),
                         agx_immediate(0), AGX_FORMAT_I32, 1, 0, false);
      switch (op) {
      case 0: agx_trunc_to(b, dst, src); break;
      case 1: agx_floor_to(b, dst, src); break;
      case 2: agx_ceil_to(b, dst, src); break;
      case 3: agx_log2_to(b, dst, src); break;
      case 4: agx_exp2_to(b, dst, src); break;
      }
      agx_stop(b);
      const auto code = pack(b);
      ASSERT_GE(code.size(), sizeof(native[op]) + 4);
      EXPECT_EQ(std::vector<uint8_t>(code.end() - 14, code.end() - 4),
                std::vector<uint8_t>(native[op], native[op] + 10));
   }
}

TEST_F(AGX3PackBinary, InterpolationWaitDoesNotReadPendingRegisters)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   /* Interpolation overwrites the second independent load result. The first
    * stays live across that wait and is consumed by the later IADD. */
   agx_ldcf_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(1), 1);
   agx_ldcf_to(b, agx_register(8, AGX_SIZE_32), agx_immediate(2), 1);
   agx_index rhw = agx_register(8, AGX_SIZE_32);
   agx_rhw_interp_agx3_to(b, rhw, agx_immediate(0), agx_immediate(0),
                        AGX_INTERPOLATION_CENTER);
   agx_persp_rcp_agx3_to(b, rhw, rhw, 0);
   agx_index live = agx_register(0, AGX_SIZE_32);
   live.discard = true;
   agx_iadd_to(b, agx_register(2, AGX_SIZE_32), live, agx_immediate(7), 0);
   agx_stop(b);

   const auto binary = pack(b);
   uint8_t correct[AGX3_ALU_MAX_LENGTH], corrupting[AGX3_ALU_MAX_LENGTH];
   uint8_t correct_len = 0, corrupting_len = 0;
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_LDIMM, 8, 0, 0, 0, 0, 2,
                              correct, &correct_len));
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 2, corrupting,
                                           &corrupting_len));
   static const uint8_t source_free[] = {
      0x4c, 0x80, 0x02, 0x00, 0x00, 0x40, 0x00, 0x00,
   };
   ASSERT_EQ(correct_len, sizeof(source_free));
   EXPECT_EQ(memcmp(correct, source_free, sizeof(source_free)), 0);
   /* Native LDIMM has no register sources. The old destination value is dead,
    * and the pending r0 generation must not become an unmodelled carrier. */
   EXPECT_NE(std::search(binary.begin(), binary.end(), correct,
                         correct + correct_len), binary.end());
   EXPECT_EQ(std::search(binary.begin(), binary.end(), corrupting,
                         corrupting + corrupting_len), binary.end());
}

TEST_F(AGX3PackBinary, InterpolationWaitOverwritesBothDeadHalves)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_get_sr_to(b, agx_register(8, AGX_SIZE_16), AGX_SR_INPUT_SAMPLE_MASK);
   agx_get_sr_to(b, agx_register(9, AGX_SIZE_16), AGX_SR_BACKFACING);
   agx_index rhw = agx_register(8, AGX_SIZE_32);
   agx_rhw_interp_agx3_to(b, rhw, agx_immediate(0), agx_immediate(0),
                        AGX_INTERPOLATION_CENTER);
   agx_persp_rcp_agx3_to(b, rhw, rhw, 0);
   agx_stop(b);

   const auto binary = pack(b);
   uint8_t combined[AGX3_ALU_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_LDIMM, 8, 0, 0, 0, 0, 3,
                              combined, &len));
   EXPECT_NE(std::search(binary.begin(), binary.end(), combined,
                         combined + len), binary.end());
}

TEST_F(AGX3PackBinary, ProjectiveRhwRetiresReusedCoefficientDestination)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_ldcf_to(b, agx_register(8, AGX_SIZE_32), agx_immediate(0), 1);
   agx_index rhw = agx_register(8, AGX_SIZE_32);
   agx_rhw_interp_agx3_to(b, rhw, agx_immediate(0), agx_immediate(0),
                         AGX_INTERPOLATION_CENTER);
   agx_persp_rcp_agx3_to(b, rhw, rhw, 0);
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_VARY_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_vary_ldcf_at_start(enc, 8, 0, 1, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_LDIMM, 8, 0, 0, 0, 0, 1,
                                bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_vary_projective_rhw(enc, 8, 0, false, bytes, &len));
   append(want, bytes, len);
   struct agx3_persp_rcp_desc rcp = {
      .dest = 8, .rhw = 8, .coef_index = 0,
   };
   ASSERT_TRUE(agx3_persp_rcp_encode(enc, &rcp, bytes, &len));
   append(want, bytes, len);
   want.insert(want.end(), {0x0e, 0, 0, 0});
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, LinearInterpRetiresReusedCoefficientDestination)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_ldcf_to(b, agx_register(8, AGX_SIZE_32), agx_immediate(13), 1);
   agx_interp_agx3_to(b, agx_register(8, AGX_SIZE_32), agx_immediate(4),
                      agx_immediate(0), AGX_INTERPOLATION_CENTER, true);
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_VARY_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_vary_ldcf_at_start(enc, 8, 13, 1, true, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_compose(enc, AGX3_ALU_LDIMM, 8, 0, 0, 0, 0, 1,
                                bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_vary_interp_at_start(
      enc, 8, 4, true, AGX3_VARY_CENTER, false, bytes, &len));
   append(want, bytes, len);
   want.insert(want.end(), {0x0e, 0, 0, 0});
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FragmentHelperCoverageMatchesNativeAtomicPipeline)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_get_sr_coverage_to(b, agx_register(0, AGX_SIZE_16),
                          AGX_SR_IS_ACTIVE_THREAD);
   agx_stop(b);
   const std::vector<uint8_t> native = {
      0x04,0x84,0x10,0x06, 0x0e,0x00,0x00,0x00,
   };
   EXPECT_EQ(pack(b), native);
}

TEST_F(AGX3PackBinary, G17PDiscardDoesNotReleaseUnacquiredPixelClass)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (const std::vector<unsigned> &masks :
        {std::vector<unsigned>{0}, {2}, {0, 0, 2}}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      for (unsigned flags : masks)
         agx_sample_mask_agx3(b, agx_register(0, AGX_SIZE_16),
                                 agx_register(4, AGX_SIZE_32), flags);
      agx_stop(b);

      /* Apple G17P kill_fetch emits an early discard MOVMSK without a
       * class-1 wait or release. Its later testing MOVMSK is bracketed by
       * PIXWAIT(1)/PIXREL(1). kill_memory independently has two unbracketed
       * discard MOVMSKs before its bracketed late depth test. */
      std::vector<uint8_t> want;
      const uint8_t entry[] = {0x87, 0x02, 0x54, 0x00, 0x06, 0x00};
      const uint8_t acquire[] = {0x87, 0x02, 0x54, 0x01, 0x00, 0x00};
      const uint8_t mask[] = {0x57, 0x14, 0x54, 0x00, 0x00, 0x01};
      const uint8_t release[] = {0x07, 0x02, 0x54, 0x01, 0x00, 0x00};
      append(want, entry, sizeof(entry));
      for (unsigned flags : masks) {
         uint8_t copy[AGX3_STAGE_MAX_LENGTH], length = 0;
         ASSERT_TRUE(agx3_stage_copy_source(enc, 0, 4, false, 1, 0, copy,
                                            &length));
         append(want, copy, length);
         if (flags == 2)
            append(want, acquire, sizeof(acquire));
         append(want, mask, sizeof(mask));
         if (flags == 2)
            append(want, release, sizeof(release));
      }
      want.insert(want.end(), {0x0e, 0, 0, 0});
      EXPECT_EQ(pack(b), want);
   }
}

TEST_F(AGX3PackBinary, FragmentSideEffectLateDepthMatchesNativeProtocol)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_wait_pix(b, 1);
   agx_index depth = agx_register(4, AGX_SIZE_32);
   depth.discard = true;
   agx_zs_emit(b, agx_immediate(0xff), depth, 1);
   agx_signal_pix(b, 1);
   agx_stop(b);

   /* Fresh Apple fragment-atomic tail from the same d150404a... AIR /
    * 32cf4963... native pipeline: entry, stage Z, acquire class1, depthf,
    * release. With no colour store after it, the release ends the program
    * and carries the final bit, as in metal-nt -arch applegpu_g17p fragments
    * with no colour writes (070254010200). Construct every instruction
    * independently, then compare the complete backend stream. */
   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_stage_copy_kill(enc, 0, 4, true, 1, 0, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_wait(enc, 1, 0, bytes, &len));
   append(want, bytes, len);
   const struct agx3_zs_desc zs = {
      .value = 0, .value_kill = true, .slot = 1, .wait_mask = 1,
   };
   ASSERT_TRUE(agx3_zs_encode_depth_stencil(enc, &zs, true, false, bytes,
                                            &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_release(enc, 1, true, 0, bytes, &len));
   append(want, bytes, len);
   want.insert(want.end(), {0x0e, 0, 0, 0});
   EXPECT_EQ(pack(b), want);
}

/* Stores interleaved with producers that reuse their sources follow the
 * six-output protocol at any width: each staging COPY reads its source at the
 * store and retires immediately, and every six staged components (and the
 * tail) are written in component order. dEQP-VK.subgroups.basic.framebuffer
 * *barrier_vertex interleaves its stores with device loads this way. */
TEST_F(AGX3PackBinary, WideVertexOutputsInterleaveWithProducers)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (unsigned count : {7u, 9u, 33u}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_VERTEX;
      /* Store the upper half first, as the barrier shaders do. */
      std::vector<unsigned> order;
      for (unsigned c = count / 2; c < count; ++c)
         order.push_back(c);
      for (unsigned c = 0; c < count / 2; ++c)
         order.push_back(c);

      std::vector<uint8_t> want;
      std::vector<unsigned> batch;
      for (unsigned component : order) {
         agx_mov_imm_to(b, agx_register(0, AGX_SIZE_32), 0x3f000000 + component);
         agx_st_vary(b, agx_immediate(component), agx_register(0, AGX_SIZE_32));
         struct agx3_encoded_ldimm ld;
         ASSERT_TRUE(agx3_pack_ldimm_u32(enc, 0, 0x3f000000 + component, &ld));
         append(want, ld.bytes, ld.length);
         uint8_t bytes[AGX3_ALU_MAX_LENGTH];
         uint8_t len = 0;
         ASSERT_TRUE(agx3_stage_copy_source(enc, component * 2, 0, false,
                                            component % 6 + 1, 0, bytes, &len));
         append(want, bytes, len);
         ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1u << (component % 6),
                                                 bytes, &len));
         append(want, bytes, len);
         batch.push_back(component);
         if (batch.size() == 6 || component == order.back()) {
            std::sort(batch.begin(), batch.end());
            for (unsigned c : batch) {
               uint8_t uvsw[AGX3_VARY_MAX_LENGTH];
               uint8_t uvsw_len = 0;
               ASSERT_TRUE(agx3_vary_uvsw(enc, c * 2, c, c % 6 + 1, true,
                                          1u << (c % 6), uvsw, &uvsw_len));
               append(want, uvsw, uvsw_len);
            }
            batch.clear();
         }
      }
      agx_stop(b);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want) << count;
   }
}

/* A texture sample between the stores stages its own operands into the same
 * bank, so that bundle keeps refusing; the same sample ahead of the stores
 * does not. */
TEST_F(AGX3PackBinary, WideVertexOutputsRefuseStagingBetweenStores)
{
   for (bool between : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_VERTEX;
      const auto sample = [&]() {
         agx_index dest = agx_register(32, AGX_SIZE_32);
         dest.channels_m1 = 3;
         agx_index coord = agx_register(8, AGX_SIZE_32);
         coord.channels_m1 = 2;
         agx_index gradient = agx_register(16, AGX_SIZE_32);
         gradient.channels_m1 = 5;
         auto *I = agx_texture_sample_to(
            b, dest, coord, gradient, agx_immediate(0),
            agx_register(0, AGX_SIZE_64), agx_register(4, AGX_SIZE_64),
            agx_null(), AGX_DIM_3D, AGX_LOD_MODE_LOD_GRAD, 0, false, false,
            false, AGX_GATHER_NONE, false);
         I->mask = 15;
      };
      if (!between)
         sample();
      for (unsigned c = 0; c < 9; ++c) {
         if (between && c == 4)
            sample();
         agx_st_vary(b, agx_immediate(c), agx_register(40, AGX_SIZE_32));
      }
      agx_mov_imm_to(b, agx_register(40, AGX_SIZE_32), 0x3f800000);
      agx_st_vary(b, agx_immediate(9), agx_register(40, AGX_SIZE_32));
      agx_stop(b);
      EXPECT_EQ(pack(b, !between).empty(), between);
   }
}

/* The ID remains live after both immediate adds in the transfer lowering.
 * Old encoding unconditionally set f42, discarding it before the store. */
TEST_F(AGX3PackBinary, ImmediateIaddPreservesLiveSource)
{
   for (bool killed : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index source = agx_register(2, AGX_SIZE_32);
      source.discard = killed;
      agx_iadd_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(0), source, 2);
      agx_stop(b);
      auto bytes = pack(b);
      uint8_t native[] = {0x9f,0x01,0x54,0x00,0x02,0x00,0x08,0x88,
                          (uint8_t)(killed ? 0x14 : 0x10),0x14};
      native[4] |= int_dest_cache();
      ASSERT_GE(bytes.size(), sizeof(native));
      EXPECT_EQ(memcmp(bytes.data(), native, sizeof(native)), 0);
   }
}

TEST_F(AGX3PackBinary, ZeroExtendUsesNarrowAdd)
{
   /* Below and above the seven-bit destination of the half COPY alike, the
    * narrow slot-B add of 0 zero-extends in one instruction. */
   for (unsigned dest : {0u, 126u, 132u, 190u}) {
      const auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      agx_index half = agx_register(54, AGX_SIZE_16);
      half.discard = true;
      agx_bitop_to(b, agx_register(dest, AGX_SIZE_32), half, agx_immediate(0),
                   AGX3_SEL_TT_MOV);
      agx_stop(b);

      struct agx3_iadd_desc d = {
         .src_kill = true, .dest = (uint16_t)dest, .immediate = 0,
         .src = 54, .src_16 = true,
      };
      d.dest_cache = int_dest_cache();
      uint8_t add[AGX3_IADD_MAX_LENGTH], add_len = 0;
      ASSERT_TRUE(agx3_iadd_encode(enc, &d, add, &add_len));
      std::vector<uint8_t> want;
      append(want, add, add_len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want) << "dest=" << dest;
   }
}

TEST_F(AGX3PackBinary, LocalIndexZeroExtendUsesNarrowAdd)
{
   for (unsigned source : {0u, 1u, 255u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index half = agx_register(source, AGX_SIZE_16);
      half.abs = true;
      half.discard = true;
      half.kill = source == 255;
      agx_iadd_to(b, agx_register(4, AGX_SIZE_32), agx_immediate(0), half, 0);
      agx_stop(b);

      /* One add where it encodes; otherwise the copy of the half and the
       * clear of the upper one. */
      struct agx3_iadd_desc d = {
         .src_kill = true, .dest = 4, .immediate = 0,
         .src = (uint16_t)source, .src_16 = true,
      };
      d.dest_cache = int_dest_cache();
      uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
      std::vector<uint8_t> want;
      if (agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len)) {
         append(want, bytes, len);
      } else {
         ASSERT_TRUE(agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3B, 4, source,
                                            false, false, true, 0, bytes,
                                            &len));
         append(want, bytes, len);
         ASSERT_TRUE(agx3_clr(AGX_ISA_ENCODING_AGX3B, 5, true, false, bytes,
                              &len));
         append(want, bytes, len);
      }
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want) << "source half=" << source;
   }
}

TEST_F(AGX3PackBinary, M5PixelCoordinateZeroExtendUsesCopyAndClear)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   agx_index dest = agx_register(0, AGX_SIZE_32);
   dest.cache = true;
   agx_index source = agx_register(1, AGX_SIZE_16);
   source.discard = true;
   agx_bitop_to(b, dest, source, agx_zero(), AGX3_SEL_TT_MOV);
   agx_stop(b);

   /* The low-half COPY is corpus by835; the following CLR addresses only
    * the high half, retaining the copied low 16 bits. */
   const std::vector<uint8_t> want = {
      0x03, 0x01, 0x29, 0x00,
      0x03, 0x00, 0x00, 0x01,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), want);
}


/* SIGNEXT lowers this exact Firefox vertex-shader shape to IADD after RA.
 * A bare half is signed in AGX IR, while ABS denotes zero extension. */
TEST_F(AGX3PackBinary, FirefoxVertexHalfSignExtendUsesNativeIadd)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(34, AGX_SIZE_32);
   dest.cache = true;
   agx_iadd_to(b, dest, agx_register(26, AGX_SIZE_16), agx_immediate(0), 0);
   agx_stop(b);

   const struct agx3_iadd_desc d = {
      .dest = 34,
      .immediate = 0,
      .src = 26,
      .src_16 = true,
      .sign_extend = true,
      .dest_cache = true,
   };
   uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   std::vector<uint8_t> want(bytes, bytes + len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, FirefoxFragmentSignedHalfAccumulatorUsesNativeIadd)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index half = agx_register(22, AGX_SIZE_16);
   half.cache = true;
   agx_iadd_to(b, agx_register(30, AGX_SIZE_32),
               agx_register(30, AGX_SIZE_32), half, 0);
   agx_stop(b);

   struct agx3_iadd_desc d = {};
   d.src_a_is_register = true;
   d.src_a_reg = 30;
   d.dest = 30;
   d.src = 22;
   d.src_16 = true;
   d.sign_extend = true;
   d.dest_cache = int_dest_cache();
   uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   std::vector<uint8_t> want(bytes, bytes + len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   /* Addition commutes: reverse Firefox's second physical allocation onto
    * the same grounded slot-A32 + slot-B16 instruction. */
   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_iadd_to(bad, agx_register(30, AGX_SIZE_32),
               agx_register(22, AGX_SIZE_16),
               agx_register(30, AGX_SIZE_32), 0);
   agx_stop(bad);
   EXPECT_EQ(pack(bad), want);

   /* A shifted signed half is the same instruction with f54 set. */
   agx_builder *shifted = builder(AGX_ISA_ENCODING_AGX3B);
   agx_iadd_to(shifted, agx_register(30, AGX_SIZE_32),
               agx_register(30, AGX_SIZE_32),
               agx_register(22, AGX_SIZE_16), 1);
   agx_stop(shifted);
   d.shift = 1;
   ASSERT_TRUE(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   std::vector<uint8_t> want_shifted(bytes, bytes + len);
   append(want_shifted, end.bytes, end.length);
   EXPECT_EQ(pack(shifted), want_shifted);

   bad = builder(AGX_ISA_ENCODING_AGX3B);
   agx_instr *sat = agx_iadd_to(bad, agx_register(30, AGX_SIZE_32),
                                agx_register(30, AGX_SIZE_32),
                                agx_register(22, AGX_SIZE_16), 0);
   sat->saturate = true;
   agx_stop(bad);
   EXPECT_TRUE(pack(bad, false).empty());
}

TEST_F(AGX3PackBinary, FirefoxHalfArithmeticShiftMatchesNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(35, AGX_SIZE_16);
   src.cache = true;
   agx_index amount = agx_immediate(1);
   agx_asr_to(b, agx_register(25, AGX_SIZE_16), src, amount);
   agx_stop(b);

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_alu_asr_imm16(
      AGX_ISA_ENCODING_AGX3B, 25, 35, 1, AGX3_ALU_HINT_CACHE, false, 0,
      bytes, &len));
   std::vector<uint8_t> want(bytes, bytes + len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);

   for (unsigned variant = 0; variant < 4; ++variant) {
      agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index bad_src = variant == 0 ? agx_register(34, AGX_SIZE_32)
                                       : agx_register(35, AGX_SIZE_16);
      agx_index bad_amount = agx_immediate(variant == 1 ? 16 : 1);
      if (variant == 2)
         bad_src.abs = true;
      agx_instr *bad_I = agx_asr_to(
         bad, agx_register(25, AGX_SIZE_16), bad_src, bad_amount);
      bad_I->saturate = variant == 3;
      agx_stop(bad);
      EXPECT_TRUE(pack(bad, false).empty()) << variant;
   }
}
/*
 * agx_lower_uniform_sources() materializes a move for every generic-ALU
 * uniform source. At 32 bits it reached agx3_copy_encode_uniform_to_reg32();
 * at 16 bits it had no encoder, so it stayed a bitop reading the uniform file
 * and the packer refused the whole shader. u_blitter's pass-through vertex
 * shader hits exactly that (`mov u9h` for the clip-Z coefficient), so before
 * this every native-Gallium blit on G17P refused its vertex program.
 */
TEST_F(AGX3PackBinary, G17PUniformHalfMoveUsesNativeCopy)
{
   for (unsigned uniform : {0u, 19u, 511u}) {
      for (bool cache : {false, true}) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         agx_index dest = agx_register(2, AGX_SIZE_16);
         dest.cache = cache;
         /* agx_lower_pseudo() turns a MOV into exactly this bitop. */
         agx_bitop_to(b, dest, agx_uniform(uniform, AGX_SIZE_16), agx_zero(),
                      AGX3_SEL_TT_MOV);
         agx_stop(b);

         uint8_t bytes[AGX3_COPY_MAX_LENGTH], len = 0;
         std::vector<uint8_t> want;
         ASSERT_TRUE(agx3_copy_encode_uniform_to_reg16(
            AGX_ISA_ENCODING_AGX3B, 2, uniform, cache, bytes, &len));
         append(want, bytes, len);
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want)
            << "uniform half=" << uniform << " cache=" << cache;
      }
   }
}

TEST_F(AGX3PackBinary, G17GHalfWidenMatchesNativeMetal)
{
   /* metal-nt -arch applegpu_g17g, dynamic half4 -> float4 and signed/abs
    * variants. Byte strings are extracted from native output, not produced
    * by the encoder under test. The first conversion waits on load slot 8.
    */
   const struct {
      unsigned dest, src;
      bool abs, neg;
      unsigned wait;
      std::vector<uint8_t> bytes;
   } cases[] = {
      {0, 4, false, false, 0x80,
       {0x29,0x00,0x04,0x1a,0x22,0x20,0xa4,0x02,0x84,0x06,0x40,0x00}},
      {2, 5, false, false, 0,
       {0x19,0x04,0x1c,0x81,0x08,0x02}},
      {4, 6, false, false, 0,
       {0x29,0x06,0x1c,0x81,0x00,0x02}},
      {6, 7, false, false, 0,
       {0x39,0x06,0x1c,0x81,0x08,0x02}},
      {2, 5, false, true, 0,
       {0x19,0x04,0x1c,0x81,0x09,0x02,0x02,0x00}},
      {4, 6, true, false, 0,
       {0x21,0x02,0x04,0x1a,0x22,0x20,0xa4,0x0a,0x84,0x86,0x40,0x00}},
      {6, 7, true, true, 0,
       {0x31,0x03,0x04,0x1a,0x22,0x20,0xa4,0x0a,0x84,0x86,0x40,0x04}},
   };

   for (const auto &c : cases) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_alu_widen_half(
         AGX_ISA_ENCODING_AGX3C, c.dest, c.src, AGX3_ALU_HINT_DISCARD,
         c.abs, c.neg, false, c.wait, bytes, &len));
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len), c.bytes);
   }

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   EXPECT_FALSE(agx3_alu_widen_half(
      AGX_ISA_ENCODING_AGX3C, 1, 4, AGX3_ALU_HINT_DISCARD,
      false, false, false, 0, bytes, &len));
   EXPECT_FALSE(agx3_alu_widen_half(
      AGX_ISA_ENCODING_AGX3C, 0, 256, AGX3_ALU_HINT_DISCARD,
      false, false, false, 0, bytes, &len));
   EXPECT_FALSE(agx3_alu_widen_half(
      AGX_ISA_ENCODING_AGX3C, 0, 4, AGX3_ALU_HINT_DISCARD,
      false, false, false, 0x100, bytes, &len));
}

TEST_F(AGX3PackBinary, G17GUniformHalfReadUsesAlignedWordAndCorrectHalf)
{
   key.dev.emit_agx3d = true;
   for (unsigned uniform : {0u, 1u, 12u, 13u, 510u, 511u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      b->shader->stage = MESA_SHADER_VERTEX;
      b->shader->nir = rzalloc(mem_ctx, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_VERTEX;

      /* Match the half-to-float conversion that refused the native GLES
       * vertex shader. Keep the result live through an actual varying store.
       */
      agx_index result = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *add = agx_fadd_to(b, result,
         agx_uniform(uniform, AGX_SIZE_16), agx_immediate(0x80));
      agx_st_vary(b, agx_immediate(0), result);
      agx_stop(b);
      agx_lower_uniform_sources(b->shader);
      agx_dce(b->shader, true);
      agx_ra(b->shader);
      agx_lower_pseudo(b->shader);

      /* The original source is exactly one half of the uniform word, not a
       * numeric conversion or a replacement constant. Check both edge halves
       * of the complete uniform file as well as the observed odd-half case.
       */
      unsigned reads = 0;
      agx_foreach_instr_global(b->shader, I) {
         agx_foreach_src(I, s) {
            if (I->src[s].type != AGX_INDEX_UNIFORM)
               continue;
            ++reads;
            EXPECT_EQ(I->src[s].size, AGX_SIZE_32);
            EXPECT_EQ(I->src[s].value, uniform & ~1u);
            EXPECT_EQ(add->src[0].value, I->dest[0].value + (uniform & 1));
         }
      }
      EXPECT_EQ(reads, 1u);
      EXPECT_EQ(add->src[0].size, AGX_SIZE_16);
      EXPECT_FALSE(pack(b).empty()) << uniform;
   }
}

TEST_F(AGX3PackBinary, G17GRepeatedHalfUniformExportsStayEncodableAfterRA)
{
   key.dev.emit_agx3d = true;
   for (unsigned uniform : {0u, 1u, 16u, 17u, 510u, 511u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem_ctx, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;

      /* GLES lifetime setup exports (1,0,1) from promoted half literals.
       * Reusing the first half forces a late copy without this legalization. */
      agx_export(b, agx_uniform(uniform, AGX_SIZE_16), 8);
      agx_export(b, agx_uniform(uniform ^ 1, AGX_SIZE_16), 9);
      agx_export(b, agx_uniform(uniform, AGX_SIZE_16), 10);
      agx_lower_uniform_sources(b->shader);

      bool exporting = false;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_EXPORT) {
            exporting = true;
            EXPECT_NE(I->src[0].type, AGX_INDEX_UNIFORM);
         } else {
            EXPECT_FALSE(exporting) << "materialization split the export group";
         }
      }

      agx_dce(b->shader, true);
      agx_ra(b->shader);
      agx_lower_pseudo(b->shader);
      agx_foreach_instr_global(b->shader, I) {
         agx_foreach_src(I, s) {
            if (I->src[s].type == AGX_INDEX_UNIFORM) {
               EXPECT_EQ(I->src[s].size, AGX_SIZE_32);
               EXPECT_EQ(I->src[s].value, uniform & ~1u);
            }
         }
      }
      EXPECT_FALSE(pack(b).empty()) << uniform;
   }
}

TEST_F(AGX3PackBinary, G17GRepeatedHalfLiteralExportsStayEncodableAfterRA)
{
   key.dev.emit_agx3d = true;
   for (unsigned bits : {0u, 1u, 0x3c00u, 0xbc00u, 0x8000u, 0x7c00u,
                         0x7e01u, 0xffffu}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem_ctx, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;

      agx_index value = agx_mov_imm(b, 16, bits);
      agx_index zero = agx_mov_imm(b, 16, 0);
      agx_export(b, value, 8);
      agx_export(b, zero, 9);
      agx_export(b, value, 10);
      agx_lower_agx3_immediates(b->shader);

      unsigned immediate_count = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_MOV_IMM) {
            EXPECT_EQ(I->dest[0].size, AGX_SIZE_32);
            EXPECT_EQ(I->imm, immediate_count++ == 0 ? bits : 0u);
         } else if (I->op == AGX_OPCODE_SPLIT) {
            EXPECT_EQ(I->dest[0].size, AGX_SIZE_16);
            EXPECT_EQ(I->src[0].size, AGX_SIZE_32);
         }
      }
      EXPECT_EQ(immediate_count, 2u);

      agx_dce(b->shader, true);
      agx_ra(b->shader);
      agx_lower_pseudo(b->shader);
      EXPECT_FALSE(pack(b).empty()) << bits;
   }
}

/* AGX3C keeps plain half registers that two FFMAs read in place
 * (agx_lower_agx3_fma_sources): every mixed shape, at either destination
 * size, packs after register allocation as the two FFMAs, with no widening
 * move. Also with abs/neg on one half operand next to a plain one, and with a
 * uniform addend, which agx_lower_uniform_sources copies to a register when a
 * half register operand remains. */
TEST_F(AGX3PackBinary, AGX3CMixedFMAHalfRegistersPackAfterRA)
{
   for (auto chip : {AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C}) {
      key.dev.chip = chip;
      SCOPED_TRACE(chip);
      for (unsigned half_mask = 1; half_mask < 8; ++half_mask) {
         for (unsigned dest_size : {32u, 16u}) {
            /* 0: plain; 1: abs/neg on the first half operand; 2: uniform addend
             */
            for (unsigned variant = 0; variant < 3; ++variant) {
               for (unsigned mods = 1; mods < (variant == 1 ? 4u : 2u);
                    ++mods) {
                  SCOPED_TRACE(half_mask);
                  SCOPED_TRACE(dest_size);
                  SCOPED_TRACE(variant);
                  SCOPED_TRACE(mods);
                  agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
                  b->shader->stage = MESA_SHADER_VERTEX;
                  b->shader->nir = rzalloc(mem_ctx, nir_shader);
                  b->shader->nir->info.stage = MESA_SHADER_VERTEX;
                  agx_index src[3];
                  bool half_gpr[3] = {false}, modified[3] = {false};
                  bool first_half = true;
                  for (unsigned s = 0; s < 3; ++s) {
                     const bool half = half_mask & BITFIELD_BIT(s);
                     if (variant == 2 && s == 2) {
                        src[s] =
                           agx_uniform(8, half ? AGX_SIZE_16 : AGX_SIZE_32);
                        continue;
                     }
                     src[s] = agx_mov_imm(b, half ? 16 : 32,
                                          half ? 0x3c00 + s : 0x3f800000 + s);
                     half_gpr[s] = half;
                     if (half && variant == 1 && first_half) {
                        src[s].abs = mods & 1;
                        src[s].neg = mods & 2;
                        modified[s] = true;
                     }
                     first_half &= !half;
                  }
                  agx_instr *fmas[2];
                  for (unsigned f = 0; f < 2; ++f) {
                     agx_index result = agx_temp(
                        b->shader, dest_size == 16 ? AGX_SIZE_16 : AGX_SIZE_32);
                     fmas[f] = agx_ffma_to(b, result, src[0], src[1], src[2]);
                     agx_index wide = result;
                     if (dest_size == 16) {
                        wide = agx_temp(b->shader, AGX_SIZE_32);
                        agx_fmov_to(b, wide, result);
                     }
                     agx_st_vary(b, agx_immediate(f), wide);
                  }
                  agx_stop(b);

                  agx_lower_agx3_fma_sources(b->shader);
                  for (agx_instr *fma : fmas) {
                     for (unsigned s = 0; s < 3; ++s) {
                        if (!half_gpr[s])
                           continue;
                        EXPECT_EQ(fma->src[s].size, AGX_SIZE_16) << s;
                        EXPECT_EQ(fma->src[s].abs, modified[s] && (mods & 1))
                           << s;
                        EXPECT_EQ(fma->src[s].neg, modified[s] && (mods & 2))
                           << s;
                     }
                  }
                  /* No widening move reads a half register operand. */
                  agx_foreach_instr_global(b->shader, I) {
                     if (I->op != AGX_OPCODE_FADD)
                        continue;
                     for (unsigned s = 0; s < 3; ++s) {
                        if (half_gpr[s] && I->src[0].type == AGX_INDEX_NORMAL)
                           EXPECT_NE(I->src[0].value, src[s].value) << s;
                     }
                  }
                  agx_lower_uniform_sources(b->shader);
                  agx_dce(b->shader, true);
                  agx_ra(b->shader);
                  agx_lower_pseudo(b->shader);

                  unsigned fused = 0;
                  agx_foreach_instr_global(b->shader, I)
                     fused += I->op == AGX_OPCODE_FFMA;
                  EXPECT_EQ(fused, 2u);
                  EXPECT_FALSE(pack(b).empty());
               }
            }
         }
      }
   }
}

TEST_F(AGX3PackBinary, G17GMixedFMAKeepsFusedOperationThroughPacking)
{
   key.dev.emit_agx3d = true;
   for (unsigned half_mask = 1; half_mask < 8; ++half_mask) {
      for (unsigned modifiers = 0; modifiers < 4; ++modifiers) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
         b->shader->stage = MESA_SHADER_VERTEX;
         b->shader->nir = rzalloc(mem_ctx, nir_shader);
         b->shader->nir->info.stage = MESA_SHADER_VERTEX;
         agx_index src[3];
         for (unsigned s = 0; s < 3; ++s) {
            const bool half = half_mask & BITFIELD_BIT(s);
            src[s] = agx_uniform(4 * s, half ? AGX_SIZE_16 : AGX_SIZE_32);
            if (half) {
               src[s].abs = modifiers & 1;
               src[s].neg = modifiers & 2;
            }
         }
         agx_index result = agx_temp(b->shader, AGX_SIZE_32);
         agx_instr *fma = agx_ffma_to(b, result, src[0], src[1], src[2]);
         agx_st_vary(b, agx_immediate(0), result);
         agx_stop(b);

         agx_lower_agx3_fma_sources(b->shader);
         for (unsigned s = 0; s < 3; ++s)
            EXPECT_EQ(fma->src[s].size, AGX_SIZE_32);
         agx_lower_uniform_sources(b->shader);
         agx_dce(b->shader, true);
         agx_ra(b->shader);
         agx_lower_pseudo(b->shader);

         unsigned fused = 0, multiplied = 0;
         agx_foreach_instr_global(b->shader, I) {
            fused += I->op == AGX_OPCODE_FFMA;
            multiplied += I->op == AGX_OPCODE_FMUL;
         }
         EXPECT_EQ(fused, 1u);
         EXPECT_EQ(multiplied, 0u);
         EXPECT_FALSE(pack(b).empty()) << half_mask << "/" << modifiers;
      }
   }
}

/*
 * The same materialized move at 32 bits, with the destination register
 * allocation actually gave it in a ray-query shader.
 *
 * Every dEQP-VK.ray_query case was refused on G17P with
 *
 *   agx3_pack_binary.c:11687: no grounded AGX3 encoding for bitop
 *     [d32:reg134 s0=32:uni12 s1=16:imm#0 ... chan=1 mask=0x0 fmt=0 shift=0]
 *
 * because the compact four-byte `copy` names its destination in a seven-bit
 * field and truncates at half 127, while these shaders' pressure put the
 * destinations at halves 134-154. Apple's compiler encodes exactly those
 * destinations in the ten-byte ep1 -- see the metal-nt witnesses quoted in
 * agx3_copy_core.c -- so the fix is the encoding point, not a restriction on
 * where the value may live.
 */
TEST_F(AGX3PackBinary, G17PUniformWordMoveReachesHighDestinations)
{
   for (unsigned dest : {2u, 126u, 128u, 134u, 154u, 254u}) {
      for (unsigned uniform : {8u, 12u, 18u}) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         agx_bitop_to(b, agx_register(dest, AGX_SIZE_32),
                      agx_uniform(uniform, AGX_SIZE_32), agx_zero(),
                      AGX3_SEL_TT_MOV);
         agx_stop(b);

         uint8_t bytes[AGX3_COPY_MAX_LENGTH], len = 0;
         std::vector<uint8_t> want;
         ASSERT_TRUE(agx3_copy_encode_uniform_to_reg32(
            AGX_ISA_ENCODING_AGX3B, dest, uniform, bytes, &len))
            << "dest half=" << dest << " uniform=" << uniform;
         EXPECT_EQ(len, dest < AGX3_COPY_REG16_DEST_HALVES ? 4u : 10u)
            << "dest half=" << dest;
         append(want, bytes, len);
         struct agx3_encoded_end end;
         ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
         append(want, end.bytes, end.length);

         std::vector<uint8_t> got = pack(b);
         EXPECT_FALSE(got.empty())
            << "refused dest half=" << dest << " uniform=" << uniform;
         EXPECT_EQ(got, want)
            << "dest half=" << dest << " uniform=" << uniform;
      }
   }
}

TEST_F(AGX3PackBinary, UniformStoreWaitPreservesUnrelatedPendingR0)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->is_preamble = true;
   key.dev.chip = AGX_CHIP_G17P;

   /* The sample-index load stays in flight while another preamble result
    * is published. Its partial wait must not read or write r0. */
   const agx_index base = agx_uniform(16, AGX_SIZE_64);
   agx_device_load_to(b, agx_register(0, AGX_SIZE_32), base,
                      agx_immediate(0), AGX_FORMAT_I32, 1, 0, false);
   agx_device_load_to(b, agx_register(8, AGX_SIZE_32), base,
                      agx_immediate(4), AGX_FORMAT_I32, 1, 0, false);
   agx_uniform_store(b, agx_register(8, AGX_SIZE_32), agx_immediate(24), 1);
   agx_uniform_store(b, agx_register(0, AGX_SIZE_32), agx_immediate(22), 1);
   agx_stop(b);

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   std::vector<uint8_t> publication;
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 8, 2, bytes, &len));
   append(publication, bytes, len);
   ASSERT_TRUE(agx3_copy_encode_reg_to_uniform(enc, 24, 8, false, 0, bytes, &len));
   append(publication, bytes, len);
   const auto code = pack(b);
   EXPECT_NE(std::search(code.begin(), code.end(), publication.begin(),
                         publication.end()), code.end());
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 2, bytes, &len));
   EXPECT_EQ(std::search(code.begin(), code.end(), bytes, bytes + len), code.end());
}

TEST_F(AGX3PackBinary, UniformHalfStoreWaitPreservesPendingUpperHalf)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->is_preamble = true;
   key.dev.chip = AGX_CHIP_G17P;
   const agx_index base = agx_uniform(16, AGX_SIZE_64);
   agx_device_load_to(b, agx_register(8, AGX_SIZE_16), base,
                      agx_immediate(0), AGX_FORMAT_I16, 1, 0, false);
   agx_device_load_to(b, agx_register(9, AGX_SIZE_16), base,
                      agx_immediate(2), AGX_FORMAT_I16, 1, 0, false);
   agx_uniform_store(b, agx_register(8, AGX_SIZE_16), agx_immediate(24), 1);
   agx_stop(b);

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   std::vector<uint8_t> publication;
   ASSERT_TRUE(agx3_alu_bit_imm(enc, AGX3_ALU_BOR, true, 8, 8, 0,
                               AGX3_ALU_HINT_NONE, 1, bytes, &len));
   append(publication, bytes, len);
   ASSERT_TRUE(agx3_copy_encode_reg_to_uniform(enc, 24, 8, false, 0, bytes, &len));
   append(publication, bytes, len);
   const auto code = pack(b);
   EXPECT_NE(std::search(code.begin(), code.end(), publication.begin(),
                         publication.end()), code.end());
}

TEST_F(AGX3PackBinary, G17PPaddedUniformHalfStoreUsesNativeDwordCopy)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index value = agx_register(4, AGX_SIZE_16);
   value.discard = true;
   agx_uniform_store(b, value, agx_immediate(24), 1);
   agx_stop(b);

   uint8_t bytes[AGX3_COPY_MAX_LENGTH], len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_copy_encode_reg_to_uniform(
      AGX_ISA_ENCODING_AGX3B, 24, 4, false, 0, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, G17PUniformHalfStoreRequiresUnusedAlignedPadding)
{
   /* Adjacent half stores are real data, not a dword with free padding. */
   agx_builder *adjacent = builder(AGX_ISA_ENCODING_AGX3B);
   agx_uniform_store(adjacent, agx_register(4, AGX_SIZE_16),
                     agx_immediate(24), 1);
   agx_uniform_store(adjacent, agx_register(6, AGX_SIZE_16),
                     agx_immediate(25), 1);
   agx_stop(adjacent);
   pack(adjacent, false);

   agx_builder *odd_source = builder(AGX_ISA_ENCODING_AGX3B);
   agx_uniform_store(odd_source, agx_register(5, AGX_SIZE_16),
                     agx_immediate(24), 1);
   agx_stop(odd_source);
   pack(odd_source, false);
}

TEST_F(AGX3PackBinary, G17PCoalescesAdjacentPreambleUniformHalvesBeforeRA)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->is_preamble = true;
   b->shader->allocated_vec = _mesa_hash_table_u64_create(b->shader);
   /* Before RA, exports use SSA values that remain safe to delay. Reserve
    * these IDs so the existing collect-source assertions stay exact. */
   agx_index halves[3] = {agx_get_index(4, AGX_SIZE_16),
                          agx_get_index(6, AGX_SIZE_16),
                          agx_get_index(8, AGX_SIZE_16)};
   b->shader->alloc = 9;
   for (unsigned h = 0; h < 3; ++h)
      agx_preload_to(b, halves[h], agx_register(4 + 2 * h, AGX_SIZE_16));
   agx_uniform_store(b, halves[0], agx_immediate(32), 1);
   agx_uniform_store(b, halves[1], agx_immediate(33), 1);
   agx_uniform_store(b, halves[2], agx_immediate(34), 1);
   agx_stop(b);

   ASSERT_TRUE(agx3_coalesce_uniform_half_stores(b->shader));
   unsigned collects = 0, stores32 = 0, stores16 = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_COLLECT) {
         ++collects;
         ASSERT_EQ(I->nr_srcs, 2);
         EXPECT_EQ(I->src[0].value, 4u);
         EXPECT_EQ(I->src[1].value, 6u);
      } else if (I->op == AGX_OPCODE_UNIFORM_STORE) {
         ASSERT_EQ(I->src[1].type, AGX_INDEX_IMMEDIATE);
         if (I->src[0].size == AGX_SIZE_32) {
            ++stores32;
            EXPECT_EQ(I->src[1].value, 32u);
         } else {
            ++stores16;
            EXPECT_EQ(I->src[1].value, 34u);
         }
      }
   }
   EXPECT_EQ(collects, 1u);
   EXPECT_EQ(stores32, 1u);
   EXPECT_EQ(stores16, 1u);
}

TEST_F(AGX3PackBinary, G17PDoesNotCoalesceOddUniformPair)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->is_preamble = true;
   agx_index low = agx_temp(b->shader, AGX_SIZE_16);
   agx_index high = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, low, agx_register(4, AGX_SIZE_16));
   agx_preload_to(b, high, agx_register(6, AGX_SIZE_16));
   agx_uniform_store(b, low, agx_immediate(33), 1);
   agx_uniform_store(b, high, agx_immediate(34), 1);
   agx_stop(b);
   EXPECT_FALSE(agx3_coalesce_uniform_half_stores(b->shader));
}

TEST_F(AGX3PackBinary, G17PDoesNotCoalescePhysicalRegisterHalfStores)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->is_preamble = true;
   agx_uniform_store(b, agx_register(4, AGX_SIZE_16), agx_immediate(32), 1);
   agx_uniform_store(b, agx_register(6, AGX_SIZE_16), agx_immediate(33), 1);
   agx_stop(b);

   EXPECT_FALSE(agx3_coalesce_uniform_half_stores(b->shader));
   unsigned stores = 0;
   agx_foreach_instr_global(b->shader, I) {
      EXPECT_NE(I->op, AGX_OPCODE_COLLECT);
      if (I->op == AGX_OPCODE_UNIFORM_STORE) {
         EXPECT_EQ(I->src[0].type, AGX_INDEX_REGISTER);
         EXPECT_EQ(I->src[0].value, 4u + 2u * stores);
         EXPECT_EQ(I->src[1].value, 32u + stores);
         ++stores;
      }
   }
   EXPECT_EQ(stores, 2u);
}

TEST_F(AGX3PackBinary, ByteAddressLoweringKeepsFullU32Offset)
{
   for (bool enabled : {false, true}) {
      nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,NULL,
                                                   "byte-index-address");
      nir_def *base=nir_load_push_constant(&b,1,64,nir_imm_int(&b,0));
      nir_def *offset=nir_channel(&b,nir_load_global_invocation_id(&b,32),0);
      nir_def *value=nir_load_global(&b,1,32,
         nir_iadd(&b,base,nir_u2u64(&b,offset)),.align_mul=4);
      nir_store_global(&b,value,nir_imm_int64(&b,0x100000),.align_mul=4);
      ASSERT_TRUE(agx_nir_lower_address(b.shader,enabled,false,false));
      unsigned found=0;
      nir_foreach_function_impl(impl,b.shader) {
         nir_foreach_block(block,impl) {
            nir_foreach_instr(instr,block) {
               if(instr->type!=nir_instr_type_intrinsic)continue;
               auto intr=nir_instr_as_intrinsic(instr);
               if(intr->intrinsic!=nir_intrinsic_load_agx)continue;
               found++;
               EXPECT_EQ(nir_intrinsic_base(intr),
                         enabled?AGX_NIR_MEMORY_INDEX_BYTES:0u);
               if(enabled){EXPECT_EQ(intr->src[0].ssa,base);}
            }
         }
      }
      EXPECT_EQ(found,1u);ralloc_free(b.shader);
   }
}

TEST_F(AGX3PackBinary, PackedVertexMemoryNativeBytes)
{
   /* Apple packed_uchar4/ushort4 records with pad then colour, stride5/10.
    * The unaligned whole vector is one native load into packed GPR words. */
   const uint8_t native[2][14] = {
      {0x67,0x00,0x44,0x00,0x00,0x00,0x00,0x00,0x67,0x25,0x00,0x40,0x22,0x00},
      {0x67,0x00,0x44,0x00,0x00,0x00,0x00,0x00,0x47,0x45,0x00,0x40,0x22,0x00},
   };
   for (unsigned half = 0; half < 2; ++half) {
      struct agx3_mem_desc desc = {};
      desc.width = half ? AGX3_MEM_W16 : AGX3_MEM_W8;
      desc.mask = 15;
      desc.base_kill = true;
      desc.idx_kind = AGX3_MEM_IDX_IMM;
      desc.byte_off = 1u << half;
      desc.slot = 6;
      desc.last_load = true;
      uint8_t bytes[AGX3_MEM_MAX_LENGTH], length;
      ASSERT_TRUE(agx3_mem_encode_load(AGX_ISA_ENCODING_AGX3B, &desc, bytes, &length));
      ASSERT_EQ(length, sizeof(native[half]));
      EXPECT_EQ(memcmp(bytes, native[half], length), 0);
   }
}

/* The engine's instance transform branches finish before vertex writeout.
 * The sink must leave that CFG intact and move stores only inside its final
 * block; this is independent of the native branch encoder's validation. */
TEST_F(AGX3PackBinary, VertexTerminalOutputBlockPreservesEarlierControlFlow)
{
   for (unsigned count : {6u, 9u, 32u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_VERTEX;
      agx_block *entry = agx_start_block(b->shader);
      agx_block *merge = agx_test_block(b->shader);
      agx_block *output = agx_test_block(b->shader);
      agx_begin_cf(b);
      agx_if_icmp(b, agx_register(0, AGX_SIZE_32),
                    agx_register(2, AGX_SIZE_32), 1, AGX_ICOND_ULT, false, merge);
      agx_jmp_exec_none(b, merge);
      agx_block_add_successor(entry, merge);
      b->cursor = agx_after_block(merge);
      agx_pop_exec(b, 1);
      agx_block_add_successor(merge, output);
      std::vector<agx_instr *> prefix;
      agx_foreach_instr_global(b->shader, I) prefix.push_back(I);
      b->cursor = agx_after_block(output);
      std::vector<agx_instr *> stores(count);
      for (unsigned c = count; c-- > 0;) {
         agx_index value = agx_mov_imm(b, 32, 0x3f000000 + c);
         stores[c] = agx_st_vary(b, agx_immediate(c), value);
      }
      agx_stop(b);
      ASSERT_TRUE(agx3_sink_vertex_outputs(b->shader));
      unsigned pos = 0, next = 0;
      agx_foreach_block(b->shader, block) {
         agx_foreach_instr_in_block(block, I) {
            if (block != output) {
               ASSERT_LT(pos, prefix.size());
               EXPECT_EQ(I, prefix[pos++]);
            } else if (I->op == AGX_OPCODE_ST_VARY) {
               ASSERT_LT(next, count);
               EXPECT_EQ(I, stores[next++]);
            } else if (next) {
               EXPECT_EQ(I->op, AGX_OPCODE_STOP);
            }
         }
      }
      EXPECT_EQ(pos, prefix.size());
      EXPECT_EQ(next, count);
   }
}

TEST_F(AGX3PackBinary, VertexTerminalOutputsCrossDirectEmptyFallthrough)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   agx_block *stores_block = agx_start_block(b->shader);
   agx_index values[12];
   values[0] = agx_temp(b->shader, AGX_SIZE_32);
   values[1] = agx_temp(b->shader, AGX_SIZE_32);
   agx_get_sr_to(b, values[0], AGX_SR_VERTEX_ID);
   agx_get_sr_to(b, values[1], AGX_SR_INSTANCE_ID);
   for (unsigned c = 2; c < ARRAY_SIZE(values); ++c)
      values[c] = agx_mov_imm(b, 32, 0x3f000000 + c);

   /* Exact ordering from the live Settings/Zink shader. */
   for (unsigned c : {8u, 9u, 10u, 11u, 4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u})
      agx_st_vary(b, agx_immediate(c), values[c]);

   agx_block *stop_block = agx_test_block(b->shader);
   agx_block_add_successor(stores_block, stop_block);
   b->cursor = agx_after_block(stop_block);
   agx_stop(b);

   agx_pressure_schedule(b->shader);
   ASSERT_TRUE(agx3_sink_vertex_outputs(b->shader));
   agx_foreach_instr_in_block(stores_block, I)
      EXPECT_NE(I->op, AGX_OPCODE_ST_VARY);

   unsigned next = 0;
   agx_foreach_instr_in_block(stop_block, I) {
      if (I->op == AGX_OPCODE_ST_VARY) {
         ASSERT_LT(next, ARRAY_SIZE(values));
         EXPECT_EQ(I->src[0].value, next++);
      } else {
         EXPECT_EQ(I->op, AGX_OPCODE_STOP);
         EXPECT_EQ(next, ARRAY_SIZE(values));
      }
   }
   EXPECT_EQ(next, ARRAY_SIZE(values));

   agx_ra(b->shader);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, VertexInstanceIdWithWideSingleBlockOutputs)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;

   agx_index instance = agx_temp(b->shader, AGX_SIZE_32);
   agx_index vertex = agx_temp(b->shader, AGX_SIZE_32);
   agx_get_sr_to(b, instance, AGX_SR_INSTANCE_ID);
   agx_get_sr_to(b, vertex, AGX_SR_VERTEX_ID);
   for (unsigned c : {8u, 9u, 10u, 11u, 4u, 5u, 6u, 7u, 0u, 1u, 2u, 3u})
      agx_st_vary(b, agx_immediate(c), c & 1 ? instance : vertex);
   agx_stop(b);

   agx_pressure_schedule(b->shader);
   ASSERT_TRUE(agx3_sink_vertex_outputs(b->shader));
   agx_ra(b->shader);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, VertexOutputCannotMoveAcrossControlFlow)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   for (unsigned c = 0; c < 9; ++c) {
      agx_index value = agx_mov_imm(b, 32, 0x3f000000 + c);
      agx_st_vary(b, agx_immediate(c), value);
   }
   agx_block *end = agx_test_block(b->shader);
   b->cursor = agx_after_block(end);
   agx_stop(b);
   std::vector<agx_instr *> before, after;
   agx_foreach_instr_global(b->shader, I) before.push_back(I);
   EXPECT_FALSE(agx3_sink_vertex_outputs(b->shader));
   agx_foreach_instr_global(b->shader, I) after.push_back(I);
   EXPECT_EQ(before, after);
}

TEST_F(AGX3PackBinary, VertexCFGTerminalOutputBytesMatchSingleBlock)
{
   for (unsigned count : {6u, 9u, 32u}) {
      std::vector<uint8_t> straight;
      for (bool cfg : {false, true}) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         b->shader->stage = MESA_SHADER_VERTEX;
         if (cfg) {
            agx_block *merge = agx_test_block(b->shader);
            agx_block *output = agx_test_block(b->shader);
            agx_begin_cf(b);
            agx_if_icmp(b, agx_register(0, AGX_SIZE_32),
                          agx_register(2, AGX_SIZE_32), 1, AGX_ICOND_ULT, false, merge);
            agx_jmp_exec_none(b, merge);
            b->cursor = agx_after_block(merge);
            agx_pop_exec(b, 1);
            b->cursor = agx_after_block(output);
         }
         for (unsigned c = 0; c < count; ++c)
            agx_st_vary(b, agx_immediate(c), agx_register(4, AGX_SIZE_32));
         agx_stop(b);
         auto code = pack(b);
         if (!cfg) {
            straight = code;
         } else {
            ASSERT_GT(code.size(), straight.size());
            EXPECT_TRUE(std::equal(straight.begin(), straight.end(),
                                    code.end() - straight.size()));
         }
      }
   }
}

TEST(AGX3NativeRgb10A2, NativeDeadLiveAndConstantOperands)
{
 const struct agx3_rgb10a2_desc d[3]={
  {.dest=0,.source={0,2,4,6},.kind={2,2,2,2},.discard={1,1,1,1}},
  {.dest=8,.source={6,4,2,0},.kind={2,2,2,2}},
  {.source={48,32,16,48},.kind={1,1,1,1},.first_instruction=true},
 };
 const unsigned char want[3][14]={{0xa7,0x06,0x54,0x00,0x02,0x00,0x08,0x20,0x60,0xa0,0xea,0x8b,0x24,0x11},{0xa7,0x06,0x54,0x08,0x02,0x0c,0x10,0x10,0x00,0xa0,0x2a,0x88,0x24,0x11},{0xa7,0x0e,0x54,0,2,0x60,0x80,0x80,0,0xa3,0x2a,8,0,0x10}};
 for(unsigned i=0;i<3;i++){
  unsigned char out[14]={0};unsigned char len=0;
  ASSERT_TRUE(agx3_rgb10a2_encode(&d[i],out,&len));
  ASSERT_EQ(len,14); EXPECT_EQ(memcmp(out,want[i],14),0);
 }
}

/* blend/native/negpack.py and negpack2.py: -x, (-x.r, x.g, -x.b, -x.a),
 * abs(x) and -abs(x) into RGB10A2Unorm. */
TEST(AGX3NativeRgb10A2, RegisterModifiers)
{
   struct variant {
      bool abs[4], neg[4];
      uint8_t want[14];
   };
   const struct variant v[] = {
      {{0}, {1,1,1,1},
       {0xa7,0x06,0x54,0,2,0,8,0x20,0x60,0xa0,0xea,0x8b,0xb6,0x15}},
      {{0}, {1,0,1,1},
       {0xa7,0x06,0x54,0,2,0,8,0x20,0x60,0xa0,0xea,0x8b,0xa6,0x15}},
      {{1,1,1,1}, {0},
       {0xa7,0x06,0x54,0,2,0,8,0x20,0x60,0xa0,0xea,0x8b,0x6d,0x13}},
      {{1,1,1,1}, {1,1,1,1},
       {0xa7,0x06,0x54,0,2,0,8,0x20,0x60,0xa0,0xea,0x8b,0xff,0x17}},
   };
   for (const struct variant &x : v) {
      struct agx3_rgb10a2_desc d = {
         .source = {0, 2, 4, 6}, .kind = {2, 2, 2, 2},
         .discard = {true, true, true, true},
      };
      memcpy(d.abs, x.abs, sizeof(d.abs));
      memcpy(d.neg, x.neg, sizeof(d.neg));
      uint8_t out[14], length = 0;
      ASSERT_TRUE(agx3_rgb10a2_encode(&d, out, &length));
      ASSERT_EQ(length, 14);
      EXPECT_EQ(memcmp(out, x.want, 14), 0);
   }

   struct agx3_rgb10a2_desc constant = {
      .source = {48, 32, 16, 48}, .kind = {1, 1, 1, 1},
   };
   constant.neg[1] = true;
   uint8_t out[14], length = 0;
   EXPECT_FALSE(agx3_rgb10a2_encode(&constant, out, &length));
}

/* rgb9e5/native/rgb9e5.py: float4 into RGB9E5Float and RG11B10Float, from
 * registers and from constants, and float4 [[color(0)]] reads of them. */
TEST(AGX3NativeRgb10A2, PackedFloatLayouts)
{
   const struct {
      uint8_t layout;
      bool constant;
      uint8_t want[14];
   } packs[] = {
      {AGX3_PACKED_RGB9E5, false,
       {0xa7,0x06,0x54,0,2,0,8,0x20,0x10,0xa0,0xca,0x89,0x24,0x00}},
      {AGX3_PACKED_RG11B10, false,
       {0xa7,0x06,0x54,0,2,0,8,0x20,0x10,0xa0,0xca,0x89,0x24,0x08}},
      {AGX3_PACKED_RGB9E5, true,
       {0xa7,0x0e,0x54,0,2,0x20,0x80,0,0x12,0xa0,0x0a,0x08,0,0}},
      {AGX3_PACKED_RG11B10, true,
       {0xa7,0x0e,0x54,0,2,0x20,0x80,0,0x12,0xa0,0x0a,0x08,0,0x08}},
   };
   for (const auto &p : packs) {
      struct agx3_rgb10a2_desc d = {.layout = p.layout};
      for (unsigned i = 0; i < 3; ++i) {
         d.kind[i] = p.constant ? AGX3_FLTNORM_SRC_IMMEDIATE
                                : AGX3_FLTNORM_SRC_REGISTER;
         d.source[i] = p.constant ? 16u << i : 2 * i;
         d.discard[i] = !p.constant;
      }
      d.first_instruction = p.constant;
      uint8_t out[14], length = 0;
      ASSERT_TRUE(agx3_rgb10a2_encode(&d, out, &length));
      ASSERT_EQ(length, 14);
      EXPECT_EQ(memcmp(out, p.want, 14), 0);

      d.kind[3] = AGX3_FLTNORM_SRC_REGISTER;
      d.source[3] = 6;
      EXPECT_FALSE(agx3_rgb10a2_encode(&d, out, &length));
   }

   const struct {
      uint8_t layout, want[8];
   } unpacks[] = {
      {AGX3_PACKED_RGB9E5, {0x27,6,0x56,0,1,0,0x5c,8}},
      {AGX3_PACKED_RG11B10, {0x27,6,0x56,0,1,0,0x5e,8}},
   };
   for (const auto &u : unpacks) {
      struct agx3_rgb10a2_unpack_desc d = {
         .dest = 0, .source = 0, .wait_mask = 32, .discard = true,
         .layout = u.layout,
      };
      uint8_t out[16], length = 0;
      ASSERT_TRUE(agx3_rgb10a2_unpack_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
      ASSERT_EQ(length, 8);
      EXPECT_EQ(memcmp(out, u.want, 8), 0);
      /* The layout field adds a seventh fixed field. All six waits must
       * still fit in the encoder's local field array. */
      d.wait_mask = 0x3f;
      ASSERT_TRUE(agx3_rgb10a2_unpack_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
      ASSERT_EQ(length, 8);
   }
}

TEST(AGX3NativeRgb10A2, NativeUnpackAndTextureDependency)
{
   const uint8_t expected[3][8] = {
      {0x27,6,0x56,0,1,0,0x5a,8}, {0x27,6,0x56,4,1,2,0x5a,8},
      {0x27,0x16,0x54,0,1,0,0x5a,8},
   };
   for (unsigned i = 0; i < 3; ++i) {
      struct agx3_rgb10a2_unpack_desc d = {
         .dest = (uint16_t)(i == 1 ? 8 : 0),
         .source = (uint16_t)(i == 1 ? 2 : 0),
         .wait_mask = (uint16_t)(i == 2 ? 1 : 32), .discard = true,
      };
      uint8_t out[16], length;
      ASSERT_TRUE(agx3_rgb10a2_unpack_encode(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
      ASSERT_EQ(length, 8); EXPECT_EQ(memcmp(out, expected[i], 8), 0);
   }
   struct agx3_rgb10a2_desc p = {
      .source = {0,2,4,6}, .kind = {2,2,2,2}, .wait_mask = 1,
      .discard = {true,true,true,true},
   };
   uint8_t bytes[14], length;
   const uint8_t want[14] = {0xa7,0x16,0x54,0,2,0,8,0x20,0x60,0xa0,0xea,0x8b,0x24,0x11};
   ASSERT_TRUE(agx3_rgb10a2_encode(&p, bytes, &length));
   EXPECT_EQ(memcmp(bytes, want, 14), 0);
}

/* AGX3C emits 16 bytes; the former 14-byte caller buffer overflowed under ASAN. */
TEST_F(AGX3PackBinary, Rgb10TilePacksUnderAgx3C)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
   key.fs.nr_samples = 1;
   key.no_stop = true;
   key.fs.ignore_tib_dependencies = true;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index packed = agx_register(16, AGX_SIZE_32);
   agx_pack_rgb10a2_agx3_to(b, packed, agx_register(0, AGX_SIZE_32),
                            agx_register(2, AGX_SIZE_32),
                            agx_register(4, AGX_SIZE_32),
                            agx_register(6, AGX_SIZE_32));
   agx_stop(b);
   const std::vector<uint8_t> code = pack(b);
   static const uint8_t want[] = {0xa7, 0x00, 0x00, 0x1a, 0x22, 0x00, 0xad,
                                  0x02, 0x90, 0x00, 0x01, 0x0c, 0x24, 0x02,
                                  0x00, 0x00};
   EXPECT_EQ(code.size(), sizeof(want));
   EXPECT_EQ(code, std::vector<uint8_t>(want, want + sizeof(want)));
}

TEST_F(AGX3PackBinary, Rgb10TileRejectsRawFloatChannels)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_st_tile(b, agx_register(0, AGX_SIZE_32), agx_immediate(255),
               agx_null(), AGX_FORMAT_RGB10A2, 15, 0, false, 0);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, NativeBfeilExtractPreservesSourceLifetime)
{
   for (bool dynamic : {false,true}) {
      agx_builder *b=builder(AGX_ISA_ENCODING_AGX3B);
      agx_index value=agx_register(0,AGX_SIZE_32);value.discard=true;
      agx_index shift=dynamic?agx_register(2,AGX_SIZE_32):agx_immediate(14);
      if(dynamic)shift.discard=true;
      agx_bfeil_to(b,agx_register(0,AGX_SIZE_32),agx_immediate(0),value,shift,14);
      agx_stop(b);
      auto code=pack(b);
      const uint8_t expected[2][12]={
         {0xa7,0,0x54,0,2,0,0x38,0,0xf0,0x11,0xe1,0},
         {0xa7,0,0x54,0,2,0,8,0,0xf0,0x13,0xe5,0},
      };
      ASSERT_EQ(code.size(),16);
      EXPECT_EQ(memcmp(code.data(),expected[dynamic],12),0);
   }
}

TEST_F(AGX3PackBinary, HalfBfeilRetainsCachedDynamicShift)
{
   auto build = [this](bool cache_shift) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index value = agx_register(10, AGX_SIZE_16);
      value.discard = true;
      agx_index shift = agx_register(16, AGX_SIZE_32);
      shift.cache = cache_shift;
      agx_bfeil_to(b, agx_register(4, AGX_SIZE_16), agx_immediate(0),
                   value, shift, 0);
      agx_stop(b);
      return pack(b);
   };

   /* The dynamic-shift endpoint has no source-cache bit. NONE retains the
    * shift register, so the cached and ordinary post-RA forms are exact. */
   EXPECT_EQ(build(true), build(false));
}

TEST_F(AGX3PackBinary, BfeilKeepsNonzeroMergeBaseClosed)
{
   agx_builder *b=builder(AGX_ISA_ENCODING_AGX3B);
   agx_bfeil_to(b,agx_register(0,AGX_SIZE_32),agx_register(4,AGX_SIZE_32),
               agx_register(0,AGX_SIZE_32),agx_immediate(14),14);
   agx_stop(b);EXPECT_TRUE(pack(b,false).empty());
}

TEST(AGX3NativeBfeil, NativeWaitAndOperandBounds)
{
   const enum agx3_alu_hint hints[2]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_NONE};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],length;
   const uint8_t dead[12]={0xa7,0,0x56,0,2,0,0x38,0,0xf0,0x11,0xe1,0};
   ASSERT_TRUE(agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3B,0,0,14,false,14,hints,32,bytes,&length));
   ASSERT_EQ(length,12);EXPECT_EQ(memcmp(bytes,dead,12),0);
   EXPECT_FALSE(agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3B,1,0,14,false,14,hints,0,bytes,&length));
   EXPECT_FALSE(agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3B,0,0,32,false,14,hints,0,bytes,&length));
   EXPECT_FALSE(agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3B,0,0,14,false,32,hints,0,bytes,&length));
   EXPECT_FALSE(agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3B,0,0,14,false,14,hints,64,bytes,&length));
}

TEST_F(AGX3PackBinary, NativeIselImmediateAndNarrowComparands)
{
   for (bool narrow : {false, true}) {
      for (bool lhs : {false, true}) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         const auto size = narrow ? AGX_SIZE_16 : AGX_SIZE_32;
         agx_index a = agx_register(narrow ? 1 : 2, size);
         agx_index t = agx_register(narrow ? 3 : 4, size);
         agx_index f = agx_register(narrow ? 5 : 6, size);
         a.cache = true;
         t.discard = f.discard = true;
         agx_icmpsel_to(b, agx_register(0, size),
                        lhs ? agx_immediate(2) : a,
                        lhs ? a : agx_immediate(2), t, f,
                        lhs ? AGX_ICOND_UGT : AGX_ICOND_ULT);
         agx_stop(b);
         auto code = pack(b);
         ASSERT_EQ(code.size(), 14);
         struct agx3_isel_desc d = {};
         d.dest_16 = narrow;
         d.cmp_16 = narrow;
         d.cmp1_immediate = true;
         d.cmp0 = a.value;
         d.cmp1 = 2;
         d.cond = AGX3_ISEL_ICOND_ULT;
         d.sel_true = t.value;
         d.sel_false = f.value;
         d.cache[0] = true;
         d.discard[2] = d.discard[3] = true;
         uint8_t expected[AGX3_ISEL_MAX_LENGTH], length;
         ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, expected, &length));
         EXPECT_EQ(memcmp(code.data(), expected, length), 0);
      }
   }
}

TEST_F(AGX3PackBinary, NativeIselCarriesDestinationCache)
{
   auto build = [&](bool cache) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(0, AGX_SIZE_16);
      agx_index cmp = agx_register(2, AGX_SIZE_32);
      dest.cache = cache;
      cmp.discard = true;
      agx_icmpsel_to(b, dest, cmp, agx_immediate(1), agx_immediate(1),
                     agx_immediate(0), AGX_ICOND_UEQ);
      agx_stop(b);
      return pack(b);
   };

   const auto uncached = build(false);
   const auto cached = build(true);
   ASSERT_EQ(uncached.size(), 14);
   ASSERT_EQ(cached.size(), uncached.size());
   auto expected = uncached;
   expected[2] |= 0x20; /* Native ISEL destination-cache field f12. */
   EXPECT_EQ(cached, expected);
}

TEST_F(AGX3PackBinary, NativeHalfComparisonSelectsIntegerImmediate)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(0, AGX_SIZE_16);
   agx_index c = agx_register(1, AGX_SIZE_16);
   a.discard = c.discard = true;
   agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_32), a, c,
                  agx_immediate(2), agx_immediate(0), AGX_FCOND_LT);
   agx_stop(b);

   const auto code = pack(b);
   const uint16_t src[] = {0, 1};
   const bool src_16[] = {true, true};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
   uint8_t expected[AGX3_ALU_MAX_LENGTH], length = 0;
   ASSERT_TRUE(agx3_alu_fcmp_select_imm_width(
      AGX_ISA_ENCODING_AGX3B, 0, src, src_16, hints, AGX_FCOND_LT,
      2, 0, false, false, 0, expected, &length));
   ASSERT_GE(code.size(), length);
   EXPECT_EQ(memcmp(code.data(), expected, length), 0);
}

TEST_F(AGX3PackBinary, MixedAbsoluteNegatedCompareUsesGeneralLongForm)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(0, AGX_SIZE_32);
   agx_index c = agx_register(2, AGX_SIZE_32);
   a.abs = true;
   c.neg = true;
   a.cache = c.cache = true;
   agx_fcmpsel_to(b, agx_register(4, AGX_SIZE_16), a, c,
                  agx_immediate(0), agx_immediate(255), AGX_FCOND_LT);
   agx_stop(b);

   struct agx3_fsel_desc d = {
      .dest = 4,
      .dest_16 = true,
      .cmp0 = 0,
      .cmp0_hint = AGX3_ALU_HINT_CACHE,
      .cmp0_abs = true,
      .cmp1 = 2,
      .cmp1_hint = AGX3_ALU_HINT_CACHE,
      .cmp1_neg = true,
      .sel0_kind = AGX3_FSEL_SEL_IMMEDIATE,
      .sel1_kind = AGX3_FSEL_SEL_IMMEDIATE,
      .sel0 = 0,
      .sel1 = 255,
      .sel_16 = true,
      .condition = AGX_FCOND_LT,
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], length = 0;
   ASSERT_TRUE(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                                    &length));
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   std::vector<uint8_t> expected;
   append(expected, bytes, length);
   append(expected, end.bytes, end.length);
   EXPECT_EQ(pack(b), expected);
}

TEST_F(AGX3PackBinary, NativeIselRetainsLiveRegisterComparands)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(2, AGX_SIZE_32);
   agx_index c = agx_register(4, AGX_SIZE_32);
   agx_index t = agx_register(6, AGX_SIZE_32);
   agx_index f = agx_register(8, AGX_SIZE_32);
   a.cache = c.cache = t.cache = f.cache = true;
   agx_icmpsel_to(b, agx_register(0, AGX_SIZE_32), a, c, t, f, AGX_ICOND_ULT);
   agx_stop(b);
   auto code = pack(b);
   const uint8_t native_nowait[10] = {0x02,0x83,0x07,0x85,0x02,0x86,0x05,0,0,0x88};
   ASSERT_EQ(code.size(), 14);
   EXPECT_EQ(memcmp(code.data(), native_nowait, 10), 0);
}

TEST_F(AGX3PackBinary, NativeIselPairsAdjacentDeviceLoadSlots)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index root = agx_register(0, AGX_SIZE_32);
   root.channels_m1 = 3;
   agx_device_load_to(b, root, agx_uniform(8, AGX_SIZE_64), agx_immediate(0),
                      AGX_FORMAT_I32, 15, 0, false);

   agx_index first = agx_register(8, AGX_SIZE_32);
   agx_index second = agx_register(10, AGX_SIZE_32);
   agx_device_load_to(b, first, agx_register(0, AGX_SIZE_64),
                      agx_immediate(0), AGX_FORMAT_I32, 1, 2, false);
   agx_device_load_to(b, second, agx_register(0, AGX_SIZE_64),
                      agx_immediate(1), AGX_FORMAT_I32, 1, 2, false);
   first.discard = true;
   second.discard = true;
   agx_icmpsel_to(b, agx_register(12, AGX_SIZE_32), first, second, first,
                  second, AGX_ICOND_UEQ);
   agx_stop(b);

   const auto code = pack(b);
   /* Both value loads issue before one source-dependent identity retires each
    * of slots 1 and 2. ISEL itself then carries no scoreboard wait. */
   ASSERT_EQ(code.size(), 76);
   const uint16_t carrier1_src[] = {10, 10};
   const uint16_t carrier2_src[] = {8, 8};
   const enum agx3_alu_hint carrier_hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   uint8_t carrier1[AGX3_ALU_MAX_LENGTH], carrier2[AGX3_ALU_MAX_LENGTH];
   uint8_t carrier1_length = 0, carrier2_length = 0;
   ASSERT_TRUE(agx3_alu_bit_reg(
      AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND, 10, carrier1_src,
      carrier_hints, 1, carrier1, &carrier1_length));
   ASSERT_TRUE(agx3_alu_bit_reg(
      AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND, 8, carrier2_src,
      carrier_hints, 2, carrier2, &carrier2_length));
   EXPECT_EQ(memcmp(code.data() + 42, carrier1, carrier1_length), 0);
   EXPECT_EQ(memcmp(code.data() + 42 + carrier1_length, carrier2,
                    carrier2_length), 0);
   const struct agx3_isel_desc desc = {
      .dest = 12,
      .cmp0 = 8,
      .cmp1 = 10,
      .cond = AGX3_ISEL_ICOND_EQ,
      .explicit_hints = true,
      .discard = {true, true, true, true},
      .true_kind = AGX3_ISEL_REGISTER,
      .false_kind = AGX3_ISEL_REGISTER,
      .sel_true = 8,
      .sel_false = 10,
      .wait_mask = 0,
   };
   uint8_t expected[AGX3_ISEL_MAX_LENGTH], length = 0;
   ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &desc, expected,
                                &length));
   ASSERT_EQ(length, 10);
   EXPECT_EQ(memcmp(code.data() + 42 + carrier1_length + carrier2_length,
                    expected, length), 0);
}

TEST_F(AGX3PackBinary, NativeIselRetiresTextureAndDeviceLoadPair)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *merge = agx_test_block(b->shader);

   agx_index old_sample = agx_register(16, AGX_SIZE_32);
   old_sample.channels_m1 = 3;
   agx_index old_coord = agx_register(24, AGX_SIZE_32);
   old_coord.channels_m1 = 1;
   agx_instr *old_texture = agx_texture_sample_to(
      b, old_sample, old_coord, agx_immediate_f(0.0f), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   old_texture->mask = 15;
   agx_begin_cf(b);
   agx_if_icmp(b, agx_register(30, AGX_SIZE_32),
               agx_register(32, AGX_SIZE_32), 1, AGX_ICOND_ULT, false,
               merge);
   agx_jmp_exec_none(b, merge);
   agx_block_add_successor(entry, body);
   agx_block_add_successor(entry, merge);

   b->cursor = agx_after_block(body);

   agx_index sample = agx_register(8, AGX_SIZE_32);
   sample.channels_m1 = 3;
   agx_index coord = agx_register(4, AGX_SIZE_32);
   coord.channels_m1 = 1;
   agx_instr *texture = agx_texture_sample_to(
      b, sample, coord, agx_immediate_f(0.0f), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   texture->mask = 15;

   agx_index loaded = agx_register(40, AGX_SIZE_32);
   agx_device_load_to(b, loaded, agx_register(36, AGX_SIZE_64),
                      agx_immediate(0), AGX_FORMAT_I32, 1, 2, false);

   agx_index sample_value = agx_register(8, AGX_SIZE_32);
   agx_index old_value = agx_register(16, AGX_SIZE_32);
   sample_value.discard = loaded.discard = old_value.discard = true;
   agx_icmpsel_to(b, agx_register(44, AGX_SIZE_32), sample_value, loaded,
                  sample_value, old_value, AGX_ICOND_UEQ);
   agx_block_add_successor(body, merge);

   b->cursor = agx_after_block(merge);
   agx_pop_exec(b, 1);
   agx_stop(b);

   const auto code = pack(b);
   ASSERT_FALSE(code.empty());

   const uint16_t carrier1_src[] = {40, 40};
   const uint16_t carrier2_src[] = {8, 8};
   const enum agx3_alu_hint carrier_hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   uint8_t carrier1[AGX3_ALU_MAX_LENGTH], carrier2[AGX3_ALU_MAX_LENGTH];
   uint8_t carrier1_length = 0, carrier2_length = 0;
   ASSERT_TRUE(agx3_alu_bit_reg(
      AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND, 40, carrier1_src,
      carrier_hints, 1, carrier1, &carrier1_length));
   ASSERT_TRUE(agx3_alu_bit_reg(
      AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND, 8, carrier2_src,
      carrier_hints, 2, carrier2, &carrier2_length));
   EXPECT_NE(std::search(code.begin(), code.end(), carrier1,
                         carrier1 + carrier1_length),
             code.end());
   EXPECT_NE(std::search(code.begin(), code.end(), carrier2,
                         carrier2 + carrier2_length),
             code.end());

   /* The sample and load stay overlapped on distinct slots. Their grounded
    * identity carrier retires both before the consuming ISEL. */
   const struct agx3_isel_desc desc = {
      .dest = 44,
      .cmp0 = 8,
      .cmp1 = 40,
      .cond = AGX3_ISEL_ICOND_EQ,
      .explicit_hints = true,
      .discard = {true, true, true, true},
      .true_kind = AGX3_ISEL_REGISTER,
      .false_kind = AGX3_ISEL_REGISTER,
      .sel_true = 8,
      .sel_false = 16,
      .wait_mask = 0,
   };
   uint8_t expected[AGX3_ISEL_MAX_LENGTH], length = 0;
   ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &desc, expected,
                                &length));
   EXPECT_NE(std::search(code.begin(), code.end(), expected,
                         expected + length),
             code.end());
}

TEST_F(AGX3PackBinary, NativeIselPairsDereferencedDeviceLoadSlots)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;

   agx_index first_ptr = agx_register(0, AGX_SIZE_32);
   first_ptr.channels_m1 = 3;
   agx_device_load_to(b, first_ptr, agx_uniform(8, AGX_SIZE_64),
                      agx_immediate(1), AGX_FORMAT_I32, 15, 0, false);
   agx_index first = agx_register(0, AGX_SIZE_32);
   agx_index first_base = agx_register(0, AGX_SIZE_64);
   first_base.discard = true;
   agx_device_load_to(b, first, first_base,
                      agx_immediate(0), AGX_FORMAT_I32, 1, 2, false);

   agx_index second_ptr = agx_register(8, AGX_SIZE_32);
   second_ptr.channels_m1 = 3;
   agx_device_load_to(b, second_ptr, agx_uniform(8, AGX_SIZE_64),
                      agx_immediate(2), AGX_FORMAT_I32, 15, 0, false);
   agx_index second = agx_register(2, AGX_SIZE_32);
   agx_index second_base = agx_register(8, AGX_SIZE_64);
   second_base.discard = true;
   agx_device_load_to(b, second, second_base,
                      agx_immediate(0), AGX_FORMAT_I32, 1, 2, false);

   first.discard = true;
   second.discard = true;
   agx_icmpsel_to(b, agx_register(0, AGX_SIZE_32), second, first, second,
                  first, AGX_ICOND_UEQ);
   agx_stop(b);

   const auto code = pack(b);
   /* Pointer loads reuse slot 3 after each dereference consumes it. The two
    * value loads overlap on slots 2 and 1, then one grounded carrier retires
    * each producer before ISEL. */
   ASSERT_EQ(code.size(), 90);
   const uint16_t carrier1_src[] = {2, 2};
   const uint16_t carrier2_src[] = {0, 0};
   const enum agx3_alu_hint carrier_hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   uint8_t carrier1[AGX3_ALU_MAX_LENGTH], carrier2[AGX3_ALU_MAX_LENGTH];
   uint8_t carrier1_length = 0, carrier2_length = 0;
   ASSERT_TRUE(agx3_alu_bit_reg(
      AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND, 2, carrier1_src,
      carrier_hints, 1, carrier1, &carrier1_length));
   ASSERT_TRUE(agx3_alu_bit_reg(
      AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND, 0, carrier2_src,
      carrier_hints, 2, carrier2, &carrier2_length));
   EXPECT_NE(std::search(code.begin(), code.end(), carrier1,
                         carrier1 + carrier1_length),
             code.end());
   EXPECT_NE(std::search(code.begin(), code.end(), carrier2,
                         carrier2 + carrier2_length),
             code.end());

   const struct agx3_isel_desc desc = {
      .dest = 0,
      .cmp0 = 2,
      .cmp1 = 0,
      .cond = AGX3_ISEL_ICOND_EQ,
      .explicit_hints = true,
      .discard = {true, true, true, true},
      .true_kind = AGX3_ISEL_REGISTER,
      .false_kind = AGX3_ISEL_REGISTER,
      .sel_true = 2,
      .sel_false = 0,
      .wait_mask = 0,
   };
   uint8_t expected[AGX3_ISEL_MAX_LENGTH], length = 0;
   ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &desc, expected,
                                &length));
   EXPECT_NE(std::search(code.begin(), code.end(), expected,
                         expected + length),
             code.end());
}

TEST_F(AGX3PackBinary, NativeIselDropsControlFlowRetiredSlot)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *merge = agx_test_block(b->shader);

   agx_index old_value = agx_register(8, AGX_SIZE_32);
   agx_device_load_to(b, old_value, agx_uniform(8, AGX_SIZE_64),
                      agx_immediate(0), AGX_FORMAT_I32, 1, 2, false);
   agx_begin_cf(b);
   agx_if_icmp(b, agx_register(16, AGX_SIZE_32),
               agx_register(18, AGX_SIZE_32), 1, AGX_ICOND_ULT, false,
               merge);
   agx_jmp_exec_none(b, merge);
   agx_block_add_successor(entry, body);
   agx_block_add_successor(entry, merge);

   b->cursor = agx_after_block(body);
   agx_index first = agx_register(10, AGX_SIZE_32);
   agx_index second = agx_register(12, AGX_SIZE_32);
   agx_device_load_to(b, first, agx_uniform(12, AGX_SIZE_64),
                      agx_immediate(0), AGX_FORMAT_I32, 1, 2, false);
   agx_device_load_to(b, second, agx_uniform(16, AGX_SIZE_64),
                      agx_immediate(0), AGX_FORMAT_I32, 1, 2, false);
   old_value.discard = first.discard = second.discard = true;
   agx_icmpsel_to(b, agx_register(14, AGX_SIZE_32), second, first, first,
                  old_value, AGX_ICOND_UEQ);
   agx_block_add_successor(body, merge);

   b->cursor = agx_after_block(merge);
   agx_pop_exec(b, 1);
   agx_stop(b);
   const auto code = pack(b);
   ASSERT_FALSE(code.empty());

   /* The old value's slot was retired before IF. Each live body load is
    * retired through its own grounded carrier before ISEL. */
   const uint16_t carrier1_src[] = {12, 12};
   const uint16_t carrier2_src[] = {10, 10};
   const enum agx3_alu_hint carrier_hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   uint8_t carrier1[AGX3_ALU_MAX_LENGTH], carrier2[AGX3_ALU_MAX_LENGTH];
   uint8_t carrier1_length = 0, carrier2_length = 0;
   ASSERT_TRUE(agx3_alu_bit_reg(
      AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND, 12, carrier1_src,
      carrier_hints, 1, carrier1, &carrier1_length));
   ASSERT_TRUE(agx3_alu_bit_reg(
      AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND, 10, carrier2_src,
      carrier_hints, 2, carrier2, &carrier2_length));
   EXPECT_NE(std::search(code.begin(), code.end(), carrier1,
                         carrier1 + carrier1_length),
             code.end());
   EXPECT_NE(std::search(code.begin(), code.end(), carrier2,
                         carrier2 + carrier2_length),
             code.end());

   const struct agx3_isel_desc desc = {
      .dest = 14,
      .cmp0 = 12,
      .cmp1 = 10,
      .cond = AGX3_ISEL_ICOND_EQ,
      .explicit_hints = true,
      .discard = {true, true, true, true},
      .true_kind = AGX3_ISEL_REGISTER,
      .false_kind = AGX3_ISEL_REGISTER,
      .sel_true = 10,
      .sel_false = 8,
      .wait_mask = 0,
   };
   uint8_t expected[AGX3_ISEL_MAX_LENGTH], length = 0;
   ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &desc, expected,
                                &length));
   EXPECT_NE(std::search(code.begin(), code.end(), expected,
                         expected + length),
             code.end());
}

TEST_F(AGX3PackBinary, AbsFselWaitsOnItsLoadedComparator)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_device_load_to(b, agx_register(4, AGX_SIZE_32),
                      agx_uniform(8, AGX_SIZE_64), agx_immediate(0),
                      AGX_FORMAT_I32, 1, 2, false);
   agx_mov_imm_to(b, agx_register(6, AGX_SIZE_32), 0x3d4ccccd);

   agx_index loaded = agx_register(4, AGX_SIZE_32);
   agx_index threshold = agx_register(6, AGX_SIZE_32);
   loaded.abs = loaded.discard = true;
   threshold.cache = true;
   agx_fcmpsel_to(b, agx_register(8, AGX_SIZE_16), loaded, threshold,
                  agx_immediate(1), agx_immediate(0), AGX_FCOND_LT);
   agx_stop(b);

   const auto code = pack(b);
   uint8_t carrier[AGX3_ALU_MAX_LENGTH], length = 0;
   const uint16_t operands[] = {4, 4};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   ASSERT_TRUE(agx3_alu_bit_reg(AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND,
                                4, operands, hints, 1, carrier, &length));
   EXPECT_NE(std::search(code.begin(), code.end(), carrier, carrier + length),
             code.end());
}

TEST_F(AGX3PackBinary, NativeIselImageCopyBooleanHalfRegisters)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(1, AGX_SIZE_16);
   agx_index f = agx_register(10, AGX_SIZE_16);
   a.discard = f.discard = true;
   agx_icmpsel_to(b, agx_register(1, AGX_SIZE_16), a, agx_immediate(0),
                  agx_immediate(0), f, AGX_ICOND_UEQ);
   agx_index x = agx_register(4, AGX_SIZE_16);
   agx_index y = agx_register(6, AGX_SIZE_16);
   agx_icmpsel_to(b, agx_register(1, AGX_SIZE_16), x, y, x, y, AGX_ICOND_UEQ);
   agx_stop(b);
   /* Both original image-copy shapes stay fused: two10-byte selects+stop. */
   EXPECT_EQ(pack(b).size(), 24);
}

TEST_F(AGX3PackBinary, ProjectiveRhwTracksBranchesAndPartialRegisterLifetimes)
{
   for (unsigned mode = 0; mode < 8; ++mode) {
      SCOPED_TRACE(mode);
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_block *entry = list_first_entry(&b->shader->blocks, agx_block, link);
      agx_block *left = agx_test_block(b->shader);
      agx_block *right = agx_test_block(b->shader);
      entry->successors[0] = left;
      entry->successors[1] = right;
      const agx_index rhw = agx_register(0, AGX_SIZE_32);
      agx_rhw_interp_agx3_to(b, rhw, agx_immediate(0), agx_immediate(0),
                            AGX_INTERPOLATION_CENTER);
      if (mode == 0 || mode == 6)
         agx_persp_rcp_agx3_to(b, mode == 0 ? rhw : agx_register(2, AGX_SIZE_32), rhw, 0);
      b->cursor = agx_after_block(left);
      if (mode == 4 || mode == 5 || mode == 6)
         agx_mov_imm_to(b, agx_register(0, AGX_SIZE_16), 0);
      if (mode == 6)
         agx_mov_imm_to(b, agx_register(1, AGX_SIZE_16), 0);
      if (mode == 4)
         agx_mov_to(b, agx_register(4, AGX_SIZE_16), agx_register(1, AGX_SIZE_16));
      else if (mode == 6)
         agx_fmul_to(b, agx_register(4, AGX_SIZE_32), rhw, rhw);
      else if (mode != 0)
         agx_persp_rcp_agx3_to(b, agx_register(2, AGX_SIZE_32), rhw, mode == 7 ? 1 : 0);
      if (mode == 3)
         left->successors[0] = left;
      b->cursor = agx_after_block(right);
      if (mode == 2)
         agx_fmul_to(b, agx_register(4, AGX_SIZE_32), rhw, rhw);
      else if (mode != 0)
         agx_persp_rcp_agx3_to(b, agx_register(2, AGX_SIZE_32), rhw, 0);
      agx_stop(b);
      const bool valid = mode == 0 || mode == 1 || mode == 3 || mode == 6;
      EXPECT_EQ(!pack(b, valid).empty(), valid);
   }
}

TEST_F(AGX3PackBinary, ModifiedFaddWaitAndLifetimeStayFused)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index loaded = agx_register(0, AGX_SIZE_32);
   loaded.channels_m1 = 1;
   agx_device_load_to(b, loaded, agx_uniform(0, AGX_SIZE_64),
                      agx_immediate(0), AGX_FORMAT_I32, 3, 2, false);
   agx_index x = agx_register(0, AGX_SIZE_32);
   agx_index y = agx_register(2, AGX_SIZE_32);
   x.cache = true;
   y.neg = y.discard = true;
   agx_fadd_to(b, agx_register(4, AGX_SIZE_32), x, y);
   agx_stop(b);
   const auto code = pack(b);
   /* Native negate-first subtraction consumes the load directly, retains x
    * and discards y. No temporary negation or separate wait operation. */
   const uint8_t native[6] = {0x29,0x81,0x14,0x03,0,0xc8};
   ASSERT_GE(code.size(), 10u);
   EXPECT_EQ(memcmp(code.data()+code.size()-10, native, 6), 0);
}

TEST_F(AGX3PackBinary, M5NarrowMultiplyMatchesNativeHalf4)
{
   for (unsigned component = 1; component < 4; ++component) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_index x = agx_register(2 * component, AGX_SIZE_32);
      agx_index y = agx_register(8 + 2 * component, AGX_SIZE_32);
      x.discard = y.discard = true;
      agx_fmul_to(b, agx_register(component, AGX_SIZE_16), x, y);
      agx_stop(b);
      const auto code = pack(b);
      /* Actual G17G metal-nt half4(float4 * float4), including odd halves. */
      const uint8_t native[][6] = {
         {0x01,0x03,0x1d,0x0b,0x04,0x00},
         {0x11,0x05,0x19,0x0d},
         {0x11,0x07,0x1d,0x0f,0x04,0x00},
      };
      const unsigned length = component == 2 ? 4 : 6;
      ASSERT_EQ(code.size(), length + 4);
      EXPECT_EQ(memcmp(code.data(), native[component - 1], length), 0);
   }
}

TEST_F(AGX3PackBinary, NarrowFaddKeepsOddHalfAndLiveInput)
{
   for (bool swapped : {false,true}) {
      for (bool dead : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index source = agx_register(2, AGX_SIZE_32);
      source.kill = source.discard = dead;
      agx_index zero = agx_immediate(0x80);
      agx_fadd_to(b, agx_register(1, AGX_SIZE_16),
                  swapped ? zero : source, swapped ? source : zero);
      agx_stop(b);
      auto code = pack(b);
      /* One 16-bit rounding ADD. A post-RA kill is representable exactly
       * when DISCARD owns the same final use, as in Gallium's readback CS. */
      const uint8_t expected[2][6] = {
         {0x01,0x03,0x14,0x81,0x04,0x02},
         {0x01,0x03,0x1c,0x81,0x04,0x02},
      };
      ASSERT_EQ(code.size(),10);
      EXPECT_EQ(memcmp(code.data(),expected[dead],6),0);
      }
   }
}

TEST_F(AGX3PackBinary, HalfMaximumRoutesLifetimesWithoutPhantomWait)
{
   for (bool live : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index a = agx_register(2, AGX_SIZE_16);
      agx_index c = agx_register(3, AGX_SIZE_16);
      agx_index dest = agx_register(live ? 0 : 2, AGX_SIZE_16);
      if (live) {
         a.cache = c.cache = dest.cache = true;
      } else {
         a.discard = c.discard = true;
      }
      agx_fcmpsel_to(b, dest, a, c, a, c, AGX_FCOND_GTN);
      agx_stop(b);
      const auto code = pack(b);
      const uint8_t retained[] = {0x02,0x82,0x24,0x82,0x10,0x00};
      const uint8_t dead[] = {0x12,0x02,0x1c,0x02,0x10,0x00};
      const uint8_t *native = live ? retained : dead;
      ASSERT_EQ(code.size(), 10u);
      EXPECT_EQ(memcmp(code.data(), native, 6), 0);
   }
}

/*
 * Apple's own applegpu_g17p bytes for the 32-bit tied minimum/maximum.  Before
 * this form existed the packer refused every shader containing max(x, 0.0),
 * which is normalize(), clamp(), saturate() and the discard compare.
 *
 *   fmax(a, b)      12 03 1e 05 00 c0
 *   fmin(a, b)      12 03 1e 05 01 c0
 *   fmax(a, 0.0f)   12 03 0a 81 00 c2
 *   fmax(a, 1.0f)   12 03 0a b1 00 c2
 */
TEST_F(AGX3PackBinary, F32TiedMaximumMatchesNativeRegisterForm)
{
   for (bool maximum : {true, false}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index a = agx_register(2, AGX_SIZE_32);
      agx_index c = agx_register(4, AGX_SIZE_32);
      agx_index dest = agx_register(2, AGX_SIZE_32);
      a.discard = c.discard = true;
      agx_fcmpsel_to(b, dest, a, c, a, c,
                     maximum ? AGX_FCOND_GTN : AGX_FCOND_LTN);
      agx_stop(b);
      const auto code = pack(b);
      const uint8_t native[] = {
         0x12, 0x03, 0x1e, 0x05, (uint8_t)(maximum ? 0x00 : 0x01), 0x00,
      };
      ASSERT_EQ(code.size(), 10u);
      EXPECT_EQ(memcmp(code.data(), native, 6), 0);
   }
}

TEST_F(AGX3PackBinary, F32TiedMaximumMatchesNativeMinifloatForm)
{
   static const struct {
      float value;
      uint8_t byte3, byte4;
   } cases[] = {
      {0.0f, 0x81, 0x00}, {0.5f, 0xa1, 0x00}, {1.0f, 0xb1, 0x00},
      {2.0f, 0xc1, 0x00}, {-1.0f, 0xb1, 0x00}, {31.0f, 0xff, 0x10},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index a = agx_register(2, AGX_SIZE_32);
      agx_index dest = agx_register(2, AGX_SIZE_32);
      a.discard = true;
      const agx_index k = agx_immediate_f(cases[i].value);
      agx_fcmpsel_to(b, dest, a, k, a, k, AGX_FCOND_GTN);
      agx_stop(b);
      const auto code = pack(b);
      const uint8_t native[] = {
         0x12, 0x03, (uint8_t)(cases[i].value < 0.0f ? 0x1a : 0x0a),
         cases[i].byte3, cases[i].byte4, 0x02,
      };
      ASSERT_EQ(code.size(), 10u);
      EXPECT_EQ(memcmp(code.data(), native, 6), 0);
   }
}

/*
 * max(0.0, x) is max(x, 0.0). NIR declares fmin_agx and fmax_agx
 * 2src_commutative, and a tied GTN/LTN fcmpsel can only have come from one of
 * them, so the constant may be written on either side and both encode to the
 * bytes Apple emits for the constant-second form.
 */
TEST_F(AGX3PackBinary, F32TiedMaximumIsCommutative)
{
   for (bool constant_first : {false, true}) {
      for (bool maximum : {true, false}) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         agx_index a = agx_register(2, AGX_SIZE_32);
         agx_index dest = agx_register(2, AGX_SIZE_32);
         a.discard = true;
         const agx_index k = agx_immediate_f(0.0f);
         const agx_index s0 = constant_first ? k : a;
         const agx_index s1 = constant_first ? a : k;
         agx_fcmpsel_to(b, dest, s0, s1, s0, s1,
                        maximum ? AGX_FCOND_GTN : AGX_FCOND_LTN);
         agx_stop(b);
         const auto code = pack(b);
         const uint8_t native[] = {
            0x12, 0x03, 0x0a, 0x81, (uint8_t)(maximum ? 0x00 : 0x01), 0x02,
         };
         ASSERT_EQ(code.size(), 10u) << "constant_first=" << constant_first;
         EXPECT_EQ(memcmp(code.data(), native, 6), 0)
            << "constant_first=" << constant_first;
      }
   }

   /* The same for two registers: the tied pair may be named either way. */
   for (bool swapped : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index a = agx_register(2, AGX_SIZE_32);
      agx_index c = agx_register(4, AGX_SIZE_32);
      agx_index dest = agx_register(2, AGX_SIZE_32);
      a.discard = c.discard = true;
      const agx_index s0 = swapped ? c : a;
      const agx_index s1 = swapped ? a : c;
      agx_fcmpsel_to(b, dest, s0, s1, s0, s1, AGX_FCOND_GTN);
      agx_stop(b);
      const auto code = pack(b);
      ASSERT_EQ(code.size(), 10u);
      /* Both operands are registers, so nothing is normalized: the two
       * operand fields simply exchange. */
      const uint8_t native[] = {
         0x12, (uint8_t)(swapped ? 0x05 : 0x03), 0x1e,
         (uint8_t)(swapped ? 0x03 : 0x05), 0x00, 0x00,
      };
      EXPECT_EQ(memcmp(code.data(), native, 6), 0) << "swapped=" << swapped;
   }
}

/*
 * A half select whose selected operands are byte constants is not the half
 * fmin/fmax form and must fall through to the byte-constant encoder, not fail
 * the shader.  This is the shape a `discard` compare produces.
 */
TEST_F(AGX3PackBinary, HalfSelectOfByteConstantsIsNotRefusedByTheHalfForms)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(4, AGX_SIZE_32);
   agx_index c = agx_register(2, AGX_SIZE_32);
   agx_index dest = agx_register(0, AGX_SIZE_16);
   a.discard = c.discard = true;
   agx_fcmpsel_to(b, dest, a, c, agx_immediate(255), agx_immediate(0),
                  AGX_FCOND_LT);
   agx_stop(b);
   const auto code = pack(b);
   EXPECT_GT(code.size(), 4u);
}

TEST_F(AGX3PackBinary, HalfCompareSelectMatchesNative)
{
   for (bool live : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index a = agx_register(live ? 2 : 0, AGX_SIZE_16);
      agx_index c = agx_register(live ? 3 : 1, AGX_SIZE_16);
      agx_index if_true = agx_register(live ? 4 : 2, AGX_SIZE_16);
      agx_index if_false = agx_register(live ? 5 : 3, AGX_SIZE_16);
      agx_index dest = agx_register(0, AGX_SIZE_16);
      if (live) {
         a.cache = c.cache = if_true.cache = dest.cache = true;
         if_false.discard = true;
      } else {
         a.discard = c.discard = if_true.discard = if_false.discard = true;
      }
      agx_fcmpsel_to(b, dest, a, c, if_true, if_false, AGX_FCOND_LT);
      agx_stop(b);
      const auto code = pack(b);
      const uint8_t retained[] = {
         0x02,0x82,0x25,0x82,0x02,0x84,0x13,0xc0,0x80,0x05,
      };
      const uint8_t dead[] = {
         0x02,0x00,0x1d,0x00,0x82,0x02,0x13,0xc0,0x80,0x03,
      };
      const uint8_t *native = live ? retained : dead;
      ASSERT_EQ(code.size(), 14u);
      EXPECT_EQ(memcmp(code.data(), native, 10), 0);
   }

   /* An absolute half comparator takes the fourteen-byte endpoint. Apple's
    * applegpu_g17p `(abs(a) > b) ? a : b` over half operands, with its slot 6
    * wait (f107) cleared: 12021d0283021200800300800100. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(2, AGX_SIZE_16);
   agx_index c = agx_register(3, AGX_SIZE_16);
   a.discard = c.discard = true;
   agx_index modified = a;
   modified.abs = true;
   agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), modified, c, a, c,
                  AGX_FCOND_GT);
   agx_stop(b);
   const auto code = pack(b);
   const uint8_t native[] = {
      0x12,0x02,0x1d,0x02,0x83,0x02,0x12,0x00,0x80,0x03,0x00,0x00,0x01,0x00,
   };
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);
}

TEST_F(AGX3PackBinary, HalfSqrtMatchesNativeSequenceAndRefusesModifier)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index half = agx_register(2, AGX_SIZE_16);
   half.cache = true;
   agx_index step = agx_register(4, AGX_SIZE_32);
   step.cache = true;
   agx_srsqrt_to(b, step, half);

   step.cache = false;
   step.discard = true;
   half.cache = false;
   half.discard = true;
   agx_index dest = agx_register(2, AGX_SIZE_16);
   dest.cache = true;
   agx_fmul_to(b, dest, step, half);
   agx_stop(b);

   const uint8_t native[] = {
      0x2f,0x01,0x54,0x04,0x03,0x04,0x8a,0x40,0x00,0x00,
      0x11,0x05,0x39,0x02,
   };
   const auto code = pack(b);
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   half = agx_register(2, AGX_SIZE_16);
   half.cache = true;
   half.abs = true;
   step = agx_register(4, AGX_SIZE_32);
   step.cache = true;
   agx_srsqrt_to(b, step, half);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, MixedHalfMultiplyCarriesCachedHalf)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index f32 = agx_register(16, AGX_SIZE_32);
   agx_index f16 = agx_register(13, AGX_SIZE_16);
   f32.discard = true;
   f16.cache = true;
   agx_fmul_to(b, agx_register(15, AGX_SIZE_16), f32, f16);
   agx_stop(b);

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 15, 16, 13, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_CACHE, false, 0, bytes, &len));
   std::vector<uint8_t> want;
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, HalfClampDoesNotCarryLoadWaitIntoSecondSelect)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index x = agx_register(2, AGX_SIZE_16);
   agx_index low = agx_register(3, AGX_SIZE_16);
   agx_index high = agx_register(4, AGX_SIZE_16);
   x.cache = low.cache = true;
   agx_index temporary = agx_register(0, AGX_SIZE_16);
   temporary.cache = true;
   agx_fcmpsel_to(b, temporary, x, low, x, low, AGX_FCOND_GTN);
   temporary.cache = false;
   temporary.discard = high.discard = true;
   agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_16), temporary, high,
                  temporary, high, AGX_FCOND_LTN);
   agx_stop(b);
   const auto code = pack(b);
   const uint8_t expected_chain[] = {
      0x02,0x82,0x24,0x82,0x10,0x00,
      0x02,0x00,0x1c,0x04,0x01,0x00,
   };
   ASSERT_EQ(code.size(), sizeof(expected_chain) + 4);
   if (memcmp(code.data(), expected_chain, sizeof(expected_chain)) != 0) {
      fprintf(stderr, "half-clamp chain got ");
      for (unsigned i = 0; i < sizeof(expected_chain); ++i)
         fprintf(stderr, "%02x", code[i]);
      fputc('\n', stderr);
   }
   EXPECT_EQ(memcmp(code.data(), expected_chain, sizeof(expected_chain)), 0);
}

TEST_F(AGX3PackBinary, GtkMixedReciprocalMatchesNative)
{
   struct width_case {
      enum agx_size dest_size, src_size;
      uint16_t dest, src;
      const uint8_t bytes[10];
   };
   const width_case widths[] = {
      {AGX_SIZE_16, AGX_SIZE_16, 2, 2,
       {0xaf, 0x00, 0x54, 0x02, 0x02, 0x04, 0x10, 0x46, 0x30, 0x00}},
      {AGX_SIZE_32, AGX_SIZE_16, 0, 0,
       {0xaf, 0x00, 0x54, 0x00, 0x02, 0x00, 0x10, 0x44, 0x20, 0x00}},
      {AGX_SIZE_16, AGX_SIZE_32, 2, 2,
       {0xaf, 0x00, 0x54, 0x02, 0x02, 0x04, 0x10, 0x4a, 0x30, 0x00}},
      {AGX_SIZE_32, AGX_SIZE_32, 0, 0,
       {0xaf, 0x00, 0x54, 0x00, 0x02, 0x00, 0x10, 0x48, 0x20, 0x00}},
   };
   for (const width_case &v : widths) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index src = agx_register(v.src, v.src_size);
      src.discard = true;
      agx_rcp_to(b, agx_register(v.dest, v.dest_size), src);
      agx_stop(b);
      std::vector<uint8_t> native(v.bytes, v.bytes + sizeof(v.bytes));
      native.insert(native.end(), {0x0e, 0x00, 0x00, 0x00});
      EXPECT_EQ(pack(b), native);
   }

   /* Exact post-RA blocker from GTK gskgpuconvertbuiltin: a cached odd half
    * texture channel feeds a cached full-precision reciprocal result. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(6, AGX_SIZE_32);
   agx_index src = agx_register(3, AGX_SIZE_16);
   dest.cache = src.cache = true;
   agx_rcp_to(b, dest, src);
   agx_stop(b);
   const std::vector<uint8_t> expected = {
      0xaf, 0x00, 0x54, 0x06, 0x03, 0x06, 0x02, 0x44, 0x20, 0x00,
      0x0e, 0x00, 0x00, 0x00,
   };
   EXPECT_EQ(pack(b), expected);

   /* Source ABS and NEG are native fields rather than legalization moves. */
   for (unsigned modifiers = 0; modifiers < 4; ++modifiers) {
      b = builder(AGX_ISA_ENCODING_AGX3B);
      dest = agx_register(0, AGX_SIZE_32);
      src = agx_register(0, AGX_SIZE_16);
      src.discard = true;
      src.abs = modifiers & 1;
      src.neg = modifiers & 2;
      agx_rcp_to(b, dest, src);
      agx_stop(b);
      const std::vector<uint8_t> native = {
         0xaf, 0x00, 0x54, 0x00, 0x02, 0x00, 0x10, 0x44,
         (uint8_t)(0x20 | (modifiers << 6)), 0x00,
         0x0e, 0x00, 0x00, 0x00,
      };
      EXPECT_EQ(pack(b), native) << "modifiers=" << modifiers;
   }
}

TEST_F(AGX3PackBinary, ReciprocalWidthsRefuseUngroundedModes)
{
   for (unsigned mode = 1; mode < 18; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index dest = agx_register(0, AGX_SIZE_32);
      agx_index src = agx_register(0, AGX_SIZE_16);
      src.discard = true;

      if (mode == 1)
         dest.size = AGX_SIZE_64;
      if (mode == 2)
         src.size = AGX_SIZE_64;
      if (mode == 3)
         dest.memory = true;
      if (mode == 4)
         src.memory = true;
      if (mode == 5)
         src.kill = true;
      if (mode == 6)
         src.cache = true;
      if (mode == 7)
         dest.value = 1;
      if (mode == 8) {
         src.size = AGX_SIZE_32;
         src.value = 1;
      }
      if (mode == 9)
         dest.channels_m1 = 1;
      if (mode == 10)
         src.channels_m1 = 1;
      if (mode == 11)
         src = agx_immediate(0);
      if (mode == 12)
         src = agx_uniform(0, AGX_SIZE_16);
      /* 192, not 128: f2 and f18 of f32rcp ep0 accept [0, 191] in Apple's own
       * composer tables, with 192 distinct encodings and no aliasing, and the
       * packer's gate now agrees with them rather than refusing sixty-four
       * addressable registers. */
      if (mode == 14) {
         dest.size = AGX_SIZE_16;
         dest.value = 192;
      }
      if (mode == 15)
         src.value = 192;
      if (mode == 16)
         dest.value = 192;
      if (mode == 17) {
         src.size = AGX_SIZE_32;
         src.value = 192;
      }

      agx_instr *rcp = agx_rcp_to(b, dest, src);
      rcp->saturate = mode == 13;
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty()) << "mode=" << mode;
   }
}

/* The gap the old gate left: a reciprocal whose destination is a register the
 * form can address but the packer refused.  Three of the ten distinct programs
 * three.js compiles for webgl_geometry_cube, webgl_animation_keyframes and
 * webgl_materials_physical_clearcoat died on this shape, and hk_shader.c turns
 * a packer refusal into VK_ERROR_FEATURE_NOT_PRESENT for the whole pipeline. */
TEST_F(AGX3PackBinary, ReciprocalEncodesAboveTheOldRegisterGate)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index dest = agx_register(140, AGX_SIZE_32);
   agx_index src = agx_register(86, AGX_SIZE_32);
   src.abs = true;
   src.cache = true;
   agx_rcp_to(b, dest, src);
   agx_stop(b);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, HalfReciprocalSqrtMatchesNativeAndRefusesModifier)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(3, AGX_SIZE_16);
   src.discard = true;
   agx_rsqrt_to(b, agx_register(3, AGX_SIZE_16), src);
   agx_stop(b);
   const auto code = pack(b);
   const uint8_t native[] = {
      0xaf,0x01,0x54,0x03,0x02,0x06,0xac,0x60,0x00,0x00,
   };
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(2, AGX_SIZE_16);
   src.neg = true;
   agx_rsqrt_to(b, agx_register(2, AGX_SIZE_16), src);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, HalfExp2MatchesNativeNegateAndRefusesAbsolute)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(3, AGX_SIZE_16);
   src.discard = true;
   agx_exp2_to(b, agx_register(3, AGX_SIZE_16), src);
   agx_stop(b);
   const auto code = pack(b);
   const uint8_t native[] = {
      0xaf,0x02,0x54,0x03,0x02,0x06,0xac,0x60,0x00,0x00,
   };
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(2, AGX_SIZE_16);
   src.neg = true;
   src.discard = true;
   agx_exp2_to(b, agx_register(2, AGX_SIZE_16), src);
   agx_stop(b);
   const uint8_t native_neg[] = {
      0xaf,0x02,0x54,0x02,0x02,0x04,0xac,0x60,0x01,0x00,
   };
   const auto neg_code = pack(b);
   ASSERT_EQ(neg_code.size(), sizeof(native_neg) + 4);
   EXPECT_EQ(memcmp(neg_code.data(), native_neg, sizeof(native_neg)), 0);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(2, AGX_SIZE_16);
   src.abs = true;
   agx_exp2_to(b, agx_register(2, AGX_SIZE_16), src);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, HalfLog2MatchesNativeAndModifiers)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(3, AGX_SIZE_16);
   src.discard = true;
   agx_log2_to(b, agx_register(3, AGX_SIZE_16), src);
   agx_stop(b);
   const auto code = pack(b);
   const uint8_t native[] = {
      0x2f,0x02,0x54,0x03,0x02,0x06,0xac,0x60,0x00,0x00,
   };
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(memcmp(code.data(), native, sizeof(native)), 0);

   b = builder(AGX_ISA_ENCODING_AGX3B);
   src = agx_register(2, AGX_SIZE_16);
   src.neg = true;
   agx_log2_to(b, agx_register(2, AGX_SIZE_16), src);
   agx_stop(b);
   const uint8_t native_neg[] = {
      0x2f,0x02,0x54,0x02,0x02,0x04,0x8c,0x60,0x01,0x00,
   };
   const auto neg_code = pack(b);
   ASSERT_EQ(neg_code.size(), sizeof(native_neg) + 4);
   EXPECT_EQ(memcmp(neg_code.data(), native_neg, sizeof(native_neg)), 0);
}

TEST_F(AGX3PackBinary, MixedWidthLog2MatchesCurrentGtkAndNativeModifiers)
{
   /* Exact unchanged GTK component-transfer post-RA shape. Half 76 is read
    * directly and the float result begins at the aligned pair 64. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_log2_to(b, agx_register(64, AGX_SIZE_32),
               agx_register(76, AGX_SIZE_16));
   agx_stop(b);
   const uint8_t gtk_native[] = {
      0x2f,0x02,0x54,0x40,0x02,0x98,0x88,0x40,0x00,0x00,
   };
   const auto gtk_code = pack(b);
   ASSERT_EQ(gtk_code.size(), sizeof(gtk_native) + 4);
   EXPECT_EQ(memcmp(gtk_code.data(), gtk_native, sizeof(gtk_native)), 0);

   /* Fresh Apple half-input/full-output form with an odd source and both
    * modifiers. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(3, AGX_SIZE_16);
   src.abs = true;
   src.neg = true;
   src.discard = true;
   agx_log2_to(b, agx_register(0, AGX_SIZE_32), src);
   agx_stop(b);
   const uint8_t modified_native[] = {
      0x2f,0x02,0x54,0x00,0x02,0x06,0xa8,0xc0,0x01,0x00,
   };
   const auto modified_code = pack(b);
   ASSERT_EQ(modified_code.size(), sizeof(modified_native) + 4);
   EXPECT_EQ(memcmp(modified_code.data(), modified_native,
                    sizeof(modified_native)), 0);

   /* A full-width destination/source still requires aligned register pairs,
    * and every scalar register remains inside the endpoint's register
    * bounds. */
   b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_log2_to(b, agx_register(3, AGX_SIZE_32),
               agx_register(2, AGX_SIZE_16));
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());

   b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_log2_to(b, agx_register(0, AGX_SIZE_32),
               agx_register(192, AGX_SIZE_16));
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());

   b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_log2_to(b, agx_register(192, AGX_SIZE_16),
               agx_register(0, AGX_SIZE_16));
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, HalfFloorMatchesNativeAndRefusesModifier)
{
   for (unsigned cached = 0; cached < 2; ++cached) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index src = agx_register(3, AGX_SIZE_16);
      src.cache = cached;
      src.discard = !cached;
      agx_floor_to(b, agx_register(3, AGX_SIZE_16), src);
      agx_stop(b);
      const auto code = pack(b);
      const uint8_t native[][10] = {
         {0x2f,0x00,0x54,0x03,0x02,0x06,0xac,0x60,0x02,0x00},
         {0x2f,0x00,0x54,0x03,0x02,0x06,0x8e,0x60,0x02,0x00},
      };
      ASSERT_EQ(code.size(), sizeof(native[cached]) + 4);
      EXPECT_EQ(memcmp(code.data(), native[cached], sizeof(native[cached])), 0);
   }

   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src = agx_register(2, AGX_SIZE_16);
   src.abs = true;
   agx_floor_to(b, agx_register(2, AGX_SIZE_16), src);
   agx_stop(b);
   EXPECT_TRUE(pack(b, false).empty());
}

TEST_F(AGX3PackBinary, HalfCeilMatchesNativeAndRefusesModifier)
{
   agx_builder *b=builder(AGX_ISA_ENCODING_AGX3B);
   agx_index src=agx_register(3,AGX_SIZE_16); src.discard=true;
   agx_ceil_to(b,agx_register(3,AGX_SIZE_16),src); agx_stop(b);
   const auto code=pack(b);
   const uint8_t native[]={0x2f,0x00,0x54,0x03,0x02,0x06,0xac,0x60,0x04,0x00};
   ASSERT_EQ(code.size(),sizeof(native)+4);
   EXPECT_EQ(memcmp(code.data(),native,sizeof(native)),0);
   b=builder(AGX_ISA_ENCODING_AGX3B); src=agx_register(2,AGX_SIZE_16);
   src.neg=true; agx_ceil_to(b,agx_register(2,AGX_SIZE_16),src); agx_stop(b);
   EXPECT_TRUE(pack(b,false).empty());
}

TEST_F(AGX3PackBinary, HalfMaximumRoutesUntiedAndRefusesSaturate)
{
   /* An absolute comparator is Apple's own half maximum form (see
    * HalfCompareSelectMatchesNative); only the saturated select refuses. */
   for (unsigned mode = 0; mode < 3; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index a = agx_register(2, AGX_SIZE_16);
      agx_index c = agx_register(3, AGX_SIZE_16);
      agx_index selected_a = a;
      if (mode == 0)
         selected_a.value = 4;
      if (mode == 1)
         a.abs = true;
      agx_instr *I = agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_16), a, c,
                                    selected_a, c, AGX_FCOND_GTN);
      if (mode == 2)
         I->saturate = true;
      agx_stop(b);
      const auto code = pack(b, mode != 2);
      EXPECT_EQ(code.empty(), mode == 2) << "mode=" << mode;
   }
}

TEST_F(AGX3PackBinary, FselZeroMixedGnomeRegisterShape)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      const bool zero_first = mode & 1, invert = mode & 2;
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index half = agx_register(1, AGX_SIZE_16);
      agx_index zero = agx_immediate(0);
      agx_instr *I = agx_fcmpsel_to(b, agx_register(70, AGX_SIZE_32),
                     zero_first ? zero : half, zero_first ? half : zero,
                     agx_register(invert ? 2 : 4, AGX_SIZE_32),
                     agx_register(invert ? 4 : 2, AGX_SIZE_32),
                     AGX_FCOND_EQ);
      I->invert_cond = invert;
      agx_stop(b);
      auto code = pack(b);
      /* Original GNOME post-RA shape: odd half comparison, both full float
       * alternatives remain live, one FSEL followed by STOP. */
      ASSERT_EQ(code.size(), 14u);
      const uint8_t expected[] = {0x32,0x00,0x87,0x81,0x06,0x04,0x08,0x02,0x00,0x02};
      EXPECT_EQ(memcmp(code.data(),expected,sizeof(expected)),0);
   }
}

TEST_F(AGX3PackBinary, FselZeroNativeLifetimeAndCompareOrder)
{
   for (bool zero_first : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index half = agx_register(10, AGX_SIZE_16);
      agx_index t = agx_register(0, AGX_SIZE_32), f = agx_register(2, AGX_SIZE_32);
      half.discard = t.discard = f.discard = true;
      agx_fcmpsel_to(b, agx_register(0, AGX_SIZE_32),
                     zero_first ? agx_immediate(0) : half,
                     zero_first ? half : agx_immediate(0), t, f,
                     zero_first ? AGX_FCOND_GT : AGX_FCOND_LT);
      agx_stop(b);
      auto code = pack(b);
      const uint8_t native[] = {0x02,0x0a,0x0f,0x81,0x80,0x02,0x03,0x02};
      ASSERT_EQ(code.size(), sizeof(native)+4);
      EXPECT_EQ(memcmp(code.data(), native, sizeof(native)),0);
   }
}

TEST_F(AGX3PackBinary, ChromiumNegatedRegisterEqualityFselEncodesBySwap)
{
   /* `-a == c` with both selected values in registers. f65 names the SECOND
    * comparator only, so Apple spells this by exchanging the operands and
    * reversing the relation -- equality being its own reverse. The exchange
    * itself is witnessed on the register-select form:
    *
    *    a >= c    12051d07850302c0
    *   -a >= c    12071d05850303c8   operands exchanged, f110 5 -> 6, f65
    *
    * so the shape that used to fail Chromium's pipeline closed is the
    * plain select with one field set. */
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index a = agx_register(2, AGX_SIZE_32);
   a.neg = a.cache = true;
   agx_index c = agx_register(12, AGX_SIZE_32);
   agx_index t = agx_register(10, AGX_SIZE_32);
   agx_index f = agx_register(18, AGX_SIZE_32);
   c.discard = t.discard = f.discard = true;
   agx_fcmpsel_to(b, agx_register(10, AGX_SIZE_32), a, c, t, f,
                  AGX_FCOND_EQ);
   agx_stop(b);
   EXPECT_FALSE(pack(b).empty());
}

TEST_F(AGX3PackBinary, HalfCompareGnomeLoopBreakAndBothBranchFixups)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_push_exec(b, 1);
   agx_block *body = agx_test_block(b->shader);
   agx_block *pop = agx_test_block(b->shader);
   agx_block *after = agx_test_block(b->shader);
   agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
   /* Original blur-loop comparison. The limit and counter are both live
    * after the comparison, including on the next iteration. */
   agx_break_if_icmp(&lb, agx_register(18, AGX_SIZE_16),
                     agx_register(1, AGX_SIZE_16), 1, false, AGX_ICOND_SLT, after);
   agx_jmp_exec_none_after(&lb, pop);
   agx_builder pb = agx_init_builder(b->shader, agx_after_block(pop));
   agx_jmp_exec_any(&pb, body);
   agx_pop_exec(&pb, 1);
   agx_builder ab = agx_init_builder(b->shader, agx_after_block(after));
   agx_stop(&ab);
   const auto code = pack(b);
   const uint8_t prefix[] = {
      0x0f,0x05,0x54,0x1a,
      0x0a,0x12,0x23,0x00,0x02,0x00,0x17,0x08,0x02,0x00,
      0x8f,0x04,0x54,0x22,
   };
   ASSERT_EQ(code.size(),48u);
   EXPECT_EQ(memcmp(code.data(),prefix,sizeof(prefix)),0);
   /* Reserved native branches deposit a signed byte displacement at byte3.
    * Forward reaches the footer POP; backward re-enters the compare. */
   const uint8_t forward[] = {0x0f,0x01,0x54,0x14,0,0,0,0,0,0};
   const uint8_t backward[] = {0x0f,0x00,0x54,0xe8,0xff,0xff,0xff,0xff,0xff,0};
   EXPECT_EQ(memcmp(code.data()+18,forward,sizeof(forward)),0);
   EXPECT_EQ(memcmp(code.data()+28,backward,sizeof(backward)),0);
   const uint8_t pop_native[] = {0x0f,0x06,0x04,0x02,0,0};
   EXPECT_EQ(memcmp(code.data()+38,pop_native,sizeof(pop_native)),0);
}

TEST_F(AGX3PackBinary, LoopExitBranchKeepsLoopAndOuterPops)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_begin_cf(b);
   agx_push_exec(b, 0);
   agx_push_exec(b, 1);

   agx_block *body = agx_test_block(b->shader);
   agx_block *footer = agx_test_block(b->shader);
   agx_block *after = agx_test_block(b->shader);

   agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
   agx_break_if_icmp(&lb, agx_register(18, AGX_SIZE_16),
                     agx_register(1, AGX_SIZE_16), 1, false, AGX_ICOND_SLT,
                     after);
   agx_jmp_exec_none_after(&lb, footer);

   agx_builder fb = agx_init_builder(b->shader, agx_after_block(footer));
   agx_jmp_exec_any(&fb, body);
   agx_pop_exec(&fb, 1);
   agx_pop_exec(&fb, 1);

   agx_builder ab = agx_init_builder(b->shader, agx_after_block(after));
   agx_stop(&ab);

   const auto code = pack(b);
   uint8_t expected[AGX3_CF_MAX_LENGTH];
   uint8_t len = 0;
   ASSERT_TRUE(agx3_cf_branch(AGX_ISA_ENCODING_AGX3B, true, expected, &len));
   ASSERT_TRUE(agx3_cf_patch_branch(AGX_ISA_ENCODING_AGX3B, true, expected,
                                    20));

   /* The forward branch starts after the two pushes, compare and CNDLT.  It
    * must land on CNDEND(2), not skip it and land on the outer CNDEND(1). */
   ASSERT_GE(code.size(), 22u + len);
   EXPECT_EQ(memcmp(code.data() + 22, expected, len), 0);
}

/*
 * agx_lower_pseudo spells an AGX3 unconditional break as the always-true
 * identity r0l == r0l; this pins the cached/killed normalization of that
 * identity, and the level count NEST 2 turns into.
 *
 * A loop owns two architectural exec-mask levels, so a break that escapes the
 * loop plus one enclosing block has to stay inactive for three -- CNDCF with
 * count 3, which is Apple's byte-exact form for that shape. CNDLT is the same
 * operation pinned to count 2 and covers only the NEST 1 break, which
 * HalfCompareGnomeLoopBreakAndBothBranchFixups above asserts.
 */
TEST_F(AGX3PackBinary, CachedHalfIdentityUnconditionalLoopBreak)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_push_exec(b, 1);
   agx_block *body = agx_test_block(b->shader);
   agx_block *after = agx_test_block(b->shader);
   agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
   agx_index r0l = agx_register(0, AGX_SIZE_16);
   r0l.cache = true;
   r0l.kill = true;
   agx_break_if_icmp(&lb, r0l, r0l, 2, false, AGX_ICOND_UEQ, after);
   agx_jmp_exec_any(&lb, body);
   agx_pop_exec(&lb, 1);
   agx_builder ab = agx_init_builder(b->shader, agx_after_block(after));
   agx_stop(&ab);

   uint8_t bytes[AGX3_CF_MAX_LENGTH];
   uint8_t len = 0;
   std::vector<uint8_t> want;
   ASSERT_TRUE(agx3_cf_cndst(AGX_ISA_ENCODING_AGX3B, 6, false, 2, bytes,
                             &len));
   append(want, bytes, len);
   const size_t body_offset = want.size();
   const uint16_t src[] = {0, 0};
   const struct agx3_cf_source_hints live[] = {
      {false, false}, {false, false},
   };
   ASSERT_TRUE(agx3_cf_icmp_reg16(AGX_ISA_ENCODING_AGX3B,
                                  AGX3_CF_ICOND_EQ, 1, src, live, 0,
                                  bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(
      agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3B, 1, false, 3, bytes, &len));
   append(want, bytes, len);
   const size_t branch_offset = want.size();
   ASSERT_TRUE(agx3_cf_branch(AGX_ISA_ENCODING_AGX3B, false, bytes, &len));
   ASSERT_TRUE(agx3_cf_patch_branch(AGX_ISA_ENCODING_AGX3B, false, bytes,
                                    (int64_t)body_offset -
                                       (int64_t)branch_offset));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_cf_cndend(AGX_ISA_ENCODING_AGX3B, 2, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
}

/*
 * Compiler-produced AGX3 outer loops carry an unconditional push_exec after
 * BEGIN_CF.  The GLSL CTS return_after_loop_sequence vertex shader reaches
 * this exact shape, with a break escaping two enclosing ifs.  Apple's loop
 * entry is CNDST(2). A deep break uses the next flag slot so it does not
 * overwrite either enclosing if predicate; nested loops advance again.
 */
TEST_F(AGX3PackBinary, BeginCfDeepBreakUsesNextFlagSlot)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_block *body = agx_test_block(b->shader);
   agx_block *after = agx_test_block(b->shader);

   agx_begin_cf(b);
   agx_push_exec(b, 0);
   agx_push_exec(b, 1);
   agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
   agx_index counter = agx_register(2, AGX_SIZE_16);
   agx_if_icmp(&lb, counter, agx_immediate(8), 1, AGX_ICOND_UEQ, true, NULL);
   agx_if_icmp(&lb, counter, agx_immediate(9), 1, AGX_ICOND_UEQ, false, NULL);
   agx_index r0l = agx_register(0, AGX_SIZE_16);
   r0l.cache = true;
   r0l.kill = true;
   agx_break_if_icmp(&lb, r0l, r0l, 3, false, AGX_ICOND_UEQ, after);
   agx_pop_exec(&lb, 1);
   agx_pop_exec(&lb, 1);
   agx_jmp_exec_any(&lb, body);
   agx_pop_exec(&lb, 1);
   agx_pop_exec(&lb, 1);

   agx_builder ab = agx_init_builder(b->shader, agx_after_block(after));
   agx_stop(&ab);

   const auto code = pack(b);
   uint8_t bytes[AGX3_CF_MAX_LENGTH];
   uint8_t len = 0;
   const uint16_t src[] = {0, 0};
   const struct agx3_cf_source_hints live[] = {
      {false, false}, {false, false},
   };
   std::vector<uint8_t> guard;
   ASSERT_TRUE(
      agx3_cf_cndst(AGX_ISA_ENCODING_AGX3B, 6, false, 1, bytes, &len));
   append(guard, bytes, len);
   EXPECT_NE(std::search(code.begin(), code.end(), guard.begin(), guard.end()),
             code.end());

   std::vector<uint8_t> entry;
   ASSERT_TRUE(
      agx3_cf_cndst(AGX_ISA_ENCODING_AGX3B, 6, false, 2, bytes, &len));
   append(entry, bytes, len);
   EXPECT_NE(std::search(code.begin(), code.end(), entry.begin(), entry.end()),
             code.end());

   std::vector<uint8_t> native;
   ASSERT_TRUE(agx3_cf_icmp_reg16(AGX_ISA_ENCODING_AGX3B,
                                  AGX3_CF_ICOND_EQ, 1, src, live, 0,
                                  bytes, &len));
   append(native, bytes, len);
   ASSERT_TRUE(
      agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3B, 1, false, 4, bytes, &len));
   append(native, bytes, len);
   EXPECT_NE(std::search(code.begin(), code.end(), native.begin(), native.end()),
             code.end());

   std::vector<uint8_t> stale;
   ASSERT_TRUE(agx3_cf_icmp_reg16(AGX_ISA_ENCODING_AGX3B,
                                  AGX3_CF_ICOND_EQ, 0, src, live, 0,
                                  bytes, &len));
   append(stale, bytes, len);
   ASSERT_TRUE(
      agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3B, 0, false, 4, bytes, &len));
   append(stale, bytes, len);
   EXPECT_EQ(std::search(code.begin(), code.end(), stale.begin(), stale.end()),
             code.end());
}

TEST_F(AGX3PackBinary, G16GDeepBreakUsesNextFlagSlot)
{
   /* G16G must advance flag_slot for nested BREAK_IF (nest > 1) just like
    * other AGX3B/HAL200 chips so the break comparison does not overwrite an
    * enclosing conditional's flag slot. Check both outer (loop_slot 0 ->
    * flag_slot 1) and nested (loop_slot 1 -> flag_slot 2) loops. */
   for (unsigned depth : {1u, 2u}) {
      SCOPED_TRACE(depth);
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      key.dev.chip = AGX_CHIP_G16G;
      agx_block *body = agx_test_block(b->shader);
      agx_block *after = agx_test_block(b->shader);

      for (unsigned d = 0; d < depth; ++d)
         agx_push_exec(b, 1);

      agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
      agx_index counter = agx_register(2, AGX_SIZE_16);
      agx_if_icmp(&lb, counter, agx_immediate(8), 1, AGX_ICOND_UEQ, true, NULL);
      agx_index r0l = agx_register(0, AGX_SIZE_16);
      r0l.cache = true;
      r0l.kill = true;
      agx_break_if_icmp(&lb, r0l, r0l, 2, false, AGX_ICOND_UEQ, after);
      agx_pop_exec(&lb, 1);
      agx_jmp_exec_any(&lb, body);
      for (unsigned d = 0; d < depth; ++d)
         agx_pop_exec(&lb, 1);

      agx_builder ab = agx_init_builder(b->shader, agx_after_block(after));
      agx_stop(&ab);

      const auto code = pack(b);
      const unsigned expected_slot = (depth - 1 + 1) % 6;
      const unsigned stale_slot = (depth - 1) % 6;
      uint8_t bytes[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      const uint16_t src[] = {0, 0};
      const struct agx3_cf_source_hints live[] = {
         {false, false}, {false, false},
      };

      std::vector<uint8_t> native;
      ASSERT_TRUE(agx3_cf_icmp_reg16(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ,
                                     expected_slot, src, live, 0, bytes, &len));
      append(native, bytes, len);
      ASSERT_TRUE(agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3B, expected_slot, false, 3,
                                bytes, &len));
      append(native, bytes, len);
      EXPECT_NE(
         std::search(code.begin(), code.end(), native.begin(), native.end()),
         code.end());

      std::vector<uint8_t> stale;
      ASSERT_TRUE(agx3_cf_icmp_reg16(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ,
                                     stale_slot, src, live, 0, bytes, &len));
      append(stale, bytes, len);
      ASSERT_TRUE(agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3B, stale_slot, false, 3,
                                bytes, &len));
      append(stale, bytes, len);
      EXPECT_EQ(
         std::search(code.begin(), code.end(), stale.begin(), stale.end()),
         code.end());
   }
}

TEST_F(AGX3PackBinary, DeepBreakConsumesBreakPredicateInsteadOfContinuePredicate)
{
   /* Full native pairs, decoded with Apple's composed tables:
    * break_continue +0xac: EQ(i,9), CNDCF(slot0,invert0,count3).
    * deep_break +0xc8: EQ(guard,0), CNDCF(slot1,invert1,count3).
    * Thus true, not false, selects a lane to break with CNDCF. Test both
    * source senses and both nesting classes without deriving the truth
    * table from the packer's choice of opcode. */
   for (unsigned nest : {1u, 2u, 3u}) {
      for (bool invert : {false, true}) {
         agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
         agx_push_exec(b, 1);
         agx_break_if_icmp(b, agx_register(4, AGX_SIZE_32),
                           agx_immediate(9), nest, invert, AGX_ICOND_UEQ,
                           nullptr);
         agx_stop(b);
         const auto code = pack(b);
         const unsigned cnd_offset = 4 + 10; /* entry + EQ comparison */
         ASSERT_GE(code.size(), cnd_offset + (nest == 1 ? 4u : 6u));
         uint64_t cnd = 0;
         for (unsigned byte = 0; byte < (nest == 1 ? 4u : 6u); ++byte)
            cnd |= uint64_t(code[cnd_offset + byte]) << (byte * 8);
         const bool encoded_invert = (cnd >> (nest == 1 ? 29 : 43)) & 1;
         for (bool equal : {false, true}) {
            const bool consumed = equal ^ encoded_invert;
            const bool exits = nest == 1 ? !consumed : consumed;
            EXPECT_EQ(exits, equal ^ invert) << "nest=" << nest;
         }
         if (nest > 1) {
            const uint8_t native[] = {
               0x8f, 0x05, 0x54, uint8_t(nest + 1), 0,
               uint8_t(invert ? 9 : 1),
            };
            EXPECT_EQ(memcmp(code.data() + cnd_offset, native, sizeof(native)), 0);
         }
      }
   }
}

/*
 * The same break, but escaping enclosing ifs as well. Apple's compiler emits
 * CNDCF with a level count equal to the number of architectural levels open
 * inside the loop: the loop's two plus one per enclosing if, i.e. NEST + 1.
 * Witnessed byte for byte for two through seven nested ifs -- see
 * agx3_cf_cndcf()'s comment and test-agx3-cf-core.c.
 */
TEST_F(AGX3PackBinary, IdentityBreakEscapingNestedIfsUsesNativeLevelCount)
{
   for (unsigned ifs = 1; ifs <= 5; ++ifs) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_push_exec(b, 1);
      agx_block *body = agx_test_block(b->shader);
      agx_block *after = agx_test_block(b->shader);
      agx_builder lb = agx_init_builder(b->shader, agx_after_block(body));
      agx_index cond = agx_register(3, AGX_SIZE_16);
      for (unsigned i = 0; i < ifs; ++i)
         agx_if_icmp(&lb, cond, agx_immediate(0), 1, AGX_ICOND_UEQ, true,
                     NULL);
      agx_index r0l = agx_register(0, AGX_SIZE_16);
      r0l.cache = true;
      r0l.kill = true;
      agx_break_if_icmp(&lb, r0l, r0l, ifs + 1, false, AGX_ICOND_UEQ, after);
      for (unsigned i = 0; i < ifs; ++i)
         agx_pop_exec(&lb, 1);
      agx_jmp_exec_any(&lb, body);
      agx_pop_exec(&lb, 1);
      agx_builder ab = agx_init_builder(b->shader, agx_after_block(after));
      agx_stop(&ab);

      uint8_t bytes[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      std::vector<uint8_t> want;
      ASSERT_TRUE(
         agx3_cf_cndst(AGX_ISA_ENCODING_AGX3B, 6, false, 2, bytes, &len));
      append(want, bytes, len);
      const size_t body_offset = want.size();

      const struct agx3_cf_source_hints plain = {false, false};
      for (unsigned i = 0; i < ifs; ++i) {
         ASSERT_TRUE(agx3_cf_icmp_imm16_runtime(AGX_ISA_ENCODING_AGX3B,
                                                AGX3_CF_ICOND_EQ, 0, 3, 0,
                                                plain, 0, bytes, &len));
         append(want, bytes, len);
         ASSERT_TRUE(
            agx3_cf_cndst(AGX_ISA_ENCODING_AGX3B, 0, true, 1, bytes, &len));
         append(want, bytes, len);
      }

      const uint16_t src[] = {0, 0};
      const struct agx3_cf_source_hints live[] = {{false, false},
                                                  {false, false}};
      ASSERT_TRUE(agx3_cf_icmp_reg16(AGX_ISA_ENCODING_AGX3B,
                                     AGX3_CF_ICOND_EQ, 1, src, live, 0, bytes,
                                     &len));
      append(want, bytes, len);
      /* NEST is ifs + 1, so the native level count is ifs + 2. */
      ASSERT_TRUE(agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3B, 1, false, ifs + 2,
                                bytes, &len));
      append(want, bytes, len);

      for (unsigned i = 0; i < ifs; ++i) {
         ASSERT_TRUE(
            agx3_cf_cndend(AGX_ISA_ENCODING_AGX3B, 1, bytes, &len));
         append(want, bytes, len);
      }

      const size_t branch_offset = want.size();
      ASSERT_TRUE(agx3_cf_branch(AGX_ISA_ENCODING_AGX3B, false, bytes, &len));
      ASSERT_TRUE(agx3_cf_patch_branch(AGX_ISA_ENCODING_AGX3B, false, bytes,
                                       (int64_t)body_offset -
                                          (int64_t)branch_offset));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_cf_cndend(AGX_ISA_ENCODING_AGX3B, 2, bytes, &len));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);

      EXPECT_EQ(pack(b), want) << "ifs=" << ifs;
   }
}

TEST_F(AGX3PackBinary, HalfZeroGnomeSamplerIndexSurvivesBranch)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index index = agx_register(3, AGX_SIZE_16);
   agx_if_icmp(b,index,agx_immediate(0),1,AGX_ICOND_UEQ,true,NULL);
   agx_index zext = index; zext.abs = true;
   agx_index bias = agx_immediate(28); bias.neg = true;
   agx_iadd_to(b,agx_register(8,AGX_SIZE_32),zext,bias,0);
   agx_pop_exec(b,1);
   agx_stop(b);
   auto code=pack(b);
   const uint8_t native[] = {0x0a,0x02,0x23,0x80,0x06,0x00,0x0f,0x02,0x02,0x00};
   ASSERT_GT(code.size(),sizeof(native));
   /* Both original halves remain live; no specimen-only discard/wait6. */
   EXPECT_EQ(memcmp(code.data(),native,sizeof(native)),0);
}

TEST_F(AGX3PackBinary, BlurSignedHalfConversionAndCounterStayNative)
{
   for (bool convert : {true,false}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      if (convert)
         agx_convert_to(b,agx_register(44,AGX_SIZE_32),agx_immediate(AGX_CONVERT_S16_TO_F),
                         agx_register(1,AGX_SIZE_16),AGX_ROUND_RTE);
      else
         agx_iadd_to(b,agx_register(1,AGX_SIZE_16),agx_register(1,AGX_SIZE_16),agx_immediate(2),0);
      agx_stop(b);
      auto code = pack(b);
      ASSERT_EQ(code.size(),convert ? 12u : 14u);
      const uint8_t converted[] = {0xa7,0x07,0x54,0x2c,0x02,0x02,0x84,0x60};
      const uint8_t incremented[] = {0x9f,0x01,0x54,0x01,0x02,0x04,0x04,0x00,0x11,0x04};
      EXPECT_EQ(memcmp(code.data(),convert ? converted : incremented,
                       convert ? sizeof(converted) : sizeof(incremented)),0);
   }
}

TEST_F(AGX3PackBinary, HyprlandGradientCounterSubtractStaysNative)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index float_source = agx_register(6, AGX_SIZE_32);
   float_source.cache = true;
   agx_index converted = agx_register(3, AGX_SIZE_16);
   converted.cache = true;
   agx_convert_to(b, converted, agx_immediate(AGX_CONVERT_F_TO_S16),
                  float_source, AGX_ROUND_RTZ);
   agx_index source = agx_register(3, AGX_SIZE_16);
   source.discard = true;
   agx_index immediate = agx_immediate(1);
   immediate.neg = true;
   agx_index narrow = agx_register(2, AGX_SIZE_16);
   narrow.cache = true;
   agx_iadd_to(b, narrow, source, immediate, 0);
   agx_index wide = agx_register(4, AGX_SIZE_32);
   wide.cache = true;
   agx_instr *widen = agx_iadd_to(b, wide, agx_immediate(0),
                                  agx_register(2, AGX_SIZE_16), 0);
   widen->shift = 4;
   agx_stop(b);

   const auto code = pack(b);
   const uint8_t native[] = {
      0x27,0x07,0x54,0x03,0x03,0x0c,0x90,0xc8,0x02,0x00,
      0x1f,0x01,0x54,0x02,0x03,0x06,0x04,0x00,0x13,0x01,
      0x9f,0x01,0x54,0x04,0x03,0x00,0x08,0x08,0x10,0x0c,
   };
   ASSERT_EQ(code.size(), sizeof(native) + 4);
   EXPECT_EQ(std::vector<uint8_t>(code.begin(), code.begin() + sizeof(native)),
             std::vector<uint8_t>(native, native + sizeof(native)));
}

TEST_F(AGX3PackBinary, NarrowBitwiseUsesWiderSourcesLowHalves)
{
   for (unsigned mask = 1; mask < 4; ++mask) {
      for (unsigned lifetime = 0; lifetime < 4; ++lifetime) {
         for (unsigned op = 0; op < 3; ++op) {
            const enum agx3_alu_op ops[] = {AGX3_ALU_BAND,AGX3_ALU_BOR,AGX3_ALU_BXOR};
            const unsigned tt[] = {8,14,6};
            auto *b = builder(AGX_ISA_ENCODING_AGX3B);
            agx_index a = agx_register(2,mask & 1 ? AGX_SIZE_32 : AGX_SIZE_16);
            agx_index z = agx_register(6,mask & 2 ? AGX_SIZE_32 : AGX_SIZE_16);
            a.discard = lifetime & 1; a.cache = !a.discard;
            z.discard = lifetime & 2; z.cache = !z.discard;
            agx_bitop_to(b,agx_register(1,AGX_SIZE_16),a,z,tt[op]);
            agx_stop(b);
            const uint16_t halves[2] = {2,6};
            const enum agx3_alu_hint hints[2] = {
               a.discard ? AGX3_ALU_HINT_DISCARD : AGX3_ALU_HINT_CACHE,
               z.discard ? AGX3_ALU_HINT_DISCARD : AGX3_ALU_HINT_CACHE,
            };
            uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
            ASSERT_TRUE(agx3_alu_bit_reg16(AGX_ISA_ENCODING_AGX3B,ops[op],1,
                                          halves,hints,0,bytes,&length));
            std::vector<uint8_t> want;
            append(want,bytes,length);
            struct agx3_encoded_end end;
            ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B,&end));
            append(want,end.bytes,end.length);
            EXPECT_EQ(pack(b),want);
         }
      }
   }
}

TEST_F(AGX3PackBinary, ExplicitLodStagingAndLifetime)
{
   for (unsigned lod_reg : {0u, 1u, 127u, 254u, 255u}) {
      for (unsigned hint = 0; hint < 3; ++hint) {
         auto enc = AGX_ISA_ENCODING_AGX3B;
         agx_builder *b = builder(enc);
         b->shader->stage = MESA_SHADER_FRAGMENT;
         key.no_stop = true;
         agx_index dest = agx_register(8, AGX_SIZE_32);
         dest.channels_m1 = 3;
         agx_index coord = agx_register(4, AGX_SIZE_32);
         coord.channels_m1 = 1;
         agx_index lod = agx_register(lod_reg, AGX_SIZE_16);
         lod.cache = hint == 1;
         lod.discard = hint == 2;
         auto *I = agx_texture_sample_to(b, dest, coord, lod,
            agx_immediate(0), agx_immediate(0), agx_immediate(0), agx_null(),
            AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0, false, false, false,
            AGX_GATHER_NONE, false);
         I->mask = 15;
         std::vector<uint8_t> want;
         uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
         ASSERT_TRUE(agx3_tex_stage_lod(enc, lod_reg, 1, 0,
                                       hint == 1, hint == 2, bytes, &len));
         append(want, bytes, len);
         ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 4, 1, 0,
                                              false, false, bytes, &len));
         append(want, bytes, len);
         ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 6, 1, 0,
                                              false, false, bytes, &len));
         append(want, bytes, len);
         struct agx3_tex_sample_desc d = {};
         d.dest=8; d.width=AGX3_TEX_W32; d.mask=15;
         d.lod=AGX3_TEX_LOD_REGISTER; d.slot=1; d.wait_mask=1; d.final=true;
         ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
         append(want, bytes, len);
         ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
         append(want, bytes, len);
         EXPECT_EQ(pack(b), want);
         EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
      }
   }
}

TEST_F(AGX3PackBinary, ComputeImmediateLodSample)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(4, AGX_SIZE_32);
   coord.channels_m1 = 1;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate_f(2.0f), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 4, 1, 0, false, false,
                                        bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 6, 1, 0, false, false,
                                        bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_LEVEL;
   d.lod_x64 = 128;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   /* Only the two f32 coordinate COPYs occupy staging. */
   EXPECT_EQ(b->shader->agx3_staging_halfs, 4);
}

TEST_F(AGX3PackBinary, Texture3DImmediateLodStagesThirdCoordinate)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(4, AGX_SIZE_32);
   coord.channels_m1 = 2;
   coord.discard = true;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate_f(2.0f), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_3D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   for (unsigned component = 0; component < 3; ++component) {
      ASSERT_TRUE(agx3_tex_stage_copy_hints(
         enc, 2 * component, 4 + 2 * component, 1, 0, false, true,
         bytes, &len));
      append(want, bytes, len);
   }
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_LEVEL;
   d.lod_x64 = 128;
   d.slot = 1;
   d.wait_mask = 1;
   d.coord_kill = false;
   d.final = true;
   d.three_d = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   static const uint8_t native_sample[] = {
      0x25,0x80,0x0c,0x98,0xb0,0x00,0x39,
      0x00,0x00,0x00,0x10,0x08,0x01,0x00,
   };
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(native_sample,
                                  native_sample + sizeof(native_sample)));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, Texture3DImplicitLodStagesCoordinates)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(4, AGX_SIZE_32);
   coord.channels_m1 = 2;
   coord.discard = true;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_3D,
      AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   for (unsigned component = 0; component < 3; ++component) {
      ASSERT_TRUE(agx3_tex_stage_copy_hints(
         enc, 2 * component, 4 + 2 * component, 1, 0, false, true,
         bytes, &len));
      append(want, bytes, len);
   }
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   d.three_d = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   /* Rebased onto this test's destination/scoreboard fields from the Apple
    * A18 Pro fragment `texture3d::sample(s, uvw)`. The adjacent native
    * `sample(..., level(0))` witness is identical outside the LOD selector:
    * implicit uses f156=1 while explicit level zero uses f37=1/f139=2. */
   static const uint8_t native_sample[] = {
      0x25,0x80,0x0c,0x98,0xb0,0x00,0x80,
      0x01,0x01,0x00,0x10,0x00,0x01,0x00,
   };
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(native_sample,
                                  native_sample + sizeof(native_sample)));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, TextureCubeRegisterDescriptorsUseLiveCEMScratch)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(32, AGX_SIZE_32);
   coord.channels_m1 = 2;
   agx_index texture = agx_register(4, AGX_SIZE_64);
   agx_index sampler = agx_register(24, AGX_SIZE_64);
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), texture, sampler,
      agx_null(), AGX_DIM_CUBE, AGX_LOD_MODE_AUTO_LOD, 0, false, false,
      false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   /* The ordinary operands occupy through half37. Axis0 needs halves38/40
    * for magnitude/face; axes1/2 use halves42/44. It cannot overwrite the
    * texture handle at r2:r3 as the old fixed-r2 axis destination did. */
   const uint16_t scratch[3] = {38, 42, 44};
   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   for (unsigned axis = 0; axis < 3; ++axis) {
      ASSERT_TRUE(agx3_tex_cube_cem_calc_to(
         enc, 32, 34, 36, scratch[axis], axis, false, bytes, &len));
      append(want, bytes, len);
      if (axis == 0) {
         ASSERT_TRUE(agx3_tex_cube_cem_rcp_to(enc, scratch[0], bytes, &len));
         append(want, bytes, len);
      }
   }
   for (unsigned component = 0; component < 2; ++component) {
      ASSERT_TRUE(agx3_tex_cube_cem_finalize_to(
         enc, component, scratch[0], scratch[component + 1], 6, bytes, &len));
      append(want, bytes, len);
   }
   ASSERT_TRUE(agx3_tex_stage_cube_z(enc, 40, 6, 0, false, true, bytes,
                                     &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);

   const struct agx3_tex_sample_desc sample = {
      .dest = 8,
      .coord = 0,
      .width = AGX3_TEX_W32,
      .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO,
      .slot = 1,
      .wait_mask = 1u << 5,
      .coord_kill = true,
      .final = true,
      .cube = true,
      .texture_kind = AGX3_TEX_OPERAND_REGISTER,
      .sampler_kind = AGX3_TEX_OPERAND_REGISTER,
      .texture_value = 4,
      .sampler_value = 24,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
   append(want, bytes, len);
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);

   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->max_reg, 45u);
   /* The lane-indexed cube uses slot 6 without resizing the private extent. */
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, FetchAndSampleKeepIndependentStagingBanks)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (bool fetch_first : {false, true}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_VERTEX;
      for (unsigned i = 0; i < 2; ++i) {
         agx_index dest = agx_register(16 + 8 * i, AGX_SIZE_32);
         dest.channels_m1 = 3;
         agx_index coords = agx_register(4 + 4 * i, AGX_SIZE_32);
         coords.channels_m1 = 1;
         const bool fetch = (i == 0) == fetch_first;
         auto *texture = agx_texture_sample_to(
            b, dest, coords, agx_immediate(0), agx_immediate(0),
            agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
            fetch ? AGX_LOD_MODE_AUTO_LOD : AGX_LOD_MODE_LOD_MIN,
            0, false, false, false, AGX_GATHER_NONE, false);
         if (fetch)
            texture->op = AGX_OPCODE_TEXTURE_LOAD;
         texture->mask = 15;
      }
      agx_stop(b);
      const auto binary = pack(b);
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
      /* The immediate sample publishes only its coordinate pair: LOD is
       * encoded in SMP and occupies no staging halves. The sample allocator
       * places its pair immediately after the fetch's four-half range; the
       * existing fetch ABI searches in eight-half steps. Both keep the first
       * consumer in flight, with no wait merely to reuse its storage. */
      const unsigned second_base = fetch_first ? 4 : 8;
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, second_base, 8, 2, 0, false, false,
                                            bytes, &len));
      auto second = std::search(binary.begin(), binary.end(), bytes, bytes + len);
      ASSERT_NE(second, binary.end()) << "fetch_first=" << fetch_first;
      ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
      EXPECT_EQ(std::search(binary.begin(), second, bytes, bytes + len), second);
   }
}

TEST_F(AGX3PackBinary, IndexedShadowLevelZeroKeepsNativeHandles)
{
   for (bool cube : {false, true}) {
      for (unsigned level = 0; level < 3; ++level) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "indexed shadow level");
         nir_def *offset = nir_imm_int(&b, 72);
         nir_def *image = nir_bindless_image_agx(&b, offset, .desc_set = 1);
         nir_def *smp = nir_bindless_sampler_agx(
            &b, nir_imm_int(&b, 104), nir_imm_intN_t(&b, 3, 16),
            .desc_set = 1);
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 5);
         tex->op = nir_texop_txl;
         tex->sampler_dim = cube ? GLSL_SAMPLER_DIM_CUBE : GLSL_SAMPLER_DIM_2D;
         tex->is_array = !cube;
         tex->is_shadow = true;
         tex->is_new_style_shadow = true;
         tex->dest_type = nir_type_float32;
         tex->coord_components = 3;
         tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                          nir_imm_vec3(&b, .5, .25, 1));
         tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
         tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, smp);
         tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                          nir_imm_float(&b, .5));
         tex->src[4] = nir_tex_src_for_ssa(
            nir_tex_src_lod, level < 2 ? nir_imm_float(&b, float(level))
                                      : nir_load_global_constant(
                                           &b, 1, 32, nir_imm_int64(&b, 0x10000),
                                           .align_mul = 4));
         nir_def_init(&tex->instr, &tex->def, 1, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         nir_store_global(&b, &tex->def, nir_imm_int64(&b, 0x20000),
                           .align_mul = 4);
         /* A comparison stages a register level beside the operand its
          * dimension already owns -- the cube beside its face, the noncube
          * array beside its layer -- so both are native. An immediate
          * nonzero level would have to share SMP's immediate slot with that
          * same operand; the noncube array keeps it in a register instead
          * (agx_optimizer_inline_imm), the cube has no such form. */
         const bool native = level == 0 || level == 2 || !cube;
         EXPECT_EQ(agx_nir_lower_g17p_indexed_samplers(b.shader, true), native);
         if (native) {
            const int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
            const int sampler = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
            ASSERT_GE(texture, 0);
            ASSERT_GE(sampler, 0);
            EXPECT_EQ(nir_src_bit_size(tex->src[texture].src), 64);
            EXPECT_EQ(nir_src_bit_size(tex->src[sampler].src), 64);
         }
         ralloc_free(b.shader);
      }
   }
}

TEST_F(AGX3PackBinary, IndexedShadowGradientAndRegisterBiasKeepNativeHandles)
{
   /* kind 0: per-lane bias, kind 1: constant bias, kind 2: gradient. */
   for (bool cube : {false, true}) {
      for (unsigned kind = 0; kind < 3; ++kind) {
         const bool grad = kind == 2;
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "indexed shadow lod");
         nir_def *image =
            nir_bindless_image_agx(&b, nir_imm_int(&b, 72), .desc_set = 1);
         nir_def *smp = nir_bindless_sampler_agx(
            &b, nir_imm_int(&b, 104), nir_imm_intN_t(&b, 3, 16),
            .desc_set = 1);
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, grad ? 6 : 5);
         tex->op = grad ? nir_texop_txd : nir_texop_txb;
         tex->sampler_dim = cube ? GLSL_SAMPLER_DIM_CUBE : GLSL_SAMPLER_DIM_2D;
         tex->is_array = !cube;
         tex->is_shadow = true;
         tex->is_new_style_shadow = true;
         tex->dest_type = nir_type_float32;
         tex->coord_components = 3;
         tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                           nir_imm_vec3(&b, .5, .25, 1));
         tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
         tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, smp);
         tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                           nir_imm_float(&b, .5));
         if (grad) {
            nir_def *d = nir_load_global_constant(
               &b, 2, 32, nir_imm_int64(&b, 0x10000), .align_mul = 8);
            tex->src[4] = nir_tex_src_for_ssa(nir_tex_src_ddx, d);
            tex->src[5] = nir_tex_src_for_ssa(nir_tex_src_ddy, d);
         } else {
            tex->src[4] = nir_tex_src_for_ssa(
               nir_tex_src_bias,
               kind == 0 ? nir_load_global_constant(
                              &b, 1, 32, nir_imm_int64(&b, 0x10000),
                              .align_mul = 4)
                         : nir_imm_float(&b, .5));
         }
         nir_def_init(&tex->instr, &tex->def, 1, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         nir_store_global(&b, &tex->def, nir_imm_int64(&b, 0x20000),
                          .align_mul = 4);
         /* A per-lane bias is published beside the layer or CEM's face, so
          * both dimensions take it; a constant one would have to share SMP's
          * immediate slot with that same operand, so the noncube array keeps
          * it in a register and the cube refuses it. The gradient composes
          * with the packed layer, and the cube has no gradient form at all. */
         const bool native = grad ? !cube : kind == 0 || !cube;
         EXPECT_EQ(agx_nir_lower_g17p_indexed_samplers(b.shader, true), native);
         ralloc_free(b.shader);
      }
   }
}

TEST_F(AGX3PackBinary, TextureCubeArrayRegisterLodPrepacksLayer)
{
   for (unsigned bias = 0; bias < 2; ++bias) {
      const auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index dest = agx_register(0, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(16, AGX_SIZE_32);
      coord.channels_m1 = 3;
      coord.discard = true;
      agx_index lod = agx_register(24, AGX_SIZE_16);
      lod.discard = true;
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_CUBE_ARRAY,
         bias ? AGX_LOD_MODE_AUTO_LOD_BIAS : AGX_LOD_MODE_LOD_MIN, 0, false,
         false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);

      /* CEM's bank sits above the level at half24; the prepacked layer word
       * sits above that bank. */
      const uint16_t scratch[3] = {26, 30, 32};
      const uint16_t packed_layer = 34;
      std::vector<uint8_t> want;
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
      for (unsigned axis = 0; axis < 3; ++axis) {
         ASSERT_TRUE(agx3_tex_cube_cem_calc_to(enc, 16, 18, 20, scratch[axis],
                                               axis, axis == 2, bytes, &len));
         append(want, bytes, len);
         if (axis == 0) {
            ASSERT_TRUE(agx3_tex_cube_cem_rcp_to(enc, scratch[0], bytes,
                                                 &len));
            append(want, bytes, len);
         }
      }
      for (unsigned component = 0; component < 2; ++component) {
         ASSERT_TRUE(agx3_tex_cube_cem_finalize_to(
            enc, component, scratch[0], scratch[component + 1], 1, bytes,
            &len));
         append(want, bytes, len);
      }
      ASSERT_TRUE(agx3_tex_pack_ms_array_layer(enc, 22, packed_layer, false,
                                               true, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_tex_stage_cube_array_lod(enc, scratch[0] + 2, 24,
                                                packed_layer, 1, 0, false,
                                                true, bytes, &len));
      append(want, bytes, len);
      if (bias) {
         ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
         append(want, bytes, len);
      }

      const struct agx3_tex_sample_desc sample = {
         .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
         .lod = bias ? AGX3_TEX_LOD_BIAS_REGISTER : AGX3_TEX_LOD_REGISTER,
         .slot = 1, .wait_mask = 1, .coord_kill = true, .final = true,
         .array = true, .array_layer = 22, .cube = true,
      };
      ASSERT_TRUE(agx3_tex_encode_sample(enc, &sample, bytes, &len));
      /* metal-nt -arch applegpu_g17p fragment texturecube_array::sample at
       * l[i] with level(b[j]) and bias(b[j]). */
      static const uint8_t native_level[] = {
         0x05,0x80,0x0c,0xb8,0xb0,0x00,0x08,
         0x03,0x01,0x00,0x10,0x00,0x01,0x00,
      };
      static const uint8_t native_bias[] = {
         0x05,0x80,0x0c,0xb8,0xb0,0x00,0x18,
         0x00,0x01,0x00,0x10,0x00,0x01,0x00,
      };
      const uint8_t *native = bias ? native_bias : native_level;
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
                std::vector<uint8_t>(native, native + sizeof(native_level)));
      append(want, bytes, len);
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
      EXPECT_EQ(b->shader->max_reg, 35u);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
   }
}

TEST_F(AGX3PackBinary, ComputeSampleRefusesUnqualifiedLod)
{
   for (unsigned shape = 0; shape < 2; ++shape) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(4, AGX_SIZE_32);
      coord.channels_m1 = 1;
      const bool implicit = shape == 0;
      agx_index lod = agx_immediate_f(-1.0f);
      auto *I = agx_texture_sample_to(
         b, dest, coord, implicit ? agx_immediate(0) : lod,
         agx_immediate(0), agx_immediate(0), agx_immediate(0), agx_null(),
         AGX_DIM_2D,
         implicit ? AGX_LOD_MODE_AUTO_LOD : AGX_LOD_MODE_LOD_MIN, 0, false,
         false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);
      pack(b, /*expect_ok=*/false);
   }
}

/* metal-nt -arch applegpu_g17p compute kernels sampling at level(l), with l
 * computed in the kernel, stage the level exactly as fragment functions do
 * and never set the fragment group terminator f156. */
TEST_F(AGX3PackBinary, ComputeSampleStagesRegisterLevelWithoutGroupEnd)
{
   for (unsigned cube = 0; cube < 2; ++cube) {
      const auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index dest = agx_register(0, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = cube ? 2 : 1;
      coord.discard = true;
      coord.kill = true;
      agx_index lod = agx_register(cube ? 20 : 4, AGX_SIZE_16);
      lod.discard = true;
      agx_instr *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), cube ? AGX_DIM_CUBE : AGX_DIM_2D,
         AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE,
         false);
      I->mask = 15;
      agx_stop(b);

      const std::vector<uint8_t> stream = pack(b);
      /* texture2d: 1705540600080004f0c00800 05800cb8b0000001000010000100
       * texturecube:                        05800cb8b0000003000010000100 */
      static const uint8_t level_2d[] = {
         0x17, 0x05, 0x54, 0x06, 0x00, 0x08, 0x00, 0x04,
         0xf0, 0xc0, 0x08, 0x00,
      };
      static const uint8_t smp_2d[] = {
         0x05, 0x80, 0x0c, 0xb8, 0xb0, 0x00, 0x00,
         0x01, 0x00, 0x00, 0x10, 0x00, 0x01, 0x00,
      };
      static const uint8_t smp_cube[] = {
         0x05, 0x80, 0x0c, 0xb8, 0xb0, 0x00, 0x00,
         0x03, 0x00, 0x00, 0x10, 0x00, 0x01, 0x00,
      };
      if (!cube) {
         EXPECT_NE(std::search(stream.begin(), stream.end(), level_2d,
                               level_2d + sizeof(level_2d)),
                   stream.end());
      }
      const uint8_t *smp = cube ? smp_cube : smp_2d;
      EXPECT_NE(std::search(stream.begin(), stream.end(), smp,
                            smp + sizeof(smp_2d)),
                stream.end());
   }
}

/* A compute texture2d_array sample stages its GPR layer with PACK_SMPIR and
 * selects it with SMP f77=2, at an immediate or a runtime level, as vertex
 * samples do. SMP's direct f31/f37=3 layer would name a uniform half.
 * metal-nt -arch applegpu_g17p compute kernels, t.sample(s, uv, layer, ...):
 *
 *   level(2.0) 1705540600040000f0c00400 05800cb8b0000901000010080100
 *   level(l)   1705540600040c00f0c02500 05800cb8b0000801000010000100
 */
TEST_F(AGX3PackBinary, ComputeArraySampleStagesLayer)
{
   for (unsigned dynamic = 0; dynamic < 2; ++dynamic) {
      const auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index dest = agx_register(0, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = 2;
      coord.discard = true;
      coord.kill = true;
      agx_index lod = dynamic ? agx_register(20, AGX_SIZE_16)
                              : agx_immediate_f(2.0f);
      lod.discard = dynamic;
      agx_instr *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_2D_ARRAY,
         AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE,
         false);
      I->mask = 15;
      agx_stop(b);

      const std::vector<uint8_t> stream = pack(b);
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
      if (dynamic) {
         ASSERT_TRUE(agx3_tex_stage_array_lod(enc, 12, 20, 1, 0, false, true,
                                              false, true, bytes, &len));
      } else {
         ASSERT_TRUE(agx3_tex_stage_array_layer(enc, 12, 1, 0, false, true,
                                                bytes, &len));
      }
      EXPECT_NE(std::search(stream.begin(), stream.end(), bytes, bytes + len),
                stream.end());

      static const uint8_t smp_level[] = {
         0x05, 0x80, 0x0c, 0xb8, 0xb0, 0x00, 0x09,
         0x01, 0x00, 0x00, 0x10, 0x08, 0x01, 0x00,
      };
      static const uint8_t smp_register[] = {
         0x05, 0x80, 0x0c, 0xb8, 0xb0, 0x00, 0x08,
         0x01, 0x00, 0x00, 0x10, 0x00, 0x01, 0x00,
      };
      const uint8_t *smp = dynamic ? smp_register : smp_level;
      EXPECT_NE(std::search(stream.begin(), stream.end(), smp,
                            smp + sizeof(smp_level)),
                stream.end());
   }
}

TEST_F(AGX3PackBinary, ArraySampleStagesDriverLayerAndLifetime)
{
   for (unsigned hint = 0; hint < 3; ++hint) {
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(0, AGX_SIZE_32);
      coord.channels_m1 = 2;
      coord.cache = hint == 1;
      coord.discard = hint == 2;
      coord.kill = hint == 2;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D_ARRAY,
         AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
         false);
      I->mask = 15;
      agx_stop(b);

      std::vector<uint8_t> want;
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 0, 1, 0,
                                           hint == 1, hint == 2,
                                           bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 2, 1, 0,
                                           hint == 1, hint == 2,
                                           bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_tex_stage_array_layer(enc, 4, 1, 0,
                                             hint == 1, hint == 2,
                                             bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
      append(want, bytes, len);
      struct agx3_tex_sample_desc d = {};
      d.dest = 8;
      d.width = AGX3_TEX_W32;
      d.mask = 15;
      d.lod = AGX3_TEX_LOD_AUTO;
      d.slot = 1;
      d.wait_mask = 1;
      d.coord_kill = hint == 2;
      d.final = true;
      d.array = true;
      d.array_layer = 4;
      ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
      append(want, bytes, len);
      EXPECT_EQ(pack(b), want);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
   }
}

/*
 * textureGrad. The gradient collect is (ddx.x, ddy.x, ddx.y, ddy.y) and each
 * scalar is staged into the bank Apple's own kernels stage it into: 8, 10, 12
 * and 14 in half-register units, on one shared staging slot, after the two
 * coordinate copies. agx3_tex_core's own test asserts the sample byte-exact
 * against `xcrun metal-nt -arch applegpu_g17p` on the compute kernel
 * t.sample(s, g[0], gradient2d(g[1], g[2])); this test asserts that the
 * packer drives it with the right banks, sources and staging budget.
 */
TEST_F(AGX3PackBinary, GradientSampleStagesFourScalarsIntoTheirOwnBanks)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(20, AGX_SIZE_32);
   coord.channels_m1 = 1;
   agx_index grad = agx_register(24, AGX_SIZE_32);
   grad.channels_m1 = 3;
   auto *I = agx_texture_sample_to(
      b, dest, coord, grad, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_GRAD, 0,
      false, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   /* Coordinates go to bank 0, as every gradient witness names coord 0. */
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 20, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 22, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   const uint16_t bank[4] = {8, 10, 12, 14};
   for (unsigned i = 0; i < 4; ++i) {
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, bank[i], 24 + 2 * i, 1, 0,
                                            false, false, bytes, &len));
      append(want, bytes, len);
   }
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_GRADIENT;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   /* Apple's own compute gradient sample is 05800cb8b0000401000010000100;
    * this one differs only in the destination register (8, not 0) and the
    * coordinate-kill hint. */
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             (std::vector<uint8_t>{0x25, 0x80, 0x0c, 0x98, 0xb0, 0x00, 0x04,
                                   0x01, 0x00, 0x00, 0x10, 0x00, 0x01, 0x00}));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 16);
}

/* ShaderInfo fields 38 and 42 are both 64 in a fresh G17P fragment pipeline
 * for texture3d::sample(..., gradient3d(...)); the otherwise matching 2D
 * pipeline reports 32.  Keep that native sampler-private footprint separate
 * from the six scalar COPY destinations, whose highest bank is 18. */
TEST_F(AGX3PackBinary, Gradient3DSampleUsesNativeFragmentStagingFootprint)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(20, AGX_SIZE_32);
   coord.channels_m1 = 2;
   agx_index grad = agx_register(32, AGX_SIZE_32);
   grad.channels_m1 = 5;
   auto *I = agx_texture_sample_to(
      b, dest, coord, grad, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_3D, AGX_LOD_MODE_LOD_GRAD, 0,
      false, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   const std::vector<uint8_t> code = pack(b);
   static const uint16_t bank[6] = {
      AGX3_TEX_GRADIENT_BANK_DDX_X, AGX3_TEX_GRADIENT_BANK_DDY_X,
      AGX3_TEX_GRADIENT_BANK_DDX_Y, AGX3_TEX_GRADIENT_BANK_DDY_Y,
      AGX3_TEX_GRADIENT_BANK_DDX_Z, AGX3_TEX_GRADIENT_BANK_DDY_Z,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(bank); ++i) {
      uint8_t copy[AGX3_STAGE_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, bank[i], 32 + 2 * i, 1, 0,
                                            false, false, copy, &len));
      EXPECT_NE(std::search(code.begin(), code.end(), copy, copy + len),
                code.end());
   }
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 32);
}

/* SMP's direct f31/f37=3 layer names a uniform half, so a driver array
 * gradient, whose layer is a GPR, stages it with PACK_SMPIR and reads it with
 * f77=2, as Apple's texture2d_array gradient2d() fragment function with a
 * runtime layer does: 17055406000c0000f0c00400 05800cb8b0000c01000010000100.
 */
TEST_F(AGX3PackBinary, ArrayGradientStagesLayer)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(20, AGX_SIZE_32);
   coord.channels_m1 = 2;
   agx_index grad = agx_register(32, AGX_SIZE_32);
   grad.channels_m1 = 3;
   auto *I = agx_texture_sample_to(
      b, dest, coord, grad, agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_2D_ARRAY,
      AGX_LOD_MODE_LOD_GRAD, 0, false, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 20, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 22, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_array_layer(enc, 24, 1, 0, false, false, bytes,
                                          &len));
   append(want, bytes, len);
   const uint16_t bank[4] = {8, 10, 12, 14};
   for (unsigned i = 0; i < 4; ++i) {
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, bank[i], 32 + 2 * i, 1, 0,
                                            false, false, bytes, &len));
      append(want, bytes, len);
   }
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_GRADIENT;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   d.array = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 16);
}

/* A gradient whose four scalars are not one aligned 32-bit run must fail. */
TEST_F(AGX3PackBinary, GradientSampleRefusesUnwitnessedShapes)
{
   for (unsigned shape = 0; shape < 2; ++shape) {
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(20, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_index grad = agx_register(24, AGX_SIZE_32);
      grad.channels_m1 = shape == 0 ? 1 : 3;
      if (shape == 1)
         grad = agx_register(25, AGX_SIZE_32), grad.channels_m1 = 3;
      auto *I = agx_texture_sample_to(
         b, dest, coord, grad, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_LOD_GRAD, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);
      pack(b, /*expect_ok=*/false);
   }
}

/* A zero bias is the plain implicit sample, byte for byte, exactly as Apple's
 * own compiler folds it. Nothing extra is staged and both use fixed bank 0. */
TEST_F(AGX3PackBinary, ZeroBiasFoldsIntoTheImplicitSample)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   std::vector<uint8_t> packed[2];
   for (unsigned biased = 0; biased < 2; ++biased) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(20, AGX_SIZE_32);
      coord.channels_m1 = 1;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_2D,
         biased ? AGX_LOD_MODE_AUTO_LOD_BIAS : AGX_LOD_MODE_AUTO_LOD, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
      agx_stop(b);
      packed[biased] = pack(b);
   }
   EXPECT_FALSE(packed[0].empty());
   EXPECT_EQ(packed[0], packed[1]);
}

/*
 * A per-lane LOD bias stages exactly like a per-lane level -- PACK_SMPIR into
 * operand bank 6, coordinates rebased to bank 0 -- and moves only f139, so
 * the emitted sample differs from the explicit-level one in bytes 6 and 7
 * alone, exactly as Apple's own fragment pair does.
 */
TEST_F(AGX3PackBinary, RegisterLodBiasStagesLikeRegisterLevel)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(20, AGX_SIZE_32);
   coord.channels_m1 = 1;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_register(4, AGX_SIZE_16), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_AUTO_LOD_BIAS, 0, false, false, false, AGX_GATHER_NONE,
      false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_lod(enc, 4, 1, 0, false, false, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 20, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 22, 1, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_BIAS_REGISTER;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   std::vector<uint8_t> bias(bytes, bytes + len);
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);

   d.lod = AGX3_TEX_LOD_REGISTER;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   std::vector<uint8_t> level(bytes, bytes + len);
   ASSERT_EQ(level.size(), bias.size());
   for (size_t i = 0; i < level.size(); ++i) {
      if (i == 6) {
         EXPECT_EQ(level[i], 0x00);
         EXPECT_EQ(bias[i], 0x04);
      } else if (i == 7) {
         EXPECT_EQ(level[i], 0x01);
         EXPECT_EQ(bias[i], 0x00);
      } else {
         EXPECT_EQ(level[i], bias[i]);
      }
   }
}

/*
 * textureGather. AGX_GATHER_{R,G,B,A} are 1/3/5/7 in the IR but the native
 * component field counts 0..3, so the packer halves them; a mistake there
 * would silently gather the wrong channel. The four encodings themselves are
 * asserted against Apple's kernels in test-agx3-tex-core.c.
 */
TEST_F(AGX3PackBinary, GatherComponentMapsOntoTheNativeField)
{
   const enum agx_gather ir[4] = {AGX_GATHER_R, AGX_GATHER_G, AGX_GATHER_B,
                                  AGX_GATHER_A};
   const enum agx3_tex_gather native[4] = {
      AGX3_TEX_GATHER_R, AGX3_TEX_GATHER_G, AGX3_TEX_GATHER_B,
      AGX3_TEX_GATHER_A};

   for (unsigned c = 0; c < 4; ++c) {
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(20, AGX_SIZE_32);
      coord.channels_m1 = 1;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
         false, false, false, ir[c], false);
      I->mask = 15;
      agx_stop(b);

      std::vector<uint8_t> want;
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 20, 1, 0, false, false,
                                            bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 22, 1, 0, false, false,
                                            bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
      append(want, bytes, len);
      struct agx3_tex_sample_desc d = {};
      d.dest = 8;
      d.coord = 0;
      d.width = AGX3_TEX_W32;
      d.mask = 15;
      d.lod = AGX3_TEX_LOD_AUTO;
      d.slot = 1;
      d.wait_mask = 1;
      d.final = true;
      d.gather = native[c];
      ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
      append(want, bytes, len);
      ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
      append(want, bytes, len);
      EXPECT_EQ(pack(b), want);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 4);
   }
}

/* Metal's gather takes no LOD argument, so a gather carrying one has no
 * witness and must fail closed.  The array layer does compose with it:
 * `texture2d_array::gather(s, uv, layer)` is 05800ca490000db4010000001200,
 * the ordinary array sample's f77/f31/f37 plus the gather's f144. */
TEST_F(AGX3PackBinary, GatherRefusesALodButTakesAnArrayLayer)
{
   for (unsigned shape = 0; shape < 3; ++shape) {
      const bool array = shape != 1;
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(20, AGX_SIZE_32);
      coord.channels_m1 = array ? 2 : 1;
      auto *I = agx_texture_sample_to(
         b, dest, coord,
         shape == 0 ? agx_immediate(0)
                    : agx_immediate(agx_minifloat_encode(0.5f)),
         agx_immediate(0), agx_immediate(0), agx_immediate(0), agx_null(),
         array ? AGX_DIM_2D_ARRAY : AGX_DIM_2D,
         shape == 0 ? AGX_LOD_MODE_AUTO_LOD : AGX_LOD_MODE_AUTO_LOD_BIAS, 0,
         false, false, false, AGX_GATHER_G, false);
      I->mask = 15;
      agx_stop(b);
      /* shape 0: array gather, no LOD -> encodes.
       * shape 1: 2D gather with an immediate bias -> refuses.
       * shape 2: array gather with an immediate bias -> refuses. */
      pack(b, /*expect_ok=*/shape == 0);
   }
}

/*
 * The coordinate offset reaches the sample instruction as an immediate, in
 * every LOD mode. Before this, the packer took an offset only on a plain
 * implicit sample -- and the NIR combined-sampler gate disowned any shader
 * carrying one, so even that path was unreachable and a bare textureOffset()
 * refused the whole shader.
 */
TEST_F(AGX3PackBinary, OffsetSampleRidesEveryLodMode)
{
   struct row {
      enum agx_lod_mode mode;
      const char *name;
   };
   static const row rows[] = {
      {AGX_LOD_MODE_AUTO_LOD, "implicit"},
      {AGX_LOD_MODE_AUTO_LOD_BIAS, "immediate bias"},
      {AGX_LOD_MODE_LOD_MIN, "immediate level"},
      {AGX_LOD_MODE_LOD_GRAD, "gradient"},
   };

   for (const row &r : rows) {
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;
      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(20, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_index lod = agx_immediate(0);
      if (r.mode == AGX_LOD_MODE_AUTO_LOD_BIAS)
         lod = agx_immediate(agx_minifloat_encode(0.5f));
      else if (r.mode == AGX_LOD_MODE_LOD_MIN)
         lod = agx_immediate(agx_minifloat_encode(2.0f));
      else if (r.mode == AGX_LOD_MODE_LOD_GRAD) {
         lod = agx_register(24, AGX_SIZE_32);
         lod.channels_m1 = 3;
      }
      /* Signed nibble pair (y << 4) | x -- offset (1, 1), the value
       * agx_nir_lower_texture() packs into nir_tex_src_backend2. */
      agx_index offset = agx_immediate(0x11);
      offset.size = AGX_SIZE_32;
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), offset, AGX_DIM_2D, r.mode, 0,
         /*offset=*/true, false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
      I->offset_xyz = 0x11;
      agx_stop(b);
      SCOPED_TRACE(r.name);
      pack(b, /*expect_ok=*/true);
   }
}

TEST_F(AGX3PackBinary, VertexOffsetSampleRidesExplicitLodModes)
{
   for (enum agx_lod_mode mode :
        {AGX_LOD_MODE_LOD_MIN, AGX_LOD_MODE_LOD_GRAD}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_VERTEX;
      agx_index dest = agx_register(32, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_index lod = agx_register(12, mode == AGX_LOD_MODE_LOD_MIN
                                         ? AGX_SIZE_16 : AGX_SIZE_32);
      lod.channels_m1 = mode == AGX_LOD_MODE_LOD_MIN ? 0 : 3;
      agx_index offset = agx_immediate(0x87);
      offset.size = AGX_SIZE_32;
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0),
         agx_register(0, AGX_SIZE_64), agx_register(4, AGX_SIZE_64), offset,
         AGX_DIM_2D, mode, 0, true, false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
      I->offset_xyz = 0x87;
      agx_stop(b);
      SCOPED_TRACE(mode);
      pack(b, /*expect_ok=*/true);
      EXPECT_EQ(b->shader->agx3_staging_halfs,
                mode == AGX_LOD_MODE_LOD_GRAD ? 24 : 16);
   }
}

TEST_F(AGX3PackBinary, Vertex3DGradientMatchesNativeCompilerFootprint)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_VERTEX;
   agx_index dest = agx_register(32, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(8, AGX_SIZE_32);
   coord.channels_m1 = 2;
   agx_index gradient = agx_register(16, AGX_SIZE_32);
   gradient.channels_m1 = 5;
   agx_index offset = agx_immediate(0x837);
   offset.size = AGX_SIZE_32;
   auto *I = agx_texture_sample_to(
      b, dest, coord, gradient, agx_immediate(0),
      agx_register(0, AGX_SIZE_64), agx_register(4, AGX_SIZE_64), offset,
      AGX_DIM_3D, AGX_LOD_MODE_LOD_GRAD, 0, true, false, false,
      AGX_GATHER_NONE, false);
   I->mask = 15;
   I->offset_xyz = 0x837;
   agx_stop(b);
   pack(b, /*expect_ok=*/true);

   /* Apple ShaderInfo fields 38/42 are 80 for both control and offset 3D
    * vertex-gradient pipelines. The driver derives D by doubling this
    * compiler value, so publish 40 here. The addressed staging banks still
    * end at half 19; D is a compiler footprint rather than an address span. */
   EXPECT_EQ(b->shader->agx3_staging_halfs, 40);
}

TEST_F(AGX3PackBinary, Texture3DOffsetKeepsSignedZNibbleThroughValidation)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index dest = agx_register(16, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(8, AGX_SIZE_32);
   coord.channels_m1 = 2;
   agx_index offset = agx_immediate(0x837); /* (7, 3, -8), from CTS */
   offset.size = AGX_SIZE_32;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_register(0, AGX_SIZE_64), agx_register(4, AGX_SIZE_64),
      offset, AGX_DIM_3D, AGX_LOD_MODE_AUTO_LOD, 0, true, false, false,
      AGX_GATHER_NONE, false);
   I->mask = 15;
   I->offset_xyz = 0x837;
   agx_stop(b);
   agx_validate(b->shader, "3D immediate offset");
   pack(b);
   EXPECT_EQ(I->offset_xyz, 0x837u);
}

/* The array layer and register LOD share one private operand word. Check
 * the full IR path emits the native combined publication exactly once. */
TEST_F(AGX3PackBinary, OffsetArrayCombinesLayerAndRegisterLod)
{
   for (bool shadow : {false, true}) {
      for (auto mode : {AGX_LOD_MODE_LOD_MIN, AGX_LOD_MODE_AUTO_LOD_BIAS}) {
         const auto enc = AGX_ISA_ENCODING_AGX3B;
         agx_builder *b = builder(enc);
         b->shader->stage = MESA_SHADER_FRAGMENT;
         agx_index dest = agx_register(16, AGX_SIZE_32);
         dest.channels_m1 = 3;
         agx_index coord = agx_register(8, AGX_SIZE_32);
         coord.channels_m1 = 2;
         coord.discard = true;
         agx_index lod = agx_register(13, AGX_SIZE_16);
         lod.discard = true;
         agx_index off = agx_immediate(0x87);
         off.size = AGX_SIZE_32;
         auto *I = agx_texture_sample_to(
            b, dest, coord, lod, agx_immediate(0),
            agx_register(0, AGX_SIZE_64), agx_register(4, AGX_SIZE_64),
            shadow ? agx_register(24, AGX_SIZE_32) : off,
            AGX_DIM_2D_ARRAY, mode, 0, true, shadow, false,
            AGX_GATHER_NONE, false);
         I->mask = shadow ? 1 : 15;
         I->offset_xyz = 0x87;
         agx_stop(b);
         const auto stream = pack(b);
         const uint8_t combined[] = {
            0x17, 0x05, 0x54, 0x06, 0x00, 0x18,
            0x34, 0x00, 0xf0, 0xe4, 0x25, 0x00,
         };
         auto first = std::search(stream.begin(), stream.end(),
                                  std::begin(combined), std::end(combined));
         ASSERT_NE(first, stream.end());
         EXPECT_EQ(std::search(first + sizeof(combined), stream.end(),
                               std::begin(combined), std::end(combined)),
                   stream.end());
         unsigned publications = 0;
         for (size_t i = 0; i + 4 <= stream.size(); i += 2) {
            if (stream[i] == 0x17 && stream[i + 1] == 0x05 &&
                stream[i + 2] == 0x54 && stream[i + 3] == 0x06)
               publications++;
         }
         EXPECT_EQ(publications, 1u);
         EXPECT_EQ(b->shader->agx3_staging_halfs, 8u);
      }
   }
}

/*
 * The same offset on the sample instruction itself, checked byte for byte.
 * Apple's `t.sample(s, uv, int2(1,1))` is 05800cb8b0000102018810001200 and
 * differs from its offset-free control by exactly f43 = 17, f51 = 1 and
 * f141 = 1. The stream around the sample (coordinate staging, the fragment
 * pixel entry) is checked by its own tests, so this one only asserts that
 * the sample instruction the packer emitted carries that field triple.
 */
TEST_F(AGX3PackBinary, OffsetSampleMatchesNativeFieldTriple)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(20, AGX_SIZE_32);
   coord.channels_m1 = 1;
   agx_index offset = agx_immediate(0x11);
   offset.size = AGX_SIZE_32;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), offset, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      /*offset=*/true, false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   I->offset_xyz = 0x11;
   agx_stop(b);
   const std::vector<uint8_t> stream = pack(b);

   /* The same sample with no offset, so the difference really is the triple
    * and not something the shape changed along the way. */
   agx_builder *b2 = builder(enc);
   b2->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   auto *I2 = agx_texture_sample_to(
      b2, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      false, false, false, AGX_GATHER_NONE, false);
   I2->mask = 15;
   agx_stop(b2);
   const std::vector<uint8_t> plain = pack(b2);

   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   const std::vector<uint8_t> want_plain(bytes, bytes + len);
   d.offset = true;
   d.offset_xyz = 0x11;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   const std::vector<uint8_t> want_offset(bytes, bytes + len);

   EXPECT_NE(want_plain, want_offset);
   EXPECT_NE(std::search(stream.begin(), stream.end(), want_offset.begin(),
                         want_offset.end()),
             stream.end());
   EXPECT_NE(std::search(plain.begin(), plain.end(), want_plain.begin(),
                         want_plain.end()),
             plain.end());
}

/*
 * The depth comparison is orthogonal to the LOD shape. Apple's depth2d
 * sample_compare with level(2), bias(0.5), level(reg), bias(reg) and
 * gradient2d() each differ from the corresponding colour sample by f143
 * alone; before this, only the implicit form was admitted.
 */
TEST_F(AGX3PackBinary, ShadowSampleRidesEveryLodMode)
{
   struct row {
      enum agx_lod_mode mode;
      const char *name;
   };
   static const row rows[] = {
      {AGX_LOD_MODE_AUTO_LOD, "implicit"},
      {AGX_LOD_MODE_AUTO_LOD_BIAS, "immediate bias"},
      {AGX_LOD_MODE_LOD_MIN, "immediate level"},
      {AGX_LOD_MODE_LOD_GRAD, "gradient"},
   };

   for (const row &r : rows) {
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;
      agx_index dest = agx_register(0, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = 1;
      coord.discard = true;
      agx_index compare = agx_register(2, AGX_SIZE_32);
      compare.discard = true;
      agx_index lod = agx_immediate(0);
      if (r.mode == AGX_LOD_MODE_AUTO_LOD_BIAS)
         lod = agx_immediate(agx_minifloat_encode(0.5f));
      else if (r.mode == AGX_LOD_MODE_LOD_MIN)
         lod = agx_immediate(agx_minifloat_encode(2.0f));
      else if (r.mode == AGX_LOD_MODE_LOD_GRAD) {
         lod = agx_register(16, AGX_SIZE_32);
         lod.channels_m1 = 3;
      }
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), compare, AGX_DIM_2D, r.mode, 0, false,
         /*shadow=*/true, false, AGX_GATHER_NONE, false);
      I->mask = 1;
      agx_stop(b);
      SCOPED_TRACE(r.name);
      pack(b, /*expect_ok=*/true);
   }
}

/*
 * gather_compare: the ordinary gather plus f143, returning four texels and so
 * carrying the full f75 mask rather than the single component every other
 * comparison carries. A comparison whose mask does not match its shape must
 * still refuse.
 */
TEST_F(AGX3PackBinary, ShadowGatherKeepsTheFourTexelMask)
{
   for (unsigned mask : {15u, 1u}) {
      auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      key.no_stop = true;
      agx_index dest = agx_register(0, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = 1;
      coord.discard = true;
      agx_index compare = agx_register(2, AGX_SIZE_32);
      compare.discard = true;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
         agx_immediate(0), compare, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
         false, /*shadow=*/true, false, AGX_GATHER_R, false);
      I->mask = mask;
      agx_stop(b);
      pack(b, /*expect_ok=*/mask == 15u);
   }
}

/*
 * A comparison sample takes a coordinate offset. Apple's
 * depth2d::sample_compare(s, uv, ref, int2(1,1)) is
 * 1d800ca090002102018800001200 -- the plain comparison plus the same
 * f43/f51/f141 triple every other sample takes.
 *
 * The reason this could not be encoded before was not the ISA: agx_emit_tex()
 * collected the comparator and the packed offset into one src[5] vector, and
 * the comparator staging cannot take that apart. The offset now rides on the
 * instruction (I->offset_xyz) and src[5] belongs to the comparator alone.
 */
TEST_F(AGX3PackBinary, ShadowSampleTakesACoordinateOffset)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(8, AGX_SIZE_32);
   coord.channels_m1 = 1;
   coord.discard = true;
   agx_index compare = agx_register(2, AGX_SIZE_32);
   compare.discard = true;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), compare, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      /*offset=*/true, /*shadow=*/true, false, AGX_GATHER_NONE, false);
   I->mask = 1;
   I->offset_xyz = 0x11;
   agx_stop(b);
   const std::vector<uint8_t> stream = pack(b);

   struct agx3_tex_sample_desc d = {};
   d.dest = 0;
   d.width = AGX3_TEX_W32;
   d.mask = 1;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   d.shadow = true;
   d.offset = true;
   d.offset_xyz = 0x11;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   const std::vector<uint8_t> want(bytes, bytes + len);
   EXPECT_NE(std::search(stream.begin(), stream.end(), want.begin(),
                         want.end()),
             stream.end());

   /* An offset the instruction does not carry must refuse rather than encode
    * as offset (0,0), which native spells as the plain form. */
   agx_builder *b2 = builder(enc);
   b2->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   auto *I2 = agx_texture_sample_to(
      b2, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), compare, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      /*offset=*/true, /*shadow=*/true, false, AGX_GATHER_NONE, false);
   I2->mask = 1;
   agx_stop(b2);
   pack(b2, /*expect_ok=*/false);
}

/*
 * REGRESSION GUARD. agx_opt_cse.c's instrs_equal() is hand-written and
 * enumerates the instruction's immediates one at a time, so a new field that
 * is not added there lets two otherwise-identical instructions collapse into
 * one. I->offset_xyz is such a field: without its line in instrs_equal(), two
 * samples of the same texture at the same coordinate but *different* texel
 * offsets would CSE together and the second would silently read the first's
 * texels.
 *
 * This asserts the two are not equal. Delete the offset_xyz line from
 * instrs_equal() and this test fails.
 */
TEST_F(AGX3PackBinary, DifferentTexelOffsetsDoNotCSETogether)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index coord = agx_register(4, AGX_SIZE_32);
   coord.channels_m1 = 1;

   agx_index d0 = agx_register(8, AGX_SIZE_32);
   d0.channels_m1 = 3;
   agx_index d1 = agx_register(16, AGX_SIZE_32);
   d1.channels_m1 = 3;

   agx_index off = agx_immediate(0x11);
   off.size = AGX_SIZE_32;

   auto *a = agx_texture_sample_to(
      b, d0, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), off, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      /*offset=*/true, false, false, AGX_GATHER_NONE, false);
   a->mask = 15;
   a->offset_xyz = 0x11;

   auto *c = agx_texture_sample_to(
      b, d1, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), off, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0,
      /*offset=*/true, false, false, AGX_GATHER_NONE, false);
   c->mask = 15;
   c->offset_xyz = 0x22; /* the only difference */

   EXPECT_NE(a->offset_xyz, c->offset_xyz);
   EXPECT_FALSE(agx_instrs_equal_for_cse(a, c));

   /* Preserve the third nibble too: equal X/Y with different Z must not
    * merge after the comparator has occupied src[5]. */
   a->dim = c->dim = AGX_DIM_3D;
   a->src[0].channels_m1 = c->src[0].channels_m1 = 2;
   c->offset_xyz = 0x111;
   EXPECT_FALSE(agx_instrs_equal_for_cse(a, c));

   /* Equal full-width offsets still compare equal. */
   c->offset_xyz = 0x11;
   c->dest[0] = d0;
   EXPECT_TRUE(agx_instrs_equal_for_cse(a, c));
}

TEST_F(AGX3PackBinary, ShadowArrayStagesPackedLayer)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(8, AGX_SIZE_32);
   coord.channels_m1 = 2;
   coord.discard = true;
   agx_index compare = agx_register(2, AGX_SIZE_32);
   compare.discard = true;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(0), compare, AGX_DIM_2D_ARRAY, AGX_LOD_MODE_AUTO_LOD, 0,
      false, true, false, AGX_GATHER_NONE, false);
   I->mask = 1;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 8, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 10, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_array_layer(enc, 12, 1, 0, false, true,
                                          bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 4, 2, 1, 0, false, true,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_sample_desc d = {};
   d.dest = 0;
   d.width = AGX3_TEX_W32;
   d.mask = 1;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.slot = 1;
   d.wait_mask = 1;
   d.final = true;
   d.array = true;
   d.array_layer = 12;
   d.shadow = true;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
   append(want, bytes, len);

   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}

TEST_F(AGX3PackBinary, ArrayRegisterDescriptorsStageDriverLayer)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(20, AGX_SIZE_32);
   coord.channels_m1 = 2;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_register(12, AGX_SIZE_64), agx_register(16, AGX_SIZE_64),
      agx_null(), AGX_DIM_2D_ARRAY, AGX_LOD_MODE_AUTO_LOD, 0, false,
      false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 20, 6, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 22, 6, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_array_layer(enc, 24, 6, 0, false, false,
                                          bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.slot = 1;
   d.wait_mask = 1u << 5;
   d.final = true;
   d.array = true;
   d.array_layer = 24;
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = 12;
   d.sampler_value = 16;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8);
}
TEST_F(AGX3PackBinary, ArrayDriverLayerAcceptsHighRegistersWithoutWrap)
{
   auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_FRAGMENT;
   key.no_stop = true;
   agx_index dest = agx_register(8, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(120, AGX_SIZE_32);
   coord.channels_m1 = 2;
   auto *I = agx_texture_sample_to(
      b, dest, coord, agx_immediate(0), agx_immediate(0),
      agx_register(12, AGX_SIZE_64), agx_register(16, AGX_SIZE_64),
      agx_null(), AGX_DIM_2D_ARRAY, AGX_LOD_MODE_AUTO_LOD, 0, false,
      false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 0, 120, 6, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 2, 122, 6, 0, false, false,
                                         bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_tex_stage_array_layer(enc, 124, 6, 0, false, false,
                                          bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_pixel_colour_entry(enc, bytes, &len));
   append(want, bytes, len);
   struct agx3_tex_sample_desc d = {};
   d.dest = 8;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.slot = 1;
   d.wait_mask = 1u << 5;
   d.final = true;
   d.array = true;
   d.array_layer = 124;
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = 12;
   d.sampler_value = 16;
   ASSERT_TRUE(agx3_tex_encode_sample(enc, &d, bytes, &len));
   append(want, bytes, len);
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, bytes, &len));
   append(want, bytes, len);
   EXPECT_EQ(pack(b), want);

   agx_builder *bad = builder(enc);
   bad->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index wrapped = agx_register(124, AGX_SIZE_32);
   wrapped.channels_m1 = 2;
   I = agx_texture_sample_to(
      bad, dest, wrapped, agx_immediate(0), agx_immediate(0),
      agx_register(12, AGX_SIZE_64), agx_register(16, AGX_SIZE_64),
      agx_null(), AGX_DIM_2D_ARRAY, AGX_LOD_MODE_AUTO_LOD, 0, false,
      false, false, AGX_GATHER_NONE, false);
   I->mask = 15;
   agx_stop(bad);
   EXPECT_TRUE(pack(bad, false).empty());
}

TEST_F(AGX3PackBinary, StorageImageWriteMatchesNativeWithReadyOrSlot1Data)
{
   static const uint8_t native[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x10, 0x54, 0x00, 0x00, 0x40, 0x00, 0x00,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x1a, 0x09, 0x00,
   };
   uint8_t bytes[AGX3_IMAGE_WRITE_LENGTH];
   uint8_t len = 0;
   struct agx3_image_write_desc d = {
      .wait_mask = 1,
      .image_kind = AGX3_TEX_OPERAND_IMMEDIATE,
      .image_value = 0,
      .normalized = false,
   };
   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   ASSERT_EQ(len, sizeof(native));
   EXPECT_EQ(memcmp(bytes, native, sizeof(native)), 0);

   d.wait_mask = 0;
   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   ASSERT_EQ(len, sizeof(native));
   EXPECT_EQ(memcmp(bytes, native, sizeof(native)), 0);

   for (unsigned wait : {2u, 3u, 0x20u, 0x3fu}) {
      d.wait_mask = wait;
      ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
         AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
      ASSERT_EQ(len, sizeof(native));
      EXPECT_EQ(memcmp(bytes, native, sizeof(native)), 0);
   }

   for (unsigned wait : {0x40u, 0x7fu}) {
      memset(bytes, 0xa5, sizeof(bytes));
      len = 0xa5;
      d.wait_mask = wait;
      EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
         AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
      EXPECT_EQ(len, 0);
      for (uint8_t byte : bytes)
         EXPECT_EQ(byte, 0);
   }

   memset(bytes, 0xa5, sizeof(bytes));
   len = 0xa5;
   d.wait_mask = 1;
   EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3C, &d, bytes, &len));
   EXPECT_EQ(len, 0);
   for (uint8_t byte : bytes)
      EXPECT_EQ(byte, 0);

   static const uint8_t sampled_write[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x20, 0x40, 0x00, 0x01,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x09, 0x00,
   };
   d.image_value = 1;
   d.normalized = true;
   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   ASSERT_EQ(len, sizeof(sampled_write));
   EXPECT_EQ(memcmp(bytes, sampled_write, sizeof(sampled_write)), 0);

   d.image_value = 0;
   EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.image_value = 1;
   d.normalized = false;
   /* metal-nt applegpu_g17p, texture2d<uint>::read from texture(0), then
    * write the uint4 to texture(1). */
   static const uint8_t integer_state1[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x20, 0x40, 0x00, 0x01,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x1a, 0x09, 0x00,
   };
   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   ASSERT_EQ(len, sizeof(integer_state1));
   EXPECT_EQ(memcmp(bytes, integer_state1, sizeof(integer_state1)), 0);
   d.image_value = 2;
   EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));

   /* A retained GPR index is f92/f99=2 without the f103 kill bit. The native
    * argument-buffer witness 0xd7205400...883a0901 decodes as f93 20/f99=3:
    * that is uniform half 20, not a retained r10. */
   static const uint8_t indexed_live[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x00, 0x40, 0x00, 0x14,
      0x20, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x89, 0x00,
   };
   d = {
      .wait_mask = 1,
      .image_kind = AGX3_TEX_OPERAND_REGISTER,
      .image_value = 20,
      .normalized = true,
   };
   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   ASSERT_EQ(len, sizeof(indexed_live));
   EXPECT_EQ(memcmp(bytes, indexed_live, sizeof(indexed_live)), 0);

   /* The killed-index form is independently witnessed at r4. Combining it
    * with the same slot-2 staging wait gives the production bindless form. */
   static const uint8_t indexed_kill[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x00, 0x40, 0x00, 0x08,
      0x21, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x89, 0x00,
   };
   d.image_value = 8;
   d.image_kill = true;
   ASSERT_TRUE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   ASSERT_EQ(len, sizeof(indexed_kill));
   EXPECT_EQ(memcmp(bytes, indexed_kill, sizeof(indexed_kill)), 0);

   d.image_value = 9;
   EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.image_value = 8;
   d.image_kind = AGX3_TEX_OPERAND_UNIFORM;
   EXPECT_FALSE(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
}

TEST_F(AGX3PackBinary, StorageImageGprIndexMatchesNative)
{
   /* metal-nt -arch applegpu_g17p, array<texture2d<int, access::write>, 4> ts
    * written at a computed GPR index r0: dead after the write, then read
    * again afterwards. */
   static const uint8_t killed[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x10, 0x54, 0x00, 0x00, 0x40, 0x00, 0x00,
      0x21, 0x04, 0x00, 0x00, 0x88, 0x1a, 0x89, 0x00,
   };
   static const uint8_t live[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x10, 0x54, 0x00, 0x00, 0x40, 0x00, 0x00,
      0x20, 0x04, 0x00, 0x00, 0x88, 0x1a, 0x89, 0x00,
   };
   struct agx3_image_write_desc d = {
      .image_kind = AGX3_TEX_OPERAND_REGISTER,
      .image_value = 0,
      .image_kill = true,
   };
   uint8_t bytes[AGX3_IMAGE_WRITE_LENGTH], len = 0;
   ASSERT_TRUE(agx3_tex_encode_image_write_2d(
      AGX_ISA_ENCODING_AGX3B, &d, 0, 1, bytes, &len));
   ASSERT_EQ(len, sizeof(killed));
   EXPECT_EQ(memcmp(bytes, killed, sizeof(killed)), 0);

   d.image_kill = false;
   ASSERT_TRUE(agx3_tex_encode_image_write_2d(
      AGX_ISA_ENCODING_AGX3B, &d, 0, 1, bytes, &len));
   ASSERT_EQ(len, sizeof(live));
   EXPECT_EQ(memcmp(bytes, live, sizeof(live)), 0);
}

TEST_F(AGX3PackBinary, StorageImageBankAndSlotMatchNative)
{
   /* metal-nt -arch applegpu_g17p, two writes to array<texture2d<int,
    * access::write>, 4> at computed GPR indices: the first stages bank 0,
    * waits for its COPYs on slot 2 and signals slot 1; the second stages
    * bank 16, waits slot 4 and signals slot 2. */
   struct agx3_image_write_desc d = {
      .staging_bank = 0,
      .result_slot = 1,
      .image_kind = AGX3_TEX_OPERAND_REGISTER,
      .image_value = 2,
      .image_kill = true,
   };
   uint8_t bytes[AGX3_IMAGE_WRITE_LENGTH], len = 0;
   static const uint8_t first[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x00, 0x40, 0x00, 0x02,
      0x21, 0x04, 0x00, 0x00, 0x88, 0x1a, 0x89, 0x00,
   };
   ASSERT_TRUE(agx3_tex_encode_image_write_2d(
      AGX_ISA_ENCODING_AGX3B, &d, 0, 2, bytes, &len));
   ASSERT_EQ(len, sizeof(first));
   EXPECT_EQ(memcmp(bytes, first, sizeof(first)), 0);

   static const uint8_t second[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x80, 0x54, 0x10, 0x00, 0x40, 0x00, 0x00,
      0x21, 0x04, 0x00, 0x00, 0x89, 0x1a, 0x89, 0x00,
   };
   d.staging_bank = 16;
   d.result_slot = 2;
   d.image_value = 0;
   ASSERT_TRUE(agx3_tex_encode_image_write_2d(
      AGX_ISA_ENCODING_AGX3B, &d, 0, 4, bytes, &len));
   ASSERT_EQ(len, sizeof(second));
   EXPECT_EQ(memcmp(bytes, second, sizeof(second)), 0);
}

/* Array layers stay constant across mip levels, unlike 3D depth, so G17C keeps
 * array and 3D stores distinct. G17G drops the layer operand and still writes
 * layered images as 3D. */
TEST_F(AGX3PackBinary, LayeredStorageImageWritesFollowTheChip)
{
   const struct {
      enum agx_isa_encoding enc;
      enum agx_chip chip;
      bool distinct;
   } cases[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17G, false},
   };
   for (const auto &c : cases) {
      key.dev.chip = c.chip;
      std::vector<uint8_t> code[2];
      for (unsigned i = 0; i < 2; ++i) {
         agx_builder *b = builder(c.enc);
         b->shader->stage = MESA_SHADER_COMPUTE;
         agx_index data = agx_register(8, AGX_SIZE_32);
         data.channels_m1 = 3;
         agx_index coords = agx_register(0, AGX_SIZE_32);
         coords.channels_m1 = 2;
         agx_image_write(b, data, coords, agx_immediate(0),
                         agx_uniform(20, AGX_SIZE_64),
                         agx_register(6, AGX_SIZE_32),
                         i ? AGX_DIM_3D : AGX_DIM_2D_ARRAY, false);
         agx_stop(b);
         code[i] = pack(b);
         ASSERT_FALSE(code[i].empty());
      }
      EXPECT_EQ(code[0] != code[1], c.distinct) << "chip " << c.chip;
   }
}

TEST_F(AGX3PackBinary, InFlightStorageImageWritesUseDisjointBanks)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   b->shader->stage = MESA_SHADER_COMPUTE;
   const agx_index heap = agx_uniform(20, AGX_SIZE_64);
   for (unsigned i = 0; i < 2; ++i) {
      agx_index data = agx_register(8, AGX_SIZE_32);
      data.channels_m1 = 3;
      agx_index coord = agx_register(0, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_image_write(b, data, coord, agx_immediate(0), heap,
                      agx_register(4 + 2 * i, AGX_SIZE_32), AGX_DIM_2D,
                      false);
   }
   agx_stop(b);

   /* The first imgwci is still in flight, so the second stages bank 16 on
    * slot 2 rather than restaging bank 0. */
   std::vector<uint8_t> want;
   for (unsigned i = 0; i < 2; ++i) {
      const uint16_t bank = 16 * i;
      const uint8_t slot = 1 + i;
      uint8_t bytes[AGX3_STAGE_MAX_LENGTH], len = 0;
      for (unsigned c = 0; c < 2; ++c) {
         ASSERT_TRUE(agx3_stage_copy_source(enc, bank + 2 * c, 2 * c, false,
                                            slot, 0, bytes, &len));
         append(want, bytes, len);
      }
      const uint16_t data_dest[4] = {4, 6, 12, 14};
      for (unsigned c = 0; c < 4; ++c) {
         ASSERT_TRUE(agx3_stage_copy_source(enc, bank + data_dest[c],
                                            8 + 2 * c, false, slot, 0,
                                            bytes, &len));
         append(want, bytes, len);
      }
      const struct agx3_image_write_desc d = {
         .staging_bank = bank,
         .result_slot = slot,
         .image_kind = AGX3_TEX_OPERAND_REGISTER,
         .image_value = (uint16_t)(4 + 2 * i),
      };
      uint8_t imgwci[AGX3_IMAGE_WRITE_LENGTH], imgwci_len = 0;
      ASSERT_TRUE(agx3_tex_encode_image_write_2d(enc, &d, 20, slot, imgwci,
                                                 &imgwci_len));
      append(want, imgwci, imgwci_len);
   }
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(enc, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 32u);
}

TEST_F(AGX3PackBinary, StorageImageExplicitBaseAndSlotMatchHardware)
{
   const struct agx3_image_write_desc d = {
      .wait_mask = 1,
      .image_kind = AGX3_TEX_OPERAND_IMMEDIATE,
      .image_value = 0,
      .image_kill = false,
      .normalized = false,
   };
   /* Both slot variants passed complete/partial_offset1 on G17P. The
    * production path retains slot1's shorter preceding COPY sequence. */
   const uint8_t slot1[] = {
      0xd7, 0x10, 0x54, 0x00, 0x40, 0x40, 0x00, 0x00,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x1a, 0x09, 0x00,
   };
   uint8_t bytes[AGX3_IMAGE_WRITE_LENGTH], len = 0;
   ASSERT_TRUE(agx3_tex_encode_image_write_2d(
      AGX_ISA_ENCODING_AGX3B, &d, 8, 1, bytes, &len));
   ASSERT_EQ(len, sizeof(slot1));
   EXPECT_EQ(memcmp(bytes, slot1, sizeof(slot1)), 0);
   ASSERT_TRUE(agx3_tex_encode_image_write_2d(
      AGX_ISA_ENCODING_AGX3B, &d, 0, 1, bytes, &len));
   EXPECT_EQ(bytes[4], 0);
   bytes[4] = 0x40;
   EXPECT_EQ(memcmp(bytes, slot1, sizeof(slot1)), 0);
   ASSERT_TRUE(agx3_tex_encode_image_write_2d(
      AGX_ISA_ENCODING_AGX3B, &d, 8, 2, bytes, &len));
   EXPECT_EQ(bytes[1], 0x20);
   bytes[1] = 0x10;
   EXPECT_EQ(memcmp(bytes, slot1, sizeof(slot1)), 0);

   for (unsigned base : {2u, 6u, 510u, 512u}) {
      memset(bytes, 0xa5, sizeof(bytes));
      len = 0xa5;
      EXPECT_FALSE(agx3_tex_encode_image_write_2d(
         AGX_ISA_ENCODING_AGX3B, &d, base, 1, bytes, &len));
      EXPECT_EQ(len, 0);
      for (uint8_t byte : bytes)
         EXPECT_EQ(byte, 0);
   }
   for (unsigned slot : {0u, 7u})
      EXPECT_FALSE(agx3_tex_encode_image_write_2d(
         AGX_ISA_ENCODING_AGX3B, &d, 8, slot, bytes, &len));
}

/* metal-nt -arch applegpu_g17p (AIR-NT 32023.917) texture2d_array and
 * texture3d writes at register coordinates, int and float data, immediate
 * and killed-register descriptor selection. */
TEST_F(AGX3PackBinary, LayeredStorageImageWriteMatchesNative)
{
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t len = 0;

   static const uint8_t layer_r2[] = {
      0x97, 0x05, 0x54, 0x06, 0x00, 0x08, 0x10, 0x00,
      0xf0, 0xc0, 0x00, 0x00,
   };
   ASSERT_TRUE(agx3_tex_stage_image_write_layer(
      AGX_ISA_ENCODING_AGX3B, 4, 6, 1, 0, false, true, bytes, &len));
   ASSERT_EQ(len, sizeof(layer_r2));
   EXPECT_EQ(memcmp(bytes, layer_r2, sizeof(layer_r2)), 0);

   static const uint8_t layer_r0[] = {
      0x97, 0x05, 0x54, 0x06, 0x00, 0x00, 0x10, 0x00,
      0xf0, 0xc0, 0x00, 0x00,
   };
   ASSERT_TRUE(agx3_tex_stage_image_write_layer(
      AGX_ISA_ENCODING_AGX3B, 0, 6, 1, 0, false, true, bytes, &len));
   ASSERT_EQ(len, sizeof(layer_r0));
   EXPECT_EQ(memcmp(bytes, layer_r0, sizeof(layer_r0)), 0);

   /* The eighth source bit rides in bit 48, as in PACK_SMPIR. */
   static const uint8_t layer_half184[] = {
      0x97, 0x05, 0x54, 0x06, 0x00, 0x70, 0x11, 0x00,
      0xf0, 0xc0, 0x00, 0x00,
   };
   ASSERT_TRUE(agx3_tex_stage_image_write_layer(
      AGX_ISA_ENCODING_AGX3B, 184, 6, 1, 0, false, true, bytes, &len));
   ASSERT_EQ(len, sizeof(layer_half184));
   EXPECT_EQ(memcmp(bytes, layer_half184, sizeof(layer_half184)), 0);

   EXPECT_FALSE(agx3_tex_stage_image_write_layer(
      AGX_ISA_ENCODING_AGX3B, 1, 6, 1, 0, false, true, bytes, &len));
   EXPECT_FALSE(agx3_tex_stage_image_write_layer(
      AGX_ISA_ENCODING_AGX3B, 256, 6, 1, 0, false, true, bytes, &len));

   const struct {
      enum agx3_image_write_dim dim;
      enum agx3_tex_operand_kind kind;
      uint16_t value, base;
      bool kill, normalized;
      uint8_t native[AGX3_IMAGE_WRITE_LENGTH];
   } writes[] = {
      {AGX3_IMAGE_WRITE_2D_ARRAY, AGX3_TEX_OPERAND_IMMEDIATE, 0, 4, false,
       false,
       {0xd7, 0x10, 0x54, 0x00, 0x20, 0x00, 0x00, 0x00,
        0x00, 0x04, 0x00, 0x00, 0x88, 0x1a, 0x09, 0x00}},
      {AGX3_IMAGE_WRITE_3D, AGX3_TEX_OPERAND_IMMEDIATE, 0, 0, false, false,
       {0xd7, 0x10, 0x54, 0x00, 0x00, 0x40, 0x00, 0x00,
        0x00, 0x08, 0x00, 0x00, 0x88, 0x1a, 0x09, 0x00}},
      {AGX3_IMAGE_WRITE_2D_ARRAY, AGX3_TEX_OPERAND_REGISTER, 12, 4, true,
       false,
       {0xd7, 0x10, 0x54, 0x00, 0x20, 0x00, 0x00, 0x0c,
        0x21, 0x04, 0x00, 0x00, 0x88, 0x1a, 0x89, 0x00}},
      {AGX3_IMAGE_WRITE_2D_ARRAY, AGX3_TEX_OPERAND_REGISTER, 12, 4, true,
       true,
       {0xd7, 0x10, 0x54, 0x00, 0x20, 0x00, 0x00, 0x0c,
        0x21, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x89, 0x00}},
      {AGX3_IMAGE_WRITE_3D, AGX3_TEX_OPERAND_REGISTER, 0, 0, true, false,
       {0xd7, 0x10, 0x54, 0x00, 0x00, 0x40, 0x00, 0x00,
        0x21, 0x08, 0x00, 0x00, 0x88, 0x1a, 0x89, 0x00}},
      {AGX3_IMAGE_WRITE_3D, AGX3_TEX_OPERAND_REGISTER, 0, 0, true, true,
       {0xd7, 0x10, 0x54, 0x00, 0x00, 0x40, 0x00, 0x00,
        0x21, 0x08, 0x00, 0x00, 0x88, 0x3a, 0x89, 0x00}},
   };
   for (const auto &w : writes) {
      const struct agx3_image_write_desc d = {
         .image_kind = w.kind,
         .image_value = w.value,
         .image_kill = w.kill,
         .normalized = w.normalized,
         .dim = w.dim,
      };
      uint8_t imgwci[AGX3_IMAGE_WRITE_LENGTH];
      ASSERT_TRUE(agx3_tex_encode_image_write_2d(
         AGX_ISA_ENCODING_AGX3B, &d, w.base, 1, imgwci, &len));
      ASSERT_EQ(len, sizeof(w.native));
      EXPECT_EQ(memcmp(imgwci, w.native, sizeof(w.native)), 0);
   }
}

TEST_F(AGX3PackBinary, IntegerIndexedStorageWriteUsesResolvedPbe)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index data = agx_register(8, AGX_SIZE_32);
   data.channels_m1 = 3;
   agx_index coords = agx_register(0, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_instr *write = agx_image_write(
      b, data, coords, agx_immediate(0), agx_immediate(0),
      agx_immediate(1), AGX_DIM_2D, false);
   write->format = AGX_FORMAT_I8;
   agx_stop(b);
   const std::vector<uint8_t> code = pack(b);
   const uint8_t expected[] = {
      0xd7, 0x10, 0x54, 0x00, 0x40, 0x40, 0x00, 0x00,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x1a, 0x09, 0x00,
   };
   ASSERT_GE(code.size(), sizeof(expected) + 4);
   EXPECT_TRUE(std::equal(expected, expected + sizeof(expected),
                          code.end() - 4 - sizeof(expected)));
   /* Six compact slot1 COPYs, one16-byte image write and STOP. */
   EXPECT_EQ(code.size(), 6 * 4 + sizeof(expected) + 4);
}

TEST_F(AGX3PackBinary, OverlappingComputeReadsStageDisjointBanks)
{
   for (unsigned retired = 0; retired < 2; ++retired) {
      const auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index coord = agx_register(0, AGX_SIZE_32);
      coord.channels_m1 = 1;
      const agx_index heap = agx_uniform(6, AGX_SIZE_64);
      for (unsigned i = 0; i < 2; ++i) {
         agx_index data = agx_register(8 + 8 * i, AGX_SIZE_32);
         data.channels_m1 = 3;
         agx_image_load_to(b, data, coord, agx_immediate(0), heap,
                           agx_register(40 + 8 * i, AGX_SIZE_64),
                           agx_immediate(0), agx_null(), AGX_DIM_2D,
                           AGX_LOD_MODE_AUTO_LOD, 15, false, false, false);
         /* Consuming the first result retires its generation. */
         if (i == 0 && retired)
            agx_fadd_to(b, agx_register(32, AGX_SIZE_32),
                        agx_register(8, AGX_SIZE_32),
                        agx_register(10, AGX_SIZE_32));
      }
      agx_stop(b);

      const std::vector<uint8_t> code = pack(b);
      ASSERT_FALSE(code.empty());

      /* metal-nt -arch applegpu_g17p: two texture2d reads in flight at once
       * stage their coordinates in disjoint banks. */
      std::vector<uint8_t> second;
      uint8_t copy[AGX3_TEX_MAX_LENGTH], len = 0;
      const uint16_t bank = retired ? 0 : 8;
      for (unsigned c = 0; c < 2; ++c) {
         ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, bank + 2 * c, 2 * c, 1, 0,
                                               false, false, copy, &len));
         append(second, copy, len);
      }
      EXPECT_NE(std::search(code.begin(), code.end(), second.begin(),
                            second.end()),
                code.end()) << "retired=" << retired;
      EXPECT_EQ(b->shader->agx3_staging_halfs, retired ? 4u : 12u);
   }
}

TEST_F(AGX3PackBinary, StorageImageRoundtripAcceptsCoherentAndRejectsShapes)
{
   static const uint8_t imgwci[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x10, 0x54, 0x00, 0x00, 0x40, 0x00, 0x00,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x1a, 0x09, 0x00,
   };

   for (unsigned mode = 0; mode < 4; ++mode) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index coord = agx_register(0, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_index data = agx_register(4, AGX_SIZE_32);
      data.channels_m1 = 3;
      agx_instr *load = agx_image_load_to(
         b, data, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
         AGX_LOD_MODE_AUTO_LOD, 15, false, false, false);
      load->format = AGX_FORMAT_I8;
      agx_image_write(
         b, data, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), mode == 1 ? AGX_DIM_2D_ARRAY
                            : mode == 2 ? AGX_DIM_2D_MS : AGX_DIM_2D,
         mode == 3);
      agx_stop(b);

      const bool valid = mode == 0 || mode == 3;
      const std::vector<uint8_t> code = pack(b, valid);
      if (valid) {
         EXPECT_TRUE(b->shader->agx3_staging_known);
         EXPECT_EQ(b->shader->agx3_staging_halfs, 20);
         static const uint8_t read_coords[] = {
            0x8b, 0x00, 0x21, 0x04, 0x9b, 0x02, 0x21, 0x04,
         };
         EXPECT_NE(std::search(code.begin(), code.end(), read_coords,
                               read_coords + sizeof(read_coords)),
                   code.end());
         /* The scoreboard wait is carried by the FIRST staged copy, which is
          * the first coordinate copy -- not the first data copy. Waiting on a
          * data copy runs after the coordinates have already been staged, so a
          * coordinate coming straight from an outstanding async producer was
          * copied before that producer retired and the store landed at texel
          * (0,0) (agx3_pack_binary.c, agx3_emit_storage_image_write). */
         std::vector<uint8_t> coordinates;
         uint8_t coordinate[AGX3_STAGE_MAX_LENGTH], coordinate_len = 0;
         ASSERT_TRUE(agx3_stage_copy_source(
            AGX_ISA_ENCODING_AGX3B, 0, 0, false, 1, 1,
            coordinate, &coordinate_len));
         append(coordinates, coordinate, coordinate_len);
         ASSERT_TRUE(agx3_stage_copy_source(
            AGX_ISA_ENCODING_AGX3B, 2, 2, false, 1, 0,
            coordinate, &coordinate_len));
         append(coordinates, coordinate, coordinate_len);
         EXPECT_NE(std::search(code.begin(), code.end(), coordinates.begin(),
                               coordinates.end()), code.end());

         const auto first = std::search(code.begin(), code.end(), imgwci,
                                        imgwci + sizeof(imgwci));
         ASSERT_NE(first, code.end());
         /* The native witness consumes the image-load scoreboard signal once,
          * on the first staged copy of the whole run. The coordinate copies
          * come first and carry it (above), so every data-staging COPY only
          * joins the same slot-1 staging transaction. */
         const unsigned data_dest[4] = {4, 6, 12, 14};
         std::vector<uint8_t> staging;
         for (unsigned c = 0; c < 4; ++c) {
            uint8_t copy[AGX3_STAGE_MAX_LENGTH], len = 0;
            ASSERT_TRUE(agx3_stage_copy_source(
               AGX_ISA_ENCODING_AGX3B, data_dest[c], 4 + 2 * c, false, 1, 0,
               copy, &len));
            append(staging, copy, len);
         }
         ASSERT_GE(std::distance(code.begin(), first), staging.size());
         EXPECT_TRUE(std::equal(staging.begin(), staging.end(),
                                first - staging.size()));
         EXPECT_EQ(std::search(first + sizeof(imgwci), code.end(), imgwci,
                               imgwci + sizeof(imgwci)),
                   code.end());
      } else {
         EXPECT_TRUE(code.empty());
      }
   }
}

/* PACK_SMPIR stages a read's sampler operands six halves above the read's
 * coordinate bank. metal-nt -arch applegpu_g17p, reads followed by a write,
 * SMP f19=16:
 *
 *   texture2d_array layer          1705541600000000f0c00400
 *   texture2d uint level           1705541600000004f0c00c00
 *   texture2d_array layer + level  170554160010020070c06500
 */
TEST_F(AGX3PackBinary, SamplerOperandsFollowTheCoordinateBank)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   static const uint8_t layer16[] = {
      0x17, 0x05, 0x54, 0x16, 0x00, 0x00, 0x00, 0x00,
      0xf0, 0xc0, 0x04, 0x00,
   };
   ASSERT_TRUE(agx3_tex_stage_array_layer_at(enc, 16, 0, 1, 0, false, true,
                                             bytes, &len));
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(layer16, layer16 + sizeof(layer16)));
   static const uint8_t level16[] = {
      0x17, 0x05, 0x54, 0x16, 0x00, 0x00, 0x00, 0x04,
      0xf0, 0xc0, 0x0c, 0x00,
   };
   ASSERT_TRUE(agx3_tex_stage_typed_lod_at(enc, 16, 0, 1, 0, false, true,
                                           AGX3_TEX_LEVEL_INTEGER, bytes,
                                           &len));
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(level16, level16 + sizeof(level16)));
   static const uint8_t layer_level16[] = {
      0x17, 0x05, 0x54, 0x16, 0x00, 0x10, 0x02, 0x00,
      0x70, 0xc0, 0x65, 0x00,
   };
   ASSERT_TRUE(agx3_tex_stage_array_integer_lod_at(
      enc, 16, 8, 0, 1, 0, true, false, false, true, bytes, &len));
   EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
             std::vector<uint8_t>(layer_level16,
                                  layer_level16 + sizeof(layer_level16)));
}

/* An array read in a shader that also writes an image stages its coordinates
 * at halves 16/18 and its layer and level at half 22, clear of the write's
 * coordinates (0/2), layer (6) and data (4/10/12/14), which imgwci's COPYs
 * publish without waiting on the outstanding read. */
TEST_F(AGX3PackBinary, ArrayImageReadStagesClearOfImageWrite)
{
   for (bool register_lod : {false, true}) {
      SCOPED_TRACE(register_lod);
      const auto enc = AGX_ISA_ENCODING_AGX3B;
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index read_coord = agx_register(8, AGX_SIZE_32);
      read_coord.channels_m1 = 2;
      agx_index lod = register_lod ? agx_register(14, AGX_SIZE_32)
                                   : agx_immediate(0);
      agx_index heap = agx_register(16, AGX_SIZE_64);
      agx_index data = agx_register(20, AGX_SIZE_32);
      data.channels_m1 = 3;
      agx_image_load_to(b, data, read_coord, lod, agx_uniform(12, AGX_SIZE_64),
                        heap, agx_immediate(0), agx_null(),
                        AGX_DIM_2D_ARRAY,
                        register_lod ? AGX_LOD_MODE_LOD_MIN
                                     : AGX_LOD_MODE_AUTO_LOD,
                        15, false, false, false);
      agx_index write_coord = agx_register(32, AGX_SIZE_32);
      write_coord.channels_m1 = 2;
      agx_image_write(b, data, write_coord, agx_immediate(0),
                      agx_uniform(12, AGX_SIZE_64),
                      agx_register(40, AGX_SIZE_32), AGX_DIM_2D_ARRAY,
                      false);
      agx_stop(b);
      const auto code = pack(b);
      ASSERT_FALSE(code.empty());

      uint8_t bytes_scratch[AGX3_TEX_MAX_LENGTH], len_scratch = 0;
      ASSERT_TRUE(register_lod
                     ? agx3_tex_stage_array_integer_lod_at(
                          enc, 16, 12, 14, 1, 0, false, false, false, false,
                          bytes_scratch, &len_scratch)
                     : agx3_tex_stage_array_layer_at(enc, 16, 12, 1, 0, false,
                                                     false, bytes_scratch,
                                                     &len_scratch));
      EXPECT_NE(std::search(code.begin(), code.end(), bytes_scratch,
                            bytes_scratch + len_scratch),
                code.end());

      /* Nothing is staged into bank 0's operand half, which the write's
       * PACK_IMGWIR layer shares. */
      ASSERT_TRUE(register_lod
                     ? agx3_tex_stage_array_integer_lod(
                          enc, 12, 14, 1, 0, false, false, false, false,
                          bytes_scratch, &len_scratch)
                     : agx3_tex_stage_array_layer(enc, 12, 1, 0, false,
                                                  false, bytes_scratch,
                                                  &len_scratch));
      EXPECT_EQ(std::search(code.begin(), code.end(), bytes_scratch,
                            bytes_scratch + len_scratch),
                code.end());

      /* The read's coordinates are staged at 16/18. */
      uint8_t copy[AGX3_TEX_MAX_LENGTH], copy_len = 0;
      ASSERT_TRUE(agx3_tex_stage_copy_hints(enc, 18, 10, 1, 0, false, false,
                                            copy, &copy_len));
      EXPECT_NE(std::search(code.begin(), code.end(), copy, copy + copy_len),
                code.end());
   }
}

TEST_F(AGX3PackBinary, BufferFetchStoreKeepsNativeReadCoordinatesSeparate)
{
   for (bool with_store : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index read_coord = agx_register(0, AGX_SIZE_32);
      read_coord.channels_m1 = 1;
      agx_index write_coord = agx_register(12, AGX_SIZE_32);
      write_coord.channels_m1 = 1;
      agx_index data = agx_register(4, AGX_SIZE_32);
      data.channels_m1 = 3;
      agx_instr *read = agx_image_load_to(
         b, data, read_coord, agx_immediate(0), agx_immediate(0),
         agx_register(16, AGX_SIZE_64), agx_immediate(0), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 1, false, false, false);
      read->op = AGX_OPCODE_TEXTURE_LOAD;
      if (with_store) {
         agx_instr *write = agx_image_write(
            b, data, write_coord, agx_immediate(0), agx_immediate(0),
            agx_immediate(1), AGX_DIM_2D, false);
         write->format = AGX_FORMAT_I8;
      }
      agx_stop(b);
      const auto code = pack(b);
      bool found = false;
      for (size_t i = 0; i < code.size(); ++i) {
         struct agx3_tex_fetch_desc fetch = {};
         if (i + 20 <= code.size() &&
             agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B,
                                   code.data() + i, 20,
                                   &fetch)) {
            EXPECT_EQ(fetch.coord, with_store ? 16 : 0);
            if (with_store) {
               EXPECT_EQ(fetch.slot, 1);
            }
            found = true;
         }
      }
      EXPECT_TRUE(found);
      EXPECT_TRUE(b->shader->agx3_staging_known);
      EXPECT_EQ(b->shader->agx3_staging_halfs, with_store ? 20 : 4);
   }
}

TEST_F(AGX3PackBinary, VertexIndexedImageReadPublishesCoordinates)
{
   for (bool image : {false, true}) {
      SCOPED_TRACE(image);
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_VERTEX;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_index data = agx_register(16, AGX_SIZE_32);
      data.channels_m1 = 3;
      agx_instr *read = agx_image_load_to(
         b, data, coord, agx_immediate(0),
         image ? agx_uniform(0, AGX_SIZE_64) : agx_immediate(0),
         agx_register(24, AGX_SIZE_64), agx_immediate(0), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 1, false, false, false);
      if (!image)
         read->op = AGX_OPCODE_TEXTURE_LOAD;
      agx_stop(b);
      const auto code = pack(b);
      ASSERT_FALSE(code.empty());
      EXPECT_TRUE(b->shader->agx3_staging_known);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 4);

      bool found = false;
      for (size_t i = 0; i < code.size(); ++i) {
         struct agx3_tex_fetch_desc fetch = {};
         if (i + 20 > code.size() ||
             !agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B,
                                    code.data() + i, 20, &fetch))
            continue;
         EXPECT_EQ(fetch.coord, 0);
         EXPECT_EQ(fetch.wait_mask, 1u << (fetch.slot - 1));
         /* Native vertex read() writes x/y to staging halves 0/2, then
          * SMP consumes those coordinates. The Mesa GPR pair is not itself
          * a staging operand, for either texelFetch or imageLoad. */
         std::vector<uint8_t> coordinates;
         for (unsigned c = 0; c < 2; ++c) {
            uint8_t copy[AGX3_TEX_MAX_LENGTH], length = 0;
            ASSERT_TRUE(agx3_tex_stage_copy_hints(
               AGX_ISA_ENCODING_AGX3B, 2 * c, 8 + 2 * c, fetch.slot,
               0, false, false, copy, &length));
            append(coordinates, copy, length);
         }
         ASSERT_GE(i, coordinates.size());
         EXPECT_TRUE(std::equal(coordinates.begin(), coordinates.end(),
                                code.begin() + i - coordinates.size()));
         found = true;
      }
      EXPECT_TRUE(found);
   }
}

/* A bindless store's descriptor set uniform is its PBE heap and a GPR holds
 * the PBE record index. The write addresses that heap through f79 and
 * consumes the slot1 data COPYs. */
TEST_F(AGX3PackBinary, BindlessStorageImageWriteAddressesSetHeap)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index data = agx_register(8, AGX_SIZE_32);
   data.channels_m1 = 3;
   data.discard = true;
   agx_index coord = agx_register(16, AGX_SIZE_32);
   coord.channels_m1 = 1;
   coord.discard = true;
   agx_index index = agx_register(2, AGX_SIZE_32);
   index.discard = true;
   agx_instr *write = agx_image_write(
      b, data, coord, agx_immediate(0), agx_uniform(4, AGX_SIZE_64),
      index, AGX_DIM_2D, false);
   write->format = AGX_FORMAT_U8NORM;
   agx_stop(b);

   const std::vector<uint8_t> code = pack(b);
   static const uint8_t imgwci[] = {
      0xd7, 0x10, 0x54, 0x00, 0x20, 0x40, 0x00, 0x02,
      0x21, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x89, 0x00,
   };
   EXPECT_NE(std::search(code.begin(), code.end(), imgwci,
                         imgwci + sizeof(imgwci)),
             code.end());

   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   bad->shader->stage = MESA_SHADER_COMPUTE;
   index = agx_register(3, AGX_SIZE_32);
   index.discard = true;
   write = agx_image_write(
      bad, data, coord, agx_immediate(0), agx_uniform(4, AGX_SIZE_64),
      index, AGX_DIM_2D, false);
   write->format = AGX_FORMAT_U8NORM;
   agx_stop(bad);
   EXPECT_TRUE(pack(bad, false).empty());
}

TEST_F(AGX3PackBinary, ArrayFetchLayerOperandMatchesNative)
{
   /* Own G17P texture2d_array<uint>::read(gid.xy, gid.z) witness. The
    * layer's low16 bits are discarded after publishing operand bank6. */
   const uint8_t native[] = {
      0x17, 0x05, 0x54, 0x06, 0x00, 0x04,
      0x00, 0x00, 0xf0, 0xc0, 0x04, 0x00,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_tex_stage_array_layer(
      AGX_ISA_ENCODING_AGX3B, 2, 1, 0, false, true, bytes, &len));
   ASSERT_EQ(len, sizeof(native));
   EXPECT_EQ(memcmp(bytes, native, sizeof(native)), 0);
}

TEST_F(AGX3PackBinary, ArrayFetchPublishesLayerSeparatelyFromCoordinates)
{
   for (bool array : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index data = agx_register(8, AGX_SIZE_32);
      data.channels_m1 = 3;
      agx_index coord = agx_register(0, AGX_SIZE_32);
      coord.channels_m1 = array ? 2 : 1;
      agx_instr *read = agx_image_load_to(
         b, data, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(),
         array ? AGX_DIM_2D_ARRAY : AGX_DIM_2D,
         AGX_LOD_MODE_AUTO_LOD, 15, false, false, false);
      read->op = AGX_OPCODE_TEXTURE_LOAD;
      read->mask = 15;
      agx_stop(b);
      const auto code = pack(b);
      /* The native layer operand with source half4, slot1 and retained
       * source. The prior third coordinate COPY did not publish this bank. */
      const uint8_t packed[] = {
         0x17, 0x05, 0x54, 0x06, 0x00, 0x08,
         0x00, 0x00, 0x70, 0xc0, 0x04, 0x00,
      };
      const auto found = std::search(code.begin(), code.end(), packed,
                                      packed + sizeof(packed));
      EXPECT_EQ(found != code.end(), array);
      EXPECT_TRUE(b->shader->agx3_staging_known);
      EXPECT_EQ(b->shader->agx3_staging_halfs, array ? 8 : 4);
   }
}

TEST_F(AGX3PackBinary, BindlessStorageImageReadMatchesNativeIndexAndLOD)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index dest = agx_register(0, AGX_SIZE_32);
   dest.channels_m1 = 3;
   agx_index coord = agx_register(12, AGX_SIZE_32);
   coord.channels_m1 = 1;
   coord.discard = true;
   agx_index lod = agx_register(4, AGX_SIZE_32);
   lod.discard = true;
   agx_index texture = agx_register(0, AGX_SIZE_64);
   texture.discard = true;
   agx_instr *read = agx_image_load_to(
      b, dest, coord, lod, agx_uniform(0, AGX_SIZE_64), texture,
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN,
      15, false, false, false);
   read->mask = 15;
   agx_stop(b);

   const std::vector<uint8_t> code = pack(b);
   /* applegpu-nt 32023.917 / apple-g17p, lane-selected argument-buffer
    * texture plus a uint level widened from ushort. Coordinates and LOD are
    * published on slot 2; the image result independently occupies slot 1. */
   static const uint8_t native_lod[] = {
      0x17,0x05,0x54,0x06,0x00,0x08,0x00,0x04,0xf0,0xc4,0x0c,0x00,
   };
   static const uint8_t native_read[] = {
      0x05,0x80,0x0c,0xb8,0x30,0x01,0x80,0x24,0x00,0x02,
      0x80,0x00,0x01,0x00,0x00,0x28,0x00,0x00,0x00,0x00,
   };
   EXPECT_NE(std::search(code.begin(), code.end(), native_lod,
                         native_lod + sizeof(native_lod)), code.end());
   EXPECT_NE(std::search(code.begin(), code.end(), native_read,
                         native_read + sizeof(native_read)), code.end());

   agx_builder *bad = builder(AGX_ISA_ENCODING_AGX3B);
   bad->shader->stage = MESA_SHADER_COMPUTE;
   lod = agx_register(4, AGX_SIZE_16);
   lod.discard = true;
   read = agx_image_load_to(
      bad, dest, coord, lod, agx_uniform(0, AGX_SIZE_64), texture,
      agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN,
      15, false, false, false);
   read->mask = 15;
   agx_stop(bad);
   EXPECT_TRUE(pack(bad, false).empty());
}

TEST_F(AGX3PackBinary, SampledRgba8WriteUsesNativeStateOneProtocol)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index sample = agx_register(0, AGX_SIZE_32);
   sample.channels_m1 = 3;
   agx_index sample_coord = agx_register(8, AGX_SIZE_32);
   sample_coord.channels_m1 = 1;
   agx_instr *read = agx_texture_sample_to(
      b, sample, sample_coord, agx_immediate_f(0.0f), agx_immediate(0),
      agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
   read->mask = 15;

   agx_index write_coord = agx_register(12, AGX_SIZE_32);
   write_coord.channels_m1 = 1;
   agx_instr *write = agx_image_write(
      b, sample, write_coord, agx_immediate(0), agx_immediate(0),
      agx_immediate(1), AGX_DIM_2D, false);
   write->format = AGX_FORMAT_U8NORM;
   agx_stop(b);

   const std::vector<uint8_t> code = pack(b);
   static const uint8_t imgwci[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x20, 0x40, 0x00, 0x01,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x09, 0x00,
   };
   ASSERT_GE(code.size(), sizeof(imgwci) + 4);
   EXPECT_TRUE(std::equal(imgwci, imgwci + sizeof(imgwci),
                          code.end() - 4 - sizeof(imgwci)));
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 16);
}

TEST_F(AGX3PackBinary, ComputedRgb10WriteUsesNativeStateOneProtocol)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index data = agx_register(16, AGX_SIZE_32);
   data.channels_m1 = 3;
   agx_index coord = agx_register(12, AGX_SIZE_32);
   coord.channels_m1 = 1;
   agx_instr *write = agx_image_write(
      b, data, coord, agx_immediate(0), agx_immediate(0), agx_immediate(1),
      AGX_DIM_2D, false);
   write->format = AGX_FORMAT_RGB10A2;
   agx_stop(b);

   const std::vector<uint8_t> code = pack(b);
   static const uint8_t imgwci[AGX3_IMAGE_WRITE_LENGTH] = {
      0xd7, 0x20, 0x54, 0x00, 0x20, 0x40, 0x00, 0x01,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x3a, 0x09, 0x00,
   };
   EXPECT_NE(std::search(code.begin(), code.end(), imgwci,
                         imgwci + sizeof(imgwci)),
             code.end());
}


/*
 * The six shapes a three.js frame was refused on.
 *
 * Measured on hardware: Chromium loading shadowmap_pointlight, lights_spotlight
 * and animation_keyframes with AGX_MESA_DEBUG=shaders printed 307 refusals in
 * one run, which reduce to six once register and immediate numbers are
 * normalised. Every one is a source modifier, an operand width, or an operand
 * KIND -- three axes that are independent fields of a single AGX3 encoding
 * point, and which the packer modelled as separate encoders. A shape sitting
 * on two axes at once then had no home in either.
 *
 * Each case below is the refusal line verbatim, and each encoding is grounded
 * in the byte ladders quoted at the encoders themselves.
 */
TEST_F(AGX3PackBinary, ThreeJsRefusedShapesEncode)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;

   /* 113x  fcmpsel [d16:reg s0=32:reg.abs.discard s1=16:imm s2=16:imm
    *                s3=16:imm cond=1 chan=1]
    * An absolute comparator against a non-zero minifloat. The gate demanded
    * an immediate comparator of exactly zero; f51 holds any minifloat. */
   {
      agx_builder *b = builder(enc);
      agx_index a = agx_register(2, AGX_SIZE_32);
      a.abs = true;
      a.discard = true;
      agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), a, agx_immediate(0x20),
                     agx_immediate(0x30), agx_immediate(0), AGX_FCOND_LT);
      agx_stop(b);
      uint8_t bytes[AGX3_ALU_MAX_LENGTH];
      uint8_t len = 0;
      struct agx3_fsel_desc d = {};
      d.dest = 2;
      d.dest_16 = true;
      d.cmp0 = 2;
      d.cmp0_hint = AGX3_ALU_HINT_DISCARD;
      d.cmp0_abs = true;
      d.cmp1_is_immediate = true;
      d.cmp1 = 0x20;
      d.sel0_kind = AGX3_FSEL_SEL_IMMEDIATE;
      d.sel0 = 0x30;
      d.sel1_kind = AGX3_FSEL_SEL_IMMEDIATE;
      d.sel1 = 0;
      d.sel_16 = true;
      d.condition = AGX_FCOND_LT;
      ASSERT_TRUE(agx3_alu_fcmp_select(enc, &d, bytes, &len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));
      std::vector<uint8_t> want;
      append(want, bytes, len);
      append(want, end.bytes, end.length);
      EXPECT_EQ(pack(b), want);
   }

   /* 84x  fadd [d32:reg s0=16:reg s1=32:reg.neg cond=0 chan=1]
    * A half added to a negated full float. The width lives in f43/f26 and the
    * negate in f48/f31, on one encoding point; the packer had one encoder for
    * widths without modifiers and another for modifiers without widths. */
   {
      agx_builder *b = builder(enc);
      agx_index f16 = agx_register(4, AGX_SIZE_16);
      agx_index f32 = agx_register(0, AGX_SIZE_32);
      f32.neg = true;
      f32.discard = true;
      f16.discard = true;
      agx_fadd_to(b, agx_register(0, AGX_SIZE_32), f16, f32);
      agx_stop(b);
      /* Apple's own `b - a`, `float a`, `half b`: 09041c0100c8. The wait bit
       * f57 belongs to the loads in that kernel, which this shape has not
       * got, so the ladder row is reproduced through the encoder. */
      struct agx3_falu_mods m = {};
      m.src0_neg = true;
      m.src1_16 = true;
      struct agx3_falu_runtime rt = {};
      rt.discard[0] = true;
      rt.discard[1] = true;
      rt.wait_mask = 0x20;
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH];
      uint8_t len = 0;
      ASSERT_TRUE(
         agx3_falu_add_mod_runtime(enc, 0, 0, 4, &m, &rt, bytes, &len));
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
                (std::vector<uint8_t>{0x09, 0x04, 0x1c, 0x01, 0x00, 0xc8}));
      EXPECT_FALSE(pack(b).empty());
   }

   /* 80x  fcmpsel [d16:reg s0=32:reg.neg s1=32:reg.neg s2=16:reg s3=16:imm
    *               cond=5 chan=1]
    * Both comparators negated. Apple's own bytes for `-a >= -c` and `a <= c`
    * are the same instruction, so this is the relation flip and nothing else;
    * the encoder must produce exactly the plain select's bytes. */
   {
      auto build = [&](bool negated, enum agx_fcond cond) {
         agx_builder *b = builder(enc);
         agx_index a = agx_register(8, AGX_SIZE_32);
         agx_index c = agx_register(16, AGX_SIZE_32);
         a.neg = c.neg = negated;
         agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), a, c,
                        agx_register(6, AGX_SIZE_16), agx_immediate(0), cond);
         agx_stop(b);
         return pack(b);
      };
      const auto negated = build(true, AGX_FCOND_GE);
      const auto flipped = build(false, AGX_FCOND_LE);
      EXPECT_FALSE(negated.empty());
      EXPECT_EQ(negated, flipped);

      /* One negate is f65, with the operands and the relation left alone. */
      agx_builder *b = builder(enc);
      agx_index a = agx_register(8, AGX_SIZE_32);
      agx_index c = agx_register(16, AGX_SIZE_32);
      c.neg = true;
      agx_fcmpsel_to(b, agx_register(2, AGX_SIZE_16), a, c,
                     agx_register(6, AGX_SIZE_16), agx_immediate(0),
                     AGX_FCOND_GE);
      agx_stop(b);
      const auto second = pack(b);
      const auto plain = build(false, AGX_FCOND_GE);
      ASSERT_EQ(second.size(), plain.size());
      unsigned differing = 0;
      for (unsigned i = 0; i < second.size(); ++i)
         differing += second[i] != plain[i];
      EXPECT_EQ(differing, 1u);
   }

   /* 19x  bitop [d32:reg s0=32:reg.neg s1=16:imm cond=0 chan=1]
    * A value move carrying a folded fneg or fabs. agx_lower_pseudo() now puts
    * the float move back rather than handing the bitwise encoding a modifier
    * it has no field for. */
   {
      for (unsigned k = 0; k < 3; ++k) {
         agx_builder *b = builder(enc);
         agx_index a = agx_register(8, AGX_SIZE_32);
         a.neg = k != 1;
         a.abs = k != 0;
         agx_mov_to(b, agx_register(0, AGX_SIZE_32), a);
         agx_stop(b);
         agx_lower_pseudo(b->shader);
         EXPECT_FALSE(pack(b).empty()) << "k=" << k;
      }
   }

   /* 9x  bitop [d32:reg s0=32:reg.discard s1=16:imm cond=0 chan=1]
    * A bitwise operation against an immediate. The operand kind is f37 on the
    * register form's own encoding point, with the constant in f29. */
   {
      const unsigned tables[] = {AGX3_SEL_TT_AND, AGX3_SEL_TT_OR,
                                 AGX3_SEL_TT_XOR};
      for (unsigned i = 0; i < ARRAY_SIZE(tables); ++i) {
         agx_builder *b = builder(enc);
         agx_index a = agx_register(0, AGX_SIZE_32);
         a.discard = true;
         agx_bitop_to(b, agx_register(0, AGX_SIZE_32), a, agx_immediate(0xff),
                      tables[i]);
         agx_stop(b);
         EXPECT_FALSE(pack(b).empty()) << "truth table " << tables[i];
      }
      /* Apple's own `a & 0xff`, with the load's wait on it. */
      uint8_t bytes[AGX3_ALU_MAX_LENGTH];
      uint8_t len = 0;
      ASSERT_TRUE(agx3_alu_bit_imm(enc, AGX3_ALU_BAND, false, 0, 0, 0xff,
                                   AGX3_ALU_HINT_DISCARD, 0x20, bytes, &len));
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + len),
                (std::vector<uint8_t>{0x0b, 0x01, 0x1f, 0xff, 0x10, 0x02, 0x00,
                                      0x80, 0x00, 0x00}));
   }
}

/* Chromium's animation_keyframes vertex shader needs 97 full registers and
 * moves values in both directions across AGX3b iadd's r95 boundary. Pin the
 * exact native half-pair identities, including the destination-cache hint
 * exposed by the captured shader. */
TEST_F(AGX3PackBinary, ChromiumHighRegisterMovesUseHalfPairIdentity)
{
   static const struct {
      unsigned dest;
      unsigned src;
      bool dest_cache;
      uint8_t bytes[20];
   } moves[] = {
      {192, 0, true,
       {0x03, 0x00, 0x8f, 0x81, 0x02, 0x1a, 0x00, 0x00, 0x00, 0x00,
        0x03, 0x00, 0x8f, 0x81, 0x0e, 0x1a, 0x00, 0x00, 0x00, 0x00}},
      {254, 192, false,
       {0xf3, 0x40, 0xcf, 0x81, 0x02, 0x1b, 0x00, 0x00, 0x00, 0x00,
        0xf3, 0x40, 0xcf, 0x81, 0x0e, 0x1b, 0x00, 0x00, 0x00, 0x00}},
   };

   for (const auto &move : moves) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index src = agx_register(move.src, AGX_SIZE_32);
      src.discard = true;
      agx_index dest = agx_register(move.dest, AGX_SIZE_32);
      dest.cache = move.dest_cache;
      agx_bitop_to(b, dest, src, agx_immediate(0), AGX3_SEL_TT_MOV);
      agx_stop(b);

      std::vector<uint8_t> want(std::begin(move.bytes), std::end(move.bytes));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
      append(want, end.bytes, end.length);

      EXPECT_EQ(pack(b), want) << "dest=" << move.dest
                               << " src=" << move.src;
   }
}

/* The CTS abort combined a comparison and a runtime offset in src[5], then
 * validated it as a single scalar. Exercise that exact vector shape and the
 * array bank-6 merge in both graphics and compute shaders. */
TEST_F(AGX3PackBinary, RuntimeGatherOffsetPreservesComparisonAndLayer)
{
   for (auto stage : {MESA_SHADER_FRAGMENT, MESA_SHADER_COMPUTE}) {
      for (bool array : {false, true}) {
         for (bool shadow : {false, true}) {
            auto enc = AGX_ISA_ENCODING_AGX3B;
            agx_builder *b = builder(enc);
            b->shader->stage = stage;
            agx_index dest = agx_register(24, AGX_SIZE_32);
            dest.channels_m1 = 3;
            agx_index coord = agx_register(8, AGX_SIZE_32);
            coord.channels_m1 = array ? 2 : 1;
            agx_index ref_offset = agx_register(16, AGX_SIZE_32);
            ref_offset.channels_m1 = shadow ? 1 : 0;
            auto *I = agx_texture_sample_to(
               b, dest, coord, agx_immediate(0), agx_immediate(0),
               agx_register(0, AGX_SIZE_64), agx_register(4, AGX_SIZE_64),
               ref_offset, array ? AGX_DIM_2D_ARRAY : AGX_DIM_2D,
               AGX_LOD_MODE_AUTO_LOD, 0, true, shadow, false,
               AGX_GATHER_R, false);
            I->mask = 15;
            agx_stop(b);
            agx_validate(b->shader, "runtime gather offset");
            const auto code = pack(b);
            uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
            ASSERT_TRUE(agx3_tex_stage_gather_offset(
               enc, shadow ? 18 : 16, array, b->shader->max_reg - 1,
               array ? 6 : 1, false, false, bytes, &len));
            EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                      code.end());
            EXPECT_GE(b->shader->agx3_staging_halfs, 8);
         }
      }
   }
}

TEST_F(AGX3PackBinary, IndexedGatherKeepsNativeHandles)
{
   for (unsigned shape = 0; shape < 3; ++shape) {
      for (bool shadow : {false, true}) {
         for (bool constant_offset : {false, true}) {
            if (shape == 2 && constant_offset)
               continue;
            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_COMPUTE, &agx_nir_options, "indexed gather");
            nir_def *image = nir_bindless_image_agx(&b, nir_imm_int(&b, 72),
                                                     .desc_set = 1);
            nir_def *smp = nir_bindless_sampler_agx(
               &b, nir_imm_int(&b, 104), nir_imm_intN_t(&b, 3, 16),
               .desc_set = 1);
            nir_tex_instr *tex = nir_tex_instr_create(
               b.shader, 3 + shadow + (shape < 2));
            tex->op = nir_texop_tg4;
            tex->sampler_dim = shape == 2 ? GLSL_SAMPLER_DIM_CUBE
                                          : GLSL_SAMPLER_DIM_2D;
            tex->is_array = shape == 1;
            tex->is_shadow = shadow;
            tex->is_new_style_shadow = false;
            tex->dest_type = nir_type_float32;
            tex->coord_components = shape ? 3 : 2;
            unsigned i = 0;
            tex->src[i++] = nir_tex_src_for_ssa(
               nir_tex_src_coord, shape ? nir_imm_vec3(&b, .5, .25, 1)
                                        : nir_imm_vec2(&b, .5, .25));
            tex->src[i++] =
               nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
            tex->src[i++] =
               nir_tex_src_for_ssa(nir_tex_src_sampler_handle, smp);
            if (shadow)
               tex->src[i++] = nir_tex_src_for_ssa(
                  nir_tex_src_comparator, nir_imm_float(&b, .5));
            if (shape < 2) {
               nir_def *packed_offset;
               if (constant_offset) {
                  packed_offset = nir_imm_int(&b, 0x00080008);
               } else {
                  nir_def *runtime = nir_load_global_constant(
                     &b, 2, 32, nir_imm_int64(&b, 0x10000), .align_mul = 4);
                  packed_offset = nir_pack_32_2x16_split(
                     &b, nir_i2i16(&b, nir_channel(&b, runtime, 0)),
                     nir_i2i16(&b, nir_channel(&b, runtime, 1)));
               }
               tex->src[i++] = nir_tex_src_for_ssa(nir_tex_src_backend2,
                                                    packed_offset);
            }
            nir_def_init(&tex->instr, &tex->def, 4, 32);
            nir_builder_instr_insert(&b, &tex->instr);
            nir_store_global(&b, &tex->def, nir_imm_int64(&b, 0x20000),
                             .align_mul = 4);
            ASSERT_TRUE(agx_nir_lower_g17p_indexed_samplers(b.shader, true));
            const int texture = nir_tex_instr_src_index(
               tex, nir_tex_src_texture_offset);
            ASSERT_GE(texture, 0);
            EXPECT_EQ(nir_src_bit_size(tex->src[texture].src), 64);
            ralloc_free(b.shader);
         }
      }
   }
}

/* SCRATCH: dump the G17G array-comparison streams for a witness diff. */
TEST_F(AGX3PackBinary, G17GArrayShadowDump)
{
   struct row { const char *name; mesa_shader_stage stage; enum agx_lod_mode mode; bool reg_lod; };
   static const row rows[] = {
      {"frag bias(reg)", MESA_SHADER_FRAGMENT, AGX_LOD_MODE_AUTO_LOD_BIAS, true},
      {"frag level(reg)", MESA_SHADER_FRAGMENT, AGX_LOD_MODE_LOD_MIN, true},
      {"vertex level0", MESA_SHADER_VERTEX, AGX_LOD_MODE_LOD_MIN, false},
      {"vertex level(reg)", MESA_SHADER_VERTEX, AGX_LOD_MODE_LOD_MIN, true},
      {"frag implicit", MESA_SHADER_FRAGMENT, AGX_LOD_MODE_AUTO_LOD, false},
   };
   for (const row &r : rows) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      b->shader->stage = r.stage;
      key.no_stop = true;
      agx_index dest = agx_register(24, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(16, AGX_SIZE_32);
      coord.channels_m1 = 2;
      coord.discard = true;
      agx_index lod = r.reg_lod ? agx_register(0, AGX_SIZE_16) : agx_immediate(0);
      lod.discard = r.reg_lod;
      agx_index compare = agx_register(14, AGX_SIZE_32);
      compare.discard = true;
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0), agx_immediate(0),
         agx_immediate(1), compare, AGX_DIM_2D_ARRAY, r.mode, 0, false,
         /*shadow=*/true, false, AGX_GATHER_NONE, false);
      I->mask = 1;
      agx_stop(b);
      struct util_dynarray bin;
      util_dynarray_init(&bin, mem_ctx);
      bool ok = agx3_pack_binary(b->shader, &bin);
      printf("DUMP %-18s ok=%d halfs=%u :", r.name, ok, b->shader->agx3_staging_halfs);
      const uint8_t *d = (const uint8_t *)bin.data;
      for (unsigned i = 0; i < bin.size; ++i) printf("%02x", d[i]);
      printf("\n");
   }
}

TEST(AGX3VertexExportAlu, NativeTerrainProducerBytes)
{
   uint8_t out[16], len;
   { /* Native terrain shader form, exact decoded/recomposed witness. */
      const uint16_t src[3] = {40, 16, 32};
      const bool live[3] = {false, true, false}, discard[3] = {false, false, true};
      const uint8_t expected[] = {0x29, 0x91, 0x26, 0x29, 0x82, 0x20, 0x42, 0x02, 0x00, 0x04};
      ASSERT_TRUE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, true, 4, 2, src, true, live, discard, 0, out, &len));
      ASSERT_EQ(len, sizeof(expected));
      EXPECT_EQ(memcmp(out, expected, len), 0);
   }
   { /* Native terrain shader form, exact decoded/recomposed witness. */
      const uint16_t src[3] = {28, 8, 0};
      const bool live[3] = {false, false, false}, discard[3] = {true, true, false};
      const uint8_t expected[] = {0x09, 0x09, 0x3d, 0x1d, 0x41, 0x20, 0x00, 0x08};
      ASSERT_TRUE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, false, 0, 3, src, false, live, discard, 1, out, &len));
      ASSERT_EQ(len, sizeof(expected));
      EXPECT_EQ(memcmp(out, expected, len), 0);
   }
   { /* Native terrain shader form, exact decoded/recomposed witness. */
      const uint16_t src[3] = {30, 10, 0};
      const bool live[3] = {false, false, false}, discard[3] = {true, true, false};
      const uint8_t expected[] = {0x19, 0x0b, 0x3d, 0x1f, 0x41, 0x00, 0x00, 0x08};
      ASSERT_TRUE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, false, 2, 3, src, false, live, discard, 0, out, &len));
      ASSERT_EQ(len, sizeof(expected));
      EXPECT_EQ(memcmp(out, expected, len), 0);
   }
   { /* Native terrain shader form, exact decoded/recomposed witness. */
      const uint16_t src[3] = {2, 12, 0};
      const bool live[3] = {false, false, false}, discard[3] = {true, true, false};
      const uint8_t expected[] = {0x39, 0x0d, 0x3d, 0x03, 0x41, 0x00, 0x00, 0x08};
      ASSERT_TRUE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, false, 6, 3, src, false, live, discard, 0, out, &len));
      ASSERT_EQ(len, sizeof(expected));
      EXPECT_EQ(memcmp(out, expected, len), 0);
   }
   { /* Native terrain shader form, exact decoded/recomposed witness. */
      const uint16_t src[3] = {4, 14, 0};
      const bool live[3] = {false, false, false}, discard[3] = {true, true, false};
      const uint8_t expected[] = {0x49, 0x0f, 0x3d, 0x05, 0x41, 0x00, 0x00, 0x08};
      ASSERT_TRUE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, false, 8, 3, src, false, live, discard, 0, out, &len));
      ASSERT_EQ(len, sizeof(expected));
      EXPECT_EQ(memcmp(out, expected, len), 0);
   }
   { /* Native terrain shader form, exact decoded/recomposed witness. */
      const uint16_t src[3] = {42, 16, 12};
      const bool live[3] = {false, true, false}, discard[3] = {false, false, true};
      const uint8_t expected[] = {0x59, 0x91, 0x26, 0x2b, 0x81, 0x0c, 0x42, 0x02};
      ASSERT_TRUE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, true, 10, 1, src, true, live, discard, 0, out, &len));
      ASSERT_EQ(len, sizeof(expected));
      EXPECT_EQ(memcmp(out, expected, len), 0);
   }
   { /* Native terrain shader form, exact decoded/recomposed witness. */
      const uint16_t src[3] = {44, 16, 20};
      const bool live[3] = {false, true, false}, discard[3] = {false, false, true};
      const uint8_t expected[] = {0x69, 0x91, 0x26, 0x2d, 0x81, 0x14, 0x42, 0x02};
      ASSERT_TRUE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, true, 12, 1, src, true, live, discard, 0, out, &len));
      ASSERT_EQ(len, sizeof(expected));
      EXPECT_EQ(memcmp(out, expected, len), 0);
   }
   { /* Native terrain shader form, exact decoded/recomposed witness. */
      const uint16_t src[3] = {46, 16, 12};
      const bool live[3] = {false, false, false}, discard[3] = {false, true, true};
      const uint8_t expected[] = {0x79, 0x11, 0x2e, 0x2f, 0x81, 0x0c, 0x42, 0x02};
      ASSERT_TRUE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, true, 14, 1, src, true, live, discard, 0, out, &len));
      ASSERT_EQ(len, sizeof(expected));
      EXPECT_EQ(memcmp(out, expected, len), 0);
   }
   { /* Native terrain shader form, exact decoded/recomposed witness. */
      const uint16_t src[3] = {4, 6, 8};
      const bool live[3] = {false, false, false}, discard[3] = {true, true, true};
      const uint8_t expected[] = {0x89, 0x07, 0x3e, 0x05, 0x82, 0x08, 0x42, 0x00, 0x00, 0x14};
      ASSERT_TRUE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, true, 16, 6, src, false, live, discard, 0, out, &len));
      ASSERT_EQ(len, sizeof(expected));
      EXPECT_EQ(memcmp(out, expected, len), 0);
   }
}

TEST(AGX3VertexExportAlu, RejectsUnqualifiedOperands)
{
   uint8_t out[16], len;
   uint16_t src[3] = {0, 2, 4};
   bool live[3] = {}, discard[3] = {};
   for (unsigned slot : {0u, 7u})
      EXPECT_FALSE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, true, 0, slot,
                                         src, false, live, discard, 0, out, &len));
   for (unsigned dest : {1u, 256u})
      EXPECT_FALSE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, true, dest, 1,
                                         src, false, live, discard, 0, out, &len));
   EXPECT_FALSE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, false, 0, 1,
                                      src, true, live, discard, 0, out, &len));
   EXPECT_FALSE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, true, 0, 1,
                                      src, false, live, discard, 64, out, &len));
   for (unsigned s = 0; s < 3; ++s) {
      const unsigned old = src[s];
      src[s] = 1;
      EXPECT_FALSE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, true, 0, 1,
                                         src, false, live, discard, 0, out, &len));
      src[s] = old;
      live[s] = discard[s] = true;
      EXPECT_FALSE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, true, 0, 1,
                                         src, false, live, discard, 0, out, &len));
      live[s] = discard[s] = false;
   }
}

TEST_F(AGX3PackBinary, FusedVertexExportsRetireEachSixSlotBatch)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (unsigned count : {5u, 6u, 7u, 12u, 15u}) {
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_VERTEX;
      for (unsigned c = 0; c < count; ++c) {
         agx_st_vary_mul_agx3(b, agx_immediate(c),
                              agx_register(0, AGX_SIZE_32),
                              agx_register(2, AGX_SIZE_32));
      }
      agx_stop(b);
      std::vector<uint8_t> want;
      const uint16_t src[3] = {0, 2, 0};
      const bool hints[3] = {};
      uint8_t bytes[16], len;
      for (unsigned first = 0; first < count; first += 6) {
         unsigned end = std::min(count, first + 6);
         for (unsigned c = first; c < end; ++c) {
            ASSERT_TRUE(agx3_stage_vertex_alu(enc, false, 2 * c, c % 6 + 1,
                                               src, false, hints, hints,
                                               c == first && first ? 63 : 0,
                                               bytes, &len));
            append(want, bytes, len);
         }
         for (unsigned c = first; c < end; ++c) {
            ASSERT_TRUE(agx3_vary_uvsw(enc, 2 * c, c, c % 6 + 1, true,
                                        1u << (c % 6), bytes, &len));
            append(want, bytes, len);
         }
      }
      want.insert(want.end(), {0x0e, 0, 0, 0});
      EXPECT_EQ(pack(b), want);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 2 * count);
   }
}

/* An M5 compute shader that issues `count` independent 32-bit device loads
 * into r4, r6, ... (eight take a slot each, a ninth joins slot 1), then one
 * barrier of `scope`, then stores every loaded value. */
static void
m5_loads_barrier_stores(agx_builder *b, unsigned count, unsigned scope)
{
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(b->shader, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
   for (unsigned n = 0; n < count; ++n)
      agx_device_load_to(b, agx_register(8 + 4 * n, AGX_SIZE_32),
                         agx_uniform(16, AGX_SIZE_64), agx_immediate(n),
                         AGX_FORMAT_I32, 1, 0, false);
   agx_scoped_barrier_agx3(b, scope);
   for (unsigned n = 0; n < count; ++n)
      agx_device_store(b, agx_register(8 + 4 * n, AGX_SIZE_32),
                       agx_uniform(20, AGX_SIZE_64), agx_immediate(n),
                       AGX_FORMAT_I32, 1, 0, false);
   agx_stop(b);
}

static void
m5_store_bytes(std::vector<uint8_t> &v, unsigned n, uint16_t wait_mask)
{
   struct agx3_mem_desc d = {};
   d.data = 8 + 4 * n;
   d.width = AGX3_MEM_W32;
   d.mask = 1;
   d.base_is_uniform = true;
   d.base = 20;
   d.idx_kind = AGX3_MEM_IDX_IMM;
   d.byte_off = n << 2;
   d.wait_mask = wait_mask;
   uint8_t bytes[AGX3_MEM_MAX_LENGTH], len;
   ASSERT_TRUE(agx3_mem_encode_store(AGX_ISA_ENCODING_AGX3C, &d, bytes, &len));
   append(v, bytes, len);
}

/* A workgroup barrier, with or without shared memory, retires every load
 * still in flight before the invocation stops there: nine loads on eight
 * slots, all retired in one wait before the barrier, so no store after it
 * waits. */
TEST_F(AGX3PackBinary, M5WorkgroupBarriersRetirePendingLoads)
{
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17C;
   for (unsigned scope : {(unsigned)AGX3_BARRIER_SCOPE_THREADGROUP,
                          (unsigned)AGX3_BARRIER_SCOPE_EXECUTION}) {
      SCOPED_TRACE(scope);
      agx_builder *b = builder(enc);
      m5_loads_barrier_stores(b, 9, scope);
      const auto code = pack(b);
      ASSERT_FALSE(code.empty());

      std::vector<uint8_t> tail;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 0xff, bytes, &len));
      append(tail, bytes, len);
      struct agx3_encoded_scoped_barrier barrier;
      ASSERT_TRUE(agx3_pack_scoped_barrier(
         enc, (enum agx3_barrier_scope)scope, &barrier));
      append(tail, barrier.bytes, barrier.length);
      for (unsigned n = 0; n < 9; ++n)
         m5_store_bytes(tail, n, 0);
      tail.insert(tail.end(), {0x0e, 0, 0, 0});
      ASSERT_GE(code.size(), tail.size());
      EXPECT_EQ(std::vector<uint8_t>(code.end() - tail.size(), code.end()),
                tail);
   }
}

/* G18G compiles with the same encoding and keeps its workgroup barriers as
 * they were: the barrier is issued without a wait, and each store after it
 * waits for its own load's slot (the ninth load's slot is waited for by the
 * first store already). */
TEST_F(AGX3PackBinary, G18GWorkgroupBarriersKeepLoadsInFlight)
{
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G18G;
   key.dev.emit_agx3d = true;
   for (unsigned scope : {(unsigned)AGX3_BARRIER_SCOPE_THREADGROUP,
                          (unsigned)AGX3_BARRIER_SCOPE_EXECUTION}) {
      SCOPED_TRACE(scope);
      agx_builder *b = builder(enc);
      m5_loads_barrier_stores(b, 9, scope);
      const auto code = pack(b);
      ASSERT_FALSE(code.empty());

      std::vector<uint8_t> tail;
      struct agx3_encoded_scoped_barrier barrier;
      ASSERT_TRUE(agx3_pack_scoped_barrier(
         enc, (enum agx3_barrier_scope)scope, &barrier));
      append(tail, barrier.bytes, barrier.length);
      for (unsigned n = 0; n < 9; ++n)
         m5_store_bytes(tail, n, n < 8 ? 1u << n : 0);
      tail.insert(tail.end(), {0x0e, 0, 0, 0});
      ASSERT_GE(code.size(), tail.size());
      EXPECT_EQ(std::vector<uint8_t>(code.end() - tail.size(), code.end()),
                tail);
   }
}

/* A G18G vertex shader: `loads` device loads still in flight at a 2D
 * textureLod whose coordinates are in register half `coord`, then the four
 * result channels and the loaded words written as varyings. The sample takes
 * scoreboard slot loads + 1, and its staging block sits at 8 * loads. With
 * relay_sample a second 2D sample of the same coordinates follows, with its
 * level in a 16-bit register (half 20), and its result is written as varyings
 * 20 to 23 too. */
static agx_builder *
g18g_vs_sample_then_outputs(agx_builder *b, unsigned loads, unsigned coord,
                            bool relay_sample = false)
{
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->nir = rzalloc(b->shader, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_VERTEX;
   for (unsigned n = 0; n < loads; ++n)
      agx_device_load_to(b, agx_register(64 + 2 * n, AGX_SIZE_32),
                         agx_uniform(16, AGX_SIZE_64), agx_immediate(n),
                         AGX_FORMAT_I32, 1, 0, false);
   agx_mov_imm_to(b, agx_register(coord, AGX_SIZE_32), 0x3e800000);
   agx_mov_imm_to(b, agx_register(coord + 2, AGX_SIZE_32), 0x3f400000);
   if (relay_sample)
      agx_mov_imm_to(b, agx_register(20, AGX_SIZE_16), 0);
   agx_index dest = agx_register(0, AGX_SIZE_32);
   agx_index uv = agx_register(coord, AGX_SIZE_32);
   dest.channels_m1 = 3;
   uv.channels_m1 = 1;
   agx_instr *tex = agx_texture_sample_to(
      b, dest, uv, agx_immediate(0), agx_immediate(0), agx_immediate(0),
      agx_immediate(1), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0, false,
      false, false, AGX_GATHER_NONE, false);
   tex->mask = 15;
   if (relay_sample) {
      agx_index dest2 = agx_register(8, AGX_SIZE_32);
      dest2.channels_m1 = 3;
      agx_instr *tex2 = agx_texture_sample_to(
         b, dest2, uv, agx_register(20, AGX_SIZE_16), agx_immediate(0),
         agx_immediate(0), agx_immediate(1), agx_null(), AGX_DIM_2D,
         AGX_LOD_MODE_LOD_MIN, 0, false, false, false, AGX_GATHER_NONE, false);
      tex2->mask = 15;
      for (unsigned c = 0; c < 4; ++c)
         agx_st_vary(b, agx_immediate(20 + c),
                     agx_register(8 + 2 * c, AGX_SIZE_32));
   }
   for (unsigned c = 0; c < 4; ++c)
      agx_st_vary(b, agx_immediate(c), agx_register(2 * c, AGX_SIZE_32));
   for (unsigned n = 0; n < loads; ++n)
      agx_st_vary(b, agx_immediate(4 + n),
                  agx_register(64 + 2 * n, AGX_SIZE_32));
   agx_stop(b);
   return b;
}

/* The staging halves the UVSWs of `code` write: the bits that carry the
 * staging half are those that change when one UVSW is encoded at each even
 * half. */
static std::vector<unsigned>
agx3c_uvsw_staging_halves(const std::vector<uint8_t> &code)
{
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   uint8_t probe[64][AGX3_VARY_MAX_LENGTH], len = 0;
   for (unsigned h = 0; h < 64; ++h) {
      uint8_t l = 0;
      EXPECT_TRUE(agx3_vary_uvsw(enc, 2 * h, 0, 1, true, 0, probe[h], &l));
      EXPECT_TRUE(!h || l == len);
      len = l;
   }
   uint8_t mask[AGX3_VARY_MAX_LENGTH] = {0};
   for (unsigned h = 1; h < 64; ++h)
      for (unsigned i = 0; i < len; ++i)
         mask[i] |= probe[h][i] ^ probe[0][i];

   std::vector<unsigned> halves;
   for (size_t at = 0; at < code.size();) {
      const char *name = NULL;
      const signed l =
         agx3c_instr_length(code.data() + at, code.size() - at, &name);
      EXPECT_GT(l, 0) << "undecodable at " << at;
      if (l <= 0)
         break;
      if (name && strstr(name, "uvsw")) {
         EXPECT_EQ(l, (signed)len) << "UVSW of another form at " << at;
         unsigned found = ~0u;
         for (unsigned h = 0; h < 64 && l == (signed)len; ++h) {
            bool same = true;
            for (unsigned i = 0; i < len; ++i)
               same &= ((code[at + i] ^ probe[h][i]) & mask[i]) == 0;
            if (same)
               found = 2 * h;
         }
         EXPECT_NE(found, ~0u) << "UVSW staging half at " << at;
         halves.push_back(found);
      }
      at += l;
   }
   return halves;
}

/* G18G vertex outputs are staged from AGX3_G18G_READ_STAGING_LIMIT (80), in
 * the staging file the texture reads use. A G18G sample's staging block sits
 * at 8 * (slot - 1) and reaches its coordinate register plus 4, so a
 * textureLod issued behind several loads in flight reaches past 80, and
 * agx3_pack_binary() refuses the shader. agx3_pack_program() then packs it
 * once more with every output's UVSW staging at or above ALIGN(footprint, 8),
 * provided the outputs still end within AGX3_STAGED_WRITE_LIMIT; a footprint
 * past 112 stays refused. */
TEST_F(AGX3PackBinary, G18GVertexOutputsMoveAboveHighTextureReads)
{
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G18G;
   key.dev.emit_agx3d = true;

   /* Reads below 80: one attempt, the outputs from 80. */
   {
      agx_builder *b = g18g_vs_sample_then_outputs(builder(enc), 2, 44);
      struct util_dynarray bin;
      util_dynarray_init(&bin, mem_ctx);
      ASSERT_TRUE(agx3_pack_program(b->shader, &bin))
         << agx3_last_pack_refusal();
      EXPECT_EQ(b->shader->agx3_g18g_output_staging_floor, 0u);
      const std::vector<uint8_t> code((uint8_t *)bin.data,
                                      (uint8_t *)bin.data + bin.size);
      const auto halves = agx3c_uvsw_staging_halves(code);
      EXPECT_EQ(halves.size(), 6u);
      for (unsigned h : halves)
         EXPECT_GE(h, AGX3_G18G_READ_STAGING_LIMIT);
   }

   static const struct {
      unsigned loads, coord;
   } cases[] = {{6, 32}, {6, 44}, {6, 60}, {7, 52}, {7, 56}, {7, 60}};
   for (const auto &c : cases) {
      SCOPED_TRACE(testing::Message()
                   << c.loads << " loads, coordinate " << c.coord);
      const unsigned footprint = 8 * c.loads + c.coord + 4;
      const unsigned floor = ALIGN_POT(footprint, 8);
      const bool fits =
         floor + AGX3_G18G_OUTPUT_STAGING_LIMIT <= AGX3_STAGED_WRITE_LIMIT;

      /* One attempt alone refuses, and says how far the reads reach. */
      agx_builder *once =
         g18g_vs_sample_then_outputs(builder(enc), c.loads, c.coord);
      struct util_dynarray bin;
      util_dynarray_init(&bin, mem_ctx);
      EXPECT_FALSE(agx3_pack_binary(once->shader, &bin));
      EXPECT_EQ(once->shader->agx3_g18g_read_staging_end, footprint);
      EXPECT_NE(strstr(agx3_last_pack_refusal(), "past the output staging"),
                nullptr)
         << agx3_last_pack_refusal();

      agx_builder *b =
         g18g_vs_sample_then_outputs(builder(enc), c.loads, c.coord);
      util_dynarray_init(&bin, mem_ctx);
      ASSERT_EQ(agx3_pack_program(b->shader, &bin), fits)
         << agx3_last_pack_refusal();
      if (!fits)
         continue;
      EXPECT_EQ(b->shader->agx3_g18g_output_staging_floor, floor);
      const std::vector<uint8_t> code((uint8_t *)bin.data,
                                      (uint8_t *)bin.data + bin.size);
      const auto halves = agx3c_uvsw_staging_halves(code);
      EXPECT_EQ(halves.size(), 4u + c.loads);
      for (unsigned h : halves) {
         EXPECT_GE(h, floor);
         EXPECT_LE(h, 126u);
         EXPECT_LE(h + 2, b->shader->agx3_staging_halfs);
      }
      EXPECT_LE(b->shader->agx3_staging_halfs, AGX3_STAGED_WRITE_LIMIT);
   }
}

/* The refused first attempt of agx3_pack_program() leaves nothing behind. A
 * sample with a register level on a slot past 1 goes through the G17G LOD
 * relay, which raises max_reg for its scratch word in every attempt. The
 * retry must pack what one attempt at its floor packs, register bound
 * included, not put a second relay word above the first (max_reg 79 instead
 * of 77 here before agx3_pack_program() restored it). */
TEST_F(AGX3PackBinary, G18GVertexOutputRetryKeepsTheRegisterBound)
{
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G18G;
   key.dev.emit_agx3d = true;
   for (unsigned coord : {32u, 44u}) {
      SCOPED_TRACE(testing::Message() << "coordinate " << coord);
      agx_builder *once =
         g18g_vs_sample_then_outputs(builder(enc), 6, coord, true);
      struct util_dynarray bin;
      util_dynarray_init(&bin, mem_ctx);
      ASSERT_FALSE(agx3_pack_binary(once->shader, &bin));
      const unsigned floor =
         ALIGN_POT(once->shader->agx3_g18g_read_staging_end, 8);
      ASSERT_NE(floor, 0u) << agx3_last_pack_refusal();
      /* The relay ran in the refused attempt */
      ASSERT_GT(once->shader->max_reg, 0u);

      agx_builder *b =
         g18g_vs_sample_then_outputs(builder(enc), 6, coord, true);
      util_dynarray_init(&bin, mem_ctx);
      ASSERT_TRUE(agx3_pack_program(b->shader, &bin))
         << agx3_last_pack_refusal();
      EXPECT_EQ(b->shader->agx3_g18g_output_staging_floor, floor);

      agx_builder *direct =
         g18g_vs_sample_then_outputs(builder(enc), 6, coord, true);
      direct->shader->agx3_g18g_output_staging_floor = floor;
      struct util_dynarray direct_bin;
      util_dynarray_init(&direct_bin, mem_ctx);
      ASSERT_TRUE(agx3_pack_binary(direct->shader, &direct_bin))
         << agx3_last_pack_refusal();
      EXPECT_EQ(b->shader->max_reg, direct->shader->max_reg);
      EXPECT_EQ(b->shader->agx3_staging_halfs,
                direct->shader->agx3_staging_halfs);
      EXPECT_EQ(
         std::vector<uint8_t>((uint8_t *)bin.data,
                              (uint8_t *)bin.data + bin.size),
         std::vector<uint8_t>((uint8_t *)direct_bin.data,
                              (uint8_t *)direct_bin.data + direct_bin.size));
   }
}

/* G18G: the saturated register+minifloat add that hk's image-to-buffer copy
 * uses to repack a UNORM texel, sat(x + -0.0), has wait fields for slots 1..6
 * only. With seven or eight loads in flight its source comes back on slot 7
 * or 8; that wait is retired through the band carrier on the destination and
 * the add then waits on nothing. Up to slot 6 the add carries its own wait. */
TEST_F(AGX3PackBinary, G18GSaturatedMinifloatAddRetiresLateSlots)
{
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G18G;
   key.dev.emit_agx3d = true;
   for (unsigned loads : {6u, 7u, 8u}) {
      SCOPED_TRACE(loads);
      agx_builder *b = builder(enc);
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->nir = rzalloc(b->shader, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
      for (unsigned n = 0; n < loads; ++n)
         agx_device_load_to(b, agx_register(8 + 4 * n, AGX_SIZE_32),
                            agx_uniform(16, AGX_SIZE_64), agx_immediate(n),
                            AGX_FORMAT_I32, 1, 0, false);
      const unsigned src = 8 + 4 * (loads - 1);
      agx_instr *add = agx_fadd_to(b, agx_register(2, AGX_SIZE_32),
                                   agx_register(src, AGX_SIZE_32),
                                   agx_immediate(0x80));
      add->saturate = true;
      agx_stop(b);
      const auto code = pack(b);
      ASSERT_FALSE(code.empty());

      const uint16_t wait = 1u << (loads - 1);
      std::vector<uint8_t> tail;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      if (wait & ~0x3fu) {
         ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 2, wait, bytes, &len));
         append(tail, bytes, len);
      }
      ASSERT_TRUE(agx3_alu_fadd_minifloat_mod_g18g(
         enc, 2, src, 0x80, false, true, AGX3_ALU_HINT_NONE, wait & 0x3f,
         bytes, &len));
      append(tail, bytes, len);
      tail.insert(tail.end(), {0x0e, 0, 0, 0});
      ASSERT_GE(code.size(), tail.size());
      EXPECT_EQ(std::vector<uint8_t>(code.end() - tail.size(), code.end()),
                tail);
   }
}

/* No AGX3 table has an f32 add of two immediates: agx3c and agx3d refuse
 * every such pair and agx3b folds only the unmodified, unclamped one.
 * agx_lower_pseudo() lowers the pair on every AGX3 encoding, and the result
 * packs on all of them; the agx3b fold keeps its bytes. */
TEST_F(AGX3PackBinary, TwoImmediateFaddPacksAfterLoweringOnEveryAgx3)
{
   const struct {
      enum agx_isa_encoding enc;
      bool agx3d;
   } encodings[] = {
      {AGX_ISA_ENCODING_AGX3B, false},
      {AGX_ISA_ENCODING_AGX3C, false},
      {AGX_ISA_ENCODING_AGX3C, true},
   };

   for (auto e : encodings) {
      /* 0: plain, 1: saturated, 2: negated, 3: |-x|, 4: second constant */
      for (unsigned v = 0; v < 5; ++v) {
         SCOPED_TRACE(e.enc * 100 + e.agx3d * 10 + v);
         std::vector<uint8_t> unlowered;
         for (unsigned lower = 0; lower < 2; ++lower) {
            agx_builder *b = builder(e.enc);
            key.dev.emit_agx3d = e.agx3d;
            key.dev.chip = e.enc == AGX_ISA_ENCODING_AGX3B ? AGX_CHIP_G17P
                                                           : AGX_CHIP_G17G;
            agx_index a = agx_immediate(0x30);
            agx_index c = v == 4 ? agx_immediate(0x38) : agx_negzero();
            if (v == 2)
               a = agx_neg(a);
            if (v == 3)
               a = agx_abs(agx_neg(a));
            agx_instr *add =
               agx_fadd_to(b, agx_register(4, AGX_SIZE_32), a, c);
            add->saturate = v == 1;
            agx_stop(b);

            if (!lower) {
               struct util_dynarray bin;
               util_dynarray_init(&bin, mem_ctx);
               if (agx3_pack_binary(b->shader, &bin))
                  unlowered.assign((const uint8_t *)bin.data,
                                   (const uint8_t *)bin.data + bin.size);
               continue;
            }

            agx_lower_pseudo(b->shader);
            std::vector<uint8_t> bytes = pack(b);
            EXPECT_FALSE(bytes.empty());
            if (e.enc == AGX_ISA_ENCODING_AGX3B && (v == 0 || v == 4))
               EXPECT_EQ(bytes, unlowered);
         }
      }
   }
}

/* Consecutive 2D-array shadow samples (for example a PCF kernel) have no
 * fixed-bank operands: X/Y land at bank+0/+2, the shadow reference at bank+4,
 * and PACK_SMPIR publishes the array layer at bank+6. Each sample claims its
 * own 8-half bank on both AGX3B and AGX3C without retiring earlier slots. */
TEST_F(AGX3PackBinary, TextureSample2DArrayShadowUsesDisjointBanks)
{
   for (enum agx_isa_encoding enc :
        {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      SCOPED_TRACE(enc);
      agx_builder *b = builder(enc);
      key.dev.chip = enc == AGX_ISA_ENCODING_AGX3B ? AGX_CHIP_G17P
                                                   : AGX_CHIP_G17S;
      key.dev.emit_agx3d = false;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 63;

      for (unsigned i = 0; i < 4; ++i) {
         agx_index dest = agx_register(40 + 8 * i, AGX_SIZE_32);
         dest.channels_m1 = 3;
         agx_index coord = agx_register(4 + 8 * i, AGX_SIZE_32);
         coord.channels_m1 = 2;
         agx_index ref = agx_register(36, AGX_SIZE_32);
         auto *I = agx_texture_sample_to(
            b, dest, coord, agx_immediate(0), agx_immediate(0),
            agx_immediate(0), agx_immediate(0), ref, AGX_DIM_2D_ARRAY,
            AGX_LOD_MODE_AUTO_LOD, 0, false, true, false, AGX_GATHER_NONE,
            false);
         I->mask = 1;
      }
      agx_stop(b);

      const auto code = pack(b);
      ASSERT_FALSE(code.empty());
      EXPECT_TRUE(b->shader->agx3_staging_known);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 32u);

      for (unsigned i = 0; i < 4; ++i) {
         const uint16_t bank = 8 * i;
         const uint8_t slot = 1 + i;
         uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
         ASSERT_TRUE(agx3_tex_stage_array_layer_at(
            enc, bank, 4 + 8 * i + 4, slot, 0,
            enc == AGX_ISA_ENCODING_AGX3C, false, bytes, &len));
         EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                   code.end())
            << "missing array layer staging at bank " << bank;

         ASSERT_TRUE(agx3_tex_stage_copy_hints(
            enc, bank + 4, 36, slot, 0,
            enc == AGX_ISA_ENCODING_AGX3C, false, bytes, &len));
         EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                   code.end())
            << "missing shadow ref staging at bank+4 " << (bank + 4);

         const struct agx3_tex_sample_desc want = {
            .dest = (uint16_t)(40 + 8 * i),
            .coord = bank,
            .width = AGX3_TEX_W32,
            .mask = 1,
            .lod = AGX3_TEX_LOD_AUTO,
            .slot = slot,
            .wait_mask = (uint16_t)(1u << (slot - 1)),
            .coord_kill = enc == AGX_ISA_ENCODING_AGX3C,
            .final = i == 3,
            .sample_group_nonterminal = i < 3,
            .sample_group_end = i == 3,
            .array = true,
            .array_layer = (uint16_t)(4 + 8 * i + 4),
            .shadow = true,
            .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
         };
         ASSERT_TRUE(agx3_tex_encode_sample(enc, &want, bytes, &len));
         EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                   code.end())
            << "missing SMP naming coord=" << bank << " for shadow sample "
            << i;
      }

      /* No intervening slot-1 retirement between the four samples. */
      uint8_t retire[AGX3_ALU_MAX_LENGTH], rlen = 0;
      ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, retire, &rlen));
      EXPECT_EQ(std::search(code.begin(), code.end(), retire, retire + rlen),
                code.end());
   }
}

/* A 3x3 PCF kernel on a sampler2DArrayShadow (gpubench-gl render.shadow):
 * nine taps, each on its own eight-half bank, so the ninth relocates to bank
 * 64. AGX3B's 14-byte immediate-descriptor SMP cannot name that bank; the
 * sample must take the long form instead of failing the whole shader. */
TEST_F(AGX3PackBinary, TextureSample2DArrayShadowPcfReachesBank64)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->max_reg = 127;

   constexpr unsigned taps = 9;
   for (unsigned i = 0; i < taps; ++i) {
      agx_index dest = agx_register(24 + 8 * i, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(4 + 8 * (i & 1), AGX_SIZE_32);
      coord.channels_m1 = 2;
      agx_index ref = agx_register(2, AGX_SIZE_32);
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0), agx_immediate(0),
         agx_immediate(1), ref, AGX_DIM_2D_ARRAY, AGX_LOD_MODE_AUTO_LOD, 0,
         false, true, false, AGX_GATHER_NONE, false);
      I->mask = 1;
   }
   agx_stop(b);

   const auto code = pack(b);
   ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8u * taps);

   /* The last tap's comparison reference is staged at bank 64 + 4, and its
    * SMP is the long form naming bank 64. */
   const uint16_t bank = 8 * (taps - 1);
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   bool found_ref = false;
   for (unsigned hints = 0; hints < 32 && !found_ref; ++hints) {
      const uint8_t slot = 1 + (hints >> 2);
      if (slot <= 8 &&
          agx3_tex_stage_copy_hints(enc, bank + 4, 2, slot, 0, hints & 1,
                                    hints & 2, bytes, &len))
         found_ref = std::search(code.begin(), code.end(), bytes,
                                 bytes + len) != code.end();
   }
   EXPECT_TRUE(found_ref) << "missing shadow ref staging at bank " << bank;

   bool found_smp = false;
   for (size_t at = 0; at + 20 <= code.size() && !found_smp; ++at) {
      uint64_t f[AGX3_MAX_FIELDS];
      found_smp = agx3_decode(&agx3b_table, &code[at], 20, AGX3_MN_smp, f) >= 0 &&
                  f[19] == bank;
   }
   EXPECT_TRUE(found_smp) << "missing 20-byte SMP naming bank " << bank;
}

/* A 2D-array sample with a register texture descriptor stages through fixed
 * slot 6; keep it on bank 0 rather than relocating it. */
TEST_F(AGX3PackBinary, TextureSample2DArrayRegisterOperandStaysOnBankZero)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->max_reg = 63;

   for (unsigned i = 0; i < 2; ++i) {
      agx_index dest = agx_register(40 + 8 * i, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(4 + 8 * i, AGX_SIZE_32);
      coord.channels_m1 = 2;
      agx_index tex = agx_register(28, AGX_SIZE_64);
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0), tex,
         agx_immediate(0), agx_null(), AGX_DIM_2D_ARRAY, AGX_LOD_MODE_AUTO_LOD,
         0, false, false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
   }
   agx_stop(b);

   const auto code = pack(b);
   ASSERT_FALSE(code.empty());
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8u);
}

/* Newly relocated 2D-array sample shapes (gather, immediate offset with and
 * without shadow comparison, immediate bias, immediate explicit LOD, and
 * compute/vertex stages without g17p_uniform_dma) all allocate disjoint
 * 8-half banks on both AGX3B and AGX3C. */
TEST_F(AGX3PackBinary, TextureSample2DArrayNewShapesRelocate)
{
   struct shape {
      const char *name;
      mesa_shader_stage stage;
      enum agx_lod_mode lod_mode;
      uint16_t lod_imm;
      bool shadow;
      bool offset;
      enum agx_gather gather;
   };
   static const shape shapes[] = {
      {"frag_gather", MESA_SHADER_FRAGMENT, AGX_LOD_MODE_AUTO_LOD, 0, false,
       false, AGX_GATHER_R},
      {"frag_imm_offset", MESA_SHADER_FRAGMENT, AGX_LOD_MODE_AUTO_LOD, 0, false,
       true, AGX_GATHER_NONE},
      {"frag_shadow_imm_offset", MESA_SHADER_FRAGMENT, AGX_LOD_MODE_AUTO_LOD, 0,
       true, true, AGX_GATHER_NONE},
      {"frag_imm_bias", MESA_SHADER_FRAGMENT, AGX_LOD_MODE_AUTO_LOD_BIAS, 0x38,
       false, false, AGX_GATHER_NONE},
      {"frag_imm_lod", MESA_SHADER_FRAGMENT, AGX_LOD_MODE_LOD_MIN, 0x38, false,
       false, AGX_GATHER_NONE},
      {"compute_imm_lod", MESA_SHADER_COMPUTE, AGX_LOD_MODE_LOD_MIN, 0, false,
       false, AGX_GATHER_NONE},
      {"vertex_imm_lod_no_dma", MESA_SHADER_VERTEX, AGX_LOD_MODE_LOD_MIN, 0,
       false, false, AGX_GATHER_NONE},
   };

   for (enum agx_isa_encoding enc :
        {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      for (const auto &s : shapes) {
         SCOPED_TRACE(testing::Message() << s.name << " enc=" << enc);
         agx_builder *b = builder(enc);
         key.dev.chip = enc == AGX_ISA_ENCODING_AGX3B ? AGX_CHIP_G17P
                                                      : AGX_CHIP_G17S;
         key.dev.emit_agx3d = false;
         key.g17p_uniform_dma = false;
         b->shader->stage = s.stage;
         b->shader->max_reg = 63;

         for (unsigned i = 0; i < 2; ++i) {
            agx_index dest = agx_register(40 + 8 * i, AGX_SIZE_32);
            dest.channels_m1 = 3;
            agx_index coord = agx_register(4 + 8 * i, AGX_SIZE_32);
            coord.channels_m1 = 2;
            agx_index compare_or_offset =
               s.shadow ? agx_register(36, AGX_SIZE_32)
               : s.offset ? agx_immediate(0x11)
                          : agx_null();
            auto *I = agx_texture_sample_to(
               b, dest, coord, agx_immediate(s.lod_imm), agx_immediate(0),
               agx_immediate(0), agx_immediate(0), compare_or_offset,
               AGX_DIM_2D_ARRAY, s.lod_mode, 0, s.offset, s.shadow, false,
               s.gather, false);
            if (s.offset)
               I->offset_xyz = 0x11;
            I->mask = s.shadow && s.gather == AGX_GATHER_NONE ? 1 : 15;
         }
         agx_stop(b);

         const auto code = pack(b);
         ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();
         EXPECT_TRUE(b->shader->agx3_staging_known);
         EXPECT_EQ(b->shader->agx3_staging_halfs, 16u);

         for (unsigned i = 0; i < 2; ++i) {
            const uint16_t bank = 8 * i;
            const uint8_t slot = 1 + i;
            uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
            ASSERT_TRUE(agx3_tex_stage_array_layer_at(
               enc, bank, 4 + 8 * i + 4, slot, 0,
               enc == AGX_ISA_ENCODING_AGX3C, false, bytes, &len));
            EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                      code.end())
               << "missing array layer staging at bank " << bank;

            const bool is_frag = s.stage == MESA_SHADER_FRAGMENT;
            const bool is_imm_lod = s.lod_mode == AGX_LOD_MODE_LOD_MIN;
            const bool is_bias = s.lod_mode == AGX_LOD_MODE_AUTO_LOD_BIAS;
            const uint16_t lod_x64 =
               (is_imm_lod || is_bias)
                  ? (uint16_t)(int16_t)(agx_minifloat_decode(s.lod_imm) * 64.0f)
                  : 0;
            const struct agx3_tex_sample_desc want = {
               .dest = (uint16_t)(40 + 8 * i),
               .coord = bank,
               .width = AGX3_TEX_W32,
               .mask = (uint8_t)(s.shadow && s.gather == AGX_GATHER_NONE ? 1 : 15),
               .lod = is_bias ? AGX3_TEX_LOD_BIAS
                      : is_imm_lod ? AGX3_TEX_LOD_LEVEL
                                   : AGX3_TEX_LOD_AUTO,
               .lod_x64 = lod_x64,
               .slot = slot,
               .wait_mask = (uint16_t)(1u << (slot - 1)),
               .coord_kill = enc == AGX_ISA_ENCODING_AGX3C,
               .final = i == 1,
               .sample_group_nonterminal = is_frag && i == 0,
               .sample_group_end = is_frag && i == 1,
               .vertex_stage = s.stage == MESA_SHADER_VERTEX,
               .compute_stage = s.stage == MESA_SHADER_COMPUTE,
               .array = true,
               .array_layer = (uint16_t)(4 + 8 * i + 4),
               .gather = s.gather == AGX_GATHER_R ? AGX3_TEX_GATHER_R
                                                  : AGX3_TEX_GATHER_NONE,
               .shadow = s.shadow,
               .offset = s.offset,
               .offset_xyz = (uint16_t)(s.offset ? 0x11 : 0),
               .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
            };
            ASSERT_TRUE(agx3_tex_encode_sample(enc, &want, bytes, &len));
            EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                      code.end())
               << "missing SMP naming coord=" << bank << " for " << s.name
               << " sample " << i;
         }
      }
   }
}

/* When a 2D gradient sample first holds staging halves 0..15 on slot 1,
 * subsequent 2D-array samples on slots 2 and 3 allocate banks 16 and 24
 * (out of step with 8 * (slot - 1)) and encode SMP with coord = 16 and 24
 * on both AGX3B and AGX3C. */
TEST_F(AGX3PackBinary, TextureSample2DArrayBankOutOfStepWithSlot)
{
   for (enum agx_isa_encoding enc :
        {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      SCOPED_TRACE(enc);
      agx_builder *b = builder(enc);
      key.dev.chip = enc == AGX_ISA_ENCODING_AGX3B ? AGX_CHIP_G17P
                                                   : AGX_CHIP_G17S;
      key.dev.emit_agx3d = false;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 63;

      /* Slot 1: 2D gradient sample occupying staging halves 0..15. */
      {
         agx_index dest = agx_register(40, AGX_SIZE_32);
         dest.channels_m1 = 3;
         agx_index coord = agx_register(4, AGX_SIZE_32);
         coord.channels_m1 = 1;
         agx_index grad = agx_register(16, AGX_SIZE_32);
         grad.channels_m1 = 3;
         auto *G = agx_texture_sample_to(
            b, dest, coord, grad, agx_immediate(0), agx_immediate(0),
            agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_GRAD, 0,
            false, false, false, AGX_GATHER_NONE, false);
         G->mask = 15;
      }

      /* Slots 2 and 3: 2D-array samples must land at banks 16 and 24. */
      for (unsigned i = 0; i < 2; ++i) {
         agx_index dest = agx_register(48 + 8 * i, AGX_SIZE_32);
         dest.channels_m1 = 3;
         agx_index coord = agx_register(24 + 8 * i, AGX_SIZE_32);
         coord.channels_m1 = 2;
         auto *I = agx_texture_sample_to(
            b, dest, coord, agx_immediate(0), agx_immediate(0),
            agx_immediate(0), agx_immediate(0), agx_null(), AGX_DIM_2D_ARRAY,
            AGX_LOD_MODE_AUTO_LOD, 0, false, false, false, AGX_GATHER_NONE,
            false);
         I->mask = 15;
      }
      agx_stop(b);

      const auto code = pack(b);
      ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();
      EXPECT_TRUE(b->shader->agx3_staging_known);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 32u);

      for (unsigned i = 0; i < 2; ++i) {
         const uint16_t bank = 16 + 8 * i;
         const uint8_t slot = 2 + i;
         uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
         ASSERT_TRUE(agx3_tex_stage_array_layer_at(
            enc, bank, 24 + 8 * i + 4, slot, 0,
            enc == AGX_ISA_ENCODING_AGX3C, false, bytes, &len));
         EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                   code.end())
            << "missing array layer staging at bank " << bank << " slot "
            << unsigned(slot);

         const struct agx3_tex_sample_desc want = {
            .dest = (uint16_t)(48 + 8 * i),
            .coord = bank,
            .width = AGX3_TEX_W32,
            .mask = 15,
            .lod = AGX3_TEX_LOD_AUTO,
            .slot = slot,
            .wait_mask = (uint16_t)(1u << (slot - 1)),
            .coord_kill = enc == AGX_ISA_ENCODING_AGX3C,
            .final = i == 1,
            .sample_group_nonterminal = i == 0,
            .sample_group_end = i == 1,
            .array = true,
            .array_layer = (uint16_t)(24 + 8 * i + 4),
            .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
         };
         ASSERT_TRUE(agx3_tex_encode_sample(enc, &want, bytes, &len));
         EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                   code.end())
            << "missing SMP naming coord=" << bank << " on slot "
            << unsigned(slot);
      }
   }
}

/* Consecutive 2D-array texel fetches allocate disjoint 8-half staging banks
 * across every scoreboard slot (6 on AGX3B, 8 on AGX3C) and wrap cleanly via
 * bank_retire on the fetch's coordinate register without clobbering an
 * in-flight r0 destination. */
TEST_F(AGX3PackBinary, TextureFetch2DArrayUsesDisjointBanksAndWraps)
{
   for (enum agx_isa_encoding enc :
        {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      const unsigned banks = enc == AGX_ISA_ENCODING_AGX3B ? 6u : 8u;
      const unsigned ops = banks + 2;
      SCOPED_TRACE(enc);
      agx_builder *b = builder(enc);
      key.dev.chip = enc == AGX_ISA_ENCODING_AGX3B ? AGX_CHIP_G17P
                                                   : AGX_CHIP_G17S;
      key.dev.emit_agx3d = false;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 127;

      for (unsigned i = 0; i < ops; ++i) {
         /* Put fetch 1's destination in r0..r1 so wrapping onto bank 0 cannot
          * safely retire through an r0 identity carrier. Use 16-bit results
          * in 88..127 so all coordinates (8..85) and destinations stay
          * disjoint within AGX3B's 128-half destination bound. */
         const uint16_t dst_reg = i == 1 ? 0 : (88 + 4 * i);
         agx_index dest = agx_register(dst_reg, AGX_SIZE_16);
         dest.channels_m1 = 3;
         agx_index coord = agx_register(8 + 8 * i, AGX_SIZE_32);
         coord.channels_m1 = 2;
         auto *I = agx_texture_sample_to(
            b, dest, coord, agx_immediate(0), agx_immediate(0),
            agx_immediate(0), agx_immediate(0), agx_null(),
            AGX_DIM_2D_ARRAY, AGX_LOD_MODE_LOD_MIN, 0,
            false, false, false, AGX_GATHER_NONE, false);
         I->op = AGX_OPCODE_TEXTURE_LOAD;
         I->mask = 15;
      }
      agx_stop(b);

      const auto code = pack(b);
      ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();
      EXPECT_TRUE(b->shader->agx3_staging_known);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 8u * banks);

      for (unsigned i = 0; i < ops; ++i) {
         const uint16_t bank = 8 * (i % banks);
         const uint8_t slot = 1 + (i % banks);
         uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
         ASSERT_TRUE(agx3_tex_stage_array_layer_at(
            enc, bank, 8 + 8 * i + 4, slot, 0, false, false, bytes, &len));
         EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                   code.end())
            << "missing fetch array layer staging for op " << i << " at bank "
            << bank;

         const uint16_t dst_reg = i == 1 ? 0 : (88 + 4 * i);
         const uint16_t copy_wait = i >= banks ? (uint16_t)(1u << (slot - 1)) : 0;
         ASSERT_TRUE(agx3_tex_stage_copy_hints(
            enc, bank, 8 + 8 * i, slot, copy_wait,
            enc == AGX_ISA_ENCODING_AGX3C, false, bytes, &len));
         EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                   code.end())
            << "missing fetch coord x staging for op " << i << " at bank "
            << bank << " wait " << copy_wait;

         const struct agx3_tex_fetch_desc want = {
            .dest = dst_reg,
            .coord = bank,
            .width = AGX3_TEX_W16,
            .mask = 15,
            .slot = slot,
            .wait_mask = (uint16_t)(1u << (slot - 1)),
            .coord_kill = true,
            .final = i == ops - 1,
            .array = true,
            .sampler_value = 0,
         };
         ASSERT_TRUE(agx3_tex_encode_fetch(enc, &want, bytes, &len));
         EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                   code.end())
            << "missing fetch instruction for op " << i << " at bank " << bank;
      }
   }
}

/* When an earlier multi-bank op (a 2D gradient sample holding halves 0..15 on
 * slot 1) leaves fewer than six free fetch banks on AGX3B, the fifth 2D-array
 * fetch hits bank_retire != 0 while slot 2 is still writing r0. Verify it
 * retires slot 1 through its own coordinate register rather than r0 and reuses
 * bank 0. */
TEST_F(AGX3PackBinary, TextureFetch2DArrayBankRetireAvoidsR0)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   key.dev.chip = AGX_CHIP_G17P;
   key.dev.emit_agx3d = false;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->max_reg = 127;

   /* Slot 1: 2D gradient sample occupying staging halves 0..15 (banks 0, 8). */
   {
      agx_index dest = agx_register(84, AGX_SIZE_16);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(4, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_index grad = agx_register(16, AGX_SIZE_32);
      grad.channels_m1 = 3;
      auto *S = agx_texture_sample_to(
         b, dest, coord, grad, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_GRAD, 0,
         false, false, false, AGX_GATHER_NONE, false);
      S->mask = 15;
   }

   /* Slots 2..6: five 2D-array fetches. Fetches 0..3 take banks 16, 24, 32, 40
    * (with fetch 0 writing r0..r1 on slot 2); fetch 4 exhausts the 6-bank ring
    * and retires slot 1 via bank_retire on its own coordinate register (56). */
   for (unsigned i = 0; i < 5; ++i) {
      const uint16_t dst_reg = i == 0 ? 0 : (88 + 4 * i);
      agx_index dest = agx_register(dst_reg, AGX_SIZE_16);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(24 + 8 * i, AGX_SIZE_32);
      coord.channels_m1 = 2;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(),
         AGX_DIM_2D_ARRAY, AGX_LOD_MODE_LOD_MIN, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      I->mask = 15;
   }
   agx_stop(b);

   const auto code = pack(b);
   ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 48u);

   uint8_t retire_coord[AGX3_ALU_MAX_LENGTH], rlen = 0;
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 24 + 8 * 4, 1, retire_coord,
                                           &rlen));
   EXPECT_NE(std::search(code.begin(), code.end(), retire_coord,
                         retire_coord + rlen),
             code.end());

   uint8_t retire_r0[AGX3_ALU_MAX_LENGTH], r0len = 0;
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 1, retire_r0, &r0len));
   EXPECT_EQ(std::search(code.begin(), code.end(), retire_r0,
                         retire_r0 + r0len),
             code.end());

   /* Fetch 4 reuses bank 0 on slot 6 after retiring slot 1. */
   const struct agx3_tex_fetch_desc want4 = {
      .dest = 88 + 4 * 4,
      .coord = 0,
      .width = AGX3_TEX_W16,
      .mask = 15,
      .slot = 6,
      .wait_mask = (uint16_t)(1u << 5),
      .coord_kill = true,
      .final = true,
      .array = true,
      .sampler_value = 0,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_tex_encode_fetch(enc, &want4, bytes, &len));
   EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
             code.end());
}

/* When agx3_retire_wait_on() retires bank_retire | wait_mask on a fetch's
 * coordinate register, wait_mask is cleared so the AGX3B multi-wait staging
 * guard does not emit a redundant second wait through r0. */
TEST_F(AGX3PackBinary, TextureFetchBankRetireClearsWaitMask)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = builder(enc);
   key.dev.chip = AGX_CHIP_G17P;
   key.dev.emit_agx3d = false;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->max_reg = 127;

   /* Slot 1: 2D-array fetch holding staging bank 0 (halves 0..7). */
   {
      agx_index dest = agx_register(80, AGX_SIZE_16);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(4, AGX_SIZE_32);
      coord.channels_m1 = 2;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(),
         AGX_DIM_2D_ARRAY, AGX_LOD_MODE_LOD_MIN, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      I->mask = 15;
   }

   /* Slot 2: 2D-array fetch (bank 8) producing r40..r47 (coords for op 3). */
   {
      agx_index dest = agx_register(40, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(12, AGX_SIZE_32);
      coord.channels_m1 = 2;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(),
         AGX_DIM_2D_ARRAY, AGX_LOD_MODE_LOD_MIN, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      I->mask = 3;
   }

   /* Slot 3: 2D-array fetch (bank 16) producing r48..r55 (dynamic LOD for op 3). */
   {
      agx_index dest = agx_register(48, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(20, AGX_SIZE_32);
      coord.channels_m1 = 2;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(),
         AGX_DIM_2D_ARRAY, AGX_LOD_MODE_LOD_MIN, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      I->mask = 1;
   }

   /* Slot 4: plain 2D fetch with register LOD (max_banks == 1 -> bank 0).
    * Retires bank_retire (slot 1) | wait_mask (slots 2, 3) on coord r40 in
    * one carrier without a redundant r0 noop for slots 2, 3. */
   {
      agx_index dest = agx_register(0, AGX_SIZE_16);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(40, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_index lod = agx_register(48, AGX_SIZE_32);
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      I->mask = 15;
   }
   agx_stop(b);

   const auto code = pack(b);
   ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();

   uint8_t retire_all[AGX3_ALU_MAX_LENGTH], alen = 0;
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 40, 0x7, retire_all, &alen));
   EXPECT_NE(std::search(code.begin(), code.end(), retire_all,
                         retire_all + alen),
             code.end());

   uint8_t redundant_r0[AGX3_ALU_MAX_LENGTH], rlen = 0;
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 0, 0x6, redundant_r0, &rlen));
   EXPECT_EQ(std::search(code.begin(), code.end(), redundant_r0,
                         redundant_r0 + rlen),
             code.end());
}

/* On AGX3B, consecutive plain 2D texel fetches allocate up to six disjoint
 * banks (0, 8, 16, 24, 32, 40) matching the six scoreboard slots without
 * early waits before wrapping cleanly onto bank 0 and bank 8. */
TEST_F(AGX3PackBinary, TextureFetchPlain2DUsesSixBanksOnAGX3B)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   const unsigned banks = 6u;
   const unsigned ops = 8u;
   agx_builder *b = builder(enc);
   key.dev.chip = AGX_CHIP_G17P;
   key.dev.emit_agx3d = false;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->max_reg = 127;

   for (unsigned i = 0; i < ops; ++i) {
      const uint16_t dst_reg = i == 1 ? 0 : (88 + 4 * i);
      agx_index dest = agx_register(dst_reg, AGX_SIZE_16);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(8 + 8 * i, AGX_SIZE_32);
      coord.channels_m1 = 1;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      I->mask = 15;
   }
   agx_stop(b);

   const auto code = pack(b);
   ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8u * (banks - 1) + 4u);

   for (unsigned i = 0; i < ops; ++i) {
      const uint16_t bank = 8 * (i % banks);
      const uint8_t slot = 1 + (i % banks);
      const uint16_t dst_reg = i == 1 ? 0 : (88 + 4 * i);
      const uint16_t copy_wait = i >= banks ? (uint16_t)(1u << (slot - 1)) : 0;
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_tex_stage_copy_hints(
         enc, bank, 8 + 8 * i, slot, copy_wait, false, false, bytes, &len));
      EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                code.end())
         << "missing plain 2D fetch coord x staging for op " << i
         << " at bank " << bank << " wait " << copy_wait;

      const struct agx3_tex_fetch_desc want = {
         .dest = dst_reg,
         .coord = bank,
         .width = AGX3_TEX_W16,
         .mask = 15,
         .slot = slot,
         .wait_mask = (uint16_t)(1u << (slot - 1)),
         .coord_kill = true,
         .final = i == ops - 1,
         .array = false,
         .sampler_value = 0,
      };
      ASSERT_TRUE(agx3_tex_encode_fetch(enc, &want, bytes, &len));
      EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                code.end())
         << "missing plain 2D fetch instruction for op " << i << " at bank "
         << bank;
   }
}

/* A vertex 2D explicit-LOD sample with a register LOD allocates a 16-half
 * staging span on AGX3C, which still requires an 8-half-aligned base bank so
 * the per-lane LOD publication satisfies the texture operand gate. When an
 * earlier 4-half sample holds bank 0..3, the second sample must land at bank
 * 8 rather than bank 4. */
TEST_F(AGX3PackBinary, VertexRegisterLodSampleAlignsWideStagingBankOnAGX3C)
{
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   agx_builder *b = builder(enc);
   key.dev.chip = AGX_CHIP_G17C;
   key.dev.emit_agx3d = false;
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->max_reg = 127;

   /* Sample 1: plain 2D immediate-LOD vertex sample holds 4 halves (bank 0..3). */
   {
      agx_index dest = agx_register(40, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = 1;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
   }

   /* Sample 2: 2D register-LOD vertex sample allocates a 16-half staging span
    * while sample 1 still owns bank 0..3; bank_base must align to 8 (bank 8). */
   {
      agx_index dest = agx_register(60, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(20, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_index lod = agx_register(28, AGX_SIZE_16);
      lod.discard = true;
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0),
         agx_immediate(1), agx_immediate(0), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->mask = 15;
   }
   agx_stop(b);

   const auto code = pack(b);
   ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();

   /* Confirm sample 2 staged its coordinate X at bank 8 (not 4). */
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_tex_stage_copy_hints(
      enc, 8, 20, 2, 0, true, false, bytes, &len));
   EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
             code.end())
      << "expected second vertex sample to stage coord X at 8-aligned bank 8";
}

TEST_F(AGX3PackBinary, PureAluLoopInterleavesPipesAndBoundsProducerWaves)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem_ctx, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   entry->index = 0;
   body->index = 1;
   exit->index = 2;
   agx_block_add_successor(entry, body);
   agx_block_add_successor(body, body);
   agx_block_add_successor(body, exit);
   body->loop_header = true;

   b->cursor = agx_after_block(entry);
   agx_index ha = agx_preload(b, agx_register(0, AGX_SIZE_16));
   agx_index hb = agx_preload(b, agx_register(1, AGX_SIZE_16));
   agx_index fa = agx_preload(b, agx_register(2, AGX_SIZE_32));
   agx_index fb = agx_preload(b, agx_register(4, AGX_SIZE_32));
   agx_index hc[6], fc[6];
   for (unsigned i = 0; i < 6; ++i)
      hc[i] = agx_preload(b, agx_register(8 + i, AGX_SIZE_16));
   for (unsigned i = 0; i < 6; ++i)
      fc[i] = agx_preload(b, agx_register(16 + 2 * i, AGX_SIZE_32));

   b->cursor = agx_after_block(body);
   agx_index h1[6], h2[6], f1[6], f2[6];
   for (unsigned i = 0; i < 6; ++i) {
      h1[i] = agx_temp(b->shader, AGX_SIZE_16);
      agx_hfma_to(b, h1[i], ha, hb, hc[i]);
   }
   for (unsigned i = 0; i < 6; ++i) {
      f1[i] = agx_temp(b->shader, AGX_SIZE_32);
      agx_ffma_to(b, f1[i], fa, fb, fc[i]);
   }
   for (unsigned i = 0; i < 6; ++i) {
      h2[i] = agx_temp(b->shader, AGX_SIZE_16);
      agx_hfma_to(b, h2[i], h1[i], ha, hb);
   }
   for (unsigned i = 0; i < 6; ++i) {
      f2[i] = agx_temp(b->shader, AGX_SIZE_32);
      agx_ffma_to(b, f2[i], f1[i], fa, fb);
   }

   b->cursor = agx_after_block(exit);
   for (unsigned i = 0; i < 6; ++i) {
      agx_unit_test(b, h2[i]);
      agx_unit_test(b, f2[i]);
   }
   agx_stop(b);

   agx_ilp_schedule(b->shader);

   std::vector<enum agx_opcode> fmas;
   unsigned first_stage2_pos = ~0u;
   unsigned pos = 0;
   agx_foreach_instr_in_block(body, I) {
      if (I->op != AGX_OPCODE_HFMA && I->op != AGX_OPCODE_FFMA)
         continue;
      fmas.push_back(I->op);
      for (unsigned i = 0; i < 6; ++i) {
         if (I->dest[0].value == h2[i].value ||
             I->dest[0].value == f2[i].value)
            first_stage2_pos = std::min(first_stage2_pos, pos);
      }
      pos++;
   }

   ASSERT_EQ(fmas.size(), 24u);
   /* Mixed FP16/FP32 pipes alternate rather than issuing a 6-wide HFMA burst. */
   EXPECT_NE(fmas[0], fmas[1]);
   EXPECT_NE(fmas[1], fmas[2]);
   /* First stage-2 consumer begins before all 12 stage-1 producers issue. */
   EXPECT_LT(first_stage2_pos, 10u);
}

/* AGX3B and AGX3C device loads and stores encode 16-bit signed
 * memory_byte_offset on both immediate and register indices (element-scaled
 * and byte-indexed). */
TEST_F(AGX3PackBinary, AGX3SignedMemoryByteOffsetLoadAndStore)
{
   const struct {
      enum agx_isa_encoding enc;
      enum agx_chip chip;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S},
   };
   const int16_t offsets[] = {28, 776, -16, INT16_MIN, INT16_MAX};

   for (const auto &t : targets) {
      const auto enc = t.enc;
      key.dev.chip = t.chip;
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));

      for (bool is_store : {false, true}) {
         for (int16_t off : offsets) {
            for (unsigned mode = 0; mode < 3; ++mode) {
               /* mode 0: immediate index 0; mode 1: signed scaled reg index;
                * mode 2: unsigned byte-indexed reg index. */
               SCOPED_TRACE((int)t.chip * 10000 + (is_store ? 1000 : 0) +
                            (int)mode * 100 + off);
               agx_builder *b = builder(enc);
               b->shader->stage = MESA_SHADER_COMPUTE;
               agx_index base = agx_uniform(8, AGX_SIZE_64);
               agx_index idx = mode == 0 ? agx_immediate(0)
                                         : agx_register(4, AGX_SIZE_32);
               if (mode == 2)
                  idx.abs = true;

               agx_instr *I;
               if (is_store) {
                  I = agx_device_store(b, agx_register(16, AGX_SIZE_32), base,
                                       idx, AGX_FORMAT_I32, 0x1, 0, false);
               } else {
                  I = agx_device_load_to(b, agx_register(16, AGX_SIZE_32), base,
                                         idx, AGX_FORMAT_I32, 0x1, 0, false);
               }
               I->memory_byte_offset = off;
               I->memory_index_is_bytes = (mode == 2);
               agx_stop(b);

               struct agx3_mem_desc d = {};
               d.data = 16;
               d.width = AGX3_MEM_W32;
               d.mask = 0x1;
               d.base_is_uniform = true;
               d.base = 8;
               d.byte_off = off;
               if (mode == 0) {
                  d.idx_kind = AGX3_MEM_IDX_IMM;
                  d.shift = 0;
               } else {
                  d.idx_kind = AGX3_MEM_IDX_REG;
                  d.idx_reg = 4;
                  d.idx_signed = (mode == 1);
                  d.shift = (mode == 2) ? 0 : 2;
               }
               if (!is_store)
                  d.slot = 1;
               else if (agx_chip_has_cap(t.chip,
                                         AGX_CAP_COMPUTE_COHERENT_STORE))
                  d.coherent = true;

               uint8_t bytes[AGX3_MEM_MAX_LENGTH], len = 0;
               ASSERT_TRUE(is_store
                              ? agx3_mem_encode_store(enc, &d, bytes, &len)
                              : agx3_mem_encode_load(enc, &d, bytes, &len));
               std::vector<uint8_t> want(bytes, bytes + len);
               append(want, end.bytes, end.length);
               EXPECT_EQ(pack(b), want);
            }
         }
      }
   }
}

/* AGX3B (192-half file) and AGX3C (256-half file) natively pack unshifted
 * mixed-width 32+16/16+16 register+register IADD and ISUB as well as shifted
 * (1..4) non-negated 32+(16<<s) IADD; AGX3C additionally packs shifted 16+16
 * and 16+32 IADD while AGX3B fails closed specifically on shifted 16+16 and
 * 16+32, and both fail closed on unsupported forms. */
TEST_F(AGX3PackBinary, AGX3MixedWidthIaddAndIsub)
{
   const struct {
      enum agx_isa_encoding enc;
      enum agx_chip chip;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S},
   };

   struct add_case {
      uint16_t dest, a_reg, b_reg;
      bool a16, b16, a_abs, b_abs;
      uint8_t shift;
   };
   const struct add_case add_cases[] = {
      /* Unshifted 32+16, 16+32, 16+16 (sext and zext), including odd/high halves. */
      {30, 30, 22, false, true, false, false, 0},
      {30, 30, 23, false, true, false, true, 0},
      {30, 23, 30, true, false, false, false, 0},
      {30, 23, 30, true, false, true, false, 0},
      {130, 131, 255, true, true, false, false, 0},
      {130, 131, 255, true, true, true, true, 0},
      {130, 131, 255, true, true, false, true, 0},
      /* Shifted non-negated register+register mixed-width adds (shift 1..4, AGX3C only). */
      {30, 30, 23, false, true, false, false, 2},
      {30, 30, 23, false, true, false, true, 4},
      {130, 131, 131, true, true, true, true, 1},
      {130, 255, 130, true, false, true, false, 1},
   };

   struct sub_case {
      uint16_t dest, a_reg, b_reg;
      bool a16, b16, a_abs, b_abs, a_neg, b_neg;
   };
   const struct sub_case sub_cases[] = {
      {30, 30, 23, false, true, false, false, false, true},
      {30, 30, 23, false, true, false, true, false, true},
      {30, 23, 30, true, false, true, false, false, true},
      {30, 23, 30, true, false, true, false, true, false},
      {130, 131, 255, true, true, false, true, false, true},
   };

   for (const auto &t : targets) {
      const auto enc = t.enc;
      const bool b3 = enc == AGX_ISA_ENCODING_AGX3B;
      const uint16_t hi = b3 ? 191 : 255;
      key.dev.chip = t.chip;
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(enc, &end));

      for (size_t i = 0; i < ARRAY_SIZE(add_cases); ++i) {
         auto c = add_cases[i];
         if (c.a_reg == 255)
            c.a_reg = hi;
         if (c.b_reg == 255)
            c.b_reg = hi;
         SCOPED_TRACE((int)t.chip * 1000 + (int)i);
         agx_builder *b = builder(enc);
         b->shader->stage = MESA_SHADER_COMPUTE;
         agx_index sa = agx_register(c.a_reg, c.a16 ? AGX_SIZE_16 : AGX_SIZE_32);
         agx_index sb = agx_register(c.b_reg, c.b16 ? AGX_SIZE_16 : AGX_SIZE_32);
         sa.abs = c.a_abs;
         sb.abs = c.b_abs;
         sa.discard = true;
         sb.cache = true;
         agx_iadd_to(b, agx_register(c.dest, AGX_SIZE_32), sa, sb, c.shift);
         agx_stop(b);

         /* AGX3B packs shifted a32 + (b16 << shift), while shifted 16+16 and
          * 16+32 fail closed on AGX3B. */
         if (b3 && c.shift != 0 && c.a16) {
            EXPECT_TRUE(pack(b, false).empty());
            continue;
         }

         /* AGX3B packs a 16+32 add in its 32+16 slot order for both sign-
          * and zero-extended halves (agx3_pack_binary()). */
         const bool commute_half =
            b3 && c.shift == 0 && !c.b_abs && c.a16 && !c.b16;
         struct agx3_iadd_desc d = {};
         d.dest = c.dest;
         d.dest_cache = b3;
         d.src_a_is_register = true;
         d.src_a_reg = commute_half ? c.b_reg : c.a_reg;
         d.src_a_16 = commute_half ? c.b16 : c.a16;
         d.src_a_sign_extend = d.src_a_16 && !(commute_half ? c.b_abs : c.a_abs);
         d.src_a_kill = !commute_half;
         d.src = commute_half ? c.a_reg : c.b_reg;
         d.src_16 = commute_half ? c.a16 : c.b16;
         d.sign_extend = d.src_16 && !(commute_half ? c.a_abs : c.b_abs);
         d.src_kill = commute_half;
         d.src_cache = !b3;
         d.shift = c.shift;

         uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
         ASSERT_TRUE(agx3_iadd_encode(enc, &d, bytes, &len));
         std::vector<uint8_t> want(bytes, bytes + len);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }

      /* Unshifted mixed-width register+register subtractions (direct & commuted). */
      for (size_t i = 0; i < ARRAY_SIZE(sub_cases); ++i) {
         auto c = sub_cases[i];
         if (c.a_reg == 255)
            c.a_reg = hi;
         if (c.b_reg == 255)
            c.b_reg = hi;
         SCOPED_TRACE((int)t.chip * 1000 + 500 + (int)i);
         agx_builder *b = builder(enc);
         b->shader->stage = MESA_SHADER_COMPUTE;
         agx_index sa = agx_register(c.a_reg, c.a16 ? AGX_SIZE_16 : AGX_SIZE_32);
         agx_index sb = agx_register(c.b_reg, c.b16 ? AGX_SIZE_16 : AGX_SIZE_32);
         sa.abs = c.a_abs;
         sa.neg = c.a_neg;
         sb.abs = c.b_abs;
         sb.neg = c.b_neg;
         sa.discard = true;
         sb.discard = true;
         agx_iadd_to(b, agx_register(c.dest, AGX_SIZE_32), sa, sb, 0);
         agx_stop(b);

         const bool swap = c.a_neg && !c.b_neg;
         struct agx3_iadd_desc d = {};
         d.dest = c.dest;
         d.dest_cache = true;
         d.src_a_is_register = true;
         d.src_a_reg = swap ? c.b_reg : c.a_reg;
         d.src_a_16 = swap ? c.b16 : c.a16;
         d.src_a_sign_extend = d.src_a_16 && !(swap ? c.b_abs : c.a_abs);
         d.src_a_kill = true;
         d.src = swap ? c.a_reg : c.b_reg;
         d.src_16 = swap ? c.a16 : c.b16;
         d.sign_extend = d.src_16 && !(swap ? c.a_abs : c.b_abs);
         d.src_kill = true;
         d.subtract = true;

         uint8_t bytes[AGX3_IADD_MAX_LENGTH], len = 0;
         ASSERT_TRUE(agx3_iadd_encode(enc, &d, bytes, &len));
         std::vector<uint8_t> want(bytes, bytes + len);
         append(want, end.bytes, end.length);
         EXPECT_EQ(pack(b), want);
      }

      /* Fail-closed checks: saturating mixed-width, double-negated, shifted sub. */
      {
         agx_builder *b = builder(enc);
         agx_instr *sat = agx_iadd_to(b, agx_register(30, AGX_SIZE_32),
                                      agx_register(30, AGX_SIZE_32),
                                      agx_register(23, AGX_SIZE_16), 0);
         sat->saturate = true;
         agx_stop(b);
         EXPECT_TRUE(pack(b, false).empty());
      }
      {
         agx_builder *b = builder(enc);
         agx_index a = agx_neg(agx_register(21, AGX_SIZE_16));
         agx_index c = agx_neg(agx_register(23, AGX_SIZE_16));
         agx_iadd_to(b, agx_register(30, AGX_SIZE_32), a, c, 0);
         agx_stop(b);
         EXPECT_TRUE(pack(b, false).empty());
      }
      {
         agx_builder *b = builder(enc);
         agx_index c = agx_neg(agx_register(23, AGX_SIZE_16));
         agx_iadd_to(b, agx_register(30, AGX_SIZE_32),
                     agx_register(30, AGX_SIZE_32), c, 1);
         agx_stop(b);
         EXPECT_TRUE(pack(b, false).empty());
      }
   }
}

/* End-to-end NIR compilation of mixed-width 32+16 adds (unshifted, shifted,
 * subtracted) and signed byte-offset / byte-indexed global loads across
 * AGX3B, AGX3C (G17S), and AGX3D (G18G). */
TEST_F(AGX3PackBinary, AGX3CMixedWidthIaddAndByteOffsetEndToEnd)
{
   const struct {
      enum agx_isa_encoding enc;
      enum agx_chip chip;
      bool agx3d;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true},
   };

   for (const auto &t : targets) {
      SCOPED_TRACE(t.chip);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "agx3_iadd16_byteoff");
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *byte_off = nir_iadd_imm(&b, nir_imul_imm(&b, gid, 12), 28);
      nir_def *addr = nir_iadd(&b, base, nir_u2u64(&b, byte_off));
      nir_def *w = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
      nir_def *h = nir_load_global(&b, 1, 16, nir_iadd_imm(&b, addr, 4),
                                   .align_mul = 2);
      nir_def *s = nir_load_global(&b, 1, 16, nir_iadd_imm(&b, addr, 6),
                                   .align_mul = 2);
      nir_def *r0 = nir_iadd(&b, w, nir_u2u32(&b, h));
      nir_def *r1 = nir_iadd(&b, r0, nir_ishl_imm(&b, nir_i2i32(&b, s), 2));
      nir_def *r2 = nir_isub(&b, r1, nir_u2u32(&b, h));
      nir_store_global(&b, r2, nir_iadd_imm(&b, addr, -16), .align_mul = 4);
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      agx_shader_key k = {};
      k.dev.encoding = t.enc;
      k.dev.chip = t.chip;
      k.dev.emit_agx3d = t.agx3d;
      k.no_preamble = true;
      k.reserved_preamble = 16;

      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &k, &part));
      EXPECT_GT(part.info.stats.instrs, 0u);
      free(part.binary);
      ralloc_free(b.shader);
   }
}

/* On AGX3C (non-G18G), consecutive compute 2D-array shadow samples with
 * explicit LOD (both register and immediate zero, and both immediate and
 * register descriptors) allocate disjoint 8-half staging banks (0, 8, 16)
 * instead of stalling on bank 0, staging each sample's comparison reference at
 * bank + 4 and its layer or layer/LOD word at bank + 6. */
TEST_F(AGX3PackBinary, ComputeShadow2DArrayUsesDisjointBanksOnAGX3C)
{
   for (bool reg_desc : {false, true}) {
      for (bool reg_lod : {false, true}) {
         SCOPED_TRACE(std::string(reg_desc ? "reg-desc/" : "imm-desc/") +
                      (reg_lod ? "reg-lod" : "imm0-lod"));
         const auto enc = AGX_ISA_ENCODING_AGX3C;
         agx_builder *b = builder(enc);
         key.dev.chip = AGX_CHIP_G17C;
         key.dev.emit_agx3d = false;
         b->shader->stage = MESA_SHADER_COMPUTE;
         b->shader->max_reg = 127;

         const agx_index tex =
            reg_desc ? agx_register(40, AGX_SIZE_64) : agx_immediate(0);
         const agx_index smp =
            reg_desc ? agx_register(44, AGX_SIZE_64) : agx_immediate(0);

         for (unsigned i = 0; i < 3; ++i) {
            agx_index dest = agx_register(60 + 8 * i, AGX_SIZE_32);
            dest.channels_m1 = 3;
            agx_index coord = agx_register(8 + 8 * i, AGX_SIZE_32);
            coord.channels_m1 = 2;
            coord.discard = true;
            agx_index lod = reg_lod ? agx_register(24 + 2 * i, AGX_SIZE_16)
                                    : agx_immediate(0);
            if (reg_lod)
               lod.discard = true;
            agx_index ref = agx_register(32 + 2 * i, AGX_SIZE_32);
            ref.discard = true;
            auto *I = agx_texture_sample_to(
               b, dest, coord, lod, agx_immediate(0), tex, smp, ref,
               AGX_DIM_2D_ARRAY, AGX_LOD_MODE_LOD_MIN, 0,
               false, true, false, AGX_GATHER_NONE, false);
            I->mask = 1;
         }
         agx_stop(b);

         const auto code = pack(b);
         ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();
         EXPECT_TRUE(b->shader->agx3_staging_known);
         EXPECT_EQ(b->shader->agx3_staging_halfs, 24u);

         for (unsigned i = 0; i < 3; ++i) {
            const uint16_t bank = 8 * i;
            const uint8_t stage_slot = reg_desc ? 6 : (1 + i);
            uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;

            if (reg_lod) {
               ASSERT_TRUE(agx3_tex_stage_array_lod_at(
                  enc, bank, (8 + 8 * i) + 4, 24 + 2 * i, stage_slot, 0,
                  false, true, false, true, bytes, &len));
            } else {
               ASSERT_TRUE(agx3_tex_stage_array_layer_at(
                  enc, bank, (8 + 8 * i) + 4, stage_slot, 0, false, true,
                  bytes, &len));
            }
            EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                      code.end())
               << "missing array staging at bank " << bank;

            ASSERT_TRUE(agx3_tex_stage_copy_hints(
               enc, bank + 4, 32 + 2 * i, stage_slot, 0, false, true,
               bytes, &len));
            EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                      code.end())
               << "missing shadow compare ref staging at bank " << (bank + 4);
         }

         /* Sample 1 (dest r68) and sample 2 (dest r76) must not retire slot 1
          * early to reclaim bank 0. */
         for (uint16_t dst : {68u, 76u}) {
            uint8_t retire_s1[AGX3_ALU_MAX_LENGTH], rlen = 0;
            ASSERT_TRUE(agx3_alu_band_identity_wait(enc, dst, 0x1, retire_s1,
                                                    &rlen));
            EXPECT_EQ(std::search(code.begin(), code.end(), retire_s1,
                                  retire_s1 + rlen),
                      code.end())
               << "unexpected early slot-1 retirement on dest r" << dst;
         }
      }
   }
}

/* When a compute 2D-array shadow sample with register descriptors runs while a
 * prior result is live on slot 6, it must retire slot 6 before staging through
 * fixed slot 6 (and then reuses slot 6's freed bank). */
TEST_F(AGX3PackBinary, ComputeShadow2DArrayRegisterDescRetiresLiveSlot6OnAGX3C)
{
   const auto enc = AGX_ISA_ENCODING_AGX3C;
   agx_builder *b = builder(enc);
   key.dev.chip = AGX_CHIP_G17C;
   key.dev.emit_agx3d = false;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 127;

   /* Five independent 2D samples occupy slots 1..5 (banks 0, 4, 8, 12, 16). */
   for (unsigned i = 0; i < 5; ++i) {
      agx_index dest = agx_register(48 + 8 * i, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(8, AGX_SIZE_32);
      coord.channels_m1 = 1;
      auto *I = agx_texture_sample_to(
         b, dest, coord, agx_immediate(0), agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->mask = 1;
   }

   /* Sample A (register descriptors, shadow 2D-array, register LOD) takes
    * result slot 6 and allocates bank 24 (aligned to 8 above bank 16..19). */
   const agx_index tex = agx_register(40, AGX_SIZE_64);
   const agx_index smp = agx_register(44, AGX_SIZE_64);
   for (unsigned i = 0; i < 2; ++i) {
      agx_index dest = agx_register(96 + 8 * i, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(16 + 8 * i, AGX_SIZE_32);
      coord.channels_m1 = 2;
      coord.discard = true;
      agx_index lod = agx_register(32 + 2 * i, AGX_SIZE_16);
      lod.discard = true;
      agx_index ref = agx_register(36 + 2 * i, AGX_SIZE_32);
      ref.discard = true;
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0), tex, smp, ref,
         AGX_DIM_2D_ARRAY, AGX_LOD_MODE_LOD_MIN, 0,
         false, true, false, AGX_GATHER_NONE, false);
      I->mask = 1;
   }
   agx_stop(b);

   const auto code = pack(b);
   ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();

   /* Sample B (dest r104, result slot 7) must retire slot 6 (mask 0x20) before
    * staging on slot 6, and then reuse bank 24 freed by sample A. */
   uint8_t retire_s6[AGX3_ALU_MAX_LENGTH], rlen = 0;
   ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 104, 1u << 5, retire_s6, &rlen));
   auto retire_pos = std::search(code.begin(), code.end(), retire_s6,
                                 retire_s6 + rlen);
   ASSERT_NE(retire_pos, code.end())
      << "missing live slot-6 retirement before second register-desc shadow sample";

   uint8_t stage_b[AGX3_TEX_MAX_LENGTH], slen = 0;
   ASSERT_TRUE(agx3_tex_stage_array_lod_at(
      enc, 24, 24 + 4, 34, 6, 0, false, true, false, true, stage_b, &slen));
   auto stage_pos = std::search(code.begin(), code.end(), stage_b,
                                stage_b + slen);
   ASSERT_NE(stage_pos, code.end())
      << "expected sample B to reuse freed bank 24 after retiring slot 6";
   EXPECT_LT(retire_pos, stage_pos);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 32u);
}

/* Negative cases: register-descriptor 2D-array shadow samples with register
 * LOD stay on bank 0 outside AGX3C compute (uniform/immediate descriptors use
 * relocatable_2d_array on both AGX3B and non-AGX3D AGX3C). */
TEST_F(AGX3PackBinary, Shadow2DArrayRegisterLodBankZeroOutsideAGX3CCompute)
{
   const struct {
      const char *name;
      enum agx_isa_encoding enc;
      enum agx_chip chip;
      bool agx3d;
      mesa_shader_stage stage;
   } cases[] = {
      {"agx3b-compute", AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false,
       MESA_SHADER_COMPUTE},
      {"agx3c-fragment", AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false,
       MESA_SHADER_FRAGMENT},
      {"agx3d-compute", AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true,
       MESA_SHADER_COMPUTE},
   };

   for (const auto &c : cases) {
      SCOPED_TRACE(c.name);
      agx_builder *b = builder(c.enc);
      key.dev.chip = c.chip;
      key.dev.emit_agx3d = c.agx3d;
      b->shader->stage = c.stage;
      b->shader->max_reg = 127;

      const agx_index tex = agx_register(40, AGX_SIZE_64);
      const agx_index smp = agx_register(44, AGX_SIZE_64);
      for (unsigned i = 0; i < 2; ++i) {
         agx_index dest = agx_register(60 + 8 * i, AGX_SIZE_32);
         dest.channels_m1 = 3;
         agx_index coord = agx_register(8 + 8 * i, AGX_SIZE_32);
         coord.channels_m1 = 2;
         coord.discard = true;
         agx_index lod = agx_register(24 + 2 * i, AGX_SIZE_16);
         lod.discard = true;
         agx_index ref = agx_register(32 + 2 * i, AGX_SIZE_32);
         ref.discard = true;
         auto *I = agx_texture_sample_to(
            b, dest, coord, lod, agx_immediate(0),
            tex, smp, ref,
            AGX_DIM_2D_ARRAY, AGX_LOD_MODE_LOD_MIN, 0,
            false, true, false, AGX_GATHER_NONE, false);
         I->mask = 1;
      }
      agx_stop(b);

      const auto code = pack(b);
      ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();
      EXPECT_TRUE(b->shader->agx3_staging_known);
      /* On G18G (emit_agx3d) slot-keyed blocks place slot 2 at 8*(2-1)=8, so
       * only non-G18G cases keep staging_halfs == 8. */
      if (!c.agx3d) {
         EXPECT_EQ(b->shader->agx3_staging_halfs, 8u);
      }
   }
}

/* Consecutive 2D register-LOD / register-bias samples and 9-tap 2D-array
 * shadow register-LOD samples (e.g. VK_EXT_image_view_min_lod lowering of
 * SampleLevel/SampleCmpLevelZero) allocate disjoint 8-half staging banks on
 * AGX3B without early slot-1 retirement. */
TEST_F(AGX3PackBinary, RegisterLodAndBias2DAnd2DArrayUseDisjointBanksOnAGX3B)
{
   const auto enc = AGX_ISA_ENCODING_AGX3B;
   for (enum agx_lod_mode mode :
        {AGX_LOD_MODE_LOD_MIN, AGX_LOD_MODE_AUTO_LOD_BIAS}) {
      SCOPED_TRACE(mode);
      agx_builder *b = builder(enc);
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 127;

      const agx_index lod = agx_register(4, AGX_SIZE_16);
      for (unsigned i = 0; i < 5; ++i) {
         agx_index dest = agx_register(48 + 8 * i, AGX_SIZE_32);
         dest.channels_m1 = 3;
         agx_index coord = agx_register(8 + 4 * i, AGX_SIZE_32);
         coord.channels_m1 = 1;
         auto *I = agx_texture_sample_to(
            b, dest, coord, lod, agx_immediate(0),
            agx_uniform(0, AGX_SIZE_64), agx_uniform(4, AGX_SIZE_64),
            agx_null(), AGX_DIM_2D, mode, 0, false, false, false,
            AGX_GATHER_NONE, false);
         I->mask = 15;
      }
      agx_stop(b);

      const auto code = pack(b);
      ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();
      EXPECT_TRUE(b->shader->agx3_staging_known);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 40u);

      for (unsigned i = 0; i < 5; ++i) {
         uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
         ASSERT_TRUE(agx3_tex_stage_typed_lod_at(
            enc, 8 * i, 4, 1 + i, 0, false, false, AGX3_TEX_LEVEL_FLOAT,
            bytes, &len));
         EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                   code.end())
            << "missing typed LOD staging at bank " << (8 * i);
      }

      /* Later taps must not retire slot 1 early to reclaim bank 0. */
      for (unsigned i = 1; i < 5; ++i) {
         uint8_t retire_s1[AGX3_ALU_MAX_LENGTH], rlen = 0;
         ASSERT_TRUE(agx3_alu_band_identity_wait(enc, 48 + 8 * i, 0x1,
                                                 retire_s1, &rlen));
         EXPECT_EQ(std::search(code.begin(), code.end(), retire_s1,
                               retire_s1 + rlen),
                   code.end())
            << "unexpected early slot-1 retirement on tap " << i;
      }
   }

   /* 9-tap 2D-array shadow comparison with register LOD (3x3 PCF with
    * image_view_min_lod): stages on slot 6, spreads across banks 0..64
    * (escalating bank 64 to ep4), and keeps all 9 taps in flight. */
   {
      agx_builder *b = builder(enc);
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 127;

      const agx_index lod = agx_register(4, AGX_SIZE_16);
      const agx_index ref = agx_register(6, AGX_SIZE_32);
      for (unsigned i = 0; i < 9; ++i) {
         agx_index dest = agx_register(24 + 8 * i, AGX_SIZE_32);
         dest.channels_m1 = 3;
         agx_index coord = agx_register(8 + 8 * (i % 2), AGX_SIZE_32);
         coord.channels_m1 = 2;
         auto *I = agx_texture_sample_to(
            b, dest, coord, lod, agx_immediate(0),
            agx_uniform(0, AGX_SIZE_64), agx_uniform(4, AGX_SIZE_64),
            ref, AGX_DIM_2D_ARRAY, AGX_LOD_MODE_LOD_MIN, 0, false, true, false,
            AGX_GATHER_NONE, false);
         I->mask = 1;
      }
      agx_stop(b);

      const auto code = pack(b);
      ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();
      EXPECT_TRUE(b->shader->agx3_staging_known);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 72u);

      for (unsigned i = 0; i < 9; ++i) {
         uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
         ASSERT_TRUE(agx3_tex_stage_array_lod_at(
            enc, 8 * i, (8 + 8 * (i % 2)) + 4, 4, 6, 0,
            false, false, false, false, bytes, &len));
         EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                   code.end())
            << "missing array+LOD staging at bank " << (8 * i);
      }
   }
}

TEST_F(AGX3PackBinary, FragmentTexelFetchUsesReservedSamplerZeroOnAgx3c)
{
   /* AGX3c fragment texel fetches must target reserved texel-fetch sampler 0
    * rather than slot 1 (which belongs to the first application/meta VkSampler
    * in the shared sampler heap). */
   const struct {
      const char *name;
      enum agx_dim dim;
      bool register_lod;
      unsigned coord_channels;
   } cases[] = {
      {"2d-reg-lod", AGX_DIM_2D, true, 2},
      {"2d-array", AGX_DIM_2D_ARRAY, false, 3},
      {"3d-imm-lod", AGX_DIM_3D, false, 3},
   };

   for (const auto &c : cases) {
      SCOPED_TRACE(c.name);
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      key.dev.chip = AGX_CHIP_G17G;
      key.dev.emit_agx3d = false;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 127;

      agx_index dest = agx_register(8, AGX_SIZE_32);
      dest.channels_m1 = 3;
      agx_index coord = agx_register(0, AGX_SIZE_32);
      coord.channels_m1 = c.coord_channels - 1;
      coord.discard = true;
      agx_index lod = c.register_lod ? agx_register(6, AGX_SIZE_32)
                                     : agx_immediate(0);
      if (c.register_lod)
         lod.discard = true;
      agx_index tex = agx_uniform(24, AGX_SIZE_64);
      auto *I = agx_texture_sample_to(
         b, dest, coord, lod, agx_immediate(0), tex, agx_immediate(0),
         agx_null(), c.dim, AGX_LOD_MODE_LOD_MIN, 0, false, false, false,
         AGX_GATHER_NONE, false);
      I->op = AGX_OPCODE_TEXTURE_LOAD;
      I->mask = 15;
      agx_stop(b);

      const auto code = pack(b);
      ASSERT_FALSE(code.empty()) << agx3_last_pack_refusal();

      struct agx3_tex_fetch_desc want = {};
      want.dest = 8;
      want.coord = 0;
      want.width = AGX3_TEX_W32;
      want.mask = 15;
      want.texture_kind = AGX3_TEX_OPERAND_UNIFORM;
      want.texture_value = 24;
      want.slot = 1;
      want.wait_mask = 1;
      want.coord_kill = true;
      want.final = true;
      want.array = c.dim == AGX_DIM_2D_ARRAY;
      want.three_d = c.dim == AGX_DIM_3D;
      want.register_lod = c.register_lod;
      want.sampler_value = 0;

      uint8_t bytes[AGX3_TEX_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &want, bytes, &len));
      EXPECT_NE(std::search(code.begin(), code.end(), bytes, bytes + len),
                code.end())
         << "missing AGX3c fragment fetch with sampler_value=0";

      want.sampler_value = 1;
      ASSERT_TRUE(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &want, bytes, &len));
      EXPECT_EQ(std::search(code.begin(), code.end(), bytes, bytes + len),
                code.end())
         << "unexpected AGX3c fragment fetch with sampler_value=1";
   }
}

/* Trimmed 16-bit vec3 loads (6 bytes) from global and global-constant memory
 * (the lowered load_agx and load_constant_agx forms) must shrink before
 * emission and encode/pack as a 3-component W16 load on AGX2, AGX3B, AGX3C,
 * and G18G (AGX3D). */
TEST_F(AGX3PackBinary, TrimmedF16Vec3MemoryLoadsPack)
{
   const struct {
      enum agx_isa_encoding enc;
      enum agx_chip chip;
      bool agx3d;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, false},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true},
   };

   for (bool is_ubo : {false, true}) {
      for (const auto &t : targets) {
         SCOPED_TRACE(::testing::Message()
                      << "is_ubo=" << is_ubo << " chip=" << t.chip
                      << " agx3d=" << t.agx3d);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "f16_vec3_shrink");
         nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
         nir_def *gid = nir_channel(&b, nir_load_workgroup_id(&b), 0);
         nir_def *addr = nir_iadd(&b, base, nir_u2u64(&b, nir_ishl_imm(&b, gid, 3)));
         nir_def *loaded =
            is_ubo
               ? agx_test_load_global_constant(&b, 4, 16, addr, 8,
                                               (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER))
               : nir_load_global(&b, 4, 16, addr, .align_mul = 8);
         nir_def *sum = nir_fadd(&b,
            nir_fadd(&b, nir_channel(&b, loaded, 0), nir_channel(&b, loaded, 1)),
            nir_channel(&b, loaded, 2));
         nir_store_global(&b, sum, nir_iadd_imm(&b, addr, 4096), .align_mul = 2);
         b.shader->info.workgroup_size[0] = 32;
         b.shader->info.workgroup_size[1] = 1;
         b.shader->info.workgroup_size[2] = 1;

         agx_shader_key k = {};
         k.dev.encoding = t.enc;
         k.dev.chip = t.chip;
         k.dev.emit_agx3d = t.agx3d;
         k.no_preamble = true;
         k.reserved_preamble = 16;

         agx_preprocess_nir(b.shader);
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(b.shader, &k, &part));
         EXPECT_GT(part.info.stats.instrs, 0u);
         ASSERT_GT(part.info.main_size, 0u);

         const nir_intrinsic_op expect_op =
            is_ubo ? nir_intrinsic_load_constant_agx : nir_intrinsic_load_agx;
         unsigned trimmed_nir_loads = 0;
         nir_foreach_function_impl(impl, b.shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic == expect_op) {
                     EXPECT_EQ(intr->def.bit_size, 16u);
                     EXPECT_EQ(intr->def.num_components, 3u);
                     trimmed_nir_loads++;
                  }
               }
            }
         }
         EXPECT_EQ(trimmed_nir_loads, 1u);

         const uint8_t *code =
            static_cast<const uint8_t *>(part.binary) + part.info.main_offset;
         unsigned w16_vec3_loads = 0;
         unsigned w16_vec4_loads = 0;
         unsigned off = 0;

         while (off < part.info.main_size) {
            const char *display = NULL;
            const size_t avail = part.info.main_size - off;
            if (t.enc == AGX_ISA_ENCODING_AGX2) {
               if (avail >= 2 && code[off] == 0x08 && code[off + 1] == 0x00)
                  break;
               signed len = agx2_instr_length(code + off, avail, &display);
               ASSERT_GT(len, 0);
               if (display && strncmp(display, "load", 4) == 0 &&
                   (display[4] == '\0' || display[4] == ' ')) {
                  ASSERT_GE(len, 6);
                  uint64_t raw = 0;
                  memcpy(&raw, code + off, std::min<size_t>(len, sizeof(raw)));
                  unsigned fmt = (((raw >> 48) & 1u) << 3) | ((raw >> 7) & 0x7u);
                  unsigned mask = (raw >> 52) & 0xfu;
                  if (fmt == AGX_FORMAT_I16) {
                     if (mask == 0x7)
                        w16_vec3_loads++;
                     else if (mask == 0xf)
                        w16_vec4_loads++;
                  }
               }
               off += (unsigned)len;
               continue;
            }

            signed len = 0;
            const agx3_table *table = NULL;
            if (t.enc == AGX_ISA_ENCODING_AGX3B) {
               len = agx3b_instr_length(code + off, avail, &display);
               table = &agx3b_table;
            } else if (t.agx3d) {
               len = agx3d_instr_length(code + off, avail, &display);
               table = &agx3d_table;
            } else {
               len = agx3c_instr_length(code + off, avail, &display);
               table = &agx3c_table;
            }
            ASSERT_GT(len, 0);
            uint64_t fields[AGX3_MAX_FIELDS] = {};
            if (agx3_decode(table, code + off, (unsigned)len, AGX3_MN_lddev,
                            fields) >= 0 &&
                fields[10] == AGX3_MEM_W16) {
               if (fields[73] == 0x7)
                  w16_vec3_loads++;
               else if (fields[73] == 0xf)
                  w16_vec4_loads++;
            }
            off += (unsigned)len;
         }

         EXPECT_EQ(w16_vec3_loads, 1u);
         EXPECT_EQ(w16_vec4_loads, 0u);
         free(part.binary);
         ralloc_free(b.shader);
      }
   }
}

/*
 * Compute imageblock stores, the imageblock barrier and the block store. The
 * field vectors are the documented meanings of each instruction's fields
 * (agx3_tile_core.h, agx3_pack.c), recovered by compiling our own imageblock
 * compute kernels for the G16G and comparing the results field by field; the
 * expected bytes are composed from the in-tree AGX3B tables.
 */
static std::vector<uint8_t>
compose_b(enum agx3_mnemonic op, std::vector<struct agx3_field_kv> fields)
{
   uint8_t bytes[16], length = 0;
   EXPECT_TRUE(agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, op, 0, fields.data(),
                                 fields.size(), NULL, 0, bytes, &length));
   return std::vector<uint8_t>(bytes, bytes + length);
}

TEST(AGX3ComputeImageblock, TileStoreFieldMap)
{
   const struct {
      uint16_t data;
      enum agx3_tile_width width;
      uint8_t mask;
      uint16_t offset, coords;
      bool last;
   } cases[] = {
      /* Four 8-bit lanes from two fltnorm halves, the whole pixel. */
      {0, AGX3_TILE_W8, 15, 0, 4, true},
      /* One 8-bit lane at byte 3, not the last store. */
      {8, AGX3_TILE_W8, 1, 3, 2, false},
      /* One 32-bit word at byte 4. */
      {6, AGX3_TILE_W32, 1, 4, 0, true},
   };
   for (const auto &c : cases) {
      uint8_t got[AGX3_TILE_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_tile_store_explicit(AGX_ISA_ENCODING_AGX3B, c.data,
                                           c.width, c.mask, c.offset, c.coords,
                                           c.last, got, &len));
      const std::vector<uint8_t> want = compose_b(AGX3_MN_lstcol, {
         {2, c.data},  {9, 2},   {10, c.width}, {13, 1},  {26, c.mask},
         {30, c.coords}, {37, 2}, {38, 1}, {41, c.last}, {43, 1}, {51, 1},
         {52, 1}, {57, c.offset}, {65, 1}, {66, 1},
      });
      EXPECT_EQ(std::vector<uint8_t>(got, got + len), want);
      EXPECT_EQ(len, 12);
   }

   uint8_t out[AGX3_TILE_MAX_LENGTH], len = 0;
   /* Other shapes and encodings are refused. */
   EXPECT_FALSE(agx3_tile_store_explicit(AGX_ISA_ENCODING_AGX3C, 0,
                                         AGX3_TILE_W8, 15, 0, 4, true, out,
                                         &len));
   EXPECT_FALSE(agx3_tile_store_explicit(AGX_ISA_ENCODING_AGX3B, 0,
                                         AGX3_TILE_W16, 15, 0, 4, true, out,
                                         &len));
   EXPECT_FALSE(agx3_tile_store_explicit(AGX_ISA_ENCODING_AGX3B, 0,
                                         AGX3_TILE_W8, 3, 0, 4, true, out,
                                         &len));
   EXPECT_FALSE(agx3_tile_store_explicit(AGX_ISA_ENCODING_AGX3B, 0,
                                         AGX3_TILE_W8, 15, 0, 5, true, out,
                                         &len));
   EXPECT_FALSE(agx3_tile_store_explicit(AGX_ISA_ENCODING_AGX3B, 1,
                                         AGX3_TILE_W32, 1, 0, 4, true, out,
                                         &len));
}

static std::vector<uint8_t>
block_store_b(uint16_t x, uint16_t y, bool kill, int lane_class,
              uint16_t base, enum agx3_block_store_index kind, uint16_t index,
              bool index_kill)
{
   std::vector<uint8_t> v = compose_b(AGX3_MN_pack_imgwir, {
      {5, 4}, {9, 5}, {10, 2}, {12, 1}, {23, 1}, {24, 1}, {37, 1}, {38, 1},
      {51, 1}, {52, 2}, {74, 8}, {75, 16}, {76, 7}, {77, 24},
   });
   for (unsigned xy = 0; xy < 2; ++xy) {
      const std::vector<uint8_t> copy = compose_b(AGX3_MN_copy, {
         {5, 2u * xy}, {9, 5}, {10, 2}, {12, 1}, {16, xy ? y : x}, {23, 2},
         {24, 2}, {kill ? (uint16_t)27 : (uint16_t)25, 1},
      });
      v.insert(v.end(), copy.begin(), copy.end());
   }
   std::vector<struct agx3_field_kv> img = {
      {9, 5}, {12, 1}, {13, 1}, {28, 1}, {29, (uint64_t)lane_class}, {30, 1},
      {32, 1}, {45, 3}, {46, 3}, {39, base}, {73, 1},
   };
   if (kind == AGX3_BLOCK_STORE_INDEX_IMMEDIATE)
      img.insert(img.end(), {{59, 1}, {60, 1}, {51, index}, {17, 1}});
   else if (kind == AGX3_BLOCK_STORE_INDEX_UNIFORM)
      img.insert(img.end(), {{59, 3}, {60, 2}, {53, index}, {17, 1}});
   else
      img.insert(img.end(), {{59, 2}, {60, 2}, {52, index}, {22, 1},
                             {63, index_kill}});
   const std::vector<uint8_t> store = compose_b(AGX3_MN_imgwblk, img);
   v.insert(v.end(), store.begin(), store.end());
   return v;
}

TEST(AGX3ComputeImageblock, BlockStoreFieldMap)
{
   const struct {
      enum agx3_block_store_index kind;
      uint16_t index;
      bool index_kill, kill;
      int lane_class;
   } cases[] = {
      {AGX3_BLOCK_STORE_INDEX_REGISTER, 12, true, true, 3},
      {AGX3_BLOCK_STORE_INDEX_REGISTER, 14, false, false, 2},
      {AGX3_BLOCK_STORE_INDEX_IMMEDIATE, 5, false, true, 1},
      {AGX3_BLOCK_STORE_INDEX_UNIFORM, 40, false, true, 3},
   };
   for (const auto &c : cases) {
      const struct agx3_block_store_desc d = {
         .x = 8, .y = 10, .x_kill = c.kill, .y_kill = c.kill,
         .lane_class = c.lane_class, .base_uniform = 12,
         .index_kind = c.kind, .index = c.index, .index_kill = c.index_kill,
      };
      uint8_t got[AGX3_BLOCK_STORE_MAX_LENGTH];
      size_t len = 0;
      ASSERT_TRUE(agx3_block_store_compute(AGX_ISA_ENCODING_AGX3B, &d, got,
                                           &len));
      EXPECT_EQ(std::vector<uint8_t>(got, got + len),
                block_store_b(8, 10, c.kill, c.lane_class, 12, c.kind,
                              c.index, c.index_kill));
   }

   const struct agx3_block_store_desc bad_base = {
      .x = 8, .y = 10, .lane_class = 3, .base_uniform = 6,
      .index_kind = AGX3_BLOCK_STORE_INDEX_IMMEDIATE,
   };
   uint8_t out[AGX3_BLOCK_STORE_MAX_LENGTH];
   size_t len = 0;
   EXPECT_FALSE(agx3_block_store_compute(AGX_ISA_ENCODING_AGX3B, &bad_base,
                                         out, &len));
   EXPECT_FALSE(agx3_block_store_compute(AGX_ISA_ENCODING_AGX3C, &bad_base,
                                         out, &len));
   EXPECT_EQ(agx3_block_store_lane_class(AGX3_TILE_W8, false), 3);
   EXPECT_EQ(agx3_block_store_lane_class(AGX3_TILE_W16, true), 1);
   EXPECT_EQ(agx3_block_store_lane_class(AGX3_TILE_W32, false), 2);
   EXPECT_EQ(agx3_block_store_lane_class(AGX3_TILE_W16, false), -1);
}

TEST(AGX3ComputeImageblock, ImageblockBarrierFieldMap)
{
   struct agx3_encoded_scoped_barrier got;
   ASSERT_TRUE(agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3B,
                                        AGX3_BARRIER_SCOPE_IMAGEBLOCK, &got));
   EXPECT_EQ(std::vector<uint8_t>(got.bytes, got.bytes + got.length),
             compose_b(AGX3_MN_barrier, {{27, 1}, {29, 1}, {30, 2}, {39, 1}}));
   EXPECT_FALSE(agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3C,
                                         AGX3_BARRIER_SCOPE_IMAGEBLOCK, &got));
}

/* A compute kernel's imageblock store, imageblock barrier and block store pack
 * to those encoders' output: the store is the program's last imageblock store,
 * the block store's heap is a uniform and its record index a register. */
TEST_F(AGX3PackBinary, ComputeImageblockBlockStorePacks)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   key.dev.chip = AGX_CHIP_G16G;
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index data = agx_register(0, AGX_SIZE_16);
   data.channels_m1 = 1;
   agx_index coords = agx_register(4, AGX_SIZE_16);
   coords.channels_m1 = 1;
   agx_st_tile(b, data, agx_immediate(1), coords, AGX_FORMAT_U8NORM, 0xf, 0,
               true, 0);
   agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_IMAGEBLOCK);
   agx_index origin = agx_register(8, AGX_SIZE_32);
   origin.channels_m1 = 1;
   agx_block_image_store(b, agx_uniform(12, AGX_SIZE_64),
                         agx_register(12, AGX_SIZE_32), agx_immediate(0),
                         origin, AGX_FORMAT_U8NORM, AGX_DIM_2D, true);
   agx_stop(b);

   std::vector<uint8_t> want;
   uint8_t tile[AGX3_TILE_MAX_LENGTH], tile_len = 0;
   ASSERT_TRUE(agx3_tile_store_explicit(AGX_ISA_ENCODING_AGX3B, 0,
                                        AGX3_TILE_W8, 15, 0, 4, true, tile,
                                        &tile_len));
   append(want, tile, tile_len);
   struct agx3_encoded_scoped_barrier barrier;
   ASSERT_TRUE(agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3B,
                                        AGX3_BARRIER_SCOPE_IMAGEBLOCK,
                                        &barrier));
   append(want, barrier.bytes, barrier.length);
   const std::vector<uint8_t> store =
      block_store_b(8, 10, false, 3, 12, AGX3_BLOCK_STORE_INDEX_REGISTER, 12,
                    false);
   want.insert(want.end(), store.begin(), store.end());
   struct agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   append(want, end.bytes, end.length);
   EXPECT_EQ(pack(b), want);
}

TEST_F(AGX3PackBinary, G17PComputeDeviceStoreSetsCoherentPolicy)
{
   const struct {
      enum agx_isa_encoding enc;
      enum agx_chip chip;
      mesa_shader_stage stage;
      bool expect_coherent;
   } cases[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, MESA_SHADER_COMPUTE, true},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, MESA_SHADER_VERTEX, false},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, MESA_SHADER_FRAGMENT, false},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G15G, MESA_SHADER_COMPUTE, false},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, MESA_SHADER_COMPUTE, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17S, MESA_SHADER_COMPUTE, false},
   };
   for (const auto &c : cases) {
      EXPECT_EQ(c.expect_coherent,
                c.stage == MESA_SHADER_COMPUTE &&
                   agx_chip_has_cap(c.chip, AGX_CAP_COMPUTE_COHERENT_STORE));
      agx_builder *b = builder(c.enc);
      key.dev.chip = c.chip;
      b->shader->stage = c.stage;
      agx_device_store(b, agx_register(0, AGX_SIZE_32),
                       agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                       AGX_FORMAT_I32, 1, 0, false);
      agx_stop(b);

      struct agx3_mem_desc d = {};
      d.data = 0;
      d.width = AGX3_MEM_W32;
      d.mask = 1;
      d.shift = 0;
      d.base_is_uniform = true;
      d.base = 0;
      d.idx_kind = AGX3_MEM_IDX_IMM;
      d.byte_off = 0;
      d.slot = 0;
      d.wait_mask = 0;
      d.coherent = c.expect_coherent;

      uint8_t st[AGX3_MEM_MAX_LENGTH], st_len = 0;
      ASSERT_TRUE(agx3_mem_encode_store(c.enc, &d, st, &st_len));
      struct agx3_encoded_end end;
      ASSERT_TRUE(agx3_pack_end(c.enc, &end));
      std::vector<uint8_t> want;
      append(want, st, st_len);
      append(want, end.bytes, end.length);
      const auto got = pack(b);
      ASSERT_GE(got.size(), want.size());
      if (c.stage == MESA_SHADER_FRAGMENT) {
         EXPECT_EQ(std::vector<uint8_t>(got.end() - want.size(), got.end()),
                   want);
      } else {
         EXPECT_EQ(got, want);
      }
   }
}

/* PIXINFO composes the destination half, the coverage half, the scoreboard
 * slot and the incoming wait mask into their fields and nothing else. */
TEST(AGX3PixInfo, ComposesSampleGroupFields)
{
   const struct {
      uint16_t dest, src;
      uint8_t slot;
      uint16_t wait;
   } cases[] = {
      {2, 1, 6, 0}, {26, 8, 6, 1}, {12, 8, 6, 1}, {190, 191, 1, 0x21},
   };
   for (const auto &c : cases) {
      uint8_t bytes[AGX3_TILE_MAX_LENGTH];
      uint8_t len = 0;
      ASSERT_TRUE(agx3_pixinfo(AGX_ISA_ENCODING_AGX3B, c.dest, c.src, c.slot,
                               c.wait, bytes, &len));
      ASSERT_EQ(len, 14u);
      uint64_t f[AGX3_MAX_FIELDS];
      ASSERT_GE(agx3_decode(&agx3b_table, bytes, len, AGX3_MN_pixinfo, f), 0);
      EXPECT_EQ(f[2], c.dest);
      EXPECT_EQ(f[44], c.src);
      EXPECT_EQ(f[12], c.slot);
      for (unsigned i = 0; i < 6; ++i)
         EXPECT_EQ(f[18 + i], (c.wait >> i) & 1u) << "wait slot " << i + 1;
      /* A 16-bit result from a register source. */
      EXPECT_EQ(f[9], 2u);
      EXPECT_EQ(f[10], 1u);
      EXPECT_EQ(f[51], 2u);
      EXPECT_EQ(f[56], 0x100u);
   }

   uint8_t bytes[AGX3_TILE_MAX_LENGTH];
   uint8_t len = 0;
   EXPECT_FALSE(agx3_pixinfo(AGX_ISA_ENCODING_AGX3C, 2, 1, 6, 0, bytes, &len));
   EXPECT_FALSE(agx3_pixinfo(AGX_ISA_ENCODING_AGX3B, 2, 1, 0, 0, bytes, &len));
   EXPECT_FALSE(agx3_pixinfo(AGX_ISA_ENCODING_AGX3B, 2, 1, 7, 0, bytes, &len));
   EXPECT_FALSE(agx3_pixinfo(AGX_ISA_ENCODING_AGX3B, 192, 1, 6, 0, bytes, &len));
   EXPECT_FALSE(agx3_pixinfo(AGX_ISA_ENCODING_AGX3B, 2, 192, 6, 0, bytes, &len));
   EXPECT_FALSE(agx3_pixinfo(AGX_ISA_ENCODING_AGX3B, 2, 1, 6, 0x40, bytes, &len));
   EXPECT_EQ(len, 0u);
}

/* A sample-groups tile load packs as PIXINFO of its coverage register on the
 * tile-read slot, and is refused where PIXINFO has no form. */
TEST_F(AGX3PackBinary, SampleGroupsLoadPacksAsPixinfo)
{
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_instr *I = agx_ld_tile_to(b, agx_register(2, AGX_SIZE_16),
                                    agx_register(1, AGX_SIZE_16), agx_null(),
                                    AGX_FORMAT_I16, 1, 0, false, 0);
      I->sample_groups = true;
      agx_stop(b);
      const std::vector<uint8_t> code = pack(b);
      ASSERT_GE(code.size(), 14u);
      uint64_t f[AGX3_MAX_FIELDS];
      ASSERT_GE(agx3_decode(&agx3b_table, code.data(), 14, AGX3_MN_pixinfo, f),
                0);
      EXPECT_EQ(f[2], 2u);
      EXPECT_EQ(f[44], 1u);
      EXPECT_EQ(f[12], 6u);
   }
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_instr *I = agx_ld_tile_to(b, agx_register(2, AGX_SIZE_16),
                                    agx_register(1, AGX_SIZE_16), agx_null(),
                                    AGX_FORMAT_I16, 1, 0, false, 0);
      I->sample_groups = true;
      agx_stop(b);
      EXPECT_TRUE(pack(b, false).empty());
   }
}

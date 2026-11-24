/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "asahi/compiler/agx_compile.h"
#include "compiler/nir/nir_builder.h"
#include "agx_tilebuffer.h"
#include "agx_linker.h"
#include "util/u_debug.h"
extern "C" {
#include "asahi/compiler/agx_nir.h"
#include "asahi/compiler/agx_debug.h"
}


class G17PMSAAEpilog : public ::testing::Test {
 protected:
   G17PMSAAEpilog() { glsl_type_singleton_init_or_ref(); }
   ~G17PMSAAEpilog() { glsl_type_singleton_decref(); }
};

TEST_F(G17PMSAAEpilog, M5SampleAlphaCoveragePrecedesAllColourStores)
{
   for (unsigned samples : {2u, 4u, 8u}) {
      SCOPED_TRACE(samples);
      agx_fs_epilog_key fs = {};
      fs.nr_samples = samples;
      fs.max_sample_size_B = 248;
      fs.max_bytes_per_tile = 65536;
      fs.max_samples_per_tile = 2048;
      fs.link.loc_written = 1;
      fs.link.size_32 = 1;
      fs.link.colour_comps = 15;
      fs.link.sample_shading = true;
      fs.rt_formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
      fs.remap[0] = 0;
      fs.blend.rt[0].colormask = 15;
      fs.blend.rt[0].mode = agx_pack_blend_standard(
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
      fs.blend.alpha_to_coverage = true;
      agx_fs_epilog_apply_device_policy(AGX_CHIP_G17S, &fs);

      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "M5 sample alpha coverage");
      agx_nir_fs_epilog_g17p(&b, &fs);
      agx_preprocess_nir(b.shader);
      nir_validate_shader(b.shader, "M5 sample alpha coverage ordering");

      unsigned updates = 0, stores = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (in->intrinsic == nir_intrinsic_sample_mask_agx) {
               EXPECT_EQ(stores, 0u);
               EXPECT_EQ(block->cf_node.parent, &b.impl->cf_node);
               updates++;
            }
            if (in->intrinsic == nir_intrinsic_store_local_pixel_agx)
               stores++;
         }
      }
      EXPECT_EQ(updates, 1u);
      EXPECT_EQ(stores, samples);
      ralloc_free(b.shader);
   }
}

/* Two testing updates lose depth samples and can hang G17P, so an epilog must
 * not update coverage again after a main part that applied alpha-to-coverage. */
TEST_F(G17PMSAAEpilog, AlphaCoverageInMainLeavesNoUpdateInTheEpilog)
{
   for (bool a2c_in_main : {false, true}) {
      SCOPED_TRACE(a2c_in_main);
      agx_fs_epilog_key fs = {};
      fs.nr_samples = 2;
      fs.max_sample_size_B = 248;
      fs.max_bytes_per_tile = 65536;
      fs.max_samples_per_tile = 2048;
      fs.link.loc_written = 1;
      fs.link.size_32 = 1;
      fs.link.colour_comps = 15;
      fs.rt_formats[0] = PIPE_FORMAT_R8G8B8A8_UNORM;
      fs.remap[0] = 0;
      fs.blend.rt[0].colormask = 15;
      fs.blend.rt[0].mode = agx_pack_blend_standard(
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
      fs.blend.alpha_to_coverage = true;
      fs.a2c_in_main = a2c_in_main;
      agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &fs);

      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P alpha coverage");
      agx_nir_fs_epilog_g17p(&b, &fs);
      nir_validate_shader(b.shader, "G17P alpha coverage epilog");

      unsigned updates = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            updates += instr->type == nir_instr_type_intrinsic &&
                       (nir_instr_as_intrinsic(instr)->intrinsic ==
                           nir_intrinsic_sample_mask_agx ||
                        nir_instr_as_intrinsic(instr)->intrinsic ==
                           nir_intrinsic_demote_samples);
         }
      }
      EXPECT_EQ(updates != 0, !a2c_in_main);
      ralloc_free(b.shader);
   }
}

TEST_F(G17PMSAAEpilog, ForceEarlyAppliesApiMaskBeforeSampleCounting)
{
   for (enum agx_chip chip : {AGX_CHIP_G13G, AGX_CHIP_G17P, AGX_CHIP_G17S}) {
      for (unsigned samples : {2u, 4u, 8u}) {
         for (bool early : {false, true}) {
            for (bool sample_shading : {false, true}) {
               for (unsigned mask : {0u, 0xaau, 0xffu}) {
                  SCOPED_TRACE(::testing::Message() << chip << ", " << samples
                     << " samples, early=" << early << ", shading="
                     << sample_shading << ", mask=" << mask);
                  nir_builder b = nir_builder_init_simple_shader(
                     MESA_SHADER_FRAGMENT, &agx_nir_options, "forced early API mask");
                  b.shader->info.fs.early_fragment_tests = early;
                  b.shader->info.fs.uses_sample_shading = sample_shading;
                  b.shader->info.fs.uses_discard = true;
                  nir_intrinsic_instr *first = nir_sample_mask_agx(
                     &b, nir_imm_intN_t(&b, 0xff, 16),
                     nir_imm_intN_t(&b, 0xff, 16));
                  nir_intrinsic_instr *last = nir_sample_mask_agx(
                     &b, nir_imm_intN_t(&b, 0xff, 16),
                     nir_imm_intN_t(&b, 0, 16));
                  const bool changed = early &&
                     (chip == AGX_CHIP_G17P ||
                      (chip == AGX_CHIP_G17S && !sample_shading));
                  EXPECT_EQ(agx_nir_apply_api_mask_to_force_early(b.shader, chip),
                            changed);
                  unsigned api_loads = 0;
                  nir_foreach_block(block, b.impl) {
                     nir_foreach_instr_safe(instr, block) {
                        if (instr->type != nir_instr_type_intrinsic)
                           continue;
                        nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
                        if (in->intrinsic != nir_intrinsic_load_api_sample_mask_agx)
                           continue;
                        api_loads++;
                        b.cursor = nir_before_instr(instr);
                        nir_def_replace(&in->def, nir_imm_intN_t(
                           &b, mask & BITFIELD_MASK(samples), in->def.bit_size));
                     }
                  }
                  EXPECT_EQ(api_loads, changed ? 1u : 0u);
                  nir_opt_constant_folding(b.shader);
                  EXPECT_EQ(nir_src_as_uint(first->src[0]), 0xffu);
                  EXPECT_EQ(nir_src_as_uint(first->src[1]),
                            changed ? (mask & BITFIELD_MASK(samples)) : 0xffu);
                  EXPECT_EQ(nir_src_as_uint(last->src[0]), 0xffu);
                  EXPECT_EQ(nir_src_as_uint(last->src[1]), 0u);
                  nir_validate_shader(b.shader, "forced early API mask");
                  ralloc_free(b.shader);
               }
            }
         }
      }
   }
}

TEST_F(G17PMSAAEpilog, PerSampleColourImportsCompile)
{
   for (unsigned samples : {2u, 4u}) {
      for (unsigned targets : {1u, 4u}) {
         for (bool blend : {false, true}) {
            for (bool depth : {false, true}) {
               for (unsigned mask : {1u, 5u, 15u}) {
                  SCOPED_TRACE(::testing::Message() << samples << " samples, "
                     << targets << " targets, blend=" << blend << " depth="
                     << depth << " mask=" << mask);
                  agx_fs_epilog_key fs = {};
                  fs.nr_samples = samples;
                  fs.max_sample_size_B = AGX_G17P_MAX_BYTES_PER_SAMPLE;
                  fs.max_bytes_per_tile = 32768;
                  fs.max_samples_per_tile = 2048;
                  fs.link.loc_written = (uint8_t)BITFIELD_MASK(targets);
                  fs.link.size_32 = (uint8_t)BITFIELD_MASK(targets);
                  fs.link.colour_comps = BITFIELD_MASK(4 * targets);
                  fs.link.sample_shading = true;
                  fs.link.write_z = depth;
                  fs.defer_api_sample_mask = true;
                  fs.api_sample_mask = mask & BITFIELD_MASK(samples);
                  agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &fs);
                  for (unsigned rt = 0; rt < targets; ++rt) {
                     fs.rt_formats[rt] = PIPE_FORMAT_R8G8B8A8_UNORM;
                     fs.remap[rt] = rt;
                     fs.blend.rt[rt].colormask = 0xf;
                     fs.blend.rt[rt].mode = agx_pack_blend_standard(
                        PIPE_BLEND_ADD, blend ? PIPE_BLENDFACTOR_SRC_ALPHA : PIPE_BLENDFACTOR_ONE,
                        blend ? PIPE_BLENDFACTOR_INV_SRC_ALPHA : PIPE_BLENDFACTOR_ZERO,
                        PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
                  }
                  nir_builder b = nir_builder_init_simple_shader(
                     MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P sample epilog");
                  agx_nir_fs_epilog_g17p(&b, &fs);
                  agx_preprocess_nir(b.shader);
                  agx_nir_lower_g17p_tile_samples(b.shader, samples);
                  agx_shader_key key = {};
                  key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
                  key.dev.chip = AGX_CHIP_G17P;
                  key.no_preamble = true;
                  key.secondary = true;
                  key.fs.nr_samples = samples;
                  for (unsigned rt = 0; rt < targets; ++rt)
                     BITSET_SET(key.g17p_tile_partitions, 4 * rt);
                  agx_shader_part part = {};
                  EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part)) << agx_last_compile_refusal();
                  free(part.binary);
                  ralloc_free(b.shader);
               }
            }
         }
      }
   }
}

TEST_F(G17PMSAAEpilog, PerSampleMainExportsCompile)
{
   for (unsigned samples : {2u, 4u}) {
      for (unsigned targets : {1u, 4u, 8u}) {
         for (unsigned components : {2u, 4u}) {
            SCOPED_TRACE(::testing::Message() << samples << " samples, "
               << targets << " targets, " << components << " components");
            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P sample main");
            b.shader->info.fs.uses_sample_shading = true;
            nir_def *sample = nir_u2f32(&b, nir_load_sample_id(&b));
            nir_def *color = components == 2 ? nir_vec2(&b, sample, sample)
               : nir_vec4(&b, sample, sample, sample, sample);
            for (unsigned rt = 0; rt < targets; ++rt) {
               nir_intrinsic_instr *store =
                  nir_intrinsic_instr_create(b.shader, nir_intrinsic_store_output);
               store->num_components = components;
               store->src[0] = nir_src_for_ssa(color);
               store->src[1] = nir_src_for_ssa(nir_imm_int(&b, 0));
               nir_intrinsic_set_base(store, rt);
               nir_intrinsic_set_write_mask(store, (1u << components) - 1);
               nir_intrinsic_set_src_type(store, nir_type_float32);
               nir_io_semantics sem = {};
               sem.location = FRAG_RESULT_DATA0 + rt;
               sem.num_slots = 1;
               nir_intrinsic_set_io_semantics(store, sem);
               nir_builder_instr_insert(&b, &store->instr);
               b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + rt);
            }
            agx_fs_epilog_link_info link = {};
            agx_nir_lower_fs_output_to_epilog_for_device(b.shader, AGX_CHIP_G17P, &link);
            bool fits = agx_nir_lower_g17p_sample_shading(b.shader, &link, samples);
            if (samples == 4 && targets == 8 && components == 4) {
               EXPECT_FALSE(fits);
               ralloc_free(b.shader);
               continue;
            }
            ASSERT_TRUE(fits);
            EXPECT_TRUE(link.sample_shading);
            EXPECT_FALSE(b.shader->info.fs.uses_sample_shading);
            if (samples == 4 && targets == 8) {
               EXPECT_TRUE(link.packed_colour);
            }
            agx_preprocess_nir(b.shader);
            agx_shader_key key = {};
            key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
            key.dev.chip = AGX_CHIP_G17P;
            key.no_preamble = true;
            key.no_stop = true;
            key.fs.nr_samples = samples;
            agx_shader_part part = {};
            EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part)) << agx_last_compile_refusal();
            free(part.binary);
            ralloc_free(b.shader);
         }
      }
   }
}

/* A sample-shaded main that kills coverage is unrolled per sample, then tests
 * its coverage once: every copy's kill goes into that one update. Kill-only
 * updates of part of a multisampled pixel lose the killed samples' previous
 * contents, and a testing update per copy can stall the GPU beside another
 * context's fragment work. */
TEST_F(G17PMSAAEpilog, PerSampleKillsUseOneCoverageUpdate)
{
   for (unsigned samples : {2u, 4u}) {
      SCOPED_TRACE(samples);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P per-sample kill");
      b.shader->info.fs.uses_sample_shading = true;
      b.shader->info.fs.uses_discard = true;

      /* Odd samples are discarded. */
      nir_def *id = nir_load_sample_id(&b);
      nir_demote_if(&b, nir_ine_imm(&b, nir_iand_imm(&b, id, 1), 0));

      nir_def *value = nir_u2f32(&b, id);
      nir_intrinsic_instr *store =
         nir_intrinsic_instr_create(b.shader, nir_intrinsic_store_output);
      store->num_components = 4;
      store->src[0] = nir_src_for_ssa(nir_vec4(&b, value, value, value, value));
      store->src[1] = nir_src_for_ssa(nir_imm_int(&b, 0));
      nir_intrinsic_set_base(store, 0);
      nir_intrinsic_set_write_mask(store, 15);
      nir_intrinsic_set_src_type(store, nir_type_float32);
      nir_io_semantics sem = {};
      sem.location = FRAG_RESULT_DATA0;
      sem.num_slots = 1;
      nir_intrinsic_set_io_semantics(store, sem);
      nir_builder_instr_insert(&b, &store->instr);
      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0);

      /* As both drivers order it: discards to demotes, the unroll, then the
       * coverage lowering. */
      agx_nir_lower_discard_zs_emit(b.shader);
      agx_fs_epilog_link_info link = {};
      agx_nir_lower_fs_output_to_epilog_for_device(b.shader, AGX_CHIP_G17P, &link);
      ASSERT_TRUE(agx_nir_lower_g17p_sample_shading(b.shader, &link, samples));
      nir_opt_constant_folding(b.shader);
      ASSERT_TRUE(agx_nir_lower_sample_mask_one_update(b.shader));
      nir_validate_shader(b.shader, "G17P per-sample kill coverage");

      unsigned updates = 0, kill_only = 0, demotes = 0;
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
               if (in->intrinsic == nir_intrinsic_demote_samples)
                  demotes++;
               if (in->intrinsic != nir_intrinsic_sample_mask_agx)
                  continue;
               updates++;
               if (nir_src_is_const(in->src[1]) && nir_src_as_uint(in->src[1]) == 0)
                  kill_only++;
            }
         }
      }
      EXPECT_EQ(demotes, 0u);
      EXPECT_EQ(updates, 1u);
      EXPECT_EQ(kill_only, 0u);

      /* The drivers bind or fold the update's test selector; here the
       * fragment always tests. */
      nir_shader_intrinsics_pass(
         b.shader,
         [](nir_builder *nb, nir_intrinsic_instr *in, void *) -> bool {
            if (in->intrinsic != nir_intrinsic_load_shader_part_tests_zs_agx)
               return false;
            nb->cursor = nir_before_instr(&in->instr);
            nir_def_replace(&in->def,
                            nir_imm_intN_t(nb, UINT16_MAX, in->def.bit_size));
            return true;
         },
         nir_metadata_control_flow, NULL);

      agx_preprocess_nir(b.shader);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.no_preamble = true;
      key.no_stop = true;
      key.fs.nr_samples = samples;
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part)) << agx_last_compile_refusal();
      free(part.binary);
      ralloc_free(b.shader);
   }
}

/* M5 unrolls sample-shaded mains like G17P (in its larger register file) and
 * collects the copies' kills into one testing update too, in M5's uniformly
 * selected form: one testing update, a kill-only update at most on the branch
 * where the draw does not test, no demote left, and the result compiles for
 * G17S at 2x and 4x. */
TEST_F(G17PMSAAEpilog, M5PerSampleKillsUseOneTestingUpdate)
{
   /* hk_native_fs_registers() on M5: the whole register file (AGX_NUM_REGS). */
   const unsigned m5_registers = 256;
   for (unsigned samples : {2u, 4u}) {
      SCOPED_TRACE(samples);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "M5 per-sample kill");
      b.shader->info.fs.uses_sample_shading = true;
      b.shader->info.fs.uses_discard = true;

      /* Odd samples are discarded. */
      nir_def *id = nir_load_sample_id(&b);
      nir_demote_if(&b, nir_ine_imm(&b, nir_iand_imm(&b, id, 1), 0));

      nir_def *value = nir_u2f32(&b, id);
      nir_intrinsic_instr *store =
         nir_intrinsic_instr_create(b.shader, nir_intrinsic_store_output);
      store->num_components = 4;
      store->src[0] = nir_src_for_ssa(nir_vec4(&b, value, value, value, value));
      store->src[1] = nir_src_for_ssa(nir_imm_int(&b, 0));
      nir_intrinsic_set_base(store, 0);
      nir_intrinsic_set_write_mask(store, 15);
      nir_intrinsic_set_src_type(store, nir_type_float32);
      nir_io_semantics sem = {};
      sem.location = FRAG_RESULT_DATA0;
      sem.num_slots = 1;
      nir_intrinsic_set_io_semantics(store, sem);
      nir_builder_instr_insert(&b, &store->instr);
      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0);

      /* As Honeykrisp orders it on M5: discards to demotes, the unroll, then
       * the coverage lowering. */
      agx_nir_lower_discard_zs_emit(b.shader);
      agx_fs_epilog_link_info link = {};
      agx_nir_lower_fs_output_to_epilog_for_device(b.shader, AGX_CHIP_G17S, &link);
      ASSERT_TRUE(agx_nir_lower_g17p_sample_shading_file(b.shader, &link, samples,
                                                         m5_registers, true));
      nir_opt_constant_folding(b.shader);
      ASSERT_TRUE(agx_nir_lower_sample_mask_one_update_m5(b.shader));
      nir_validate_shader(b.shader, "M5 per-sample kill coverage");

      unsigned testing = 0, kill_only = 0, demotes = 0;
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
               if (in->intrinsic == nir_intrinsic_demote_samples)
                  demotes++;
               if (in->intrinsic != nir_intrinsic_sample_mask_agx)
                  continue;
               if (nir_src_is_const(in->src[1]) && nir_src_as_uint(in->src[1]) == 0)
                  kill_only++;
               else
                  testing++;
            }
         }
      }
      EXPECT_EQ(demotes, 0u);
      EXPECT_EQ(testing, 1u);
      EXPECT_LE(kill_only, 1u);

      /* The drivers bind or fold the update's test selector; here the
       * fragment always tests. */
      nir_shader_intrinsics_pass(
         b.shader,
         [](nir_builder *nb, nir_intrinsic_instr *in, void *) -> bool {
            if (in->intrinsic != nir_intrinsic_load_shader_part_tests_zs_agx)
               return false;
            nb->cursor = nir_before_instr(&in->instr);
            nir_def_replace(&in->def,
                            nir_imm_intN_t(nb, UINT16_MAX, in->def.bit_size));
            return true;
         },
         nir_metadata_control_flow, NULL);

      agx_preprocess_nir(b.shader);
      agx_nir_lower_g17p_tile_samples(b.shader, samples);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = AGX_CHIP_G17S;
      key.no_preamble = true;
      key.secondary = true;
      key.fs.nr_samples = samples;
      BITSET_SET(key.g17p_tile_partitions, 0);
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part)) << agx_last_compile_refusal();
      free(part.binary);
      ralloc_free(b.shader);
   }
}

TEST_F(G17PMSAAEpilog, NativeSpillEpilogsCompile)
{
   for (unsigned samples : {2u, 4u}) {
      for (unsigned targets : {4u, 8u}) {
         for (unsigned variant = 0; variant < 3; ++variant) {
            const bool blend = variant == 1, a2c = variant == 2;
            agx_fs_epilog_key fs = {};
            fs.nr_samples = samples;
            fs.blend.alpha_to_coverage = a2c;
            fs.max_sample_size_B = AGX_G17P_MAX_BYTES_PER_SAMPLE;
            fs.max_bytes_per_tile = 32768;
            fs.max_samples_per_tile = 2048;
            fs.native_spills = true;
            fs.emrt_heap_base = 16;
            fs.blend_constant_base = 20;
            fs.link.loc_written = (uint8_t)BITFIELD_MASK(targets);
            fs.link.size_32 = (uint8_t)BITFIELD_MASK(targets);
            fs.link.colour_comps = BITFIELD_MASK(4 * targets);
            fs.link.write_z = true;
            fs.defer_api_sample_mask = true;
            fs.api_sample_mask = 5 & BITFIELD_MASK(samples);
            agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &fs);
            for (unsigned rt = 0; rt < targets; ++rt) {
               fs.rt_formats[rt] = PIPE_FORMAT_R32G32B32A32_FLOAT;
               fs.remap[rt] = rt;
               fs.blend.rt[rt].colormask = 0xf;
               fs.blend.rt[rt].mode = agx_pack_blend_standard(
                  PIPE_BLEND_ADD,
                  blend ? PIPE_BLENDFACTOR_CONST_COLOR : PIPE_BLENDFACTOR_ONE,
                  blend ? PIPE_BLENDFACTOR_INV_CONST_COLOR : PIPE_BLENDFACTOR_ZERO,
                  PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
            }
            const agx_tilebuffer_layout tib = agx_build_tilebuffer_layout_with_limits(
               fs.rt_formats, targets, samples, false, fs.max_sample_size_B,
               fs.max_bytes_per_tile, fs.max_samples_per_tile);
            if (!agx_tilebuffer_spills(&tib))
               continue;
            SCOPED_TRACE(::testing::Message() << samples << " samples, "
                         << targets << " targets, blend=" << blend
                         << ", alpha-to-coverage=" << a2c);
            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P spilled MRT epilog");
            agx_nir_fs_epilog_g17p(&b, &fs);
            agx_preprocess_nir(b.shader);
            agx_nir_hoist_g17p_zs_writes(b.shader);
            agx_nir_lower_g17p_tile_samples(b.shader, samples);
            agx_shader_key key = {};
            key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
            key.dev.chip = AGX_CHIP_G17P;
            key.no_preamble = true;
            key.secondary = true;
            key.g17p_emrt = true;
            for (unsigned rt = 0; rt < targets; ++rt) {
               if (!tib.spilled[rt])
                  BITSET_SET(key.g17p_tile_partitions, tib._offset_B[rt]);
            }
            if (BITSET_COUNT(key.g17p_tile_partitions) < 2)
               BITSET_ZERO(key.g17p_tile_partitions);
            key.fs.nr_samples = samples;
            agx_shader_part part = {};
            EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part)) << agx_last_compile_refusal();
            free(part.binary);
            ralloc_free(b.shader);
         }
      }
   }
}

TEST_F(G17PMSAAEpilog, SingleSampleCollapsePreservesHelperCoverage)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "single-sample helper check");
   b.shader->info.fs.uses_sample_shading = true;
   nir_def *helper = nir_load_helper_invocation(&b, 1);
   nir_intrinsic_instr *exported =
      nir_intrinsic_instr_create(b.shader, nir_intrinsic_export_agx);
   exported->num_components = 1;
   exported->src[0] = nir_src_for_ssa(nir_b2i32(&b, helper));
   nir_intrinsic_set_base(exported, 8);
   nir_builder_instr_insert(&b, &exported->instr);
   ASSERT_TRUE(agx_nir_lower_sample_intrinsics(b.shader, true));
   ASSERT_TRUE(agx_nir_lower_g17p_1x_sample_shading(b.shader));
   EXPECT_FALSE(b.shader->info.fs.uses_sample_shading);
   unsigned active = 0, coverage = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            active += op == nir_intrinsic_load_active_samples_agx;
            coverage += op == nir_intrinsic_load_sample_mask;
         }
      }
   }
   EXPECT_EQ(active, 0u);
   EXPECT_EQ(coverage, 1u);
   ralloc_free(b.shader);
}

TEST_F(G17PMSAAEpilog, BroadcastNoopPreservesBoundTargets)
{
   for (unsigned samples : {1u, 2u, 4u}) {
      for (unsigned targets : {1u, 2u, 4u}) {
         SCOPED_TRACE(::testing::Message() << samples << " samples, "
                                           << targets << " targets");
         agx_fs_epilog_key fs = {};
         fs.nr_samples = samples;
         fs.max_sample_size_B = AGX_G17P_MAX_BYTES_PER_SAMPLE;
         fs.max_bytes_per_tile = 32768;
         fs.max_samples_per_tile = 2048;
         fs.link.loc_written = 1;
         fs.link.size_32 = 1;
         fs.link.colour_comps = 15;
         fs.link.broadcast_rt0 = true;
         fs.blend.logicop_enable = true;
         fs.blend.logicop_func = PIPE_LOGICOP_NOOP;
         agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &fs);
         for (unsigned rt = 0; rt < targets; ++rt) {
            fs.rt_formats[rt] = PIPE_FORMAT_R8G8B8A8_UNORM;
            fs.blend.rt[rt].colormask = 15;
            fs.blend.rt[rt].mode = agx_pack_blend_standard(
               PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
               PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
         }

         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "broadcast noop epilog");
         agx_nir_fs_epilog_g17p(&b, &fs);
         agx_preprocess_nir(b.shader);
         agx_nir_lower_g17p_tile_samples(b.shader, samples);
         unsigned loads = 0, stores = 0;
         nir_foreach_function_impl(impl, b.shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
                  loads += op == nir_intrinsic_load_local_pixel_agx;
                  stores += op == nir_intrinsic_store_local_pixel_agx;
               }
            }
         }
         EXPECT_EQ(loads, targets);
         EXPECT_EQ(stores, targets);
         EXPECT_TRUE(b.shader->info.fs.uses_fbfetch_output);
         ralloc_free(b.shader);
      }
   }
}

TEST(G17pMsaaEpilog, NativeLateSampleMaskCrossesEpilogBoundary)
{
   for (enum agx_chip chip : {AGX_CHIP_G13G, AGX_CHIP_G17P, AGX_CHIP_G17S}) {
      for (bool sample_mask : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "late coverage boundary");
      b.shader->info.io_lowered = true;
      b.shader->info.fs.uses_discard = true;
      b.shader->info.writes_memory = !sample_mask;
      b.shader->info.outputs_written =
         BITFIELD64_BIT(FRAG_RESULT_DATA0) |
         (sample_mask ? BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK) : 0);
      nir_demote_samples(&b, nir_imm_intN_t(&b, 0x5555, 16));
      nir_store_output(&b, nir_imm_vec4(&b, 1, 1, 0, 1),
         nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = FRAG_RESULT_DATA0, .num_slots = 1});
      struct agx_fs_epilog_link_info info = {};
      ASSERT_TRUE(agx_nir_lower_fs_output_to_epilog_for_device(b.shader, chip, &info));
      unsigned demotes = 0, masks = 0;
      nir_foreach_block(block, b.impl) nir_foreach_instr(in, block) {
         if (in->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(in);
         demotes += intr->intrinsic == nir_intrinsic_demote_samples;
         masks += intr->intrinsic == nir_intrinsic_export_agx &&
            nir_intrinsic_base(intr) == AGX_ABI_FOUT_WRITE_SAMPLES;
      }
      const bool native = chip == AGX_CHIP_G17P ||
         (chip == AGX_CHIP_G17S && sample_mask);
      EXPECT_EQ(info.sample_mask_after_force_early, native);
      EXPECT_EQ(info.loc0_w_1, chip != AGX_CHIP_G17P);
      EXPECT_EQ(demotes, native ? 0u : 1u);
      EXPECT_EQ(masks, native ? 1u : 0u);
      nir_validate_shader(b.shader, "late coverage transport");
      ralloc_free(b.shader);
      }
   }
}

TEST(LegacyFragments, EpilogUsesUpstreamDepthCoverage)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "legacy depth epilog");
   agx_fs_epilog_key key = {};
   key.nr_samples = 2;
   key.link.write_z = true;
   key.link.already_ran_zs = true;
   key.link.sample_mask_after_force_early = true;
   for (auto &r : key.remap) r = -1;
   agx_nir_fs_epilog(&b, &key);
   unsigned zs = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic) continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_zs_agx) continue;
         zs++;
         EXPECT_TRUE(nir_src_is_const(intr->src[0]));
         if (nir_src_is_const(intr->src[0])) {
            EXPECT_EQ(nir_src_as_uint(intr->src[0]), 0xffu);
         }
      }
   }
   EXPECT_EQ(zs, 1u);
   ralloc_free(b.shader);
}

TEST(LegacyFragments, EarlyCoverageIsPerSample)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "legacy early coverage");
   b.shader->info.fs.early_fragment_tests = true;
   nir_intrinsic_instr *mask = nir_sample_mask_agx(
      &b, nir_imm_intN_t(&b, 0xff, 16), nir_imm_intN_t(&b, 0xff, 16));
   EXPECT_TRUE(agx_nir_lower_to_per_sample_agx2(b.shader));
   EXPECT_FALSE(nir_src_is_const(mask->src[0]));
   ralloc_free(b.shader);
}

TEST_F(G17PMSAAEpilog, PerSampleFramebufferFetchMainCompilesForNativeAGX3)
{
   for (enum agx_chip chip : {AGX_CHIP_G17P, AGX_CHIP_G16G}) {
      for (unsigned samples : {2u, 4u}) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "sample framebuffer fetch");
         b.shader->info.fs.uses_sample_shading = true;
         nir_def *color = nir_load_local_pixel_agx(
            &b, 4, 32, nir_imm_intN_t(&b, 0xff, 16),
            .format = PIPE_FORMAT_R8_UNORM);
         nir_store_output(&b, color, nir_imm_int(&b, 0),
                          .write_mask = 0xf, .src_type = nir_type_float32,
                          .io_semantics = {.location = FRAG_RESULT_DATA0});
         b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
         agx_fs_epilog_link_info link = {};
         agx_nir_lower_fs_output_to_epilog_for_device(b.shader, chip, &link);
         ASSERT_TRUE(agx_nir_lower_g17p_sample_shading(b.shader, &link, samples));
         nir_opt_constant_folding(b.shader);
         agx_nir_lower_g17p_tile_samples(b.shader, samples);
         agx_preprocess_nir(b.shader);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         key.dev.chip = chip;
         key.no_preamble = true;
         key.no_stop = true;
         key.fs.nr_samples = samples;
         agx_shader_part part = {};
         EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
            << agx_last_compile_refusal();
         free(part.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST_F(G17PMSAAEpilog, ReadOnlyEarlyRejectPrologCompilesForAGX3)
{
   for (enum agx_chip chip : {AGX_CHIP_G17P, AGX_CHIP_G17S}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "read-only early reject");
      struct agx_fs_prolog_key pkey = {};
      pkey.api_sample_mask = 0xff;
      pkey.run_zs_tests = true;
      pkey.nr_samples = 1;
      agx_nir_fs_prolog(&b, &pkey);
      agx_preprocess_nir(b.shader);

      agx_shader_key key = {};
      key.dev.encoding = chip == AGX_CHIP_G17S ? AGX_ISA_ENCODING_AGX3C
                                               : AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = chip;
      key.no_preamble = true;
      key.no_stop = true;
      key.fs.nr_samples = 1;

      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
         << "chip " << chip << ": " << agx_last_compile_refusal();
      EXPECT_TRUE(part.info.writes_sample_mask) << "chip " << chip;
      free(part.binary);
      ralloc_free(b.shader);
   }
}

TEST_F(G17PMSAAEpilog, MonolithicLocalReadSampleEquivFastPathCompiles)
{
   for (enum agx_chip chip : {AGX_CHIP_G16G, AGX_CHIP_G17P}) {
      SCOPED_TRACE(chip);
      const unsigned samples = 4;
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "local-read sample equiv");
      b.shader->info.fs.uses_sample_shading = true;
      b.shader->info.fs.uses_fbfetch_output = true;

      nir_def *src = nir_load_local_pixel_agx(
         &b, 4, 16, nir_imm_intN_t(&b, 0xff, 16), .base = 0,
         .format = PIPE_FORMAT_R8_UNORM);
      nir_def *half = nir_fmul_imm(&b, src, 0.5);
      nir_store_local_pixel_agx(&b, half, nir_imm_intN_t(&b, 0xff, 16),
                                nir_undef(&b, 2, 16), .base = 0,
                                .write_mask = 0xf,
                                .format = PIPE_FORMAT_R8_UNORM);

      struct agx_g17p_colour_layout layout;
      agx_g17p_colour_layout_init(&layout, 1, 0, 0xf, false);
      ASSERT_TRUE(agx_nir_lower_to_per_sample(b.shader));
      ASSERT_TRUE(agx_nir_unroll_g17p_samples_equiv(b.shader, samples, &layout));

      bool progress;
      do {
         progress = false;
         progress |= nir_opt_constant_folding(b.shader);
         progress |= nir_opt_algebraic(b.shader);
      } while (progress);
      agx_nir_hoist_g17p_zs_writes(b.shader);
      agx_nir_lower_g17p_tile_samples(b.shader, samples);
      b.shader->info.fs.uses_sample_shading = false;
      nir_validate_shader(b.shader, "local-read sample equiv");

      unsigned ballots = 0, loads = 0, stores = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            const nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            ballots += op == nir_intrinsic_ballot;
            loads += op == nir_intrinsic_load_local_pixel_agx;
            stores += op == nir_intrinsic_store_local_pixel_agx;
         }
      }
      EXPECT_EQ(ballots, 1u);
      /* 4 probe loads + 0 in fast path + 3 in slow path (samples 1..3). */
      EXPECT_EQ(loads, 7u);
      /* 1 whole-pixel store in fast path + 4 per-sample stores in slow path. */
      EXPECT_EQ(stores, 5u);

      agx_preprocess_nir(b.shader);

      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = chip;
      key.no_preamble = true;
      key.agx3_local_read = true;
      key.fs.nr_samples = samples;
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
         << agx_last_compile_refusal();
      EXPECT_EQ(part.info.scratch_size, 0u);
      EXPECT_LE(part.info.nr_gprs, 32u);
      free(part.binary);
      ralloc_free(b.shader);
   }

   /* Ineligible: reads gl_SampleID. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "local-read sample_id");
      nir_def *src = nir_load_local_pixel_agx(
         &b, 4, 16, nir_imm_intN_t(&b, 0xff, 16), .base = 0,
         .format = PIPE_FORMAT_R8_UNORM);
      nir_def *sid = nir_f2f16(&b, nir_u2f32(&b, nir_load_sample_id(&b)));
      nir_def *out = nir_fadd(&b, src, nir_vec4(&b, sid, sid, sid, sid));
      nir_store_local_pixel_agx(&b, out, nir_imm_intN_t(&b, 0xff, 16),
                                nir_undef(&b, 2, 16), .base = 0,
                                .write_mask = 0xf,
                                .format = PIPE_FORMAT_R8_UNORM);
      struct agx_g17p_colour_layout layout;
      agx_g17p_colour_layout_init(&layout, 1, 0, 0xf, false);
      ASSERT_TRUE(agx_nir_lower_to_per_sample(b.shader));
      EXPECT_FALSE(agx_nir_unroll_g17p_samples_equiv(b.shader, 4, &layout));
      ralloc_free(b.shader);
   }

   /* Ineligible: uses discard. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "local-read discard");
      b.shader->info.fs.uses_discard = true;
      nir_def *src = nir_load_local_pixel_agx(
         &b, 4, 16, nir_imm_intN_t(&b, 0xff, 16), .base = 0,
         .format = PIPE_FORMAT_R8_UNORM);
      nir_store_local_pixel_agx(&b, src, nir_imm_intN_t(&b, 0xff, 16),
                                nir_undef(&b, 2, 16), .base = 0,
                                .write_mask = 0xf,
                                .format = PIPE_FORMAT_R8_UNORM);
      struct agx_g17p_colour_layout layout;
      agx_g17p_colour_layout_init(&layout, 1, 0, 0xf, false);
      ASSERT_TRUE(agx_nir_lower_to_per_sample(b.shader));
      EXPECT_FALSE(agx_nir_unroll_g17p_samples_equiv(b.shader, 4, &layout));
      ralloc_free(b.shader);
   }
}

TEST(G17PMsaaEpilog, MonolithicLocalReadDeferred4MRTMsaa4EquivCompiles)
{
   const enum agx_chip chips[] = {AGX_CHIP_G17P, AGX_CHIP_G16G};
   const unsigned samples = 4;
   static const struct {
      int base;
      unsigned target;
      enum pipe_format fmt;
   } inputs[4] = {
      {0, 0, PIPE_FORMAT_R8_UNORM},
      {4, 1, PIPE_FORMAT_R16_FLOAT},
      {12, 2, PIPE_FORMAT_R8_UNORM},
      {16, 3, PIPE_FORMAT_R8_UNORM},
   };
   static const unsigned partition_offsets[] = {0, 4, 12, 16, 20};

   for (enum agx_chip chip : chips) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options,
         "local-read deferred 4-MRT msaa4 equiv");
      b.shader->info.fs.uses_sample_shading = true;
      b.shader->info.fs.uses_fbfetch_output = true;

      nir_def *acc = nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 0.0f);
      for (unsigned k = 0; k < ARRAY_SIZE(inputs); ++k) {
         nir_def *v = nir_load_local_pixel_agx(
            &b, 4, 32, nir_imm_intN_t(&b, 0xff, 16), .base = inputs[k].base,
            .format = inputs[k].fmt, .target = inputs[k].target);
         acc = nir_fadd(&b, acc, v);
      }
      nir_def *lit = nir_fmul_imm(&b, acc, 0.25f);
      nir_store_local_pixel_agx(&b, lit, nir_imm_intN_t(&b, 0xff, 16),
                                nir_undef(&b, 2, 16), .base = 20,
                                .write_mask = 0xf,
                                .format = PIPE_FORMAT_R8_UNORM, .target = 4);

      struct agx_g17p_colour_layout layout;
      agx_g17p_colour_layout_init(&layout, BITFIELD_BIT(4), BITFIELD_BIT(4),
                                  0xf, false);
      ASSERT_TRUE(agx_nir_lower_to_per_sample(b.shader));
      ASSERT_TRUE(agx_nir_unroll_g17p_samples_equiv(b.shader, samples, &layout));

      bool progress;
      do {
         progress = false;
         progress |= nir_opt_constant_folding(b.shader);
         progress |= nir_opt_algebraic(b.shader);
         progress |= nir_opt_dce(b.shader);
      } while (progress);
      agx_nir_hoist_g17p_zs_writes(b.shader);
      agx_nir_lower_g17p_tile_samples(b.shader, samples);
      b.shader->info.fs.uses_sample_shading = false;
      nir_validate_shader(b.shader, "deferred 4-MRT local-read equiv");

      unsigned ballots = 0, loads = 0, stores = 0, f16_probes = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            const nir_intrinsic_op op = intr->intrinsic;
            ballots += op == nir_intrinsic_ballot;
            if (op == nir_intrinsic_load_local_pixel_agx) {
               loads++;
               if (intr->def.bit_size == 16)
                  f16_probes++;
            }
            stores += op == nir_intrinsic_store_local_pixel_agx;
         }
      }
      EXPECT_EQ(ballots, 1u);
      /* 16 probe loads (3 R8_UNORM narrowed to f16 = 12) + 4 fast-path
       * sample-0 loads + 16 slow-path loads = 36 total loads. */
      EXPECT_EQ(loads, 36u);
      EXPECT_EQ(f16_probes, 12u);
      /* 1 whole-pixel store in fast path + 4 per-sample stores in slow path. */
      EXPECT_EQ(stores, 5u);

      agx_preprocess_nir(b.shader);

      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = chip;
      key.no_preamble = true;
      key.agx3_local_read = true;
      key.fs.nr_samples = samples;
      for (unsigned off : partition_offsets)
         BITSET_SET(key.g17p_tile_partitions, off);
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
         << agx_last_compile_refusal();
      EXPECT_EQ(part.info.scratch_size, 0u);
      free(part.binary);
      ralloc_free(b.shader);
   }
}

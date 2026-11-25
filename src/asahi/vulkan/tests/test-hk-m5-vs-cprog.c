/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_agx3_vdm_decode.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_instance.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "asahi/compiler/agx_compile.h"
#include "asahi/lib/agx_vdm_state_update_core.h"
#include "compiler/nir/nir_builder.h"
#include "vk_alloc.h"
#include "vk_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

struct fake_code_bo {
   struct agx_bo bo;
   struct agx_va va;
   uint8_t *data;
};

static struct fake_code_bo fake_code_storage[64];
static unsigned fake_code_next;

static void
free_fake_bos(void)
{
   for (unsigned i = 0; i < fake_code_next; ++i) {
      free(fake_code_storage[i].data);
      fake_code_storage[i].data = NULL;
   }
   fake_code_next = 0;
}

static struct agx_bo *
alloc_fake_bo(struct agx_device *dev, size_t size)
{
   CHECK(fake_code_next < ARRAY_SIZE(fake_code_storage));
   const size_t alloc_size = size ? ALIGN_POT(size, 4096) : 4096;
   struct fake_code_bo *slot = &fake_code_storage[fake_code_next++];
   memset(slot, 0, sizeof(*slot));
   slot->data = calloc(1, alloc_size);
   CHECK(slot->data);
   slot->va.addr = dev->shader_base + 0x100000 +
                   (uint64_t)fake_code_next * 0x100000;
   slot->va.size_B = alloc_size;
   slot->bo.dev = dev;
   slot->bo.va = &slot->va;
   slot->bo.size = alloc_size;
   slot->bo._map = slot->data;
   return &slot->bo;
}

struct agx_bo *__wrap_hk_code_alloc(struct hk_device *dev, size_t size,
                                    uint64_t *offset_out);
struct agx_bo *__wrap_agx_bo_create(struct agx_device *dev, unsigned size,
                                    unsigned align, enum agx_bo_flags flags,
                                    const char *label);
VkResult __wrap_hk_device_ensure_geometry_heap(struct hk_device *dev);

struct agx_bo *
__wrap_hk_code_alloc(struct hk_device *dev, size_t size, uint64_t *offset_out)
{
   *offset_out = 0;
   return alloc_fake_bo(&dev->dev, size);
}

struct agx_bo *
__wrap_agx_bo_create(struct agx_device *dev, unsigned size, unsigned align,
                     enum agx_bo_flags flags, const char *label)
{
   return alloc_fake_bo(dev, size);
}

VkResult
__wrap_hk_device_ensure_geometry_heap(struct hk_device *dev)
{
   return VK_SUCCESS;
}

static void
check_policy(void)
{
   const uint64_t ep = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   const uint64_t v1 = DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES;
   const uint64_t v2 = DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES_V2;

   /* G17P hardware vertex shaders keep native constant programs regardless of
    * stage-prefix features. */
   CHECK(hk_allow_vs_cprog(AGX_CHIP_G17P, 0, true, MESA_SHADER_VERTEX, false));
   CHECK(hk_allow_vs_cprog(AGX_CHIP_G17P, ep | v2, true, MESA_SHADER_VERTEX,
                           false));
   CHECK(!hk_allow_vs_cprog(AGX_CHIP_G17P, ep | v2, false, MESA_SHADER_VERTEX,
                            false));
   CHECK(!hk_allow_vs_cprog(AGX_CHIP_G17P, ep | v2, true, MESA_SHADER_FRAGMENT,
                            false));
   CHECK(!hk_allow_vs_cprog(AGX_CHIP_G17P, ep | v2, true, MESA_SHADER_COMPUTE,
                            false));
   CHECK(!hk_allow_vs_cprog(AGX_CHIP_G17P, ep | v2, true, MESA_SHADER_VERTEX,
                            true));

   /* M5 parts omit vertex constant programs only when the kernel advertises
    * both exact prior barriers and render stage prefixes (matching hk_queue.c
    * staged render admission). Missing either bit preserves the CProg. */
   const enum agx_chip m5_chips[] = {
      AGX_CHIP_G17G,
      AGX_CHIP_G17S,
      AGX_CHIP_G17C,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(m5_chips); ++i) {
      CHECK(hk_allow_vs_cprog(m5_chips[i], 0, true, MESA_SHADER_VERTEX, false));
      CHECK(hk_allow_vs_cprog(m5_chips[i], ep, true, MESA_SHADER_VERTEX,
                              false));
      CHECK(hk_allow_vs_cprog(m5_chips[i], v1, true, MESA_SHADER_VERTEX,
                              false));
      CHECK(hk_allow_vs_cprog(m5_chips[i], v2, true, MESA_SHADER_VERTEX,
                              false));
      CHECK(hk_allow_vs_cprog(m5_chips[i], v1 | v2, true, MESA_SHADER_VERTEX,
                              false));
      CHECK(!hk_allow_vs_cprog(m5_chips[i], ep | v1, true, MESA_SHADER_VERTEX,
                               false));
      CHECK(!hk_allow_vs_cprog(m5_chips[i], ep | v2, true, MESA_SHADER_VERTEX,
                               false));
      CHECK(!hk_allow_vs_cprog(m5_chips[i], ep | v1 | v2, true,
                               MESA_SHADER_VERTEX, false));
      CHECK(!hk_allow_vs_cprog(m5_chips[i], 0, false, MESA_SHADER_VERTEX,
                               false));
      CHECK(!hk_allow_vs_cprog(m5_chips[i], 0, true, MESA_SHADER_VERTEX, true));
   }

   /* Chips without native preambles remain disallowed. */
   const enum agx_chip other_chips[] = {
      AGX_CHIP_G13G,
      AGX_CHIP_G15G,
      AGX_CHIP_G15S,
      AGX_CHIP_G18G,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(other_chips); ++i) {
      CHECK(!hk_allow_vs_cprog(other_chips[i], 0, true, MESA_SHADER_VERTEX,
                               false));
      CHECK(!hk_allow_vs_cprog(other_chips[i], ep | v2, true,
                               MESA_SHADER_VERTEX, false));
   }
}

static nir_shader *
build_stage_nir(mesa_shader_stage stage)
{
   nir_builder b =
      nir_builder_init_simple_shader(stage, &agx_nir_options, "m5_vs_cprog");
   const nir_address_format ubo_fmt =
      hk_ubo_addr_format(vk_robustness_disabled.uniform_buffers);
   const unsigned ubo_comps = nir_address_format_num_components(ubo_fmt);
   const unsigned ubo_bits = nir_address_format_bit_size(ubo_fmt);
   nir_def *idx = nir_vulkan_resource_index(
      &b, ubo_comps, ubo_bits, nir_imm_int(&b, 0),
      .desc_set = 0, .binding = 0,
      .desc_type = nir_descriptor_type_uniform_buffer);
   nir_def *desc = nir_load_vulkan_descriptor(
      &b, ubo_comps, ubo_bits, idx,
      .desc_type = nir_descriptor_type_uniform_buffer);
   nir_deref_instr *base_deref = nir_build_deref_cast(
      &b, desc, nir_var_mem_ubo, glsl_vec4_type(), 16);
   nir_deref_instr *deref0 =
      nir_build_deref_ptr_as_array(&b, base_deref, nir_imm_int(&b, 0));
   nir_deref_instr *deref1 =
      nir_build_deref_ptr_as_array(&b, base_deref, nir_imm_int(&b, 1));
   nir_deref_instr *deref2 =
      nir_build_deref_ptr_as_array(&b, base_deref, nir_imm_int(&b, 2));
   nir_def *u0 = nir_load_deref(&b, deref0);
   nir_def *u1 = nir_load_deref(&b, deref1);
   nir_def *u2 = nir_load_deref(&b, deref2);
   /* Enough uniform math (like a draw's MVP/normal transform) that
    * nir_opt_preamble hoists it whenever a vertex constant program is allowed. */
   nir_def *ubo = nir_ffma(&b, u0, u1, u2);
   ubo = nir_fdiv(&b, nir_fsqrt(&b, nir_fabs(&b, nir_ffma(&b, ubo, u0, u1))),
                  nir_fadd(&b, nir_fabs(&b, u2), nir_imm_float(&b, 1.0f)));

   if (stage == MESA_SHADER_VERTEX) {
      nir_def *vid = nir_i2f32(&b, nir_load_vertex_id(&b));
      nir_def *pos = nir_vec4(&b, nir_fadd(&b, nir_channel(&b, ubo, 0), vid),
                              nir_channel(&b, ubo, 1), nir_channel(&b, ubo, 2),
                              nir_channel(&b, ubo, 3));
      nir_store_output(&b, pos, nir_imm_int(&b, 0),
                       .io_semantics = {
                          .location = VARYING_SLOT_POS,
                          .num_slots = 1,
                       },
                       .write_mask = 0xf);
      b.shader->info.outputs_written = VARYING_BIT_POS;
   } else if (stage == MESA_SHADER_TESS_EVAL) {
      b.shader->info.tess._primitive_mode = TESS_PRIMITIVE_TRIANGLES;
      b.shader->info.tess.tcs_vertices_out = 3;
      b.shader->info.tess.spacing = TESS_SPACING_EQUAL;
      nir_def *tc = nir_load_tess_coord(&b);
      nir_def *pos = nir_vec4(&b, nir_fadd(&b, nir_channel(&b, ubo, 0),
                                           nir_channel(&b, tc, 0)),
                              nir_channel(&b, ubo, 1), nir_channel(&b, ubo, 2),
                              nir_channel(&b, ubo, 3));
      nir_store_output(&b, pos, nir_imm_int(&b, 0),
                       .io_semantics = {
                          .location = VARYING_SLOT_POS,
                          .num_slots = 1,
                       },
                       .write_mask = 0xf);
      b.shader->info.outputs_written = VARYING_BIT_POS;
   } else if (stage == MESA_SHADER_GEOMETRY) {
      b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
      b.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
      b.shader->info.gs.vertices_in = 3;
      b.shader->info.gs.vertices_out = 3;
      b.shader->info.gs.invocations = 1;
      b.shader->info.gs.active_stream_mask = 1;
      for (unsigned v = 0; v < 3; ++v) {
         nir_def *pos = nir_vec4(
            &b, nir_fadd(&b, nir_channel(&b, ubo, 0), nir_imm_float(&b, v)),
            nir_channel(&b, ubo, 1), nir_channel(&b, ubo, 2),
            nir_channel(&b, ubo, 3));
         nir_store_output(&b, pos, nir_imm_int(&b, 0),
                          .io_semantics = {
                             .location = VARYING_SLOT_POS,
                             .num_slots = 1,
                          },
                          .write_mask = 0xf);
         nir_emit_vertex(&b, .stream_id = 0);
      }
      nir_end_primitive(&b, .stream_id = 0);
      b.shader->info.outputs_written = VARYING_BIT_POS;
   } else {
      abort();
   }

   nir_shader_gather_info(b.shader, b.impl);
   return b.shader;
}

static void
compile_production_case(enum agx_chip chip, uint64_t features,
                        mesa_shader_stage sw_stage, bool expect_cprog)
{
   fake_code_next = 0;

   struct hk_descriptor_set_layout *layout =
      calloc(1, sizeof(*layout) + sizeof(layout->binding[0]));
   CHECK(layout);
   layout->descriptor_generation = agx_chip_descriptor_generation(chip);
   layout->binding_count = 1;
   layout->vk.dynamic_descriptor_count = 1;
   layout->binding[0] = (struct hk_descriptor_set_binding_layout){
      .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
      .array_size = 1,
      .dynamic_buffer_index = 0,
   };
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};

   struct hk_instance *instance = calloc(1, sizeof(*instance));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   struct hk_device *dev = calloc(1, sizeof(*dev));
   CHECK(instance && pdev && dev);
   pdev->vk.instance = &instance->vk;
   pdev->dev.chip = chip;
   pdev->dev.params.features = features;
   pdev->dev.params.num_clusters_total = 4;
   pdev->dev.params.num_cores_per_cluster = 10;
   pdev->dev.shader_base = UINT64_C(0x100000000);
   dev->vk.base.device = &dev->vk;
   dev->vk.physical = &pdev->vk;
   dev->vk.alloc = *vk_default_allocator();
   dev->dev = pdev->dev;

   nir_shader *nir = build_stage_nir(sw_stage);
   struct vk_shader_compile_info info = {
      .stage = sw_stage,
      .nir = nir,
      .next_stage_mask = VK_SHADER_STAGE_FRAGMENT_BIT,
      .robustness = &vk_robustness_disabled,
      .set_layout_count = 1,
      .set_layouts = layouts,
   };
   struct vk_features vk_feats = {0};
   struct hk_api_shader *api_shader = NULL;
   VkResult res = hk_compile_shader(dev, &info, NULL, &vk_feats,
                                    &dev->vk.alloc, &api_shader);
   CHECK(res == VK_SUCCESS && api_shader != NULL);

   const unsigned variant =
      sw_stage == MESA_SHADER_GEOMETRY ? HK_GS_VARIANT_RAST : HK_VS_VARIANT_HW;
   struct hk_shader *shader = &api_shader->variants[variant];
   CHECK(shader->compiled);
   CHECK(shader->b.info.stage == MESA_SHADER_VERTEX);

   if (expect_cprog) {
      CHECK(shader->b.info.has_preamble);
      CHECK(shader->b.info.preamble_scratch_size > 0);
      CHECK(shader->agx3_vdm_state_update.source_layout ==
            AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT);
      CHECK(shader->agx3_vdm_state_update.template_info.raw_state_load_count_12 != 0 ||
            shader->agx3_vdm_state_update.template_info.raw_state_load_count_18 != 0);
   } else {
      CHECK(!shader->b.info.has_preamble);
      CHECK(shader->b.info.preamble_scratch_size == 0);
      CHECK(shader->b.info.scratch_size == 0);
      CHECK(shader->agx3_vdm_state_update.source_layout ==
            AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT);
      CHECK(shader->agx3_vdm_state_update.template_info.raw_state_load_count_12 == 0);
      CHECK(shader->agx3_vdm_state_update.template_info.raw_state_load_count_18 == 0);
   }

   /* Encode the draw-time VDM state update from the compiled template and
    * verify the emitted thread-invariant resource fields. */
   struct agx_vdm_state_update_draw_info draw_info = {
      .source_layout = shader->agx3_vdm_state_update.source_layout,
      .draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED,
      .rasterization_enabled = true,
      .raw_provoking_selector = 0,
      .template_info = shader->agx3_vdm_state_update.template_info,
   };
   draw_info.template_info.pipeline = UINT64_C(0x10000000000);
   draw_info.template_info.raw_active_context_selector = UINT32_MAX;

   struct agx_vdm_state_update encoded;
   struct agx_vdm_state_update_info decoded;
   CHECK(hk_agx3_encode_vdm_state_update_draw(chip, &draw_info, &encoded));
   CHECK(agx_vdm_state_update_unpack(chip, encoded.bytes, encoded.length,
                                     &decoded));
   const uint32_t ti_ipr_quads =
      (decoded.raw_vertex_shader_word0 >> 8) & UINT32_C(0x0fff);
   const uint32_t ti_spill_quads =
      (decoded.raw_pipeline_template >> 3) & UINT32_C(0x0003ffff);
   CHECK(ti_ipr_quads ==
         shader->agx3_vdm_state_update.template_info.raw_state_load_count_12);
   CHECK(ti_spill_quads ==
         shader->agx3_vdm_state_update.template_info.raw_state_load_count_18);
   if (expect_cprog) {
      CHECK(ti_ipr_quads != 0 || ti_spill_quads != 0);
   } else {
      CHECK(ti_ipr_quads == 0 && ti_spill_quads == 0);
   }

   /* Exercise production graphics loader uniform range collection. */
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   struct hk_linked_shader dummy_linked = {0};
   CHECK(cmd);
   cmd->vk.base.device = &dev->vk;
   const uint64_t root_va = UINT64_C(0x200000000);
   const uint64_t ubo_va = UINT64_C(0x200010000);
   cmd->state.gfx.descriptors.root.dynamic_buffers[0] =
      (struct hk_buffer_address){
         .base_addr = ubo_va,
         .size = 256,
      };
   struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   uint8_t sources[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   size_t count = 0;

   const struct hk_linked_shader *linked =
      shader->only_linked ? shader->only_linked : &dummy_linked;
   CHECK(hk_g17p_graphics_uniform_ranges(cmd, shader, linked, root_va, ranges,
                                         &count, sources));

   if (sw_stage == MESA_SHADER_VERTEX && !expect_cprog) {
      /* API VS uses uniform DMA for its dynamic UBO buffer; verify both the
       * compiler span and the emitted graphics loader range (address,
       * destination half, and half count). */
      CHECK(shader->b.info.agx3_uniform_dma_count >= 1);
      const struct agx3_uniform_dma_span *ubo_span = NULL;
      for (unsigned i = 0; i < shader->b.info.agx3_uniform_dma_count; ++i) {
         const struct agx3_uniform_dma_span *span =
            &shader->b.info.agx3_uniform_dma[i];
         if (span->source == AGX3_UNIFORM_DMA_BUFFER &&
             span->descriptor_B ==
                hk_root_descriptor_offset(dynamic_buffers)) {
            ubo_span = span;
            break;
         }
      }
      CHECK(ubo_span != NULL);
      CHECK(ubo_span->halfs >= 24);

      bool found_ubo_dma = false;
      for (size_t i = 0; i < count; ++i) {
         if (ranges[i].address == ubo_va + ubo_span->offset_B &&
             ranges[i].start_halfs == ubo_span->dest &&
             ranges[i].size_halfs == ubo_span->halfs) {
            found_ubo_dma = true;
            break;
         }
      }
      CHECK(found_ubo_dma);
   } else if (sw_stage == MESA_SHADER_TESS_EVAL ||
              sw_stage == MESA_SHADER_GEOMETRY) {
      /* Native TES and GS raster stay outside vs_dma (zero DMA spans) and keep
       * the root range for direct descriptor loads. */
      CHECK(shader->info.agx3_uses_root);
      CHECK(shader->b.info.agx3_uniform_dma_count == 0);
      bool found_root = false;
      for (size_t i = 0; i < count; ++i) {
         CHECK(ranges[i].address != ubo_va);
         if (sources[i] == HK_G17P_RANGE_SOURCE_ROOT &&
             ranges[i].address == root_va &&
             ranges[i].start_halfs == shader->info.agx3_root_uniform &&
             ranges[i].size_halfs > 0) {
            found_root = true;
         }
      }
      CHECK(found_root);
   }

   free(cmd);

   /* Zero fake BO pointers before hk_api_shader_destroy so it does not try to
    * unreference static test storage. */
   const unsigned nvariants = hk_num_variants(sw_stage);
   for (unsigned i = 0; i < nvariants; ++i) {
      struct hk_shader *v = &api_shader->variants[i];
      v->bo = NULL;
      if (v->only_linked) {
         v->only_linked->b.bo = NULL;
         if (v->only_linked->g17p_layout)
            v->only_linked->g17p_layout->bo = NULL;
         if (v->only_linked->variant_source)
            v->only_linked->variant_source->bo = NULL;
      }
   }
   hk_api_shader_destroy(&dev->vk, &api_shader->vk, &dev->vk.alloc);
   free_fake_bos();
   free(dev);
   free(pdev);
   free(instance);
   free(layout);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   check_policy();

   const uint64_t ep = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   const uint64_t v1 = DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES;
   const uint64_t v2 = DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES_V2;

   /* M5 with exact prior barriers + stage prefixes (v2 or v1) omits the VS
    * CProg and uses uniform DMA. */
   compile_production_case(AGX_CHIP_G17C, ep | v2, MESA_SHADER_VERTEX, false);
   compile_production_case(AGX_CHIP_G17C, ep | v1, MESA_SHADER_VERTEX, false);

   /* M5 lacking either exact prior barriers or stage prefixes preserves the
    * VS CProg. */
   compile_production_case(AGX_CHIP_G17C, 0, MESA_SHADER_VERTEX, true);
   compile_production_case(AGX_CHIP_G17C, ep, MESA_SHADER_VERTEX, true);
   compile_production_case(AGX_CHIP_G17C, v2, MESA_SHADER_VERTEX, true);
   compile_production_case(AGX_CHIP_G17C, v1, MESA_SHADER_VERTEX, true);

   /* G17P preserves the VS CProg even with exact prior + stage prefixes v2. */
   compile_production_case(AGX_CHIP_G17P, ep | v2, MESA_SHADER_VERTEX, true);

   /* Native TES and GS raster hardware vertex variants compile cleanly without
    * a CProg on stage-prefix-capable M5 and retain direct root loads. */
   compile_production_case(AGX_CHIP_G17C, ep | v2, MESA_SHADER_TESS_EVAL, false);
   compile_production_case(AGX_CHIP_G17C, ep | v2, MESA_SHADER_GEOMETRY, false);

   glsl_type_singleton_decref();
   return 0;
}

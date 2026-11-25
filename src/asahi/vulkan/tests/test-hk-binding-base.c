/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_descriptor_set_layout.h"
#include "hk_shader.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "vk_pipeline.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x);               \
         abort();                                                              \
      }                                                                        \
   } while (0)

static bool
lower_offset_loads(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_global_constant_offset &&
       intr->intrinsic != nir_intrinsic_load_global_constant_bounded)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *addr = nir_iadd(b, intr->src[0].ssa, nir_u2u64(b, intr->src[1].ssa));
   enum gl_access_qualifier access = nir_intrinsic_access(intr);
   nir_def *val;
   if (intr->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      val = nir_load_global(
         b, intr->def.num_components, intr->def.bit_size, addr,
         .align_mul = nir_intrinsic_align_mul(intr),
         .align_offset = nir_intrinsic_align_offset(intr),
         .access = access | ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
   } else {
      val = nir_load_global_constant(
         b, intr->def.num_components, intr->def.bit_size, addr,
         .align_mul = nir_intrinsic_align_mul(intr),
         .align_offset = nir_intrinsic_align_offset(intr), .access = access);
   }
   nir_def_replace(&intr->def, val);
   return true;
}

static bool
lower_root_and_sets(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_root_agx &&
       intr->intrinsic != nir_intrinsic_load_descriptor_set_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   unsigned base =
      intr->intrinsic == nir_intrinsic_load_root_agx
         ? 0
         : 4 + 4 * nir_intrinsic_desc_set(intr);
   nir_def_replace(&intr->def, nir_load_preamble(b, 1, 64, .base = base));
   return true;
}

static void
lower_for_agx_passes(nir_shader *s, struct hk_descriptor_set_layout *layout)
{
   struct vk_descriptor_set_layout *layouts[1] = {&layout->vk};
   CHECK(hk_nir_lower_descriptors(s, &vk_robustness_disabled, 1, layouts, false,
                                  false, 0, false, false, false, false, false));
   NIR_PASS(_, s, nir_lower_explicit_io, nir_var_mem_ssbo,
            hk_buffer_addr_format(
               VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT, false,
               true));
   NIR_PASS(_, s, nir_lower_explicit_io, nir_var_mem_ubo,
            hk_ubo_addr_format(
               VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT));
   NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_offset_loads,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_root_and_sets,
            nir_metadata_control_flow, NULL);

   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_cse);
      NIR_PASS(progress, s, nir_opt_dce);
   } while (progress);
   nir_validate_shader(s, "after descriptor and io lowering");
}

static struct hk_descriptor_set_layout *
make_layout(VkDescriptorType type)
{
   struct hk_descriptor_set_layout *layout =
      calloc(1, sizeof(*layout) + sizeof(layout->binding[0]));
   CHECK(layout);
   layout->descriptor_generation = AGX_DESCRIPTOR_GENERATION_10;
   layout->binding_count = 1;
   layout->binding[0] = (struct hk_descriptor_set_binding_layout){
      .type = type,
      .array_size = 2,
      .stride = 16,
      .offset = 0,
      .dynamic_buffer_index = 0,
   };
   if (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
       type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
      layout->vk.dynamic_descriptor_count = 1;
   return layout;
}

static nir_def *
make_binding_desc(nir_builder *b, bool is_ssbo, bool complex)
{
   const nir_descriptor_type dt = is_ssbo ? nir_descriptor_type_storage_buffer
                                          : nir_descriptor_type_uniform_buffer;
   const nir_resource_type rt = is_ssbo
                                   ? nir_resource_type_read_write_storage_buffer
                                   : nir_resource_type_uniform_buffer;
   const unsigned comps = is_ssbo ? 5 : 4;
   nir_def *idx = nir_vulkan_resource_index(b, comps, 32, nir_imm_int(b, 0),
                                            .desc_set = 0, .binding = 0,
                                            .desc_type = dt,
                                            .resource_type = rt);
   if (complex) {
      nir_def *idx1 = nir_vulkan_resource_index(b, comps, 32, nir_imm_int(b, 1),
                                                .desc_set = 0, .binding = 0,
                                                .desc_type = dt,
                                                .resource_type = rt);
      nir_def *cond = nir_ine_imm(
         b, nir_load_push_constant(b, 1, 32, nir_imm_int(b, 4), .range = 32), 0);
      idx = nir_bcsel(b, cond, idx, idx1);
   }
   return nir_load_vulkan_descriptor(b, comps, 32, idx, .desc_type = dt);
}

/* Verify that hk_nir_lower_descriptors marks real buffer bindings (UBO, SSBO,
 * dynamic UBO/SSBO, complex SSBO) so both agx_nir_merge_global_words and
 * agx_nir_fold_m5_offsets match them. */
static void
check_binding_accepted(VkDescriptorType type, bool is_ssbo, bool complex)
{
   struct hk_descriptor_set_layout *layout = make_layout(type);
   const nir_variable_mode mode = is_ssbo ? nir_var_mem_ssbo : nir_var_mem_ubo;
   const enum gl_access_qualifier ro =
      ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;

   /* Merge consecutive words */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "hk-binding-merge");
      nir_def *desc = make_binding_desc(&b, is_ssbo, complex);
      nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0),
                                          .range = 32);
      nir_deref_instr *root = nir_build_deref_cast_with_alignment(
         &b, desc, mode, glsl_uint_type(), 4, 4, 0);
      nir_def *sum = nir_imm_int(&b, 0);
      for (unsigned k = 0; k < 4; ++k) {
         nir_deref_instr *elem = nir_build_deref_ptr_as_array(
            &b, root, nir_iadd_imm(&b, x, k));
         nir_def *v = nir_load_deref_with_access(&b, elem, ro);
         sum = nir_iadd(&b, sum, v);
      }
      nir_store_global(&b, sum, nir_imm_int64(&b, 0x100000), .align_mul = 4);
      lower_for_agx_passes(b.shader, layout);
      CHECK(agx_nir_merge_global_words(b.shader));
      nir_validate_shader(b.shader, "after hk binding merge");
      ralloc_free(b.shader);
   }

   /* Fold byte siblings */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "hk-binding-fold");
      nir_def *desc = make_binding_desc(&b, is_ssbo, complex);
      nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0),
                                          .range = 32);
      nir_deref_instr *root = nir_build_deref_cast(
         &b, desc, mode, glsl_uint8_t_type(), 1);
      nir_def *sum = nir_imm_int(&b, 0);
      for (unsigned k = 0; k < 3; ++k) {
         nir_deref_instr *elem = nir_build_deref_ptr_as_array(
            &b, root, nir_iadd_imm(&b, x, k));
         nir_def *v = nir_load_deref_with_access(&b, elem, ro);
         sum = nir_iadd(&b, sum, nir_u2u32(&b, v));
      }
      nir_store_global(&b, sum, nir_imm_int64(&b, 0x100000), .align_mul = 4);
      lower_for_agx_passes(b.shader, layout);
      NIR_PASS(_, b.shader, nir_lower_pack);
      CHECK(agx_nir_fold_m5_offsets(b.shader, false));
      nir_validate_shader(b.shader, "after hk binding fold");
      ralloc_free(b.shader);
   }

   free(layout);
}

/* Verify that a BDA pointer packed from a push-constant uvec4 or a UBO uvec4
 * after real hk_nir_lower_descriptors lowering is refused by both passes. */
static void
check_packed_uvec4_refused(bool from_ubo)
{
   struct hk_descriptor_set_layout *layout =
      make_layout(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
   const enum gl_access_qualifier ro =
      ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;

   /* Merge refusal */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "hk-uvec4-merge-refuse");
      nir_def *v;
      if (from_ubo) {
         nir_def *desc = make_binding_desc(&b, false, false);
         nir_deref_instr *root = nir_build_deref_cast(
            &b, desc, nir_var_mem_ubo, glsl_vector_type(GLSL_TYPE_UINT, 4), 16);
         v = nir_load_deref_with_access(&b, root, ro);
      } else {
         v = nir_load_push_constant(&b, 4, 32, nir_imm_int(&b, 16), .range = 32);
      }
      nir_def *ptr = nir_pack_64_2x32(&b, nir_trim_vector(&b, v, 2));
      nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0),
                                          .range = 32);
      nir_def *sum = nir_imm_int(&b, 0);
      for (unsigned k = 0; k < 2; ++k) {
         nir_def *addr = nir_iadd(&b, ptr, nir_u2u64(&b, nir_iadd_imm(&b, x, 4 * k)));
         sum = nir_iadd(&b, sum,
                        nir_load_global(&b, 1, 32, addr, .access = ro,
                                        .align_mul = 4));
      }
      nir_store_global(&b, sum, nir_imm_int64(&b, 0x100000), .align_mul = 4);
      lower_for_agx_passes(b.shader, layout);
      CHECK(!agx_nir_merge_global_words(b.shader));
      ralloc_free(b.shader);
   }

   /* Fold refusal */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "hk-uvec4-fold-refuse");
      nir_def *v;
      if (from_ubo) {
         nir_def *desc = make_binding_desc(&b, false, false);
         nir_deref_instr *root = nir_build_deref_cast(
            &b, desc, nir_var_mem_ubo, glsl_vector_type(GLSL_TYPE_UINT, 4), 16);
         v = nir_load_deref_with_access(&b, root, ro);
      } else {
         v = nir_load_push_constant(&b, 4, 32, nir_imm_int(&b, 16), .range = 32);
      }
      nir_def *ptr = nir_pack_64_2x32(&b, nir_trim_vector(&b, v, 2));
      nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0),
                                          .range = 32);
      nir_def *sum = nir_imm_int(&b, 0);
      for (unsigned k = 1; k <= 3; ++k) {
         nir_def *addr = nir_iadd(&b, ptr, nir_u2u64(&b, nir_iadd_imm(&b, x, k)));
         nir_def *byte = nir_load_global(&b, 1, 8, addr, .align_mul = 1);
         sum = nir_iadd(&b, sum, nir_u2u32(&b, byte));
      }
      nir_store_global(&b, sum, nir_imm_int64(&b, 0x100000), .align_mul = 4);
      lower_for_agx_passes(b.shader, layout);
      NIR_PASS(_, b.shader, nir_lower_pack);
      CHECK(!agx_nir_fold_m5_offsets(b.shader, false));
      ralloc_free(b.shader);
   }

   free(layout);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   check_binding_accepted(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, true, false);
   check_binding_accepted(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, false, false);
   check_binding_accepted(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, true, false);
   check_binding_accepted(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, false, false);
   check_binding_accepted(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, true, true);
   check_packed_uvec4_refused(false);
   check_packed_uvec4_refused(true);
   glsl_type_singleton_decref();
   return 0;
}

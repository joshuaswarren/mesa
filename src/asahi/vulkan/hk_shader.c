/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2023 Advanced Micro Devices, Inc.
 * Copyright 2018 Intel Corporation
 * SPDX-License-Identifier: MIT
 */
#include "hk_shader.h"
#include "hk_buffer.h"
#include "hk_frame_anatomy.h"
#include "hk_agx3_ioto_program_core.h"
#include "hk_nir_lower_gs.h"
#include "hk_nir_vertex_preamble.h"
#include "hk_nir_root_size.h"
#include "hk_nir_lower_tes_rq.h"
#include "hk_native_write_class.h"
#include "asahi/compiler/agx_nir_m5_mesh.h"
#include "asahi/compiler/agx3_mesh_core.h"
#include "hk_nir_sampler_bias.h"
#include "hk_gen1_rt_indirect_contract.h"
#include "asahi/compiler/agx_nir_m5_mesh.h"
#include "asahi/compiler/agx3_mesh_core.h"
#include "asahi/compiler/agx_nir_gen1_ray_query.h"
#include "hk_nir_border_index.h"
#include "util/u_debug.h"
#include "util/u_atomic.h"

#include "hk_nir_ray_tracing.h"
#include "hk_nir_forward_image_stores.h"
#include "hk_nir_widen_16bit_varyings.h"
#include "hk_nir_lower_robust_image.h"
#include "hk_nir_lower_image_view_min_lod.h"
#include "hk_entrypoints.h"

#include "poly/nir/poly_nir.h"
#include "agx_compiler.h"
#include "agx_debug.h"
#include "agx_device.h"
#include "agx_helpers.h"
#include "agx_nir_lower_vbo.h"
#include "glsl_types.h"
#include "hk_instance.h"
#include "hk_image.h"
#include "hk_agx3_vdm_decode.h"
#include "libagx.h"
#include "libagx_dgc.h"
#include "nir.h"
#include "nir_serialize.h"
#include "util/blob.h"
#include "nir_builder.h"
#include "nir_deref.h"

#include "agx_bo.h"
#include "hk_cmd_buffer.h"
#include "pipe/p_defines.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_image.h"
#include "hk_native_descriptor.h"
#include "hk_physical_device.h"
#include "hk_sampler.h"
#include "hk_shader.h"
#include "hk_nir_sampler_bias.h"
#include "asahi/compiler/agx_nir_gen1_ray_query.h"

#include "nir_builder_opcodes.h"
#include "nir_builtin_builder.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "nir_xfb_info.h"
#include "shader_enums.h"
#include "vk_graphics_state.h"
#include "vk_nir_convert_ycbcr.h"
#include "vk_physical_device_features.h"
#include "vk_common_entrypoints.h"
#include "vk_enum_to_str.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"
#include "vk_shader.h"
#include "vk_shader_module.h"
#include "vk_ycbcr_conversion.h"

#include "asahi/compiler/agx3_vary_core.h"
#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_nir.h"
#include "asahi/compiler/agx_nir_texture.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx_linker.h"
#include "asahi/lib/agx_tilebuffer.h"
#include "asahi/lib/agx_uvs.h"
#include "compiler/spirv/nir_spirv.h"

#include <stdarg.h>

#include "util/blob.h"
#include "util/hash_table.h"
#include "util/macros.h"
#include "util/disk_cache.h"
#include "util/mesa-blake3.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "vulkan/vulkan_core.h"

struct hk_fs_key {
   bool zs_self_dep;

   /** True if sample shading is forced on via an API knob such as
    * VkPipelineMultisampleStateCreateInfo::minSampleShading
    */
   bool force_sample_shading;

   /** G17P: static rasterization samples the fragment's coverage updates
    * are compiled for, or 0 when the count is dynamic or unknown.
    */
   uint8_t g17p_samples;

   /** M5 sample-position width selection; 0 means dynamic or unknown. */
   uint8_t m5_position_samples;

   /** AGX3: alpha-to-coverage is statically disabled, so no fragment epilog of
    * this shader discards unless the shader writes depth or stencil
    * (agx_fs_epilog_may_discard()). The main part then always runs the
    * depth/stencil tests itself and need not select that at draw time; the
    * pipeline state also rules out a read-only early reject.
    */
   bool static_tests : 1;

   /** G17P: complete pipeline state proves that no colour output can be
    * consumed (no colour attachment, alpha-to-coverage statically off), so
    * colour stores are pruned before varying linking.
    */
   bool discard_color_outputs : 1;

   /** G17P: multisampled with alpha-to-coverage statically on and every sample
    * enabled, so the main part can apply it (agx_fs_epilog_key::a2c_in_main).
    */
   bool static_a2c : 1;

   /** The bound depth/stencil feedback loop is only known at draw time, so
    * keep the NIR to compile the zs_self_dep twin (hk_fbl_twin()).
    */
   bool dynamic_zs_self_dep : 1;

   uint8_t pad[3];
};

struct hk_shader_binary_cache_entry {
   uint8_t *binary;
   size_t binary_size;
   uint32_t binary_version;
   struct hk_api_shader *shader;

   /* Protected by hk_device::shader_binary_cache.lock.  The cache owns one
    * reference and each live API wrapper owns one. */
   uint32_t refcount;
};

static_assert(sizeof(struct hk_fs_key) == 8, "packed");

struct hk_vs_key {
   struct agx_velem_key attribs[32];
   bool skip_prolog;
   bool static_strides;
   bool kill_psiz;

   /** Compiled together with a fragment shader that never reads gl_Layer:
    * no default layer output (agx_nir_lower_default_layer()) is needed.
    */
   bool no_default_layer;
};
static_assert(sizeof(struct hk_vs_key) == 260, "packed");

struct hk_mesh_key {
   /** An M5 mesh shader compiled without a fragment shader: write a zero
    * gl_PrimitiveLayerEXT for every primitive (hk_m5_mesh_default_layer()).
    */
   bool default_layer;
};

union hk_key {
   struct hk_vs_key vs;
   struct hk_fs_key fs;
   struct hk_mesh_key mesh;
};

static void
shared_var_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size =
      glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length, *align = comp_size;
}

/* Inline blending changes the fragment programs fast-link builds. */
static uint64_t
hk_fs_structure_key_bits(void)
{
   return (uint64_t)!!(agx_get_fs_structure_flags() &
                       AGX_FS_STRUCTURE_INLINE_BLEND) << 30;
}

uint64_t
hk_physical_device_compiler_flags(const struct hk_physical_device *pdev)
{
   /* This could be optimized but it doesn't matter. */
   return pdev->dev.debug |
          hk_fs_structure_key_bits();
}

const nir_shader_compiler_options *
hk_get_nir_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                   UNUSED const struct vk_pipeline_robustness_state *rs)
{
   return &agx_nir_options;
}

struct spirv_to_nir_options
hk_get_spirv_options(struct vk_physical_device *vk_pdev,
                     UNUSED mesa_shader_stage stage,
                     const struct vk_pipeline_robustness_state *rs)
{
   return (struct spirv_to_nir_options){
      .ssbo_addr_format = hk_buffer_addr_format(
         rs->storage_buffers, rs->null_storage_buffer_descriptor,
         agx_chip_g17_m5(((struct hk_physical_device *)vk_pdev)->dev.chip)),
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .ubo_addr_format = hk_ubo_addr_format(rs->uniform_buffers),
      .shared_addr_format = nir_address_format_32bit_offset,
      .min_ssbo_alignment = HK_MIN_SSBO_ALIGNMENT,
      .min_ubo_alignment = HK_MIN_UBO_ALIGNMENT,
   };
}

/* Whether agx_preprocess_nir() would lower a local array to scratch. */
static bool
hk_has_scratch_sized_array(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_function_temp_variable(var, impl) {
         unsigned size, align;
         glsl_get_natural_size_align_bytes(var->type, &size, &align);
         if (size > AGX_SCRATCH_ARRAY_THRESHOLD_B)
            return true;
      }
   }

   return false;
}

/*
 * A fragment shader in a depth/stencil feedback loop samples the same depth
 * attachment it renders to. Measured on G17P with a single draw and no
 * barriers: subpassLoad() of the depth attachment returned this draw's own
 * interpolated z instead of the pre-draw contents. With depth writes disabled
 * the same read returns the clear correctly, so the input-attachment read path
 * is sound -- what is wrong is that the fixed-function depth write lands
 * before the shader's read.
 *
 * Forcing late-Z is not enough, because late-Z only moves the depth *test*.
 * Writing gl_FragDepth moves the depth *write* into the shader
 * (agx_nir_lower_discard_zs_emit turns a FRAG_RESULT_DEPTH store into
 * store_zs_agx, emitted after the reads), which is why the same CTS cases pass
 * at the sample counts where the test writes gl_FragDepth itself and fail at
 * sample count 1 where it does not.
 *
 * gl_FragCoord.z is exactly the value fixed-function would have written: a
 * depth round-trip probe on this hardware showed the interpolated value, the
 * stored value and the vertex z bit-identical across every sample tested.
 *
 * This costs early depth, so it is gated on
 * VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT
 * (hk_fs_key::zs_self_dep). A pipeline that declares that bit has already
 * given up early depth by definition, so the production path is unchanged.
 *
 * Runs after nir_lower_io, so the write is emitted as a store_output.
 */
static void
hk_fs_force_shader_depth_write(nir_shader *nir)
{
   /* A shader that already writes gl_FragDepth orders its own write. */
   if (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH))
      return;

   /* Declaring early_fragment_tests is the application asking for the write to
    * happen before the shader, and agx_nir_lower_discard_zs_emit() deletes the
    * z store in that case. Leave it alone.
    */
   if (nir->info.fs.early_fragment_tests)
      return;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_after_impl(impl));

   nir_store_output(
      &b, nir_channel(&b, nir_load_frag_coord(&b), 2), nir_imm_int(&b, 0),
      .base = 0, .write_mask = 0x1, .component = 0,
      .src_type = nir_type_float32,
      .io_semantics = {.location = FRAG_RESULT_DEPTH, .num_slots = 1});

   nir->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DEPTH);
}

static bool
hk_cf_list_has_ray_query(struct exec_list *cf_list)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block:
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_intrinsic_is_ray_query(nir_instr_as_intrinsic(instr)->intrinsic))
               return true;
         }
         break;
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         if (hk_cf_list_has_ray_query(&nif->then_list) ||
             hk_cf_list_has_ray_query(&nif->else_list))
            return true;
         break;
      }
      case nir_cf_node_loop: {
         nir_loop *loop = nir_cf_node_as_loop(node);
         if (hk_cf_list_has_ray_query(&loop->body))
            return true;
         break;
      }
      default:
         break;
      }
   }
   return false;
}

/* A ray query addressed through a non-constant array index. Unrolling the
 * loop that produces the index lets query arrays split into fixed native
 * contexts, so such loops are left to the unroller. */
static bool
hk_cf_list_has_indexed_ray_query(struct exec_list *cf_list)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      switch (node->type) {
      case nir_cf_node_block:
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            if (instr->type != nir_instr_type_intrinsic ||
                !nir_intrinsic_is_ray_query(nir_instr_as_intrinsic(instr)->intrinsic))
               continue;
            nir_deref_instr *deref =
               nir_src_as_deref(nir_instr_as_intrinsic(instr)->src[0]);
            for (; deref && deref->deref_type != nir_deref_type_var;
                 deref = nir_deref_instr_parent(deref)) {
               if (deref->deref_type == nir_deref_type_array &&
                   !nir_src_is_const(deref->arr.index))
                  return true;
            }
         }
         break;
      case nir_cf_node_if: {
         nir_if *nif = nir_cf_node_as_if(node);
         if (hk_cf_list_has_indexed_ray_query(&nif->then_list) ||
             hk_cf_list_has_indexed_ray_query(&nif->else_list))
            return true;
         break;
      }
      case nir_cf_node_loop:
         if (hk_cf_list_has_indexed_ray_query(&nir_cf_node_as_loop(node)->body))
            return true;
         break;
      default:
         break;
      }
   }
   return false;
}

/* Keep loops that run whole ray queries. Unrolled, each iteration copies
 * the complete traversal. Opt-in (ASAHI_M5_RQ_KEEP_LOOPS) until its CTS
 * qualification is complete. */
static void
hk_keep_m5_ray_query_loops(struct exec_list *cf_list)
{
   foreach_list_typed(nir_cf_node, node, node, cf_list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         hk_keep_m5_ray_query_loops(&nif->then_list);
         hk_keep_m5_ray_query_loops(&nif->else_list);
      } else if (node->type == nir_cf_node_loop) {
         nir_loop *loop = nir_cf_node_as_loop(node);
         if (hk_cf_list_has_ray_query(&loop->body) &&
             !hk_cf_list_has_indexed_ray_query(&loop->body))
            loop->control = nir_loop_control_dont_unroll;
         hk_keep_m5_ray_query_loops(&loop->body);
      }
   }
}

static bool
hk_nir_has_ray_query(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      if (hk_cf_list_has_ray_query(&impl->body))
         return true;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_intersect_ray_gen1_agx)
               return true;
         }
      }
   }
   return false;
}

void
hk_preprocess_nir_internal(struct vk_physical_device *vk_pdev, nir_shader *nir)
{
   /* Matrices of flexible dimensions become 16x16 pieces, then tensor loads
    * and stores are lowered: they inline the decode functions they call,
    * which the rest of the lowering must then see. The pieces of a matrix
    * may pair K 64 deep, which the tensor loads then take. */
   if (agx_chip_g17_m5(((struct hk_physical_device *)vk_pdev)->dev.chip)) {
      bool split = false, pairing = false;
      NIR_PASS(split, nir, agx_nir_lower_cmat_flexible, &pairing);
      if (split) {
         NIR_PASS(_, nir, nir_opt_deref);
         NIR_PASS(_, nir, nir_opt_dce);
         NIR_PASS(_, nir, nir_remove_dead_variables,
                  nir_var_function_temp | nir_var_shader_temp, NULL);
      }
      NIR_PASS(_, nir, agx_nir_lower_cmat_tensor, pairing);
   }

   /* Then, before anything else lowers the shader: the functions of matrix
    * per-element operations and reductions are inlined here.
    */
   if (agx_chip_g17_m5(((struct hk_physical_device *)vk_pdev)->dev.chip))
      NIR_PASS(_, nir, agx_nir_lower_cmat_calls);

   if (agx_chip_g17_m5(((struct hk_physical_device *)vk_pdev)->dev.chip) &&
       debug_get_bool_option("ASAHI_M5_RQ_KEEP_LOOPS", false)) {
      nir_foreach_function_impl(impl, nir)
         hk_keep_m5_ray_query_loops(&impl->body);
   }

   /* Must lower before io to temps */
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_terminate_to_demote);
      NIR_PASS(_, nir, nir_lower_halt_to_return);
      NIR_PASS(_, nir, nir_lower_returns);
   }

   /* Unroll loops before lowering indirects via
    * nir_lower_io_vars_to_temporaries */
   UNUSED bool progress = false;
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);

   do {
      progress = false;
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_loop);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      struct nir_lower_sysvals_to_varyings_options sysvals_opts = {
         .point_coord = true,
         .primitive_id = nir->info.stage == MESA_SHADER_FRAGMENT,
      };

      nir_lower_sysvals_to_varyings(nir, &sysvals_opts);

   }

   NIR_PASS(_, nir, nir_lower_system_values);

   /* Gather info before preprocess_nir but after some general lowering, so
    * inputs_read and system_values_read are accurately set.
    */
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
            nir_shader_get_entrypoint(nir), nir_var_shader_out);

   NIR_PASS(_, nir, nir_lower_global_vars_to_local);

   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_split_struct_vars, nir_var_function_temp);

   /* G17P has no legacy USC constant-data upload.  Lower compute, vertex and
    * fragment constants to load_constant so the private native path can
    * address the immutable blob through its root descriptor
    * (hk_nir_lower_descriptors.c lower_load_constant).  hk_cmd_dispatch.c
    * uploads a compute blob per dispatch and hk_flush_shaders() publishes the
    * bound vertex and fragment blobs.  This is also Apple's shape: metal-nt
    * -arch applegpu_g17p compiles a fragment function's static constant
    * table lookup to an lddev from a uniform-supplied base, not to thread
    * private storage.
    *
    * This MUST run before agx_preprocess_nir.  That helper's first act is
    * nir_lower_vars_to_scratch with a 256-byte threshold (agx_compile.c:3852),
    * which claims exactly the large read-only arrays this pass exists to
    * capture -- and it claims them as scratch, which the AGX3 backend cannot
    * encode and hk_g17p_compute_shader_supported() refuses anyway.  Running
    * afterwards silently does nothing: there is no function_temp array left to
    * move, and nir_opt_large_constants only ever looks at function_temp
    * (nir_opt_large_constants.c:545-561).
    */
   if (agx_native_graphics(&((struct hk_physical_device *)vk_pdev)->dev) ||
       agx_chip_g17_m5(((struct hk_physical_device *)vk_pdev)->dev.chip)) {
      if (nir->info.stage == MESA_SHADER_COMPUTE) {
         NIR_PASS(_, nir, nir_opt_large_constants, NULL, 32);
      } else if ((nir->info.stage == MESA_SHADER_VERTEX ||
                  nir->info.stage == MESA_SHADER_FRAGMENT) &&
                 hk_has_scratch_sized_array(nir)) {
         /* Vertex and fragment arrays within the scratch threshold keep their
          * csel ladders, so leave shaders without a larger array untouched.
          * nir_opt_large_constants cannot lower a whole-array copy_deref. */
         NIR_PASS(_, nir, nir_lower_var_copies);
         NIR_PASS(_, nir, nir_opt_large_constants, NULL,
                  AGX_SCRATCH_ARRAY_THRESHOLD_B);
      }
   }

   /* Optimize but allow copies because we haven't lowered them yet */
   if (agx_native_graphics(&((struct hk_physical_device *)vk_pdev)->dev) &&
       nir->info.stage != MESA_SHADER_MESH)
      agx_preprocess_nir_native(
         nir, ((struct hk_physical_device *)vk_pdev)->dev.chip);
   else if (((struct hk_physical_device *)vk_pdev)->dev.chip == AGX_CHIP_G18G)
      agx_preprocess_nir_g18g(nir);
   else if (agx_chip_g17_m5(((struct hk_physical_device *)vk_pdev)->dev.chip))
      agx_preprocess_nir_m5(nir);
   else
      agx_preprocess_nir(nir);

   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_memcpy);
}

static void
hk_preprocess_nir(struct vk_physical_device *vk_pdev, nir_shader *nir,
                  UNUSED const struct vk_pipeline_robustness_state *rs)
{
   hk_preprocess_nir_internal(vk_pdev, nir);
   const enum agx_chip chip = ((struct hk_physical_device *)vk_pdev)->dev.chip;
   /* G17P task shaders run as compute kernels, and mesh workgroups map
    * invocations to lanes as compute does, so they take the same derivatives
    * (meshAndTaskShaderDerivatives). */
   const bool g17p_compute_derivatives =
      (((chip == AGX_CHIP_G17P || agx_chip_g17_m5(chip)) &&
        nir->info.stage == MESA_SHADER_COMPUTE) ||
       (chip == AGX_CHIP_G17P && (nir->info.stage == MESA_SHADER_TASK ||
                                  nir->info.stage == MESA_SHADER_MESH))) &&
      nir->info.derivative_group != DERIVATIVE_GROUP_NONE;
   nir_lower_compute_system_values_options csv_options = {
      /* Only vkCmdDispatchBase has a base workgroup. Task/mesh draws have
       * none, and their root is the graphics arm: root.cs.base_group
       * (byte 88) overlays root.draw.attrib_clamps[2..4] there. */
      .has_base_workgroup_id = nir->info.stage == MESA_SHADER_COMPUTE,
   };
   if (g17p_compute_derivatives &&
       nir->info.derivative_group == DERIVATIVE_GROUP_QUADS) {
      NIR_PASS(_, nir, agx_nir_lower_quad_derivative_ids);
      csv_options.shuffle_local_ids_for_quad_derivatives = true;
   }
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);
   if (g17p_compute_derivatives)
      NIR_PASS(_, nir, agx_nir_lower_compute_derivatives);
}

static void
hk_populate_vs_key(struct hk_vs_key *key,
                   const struct vk_graphics_pipeline_state *state)
{
   memset(key, 0, sizeof(*key));

   if (state && state->ia &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_IA_PRIMITIVE_TOPOLOGY)) {

      key->kill_psiz = state->ia->primitive_topology != MESA_PRIM_POINTS;
   }

   if (state && state->vi && !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI) &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI_BINDINGS_VALID)) {

      agx_fill_velem_keys(state->vi, ~0 /* compacted on use */, key->attribs);
      key->skip_prolog = true;
      key->static_strides =
         !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_VI_BINDING_STRIDES);

      if (!key->static_strides) {
         for (unsigned i = 0; i < ARRAY_SIZE(key->attribs); ++i) {
            key->attribs[i].stride = 0;
         }
      }
   }
}

/* The native AGX3B parts (G15, G17P) run a discarding shader's read-only
 * depth test in the prolog instead (hk_g17p_read_only_early_reject_in()),
 * with the main part's selector off. Whether a draw takes that path is
 * draw-time state; these pipeline states rule it out for every draw. */
static bool
hk_g17p_static_no_early_reject(const struct vk_graphics_pipeline_state *state)
{
   if (state->ms != NULL &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES) &&
       state->ms->rasterization_samples > 1)
      return true;

   const struct vk_depth_stencil_state *ds = state->ds;
   if (ds == NULL)
      return false;

   if (!BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE) &&
       !ds->depth.test_enable)
      return true;
   if (!BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_DEPTH_WRITE_ENABLE) &&
       ds->depth.write_enable)
      return true;
   if (!BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_DS_DEPTH_COMPARE_OP) &&
       ds->depth.compare_op == VK_COMPARE_OP_ALWAYS)
      return true;
   return !BITSET_TEST(state->dynamic,
                       MESA_VK_DYNAMIC_DS_DEPTH_BOUNDS_TEST_ENABLE) &&
          ds->depth.bounds_test.enable;
}

static void
hk_populate_fs_key(struct hk_fs_key *key,
                   const struct vk_graphics_pipeline_state *state,
                   enum agx_chip chip)
{
   memset(key, 0, sizeof(*key));

   if (state == NULL)
      return;

   /* AGX2 coverage updates are sample-count independent. G17P fragments build
    * their coverage word differently at one and at several samples. */
   if (agx_chip_native_agx3(chip) && state->ms != NULL &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES))
      key->g17p_samples = state->ms->rasterization_samples;

   if (agx_chip_g17_m5(chip) && state->ms != NULL &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES))
      key->m5_position_samples = state->ms->rasterization_samples;

   /* M5 has no read-only early reject to rule out. */
   if ((agx_chip_native_agx3(chip) || agx_chip_g17_m5(chip)) &&
       state->ms != NULL &&
       !BITSET_TEST(state->dynamic,
                    MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE) &&
       !state->ms->alpha_to_coverage_enable &&
       (agx_chip_g17_m5(chip) || hk_g17p_static_no_early_reject(state)))
      key->static_tests = true;

   if (chip == AGX_CHIP_G17P && state->ms != NULL &&
       !BITSET_TEST(state->dynamic,
                    MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE) &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES) &&
       !BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_MS_SAMPLE_MASK) &&
       state->ms->alpha_to_coverage_enable &&
       state->ms->rasterization_samples > 1 &&
       (state->ms->sample_mask &
        BITFIELD_MASK(state->ms->rasterization_samples)) ==
          BITFIELD_MASK(state->ms->rasterization_samples))
      key->static_a2c = true;

   if (state->pipeline_flags &
       VK_PIPELINE_CREATE_2_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT)
      key->zs_self_dep = true;

   /* We force per-sample interpolation whenever sampleShadingEnable is set
    * regardless of minSampleShading or rasterizationSamples.
    *
    * When sampleShadingEnable is set, few guarantees are made about the
    * location of interpolation of the inputs.  The only real guarantees are
    * that the inputs are interpolated within the pixel and that you get at
    * least `rasterizationSamples * minSampleShading` unique positions.
    * Importantly, it does not require that when `rasterizationSamples *
    * minSampleShading <= 1.0` that those positions are at the fragment
    * center.  Therefore, it's valid to just always do per-sample all the time.
    *
    * The one caveat here is that we have to be careful about gl_SampleMaskIn.
    * When `hk_fs_key::force_sample_shading = true` we also turn any reads of
    * gl_SampleMaskIn into `1 << gl_SampleID` because the hardware sample mask
    * is actually per-fragment, not per-pass.  We handle this by smashing
    * minSampleShading to 1.0 whenever gl_SampleMaskIn is read.
    */
   const struct vk_multisample_state *ms = state->ms;
   if (ms != NULL && ms->sample_shading_enable)
      key->force_sample_shading = true;

   /* A fragment library without output-interface state may acquire color
    * attachments later. Alpha-to-coverage also consumes color even without
    * an attachment, so require it to be statically disabled. */
   if ((chip == AGX_CHIP_G17P || agx_chip_g17_m5(chip)) &&
       state->rp &&
       (state->cal || BITSET_TEST(state->dynamic,
                                   MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP)) &&
       vk_render_pass_state_has_attachment_info(state->rp) &&
       state->rp->color_attachment_count == 0 && ms &&
       !BITSET_TEST(state->dynamic,
                    MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE) &&
       !ms->alpha_to_coverage_enable)
      key->discard_color_outputs = true;
}

static bool
hk_remove_unused_color_store(UNUSED nir_builder *b, nir_intrinsic_instr *intr,
                             UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   const unsigned location = nir_intrinsic_io_semantics(intr).location;
   if (location != FRAG_RESULT_COLOR &&
       (location < FRAG_RESULT_DATA0 || location > FRAG_RESULT_DATA7))
      return false;

   nir_instr_remove(&intr->instr);
   return true;
}

/* Whether a fragment shader may draw in a depth/stencil feedback loop that
 * only the draw-time state reveals: shader objects, a dynamic loop, or a
 * fragment library whose output interface comes later. A shader object's
 * binary may load on any device with the feature, so it follows the physical
 * device. Internal shaders have no features. */
static bool
hk_fs_zs_self_dep_dynamic(const struct vk_physical_device *pdev,
                          const struct vk_graphics_pipeline_state *state,
                          const struct vk_features *features)
{
   if (!features)
      return false;
   if (!state)
      return pdev->supported_features.attachmentFeedbackLoopDynamicState;

   return features->attachmentFeedbackLoopDynamicState &&
          (BITSET_TEST(state->dynamic,
                       MESA_VK_DYNAMIC_ATTACHMENT_FEEDBACK_LOOP_ENABLE) ||
           !(state->cal || BITSET_TEST(state->dynamic,
                                       MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP)));
}

static void
hk_prune_unused_fragment_colors(nir_shader *nir, const struct hk_fs_key *key)
{
   if (!key->discard_color_outputs)
      return;

   assert(nir->info.stage == MESA_SHADER_FRAGMENT && nir->info.io_lowered);
   /* An output read may still consume a preceding shader-local write. */
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_output) {
               const unsigned loc = nir_intrinsic_io_semantics(intr).location;
               if (loc == FRAG_RESULT_COLOR ||
                   (loc >= FRAG_RESULT_DATA0 && loc <= FRAG_RESULT_DATA7))
                  return;
            }
         }
      }
   }
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_remove_unused_color_store,
            nir_metadata_control_flow, NULL);

   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
   } while (progress);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
}

/* Every enabled device feature that changes a shader's code must be in this
 * key, since it is part of the cached shader's hash.
 */
enum hk_feature_key {
   HK_FEAT_CUSTOM_BORDER = BITFIELD_BIT(0),

   /* G17P only lowers sampled images to its native heaps when they cannot be
    * updated after bind. */
   HK_FEAT_SAMPLED_IMAGE_UPDATE_AFTER_BIND = BITFIELD_BIT(1),

   /* Select the vertex fetch robustness level of a monolithic vertex shader.
    */
   HK_FEAT_ROBUST_BUFFER_ACCESS = BITFIELD_BIT(2),
   HK_FEAT_ROBUST_BUFFER_ACCESS_2 = BITFIELD_BIT(3),
   HK_FEAT_PIPELINE_ROBUSTNESS = BITFIELD_BIT(4),

   /* G17P compiles against its native ray-query state only on a device that
    * enables a ray-tracing feature (hk_rt_features_enabled()). Ray-query
    * shaders target that state's uniform and SCS layout, and Poly's geometry
    * and evaluation lowering also runs extra passes with it, query or not.
    * The bit is never set where ray tracing is not advertised. */
   HK_FEAT_RAY_TRACING = BITFIELD_BIT(5),
   HK_FEAT_NULL_DESCRIPTOR = BITFIELD_BIT(6),

   /* The last pre-rasterization stage reads the clip z coefficient. */
   HK_FEAT_DEPTH_CLIP_CONTROL = BITFIELD_BIT(7),

   /* Not a device feature: the shader's set layouts are descriptor-buffer
    * layouts (hk_set_layouts_use_buffers()). Their contents can change until
    * the commands execute, so G17P keeps the lowerings that decode set
    * contents when a draw or dispatch is recorded away from them. The layout
    * hash carries the flag into the shader cache key. */
   HK_FEAT_DESCRIPTOR_BUFFER = BITFIELD_BIT(8),

   HK_FEAT_IMAGE_VIEW_MIN_LOD = BITFIELD_BIT(9),
};

/* What a dynamic-sample-count G17P fragment shader needs to compile its
 * multisample variants later: its NIR before any sample lowering and the
 * inputs of hk_compile_nir(). */
struct hk_fs_sample_source {
   simple_mtx_t lock;
   void *nir;
   size_t nir_size;
   union hk_key key;
   VkShaderCreateFlagsEXT flags;
   struct vk_pipeline_robustness_state robustness;
   enum hk_feature_key features;
   uint32_t set_layout_count;
   uint8_t static_set_mask;

   /* Variants usable through hk_fs_variant(), published with p_atomic_set
    * once complete. */
   uint32_t ready_mask;

   /* Variants built from this source; they belong to this object even when
    * it borrows the others from the binary cache. */
   uint32_t owned_mask;
};

/* What a shader compiled without custom-border emulation needs to compile
 * its emulating twin: its NIR after hk_lower_nir(), emulation included, and
 * the other inputs of hk_compile_shader_impl(). */
struct hk_border_source {
   simple_mtx_t lock;
   void *nir;
   size_t nir_size;
   union hk_key key;
   bool has_key;
   VkShaderCreateFlagsEXT flags;
   VkShaderStageFlags next_stage_mask;
   struct vk_pipeline_robustness_state robustness;
   enum hk_feature_key features;
   uint32_t set_layout_count;
   uint8_t static_set_mask;

   /* Built by hk_border_twin() and owned by this source; published with
    * p_atomic_set once complete. */
   struct hk_api_shader *twin;
};

/* A fragment twin built for one G17P mesh per-primitive record layout. */
struct hk_mesh_fs_twin {
   uint8_t comps[64];
   struct hk_api_shader *shader;
   struct hk_mesh_fs_twin *next;
};

static enum hk_feature_key
hk_make_feature_key(const struct vk_features *feats)
{
   /* Shader objects and internal shaders have no features to go by. A shader
    * object binary must not depend on the features its device enabled, so
    * assume every state they can draw with. */
   if (!feats)
      return HK_FEAT_CUSTOM_BORDER | HK_FEAT_DEPTH_CLIP_CONTROL |
             HK_FEAT_IMAGE_VIEW_MIN_LOD;

   return (feats->customBorderColors ? HK_FEAT_CUSTOM_BORDER : 0) |
          (feats->descriptorBindingSampledImageUpdateAfterBind
              ? HK_FEAT_SAMPLED_IMAGE_UPDATE_AFTER_BIND
              : 0) |
          (feats->robustBufferAccess ? HK_FEAT_ROBUST_BUFFER_ACCESS : 0) |
          (feats->robustBufferAccess2 ? HK_FEAT_ROBUST_BUFFER_ACCESS_2 : 0) |
          (feats->pipelineRobustness ? HK_FEAT_PIPELINE_ROBUSTNESS : 0) |
          (hk_rt_features_enabled(feats) ? HK_FEAT_RAY_TRACING : 0) |
          (feats->nullDescriptor ? HK_FEAT_NULL_DESCRIPTOR : 0) |
          (feats->depthClipControl ? HK_FEAT_DEPTH_CLIP_CONTROL : 0) |
          (feats->minLod ? HK_FEAT_IMAGE_VIEW_MIN_LOD : 0);
}

/*
 * Whether no fragment shader can ever consume the user varyings of this
 * pipeline's pre-rasterization stages. Fragment shader state is part of every
 * complete pipeline that may rasterize, so a missing fragment stage is final
 * there. A pre-rasterization library has no fragment shader state and may be
 * linked with a fragment shader library later.
 *
 * The runtime does not keep VK_PIPELINE_CREATE_2_LIBRARY_BIT_KHR in
 * vk_graphics_pipeline_state::pipeline_flags, so look for the state group
 * instead: fragment shader state always includes the input attachment map.
 */
static bool
hk_pipeline_has_no_fragment_shader(const struct vk_graphics_pipeline_state *state)
{
   return state && !(state->shader_stages & VK_SHADER_STAGE_FRAGMENT_BIT) &&
          (state->ial != NULL ||
           BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_INPUT_ATTACHMENT_MAP));
}

/* Whether the last pre-rasterization stage applies agx_nir_lower_clip_m1_1.
 * Every shader of a device that enabled depthClipControl needs it, internal
 * ones included. A shader object applies it wherever the feature exists, so
 * its binary is the same whether or not its device enabled the feature; the
 * coefficient is 0 unless negativeOneToOne is set. */
static bool
hk_clip_m1_1(const struct hk_device *dev, enum hk_feature_key features)
{
   return dev->vk.enabled_features.depthClipControl ||
          ((features & HK_FEAT_DEPTH_CLIP_CONTROL) &&
           dev->vk.physical->supported_features.depthClipControl);
}

/* Whether a mesh shader can draw with negativeOneToOne. Shader objects and
 * dynamic state cannot rule it out. */
static bool
hk_mesh_clip_m1_1(bool enabled, const struct vk_graphics_pipeline_state *state)
{
   return enabled &&
          (!state || !state->vp ||
           BITSET_TEST(state->dynamic,
                       MESA_VK_DYNAMIC_VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) ||
           state->vp->depth_clip_negative_one_to_one);
}

static void
hk_hash_graphics_state(struct vk_physical_device *device,
                       const struct vk_graphics_pipeline_state *state,
                       const struct vk_features *features,
                       VkShaderStageFlags stages, blake3_hash blake3_out)
{
   struct mesa_blake3 blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);
   /* A monolithic pipeline compiles every stage in one partition, so the
    * fragment key must be hashed alongside the vertex key. */
   if (state && (stages & VK_SHADER_STAGE_VERTEX_BIT)) {
      struct hk_vs_key key;
      hk_populate_vs_key(&key, state);
      _mesa_blake3_update(&blake3_ctx, &key, sizeof(key));
   }

   if (state && (stages & VK_SHADER_STAGE_FRAGMENT_BIT)) {
      struct hk_fs_key key;
      hk_populate_fs_key(&key, state,
                         container_of(device, struct hk_physical_device, vk)
                            ->dev.chip);
      key.dynamic_zs_self_dep =
         !key.zs_self_dep &&
         hk_fs_zs_self_dep_dynamic(device, state, features);
      _mesa_blake3_update(&blake3_ctx, &key, sizeof(key));
   }

   if (state &&
       (stages & (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT |
                  VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
                  VK_SHADER_STAGE_GEOMETRY_BIT))) {
      const bool is_multiview = state->mv->view_mask != 0;
      _mesa_blake3_update(&blake3_ctx, &is_multiview, sizeof(is_multiview));
   }

   /* G17P removes user varyings from fragmentless pipelines, so the same
    * pre-rasterization shader compiles differently for a library. */
   if (agx_native_graphics(&((struct hk_physical_device *)device)->dev) &&
       !(stages & VK_SHADER_STAGE_FRAGMENT_BIT)) {
      const bool no_fragment_shader = hk_pipeline_has_no_fragment_shader(state);
      _mesa_blake3_update(&blake3_ctx, &no_fragment_shader,
                          sizeof(no_fragment_shader));
   }

   /* M5 removes user varyings from the mesh shader of a fragmentless
    * pipeline (hk_m5_prune_fragmentless_mesh_outputs), so that mesh shader
    * compiles differently for a pre-rasterization library. */
   if (agx_chip_g17_m5(((struct hk_physical_device *)device)->dev.chip) &&
       (stages & VK_SHADER_STAGE_MESH_BIT_EXT) &&
       !(stages & VK_SHADER_STAGE_FRAGMENT_BIT)) {
      const bool no_fragment_shader = hk_pipeline_has_no_fragment_shader(state);
      _mesa_blake3_update(&blake3_ctx, &no_fragment_shader,
                          sizeof(no_fragment_shader));
   }

   /* M5 triangle mesh point-size removal is a pipeline-only lowering. */
   if (agx_chip_g17_m5(((struct hk_physical_device *)device)->dev.chip) &&
       (stages & VK_SHADER_STAGE_MESH_BIT_EXT)) {
      const bool pipeline_mesh = state != NULL;
      _mesa_blake3_update(&blake3_ctx, &pipeline_mesh, sizeof(pipeline_mesh));
   }

   enum hk_feature_key feature_key = hk_make_feature_key(features);
   _mesa_blake3_update(&blake3_ctx, &feature_key, sizeof(feature_key));

   if (stages & VK_SHADER_STAGE_MESH_BIT_EXT) {
      const bool clip_m1_1 =
         hk_mesh_clip_m1_1(features && features->depthClipControl, state);
      _mesa_blake3_update(&blake3_ctx, &clip_m1_1, sizeof(clip_m1_1));
   }

   /* Whether G17P and M5 shaders compile the custom-border emulation in
    * (hk_compile_shader_impl): a cached shader was compiled the same way. */
   struct hk_physical_device *pdev =
      container_of(device, struct hk_physical_device, vk);
   if (agx_native_graphics(&pdev->dev) || agx_chip_g17_m5(pdev->dev.chip)) {
      const uint32_t border = p_atomic_read(&pdev->border_emulation);
      _mesa_blake3_update(&blake3_ctx, &border, sizeof(border));
   }

   _mesa_blake3_final(&blake3_ctx, blake3_out);
}

static nir_def *
bounds_check(nir_builder *b, nir_def *data, nir_def *offs, nir_def *bound)
{
   if (data->bit_size == 32 && data->num_components == 1) {
      return nir_bounds_agx(b, data, offs, bound);
   } else {
      /* TODO: Optimize */
      return nir_bcsel(b, nir_uge(b, bound, offs), data,
                       nir_imm_zero(b, data->num_components, data->bit_size));
   }
}

static bool
is_op(nir_scalar s, nir_op op)
{
   return nir_scalar_is_alu(s) && nir_scalar_alu_op(s) == op;
}

/* Prove low offset bits from SSA, independently of the final-address
 * alignment metadata: an aligned address does not imply an aligned offset
 * when its base is unaligned. Fail closed on phis, calls and unknown values. */
static bool
m5_offset_aligned(nir_scalar s, uint32_t mask, unsigned depth)
{
   if (!mask)
      return true;
   if (nir_scalar_is_const(s))
      return (nir_scalar_as_uint(s) & mask) == 0;
   if (!depth || s.def->bit_size != 32 || !nir_scalar_is_alu(s))
      return false;
   const nir_op op = nir_scalar_alu_op(s);
   if (op == nir_op_mov)
      return m5_offset_aligned(nir_scalar_chase_alu_src(s, 0), mask, depth - 1);
   nir_scalar x = nir_scalar_chase_alu_src(s, 0);
   nir_scalar y;
   switch (op) {
   case nir_op_iadd: case nir_op_isub: case nir_op_ior: case nir_op_ixor:
      y = nir_scalar_chase_alu_src(s, 1);
      return m5_offset_aligned(x, mask, depth - 1) &&
             m5_offset_aligned(y, mask, depth - 1);
   case nir_op_imul: case nir_op_amul: case nir_op_iand:
      y = nir_scalar_chase_alu_src(s, 1);
      return m5_offset_aligned(x, mask, depth - 1) ||
             m5_offset_aligned(y, mask, depth - 1);
   case nir_op_ishl:
      y = nir_scalar_chase_alu_src(s, 1);
      return nir_scalar_is_const(y) &&
             m5_offset_aligned(x, mask >> (nir_scalar_as_uint(y) & 31),
                               depth - 1);
   case nir_op_bcsel:
      return m5_offset_aligned(nir_scalar_chase_alu_src(s, 1), mask, depth - 1) &&
             m5_offset_aligned(nir_scalar_chase_alu_src(s, 2), mask, depth - 1);
   default:
      return false;
   }
}

static nir_def *
check_in_bounds(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_def *offset = intr->src[1].ssa;
   nir_def *bound = intr->src[2].ssa;

   unsigned bit_size = intr->def.bit_size;
   assert(bit_size >= 8 && bit_size % 8 == 0);
   unsigned byte_size = bit_size / 8;
   unsigned load_size = byte_size * intr->num_components;

   /* Try to bounds check in terms of elements */
   nir_scalar offset_s = nir_scalar_resolved(offset, 0);
   if (is_op(offset_s, nir_op_amul)) {
      nir_scalar srcs[] = {nir_scalar_chase_alu_src(offset_s, 0),
                           nir_scalar_chase_alu_src(offset_s, 1)};
      unsigned i = nir_scalar_is_const(srcs[0]) ? 1 : 0;
      if (nir_scalar_is_const(srcs[1 - i]) &&
          nir_scalar_as_uint(srcs[1 - i]) == load_size) {

         nir_def *index = nir_mov_scalar(b, srcs[i]);
         return nir_ult(b, index, nir_udiv_imm(b, bound, load_size));
      }
   }

   /* Byte offsets below bound - (size - 1) fit the whole load. The
    * saturating subtraction rejects every offset when the bound is smaller
    * than the load. Loads of the same width can share the adjusted bound.
    */
   if (offset->bit_size == 32 && bound->bit_size == 32)
      return nir_ult(b, offset,
                     nir_usub_sat(b, bound, nir_imm_int(b, load_size - 1)));

   /* Otherwise bounds check in bytes */
   nir_def *sat_offset = nir_umin_imm(b, offset, UINT32_MAX - (load_size - 1));
   return nir_ult(b, nir_iadd_imm(b, sat_offset, load_size - 1), bound);
}

struct m5_bounded_offset {
   nir_scalar term;
   uint32_t constant;
   uint32_t scale;
   bool valid;
};

static struct m5_bounded_offset
m5_bounded_offset(nir_scalar s, unsigned depth)
{
   if (nir_scalar_is_const(s))
      return (struct m5_bounded_offset){.constant = nir_scalar_as_uint(s), .valid = true};
   if (depth && is_op(s, nir_op_iadd)) {
      struct m5_bounded_offset x = m5_bounded_offset(nir_scalar_chase_alu_src(s, 0), depth - 1);
      struct m5_bounded_offset y = m5_bounded_offset(nir_scalar_chase_alu_src(s, 1), depth - 1);
      if (!x.valid || !y.valid || (x.term.def && y.term.def))
         return (struct m5_bounded_offset){0};
      return (struct m5_bounded_offset){.term = x.term.def ? x.term : y.term,
         .constant = x.constant + y.constant,
         .scale = x.term.def ? x.scale : y.scale, .valid = true};
   }
   if (depth && is_op(s, nir_op_ior)) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      if (nir_scalar_is_const(a) || nir_scalar_is_const(b)) {
         uint32_t k = nir_scalar_as_uint(nir_scalar_is_const(a) ? a : b);
         nir_scalar term = nir_scalar_is_const(a) ? b : a;
         uint32_t mask = k ? UINT32_MAX >> (32 - util_last_bit(k)) : 0;
         if (m5_offset_aligned(term, mask, depth)) {
            struct m5_bounded_offset value = m5_bounded_offset(term, depth - 1);
            value.constant += k;
            return value;
         }
      }
   }
   if (depth && is_op(s, nir_op_ishl)) {
      nir_scalar shift = nir_scalar_chase_alu_src(s, 1);
      if (nir_scalar_is_const(shift)) {
         unsigned bits = nir_scalar_as_uint(shift) & 31;
         struct m5_bounded_offset value = m5_bounded_offset(
            nir_scalar_chase_alu_src(s, 0), depth - 1);
         value.constant <<= bits;
         value.scale <<= bits;
         return value;
      }
   }
   if (depth && (is_op(s, nir_op_amul) || is_op(s, nir_op_imul))) {
      nir_scalar a = nir_scalar_chase_alu_src(s, 0);
      nir_scalar b = nir_scalar_chase_alu_src(s, 1);
      if (nir_scalar_is_const(a) || nir_scalar_is_const(b)) {
         uint32_t k = nir_scalar_as_uint(nir_scalar_is_const(a) ? a : b);
         struct m5_bounded_offset value = m5_bounded_offset(
            nir_scalar_is_const(a) ? b : a, depth - 1);
         value.constant *= k;
         value.scale *= k;
         return value;
      }
   }
   return (struct m5_bounded_offset){.term = s, .scale = 1, .valid = true};
}

/* Keep memory/control barriers intact even for readonly loads. */
static bool
m5_bounded_can_move(nir_instr *first, nir_instr *last)
{
   for (nir_instr *instr = nir_instr_next(first); instr != last;
        instr = nir_instr_next(instr)) {
      if (instr->type == nir_instr_type_intrinsic &&
          !nir_intrinsic_can_reorder(nir_instr_as_intrinsic(instr)))
         return false;
      if (instr->type == nir_instr_type_call || instr->type == nir_instr_type_jump)
         return false;
   }
   return true;
}

struct m5_word_read {
   nir_intrinsic_instr *load;
   /* The 64-bit base and, for a robust read, the descriptor's byte size. */
   nir_def *base, *bound;
   /* The read's 32-bit byte offset from base. */
   nir_def *offset_def;
   struct m5_bounded_offset offset;
};

static int
m5_word_read_cmp(const void *pa, const void *pb)
{
   const struct m5_word_read *a = pa, *b = pb;
   const nir_intrinsic_instr *x = a->load, *y = b->load;
#define M5_CMP(u, v) if ((u) != (v)) return (u) < (v) ? -1 : 1
   M5_CMP(x->intrinsic, y->intrinsic);
   M5_CMP(a->base->index, b->base->index);
   M5_CMP(a->bound ? a->bound->index + 1 : 0, b->bound ? b->bound->index + 1 : 0);
   M5_CMP(nir_intrinsic_access(x), nir_intrinsic_access(y));
   M5_CMP(a->offset.term.def ? a->offset.term.def->index + 1 : 0,
          b->offset.term.def ? b->offset.term.def->index + 1 : 0);
   M5_CMP(a->offset.term.comp, b->offset.term.comp);
   M5_CMP(a->offset.scale, b->offset.scale);
   M5_CMP(a->offset.constant, b->offset.constant);
   M5_CMP(x->instr.index, y->instr.index);
#undef M5_CMP
   return 0;
}


static bool
m5_word_reads_same_record(const struct m5_word_read *a, const struct m5_word_read *b)
{
   return a->load->intrinsic == b->load->intrinsic && a->base == b->base &&
          a->bound == b->bound &&
          nir_intrinsic_access(a->load) == nir_intrinsic_access(b->load) &&
          a->offset.term.def == b->offset.term.def &&
          a->offset.term.comp == b->offset.term.comp &&
          a->offset.scale == b->offset.scale;
}

static bool
m5_word_reads_adjacent(const struct m5_word_read *a, const struct m5_word_read *b)
{
   return m5_word_reads_same_record(a, b) &&
          b->offset.constant == a->offset.constant + 4;
}

/* Match a 64-bit address base + u2u64(offset) with a 32-bit offset. */
static bool
m5_split_address(nir_def *addr, nir_def **base, nir_def **offset)
{
   nir_scalar s = nir_scalar_resolved(addr, 0);
   if (addr->bit_size != 64 || !is_op(s, nir_op_iadd))
      return false;
   for (unsigned i = 0; i < 2; i++) {
      nir_scalar u = nir_scalar_chase_alu_src(s, i);
      nir_scalar other = nir_scalar_chase_alu_src(s, 1 - i);
      if (is_op(u, nir_op_u2u64) && other.def->num_components == 1) {
         nir_scalar off = nir_scalar_chase_alu_src(u, 0);
         if (off.def->bit_size == 32 && off.def->num_components == 1 &&
             !is_op(other, nir_op_u2u64)) {
            *base = other.def;
            *offset = off.def;
            return true;
         }
      }
   }
   return false;
}

/* M5: a structure read field by field (a light record in a loop) issues one
 * load per 32-bit word, each with its own address. Merge the reads of up to
 * four consecutive words at one base and offset term into one vector load.
 *
 * Buffer reads address base + u2u64(X + c) with a 32-bit offset, so the words
 * of one vector load differ from their separate reads only where X + c wraps
 * 2^32. Such an offset lies past the end of any buffer: undefined without
 * robustness, and zeroed by the robust forms here. A robust read (soft fault:
 * load, then zero the word when its end lies past the descriptor's byte size)
 * keeps each word's own check:
 *
 *    word j of a load at X: bounds_agx(v.j, uadd_sat(X, 4 * (j + 1)), size)
 *
 * which is exact, except that a word whose offset wraps is now zeroed instead
 * of read at its wrapped offset. Only such a word's address differs from its
 * separate read, and soft fault covers it. */
static bool
m5_merge_buffer_words(nir_shader *nir)
{
   bool any = false;
   struct util_dynarray reads;
   util_dynarray_init(&reads, NULL);
   nir_foreach_function_impl(impl, nir) {
      bool progress = false;
      nir_index_ssa_defs(impl);
      nir_index_instrs(impl);
      nir_foreach_block(block, impl) {
         util_dynarray_clear(&reads);
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *in = nir_instr_as_intrinsic(instr);
            if (in->num_components != 1 || in->def.bit_size != 32 ||
                !nir_intrinsic_can_reorder(in) || nir_intrinsic_align(in) < 4)
               continue;
            struct m5_word_read r = {.load = in};
            switch (in->intrinsic) {
            case nir_intrinsic_load_global_bounded:
            case nir_intrinsic_load_global_constant_bounded:
               r.bound = in->src[2].ssa;
               FALLTHROUGH;
            case nir_intrinsic_load_global_constant_offset:
               r.base = in->src[0].ssa;
               r.offset_def = in->src[1].ssa;
               break;
            case nir_intrinsic_load_global:
            case nir_intrinsic_load_global_constant:
               if (!m5_split_address(in->src[0].ssa, &r.base, &r.offset_def))
                  continue;
               break;
            default:
               continue;
            }
            if (r.offset_def->bit_size != 32)
               continue;
            r.offset = m5_bounded_offset(nir_scalar_resolved(r.offset_def, 0), 12);
            if (r.offset.valid)
               util_dynarray_append(&reads, r);
         }

         unsigned n = util_dynarray_num_elements(&reads, struct m5_word_read);
         if (n < 2)
            continue;
         struct m5_word_read *r = reads.data;
         qsort(r, n, sizeof(*r), m5_word_read_cmp);

         for (unsigned g = 0; g < n;) {
            /* The reads of one record: one base, size and offset term. */
            unsigned g_end = g + 1, earliest = g;
            while (g_end < n && m5_word_reads_same_record(&r[g], &r[g_end])) {
               if (r[g_end].load->instr.index < r[earliest].load->instr.index)
                  earliest = g_end;
               g_end++;
            }

            /* Split the record into loads of up to four consecutive words. */
            unsigned units[64][2], nr_units = 0;
            for (unsigned i = g; i < g_end && nr_units < ARRAY_SIZE(units);) {
               unsigned count = 1;
               while (i + count < g_end && count < 4 &&
                      m5_word_reads_adjacent(&r[i + count - 1], &r[i + count]))
                  count++;
               while (count > 1) {
                  unsigned first = i, last = i;
                  for (unsigned j = i + 1; j < i + count; j++) {
                     if (r[j].load->instr.index < r[first].load->instr.index)
                        first = j;
                     if (r[j].load->instr.index > r[last].load->instr.index)
                        last = j;
                  }
                  if (m5_bounded_can_move(&r[first].load->instr, &r[last].load->instr))
                     break;
                  count = 1;
               }
               units[nr_units][0] = i;
               units[nr_units][1] = count;
               nr_units++;
               i += count;
            }
            const unsigned covered = units[nr_units - 1][0] +
                                     units[nr_units - 1][1] - g;

            bool merged = false;
            for (unsigned u = 0; u < nr_units; u++)
               merged |= units[u][1] > 1;
            if (covered != g_end - g || !merged) {
               g = g_end;
               continue;
            }

            nir_builder b = nir_builder_at(nir_before_instr(&r[earliest].load->instr));
            nir_def *x_min = nir_iadd_imm(&b, r[earliest].offset_def,
               (uint32_t)(r[g].offset.constant - r[earliest].offset.constant));

            for (unsigned u = 0; u < nr_units; u++) {
               const unsigned i = units[u][0], count = units[u][1];
               if (count < 2)
                  continue;

               unsigned first = i;
               for (unsigned j = i + 1; j < i + count; j++) {
                  if (r[j].load->instr.index < r[first].load->instr.index)
                     first = j;
               }
               const uint32_t delta = r[i].offset.constant - r[g].offset.constant;
               b.cursor = nir_before_instr(&r[first].load->instr);

               nir_intrinsic_instr *low = r[i].load;
               const unsigned align_mul = nir_intrinsic_align_mul(low);
               const unsigned align_offset = nir_intrinsic_align_offset(low);
               enum gl_access_qualifier access = nir_intrinsic_access(low);
               const bool constant =
                  low->intrinsic == nir_intrinsic_load_global_constant ||
                  low->intrinsic == nir_intrinsic_load_global_constant_offset ||
                  low->intrinsic == nir_intrinsic_load_global_constant_bounded;
               if (r[i].bound && constant)
                  access |= ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;

               nir_def *x = nir_iadd_imm(&b, x_min, delta);
               nir_def *addr = nir_iadd(&b, r[i].base, nir_u2u64(&b, x));
               nir_def *v = constant
                  ? nir_load_global_constant(&b, count, 32, addr,
                       .align_mul = align_mul, .align_offset = align_offset,
                       .access = access)
                  : nir_load_global(&b, count, 32, addr,
                       .align_mul = align_mul, .align_offset = align_offset,
                       .access = access);
               for (unsigned j = 0; j < count; j++) {
                  nir_def *word = nir_channel(&b, v, j);
                  if (r[i].bound) {
                     nir_def *end = nir_uadd_sat(&b, x, nir_imm_int(&b, 4 * (j + 1)));
                     word = nir_bounds_agx(&b, word, end, r[i].bound);
                  }
                  nir_def_replace(&r[i + j].load->def, word);
               }
            }
            progress = true;
            g = g_end;
         }
      }
      any |= nir_progress(progress, impl, nir_metadata_control_flow);
   }
   util_dynarray_fini(&reads);
   return any;
}

static nir_def *
bound_offset(nir_builder *b, nir_def *valid, nir_scalar offset)
{
   /* If we can, clamp the source of an amul instead of the result, to remain
    * compatible with the hardware address mode.
    */
   if (is_op(offset, nir_op_amul)) {
      nir_scalar srcs[] = {
         nir_scalar_chase_alu_src(offset, 0),
         nir_scalar_chase_alu_src(offset, 1),
      };
      unsigned i = nir_scalar_is_const(srcs[0]) ? 1 : 0;
      nir_def *x = nir_mov_scalar(b, srcs[i]);
      nir_def *y = nir_mov_scalar(b, srcs[1 - i]);

      return nir_amul(b, nir_bcsel(b, valid, x, nir_imm_int(b, 0)), y);
   }

   /* Similar case for when there's an addition (chain) in the way */
   if (is_op(offset, nir_op_iadd)) {
      nir_scalar x = nir_scalar_chase_alu_src(offset, 0);
      nir_scalar y = nir_scalar_chase_alu_src(offset, 1);

      if (is_op(x, nir_op_amul) || is_op(y, nir_op_amul)) {
         return nir_iadd(b, bound_offset(b, valid, x),
                         bound_offset(b, valid, y));
      }
   }

   nir_def *def = nir_mov_scalar(b, offset);

   /* If the offset fits within the zero page, clamping is pointless */
   if (nir_scalar_is_const(offset) &&
       (nir_scalar_as_uint(offset) + 16) < AGX_ZERO_PAGE_SIZE)
      return def;

   /* Otherwise, fallback on clamping the offset */
   return nir_bcsel(b, valid, def, nir_imm_int(b, 0));
}

static void
lower_load_global_bounded(nir_builder *b, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *base = intr->src[0].ssa;
   nir_def *offset = intr->src[1].ssa;

   /* Bounds check compatibly with the hardware address mode */
   nir_def *valid = check_in_bounds(b, intr);
   base = nir_bcsel(b, valid, base, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
   offset = bound_offset(b, valid, nir_scalar_resolved(offset, 0));

   enum gl_access_qualifier access = nir_intrinsic_access(intr);

   if (intr->intrinsic == nir_intrinsic_load_global_constant_bounded) {
      access |= ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER;
   }

   nir_def *val = nir_load_global(
      b, intr->def.num_components, intr->def.bit_size,
      nir_iadd(b, base, nir_u2u64(b, offset)),
      .align_mul = nir_intrinsic_align_mul(intr),
      .align_offset = nir_intrinsic_align_offset(intr), .access = access);

   nir_def_replace(&intr->def, val);
}

struct global_load_lowering {
   bool soft_fault;
   /* G17P and M5: storage-buffer loads use the soft-fault form as well:
    * load, then zero the value when the access is out of bounds. */
   bool storage_result_select;
};

static bool
lower_physical_sparse_load(nir_builder *b, nir_intrinsic_instr *intr,
                           UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;

   nir_deref_instr *leaf = nir_src_as_deref(intr->src[0]);
   if (!leaf || leaf->modes != nir_var_mem_global)
      return false;

   nir_deref_path path;
   nir_deref_path_init(&path, leaf, NULL);
   nir_deref_instr *root = path.path[0];
   if (root->deref_type != nir_deref_type_cast ||
       root->parent.ssa->num_components != 1 ||
       root->parent.ssa->bit_size != 64) {
      nir_deref_path_finish(&path);
      return false;
   }

   /* Generated read-only aliases are not coherent with the writable VA
    * within an M5 dispatch. Only unbound sparse pages need substituted
    * storage; resident reads must keep the application's original VA. */
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *table = nir_load_root_agx(b);
#define MAP_FIELD(field) nir_load_global_constant( \
   b, 1, 64, nir_iadd_imm(b, table, hk_root_descriptor_offset(field)), \
   .align_mul = 8)
   nir_def *map = MAP_FIELD(physical_sparse_map);
   nir_def *heap_base = MAP_FIELD(physical_sparse_base);
   nir_def *heap_size = MAP_FIELD(physical_sparse_size);
#undef MAP_FIELD
   nir_def *address = path.path[0]->parent.ssa;
   for (unsigned i = 1; path.path[i]; i++)
      address = nir_explicit_io_address_from_deref(
         b, path.path[i], address, nir_address_format_64bit_global);
   const unsigned bits = intr->def.bit_size;
   const unsigned memory_bits = bits == 1 ? 32 : bits;
   const unsigned units = memory_bits == 64 ? 2 : 1;
   const unsigned unit_bits = MIN2(memory_bits, 32);
   const unsigned stride = glsl_get_explicit_stride(leaf->type);
   uint32_t align_mul, align_offset;
   if (!nir_get_explicit_deref_align(leaf, true, &align_mul, &align_offset)) {
      align_mul = memory_bits / 8;
      align_offset = 0;
   }
   nir_def *components[NIR_MAX_VEC_COMPONENTS];
   nir_def *direct[NIR_MAX_VEC_COMPONENTS];
   nir_def *count = nir_load_global_constant(b, 1, 32, map, .align_mul = 4);
   nir_if *sparse_if = nir_push_if(b, nir_ine_imm(b, count, 0));
   for (unsigned c = 0; c < intr->def.num_components; c++) {
      nir_def *parts[2];
      for (unsigned u = 0; u < units; u++) {
         const unsigned byte = c * (stride ? stride : memory_bits / 8) +
                               u * unit_bits / 8;
         nir_def *ptr = nir_iadd_imm(b, address, byte);
         nir_def *relative = nir_isub(b, ptr, heap_base);
         nir_def *valid = nir_ult(b, relative, heap_size);
         nir_def *page = nir_ushr_imm(b, relative, 14);
         nir_def *word = nir_bcsel(b, valid, nir_ushr_imm(b, page, 5),
                                   nir_imm_int64(b, 0));
         nir_def *entry = nir_load_global_constant(
            b, 1, 32, nir_iadd_imm(b, nir_iadd(b, map, nir_ishl_imm(b, word, 2)), 16),
            .align_mul = 4);
         nir_def *mask = nir_ishl(b, nir_imm_int(b, 1),
            nir_iand_imm(b, nir_u2u32(b, page), 31));
         nir_def *nonresident = nir_iand(b, valid,
            nir_ine_imm(b, nir_iand(b, entry, mask), 0));
         nir_def *read_address = nir_bcsel(b, nonresident,
            nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS), ptr);
         parts[u] = nir_load_global(b, 1, unit_bits, read_address,
            .align_mul = MIN2(align_mul, unit_bits / 8),
            .align_offset = (align_offset + byte) % MIN2(align_mul, unit_bits / 8),
            .access = nir_intrinsic_access(intr));
      }
      components[c] = units == 2
         ? nir_pack_64_2x32(b, nir_vec2(b, parts[0], parts[1]))
         : bits == 1 ? nir_ine_imm(b, parts[0], 0) : parts[0];
   }
   nir_push_else(b, sparse_if);
   for (unsigned c = 0; c < intr->def.num_components; c++) {
      const unsigned byte = c * (stride ? stride : memory_bits / 8);
      direct[c] = nir_load_global(b, 1, memory_bits,
         nir_iadd_imm(b, address, byte),
         .align_mul = align_mul,
         .align_offset = (align_offset + byte) % align_mul,
         .access = nir_intrinsic_access(intr));
      if (bits == 1)
         direct[c] = nir_ine_imm(b, direct[c], 0);
   }
   nir_pop_if(b, sparse_if);
   for (unsigned c = 0; c < intr->def.num_components; c++)
      components[c] = nir_if_phi(b, components[c], direct[c]);
   nir_def_rewrite_uses(&intr->def,
                        nir_vec(b, components, intr->def.num_components));
   nir_instr_remove(&intr->instr);
   nir_deref_path_finish(&path);
   return true;
}


/*
 * Sparse residency for a texel buffer on G13 and G14.
 *
 * hk_buffer_view.c builds a texel buffer as a plain linear 2D texture over
 * the buffer's read-only alias, so the texture unit has no residency to
 * report and every sparse texel fetch or read comes back resident.  Take the
 * residency of the texel's page from the device-wide map of non-resident
 * sparse pages instead.  The data needs no masking: unbound pages read as
 * zero through the read-only alias.
 */
static bool
lower_sparse_texel_buffer_residency(nir_builder *b, nir_instr *instr,
                                    void *data)
{
   const struct hk_device *dev = data;
   nir_def *handle, *coord, *def;

   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      if (!tex->is_sparse || tex->sampler_dim != GLSL_SAMPLER_DIM_BUF)
         return false;

      const int handle_idx =
         nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
      const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      if (handle_idx < 0 || coord_idx < 0)
         return false;

      handle = tex->src[handle_idx].src.ssa;
      coord = tex->src[coord_idx].src.ssa;
      def = &tex->def;
   } else if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic != nir_intrinsic_bindless_image_sparse_load ||
          nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_BUF)
         return false;

      handle = intr->src[0].ssa;
      coord = intr->src[1].ssa;
      def = &intr->def;
   } else {
      return false;
   }

   /* The lookup has control flow, so build it ahead of the access: the code
    * that patches the result below must share the access's block. */
   b->cursor = nir_before_instr(instr);

   nir_def *table = nir_load_root_agx(b);
#define ROOT_FIELD(field) nir_load_global_constant( \
   b, 1, 64, nir_iadd_imm(b, table, hk_root_descriptor_offset(field)), \
   .align_mul = 8)
   nir_def *map = ROOT_FIELD(physical_sparse_map);
   nir_def *heap_base = ROOT_FIELD(physical_sparse_base);
   nir_def *heap_size = ROOT_FIELD(physical_sparse_size);
#undef ROOT_FIELD

   nir_def *nonresident = libagx_buffer_texture_nonresident(
      b, nir_load_from_texture_handle_agx(b, handle),
      nir_channel(b, coord, 0), map, heap_base, heap_size,
      nir_imm_int64(b, dev->dev.sparse_ro_offset));

   b->cursor = nir_after_instr(instr);

   /* The residency code is the last component; bit 0 set is non-resident. */
   const unsigned code_idx = def->num_components - 1;
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned c = 0; c < code_idx; c++)
      comps[c] = nir_channel(b, def, c);
   comps[code_idx] =
      nir_bcsel(b, nir_ine_imm(b, nonresident, 0),
                nir_imm_intN_t(b, 1, def->bit_size),
                nir_channel(b, def, code_idx));

   nir_def_rewrite_uses_after(def, nir_vec(b, comps, def->num_components));
   return true;
}

static bool
lower_load_global_constant_offset_instr(nir_builder *b,
                                        nir_intrinsic_instr *intrin, void *data)
{
   const struct global_load_lowering *state = data;
   if ((intrin->intrinsic == nir_intrinsic_load_global_bounded &&
        !(state->soft_fault && state->storage_result_select)) ||
       (intrin->intrinsic == nir_intrinsic_load_global_constant_bounded &&
        !state->soft_fault)) {

      lower_load_global_bounded(b, intrin);
      return true;
   }

   if (intrin->intrinsic != nir_intrinsic_load_global_constant_offset &&
       intrin->intrinsic != nir_intrinsic_load_global_constant_bounded &&
       intrin->intrinsic != nir_intrinsic_load_global_bounded)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *base_addr = intrin->src[0].ssa;
   nir_def *offset = intrin->src[1].ssa;
   nir_def *bound = NULL;

   unsigned bit_size = intrin->def.bit_size;
   assert(bit_size >= 8 && bit_size % 8 == 0);
   unsigned byte_size = bit_size / 8;
   unsigned load_size = byte_size * intrin->num_components;

   const bool bounded =
      intrin->intrinsic == nir_intrinsic_load_global_constant_bounded ||
      intrin->intrinsic == nir_intrinsic_load_global_bounded;
   if (bounded) {
      bound = intrin->src[2].ssa;
   }

   unsigned align_mul = nir_intrinsic_align_mul(intrin);
   unsigned align_offset = nir_intrinsic_align_offset(intrin);

   nir_def *addr = nir_iadd(b, base_addr, nir_u2u64(b, offset));
   nir_def *val =
      intrin->intrinsic == nir_intrinsic_load_global_bounded
         ? nir_load_global(b, intrin->def.num_components, intrin->def.bit_size,
                           addr, .align_mul = align_mul,
                           .align_offset = align_offset,
                           .access = nir_intrinsic_access(intrin))
         : nir_load_global_constant(b, intrin->def.num_components,
                                    intrin->def.bit_size, addr,
                                    .align_mul = align_mul,
                                    .align_offset = align_offset,
                                    .access = nir_intrinsic_access(intrin));

   if (bounded) {
      nir_scalar offs = nir_scalar_resolved(offset, 0);
      if (nir_scalar_is_const(offs)) {
         /* Compare the exclusive end against the descriptor's byte size.
          * Do not round the access down: a descriptor ending exactly where
          * this load begins must return zero with robustBufferAccess2. */
         uint64_t end = (uint64_t)nir_scalar_as_uint(offs) + load_size;
         val = end <= UINT32_MAX
                  ? bounds_check(b, val, nir_imm_int(b, end), bound)
                  : nir_imm_zero(b, val->num_components, val->bit_size);
      } else {
         /* Saturation makes a wrapping address fail the unsigned check. */
         offset = nir_uadd_sat(b, offset, nir_imm_int(b, load_size));
         val = bounds_check(b, val, offset, bound);
      }
   }

   nir_def_replace(&intrin->def, val);
   return true;
}

struct lower_ycbcr_state {
   uint32_t set_layout_count;
   struct vk_descriptor_set_layout *const *set_layouts;
};

static const struct hk_sampler *
lookup_ycbcr_sampler(const void *_state, uint32_t set, uint32_t binding,
                     uint32_t array_index)
{
   const struct lower_ycbcr_state *state = _state;
   assert(set < state->set_layout_count);
   assert(state->set_layouts[set] != NULL);
   const struct hk_descriptor_set_layout *set_layout =
      vk_to_hk_descriptor_set_layout(state->set_layouts[set]);
   assert(binding < set_layout->binding_count);

   const struct hk_descriptor_set_binding_layout *bind_layout =
      &set_layout->binding[binding];

   if (bind_layout->immutable_samplers == NULL)
      return NULL;

   array_index = MIN2(array_index, bind_layout->array_size - 1);

   const struct hk_sampler *sampler =
      bind_layout->immutable_samplers[array_index];

   return sampler && sampler->vk.ycbcr_conversion ? sampler : NULL;
}

static const struct hk_sampler *
lookup_packed_ycbcr_sampler(nir_tex_instr *tex, const void *state)
{
   int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (handle < 0)
      return NULL;
   nir_deref_instr *deref = nir_src_as_deref(tex->src[handle].src);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   unsigned index = 0;
   if (deref->deref_type == nir_deref_type_array) {
      if (!nir_src_is_const(deref->arr.index))
         return NULL;
      index = nir_src_as_uint(deref->arr.index);
   }
   const struct hk_sampler *sampler = lookup_ycbcr_sampler(
      state, var->data.descriptor_set, var->data.binding, index);
   return sampler && hk_packed_ycbcr_bits(sampler->ycbcr.format) ? sampler : NULL;
}

/* API size queries return luma pixels. Run before adding private word queries. */
static bool
lower_packed_ycbcr_size(nir_builder *b, nir_tex_instr *tex, void *state)
{
   if (tex->op != nir_texop_txs || !lookup_packed_ycbcr_sampler(tex, state))
      return false;
   b->cursor = nir_after_instr(&tex->instr);
   nir_def *size = nir_vector_insert_imm(b, &tex->def,
      nir_imul_imm(b, nir_channel(b, &tex->def, 0), 2), 0);
   nir_def_rewrite_uses_after(&tex->def, size);
   return true;
}

static nir_def *
packed_ycbcr_word(nir_builder *b, nir_tex_instr *orig, nir_def *x,
                  nir_def *y, nir_def *layer, nir_def *word_size)
{
   x = nir_imin(b, nir_imax(b, x, nir_imm_int(b, 0)),
                  nir_iadd_imm(b, nir_channel(b, word_size, 0), -1));
   nir_def *coord = layer ? nir_vec3(b, x, y, layer) : nir_vec2(b, x, y);
   nir_tex_instr *fetch = nir_tex_instr_create(b->shader, 3);
   fetch->op = nir_texop_txf;
   fetch->sampler_dim = GLSL_SAMPLER_DIM_2D;
   fetch->is_array = orig->is_array;
   fetch->dest_type = nir_type_float32;
   fetch->coord_components = coord->num_components;
   fetch->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref,
      orig->src[nir_tex_instr_src_index(orig, nir_tex_src_texture_deref)].src.ssa);
   fetch->src[1] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   fetch->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
   nir_def_init(&fetch->instr, &fetch->def, 4, 32);
   nir_builder_instr_insert(b, &fetch->instr);
   return &fetch->def;
}

/* Reconstruct chroma at one luma centre. The outer explicit pass filters these
 * reconstructed samples, including separately chosen luma/chroma filters. */
static bool
lower_packed_ycbcr_center(nir_builder *b, nir_tex_instr *tex, void *state)
{
   if (tex->op != nir_texop_txl)
      return false;
   const struct hk_sampler *sampler = lookup_packed_ycbcr_sampler(tex, state);
   if (!sampler)
      return false;
   const struct vk_ycbcr_conversion_state *conversion = &sampler->ycbcr;
   unsigned bits = hk_packed_ycbcr_bits(conversion->format);
   bool first = hk_packed_ycbcr_luma_first(conversion->format);
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *word_size = nir_get_texture_size(b, tex);
   nir_def *size = nir_vec2(b, nir_imul_imm(b, nir_channel(b, word_size, 0), 2),
                             nir_channel(b, word_size, 1));
   nir_def *coord = tex->src[nir_tex_instr_src_index(tex, nir_tex_src_coord)].src.ssa;
   nir_def *ij = nir_f2i32(b, nir_ffloor(b,
      nir_fmul(b, nir_channels(b, coord, 3), nir_i2f32(b, size))));
   ij = nir_imin(b, nir_imax(b, ij, nir_imm_int(b, 0)), nir_iadd_imm(b, size, -1));
   nir_def *i = nir_channel(b, ij, 0), *j = nir_channel(b, ij, 1);
   nir_def *layer = NULL;
   if (tex->is_array) {
      layer = nir_f2i32(b, nir_fround_even(b, nir_channel(b, coord, 2)));
      layer = nir_imin(b, nir_imax(b, layer, nir_imm_int(b, 0)),
                         nir_iadd_imm(b, nir_channel(b, word_size, 2), -1));
   }
   nir_def *word_x = nir_ishr_imm(b, i, 1);
   nir_def *word = packed_ycbcr_word(b, tex, word_x, j, layer, word_size);
   nir_def *y = nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, i, 1), 0),
      nir_channel(b, word, first ? 2 : 3), nir_channel(b, word, first ? 0 : 1));
   nir_def *chroma = word;
   if (conversion->chroma_filter == VK_FILTER_LINEAR) {
      nir_def *x = nir_fmul_imm(b, nir_i2f32(b, i), 0.5);
      if (conversion->chroma_offsets[0] == VK_CHROMA_LOCATION_MIDPOINT)
         x = nir_fadd_imm(b, x, -0.25);
      nir_def *base = nir_f2i32(b, nir_ffloor(b, x));
      nir_def *a = packed_ycbcr_word(b, tex, base, j, layer, word_size);
      nir_def *z = packed_ycbcr_word(b, tex, nir_iadd_imm(b, base, 1), j, layer, word_size);
      chroma = nir_flrp(b, a, z, nir_ffract(b, x));
      b->shader->info.flrp_lowered = false;
   }
   const float scale = bits == 8 ? 1.0f :
      65535.0f / (((1u << bits) - 1u) << (16 - bits));
   nir_def *raw[4] = {
      nir_fmul_imm(b, nir_channel(b, chroma, first ? 3 : 2), scale),
      nir_fmul_imm(b, y, scale),
      nir_fmul_imm(b, nir_channel(b, chroma, first ? 1 : 0), scale),
      nir_imm_float(b, 1),
   };
   nir_def *mapped[4];
   for (unsigned c = 0; c < 4; ++c) {
      VkComponentSwizzle sw = conversion->mapping[c];
      mapped[c] = sw == VK_COMPONENT_SWIZZLE_ZERO ? nir_imm_float(b, 0) :
                  sw == VK_COMPONENT_SWIZZLE_ONE ? nir_imm_float(b, 1) :
                  raw[sw == VK_COMPONENT_SWIZZLE_IDENTITY ? c : sw - VK_COMPONENT_SWIZZLE_R];
   }
   nir_def *result = nir_vec(b, mapped, 4);
   uint32_t bpcs[4] = {bits, bits, bits, bits};
   if (conversion->ycbcr_model != VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY)
      result = nir_convert_ycbcr_to_rgb(b, conversion->ycbcr_model,
                                      conversion->ycbcr_range, result, bpcs);
   nir_def_replace(&tex->def, result);
   return true;
}

static const struct vk_ycbcr_conversion_state *
lookup_ycbcr_conversion(const void *state, uint32_t set, uint32_t binding,
                        uint32_t array_index)
{
   const struct hk_sampler *sampler =
      lookup_ycbcr_sampler(state, set, binding, array_index);
   /* Packed images were reconstructed by the native word-sampling pass. */
   return sampler && !hk_packed_ycbcr_bits(sampler->ycbcr.format) ? &sampler->ycbcr : NULL;
}

/* Reconstruct at full-resolution luma texel centres, then filter the converted
 * values. The common YCbCr pass lowers each centre sample into plane accesses
 * and colour conversion. YCbCr samplers use clamp-to-edge and a single mip. */
static bool
lower_ycbcr_explicit(nir_builder *b, nir_tex_instr *tex, void *state)
{
   switch (tex->op) {
   case nir_texop_tex:
   case nir_texop_txb:
   case nir_texop_txl:
   case nir_texop_txd:
      break;
   default:
      return false;
   }
   const int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (handle < 0)
      return false;
   nir_deref_instr *deref = nir_src_as_deref(tex->src[handle].src);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   unsigned index = 0;
   if (deref->deref_type == nir_deref_type_array) {
      if (!nir_src_is_const(deref->arr.index))
         return false;
      index = nir_src_as_uint(deref->arr.index);
   }
   const struct hk_sampler *sampler = lookup_ycbcr_sampler(
      state, var->data.descriptor_set, var->data.binding, index);
   if (!sampler || !sampler->ycbcr.force_explicit_reconstruction)
      return false;

   assert(tex->sampler_dim == GLSL_SAMPLER_DIM_2D);
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *size =
      nir_u2f32(b, nir_channels(b, nir_get_texture_size(b, tex), 3));
   if (hk_packed_ycbcr_bits(sampler->ycbcr.format))
      size = nir_fmul(b, size, nir_imm_vec2(b, 2, 1));
   nir_def *coord =
      tex->src[nir_tex_instr_src_index(tex, nir_tex_src_coord)].src.ssa;
   nir_def *uv = nir_fmul(b, nir_channels(b, coord, 3), size);
   int offset = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   if (offset >= 0)
      uv = nir_fadd(b, uv, nir_i2f32(b, tex->src[offset].src.ssa));

   nir_def *linear;
   if (sampler->min_filter == sampler->mag_filter) {
      linear = nir_imm_bool(b, sampler->min_filter == VK_FILTER_LINEAR);
   } else {
      nir_def *lod;
      if (tex->op == nir_texop_txl) {
         lod = tex->src[nir_tex_instr_src_index(tex, nir_tex_src_lod)].src.ssa;
         lod = nir_fadd_imm(b, lod, sampler->mip_lod_bias);
      } else if (tex->op == nir_texop_txd) {
         nir_def *dx = nir_fmul(
            b, size,
            tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddx)].src.ssa);
         nir_def *dy = nir_fmul(
            b, size,
            tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddy)].src.ssa);
         nir_def *log2_rho =
            nir_flog2(b, nir_fmax(b, nir_fdot(b, dx, dx), nir_fdot(b, dy, dy)));
         lod = nir_ffma(b, log2_rho,
                        nir_imm_floatN_t(b, 0.5, log2_rho->bit_size),
                        nir_imm_floatN_t(b, sampler->mip_lod_bias,
                                         log2_rho->bit_size));
      } else {
         lod = nir_get_texture_lod(b, tex);
         if (tex->op == nir_texop_txb)
            lod = nir_fadd(
               b, lod,
               tex->src[nir_tex_instr_src_index(tex, nir_tex_src_bias)].src.ssa);
      }
      linear =
         nir_bcsel(b, nir_fle_imm(b, lod, 0.0),
                   nir_imm_bool(b, sampler->mag_filter == VK_FILTER_LINEAR),
                   nir_imm_bool(b, sampler->min_filter == VK_FILTER_LINEAR));
   }

   nir_def *pos = nir_bcsel(b, linear, nir_fadd_imm(b, uv, -0.5), uv);
   nir_def *base = nir_ffloor(b, pos);
   nir_def *weight =
      nir_bcsel(b, linear, nir_ffract(b, pos), nir_imm_vec2(b, 0, 0));
   nir_def *taps[4];
   for (unsigned tap = 0; tap < 4; ++tap) {
      nir_def *xy = nir_fadd(b, base, nir_imm_vec2(b, tap & 1, tap >> 1));
      xy = nir_fmin(b, nir_fmax(b, xy, nir_imm_float(b, 0)),
                    nir_fadd_imm(b, size, -1));
      xy = nir_fdiv(b, nir_fadd_imm(b, xy, 0.5), size);
      nir_def *center = tex->is_array ? nir_vec3(b, nir_channel(b, xy, 0),
                                                 nir_channel(b, xy, 1),
                                                 nir_channel(b, coord, 2))
                                      : xy;
      nir_def *lod_zero = nir_imm_float(b, 0);
      nir_tex_instr *sample =
         nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
      /* Register cloned source uses before removing or rewriting them. */
      nir_builder_instr_insert(b, &sample->instr);
      for (int s = sample->num_srcs - 1; s >= 0; --s) {
         switch (sample->src[s].src_type) {
         case nir_tex_src_lod:
         case nir_tex_src_bias:
         case nir_tex_src_ddx:
         case nir_tex_src_ddy:
         case nir_tex_src_offset:
            nir_tex_instr_remove_src(sample, s);
            break;
         case nir_tex_src_coord:
            nir_src_rewrite(&sample->src[s].src, center);
            break;
         default:
            break;
         }
      }
      sample->op = nir_texop_txl;
      nir_tex_instr_add_src(sample, nir_tex_src_lod, lod_zero);
      taps[tap] = &sample->def;
   }
   nir_def *x = nir_channel(b, weight, 0), *y = nir_channel(b, weight, 1);
   /* Frontend preprocessing may already have lowered the original flrps. */
   b->shader->info.flrp_lowered = false;
   nir_def *result = nir_flrp(b, nir_flrp(b, taps[0], taps[1], x),
                              nir_flrp(b, taps[2], taps[3], x), y);
   nir_def_replace(&tex->def, result);
   return true;
}

/* Look up the conversion of a plane access produced by the common pass. */
static const struct vk_ycbcr_conversion_state *
plane_tex_conversion(nir_tex_instr *tex, const void *state, unsigned *plane)
{
   const int plane_src = nir_tex_instr_src_index(tex, nir_tex_src_plane);
   const int deref_src = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (plane_src < 0 || deref_src < 0 || !nir_src_is_const(tex->src[plane_src].src))
      return NULL;

   nir_deref_instr *deref = nir_src_as_deref(tex->src[deref_src].src);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   unsigned array_index = 0;
   if (deref->deref_type != nir_deref_type_var) {
      if (deref->deref_type != nir_deref_type_array ||
          !nir_src_is_const(deref->arr.index))
         return NULL;
      array_index = nir_src_as_uint(deref->arr.index);
   }
   *plane = nir_src_as_uint(tex->src[plane_src].src);
   return lookup_ycbcr_conversion(state, var->data.descriptor_set,
                                  var->data.binding, array_index);
}

/* The texture unit filters 16-bit UNORM planes at 16-bit precision: a
 * linear blend of two texels rounds to a whole code, half a code away from
 * the exact result. Chroma reconstruction must keep the format's precision
 * (the conversion tests bound it to a 2^-16 step), so blend 16-bit chroma
 * planes in the shader from nearest fetches. Multi-planar images have one
 * mip level and YCbCr samplers clamp to edge, so the blend of the four
 * clamped neighbours at the sample position is the linear filter result. */
static bool
lower_ycbcr_chroma_blend(nir_builder *b, nir_tex_instr *tex, void *state)
{
   switch (tex->op) {
   case nir_texop_tex:
   case nir_texop_txb:
   case nir_texop_txl:
   case nir_texop_txd:
      break;
   default:
      return false;
   }
   const int coord_src = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord_src < 0 || tex->sampler_dim != GLSL_SAMPLER_DIM_2D)
      return false;

   unsigned plane;
   const struct vk_ycbcr_conversion_state *conversion =
      plane_tex_conversion(tex, state, &plane);
   if (!conversion || conversion->chroma_filter != VK_FILTER_LINEAR)
      return false;
   const struct vk_format_ycbcr_info *info =
      vk_format_get_ycbcr_info(conversion->format);
   assert(info && plane < info->n_planes);
   if (!info->planes[plane].has_chroma)
      return false;
   switch (info->planes[plane].format) {
   case VK_FORMAT_R16_UNORM:
   case VK_FORMAT_R16G16_UNORM:
      break;
   default:
      return false;
   }

   const int deref_src = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   nir_def *texture = tex->src[deref_src].src.ssa;
   b->cursor = nir_before_instr(&tex->instr);

   nir_tex_instr *txs = nir_tex_instr_create(b->shader, 2);
   txs->op = nir_texop_txs;
   txs->sampler_dim = tex->sampler_dim;
   txs->is_array = tex->is_array;
   txs->dest_type = nir_type_int32;
   txs->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, texture);
   txs->src[1] = nir_tex_src_for_ssa(nir_tex_src_plane, nir_imm_int(b, plane));
   nir_def_init(&txs->instr, &txs->def, nir_tex_instr_dest_size(txs), 32);
   nir_builder_instr_insert(b, &txs->instr);
   nir_def *size = nir_channels(b, &txs->def, 3);

   /* The texture unit converts the coordinate to fixed point with
    * subTexelPrecisionBits fractional bits before filtering. Do the same, so
    * a sample that lies on a texel centre or boundary in that precision gets
    * the exact weight rather than a rounding residue of the float scaling. */
   nir_def *coord = tex->src[coord_src].src.ssa;
   nir_def *pos = nir_fadd_imm(
      b, nir_fmul(b, nir_channels(b, coord, 3), nir_i2f32(b, size)), -0.5);
   pos = nir_fmul_imm(b, nir_fround_even(b, nir_fmul_imm(b, pos, 256.0)),
                      1.0 / 256.0);
   nir_def *base = nir_ffloor(b, pos);
   nir_def *weight = nir_fsub(b, pos, base);
   nir_def *ibase = nir_f2i32(b, base);
   const int offset_src = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   if (offset_src >= 0)
      ibase = nir_iadd(b, ibase, nir_channels(b, tex->src[offset_src].src.ssa, 3));

   nir_def *layer = NULL;
   if (tex->is_array) {
      layer = nir_f2i32(b, nir_fround_even(b, nir_channel(b, coord, 2)));
      layer = nir_imin(b, nir_imax(b, layer, nir_imm_int(b, 0)),
                       nir_iadd_imm(b, nir_channel(b, &txs->def, 2), -1));
   }

   nir_def *taps[4];
   for (unsigned tap = 0; tap < 4; ++tap) {
      nir_def *xy = nir_iadd(b, ibase, nir_imm_ivec2(b, tap & 1, tap >> 1));
      xy = nir_imin(b, nir_imax(b, xy, nir_imm_int(b, 0)),
                    nir_iadd_imm(b, size, -1));
      nir_def *texel = layer ? nir_vec3(b, nir_channel(b, xy, 0),
                                        nir_channel(b, xy, 1), layer)
                             : xy;
      nir_tex_instr *fetch = nir_tex_instr_create(b->shader, 4);
      fetch->op = nir_texop_txf;
      fetch->sampler_dim = tex->sampler_dim;
      fetch->is_array = tex->is_array;
      fetch->dest_type = tex->dest_type;
      fetch->coord_components = texel->num_components;
      fetch->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, texture);
      fetch->src[1] = nir_tex_src_for_ssa(nir_tex_src_coord, texel);
      fetch->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
      fetch->src[3] = nir_tex_src_for_ssa(nir_tex_src_plane, nir_imm_int(b, plane));
      nir_def_init(&fetch->instr, &fetch->def, tex->def.num_components,
                   tex->def.bit_size);
      nir_builder_instr_insert(b, &fetch->instr);
      taps[tap] = &fetch->def;
   }

   nir_def *x = nir_channel(b, weight, 0), *y = nir_channel(b, weight, 1);
   if (tex->def.bit_size != 32) {
      x = nir_f2fN(b, x, tex->def.bit_size);
      y = nir_f2fN(b, y, tex->def.bit_size);
   }
   b->shader->info.flrp_lowered = false;
   nir_def *result = nir_flrp(b, nir_flrp(b, taps[0], taps[1], x),
                              nir_flrp(b, taps[2], taps[3], x), y);
   nir_def_replace(&tex->def, result);
   return true;
}

/* hk_format_to_pipe_format samples X6R10/X4R12 planes through R16_UNORM.
 * Restore their logical UNORM range before range expansion and YCbCr matrix
 * conversion. The correction commutes with filtering; absent channels and
 * alpha must retain their descriptor defaults. */
static bool
lower_ycbcr_plane_range(nir_builder *b, nir_tex_instr *tex, void *data)
{
   const int plane_src = nir_tex_instr_src_index(tex, nir_tex_src_plane);
   const int deref_src = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (plane_src < 0 || deref_src < 0 || tex->op == nir_texop_txs ||
       tex->op == nir_texop_query_levels || tex->op == nir_texop_lod)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(tex->src[deref_src].src);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   unsigned array_index = 0;
   if (deref->deref_type != nir_deref_type_var) {
      if (deref->deref_type != nir_deref_type_array ||
          !nir_src_is_const(deref->arr.index))
         return false;
      array_index = nir_src_as_uint(deref->arr.index);
   }

   const struct vk_ycbcr_conversion_state *conversion =
      lookup_ycbcr_conversion(data, var->data.descriptor_set,
                              var->data.binding, array_index);
   if (!conversion)
      return false;

   const struct vk_format_ycbcr_info *info =
      vk_format_get_ycbcr_info(conversion->format);
   const unsigned plane = nir_src_as_uint(tex->src[plane_src].src);
   if (!info || plane >= info->n_planes)
      return false;

   const VkFormat format = info->planes[plane].format;
   unsigned bits;
   switch (format) {
   case VK_FORMAT_R10X6_UNORM_PACK16:
   case VK_FORMAT_R10X6G10X6_UNORM_2PACK16:
      bits = 10;
      break;
   case VK_FORMAT_R12X4_UNORM_PACK16:
   case VK_FORMAT_R12X4G12X4_UNORM_2PACK16:
      bits = 12;
      break;
   default:
      return false;
   }

   /* hk_format_to_pipe_format samples these MSB-aligned channels as UNORM16.
    * Normalize by the actual channel maximum before YCbCr range/model math.
    * This is linear, so it also commutes with the chroma filtering above. */
   const float scale = 65535.0f / (BITFIELD_MASK(bits) << (16 - bits));
   const unsigned comps =
      util_format_get_nr_components(hk_format_to_pipe_format(format));
   b->cursor = nir_after_instr(&tex->instr);
   nir_def *channels[4];
   for (unsigned c = 0; c < tex->def.num_components; ++c) {
      channels[c] = nir_channel(b, &tex->def, c);
      if (c < comps)
         channels[c] = nir_fmul_imm(b, channels[c], scale);
   }
   nir_def_rewrite_uses_after(&tex->def,
                             nir_vec(b, channels, tex->def.num_components));
   return true;
}

static unsigned
glsl_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/*
 * This is the world's worst multiview implementation. We simply duplicate each
 * draw on the CPU side, changing a uniform in between, and then plumb the view
 * index into the layer ID here. Whatever, it works.
 *
 * The "proper" implementation on AGX would use vertex amplification, but a
 * MacBook is not a VR headset.
 *
 * This must run on the last pre-rasterization stage, since that is the stage
 * whose layer output the rasterizer consumes: see hk_lower_multiview_stage().
 */
/* A multiview fragment shader's gl_Layer is the view index, as
 * hk_lower_multiview() makes the last pre-rasterization stage write it. Read
 * it from the root, like gl_ViewIndex: a G17P mesh shader carries no
 * per-primitive layer for it (hk_lower_multiview()). */
static bool
hk_lower_multiview_fs_layer(nir_builder *b, nir_intrinsic_instr *intr,
                            UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_layer_id)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_u2uN(b, nir_load_view_index(b),
                                        intr->def.bit_size));
   return true;
}

static void
hk_lower_multiview(nir_shader *nir, bool g17p)
{
   /* If there's an existing layer ID write, ignore it. This avoids validation
    * splat with vk_meta.
    */
   nir_variable *existing = nir_find_variable_with_location(
      nir, nir_var_shader_out, VARYING_SLOT_LAYER);

   if (existing) {
      existing->data.mode = nir_var_shader_temp;
      existing->data.location = 0;
      nir_fixup_deref_modes(nir);
   }

   /* Now write the view index as the layer.
    *
    * A geometry shader's EmitVertex snapshots the output variables, so the
    * store has to precede them; the view index is a per-draw root constant,
    * so hoisting it to the top of main is correct. Every other stage writes
    * its outputs once, and on G17P the store must stay with them: a store
    * separated from the terminal writeout bundle leaves
    * agx3_g17p_vertex_output_count() with a partial bundle, and the packer
    * then refuses the stranded UVS slot (agx3_pack_binary.c, "User-output
    * graphs outside the qualified scalar shapes").
    */
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir->info.stage == MESA_SHADER_GEOMETRY
                                     ? nir_before_impl(impl)
                                     : nir_after_impl(impl));

   if (nir->info.stage == MESA_SHADER_MESH && g17p &&
       !nir_find_variable_with_location(nir, nir_var_shader_out,
                                        VARYING_SLOT_VIEWPORT)) {
      /* The view is the same for the whole workgroup, so on G17P it goes to
       * every vertex's layer word (agx_nir_lower_gen1_mesh, vertex_layer):
       * a per-primitive layer would give each primitive private vertices
       * and a primitive record. Stored first, before any early return: a
       * vertex word nothing writes is undefined. */
      const unsigned verts = nir->info.mesh.max_vertices_out;
      const unsigned threads = nir->info.workgroup_size[0] *
                               nir->info.workgroup_size[1] *
                               nir->info.workgroup_size[2];
      b.cursor = nir_before_impl(impl);
      nir_variable *layer = nir_variable_create(
         nir, nir_var_shader_out, glsl_array_type(glsl_uint_type(), verts, 0),
         "view_layer");
      layer->data.location = VARYING_SLOT_LAYER;
      nir_def *view = nir_load_view_index(&b);
      nir_def *lane = nir_load_local_invocation_index(&b);
      for (unsigned first = 0; first < verts; first += threads) {
         nir_def *vertex = nir_iadd_imm(&b, lane, first);
         nir_push_if(&b, nir_ult_imm(&b, vertex, verts));
         nir_store_deref(&b, nir_build_deref_array(&b,
                            nir_build_deref_var(&b, layer), vertex), view, 1);
         nir_pop_if(&b, NULL);
      }
      nir->info.outputs_written |= VARYING_BIT_LAYER;
      return;
   }

   if (nir->info.stage == MESA_SHADER_MESH) {
      /* Every primitive the workgroup may output goes to the view's layer. */
      const unsigned prims = nir->info.mesh.max_primitives_out;
      const unsigned threads = nir->info.workgroup_size[0] *
                               nir->info.workgroup_size[1] *
                               nir->info.workgroup_size[2];
      nir_variable *layer = nir_variable_create(
         nir, nir_var_shader_out, glsl_array_type(glsl_uint_type(), prims, 0),
         "view_layer");
      layer->data.location = VARYING_SLOT_LAYER;
      layer->data.per_primitive = true;
      nir_def *view = nir_load_view_index(&b);
      nir_def *lane = nir_load_local_invocation_index(&b);
      for (unsigned first = 0; first < prims; first += threads) {
         nir_def *prim = nir_iadd_imm(&b, lane, first);
         nir_push_if(&b, nir_ult_imm(&b, prim, prims));
         nir_store_deref(&b, nir_build_deref_array(&b,
                            nir_build_deref_var(&b, layer), prim), view, 1);
         nir_pop_if(&b, NULL);
      }
      nir->info.outputs_written |= VARYING_BIT_LAYER;
      nir->info.per_primitive_outputs |= VARYING_BIT_LAYER;
      return;
   }

   nir_variable *layer =
      nir_variable_create(nir, nir_var_shader_out, glsl_uint_type(), NULL);

   layer->data.location = VARYING_SLOT_LAYER;

   nir_store_var(&b, layer, nir_load_view_index(&b), nir_component_mask(1));
   b.shader->info.outputs_written |= VARYING_BIT_LAYER;
}

static nir_def *
query_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   nir_def *border = nir_build_texture_query(
      b, tex, nir_texop_custom_border_color_agx, 4, tex->dest_type, false, false);
   if (tex->op == nir_texop_tg4)
      nir_instr_as_tex(nir_def_instr(border))->backend_flags |=
         AGX_TEXTURE_FLAG_BORDER_UNFILTERED;
   return border;
}

static nir_def *
has_custom_border(nir_builder *b, nir_tex_instr *tex)
{
   return nir_build_texture_query(b, tex, nir_texop_has_custom_border_color_agx,
                                  1, nir_type_bool1, false, false);
}

static bool
lower(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (!nir_tex_instr_need_sampler(tex) || nir_tex_instr_is_query(tex))
      return false;

   /* XXX: this is a really weird edge case, is this even well-defined? */
   if (tex->is_shadow)
      return false;

   b->cursor = nir_after_instr(&tex->instr);
   nir_def *has_custom = has_custom_border(b, tex);

   nir_instr *orig = nir_instr_clone(b->shader, &tex->instr);
   nir_builder_instr_insert(b, orig);
   nir_def *clamp_to_1 = &nir_instr_as_tex(orig)->def;

   nir_push_if(b, has_custom);
   nir_def *replaced = NULL;
   {
      /* Sample again, this time with clamp-to-0 instead of clamp-to-1 */
      nir_instr *clone_instr = nir_instr_clone(b->shader, &tex->instr);
      nir_builder_instr_insert(b, clone_instr);

      nir_tex_instr *tex_0 = nir_instr_as_tex(clone_instr);
      nir_def *clamp_to_0 = &tex_0->def;

      tex_0->backend_flags |= AGX_TEXTURE_FLAG_CLAMP_TO_0;

      /* Grab the border colour */
      nir_def *border = query_custom_border(b, tex_0);

      if (tex->op == nir_texop_tg4) {
         border = nir_replicate(b, nir_channel(b, border, tex->component), 4);
      }

      /* Combine together with the border */
      if (nir_alu_type_get_base_type(tex->dest_type) == nir_type_float &&
          tex->op != nir_texop_tg4) {

         /* For floats, lerp together:
          *
          * For border texels:  (1 * border) + (0 * border      ) = border
          * For regular texels: (x * border) + (x * (1 - border)) = x.
          *
          * Linear filtering is linear (duh), so lerping is compatible.
          */
         replaced = nir_flrp(b, clamp_to_0, clamp_to_1, border);
         b->shader->info.flrp_lowered = false;
      } else {
         /* For integers, just select componentwise since there is no linear
          * filtering. Gathers also use this path since they are unfiltered in
          * each component.
          */
         replaced = nir_bcsel(b, nir_ieq(b, clamp_to_0, clamp_to_1), clamp_to_0,
                              border);
      }
   }
   nir_pop_if(b, NULL);

   /* Put it together with a phi */
   nir_def *phi = nir_if_phi(b, replaced, clamp_to_1);
   nir_def_replace(&tex->def, phi);
   return true;
}

static bool
agx_nir_lower_custom_border(nir_shader *nir)
{
   return nir_shader_tex_pass(nir, lower, nir_metadata_none, NULL);
}

/* Whether nif is the branch agx_nir_lower_custom_border() put around
 * clamp-to-0 sample tex_0, after descriptor lowering: taken iff the sampler's
 * clamp-to-0 index c is not negative, i.e. `if (c >= 0)`, with tex_0 sampling
 * c without its integer-border bit (hk_nir_lower_descriptors()). */
static bool
hk_is_border_emulation_if(nir_if *nif, nir_tex_instr *tex_0, bool g17p)
{
   const int s = nir_tex_instr_src_index(tex_0, nir_tex_src_sampler_handle);
   if (s < 0)
      return false;

   nir_scalar index =
      nir_scalar_chase_movs(nir_get_scalar(tex_0->src[s].src.ssa, 0));
   if (!nir_scalar_is_alu(index) || nir_scalar_alu_op(index) != nir_op_iand)
      return false;

   nir_scalar mask = nir_scalar_chase_alu_src(index, 1);
   nir_scalar c = nir_scalar_chase_alu_src(index, 0);
   if (!nir_scalar_is_const(mask) ||
       nir_scalar_as_uint(mask) != HK_BORDER_IS_INT_BIT - 1)
      return false;

   nir_scalar cond =
      nir_scalar_chase_movs(nir_get_scalar(nif->condition.ssa, 0));
   if (!nir_scalar_is_alu(cond) || nir_scalar_alu_op(cond) != nir_op_ige)
      return false;

   nir_scalar tested = nir_scalar_chase_movs(nir_scalar_chase_alu_src(cond, 0));
   c = nir_scalar_chase_movs(c);
   if (!nir_scalar_is_zero(nir_scalar_chase_alu_src(cond, 1)))
      return false;
   if (nir_scalar_equal(tested, c))
      return true;
   if (!g17p)
      return false;
   const struct hk_sampled_image_descriptor_layout native =
      hk_sampled_image_descriptor_layout(9);
   return hk_nir_border_indices_same_descriptor(
      tested, c,
      native.clamp_0_sampler_index - native.sampler -
         offsetof(struct hk_sampled_image_descriptor_sw,
                  g17p.unbiased_clamp_0_sampler_index));
}

/* The branches of the custom-border emulation, which hold the only
 * clamp-to-0 samples hk makes. Returns false, and the shader keeps the
 * emulation, unless every clamp-to-0 sample sits directly in the then-list
 * of its branch (hk_is_border_emulation_if()). */
static bool
hk_find_border_emulation(nir_shader *nir, struct util_dynarray *ifs, bool g17p)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex ||
                !(nir_instr_as_tex(instr)->backend_flags &
                  AGX_TEXTURE_FLAG_CLAMP_TO_0))
               continue;

            nir_cf_node *parent = block->cf_node.parent;
            if (parent->type != nir_cf_node_if)
               return false;

            nir_if *nif = nir_cf_node_as_if(parent);
            bool in_then = false;
            foreach_list_typed(nir_cf_node, node, node, &nif->then_list)
               in_then |= node == &block->cf_node;
            if (!in_then ||
                !hk_is_border_emulation_if(nif, nir_instr_as_tex(instr), g17p)) {
               return false;
            }

            bool known = false;
            util_dynarray_foreach(ifs, nir_if *, seen)
               known |= *seen == nif;
            if (!known)
               util_dynarray_append(ifs, nif);
         }
      }
   }
   return true;
}

/* Compile without the custom-border emulation: no sampler the device can
 * bind has a custom border, so none of its branches is taken. */
static void
hk_drop_border_emulation(struct util_dynarray *ifs)
{
   util_dynarray_foreach(ifs, nir_if *, nif) {
      nir_builder b = nir_builder_at(nir_before_cf_node(&(*nif)->cf_node));
      nir_src_rewrite(&(*nif)->condition, nir_imm_false(&b));
      nir_progress(true, nir_cf_node_get_function(&(*nif)->cf_node),
                   nir_metadata_control_flow);
   }
}

/* Unformatted float images may name scaled views. Their native descriptor
 * uses integer texels; the shader performs the Vulkan numerical conversion.
 * The descriptor's existing type/channel fields avoid a new software ABI. */
static bool
lower_scaled_storage(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct hk_physical_device *pdev = data;
   if (!b->shader->info.internal && pdev && !pdev->scaled_storage)
      return false;

   const bool load = intr->intrinsic == nir_intrinsic_bindless_image_load;
   const bool store = intr->intrinsic == nir_intrinsic_bindless_image_store;
   if ((!load && !store) || nir_intrinsic_format(intr) != PIPE_FORMAT_NONE ||
       (load ? nir_intrinsic_dest_type(intr)
             : nir_intrinsic_src_type(intr)) != nir_type_float32)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *address = nir_load_from_texture_handle_agx(b, intr->src[0].ssa);
   nir_def *header =
      nir_load_global_constant(b, 1, 32, address, .align_mul = 4);
   nir_def *type = nir_iand_imm(b, nir_ushr_imm(b, header, 13), 7);
   nir_def *is_signed = nir_ieq_imm(b, type, AGX_TEXTURE_TYPE_SINT);
   nir_def *is_integer = nir_ior(
      b, is_signed, nir_ieq_imm(b, type, AGX_TEXTURE_TYPE_UINT));
   if (load) {
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *value = &intr->def;
      nir_def *converted =
         nir_bcsel(b, is_signed, nir_i2f32(b, value), nir_u2f32(b, value));
      nir_def_rewrite_uses_after(
         value, nir_bcsel(b, is_integer, converted, value));
      return true;
   }

   nir_def *channels = nir_iand_imm(b, nir_ushr_imm(b, header, 6), 127);
   nir_def *is_8 = nir_ior(
      b, nir_ieq_imm(b, channels, AGX_CHANNELS_R8),
      nir_ior(b, nir_ieq_imm(b, channels, AGX_CHANNELS_R8G8),
              nir_ieq_imm(b, channels, AGX_CHANNELS_R8G8B8A8)));
   nir_def *is_16 = nir_ior(
      b, nir_ieq_imm(b, channels, AGX_CHANNELS_R16),
      nir_ior(b, nir_ieq_imm(b, channels, AGX_CHANNELS_R16G16),
              nir_ieq_imm(b, channels, AGX_CHANNELS_R16G16B16A16)));
   nir_push_if(b, nir_iand(b, is_integer, nir_ior(b, is_8, is_16)));
   {
      nir_def *limit = nir_bcsel(b, is_8, nir_imm_float(b, 256),
                                 nir_imm_float(b, 65536));
      nir_def *signed_limit = nir_fmul_imm(b, limit, 0.5);
      nir_def *upper = nir_fadd_imm(
         b, nir_bcsel(b, is_signed, signed_limit, limit), -1);
      nir_def *lower = nir_bcsel(b, is_signed, nir_fneg(b, signed_limit),
                                 nir_imm_float(b, 0));
      nir_def *rounded = nir_fround_even(b, intr->src[3].ssa);
      nir_def *clamped = nir_fmin(b, nir_fmax(b, rounded, lower), upper);
      nir_def *value = nir_bcsel(b, is_signed, nir_f2i32(b, clamped),
                                 nir_f2u32(b, clamped));
      nir_push_if(b, is_8);
      for (unsigned wide = 0; wide < 2; ++wide) {
         if (wide)
            nir_push_else(b, NULL);
         nir_intrinsic_instr *copy = nir_instr_as_intrinsic(
            nir_instr_clone(b->shader, &intr->instr));
         nir_builder_instr_insert(b, &copy->instr);
         nir_src_rewrite(&copy->src[3], value);
         /* The descriptor retains signedness and channel count. The typed
          * intrinsic selects the matching raw I8/I16 staging instruction. */
         nir_intrinsic_set_format(copy, wide ? PIPE_FORMAT_R16G16B16A16_UINT
                                             : PIPE_FORMAT_R8G8B8A8_UINT);
         nir_intrinsic_set_src_type(copy, nir_type_uint32);
      }
      nir_pop_if(b, NULL);
   }
   nir_push_else(b, NULL);
   nir_builder_instr_insert(b, nir_instr_clone(b->shader, &intr->instr));
   nir_pop_if(b, NULL);
   nir_instr_remove(&intr->instr);
   return true;
}

/* Driver-internal shaders branch around the scaled-view texel conversion. */
static bool
hk_scaled_sample_guard(const nir_shader *nir)
{
   return nir->info.internal;
}

/* Image-view formats are dynamic descriptor state. Recover scaled samples
 * from the integer descriptor result without changing ordinary float samples. */
static bool
lower_scaled_sample(nir_builder *b, nir_tex_instr *tex, void *data)
{
   const struct hk_physical_device *pdev = data;
   /* No scaled format is sampleable on this device, so no valid application
    * descriptor carries a nonzero kind and the conversion below is dead code
    * that every float sample would otherwise pay for (about 50 ALU
    * instructions each, unguarded). Driver-internal shaders keep their
    * guarded conversion. */
   if (!b->shader->info.internal && pdev && !pdev->scaled_sampling)
      return false;
   if (tex->dest_type != nir_type_float32 || tex->is_shadow ||
       tex->sampler_dim == GLSL_SAMPLER_DIM_BUF || tex->is_sparse)
      return false;
   switch (tex->op) {
   case nir_texop_tex: case nir_texop_txb: case nir_texop_txl:
   case nir_texop_txd: case nir_texop_txf: case nir_texop_txf_ms:
   case nir_texop_tg4:
      break;
   default:
      return false;
   }
   const int handle = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   if (handle < 0)
      return false;
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *desc = nir_load_from_texture_handle_agx(b, tex->src[handle].src.ssa);
   /* Read scaled_conversion through an aligned 32-bit load.
    *
    * MEASURED on G17P: as a 16-bit load at descriptor offset 50 this returned
    * the offset-40 word -- sampler_index, the application sampler heap index
    * plus its +28 bias, so kind came back as 29 instead of 0 -- on a subset of
    * waves. A non-zero kind sends an already-float sample through u2f32(), and
    * the affected lanes returned exactly u2f32(bits(value)).
    *
    * The 16-bit form lowers to a load_constant_agx whose offset cannot fold
    * into the scaled index, so the offset is added to the descriptor-set base
    * and the load issues against a general-register base rather than the
    * uniform-register one the other descriptor reads use. Reading the field as
    * the upper half of the aligned 32-bit word at offset 48 (min_lod_uint16
    * occupies 48..49, scaled_conversion 50..51) restores the folded, uniform
    * base form and fixes it.
    *
    * Note this does not fully explain the hardware behaviour: sampled_heap_index
    * at offset 68 is likewise not eight-byte aligned and reads correctly, so
    * alignment alone is not the discriminator. The fix is justified by
    * measurement, not by a complete model, and other narrow descriptor reads
    * (clamp_0 at 42, the lod fields at 44/46) have not been audited for this.
    */
   nir_def *pair = nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, desc,
                   32 + offsetof(struct hk_sampled_image_descriptor_sw,
                                 min_lod_uint16)),
      .align_mul = 4);
   nir_def *kind = nir_u2u16(b, nir_ushr_imm(b, pair, 16));
   b->cursor = nir_after_instr(&tex->instr);
   nir_def *value = &tex->def;

   /* Scaled-format sample guard: kind is zero for every
    * view that is not USCALED/SSCALED, and the conversion below is then the
    * identity. Without the guard, every float sample in every shader pays
    * ~50 unconditional ALU instructions (4x i2f/u2f, the per-channel factor
    * select chains, the multiply and the final selects). Branch around the
    * whole conversion instead: kind comes from the descriptor, so the branch
    * is uniform for uniform descriptors and still exact for divergent ones.
    * dont_flatten keeps nir_opt_peephole_select (limit 64) from turning it
    * back into selects. The bits returned for kind == 0 are the raw sample,
    * exactly as the unguarded final bcsel returns.
    *
    * Driver-internal shaders (vk_meta blits, resolves and copies) always take
    * the guard. Their views are almost never scaled, they run once per pixel
    * of a whole surface, and they have no constant program. The guard is not
    * a global default: its branch reaches four Minecraft fragment constant
    * programs as a uniform_store the AGX3 packer cannot encode, and those
    * fragments then build without their constant program
    * (hk_g17p_compile_fs_cprog()), a trade not yet measured. A vk_meta colour
    * blit FS executes ~105 instructions per pixel on the path without a
    * scaled view or custom border unguarded and ~57 guarded (static
    * 211 -> 244). */
   const bool guard = hk_scaled_sample_guard(b->shader);
   nir_def *placeholder = NULL;
   nir_if *guard_if = NULL;
   if (guard) {
      /* Route every existing consumer through a placeholder so the phi
       * below can replace them even though it lives in a later block. */
      placeholder = nir_mov(b, value);
      nir_def_rewrite_uses_after(value, placeholder);
      guard_if = nir_push_if(b, nir_ine_imm(b, kind, 0));
      guard_if->control = nir_selection_control_dont_flatten;
   }

   nir_def *converted = nir_bcsel(b, nir_ieq_imm(b, kind, 1),
      nir_i2f32(b, value), nir_u2f32(b, value));
   nir_def *scale = nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, kind, 4), 0),
                             nir_imm_float(b, 65535), nir_imm_float(b, 255));
   scale = nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, kind, 128), 0),
                     nir_imm_float(b, 1023), scale);
   nir_def *factors[4];
   for (unsigned c = 0; c < value->num_components; ++c) {
      unsigned component = tex->op == nir_texop_tg4 ? tex->component : c;
      factors[c] = nir_bcsel(b,
         nir_ine_imm(b, nir_iand_imm(b, kind, BITFIELD_BIT(3 + component)), 0),
         nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(
            b, kind, BITFIELD_BIT(8 + component)), 0),
            nir_imm_float(b, 3), scale),
         nir_imm_float(b, 1));
   }
   nir_def *normalized = nir_fmul(b, value, nir_vec(b, factors, value->num_components));
   if (tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms || tex->op == nir_texop_tg4)
      normalized = nir_fround_even(b, normalized);
   converted = nir_bcsel(b, nir_ieq_imm(b, nir_iand_imm(b, kind, 3), 3),
                           normalized, converted);
   if (guard) {
      nir_pop_if(b, guard_if);
      nir_def_replace(placeholder, nir_if_phi(b, converted, value));
      return true;
   }
   nir_def *result = nir_bcsel(b, nir_ine_imm(b, kind, 0), converted, value);
   nir_def_rewrite_uses_after(value, result);
   return true;
}

/*
 * In Vulkan, the VIEWPORT should read 0 in the fragment shader if it is not
 * written by the vertex shader, but in our implementation, the varying would
 * otherwise be undefined. This small pass predicates VIEWPORT reads based on
 * whether the hardware vertex shader writes the VIEWPORT (nonzero UVS index).
 */
static bool
lower_viewport_fs(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_VIEWPORT)
      return false;

   b->cursor = nir_after_instr(&intr->instr);
   nir_def *orig = &intr->def;

   nir_def *uvs = nir_load_uvs_index_agx(b, .io_semantics = sem);
   nir_def *def = nir_bcsel(b, nir_ine_imm(b, uvs, 0), orig, nir_imm_int(b, 0));

   nir_def_rewrite_uses_after(orig, def);
   return true;
}

static bool
lower_subpass_dim(nir_builder *b, nir_tex_instr *tex, UNUSED void *_data)
{
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   else if (tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS_MS)
      tex->sampler_dim = GLSL_SAMPLER_DIM_MS;
   else
      return false;

   return true;
}

/* Native texel fetches are memory-safe but can alias another texel within
 * the allocation. Enforce the Vulkan view bounds before narrowing coordinates
 * or LODs for the hardware, and return zero for invalid fetches. */
static bool
hk_lower_native_robust_txf(nir_builder *b, nir_tex_instr *tex, UNUSED void *data)
{
   if (tex->op != nir_texop_txf && tex->op != nir_texop_txf_ms)
      return false;

   b->cursor = nir_before_instr(&tex->instr);
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   const int sample_idx = nir_tex_instr_src_index(tex, nir_tex_src_ms_index);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   const unsigned components = tex->coord_components;
   nir_def *size = nir_build_texture_query(b, tex, nir_texop_txs,
                                           components, nir_type_uint32,
                                           false, lod_idx >= 0);
   if (lod_idx >= 0) {
      nir_tex_instr *query = nir_instr_as_tex(nir_def_instr(size));
      const int query_lod = nir_tex_instr_src_index(query, nir_tex_src_lod);
      nir_src_rewrite(&query->src[query_lod].src, tex->src[lod_idx].src.ssa);
   }
   nir_def *valid = nir_ball(b, nir_ult(b, coord, size));
   if (lod_idx >= 0 && tex->sampler_dim != GLSL_SAMPLER_DIM_BUF) {
      nir_def *levels = nir_build_texture_query(b, tex, nir_texop_query_levels,
                                               1, nir_type_uint32, false, false);
      nir_def *lod = tex->src[lod_idx].src.ssa;
      valid = nir_iand(b, valid, nir_ult(b, lod, levels));
      nir_src_rewrite(&tex->src[lod_idx].src,
                      nir_bcsel(b, valid, lod, nir_imm_int(b, 0)));
   }
   if (sample_idx >= 0) {
      nir_def *samples = nir_build_texture_query(b, tex, nir_texop_texture_samples,
                                                1, nir_type_uint32, false, false);
      nir_def *sample = tex->src[sample_idx].src.ssa;
      valid = nir_iand(b, valid, nir_ult(b, sample, samples));
      nir_src_rewrite(&tex->src[sample_idx].src,
                      nir_bcsel(b, valid, sample, nir_imm_int(b, 0)));
   }
   nir_src_rewrite(&tex->src[coord_idx].src,
                   nir_bcsel(b, valid, coord, nir_imm_zero(b, components, 32)));
   /* robustImageAccess2: an invalid fetch returns (0,0,0,1) for a view whose
    * format has no alpha channel and (0,0,0,0) otherwise. The descriptor's
    * alpha swizzle is the constant one exactly for alpha-less formats. */
   nir_def *invalid_value = nir_imm_zero(b, tex->def.num_components,
                                         tex->def.bit_size);
   const int deref_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (tex->def.num_components == 4 && deref_idx >= 0) {
      nir_def *word0 = nir_load_texture_descriptor_word_agx(
         b, tex->src[deref_idx].src.ssa);
      nir_def *swizzle_a = nir_ubitfield_extract_imm(b, word0, 25, 3);
      nir_def *one = nir_alu_type_get_base_type(tex->dest_type) == nir_type_float
                        ? nir_imm_floatN_t(b, 1, tex->def.bit_size)
                        : nir_imm_intN_t(b, 1, tex->def.bit_size);
      nir_def *alpha = nir_bcsel(b, nir_ieq_imm(b, swizzle_a, AGX_CHANNEL_1),
                                 one, nir_imm_zero(b, 1, tex->def.bit_size));
      invalid_value = nir_vector_insert_imm(b, invalid_value, alpha, 3);
   }
   b->cursor = nir_after_instr(&tex->instr);
   nir_def *result = nir_bcsel(b, valid, &tex->def, invalid_value);
   nir_def_rewrite_uses_after(&tex->def, result);
   return true;
}

static bool
should_lower_robust(const nir_intrinsic_instr *intr, const void *data)
{
   /* The hardware is robust, but our software image atomics are not. Unlike the
    * GL driver, we don't use the common buffer image lowering, using the
    * agx_nir_lower_texture lowering for robustImageAccess2 semantics.
    */
   if (data && *(const bool *)data &&
       (intr->intrinsic == nir_intrinsic_image_deref_load ||
        intr->intrinsic == nir_intrinsic_image_deref_store))
      return true;

   return intr->intrinsic == nir_intrinsic_image_deref_atomic ||
          intr->intrinsic == nir_intrinsic_image_deref_atomic_swap;
}

/* robustImageAccess2 returns (0,0,0,1) out of bounds for a view whose format
 * has no alpha channel and (0,0,0,0) otherwise. A load without a format
 * qualifier reads that from the native texture descriptor: hk packs the
 * format's swizzle into it, so the alpha swizzle is the constant one exactly
 * for alpha-less formats, and the null descriptor swizzles alpha to zero. */
static nir_def *
hk_native_robust_oob_alpha(nir_builder *b, nir_intrinsic_instr *load,
                           const void *data)
{
   const nir_alu_type type = nir_intrinsic_dest_type(load);
   const unsigned bit_size = load->def.bit_size;
   nir_def *zero = nir_imm_zero(b, 1, bit_size);
   nir_def *one = nir_alu_type_get_base_type(type) == nir_type_float
                     ? nir_imm_floatN_t(b, 1, bit_size)
                     : nir_imm_intN_t(b, 1, bit_size);

   /* The descriptor read is resolved by hk_nir_lower_descriptors, which runs
    * after this pass; the wrap itself must precede descriptor lowering so
    * the native read path's own coordinate clamp applies inside the branch. */
   if (!(data && *(const bool *)data) ||
       load->intrinsic != nir_intrinsic_image_deref_load)
      return one;

   nir_def *word0 = nir_load_image_descriptor_word_agx(b, load->src[0].ssa);
   nir_def *swizzle_a = nir_ubitfield_extract_imm(b, word0, 25, 3);
   return nir_bcsel(b, nir_ieq_imm(b, swizzle_a, AGX_CHANNEL_1), one, zero);
}

/*
 * ViewIndex reaches the rasterizer as the layer ID, so the ViewIndex -> Layer
 * store belongs in the last pre-rasterization stage and nowhere else: a store
 * in an earlier stage feeds the next shader stage instead of the rasterizer
 * and is dropped. next_stage_mask is the exact next stage in the pipeline
 * (vk_pipeline.c fills it from the sorted stage array, or FRAGMENT for the
 * last geometry stage), so it identifies that stage without a separate
 * stage-set query.
 */
static bool
hk_lower_multiview_stage(mesa_shader_stage stage,
                         VkShaderStageFlags next_stage_mask)
{
   const VkShaderStageFlags later_geom =
      VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
      VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
      VK_SHADER_STAGE_GEOMETRY_BIT;

   switch (stage) {
   case MESA_SHADER_VERTEX:
      return !(next_stage_mask & later_geom);
   case MESA_SHADER_TESS_EVAL:
      return !(next_stage_mask & VK_SHADER_STAGE_GEOMETRY_BIT);
   case MESA_SHADER_GEOMETRY:
   case MESA_SHADER_MESH:
      return true;
   default:
      return false;
   }
}

/* Select the query ABI from the stage that will execute the query. */
static struct agx_gen1_ray_query_layout
hk_gen1_ray_query_layout(struct hk_device *dev, const nir_shader *nir,
                         VkShaderStageFlags next_stage_mask)
{
   return (struct agx_gen1_ray_query_layout){
      .payload_uniform = dev->gen1_rq_candidates ? 118 : 124,
      .bvh_uniform = 96,
      .status_uniform = dev->gen1_rq_candidates ? 128 :
                        dev->gen1_rq_instances ? 130 : 128,
      .instance_traversal = dev->gen1_rq_instances,
      .candidates = dev->gen1_rq_candidates,
      .as_table_first = dev->gen1_rq_instances ? HK_GEN1_AS_TABLE_AS_SLOTS : 1,
      .launch_header = dev->gen1_rq_candidates ? 0x210000 : 0x110000,
      .context_uniform = 160,
      .as_table_uniform = 164,
      .fragment_shader = dev->gen1_native_fragment_rq,
      .graphics_stage = dev->gen1_native_graphics_rq &&
         ((nir->info.stage == MESA_SHADER_VERTEX &&
           !(next_stage_mask & (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                                VK_SHADER_STAGE_GEOMETRY_BIT))) ||
          (nir->info.stage == MESA_SHADER_TESS_EVAL &&
           !(next_stage_mask & VK_SHADER_STAGE_GEOMETRY_BIT))),
      .compute_kernel = dev->gen1_native_graphics_rq,
      .position_fetch = dev->vk.enabled_features.rayTracingPositionFetch &&
                        dev->gen1_vertex_directory != NULL,
      .dispatch_group_count_offset = nir->info.stage == MESA_SHADER_COMPUTE
         ? hk_root_descriptor_offset(cs.group_count_addr) : 0,
   };
}

/*
 * A native ray query in the post-tessellation hardware vertex program does
 * not execute on G17P: every query-bearing TES case reads back zero, even
 * with a null acceleration structure, where no traversal is issued and only
 * the result store remains. The same lowering, uniform layout and render
 * state work for an API vertex shader drawn directly. Run such a TES in
 * Poly's compute variant instead, whose query context is the one TCS, GS and
 * compute queries use, and rasterize its outputs through a passthrough GS.
 */
/* Whether a GS or TES compiles under the G17P native ray-query profile.
 * The profile also runs passes on shaders without queries, and a shader
 * binary must not depend on which features its device enabled: shader
 * object binaries of one shader from devices with different features must
 * match. A device that enables no ray-tracing feature therefore compiles
 * the query-free shaders it can have the way a ray-tracing device does.
 */
static bool
hk_native_rq_profile(const struct hk_device *dev)
{
   return dev->gen1_native_rq ||
          (hk_g17p_rt_enabled(&dev->dev) &&
           !hk_rt_features_enabled(&dev->vk.enabled_features));
}

bool
hk_tes_ray_queries_in_compute(const struct hk_device *dev)
{
   return dev->dev.chip == AGX_CHIP_G17P &&
          dev->gen1_native_rq && dev->gen1_native_graphics_rq;
}

/* A uniform buffer does not change during a draw or dispatch. */
static bool
hk_ubo_loads_reorderable(nir_builder *b, nir_intrinsic_instr *intr,
                         UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_deref ||
       !nir_deref_mode_is(nir_src_as_deref(intr->src[0]), nir_var_mem_ubo))
      return false;

   nir_intrinsic_set_access(intr, nir_intrinsic_access(intr) |
                                     ACCESS_CAN_REORDER |
                                     ACCESS_NON_WRITEABLE);
   return true;
}

/* See image_load_needs_memory_path() in hk_nir_lower_descriptors.c: the
 * multi-cluster AGX2 GPUs (G13 and G14 Pro/Max/Ultra). */
static bool
hk_coherent_images_in_memory(const struct agx_device *dev)
{
   return agx_chip_isa_encoding(dev->chip) == AGX_ISA_ENCODING_AGX2 &&
          dev->params.num_clusters_total > 1;
}

static void
hk_lower_nir(struct hk_device *dev, nir_shader *nir,
             const struct vk_pipeline_robustness_state *rs, bool is_multiview,
             VkShaderStageFlags next_stage_mask, uint32_t set_layout_count,
             struct vk_descriptor_set_layout *const *set_layouts,
             enum hk_feature_key features)
{
   if (HK_PERF(dev, NOROBUST)) {
      rs = &vk_robustness_disabled;
   }

   const nir_opt_access_options access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &access_options);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_lower_input_attachments,
               &(nir_input_attachment_options){
                  .use_view_id_for_layer = is_multiview,
               });

      NIR_PASS(_, nir, nir_shader_tex_pass, lower_subpass_dim, nir_metadata_all,
               NULL);
      NIR_PASS(_, nir, nir_lower_wpos_center);

      if (agx_native_graphics(&dev->dev)) {
         NIR_PASS(_, nir, agx_nir_lower_g17p_helper_samples);
      }
   }

   if (is_multiview &&
       hk_lower_multiview_stage(nir->info.stage, next_stage_mask))
      hk_lower_multiview(nir, dev->dev.chip == AGX_CHIP_G17P);

   if (is_multiview && nir->info.stage == MESA_SHADER_FRAGMENT &&
       dev->dev.chip == AGX_CHIP_G17P)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_lower_multiview_fs_layer,
               nir_metadata_control_flow, NULL);

   if (nir->info.stage == MESA_SHADER_TESS_EVAL) {
      NIR_PASS(_, nir, nir_lower_patch_vertices,
               nir->info.tess.tcs_vertices_out, NULL);
   }

   const struct lower_ycbcr_state ycbcr_state = {
      .set_layout_count = set_layout_count,
      .set_layouts = set_layouts,
   };
   NIR_PASS(_, nir, nir_shader_tex_pass, lower_packed_ycbcr_size,
            nir_metadata_control_flow, (void *)&ycbcr_state);
   NIR_PASS(_, nir, nir_shader_tex_pass, lower_ycbcr_explicit,
            nir_metadata_none, (void *)&ycbcr_state);
   NIR_PASS(_, nir, nir_shader_tex_pass, lower_packed_ycbcr_center,
            nir_metadata_none, (void *)&ycbcr_state);
   NIR_PASS(_, nir, nir_vk_lower_ycbcr_tex, lookup_ycbcr_conversion,
            &ycbcr_state);
   NIR_PASS(_, nir, nir_shader_tex_pass, lower_ycbcr_chroma_blend,
            nir_metadata_none, (void *)&ycbcr_state);
   if (agx_native_graphics(&dev->dev)) {
      NIR_PASS(_, nir, nir_shader_tex_pass, lower_ycbcr_plane_range,
               nir_metadata_control_flow, (void *)&ycbcr_state);
   }

   /* Lower push constants before lower_descriptors */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);

   /* The cooperative-matrix lowering serves the 8x8 matmad on AGX3B
    * (G15/G16/G17P) as well as M5's AGX3C; it no-ops on shaders that carry no
    * cooperative matrix, so gating on the encoding is sufficient. */
   if (agx_chip_g17_m5(dev->dev.chip) ||
       agx_chip_isa_encoding(dev->dev.chip) == AGX_ISA_ENCODING_AGX3B)
      NIR_PASS(_, nir, agx_nir_lower_cmat);

   /* Turn cache flushes into image coherency bits while we still have derefs */
   NIR_PASS(_, nir, nir_lower_memory_model);

   /* G17P's texture/PBE accesses stay inside mapped memory, but native 2D
    * image accesses outside the Vulkan view can alias another texel in the
    * same image. Use the common NIR image-size predicate when the selected
    * pipeline robustness mode requires view-bounds semantics. Older Apple
    * generations already return the specified value in hardware. */
   const bool lower_g17p_robust_images =
      dev->dev.chip == AGX_CHIP_G17P &&
      rs->images != VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT;
   const bool lower_agx3_robust_fetches =
      (dev->dev.chip == AGX_CHIP_G17P || agx_chip_g17_m5(dev->dev.chip)) &&
      rs->images != VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT;
   const bool native_robust_images =
      agx_chip_explicit_robust_image_bounds(dev->dev.chip) &&
      rs->images != VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT;

   /* A G17P image load does not see a store made earlier in the same
    * invocation; forward the stored value. Run it before the robust lowering
    * so the loads and stores it tracks are still unconditional. */
   if (agx_chip_image_load_misses_own_store(dev->dev.chip) ||
       (agx_chip_g17_m5(dev->dev.chip) &&
        debug_get_bool_option("ASAHI_M5_IMAGE_STORE_FORWARD", false)))
      NIR_PASS(_, nir, hk_nir_forward_image_stores,
               lower_g17p_robust_images || native_robust_images ||
                  (features & HK_FEAT_NULL_DESCRIPTOR));
   NIR_PASS(_, nir, hk_nir_lower_robust_image_access,
            lower_g17p_robust_images);
   if (native_robust_images) {
      NIR_PASS(_, nir, nir_lower_robust_access_cb, should_lower_robust,
               hk_native_robust_oob_alpha, &native_robust_images);
   }
   if (native_robust_images) {
      const nir_lower_tex_options offsets = {.lower_txf_offset = true};
      NIR_PASS(_, nir, nir_lower_tex, &offsets);
      NIR_PASS(_, nir, nir_shader_tex_pass, hk_lower_native_robust_txf,
               nir_metadata_control_flow, NULL);
   }

   /* We must do early lowering before hk_nir_lower_descriptors, since this will
    * create lod_bias instructions.
    *
    * Only the Gen4 sampler word has no LOD-bias field. On every later
    * generation hk_CreateSampler writes the API bias into the hardware
    * sampler (SAMPLER_HAL200.lod_bias, SAMPLER_GEN5.lod_bias) and
    * keeps ordinary sampling in hardware. Lowering a zero software bias
    * into every shader would add a descriptor crawl that computes
    * nothing: it turns every tex into txb(load 0), and every txd into
    * gradients scaled by exp2(load 0). Skipping it is a no-op for the value
    * and keeps a genuine API bias or gradient as the shader's own operand.
    * G17P txb is handled separately below: its two biases must be added
    * before the sampler/instruction's fixed-point quantization.
    */
   const bool support_lod_bias =
      agx_chip_sampler_generation(dev->dev.chip) == AGX_SAMPLER_GENERATION_GEN4;

   /* G17P's texture_load has no witnessed integer-offset form, and a txf
    * offset is defined as plain integer coordinate addition, so fold it into
    * the coordinate before descriptor binding rather than carrying a backend
    * source the fetch encoder cannot express. Gallium already selects this
    * path; without it here, texelFetchOffset() reaches the packer as a
    * texture_load with a 32-bit immediate source, which trips
    * agx_validate_sources() in a debug build and, in a release build, reaches
    * an encoder with no offset field at all. */
   /* M5 keeps a depth comparison's bias in the sampler descriptor, as the
    * M5 path always has; only a Gen4 sampler moves it into the shader. */
   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip))
      NIR_PASS(_, nir, agx_nir_lower_g17p_texture_early, support_lod_bias,
               agx_chip_sampler_has_lod_bias(dev->dev.chip) ||
                  agx_chip_g17_m5(dev->dev.chip));
   else
      NIR_PASS(_, nir, agx_nir_lower_texture_early, support_lod_bias);

   if (agx_native_graphics(&dev->dev) &&
       agx_chip_sampler_has_lod_bias(dev->dev.chip))
      NIR_PASS(_, nir, nir_shader_tex_pass, hk_lower_g17p_sampler_bias,
               nir_metadata_control_flow, NULL);

   struct hk_instance *instance = hk_physical_device_instance(
      (struct hk_physical_device *)dev->vk.physical);

   if (instance->drirc.misc.image_view_min_lod &&
       (features & HK_FEAT_IMAGE_VIEW_MIN_LOD)) {
      NIR_PASS(_, nir, hk_nir_lower_image_view_min_lod);
   }

   if (((features & HK_FEAT_CUSTOM_BORDER) || agx_native_graphics(&dev->dev)) &&
       !HK_PERF(dev, NOBORDER)) {
      NIR_PASS(_, nir, agx_nir_lower_custom_border);
   }

   if (dev->vk.enabled_features.sparseResidencyBuffer &&
       agx_chip_g17_m5(dev->dev.chip)) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               lower_physical_sparse_load, nir_metadata_none, NULL);
   }

   /* Unbound pages read as zero through soft faults, but a dropped store
    * stays visible to later reads of its address, so storage-buffer stores
    * still check residency. */
   const bool sparse_store_guards =
      dev->vk.enabled_features.sparseResidencyBuffer &&
      hk_sparse_buffer_unmapped(&dev->dev);

   NIR_PASS(_, nir, hk_nir_lower_descriptors, rs, set_layout_count,
            set_layouts, sparse_store_guards, sparse_store_guards,
            dev->dev.sparse_ro_offset,
            hk_device_physical(dev)->scaled_sampling,
            agx_chip_sampler_has_lod_bias(dev->dev.chip),
            instance->drirc.misc.d3d_nearest_mip,
            hk_coherent_images_in_memory(&dev->dev),
            dev->dev.chip == AGX_CHIP_G17G || dev->dev.chip == AGX_CHIP_G17S);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo,
            hk_buffer_addr_format(rs->storage_buffers,
                                  rs->null_storage_buffer_descriptor,
                                  agx_chip_g17_m5(dev->dev.chip)));
   /* While UBO offsets are still 32-bit deref offsets, merge the loads of a
    * block that read the same record: once lowered, their 64-bit addresses
    * hide how they relate. Components of a vector become loads of the whole
    * vector, so that reads of one record's .xy and .z are the same load. A
    * uniform buffer does not change during a draw or dispatch, so its loads
    * may be reordered, as load_ubo always may: repeated ones are then merged
    * and loop invariant ones may leave the loop. */
   NIR_PASS(_, nir, nir_lower_array_deref_of_vec, nir_var_mem_ubo, NULL,
            nir_lower_direct_array_deref_of_vec_load);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_ubo_loads_reorderable,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_opt_cse);
   NIR_PASS(_, nir, nir_opt_load_store_vectorize,
            &(const nir_load_store_vectorize_options){
               .modes = nir_var_mem_ubo,
               .robust_modes =
                  rs->uniform_buffers ==
                        VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT
                     ? 0
                     : nir_var_mem_ubo,
               .callback = agx_mem_vectorize_cb,
            });
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo,
            hk_ubo_addr_format(rs->uniform_buffers));

   /* Ray queries lower to a software traversal loop of ALU and global loads.
    * It has to run after descriptor lowering, because rq_initialize's
    * acceleration-structure source must already be a resolved 64-bit address
    * by the time the traversal reads a node through it, and before shared
    * memory is laid out, because the pass introduces shader temporaries that
    * nir_lower_vars_to_ssa still has to promote. */
   /* Poly replays a GS in a hardware vertex shader. Keep native queries until
    * that split so the raster variant gets the graphics query ABI and Poly
    * cannot strip stores introduced by query lowering as API side effects.
    */
   /* A TES likewise has a hardware vertex variant and a CDM variant, so its
    * native queries are lowered per variant in hk_compile_nir().
    */
   if (nir->info.ray_queries > 0 &&
       !(dev->gen1_native_rq &&
         (nir->info.stage == MESA_SHADER_GEOMETRY ||
          nir->info.stage == MESA_SHADER_TESS_EVAL))) {
      if(dev->gen1_native_rq) {
         NIR_PASS(_, nir, nir_opt_ray_queries);
         NIR_PASS(_, nir, nir_opt_ray_query_ranges);
         const struct agx_gen1_ray_query_layout layout =
            hk_gen1_ray_query_layout(dev, nir, next_stage_mask);
         if(!agx_nir_lower_gen1_ray_query(nir,&layout)) {
            mesa_loge("Gen1 native ray query: unsupported shape or operation");
            return;
         }
      } else {
         NIR_PASS(_, nir, hk_nir_lower_ray_queries);
      }

      /* The pass replaces every ray-query variable with ordinary temporaries
       * and lowers every rq_* intrinsic, so the source count is now stale.
       * That is not cosmetic: the AGX3 backend refuses outright any shader
       * whose info.ray_queries is nonzero (`agx_compile.c:8610-8613`), and it
       * also derives uses_ray_traversal / ray_traversal_depth_nonzero from it
       * (`:8776-8781`).  Measured on the g17p drm-shim: an ordinary ray-query
       * compute pipeline already reaches agx_compile_shader_nir() with
       * ray_queries == 0 -- refreshed somewhere downstream of this call -- so
       * zeroing it here changes nothing for the path that measured 43/45 on
       * Neo.  The ray-tracing megakernel gets no such refresh, and without
       * this every ray query inside a ray-tracing-pipeline stage fails
       * creation with "AGX backend cannot encode compute shader for G17P". */
      nir->info.ray_queries = 0;
   }

   /* Before inserting bounds checks, we want to do a fair bit of optimization.
    * lower_load_global_constant_offset_instr has special optimizations for
    * constant offsets, so we want as many offsets to be constant as possible.
    */
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
   } while (progress);

   nir_move_options opts = nir_move_load_ssbo | nir_move_load_ubo;

   /* Unbounded storage reads are load_global here. Sinking the reorderable
    * ones to their uses keeps M5's large compute kernels from holding every
    * loaded struct across unrelated branches, which is where they spill.
    * nir_opt_sink never moves these into or out of a loop.
    */
   nir_move_options sink_opts = opts;
   if (agx_chip_g17_m5(dev->dev.chip))
      sink_opts |= nir_move_load_global;

   NIR_PASS(progress, nir, nir_opt_sink, sink_opts);
   NIR_PASS(progress, nir, nir_opt_move, opts);

   struct global_load_lowering global_loads = {
      .soft_fault = agx_has_soft_fault(&dev->dev),
      .storage_result_select = dev->dev.chip == AGX_CHIP_G17P ||
                               agx_chip_g17_m5(dev->dev.chip),
   };
   if (agx_chip_g17_m5(dev->dev.chip) && global_loads.soft_fault) {
      /* Reads of one descriptor share its base and size only after CSE. */
      NIR_PASS(_, nir, nir_opt_cse);
      NIR_PASS(_, nir, m5_merge_buffer_words);
   }
   NIR_PASS(_, nir, nir_shader_intrinsics_pass,
            lower_load_global_constant_offset_instr, nir_metadata_none,
            &global_loads);


   if (nir->info.stage == MESA_SHADER_TASK || nir->info.stage == MESA_SHADER_MESH) {
      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_task_payload,
               shared_var_info);
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_task_payload,
               nir_address_format_32bit_offset);
   }
   assert(nir->info.shared_size == 0);

   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
            shared_var_info);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
            nir_address_format_32bit_offset);

   if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
      /* Align everything up to 16B so we can write whole vec4s. */
      nir->info.shared_size = align(nir->info.shared_size, 16);
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory, nir->info.shared_size,
               16);

      /* We need to call lower_compute_system_values again because
       * nir_zero_initialize_shared_memory generates load_invocation_id which
       * has to be lowered to load_invocation_index.
       */
      NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
   }

   /* G17P's f32interp names its coefficient with an immediate field. Apple's
    * native compiler therefore lowers a dynamically indexed fragment-input
    * array to direct interpolations and selection. Do the same while the I/O
    * is still represented by derefs, so each resulting interpolation keeps a
    * constant coefficient and no temporary array reaches scratch memory.
    * AGX2 retains its native indirect fragment-input path. */
   nir_variable_mode lower_indirect_modes = 0;
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      lower_indirect_modes |= nir_var_shader_out;

      if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip))
         lower_indirect_modes |= nir_var_shader_in;
   } else if (nir->info.stage == MESA_SHADER_VERTEX) {
      lower_indirect_modes |= nir_var_shader_in | nir_var_shader_out;
   }

   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
            lower_indirect_modes, UINT32_MAX);

   /* SPIR-V represents a fragment gl_Layer read as load_layer_id, and the
    * generic AGX options deliberately keep that system-value form for AGX2.
    * AGX3 reads the API fragment layer as a flat input: Apple's shaders consume the
    * raster layer through a flat coefficient. Convert the sysval back to a
    * shader input here, after the generic system-value lowering but before IO
    * lowering, so the regular varying linker builds the coefficient binding. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip))) {
      const nir_lower_sysvals_to_varyings_options sysvals_to_varyings = {
         .layer_id = true,
      };
      NIR_PASS(_, nir, nir_lower_sysvals_to_varyings,
               &sysvals_to_varyings);
   }

   /* Tess levels are compact float arrays: nir_lower_io turns an indirect
    * element index into an IO offset counted in components, while poly's
    * TCS/TES buffer addressing counts IO offsets in slots. A control shader
    * writing gl_TessLevelOuter[gl_InvocationID] therefore stored levels 1/2
    * into other slots, left them zero and culled every patch. Make the
    * levels vectors so indexed accesses become per-component accesses. */
   if (agx_chip_g17_m5(dev->dev.chip) &&
       (nir->info.stage == MESA_SHADER_TESS_CTRL ||
        nir->info.stage == MESA_SHADER_TESS_EVAL))
      NIR_PASS(_, nir, nir_lower_tess_level_array_vars_to_vec);

   /* Mesh outputs are per-vertex/per-primitive arrays, so a component
    * write such as gl_MeshVerticesEXT[i].gl_Position.y reaches here as an
    * array deref of a vector, which nir_lower_io would store to component
    * 0. Turn those into masked vector accesses first. */
   if (nir->info.stage == MESA_SHADER_MESH)
      NIR_PASS(_, nir, nir_lower_array_deref_of_vec, nir_var_shader_out, NULL,
               nir_lower_direct_array_deref_of_vec_load |
               nir_lower_indirect_array_deref_of_vec_load |
               nir_lower_direct_array_deref_of_vec_store |
               nir_lower_indirect_array_deref_of_vec_store);
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            glsl_type_size,
            nir_lower_io_lower_64bit_to_32 |
               nir_lower_io_use_interpolated_input_intrinsics);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_viewport_fs,
               nir_metadata_control_flow, NULL);
   }

   if (agx_chip_descriptor_generation(dev->dev.chip) == 9) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_scaled_storage,
               nir_metadata_none, hk_device_physical(dev));
      NIR_PASS(_, nir, nir_shader_tex_pass, lower_scaled_sample,
               hk_scaled_sample_guard(nir) ? nir_metadata_none
                                           : nir_metadata_control_flow,
               hk_device_physical(dev));
   }

   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
      /* hk_nir_lower_robust_image_access() leaves formatless loads to this
       * pass. */
      NIR_PASS(_, nir, agx_nir_lower_texture_with_buffer_metadata,
               agx_chip_descriptor_generation(dev->dev.chip) |
                  (agx_chip_texture_descriptor_spare_word(dev->dev.chip) &&
                         !nir->info.internal
                      ? AGX_NIR_TEXTURE_LEVEL_ZERO_EXTENT
                      : 0),
               lower_agx3_robust_fetches, dev->dev.sparse_ro_offset,
               dev->vk.enabled_features.sparseResidencyBuffer);
   } else {
      if (dev->physical_sparse_map &&
          hk_sparse_texel_residency_from_map(&dev->dev)) {
         NIR_PASS(_, nir, nir_shader_instructions_pass,
                  lower_sparse_texel_buffer_residency, nir_metadata_none,
                  (void *)dev);
      }

      /* G18G storage images use the 96-byte record, whose image-atomic
       * word hk_descriptor_set.c writes after the texture/PBE pair. */
      NIR_PASS(_, nir, agx_nir_lower_texture_for_generation,
               agx_chip_descriptor_generation(dev->dev.chip) |
                  (dev->dev.chip == AGX_CHIP_G18G
                      ? AGX_NIR_TEXTURE_IMAGE_SIDEBAND
                      : 0));
   }
   NIR_PASS(_, nir, agx_nir_lower_multisampled_image_store);

   /* Mesh adds its native UVB shared arena after this frontend lowering.
    * Keep application shared accesses byte addressed until that arena exists;
    * the mesh branch then preprocesses every shared access exactly once.
    * Reprocessing here and there divides preexisting element indices again.
    * The M5 task (object) branch likewise preprocesses after adding its
    * child-grid words, and the G17P task kernel after its payload lowering. */
   if (nir->info.stage != MESA_SHADER_MESH &&
       !(nir->info.stage == MESA_SHADER_TASK &&
         (agx_chip_g17_m5(dev->dev.chip) || dev->dev.chip == AGX_CHIP_G17P))) {
      if (agx_native_graphics(&dev->dev))
         agx_preprocess_nir_native(nir, dev->dev.chip);
      else if (dev->dev.chip == AGX_CHIP_G18G)
         agx_preprocess_nir_g18g(nir);
      else if (agx_chip_g17_m5(dev->dev.chip))
         agx_preprocess_nir_m5(nir);
      else
         agx_preprocess_nir(nir);
   }

   nir_opt_peephole_select_options peephole_select_options = {
      .limit = 0,
      .discard_ok = true,
   };
   NIR_PASS(_, nir, nir_opt_peephole_select, &peephole_select_options);
   NIR_PASS(_, nir, nir_opt_if,
            nir_opt_if_optimize_phi_true_false | nir_opt_if_avoid_64bit_phis);
}

/* Report a failed executable-preamble upload, so the caller's error path
 * destroys the half-built shader instead of keeping one with preamble_addr
 * 0. */
static VkResult
hk_upload_shader(struct hk_device *dev, struct hk_shader *shader)
{
   /* Allocator addresses can be recycled after a shader is destroyed while
    * earlier commands still own an upload of its constants. Do not key those
    * uploads by a shader or malloc pointer. This is not serialized. */
   static uint64_t next_constant_data_id;
   if (shader->data_size && !shader->constant_data_id)
      shader->constant_data_id = p_atomic_inc_return(&next_constant_data_id);

   if (shader->b.info.has_preamble || shader->b.info.rodata.size_16) {
      /* TODO: Do we wnat to compact? Revisit when we rework prolog/epilogs. */
      size_t size = shader->b.info.binary_size;
      assert(size > 0);

      shader->bo = hk_code_alloc(dev, size, &shader->bo_offset);
      if (!shader->bo)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;

      memcpy((uint8_t *)agx_bo_map(shader->bo) + shader->bo_offset,
             shader->b.binary, size);
      agx_bo_assert_range(shader->bo, shader->bo_offset, size);
      shader->preamble_addr =
         hk_shader_code_va(shader) + shader->b.info.preamble_offset;
   }

   if (!shader->linked.ht) {
      /* If we only have a single variant, link now. */
      shader->only_linked = hk_fast_link_vertex(dev, shader, NULL, 0, 0);
      if (!shader->only_linked)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   if (shader->info.stage == MESA_SHADER_FRAGMENT) {
      agx_pack_fragment_face_2(&shader->frag_face, 0, &shader->b.info);
   }

   agx_pack(&shader->counts, COUNTS, cfg) {
      cfg.uniform_register_count = shader->b.info.push_count;
      cfg.preshader_register_count = shader->b.info.nr_preamble_gprs;
      cfg.sampler_state_register_count = agx_translate_sampler_state_count(
         shader->b.info.sampler_state_count, false);
      cfg.texture_state_register_count = shader->b.info.texture_state_count;
   }

   return VK_SUCCESS;
}

DERIVE_HASH_TABLE(hk_fast_link_key_vs);
DERIVE_HASH_TABLE(hk_fast_link_key_fs);

static VkResult
hk_init_link_ht(struct hk_shader *shader, mesa_shader_stage sw_stage)
{
   simple_mtx_init(&shader->linked.lock, mtx_plain);

   bool multiple_variants =
      sw_stage == MESA_SHADER_VERTEX || sw_stage == MESA_SHADER_FRAGMENT;

   if (!multiple_variants)
      return VK_SUCCESS;

   if (sw_stage == MESA_SHADER_VERTEX)
      shader->linked.ht = hk_fast_link_key_vs_table_create(NULL);
   else
      shader->linked.ht = hk_fast_link_key_fs_table_create(NULL);

   return (shader->linked.ht == NULL) ? VK_ERROR_OUT_OF_HOST_MEMORY
                                      : VK_SUCCESS;
}

bool
hk_shader_ensure_link_ht_locked(struct hk_shader *shader)
{
   simple_mtx_assert_locked(&shader->linked.lock);

   if (shader->linked.ht)
      return true;
   if (!shader->linked.lazy_ht)
      return false;

   if (shader->info.stage == MESA_SHADER_VERTEX)
      shader->linked.ht = hk_fast_link_key_vs_table_create(NULL);
   else if (shader->info.stage == MESA_SHADER_FRAGMENT)
      shader->linked.ht = hk_fast_link_key_fs_table_create(NULL);
   else
      return false;

   if (shader->linked.ht)
      shader->linked.lazy_ht = false;
   return shader->linked.ht != NULL;
}

static bool
lower_uniforms(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   /* Root is first, descriptor sets follow immediately. */
   unsigned *root_ = data;
   unsigned root = *root_;
   unsigned sets = root + 4;

   if (intr->intrinsic == nir_intrinsic_bindless_image_agx ||
       intr->intrinsic == nir_intrinsic_bindless_sampler_agx) {
      /* Change of units from sets to uniforms */
      nir_intrinsic_set_desc_set(intr,
                                 sets + (nir_intrinsic_desc_set(intr) * 4));
      return true;
   }

   if (intr->intrinsic != nir_intrinsic_load_root_agx &&
       intr->intrinsic != nir_intrinsic_load_descriptor_set_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *rep;

   if (intr->intrinsic == nir_intrinsic_load_descriptor_set_agx) {
      unsigned s = nir_intrinsic_desc_set(intr);
      rep = nir_load_preamble(b, 1, 64, .base = sets + (4 * s));
   } else {
      rep = nir_load_preamble(b, 1, 64, .base = root);
   }

   nir_def_replace(&intr->def, rep);
   return true;
}

/* Divide a descriptor byte offset by the G17P PBE record size. Descriptor
 * layouts keep every storage PBE on a record boundary, and the offset is built
 * from the binding offset and stride, so it divides term by term. Rebuilding
 * the expression keeps an ordinary array index on the multiply-add the backend
 * already encodes instead of adding a run-time shift.
 */
static nir_def *
hk_g17p_pbe_record(nir_builder *b, nir_scalar s)
{
   const unsigned record_log2 = util_logbase2(HK_G17P_PBE_RECORD_SIZE);

   if (nir_scalar_is_const(s)) {
      const uint64_t v = nir_scalar_as_uint(s);
      return (v % HK_G17P_PBE_RECORD_SIZE) == 0
                ? nir_imm_int(b, v >> record_log2)
                : NULL;
   }

   if (!nir_scalar_is_alu(s))
      return NULL;

   nir_op op = nir_scalar_alu_op(s);
   nir_scalar x = nir_scalar_chase_alu_src(s, 0);
   nir_scalar y = nir_scalar_chase_alu_src(s, 1);

   switch (op) {
   case nir_op_iadd: {
      nir_def *x_record = hk_g17p_pbe_record(b, x);
      nir_def *y_record = x_record ? hk_g17p_pbe_record(b, y) : NULL;
      return y_record ? nir_iadd(b, x_record, y_record) : NULL;
   }

   case nir_op_imul:
   case nir_op_amul:
      if (nir_scalar_is_const(x))
         SWAP(x, y);
      if (!nir_scalar_is_const(y) ||
          (nir_scalar_as_uint(y) % HK_G17P_PBE_RECORD_SIZE) != 0)
         return NULL;
      return nir_build_alu2(b, op, nir_mov_scalar(b, x),
                            nir_imm_int(b, nir_scalar_as_uint(y) >>
                                           record_log2));

   case nir_op_ishl:
      if (!nir_scalar_is_const(y) || nir_scalar_as_uint(y) < record_log2)
         return NULL;
      return nir_ishl_imm(b, nir_mov_scalar(b, x),
                          nir_scalar_as_uint(y) - record_log2);

   default:
      return NULL;
   }
}

/* Honeykrisp names a storage image by the byte offset of its PBE in the
 * descriptor set. G17P's imgwci instead takes a PBE heap from a uniform and
 * the index of a PBE record in that heap, so the set itself is the heap and
 * the store carries its record index.
 */
static bool
hk_g17p_lower_storage_records(nir_builder *b, nir_intrinsic_instr *intr,
                              UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_bindless_image_store)
      return false;

   nir_intrinsic_instr *handle = nir_src_as_intrinsic(intr->src[0]);
   assert(handle->intrinsic == nir_intrinsic_bindless_image_agx);

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *record =
      hk_g17p_pbe_record(b, nir_get_scalar(handle->src[0].ssa, 0));
   if (!record) {
      record = nir_ushr_imm(b, handle->src[0].ssa,
                            util_logbase2(HK_G17P_PBE_RECORD_SIZE));
   }

   nir_src_rewrite(&intr->src[0],
                   nir_bindless_image_agx(
                      b, record, .desc_set = nir_intrinsic_desc_set(handle)));
   return true;
}

/* A first-provoking triangle fan stores the flat value in coefficient B;
 * every other supported topology stores it in C. agx_nir_lower_interpolation
 * introduces this intrinsic after descriptor lowering, so lower it to the
 * draw-root provoking selector here. The later G17P root-resource pass then
 * publishes the root pointer through the normal native uniform contract. */
static bool
hk_g17p_lower_is_first_fan(nir_builder *b, nir_intrinsic_instr *intr,
                           UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_is_first_fan_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   const unsigned offset = hk_root_descriptor_offset(draw.provoking);
   nir_def *addr = nir_iadd(
      b, nir_load_root_agx(b), nir_imm_int64(b, offset));
   nir_def *provoking = nir_load_global_constant(
      b, 1, 16, addr, .align_mul = 2, .access = ACCESS_CAN_SPECULATE);
   nir_def_replace(&intr->def, nir_ieq_imm(b, provoking, 1));
   return true;
}


static bool
kill_psiz_write(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_PSIZ)
      return false;

   if (nir_remove_sysval_output(intr, MESA_SHADER_FRAGMENT)) {
      b->shader->info.outputs_written &= ~VARYING_BIT_PSIZ;
   }

   return true;
}

static bool
hk_mesh_default_point_size(nir_builder *b, nir_intrinsic_instr *intr,
                           void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_per_vertex_output ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_POS)
      return false;

   b->cursor = nir_after_instr(&intr->instr);
   nir_store_per_vertex_output(b, nir_imm_float(b, 1.0), intr->src[1].ssa,
                               nir_imm_int(b, 0), .write_mask = 1,
                               .src_type = nir_type_float32,
                               .io_semantics = {
                                  .location = VARYING_SLOT_PSIZ,
                                  .num_slots = 1});
   b->shader->info.outputs_written |= VARYING_BIT_PSIZ;
   return true;
}

/* Whether a lowered store writes this varying slot. */
static bool
hk_store_writes_slot(const nir_intrinsic_instr *intr, gl_varying_slot slot)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
   case nir_intrinsic_store_per_primitive_output:
   case nir_intrinsic_store_per_view_output:
      break;
   default:
      return false;
   }
   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   return sem.location <= slot && slot < sem.location + sem.num_slots;
}

/* A fragment shader reading gl_Layer links that input to the layer word of
 * the mesh shader's UVS, and an M5 mesh shader that writes neither
 * gl_PrimitiveLayerEXT nor gl_ViewportIndex has no such word. A vertex shader
 * compiled without its fragment shader gets a zero layer output for that
 * reason (agx_nir_lower_default_layer()). Give an M5 mesh shader compiled
 * without its fragment shader the same: store gl_PrimitiveLayerEXT = 0 for
 * every primitive the workgroup may output, as an application would, so that
 * agx_m5_mesh_gather_varyings() and agx_nir_lower_m5_mesh() treat it like an
 * application-written per-primitive layer. The stores come first in main,
 * where every invocation of the workgroup executes them. A written viewport
 * already exports a zero layer word.
 *
 * Runs on lowered I/O, after nir_lower_io_to_scalar(), and returns whether
 * it added the stores.
 */
static bool
hk_m5_mesh_default_layer(nir_shader *nir)
{
   const unsigned prims = nir->info.mesh.max_primitives_out;
   const unsigned threads = nir->info.workgroup_size[0] *
                            nir->info.workgroup_size[1] *
                            nir->info.workgroup_size[2];
   if (nir->info.stage != MESA_SHADER_MESH ||
       (nir->info.outputs_written & (VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT)) ||
       nir->info.workgroup_size_variable || prims == 0 || threads == 0)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (hk_store_writes_slot(intr, VARYING_SLOT_LAYER) ||
             hk_store_writes_slot(intr, VARYING_SLOT_VIEWPORT))
            return false;
      }
   }

   /* Invocation i stores primitives i, i + threads, ... below prims. */
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *lane = nir_load_local_invocation_index(&b);
   for (unsigned first = 0; first < prims; first += threads) {
      nir_def *prim = nir_iadd_imm(&b, lane, first);
      nir_if *partial = NULL;
      if (first + threads > prims)
         partial = nir_push_if(&b, nir_ult_imm(&b, prim, prims));
      /* nir_lower_io's form of a gl_PrimitiveLayerEXT store. */
      nir_store_per_primitive_output(&b, nir_imm_int(&b, 0), prim,
                                     nir_imm_int(&b, 0), .base = 0,
                                     .range = 1, .write_mask = 1,
                                     .component = 0,
                                     .src_type = nir_type_int32,
                                     .io_semantics = {
                                        .location = VARYING_SLOT_LAYER,
                                        .num_slots = 1});
      if (partial)
         nir_pop_if(&b, partial);
   }

   nir->info.outputs_written |= VARYING_BIT_LAYER;
   nir->info.per_primitive_outputs |= VARYING_BIT_LAYER;
   return nir_progress(true, impl, nir_metadata_none);
}

static bool
kill_mesh_triangle_psiz_write(nir_builder *b, nir_intrinsic_instr *intr,
                               void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_per_vertex_output ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_PSIZ)
      return false;

   nir_instr_remove(&intr->instr);
   b->shader->info.outputs_written &= ~VARYING_BIT_PSIZ;
   return true;
}

static bool
hk_nir_reads_intrinsic(const nir_shader *nir, nir_intrinsic_op op)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               return true;
         }
      }
   }

   return false;
}

static bool
hk_nir_uses_root(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_load_root_agx)
               return true;
         }
      }
   }

   return false;
}

/* The value of an integer offset built from immediates: the descriptor
 * lowering adds, scales and widens immediates without folding them. */
static bool
hk_nir_constant_offset(nir_scalar s, unsigned depth, uint64_t *value)
{
   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_const(s)) {
      *value = nir_scalar_as_uint(s);
      return true;
   }
   if (!depth || !nir_scalar_is_alu(s))
      return false;

   const nir_op op = nir_scalar_alu_op(s);
   uint64_t a, b = 0;
   if (!hk_nir_constant_offset(nir_scalar_chase_alu_src(s, 0), depth - 1, &a))
      return false;
   if (nir_op_infos[op].num_inputs == 2 &&
       !hk_nir_constant_offset(nir_scalar_chase_alu_src(s, 1), depth - 1, &b))
      return false;

   switch (op) {
   case nir_op_u2u32:
   case nir_op_u2u64:
      *value = a;
      break;
   case nir_op_iadd:
      *value = a + b;
      break;
   case nir_op_imul:
   case nir_op_amul:
      *value = a * b;
      break;
   case nir_op_ishl:
      *value = a << (b & (s.def->bit_size - 1));
      break;
   default:
      return false;
   }
   *value &= BITFIELD64_MASK(s.def->bit_size);
   return true;
}

/* The HK_ROOT_READS_* classes of the root bytes one use of load_root_agx can
 * read: an immediate offset loaded as is falls in the classes of its bytes,
 * anything else could read any byte. */
static uint8_t
hk_nir_root_use_reads(nir_src *use)
{
   const uint8_t any = HK_ROOT_READS_ADDRESS | HK_ROOT_READS_VERTEX_INPUT |
                       HK_ROOT_READS_DYNAMIC_BUFFERS;
   if (nir_src_is_if(use) ||
       nir_src_use_instr(use)->type != nir_instr_type_alu)
      return any;

   nir_alu_instr *add = nir_instr_as_alu(nir_src_use_instr(use));
   if (add->op != nir_op_iadd)
      return any;

   const unsigned other = use == &add->src[0].src ? 1 : 0;
   uint64_t offset;
   if (!hk_nir_constant_offset(nir_get_scalar(add->src[other].src.ssa,
                                              add->src[other].swizzle[0]),
                               8, &offset))
      return any;

   uint8_t reads = 0;
   nir_foreach_use_including_if(load_use, &add->def) {
      if (nir_src_is_if(load_use) ||
          nir_src_use_instr(load_use)->type != nir_instr_type_intrinsic)
         return any;

      nir_intrinsic_instr *load =
         nir_instr_as_intrinsic(nir_src_use_instr(load_use));
      if ((load->intrinsic != nir_intrinsic_load_global_constant &&
           load->intrinsic != nir_intrinsic_load_global) ||
          load_use != &load->src[0])
         return any;

      reads |= hk_root_reads_of(
         offset, offset + load->def.num_components * load->def.bit_size / 8);
   }
   return reads;
}

/* hk_shader_info::agx3_root_reads, gathered where hk_nir_uses_root() is. */
static uint8_t
hk_nir_root_reads(const nir_shader *nir)
{
   uint8_t reads = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic !=
                   nir_intrinsic_load_root_agx)
               continue;

            nir_foreach_use_including_if(use,
                                         &nir_instr_as_intrinsic(instr)->def)
               reads |= hk_nir_root_use_reads(use);
         }
      }
   }

   return reads;
}

/* The layer and viewport index each occupy a 16-bit half of the shared UVS
 * scalar. Apple's G17P compiler saturates a wider index into its half rather
 * than truncating it: every vertex program writing render_target_array_index
 * or viewport_array_index from a value that may exceed 16 bits carries one
 * isel, identical to the one it emits for an explicit min(value, 0xffff),
 * which it drops once the value is known to fit (value & 0xffff). metal-nt
 * -arch applegpu_g17p, layer = vid: 02012e030502; layer = min(vid, 0xffff):
 * the same single 02012e030502; layer = vid & 0xffff: no isel. */
static bool
hk_g17p_saturate_layer_viewport(nir_builder *b, nir_intrinsic_instr *intr,
                                UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   const unsigned location = nir_intrinsic_io_semantics(intr).location;
   nir_def *value = intr->src[0].ssa;
   if ((location != VARYING_SLOT_LAYER && location != VARYING_SLOT_VIEWPORT) ||
       value->bit_size <= 16)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(&intr->src[0],
                   nir_umin(b, value, nir_imm_intN_t(b, 0xffff,
                                                     value->bit_size)));
   return true;
}

static void
hk_lower_hw_vs(struct hk_device *dev, nir_shader *nir, struct hk_shader *shader,
               bool kill_psiz, bool layer_as_varying, bool default_layer,
               bool clip_m1_1, int layer_zero_root_offset)
{
   /* Point size must be clamped, excessively large points don't render
    * properly on G13.
    *
    * Must be synced with pointSizeRange.
    */
   NIR_PASS(_, nir, nir_lower_point_size, 1.0f, 511.95f);

   if (kill_psiz) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, kill_psiz_write,
               nir_metadata_control_flow, NULL);
   }

   if (clip_m1_1)
      NIR_PASS(_, nir, agx_nir_lower_clip_m1_1);

   NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_out, NULL, NULL);
   NIR_PASS(_, nir, agx_nir_lower_cull_distance_vs);

   /* Native fragment Layer reads use a flat coefficient, including for
    * independently compiled stages whose VS does not explicitly write Layer.
    * This program is the last stage before the rasterizer whatever API stage
    * it came from, so the same holds for tessellation and geometry. */
   shader->info.g17p_default_layer_zero =
      layer_as_varying && !nir->info.layer_viewport_relative &&
      !(nir->info.outputs_written & VARYING_BIT_LAYER);
   if (layer_as_varying && default_layer)
      NIR_PASS(_, nir, agx_nir_lower_default_layer);

   /* On G17P a vertex shader that stores a varying at a UVS index it loaded
    * at run time can have the varying land in the wrong place while another
    * context renders, so give a shader compiled without its fragment shader
    * the canonical layout, whose indices are constants.
    */
   NIR_PASS(_, nir, agx_nir_lower_uvs, &shader->info.uvs,
            layer_as_varying, dev->dev.chip == AGX_CHIP_G17P,
            layer_zero_root_offset);
}

/* G17P unrolls per-sample fragment execution for a pipeline that fixes two or
 * four samples. Every sample's colours, depth and stencil need their own
 * export registers inside the register file.
 *
 * The fragment ABI gives each colour location a four-component slot, and the
 * unroll repeats the whole set once per sample, so eight two-component
 * targets at four samples ask for 276 halves of a 192-half file -- 128 of
 * them for components the shader never writes. Pack the exports by what each
 * location actually writes and the same fragment needs 148, which is what
 * lets it run per sample at all. hk_compile_nir() lays out the same thing
 * from the link info once the split has recorded it exactly.
 */
static unsigned
hk_native_fs_samples(enum agx_chip chip, const struct hk_fs_key *key)
{
   return agx_chip_g17_m5(chip) ? key->m5_position_samples : key->g17p_samples;
}

static unsigned
hk_native_fs_registers(enum agx_chip chip)
{
   return agx_chip_g17_m5(chip) ? AGX_NUM_REGS : AGX3B_NUM_REGS;
}

static bool
hk_native_sample_shading_supported(const nir_shader *nir,
                               const struct hk_fs_key *key, enum agx_chip chip)
{
   const unsigned samples = hk_native_fs_samples(chip, key);
   if (samples != 2 && samples != 4 &&
       !(agx_chip_g17_m5(chip) && samples == 8))
      return false;

   uint8_t locations =
      (nir->info.outputs_written >> FRAG_RESULT_DATA0) & BITFIELD_MASK(8);
   if (nir->info.fs.color_is_dual_source)
      locations |= BITFIELD_BIT(1);

   uint8_t gathered = 0, size_32 = 0;
   uint32_t comps = 0;
   agx_nir_gather_fs_colour_exports(nir, &gathered, &size_32, &comps);

   /* Charge a location the gather did not see the ABI's whole slot, so this
    * never admits a shader the exact layout would not fit. */
   u_foreach_bit(loc, (unsigned)(locations & ~gathered)) {
      comps |= 0xfu << (4 * loc);
      size_32 |= BITFIELD_BIT(loc);
   }

   struct agx_g17p_colour_layout layout;

   agx_g17p_colour_layout_init(&layout, locations, size_32, comps, false);
   if (agx_g17p_sample_exports_end(&layout, samples) <=
       hk_native_fs_registers(chip))
      return true;

   agx_g17p_colour_layout_init(&layout, locations, size_32, comps, true);
   return agx_g17p_sample_exports_end(&layout, samples) <=
          hk_native_fs_registers(chip) ||
          (agx_chip_g17_m5(chip) && samples == 8);
}

static void
hk_g17p_vertex_compile_info_init(const struct hk_shader *shader,
                                 bool constant_program_allowed,
                                 struct hk_agx3_g17p_vertex_compile_info *info)
{
   *info = (struct hk_agx3_g17p_vertex_compile_info){
      .hardware_vertex = shader->b.info.stage == MESA_SHADER_VERTEX,
      .has_preamble = shader->b.info.has_preamble,
      .constant_program_allowed = constant_program_allowed,
      .preamble_staging_known = shader->b.info.agx3_preamble_staging_known,
      .preamble_staging_halfs = shader->b.info.agx3_preamble_staging_halfs,
      .nr_preamble_gprs = shader->b.info.nr_preamble_gprs,
      .preamble_scratch_size = shader->b.info.preamble_scratch_size,
      .uvs_written = shader->info.uvs.written,
      .position_off = shader->info.uvs.group_offs[UVS_POSITION],
      .varyings_off = shader->info.uvs.group_offs[UVS_VARYINGS],
      .psiz_off = shader->info.uvs.group_offs[UVS_PSIZ],
      .layer_viewport_off = shader->info.uvs.group_offs[UVS_LAYER_VIEWPORT],
      .clip_off = shader->info.uvs.group_offs[UVS_CLIP_DIST],
      .uvs_size = shader->info.uvs.size,
      .user_size = shader->info.uvs.user_size,
   };
}

/*
 * Compile a G17P hardware vertex program with its native constant program.
 * The compiler hoists uniform-only work into the preamble (nothing, for a
 * shader without any) and the launch runs it once per draw.
 *
 * A preamble outside the bounded native protocol -- a CProg that spills
 * beyond its imported-input TPS frame, a texture/sampler state register, or
 * TI resources the VertexProgramVariant cannot encode -- is not launchable.
 * Such a shader is reported loudly and compiled without the preamble instead
 * of failing pipeline creation; every occurrence is a protocol gap to close,
 * not an alternative path (the bench corpus hits none). The compile uses a
 * clone so that guard keeps the original NIR. On success the clone replaces
 * `*nir`; otherwise `shader->b` is cleared and `*nir` is untouched.
 */
static bool
hk_g17p_compile_vs_cprog(struct hk_device *dev, nir_shader **nir,
                         const struct agx_shader_key *key,
                         struct hk_shader *shader)
{
   nir_shader *attempt = nir_shader_clone(NULL, *nir);
   if (!attempt)
      return false;

   struct agx_shader_key cprog_key = *key;
   cprog_key.no_preamble = false;
   cprog_key.agx3_native_cprog = true;

   const char *declined = "compile";
   if (agx_compile_shader_nir(attempt, &cprog_key, &shader->b)) {
      const struct agx_shader_info *info = &shader->b.info;
      struct hk_agx3_g17p_vertex_compile_info compiler_info;
      struct agx_vdm_state_update_program_info program = {0};
      hk_g17p_vertex_compile_info_init(shader, true, &compiler_info);

      if (!info->has_preamble)
         declined = NULL;
      else if (!hk_g17p_native_cprog_qualified(info))
         declined = "cprog-shape";
      else if (info->texture_state_count != 0 || info->uses_sampler_heap ||
               (info->sampler_state_count != 0 &&
                !(shader->info.agx3_indexed_sampler_valid &&
                  info->sampler_state_count == 1)))
         declined = "texture-state";
      else if (!hk_agx3_g17p_vertex_program_info(dev->dev.chip,
                                                 &compiler_info, &program))
         declined = "vdm-state";
      else
         declined = NULL;

      if (agx_get_compiler_debug() & AGX_DBG_SHADERDB) {
         fprintf(stderr,
                 "HK G17P VS CProg: %s%s preamble=%uB inputs=%u tps=%uB "
                 "gprs=%u staging=%u uniforms=%u main=%uB state=%u/%u\n",
                 declined ? "declined:"
                 : info->has_preamble ? "admitted" : "none",
                 declined ? declined : "",
                 info->preamble_size, info->agx3_preamble_inputs.count,
                 info->preamble_scratch_size, info->nr_preamble_gprs,
                 info->agx3_preamble_staging_halfs, info->push_count,
                 info->main_size,
                 program.template_info.raw_state_load_count_12,
                 program.template_info.raw_state_load_count_18);
      }

      if (!declined) {
         ralloc_free(*nir);
         *nir = attempt;
         return true;
      }

      mesa_loge("G17P vertex constant program not launchable (%s): "
                "preamble=%uB tps=%uB inputs=%uB gprs=%u staging_known=%u "
                "textures=%u samplers=%u; compiling without it",
                declined, info->preamble_size, info->preamble_scratch_size,
                info->agx3_preamble_inputs.storage_bytes,
                info->nr_preamble_gprs, info->agx3_preamble_staging_known,
                info->texture_state_count, info->sampler_state_count);
      free(shader->b.binary);
   } else {
      if (agx_get_compiler_debug() & AGX_DBG_SHADERDB)
         fprintf(stderr, "HK G17P VS CProg: declined:%s\n", declined);
      mesa_loge("G17P vertex constant program failed to compile (%s); "
                "compiling without it", agx_last_compile_refusal());
   }

   memset(&shader->b, 0, sizeof(shader->b));
   ralloc_free(attempt);
   return false;
}

/*
 * The fragment constant program (formed when a fragment reads
 * descriptor-backed memory it can hoist) is launched by the same bounded
 * native protocol as the vertex one, and hk_g17p_basic_graphics_program_
 * supported() refuses every draw of a fragment whose preamble falls outside
 * it -- a CProg that spills beyond its imported-input TPS frame above all.
 * Decide that once, here, exactly as hk_g17p_compile_vs_cprog() does for the
 * vertex stage: compile a clone with the constant program and keep it only
 * when it is launchable; otherwise build the fragment the way a
 * push-constant-only fragment is built (no preamble). A shader whose
 * constant program does not compile at all (an
 * unencodable preamble instruction) takes the same route.
 *
 * On success the clone replaces `*nir`. Otherwise `shader->b` is cleared,
 * `*nir` is untouched and `key` (and the monolithic relink key) no longer
 * asks for a preamble.
 */
static bool
hk_g17p_compile_fs_cprog(nir_shader **nir, struct agx_shader_key *key,
                         struct hk_shader *shader)
{
   nir_shader *attempt = nir_shader_clone(NULL, *nir);
   if (!attempt)
      return false;

   const char *declined = "compile";
   if (agx_compile_shader_nir(attempt, key, &shader->b)) {
      const struct agx_shader_info *info = &shader->b.info;
      if (!info->has_preamble || hk_g17p_native_cprog_qualified(info)) {
         ralloc_free(*nir);
         *nir = attempt;
         return true;
      }

      declined = "cprog-shape";
      mesa_logw("G17P fragment constant program not launchable (%s): "
                "preamble=%uB tps=%uB inputs=%uB gprs=%u staging_known=%u; "
                "compiling without it",
                declined, info->preamble_size, info->preamble_scratch_size,
                info->agx3_preamble_inputs.storage_bytes,
                info->nr_preamble_gprs, info->agx3_preamble_staging_known);
      free(shader->b.binary);
   } else {
      mesa_logw("G17P fragment constant program failed to compile (%s); "
                "compiling without it", agx_last_compile_refusal());
   }

   memset(&shader->b, 0, sizeof(shader->b));
   ralloc_free(attempt);

   key->no_preamble = true;
   key->g17p_native_uniforms = false;
   if (shader->g17p_mono_nir) {
      shader->g17p_mono_key.no_preamble = true;
      shader->g17p_mono_key.g17p_native_uniforms = false;
   }
   return false;
}

/* This deliberately accepts only direct constant-offset push loads. The root
 * access set is unchanged when their arithmetic moves into a constant program. */
static bool
hk_m5_push_colour_load(const nir_intrinsic_instr *load)
{
   if (load->intrinsic != nir_intrinsic_load_global_constant ||
       load->def.bit_size != 32 || load->def.num_components > 4)
      return false;

   nir_scalar address = nir_scalar_chase_movs(nir_get_scalar(load->src[0].ssa, 0));
   if (!nir_scalar_is_alu(address) || nir_scalar_alu_op(address) != nir_op_iadd)
      return false;

   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar root = nir_scalar_chase_movs(nir_scalar_chase_alu_src(address, i));
      uint64_t offset;
      if (nir_def_instr(root.def)->type != nir_instr_type_intrinsic ||
          !hk_nir_constant_offset(nir_scalar_chase_alu_src(address, 1 - i),
                                  8, &offset))
         continue;
      const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(nir_def_instr(root.def));
      const uint64_t start = hk_root_descriptor_offset(push);
      if (intr->intrinsic == nir_intrinsic_load_preamble &&
          intr->def.bit_size == 64 && intr->def.num_components == 1 &&
          nir_intrinsic_base(intr) == AGX_ABI_FUNI_ROOT && offset >= start &&
          offset - start <= HK_MAX_PUSH_SIZE &&
          load->def.num_components * 4 <= HK_MAX_PUSH_SIZE - (offset - start))
         return true;
   }
   return false;
}

/* Keep straight-line uniform RT0 exports without descriptors or hidden effects.
 * The clone retains the original clamps and floating-point controls. */
static bool
hk_m5_uniform_colour(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_FRAGMENT || nir->info.internal ||
       nir->info.writes_memory || nir->info.fs.uses_discard ||
       nir->info.fs.uses_sample_shading || nir->info.fs.early_fragment_tests ||
       nir->info.outputs_written != BITFIELD64_BIT(FRAG_RESULT_DATA0) ||
       nir->info.num_ubos || nir->info.num_ssbos || nir->info.num_textures ||
       nir->info.num_images)
      return false;

   nir_divergence_analysis(nir);
   unsigned blocks = 0, exports = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      if (++blocks != 1)
         return false;
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_alu || instr->type == nir_instr_type_load_const)
            continue;
         if (instr->type != nir_instr_type_intrinsic)
            return false;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_export_agx:
            if (intr->src[0].ssa->divergent)
               return false;
            exports++;
            break;
         case nir_intrinsic_load_preamble:
            if (intr->def.bit_size != 64 || intr->def.num_components != 1 ||
                nir_intrinsic_base(intr) != AGX_ABI_FUNI_ROOT)
               return false;
            break;
         case nir_intrinsic_load_global_constant:
            if (!hk_m5_push_colour_load(intr))
               return false;
            break;
         default:
            return false;
         }
      }
   }
   return exports != 0;
}

/* Whether any set layout is a descriptor-buffer layout. */
static bool
hk_set_layouts_use_buffers(uint32_t set_layout_count,
                           struct vk_descriptor_set_layout *const *set_layouts)
{
   for (unsigned s = 0; set_layouts && s < set_layout_count; ++s) {
      if (set_layouts[s] &&
          (set_layouts[s]->flags &
           (VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT |
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_EMBEDDED_IMMUTABLE_SAMPLERS_BIT_EXT)))
         return true;
   }
   return false;
}

/* The descriptor sets whose buffer descriptors a draw's launch may copy
 * from when it is recorded (AGX3_UNIFORM_DMA_BUFFER): sets no binding of
 * which may be updated after it is bound or while a command buffer using it
 * is pending. Push descriptor sets are recorded too; descriptor buffers can
 * change until the commands execute. */
static uint8_t
hk_static_set_mask(uint32_t set_layout_count,
                   struct vk_descriptor_set_layout *const *set_layouts)
{
   const VkDescriptorBindingFlags late =
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
      VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
   uint8_t mask = 0;
   for (unsigned s = 0; s < MIN2(set_layout_count, 8); ++s) {
      if (!set_layouts || !set_layouts[s] ||
          (set_layouts[s]->flags &
           VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT))
         continue;
      if ((set_layouts[s]->flags &
           (VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT |
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_EMBEDDED_IMMUTABLE_SAMPLERS_BIT_EXT)) &&
          !(set_layouts[s]->flags &
            VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR))
         continue;

      const struct hk_descriptor_set_layout *layout =
         vk_to_hk_descriptor_set_layout(set_layouts[s]);
      bool fixed = true;
      for (unsigned b = 0; b < layout->binding_count; ++b)
         fixed &= !(layout->binding[b].flags & late);
      if (fixed)
         mask |= BITFIELD_BIT(s);
   }
   return mask;
}

/* Bindless (descriptor-heap) samples leave num_textures at zero, yet their
 * descriptor reads are exactly the per-draw work a constant program hoists. */
static bool
hk_nir_has_tex(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               return true;
         }
      }
   }
   return false;
}

static bool
hk_fold_tests_zs(nir_builder *b, nir_intrinsic_instr *intr,
                 UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_shader_part_tests_zs_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_imm_intN_t(b, UINT16_MAX, intr->def.bit_size));
   return true;
}

/* Whether a function writes external memory, by the predicate
 * nir_shader_gather_info uses, following any remaining direct calls. */
static bool
hk_nir_function_writes_external_memory(nir_function_impl *impl)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_intrinsic_writes_external_memory(nir_instr_as_intrinsic(instr)))
            return true;
         if (instr->type == nir_instr_type_call) {
            nir_function *callee = nir_instr_as_call(instr)->callee;
            if (callee->impl &&
                hk_nir_function_writes_external_memory(callee->impl))
               return true;
         }
      }
   }
   return false;
}

/* Transform feedback counts as a write before its buffer-store lowering. */
static bool
hk_shader_writes_external_memory(nir_shader *nir)
{
   return nir->info.has_transform_feedback_varyings ||
          hk_nir_function_writes_external_memory(nir_shader_get_entrypoint(nir));
}

/* Whether a shader accesses image texel data: a texture instruction other
 * than a size or sample-count query, or any image intrinsic other than those
 * queries. Bindless handles count, so the answer holds for any descriptor. */
static bool
hk_shader_reads_images(const nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               const nir_texop op = nir_instr_as_tex(instr)->op;
               if (op != nir_texop_txs && op != nir_texop_query_levels &&
                   op != nir_texop_texture_samples)
                  return true;
            } else if (instr->type == nir_instr_type_intrinsic) {
               const char *name = nir_intrinsic_infos[
                  nir_instr_as_intrinsic(instr)->intrinsic].name;
               if (strstr(name, "image") && !strstr(name, "_size") &&
                   !strstr(name, "_samples") && !strstr(name, "descriptor"))
                  return true;
            }
         }
      }
   }
   return false;
}

/* Whether a fragment shader may leave a covered pixel unwritten: it discards,
 * demotes, terminates or writes the sample mask. */
static bool
hk_fs_may_kill(const nir_shader *fs)
{
   nir_foreach_function_impl(impl, fs) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_demote:
            case nir_intrinsic_demote_if:
            case nir_intrinsic_terminate:
            case nir_intrinsic_terminate_if:
            case nir_intrinsic_demote_samples:
               return true;
            case nir_intrinsic_store_output:
               if (nir_intrinsic_io_semantics(intr).location ==
                   FRAG_RESULT_SAMPLE_MASK)
                  return true;
               break;
            default:
               break;
            }
         }
      }
   }
   return false;
}

/* The clip-space position a vertex shader stores for vertex k of a direct
 * draw that starts at vertex 0 and instance 0, when it is a constant once the
 * vertex index is: written once, at the top level of the shader, every
 * component by a constant. */
static bool
hk_vs_constant_position(const nir_shader *vs, unsigned k, float pos[4])
{
   nir_shader *s = nir_shader_clone(NULL, vs);
   /* The shader as the front end produced it: system values still read from
    * the draw, outputs possibly still variables (io_lowered is set before the
    * outputs are). Lowering what is already lowered changes nothing. */
   NIR_PASS(_, s, nir_lower_system_values);
   NIR_PASS(_, s, nir_lower_io, nir_var_shader_out, glsl_type_size, 0);
   nir_function_impl *impl = nir_shader_get_entrypoint(s);

   /* Substitute the draw's system values. */
   nir_foreach_function_impl(fimpl, s) {
      nir_builder b = nir_builder_create(fimpl);
      nir_foreach_block(block, fimpl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            uint64_t v;
            switch (intr->intrinsic) {
            case nir_intrinsic_load_vertex_id:
            case nir_intrinsic_load_vertex_id_zero_base:
               v = k;
               break;
            case nir_intrinsic_load_first_vertex:
            case nir_intrinsic_load_base_vertex:
            case nir_intrinsic_load_base_instance:
            case nir_intrinsic_load_instance_id:
            case nir_intrinsic_load_draw_id:
            /* Base vertex lowers to is_indexed_draw ? first_vertex : 0. */
            case nir_intrinsic_load_is_indexed_draw:
               v = 0;
               break;
            default:
               continue;
            }
            b.cursor = nir_before_instr(instr);
            nir_def_replace(&intr->def,
                            nir_imm_intN_t(&b, v, intr->def.bit_size));
         }
      }
   }

   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, s, nir_opt_copy_prop);
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_dead_cf);
      NIR_PASS(progress, s, nir_opt_remove_phis);
      NIR_PASS(progress, s, nir_opt_dce);
   } while (progress);

   unsigned written = 0;
   bool ok = true;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output ||
             nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_POS)
            continue;
         const unsigned mask = nir_intrinsic_write_mask(intr)
                               << nir_intrinsic_component(intr);
         if (block->cf_node.parent != &impl->cf_node ||
             !nir_src_is_const(intr->src[0]) || intr->src[0].ssa->bit_size != 32 ||
             !nir_src_is_const(intr->src[1]) || nir_src_as_uint(intr->src[1]) ||
             (written & mask)) {
            ok = false;
            continue;
         }
         u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
            pos[c + nir_intrinsic_component(intr)] =
               nir_src_comp_as_float(intr->src[0], c);
         }
         written |= mask;
      }
   }
   ralloc_free(s);
   return ok && written == 0xf;
}

/* A vertex shader whose first three vertices, by the vertex index alone, form
 * a triangle that contains the whole clip-space square in front of the viewer
 * and inside the depth range: the full-screen triangle of a post-processing
 * pass. One such triangle covers every sample of a render area its viewport
 * and scissor cover. Clip and cull distances could cut it, so a shader writing
 * them never qualifies. */
static bool
hk_vs_covers_clip_square(const nir_shader *vs)
{
   if (vs->info.stage != MESA_SHADER_VERTEX ||
       (vs->info.outputs_written &
        (VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1 |
         VARYING_BIT_CULL_DIST0 | VARYING_BIT_CULL_DIST1 |
         VARYING_BIT_CLIP_VERTEX)))
      return false;

   float x[3], y[3];
   for (unsigned k = 0; k < 3; ++k) {
      float pos[4];
      if (!hk_vs_constant_position(vs, k, pos) || !(pos[3] > 0.0f) ||
          !(pos[2] >= 0.0f && pos[2] <= pos[3]))
         return false;
      x[k] = pos[0] / pos[3];
      y[k] = pos[1] / pos[3];
   }

   /* Every corner of [-1, 1]^2 on the same side of each edge as the
    * opposite vertex (or on the edge). */
   for (unsigned e = 0; e < 3; ++e) {
      const unsigned a = e, b2 = (e + 1) % 3, o = (e + 2) % 3;
      const float ex = x[b2] - x[a], ey = y[b2] - y[a];
      const float side = ex * (y[o] - y[a]) - ey * (x[o] - x[a]);
      if (side == 0.0f)
         return false;
      for (unsigned c = 0; c < 4; ++c) {
         const float cx = (c & 1) ? 1.0f : -1.0f, cy = (c & 2) ? 1.0f : -1.0f;
         const float d = ex * (cy - y[a]) - ey * (cx - x[a]);
         if (d * side < 0.0f)
            return false;
      }
   }
   return true;
}

/* G18G compute: keep a lowered clone of the shader if its pass succeeded,
 * otherwise drop the clone and keep the shader as it was. */
static bool
hk_g18g_keep_lowered(nir_shader **nir, nir_shader *clone, bool lowered)
{
   if (!lowered) {
      ralloc_free(clone);
      return false;
   }

   ralloc_free(*nir);
   *nir = clone;
   return true;
}

/* G18G compute: whether an image access is left on the AGX2 bindless or
 * deref form. Such accesses compiled but silently did nothing on G18G when
 * tested, so they are refused at dispatch. bindless_image_agx only forms a
 * descriptor handle (the indexed sampler route keeps it); it is not an
 * access. */
static bool
hk_g18g_has_unlowered_image(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            const char *name =
               nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic]
                  .name;
            if ((!strncmp(name, "bindless_image_", 15) &&
                 strcmp(name, "bindless_image_agx")) ||
                !strncmp(name, "image_deref_", 12))
               return true;
         }
      }
   }
   return false;
}

static VkResult
hk_compile_nir(struct hk_device *dev, const VkAllocationCallbacks *pAllocator,
               nir_shader *nir, VkShaderCreateFlagsEXT shader_flags,
               const struct vk_pipeline_robustness_state *rs,
               const union hk_key *key, enum hk_feature_key features,
               struct hk_shader *shader, mesa_shader_stage sw_stage, bool hw,
               nir_xfb_info *xfb_info, unsigned set_count,
               uint8_t static_set_mask)
{
   const bool writes_external_memory = hk_shader_writes_external_memory(nir);
   const bool reads_images = hk_shader_reads_images(nir);
   /* Facts hk_g17p_note_first_draw() needs, before lowering obscures them. */
   if ((dev->dev.chip == AGX_CHIP_G17P || agx_chip_g17_m5(dev->dev.chip)) &&
       sw_stage == MESA_SHADER_FRAGMENT)
      shader->info.fs.may_kill = hk_fs_may_kill(nir);
   unsigned nr_vbos = 0;
   bool kill_psiz = false;
   bool default_layer = true;
   if ((sw_stage == MESA_SHADER_FRAGMENT && dev->gen1_native_fragment_rq) ||
       (sw_stage != MESA_SHADER_FRAGMENT && sw_stage != MESA_SHADER_COMPUTE &&
        dev->gen1_native_graphics_rq)) {
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic ==
                      nir_intrinsic_intersect_ray_gen1_agx)
                  shader->info.agx3_gen1_fragment_rq = true;
            }
         }
      }
   }

   /* The last VS/TES variant may pass xfb_info owned by this very NIR.
    * Resource lowering can replace and free that NIR below. Preserve its
    * transform-feedback layout before any such replacement or stage lowering,
    * and use the shader-owned copy for the remaining compile decisions. */
   if (xfb_info) {
      if (xfb_info->output_count > ARRAY_SIZE(shader->info.xfb_outputs)) {
         ralloc_free(nir);
         return vk_errorf(dev, VK_ERROR_UNKNOWN,
                          "%u transform feedback outputs exceed %zu",
                          xfb_info->output_count,
                          ARRAY_SIZE(shader->info.xfb_outputs));
      }
      memcpy(&shader->info.xfb_info, xfb_info,
             nir_xfb_info_size(xfb_info->output_count));
      typed_memcpy(shader->info.xfb_stride, nir->info.xfb_stride, 4);
      xfb_info = &shader->info.xfb_info;
   }

   /* Only G17P's backend keeps an array comparison's constant level or bias
    * in a register (agx_optimizer_inline_imm), so only G17P admits one to
    * the sampler lowerings below. */
   const bool g17p = dev->dev.chip == AGX_CHIP_G17P;
   if (agx_native_graphics(&dev->dev) &&
       nir->info.stage == MESA_SHADER_VERTEX) {
      /* Software vertex variants feed geometry/tessellation through CDM.
       * They use the same indexed texture heap and shifted vertex uniforms
       * as hardware VS variants; legacy descriptor-base/index operands do
       * not encode as native G17P texture fetches. Lower before changing the
       * NIR stage so both variants retain the vertex preamble convention. */
      if (agx_nir_lower_g17p_indexed_samplers(nir, g17p)) {
         shader->info.agx3_indexed_sampler_valid = true;

         /* Native G17P publishes texture/sampler heap pointers in halves
          * 0..7. The monolithic VBO lowering may already have materialized
          * vertex ABI preamble loads, so move those as one unit before root
          * and descriptor-set lowering allocates the following ranges.
          * A native query owns halves 0..167 and its heaps start at 168, so
          * only the vertex ABI above the query state moves: the query
          * lowering has already read its state from fixed halves, which the
          * driver publishes there (hk_gen1_upload_fragment_query_state()).
          * This covers an API vertex shader and a geometry shader's raster
          * replay alike. */
         NIR_PASS(_, nir, hk_nir_shift_vertex_preamble,
                  AGX_ABI_G17P_TEXTURE_POINTER_HALFS,
                  shader->info.agx3_gen1_fragment_rq);
      }
   } else if (agx_native_graphics(&dev->dev) &&
              (nir->info.stage == MESA_SHADER_GEOMETRY ||
               nir->info.stage == MESA_SHADER_TESS_CTRL ||
               nir->info.stage == MESA_SHADER_TESS_EVAL ||
               nir->info.stage == MESA_SHADER_MESH ||
               sw_stage == MESA_SHADER_TASK)) {
      /* Poly executes geometry variants through CDM, so they use the native
       * indexed texture/sampler heaps just like compute shaders; so does the
       * G17P software task kernel. Metal has no
       * geometry stage, but its indexed explicit-level vertex witness emits
       * the same register-operand SMP form selected after this rewrite. The
       * VDM-launched mesh program takes the heaps in uniform halves 0..7 like
       * the vertex program (hk_g17p_graphics_uniform_ranges()). */
      if (agx_nir_lower_g17p_indexed_samplers(nir, g17p))
         shader->info.agx3_indexed_sampler_valid = true;
   } else if ((agx_native_graphics(&dev->dev) ||
               dev->dev.chip == AGX_CHIP_G18G) &&
              nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* The combined-sampler lowering can refuse after changing the shader,
       * so it works on a copy -- made only for a shader it may apply to. */
      nir_shader *candidate =
         !dev->vk.enabled_features.descriptorBindingSampledImageUpdateAfterBind &&
               !(features & HK_FEAT_DESCRIPTOR_BUFFER) &&
               agx_nir_g17p_combined_sampler_candidate(nir, g17p)
            ? nir_shader_clone(NULL, nir)
            : NULL;
      struct agx_g17p_combined_sampler_binding binding;
      if (candidate &&
          agx_nir_lower_g17p_combined_sampler(candidate, &binding, g17p)) {
         ralloc_free(nir);
         nir = candidate;
         shader->info.agx3_combined_sampler_valid = true;
         shader->info.agx3_combined_sampler = binding;

         /* Texture and sampler table pointers occupy uniform halves 0..7.
          * Keep ordinary push-root accesses at AGX_ABI_FUNI_ROOT (half 12);
          * the lowering removes descriptor accesses, not push constants.
          */
         set_count = 0;
      } else {
         ralloc_free(candidate);
         if (agx_nir_lower_g17p_indexed_samplers(nir, g17p))
            shader->info.agx3_indexed_sampler_valid = true;
      }
   } else if (dev->dev.chip == AGX_CHIP_G18G &&
              nir->info.stage == MESA_SHADER_COMPUTE) {
      /* The storage routes go first: the state-0 route takes a single
       * storage image (write-only, or Gallium's read-write texture/PBE
       * pair) and declines shaders with texture instructions, which the
       * indexed sampler lowering would otherwise claim -- including the
       * image_load of a read-write image, which that lowering cannot encode
       * on G18G. Then several images in one set, then textures and images
       * through per-dispatch tables, then the indexed samplers. */
      struct agx_g17p_storage_image_binding storage;
      struct agx_g18g_table_binding tables;
      nir_shader *c = nir_shader_clone(NULL, nir);
      bool storage_ok = hk_g18g_keep_lowered(
         &nir, c, c && agx_nir_lower_g18g_storage_image(c, &storage));
      if (!storage_ok) {
         c = nir_shader_clone(NULL, nir);
         storage_ok = hk_g18g_keep_lowered(
            &nir, c, c && agx_nir_lower_g18g_storage_images(c, &storage));
      }

      if (storage_ok) {
         shader->info.agx3_storage_image_valid = true;
         shader->info.agx3_storage_image = storage;
      } else {
         c = nir_shader_clone(NULL, nir);
         if (hk_g18g_keep_lowered(
                &nir, c, c && agx_nir_lower_g18g_combined_tables(c, &tables))) {
            shader->info.agx3_storage_image_valid = true;
            shader->info.agx3_storage_image =
               (struct agx_g17p_storage_image_binding){0};
            shader->info.agx3_g18_tables_valid = true;
            shader->info.agx3_g18_tables = tables;
         } else {
            c = nir_shader_clone(NULL, nir);
            shader->info.agx3_indexed_sampler_valid = hk_g18g_keep_lowered(
               &nir, c, c && agx_nir_lower_g17p_indexed_samplers(c, g17p));
         }
      }
      shader->info.g18_unlowered_image = hk_g18g_has_unlowered_image(nir);
   } else if (agx_native_graphics(&dev->dev) &&
              nir->info.stage == MESA_SHADER_COMPUTE) {
      /* Descriptor buffers keep only the heap-indexed paths, which never
       * decode set contents while recording. */
      const bool record_time_sets = !(features & HK_FEAT_DESCRIPTOR_BUFFER);
      nir_shader *candidate = nir_shader_clone(NULL, nir);
      struct agx_g17p_sampled_storage_binding sampled_storage;
      if (candidate && record_time_sets &&
          !dev->vk.enabled_features.descriptorBindingSampledImageUpdateAfterBind &&
          agx_nir_lower_g17p_sampled_storage(candidate, &sampled_storage)) {
         ralloc_free(nir);
         nir = candidate;
         shader->info.agx3_sampled_storage_valid = true;
         shader->info.agx3_sampled_storage = sampled_storage;
      } else {
         ralloc_free(candidate);
         candidate = nir_shader_clone(NULL, nir);
         struct agx_g17p_storage_image_binding storage;
         if (candidate && record_time_sets &&
             agx_nir_lower_g17p_storage_image(candidate, &storage)) {
            ralloc_free(nir);
            nir = candidate;
            shader->info.agx3_storage_image_valid = true;
            shader->info.agx3_storage_image = storage;
         } else {
            ralloc_free(candidate);
            candidate = nir_shader_clone(NULL, nir);
            struct agx_g17p_combined_sampler_binding sampled_binding;
            if (candidate && record_time_sets &&
                !dev->vk.enabled_features.descriptorBindingSampledImageUpdateAfterBind &&
                agx_nir_lower_g17p_combined_sampler(candidate,
                                                    &sampled_binding, g17p)) {
               ralloc_free(nir);
               nir = candidate;
               shader->info.agx3_combined_sampler_valid = true;
               shader->info.agx3_combined_sampler = sampled_binding;
               shader->info.agx3_txf_texture_valid = true;
               shader->info.agx3_txf_texture.texture_offset_B =
                  sampled_binding.texture_offset_B;
            } else {
               ralloc_free(candidate);
               candidate = nir_shader_clone(NULL, nir);
               if (candidate &&
                   agx_nir_lower_g17p_indexed_samplers(candidate, g17p)) {
                  /* Keep the storage image BINDLESS alongside indexed
                   * samplers.
                   *
                   * agx_nir_lower_g17p_indexed_storage() rewrites the image
                   * store to the immediate-slot form: image_write with a zero
                   * base and a constant resource index, with the PBE
                   * descriptor delivered through the launch's argument slots
                   * by hk_agx3_g17p_indexed_storage_pointers()
                   * (hk_cmd_dispatch.c).  Measured on G17P with the
                   * image_2d_view_3d_image compute sampler cases: in that
                   * combination the store lands nowhere at all -- the
                   * destination reads back byte-identical to its pre-dispatch
                   * clear -- at resource index 1 and at index 2 alike.  The
                   * ordinary bindless image write, which the fragment stage
                   * and every non-sampling compute shader already use on this
                   * chip, stores correctly: 8 fail -> 0 fail, with 2823
                   * neighbouring cases per-case identical.
                   *
                   * WHY the immediate-slot store fails here is NOT understood.
                   * Only that it does, and that the bindless form does not.
                   * The slot layout is a plausible suspect -- slots 0/1 carry
                   * the texture and sampler heap bases that the shader
                   * dereferences itself -- but that is a hypothesis, not a
                   * measurement, and the driver does build a combined pointer
                   * set for exactly this case rather than ignoring it.
                   */
                  ralloc_free(nir);
                  nir = candidate;
                  shader->info.agx3_indexed_sampler_valid = true;
                  shader->info.agx3_indexed_storage_valid = false;
               } else {
                  ralloc_free(candidate);
                  candidate = nir_shader_clone(NULL, nir);
                  struct agx_g17p_txf_binding binding;
                  if (candidate && record_time_sets &&
                      agx_nir_lower_g17p_txf(candidate, &binding) &&
                      (!(binding.texture_offset_B &
                         (HK_G17P_TEXTURE_STATE_ALIGN - 1)) ||
                       !dev->vk.enabled_features
                           .descriptorBindingSampledImageUpdateAfterBind)) {
                     ralloc_free(nir);
                     nir = candidate;
                     shader->info.agx3_txf_texture_valid = true;
                     shader->info.agx3_txf_texture = binding;
                  } else {
                     ralloc_free(candidate);
                  }
               }
            }
         }
      }
   }

   if (agx_chip_g17_m5(dev->dev.chip) &&
       agx_nir_lower_g17g_indexed_reads(nir)) {
      shader->info.agx3_indexed_sampler_valid = true;
      if (nir->info.stage == MESA_SHADER_VERTEX) {
         /* Both native AGX3 shader ABIs reserve halves 0..7 for the texture
          * and sampler heaps. Move already-lowered VBO uniforms together.
          * A native query owns halves 0..167 and its heaps start at 168, so
          * only the vertex ABI above the query state moves. */
         NIR_PASS(_, nir, hk_nir_shift_vertex_preamble,
                  AGX_ABI_G17P_TEXTURE_POINTER_HALFS,
                  shader->info.agx3_gen1_fragment_rq);
      }
   }

   /* For now, only shader objects are supported */
   if (sw_stage == MESA_SHADER_VERTEX) {
      nr_vbos = DIV_ROUND_UP(
         BITSET_LAST_BIT(shader->info.vs.attrib_components_read), 4);
      kill_psiz = key->vs.kill_psiz;
      default_layer = !key->vs.no_default_layer;
   } else if (sw_stage == MESA_SHADER_FRAGMENT) {
      shader->info.fs.writes_memory = nir->info.writes_memory;
      shader->info.fs.interp.linear = nir->info.linear_varyings;
      shader->info.fs.interp.flat =
         ~(nir->info.linear_varyings | nir->info.perspective_varyings);

      /* Discards must be lowering before lowering MSAA to handle discards */
      NIR_PASS(_, nir, agx_nir_lower_discard_zs_emit);
      /* A shader that writes gl_SampleMask owns its coverage update. Apply
       * draw-time alpha-to-coverage before that update so killed samples also
       * skip depth/stencil tests, colour stores and occlusion accounting. */
      if (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK)) {
         nir_builder b = nir_builder_at(
            nir_before_impl(nir_shader_get_entrypoint(nir)));
         nir_def *enable =
            nir_ine_imm(&b, nir_load_alpha_to_coverage_agx(&b), 0);
         NIR_PASS(_, nir, nir_lower_alpha_to_coverage, false, enable);
      }

      if (agx_native_graphics(&dev->dev))
         shader->info.fs.g17p_colour_unit_range =
            agx_nir_fs_colour_outputs_unit_range(nir);

      /* Only a shader that already discards has an early coverage update to
       * carry the kill. A shader with no way to apply it keeps it in the
       * epilog. Early fragment tests and memory writes make the epilog
       * lowering fold every demote into the exported sample mask, so the
       * alpha-to-coverage demote would not survive to the coverage update:
       * those shaders keep alpha-to-coverage in the epilog too. */
      /* A fragment tests its coverage once. G17P unrolls a sample-shaded
       * main into one copy of its body per sample (below); lowered before
       * that, the testing coverage update after the last discard is cloned
       * with the body and the fragment tests once per sample. A pass of such
       * fragments can stall the GPU when another context's fragment work
       * runs beside it. So these shaders unroll their demotes -- each copy
       * keeps its own sample's -- and lower them afterwards: one update
       * tests whatever the copies left of the pixel, as in a per-pixel
       * shader that discards several times. Alpha-to-coverage stays in the
       * epilog for them, as for a shader that never discards.
       *
       * Forced-early fragments test once at the start already (the unroll
       * hoists that update ahead of the copies) and keep their order. */
      const bool g17p_test_once =
         dev->dev.chip == AGX_CHIP_G17P && nir->info.fs.uses_sample_shading &&
         !nir->info.fs.early_fragment_tests;
      /* M5 unrolls its sample-shaded mains the same way. A main too large
       * to unroll here (inlined with its epilog at draw time) keeps the
       * lowering before that unroll. */
      const bool m5_test_once =
         agx_chip_g17_m5(dev->dev.chip) && nir->info.fs.uses_sample_shading &&
         !nir->info.fs.early_fragment_tests;

      bool a2c_in_main = false;
      if (dev->dev.chip == AGX_CHIP_G17P && key && key->fs.static_a2c &&
          !g17p_test_once &&
          nir->info.fs.uses_discard && !nir->info.fs.early_fragment_tests &&
          !nir->info.writes_memory &&
          !(nir->info.outputs_written &
            BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK)))
         NIR_PASS(a2c_in_main, nir, nir_lower_alpha_to_coverage, false, NULL);

      NIR_PASS(_, nir, agx_nir_lower_fs_output_to_epilog_for_device,
               dev->dev.chip, &shader->info.fs.epilog_key);
      if (agx_chip_g17_m5(dev->dev.chip)) {
         if (!m5_test_once)
            NIR_PASS(_, nir, agx_nir_lower_sample_mask_deferred_m5);
      } else if (dev->dev.chip == AGX_CHIP_G17P) {
         if (!g17p_test_once) {
            NIR_PASS(_, nir, agx_nir_lower_sample_mask_deferred_g17p,
                     a2c_in_main ? &a2c_in_main : NULL);
         }
         shader->info.fs.a2c_in_main = a2c_in_main;
      } else if (agx_native_graphics(&dev->dev))
         NIR_PASS(_, nir, agx_nir_lower_sample_mask_deferred);
      else
         NIR_PASS(_, nir, agx_nir_lower_sample_mask);

      /* With alpha-to-coverage statically off or in the main part and no
       * depth/stencil export, no epilog of this shader discards, so on the
       * native AGX3 parts the main part always owns the depth/stencil tests
       * while no read-only early reject can apply
       * (hk_g17p_static_no_early_reject()): the draw-time
       * selector (hk_root_descriptor_table::draw.no_epilog_discard) is all
       * ones for every draw that can use it. Fold it, leaving the single
       * unconditional testing coverage update of the native form instead of
       * a uniform branch between a testing and a killing one. */
      const bool fold_tests =
         (agx_chip_native_agx3(dev->dev.chip) ||
          agx_chip_g17_m5(dev->dev.chip)) &&
         key && (key->fs.static_tests || a2c_in_main) &&
         !shader->info.fs.epilog_key.write_z &&
         !shader->info.fs.epilog_key.write_s;
      if (fold_tests && !g17p_test_once && !m5_test_once) {
         bool folded = false;
         NIR_PASS(folded, nir, nir_shader_intrinsics_pass,
                  hk_fold_tests_zs, nir_metadata_control_flow, NULL);
         shader->info.fs.static_tests = folded;
      }

      NIR_PASS(_, nir, agx_nir_apply_api_mask_to_force_early, dev->dev.chip);

      if (nir->info.fs.uses_sample_shading &&
          (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip))) {
         /* A split shader exports each sample through fixed registers. M5
          * fragments that exceed that file retain one sample's NIR until
          * the epilog can be inlined, then unroll the complete body. */
         const unsigned raster_samples = hk_native_fs_samples(dev->dev.chip, &key->fs);
         const unsigned registers = hk_native_fs_registers(dev->dev.chip);
         assert(raster_samples == 2 || raster_samples == 4 ||
                (agx_chip_g17_m5(dev->dev.chip) && raster_samples == 8));

         /* M5 unrolls up to eight samples in its larger register file, the
          * same count and file hk_native_sample_shading_supported() admitted. */
         if (!agx_nir_lower_g17p_sample_shading_file(
                nir, &shader->info.fs.epilog_key, raster_samples, registers,
                agx_chip_g17_m5(dev->dev.chip))) {
            if (agx_chip_g17_m5(dev->dev.chip) && raster_samples == 8) {
               shader->info.fs.m5_inline_sample_shading = true;
               /* Not unrolled here: lower as before. */
               if (m5_test_once) {
                  NIR_PASS(_, nir, agx_nir_lower_sample_mask_deferred_m5);
                  if (fold_tests) {
                     bool folded = false;
                     NIR_PASS(folded, nir, nir_shader_intrinsics_pass,
                              hk_fold_tests_zs, nir_metadata_control_flow,
                              NULL);
                     shader->info.fs.static_tests = folded;
                  }
               }
            } else {
               ralloc_free(nir);
               return vk_errorf(dev, VK_ERROR_UNKNOWN,
                                "G17P: %u-sample colour exports exceed the %u-half register file",
                                raster_samples, registers);
            }
         }

         if (g17p_test_once) {
            /* The copies' one-hot sample masks, so that the lowering sees
             * each demote's constant sample. */
            NIR_PASS(_, nir, nir_opt_constant_folding);
            /* Kills go into the one testing update where they can: a
             * kill-only update of part of a multisampled pixel loses the
             * killed samples' previous contents. Depth/stencil exports and
             * memory writes keep the kills where they happen. */
            const uint64_t zs = BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
                                BITFIELD64_BIT(FRAG_RESULT_STENCIL);
            if (!(nir->info.outputs_written & zs) && !nir->info.writes_memory)
               NIR_PASS(_, nir, agx_nir_lower_sample_mask_one_update);
            else
               NIR_PASS(_, nir, agx_nir_lower_sample_mask_deferred_g17p, NULL);
            if (fold_tests) {
               bool folded = false;
               NIR_PASS(folded, nir, nir_shader_intrinsics_pass,
                        hk_fold_tests_zs, nir_metadata_control_flow, NULL);
               shader->info.fs.static_tests = folded;
            }
         } else if (m5_test_once && !shader->info.fs.m5_inline_sample_shading) {
            NIR_PASS(_, nir, nir_opt_constant_folding);
            /* As on G17P: kills go into the one testing update where they
             * can. */
            const uint64_t zs = BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
                                BITFIELD64_BIT(FRAG_RESULT_STENCIL);
            if (!(nir->info.outputs_written & zs) && !nir->info.writes_memory)
               NIR_PASS(_, nir, agx_nir_lower_sample_mask_one_update_m5);
            else
               NIR_PASS(_, nir, agx_nir_lower_sample_mask_deferred_m5);
            if (fold_tests) {
               bool folded = false;
               NIR_PASS(folded, nir, nir_shader_intrinsics_pass,
                        hk_fold_tests_zs, nir_metadata_control_flow, NULL);
               shader->info.fs.static_tests = folded;
            }
         }
      } else if (nir->info.fs.uses_sample_shading) {
         /* Ensure the sample mask is preserved in register */
         nir_builder b =
            nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));

         nir_def *mask =
            nir_load_exported_agx(&b, 1, 16, .base = AGX_ABI_FIN_SAMPLE_MASK);

         nir_export_agx(&b, mask, .base = AGX_ABI_FOUT_SAMPLE_MASK);

         NIR_PASS(_, nir, agx_nir_lower_to_per_sample_agx2);
      }

      /* Explicit interpolateAtSample() is not sample shading. Fold its
       * single-sample clamp too, without cloning the per-pixel shader body
       * or guessing a samples-log2 special-register encoding: neither G17P
       * nor M5 has a grounded AGX3 encoding for that SR, and the packer
       * refuses it. M5 keys the count as m5_position_samples. */
      const unsigned native_samples =
         hk_native_fs_samples(dev->dev.chip, &key->fs);
      if ((agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
          native_samples)
         NIR_PASS(_, nir, agx_nir_specialize_sample_count, native_samples);

      if (!shader->info.fs.m5_inline_sample_shading)
         NIR_PASS(_, nir, agx_nir_lower_fs_active_samples_to_register);
      /* G17P exposes the raster layer through the flat coefficient table.
       * SR28 is the compute grid-Z register and does not hold a fragment
       * layer on this generation. */
      NIR_PASS(_, nir, agx_nir_lower_interpolation,
               agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip));
      if (agx_native_graphics(&dev->dev)) {
         NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                  hk_g17p_lower_is_first_fan,
                  nir_metadata_control_flow, NULL);
      }
   } else if (sw_stage == MESA_SHADER_TESS_EVAL ||
              sw_stage == MESA_SHADER_TESS_CTRL) {

      shader->info.tess.info.ccw = nir->info.tess.ccw;
      shader->info.tess.info.points = nir->info.tess.point_mode;
      shader->info.tess.info.spacing = nir->info.tess.spacing;
      shader->info.tess.info.mode = nir->info.tess._primitive_mode;

      if (sw_stage == MESA_SHADER_TESS_CTRL) {
         shader->info.tess.tcs_output_patch_size =
            nir->info.tess.tcs_vertices_out;
         shader->info.tess.tcs_per_vertex_outputs =
            poly_tcs_per_vertex_outputs(nir);
         shader->info.tess.tcs_nr_patch_outputs =
            util_last_bit(nir->info.patch_outputs_written);
         shader->info.tess.tcs_output_stride = poly_tcs_output_stride(nir);
      } else {
         /* This destroys info so it needs to happen after the gather */
         struct agx_gen1_ray_query_layout rq_layout =
            hk_gen1_ray_query_layout(dev, nir, 0);
         if (!hk_nir_lower_tes_variant(nir, hw,
                                       hk_native_rq_profile(dev) ? &rq_layout
                                                                 : NULL)) {
            ralloc_free(nir);
            return vk_errorf(dev, VK_ERROR_UNKNOWN,
                             "Gen1 native TES ray query lowering failed");
         }

         /* The query-state flag above was gathered before this lowering. */
         if (dev->gen1_native_graphics_rq)
            shader->info.agx3_gen1_fragment_rq = hk_nir_has_ray_query(nir);
      }
   }

   /* Normally, vertex shaders need to write a default point size. However, if
    * we have a geometry/tessellation shader, the hardware vertex (software
    * GS/TES) will handle this itself instead.
    */
   if ((sw_stage == MESA_SHADER_VERTEX || sw_stage == MESA_SHADER_TESS_EVAL) &&
       hw && !kill_psiz) {
      /* An evaluation shader is the hardware vertex shader too, and with
       * shader objects point mode may come from the control shader alone;
       * HLSL domain shaders write no point size
       * (shader_object.tessellation.hlsl.point_mode*). */
      NIR_PASS(_, nir, nir_lower_default_point_size);
   }

   uint64_t outputs = nir->info.outputs_written;
   if (sw_stage == MESA_SHADER_VERTEX || sw_stage == MESA_SHADER_TESS_EVAL) {
      if (hw) {
         if (agx_native_graphics(&dev->dev)) {
            NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                     hk_g17p_saturate_layer_viewport,
                     nir_metadata_control_flow, NULL);
         }
         if (sw_stage == MESA_SHADER_VERTEX &&
             (dev->dev.chip == AGX_CHIP_G17P ||
              agx_chip_g17_m5(dev->dev.chip)))
            NIR_PASS(_, nir, agx_nir_preserve_position_order);
         hk_lower_hw_vs(dev, nir, shader, kill_psiz,
                        agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip),
                        default_layer, hk_clip_m1_1(dev, features),
                        agx_native_graphics(&dev->dev)
                           ? hk_root_descriptor_offset(draw.no_attachments)
                           : -1);
         outputs = nir->info.outputs_written;
      } else {
         NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
         nir->info.stage = MESA_SHADER_COMPUTE;
         memset(&nir->info.cs, 0, sizeof(nir->info.cs));
         nir->xfb_info = NULL;
      }
   }

   unsigned root =
      dev->dev.chip == AGX_CHIP_G18G && shader->info.agx3_storage_image_valid
         ? AGX_G18G_STORAGE_ROOT_HALFS
      : (shader->info.agx3_storage_image_valid ||
                    shader->info.agx3_sampled_storage_valid ||
                    shader->info.agx3_indexed_storage_valid)
                      ? AGX_G17P_STORAGE_ROOT_HALFS
                   : (shader->info.agx3_txf_texture_valid ||
                      shader->info.agx3_indexed_sampler_valid)
                      ? AGX_G17P_TXF_ROOT_HALFS
                      : 0;
   const unsigned vertex_uniform_base =
      (sw_stage == MESA_SHADER_VERTEX &&
       shader->info.agx3_indexed_sampler_valid
         ? AGX_ABI_G17P_TEXTURE_POINTER_HALFS
         : 0) +
      (sw_stage == MESA_SHADER_VERTEX && shader->info.agx3_gen1_fragment_rq
         ? 168 : 0);
   if (sw_stage == MESA_SHADER_FRAGMENT)
      root = AGX_ABI_FUNI_ROOT;
   else if (sw_stage == MESA_SHADER_VERTEX)
      root = vertex_uniform_base + AGX_ABI_VUNI_COUNT_VK(nr_vbos);

   const unsigned texture_uniform_base =
      ((sw_stage == MESA_SHADER_COMPUTE && dev->gen1_native_rq &&
        hk_nir_has_ray_query(nir)) ||
       shader->info.agx3_gen1_fragment_rq) ? 168 : 0;
   if (sw_stage != MESA_SHADER_VERTEX || !shader->info.agx3_gen1_fragment_rq)
      root += texture_uniform_base;
   shader->info.set_count = set_count;
   shader->info.agx3_root_uniform = root;

   /* XXX: rename */
   NIR_PASS(_, nir, hk_lower_uvs_index, sw_stage, nr_vbos,
            vertex_uniform_base);
   /* Stage lowering can introduce driver sysvals (for example discard's
    * no_epilog_discard mask). hk_lower_uvs_index turns those into root
    * accesses, so gather the resource requirement after that pass and before
    * lower_uniforms removes the load_root_agx markers.
    *
    * G18G publishes its graphics and compute uniforms through the same
    * range builders (hk_g17p_graphics_uniform_ranges(), which G18G's
    * hk_upload_usc_words_internal() calls), so it needs the flag too:
    * without it a shader that reads the root but binds no descriptor set --
    * push constants -- got no root range and read its push constants from
    * address 0 + offset, i.e. as zero. */
   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip) ||
       dev->dev.chip == AGX_CHIP_G18G) {
      shader->info.agx3_uses_root = hk_nir_uses_root(nir);
      shader->info.agx3_root_reads = hk_nir_root_reads(nir);
   }

   /* Measure while root markers are still explicit. Uniform lowering loads
    * the root pointer and set bases separately, so retain that prefix too.
    * Round up for native uniform/global load granularity. */
   if (dev->dev.chip == AGX_CHIP_G17P)
      shader->root_descriptor_size = sizeof(struct hk_root_descriptor_table);
   if (dev->dev.chip == AGX_CHIP_G17P && sw_stage == MESA_SHADER_COMPUTE) {
      unsigned bytes = hk_nir_root_size(
         nir, sizeof(struct hk_root_descriptor_table));
      bytes = MAX2(bytes, hk_root_descriptor_offset(sets) +
                            set_count * sizeof(uint64_t));
      shader->root_descriptor_size = MIN2(
         ALIGN_POT(bytes, 16), sizeof(struct hk_root_descriptor_table));
   }

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_uniforms,
            nir_metadata_control_flow, &root);

   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               hk_g17p_lower_storage_records, nir_metadata_control_flow,
               NULL);
   }

#if 0
   /* TODO */
   nir_variable_mode robust2_modes = 0;
   if (rs->uniform_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT)
      robust2_modes |= nir_var_mem_ubo;
   if (rs->storage_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2_EXT)
      robust2_modes |= nir_var_mem_ssbo;
#endif

   /* Native M5 compute launches publish invariant inputs through TPS and
    * execute a separate constant program. Apply that same compile/launch
    * contract to every compute pipeline, including ordinary compute and
    * meta compute, while excluding graphics variants dispatched on CDM. */
   const bool m5_compute_cprog = agx_chip_g17_m5(dev->dev.chip) &&
      sw_stage == MESA_SHADER_COMPUTE && nir->info.stage == MESA_SHADER_COMPUTE;

   /* A fragment shader that reads descriptor-backed memory forms its native
    * constant program, as on G17P. The native HAL300 fragment program state
    * uses the same constant-calculation program setup of the state-load
    * encoder as compute and publishes the TI
    * spill/IPR quads in its PPP records (agx_ppp_hal300_graphics_core_pack).
    * An M5 fragment unrolled inline over 8 samples keeps no preamble: its
    * launch needs the monolithic program, whose preamble need not match.
    * Ray-query fragments keep their qualified launch unchanged, as their
    * vertex programs do. */
   const bool fs_reads_descriptors =
      nir->info.stage == MESA_SHADER_FRAGMENT &&
      (nir->info.num_ubos || nir->info.num_ssbos || nir->info.num_textures ||
       nir->info.num_images || hk_nir_has_tex(nir));
   const bool m5_fs_cprog = agx_chip_g17_m5(dev->dev.chip) &&
      sw_stage == MESA_SHADER_FRAGMENT && fs_reads_descriptors &&
      !shader->info.fs.m5_inline_sample_shading &&
      !shader->info.agx3_gen1_fragment_rq;

   const bool relocatable_uvs = hw && nir->info.stage == MESA_SHADER_VERTEX &&
                                shader->info.uvs.relocatable;
   struct agx_shader_key backend_key = {
      .reserved_preamble = root + (4 * (1 + set_count)),
      .g17p_texture_uniform_base = texture_uniform_base,
      .g17p_storage_image_pair = shader->info.agx3_storage_image_valid &&
                                 !shader->info.agx3_storage_image.write_only,
      .dev = agx_gather_device_key(&dev->dev),
      .no_stop = nir->info.stage == MESA_SHADER_FRAGMENT,
      .has_scratch = !nir->info.internal,
      .promote_constants = true,
      .promote_textures = true,
      /* hk_buffer_addr_format() and hk_ubo_addr_format() bound robust
       * buffers, and nir_lower_explicit_io then checks each access's 32-bit
       * offset. Null-descriptor bounds need no exact offset: their size is
       * zero, so every access fails the check. */
      .checked_offsets =
         rs && (rs->storage_buffers !=
                   VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT ||
                rs->uniform_buffers !=
                   VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT),
      .g17p_sparse_vertex_outputs =
         agx_native_graphics(&dev->dev) && hw &&
         hk_g17p_sparse_vertex_outputs(&nir->info, sw_stage,
                                       xfb_info && xfb_info->output_count),
      /* The placeholder indices of a relocatable UVS layout, which the
       * fast link rewrites for the bound fragment shader. */
      .g17p_uvs_reloc_base = relocatable_uvs
                                ? shader->info.uvs.group_offs[UVS_VARYINGS]
                                : 0,
      .g17p_uvs_reloc_size = relocatable_uvs ? shader->info.uvs.user_size : 0,
      /* The fragment constant program is formed only when the fragment reads
       * descriptor-backed memory (UBO/SSBO/texture/image) it can hoist. A
       * push-constant-only fragment keeps uniform DMA: its CProg measured
       * longer offline (light FS main 48 -> 52 + 12-instruction CProg, and
       * the loop's uniform compare/early compare were lost). Descriptor
       * lowering above uses immediate native state slots, never the AGX2
       * texture/sampler-state-store protocol. Only G17P's constant program
       * is qualified among the AGX3B chips; G15 never forms a preamble.
       * G18G never forms a preamble. */
      .agx3_native_cprog = m5_compute_cprog,
      .no_preamble = dev->dev.chip == AGX_CHIP_G18G ||
         (agx_chip_g17_m5(dev->dev.chip) && !m5_compute_cprog &&
          !m5_fs_cprog) ||
         (agx_native_graphics(&dev->dev) &&
          (!agx_chip_native_preamble(dev->dev.chip) || !fs_reads_descriptors)),
   };

   /* The native constant-program lowering applies exactly when G17P or an
    * M5 graphics stage lets this compile form a preamble (fragment and
    * vertex constant programs); carried in the key, never in the global
    * debug word. */
   backend_key.g17p_native_uniforms =
      (agx_native_graphics(&dev->dev) || m5_fs_cprog) &&
      !backend_key.no_preamble;

   /* Uniform DMA: the root table, every descriptor set pointer, the dynamic
    * buffers and the buffers of static sets are driver-resolvable
    * (hk_g17p_graphics_uniform_ranges publishes the spans). A hardware vertex
    * program copies into the uniforms its constant program leaves free; a
    * fragment program copies only when it forms none. */
   const bool vs_dma = sw_stage == MESA_SHADER_VERTEX && hw &&
                       nir->info.stage == MESA_SHADER_VERTEX;
   /* An ordinary compute pipeline forms no preamble, so without copies
    * every thread loads its push data, dispatch base and descriptors from
    * memory, one dependent load after another. On native AGX3 its launch
    * copies them like a graphics stage's (hk_g17p_compute_uniform_ranges()),
    * including meta compute pipelines; on M5 one with a native constant
    * program does (hk_m5_compute_uniform_dma_ranges()). Compute-stage
    * programs of graphics keep their own launch ABI; precompiled library
    * helpers do not use this shader path. */
   const bool cs_dma = sw_stage == MESA_SHADER_COMPUTE &&
                       nir->info.stage == MESA_SHADER_COMPUTE;
   const bool fs_dma = backend_key.no_preamble &&
                       sw_stage == MESA_SHADER_FRAGMENT &&
                       nir->info.stage == MESA_SHADER_FRAGMENT;
   if ((agx_native_graphics(&dev->dev) && (fs_dma || vs_dma || cs_dma)) ||
       (agx_chip_g17_m5(dev->dev.chip) &&
        (fs_dma || vs_dma || (cs_dma && m5_compute_cprog)))) {
      backend_key.g17p_uniform_dma = true;
      backend_key.g17p_dma_root = root;
      backend_key.g17p_dma_set_mask =
         (uint8_t)BITFIELD_MASK(MIN2(set_count, 8));
      backend_key.g17p_dma_static_set_mask =
         static_set_mask & backend_key.g17p_dma_set_mask;
      backend_key.g17p_dma_buffers_B =
         hk_root_descriptor_offset(dynamic_buffers);
      backend_key.g17p_dma_buffer_count = HK_MAX_DYNAMIC_BUFFERS;
   }

   /* For now, sample shading is always dynamic. Indicate that. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       nir->info.fs.uses_sample_shading)
      backend_key.fs.inside_sample_loop = true;

   if (nir->info.stage == MESA_SHADER_FRAGMENT && key)
      backend_key.fs.nr_samples = agx_chip_g17_m5(dev->dev.chip)
                                    ? key->fs.m5_position_samples
                                    : key->fs.g17p_samples;

   /* Serialize compiles only while they print, so debug output
    * stays readable. Behaviour flags (native_uniforms, nopreamble, ...)
    * change no shared state; locking on them serialized every pipeline
    * compile across the application's threads (4 threads: 268 -> 973 ms
    * for the 193 captured Minecraft pipelines). */
   simple_mtx_t *lock = NULL;
   if (agx_get_compiler_debug() &
       (AGX_DBG_SHADERS | AGX_DBG_SHADERDB | AGX_DBG_VERBOSE |
        AGX_DBG_INTERNAL))
      lock = &hk_device_physical(dev)->debug_compile_lock;

   if (lock)
      simple_mtx_lock(lock);

   assert(nir->info.io_lowered);

   /* Preserve NIR for optional native inline blending, M5 uniform-colour
    * blending and mandatory M5 sample shading whose split exports exceed the
    * register file. */
   if ((agx_chip_g17_m5(dev->dev.chip) &&
        sw_stage == MESA_SHADER_FRAGMENT && hk_m5_uniform_colour(nir)) ||
       (sw_stage == MESA_SHADER_FRAGMENT &&
        shader->info.fs.m5_inline_sample_shading) ||
       (agx_native_graphics(&dev->dev) &&
        (agx_get_fs_structure_flags() & AGX_FS_STRUCTURE_INLINE_BLEND) &&
        sw_stage == MESA_SHADER_FRAGMENT &&
        nir->info.stage == MESA_SHADER_FRAGMENT && !nir->info.internal &&
        !nir->info.fs.uses_sample_shading)) {
      struct blob blob;
      blob_init(&blob);
      nir_serialize(&blob, nir, false);
      if (!blob.out_of_memory) {
         shader->g17p_mono_nir = malloc(blob.size);
         if (shader->g17p_mono_nir) {
            memcpy(shader->g17p_mono_nir, blob.data, blob.size);
            shader->g17p_mono_nir_size = blob.size;
            shader->g17p_mono_key = backend_key;
         }
      }
      blob_finish(&blob);
      if (shader->info.fs.m5_inline_sample_shading) {
         if (!shader->g17p_mono_nir) {
            ralloc_free(nir);
            return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         }
         /* This compile supplies resource metadata only. Command recording
          * requires the complete inline program before any draw can run. */
         NIR_PASS(_, nir, agx_nir_lower_g17p_1x_sample_shading);
         backend_key.fs.inside_sample_loop = false;
      }
   }

   /* An M4 Pro API compute shader that shares nothing across its workgroup
    * may launch its big workgroups as smaller hardware workgroups
    * (agx_nir_lower_workgroup_split, hk dispatch_grid). Measured on T6040
    * only, so A18 Pro keeps the API shape. */
   uint16_t launch_size[3] = {0};
   if (agx_is_m4_pro(&dev->dev) && sw_stage == MESA_SHADER_COMPUTE &&
       nir->info.stage == MESA_SHADER_COMPUTE && !nir->info.internal &&
       agx_workgroup_split_size(nir, launch_size))
      NIR_PASS(_, nir, agx_nir_lower_workgroup_split, launch_size);

   /* G17P hardware vertex programs (API VS, GS raster and native TES
    * variants) run per-draw uniform work once in a native vertex constant
    * program, as Apple's compiler does: e.g. ProjMat * ModelViewMat. */
   /* HAL300 sets the same vertex constant program and packs its TI IPR/spill
    * quads into every VDM template exactly as HAL200 does. */
   const bool vs_cprog =
      (agx_chip_native_preamble(dev->dev.chip) ||
       agx_chip_g17_m5(dev->dev.chip)) &&
      hw && nir->info.stage == MESA_SHADER_VERTEX &&
      !shader->info.agx3_gen1_fragment_rq;
   bool compiled = vs_cprog &&
      hk_g17p_compile_vs_cprog(dev, &nir, &backend_key, shader);
   if (!compiled &&
       (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
       nir->info.stage == MESA_SHADER_FRAGMENT && !backend_key.no_preamble)
      compiled = hk_g17p_compile_fs_cprog(&nir, &backend_key, shader);
   /* An M5 compute constant program can need a form the AGX3 packer does not
    * encode, such as a 16-bit uniform write beside a used uniform half.
    * Compile such a kernel without its constant program, as M5 runs the
    * compute variants of graphics stages, rather than refusing the pipeline.
    * Kernels that compile with one are unchanged. */
   if (!compiled && m5_compute_cprog && !backend_key.no_preamble) {
      nir_shader *attempt = nir_shader_clone(NULL, nir);
      compiled = attempt &&
                 agx_compile_shader_nir(attempt, &backend_key, &shader->b);
      if (compiled) {
         ralloc_free(nir);
         nir = attempt;
      } else {
         mesa_logw("M5 compute constant program failed to compile (%s); "
                   "compiling without it", agx_last_compile_refusal());
         ralloc_free(attempt);
         memset(&shader->b, 0, sizeof(shader->b));
         backend_key.agx3_native_cprog = false;
         backend_key.no_preamble = true;
         backend_key.g17p_uniform_dma = false;
      }
   }
   if (!compiled)
      compiled = agx_compile_shader_nir(nir, &backend_key, &shader->b);

   if (lock)
      simple_mtx_unlock(lock);

   if (!compiled) {
      /* On G17P a refusal here is a missing AGX3 *instruction encoding*, and
       * it fails the whole shader closed.
       *
       * `AGX_MESA_DEBUG=shaders` makes the packer name the instruction it
       * refused, which is the only practical way to attribute one of these:
       *
       *    agx3_pack_binary.c:6740: no grounded AGX3 encoding for imad
       *      [d16:reg1 s0=16:reg1 s1=16:imm7 s2=16:imm3]
       *
       * The reproduction needs no hardware -- it runs under the G17P
       * drm-shim. */
      /* vk_errorf alone is invisible unless the application enabled debug
       * logging, so a refused shader reaches the application as a bare
       * VkResult and it typically crashes binding the pipeline that was never
       * created. Name the stage unconditionally: this is a driver gap, and
       * whoever hits it needs to know which shader hit it. */
      mesa_loge("AGX backend cannot encode %s shader for %s: %s",
                _mesa_shader_stage_to_string(sw_stage),
                agx_chip_name(dev->dev.chip),
                agx_last_compile_refusal()[0] ? agx_last_compile_refusal()
                                              : "no reason recorded");
      ralloc_free(nir);
      /* This path is reached from vkCreateGraphicsPipelines,
       * vkCreateComputePipelines and vkCreateShadersEXT.  None of the three
       * lists VK_ERROR_FEATURE_NOT_PRESENT among its return codes, so the old
       * value was outside the API contract for every caller.  The three legal
       * sets intersect in OUT_OF_HOST_MEMORY, OUT_OF_DEVICE_MEMORY, UNKNOWN
       * and VALIDATION_FAILED; UNKNOWN is the only one that describes a
       * backend that cannot encode an otherwise valid shader. */
      return vk_errorf(dev, VK_ERROR_UNKNOWN,
                       "AGX backend cannot encode shader for %s",
                       agx_chip_name(dev->dev.chip));
   }

   /* hk_lower_nir() already cleared info.ray_queries; the ray context of a
    * traversal shader is built for the API workgroup size, so keep it. The
    * rewritten builtins are exact for that size too. */
   if (launch_size[0] && !shader->b.info.uses_ray_traversal)
      memcpy(shader->b.info.launch_size, launch_size, sizeof(launch_size));

   shader->code_ptr = shader->b.binary;
   shader->code_size = shader->b.info.binary_size;

   if (unlikely(dev->frame_anatomy))
      hk_frame_anatomy_dump_nir(dev, shader, nir, sw_stage);

   if (sw_stage == MESA_SHADER_FRAGMENT)
      shader->info.fs.g17p_samples = MAX2(backend_key.fs.nr_samples, 1);

   shader->info.stage = sw_stage;
   shader->info.writes_external_memory = writes_external_memory;
   shader->info.reads_images = reads_images;
   shader->info.clip_distance_array_size = nir->info.clip_distance_array_size;
   shader->info.cull_distance_array_size = nir->info.cull_distance_array_size;
   shader->b.info.outputs = outputs;

   /* A geometry raster variant is a hardware vertex program too: software
    * geometry only changes how its input vertices were produced.  Build the
    * same byte-grounded VertexProgramVariant payload from the final UVS
    * layout, while leaving compute-only GS variants untouched. */
   if ((agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip) ||
        dev->dev.chip == AGX_CHIP_G18G) &&
       hw && shader->b.info.stage == MESA_SHADER_VERTEX) {
      struct hk_agx3_g17p_vertex_compile_info compiler_info;
      hk_g17p_vertex_compile_info_init(shader, vs_cprog, &compiler_info);
      /* On failure the program info is cleared, so the draw paths see no
       * vertex program variant and refuse the shader there. */
      if (dev->dev.chip == AGX_CHIP_G18G)
         hk_agx3_g17g_vertex_program_info(dev->dev.chip, &compiler_info,
                                          &shader->agx3_vdm_state_update);
      else
         hk_agx3_g17p_vertex_program_info(dev->dev.chip, &compiler_info,
                                          &shader->agx3_vdm_state_update);
   }

   if (nir->constant_data_size > 0) {
      uint32_t data_size = align(nir->constant_data_size, HK_MIN_UBO_ALIGNMENT);

      void *data = malloc(data_size);
      if (data == NULL) {
         ralloc_free(nir);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }

      memcpy(data, nir->constant_data, nir->constant_data_size);

      assert(nir->constant_data_size <= data_size);
      memset(data + nir->constant_data_size, 0,
             data_size - nir->constant_data_size);

      shader->data_ptr = data;
      shader->data_size = data_size;
   }

   ralloc_free(nir);

   VkResult result = hk_init_link_ht(shader, sw_stage);
   if (result != VK_SUCCESS)
      return vk_error(dev, result);

   result = hk_upload_shader(dev, shader);
   if (result != VK_SUCCESS)
      return vk_error(dev, result);
   shader->compiled = true;
   return VK_SUCCESS;
}

static const struct vk_shader_ops hk_shader_ops;

struct hk_shader *
hk_g17p_layout_shader_create(struct hk_device *dev, const struct hk_shader *api,
                             struct agx_shader_part *part)
{
   struct hk_shader *s = malloc(sizeof(*s));
   if (!s)
      return NULL;

   /* Everything but the program and its uniform layout is the API shader's:
    * the draw still builds link keys, varyings and constant data from it.
    * The copy shares the API shader's constant data, owns its code and
    * links nothing itself. */
   *s = *api;
   s->b = *part;
   part->binary = NULL;
   s->code_ptr = s->b.binary;
   s->code_size = s->b.info.binary_size;
   s->bo = NULL;
   s->bo_offset = 0;
   s->preamble_addr = 0;
   memset(&s->linked, 0, sizeof(s->linked));
   s->only_linked = NULL;
   s->uvs_linked = NULL;
   s->g17p_mono_nir = NULL;
   s->g17p_mono_nir_size = 0;

   if (s->b.info.has_preamble || s->b.info.rodata.size_16) {
      const size_t size = s->b.info.binary_size;
      s->bo = size ? hk_code_alloc(dev, size, &s->bo_offset) : NULL;
      if (!s->bo) {
         free(s->b.binary);
         free(s);
         return NULL;
      }
      memcpy((uint8_t *)agx_bo_map(s->bo) + s->bo_offset, s->b.binary, size);
      agx_bo_assert_range(s->bo, s->bo_offset, size);
      s->preamble_addr = hk_shader_code_va(s) + s->b.info.preamble_offset;
   }
   return s;
}

static void
hk_destroy_linked_shader(struct hk_device *dev, struct hk_linked_shader *linked)
{
   hk_g17p_linked_compute_cache_finish(linked);
   hk_g17p_linked_graphics_cache_finish(dev, linked);
   agx_bo_unreference(&dev->dev, linked->b.bo);
   if (linked->g17p_layout) {
      free(linked->g17p_layout->b.binary);
      agx_bo_unreference(&dev->dev, linked->g17p_layout->bo);
      free(linked->g17p_layout);
   }
   if (linked->variant_source) {
      agx_bo_unreference(&dev->dev, linked->variant_source->bo);
      free(linked->variant_source->b.binary);
   }
   ralloc_free(linked);
}

static void
hk_shader_destroy(struct hk_device *dev, struct hk_shader *s,
                  bool owns_payload)
{
   if (owns_payload) {
      free((void *)s->code_ptr);
      free((void *)s->data_ptr);
      agx_bo_unreference(&dev->dev, s->bo);
   }

   simple_mtx_destroy(&s->linked.lock);
   free(s->g17p_mono_nir);
   s->g17p_mono_nir = NULL;

   if (s->only_linked)
      hk_destroy_linked_shader(dev, s->only_linked);

   if (s->uvs_linked) {
      hash_table_foreach(s->uvs_linked, entry) {
         hk_destroy_linked_shader(dev, entry->data);
      }
      _mesa_hash_table_destroy(s->uvs_linked, NULL);
   }

   if (s->linked.ht) {
      hash_table_foreach(s->linked.ht, entry) {
         hk_destroy_linked_shader(dev, entry->data);
      }
      _mesa_hash_table_destroy(s->linked.ht, NULL);
   }
}

static void
hk_shader_binary_cache_entry_destroy(
   struct hk_device *dev, struct hk_shader_binary_cache_entry *entry)
{
   hk_api_shader_destroy(&dev->vk, &entry->shader->vk, NULL);
   free(entry->binary);
   free(entry);
}

static void
hk_shader_binary_cache_entry_unref(
   struct hk_device *dev, struct hk_shader_binary_cache_entry *entry)
{
   bool destroy;

   simple_mtx_lock(&dev->shader_binary_cache.lock);
   assert(entry->refcount > 0);
   destroy = --entry->refcount == 0;
   simple_mtx_unlock(&dev->shader_binary_cache.lock);

   if (destroy)
      hk_shader_binary_cache_entry_destroy(dev, entry);
}
static void hk_fs_sample_source_destroy(struct hk_fs_sample_source *src);

static void
hk_border_source_destroy(struct hk_device *dev, struct hk_border_source *src)
{
   if (!src)
      return;
   if (src->twin)
      hk_api_shader_destroy(&dev->vk, &src->twin->vk, NULL);
   simple_mtx_destroy(&src->lock);
   free(src->nir);
   free(src);
}

void
hk_api_shader_destroy(struct vk_device *vk_dev, struct vk_shader *vk_shader,
                      const VkAllocationCallbacks *pAllocator)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   const bool owns_payload = obj->binary_cache_entry == NULL;
   const uint32_t owned_mask =
      obj->sample_source ? obj->sample_source->owned_mask : 0;

   hk_foreach_variant(obj, shader) {
      const unsigned i = shader - obj->variants;
      hk_shader_destroy(dev, shader,
                        owns_payload || (owned_mask & BITFIELD_BIT(i)));
   }

   hk_fs_sample_source_destroy(obj->sample_source);
   hk_border_source_destroy(dev, obj->border_source);
   for (struct hk_mesh_fs_twin *twin = obj->mesh_twins, *next; twin;
        twin = next) {
      next = twin->next;
      hk_api_shader_destroy(&dev->vk, &twin->shader->vk, NULL);
      free(twin);
   }
   hk_border_source_destroy(dev, obj->mesh_source);
   hk_border_source_destroy(dev, obj->fbl_source);

   if (obj->binary_cache_entry)
      hk_shader_binary_cache_entry_unref(dev, obj->binary_cache_entry);

   vk_shader_free(&dev->vk, pAllocator, &obj->vk);
}

void
hk_shader_binary_cache_finish(struct hk_device *dev)
{
   for (unsigned i = 0; i < HK_SHADER_BINARY_CACHE_MAX; ++i) {
      struct hk_shader_binary_cache_entry *entry =
         dev->shader_binary_cache.entries[i];
      dev->shader_binary_cache.entries[i] = NULL;
      if (entry)
         hk_shader_binary_cache_entry_unref(dev, entry);
   }

   simple_mtx_destroy(&dev->shader_binary_cache.lock);
}

/*
 * A draw that runs geometry or tessellation allocates the 128 MB geometry heap,
 * and until now that allocation was reached only from inside vkCmdDraw().  It
 * costs 15.5 ms on G17P, so the first such draw in a process paid a 15.5 ms
 * stall mid-render-pass while every later draw with the same shaders cost
 * ~110 us.  The heap depends on nothing but the device, so pull it forward to
 * the creation of the stage that will need it -- a creation-time allocation is
 * where a client expects to pay for one.  hk_heap() keeps its own
 * util_call_once(), so this is a warm-up, not a new requirement.
 */
static VkResult
hk_shader_prepare_device_for_stage(struct hk_device *dev,
                                   mesa_shader_stage stage)
{
   if (stage == MESA_SHADER_GEOMETRY || stage == MESA_SHADER_TESS_EVAL)
      return hk_device_ensure_geometry_heap(dev);

   return VK_SUCCESS;
}

static VkResult
hk_fs_sample_source_init(struct hk_api_shader *obj, const nir_shader *nir,
                         const union hk_key *key,
                         const struct vk_shader_compile_info *info,
                         enum hk_feature_key features,
                         uint8_t static_set_mask)
{
   struct hk_fs_sample_source *src = calloc(1, sizeof(*src));
   if (!src)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   struct blob blob;
   blob_init(&blob);
   nir_serialize(&blob, nir, false);
   if (blob.out_of_memory) {
      blob_finish(&blob);
      free(src);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   blob_finish_get_buffer(&blob, &src->nir, &src->nir_size);

   simple_mtx_init(&src->lock, mtx_plain);
   src->key = *key;
   src->flags = info->flags;
   src->robustness = *info->robustness;
   src->features = features;
   src->set_layout_count = info->set_layout_count;
   src->static_set_mask = static_set_mask;
   obj->sample_source = src;
   return VK_SUCCESS;
}

static void
hk_fs_sample_source_destroy(struct hk_fs_sample_source *src)
{
   if (!src)
      return;
   simple_mtx_destroy(&src->lock);
   free(src->nir);
   free(src);
}

/* Compile the `samples` variant of obj from nir, which it consumes. */
static VkResult
hk_compile_fs_sample_variant(struct hk_device *dev,
                             const VkAllocationCallbacks *pAllocator,
                             nir_shader *nir, struct hk_api_shader *obj,
                             unsigned samples)
{
   struct hk_fs_sample_source *src = obj->sample_source;
   const unsigned i = util_logbase2(samples);
   union hk_key key = src->key;
   key.fs.g17p_samples = samples;
   if (agx_chip_g17_m5(dev->dev.chip))
      key.fs.m5_position_samples = samples;

   if (samples > 1 && nir->info.fs.uses_sample_shading &&
       !hk_native_sample_shading_supported(nir, &key.fs, dev->dev.chip)) {
      ralloc_free(nir);
      return vk_error(dev, VK_ERROR_UNKNOWN);
   }

   if (samples == 1)
      NIR_PASS(_, nir, agx_nir_lower_g17p_1x_sample_shading);
   if (agx_chip_g17_m5(dev->dev.chip) && samples > 4)
      NIR_PASS(_, nir, agx_nir_lower_sample_intrinsics_wide, false);
   else
      NIR_PASS(_, nir, agx_nir_lower_sample_intrinsics, false);

   VkResult result = hk_compile_nir(dev, pAllocator, nir, src->flags,
                                    &src->robustness, &key, src->features,
                                    &obj->variants[i], MESA_SHADER_FRAGMENT,
                                    true, NULL, src->set_layout_count,
                                    src->static_set_mask);
   if (result == VK_SUCCESS)
      p_atomic_set(&src->ready_mask, src->ready_mask | BITFIELD_BIT(i));
   return result;
}

static VkResult hk_deserialize_shader(struct hk_device *dev,
                                      struct blob_reader *blob,
                                      struct hk_shader *shader, bool absent,
                                      bool cache_candidate);
static void hk_shader_serialize(struct vk_device *vk_dev,
                                const struct hk_shader *shader,
                                struct blob *blob);

/* A multisample variant is built at its first draw, after its shader object
 * went into the pipeline cache, so the cached object never holds it and every
 * process compiled it again on the draw that needs it. Keep each built variant
 * in the disk cache under its own key: the saved source and compile inputs,
 * which name its NIR and key, and the sample count. */
static bool
hk_fs_sample_variant_cache_key(struct hk_device *dev,
                               const struct hk_fs_sample_source *src,
                               unsigned samples, cache_key out)
{
   struct disk_cache *cache = hk_device_physical(dev)->vk.disk_cache;
   if (!cache)
      return false;
   struct mesa_blake3 ctx;
   blake3_hash hash;
   _mesa_blake3_init(&ctx);
   static const char tag[] = "hk fs sample variant";
   _mesa_blake3_update(&ctx, tag, sizeof(tag));
   _mesa_blake3_update(&ctx, &samples, sizeof(samples));
   _mesa_blake3_update(&ctx, &src->key, sizeof(src->key));
   _mesa_blake3_update(&ctx, &src->flags, sizeof(src->flags));
   _mesa_blake3_update(&ctx, &src->robustness, sizeof(src->robustness));
   _mesa_blake3_update(&ctx, &src->features, sizeof(src->features));
   _mesa_blake3_update(&ctx, &src->set_layout_count,
                       sizeof(src->set_layout_count));
   _mesa_blake3_update(&ctx, &src->static_set_mask,
                       sizeof(src->static_set_mask));
   _mesa_blake3_update(&ctx, src->nir, src->nir_size);
   _mesa_blake3_final(&ctx, hash);
   disk_cache_compute_key(cache, hash, sizeof(hash), out);
   return true;
}

/* The variant and the NIR its inline blending is rebuilt from at fast-link,
 * as hk_api_shader_serialize() writes them for one variant. */
static bool
hk_fs_sample_variant_load(struct hk_device *dev, struct hk_shader *shader,
                          const cache_key key)
{
   size_t size;
   void *data = disk_cache_get(hk_device_physical(dev)->vk.disk_cache, key,
                               &size);
   if (!data)
      return false;
   struct blob_reader blob;
   blob_reader_init(&blob, data, size);
   bool ok = hk_deserialize_shader(dev, &blob, shader, false, false) ==
             VK_SUCCESS;
   const uint32_t mono_size = ok ? blob_read_uint32(&blob) : 0;
   if (ok && mono_size) {
      const void *mono_nir = blob_read_bytes(&blob, mono_size);
      const void *mono_key =
         blob_read_bytes(&blob, sizeof(shader->g17p_mono_key));
      shader->g17p_mono_nir = blob.overrun ? NULL : malloc(mono_size);
      ok = shader->g17p_mono_nir != NULL;
      if (ok) {
         memcpy(shader->g17p_mono_nir, mono_nir, mono_size);
         shader->g17p_mono_nir_size = mono_size;
         memcpy(&shader->g17p_mono_key, mono_key,
                sizeof(shader->g17p_mono_key));
      }
   }
   ok &= !blob.overrun && blob.current == blob.end;
   free(data);
   if (!ok) {
      /* A stale or damaged entry: compile instead, from a clean variant. */
      hk_shader_destroy(dev, shader, true);
      memset(shader, 0, sizeof(*shader));
   }
   return ok;
}

static void
hk_fs_sample_variant_store(struct hk_device *dev,
                           const struct hk_shader *shader,
                           const cache_key key)
{
   struct blob blob;
   blob_init(&blob);
   hk_shader_serialize(&dev->vk, shader, &blob);
   const uint32_t mono_size = shader->g17p_mono_nir ?
      shader->g17p_mono_nir_size : 0;
   blob_write_uint32(&blob, mono_size);
   if (mono_size) {
      blob_write_bytes(&blob, shader->g17p_mono_nir, mono_size);
      blob_write_bytes(&blob, &shader->g17p_mono_key,
                       sizeof(shader->g17p_mono_key));
   }
   if (!blob.out_of_memory)
      disk_cache_put(hk_device_physical(dev)->vk.disk_cache, key, blob.data,
                     blob.size, NULL);
   blob_finish(&blob);
}

VkResult
hk_fs_build_sample_variant(struct hk_device *dev, struct hk_api_shader *obj,
                           unsigned samples)
{
   if (!obj || !obj->sample_source || samples <= 1)
      return VK_SUCCESS;

   struct hk_fs_sample_source *src = obj->sample_source;
   const unsigned i = util_logbase2(samples);
   if (p_atomic_read(&src->ready_mask) & BITFIELD_BIT(i))
      return VK_SUCCESS;

   VkResult result = VK_SUCCESS;
   simple_mtx_lock(&src->lock);
   if (!(src->ready_mask & BITFIELD_BIT(i))) {
      cache_key key;
      const bool cacheable =
         hk_fs_sample_variant_cache_key(dev, src, samples, key);
      if (cacheable && hk_fs_sample_variant_load(dev, &obj->variants[i], key)) {
         src->owned_mask |= BITFIELD_BIT(i);
         p_atomic_set(&src->ready_mask, src->ready_mask | BITFIELD_BIT(i));
      } else {
         struct blob_reader reader;
         blob_reader_init(&reader, src->nir, src->nir_size);
         nir_shader *nir = nir_deserialize(NULL, &agx_nir_options, &reader);
         if (nir) {
            src->owned_mask |= BITFIELD_BIT(i);
            result = hk_compile_fs_sample_variant(dev, NULL, nir, obj, samples);
            if (result == VK_SUCCESS && cacheable)
               hk_fs_sample_variant_store(dev, &obj->variants[i], key);
         } else {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
         }
      }
   }
   simple_mtx_unlock(&src->lock);
   return result;
}

/* Launch `launch` of a mesh draw of `rows` rows of x workgroups: count rows
 * from row first, published as y rows by z slices (hk_mesh_launch_rows()). */
struct hk_mesh_launch {
   nir_def *first, *count, *y, *z;
};

static struct hk_mesh_launch
hk_mesh_launch_shape(nir_builder *b, nir_def *x, nir_def *rows, nir_def *launch)
{
   struct hk_mesh_launch l;
   nir_def *zero = nir_imm_int(b, 0), *one = nir_imm_int(b, 1);
   nir_def *per = nir_umax(b, nir_udiv(b, nir_imm_int(b, HK_MESH_LAUNCH_WGS),
                                       nir_umax(b, x, one)), one);
   nir_def *slices = nir_udiv_imm(b, nir_iadd_imm(b, per, HK_MESH_LAUNCH_MAX_Y - 1),
                                  HK_MESH_LAUNCH_MAX_Y);
   nir_def *slice_rows = nir_udiv(b, per, slices);
   nir_def *launch_rows = nir_imul(b, slice_rows, slices);
   l.first = nir_imul(b, launch, launch_rows);
   nir_def *left = nir_bcsel(b, nir_ult(b, l.first, rows), nir_isub(b, rows, l.first), zero);
   l.count = nir_bcsel(b, nir_ieq_imm(b, x, 0), zero, nir_umin(b, left, launch_rows));
   l.y = nir_umin(b, l.count, slice_rows);
   l.z = nir_bcsel(b, nir_ieq_imm(b, l.count, 0), zero,
                   nir_udiv(b, nir_iadd(b, l.count, nir_iadd_imm(b, l.y, -1)),
                            nir_umax(b, l.y, one)));
   return l;
}

/* A mesh workgroup's row within its launch (hk_mesh_launch_shape()). */
static nir_def *
hk_mesh_launch_row(nir_builder *b, nir_def *wg, const struct hk_mesh_launch *l)
{
   return nir_iadd(b, nir_channel(b, wg, 1), nir_imul(b, nir_channel(b, wg, 2), l->y));
}

/* Whether an x by y by z grid of mesh workgroups fits one launch as it is.
 * The object then publishes the grid itself at launch 0 (none at later
 * launches) and the mesh workgroup IDs need no decode; otherwise the draw
 * goes out in rows (hk_mesh_launch_shape()). */
static nir_def *
hk_mesh_launch_whole(nir_builder *b, nir_def *x, nir_def *y, nir_def *z)
{
   /* y and z are at most 65535 here, so only x * (y * z) can overflow. */
   nir_def *rows = nir_imul(b, y, z);
   nir_def *fits = nir_iand(b, nir_ieq_imm(b, nir_umul_high(b, x, rows), 0),
                            nir_ule_imm(b, nir_imul(b, x, rows), HK_MESH_LAUNCH_WGS));
   return nir_iand(b, fits, nir_ule_imm(b, y, HK_MESH_LAUNCH_MAX_Y));
}

/* Parameter block addressed by the predicated object's uniform0 pointer.
 * The ordinary mesh-only object keeps its smaller native passthrough ABI. */
static nir_shader *
hk_build_predicated_mesh_object(enum agx_chip chip)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "native predicated mesh object");
   b.shader->info.io_lowered = true;
   b.shader->info.internal = true;
   b.shader->info.workgroup_size[0] = 3;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *params = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *grid = nir_load_global_constant(&b, 1, 64, params, .align_mul = 8);
   nir_def *count = nir_load_global_constant(&b, 1, 64,
      nir_iadd_imm(&b, params, 8), .align_mul = 8);
   nir_def *predicate = nir_load_global_constant(&b, 1, 64,
      nir_iadd_imm(&b, params, 16), .align_mul = 8);
   nir_variable *enabled = nir_local_variable_create(nir_shader_get_entrypoint(b.shader),
                                                    glsl_bool_type(), "enabled");
   nir_store_var(&b, enabled, nir_imm_true(&b), 1);
   nir_push_if(&b, nir_ine_imm(&b, count, 0));
   {
      nir_def *limit = nir_load_global_constant(&b, 1, 32, count, .align_mul = 4);
      nir_def *draw_id = nir_load_global_constant(&b, 1, 32,
         nir_iadd_imm(&b, params, 24), .align_mul = 4);
      nir_store_var(&b, enabled, nir_ult(&b, draw_id, limit), 1);
   }
   nir_pop_if(&b, NULL);
   nir_push_if(&b, nir_ine_imm(&b, predicate, 0));
   {
      nir_def *value = nir_load_global_constant(&b, 1, 32, predicate, .align_mul = 4);
      nir_def *invert = nir_load_global_constant(&b, 1, 32,
         nir_iadd_imm(&b, params, 28), .align_mul = 4);
      nir_def *pass = nir_ixor(&b, nir_ine_imm(&b, value, 0), nir_ine_imm(&b, invert, 0));
      nir_store_var(&b, enabled, nir_iand(&b, pass, nir_load_var(&b, enabled)), 1);
   }
   nir_pop_if(&b, NULL);
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *value;
   if (chip == AGX_CHIP_G17P) {
      /* This launch's rows of the grid (hk_mesh_launch_shape()). */
      nir_def *g = nir_load_global_constant(&b, 3, 32, grid, .align_mul = 4);
      nir_def *launch = nir_load_global_constant(&b, 1, 32,
         nir_iadd_imm(&b, params, 32), .align_mul = 4);
      nir_def *x = nir_channel(&b, g, 0);
      nir_def *gy = nir_channel(&b, g, 1), *gz = nir_channel(&b, g, 2);
      struct hk_mesh_launch l = hk_mesh_launch_shape(&b, x, nir_imul(&b, gy, gz), launch);
      nir_def *whole = hk_mesh_launch_whole(&b, x, gy, gz);
      nir_def *any = nir_bcsel(&b, whole,
         nir_iand(&b, nir_ieq_imm(&b, launch, 0),
                  nir_ine_imm(&b, nir_imul(&b, x, nir_imul(&b, gy, gz)), 0)),
         nir_ine_imm(&b, l.count, 0));
      nir_store_var(&b, enabled, nir_iand(&b, nir_load_var(&b, enabled), any), 1);
      value = nir_bcsel(&b, nir_ieq_imm(&b, lane, 0), x,
                        nir_bcsel(&b, nir_ieq_imm(&b, lane, 1),
                                  nir_bcsel(&b, whole, gy, l.y),
                                  nir_bcsel(&b, whole, gz, l.z)));
   } else {
      value = nir_load_global_constant(&b, 1, 32,
         nir_iadd(&b, grid, nir_u2u64(&b, nir_imul_imm(&b, lane, 4))), .align_mul = 4);
   }
   nir_store_object_childcount_agx(&b, nir_bcsel(&b, nir_load_var(&b, enabled),
                                               value, nir_imm_int(&b, 0)));
   if (agx_chip_g17_m5(chip))
      agx_preprocess_nir_m5(b.shader);
   else
      agx_preprocess_nir_native(b.shader, chip);
   return b.shader;
}

/* The mesh object of a software task draw: launch the children of every task
 * entry (hk_g17p_mesh_task_record() decodes the grid). uniform0 points at the
 * task record. */
static nir_shader *hk_build_task_mesh_object(enum agx_chip chip);

/* twin_of: compile the custom-border-emulating twin of the shader that
 * saved these inputs; info then names its NIR and saved create inputs. */





static nir_def *
hk_mesh_query_root(nir_builder *b, unsigned bit_size, unsigned offset)
{
   nir_def *addr = nir_iadd_imm(b, nir_load_root_agx(b), offset);
   return nir_load_global_constant(b, 1, bit_size, addr,
                                   .align_mul = bit_size / 8,
                                   .access = ACCESS_CAN_SPECULATE);
}

/* invocations: count mesh invocations here. A mesh shader launched by the
 * software task stage leaves that to the task kernel, which knows how many of
 * its padded workgroups are real. */
static bool
hk_lower_mesh_queries(nir_shader *nir, bool invocations)
{
   if (nir->info.stage != MESA_SHADER_MESH)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   const unsigned threads = nir->info.workgroup_size[0] *
                            nir->info.workgroup_size[1] *
                            nir->info.workgroup_size[2];
   if (invocations) {
      nir_def *flags = hk_mesh_query_root(
         &b, 32, hk_root_descriptor_offset(draw.pipeline_stats_flags));
      nir_def *first = nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0);
      nir_push_if(&b, nir_iand(&b, first, nir_ine_imm(&b, nir_iand_imm(&b, flags,
         VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT), 0)));
      {
         nir_def *addr = nir_load_stat_query_address_poly(
            &b, .base = PIPE_STAT_QUERY_MS_INVOCATIONS);
         nir_global_atomic(&b, 32, addr, nir_imm_int(&b, threads),
                           .atomic_op = nir_atomic_op_iadd);
      }
      nir_pop_if(&b, NULL);
   }

   /* Collect first: the query code below splits blocks. */
   struct util_dynarray sites;
   util_dynarray_init(&sites, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(instr)->intrinsic !=
                nir_intrinsic_set_vertex_and_primitive_count)
            continue;
         nir_intrinsic_instr *counts = nir_instr_as_intrinsic(instr);
         if (!nir_src_is_const(counts->src[1]) ||
             nir_src_as_uint(counts->src[1]) != 0)
            util_dynarray_append(&sites, counts);
      }
   }
   util_dynarray_foreach(&sites, nir_intrinsic_instr *, site) {
      {
         nir_intrinsic_instr *counts = *site;
         b.cursor = nir_after_instr(&counts->instr);
         nir_def *query = hk_mesh_query_root(
            &b, 64, hk_root_descriptor_offset(mesh_prims_query));
         nir_def *lane0 = nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0);
         nir_push_if(&b, nir_iand(&b, lane0, nir_ine_imm(&b, query, 0)));
         nir_global_atomic(&b, 32, query, nir_u2u32(&b, counts->src[1].ssa),
                           .atomic_op = nir_atomic_op_iadd);
         nir_pop_if(&b, NULL);
      }
   }
   util_dynarray_fini(&sites);

   return nir_progress(true, impl, nir_metadata_none);
}

/* A mesh shader used with a task stage reads its child grid
 * (gl_NumWorkGroups) from the native object->mesh record, not the API grid in
 * the draw root. Run before hk_lower_nir, which would lower it to the root.
 * Without the record (no kernel mesh rendering buffer) there is no source for
 * it: returns false. */
/* M5 task draws whose grid, draw count or predicate live in GPU memory
 * (hk_draw_mesh_native) record one object workgroup and let
 * libagx_m5_task_dispatch publish whether the draw executes. A task shader
 * checks that word on entry: a disabled draw launches no mesh children. The
 * root word is zero for every other draw, so only those loads the flag. */
static void
hk_m5_task_predicate(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *flag = nir_load_global_constant(&b, 1, 64,
      nir_iadd_imm(&b, nir_load_root_agx(&b),
                   hk_root_descriptor_offset(mesh_task_flag)),
      .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
   nir_push_if(&b, nir_ine_imm(&b, flag, 0));
   {
      nir_def *execute = nir_load_global(&b, 1, 32, flag, .align_mul = 4);
      nir_push_if(&b, nir_ieq_imm(&b, execute, 0));
      {
         nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 0, 0, 0));
         nir_jump(&b, nir_jump_halt);
      }
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);
   nir_progress(true, impl, nir_metadata_none);
}

static bool
hk_m5_mesh_task_grid(struct hk_device *dev, nir_shader *nir,
                     VkShaderCreateFlagsEXT flags)
{
   if (nir->info.stage != MESA_SHADER_MESH ||
       (flags & VK_SHADER_CREATE_NO_TASK_SHADER_BIT_EXT) ||
       !hk_nir_reads_intrinsic(nir, nir_intrinsic_load_num_workgroups))
      return true;
   if (!hk_m5_task_record(dev->dev.params.features))
      return false;
   NIR_PASS(_, nir, agx_nir_lower_m5_mesh_task_record);
   return true;
}

/* Software task stage (G17P). The task kernel leaves one entry per task
 * workgroup (libagx_task_record). The mesh objects launch every task's
 * children in rows: x indexes the child within a row (32768 per row) and y
 * the row, task after task, split into launches (hk_task_rows()). Tasks with
 * fewer children than the largest leave empty workgroups, which output
 * nothing.
 */
#define HK_TASK_CHILD_ROW 32768

static nir_def *
hk_task_record_addr(nir_builder *b)
{
   return nir_load_global_constant(b, 1, 64,
      nir_iadd_imm(b, nir_load_root_agx(b),
                   hk_root_descriptor_offset(mesh_task_flag)),
      .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
}

/* The software task draw's mesh launch: each task gets rows of x children
 * (x = the largest child count, at most HK_TASK_CHILD_ROW). */
struct hk_task_rows {
   nir_def *x, *task_rows, *tasks;
   struct hk_mesh_launch l;
};

static struct hk_task_rows
hk_task_rows(nir_builder *b, nir_def *max_children, nir_def *tasks,
             nir_def *launch)
{
   struct hk_task_rows r = {.tasks = tasks};
   r.x = nir_umin(b, max_children, nir_imm_int(b, HK_TASK_CHILD_ROW));
   r.task_rows = nir_bcsel(b, nir_ugt_imm(b, max_children, HK_TASK_CHILD_ROW),
                           nir_ushr_imm(b, nir_iadd_imm(b, max_children,
                                                        HK_TASK_CHILD_ROW - 1), 15),
                           nir_imm_int(b, 1));
   r.l = hk_mesh_launch_shape(b, r.x, nir_imul(b, r.task_rows, tasks), launch);
   return r;
}

/* Whether a software task draw's children fit one launch as one row per
 * task: the object publishes (children, tasks, 1) and a mesh workgroup's task
 * and child are its workgroup ID's y and x (hk_mesh_launch_whole()). */
static nir_def *
hk_task_whole(nir_builder *b, nir_def *max_children, nir_def *tasks)
{
   return nir_iand(b, nir_ule_imm(b, max_children, HK_TASK_CHILD_ROW),
                   hk_mesh_launch_whole(b, max_children, tasks, nir_imm_int(b, 1)));
}

static nir_def *
hk_mesh_launch_index(nir_builder *b)
{
   return nir_load_global_constant(b, 1, 32,
      nir_iadd_imm(b, nir_load_root_agx(b),
                   hk_root_descriptor_offset(mesh_launch)),
      .align_mul = 4, .access = ACCESS_CAN_SPECULATE);
}

static bool
hk_mesh_task_rewrite(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_def **decoded = data;
   nir_def *id = decoded[0], *grid = decoded[1], *payload = decoded[2];
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_workgroup_id:
      if (&intr->def == decoded[3])
         return false;
      nir_def_replace(&intr->def, nir_u2uN(b, id, intr->def.bit_size));
      return true;
   case nir_intrinsic_load_num_workgroups:
      if (!grid || &intr->def == decoded[4])
         return false;
      nir_def_replace(&intr->def, nir_u2uN(b, grid, intr->def.bit_size));
      return true;
   case nir_intrinsic_load_global_invocation_id: {
      const uint16_t *size = b->shader->info.workgroup_size;
      nir_def *global = nir_iadd(b,
         nir_imul(b, id, nir_imm_ivec3(b, size[0], size[1], size[2])),
         nir_load_local_invocation_id(b));
      nir_def_replace(&intr->def, nir_u2uN(b, global, intr->def.bit_size));
      return true;
   }
   case nir_intrinsic_load_task_payload: {
      if (!payload)
         return false;
      nir_def *addr = nir_iadd(b, payload, nir_u2u64(b,
         nir_iadd_imm(b, intr->src[0].ssa, nir_intrinsic_base(intr))));
      nir_def_replace(&intr->def, nir_load_global(b,
         intr->def.num_components, intr->def.bit_size, addr,
         .align_mul = nir_intrinsic_align_mul(intr),
         .align_offset = nir_intrinsic_align_offset(intr),
         .access = ACCESS_NON_WRITEABLE));
      return true;
   }
   default:
      return false;
   }
}

/* A mesh shader that never calls SetMeshOutputsEXT outputs nothing. */
static bool
hk_g17p_mesh_default_counts(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_set_vertex_and_primitive_count)
            return false;
      }
   }
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                                      nir_undef(&b, 1, 32));
   return nir_progress(true, impl, nir_metadata_control_flow);
}

/* A G17P mesh shader used with a task stage: find the launching task's entry
 * from the hardware workgroup ID, read the child grid and payload from it, and
 * run the body only for a real child. Runs before hk_lower_nir, which would
 * turn gl_NumWorkGroups into the API grid of the draw. */
static void
hk_g17p_mesh_task_record(nir_shader *nir)
{
   NIR_PASS(_, nir, hk_g17p_mesh_default_counts);
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_task_payload,
            shared_var_info);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_task_payload,
            nir_address_format_32bit_offset);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   nir_def *rec = hk_task_record_addr(&b);
   nir_def *hdr = nir_load_global(&b, 3, 32, rec, .align_mul = 16,
                                  .access = ACCESS_NON_WRITEABLE);
   nir_def *entries = nir_load_global(&b, 1, 64, nir_iadd_imm(&b, rec,
      offsetof(struct libagx_task_record, entries)), .align_mul = 8,
      .access = ACCESS_NON_WRITEABLE);
   nir_def *max_children = nir_channel(&b, hdr, 0);
   nir_def *tasks = nir_channel(&b, hdr, 1);
   nir_def *stride = nir_channel(&b, hdr, 2);

   nir_def *wg = nir_load_workgroup_id(&b);
   /* One row per task in one launch: the task and child are the workgroup
    * ID's y and x (hk_task_whole()); otherwise decode the launch's rows. */
   nir_def *true_ = nir_imm_true(&b);
   nir_def *wg_x = nir_channel(&b, wg, 0), *wg_y = nir_channel(&b, wg, 1);
   nir_push_if(&b, hk_task_whole(&b, max_children, tasks))->control =
      nir_selection_control_dont_flatten;
   nir_push_else(&b, NULL);
   struct hk_task_rows r = hk_task_rows(&b, max_children, tasks,
                                        hk_mesh_launch_index(&b));
   nir_def *launch_row = hk_mesh_launch_row(&b, wg, &r.l);
   nir_def *global_row = nir_iadd(&b, r.l.first, launch_row);
   nir_def *task_rows = nir_udiv(&b, global_row, r.task_rows);
   nir_def *row = nir_isub(&b, global_row, nir_imul(&b, task_rows, r.task_rows));
   nir_def *child_rows = nir_iadd(&b, nir_channel(&b, wg, 0), nir_imul(&b, row, r.x));
   nir_def *row_ok = nir_ult(&b, launch_row, r.l.count);
   nir_pop_if(&b, NULL);
   nir_def *task = nir_if_phi(&b, wg_y, task_rows);
   nir_def *child = nir_if_phi(&b, wg_x, child_rows);
   nir_def *in_launch = nir_if_phi(&b, true_, row_ok);
   nir_def *task_ok = nir_ult(&b, task, tasks);
   nir_def *entry = nir_iadd(&b, entries,
      nir_umul_2x32_64(&b, nir_bcsel(&b, task_ok, task, nir_imm_int(&b, 0)), stride));
   nir_def *grid = nir_load_global(&b, 3, 32, entry, .align_mul = 16,
                                   .access = ACCESS_NON_WRITEABLE);
   grid = nir_bcsel(&b, task_ok, grid, nir_imm_ivec3(&b, 0, 0, 0));
   nir_def *gx = nir_channel(&b, grid, 0), *gy = nir_channel(&b, grid, 1);
   nir_def *count = nir_imul(&b, nir_imul(&b, gx, gy), nir_channel(&b, grid, 2));
   nir_def *valid = nir_iand(&b, nir_ult(&b, child, count), in_launch);
   /* A one-row child grid (n, 1, 1) needs no division. */
   nir_def *id_row = nir_vec3(&b, child, nir_imm_int(&b, 0), nir_imm_int(&b, 0));
   nir_push_if(&b, nir_iand(&b, nir_ule_imm(&b, gy, 1),
                            nir_ule_imm(&b, nir_channel(&b, grid, 2), 1)))->control =
      nir_selection_control_dont_flatten;
   nir_push_else(&b, NULL);
   gx = nir_umax(&b, gx, nir_imm_int(&b, 1));
   gy = nir_umax(&b, gy, nir_imm_int(&b, 1));
   nir_def *rest = nir_udiv(&b, child, gx);
   nir_def *id_grid = nir_vec3(&b, nir_isub(&b, child, nir_imul(&b, rest, gx)),
                               nir_umod(&b, rest, gy), nir_udiv(&b, rest, gy));
   nir_pop_if(&b, NULL);
   nir_def *id = nir_if_phi(&b, id_row, id_grid);
   nir_def *payload = nir_iadd_imm(&b, entry, LIBAGX_TASK_ENTRY_PAYLOAD_B);

   nir_push_if(&b, valid);
   b.cursor = nir_cf_reinsert(&body, b.cursor);
   nir_push_else(&b, NULL);
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                                      nir_undef(&b, 1, 32));
   nir_pop_if(&b, NULL);
   nir_progress(true, impl, nir_metadata_none);

   nir_def *decoded[5] = {id, grid, payload, wg, NULL};
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_mesh_task_rewrite,
            nir_metadata_control_flow, decoded);
}

struct hk_task_kernel {
   nir_def *rec, *entry, *mesh_threads;
};

static bool
hk_task_kernel_rewrite(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct hk_task_kernel *k = data;
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_barrier: {
      const nir_variable_mode modes = nir_intrinsic_memory_modes(intr);
      if (!(modes & nir_var_mem_task_payload))
         return false;
      nir_intrinsic_set_memory_modes(intr,
         (modes & ~nir_var_mem_task_payload) | nir_var_mem_global);
      return true;
   }
   case nir_intrinsic_load_task_payload:
   case nir_intrinsic_store_task_payload: {
      const bool store = intr->intrinsic == nir_intrinsic_store_task_payload;
      nir_def *offset = intr->src[store ? 1 : 0].ssa;
      nir_def *addr = nir_iadd(b, k->entry, nir_u2u64(b,
         nir_iadd_imm(b, offset, LIBAGX_TASK_ENTRY_PAYLOAD_B +
                                 nir_intrinsic_base(intr))));
      if (store) {
         nir_store_global(b, intr->src[0].ssa, addr,
                          .write_mask = nir_intrinsic_write_mask(intr),
                          .align_mul = nir_intrinsic_align_mul(intr),
                          .align_offset = nir_intrinsic_align_offset(intr));
         nir_instr_remove(&intr->instr);
      } else {
         nir_def_replace(&intr->def, nir_load_global(b,
            intr->def.num_components, intr->def.bit_size, addr,
            .align_mul = nir_intrinsic_align_mul(intr),
            .align_offset = nir_intrinsic_align_offset(intr)));
      }
      return true;
   }
   case nir_intrinsic_launch_mesh_workgroups: {
      /* The arguments of the first invocation launch the children. */
      nir_def *grid = intr->src[0].ssa;
      nir_push_if(b, nir_ieq_imm(b, nir_load_local_invocation_index(b), 0));
      {
         nir_store_global(b, nir_pad_vector_imm_int(b, grid, 0, 4), k->entry,
                          .write_mask = 0xf, .align_mul = 16);
         nir_def *count = nir_imul(b, nir_imul(b, nir_channel(b, grid, 0),
                                               nir_channel(b, grid, 1)),
                                   nir_channel(b, grid, 2));
         nir_global_atomic(b, 32, k->rec, count, .atomic_op = nir_atomic_op_umax);
         nir_def *flags = hk_mesh_query_root(
            b, 32, hk_root_descriptor_offset(draw.pipeline_stats_flags));
         nir_push_if(b, nir_ine_imm(b, nir_iand_imm(b, flags,
            VK_QUERY_PIPELINE_STATISTIC_MESH_SHADER_INVOCATIONS_BIT_EXT), 0));
         {
            nir_def *addr = nir_load_stat_query_address_poly(
               b, .base = PIPE_STAT_QUERY_MS_INVOCATIONS);
            nir_global_atomic(b, 32, addr, nir_imul(b, count, k->mesh_threads),
                              .atomic_op = nir_atomic_op_iadd);
         }
         nir_pop_if(b, NULL);
      }
      nir_pop_if(b, NULL);
      nir_instr_remove(&intr->instr);
      return true;
   }
   default:
      return false;
   }
}

/* nir_lower_task_shader() addresses the payload it keeps in shared memory
 * with large constant bases; the AGX3 threadgroup forms take at most 255 in
 * their byte-offset field, so move every base into the offset. */
static bool
hk_shared_base_to_offset(nir_builder *b, nir_intrinsic_instr *intr,
                         UNUSED void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_load_shared:
   case nir_intrinsic_store_shared:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_shared_atomic_swap:
      break;
   default:
      return false;
   }
   const unsigned base = nir_intrinsic_base(intr);
   if (!base)
      return false;
   nir_src *offset = nir_get_io_offset_src(intr);
   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(offset, nir_iadd_imm(b, offset->ssa, base));
   nir_intrinsic_set_base(intr, 0);
   return true;
}

/* Lower a G17P API task shader to the compute kernel of the software task
 * stage: the draw runs it ahead of the render (hk_draw_mesh_native), and each
 * workgroup writes its child grid and payload to its entry of the task
 * record. Payload accesses go to the entry; payload atomics and sub-dword
 * payload accesses use shared memory, copied out before the launch. */
static void
hk_g17p_lower_task_kernel(nir_shader *nir)
{
   NIR_PASS(_, nir, nir_lower_task_shader,
            (nir_lower_task_shader_options){
               .payload_to_shared_for_atomics = true,
               .payload_to_shared_for_small_types = true,
            });
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_shared_base_to_offset,
            nir_metadata_control_flow, NULL);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   struct hk_task_kernel k = {.rec = hk_task_record_addr(&b)};
   nir_def *hdr = nir_load_global(&b, 2, 32, nir_iadd_imm(&b, k.rec,
      offsetof(struct libagx_task_record, stride_B)), .align_mul = 8,
      .access = ACCESS_NON_WRITEABLE);
   nir_def *entries = nir_load_global(&b, 1, 64, nir_iadd_imm(&b, k.rec,
      offsetof(struct libagx_task_record, entries)), .align_mul = 8,
      .access = ACCESS_NON_WRITEABLE);
   k.mesh_threads = nir_channel(&b, hdr, 1);
   nir_def *grid_ptr = nir_load_global_constant(&b, 1, 64,
      nir_iadd_imm(&b, nir_load_root_agx(&b),
                   hk_root_descriptor_offset(mesh_grid_addr)),
      .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
   nir_def *groups = nir_load_global_constant(&b, 3, 32, grid_ptr,
      .align_mul = 4, .access = ACCESS_CAN_SPECULATE);
   nir_def *wg = nir_load_workgroup_id(&b);
   nir_def *task = nir_iadd(&b, nir_channel(&b, wg, 0),
      nir_imul(&b, nir_channel(&b, groups, 0),
               nir_iadd(&b, nir_channel(&b, wg, 1),
                        nir_imul(&b, nir_channel(&b, groups, 1),
                                 nir_channel(&b, wg, 2)))));
   k.entry = nir_iadd(&b, entries, nir_imul(&b, nir_u2u64(&b, task),
                      nir_u2u64(&b, nir_channel(&b, hdr, 0))));

   /* An entry starts with no children, for a workgroup that never launches. */
   nir_def *lane0 = nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0);
   nir_push_if(&b, lane0);
   {
      nir_store_global(&b, nir_imm_ivec4(&b, 0, 0, 0, 0), k.entry,
                       .write_mask = 0xf, .align_mul = 16);
      nir_def *flags = hk_mesh_query_root(
         &b, 32, hk_root_descriptor_offset(draw.pipeline_stats_flags));
      nir_push_if(&b, nir_ine_imm(&b, nir_iand_imm(&b, flags,
         VK_QUERY_PIPELINE_STATISTIC_TASK_SHADER_INVOCATIONS_BIT_EXT), 0));
      {
         const unsigned threads = nir->info.workgroup_size[0] *
                                  nir->info.workgroup_size[1] *
                                  nir->info.workgroup_size[2];
         nir_def *addr = nir_load_stat_query_address_poly(
            &b, .base = PIPE_STAT_QUERY_TS_INVOCATIONS);
         nir_global_atomic(&b, 32, addr, nir_imm_int(&b, threads),
                           .atomic_op = nir_atomic_op_iadd);
      }
      nir_pop_if(&b, NULL);
   }
   nir_pop_if(&b, NULL);
   nir_progress(true, impl, nir_metadata_none);

   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_task_kernel_rewrite,
            nir_metadata_none, &k);

   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
}

static nir_shader *
hk_build_task_mesh_object(enum agx_chip chip)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "software task mesh object");
   b.shader->info.io_lowered = true;
   b.shader->info.internal = true;
   b.shader->info.workgroup_size[0] = 3;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   /* uniform0: {task record, launch} */
   nir_def *params = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *rec = nir_load_global_constant(&b, 1, 64, params, .align_mul = 8);
   nir_def *launch = nir_load_global_constant(&b, 1, 32,
      nir_iadd_imm(&b, params, 8), .align_mul = 4);
   nir_def *hdr = nir_load_global(&b, 2, 32, rec, .align_mul = 8);
   struct hk_task_rows r = hk_task_rows(&b, nir_channel(&b, hdr, 0),
                                        nir_channel(&b, hdr, 1), launch);
   nir_def *tasks = nir_channel(&b, hdr, 1);
   nir_def *whole = hk_task_whole(&b, nir_channel(&b, hdr, 0), tasks);
   nir_def *enabled = nir_bcsel(&b, whole,
      nir_iand(&b, nir_ieq_imm(&b, launch, 0),
               nir_ine_imm(&b, nir_imul(&b, r.x, tasks), 0)),
      nir_ine_imm(&b, r.l.count, 0));
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *value = nir_bcsel(&b, nir_ieq_imm(&b, lane, 0), r.x,
                              nir_bcsel(&b, nir_ieq_imm(&b, lane, 1),
                                        nir_bcsel(&b, whole, tasks, r.l.y),
                                        nir_bcsel(&b, whole, nir_imm_int(&b, 1), r.l.z)));
   nir_store_object_childcount_agx(&b, nir_bcsel(&b, enabled, value,
                                                nir_imm_int(&b, 0)));
   agx_preprocess_nir_native(b.shader, chip);
   return b.shader;
}

/* A G17P mesh shader without a task stage: its workgroups arrive in rows of
 * the API grid, launch by launch, as y rows by z slices
 * (hk_build_predicated_mesh_object()), so rebuild the workgroup ID from the
 * row, and run the body only for a row of the launch (the last slice can be
 * partial). Runs before hk_lower_nir, which lowers gl_NumWorkGroups to the
 * API grid. */
static void
hk_g17p_mesh_rows(nir_shader *nir)
{
   NIR_PASS(_, nir, hk_g17p_mesh_default_counts);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *wg = nir_load_workgroup_id(&b);
   nir_def *valid_whole = nir_imm_true(&b);
   /* A grid that fits one launch arrives as it is (hk_mesh_launch_whole()):
    * the recording says so for a grid it knows, otherwise test the grid. */
   nir_def *recorded_whole = nir_load_global_constant(&b, 1, 32,
      nir_iadd_imm(&b, nir_load_root_agx(&b), hk_root_descriptor_offset(mesh_whole)),
      .align_mul = 4, .access = ACCESS_CAN_SPECULATE);
   nir_push_if(&b, nir_ine_imm(&b, recorded_whole, 0))->control =
      nir_selection_control_dont_flatten;
   nir_push_else(&b, NULL);
   nir_def *grid = nir_load_num_workgroups(&b);
   nir_push_if(&b, hk_mesh_launch_whole(&b, nir_channel(&b, grid, 0),
                                        nir_channel(&b, grid, 1),
                                        nir_channel(&b, grid, 2)))->control =
      nir_selection_control_dont_flatten;
   nir_push_else(&b, NULL);
   nir_def *gy = nir_umax(&b, nir_channel(&b, grid, 1), nir_imm_int(&b, 1));
   struct hk_mesh_launch l = hk_mesh_launch_shape(&b, nir_channel(&b, grid, 0),
      nir_imul(&b, nir_channel(&b, grid, 1), nir_channel(&b, grid, 2)),
      hk_mesh_launch_index(&b));
   nir_def *launch_row = hk_mesh_launch_row(&b, wg, &l);
   nir_def *row = nir_iadd(&b, l.first, launch_row);
   nir_def *z = nir_udiv(&b, row, gy);
   nir_def *id_rows = nir_vec3(&b, nir_channel(&b, wg, 0),
                               nir_isub(&b, row, nir_imul(&b, z, gy)), z);
   nir_def *valid_rows = nir_ult(&b, launch_row, l.count);
   nir_pop_if(&b, NULL);
   nir_def *id_tested = nir_if_phi(&b, wg, id_rows);
   nir_def *valid_tested = nir_if_phi(&b, valid_whole, valid_rows);
   nir_pop_if(&b, NULL);
   nir_def *id = nir_if_phi(&b, wg, id_tested);
   nir_def *valid = nir_if_phi(&b, valid_whole, valid_tested);

   nir_push_if(&b, valid);
   b.cursor = nir_cf_reinsert(&body, b.cursor);
   nir_push_else(&b, NULL);
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                                      nir_undef(&b, 1, 32));
   nir_pop_if(&b, NULL);
   nir_progress(true, impl, nir_metadata_none);

   /* The mesh-only rewrite leaves gl_NumWorkGroups and payload alone. */
   nir_def *decoded[5] = {id, NULL, NULL, wg, grid};
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_mesh_task_rewrite,
            nir_metadata_control_flow, decoded);
}

/* gl_NumWorkGroups and the payload of a mesh shader used with a task stage
 * come from the task: run before hk_lower_nir. M5 reads them from the native
 * object->mesh record and returns false without one; G17P reads the software
 * task record. */
static bool
hk_mesh_task_grid(struct hk_device *dev, nir_shader *nir,
                  VkShaderCreateFlagsEXT flags)
{
   if (dev->dev.chip == AGX_CHIP_G17P) {
      if (nir->info.stage == MESA_SHADER_MESH) {
         if (flags & VK_SHADER_CREATE_NO_TASK_SHADER_BIT_EXT)
            hk_g17p_mesh_rows(nir);
         else
            hk_g17p_mesh_task_record(nir);
      }
      return true;
   }
   return hk_m5_mesh_task_grid(dev, nir, flags);
}

/* The native three-thread passthrough object program of an M5 mesh-only
 * draw is composed directly (agx3_mesh_passthrough_object), so the compiler
 * reports no staging size for it; its program declares compiler word d = 1.
 * Applied when the program is built and when it is deserialized. */
static void
hk_m5_mesh_passthrough_linked(struct hk_shader *object)
{
   object->only_linked->b.g17p_compiler_d_valid = true;
   object->only_linked->b.g17p_compiler_d = 1;
}

static bool
hk_fs_mesh_builtin_per_primitive(nir_builder *b, nir_intrinsic_instr *intr,
                                 void *data);

static bool
hk_fs_reads_primitive_input(nir_builder *b, nir_intrinsic_instr *intr,
                            void *data)
{
   if (intr->intrinsic == nir_intrinsic_load_per_primitive_input ||
       (intr->intrinsic == nir_intrinsic_load_input &&
        (nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_PRIMITIVE_ID ||
         nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_LAYER ||
         nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_VIEWPORT)))
      *(bool *)data = true;
   return false;
}

static struct hk_border_source *
hk_compile_source_create(void *nir, size_t nir_size, const union hk_key *key,
                         bool has_key, const struct vk_shader_compile_info *info,
                         enum hk_feature_key features, uint8_t static_set_mask)
{
   struct hk_border_source *src = calloc(1, sizeof(*src));
   if (!src)
      return NULL;

   simple_mtx_init(&src->lock, mtx_plain);
   src->nir = nir;
   src->nir_size = nir_size;
   src->key = *key;
   src->has_key = has_key;
   src->flags = info->flags;
   src->next_stage_mask = info->next_stage_mask;
   src->robustness = *info->robustness;
   src->features = features;
   src->set_layout_count = info->set_layout_count;
   src->static_set_mask = static_set_mask;
   return src;
}

/* twin_of: compile a twin of the shader that saved these inputs, with
 * custom-border emulation; mesh_fs: the twin is the fragment shader for a
 * G17P mesh shader with these per-primitive records (hk_mesh_fs_twin());
 * mesh_default_layer: an M5 mesh shader compiled without a fragment shader
 * (hk_m5_mesh_default_layer());
 * m5_no_grid_record: an M5 task or mesh shader compiled with its partner
 * stage, leaving the grid record out (hk_m5_no_grid_record()). */
static VkResult
hk_compile_shader_impl(struct hk_device *dev, struct vk_shader_compile_info *info,
                  const struct vk_graphics_pipeline_state *state,
                  const struct vk_features *vk_features,
                  const VkAllocationCallbacks *pAllocator,
                  const struct hk_border_source *twin_of,
                  const struct agx_m5_mesh_varyings *mesh_fs,
                  bool no_default_layer, bool mesh_default_layer,
                  int vs_covers, bool m5_no_grid_record,
                  struct hk_api_shader **shader_out)
{
   VkResult result;
   /* hk_vs_covers_clip_square() needs the system values hk_lower_nir()
    * lowers: vs_covers is its result when the caller lowered the shader
    * already, else -1; a shader this function does not lower itself below
    * never qualifies then. */
   const bool vs_covers_clip_square =
      vs_covers >= 0 ? vs_covers > 0
                     : (dev->dev.chip == AGX_CHIP_G17P ||
                        agx_chip_g17_m5(dev->dev.chip)) &&
                          info->stage == MESA_SHADER_VERTEX &&
                          !info->nir->info.io_lowered &&
                          hk_vs_covers_clip_square(info->nir);
   enum hk_feature_key features =
      twin_of ? twin_of->features : hk_make_feature_key(vk_features);
   if (!twin_of &&
       hk_set_layouts_use_buffers(info->set_layout_count, info->set_layouts))
      features |= HK_FEAT_DESCRIPTOR_BUFFER;
   const uint8_t static_set_mask =
      twin_of ? twin_of->static_set_mask
              : hk_static_set_mask(info->set_layout_count, info->set_layouts);

   /* Retain the draw-time guard for a fixed-count shader whose exports cannot
    * be unrolled. Dynamic sample-rate shaders use native variants below.
    */
   bool g17p_collapsed_sample_shading = false;

   /* We consume the NIR, regardless of success or failure */
   nir_shader *nir = info->nir;

   /* hk_lower_nir turns these into draw-root loads. Scan for them first
    * (after compute system-value lowering, which can introduce
    * num_workgroups); NIR that arrives already lowered keeps the
    * conservative answer. */
   const bool mesh_reads_draw_id =
      (nir->info.stage == MESA_SHADER_MESH ||
       nir->info.stage == MESA_SHADER_TASK) &&
      (nir->info.io_lowered ||
       hk_nir_reads_intrinsic(nir, nir_intrinsic_load_draw_id));
   const bool mesh_reads_num_workgroups =
      (nir->info.stage == MESA_SHADER_MESH ||
       nir->info.stage == MESA_SHADER_TASK) &&
      (nir->info.io_lowered ||
       hk_nir_reads_intrinsic(nir, nir_intrinsic_load_num_workgroups));

   if (!nir->info.io_lowered &&
       !hk_mesh_task_grid(dev, nir, info->flags)) {
      ralloc_free(nir);
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "mesh gl_NumWorkGroups under a task stage needs the task record");
   }
   if (!nir->info.io_lowered) {
       hk_lower_nir(dev, nir, info->robustness, false, info->next_stage_mask,
                    info->set_layout_count, info->set_layouts, features);
   }
   /* A G17P fragment shader compiled without its mesh shader cannot know
    * where that shader's per-primitive records keep each input. Keep the NIR,
    * before custom-border emulation may be dropped below, to compile the
    * twin a mesh draw needs (hk_mesh_fs_twin()). */
   void *mesh_nir = NULL;
   size_t mesh_nir_size = 0;
   bool reads_primitive_input = false;
   if (dev->dev.chip == AGX_CHIP_G17P && !mesh_fs &&
       nir->info.stage == MESA_SHADER_FRAGMENT &&
       dev->vk.enabled_features.meshShader)
      nir_shader_intrinsics_pass(nir, hk_fs_reads_primitive_input,
                                 nir_metadata_all, &reads_primitive_input);
   if (reads_primitive_input) {
      struct blob blob;
      blob_init(&blob);
      nir_serialize(&blob, nir, false);
      if (blob.out_of_memory) {
         blob_finish(&blob);
         ralloc_free(nir);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      blob_finish_get_buffer(&blob, &mesh_nir, &mesh_nir_size);
   }
   /* M5 mesh shaders export per-primitive words on private vertices. */
   if (agx_chip_g17_m5(dev->dev.chip) &&
       nir->info.stage == MESA_SHADER_FRAGMENT)
      NIR_PASS(_, nir, agx_nir_lower_m5_mesh_fragment);


   /* G17P and M5 compile the custom-border emulation in only once the
    * physical device has a sampler that needs it (hk_CreateSampler). Until
    * then, drop its branches and keep this NIR to compile the emulating twin
    * from. A shader whose emulation lost the shape it was built with keeps
    * it. */
   void *border_nir = NULL;
   size_t border_nir_size = 0;
   bool border_emulating = false;
   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
      struct hk_physical_device *pdev = hk_device_physical(dev);
      struct util_dynarray ifs;
      util_dynarray_init(&ifs, NULL);
      if (hk_find_border_emulation(nir, &ifs, dev->dev.chip == AGX_CHIP_G17P) &&
          ifs.size) {
         if (twin_of || p_atomic_read(&pdev->border_emulation)) {
            border_emulating = true;
         } else {
            struct blob blob;
            blob_init(&blob);
            nir_serialize(&blob, nir, false);
            if (blob.out_of_memory) {
               blob_finish(&blob);
               util_dynarray_fini(&ifs);
               free(mesh_nir);
               ralloc_free(nir);
               return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
            }
            blob_finish_get_buffer(&blob, &border_nir, &border_nir_size);
            hk_drop_border_emulation(&ifs);
         }
      }
      util_dynarray_fini(&ifs);
   }

   union hk_key key_tmp = {0}, *key = NULL;
   bool has_key = false;
   if (twin_of) {
      key_tmp = twin_of->key;
      has_key = twin_of->has_key;
   } else if (info->stage == MESA_SHADER_FRAGMENT) {
      hk_populate_fs_key(&key_tmp.fs, state, dev->dev.chip);
      key_tmp.fs.dynamic_zs_self_dep =
         !key_tmp.fs.zs_self_dep &&
         hk_fs_zs_self_dep_dynamic(dev->vk.physical, state, vk_features);
      has_key = true;
   } else if (info->stage == MESA_SHADER_VERTEX) {
      hk_populate_vs_key(&key_tmp.vs, state);
      key_tmp.vs.no_default_layer = no_default_layer;
      has_key = true;
   } else if (info->stage == MESA_SHADER_MESH) {
      /* Only the twins this shader keeps the NIR for read it back. */
      key_tmp.mesh.default_layer = mesh_default_layer;
   }

   /* A fragment shader whose depth/stencil feedback loop is only known at
    * draw time keeps this NIR to compile its zs_self_dep twin from. */
   void *fbl_nir = NULL;
   size_t fbl_nir_size = 0;
   if (info->stage == MESA_SHADER_FRAGMENT && !mesh_fs &&
       key_tmp.fs.dynamic_zs_self_dep && !key_tmp.fs.zs_self_dep) {
      struct blob blob;
      blob_init(&blob);
      nir_serialize(&blob, nir, false);
      if (blob.out_of_memory) {
         blob_finish(&blob);
         free(border_nir);
         free(mesh_nir);
         ralloc_free(nir);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      blob_finish_get_buffer(&blob, &fbl_nir, &fbl_nir_size);
   }

   /* Per-primitive inputs read zero without a mesh shader's records: a mesh
    * draw uses the twin instead. */
   if (dev->dev.chip == AGX_CHIP_G17P &&
       nir->info.stage == MESA_SHADER_FRAGMENT) {
      static const struct agx_m5_mesh_varyings no_mesh;
      if (mesh_fs)
         NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                  hk_fs_mesh_builtin_per_primitive, nir_metadata_control_flow,
                  NULL);
      if (!agx_nir_lower_gen1_mesh_fragment_layout(nir,
                                                   mesh_fs ? mesh_fs : &no_mesh)) {
         free(border_nir);
         free(mesh_nir);
         free(fbl_nir);
         ralloc_free(nir);
         return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
      }
   }

   if (info->stage == MESA_SHADER_FRAGMENT)
      hk_prune_unused_fragment_colors(nir, &key_tmp.fs);

   bool sample_variants = false;
   if (info->stage == MESA_SHADER_FRAGMENT) {
      /* Coverage updates also embed the raster sample count, even when the
       * application shades once per pixel. These include SampleMask writes,
       * discards, and early tests or late tests guarding memory writes. */
      const bool coverage_updates = nir->info.fs.uses_discard ||
         nir->info.writes_memory ||
         (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK)) ||
         (nir->info.fs.early_fragment_tests &&
          (nir->info.outputs_written &
           (BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
            BITFIELD64_BIT(FRAG_RESULT_STENCIL))));
      sample_variants = (agx_native_graphics(&dev->dev) ||
                         agx_chip_g17_m5(dev->dev.chip)) &&
         hk_native_fs_samples(dev->dev.chip, &key_tmp.fs) == 0 &&
         (nir->info.fs.uses_sample_shading || key_tmp.fs.force_sample_shading ||
          coverage_updates || agx_nir_needs_sample_count(nir));
   }

   const unsigned sample_variant_count = sample_variants
      ? (agx_chip_g17_m5(dev->dev.chip) ? 4 : 3) : 0;
   size_t size = hk_api_shader_size(info->stage, sample_variant_count);
   struct hk_api_shader *obj =
      vk_shader_zalloc(&dev->vk, &hk_shader_ops, info->stage, pAllocator, size);

   if (obj == NULL) {
      free(border_nir);
      free(mesh_nir);
      free(fbl_nir);
      ralloc_free(nir);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   obj->g17p_sample_variants = sample_variant_count;
   obj->border_emulating = border_emulating;
   /* Mesh shaders decide with the pipeline state (hk_mesh_clip_m1_1()). */
   obj->clip_m1_1 = info->stage == MESA_SHADER_MESH
                       ? dev->vk.enabled_features.depthClipControl
                       : hk_clip_m1_1(dev, features);

   if (border_nir) {
      obj->border_source = hk_compile_source_create(
         border_nir, border_nir_size, &key_tmp, has_key, info, features,
         static_set_mask);
      if (!obj->border_source)
         free(border_nir);
   }
   if (mesh_nir) {
      obj->mesh_source = hk_compile_source_create(
         mesh_nir, mesh_nir_size, &key_tmp, has_key, info, features,
         static_set_mask);
      if (!obj->mesh_source)
         free(mesh_nir);
   }
   if (fbl_nir) {
      union hk_key fbl_key = key_tmp;
      fbl_key.fs.zs_self_dep = true;
      obj->fbl_source = hk_compile_source_create(
         fbl_nir, fbl_nir_size, &fbl_key, has_key, info, features,
         static_set_mask);
      if (!obj->fbl_source)
         free(fbl_nir);
   }
   if ((border_nir && !obj->border_source) || (mesh_nir && !obj->mesh_source) ||
       (fbl_nir && !obj->fbl_source)) {
      ralloc_free(nir);
      hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   mesa_shader_stage sw_stage = nir->info.stage;
   const bool api_writes_external_memory =
      hk_shader_writes_external_memory(nir);

   if (sw_stage == MESA_SHADER_FRAGMENT) {
      key = &key_tmp;

      nir->info.fs.uses_sample_shading |= key->fs.force_sample_shading;

      /* Fixed-count fragments unroll in hk_compile_nir. Dynamic sample-rate
       * shaders retain their sample inputs until each variant is compiled.
       * Canonicalization is exact only at one sample; keep the draw-time
       * refusal for unsupported fixed-count export footprints.
       */
      if ((agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
          nir->info.fs.uses_sample_shading &&
          !sample_variants &&
          !hk_native_sample_shading_supported(nir, &key->fs, dev->dev.chip)) {
         g17p_collapsed_sample_shading = true;
         NIR_PASS(_, nir, agx_nir_lower_g17p_1x_sample_shading);
      }

      /* Force late-Z for Z/S self-deps. TODO: There's probably a less silly way
       * to do this.
       */
      if (key->fs.zs_self_dep) {
         nir_builder b =
            nir_builder_at(nir_before_impl(nir_shader_get_entrypoint(nir)));
         nir_discard_if(&b, nir_imm_false(&b));
         nir->info.fs.uses_discard = true;

         /* Late-Z alone only moves the depth test. Move the depth write too. */
         hk_fs_force_shader_depth_write(nir);
      }

      if (!sample_variants) {
         if (agx_chip_g17_m5(dev->dev.chip) &&
             (key_tmp.fs.m5_position_samples == 0 ||
              key_tmp.fs.m5_position_samples > 4))
            NIR_PASS(_, nir, agx_nir_lower_sample_intrinsics_wide, false);
         else
            NIR_PASS(_, nir, agx_nir_lower_sample_intrinsics, false);
      }
   } else if (sw_stage == MESA_SHADER_VERTEX) {
      key = &key_tmp;
   } else if (sw_stage == MESA_SHADER_TESS_CTRL) {
      /* A patch fits in a subgroup, so the AGX2 encoding can drop barriers
       * that only order TCS outputs, as the GL driver does. Its backend only
       * emits a threadgroup barrier in compute stages. The AGX3 encodings
       * keep them as device barriers on the lowered output buffer.
       */
      bool agx2 =
         agx_chip_isa_encoding(dev->dev.chip) == AGX_ISA_ENCODING_AGX2;

      NIR_PASS(_, nir, poly_nir_lower_tcs_with_options, agx2,
               dev->dev.chip == AGX_CHIP_G17P);
   }

   



   if (sw_stage == MESA_SHADER_MESH) {
      struct hk_shader *shader = hk_main_mesh_variant(obj);
      if (dev->dev.chip != AGX_CHIP_G17P &&
          dev->dev.chip != AGX_CHIP_G17S) {
         ralloc_free(nir);
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_error(dev, VK_ERROR_UNKNOWN);
      }
      if (agx_chip_g17_m5(dev->dev.chip))
         NIR_PASS(_, nir, agx_nir_m5_canonicalize_empty_mesh);
      else
         NIR_PASS(_, nir, hk_g17p_mesh_default_counts);
      if ((dev->dev.chip == AGX_CHIP_G17P || agx_chip_g17_m5(dev->dev.chip)) &&
          nir->info.mesh.primitive_type == MESA_PRIM_POINTS &&
          !(nir->info.outputs_written & VARYING_BIT_PSIZ))
         NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                  hk_mesh_default_point_size, nir_metadata_control_flow, NULL);
      if ((dev->dev.chip == AGX_CHIP_G17P ||
           (state && agx_chip_g17_m5(dev->dev.chip))) &&
          nir->info.mesh.primitive_type == MESA_PRIM_TRIANGLES &&
          !(nir->info.outputs_read & VARYING_BIT_PSIZ))
         NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                  kill_mesh_triangle_psiz_write,
                  nir_metadata_control_flow, NULL);

      shader->info.mesh.max_vertices = nir->info.mesh.max_vertices_out;
      shader->info.mesh.max_primitives = nir->info.mesh.max_primitives_out;
      shader->info.mesh.prim = nir->info.mesh.primitive_type;
      shader->info.mesh.reads_draw_id = mesh_reads_draw_id;
      shader->info.mesh.reads_num_workgroups = mesh_reads_num_workgroups;
      shader->info.mesh.m5_no_grid_record = m5_no_grid_record;
      for (unsigned i = 0; i < 3; ++i)
         shader->info.mesh.local_size[i] = nir->info.workgroup_size[i];
      


      shader->info.mesh.vertex_outputs =
         nir->info.outputs_written & ~nir->info.per_primitive_outputs;
      shader->info.mesh.primitive_outputs =
         nir->info.per_primitive_outputs &
         (agx_chip_g17_m5(dev->dev.chip) ? nir->info.outputs_written : ~UINT64_C(0)) &
         ~VARYING_BIT_PRIMITIVE_INDICES;
      




      NIR_PASS(_, nir, nir_opt_constant_folding);
      const bool sw_task = dev->dev.chip == AGX_CHIP_G17P &&
         !(info->flags & VK_SHADER_CREATE_NO_TASK_SHADER_BIT_EXT);
      if (dev->dev.chip == AGX_CHIP_G17P)
         NIR_PASS(_, nir, hk_lower_mesh_queries, !sw_task);
      const bool clip_m1_1 =
         hk_mesh_clip_m1_1(dev->vk.enabled_features.depthClipControl, state);
      /* G17P remaps the exported z instead (agx_nir_lower_gen1_mesh). */
      if (clip_m1_1 && dev->dev.chip != AGX_CHIP_G17P) {
         /* Mesh outputs cannot go through output temporaries. Refuse what
          * the lowering cannot convert rather than render the wrong depth. */
         if (!agx_nir_clip_m1_1_complete(nir)) {
            ralloc_free(nir);
            hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
            return vk_errorf(dev, VK_ERROR_UNKNOWN,
                             "depthClipControl needs whole z+w mesh position stores");
         }
         NIR_PASS(_, nir, agx_nir_lower_clip_m1_1);
      }
      struct agx_m5_mesh_varyings varyings;
      if (agx_chip_g17_m5(dev->dev.chip)) {
         NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_out, NULL, NULL);
         NIR_PASS(_, nir, agx_nir_lower_cull_distance_vs);
      }
      const bool m5_mesh = agx_chip_g17_m5(dev->dev.chip);
      /* The layer word gives every primitive private vertices, which must
       * fit the native output limits (agx_nir_lower_m5_mesh()). Keep the
       * shader as it was to compile instead where they do not. */
      nir_shader *m5_without_layer = NULL;
      if (dev->dev.chip == AGX_CHIP_G17S && key_tmp.mesh.default_layer) {
         m5_without_layer = nir_shader_clone(NULL, nir);
         if (m5_without_layer && !hk_m5_mesh_default_layer(nir)) {
            ralloc_free(m5_without_layer);
            m5_without_layer = NULL;
         }
      }
   m5_mesh_layout:;
      bool layout_ok = m5_mesh ? agx_m5_mesh_gather_varyings(nir, &varyings)
                              : agx_gen1_mesh_gather_varyings(nir, &varyings);
      varyings.clip_m1_1 = clip_m1_1 && dev->dev.chip == AGX_CHIP_G17P;
      if ((dev->dev.chip == AGX_CHIP_G17P || m5_mesh) &&
          nir->info.mesh.primitive_type == MESA_PRIM_LINES)
         varyings.line_bias_root_B = hk_root_descriptor_offset(line_y_bias);

      nir_builder layout = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "native mesh raster interface");
      layout.shader->info.io_lowered = true;
      if (m5_mesh) {
         layout.shader->info.clip_distance_array_size = nir->info.clip_distance_array_size;
         layout.shader->info.cull_distance_array_size = nir->info.cull_distance_array_size;
      }
      layout.shader->info.outputs_written = VARYING_BIT_POS;
      nir_store_output(&layout, nir_undef(&layout, 4, 32),
         nir_imm_int(&layout, 0), .write_mask = 15,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      if (layout_ok && !m5_mesh && varyings.lv) {
         layout.shader->info.outputs_written |= VARYING_BIT_LAYER;
         nir_store_output(&layout, nir_undef(&layout, 1, 32),
            nir_imm_int(&layout, 0), .write_mask = 1,
            .src_type = nir_type_uint32,
            .io_semantics = {.location = VARYING_SLOT_LAYER, .num_slots = 1});
      }
      /* The CLIP_DIST slots also carry cull distances (user words only). */
      if (layout_ok && !m5_mesh) {
         layout.shader->info.clip_distance_array_size = varyings.clip_count;
         for (unsigned slot = VARYING_SLOT_CLIP_DIST0;
              slot <= VARYING_SLOT_CLIP_DIST1; ++slot) {
            const unsigned n = varyings.components[slot];
            if (!n)
               continue;
            layout.shader->info.outputs_written |= BITFIELD64_BIT(slot);
            nir_store_output(&layout, nir_undef(&layout, n, 32),
               nir_imm_int(&layout, 0), .write_mask = BITFIELD_MASK(n),
               .src_type = nir_type_float32,
               .io_semantics = {.location = slot, .num_slots = 1});
         }
      }
      if (layout_ok && !m5_mesh && varyings.components[VARYING_SLOT_PSIZ]) {
         layout.shader->info.outputs_written |= VARYING_BIT_PSIZ;
         nir_store_output(&layout, nir_undef(&layout, 1, 32),
            nir_imm_int(&layout, 0), .write_mask = 1,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_PSIZ, .num_slots = 1});
      }
      for (unsigned slot = m5_mesh ? 0 : VARYING_SLOT_VAR0; layout_ok && slot < 64; ++slot) {
         const unsigned n = varyings.components[slot];
         if (!n)
            continue;
         layout.shader->info.outputs_written |= BITFIELD64_BIT(slot);
         nir_store_output(&layout, nir_undef(&layout, n, 32),
            nir_imm_int(&layout, 0), .write_mask = BITFIELD_MASK(n),
            .src_type = nir_type_float32,
            .io_semantics = {.location = slot, .num_slots = 1});
      }
      if (layout_ok && varyings.primitive_stride_B) {
         const unsigned page_words = varyings.components[AGX_MESH_PRIMITIVE_PAGE_SLOT];
         layout.shader->info.outputs_written |= BITFIELD64_BIT(AGX_MESH_PRIMITIVE_PAGE_SLOT);
         nir_store_output(&layout, nir_undef(&layout, page_words, 32), nir_imm_int(&layout, 0),
            .write_mask = BITFIELD_MASK(page_words), .src_type = nir_type_uint32,
            .io_semantics = {.location = AGX_MESH_PRIMITIVE_PAGE_SLOT, .num_slots = 1});
      }
      /* An M5 mesh layer/viewport is also a user word (see
       * agx_m5_mesh_gather_varyings), as for the M5 vertex stage. */
      agx_nir_lower_uvs(layout.shader, &shader->info.uvs,
                        m5_mesh && layout_ok && varyings.lv, false, -1);
      shader->info.uvs.mesh_primitive_page = varyings.primitive_stride_B != 0;
      shader->info.mesh.primitive_stride_B = varyings.primitive_stride_B;
      if (!m5_mesh)
         memcpy(shader->g17p_mesh_primitive_comps, varyings.primitive_comps,
                sizeof(shader->g17p_mesh_primitive_comps));
      ralloc_free(layout.shader);
      if (layout_ok && varyings.dynamic &&
          nir->info.known_interpolation_qualifiers) {
         


         struct agx_varyings_vs linked;
         agx_assign_uvs(&linked, &shader->info.uvs,
                        ~(nir->info.linear_varyings |
                          nir->info.perspective_varyings),
                        nir->info.linear_varyings);
         for (unsigned slot = 0; slot < 64; ++slot) {
            if (varyings.components[slot] && slot != VARYING_SLOT_PSIZ)
               varyings.first[slot] = linked.slots[slot];
         }
         if (varyings.components[VARYING_SLOT_PSIZ])
            varyings.first[VARYING_SLOT_PSIZ] = shader->info.uvs.group_offs[UVS_PSIZ];
         if (varyings.lv)
            varyings.lv_first = shader->info.uvs.group_offs[UVS_LAYER_VIEWPORT];
         varyings.dynamic = false;
      }
      layout_ok &= varyings.size == shader->info.uvs.size;
      layout_ok &= !varyings.clip_distance_count ||
         varyings.clip_distance_first == shader->info.uvs.group_offs[UVS_CLIP_DIST];
      layout_ok &= !m5_mesh || !varyings.lv ||
         varyings.lv_first == shader->info.uvs.group_offs[UVS_LAYER_VIEWPORT];
      /* Nor may the layer word take the user words past what a draw links
       * (hk_g17p_smooth_varyings_supported_uncached()). */
      layout_ok &= !m5_without_layer ||
         shader->info.uvs.user_size <= HK_AGX3_G17P_MAX_USER_WORDS;
      if (!layout_ok ||
          !(m5_mesh ? agx_nir_lower_m5_task_payload(nir,
                         hk_m5_task_payload_limit(dev->dev.params.features),
                         hk_m5_task_record(dev->dev.params.features) &&
                            !m5_no_grid_record ? 16 : 0)
                    : agx_nir_lower_gen1_task_payload_probe(nir)) ||
          !(m5_mesh ? agx_nir_lower_m5_mesh(nir, &varyings)
                    : agx_nir_lower_gen1_mesh(nir, &varyings))) {
         if (m5_without_layer) {
            /* Drop this attempt, which the payload lowering may already
             * have changed, and lay out the shader as it was before the
             * layer stores, as without them. */
            ralloc_free(nir);
            nir = m5_without_layer;
            m5_without_layer = NULL;
            memset(&shader->info.uvs, 0, sizeof(shader->info.uvs));
            goto m5_mesh_layout;
         }
         ralloc_free(nir);
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_errorf(dev, VK_ERROR_UNKNOWN,
                          "native mesh layout outside captured triangle contract");
      }
      if (key_tmp.mesh.default_layer) {
         const bool m5_default_layer = m5_without_layer != NULL;
         ralloc_free(m5_without_layer);
         if (m5_default_layer)
            shader->info.mesh.primitive_outputs |= VARYING_BIT_LAYER;
         /* A twin compiles with what this shader compiled with. */
         if (obj->border_source)
            obj->border_source->key.mesh.default_layer = m5_default_layer;
      }
      /* Per-primitive outputs can raise the exported vertex count. */
      shader->info.mesh.max_vertices = nir->info.mesh.max_vertices_out;
      if (agx_chip_g17_m5(dev->dev.chip))
         agx_preprocess_nir_m5(nir);
      else
         agx_preprocess_nir_native(nir, dev->dev.chip);
      result = hk_compile_nir(dev, pAllocator, nir, info->flags,
                             info->robustness, NULL, features, shader,
                             sw_stage, true, NULL, info->set_layout_count,
                             static_set_mask);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return result;
      }
      /* Native mesh lowering introduces private transport stores after the
       * API write class was gathered. They do not change this declaration. */
      shader->info.writes_external_memory = api_writes_external_memory;
      struct hk_shader *object = &obj->variants[HK_MESH_VARIANT_RAST];
      if (dev->dev.chip == AGX_CHIP_G17P) {
      nir_shader *object_nir = sw_task
         ? hk_build_task_mesh_object(dev->dev.chip)
         : hk_build_predicated_mesh_object(dev->dev.chip);
      if (!object_nir) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_error(dev, VK_ERROR_UNKNOWN);
      }
      result = hk_compile_nir(dev, pAllocator, object_nir,
                             info->flags, info->robustness, NULL, features,
                             object, MESA_SHADER_TASK, true, NULL, 0, 0);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return result;
      }
      object->info.writes_external_memory = false;
      } else {
      /* Mesh-only Vulkan pipelines use Apple's native three-thread
       * passthrough object stage, whose operand-bearing code is composed,
       * not copied from a fixed shader blob. */
      uint8_t body[64], size;
      if (!agx3_mesh_passthrough_object(0, body, &size)) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_error(dev, VK_ERROR_UNKNOWN);
      }
      object->b.binary = malloc(size);
      if (!object->b.binary) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      memcpy(object->b.binary, body, size);
      object->code_ptr = object->b.binary;
      object->code_size = size;
      object->info.stage = MESA_SHADER_TASK;
      object->b.info.stage = MESA_SHADER_TASK;
      object->b.info.binary_size = size;
      object->b.info.main_size = size;
      object->b.info.nr_gprs = 8;
      object->b.info.workgroup_size[0] = 3;
      object->b.info.workgroup_size[1] = object->b.info.workgroup_size[2] = 1;
      result = hk_init_link_ht(object, MESA_SHADER_TASK);
      if (result == VK_SUCCESS)
         result = hk_upload_shader(dev, object);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_error(dev, result);
      }
      hk_m5_mesh_passthrough_linked(object);
      object->compiled = true;
      }
      struct hk_shader *predicated =
         &obj->variants[HK_MESH_VARIANT_PREDICATED_OBJECT];
      nir_shader *object_nir = sw_task
         ? hk_build_task_mesh_object(dev->dev.chip)
         : hk_build_predicated_mesh_object(dev->dev.chip);
      if (!object_nir) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_error(dev, VK_ERROR_UNKNOWN);
      }
      result = hk_compile_nir(dev, pAllocator, object_nir,
                             info->flags, info->robustness, NULL, features,
                             predicated, MESA_SHADER_TASK, true, NULL, 0, 0);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return result;
      }
      /* This compiler-generated object only transports the mesh launch. */
      predicated->info.writes_external_memory = false;
   } else if (sw_stage == MESA_SHADER_TASK &&
              dev->dev.chip == AGX_CHIP_G17P) {
      /* The software task stage: a compute kernel run ahead of the render. */
      const unsigned stride_B = LIBAGX_TASK_ENTRY_PAYLOAD_B +
                                ALIGN_POT(nir->info.task_payload_size, 16);
      hk_g17p_lower_task_kernel(nir);
      agx_preprocess_nir_native(nir, dev->dev.chip);
      result = hk_compile_nir(dev, pAllocator, nir, info->flags,
                             info->robustness, NULL, features, &obj->variants[0],
                             sw_stage, false, NULL, info->set_layout_count,
                             static_set_mask);
      if (result == VK_SUCCESS) {
         obj->variants[0].info.writes_external_memory = api_writes_external_memory;
         obj->variants[0].info.mesh.payload_size_B = stride_B;
      }
   } else if (sw_stage == MESA_SHADER_TASK) {
      /* Compile the actual API object shader through captured native payload
       * addressing and child-grid UVSW, never as a CDM mesh producer.
       * taskShader is advertised only with the kernel's task mesh storage
       * (hk_m5_task_record). */
      hk_m5_task_predicate(nir);
      if (dev->dev.chip != AGX_CHIP_G17S ||
          !agx_nir_lower_m5_task(nir,
              hk_m5_task_payload_limit(dev->dev.params.features),
              hk_m5_task_record(dev->dev.params.features) &&
                 !m5_no_grid_record)) {
         ralloc_free(nir);
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_errorf(dev, VK_ERROR_UNKNOWN,
                          "native task layout outside captured object contract");
      }
      if (agx_chip_g17_m5(dev->dev.chip))
         agx_preprocess_nir_m5(nir);
      else
         agx_preprocess_nir_native(nir, dev->dev.chip);
      result = hk_compile_nir(dev, pAllocator, nir, info->flags,
                             info->robustness, NULL, features, &obj->variants[0],
                             sw_stage, true, NULL, info->set_layout_count,
                             static_set_mask);
      if (result == VK_SUCCESS) {
         obj->variants[0].info.writes_external_memory = api_writes_external_memory;
         /* hk_draw_mesh_native publishes the API grid and draw index to the
          * object stage's draw root only when it reads them. */
         obj->variants[0].info.mesh.reads_draw_id = mesh_reads_draw_id;
         obj->variants[0].info.mesh.reads_num_workgroups =
            mesh_reads_num_workgroups;
         obj->variants[0].info.mesh.m5_no_grid_record = m5_no_grid_record;
      }
   } else if (sw_stage == MESA_SHADER_GEOMETRY) {
      struct hk_shader *main_variant = hk_main_gs_variant(obj);
      struct hk_shader *count_variant = hk_count_gs_variant(obj);

      nir_shader *count = NULL, *rast = NULL, *pre_gs = NULL;

      struct agx_gen1_ray_query_layout rq_layout =
         hk_gen1_ray_query_layout(dev, nir, 0);
      /* The raster replay is always a hardware vertex shader. */
      rq_layout.graphics_stage = dev->gen1_native_graphics_rq;
      /* On M5 the main kernel performs the invocation's API stores and
       * atomics; the count and raster variants drop them. */
      const bool m5 = agx_chip_g17_m5(dev->dev.chip);
      if (!hk_nir_lower_gs(nir, &count, &rast, &pre_gs,
                           &count_variant->info.gs,
                           hk_native_rq_profile(dev) ? &rq_layout : NULL, m5,
                           NULL)) {
         ralloc_free(nir);
         ralloc_free(count);
         ralloc_free(rast);
         ralloc_free(pre_gs);
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_errorf(dev, VK_ERROR_UNKNOWN,
                          "Gen1 native GS ray query lowering failed");
      }

      agx_preprocess_nir(count);
      agx_preprocess_nir(rast);
      agx_preprocess_nir(pre_gs);

      /* The raster variant is a hardware vertex shader. On M5, match the
       * ordinary VS/TES default when the geometry producer has no point-size
       * output, including a producer that emits no vertices. Other chips keep
       * agx3b's raster variant, which writes no default point size.
       */
      if (agx_chip_g17_m5(dev->dev.chip))
         NIR_PASS(_, rast, nir_lower_default_point_size);

      struct hk_shader *shader = &obj->variants[HK_GS_VARIANT_RAST];
      if (agx_native_graphics(&dev->dev)) {
         NIR_PASS(_, rast, nir_shader_intrinsics_pass,
                  hk_g17p_saturate_layer_viewport, nir_metadata_control_flow,
                  NULL);
      }
      hk_lower_hw_vs(dev, rast, shader, false,
                     agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip),
                     true, hk_clip_m1_1(dev, features),
                     agx_native_graphics(&dev->dev)
                        ? hk_root_descriptor_offset(draw.no_attachments)
                        : -1);
      shader->info.gs = count_variant->info.gs;
      main_variant->info.gs = count_variant->info.gs;

      struct {
         nir_shader *in;
         struct hk_shader *out;
      } variants[] = {
         {nir, hk_main_gs_variant(obj)},
         {pre_gs, hk_pre_gs_variant(obj)},
         {count, count_variant},
         {rast, &obj->variants[HK_GS_VARIANT_RAST]},
      };

      for (unsigned v = 0; v < ARRAY_SIZE(variants); ++v) {
         if (variants[v].in) {
            /* hk_compile_nir() takes ownership, including on failure. */
            nir_shader *variant_nir = variants[v].in;
            variants[v].in = NULL;
            result =
               hk_compile_nir(dev, pAllocator, variant_nir, info->flags,
                              info->robustness, NULL, features, variants[v].out,
                              sw_stage, true, NULL, info->set_layout_count,
                              static_set_mask);

            if (result != VK_SUCCESS) {
               hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
               for (unsigned remaining = v + 1;
                    remaining < ARRAY_SIZE(variants); ++remaining)
                  ralloc_free(variants[remaining].in);
               return result;
            }
         }
      }
   } else if (sw_stage == MESA_SHADER_VERTEX ||
              sw_stage == MESA_SHADER_TESS_EVAL) {

      VkShaderStageFlags next_stage = info->next_stage_mask;

      /* Transform feedback is layered on top of geometry shaders. If there is
       * not a geometry shader in the pipeline, we will compile a geometry
       * shader for the purpose. Update the next_stage mask accordingly.
       */
      if (nir->xfb_info != NULL) {
         next_stage |= VK_SHADER_STAGE_GEOMETRY_BIT;
      }

      /* Native TES ray queries execute in Poly's compute variant, which
       * needs the software variant (see hk_tes_ray_queries_in_compute()).
       */
      if (sw_stage == MESA_SHADER_TESS_EVAL && nir->info.ray_queries > 0 &&
          hk_tes_ray_queries_in_compute(dev)) {
         next_stage |= VK_SHADER_STAGE_GEOMETRY_BIT;
      }

      if (sw_stage == MESA_SHADER_VERTEX) {
         assert(
            !(nir->info.inputs_read & BITFIELD64_MASK(VERT_ATTRIB_GENERIC0)) &&
            "Fixed-function attributes not used in Vulkan");

         /* Attributes are compacted into slots indexing fixed-size keys,
          * component masks and root tables, and the prolog ABI hands them
          * over in registers. A shader reading more than the advertised
          * maxVertexInputAttributes is invalid API usage; refuse it here
          * instead of indexing past those tables (DXVK's 18-attribute D3D9
          * fixed-function vertex shader did exactly that under the old
          * 16-attribute limit and crashed in the vertex fetch lowering).
          */
         const unsigned nr_attribs_read =
            util_bitcount64(nir->info.inputs_read >> VERT_ATTRIB_GENERIC0);
         const struct agx_device_key device_key =
            agx_gather_device_key(&dev->dev);
         const unsigned max_attribs = agx_max_vertex_attribs(&device_key);
         if (nr_attribs_read > max_attribs) {
            hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
            ralloc_free(nir);
            return vk_errorf(dev, VK_ERROR_UNKNOWN,
                             "vertex shader reads %u attributes, device "
                             "supports %u",
                             nr_attribs_read, max_attribs);
         }

         NIR_PASS(_, nir, nir_recompute_io_bases, nir_var_shader_in);
      }

      /* Fold constant offset srcs for IO. */
      NIR_PASS(_, nir, nir_opt_constant_folding);

      /* Only compile the software variant if we might use this shader with
       * geometry/tessellation. We need to compile the hardware variant
       * unconditionally to handle the VS -> null FS case, which does not
       * require setting the FRAGMENT bit.
       */
      const bool sw_variant =
         next_stage & (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                       VK_SHADER_STAGE_GEOMETRY_BIT);
      const enum hk_vs_variant last_variant =
         sw_variant ? HK_VS_VARIANT_SW : HK_VS_VARIANT_HW;

      for (enum hk_vs_variant v = 0; v < HK_VS_VARIANTS; ++v) {
         if (v == HK_VS_VARIANT_SW && !sw_variant)
            continue;

         struct hk_shader *shader = &obj->variants[v];
         bool hw = v == HK_VS_VARIANT_HW;
         bool last = v == last_variant;

         /* Each variant gets its own NIR. To save an extra clone, we use the
          * original NIR for the last variant compiled. (Using it only for the
          * software variant leaked the original whenever that variant was not
          * needed, the common vertex -> fragment case.)
          */
         nir_shader *clone = last ? nir : nir_shader_clone(NULL, nir);

         /* Without a declared fragment successor this hardware variant is
          * only used for VS -> null FS. Geometry/tessellation and transform
          * feedback consume the separate software variant's outputs.
          * Remove unconsumed user outputs before assigning a UVS layout;
          * fixed-function outputs and side effects remain intact. */
         if (hw && !(next_stage & VK_SHADER_STAGE_FRAGMENT_BIT)) {
            NIR_PASS(_, clone, nir_remove_outputs, MESA_SHADER_FRAGMENT,
                     BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32), 0);
         }

         NIR_PASS(_, clone, agx_nir_gather_vs_inputs,
                  shader->info.vs.attrib_components_read);

         if (sw_stage == MESA_SHADER_VERTEX) {
            shader->info.vs.use_prolog = !(key && key->vs.skip_prolog);
            shader->info.vs.attribs_read =
               nir->info.inputs_read >> VERT_ATTRIB_GENERIC0;

            if (shader->info.vs.use_prolog) {
               NIR_PASS(_, clone, agx_nir_lower_vs_input_to_prolog);
            } else {
               struct agx_velem_key attribs[AGX_MAX_ATTRIBS] = {0};
               for (unsigned a = 0; a < AGX_MAX_ATTRIBS; ++a) {
                  if (key->vs.attribs[a].format &&
                      (shader->info.vs.attribs_read & BITFIELD_BIT(a))) {
                     unsigned slot = util_bitcount64(
                        shader->info.vs.attribs_read & BITFIELD_MASK(a));

                     attribs[slot] = key->vs.attribs[a];
                  }
               }

               struct agx_robustness agx_rs = {
                  .soft_fault = agx_has_soft_fault(&dev->dev),

                  /* Correctly handling GPL + pipeline-robustness requires
                   * runtime changes, so derive the level from the enabled
                   * device features exactly as the vertex prolog does
                   * (hk_cmd_draw.c:3200-3208) rather than pessimising every
                   * pipeline to D3D.  Hardcoding D3D here made every vertex
                   * fetch emit an out-of-bounds select even when the
                   * application never asked for robustness; the two paths
                   * must agree anyway, since a monolithic VS and a prologue
                   * fetch the same attributes.
                   */
                  .level =
                     (dev->vk.enabled_features.robustBufferAccess2 ||
                      dev->vk.enabled_features.pipelineRobustness)
                        ? AGX_ROBUSTNESS_D3D
                     : dev->vk.enabled_features.robustBufferAccess
                        ? AGX_ROBUSTNESS_GL
                        : AGX_ROBUSTNESS_DISABLED,
               };

               if (agx_native_graphics(&dev->dev) ||
                   agx_chip_g17_m5(dev->dev.chip))
                  agx_nir_lower_vbo_raw(clone, attribs, agx_rs,
                                        !key->vs.static_strides);
               else
                  agx_nir_lower_vbo(clone, attribs, agx_rs,
                                    !key->vs.static_strides);

               unsigned nr = DIV_ROUND_UP(
                  BITSET_LAST_BIT(shader->info.vs.attrib_components_read), 4);
               /* Native query state owns uniform halves 0..167; the vertex
                * ABI then starts at 168, as the prolog key and the bound
                * uniform ranges already assume. */
               const bool query_prefix =
                  dev->gen1_native_graphics_rq && hk_nir_has_ray_query(clone);
               agx_nir_lower_non_monolithic_uniforms_at(clone, nr,
                                                        query_prefix ? 168 : 0);
            }
         }

         /* hk_compile_nir takes ownership of the clone */
         result =
            hk_compile_nir(dev, pAllocator, clone, info->flags,
                           info->robustness, key, features, shader, sw_stage,
                           hw, nir->xfb_info, info->set_layout_count,
                           static_set_mask);
         if (result != VK_SUCCESS) {
            hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
            if (!last)
               ralloc_free(nir);
            return result;
         }
         shader->info.vs.covers_clip_square = vs_covers_clip_square;
      }
   } else if (sample_variants) {
      /* The sample count is dynamic API state, but native coverage words and
       * export registers depend on it. Compile the same native specialization
       * as a fixed-count pipeline, without changing GPU batching or inserting
       * an AGX2 machine-code loop into a G17P fragment. The one-sample
       * program compiles now; the multisample ones on first use, so a
       * library that never meets a multisampled draw does not pay for them.
       * Refuse a shape no sample count can export now, as before. */
      if (nir->info.fs.uses_sample_shading) {
         const unsigned max_samples = agx_chip_g17_m5(dev->dev.chip) ? 8 : 4;
         for (unsigned samples = 2; samples <= max_samples; samples *= 2) {
            key->fs.g17p_samples = samples;
            if (agx_chip_g17_m5(dev->dev.chip))
               key->fs.m5_position_samples = samples;
            if (!hk_native_sample_shading_supported(nir, &key->fs, dev->dev.chip)) {
               ralloc_free(nir);
               hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
               /* Same contract as above: not a legal code for any caller of
                * hk_compile_shaders(). */
               return vk_error(dev, VK_ERROR_UNKNOWN);
            }
         }
      }
      key->fs.g17p_samples = 0;
      if (agx_chip_g17_m5(dev->dev.chip))
         key->fs.m5_position_samples = 0;

      result = hk_fs_sample_source_init(obj, nir, key, info, features,
                                        static_set_mask);
      if (result == VK_SUCCESS)
         result = hk_compile_fs_sample_variant(dev, pAllocator, nir, obj, 1);
      else
         ralloc_free(nir);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return result;
      }
   } else {
      struct hk_shader *shader = hk_only_variant(obj);

      /* hk_compile_nir takes ownership of nir */
      result = hk_compile_nir(dev, pAllocator, nir, info->flags,
                              info->robustness, key, features, shader, sw_stage,
                              true, NULL, info->set_layout_count,
                              static_set_mask);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return result;
      }
   }

   /* hk_compile_nir() runs after the collapse above, so by the time it fills
    * hk_shader_info the NIR flag is already cleared.  Publish the API-level
    * intent here, where every variant exists and has been compiled.
    */
   if (sw_stage == MESA_SHADER_FRAGMENT && g17p_collapsed_sample_shading)
      hk_only_variant(obj)->info.fs.uses_sample_shading = true;

   result = hk_shader_prepare_device_for_stage(dev, sw_stage);
   if (result != VK_SUCCESS) {
      hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
      return result;
   }

   *shader_out = obj;
   return VK_SUCCESS;
}


VkResult
hk_compile_shader(struct hk_device *dev, struct vk_shader_compile_info *info,
                  const struct vk_graphics_pipeline_state *state,
                  const struct vk_features *features,
                  const VkAllocationCallbacks *allocator,
                  struct hk_api_shader **shader_out)
{
   return hk_compile_shader_impl(dev, info, state, features, allocator, NULL,
                                 NULL, false, false, -1, false, shader_out);
}

static VkResult
hk_compile_twin(struct hk_device *dev, const struct hk_api_shader *obj,
                const struct hk_border_source *src,
                const struct agx_m5_mesh_varyings *mesh_fs,
                struct hk_api_shader **built)
{
   *built = NULL;
   struct blob_reader reader;
   blob_reader_init(&reader, src->nir, src->nir_size);
   nir_shader *nir = nir_deserialize(NULL, &agx_nir_options, &reader);
   if (!nir)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_shader_compile_info info = {
      .stage = obj->vk.stage,
      .flags = src->flags,
      .next_stage_mask = src->next_stage_mask,
      .nir = nir,
      .robustness = &src->robustness,
      .set_layout_count = src->set_layout_count,
   };
   struct hk_api_shader *shader = NULL;
   /* A twin keeps the payload layout of the shader it stands in for: it is
    * drawn with the same partner stage. */
   const bool m5_no_grid_record = obj->variants[0].info.mesh.m5_no_grid_record;
   VkResult result = hk_compile_shader_impl(dev, &info, NULL, NULL, NULL, src,
                                            mesh_fs, false, false, -1,
                                            m5_no_grid_record, &shader);
   *built = result == VK_SUCCESS ? shader : NULL;
   return result;
}

static struct hk_api_shader *
hk_source_twin(struct hk_device *dev, struct hk_api_shader *obj,
               struct hk_border_source *src)
{
   struct hk_api_shader *twin = p_atomic_read(&src->twin);
   if (twin)
      return twin;

   simple_mtx_lock(&src->lock);
   if (!src->twin) {
      hk_compile_twin(dev, obj, src, NULL, &twin);
      p_atomic_set(&src->twin, twin);
   }
   twin = src->twin;
   simple_mtx_unlock(&src->lock);
   return twin;
}

struct hk_api_shader *
hk_border_twin(struct hk_device *dev, struct hk_api_shader *obj)
{
   return hk_source_twin(dev, obj, obj->border_source);
}

struct hk_api_shader *
hk_fbl_twin(struct hk_device *dev, struct hk_api_shader *obj)
{
   return hk_source_twin(dev, obj, obj->fbl_source);
}

VkResult
hk_mesh_fs_twin(struct hk_device *dev, struct hk_api_shader *obj,
                const struct hk_shader *mesh, struct hk_api_shader **out)
{
   struct hk_border_source *src = obj->mesh_source;
   const uint8_t *comps = mesh->g17p_mesh_primitive_comps;
   VkResult result = VK_SUCCESS;

   simple_mtx_lock(&src->lock);
   struct hk_mesh_fs_twin *twin = obj->mesh_twins;
   while (twin && memcmp(twin->comps, comps, sizeof(twin->comps)))
      twin = twin->next;
   if (!twin && !(twin = calloc(1, sizeof(*twin)))) {
      result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   } else if (!twin->shader) {
      /* A listed twin always has a shader, so this one is new. */
      struct agx_m5_mesh_varyings layout = {0};
      memcpy(layout.primitive_comps, comps, sizeof(twin->comps));
      agx_gen1_mesh_record_layout(&layout);
      result = hk_compile_twin(dev, obj, src, &layout, &twin->shader);
      if (result == VK_SUCCESS) {
         memcpy(twin->comps, comps, sizeof(twin->comps));
         twin->next = obj->mesh_twins;
         obj->mesh_twins = twin;
      } else {
         free(twin);
         twin = NULL;
      }
   }
   simple_mtx_unlock(&src->lock);
   *out = twin ? twin->shader : NULL;
   return result;
}

static void
nir_opts(nir_shader *nir, void *data)
{
   bool progress;

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_loop);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);

      NIR_PASS(progress, nir, nir_opt_if, 0);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);

      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &(nir_opt_peephole_select_options){
                  .limit = 8,
                  .expensive_alu_ok = true,
                  .discard_ok = true,
               });

      NIR_PASS(progress, nir, nir_opt_phi_precision);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);

      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);
}

/* Native varyings use 32-bit components. Widen narrow integer interfaces
 * before interstage optimization: nir_opt_varyings may retype flat IO as
 * float, after which a numeric half-float conversion loses integer NaN bits.
 * Keep narrowing inside the consumer, where it is ordinary integer ALU.
 */
static bool
hk_widen_integer_io(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   bool store;
   switch (intr->intrinsic) {
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
      if (b->shader->info.stage == MESA_SHADER_FRAGMENT)
         return false;
      store = true;
      break;
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_per_vertex_input:
      if (b->shader->info.stage == MESA_SHADER_VERTEX)
         return false;
      store = false;
      break;
   case nir_intrinsic_load_output:
   case nir_intrinsic_load_per_vertex_output:
      store = false;
      break;
   default:
      return false;
   }

   nir_alu_type type = store ? nir_intrinsic_src_type(intr)
                            : nir_intrinsic_dest_type(intr);
   nir_alu_type base = nir_alu_type_get_base_type(type);
   unsigned bits = store ? intr->src[0].ssa->bit_size : intr->def.bit_size;
   if (bits != 16 || (base != nir_type_int && base != nir_type_uint))
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (store && sem.no_varying)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   if (store) {
      /* XFB observes the original component width independently of the
       * internal varying representation. Retain a separate XFB-only store. */
      if (nir_instr_xfb_write_mask(intr)) {
         nir_intrinsic_instr *xfb = nir_instr_as_intrinsic(
            nir_instr_clone(b->shader, &intr->instr));
         nir_io_semantics xfb_sem = sem;
         xfb_sem.no_varying = true;
         nir_intrinsic_set_io_semantics(xfb, xfb_sem);
         nir_builder_instr_insert(b, &xfb->instr);
         nir_intrinsic_set_io_xfb(intr, (nir_io_xfb){0});
      }
      nir_def *value = nir_convert_to_bit_size(b, intr->src[0].ssa, base, 32);
      nir_src_rewrite(&intr->src[0], value);
      nir_intrinsic_set_src_type(intr, base | 32);
   } else {
      intr->def.bit_size = 32;
      nir_intrinsic_set_dest_type(intr, base | 32);
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *value = nir_u2u16(b, &intr->def);
      nir_def_rewrite_uses_after(&intr->def, value);
   }
   return true;
}

static bool
hk_fs_mesh_builtin_per_primitive(nir_builder *b, nir_intrinsic_instr *intr,
                                 void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input)
      return false;
   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != VARYING_SLOT_LAYER &&
       sem.location != VARYING_SLOT_VIEWPORT &&
       sem.location != VARYING_SLOT_PRIMITIVE_ID)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *repl = nir_load_per_primitive_input(
      b, intr->def.num_components, intr->def.bit_size, intr->src[0].ssa,
      .base = nir_intrinsic_base(intr),
      .component = nir_intrinsic_component(intr),
      .dest_type = nir_intrinsic_dest_type(intr), .io_semantics = sem);
   nir_def_replace(&intr->def, repl);
   return true;
}

/* M5 task and mesh shaders agree on where the task payload sits at the
 * payload address. A mesh that reads gl_NumWorkGroups takes it from the
 * child grid record the task writes at +0, with the payload from +16
 * (hk_m5_mesh_task_grid()). Any other mesh needs no record, so the pair can
 * drop it and keep the payload at +0, but only when both are compiled here
 * together, linked: the stages of a pipeline, or shader objects created with
 * VK_SHADER_CREATE_LINK_STAGE_BIT_EXT, which must be bound together. A task
 * or mesh shader compiled alone may be drawn with any partner, so it keeps
 * the record layout, as do both stages of every other pairing. Run before
 * hk_m5_mesh_task_grid() and hk_lower_nir(), with the same gl_NumWorkGroups
 * test as the former. */
static bool
hk_m5_no_grid_record(const struct hk_device *dev, uint32_t shader_count,
                     const struct vk_shader_compile_info *infos)
{
   if (dev->dev.chip != AGX_CHIP_G17S ||
       !hk_m5_task_record(dev->dev.params.features))
      return false;
   const struct vk_shader_compile_info *task = NULL, *mesh = NULL;
   for (uint32_t i = 0; i < shader_count; i++) {
      if (infos[i].stage == MESA_SHADER_TASK)
         task = &infos[i];
      else if (infos[i].stage == MESA_SHADER_MESH)
         mesh = &infos[i];
   }
   return task && mesh &&
          (task->flags & VK_SHADER_CREATE_LINK_STAGE_BIT_EXT) &&
          (mesh->flags & VK_SHADER_CREATE_LINK_STAGE_BIT_EXT) &&
          !hk_nir_reads_intrinsic(mesh->nir, nir_intrinsic_load_num_workgroups);
}

static VkResult
hk_compile_shaders(struct vk_device *vk_dev, uint32_t shader_count,
                   struct vk_shader_compile_info *infos,
                   const struct vk_graphics_pipeline_state *state,
                   const struct vk_features *features,
                   const VkAllocationCallbacks *pAllocator,
                   struct vk_shader **shaders_out)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);
   nir_shader *shaders[shader_count];
   int vs_covers[shader_count];
   for (uint32_t i = 0; i < shader_count; i++)
      vs_covers[i] = -1;
   bool has_mesh = false;
   const bool m5_no_grid_record =
      hk_m5_no_grid_record(dev, shader_count, infos);

   for (uint32_t i = 0; i < shader_count; i++) {
      if (hk_nir_layout_has_descriptors(infos[i].nir, infos[i].set_layout_count,
                                        infos[i].set_layouts))
         continue;

      const mesa_shader_stage stage = infos[i].nir->info.stage;
      for (uint32_t j = 0; j < shader_count; j++)
         ralloc_free(infos[j].nir);
      memset(shaders_out, 0, shader_count * sizeof(*shaders_out));
      return vk_errorf(dev, VK_ERROR_UNKNOWN,
                       "%s shader uses a descriptor missing from its pipeline "
                       "layout", mesa_shader_stage_name(stage));
   }

   /* Lower shaders, notably lowering I/O. This is a prerequisite for
    * intershader optimization.
    */
   for (uint32_t i = 0; i < shader_count; i++) {
      const struct vk_shader_compile_info *info = &infos[i];
      /* TODO: Multiview with ESO */
      const bool is_multiview = state && state->mv->view_mask != 0;
      enum hk_feature_key hk_features = hk_make_feature_key(features);
      nir_shader *nir = info->nir;

      if (nir->info.stage == MESA_SHADER_MESH)
         has_mesh = true;

      if (nir->info.stage == MESA_SHADER_FRAGMENT && has_mesh) {
         nir_foreach_shader_in_variable(var, nir) {
            if (var->data.location == VARYING_SLOT_PRIMITIVE_ID ||
                var->data.location == VARYING_SLOT_VIEWPORT ||
                var->data.location == VARYING_SLOT_LAYER)
               var->data.per_primitive = true;
         }
      }

      vs_covers[i] = (dev->dev.chip == AGX_CHIP_G17P ||
                      agx_chip_g17_m5(dev->dev.chip)) &&
                     nir->info.stage == MESA_SHADER_VERTEX &&
                     hk_vs_covers_clip_square(nir);
      if (!hk_mesh_task_grid(dev, nir, info->flags)) {
         for (uint32_t j = 0; j < shader_count; j++)
            ralloc_free(infos[j].nir);
         memset(shaders_out, 0, shader_count * sizeof(*shaders_out));
         return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                          "mesh gl_NumWorkGroups under a task stage needs the task record");
      }
      /* Matrices of flexible dimensions are 16x16 or 8x8 pieces by now. What
       * is left must be a size and an operation the lowering takes. */
      if (agx_chip_g17_m5(dev->dev.chip) &&
          nir->info.stage == MESA_SHADER_COMPUTE &&
          !agx_nir_cmat_supported(nir)) {
         for (uint32_t j = 0; j < shader_count; j++)
            ralloc_free(infos[j].nir);
         memset(shaders_out, 0, shader_count * sizeof(*shaders_out));
         return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                          "cooperative matrix size or operation not supported");
      }
      hk_lower_nir(dev, nir, info->robustness, is_multiview,
                   info->next_stage_mask, info->set_layout_count,
                    info->set_layouts, hk_features);
      if (nir->info.stage == MESA_SHADER_FRAGMENT && has_mesh)
         NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                  hk_fs_mesh_builtin_per_primitive, nir_metadata_control_flow,
                  NULL);

      if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
         enum hk_nir_widen_16bit_varyings_mode mode = 0;

         if (nir->info.stage == MESA_SHADER_FRAGMENT) {
            mode = HK_NIR_WIDEN_16BIT_INPUTS;
         } else if (nir->info.stage == MESA_SHADER_MESH) {
            /* Native mesh output reaches the same 32-bit UVS words and
             * fragment interpolation as a vertex stage. */
            mode = HK_NIR_WIDEN_16BIT_OUTPUTS;
         } else if (nir->info.stage == MESA_SHADER_VERTEX ||
                    nir->info.stage == MESA_SHADER_TESS_EVAL ||
                    nir->info.stage == MESA_SHADER_GEOMETRY) {
            const VkShaderStageFlags poly_stages =
               VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
               VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT |
               VK_SHADER_STAGE_GEOMETRY_BIT;

            /* Poly interfaces are raw words in memory. Only the final
             * pre-raster interface needs numeric widening for G17P's 32-bit
             * UVS/interpolation path. */
            if (!(info->next_stage_mask & poly_stages))
               mode = HK_NIR_WIDEN_16BIT_OUTPUTS;
         }

         if (mode)
            NIR_PASS(_, nir, hk_nir_widen_16bit_varyings, mode);
      }

      if (nir->xfb_info) {
         /* Fold constant offset srcs for IO. */
         NIR_PASS(_, nir, nir_opt_constant_folding);

         nir_io_add_intrinsic_xfb_info(nir);
      }

      if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip))
         NIR_PASS(_, nir, nir_shader_intrinsics_pass, hk_widen_integer_io,
                  nir_metadata_control_flow, NULL);

      shaders[i] = nir;
   }

   /* A complete raster pipeline without a fragment shader cannot consume
    * user varyings.  Remove them before AGX assigns the unlinked UVS layout,
    * while preserving fixed-function outputs such as position and any XFB
    * writes.  Besides avoiding pointless vertex work, this keeps the G17P
    * backend on its native immediate-index UVSW form; the dynamic UVS index
    * is only needed when a later fragment shader may still choose the
    * interpolation-class ordering.
    *
    * Do not apply this to pipeline libraries: their pre-raster shader can be
    * linked with a fragment library later. */
   if (agx_native_graphics(&dev->dev) && shader_count > 0 &&
       hk_pipeline_has_no_fragment_shader(state)) {
      nir_shader *last = shaders[shader_count - 1];
      if (last->info.stage <= MESA_SHADER_GEOMETRY)
         NIR_PASS(_, last, nir_remove_outputs, MESA_SHADER_FRAGMENT,
                  BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32), 0);
   }

   /* The same holds for an M5 mesh shader, and there it also decides the
    * output form: agx_nir_lower_m5_mesh gives every primitive private
    * vertices when a per-primitive user word is written
    * (agx_m5_mesh_unwelds), and once corners * max_primitives exceeds the
    * 256 vertices its 8-bit index bytes can name it carries the words on
    * provoking vertices instead (mesh_layout_provoking), with a serial pass
    * per workgroup. A fragmentless triangle mesh declaring max_primitives=256
    * with a per-primitive colour (dEQP-VK.mesh_shader.ext.smoke.monolithic.
    * mesh_shader_triangle_rasterization_disabled) needs neither for a word
    * nothing reads. Layer, viewport, primitive ID, clip and cull
    * words stay: they are not in VAR0..VAR31. A shader that reads its
    * outputs back is left alone (the lowering refuses those regardless). */
   if (agx_chip_g17_m5(dev->dev.chip) && shader_count > 0 &&
       hk_pipeline_has_no_fragment_shader(state)) {
      nir_shader *last = shaders[shader_count - 1];
      if (last->info.stage == MESA_SHADER_MESH) {
         nir_shader_gather_info(last, nir_shader_get_entrypoint(last));
         if (!last->info.outputs_read)
            NIR_PASS(_, last, nir_remove_outputs, MESA_SHADER_FRAGMENT,
                     BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32), 0);
      }
   }

   struct hk_fs_key color_key;
   hk_populate_fs_key(&color_key, state, dev->dev.chip);
   for (uint32_t i = 0; i < shader_count; i++) {
      if (shaders[i]->info.stage == MESA_SHADER_FRAGMENT)
         hk_prune_unused_fragment_colors(shaders[i], &color_key);
   }

   nir_opt_varyings_bulk(shaders, shader_count, true, UINT32_MAX, UINT32_MAX,
                         nir_opts, NULL);

   /* A fragment shader compiled together with its mesh shader reads the
    * per-primitive records with that mesh shader's record layout. */
   if (dev->dev.chip == AGX_CHIP_G17P && has_mesh) {
      nir_shader *mesh_nir = NULL, *fs_nir = NULL;
      for (uint32_t i = 0; i < shader_count; i++) {
         if (shaders[i]->info.stage == MESA_SHADER_MESH)
            mesh_nir = shaders[i];
         else if (shaders[i]->info.stage == MESA_SHADER_FRAGMENT)
            fs_nir = shaders[i];
      }
      struct agx_m5_mesh_varyings mesh_varyings;
      if (mesh_nir && fs_nir &&
          agx_gen1_mesh_gather_varyings(mesh_nir, &mesh_varyings))
         agx_nir_lower_gen1_mesh_fragment_layout(fs_nir, &mesh_varyings);
   }

   /* Only a complete pipeline known to have no fragment shader can fix an
    * empty interpolation contract. A pre-raster library or shader object
    * must retain draw-time UVS indices for its eventual fragment consumer. */
   if (agx_native_graphics(&dev->dev) && shader_count > 0 &&
       hk_pipeline_has_no_fragment_shader(state)) {
      nir_shader *last = shaders[shader_count - 1];
      if (last->info.stage <= MESA_SHADER_GEOMETRY) {
         last->info.known_interpolation_qualifiers = true;
         last->info.linear_varyings = 0;
         last->info.perspective_varyings = 0;
      }
   }

   /* A vertex shader compiled together with its fragment shader writes no
    * default layer when that fragment shader never reads gl_Layer: nothing
    * else consumes the output, and its UVS words would be stored for every
    * vertex. A vertex shader compiled alone keeps it for whichever fragment
    * shader it meets. */
   bool fs_reads_layer = true, has_fragment = false;
   for (uint32_t i = 0; i < shader_count; i++) {
      if (shaders[i]->info.stage == MESA_SHADER_FRAGMENT) {
         has_fragment = true;
         nir_shader_gather_info(shaders[i],
                                nir_shader_get_entrypoint(shaders[i]));
         fs_reads_layer =
            (shaders[i]->info.inputs_read & VARYING_BIT_LAYER) ||
            BITSET_TEST(shaders[i]->info.system_values_read,
                        SYSTEM_VALUE_LAYER_ID);
      }
   }

   /* An M5 mesh shader compiled without a fragment shader, alone or with
    * its task shader, writes a zero layer when it writes none
    * (hk_m5_mesh_default_layer()): a pre-rasterization library or an
    * unlinked shader object may meet a fragment shader that reads gl_Layer.
    * A complete pipeline without a fragment shader cannot. */
   const bool mesh_default_layer =
      dev->dev.chip == AGX_CHIP_G17S && !has_fragment &&
      !hk_pipeline_has_no_fragment_shader(state);

   for (uint32_t i = 0; i < shader_count; i++) {
      const bool no_default_layer =
         !fs_reads_layer && infos[i].stage == MESA_SHADER_VERTEX &&
         infos[i].next_stage_mask == VK_SHADER_STAGE_FRAGMENT_BIT;
      const bool no_grid_record =
         m5_no_grid_record && (infos[i].stage == MESA_SHADER_TASK ||
                               infos[i].stage == MESA_SHADER_MESH);
      VkResult result =
         hk_compile_shader_impl(dev, &infos[i], state, features, pAllocator,
                                NULL, NULL, no_default_layer,
                                mesh_default_layer &&
                                   infos[i].stage == MESA_SHADER_MESH,
                                vs_covers[i], no_grid_record,
                                (struct hk_api_shader **)&shaders_out[i]);
      if (result != VK_SUCCESS) {
         /* Clean up all the shaders before this point */
         for (uint32_t j = 0; j < i; j++)
            hk_api_shader_destroy(&dev->vk, shaders_out[j], pAllocator);

         /* Clean up all the NIR after this point */
         for (uint32_t j = i + 1; j < shader_count; j++)
            ralloc_free(infos[j].nir);

         /* Memset the output array */
         memset(shaders_out, 0, shader_count * sizeof(*shaders_out));

         return result;
      }
   }

   /* G17P and M5 link a fragment shader with its epilog into one program, a
    * full compile, and a vertex prolog is compiled on its first use. Do both
    * now rather than at the first draw on the recording thread: for the state
    * a complete pipeline fixes, or for the most common state when compiling
    * shader objects. */
   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
      struct hk_api_shader *api[MESA_SHADER_MESH + 1] = {NULL};
      for (uint32_t i = 0; i < shader_count; i++) {
         if (infos[i].stage <= MESA_SHADER_MESH)
            api[infos[i].stage] =
               container_of(shaders_out[i], struct hk_api_shader, vk);
      }

      struct hk_api_shader *vs = api[MESA_SHADER_VERTEX];
      struct hk_api_shader *tes = api[MESA_SHADER_TESS_EVAL];
      struct hk_api_shader *gs = api[MESA_SHADER_GEOMETRY];
      struct hk_api_shader *fs = api[MESA_SHADER_FRAGMENT];

      /* A pipeline runs the vertex shader as its hardware program unless
       * tessellation or geometry follow; a shader object may be drawn
       * either way. */
      if (vs) {
         const bool pre_raster_after_vs = tes || gs;
         hk_prelink_vertex(dev, vs, !state || !pre_raster_after_vs,
                           !state || pre_raster_after_vs);
      }

      if (fs && !state) {
         hk_prelink_object_fragment(dev, fs);
      } else if (fs && vs && !api[MESA_SHADER_TASK] && !api[MESA_SHADER_MESH]) {
         const struct hk_shader *hw_vs =
            gs    ? &gs->variants[HK_GS_VARIANT_RAST]
            : tes ? &tes->variants[HK_VS_VARIANT_HW]
                  : &vs->variants[HK_VS_VARIANT_HW];
         hk_prelink_pipeline_fragment(dev, hw_vs, fs, state);
      }
   }

   return VK_SUCCESS;
}

/* Per-variant marker in a serialized hk_api_shader (shader binary version
 * 25).  ABSENT replaces the size-based guess binary version 24 used, which
 * mistook a compiled zero-byte fragment main for a never-compiled variant. */
enum hk_serialized_variant {
   HK_SERIALIZED_COMPILED = 0,
   HK_SERIALIZED_ABSENT = 2,
};

static VkResult
hk_deserialize_shader(struct hk_device *dev, struct blob_reader *blob,
                      struct hk_shader *shader, bool absent,
                      bool cache_candidate)
{
   struct hk_shader_info info;
   blob_copy_bytes(blob, &info, sizeof(info));
   const uint32_t root_descriptor_size = dev->dev.chip == AGX_CHIP_G17P
      ? blob_read_uint32(blob) : 0;
   uint8_t mesh_primitive_comps[64] = {0};
   if (dev->dev.chip == AGX_CHIP_G17P && info.stage == MESA_SHADER_MESH)
      blob_copy_bytes(blob, mesh_primitive_comps, sizeof(mesh_primitive_comps));

   struct agx_shader_info b_info;
   blob_copy_bytes(blob, &b_info, sizeof(b_info));

   struct agx_vdm_state_update_program_info agx3_vdm_state_update;
   if (!hk_agx3_vdm_program_cache_deserialize(blob,
                                               &agx3_vdm_state_update))
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   const uint32_t code_size = blob_read_uint32(blob);
   const uint32_t data_size = blob_read_uint32(blob);
   if (blob->overrun)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   /* A shader with native ray queries runs against the device's ray-query
    * state, which a device enabling no ray-tracing feature does not create.
    * Cache keys keep such binaries apart; refuse one that arrives as an
    * application-supplied binary from a ray-tracing device. */
   if ((info.agx3_gen1_fragment_rq || b_info.uses_ray_traversal) &&
       hk_g17p_rt_enabled(&dev->dev) && !dev->gen1_native_rq)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   /* A variant that was never compiled (for instance the software
    * vertex variant of a hardware-only vertex shader) has no code. The
    * compile leaves it without link table, upload or packed counts; restore
    * exactly that, so a deserialized object equals the compiled one field
    * for field (tools/objdiff.py).
    *
    * The serializer marks such a variant explicitly (HK_SERIALIZED_ABSENT).
    * Empty code alone does not identify it: a compiled G17P fragment shader
    * whose main does nothing has code_size == 0 and data_size == 0, yet it
    * needs its link table, upload and packed state like any other compiled
    * variant.  Treating it as never compiled left it with neither a link
    * table nor a lazy one, so every draw failed hk_get_fast_linked() and
    * vkEndCommandBuffer returned VK_ERROR_FEATURE_NOT_PRESENT (zink: device
    * lost) whenever the shader came from a pipeline cache or binary. */
   if (absent) {
      if (code_size != 0 || data_size != 0)
         return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      shader->b.info = b_info;
      shader->info = info;
      shader->root_descriptor_size = root_descriptor_size;
      memcpy(shader->g17p_mesh_primitive_comps, mesh_primitive_comps,
             sizeof(mesh_primitive_comps));
      shader->agx3_vdm_state_update = agx3_vdm_state_update;
      shader->b.info.binary_size = 0;
      return VK_SUCCESS;
   }

   VkResult result = hk_init_link_ht(shader, info.stage);
   if (result != VK_SUCCESS)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* Cache templates may only own shareable CPU payloads. Decline before
    * copying/uploading an executable or eagerly linking it, so the caller
    * can deserialize once with the final object's allocation callbacks.
    * VK_NOT_READY is private to this probe and is consumed by the wrapper. */
   if (cache_candidate &&
       (b_info.has_preamble || b_info.rodata.size_16 || !shader->linked.ht))
      return VK_NOT_READY;

   shader->b.info = b_info;
   shader->info = info;
   shader->root_descriptor_size = root_descriptor_size;
   memcpy(shader->g17p_mesh_primitive_comps, mesh_primitive_comps,
          sizeof(mesh_primitive_comps));
   shader->agx3_vdm_state_update = agx3_vdm_state_update;
   shader->code_size = code_size;
   shader->data_size = data_size;
   shader->b.info.binary_size = code_size;

   if (code_size > 0) {
      shader->code_ptr = malloc(code_size);
      if (shader->code_ptr == NULL)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   if (data_size > 0) {
      shader->data_ptr = malloc(data_size);
      if (shader->data_ptr == NULL)
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   blob_copy_bytes(blob, (void *)shader->code_ptr, shader->code_size);
   blob_copy_bytes(blob, (void *)shader->data_ptr, shader->data_size);
   if (blob->overrun)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   shader->b.binary = (void *)shader->code_ptr;
   result = hk_upload_shader(dev, shader);
   if (result != VK_SUCCESS)
      return vk_error(dev, result);
   shader->compiled = true;
   return VK_SUCCESS;
}

static VkResult
hk_fs_sample_source_deserialize(struct hk_api_shader *obj,
                                struct blob_reader *blob)
{
   struct hk_fs_sample_source *src = calloc(1, sizeof(*src));
   if (!src)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   blob_copy_bytes(blob, &src->key, sizeof(src->key));
   src->flags = blob_read_uint32(blob);
   blob_copy_bytes(blob, &src->robustness, sizeof(src->robustness));
   src->features = blob_read_uint32(blob);
   src->set_layout_count = blob_read_uint32(blob);
   src->static_set_mask = blob_read_uint8(blob);
   src->nir_size = blob_read_uint32(blob);
   const void *nir = blob_read_bytes(blob, src->nir_size);
   src->nir = blob->overrun ? NULL : malloc(src->nir_size);
   if (!src->nir) {
      free(src);
      return blob->overrun ? VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT
                           : VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   memcpy(src->nir, nir, src->nir_size);
   simple_mtx_init(&src->lock, mtx_plain);

   hk_foreach_variant(obj, shader) {
      if (shader->compiled)
         src->ready_mask |= BITFIELD_BIT(shader - obj->variants);
   }
   obj->sample_source = src;
   return VK_SUCCESS;
}

static VkResult
hk_border_source_deserialize(struct hk_border_source **out,
                             struct blob_reader *blob)
{
   struct hk_border_source *src = calloc(1, sizeof(*src));
   if (!src)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   src->has_key = blob_read_uint8(blob);
   blob_copy_bytes(blob, &src->key, sizeof(src->key));
   src->flags = blob_read_uint32(blob);
   src->next_stage_mask = blob_read_uint32(blob);
   blob_copy_bytes(blob, &src->robustness, sizeof(src->robustness));
   src->features = blob_read_uint32(blob);
   src->set_layout_count = blob_read_uint32(blob);
   src->static_set_mask = blob_read_uint8(blob);
   src->nir_size = blob_read_uint32(blob);
   const void *nir = blob_read_bytes(blob, src->nir_size);
   src->nir = blob->overrun ? NULL : malloc(src->nir_size);
   if (!src->nir) {
      free(src);
      return blob->overrun ? VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT
                           : VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   memcpy(src->nir, nir, src->nir_size);
   simple_mtx_init(&src->lock, mtx_plain);
   *out = src;
   return VK_SUCCESS;
}

static VkResult
hk_deserialize_api_shader_impl(struct vk_device *vk_dev,
                               struct blob_reader *blob,
                               uint32_t binary_version,
                               const VkAllocationCallbacks *pAllocator,
                               bool cache_candidate,
                               struct vk_shader **shader_out)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);

   if (binary_version != hk_shader_binary_version(dev->dev.chip))
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   mesa_shader_stage stage = blob_read_uint8(blob);
   const unsigned variant_flags = blob_read_uint8(blob);
   const bool sample_variants = variant_flags & 1;
   const bool m5_sample_variants = variant_flags & 32;
   const bool has_source = variant_flags & 2;
   const bool has_mono = variant_flags & 4;
   const bool has_border_source = variant_flags & 8;
   const bool border_emulating = variant_flags & 16;
   const bool has_mesh_source = variant_flags & 64;
   /* Compiled with or without the depth clip control lowering. With it, the
    * shader is right on any device that has the feature; without it, only on
    * one that did not enable it. */
   const bool clip_z = variant_flags & 128;
   const unsigned variant_flags2 = blob_read_uint8(blob);
   const bool has_fbl_source = variant_flags2 & 1;
   /* Graphics, compute, task and mesh stages. */
   if (blob->overrun || stage >= MESA_SHADER_MESH_STAGES ||
       (clip_z ? !dev->vk.physical->supported_features.depthClipControl
               : dev->vk.enabled_features.depthClipControl) ||
       (has_mesh_source && (stage != MESA_SHADER_FRAGMENT ||
                            dev->dev.chip != AGX_CHIP_G17P)) ||
       (variant_flags2 & ~1u) ||
       (has_fbl_source && stage != MESA_SHADER_FRAGMENT) ||
       (has_source && !sample_variants) ||
       (m5_sample_variants &&
        (!sample_variants || !agx_chip_g17_m5(dev->dev.chip))) ||
       (sample_variants &&
        (stage != MESA_SHADER_FRAGMENT ||
         (agx_chip_g17_m5(dev->dev.chip) ? !m5_sample_variants
                                        : !agx_native_graphics(&dev->dev)))) ||
       (has_mono && (stage != MESA_SHADER_FRAGMENT ||
                    (!agx_native_graphics(&dev->dev) &&
                     !agx_chip_g17_m5(dev->dev.chip)))))
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   /* Compiled with a custom-border emulation that this GPU, with no sampler
    * needing it, leaves out: refuse it, so the caller compiles the shader
    * without it (a cache entry is replaced, not reused). */
   if (border_emulating &&
       !p_atomic_read(&hk_device_physical(dev)->border_emulation))
      return VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT;

   const unsigned sample_variant_count = sample_variants
      ? (m5_sample_variants ? 4 : 3) : 0;
   size_t size = hk_api_shader_size(stage, sample_variant_count);

   struct hk_api_shader *obj =
      vk_shader_zalloc(&dev->vk, &hk_shader_ops, stage, pAllocator, size);

   if (obj == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   obj->g17p_sample_variants = sample_variant_count;
   obj->border_emulating = border_emulating;
   obj->clip_m1_1 = clip_z;

   hk_foreach_variant(obj, shader) {
      VkResult result;
      const uint8_t marker = blob_read_uint8(blob);
      if (marker == HK_SERIALIZED_COMPILED ||
          marker == HK_SERIALIZED_ABSENT) {
         result = hk_deserialize_shader(dev, blob, shader,
                                        marker == HK_SERIALIZED_ABSENT,
                                        cache_candidate);
      } else {
         result = vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      }

      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return result;
      }
   }

   /* Restore what the compile set on the composed passthrough object. */
   if (stage == MESA_SHADER_MESH && agx_chip_g17_m5(dev->dev.chip)) {
      struct hk_shader *object = &obj->variants[HK_MESH_VARIANT_RAST];
      if (object->compiled && object->only_linked)
         hk_m5_mesh_passthrough_linked(object);
   }

   if (has_source) {
      VkResult result = hk_fs_sample_source_deserialize(obj, blob);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_error(dev, result);
      }
   }

   if (has_mono) {
      hk_foreach_variant(obj, shader) {
         const uint32_t mono_size = blob_read_uint32(blob);
         if (!mono_size)
            continue;
         const void *mono_nir = blob_read_bytes(blob, mono_size);
         const void *mono_key =
            blob_read_bytes(blob, sizeof(shader->g17p_mono_key));
         shader->g17p_mono_nir = blob->overrun ? NULL : malloc(mono_size);
         if (!shader->g17p_mono_nir) {
            const VkResult result = blob->overrun
               ? VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT
               : VK_ERROR_OUT_OF_HOST_MEMORY;
            hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
            return vk_error(dev, result);
         }
         memcpy(shader->g17p_mono_nir, mono_nir, mono_size);
         shader->g17p_mono_nir_size = mono_size;
         memcpy(&shader->g17p_mono_key, mono_key,
                sizeof(shader->g17p_mono_key));
      }
   }

   if (has_border_source) {
      VkResult result = hk_border_source_deserialize(&obj->border_source, blob);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_error(dev, result);
      }
   }

   if (has_mesh_source) {
      VkResult result = hk_border_source_deserialize(&obj->mesh_source, blob);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_error(dev, result);
      }
   }

   if (has_fbl_source) {
      VkResult result = hk_border_source_deserialize(&obj->fbl_source, blob);
      if (result != VK_SUCCESS) {
         hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
         return vk_error(dev, result);
      }
   }

   VkResult result = hk_shader_prepare_device_for_stage(dev, stage);
   if (result != VK_SUCCESS) {
      hk_api_shader_destroy(&dev->vk, &obj->vk, pAllocator);
      return result;
   }

   *shader_out = &obj->vk;
   return VK_SUCCESS;
}

/* A copy of a source without its twins. */
static struct hk_border_source *
hk_border_source_clone(const struct hk_border_source *from)
{
   struct hk_border_source *src = calloc(1, sizeof(*src));
   void *nir = src ? malloc(from->nir_size) : NULL;
   if (!nir) {
      free(src);
      return NULL;
   }
   *src = *from;
   memcpy(nir, from->nir, from->nir_size);
   src->nir = nir;
   src->twin = NULL;
   simple_mtx_init(&src->lock, mtx_plain);
   return src;
}

static struct hk_shader_binary_cache_entry *
hk_shader_binary_cache_find_locked(struct hk_device *dev,
                                   const void *binary, size_t binary_size,
                                   uint32_t binary_version)
{
   for (unsigned i = 0; i < HK_SHADER_BINARY_CACHE_MAX; ++i) {
      struct hk_shader_binary_cache_entry *entry =
         dev->shader_binary_cache.entries[i];

      if (entry && entry->binary_version == binary_version &&
          entry->binary_size == binary_size &&
          memcmp(entry->binary, binary, binary_size) == 0)
         return entry;
   }

   return NULL;
}

static bool
hk_shader_binary_payload_cacheable(struct hk_api_shader *obj)
{
   /* A cached payload is immutable CPU code/data only.  Hardware allocations
    * and eager linked variants keep their existing per-object ownership. */
   hk_foreach_variant(obj, shader) {
      /* A never-compiled variant carries nothing to share. */
      if (!shader->compiled)
         continue;
      if (shader->bo || shader->only_linked || !shader->linked.ht)
         return false;
   }

   return true;
}

static VkResult
hk_shader_binary_cache_clone(struct hk_device *dev,
                             struct hk_shader_binary_cache_entry *entry,
                             const VkAllocationCallbacks *pAllocator,
                             struct vk_shader **shader_out)
{
   const struct hk_api_shader *src_obj = entry->shader;
   const size_t size = hk_api_shader_size(src_obj->vk.stage,
                                          src_obj->g17p_sample_variants);
   struct hk_api_shader *obj =
      vk_shader_zalloc(&dev->vk, &hk_shader_ops, src_obj->vk.stage,
                       pAllocator, size);
   if (!obj) {
      hk_shader_binary_cache_entry_unref(dev, entry);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   obj->is_passthrough = src_obj->is_passthrough;
   obj->g17p_sample_variants = src_obj->g17p_sample_variants;
   obj->border_emulating = src_obj->border_emulating;
   obj->clip_m1_1 = src_obj->clip_m1_1;
   obj->binary_cache_entry = entry;

   if (src_obj->sample_source) {
      const struct hk_fs_sample_source *from = src_obj->sample_source;
      struct hk_fs_sample_source *src = calloc(1, sizeof(*src));
      void *nir = src ? malloc(from->nir_size) : NULL;
      if (!nir) {
         free(src);
         vk_shader_free(&dev->vk, pAllocator, &obj->vk);
         hk_shader_binary_cache_entry_unref(dev, entry);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      *src = *from;
      memcpy(nir, from->nir, from->nir_size);
      src->nir = nir;
      src->owned_mask = 0;
      src->ready_mask = 0;
      simple_mtx_init(&src->lock, mtx_plain);
      for (unsigned i = 0; i < hk_api_num_variants(src_obj); ++i) {
         if (src_obj->variants[i].compiled)
            src->ready_mask |= BITFIELD_BIT(i);
      }
      obj->sample_source = src;
   }

   if (src_obj->border_source) {
      obj->border_source = hk_border_source_clone(src_obj->border_source);
      if (!obj->border_source) {
         hk_fs_sample_source_destroy(obj->sample_source);
         vk_shader_free(&dev->vk, pAllocator, &obj->vk);
         hk_shader_binary_cache_entry_unref(dev, entry);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
   }

   if (src_obj->mesh_source) {
      obj->mesh_source = hk_border_source_clone(src_obj->mesh_source);
      if (!obj->mesh_source) {
         hk_border_source_destroy(dev, obj->border_source);
         hk_fs_sample_source_destroy(obj->sample_source);
         vk_shader_free(&dev->vk, pAllocator, &obj->vk);
         hk_shader_binary_cache_entry_unref(dev, entry);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
   }

   if (src_obj->fbl_source) {
      obj->fbl_source = hk_border_source_clone(src_obj->fbl_source);
      if (!obj->fbl_source) {
         hk_border_source_destroy(dev, obj->mesh_source);
         hk_border_source_destroy(dev, obj->border_source);
         hk_fs_sample_source_destroy(obj->sample_source);
         vk_shader_free(&dev->vk, pAllocator, &obj->vk);
         hk_shader_binary_cache_entry_unref(dev, entry);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
   }

   for (unsigned i = 0; i < hk_api_num_variants(src_obj); ++i) {
      struct hk_shader *dst = &obj->variants[i];
      const struct hk_shader *src = &src_obj->variants[i];

      *dst = *src;
      memset(&dst->linked, 0, sizeof(dst->linked));
      dst->only_linked = NULL;
      dst->uvs_linked = NULL;
      dst->bo = NULL;
      /* The clone owns its own copy of the inline-blend NIR. */
      dst->g17p_mono_nir = NULL;
      dst->g17p_mono_nir_size = 0;
      if (src->g17p_mono_nir) {
         dst->g17p_mono_nir = malloc(src->g17p_mono_nir_size);
         if (dst->g17p_mono_nir) {
            memcpy(dst->g17p_mono_nir, src->g17p_mono_nir,
                   src->g17p_mono_nir_size);
            dst->g17p_mono_nir_size = src->g17p_mono_nir_size;
         }
      }
      simple_mtx_init(&dst->linked.lock, mtx_plain);
      /* Every compiled variant links on demand, including one whose main
       * compiled to zero bytes. */
      dst->linked.lazy_ht = src->compiled;
   }

   *shader_out = &obj->vk;
   return VK_SUCCESS;
}

static VkResult
hk_deserialize_api_shader(struct vk_device *vk_dev, struct blob_reader *blob,
                          uint32_t binary_version,
                          const VkAllocationCallbacks *pAllocator,
                          struct vk_shader **shader_out)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);
   const size_t bytes = (size_t)(blob->end - blob->current);
   const void *binary = blob->current;
   const size_t binary_size = bytes;

   /* This callback is used by both standalone shader-binary import and the
    * pipeline cache.  Only the former has a vk_shader_bin_header immediately
    * before blob->current.  Compare the exact driver payload for cache hits;
    * the common Vulkan layer has already validated standalone binaries with
    * BLAKE3, and a second hash here would only repeat work.  This also keeps
    * pipeline-cache readers from accessing memory before their allocation. */

   simple_mtx_lock(&dev->shader_binary_cache.lock);
   struct hk_shader_binary_cache_entry *entry =
      hk_shader_binary_cache_find_locked(dev, binary, binary_size,
                                         binary_version);
   if (entry)
      entry->refcount++;
   bool cache_has_room = false;
   if (!entry) {
      for (unsigned i = 0; i < HK_SHADER_BINARY_CACHE_MAX; ++i) {
         if (!dev->shader_binary_cache.entries[i]) {
            cache_has_room = true;
            break;
         }
      }
   }
   simple_mtx_unlock(&dev->shader_binary_cache.lock);

   /* Entries remain installed until device destruction. A full-cache miss
    * cannot gain a slot, so decode directly with the caller's allocator.
    * Existing hits still share their template; racing insertions below keep
    * their locked recheck when a slot was available at this first lookup. */
   if (!entry && !cache_has_room)
      return hk_deserialize_api_shader_impl(
         vk_dev, blob, binary_version, pAllocator, false, shader_out);

   VkResult result;
   if (entry) {
      blob->current = blob->end;
      result = hk_shader_binary_cache_clone(dev, entry, pAllocator, shader_out);
   } else {
      struct blob_reader template_blob = *blob;
      struct vk_shader *template_vk = NULL;
      result = hk_deserialize_api_shader_impl(
         vk_dev, &template_blob, binary_version, NULL, true, &template_vk);

      if (result == VK_NOT_READY) {
         /* The probe released its partial CPU-only object. Re-read the
          * original blob with the final allocator, including the explicit
          * device callbacks used by pipeline-cache imports. */
         return hk_deserialize_api_shader_impl(
            vk_dev, blob, binary_version, pAllocator, false, shader_out);
      }

      struct hk_api_shader *template_obj = result == VK_SUCCESS
         ? container_of(template_vk, struct hk_api_shader, vk)
         : NULL;
      if (result == VK_SUCCESS &&
          hk_shader_binary_payload_cacheable(template_obj)) {
         struct hk_shader_binary_cache_entry *new_entry =
            calloc(1, sizeof(*new_entry));
         if (new_entry)
            new_entry->binary = malloc(binary_size);
         if (new_entry && !new_entry->binary) {
            free(new_entry);
            new_entry = NULL;
         }

         if (new_entry && new_entry->binary) {
            memcpy(new_entry->binary, binary, binary_size);
            new_entry->binary_size = binary_size;
            new_entry->binary_version = binary_version;
            new_entry->shader = template_obj;
            new_entry->refcount = 2; /* cache plus the wrapper below */

            simple_mtx_lock(&dev->shader_binary_cache.lock);
            entry = hk_shader_binary_cache_find_locked(
               dev, binary, binary_size, binary_version);
            if (entry) {
               entry->refcount++;
            } else {
               for (unsigned i = 0; i < HK_SHADER_BINARY_CACHE_MAX; ++i) {
                  if (!dev->shader_binary_cache.entries[i]) {
                     dev->shader_binary_cache.entries[i] = new_entry;
                     entry = new_entry;
                     new_entry = NULL;
                     break;
                  }
               }
            }
            simple_mtx_unlock(&dev->shader_binary_cache.lock);
         }

         if (new_entry) {
            hk_shader_binary_cache_entry_destroy(dev, new_entry);
            new_entry = NULL;
            template_obj = NULL;
         } else if (entry && entry->shader != template_obj) {
            hk_api_shader_destroy(vk_dev, &template_obj->vk, NULL);
            template_obj = NULL;
         }

         if (entry) {
            blob->current = blob->end;
            result = hk_shader_binary_cache_clone(
               dev, entry, pAllocator, shader_out);
         } else {
            if (template_obj)
               hk_api_shader_destroy(vk_dev, &template_obj->vk, NULL);
            result = hk_deserialize_api_shader_impl(
               vk_dev, blob, binary_version, pAllocator, false, shader_out);
         }
      } else if (result == VK_SUCCESS) {
         hk_api_shader_destroy(vk_dev, template_vk, NULL);
         result = hk_deserialize_api_shader_impl(
            vk_dev, blob, binary_version, pAllocator, false, shader_out);
      }
   }

   return result;
}

static void
hk_shader_serialize(struct vk_device *vk_dev, const struct hk_shader *shader,
                    struct blob *blob)
{
   struct hk_device *dev = container_of(vk_dev, struct hk_device, vk);
   blob_write_bytes(blob, &shader->info, sizeof(shader->info));
   if (dev->dev.chip == AGX_CHIP_G17P)
      blob_write_uint32(blob, shader->root_descriptor_size);
   if (dev->dev.chip == AGX_CHIP_G17P &&
       shader->info.stage == MESA_SHADER_MESH)
      blob_write_bytes(blob, shader->g17p_mesh_primitive_comps,
                       sizeof(shader->g17p_mesh_primitive_comps));
   blob_write_bytes(blob, &shader->b.info, sizeof(shader->b.info));
   if (!hk_agx3_vdm_program_cache_serialize(
          blob, &shader->agx3_vdm_state_update))
      return;

   blob_write_uint32(blob, shader->code_size);
   blob_write_uint32(blob, shader->data_size);
   blob_write_bytes(blob, shader->code_ptr, shader->code_size);
   blob_write_bytes(blob, shader->data_ptr, shader->data_size);
}

static bool
hk_api_shader_serialize(struct vk_device *vk_dev,
                        const struct vk_shader *vk_shader, struct blob *blob)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   const struct hk_fs_sample_source *src = obj->sample_source;
   bool mono = false;
   hk_foreach_variant(obj, shader)
      mono |= shader->g17p_mono_nir != NULL;
   const struct hk_border_source *border = obj->border_source;
   blob_write_uint8(blob, vk_shader->stage);
   blob_write_uint8(blob, (obj->g17p_sample_variants ? 1 : 0) | (src ? 2 : 0) |
                             (mono ? 4 : 0) | (border ? 8 : 0) |
                             (obj->border_emulating ? 16 : 0) |
                             (obj->g17p_sample_variants == 4 ? 32 : 0) |
                             (obj->mesh_source ? 64 : 0) |
                             (obj->clip_m1_1 ? 128 : 0));
   blob_write_uint8(blob, obj->fbl_source ? 1 : 0);

   hk_foreach_variant(obj, shader) {
      const unsigned i = shader - obj->variants;
      const bool ready =
         src ? (p_atomic_read(&src->ready_mask) & BITFIELD_BIT(i))
             : shader->compiled;
      blob_write_uint8(blob, ready ? HK_SERIALIZED_COMPILED
                                   : HK_SERIALIZED_ABSENT);
      if (ready) {
         hk_shader_serialize(vk_dev, shader, blob);
      } else {
         /* Possibly being built right now: describe it as never compiled. */
         const struct hk_shader absent = {0};
         hk_shader_serialize(vk_dev, src ? &absent : shader, blob);
      }
   }

   /* Multisample variants not built yet keep their source. */
   if (src) {
      blob_write_bytes(blob, &src->key, sizeof(src->key));
      blob_write_uint32(blob, src->flags);
      blob_write_bytes(blob, &src->robustness, sizeof(src->robustness));
      blob_write_uint32(blob, src->features);
      blob_write_uint32(blob, src->set_layout_count);
      blob_write_uint8(blob, src->static_set_mask);
      blob_write_uint32(blob, src->nir_size);
      blob_write_bytes(blob, src->nir, src->nir_size);
   }

   /* The NIR an inline-blend fragment program is rebuilt from at fast-link,
    * so a shader that comes from a cache or a binary blends inline exactly
    * like a freshly compiled one. */
   if (mono) {
      hk_foreach_variant(obj, shader) {
         const unsigned i = shader - obj->variants;
         const bool ready =
            src ? (p_atomic_read(&src->ready_mask) & BITFIELD_BIT(i))
                : shader->compiled;
         const uint32_t size =
            ready && shader->g17p_mono_nir ? shader->g17p_mono_nir_size : 0;
         blob_write_uint32(blob, size);
         if (size) {
            blob_write_bytes(blob, shader->g17p_mono_nir, size);
            blob_write_bytes(blob, &shader->g17p_mono_key,
                             sizeof(shader->g17p_mono_key));
         }
      }
   }

   /* A shader compiled without custom-border emulation keeps what its
    * emulating twin compiles from, a G17P fragment shader compiled without
    * its mesh shader what its mesh twins compile from, and a fragment shader
    * with a dynamic feedback loop what its zs_self_dep twin compiles from. */
   const struct hk_border_source *sources[] = {border, obj->mesh_source,
                                               obj->fbl_source};
   for (unsigned i = 0; i < ARRAY_SIZE(sources); ++i) {
      if (!sources[i])
         continue;
      blob_write_uint8(blob, sources[i]->has_key);
      blob_write_bytes(blob, &sources[i]->key, sizeof(sources[i]->key));
      blob_write_uint32(blob, sources[i]->flags);
      blob_write_uint32(blob, sources[i]->next_stage_mask);
      blob_write_bytes(blob, &sources[i]->robustness,
                       sizeof(sources[i]->robustness));
      blob_write_uint32(blob, sources[i]->features);
      blob_write_uint32(blob, sources[i]->set_layout_count);
      blob_write_uint8(blob, sources[i]->static_set_mask);
      blob_write_uint32(blob, sources[i]->nir_size);
      blob_write_bytes(blob, sources[i]->nir, sources[i]->nir_size);
   }

   return !blob->out_of_memory;
}

static VkResult
hk_shader_get_executable_properties(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t *executable_count, VkPipelineExecutablePropertiesKHR *properties)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutablePropertiesKHR, out, properties,
                          executable_count);

   vk_outarray_append_typed(VkPipelineExecutablePropertiesKHR, &out, props)
   {
      props->stages = mesa_to_vk_shader_stage(obj->vk.stage);
      props->subgroupSize = 32;
      VK_COPY_STR(props->name, _mesa_shader_stage_to_string(obj->vk.stage));
      VK_PRINT_STR(props->description, "%s shader",
                   _mesa_shader_stage_to_string(obj->vk.stage));
   }

   return vk_outarray_status(&out);
}

static VkResult
hk_shader_get_executable_statistics(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *statistic_count,
   VkPipelineExecutableStatisticKHR *statistics)
{
   struct hk_api_shader *obj =
      container_of(vk_shader, struct hk_api_shader, vk);

   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableStatisticKHR, out, statistics,
                          statistic_count);

   assert(executable_index == 0);

   /* TODO: find a sane way to report multiple variants and have that play nice
    * with zink.
    */
   struct hk_shader *shader = hk_any_variant(obj);
   vk_add_agx2_stats(out, &shader->b.info.stats);
   return vk_outarray_status(&out);
}

static bool
write_ir_text(VkPipelineExecutableInternalRepresentationKHR *ir,
              const char *data)
{
   ir->isText = VK_TRUE;

   size_t data_len = strlen(data) + 1;

   if (ir->pData == NULL) {
      ir->dataSize = data_len;
      return true;
   }

   strncpy(ir->pData, data, ir->dataSize);
   if (ir->dataSize < data_len)
      return false;

   ir->dataSize = data_len;
   return true;
}

static VkResult
hk_shader_get_executable_internal_representations(
   UNUSED struct vk_device *device, const struct vk_shader *vk_shader,
   uint32_t executable_index, uint32_t *internal_representation_count,
   VkPipelineExecutableInternalRepresentationKHR *internal_representations)
{
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableInternalRepresentationKHR, out,
                          internal_representations,
                          internal_representation_count);
   bool incomplete_text = false;

   assert(executable_index == 0);

   /* TODO */
#if 0
   vk_outarray_append_typed(VkPipelineExecutableInternalRepresentationKHR, &out, ir) {
      VK_COPY_STR(ir->name, "AGX assembly");
      VK_COPY_STR(ir->description, "AGX assembly");
      if (!write_ir_text(ir, TODO))
         incomplete_text = true;
   }
#endif

   return incomplete_text ? VK_INCOMPLETE : vk_outarray_status(&out);
}

static const struct vk_shader_ops hk_shader_ops = {
   .destroy = hk_api_shader_destroy,
   .serialize = hk_api_shader_serialize,
   .get_executable_properties = hk_shader_get_executable_properties,
   .get_executable_statistics = hk_shader_get_executable_statistics,
   .get_executable_internal_representations =
      hk_shader_get_executable_internal_representations,
};

const struct vk_device_shader_ops hk_device_shader_ops = {
   .get_nir_options = hk_get_nir_options,
   .get_spirv_options = hk_get_spirv_options,
   .preprocess_nir = hk_preprocess_nir,
   .hash_state = hk_hash_graphics_state,
   .compile = hk_compile_shaders,
   .deserialize = hk_deserialize_api_shader,
   .cmd_set_dynamic_graphics_state = vk_cmd_set_dynamic_graphics_state,
   .cmd_bind_shaders = hk_cmd_bind_shaders,
};

/*
 * A refused graphics pipeline is invisible.  vkCreateGraphicsPipelines hands
 * the application a bare VkResult; the application drops every draw that
 * would have used the pipeline, the frame comes out blank, and nothing in the
 * log says which pipeline died.  That is exactly how one unsupported pipeline
 * shape hides inside a page whose other pipelines render.
 *
 * Only one refusal inside this driver is reachable from pipeline creation --
 * the AGX3 packer in hk_compile_nir(), which names the stage it could not
 * encode -- so a -8 with no packer line is not this driver refusing at all.
 * Making that distinction requires the driver to say something on *every*
 * failure, so wrap the common entrypoint and print the refused pipeline's
 * shape: its index, its stages and their SPIR-V sizes, its create flags and
 * graphics-pipeline-library subset, its dynamic state, and the fixed-function
 * state a chip gate can plausibly reject.  Nothing is printed on success.
 */
struct hk_pipeline_strbuf {
   char data[512];
   size_t len;
};

static void PRINTFLIKE(2, 3)
hk_pipeline_strbuf_appendf(struct hk_pipeline_strbuf *b, const char *fmt, ...)
{
   if (b->len + 1 >= sizeof(b->data))
      return;

   va_list ap;
   va_start(ap, fmt);
   int n = vsnprintf(b->data + b->len, sizeof(b->data) - b->len, fmt, ap);
   va_end(ap);

   if (n > 0)
      b->len = MIN2(b->len + (size_t)n, sizeof(b->data) - 1);
}

static const char *
hk_shader_stage_bit_name(VkShaderStageFlagBits stage)
{
   switch (stage) {
   case VK_SHADER_STAGE_VERTEX_BIT:
      return "vs";
   case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
      return "tcs";
   case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
      return "tes";
   case VK_SHADER_STAGE_GEOMETRY_BIT:
      return "gs";
   case VK_SHADER_STAGE_FRAGMENT_BIT:
      return "fs";
   case VK_SHADER_STAGE_TASK_BIT_EXT:
      return "task";
   case VK_SHADER_STAGE_MESH_BIT_EXT:
      return "mesh";
   default:
      return "unknown";
   }
}

static size_t
hk_pipeline_stage_spirv_size(const VkPipelineShaderStageCreateInfo *stage)
{
   if (stage->module != VK_NULL_HANDLE)
      return vk_shader_module_from_handle(stage->module)->size;

   /* VK_KHR_maintenance5 lets the SPIR-V ride in the stage's pNext with no
    * VkShaderModule at all. */
   const VkShaderModuleCreateInfo *inline_module =
      vk_find_struct_const(stage->pNext, SHADER_MODULE_CREATE_INFO);

   return inline_module ? inline_module->codeSize : 0;
}

static void
hk_log_refused_graphics_pipeline(const VkGraphicsPipelineCreateInfo *info,
                                 uint32_t index, uint32_t count,
                                 VkResult result)
{
   mesa_loge("hk: vkCreateGraphicsPipelines refused pipeline %u of %u: %s",
             index, count, vk_Result_to_str(result));

   const VkPipelineCreateFlags2KHR flags =
      vk_graphics_pipeline_create_flags(info);
   const VkGraphicsPipelineLibraryCreateInfoEXT *gpl = vk_find_struct_const(
      info->pNext, GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT);
   const VkPipelineLibraryCreateInfoKHR *libs =
      vk_find_struct_const(info->pNext, PIPELINE_LIBRARY_CREATE_INFO_KHR);

   mesa_loge("hk:   flags=0x%" PRIx64 " gpl_subset=0x%x linked_libraries=%u",
             (uint64_t)flags, gpl ? gpl->flags : 0u,
             libs ? libs->libraryCount : 0u);

   for (uint32_t s = 0; s < info->stageCount; s++) {
      const VkPipelineShaderStageCreateInfo *stage = &info->pStages[s];

      mesa_loge("hk:   stage %-4s entry \"%s\" spirv=%zu bytes flags=0x%x",
                hk_shader_stage_bit_name(stage->stage),
                stage->pName ? stage->pName : "(null)",
                hk_pipeline_stage_spirv_size(stage), stage->flags);
   }

   if (info->pDynamicState) {
      struct hk_pipeline_strbuf dyn = {0};

      for (uint32_t d = 0; d < info->pDynamicState->dynamicStateCount; d++) {
         hk_pipeline_strbuf_appendf(
            &dyn, "%s%s", d ? " " : "",
            vk_DynamicState_to_str(info->pDynamicState->pDynamicStates[d]));
      }

      mesa_loge("hk:   dynamic(%u): %s",
                info->pDynamicState->dynamicStateCount, dyn.data);
   } else {
      mesa_loge("hk:   dynamic(0): all state static");
   }

   if (info->pInputAssemblyState) {
      mesa_loge("hk:   ia: topology=%u restart=%u",
                info->pInputAssemblyState->topology,
                info->pInputAssemblyState->primitiveRestartEnable);
   }

   if (info->pVertexInputState) {
      mesa_loge("hk:   vi: bindings=%u attributes=%u",
                info->pVertexInputState->vertexBindingDescriptionCount,
                info->pVertexInputState->vertexAttributeDescriptionCount);
   }

   if (info->pTessellationState) {
      mesa_loge("hk:   tess: patch_control_points=%u",
                info->pTessellationState->patchControlPoints);
   }

   if (info->pViewportState) {
      mesa_loge("hk:   vp: viewports=%u scissors=%u",
                info->pViewportState->viewportCount,
                info->pViewportState->scissorCount);
   }

   if (info->pRasterizationState) {
      const VkPipelineRasterizationStateCreateInfo *rs =
         info->pRasterizationState;

      mesa_loge("hk:   rs: discard=%u polygon=%u cull=0x%x front=%u "
                "depth_clamp=%u depth_bias=%u line_width=%f",
                rs->rasterizerDiscardEnable, rs->polygonMode, rs->cullMode,
                rs->frontFace, rs->depthClampEnable, rs->depthBiasEnable,
                rs->lineWidth);
   }

   if (info->pMultisampleState) {
      const VkPipelineMultisampleStateCreateInfo *ms = info->pMultisampleState;

      mesa_loge("hk:   ms: samples=%u sample_shading=%u min=%f "
                "sample_mask=0x%x alpha_to_coverage=%u alpha_to_one=%u",
                ms->rasterizationSamples, ms->sampleShadingEnable,
                ms->minSampleShading,
                ms->pSampleMask ? ms->pSampleMask[0] : 0xffffffffu,
                ms->alphaToCoverageEnable, ms->alphaToOneEnable);
   }

   if (info->pDepthStencilState) {
      const VkPipelineDepthStencilStateCreateInfo *ds =
         info->pDepthStencilState;

      mesa_loge("hk:   ds: test=%u write=%u op=%u bounds=%u stencil=%u",
                ds->depthTestEnable, ds->depthWriteEnable, ds->depthCompareOp,
                ds->depthBoundsTestEnable, ds->stencilTestEnable);
   }

   if (info->pColorBlendState) {
      const VkPipelineColorBlendStateCreateInfo *cb = info->pColorBlendState;
      struct hk_pipeline_strbuf att = {0};

      for (uint32_t a = 0; a < cb->attachmentCount && cb->pAttachments; a++) {
         hk_pipeline_strbuf_appendf(&att, "%s[%u]=en%u/mask%x",
                                    a ? " " : "", a,
                                    cb->pAttachments[a].blendEnable,
                                    cb->pAttachments[a].colorWriteMask);
      }

      mesa_loge("hk:   cb: logic_op=%u/%u attachments=%u %s",
                cb->logicOpEnable, cb->logicOp, cb->attachmentCount, att.data);
   }

   const VkPipelineRenderingCreateInfo *rendering =
      vk_find_struct_const(info->pNext, PIPELINE_RENDERING_CREATE_INFO);

   if (rendering) {
      struct hk_pipeline_strbuf colors = {0};

      for (uint32_t a = 0;
           a < rendering->colorAttachmentCount &&
           rendering->pColorAttachmentFormats;
           a++) {
         hk_pipeline_strbuf_appendf(
            &colors, "%s%s", a ? " " : "",
            vk_Format_to_str(rendering->pColorAttachmentFormats[a]));
      }

      mesa_loge("hk:   rendering: view_mask=0x%x color=%u [%s] depth=%s "
                "stencil=%s",
                rendering->viewMask, rendering->colorAttachmentCount,
                colors.data, vk_Format_to_str(rendering->depthAttachmentFormat),
                vk_Format_to_str(rendering->stencilAttachmentFormat));
   } else {
      mesa_loge("hk:   render_pass=%s subpass=%u",
                info->renderPass != VK_NULL_HANDLE ? "yes" : "none",
                info->subpass);
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateGraphicsPipelines(VkDevice device, VkPipelineCache pipelineCache,
                           uint32_t createInfoCount,
                           const VkGraphicsPipelineCreateInfo *pCreateInfos,
                           const VkAllocationCallbacks *pAllocator,
                           VkPipeline *pPipelines)
{
   VkResult result = vk_common_CreateGraphicsPipelines(
      device, pipelineCache, createInfoCount, pCreateInfos, pAllocator,
      pPipelines);

   /* VK_PIPELINE_COMPILE_REQUIRED is the answer the application asked for
    * with VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT, not a
    * refusal. */
   if (result == VK_SUCCESS || result == VK_PIPELINE_COMPILE_REQUIRED ||
       createInfoCount == 0)
      return result;

   /* vk_common_CreateGraphicsPipelines() zeroes pPipelines up front and
    * returns on the first hard failure, so the first null handle names the
    * pipeline that was refused. */
   uint32_t failed = 0;
   while (failed + 1 < createInfoCount && pPipelines[failed] != VK_NULL_HANDLE)
      failed++;

   hk_log_refused_graphics_pipeline(&pCreateInfos[failed], failed,
                                    createInfoCount, result);
   return result;
}

/*
 * The compute entrypoint has the same blind spot, and a translation layer's
 * internal compute pipelines (format conversion, mip generation, blits) fail
 * the same way: the operation silently does nothing and the frame is wrong
 * somewhere else entirely.
 */
VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateComputePipelines(VkDevice device, VkPipelineCache pipelineCache,
                          uint32_t createInfoCount,
                          const VkComputePipelineCreateInfo *pCreateInfos,
                          const VkAllocationCallbacks *pAllocator,
                          VkPipeline *pPipelines)
{
   VkResult result = vk_common_CreateComputePipelines(
      device, pipelineCache, createInfoCount, pCreateInfos, pAllocator,
      pPipelines);

   if (result == VK_SUCCESS || result == VK_PIPELINE_COMPILE_REQUIRED ||
       createInfoCount == 0)
      return result;

   uint32_t failed = 0;
   while (failed + 1 < createInfoCount && pPipelines[failed] != VK_NULL_HANDLE)
      failed++;

   const VkComputePipelineCreateInfo *info = &pCreateInfos[failed];
   mesa_loge("hk: vkCreateComputePipelines refused pipeline %u of %u: %s",
             failed, createInfoCount, vk_Result_to_str(result));
   mesa_loge("hk:   flags=0x%x stage entry \"%s\" spirv=%zu bytes "
             "stage_flags=0x%x",
             info->flags, info->stage.pName ? info->stage.pName : "(null)",
             hk_pipeline_stage_spirv_size(&info->stage), info->stage.flags);
   return result;
}

struct hk_linked_shader *
hk_fast_link(struct hk_device *dev, bool fragment, struct hk_shader *main,
             struct agx_shader_part *prolog, struct agx_shader_part *epilog,
             unsigned nr_samples_shaded)
{
   return hk_fast_link_part(dev, fragment, main, NULL, prolog, epilog,
                            nr_samples_shaded, NULL);
}

struct hk_linked_shader *
hk_fast_link_vertex(struct hk_device *dev, struct hk_shader *main,
                    struct agx_shader_part *prolog, uint64_t flat,
                    uint64_t linear)
{
   uint8_t map[AGX3_UVS_RELOC_MAP_SIZE];
   const bool relocatable = main->info.uvs.relocatable;
   if (relocatable)
      agx_uvs_relocation_map(&main->info.uvs, flat, linear, map);

   return hk_fast_link_part(dev, false, main, NULL, prolog, NULL, 0,
                            relocatable ? map : NULL);
}

struct hk_uvs_fs_key {
   uint64_t flat, linear;
};
DERIVE_HASH_TABLE(hk_uvs_fs_key);

struct hk_linked_shader *
hk_only_linked_for_fs(struct hk_device *dev, struct hk_shader *shader,
                      uint64_t flat, uint64_t linear)
{
   const uint64_t written = shader->info.uvs.written;
   const struct hk_uvs_fs_key key = {flat & written, linear & written};

   /* only_linked itself places the varyings for no flat or linear input. */
   if (!shader->info.uvs.relocatable || (!key.flat && !key.linear))
      return shader->only_linked;

   struct hk_linked_shader *linked = NULL;
   simple_mtx_lock(&shader->linked.lock);
   if (!shader->uvs_linked)
      shader->uvs_linked = hk_uvs_fs_key_table_create(NULL);

   struct hash_entry *ent =
      shader->uvs_linked ? _mesa_hash_table_search(shader->uvs_linked, &key)
                         : NULL;
   if (ent) {
      linked = ent->data;
   } else if (shader->uvs_linked) {
      struct hk_uvs_fs_key *clone =
         ralloc_memdup(shader->uvs_linked, &key, sizeof(key));
      linked = clone ? hk_fast_link_vertex(dev, shader, NULL, key.flat,
                                           key.linear)
                     : NULL;
      if (linked)
         _mesa_hash_table_insert(shader->uvs_linked, clone, linked);
      else
         ralloc_free(clone);
   }
   simple_mtx_unlock(&shader->linked.lock);
   return linked;
}

/* Prove the complete new loader before exposing a specialized executable.
 * The only external inputs are the original root and the new program's own
 * constant pool; every imported uniform half must be covered by those ranges. */
static bool
hk_m5_fragment_variant_preflight(struct hk_device *dev,
                                 const struct hk_shader *s,
                                 const struct hk_linked_shader *linked,
                                 const struct agx_fs_epilog_key *key)
{
   if (!s->info.agx3_uses_root || s->info.set_count ||
       s->info.agx3_root_uniform != AGX_ABI_FUNI_ROOT ||
       s->b.info.agx3_uniform_dma_count)
      return false;

   struct agx3_ioto_uniform_range ranges[2];
   size_t count = 1;
   if (!hk_agx3_g17p_root_range(AGX_ABI_FUNI_ROOT, 0,
                                 dev->dev.shader_base + 0x10000, &ranges[0]))
      return false;
   if (s->b.info.rodata.size_16) {
      const struct agx_rodata *rodata = &s->b.info.rodata;
      if (!hk_agx3_g17p_rodata_range(rodata->base_uniform, rodata->size_16,
                                      hk_shader_code_va(s), rodata->offset,
                                      s->b.info.binary_size, &ranges[count]))
         return false;
      count++;
   }
   for (unsigned half = 0; half < AGX3_CPROG_INPUT_HALVES; ++half) {
      if (!(s->b.info.agx3_preamble_inputs.halves[half / 64] & BITFIELD64_BIT(half % 64)))
         continue;
      bool covered = false;
      for (unsigned i = 0; i < count; ++i)
         covered |= half >= ranges[i].start_halfs &&
                    half - ranges[i].start_halfs < ranges[i].size_halfs;
      if (!covered)
         return false;
   }

   const struct agx_tilebuffer_layout tib = key->native_imageblock
      ? agx_build_tilebuffer_layout_resident(key->rt_formats, 8, 1, true,
           key->max_sample_size_B, key->max_bytes_per_tile,
           key->max_samples_per_tile, key->resident_mask)
      : agx_build_tilebuffer_layout_with_limits(key->rt_formats, 8, 1, true,
           key->max_sample_size_B, key->max_bytes_per_tile,
           key->max_samples_per_tile);
   const uint32_t pass = linked->b.fragment_control.opaque[0] >> 29;
   struct hk_agx3_g17p_graphics_program_desc desc = {
      .program_va = dev->dev.shader_base + 0x20000,
      .shader_va = linked->b.bo->va->addr + linked->b.entry_offset,
      .uniform_ranges = ranges, .uniform_range_count = count,
      .shader_binary_size = linked->b.size,
      .scratch_size = MAX2(linked->b.scratch_size, linked->b.agx3_spectpr_scratch_size),
      .nr_gprs = linked->b.nr_gprs,
      .compiler_d_valid = linked->b.g17p_compiler_d_valid,
      .compiler_d = linked->b.g17p_compiler_d,
      .fragment = true,
      .writes_sample_mask = pass == AGX_PASS_TYPE_PUNCH_THROUGH ||
                           pass == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH,
      .colour_classes = linked->b.g17p_colour_classes,
      .imageblock = { .sample_size_B = tib.sample_size_B, .nr_samples = 1,
                      .tile_size_px = tib.tile_size },
      .patchable = true,
      .cprog_inputs = &s->b.info.agx3_preamble_inputs,
   };
   if (!hk_agx3_g17p_constant_program_from_binary(s->code_ptr, s->code_size,
          s->b.info.preamble_offset, s->b.info.preamble_size, hk_shader_code_va(s),
          s->b.info.binary_size, &desc.constant_program))
      return false;
   desc.constant_program.request_prelude = s->b.info.agx3_preamble_uniform_inputs;
   struct agx3_g17p_graphics_program_template *program = malloc(sizeof(*program));
   if (!program)
      return false;
   bool valid = hk_agx3_g17p_graphics_program_template_generate(dev->dev.chip, &desc, program);
   free(program);
   return valid;
}

struct hk_linked_shader *
hk_fast_link_fragment_variant(struct hk_device *dev,
                              const struct hk_shader *source,
                              const struct agx_shader_part *part,
                              const struct agx_fs_epilog_key *key)
{
   if (!part || !part->binary || !part->info.binary_size ||
       part->info.preamble_offset > part->info.binary_size ||
       part->info.preamble_size > part->info.binary_size - part->info.preamble_offset ||
       !agx_chip_g17_m5(dev->dev.chip) ||
       !hk_g17p_native_cprog_qualified(&part->info))
      return NULL;

   struct hk_shader *variant = rzalloc(NULL, struct hk_shader);
   if (!variant)
      return NULL;
   /* Only original constant-offset push loads are retained, and the epilog
    * admits no external loads. Hoisting cannot add a root access class. The
    * new constant pool and input layout belong to part->info below. */
   variant->info = source->info;
   variant->b = *part;
   variant->b.binary = malloc(part->info.binary_size);
   if (!variant->b.binary)
      goto fail;
   memcpy(variant->b.binary, part->binary, part->info.binary_size);
   variant->code_ptr = variant->b.binary;
   variant->code_size = part->info.binary_size;
   variant->compiled = true;
   variant->bo = hk_code_alloc(dev, variant->code_size, &variant->bo_offset);
   if (!variant->bo)
      goto fail;
   memcpy((uint8_t *)agx_bo_map(variant->bo) + variant->bo_offset,
          variant->code_ptr, variant->code_size);
   agx_bo_assert_range(variant->bo, variant->bo_offset, variant->code_size);
   variant->preamble_addr = hk_shader_code_va(variant) + part->info.preamble_offset;
   struct hk_linked_shader *linked = hk_fast_link(dev, true, variant, NULL, NULL, 0);
   if (!linked)
      goto fail;
   linked->variant_source = variant;
   ralloc_steal(linked, variant);
   if (!hk_m5_fragment_variant_preflight(dev, variant, linked, key)) {
      hk_destroy_linked_shader(dev, linked);
      return NULL;
   }
   if (agx_get_compiler_debug() & AGX_DBG_SHADERDB)
      fprintf(stderr, "M5 uniform blend: selected preflighted variant\n");
   return linked;

fail:
   if (variant->bo)
      agx_bo_unreference(&dev->dev, variant->bo);
   free(variant->b.binary);
   ralloc_free(variant);
   return NULL;
}

struct hk_linked_shader *
hk_fast_link_part(struct hk_device *dev, bool fragment, struct hk_shader *main,
                  struct agx_shader_part *main_part,
                  struct agx_shader_part *prolog, struct agx_shader_part *epilog,
                  unsigned nr_samples_shaded, const uint8_t *uvs_map)
{
   struct hk_linked_shader *s = rzalloc(NULL, struct hk_linked_shader);
   if (!s)
      return NULL;
   if (!agx_fast_link(&s->b, &dev->dev, fragment,
                      main_part ? main_part : &main->b, prolog, epilog,
                      nr_samples_shaded, uvs_map)) {
      ralloc_free(s);
      return NULL;
   }

   if (fragment) {
      unsigned samplers = main->b.info.sampler_state_count;
      if (s->b.uses_txf)
         samplers = MAX2(samplers, 1);

      agx_pack(&s->fs_counts, FRAGMENT_SHADER_WORD_0, cfg) {
         cfg.cf_binding_count = s->b.cf.nr_bindings;
         cfg.uniform_register_count = main->b.info.push_count;
         cfg.preshader_register_count = main->b.info.nr_preamble_gprs;
         cfg.texture_state_register_count = main->b.info.texture_state_count;
         cfg.sampler_state_register_count =
            agx_translate_sampler_state_count(samplers, false);
      }
   }

   /* HAL200 consumes the joined Ioto program built at command-recording time.
    * The legacy USC list below is not part of that contract. */
   if (agx_chip_command_stream_generation(dev->dev.chip) !=
       AGX_CS_GENERATION_AGX2) {
      hk_g17p_prewarm_linked_compute_shader(dev, main, s);
      return s;
   }

   /* Now that we've linked, bake the USC words to bind this program */
   struct agx_usc_builder b = agx_usc_builder(s->usc.data, sizeof(s->usc.data));

   if (main && main->b.info.rodata.size_16) {
      agx_usc_immediates(&b, &main->b.info.rodata, hk_shader_code_va(main));
   }

   if (s->b.uses_txf)
      agx_usc_push_packed(&b, SAMPLER, dev->dev.txf_sampler);

   agx_usc_shared_non_fragment(&b, &main->b.info, 0);
   agx_usc_push_packed(&b, SHADER, s->b.shader);
   agx_usc_push_packed(&b, REGISTERS, s->b.regs);

   if (fragment)
      agx_usc_push_packed(&b, FRAGMENT_PROPERTIES, s->b.fragment_props);

   if (main && main->b.info.has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code = agx_usc_addr(&dev->dev, main->preamble_addr);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   s->usc.size = b.head - s->usc.data;
   return s;
}

/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_physical_device.h"

#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_nir_texture.h"
#include "asahi/compiler/agx_debug.h"
#include "asahi/lib/agx_device.h"
#include "asahi/lib/agx_nir_lower_vbo.h"
#include "util/disk_cache.h"
#include "util/hex.h"
#include "util/mesa-blake3.h"
#include "util/u_process.h"
#include "util/os_misc.h"
#include "git_sha1.h"
#include "hk_buffer.h"
#include "hk_buffer_view.h"
#include "hk_descriptor_set_layout.h"
#include "hk_image_view.h"
#include "hk_sampler.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_instance.h"
#include "hk_m5_acceleration_structure.h"
#include "hk_memory_budget_core.h"
#include "hk_agx3_vdm_decode.h"
#include "hk_native_descriptor.h"
#include "hk_nir_lower_mesh.h"
#include "hk_private.h"
#include "hk_ray_tracing_core.h"
#include "hk_shader.h"
#include "hk_wsi.h"

#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "vulkan/vulkan_core.h"
#include "vulkan/wsi/wsi_common.h"
#include "vk_drm_syncobj.h"
#include "vk_physical_device.h"
#include "vk_shader_module.h"

#include <fcntl.h>
#include <string.h>
#include <xf86drm.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>

static uint32_t
hk_get_vk_version()
{
   /* Version override takes priority */
   const uint32_t version_override = vk_get_version_override();
   if (version_override)
      return version_override;

   return VK_MAKE_VERSION(1, 4, VK_HEADER_VERSION);
}

/* Vulkan cache and opaque-layout UUIDs must not alias across Apple GPU
 * contracts.  In particular, G15+ changes the ISA, command streams and
 * descriptor layouts independently, while G17 splits AGX3b/AGX3c and
 * HAL200/HAL300 by variant.  Hash explicit scalars instead of a C struct so
 * padding and host ABI cannot enter the persistent identity. */
static void
hk_hash_device_generation_contract(blake3_hasher *ctx,
                                   const struct agx_device *dev)
{
   const uint32_t values[] = {
      dev->params.gpu_generation,
      (uint8_t)dev->params.gpu_variant,
      dev->params.gpu_revision,
      dev->chip,
      agx_chip_isa_encoding(dev->chip),
      agx_chip_command_stream_generation(dev->chip),
      agx_chip_descriptor_generation(dev->chip),
      agx_chip_sampler_generation(dev->chip),
      agx_chip_sampler_heap_element_size(dev->chip),
      hk_gen1_as_enabled(dev),
      hk_g17p_rt_enabled(dev),
   };

   _mesa_blake3_update(ctx, values, sizeof(values));
}

/* VK_KHR_shader_fma: G17P with its explicit-RTE tininess repair
 * (agx_nir_lower_fma_rte()) and M5. G13-G15 do not pass the newly exposed
 * cases. */
static bool
hk_shader_fma_supported(enum agx_chip chip)
{
   return (chip == AGX_CHIP_G17P || agx_chip_g17_m5(chip)) &&
          agx_chip_exact_fma32(chip);
}

static void
hk_get_device_extensions(const struct agx_device *dev,
                         const struct hk_instance *instance,
                         struct vk_device_extension_table *ext)
{
   /* Keep the upstream AGX2 extension profile until the newly added common
    * implementations have been qualified there. Native AGX3 and M5 keep
    * their existing exposure.
    */
   const bool native_agx3 = (agx_native_graphics(dev) &&
                             dev->params.gpu_generation > 15) ||
                            agx_chip_g17_m5(dev->chip);

   /*
    * VK_KHR_ray_query and everything it depends on.  Software traversal:
    * grounded end to end in shader ALU and global loads, with no dependence
    * on the partially mapped AGX3 intersect_ray SCS ABI -- but it has never
    * executed on hardware, and an extension bit is a promise.  It stays off
    * until the exact-result oracle passes on a real G17P.
    */
   const bool ray_query = false;

   /*
    * VK_KHR_ray_tracing_pipeline.  The whole pipeline compiles to one compute
    * megakernel over the same software traversal ray_query uses, so it needs
    * exactly what ray_query needs and nothing more -- but it has never
    * executed on hardware either, and an extension bit is a promise.
    */
   const bool ray_tracing_pipeline = ray_query;

   *ext = (struct vk_device_extension_table){
      .KHR_8bit_storage = true,
      .KHR_16bit_storage = true,
      .KHR_acceleration_structure = ray_query,
      .KHR_deferred_host_operations = ray_query,
      .KHR_ray_query = ray_query,
      .KHR_ray_tracing_pipeline = ray_tracing_pipeline,
      .KHR_ray_tracing_maintenance1 = ray_tracing_pipeline,
      .KHR_bind_memory2 = true,
      .KHR_buffer_device_address = true,
      .KHR_calibrated_timestamps = true,
      .KHR_copy_commands2 = true,
      .KHR_create_renderpass2 = true,
      .KHR_dedicated_allocation = true,
      .KHR_depth_stencil_resolve = true,
      .KHR_descriptor_update_template = true,
      .KHR_device_group = true,
      .KHR_draw_indirect_count = true,
      .KHR_driver_properties = true,
      .KHR_dynamic_rendering = true,
      .KHR_dynamic_rendering_local_read = true,
      .KHR_external_fence = true,
      .KHR_external_fence_fd = true,
      .KHR_external_memory = true,
      .KHR_external_memory_fd = true,
      .KHR_external_semaphore = true,
      .KHR_external_semaphore_fd = true,
      .KHR_format_feature_flags2 = true,
      .KHR_fragment_shader_barycentric = false,
      .KHR_get_memory_requirements2 = true,
      .KHR_global_priority = true,
      .KHR_image_format_list = true,
      .KHR_imageless_framebuffer = true,
#ifdef HK_USE_WSI_PLATFORM
      .KHR_incremental_present = true,
#endif
      .KHR_index_type_uint8 = true,
      .KHR_line_rasterization = true,
      .KHR_load_store_op_none = true,
      .KHR_maintenance1 = true,
      .KHR_maintenance2 = true,
      .KHR_maintenance3 = true,
      .KHR_maintenance4 = true,
      .KHR_maintenance5 = true,
      .KHR_maintenance6 = true,
      .KHR_maintenance7 = true,
      .KHR_maintenance8 = true,
      .KHR_maintenance9 = true,
      .KHR_maintenance10 = true,
      .KHR_map_memory2 = true,
      .KHR_multiview = true,
      .KHR_pipeline_binary = true,
      .KHR_pipeline_executable_properties = true,
      .KHR_pipeline_library = true,
#ifdef HK_USE_WSI_PLATFORM
      .KHR_present_id = true,
      .KHR_present_id2 = true,
      .KHR_present_wait = true,
      .KHR_present_wait2 = true,
#endif
      .KHR_push_descriptor = true,
      .KHR_relaxed_block_layout = true,
      .KHR_robustness2 = true,
      .KHR_sampler_mirror_clamp_to_edge = true,
      .KHR_sampler_ycbcr_conversion = true,
      .KHR_separate_depth_stencil_layouts = true,
      .KHR_shader_atomic_int64 = false,
      .KHR_shader_clock = false,
      .KHR_shader_draw_parameters = true,
      .KHR_shader_expect_assume = true,
      .KHR_shader_float_controls = true,
      .KHR_shader_float_controls2 = true,
      .KHR_shader_float16_int8 = true,
      .KHR_shader_fma = true,
      .KHR_shader_integer_dot_product = true,
      .KHR_shader_maximal_reconvergence = true,
      .KHR_shader_non_semantic_info = true,
      .KHR_shader_quad_control = true,
      .KHR_shader_relaxed_extended_instruction = true,
      .KHR_shader_subgroup_extended_types = true,
      .KHR_shader_subgroup_rotate = true,
      .KHR_shader_subgroup_uniform_control_flow = true,
      .KHR_shader_terminate_invocation = true,
      .KHR_shader_untyped_pointers = true,
      .KHR_spirv_1_4 = true,
      .KHR_storage_buffer_storage_class = true,
      .KHR_timeline_semaphore = true,
#ifdef HK_USE_WSI_PLATFORM
      .KHR_swapchain = true,
      .KHR_swapchain_maintenance1 = true,
      .KHR_swapchain_mutable_format = true,
#endif
      .KHR_synchronization2 = true,
      .KHR_unified_image_layouts = native_agx3,
      .KHR_uniform_buffer_standard_layout = true,
      .KHR_variable_pointers = true,
      .KHR_vertex_attribute_divisor = true,
      .KHR_vulkan_memory_model = true,
      .KHR_workgroup_memory_explicit_layout = true,
      .KHR_zero_initialize_workgroup_memory = true,
      .EXT_4444_formats = true,
      .EXT_attachment_feedback_loop_layout = true,
      .EXT_blend_operation_advanced = true,
      .EXT_border_color_swizzle = true,
      .EXT_buffer_device_address = true,
      .EXT_calibrated_timestamps = true,
      .EXT_conditional_rendering = native_agx3,
      .EXT_color_write_enable = true,
      .EXT_custom_border_color = true,
      .EXT_debug_marker = true,
      .EXT_depth_bias_control = true,
      /* Tested on G17P only. */
      .EXT_depth_clip_control = dev->chip == AGX_CHIP_G17P,
      .EXT_depth_clip_enable = true,
      /* Qualified on G17P only (the whole binding_model.descriptor_buffer
       * group); other chips share the code but need their own run. */
      .EXT_descriptor_buffer = dev->chip == AGX_CHIP_G17P,
      .EXT_descriptor_indexing = true,
#ifdef VK_USE_PLATFORM_DISPLAY_KHR
      .EXT_display_control = false,
#endif
      .EXT_dynamic_rendering_unused_attachments = true,
      .EXT_extended_dynamic_state = true,
      .EXT_extended_dynamic_state2 = true,
      .EXT_extended_dynamic_state3 = true,
      .EXT_external_memory_dma_buf = true,
      .EXT_external_memory_host = agx_has_userptr(dev),
      .EXT_fragment_shader_interlock = agx_native_graphics(dev),
      .EXT_global_priority = true,
      .EXT_global_priority_query = true,
      .EXT_graphics_pipeline_library = true,
      .EXT_hdr_metadata = true,
      .EXT_host_query_reset = true,
      .EXT_host_image_copy = true,
      .EXT_image_2d_view_of_3d = true,
      .EXT_image_drm_format_modifier = true,
      .EXT_image_robustness = true,
      .EXT_image_sliced_view_of_3d = false,
      .EXT_image_view_min_lod = instance->drirc.misc.image_view_min_lod,
      .EXT_index_type_uint8 = true,
      .EXT_inline_uniform_block = true,
      .EXT_line_rasterization = true,
      .EXT_load_store_op_none = true,
      .EXT_map_memory_placed = true,
      .EXT_memory_budget = native_agx3,
      .EXT_memory_priority = agx_has_residency_control(dev),
      .EXT_multi_draw = true,
      .EXT_mutable_descriptor_type = true,
      /* Qualified on G17P and G17S; nested replay is shared code. */
      .EXT_nested_command_buffer =
         dev->chip == AGX_CHIP_G17P || dev->chip == AGX_CHIP_G17S,
      .EXT_non_seamless_cube_map = true,
      .EXT_pageable_device_local_memory = agx_has_residency_control(dev),
      .EXT_pipeline_creation_cache_control = true,
      .EXT_pipeline_creation_feedback = true,
      .EXT_pipeline_protected_access = true,
      .EXT_pipeline_robustness = true,
      .EXT_physical_device_drm = true,
#ifdef HK_USE_WSI_PLATFORM
      .EXT_present_timing = true,
#endif
      .EXT_primitive_restart_index = true,
      .EXT_primitive_topology_list_restart = true,
      .EXT_private_data = true,
      .EXT_primitives_generated_query = false,
      .EXT_provoking_vertex = true,
      .EXT_queue_family_foreign = true,
      .EXT_robustness2 = true,
      .EXT_sample_locations = true,
      .EXT_sampler_filter_minmax = false,
      .EXT_scalar_block_layout = true,
      .EXT_separate_stencil_usage = true,
      .EXT_shader_atomic_float = false,
      .EXT_shader_atomic_float2 = false,
      .EXT_shader_image_atomic_int64 = false,
      .EXT_shader_demote_to_helper_invocation = true,
      .EXT_shader_module_identifier = true,
      .EXT_shader_object = true,
      .EXT_shader_replicated_composites = true,
      .EXT_shader_stencil_export = true,
      .EXT_shader_subgroup_ballot = true,
      .EXT_shader_subgroup_vote = true,
      .EXT_shader_uniform_buffer_unsized_array = true,
      .EXT_shader_viewport_index_layer = true,
      .EXT_subgroup_size_control = true,
#ifdef HK_USE_WSI_PLATFORM
      .EXT_swapchain_maintenance1 = true,
#endif
      .EXT_texel_buffer_alignment = true,
      .EXT_tooling_info = true,
      .EXT_transform_feedback = true,
      

      .EXT_mesh_shader = false,
      .EXT_vertex_attribute_divisor = true,
      .EXT_vertex_input_dynamic_state = true,
      .EXT_ycbcr_2plane_444_formats = false,
      .EXT_ycbcr_image_arrays = false,
      .GOOGLE_decorate_string = true,
#ifdef HK_USE_WSI_PLATFORM
      .GOOGLE_display_timing = wsi_instance_supports_google_display_timing(&instance->vk, &instance->drirc.options),
#endif
      .GOOGLE_hlsl_functionality1 = true,
      .GOOGLE_user_type = true,
      .VALVE_mutable_descriptor_type = true,
   };
}

static void
hk_get_device_features(
   const struct agx_device *dev, const struct hk_instance *instance,
   const struct vk_device_extension_table *supported_extensions,
   struct vk_features *features)
{
   *features = (struct vk_features){
      /* Vulkan 1.0 */
      .robustBufferAccess = true,
      .fullDrawIndexUint32 = true,
      .imageCubeArray = true,
      .independentBlend = true,
      .geometryShader = true,
      .tessellationShader = true,

      
      .taskShader = supported_extensions->EXT_mesh_shader &&
                    (dev->chip == AGX_CHIP_G17P ||
                     (dev->chip == AGX_CHIP_G17S &&
                      hk_m5_task_record(dev->params.features))),
      .meshShader = supported_extensions->EXT_mesh_shader,
      /* G17P replays a multiview mesh draw per view, as other draws. */
      .multiviewMeshShader = dev->chip == AGX_CHIP_G17P &&
                             supported_extensions->EXT_mesh_shader,
      .primitiveFragmentShadingRateMeshShader = false,
      .meshShaderQueries =
         !agx_chip_g17_m5(dev->chip) && supported_extensions->EXT_mesh_shader,
      .sampleRateShading = true,
      .dualSrcBlend = true,
      .logicOp = true,
      .multiDrawIndirect = true,
      .drawIndirectFirstInstance = true,
      .depthClamp = true,
      .depthBiasClamp = true,
      .fillModeNonSolid = true,
      .depthBounds = false,
      .wideLines = true,
      .largePoints = true,
      .alphaToOne = true,
      .multiViewport = true,
      .samplerAnisotropy = true,
      .textureCompressionETC2 = true,
      .textureCompressionBC = true,
      .textureCompressionASTC_LDR = true,
      .occlusionQueryPrecise = true,
      .pipelineStatisticsQuery = true,
      .vertexPipelineStoresAndAtomics =
         instance->drirc.misc.enable_vertex_pipeline_stores_atomics ||
         agx_chip_g17_m5(dev->chip),
      .fragmentStoresAndAtomics = true,
      .shaderTessellationAndGeometryPointSize = true,
      .shaderImageGatherExtended = true,
      .shaderStorageImageExtendedFormats = true,
      /* TODO: hitting the vertex shader timeout in CTS, but should work */
      .shaderStorageImageMultisample = false,
      .shaderStorageImageReadWithoutFormat = true,
      .shaderStorageImageWriteWithoutFormat = true,
      .shaderUniformBufferArrayDynamicIndexing = true,
      .shaderSampledImageArrayDynamicIndexing = true,
      .shaderStorageBufferArrayDynamicIndexing = true,
      .shaderStorageImageArrayDynamicIndexing = true,
      .shaderClipDistance = true,
      .shaderCullDistance = true,
      .shaderFloat64 = false,
      .shaderInt64 = true,
      .shaderInt16 = true,
      .shaderResourceResidency = true,
      .shaderResourceMinLod = true,
      .sparseBinding = true,

      /* We probably could advertise multisampled sparse but we don't have a use
       * case yet and it isn't trivial.
       */
      .sparseResidency2Samples = false,
      .sparseResidency4Samples = false,
      .sparseResidency8Samples = false,
      .sparseResidencyAliased = true,
      .sparseResidencyImage2D = true,
      .sparseResidencyBuffer = true,

      /* This needs investigation. */
      .sparseResidencyImage3D = false,
      .variableMultisampleRate = false,
      .inheritedQueries = true,

      /* Vulkan 1.1 */
      .storageBuffer16BitAccess = true,
      .uniformAndStorageBuffer16BitAccess = true,
      .storagePushConstant16 = true,
      /* HAL200 interpolation is 32-bit, but the compiler performs matching
       * type-correct widening at ST_VARY and narrowing after interpolation. */
      .storageInputOutput16 = agx_chip_native_agx3(dev->chip) ||
                              agx_chip_g17_m5(dev->chip),
      .multiview = true,
      /* ViewIndex is lowered into the layer output of the last
       * pre-rasterization stage and the draw is duplicated per view, so
       * tessellation and geometry shaders work with multiview too. */
      .multiviewGeometryShader = true,
      .multiviewTessellationShader = true,
      .variablePointersStorageBuffer = true,
      .variablePointers = true,
      .shaderDrawParameters = true,
      .samplerYcbcrConversion = true,

      /* Vulkan 1.2 */
      .samplerMirrorClampToEdge = true,
      .drawIndirectCount = supported_extensions->KHR_draw_indirect_count,
      .storageBuffer8BitAccess = true,
      .uniformAndStorageBuffer8BitAccess = true,
      .storagePushConstant8 = true,
      .shaderBufferInt64Atomics = false,
      .shaderSharedInt64Atomics = false,
      .shaderFloat16 = true,
      .cooperativeMatrix = supported_extensions->KHR_cooperative_matrix,
      .cooperativeMatrixRobustBufferAccess = false,
      .shaderBFloat16Type = supported_extensions->KHR_shader_bfloat16,
      .shaderBFloat16CooperativeMatrix = supported_extensions->KHR_shader_bfloat16,
      .shaderBFloat16DotProduct = false,

      /* VK_NV_cooperative_matrix2: each feature is enabled where it is
       * implemented. */
      .cooperativeMatrixWorkgroupScope = false,
      .cooperativeMatrixFlexibleDimensions = supported_extensions->NV_cooperative_matrix2,
      .cooperativeMatrixReductionsNV = supported_extensions->NV_cooperative_matrix2,
      .cooperativeMatrixConversionsNV = supported_extensions->NV_cooperative_matrix2,
      .cooperativeMatrixPerElementOperationsNV = supported_extensions->NV_cooperative_matrix2,
      .cooperativeMatrixTensorAddressing = supported_extensions->NV_cooperative_matrix2,
      .cooperativeMatrixBlockLoads = supported_extensions->NV_cooperative_matrix2,

      .shaderInt8 = true,
      .descriptorIndexing = true,
      .shaderInputAttachmentArrayDynamicIndexing = true,
      .shaderUniformTexelBufferArrayDynamicIndexing = true,
      .shaderStorageTexelBufferArrayDynamicIndexing = true,
      .shaderUniformBufferArrayNonUniformIndexing = true,
      .shaderSampledImageArrayNonUniformIndexing = true,
      .shaderStorageBufferArrayNonUniformIndexing = true,
      .shaderStorageImageArrayNonUniformIndexing = true,
      .shaderInputAttachmentArrayNonUniformIndexing = true,
      .shaderUniformTexelBufferArrayNonUniformIndexing = true,
      .shaderStorageTexelBufferArrayNonUniformIndexing = true,
      .descriptorBindingUniformBufferUpdateAfterBind = true,
      .descriptorBindingSampledImageUpdateAfterBind = true,
      .descriptorBindingStorageImageUpdateAfterBind = true,
      .descriptorBindingStorageBufferUpdateAfterBind = true,
      .descriptorBindingUniformTexelBufferUpdateAfterBind = true,
      .descriptorBindingStorageTexelBufferUpdateAfterBind = true,
      .descriptorBindingUpdateUnusedWhilePending = true,
      .descriptorBindingPartiallyBound = true,
      .descriptorBindingVariableDescriptorCount = true,
      .runtimeDescriptorArray = true,
      .samplerFilterMinmax = supported_extensions->EXT_sampler_filter_minmax ||
                             instance->drirc.misc.fake_minmax,
      .scalarBlockLayout = true,
      .imagelessFramebuffer = true,
      .uniformBufferStandardLayout = true,
      .shaderSubgroupExtendedTypes = true,
      .separateDepthStencilLayouts = true,
      .hostQueryReset = true,
      .timelineSemaphore = true,
      .bufferDeviceAddress = true,
      .bufferDeviceAddressCaptureReplay = true,

      /* VK_KHR_acceleration_structure / VK_KHR_ray_query, gated with the
       * extensions above.  Host-side builds stay off: they are unimplemented
       * and a conformant application therefore never reaches them. */
      .accelerationStructure = supported_extensions->KHR_acceleration_structure,
      .accelerationStructureCaptureReplay =
         !agx_chip_g17_m5(dev->chip) &&
            supported_extensions->KHR_acceleration_structure,
      .accelerationStructureIndirectBuild =
         !agx_chip_g17_m5(dev->chip) &&
            supported_extensions->KHR_acceleration_structure,
      .accelerationStructureHostCommands =
         !agx_chip_g17_m5(dev->chip) &&
            supported_extensions->KHR_acceleration_structure,
      .descriptorBindingAccelerationStructureUpdateAfterBind =
         supported_extensions->KHR_acceleration_structure,
      .rayQuery = supported_extensions->KHR_ray_query,
      .rayTraversalPrimitiveCulling = supported_extensions->KHR_ray_query ||
                                      supported_extensions->KHR_ray_tracing_pipeline,

      /* VK_KHR_ray_tracing_pipeline.  The pipeline stack size
       * is static scratch in the compiled megakernel, so the dynamic state
       * cannot change it and is not advertised either. */
      .rayTracingPipeline = supported_extensions->KHR_ray_tracing_pipeline,
      .pipelineLibraryGroupHandles =
         supported_extensions->EXT_pipeline_library_group_handles,
      .rayTracingPipelineShaderGroupHandleCaptureReplay =
         supported_extensions->KHR_ray_tracing_pipeline,
      .rayTracingPipelineShaderGroupHandleCaptureReplayMixed =
         supported_extensions->KHR_ray_tracing_pipeline,
      .rayTracingPipelineTraceRaysIndirect =
         supported_extensions->KHR_ray_tracing_pipeline,

      /* VK_KHR_ray_tracing_maintenance1 */
      .rayTracingMaintenance1 =
         supported_extensions->KHR_ray_tracing_maintenance1,
      .rayTracingPositionFetch =
         supported_extensions->KHR_ray_tracing_position_fetch,
      .rayTracingPipelineTraceRaysIndirect2 =
         supported_extensions->KHR_ray_tracing_maintenance1,

      /* VK_EXT_ray_tracing_invocation_reorder */
      .rayTracingInvocationReorder =
         supported_extensions->EXT_ray_tracing_invocation_reorder,
      .bufferDeviceAddressMultiDevice = false,
      .vulkanMemoryModel = true,
      .vulkanMemoryModelDeviceScope = true,
      .vulkanMemoryModelAvailabilityVisibilityChains = false,
      .shaderOutputViewportIndex = true,
      .shaderOutputLayer = true,
      .subgroupBroadcastDynamicId = true,

      /* Vulkan 1.3 */
      .robustImageAccess = true,
      .inlineUniformBlock = true,
      .descriptorBindingInlineUniformBlockUpdateAfterBind = true,
      .pipelineCreationCacheControl = true,
      .privateData = true,
      .shaderDemoteToHelperInvocation = true,
      .shaderTerminateInvocation = true,
      .subgroupSizeControl = true,
      .computeFullSubgroups = true,
      .synchronization2 = true,
      .shaderZeroInitializeWorkgroupMemory = true,
      .dynamicRendering = true,
      .shaderIntegerDotProduct = true,
      .maintenance4 = true,

      /* Vulkan 1.4 */
      .pushDescriptor = true,

      /* VK_KHR_dynamic_rendering_local_read */
      .dynamicRenderingLocalRead = true,

      /* VK_KHR_fragment_shader_barycentric */
      .fragmentShaderBarycentric = false,

      /* VK_KHR_global_priority */
      .globalPriorityQuery = true,

      /* VK_KHR_index_type_uint8 */
      .indexTypeUint8 = true,

      /* VK_KHR_line_rasterization */
      .rectangularLines = false,
      .bresenhamLines = true,
      .smoothLines = false,
      .stippledRectangularLines = false,
      .stippledBresenhamLines = false,
      .stippledSmoothLines = false,

      /* VK_KHR_maintenance5 */
      .maintenance5 = true,

      /* VK_KHR_maintenance6 */
      .maintenance6 = true,

      /* VK_KHR_maintenance7 */
      .maintenance7 = true,

      /* VK_KHR_maintenance8 */
      .maintenance8 = true,

      /* VK_KHR_maintenance9 */
      .maintenance9 = true,
      .maintenance10 = true,

      /* VK_KHR_maintenance10 */
      .maintenance10 = true,

      /* VK_KHR_pipeline_binary */
      .pipelineBinaries = true,

      /* VK_KHR_pipeline_executable_properties */
      .pipelineExecutableInfo = true,

      /* VK_KHR_present_id */
      .presentId = true,

      /* VK_KHR_present_id2 */
      .presentId2 = true,

      /* VK_KHR_present_wait */
      .presentWait = true,

      /* VK_KHR_present_wait2 */
      .presentWait2 = true,

      /* VK_KHR_shader_clock */
      .shaderSubgroupClock = false,
      .shaderDeviceClock = false,

      /* VK_KHR_shader_expect_assume */
      .shaderExpectAssume = true,

      /* VK_KHR_shader_float_controls2 */
      .shaderFloatControls2 = true,
      .shaderFmaFloat16 = true,
      .shaderFmaFloat32 = true,
      .shaderFmaFloat64 = false,

      /* VK_KHR_shader_fma */
      .shaderFmaFloat16 = hk_shader_fma_supported(dev->chip),
      .shaderFmaFloat32 = hk_shader_fma_supported(dev->chip),
      .shaderFmaFloat64 = false,

      /* VK_KHR_shader_maximal_reconvergence */
      .shaderMaximalReconvergence = true,

      /* VK_KHR_shader_quad_control */
      .shaderQuadControl = true,

      /* VK_KHR_shader_subgroup_rotate */
      .shaderSubgroupRotate = true,
      .shaderSubgroupRotateClustered = true,

      /* VK_KHR_unified_image_layouts
       *
       * Honeykrisp does not act on VkImageLayout anywhere: barriers
       * (hk_CmdPipelineBarrier2) discard the dependency info and simply end the
       * open batches, hk_CmdBeginRendering ignores the attachment layouts, and
       * hk_TransitionImageLayoutEXT is a documented no-op.  Compression is
       * decided once at image creation and is never a function of the layout,
       * so VK_IMAGE_LAYOUT_GENERAL is already equivalent to every other layout.
       *
       * Video is not supported, so the video half stays false.
       */
      .unifiedImageLayouts = supported_extensions->KHR_unified_image_layouts,
      .unifiedImageLayoutsVideo = false,

      /* VK_KHR_vertex_attribute_divisor */
      .vertexAttributeInstanceRateDivisor = true,
      .vertexAttributeInstanceRateZeroDivisor = true,

      /* VK_KHR_workgroup_memory_explicit_layout */
      .workgroupMemoryExplicitLayout = true,
      .workgroupMemoryExplicitLayoutScalarBlockLayout = true,
      .workgroupMemoryExplicitLayout8BitAccess = true,
      .workgroupMemoryExplicitLayout16BitAccess = true,

      /* VK_EXT_4444_formats */
      .formatA4R4G4B4 = true,
      .formatA4B4G4R4 = true,

      /* VK_EXT_attachment_feedback_loop_layout */
      .attachmentFeedbackLoopLayout = true,

      /* VK_EXT_border_color_swizzle */
      .borderColorSwizzle = true,
      .borderColorSwizzleFromImage = false,

      /* VK_EXT_buffer_device_address */
      .bufferDeviceAddressCaptureReplayEXT = true,

      /* VK_EXT_color_write_enable */
      .colorWriteEnable = true,

      /* VK_EXT_conditional_rendering */
      .conditionalRendering = supported_extensions->EXT_conditional_rendering,
      .inheritedConditionalRendering =
         supported_extensions->EXT_conditional_rendering,

      /* VK_EXT_fragment_shader_interlock */
      .fragmentShaderSampleInterlock =
         supported_extensions->EXT_fragment_shader_interlock,
      /* Pixel ordering covers only the samples a fragment covers, so two
       * pixel-rate fragments splitting one MSAA pixel overlap
       * (fragment_shader_interlock.basic.*.pixel_*.4xaa.no_sample_shading).
       */
      .fragmentShaderPixelInterlock = false,
      .fragmentShaderShadingRateInterlock = false,

      /* VK_EXT_custom_border_color */
      .customBorderColors = true,
      .customBorderColorWithoutFormat = true,

      /* VK_EXT_depth_bias_control */
      .depthBiasControl = true,
      .leastRepresentableValueForceUnormRepresentation = true,
      .floatRepresentation = false,
      .depthBiasExact = true,

      /* VK_EXT_depth_clip_control, tested on G17P only */
      .depthClipControl = dev->chip == AGX_CHIP_G17P,

      /* VK_EXT_depth_clip_enable */
      .depthClipEnable = true,

      /* VK_EXT_descriptor_buffer. Descriptors never depend on the image
       * layout, and push descriptors keep their own command-buffer storage.
       */
      .descriptorBuffer = supported_extensions->EXT_descriptor_buffer,
      .descriptorBufferCaptureReplay =
         supported_extensions->EXT_descriptor_buffer,
      .descriptorBufferImageLayoutIgnored =
         supported_extensions->EXT_descriptor_buffer,
      .descriptorBufferPushDescriptors =
         supported_extensions->EXT_descriptor_buffer,

      /* VK_EXT_dynamic_rendering_unused_attachments */
      .dynamicRenderingUnusedAttachments = true,

      /* VK_EXT_extended_dynamic_state */
      .extendedDynamicState = true,

      /* VK_EXT_extended_dynamic_state2 */
      .extendedDynamicState2 = true,
      .extendedDynamicState2LogicOp = true,
      .extendedDynamicState2PatchControlPoints = true,

      /* VK_EXT_extended_dynamic_state3 */
      .extendedDynamicState3TessellationDomainOrigin = true,
      .extendedDynamicState3DepthClampEnable = true,
      .extendedDynamicState3PolygonMode = true,
      .extendedDynamicState3RasterizationSamples = true,
      .extendedDynamicState3SampleMask = true,
      .extendedDynamicState3AlphaToCoverageEnable = true,
      .extendedDynamicState3AlphaToOneEnable = true,
      .extendedDynamicState3LogicOpEnable = true,
      .extendedDynamicState3ColorBlendEnable = true,
      .extendedDynamicState3ColorBlendEquation = true,
      .extendedDynamicState3ColorWriteMask = true,
      .extendedDynamicState3RasterizationStream = true,
      .extendedDynamicState3ConservativeRasterizationMode = false,
      .extendedDynamicState3ExtraPrimitiveOverestimationSize = false,
      .extendedDynamicState3DepthClipEnable = true,
      .extendedDynamicState3SampleLocationsEnable = true,
      .extendedDynamicState3ColorBlendAdvanced = false,
      .extendedDynamicState3ProvokingVertexMode = true,
      .extendedDynamicState3LineRasterizationMode = true,
      .extendedDynamicState3LineStippleEnable = false,
      .extendedDynamicState3DepthClipNegativeOneToOne =
         dev->chip == AGX_CHIP_G17P,
      .extendedDynamicState3ViewportWScalingEnable = false,
      .extendedDynamicState3ViewportSwizzle = false,
      .extendedDynamicState3CoverageToColorEnable = false,
      .extendedDynamicState3CoverageToColorLocation = false,
      .extendedDynamicState3CoverageModulationMode = false,
      .extendedDynamicState3CoverageModulationTableEnable = false,
      .extendedDynamicState3CoverageModulationTable = false,
      .extendedDynamicState3CoverageReductionMode = false,
      .extendedDynamicState3RepresentativeFragmentTestEnable = false,
      .extendedDynamicState3ShadingRateImageEnable = false,

      /* VK_EXT_graphics_pipeline_library */
      .graphicsPipelineLibrary = true,

      /* VK_EXT_host_image_copy */
      .hostImageCopy = true,

      /* VK_EXT_image_2d_view_of_3d */
      .image2DViewOf3D = true,
      .sampler2DViewOf3D = true,

      /* VK_EXT_image_sliced_view_of_3d */
      .imageSlicedViewOf3D = false,

#ifdef HK_USE_WSI_PLATFORM
      /* VK_KHR_swapchain_maintenance1 */
      .swapchainMaintenance1 = true,
#endif

      /* VK_EXT_image_view_min_lod */
      .minLod = instance->drirc.misc.image_view_min_lod,

      /* VK_EXT_map_memory_placed */
      .memoryMapPlaced = true,
      .memoryMapRangePlaced = false,
      .memoryUnmapReserve = true,

      /* VK_EXT_memory_priority */
      .memoryPriority = agx_has_residency_control(dev),

      /* VK_EXT_pageable_device_local_memory */
      .pageableDeviceLocalMemory = agx_has_residency_control(dev),

      /* VK_EXT_multi_draw */
      .multiDraw = true,

      /* VK_EXT_mutable_descriptor_type */
      .mutableDescriptorType = true,

      /* VK_EXT_non_seamless_cube_map */
      .nonSeamlessCubeMap = true,

      /* VK_EXT_nested_command_buffer: secondaries are recorded into the
       * common command queue and replayed into the primary, so a nested
       * vkCmdExecuteCommands is one more replayed command at any depth.
       */
      .nestedCommandBuffer = supported_extensions->EXT_nested_command_buffer,
      .nestedCommandBufferRendering =
         supported_extensions->EXT_nested_command_buffer,
      .nestedCommandBufferSimultaneousUse =
         supported_extensions->EXT_nested_command_buffer,

      /* VK_EXT_pipeline_protected_access */
      .pipelineProtectedAccess = true,

      /* VK_EXT_pipeline_robustness */
      .pipelineRobustness = true,

      /* VK_EXT_primitive_restart_index */
      .primitiveRestartIndex = true,

      /* VK_EXT_primitive_topology_list_restart */
      .primitiveTopologyListRestart = true,
      .primitiveTopologyPatchListRestart = false,

      /* VK_EXT_primitives_generated_query: counted in the pre-graphics
       * helpers (assembled, tessellated or geometry-shader primitives per
       * stream), independent of rasterizer discard. */
      .primitivesGeneratedQuery =
         supported_extensions->EXT_primitives_generated_query,
      .primitivesGeneratedQueryWithNonZeroStreams =
         supported_extensions->EXT_primitives_generated_query,
      .primitivesGeneratedQueryWithRasterizerDiscard =
         supported_extensions->EXT_primitives_generated_query,

      /* VK_EXT_provoking_vertex */
      .provokingVertexLast = true,
      .transformFeedbackPreservesProvokingVertex = true,

      /* VK_KHR_robustness2 */
      .robustBufferAccess2 = true,
      .robustImageAccess2 = true,
      .nullDescriptor = true,

      /* VK_EXT_shader_atomic_float */
      .shaderBufferFloat32Atomics = supported_extensions->EXT_shader_atomic_float,
      .shaderBufferFloat32AtomicAdd = supported_extensions->EXT_shader_atomic_float,
      .shaderBufferFloat64Atomics = false,
      .shaderBufferFloat64AtomicAdd = false,
      .shaderSharedFloat32Atomics = supported_extensions->EXT_shader_atomic_float,
      .shaderSharedFloat32AtomicAdd = supported_extensions->EXT_shader_atomic_float,
      .shaderSharedFloat64Atomics = false,
      .shaderSharedFloat64AtomicAdd = false,
      .shaderImageFloat32Atomics = supported_extensions->EXT_shader_atomic_float,
      .shaderImageFloat32AtomicAdd = supported_extensions->EXT_shader_atomic_float,
      .sparseImageFloat32Atomics = false,
      .sparseImageFloat32AtomicAdd = false,

      /* VK_EXT_shader_atomic_float2 */
      .shaderBufferFloat16Atomics = false,
      .shaderBufferFloat16AtomicAdd = false,
      .shaderBufferFloat16AtomicMinMax = false,
      .shaderBufferFloat32AtomicMinMax = supported_extensions->EXT_shader_atomic_float2,
      .shaderBufferFloat64AtomicMinMax = false,
      .shaderSharedFloat16Atomics = false,
      .shaderSharedFloat16AtomicAdd = false,
      .shaderSharedFloat16AtomicMinMax = false,
      .shaderSharedFloat32AtomicMinMax = supported_extensions->EXT_shader_atomic_float2,
      .shaderSharedFloat64AtomicMinMax = false,
      .shaderImageFloat32AtomicMinMax = supported_extensions->EXT_shader_atomic_float2,
      .sparseImageFloat32AtomicMinMax = false,

      /* VK_EXT_shader_image_atomic_int64 */
      .shaderImageInt64Atomics = false,
      .sparseImageInt64Atomics = false,

      /* VK_EXT_shader_module_identifier */
      .shaderModuleIdentifier = true,

      /* VK_EXT_shader_object */
      .shaderObject = true,

      /* VK_EXT_shader_replicated_composites */
      .shaderReplicatedComposites = true,

      /* VK_KHR_shader_subgroup_uniform_control_flow */
      .shaderSubgroupUniformControlFlow = true,

      /* VK_KHR_shader_untyped_pointers */
      .shaderUntypedPointers = true,

      /* VK_EXT_shader_uniform_buffer_unsized_array */
      .shaderUniformBufferUnsizedArray = true,

      /* VK_EXT_texel_buffer_alignment */
      .texelBufferAlignment = true,

      /* VK_EXT_transform_feedback */
      .transformFeedback = true,
      .geometryStreams = true,

      /* VK_EXT_vertex_input_dynamic_state */
      .vertexInputDynamicState = true,

      /* VK_EXT_ycbcr_2plane_444_formats */
      .ycbcr2plane444Formats = false,

      /* VK_EXT_ycbcr_image_arrays */
      .ycbcrImageArrays = false,

      /* VK_KHR_shader_relaxed_extended_instruction */
      .shaderRelaxedExtendedInstruction = true,

      /* VK_EXT_blend_operation_advanced */
      .advancedBlendCoherentOperations = true,

#ifdef HK_USE_WSI_PLATFORM
      /* VK_EXT_present_timing */
      .presentTiming = true,
      .presentAtRelativeTime = true,
      .presentAtAbsoluteTime = true,
#endif
   };

   if (agx_chip_native_agx3(dev->chip)) {
      /* The first HAL200 path is intentionally one ordinary, single-sample
       * graphics pipeline. Keep optional feature reporting within that path
       * until the corresponding joined Ioto programs are implemented. */
      /* Geometry and tessellation shaders run in compute and draw through
       * their raster and evaluation vertex programs, as on AGX2. */
      /* Known issue: 16-bit varyings across the tessellation-to-geometry
       * handoff (interface_matching.*.vert_tesc_tese_geom_frag float16). */
      features->geometryShader = true;
      features->tessellationShader = true;
      features->shaderTessellationAndGeometryPointSize =
         features->geometryShader || features->tessellationShader;
      features->geometryStreams = features->geometryShader;
      features->extendedDynamicState3RasterizationStream =
         features->geometryShader;
      features->extendedDynamicState2PatchControlPoints =
         features->tessellationShader;
      features->extendedDynamicState3TessellationDomainOrigin =
         features->tessellationShader;
      /* IA vertex/primitive recording, including reset/reuse/unavailable
       * handling, has passed its exact hardware oracle twice. Every applicable
       * graphics/compute counter now has a strict native helper or fragment-
       * prolog compiler oracle and shares the guarded compact-report ABI. */
      features->pipelineStatisticsQuery = true;
      /* Vertex, tessellation and geometry storage operations share the
       * native compiler and compute-backed geometry/tessellation paths.
       */
      features->vertexPipelineStoresAndAtomics = true;
      /* Qualified on hardware: g17p-vulkan-fragment-side-effects renders
       * 1024/1024 exact pixels, 0 cleared, 0 wrong, with 0/256 mixed 2x2
       * quads and 0/32 mixed 8x4 SIMD groups, and its fragment stores,
       * atomics, untouched vertex fields, guard words and three-second
       * stability all verify exactly. No GPU faults.
       *
       * The earlier 513/1024 that kept this gated was measured against a
       * build predating 53aac78d1aa, which replaced the FragDepth write in
       * agx_nir_lower_frag_sidefx with a late test and deleted the depth path
       * that failure ran through. That change also lost the pass's
       * idempotence -- Honeykrisp preprocesses every shader twice and a
       * sample mask leaves no trace in shader_info -- so every side-effecting
       * fragment shader emitted sample_mask twice, which
       * agx_nir_lower_sample_mask documents as incorrect. Both are fixed.
       *
       * Dawn requires this feature; without it Chromium's WebGPU silently
       * falls back to SwiftShader. */
      features->fragmentStoresAndAtomics = true;
      features->advancedBlendCoherentOperations = true;
      /* Sparse binding and residency reuse the AGX2 architecture wholesale:
       * per-page VM_BIND with unbound pages mapped to the scratch/zero page,
       * a read-only shadow at +sparse_ro_offset, and a userspace sparse map
       * that the G17P texture descriptor points at in Sparse mode. The kernel
       * already accepts DRM_ASAHI_BIND_SINGLE_PAGE binds, and the G17P user VA
       * window is split for the RO alias exactly as AGX2 is. Sparse texture
       * updates flush the SSTT with a HAL200 CDM record, so other native
       * chips do not advertise it. */
      const bool sparse = dev->chip == AGX_CHIP_G17P;
      features->shaderResourceResidency = sparse;
      features->sparseBinding = sparse;
      /* A memory range may back several sparse VAs, including several
       * locations within one resource. The common binder preserves the BO
       * offset for each mapping and its RO shadow; G17P's transactional
       * VM_BIND path retains each mapping independently. Residency metadata
       * is per resource VA, not per backing allocation. */
      features->sparseResidencyAliased = sparse;
      features->sparseResidencyImage2D = sparse;
      /* The sparse binder and SSTT use the same slice layout for 3D images
       * as for 2D arrays. Report the nonstandard, one-slice block shape rather
       * than requiring the standard 3D block shape.
       */
      features->sparseResidencyImage3D = sparse;
      features->sparseResidencyBuffer = sparse;
      /* The common Honeykrisp lowering duplicates a draw per view, writes
       * ViewIndex through the draw root and routes it through the packed
       * layer/viewport UVS output.  G17P now also has exact array BG/EOT
       * coordinates, completing the two-view render path. */
      features->multiview = true;
      /* The direct triangle path uses the native geometry pipeline plus the
       * G17P-compiled libagx XFB counter-copy kernel. */
      features->transformFeedback = true;
      /* Precompiled predicate kernels use the same direct HAL200 Ioto launch
       * as the public compute path, followed by native indirect Index Lists.
       * Both indexed and non-indexed count/clamp/stride paths are qualified. */
      features->drawIndirectCount = true;
      /* G17P's native layer output occupies the high half of the shared
       * layer/viewport UVS scalar.  The public bit is backed by the exact
       * two-instance/two-layer image qualification. */
      features->shaderOutputLayer = true;
      /* G17P uses the common reconvergence and subgroup lowering with AGX3
       * quad encodings. The execution modes also select helper-capable
       * fragment entry and prevent merging triangles across derivative quads.
       */
      features->shaderQuadControl = dev->chip == AGX_CHIP_G17P;

      /* Qualified on A18 Pro only. */
      if (dev->chip == AGX_CHIP_G17P) {
         features->computeDerivativeGroupQuads = true;
         features->computeDerivativeGroupLinear = true;
         features->textureCompressionASTC_HDR = true;
      }
   }

   /* Compute derivatives are NIR quad-swap lowering plus explicit-gradient
    * samples, both available on M5; quad control remains restricted to G17P. */
   if (agx_chip_g17_m5(dev->chip)) {
      features->computeDerivativeGroupQuads = true;
      features->computeDerivativeGroupLinear = true;
   }

   /* A multiview promise for a stage this device does not expose is
    * meaningless, so keep the pair consistent however the stage bits above
    * resolved. */
   features->multiviewGeometryShader &= features->geometryShader;
   features->multiviewTessellationShader &= features->tessellationShader;

   /* G16G inserts explicit view bounds for robust image operations in NIR.
    * It qualifies both robust modes, and its descriptor swizzle supplies the
    * required out-of-bounds alpha. */
   if (dev->chip == AGX_CHIP_G16G) {
      features->robustImageAccess = true;
      features->robustImageAccess2 = true;
   }
}

#define HK_RAY_TRACING_STAGES                                                 \
   (VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |         \
    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |       \
    VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CALLABLE_BIT_KHR)

/* Stages a pipeline or shader object may contain on this device. Every
 * shader-stage mask reported in the properties must stay inside this set.
 */
static VkShaderStageFlags
hk_supported_shader_stages(const struct vk_features *features)
{
   VkShaderStageFlags stages = VK_SHADER_STAGE_VERTEX_BIT |
                               VK_SHADER_STAGE_FRAGMENT_BIT |
                               VK_SHADER_STAGE_COMPUTE_BIT;

   if (features->tessellationShader) {
      stages |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
   }

   if (features->geometryShader)
      stages |= VK_SHADER_STAGE_GEOMETRY_BIT;

   if (features->taskShader)
      stages |= VK_SHADER_STAGE_TASK_BIT_EXT;

   if (features->meshShader)
      stages |= VK_SHADER_STAGE_MESH_BIT_EXT;

   /* Ray tracing stages are inlined into compute kernels, which run subgroup
    * operations like any other compute code. */
   if (features->rayTracingPipeline)
      stages |= HK_RAY_TRACING_STAGES;

   return stages;
}

static void
hk_get_device_properties(const struct agx_device *dev,
                         const struct hk_instance *instance,
                         const struct vk_features *features,
                         struct vk_properties *properties)
{
   const unsigned descriptor_generation =
      agx_chip_descriptor_generation(dev->chip);
   const bool texel_buffer_single_texel_alignment =
      hk_descriptor_generation_is_agx2(descriptor_generation) ||
      agx_chip_native_agx3(dev->chip);
   /* G17P: what agx_nir_lower_gen1_mesh accepts and the CTS runs there
    * (misc.maximize_invocations_256); no larger mesh workgroup was run. */
   const uint32_t mesh_threads = dev->chip == AGX_CHIP_G17P ? 256 : 1024;

   /* Note: update nir_shader_compiler_options.max_samples when changing this. */
   const VkSampleCountFlagBits sample_counts =
      dev->chip == AGX_CHIP_G18G
         ? VK_SAMPLE_COUNT_1_BIT
         : (VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_2_BIT |
            VK_SAMPLE_COUNT_4_BIT |
            (agx_chip_g17_m5(dev->chip) ? VK_SAMPLE_COUNT_8_BIT : 0));

   /* Integer colour attachments and sampled multisample images use the same
    * native tile store/reload and texel-fetch paths as other colour formats.
    * Storage-image multisampling is an independent capability: keep its
    * G17P restriction until that image descriptor path is qualified.
    */
   const VkSampleCountFlagBits storage_sample_counts =
      agx_chip_native_agx3(dev->chip) ? VK_SAMPLE_COUNT_1_BIT : sample_counts;

   uint64_t os_page_size = 16384;
   os_get_page_size(&os_page_size);

   const struct agx_device_key compiler_key = {
      .chip = dev->chip,
      .encoding = agx_chip_isa_encoding(dev->chip),
   };

   /* The hardware limit is 128. However, we need to set a lower limit to
    * account for duplicated system values: clip/cull and viewport targets.
    * There are up to 8 combined clip/cull planes and only 1 viewport target,
    * giving a theoretical limit of 119. We round down to 116 to give a sensible
    * limit in vec4s.
    */
   /* G17P's 8-bit UVS indices and native vertex-state counts also cover the
    * fixed-function copies above and the unlinked shader's default Layer and
    * PointSize. The API's 128 components include position; these defaults and
    * fixed copies are internal reservations. Fragment coefficients retain
    * their independent bound. The 136-word clip boundary is hardware-tested.
    */
   unsigned max_vgt_output_components = dev->chip == AGX_CHIP_G17P ? 128 : 116;

   *properties = (struct vk_properties){
      .cooperativeMatrixSupportedStages = features->cooperativeMatrix
         ? VK_SHADER_STAGE_COMPUTE_BIT : 0,
      .cooperativeMatrixFlexibleDimensionsMaxDimension =
         features->cooperativeMatrixFlexibleDimensions ? 256 : 0,
      .apiVersion = hk_get_vk_version(),
      .driverVersion = vk_get_driver_version(),
      .vendorID = instance->drirc.debug.force_vk_vendor ?: VK_VENDOR_ID_MESA,
      .deviceID = (agx_chip_native_agx3(dev->chip) || agx_chip_g17_m5(dev->chip))
                     ? dev->params.chip_id : 0,
      .deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,

      /* Vulkan 1.0 limits */
      .maxImageDimension1D = 16384,
      .maxImageDimension2D = 16384,
      /* G17P and G16G use a 2048 limit for 3D base dimensions, although the
       * shared texture descriptor fields can encode larger arrays.
       */
      .maxImageDimension3D =
         (dev->chip == AGX_CHIP_G17P || dev->chip == AGX_CHIP_G16G ||
          agx_chip_g17_m5(dev->chip)) ? 2048 : 16384,
      .maxImageDimensionCube = 16384,
      .maxImageArrayLayers = 2048,
      /* The G17P texture/PBE includes the software alignment tail in its
       * height. Reserve the worst-case 15 R8 texels so a maximally sized
       * view at a single-texel offset still fits the native 2D descriptor.
       */
      .maxTexelBufferElements = AGX_TEXTURE_BUFFER_MAX_SIZE -
                               (agx_chip_native_agx3(dev->chip) ? 15 : 0),
      .maxUniformBufferRange = 65536,

      /* Storage buffers are lowered to global address arithmetic with a
       * zero-extended 32-bit offset, so the whole 32-bit range works. Direct3D
       * views reach just under 4 GiB.
       *
       * M5 and later fold constant index addends into a signed byte offset
       * (agx_nir_lower_address() with byte_offsets), which sign-extends a
       * scaled index and relies on every valid offset staying below 2^31.
       * Keep them at INT32_MAX; G13/G14 and G17P take the full range.
       */
      .maxStorageBufferRange =
         agx_chip_isa_encoding(dev->chip) == AGX_ISA_ENCODING_AGX2 ||
               agx_chip_isa_encoding(dev->chip) == AGX_ISA_ENCODING_AGX3B
            ? UINT32_MAX
            : INT32_MAX,
      .maxPushConstantsSize = HK_MAX_PUSH_SIZE,
      .maxMemoryAllocationCount = 4096,
      .maxSamplerAllocationCount = 1 << 20,
      .bufferImageGranularity = 0x400,
      .sparseAddressSpaceSize = agx_chip_g17_m5(dev->chip)
                                   ? HK_M5_SPARSE_ADDR_SPACE_SIZE
                                   : HK_SPARSE_ADDR_SPACE_SIZE,
      /* The classic AGX2 root upload has one USC uniform record, limited
       * to 64 halfwords. Keep upstream's eight-set limit: the native
       * sixteen-set root needs 68 halfwords and uses a separate loader.
       */
      .maxBoundDescriptorSets =
         dev->params.gpu_generation <= 15 ? 8 : HK_MAX_SETS,
      /* VK_KHR_acceleration_structure limits.  The counts are the
       * specification's own 24/29-bit encoding maxima, which is what bounds
       * us: the geometry, instance and primitive ids all live in packed
       * fields.  Scratch alignment is 128 because the common builder's
       * shaders address scratch as uvec4. */
      .maxGeometryCount = (1u << 24) - 1,
      .maxInstanceCount = (1u << 24) - 1,
      .maxPrimitiveCount = (1u << 29) - 1,
      .maxPerStageDescriptorAccelerationStructures = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindAccelerationStructures =
         HK_MAX_DESCRIPTORS,
      .maxDescriptorSetAccelerationStructures = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindAccelerationStructures =
         HK_MAX_DESCRIPTORS,
      .minAccelerationStructureScratchOffsetAlignment = 128,

      /* VK_KHR_ray_tracing_pipeline.  shaderGroupHandleSize is fixed at 32 by
       * the specification.  The base alignment is the largest the
       * specification allows. G17P's SBT fetch takes records at any byte
       * address (hk_load_sbt_entry), so it reports a handle alignment of 1;
       * other chips keep the largest allowed, 32.
       *
       * Trace recursion is unrolled into the megakernel at compile time up
       * to three levels. G17P runs deeper pipelines as continuations
       * (hk_rt_cont_split) and reports the specification's maximum of 31;
       * legacy generations retain their limit of three.
       *
       * maxRayDispatchInvocationCount is bounded by what one compute dispatch
       * can launch, since the launch is exactly a compute dispatch. */
      .shaderGroupHandleSize = HK_RAY_TRACING_GROUP_HANDLE_SIZE,
      .maxRayRecursionDepth = dev->chip == AGX_CHIP_G17P ?
         HK_G17P_MAX_RAY_RECURSION_DEPTH : HK_MAX_RAY_RECURSION_DEPTH,
      .maxShaderGroupStride = 16384,
      .shaderGroupBaseAlignment = 64,
      .shaderGroupHandleCaptureReplaySize = sizeof(uint32_t),
      .maxRayDispatchInvocationCount = 1u << 30,
      .shaderGroupHandleAlignment = dev->chip == AGX_CHIP_G17P ? 1 : 32,
      .maxRayHitAttributeSize = HK_RAY_HIT_ATTRIBS_SIZE,

      .rayTracingInvocationReorderReorderingHint =
         VK_RAY_TRACING_INVOCATION_REORDER_MODE_NONE_EXT,

      .maxPerStageDescriptorSamplers = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUniformBuffers = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorStorageBuffers = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorSampledImages = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorStorageImages = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorInputAttachments = HK_MAX_DESCRIPTORS,
      .maxPerStageResources = UINT32_MAX,
      .maxDescriptorSetSamplers = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUniformBuffers = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUniformBuffersDynamic = HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetStorageBuffers = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetStorageBuffersDynamic = HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetSampledImages = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetStorageImages = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetInputAttachments = HK_MAX_DESCRIPTORS,
      /* Bounded by the vertex prolog register ABI, not by the key tables:
       * 21 on G17P, 29 on AGX2 (agx_max_vertex_attribs). DXVK's D3D9
       * fixed-function vertex shader needs 18. */
      .maxVertexInputAttributes = agx_max_vertex_attribs(&compiler_key),
      .maxVertexInputBindings = AGX_MAX_VBUFS,
      .maxVertexInputAttributeOffset = 65535,
      .maxVertexInputBindingStride = 2048,

      .maxVertexOutputComponents = max_vgt_output_components,
      .maxGeometryShaderInvocations = 32,
      .maxGeometryInputComponents = 128,
      .maxGeometryOutputComponents = max_vgt_output_components,
      .maxGeometryOutputVertices = 1024,
      .maxGeometryTotalOutputComponents = 1024,
      .maxTessellationGenerationLevel = 64,
      .maxTessellationPatchSize = 32,
      .maxTessellationControlPerVertexInputComponents = 128,
      .maxTessellationControlPerVertexOutputComponents = 128,
      .maxTessellationControlPerPatchOutputComponents = 120,
      .maxTessellationControlTotalOutputComponents = 4216,
      .maxTessellationEvaluationInputComponents = 128,
      .maxTessellationEvaluationOutputComponents = max_vgt_output_components,

      




      .maxTaskWorkGroupTotalCount = 1u << 22,
      .maxTaskWorkGroupCount = {65535, 65535, 65535},
      .maxTaskWorkGroupInvocations = 1024,
      .maxTaskWorkGroupSize = {1024, 1024, 1024},
      .maxTaskPayloadSize = 16384,
      .maxTaskSharedMemorySize = 32768,
      .maxTaskPayloadAndSharedMemorySize = 32768,
      .maxMeshWorkGroupTotalCount = 1u << 22,
      .maxMeshWorkGroupCount = {65535, 65535, 65535},
      .maxMeshWorkGroupInvocations = mesh_threads,
      .maxMeshWorkGroupSize = {mesh_threads, mesh_threads, mesh_threads},
      .maxMeshSharedMemorySize = 28672,
      .maxMeshPayloadAndSharedMemorySize = 28672 + 16384,
      .maxMeshOutputMemorySize = 32768,
      .maxMeshPayloadAndOutputMemorySize = 32768 + 16384,
      .maxMeshOutputComponents = 128,
      .maxMeshOutputVertices = 256,
      .maxMeshOutputPrimitives = 256,
      .maxMeshOutputLayers = 8,
      /* A multiview mesh draw replays once per view (hk_foreach_view), which
       * works up to HK_MAX_MULTIVIEW_VIEW_COUNT. Temporarily 16: CTS
       * mesh_shader.ext.properties.max_view_index builds its view mask as
       * (1u << count) - 1, undefined at 32. Raise it once the CTS is fixed. */
      .maxMeshMultiviewViewCount = features->multiviewMeshShader ? 16 : 1,
      .meshOutputPerVertexGranularity = 32,
      .meshOutputPerPrimitiveGranularity = 32,
      .maxPreferredTaskWorkGroupInvocations = 32,
      .maxPreferredMeshWorkGroupInvocations = 32,
      .prefersLocalInvocationVertexOutput = true,
      .prefersLocalInvocationPrimitiveOutput = true,
      .prefersCompactVertexOutput = true,
      .prefersCompactPrimitiveOutput = true,

      /* Set to match maxVertexOutputComponents, hardware limit is higher. */
      .maxFragmentInputComponents = max_vgt_output_components,
      .maxFragmentOutputAttachments = HK_MAX_RTS,
      .maxFragmentDualSrcAttachments = 1,
      .maxFragmentCombinedOutputResources = HK_MAX_RTS + HK_MAX_DESCRIPTORS,
      .maxComputeSharedMemorySize = HK_MAX_SHARED_SIZE,
      .maxComputeWorkGroupCount = {0x7fffffff, 65535, 65535},
      .maxComputeWorkGroupInvocations = 1024,
      .maxComputeWorkGroupSize = {
         1024, 1024, agx_chip_g17_m5(dev->chip) ? 1024 : 64},
      .subPixelPrecisionBits = 8,
      .subTexelPrecisionBits = 8,
      .mipmapPrecisionBits = 8,
      .maxDrawIndexedIndexValue = UINT32_MAX,
      .maxDrawIndirectCount = UINT16_MAX,
      .maxSamplerLodBias = 15,
      .maxSamplerAnisotropy = 16,
      .maxViewports = HK_MAX_VIEWPORTS,
      .maxViewportDimensions = {32768, 32768},
      .viewportBoundsRange = {-65536, 65536},
      .viewportSubPixelBits = 8,
      .minMemoryMapAlignment = os_page_size,
      .minTexelBufferOffsetAlignment = HK_MIN_TEXEL_BUFFER_ALIGNMENT,
      .minUniformBufferOffsetAlignment = HK_MIN_UBO_ALIGNMENT,
      .minStorageBufferOffsetAlignment = HK_MIN_SSBO_ALIGNMENT,
      .minTexelOffset = -8,
      .maxTexelOffset = 7,
      .minTexelGatherOffset = -8,
      .maxTexelGatherOffset = 7,
      .minInterpolationOffset = -0.5,
      .maxInterpolationOffset = 0.4375,
      .subPixelInterpolationOffsetBits = 4,
      .maxFramebufferHeight = 16384,
      .maxFramebufferWidth = 16384,
      .maxFramebufferLayers = 2048,
      .framebufferColorSampleCounts = sample_counts,
      .framebufferDepthSampleCounts = sample_counts,
      .framebufferNoAttachmentsSampleCounts = sample_counts,
      .framebufferStencilSampleCounts = sample_counts,
      .maxColorAttachments = HK_MAX_RTS,
      .sampledImageColorSampleCounts = sample_counts,
      .sampledImageIntegerSampleCounts = sample_counts,
      .sampledImageDepthSampleCounts = sample_counts,
      .sampledImageStencilSampleCounts = sample_counts,
      .storageImageSampleCounts = storage_sample_counts,
      .maxSampleMaskWords = 1,
      .timestampComputeAndGraphics = true,
      /* FIXME: Is timestamp period actually 1? */
      .timestampPeriod = 1.0f,
      .maxClipDistances = 8,
      .maxCullDistances = 8,
      .maxCombinedClipAndCullDistances = 8,
      .discreteQueuePriorities = 2,
      .pointSizeRange = {1.0, 512.f - 0.0625f},
      .lineWidthRange = {1.0, 16.0f},
      .pointSizeGranularity = 0.0625,
      .lineWidthGranularity = 1.0f / 16.0f,
      .strictLines = false,
      .standardSampleLocations = true,
      .optimalBufferCopyOffsetAlignment = 1,
      .optimalBufferCopyRowPitchAlignment = 1,
      .nonCoherentAtomSize = 64,

      /* Vulkan 1.0 sparse properties */
      .sparseResidencyNonResidentStrict = true,
      .sparseResidencyAlignedMipSize = false,
      .sparseResidencyStandard2DBlockShape = true,

      /* We can implement the standard block size for MSAA 4x but maybe not MSAA
       * 2x?
       */
      .sparseResidencyStandard2DMultisampleBlockShape = false,

      /* As far as I can tell, there is no way to implement this on G13. This
       * is a shame because D3D12 requires it for FL12.2.
       */
      .sparseResidencyStandard3DBlockShape = false,

      /* Vulkan 1.1 properties */
      .subgroupSize = 32,
      .subgroupSupportedStages = hk_supported_shader_stages(features),
      .subgroupSupportedOperations =
         VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
         VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_QUAD_BIT |
         VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
         VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
         VK_SUBGROUP_FEATURE_ROTATE_BIT_KHR |
         VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
         VK_SUBGROUP_FEATURE_CLUSTERED_BIT |
         VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT_KHR,
      .subgroupQuadOperationsInAllStages = true,
      .meshAndTaskShaderDerivatives =
         dev->chip == AGX_CHIP_G17P && features->meshShader,
      .pointClippingBehavior = VK_POINT_CLIPPING_BEHAVIOR_USER_CLIP_PLANES_ONLY,
      .maxMultiviewViewCount = HK_MAX_MULTIVIEW_VIEW_COUNT,
      .maxMultiviewInstanceIndex = UINT32_MAX,
      .maxPerSetDescriptors = UINT32_MAX,
      .maxMemoryAllocationSize = (1ull << 37),

      /* Vulkan 1.2 properties */
      .supportedDepthResolveModes =
         VK_RESOLVE_MODE_SAMPLE_ZERO_BIT | VK_RESOLVE_MODE_AVERAGE_BIT |
         VK_RESOLVE_MODE_MIN_BIT | VK_RESOLVE_MODE_MAX_BIT,
      .supportedStencilResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT |
                                      VK_RESOLVE_MODE_MIN_BIT |
                                      VK_RESOLVE_MODE_MAX_BIT,
      .independentResolveNone = true,
      .independentResolve = true,
      .driverID = instance->drirc.misc.report_honeykrisp_driver_id
                     ? VK_DRIVER_ID_MESA_HONEYKRISP
                     : VK_DRIVER_ID_AURORA_BOREALIS,
      /* Conformance is submitted per driver ID. Borealis has not made a
       * submission under its own ID yet, so it claims no CTS version on any
       * chip, including the ones Honeykrisp is conformant on.
       */
      .conformanceVersion = (VkConformanceVersion){0, 0, 0, 0},
      .denormBehaviorIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
      .roundingModeIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
      .shaderSignedZeroInfNanPreserveFloat16 = true,
      .shaderSignedZeroInfNanPreserveFloat32 = true,
      .shaderSignedZeroInfNanPreserveFloat64 = false,
      .shaderDenormPreserveFloat16 = true,
      /* agx_nir_lower_denorm_preserve() computes flushed cases exactly. */
      .shaderDenormPreserveFloat32 = dev->chip == AGX_CHIP_G17P,
      .shaderDenormPreserveFloat64 = false,
      .shaderDenormFlushToZeroFloat16 = false,
      .shaderDenormFlushToZeroFloat32 = true,
      .shaderDenormFlushToZeroFloat64 = false,
      .shaderRoundingModeRTEFloat16 = true,
      /* FP32 arithmetic rounds to nearest even on every generation. On
       * G13-G15 an RTE execution mode leaves the program unchanged, as
       * upstream does; G17P adds its FMA repair (agx_nir_lower_fma_rte).
       * VK_KHR_shader_fma is gated separately.
       */
      .shaderRoundingModeRTEFloat32 = true,
      .shaderRoundingModeRTEFloat64 = false,
      .shaderRoundingModeRTZFloat16 = false,
      .shaderRoundingModeRTZFloat32 = false,
      .shaderRoundingModeRTZFloat64 = false,
      .maxUpdateAfterBindDescriptorsInAllPools = UINT32_MAX,
      .shaderUniformBufferArrayNonUniformIndexingNative = true,
      .shaderSampledImageArrayNonUniformIndexingNative = true,
      .shaderStorageBufferArrayNonUniformIndexingNative = true,
      .shaderStorageImageArrayNonUniformIndexingNative = true,
      .shaderInputAttachmentArrayNonUniformIndexingNative = true,
      .robustBufferAccessUpdateAfterBind = true,
      .quadDivergentImplicitLod = false,
      .maxPerStageDescriptorUpdateAfterBindSamplers = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindUniformBuffers = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindStorageBuffers = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindSampledImages = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindStorageImages = HK_MAX_DESCRIPTORS,
      .maxPerStageDescriptorUpdateAfterBindInputAttachments =
         HK_MAX_DESCRIPTORS,
      .maxPerStageUpdateAfterBindResources = UINT32_MAX,
      .maxDescriptorSetUpdateAfterBindSamplers = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindUniformBuffers = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindUniformBuffersDynamic =
         HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetUpdateAfterBindStorageBuffers = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindStorageBuffersDynamic =
         HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetUpdateAfterBindSampledImages = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindStorageImages = HK_MAX_DESCRIPTORS,
      .maxDescriptorSetUpdateAfterBindInputAttachments = HK_MAX_DESCRIPTORS,
      .filterMinmaxSingleComponentFormats =
         agx_chip_sampler_generation(dev->chip) == AGX_SAMPLER_GENERATION_GEN5 ||
         instance->drirc.misc.fake_minmax,
      .filterMinmaxImageComponentMapping =
         agx_chip_sampler_generation(dev->chip) == AGX_SAMPLER_GENERATION_GEN5,
      .maxTimelineSemaphoreValueDifference = UINT64_MAX,
      .framebufferIntegerColorSampleCounts = sample_counts,

      /* Vulkan 1.3 properties */
      .minSubgroupSize = 32,
      .maxSubgroupSize = 32,
      .maxComputeWorkgroupSubgroups = 1024 / 32,
      /* The only subgroup size is 32, so requiring it is always satisfied,
       * also in ray tracing stages, which are inlined into compute kernels,
       * and in G17P task (compute) and mesh stages. */
      .requiredSubgroupSizeStages =
         VK_SHADER_STAGE_COMPUTE_BIT |
         (features->rayTracingPipeline ? HK_RAY_TRACING_STAGES : 0) |
         (dev->chip == AGX_CHIP_G17P && features->meshShader
             ? VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT
             : 0),
      .maxInlineUniformBlockSize = HK_MAX_INLINE_UNIFORM_BLOCK_SIZE,
      .maxPerStageDescriptorInlineUniformBlocks = 32,
      .maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks = 32,
      .maxDescriptorSetInlineUniformBlocks = 6 * 32,
      .maxDescriptorSetUpdateAfterBindInlineUniformBlocks = 6 * 32,
      .maxInlineUniformTotalSize = 1 << 16,
      .integerDotProduct4x8BitPackedUnsignedAccelerated = false,
      .integerDotProduct4x8BitPackedSignedAccelerated = false,
      .integerDotProduct4x8BitPackedMixedSignednessAccelerated = false,
      .storageTexelBufferOffsetAlignmentBytes = HK_MIN_TEXEL_BUFFER_ALIGNMENT,
      .storageTexelBufferOffsetSingleTexelAlignment =
         texel_buffer_single_texel_alignment,
      .uniformTexelBufferOffsetAlignmentBytes = HK_MIN_TEXEL_BUFFER_ALIGNMENT,
      .uniformTexelBufferOffsetSingleTexelAlignment =
         texel_buffer_single_texel_alignment,
      .maxBufferSize = HK_MAX_BUFFER_SIZE,

      /* Vulkan 1.4 properties */
      .dynamicRenderingLocalReadDepthStencilAttachments = false,
      .dynamicRenderingLocalReadMultisampledAttachments = true,

      /* VK_KHR_push_descriptor */
      .maxPushDescriptors = HK_MAX_PUSH_DESCRIPTORS,

      /* VK_EXT_custom_border_color */
      .maxCustomBorderColorSamplers = 1 << 20,

      /* VK_EXT_descriptor_buffer. A descriptor-buffer set is the same memory
       * a descriptor pool hands out, so every size is the binding stride of
       * that type, and set offsets keep the 64-byte inline-uniform and
       * 32-byte storage PBE record alignment pool sets have. Push descriptors
       * keep their own command-buffer storage; the push buffer the
       * application then binds is never read. Embedded samplers and samplers
       * share the device sampler heap.
       */
      .combinedImageSamplerDescriptorSingleArray = true,
      .bufferlessPushDescriptors = false,
      .allowSamplerImageViewPostSubmitCreation = false,
      .descriptorBufferOffsetAlignment = HK_MIN_UBO_ALIGNMENT,
      .maxDescriptorBufferBindings = HK_MAX_DESCRIPTOR_BUFFERS,
      .maxResourceDescriptorBufferBindings = HK_MAX_DESCRIPTOR_BUFFERS,
      .maxSamplerDescriptorBufferBindings = HK_MAX_DESCRIPTOR_BUFFERS,
      .maxEmbeddedImmutableSamplerBindings = HK_MAX_DESCRIPTOR_BUFFERS,
      .maxEmbeddedImmutableSamplers = 2032,
      /* Capture/replay: memory, buffers and images keep their addresses, an
       * image view its heap textures and a sampler its heap samplers. An
       * acceleration structure descriptor is its buffer's address. */
      .bufferCaptureReplayDescriptorDataSize =
         sizeof(struct hk_buffer_capture),
      .imageCaptureReplayDescriptorDataSize = sizeof(struct hk_image_capture),
      .imageViewCaptureReplayDescriptorDataSize =
         sizeof(struct hk_image_view_capture),
      .samplerCaptureReplayDescriptorDataSize =
         sizeof(struct hk_sampler_capture),
      .accelerationStructureCaptureReplayDescriptorDataSize = 0,
      .EDBsamplerDescriptorSize =
         hk_descriptor_buffer_size(dev, VK_DESCRIPTOR_TYPE_SAMPLER),
      .combinedImageSamplerDescriptorSize = hk_descriptor_buffer_size(
         dev, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER),
      .sampledImageDescriptorSize =
         hk_descriptor_buffer_size(dev, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
      .storageImageDescriptorSize =
         hk_descriptor_buffer_size(dev, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE),
      .uniformTexelBufferDescriptorSize = hk_descriptor_buffer_size(
         dev, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER),
      .robustUniformTexelBufferDescriptorSize = hk_descriptor_buffer_size(
         dev, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER),
      .storageTexelBufferDescriptorSize = hk_descriptor_buffer_size(
         dev, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER),
      .robustStorageTexelBufferDescriptorSize = hk_descriptor_buffer_size(
         dev, VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER),
      .uniformBufferDescriptorSize =
         hk_descriptor_buffer_size(dev, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
      .robustUniformBufferDescriptorSize =
         hk_descriptor_buffer_size(dev, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
      .storageBufferDescriptorSize =
         hk_descriptor_buffer_size(dev, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
      .robustStorageBufferDescriptorSize =
         hk_descriptor_buffer_size(dev, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
      .inputAttachmentDescriptorSize =
         hk_descriptor_buffer_size(dev, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT),
      .accelerationStructureDescriptorSize = hk_descriptor_buffer_size(
         dev, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR),
      .maxSamplerDescriptorBufferRange = HK_MAX_DESCRIPTOR_SET_SIZE,
      .maxResourceDescriptorBufferRange = HK_MAX_DESCRIPTOR_SET_SIZE,
      /* Descriptor buffers may live anywhere in the application's VA. */
      .samplerDescriptorBufferAddressSpaceSize = dev->main_heap_size,
      .resourceDescriptorBufferAddressSpaceSize = dev->main_heap_size,
      .descriptorBufferAddressSpaceSize = dev->main_heap_size,

      /* VK_EXT_extended_dynamic_state3 */
      .dynamicPrimitiveTopologyUnrestricted = true,

      /* VK_EXT_graphics_pipeline_library */
      .graphicsPipelineLibraryFastLinking = true,
      .graphicsPipelineLibraryIndependentInterpolationDecoration = true,

      /* VK_EXT_host_image_copy */

      /* VK_KHR_line_rasterization */
      .lineSubPixelPrecisionBits = 8,

      /* VK_KHR_maintenance5 */
      .earlyFragmentMultisampleCoverageAfterSampleCounting = false,
      .earlyFragmentSampleMaskTestBeforeSampleCounting =
         !agx_chip_g17_m5(dev->chip),
      .depthStencilSwizzleOneSupport = true,
      .polygonModePointSize = false,
      .nonStrictSinglePixelWideLinesUseParallelogram = false,
      .nonStrictWideLinesUseParallelogram = false,

      /* VK_KHR_maintenance6 */
      .blockTexelViewCompatibleMultipleLayers = false,
      .maxCombinedImageSamplerDescriptorCount = 3,
      .fragmentShadingRateClampCombinerInputs = false,

      /* VK_KHR_maintenance7 */
      .robustFragmentShadingRateAttachmentAccess = false,
      .separateDepthStencilAttachmentAccess = false, /* TODO */
      .maxDescriptorSetTotalUniformBuffersDynamic = HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetTotalStorageBuffersDynamic = HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetTotalBuffersDynamic = HK_MAX_DYNAMIC_BUFFERS,
      .maxDescriptorSetUpdateAfterBindTotalUniformBuffersDynamic =
         HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetUpdateAfterBindTotalStorageBuffersDynamic =
         HK_MAX_DYNAMIC_BUFFERS / 2,
      .maxDescriptorSetUpdateAfterBindTotalBuffersDynamic =
         HK_MAX_DYNAMIC_BUFFERS,

      /* VK_KHR_maintenance9 */
      .image2DViewOf3DSparse = false,
      .defaultVertexAttributeValue =
         VK_DEFAULT_VERTEX_ATTRIBUTE_VALUE_ZERO_ZERO_ZERO_ONE_KHR,

      /* VK_KHR_maintenance10 */
      .rgba4OpaqueBlackSwizzled = true,
      .resolveSrgbFormatAppliesTransferFunction = true,
      .resolveSrgbFormatSupportsTransferFunctionControl = true,

      /* VK_EXT_map_memory_placed */
      .minPlacedMemoryMapAlignment = os_page_size,

      /* VK_EXT_external_memory_host: whole pages of both the CPU and the
       * GPU, which binds in 16 KiB pages. */
      .minImportedHostPointerAlignment = MAX2(os_page_size, 16384),

      /* VK_EXT_multi_draw */
      .maxMultiDrawCount = UINT16_MAX,

      /* VK_EXT_nested_command_buffer */
      .maxCommandBufferNestingLevel = UINT32_MAX,

      /* VK_KHR_pipeline_binary
       *
       * InternalCache properties are set by
       * hk_physical_device_init_pipeline_cache()
       */
      .pipelineBinaryCompressedData = false,

      /* VK_EXT_pipeline_robustness */
      .defaultRobustnessStorageBuffers =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .defaultRobustnessUniformBuffers =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .defaultRobustnessVertexInputs =
         VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .defaultRobustnessImages =
         VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2_EXT,

      /* VK_EXT_physical_device_drm gets populated later */

      /* VK_EXT_provoking_vertex */
      .provokingVertexModePerPipeline = true,
      .transformFeedbackPreservesTriangleFanProvokingVertex = true,

      /* VK_KHR_robustness2 */
      .robustStorageBufferAccessSizeAlignment = HK_SSBO_BOUNDS_CHECK_ALIGNMENT,
      .robustUniformBufferAccessSizeAlignment = HK_MIN_UBO_ALIGNMENT,

      /* VK_EXT_sample_locations */
      .sampleLocationSampleCounts = sample_counts & ~VK_SAMPLE_COUNT_1_BIT,
      .maxSampleLocationGridSize = (VkExtent2D){1, 1},
      .sampleLocationCoordinateRange[0] = 0.0f,
      .sampleLocationCoordinateRange[1] = 0.9375f,
      .sampleLocationSubPixelBits = 4,
      .variableSampleLocations = false,

      /* VK_EXT_shader_object */
      .shaderBinaryVersion = hk_shader_binary_version(dev->chip),

      /* VK_EXT_transform_feedback */
      .maxTransformFeedbackStreams = 4,
      .maxTransformFeedbackBuffers = 4,
      .maxTransformFeedbackBufferSize = UINT32_MAX,
      .maxTransformFeedbackStreamDataSize = 2048,
      .maxTransformFeedbackBufferDataSize = 512,
      .maxTransformFeedbackBufferDataStride = 2048,
      .transformFeedbackQueries = true,
      .transformFeedbackStreamsLinesTriangles = true,
      .transformFeedbackRasterizationStreamSelect = true,
      .transformFeedbackDraw = true,

      /* VK_KHR_vertex_attribute_divisor */
      .maxVertexAttribDivisor = UINT32_MAX,
      .supportsNonZeroFirstInstance = true,

      /* VK_KHR_fragment_shader_barycentric */
      .triStripVertexOrderIndependentOfProvokingVertex = false,

      /* VK_EXT_blend_operation_advanced */
      .advancedBlendMaxColorAttachments = HK_MAX_RTS,
      .advancedBlendIndependentBlend = true,
      .advancedBlendNonPremultipliedSrcColor = true,
      .advancedBlendNonPremultipliedDstColor = true,
      .advancedBlendCorrelatedOverlap = true,
      .advancedBlendAllOperations = true,
   };

   /* Strict residency on M5 relies on soft faults; see
    * hk_sparse_buffer_unmapped(). */
   if (agx_chip_g17_m5(dev->chip))
      properties->sparseResidencyNonResidentStrict = agx_has_soft_fault(dev);

   /* G17P maps unbound pages to a zeroed fill page, so they read zero, but
    * the page is writable: a store or copy into an unbound page is read back
    * afterwards, which strict residency forbids. Report it only where the
    * driconf asks (vkd3d-proton, for tiled resources tier 2). */
   if (dev->chip == AGX_CHIP_G17P)
      properties->sparseResidencyNonResidentStrict =
         instance->drirc.misc.sparse_strict_residency;

   /* G13 and G14 back the writable alias of an unbound page with the shared
    * fill page, as G17P does: a store or copy into an unbound page is read
    * back afterwards, and through the writable alias any unbound page returns
    * what another one received.  Same rule as G17P. */
   if (dev->chip == AGX_CHIP_G13G || dev->chip == AGX_CHIP_G13X ||
       dev->chip == AGX_CHIP_G14G || dev->chip == AGX_CHIP_G14X)
      properties->sparseResidencyNonResidentStrict =
         instance->drirc.misc.sparse_strict_residency;

   if (agx_chip_native_agx3(dev->chip)) {
      /* Measured per operation under the G17P drm-shim:
       *
       *   BASIC             g22_subgroup_basic          encodes
       *   VOTE              g21_subgroup_vote           encodes
       *   BALLOT            g05_subgroup_ballot         encodes
       *   ARITHMETIC        g06_subgroup_arith          encodes
       *   SHUFFLE           g07_subgroup_shuffle        encodes
       *   SHUFFLE_RELATIVE  g25_subgroup_shuffle_rel    encodes
       *   CLUSTERED         g23_subgroup_clustered      encodes
       *   QUAD              g08_subgroup_quad           encodes
       *   ROTATE/CLUSTERED  g24_subgroup_rotate         encodes
       *
       * G17P uses the same lane instructions in native vertex/fragment
       * programs and the compute programs implementing tessellation and
       * geometry. Other AGX3 targets retain their stage restriction. */
      properties->subgroupQuadOperationsInAllStages =
         dev->chip == AGX_CHIP_G17P;
   }

   snprintf(properties->deviceName, sizeof(properties->deviceName), "%s",
            (strlen(instance->drirc.debug.force_vk_devicename) > 0) ?
            instance->drirc.debug.force_vk_devicename : dev->name);

   /* VK_EXT_shader_module_identifier */
   static_assert(sizeof(vk_shaderModuleIdentifierAlgorithmUUID) ==
                 sizeof(properties->shaderModuleIdentifierAlgorithmUUID));
   memcpy(properties->shaderModuleIdentifierAlgorithmUUID,
          vk_shaderModuleIdentifierAlgorithmUUID,
          sizeof(properties->shaderModuleIdentifierAlgorithmUUID));

   uint8_t dev_uuid[VK_UUID_SIZE];
   agx_get_device_uuid(dev, &dev_uuid);
   static_assert(sizeof(dev_uuid) == VK_UUID_SIZE);
   memcpy(properties->deviceUUID, &dev_uuid, VK_UUID_SIZE);
   static_assert(sizeof(instance->driver_build_sha) >= VK_UUID_SIZE);
   memcpy(properties->driverUUID, instance->driver_build_sha, VK_UUID_SIZE);

   strncpy(properties->driverName, HK_DRIVER_NAME, VK_MAX_DRIVER_NAME_SIZE);
   snprintf(properties->driverInfo, VK_MAX_DRIVER_INFO_SIZE,
            "Mesa " PACKAGE_VERSION MESA_GIT_SHA1 HK_DRIVER_INFO_SUFFIX);

   /* We don't use the layouts ATM so just report all layouts from
    * extensions that we support as compatible.
    */
   static const VkImageLayout supported_layouts[] = {
      VK_IMAGE_LAYOUT_GENERAL, /* required by spec */
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_PREINITIALIZED,
      VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
      /* VK_KHR_dynamic_rendering_local_read is supported, and
       * unifiedImageLayouts makes this layout interchangeable with the rest,
       * so host copy has to accept it too. */
      VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ,
      // VK_IMAGE_LAYOUT_FRAGMENT_DENSITY_MAP_OPTIMAL_EXT,
      VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
   };

   properties->pCopySrcLayouts = (VkImageLayout *)supported_layouts;
   properties->copySrcLayoutCount = ARRAY_SIZE(supported_layouts);
   properties->pCopyDstLayouts = (VkImageLayout *)supported_layouts;
   properties->copyDstLayoutCount = ARRAY_SIZE(supported_layouts);

   /* We're a UMR so we can always map every kind of memory */
   properties->identicalMemoryTypeRequirements = true;

   {
      blake3_hasher blake3_ctx;
      uint8_t blake3[BLAKE3_KEY_LEN];

      _mesa_blake3_init(&blake3_ctx);
      /* Make sure we don't match with other vendors */
      const char *driver = "honeykrisp-v1";
      _mesa_blake3_update(&blake3_ctx, driver, strlen(driver));
      hk_hash_device_generation_contract(&blake3_ctx, dev);
      _mesa_blake3_final(&blake3_ctx, blake3);

      memcpy(properties->optimalTilingLayoutUUID, blake3, VK_UUID_SIZE);
   }
}

/* Every AGX_*, HK_* and ASAHI_* environment variable that can change generated
 * code or serialized objects is part of the cache identity, not only the bits
 * hk_physical_device_compiler_flags() packs. Variables that only print or pick
 * a runtime path are excluded so tracing keeps a warm cache. */
extern char **environ;
static int
hk_env_cmp(const void *a, const void *b)
{
   return strcmp(*(const char *const *)a, *(const char *const *)b);
}

uint64_t
hk_codegen_env_hash(void)
{
   static const char *const runtime_only[] = {
      "TRACE", "DUMP", "_LOG", "PERF", "VERBOSE", "DEBUG_OBJECT", "SLOW_DRAW",
      "DISK_CACHE_ID", "LAZY_GENERIC", "FORCE_GENERIC", "BIND_PREDICT",
      "VARIANT_TRACE", "SLOW_SUBMIT", "PROLOG_DISK_CACHE",
   };
   const char *list[512];
   unsigned n = 0;
   for (char **e = environ; e && *e && n < ARRAY_SIZE(list); ++e) {
      if (strncmp(*e, "AGX_", 4) && strncmp(*e, "HK_", 3) &&
          strncmp(*e, "ASAHI_", 6))
         continue;
      const char *eq = strchr(*e, '=');
      const size_t name_len = eq ? (size_t)(eq - *e) : strlen(*e);
      bool skip = false;
      for (unsigned i = 0; i < ARRAY_SIZE(runtime_only) && !skip; ++i) {
         const size_t l = strlen(runtime_only[i]);
         for (size_t o = 0; o + l <= name_len && !skip; ++o)
            skip = !strncmp(*e + o, runtime_only[i], l);
      }
      /* HK_PERFTEST includes code-generation choices such as noborder. */
      if (!skip || (name_len == strlen("HK_PERFTEST") &&
                    !strncmp(*e, "HK_PERFTEST", name_len)))
         list[n++] = *e;
   }
   qsort(list, n, sizeof(list[0]), hk_env_cmp);
   uint64_t h = 0xcbf29ce484222325ull;
   for (unsigned i = 0; i < n; ++i)
      for (const char *c = list[i]; ; ++c) {
         h = (h ^ (uint8_t)*c) * 0x100000001b3ull;
         if (!*c)
            break;
      }
   return h;
}

/* Whether a program's shaders emulate custom border colours is decided at
 * the first sampler that needs it (hk_sampler.c), which lands at a different
 * point among the shaders compiled on worker threads each time the program
 * runs. Shaders compiled before it leave the emulation out and need an
 * emulating twin, built on the draw that binds them, and the shader cache key
 * carries the setting, so a run that met the sampler at another point missed
 * the cache. Remember in the disk cache that this program needed it, so its
 * next run emulates from the start. The emulation is a branch on the bound
 * sampler, correct with or without a custom border. */
static bool
hk_border_emulation_key(struct hk_physical_device *pdev, cache_key key)
{
#ifdef ENABLE_SHADER_CACHE
   if (!pdev->vk.disk_cache)
      return false;
   static const char tag[] = "hk border emulation";
   const char *process = util_get_process_name();
   struct mesa_blake3 ctx;
   blake3_hash hash;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, tag, sizeof(tag));
   if (process)
      _mesa_blake3_update(&ctx, process, strlen(process));
   _mesa_blake3_final(&ctx, hash);
   disk_cache_compute_key(pdev->vk.disk_cache, hash, sizeof(hash), key);
   return true;
#else
   return false;
#endif
}

void
hk_border_emulation_recall(struct hk_physical_device *pdev)
{
#ifdef ENABLE_SHADER_CACHE
   cache_key key;
   size_t size;
   if (!hk_border_emulation_key(pdev, key))
      return;
   void *seen = disk_cache_get(pdev->vk.disk_cache, key, &size);
   if (seen)
      p_atomic_set(&pdev->border_emulation, 1);
   free(seen);
#endif
}

void
hk_border_emulation_remember(struct hk_physical_device *pdev)
{
#ifdef ENABLE_SHADER_CACHE
   static const uint8_t seen = 1;
   cache_key key;
   if (hk_border_emulation_key(pdev, key))
      disk_cache_put(pdev->vk.disk_cache, key, &seen, sizeof(seen), NULL);
#endif
}

static void
hk_physical_device_init_pipeline_cache(struct hk_physical_device *pdev)
{
   struct hk_instance *instance = hk_physical_device_instance(pdev);

   blake3_hasher blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);

   _mesa_blake3_update(&blake3_ctx, instance->driver_build_sha,
                     sizeof(instance->driver_build_sha));
   hk_hash_device_generation_contract(&blake3_ctx, &pdev->dev);

   const uint64_t compiler_flags = hk_physical_device_compiler_flags(pdev);
   _mesa_blake3_update(&blake3_ctx, &compiler_flags, sizeof(compiler_flags));
   const uint64_t env_hash = hk_codegen_env_hash();
   _mesa_blake3_update(&blake3_ctx, &env_hash, sizeof(env_hash));

   unsigned char blake3[BLAKE3_KEY_LEN];
   _mesa_blake3_final(&blake3_ctx, blake3);

   static_assert(BLAKE3_KEY_LEN >= VK_UUID_SIZE);
   memcpy(pdev->vk.properties.pipelineCacheUUID, blake3, VK_UUID_SIZE);
   memcpy(pdev->vk.properties.shaderBinaryUUID, blake3, VK_UUID_SIZE);

#ifdef ENABLE_SHADER_CACHE
   char renderer[10];
   ASSERTED int len =
      snprintf(renderer, sizeof(renderer), "HK_G%u%c_",
               pdev->dev.params.gpu_generation, pdev->dev.params.gpu_variant);

   assert(len == sizeof(renderer) - 2);

   /* The driver id names the single-file cache directory: keep it to the
    * 20-byte build-id (40 hex digits), not the zero-padded 32-byte buffer.
    * Steam records single-file cache directories as "L7:<driver id>" cut
    * to 64 characters and prunes directories that match no bucket, so a
    * longer id would get the live cache deleted.
    */
   const unsigned id_bytes = 20;
   static_assert(sizeof(instance->driver_build_sha) >= 20);
   char timestamp[2 * 20 + 1];
   mesa_bytes_to_hex(timestamp, instance->driver_build_sha, id_bytes);
   const uint64_t driver_flags = hk_physical_device_compiler_flags(pdev);
   pdev->vk.disk_cache = disk_cache_create(renderer, timestamp, driver_flags ^
      (env_hash * UINT64_C(0xff51afd7ed558ccd)));
   if (pdev->vk.disk_cache != NULL) {
      hk_border_emulation_recall(pdev);
      pdev->vk.properties.pipelineBinaryInternalCache = true;
      pdev->vk.properties.pipelineBinaryInternalCacheControl = true;
      pdev->vk.properties.pipelineBinaryPrefersInternalCache = true;
      pdev->vk.properties.pipelineBinaryPrecompiledInternalCache = true;
   }
#endif
}

static void
hk_physical_device_free_disk_cache(struct hk_physical_device *pdev)
{
#ifdef ENABLE_SHADER_CACHE
   if (pdev->vk.disk_cache) {
      disk_cache_destroy(pdev->vk.disk_cache);
      pdev->vk.disk_cache = NULL;
   }
#else
   assert(pdev->vk.disk_cache == NULL);
#endif
}

static uint64_t
hk_get_sysmem_heap_size(struct hk_physical_device *pdev)
{
   if (pdev->sysmem)
      return pdev->sysmem;

   struct hk_instance *instance = hk_physical_device_instance(pdev);
   return os_get_gpu_heap_size(instance->drirc.misc.heap_memory_percent,
                               &instance->drirc.misc.heap_memory_percent);
}

static simple_mtx_t hk_heap_usage_lock = SIMPLE_MTX_INITIALIZER;
static struct hk_heap_usage *hk_heap_usages;

/* Return the process-wide usage record for a render node, creating it on
 * first use. It outlives any one instance, so it is not allocated with an
 * instance's allocation callbacks. */
static struct hk_heap_usage *
hk_heap_usage_get(dev_t render_dev)
{
   simple_mtx_lock(&hk_heap_usage_lock);

   struct hk_heap_usage *usage = hk_heap_usages;
   while (usage && usage->render_dev != render_dev)
      usage = usage->next;

   if (usage) {
      usage->refcount++;
   } else {
      usage = calloc(1, sizeof(*usage));
      if (usage) {
         usage->render_dev = render_dev;
         usage->refcount = 1;
         usage->next = hk_heap_usages;
         hk_heap_usages = usage;
      }
   }

   simple_mtx_unlock(&hk_heap_usage_lock);
   return usage;
}

static void
hk_heap_usage_put(struct hk_heap_usage *usage)
{
   if (!usage)
      return;

   simple_mtx_lock(&hk_heap_usage_lock);

   if (--usage->refcount == 0) {
      struct hk_heap_usage **link = &hk_heap_usages;
      while (*link != usage)
         link = &(*link)->next;

      *link = usage->next;
      free(usage);
   }

   simple_mtx_unlock(&hk_heap_usage_lock);
}

static uint64_t
hk_get_sysmem_heap_available(struct hk_physical_device *pdev)
{
   assert(pdev->sysmem);

   uint64_t total_used = 0;
   for (unsigned i = 0; i < pdev->mem_heap_count; i++) {
      const struct hk_memory_heap *heap = &pdev->mem_heaps[i];
      uint64_t used = p_atomic_read(heap->used) +
                         p_atomic_read(&heap->cached->value);
      total_used += used;
   }

   return total_used < pdev->sysmem ? pdev->sysmem - total_used : 0;
}

VkResult
hk_create_drm_physical_device(struct vk_instance *_instance,
                              drmDevicePtr drm_device,
                              struct vk_physical_device **pdev_out)
{
   struct hk_instance *instance = (struct hk_instance *)_instance;
   VkResult result;

   if (!(drm_device->available_nodes & (1 << DRM_NODE_RENDER)) ||
       drm_device->bustype != DRM_BUS_PLATFORM)
      return VK_ERROR_INCOMPATIBLE_DRIVER;

   const char *path = drm_device->nodes[DRM_NODE_RENDER];
   int fd = open(path, O_RDWR | O_CLOEXEC);
   if (fd < 0) {
      return vk_errorf(instance, VK_ERROR_INCOMPATIBLE_DRIVER,
                       "failed to open device %s", path);
   }

   drmVersionPtr version = drmGetVersion(fd);
   if (!version) {
      result =
         vk_errorf(instance, VK_ERROR_INCOMPATIBLE_DRIVER,
                   "failed to query kernel driver version for device %s", path);
      goto fail_fd;
   }

   bool is_asahi = agx_is_native_drm_driver(version->name);
   is_asahi |= strcmp(version->name, "virtio_gpu") == 0;
   drmFreeVersion(version);

   if (!is_asahi) {
      /* Fail silently */
      result = VK_ERROR_INCOMPATIBLE_DRIVER;
      goto fail_fd;
   }

   struct stat st;
   if (stat(drm_device->nodes[DRM_NODE_RENDER], &st)) {
      result = vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                         "fstat() failed on %s: %m",
                         drm_device->nodes[DRM_NODE_RENDER]);
      goto fail_fd;
   }
   const dev_t render_dev = st.st_rdev;

   struct hk_physical_device *pdev =
      vk_zalloc(&instance->vk.alloc, sizeof(*pdev), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);

   if (pdev == NULL) {
      result = vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail_fd;
   }

   /* We're render-only */
   pdev->master_fd = -1;
   pdev->render_dev = render_dev;
   pdev->dev.fd = fd;

   if (!agx_open_device(NULL, &pdev->dev)) {
      /* Fail silently, for virtgpu */
      result = VK_ERROR_INCOMPATIBLE_DRIVER;
      goto fail_pdev_alloc;
   }

   struct vk_physical_device_dispatch_table dispatch_table;
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &hk_physical_device_entrypoints, true);
   vk_physical_device_dispatch_table_from_entrypoints(
      &dispatch_table, &wsi_physical_device_entrypoints, false);

   struct vk_device_extension_table supported_extensions;
   hk_get_device_extensions(&pdev->dev, instance, &supported_extensions);
   if (hk_g16s_as_enabled(&pdev->dev)) {
      supported_extensions.KHR_acceleration_structure = true;
      supported_extensions.KHR_deferred_host_operations = true;
   }
   if (hk_g17p_rt_enabled(&pdev->dev)) {
      supported_extensions.KHR_acceleration_structure = true;
      supported_extensions.KHR_deferred_host_operations = true;
      supported_extensions.KHR_ray_query =
         (pdev->dev.params.features & DRM_ASAHI_FEATURE_RENDER_RAY_QUERY) != 0;
      supported_extensions.KHR_ray_tracing_pipeline = true;
      supported_extensions.EXT_pipeline_library_group_handles = true;
      supported_extensions.EXT_ray_tracing_invocation_reorder = true;
      supported_extensions.KHR_ray_tracing_maintenance1 = true;
      supported_extensions.KHR_ray_tracing_position_fetch = true;
   }

   /* Explicit bring-up profile for the native M5 acceleration structure
    * builder. Nothing traverses the native format yet, so ray queries and
    * ray tracing pipelines stay off. */
   if (agx_chip_g17_m5(pdev->dev.chip) &&
       debug_get_bool_option("ASAHI_M5_NATIVE_AS", false)) {
      supported_extensions.KHR_acceleration_structure = true;
      supported_extensions.KHR_deferred_host_operations = true;
   }
   /* The 8x8 f16/f32 matmad underlying a subgroup-scope cooperative matrix
    * exists on the AGX3B ISA (G15/G16/G17P) just as it does on the M5 AGX3C
    * encoding; only the M5-exclusive 16x16 GEMM needs the kernel
    * DRM_ASAHI_FEATURE_COMPUTE_MATRIX, and the 8x8 matmad dispatch never sets
    * uses_matrix16, so AGX3B coopmat is purely a userspace capability. The
    * default is decided per chip: a part advertises coopmat only once its
    * dEQP-VK.compute.*cooperative_matrix* run has passed on real silicon
    * (agx_chip_cooperative_matrix_qualified, G16S today); every other AGX3B
    * part stays opt-in via ASAHI_AGX3B_COOPERATIVE_MATRIX=1. The env var
    * overrides the default either way. M5 behaviour is unchanged.
    *
    * The qualification is per product, not per architecture profile: an M4
    * Pro on the G17 kernel runtime takes the G17P profile (agx_chip.c) but
    * is the same G16S silicon, so it keeps its default too. */
   const bool m5_cmat = agx_chip_g17_m5(pdev->dev.chip) &&
      debug_get_bool_option("ASAHI_M5_COOPERATIVE_MATRIX", true);
   const bool agx3b_cmat =
      agx_chip_isa_encoding(pdev->dev.chip) == AGX_ISA_ENCODING_AGX3B &&
      debug_get_bool_option(
         "ASAHI_AGX3B_COOPERATIVE_MATRIX",
         agx_chip_cooperative_matrix_qualified(pdev->dev.chip) ||
            agx_is_m4_pro(&pdev->dev));
   supported_extensions.KHR_cooperative_matrix = m5_cmat || agx3b_cmat;

   /* bf16 inputs ride the same 8x8 form, but no AGX3B bf16 matmad has been
    * witnessed from Apple yet, so KHR_shader_bfloat16 stays M5-only until a
    * fixture (and bf16 CTS) proves it on AGX3B. */
   supported_extensions.KHR_shader_bfloat16 =
      m5_cmat && debug_get_bool_option("ASAHI_M5_BFLOAT16", true);

   /* Most of VK_NV_cooperative_matrix2 (flexible dimensions, tensor
    * addressing, block loads, per-element operations, reductions) is lowered
    * for the M5 only, so an AGX3B part that enables the 8x8
    * VK_KHR_cooperative_matrix does not advertise it. */
   supported_extensions.NV_cooperative_matrix2 = m5_cmat;

   /* Counted by the pre-graphics helpers, which every native chip runs;
    * hardware-qualified on M5 and G17P. */
   supported_extensions.EXT_primitives_generated_query =
      agx_chip_g17_m5(pdev->dev.chip) || pdev->dev.chip == AGX_CHIP_G17P;

   supported_extensions.EXT_mesh_shader = pdev->dev.chip == AGX_CHIP_G17S &&
      hk_mesh_mode() == HK_MESH_MODE_NATIVE_PROBE &&
      (pdev->dev.params.features & DRM_ASAHI_FEATURE_RENDER_RESOURCES_V2);

   /* Gen5 adds native component-wise minimum/maximum sampler reduction. */
   supported_extensions.EXT_sampler_filter_minmax =
      agx_chip_sampler_generation(pdev->dev.chip) == AGX_SAMPLER_GENERATION_GEN5;

   if (agx_native_graphics(&pdev->dev)) {
      supported_extensions.KHR_shader_quad_control =
         pdev->dev.chip == AGX_CHIP_G17P;
   }

   /* VK_KHR_shader_fma makes shaderFmaFloat32 mandatory. */
   if (!hk_shader_fma_supported(pdev->dev.chip))
      supported_extensions.KHR_shader_fma = false;

   if (pdev->dev.chip == AGX_CHIP_G17P) {
      /* Advanced blending is hardware-qualified on A18 Pro for every
       * operation and advertised overlap/premultiplication mode, including
       * coherent sequencing, across the bounded one-attachment profile. */
      supported_extensions.KHR_compute_shader_derivatives = true;
      supported_extensions.EXT_shader_atomic_float = true;
      supported_extensions.EXT_shader_atomic_float2 = true;
      supported_extensions.EXT_texture_compression_astc_hdr = true;
   }

   if (agx_chip_g17_m5(pdev->dev.chip))
      supported_extensions.KHR_compute_shader_derivatives = true;

   /* AGX3C has no witnessed native float atomic; agx_nir_lower_atomics turns
    * fp32 add/min/max on buffers, shared memory and (lowered) images into
    * compare-and-swap loops, so the same feature set is exact on M5. */
   if (agx_chip_g17_m5(pdev->dev.chip)) {
      supported_extensions.EXT_shader_atomic_float = true;
      supported_extensions.EXT_shader_atomic_float2 = true;
   }

   struct vk_features supported_features;
   /* G17P runs mesh shaders natively and task shaders in software. */
   if (pdev->dev.chip == AGX_CHIP_G17P)
      supported_extensions.EXT_mesh_shader = true;
   hk_get_device_features(&pdev->dev, instance, &supported_extensions,
                          &supported_features);

   struct vk_properties properties;
   hk_get_device_properties(&pdev->dev, instance, &supported_features,
                            &properties);

   properties.drmHasRender = true;
   properties.drmRenderMajor = major(render_dev);
   properties.drmRenderMinor = minor(render_dev);

   result = vk_physical_device_init(&pdev->vk, &instance->vk,
                                    &supported_extensions, &supported_features,
                                    &properties, &dispatch_table);
   if (result != VK_SUCCESS)
      goto fail_agx_device;

   /* Descriptor writes encode a scaled_conversion kind only for views whose
    * format is [SU]SCALED (hk_scaled_texture_format()). Such a view needs
    * sampled-image or input-attachment features for its format, so if no
    * scaled format has them, no valid descriptor carries a nonzero kind. */
   pdev->scaled_sampling = false;
   pdev->scaled_storage = false;
   for (VkFormat f = VK_FORMAT_UNDEFINED + 1;
        f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK &&
        !(pdev->scaled_sampling && pdev->scaled_storage); ++f) {
      if (hk_scaled_texture_format(hk_format_to_pipe_format(f)) ==
          PIPE_FORMAT_NONE)
         continue;
      const VkFormatFeatureFlags2 sampled =
         VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
         VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT |
         VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT;
      const VkFormatFeatureFlags2 images =
         hk_get_image_format_features(pdev, f, VK_IMAGE_TILING_OPTIMAL) |
         hk_get_image_format_features(pdev, f, VK_IMAGE_TILING_LINEAR);
      pdev->scaled_sampling |= (images & sampled) != 0;
      pdev->scaled_storage |=
         (images & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT) != 0 ||
         (hk_get_buffer_format_features(pdev, f) &
          VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT) != 0;
   }

   hk_physical_device_init_pipeline_cache(pdev);

   const char *hk_sysmem = os_get_option("HK_SYSMEM");
   if (hk_sysmem) {
      uint64_t sysmem = strtoll(hk_sysmem, NULL, 10);
      if (sysmem != LLONG_MIN && sysmem != LLONG_MAX) {
         pdev->sysmem = sysmem;
      }
   }

   uint64_t sysmem_size_B = hk_get_sysmem_heap_size(pdev);
   if (sysmem_size_B == 0) {
      result = vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                         "Failed to query total system memory");
      goto fail_disk_cache;
   }

   pdev->heap_usage = hk_heap_usage_get(render_dev);
   if (!pdev->heap_usage) {
      result = vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail_disk_cache;
   }

   uint32_t sysmem_heap_idx = pdev->mem_heap_count++;
   pdev->mem_heaps[sysmem_heap_idx] = (struct hk_memory_heap){
      .size = sysmem_size_B,
      .used = &pdev->heap_usage->used[sysmem_heap_idx],
      .cached = &pdev->heap_usage->cached[sysmem_heap_idx],
      .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
   };

   pdev->mem_types[pdev->mem_type_count++] = (VkMemoryType){
      .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      .heapIndex = sysmem_heap_idx,
   };

   pdev->mem_types[pdev->mem_type_count++] = (VkMemoryType){
      .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
      .heapIndex = sysmem_heap_idx,
   };

   assert(pdev->mem_heap_count <= ARRAY_SIZE(pdev->mem_heaps));
   assert(pdev->mem_type_count <= ARRAY_SIZE(pdev->mem_types));

   VkQueueFlags queue_flags =
      VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT |
      VK_QUEUE_SPARSE_BINDING_BIT;
   if (agx_native_graphics(&pdev->dev)) {
      /* Graphics, compute and transfers use the native HAL200 queue path.
       * Sparse binding rides the shared per-page VM_BIND path; advertise it
       * only when the same opt-in that turns on the sparse features above is
       * set, so the queue family and the device features stay consistent. */
      queue_flags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT |
                    VK_QUEUE_TRANSFER_BIT;
      if (pdev->vk.supported_features.sparseBinding)
         queue_flags |= VK_QUEUE_SPARSE_BINDING_BIT;
   }

   uint32_t queue_count = 1;
   /* Under virtio the simple-ioctl path forwards the request struct, not the
    * buffer .pointer refers to, so the queue-limits query cannot work there.
    */
   if ((pdev->dev.params.features & DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES) &&
       !pdev->dev.is_virtio) {
#define HK_ADVERTISED_QUEUES 8u
      struct drm_asahi_queue_limits limits = {0};
      struct drm_asahi_get_params query = {
         .param_group = DRM_ASAHI_PARAM_GROUP_QUEUE_LIMITS,
         .pointer = (uintptr_t)&limits,
         .size = sizeof(limits),
      };
      if (pdev->dev.ops.simple_ioctl(&pdev->dev, DRM_IOCTL_ASAHI_GET_PARAMS,
                                    &query)) {
         result = vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                            "Failed to query kernel queue limits: %m");
         goto fail_disk_cache;
      }
      if (!limits.max_queues || !limits.max_in_flight_per_queue) {
         result = vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                            "Kernel reports no usable command queues");
         goto fail_disk_cache;
      }
      /* limits.max_queues is the machine-wide pool of application GART
       * contexts (g17_queue_limits.rs: contexts 5..62), shared with every
       * other client on the device. VkQueueFamilyProperties::queueCount is a
       * per-device guarantee: an application may create a device with that
       * many queues and expect it to succeed. Handing the shared pool out as
       * the guarantee makes any application that asks for queueCount queues
       * fail as soon as other clients hold contexts. Advertise a count we
       * can deliver; a real guarantee needs a per-client reservation in the
       * kernel. */
      queue_count = MIN2(limits.max_queues, HK_ADVERTISED_QUEUES);
   }

   pdev->queue_families[pdev->queue_family_count++] = (struct hk_queue_family){
      .queue_flags = queue_flags,

      .queue_count = queue_count,
   };

   /* With independent kernel queues, also expose compute-only and
    * transfer-only families. Kernel queues carry no capability class: every
    * queue accepts render and compute commands, and hk records copies,
    * clears and blits the same way on any family. These families only
    * narrow what the application may record, so they reuse the universal
    * queue implementation, and like it they create their kernel queue on
    * first submission, in the device's one VM. The G17P and M3 kernels
    * report independent queues too, but the families are qualified on M5. */
   if (agx_chip_g17_m5(pdev->dev.chip) &&
       (pdev->dev.params.features & DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES)) {
      pdev->queue_families[pdev->queue_family_count++] = (struct hk_queue_family){
         .queue_flags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
         .queue_count = queue_count,
      };
      pdev->queue_families[pdev->queue_family_count++] = (struct hk_queue_family){
         .queue_flags = VK_QUEUE_TRANSFER_BIT,
         .queue_count = queue_count,
      };
   }
   assert(pdev->queue_family_count <= ARRAY_SIZE(pdev->queue_families));

   unsigned st_idx = 0;
   pdev->syncobj_sync_type = vk_drm_syncobj_get_type(fd);
   pdev->sync_types[st_idx++] = &pdev->syncobj_sync_type;
   pdev->sync_types[st_idx++] = NULL;
   assert(st_idx <= ARRAY_SIZE(pdev->sync_types));
   pdev->vk.supported_sync_types = pdev->sync_types;

   result = hk_init_wsi(pdev);
   if (result != VK_SUCCESS)
      goto fail_disk_cache;

   simple_mtx_init(&pdev->debug_compile_lock, mtx_plain);
   *pdev_out = &pdev->vk;

   return VK_SUCCESS;

fail_disk_cache:
   hk_heap_usage_put(pdev->heap_usage);
   hk_physical_device_free_disk_cache(pdev);
   vk_physical_device_finish(&pdev->vk);
fail_agx_device:
   agx_close_device(&pdev->dev);
fail_pdev_alloc:
   if (pdev->master_fd)
      close(pdev->master_fd);

   vk_free(&instance->vk.alloc, pdev);
fail_fd:
   close(fd);
   return result;
}

void
hk_physical_device_destroy(struct vk_physical_device *vk_pdev)
{
   struct hk_physical_device *pdev =
      container_of(vk_pdev, struct hk_physical_device, vk);

   hk_finish_wsi(pdev);

   if (pdev->master_fd >= 0)
      close(pdev->master_fd);

   simple_mtx_destroy(&pdev->debug_compile_lock);
   hk_heap_usage_put(pdev->heap_usage);
   hk_physical_device_free_disk_cache(pdev);
   agx_close_device(&pdev->dev);
   vk_physical_device_finish(&pdev->vk);
   vk_free(&pdev->vk.instance->alloc, pdev);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetPhysicalDeviceMemoryProperties2(
   VkPhysicalDevice physicalDevice,
   VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   struct hk_instance *instance = hk_physical_device_instance(pdev);

   pMemoryProperties->memoryProperties.memoryHeapCount = pdev->mem_heap_count;
   for (int i = 0; i < pdev->mem_heap_count; i++) {
      pMemoryProperties->memoryProperties.memoryHeaps[i] = (VkMemoryHeap){
         .size = pdev->mem_heaps[i].size,
         .flags = pdev->mem_heaps[i].flags,
      };
   }

   pMemoryProperties->memoryProperties.memoryTypeCount = pdev->mem_type_count;
   for (int i = 0; i < pdev->mem_type_count; i++) {
      pMemoryProperties->memoryProperties.memoryTypes[i] = pdev->mem_types[i];
   }

   vk_foreach_struct(sType, ext, pMemoryProperties->pNext) {
      switch (sType) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT: {
         VkPhysicalDeviceMemoryBudgetPropertiesEXT *p = ext;

         uint64_t sysmem_budget =
            pdev->sysmem ? hk_get_sysmem_heap_available(pdev) : 0;

         for (unsigned i = 0; i < pdev->mem_heap_count; i++) {
            const struct hk_memory_heap *heap = &pdev->mem_heaps[i];
            uint64_t allocated = p_atomic_read(heap->used);
            uint64_t used = allocated + p_atomic_read(&heap->cached->value);

            /* From the Vulkan 1.3.278 spec:
             *
             *    "heapUsage is an array of VK_MAX_MEMORY_HEAPS VkDeviceSize
             *    values in which memory usages are returned, with one element
             *    for each memory heap. A heap’s usage is an estimate of how
             *    much memory the process is currently using in that heap."
             *
             * Idle BOs in the BO cache are reclaimable, and an allocation
             * that reuses one must still raise the reported usage, so they
             * only lower the budget below.
             *
             * TODO: Include internal allocations?
             */
            p->heapUsage[i] = allocated;

            /* Set the budget at 90% to avoid thrashing. */
            float percent = 0.9f;

            uint64_t budget;
            if (sysmem_budget) {
               budget = MIN2(sysmem_budget + used, heap->size);
               budget = ROUND_DOWN_TO((uint64_t)(budget * percent), 1 << 20);
            } else {
               /* Scale the budget the same way the heap was scaled. */
               percent *= instance->drirc.misc.heap_memory_percent;

               budget = vk_physical_device_heap_budget_from_system(
                  &pdev->vk, percent, heap->size, used);
            }

            p->heapBudget[i] = budget;
         }

         /* From the Vulkan 1.3.278 spec:
          *
          *    "The heapBudget and heapUsage values must be zero for array
          *    elements greater than or equal to
          *    VkPhysicalDeviceMemoryProperties::memoryHeapCount. The
          *    heapBudget value must be non-zero for array elements less than
          *    VkPhysicalDeviceMemoryProperties::memoryHeapCount."
          */
         for (unsigned i = pdev->mem_heap_count; i < VK_MAX_MEMORY_HEAPS; i++) {
            p->heapBudget[i] = 0u;
            p->heapUsage[i] = 0u;
         }
         break;
      }
      default:
         vk_debug_ignored_stype(sType);
         break;
      }
   }
}

static const VkQueueGlobalPriorityKHR hk_global_queue_priorities[] = {
   VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR,
   VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR,
   VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR,
};

VKAPI_ATTR void VKAPI_CALL
hk_GetPhysicalDeviceQueueFamilyProperties2(
   VkPhysicalDevice physicalDevice, uint32_t *pQueueFamilyPropertyCount,
   VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out, pQueueFamilyProperties,
                          pQueueFamilyPropertyCount);

   for (uint8_t i = 0; i < pdev->queue_family_count; i++) {
      const struct hk_queue_family *queue_family = &pdev->queue_families[i];

      vk_outarray_append_typed(VkQueueFamilyProperties2, &out, p)
      {
         p->queueFamilyProperties.queueFlags = queue_family->queue_flags;
         p->queueFamilyProperties.queueCount = queue_family->queue_count;
         p->queueFamilyProperties.timestampValidBits = 64;
         p->queueFamilyProperties.minImageTransferGranularity =
            (VkExtent3D){1, 1, 1};

         VkQueueFamilyGlobalPriorityPropertiesKHR *prio = vk_find_struct(
            p->pNext, QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_KHR);
         if (prio) {
            STATIC_ASSERT(ARRAY_SIZE(hk_global_queue_priorities) <=
                          VK_MAX_GLOBAL_PRIORITY_SIZE_KHR);
            /* The native kernel reserves REALTIME and only accepts HIGH on
             * M5. Report the same classes that queue creation can admit. */
            prio->priorityCount = agx_chip_g17_m5(pdev->dev.chip) ? 3 : 2;
            memcpy(&prio->priorities, hk_global_queue_priorities,
                   prio->priorityCount * sizeof(hk_global_queue_priorities[0]));
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_GetPhysicalDeviceMultisamplePropertiesEXT(
   VkPhysicalDevice physicalDevice, VkSampleCountFlagBits samples,
   VkMultisamplePropertiesEXT *pMultisampleProperties)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);

   if (samples & pdev->vk.properties.sampleLocationSampleCounts) {
      pMultisampleProperties->maxSampleLocationGridSize = (VkExtent2D){1, 1};
   } else {
      pMultisampleProperties->maxSampleLocationGridSize = (VkExtent2D){0, 0};
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetPhysicalDeviceCooperativeMatrixPropertiesKHR(
   VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
   VkCooperativeMatrixPropertiesKHR *pProperties)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkCooperativeMatrixPropertiesKHR, out, pProperties,
                          pPropertyCount);
   if (pdev->vk.supported_features.cooperativeMatrix) {
      /* The order of this list is implementation-defined. List the 8x8x8
       * FP32 x FP32 configuration first: CTS
       * cooperative_matrix.op_constant_null steps through the device's
       * configurations and, for the last one, needs it in its own table,
       * which has no FP32 A/B operands; when the last configuration is
       * missing there it builds a zero configuration and crashes. */
      vk_outarray_append_typed(VkCooperativeMatrixPropertiesKHR, &out, p) {
         *p = (VkCooperativeMatrixPropertiesKHR) {
            .sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
            .MSize = 8, .NSize = 8, .KSize = 8,
            .AType = VK_COMPONENT_TYPE_FLOAT32_KHR,
            .BType = VK_COMPONENT_TYPE_FLOAT32_KHR,
            .CType = VK_COMPONENT_TYPE_FLOAT32_KHR,
            .ResultType = VK_COMPONENT_TYPE_FLOAT32_KHR,
            .saturatingAccumulation = VK_FALSE,
            .scope = VK_SCOPE_SUBGROUP_KHR,
         };
      }
      /* The 16x16 GEMM form that the kernel COMPUTE_MATRIX feature unlocks is
       * AGX3C (M5) only; AGX3B offers just the 8x8 matmad, so cap it at 8 even
       * if a kernel were to report the feature there. */
      unsigned max_size =
         ((pdev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_MATRIX) &&
          agx_chip_isa_encoding(pdev->dev.chip) == AGX_ISA_ENCODING_AGX3C) ? 16 : 8;
      for (unsigned size = max_size; size >= 8; size /= 2) {
         const VkComponentTypeKHR types[] = {VK_COMPONENT_TYPE_FLOAT16_KHR,
            VK_COMPONENT_TYPE_BFLOAT16_KHR};
         for (unsigned i = 0; i < ARRAY_SIZE(types); ++i) {
            if (types[i] == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
                !pdev->vk.supported_features.shaderBFloat16CooperativeMatrix)
               continue;
            unsigned modes = types[i] == VK_COMPONENT_TYPE_BFLOAT16_KHR ? 2 : 1;
            for (unsigned accum = 0; accum < modes; ++accum) {
               vk_outarray_append_typed(VkCooperativeMatrixPropertiesKHR, &out, p) {
                  *p = (VkCooperativeMatrixPropertiesKHR) {
                     .sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
                     .MSize = size, .NSize = size, .KSize = size,
                     .AType = types[i], .BType = types[i],
                     .CType = accum ? VK_COMPONENT_TYPE_BFLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR,
                     .ResultType = accum ? VK_COMPONENT_TYPE_BFLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR,
                     .saturatingAccumulation = VK_FALSE,
                     .scope = VK_SCOPE_SUBGROUP_KHR,
                  };
               }
            }
         }
         if (size == 16 && pdev->vk.supported_features.shaderBFloat16CooperativeMatrix) {
            for (unsigned swap = 0; swap < 2; ++swap) {
               vk_outarray_append_typed(VkCooperativeMatrixPropertiesKHR, &out, p) {
                  *p = (VkCooperativeMatrixPropertiesKHR) {
                     .sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
                     .MSize = 16, .NSize = 16, .KSize = 16,
                     .AType = swap ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_BFLOAT16_KHR,
                     .BType = swap ? VK_COMPONENT_TYPE_BFLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT16_KHR,
                     .CType = VK_COMPONENT_TYPE_FLOAT32_KHR,
                     .ResultType = VK_COMPONENT_TYPE_FLOAT32_KHR,
                     .saturatingAccumulation = VK_FALSE,
                     .scope = VK_SCOPE_SUBGROUP_KHR,
                  };
               }
            }
         }
      }
   }
   return vk_outarray_status(&out);
}

/* The 16x16x16 configurations, at subgroup scope, in multiples of 16 in
 * each dimension: the compiler splits such a matrix into 16x16 pieces.
 */
VKAPI_ATTR VkResult VKAPI_CALL
hk_GetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(
   VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
   VkCooperativeMatrixFlexibleDimensionsPropertiesNV *pProperties)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkCooperativeMatrixFlexibleDimensionsPropertiesNV,
                          out, pProperties, pPropertyCount);
   if (!pdev->vk.supported_features.cooperativeMatrixFlexibleDimensions ||
       !(pdev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_MATRIX))
      return vk_outarray_status(&out);

   const bool bf16 = pdev->vk.supported_features.shaderBFloat16CooperativeMatrix;
   const struct {
      VkComponentTypeKHR a, b, c;
   } configs[] = {
      {VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR},
      {VK_COMPONENT_TYPE_BFLOAT16_KHR, VK_COMPONENT_TYPE_BFLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR},
      {VK_COMPONENT_TYPE_BFLOAT16_KHR, VK_COMPONENT_TYPE_BFLOAT16_KHR, VK_COMPONENT_TYPE_BFLOAT16_KHR},
      {VK_COMPONENT_TYPE_BFLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR},
      {VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_BFLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(configs); ++i) {
      if (!bf16 && (configs[i].a == VK_COMPONENT_TYPE_BFLOAT16_KHR ||
                    configs[i].b == VK_COMPONENT_TYPE_BFLOAT16_KHR))
         continue;
      vk_outarray_append_typed(VkCooperativeMatrixFlexibleDimensionsPropertiesNV, &out, p) {
         *p = (VkCooperativeMatrixFlexibleDimensionsPropertiesNV) {
            .sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV,
            .MGranularity = 16, .NGranularity = 16, .KGranularity = 16,
            .AType = configs[i].a, .BType = configs[i].b,
            .CType = configs[i].c, .ResultType = configs[i].c,
            .saturatingAccumulation = VK_FALSE,
            .scope = VK_SCOPE_SUBGROUP_KHR,
            .workgroupInvocations = 0,
         };
      }
   }
   return vk_outarray_status(&out);
}

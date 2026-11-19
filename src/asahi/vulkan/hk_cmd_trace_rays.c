/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * vkCmdTraceRaysKHR and its indirect forms.
 *
 * Every form publishes the same record, VkTraceRaysIndirectCommand2KHR, and
 * dispatches the pipeline's kernels with the record's address in the compute
 * root descriptor table.  The direct form fills the record on the host; the
 * indirect forms let one GPU invocation fill in the launch size, because that
 * is the only part that is not known until the command executes.
 *
 * Elsewhere a pipeline is one megakernel; on M5 it has one kernel per
 * ray-generation group.  Which group a trace runs
 * is named by the raygen shader binding table record, and that record is
 * application memory whose contents are only required to be valid when the
 * command executes -- not when it is recorded.  So for a pipeline with several
 * raygen groups the selection is made on the GPU: a setup kernel reads the
 * record's group handle and publishes one VkDispatchIndirectCommand per kernel
 * (the real threadgroup count for the matching kernel, zero for the others),
 * and one indirect dispatch is recorded per kernel.  A single-raygen pipeline
 * needs no selection and keeps the previous direct dispatch.
 */

#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_ray_tracing_pipeline.h"
#include "hk_shader.h"

#include "asahi/lib/agx_bo.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx3_compute_program_core.h"
#include "util/u_debug.h"
#include <inttypes.h>
#include <stdio.h>
#include "libagx_shaders.h"

static VkTraceRaysIndirectCommand2KHR
hk_trace_rays_record(const VkStridedDeviceAddressRegionKHR *raygen,
                     const VkStridedDeviceAddressRegionKHR *miss,
                     const VkStridedDeviceAddressRegionKHR *hit,
                     const VkStridedDeviceAddressRegionKHR *callable,
                     uint32_t width, uint32_t height, uint32_t depth)
{
   return (VkTraceRaysIndirectCommand2KHR){
      .raygenShaderRecordAddress = raygen->deviceAddress,
      .raygenShaderRecordSize = raygen->size,

      .missShaderBindingTableAddress = miss->deviceAddress,
      .missShaderBindingTableSize = miss->size,
      .missShaderBindingTableStride = miss->stride,

      .hitShaderBindingTableAddress = hit->deviceAddress,
      .hitShaderBindingTableSize = hit->size,
      .hitShaderBindingTableStride = hit->stride,

      .callableShaderBindingTableAddress = callable->deviceAddress,
      .callableShaderBindingTableSize = callable->size,
      .callableShaderBindingTableStride = callable->stride,

      .width = width,
      .height = height,
      .depth = depth,
   };
}

static struct hk_ray_tracing_pipeline *
hk_bound_rt_pipeline(struct hk_cmd_buffer *cmd)
{
   struct hk_ray_tracing_pipeline *pipeline = cmd->state.rt.pipeline;

   if (pipeline == NULL ||
       (pipeline->shader == NULL && pipeline->kernel_count == 0)) {
      /* A pipeline library is never executable and a failed create returns
       * VK_NULL_HANDLE.  Recording a trace against either is application
       * error; report it rather than dispatching nothing silently. */
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
      return NULL;
   }

   return pipeline;
}

/* The one kernel a trace dispatches without a GPU-side selection: the
 * megakernel, or the only per-raygen kernel of an M5 pipeline. */
static struct hk_api_shader *
hk_rt_single_kernel(const struct hk_ray_tracing_pipeline *pipeline)
{
   return pipeline->shader ? pipeline->shader : pipeline->kernels[0].shader;
}

/*
 * Single-kick form of the per-raygen selection, used on M5. It is shaped like
 * Metal's indirect command buffers (as the native M5 driver executes them):
 * the host encodes each kernel's direct launch as a template,
 * one select invocation writes the chosen launch into a slot, and the control
 * stream reaches the slot through barrier 0x60000960 and a Stream Link with
 * return, so the CDM sees exactly one launch instead of K indirect ones.
 * Returns false (and records nothing) if a kernel could not be templated.
 */
static bool
hk_trace_rays_single_kick(struct hk_cmd_buffer *cmd,
                          struct hk_ray_tracing_pipeline *pipeline,
                          uint64_t record_addr, uint64_t record_dims_addr,
                          uint64_t dims_addr, uint64_t handles_addr)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const uint32_t count = pipeline->kernel_count;
   const unsigned T = AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE;
   if (T != 44)
      return false;

   struct agx_ptr templates = hk_pool_alloc(cmd, count * T, 64);
   struct agx_ptr slot = hk_pool_alloc(cmd, 13 * 4, 64);
   struct agx_ptr groups =
      hk_pool_alloc(cmd, sizeof(VkDispatchIndirectCommand), 8);
   if (!templates.cpu || !slot.cpu || !groups.cpu)
      return false;
   memset(templates.cpu, 0, count * T);
   memset(groups.cpu, 0, sizeof(VkDispatchIndirectCommand));

   /* Template every kernel first: this records no stream words, only each
    * kernel's IOTO program and resource profile. */
   bool ok = true;
   for (uint32_t k = 0; k < count && ok; ++k) {
      cmd->m5_rt_capture = (uint8_t *)templates.cpu + T * k;
      cmd->m5_rt_captured = false;
      hk_cmd_dispatch_rays(cmd, pipeline->kernels[k].shader, record_addr,
                           groups.gpu, agx_3d(1, 1, 1));
      ok = cmd->m5_rt_captured;
   }
   cmd->m5_rt_capture = NULL;
   cmd->m5_rt_captured = false;
   if (!ok)
      return false;

   uint32_t *words = slot.cpu;
   memset(words, 0, 13 * 4);
   words[0] = 0x60000000; /* the empty barrier slot Metal leads a record with */
   words[12] = 0x80000000; /* Stream Return */

   libagx_trace_rays_select_launch(cmd, agx_1d(1), AGX_BARRIER_ALL,
                                   record_addr, record_dims_addr, groups.gpu,
                                   dims_addr, handles_addr, templates.gpu,
                                   slot.gpu, count,
                                   HK_RT_WORKGROUP_SIZE_X_LOG2);

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, true /* compute */);
   if (!cs)
      return true;
   hk_ensure_cs_has_space(cmd, cs, 0x40);
   const struct agx3_cdm_barrier_info barrier = {
      .chip = dev->dev.chip, .flags = 2, /* 0x60000960, as Metal's */
   };
   const struct agx3_cdm_stream_link_info link = {
      .chip = dev->dev.chip, .target = slot.gpu, .with_return = true,
   };
   uint8_t *out = (uint8_t *)cs->current;
   if (!agx3_cdm_barrier_pack(&barrier, out) ||
       !agx3_cdm_stream_link_pack(&link, out + AGX3_CDM_BARRIER_SIZE)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return true;
   }
   cs->current = (uint32_t *)(out + AGX3_CDM_BARRIER_SIZE +
                              AGX3_CDM_STREAM_LINK_SIZE);
   cs->stats.cmds++;
   return true;
}

/*
 * Publish the per-kernel threadgroup counts from the GPU and record one
 * indirect dispatch per kernel.  `record_addr` is the record the kernels will
 * read; `dims_addr` is where the three launch dimensions live right now;
 * `record_dims_addr` is where they must be copied to, or zero when the record
 * already contains them.
 *
 * On M5 this is done in a single kick (hk_trace_rays_single_kick()); the
 * indirect dispatches remain for the other chips and for a pipeline whose
 * kernels could not be templated.
 */
static void
hk_trace_rays_select_and_dispatch(struct hk_cmd_buffer *cmd,
                                  struct hk_ray_tracing_pipeline *pipeline,
                                  uint64_t record_addr,
                                  uint64_t record_dims_addr,
                                  uint64_t dims_addr)
{
   const uint32_t count = pipeline->kernel_count;

   uint32_t handles[count];
   for (uint32_t k = 0; k < count; ++k)
      handles[k] = pipeline->kernels[k].raygen_handle_index;

   const uint64_t handles_addr =
      hk_pool_upload(cmd, handles, sizeof(handles), 4);
   if (!handles_addr)
      return;

   if (agx_chip_g17_m5(hk_cmd_buffer_device(cmd)->dev.chip) &&
       hk_trace_rays_single_kick(cmd, pipeline, record_addr, record_dims_addr,
                                 dims_addr, handles_addr))
      return;

   const uint64_t groups_addr =
      hk_pool_alloc(cmd, count * sizeof(VkDispatchIndirectCommand), 8).gpu;
   if (!groups_addr)
      return;

   libagx_trace_rays_select_setup(cmd, agx_1d(1), AGX_BARRIER_ALL, record_addr,
                                  record_dims_addr, groups_addr, dims_addr,
                                  handles_addr, count,
                                  HK_RT_WORKGROUP_SIZE_X_LOG2);

   /* Exactly one kernel gets threadgroups; the others are empty dispatches. */
   for (uint32_t k = 0; k < count; ++k) {
      const uint64_t group_count_addr =
         groups_addr + k * sizeof(VkDispatchIndirectCommand);

      hk_cmd_dispatch_rays(cmd, pipeline->kernels[k].shader, record_addr,
                           group_count_addr,
                           agx_grid_indirect(group_count_addr));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdTraceRaysKHR(VkCommandBuffer commandBuffer,
                   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
                   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
                   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
                   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
                   uint32_t width, uint32_t height, uint32_t depth)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   struct hk_ray_tracing_pipeline *pipeline = hk_bound_rt_pipeline(cmd);
   if (!pipeline)
      return;

   if (width == 0 || height == 0 || depth == 0)
      return;

   const VkTraceRaysIndirectCommand2KHR record = hk_trace_rays_record(
      pRaygenShaderBindingTable, pMissShaderBindingTable,
      pHitShaderBindingTable, pCallableShaderBindingTable, width, height,
      depth);

   const uint64_t record_addr =
      hk_pool_upload(cmd, &record, sizeof(record), 8);
   if (!record_addr)
      return;

   if (pipeline->kernel_count > 1) {
      /* The record already holds the dimensions; only the selection is
       * execution-time work. */
      hk_trace_rays_select_and_dispatch(cmd, pipeline, record_addr, 0,
                                        record_addr + HK_RT_CMD_WIDTH);
      return;
   }

   uint32_t groups[3];
   hk_rt_launch_group_count(width, height, depth, groups);

   const VkDispatchIndirectCommand group_count = {
      .x = groups[0],
      .y = groups[1],
      .z = groups[2],
   };
   const uint64_t group_count_addr =
      hk_pool_upload(cmd, &group_count, sizeof(group_count), 8);
   if (!group_count_addr)
      return;

   /* hk_cmd_dispatch_rays() takes the same grid every API dispatch takes:
    * threadgroups, which the compute path multiplies by the workgroup size. */
   hk_cmd_dispatch_rays(cmd, hk_rt_single_kernel(pipeline), record_addr,
                        group_count_addr,
                        agx_3d(groups[0], groups[1], groups[2]));
}

/*
 * The shared tail of both indirect forms.  `record_addr` is the record the
 * kernels will read; `dims_addr` is where the three launch dimensions live
 * right now; `record_dims_addr` is where they must be copied to, or zero when
 * the record already contains them.
 */
static void
hk_trace_rays_indirect(struct hk_cmd_buffer *cmd,
                       struct hk_ray_tracing_pipeline *pipeline,
                       uint64_t record_addr, uint64_t record_dims_addr,
                       uint64_t dims_addr)
{
   if (pipeline->kernel_count > 1) {
      hk_trace_rays_select_and_dispatch(cmd, pipeline, record_addr,
                                        record_dims_addr, dims_addr);
      return;
   }

   const uint64_t group_count_addr =
      hk_pool_alloc(cmd, sizeof(VkDispatchIndirectCommand), 8).gpu;
   if (!group_count_addr)
      return;

   libagx_trace_rays_indirect_setup(cmd, agx_1d(1), AGX_BARRIER_ALL,
                                    record_dims_addr, group_count_addr,
                                    dims_addr, HK_RT_WORKGROUP_SIZE_X_LOG2);

   hk_cmd_dispatch_rays(cmd, hk_rt_single_kernel(pipeline), record_addr,
                        group_count_addr, agx_grid_indirect(group_count_addr));
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdTraceRaysIndirectKHR(
   VkCommandBuffer commandBuffer,
   const VkStridedDeviceAddressRegionKHR *pRaygenShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pMissShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pHitShaderBindingTable,
   const VkStridedDeviceAddressRegionKHR *pCallableShaderBindingTable,
   VkDeviceAddress indirectDeviceAddress)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   struct hk_ray_tracing_pipeline *pipeline = hk_bound_rt_pipeline(cmd);
   if (!pipeline)
      return;

   /* The application's buffer holds only a VkTraceRaysIndirectCommandKHR, so
    * the shader binding tables are assembled here and the three dimensions are
    * copied in by the setup kernel. */
   const VkTraceRaysIndirectCommand2KHR record = hk_trace_rays_record(
      pRaygenShaderBindingTable, pMissShaderBindingTable,
      pHitShaderBindingTable, pCallableShaderBindingTable, 0, 0, 0);

   const uint64_t record_addr =
      hk_pool_upload(cmd, &record, sizeof(record), 8);
   if (!record_addr)
      return;

   hk_trace_rays_indirect(cmd, pipeline, record_addr,
                          record_addr + HK_RT_CMD_WIDTH, indirectDeviceAddress);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdTraceRaysIndirect2KHR(VkCommandBuffer commandBuffer,
                            VkDeviceAddress indirectDeviceAddress)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   struct hk_ray_tracing_pipeline *pipeline = hk_bound_rt_pipeline(cmd);
   if (!pipeline)
      return;

   /* The application supplied the whole record, so the kernels can read it
    * where it is; only the threadgroup counts still have to be derived. */
   hk_trace_rays_indirect(cmd, pipeline, indirectDeviceAddress, 0,
                          indirectDeviceAddress + HK_RT_CMD_WIDTH);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdSetRayTracingPipelineStackSizeKHR(VkCommandBuffer commandBuffer,
                                        uint32_t pipelineStackSize)
{
   /* Each kernel's stack is static scratch inside its compiled binary, sized
    * at pipeline creation for maxPipelineRayRecursionDepth.  A dynamic size
    * cannot shrink it and must not grow it, so this is deliberately a no-op
    * rather than a partial implementation.  Documented in
    * docs/G17P_RAYTRACING_PIPELINE_ABI.md. */
}

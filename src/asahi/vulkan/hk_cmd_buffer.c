/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_cmd_buffer.h"
#include "util/format/u_format.h"
#include "hk_frame_anatomy.h"
#include "hk_ppp_state.h"
#include "asahi/compiler/agx3_graphics_program.h"
#include "hk_g17p_patch.h"
#include "asahi/lib/agx_scs_layout.h"

#include "agx_abi.h"
#include "agx_bo.h"
#include "agx_device.h"
#include "agx_linker.h"
#include "agx_tilebuffer.h"
#include "agx_usc.h"
#include "decode.h"
#include "hk_agx3_ioto_program_core.h"
#include "asahi/compiler/agx3_compute_program.h"
#include "asahi/compiler/agx_debug.h"
#include "hk_agx3_vdm_decode.h"
#include "asahi/lib/agx_ioto_load_core.h"
#include "hk_buffer.h"
#include "hk_acceleration_structure.h"
#include "hk_cmd_pool.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "libagx_shaders.h"

#include "util/cache_ops.h"
#include "util/os_misc.h"
#include "util/u_atomic.h"
#include <inttypes.h>
#include <stdio.h>
#include <unistd.h>
#include "util/log.h"
#include "util/hash_table.h"
#include "util/set.h"
#include "util/u_debug.h"
#include "util/os_time.h"

#include <inttypes.h>
#include <stdio.h>

#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx_ppp_hal300_depth_stencil_core.h"
#include "asahi/lib/agx_ppp_w_clamp_cull2_core.h"
#include "asahi/lib/agx_vdm_barrier_core.h"
#include "asahi/lib/agx_vdm_stream_link_core.h"
#include "asahi/lib/agx_m5_va.h"
#include "asahi/lib/agx_vdm_stream_terminate_core.h"
#include "asahi/lib/agx_va_runtime_hole.h"
#include "libagx_dgc.h"
#include "pool.h"
#include "shader_enums.h"
#include "vk_pipeline_layout.h"
#include "vk_common_entrypoints.h"
#include "vk_synchronization.h"

#include "util/list.h"
#include "util/macros.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"

static uint64_t
hk_g17p_vdm_lookup(struct hk_cmd_buffer *cmd, uint64_t primary, bool bind)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!hk_g17p_vdm_low_window_enabled(&dev->dev))
      return primary;

   struct hk_vdm_allocation *last = cmd->g17p_vdm_last_allocation;
   if (last && primary >= last->addr && primary - last->addr < last->size) {
      uint32_t compact = 0;
      if (agx_g17p_vdm_alias_compact(
             last->alias + (primary - last->addr), &compact))
         return compact;
   }

   /* Allocations never overlap, so the search order cannot change the
    * answer. New control streams live in the most recent allocation: search
    * from the tail so the lookup at each stream start stays constant time
    * instead of growing with every render pass recorded. */
   list_for_each_entry_rev(struct hk_vdm_allocation, allocation,
                           &cmd->vdm_allocations, link) {
      if (primary < allocation->addr ||
          primary - allocation->addr >= allocation->size)
         continue;

      const uint64_t alias =
         allocation->alias + (primary - allocation->addr);
      uint32_t compact = 0;
      if (agx_g17p_vdm_alias_compact(alias, &compact)) {
         /* Recording owns the command buffer, but submission may translate a
          * SIMULTANEOUS_USE buffer from several queues at once.  The latter
          * must remain read-only: it may consume the stable last recording
          * span above, never race to replace it here. */
         if (bind)
            cmd->g17p_vdm_last_allocation = allocation;
         return compact;
      }
   }

   /* The constrained G17P main heap is wholly inside the lower-root compact
    * VDM aperture. Control-stream roots and links can therefore name their
    * primary mapping directly. Keep the scarce +8..+32 MiB aliases above for
    * draw-state payloads whose independent HAL200 fields are narrower. */
   uint64_t direct = 0;
   if (agx_g17p_vdm_payload_compact(primary, &direct))
      return direct;

   {
      list_for_each_entry(struct hk_cmd_bo, bo, &cmd->uploader.main.bos, link) {
         const uint64_t start = bo->bo->va->addr;
         if (primary < start || primary - start >= bo->bo->size)
            continue;

         if (bind) {
            const VkResult result =
               hk_cmd_bo_ensure_vdm_alias(hk_cmd_buffer_pool(cmd), bo);
            if (result != VK_SUCCESS) {
               vk_command_buffer_set_error(&cmd->vk, result);
               return 0;
            }
         }

         const uint64_t alias = bo->g17p_vdm_alias + (primary - start);
         uint32_t compact = 0;
         if (agx_g17p_vdm_alias_compact(alias, &compact))
            return compact;
         break;
      }
   }

   mesa_loge("G17P VDM address %#" PRIx64 " has no low command-BO alias",
             primary);
   return 0;
}

uint64_t
hk_g17p_vdm_compact(struct hk_cmd_buffer *cmd, uint64_t primary)
{
   return hk_g17p_vdm_lookup(cmd, primary, true);
}

uint64_t
hk_g17p_vdm_alias(struct hk_cmd_buffer *cmd, uint64_t primary)
{
   /* Submission may read a simultaneous-use CB from multiple queues. All
    * VDM roots/links were mapped during recording; never mutate them here. */
   const uint64_t compact = hk_g17p_vdm_lookup(cmd, primary, false);
   return compact ? AGX_G17P_VDM_USER_BASE + compact : 0;
}

static void
hk_descriptor_state_fini(struct hk_cmd_buffer *cmd,
                         struct hk_descriptor_state *desc)
{
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);

   for (unsigned i = 0; i < HK_MAX_SETS; i++) {
      vk_free(&pool->vk.alloc, desc->push[i]);
      desc->push[i] = NULL;
   }
}

struct hk_m5_rt_state_diag_entry {
   uint8_t *cpu;
   uint64_t gpu, size;
   uint8_t *initial;
};

/* Opt-in diagnostic (ASAHI_M5_RT_STATE_DIAG=<dir>): remember the initial
 * contents of each SCS state block this recording allocates, so the bytes the
 * GPU changed can be reported once the application has finished with the
 * command buffer. No effect on the recorded commands. */
void
hk_m5_rt_state_diag_record(struct hk_cmd_buffer *cmd, struct agx_ptr state,
                           uint64_t size)
{
   struct hk_m5_rt_state_diag_entry e = {
      .cpu = state.cpu, .gpu = state.gpu, .size = size, .initial = malloc(size),
   };
   if (!e.initial)
      return;
   memcpy(e.initial, state.cpu, size);
   util_dynarray_append(&cmd->m5_rt_state.diag, e);
}

static void
hk_m5_rt_state_diag_flush(struct hk_cmd_buffer *cmd)
{
   const char *dir = os_get_option("ASAHI_M5_RT_STATE_DIAG");
   static unsigned seq;
   util_dynarray_foreach(&cmd->m5_rt_state.diag,
                         struct hk_m5_rt_state_diag_entry, e) {
      const unsigned n = p_atomic_inc_return(&seq);
      char path[512];
      snprintf(path, sizeof(path), "%s/state-%d-%u.txt", dir ? dir : "/tmp",
               (int)getpid(), n);
      FILE *f = fopen(path, "w");
      if (f) {
         uint64_t changed = 0, ranges = 0, start = UINT64_MAX, last = 0;
         fprintf(f, "gpu 0x%" PRIx64 " size 0x%" PRIx64 "\n", e->gpu, e->size);
         for (uint64_t o = 0; o < e->size; o += 16) {
            if (!memcmp(e->cpu + o, e->initial + o, 16))
               continue;
            changed += 16;
            if (start != UINT64_MAX && o > last + 256) {
               fprintf(f, "range 0x%" PRIx64 " 0x%" PRIx64 "\n", start, last + 16);
               ranges++;
               start = UINT64_MAX;
            }
            if (start == UINT64_MAX)
               start = o;
            last = o;
         }
         if (start != UINT64_MAX) {
            fprintf(f, "range 0x%" PRIx64 " 0x%" PRIx64 "\n", start, last + 16);
            ranges++;
         }
         fprintf(f, "changed16 %" PRIu64 " ranges %" PRIu64 "\n", changed, ranges);
         fclose(f);
      }
      if (n <= 4) {
         snprintf(path, sizeof(path), "%s/state-%d-%u.final.bin", dir ? dir : "/tmp",
                  (int)getpid(), n);
         f = fopen(path, "wb");
         if (f) {
            fwrite(e->cpu, 1, e->size, f);
            fclose(f);
         }
         snprintf(path, sizeof(path), "%s/state-%d-%u.initial.bin", dir ? dir : "/tmp",
                  (int)getpid(), n);
         f = fopen(path, "wb");
         if (f) {
            fwrite(e->initial, 1, e->size, f);
            fclose(f);
         }
      }
      free(e->initial);
   }
   util_dynarray_fini(&cmd->m5_rt_state.diag);
   memset(&cmd->m5_rt_state.diag, 0, sizeof(cmd->m5_rt_state.diag));
}

static void
hk_free_resettable_cmd_buffer(struct hk_cmd_buffer *cmd)
{
   if (cmd->m5_rt_state.diag.size)
      hk_m5_rt_state_diag_flush(cmd);

   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);
   struct hk_device *dev = hk_cmd_pool_device(pool);

   /* The cached span is only valid while this recording owns its leases. */
   cmd->g17p_vdm_last_allocation = NULL;

   hk_descriptor_state_fini(cmd, &cmd->state.gfx.descriptors);
   hk_descriptor_state_fini(cmd, &cmd->state.cs.descriptors);
   hk_descriptor_state_fini(cmd, &cmd->state.rt.descriptors);

   hk_cmd_pool_free_gen1_as_chunks(pool, &cmd->gen1_as_table_chunks);

   hk_cmd_pool_free_bo_list(pool, HK_CMD_BO_MAIN, &cmd->uploader.main.bos);
   hk_cmd_pool_free_bo_list(pool, HK_CMD_BO_USC, &cmd->uploader.usc.bos);
   hk_cmd_pool_free_bo_list(pool, HK_CMD_BO_ROOT, &cmd->uploader.root.bos);
   list_for_each_entry_safe(struct hk_vdm_allocation, allocation,
                            &cmd->vdm_allocations, link) {
      list_del(&allocation->link);
      hk_cmd_pool_free_vdm(pool, allocation);
   }

   list_for_each_entry_safe(struct hk_cs, it, &cmd->control_streams, node) {
      list_del(&it->node);
      hk_cs_destroy(it);
   }

   hk_ioto_arena_finish(cmd);
   hk_cmd_buffer_release_bos(cmd, &dev->dev);
}

static void
hk_destroy_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer)
{
   struct hk_cmd_buffer *cmd =
      container_of(vk_cmd_buffer, struct hk_cmd_buffer, vk);
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);

   hk_free_resettable_cmd_buffer(cmd);
   util_dynarray_fini(&cmd->large_bos);
   free(cmd->retained_set);
   if (cmd->m5_pushed_images)
      _mesa_set_destroy(cmd->m5_pushed_images, NULL);
   util_dynarray_fini(&cmd->sync_pass_records);
   util_dynarray_fini(&cmd->oq_ends);
   vk_command_buffer_finish(&cmd->vk);
   vk_free(&pool->vk.alloc, cmd);
}

static VkResult
hk_create_cmd_buffer(struct vk_command_pool *vk_pool,
                     VkCommandBufferLevel level,
                     struct vk_command_buffer **cmd_buffer_out)
{
   struct hk_cmd_pool *pool = container_of(vk_pool, struct hk_cmd_pool, vk);
   struct hk_device *dev = hk_cmd_pool_device(pool);
   struct hk_cmd_buffer *cmd;
   VkResult result;

   cmd = vk_zalloc(&pool->vk.alloc, sizeof(*cmd), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (cmd == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = vk_command_buffer_init_with_params(&cmd->vk,
                                               &(struct vk_command_buffer_init_params) {
                                                  .pool = &pool->vk,
                                                  .ops = &hk_cmd_buffer_ops,
                                                  .level = level,
                                                  .needs_cmd_queue = level == VK_COMMAND_BUFFER_LEVEL_SECONDARY,
                                               });
   if (result != VK_SUCCESS) {
      vk_free(&pool->vk.alloc, cmd);
      return result;
   }

   cmd->large_bos = UTIL_DYNARRAY_INIT;

   cmd->vk.dynamic_graphics_state.vi = &cmd->state.gfx._dynamic_vi;
   cmd->vk.dynamic_graphics_state.ms.sample_locations =
      &cmd->state.gfx._dynamic_sl;

   list_inithead(&cmd->uploader.main.bos);
   list_inithead(&cmd->uploader.usc.bos);
   list_inithead(&cmd->uploader.root.bos);
   list_inithead(&cmd->vdm_allocations);
   list_inithead(&cmd->gen1_as_table_chunks);
   list_inithead(&cmd->control_streams);

   *cmd_buffer_out = &cmd->vk;

   return VK_SUCCESS;
}

static void
hk_reset_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer,
                    UNUSED VkCommandBufferResetFlags flags)
{
   struct hk_cmd_buffer *cmd =
      container_of(vk_cmd_buffer, struct hk_cmd_buffer, vk);

   vk_command_buffer_reset(&cmd->vk);
   hk_free_resettable_cmd_buffer(cmd);

   cmd->uploader.main.map = NULL;
   cmd->uploader.main.base = 0;
   cmd->uploader.main.offset = 0;
   cmd->uploader.usc.map = NULL;
   cmd->uploader.usc.base = 0;
   cmd->uploader.usc.offset = 0;
   cmd->uploader.root.map = NULL;
   cmd->uploader.root.base = 0;
   cmd->uploader.root.offset = 0;
   cmd->uploader.root.capacity = 0;
   cmd->uploader.vdm.map = NULL;
   cmd->uploader.vdm.base = 0;
   cmd->uploader.vdm.offset = 0;
   cmd->uploader.vdm.capacity = 0;
   cmd->uploader.vdm.vdm_alias = 0;
   cmd->uploader.streams.map = NULL;
   cmd->uploader.streams.base = 0;
   cmd->uploader.streams.offset = 0;

   cmd->current_cs.gfx = NULL;
   cmd->current_cs.cs = NULL;
   cmd->current_cs.post_gfx = NULL;
   cmd->current_cs.pre_gfx = NULL;

   assert(!cmd->in_meta);
   assert(!cmd->conditional_meta);
   cmd->geom_index_buffer = 0;
   cmd->geom_index_count = 0;
   cmd->geom_instance_count = 0;
   cmd->uses_heap = false;
   hk_sync_reset(cmd);
   if (cmd->m5_pushed_images)
      _mesa_set_clear(cmd->m5_pushed_images, NULL);
   util_dynarray_clear(&cmd->oq_ends);

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (dev->dev.params.gpu_generation == 13 ||
       dev->dev.params.gpu_generation == 14) {
      /* These native graphics caches are zero-initialized at allocation and
       * never used on AGX2. Reset the common state on either side without
       * clearing their 8.5 KiB again for every recording. */
      const size_t first =
         offsetof(struct hk_cmd_buffer, state.gfx.g17p_ioto) -
         offsetof(struct hk_cmd_buffer, state);
      const size_t end =
         offsetof(struct hk_cmd_buffer, state.gfx.g17p_spilled_rt_descs) +
         sizeof(cmd->state.gfx.g17p_spilled_rt_descs) -
         offsetof(struct hk_cmd_buffer, state);
      memset(&cmd->state, 0, first);
      memset((char *)&cmd->state + end, 0, sizeof(cmd->state) - end);
   } else {
      memset(&cmd->state, 0, sizeof(cmd->state));
      for (unsigned i = 0; i < HK_G17P_BG_EOT_CACHE_ENTRIES; ++i)
         cmd->g17p_bg_eot_cache[i].valid = false;
      cmd->g17p_bg_eot_cache_next = 0;
   }
   /* Nothing has been recorded since this reset. */
   cmd->g17p_pristine = true;
}

const struct vk_command_buffer_ops hk_cmd_buffer_ops = {
   .create = hk_create_cmd_buffer,
   .reset = hk_reset_cmd_buffer,
   .destroy = hk_destroy_cmd_buffer,
};

VKAPI_ATTR void VKAPI_CALL
hk_CmdExecuteCommands(VkCommandBuffer commandBuffer,
                      uint32_t commandBufferCount,
                      const VkCommandBuffer *pCommandBuffers)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   const struct vk_device_dispatch_table *disp =
      cmd->vk.base.device->command_dispatch_table;
   const struct hk_conditional_render_state primary_conditional =
      cmd->state.conditional;

   hk_flush_deferred_clears(cmd);

   /* Secondary command buffers are materialized by replaying their common
    * command queues into the primary.  Their last bound state is therefore
    * also the driver's cached state in the primary after replay.  Vulkan
    * makes that state undefined (except for the active render pass), so an
    * application is required to bind it again.  Invalidate here so binding
    * the same object used by the secondary cannot be incorrectly elided.
    */
   for (uint32_t i = 0; i < commandBufferCount; ++i) {
      VK_FROM_HANDLE(hk_cmd_buffer, secondary, pCommandBuffers[i]);

      /* The queued commands replay into `cmd`, so establish the conditional
       * state promised by the secondary's inheritance declaration. Explicit
       * begin/end scopes recorded inside the secondary may change this state
       * during replay; always restore the primary at the command-buffer
       * boundary. */
      cmd->state.conditional = hk_conditional_render_replay_state(
         &primary_conditional, secondary->state.conditional.inherited);
      vk_cmd_queue_execute(&secondary->vk.cmd_queue, commandBuffer, disp);
      cmd->state.conditional = primary_conditional;
   }

   hk_cmd_invalidate_graphics_state(cmd);
   hk_cmd_invalidate_compute_state(cmd);
}

static VkResult
hk_cmd_buffer_alloc_bo(struct hk_cmd_buffer *cmd, enum hk_cmd_bo_type type,
                       struct hk_cmd_bo **bo_out)
{
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);
   VkResult result = hk_cmd_pool_alloc_bo(pool, type, bo_out);
   if (result != VK_SUCCESS)
      return result;

   if (type == HK_CMD_BO_USC)
      list_addtail(&(*bo_out)->link, &cmd->uploader.usc.bos);
   else if (type == HK_CMD_BO_ROOT)
      list_addtail(&(*bo_out)->link, &cmd->uploader.root.bos);
   else
      list_addtail(&(*bo_out)->link, &cmd->uploader.main.bos);

   return VK_SUCCESS;
}

struct agx_ptr
hk_pool_alloc_from_slow(struct hk_cmd_buffer *cmd,
                        struct hk_uploader *uploader, uint32_t size,
                        uint32_t alignment, enum hk_cmd_bo_type type)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* Specially handle large allocations owned by the command buffer, e.g. used
    * for statically allocated vertex output buffers with geometry shaders.
    */
   if (size > HK_CMD_BO_SIZE) {
      enum agx_bo_flags flags =
         type == HK_CMD_BO_USC    ? AGX_BO_LOW_VA
         : type == HK_CMD_BO_ROOT ? AGX_BO_WRITEBACK
                                  : 0;
      struct agx_bo *bo =
         agx_bo_create(&dev->dev, size, 0, flags, "Large pool allocation");

      if (unlikely(bo == NULL)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return (struct agx_ptr){0};
      }

      util_dynarray_append(&cmd->large_bos, bo);
      return (struct agx_ptr){
         .gpu = bo->va->addr,
         .cpu = agx_bo_map(bo),
      };
   }

   assert(size <= HK_CMD_BO_SIZE);
   assert(alignment > 0);

   uint32_t offset = align(uploader->offset, alignment);

   assert(offset <= HK_CMD_BO_SIZE);
   if (uploader->map != NULL && size <= HK_CMD_BO_SIZE - offset) {
      uploader->offset = offset + size;

      return (struct agx_ptr){
         .gpu = uploader->base + offset,
         .cpu = uploader->map + offset,
      };
   }

   struct hk_cmd_bo *bo;
   VkResult result = hk_cmd_buffer_alloc_bo(cmd, type, &bo);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return (struct agx_ptr){0};
   }
   bo->stream_only = type == HK_CMD_BO_MAIN &&
                     uploader == &cmd->uploader.streams;

   /* Pick whichever of the current upload BO and the new BO will have more
    * room left to be the BO for the next upload.  If our upload size is
    * bigger than the old offset, we're better off burning the whole new
    * upload BO on this one allocation and continuing on the current upload
    * BO.
    */
   if (uploader->map == NULL || size < uploader->offset) {
      uploader->map = agx_bo_map(bo->bo);
      uploader->base = bo->bo->va->addr;
      uploader->offset = size;
   }

   return (struct agx_ptr){
      .gpu = bo->bo->va->addr,
      .cpu = bo->map,
   };
}

static __attribute__((hot)) struct agx_ptr
hk_pool_alloc_internal(struct hk_cmd_buffer *cmd, uint32_t size,
                       uint32_t alignment, bool usc)
{
   struct hk_uploader *uploader =
      usc ? &cmd->uploader.usc : &cmd->uploader.main;
   return hk_pool_alloc_from(cmd, uploader, size, alignment,
                             usc ? HK_CMD_BO_USC : HK_CMD_BO_MAIN);
}

struct agx_ptr
hk_pool_cs_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* A G17P compute root consumes a whole command BO. Keep control-stream
    * roots on their own bump cursor so allocating one cannot evict a partially
    * used main-upload BO and move that allocator's slow path into the next
    * application command. Ownership remains on the ordinary main BO list. */
   struct hk_uploader *uploader = agx_native_graphics(&dev->dev)
                                     ? &cmd->uploader.streams
                                     : &cmd->uploader.main;
   return hk_pool_alloc_from(cmd, uploader, size, alignment, HK_CMD_BO_MAIN);
}


struct agx_ptr
hk_pool_vdm_primary_alloc(struct hk_cmd_buffer *cmd, uint32_t size,
                          uint32_t alignment)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!hk_g17p_vdm_low_window_enabled(&dev->dev))
      return hk_pool_alloc(cmd, size, alignment);

   struct hk_uploader *uploader = &cmd->uploader.vdm;
   uint32_t offset = align(uploader->offset, alignment);
   if (uploader->map && offset <= uploader->capacity &&
       size <= uploader->capacity - offset) {
      uploader->offset = offset + size;
      return (struct agx_ptr){
         .gpu = uploader->base + offset,
         .cpu = uploader->map + offset,
      };
   }

   const uint32_t chunk_size =
      align(MAX2(size, HK_G17P_VDM_UPLOAD_CHUNK_SIZE),
            HK_G17P_VDM_UPLOAD_CHUNK_SIZE);
   const uint32_t chunk_alignment =
      MAX2(alignment, HK_G17P_VDM_UPLOAD_CHUNK_SIZE);
   struct agx_ptr chunk = {0};
   struct hk_vdm_allocation *allocation = NULL;
   const VkResult result = hk_cmd_pool_alloc_vdm(
      hk_cmd_buffer_pool(cmd), chunk_size, chunk_alignment, &chunk,
      &allocation);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return (struct agx_ptr){0};
   }

   list_addtail(&allocation->link, &cmd->vdm_allocations);
   cmd->g17p_vdm_last_allocation = allocation;
   uploader->map = chunk.cpu;
   uploader->base = chunk.gpu;
   uploader->offset = size;
   uploader->capacity = chunk_size;
   uploader->vdm_alias = allocation->alias + (chunk.gpu - allocation->addr);
   return chunk;
}

__attribute__((hot)) uint64_t
hk_pool_upload(struct hk_cmd_buffer *cmd, const void *data, uint32_t size,
               uint32_t alignment)
{
   struct agx_ptr T = hk_pool_alloc(cmd, size, alignment);
   if (unlikely(T.cpu == NULL))
      return 0;

   memcpy(T.cpu, data, size);
   return T.gpu;
}

uint64_t
hk_g17p_texture_state_pointer(struct hk_cmd_buffer *cmd, uint64_t texture_va,
                              const void *descriptor, uint32_t available_B)
{
   if (!(texture_va & (HK_G17P_TEXTURE_STATE_ALIGN - 1)))
      return texture_va;

   if (!descriptor || available_B < HK_TEXTURE_DESCRIPTOR_AGX3_SIZE)
      return 0;

   return hk_pool_upload(
      cmd, descriptor,
      MIN2(available_B, (uint32_t)sizeof(struct hk_sampled_image_descriptor)),
      HK_G17P_TEXTURE_STATE_ALIGN);
}

uint64_t
hk_pool_usc_upload(struct hk_cmd_buffer *cmd, const void *data, uint32_t size,
                   uint32_t alignment)
{
   struct agx_ptr T = hk_pool_usc_alloc(cmd, size, alignment);
   if (unlikely(T.cpu == NULL))
      return 0;

   memcpy(T.cpu, data, size);
   return T.gpu;
}

void *
hk_cmd_buffer_emit_cs_jump(struct hk_cmd_buffer *cmd, void *current,
                           uint64_t target, bool vdm)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (vdm && hk_g17p_vdm_low_window_enabled(&dev->dev))
      target = hk_g17p_vdm_compact(cmd, target);
   else if (vdm && agx_chip_g17_m5(dev->dev.chip)) {
      if (!agx_m5_vdm_compact(target, &target)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return current;
      }
   }

   switch (agx_chip_command_stream_generation(dev->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      return agx_cs_jump(current, target, vdm);

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300:
      if (vdm) {
         struct agx_vdm_stream_link_core link;
         if (agx_vdm_stream_link_core_pack(dev->dev.chip, target, &link)) {
            memcpy(current, link.bytes, link.length);
            return (uint8_t *)current + link.length;
         }
      } else {
         const struct agx3_cdm_stream_link_info info = {
            .chip = dev->dev.chip,
            .target = target,
            .with_return = false,
         };
         uint8_t link[AGX3_CDM_STREAM_LINK_SIZE];
         if (agx3_cdm_stream_link_pack(&info, link)) {
            memcpy(current, link, sizeof(link));
            return (uint8_t *)current + sizeof(link);
         }
      }
      break;

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      break;
   }

   mesa_loge("no grounded %s stream link for %s", vdm ? "VDM" : "CDM",
             agx_chip_name(dev->dev.chip));
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   return current;
}

static void *
hk_cmd_buffer_emit_cdm_terminate(struct hk_cmd_buffer *cmd, void *current)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   switch (agx_chip_command_stream_generation(dev->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      return agx_cdm_terminate(current);

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
      if (agx3_cdm_stream_terminate_pack(dev->dev.chip, terminate)) {
         memcpy(current, terminate, sizeof(terminate));
         return (uint8_t *)current + sizeof(terminate);
      }
      break;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      break;
   }

   mesa_loge("no grounded CDM stream terminator for %s",
             agx_chip_name(dev->dev.chip));
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   return current;
}

void *
hk_cmd_buffer_emit_vdm_terminate(struct hk_cmd_buffer *cmd, void *current)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* Draw emission leaves the stream open for further state, draws or links.
    * EndCommandStream supplies exactly one terminal record here. Do not infer
    * termination from the preceding dword: it may be a draw's vertex_start.
    */
   switch (agx_chip_command_stream_generation(dev->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      return agx_vdm_terminate(current);

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      struct agx_vdm_stream_terminate_core terminate;
      if (agx_vdm_stream_terminate_core_pack(dev->dev.chip, &terminate)) {
         memcpy(current, terminate.bytes, terminate.length);
         return (uint8_t *)current + terminate.length;
      }
      break;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      break;
   }

   mesa_loge("no grounded VDM stream terminator for %s",
             agx_chip_name(dev->dev.chip));
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   return current;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                      const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* An explicit vkResetCommandBuffer/pool reset already left this buffer in
    * exactly the state the implicit reset below produces (nothing can be
    * recorded between that reset and this begin), so doing it again would
    * only repeat the teardown and the full state memset. */
   if (!cmd->g17p_pristine)
      hk_reset_cmd_buffer(&cmd->vk, 0);
   cmd->g17p_pristine = false;
   vk_command_buffer_begin(&cmd->vk, pBeginInfo);

   if (cmd->vk.level == VK_COMMAND_BUFFER_LEVEL_SECONDARY &&
       pBeginInfo->pInheritanceInfo) {
      const VkCommandBufferInheritanceConditionalRenderingInfoEXT *conditional =
         vk_find_struct_const(
            pBeginInfo->pInheritanceInfo->pNext,
            COMMAND_BUFFER_INHERITANCE_CONDITIONAL_RENDERING_INFO_EXT);

      cmd->state.conditional.inherited =
         conditional && conditional->conditionalRenderingEnable;
   }

   perf_debug(cmd, "Begin command buffer");
   hk_cmd_buffer_begin_compute(cmd, pBeginInfo);
   hk_cmd_buffer_begin_graphics(cmd, pBeginInfo);

   return VK_SUCCESS;
}

/*
 * Merge adjacent compute control streams. Except for reading timestamps, there
 * is no reason to submit two CDM streams back-to-back in the same command
 * buffer. However, it is challenging to avoid constructing such sequences due
 * to the gymnastics required to reorder compute around graphics. Merging at
 * EndCommandBuffer is cheap O(# of control streams) and lets us get away with
 * the sloppiness. On M5 a stream that waits for a render the one before it
 * does not wait for stays separate (hk_sync_merge_delays()): merged, the
 * earlier work would wait for that render too.
 */
static void
merge_control_streams(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *last = NULL;

   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      if (cs->type == HK_CS_CDM && last && last->type == HK_CS_CDM &&
          !last->timestamp.end.handle &&
          !(last->timestamp.start.handle && cs->g17p_timestamp_query) &&
          !(last->timestamp.start.handle && cs->timestamp.start.handle) &&
          !hk_sync_merge_delays(cmd, last, cs)) {

         hk_sync_merge(cmd, last, cs);
         hk_cs_merge_cdm(last, cs);
         list_del(&cs->node);
         hk_cs_destroy(cs);
      } else {
         last = cs;
      }
   }
}

static void
hk_cmd_buffer_publish_root_uploads(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!agx_native_graphics(&dev->dev) ||
       list_is_empty(&cmd->uploader.root.bos))
      return;

   if (util_has_cache_ops()) {
      /* Every root range was cleaned when written. Complete those cache
       * operations once, after recording has made all roots immutable. */
      util_post_flush_fence();
      return;
   }

   list_for_each_entry(struct hk_cmd_bo, bo, &cmd->uploader.root.bos, link) {
      agx_bo_assert_range(bo->bo, 0, bo->bo->size);
   }
}

/* A dedicated stream BO can reserve much more space than the stream wrote.
 * For an unlinked stream, its current pointer and end delimit the written
 * records plus the existing prefetch padding. Removed inline-merged streams
 * are no longer referenced. Linked layouts retain the conservative path. */
static bool
hk_cmd_bo_clean_unlinked_streams(struct hk_cmd_buffer *cmd,
                                 const struct hk_cmd_bo *bo)
{
   const uintptr_t base = (uintptr_t)bo->map;
   if (bo->bo->size > UINTPTR_MAX - base)
      return false;
   const uintptr_t limit = base + bo->bo->size;

   list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
      const uintptr_t start = (uintptr_t)cs->start;
      if (start < base || start >= limit)
         continue;
      const uintptr_t current = (uintptr_t)cs->current;
      const uintptr_t end = (uintptr_t)cs->end;
      if (current < start || current > end || end > limit)
         return false;
      const size_t padding = MIN2(end - current, HK_CS_SPACE_PADDING);
      const size_t bytes = current - start + padding;
      if (bytes)
         util_flush_range_no_fence((void *)start, bytes);
   }
   return true;
}

/* Clean everything recording wrote to CPU-cached command memory (see
 * hk_g17p_cached_cmd_memory()); hk_cmd_buffer_publish_root_uploads() and
 * queue submission fence the cleans. */
static void
hk_cmd_buffer_publish_cached_uploads(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!hk_g17p_cached_cmd_memory(&dev->dev))
      return;

   bool linked_stream = false;
   list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node)
      linked_stream |= cs->stream_linked;

   /* An uploader's current BO is written up to its offset; a BO it moved
    * on from may have been written anywhere. Dedicated unlinked stream BOs
    * instead have exact written extents, independent of reserved capacity. */
   const struct hk_uploader *current[] = {
      &cmd->uploader.main, &cmd->uploader.streams, &cmd->uploader.usc,
   };
   for (unsigned l = 0; l < 2; ++l) {
      struct list_head *bos =
         l == 0 ? &cmd->uploader.main.bos : &cmd->uploader.usc.bos;
      list_for_each_entry(struct hk_cmd_bo, bo, bos, link) {
         if (bo->stream_only && !linked_stream &&
             hk_cmd_bo_clean_unlinked_streams(cmd, bo))
            continue;
         size_t size = bo->bo->size;
         for (unsigned u = 0; u < ARRAY_SIZE(current); ++u) {
            if (current[u]->map == bo->map)
               size = MIN2(size, current[u]->offset);
         }
         if (size)
            util_flush_range_no_fence(bo->map, size);
      }
   }
   list_for_each_entry(struct hk_vdm_allocation, allocation,
                       &cmd->vdm_allocations, link)
      hk_cmd_pool_clean_vdm(allocation);
   hk_ioto_arena_clean(cmd);
   util_post_flush_fence();
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   assert(cmd->current_cs.gfx == NULL && cmd->current_cs.pre_gfx == NULL &&
          "must end rendering before ending the command buffer");

   perf_debug(cmd, "End command buffer");
   hk_flush_deferred_clears(cmd);
   hk_sync_flush_pending_timestamp(cmd);
   hk_cmd_buffer_end_compute(cmd);
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);
   hk_resolve_deferred_oq_available(cmd);

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (likely(!(dev->dev.debug & AGX_DBG_NOMERGE))) {
      merge_control_streams(cmd);
   }

   /* We cannot terminate CDM control streams until after merging, since merging
    * needs to append stream links late. Now that we've merged, insert all the
    * missing stream terminates.
    */
   list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
      if (cs->type == HK_CS_CDM) {
         cs->current = hk_cmd_buffer_emit_cdm_terminate(cmd, cs->current);
      }
   }

   /* G17P roots use cached mappings. Publish the completed recording once,
    * after every root table is immutable and before submission can consume
    * any address embedded in the command streams. */
   hk_cmd_buffer_publish_cached_uploads(cmd);
   hk_cmd_buffer_publish_root_uploads(cmd);

   hk_cmd_print_upload_stats(cmd);
   return vk_command_buffer_end(&cmd->vk);
}

static bool
hk_dependency_has_host_read(const VkDependencyInfo *info)
{
   for (uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
      if (info->pMemoryBarriers[i].dstAccessMask & VK_ACCESS_2_HOST_READ_BIT)
         return true;
   }

   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) {
      if (info->pBufferMemoryBarriers[i].dstAccessMask &
          VK_ACCESS_2_HOST_READ_BIT)
         return true;
   }

   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
      if (info->pImageMemoryBarriers[i].dstAccessMask &
          VK_ACCESS_2_HOST_READ_BIT)
         return true;
   }

   return false;
}

/* Images are compared as whole resources. Subresource-disjoint barriers may
 * still split, and aliasable/sparse attachments disable the unwritten-image
 * proof. Never infer disjointness from different Vulkan handles alone.
 */
static uint32_t
hk_classify_pass_barrier(struct hk_cmd_buffer *cmd, const VkDependencyInfo *info)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   VkImage images[HK_MAX_RTS + 2];
   struct hk_barrier_state state = {
      .shader_writes = render->barrier_shader_writes,
      .pre_graphics = cmd->current_cs.pre_gfx != NULL,
      .post_graphics = cmd->current_cs.post_gfx != NULL,
      .tilebuffer_spilled = agx_tilebuffer_spills(&render->tilebuffer),
      .attachments = images,
   };
   const struct hk_attachment *attachments[HK_MAX_RTS + 2];
   unsigned count = 0;
   for (unsigned i = 0; i < render->color_att_count; ++i)
      attachments[count++] = &render->color_att[i];
   attachments[count++] = &render->depth_att;
   attachments[count++] = &render->stencil_att;

   bool aliases = false;
   for (unsigned i = 0; i < count; ++i) {
      if (!attachments[i]->iview)
         continue;
      struct vk_image *image = attachments[i]->iview->vk.image;
      aliases |= hk_barrier_image_may_alias(
         image->create_flags, image->tiling, image->external_handle_types);
      const VkImageAspectFlags aspect =
         i < render->color_att_count ? VK_IMAGE_ASPECT_COLOR_BIT :
         i == render->color_att_count ? VK_IMAGE_ASPECT_DEPTH_BIT :
                                       VK_IMAGE_ASPECT_STENCIL_BIT;
      if (!hk_barrier_attachment_may_write(
             attachments[i]->layout, aspect,
             attachments[i]->clear ? VK_ATTACHMENT_LOAD_OP_CLEAR :
                                      attachments[i]->load_op,
             attachments[i]->resolve_mode))
         continue;
      /* Do not mark a shared image client_visible while recording in parallel. */
      images[state.attachment_count++] =
         (VkImage)(uintptr_t)vk_object_to_u64_handle(&image->base);
   }
   state.attachments_alias = state.attachment_count && aliases;

   return hk_barrier_classify(info, &state);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                       const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (HK_PERF(dev, NOBARRIER))
      return;

   perf_debug(cmd, "Pipeline barrier");

   if (unlikely(dev->frame_anatomy)) {
      const VkDependencyInfo *dep = pDependencyInfo;
      for (uint32_t i = 0; i < dep->memoryBarrierCount; ++i) {
         cmd->anatomy_bar.src |= dep->pMemoryBarriers[i].srcStageMask;
         cmd->anatomy_bar.dst |= dep->pMemoryBarriers[i].dstStageMask;
         cmd->anatomy_bar.src_access |= dep->pMemoryBarriers[i].srcAccessMask;
         cmd->anatomy_bar.dst_access |= dep->pMemoryBarriers[i].dstAccessMask;
      }
      for (uint32_t i = 0; i < dep->bufferMemoryBarrierCount; ++i) {
         cmd->anatomy_bar.src |= dep->pBufferMemoryBarriers[i].srcStageMask;
         cmd->anatomy_bar.dst |= dep->pBufferMemoryBarriers[i].dstStageMask;
         cmd->anatomy_bar.src_access |= dep->pBufferMemoryBarriers[i].srcAccessMask;
         cmd->anatomy_bar.dst_access |= dep->pBufferMemoryBarriers[i].dstAccessMask;
      }
      for (uint32_t i = 0; i < dep->imageMemoryBarrierCount; ++i) {
         cmd->anatomy_bar.src |= dep->pImageMemoryBarriers[i].srcStageMask;
         cmd->anatomy_bar.dst |= dep->pImageMemoryBarriers[i].dstStageMask;
         cmd->anatomy_bar.src_access |= dep->pImageMemoryBarriers[i].srcAccessMask;
         cmd->anatomy_bar.dst_access |= dep->pImageMemoryBarriers[i].dstAccessMask;
      }
      cmd->anatomy_bar.mem += dep->memoryBarrierCount;
      cmd->anatomy_bar.buf += dep->bufferMemoryBarrierCount;
      cmd->anatomy_bar.img += dep->imageMemoryBarrierCount;
      cmd->anatomy_bar.count++;
   }

   /* M5 records its barriers with the native dependency tracker. */
   if (!agx_native_graphics(&dev->dev) && !agx_chip_g17_m5(dev->dev.chip)) {
      hk_cmd_buffer_end_compute(cmd);
      hk_cmd_buffer_end_graphics(cmd);
      /* Preserve the same conservative ordering in the shared submit graph,
       * including kernels that advertise exact prior barriers on AGX2. */
      hk_sync_barrier(cmd, HK_SYNC_ALL, HK_SYNC_ALL);
      return;
   }

   /* M5: a render pass instance that ended may still be continued by the
    * next one if this dependency is framebuffer-local. */
   if (unlikely(cmd->state.pass_merge.valid))
      hk_m5_pass_merge_barrier(cmd, pDependencyInfo);

   /* Host availability is recorded against the work already recorded. */
   if (cmd->state.deferred_clear_count &&
       hk_dependency_has_host_read(pDependencyInfo))
      hk_flush_deferred_clears(cmd);

   /* In-pass dependencies stay scoped to the API pass, even when its
    * physical render must end. Keep local attachment operations in tile
    * memory; descriptor/input-attachment reads still require a store because
    * the compiler lowers them to texture fetches. Software vertex work needs
    * a boundary only when the destination scope can name it.
    */
   if (cmd->sync.in_pass) {
      uint8_t src_kinds, dst_kinds;
      hk_sync_dependency_kinds(pDependencyInfo, hk_sync_image_class(cmd),
                            &src_kinds, &dst_kinds);
      const uint32_t reasons = hk_classify_pass_barrier(cmd, pDependencyInfo);
      if (reasons) {
         if (hk_cmd_buffer_end_graphics(cmd)) {
            cmd->state.barrier_stats.splits++;
            u_foreach_bit(r, reasons)
               cmd->state.barrier_stats.reason[r]++;
         }
      } else if (cmd->current_cs.gfx) {
         cmd->state.barrier_stats.kept++;
      }
      /* Ending an empty render can replace its clear with a CDM job. Capture
       * the first scope only after that replacement has its physical stream. */
      hk_sync_pass_barrier(cmd, src_kinds, dst_kinds);
      return;
   }

   uint8_t src_kinds, dst_kinds;
   hk_sync_dependency_kinds(pDependencyInfo, hk_sync_image_class(cmd),
                            &src_kinds, &dst_kinds);

   /* Compute work that stays in the open compute stream is ordered by the
    * stream itself; what it needs at a barrier is visibility. Every native
    * launch ends in a weak ordering barrier record, and the strong cache
    * flush is that same record with Apple's flags, upgraded in place. While
    * memory can be reached through two addresses, the flush needs memory
    * scope (hk_cdm_alias_scope()). */
   struct hk_cs *cs = cmd->current_cs.cs;
   const bool alias_scope = hk_cdm_alias_scope(dev);
   if (cs && (src_kinds & (HK_SYNC_COMPUTE | HK_SYNC_XFER)) && dst_kinds &&
       hk_cdm_visibility_flush_needed(dev, cs, alias_scope)) {
      hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
      hk_cdm_visibility_flush(dev, cs, alias_scope);
   }

   hk_sync_barrier(cmd, src_kinds, dst_kinds);

   /* A barrier-only command buffer otherwise records no GPU work, so it
    * cannot make a preceding fragment store visible to the host. Emit the
    * native strong CDM cache-maintenance record only when the API requests
    * host-read availability, in a stream ordered after the source scope.
    * The host reads memory, so on G17P the record has memory scope: a strong
    * flush alone leaves the data in the GPU caches, and the end of the
    * command does not reliably write it back before completion is reported
    * when render work shares the queue. */
   if ((dev->dev.chip == AGX_CHIP_G17P || agx_chip_g17_m5(dev->dev.chip)) &&
       hk_dependency_has_host_read(pDependencyInfo)) {
      struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
      hk_sync_src_dep(cmd, src_kinds, cmd->current_cs.cs, dep);

      cmd->sync.kind_override = HK_SYNC_ALL;
      struct hk_cs *flush = hk_sync_get_cs_after(cmd, dep);
      cmd->sync.kind_override = 0;
      if (!flush)
         return;

      hk_ensure_cs_has_space(cmd, flush, AGX3_CDM_BARRIER_SIZE);
      hk_cdm_visibility_flush(dev, flush, true);
   }

   /* The dependency this barrier recorded is the one a continued instance
    * applies to its stream (hk_sync_merge_rendering()); a stream it created
    * or extended still ends the candidacy there. */
   if (unlikely(cmd->state.pass_merge.valid))
      cmd->state.pass_merge.sync_epoch = cmd->sync.epoch;
}

__attribute__((hot)) void
hk_cmd_bind_shaders(struct vk_command_buffer *vk_cmd, uint32_t stage_count,
                    const mesa_shader_stage *stages,
                    struct vk_shader **const shaders)
{
   struct hk_cmd_buffer *cmd = container_of(vk_cmd, struct hk_cmd_buffer, vk);

   for (uint32_t i = 0; i < stage_count; i++) {
      struct hk_api_shader *shader = shaders[i]
         ? container_of(shaders[i], struct hk_api_shader, vk)
         : NULL;
      if (stages[i] == MESA_SHADER_COMPUTE || stages[i] == MESA_SHADER_KERNEL)
         hk_cmd_bind_compute_shader(cmd, shader);
      else
         hk_cmd_bind_graphics_shader(cmd, stages[i], shader);
   }

}

/* Only ordinary G17P graphics programs whose descriptor values are fetched
 * through the root/table in GPU memory qualify.  CPU-decoded combined sampler
 * programs and variant selection retain the existing publication path.
 */
static bool
hk_g17p_can_retain_descriptor_root(struct hk_cmd_buffer *cmd,
                                  const struct hk_descriptor_state *desc)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   if (!agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev) ||
       desc != &gfx->descriptors ||
       gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL])
      return false;

   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_api_shader *fs = gfx->shaders[MESA_SHADER_FRAGMENT];
   if (!vs || !fs || fs->g17p_sample_variants)
      return false;

   const struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   const struct hk_shader *hw_fs = hk_fs_variant(fs, 1);
   return hw_vs && hw_fs && hw_vs->compiled && hw_fs->compiled &&
          !hw_fs->info.agx3_combined_sampler_valid;
}

static void
hk_bind_descriptor_sets(struct hk_cmd_buffer *cmd,
                        struct hk_descriptor_state *desc,
                        const VkBindDescriptorSetsInfoKHR *info)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, info->layout);

   /* Fro the Vulkan 1.3.275 spec:
    *
    *    "When binding a descriptor set (see Descriptor Set Binding) to
    *    set number N...
    *
    *    If, additionally, the previously bound descriptor set for set
    *    N was bound using a pipeline layout not compatible for set N,
    *    then all bindings in sets numbered greater than N are
    *    disturbed."
    *
    * This means that, if some earlier set gets bound in such a way that
    * it changes set_dynamic_buffer_start[s], this binding is implicitly
    * invalidated.
    */
   uint8_t dyn_buffer_start =
      pipeline_layout->dynamic_descriptor_offset[info->firstSet];

   bool root_changed = false;
   bool binding_changed = false;
   uint32_t next_dyn_offset = 0;
   for (uint32_t i = 0; i < info->descriptorSetCount; ++i) {
      unsigned s = i + info->firstSet;
      VK_FROM_HANDLE(hk_descriptor_set, set, info->pDescriptorSets[i]);

      if (desc->sets[s] != set ||
          (desc->push_active & BITFIELD_BIT(s))) {
         binding_changed = true;
         if (set != NULL) {
            root_changed |= desc->root.sets[s] != hk_descriptor_set_addr(set);
            desc->root.sets[s] = hk_descriptor_set_addr(set);
            desc->set_sizes[s] = set->size;
         } else {
            root_changed |= desc->root.sets[s] != 0;
            desc->root.sets[s] = 0;
            desc->set_sizes[s] = 0;
         }
         desc->sets[s] = set;
         desc->sets_dirty |= BITFIELD_BIT(s);

         /* Binding descriptors invalidates push descriptors */
      }
      desc->push_active &= ~BITFIELD_BIT(s);
      desc->push_dirty &= ~BITFIELD_BIT(s);

      root_changed |=
         desc->root.set_dynamic_buffer_start[s] != dyn_buffer_start;
      desc->root.set_dynamic_buffer_start[s] = dyn_buffer_start;

      if (pipeline_layout->set_layouts[s] != NULL) {
         const struct hk_descriptor_set_layout *set_layout =
            vk_to_hk_descriptor_set_layout(pipeline_layout->set_layouts[s]);

         if (set != NULL && set_layout->vk.dynamic_descriptor_count > 0) {
            for (uint32_t j = 0; j < set_layout->vk.dynamic_descriptor_count; j++) {
               struct hk_buffer_address addr = set->dynamic_buffers[j];
               /* A null buffer stays at the zero page (hk_ubo_addr_format). */
               if (addr.size)
                  addr.base_addr += info->pDynamicOffsets[next_dyn_offset + j];
               const struct hk_buffer_address *old =
                  &desc->root.dynamic_buffers[dyn_buffer_start + j];
               root_changed |= old->base_addr != addr.base_addr ||
                               old->size != addr.size ||
                               old->sparse_map_page != addr.sparse_map_page;
               desc->root.dynamic_buffers[dyn_buffer_start + j] = addr;
            }
            next_dyn_offset += set->layout->vk.dynamic_descriptor_count;
         }

         dyn_buffer_start += set_layout->vk.dynamic_descriptor_count;
      } else {
         assert(set == NULL);
      }
   }
   assert(dyn_buffer_start <= HK_MAX_DYNAMIC_BUFFERS);
   assert(next_dyn_offset <= info->dynamicOffsetCount);
   root_changed |= desc->dynamic_buffer_end < dyn_buffer_start;
   desc->dynamic_buffer_end = MAX2(desc->dynamic_buffer_end, dyn_buffer_start);

   hk_upload_stat(cmd, descriptor_binds, 1);
   const bool retain_root = hk_g17p_can_retain_descriptor_root(cmd, desc);
   if (!root_changed && !binding_changed && retain_root) {
      hk_upload_stat(cmd, descriptor_noops, 1);
      return;
   }

   /* CPU-decoded descriptor programs retain their conservative invalidation:
    * their inputs can change in set memory without changing its address. */
   desc->root_dirty |= root_changed || !retain_root;
   desc->generation++;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindDescriptorSets2KHR(
   VkCommandBuffer commandBuffer,
   const VkBindDescriptorSetsInfoKHR *pBindDescriptorSetsInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pBindDescriptorSetsInfo->stageFlags & (VK_SHADER_STAGE_ALL_GRAPHICS |
       VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT)) {
      hk_m5_pass_merge_note_set_bind(cmd);
      hk_bind_descriptor_sets(cmd, &cmd->state.gfx.descriptors,
                              pBindDescriptorSetsInfo);
   }

   if (pBindDescriptorSetsInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
      hk_bind_descriptor_sets(cmd, &cmd->state.cs.descriptors,
                              pBindDescriptorSetsInfo);
   }

   if (pBindDescriptorSetsInfo->stageFlags & HK_RT_SHADER_STAGES) {
      hk_bind_descriptor_sets(cmd, &cmd->state.rt.descriptors,
                              pBindDescriptorSetsInfo);
   }
}

static void
hk_push_constants(UNUSED struct hk_cmd_buffer *cmd,
                  struct hk_descriptor_state *desc,
                  const VkPushConstantsInfoKHR *info)
{
   /* D3D translation layers re-push unchanged constants before most draws.
    * While the root is clean the uploaded root holds exactly these bytes, so
    * an identical push changes nothing; keep the root and every preparation
    * keyed to the current descriptor generation. */
   if (memcmp(desc->root.push + info->offset, info->pValues, info->size) == 0)
      return;

   memcpy(desc->root.push + info->offset, info->pValues, info->size);
   desc->root_dirty = true;
   desc->generation++;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdPushConstants2KHR(VkCommandBuffer commandBuffer,
                        const VkPushConstantsInfoKHR *pPushConstantsInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pPushConstantsInfo->stageFlags & (VK_SHADER_STAGE_ALL_GRAPHICS |
       VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT))
      hk_push_constants(cmd, &cmd->state.gfx.descriptors, pPushConstantsInfo);

   if (pPushConstantsInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
      hk_push_constants(cmd, &cmd->state.cs.descriptors, pPushConstantsInfo);

   if (pPushConstantsInfo->stageFlags & HK_RT_SHADER_STAGES)
      hk_push_constants(cmd, &cmd->state.rt.descriptors, pPushConstantsInfo);
}

static struct hk_push_descriptor_set *
hk_cmd_push_descriptors(struct hk_cmd_buffer *cmd,
                        struct hk_descriptor_state *desc, uint32_t set)
{
   assert(set < HK_MAX_SETS);
   if (unlikely(desc->push[set] == NULL)) {
      desc->push[set] =
         vk_zalloc(&cmd->vk.pool->alloc, sizeof(*desc->push[set]), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (unlikely(desc->push[set] == NULL)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return NULL;
      }
   }

   /* Pushing descriptors replaces whatever sets are bound */
   desc->sets[set] = NULL;
   desc->push_dirty |= BITFIELD_BIT(set);
   desc->push_active |= BITFIELD_BIT(set);
   desc->generation++;

   return desc->push[set];
}

/* Whether a descriptor of this type names an image view. */
static inline bool
hk_descriptor_type_has_image_view(VkDescriptorType type)
{
   return type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
          type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
          type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ||
          type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
}

/* AGX3 render pass continuation: the image views a graphics push descriptor
 * update writes (hk_m5_pass_merge_note_push()). */
static void
hk_m5_note_push_writes(struct hk_cmd_buffer *cmd, uint32_t count,
                       const VkWriteDescriptorSet *writes)
{
   const enum agx_chip chip = hk_cmd_buffer_device(cmd)->dev.chip;
   if (!agx_chip_native_agx3(chip) && !agx_chip_g17_m5(chip))
      return;

   for (uint32_t i = 0; i < count; ++i) {
      if (!hk_descriptor_type_has_image_view(writes[i].descriptorType))
         continue;
      for (uint32_t j = 0; j < writes[i].descriptorCount; ++j) {
         VK_FROM_HANDLE(hk_image_view, iview,
                        writes[i].pImageInfo[j].imageView);
         if (iview)
            hk_m5_pass_merge_note_push(cmd, iview);
      }
   }
}

static void
hk_m5_note_push_template(struct hk_cmd_buffer *cmd,
                         const struct vk_descriptor_update_template *template,
                         const void *data)
{
   const enum agx_chip chip = hk_cmd_buffer_device(cmd)->dev.chip;
   if (!agx_chip_native_agx3(chip) && !agx_chip_g17_m5(chip))
      return;

   for (uint32_t i = 0; i < template->entry_count; ++i) {
      const struct vk_descriptor_template_entry *entry = &template->entries[i];
      if (!hk_descriptor_type_has_image_view(entry->type))
         continue;
      for (uint32_t j = 0; j < entry->array_count; ++j) {
         const VkDescriptorImageInfo *info =
            (const void *)((const uint8_t *)data + entry->offset +
                           j * entry->stride);
         VK_FROM_HANDLE(hk_image_view, iview, info->imageView);
         if (iview)
            hk_m5_pass_merge_note_push(cmd, iview);
      }
   }
}

static void
hk_push_descriptor_set(struct hk_cmd_buffer *cmd,
                       struct hk_descriptor_state *desc,
                       const VkPushDescriptorSetInfoKHR *info)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, info->layout);

   if (desc == &cmd->state.gfx.descriptors)
      hk_m5_note_push_writes(cmd, info->descriptorWriteCount,
                             info->pDescriptorWrites);

   struct hk_push_descriptor_set *push_set =
      hk_cmd_push_descriptors(cmd, desc, info->set);
   if (unlikely(push_set == NULL))
      return;

   struct hk_descriptor_set_layout *set_layout =
      vk_to_hk_descriptor_set_layout(pipeline_layout->set_layouts[info->set]);

   hk_push_descriptor_set_update(push_set, set_layout,
                                 info->descriptorWriteCount,
                                 info->pDescriptorWrites);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdPushDescriptorSet2KHR(
   VkCommandBuffer commandBuffer,
   const VkPushDescriptorSetInfoKHR *pPushDescriptorSetInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pPushDescriptorSetInfo->stageFlags & (VK_SHADER_STAGE_ALL_GRAPHICS |
       VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT)) {
      hk_push_descriptor_set(cmd, &cmd->state.gfx.descriptors,
                             pPushDescriptorSetInfo);
   }

   if (pPushDescriptorSetInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
      hk_push_descriptor_set(cmd, &cmd->state.cs.descriptors,
                             pPushDescriptorSetInfo);
   }

   if (pPushDescriptorSetInfo->stageFlags & HK_RT_SHADER_STAGES) {
      hk_push_descriptor_set(cmd, &cmd->state.rt.descriptors,
                             pPushDescriptorSetInfo);
   }
}

void
hk_cmd_buffer_flush_push_descriptors(struct hk_cmd_buffer *cmd,
                                     struct hk_descriptor_state *desc)
{
   u_foreach_bit(set_idx, desc->push_dirty) {
      struct hk_push_descriptor_set *push_set = desc->push[set_idx];
      /* Only the bytes the pushed layout uses; the rest is never read */
      const uint32_t size = MAX2(push_set->size, HK_MIN_UBO_ALIGNMENT);
      assert(size <= sizeof(push_set->data));
      const uint64_t push_set_addr =
         hk_pool_upload(cmd, push_set->data, size, HK_MIN_UBO_ALIGNMENT);

      desc->root.sets[set_idx] = push_set_addr;
      desc->set_sizes[set_idx] = size;
   }

   desc->root_dirty = true;
   desc->push_dirty = 0;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdPushDescriptorSetWithTemplate2KHR(
   VkCommandBuffer commandBuffer, const VkPushDescriptorSetWithTemplateInfoKHR
                                     *pPushDescriptorSetWithTemplateInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_descriptor_update_template, template,
                  pPushDescriptorSetWithTemplateInfo->descriptorUpdateTemplate);
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout,
                  pPushDescriptorSetWithTemplateInfo->layout);

   struct hk_descriptor_state *desc =
      hk_get_descriptors_state(cmd, template->bind_point);
   struct hk_push_descriptor_set *push_set = hk_cmd_push_descriptors(
      cmd, desc, pPushDescriptorSetWithTemplateInfo->set);
   if (unlikely(push_set == NULL))
      return;

   struct hk_descriptor_set_layout *set_layout = vk_to_hk_descriptor_set_layout(
      pipeline_layout->set_layouts[pPushDescriptorSetWithTemplateInfo->set]);

   if (desc == &cmd->state.gfx.descriptors)
      hk_m5_note_push_template(cmd, template,
                               pPushDescriptorSetWithTemplateInfo->pData);

   hk_push_descriptor_set_update_template(
      push_set, set_layout, template,
      pPushDescriptorSetWithTemplateInfo->pData);
}

DEBUG_GET_ONCE_BOOL_OPTION(m5_fragment_root_memo, "ASAHI_M5_FRAGMENT_ROOT_MEMO",
                           true)
DEBUG_GET_ONCE_BOOL_OPTION(m5_fragment_root_shadow,
                           "ASAHI_M5_FRAGMENT_ROOT_SHADOW", true)
DEBUG_GET_ONCE_BOOL_OPTION(m5_verify_fragment_root_shadow,
                           "ASAHI_M5_VERIFY_FRAGMENT_ROOT_SHADOW", false)

static uint64_t
hk_upload_native_spilled_rt_descs(struct hk_cmd_buffer *cmd)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   uint8_t entries[HK_MAX_RTS][AGX_G17P_EMRT_DESCRIPTOR_SIZE] = {0};
   for (unsigned rt = 0; rt < render->color_att_count; ++rt) {
      if (!render->tilebuffer.spilled[rt])
         continue;
      const struct hk_image_view *view = render->color_att[rt].iview;
      if (!view || !view->planes[0].emrt_pbe_valid ||
          !view->planes[0].emrt_read_heap_index) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FORMAT_NOT_SUPPORTED);
         return 0;
      }
      memcpy(entries[rt], &view->planes[0].emrt_texture, 32);
      memcpy(entries[rt] + 32, &view->planes[0].emrt_pbe, 32);
      memcpy(entries[rt] + AGX_G17P_STORAGE_HEAP_INDEX_OFFSET,
             &view->planes[0].emrt_read_heap_index, 4);
      memcpy(entries[rt] + AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET,
             &view->planes[0].emrt_read_heap_index, 4);
      memcpy(entries[rt] + AGX_G17P_IMAGE_ATOMIC_SW_OFFSET,
             &view->planes[0].emrt_atomic_sw, 8);
      memcpy(entries[rt] + AGX_G17P_EMRT_ALIGNED_WIDTH_OFFSET,
             &view->planes[0].emrt_aligned_width, 4);
   }

   const size_t size = render->color_att_count * sizeof(entries[0]);
   __typeof__(cmd->state.gfx.g17p_spilled_rt_descs) *cached =
      &cmd->state.gfx.g17p_spilled_rt_descs;
   if (cached->valid && cached->color_att_count == render->color_att_count &&
       !memcmp(cached->entries, entries, size))
      return cached->pointer;

   const uint64_t address = hk_pool_upload(cmd, entries, size, 32);
   const uint64_t pointer = address ? hk_pool_upload(cmd, &address, 8, 8) : 0;
   if (pointer) {
      memcpy(cached->entries, entries, size);
      cached->address = address;
      cached->pointer = pointer;
      cached->color_att_count = render->color_att_count;
      cached->valid = true;
   }
   return pointer;
}

void
hk_usc_upload_spilled_rt_descs(struct agx_usc_builder *b,
                               struct hk_cmd_buffer *cmd)
{
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const unsigned generation =
      agx_chip_descriptor_generation(dev->dev.chip);

   if (!hk_native_pbe_hardware_supported(generation)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   const unsigned texture_size = hk_texture_descriptor_size(generation);
   const unsigned pbe_size = hk_pbe_descriptor_size(generation);
   const unsigned pair_size = texture_size + pbe_size;

   /* Upload texture/PBE descriptors for each render target so we can clear
    * spilled render targets.
    */
   struct agx_ptr descs =
      hk_pool_alloc(cmd, pair_size * render->color_att_count, 64);
   uint8_t *desc = descs.cpu;
   if (!desc)
      return;

   for (unsigned i = 0; i < render->color_att_count; ++i) {
      struct hk_image_view *iview = render->color_att[i].iview;
      if (!iview) {
         /* XXX: probably should emit a null descriptor here...? */
         continue;
      }

      if (!iview->planes[0].emrt_pbe_valid) {
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FORMAT_NOT_SUPPORTED);
         return;
      }

      memcpy(desc + i * pair_size, &iview->planes[0].emrt_texture,
             texture_size);
      memcpy(desc + i * pair_size + texture_size,
             &iview->planes[0].emrt_pbe, pbe_size);
   }

   /* Bind the base as u0_u1 for bindless access */
   agx_usc_uniform(b, 0, 4, hk_pool_upload(cmd, &descs.gpu, 8, 8));
}

void
hk_reserve_scratch_slow(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        struct hk_shader *s)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   uint32_t max_scratch_size =
      MAX2(s->b.info.scratch_size, s->b.info.preamble_scratch_size);

   unsigned preamble_size = (s->b.info.preamble_scratch_size > 0) ? 1 : 0;

   /* Note: this uses the hardware stage, not the software stage */
   hk_device_alloc_scratch(dev, s->b.info.stage, max_scratch_size);
   perf_debug(cmd, "Reserving %u (%u) bytes of scratch for stage %s",
              s->b.info.scratch_size, s->b.info.preamble_scratch_size,
              _mesa_shader_stage_to_abbrev(s->b.info.stage));

   switch (s->b.info.stage) {
   case MESA_SHADER_FRAGMENT:
      cs->scratch.fs.main = true;
      cs->scratch.fs.preamble = MAX2(cs->scratch.fs.preamble, preamble_size);
      break;
   case MESA_SHADER_VERTEX:
      cs->scratch.vs.main = true;
      cs->scratch.vs.preamble = MAX2(cs->scratch.vs.preamble, preamble_size);
      break;
   default:
      cs->scratch.cs.main = true;
      cs->scratch.cs.preamble = MAX2(cs->scratch.cs.preamble, preamble_size);
      break;
   }
}

static bool
hk_g17p_basic_fragment_layout(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct agx_tilebuffer_layout *tib = &cmd->state.gfx.render.tilebuffer;
   if (agx_chip_g17_m5(dev->dev.chip) && tib->nr_samples > 1) {
      return (tib->nr_samples == 2 || tib->nr_samples == 4 ||
              tib->nr_samples == 8) &&
             (tib->tile_size == 32 * 32 || tib->tile_size == 32 * 16 ||
              tib->tile_size == 16 * 16) &&
             tib->sample_size_B >= 8 && !(tib->sample_size_B & 7) &&
             tib->sample_size_B <= tib->max_sample_size_B &&
             tib->nr_samples * tib->tile_size <= tib->max_samples_per_tile &&
             tib->sample_size_B * tib->nr_samples * tib->tile_size <=
                tib->max_bytes_per_tile;
   }
   return cmd->state.gfx.render.g17p_rt0_layout_supported;
}

/* Keep native interpolation bounded to the f32 UVS layout already supported
 * by hk_agx3_g17p_vertex_program_info: position plus up to 64 user words.
 * HAL200 updateVaryingTable (0x3efa58) uses the same CF entry encoding for
 * smooth, linear and flat inputs, including their mixture in one shader.
 * Flat inputs select C. Native G17P Point and Triangle pipelines have identical
 * complete fragment text for flat vec2/vec3/vec4 inputs. The provoking selector
 * is topology-independent, so line lists and strips use the same witnessed
 * first/last selector values as triangles. First-provoking fans use their
 * dedicated selector and the fragment shader's draw-root B/C coefficient
 * choice. Position Z and reciprocal W use native coefficient sources.
 */
static bool
hk_g17p_smooth_varyings_supported_uncached(struct hk_cmd_buffer *cmd,
                                          const struct hk_linked_shader *linked)
{
   if (!linked)
      return false;
   const struct agx_varyings_fs *cf = &linked->b.cf;
   if (cf->nr_bindings == 0)
      return cf->nr_cf == 0;

   /* The coefficients describe the hardware vertex program the draw
    * launches: the API vertex shader, the post-tessellation vertex program,
    * or the geometry raster-copy variant. */
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const bool mesh=gfx->shaders[MESA_SHADER_MESH]!=NULL;
   const struct hk_shader *vs =
      (gfx->shaders[MESA_SHADER_VERTEX]||mesh) ? hk_bound_hw_vs(gfx) : NULL;
   if (!vs || cf->nr_bindings > HK_AGX3_G17P_MAX_COEFFICIENTS ||
       cf->nr_cf > HK_AGX3_G17P_MAX_COEFFICIENTS ||
       ((!mesh || vs->info.stage!=MESA_SHADER_MESH) &&
        vs->agx3_vdm_state_update.source_layout !=
          AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT) ||
       vs->info.uvs.user_size > HK_AGX3_G17P_MAX_USER_WORDS)
      return false;

   unsigned next_cf = 0;
   bool has_rhw = false, has_z = false;
   unsigned user_bindings = 0;
   for (unsigned i = 0; i < cf->nr_bindings; ++i) {
      const struct agx_cf_binding *b = &cf->bindings[i];
      if (b->count == 0 || b->count > 4 || b->cf_base != next_cf)
         return false;
      if (!b->smooth) {
         const bool point = gfx->topology == AGX_PRIMITIVE_POINTS &&
                            gfx->provoking == 0;
         const bool line =
            (gfx->topology == AGX_PRIMITIVE_LINES ||
             gfx->topology == AGX_PRIMITIVE_LINE_STRIP) &&
            (gfx->provoking == 0 || gfx->provoking == 2);
         const bool triangle =
            (gfx->topology == AGX_PRIMITIVE_TRIANGLES ||
             gfx->topology == AGX_PRIMITIVE_TRIANGLE_STRIP) &&
            (gfx->provoking == 0 || gfx->provoking == 2);
         const bool fan = gfx->topology == AGX_PRIMITIVE_TRIANGLE_FAN &&
                          (gfx->provoking == 1 || gfx->provoking == 2);
         if (b->perspective || (!point && !line && !triangle && !fan))
            return false;
      }
      if (b->slot == VARYING_SLOT_POS) {
         /* System coefficients precede user coefficients: optional W,
          * then optional Z. agx_link_varyings assigns native Z source1. */
         if (user_bindings || b->count != 1 || b->perspective || !b->smooth)
            return false;
         if (b->offset == 3) {
            if (i != 0 || has_rhw || has_z)
               return false;
            has_rhw = true;
         } else if (b->offset == 2) {
            if (has_z || !cf->reads_z)
               return false;
            has_z = true;
         } else {
            return false;
         }
      } else {
         /* Fragment layer and gl_ViewportIndex are ordinary flat varyings
          * that the vertex shader also writes out as user words. Clip-distance
          * inputs use the compiler's compact coefficient layout while the
          * fixed-function clipper consumes the same VS output. Cull-distance
          * emulation similarly links the VS-generated CULL_PRIMITIVE words
          * into its FS prolog as noperspective coefficients. */
         const bool cull_slot =
            b->slot == VARYING_SLOT_CULL_PRIMITIVE ||
            b->slot == VARYING_SLOT_CULL_PRIMITIVE + 1;
         const bool clip_slot = b->slot == VARYING_SLOT_CLIP_DIST0 ||
                                b->slot == VARYING_SLOT_CLIP_DIST1;
         /* gl_PrimitiveID that no vertex shader writes is not a UVS word at
          * all: agx_link_varyings_vs_fs() gives its binding
          * AGX_COEFFICIENT_SOURCE_PRIMITIVE_ID and asks the VDM to generate
          * the value, which hk_flush_dynamic_state() already plumbs through
          * generate_primitive_id. Admit it as a system coefficient beside
          * FragCoord Z and 1/W -- scalar, flat, and indexing no vertex
          * output. A VS-written gl_PrimitiveID would take the ordinary user
          * word path instead and stays refused, having no witness. A mesh
          * shader's per-primitive gl_PrimitiveID is a flat user word of the
          * primitive's private vertices (agx_m5_mesh_gather_varyings) and
          * links like any other user word below. */
         const bool mesh_primitive_id =
            mesh && vs->info.stage == MESA_SHADER_MESH &&
            b->slot == VARYING_SLOT_PRIMITIVE_ID &&
            gfx->linked_varyings.slots[VARYING_SLOT_PRIMITIVE_ID] &&
            b->count == 1 && b->offset == 0 && !b->smooth && !b->perspective;
         if (b->slot == VARYING_SLOT_PRIMITIVE_ID && !mesh_primitive_id) {
            if (gfx->linked_varyings.slots[VARYING_SLOT_PRIMITIVE_ID] ||
                b->count != 1 || b->offset != 0 || b->smooth || b->perspective)
               return false;
            ++user_bindings;
            next_cf += b->count;
            continue;
         }
         /* gl_PointCoord is the third hardware-generated coefficient source,
          * beside FragCoord Z and gl_PrimitiveID: nir_lower_sysvals_to_varyings
          * turns the system value into a VARYING_SLOT_PNTC input and
          * agx_link_varyings_vs_fs() gives that binding
          * AGX_COEFFICIENT_SOURCE_POINT_COORD with base slot 0, so it indexes
          * no vertex output and needs no UVS word. The rasterizer supplies the
          * sprite UV, selected by the Object Type nibble that
          * translate_object_type() already sets to POINT_SPRITE_UV01 for point
          * topology and that hk_g17p_emit_native_bind_tail() already packs into
          * both face words. The CF entry is the same 4-byte encoding HAL200
          * updateVaryingTable consumes for every other source, so nothing else
          * in the native tail changes.
          *
          * agx_link_varyings_vs_fs() asserts base offset 0 and overwrites the
          * base slot, so a partial read starting at .y has no correct encoding
          * on any chip; keep it refused rather than rendering it wrong.
          *
          * The CF entry carries no topology or provoking selector, so this is
          * not gated on point topology: a dynamic-topology pipeline may bind
          * the same fragment shader for a triangle draw, where Vulkan leaves
          * gl_PointCoord undefined. Refusing there would turn a legal draw
          * into VK_ERROR_FEATURE_NOT_PRESENT for no conformance gain. */
         if (b->slot == VARYING_SLOT_PNTC) {
            if (b->offset != 0 || b->count > 2 || !b->smooth ||
                (b->perspective && !has_rhw))
               return false;
            ++user_bindings;
            next_cf += b->count;
            continue;
         }
         const bool primitive_page = mesh &&
            vs->info.uvs.mesh_primitive_page &&
            b->slot == AGX_MESH_PRIMITIVE_PAGE_SLOT &&
            b->count == vs->info.uvs.components[AGX_MESH_PRIMITIVE_PAGE_SLOT] &&
            b->count >= 1 && b->count <= 2 &&
            b->offset == 0 && !b->smooth && !b->perspective;
         const bool user_slot = primitive_page || mesh_primitive_id ||
                                (b->slot >= VARYING_SLOT_VAR0 &&
                                 b->slot <= VARYING_SLOT_VAR31) ||
                                b->slot == VARYING_SLOT_LAYER ||
                                b->slot == VARYING_SLOT_VIEWPORT || clip_slot ||
                                cull_slot;
         if (!user_slot || (cull_slot && (b->perspective || !b->smooth)) ||
             (b->perspective && !has_rhw) ||
             b->offset + b->count > vs->info.uvs.components[b->slot])
            return false;
         ++user_bindings;
         const unsigned slot = gfx->linked_varyings.slots[b->slot];
         if (slot < 4 || slot + b->offset + b->count > vs->info.uvs.size)
            return false;
      }
      next_cf += b->count;
   }
   return (user_bindings || has_rhw || has_z) &&
          has_z == cf->reads_z && next_cf == cf->nr_cf;
}

/* Evaluated for every draw that republishes its fragment program. The
 * answer depends only on the linked fragment coefficients, the hardware
 * vertex shader's outputs, the topology, the provoking vertex and the
 * current linked_varyings slot table, so remember the last evaluation. */
bool
hk_g17p_smooth_varyings_supported(struct hk_cmd_buffer *cmd,
                                 const struct hk_linked_shader *linked)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_shader *vs =
      gfx->shaders[MESA_SHADER_VERTEX] ? hk_bound_hw_vs(gfx) : NULL;
   __typeof__(gfx->g17p_smooth_memo) *memo = &gfx->g17p_smooth_memo;
   if (linked && memo->valid && memo->linked == linked && memo->vs == vs &&
       memo->topology == gfx->topology && memo->provoking == gfx->provoking)
      return memo->result;

   const bool result = hk_g17p_smooth_varyings_supported_uncached(cmd, linked);
   if (linked) {
      *memo = (__typeof__(*memo)){
         .linked = linked, .vs = vs, .topology = gfx->topology,
         .provoking = gfx->provoking, .valid = true, .result = result,
      };
   }
   return result;
}

static bool
hk_g17p_basic_graphics_program_supported(struct hk_cmd_buffer *cmd,
                                         struct hk_shader *s,
                                         struct hk_linked_shader *linked)
{
   const bool native_spills = s->info.stage == MESA_SHADER_FRAGMENT &&
      agx_tilebuffer_spills(&cmd->state.gfx.render.tilebuffer);
   /* First native CProg qualification covers the imported-input region.
    * Local/spill frame use needs its own native metadata witnesses before
    * admission. This is not a fallback:
    * a candidate outside the bounded protocol is refused. */
   /* Every G17P hardware vertex program (API VS, GS raster and native TES
    * variants) is compiled with a native CProg when anything is hoistable;
    * its VertexProgramVariant carries the TI counts. The stage branches
    * below still require that variant. */
   const bool native_preamble =
      (s->b.info.stage == MESA_SHADER_VERTEX ||
       s->info.stage == MESA_SHADER_FRAGMENT) &&
      hk_g17p_native_cprog_qualified(&s->b.info);
   if (!linked || !linked->b.bo || !linked->b.bo->va || linked->b.size == 0 ||
       linked->b.size > UINT32_MAX || (s->b.info.has_preamble && !native_preamble) ||
       s->b.info.scratch_size > AGX3_G17P_MAX_SPILL_B ||
       (s->b.info.preamble_scratch_size != 0 && !native_preamble) ||
       linked->b.scratch_size > AGX3_G17P_MAX_SPILL_B ||
       s->b.info.texture_state_count != 0 ||
       (s->b.info.sampler_state_count != 0 &&
        !(s->info.agx3_indexed_sampler_valid &&
          s->b.info.sampler_state_count == 1)) ||
       s->b.info.uses_sampler_heap ||
       (linked->b.uses_txf && !s->info.agx3_indexed_sampler_valid && !native_spills) ||
       s->info.set_count > HK_AGX3_G17P_MAX_ROOT_SETS) {
      mesa_loge_once(
         "G17P graphics program refused: linked=%u bo=%u va=%u size=%zu "
         "preamble=%u scratch=%u/%u/%u textures=%u samplers=%u "
         "indexed_sampler=%u sampler_heap=%u txf=%u sets=%u",
         linked != NULL, linked && linked->b.bo != NULL,
         linked && linked->b.bo && linked->b.bo->va != NULL,
         linked ? linked->b.size : 0, s->b.info.has_preamble,
         s->b.info.scratch_size, s->b.info.preamble_scratch_size,
         linked ? linked->b.scratch_size : 0,
         s->b.info.texture_state_count, s->b.info.sampler_state_count,
         s->info.agx3_indexed_sampler_valid, s->b.info.uses_sampler_heap,
         linked ? linked->b.uses_txf : 0, s->info.set_count);
      return false;
   }

   if (s->info.stage == MESA_SHADER_VERTEX) {
      const unsigned nr_attribs = DIV_ROUND_UP(
         BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4);
      /* Software indexing is admitted when the bound program is still the
       * hardware vertex shader and only fetches its own indices, which is
       * what an adjacency topology without a geometry shader compiles to.
       * Its extra state is the VERTEX_PARAMS uniform, which
       * hk_flush_dynamic_state() already uploads for exactly this case. The
       * geometry and tessellation shape, where the software vertex shader
       * runs as a compute kernel, stays refused. */
      const bool ok =
         (!linked->sw_indexing || linked->sw_indexing_hw_vs) &&
         nr_attribs <= AGX_MAX_VBUFS &&
         s->agx3_vdm_state_update.source_layout ==
            AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT;

      if (!ok) {
         mesa_loge_once("G17P graphics program refused: stage=vs "
                        "sw_indexing=%u attribs=%u/%u source_layout=%u "
                        "(want %u)",
                        linked->sw_indexing, nr_attribs, AGX_MAX_VBUFS,
                        (unsigned)s->agx3_vdm_state_update.source_layout,
                        (unsigned)
                        AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT);
      }

      return ok;
   }

   /* The raster variant produced by poly_nir_lower_gs is a hardware vertex
    * program while retaining the API geometry stage in hk_shader_info.  It
    * pulls generated vertices through the root ABI rather than vertex
    * attributes, then performs the XFB stores before rasterization.  The
    * post-tessellation vertex program likewise keeps the evaluation stage and
    * reads the control shader's outputs through the root ABI. */
   if ((s->info.stage == MESA_SHADER_GEOMETRY ||
        s->info.stage == MESA_SHADER_TESS_EVAL) &&
       s->b.info.stage == MESA_SHADER_VERTEX) {
      const bool ok =
         !linked->sw_indexing &&
         s->agx3_vdm_state_update.source_layout ==
            AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT;

      if (!ok) {
         mesa_loge_once("G17P graphics program refused: stage=gs-raster "
                        "sw_indexing=%u source_layout=%u (want %u)",
                        linked->sw_indexing,
                        (unsigned)s->agx3_vdm_state_update.source_layout,
                        (unsigned)
                        AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT);
      }

      return ok;
   }

   if (s->info.stage != MESA_SHADER_FRAGMENT) {
      mesa_loge_once("G17P graphics program refused: stage=%u has no native "
                     "G17P graphics path (sw stage %u, hw stage %u)",
                     (unsigned)s->info.stage, (unsigned)s->info.stage,
                     (unsigned)s->b.info.stage);
      return false;
   }

   {
      const bool smooth = hk_g17p_smooth_varyings_supported(cmd, linked);
      const bool layout = hk_g17p_basic_fragment_layout(cmd);

      if (!smooth || !layout) {
         mesa_loge_once("G17P graphics program refused: stage=fs "
                        "smooth_varyings=%u basic_fragment_layout=%u",
                        smooth, layout);
      }

      return smooth && layout;
   }
}

/*
 * Every uniform range hk_g17p_graphics_uniform_ranges() below can emit, by the
 * branch that emits it. A vertex program can request all of the first block;
 * a fragment program the second. Keeping the budget derived from this instead
 * of hand-set is what stops a legal program from being refused fail-closed --
 * a budget of five silently refused every Firefox/Zink vertex program that
 * sampled a texture and promoted a constant.
 */
#define HK_G17P_VERTEX_RANGES                                                  \
   (2 /* attribute bases and clamps */ + 1 /* draw parameters */ +             \
    1 /* draw ID */ + 1 /* indexed sampler heap */ + 1 /* root */ +            \
    1 /* promoted constants */)
/* Each uniform DMA span publishes at most two ranges: its buffer part and a
 * zero-page tail (hk_g17p_graphics_uniform_ranges()). */
#define HK_G17P_DMA_RANGES (2 * AGX3_UNIFORM_DMA_MAX_SPANS)

#define HK_G17P_FRAGMENT_RANGES                                                \
   (1 /* blend constants */ + 1 /* combined sampler heap */ +                  \
    1 /* indexed sampler heap */ + 1 /* root */ + 1 /* promoted constants */ + \
    1 /* native spill table */)

static_assert(HK_AGX3_G17P_GRAPHICS_MAX_RANGES >=
                    HK_G17P_VERTEX_RANGES + HK_G17P_DMA_RANGES &&
                 HK_AGX3_G17P_GRAPHICS_MAX_RANGES >=
                    HK_G17P_FRAGMENT_RANGES + HK_G17P_DMA_RANGES,
              "the graphics range budget must cover every range "
              "hk_g17p_graphics_uniform_ranges() can emit, or a legal "
              "program is refused and ranges[] is written past its end");
static_assert(HK_G17P_PATCH_MAX_RANGES >= HK_AGX3_G17P_GRAPHICS_MAX_RANGES,
              "every graphics range layout needs a patchable loader");

/*
 * Record how many uniform ranges the collector had already committed when it
 * refused. Without it the refusal diagnostic cannot distinguish "this program
 * needs more ranges than the budget allows" from the several structural
 * rejections that share the same message, which is exactly the ambiguity that
 * made the Firefox hardware refusal unreadable.
 */
static bool
hk_g17p_graphics_ranges_refused(size_t *range_count, size_t count)
{
   *range_count = count;
   return false;
}

/*
 * AGX_G17P_UNIFORM_DMA sources its span from a pointer uniform: either the
 * root table (a root-relative address, so it follows the root's identity) or
 * a descriptor-set pointer.  For a set pointer the span address is the set's
 * address *value*, copied from the CPU-side root when the launch program is
 * built.  A program reused on root identity alone (the exact-last cache, a
 * program prepared at vkCmdBindPipeline, or the bind tail that embeds it)
 * therefore keeps reading the set that was bound when it was built.
 * Minecraft binds its pipeline before its descriptor sets, so every draw
 * after a pipeline switch read the previous pipeline's set: wrong UBOs,
 * wrong colours, and UBO-bounded loops that never terminate.
 *
 * Return that value (0 when the span is root-relative or absent) so every
 * reuse path can require it to be unchanged.
 */
uint64_t
hk_g17p_dma_set_base(const struct hk_cmd_buffer *cmd,
                     const struct hk_shader *s)
{
   if (!s)
      return 0;

   const unsigned root_uniform = s->info.agx3_root_uniform;
   for (unsigned i = 0; i < s->b.info.agx3_uniform_dma_count; ++i) {
      const struct agx3_uniform_dma_span *span =
         &s->b.info.agx3_uniform_dma[i];
      const unsigned pointer = span->pointer;
      if (span->source != AGX3_UNIFORM_DMA_POINTER ||
          pointer == root_uniform || pointer < root_uniform + 4 ||
          (pointer - root_uniform - 4) % 4 != 0 ||
          (pointer - root_uniform - 4) / 4 >= HK_MAX_SETS)
         continue;

      /* The compiler copies through at most one set pointer. */
      const uint64_t base =
         cmd->state.gfx.descriptors.root.sets[(pointer - root_uniform - 4) / 4];
      /* Distinguish "set pointer, currently unbound" (zero page) from "no set
       * span" so that binding a set later still invalidates. */
      return base ? base : AGX_ZERO_PAGE_ADDRESS;
   }
   return 0;
}

/* The buffer descriptor an AGX3_UNIFORM_DMA_BUFFER span names, as the root
 * this draw publishes (set < 0) or the descriptor set bound or pushed at
 * `set` holds it. The shader was compiled for a static layout of that set
 * (agx_shader_key::g17p_dma_static_set_mask), so its contents cannot change
 * before the launch runs. An unbound set reads as a null buffer. */
static bool
hk_g17p_dma_buffer_in(const struct hk_descriptor_state *desc, int set,
                      unsigned descriptor_B, struct hk_buffer_address *buffer)
{
   const void *table;
   size_t table_B;
   if (set < 0) {
      const unsigned buffers_B = hk_root_descriptor_offset(dynamic_buffers);
      if (descriptor_B < buffers_B ||
          descriptor_B + sizeof(*buffer) >
             buffers_B + sizeof(desc->root.dynamic_buffers))
         return false;
      table = &desc->root;
      table_B = sizeof(desc->root);
   } else if (desc->push_active & BITFIELD_BIT(set)) {
      table = desc->push[set]->data;
      table_B = sizeof(desc->push[set]->data);
   } else if (desc->sets[set]) {
      table = desc->sets[set]->mapped_ptr;
      table_B = desc->sets[set]->size;
   } else {
      *buffer = (struct hk_buffer_address){0};
      return true;
   }

   if (descriptor_B + sizeof(*buffer) > table_B)
      return false;
   memcpy(buffer, (const uint8_t *)table + descriptor_B, sizeof(*buffer));
   return true;
}

/* The buffer descriptor of a uniform DMA span from the descriptors bound
 * at bind_point. */
static bool
hk_g17p_dma_buffer(const struct hk_cmd_buffer *cmd,
                   VkPipelineBindPoint bind_point, int set,
                   unsigned descriptor_B, struct hk_buffer_address *buffer)
{
   const struct hk_descriptor_state *desc =
      bind_point == VK_PIPELINE_BIND_POINT_COMPUTE ? &cmd->state.cs.descriptors
                                                   : &cmd->state.gfx.descriptors;
   return hk_g17p_dma_buffer_in(desc, set, descriptor_B, buffer);
}

/* Native compute IOTO reads immutable addresses when the launch runs, as
 * the original constant loads did. Buffer descriptors are eligible only for
 * layouts whose contents cannot be updated after recording (compiler mask).
 * Every dispatch resolves the currently bound compute descriptor state; no
 * pointer-value cache or GPU prepass is introduced. */
bool
hk_m5_compute_uniform_dma_ranges(
   struct hk_cmd_buffer *cmd, const struct hk_shader *s, uint64_t root_ptr,
   struct agx3_ioto_uniform_range *ranges, size_t capacity, size_t *count)
{
   if (!cmd || !s || !ranges || !count || *count > capacity)
      return false;
   const unsigned root = s->info.agx3_root_uniform;
   for (unsigned i = 0; i < s->b.info.agx3_uniform_dma_count; ++i) {
      const struct agx3_uniform_dma_span *span = &s->b.info.agx3_uniform_dma[i];
      const unsigned pointer = span->pointer;
      const int set = pointer == root ? -1
         : pointer >= root + 4 && (pointer - root - 4) % 4 == 0 &&
           (pointer - root - 4) / 4 < HK_MAX_SETS
            ? (int)(pointer - root - 4) / 4 : -2;
      if (set == -2 || (span->dest & 1) || (span->halfs & 1) ||
          (span->offset_B & 3) || *count >= capacity)
         return false;
      if (span->source == AGX3_UNIFORM_DMA_BUFFER) {
         struct hk_buffer_address buffer;
         if (!hk_g17p_dma_buffer(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, set,
                                span->descriptor_B, &buffer))
            return false;
         const uint64_t size_B = (uint64_t)buffer.size & ~UINT64_C(3);
         const unsigned covered = size_B > span->offset_B
            ? MIN2(span->halfs, (size_B - span->offset_B) / 2) : 0;
         if (covered) {
            const uint64_t address = buffer.base_addr + span->offset_B;
            if (address & 3)
               return false;
            ranges[(*count)++] = (struct agx3_ioto_uniform_range){
               .start_halfs = span->dest, .size_halfs = covered,
               .address = address,
            };
         }
         if (covered < span->halfs) {
            if (*count >= capacity)
               return false;
            ranges[(*count)++] = (struct agx3_ioto_uniform_range){
               .start_halfs = span->dest + covered,
               .size_halfs = span->halfs - covered,
               .address = AGX_ZERO_PAGE_ADDRESS,
            };
         }
      } else if (span->source == AGX3_UNIFORM_DMA_POINTER) {
         const uint64_t base = set < 0 ? root_ptr
            : cmd->state.cs.descriptors.root.sets[set];
         if (set < 0 && !base)
            return false;
         const uint64_t address = base ? base + span->offset_B
                                       : AGX_ZERO_PAGE_ADDRESS;
         if (address & 3)
            return false;
         ranges[(*count)++] = (struct agx3_ioto_uniform_range){
            .start_halfs = span->dest, .size_halfs = span->halfs,
            .address = address,
         };
      } else {
         return false;
      }
   }
   return true;
}

static bool
hk_m5_graphics_query_uniforms(struct hk_cmd_buffer *cmd, bool vertex,
                             struct agx3_ioto_uniform_range *range)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const uint64_t packed = UINT64_C(0x01140000600100e8);
   struct agx_scs_layout_gen2 scs;
   if (!dev->m5_native_rq || !dev->m5_rq_instances ||
       !agx_scs_layout_gen2_init(packed, &scs) || !cmd->current_cs.gfx)
      return false;
   const uint64_t size = MAX2(UINT64_C(0x800000),
      ALIGN_POT((uint64_t)scs.end[12], 0x4000));
   struct agx_ptr state = hk_pool_alloc(cmd, size, 0x4000);
   struct agx_ptr uniforms = hk_pool_alloc(cmd, 512, 16);
   uint64_t table = hk_m5_upload_as_table(cmd);
   if (!state.cpu || !uniforms.cpu || !table)
      return false;
   memcpy(state.cpu, dev->m5_rq_user_image, 0x800000);
   if (size > 0x800000)
      memset((uint8_t *)state.cpu + 0x800000, 0, size - 0x800000);
   agx_scs_layout_gen2_parameters(&scs, (uint8_t *)state.cpu + 0x80020);
   memset(uniforms.cpu, 0, 512);
   memcpy(uniforms.cpu, dev->m5_rq_uniforms, sizeof(dev->m5_rq_uniforms));
   uint64_t sizes = (uint64_t)scs.end[6] | ((uint64_t)scs.end[11] << 32);
   uint64_t driver = scs.end[5];
   memcpy((uint8_t *)uniforms.cpu + 48*2, &sizes, 8);
   memcpy((uint8_t *)uniforms.cpu + 52*2, &packed, 8);
   memcpy((uint8_t *)uniforms.cpu + 56*2, &driver, 8);
   uint64_t query_payload = UINT64_C(0x3fffffff00008000);
   memcpy((uint8_t *)uniforms.cpu + 152*2, &query_payload, 8);
   for (unsigned i = 0; i < 80; i += 2) {
      uint64_t pointer;
      memcpy(&pointer, (uint8_t *)uniforms.cpu + i*4, 8);
      if (pointer >= UINT64_C(0x10000000000) && pointer < UINT64_C(0x10000800000)) {
         pointer += state.gpu - UINT64_C(0x10000000000);
         memcpy((uint8_t *)uniforms.cpu + i*4, &pointer, 8);
      }
   }
   memcpy((uint8_t *)uniforms.cpu + 32*2, &table, 8);
   uint32_t context[32] = {0, 0, (uint32_t)packed, (uint32_t)(packed >> 32)};
   const uint32_t offsets[] = {0x18000,0x18200,0x18400,0x18500,0x18700,0x19b00};
   for (unsigned i = 0; i < 6; ++i) {
      uint64_t pointer = i == 4 ? table : state.gpu + offsets[i];
      memcpy(context + 4 + 2*i, &pointer, 8);
   }
   context[20] = scs.end[5];
   context[22] = scs.end[10] - scs.end[5];
   const unsigned levels = (packed >> 16) & 0xff;
   context[23] = (levels + (levels > 1)) * 8;
   uint64_t extra = state.gpu + 0x80050;
   memcpy(context + 26, &extra, 8);
   extra = state.gpu + 0x18900;
   memcpy(context + 30, &extra, 8);
   uint64_t ctx = uniforms.gpu + 352;
   memcpy((uint8_t *)uniforms.cpu + 352, context, sizeof(context));
   memcpy((uint8_t *)uniforms.cpu + 320, &ctx, 8);
   memcpy((uint8_t *)uniforms.cpu + 328, &table, 8);
   *range = (struct agx3_ioto_uniform_range){0, 168, uniforms.gpu};
   hk_cmd_buffer_retain_bo(cmd, dev->m5_rq_code);
   hk_cmd_buffer_retain_bo(cmd, dev->m5_rq_ift);
   if (vertex) cmd->current_cs.gfx->gen1_native_vertex_rq = true;
   else cmd->current_cs.gfx->gen1_native_fragment_rq = true;
   return true;
}

/*
 * Publish the uniform DMA spans of `s` (agx_shader_info::agx3_uniform_dma)
 * after the *count_io ranges already collected: root-relative spans from
 * root_ptr, set-pointer spans from the set addresses of `desc`, buffer spans
 * from the root's or a static set's buffer descriptor (zero page past the
 * buffer's end). Shared by graphics stages and compute kernels; the launch
 * reads the memory when it runs, as the loads did. sources may be NULL.
 */
bool
hk_g17p_dma_ranges(const struct hk_descriptor_state *desc,
                   const struct hk_shader *s, uint64_t root_ptr,
                   struct agx3_ioto_uniform_range *ranges, size_t max_ranges,
                   size_t *count_io, uint8_t *sources)
{
   uint8_t ignored_sources[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   if (!sources)
      sources = ignored_sources;
   if (max_ranges > HK_AGX3_G17P_GRAPHICS_MAX_RANGES)
      return false;
   const unsigned root_uniform = s->info.agx3_root_uniform;
   for (unsigned i = 0; i < s->b.info.agx3_uniform_dma_count; ++i) {
      const struct agx3_uniform_dma_span *span =
         &s->b.info.agx3_uniform_dma[i];
      const unsigned pointer = span->pointer;
      const int set = pointer == root_uniform ? -1
                      : pointer >= root_uniform + 4 &&
                            (pointer - root_uniform - 4) % 4 == 0 &&
                            (pointer - root_uniform - 4) / 4 < HK_MAX_SETS
                         ? (int)(pointer - root_uniform - 4) / 4
                         : -2;
      if (*count_io >= max_ranges || set == -2 ||
          (span->dest & 1) || (span->halfs & 1) || (span->offset_B & 3))
         return false;

      if (span->source == AGX3_UNIFORM_DMA_BUFFER) {
         struct hk_buffer_address buffer;
         if (!hk_g17p_dma_buffer_in(desc, set, span->descriptor_B, &buffer))
            return false;

         /* Dwords below the size rounded up to 4 bytes come from the buffer,
          * later ones from the zero page (AGX3_UNIFORM_DMA_BUFFER). A buffer
          * holding none of the span, such as a null one, is all zero page.
          * Both addresses come from the root's or a static set's contents.
          * A launch replays the buffer address of a span the buffer covers
          * (hk_g17p_launch_replay()); a split or zero-page span depends on
          * the buffer's size, so it is collected again for every draw. */
         const uint64_t size_B = ALIGN_POT((uint64_t)buffer.size, 4);
         const unsigned buffer_halfs =
            size_B > span->offset_B
               ? MIN2(span->halfs, (size_B - span->offset_B) / 2)
               : 0;
         if (buffer_halfs) {
            const uint64_t address = buffer.base_addr + span->offset_B;
            if (address & 3)
               return false;
            sources[*count_io] = buffer_halfs == span->halfs
                                ? HK_G17P_RANGE_SOURCE_BUFFER
                                : HK_G17P_RANGE_SOURCE_UPLOADED;
            ranges[(*count_io)++] = (struct agx3_ioto_uniform_range){
               .start_halfs = span->dest,
               .size_halfs = buffer_halfs,
               .address = address,
            };
         }
         if (buffer_halfs < span->halfs) {
            if (*count_io >= max_ranges)
               return false;
            sources[*count_io] = HK_G17P_RANGE_SOURCE_UPLOADED;
            ranges[(*count_io)++] = (struct agx3_ioto_uniform_range){
               .start_halfs = span->dest + buffer_halfs,
               .size_halfs = span->halfs - buffer_halfs,
               .address = AGX_ZERO_PAGE_ADDRESS,
            };
         }
         continue;
      }

      uint64_t base;
      if (set < 0) {
         base = root_ptr;
         if (!base)
            return false;
      } else {
         base = desc->root.sets[set];
      }
      /* A set the draw never binds is only read on statically dead paths
       * (reading it is invalid usage); give the launch a mapped zero page
       * instead of a fault. */
      const uint64_t address =
         base ? base + span->offset_B : AGX_ZERO_PAGE_ADDRESS;
      if (address & 3)
         return false;

      /* A root-relative span follows the root, a set-pointer span the set's
       * address (hk_g17p_dma_set_base(), which every reuse of a published
       * launcher still compares). */
      sources[*count_io] = set < 0 ? HK_G17P_RANGE_SOURCE_ROOT
                               : HK_G17P_RANGE_SOURCE_SET;
      ranges[(*count_io)++] = (struct agx3_ioto_uniform_range){
         .start_halfs = span->dest,
         .size_halfs = span->halfs,
         .address = address,
      };
   }

   return true;
}

/* sources, when not NULL, receives each range's enum hk_g17p_range_source.
 * Root ranges address root_ptr, the root this stage's program follows. */
static bool
hk_g17p_graphics_uniform_ranges(
   struct hk_cmd_buffer *cmd, struct hk_shader *s,
   const struct hk_linked_shader *linked, uint64_t root_ptr,
   struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES],
   size_t *range_count, uint8_t *sources)
{
   uint8_t ignored_sources[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   if (!sources)
      sources = ignored_sources;
   /* A range whose source is not recorded below is not re-derivable: the
    * compact launcher cache must then republish rather than patch it. */
   memset(sources, HK_G17P_RANGE_SOURCE_UPLOADED,
          HK_AGX3_G17P_GRAPHICS_MAX_RANGES);

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   size_t count = 0;
   const unsigned root_uniform = s->info.agx3_root_uniform;
   const unsigned texture_uniform_base = s->info.agx3_gen1_fragment_rq ? 168 : 0;
   if (s->info.agx3_gen1_fragment_rq) {
      if (linked->g17p_prolog_uses_root ||
          agx_tilebuffer_spills(&cmd->state.gfx.render.tilebuffer) ||
          hk_cmd_uses_blend_constant(cmd) ||
          !(agx_chip_g17_m5(dev->dev.chip)
             ? hk_m5_graphics_query_uniforms(cmd, s->info.stage != MESA_SHADER_FRAGMENT, &ranges[count])
             : hk_gen1_upload_fragment_query_state(cmd, s->info.stage != MESA_SHADER_FRAGMENT, &ranges[count])))
         return hk_g17p_graphics_ranges_refused(range_count,count);
      count++;
   }

   const bool native_spills = s->info.stage == MESA_SHADER_FRAGMENT &&
      agx_tilebuffer_spills(&cmd->state.gfx.render.tilebuffer);

   if (s->info.stage == MESA_SHADER_VERTEX) {
      const unsigned nr_attribs = DIV_ROUND_UP(
         BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4);
      const unsigned uniform_base = (s->info.agx3_indexed_sampler_valid
         ? AGX_ABI_G17P_TEXTURE_POINTER_HALFS
         : 0) + texture_uniform_base;
      if (nr_attribs > AGX_MAX_VBUFS)
         return hk_g17p_graphics_ranges_refused(range_count, count);

      if (nr_attribs != 0) {
         if (!root_ptr)
            return hk_g17p_graphics_ranges_refused(range_count, count);

         sources[count] = HK_G17P_RANGE_SOURCE_ROOT;
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = uniform_base + AGX_ABI_VUNI_VBO_BASE(0),
            .size_halfs = 4 * nr_attribs,
            .address =
               root_ptr + hk_root_descriptor_offset(draw.attrib_base),
         };
         sources[count] = HK_G17P_RANGE_SOURCE_ROOT;
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = uniform_base +
                           AGX_ABI_VUNI_VBO_CLAMP(nr_attribs, 0),
            .size_halfs = 2 * nr_attribs,
            .address =
               root_ptr + hk_root_descriptor_offset(draw.attrib_clamps),
         };
      }

      /* The compiler's vertex ABI reserves these independent ranges after
       * attribute bases/clamps. Direct draws already upload baseVertex and
       * firstInstance together; native Ioto can bind them without a prolog
       * specialization, shader patch or pre-render compute dispatch. */
      if (cmd->state.gfx.draw_params) {
         sources[count] = HK_G17P_RANGE_SOURCE_DRAW_PARAMS;
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = uniform_base +
                           AGX_ABI_VUNI_FIRST_VERTEX(nr_attribs),
            .size_halfs = 4,
            .address = cmd->state.gfx.draw_params,
         };
      }
      if (cmd->state.gfx.draw_id_ptr) {
         /* The shader reads gl_DrawID as a single half, but a uniform range
          * is loaded in 32-bit dwords -- agx3_ioto_uniform_program.c rejects
          * an odd size_halfs outright -- so a one-half range could never be
          * published and every gl_DrawID vertex program was refused. Publish
          * the containing dword instead. AGX_ABI_VUNI_DRAW_ID is always
          * FIRST_VERTEX + 4 and AGX_ABI_VUNI_VERTEX_PARAMS is at least
          * FIRST_VERTEX + 8, so the second half is reserved ABI padding that
          * no other range claims. hk_cmd_draw.c uploads the matching dword. */
         sources[count] = HK_G17P_RANGE_SOURCE_DRAW_ID;
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = uniform_base +
                           AGX_ABI_VUNI_DRAW_ID(nr_attribs),
            .size_halfs = 2,
            .address = cmd->state.gfx.draw_id_ptr,
         };
      }

   } else if (s->info.stage == MESA_SHADER_MESH ||
              s->info.stage == MESA_SHADER_TASK) {
      /* Native TA mesh and object bodies have the same driver-root/descriptor
       * ranges, without vertex attribute or base-instance ABI ranges. */
   } else if ((s->info.stage == MESA_SHADER_GEOMETRY ||
               s->info.stage == MESA_SHADER_TESS_EVAL) &&
              s->b.info.stage == MESA_SHADER_VERTEX) {
      /* The compiled geometry/tessellation raster variant owns its root
       * layout. Do not infer it from the original API stage. */
   } else if (s->info.stage == MESA_SHADER_FRAGMENT) {
      if (native_spills) {
         const uint64_t pointer = hk_upload_native_spilled_rt_descs(cmd);
         if (!pointer || !linked->g17p_emrt_heap_base)
            return hk_g17p_graphics_ranges_refused(range_count, count);
         sources[count] = HK_G17P_RANGE_SOURCE_UPLOADED;
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = linked->g17p_emrt_heap_base,
            .size_halfs = 4, .address = pointer,
         };
      }
      if (hk_cmd_uses_blend_constant(cmd)) {
         /* The non-monolithic blend epilog reads the four constants from
          * the base selected by its linked key. It is half4 for the standard
          * ABI and moves beyond the main shader footprint when HAL200 native
          * texture/sampler pointers occupy halves0..7. */
         const uint16_t blend_base = linked->g17p_blend_constant_base
                                        ? linked->g17p_blend_constant_base
                                        : AGX_ABI_FUNI_BLEND_R;
         if (!root_ptr ||
             !hk_agx3_g17p_fragment_blend_constants_range(
                blend_base,
                root_ptr + hk_root_descriptor_offset(draw.blend_constant),
                &ranges[count]))
            return hk_g17p_graphics_ranges_refused(range_count, count);
         sources[count] = HK_G17P_RANGE_SOURCE_ROOT;
         count++;
      }
   } else if (s->info.stage != MESA_SHADER_MESH &&
              s->info.stage != MESA_SHADER_TASK) {
      return hk_g17p_graphics_ranges_refused(range_count, count);
   }

   if (s->info.agx3_combined_sampler_valid) {
      if (s->info.stage != MESA_SHADER_FRAGMENT || s->info.set_count != 0)
         return hk_g17p_graphics_ranges_refused(range_count, count);

      struct hk_descriptor_state *descriptors =
         &cmd->state.gfx.descriptors;
      struct hk_descriptor_set *set = descriptors->sets[0];
      const void *set_map = set ? set->mapped_ptr : NULL;
      uint32_t set_size = set ? set->size : 0;
      uint64_t set_addr = set ? hk_descriptor_set_addr(set) : 0;
      if (!set && descriptors->push[0] && descriptors->root.sets[0]) {
         set_map = descriptors->push[0]->data;
         set_size = sizeof(descriptors->push[0]->data);
         set_addr = descriptors->root.sets[0];
      }
      const struct agx_g17p_combined_sampler_binding *binding =
         &s->info.agx3_combined_sampler;
      if (!set_map || !set_addr ||
          binding->texture_offset_B > set_size ||
          set_size - binding->texture_offset_B < 32 ||
          binding->sampler_index_offset_B > set_size ||
          set_size - binding->sampler_index_offset_B < sizeof(uint16_t))
         return hk_g17p_graphics_ranges_refused(range_count, count);

      uint16_t sampler_index;
      memcpy(&sampler_index,
             (const uint8_t *)set_map +
                binding->sampler_index_offset_B,
             sizeof(sampler_index));

      const unsigned sampler_element_size =
         agx_chip_sampler_heap_element_size(dev->dev.chip);
      if (!dev->samplers.table.bo || !dev->samplers.table.bo->va ||
          dev->samplers.table.alloc > UINT16_MAX ||
          sampler_element_size > UINT16_MAX)
         return hk_g17p_graphics_ranges_refused(range_count, count);

      const struct hk_agx3_g17p_combined_sampler_heap heap = {
         .descriptor_set_va = set_addr,
         .sampler_heap_va = dev->samplers.table.bo->va->addr,
         .texture_offset_B = binding->texture_offset_B,
         .sampler_index = sampler_index,
         .sampler_count = dev->samplers.table.alloc,
         .sampler_element_size_B = sampler_element_size,
      };
      uint64_t pointers[2];
      if (!hk_agx3_g17p_combined_sampler_pointers(&heap, pointers))
         return hk_g17p_graphics_ranges_refused(range_count, count);

      /* HAL200 FragmentProgramVariant::finalize (374968..3749c0) maps
       * compiler bindpoints 0x31/0x35 to driver qwords +0xc0/+0xe0.
       * encodeAndEmitRenderState writes the raw texture/sampler table VAs
       * there (4c5580/4c5598); ESL copies them to uniform dwords 0..3.
       * Keep this upload immutable for the command-buffer lifetime, just
       * like the per-draw graphics state. Both shader indices are zero.
       */
      const unsigned cache_index =
         ((pointers[0] >> 4) ^ (pointers[0] >> 32) ^
          (pointers[1] >> 4) ^ (pointers[1] >> 32)) &
         (ARRAY_SIZE(cmd->state.gfx.g17p_combined_sampler_pointers) - 1);
      __typeof__(cmd->state.gfx.g17p_combined_sampler_pointers[0]) *cached =
         &cmd->state.gfx.g17p_combined_sampler_pointers[cache_index];
      uint64_t pointers_va;
      if (cached->valid && !memcmp(cached->pointers, pointers,
                                   sizeof(pointers))) {
         pointers_va = cached->address;
      } else {
         /* The cache stays keyed on the set address; what is published is
          * the 32-byte-aligned texture state (hk_g17p_texture_state_pointer). */
         const uint64_t published[2] = {
            hk_g17p_texture_state_pointer(
               cmd, pointers[0],
               (const uint8_t *)set_map + binding->texture_offset_B,
               set_size - binding->texture_offset_B),
            pointers[1],
         };
         pointers_va = published[0]
            ? hk_pool_upload(cmd, published, sizeof(published), 8)
            : 0;
         if (pointers_va) {
            memcpy(cached->pointers, pointers, sizeof(pointers));
            memcpy(cached->published, published, sizeof(published));
            cached->address = pointers_va;
            cached->valid = true;
         }
      }
      if (!pointers_va)
         return hk_g17p_graphics_ranges_refused(range_count, count);

      sources[count] = HK_G17P_RANGE_SOURCE_UPLOADED;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = texture_uniform_base,
         .size_halfs = 8,
         .address = pointers_va,
      };
   }

   if (native_spills && s->info.agx3_combined_sampler_valid)
      return hk_g17p_graphics_ranges_refused(range_count, count);

   if (s->info.agx3_indexed_sampler_valid || native_spills) {
      if ((s->b.info.stage != MESA_SHADER_FRAGMENT &&
           s->b.info.stage != MESA_SHADER_VERTEX &&
           !(agx_chip_g17_m5(dev->dev.chip) &&
             (s->b.info.stage == MESA_SHADER_MESH ||
              s->b.info.stage == MESA_SHADER_TASK))) ||
          !dev->sampled_images.base_va || !dev->samplers.table.bo ||
          !dev->rodata.indexed_heap_pointers)
         return hk_g17p_graphics_ranges_refused(range_count, count);
      const uint64_t pointers_va = dev->rodata.indexed_heap_pointers;
      sources[count] = HK_G17P_RANGE_SOURCE_CONSTANT;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = texture_uniform_base, .size_halfs = 8, .address = pointers_va,
      };
   }

   if (hk_agx3_g17p_graphics_needs_root(
          s->info.agx3_uses_root, linked->g17p_prolog_uses_root,
          s->info.set_count)) {
      if (!root_ptr)
         return hk_g17p_graphics_ranges_refused(range_count, count);

      if (!hk_agx3_g17p_root_range(root_uniform, s->info.set_count, root_ptr,
                                   &ranges[count]))
         return hk_g17p_graphics_ranges_refused(range_count, count);
      sources[count] = HK_G17P_RANGE_SOURCE_ROOT;
      count++;
   }

   /*
    * Promoted constants. agx_opt_promote_constants() parks a shader's
    * constant pool in the binary's rodata section and addresses it through a
    * uniform, so the driver has to publish that address as a uniform range --
    * exactly as the compute path does (hk_cmd_dispatch.c). Without it a
    * fragment shader as simple as a constant colour is refused, because the
    * compiler promotes its three floats.
    *
    * The data is read through the uniform, not the program counter, so it is
    * addressed in the shader's own BO even though graphics executes from the
    * linked binary. Same bounds checks as compute: the range must lie inside
    * that BO. G17P loads dwords; an odd final half is covered by the zeroed
    * alignment padding before the shader code.
    */
   if (count < HK_AGX3_G17P_GRAPHICS_MAX_RANGES && s->b.info.rodata.size_16 != 0) {
      const struct agx_rodata *rodata = &s->b.info.rodata;
      if (!s->bo || !s->bo->va ||
          !hk_agx3_g17p_rodata_range(
             rodata->base_uniform, rodata->size_16, hk_shader_code_va(s),
             rodata->offset, s->b.info.binary_size, &ranges[count]))
         return hk_g17p_graphics_ranges_refused(range_count, count);
      sources[count] = HK_G17P_RANGE_SOURCE_CONSTANT;
      count++;
   } else if (s->b.info.rodata.size_16 != 0) {
      /* No room left in the program's range budget. */
      return hk_g17p_graphics_ranges_refused(range_count, count);
   }

   /* Uniform DMA: the compiler replaced constant-index loads with reads of
    * uniform spans (agx_shader_info::agx3_uniform_dma). Publish each from the
    * address this draw gives it; the launch reads the memory when it runs,
    * as the loads did. */
   if (!hk_g17p_dma_ranges(&cmd->state.gfx.descriptors, s, root_ptr, ranges,
                           HK_AGX3_G17P_GRAPHICS_MAX_RANGES, &count, sources))
      return hk_g17p_graphics_ranges_refused(range_count, count);

   *range_count = count;
   return count <= HK_AGX3_G17P_GRAPHICS_MAX_RANGES;
}

/* A G17P graphics stage program of one linked executable for one uniform
 * layout and address shape (hk_g17p_patch.h). A draw copies it with its own
 * range base immediates. */
#define HK_G17P_GFX_LOADER_MAX_SHAPES 4

struct hk_g17p_gfx_loader {
   struct hk_g17p_gfx_loader *next;
   const struct hk_shader *s;
   struct agx_ioto_g17p_imageblock imageblock;
   uint16_t range_start[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   uint16_t range_size[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   uint8_t range_count;
   uint16_t length;
   struct hk_g17p_patch_plan plan;
   uint8_t bytes[];
};

static inline bool
hk_g17p_gfx_loader_matches(const struct hk_g17p_gfx_loader *l,
                           const struct hk_shader *s,
                           const struct agx_ioto_g17p_imageblock *imageblock,
                           const struct agx3_ioto_uniform_range *ranges,
                           size_t range_count)
{
   if (l->s != s || l->range_count != range_count ||
       l->imageblock.tile_size_px != imageblock->tile_size_px ||
       l->imageblock.sample_size_B != imageblock->sample_size_B ||
       l->imageblock.nr_samples != imageblock->nr_samples)
      return false;

   for (unsigned i = 0; i < range_count; ++i) {
      if (l->range_start[i] != ranges[i].start_halfs ||
          l->range_size[i] != ranges[i].size_halfs)
         return false;
   }
   return true;
}

static struct hk_g17p_gfx_loader *
hk_g17p_gfx_loader_create(struct hk_device *dev, const struct hk_shader *s,
                          const struct hk_agx3_g17p_graphics_program_desc *desc,
                          bool independent)
{
   struct hk_agx3_g17p_graphics_program_desc composed = *desc;
   /* Any aligned address inside the USC window: with absolute constant
    * program branches the bytes do not depend on it. */
   composed.program_va = dev->dev.shader_base + 0x10000;
   composed.patchable = independent;

   struct agx3_g17p_graphics_program_template *t = malloc(sizeof(*t));
   if (!t)
      return NULL;

   struct hk_g17p_gfx_loader *l = NULL;
   if (!hk_agx3_g17p_graphics_program_template_generate(dev->dev.chip,
                                                         &composed, t) ||
       (desc->constant_program.code_size &&
        !desc->constant_program.absolute_branch))
      goto out;

   l = calloc(1, sizeof(*l) + t->program.length);
   if (!l)
      goto out;

   l->s = s;
   l->imageblock = desc->imageblock;
   l->range_count = desc->uniform_range_count;
   l->length = t->program.length;
   memcpy(l->bytes, t->program.bytes, t->program.length);
   for (unsigned i = 0; i < l->range_count; ++i) {
      l->range_start[i] = desc->uniform_ranges[i].start_halfs;
      l->range_size[i] = desc->uniform_ranges[i].size_halfs;
   }

   if (!hk_g17p_patch_plan_init(&l->plan, desc->uniform_ranges,
                                desc->uniform_range_count, t->relocations,
                                t->relocation_count, t->program.length,
                                independent)) {
      free(l);
      l = NULL;
   }

out:
   free(t);
   return l;
}

static uint64_t
hk_g17p_gfx_loader_emit(struct hk_cmd_buffer *cmd,
                        const struct hk_g17p_gfx_loader *l,
                        const struct agx3_ioto_uniform_range *ranges)
{
   struct hk_ioto_reservation r;
   if (!hk_cmd_reserve_ioto_program(cmd, l->length, "G17P graphics loader",
                                    &r))
      return 0;

   memcpy(r.map, l->bytes, l->length);
   hk_g17p_patch_plan_apply(&l->plan, ranges, r.map);

   return hk_cmd_publish_ioto_program(cmd, &r, l->length);
}

static bool
hk_g17p_graphics_program_desc_init(
   struct hk_cmd_buffer *cmd, struct hk_shader *s,
   struct hk_linked_shader *linked, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, struct hk_agx3_g17p_graphics_program_desc *desc);

/* Publish l with the draw's range addresses, or the last program when they
 * are the addresses it was published with. */
static uint64_t
hk_g17p_gfx_loader_publish(struct hk_cmd_buffer *cmd,
                           struct hk_linked_shader *linked,
                           struct hk_g17p_graphics_ioto_cache_entry *cached,
                           const struct hk_g17p_gfx_loader *l,
                           const struct agx3_ioto_uniform_range *ranges,
                           size_t range_count)
{
   /* The program is a function of the loader and the range addresses. */
   if (cached->loader == l && cached->loader_linked == linked &&
       cached->loader_program &&
       !memcmp(cached->loader_ranges, ranges, range_count * sizeof(ranges[0])))
      return cached->loader_program;

   const uint64_t published = hk_g17p_gfx_loader_emit(cmd, l, ranges);
   if (!published)
      return 0;

   cached->loader = l;
   cached->loader_linked = linked;
   cached->loader_program = published;
   memcpy(cached->loader_ranges, ranges, range_count * sizeof(ranges[0]));
   return published;
}

/* The draw state that selects which ranges hk_g17p_graphics_uniform_ranges()
 * collects for a shader; everything else in their layout is the shader's. */
static inline uint8_t
hk_g17p_launch_flags(const struct hk_cmd_buffer *cmd, const struct hk_shader *s)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   uint8_t flags = gfx->root ? 1 : 0;
   if (s->info.stage == MESA_SHADER_VERTEX) {
      flags |= (gfx->draw_params ? 2 : 0) | (gfx->draw_id_ptr ? 4 : 0);
   } else if (s->info.stage == MESA_SHADER_FRAGMENT) {
      flags |= (hk_cmd_uses_blend_constant(cmd) ? 8 : 0) |
               (agx_tilebuffer_spills(&gfx->render.tilebuffer) ? 16 : 0);
   }
   return flags;
}

static inline struct hk_g17p_launch *
hk_g17p_launch_slot(struct hk_cmd_buffer *cmd, unsigned stage,
                    const struct hk_linked_shader *linked)
{
   const uintptr_t key = (uintptr_t)linked;
   return &cmd->state.gfx.g17p_launch[stage][((key >> 6) ^ (key >> 12)) &
                                             (HK_G17P_LAUNCH_SLOTS - 1)];
}

/* Remember how ranges, just collected with sources and published with l,
 * derive from the draw state. */
static void
hk_g17p_launch_record(struct hk_cmd_buffer *cmd, struct hk_g17p_launch *launch,
                      const struct hk_shader *s,
                      const struct hk_linked_shader *linked,
                      const struct hk_g17p_gfx_loader *l, uint8_t flags,
                      const struct hk_shader *hw_vs, uint64_t root,
                      const struct agx3_ioto_uniform_range *ranges,
                      size_t count, const uint8_t *sources)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct agx_tilebuffer_layout *tib = &gfx->render.tilebuffer;

   launch->valid = false;
   if (count > ARRAY_SIZE(launch->source))
      return;

   for (unsigned i = 0; i < count; ++i) {
      uint64_t offset = 0;
      uint8_t set = 0;
      uint16_t descriptor_B = 0;
      switch (sources[i]) {
      case HK_G17P_RANGE_SOURCE_CONSTANT:
         offset = ranges[i].address;
         break;
      case HK_G17P_RANGE_SOURCE_ROOT:
         offset = ranges[i].address - root;
         break;
      case HK_G17P_RANGE_SOURCE_DRAW_PARAMS:
      case HK_G17P_RANGE_SOURCE_DRAW_ID:
         break;
      case HK_G17P_RANGE_SOURCE_SET: {
         /* The set-pointer span the range was collected from: the one
          * copied to its first uniform. */
         const struct agx3_uniform_dma_span *span = NULL;
         for (unsigned j = 0; j < s->b.info.agx3_uniform_dma_count; ++j) {
            if (s->b.info.agx3_uniform_dma[j].dest == ranges[i].start_halfs)
               span = &s->b.info.agx3_uniform_dma[j];
         }
         if (!span || span->source != AGX3_UNIFORM_DMA_POINTER)
            return;
         set = (span->pointer - s->info.agx3_root_uniform - 4) / 4;
         offset = span->offset_B;
         break;
      }
      case HK_G17P_RANGE_SOURCE_BUFFER: {
         /* The buffer span the range was collected from, as above. */
         const struct agx3_uniform_dma_span *span = NULL;
         for (unsigned j = 0; j < s->b.info.agx3_uniform_dma_count; ++j) {
            if (s->b.info.agx3_uniform_dma[j].dest == ranges[i].start_halfs)
               span = &s->b.info.agx3_uniform_dma[j];
         }
         if (!span || span->source != AGX3_UNIFORM_DMA_BUFFER ||
             span->halfs != ranges[i].size_halfs ||
             span->descriptor_B > UINT16_MAX)
            return;
         set = span->pointer == s->info.agx3_root_uniform
                  ? HK_G17P_LAUNCH_ROOT_SET
                  : (span->pointer - s->info.agx3_root_uniform - 4) / 4;
         descriptor_B = span->descriptor_B;
         offset = span->offset_B;
         break;
      }
      default:
         return;
      }
      launch->source[i] = sources[i];
      launch->set[i] = set;
      launch->descriptor_B[i] = descriptor_B;
      launch->offset[i] = offset;
      launch->start_halfs[i] = ranges[i].start_halfs;
      launch->size_halfs[i] = ranges[i].size_halfs;
   }

   launch->s = s;
   launch->linked = linked;
   launch->loader = l;
   launch->hw_vs = hw_vs;
   launch->tile_size_px = tib->tile_size;
   launch->sample_size_B = tib->sample_size_B;
   launch->nr_samples = tib->nr_samples;
   launch->topology = gfx->topology;
   launch->provoking = gfx->provoking;
   launch->flags = flags;
   launch->count = count;
   launch->valid = true;
}

/* Whether the recorded launch applies to this draw, admission included: the
 * admission of a vertex program depends only on the program, that of a
 * fragment program also on the spill state (in the flags), the native RT0
 * layout and the varying-coefficient inputs (hk_g17p_basic_graphics_
 * program_supported()). */
static inline bool
hk_g17p_launch_matches(const struct hk_cmd_buffer *cmd,
                       const struct hk_g17p_launch *launch,
                       const struct hk_shader *s,
                       const struct hk_linked_shader *linked, uint8_t flags,
                       const struct hk_shader *hw_vs)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct agx_tilebuffer_layout *tib = &gfx->render.tilebuffer;

   if (!launch->valid || launch->s != s || launch->linked != linked ||
       launch->flags != flags || launch->tile_size_px != tib->tile_size ||
       launch->sample_size_B != tib->sample_size_B ||
       launch->nr_samples != tib->nr_samples)
      return false;

   return s->info.stage != MESA_SHADER_FRAGMENT ||
          (gfx->render.g17p_rt0_layout_supported && hw_vs &&
           launch->hw_vs == hw_vs && launch->topology == gfx->topology &&
           launch->provoking == gfx->provoking);
}

/* The stage program of s from a matching recorded launch: the ranges
 * hk_g17p_graphics_uniform_ranges() would collect for root, published with
 * the loader of the recording. 0 when they do not fit that loader. */
static inline uint64_t
hk_g17p_launch_replay(struct hk_cmd_buffer *cmd,
                      const struct hk_g17p_launch *launch,
                      struct hk_linked_shader *linked,
                      struct hk_g17p_graphics_ioto_cache_entry *cached,
                      uint64_t root)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const unsigned count = launch->count;
   struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   for (unsigned i = 0; i < count; ++i) {
      uint64_t address;
      switch (launch->source[i]) {
      case HK_G17P_RANGE_SOURCE_ROOT:
         address = root + launch->offset[i];
         break;
      case HK_G17P_RANGE_SOURCE_DRAW_PARAMS:
         address = gfx->draw_params;
         break;
      case HK_G17P_RANGE_SOURCE_DRAW_ID:
         address = gfx->draw_id_ptr;
         break;
      case HK_G17P_RANGE_SOURCE_SET: {
         /* As hk_g17p_graphics_uniform_ranges(): an unbound set reads the
          * zero page. */
         const uint64_t base = gfx->descriptors.root.sets[launch->set[i]];
         address = base ? base + launch->offset[i] : AGX_ZERO_PAGE_ADDRESS;
         if (address & 3)
            return 0;
         break;
      }
      case HK_G17P_RANGE_SOURCE_BUFFER: {
         /* As hk_g17p_graphics_uniform_ranges(), for a buffer that still
          * covers the whole span; otherwise the span splits differently. */
         struct hk_buffer_address buffer;
         if (!hk_g17p_dma_buffer(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 launch->set[i] == HK_G17P_LAUNCH_ROOT_SET
                                    ? -1 : launch->set[i],
                                 launch->descriptor_B[i], &buffer) ||
             ALIGN_POT((uint64_t)buffer.size, 4) <
                launch->offset[i] + 2 * launch->size_halfs[i])
            return 0;
         address = buffer.base_addr + launch->offset[i];
         if (address & 3)
            return 0;
         break;
      }
      default:
         address = launch->offset[i];
         break;
      }
      ranges[i] = (struct agx3_ioto_uniform_range){
         .start_halfs = launch->start_halfs[i],
         .size_halfs = launch->size_halfs[i],
         .address = address,
      };
   }

   if (!hk_g17p_patch_plan_fits(&launch->loader->plan, ranges))
      return 0;

   return hk_g17p_gfx_loader_publish(cmd, linked, cached, launch->loader,
                                     ranges, count);
}

/* The stage program of s for the current draw: the loader of its range
 * layout, copied with the draw's range addresses. 0 if the layout has no
 * loader (or on error, which is then set on the command buffer). */
static uint64_t
hk_g17p_publish_gfx_loader(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                           struct hk_linked_shader *linked,
                           struct hk_g17p_graphics_ioto_cache_entry *cached,
                           const struct agx3_ioto_uniform_range *ranges,
                           size_t range_count)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct agx_tilebuffer_layout *tib = &cmd->state.gfx.render.tilebuffer;
   const struct agx_ioto_g17p_imageblock imageblock = {
      .sample_size_B = tib->sample_size_B,
      /* Empty M5 fragment programs use the single-sample loader form. */
      .nr_samples = agx_chip_g17_m5(dev->dev.chip) &&
                       s->info.stage == MESA_SHADER_FRAGMENT &&
                       !tib->sample_size_B && !tib->nr_samples
                       ? 1 : tib->nr_samples,
      .tile_size_px = tib->tile_size,
   };

   const struct hk_g17p_gfx_loader *l = cached->loader;
   unsigned shapes = 0;
   if (!l || cached->loader_linked != linked ||
       !hk_g17p_gfx_loader_matches(l, s, &imageblock, ranges, range_count) ||
       !hk_g17p_patch_plan_fits(&l->plan, ranges)) {
      l = NULL;
      for (struct hk_g17p_gfx_loader *it =
              p_atomic_read(&linked->g17p_gfx_loaders);
           it; it = it->next) {
         if (!hk_g17p_gfx_loader_matches(it, s, &imageblock, ranges,
                                         range_count))
            continue;
         if (hk_g17p_patch_plan_fits(&it->plan, ranges)) {
            l = it;
            break;
         }
         ++shapes;
      }
   }

   if (!l) {
      struct hk_agx3_g17p_graphics_program_desc desc;
      if (!hk_g17p_graphics_program_desc_init(cmd, s, linked, ranges,
                                              range_count, &desc))
         return 0;

      /* A layout whose ranges keep moving relative to each other gets one
       * program with a base per range, which fits any addresses. */
      struct hk_g17p_gfx_loader *created = hk_g17p_gfx_loader_create(
         dev, s, &desc, shapes >= HK_G17P_GFX_LOADER_MAX_SHAPES);
      if (!created)
         return 0;

      while (true) {
         struct hk_g17p_gfx_loader *head =
            p_atomic_read(&linked->g17p_gfx_loaders);
         created->next = head;
         if (p_atomic_cmpxchg_ptr(&linked->g17p_gfx_loaders, head, created) ==
             head)
            break;
      }
      l = created;
   }

   const uint64_t published =
      hk_g17p_gfx_loader_publish(cmd, linked, cached, l, ranges, range_count);
   if (!published)
      return 0;

   /* The program names the linked executable, and promoted constants and the
    * constant program live with the shader's code. */
   hk_cmd_buffer_retain_bo(cmd, linked->b.bo);
   if (s->b.info.rodata.size_16 != 0 || s->b.info.has_preamble)
      hk_cmd_buffer_retain_bo(cmd, s->bo);
   return published;
}

void
hk_g17p_linked_graphics_cache_finish(struct hk_device *dev,
                                     struct hk_linked_shader *linked)
{
   struct hk_g17p_gfx_loader *loader = p_atomic_read(&linked->g17p_gfx_loaders);
   while (loader) {
      struct hk_g17p_gfx_loader *next = loader->next;
      free(loader);
      loader = next;
   }
   linked->g17p_gfx_loaders = NULL;
}

/* agx_usc_uniform() publishes at most 64 halves per record. */
static void
hk_usc_uniform_split(struct agx_usc_builder *b, unsigned start_halfs,
                     unsigned size_halfs, uint64_t buffer)
{
   for (unsigned offset = 0; offset < size_halfs; offset += 64) {
      agx_usc_uniform(b, start_halfs + offset, MIN2(64, size_halfs - offset),
                      buffer + offset * 2);
   }
}

static bool
hk_g17p_graphics_program_desc_init(
   struct hk_cmd_buffer *cmd, struct hk_shader *s,
   struct hk_linked_shader *linked, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, struct hk_agx3_g17p_graphics_program_desc *desc)
{
   const bool fragment = s->info.stage == MESA_SHADER_FRAGMENT;
   const uint32_t pass_type = linked->b.fragment_control.opaque[0] >> 29;
   *desc = (struct hk_agx3_g17p_graphics_program_desc){
      /* The container's entry point, not the allocation base: the shader
       * loader reads the block-size dword at entry - 0x80. */
      .shader_va = linked->b.bo->va->addr + linked->b.entry_offset,
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
      .shader_binary_size = linked->b.size,
      .scratch_size =
         MAX2(linked->b.scratch_size, linked->b.agx3_spectpr_scratch_size),
      .nr_gprs = linked->b.nr_gprs,
      .compiler_d_valid = linked->b.g17p_compiler_d_valid,
      .compiler_d = linked->b.g17p_compiler_d,
      .fragment = fragment,
      .writes_sample_mask =
         fragment && (pass_type == AGX_PASS_TYPE_PUNCH_THROUGH ||
                      pass_type == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH),
      .colour_classes = linked->b.g17p_colour_classes,
      .imageblock = {
         .sample_size_B = cmd->state.gfx.render.tilebuffer.sample_size_B,
         .nr_samples = cmd->state.gfx.render.tilebuffer.nr_samples,
         .tile_size_px = cmd->state.gfx.render.tilebuffer.tile_size,
      },
   };

   /* Attachmentless discard never resolves a raster sample count. Its inert
    * fragment loader uses the native empty single-sample imageblock without
    * changing the render state's deferred sample count. */
   if (agx_chip_g17_m5(hk_cmd_buffer_device(cmd)->dev.chip) && fragment &&
       desc->imageblock.nr_samples == 0 && desc->imageblock.sample_size_B == 0)
      desc->imageblock.nr_samples = 1;

   if (s->b.info.has_preamble &&
       (!s->bo || !s->bo->va || s->code_size != s->b.info.binary_size ||
        !hk_agx3_g17p_constant_program_from_binary(
           s->code_ptr, s->code_size, s->b.info.preamble_offset,
           s->b.info.preamble_size, hk_shader_code_va(s),
           s->b.info.binary_size, &desc->constant_program))) {
      mesa_loge_once("G17P graphics constant-program extent refused");
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   desc->constant_program.request_prelude =
      s->b.info.agx3_preamble_uniform_inputs;
   if (s->b.info.agx3_preamble_tps_inputs) {
      if (!s->b.info.has_preamble ||
          s->b.info.preamble_scratch_size <
             s->b.info.agx3_preamble_inputs.storage_bytes) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }
      desc->cprog_inputs = &s->b.info.agx3_preamble_inputs;
   }
   return true;
}

uint64_t
hk_upload_native_mesh_program(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                              struct hk_linked_shader *linked, bool object,
                              bool task)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!linked ||
       (dev->dev.chip == AGX_CHIP_G17P
          ? (!linked->b.g17p_compiler_d_valid || hk_mesh_mode() != HK_MESH_MODE_NATIVE_PROBE)
          : (dev->dev.chip != AGX_CHIP_G17S ||
             !(dev->dev.params.features & DRM_ASAHI_FEATURE_RENDER_RESOURCES_V2))))
      goto refused;
   struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   size_t count = 0;
   /* An API task shader is a compiled object program with the ordinary
    * driver-root/descriptor ranges and its own threadgroup memory. */
   const bool task_object = object && task && dev->dev.chip == AGX_CHIP_G17S &&
                            s->info.stage == MESA_SHADER_TASK;
   if (task_object) {
      if (!hk_g17p_graphics_uniform_ranges(cmd, s, linked,
                                           cmd->state.gfx.root, ranges, &count,
                                           NULL))
         goto refused;
   } else if (object) {
      const uint64_t grid = cmd->state.gfx.mesh.object_params ?: cmd->state.gfx.mesh.grid;
      if (!grid)
         goto refused;
      ranges[count++] = (struct agx3_ioto_uniform_range) {
         .start_halfs = 0, .size_halfs = 4,
         .address = hk_pool_upload(cmd, &grid, sizeof(grid), 8),
      };
   } else if (!hk_g17p_graphics_uniform_ranges(cmd, s, linked,
                                               cmd->state.gfx.root, ranges,
                                               &count, NULL)) {
      goto refused;
   }
   struct agx3_g17p_graphics_program_desc d = {
      .shader_va = linked->b.bo->va->addr + linked->b.entry_offset,
      .uniform_ranges = ranges, .uniform_range_count = count,
      .shader_binary_size = linked->b.size,
      .scratch_size = MAX2(linked->b.scratch_size,
                           dev->dev.chip == AGX_CHIP_G17P
                              ? s->b.info.scratch_size : linked->b.agx3_spectpr_scratch_size),
      .nr_gprs = linked->b.nr_gprs,
      .compiler_d_valid = linked->b.g17p_compiler_d_valid,
      .compiler_d = linked->b.g17p_compiler_d,
      .threadgroup_memory_size_B = dev->dev.chip == AGX_CHIP_G17P && object
                                    ? 0 : s->b.info.local_size,
   };
   size_t bound;
   if (!(dev->dev.chip == AGX_CHIP_G17P
          ? agx3_g17p_graphics_program_bound(dev->dev.chip, ranges, count, &bound)
          : agx3_g17g_graphics_program_bound(dev->dev.chip, ranges, count, &bound)))
      goto refused;
   struct agx_ptr allocation = hk_pool_usc_alloc(cmd, bound, 64);
   if (!allocation.cpu)
      return 0;
   d.program_va = allocation.gpu;
   struct agx3_g17p_graphics_program program;
   if (!(dev->dev.chip == AGX_CHIP_G17P
          ? agx3_g17p_graphics_program_generate(dev->dev.chip, &d, &program)
          : agx3_g17g_mesh_program_generate(dev->dev.chip, &d, object, &program)))
      goto refused;
   memcpy(allocation.cpu, program.bytes, program.length);
   return allocation.gpu;
refused:
   mesa_loge("native mesh/object program refused: object=%u", object);
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   return 0;
}

/* Read the uncached mappings only for this opt-in validation oracle. Keep
 * its temporary snapshot and checks out of the normal recording path. */
__attribute__((cold, noinline)) static void
hk_m5_verify_fragment_root_shadow(const struct hk_graphics_state *gfx)
{
   struct hk_root_fragment_snapshot reference;
   if (gfx->root_cpu) {
      hk_root_fragment_snapshot_store(&reference, gfx->root_cpu);
      if (gfx->root_fragment_snapshot_cpu != gfx->root_cpu ||
          memcmp(&reference, &gfx->root_fragment_snapshot, sizeof(reference))) {
         mesa_loge("M5 current root snapshot mismatch");
         abort();
      }
   }
   if (gfx->fs_root_cpu) {
      hk_root_fragment_snapshot_store(&reference, gfx->fs_root_cpu);
      if (gfx->fs_root_fragment_snapshot_cpu != gfx->fs_root_cpu ||
          memcmp(&reference, &gfx->fs_root_fragment_snapshot,
                 sizeof(reference))) {
         mesa_loge("M5 prior root snapshot mismatch");
         abort();
      }
   }
}

/* The root the M5 fragment program of s follows. A root republished only
 * for bytes the fragment shader cannot read -- a per-draw dynamic offset of a
 * vertex-stage uniform buffer, a vertex buffer bind -- keeps the program the
 * stage last published, and with it the whole PPP state: that root stays
 * valid for the command buffer and holds the same bytes wherever the shader
 * reads. */
static uint64_t
hk_m5_fragment_root(struct hk_cmd_buffer *cmd, const struct hk_shader *s)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const bool shadow = debug_get_option_m5_fragment_root_shadow();

   if (unlikely(shadow && debug_get_option_m5_verify_fragment_root_shadow()))
      hk_m5_verify_fragment_root_shadow(gfx);

   return hk_m5_fragment_root_select(gfx, s->info.agx3_root_reads,
                                     debug_get_option_m5_fragment_root_memo(),
                                     shadow);
}

static uint64_t
hk_m5_upload_graphics_program(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                              struct hk_linked_shader *linked)
{
   const unsigned stage = s->info.stage == MESA_SHADER_FRAGMENT ? 1 : 0;
   struct hk_g17p_graphics_ioto_cache_entry *cached =
      &cmd->state.gfx.g17p_ioto[stage];
   const uint64_t root = stage && !s->info.agx3_gen1_fragment_rq
                            ? hk_m5_fragment_root(cmd, s)
                            : cmd->state.gfx.root;

   /* As on HAL200: a launch recorded for the same program and draw state
    * already passed admission, and replays the ranges the collector below
    * would build, so the same loader publishes the same program. */
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_shader *hw_vs =
      gfx->shaders[MESA_SHADER_VERTEX] ? hk_bound_hw_vs(gfx) : NULL;
   const uint8_t flags = hk_g17p_launch_flags(cmd, s);
   struct hk_g17p_launch *launch = hk_g17p_launch_slot(cmd, stage, linked);
   if (!s->info.agx3_gen1_fragment_rq &&
       hk_g17p_launch_matches(cmd, launch, s, linked, flags, hw_vs)) {
      const uint64_t published =
         hk_g17p_launch_replay(cmd, launch, linked, cached, root);
      if (published)
         return published;
      if (vk_command_buffer_has_error(&cmd->vk))
         return 0;
   }

   if (!hk_g17p_basic_graphics_program_supported(cmd, s, linked))
      goto refused;

   struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   uint8_t sources[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   size_t count = 0;
   if (!hk_g17p_graphics_uniform_ranges(cmd, s, linked, root, ranges, &count,
                                        sources))
      goto refused;

   /* Linked shader-owned loader templates use the same range shape and
    * relocation plan as HAL200. Each recording publishes immutable bytes
    * with its own addresses; executable lifetime follows the linked shader. */
   const uint64_t program =
      hk_g17p_publish_gfx_loader(cmd, s, linked, cached, ranges, count);
   if (!program && !vk_command_buffer_has_error(&cmd->vk))
      goto refused;
   if (program)
      hk_g17p_launch_record(cmd, launch, s, linked, cached->loader, flags,
                            hw_vs, root, ranges, count, sources);
   return program;

refused:
   mesa_loge_once("M5 graphics launch refused: stage=%u", (unsigned)s->info.stage);
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   return 0;
}

static uint64_t
hk_upload_usc_words_internal(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                             struct hk_linked_shader *linked);

uint64_t
hk_m5_multi_draw_vertex_program(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                                struct hk_linked_shader *linked)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_g17p_graphics_ioto_cache_entry *cached = &gfx->g17p_ioto[0];
   const struct hk_g17p_launch *launch = hk_g17p_launch_slot(cmd, 0, linked);

   /* No command was recorded between the records of one multi-draw, so
    * every range the previous record's launch replay derived from the root,
    * the descriptor sets and their buffers (hk_g17p_launch_replay()) is
    * unchanged; only the draw-parameter and draw-index ranges follow the
    * record. The previous record left that launch's loader and ranges as the
    * stage's last publication. Anything else takes the general path. */
   if (s->info.stage == MESA_SHADER_VERTEX && !s->info.agx3_gen1_fragment_rq &&
       launch->valid && launch->s == s && launch->linked == linked &&
       launch->loader && cached->loader == launch->loader &&
       cached->loader_linked == linked && cached->loader_program &&
       launch->flags == hk_g17p_launch_flags(cmd, s)) {
      struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
      const unsigned count = launch->count;
      memcpy(ranges, cached->loader_ranges, count * sizeof(ranges[0]));
      for (unsigned i = 0; i < count; ++i) {
         if (launch->source[i] == HK_G17P_RANGE_SOURCE_DRAW_PARAMS)
            ranges[i].address = gfx->draw_params;
         else if (launch->source[i] == HK_G17P_RANGE_SOURCE_DRAW_ID)
            ranges[i].address = gfx->draw_id_ptr;
      }

      if (hk_g17p_patch_plan_fits(&launch->loader->plan, ranges)) {
         return hk_g17p_gfx_loader_publish(cmd, linked, cached, launch->loader,
                                           ranges, count);
      }
   }

   return hk_upload_usc_words_internal(cmd, s, linked);
}

static uint64_t
hk_upload_usc_words_internal(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                             struct hk_linked_shader *linked)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (agx_chip_g17_m5(dev->dev.chip))
      return hk_m5_upload_graphics_program(cmd, s, linked);

   if (agx_native_graphics(&dev->dev)) {
      const unsigned stage = s->info.stage == MESA_SHADER_FRAGMENT ? 1 : 0;
      struct hk_g17p_graphics_ioto_cache_entry *cached =
         &cmd->state.gfx.g17p_ioto[stage];

      /* The admission below depends only on the shader and its linked
       * program for a vertex shader, plus the tile-buffer spill state, the
       * native RT0 layout and the varying-coefficient inputs (topology,
       * provoking vertex, hardware vertex shader) for a fragment shader.  An
       * earlier compact publication in this recording with the same inputs
       * was already admitted. */
      struct hk_graphics_state *agfx = &cmd->state.gfx;
      const struct hk_shader *hw_vs =
         agfx->shaders[MESA_SHADER_VERTEX] ? hk_bound_hw_vs(agfx) : NULL;
      const uint8_t flags = hk_g17p_launch_flags(cmd, s);
      struct hk_g17p_launch *launch = hk_g17p_launch_slot(cmd, stage, linked);
      if (!s->info.agx3_gen1_fragment_rq &&
          hk_g17p_launch_matches(cmd, launch, s, linked, flags, hw_vs)) {
         const uint64_t published =
            hk_g17p_launch_replay(cmd, launch, linked, cached, agfx->root);
         if (published)
            return published;
         if (vk_command_buffer_has_error(&cmd->vk))
            return 0;
      }

      /* The admission below depends only on the shader and its linked
       * program for a vertex shader, plus the tile-buffer spill state, the
       * native RT0 layout and the varying-coefficient inputs (topology,
       * provoking vertex, hardware vertex shader) for a fragment shader.  An
       * earlier compact publication in this recording with the same inputs
       * was already admitted. */
      const bool spills = agx_tilebuffer_spills(&agfx->render.tilebuffer);
      const bool same_program = agx_native_graphics(&dev->dev) &&
                                cached->admit.s == s &&
                                cached->admit.linked == linked;
      const bool vertex_admitted =
         same_program && s->info.stage == MESA_SHADER_VERTEX;
      const bool fragment_admitted =
         same_program && s->info.stage == MESA_SHADER_FRAGMENT &&
         agfx->render.g17p_rt0_layout_supported &&
         cached->admit.spills == spills &&
         cached->admit.topology == agfx->topology &&
         cached->admit.provoking == agfx->provoking && hw_vs &&
         cached->admit.hw_vs == hw_vs;
      if (!vertex_admitted && !fragment_admitted) {
         if (!agx_native_graphics(&dev->dev) ||
             !hk_g17p_basic_graphics_program_supported(cmd, s, linked)) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return 0;
         }
         cached->admit.s = s;
         cached->admit.linked = linked;
         cached->admit.spills = spills;
         cached->admit.topology = agfx->topology;
         cached->admit.provoking = agfx->provoking;
         cached->admit.hw_vs = hw_vs;
      }

      struct agx3_ioto_uniform_range
         loader_ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
      uint8_t sources[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
      size_t loader_range_count = 0;
      if (hk_g17p_graphics_uniform_ranges(cmd, s, linked, agfx->root,
                                          loader_ranges, &loader_range_count,
                                          sources)) {
         const uint64_t published = hk_g17p_publish_gfx_loader(
            cmd, s, linked, cached, loader_ranges, loader_range_count);
         if (published) {
            hk_g17p_launch_record(cmd, launch, s, linked, cached->loader,
                                  flags, hw_vs, agfx->root, loader_ranges,
                                  loader_range_count, sources);
            return published;
         }
      }

      /* The loader path above covers every admitted program; a refused
       * range layout or loader leaves nothing to launch. */
      if (!vk_command_buffer_has_error(&cmd->vk)) {
         mesa_loge_once("G17P graphics stage program refused: stage=%u "
                        "sets=%u ranges",
                        (unsigned)s->info.stage, s->info.set_count);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      }
      return 0;
   }

   if (agx_chip_command_stream_generation(dev->dev.chip) !=
       AGX_CS_GENERATION_AGX2) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return 0;
   }

   mesa_shader_stage sw_stage = s->info.stage;

   unsigned constant_push_ranges = DIV_ROUND_UP(s->b.info.rodata.size_16, 64);
   unsigned push_ranges = 2;
   unsigned stage_ranges = 3;

   size_t usc_size =
      agx_usc_size(constant_push_ranges + push_ranges + stage_ranges + 4);
   struct agx_ptr t = hk_pool_usc_alloc(cmd, usc_size, 64);
   if (!t.cpu)
      return 0;

   struct agx_usc_builder b = agx_usc_builder(t.cpu, usc_size);

   uint64_t root_ptr;

   if (sw_stage == MESA_SHADER_COMPUTE) {
      root_ptr = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
   } else {
      root_ptr = cmd->state.gfx.root;
   }

   static_assert(offsetof(struct hk_root_descriptor_table, root_desc_addr) == 0,
                 "self-reflective");

   unsigned root_unif = 0;
   if (sw_stage == MESA_SHADER_VERTEX) {
      unsigned count =
         DIV_ROUND_UP(BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4);

      if (count) {
         /* One USC uniform record covers at most 64 halves: 16 attribute
          * bases or 32 clamps. Beyond 16 attributes, split like
          * agx_usc_immediates() does for rodata. */
         hk_usc_uniform_split(
            &b, AGX_ABI_VUNI_VBO_BASE(0), 4 * count,
            root_ptr + hk_root_descriptor_offset(draw.attrib_base));

         hk_usc_uniform_split(
            &b, AGX_ABI_VUNI_VBO_CLAMP(count, 0), 2 * count,
            root_ptr + hk_root_descriptor_offset(draw.attrib_clamps));
      }

      if (cmd->state.gfx.draw_params) {
         agx_usc_uniform(&b, AGX_ABI_VUNI_FIRST_VERTEX(count), 4,
                         cmd->state.gfx.draw_params);
      }

      if (cmd->state.gfx.draw_id_ptr) {
         agx_usc_uniform(&b, AGX_ABI_VUNI_DRAW_ID(count), 1,
                         cmd->state.gfx.draw_id_ptr);
      }

      if (linked->sw_indexing) {
         agx_usc_uniform(
            &b, AGX_ABI_VUNI_VERTEX_PARAMS(count), 4,
            root_ptr + hk_root_descriptor_offset(draw.vertex_params));
      }

      root_unif = AGX_ABI_VUNI_COUNT_VK(count);
   } else if (sw_stage == MESA_SHADER_FRAGMENT) {
      if (agx_tilebuffer_spills(&cmd->state.gfx.render.tilebuffer)) {
         hk_usc_upload_spilled_rt_descs(&b, cmd);
      }

      if (hk_cmd_uses_blend_constant(cmd)) {
         agx_usc_uniform(
            &b, 4, 8,
            root_ptr + hk_root_descriptor_offset(draw.blend_constant));
      }

      /* The SHARED state is baked into linked->usc for non-fragment shaders. We
       * don't pass around the information to bake the tilebuffer layout.
       *
       * TODO: We probably could with some refactor.
       */
      agx_usc_push_packed(&b, SHARED, &cmd->state.gfx.render.tilebuffer.usc);
      root_unif = AGX_ABI_FUNI_ROOT;
   }

   /* Address for the root and each set */
   agx_usc_uniform(&b, root_unif, 4 * (1 + s->info.set_count), root_ptr);

   agx_usc_push_blob(&b, linked->usc.data, linked->usc.size);
   return agx_usc_addr(&dev->dev, t.gpu);
}

uint64_t
hk_upload_usc_words(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                    struct hk_linked_shader *linked)
{
   /* An inline-blend program with its own constant program launches with
    * its own uniform layout (hk_linked_shader::g17p_layout). */
   if (linked && linked->g17p_layout)
      s = linked->g17p_layout;
   return hk_upload_usc_words_internal(cmd, s, linked);
}

void
hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                    enum agx_barrier barrier, enum libagx_program idx,
                    void *data, size_t data_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (agx_native_graphics(&dev->dev)) {
      hk_dispatch_precomp_g17p(cmd, grid, barrier, idx, data, data_size);
      return;
   }

   if (agx_chip_g17_m5(dev->dev.chip)) {
      hk_dispatch_precomp_m5(cmd, grid, barrier, idx, data, data_size);
      return;
   }

   struct agx_precompiled_shader *prog = agx_get_precompiled(&dev->bg_eot, idx);
   if (!prog) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   struct hk_cs **target = (barrier & AGX_POSTGFX)  ? &cmd->current_cs.post_gfx
                           : (barrier & AGX_PREGFX) ? &cmd->current_cs.pre_gfx
                                                    : &cmd->current_cs.cs;

   struct hk_cs *cs = hk_cmd_buffer_get_cs_general(cmd, target, true);
   if (!cs)
      return;

   struct agx_ptr t = hk_pool_usc_alloc(cmd, agx_usc_size(15), 64);
   uint64_t uploaded_data = hk_pool_upload(cmd, data, data_size, 4);

   agx_usc_words_precomp(t.cpu, &prog->b, uploaded_data, data_size);

   hk_dispatch_with_usc_launch(dev, cs, prog->b.launch,
                               agx_usc_addr(&dev->dev, t.gpu), grid,
                               prog->b.workgroup);
}

uint64_t
hk_apply_conditional_rendering(struct hk_cmd_buffer *cmd, uint64_t source,
                               uint32_t words, enum agx_barrier barrier)
{
   const struct hk_conditional_render_state *conditional =
      &cmd->state.conditional;

   if (!hk_conditional_render_is_active(conditional, cmd->in_meta,
                                         cmd->conditional_meta))
      return source;

   if (!source || !conditional->address || words == 0 ||
       words > HK_CONDITIONAL_RENDER_MAX_WORDS) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return 0;
   }

   const struct agx_ptr output =
      hk_pool_alloc(cmd, words * sizeof(uint32_t), 4);
   if (!output.gpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return 0;
   }

   libagx_conditional_render(cmd, agx_1d(1), barrier, output.gpu, source,
                             conditional->address, words,
                             conditional->inverted);
   return vk_command_buffer_has_error(&cmd->vk) ? 0 : output.gpu;
}

static void
hk_cs_capture_fragment_attachments(const struct hk_device *dev,
                                   const struct hk_rendering_state *render,
                                   struct hk_cs *cs)
{
   /* Retain the image allocation spans now. A command buffer may contain
    * several VDM commands with different render state, while queue submission
    * happens only after all of them have been recorded. */
   struct hk_attachment_view_ref {
      struct hk_image_view *view;
      VkImageAspectFlags aspect;
   } views[(2 * HK_MAX_RTS) + 4];
   static_assert(ARRAY_SIZE(views) == HK_DRM_MAX_FRAGMENT_ATTACHMENTS,
                 "fragment attachment capture capacity");
   unsigned view_count = 0;

   /* Each retained span is a G17P fragment write range, and the kernel
    * encodes at most sixteen of them next to its own. Only a direct colour
    * resolve writes a resolve attachment during the pass; every other resolve
    * is a later meta pass that retains its own attachments. Eight multisample
    * targets and their resolve targets would otherwise overflow the table and
    * fail the submit. */
   const bool g17p = agx_native_graphics(&dev->dev);
   const bool resolves_written = !g17p || render->g17p_direct_color_resolve;

   for (unsigned i = 0; i < render->color_att_count; ++i) {
      views[view_count++] = (struct hk_attachment_view_ref){
         .view = render->color_att[i].iview,
         .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      };
      views[view_count++] = (struct hk_attachment_view_ref){
         .view = resolves_written ? render->color_att[i].resolve_iview : NULL,
         .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      };
   }

   views[view_count++] = (struct hk_attachment_view_ref){
      .view = render->depth_att.iview,
      .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
   };
   views[view_count++] = (struct hk_attachment_view_ref){
      .view = g17p ? NULL : render->depth_att.resolve_iview,
      .aspect = VK_IMAGE_ASPECT_DEPTH_BIT,
   };
   views[view_count++] = (struct hk_attachment_view_ref){
      .view = render->stencil_att.iview,
      .aspect = VK_IMAGE_ASPECT_STENCIL_BIT,
   };
   views[view_count++] = (struct hk_attachment_view_ref){
      .view = g17p ? NULL : render->stencil_att.resolve_iview,
      .aspect = VK_IMAGE_ASPECT_STENCIL_BIT,
   };

   assert(view_count <= ARRAY_SIZE(views));
   for (unsigned i = 0; i < view_count; ++i) {
      struct hk_image_view *view = views[i].view;
      if (!view)
         continue;

      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);
      const uint8_t image_plane =
         views[i].aspect == VK_IMAGE_ASPECT_COLOR_BIT
            ? view->planes[0].image_plane
            : hk_image_aspects_to_plane(image, views[i].aspect);
      if (image_plane >= image->plane_count) {
         cs->fragment_attachments.complete = false;
         continue;
      }

      const struct hk_image_plane *plane = &image->planes[image_plane];
      const uint64_t size = plane->layout.size_B;
      if (!plane->addr || !size) {
         cs->fragment_attachments.complete = false;
         continue;
      }

      hk_drm_attachment_set_add(&cs->fragment_attachments, plane->addr, size);
   }
}

bool
hk_cs_init_graphics(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   uint8_t *map = cs->current;

   /* Even a clear-only stream needs a stable root alias before EndCommandBuffer.
    * Linked chunks acquire theirs in hk_cmd_buffer_emit_cs_jump. */
   if (hk_g17p_vdm_low_window_enabled(&dev->dev) &&
       !hk_g17p_vdm_compact(cmd, cs->addr))
      return false;

   hk_drm_attachment_set_init(&cs->fragment_attachments);

   /* Only the native AGX3 path submits the spans (hk_queue.c). */
   if (agx_native_graphics(&dev->dev))
      hk_cs_capture_fragment_attachments(dev, render, cs);

   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
      /* The kext preserves a zero final VDM word when this selector is the
       * UINT32_MAX sentinel.  This reproduces m1n1's direct-triangle prefix. */
      cs->agx3_active_context_selector_valid = true;
      cs->agx3_raw_active_context_selector = UINT32_MAX;
   }

   cs->tib = render->tilebuffer;
   /* Attachmentless rendering defers its API sample count. Publish the
    * native one-sample no-imageblock profile in this control stream while
    * preserving the deferred count in render state. */
   if (agx_chip_g17_m5(dev->dev.chip) && !cs->tib.sample_size_B &&
       !cs->tib.nr_samples && !render->color_att_count &&
       !render->depth_att.iview && !render->stencil_att.iview)
      cs->tib.nr_samples = 1;
   if (agx_chip_g17_m5(dev->dev.chip)) {
      const struct hk_bg_eot *programs[] = {
         &render->cr.bg.main, &render->cr.bg.partial,
         &render->cr.eot.main, &render->cr.eot.partial,
      };
      cs->tile_resources = (struct drm_asahi_render_program){
         .stage = 1,
         .imageblock_bytes = cs->tib.tile_size * cs->tib.sample_size_B *
                             cs->tib.nr_samples,
         .multisample_bytes = cs->tib.nr_samples == 2
                                ? cs->tib.tile_size * cs->tib.sample_size_B / 4
                                : cs->tib.nr_samples == 4
                                     ? cs->tib.tile_size * cs->tib.sample_size_B
                                     : cs->tib.nr_samples == 8
                                          ? 4 * cs->tib.tile_size * cs->tib.sample_size_B
                                          : 0,
      };
      for (unsigned i = 0; i < ARRAY_SIZE(programs); i++) {
         cs->tile_resources.primary_bytes = MAX2(cs->tile_resources.primary_bytes,
                                                  programs[i]->primary_storage_B);
         cs->tile_resources.secondary_bytes = MAX2(cs->tile_resources.secondary_bytes,
                                                    programs[i]->staging_storage_B);
      }
   }

   /* Assume this is not the first control stream of the render pass, so
    * initially use the partial background/EOT program and ZLS control.
    * hk_BeginRendering/hk_EndRendering will override.
    */
   cs->cr = render->cr;
   cs->cr.bg.main = render->cr.bg.partial;
   cs->cr.eot.main = render->cr.eot.partial;
   cs->cr.zls_control = render->cr.zls_control_partial;

   /* Barrier to enforce GPU-CPU coherency, in case this batch is back to back
    * with another that caused stale data to be cached and the CPU wrote to it
    * in the meantime.
    */
   switch (agx_chip_command_stream_generation(dev->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      agx_push(map, VDM_BARRIER, cfg) {
         cfg.usc_cache_inval = true;
      }
      break;

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      struct agx_vdm_barrier_core barrier;
      if (!agx_vdm_barrier_core_pack(dev->dev.chip, &barrier)) {
         mesa_loge("no grounded AGX3 initial VDM barrier for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }

      memcpy(map, barrier.bytes, barrier.length);
      map += barrier.length;
      break;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      mesa_loge("unknown VDM command-stream generation for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   if (agx_chip_command_stream_generation(dev->dev.chip) ==
       AGX_CS_GENERATION_HAL300) {
      struct agx_ppp_w_clamp_cull2_core initial;
      const struct agx_ppp_w_clamp_cull2_fields fields = {
         /* Exact G17G/G17X default: W clamp 0.0, clamp-W enabled. */
         .w_clamp_raw = 0,
         .cull2_raw = UINT32_C(0x80),
      };
      struct agx_ppp_hal300_depth_stencil_core depth_stencil;
      const struct agx_ppp_hal300_depth_stencil_fields depth_stencil_fields = {
         /* Exact G17G/G17X fixed-layout initial render-state witness. */
         .fragment_control_raw = UINT32_C(0x000007aa),
         .fragment_front_face_raw = UINT32_C(0xe2e00000),
         .fragment_back_face_raw = UINT32_C(0xf0c00000),
         .fragment_front_stencil_raw = UINT32_C(0x00023020),
         .fragment_back_stencil_raw = UINT32_C(0x0001ffff),
      };
      if (!agx_ppp_hal300_depth_stencil_core_pack(
             dev->dev.chip, &depth_stencil_fields, &depth_stencil) ||
          !agx_ppp_w_clamp_cull2_core_pack(dev->dev.chip, &fields,
                                            &initial)) {
         mesa_loge("no grounded HAL300 initial PPP payload for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }

      const uint64_t depth_stencil_gpu = hk_pool_upload(
         cmd, depth_stencil.bytes, depth_stencil.length,
         AGX_PPP_HEADER_ALIGN);
      const uint64_t gpu = hk_pool_upload(
         cmd, initial.bytes, initial.length, AGX_PPP_HEADER_ALIGN);
      uint8_t records[2 * AGX_PPP_STATE_LENGTH];
      uint8_t *record = records;
      if (!depth_stencil_gpu || !gpu ||
          !hk_ppp_emit_state(dev->dev.chip, &record, depth_stencil_gpu,
                              depth_stencil.length) ||
          !hk_ppp_emit_state(dev->dev.chip, &record, gpu, initial.length) ||
          record != records + sizeof(records)) {
         mesa_loge("failed to emit HAL300 initial PPP state for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return false;
      }

      memcpy(map, records, sizeof(records));
      map += sizeof(records);
   } else if (agx_native_graphics(&dev->dev)) {
      /* PPPEncoderGen6's inspected G17P fixed-layout opening is one exact
       * W-clamp/Cull-2 payload, not the legacy AGX2 selector graph below.
       * Keep the payload in the compact VDM window just like every other
       * nested HAL200 PPP pointer.
       */
      struct agx_ppp_w_clamp_cull2_core initial;
      const struct agx_ppp_w_clamp_cull2_fields fields = {
         .w_clamp_raw = 0,
         .cull2_raw = UINT32_C(0x80),
      };

      if (!agx_ppp_w_clamp_cull2_core_pack(dev->dev.chip, &fields,
                                            &initial)) {
         mesa_loge("no grounded G17P initial PPP payload for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }

      struct agx_ptr T =
         hk_pool_vdm_alloc(cmd, initial.length, AGX_PPP_HEADER_ALIGN);
      if (!T.cpu) {
         /* Returning without an error left the control stream stopped after
          * the VDM barrier, short of HK_G17P_VDM_INITIAL_SIZE, with the
          * command buffer still reporting VK_SUCCESS.  Every following draw
          * then appended to a half-built prelude, and the size assert below
          * aborts a debug build instead of reporting anything. */
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return false;
      }

      memcpy(T.cpu, initial.bytes, initial.length);
      if (!hk_ppp_emit_state(dev->dev.chip, &map, T.gpu, initial.length)) {
         mesa_loge("failed to emit G17P initial PPP state for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return false;
      }
   } else {
      struct AGX_PPP_HEADER present = {
         .w_clamp = true,
         .occlusion_query_2 = true,
         .output_unknown = true,
         .varying_word_2 = true,
         .viewport_count = 1, /* irrelevant */
      };

      size_t size = agx_ppp_update_size(&present);
      struct agx_ptr T = hk_pool_vdm_alloc(cmd, size, AGX_PPP_HEADER_ALIGN);
      if (!T.cpu)
         return false;

      struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &present);

      /* clang-format off */
      agx_ppp_push(&ppp, W_CLAMP, cfg) cfg.w_clamp = 1e-10;
      agx_ppp_push(&ppp, FRAGMENT_OCCLUSION_QUERY_2, cfg);
      agx_ppp_push(&ppp, OUTPUT_UNKNOWN, cfg);
      agx_ppp_push(&ppp, VARYING_2, cfg);
      /* clang-format on */

      const bool ppp_ok = agx_ppp_legacy_payload_supported(dev->dev.chip) &&
                          agx_ppp_fini(dev->dev.chip, &map, &ppp);
      if (!ppp_ok) {
         mesa_loge("no grounded legacy PPP payload/finalizer for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }
   }
   cs->current = map;

   cs->scissor = UTIL_DYNARRAY_INIT;
   cs->depth_bias = UTIL_DYNARRAY_INIT;
   if (!util_dynarray_ensure_cap(&cs->scissor,
                                 HK_MAX_VIEWPORTS * AGX_SCISSOR_LENGTH) ||
       !util_dynarray_ensure_cap(&cs->depth_bias,
                                 HK_MAX_VIEWPORTS * AGX_DEPTH_BIAS_LENGTH)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return false;
   }

   /* All graphics state must be reemited in each control stream */
   hk_cmd_buffer_dirty_all(cmd);

   if (agx_native_graphics(&dev->dev))
      assert((cs->current - cs->start) == HK_G17P_VDM_INITIAL_SIZE);

   return true;
}

__attribute__((hot)) void
hk_ensure_cs_has_space(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       size_t space)
{
   bool vdm = cs->type == HK_CS_VDM;

   size_t link_length =
      vdm ? AGX_VDM_STREAM_LINK_LENGTH : AGX_CDM_STREAM_LINK_LENGTH;

   /* Assert that we have space for a link tag */
   assert((cs->current + link_length) <= cs->end && "Encoder overflowed");

   /* Always leave room for a link tag, in case we run out of space later,
    * plus padding because VDM apparently overreads?
    *
    * 0x200 is not enough. 0x400 seems to work. 0x800 for safety.
    */
   /* If there is room in the command buffer, we're done */
   if (likely(hk_cs_space_fits(cs->end - cs->current, space, link_length)))
      return;

   /* Otherwise, we need to allocate a new command buffer. We use memory owned
    * by the batch to simplify lifetime management for the BO.
    */
   size_t size = 65536;
   struct agx_ptr T = hk_pool_alloc(cmd, size, 256);
   if (!T.cpu)
      return;

   /* Jump from the old control stream to the new control stream. */
   void *link_end =
      hk_cmd_buffer_emit_cs_jump(cmd, cs->current, T.gpu, vdm);
   const bool linked = link_end != cs->current;
   if (!linked)
      return;

   /* Swap out the control stream */
   cs->current = T.cpu;
   cs->end = cs->current + size;
   cs->chunk = T;
   cs->stream_linked |= linked;
}

static void
clear_attachment_as_image(struct hk_cmd_buffer *cmd,
                          struct hk_rendering_state *render,
                          struct hk_attachment *att, unsigned aspect)
{
   struct hk_image_view *view = att->iview;
   if (!att->clear || !view)
      return;

   /* A depth/stencil attachment uses every aspect of its image whatever the
    * view's aspect mask says (the mask is ignored for render pass depth/stencil
    * attachments), so a depth-only view bound as the stencil attachment must
    * still have its stencil cleared. Its view format then names only the depth
    * aspect, so take the format from the image. Colour keeps the view format,
    * in which the clear value is expressed.
    */
   const bool zs = aspect & (VK_IMAGE_ASPECT_DEPTH_BIT |
                             VK_IMAGE_ASPECT_STENCIL_BIT);
   const VkImageAspectFlags aspects = zs ? view->vk.image->aspects
                                         : view->vk.aspects;
   if (!(aspects & aspect))
      return;

   const uint32_t layer_count = render->view_mask
                                   ? util_last_bit(render->view_mask)
                                   : render->layer_count;

   const VkImageSubresourceRange range = {
      .layerCount = layer_count,
      .levelCount = 1,
      .baseArrayLayer = view->vk.base_array_layer,
      .baseMipLevel = view->vk.base_mip_level,
      .aspectMask = aspects & aspect,
   };

   assert(util_bitcount(range.aspectMask) == 1);
   VkFormat format = zs ? view->vk.image->format : att->vk_format;

   if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
      format = vk_format_depth_only(format);
   } else if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) {
      format = vk_format_stencil_only(format);
   }

   struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   hk_clear_image(cmd, image, vk_format_to_pipe_format(format),
                  att->clear_colour, &range, false /* partially clear 3D */);
}

bool
hk_optimize_empty_vdm(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.gfx;
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* An empty final stream of a pass with earlier draws is not an empty
    * render. Replacing it with the load-op clears would erase those draws. */
   const bool continuation = cmd->sync.pass_vdm != cs;
   if (continuation && !agx_native_graphics(&dev->dev) &&
       !agx_chip_g17_m5(dev->dev.chip))
      return false;

   /* Two reasons to keep the graphics representation on AGX3 rather than
    * replace it with the image clears below.
    *
    * The first is capability: G17P has native BG/EOT programs for the
    * supported render-target clear path, but had no precompiled fill_uint4
    * for the compute replacement. That one retires itself once the shader is
    * available.
    *
    * The second is correctness, and it does not retire. A direct colour
    * resolve is performed by this render's OWN end-of-tile program -- it is
    * selected precisely when storeOp is not STORE, so the multisample
    * attachment is never stored and the pass's only output is the resolve
    * target. Clearing the attachment as an image then writes a surface
    * nobody reads and leaves the resolve target untouched, which is exactly
    * why a clear-only multisample pass resolved to 00 00 00 00 while a pass
    * with any draw in it was correct. Whether fill_uint4 happens to be
    * available cannot be allowed to decide that.
    *
    * BeginRendering retained the attachments, initialized TA/PPP and selected
    * the BG clear; EndRendering selected the final EOT. process_empty_tiles
    * is the existing clear-and-store predicate, not a fabricated draw. The
    * common finalizer will append the native VDM terminator without uploading
    * any ordinary vertex or fragment program.
    */
   const bool native_eot = agx_native_graphics(&dev->dev) ||
                           agx_chip_g17_m5(dev->dev.chip);
   const bool output_only_through_eot =
      native_eot && render->g17p_direct_color_resolve;

   /* A timestamp rides on this exact command; replacing the render with
    * image clears would drop it. Keep the render. */
   if (cs->timestamp.start.handle || cs->timestamp.end.handle)
      return false;

   /* M5: a multisample clear-only pass stays a render. As compute it is a
    * separate job that every later render's vertex half must wait for; as a
    * render that writes only its attachments it is not. */
   const bool m5_ms_clear = agx_chip_g17_m5(dev->dev.chip) &&
                            cs->cr.process_empty_tiles && cs->tib.nr_samples > 1;
   if (native_eot &&
       (continuation ||
        (cs->cr.process_empty_tiles &&
         (output_only_through_eot || m5_ms_clear ||
          !agx_get_precompiled(&dev->bg_eot, LIBAGX_FILL_UINT4))))) {
      /* Normally supplied by the first draw. A clear has no application
       * sample-location state; use the attachment sample count and leave
       * has_sample_locations uncommitted, just like other meta operations.
       */
      cs->ppp_multisamplectl =
         hk_default_sample_positions(MAX2(cs->tib.nr_samples, 1));

      /* The render submission still binds these arrays even without PPP draw
       * state referencing them. Give it an owned, neutral full-frame entry
       * rather than a zero-byte upload. Render-area handling and empty-tile
       * selection remain in the existing render setup.
       */
      if (cs->scissor.size == 0) {
         uint8_t *scissor =
            util_dynarray_grow_bytes(&cs->scissor, 1, AGX_SCISSOR_LENGTH);
         agx_pack(scissor, SCISSOR, cfg) {
            cfg.min_x = cfg.min_y = 0;
            cfg.max_x = cs->cr.width;
            cfg.max_y = cs->cr.height;
            cfg.min_z = 0.0f;
            cfg.max_z = 1.0f;
         }
      }

      if (cs->depth_bias.size == 0) {
         void *bias = util_dynarray_grow_bytes(&cs->depth_bias, 1,
                                              AGX_DEPTH_BIAS_LENGTH);
         memset(bias, 0, AGX_DEPTH_BIAS_LENGTH);
      }

      return false;
   }

   /* The clears stand in for the render: they are graphics work for every
    * later dependency, and they were recorded under the same requirements
    * the render was created under. */
   /* Give the replacement an owned stream: extending an older compute job
    * could place its clear before a render it must wait for. */
   hk_cmd_buffer_end_compute(cmd);
   cmd->sync.kind_override = HK_SYNC_GFX;

   for (unsigned i = 0; i < render->color_att_count; ++i) {
      clear_attachment_as_image(cmd, render, &render->color_att[i], ~0);
   }

   clear_attachment_as_image(cmd, render, &render->depth_att,
                             VK_IMAGE_ASPECT_DEPTH_BIT);

   clear_attachment_as_image(cmd, render, &render->stencil_att,
                             VK_IMAGE_ASPECT_STENCIL_BIT);

   cmd->sync.kind_override = 0;
   if (cmd->current_cs.cs) {
      hk_sync_replace_render(cmd, cs, cmd->current_cs.cs);
      hk_cmd_buffer_end_compute(cmd);
   }

   /* Remove the VDM control stream from the command buffer, now that it is
    * replaced by equivalent other operations.
    */
   hk_sync_forget_stream(cmd, cs);
   list_del(&cs->node);
   hk_cs_destroy(cs);
   return true;
}

/*
 * Deferred whole-image clears (native AGX3).
 *
 * A vkCmdClearColorImage or vkCmdClearDepthStencilImage of a whole
 * single-level, single-layer attachment image is a compute job: a fill of
 * the entire allocation for a plain image, a metadata fast clear for a
 * compressed one. Applications commonly clear an image this way and
 * then begin a full-area render pass that loads it, which reads the whole
 * image back into the tilebuffer. Minecraft clears six 2976x1872 images
 * per frame like that: 138 MB of fills, then the loads.
 *
 * Instead such a clear is remembered. When the next render pass covers the
 * whole image with it as an attachment that it loads (or does not care
 * about) and stores (or does not care about), the attachment's load
 * becomes a load-op clear of the same value: the tilebuffer starts with the
 * clear value and the pass stores the image, which is what the fill
 * followed by the load and the store produces. Anything else records the
 * fill first: any other work, through hk_cmd_buffer_get_cs_general(); a
 * render pass that does not take the clear; the end of the command buffer;
 * and the commands below that order or observe earlier work without
 * recording any. A pass that may also read the image's memory takes the
 * load-op clear and records the fill as well, before the pass
 * (hk_deferred_clear_layout_use()). The fill is then recorded after the
 * barriers that followed the clear, so it is followed by a transfer-to-all
 * barrier of its own: later work still waits for it.
 */
static bool
hk_deferred_clear_image_whole(const struct hk_image *image,
                              uint32_t range_count,
                              const VkImageSubresourceRange *ranges,
                              VkImageAspectFlags aspects)
{
   if (range_count != 1 || image->vk.image_type != VK_IMAGE_TYPE_2D ||
       image->vk.mip_levels != 1 || image->vk.array_layers != 1 ||
       image->vk.extent.depth != 1 || image->vk.samples != 1 ||
       image->vk.tiling != VK_IMAGE_TILING_OPTIMAL ||
       ranges[0].aspectMask != aspects || ranges[0].baseMipLevel != 0 ||
       vk_image_subresource_level_count(&image->vk, &ranges[0]) != 1 ||
       ranges[0].baseArrayLayer != 0 ||
       vk_image_subresource_layer_count(&image->vk, &ranges[0]) != 1)
      return false;

   return true;
}

bool
hk_defer_clear(struct hk_cmd_buffer *cmd, struct hk_image *image,
               VkImageAspectFlags aspects, const VkClearValue *value,
               uint32_t range_count, const VkImageSubresourceRange *ranges)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   const VkImageUsageFlags attachment =
      aspects == VK_IMAGE_ASPECT_COLOR_BIT
         ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
         : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

   if ((!agx_chip_native_agx3(dev->dev.chip) &&
        !agx_chip_g17_m5(dev->dev.chip)) ||
       cmd->in_meta ||
       cmd->vk.pool->queue_family_index >= pdev->queue_family_count ||
       !(pdev->queue_families[cmd->vk.pool->queue_family_index].queue_flags &
         VK_QUEUE_GRAPHICS_BIT) ||
       !(image->vk.usage & attachment) ||
       aspects != (image->vk.aspects &
                   (aspects == VK_IMAGE_ASPECT_COLOR_BIT
                       ? VK_IMAGE_ASPECT_COLOR_BIT
                       : (VK_IMAGE_ASPECT_DEPTH_BIT |
                          VK_IMAGE_ASPECT_STENCIL_BIT))) ||
       !hk_deferred_clear_image_whole(image, range_count, ranges, aspects))
      return false;

   /* The deferred clear overwrites the whole image, whether it becomes a
    * load-op clear or a fill: a pending store of it is dead. */
   if (cmd->state.pending_z_store.cs &&
       cmd->state.pending_z_store.image == image &&
       (aspects & VK_IMAGE_ASPECT_DEPTH_BIT))
      hk_kill_pending_z_store(cmd);

   /* A later clear of the same image replaces the earlier one. */
   for (unsigned i = 0; i < cmd->state.deferred_clear_count; ++i) {
      struct hk_deferred_clear *e = &cmd->state.deferred_clear[i];
      if (e->image == image) {
         e->aspects = aspects;
         e->value = *value;
         return true;
      }
   }

   if (cmd->state.deferred_clear_count == HK_MAX_DEFERRED_CLEARS)
      hk_flush_deferred_clears(cmd);

   cmd->state.deferred_clear[cmd->state.deferred_clear_count++] =
      (struct hk_deferred_clear){
         .image = image,
         .aspects = aspects,
         .value = *value,
      };
   return true;
}

/* A buffer command (copy, fill, update) cannot access an image except
 * through memory the two share. Keep the deferred clears across one whose
 * buffer ranges miss every image they clear; call it for each range of the
 * command before recording it, and clear deferred_clear_hold after. */
void
hk_deferred_clears_hold_buffer(struct hk_cmd_buffer *cmd, uint64_t addr,
                               uint64_t size)
{
   /* The same holds for the depth image of a pending store. */
   const struct hk_image *z_image = cmd->state.pending_z_store.cs
                                       ? cmd->state.pending_z_store.image
                                       : NULL;
   if (z_image) {
      for (unsigned p = 0; p < z_image->plane_count; ++p) {
         const uint64_t base = z_image->planes[p].addr;
         const uint64_t end = base + z_image->planes[p].layout.size_B;
         if (!base || (addr < end && base < addr + size)) {
            cmd->state.pending_z_store.cs = NULL;
            break;
         }
      }
   }

   if (!cmd->state.deferred_clear_count) {
      if (cmd->state.pending_z_store.cs)
         cmd->state.deferred_clear_hold = true;
      return;
   }

   for (unsigned i = 0; i < cmd->state.deferred_clear_count; ++i) {
      const struct hk_image *image = cmd->state.deferred_clear[i].image;
      for (unsigned p = 0; p < image->plane_count; ++p) {
         const uint64_t base = image->planes[p].addr;
         const uint64_t end = base + image->planes[p].layout.size_B;
         if (!base || (addr < end && base < addr + size)) {
            hk_flush_deferred_clears(cmd);
            return;
         }
      }
   }
   cmd->state.deferred_clear_hold = true;
}

void
hk_flush_deferred_clears_why(struct hk_cmd_buffer *cmd, const char *why)
{
   /* Whatever observes the deferred clears may also read a pending store. */
   cmd->state.pending_z_store.cs = NULL;
   cmd->state.deferred_clear_hold = false;
   if (!cmd->state.deferred_clear_count || cmd->state.deferred_clear_flushing)
      return;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   cmd->state.deferred_clear_flushing = true;
   for (unsigned i = 0; i < cmd->state.deferred_clear_count; ++i) {
      const struct hk_deferred_clear *e = &cmd->state.deferred_clear[i];
      if (unlikely(dev->frame_anatomy)) {
         hk_frame_anatomy_note(
            dev, "deferred-clear fill by %s: %ux%u %s aspects=0x%x compressed=%u",
            why, e->image->vk.extent.width, e->image->vk.extent.height,
            util_format_short_name(
               hk_format_to_pipe_format(e->image->vk.format)),
            e->aspects, e->image->planes[0].layout.compressed);
      }
      const VkImageSubresourceRange range = {
         .aspectMask = e->aspects,
         .levelCount = 1,
         .layerCount = 1,
      };
      uint32_t colour[4];
      if (e->aspects == VK_IMAGE_ASPECT_COLOR_BIT) {
         memcpy(colour, e->value.color.uint32, sizeof(colour));
      } else {
         colour[0] = fui(e->value.depthStencil.depth);
         colour[1] = e->value.depthStencil.stencil;
         colour[2] = colour[3] = 0;
      }

      hk_sync_xfer_begin(cmd);
      hk_clear_image(cmd, e->image, PIPE_FORMAT_NONE, colour, &range, true);
      hk_sync_xfer_end(cmd);
   }
   cmd->state.deferred_clear_count = 0;

   /* The barriers that followed the clears were recorded before the fills;
    * order everything recorded from now on after them, as
    * hk_CmdPipelineBarrier2() does outside a render pass. */
   struct hk_cs *cs = cmd->current_cs.cs;
   const bool alias_scope = hk_cdm_alias_scope(dev);
   if (cs && hk_cdm_visibility_flush_needed(dev, cs, alias_scope)) {
      hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
      hk_cdm_visibility_flush(dev, cs, alias_scope);
   }
   hk_sync_barrier(cmd, HK_SYNC_XFER, HK_SYNC_ALL);
   cmd->state.deferred_clear_flushing = false;
}

/* How a pass may use a deferred clear for an attachment of the given aspect
 * in this layout. A load-op clear lands in the tilebuffer, not in memory.
 *
 * In an attachment layout the pass can access the image only as that
 * attachment, so the load-op clear replaces the fill. In GENERAL, a
 * feedback-loop or the local-read layout the pass may also read the image's
 * memory, as a sampled or storage image, an input attachment (lowered to a
 * texture load) or through local read; it still takes the load-op clear in
 * place of the load, but the fill is recorded before the pass. Every layout
 * is GENERAL under unified image layouts, which zink uses. A read-only
 * aspect takes nothing: the clear is recorded as a fill. */
enum hk_deferred_clear_use {
   HK_DEFERRED_CLEAR_UNUSED,
   HK_DEFERRED_CLEAR_LOAD,
   HK_DEFERRED_CLEAR_LOAD_AND_FILL,
};

static enum hk_deferred_clear_use
hk_deferred_clear_layout_use(VkImageLayout layout, VkImageAspectFlags aspect)
{
   bool attachment;
   switch (layout) {
   case VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL:
      attachment = true;
      break;
   case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      attachment = aspect == VK_IMAGE_ASPECT_COLOR_BIT;
      break;
   case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      attachment = aspect != VK_IMAGE_ASPECT_COLOR_BIT;
      break;
   case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
   case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL:
      attachment = aspect == VK_IMAGE_ASPECT_DEPTH_BIT;
      break;
   case VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL:
   case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL:
      attachment = aspect == VK_IMAGE_ASPECT_STENCIL_BIT;
      break;
   case VK_IMAGE_LAYOUT_GENERAL:
   case VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT:
   case VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ_KHR:
      return HK_DEFERRED_CLEAR_LOAD_AND_FILL;
   default:
      return HK_DEFERRED_CLEAR_UNUSED;
   }
   return attachment ? HK_DEFERRED_CLEAR_LOAD : HK_DEFERRED_CLEAR_UNUSED;
}

/* The deferred clear of the image an attachment views, if the attachment can
 * take it as its load-op clear (see above); *fill is set when the clear must
 * also be recorded as a fill before the pass. */
static struct hk_deferred_clear *
hk_deferred_clear_for(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                      const VkRenderingAttachmentInfo *att,
                      VkImageAspectFlags aspect, bool *fill)
{
   if (!att || !att->imageView)
      return NULL;

   enum hk_deferred_clear_use use =
      hk_deferred_clear_layout_use(att->imageLayout, aspect);
   if (use == HK_DEFERRED_CLEAR_UNUSED)
      return NULL;

   /* An input-attachment feedback flag reads the attachment in the pass. */
   const VkRenderingAttachmentFlagsInfoKHR *flags =
      vk_find_struct_const(att->pNext, RENDERING_ATTACHMENT_FLAGS_INFO_KHR);
   if (flags && flags->flags)
      use = HK_DEFERRED_CLEAR_LOAD_AND_FILL;

   VK_FROM_HANDLE(hk_image_view, view, att->imageView);
   struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
   struct hk_deferred_clear *e = NULL;
   for (unsigned i = 0; i < cmd->state.deferred_clear_count; ++i) {
      if (cmd->state.deferred_clear[i].image == image)
         e = &cmd->state.deferred_clear[i];
   }

   if (!e || !(e->aspects & aspect) ||
       (att->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD &&
        att->loadOp != VK_ATTACHMENT_LOAD_OP_DONT_CARE) ||
       (att->storeOp != VK_ATTACHMENT_STORE_OP_STORE &&
        att->storeOp != VK_ATTACHMENT_STORE_OP_DONT_CARE) ||
       view->vk.view_type != VK_IMAGE_VIEW_TYPE_2D ||
       view->vk.format != image->vk.format ||
       view->vk.base_mip_level != 0 || view->vk.level_count != 1 ||
       view->vk.base_array_layer != 0 || view->vk.layer_count != 1 ||
       info->renderArea.offset.x != 0 || info->renderArea.offset.y != 0 ||
       info->renderArea.extent.width != image->vk.extent.width ||
       info->renderArea.extent.height != image->vk.extent.height)
      return NULL;

   *fill = use == HK_DEFERRED_CLEAR_LOAD_AND_FILL;
   if (unlikely(hk_cmd_buffer_device(cmd)->frame_anatomy)) {
      hk_frame_anatomy_note(hk_cmd_buffer_device(cmd),
                            "deferred-clear taken: %ux%u layout=%u fill=%u",
                            image->vk.extent.width, image->vk.extent.height,
                            att->imageLayout, *fill);
   }
   return e;
}

const VkRenderingInfo *
hk_take_deferred_clears(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                        VkRenderingInfo *copy,
                        VkRenderingAttachmentInfo *color,
                        VkRenderingAttachmentInfo *depth,
                        VkRenderingAttachmentInfo *stencil)
{
   if (cmd->in_meta || (info->flags & VK_RENDERING_RESUMING_BIT) ||
       info->viewMask || info->layerCount != 1 ||
       info->colorAttachmentCount > HK_MAX_RTS)
      return info;

   struct hk_deferred_clear *col[HK_MAX_RTS] = {NULL};
   bool col_fill[HK_MAX_RTS] = {false}, z_fill = false, s_fill = false;
   for (unsigned i = 0; i < info->colorAttachmentCount; ++i) {
      col[i] = hk_deferred_clear_for(cmd, info, &info->pColorAttachments[i],
                                     VK_IMAGE_ASPECT_COLOR_BIT, &col_fill[i]);
   }
   struct hk_deferred_clear *z = hk_deferred_clear_for(
      cmd, info, info->pDepthAttachment, VK_IMAGE_ASPECT_DEPTH_BIT, &z_fill);
   struct hk_deferred_clear *s = hk_deferred_clear_for(
      cmd, info, info->pStencilAttachment, VK_IMAGE_ASPECT_STENCIL_BIT,
      &s_fill);

   /* A depth/stencil clear is taken only if every aspect it cleared is. */
   struct hk_deferred_clear *zs[2] = {z, s};
   for (unsigned i = 0; i < 2; ++i) {
      struct hk_deferred_clear *e = zs[i];
      if (e && (((e->aspects & VK_IMAGE_ASPECT_DEPTH_BIT) && z != e) ||
                ((e->aspects & VK_IMAGE_ASPECT_STENCIL_BIT) && s != e))) {
         if (z == e)
            z = NULL;
         if (s == e)
            s = NULL;
      }
   }

   bool any = z || s;
   for (unsigned i = 0; i < info->colorAttachmentCount; ++i)
      any |= col[i] != NULL;
   if (!any)
      return info;

   *copy = *info;
   for (unsigned i = 0; i < info->colorAttachmentCount; ++i) {
      color[i] = info->pColorAttachments[i];
      if (col[i]) {
         color[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
         color[i].clearValue = col[i]->value;
      }
   }
   copy->pColorAttachments = color;
   if (z) {
      *depth = *info->pDepthAttachment;
      depth->loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      depth->clearValue = z->value;
      copy->pDepthAttachment = depth;
   }
   if (s) {
      *stencil = *info->pStencilAttachment;
      stencil->loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      stencil->clearValue = s->value;
      copy->pStencilAttachment = stencil;
   }

   /* Drop the taken clears that need no fill; the caller records the rest
    * before the pass. */
   unsigned kept = 0;
   for (unsigned i = 0; i < cmd->state.deferred_clear_count; ++i) {
      struct hk_deferred_clear *e = &cmd->state.deferred_clear[i];
      bool taken = e == z || e == s;
      bool fill = (e == z && z_fill) || (e == s && s_fill);
      for (unsigned j = 0; j < info->colorAttachmentCount; ++j) {
         taken |= e == col[j];
         fill |= e == col[j] && col_fill[j];
      }
      if (!taken || fill)
         cmd->state.deferred_clear[kept++] = *e;
   }
   cmd->state.deferred_clear_count = kept;
   return copy;
}

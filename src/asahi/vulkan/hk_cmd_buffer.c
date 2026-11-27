/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_cmd_buffer.h"
#include "hk_compute_pair.h"
#include "hk_identity_copy4.h"
#include "util/format/u_format.h"
#include "hk_frame_anatomy.h"
#include "hk_ppp_state.h"
#include "hk_sampler_index.h"
#include "asahi/compiler/agx3_graphics_program.h"
#include "hk_g17p_compression.h"
#include "hk_g17p_patch.h"
#include "hk_g17p_varyings.h"

#include "agx_abi.h"
#include "agx_bo.h"
#include "agx_device.h"
#include "agx_linker.h"
#include "agx_tilebuffer.h"
#include "agx_usc.h"
#include "decode.h"
#include "hk_agx3_ioto_program_core.h"
#include "asahi/compiler/agx3_compute_program.h"
#include "asahi/compiler/agx3_transcode.h"
#include "asahi/lib/agx3_compute_program_core.h"
#include "asahi/lib/agx3_cdm_direct_launch_semantic_core.h"
#include "asahi/compiler/agx_debug.h"
#include "hk_agx3_vdm_decode.h"
#include "asahi/lib/agx_ioto_load_core.h"
#include "asahi/lib/agx_chip_caps.h"
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
#include "hk_queue.h"
#include "libagx_shaders.h"

#include "util/cache_ops.h"
#include "util/os_misc.h"
#include "util/u_atomic.h"
#include <inttypes.h>
#include <stdio.h>
#include <unistd.h>
#include "util/log.h"
#include "util/u_debug.h"
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
   if (last && last->alias && primary >= last->addr &&
       primary - last->addr < last->size) {
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
      if (!allocation->alias || primary < allocation->addr ||
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
         const uint64_t start = bo->addr;
         if (bo->span || primary < start || primary - start >= bo->size)
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

/* Writes the stream words of the patch sites. Store whole words (the GPU may
 * read the stream meanwhile when the alias count leaves 0), clean each patched
 * cache line when command memory is CPU-cached, and fence before the
 * submission or host signal that lets the GPU run them. */
void
hk_cmd_buffer_patch_alias_sites(struct hk_cmd_buffer *cmd)
{
   if (p_atomic_read(&cmd->alias_patched) || !cmd->alias_sites.size)
      return;

   const bool cached =
      hk_g17p_cached_cmd_memory(&hk_cmd_buffer_device(cmd)->dev);
   util_dynarray_foreach(&cmd->alias_sites, struct hk_alias_site, site) {
      __atomic_store_n(site->word, site->value, __ATOMIC_RELAXED);
      if (cached)
         util_flush_range_no_fence(site->word, sizeof(*site->word));
   }

   __sync_synchronize();
   if (util_has_cache_ops())
      util_post_flush_fence();
   p_atomic_set(&cmd->alias_patched, 1);
}

void
hk_cmd_buffer_move_sparse_sites(struct hk_cmd_buffer *cmd, const void *from,
                                const void *from_end, uint8_t *to, size_t skip)
{
   util_dynarray_foreach(&cmd->sparse_sites, struct hk_sparse_site, site) {
      const uint8_t *word = (const uint8_t *)site->word;
      if (word < (const uint8_t *)from || word >= (const uint8_t *)from_end)
         continue;

      /* As the alias sites: a site in the folded leading record moves to
       * the trailing record that absorbed it. */
      const size_t offset = word - (const uint8_t *)from;
      site->word = (uint32_t *)(offset < skip ? to - AGX3_CDM_BARRIER_SIZE
                                             : to + offset - skip);
   }
}

void
hk_cmd_buffer_move_alias_sites(struct hk_cmd_buffer *cmd, const void *from,
                               const void *from_end, uint8_t *to, size_t skip)
{
   util_dynarray_foreach(&cmd->alias_sites, struct hk_alias_site, site) {
      const uint8_t *word = (const uint8_t *)site->word;
      if (word < (const uint8_t *)from || word >= (const uint8_t *)from_end)
         continue;

      /* A site in the folded leading record moves to the trailing record
       * that absorbed it. */
      const size_t offset = word - (const uint8_t *)from;
      site->word = (uint32_t *)(offset < skip ? to - AGX3_CDM_BARRIER_SIZE
                                              : to + offset - skip);
   }
}

/* At vkEndCommandBuffer, once nothing moves the streams any more: a command
 * buffer that uses events gets memory scope throughout
 * (hk_cmd_alias_scope()), one recorded while a resource may alias by now is
 * written at once, and any other waits in host_alias.cmd_buffers for
 * hk_device_patch_alias_sites(), besides the check at submission. */
static void
hk_cmd_buffer_end_alias_sites(struct hk_cmd_buffer *cmd)
{
   if (!cmd->alias_sites.size)
      return;

   if (cmd->state.uses_events) {
      hk_cmd_buffer_patch_alias_sites(cmd);
      return;
   }

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   simple_mtx_lock(&dev->host_alias.cmd_lock);
   const bool scope = hk_cdm_alias_scope(dev);
   if (!scope) {
      list_addtail(&cmd->alias_link, &dev->host_alias.cmd_buffers);
      p_atomic_set(&cmd->alias_registered, 1);
   }
   simple_mtx_unlock(&dev->host_alias.cmd_lock);

   if (scope)
      hk_cmd_buffer_patch_alias_sites(cmd);
}

/* Before the recording's command memory is released. */
static void
hk_cmd_buffer_reset_alias_sites(struct hk_cmd_buffer *cmd)
{
   if (p_atomic_read(&cmd->alias_registered)) {
      struct hk_device *dev = hk_cmd_buffer_device(cmd);
      simple_mtx_lock(&dev->host_alias.cmd_lock);
      if (cmd->alias_registered) {
         list_del(&cmd->alias_link);
         p_atomic_set(&cmd->alias_registered, 0);
      }
      simple_mtx_unlock(&dev->host_alias.cmd_lock);
   }

   util_dynarray_clear(&cmd->alias_sites);
   p_atomic_set(&cmd->alias_patched, 0);
}

/* A command buffer's sites are written once, under host_alias.patch_lock
 * held exclusively unless nothing can read its streams yet
 * (hk_cmd_buffer_end_alias_sites()): submit packing copies a submitted
 * command buffer's streams under the shared lock, and could otherwise copy a
 * word half written. Each word is stored whole, so the GPU itself reads
 * either record. */
unsigned
hk_device_patch_alias_sites(struct hk_device *dev)
{
   unsigned patched = 0;
   simple_mtx_lock(&dev->host_alias.cmd_lock);
   if (!list_is_empty(&dev->host_alias.cmd_buffers)) {
      u_rwlock_wrlock(&dev->host_alias.patch_lock);
      list_for_each_entry_safe(struct hk_cmd_buffer, cmd,
                               &dev->host_alias.cmd_buffers, alias_link) {
         hk_cmd_buffer_patch_alias_sites(cmd);
         list_del(&cmd->alias_link);
         /* After the words: a reset that sees 0 may release them. */
         p_atomic_set(&cmd->alias_registered, 0);
         patched++;
      }
      u_rwlock_wrunlock(&dev->host_alias.patch_lock);
   }
   simple_mtx_unlock(&dev->host_alias.cmd_lock);
   return patched;
}

void
hk_device_alias_began(struct hk_device *dev, enum hk_alias_cause cause)
{
   const unsigned patched = hk_device_patch_alias_sites(dev);
   if (dev->host_alias.trace) {
      static const char *const names[] = {
         [HK_ALIAS_CAUSE_IMPORT] = "import",
         [HK_ALIAS_CAUSE_SPARSE] = "sparse",
         [HK_ALIAS_CAUSE_CAPTURE_REPLAY] = "capture-replay",
         [HK_ALIAS_CAUSE_FD] = "fd",
      };
      fprintf(stderr, "M5_HOST_ALIAS began cause=%s patched=%u\n",
              names[cause], patched);
   }
}

void
hk_submit_patch_alias_sites(struct hk_device *dev,
                            struct vk_command_buffer *const *cmds,
                            unsigned count)
{
   bool unpatched = false;
   for (unsigned i = 0; i < count && !unpatched; ++i) {
      const struct hk_cmd_buffer *cmd = (const void *)cmds[i];
      unpatched = cmd->alias_sites.size && !p_atomic_read(&cmd->alias_patched);
   }
   if (!unpatched)
      return;

   u_rwlock_wrlock(&dev->host_alias.patch_lock);
   for (unsigned i = 0; i < count; ++i)
      hk_cmd_buffer_patch_alias_sites((struct hk_cmd_buffer *)cmds[i]);
   u_rwlock_wrunlock(&dev->host_alias.patch_lock);
}

void
hk_cdm_note_sparse_site(struct hk_cs *cs, uint32_t *word)
{
   struct hk_sparse_site *site =
      util_dynarray_grow(&cs->cmd->sparse_sites, struct hk_sparse_site, 1);
   if (site) {
      *site = (struct hk_sparse_site){
         .word = word,
         .value = AGX_CDM_BARRIER_FULL_WORD |
                  (AGX_CDM_BLOCK_TYPE_BARRIER << 29),
      };
      cs->sparse_sites++;
   }
}

/* Writes the stream words of the sparse patch sites. Same store discipline
 * as the alias sites: the words are stored whole, then made visible before
 * the submission that lets the GPU run them. */
static void
hk_cmd_buffer_patch_sparse_sites(struct hk_cmd_buffer *cmd)
{
   util_dynarray_foreach(&cmd->sparse_sites, struct hk_sparse_site, site)
      __atomic_store_n(site->word, site->value, __ATOMIC_RELAXED);

   __sync_synchronize();
   if (util_has_cache_ops())
      util_post_flush_fence();
}

void
hk_submit_patch_sparse_sites(struct hk_device *dev,
                             struct vk_command_buffer *const *cmds,
                             unsigned count)
{
   bool unpatched = false;
   for (unsigned i = 0; i < count && !unpatched; ++i) {
      const struct hk_cmd_buffer *cmd = (const void *)cmds[i];
      unpatched = cmd->sparse_sites.size != 0;
   }
   if (!unpatched)
      return;

   u_rwlock_wrlock(&dev->host_alias.patch_lock);
   for (unsigned i = 0; i < count; ++i)
      hk_cmd_buffer_patch_sparse_sites((struct hk_cmd_buffer *)cmds[i]);
   u_rwlock_wrunlock(&dev->host_alias.patch_lock);
}

static void
hk_free_resettable_cmd_buffer(struct hk_cmd_buffer *cmd)
{
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);
   struct hk_device *dev = hk_cmd_pool_device(pool);
   hk_cmd_buffer_reset_alias_sites(cmd);
   util_dynarray_clear(&cmd->sparse_sites);
   hk_cmd_pending_reap(dev);
   hk_cmd_pending_retain_lost(dev, cmd);

   /* The cached span is only valid while this recording owns its leases. */
   cmd->g17p_vdm_last_allocation = NULL;

   hk_descriptor_state_fini(cmd, &cmd->state.gfx.descriptors);
   hk_descriptor_state_fini(cmd, &cmd->state.cs.descriptors);
   if (unlikely(cmd->used_rt))
      hk_descriptor_state_fini(cmd, &cmd->rt.descriptors);

   hk_cmd_pool_free_bo_list(pool, HK_CMD_BO_MAIN, &cmd->uploader.main.bos);
   hk_cmd_pool_free_bo_list(pool, HK_CMD_BO_USC, &cmd->uploader.usc.bos);
   hk_cmd_pool_free_bo_list(pool, HK_CMD_BO_ROOT, &cmd->uploader.root.bos);
   list_for_each_entry_safe(struct hk_vdm_allocation, allocation,
                            &cmd->vdm_allocations, link) {
      list_del(&allocation->link);
      hk_cmd_pool_free_vdm_span(pool, allocation);
   }

   list_for_each_entry_safe(struct hk_cs, it, &cmd->control_streams, node) {
      list_del(&it->node);
      hk_cs_recycle(pool, it);
   }
   hk_tess_batch_reset(cmd);

   hk_ioto_arena_finish(cmd);
   hk_cmd_buffer_release_bos(cmd, &dev->dev);
   util_dynarray_clear(&cmd->meta_push_saves);
}

static void
hk_destroy_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer)
{
   struct hk_cmd_buffer *cmd =
      container_of(vk_cmd_buffer, struct hk_cmd_buffer, vk);
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);

   hk_device_quiesce_lost_submissions(hk_cmd_pool_device(pool));
   hk_free_resettable_cmd_buffer(cmd);
   util_dynarray_fini(&cmd->alias_sites);
   util_dynarray_fini(&cmd->sparse_sites);
   util_dynarray_fini(&cmd->large_bos);
   util_dynarray_fini(&cmd->meta_push_saves);
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
   cmd->alias_sites = UTIL_DYNARRAY_INIT;
   cmd->sparse_sites = UTIL_DYNARRAY_INIT;
   cmd->meta_push_saves = UTIL_DYNARRAY_INIT;
   list_inithead(&cmd->alias_link);

   cmd->vk.dynamic_graphics_state.vi = &cmd->state.gfx._dynamic_vi;
   cmd->vk.dynamic_graphics_state.ms.sample_locations =
      &cmd->state.gfx._dynamic_sl;

   list_inithead(&cmd->uploader.main.bos);
   list_inithead(&cmd->uploader.usc.bos);
   list_inithead(&cmd->uploader.root.bos);
   list_inithead(&cmd->vdm_allocations);
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

   hk_device_quiesce_lost_submissions(hk_cmd_pool_device(hk_cmd_buffer_pool(cmd)));
   vk_command_buffer_reset(&cmd->vk);
   hk_free_resettable_cmd_buffer(cmd);

   cmd->uploader.main.map = NULL;
   cmd->uploader.main.base = 0;
   cmd->uploader.main.offset = 0;
   cmd->uploader.main.capacity = 0;
   cmd->uploader.main.cpu_cached = false;
   cmd->uploader.usc.map = NULL;
   cmd->uploader.usc.base = 0;
   cmd->uploader.usc.offset = 0;
   cmd->uploader.usc.capacity = 0;
   cmd->uploader.usc.cpu_cached = false;
   cmd->uploader.root.map = NULL;
   cmd->uploader.root.base = 0;
   cmd->uploader.root.offset = 0;
   cmd->uploader.root.capacity = 0;
   cmd->uploader.root.cpu_cached = false;
   cmd->uploader.vdm.map = NULL;
   cmd->uploader.vdm.base = 0;
   cmd->uploader.vdm.offset = 0;
   cmd->uploader.vdm.capacity = 0;
   cmd->uploader.vdm.vdm_alias = 0;
   cmd->uploader.vdm.cpu_cached = false;
   cmd->uploader.streams.map = NULL;
   cmd->uploader.streams.base = 0;
   cmd->uploader.streams.offset = 0;
   cmd->uploader.streams.capacity = 0;
   cmd->uploader.streams.cpu_cached = false;

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
   cmd->heap_slot_assigned = false;
   cmd->heap_slot = 0;
   hk_sync_reset(cmd);
   if (cmd->m5_pushed_images)
      _mesa_set_clear(cmd->m5_pushed_images, NULL);
   util_dynarray_clear(&cmd->oq_ends);

   /* Reset the common state on either side of the four contiguous native
    * graphics caches (g17p_ioto..g17p_spilled_rt_descs, ~8.5 KiB) without
    * zeroing their whole payload on every reset. When populated, clear only
    * their validity/identity tags. */
   const size_t first =
      offsetof(struct hk_cmd_buffer, state.gfx.g17p_ioto) -
      offsetof(struct hk_cmd_buffer, state);
   const size_t end =
      offsetof(struct hk_cmd_buffer, state.gfx.g17p_spilled_rt_descs) +
      sizeof(cmd->state.gfx.g17p_spilled_rt_descs) -
      offsetof(struct hk_cmd_buffer, state);
   memset(&cmd->state, 0, first);
   memset((char *)&cmd->state + end, 0, sizeof(cmd->state) - end);
   if (cmd->g17p_gfx_caches_used) {
      for (unsigned stage = 0; stage < ARRAY_SIZE(cmd->state.gfx.g17p_ioto);
           ++stage) {
         cmd->state.gfx.g17p_ioto[stage].loader = NULL;
         cmd->state.gfx.g17p_ioto[stage].loader_linked = NULL;
         cmd->state.gfx.g17p_ioto[stage].loader_program = 0;
         cmd->state.gfx.g17p_ioto[stage].admit.s = NULL;
         cmd->state.gfx.g17p_ioto[stage].admit.linked = NULL;
      }
      for (unsigned stage = 0; stage < ARRAY_SIZE(cmd->state.gfx.g17p_launch);
           ++stage) {
         for (unsigned i = 0; i < ARRAY_SIZE(cmd->state.gfx.g17p_launch[stage]);
              ++i)
            cmd->state.gfx.g17p_launch[stage][i].valid = false;
      }
      for (unsigned i = 0;
           i < ARRAY_SIZE(cmd->state.gfx.g17p_combined_sampler_pointers); ++i)
         cmd->state.gfx.g17p_combined_sampler_pointers[i].valid = false;
      cmd->state.gfx.g17p_spilled_rt_descs.valid = false;
      cmd->g17p_gfx_caches_used = false;
   }
   if (cmd->g17p_bg_eot_cache_used) {
      for (unsigned i = 0; i < HK_G17P_BG_EOT_CACHE_ENTRIES; ++i)
         cmd->g17p_bg_eot_cache[i].valid = false;
      cmd->g17p_bg_eot_cache_next = 0;
      cmd->g17p_bg_eot_cache_used = false;
   }
   if (unlikely(cmd->used_rt)) {
      memset(&cmd->rt, 0, sizeof(cmd->rt));
      cmd->used_rt = false;
   }
   if (cmd->used_group_count_cache) {
      for (unsigned i = 0; i < ARRAY_SIZE(cmd->group_count_cache); ++i)
         cmd->group_count_cache[i].valid = 0;
      cmd->used_group_count_cache = false;
   }
   cmd->pass_merge.valid = false;
   memset(&cmd->render_pack, 0, sizeof(cmd->render_pack));
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
   cmd->state.cs.cdm_barrier_full_needed = true;
   cmd->state.cs.cdm_stream_has_images = true;

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
hk_cmd_buffer_alloc_bo(struct hk_cmd_buffer *cmd,
                       const struct hk_uploader *uploader, uint32_t size,
                       enum hk_cmd_bo_type type, struct hk_cmd_bo **bo_out)
{
   struct hk_cmd_pool *pool = hk_cmd_buffer_pool(cmd);
   const uint32_t span = uploader == &cmd->uploader.streams
                            ? HK_CMD_STREAM_SPAN_SIZE
                            : HK_CMD_SPAN_SIZE;
   VkResult result =
      !uploader->map && size <= span && list_is_empty(&pool->free_bos[type]) &&
            hk_g17p_cached_cmd_memory(&hk_cmd_buffer_device(cmd)->dev)
         ? hk_cmd_pool_alloc_bo_span(pool, type, span, bo_out)
         : hk_cmd_pool_alloc_bo(pool, type, bo_out);
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
hk_pool_alloc_from_slow_cached(struct hk_cmd_buffer *cmd,
                               struct hk_uploader *uploader, uint32_t size,
                               uint32_t alignment, enum hk_cmd_bo_type type,
                               bool *cpu_cached_out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (cpu_cached_out)
      *cpu_cached_out = false;

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

      void *cpu = agx_bo_map(bo);
      if (unlikely(cpu == NULL)) {
         agx_bo_unreference(&dev->dev, bo);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return (struct agx_ptr){0};
      }

      struct agx_bo **slot =
         util_dynarray_grow(&cmd->large_bos, struct agx_bo *, 1);
      if (unlikely(slot == NULL)) {
         agx_bo_unreference(&dev->dev, bo);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return (struct agx_ptr){0};
      }
      *slot = bo;
      if (cpu_cached_out)
         *cpu_cached_out = (bo->flags & AGX_BO_WRITEBACK) != 0;
      return (struct agx_ptr){
         .gpu = bo->va->addr,
         .cpu = cpu,
      };
   }

   assert(size <= HK_CMD_BO_SIZE);
   assert(alignment > 0);

   uint32_t offset = align(uploader->offset, alignment);

   if (uploader->map != NULL && offset <= uploader->capacity &&
       size <= uploader->capacity - offset) {
      uploader->offset = offset + size;
      if (cpu_cached_out)
         *cpu_cached_out = uploader->cpu_cached;

      return (struct agx_ptr){
         .gpu = uploader->base + offset,
         .cpu = uploader->map + offset,
      };
   }

   struct hk_cmd_bo *bo;
   VkResult result = hk_cmd_buffer_alloc_bo(cmd, uploader, size, type, &bo);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return (struct agx_ptr){0};
   }
   bo->stream_only = type == HK_CMD_BO_MAIN &&
                     uploader == &cmd->uploader.streams;

   const bool bo_cpu_cached = bo->bo && (bo->bo->flags & AGX_BO_WRITEBACK);

   /* Pick whichever of the current upload BO and the new BO will have more
    * room left to be the BO for the next upload.  If our upload size is
    * bigger than the old offset, we're better off burning the whole new
    * upload BO on this one allocation and continuing on the current upload
    * BO.
    */
   if (uploader->map == NULL ||
       bo->size - size > uploader->capacity - uploader->offset) {
      uploader->map = bo->map;
      uploader->base = bo->addr;
      uploader->offset = size;
      uploader->capacity = bo->size;
      uploader->cpu_cached = bo_cpu_cached;
   }

   if (cpu_cached_out)
      *cpu_cached_out = bo_cpu_cached;

   return (struct agx_ptr){
      .gpu = bo->addr,
      .cpu = bo->map,
   };
}

struct agx_ptr
hk_pool_alloc_from_slow(struct hk_cmd_buffer *cmd,
                        struct hk_uploader *uploader, uint32_t size,
                        uint32_t alignment, enum hk_cmd_bo_type type)
{
   return hk_pool_alloc_from_slow_cached(cmd, uploader, size, alignment, type,
                                         NULL);
}

static inline struct agx_ptr
hk_pool_alloc_cached(struct hk_cmd_buffer *cmd, uint32_t size,
                     uint32_t alignment, bool *cpu_cached_out)
{
   struct hk_uploader *uploader = &cmd->uploader.main;
   assert(alignment > 0);
   const uint32_t offset = align(uploader->offset, alignment);
   if (likely(uploader->map != NULL && offset <= uploader->capacity &&
              size <= uploader->capacity - offset)) {
      uploader->offset = offset + size;
      *cpu_cached_out = uploader->cpu_cached;
      return (struct agx_ptr){
         .gpu = uploader->base + offset,
         .cpu = uploader->map + offset,
      };
   }

   return hk_pool_alloc_from_slow_cached(cmd, uploader, size, alignment,
                                         HK_CMD_BO_MAIN, cpu_cached_out);
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

   /* Keep control-stream roots on their own bump cursor on native AGX3 and M5
    * so allocating one cannot evict a partially used main-upload BO and so
    * unlinked stream BOs are marked stream_only for exact-extent cache cleans
    * at vkEndCommandBuffer(). Ownership remains on the ordinary main BO list. */
   struct hk_uploader *uploader = agx_chip_native_or_m5(dev->dev.chip)
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
   const VkResult result = hk_cmd_pool_alloc_vdm_span(
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
   else if (vdm && agx_chip_hal300_path(dev->dev.chip)) {
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

void *
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
   cmd->usage_flags = pBeginInfo->flags;
   vk_command_buffer_begin(&cmd->vk, pBeginInfo);
   cmd->state.compute_pair_simultaneous =
      pBeginInfo->flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

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

   /* Submit-time render packing: until something but a pipeline barrier is
    * recorded, the first render pass instance may continue the previous
    * command buffer's last one (hk_render_pack.h). */
   cmd->render_pack.head_open =
      cmd->vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   cmd->render_pack.head_epoch = cmd->sync.epoch;

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
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const bool g18 = dev->dev.chip == AGX_CHIP_G18G;
   struct hk_cs *last = NULL;

   list_for_each_entry_safe(struct hk_cs, cs, &cmd->control_streams, node) {
      /* G18G CDM streams never merge: launches would add past the
       * HK_G18_BATCH_LAUNCHES cap and hk_cs_merge_cdm does not combine
       * g18_compute_resources, so the merged command would keep only the
       * first stream's registers, staging and local size. */
      if (cs->type == HK_CS_CDM && last && last->type == HK_CS_CDM && !g18 &&
          !last->g18_dispatch_sealed && !cs->g18_dispatch_sealed &&
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
hk_cmd_buffer_publish_root_uploads(struct hk_cmd_buffer *cmd,
                                   bool already_fenced)
{
   if (list_is_empty(&cmd->uploader.root.bos))
      return;

   if (util_has_cache_ops()) {
      /* Every root range was cleaned when written. Complete those cache
       * operations once, after recording has made all roots immutable. */
      if (!already_fenced)
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
                                 const struct hk_cmd_bo *bo, bool *cleaned)
{
   const uintptr_t base = (uintptr_t)bo->map;
   if (bo->size > UINTPTR_MAX - base)
      return false;
   const uintptr_t limit = base + bo->size;

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
      if (bytes) {
         util_flush_range_no_fence((void *)start, bytes);
         *cleaned = true;
      }
   }
   return true;
}

/* Clean everything recording wrote to CPU-cached command memory (see
 * hk_g17p_cached_cmd_memory()) and issue one completion fence covering both
 * these cleans and any per-upload root cleans. */
static bool
hk_cmd_buffer_publish_cached_uploads(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!hk_g17p_cached_cmd_memory(&dev->dev))
      return false;

   bool linked_stream = false;
   list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node)
      linked_stream |= cs->stream_linked;

   /* An uploader's current BO is written up to its offset; a BO it moved
    * on from may have been written anywhere. Dedicated unlinked stream BOs
    * instead have exact written extents, independent of reserved capacity. */
   const struct hk_uploader *current[] = {
      &cmd->uploader.main, &cmd->uploader.streams, &cmd->uploader.usc,
   };
   bool cleaned = false;
   for (unsigned l = 0; l < 2; ++l) {
      struct list_head *bos =
         l == 0 ? &cmd->uploader.main.bos : &cmd->uploader.usc.bos;
      list_for_each_entry(struct hk_cmd_bo, bo, bos, link) {
         if (bo->stream_only && !linked_stream &&
             hk_cmd_bo_clean_unlinked_streams(cmd, bo, &cleaned))
            continue;
         size_t size = bo->size;
         for (unsigned u = 0; u < ARRAY_SIZE(current); ++u) {
            if (current[u]->map == bo->map)
               size = MIN2(size, current[u]->offset);
         }
         if (size) {
            util_flush_range_no_fence(bo->map, size);
            cleaned = true;
         }
      }
   }
   list_for_each_entry(struct hk_vdm_allocation, allocation,
                       &cmd->vdm_allocations, link) {
      hk_cmd_pool_clean_vdm(allocation);
      cleaned = true;
   }
   cleaned |= hk_ioto_arena_clean(cmd);
   if (!cleaned && list_is_empty(&cmd->uploader.root.bos))
      return false;

   util_post_flush_fence();
   return true;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   assert(cmd->current_cs.gfx == NULL && cmd->current_cs.pre_gfx == NULL &&
          "must end rendering before ending the command buffer");

   perf_debug(cmd, "End command buffer");
   hk_flush_deferred_clears(cmd);
   hk_sync_flush_stamp_copies(cmd);
   hk_sync_flush_pending_timestamp(cmd);
   hk_cmd_buffer_end_compute(cmd);
   cmd->state.cs.cdm_barrier_pending = false;
   cmd->state.cs.cdm_barrier_full_needed = false;
   cmd->state.cs.cdm_stream_has_images = false;

   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);
   hk_resolve_pending_oq_available(cmd);
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
      /* A G18G native dispatch stream is sealed: the launch is followed by its
       * barrier and its own stream terminate (hk_cmd_dispatch.c, hk_cmd_buffer.c
       * helper path).  Appending a second terminate makes the stream 52 bytes;
       * the G18 kernel admits at most 48 (one dispatch) and refuses the submit
       * with ENOTSUPP ("G18 compute admission rejected ... stream_bytes=52").
       */
      if (cs->type == HK_CS_CDM && !cs->g18_dispatch_sealed) {
         /* With host_alias.dynamic, every stream holding work ends with the
          * memory-scoped strong flush, also where its last record is an
          * application barrier recorded without memory scope. Each command
          * then writes back what it wrote by its end, as the memory scope of
          * each barrier did under the per-device rule, so a later command
          * reads it through any address, also through a resource that
          * aliases created in between, whose patch sites cover only the
          * commands recorded before it (hk_cdm_alias_scope()), or through
          * another device's, process's or API's mapping of the same pages,
          * which Vulkan orders after this work only at a submission
          * boundary. Submit packing keeps the scope only at a packet's end
          * while no resource may alias. */
         if (dev->host_alias.dynamic &&
             hk_cdm_visibility_flush_needed(dev, cs, true)) {
            if (cs->current != cs->weak_barrier_end &&
                cs->current != cs->strong_flush_end)
               hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
            hk_cdm_visibility_flush(dev, cs, true);
         }

         /* Where the kernel lets another queue's kick wait for this stream's
          * completion in firmware, which a weak tail reports before the
          * stream's writes are visible, end the stream with the strong
          * memory-scoped flush, upgrading the weak record in place, and say
          * so in the command (DRM_ASAHI_COMPUTE_VISIBLE_COMPLETION).
          */
         if (hk_cdm_visible_completion(dev)) {
            if (cs->current == cs->weak_barrier_end)
               hk_cdm_visibility_flush(dev, cs, true);
            cs->visible_completion = cs->current == cs->strong_flush_end &&
                                     cs->current == cs->memory_flush_end;
         }
         cs->current = hk_cmd_buffer_emit_cdm_terminate(cmd, cs->current);
      }
   }

   /* G17P roots use cached mappings. Publish the completed recording once,
    * after every root table is immutable and before submission can consume
    * any address embedded in the command streams. */
   const bool fenced = hk_cmd_buffer_publish_cached_uploads(cmd);
   hk_cmd_buffer_publish_root_uploads(cmd, fenced);
   hk_cmd_buffer_end_alias_sites(cmd);

   hk_render_pack_end_cmd(cmd);

   hk_cmd_print_upload_stats(cmd);
   return vk_command_buffer_end(&cmd->vk);
}

/* Whether any barrier in the dependency requests write availability (source
 * write accesses) or read/host visibility (destination read or host write
 * accesses), or performs a memory operation of its own. A barrier with none
 * is execution-only or WAR and requires execution ordering rather than a
 * cache flush/invalidation.
 *
 * Image layout transitions and queue family ownership transfers are such
 * operations whatever the access masks say: a buffer or image barrier whose
 * srcQueueFamilyIndex and dstQueueFamilyIndex differ is a release or an
 * acquire, and Vulkan defines it through the queue family indices even when
 * the other half's synchronization scope does not apply. Both stay
 * conservative here, as in hk_barrier_classify_ex(). */
static bool
hk_dependency_needs_cache_flush(const VkDependencyInfo *info)
{
   for (uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
      const VkMemoryBarrier2 *b = &info->pMemoryBarriers[i];
      if (vk_filter_src_access_flags2(
             vk_expand_src_stage_flags2(b->srcStageMask), b->srcAccessMask) ||
          vk_filter_dst_access_flags2(
             vk_expand_dst_stage_flags2(b->dstStageMask), b->dstAccessMask))
         return true;
   }

   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) {
      const VkBufferMemoryBarrier2 *b = &info->pBufferMemoryBarriers[i];
      if (b->srcQueueFamilyIndex != b->dstQueueFamilyIndex)
         return true;
      if (vk_filter_src_access_flags2(
             vk_expand_src_stage_flags2(b->srcStageMask), b->srcAccessMask) ||
          vk_filter_dst_access_flags2(
             vk_expand_dst_stage_flags2(b->dstStageMask), b->dstAccessMask))
         return true;
   }

   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
      const VkImageMemoryBarrier2 *b = &info->pImageMemoryBarriers[i];
      /* Equal layouts, including UNDEFINED, are not transitions (sync2). */
      if (b->oldLayout != b->newLayout ||
          b->srcQueueFamilyIndex != b->dstQueueFamilyIndex)
         return true;
      if (vk_filter_src_access_flags2(
             vk_expand_src_stage_flags2(b->srcStageMask), b->srcAccessMask) ||
          vk_filter_dst_access_flags2(
             vk_expand_dst_stage_flags2(b->dstStageMask), b->dstAccessMask))
         return true;
   }

   return false;
}

/* Images are compared as whole resources. Subresource-disjoint barriers may
 * still split, and aliasable/sparse attachments disable the unwritten-image
 * proof. Never infer disjointness from different Vulkan handles alone.
 */
static uint32_t
hk_classify_pass_barrier(struct hk_cmd_buffer *cmd, const VkDependencyInfo *info,
                         bool *used_local_read)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   VkImage images[HK_MAX_RTS + 2];
   struct hk_barrier_state state = {
      .shader_writes = render->barrier_shader_writes,
      .pre_graphics = cmd->current_cs.pre_gfx != NULL,
      .post_graphics = cmd->current_cs.post_gfx != NULL ||
                       cmd->state.oq_pending_render != NULL,
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
      if (i < render->color_att_count)
         state.color_attachment_count++;
   }
   state.attachments_alias = state.attachment_count && aliases;
   state.depth_stencil_writes =
      state.attachment_count > state.color_attachment_count;
   state.color_input_local_read =
      hk_local_read_tilebuffer(dev->dev.chip) && render->view_mask == 0 &&
      cmd->vk.level == VK_COMMAND_BUFFER_LEVEL_PRIMARY &&
      !(render->flags & (VK_RENDERING_SUSPENDING_BIT |
                         VK_RENDERING_RESUMING_BIT |
                         VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT)) &&
      !state.tilebuffer_spilled && !state.attachments_alias;

   return hk_barrier_classify_ex(info, &state, used_local_read);
}

/* Whether an image barrier discards the contents of a compressed storage
 * image that this queue must initialize. A shader store can update only part
 * of a compression tile, so it must not meet the metadata the memory held
 * before. Of an ownership transfer, only the acquire initializes (*acquire). */
static bool
brl_barrier_needs_metadata_init(const struct hk_cmd_buffer *cmd,
                                const VkImageMemoryBarrier2 *bar,
                                bool *acquire)
{
   if (bar->oldLayout != VK_IMAGE_LAYOUT_UNDEFINED ||
       bar->newLayout == VK_IMAGE_LAYOUT_UNDEFINED)
      return false;

   VK_FROM_HANDLE(hk_image, image, bar->image);
   if (!(image->vk.usage & VK_IMAGE_USAGE_STORAGE_BIT) ||
       !image->planes[0].layout.compressed)
      return false;

   /* Gen10 (HAL300) compressed storage PBE merges partial stores directly
    * into compressed tiles regardless of prior metadata contents
    * (tests/vulkan_storage_alias.c), so discarding from UNDEFINED does not
    * require pre-clearing compression metadata. */
   if (hk_chip_supports_compressed_storage_pbe(
          hk_cmd_buffer_device(cmd)->dev.chip))
      return false;

   *acquire = false;
   if (image->vk.sharing_mode != VK_SHARING_MODE_EXCLUSIVE ||
       bar->srcQueueFamilyIndex == bar->dstQueueFamilyIndex)
      return true;

   *acquire = bar->dstQueueFamilyIndex == cmd->vk.pool->queue_family_index;
   return *acquire;
}

bool
brl_discard_init_prepare(struct hk_cmd_buffer *cmd,
                         const VkDependencyInfo *info, uint8_t *dst_kinds)
{
   /* A valid dependency inside a render pass instance keeps every layout. */
   if (cmd->sync.in_pass)
      return false;

   uint8_t src_kinds = 0;
   bool any = false;
   *dst_kinds = 0;
   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; i++) {
      const VkImageMemoryBarrier2 *bar = &info->pImageMemoryBarriers[i];
      bool acquire;
      if (!brl_barrier_needs_metadata_init(cmd, bar, &acquire))
         continue;

      brl_sync_transition_kinds(info, bar, hk_sync_image_class(cmd), acquire,
                                cmd->vk.runtime_rp_barrier, &src_kinds,
                                dst_kinds);
      any = true;
   }

   if (!any)
      return false;

   /* A deferred clear was recorded before the dependency: it belongs to
    * the first scope like any other earlier work. */
   hk_flush_deferred_clears(cmd);
   brl_sync_transition_prepare(cmd, src_kinds);
   return true;
}

void
brl_discard_init_record(struct hk_cmd_buffer *cmd,
                        const VkDependencyInfo *info, uint8_t dst_kinds)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* The clears get streams of their own, which wait on the transition's
    * requirement alone. */
   hk_cmd_buffer_end_compute(cmd);
   brl_sync_transition_begin(cmd);

   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; i++) {
      const VkImageMemoryBarrier2 *bar = &info->pImageMemoryBarriers[i];
      bool acquire;
      if (!brl_barrier_needs_metadata_init(cmd, bar, &acquire))
         continue;

      VK_FROM_HANDLE(hk_image, image, bar->image);
      const uint32_t zero[4] = {0};
      hk_clear_image(cmd, image, PIPE_FORMAT_NONE, zero, &bar->subresourceRange,
                     true);
   }

   /* The clears are device writes that the second scope reads, as an
    * application transfer-to-shader dependency would order them. */
   const bool alias_scope = hk_cmd_alias_scope(cmd);
   struct hk_cs *cs = cmd->current_cs.cs;
   if (cs && dst_kinds &&
       hk_cdm_visibility_flush_needed(dev, cs, alias_scope)) {
      hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
      hk_cdm_visibility_flush(dev, cs, alias_scope);
   }
   hk_cmd_buffer_end_compute(cmd);
   brl_sync_transition_end(cmd, dst_kinds);

   /* The transition is work now: an ended pass instance cannot continue
    * across it (hk_m5_pass_merge_classify() assumes transitions do none). */
   cmd->pass_merge.valid = false;
   cmd->render_pack.tail_open = false;
   cmd->render_pack.tail_why = "layout transition work";
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
   if (!agx_chip_native_or_m5(dev->dev.chip)) {
      /* A BY_REGION colour input may stay in this render. The consuming
       * draw checks the bound descriptors and compiled tile-read program
       * before emitting geometry. A failed check splits the render. */
      bool local_read = false;
      if (cmd->sync.in_pass && cmd->current_cs.gfx &&
          (pDependencyInfo->dependencyFlags & VK_DEPENDENCY_BY_REGION_BIT) &&
          !hk_classify_pass_barrier(cmd, pDependencyInfo, &local_read) &&
          local_read) {
         uint8_t src, dst;
         hk_sync_dependency_kinds(pDependencyInfo, hk_sync_image_class(cmd),
                                  &src, &dst);
         cmd->state.gfx.render.kept_local_read_barrier = true;
         cmd->state.barrier_stats.kept++;
         hk_sync_pass_barrier(cmd, src, dst);
         return;
      }

      /* On AGX2 outside a render pass, keep an open compute stream alive
       * across the barrier so the next pure compute SSBO launch can narrow
       * the previous launch's trailing barrier word in place (hk_cmd_dispatch.c).
       * Non-AGX2 command streams (such as G18G/HAL300), fullcdmbarrier, and
       * open graphics batches still end the compute stream as on the base. */
      if (hk_cdm_record_pipeline_barrier(cmd, pDependencyInfo))
         hk_cmd_buffer_end_compute(cmd);

      hk_cmd_buffer_end_graphics(cmd);
      /* Preserve the same conservative ordering in the shared submit graph,
       * including kernels that advertise exact prior barriers on AGX2. */
      uint8_t init_dst;
      const bool init =
         brl_discard_init_prepare(cmd, pDependencyInfo, &init_dst);
      hk_sync_barrier(cmd, HK_SYNC_ALL, HK_SYNC_ALL);
      if (init)
         brl_discard_init_record(cmd, pDependencyInfo, init_dst);
      return;
   }

   /* M5: a render pass instance that ended may still be continued by the
    * next one if this dependency is framebuffer-local. */
   if (unlikely(cmd->pass_merge.valid))
      hk_m5_pass_merge_barrier(cmd, pDependencyInfo);
   /* Likewise across command buffers, at submission (hk_render_pack.h): a
    * barrier before the first instance or after the last. */
   if (cmd->render_pack.head_open || cmd->render_pack.tail_open)
      hk_render_pack_note_barrier(cmd, pDependencyInfo);

   /* Host availability is recorded against the work already recorded. */
   if (cmd->state.deferred_clear_count &&
       hk_sync_dependency_has_host_access(pDependencyInfo))
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
      bool used_local_read = false;
      const uint32_t reasons =
         hk_classify_pass_barrier(cmd, pDependencyInfo, &used_local_read);
      if (reasons) {
         if (hk_cmd_buffer_end_graphics(cmd)) {
            cmd->state.barrier_stats.splits++;
            u_foreach_bit(r, reasons)
               cmd->state.barrier_stats.reason[r]++;
         }
      } else {
         if (cmd->current_cs.gfx)
            cmd->state.barrier_stats.kept++;
         if (used_local_read)
            cmd->state.gfx.render.kept_local_read_barrier = true;
      }
      /* Ending an empty render can replace its clear with a CDM job. Capture
       * the first scope only after that replacement has its physical stream. */
      hk_sync_pass_barrier(cmd, src_kinds, dst_kinds);
      return;
   }

   /* Metadata initialization of discarded compressed storage images: its
    * first scope is taken before this dependency is recorded. */
   uint8_t init_dst;
   const bool init = brl_discard_init_prepare(cmd, pDependencyInfo, &init_dst);

   uint8_t src_kinds, dst_kinds;
   hk_sync_dependency_kinds(pDependencyInfo, hk_sync_image_class(cmd),
                            &src_kinds, &dst_kinds);

   /* Compute work that stays in the open compute stream is ordered by the
    * stream itself; what it needs at a barrier is visibility. Every native
    * launch ends in a weak ordering barrier record, and the strong cache
    * flush is that same record with Apple's flags, upgraded in place. While
    * memory can be reached through two addresses, the flush needs memory
    * scope (hk_cdm_alias_scope()); where a later host import could add one,
    * the record is a patch site. */
   struct hk_cs *cs = cmd->current_cs.cs;
   const bool alias_scope = hk_cmd_alias_scope(cmd);
   const bool weak_only =
      HK_PERF(dev, WEAKBARRIER) ||
      (agx_chip_has_cap(dev->dev.chip, AGX_CAP_CDM_WAR_WEAK_BARRIER) &&
       !hk_dependency_needs_cache_flush(pDependencyInfo));
   if (cs && weak_only && !alias_scope &&
       src_kinds && !(src_kinds & ~(HK_SYNC_COMPUTE | HK_SYNC_XFER)) &&
       dst_kinds && !(dst_kinds & ~(HK_SYNC_COMPUTE | HK_SYNC_XFER)) &&
       !hk_sync_xfer_pending(&cmd->sync) &&
       !hk_sync_dependency_has_host_access(pDependencyInfo) &&
       (cs->current == cs->weak_barrier_end ||
        !hk_cdm_visibility_flush_needed(dev, cs, alias_scope))) {
      /* Keep the preceding launch's weak ordering record in place (do not let
       * the next application dispatch elide it) and skip the flags-5 cache
       * flush when the barrier is execution-only/WAR or when testing
       * ordering-only compute barriers. Where a later host import could alias,
       * the record is a patch site like any other. */
      cs->app_dispatch_end = NULL;
      hk_cdm_note_alias_site(cmd, cs);
   } else if (cs && agx_cdm_ordering_memory_scope(&dev->dev) &&
              src_kinds == HK_SYNC_COMPUTE && dst_kinds &&
              !(dst_kinds & ~(HK_SYNC_COMPUTE | HK_SYNC_XFER)) &&
              pDependencyInfo->imageMemoryBarrierCount == 0 &&
              !hk_sync_xfer_pending(&cmd->sync) &&
              !hk_sync_dependency_has_host_access(pDependencyInfo) &&
              cs->current == cs->weak_barrier_end) {
      /* Where every launch's ordering record has memory scope
       * (agx_cdm_ordering_memory_scope()), a dependency from compute to
       * compute or transfer work with no image barrier keeps the preceding
       * launch's record in place of the strong flush; the next application
       * dispatch may not drop it. Image barriers, transfer sources and host
       * access take the memory-scoped strong flush below. */
      cs->app_dispatch_end = NULL;
   } else if (cs && (src_kinds & (HK_SYNC_COMPUTE | HK_SYNC_XFER)) &&
              dst_kinds) {
      if (hk_cdm_visibility_flush_needed(dev, cs, alias_scope)) {
         hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
         hk_cdm_visibility_flush(dev, cs, alias_scope);
      }
      if (!alias_scope)
         hk_cdm_note_alias_site(cmd, cs);
   }

   hk_sync_barrier(cmd, src_kinds, dst_kinds);

   if (init)
      brl_discard_init_record(cmd, pDependencyInfo, init_dst);

   /* Host availability: where the strong CDM flush has a memory-scoped form,
    * that record writes the device's writes back for the host (a strong
    * flush alone leaves them in the GPU caches, and the end of a command
    * does not reliably write them back before completion is reported when
    * render work shares the queue). An open compute stream holding work
    * after the dependency's first scope takes the record, so the command
    * that ends with the scope's work ends with the flush. Otherwise the
    * host, which can only observe it at the end of the submission or
    * through an event, gets it there (hk_sync_defer_host_flush()) instead of
    * from a compute command of its own between the work around it. */
   if (hk_cdm_memory_scope_form(dev)) {
      struct hk_cs *flush = hk_sync_host_flush(cmd, pDependencyInfo);
      if (flush) {
         hk_ensure_cs_has_space(cmd, flush, AGX3_CDM_BARRIER_SIZE);
         hk_cdm_visibility_flush(dev, flush, true);
      }
   }

   /* The dependency this barrier recorded is the one a continued instance
    * applies to its stream (hk_sync_merge_rendering()); a stream it created
    * or extended still ends the candidacy there. */
   if (unlikely(cmd->pass_merge.valid))
      cmd->pass_merge.sync_epoch = cmd->sync.epoch;
   hk_render_pack_barrier_done(cmd);
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

static bool
hk_descriptor_sets_unchanged(const struct hk_descriptor_state *desc)
{
   if (desc->push_active || desc->buffer_active)
      return false;

   for (unsigned s = 0; s < HK_MAX_SETS; ++s) {
      const struct hk_descriptor_set *set = desc->sets[s];
      if (!set)
         continue;
      if (hk_descriptor_set_layout_allows_update_after_bind(set->layout))
         return false;
      if (p_atomic_read(&set->update_seq) != desc->set_update_seq[s])
         return false;
   }

   return true;
}

/* Ordinary graphics programs whose descriptor values are fetched through the
 * root/table in GPU memory qualify, as do CPU-decoded combined-sampler
 * programs while every bound set uses a static (non-update-after-bind) layout
 * and has not been modified since it was bound.
 */
static bool
hk_can_retain_descriptor_root(struct hk_cmd_buffer *cmd,
                              const struct hk_descriptor_state *desc)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   if (desc != &gfx->descriptors ||
       gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_MESH] ||
       gfx->shaders[MESA_SHADER_TASK])
      return false;

   struct hk_api_shader *vs = gfx->shaders[MESA_SHADER_VERTEX];
   struct hk_api_shader *fs = hk_bound_fs_obj(gfx);
   if (!vs || !fs)
      return false;

   const unsigned samples =
      cmd->vk.dynamic_graphics_state.ms.rasterization_samples;
   if (fs->g17p_sample_variants &&
       !(samples == 0 || samples == 1 || samples == 2 || samples == 4 ||
         (samples == 8 && fs->g17p_sample_variants == 4)))
      return false;

   const unsigned fs_idx =
      fs->g17p_sample_variants && samples > 1 ? util_logbase2(samples) : 0;
   if (!hk_fs_sample_variant_ready(fs, fs_idx))
      return false;

   const struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   const struct hk_shader *hw_fs = &fs->variants[fs_idx];
   if (!hw_vs || !hw_vs->compiled || !hw_fs->compiled)
      return false;

   bool uses_combined_sampler = hw_fs->info.agx3_combined_sampler_valid;
   if (!uses_combined_sampler) {
      /* If another published sample variant of the same API shader uses
       * CPU-decoded combined samplers, require unchanged static sets in case
       * rasterization_samples switches before the next descriptor bind. */
      for (unsigned i = 0; i < fs->g17p_sample_variants; ++i) {
         if (i == fs_idx || !hk_fs_sample_variant_ready(fs, i))
            continue;
         if (fs->variants[i].info.agx3_combined_sampler_valid) {
            uses_combined_sampler = true;
            break;
         }
      }
   }

   if (uses_combined_sampler)
      return hk_descriptor_sets_unchanged(desc);

   return true;
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

   bool non_dynbuf_root_changed = false;
   uint64_t dynbuf_changed_mask = 0;
   bool binding_changed = false;
   bool set_updated = false;
   uint32_t next_dyn_offset = 0;
   for (uint32_t i = 0; i < info->descriptorSetCount; ++i) {
      unsigned s = i + info->firstSet;
      VK_FROM_HANDLE(hk_descriptor_set, set, info->pDescriptorSets[i]);
      const uint32_t seq = set != NULL ? p_atomic_read(&set->update_seq) : 0;

      if (desc->sets[s] != set ||
          ((desc->push_active | desc->buffer_active) & BITFIELD_BIT(s))) {
         binding_changed = true;
         if (set != NULL) {
            non_dynbuf_root_changed |=
               desc->root.sets[s] != hk_descriptor_set_addr(set);
            desc->root.sets[s] = hk_descriptor_set_addr(set);
            desc->set_sizes[s] = set->size;
         } else {
            non_dynbuf_root_changed |= desc->root.sets[s] != 0;
            desc->root.sets[s] = 0;
            desc->set_sizes[s] = 0;
         }
         desc->sets[s] = set;
         desc->set_update_seq[s] = seq;
         desc->sets_dirty |= BITFIELD_BIT(s);

         /* Binding descriptors invalidates push descriptors */
      } else if (set != NULL && desc->set_update_seq[s] != seq) {
         set_updated = true;
         desc->set_update_seq[s] = seq;
         desc->sets_dirty |= BITFIELD_BIT(s);
      }
      desc->push_active &= ~BITFIELD_BIT(s);
      desc->push_dirty &= ~BITFIELD_BIT(s);
      desc->buffer_active &= ~BITFIELD_BIT(s);
      if (desc->push[s])
         hk_push_descriptor_set_reset_upload(desc->push[s]);

      non_dynbuf_root_changed |=
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
               const unsigned slot = dyn_buffer_start + j;
               const struct hk_buffer_address *old =
                  &desc->root.dynamic_buffers[slot];
               if (old->base_addr != addr.base_addr ||
                   old->size != addr.size ||
                   old->sparse_map_page != addr.sparse_map_page) {
                  dynbuf_changed_mask |= BITFIELD64_BIT(slot);
                  desc->root.dynamic_buffers[slot] = addr;
               }
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
   non_dynbuf_root_changed |= desc->dynamic_buffer_end < dyn_buffer_start;
   desc->dynamic_buffer_end = MAX2(desc->dynamic_buffer_end, dyn_buffer_start);

   hk_upload_stat(cmd, descriptor_binds, 1);
   const bool retain_root = hk_can_retain_descriptor_root(cmd, desc);
   if (!non_dynbuf_root_changed && !binding_changed && !set_updated &&
       retain_root) {
      if (!dynbuf_changed_mask) {
         hk_upload_stat(cmd, descriptor_noops, 1);
         return;
      }
      hk_root_mark_dynbuf_dirty(desc, dynbuf_changed_mask);
      desc->generation++;
      desc->non_push_generation++;
      return;
   }

   /* CPU-decoded descriptor programs retain their conservative invalidation:
    * their inputs can change in set memory without changing its address. */
   if (non_dynbuf_root_changed || dynbuf_changed_mask || set_updated ||
       !retain_root)
      hk_root_mark_dirty(desc);
   desc->generation++;
   desc->non_push_generation++;
   desc->non_dynbuf_generation++;
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
      cmd->used_rt = true;
      hk_bind_descriptor_sets(cmd, &cmd->rt.descriptors,
                              pBindDescriptorSetsInfo);
   }
}

static void
hk_push_constants(struct hk_cmd_buffer *cmd,
                  struct hk_descriptor_state *desc,
                  const VkPushConstantsInfoKHR *info)
{
   const uint32_t first_dw = info->offset >> 2;
   const uint32_t dw_count = info->size >> 2;
   if (first_dw < 64 && dw_count > 0) {
      const uint32_t clamped = MIN2(dw_count, 64 - first_dw);
      const uint64_t mask =
         clamped == 64 ? UINT64_MAX : ((UINT64_C(1) << clamped) - 1);
      desc->push_written_dwords |= mask << first_dw;
   }

   if (!cmd->in_meta && desc == &cmd->state.cs.descriptors &&
       info->offset < HK_COMPUTE_OPT_PUSH_BYTES) {
      unsigned size = MIN2(info->size, HK_COMPUTE_OPT_PUSH_BYTES - info->offset);
      desc->identity_copy4_push_mask |= BITFIELD64_MASK(size) << info->offset;
   }

   /* D3D translation layers re-push unchanged constants before most draws.
    * While the root is clean the uploaded root holds exactly these bytes, so
    * an identical push changes nothing; keep the root and every preparation
    * keyed to the current descriptor generation. */
   if (memcmp(desc->root.push + info->offset, info->pValues, info->size) == 0)
      return;

   memcpy(desc->root.push + info->offset, info->pValues, info->size);
   hk_root_mark_push_dirty(desc);
   /* non_push_generation intentionally does not advance on push constants
    * (push constants carry no tracked resource addresses; BDA transfers
    * already set sync.xfer_any). */
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

   if (pPushConstantsInfo->stageFlags & HK_RT_SHADER_STAGES) {
      cmd->used_rt = true;
      hk_push_constants(cmd, &cmd->rt.descriptors, pPushConstantsInfo);
   }
}

/* VK_EXT_descriptor_buffer */
VKAPI_ATTR void VKAPI_CALL
hk_CmdBindDescriptorBuffersEXT(
   VkCommandBuffer commandBuffer, uint32_t bufferCount,
   const VkDescriptorBufferBindingInfoEXT *pBindingInfos)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* The push descriptor buffer (bufferlessPushDescriptors is false) is never
    * read: push descriptors keep their own command-buffer storage. */
   assert(bufferCount <= HK_MAX_DESCRIPTOR_BUFFERS);
   for (uint32_t i = 0; i < bufferCount; i++)
      cmd->state.descriptor_buffers[i] = pBindingInfos[i].address;
}

/* Bind set s to `size` bytes of descriptors at `addr`. */
static void
hk_bind_buffer_set(struct hk_descriptor_state *desc,
                   const struct vk_pipeline_layout *pipeline_layout,
                   uint32_t s, uint64_t addr, uint32_t size)
{
   desc->root.sets[s] = addr;
   desc->set_sizes[s] = size;
   desc->sets[s] = NULL;
   desc->set_update_seq[s] = 0;
   desc->sets_dirty |= BITFIELD_BIT(s);
   desc->push_active &= ~BITFIELD_BIT(s);
   desc->push_dirty &= ~BITFIELD_BIT(s);
   desc->buffer_active |= BITFIELD_BIT(s);
   if (desc->push[s])
      hk_push_descriptor_set_reset_upload(desc->push[s]);

   /* Descriptor-buffer layouts have no dynamic buffers. */
   desc->root.set_dynamic_buffer_start[s] =
      pipeline_layout->dynamic_descriptor_offset[s];
}

static void
hk_set_descriptor_buffer_offsets(struct hk_cmd_buffer *cmd,
                                 struct hk_descriptor_state *desc,
                                 const VkSetDescriptorBufferOffsetsInfoEXT *info)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, info->layout);

   for (uint32_t i = 0; i < info->setCount; ++i) {
      const uint32_t s = info->firstSet + i;
      const struct hk_descriptor_set_layout *set_layout =
         pipeline_layout->set_layouts[s]
            ? vk_to_hk_descriptor_set_layout(pipeline_layout->set_layouts[s])
            : NULL;
      assert(info->pBufferIndices[i] < HK_MAX_DESCRIPTOR_BUFFERS);
      const uint64_t addr =
         cmd->state.descriptor_buffers[info->pBufferIndices[i]] +
         info->pOffsets[i];

      hk_bind_buffer_set(desc, pipeline_layout, s, addr,
                         set_layout ? set_layout->max_buffer_size : 0);
   }

   hk_root_mark_dirty(desc);
   desc->generation++;
   desc->non_push_generation++;
   desc->non_dynbuf_generation++;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdSetDescriptorBufferOffsets2EXT(
   VkCommandBuffer commandBuffer,
   const VkSetDescriptorBufferOffsetsInfoEXT *pInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pInfo->stageFlags & (VK_SHADER_STAGE_ALL_GRAPHICS |
                            VK_SHADER_STAGE_TASK_BIT_EXT |
                            VK_SHADER_STAGE_MESH_BIT_EXT)) {
      hk_m5_pass_merge_note_set_bind(cmd);
      hk_set_descriptor_buffer_offsets(cmd, &cmd->state.gfx.descriptors, pInfo);
   }

   if (pInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
      hk_set_descriptor_buffer_offsets(cmd, &cmd->state.cs.descriptors, pInfo);

   if (pInfo->stageFlags & HK_RT_SHADER_STAGES) {
      cmd->used_rt = true;
      hk_set_descriptor_buffer_offsets(cmd, &cmd->rt.descriptors, pInfo);
   }
}

static void
hk_bind_embedded_samplers(struct hk_cmd_buffer *cmd,
                          struct hk_descriptor_state *desc,
                          const VkBindDescriptorBufferEmbeddedSamplersInfoEXT *info)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, info->layout);
   const struct hk_descriptor_set_layout *set_layout =
      vk_to_hk_descriptor_set_layout(pipeline_layout->set_layouts[info->set]);
   assert(set_layout->embedded_samplers != NULL);

   /* Uploaded per bind, so the descriptors outlive the layout for as long as
    * the command buffer needs them. */
   const uint32_t size = MAX2(set_layout->non_variable_descriptor_buffer_size,
                              HK_MIN_UBO_ALIGNMENT);
   const uint64_t addr = hk_pool_upload(cmd, set_layout->embedded_samplers,
                                        size, HK_MIN_UBO_ALIGNMENT);

   hk_bind_buffer_set(desc, pipeline_layout, info->set, addr, size);
   hk_root_mark_dirty(desc);
   desc->generation++;
   desc->non_push_generation++;
   desc->non_dynbuf_generation++;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindDescriptorBufferEmbeddedSamplers2EXT(
   VkCommandBuffer commandBuffer,
   const VkBindDescriptorBufferEmbeddedSamplersInfoEXT *pInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pInfo->stageFlags & (VK_SHADER_STAGE_ALL_GRAPHICS |
                            VK_SHADER_STAGE_TASK_BIT_EXT |
                            VK_SHADER_STAGE_MESH_BIT_EXT)) {
      hk_m5_pass_merge_note_set_bind(cmd);
      hk_bind_embedded_samplers(cmd, &cmd->state.gfx.descriptors, pInfo);
   }

   if (pInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
      hk_bind_embedded_samplers(cmd, &cmd->state.cs.descriptors, pInfo);

   if (pInfo->stageFlags & HK_RT_SHADER_STAGES) {
      cmd->used_rt = true;
      hk_bind_embedded_samplers(cmd, &cmd->rt.descriptors, pInfo);
   }
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
   desc->set_update_seq[set] = 0;
   desc->buffer_active &= ~BITFIELD_BIT(set);
   desc->push_dirty |= BITFIELD_BIT(set);
   desc->push_active |= BITFIELD_BIT(set);

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
   if (!agx_chip_native_or_m5(chip))
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
   if (!agx_chip_native_or_m5(chip))
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

   if (!hk_push_descriptor_set_update(push_set, set_layout,
                                      info->descriptorWriteCount,
                                      info->pDescriptorWrites))
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
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
      cmd->used_rt = true;
      hk_push_descriptor_set(cmd, &cmd->rt.descriptors,
                             pPushDescriptorSetInfo);
   }
}

void
hk_cmd_buffer_flush_push_descriptors(struct hk_cmd_buffer *cmd,
                                     struct hk_descriptor_state *desc)
{
   bool root_changed = false;

   u_foreach_bit(set_idx, desc->push_dirty) {
      struct hk_push_descriptor_set *push_set = desc->push[set_idx];
      /* Only the bytes the pushed layout uses; the rest is never read.
       * hk_push_descriptor_set_begin() keeps that within the storage. */
      STATIC_ASSERT(HK_MIN_UBO_ALIGNMENT <= HK_PUSH_DESCRIPTOR_SET_SIZE);
      const uint32_t size = MAX2(push_set->size, HK_MIN_UBO_ALIGNMENT);
      uint64_t push_set_addr;

      /* last_cpu_ptr is only populated when the backing command-pool allocation
       * is mapped CPU write-back (AGX_BO_WRITEBACK); write-combined allocations
       * leave it NULL so this path never reads back WC pages or keeps a
       * duplicate 10 KiB shadow copy. */
      if (push_set->last_gpu_addr && push_set->last_cpu_ptr &&
          push_set->last_size == size &&
          memcmp(push_set->last_cpu_ptr, push_set->data, size) == 0) {
         push_set_addr = push_set->last_gpu_addr;
      } else {
         bool cpu_cached = false;
         struct agx_ptr uploaded =
            hk_pool_alloc_cached(cmd, size, HK_MIN_UBO_ALIGNMENT, &cpu_cached);
         if (likely(uploaded.cpu != NULL)) {
            memcpy(uploaded.cpu, push_set->data, size);
            if (cpu_cached) {
               push_set->last_gpu_addr = uploaded.gpu;
               push_set->last_cpu_ptr = uploaded.cpu;
               push_set->last_size = size;
            } else {
               hk_push_descriptor_set_reset_upload(push_set);
            }
            push_set_addr = uploaded.gpu;
         } else {
            hk_push_descriptor_set_reset_upload(push_set);
            push_set_addr = 0;
         }
      }

      if (desc->root.sets[set_idx] != push_set_addr ||
          desc->set_sizes[set_idx] != size) {
         desc->root.sets[set_idx] = push_set_addr;
         desc->set_sizes[set_idx] = size;
         root_changed = true;
      }
   }

   if (root_changed) {
      hk_root_mark_dirty(desc);
      desc->generation++;
      desc->non_push_generation++;
      desc->non_dynbuf_generation++;
   }
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

   if (!hk_push_descriptor_set_update_template(
          push_set, set_layout, template,
          pPushDescriptorSetWithTemplateInfo->pData))
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
}

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
      cmd->g17p_gfx_caches_used = true;
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

static inline const struct hk_shader *
hk_bound_hw_vs_or_null(struct hk_graphics_state *gfx)
{
   return (gfx->shaders[MESA_SHADER_VERTEX] || gfx->shaders[MESA_SHADER_MESH])
             ? hk_bound_hw_vs(gfx)
             : NULL;
}

static inline bool
hk_g17p_basic_fragment_layout(const struct hk_device *dev,
                              const struct hk_cmd_buffer *cmd)
{
   const struct agx_tilebuffer_layout *tib = &cmd->state.gfx.render.tilebuffer;
   if (likely(tib->nr_samples <= 1))
      return cmd->state.gfx.render.g17p_rt0_layout_supported;
   if (agx_chip_hal300_path(dev->dev.chip)) {
      return (tib->nr_samples == 2 || tib->nr_samples == 4 ||
              (agx_chip_g17_m5(dev->dev.chip) && tib->nr_samples == 8)) &&
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
 * The HAL200 varying table uses the same CF entry encoding for
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
             (b->perspective && !has_rhw))
            return false;
         ++user_bindings;
         if (!hk_g17p_user_varying_binding_fits(
                b->offset, b->count, vs->info.uvs.components[b->slot],
                gfx->linked_varyings.slots[b->slot], vs->info.uvs.size))
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
   const struct hk_shader *vs = hk_bound_hw_vs_or_null(gfx);
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
   const uint32_t max_scratch_B = hk_graphics_max_scratch_B(
      &hk_cmd_buffer_device(cmd)->dev, s->b.info.stage);
   if (!linked || !linked->b.bo || !linked->b.bo->va || linked->b.size == 0 ||
       linked->b.size > UINT32_MAX || (s->b.info.has_preamble && !native_preamble) ||
       MAX2(s->b.info.scratch_size, s->b.info.agx3_spectpr_scratch_size) >
          max_scratch_B ||
       (s->b.info.preamble_scratch_size != 0 && !native_preamble) ||
       MAX2(linked->b.scratch_size, linked->b.agx3_spectpr_scratch_size) >
          max_scratch_B ||
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
      /* The narrow RT0 colour-link layout is a G17G witness-profile guard.
       * The G18G general path binds any tile layout through the general
       * graphics core and the separately validated EEOT, so it is not a
       * composer constraint there. */
      const struct hk_device *dev = hk_cmd_buffer_device(cmd);
      const bool layout =
         dev->dev.chip == AGX_CHIP_G18G ||
         hk_g17p_basic_fragment_layout(dev, cmd);

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
int8_t
hk_g17p_dma_set_index(const struct hk_shader *s)
{
   static_assert(HK_MAX_SETS <= INT8_MAX,
                 "descriptor set index fits in int8_t");
   if (!s)
      return -1;

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
      return (int8_t)((pointer - root_uniform - 4) / 4);
   }
   return -1;
}

uint64_t
hk_g17p_dma_set_base(const struct hk_cmd_buffer *cmd,
                     const struct hk_shader *s)
{
   return hk_g17p_dma_set_base_from_index(cmd, hk_g17p_dma_set_index(s));
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
   } else if (desc->buffer_active & BITFIELD_BIT(set)) {
      /* Descriptor buffers are never static (hk_static_set_mask()). */
      return false;
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

      /* A root-relative span follows the root (or root.push[] when wholly
       * inside the push-constant window), a set-pointer span the set's
       * address (hk_g17p_dma_set_base(), which every reuse of a published
       * launcher still compares). */
      if (set < 0) {
         const size_t push_lo = hk_root_descriptor_offset(push);
         const size_t push_hi = push_lo + HK_MAX_PUSH_SIZE;
         const bool in_push =
            span->offset_B >= push_lo && span->offset_B <= push_hi &&
            2u * span->halfs <= push_hi - span->offset_B;
         sources[*count_io] = in_push ? HK_G17P_RANGE_SOURCE_PUSH
                                      : HK_G17P_RANGE_SOURCE_ROOT;
      } else {
         sources[*count_io] = HK_G17P_RANGE_SOURCE_SET;
      }
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
bool
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
          agx_chip_g17_m5(dev->dev.chip) ||
          !hk_gen1_upload_fragment_query_state(cmd, s->info.stage != MESA_SHADER_FRAGMENT, &ranges[count]))
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

         sources[count] = HK_G17P_RANGE_SOURCE_ATTRIB_BASE;
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = uniform_base + AGX_ABI_VUNI_VBO_BASE(0),
            .size_halfs = 4 * nr_attribs,
            .address =
               root_ptr + hk_root_descriptor_offset(draw.attrib_base),
         };
         sources[count] = HK_G17P_RANGE_SOURCE_ATTRIB_CLAMP;
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
      if (!set && descriptors->push[0] && descriptors->root.sets[0] &&
          !(descriptors->buffer_active & BITFIELD_BIT(0))) {
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
      sampler_index = hk_sampler_heap_index(
         set ? set->layout->native_sampler_indices
             : agx_chip_agx3_path(dev->dev.chip),
         sampler_index);

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

      /* On HAL200 compiler bindpoints 0x31/0x35 are the texture and sampler
       * table addresses, at driver qwords +0xc0/+0xe0. Render state carries
       * the raw texture/sampler table VAs there; ESL copies them to uniform
       * dwords 0..3. Keep this upload immutable for the command-buffer
       * lifetime, just like the per-draw graphics state. Both shader indices
       * are zero.
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
            cmd->g17p_gfx_caches_used = true;
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
           s->b.info.stage != MESA_SHADER_MESH &&
           !(agx_chip_g17_m5(dev->dev.chip) &&
             s->b.info.stage == MESA_SHADER_TASK)) ||
          !dev->sampled_images.base_va || !dev->samplers.table.bo ||
          !dev->rodata.indexed_heap_pointers)
         return hk_g17p_graphics_ranges_refused(range_count, count);
      const uint64_t pointers_va = dev->rodata.indexed_heap_pointers;
      sources[count] = HK_G17P_RANGE_SOURCE_CONSTANT;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = texture_uniform_base, .size_halfs = 8, .address = pointers_va,
      };
   }

   bool effective_uses_root = s->info.agx3_uses_root;
   uint8_t live_set_count = s->info.set_count;
   if (s->info.stage == MESA_SHADER_VERTEX &&
       s->b.info.agx3_root_uniform_usage_valid) {
      const bool hoisted = cmd->state.gfx.g17p_hoisted_preamble_va != 0;
      effective_uses_root =
         s->b.info.agx3_main_uses_root_uniform ||
         (!hoisted && s->b.info.has_preamble &&
          s->b.info.agx3_preamble_uses_root_uniform);
      /* A vertex shader reading only push constants through uniform DMA
       * (draw-ID indexed tiles, per-draw transforms) then has no root range
       * at all, and its launches become push-only
       * (hk_agx3_try_push_only_vs_program()). */
      live_set_count = hk_agx3_g17p_graphics_live_set_count(
         s->b.info.agx3_uniform_halfs,
         ARRAY_SIZE(s->b.info.agx3_uniform_halfs),
         s->b.info.agx3_uniform_halfs_valid, root_uniform, s->info.set_count);
   }

   if (hk_agx3_g17p_graphics_needs_root(
          effective_uses_root, linked->g17p_prolog_uses_root,
          live_set_count)) {
      if (!root_ptr)
         return hk_g17p_graphics_ranges_refused(range_count, count);

      if (!hk_agx3_g17p_root_range(root_uniform, s->info.set_count, root_ptr,
                                   hk_shader_root_size(s), &ranges[count]))
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

   if (s->info.stage == MESA_SHADER_VERTEX &&
       cmd->state.gfx.g17p_hoisted_preamble_va) {
      if (count >= HK_AGX3_G17P_GRAPHICS_MAX_RANGES ||
          !s->b.info.agx3_preamble_out_size_halfs)
         return hk_g17p_graphics_ranges_refused(range_count, count);
      sources[count] = HK_G17P_RANGE_SOURCE_PREAMBLE_OUT;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = s->b.info.agx3_preamble_out_start_half,
         .size_halfs = s->b.info.agx3_preamble_out_size_halfs,
         .address = cmd->state.gfx.g17p_hoisted_preamble_va,
      };
   }

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
   bool has_constant_program;
   uint16_t length;
   struct hk_g17p_patch_plan plan;
   uint8_t bytes[];
};

static inline bool
hk_g17p_gfx_loader_matches(const struct hk_g17p_gfx_loader *l,
                           const struct hk_shader *s,
                           const struct agx_ioto_g17p_imageblock *imageblock,
                           const struct agx3_ioto_uniform_range *ranges,
                           size_t range_count, bool has_constant_program)
{
   if (l->s != s || l->range_count != range_count ||
       l->has_constant_program != has_constant_program ||
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
   l->has_constant_program = desc->constant_program.code_size != 0;
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
   struct hk_ioto_arena *arena = &cmd->ioto;
   const uint32_t len = l->length;
   const uint32_t mask = HK_IOTO_PROGRAM_ALIGNMENT - 1;

   if (likely(len != 0 && arena->map && arena->cursor <= arena->capacity &&
              arena->cursor <= UINT32_MAX - mask)) {
      const uint32_t offset = (arena->cursor + mask) & ~mask;
      if (likely(offset <= arena->capacity && len <= arena->capacity - offset)) {
         uint8_t *dst = arena->map + offset;
         const uint64_t va = arena->base + offset;
         arena->cursor = offset + len;
         arena->programs++;
         arena->program_bytes += len;
         if (unlikely(!hk_g17p_cached_cmd_memory(&hk_cmd_buffer_device(cmd)->dev)))
            agx_bo_assert_range(arena->bo, offset, len);
         memcpy(dst, l->bytes, len);
         hk_g17p_patch_plan_apply(&l->plan, ranges, dst);
         return va;
      }
   }

   struct hk_ioto_reservation r;
   if (!hk_cmd_reserve_ioto_program_slow(cmd, len, "G17P graphics loader", &r))
      return 0;

   memcpy(r.map, l->bytes, len);
   hk_g17p_patch_plan_apply(&l->plan, ranges, r.map);

   return hk_cmd_publish_ioto_program(cmd, &r, len);
}

static bool
hk_g17p_graphics_program_desc_init(
   struct hk_cmd_buffer *cmd, struct hk_shader *s,
   struct hk_linked_shader *linked, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, struct hk_agx3_g17p_graphics_program_desc *desc);

static inline bool
hk_g17p_gfx_loader_ranges_equal(const struct agx3_ioto_uniform_range *a,
                                const struct agx3_ioto_uniform_range *b,
                                size_t range_count)
{
   for (size_t i = 0; i < range_count; ++i) {
      if (a[i].address != b[i].address ||
          a[i].start_halfs != b[i].start_halfs ||
          a[i].size_halfs != b[i].size_halfs)
         return false;
   }
   return true;
}

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
   /* The program is a function of the loader and the range addresses.
    * Compare fields explicitly rather than with memcmp():
    * struct agx3_ioto_uniform_range has 4 bytes of internal padding that
    * compilers leave indeterminate on struct assignment. */
   if (cached->loader == l && cached->loader_linked == linked &&
       cached->loader_program &&
       hk_g17p_gfx_loader_ranges_equal(cached->loader_ranges, ranges,
                                       range_count))
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
      flags |= (gfx->draw_params ? 2 : 0) | (gfx->draw_id_ptr ? 4 : 0) |
               (gfx->g17p_hoisted_preamble_va ? 32 : 0);
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
   launch->bypass_loader = NULL;
   launch->root_bypass_ok = false;
   launch->has_general_root = false;
   launch->push_min_B = 0;
   launch->push_size_B = 0;
   if (count > ARRAY_SIZE(launch->source))
      return;

   bool has_push = false;
   bool has_general_root = false;
   uint16_t push_min_B = HK_MAX_PUSH_SIZE;
   uint16_t push_max_B = 0;

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
         /* The root/set table header range at offset 0 only exposes the
          * root_desc_addr pointer when the main body, preamble, or prolog
          * reads it; otherwise it only supplies root.sets[], which never
          * change on push, dynamic-buffer, or VB base/clamp updates. */
         if (offset != 0 || !s->b.info.agx3_root_uniform_usage_valid ||
             s->b.info.agx3_main_uses_root_uniform ||
             s->b.info.has_preamble || linked->g17p_prolog_uses_root)
            has_general_root = true;
         break;
      case HK_G17P_RANGE_SOURCE_ATTRIB_BASE:
         offset = ranges[i].address - root;
         if (offset != hk_root_descriptor_offset(draw.attrib_base) ||
             2u * ranges[i].size_halfs >
                sizeof(gfx->descriptors.root.draw.attrib_base))
            return;
         break;
      case HK_G17P_RANGE_SOURCE_ATTRIB_CLAMP:
         offset = ranges[i].address - root;
         if (offset != hk_root_descriptor_offset(draw.attrib_clamps) ||
             2u * ranges[i].size_halfs >
                sizeof(gfx->descriptors.root.draw.attrib_clamps))
            return;
         break;
      case HK_G17P_RANGE_SOURCE_PUSH: {
         const uint64_t push_off = hk_root_descriptor_offset(push);
         offset = ranges[i].address - root;
         if (offset < push_off ||
             offset - push_off + 2u * ranges[i].size_halfs > HK_MAX_PUSH_SIZE)
            return;
         const uint16_t rel = (uint16_t)(offset - push_off);
         const uint16_t end = (uint16_t)(rel + 2u * ranges[i].size_halfs);
         push_min_B = MIN2(push_min_B, rel);
         push_max_B = MAX2(push_max_B, end);
         has_push = true;
         break;
      }
      case HK_G17P_RANGE_SOURCE_DRAW_PARAMS:
      case HK_G17P_RANGE_SOURCE_DRAW_ID:
      case HK_G17P_RANGE_SOURCE_PREAMBLE_OUT:
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
   launch->has_general_root = has_general_root;
   launch->root_bypass_ok =
      s->info.stage == MESA_SHADER_VERTEX &&
      (!has_general_root || !(s->info.agx3_root_reads & HK_ROOT_READS_ADDRESS));
   if (has_push && push_max_B > push_min_B) {
      launch->push_min_B = push_min_B;
      launch->push_size_B = (uint16_t)(push_max_B - push_min_B);
   }
   launch->valid = true;
}

/* Whether the recorded launch applies to this draw, admission included: the
 * admission of a vertex program depends only on the program, that of a
 * fragment program also on the spill state (in the flags), the native RT0
 * layout and the varying-coefficient inputs (hk_g17p_basic_graphics_
 * program_supported()). */
static inline bool
hk_g17p_launch_matches(const struct hk_device *dev,
                       const struct hk_cmd_buffer *cmd,
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
          (hk_g17p_basic_fragment_layout(dev, cmd) && hw_vs &&
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
      case HK_G17P_RANGE_SOURCE_PUSH:
      case HK_G17P_RANGE_SOURCE_ATTRIB_BASE:
      case HK_G17P_RANGE_SOURCE_ATTRIB_CLAMP:
         address = root + launch->offset[i];
         break;
      case HK_G17P_RANGE_SOURCE_DRAW_PARAMS:
         address = gfx->draw_params;
         break;
      case HK_G17P_RANGE_SOURCE_DRAW_ID:
         address = gfx->draw_id_ptr;
         break;
      case HK_G17P_RANGE_SOURCE_PREAMBLE_OUT:
         address = gfx->g17p_hoisted_preamble_va;
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

   const bool has_constant_program =
      s->b.info.has_preamble &&
      !(s->info.stage == MESA_SHADER_VERTEX &&
        cmd->state.gfx.g17p_hoisted_preamble_va != 0);
   const struct hk_g17p_gfx_loader *l = cached->loader;
   unsigned shapes = 0;
   if (!l || cached->loader_linked != linked ||
       !hk_g17p_gfx_loader_matches(l, s, &imageblock, ranges, range_count,
                                   has_constant_program) ||
       !hk_g17p_patch_plan_fits(&l->plan, ranges)) {
      l = NULL;
      for (struct hk_g17p_gfx_loader *it =
              p_atomic_read(&linked->g17p_gfx_loaders);
           it; it = it->next) {
         if (!hk_g17p_gfx_loader_matches(it, s, &imageblock, ranges,
                                         range_count, has_constant_program))
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
      .usc_clusters =
         hk_cmd_buffer_device(cmd)->dev.params.num_clusters_total,
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

   const bool use_cprog =
      s->b.info.has_preamble &&
      !(s->info.stage == MESA_SHADER_VERTEX &&
        cmd->state.gfx.g17p_hoisted_preamble_va != 0);

   if (use_cprog &&
       (!s->bo || !s->bo->va || s->code_size != s->b.info.binary_size ||
        !hk_agx3_g17p_constant_program_from_binary(
           s->code_ptr, s->code_size, s->b.info.preamble_offset,
           s->b.info.preamble_size, hk_shader_code_va(s),
           s->b.info.binary_size, &desc->constant_program))) {
      mesa_loge_once("G17P graphics constant-program extent refused");
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   if (use_cprog) {
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
          ? !linked->b.g17p_compiler_d_valid
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
      /* Promoted constants, as for the other graphics programs
       * (hk_g17p_graphics_uniform_ranges()). */
      const struct agx_rodata *rodata = &s->b.info.rodata;
      if (dev->dev.chip == AGX_CHIP_G17P && rodata->size_16 != 0 &&
          (!s->bo || !s->bo->va ||
           !hk_agx3_g17p_rodata_range(rodata->base_uniform, rodata->size_16,
                                      hk_shader_code_va(s), rodata->offset,
                                      s->b.info.binary_size, &ranges[count++])))
         goto refused;
   } else if (!hk_g17p_graphics_uniform_ranges(cmd, s, linked,
                                               cmd->state.gfx.root, ranges,
                                               &count, NULL)) {
      goto refused;
   }
   struct agx3_g17p_graphics_program_desc d = {
      .usc_clusters = dev->dev.params.num_clusters_total,
      .shader_va = linked->b.bo->va->addr + linked->b.entry_offset,
      .uniform_ranges = ranges, .uniform_range_count = count,
      .shader_binary_size = linked->b.size,
      /* The thread-private frame also holds returned atomic words
       * (agx3_spectpr_scratch_size), as for the vertex and fragment programs. */
      .scratch_size = dev->dev.chip == AGX_CHIP_G17P
                         ? MAX3(linked->b.scratch_size, s->b.info.scratch_size,
                                linked->b.agx3_spectpr_scratch_size)
                         : MAX2(linked->b.scratch_size,
                                linked->b.agx3_spectpr_scratch_size),
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

/* Read the uncached mappings only for this validation oracle
 * (ASAHI_MESA_DEBUG=verify). Keep its temporary snapshot and checks out of
 * the normal recording path. */
__attribute__((cold, noinline)) static void
hk_agx3_verify_fragment_root_shadow(const struct hk_graphics_state *gfx)
{
   struct hk_root_fragment_snapshot reference;
   if (gfx->root_cpu) {
      hk_root_fragment_snapshot_store(&reference, gfx->root_cpu,
                                      gfx->root_size);
      if (gfx->root_fragment_snapshot_cpu != gfx->root_cpu ||
          memcmp(&reference, &gfx->root_fragment_snapshot, sizeof(reference))) {
         mesa_loge("AGX3 current root snapshot mismatch");
         abort();
      }
   }
   if (gfx->fs_root_cpu) {
      hk_root_fragment_snapshot_store(&reference, gfx->fs_root_cpu,
                                      gfx->fs_root_size);
      if (gfx->fs_root_fragment_snapshot_cpu != gfx->fs_root_cpu ||
          memcmp(&reference, &gfx->fs_root_fragment_snapshot,
                 sizeof(reference))) {
         mesa_loge("AGX3 prior root snapshot mismatch");
         abort();
      }
   }
}

/* The root the AGX3 fragment program of s follows. A root republished only
 * for bytes the fragment shader cannot read -- a per-draw dynamic offset of a
 * vertex-stage uniform buffer, a vertex-only push constant update, a vertex
 * buffer bind -- keeps the program the stage last published, and with it the
 * whole PPP state: that root stays valid for the command buffer and holds the
 * same bytes wherever the shader reads. */
static uint64_t
hk_agx3_fragment_root(struct hk_cmd_buffer *cmd, const struct hk_shader *s)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   const bool verify = unlikely(dev->dev.debug & AGX_DBG_VERIFY);

   if (verify)
      hk_agx3_verify_fragment_root_shadow(gfx);

   const bool use_shadow =
      verify || !gfx->root_cpu_cached || !gfx->fs_root_cpu_cached;
   return hk_agx3_fragment_root_select(gfx, s->info.agx3_root_reads,
                                       s->info.agx3_dynbuf_reads,
                                       s->root_descriptor_size, true,
                                       use_shadow);
}

static inline uint64_t
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

bool
hk_agx3_multi_draw_ioto_prepare(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                                struct hk_linked_shader *linked,
                                bool uses_base_param, bool uses_draw_id,
                                struct hk_agx3_multi_draw_ioto_tpl *tpl)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_g17p_graphics_ioto_cache_entry *cached = &gfx->g17p_ioto[0];
   const struct hk_g17p_launch *launch = hk_g17p_launch_slot(cmd, 0, linked);

   const uint8_t expected_flags =
      (gfx->root ? 1 : 0) | (uses_base_param ? 2 : 0) |
      (uses_draw_id ? 4 : 0) | (gfx->g17p_hoisted_preamble_va ? 32 : 0);

   if (s->info.stage != MESA_SHADER_VERTEX || s->info.agx3_gen1_fragment_rq ||
       !launch->valid || launch->s != s || launch->linked != linked ||
       !launch->loader || cached->loader != launch->loader ||
       cached->loader_linked != linked || !cached->loader_program ||
       launch->flags != expected_flags)
      return false;

   const struct hk_g17p_gfx_loader *l = launch->loader;
   if (!l->length || l->length > sizeof(tpl->bytes) ||
       launch->count != l->plan.range_count ||
       launch->count > HK_G17P_PATCH_MAX_RANGES)
      return false;

   uint32_t draw_params_mask = 0;
   uint32_t draw_id_mask = 0;
   for (unsigned i = 0; i < launch->count; ++i) {
      if (launch->source[i] == HK_G17P_RANGE_SOURCE_DRAW_PARAMS)
         draw_params_mask |= BITFIELD_BIT(i);
      else if (launch->source[i] == HK_G17P_RANGE_SOURCE_DRAW_ID)
         draw_id_mask |= BITFIELD_BIT(i);
   }
   const uint32_t dyn_mask = draw_params_mask | draw_id_mask;

   /* Static ranges stay at cached->loader_ranges; require that no dynamic
    * range shares a cluster with any other range so moving DRAW_PARAMS or
    * DRAW_ID cannot invalidate cluster sharing. */
   if (l->plan.shared) {
      for (unsigned i = 0; i < l->plan.range_count; ++i) {
         const unsigned leader = l->plan.leader[i];
         if (dyn_mask & BITFIELD_BIT(i)) {
            if (leader != i)
               return false;
         } else if (dyn_mask & BITFIELD_BIT(leader)) {
            return false;
         }
      }
   }

   if (!hk_g17p_patch_plan_fits(&l->plan, cached->loader_ranges))
      return false;

   /* Bake all static relocations into the template once. */
   memcpy(tpl->bytes, l->bytes, l->length);
   hk_g17p_patch_plan_apply(&l->plan, cached->loader_ranges, tpl->bytes);

   unsigned dyn_reloc_count = 0;
   for (unsigned r = 0; r < l->plan.reloc_count; ++r) {
      const struct hk_g17p_patch_reloc *rel = &l->plan.relocs[r];
      if (!(dyn_mask & BITFIELD_BIT(rel->range)))
         continue;
      tpl->dyn_relocs[dyn_reloc_count++] = (struct hk_agx3_multi_draw_ioto_reloc){
         .dest = {rel->dest[0], rel->dest[1]},
         .delta = rel->delta,
         .offset = rel->offset,
         .short_mask = rel->short_mask,
         .is_draw_id = (draw_id_mask & BITFIELD_BIT(rel->range)) != 0,
      };
   }

   if ((dyn_mask != 0) != (dyn_reloc_count != 0))
      return false;

   tpl->length = l->length;
   tpl->cached_cmd_mem = hk_g17p_cached_cmd_memory(&dev->dev);
   tpl->dyn_reloc_count = dyn_reloc_count;
   tpl->draw_params_mask = draw_params_mask;
   tpl->draw_id_mask = draw_id_mask;
   return true;
}

void
hk_agx3_multi_draw_ioto_finish(struct hk_cmd_buffer *cmd,
                               const struct hk_agx3_multi_draw_ioto_tpl *tpl,
                               uint64_t pipeline_program, uint64_t draw_params,
                               uint64_t draw_id_ptr)
{
   struct hk_g17p_graphics_ioto_cache_entry *cached =
      &cmd->state.gfx.g17p_ioto[0];
   u_foreach_bit(r, tpl->draw_params_mask)
      cached->loader_ranges[r].address = draw_params;
   u_foreach_bit(r, tpl->draw_id_mask)
      cached->loader_ranges[r].address = draw_id_ptr;
   cached->loader_program = pipeline_program;
}

static bool
hk_agx3_indirect_loader_fits(const struct hk_g17p_gfx_loader *l,
                             const struct hk_g17p_launch *launch,
                             const struct agx3_ioto_uniform_range *first,
                             const struct agx3_ioto_uniform_range *last)
{
   if (!l || l->length == 0 || l->length > AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE)
      return false;

   if (!hk_g17p_patch_plan_fits(&l->plan, first) ||
       !hk_g17p_patch_plan_fits(&l->plan, last))
      return false;

   unsigned varying = 0;
   for (unsigned r = 0; r < l->plan.reloc_count; ++r) {
      const struct hk_g17p_patch_reloc *rel = &l->plan.relocs[r];
      const uint8_t src = launch->source[rel->range];
      if (src != HK_G17P_RANGE_SOURCE_DRAW_PARAMS &&
          src != HK_G17P_RANGE_SOURCE_DRAW_ID)
         continue;

      if ((rel->offset & 1u) != 0)
         return false;

      const uint64_t base_first = first[rel->range].address + rel->delta;
      const uint64_t base_last = last[rel->range].address + rel->delta;
      if (base_last < base_first)
         return false;

      /* A 2-byte low LDIMM requires every intermediate draw's low 32 bits to
       * remain < 128; forbid crossing a 4 GiB wrap boundary. */
      if ((rel->short_mask & 1u) && (base_first >> 32) != (base_last >> 32))
         return false;

      if (++varying > LIBAGX_INDIRECT_VDM_MAX_RELOCS)
         return false;
   }

   return true;
}

bool
hk_agx3_indirect_ioto_prepare(
   struct hk_cmd_buffer *cmd, struct hk_shader *s,
   struct hk_linked_shader *linked, uint64_t in_va, uint32_t stride_B,
   uint32_t param_offset_B, uint64_t draw_ids_va, uint32_t max_draw_count,
   uint64_t *ioto_template_gpu, uint32_t *ioto_copy_dwords,
   uint32_t *ioto_stride_B, struct libagx_indirect_ioto_reloc *relocs,
   uint32_t *reloc_count)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_g17p_graphics_ioto_cache_entry *cached = &gfx->g17p_ioto[0];
   const struct hk_g17p_launch *launch = hk_g17p_launch_slot(cmd, 0, linked);

   if (!max_draw_count || s->info.stage != MESA_SHADER_VERTEX ||
       s->info.agx3_gen1_fragment_rq || !launch->valid || launch->s != s ||
       launch->linked != linked || !launch->loader ||
       cached->loader != launch->loader || cached->loader_linked != linked ||
       !cached->loader_program ||
       launch->flags != hk_g17p_launch_flags(cmd, s))
      return false;

   const unsigned count = launch->count;
   struct agx3_ioto_uniform_range first[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   struct agx3_ioto_uniform_range last[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   memcpy(first, cached->loader_ranges, count * sizeof(first[0]));
   memcpy(last, cached->loader_ranges, count * sizeof(last[0]));

   const uint64_t last_idx = max_draw_count - 1;
   for (unsigned i = 0; i < count; ++i) {
      if (launch->source[i] == HK_G17P_RANGE_SOURCE_DRAW_PARAMS) {
         first[i].address = in_va + param_offset_B;
         last[i].address = in_va + last_idx * stride_B + param_offset_B;
      } else if (launch->source[i] == HK_G17P_RANGE_SOURCE_DRAW_ID) {
         if (!draw_ids_va)
            return false;
         first[i].address = draw_ids_va;
         last[i].address = draw_ids_va + last_idx * sizeof(uint32_t);
      }
   }

   const struct hk_g17p_gfx_loader *l = launch->loader;
   if (!hk_agx3_indirect_loader_fits(l, launch, first, last)) {
      const struct agx_tilebuffer_layout *tib = &gfx->render.tilebuffer;
      const struct agx_ioto_g17p_imageblock imageblock = {
         .sample_size_B = tib->sample_size_B,
         .nr_samples = tib->nr_samples,
         .tile_size_px = tib->tile_size,
      };

      const bool has_constant_program =
         s->b.info.has_preamble && gfx->g17p_hoisted_preamble_va == 0;

      l = NULL;
      for (struct hk_g17p_gfx_loader *it =
              p_atomic_read(&linked->g17p_gfx_loaders);
           it; it = it->next) {
         if (hk_g17p_gfx_loader_matches(it, s, &imageblock, first, count,
                                        has_constant_program) &&
             hk_agx3_indirect_loader_fits(it, launch, first, last)) {
            l = it;
            break;
         }
      }

      if (!l) {
         struct hk_agx3_g17p_graphics_program_desc desc;
         if (!hk_g17p_graphics_program_desc_init(cmd, s, linked, first, count,
                                                 &desc))
            return false;

         struct hk_g17p_gfx_loader *created =
            hk_g17p_gfx_loader_create(dev, s, &desc, true);
         if (!created)
            return false;

         while (true) {
            struct hk_g17p_gfx_loader *head =
               p_atomic_read(&linked->g17p_gfx_loaders);
            created->next = head;
            if (p_atomic_cmpxchg_ptr(&linked->g17p_gfx_loaders, head,
                                     created) == head)
               break;
         }

         if (!hk_agx3_indirect_loader_fits(created, launch, first, last))
            return false;
         l = created;
      }
   }

   static_assert((AGX_IOTO_PROGRAM_ALIGNMENT % 16) == 0,
                 "Ioto program stride must remain 16-byte aligned");
   const uint32_t copy_B = ALIGN_POT(l->length, 16);
   struct agx_ptr tmpl = hk_pool_alloc(cmd, copy_B, 16);
   if (!tmpl.cpu)
      return false;

   if (copy_B > l->length)
      memset((uint8_t *)tmpl.cpu + l->length, 0, copy_B - l->length);
   memcpy(tmpl.cpu, l->bytes, l->length);
   hk_g17p_patch_plan_apply(&l->plan, first, tmpl.cpu);

   uint32_t nrelocs = 0;
   for (unsigned r = 0; r < l->plan.reloc_count; ++r) {
      const struct hk_g17p_patch_reloc *rel = &l->plan.relocs[r];
      const uint8_t src = launch->source[rel->range];
      if (src != HK_G17P_RANGE_SOURCE_DRAW_PARAMS &&
          src != HK_G17P_RANGE_SOURCE_DRAW_ID)
         continue;

      assert(nrelocs < LIBAGX_INDIRECT_VDM_MAX_RELOCS);
      relocs[nrelocs++] = (struct libagx_indirect_ioto_reloc){
         .dest_lo = rel->dest[0],
         .dest_hi = rel->dest[1],
         .delta = rel->delta,
         .offset = rel->offset,
         .kind = src == HK_G17P_RANGE_SOURCE_DRAW_PARAMS ? 0u : 1u,
         .short_mask = rel->short_mask,
      };
   }

   *ioto_template_gpu = tmpl.gpu;
   *ioto_copy_dwords = copy_B / 4u;
   *ioto_stride_B = ALIGN_POT(l->length, AGX_IOTO_PROGRAM_ALIGNMENT);
   *reloc_count = nrelocs;
   return true;
}

static ATTRIBUTE_NOINLINE uint64_t
hk_g18g_upload_graphics_program(struct hk_cmd_buffer *cmd,
                                struct hk_device *dev, struct hk_shader *s,
                                struct hk_linked_shader *linked)
{
   struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   uint8_t sources[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   size_t range_count = 0;
   struct hk_agx3_g17p_graphics_program_desc desc;
   size_t capacity;

   /* G17G retains its native uncached composer. G17P's new loader and
    * replay records describe a different instruction encoding. */
   if (!hk_g17p_basic_graphics_program_supported(cmd, s, linked) ||
       !hk_g17p_graphics_uniform_ranges(cmd, s, linked, cmd->state.gfx.root,
                                        ranges, &range_count, sources) ||
       !hk_g17p_graphics_program_desc_init(cmd, s, linked, ranges,
                                           range_count, &desc) ||
       !hk_agx3_g17g_graphics_program_bound(
          dev->dev.chip == AGX_CHIP_G18G ? AGX_CHIP_G17G : dev->dev.chip,
          ranges, range_count, &capacity)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return 0;
   }

   /* Room for the agx3d re-encoding: twice the G17G bound covers every
    * witness (agx_ioto_capacity). Callers refuse a longer program. */
   const size_t reserve_capacity =
      dev->dev.chip == AGX_CHIP_G18G ? 2 * capacity : capacity;
   hk_cmd_buffer_retain_bo(cmd, linked->b.bo);
   if (s->b.info.rodata.size_16 != 0 || s->b.info.has_preamble)
      hk_cmd_buffer_retain_bo(cmd, s->bo);

   struct hk_ioto_reservation reservation;
   if (!hk_cmd_reserve_ioto_program(cmd, reserve_capacity,
                                     "G17G Ioto program", &reservation))
      return 0;
   desc.program_va = reservation.va;
   if (!agx_va_in_usc_window(&dev->dev, desc.program_va) ||
       !agx_va_in_usc_window(&dev->dev, desc.shader_va)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
      return 0;
   }

   struct hk_agx3_g17p_graphics_program program;
   if (!hk_agx3_g17g_basic_graphics_program_generate(dev->dev.chip, &desc,
                                                     &program)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return 0;
   }
   if (program.length > capacity) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return 0;
   }
   if (dev->dev.chip == AGX_CHIP_G18G) {
      /* G18G consumes the canonical GPU1 record re-encoded as agx3d. The
       * G17G program embeds no address of its own, so the longer text
       * stays position-independent (agx_ioto_capacity). */
      struct util_dynarray code;
      util_dynarray_init(&code, NULL);
      memcpy(util_dynarray_grow_bytes(&code, 1, program.length),
             program.bytes, program.length);
      const struct agx3_transcode_ioto ioto = {
         .api_program = true,
         .shader_gprs = desc.nr_gprs,
      };
      const char *why = NULL;
      unsigned at = 0;
      const bool ok =
         agx3_transcode_agx3d_ioto(&code, 0, &ioto, &why, &at) &&
         code.size <= reserve_capacity;
      if (!ok) {
         mesa_loge("hk: G18G Ioto program not re-encoded: %s at +0x%x",
                   why ? why : "longer than the reservation", at);
         util_dynarray_fini(&code);
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return 0;
      }
      const size_t published_size = code.size;
      memcpy(reservation.map, code.data, code.size);
      util_dynarray_fini(&code);
      return hk_cmd_publish_ioto_program(cmd, &reservation,
                                         published_size);
   }
   memcpy(reservation.map, program.bytes, program.length);
   return hk_cmd_publish_ioto_program(cmd, &reservation, program.length);
}

static ATTRIBUTE_NOINLINE uint64_t
hk_agx3_upload_graphics_program_slow(
   struct hk_cmd_buffer *cmd, struct hk_device *dev, struct hk_shader *s,
   struct hk_linked_shader *linked,
   struct hk_g17p_graphics_ioto_cache_entry *cached,
   struct hk_g17p_launch *launch, uint64_t root, const struct hk_shader *hw_vs,
   uint8_t flags)
{
   struct hk_graphics_state *agfx = &cmd->state.gfx;

   /* The admission below depends only on the shader and its linked
    * program for a hardware vertex stage (API VS, GS raster, or TES
    * raster), plus the tile-buffer spill state, the admitted fragment
    * layout, and the varying-coefficient inputs (topology, provoking
    * vertex, hardware vertex/mesh shader) for a fragment shader. An
    * earlier compact publication in this recording with the same inputs
    * was already admitted. */
   const bool spills = agx_tilebuffer_spills(&agfx->render.tilebuffer);
   const bool same_program =
      cached->admit.s == s && cached->admit.linked == linked;
   const bool vertex_admitted =
      same_program &&
      (s->info.stage == MESA_SHADER_VERTEX ||
       ((s->info.stage == MESA_SHADER_GEOMETRY ||
         s->info.stage == MESA_SHADER_TESS_EVAL) &&
        s->b.info.stage == MESA_SHADER_VERTEX));
   const bool fragment_admitted =
      same_program && s->info.stage == MESA_SHADER_FRAGMENT &&
      hk_g17p_basic_fragment_layout(dev, cmd) &&
      cached->admit.spills == spills &&
      cached->admit.topology == agfx->topology &&
      cached->admit.provoking == agfx->provoking && hw_vs &&
      cached->admit.hw_vs == hw_vs;
   if (!vertex_admitted && !fragment_admitted) {
      if (!hk_g17p_basic_graphics_program_supported(cmd, s, linked)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
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
   if (hk_g17p_graphics_uniform_ranges(cmd, s, linked, root, loader_ranges,
                                       &loader_range_count, sources)) {
      const uint64_t published = hk_g17p_publish_gfx_loader(
         cmd, s, linked, cached, loader_ranges, loader_range_count);
      if (published) {
         hk_g17p_launch_record(cmd, launch, s, linked, cached->loader, flags,
                               hw_vs, root, loader_ranges, loader_range_count,
                               sources);
         return published;
      }
   }

   /* The loader path above covers every admitted program; a refused
    * range layout or loader leaves nothing to launch. */
   if (!vk_command_buffer_has_error(&cmd->vk)) {
      mesa_loge_once("%s graphics stage program refused: stage=%u "
                     "sets=%u ranges",
                     agx_chip_name(dev->dev.chip), (unsigned)s->info.stage,
                     s->info.set_count);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   }
   return 0;
}

static ATTRIBUTE_NOINLINE uint64_t
hk_agx2_upload_usc_words(struct hk_cmd_buffer *cmd, struct hk_device *dev,
                         struct hk_shader *s, struct hk_linked_shader *linked)
{
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

static ATTRIBUTE_NOINLINE uint64_t
hk_agx3_upload_graphics_program(struct hk_cmd_buffer *cmd,
                                struct hk_device *dev, struct hk_shader *s,
                                struct hk_linked_shader *linked)
{
   cmd->g17p_gfx_caches_used = true;
   s = hk_linked_shader_source(s, linked);
   const unsigned stage = s->info.stage == MESA_SHADER_FRAGMENT ? 1 : 0;
   struct hk_graphics_state *agfx = &cmd->state.gfx;
   struct hk_g17p_graphics_ioto_cache_entry *cached = &agfx->g17p_ioto[stage];
   const uint64_t root =
      stage && !s->info.agx3_gen1_fragment_rq &&
            (agx_chip_g17_m5(dev->dev.chip) ||
             agx_chip_native_hal200(dev->dev.chip))
         ? hk_agx3_fragment_root(cmd, s)
         : agfx->root;
   const struct hk_shader *hw_vs =
      stage ? hk_bound_hw_vs_or_null(agfx) : NULL;
   const uint8_t flags = hk_g17p_launch_flags(cmd, s);
   struct hk_g17p_launch *launch = hk_g17p_launch_slot(cmd, stage, linked);
   if (likely(!s->info.agx3_gen1_fragment_rq &&
              hk_g17p_launch_matches(dev, cmd, launch, s, linked, flags,
                                     hw_vs))) {
      const uint64_t published =
         hk_g17p_launch_replay(cmd, launch, linked, cached, root);
      if (likely(published))
         return published;
      if (vk_command_buffer_has_error(&cmd->vk))
         return 0;
   }

   return hk_agx3_upload_graphics_program_slow(cmd, dev, s, linked, cached,
                                               launch, root, hw_vs, flags);
}

static inline uint64_t
hk_upload_usc_words_internal(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                             struct hk_linked_shader *linked)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (likely(agx_chip_native_or_m5(dev->dev.chip)))
      return hk_agx3_upload_graphics_program(cmd, dev, s, linked);

   if (dev->dev.chip == AGX_CHIP_G18G)
      return hk_g18g_upload_graphics_program(cmd, dev, s, linked);

   return hk_agx2_upload_usc_words(cmd, dev, s, linked);
}

uint64_t
hk_agx3_try_push_only_vs_program(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                                 struct hk_linked_shader *linked)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;

   s = hk_linked_shader_source(s, linked);
   if ((!agx_chip_native_hal200(dev->dev.chip) &&
        !agx_chip_g17_m5(dev->dev.chip)) ||
       s->info.stage != MESA_SHADER_VERTEX || s->info.agx3_gen1_fragment_rq ||
       !gfx->root || gfx->g17p_hoisted_preamble_va ||
       !hk_root_bypass_dirty(desc))
      return 0;

   struct hk_g17p_launch *launch = hk_g17p_launch_slot(cmd, 0, linked);
   const uint8_t flags = hk_g17p_launch_flags(cmd, s);
   if (!launch->valid || !launch->root_bypass_ok || !launch->loader ||
       !hk_g17p_launch_matches(dev, cmd, launch, s, linked, flags, NULL))
      return 0;

   if (launch->has_general_root &&
       !hk_root_reads_ignore_dirty(s->info.agx3_root_reads,
                                   s->info.agx3_dynbuf_reads, desc))
      return 0;

   const uint8_t dirty_mask = desc->root_change.dirty_mask;
   const unsigned count = launch->count;
   uint32_t vb_base_B = 0;
   uint32_t vb_clamp_B = 0;
   if (dirty_mask & (HK_ROOT_DIRTY_VB_BASE | HK_ROOT_DIRTY_VB_CLAMP)) {
      for (unsigned i = 0; i < count; ++i) {
         if (launch->source[i] == HK_G17P_RANGE_SOURCE_ATTRIB_BASE &&
             (dirty_mask & HK_ROOT_DIRTY_VB_BASE))
            vb_base_B = 2u * launch->size_halfs[i];
         else if (launch->source[i] == HK_G17P_RANGE_SOURCE_ATTRIB_CLAMP &&
                  (dirty_mask & HK_ROOT_DIRTY_VB_CLAMP))
            vb_clamp_B = 2u * launch->size_halfs[i];
      }
   }

   const uint32_t push_B =
      (dirty_mask & HK_ROOT_DIRTY_PUSH) ? launch->push_size_B : 0;
   if (push_B && (unsigned)launch->push_min_B + push_B > HK_MAX_PUSH_SIZE)
      return 0;

   uint64_t vb_base_va = 0;
   uint64_t vb_clamp_va = 0;
   uint64_t push_slice_va = 0;
   const uint32_t total_upload_B = vb_base_B + vb_clamp_B + push_B;
   if (total_upload_B) {
      /* Pack all dirty narrow slices into one contiguous pool allocation so
       * they share a single base cluster in the Ioto patch plan. */
      struct agx_ptr up =
         hk_pool_alloc(cmd, total_upload_B, vb_base_B ? 8 : 4);
      if (!up.cpu)
         return 0;
      uint32_t off = 0;
      if (vb_base_B) {
         memcpy((uint8_t *)up.cpu + off, desc->root.draw.attrib_base, vb_base_B);
         vb_base_va = up.gpu + off;
         off += vb_base_B;
      }
      if (vb_clamp_B) {
         memcpy((uint8_t *)up.cpu + off, desc->root.draw.attrib_clamps,
                vb_clamp_B);
         vb_clamp_va = up.gpu + off;
         off += vb_clamp_B;
      }
      if (push_B) {
         memcpy((uint8_t *)up.cpu + off, desc->root.push + launch->push_min_B,
                push_B);
         push_slice_va = up.gpu + off;
      }
   }

   const uint64_t slice_origin =
      hk_root_descriptor_offset(push) + launch->push_min_B;
   struct agx3_ioto_uniform_range ranges[HK_AGX3_G17P_GRAPHICS_MAX_RANGES];
   for (unsigned i = 0; i < count; ++i) {
      uint64_t address;
      switch (launch->source[i]) {
      case HK_G17P_RANGE_SOURCE_ROOT:
         address = gfx->root + launch->offset[i];
         break;
      case HK_G17P_RANGE_SOURCE_ATTRIB_BASE:
         address = vb_base_va ? vb_base_va : gfx->root + launch->offset[i];
         break;
      case HK_G17P_RANGE_SOURCE_ATTRIB_CLAMP:
         address = vb_clamp_va ? vb_clamp_va : gfx->root + launch->offset[i];
         break;
      case HK_G17P_RANGE_SOURCE_PUSH:
         address = push_slice_va
                      ? push_slice_va + (launch->offset[i] - slice_origin)
                      : gfx->root + launch->offset[i];
         break;
      case HK_G17P_RANGE_SOURCE_DRAW_PARAMS:
         address = gfx->draw_params;
         break;
      case HK_G17P_RANGE_SOURCE_DRAW_ID:
         address = gfx->draw_id_ptr;
         break;
      case HK_G17P_RANGE_SOURCE_SET: {
         const uint64_t base = desc->root.sets[launch->set[i]];
         address = base ? base + launch->offset[i] : AGX_ZERO_PAGE_ADDRESS;
         if (address & 3)
            return 0;
         break;
      }
      case HK_G17P_RANGE_SOURCE_BUFFER: {
         struct hk_buffer_address buffer;
         if (!hk_g17p_dma_buffer(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 launch->set[i] == HK_G17P_LAUNCH_ROOT_SET
                                    ? -1 : launch->set[i],
                                 launch->descriptor_B[i], &buffer) ||
             ALIGN_POT((uint64_t)buffer.size, 4) <
                launch->offset[i] + 2u * launch->size_halfs[i])
            return 0;
         address = buffer.base_addr + launch->offset[i];
         if (address & 3)
            return 0;
         break;
      }
      case HK_G17P_RANGE_SOURCE_CONSTANT:
         address = launch->offset[i];
         break;
      default:
         return 0;
      }
      ranges[i] = (struct agx3_ioto_uniform_range){
         .start_halfs = launch->start_halfs[i],
         .size_halfs = launch->size_halfs[i],
         .address = address,
      };
   }

   const struct hk_g17p_gfx_loader *l =
      launch->bypass_loader ? launch->bypass_loader : launch->loader;
   if (likely(hk_g17p_patch_plan_fits(&l->plan, ranges)))
      return hk_g17p_gfx_loader_publish(cmd, linked, &gfx->g17p_ioto[0], l,
                                        ranges, count);

   /* Split bypass ranges (e.g. root-resident attrib_base plus pool-resident
    * push constants) have a different cluster shape than the initial full-root
    * loader. Build or look up a matching loader once and cache it on launch. */
   const uint64_t published = hk_g17p_publish_gfx_loader(
      cmd, s, linked, &gfx->g17p_ioto[0], ranges, count);
   if (published)
      launch->bypass_loader = gfx->g17p_ioto[0].loader;
   return published;
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

static bool
hk_g18_prepare_precomp_launch(
   struct hk_cmd_buffer *cmd, enum libagx_program idx,
   struct agx_precompiled_shader *prog, const void *data, size_t data_size,
   struct agx3_compute_program *launch_out,
   struct drm_asahi_g18_compute_resources *res_out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const uint32_t scratch_bytes =
      MAX3(prog->info.scratch_size, prog->info.agx3_spectpr_scratch_size,
           prog->info.agx3_tps_size);
   const uint32_t max_spill = 1u << 16;
   const uint32_t shared_bytes =
      prog->info.local_size
         ? MAX2(256u, ALIGN_POT(prog->info.local_size * 4u, 64u))
         : 0;
   if (!prog->bo || !prog->info.main_size || !prog->info.nr_gprs ||
       prog->info.nr_preamble_gprs || scratch_bytes > max_spill ||
       shared_bytes > 32768u ||
       (prog->info.rodata.size_16 && !prog->rodata_offset) ||
       (prog->info.printf_uniform &&
        prog->info.printf_uniform < AGX_ABI_AGX3_PRECOMP_ARG_HALFS) ||
       !prog->info.agx3_staging_known || !data || !data_size ||
       data_size > UINT16_MAX * 2u ||
       prog->entry_offset > UINT64_MAX - prog->ptr ||
       !prog->b.workgroup.x || !prog->b.workgroup.y ||
       !prog->b.workgroup.z) {
      mesa_loge("hk: refusing precompiled G18G program %u shape "
                "(preamble_gprs=%u scratch=%u tps=%u local=%u "
                "staging_known=%u)",
                idx, prog->info.nr_preamble_gprs, scratch_bytes,
                prog->info.agx3_tps_size, prog->info.local_size,
                prog->info.agx3_staging_known);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   const size_t printf_offset =
      prog->info.printf_uniform
         ? (size_t)(prog->info.printf_uniform -
                    AGX_ABI_AGX3_PRECOMP_ARG_HALFS) * 2
         : 0;
   const size_t upload_size =
      ALIGN_POT(prog->info.printf_uniform
                   ? MAX2(data_size, printf_offset + 8)
                   : data_size, 4);
   struct agx_ptr uploaded = hk_pool_alloc(cmd, upload_size, 4);
   if (!uploaded.cpu || upload_size / 2 > UINT16_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }
   memcpy(uploaded.cpu, data, data_size);
   memset((uint8_t *)uploaded.cpu + data_size, 0, upload_size - data_size);
   if (prog->info.printf_uniform)
      memcpy((uint8_t *)uploaded.cpu + printf_offset,
             &dev->dev.printf_buffer_address, sizeof(uint64_t));
   struct agx3_ioto_uniform_range ranges[2] = {0};
   unsigned range_count = 0;
   ranges[range_count++] = (struct agx3_ioto_uniform_range){
      .start_halfs = AGX_ABI_AGX3_PRECOMP_ARG_HALFS,
      .size_halfs = upload_size / 2,
      .address = uploaded.gpu,
   };
   if (prog->info.rodata.size_16) {
      ranges[range_count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = prog->info.rodata.base_uniform,
         .size_halfs = prog->info.rodata.size_16,
         .address = prog->ptr + prog->rodata_offset,
      };
   }

   struct agx_ptr program_t = hk_pool_usc_alloc(
      cmd, AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE, AGX_IOTO_PROGRAM_ALIGNMENT);
   if (!program_t.cpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }
   if (!agx_va_in_usc_window(&dev->dev, program_t.gpu) ||
       !agx_va_in_usc_window(&dev->dev, prog->ptr + prog->entry_offset)) {
      mesa_loge("hk: G18G helper %u: program %" PRIx64 " or entry %" PRIx64
                " outside the USC window (base %" PRIx64 ")",
                idx, (uint64_t)program_t.gpu,
                (uint64_t)(prog->ptr + prog->entry_offset),
                (uint64_t)dev->dev.shader_base);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
      return false;
   }
   const struct agx3_g17p_graphics_program_desc desc = {
      .program_va = program_t.gpu,
      .shader_va = prog->ptr + prog->entry_offset,
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
      .shader_binary_size = prog->info.main_size,
      .scratch_size = scratch_bytes,
      .nr_gprs = prog->info.nr_gprs,
      .compiler_d_valid = prog->info.agx3_staging_known,
      .compiler_d = (uint32_t)prog->info.agx3_staging_halfs * 2,
      .threadgroup_memory_size_B = shared_bytes,
   };
   struct agx3_g17p_graphics_program generated;
   if (!agx3_g18g_compute_program_generate(&desc, &generated) ||
       generated.length > AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE) {
      mesa_loge("hk: G18G compute program %u generation failed", idx);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }
   struct util_dynarray code;
   util_dynarray_init(&code, NULL);
   memcpy(util_dynarray_grow_bytes(&code, 1, generated.length),
          generated.bytes, generated.length);
   const struct agx3_transcode_ioto ioto = {
      .api_program = true,
      .shader_gprs = prog->info.nr_gprs,
   };
   const char *why = NULL;
   unsigned at = 0;
   const bool ok = agx3_transcode_agx3d_ioto(&code, 0, &ioto, &why, &at) &&
                   code.size <= AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE;
   if (!ok) {
      mesa_loge("hk: G18G compute program %u not re-encoded: %s at +0x%x", idx,
                why ? why : "too long", at);
      util_dynarray_fini(&code);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }
   memcpy(program_t.cpu, code.data, code.size);
   util_dynarray_fini(&code);

   struct agx3_g17p_native_compute_program_desc bind_desc = {
      .chip = AGX_CHIP_G17G,
      .program_va = program_t.gpu,
      .shader_va = prog->ptr + prog->entry_offset,
   };
   if (!agx3_cdm_direct_template_sources_compute(false, false, false,
                                                 &bind_desc.launch_sources) ||
       !agx3_g17g_native_compute_program_bind(&bind_desc, launch_out)) {
      mesa_loge("hk: G18G compute program %u bind failed", idx);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }
   if (prog->info.local_size)
      launch_out->launch_template.word0 |= UINT32_C(1) << 23;

   const uint32_t main_scratch =
      MAX2(prog->info.scratch_size, prog->info.agx3_tps_size);
   *res_out = (struct drm_asahi_g18_compute_resources){
      .version = 1,
      .flags = 1u,
      .registers_16 = prog->info.nr_gprs,
      .scratch_bytes = main_scratch,
      .spectpr_scratch_bytes = prog->info.agx3_spectpr_scratch_size,
      .staging_16 = prog->info.agx3_staging_halfs,
      .local_bytes = prog->info.local_size,
      .imageblock_stride = prog->info.imageblock_stride,
      .local_size = {prog->b.workgroup.x, prog->b.workgroup.y,
                     prog->b.workgroup.z},
   };
   return true;
}

bool
hk_g18_emit_indirect_launch(
   struct hk_cmd_buffer *cmd, struct hk_cs **target,
   const struct agx3_compute_program *launch, struct agx_grid grid,
   struct agx_workgroup local_size,
   const struct drm_asahi_g18_compute_resources *target_res)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!agx_is_indirect(grid) || !grid.ptr || !launch || !target || !target_res) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   /* Build the 48-byte direct CDM template (40B launch + 4B strong barrier +
    * 4B terminate) with 3D bits preserved; libagx_g18_indirect_launch clears
    * them when global_y < 2 && global_z < 2. */
   uint32_t tmpl[12] = {0};
   struct agx3_compute_dispatch_sink tmpl_sink = {
      .data = (uint8_t *)tmpl,
      .capacity = sizeof(tmpl),
   };
   const struct agx3_compute_direct_grid dummy_3d = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {2, 2, 2},
      .local_size = {1, 1, 1},
      .dimensions = 3,
   };
   uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
   if (!agx3_g17g_native_compute_dispatch_append(launch, &dummy_3d, 5,
                                                 &tmpl_sink) ||
       !agx3_cdm_stream_terminate_pack(dev->dev.chip, terminate) ||
       tmpl_sink.length != 44) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }
   memcpy((uint8_t *)tmpl + 44, terminate, sizeof(terminate));

   /* Build the 48-byte 1-thread no-op fallback launch for zero grids so G18
    * CDM never sees an empty or zero-dimension dispatch. */
   struct agx_precompiled_shader *noop_prog =
      agx_get_precompiled(&dev->bg_eot, LIBAGX_G18_NOOP);
   if (!noop_prog) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }
   struct agx_ptr scratch = hk_pool_alloc(cmd, 16, 16);
   if (!scratch.gpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }
   struct agx3_compute_program noop_launch;
   struct drm_asahi_g18_compute_resources noop_res;
   if (!hk_g18_prepare_precomp_launch(cmd, LIBAGX_G18_NOOP, noop_prog,
                                      &scratch.gpu, sizeof(scratch.gpu),
                                      &noop_launch, &noop_res))
      return false;
   noop_launch.launch_template.word0 &= UINT32_C(0xfff1ffff);

   uint32_t noop_tmpl[12] = {0};
   struct agx3_compute_dispatch_sink noop_sink = {
      .data = (uint8_t *)noop_tmpl,
      .capacity = sizeof(noop_tmpl),
   };
   const struct agx3_compute_direct_grid noop_direct = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {1, 1, 1},
      .local_size = {1, 1, 1},
      .dimensions = 1,
   };
   if (!agx3_g17g_native_compute_dispatch_append(&noop_launch, &noop_direct, 5,
                                                 &noop_sink) ||
       noop_sink.length != 44) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }
   memcpy((uint8_t *)noop_tmpl + 44, terminate, sizeof(terminate));

   const uint64_t tmpl_gpu = hk_pool_upload(cmd, tmpl, sizeof(tmpl), 16);
   const uint64_t noop_gpu =
      hk_pool_upload(cmd, noop_tmpl, sizeof(noop_tmpl), 16);
   struct agx_ptr stream_mem =
      hk_pool_cs_alloc(cmd, hk_cs_root_size(dev->dev.chip, true), 1024);
   if (!tmpl_gpu || !noop_gpu || !stream_mem.cpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }
   memcpy(stream_mem.cpu, noop_tmpl, sizeof(noop_tmpl));
   memset((uint8_t *)stream_mem.cpu + sizeof(noop_tmpl), 0,
          HK_CS_SPACE_PADDING);

   const enum agx_barrier helper_barrier =
      AGX_BARRIER_ALL |
      (target == &cmd->current_cs.pre_gfx    ? AGX_PREGFX
       : target == &cmd->current_cs.post_gfx ? AGX_POSTGFX
                                             : 0);
   libagx_g18_indirect_launch(cmd, agx_1d(1), helper_barrier, grid.ptr,
                              tmpl_gpu, noop_gpu, stream_mem.gpu, local_size.x,
                              local_size.y, local_size.z,
                              grid.mode == AGX_CDM_MODE_INDIRECT_LOCAL ? 1u : 0u);
   if (vk_command_buffer_has_error(&cmd->vk))
      return false;

   struct hk_cs *helper_cs = *target;
   struct hk_cs *exec_cs = hk_cs_create_at(cmd, true, stream_mem);
   if (!exec_cs) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return false;
   }
   if (target == &cmd->current_cs.pre_gfx && cmd->current_cs.gfx) {
      list_addtail(&exec_cs->node, &cmd->current_cs.gfx->node);
   } else {
      list_addtail(&exec_cs->node, &cmd->control_streams);
   }
   *target = exec_cs;
   hk_sync_stream_created(cmd, exec_cs, target, true);
   if (helper_cs) {
      exec_cs->sync.dep[HK_SYNC_CDM].cut =
         hk_sync_later(exec_cs->sync.dep[HK_SYNC_CDM].cut, helper_cs);
      exec_cs->sync.dep_ta[HK_SYNC_CDM].cut =
         hk_sync_later(exec_cs->sync.dep_ta[HK_SYNC_CDM].cut, helper_cs);
   }
   exec_cs->current = (uint8_t *)exec_cs->current + sizeof(tmpl);
   exec_cs->stats.cmds = 1;
   exec_cs->g18_dispatch_sealed = true;

   struct drm_asahi_g18_compute_resources exec_res = *target_res;
   exec_res.registers_16 = MAX2(exec_res.registers_16, noop_res.registers_16);
   exec_res.scratch_bytes =
      MAX2(exec_res.scratch_bytes, noop_res.scratch_bytes);
   exec_res.spectpr_scratch_bytes =
      MAX2(exec_res.spectpr_scratch_bytes, noop_res.spectpr_scratch_bytes);
   exec_res.staging_16 = MAX2(exec_res.staging_16, noop_res.staging_16);
   if (grid.mode == AGX_CDM_MODE_INDIRECT_LOCAL) {
      exec_res.flags = (1u << 0) | (1u << 3);
      exec_res.local_size[0] = 0;
      exec_res.local_size[1] = 0;
      exec_res.local_size[2] = 0;
   } else {
      exec_res.flags = (1u << 0);
      exec_res.local_size[0] = local_size.x;
      exec_res.local_size[1] = local_size.y;
      exec_res.local_size[2] = local_size.z;
   }
   exec_cs->g18_compute_resources = exec_res;
   hk_cmd_buffer_retain_bo(cmd, noop_prog->bo);
   return true;
}

void
hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                    enum agx_barrier barrier, enum libagx_program idx,
                    void *data, size_t data_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (agx_chip_native_agx3(dev->dev.chip)) {
      hk_dispatch_precomp_g17p(cmd, grid, barrier, idx, data, data_size);
      return;
   }

   if (agx_chip_g17_m5(dev->dev.chip)) {
      hk_dispatch_precomp_m5(cmd, grid, barrier, idx, data, data_size);
      return;
   }

   struct agx_precompiled_shader *prog = agx_get_precompiled(&dev->bg_eot, idx);
   if (!prog) {
      if (dev->dev.chip == AGX_CHIP_G18G)
         mesa_loge("missing precompiled helper %u on %s (barrier=%u)",
                   idx, agx_chip_name(dev->dev.chip), barrier);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   if (dev->dev.chip == AGX_CHIP_G18G) {
      struct hk_cs **target =
         (barrier & AGX_POSTGFX)  ? &cmd->current_cs.post_gfx
         : (barrier & AGX_PREGFX) ? &cmd->current_cs.pre_gfx
                                  : &cmd->current_cs.cs;
      if (!agx_is_indirect(grid) &&
          (!grid.count[0] || !grid.count[1] || !grid.count[2]))
         return;

      struct agx3_compute_program launch;
      struct drm_asahi_g18_compute_resources prog_res;
      if (!hk_g18_prepare_precomp_launch(cmd, idx, prog, data, data_size,
                                         &launch, &prog_res))
         return;

      if (agx_is_indirect(grid)) {
         if (hk_g18_emit_indirect_launch(cmd, target, &launch, grid,
                                         prog->b.workgroup, &prog_res))
            hk_cmd_buffer_retain_bo(cmd, prog->bo);
         return;
      }

      if (*target && (*target)->stats.cmds > 0 &&
          !(*target)->g18_dispatch_sealed && (*target)->imm_writes.size == 0)
         *target = NULL;
      struct hk_cs *cs = hk_cmd_buffer_get_cs_general(cmd, target, true);
      if (!cs)
         return;

      const uint32_t shared_bytes =
         prog->info.local_size
            ? MAX2(256u, ALIGN_POT(prog->info.local_size * 4u, 64u))
            : 0;
      hk_ensure_cs_has_space(cmd, cs, 0x200);
      struct agx3_compute_dispatch_sink sink = {
         .data = cs->current,
         .capacity = (size_t)((uint8_t *)cs->end - (uint8_t *)cs->current),
      };
      const struct agx3_compute_direct_grid direct = {
         .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
         .size = {grid.count[0], grid.count[1], grid.count[2]},
         .local_size = {prog->b.workgroup.x, prog->b.workgroup.y,
                        prog->b.workgroup.z},
         .dimensions = grid.count[2] != 1 || prog->b.workgroup.z != 1
                          ? 3
                          : (grid.count[1] != 1 || prog->b.workgroup.y != 1
                                ? 2
                                : 1),
      };
      if (prog->info.local_size && direct.size[1] < 2 && direct.size[2] < 2)
         launch.launch_template.word0 &= UINT32_C(0xfff1ffff);
      const bool launched = agx3_g17g_native_compute_dispatch_append(
         &launch, &direct, 5, &sink);
      uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
      if (!launched ||
          !agx3_cdm_stream_terminate_pack(dev->dev.chip, terminate) ||
          sink.capacity - sink.length < sizeof(terminate)) {
         mesa_loge("hk: G18G compute program %u dispatch append failed",
                   idx);
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      memcpy(sink.data + sink.length, terminate, sizeof(terminate));
      sink.length += sizeof(terminate);
      cs->current = sink.data + sink.length;
      const bool had_cmds = cs->stats.cmds > 0;
      cs->stats.cmds++;
      cs->g18_dispatch_sealed = true;
      if (!had_cmds) {
         cs->g18_compute_resources = prog_res;
      } else {
         const uint32_t prod = prog->b.workgroup.x * prog->b.workgroup.y *
                               prog->b.workgroup.z;
         const uint32_t prev_prod = cs->g18_compute_resources.local_size[0] *
                                    cs->g18_compute_resources.local_size[1] *
                                    cs->g18_compute_resources.local_size[2];
         cs->g18_compute_resources.registers_16 =
            MAX2(cs->g18_compute_resources.registers_16, prog_res.registers_16);
         cs->g18_compute_resources.scratch_bytes =
            MAX2(cs->g18_compute_resources.scratch_bytes,
                 prog_res.scratch_bytes);
         cs->g18_compute_resources.spectpr_scratch_bytes =
            MAX2(cs->g18_compute_resources.spectpr_scratch_bytes,
                 prog_res.spectpr_scratch_bytes);
         cs->g18_compute_resources.staging_16 =
            MAX2(cs->g18_compute_resources.staging_16, prog_res.staging_16);
         const uint32_t helper_local =
            cs->g18_compute_resources.version == 2 ? shared_bytes
                                                   : prog_res.local_bytes;
         cs->g18_compute_resources.local_bytes =
            MAX2(cs->g18_compute_resources.local_bytes, helper_local);
         cs->g18_compute_resources.imageblock_stride =
            MAX2(cs->g18_compute_resources.imageblock_stride,
                 prog_res.imageblock_stride);
         if (prod >= prev_prod) {
            cs->g18_compute_resources.local_size[0] = prog->b.workgroup.x;
            cs->g18_compute_resources.local_size[1] = prog->b.workgroup.y;
            cs->g18_compute_resources.local_size[2] = prog->b.workgroup.z;
         }
      }
      hk_cmd_buffer_retain_bo(cmd, prog->bo);
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

   cmd->state.cs.cdm_barrier_full_needed = true;
   cmd->state.cs.cdm_stream_has_images = true;
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
   const bool g17p = agx_chip_native_agx3(dev->dev.chip);
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

   /* Only the native AGX3 path submits the spans (hk_queue.c). G18G
    * submits them too, as it did before the paths were split. */
   if (agx_chip_native_or_g18g(dev->dev.chip))
      hk_cs_capture_fragment_attachments(dev, render, cs);

   if (agx_chip_agx3_path(dev->dev.chip)) {
      /* The final VDM word stays zero when this selector is the
       * UINT32_MAX sentinel.  This reproduces m1n1's direct-triangle prefix. */
      cs->agx3_active_context_selector_valid = true;
      cs->agx3_raw_active_context_selector = UINT32_MAX;
   }

   cs->tib = render->tilebuffer;
   cs->area_x1 = render->area.offset.x + render->area.extent.width;
   cs->area_y1 = render->area.offset.y + render->area.extent.height;
   /* Attachmentless rendering defers its API sample count. Publish the
    * native one-sample no-imageblock profile in this control stream while
    * preserving the deferred count in render state. */
   if (agx_chip_hal300_path(dev->dev.chip) && !cs->tib.sample_size_B &&
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

   /* Conservative default for continuation/split streams where bg.main is
    * bg.partial: every bound colour attachment has live BG contents and cannot
    * be synthesized by constant epilog clear-fill. hk_begin_rendering()
    * overrides these on the first stream of a non-resuming pass. */
   uint8_t bound_color_mask = 0;
   for (uint32_t i = 0; i < render->color_att_count; ++i) {
      if (render->color_att[i].iview)
         bound_color_mask |= BITFIELD_BIT(i);
   }
   cs->bg_live_mask = bound_color_mask;
   cs->bg_clear_fill_rt = 0;
   cs->user_color_written_mask = 0;
   cs->opaque_color_written = 0;

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
      const bool admits_inval =
         dev->dev.params.features & DRM_ASAHI_FEATURE_VDM_USC_INVALIDATE;
      if (!agx_vdm_barrier_core_pack(dev->dev.chip, admits_inval, &barrier)) {
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
   } else if (agx_chip_native_agx3(dev->dev.chip)) {
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

   util_dynarray_clear(&cs->scissor);
   util_dynarray_clear(&cs->depth_bias);
   if (!util_dynarray_ensure_cap(&cs->scissor,
                                 HK_MAX_VIEWPORTS * AGX_SCISSOR_LENGTH) ||
       !util_dynarray_ensure_cap(&cs->depth_bias,
                                 HK_MAX_VIEWPORTS * AGX_DEPTH_BIAS_LENGTH)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return false;
   }

   /* All graphics state must be reemited in each control stream */
   hk_cmd_buffer_dirty_all(cmd);

   if (agx_chip_native_agx3(dev->dev.chip))
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
   /* Conservative: AGX3 path chips re-emit their restart state after a
    * link, and HAL200 re-publishes its full bind tail before relying on
    * incremental FS PPP updates or tail elision in the new chunk. */
   if (vdm && agx_chip_agx3_path(hk_cmd_buffer_device(cmd)->dev.chip))
      cs->restart_index_set = false;
   if (vdm && agx_chip_native_hal200(hk_cmd_buffer_device(cmd)->dev.chip))
      cs->g17p_bind_tail.valid = false;
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
   const bool zs = view->vk.image->aspects & (VK_IMAGE_ASPECT_DEPTH_BIT |
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

static bool
hk_defer_clear_internal(struct hk_cmd_buffer *cmd, struct hk_image *image,
                        VkImageAspectFlags aspects, const VkClearValue *value,
                        uint32_t range_count,
                        const VkImageSubresourceRange *ranges, bool from_gfx,
                        const struct hk_sync_dep *gfx_dep);

static bool
hk_deferred_clear_eligible(struct hk_cmd_buffer *cmd,
                           const struct hk_image *image,
                           VkImageAspectFlags aspects, uint32_t range_count,
                           const VkImageSubresourceRange *ranges);

/* Try to fold an empty clear-only render pass into deferred whole-image clears
 * so a following pass that loads (or re-clears) the attachment absorbs the
 * clear directly as a tilebuffer load-op clear without an extra compute fill
 * job or memory round-trip. Only folds when every cleared attachment in the
 * pass is eligible; otherwise returns false so the caller emits compute fills. */
static bool
hk_try_defer_empty_render_clears(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (!agx_chip_native_or_m5(dev->dev.chip) || cmd->in_meta ||
       render->view_mask || render->layer_count != 1 ||
       (render->flags &
        (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT |
         VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT)) ||
       hk_sync_xfer_pending(&cmd->sync))
      return false;

   for (unsigned i = 0; i < render->color_att_count; ++i) {
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE)
         return false;
   }
   if (render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE ||
       render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      return false;

   struct empty_clear_cand {
      struct hk_image *image;
      VkImageAspectFlags aspects;
      VkClearValue value;
   } cands[HK_MAX_RTS + 1];
   unsigned cand_count = 0;

   for (unsigned i = 0; i < render->color_att_count; ++i) {
      const struct hk_attachment *att = &render->color_att[i];
      if (!att->clear || !att->iview)
         continue;
      if (att->store_op != VK_ATTACHMENT_STORE_OP_STORE)
         return false;

      struct hk_image_view *view = att->iview;
      struct hk_image *image = container_of(view->vk.image, struct hk_image, vk);
      const VkImageSubresourceRange range = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .baseMipLevel = view->vk.base_mip_level,
         .levelCount = view->vk.level_count,
         .baseArrayLayer = view->vk.base_array_layer,
         .layerCount = view->vk.layer_count,
      };
      if (view->vk.aspects != VK_IMAGE_ASPECT_COLOR_BIT ||
          image->vk.aspects != VK_IMAGE_ASPECT_COLOR_BIT ||
          view->vk.view_type != VK_IMAGE_VIEW_TYPE_2D ||
          att->vk_format != image->vk.format ||
          (image->vk.create_flags &
           (VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_SPARSE_BINDING_BIT)) ||
          render->area.offset.x != 0 || render->area.offset.y != 0 ||
          render->area.extent.width != image->vk.extent.width ||
          render->area.extent.height != image->vk.extent.height ||
          !hk_deferred_clear_eligible(cmd, image, VK_IMAGE_ASPECT_COLOR_BIT, 1,
                                      &range))
         return false;

      VkClearValue val = {0};
      memcpy(val.color.uint32, att->clear_colour, sizeof(val.color.uint32));
      cands[cand_count++] = (struct empty_clear_cand){
         .image = image,
         .aspects = VK_IMAGE_ASPECT_COLOR_BIT,
         .value = val,
      };
   }

   const bool has_z = render->depth_att.clear && render->depth_att.iview;
   const bool has_s = render->stencil_att.clear && render->stencil_att.iview;
   if (has_z || has_s) {
      if ((has_z && render->depth_att.store_op != VK_ATTACHMENT_STORE_OP_STORE) ||
          (has_s &&
           render->stencil_att.store_op != VK_ATTACHMENT_STORE_OP_STORE))
         return false;

      struct hk_image_view *z_view = has_z ? render->depth_att.iview : NULL;
      struct hk_image_view *s_view = has_s ? render->stencil_att.iview : NULL;
      struct hk_image *z_img =
         z_view ? container_of(z_view->vk.image, struct hk_image, vk) : NULL;
      struct hk_image *s_img =
         s_view ? container_of(s_view->vk.image, struct hk_image, vk) : NULL;
      if (has_z && has_s && z_img != s_img)
         return false;

      struct hk_image *image = z_img ? z_img : s_img;
      struct hk_image_view *view = z_view ? z_view : s_view;
      const VkImageAspectFlags aspects =
         (has_z ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
         (has_s ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
      const VkImageSubresourceRange range = {
         .aspectMask = aspects,
         .baseMipLevel = view->vk.base_mip_level,
         .levelCount = view->vk.level_count,
         .baseArrayLayer = view->vk.base_array_layer,
         .layerCount = view->vk.layer_count,
      };
      if (view->vk.view_type != VK_IMAGE_VIEW_TYPE_2D ||
          (has_z && has_s &&
           (z_view->vk.base_mip_level != s_view->vk.base_mip_level ||
            z_view->vk.level_count != s_view->vk.level_count ||
            z_view->vk.base_array_layer != s_view->vk.base_array_layer ||
            z_view->vk.layer_count != s_view->vk.layer_count)) ||
          (image->vk.create_flags &
           (VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_SPARSE_BINDING_BIT)) ||
          render->area.offset.x != 0 || render->area.offset.y != 0 ||
          render->area.extent.width != image->vk.extent.width ||
          render->area.extent.height != image->vk.extent.height ||
          !hk_deferred_clear_eligible(cmd, image, aspects, 1, &range))
         return false;

      VkClearValue val = {0};
      if (has_z)
         val.depthStencil.depth = uif(render->depth_att.clear_colour[0]);
      if (has_s)
         val.depthStencil.stencil = render->stencil_att.clear_colour[1];
      cands[cand_count++] = (struct empty_clear_cand){
         .image = image,
         .aspects = aspects,
         .value = val,
      };
   }

   if (cand_count == 0 ||
       cmd->state.deferred_clear_count + cand_count > HK_MAX_DEFERRED_CLEARS)
      return false;

   for (unsigned i = 0; i < cand_count; ++i) {
      for (unsigned j = i + 1; j < cand_count; ++j) {
         if (cands[i].image == cands[j].image)
            return false;
      }
   }

   struct hk_sync_dep saved_dep[HK_SYNC_NUM_SQ];
   memcpy(saved_dep, cs->sync.dep, sizeof(saved_dep));

   for (unsigned i = 0; i < cand_count; ++i) {
      const VkImageSubresourceRange range = {
         .aspectMask = cands[i].aspects,
         .levelCount = 1,
         .layerCount = 1,
      };
      bool ok = hk_defer_clear_internal(cmd, cands[i].image, cands[i].aspects,
                                        &cands[i].value, 1, &range, true,
                                        saved_dep);
      assert(ok);
   }

   return true;
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
   if (continuation && !agx_chip_native_or_m5(dev->dev.chip))
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
    * resolve is performed by this render's OWN end-of-tile program, and no
    * resolve runs after the pass: with storeOp other than STORE the
    * multisample attachment is never stored and the pass's only output is
    * the resolve target, and with STORE the same program stores the samples
    * and resolves them. Clearing the attachment as an image then leaves the
    * resolve target untouched, which is exactly
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
   const bool native_eot = agx_chip_agx3_path(dev->dev.chip);
   const bool output_only_through_eot =
      (native_eot && render->g17p_direct_color_resolve) ||
      render->agx2_resolve_mask;

   /* A timestamp belongs to this render. A direct AGX2 resolve also needs
    * the neutral raster state below when there are no draws.
    */
   if ((cs->timestamp.start.handle || cs->timestamp.end.handle ||
        cs->timestamp.vtx_end.handle) &&
       !(render->agx2_resolve_mask && cs->cr.process_empty_tiles))
      return false;

   /* HAL300: a multisample clear-only pass stays a render. As compute it is a
    * separate job that every later render's vertex half must wait for; as a
    * render that writes only its attachments it is not. */
   const bool m5_ms_clear = agx_chip_hal300_path(dev->dev.chip) &&
                            cs->cr.process_empty_tiles && cs->tib.nr_samples > 1;
   if ((cs->cr.process_empty_tiles && output_only_through_eot) ||
       (native_eot &&
        (continuation || (cs->cr.process_empty_tiles &&
                          (m5_ms_clear ||
                           !agx_get_precompiled(&dev->bg_eot, LIBAGX_FILL_UINT4)))))) {
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

   /* Flush inline occlusion availability before replacing and destroying cs. */
   if (unlikely(cmd->state.oq_pending_render == cs))
      hk_flush_oq_pending(cmd);

   /* Fold whole-image single-sample clear-only render passes into deferred
    * clears so a following pass that loads (or re-clears) the attachment
    * absorbs the clear directly without an extra compute fill kick. */
   if (hk_try_defer_empty_render_clears(cmd, cs)) {
      hk_sync_forget_stream(cmd, cs);
      list_del(&cs->node);
      hk_cs_destroy(cs);
      return true;
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

static bool
hk_deferred_clear_eligible(struct hk_cmd_buffer *cmd,
                           const struct hk_image *image,
                           VkImageAspectFlags aspects, uint32_t range_count,
                           const VkImageSubresourceRange *ranges)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   const VkImageUsageFlags attachment =
      aspects == VK_IMAGE_ASPECT_COLOR_BIT
         ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
         : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

   return agx_chip_native_or_m5(dev->dev.chip) && !cmd->in_meta &&
          cmd->vk.pool->queue_family_index < pdev->queue_family_count &&
          (pdev->queue_families[cmd->vk.pool->queue_family_index].queue_flags &
           VK_QUEUE_GRAPHICS_BIT) &&
          (image->vk.usage & attachment) &&
          aspects == (image->vk.aspects &
                      (aspects == VK_IMAGE_ASPECT_COLOR_BIT
                          ? VK_IMAGE_ASPECT_COLOR_BIT
                          : (VK_IMAGE_ASPECT_DEPTH_BIT |
                             VK_IMAGE_ASPECT_STENCIL_BIT))) &&
          hk_deferred_clear_image_whole(image, range_count, ranges, aspects);
}

static bool
hk_defer_clear_internal(struct hk_cmd_buffer *cmd, struct hk_image *image,
                        VkImageAspectFlags aspects, const VkClearValue *value,
                        uint32_t range_count,
                        const VkImageSubresourceRange *ranges, bool from_gfx,
                        const struct hk_sync_dep *gfx_dep)
{
   if (!hk_deferred_clear_eligible(cmd, image, aspects, range_count, ranges))
      return false;

   /* The deferred clear overwrites the whole image, whether it becomes a
    * load-op clear or a fill: a pending store of those aspects is dead. */
   if (cmd->state.pending_z_store.cs &&
       cmd->state.pending_z_store.image == image)
      hk_kill_pending_z_store_aspects(cmd, aspects);

   /* A later clear of the same image replaces the earlier one. */
   for (unsigned i = 0; i < cmd->state.deferred_clear_count; ++i) {
      struct hk_deferred_clear *e = &cmd->state.deferred_clear[i];
      if (e->image == image) {
         e->aspects = aspects;
         e->value = *value;
         e->from_gfx = from_gfx;
         if (from_gfx && gfx_dep)
            memcpy(e->gfx_dep, gfx_dep, sizeof(e->gfx_dep));
         else
            memset(e->gfx_dep, 0, sizeof(e->gfx_dep));
         return true;
      }
   }

   if (cmd->state.deferred_clear_count == HK_MAX_DEFERRED_CLEARS)
      hk_flush_deferred_clears(cmd);

   struct hk_deferred_clear *slot =
      &cmd->state.deferred_clear[cmd->state.deferred_clear_count++];
   *slot = (struct hk_deferred_clear){
      .image = image,
      .aspects = aspects,
      .value = *value,
      .from_gfx = from_gfx,
   };
   if (from_gfx && gfx_dep)
      memcpy(slot->gfx_dep, gfx_dep, sizeof(slot->gfx_dep));
   return true;
}

bool
hk_defer_clear(struct hk_cmd_buffer *cmd, struct hk_image *image,
               VkImageAspectFlags aspects, const VkClearValue *value,
               uint32_t range_count, const VkImageSubresourceRange *ranges)
{
   return hk_defer_clear_internal(cmd, image, aspects, value, range_count,
                                  ranges, false, NULL);
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
   bool any_xfer = false, any_gfx = false;
   for (unsigned i = 0; i < cmd->state.deferred_clear_count; ++i) {
      const struct hk_deferred_clear *e = &cmd->state.deferred_clear[i];
      if (unlikely(dev->frame_anatomy)) {
         hk_frame_anatomy_note(
            dev,
            "deferred-clear fill by %s: %ux%u %s aspects=0x%x compressed=%u gfx=%u",
            why, e->image->vk.extent.width, e->image->vk.extent.height,
            util_format_short_name(
               hk_format_to_pipe_format(e->image->vk.format)),
            e->aspects, e->image->planes[0].layout.compressed, e->from_gfx);
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

      if (e->from_gfx) {
         hk_cmd_buffer_end_compute(cmd);
         cmd->sync.kind_override = HK_SYNC_GFX;
         hk_clear_image(cmd, e->image, PIPE_FORMAT_NONE, colour, &range, true);
         if (cmd->current_cs.cs) {
            hk_sync_apply_dep(cmd, cmd->current_cs.cs, e->gfx_dep);
            hk_cmd_buffer_end_compute(cmd);
         }
         cmd->sync.kind_override = 0;
         any_gfx = true;
      } else {
         hk_sync_xfer_begin(cmd);
         hk_clear_image(cmd, e->image, PIPE_FORMAT_NONE, colour, &range, true);
         hk_sync_xfer_end(cmd);
         any_xfer = true;
      }
   }
   cmd->state.deferred_clear_count = 0;

   /* The barriers that followed the clears were recorded before the fills;
    * order everything recorded from now on after them, as
    * hk_CmdPipelineBarrier2() does outside a render pass. */
   struct hk_cs *cs = cmd->current_cs.cs;
   const bool alias_scope = hk_cmd_alias_scope(cmd);
   if (cs && hk_cdm_visibility_flush_needed(dev, cs, alias_scope)) {
      hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
      hk_cdm_visibility_flush(dev, cs, alias_scope);
   }
   if (cs && !alias_scope)
      hk_cdm_note_alias_site(cmd, cs);
   if (any_xfer)
      hk_sync_barrier(cmd, HK_SYNC_XFER, HK_SYNC_ALL);
   if (any_gfx)
      hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_ALL);
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
 * take it as its load-op clear (or overwrite it cleanly with its own
 * LOAD_OP_CLEAR in a strict attachment layout); *fill is set when the clear
 * must also be recorded as a fill before the pass, and *overwrite is set when
 * the attachment already specifies LOAD_OP_CLEAR so its own clear value must
 * be preserved while dropping the dead earlier clear. */
static struct hk_deferred_clear *
hk_deferred_clear_for(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                      const VkRenderingAttachmentInfo *att,
                      VkImageAspectFlags aspect, bool *fill, bool *overwrite)
{
   *fill = false;
   *overwrite = false;
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

   const bool load_overwrite =
      att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR &&
      use == HK_DEFERRED_CLEAR_LOAD;
   if (!e || !(e->aspects & aspect) ||
       (att->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD &&
        att->loadOp != VK_ATTACHMENT_LOAD_OP_DONT_CARE && !load_overwrite) ||
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
   *overwrite = load_overwrite;
   if (unlikely(hk_cmd_buffer_device(cmd)->frame_anatomy)) {
      hk_frame_anatomy_note(
         hk_cmd_buffer_device(cmd),
         "deferred-clear %s: %ux%u layout=%u fill=%u",
         load_overwrite ? "overwritten" : "taken", image->vk.extent.width,
         image->vk.extent.height, att->imageLayout, *fill);
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
   bool col_overwrite[HK_MAX_RTS] = {false}, z_overwrite = false,
        s_overwrite = false;
   for (unsigned i = 0; i < info->colorAttachmentCount; ++i) {
      col[i] = hk_deferred_clear_for(cmd, info, &info->pColorAttachments[i],
                                     VK_IMAGE_ASPECT_COLOR_BIT, &col_fill[i],
                                     &col_overwrite[i]);
   }
   struct hk_deferred_clear *z = hk_deferred_clear_for(
      cmd, info, info->pDepthAttachment, VK_IMAGE_ASPECT_DEPTH_BIT, &z_fill,
      &z_overwrite);
   struct hk_deferred_clear *s = hk_deferred_clear_for(
      cmd, info, info->pStencilAttachment, VK_IMAGE_ASPECT_STENCIL_BIT, &s_fill,
      &s_overwrite);

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
      if (col[i] && !col_overwrite[i]) {
         color[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
         color[i].clearValue = col[i]->value;
      }
   }
   copy->pColorAttachments = color;
   if (z && !z_overwrite) {
      *depth = *info->pDepthAttachment;
      depth->loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
      depth->clearValue = z->value;
      copy->pDepthAttachment = depth;
   }
   if (s && !s_overwrite) {
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

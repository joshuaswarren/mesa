/* SPDX-License-Identifier: MIT */
#include <inttypes.h>
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_shader.h"
#include "util/u_cpu_detect.h"
#include "util/u_debug.h"

static struct agx_ptr
hk_pool_root_alloc(struct hk_cmd_buffer *cmd, uint32_t size,
                   uint32_t alignment)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!agx_chip_native_agx3(dev->dev.chip))
      return hk_pool_alloc(cmd, size, alignment);

   return hk_pool_alloc_from(cmd, &cmd->uploader.root, size, alignment,
                             HK_CMD_BO_ROOT);
}

__attribute__((hot)) uint64_t
hk_cmd_buffer_upload_root(struct hk_cmd_buffer *cmd,
                          VkPipelineBindPoint bind_point)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_descriptor_state *desc = hk_get_descriptors_state(cmd, bind_point);
   struct hk_root_descriptor_table *root = &desc->root;

   size_t root_size = sizeof(*root);
   if (agx_chip_native_hal200(dev->dev.chip)) {
      if (bind_point == VK_PIPELINE_BIND_POINT_COMPUTE) {
         const struct hk_shader *s = hk_only_variant(cmd->state.cs.shader);
         if (s && s->root_descriptor_size)
            root_size = MIN2(root_size, s->root_descriptor_size);
      } else if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
         root_size = MIN2(root_size, hk_gfx_root_descriptor_size(cmd));
      }
   }

   root->printf_buffer_address = dev->dev.printf_buffer_address;
   root->sampler_heap_address =
      dev->samplers.table.bo && dev->samplers.table.bo->va
         ? dev->samplers.table.bo->va->addr
         : 0;
   root->physical_sparse_map = dev->physical_sparse_map
      ? dev->physical_sparse_map->va->addr : 0;
   root->physical_sparse_base = dev->dev.main_heap_base;
   root->physical_sparse_size = dev->dev.main_heap_size;

   /* Dynamic buffers past the highest one ever bound are read by no bound
    * layout. Never read the uninitialized hole in an earlier upload. */
   const size_t head_B = MIN2(root_size,
      hk_root_descriptor_offset(dynamic_buffers) +
      desc->dynamic_buffer_end * sizeof(root->dynamic_buffers[0]));
   const size_t tail_B = hk_root_descriptor_offset(set_dynamic_buffer_start);
   /* CPU-decoded descriptor programs use a new root identity to invalidate
    * their native program caches, even when only set memory changed. */
   if (agx_chip_native_agx3(dev->dev.chip) && desc->uploaded_root.gpu &&
       desc->uploaded_root_size == root_size &&
       desc->generation == desc->uploaded_root_generation &&
       desc->dynamic_buffer_end == desc->uploaded_dynamic_buffer_end) {
      root->root_desc_addr = desc->uploaded_root.gpu;
      if (memcmp(desc->uploaded_root.cpu, root, head_B) == 0 &&
          (tail_B >= root_size ||
           memcmp((const uint8_t *)desc->uploaded_root.cpu + tail_B,
                  (const uint8_t *)root + tail_B, root_size - tail_B) == 0)) {
         if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS &&
             cmd->state.gfx.root &&
             desc->uploaded_root.gpu != cmd->state.gfx.root)
            cmd->state.gfx.g17p_bind_tail.valid = false;
         if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
            cmd->state.gfx.root_cpu = desc->uploaded_root.cpu;
            cmd->state.gfx.root_size = root_size;
            if ((agx_chip_native_hal200(dev->dev.chip) ||
                 agx_chip_g17_m5(dev->dev.chip)) &&
                cmd->state.gfx.root_fragment_snapshot_cpu !=
                   desc->uploaded_root.cpu) {
               hk_root_fragment_snapshot_publish(&cmd->state.gfx,
                                                 desc->uploaded_root.cpu, root,
                                                 root_size);
            }
         }
         hk_upload_stat(cmd, root_reuses, 1);
         return desc->uploaded_root.gpu;
      }
   }

   struct agx_ptr root_ptr = hk_pool_root_alloc(cmd, root_size, 8);
   if (!root_ptr.gpu)
      return 0;

   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
      cmd->state.gfx.root_cpu = root_ptr.cpu;
      cmd->state.gfx.root_size = root_size;
   }

   /* A HAL200 tail built for the old root names it and must be rebuilt
    * before publication. Byte-identical roots keep that identity. */
   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS &&
       cmd->state.gfx.root && root_ptr.gpu != cmd->state.gfx.root)
      cmd->state.gfx.g17p_bind_tail.valid = false;
   root->root_desc_addr = root_ptr.gpu;
   /* Capture from the CPU source after every publication-time augmentation.
    * Reading the GPU-mapped root back here would pay uncached load latency. */
   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS &&
       (agx_chip_native_hal200(dev->dev.chip) ||
        agx_chip_g17_m5(dev->dev.chip))) {
      struct hk_graphics_state *gfx = &cmd->state.gfx;
      hk_root_fragment_snapshot_publish(gfx, root_ptr.cpu, root, root_size);
   }

   memcpy(root_ptr.cpu, root, head_B);
   if (tail_B < root_size)
      memcpy((uint8_t *)root_ptr.cpu + tail_B, (const uint8_t *)root + tail_B,
             root_size - tail_B);

   /* Cached G17P roots need explicit CPU-to-GPU publication. Clean just this
    * table now; vkEndCommandBuffer completes all recorded root cleans with a
    * single fence before submission can consume them. */
   if (agx_chip_native_agx3(dev->dev.chip) && util_has_cache_ops()) {
      util_flush_range_no_fence(root_ptr.cpu, head_B);
      if (tail_B < root_size)
         util_flush_range_no_fence((uint8_t *)root_ptr.cpu + tail_B,
                                  root_size - tail_B);
   }

   desc->uploaded_root = root_ptr;
   desc->uploaded_root_size = root_size;
   desc->uploaded_root_generation = desc->generation;
   desc->uploaded_dynamic_buffer_end = desc->dynamic_buffer_end;
   hk_upload_stat(cmd, root_uploads, 1);
   hk_upload_stat(cmd, root_bytes, head_B + (tail_B < root_size ? root_size - tail_B : 0));
   return root_ptr.gpu;
}

uint64_t
hk_cmd_upload_constants(struct hk_cmd_buffer *cmd, const struct hk_shader *s,
                        mesa_shader_stage stage)
{
   if (!s || !s->data_size)
      return 0;

   assert(stage < MESA_SHADER_STAGES);
   const bool native =
      agx_chip_native_agx3(hk_cmd_buffer_device(cmd)->dev.chip);
   if (native && s->constant_data_id &&
       cmd->state.constant_data[stage].id == s->constant_data_id) {
      hk_upload_stat(cmd, constant_reuses, 1);
      return cmd->state.constant_data[stage].addr;
   }

   uint64_t addr = hk_pool_upload(cmd, s->data_ptr, s->data_size,
                                  HK_MIN_UBO_ALIGNMENT);
   if (!addr)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   else {
      cmd->state.constant_data[stage].id = s->constant_data_id;
      cmd->state.constant_data[stage].addr = addr;
      hk_upload_stat(cmd, constant_uploads, 1);
      hk_upload_stat(cmd, constant_bytes, s->data_size);
   }
   return addr;
}

void
hk_cmd_upload_draw_params(struct hk_cmd_buffer *cmd, uint32_t first,
                          uint32_t first_instance)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   uint32_t params[] = {first, first_instance};
   if (agx_chip_native_agx3(hk_cmd_buffer_device(cmd)->dev.chip) &&
       gfx->g17p_draw_params_host_valid && gfx->draw_params &&
       memcmp(gfx->g17p_draw_params_value, params, sizeof(params)) == 0) {
      hk_upload_stat(cmd, draw_param_reuses, 1);
      return;
   }

   const uint64_t addr = hk_pool_upload(cmd, params, sizeof(params), 4);
   gfx->g17p_draw_params_host_valid = addr != 0;
   gfx->draw_params = addr;
   if (!addr)
      return;
   memcpy(gfx->g17p_draw_params_value, params, sizeof(params));
   hk_upload_stat(cmd, draw_param_uploads, 1);
   hk_upload_stat(cmd, draw_param_bytes, sizeof(params));
}

void
hk_cmd_upload_draw_id(struct hk_cmd_buffer *cmd, uint32_t draw_id)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   if (agx_chip_native_agx3(hk_cmd_buffer_device(cmd)->dev.chip) &&
       gfx->g17p_draw_id_host_valid && gfx->draw_id_ptr &&
       gfx->g17p_draw_id_value == draw_id) {
      hk_upload_stat(cmd, draw_id_reuses, 1);
      return;
   }

   gfx->g17p_draw_id_value = draw_id;
   /* Native uniform DMA reads a whole dword, including the high half. */
   gfx->draw_id_ptr = hk_pool_upload(cmd, &draw_id, sizeof(draw_id), 4);
   gfx->g17p_draw_id_host_valid = gfx->draw_id_ptr != 0;
   if (!gfx->draw_id_ptr)
      return;
   hk_upload_stat(cmd, draw_id_uploads, 1);
   hk_upload_stat(cmd, draw_id_bytes, sizeof(draw_id));
}

uint64_t
hk_cmd_upload_group_count(struct hk_cmd_buffer *cmd,
                          VkDispatchIndirectCommand group_count)
{
   const uint32_t hash =
      ((group_count.x * 2654435761u) ^ (group_count.y * 2246822519u) ^
       (group_count.z * 3266489917u)) >>
      28;
   struct hk_group_count_cache_entry *entry = &cmd->group_count_cache[hash];
   if (likely(cmd->used_group_count_cache && entry->valid &&
              entry->x == group_count.x && entry->y == group_count.y &&
              entry->z == group_count.z))
      return entry->addr;

   const uint64_t addr =
      hk_pool_upload(cmd, &group_count, sizeof(group_count), 8);
   if (!addr) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return 0;
   }

   entry->x = group_count.x;
   entry->y = group_count.y;
   entry->z = group_count.z;
   entry->valid = 1;
   entry->addr = addr;
   cmd->used_group_count_cache = true;
   return addr;
}

void
hk_cmd_print_upload_stats(struct hk_cmd_buffer *cmd)
{
   const struct hk_upload_stats *s = &cmd->state.upload_stats;
   perf_debug(cmd, "CPU uploads: draws=%" PRIu64
              " descriptor_binds=%" PRIu64 " descriptor_noops=%" PRIu64
              " root=%" PRIu64 "/%" PRIu64 "B root_reuse=%" PRIu64
              " draw_params=%" PRIu64 "/%" PRIu64 "B draw_param_reuse=%" PRIu64
              " draw_id=%" PRIu64 "/%" PRIu64 "B draw_id_reuse=%" PRIu64
              " constants=%" PRIu64 "/%" PRIu64 "B constant_reuse=%" PRIu64,
              s->draws, s->descriptor_binds, s->descriptor_noops,
              s->root_uploads, s->root_bytes, s->root_reuses,
              s->draw_param_uploads, s->draw_param_bytes, s->draw_param_reuses,
              s->draw_id_uploads, s->draw_id_bytes, s->draw_id_reuses,
              s->constant_uploads, s->constant_bytes, s->constant_reuses);
}

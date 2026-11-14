/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "asahi/compiler/agx_nir_texture.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/u_math.h"
#include "vulkan/vulkan_core.h"
#include "agx_pack.h"
#include "hk_buffer.h"
#include "hk_buffer_view.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_g17p_meta_copy_core.h"
#include "hk_image.h"
#include "hk_physical_device.h"

#include "layout.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_format_convert.h"
#include "shader_enums.h"
#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_meta.h"
#include "vk_pipeline.h"

/* For block based blit kernels, we hardcode the maximum tile size which we can
 * always achieve. This simplifies our life.
 */
#define TILE_WIDTH  32
#define TILE_HEIGHT 32

/* The block-based copy kernel stages the destination tile in the workgroup
 * imageblock (st_tile). G18G has no compute imageblock encoding, so it copies
 * through plain image stores; only non-1D, single-sample images qualify
 * either way. */
static bool
hk_meta_copy_block_based(struct vk_command_buffer *vk_cmd,
                         const struct vk_image *image)
{
   struct hk_cmd_buffer *cmd = container_of(vk_cmd, struct hk_cmd_buffer, vk);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   return dev->dev.chip != AGX_CHIP_G18G &&
          image->image_type != VK_IMAGE_TYPE_1D && image->samples == 1;
}

static VkResult
hk_cmd_bind_map_buffer(struct vk_command_buffer *vk_cmd,
                       struct vk_meta_device *meta, VkBuffer _buffer,
                       void **map_out)
{
   struct hk_cmd_buffer *cmd = container_of(vk_cmd, struct hk_cmd_buffer, vk);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   assert(buffer->vk.size < UINT_MAX);
   struct agx_ptr T = hk_pool_alloc(cmd, buffer->vk.size, 16);
   if (unlikely(T.cpu == NULL))
      return VK_ERROR_OUT_OF_POOL_MEMORY;

   assert(buffer->vk.device_address == 0);
   buffer->vk.device_address = T.gpu;
   *map_out = T.cpu;
   return VK_SUCCESS;
}

VkResult
hk_device_init_meta(struct hk_device *dev)
{
   VkResult result = vk_meta_device_init(&dev->vk, &dev->meta);
   if (result != VK_SUCCESS)
      return result;

   dev->meta.use_gs_for_layer = false;
   dev->meta.use_layer_output = true;
   dev->meta.use_view_relative_3d_layer = agx_native_graphics(&dev->dev);
   dev->meta.specialize_single_layer_buffer_upload = agx_is_m4_pro(&dev->dev);
   /* Other native chips lower a stencil blit to eight bit-selective discard
    * passes. G17P exports stencil in one draw, which also resolves depth and
    * stencil together: 1.50 to 0.48 ms for a 2x D32S8 at 3456x2234. */
   dev->meta.use_stencil_export = !agx_native_graphics(&dev->dev) ||
                                  dev->dev.chip == AGX_CHIP_G17P;
   /* A scaled image is read through an exact integer descriptor or a
    * filterable normalized one depending on whether the sampler filters, so
    * a blit source has to arrive together with its sampler for the driver to
    * choose.  See hk_descriptor_set.c. */
   dev->meta.use_combined_blit_descriptors = agx_native_graphics(&dev->dev);
   dev->meta.use_masked_rgba8_buffer_unpack =
      agx_native_graphics(&dev->dev);
   dev->meta.use_rect_list_pipeline = true;
   dev->meta.cmd_bind_map_buffer = hk_cmd_bind_map_buffer;
   dev->meta.max_bind_map_buffer_size_B = 64 * 1024;
   dev->meta.buffer_access.max_fill_chunk_size =
      agx_native_graphics(&dev->dev) ? 16 : 0;

   for (unsigned i = 0; i < VK_META_BUFFER_CHUNK_SIZE_COUNT; ++i) {
      dev->meta.buffer_access.optimal_wg_size[i] = 64;
   }

   return VK_SUCCESS;
}

void
hk_device_finish_meta(struct hk_device *dev)
{
   vk_meta_device_finish(&dev->vk, &dev->meta);
}

/* A push descriptor set without the bytes its layout does not use */
static void
hk_copy_push_descriptor_set(struct hk_push_descriptor_set *dst,
                            const struct hk_push_descriptor_set *src)
{
   memcpy(dst->layout_hash, src->layout_hash, sizeof(dst->layout_hash));
   dst->layout_flags = src->layout_flags;
   dst->size = src->size;
   memcpy(dst->data, src->data, src->size);
}

void
hk_meta_begin(struct hk_cmd_buffer *cmd, struct hk_meta_save *save,
              VkPipelineBindPoint bind_point)
{
   struct hk_descriptor_state *desc = hk_get_descriptors_state(cmd, bind_point);
   save->in_meta = cmd->in_meta;
   save->pipeline_shader_stages = cmd->vk.pipeline_shader_stages;

   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
      save->dynamic = cmd->vk.dynamic_graphics_state;
      save->_dynamic_vi = cmd->state.gfx._dynamic_vi;
      save->_dynamic_sl = cmd->state.gfx._dynamic_sl;

      static_assert(sizeof(cmd->state.gfx.shaders) == sizeof(save->shaders));
      memcpy(save->shaders, cmd->state.gfx.shaders, sizeof(save->shaders));

      /* Pause queries */
      save->occlusion = cmd->state.gfx.occlusion.mode;
      cmd->state.gfx.occlusion.mode = AGX_VISIBILITY_MODE_NONE;
      cmd->state.gfx.dirty |= HK_DIRTY_OCCLUSION;

      save->pipeline_stats_flags = desc->root.draw.pipeline_stats_flags;
      desc->root.draw.pipeline_stats_flags = 0;
      desc->root_dirty = true;
   } else {
      save->shaders[MESA_SHADER_COMPUTE] = cmd->state.cs.shader;
   }

   save->vb0 = cmd->state.gfx.vb[0];

   save->desc0 = desc->sets[0];
   save->has_push_desc0 = desc->push[0];
   save->push_active0 = !!(desc->push_active & BITFIELD_BIT(0));
   save->buffer_active0 = !!(desc->buffer_active & BITFIELD_BIT(0));
   save->buffer_set0 = desc->root.sets[0];
   save->buffer_set_size0 = desc->set_sizes[0];
   if (save->has_push_desc0)
      hk_copy_push_descriptor_set(&save->push_desc0, desc->push[0]);

   static_assert(sizeof(save->push) == sizeof(desc->root.push));
   memcpy(save->push, desc->root.push, sizeof(save->push));

   cmd->in_meta = true;
}

static void
hk_meta_init_render(struct hk_cmd_buffer *cmd,
                    struct vk_meta_rendering_info *info)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;

   *info = (struct vk_meta_rendering_info){
      .samples = MAX2(render->tilebuffer.nr_samples, 1),
      .view_mask = render->view_mask,
      .color_attachment_count = render->color_att_count,
      .depth_attachment_format = render->depth_att.vk_format,
      .stencil_attachment_format = render->stencil_att.vk_format,
   };
   for (uint32_t a = 0; a < render->color_att_count; a++) {
      info->color_attachment_formats[a] = render->color_att[a].vk_format;
      info->color_attachment_write_masks[a] =
         VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
   }
}

void
hk_meta_end(struct hk_cmd_buffer *cmd, struct hk_meta_save *save,
            VkPipelineBindPoint bind_point)
{
   struct hk_descriptor_state *desc = hk_get_descriptors_state(cmd, bind_point);
   desc->root_dirty = true;

   desc->buffer_active &= ~BITFIELD_BIT(0);
   if (save->buffer_active0) {
      /* Set 0 came from a descriptor buffer: rebind its address. A push
       * descriptor set the application pushed earlier stays inactive. */
      if (save->has_push_desc0)
         hk_copy_push_descriptor_set(desc->push[0], &save->push_desc0);
      desc->sets[0] = NULL;
      desc->root.sets[0] = save->buffer_set0;
      desc->set_sizes[0] = save->buffer_set_size0;
      desc->sets_dirty |= BITFIELD_BIT(0);
      desc->push_dirty &= ~BITFIELD_BIT(0);
      desc->push_active &= ~BITFIELD_BIT(0);
      desc->buffer_active |= BITFIELD_BIT(0);
   } else if (save->desc0) {
      desc->sets[0] = save->desc0;
      desc->root.sets[0] = hk_descriptor_set_addr(save->desc0);
      desc->sets_dirty |= BITFIELD_BIT(0);
      desc->push_dirty &= ~BITFIELD_BIT(0);
      desc->push_active &= ~BITFIELD_BIT(0);
   } else if (save->has_push_desc0) {
      hk_copy_push_descriptor_set(desc->push[0], &save->push_desc0);
      desc->push_dirty |= BITFIELD_BIT(0);
      if (save->push_active0)
         desc->push_active |= BITFIELD_BIT(0);
      else
         desc->push_active &= ~BITFIELD_BIT(0);
   } else {
      desc->push_active &= ~BITFIELD_BIT(0);
   }

   if (bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
      /* Restore the dynamic state */
      cmd->vk.last_graphics_pipeline = NULL;
      assert(save->dynamic.vi == &cmd->state.gfx._dynamic_vi);
      assert(save->dynamic.ms.sample_locations == &cmd->state.gfx._dynamic_sl);
      /* The generation must not go back with the state it describes.
       * Restoring it would let the next meta operation, which repeats the
       * same state changes from the same starting point, reach the very
       * generation an earlier one had: a bind tail prepared for that earlier
       * operation then "matches" with stale viewport/scissor state (two
       * vkCmdClearAttachments rectangles in one dynamic-rendering instance
       * split by a barrier drew the second one through the first's
       * viewport). Everything restored is dirty below anyway. */
      const uint64_t generation = cmd->vk.dynamic_graphics_state.generation;
      cmd->vk.dynamic_graphics_state = save->dynamic;
      cmd->vk.dynamic_graphics_state.generation = generation + 1;
      cmd->state.gfx._dynamic_vi = save->_dynamic_vi;
      cmd->state.gfx._dynamic_sl = save->_dynamic_sl;
      memcpy(cmd->vk.dynamic_graphics_state.dirty,
             cmd->vk.dynamic_graphics_state.set,
             sizeof(cmd->vk.dynamic_graphics_state.set));

      for (uint32_t stage = 0; stage < ARRAY_SIZE(save->shaders); stage++) {
         hk_cmd_bind_graphics_shader(cmd, stage, save->shaders[stage]);
      }

      hk_cmd_bind_vertex_buffer(cmd, 0, save->vb0);

      /* Restore queries */
      cmd->state.gfx.occlusion.mode = save->occlusion;
      cmd->state.gfx.dirty |= HK_DIRTY_OCCLUSION;

      desc->root.draw.pipeline_stats_flags = save->pipeline_stats_flags;
      desc->root_dirty = true;
   } else {
      hk_cmd_bind_compute_shader(cmd, save->shaders[MESA_SHADER_COMPUTE]);
   }

   memcpy(desc->root.push, save->push, sizeof(save->push));
   cmd->in_meta = save->in_meta;

   /* The meta pipeline's bind recorded every stage it bound as bound from a
    * pipeline. The state restored above is the application's, so restore
    * that record too: otherwise the application's next vkCmdBindShadersEXT,
    * even of only the task and mesh stages, unbinds the "pipeline" and with
    * it every graphics shader object restored here. */
   cmd->vk.pipeline_shader_stages = save->pipeline_shader_stages;
}

void
hk_g17p_meta_fill_addr(struct hk_cmd_buffer *cmd, struct hk_cs **target,
                       uint64_t address, uint64_t size, uint32_t value)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!agx_native_graphics(&dev->dev) || !target || !address ||
       ((address | size) & 3)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   /* vk_meta dispatches into current_cs.cs.  Select the caller's exact CDM
    * stream for the duration of this internal dispatch so a post-graphics
    * availability store cannot migrate ahead of its producing VDM stream. */
   struct hk_cs *saved = NULL;
   const bool redirect = target != &cmd->current_cs.cs;
   const bool pinned = cmd->sync.pin_cs;
   if (redirect) {
      saved = cmd->current_cs.cs;
      cmd->current_cs.cs = *target;
      /* The caller chose this exact stream; dependency tracking must not
       * replace it with a later one. */
      cmd->sync.pin_cs = true;
   }

   struct hk_meta_save save;
   hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
   vk_meta_fill_buffer_addr(&cmd->vk, &dev->meta, address, size, value);
   hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
   if (size && cmd->current_cs.cs)
      hk_drm_attachment_set_add(&cmd->current_cs.cs->compute_attachments,
                                address, size);

   if (redirect) {
      *target = cmd->current_cs.cs;
      cmd->current_cs.cs = saved;
      cmd->sync.pin_cs = pinned;
   }
}

enum hk_meta_object_key_type {
   HK_META_OBJECT_KEY_EVENT_WAIT = VK_META_OBJECT_KEY_DRIVER_OFFSET,
   HK_META_OBJECT_KEY_SPARSE_BUFFER_COPY,
   HK_META_OBJECT_KEY_BLOCK_COPY,
   HK_META_OBJECT_KEY_G17P_MIP,
   HK_META_OBJECT_KEY_G17P_BUFFER_FILL_PBE,
   HK_META_OBJECT_KEY_G17P_TILED_UPLOAD,
};

struct hk_meta_event_wait_info {
   uint64_t address;
   uint32_t value;
   uint32_t pad;
};

struct hk_meta_sparse_buffer_copy_key {
   enum hk_meta_object_key_type key_type;
   uint32_t chunk_size;
};

struct hk_meta_sparse_buffer_copy_info {
   uint64_t src_addr;
   uint64_t src_nonresident_addr;
   uint64_t dst_addr;
   uint64_t map_addr;
   uint64_t resource_base;
   uint32_t size;
   uint32_t page_count;
};
static_assert(sizeof(struct hk_meta_sparse_buffer_copy_info) == 48,
              "sparse buffer copy push data must be tightly packed");

static nir_def *
hk_load_sparse_buffer_copy_info(nir_builder *b, unsigned bit_size,
                                size_t offset)
{
   return nir_load_push_constant(b, 1, bit_size, nir_imm_int(b, offset));
}

static nir_shader *
build_sparse_buffer_copy_shader(const struct hk_device *dev,
                                const struct hk_meta_sparse_buffer_copy_key *key)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, NULL, "hk-meta-sparse-buffer-copy");

   b.shader->info.workgroup_size[0] =
      vk_meta_buffer_access_wg_size(&dev->meta, key->chunk_size);
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   assert(util_is_power_of_two_nonzero(key->chunk_size));
   uint32_t bit_size;
   uint32_t components;
   if (key->chunk_size <= 4) {
      bit_size = key->chunk_size * 8;
      components = 1;
   } else {
      bit_size = 32;
      components = key->chunk_size / 4;
   }
   assert(components < NIR_MAX_VEC_COMPONENTS);

   nir_def *global_id = nir_load_global_invocation_id(&b, 32);
   nir_def *offset32 =
      nir_imul_imm(&b, nir_channel(&b, global_id, 0), key->chunk_size);
   nir_def *size = hk_load_sparse_buffer_copy_info(
      &b, 32, offsetof(struct hk_meta_sparse_buffer_copy_info, size));

   nir_push_if(&b, nir_ult(&b, offset32, size));

   nir_def *offset = nir_u2u64(&b, offset32);
   nir_def *src_addr = hk_load_sparse_buffer_copy_info(
      &b, 64, offsetof(struct hk_meta_sparse_buffer_copy_info, src_addr));
   nir_def *src_nonresident_addr = hk_load_sparse_buffer_copy_info(
      &b, 64,
      offsetof(struct hk_meta_sparse_buffer_copy_info, src_nonresident_addr));
   nir_def *dst_addr = hk_load_sparse_buffer_copy_info(
      &b, 64, offsetof(struct hk_meta_sparse_buffer_copy_info, dst_addr));
   nir_def *map_addr = hk_load_sparse_buffer_copy_info(
      &b, 64, offsetof(struct hk_meta_sparse_buffer_copy_info, map_addr));
   nir_def *resource_base = hk_load_sparse_buffer_copy_info(
      &b, 64,
      offsetof(struct hk_meta_sparse_buffer_copy_info, resource_base));
   nir_def *page_count = hk_load_sparse_buffer_copy_info(
      &b, 32, offsetof(struct hk_meta_sparse_buffer_copy_info, page_count));

   static_assert(AIL_PAGESIZE == (1u << 14), "update sparse copy lookup");
   nir_def *access_addr = nir_iadd(&b, src_addr, offset);
   nir_def *page =
      nir_ushr_imm(&b, nir_isub(&b, access_addr, resource_base), 14);
   nir_def *page_in_range =
      nir_ult(&b, page, nir_u2u64(&b, page_count));
   nir_def *word_addr = nir_iadd(
      &b, map_addr,
      nir_imul_imm(&b, nir_ushr_imm(&b, page, 5), sizeof(uint32_t)));
   nir_def *safe_word_addr = nir_bcsel(
      &b, page_in_range, word_addr, nir_imm_int64(&b, AGX_ZERO_PAGE_ADDRESS));
   nir_def *word = nir_load_global(
      &b, 1, 32, safe_word_addr, .align_mul = sizeof(uint32_t),
      .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
   nir_def *resident = nir_iand(
      &b, page_in_range,
      nir_ine_imm(&b,
                  nir_iand_imm(
                     &b, nir_ushr(&b, word,
                                  nir_iand_imm(&b, nir_u2u32(&b, page), 31)),
                     1),
                  0));
   nir_def *load_addr = nir_bcsel(
      &b, resident, access_addr, nir_iadd(&b, src_nonresident_addr, offset));
   nir_def *data = nir_load_global(&b, components, bit_size, load_addr,
                                   .align_mul = bit_size / 8);
   nir_store_global(&b, data, nir_iadd(&b, dst_addr, offset),
                    .align_mul = key->chunk_size);

   nir_pop_if(&b, NULL);
   return b.shader;
}

static VkResult
get_sparse_buffer_copy_pipeline(struct hk_device *dev,
                                const struct hk_meta_sparse_buffer_copy_key *key,
                                VkPipelineLayout *layout_out,
                                VkPipeline *pipeline_out)
{
   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(struct hk_meta_sparse_buffer_copy_info),
   };
   VkResult result = vk_meta_get_pipeline_layout(
      &dev->vk, &dev->meta, NULL, &push_range, key, sizeof(*key), layout_out);
   if (unlikely(result != VK_SUCCESS))
      return result;

   VkPipeline cached =
      vk_meta_lookup_pipeline(&dev->meta, key, sizeof(*key));
   if (cached != VK_NULL_HANDLE) {
      *pipeline_out = cached;
      return VK_SUCCESS;
   }

   const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
      .nir = build_sparse_buffer_copy_shader(dev, key),
   };
   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext = &nir_info,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .pName = "main",
      },
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(
      &dev->vk, &dev->meta, &info, key, sizeof(*key), pipeline_out);
   ralloc_free(nir_info.nir);
   return result;
}

static void
hk_g17p_sparse_buffer_copy(struct hk_cmd_buffer *cmd,
                           struct hk_buffer *src_buffer, uint64_t src_addr,
                           uint64_t dst_addr, VkDeviceSize size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   assert((agx_native_graphics(&dev->dev) ||
           agx_chip_g17_m5(dev->dev.chip)) && src_buffer->sparse_map);

   /* Resident writes use the primary VA while nonresident reads must use the
    * zero-page shadow. Read the live map in the copy shader so a sparse bind
    * submitted after command-buffer recording still affects this copy. */
   uint64_t src_nonresident_addr = agx_rw_addr_to_ro(&dev->dev, src_addr);
   uint64_t align = src_addr | src_nonresident_addr | dst_addr | size;
   assert(align != 0);

   const struct hk_meta_sparse_buffer_copy_key key = {
      .key_type = HK_META_OBJECT_KEY_SPARSE_BUFFER_COPY,
      .chunk_size = 1u << MIN2(4, ffsll(align) - 1),
   };
   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result =
      get_sparse_buffer_copy_pipeline(dev, &key, &layout, &pipeline);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   const struct vk_physical_device *pdev = dev->vk.physical;
   const uint32_t workgroup_size =
      vk_meta_buffer_access_wg_size(&dev->meta, key.chunk_size);
   const uint32_t bytes_per_workgroup = workgroup_size * key.chunk_size;
   const uint32_t max_dispatch_size =
      pdev->properties.maxComputeWorkGroupCount[0] * bytes_per_workgroup;
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkCommandBuffer command_buffer = vk_command_buffer_to_handle(&cmd->vk);

   disp->CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                         pipeline);

   while (size) {
      const struct hk_meta_sparse_buffer_copy_info args = {
         .src_addr = src_addr,
         .src_nonresident_addr = src_nonresident_addr,
         .dst_addr = dst_addr,
         .map_addr = src_buffer->sparse_map->va->addr +
                     sizeof(struct hk_sparse_buffer_map_header),
         .resource_base = src_buffer->va->addr,
         .size = MIN2(size, max_dispatch_size),
         .page_count = src_buffer->sparse_map_entries,
      };
      const uint32_t workgroups =
         DIV_ROUND_UP(args.size, bytes_per_workgroup);

      disp->CmdPushConstants(command_buffer, layout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(args),
                             &args);
      disp->CmdDispatch(command_buffer, workgroups, 1, 1);

      src_addr += args.size;
      src_nonresident_addr += args.size;
      dst_addr += args.size;
      size -= args.size;
   }
}

static nir_shader *
build_event_wait_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, NULL, "hk-meta-event-wait");

   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *address = nir_load_push_constant(
      &b, 1, 64, nir_imm_int(&b, offsetof(struct hk_meta_event_wait_info,
                                          address)));
   nir_def *value = nir_load_push_constant(
      &b, 1, 32, nir_imm_int(&b, offsetof(struct hk_meta_event_wait_info,
                                          value)));
   nir_def *zero = nir_load_push_constant(
      &b, 1, 32,
      nir_imm_int(&b, offsetof(struct hk_meta_event_wait_info, pad)));

   /* Every recovered AGX3 loop is nested beneath an ordinary conditional:
    * cndst(push=1), then the loop's unconditional cndst(push=2), cndlt,
    * backward branch and paired cndend pops.  A bare top-level loop has no
    * native control-flow witness.  The caller supplies a positive SET value
    * and the zero pad, making this guard true while retaining the exact
    * recovered nesting protocol.
    */
   nir_if *valid = nir_push_if(&b, nir_ult(&b, zero, value));
   nir_loop *loop = nir_push_loop(&b);
   nir_def *status = nir_load_global(
      &b, 1, 32, address, .align_mul = sizeof(uint32_t),
      .access = ACCESS_COHERENT | ACCESS_VOLATILE);
   nir_break_if(&b, nir_ieq(&b, status, value));
   nir_pop_loop(&b, loop);
   nir_pop_if(&b, valid);

   return b.shader;
}

static VkResult
get_native_meta_pipeline(struct hk_device *dev,
                         enum hk_meta_object_key_type key, unsigned push_size,
                         nir_shader *(*build_shader)(void),
                         VkPipelineLayout *layout_out,
                         VkPipeline *pipeline_out)
{
   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = push_size,
   };

   VkResult result = vk_meta_get_pipeline_layout(
      &dev->vk, &dev->meta, NULL, push_size ? &push_range : NULL,
      &key, sizeof(key), layout_out);
   if (unlikely(result != VK_SUCCESS))
      return result;

   VkPipeline cached =
      vk_meta_lookup_pipeline(&dev->meta, &key, sizeof(key));
   if (cached != VK_NULL_HANDLE) {
      *pipeline_out = cached;
      return VK_SUCCESS;
   }

   const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
      .nir = build_shader(),
   };
   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext = &nir_info,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .pName = "main",
      },
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(
      &dev->vk, &dev->meta, &info, &key, sizeof(key), pipeline_out);
   ralloc_free(nir_info.nir);
   return result;
}

void
hk_g17p_meta_wait_addr(struct hk_cmd_buffer *cmd, uint64_t address,
                       uint32_t value)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!agx_native_graphics(&dev->dev) || !address || (address & 3)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   struct hk_meta_save save;
   hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);

   VkPipelineLayout layout;
   VkPipeline pipeline;
   VkResult result = get_native_meta_pipeline(
      dev, HK_META_OBJECT_KEY_EVENT_WAIT,
      sizeof(struct hk_meta_event_wait_info), build_event_wait_shader,
      &layout, &pipeline);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(&cmd->vk, result);
      hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      return;
   }

   const struct hk_meta_event_wait_info args = {
      .address = address,
      .value = value,
   };
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkCommandBuffer command_buffer = vk_command_buffer_to_handle(&cmd->vk);

   disp->CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                         pipeline);
   disp->CmdPushConstants(command_buffer, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(args), &args);
   disp->CmdDispatch(command_buffer, 1, 1, 1);

   hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
}

#define BINDING_OUTPUT 0
#define BINDING_INPUT  1

static VkFormat
aspect_format(VkFormat fmt, VkImageAspectFlags aspect)
{
   bool depth = (aspect & VK_IMAGE_ASPECT_DEPTH_BIT);
   bool stencil = (aspect & VK_IMAGE_ASPECT_STENCIL_BIT);

   assert(!(depth && stencil));

   switch (fmt) {
   case VK_FORMAT_D32_SFLOAT:
      return VK_FORMAT_R32_SFLOAT;
   case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return stencil ? VK_FORMAT_R8_UINT : VK_FORMAT_R32_SFLOAT;
   case VK_FORMAT_D16_UNORM:
      return VK_FORMAT_R16_UNORM;
   case VK_FORMAT_S8_UINT:
      return VK_FORMAT_R8_UINT;
   case VK_FORMAT_D16_UNORM_S8_UINT:
      return stencil ? VK_FORMAT_R8_UINT : VK_FORMAT_R16_UNORM;
   default:
      break;
   }

   const struct vk_format_ycbcr_info *ycbcr_info =
      vk_format_get_ycbcr_info(fmt);

   /* A single-plane YCbCr format -- the interleaved 4:2:2 families -- is still
    * a YCbCr format, but its one plane is addressed by COLOR, not PLANE_0, so
    * it has no plane aspect to translate. Answer with the format itself.
    * Falling into the switch reached UNREACHABLE(), which a release build
    * compiles to __builtin_unreachable(): the caller then continued on an
    * undefined VkFormat rather than trapping. */
   if (ycbcr_info && ycbcr_info->n_planes > 1) {
      switch (aspect) {
      case VK_IMAGE_ASPECT_PLANE_0_BIT:
         return ycbcr_info->planes[0].format;
      case VK_IMAGE_ASPECT_PLANE_1_BIT:
         return ycbcr_info->planes[1].format;
      case VK_IMAGE_ASPECT_PLANE_2_BIT:
         return ycbcr_info->planes[2].format;
      default:
         UNREACHABLE("invalid ycbcr aspect");
      }
   }

   return fmt;
}

/*
 * Canonicalize formats to simplify the copies. The returned format must in the
 * same compression class, and should roundtrip lossless (minifloat formats are
 * the unfortunate exception).
 */
static enum pipe_format
canonical_format_pipe(enum pipe_format fmt)
{
   assert(ail_is_valid_pixel_format(fmt));

   if (util_format_is_compressed(fmt)) {
      unsigned size_B = util_format_get_blocksize(fmt);
      assert(size_B == 8 || size_B == 16);

      return size_B == 16 ? PIPE_FORMAT_R32G32B32A32_UINT
                          : PIPE_FORMAT_R32G32_UINT;
   }

#define CASE(x, y) [AGX_CHANNELS_##x] = PIPE_FORMAT_##y
   /* clang-format off */
   static enum pipe_format map[] = {
      CASE(R8,           R8_UINT),
      CASE(R16,          R16_UNORM /* XXX: Hack for Z16 copies */),
      CASE(R8G8,         R8G8_UINT),
      CASE(R5G6B5,       R5G6B5_UNORM),
      CASE(R4G4B4A4,     R4G4B4A4_UNORM),
      CASE(A1R5G5B5,     A1R5G5B5_UNORM),
      CASE(R5G5B5A1,     B5G5R5A1_UNORM),
      CASE(R32,          R32_UINT),
      CASE(R16G16,       R16G16_UINT),
      CASE(R11G11B10,    R11G11B10_FLOAT),
      CASE(R10G10B10A2,  R10G10B10A2_UNORM),
      CASE(R9G9B9E5,     R9G9B9E5_FLOAT),
      CASE(R8G8B8A8,     R8G8B8A8_UINT),
      CASE(R32G32,       R32G32_UINT),
      CASE(R16G16B16A16, R16G16B16A16_UINT),
      CASE(R32G32B32A32, R32G32B32A32_UINT),
   };
   /* clang-format on */
#undef CASE

   enum agx_channels channels = ail_pixel_format[fmt].channels;
   assert(channels < ARRAY_SIZE(map) && "all valid channels handled");
   assert(map[channels] != PIPE_FORMAT_NONE && "all valid channels handled");
   return map[channels];
}

/* An uncompressed view of a blocked image has its extent divided by the block
 * (hk_image_view.c:view_denominator), and that path addresses one layer of one
 * level at a time.  Copies of such an image therefore have to be issued layer
 * by layer.  Block-compressed formats are the familiar case; the single-plane
 * 4:2:2 formats are 2x1-blocked without being compressed, so test the block
 * rather than util_format_is_compressed().
 */
static bool
hk_copy_format_is_blocked(VkFormat fmt)
{
   const enum pipe_format p = hk_format_to_pipe_format(fmt);
   return util_format_get_blockwidth(p) > 1 ||
          util_format_get_blockheight(p) > 1 ||
          util_format_get_blockdepth(p) > 1;
}

static VkFormat
canonical_format(VkFormat fmt)
{
   return vk_format_from_pipe_format(
      canonical_format_pipe(hk_format_to_pipe_format(fmt)));
}

static inline VkFormat hk_meta_bitwise_copy_format(VkFormat format);

static VkFormat
hk_meta_compute_copy_format(struct vk_device *device, VkFormat format)
{
   const struct hk_device *dev = container_of(device, struct hk_device, vk);
   return hk_agx2_packed_float_copy(dev->dev.chip, format)
             ? VK_FORMAT_R32_UINT : canonical_format(format);
}

/*
 * Copy view for a losslessly compressed G17P colour plane.
 *
 * Compressed data is only decodable through a view of the same compression
 * (channel) class: ail_formats_compatible(). G17P's usual packed-word aliases
 * (R32_UINT for any 4-byte format, R16_UINT for any 2-byte format, ...) are
 * not: an RGBA8 image written through an R32_UINT PBE is compressed as one
 * 32-bit channel and decodes as garbage through the application's RGBA8
 * view, and vice versa (magenta sampled sources, white readbacks of
 * normalized stores).
 *
 * Use the canonical view of the image's own channel class, and prefer its
 * integer member where one exists so the copy stays bitwise: R16 and
 * R10G10B10A2 canonicalize to UNORM for AGX2's Z16 copies, but their UINT
 * members are in the same class.
 */
static VkFormat
hk_meta_compressed_copy_format(VkFormat fmt)
{
   const enum pipe_format p_format = hk_format_to_pipe_format(fmt);
   enum pipe_format view = canonical_format_pipe(p_format);

   if (view == PIPE_FORMAT_R16_UNORM)
      view = PIPE_FORMAT_R16_UINT;
   else if (view == PIPE_FORMAT_R10G10B10A2_UNORM)
      view = PIPE_FORMAT_R10G10B10A2_UINT;

   assert(ail_formats_compatible(p_format, view));
   return vk_format_from_pipe_format(view);
}

enum copy_type {
   BUF2IMG,
   IMG2BUF,
   IMG2IMG,
};

struct vk_meta_push_data {
   uint64_t buffer;

   uint32_t row_extent;
   uint32_t slice_or_layer_extent;

   int32_t src_offset_el[4];
   int32_t dst_offset_el[4];
   uint32_t grid_el[3];
} PACKED;

#define get_push(b, name)                                                      \
   nir_load_push_constant(                                                     \
      b, 1, sizeof(((struct vk_meta_push_data *)0)->name) * 8,                 \
      nir_imm_int(b, offsetof(struct vk_meta_push_data, name)))

struct vk_meta_image_copy_key {
   enum vk_meta_object_key_type key_type;
   enum copy_type type;
   enum pipe_format src_format, dst_format;
   unsigned block_size;
   unsigned nr_samples;
   bool block_based;
};

/* Texels each thread of an image-to-buffer copy moves. At one 4-byte texel a
 * thread the copy costs per thread, not per byte (1280x720 RGBA8 on G17P:
 * 72 us; the same bytes as 16-byte texels: 25 us). A thread takes texels a
 * tile width apart, so each store of a SIMD group stays one contiguous run.
 */
static unsigned
copy_texels_per_thread(const struct vk_meta_image_copy_key *key)
{
   return key->type == IMG2BUF && key->block_size <= 4 &&
                key->nr_samples == 1 && !key->block_based
             ? 4
             : 1;
}

static nir_def *
linearize_coords(nir_builder *b, nir_def *coord,
                 const struct vk_meta_image_copy_key *key)
{
   assert(key->nr_samples == 1 && "buffer<-->image copies not multisampled");

   nir_def *row_extent = get_push(b, row_extent);
   nir_def *slice_or_layer_extent = get_push(b, slice_or_layer_extent);
   nir_def *x = nir_channel(b, coord, 0);
   nir_def *y = nir_channel(b, coord, 1);
   nir_def *z_or_layer = nir_channel(b, coord, 2);

   nir_def *v = nir_imul_imm(b, x, key->block_size);

   v = nir_iadd(b, v, nir_imul(b, y, row_extent));
   v = nir_iadd(b, v, nir_imul(b, z_or_layer, slice_or_layer_extent));

   return nir_udiv_imm(b, v, key->block_size);
}

static bool
is_format_native(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R16_UNORM:
      /* TODO: debug me .. why do these fail */
      return false;
   case PIPE_FORMAT_R11G11B10_FLOAT:
   case PIPE_FORMAT_R9G9B9E5_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      return true;
   case PIPE_FORMAT_R5G6B5_UNORM:
   case PIPE_FORMAT_R4G4B4A4_UNORM:
   case PIPE_FORMAT_A1R5G5B5_UNORM:
   case PIPE_FORMAT_B5G5R5A1_UNORM:
      return false;
   default:
      UNREACHABLE("expected canonical");
   }
}

static nir_def *
load_store_formatted(nir_builder *b, nir_def *base, nir_def *index,
                     nir_def *value, enum pipe_format format, bool native_formatted_memory)
{
   if (util_format_is_depth_or_stencil(format))
      format = canonical_format_pipe(format);

   /* G17P now lowers this formatted intrinsic to a native packed
    * conversion plus one raw32-bit transfer. Keep the original compatible
    * compression-class view, without scalar normalize/shift/or expansion. */
   if ((native_formatted_memory && is_format_native(format)) ||
       (!native_formatted_memory && format == PIPE_FORMAT_R10G10B10A2_UNORM)) {
      enum pipe_format isa = ail_pixel_format[format].renderable;
      unsigned isa_size = util_format_get_blocksize(isa);
      unsigned isa_components = util_format_get_blocksize(format) / isa_size;
      unsigned shift = util_logbase2(isa_components);

      if (value) {
         nir_store_agx(b, value, base, index, .format = isa, .base = shift);
      } else {
         return nir_load_agx(b, 4, 32, base, index, .format = isa,
                             .base = shift);
      }
   } else {
      unsigned blocksize_B = util_format_get_blocksize(format);
      nir_def *addr =
         nir_iadd(b, base, nir_imul_imm(b, nir_u2u64(b, index), blocksize_B));

      if (value) {
         /* UINT texel fetch (or the format conversion above) already
          * guarantees each RGBA8 component fits eight bits. Pack without
          * redundant clamps, retaining one aligned 32-bit global store. */
         const unsigned rgba8_bits[] = {8, 8, 8, 8};
         nir_def *raw = !native_formatted_memory &&
                       format == PIPE_FORMAT_R8G8B8A8_UINT
            ? nir_format_pack_uint_unmasked(b, value, rgba8_bits, 4)
            : nir_format_pack_rgba(b, format, value);

         if (blocksize_B <= 4) {
            assert(raw->num_components == 1);
            raw = nir_u2uN(b, raw, blocksize_B * 8);
         } else {
            assert(raw->bit_size == 32);
            raw = nir_trim_vector(b, raw, blocksize_B / 4);
         }

         nir_store_global(b, raw, addr, .align_mul = blocksize_B);
      } else {
         nir_def *raw = nir_load_global(b, DIV_ROUND_UP(blocksize_B, 4),
                                        MIN2(32, blocksize_B * 8), addr,
                                        .align_mul = blocksize_B);

         return nir_format_unpack_rgba(b, raw, format);
      }
   }

   return NULL;
}

static nir_shader *
build_image_copy_shader(const struct vk_meta_image_copy_key *key,
                        bool native_formatted_memory, bool g18g)
{
   nir_builder build =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "hk-meta-copy");

   nir_builder *b = &build;
   b->shader->info.workgroup_size[0] = TILE_WIDTH;
   b->shader->info.workgroup_size[1] = TILE_HEIGHT;

   bool src_is_buf = key->type == BUF2IMG;
   bool dst_is_buf = key->type == IMG2BUF;

   bool msaa = key->nr_samples > 1;
   enum glsl_sampler_dim dim_2d =
      msaa ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D;
   enum glsl_sampler_dim dim_src = src_is_buf ? GLSL_SAMPLER_DIM_BUF : dim_2d;
   enum glsl_sampler_dim dim_dst = dst_is_buf ? GLSL_SAMPLER_DIM_BUF : dim_2d;

   const struct glsl_type *texture_type =
      glsl_sampler_type(dim_src, false, !src_is_buf, GLSL_TYPE_UINT);

   const struct glsl_type *image_type =
      glsl_image_type(dim_dst, !dst_is_buf, GLSL_TYPE_UINT);

   nir_variable *texture =
      nir_variable_create(b->shader, nir_var_uniform, texture_type, "source");
   nir_variable *image =
      nir_variable_create(b->shader, nir_var_image, image_type, "dest");

   image->data.descriptor_set = 0;
   image->data.binding = BINDING_OUTPUT;
   /* Partial tile copies read the destination before preserving its pixels.
    *
    * The block-based variant reads the destination back (the else branch's
    * nir_image_deref_load). Declaring the image write-only makes nir_opt_access
    * stamp ACCESS_NON_READABLE on that load, which the G17P/G18G indexed-image
    * lowering declines by design (agx_nir_lower_g17p_combined_sampler.c,
    * g17p_indexed_image_fetch_shape); the pass is all-or-nothing per shader, so
    * the load kept its AGX2 bindless handle and the packer refused it. */
   image->data.access = key->block_based ? 0 : ACCESS_NON_READABLE;

   /* Multisampled stores use a linear PBE view. AGX2's address lowering
    * needs the texel size to convert the byte layer stride to samples.
    * Leaving the format unspecified makes that division use a zero texel
    * size, corrupting copies spanning multiple array layers. */
   if (msaa)
      image->data.image.format = key->dst_format;

   texture->data.descriptor_set = 0;
   texture->data.binding = BINDING_INPUT;

   /* Grab the offset vectors */
   nir_def *src_offset_el = nir_load_push_constant(
      b, 3, 32,
      nir_imm_int(b, offsetof(struct vk_meta_push_data, src_offset_el)));

   nir_def *dst_offset_el = nir_load_push_constant(
      b, 3, 32,
      nir_imm_int(b, offsetof(struct vk_meta_push_data, dst_offset_el)));

   nir_def *grid_2d_el = nir_load_push_constant(
      b, 2, 32, nir_imm_int(b, offsetof(struct vk_meta_push_data, grid_el)));

   /* We're done setting up variables, do the copy */
   unsigned tpt = copy_texels_per_thread(key);
   nir_def *coord0 = nir_load_global_invocation_id(b, 32);
   if (tpt > 1) {
      /* A group covers tpt tiles along x: wg.x * TILE_WIDTH * tpt + lid.x */
      nir_def *x = nir_channel(b, coord0, 0);
      nir_def *lx = nir_channel(b, nir_load_local_invocation_id(b), 0);
      x = nir_iadd(b, nir_imul_imm(b, nir_isub(b, x, lx), tpt), lx);
      coord0 = nir_vector_insert_imm(b, coord0, x, 0);
   }

   /* The destination format is already canonical, convert to an ISA format */
   enum pipe_format isa_format = PIPE_FORMAT_NONE;
   if (key->block_based) {
      enum pipe_format pipe = canonical_format_pipe(key->dst_format);
      isa_format = ail_pixel_format[pipe].renderable;
      assert(isa_format != PIPE_FORMAT_NONE);
   }

   nir_def *local_offset = nir_imm_int(b, 0);
   nir_def *lid = nir_trim_vector(b, nir_load_local_invocation_id(b), 2);
   lid = nir_u2u16(b, lid);

   nir_def *image_deref = &nir_build_deref_var(b, image)->def;
   nir_def *dst_coord = NULL;

   for (unsigned t = 0; t < tpt; ++t) {
      nir_def *coord =
         t ? nir_iadd(b, coord0, nir_imm_ivec3(b, TILE_WIDTH * t, 0, 0))
           : coord0;
      nir_def *src_coord =
         src_is_buf ? coord : nir_iadd(b, coord, src_offset_el);
      dst_coord = dst_is_buf ? coord : nir_iadd(b, coord, dst_offset_el);

      nir_def *coord_2d_el = nir_trim_vector(b, coord, 2);
      nir_def *in_bounds;
      if (key->block_based) {
         nir_def *offset_in_block_el =
            nir_umod_imm(b, nir_trim_vector(b, dst_offset_el, 2), TILE_WIDTH);

         dst_coord =
            nir_vector_insert_imm(b, nir_isub(b, dst_coord, offset_in_block_el),
                                  nir_channel(b, dst_coord, 2), 2);

         src_coord =
            nir_vector_insert_imm(b, nir_isub(b, src_coord, offset_in_block_el),
                                  nir_channel(b, src_coord, 2), 2);

         in_bounds = nir_uge(b, coord_2d_el, offset_in_block_el);
         in_bounds = nir_iand(
            b, in_bounds,
            nir_ult(b, coord_2d_el, nir_iadd(b, offset_in_block_el, grid_2d_el)));
      } else {
         in_bounds = nir_ult(b, coord_2d_el, grid_2d_el);
      }

      /* Special case handle buffer indexing */
      if (dst_is_buf) {
         assert(!key->block_based);
         dst_coord = linearize_coords(b, dst_coord, key);
      } else if (src_is_buf) {
         src_coord = linearize_coords(b, src_coord, key);
      }

      for (unsigned s = 0; s < key->nr_samples; ++s) {
         nir_def *ms_index = nir_imm_int(b, s);
         nir_def *value1 = NULL, *value2 = NULL;

         nir_push_if(b, nir_ball(b, in_bounds));
         {
            /* Copy formatted texel from texture to storage image */
            nir_deref_instr *deref = nir_build_deref_var(b, texture);

            if (src_is_buf) {
               value1 = load_store_formatted(b, get_push(b, buffer), src_coord,
                                             NULL, key->dst_format, native_formatted_memory);
            } else {
               value1 = nir_txf(b, src_coord, .texture_deref = deref,
                                .ms_index = msaa ? ms_index : NULL);
               nir_def_as_tex(value1)->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;

               /* Munge according to the implicit conversions so we get a bit copy */
               if (key->src_format != key->dst_format) {
                  nir_def *packed =
                     nir_format_pack_rgba(b, key->src_format, value1);

                  value1 = nir_format_unpack_rgba(b, packed, key->dst_format);
               }
            }

            if (dst_is_buf) {
               load_store_formatted(b, get_push(b, buffer), dst_coord, value1,
                                    key->dst_format, native_formatted_memory);
            } else if (!key->block_based) {
               nir_image_deref_store(b, image_deref, nir_pad_vec4(b, dst_coord),
                                     ms_index, value1, nir_imm_int(b, 0),
                                     .image_dim = dim_dst,
                                     .image_array = !dst_is_buf,
                                     /* G18G derives the imgwci format from the
                                      * stored type; this kernel always writes
                                      * the canonical 4x32 uint texel. */
                                     .src_type = g18g ? nir_type_uint
                                                      : nir_type_invalid);
            }
         }
         nir_push_else(b, NULL);
         if (key->block_based) {
            /* Copy back the existing destination content */
            value2 = nir_image_deref_load(b, 4, 32, image_deref,
                                          nir_pad_vec4(b, dst_coord), ms_index,
                                          nir_imm_int(b, 0), .image_dim = dim_dst,
                                          .image_array = !dst_is_buf);
         }
         nir_pop_if(b, NULL);

         if (key->block_based) {
            /* Must define the phi first so we validate. */
            nir_def *phi = nir_if_phi(b, value1, value2);
            nir_def *mask = nir_imm_intN_t(b, 1 << s, 16);

            nir_store_local_pixel_agx(b, phi, mask, lid, .base = 0,
                                      .write_mask = 0xf, .format = isa_format,
                                      .explicit_coord = true);
         }
      }
   }

   if (key->block_based) {
      assert(!dst_is_buf);

      nir_barrier(b, .execution_scope = SCOPE_WORKGROUP);

      nir_push_if(b, nir_ball(b, nir_ieq_imm(b, lid, 0)));
      {
         nir_image_deref_store_block_agx(
            b, image_deref, local_offset, dst_coord, .format = isa_format,
            .image_dim = dim_2d, .image_array = true, .explicit_coord = true);
      }
      nir_pop_if(b, NULL);
      b->shader->info.cs.image_block_size_per_thread_agx =
         util_format_get_blocksize(key->dst_format);
   }

   return b->shader;
}

static VkResult
get_image_copy_descriptor_set_layout(struct vk_device *device,
                                     struct vk_meta_device *meta,
                                     VkDescriptorSetLayout *layout_out,
                                     enum copy_type type)
{
   const char *keys[] = {
      [IMG2BUF] = "vk-meta-copy-image-to-buffer-descriptor-set-layout",
      [BUF2IMG] = "vk-meta-copy-buffer-to-image-descriptor-set-layout",
      [IMG2IMG] = "vk-meta-copy-image-to-image-descriptor-set-layout",
   };

   VkDescriptorSetLayout from_cache = vk_meta_lookup_descriptor_set_layout(
      meta, keys[type], strlen(keys[type]));
   if (from_cache != VK_NULL_HANDLE) {
      *layout_out = from_cache;
      return VK_SUCCESS;
   }

   const VkDescriptorSetLayoutBinding bindings[] = {
      {
         .binding = BINDING_OUTPUT,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
      {
         .binding = BINDING_INPUT,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      },
   };

   const VkDescriptorSetLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = ARRAY_SIZE(bindings),
      .pBindings = bindings,
   };

   return vk_meta_create_descriptor_set_layout(device, meta, &info, keys[type],
                                               strlen(keys[type]), layout_out);
}

static VkResult
get_image_copy_pipeline_layout(struct vk_device *device,
                               struct vk_meta_device *meta,
                               struct vk_meta_image_copy_key *key,
                               VkDescriptorSetLayout set_layout,
                               VkPipelineLayout *layout_out,
                               enum copy_type type)
{
   const char *keys[] = {
      [IMG2BUF] = "vk-meta-copy-image-to-buffer-pipeline-layout",
      [BUF2IMG] = "vk-meta-copy-buffer-to-image-pipeline-layout",
      [IMG2IMG] = "vk-meta-copy-image-to-image-pipeline-layout",
   };

   VkPipelineLayout from_cache =
      vk_meta_lookup_pipeline_layout(meta, keys[type], strlen(keys[type]));
   if (from_cache != VK_NULL_HANDLE) {
      *layout_out = from_cache;
      return VK_SUCCESS;
   }

   VkPipelineLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
   };

   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(struct vk_meta_push_data),
   };

   info.pushConstantRangeCount = 1;
   info.pPushConstantRanges = &push_range;

   return vk_meta_create_pipeline_layout(device, meta, &info, keys[type],
                                         strlen(keys[type]), layout_out);
}

static VkResult
get_image_copy_pipeline(struct vk_device *device, struct vk_meta_device *meta,
                        const struct vk_meta_image_copy_key *key,
                        VkPipelineLayout layout, VkPipeline *pipeline_out)
{
   VkPipeline from_cache = vk_meta_lookup_pipeline(meta, key, sizeof(*key));
   if (from_cache != VK_NULL_HANDLE) {
      *pipeline_out = from_cache;
      return VK_SUCCESS;
   }

   const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
      /* Native LDDEV/STDEV transfer raw packed bytes; they do not implement
       * AGX2's component-wise formatted conversion from four 32-bit values.
       * The existing NIR pack/unpack path retains a single aligned raw store.
       * Meta pipelines are cached per device, so no extra key is needed. */
      .nir = build_image_copy_shader(key,
         !agx_native_graphics(&container_of(device, struct hk_device, vk)->dev) &&
         !agx_chip_g17_m5(container_of(device, struct hk_device, vk)->dev.chip) &&
         container_of(device, struct hk_device, vk)->dev.chip != AGX_CHIP_G16G &&
         /* G18G (AGX3) stores raw packed bytes too. */
         container_of(device, struct hk_device, vk)->dev.chip != AGX_CHIP_G18G,
         container_of(device, struct hk_device, vk)->dev.chip ==
            AGX_CHIP_G18G),
   };
   const VkPipelineShaderStageCreateInfo cs_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = &nir_info,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .pName = "main",
   };

   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = cs_info,
      .layout = layout,
   };

   VkResult result = vk_meta_create_compute_pipeline(
      device, meta, &info, key, sizeof(*key), pipeline_out);
   ralloc_free(nir_info.nir);

   return result;
}

static void
hk_meta_copy_image_to_buffer2(struct vk_command_buffer *cmd,
                              struct vk_meta_device *meta,
                              const VkCopyImageToBufferInfo2 *pCopyBufferInfo)
{
   VK_FROM_HANDLE(vk_image, image, pCopyBufferInfo->srcImage);
   VK_FROM_HANDLE(vk_image, src_image, pCopyBufferInfo->srcImage);
   VK_FROM_HANDLE(hk_buffer, buffer, pCopyBufferInfo->dstBuffer);

   struct vk_device *device = cmd->base.device;
   const struct vk_device_dispatch_table *disp = &device->dispatch_table;

   VkResult result;

   VkDescriptorSetLayout set_layout;
   result =
      get_image_copy_descriptor_set_layout(device, meta, &set_layout, IMG2BUF);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(cmd, result);
      return;
   }

   bool per_layer = hk_copy_format_is_blocked(image->format);

   for (unsigned i = 0; i < pCopyBufferInfo->regionCount; ++i) {
      const VkBufferImageCopy2 *region = &pCopyBufferInfo->pRegions[i];

      unsigned layers = MAX2(region->imageExtent.depth,
                             vk_image_subresource_layer_count(
                                src_image, &region->imageSubresource));
      unsigned layer_iters = per_layer ? layers : 1;

      for (unsigned layer_offs = 0; layer_offs < layer_iters; ++layer_offs) {

         VkImageAspectFlags aspect = region->imageSubresource.aspectMask;
         VkFormat aspect_fmt = aspect_format(image->format, aspect);
         /* M5 reads uncompressed colour through the same integer alias as
          * G17P: a canonical R9G9B9E5_FLOAT (or UNORM) view decodes and
          * re-encodes, which renormalizes shared-exponent texels and is not
          * the bit-exact copy Vulkan specifies (queue_transfer e5b9g9r9). */
         const enum agx_chip chip =
            container_of(device, struct hk_device, vk)->dev.chip;
         const bool g17p_color =
            (agx_native_graphics(
                &container_of(device, struct hk_device, vk)->dev) ||
             agx_chip_g17_m5(chip)) &&
            !vk_format_is_depth_or_stencil(image->format);
         const struct hk_image *hk_image =
            container_of(image, struct hk_image, vk);
         const unsigned plane = hk_image_aspects_to_plane(hk_image, aspect);
         /* Lossless compression is channel-layout dependent. The packed-word
          * alias used for plain G17P copies is incompatible with, for example,
          * a compressed RGBA8 image. Preserve its compression class. */
         VkFormat canonical =
            !g17p_color ? hk_meta_compute_copy_format(device, aspect_fmt)
            : hk_image->planes[plane].layout.compressed
               ? hk_meta_compressed_copy_format(aspect_fmt)
               : hk_meta_bitwise_copy_format(aspect_fmt);

         uint32_t blocksize_B =
            util_format_get_blocksize(hk_format_to_pipe_format(canonical));

         enum pipe_format p_format = hk_format_to_pipe_format(image->format);

         unsigned row_extent = util_format_get_nblocksx(
                                  p_format, MAX2(region->bufferRowLength,
                                                 region->imageExtent.width)) *
                               blocksize_B;
         unsigned slice_extent =
            util_format_get_nblocksy(
               p_format,
               MAX2(region->bufferImageHeight, region->imageExtent.height)) *
            row_extent;
         unsigned layer_extent =
            util_format_get_nblocksz(p_format, region->imageExtent.depth) *
            slice_extent;

         bool is_3d = region->imageExtent.depth > 1;

         struct vk_meta_image_copy_key key = {
            .key_type = VK_META_OBJECT_KEY_COPY_IMAGE_TO_BUFFER,
            .type = IMG2BUF,
            .block_size = blocksize_B,
            .nr_samples = image->samples,
            .src_format = hk_format_to_pipe_format(canonical),
            .dst_format = hk_format_to_pipe_format(canonical),
         };

         VkPipelineLayout pipeline_layout;
         result = get_image_copy_pipeline_layout(device, meta, &key, set_layout,
                                                 &pipeline_layout, false);
         if (unlikely(result != VK_SUCCESS)) {
            vk_command_buffer_set_error(cmd, result);
            return;
         }

         VkImageView src_view;
         const VkImageViewUsageCreateInfo src_view_usage = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
         };
         const VkImageViewCreateInfo src_view_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
            .pNext = &src_view_usage,
            .image = pCopyBufferInfo->srcImage,
            .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            .format = canonical,
            .subresourceRange =
               {
                  .aspectMask = region->imageSubresource.aspectMask,
                  .baseMipLevel = region->imageSubresource.mipLevel,
                  .baseArrayLayer =
                     MAX2(region->imageOffset.z,
                          region->imageSubresource.baseArrayLayer) +
                     layer_offs,
                  .layerCount = per_layer ? 1 : layers,
                  .levelCount = 1,
               },
         };

         result =
            vk_meta_create_image_view(cmd, meta, &src_view_info, &src_view);
         if (unlikely(result != VK_SUCCESS)) {
            vk_command_buffer_set_error(cmd, result);
            return;
         }

         VkDescriptorImageInfo src_info = {
            .imageLayout = pCopyBufferInfo->srcImageLayout,
            .imageView = src_view,
         };

         VkWriteDescriptorSet desc_write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = 0,
            .dstBinding = BINDING_INPUT,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = 1,
            .pImageInfo = &src_info,
         };

         disp->CmdPushDescriptorSetKHR(vk_command_buffer_to_handle(cmd),
                                       VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline_layout, 0, 1, &desc_write);

         VkPipeline pipeline;
         result = get_image_copy_pipeline(device, meta, &key, pipeline_layout,
                                          &pipeline);
         if (unlikely(result != VK_SUCCESS)) {
            /* The only reason an image<->buffer copy fails with nothing else
             * logged: the internal copy shader samples its source, and if that
             * texture op has no grounded encoding the pipeline never builds.
             * Without this the app just sees the command buffer's deferred
             * error at vkEndCommandBuffer, with nothing naming the site.
             * AGX_MESA_DEBUG=shaders prints the refused opcode. */
            mesa_loge("meta image-copy pipeline unavailable (%d)", (int)result);
            vk_command_buffer_set_error(cmd, result);
            return;
         }

         disp->CmdBindPipeline(vk_command_buffer_to_handle(cmd),
                               VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

         enum pipe_format p_src_fmt =
            hk_format_to_pipe_format(src_image->format);

         struct vk_meta_push_data push = {
            .buffer = hk_buffer_address_rw(buffer, region->bufferOffset),
            .row_extent = row_extent,
            .slice_or_layer_extent = is_3d ? slice_extent : layer_extent,

            .src_offset_el[0] =
               util_format_get_nblocksx(p_src_fmt, region->imageOffset.x),
            .src_offset_el[1] =
               util_format_get_nblocksy(p_src_fmt, region->imageOffset.y),

            .grid_el[0] =
               util_format_get_nblocksx(p_format, region->imageExtent.width),
            .grid_el[1] =
               util_format_get_nblocksy(p_format, region->imageExtent.height),
            .grid_el[2] = per_layer ? 1 : layers,
         };

         push.buffer += push.slice_or_layer_extent * layer_offs;

         disp->CmdPushConstants(vk_command_buffer_to_handle(cmd),
                                pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                sizeof(push), &push);

         disp->CmdDispatch(vk_command_buffer_to_handle(cmd),
                           DIV_ROUND_UP(push.grid_el[0],
                                        32 * copy_texels_per_thread(&key)),
                           DIV_ROUND_UP(push.grid_el[1], 32), push.grid_el[2]);
      }
   }
}

static void
hk_meta_dispatch_to_image(struct vk_command_buffer *cmd,
                          const struct vk_device_dispatch_table *disp,
                          VkPipelineLayout pipeline_layout,
                          struct vk_meta_push_data *push, VkOffset3D offset,
                          VkExtent3D extent, bool per_layer, unsigned layers,
                          enum pipe_format p_dst_fmt, enum pipe_format p_format,
                          bool block_based)
{
   push->dst_offset_el[0] = util_format_get_nblocksx(p_dst_fmt, offset.x);
   push->dst_offset_el[1] = util_format_get_nblocksy(p_dst_fmt, offset.y);
   push->dst_offset_el[2] = 0;

   push->grid_el[0] = util_format_get_nblocksx(p_format, extent.width);
   push->grid_el[1] = util_format_get_nblocksy(p_format, extent.height);
   push->grid_el[2] = per_layer ? 1 : layers;

   if (!block_based &&
       hk_cmd_buffer_device(container_of(cmd, struct hk_cmd_buffer, vk))
             ->dev.chip == AGX_CHIP_G18G) {
      /* Plain image stores index the destination by global invocation id plus
       * dst_offset_el in the kernel: no tile expansion, just cover the grid. */
      disp->CmdPushConstants(vk_command_buffer_to_handle(cmd), pipeline_layout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*push),
                             push);
      disp->CmdDispatch(vk_command_buffer_to_handle(cmd),
                        DIV_ROUND_UP(push->grid_el[0], TILE_WIDTH),
                        DIV_ROUND_UP(push->grid_el[1], TILE_HEIGHT),
                        push->grid_el[2]);
      return;
   }

   unsigned w_el = util_format_get_nblocksx(p_format, extent.width);
   unsigned h_el = util_format_get_nblocksy(p_format, extent.height);

   /* Expand the grid so destinations are in tiles */
   unsigned expanded_x0 = push->dst_offset_el[0] & ~(TILE_WIDTH - 1);
   unsigned expanded_y0 = push->dst_offset_el[1] & ~(TILE_HEIGHT - 1);
   unsigned expanded_x1 = align(push->dst_offset_el[0] + w_el, TILE_WIDTH);
   unsigned expanded_y1 = align(push->dst_offset_el[1] + h_el, TILE_HEIGHT);

   /* TODO: clamp to the destination size to save some redundant threads? */

   disp->CmdPushConstants(vk_command_buffer_to_handle(cmd), pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*push), push);

   disp->CmdDispatch(vk_command_buffer_to_handle(cmd),
                     (expanded_x1 - expanded_x0) / TILE_WIDTH,
                     (expanded_y1 - expanded_y0) / TILE_HEIGHT,
                     push->grid_el[2]);
}

static void
hk_meta_copy_buffer_to_image2(struct vk_command_buffer *cmd,
                              struct vk_meta_device *meta,
                              const struct VkCopyBufferToImageInfo2 *info)
{
   VK_FROM_HANDLE(vk_image, image, info->dstImage);
   VK_FROM_HANDLE(hk_buffer, buffer, info->srcBuffer);

   struct vk_device *device = cmd->base.device;
   const struct vk_device_dispatch_table *disp = &device->dispatch_table;

   VkDescriptorSetLayout set_layout;
   VkResult result =
      get_image_copy_descriptor_set_layout(device, meta, &set_layout, BUF2IMG);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(cmd, result);
      return;
   }

   bool per_layer = hk_copy_format_is_blocked(image->format);

   for (unsigned r = 0; r < info->regionCount; ++r) {
      const VkBufferImageCopy2 *region = &info->pRegions[r];

      unsigned layers = MAX2(
         region->imageExtent.depth,
         vk_image_subresource_layer_count(image, &region->imageSubresource));
      unsigned layer_iters = per_layer ? layers : 1;

      for (unsigned layer_offs = 0; layer_offs < layer_iters; ++layer_offs) {
         VkImageAspectFlags aspect = region->imageSubresource.aspectMask;
         VkFormat aspect_fmt = aspect_format(image->format, aspect);
         VkFormat canonical = hk_meta_compute_copy_format(device, aspect_fmt);
         enum pipe_format p_format = hk_format_to_pipe_format(aspect_fmt);
         uint32_t blocksize_B = util_format_get_blocksize(p_format);
         bool is_3d = region->imageExtent.depth > 1;

         struct vk_meta_image_copy_key key = {
            .key_type = VK_META_OBJECT_KEY_COPY_IMAGE_TO_BUFFER,
            .type = BUF2IMG,
            .block_size = blocksize_B,
            .nr_samples = image->samples,
            .src_format = hk_format_to_pipe_format(canonical),
            .dst_format = hk_format_to_pipe_format(canonical),

            /* TODO: MSAA path */
            .block_based = hk_meta_copy_block_based(cmd, image),
         };

         VkPipelineLayout pipeline_layout;
         result = get_image_copy_pipeline_layout(device, meta, &key, set_layout,
                                                 &pipeline_layout, true);
         if (unlikely(result != VK_SUCCESS)) {
            vk_command_buffer_set_error(cmd, result);
            return;
         }

         unsigned row_extent = util_format_get_nblocksx(
                                  p_format, MAX2(region->bufferRowLength,
                                                 region->imageExtent.width)) *
                               blocksize_B;
         unsigned slice_extent =
            util_format_get_nblocksy(
               p_format,
               MAX2(region->bufferImageHeight, region->imageExtent.height)) *
            row_extent;
         unsigned layer_extent =
            util_format_get_nblocksz(p_format, region->imageExtent.depth) *
            slice_extent;

         VkImageView dst_view;
         const VkImageViewUsageCreateInfo dst_view_usage = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
            .usage = VK_IMAGE_USAGE_STORAGE_BIT,
         };
         const VkImageViewCreateInfo dst_view_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
            .pNext = &dst_view_usage,
            .image = info->dstImage,
            .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            .format = canonical,
            .subresourceRange =
               {
                  .aspectMask = region->imageSubresource.aspectMask,
                  .baseMipLevel = region->imageSubresource.mipLevel,
                  .baseArrayLayer =
                     MAX2(region->imageOffset.z,
                          region->imageSubresource.baseArrayLayer) +
                     layer_offs,
                  .layerCount = per_layer ? 1 : layers,
                  .levelCount = 1,
               },
         };

         result =
            vk_meta_create_image_view(cmd, meta, &dst_view_info, &dst_view);
         if (unlikely(result != VK_SUCCESS)) {
            vk_command_buffer_set_error(cmd, result);
            return;
         }

         const VkDescriptorImageInfo dst_info = {
            .imageView = dst_view,
            .imageLayout = info->dstImageLayout,
         };

         VkWriteDescriptorSet desc_write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = 0,
            .dstBinding = BINDING_OUTPUT,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .pImageInfo = &dst_info,
         };

         disp->CmdPushDescriptorSetKHR(vk_command_buffer_to_handle(cmd),
                                       VK_PIPELINE_BIND_POINT_COMPUTE,
                                       pipeline_layout, 0, 1, &desc_write);

         VkPipeline pipeline;
         result = get_image_copy_pipeline(device, meta, &key, pipeline_layout,
                                          &pipeline);
         if (unlikely(result != VK_SUCCESS)) {
            /* The only reason an image<->buffer copy fails with nothing else
             * logged: the internal copy shader samples its source, and if that
             * texture op has no grounded encoding the pipeline never builds.
             * Without this the app just sees the command buffer's deferred
             * error at vkEndCommandBuffer, with nothing naming the site.
             * AGX_MESA_DEBUG=shaders prints the refused opcode. */
            mesa_loge("meta image-copy pipeline unavailable (%d)", (int)result);
            vk_command_buffer_set_error(cmd, result);
            return;
         }

         disp->CmdBindPipeline(vk_command_buffer_to_handle(cmd),
                               VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

         struct vk_meta_push_data push = {
            .buffer = hk_buffer_address_ro(buffer, region->bufferOffset),
            .row_extent = row_extent,
            .slice_or_layer_extent = is_3d ? slice_extent : layer_extent,
         };

         push.buffer += push.slice_or_layer_extent * layer_offs;

         hk_meta_dispatch_to_image(cmd, disp, pipeline_layout, &push,
                                   region->imageOffset, region->imageExtent,
                                   per_layer, layers, p_format, p_format,
                                   key.block_based);
      }
   }
}

static void
hk_meta_copy_image2(struct vk_command_buffer *cmd, struct vk_meta_device *meta,
                    const struct VkCopyImageInfo2 *info)
{
   VK_FROM_HANDLE(vk_image, src_image, info->srcImage);
   VK_FROM_HANDLE(vk_image, dst_image, info->dstImage);

   struct vk_device *device = cmd->base.device;
   const struct vk_device_dispatch_table *disp = &device->dispatch_table;

   VkDescriptorSetLayout set_layout;
   VkResult result =
      get_image_copy_descriptor_set_layout(device, meta, &set_layout, BUF2IMG);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(cmd, result);
      return;
   }

   bool per_layer = hk_copy_format_is_blocked(src_image->format) ||
                    hk_copy_format_is_blocked(dst_image->format);

   for (unsigned r = 0; r < info->regionCount; ++r) {
      const VkImageCopy2 *region = &info->pRegions[r];

      unsigned layers = MAX2(
         vk_image_subresource_layer_count(src_image, &region->srcSubresource),
         region->extent.depth);
      unsigned layer_iters = per_layer ? layers : 1;

      for (unsigned layer_offs = 0; layer_offs < layer_iters; ++layer_offs) {
         u_foreach_bit(aspect, region->srcSubresource.aspectMask) {
            /* We use the source format throughout for consistent scaling with
             * compressed<-->uncompressed copies, where the extents are defined
             * to follow the source.
             */
            VkFormat aspect_fmt = aspect_format(src_image->format, 1 << aspect);
            VkFormat canonical = hk_meta_compute_copy_format(device, aspect_fmt);
            uint32_t blocksize_B =
               util_format_get_blocksize(hk_format_to_pipe_format(canonical));

            bool use_dst_aspect_mask =
               vk_format_get_ycbcr_info(dst_image->format) ||
               vk_format_get_ycbcr_info(src_image->format) ||
               (vk_format_is_depth_or_stencil(dst_image->format) &&
                !vk_format_is_depth_or_stencil(src_image->format));

            VkImageAspectFlagBits dst_aspect_mask =
               use_dst_aspect_mask ? region->dstSubresource.aspectMask
                                   : (1 << aspect);

            struct vk_meta_image_copy_key key = {
               .key_type = VK_META_OBJECT_KEY_COPY_IMAGE_TO_BUFFER,
               .type = IMG2IMG,
               .block_size = blocksize_B,
               .nr_samples = dst_image->samples,
               .src_format = hk_format_to_pipe_format(canonical),
               .dst_format = hk_format_to_pipe_format(hk_meta_compute_copy_format(
                  device, aspect_format(dst_image->format, dst_aspect_mask))),

               /* TODO: MSAA path */
               .block_based = hk_meta_copy_block_based(cmd, dst_image),
            };

            assert(key.nr_samples == src_image->samples);

            VkPipelineLayout pipeline_layout;
            result = get_image_copy_pipeline_layout(
               device, meta, &key, set_layout, &pipeline_layout, true);
            if (unlikely(result != VK_SUCCESS)) {
               vk_command_buffer_set_error(cmd, result);
               return;
            }

            VkWriteDescriptorSet desc_writes[2];

            VkImageView src_view;
            const VkImageViewUsageCreateInfo src_view_usage = {
               .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
               .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
            };
            const VkImageViewCreateInfo src_view_info = {
               .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
               .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
               .pNext = &src_view_usage,
               .image = info->srcImage,
               .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
               .format = canonical,
               .subresourceRange =
                  {
                     .aspectMask =
                        region->srcSubresource.aspectMask & (1 << aspect),
                     .baseMipLevel = region->srcSubresource.mipLevel,
                     .baseArrayLayer =
                        MAX2(region->srcOffset.z,
                             region->srcSubresource.baseArrayLayer) +
                        layer_offs,
                     .layerCount = per_layer ? 1 : layers,
                     .levelCount = 1,
                  },
            };

            result =
               vk_meta_create_image_view(cmd, meta, &src_view_info, &src_view);
            if (unlikely(result != VK_SUCCESS)) {
               vk_command_buffer_set_error(cmd, result);
               return;
            }

            VkDescriptorImageInfo src_info = {
               .imageLayout = info->srcImageLayout,
               .imageView = src_view,
            };

            VkImageView dst_view;
            const VkImageViewUsageCreateInfo dst_view_usage = {
               .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
               .usage = VK_IMAGE_USAGE_STORAGE_BIT,
            };
            const VkImageViewCreateInfo dst_view_info = {
               .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
               .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
               .pNext = &dst_view_usage,
               .image = info->dstImage,
               .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
               .format = vk_format_from_pipe_format(key.dst_format),
               .subresourceRange =
                  {
                     .aspectMask = dst_aspect_mask,
                     .baseMipLevel = region->dstSubresource.mipLevel,
                     .baseArrayLayer =
                        MAX2(region->dstOffset.z,
                             region->dstSubresource.baseArrayLayer) +
                        layer_offs,
                     .layerCount = per_layer ? 1 : layers,
                     .levelCount = 1,
                  },
            };

            result =
               vk_meta_create_image_view(cmd, meta, &dst_view_info, &dst_view);
            if (unlikely(result != VK_SUCCESS)) {
               vk_command_buffer_set_error(cmd, result);
               return;
            }

            const VkDescriptorImageInfo dst_info = {
               .imageView = dst_view,
               .imageLayout = info->dstImageLayout,
            };

            desc_writes[0] = (VkWriteDescriptorSet){
               .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               .dstSet = 0,
               .dstBinding = BINDING_OUTPUT,
               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
               .descriptorCount = 1,
               .pImageInfo = &dst_info,
            };

            desc_writes[1] = (VkWriteDescriptorSet){
               .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               .dstSet = 0,
               .dstBinding = BINDING_INPUT,
               .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
               .descriptorCount = 1,
               .pImageInfo = &src_info,
            };

            disp->CmdPushDescriptorSetKHR(
               vk_command_buffer_to_handle(cmd), VK_PIPELINE_BIND_POINT_COMPUTE,
               pipeline_layout, 0, ARRAY_SIZE(desc_writes), desc_writes);

            VkPipeline pipeline;
            result = get_image_copy_pipeline(device, meta, &key,
                                             pipeline_layout, &pipeline);
            if (unlikely(result != VK_SUCCESS)) {
               vk_command_buffer_set_error(cmd, result);
               return;
            }

            disp->CmdBindPipeline(vk_command_buffer_to_handle(cmd),
                                  VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

            enum pipe_format p_src_fmt =
               hk_format_to_pipe_format(src_image->format);
            enum pipe_format p_dst_fmt =
               hk_format_to_pipe_format(dst_image->format);
            enum pipe_format p_format = hk_format_to_pipe_format(aspect_fmt);

            struct vk_meta_push_data push = {
               .src_offset_el[0] =
                  util_format_get_nblocksx(p_src_fmt, region->srcOffset.x),
               .src_offset_el[1] =
                  util_format_get_nblocksy(p_src_fmt, region->srcOffset.y),
            };

            hk_meta_dispatch_to_image(cmd, disp, pipeline_layout, &push,
                                      region->dstOffset, region->extent,
                                      per_layer, layers, p_dst_fmt, p_format,
                                      key.block_based);
         }
      }
   }
}

/* Copy through integer views to preserve every bit, including normalized
 * channel precision, floating-point NaN payloads and YCbCr padding bits.
 * Compressed images use one integer texel per 8- or 16-byte block.
 */
static inline VkFormat
hk_meta_bitwise_copy_format(VkFormat format)
{
   switch (vk_format_get_blocksize(format)) {
   case 1:
      return VK_FORMAT_R8_UINT;
   case 2:
      return VK_FORMAT_R16_UINT;
   case 4:
      return VK_FORMAT_R32_UINT;
   case 8:
      return VK_FORMAT_R32G32_UINT;
   case 16:
      return VK_FORMAT_R32G32B32A32_UINT;
   default:
      return format;
   }
}

static inline struct vk_meta_copy_image_properties
hk_meta_copy_get_image_properties(struct hk_image *img, struct hk_device *dev,
                                  bool g17p, bool msaa_depth_stencil_aspect_views)
{
   const bool agx2 =
      agx_chip_isa_encoding(dev->dev.chip) == AGX_ISA_ENCODING_AGX2;
   /* A split depth/stencil image has separate depth and stencil planes. The
    * interleaved R32G32_UINT alias below is a colour view, which selects
    * image plane 0 (the depth plane) for both aspects, so on M5 copy each
    * plane through its own raw alias at every sample count. So does a
    * compressed depth plane (2x on M4 Pro): the alias is a different
    * compression class. */
   const bool aspect_views =
      g17p && hk_split_depth_stencil(img->vk.format) &&
      (img->vk.samples == VK_SAMPLE_COUNT_1_BIT ||
       msaa_depth_stencil_aspect_views || img->planes[0].layout.compressed);
   /* Keep the existing multisample render-target formats: G17P's native
    * multisample tile path does not support these integer aliases yet. */
   g17p &= img->vk.samples == VK_SAMPLE_COUNT_1_BIT;
   struct vk_meta_copy_image_properties props = {
      .tile_size = {16, 16, 1},
      /* D32/S8 and D16/S8 are separate allocations on AGX2 too. Selecting
       * COLOR instead of the original aspect would view the depth plane as
       * interleaved RG32 data and lose the stencil plane altogether. */
      .depth_stencil_aspect_views =
         aspect_views || (agx2 && hk_split_depth_stencil(img->vk.format)),
   };
   const struct vk_format_ycbcr_info *ycbcr =
      vk_format_get_ycbcr_info(img->vk.format);

   if (ycbcr && ycbcr->n_planes > 1) {
      assert(ycbcr->n_planes <= ARRAY_SIZE(props.plane));
      for (uint32_t plane = 0; plane < ycbcr->n_planes; ++plane) {
         const VkFormat format = ycbcr->planes[plane].format;
         props.plane[plane].view_format =
            g17p ? hk_meta_bitwise_copy_format(format) : format;
      }
   } else if (!vk_format_is_depth_or_stencil(img->vk.format)) {
      /* vk_meta_copy_image_properties documents that view_format "must not be
       * compressed": the common copy shaders convert texels with
       * nir_format_{pack,unpack}_rgba, which accept only UTIL_FORMAT_LAYOUT_PLAIN
       * and abort on a block-compressed description.  A compressed image is
       * therefore copied through its uncompressed block-sized twin, which
       * hk_image_view already builds (it divides the extent by the block and
       * offsets into the level), and which makes the copy the bit-exact one
       * Vulkan specifies for compressed/uncompressed pairs.
       *
       * Only G17P reached this: hk_CmdCopyImage2 routes every G17P copy
       * through the graphics meta path, while AGX2 uses the compute path for
       * all but a few destinations.
       * G17P also uses integer aliases for uncompressed copies: a normalized
       * render target may round the copied values or untouched pixels reloaded
       * by the background shader, violating the bit-exact copy contract.
       */
      if (g17p && img->planes[0].layout.compressed) {
         /* R32_UINT and RGBA8_UINT have the same texel size but different
          * lossless compression classes. Uploads and image copies must write
          * the channel layout that ordinary image views will consume. */
         props.color.view_format = hk_meta_compressed_copy_format(img->vk.format);
         assert(ail_formats_compatible(
            img->planes[0].layout.format,
            hk_format_to_pipe_format(props.color.view_format)));
      } else {
         props.color.view_format =
            (g17p || hk_agx2_packed_float_copy(dev->dev.chip, img->vk.format) ||
             vk_format_is_compressed(img->vk.format))
               ? hk_meta_bitwise_copy_format(img->vk.format)
               : img->vk.format;
      }

      /* That uncompressed view addresses one level of a block-compressed
       * image as a standalone image (hk_image_view.c), so it reaches one
       * layer only: the layer stride the hardware derives for it is not the
       * image's. G17P copies one layer or slice per region, so view just
       * that one. */
      props.copied_layers_src_view =
         g17p && vk_format_is_compressed(img->vk.format);
   } else {
      switch (img->vk.format) {
      case VK_FORMAT_S8_UINT:
         props.stencil.view_format = VK_FORMAT_R8_UINT;
         props.stencil.component_mask = BITFIELD_MASK(1);
         break;
      case VK_FORMAT_D32_SFLOAT_S8_UINT:
      case VK_FORMAT_D16_UNORM_S8_UINT:
         if (props.depth_stencil_aspect_views) {
            /* Internal views preserve the selected backing plane and use
             * its raw element width, not the interleaved API format. */
            props.depth.view_format = img->vk.format == VK_FORMAT_D16_UNORM_S8_UINT
                                         ? VK_FORMAT_R16_UINT : VK_FORMAT_R32_UINT;
            props.depth.component_mask = BITFIELD_BIT(0);
            props.stencil.view_format = VK_FORMAT_R8_UINT;
            props.stencil.component_mask = BITFIELD_BIT(0);
            break;
         }
         props.depth.view_format = VK_FORMAT_R32G32_UINT;
         props.depth.component_mask = BITFIELD_BIT(0);
         props.stencil.view_format = VK_FORMAT_R32G32_UINT;
         props.stencil.component_mask = BITFIELD_BIT(1);
         break;
      case VK_FORMAT_D16_UNORM:
         props.depth.view_format = VK_FORMAT_R16_UINT;
         props.depth.component_mask = BITFIELD_BIT(0);
         break;
      case VK_FORMAT_D32_SFLOAT:
         props.depth.view_format = VK_FORMAT_R32_UINT;
         props.depth.component_mask = BITFIELD_BIT(0);
         break;
      default:
         UNREACHABLE("Invalid ZS format");
      }
   }

   return props;
}

/*
 * G17P has no general compute imageblock path yet, so the common Vulkan meta
 * blitter turns every mip level into a separate render firmware job.  STK and
 * GTK generate complete mip chains for hundreds of small RGBA/BGRA images at
 * startup; making each level a render is thousands of jobs before the first
 * frame can be presented.
 *
 * Use raw AIL compute kernels for complete half-size UNORM8 mip levels in the
 * same RGBA or BGRA format (or 1x along an axis already at one pixel).  Odd
 * extents use CPU-built Q8 coordinate tables matching G17P's linear sampler,
 * so their device path has no divide or 64-bit arithmetic.  The fallback
 * remains the general graphics meta path, so this does not narrow Vulkan
 * functionality.
 */
static bool
hk_g17p_compute_blit_format(VkFormat format)
{
   return format == VK_FORMAT_R8G8B8A8_UNORM ||
          format == VK_FORMAT_B8G8R8A8_UNORM;
}

static bool
hk_g17p_compute_blit_region_supported(const struct hk_image *src,
                                      const struct hk_image *dst,
                                      const VkImageBlit2 *region,
                                      VkFilter filter, bool compressed)
{
   if (filter != VK_FILTER_LINEAR || src->vk.image_type != VK_IMAGE_TYPE_2D ||
       dst->vk.image_type != VK_IMAGE_TYPE_2D ||
       src->vk.samples != VK_SAMPLE_COUNT_1_BIT ||
       dst->vk.samples != VK_SAMPLE_COUNT_1_BIT || src->plane_count != 1 ||
       dst->plane_count != 1 ||
       src->planes[0].layout.tiling != AIL_TILING_GPU ||
       dst->planes[0].layout.tiling != AIL_TILING_GPU ||
       (!compressed && (src->planes[0].layout.compressed ||
                        dst->planes[0].layout.compressed)) ||
       src->vk.format != dst->vk.format ||
       !hk_g17p_compute_blit_format(src->vk.format) ||
       region->srcSubresource.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
       region->dstSubresource.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
       vk_image_subresource_layer_count(&src->vk, &region->srcSubresource) != 1 ||
       vk_image_subresource_layer_count(&dst->vk, &region->dstSubresource) != 1)
      return false;

   const VkOffset3D s0 = region->srcOffsets[0];
   const VkOffset3D s1 = region->srcOffsets[1];
   const VkOffset3D d0 = region->dstOffsets[0];
   const VkOffset3D d1 = region->dstOffsets[1];
   if (s0.x != 0 || s0.y != 0 || s0.z != 0 || s1.z != 1 ||
       d0.x != 0 || d0.y != 0 || d0.z != 0 || d1.z != 1 ||
       s1.x <= 0 || s1.y <= 0 || d1.x <= 0 || d1.y <= 0)
      return false;

   const uint32_t width = (uint32_t)d1.x;
   const uint32_t height = (uint32_t)d1.y;
   const uint32_t src_width = (uint32_t)s1.x;
   const uint32_t src_height = (uint32_t)s1.y;
   const bool scale_x =
      src_width == width || src_width == (uint64_t)width * 2 ||
      src_width == (uint64_t)width * 2 + 1;
   const bool scale_y =
      src_height == height || src_height == (uint64_t)height * 2 ||
      src_height == (uint64_t)height * 2 + 1;
   return scale_x && scale_y &&
          (src_width != width || src_height != height) &&
          src_width <= UINT16_MAX + 1u && src_height <= UINT16_MAX + 1u &&
          (uint64_t)width * height <= UINT32_MAX;
}

static bool
hk_g17p_compute_blit_supported(const struct hk_device *dev,
                               const struct hk_image *src,
                               const struct hk_image *dst,
                               const VkBlitImageInfo2 *info)
{
   if (!agx_native_graphics(&dev->dev) || info->regionCount == 0)
      return false;

   for (uint32_t r = 0; r < info->regionCount; ++r) {
      if (!hk_g17p_compute_blit_region_supported(
             src, dst, &info->pRegions[r], info->filter, false))
         return false;
   }
   return true;
}

/*
 * G17P downsample into or out of a compressed image, in compute.
 *
 * A compressed G17P colour image is otherwise written only through the PBE,
 * so a linear blit into one is a render pass per region: a GL
 * glGenerateMipmap of a 1080p target becomes ten dependent passes where an
 * uncompressed one takes the libagx_mipmap_unorm8x4 dispatches above.
 *
 * Here one dispatch per region reads the source level through the texture
 * unit, which decodes whatever state its tiles are in, writes the
 * destination texels raw in the GPU-tiled layout, and, when the destination
 * level is compressed, sets each of its 16x16 metadata tiles to the
 * uncompressed mode (ail_tile_mode_uncompressed()). That is the state
 * libagx_decompress leaves a tile in on AGX2; on G17P libagx_fast_clear
 * writes the same metadata words, indexed the same way, with the solid
 * mode. The tests/g17p_vulkan_compressed_mip_blit.c readback checks the
 * uncompressed mode on hardware. Reads, background loads and render passes
 * then treat the level as usual, and a pass that stores a tile
 * recompresses it.
 *
 * The arithmetic is libagx_mipmap_unorm8x4_odd's, which reproduces the
 * linear blit exactly. Exact 2x and 1x axes take its weights (128 and 0)
 * directly instead of from a table.
 */
static bool
hk_g17p_compressed_blit_supported(const struct hk_device *dev,
                                  const struct hk_image *src,
                                  const struct hk_image *dst,
                                  const VkBlitImageInfo2 *info)
{
   if (dev->dev.chip != AGX_CHIP_G17P || info->regionCount == 0 ||
       !(src->planes[0].layout.compressed || dst->planes[0].layout.compressed))
      return false;

   for (uint32_t i = 0; i < 2; ++i) {
      const struct ail_layout *layout =
         i ? &dst->planes[0].layout : &src->planes[0].layout;
      if (layout->compressed && layout->policy == AIL_LAYOUT_POLICY_M5)
         return false;
   }

   for (uint32_t r = 0; r < info->regionCount; ++r) {
      const VkImageBlit2 *region = &info->pRegions[r];
      if (!hk_g17p_compute_blit_region_supported(src, dst, region,
                                                 info->filter, true))
         return false;

      /* Marking a level's tiles uncompressed needs all of its texels, and
       * large compressed levels are cheaper stored compressed by a pass */
      const uint32_t level = region->dstSubresource.mipLevel;
      const uint32_t width = u_minify(dst->vk.extent.width, level);
      const uint32_t height = u_minify(dst->vk.extent.height, level);
      if (ail_is_level_logically_compressed(&dst->planes[0].layout, level) &&
          ((uint32_t)region->dstOffsets[1].x != width ||
           (uint32_t)region->dstOffsets[1].y != height ||
           hk_g17p_mip_level_through_pbe(width, height)))
         return false;
   }
   return true;
}

struct hk_g17p_mip_key {
   enum hk_meta_object_key_type key_type;
   bool odd;
   bool metadata;
};

struct hk_g17p_mip_push {
   uint64_t dst;
   uint64_t metadata;
   uint64_t axis;
   uint32_t width;
   uint32_t height;
   uint32_t scale_x;
   uint32_t scale_y;
   uint32_t tiles_per_row;
   uint32_t tile_width_log2;
   uint32_t tile_height_log2;
   uint32_t metadata_width_tl;
   uint32_t metadata_height_tl;
   uint32_t metadata_interleave_bits;
   uint32_t metadata_major_is_x;
   uint32_t uncompressed_mode;
};
static_assert(sizeof(struct hk_g17p_mip_push) == 72,
              "G17P mip push data must be tightly packed");

#define mip_push(b, field)                                                     \
   nir_load_push_constant(                                                     \
      b, 1, sizeof(((struct hk_g17p_mip_push *)0)->field) * 8,                 \
      nir_imm_int(b, offsetof(struct hk_g17p_mip_push, field)))

/* x with a zero bit inserted above each of its low 16 bits */
static nir_def *
hk_nir_spread_bits(nir_builder *b, nir_def *v)
{
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 8)), 0x00ff00ff);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 4)), 0x0f0f0f0f);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 2)), 0x33333333);
   return nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 1)), 0x55555555);
}

/* The even bits of the low 10 bits of v, packed: inverse of the spread */
static nir_def *
hk_nir_compact_bits(nir_builder *b, nir_def *v)
{
   v = nir_iand_imm(b, v, 0x155);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ushr_imm(b, v, 1)), 0x133);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ushr_imm(b, v, 2)), 0x10f);
   return nir_iand_imm(b, nir_ior(b, v, nir_ushr_imm(b, v, 4)), 0x1f);
}

static nir_shader *
build_g17p_mip_shader(const struct hk_g17p_mip_key *key)
{
   nir_builder build =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "hk-g17p-mip");
   nir_builder *b = &build;
   b->shader->info.workgroup_size[0] = TILE_WIDTH;
   b->shader->info.workgroup_size[1] = TILE_HEIGHT;

   nir_variable *texture = nir_variable_create(
      b->shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, true, GLSL_TYPE_UINT),
      "source");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;

   /* Lay the 32x32 invocations of a workgroup out in Morton order, as the
    * GPU-tiled layout stores texels: each 32-invocation group then covers an
    * 8x4 block, whose texels are 128 contiguous bytes, instead of a 32-texel
    * row scattered over two tiles.
    */
   nir_def *lid = nir_load_local_invocation_id(b);
   nir_def *index = nir_iadd(b, nir_channel(b, lid, 0),
                             nir_imul_imm(b, nir_channel(b, lid, 1), TILE_WIDTH));
   nir_def *wg = nir_load_workgroup_id(b);
   nir_def *x = nir_iadd(b, nir_imul_imm(b, nir_channel(b, wg, 0), TILE_WIDTH),
                         hk_nir_compact_bits(b, index));
   nir_def *y = nir_iadd(b, nir_imul_imm(b, nir_channel(b, wg, 1), TILE_HEIGHT),
                         hk_nir_compact_bits(b, nir_ushr_imm(b, index, 1)));
   nir_def *width = mip_push(b, width);
   nir_def *height = mip_push(b, height);

   nir_push_if(b, nir_iand(b, nir_ult(b, x, width), nir_ult(b, y, height)));
   {
      nir_def *x0, *y0, *wx, *wy;
      if (key->odd) {
         nir_def *axis = mip_push(b, axis);
         nir_def *x_entry = nir_load_global(
            b, 1, 32, nir_iadd(b, axis, nir_u2u64(b, nir_imul_imm(b, x, 4))),
            .align_mul = 4);
         nir_def *y_entry = nir_load_global(
            b, 1, 32,
            nir_iadd(b, axis,
                     nir_u2u64(b, nir_imul_imm(b, nir_iadd(b, width, y), 4))),
            .align_mul = 4);
         x0 = nir_iand_imm(b, x_entry, 0xffff);
         y0 = nir_iand_imm(b, y_entry, 0xffff);
         wx = nir_iand_imm(b, nir_ushr_imm(b, x_entry, 16), 0xff);
         wy = nir_iand_imm(b, nir_ushr_imm(b, y_entry, 16), 0xff);
      } else {
         nir_def *scale_x = mip_push(b, scale_x);
         nir_def *scale_y = mip_push(b, scale_y);
         x0 = nir_imul(b, x, scale_x);
         y0 = nir_imul(b, y, scale_y);
         wx = nir_ishl_imm(b, nir_iadd_imm(b, scale_x, -1), 7);
         wy = nir_ishl_imm(b, nir_iadd_imm(b, scale_y, -1), 7);
      }

      nir_def *x1 = nir_iadd(b, x0, nir_b2i32(b, nir_ine_imm(b, wx, 0)));
      nir_def *y1 = nir_iadd(b, y0, nir_b2i32(b, nir_ine_imm(b, wy, 0)));
      nir_def *ix = nir_isub(b, nir_imm_int(b, 256), wx);
      nir_def *iy = nir_isub(b, nir_imm_int(b, 256), wy);

      nir_def *zero = nir_imm_int(b, 0);
      nir_def *coords[4] = {
         nir_vec3(b, x0, y0, zero),
         nir_vec3(b, x1, y0, zero),
         nir_vec3(b, x0, y1, zero),
         nir_vec3(b, x1, y1, zero),
      };
      nir_def *weights[4] = {
         nir_imul(b, ix, iy),
         nir_imul(b, wx, iy),
         nir_imul(b, ix, wy),
         nir_imul(b, wx, wy),
      };
      nir_def *texels[4];
      for (unsigned i = 0; i < 4; ++i) {
         texels[i] = nir_txf(b, coords[i],
                             .texture_deref = nir_build_deref_var(b, texture));
         nir_def_as_tex(texels[i])->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
      }

      nir_def *packed = NULL;
      for (unsigned c = 0; c < 4; ++c) {
         nir_def *value = NULL;
         for (unsigned i = 0; i < 4; ++i) {
            nir_def *term = nir_imul(b, nir_channel(b, texels[i], c), weights[i]);
            value = value ? nir_iadd(b, value, term) : term;
         }

         /* Rounded to four fractional bits, then to the nearest code with
          * exact ties toward 128, as libagx_mipmap_unorm8x4_odd */
         nir_def *fixed4 = nir_ushr_imm(b, nir_iadd_imm(b, value, 2048), 12);
         nir_def *out = nir_ushr_imm(b, fixed4, 4);
         nir_def *rem = nir_iand_imm(b, fixed4, 15);
         nir_def *up = nir_ior(b, nir_ugt_imm(b, rem, 8),
                               nir_iand(b, nir_ieq_imm(b, rem, 8),
                                        nir_ult_imm(b, out, 128)));
         out = nir_iadd(b, out, nir_b2i32(b, up));

         nir_def *byte = nir_ishl_imm(b, out, 8 * c);
         packed = packed ? nir_ior(b, packed, byte) : byte;
      }

      nir_def *tile_width_log2 = mip_push(b, tile_width_log2);
      nir_def *tile_height_log2 = mip_push(b, tile_height_log2);
      nir_def *tile_x_mask =
         nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), tile_width_log2), -1);
      nir_def *tile_y_mask =
         nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), tile_height_log2), -1);
      nir_def *tile = nir_iadd(
         b, nir_imul(b, nir_ushr(b, y, tile_height_log2),
                     mip_push(b, tiles_per_row)),
         nir_ushr(b, x, tile_width_log2));
      nir_def *morton =
         nir_ior(b, hk_nir_spread_bits(b, nir_iand(b, x, tile_x_mask)),
                 nir_ishl_imm(b, hk_nir_spread_bits(b, nir_iand(b, y, tile_y_mask)), 1));
      nir_def *element = nir_ior(
         b, nir_ishl(b, tile, nir_iadd(b, tile_width_log2, tile_height_log2)),
         morton);

      nir_store_global(b, packed,
                       nir_iadd(b, mip_push(b, dst),
                                nir_u2u64(b, nir_imul_imm(b, element, 4))),
                       .align_mul = 4);
   }
   nir_pop_if(b, NULL);

   if (key->metadata) {
      nir_def *x_tl = nir_ushr_imm(b, x, 4);
      nir_def *y_tl = nir_ushr_imm(b, y, 4);
      nir_def *origin =
         nir_ieq_imm(b, nir_iand_imm(b, nir_ior(b, x, y), 15), 0);
      nir_def *in_grid =
         nir_iand(b, nir_ult(b, x_tl, mip_push(b, metadata_width_tl)),
                  nir_ult(b, y_tl, mip_push(b, metadata_height_tl)));

      nir_push_if(b, nir_iand(b, origin, in_grid));
      {
         nir_def *bits = mip_push(b, metadata_interleave_bits);
         nir_def *mask =
            nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), bits), -1);
         nir_def *major = nir_bcsel(
            b, nir_ine_imm(b, mip_push(b, metadata_major_is_x), 0), x_tl, y_tl);
         nir_def *index = nir_iadd(
            b,
            nir_ior(b, hk_nir_spread_bits(b, nir_iand(b, x_tl, mask)),
                    nir_ishl_imm(b, hk_nir_spread_bits(b, nir_iand(b, y_tl, mask)), 1)),
            nir_ishl(b, nir_iand(b, major, nir_inot(b, mask)), bits));
         nir_def *mode = mip_push(b, uncompressed_mode);

         nir_store_global(b, nir_vec2(b, mode, mode),
                          nir_iadd(b, mip_push(b, metadata),
                                   nir_u2u64(b, nir_imul_imm(b, index, 8))),
                          .align_mul = 8);
      }
      nir_pop_if(b, NULL);
   }

   return b->shader;
}

static VkResult
hk_g17p_mip_pipeline(struct hk_device *dev, const struct hk_g17p_mip_key *key,
                     VkPipelineLayout *layout_out, VkPipeline *pipeline_out)
{
   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo set_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(struct hk_g17p_mip_push),
   };
   static const char layout_key[] = "hk-g17p-mip-layout";

   VkResult result = vk_meta_get_pipeline_layout(
      &dev->vk, &dev->meta, &set_info, &push_range, layout_key,
      sizeof(layout_key), layout_out);
   if (unlikely(result != VK_SUCCESS))
      return result;

   VkPipeline cached = vk_meta_lookup_pipeline(&dev->meta, key, sizeof(*key));
   if (cached != VK_NULL_HANDLE) {
      *pipeline_out = cached;
      return VK_SUCCESS;
   }

   const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
      .nir = build_g17p_mip_shader(key),
   };
   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .pNext = &nir_info,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .pName = "main",
      },
      .layout = *layout_out,
   };

   result = vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &info, key,
                                            sizeof(*key), pipeline_out);
   ralloc_free(nir_info.nir);
   return result;
}

static void
hk_g17p_compressed_blit(struct hk_cmd_buffer *cmd, struct hk_image *src,
                        struct hk_image *dst, const VkBlitImageInfo2 *info)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkCommandBuffer handle = vk_command_buffer_to_handle(&cmd->vk);
   const struct ail_layout *dst_layout = &dst->planes[0].layout;

   for (uint32_t r = 0; r < info->regionCount; ++r) {
      const VkImageBlit2 *region = &info->pRegions[r];
      const uint32_t src_level = region->srcSubresource.mipLevel;
      const uint32_t dst_level = region->dstSubresource.mipLevel;
      const uint32_t dst_layer = region->dstSubresource.baseArrayLayer;
      const uint32_t width = (uint32_t)region->dstOffsets[1].x;
      const uint32_t height = (uint32_t)region->dstOffsets[1].y;
      const uint32_t src_width = (uint32_t)region->srcOffsets[1].x;
      const uint32_t src_height = (uint32_t)region->srcOffsets[1].y;
      const bool odd = src_width == (uint64_t)width * 2 + 1 ||
                       src_height == (uint64_t)height * 2 + 1;
      const bool metadata =
         ail_is_level_logically_compressed(dst_layout, dst_level);

      const struct ail_tile dst_tile = dst_layout->tilesize_el[dst_level];
      struct hk_g17p_tile_address_config tiles;
      struct hk_g17p_metadata_grid grid = {0};
      if (!hk_g17p_tile_address_config_init(dst_layout->stride_el[dst_level],
                                            dst_tile.width_el,
                                            dst_tile.height_el, &tiles) ||
          (metadata &&
           !hk_g17p_metadata_grid_init(
              ail_metadata_width_tl((struct ail_layout *)dst_layout, dst_level),
              ail_metadata_height_tl((struct ail_layout *)dst_layout, dst_level),
              &grid))) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
         return;
      }

      const struct hk_g17p_mip_key key = {
         .key_type = HK_META_OBJECT_KEY_G17P_MIP,
         .odd = odd,
         .metadata = metadata,
      };
      VkPipelineLayout layout;
      VkPipeline pipeline;
      VkResult result = hk_g17p_mip_pipeline(dev, &key, &layout, &pipeline);
      if (unlikely(result != VK_SUCCESS)) {
         vk_command_buffer_set_error(&cmd->vk, result);
         return;
      }

      struct hk_g17p_mip_push push = {
         .dst = hk_image_base_address(dst, 0) +
                ail_get_layer_level_B(dst_layout, dst_layer, dst_level),
         .width = width,
         .height = height,
         .scale_x = src_width == width ? 1 : 2,
         .scale_y = src_height == height ? 1 : 2,
         .tiles_per_row = tiles.tiles_per_row,
         .tile_width_log2 = tiles.tile_width_log2,
         .tile_height_log2 = tiles.tile_height_log2,
      };

      if (metadata) {
         push.metadata = hk_image_base_address(dst, 0) +
                         dst_layout->metadata_offset_B +
                         (uint64_t)dst_layer *
                            dst_layout->compression_layer_stride_B +
                         dst_layout->level_offsets_compressed_B[dst_level];
         push.metadata_width_tl = grid.width_tl;
         push.metadata_height_tl = grid.height_tl;
         push.metadata_interleave_bits = grid.interleave_bits;
         push.metadata_major_is_x = grid.major_is_x;
         push.uncompressed_mode =
            (uint32_t)ail_tile_mode_uncompressed(dst_layout->format);
      }

      if (odd) {
         struct agx_ptr axis =
            hk_pool_alloc(cmd, (width + height) * sizeof(uint32_t), 4);
         if (unlikely(axis.cpu == NULL))
            return;

         uint32_t *entries = axis.cpu;
         ASSERTED const bool x_ok =
            hk_g17p_build_mip_axis(src_width, width, entries);
         ASSERTED const bool y_ok =
            hk_g17p_build_mip_axis(src_height, height, entries + width);
         assert(x_ok && y_ok);
         push.axis = axis.gpu;
      }

      VkImageView src_view;
      const VkImageViewUsageCreateInfo src_view_usage = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
         .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      };
      const VkImageViewCreateInfo src_view_info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .flags = VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA,
         .pNext = &src_view_usage,
         .image = hk_image_to_handle(src),
         .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
         .format = hk_meta_compressed_copy_format(src->vk.format),
         .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = src_level,
            .levelCount = 1,
            .baseArrayLayer = region->srcSubresource.baseArrayLayer,
            .layerCount = 1,
         },
      };
      result =
         vk_meta_create_image_view(&cmd->vk, &dev->meta, &src_view_info,
                                   &src_view);
      if (unlikely(result != VK_SUCCESS)) {
         vk_command_buffer_set_error(&cmd->vk, result);
         return;
      }

      const VkDescriptorImageInfo src_info = {
         .imageLayout = info->srcImageLayout,
         .imageView = src_view,
      };
      const VkWriteDescriptorSet write = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstBinding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
         .descriptorCount = 1,
         .pImageInfo = &src_info,
      };

      disp->CmdBindPipeline(handle, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      disp->CmdPushDescriptorSetKHR(handle, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    layout, 0, 1, &write);
      disp->CmdPushConstants(handle, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             sizeof(push), &push);

      /* Cover the metadata grid too, which may extend past the texels */
      const uint32_t grid_w = MAX2(width, grid.width_tl * 16);
      const uint32_t grid_h = MAX2(height, grid.height_tl * 16);
      disp->CmdDispatch(handle, DIV_ROUND_UP(grid_w, TILE_WIDTH),
                        DIV_ROUND_UP(grid_h, TILE_HEIGHT), 1);
   }
}

static void
hk_g17p_compute_blit(struct hk_cmd_buffer *cmd, struct hk_image *src,
                     struct hk_image *dst, const VkBlitImageInfo2 *info)
{
   const struct ail_layout *src_layout = &src->planes[0].layout;
   const struct ail_layout *dst_layout = &dst->planes[0].layout;

   for (uint32_t r = 0; r < info->regionCount; ++r) {
      const VkImageBlit2 *region = &info->pRegions[r];
      const uint32_t src_level = region->srcSubresource.mipLevel;
      const uint32_t dst_level = region->dstSubresource.mipLevel;
      const uint32_t width =
         (uint32_t)(region->dstOffsets[1].x - region->dstOffsets[0].x);
      const uint32_t height =
         (uint32_t)(region->dstOffsets[1].y - region->dstOffsets[0].y);
      const uint32_t src_width = (uint32_t)region->srcOffsets[1].x;
      const uint32_t src_height = (uint32_t)region->srcOffsets[1].y;
      const struct ail_tile src_tile = src_layout->tilesize_el[src_level];
      const struct ail_tile dst_tile = dst_layout->tilesize_el[dst_level];
      struct hk_g17p_tile_address_config src_tile_config;
      struct hk_g17p_tile_address_config dst_tile_config;
      if (!hk_g17p_tile_address_config_init(
             src_layout->stride_el[src_level], src_tile.width_el,
             src_tile.height_el, &src_tile_config) ||
          !hk_g17p_tile_address_config_init(
             dst_layout->stride_el[dst_level], dst_tile.width_el,
             dst_tile.height_el, &dst_tile_config)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
         return;
      }
      const bool odd = src_width == (uint64_t)width * 2 + 1 ||
                       src_height == (uint64_t)height * 2 + 1;

      if (odd) {
         struct agx_ptr axis =
            hk_pool_alloc(cmd, (width + height) * sizeof(uint32_t), 4);
         if (unlikely(axis.cpu == NULL))
            return;

         uint32_t *entries = axis.cpu;
         const bool x_ok =
            hk_g17p_build_mip_axis(src_width, width, entries);
         const bool y_ok =
            hk_g17p_build_mip_axis(src_height, height, entries + width);
         assert(x_ok && y_ok);

         const struct libagx_mipmap_unorm8x4_odd_args args = {
            .dest = hk_image_base_address(dst, 0) +
                    ail_get_layer_level_B(
                       dst_layout, region->dstSubresource.baseArrayLayer,
                       dst_level),
            .src = hk_image_base_address(src, 0) +
                   ail_get_layer_level_B(
                      src_layout, region->srcSubresource.baseArrayLayer,
                      src_level),
            .axis = axis.gpu,
            .dest_width_px = width,
            .dest_height_px = height,
            .dest_tiles_per_row = dst_tile_config.tiles_per_row,
            .src_tiles_per_row = src_tile_config.tiles_per_row,
            .dest_tile_width_log2 = dst_tile_config.tile_width_log2,
            .dest_tile_height_log2 = dst_tile_config.tile_height_log2,
            .src_tile_width_log2 = src_tile_config.tile_width_log2,
            .src_tile_height_log2 = src_tile_config.tile_height_log2,
         };

         libagx_mipmap_unorm8x4_odd_struct(
            cmd, agx_2d(width, height), AGX_BARRIER_ALL, args);
         if (vk_command_buffer_has_error(&cmd->vk))
            return;
         continue;
      }

      const struct libagx_mipmap_unorm8x4_args args = {
         .dest = hk_image_base_address(dst, 0) +
                 ail_get_layer_level_B(dst_layout,
                                       region->dstSubresource.baseArrayLayer,
                                       dst_level),
         .src = hk_image_base_address(src, 0) +
                ail_get_layer_level_B(src_layout,
                                      region->srcSubresource.baseArrayLayer,
                                      src_level),
         .dest_width_px = width,
         .dest_height_px = height,
         .src_scale_x = src_width == width ? 1 : 2,
         .src_scale_y = src_height == height ? 1 : 2,
         .dest_tiles_per_row = dst_tile_config.tiles_per_row,
         .src_tiles_per_row = src_tile_config.tiles_per_row,
         .dest_tile_width_log2 = dst_tile_config.tile_width_log2,
         .dest_tile_height_log2 = dst_tile_config.tile_height_log2,
         .src_tile_width_log2 = src_tile_config.tile_width_log2,
         .src_tile_height_log2 = src_tile_config.tile_height_log2,
      };

      libagx_mipmap_unorm8x4_struct(cmd, agx_2d(width, height),
                                    AGX_BARRIER_ALL, args);
      if (vk_command_buffer_has_error(&cmd->vk))
         return;
   }
}

static void
hk_xfer_blitimage2(VkCommandBuffer commandBuffer,
                 const VkBlitImageInfo2 *pBlitImageInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_image, src_image, pBlitImageInfo->srcImage);
   VK_FROM_HANDLE(hk_image, dst_image, pBlitImageInfo->dstImage);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   perf_debug(cmd, "Blit image");

   if (hk_g17p_compute_blit_supported(dev, src_image, dst_image,
                                      pBlitImageInfo)) {
      struct hk_meta_save save;
      hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      hk_g17p_compute_blit(cmd, src_image, dst_image, pBlitImageInfo);
      hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      return;
   }

   if (hk_g17p_compressed_blit_supported(dev, src_image, dst_image,
                                         pBlitImageInfo)) {
      struct hk_meta_save save;
      hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      hk_g17p_compressed_blit(cmd, src_image, dst_image, pBlitImageInfo);
      hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      return;
   }

   struct hk_meta_save save;
   hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
   if (agx_native_graphics(&dev->dev) &&
       src_image->vk.image_type == VK_IMAGE_TYPE_2D &&
       dst_image->vk.image_type == VK_IMAGE_TYPE_2D) {
      for (uint32_t r = 0; r < pBlitImageInfo->regionCount; ++r) {
         const VkImageBlit2 *region = &pBlitImageInfo->pRegions[r];
         const uint32_t src_layers = vk_image_subresource_layer_count(
            &src_image->vk, &region->srcSubresource);
         const uint32_t dst_layers = vk_image_subresource_layer_count(
            &dst_image->vk, &region->dstSubresource);

         for (uint32_t layer = 0; layer < src_layers; ++layer) {
            VkImageBlit2 one;
            if (!hk_g17p_image_blit_slice(
                   dev->dev.chip, src_image->vk.image_type,
                   dst_image->vk.image_type, src_layers, dst_layers,
                   region, layer, &one)) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               break;
            }

            VkBlitImageInfo2 info = *pBlitImageInfo;
            info.regionCount = 1;
            info.pRegions = &one;
            /* Native optimal RGB images have four-component physical
             * storage.  Render the blit through that physical destination
             * view while preserving the logical source sampling format. */
            enum pipe_format logical =
               hk_format_to_pipe_format(dst_image->vk.format);
            enum pipe_format physical = hk_padded_image_format(logical);
            enum pipe_format scaled = hk_scaled_texture_format(physical);
            /* Scaled blits keep a floating-point shader interface. The
             * rendering and PBE paths select their integer physical storage
             * independently, avoiding an integer sampler type for a scaled
             * source (and preserving normalized USCALED filtering). */
            VkFormat dst_format = scaled != PIPE_FORMAT_NONE
                                     ? dst_image->vk.format
                                     : physical == logical
                                          ? dst_image->vk.format
                                          : vk_format_from_pipe_format(physical);
            vk_meta_blit_image(&cmd->vk, &dev->meta, &src_image->vk,
                               src_image->vk.format, info.srcImageLayout,
                               &dst_image->vk, dst_format, info.dstImageLayout,
                               1, &one, info.filter);
            if (vk_command_buffer_has_error(&cmd->vk))
               break;
         }

         if (vk_command_buffer_has_error(&cmd->vk))
            break;
      }
   } else {
      vk_meta_blit_image2(&cmd->vk, &dev->meta, pBlitImageInfo);
   }
   hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

static void
hk_xfer_resolveimage2(VkCommandBuffer commandBuffer,
                    const VkResolveImageInfo2 *pResolveImageInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_image, src_image, pResolveImageInfo->srcImage);
   VK_FROM_HANDLE(hk_image, dst_image, pResolveImageInfo->dstImage);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   perf_debug(cmd, "Resolve");

   const struct ail_layout *src_layout = &src_image->planes[0].layout;
   const struct ail_layout *dst_layout = &dst_image->planes[0].layout;
   bool raw_unorm8x4 =
      agx_native_graphics(&dev->dev) &&
      src_image->vk.image_type == VK_IMAGE_TYPE_2D &&
      dst_image->vk.image_type == VK_IMAGE_TYPE_2D &&
      src_image->vk.format == VK_FORMAT_R8G8B8A8_UNORM &&
      dst_image->vk.format == VK_FORMAT_R8G8B8A8_UNORM &&
      (src_image->vk.samples == VK_SAMPLE_COUNT_2_BIT ||
       src_image->vk.samples == VK_SAMPLE_COUNT_4_BIT) &&
      dst_image->vk.samples == VK_SAMPLE_COUNT_1_BIT &&
      src_image->plane_count == 1 && dst_image->plane_count == 1 &&
      src_layout->tiling == AIL_TILING_GPU &&
      (dst_layout->tiling == AIL_TILING_GPU ||
       dst_layout->tiling == AIL_TILING_LINEAR) &&
      !src_layout->compressed && !dst_layout->compressed &&
      pResolveImageInfo->regionCount != 0 &&
      hk_g17p_transfer_src_layout_supported(pResolveImageInfo->srcImageLayout) &&
      hk_g17p_transfer_dst_layout_supported(pResolveImageInfo->dstImageLayout);

   for (uint32_t r = 0; raw_unorm8x4 &&
                        r < pResolveImageInfo->regionCount; ++r) {
      const VkImageResolve2 *region = &pResolveImageInfo->pRegions[r];
      if (region->srcSubresource.mipLevel >= src_image->vk.mip_levels ||
          region->dstSubresource.mipLevel >= dst_image->vk.mip_levels) {
         raw_unorm8x4 = false;
         break;
      }
      const uint32_t src_layers = vk_image_subresource_layer_count(
         &src_image->vk, &region->srcSubresource);
      const uint32_t dst_layers = vk_image_subresource_layer_count(
         &dst_image->vk, &region->dstSubresource);
      const VkExtent3D src_extent = vk_image_mip_level_extent(
         &src_image->vk, region->srcSubresource.mipLevel);
      const VkExtent3D dst_extent = vk_image_mip_level_extent(
         &dst_image->vk, region->dstSubresource.mipLevel);
      raw_unorm8x4 =
         region->srcSubresource.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT &&
         region->dstSubresource.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT &&
         src_layers != 0 && src_layers == dst_layers &&
         region->extent.width != 0 && region->extent.height != 0 &&
         region->extent.depth == 1 && region->srcOffset.x >= 0 &&
         region->srcOffset.y >= 0 && region->srcOffset.z == 0 &&
         region->dstOffset.x >= 0 && region->dstOffset.y >= 0 &&
         region->dstOffset.z == 0 &&
         (uint32_t)region->srcOffset.x <= src_extent.width &&
         region->extent.width <=
            src_extent.width - (uint32_t)region->srcOffset.x &&
         (uint32_t)region->srcOffset.y <= src_extent.height &&
         region->extent.height <=
            src_extent.height - (uint32_t)region->srcOffset.y &&
         (uint32_t)region->dstOffset.x <= dst_extent.width &&
         region->extent.width <=
            dst_extent.width - (uint32_t)region->dstOffset.x &&
         (uint32_t)region->dstOffset.y <= dst_extent.height &&
         region->extent.height <=
            dst_extent.height - (uint32_t)region->dstOffset.y &&
         (uint64_t)region->extent.width * region->extent.height <= UINT32_MAX;
   }

   if (raw_unorm8x4) {
      struct hk_meta_save save;
      hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      for (uint32_t r = 0; r < pResolveImageInfo->regionCount; ++r) {
         const VkImageResolve2 *region = &pResolveImageInfo->pRegions[r];
         const uint32_t src_level = region->srcSubresource.mipLevel;
         const uint32_t dst_level = region->dstSubresource.mipLevel;
         const uint32_t layers = vk_image_subresource_layer_count(
            &src_image->vk, &region->srcSubresource);
         const struct ail_tile src_tile = src_layout->tilesize_el[src_level];
         const struct ail_tile dst_tile = dst_layout->tilesize_el[dst_level];
         const bool dst_tiled = dst_layout->tiling == AIL_TILING_GPU;

         for (uint32_t layer = 0; layer < layers; ++layer) {
            const struct libagx_resolve_unorm8x4_args args = {
               .dest = hk_image_base_address(dst_image, 0) +
                       ail_get_layer_level_B(
                          dst_layout,
                          region->dstSubresource.baseArrayLayer + layer,
                          dst_level),
               .src = hk_image_base_address(src_image, 0) +
                      ail_get_layer_level_B(
                         src_layout,
                         region->srcSubresource.baseArrayLayer + layer,
                         src_level),
               .width_px = region->extent.width,
               .height_px = region->extent.height,
               .dest_x_px = region->dstOffset.x,
               .dest_y_px = region->dstOffset.y,
               .src_x_px = region->srcOffset.x,
               .src_y_px = region->srcOffset.y,
               .dest_pitch_B = dst_tiled
                                  ? 0
                                  : ail_get_linear_stride_B(dst_layout,
                                                            dst_level),
               .dest_stride_el = dst_layout->stride_el[dst_level],
               .src_stride_el = src_layout->stride_el[src_level],
               .dest_tile_width_el = dst_tile.width_el,
               .dest_tile_height_el = dst_tile.height_el,
               .src_tile_width_el = src_tile.width_el,
               .src_tile_height_el = src_tile.height_el,
               .sample_count = src_image->vk.samples,
               .dest_tiled = dst_tiled,
               .src_tiled = true,
               .resolve_mode = 0,
            };
            libagx_resolve_unorm8x4_struct(
               cmd, agx_1d((uint64_t)region->extent.width *
                           region->extent.height),
               AGX_BARRIER_ALL, args);
            if (vk_command_buffer_has_error(&cmd->vk))
               break;
         }
         if (vk_command_buffer_has_error(&cmd->vk))
            break;
      }
      hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      return;
   }

   struct hk_meta_save save;
   hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
   vk_meta_resolve_image2(&cmd->vk, &dev->meta, pResolveImageInfo);
   hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

void
hk_meta_resolve_rendering(struct hk_cmd_buffer *cmd,
                          const VkRenderingInfo *pRenderingInfo)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   struct hk_meta_save save;
   hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
   vk_meta_resolve_rendering(&cmd->vk, &dev->meta, pRenderingInfo);
   hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

static void
hk_cmd_copy(struct hk_cmd_buffer *cmd, uint64_t dst, uint64_t src, size_t size)
{
   /* Use vectorized copies for as much of the buffer as possible. This requires
    * that dst, src, and size are all properly aligned. Failing to check for
    * alignment on the buffers causes subtle and hard-to-debug issues!
    */
   if (size >= 16 && (dst & 0xf) == 0 && (src & 0xf) == 0) {
      unsigned uint4s = size / 16;
      unsigned bytes = uint4s * 16;

      libagx_copy_uint4(cmd, agx_1d(uint4s), AGX_BARRIER_ALL, dst, src);

      dst += bytes;
      src += bytes;
      size -= bytes;
   }

   if (size) {
      libagx_copy_uchar(cmd, agx_1d(size), AGX_BARRIER_ALL, dst, src);
   }
}

static void
hk_xfer_copybuffer2(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 *info)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (agx_native_graphics(&dev->dev) ||
       agx_chip_g17_m5(dev->dev.chip)) {
      struct hk_meta_save save;
      hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      /* Retain the aligned 16-byte bulk path even when the original
       * region has a scalar tail, as hk_cmd_copy does for libagx. */
      for (unsigned i = 0; i < info->regionCount; i++) {
         VK_FROM_HANDLE(hk_buffer, src_buffer, info->srcBuffer);
         VkBufferCopy2 region = info->pRegions[i];
         uint64_t src = hk_buffer_address_rw(src_buffer, region.srcOffset);
         uint64_t dst = vk_meta_buffer_address(
            &dev->vk, info->dstBuffer, region.dstOffset, region.size);
         if (region.size > 16 && ((src | dst) & 15) == 0 &&
             (region.size & 15) != 0) {
            const VkDeviceSize tail = region.size & 15;
            region.size -= tail;
            if (src_buffer->sparse_map)
               hk_g17p_sparse_buffer_copy(cmd, src_buffer, src, dst,
                                          region.size);
            else
               vk_meta_copy_buffer_addresses(&cmd->vk, &dev->meta, src, dst,
                                             region.size);
            src += region.size;
            dst += region.size;
            region.size = tail;
         }
         if (src_buffer->sparse_map)
            hk_g17p_sparse_buffer_copy(cmd, src_buffer, src, dst, region.size);
         else
            vk_meta_copy_buffer_addresses(&cmd->vk, &dev->meta, src, dst,
                                          region.size);
      }
      hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      return;
   }

   for (unsigned i = 0; i < info->regionCount; i++) {
      const VkBufferCopy2 *region = &info->pRegions[i];
      VK_FROM_HANDLE(hk_buffer, src_buffer, info->srcBuffer);
      uint64_t src = hk_buffer_address_ro(src_buffer, region->srcOffset);
      uint64_t dst = vk_meta_buffer_address(&dev->vk, info->dstBuffer,
                                            region->dstOffset, region->size);
      hk_cmd_copy(cmd, dst, src, region->size);
   }
}

static bool
hk_copy_requires_gfx(struct hk_image *img)
{
   return img->vk.samples > 1 && img->planes[0].layout.compressed;
}

/* The VkFormat of the plane an aspect selects. Single-plane images are their
 * own plane; a multi-planar image's aspect picks one of the ycbcr planes,
 * which is the format the copy will actually render into. */
static VkFormat
hk_g17p_copy_plane_format(const struct hk_image *image, VkImageAspectFlags aspect)
{
   const struct vk_format_ycbcr_info *ycbcr =
      vk_format_get_ycbcr_info(image->vk.format);
   if (!ycbcr)
      return image->vk.format;

   const uint8_t plane = hk_image_aspects_to_plane(image, aspect);
   if (plane >= ycbcr->n_planes)
      return VK_FORMAT_UNDEFINED;
   return ycbcr->planes[plane].format;
}

static bool
hk_g17p_copy_buffer_to_image_supported(
   struct hk_device *dev, const struct hk_image *image,
   const VkCopyBufferToImageInfo2 *info)
{
   for (unsigned i = 0; i < info->regionCount; ++i) {
      const VkBufferImageCopy2 *region = &info->pRegions[i];
      /* The gate needs colour-attachment support, which is what
       * hk_get_format_features() reports from ail_pixel_format[].renderable.
       * For a multi-planar destination the core selects the plane format and
       * this is the aspect's own plane; ask about that one. */
      VkFormat plane_vk_format = hk_g17p_copy_plane_format(
         image, region->imageSubresource.aspectMask);
      /* Match hk_meta_copy_get_image_properties(): D32/D16 uploads use a raw
       * R32_UINT/R16_UINT color view, not a depth fragment output. Query the
       * format the graphics copy actually renders, preserving every bit. */
      if (region->imageSubresource.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT) {
         /* The depth plane of a split depth/stencil image. */
         plane_vk_format = hk_depth_stencil_plane_format(plane_vk_format, 0);
         if (plane_vk_format == VK_FORMAT_D32_SFLOAT)
            plane_vk_format = VK_FORMAT_R32_UINT;
         else if (plane_vk_format == VK_FORMAT_D16_UNORM)
            plane_vk_format = VK_FORMAT_R16_UINT;
      } else if (region->imageSubresource.aspectMask ==
                 VK_IMAGE_ASPECT_STENCIL_BIT) {
         /* Likewise the stencil plane renders through its raw R8_UINT view. */
         plane_vk_format = VK_FORMAT_R8_UINT;
      }
      const enum pipe_format plane_pipe =
         hk_format_to_pipe_format(plane_vk_format);
      const bool plane_renderable =
         plane_pipe != PIPE_FORMAT_NONE &&
         ail_pixel_format[plane_pipe].renderable;
      if (!hk_g17p_buffer_to_image_copy_supported(
      &(const struct hk_g17p_buffer_to_image_copy){
         .chip = dev->dev.chip,
         .format = image->vk.format,
         .plane_format_renderable = plane_renderable,
         .layout = info->dstImageLayout,
         .image_type = image->vk.image_type,
         .samples = image->vk.samples,
         .plane_count = image->plane_count,
         .mip_levels = image->vk.mip_levels,
         .array_layers = image->vk.array_layers,
         .image_extent = image->vk.extent,
         .aspect = region->imageSubresource.aspectMask,
         .mip_level = region->imageSubresource.mipLevel,
         .base_array_layer = region->imageSubresource.baseArrayLayer,
         .layer_count = region->imageSubresource.layerCount,
         .image_offset = region->imageOffset,
         .copy_extent = region->imageExtent,
         .buffer_row_length = region->bufferRowLength,
         .buffer_image_height = region->bufferImageHeight,
      }))
         return false;
   }
   return true;
}

static bool
hk_g17p_block_copy_format(struct hk_device *dev, VkFormat format)
{
   if (!dev || (!agx_native_graphics(&dev->dev) &&
                !agx_chip_g17_m5(dev->dev.chip)))
      return false;

   switch (hk_g17p_block_format_family(format)) {
   case HK_G17P_BLOCK_FORMAT_BC:
   case HK_G17P_BLOCK_FORMAT_ETC2_EAC:
   case HK_G17P_BLOCK_FORMAT_ASTC_LDR:
   case HK_G17P_BLOCK_FORMAT_ASTC_HDR:
      return true;
   default:
      return false;
   }
}

static struct hk_g17p_buffer_to_image_copy
hk_g17p_buffer_upload_desc(struct hk_device *dev, const struct hk_image *image,
                           const VkCopyBufferToImageInfo2 *info,
                           const VkBufferImageCopy2 *region, uint8_t plane,
                           VkFormat format)
{
   const struct ail_layout *layout = &image->planes[plane].layout;
   return (struct hk_g17p_buffer_to_image_copy){
      .chip = dev->dev.chip,
      /* The raw compute path writes the image with a compute kernel, so it
       * needs no EOT-capable format at all; it reuses the shared gate only for
       * that gate's dimension, layer and mip checks, and the gate insists on a
       * renderable format for the graphics fallback's sake.  Pass it as true
       * here so the rest of those checks still run.  This is also why
       * single-plane 4:2:2 uploads have to come through this path at all: no
       * subsampled pipe format has an ail_pixel_format[] entry, so none of them
       * is renderable, and the graphics path refuses the copy outright with
       * VK_ERROR_FEATURE_NOT_PRESENT. */
      .plane_format_renderable = true,
      .format = format,
      .layout = info->dstImageLayout,
      .image_type = image->vk.image_type,
      .samples = image->vk.samples,
      .plane_count = 1,
      .mip_levels = image->vk.mip_levels,
      .array_layers = image->vk.array_layers,
      .image_extent = {layout->width_px, layout->height_px,
                       image->vk.extent.depth},
      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      .mip_level = region->imageSubresource.mipLevel,
      .base_array_layer = region->imageSubresource.baseArrayLayer,
      .layer_count = region->imageSubresource.layerCount,
      .image_offset = region->imageOffset,
      .copy_extent = region->imageExtent,
      .buffer_row_length = region->bufferRowLength,
      .buffer_image_height = region->bufferImageHeight,
   };
}

static bool
hk_g17p_buffer_upload_plane(const struct hk_image *image,
                            VkImageAspectFlags aspect, uint8_t *plane,
                            VkFormat *format)
{
   if (aspect == VK_IMAGE_ASPECT_COLOR_BIT) {
      if (image->plane_count != 1)
         return false;
   } else if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
      if (!vk_format_has_depth(image->vk.format))
         return false;
   } else if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) {
      if (!vk_format_has_stencil(image->vk.format))
         return false;
   } else if (aspect == VK_IMAGE_ASPECT_PLANE_0_BIT ||
              aspect == VK_IMAGE_ASPECT_PLANE_1_BIT ||
              aspect == VK_IMAGE_ASPECT_PLANE_2_BIT) {
      const struct vk_format_ycbcr_info *ycbcr =
         vk_format_get_ycbcr_info(image->vk.format);
      if (!ycbcr || ycbcr->n_planes <= 1 ||
          hk_image_aspects_to_plane(image, aspect) >= ycbcr->n_planes)
         return false;
   } else {
      return false;
   }

   *plane = hk_image_aspects_to_plane(image, aspect);
   *format = aspect_format(image->vk.format, aspect);
   return *plane < image->plane_count && *format != VK_FORMAT_UNDEFINED;
}

static bool
hk_g17p_padded_rgb_layout(const struct hk_image *image, uint8_t plane,
                          VkFormat format, uint32_t *api_block_B,
                          uint32_t *physical_block_B)
{
   const enum pipe_format api = hk_format_to_pipe_format(format);
   const enum pipe_format physical = hk_padded_image_format(api);
   if (api == PIPE_FORMAT_NONE || physical == api ||
       image->planes[plane].layout.format != physical)
      return false;

   *api_block_B = util_format_get_blocksize(api);
   *physical_block_B = util_format_get_blocksize(physical);
   return *api_block_B != 0 && *physical_block_B != 0 &&
          *api_block_B * 4 == *physical_block_B * 3;
}

/* The M1/M2 path stores three-component formats in four-component tiled
 * images.  Buffer transfers still use the packed Vulkan representation, so
 * the GPU copy needs independent source and destination element sizes. */
struct hk_block_copy_key {
   enum hk_meta_object_key_type type;
   uint32_t block_B;
   uint32_t packed_rgb_buffer_B;
   bool to_image;
   int32_t src_tile_log2_w, src_tile_log2_h;
   int32_t dst_tile_log2_w, dst_tile_log2_h;
};

struct hk_block_copy_push {
   uint64_t src, dst;
   uint32_t src_stride, dst_stride;
   uint32_t src_x, src_y, dst_x, dst_y;
   uint32_t width, height;
};

#define block_push(b, field)                                                   \
   nir_load_push_constant(                                                     \
      b, 1, sizeof(((struct hk_block_copy_push *)0)->field) * 8,               \
      nir_imm_int(b, offsetof(struct hk_block_copy_push, field)))

static nir_def *
hk_block_copy_index(nir_builder *b, nir_def *x, nir_def *y, nir_def *stride,
                    int tile_log2_w, int tile_log2_h)
{
   if (tile_log2_w < 0)
      return nir_iadd(b, nir_umul_2x32_64(b, y, stride), nir_u2u64(b, x));

   nir_def *tile =
      nir_iadd(b, nir_imul(b, nir_ushr_imm(b, y, tile_log2_h), stride),
               nir_ushr_imm(b, x, tile_log2_w));
   nir_def *index = nir_ishl_imm(b, tile, tile_log2_w + tile_log2_h);
   for (unsigned bit = 0; bit < tile_log2_w; bit++)
      index =
         nir_ior(b, index, nir_ishl_imm(b, nir_iand_imm(b, x, 1u << bit), bit));
   for (unsigned bit = 0; bit < tile_log2_h; bit++)
      index = nir_ior(b, index,
                      nir_ishl_imm(b, nir_iand_imm(b, y, 1u << bit), bit + 1));
   return nir_u2u64(b, index);
}

static nir_shader *
hk_build_block_copy(const struct hk_block_copy_key *key)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL,
                                                  "hk-meta-block-copy");
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 8;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *id = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, id, 0);
   nir_def *y = nir_channel(&b, id, 1);
   nir_push_if(&b, nir_iand(&b, nir_ult(&b, x, block_push(&b, width)),
                            nir_ult(&b, y, block_push(&b, height))));

   nir_def *src_el = hk_block_copy_index(
      &b, nir_iadd(&b, x, block_push(&b, src_x)),
      nir_iadd(&b, y, block_push(&b, src_y)), block_push(&b, src_stride),
      key->src_tile_log2_w, key->src_tile_log2_h);
   nir_def *dst_el = hk_block_copy_index(
      &b, nir_iadd(&b, x, block_push(&b, dst_x)),
      nir_iadd(&b, y, block_push(&b, dst_y)), block_push(&b, dst_stride),
      key->dst_tile_log2_w, key->dst_tile_log2_h);

   const unsigned src_B = key->to_image ? key->packed_rgb_buffer_B
                                        : key->block_B;
   const unsigned dst_B = key->to_image ? key->block_B
                                        : key->packed_rgb_buffer_B;
   nir_def *src =
      nir_iadd(&b, block_push(&b, src), nir_imul_imm(&b, src_el, src_B));
   nir_def *dst =
      nir_iadd(&b, block_push(&b, dst), nir_imul_imm(&b, dst_el, dst_B));

   const unsigned bits = (key->packed_rgb_buffer_B / 3) * 8;
   const unsigned component_B = bits / 8;
   /* Apple's G17P compiler witnesses the packed transfer as three explicit
    * component loads and stores.  Preserve that access contract instead of
    * relying on a three-vector global operation in the AGX3 backend. */
   for (unsigned c = 0; c < 3; ++c) {
      nir_def *value = nir_load_global(
         &b, 1, bits, nir_iadd_imm(&b, src, c * component_B),
         .align_mul = component_B);
      nir_store_global(&b, value,
                       nir_iadd_imm(&b, dst, c * component_B),
                       .align_mul = component_B);
   }
   nir_pop_if(&b, NULL);
   return b.shader;
}

#undef block_push

static void
hk_dispatch_block_copy(struct hk_cmd_buffer *cmd,
                       const struct hk_block_copy_key *key,
                       const struct hk_block_copy_push *push)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const VkPushConstantRange range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(*push),
   };
   VkPipelineLayout layout;
   VkResult result =
      vk_meta_get_pipeline_layout(&dev->vk, &dev->meta, NULL, &range,
                                  &key->type, sizeof(key->type), &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   VkPipeline pipeline = vk_meta_lookup_pipeline(&dev->meta, key, sizeof(*key));
   if (pipeline == VK_NULL_HANDLE) {
      const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
         .nir = hk_build_block_copy(key),
      };
      const VkComputePipelineCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage =
            {
               .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
               .pNext = &nir_info,
               .stage = VK_SHADER_STAGE_COMPUTE_BIT,
               .pName = "main",
            },
         .layout = layout,
      };
      result = vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &info,
                                               key, sizeof(*key), &pipeline);
      ralloc_free(nir_info.nir);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, result);
         return;
      }
   }

   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkCommandBuffer cb = vk_command_buffer_to_handle(&cmd->vk);
   disp->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   disp->CmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                          sizeof(*push), push);
   disp->CmdDispatch(cb, DIV_ROUND_UP(push->width, 8),
                     DIV_ROUND_UP(push->height, 8), 1);
}

static int
hk_block_copy_tile_log2(const struct ail_layout *layout, unsigned level,
                        bool height)
{
   if (layout->tiling == AIL_TILING_LINEAR)
      return -1;

   const struct ail_tile tile = layout->tilesize_el[level];
   return util_logbase2(height ? tile.height_el : tile.width_el);
}

static uint32_t
hk_block_copy_stride(const struct ail_layout *layout, unsigned level)
{
   if (layout->tiling == AIL_TILING_LINEAR)
      return ail_get_linear_stride_B(layout, level) /
             ail_get_blocksize_B(layout);

   return DIV_ROUND_UP(layout->stride_el[level],
                       layout->tilesize_el[level].width_el);
}

static bool
hk_g17p_copy_padded_rgb_buffer_image(
   struct hk_cmd_buffer *cmd, struct hk_image *image, VkBuffer buffer_handle,
   uint32_t region_count, const VkBufferImageCopy2 *regions,
   VkImageLayout image_layout, bool to_image)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if ((!agx_native_graphics(&dev->dev) && !agx_chip_g17_m5(dev->dev.chip)) ||
       region_count == 0 ||
       image->vk.samples != VK_SAMPLE_COUNT_1_BIT ||
       (image_layout != (to_image ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                                  : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) &&
        image_layout != VK_IMAGE_LAYOUT_GENERAL))
      return false;

   VK_FROM_HANDLE(hk_buffer, buffer, buffer_handle);
   for (unsigned r = 0; r < region_count; ++r) {
      const VkBufferImageCopy2 *region = &regions[r];
      uint8_t plane;
      VkFormat format;
      if (!hk_g17p_buffer_upload_plane(
             image, region->imageSubresource.aspectMask, &plane, &format))
         return false;

      const struct ail_layout *layout = &image->planes[plane].layout;
      uint32_t buffer_block_B, block_B;
      if (!hk_g17p_padded_rgb_layout(image, plane, format, &buffer_block_B,
                                     &block_B) ||
          (layout->tiling != AIL_TILING_GPU &&
           layout->tiling != AIL_TILING_LINEAR) ||
          layout->compressed)
         return false;

      const uint32_t block_w = util_format_get_blockwidth(layout->format);
      const uint32_t block_h = util_format_get_blockheight(layout->format);
      const uint32_t level = region->imageSubresource.mipLevel;
      if (level >= image->vk.mip_levels)
         return false;

      const bool is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D;
      const uint32_t layers = vk_image_subresource_layer_count(
         &image->vk, &region->imageSubresource);
      const uint32_t mip_w = u_minify(layout->width_px, level);
      const uint32_t mip_h = u_minify(layout->height_px, level);
      const uint32_t mip_d = u_minify(image->vk.extent.depth, level);
      const uint32_t z = region->imageOffset.z;
      const uint32_t depth = region->imageExtent.depth;
      const uint32_t slices = is_3d ? depth : layers;
      const uint32_t base_slice =
         is_3d ? z : region->imageSubresource.baseArrayLayer;
      const uint32_t x = region->imageOffset.x;
      const uint32_t y = region->imageOffset.y;
      const uint32_t width = region->imageExtent.width;
      const uint32_t height = region->imageExtent.height;
      const uint32_t row_px =
         region->bufferRowLength ? region->bufferRowLength : width;
      const uint32_t image_h_px =
         region->bufferImageHeight ? region->bufferImageHeight : height;

      if (layers == 0 ||
          region->imageSubresource.baseArrayLayer >= image->vk.array_layers ||
          layers > image->vk.array_layers -
                      region->imageSubresource.baseArrayLayer ||
          region->imageOffset.x < 0 || region->imageOffset.y < 0 ||
          region->imageOffset.z < 0 || !depth ||
          (is_3d ? (region->imageSubresource.baseArrayLayer != 0 ||
                    layers != 1 || z >= mip_d || depth > mip_d - z)
                 : (z != 0 || depth != 1)) ||
          (image->vk.image_type == VK_IMAGE_TYPE_1D &&
           (y != 0 || height != 1)) ||
          !width || !height || x >= mip_w || y >= mip_h ||
          width > mip_w - x || height > mip_h - y || (x % block_w) ||
          (y % block_h) ||
          ((width % block_w) && x + width != mip_w) ||
          ((height % block_h) && y + height != mip_h) || row_px < width ||
          image_h_px < height ||
          (region->bufferOffset % (buffer_block_B / 3)))
         return false;

      const uint32_t blocks_x =
         util_format_get_nblocksx(layout->format, width);
      const uint32_t blocks_y =
         util_format_get_nblocksy(layout->format, height);
      const uint64_t buffer_row_B =
         (uint64_t)util_format_get_nblocksx(layout->format, row_px) *
         buffer_block_B;
      const uint64_t buffer_layer_B =
         (uint64_t)util_format_get_nblocksy(layout->format, image_h_px) *
         buffer_row_B;
      /* Vulkan requires the buffer to hold only the bytes the copy actually
       * addresses, not whole padded layers. The dispatch below walks
       * blocks_x x blocks_y blocks per slice at buffer_base + slice *
       * buffer_layer_B with row stride buffer_row_B, so the last byte it
       * touches is:
       *
       *   (slices - 1) layer strides
       * + (blocks_y - 1) row strides
       * + blocks_x blocks of the final row
       *
       * Bounding on buffer_layer_B instead demanded a whole padded layer from
       * bufferOffset, which rejected every legal tightly-sized copy --
       * dEQP-VK.api.copy_and_blit.*.tightly_sized_buffer_offset_* and
       * use_after_copy.r8g8b8_unorm.* -- and, because this path defers its
       * failure, surfaced as VK_ERROR_FEATURE_NOT_PRESENT from
       * vkEndCommandBuffer. AGX2 reports those cases NotSupported because it
       * does not advertise the format at all; G17P advertises it and then
       * refused, which is worse.
       */
      if (!slices)
         return false;
      const uint64_t copied_B = (uint64_t)(slices - 1) * buffer_layer_B +
                                (uint64_t)(blocks_y - 1) * buffer_row_B +
                                (uint64_t)blocks_x * buffer_block_B;
      if (region->bufferOffset > buffer->vk.size ||
          copied_B > buffer->vk.size - region->bufferOffset)
         return false;

      const uint64_t buffer_base =
         to_image ? hk_buffer_address_ro(buffer, region->bufferOffset)
                  : hk_buffer_address_rw(buffer, region->bufferOffset);
      const uint64_t image_base = hk_image_base_address(image, plane);
      struct hk_block_copy_key key = {
         .type = HK_META_OBJECT_KEY_BLOCK_COPY,
         .block_B = block_B,
         .packed_rgb_buffer_B = buffer_block_B,
         .to_image = to_image,
         .src_tile_log2_w = -1,
         .src_tile_log2_h = -1,
         .dst_tile_log2_w = hk_block_copy_tile_log2(layout, level, false),
         .dst_tile_log2_h = hk_block_copy_tile_log2(layout, level, true),
      };
      if (!to_image) {
         key.src_tile_log2_w = key.dst_tile_log2_w;
         key.src_tile_log2_h = key.dst_tile_log2_h;
         key.dst_tile_log2_w = key.dst_tile_log2_h = -1;
      }

      for (uint32_t slice = 0; slice < slices; ++slice) {
         struct hk_block_copy_push push = {
            .src = buffer_base + slice * buffer_layer_B,
            .dst = image_base +
                   ail_get_layer_level_B(layout, base_slice + slice, level),
            .src_stride = buffer_row_B / buffer_block_B,
            .dst_stride = hk_block_copy_stride(layout, level),
            .dst_x = x / block_w,
            .dst_y = y / block_h,
            .width = blocks_x,
            .height = blocks_y,
         };
         if (!to_image) {
            uint64_t address = push.src;
            push.src = push.dst;
            push.dst = address;
            uint32_t stride = push.src_stride;
            push.src_stride = push.dst_stride;
            push.dst_stride = stride;
            push.src_x = push.dst_x;
            push.src_y = push.dst_y;
            push.dst_x = push.dst_y = 0;
         }

         hk_dispatch_block_copy(cmd, &key, &push);
         if (vk_command_buffer_has_error(&cmd->vk))
            return true;
      }

      if (cmd->current_cs.cs) {
         const uint64_t dst = to_image ? image_base : buffer_base;
         /* Declare the range the dispatch actually touches. Whole padded
          * layers would now run past the end of a tightly-sized buffer. */
         const uint64_t size = to_image ? layout->size_B : copied_B;
         hk_drm_attachment_set_add(&cmd->current_cs.cs->compute_attachments,
                                   dst, size);
      }
   }

   return true;
}

/* G17P uploads into a plain GPU-tiled image walk the destination in memory
 * (Morton) order instead of libagx_copy_image's row-major byte loop. Each
 * workgroup of 64 covers one run of 64 * elements_per_thread consecutive
 * elements of one tile, so every store instruction of a SIMD group writes 32
 * adjacent elements, whole 128-byte lines from 4-byte elements up, and each
 * element moves as a single load and store of its full size. On the A18 Pro
 * this cuts a 4096x4096 RGBA8 upload by about 35 %, close to a plain
 * buffer-to-buffer copy of the same size. Tiles that are neither square nor
 * twice as wide as tall, or smaller than 64 elements, keep libagx_copy_image.
 */
#define HK_G17P_TILED_UPLOAD_ELEMENTS_PER_THREAD 4

struct hk_g17p_tiled_upload_key {
   uint32_t object;
   uint32_t block_B;
   uint32_t elements_per_thread;
};

struct hk_g17p_tiled_upload_push {
   uint64_t src, dst;
   uint32_t src_pitch_B;
   uint32_t x0, y0, width, height;
   uint32_t tiles_per_row;
   uint32_t tile_x0, tile_y0;
   uint32_t tile_w_log2, tile_h_log2;
};
static_assert(sizeof(struct hk_g17p_tiled_upload_push) == 56,
              "packed push data");

#define tiled_upload_push(b, field)                                            \
   nir_load_push_constant(                                                     \
      b, 1, sizeof(((struct hk_g17p_tiled_upload_push *)0)->field) * 8,        \
      nir_imm_int(b, offsetof(struct hk_g17p_tiled_upload_push, field)))

/* The even bits of the low 16 bits of v, packed. */
static nir_def *
hk_g17p_compact_even16(nir_builder *b, nir_def *v)
{
   v = nir_iand_imm(b, v, 0x5555);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ushr_imm(b, v, 1)), 0x3333);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ushr_imm(b, v, 2)), 0x0f0f);
   return nir_iand_imm(b, nir_ior(b, v, nir_ushr_imm(b, v, 4)), 0x00ff);
}

static nir_shader *
hk_g17p_build_tiled_upload(const struct hk_g17p_tiled_upload_key *key)
{
   nir_builder build = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, NULL, "hk-meta-g17p-tiled-upload");
   nir_builder *b = &build;
   b->shader->info.workgroup_size[0] = 64;
   b->shader->info.workgroup_size[1] = 1;
   b->shader->info.workgroup_size[2] = 1;

   const unsigned bits = MIN2(key->block_B, 4) * 8;
   const unsigned comps = MAX2(key->block_B / 4, 1);

   nir_def *lid = nir_channel(b, nir_load_local_invocation_id(b), 0);
   nir_def *wg = nir_load_workgroup_id(b);
   nir_def *twl = tiled_upload_push(b, tile_w_log2);
   nir_def *thl = tiled_upload_push(b, tile_h_log2);
   nir_def *tile_x =
      nir_iadd(b, tiled_upload_push(b, tile_x0), nir_channel(b, wg, 1));
   nir_def *tile_y =
      nir_iadd(b, tiled_upload_push(b, tile_y0), nir_channel(b, wg, 2));
   nir_def *tile_el = nir_ishl(
      b,
      nir_iadd(b, nir_imul(b, tile_y, tiled_upload_push(b, tiles_per_row)),
               tile_x),
      nir_iadd(b, twl, thl));
   nir_def *origin_x = nir_ishl(b, tile_x, twl);
   nir_def *origin_y = nir_ishl(b, tile_y, thl);
   /* Morton order: X in the even bits, Y in the odd bits. */
   nir_def *lane_x = hk_g17p_compact_even16(b, lid);
   nir_def *lane_y = hk_g17p_compact_even16(b, nir_ushr_imm(b, lid, 1));
   nir_def *x0 = tiled_upload_push(b, x0), *y0 = tiled_upload_push(b, y0);
   nir_def *width = tiled_upload_push(b, width);
   nir_def *height = tiled_upload_push(b, height);
   nir_def *pitch = tiled_upload_push(b, src_pitch_B);
   nir_def *src = tiled_upload_push(b, src), *dst = tiled_upload_push(b, dst);

   for (unsigned j = 0; j < key->elements_per_thread; ++j) {
      /* The index of this 64-element run within the tile. */
      nir_def *run = nir_iadd_imm(
         b, nir_imul_imm(b, nir_channel(b, wg, 0), key->elements_per_thread),
         j);
      nir_def *m = nir_ior(b, nir_ishl_imm(b, run, 6), lid);
      nir_def *x = nir_ior(
         b, origin_x,
         nir_ior(b, nir_ishl_imm(b, hk_g17p_compact_even16(b, run), 3),
                 lane_x));
      nir_def *y = nir_ior(
         b, origin_y,
         nir_ior(b,
                 nir_ishl_imm(b, hk_g17p_compact_even16(
                                    b, nir_ushr_imm(b, run, 1)), 3),
                 lane_y));
      nir_def *rx = nir_isub(b, x, x0), *ry = nir_isub(b, y, y0);
      nir_push_if(b, nir_iand(b, nir_ult(b, rx, width), nir_ult(b, ry, height)));
      nir_def *src_off = nir_iadd(b, nir_imul(b, ry, pitch),
                                  nir_imul_imm(b, rx, key->block_B));
      nir_def *data =
         nir_load_global(b, comps, bits, nir_iadd(b, src, nir_u2u64(b, src_off)),
                         .align_mul = key->block_B,
                         .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
      nir_def *dst_off =
         nir_imul_imm(b, nir_iadd(b, tile_el, m), key->block_B);
      nir_store_global(b, data, nir_iadd(b, dst, nir_u2u64(b, dst_off)),
                       .align_mul = key->block_B);
      nir_pop_if(b, NULL);
   }

   return b->shader;
}

/* Whether this upload takes hk_g17p_tiled_upload(), and its shape. */
static bool
hk_g17p_tiled_upload_shape(const struct hk_device *dev, bool dest_tiled,
                           const struct hk_g17p_tile_address_config *tile,
                           const struct hk_g17p_uncompressed_copy_plan *plan,
                           struct hk_g17p_tiled_upload_key *key,
                           uint32_t groups[3])
{
   if (dev->dev.chip != AGX_CHIP_G17P || !dest_tiled || !plan->width_el ||
       !plan->height_el)
      return false;

   /* Morton order is dense only for square tiles or tiles twice as wide. */
   const uint32_t twl = tile->tile_width_log2, thl = tile->tile_height_log2;
   const uint32_t area_log2 = twl + thl;
   if ((twl != thl && twl != thl + 1) || area_log2 < 6)
      return false;

   const uint32_t per_thread =
      MIN2(HK_G17P_TILED_UPLOAD_ELEMENTS_PER_THREAD, 1u << (area_log2 - 6));
   const uint32_t tx0 = plan->base_x_el >> twl;
   const uint32_t ty0 = plan->base_y_el >> thl;
   const uint64_t x1 = (uint64_t)plan->base_x_el + plan->width_el - 1;
   const uint64_t y1 = (uint64_t)plan->base_y_el + plan->height_el - 1;
   const struct vk_physical_device *pdev = dev->vk.physical;
   groups[0] = 1u << (area_log2 - 6 - util_logbase2(per_thread));
   groups[1] = (x1 >> twl) - tx0 + 1;
   groups[2] = (y1 >> thl) - ty0 + 1;
   for (unsigned i = 0; i < 3; ++i) {
      if (groups[i] > pdev->properties.maxComputeWorkGroupCount[i])
         return false;
   }

   /* The shader addresses source and destination with 32-bit offsets. */
   const uint64_t src_span = (uint64_t)(plan->height_el - 1) *
                                plan->buffer_row_stride_B +
                             (uint64_t)plan->width_el * plan->block_size_B;
   const uint64_t dst_end =
      (((uint64_t)(y1 >> thl) * tile->tiles_per_row + (x1 >> twl) + 1)
       << area_log2) * plan->block_size_B;
   if (src_span > UINT32_MAX || dst_end > UINT32_MAX ||
       plan->buffer_row_stride_B > UINT32_MAX)
      return false;

   *key = (struct hk_g17p_tiled_upload_key){
      .object = HK_META_OBJECT_KEY_G17P_TILED_UPLOAD,
      .block_B = plan->block_size_B,
      .elements_per_thread = per_thread,
   };
   return true;
}

static void
hk_g17p_tiled_upload(struct hk_cmd_buffer *cmd,
                     const struct hk_g17p_tiled_upload_key *key,
                     const uint32_t groups[3],
                     const struct hk_g17p_tiled_upload_push *push)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(*push),
   };
   VkPipelineLayout layout;
   VkResult result = vk_meta_get_pipeline_layout(
      &dev->vk, &dev->meta, NULL, &push_range, key, sizeof(*key), &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   VkPipeline pipeline = vk_meta_lookup_pipeline(&dev->meta, key, sizeof(*key));
   if (pipeline == VK_NULL_HANDLE) {
      const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
         .nir = hk_g17p_build_tiled_upload(key),
      };
      const VkComputePipelineCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &nir_info,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = "main",
         },
         .layout = layout,
      };
      result = vk_meta_create_compute_pipeline(
         &dev->vk, &dev->meta, &info, key, sizeof(*key), &pipeline);
      ralloc_free(nir_info.nir);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, result);
         return;
      }
   }

   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkCommandBuffer cb = vk_command_buffer_to_handle(&cmd->vk);
   disp->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   disp->CmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                          sizeof(*push), push);
   disp->CmdDispatch(cb, groups[0], groups[1], groups[2]);
}

static bool
hk_g17p_copy_uncompressed_buffer_to_image_supported(
   struct hk_device *dev, const struct hk_buffer *buffer,
   const struct hk_image *image, const VkCopyBufferToImageInfo2 *info)
{
   if ((!agx_native_graphics(&dev->dev) &&
        !agx_chip_g17_m5(dev->dev.chip)) || info->regionCount == 0)
      return false;

   for (uint32_t r = 0; r < info->regionCount; ++r) {
      const VkBufferImageCopy2 *region = &info->pRegions[r];
      uint8_t plane;
      VkFormat format;
      if (!hk_g17p_buffer_upload_plane(
             image, region->imageSubresource.aspectMask, &plane, &format))
         return false;

      const struct ail_layout *layout = &image->planes[plane].layout;
      if ((layout->tiling != AIL_TILING_GPU &&
           layout->tiling != AIL_TILING_LINEAR) ||
          layout->compressed ||
          (layout->tiling == AIL_TILING_LINEAR && image->vk.mip_levels != 1))
         return false;

      const struct hk_g17p_buffer_to_image_copy copy =
         hk_g17p_buffer_upload_desc(dev, image, info, region, plane, format);
      struct hk_g17p_uncompressed_copy_plan plan;
      if (!hk_g17p_uncompressed_buffer_to_image_copy_plan(
             &copy, region->bufferOffset, buffer->vk.size, &plan))
         return false;
   }

   return true;
}

static void
hk_g17p_copy_uncompressed_buffer_to_image(
   struct hk_cmd_buffer *cmd, struct hk_buffer *buffer,
   struct hk_image *image, const VkCopyBufferToImageInfo2 *info)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   for (uint32_t r = 0; r < info->regionCount; ++r) {
      const VkBufferImageCopy2 *region = &info->pRegions[r];
      uint8_t plane;
      VkFormat format;
      if (!hk_g17p_buffer_upload_plane(
             image, region->imageSubresource.aspectMask, &plane, &format)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      const struct ail_layout *layout = &image->planes[plane].layout;
      const uint64_t base_dst = hk_image_base_address(image, plane);
      const bool dest_tiled = layout->tiling == AIL_TILING_GPU;
      const struct hk_g17p_buffer_to_image_copy copy =
         hk_g17p_buffer_upload_desc(dev, image, info, region, plane, format);
      struct hk_g17p_uncompressed_copy_plan plan;
      if (!hk_g17p_uncompressed_buffer_to_image_copy_plan(
             &copy, region->bufferOffset, buffer->vk.size, &plan)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      const uint32_t level = region->imageSubresource.mipLevel;
      struct hk_g17p_tile_address_config tile_config = {0};
      if (dest_tiled) {
         const struct ail_tile tile = layout->tilesize_el[level];
         if (!hk_g17p_tile_address_config_init(
                layout->stride_el[level], tile.width_el, tile.height_el,
                &tile_config)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
            return;
         }
      }
      const uint64_t base_src =
         hk_buffer_address_ro(buffer, region->bufferOffset);
      const struct libagx_copy_image_args args_template = {
         .width_el = plan.width_el,
         .height_el = plan.height_el,
         .dest_x_el = plan.base_x_el,
         .dest_y_el = plan.base_y_el,
         .dest_pitch_B =
            dest_tiled ? 0 : ail_get_linear_stride_B(layout, level),
         .src_pitch_B = plan.buffer_row_stride_B,
         .dest_tiles_per_row = tile_config.tiles_per_row,
         .dest_tile_width_log2 = tile_config.tile_width_log2,
         .dest_tile_height_log2 = tile_config.tile_height_log2,
         .blocksize_B = plan.block_size_B,
         .dest_tiled = dest_tiled,
         .src_tiled = false,
      };
      struct hk_g17p_tiled_upload_key tiled_key;
      uint32_t tiled_groups[3];
      const bool tiled = hk_g17p_tiled_upload_shape(
         dev, dest_tiled, &tile_config, &plan, &tiled_key, tiled_groups);

      for (uint32_t slice = 0; slice < plan.slice_count; ++slice) {
         struct libagx_copy_image_args args = args_template;
         args.dest = base_dst +
                     ail_get_layer_level_B(layout, plan.first_slice + slice,
                                           level);
         args.src = base_src + slice * plan.buffer_slice_stride_B;

         if (tiled) {
            const struct hk_g17p_tiled_upload_push push = {
               .src = args.src,
               .dst = args.dest,
               .src_pitch_B = plan.buffer_row_stride_B,
               .x0 = plan.base_x_el,
               .y0 = plan.base_y_el,
               .width = plan.width_el,
               .height = plan.height_el,
               .tiles_per_row = tile_config.tiles_per_row,
               .tile_x0 = plan.base_x_el >> tile_config.tile_width_log2,
               .tile_y0 = plan.base_y_el >> tile_config.tile_height_log2,
               .tile_w_log2 = tile_config.tile_width_log2,
               .tile_h_log2 = tile_config.tile_height_log2,
            };
            hk_g17p_tiled_upload(cmd, &tiled_key, tiled_groups, &push);
            if (vk_command_buffer_has_error(&cmd->vk))
               return;
            continue;
         }

         /* Upload regions and layers are independent.  The consumer's
          * transfer barrier supplies ordering before mip generation or
          * sampling, so do not serialize the upload batch itself. */
         libagx_copy_image_struct(
            cmd, agx_2d(plan.width_el, plan.height_el), AGX_BARRIER_NONE,
            args);
         if (vk_command_buffer_has_error(&cmd->vk))
            return;
      }

      /* Declare the tiled upload's writes as the other compute meta copies
       * do, for kernels that pass compute attachments on as write ranges. */
      if (tiled && cmd->current_cs.cs)
         hk_drm_attachment_set_add(&cmd->current_cs.cs->compute_attachments,
                                   base_dst, layout->size_B);
   }
}

static bool
hk_g17p_copy_compressed_buffer_to_image(
   struct hk_cmd_buffer *cmd, struct hk_image *image,
   const VkCopyBufferToImageInfo2 *info)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!hk_g17p_block_copy_format(dev, image->vk.format) ||
       image->plane_count != 1 ||
       image->planes[0].layout.tiling != AIL_TILING_GPU ||
       !hk_g17p_transfer_dst_layout_supported(info->dstImageLayout))
      return false;

   VK_FROM_HANDLE(hk_buffer, buffer, info->srcBuffer);

   for (unsigned r = 0; r < info->regionCount; ++r) {
      const VkBufferImageCopy2 *region = &info->pRegions[r];
      const uint32_t level = region->imageSubresource.mipLevel;
      struct hk_g17p_compressed_copy_plan plan;
      if (!hk_g17p_compressed_buffer_to_image_copy_plan(
             &(const struct hk_g17p_buffer_to_image_copy){
                .chip = dev->dev.chip,
                .format = image->vk.format,
                .layout = info->dstImageLayout,
                .image_type = image->vk.image_type,
                .samples = image->vk.samples,
                .plane_count = image->plane_count,
                .mip_levels = image->vk.mip_levels,
                .array_layers = image->vk.array_layers,
                .image_extent = image->vk.extent,
                .aspect = region->imageSubresource.aspectMask,
                .mip_level = level,
                .base_array_layer =
                   region->imageSubresource.baseArrayLayer,
                .layer_count = region->imageSubresource.layerCount,
                .image_offset = region->imageOffset,
                .copy_extent = region->imageExtent,
                .buffer_row_length = region->bufferRowLength,
                .buffer_image_height = region->bufferImageHeight,
             },
             region->bufferOffset, buffer->vk.size, &plan))
         return false;

      const uint64_t base_src =
         hk_buffer_address_ro(buffer, region->bufferOffset);
      const uint64_t base_dst = hk_image_base_address(image, 0);

      const struct ail_layout *layout = &image->planes[0].layout;
      const struct ail_tile tile = layout->tilesize_el[level];
      struct hk_g17p_tile_address_config tile_config;
      if (!hk_g17p_tile_address_config_init(
             layout->stride_el[level], tile.width_el, tile.height_el,
             &tile_config)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
         return true;
      }
      const struct libagx_copy_image_args args_template = {
         .width_el = plan.blocks_x,
         .height_el = plan.blocks_y,
         .dest_x_el = plan.base_x_el,
         .dest_y_el = plan.base_y_el,
         .src_pitch_B = plan.buffer_row_stride_B,
         .dest_tiles_per_row = tile_config.tiles_per_row,
         .dest_tile_width_log2 = tile_config.tile_width_log2,
         .dest_tile_height_log2 = tile_config.tile_height_log2,
         .blocksize_B = plan.block_size_B,
         .dest_tiled = true,
         .src_tiled = false,
      };

      for (uint32_t slice = 0; slice < plan.slice_count; ++slice) {
         const uint32_t dst_slice = plan.first_slice + slice;
         struct libagx_copy_image_args args = args_template;
         args.dest = base_dst + ail_get_layer_level_B(layout, dst_slice, level);
         args.src = base_src + slice * plan.buffer_slice_stride_B;

         /* One precompiled kernel walks the complete rectangle in AIL's
          * Morton order. The former nested loop dispatched a separate generic
          * vk_meta copy for every 8- or 16-byte compressed block, repeatedly
          * rebuilding and uploading the same Ioto program during STK texture
          * loading. Keep the same weak inter-dispatch barrier as that path;
          * slices are independent. */
         libagx_copy_image_struct(
            cmd, agx_2d(plan.blocks_x, plan.blocks_y), AGX_BARRIER_NONE,
            args);
         if (vk_command_buffer_has_error(&cmd->vk))
            return true;
      }
   }

   return true;
}

static bool
hk_g17p_copy_compressed_image_to_buffer(
   struct hk_cmd_buffer *cmd, struct hk_image *image,
   const VkCopyImageToBufferInfo2 *info)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!hk_g17p_block_copy_format(dev, image->vk.format) ||
       image->plane_count != 1 ||
       image->planes[0].layout.tiling != AIL_TILING_GPU ||
       !hk_g17p_transfer_src_layout_supported(info->srcImageLayout))
      return false;

   VK_FROM_HANDLE(hk_buffer, buffer, info->dstBuffer);

   for (unsigned r = 0; r < info->regionCount; ++r) {
      const VkBufferImageCopy2 *region = &info->pRegions[r];
      const uint32_t level = region->imageSubresource.mipLevel;
      struct hk_g17p_compressed_copy_plan plan;
      if (!hk_g17p_compressed_image_to_buffer_copy_plan(
             &(const struct hk_g17p_buffer_to_image_copy){
                .chip = dev->dev.chip,
                .format = image->vk.format,
                .layout = info->srcImageLayout,
                .image_type = image->vk.image_type,
                .samples = image->vk.samples,
                .plane_count = image->plane_count,
                .mip_levels = image->vk.mip_levels,
                .array_layers = image->vk.array_layers,
                .image_extent = image->vk.extent,
                .aspect = region->imageSubresource.aspectMask,
                .mip_level = level,
                .base_array_layer =
                   region->imageSubresource.baseArrayLayer,
                .layer_count = region->imageSubresource.layerCount,
                .image_offset = region->imageOffset,
                .copy_extent = region->imageExtent,
                .buffer_row_length = region->bufferRowLength,
                .buffer_image_height = region->bufferImageHeight,
             },
             region->bufferOffset, buffer->vk.size, &plan))
         return false;

      const uint64_t base_src = hk_image_base_address(image, 0);
      const uint64_t base_dst =
         hk_buffer_address_rw(buffer, region->bufferOffset);

      const struct ail_layout *layout = &image->planes[0].layout;
      const struct ail_tile tile = layout->tilesize_el[level];
      struct hk_g17p_tile_address_config tile_config;
      if (!hk_g17p_tile_address_config_init(
             layout->stride_el[level], tile.width_el, tile.height_el,
             &tile_config)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_UNKNOWN);
         return true;
      }
      const struct libagx_copy_image_args args_template = {
         .width_el = plan.blocks_x,
         .height_el = plan.blocks_y,
         .src_x_el = plan.base_x_el,
         .src_y_el = plan.base_y_el,
         .dest_pitch_B = plan.buffer_row_stride_B,
         .src_tiles_per_row = tile_config.tiles_per_row,
         .src_tile_width_log2 = tile_config.tile_width_log2,
         .src_tile_height_log2 = tile_config.tile_height_log2,
         .blocksize_B = plan.block_size_B,
         .dest_tiled = false,
         .src_tiled = true,
      };

      for (uint32_t slice = 0; slice < plan.slice_count; ++slice) {
         const uint32_t src_slice = plan.first_slice + slice;
         struct libagx_copy_image_args args = args_template;
         args.dest = base_dst + slice * plan.buffer_slice_stride_B;
         args.src = base_src + ail_get_layer_level_B(layout, src_slice, level);

         libagx_copy_image_struct(
            cmd, agx_2d(plan.blocks_x, plan.blocks_y), AGX_BARRIER_NONE,
            args);
         if (vk_command_buffer_has_error(&cmd->vk))
            return true;
      }
   }

   return true;
}

static bool
hk_g17p_raw_image_copy_supported(struct hk_device *dev,
                                 const struct hk_image *src,
                                 const struct hk_image *dst,
                                 const VkCopyImageInfo2 *info)
{
   if ((!agx_native_graphics(&dev->dev) && !agx_chip_g17_m5(dev->dev.chip)) ||
       info->regionCount == 0 ||
       src->vk.samples != dst->vk.samples ||
       (src->vk.samples != VK_SAMPLE_COUNT_1_BIT &&
        src->vk.samples != VK_SAMPLE_COUNT_2_BIT &&
        src->vk.samples != VK_SAMPLE_COUNT_4_BIT) ||
       !hk_g17p_transfer_src_layout_supported(info->srcImageLayout) ||
       !hk_g17p_transfer_dst_layout_supported(info->dstImageLayout))
      return false;

   for (uint32_t r = 0; r < info->regionCount; ++r) {
      const VkImageCopy2 *region = &info->pRegions[r];
      struct hk_g17p_copy_aspect_plan aspects;
      if (!hk_g17p_copy_aspect_plan(region->srcSubresource.aspectMask,
                                    region->dstSubresource.aspectMask,
                                    &aspects) ||
          region->srcSubresource.mipLevel >= src->vk.mip_levels ||
          region->dstSubresource.mipLevel >= dst->vk.mip_levels)
         return false;

      for (unsigned a = 0; a < aspects.count; ++a) {
         const uint8_t src_plane =
            hk_image_aspects_to_plane(src, aspects.src[a]);
         const uint8_t dst_plane =
            hk_image_aspects_to_plane(dst, aspects.dst[a]);
         if (src_plane >= src->plane_count || dst_plane >= dst->plane_count)
            return false;

         const VkFormat src_format =
            aspect_format(src->vk.format, aspects.src[a]);
         const VkFormat dst_format =
            aspect_format(dst->vk.format, aspects.dst[a]);
         struct hk_g17p_image_copy_element_plan plan;
         if (!hk_g17p_image_copy_element_plan(
                src_format, dst_format, region, &plan))
            return false;

         const struct ail_layout *src_layout =
            &src->planes[src_plane].layout;
         const struct ail_layout *dst_layout =
            &dst->planes[dst_plane].layout;
         uint32_t storage_B;
         if (!hk_g17p_image_copy_storage_block(
                plan.block_size_B, ail_get_blocksize_B(src_layout),
                ail_get_blocksize_B(dst_layout), &storage_B) ||
             (uint64_t)storage_B * src->vk.samples > 16)
            return false;
         if ((src_layout->tiling != AIL_TILING_GPU &&
              src_layout->tiling != AIL_TILING_LINEAR) ||
             (dst_layout->tiling != AIL_TILING_GPU &&
              dst_layout->tiling != AIL_TILING_LINEAR) ||
             src_layout->compressed || dst_layout->compressed)
            return false;

         const uint32_t src_level = region->srcSubresource.mipLevel;
         const uint32_t dst_level = region->dstSubresource.mipLevel;
         const struct ail_tile src_tile = src_layout->tilesize_el[src_level];
         const struct ail_tile dst_tile = dst_layout->tilesize_el[dst_level];
         struct hk_g17p_tile_address_config src_tile_config;
         struct hk_g17p_tile_address_config dst_tile_config;
         if ((src_layout->tiling == AIL_TILING_GPU &&
              !hk_g17p_tile_address_config_init(
                 src_layout->stride_el[src_level], src_tile.width_el,
                 src_tile.height_el, &src_tile_config)) ||
             (dst_layout->tiling == AIL_TILING_GPU &&
              !hk_g17p_tile_address_config_init(
                 dst_layout->stride_el[dst_level], dst_tile.width_el,
                 dst_tile.height_el, &dst_tile_config)))
            return false;
      }

      const uint32_t src_slices =
         src->vk.image_type == VK_IMAGE_TYPE_3D
            ? region->extent.depth
            : vk_image_subresource_layer_count(&src->vk,
                                                &region->srcSubresource);
      const uint32_t dst_slices =
         dst->vk.image_type == VK_IMAGE_TYPE_3D
            ? region->extent.depth
            : vk_image_subresource_layer_count(&dst->vk,
                                                &region->dstSubresource);
      VkImageCopy2 last;
      if (!src_slices || !hk_g17p_image_copy_slice(
            dev->dev.chip, src->vk.image_type, dst->vk.image_type,
            src_slices, dst_slices, region, src_slices - 1, &last))
         return false;

   }

   return true;
}

static void
hk_g17p_copy_raw_image(struct hk_cmd_buffer *cmd, struct hk_image *src,
                       struct hk_image *dst, const VkCopyImageInfo2 *info)
{
   for (uint32_t r = 0; r < info->regionCount; ++r) {
      const VkImageCopy2 *region = &info->pRegions[r];
      const uint32_t src_level = region->srcSubresource.mipLevel;
      const uint32_t dst_level = region->dstSubresource.mipLevel;
      const uint32_t src_slices =
         src->vk.image_type == VK_IMAGE_TYPE_3D
            ? region->extent.depth
            : vk_image_subresource_layer_count(&src->vk,
                                                &region->srcSubresource);
      const uint32_t dst_slices =
         dst->vk.image_type == VK_IMAGE_TYPE_3D
            ? region->extent.depth
            : vk_image_subresource_layer_count(&dst->vk,
                                                &region->dstSubresource);
      struct hk_g17p_copy_aspect_plan aspects;
      const bool mapped = hk_g17p_copy_aspect_plan(
         region->srcSubresource.aspectMask, region->dstSubresource.aspectMask,
         &aspects);
      assert(mapped);
      for (unsigned a = 0; a < aspects.count; ++a) {
         const uint8_t src_plane =
            hk_image_aspects_to_plane(src, aspects.src[a]);
         const uint8_t dst_plane =
            hk_image_aspects_to_plane(dst, aspects.dst[a]);
         const VkFormat src_format =
            aspect_format(src->vk.format, aspects.src[a]);
         const VkFormat dst_format =
            aspect_format(dst->vk.format, aspects.dst[a]);
         struct hk_g17p_image_copy_element_plan plan;
         const bool planned = hk_g17p_image_copy_element_plan(
            src_format, dst_format, region, &plan);
         assert(planned);

         const struct ail_layout *src_layout =
            &src->planes[src_plane].layout;
         const struct ail_layout *dst_layout =
            &dst->planes[dst_plane].layout;
         /* A raw copy moves storage, not API texels: a three-component image
          * is stored four-component, so the element the tiled addressing
          * indexes is wider than the packed Vulkan block. */
         uint32_t storage_B;
         const bool sized = hk_g17p_image_copy_storage_block(
            plan.block_size_B, ail_get_blocksize_B(src_layout),
            ail_get_blocksize_B(dst_layout), &storage_B);
         assert(sized);
         const uint32_t raw_block_size_B = storage_B * src->vk.samples;
         const struct ail_tile src_tile = src_layout->tilesize_el[src_level];
         const struct ail_tile dst_tile = dst_layout->tilesize_el[dst_level];
         const bool src_tiled = src_layout->tiling == AIL_TILING_GPU;
         const bool dst_tiled = dst_layout->tiling == AIL_TILING_GPU;
         struct hk_g17p_tile_address_config src_tile_config = {0};
         struct hk_g17p_tile_address_config dst_tile_config = {0};
         const bool addressed =
            (!src_tiled || hk_g17p_tile_address_config_init(
                               src_layout->stride_el[src_level],
                               src_tile.width_el, src_tile.height_el,
                               &src_tile_config)) &&
            (!dst_tiled || hk_g17p_tile_address_config_init(
                               dst_layout->stride_el[dst_level],
                               dst_tile.width_el, dst_tile.height_el,
                               &dst_tile_config));
         assert(addressed);

         for (uint32_t slice = 0; slice < src_slices; ++slice) {
            VkImageCopy2 one;
            const bool sliced = hk_g17p_image_copy_slice(
               AGX_CHIP_G17P, src->vk.image_type, dst->vk.image_type,
               src_slices, dst_slices, region, slice, &one);
            assert(sliced);
            const uint32_t src_slice =
               src->vk.image_type == VK_IMAGE_TYPE_3D
                  ? one.srcOffset.z
                  : one.srcSubresource.baseArrayLayer;
            const uint32_t dst_slice =
               dst->vk.image_type == VK_IMAGE_TYPE_3D
                  ? one.dstOffset.z
                  : one.dstSubresource.baseArrayLayer;
            const struct libagx_copy_image_args args = {
               .dest = hk_image_base_address(dst, dst_plane) +
                       ail_get_layer_level_B(dst_layout, dst_slice, dst_level),
               .src = hk_image_base_address(src, src_plane) +
                      ail_get_layer_level_B(src_layout, src_slice, src_level),
               .width_el = plan.width_el,
               .height_el = plan.height_el,
               .dest_x_el = plan.dst_x_el,
               .dest_y_el = plan.dst_y_el,
               .src_x_el = plan.src_x_el,
               .src_y_el = plan.src_y_el,
               .dest_pitch_B = dst_tiled
                                  ? 0
                                  : ail_get_linear_stride_B(dst_layout,
                                                            dst_level),
               .src_pitch_B = src_tiled
                                 ? 0
                                 : ail_get_linear_stride_B(src_layout,
                                                           src_level),
               .dest_tiles_per_row = dst_tile_config.tiles_per_row,
               .src_tiles_per_row = src_tile_config.tiles_per_row,
               .dest_tile_width_log2 = dst_tile_config.tile_width_log2,
               .dest_tile_height_log2 = dst_tile_config.tile_height_log2,
               .src_tile_width_log2 = src_tile_config.tile_width_log2,
               .src_tile_height_log2 = src_tile_config.tile_height_log2,
               .blocksize_B = raw_block_size_B,
               .dest_tiled = dst_tiled,
               .src_tiled = src_tiled,
            };

            libagx_copy_image_struct(
               cmd, agx_2d(plan.width_el, plan.height_el), AGX_BARRIER_NONE,
               args);
            if (vk_command_buffer_has_error(&cmd->vk))
               return;
         }
      }
   }
}

static bool
hk_bind_point(bool gfx)
{
   return gfx ? VK_PIPELINE_BIND_POINT_GRAPHICS
              : VK_PIPELINE_BIND_POINT_COMPUTE;
}

static void
hk_xfer_copybuffertoimage2(VkCommandBuffer commandBuffer,
                         const VkCopyBufferToImageInfo2 *pCopyBufferToImageInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, src_buffer, pCopyBufferToImageInfo->srcBuffer);
   VK_FROM_HANDLE(hk_image, dst_image, pCopyBufferToImageInfo->dstImage);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* Share the native raw tiled upload with M5. It preserves partial
    * rectangles without an imageblock round trip or extra firmware job. */
   if (agx_chip_g17_m5(dev->dev.chip) &&
       hk_g17p_copy_uncompressed_buffer_to_image_supported(
          dev, src_buffer, dst_image, pCopyBufferToImageInfo)) {
      struct hk_meta_save save;
      hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      hk_g17p_copy_uncompressed_buffer_to_image(
         cmd, src_buffer, dst_image, pCopyBufferToImageInfo);
      hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      return;
   }

   bool gfx = hk_copy_requires_gfx(dst_image);
   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
      uint8_t plane;
      VkFormat format;
      uint32_t api_block_B, physical_block_B;
      const bool padded_rgb =
         pCopyBufferToImageInfo->regionCount > 0 &&
         hk_g17p_buffer_upload_plane(
            dst_image,
            pCopyBufferToImageInfo->pRegions[0].imageSubresource.aspectMask,
            &plane, &format) &&
         hk_g17p_padded_rgb_layout(dst_image, plane, format, &api_block_B,
                                   &physical_block_B);
      if (padded_rgb) {
         struct hk_meta_save save;
         hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
         if (!hk_g17p_copy_padded_rgb_buffer_image(
                cmd, dst_image, pCopyBufferToImageInfo->srcBuffer,
                pCopyBufferToImageInfo->regionCount,
                pCopyBufferToImageInfo->pRegions,
                pCopyBufferToImageInfo->dstImageLayout, true))
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
         hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
         return;
      }
      if (hk_g17p_block_copy_format(dev, dst_image->vk.format)) {
         struct hk_meta_save save;
         hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
         if (!hk_g17p_copy_compressed_buffer_to_image(
                cmd, dst_image, pCopyBufferToImageInfo))
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
         hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
         return;
      }
      if (hk_g17p_copy_uncompressed_buffer_to_image_supported(
             dev, src_buffer, dst_image, pCopyBufferToImageInfo)) {
         struct hk_meta_save save;
         hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
         hk_g17p_copy_uncompressed_buffer_to_image(
            cmd, src_buffer, dst_image, pCopyBufferToImageInfo);
         hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
         return;
      }
      /* The raw path takes all regions or none. A depth/stencil upload
       * whose depth plane is compressed mixes a graphics-only region with a
       * raw stencil region, so route each region on its own. */
      if (pCopyBufferToImageInfo->regionCount > 1) {
         for (uint32_t r = 0; r < pCopyBufferToImageInfo->regionCount; ++r) {
            VkCopyBufferToImageInfo2 one = *pCopyBufferToImageInfo;
            one.regionCount = 1;
            one.pRegions = &pCopyBufferToImageInfo->pRegions[r];
            hk_xfer_copybuffertoimage2(commandBuffer, &one);
            if (vk_command_buffer_has_error(&cmd->vk))
               return;
         }
         return;
      }
      if (!hk_g17p_copy_buffer_to_image_supported(
             dev, dst_image, pCopyBufferToImageInfo)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      gfx = true;
   }
   VkPipelineBindPoint bind_point = hk_bind_point(gfx);

   struct hk_meta_save save;
   hk_meta_begin(cmd, &save, bind_point);

   if (gfx) {
      /* M5 reaches this graphics upload only for a destination the raw
       * upload above refuses, such as a compressed depth plane or colour
       * image. A split depth/stencil destination needs the per-plane aspect
       * views here as well: the interleaved R32G32_UINT alias is a view of
       * plane 0 alone. A colour destination is written through the same
       * integer alias image copies and readbacks use: through its own
       * format, a normalized view would canonicalize the bits it copies
       * (SNORM -1.0 has two encodings). */
      const bool m5 = agx_chip_g17_m5(dev->dev.chip);
      const bool m5_split_zs =
         m5 && hk_split_depth_stencil(dst_image->vk.format);
      const bool m5_color =
         m5 && !vk_format_is_depth_or_stencil(dst_image->vk.format);
      struct vk_meta_copy_image_properties dst_props =
         hk_meta_copy_get_image_properties(
            dst_image, dev,
            agx_native_graphics(&dev->dev) || m5_split_zs || m5_color,
            m5_split_zs);

      if (agx_native_graphics(&dev->dev)) {
         /* Native single-slice EOT views already select the mip and array
          * layer or 3D Z slice. Keep source row/image strides intact while
          * splitting multi-layer and volume regions; no staging copy. */
         for (unsigned r = 0; r < pCopyBufferToImageInfo->regionCount; ++r) {
            VkBufferImageCopy2 region = pCopyBufferToImageInfo->pRegions[r];
            const bool volume =
               dst_image->vk.image_type == VK_IMAGE_TYPE_3D;
            const uint32_t slices = volume
               ? region.imageExtent.depth
               : vk_image_subresource_layer_count(
                    &dst_image->vk, &region.imageSubresource);
            const uint64_t rows = region.bufferImageHeight ?
                                     region.bufferImageHeight : region.imageExtent.height;
            const uint64_t columns = region.bufferRowLength ?
                                        region.bufferRowLength : region.imageExtent.width;
            const VkFormat copy_format = aspect_format(
               dst_image->vk.format, region.imageSubresource.aspectMask);
            const uint64_t pixel_size = vk_format_get_blocksize(copy_format);
            if (columns > UINT64_MAX / rows ||
                columns * rows > UINT64_MAX / pixel_size) {
               vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
               break;
            }
            const uint64_t stride = columns * rows * pixel_size;
            const uint64_t offset = region.bufferOffset;
            if (slices > 1 &&
                stride > (UINT64_MAX - offset) / (slices - 1)) {
               vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
               break;
            }
            const uint32_t base_layer = region.imageSubresource.baseArrayLayer;
            const int32_t base_z = region.imageOffset.z;
            region.imageSubresource.layerCount = 1;
            region.imageExtent.depth = 1;
            VkCopyBufferToImageInfo2 info = *pCopyBufferToImageInfo;
            info.regionCount = 1;
            info.pRegions = &region;
            for (uint32_t slice = 0; slice < slices; ++slice) {
               if (volume)
                  region.imageOffset.z = base_z + (int32_t)slice;
               else
                  region.imageSubresource.baseArrayLayer = base_layer + slice;
               region.bufferOffset = offset + slice * stride;
               vk_meta_copy_buffer_to_image(&cmd->vk, &dev->meta, &info,
                                            &dst_props, bind_point);
               if (vk_command_buffer_has_error(&cmd->vk))
                  break;
            }
            if (vk_command_buffer_has_error(&cmd->vk))
               break;
         }
      } else {
         vk_meta_copy_buffer_to_image(&cmd->vk, &dev->meta, pCopyBufferToImageInfo,
                                      &dst_props, bind_point);
      }
   } else {
      hk_meta_copy_buffer_to_image2(&cmd->vk, &dev->meta,
                                    pCopyBufferToImageInfo);
   }

   hk_meta_end(cmd, &save, bind_point);
}

struct hk_r32_volume_copy_push {
   uint64_t src, dst, layer_stride;
   uint32_t depth;
};
static bool
hk_g17p_r32_volume_copy_diagnostic(
   struct hk_cmd_buffer *cmd, struct hk_image *image,
   const VkCopyImageToBufferInfo2 *info)
{
   if (!debug_get_bool_option("ASAHI_G17P_R32_3D_ADDRESS_DIAGNOSTIC", false) ||
       !agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev) ||
       image->vk.image_type != VK_IMAGE_TYPE_3D ||
       image->vk.format != VK_FORMAT_R32_SINT || image->plane_count != 1 ||
       image->vk.extent.width != 1 || image->vk.extent.height != 1 ||
       image->vk.mip_levels != 1 || image->vk.samples != VK_SAMPLE_COUNT_1_BIT ||
       image->planes[0].layout.compressed ||
       image->planes[0].layout.tiling != AIL_TILING_GPU ||
       !hk_g17p_transfer_src_layout_supported(info->srcImageLayout))
      return false;

   VK_FROM_HANDLE(hk_buffer, buffer, info->dstBuffer);
   for (unsigned r = 0; r < info->regionCount; r++) {
      const VkBufferImageCopy2 *region = &info->pRegions[r];
      if (region->imageSubresource.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT ||
          region->imageSubresource.mipLevel ||
          region->imageSubresource.baseArrayLayer ||
          region->imageSubresource.layerCount != 1 || region->imageOffset.x ||
          region->imageOffset.y || region->imageOffset.z < 0 ||
          region->imageExtent.width != 1 || region->imageExtent.height != 1 ||
          !region->imageExtent.depth ||
          region->imageOffset.z >= image->vk.extent.depth ||
          region->imageExtent.depth > image->vk.extent.depth - region->imageOffset.z ||
          (region->bufferRowLength && region->bufferRowLength != 1) ||
          (region->bufferImageHeight && region->bufferImageHeight != 1) ||
          (region->bufferOffset & 3) || region->bufferOffset > buffer->vk.size ||
          (uint64_t)region->imageExtent.depth * 4 > buffer->vk.size - region->bufferOffset)
         return false;
   }

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct { enum hk_meta_object_key_type type; uint32_t diagnostic; } key = {
      HK_META_OBJECT_KEY_BLOCK_COPY, 0x52335644,
   };
   const VkPushConstantRange range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(struct hk_r32_volume_copy_push),
   };
   VkPipelineLayout layout;
   VkResult result = vk_meta_get_pipeline_layout(&dev->vk, &dev->meta, NULL,
                                                &range, &key, sizeof(key), &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return true;
   }
   VkPipeline pipeline = vk_meta_lookup_pipeline(&dev->meta, &key, sizeof(key));
   if (pipeline == VK_NULL_HANDLE) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL,
                                                      "hk-r32-volume-address-diagnostic");
      b.shader->info.workgroup_size[0] = 64;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
      nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
#define r32_push(field) nir_load_push_constant(&b, 1, \
   sizeof(((struct hk_r32_volume_copy_push *)0)->field) * 8, \
   nir_imm_int(&b, offsetof(struct hk_r32_volume_copy_push, field)))
      nir_push_if(&b, nir_ult(&b, id, r32_push(depth)));
      nir_def *src = nir_iadd(&b, r32_push(src),
                              nir_imul(&b, nir_u2u64(&b, id), r32_push(layer_stride)));
      nir_def *dst = nir_iadd(&b, r32_push(dst),
                              nir_ishl_imm(&b, nir_u2u64(&b, id), 2));
      nir_store_global(&b, nir_load_global(&b, 1, 32, src, .align_mul = 4),
                       dst, .align_mul = 4);
      nir_pop_if(&b, NULL);
#undef r32_push
      const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
         .nir = b.shader,
      };
      const VkComputePipelineCreateInfo pci = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                   .pNext = &nir_info, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .pName = "main"},
         .layout = layout,
      };
      result = vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &pci,
                                               &key, sizeof(key), &pipeline);
      ralloc_free(b.shader);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, result);
         return true;
      }
   }
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkCommandBuffer cb = vk_command_buffer_to_handle(&cmd->vk);
   disp->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   const struct ail_layout *ail = &image->planes[0].layout;
   for (unsigned r = 0; r < info->regionCount; r++) {
      const VkBufferImageCopy2 *region = &info->pRegions[r];
      const struct hk_r32_volume_copy_push push = {
         .src = hk_image_base_address(image, 0) +
                ail_get_layer_level_B(ail, region->imageOffset.z, 0),
         .dst = hk_buffer_address_rw(buffer, region->bufferOffset),
         .layer_stride = ail->layer_stride_B,
         .depth = region->imageExtent.depth,
      };
      disp->CmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             sizeof(push), &push);
      disp->CmdDispatch(cb, DIV_ROUND_UP(push.depth, 64), 1, 1);
      if (cmd->current_cs.cs)
         hk_drm_attachment_set_add(&cmd->current_cs.cs->compute_attachments,
                                   push.dst, (uint64_t)push.depth * 4);
   }
   return true;
}

static void
hk_xfer_copyimagetobuffer2(VkCommandBuffer commandBuffer,
                         const VkCopyImageToBufferInfo2 *pCopyImageToBufferInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_image, src_image, pCopyImageToBufferInfo->srcImage);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   struct hk_meta_save save;
   hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
   uint8_t plane;
   VkFormat format;
   uint32_t api_block_B, physical_block_B;
   const bool padded_rgb =
      (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
      pCopyImageToBufferInfo->regionCount > 0 &&
      hk_g17p_buffer_upload_plane(
         src_image,
         pCopyImageToBufferInfo->pRegions[0].imageSubresource.aspectMask,
         &plane, &format) &&
      hk_g17p_padded_rgb_layout(src_image, plane, format, &api_block_B,
                                &physical_block_B);
   if (hk_g17p_r32_volume_copy_diagnostic(cmd, src_image, pCopyImageToBufferInfo)) {
   } else if (padded_rgb) {
      if (!hk_g17p_copy_padded_rgb_buffer_image(
             cmd, src_image, pCopyImageToBufferInfo->dstBuffer,
             pCopyImageToBufferInfo->regionCount,
             pCopyImageToBufferInfo->pRegions,
             pCopyImageToBufferInfo->srcImageLayout, false))
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   } else if (hk_g17p_block_copy_format(dev, src_image->vk.format)) {
      /* A compressed Vulkan copy preserves the encoded block bytes.  The
       * common meta path samples and decodes the source, so use the symmetric
       * raw AIL-addressed path on G17P instead. */
      if (!hk_g17p_copy_compressed_image_to_buffer(
             cmd, src_image, pCopyImageToBufferInfo))
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   } else {
      hk_meta_copy_image_to_buffer2(&cmd->vk, &dev->meta,
                                    pCopyImageToBufferInfo);
   }
   hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
}

static void
hk_xfer_copyimage2(VkCommandBuffer commandBuffer,
                 const VkCopyImageInfo2 *pCopyImageInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_image, src_image, pCopyImageInfo->srcImage);
   VK_FROM_HANDLE(hk_image, dst_image, pCopyImageInfo->dstImage);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   /* VkCmdCopyImage preserves the source bits; compatible source and
    * destination formats do not imply a format conversion.  The element
    * planner checks their block sizes, so prefer the raw tiled copier for
    * every eligible G17P single-sample colour copy, not only block-compressed
    * formats. */
   if (hk_g17p_raw_image_copy_supported(
          dev, src_image, dst_image, pCopyImageInfo)) {
      struct hk_meta_save save;
      hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      hk_g17p_copy_raw_image(cmd, src_image, dst_image, pCopyImageInfo);
      hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      return;
   }
   /* G17P copies through its native tile renderer, as buffer uploads do.
    * The common meta path preserves partial rectangles, layers and format
    * reinterpretation without the AGX2 compute imageblock protocol. */
   bool gfx = agx_native_graphics(&dev->dev) ||
              agx_chip_g17_m5(dev->dev.chip) ||
              hk_copy_requires_gfx(dst_image);
   VkPipelineBindPoint bind_point = hk_bind_point(gfx);

   struct hk_meta_save save;
   hk_meta_begin(cmd, &save, bind_point);

   if (gfx) {
      const bool native = agx_native_graphics(&dev->dev) ||
                          agx_chip_g17_m5(dev->dev.chip);
      /* Both sides of a depth/stencil copy use the same views: per-plane
       * aliases when either depth plane is compressed. */
      const bool zs_planes =
         agx_chip_g17_m5(dev->dev.chip) ||
         (hk_split_depth_stencil(src_image->vk.format) &&
          (src_image->planes[0].layout.compressed ||
           dst_image->planes[0].layout.compressed));
      struct vk_meta_copy_image_properties src_props =
         hk_meta_copy_get_image_properties(src_image, dev, native, zs_planes);
      struct vk_meta_copy_image_properties dst_props =
         hk_meta_copy_get_image_properties(dst_image, dev, native, zs_planes);

      if (agx_native_graphics(&dev->dev)) {
         /* The qualified HAL200 BG/EOT contract is single-layer.  Preserve
          * arbitrary mip and rectangle selection while expanding array and
          * 3D regions into per-slice native meta draws. */
         for (uint32_t r = 0; r < pCopyImageInfo->regionCount; ++r) {
            const VkImageCopy2 *region = &pCopyImageInfo->pRegions[r];
            const uint32_t src_slices =
               src_image->vk.image_type == VK_IMAGE_TYPE_3D
                  ? region->extent.depth
                  : vk_image_subresource_layer_count(
                       &src_image->vk, &region->srcSubresource);
            const uint32_t dst_slices =
               dst_image->vk.image_type == VK_IMAGE_TYPE_3D
                  ? region->extent.depth
                  : vk_image_subresource_layer_count(
                       &dst_image->vk, &region->dstSubresource);

            for (uint32_t slice = 0; slice < src_slices; ++slice) {
               VkImageCopy2 one;
               if (!hk_g17p_image_copy_slice(
                      dev->dev.chip, src_image->vk.image_type,
                      dst_image->vk.image_type, src_slices, dst_slices,
                      region, slice, &one)) {
                  vk_command_buffer_set_error(&cmd->vk,
                                              VK_ERROR_FEATURE_NOT_PRESENT);
                  break;
               }

               VkCopyImageInfo2 info = *pCopyImageInfo;
               info.regionCount = 1;
               info.pRegions = &one;
               vk_meta_copy_image(&cmd->vk, &dev->meta, &info, &src_props,
                                  &dst_props, bind_point);
               if (vk_command_buffer_has_error(&cmd->vk))
                  break;
            }

            if (vk_command_buffer_has_error(&cmd->vk))
               break;
         }
      } else {
         vk_meta_copy_image(&cmd->vk, &dev->meta, pCopyImageInfo, &src_props,
                            &dst_props, bind_point);
      }
   } else {
      hk_meta_copy_image2(&cmd->vk, &dev->meta, pCopyImageInfo);
   }

   hk_meta_end(cmd, &save, bind_point);
}

static void
hk_g17p_fill_buffer_pbe(struct hk_cmd_buffer *cmd, VkBuffer buffer,
                        VkDeviceSize offset, VkDeviceSize size, uint32_t data,
                        const struct hk_native_pbe_descriptor *expected)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const uint32_t key = HK_META_OBJECT_KEY_G17P_BUFFER_FILL_PBE;
   const VkDescriptorSetLayoutBinding binding = {
      .binding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
      .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   const VkDescriptorSetLayoutCreateInfo desc_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   const struct {
      uint32_t elements;
      uint32_t data;
   } push = {size / 16, data};
   const VkPushConstantRange push_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .size = sizeof(push),
   };
   VkPipelineLayout layout;
   VkResult result = vk_meta_get_pipeline_layout(
      &dev->vk, &dev->meta, &desc_info, &push_range,
      &key, sizeof(key), &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   VkPipeline pipeline = vk_meta_lookup_pipeline(&dev->meta, &key, sizeof(key));
   if (pipeline == VK_NULL_HANDLE) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, NULL, "hk-meta-fill-buffer-pbe");
      b.shader->info.workgroup_size[0] = 64;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
      nir_variable *image = nir_variable_create(
         b.shader, nir_var_image,
         glsl_image_type(GLSL_SAMPLER_DIM_BUF, false, GLSL_TYPE_UINT), "dest");
      image->data.descriptor_set = 0;
      image->data.binding = 0;
      image->data.image.format = PIPE_FORMAT_R32G32B32A32_UINT;
      image->data.access = ACCESS_NON_READABLE;
      nir_def *index = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *elements = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0));
      nir_def *value = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4));
      nir_push_if(&b, nir_ult(&b, index, elements));
      nir_image_deref_store(
         &b, &nir_build_deref_var(&b, image)->def, nir_pad_vec4(&b, index),
         nir_imm_int(&b, 0), nir_replicate(&b, value, 4), nir_imm_int(&b, 0),
         .image_dim = GLSL_SAMPLER_DIM_BUF, .image_array = false,
         .format = PIPE_FORMAT_R32G32B32A32_UINT, .src_type = nir_type_uint32);
      nir_pop_if(&b, NULL);

      const VkPipelineShaderStageNirCreateInfoMESA nir_info = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_NIR_CREATE_INFO_MESA,
         .nir = b.shader,
      };
      const VkComputePipelineCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &nir_info,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = "main",
         },
         .layout = layout,
      };
      result = vk_meta_create_compute_pipeline(
         &dev->vk, &dev->meta, &info, &key, sizeof(key), &pipeline);
      ralloc_free(b.shader);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, result);
         return;
      }
   }

   /* The view is owned by this recording until its normal reset/destruction.
    * Its PBE addresses the original buffer; no image or alias is allocated. */
   const VkBufferViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
      .buffer = buffer,
      .format = VK_FORMAT_R32G32B32A32_UINT,
      .offset = offset,
      .range = size,
   };
   VkBufferView view;
   result = vk_meta_create_buffer_view(&cmd->vk, &dev->meta, &view_info, &view);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   /* CreateBufferView can represent a refused format with a null PBE.
    * Require the exact descriptor accepted by our existing codec preflight. */
   VK_FROM_HANDLE(hk_buffer_view, native_view, view);
   if (memcmp(&native_view->pbe, expected, sizeof(*expected)) != 0) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FORMAT_NOT_SUPPORTED);
      return;
   }

   const VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstBinding = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
      .pTexelBufferView = &view,
   };
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkCommandBuffer cb = vk_command_buffer_to_handle(&cmd->vk);
   disp->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   disp->CmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                                layout, 0, 1, &write);
   disp->CmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(push), &push);
   disp->CmdDispatch(cb, DIV_ROUND_UP(push.elements, 64), 1, 1);
}

static bool
hk_g17p_fill_buffer_pbe_supported(const struct hk_device *dev,
                                  const struct hk_buffer *buffer,
                                  uint64_t address, uint64_t offset,
                                  uint64_t size,
                                  struct hk_native_pbe_descriptor *pbe)
{
   const struct hk_physical_device *pdev =
      (const struct hk_physical_device *)dev->vk.physical;
   if (dev->dev.chip != AGX_CHIP_G17P || buffer->va ||
       size < (64u << 20) || ((address | offset | size) & 15) ||
       !address || size > UINT64_MAX - address ||
       size / 16 > pdev->vk.properties.maxTexelBufferElements)
      return false;

   /* Match CreateBufferView's ordinary RGBA32_UINT linear PBE. Let the
    * existing codec prove address, stride and dimension representability. */
   struct AGX_PBE semantic = {
      AGX_PBE_header,
      .dimension = AGX_TEXTURE_DIMENSION_2D,
      .layout = AGX_LAYOUT_LINEAR,
      .channels = AGX_CHANNELS_R32G32B32A32,
      .type = AGX_TEXTURE_TYPE_UINT,
      .swizzle_r = 0, .swizzle_g = 1, .swizzle_b = 2, .swizzle_a = 3,
      .buffer = address,
      .width = AGX_TEXTURE_BUFFER_WIDTH,
      .height = DIV_ROUND_UP(size / 16, AGX_TEXTURE_BUFFER_WIDTH),
      .stride = AGX_TEXTURE_BUFFER_WIDTH * 16 - 4,
      .layers = 1,
      .levels = 1,
   };
   const unsigned generation = agx_chip_descriptor_generation(dev->dev.chip);
   if (!hk_pack_native_pbe(generation, pbe, &semantic))
      return false;

   /* The last stored texel must stay in the codec's supported address domain
    * too; size and address were already checked for alignment and overflow. */
   semantic.buffer = address + size - 16;
   struct hk_native_pbe_descriptor last;
   return hk_pack_native_pbe(generation, &last, &semantic);
}

static void
hk_xfer_fillbuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
                 VkDeviceSize dstOffset, VkDeviceSize dstRange, uint32_t data)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, dstBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (agx_native_graphics(&dev->dev)) {
      /* VK_WHOLE_SIZE fills only complete 32-bit words. */
      VkDeviceSize size =
         vk_buffer_range(&buffer->vk, dstOffset, dstRange) & ~(VkDeviceSize)3;
      uint64_t address =
         vk_meta_buffer_address(&dev->vk, dstBuffer, dstOffset, dstRange);
      struct hk_meta_save save;
      hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      struct hk_native_pbe_descriptor pbe;
      if (hk_g17p_fill_buffer_pbe_supported(dev, buffer, address, dstOffset,
                                           size, &pbe))
         hk_g17p_fill_buffer_pbe(cmd, dstBuffer, dstOffset, size, data, &pbe);
      else
         vk_meta_fill_buffer_addr(&cmd->vk, &dev->meta, address, size, data);
      hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      if (size && cmd->current_cs.cs) {
         hk_drm_attachment_set_add(&cmd->current_cs.cs->compute_attachments,
                                   address, size);
      }
      return;
   }

   size_t range = vk_buffer_range(&buffer->vk, dstOffset, dstRange);
   uint64_t addr =
      vk_meta_buffer_address(&dev->vk, dstBuffer, dstOffset, dstRange);

   if (util_is_aligned(addr, 16) && util_is_aligned(range, 16)) {
      libagx_fill_uint4(cmd, agx_2d(range / 16, 1), AGX_BARRIER_ALL,
                        addr, 0, data, data, data, data);
   } else {
      libagx_fill(cmd, agx_1d(range / 4), AGX_BARRIER_ALL, addr, data);
   }
}

static void
hk_xfer_updatebuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
                   VkDeviceSize dstOffset, VkDeviceSize dstRange,
                   const void *pData)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, dstBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (agx_native_graphics(&dev->dev)) {
      struct hk_meta_save save;
      hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      vk_meta_update_buffer(&cmd->vk, &dev->meta, dstBuffer, dstOffset, dstRange,
                            pData);
      hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
      return;
   }

   size_t range = vk_buffer_range(&buffer->vk, dstOffset, dstRange);

   hk_cmd_copy(cmd,
               vk_meta_buffer_address(&dev->vk, dstBuffer, dstOffset, dstRange),
               hk_pool_upload(cmd, pData, range, 4), range);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdClearAttachments(VkCommandBuffer commandBuffer, uint32_t attachmentCount,
                       const VkClearAttachment *pAttachments,
                       uint32_t rectCount, const VkClearRect *pRects)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   struct vk_meta_rendering_info render_info;
   hk_meta_init_render(cmd, &render_info);

   for (uint32_t i = 0; i < attachmentCount; ++i) {
      if (pAttachments[i].aspectMask & (VK_IMAGE_ASPECT_DEPTH_BIT |
                                        VK_IMAGE_ASPECT_STENCIL_BIT))
         hk_g17p_note_depth_clear(cmd, pAttachments[i].aspectMask);
      if (pAttachments[i].aspectMask & VK_IMAGE_ASPECT_COLOR_BIT)
         cmd->state.gfx.render.color_written |=
            BITFIELD_BIT(pAttachments[i].colorAttachment);
   }

   struct hk_meta_save save;
   hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);

   /* Unlike transfer/meta operations, CmdClearAttachments is explicitly a
    * rendering command and remains inside an active conditional scope. */
   assert(!cmd->conditional_meta);
   cmd->conditional_meta = true;
   vk_meta_clear_attachments(&cmd->vk, &dev->meta, &render_info,
                             attachmentCount, pAttachments, rectCount, pRects);
   cmd->conditional_meta = false;
   hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
}

/* Transfer entry points name their destination for the G17P transfer write
 * tracking (hk_sync_xfer_write in hk_sync.h). */
static void
hk_xfer_begin_image(struct hk_cmd_buffer *cmd, VkImage _image)
{
   VK_FROM_HANDLE(hk_image, image, _image);
   uint64_t lo, hi;
   hk_image_va_range(image, &lo, &hi);
   hk_sync_xfer_begin_dst(cmd, lo, hi, false);
}

static void
hk_xfer_begin_buffer(struct hk_cmd_buffer *cmd, VkBuffer _buffer,
                     uint64_t lo_B, uint64_t hi_B)
{
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   const uint64_t base = vk_buffer_address(&buffer->vk, 0);
   /* A sparse buffer is also read through its shadow mapping. */
   const bool device_address =
      (buffer->vk.usage & VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT) ||
      buffer->va;
   hk_sync_xfer_begin_dst(cmd, base + lo_B, base + MIN2(hi_B, buffer->vk.size),
                          device_address);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBlitImage2(VkCommandBuffer commandBuffer,
                 const VkBlitImageInfo2 *pBlitImageInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* Transfer stage: the streams recorded here belong to the transfer
    * class for every recorded dependency. */
   hk_xfer_begin_image(cmd, pBlitImageInfo->dstImage);
   hk_xfer_blitimage2(commandBuffer, pBlitImageInfo);
   hk_sync_xfer_end(cmd);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdResolveImage2(VkCommandBuffer commandBuffer,
                    const VkResolveImageInfo2 *pResolveImageInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* Transfer stage: the streams recorded here belong to the transfer
    * class for every recorded dependency. */
   hk_xfer_begin_image(cmd, pResolveImageInfo->dstImage);
   hk_xfer_resolveimage2(commandBuffer, pResolveImageInfo);
   hk_sync_xfer_end(cmd);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyBuffer2(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 *info)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (cmd->state.deferred_clear_count) {
      VK_FROM_HANDLE(hk_buffer, src, info->srcBuffer);
      VK_FROM_HANDLE(hk_buffer, dst, info->dstBuffer);
      for (uint32_t r = 0; r < info->regionCount; ++r) {
         const VkBufferCopy2 *region = &info->pRegions[r];
         hk_deferred_clears_hold_buffer(
            cmd, hk_buffer_address_ro(src, region->srcOffset), region->size);
         hk_deferred_clears_hold_buffer(
            cmd, hk_buffer_address_rw(dst, region->dstOffset), region->size);
      }
   }

   /* Transfer stage: the streams recorded here belong to the transfer
    * class for every recorded dependency. */
   uint64_t lo = UINT64_MAX, hi = 0;
   for (uint32_t r = 0; r < info->regionCount; ++r) {
      lo = MIN2(lo, info->pRegions[r].dstOffset);
      hi = MAX2(hi, info->pRegions[r].dstOffset + info->pRegions[r].size);
   }
   hk_xfer_begin_buffer(cmd, info->dstBuffer, lo, hi);
   hk_xfer_copybuffer2(commandBuffer, info);
   hk_sync_xfer_end(cmd);
   cmd->state.deferred_clear_hold = false;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyBufferToImage2(VkCommandBuffer commandBuffer,
                         const VkCopyBufferToImageInfo2 *pCopyBufferToImageInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* Transfer stage: the streams recorded here belong to the transfer
    * class for every recorded dependency. */
   hk_xfer_begin_image(cmd, pCopyBufferToImageInfo->dstImage);
   hk_xfer_copybuffertoimage2(commandBuffer, pCopyBufferToImageInfo);
   hk_sync_xfer_end(cmd);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyImageToBuffer2(VkCommandBuffer commandBuffer,
                         const VkCopyImageToBufferInfo2 *pCopyImageToBufferInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* Transfer stage: the streams recorded here belong to the transfer
    * class for every recorded dependency. */
   hk_xfer_begin_buffer(cmd, pCopyImageToBufferInfo->dstBuffer, 0, UINT64_MAX);
   hk_xfer_copyimagetobuffer2(commandBuffer, pCopyImageToBufferInfo);
   hk_sync_xfer_end(cmd);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyImage2(VkCommandBuffer commandBuffer,
                 const VkCopyImageInfo2 *pCopyImageInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* Transfer stage: the streams recorded here belong to the transfer
    * class for every recorded dependency. */
   hk_xfer_begin_image(cmd, pCopyImageInfo->dstImage);
   hk_xfer_copyimage2(commandBuffer, pCopyImageInfo);
   hk_sync_xfer_end(cmd);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
                 VkDeviceSize dstOffset, VkDeviceSize dstRange, uint32_t data)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (cmd->state.deferred_clear_count) {
      VK_FROM_HANDLE(hk_buffer, buffer, dstBuffer);
      hk_deferred_clears_hold_buffer(
         cmd, hk_buffer_address_rw(buffer, dstOffset),
         vk_buffer_range(&buffer->vk, dstOffset, dstRange));
   }

   /* Transfer stage: the streams recorded here belong to the transfer
    * class for every recorded dependency. */
   {
      VK_FROM_HANDLE(hk_buffer, buffer, dstBuffer);
      hk_xfer_begin_buffer(cmd, dstBuffer, dstOffset,
                           dstOffset +
                              vk_buffer_range(&buffer->vk, dstOffset, dstRange));
   }
   hk_xfer_fillbuffer(commandBuffer, dstBuffer, dstOffset, dstRange, data);
   hk_sync_xfer_end(cmd);
   cmd->state.deferred_clear_hold = false;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer,
                   VkDeviceSize dstOffset, VkDeviceSize dstRange,
                   const void *pData)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (cmd->state.deferred_clear_count) {
      VK_FROM_HANDLE(hk_buffer, buffer, dstBuffer);
      hk_deferred_clears_hold_buffer(
         cmd, hk_buffer_address_rw(buffer, dstOffset), dstRange);
   }

   /* Transfer stage: the streams recorded here belong to the transfer
    * class for every recorded dependency. */
   hk_xfer_begin_buffer(cmd, dstBuffer, dstOffset, dstOffset + dstRange);
   hk_xfer_updatebuffer(commandBuffer, dstBuffer, dstOffset, dstRange, pData);
   hk_sync_xfer_end(cmd);
   cmd->state.deferred_clear_hold = false;
}

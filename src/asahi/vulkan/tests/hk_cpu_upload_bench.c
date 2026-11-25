/* SPDX-License-Identifier: MIT
 * Host-only command-recording microbenchmark. No Vulkan device is created and
 * nothing is submitted. The real descriptor-bind and upload paths run against
 * preallocated command-buffer storage. Run with ASAHI_MESA_DEBUG=perf.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "vk_pipeline_layout.h"

#define CHECK(c) do { if (!(c)) { \
   fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #c); \
   exit(EXIT_FAILURE); } } while (0)

struct fixture {
   struct hk_cmd_buffer cmd;
   struct hk_device dev;
   struct vk_pipeline_layout layout;
   struct hk_descriptor_set_layout set_layout;
   /* hk_descriptor_set ends in a dynamic-buffer array. */
   struct hk_descriptor_set *set;
   struct hk_shader vs, fs;
   struct hk_api_shader *vs_api, *fs_api;
   uint8_t constants[128];
   void *main_map, *root_map;
};

static void
reset(struct fixture *f)
{
   memset(&f->cmd.state, 0, sizeof(f->cmd.state));
   if (f->cmd.used_group_count_cache) {
      for (unsigned i = 0; i < ARRAY_SIZE(f->cmd.group_count_cache); ++i)
         f->cmd.group_count_cache[i].valid = 0;
      f->cmd.used_group_count_cache = false;
   }
   f->cmd.uploader.main.offset = f->cmd.uploader.root.offset = 0;
   f->cmd.state.gfx.shaders[MESA_SHADER_VERTEX] = f->vs_api;
   f->cmd.state.gfx.shaders[MESA_SHADER_FRAGMENT] = f->fs_api;
}

static struct fixture *
fixture(void)
{
   struct fixture *f = calloc(1, sizeof(*f));
   CHECK(f);
   f->main_map = calloc(1, HK_CMD_BO_SIZE);
   f->root_map = calloc(1, HK_CMD_BO_SIZE);
   f->set = calloc(1, sizeof(*f->set) + sizeof(f->set->dynamic_buffers[0]));
   CHECK(f->main_map && f->root_map && f->set);
   f->dev.dev.chip = AGX_CHIP_G17P;
   f->dev.dev.debug = AGX_DBG_PERF;
   f->cmd.vk.base.device = &f->dev.vk;
   f->cmd.vk.base.type = VK_OBJECT_TYPE_COMMAND_BUFFER;
   list_inithead(&f->cmd.uploader.main.bos);
   f->cmd.uploader.main.map = f->main_map;
   f->cmd.uploader.main.base = 0x100000;
   f->cmd.uploader.main.capacity = HK_CMD_BO_SIZE;
   f->cmd.uploader.main.cpu_cached = true;
   list_inithead(&f->cmd.uploader.root.bos);
   f->cmd.uploader.root.map = f->root_map;
   f->cmd.uploader.root.base = 0x200000;
   f->cmd.uploader.root.capacity = HK_CMD_BO_SIZE;
   f->cmd.uploader.root.cpu_cached = true;
   f->layout.base.type = VK_OBJECT_TYPE_PIPELINE_LAYOUT;
   f->set_layout.vk.base.type = VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT;
   f->set_layout.vk.dynamic_descriptor_count = 1;
   f->layout.set_layouts[0] = &f->set_layout.vk;
   f->set->base.type = VK_OBJECT_TYPE_DESCRIPTOR_SET;
   f->set->layout = &f->set_layout;
   f->set->addr = 0x300000;
   f->set->size = 64;
   f->set->dynamic_buffers[0].base_addr = 0x400000;
   f->set->dynamic_buffers[0].size = 4096;
   f->vs.data_ptr = f->fs.data_ptr = f->constants;
   f->vs.data_size = f->fs.data_size = sizeof(f->constants);
   f->vs.constant_data_id = 1;
   f->fs.constant_data_id = 2;
   f->vs_api = calloc(1, hk_api_shader_size(MESA_SHADER_VERTEX, false));
   f->fs_api = calloc(1, hk_api_shader_size(MESA_SHADER_FRAGMENT, false));
   CHECK(f->vs_api && f->fs_api);
   f->vs_api->vk.stage = MESA_SHADER_VERTEX;
   f->fs_api->vk.stage = MESA_SHADER_FRAGMENT;
   f->vs_api->variants[HK_VS_VARIANT_HW].compiled = true;
   f->fs_api->variants[0].compiled = true;
   reset(f);
   CHECK(16 * sizeof(struct hk_root_descriptor_table) <= HK_CMD_BO_SIZE);
   return f;
}

static void
bind(struct fixture *f, uint32_t offset, VkShaderStageFlags stages)
{
   VkDescriptorSet set = hk_descriptor_set_to_handle(f->set);
   VkBindDescriptorSetsInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO_KHR,
      .stageFlags = stages,
      .layout = vk_pipeline_layout_to_handle(&f->layout),
      .descriptorSetCount = 1,
      .pDescriptorSets = &set,
      .dynamicOffsetCount = 1,
      .pDynamicOffsets = &offset,
   };
   hk_CmdBindDescriptorSets2KHR(hk_cmd_buffer_to_handle(&f->cmd), &info);
}

static void push_descriptor_and_msaa_correctness(struct fixture *f);

static void
correctness(struct fixture *f)
{
   struct hk_cmd_buffer *cmd = &f->cmd;
   bind(f, 16, VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT);
   CHECK(cmd->state.gfx.descriptors.root.dynamic_buffers[0].base_addr == 0x400010);
   CHECK(cmd->state.cs.descriptors.root.dynamic_buffers[0].base_addr == 0x400010);
   uint64_t root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   CHECK(root);
   uint8_t old[sizeof(struct hk_root_descriptor_table)];
   memcpy(old, f->root_map, sizeof(old));
   bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
   CHECK(cmd->state.gfx.descriptors.root.dynamic_buffers[0].base_addr == 0x400020);
   CHECK(cmd->state.cs.descriptors.root.dynamic_buffers[0].base_addr == 0x400010);
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS) != root);
   CHECK(memcmp(old, f->root_map, sizeof(old)) == 0);
   /* Switching from push descriptors back to the same set is a real bind. */
   cmd->state.gfx.descriptors.push_active = 1;
   cmd->state.gfx.descriptors.root.sets[0] = 0x500000;
   bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
   CHECK(!cmd->state.gfx.descriptors.push_active);
   CHECK(cmd->state.gfx.descriptors.root.sets[0] == f->set->addr);
   /* An identical set with changed dynamic-layout indexing is a real change. */
   f->layout.dynamic_descriptor_offset[0] = 1;
   bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
   CHECK(cmd->state.gfx.descriptors.root.set_dynamic_buffer_start[0] == 1);
   CHECK(cmd->state.gfx.descriptors.root.dynamic_buffers[1].base_addr == 0x400020);
   f->layout.dynamic_descriptor_offset[0] = 0;
   hk_cmd_upload_draw_params(cmd, (uint32_t)-7, 3);
   uint64_t params = cmd->state.gfx.draw_params;
   const uint32_t *data = (uint32_t *)((uint8_t *)f->main_map + params - 0x100000);
   CHECK(data[0] == (uint32_t)-7 && data[1] == 3);
   /* Indirect parameters are read at execution time; never compare GPU data. */
   cmd->state.gfx.draw_params = 0x600000;
   cmd->state.gfx.g17p_draw_params_host_valid = false;
   hk_cmd_upload_draw_params(cmd, (uint32_t)-7, 3);
   CHECK(cmd->state.gfx.draw_params != 0x600000);
   hk_cmd_upload_draw_params(cmd, 9, 4);
   CHECK(data[0] == (uint32_t)-7 && data[1] == 3);
   hk_cmd_upload_draw_id(cmd, 65535);
   const uint32_t *id = (uint32_t *)((uint8_t *)f->main_map +
                                    cmd->state.gfx.draw_id_ptr - 0x100000);
   CHECK(*id == 65535);
   hk_cmd_upload_draw_id(cmd, 0);
   CHECK(*id == 65535);
   CHECK(cmd->vk.record_result == VK_SUCCESS);
   reset(f);
   push_descriptor_and_msaa_correctness(f);
}

static void
push_descriptor_and_msaa_correctness(struct fixture *f)
{
   struct hk_cmd_buffer *cmd = &f->cmd;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   /* Sample-variant fragment shaders retain descriptor roots when the bound
    * sample variant is published (ready) and neither it nor any ready sibling
    * uses CPU-decoded combined samplers. */
   {
      struct hk_api_shader *msaa_fs =
         calloc(1, hk_api_shader_size(MESA_SHADER_FRAGMENT, 3));
      CHECK(msaa_fs);
      msaa_fs->vk.stage = MESA_SHADER_FRAGMENT;
      msaa_fs->g17p_sample_variants = 3;
      msaa_fs->variants[2].compiled = true; /* 4x MSAA variant only */
      CHECK(hk_fs_sample_variant_ready(msaa_fs, 2));
      CHECK(!hk_fs_sample_variant_ready(msaa_fs, 1));
      CHECK(!hk_fs_sample_variant_ready(msaa_fs, 3));
      cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = msaa_fs;
      cmd->vk.dynamic_graphics_state.ms.rasterization_samples = 4;

      bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
      uint64_t gen = desc->generation;
      hk_root_mark_clean(desc);
      desc->sets_dirty = 0;
      bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
      CHECK(desc->generation == gen && !hk_root_dirty(desc) && !desc->sets_dirty);

      /* Bound 4x variant using CPU-decoded combined samplers must not retain. */
      msaa_fs->variants[2].info.agx3_combined_sampler_valid = true;
      hk_root_mark_clean(desc);
      bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
      CHECK(desc->generation == gen + 1 && hk_root_dirty(desc));
      msaa_fs->variants[2].info.agx3_combined_sampler_valid = false;

      /* Published sibling 1x variant with combined samplers stays conservative. */
      msaa_fs->variants[0].compiled = true;
      msaa_fs->variants[0].info.agx3_combined_sampler_valid = true;
      gen = desc->generation;
      hk_root_mark_clean(desc);
      bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
      CHECK(desc->generation == gen + 1 && hk_root_dirty(desc));
      msaa_fs->variants[0].info.agx3_combined_sampler_valid = false;

      /* Uncompiled bound variant (2x) must not retain. */
      cmd->vk.dynamic_graphics_state.ms.rasterization_samples = 2;
      gen = desc->generation;
      hk_root_mark_clean(desc);
      bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
      CHECK(desc->generation == gen + 1 && hk_root_dirty(desc));

      /* Out-of-range sample count (8x on 3-variant shader) fails closed. */
      cmd->vk.dynamic_graphics_state.ms.rasterization_samples = 8;
      gen = desc->generation;
      hk_root_mark_clean(desc);
      bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
      CHECK(desc->generation == gen + 1 && hk_root_dirty(desc));

      cmd->vk.dynamic_graphics_state.ms.rasterization_samples = 1;
      cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = f->fs_api;
      free(msaa_fs);
   }

   /* Push-descriptor elision, WC guard, invalidation paths, and transfer sync. */
   {
      struct hk_descriptor_set_layout *push_layout =
         calloc(1, sizeof(*push_layout) + sizeof(push_layout->binding[0]));
      struct hk_push_descriptor_set *push = calloc(1, sizeof(*push));
      struct hk_push_descriptor_set *cs_push = calloc(1, sizeof(*cs_push));
      CHECK(push_layout && push && cs_push);

      push_layout->vk.base.type = VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT;
      push_layout->vk.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
      push_layout->descriptor_generation = 9;
      push_layout->non_variable_descriptor_buffer_size = 64;
      push_layout->max_buffer_size = 64;
      push_layout->binding_count = 1;
      push_layout->binding[0] = (struct hk_descriptor_set_binding_layout){
         .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .array_size = 1,
         .offset = 0,
         .stride = sizeof(struct hk_buffer_address),
      };

      struct vk_pipeline_layout push_pl = {0};
      push_pl.base.type = VK_OBJECT_TYPE_PIPELINE_LAYOUT;
      push_pl.set_layouts[0] = &push_layout->vk;

      struct hk_buffer buf = {0};
      buf.vk.base.type = VK_OBJECT_TYPE_BUFFER;
      buf.vk.base.device = &f->dev.vk;
      buf.vk.device_address = 0x800000;
      buf.vk.size = 4096;

      desc->push[0] = push;
      VkDescriptorBufferInfo bi = {
         .buffer = hk_buffer_to_handle(&buf),
         .offset = 0,
         .range = 64,
      };
      VkWriteDescriptorSet write = {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         .pBufferInfo = &bi,
      };
      VkPushDescriptorSetInfoKHR push_info = {
         .sType = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_INFO_KHR,
         .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
         .layout = vk_pipeline_layout_to_handle(&push_pl),
         .set = 0,
         .descriptorWriteCount = 1,
         .pDescriptorWrites = &write,
      };

      /* Initial push + flush uploads, dirties root, and advances generation. */
      hk_root_mark_clean(desc);
      uint64_t gen = desc->generation;
      const uint32_t off0 = cmd->uploader.main.offset;
      hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
      CHECK(desc->push_dirty == BITFIELD_BIT(0) && desc->generation == gen);
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      const uint64_t addr1 = desc->root.sets[0];
      const uint32_t off1 = cmd->uploader.main.offset;
      CHECK(addr1 != 0 && off1 > off0);
      CHECK(hk_root_dirty(desc) && desc->generation == gen + 1);
      CHECK(desc->set_sizes[0] == 64 && !desc->push_dirty);

      /* Identical re-push on G17P elides allocation, root dirty, and gen bump. */
      hk_root_mark_clean(desc);
      gen = desc->generation;
      hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      CHECK(desc->root.sets[0] == addr1 && cmd->uploader.main.offset == off1);
      CHECK(!hk_root_dirty(desc) && desc->generation == gen);

      /* Mutating the pushed UBO offset allocates a fresh immutable upload. */
      bi.offset = 64;
      hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      const uint64_t addr2 = desc->root.sets[0];
      CHECK(addr2 != addr1 && cmd->uploader.main.offset > off1);
      struct hk_buffer_address uploaded1, uploaded2;
      memcpy(&uploaded1, (const uint8_t *)f->main_map + addr1 - 0x100000,
             sizeof(uploaded1));
      memcpy(&uploaded2, (const uint8_t *)f->main_map + addr2 - 0x100000,
             sizeof(uploaded2));
      CHECK(uploaded1.base_addr == 0x800000 && uploaded2.base_addr == 0x800040);
      CHECK(hk_root_dirty(desc) && desc->generation == gen + 1);

      /* Changing pushed layout size forces a fresh upload even with matching prefix. */
      hk_root_mark_clean(desc);
      gen = desc->generation;
      push_layout->non_variable_descriptor_buffer_size = 96;
      hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      CHECK(desc->root.sets[0] != addr2 && desc->set_sizes[0] == 96);
      CHECK(hk_root_dirty(desc) && desc->generation == gen + 1);
      push_layout->non_variable_descriptor_buffer_size = 64;

      /* Reuse follows the backing command-pool BO cache mode (AGX_BO_WRITEBACK),
       * not the GPU chip: CPU-cached pool BOs elide on non-G17P chips, while
       * write-combined (!AGX_BO_WRITEBACK) pool BOs never record last_cpu_ptr
       * or read mapped memory even on G17P. */
      {
         struct hk_cmd_pool mock_pool = {0};
         mock_pool.vk.base.device = &f->dev.vk;
         for (enum hk_cmd_bo_type t = 0; t < HK_CMD_BO_TYPE_COUNT; ++t)
            list_inithead(&mock_pool.free_bos[t]);
         cmd->vk.pool = &mock_pool.vk;

         uint8_t wb_storage[256] = {0};
         uint8_t wc_storage[256] = {0};
         struct agx_bo wb_agx_bo = {.flags = AGX_BO_WRITEBACK};
         struct agx_bo wc_agx_bo = {.flags = 0};
         struct hk_cmd_bo wb_bo = {
            .bo = &wb_agx_bo,
            .map = wb_storage,
            .addr = 0x1000000,
            .size = sizeof(wb_storage),
         };
         struct hk_cmd_bo wc_bo = {
            .bo = &wc_agx_bo,
            .map = wc_storage,
            .addr = 0x2000000,
            .size = sizeof(wc_storage),
         };

         const struct hk_uploader saved_main = cmd->uploader.main;

         /* Non-G17P chips with an AGX_BO_WRITEBACK command-pool BO elide. */
         const enum agx_chip other_chips[] = {AGX_CHIP_G13G, AGX_CHIP_G17S};
         for (unsigned i = 0; i < ARRAY_SIZE(other_chips); ++i) {
            f->dev.dev.chip = other_chips[i];
            hk_push_descriptor_set_reset_upload(push);
            desc->root.sets[0] = 0;
            list_inithead(&cmd->uploader.main.bos);
            cmd->uploader.main.map = NULL;
            cmd->uploader.main.base = 0;
            cmd->uploader.main.offset = 0;
            cmd->uploader.main.capacity = 0;
            cmd->uploader.main.cpu_cached = false;
            list_addtail(&wb_bo.link, &mock_pool.free_bos[HK_CMD_BO_MAIN]);
            mock_pool.num_free_bos[HK_CMD_BO_MAIN] = 1;

            hk_root_mark_clean(desc);
            gen = desc->generation;
            hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd),
                                        &push_info);
            hk_cmd_buffer_flush_push_descriptors(cmd, desc);
            const uint64_t wb_addr = desc->root.sets[0];
            const uint32_t wb_off = cmd->uploader.main.offset;
            CHECK(wb_addr == wb_bo.addr && cmd->uploader.main.cpu_cached);
            CHECK(push->last_gpu_addr == wb_addr &&
                  push->last_cpu_ptr == wb_bo.map && push->last_size == 64);
            CHECK(hk_root_dirty(desc) && desc->generation == gen + 1);

            hk_root_mark_clean(desc);
            gen = desc->generation;
            hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd),
                                        &push_info);
            hk_cmd_buffer_flush_push_descriptors(cmd, desc);
            CHECK(desc->root.sets[0] == wb_addr &&
                  cmd->uploader.main.offset == wb_off);
            CHECK(!hk_root_dirty(desc) && desc->generation == gen);
            list_del(&wb_bo.link);
         }

         /* Write-combined (!AGX_BO_WRITEBACK) pool BOs never read mapped
          * memory on either G17P or non-G17P chips. */
         const enum agx_chip wc_chips[] = {
            AGX_CHIP_G17P, AGX_CHIP_G13G, AGX_CHIP_G17S,
         };
         for (unsigned i = 0; i < ARRAY_SIZE(wc_chips); ++i) {
            f->dev.dev.chip = wc_chips[i];
            hk_push_descriptor_set_reset_upload(push);
            list_inithead(&cmd->uploader.main.bos);
            cmd->uploader.main.map = NULL;
            cmd->uploader.main.base = 0;
            cmd->uploader.main.offset = 0;
            cmd->uploader.main.capacity = 0;
            cmd->uploader.main.cpu_cached = true;
            list_addtail(&wc_bo.link, &mock_pool.free_bos[HK_CMD_BO_MAIN]);
            mock_pool.num_free_bos[HK_CMD_BO_MAIN] = 1;

            hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd),
                                        &push_info);
            hk_cmd_buffer_flush_push_descriptors(cmd, desc);
            const uint64_t wc_addr1 = desc->root.sets[0];
            CHECK(wc_addr1 == wc_bo.addr && !cmd->uploader.main.cpu_cached);
            CHECK(push->last_gpu_addr == 0 && push->last_cpu_ptr == NULL &&
                  push->last_size == 0);

            hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd),
                                        &push_info);
            hk_cmd_buffer_flush_push_descriptors(cmd, desc);
            CHECK(desc->root.sets[0] != wc_addr1 &&
                  push->last_gpu_addr == 0 && push->last_cpu_ptr == NULL &&
                  push->last_size == 0);
            list_del(&wc_bo.link);
         }

         /* Slow-path BO burn without replacing the current uploader must use
          * the returned BO's cache mode, not the retained uploader's. */
         f->dev.dev.chip = AGX_CHIP_G17P;
         hk_push_descriptor_set_reset_upload(push);
         list_inithead(&cmd->uploader.main.bos);
         cmd->uploader.main.map = wb_storage;
         cmd->uploader.main.base = 0x1000000;
         cmd->uploader.main.capacity = 256;
         cmd->uploader.main.offset = 200; /* 56 B left < 64 B push set */
         cmd->uploader.main.cpu_cached = true;
         wc_bo.size = 100; /* 100 - 64 = 36 <= 56: old cached BO stays current */
         list_addtail(&wc_bo.link, &mock_pool.free_bos[HK_CMD_BO_MAIN]);
         mock_pool.num_free_bos[HK_CMD_BO_MAIN] = 1;
         hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
         hk_cmd_buffer_flush_push_descriptors(cmd, desc);
         CHECK(desc->root.sets[0] == wc_bo.addr &&
               cmd->uploader.main.map == wb_storage &&
               cmd->uploader.main.cpu_cached);
         CHECK(push->last_gpu_addr == 0 && push->last_cpu_ptr == NULL &&
               push->last_size == 0);
         list_del(&wc_bo.link);

         /* Inverse: current uploader is WC, burned BO is AGX_BO_WRITEBACK. */
         hk_push_descriptor_set_reset_upload(push);
         list_inithead(&cmd->uploader.main.bos);
         cmd->uploader.main.map = wc_storage;
         cmd->uploader.main.base = 0x2000000;
         cmd->uploader.main.capacity = 256;
         cmd->uploader.main.offset = 200;
         cmd->uploader.main.cpu_cached = false;
         wb_bo.size = 100;
         list_addtail(&wb_bo.link, &mock_pool.free_bos[HK_CMD_BO_MAIN]);
         mock_pool.num_free_bos[HK_CMD_BO_MAIN] = 1;
         hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
         hk_cmd_buffer_flush_push_descriptors(cmd, desc);
         CHECK(desc->root.sets[0] == wb_bo.addr &&
               cmd->uploader.main.map == wc_storage &&
               !cmd->uploader.main.cpu_cached);
         CHECK(push->last_gpu_addr == wb_bo.addr &&
               push->last_cpu_ptr == wb_bo.map && push->last_size == 64);
         hk_root_mark_clean(desc);
         gen = desc->generation;
         hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
         hk_cmd_buffer_flush_push_descriptors(cmd, desc);
         CHECK(desc->root.sets[0] == wb_bo.addr &&
               !hk_root_dirty(desc) && desc->generation == gen);
         list_del(&wb_bo.link);

         cmd->vk.pool = NULL;
         cmd->uploader.main = saved_main;
         list_inithead(&cmd->uploader.main.bos);
         hk_push_descriptor_set_reset_upload(push);
         f->dev.dev.chip = AGX_CHIP_G17P;
      }

      /* Binding a regular descriptor set resets the push upload snapshot. */
      hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      CHECK(push->last_gpu_addr != 0 && push->last_cpu_ptr != NULL);
      bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
      CHECK(push->last_gpu_addr == 0 && push->last_cpu_ptr == NULL &&
            push->last_size == 0);

      /* Binding a descriptor buffer (hk_bind_buffer_set) resets the snapshot. */
      hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      CHECK(push->last_gpu_addr != 0 && push->last_cpu_ptr != NULL);
      cmd->state.descriptor_buffers[0] = 0x900000;
      const uint32_t buf_idx = 0;
      const VkDeviceSize buf_off = 0;
      const VkSetDescriptorBufferOffsetsInfoEXT dbo = {
         .sType = VK_STRUCTURE_TYPE_SET_DESCRIPTOR_BUFFER_OFFSETS_INFO_EXT,
         .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
         .layout = vk_pipeline_layout_to_handle(&push_pl),
         .firstSet = 0,
         .setCount = 1,
         .pBufferIndices = &buf_idx,
         .pOffsets = &buf_off,
      };
      hk_CmdSetDescriptorBufferOffsets2EXT(hk_cmd_buffer_to_handle(cmd), &dbo);
      CHECK(push->last_gpu_addr == 0 && push->last_cpu_ptr == NULL &&
            push->last_size == 0);

      /* Meta save/restore resets push upload snapshots across all branches. */
      cmd->vk.dynamic_graphics_state.vi = &cmd->state.gfx._dynamic_vi;
      cmd->vk.dynamic_graphics_state.ms.sample_locations =
         &cmd->state.gfx._dynamic_sl;
      struct hk_meta_save save = {0};

      /* Branch 1: save->has_push_desc0 restores saved bytes and resets snapshot. */
      bi.offset = 0;
      hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
      bi.offset = 128;
      hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      CHECK(push->last_gpu_addr != 0);
      hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
      CHECK(push->last_gpu_addr == 0 && push->last_cpu_ptr == NULL &&
            push->last_size == 0);
      struct hk_buffer_address restored;
      memcpy(&restored, push->data, sizeof(restored));
      CHECK(restored.base_addr == 0x800000);

      /* Branch 2: save->desc0 resets desc->push[0] if meta pushed to set 0. */
      bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
      hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
      hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      CHECK(push->last_gpu_addr != 0);
      hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
      CHECK(push->last_gpu_addr == 0 && push->last_cpu_ptr == NULL &&
            push->last_size == 0);

      /* Branch 3: trailing else (neither set0 nor push0 at begin) resets snapshot. */
      desc->sets[0] = NULL;
      desc->push[0] = NULL;
      desc->push_active = 0;
      desc->buffer_active = 0;
      hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
      desc->push[0] = push;
      hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      CHECK(push->last_gpu_addr != 0);
      hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_GRAPHICS);
      CHECK(push->last_gpu_addr == 0 && push->last_cpu_ptr == NULL &&
            push->last_size == 0);

      /* Secondary command-buffer state invalidation resets gfx and cs snapshots. */
      hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
      CHECK(push->last_gpu_addr != 0);
      hk_cmd_invalidate_graphics_state(cmd);
      CHECK(push->last_gpu_addr == 0 && push->last_cpu_ptr == NULL &&
            push->last_size == 0);

      cmd->state.cs.descriptors.push[0] = cs_push;
      push_info.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
      hk_cmd_buffer_flush_push_descriptors(cmd, &cmd->state.cs.descriptors);
      CHECK(cs_push->last_gpu_addr != 0);
      hk_cmd_invalidate_compute_state(cmd);
      CHECK(cs_push->last_gpu_addr == 0 && cs_push->last_cpu_ptr == NULL &&
            cs_push->last_size == 0);

      /* Transfer-to-draw xfer_clean cache must not skip hk_xfer_match_descriptors()
       * when push_dirty is set before desc->generation advances at flush. */
      push_info.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS;
      f->set_layout.descriptor_generation = 9;
      bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
      struct hk_cs xfer_cs = {
         .type = HK_CS_CDM,
         .sync = {.seq = 1},
      };
      memset(&cmd->sync, 0, sizeof(cmd->sync));
      cmd->sync.xfer_writes[0] = (struct hk_sync_xfer_write){
         .lo = 0x900000,
         .hi = 0x901000,
         .cs = &xfer_cs,
      };
      cmd->sync.xfer_write_count = 1;
      cmd->sync.xfer_version = 1;
      hk_xfer_check_draw_slow(cmd, 0, 0);
      CHECK(cmd->sync.xfer_clean.valid && cmd->sync.xfer_write_count == 1);
      CHECK(cmd->sync.xfer_clean.descriptors == desc->generation);

      hk_CmdPushDescriptorSet2KHR(hk_cmd_buffer_to_handle(cmd), &push_info);
      CHECK(desc->push_dirty == BITFIELD_BIT(0));
      CHECK(cmd->sync.xfer_clean.descriptors == desc->generation);
      hk_xfer_check_draw_slow(cmd, 0, 0);
      CHECK(cmd->sync.xfer_write_count == 0 && !cmd->sync.xfer_clean.valid);

      memset(&cmd->sync, 0, sizeof(cmd->sync));
      f->set_layout.descriptor_generation = 0;
      util_dynarray_fini(&cmd->meta_push_saves);
      desc->push[0] = NULL;
      cmd->state.cs.descriptors.push[0] = NULL;
      free(cs_push);
      free(push);
      free(push_layout);
   }

   reset(f);
}

static void
reuse_correctness(struct fixture *f)
{
   struct hk_cmd_buffer *cmd = &f->cmd;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   bind(f, 16, VK_SHADER_STAGE_ALL_GRAPHICS);
   const uint64_t generation = desc->generation;
   hk_root_mark_clean(desc);
   desc->sets_dirty = 0;
   bind(f, 16, VK_SHADER_STAGE_ALL_GRAPHICS);
   CHECK(desc->generation == generation && !hk_root_dirty(desc) && !desc->sets_dirty);
   bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
   CHECK(desc->generation == generation + 1 && hk_root_dirty(desc));

   uint64_t root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS) == root);
   cmd->state.gfx.root = root;
   cmd->state.gfx.g17p_bind_tail.valid = true;
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS) == root);
   CHECK(cmd->state.gfx.g17p_bind_tail.valid);
   const size_t head = hk_root_descriptor_offset(dynamic_buffers) +
      desc->dynamic_buffer_end * sizeof(desc->root.dynamic_buffers[0]);
   const size_t tail = hk_root_descriptor_offset(set_dynamic_buffer_start);
   memset((uint8_t *)desc->uploaded_root.cpu + head, 0xa5, tail - head);
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS) == root);
   /* Restoring descriptor state can return to an older cached root. */
   cmd->state.gfx.root = root + 8;
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS) == root);
   CHECK(!cmd->state.gfx.g17p_bind_tail.valid);
   cmd->state.gfx.root = root;

   const uint64_t compute = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
   CHECK(compute != root);
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE) == compute);
   /* A shader switch must not reuse a shorter root allocation. Test the
    * copied prefix, omitted footer, reuse, expansion and contraction. */
   struct hk_api_shader *cs = calloc(1, hk_api_shader_size(MESA_SHADER_COMPUTE, false));
   CHECK(cs);
   cs->vk.stage = MESA_SHADER_COMPUTE;
   cmd->state.cs.shader = cs;
   struct hk_descriptor_state *cd = &cmd->state.cs.descriptors;
   const size_t small_B = 64;
   cs->variants[0].root_descriptor_size = small_B;
   cd->root.push[0] = 0x78;
   uint64_t small = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
   CHECK(small != compute && cd->uploaded_root_size == small_B);
   CHECK(memcmp(cd->uploaded_root.cpu, &cd->root, small_B) == 0);
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE) == small);
   cs->variants[0].root_descriptor_size = sizeof(cd->root);
   uint64_t full = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
   CHECK(full != small && cd->uploaded_root_size == sizeof(cd->root));
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE) == full);
   cs->variants[0].root_descriptor_size = small_B;
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE) != full);
   CHECK(cd->uploaded_root_size == small_B);
   /* A retained or adversarial short-prefix value must have no effect on
    * any non-HAL200 chip, including the native-encoding M3 (G15) path. */
   for (enum agx_chip chip = AGX_CHIP_G13G; chip < AGX_CHIP_UNKNOWN; ++chip) {
      if (agx_chip_native_hal200(chip))
         continue;
      f->dev.dev.chip = chip;
      CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE));
      CHECK(cd->uploaded_root_size == sizeof(cd->root));
   }
   const enum agx_chip hal200_chips[] = {AGX_CHIP_G17P, AGX_CHIP_G16G};
   const size_t gfx_small_B =
      ALIGN_POT(hk_root_descriptor_offset(push) + 16, 16);
   const size_t gfx_large_B =
      ALIGN_POT(hk_root_descriptor_offset(push) + 64, 16);
   for (unsigned i = 0; i < ARRAY_SIZE(hal200_chips); ++i) {
      f->dev.dev.chip = hal200_chips[i];
      CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE));
      CHECK(cd->uploaded_root_size == small_B);

      /* Graphics root trimming + growth detection on both G17P and G16G. */
      f->vs_api->variants[HK_VS_VARIANT_HW].root_descriptor_size = gfx_small_B;
      f->fs_api->variants[0].root_descriptor_size =
         ALIGN_POT(hk_root_descriptor_offset(push), 16);
      CHECK(hk_gfx_root_descriptor_size(cmd) == gfx_small_B);
      uint64_t gsmall =
         hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
      CHECK(gsmall && desc->uploaded_root_size == gfx_small_B);
      CHECK(!hk_root_upload_needs_growth(desc->uploaded_root_size,
                                         hk_gfx_root_descriptor_size(cmd)));

      /* Binding a shader with a larger root forces a re-upload. */
      f->fs_api->variants[0].root_descriptor_size = gfx_large_B;
      CHECK(hk_gfx_root_descriptor_size(cmd) == gfx_large_B);
      CHECK(hk_root_upload_needs_growth(desc->uploaded_root_size,
                                        hk_gfx_root_descriptor_size(cmd)));
      uint64_t glarge =
         hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
      CHECK(glarge && glarge != gsmall &&
            desc->uploaded_root_size == gfx_large_B);

      /* Missing FS trimmed footprint falls back to full root table. */
      f->fs_api->variants[0].root_descriptor_size = 0;
      CHECK(hk_gfx_root_descriptor_size(cmd) == sizeof(desc->root));
      CHECK(hk_root_upload_needs_growth(desc->uploaded_root_size,
                                        hk_gfx_root_descriptor_size(cmd)));
   }
   /* Non-HAL200 chips (including M3 G15G) keep full graphics root uploads. */
   f->vs_api->variants[HK_VS_VARIANT_HW].root_descriptor_size = gfx_small_B;
   f->fs_api->variants[0].root_descriptor_size = gfx_small_B;
   f->dev.dev.chip = AGX_CHIP_G15G;
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS));
   CHECK(desc->uploaded_root_size == sizeof(desc->root));

   f->dev.dev.chip = AGX_CHIP_G17P;
   f->vs_api->variants[HK_VS_VARIANT_HW].root_descriptor_size = 0;
   f->fs_api->variants[0].root_descriptor_size = 0;
   root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   CHECK(root && desc->uploaded_root_size == sizeof(desc->root));
   cmd->state.cs.shader = NULL;
   free(cs);
   /* Test both sides of the omitted dynamic-buffer region. */
   desc->root.push[3] = 42;
   uint64_t changed = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   CHECK(changed != root);
   CHECK(((struct hk_root_descriptor_table *)f->root_map)->push[3] == 0);
   root = changed;
   f->dev.dev.printf_buffer_address = 0x700000;
   changed = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   CHECK(changed != root);
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS) == changed);
   root = changed;
   desc->dynamic_buffer_end++;
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS) != root);
   f->dev.dev.printf_buffer_address = 0;

   /* Redundant graphics descriptor binds retain the uploaded root across all
    * GPU generations, not only native AGX3. */
   const enum agx_chip retain_chips[] = {
      AGX_CHIP_G13G, AGX_CHIP_G14G, AGX_CHIP_G16G,
      AGX_CHIP_G17P, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(retain_chips); ++i) {
      f->dev.dev.chip = retain_chips[i];
      bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
      const uint64_t gen = desc->generation;
      hk_root_mark_clean(desc);
      desc->sets_dirty = 0;
      bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
      CHECK(desc->generation == gen && !hk_root_dirty(desc) && !desc->sets_dirty);
      bind(f, 64, VK_SHADER_STAGE_ALL_GRAPHICS);
      CHECK(desc->generation == gen + 1 && hk_root_dirty(desc));
   }
   f->dev.dev.chip = AGX_CHIP_G17P;
   bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);

   /* CPU-decoded descriptors deliberately keep their publication path. */
   f->fs_api->variants[0].info.agx3_combined_sampler_valid = true;
   root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   const uint64_t before = desc->generation;
   hk_root_mark_clean(desc);
   bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
   CHECK(desc->generation == before + 1 && hk_root_dirty(desc));
   CHECK(hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS) != root);
   f->fs_api->variants[0].info.agx3_combined_sampler_valid = false;

   hk_cmd_upload_draw_params(cmd, 7, 3);
   const uint64_t params = cmd->state.gfx.draw_params;
   hk_cmd_upload_draw_params(cmd, 7, 3);
   CHECK(cmd->state.gfx.draw_params == params);
   hk_cmd_upload_draw_params(cmd, 7, 4);
   CHECK(cmd->state.gfx.draw_params != params);
   hk_cmd_upload_draw_id(cmd, 1234);
   const uint64_t id = cmd->state.gfx.draw_id_ptr;
   hk_cmd_upload_draw_id(cmd, 1234);
   CHECK(cmd->state.gfx.draw_id_ptr == id);

   const uint64_t data = hk_cmd_upload_constants(cmd, &f->vs, MESA_SHADER_VERTEX);
   CHECK(hk_cmd_upload_constants(cmd, &f->vs, MESA_SHADER_VERTEX) == data);
   CHECK(hk_cmd_upload_constants(cmd, &f->vs, MESA_SHADER_COMPUTE) != data);
   /* A new shader may occupy the same malloc address. Its unique payload ID
    * must force a new immutable upload without changing the earlier bytes. */
   const uint8_t *old = (uint8_t *)f->main_map + data - 0x100000;
   CHECK(old[0] == 0);
   f->constants[0] = 99;
   f->vs.constant_data_id = 3;
   CHECK(hk_cmd_upload_constants(cmd, &f->vs, MESA_SHADER_VERTEX) != data);
   CHECK(old[0] == 0);
   f->constants[0] = 0;
   f->vs.constant_data_id = 1;

   reset(f);
   CHECK(cmd->state.gfx.descriptors.uploaded_root.gpu == 0);
   CHECK(cmd->state.constant_data[MESA_SHADER_VERTEX].id == 0);
   CHECK(!cmd->state.gfx.g17p_draw_params_host_valid);
   hk_cmd_upload_draw_params(cmd, 7, 3);
   hk_cmd_upload_constants(cmd, &f->vs, MESA_SHADER_VERTEX);
   CHECK(cmd->state.upload_stats.draw_param_uploads == 1);
   CHECK(cmd->state.upload_stats.constant_uploads == 1);

   const VkDispatchIndirectCommand gc_a = {32, 4, 1};
   const VkDispatchIndirectCommand gc_b = {64, 1, 1};
   CHECK(!cmd->used_group_count_cache);
   const uint64_t gc_addr_a = hk_cmd_upload_group_count(cmd, gc_a);
   CHECK(gc_addr_a != 0);
   CHECK(cmd->used_group_count_cache);
   const uint32_t main_off_after_a = cmd->uploader.main.offset;
   CHECK(hk_cmd_upload_group_count(cmd, gc_a) == gc_addr_a);
   CHECK(cmd->uploader.main.offset == main_off_after_a);
   const uint64_t gc_addr_b = hk_cmd_upload_group_count(cmd, gc_b);
   CHECK(gc_addr_b != 0 && gc_addr_b != gc_addr_a);
   CHECK(hk_cmd_upload_group_count(cmd, gc_a) == gc_addr_a);
   reset(f);
   CHECK(!cmd->used_group_count_cache);
   for (unsigned i = 0; i < ARRAY_SIZE(cmd->group_count_cache); ++i)
      CHECK(!cmd->group_count_cache[i].valid);
   /* After reset, uploading gc_a again must not return a stale address from
    * the previous recording even when the bump allocator rewinds to 0. */
   const uint64_t gc_addr_a2 = hk_cmd_upload_group_count(cmd, gc_a);
   CHECK(gc_addr_a2 != 0 && cmd->used_group_count_cache);
   CHECK(cmd->uploader.main.offset > 0);
   reset(f);
   CHECK(!cmd->used_group_count_cache);
}

static uint64_t
now_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (uint64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

static void
run(struct fixture *f, const char *name, unsigned change_every, unsigned batches)
{
   struct hk_upload_stats totals = {0};
   const uint64_t start = now_ns();
   for (unsigned b = 0; b < batches; ++b) {
      reset(f);
      for (unsigned d = 0; d < 16; ++d) {
         const uint32_t value = d / change_every;
         bind(f, value * 16, VK_SHADER_STAGE_ALL_GRAPHICS);
         struct hk_cmd_buffer *cmd = &f->cmd;
         /* The general draw path calls this only for a dirty root. Force the
          * call too, to measure pass-boundary/over-invalidation reuse. */
         cmd->state.gfx.root =
            hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
         hk_cmd_upload_draw_params(cmd, value, 0);
         hk_cmd_upload_draw_id(cmd, value);
         CHECK(hk_cmd_upload_constants(cmd, &f->vs, MESA_SHADER_VERTEX));
         CHECK(hk_cmd_upload_constants(cmd, &f->fs, MESA_SHADER_FRAGMENT));
         hk_upload_stat(cmd, draws, 1);
      }
#define SUM(field) totals.field += f->cmd.state.upload_stats.field
      SUM(draws); SUM(descriptor_binds); SUM(descriptor_noops);
      SUM(root_uploads); SUM(root_bytes); SUM(root_reuses);
      SUM(draw_param_uploads); SUM(draw_param_bytes); SUM(draw_param_reuses);
      SUM(draw_id_uploads); SUM(draw_id_bytes); SUM(draw_id_reuses);
      SUM(constant_uploads); SUM(constant_bytes); SUM(constant_reuses);
#undef SUM
      CHECK(f->cmd.vk.record_result == VK_SUCCESS);
   }
   const uint64_t elapsed = now_ns() - start;
   const uint64_t changes = (15 / change_every) + 1;
   CHECK(totals.draws == (uint64_t)batches * 16);
   CHECK(totals.descriptor_binds == totals.draws);
   CHECK(totals.descriptor_noops == (16 - changes) * batches);
   CHECK(f->cmd.state.gfx.descriptors.generation == changes);
   CHECK(totals.root_uploads == changes * batches);
   CHECK(totals.root_reuses == (16 - changes) * batches);
   CHECK(totals.draw_param_uploads == changes * batches);
   CHECK(totals.draw_param_reuses == (16 - changes) * batches);
   CHECK(totals.draw_id_uploads == changes * batches);
   CHECK(totals.draw_id_reuses == (16 - changes) * batches);
   CHECK(totals.constant_uploads == (uint64_t)batches * 2);
   CHECK(totals.constant_reuses == (uint64_t)batches * 30);
   f->cmd.state.upload_stats = totals;
   printf("%s batches=%u draws=%" PRIu64 " ns_per_draw=%.2f\n", name, batches,
          totals.draws, (double)elapsed / totals.draws);
   hk_cmd_print_upload_stats(&f->cmd);
   printf("generation_per_batch=%" PRIu64 " root_struct_bytes=%zu\n",
          f->cmd.state.gfx.descriptors.generation,
          sizeof(struct hk_root_descriptor_table));
}

int
main(int argc, char **argv)
{
   unsigned batches = argc > 1 ? strtoul(argv[1], NULL, 10) : 1000;
   CHECK(batches && batches <= 1000000);
   struct fixture *f = fixture();
   correctness(f);
   reuse_correctness(f);
   run(f, "identical", 16, batches);
   run(f, "change_every_4", 4, batches);
   run(f, "change_every_draw", 1, batches);
   free(f->vs_api);
   free(f->fs_api);
   free(f->set);
   free(f->main_map);
   free(f->root_map);
   free(f);
   return 0;
}

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
   f->cmd.uploader.main.map = f->main_map;
   f->cmd.uploader.main.base = 0x100000;
   f->cmd.uploader.main.capacity = HK_CMD_BO_SIZE;
   f->cmd.uploader.root.map = f->root_map;
   f->cmd.uploader.root.base = 0x200000;
   f->cmd.uploader.root.capacity = HK_CMD_BO_SIZE;
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
}

static void
reuse_correctness(struct fixture *f)
{
   struct hk_cmd_buffer *cmd = &f->cmd;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   bind(f, 16, VK_SHADER_STAGE_ALL_GRAPHICS);
   const uint64_t generation = desc->generation;
   desc->root_dirty = false;
   desc->sets_dirty = 0;
   bind(f, 16, VK_SHADER_STAGE_ALL_GRAPHICS);
   CHECK(desc->generation == generation && !desc->root_dirty && !desc->sets_dirty);
   bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
   CHECK(desc->generation == generation + 1 && desc->root_dirty);

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

   /* CPU-decoded descriptors deliberately keep their publication path. */
   f->fs_api->variants[0].info.agx3_combined_sampler_valid = true;
   root = hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   const uint64_t before = desc->generation;
   desc->root_dirty = false;
   bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
   CHECK(desc->generation == before + 1 && desc->root_dirty);
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
   const uint64_t gc_addr_a = hk_cmd_upload_group_count(cmd, gc_a);
   CHECK(gc_addr_a != 0);
   const uint32_t main_off_after_a = cmd->uploader.main.offset;
   CHECK(hk_cmd_upload_group_count(cmd, gc_a) == gc_addr_a);
   CHECK(cmd->uploader.main.offset == main_off_after_a);
   const uint64_t gc_addr_b = hk_cmd_upload_group_count(cmd, gc_b);
   CHECK(gc_addr_b != 0 && gc_addr_b != gc_addr_a);
   CHECK(hk_cmd_upload_group_count(cmd, gc_a) == gc_addr_a);
   reset(f);
   for (unsigned i = 0; i < ARRAY_SIZE(cmd->state.group_count_cache); ++i)
      CHECK(!cmd->state.group_count_cache[i].valid);
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

static void
mixed_gfx_compute_reset_reuse_regression(struct fixture *f)
{
   /* Verify the two independent capability predicates across chip families:
    * stream-uploader split follows control-stream allocation layout, while
    * descriptor-root retention follows the Ioto GPU-memory root contract. */
   const enum agx_chip split_chips[] = {
      AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G16G,
      AGX_CHIP_G17P, AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };
   const enum agx_chip nonsplit_chips[] = {
      AGX_CHIP_G13G, AGX_CHIP_G13X, AGX_CHIP_G14G, AGX_CHIP_G14X,
      AGX_CHIP_G18G,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(split_chips); ++i) {
      CHECK(hk_cs_uses_dedicated_stream_uploader(split_chips[i]));
      CHECK(hk_chip_uses_ioto_descriptor_root(split_chips[i]));
   }
   for (unsigned i = 0; i < ARRAY_SIZE(nonsplit_chips); ++i) {
      CHECK(!hk_cs_uses_dedicated_stream_uploader(nonsplit_chips[i]));
      CHECK(!hk_chip_uses_ioto_descriptor_root(nonsplit_chips[i]));
   }

   struct hk_cmd_buffer *cmd = &f->cmd;
   struct hk_cmd_pool pool = {0};
   pool.vk.base.device = &f->dev.vk;
   for (unsigned i = 0; i < HK_CMD_BO_TYPE_COUNT; ++i) {
      list_inithead(&pool.free_bos[i]);
      list_inithead(&pool.free_spans[i]);
   }

   cmd->vk.pool = &pool.vk;
   cmd->vk.ops = &hk_cmd_buffer_ops;
   cmd->vk.dynamic_graphics_state.vi = &cmd->state.gfx._dynamic_vi;
   cmd->vk.dynamic_graphics_state.ms.sample_locations =
      &cmd->state.gfx._dynamic_sl;
   list_inithead(&cmd->uploader.main.bos);
   list_inithead(&cmd->uploader.usc.bos);
   list_inithead(&cmd->uploader.root.bos);
   list_inithead(&cmd->vdm_allocations);
   list_inithead(&cmd->control_streams);
   list_inithead(&cmd->alias_link);

   void *streams_map = calloc(1, HK_CMD_BO_SIZE);
   struct hk_api_shader *gs_api =
      calloc(1, hk_api_shader_size(MESA_SHADER_GEOMETRY, false));
   CHECK(streams_map && gs_api);
   gs_api->vk.stage = MESA_SHADER_GEOMETRY;

   const enum agx_chip m5_chips[] = {
      AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C, AGX_CHIP_G17P,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(m5_chips); ++i) {
      const enum agx_chip chip = m5_chips[i];
      f->dev.dev.chip = chip;
      f->dev.dev.params.gpu_generation = agx_chip_g17_m5(chip) ? 17 : 16;

      /* Run two full mixed graphics/compute record -> reset -> reuse cycles. */
      for (unsigned cycle = 0; cycle < 2; ++cycle) {
         hk_cmd_buffer_ops.reset(&cmd->vk, 0);
         CHECK(cmd->uploader.main.map == NULL && cmd->uploader.main.offset == 0);
         CHECK(cmd->uploader.streams.map == NULL &&
               cmd->uploader.streams.offset == 0);
         CHECK(cmd->state.gfx.descriptors.generation == 0);
         CHECK(cmd->state.cs.descriptors.generation == 0);

         /* Arm distinct main and control-stream upload windows. */
         cmd->uploader.main.map = f->main_map;
         cmd->uploader.main.base = 0x100000;
         cmd->uploader.main.offset = 0;
         cmd->uploader.main.capacity = HK_CMD_BO_SIZE;
         cmd->uploader.root.map = f->root_map;
         cmd->uploader.root.base = 0x200000;
         cmd->uploader.root.offset = 0;
         cmd->uploader.root.capacity = HK_CMD_BO_SIZE;
         cmd->uploader.streams.map = streams_map;
         cmd->uploader.streams.base = 0x500000;
         cmd->uploader.streams.offset = 0;
         cmd->uploader.streams.capacity = HK_CMD_BO_SIZE;

         cmd->state.gfx.shaders[MESA_SHADER_VERTEX] = f->vs_api;
         cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = f->fs_api;

         /* Mixed graphics (VDM) + compute (CDM) control-stream roots must use
          * uploader.streams without moving uploader.main. */
         const struct agx_ptr main_before = hk_pool_alloc(cmd, 64, 16);
         CHECK(main_before.gpu == 0x100000);
         const uint32_t main_off_before = cmd->uploader.main.offset;

         const size_t vdm_sz = hk_cs_root_size(chip, false);
         const size_t cdm_sz = hk_cs_root_size(chip, true);
         const struct agx_ptr vdm_root = hk_pool_cs_alloc(cmd, vdm_sz, 1024);
         const struct agx_ptr cdm_root = hk_pool_cs_alloc(cmd, cdm_sz, 1024);
         CHECK(vdm_root.gpu == 0x500000);
         CHECK(cdm_root.gpu == 0x500000 + vdm_sz);
         CHECK(cmd->uploader.streams.offset == vdm_sz + cdm_sz);
         CHECK(cmd->uploader.main.offset == main_off_before);

         /* First mixed bind after reset must dirty both graphics and compute
          * roots (never falsely retained across a reset). */
         bind(f, 16, VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT);
         CHECK(cmd->state.gfx.descriptors.root_dirty);
         CHECK(cmd->state.cs.descriptors.root_dirty);
         CHECK(cmd->state.gfx.descriptors.generation == 1);
         CHECK(cmd->state.cs.descriptors.generation == 1);

         const uint64_t gfx_root =
            hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
         const uint64_t cs_root =
            hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
         CHECK(gfx_root != 0 && cs_root != 0 && gfx_root != cs_root);
         cmd->state.gfx.descriptors.root_dirty = false;
         cmd->state.cs.descriptors.root_dirty = false;

         /* Identical mixed re-bind: graphics retains its root as a no-op while
          * compute (outside graphics retention) stays conservatively dirty. */
         const uint64_t noops_before = cmd->state.upload_stats.descriptor_noops;
         bind(f, 16, VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT);
         CHECK(!cmd->state.gfx.descriptors.root_dirty);
         CHECK(cmd->state.gfx.descriptors.generation == 1);
         CHECK(cmd->state.upload_stats.descriptor_noops == noops_before + 1);
         CHECK(cmd->state.cs.descriptors.root_dirty);
         CHECK(cmd->state.cs.descriptors.generation == 2);

         /* Dynamic-offset change dirties graphics root and advances generation. */
         bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
         CHECK(cmd->state.gfx.descriptors.root_dirty);
         CHECK(cmd->state.gfx.descriptors.generation == 2);
         cmd->state.gfx.descriptors.root_dirty = false;

         /* Shader-stage exclusion (geometry shader) disables retention even
          * for an identical bind. */
         cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] = gs_api;
         bind(f, 32, VK_SHADER_STAGE_ALL_GRAPHICS);
         CHECK(cmd->state.gfx.descriptors.root_dirty);
         CHECK(cmd->state.gfx.descriptors.generation == 3);
         cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] = NULL;
      }
   }

   /* Non-split / non-Ioto chips (AGX2 G13G, G18G) keep cs roots on main and
    * do not enable graphics descriptor-root retention. */
   const enum agx_chip legacy_chips[] = {AGX_CHIP_G13G, AGX_CHIP_G18G};
   for (unsigned i = 0; i < ARRAY_SIZE(legacy_chips); ++i) {
      f->dev.dev.chip = legacy_chips[i];
      f->dev.dev.params.gpu_generation =
         legacy_chips[i] == AGX_CHIP_G13G ? 13 : 18;
      hk_cmd_buffer_ops.reset(&cmd->vk, 0);
      cmd->uploader.main.map = f->main_map;
      cmd->uploader.main.base = 0x100000;
      cmd->uploader.main.capacity = HK_CMD_BO_SIZE;
      cmd->uploader.streams.map = streams_map;
      cmd->uploader.streams.base = 0x500000;
      cmd->uploader.streams.capacity = HK_CMD_BO_SIZE;
      cmd->state.gfx.shaders[MESA_SHADER_VERTEX] = f->vs_api;
      cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT] = f->fs_api;

      const struct agx_ptr cs_root = hk_pool_cs_alloc(cmd, 4096, 1024);
      CHECK(cs_root.gpu == 0x100000);
      CHECK(cmd->uploader.streams.offset == 0);

      bind(f, 16, VK_SHADER_STAGE_ALL_GRAPHICS);
      cmd->state.gfx.descriptors.root_dirty = false;
      bind(f, 16, VK_SHADER_STAGE_ALL_GRAPHICS);
      CHECK(cmd->state.gfx.descriptors.root_dirty);
      CHECK(cmd->state.gfx.descriptors.generation == 2);
   }

   hk_cmd_buffer_ops.reset(&cmd->vk, 0);
   cmd->vk.pool = NULL;
   cmd->vk.ops = NULL;
   free(gs_api);
   free(streams_map);

   f->dev.dev.chip = AGX_CHIP_G17P;
   f->dev.dev.params.gpu_generation = 17;
   cmd->uploader.main.map = f->main_map;
   cmd->uploader.main.base = 0x100000;
   cmd->uploader.main.capacity = HK_CMD_BO_SIZE;
   cmd->uploader.root.map = f->root_map;
   cmd->uploader.root.base = 0x200000;
   cmd->uploader.root.capacity = HK_CMD_BO_SIZE;
   reset(f);
}

int
main(int argc, char **argv)
{
   unsigned batches = argc > 1 ? strtoul(argv[1], NULL, 10) : 1000;
   CHECK(batches && batches <= 1000000);
   struct fixture *f = fixture();
   correctness(f);
   reuse_correctness(f);
   mixed_gfx_compute_reset_reuse_regression(f);
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

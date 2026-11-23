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
#include "vk_instance.h"
#include "vk_physical_device.h"
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

   const enum agx_chip upload_reuse_chips[] = {
      AGX_CHIP_G16G, AGX_CHIP_G17P, AGX_CHIP_G17G, AGX_CHIP_G17S,
      AGX_CHIP_G17C,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(upload_reuse_chips); ++i) {
      f->dev.dev.chip = upload_reuse_chips[i];
      reset(f);
      hk_cmd_upload_draw_params(cmd, 7, 3);
      const uint64_t params = cmd->state.gfx.draw_params;
      CHECK(params != 0);
      hk_cmd_upload_draw_params(cmd, 7, 3);
      CHECK(cmd->state.gfx.draw_params == params);
      CHECK(cmd->state.upload_stats.draw_param_uploads == 1);
      CHECK(cmd->state.upload_stats.draw_param_reuses == 1);
      /* Indirect draw invalidation forces a fresh direct upload. */
      cmd->state.gfx.g17p_draw_params_host_valid = false;
      hk_cmd_upload_draw_params(cmd, 7, 3);
      CHECK(cmd->state.gfx.draw_params != params);
      const uint64_t params2 = cmd->state.gfx.draw_params;
      hk_cmd_upload_draw_params(cmd, 7, 4);
      CHECK(cmd->state.gfx.draw_params != params2);

      hk_cmd_upload_draw_id(cmd, 1234);
      const uint64_t id = cmd->state.gfx.draw_id_ptr;
      CHECK(id != 0);
      hk_cmd_upload_draw_id(cmd, 1234);
      CHECK(cmd->state.gfx.draw_id_ptr == id);
      CHECK(cmd->state.upload_stats.draw_id_uploads == 1);
      CHECK(cmd->state.upload_stats.draw_id_reuses == 1);
      hk_cmd_upload_draw_id(cmd, 1235);
      CHECK(cmd->state.gfx.draw_id_ptr != id);

      const uint64_t data = hk_cmd_upload_constants(cmd, &f->vs, MESA_SHADER_VERTEX);
      CHECK(data != 0);
      CHECK(hk_cmd_upload_constants(cmd, &f->vs, MESA_SHADER_VERTEX) == data);
      CHECK(hk_cmd_upload_constants(cmd, &f->vs, MESA_SHADER_COMPUTE) != data);
      CHECK(cmd->state.upload_stats.constant_uploads == 2);
      CHECK(cmd->state.upload_stats.constant_reuses == 1);
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
   }

   /* Non-AGX3 chips (G13G, G14G) do not use the native upload reuse caches. */
   const enum agx_chip non_reuse_chips[] = {AGX_CHIP_G13G, AGX_CHIP_G14G};
   for (unsigned i = 0; i < ARRAY_SIZE(non_reuse_chips); ++i) {
      f->dev.dev.chip = non_reuse_chips[i];
      reset(f);
      hk_cmd_upload_draw_params(cmd, 7, 3);
      const uint64_t prev_params = cmd->state.gfx.draw_params;
      hk_cmd_upload_draw_params(cmd, 7, 3);
      CHECK(cmd->state.gfx.draw_params != prev_params);
      hk_cmd_upload_draw_id(cmd, 1234);
      const uint64_t prev_id = cmd->state.gfx.draw_id_ptr;
      hk_cmd_upload_draw_id(cmd, 1234);
      CHECK(cmd->state.gfx.draw_id_ptr != prev_id);
      const uint64_t prev_data =
         hk_cmd_upload_constants(cmd, &f->vs, MESA_SHADER_VERTEX);
      CHECK(hk_cmd_upload_constants(cmd, &f->vs, MESA_SHADER_VERTEX) !=
            prev_data);
   }
   f->dev.dev.chip = AGX_CHIP_G17P;

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
graphics_launch_replay_correctness(struct fixture *f)
{
   struct hk_cmd_buffer *cmd = &f->cmd;
   const uint64_t shader_base = UINT64_C(0x1000000000);
   const size_t ioto_capacity = 64 * 1024;
   uint8_t *ioto_map = calloc(1, ioto_capacity);
   struct hk_api_shader *mesh_api =
      calloc(1, hk_api_shader_size(MESA_SHADER_MESH, false));
   CHECK(ioto_map && mesh_api);
   mesh_api->vk.stage = MESA_SHADER_MESH;
   mesh_api->variants[HK_MESH_VARIANT_MAIN].compiled = true;
   mesh_api->variants[HK_MESH_VARIANT_MAIN].info.stage = MESA_SHADER_MESH;

   struct agx_va arena_va = {.addr = shader_base + 0x20000, .size_B = ioto_capacity};
   struct agx_bo arena_bo = {
      .dev = &f->dev.dev,
      .va = &arena_va,
      .size = ioto_capacity,
      .refcnt = 1,
   };
   struct agx_va shader_va = {.addr = shader_base + 0x40000, .size_B = 4096};
   struct agx_bo shader_bo = {
      .dev = &f->dev.dev,
      .va = &shader_va,
      .size = 4096,
      .refcnt = 1,
   };

   struct hk_shader *hw_vs = &f->vs_api->variants[HK_VS_VARIANT_HW];
   struct hk_shader *hw_fs = &f->fs_api->variants[0];
   const struct hk_shader saved_vs = *hw_vs;
   const struct hk_shader saved_fs = *hw_fs;

   hw_vs->info.stage = MESA_SHADER_VERTEX;
   hw_vs->b.info.stage = MESA_SHADER_VERTEX;
   hw_vs->info.agx3_uses_root = true;
   hw_vs->info.agx3_root_uniform = 8;
   hw_vs->info.set_count = 1;
   hw_vs->agx3_vdm_state_update.source_layout =
      AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT;

   hw_fs->info.stage = MESA_SHADER_FRAGMENT;
   hw_fs->b.info.stage = MESA_SHADER_FRAGMENT;
   hw_fs->info.agx3_uses_root = true;
   hw_fs->info.agx3_root_uniform = 8;
   hw_fs->info.set_count = 1;

   struct hk_linked_shader linked_vs = {0};
   linked_vs.b.bo = &shader_bo;
   linked_vs.b.entry_offset = 0x80;
   linked_vs.b.size = 256;
   linked_vs.b.nr_gprs = 16;
   linked_vs.b.g17p_compiler_d_valid = true;
   linked_vs.b.g17p_compiler_d = 16;

   struct hk_linked_shader linked_fs = {0};
   linked_fs.b.bo = &shader_bo;
   linked_fs.b.entry_offset = 0x280;
   linked_fs.b.size = 256;
   linked_fs.b.nr_gprs = 16;
   linked_fs.b.g17p_compiler_d_valid = true;
   linked_fs.b.g17p_compiler_d = 16;
   linked_fs.b.g17p_colour_classes = 1;

   struct vk_instance inst = {0};
   list_inithead(&inst.debug_utils.callbacks);
   list_inithead(&inst.debug_utils.instance_callbacks);
   mtx_init(&inst.debug_utils.callbacks_mutex, mtx_plain);
   struct vk_physical_device phys = {.instance = &inst};
   f->dev.vk.physical = &phys;
   f->dev.dev.shader_base = shader_base;
   util_dynarray_init(&cmd->large_bos, NULL);

   const enum agx_chip chips[] = {
      AGX_CHIP_G16G, AGX_CHIP_G17P, AGX_CHIP_G17G, AGX_CHIP_G17S,
      AGX_CHIP_G17C,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(chips); ++i) {
      f->dev.dev.chip = chips[i];
      reset(f);
      cmd->vk.record_result = VK_SUCCESS;
      cmd->ioto = (struct hk_ioto_arena){
         .bo = &arena_bo,
         .map = ioto_map,
         .base = arena_va.addr,
         .capacity = ioto_capacity,
      };
      cmd->state.gfx.root = 0x800000;
      cmd->state.gfx.render.g17p_rt0_layout_supported = true;
      cmd->state.gfx.render.tilebuffer = (struct agx_tilebuffer_layout){
         .tile_size = 32 * 32,
         .sample_size_B = 8,
         .nr_samples = 1,
         .max_sample_size_B = 64,
         .max_samples_per_tile = 2048,
         .max_bytes_per_tile = 32768,
      };

      /* VS: identical draw params keep loader_program stable across draws
       * with zero new Ioto emissions; changing draw params replays via the
       * recorded launch and cached loader without creating a second loader
       * (root at 0x800000 and draw_params at 0x100000 have separate bases). */
      hk_cmd_upload_draw_params(cmd, 7, 3);
      const uint64_t vs_prog1 = hk_upload_usc_words(cmd, hw_vs, &linked_vs);
      CHECK(vs_prog1 != 0 && cmd->vk.record_result == VK_SUCCESS);
      CHECK(cmd->ioto.programs == 1);
      CHECK(cmd->state.gfx.g17p_ioto[0].admit.s == hw_vs);
      CHECK(cmd->state.gfx.g17p_ioto[0].admit.linked == &linked_vs);
      const struct hk_g17p_gfx_loader *vs_loader =
         cmd->state.gfx.g17p_ioto[0].loader;
      CHECK(vs_loader != NULL);

      hk_cmd_upload_draw_params(cmd, 7, 3);
      const uint64_t vs_prog2 = hk_upload_usc_words(cmd, hw_vs, &linked_vs);
      CHECK(vs_prog2 == vs_prog1);
      CHECK(cmd->ioto.programs == 1);

      hk_cmd_upload_draw_params(cmd, 9, 4);
      const uint64_t vs_prog3 = hk_upload_usc_words(cmd, hw_vs, &linked_vs);
      CHECK(vs_prog3 != 0 && vs_prog3 != vs_prog1);
      CHECK(cmd->ioto.programs == 2);
      CHECK(cmd->state.gfx.g17p_ioto[0].loader == vs_loader);

      /* Adding identical draw ID reuses draw_id_ptr and keeps loader_program
       * stable across consecutive draws. */
      hk_cmd_upload_draw_id(cmd, 5);
      const uint64_t vs_id1 = hk_upload_usc_words(cmd, hw_vs, &linked_vs);
      CHECK(vs_id1 != 0 && cmd->ioto.programs == 3);
      hk_cmd_upload_draw_id(cmd, 5);
      const uint64_t vs_id2 = hk_upload_usc_words(cmd, hw_vs, &linked_vs);
      CHECK(vs_id2 == vs_id1 && cmd->ioto.programs == 3);

      /* Evicting the launch slot still hits cached->admit on every chip. */
      const struct hk_g17p_gfx_loader *vs_id_loader =
         cmd->state.gfx.g17p_ioto[0].loader;
      for (unsigned s = 0; s < HK_G17P_LAUNCH_SLOTS; ++s)
         cmd->state.gfx.g17p_launch[0][s].valid = false;
      const uint64_t vs_prog4 = hk_upload_usc_words(cmd, hw_vs, &linked_vs);
      CHECK(vs_prog4 == vs_id1);
      CHECK(cmd->ioto.programs == 3);
      CHECK(cmd->state.gfx.g17p_ioto[0].loader == vs_id_loader);

      /* FS with mesh shader bound (no vertex shader): populates launch->hw_vs
       * and cached->admit.hw_vs and replays cleanly. */
      cmd->state.gfx.shaders[MESA_SHADER_VERTEX] = NULL;
      cmd->state.gfx.shaders[MESA_SHADER_MESH] = mesh_api;
      const uint64_t fs_mesh1 = hk_upload_usc_words(cmd, hw_fs, &linked_fs);
      CHECK(fs_mesh1 != 0 && cmd->vk.record_result == VK_SUCCESS);
      CHECK(cmd->state.gfx.g17p_ioto[1].admit.hw_vs ==
            &mesh_api->variants[HK_MESH_VARIANT_MAIN]);
      const uint32_t progs_after_mesh1 = cmd->ioto.programs;
      const uint64_t fs_mesh2 = hk_upload_usc_words(cmd, hw_fs, &linked_fs);
      CHECK(fs_mesh2 == fs_mesh1);
      CHECK(cmd->ioto.programs == progs_after_mesh1);

      /* M5 MSAA/MRT layout where g17p_rt0_layout_supported is false: M5 admits,
       * caches admission, and hits launch replay; HAL200 stays refused. */
      reset(f);
      cmd->vk.record_result = VK_SUCCESS;
      cmd->ioto = (struct hk_ioto_arena){
         .bo = &arena_bo,
         .map = ioto_map,
         .base = arena_va.addr,
         .capacity = ioto_capacity,
      };
      cmd->state.gfx.root = 0x800000;
      cmd->state.gfx.render.g17p_rt0_layout_supported = false;
      cmd->state.gfx.render.tilebuffer = (struct agx_tilebuffer_layout){
         .tile_size = 32 * 16,
         .sample_size_B = 16,
         .nr_samples = 4,
         .max_sample_size_B = 64,
         .max_samples_per_tile = 2048,
         .max_bytes_per_tile = 32768,
      };
      const uint64_t msaa1 = hk_upload_usc_words(cmd, hw_fs, &linked_fs);
      if (agx_chip_g17_m5(chips[i])) {
         CHECK(msaa1 != 0 && cmd->vk.record_result == VK_SUCCESS);
         CHECK(cmd->ioto.programs == 1);
         CHECK(cmd->state.gfx.g17p_ioto[1].admit.s == hw_fs);
         const struct hk_g17p_gfx_loader *msaa_loader =
            cmd->state.gfx.g17p_ioto[1].loader;
         CHECK(msaa_loader != NULL);
         const uint64_t msaa2 = hk_upload_usc_words(cmd, hw_fs, &linked_fs);
         CHECK(msaa2 == msaa1);
         CHECK(cmd->ioto.programs == 1);
         /* Moving the fragment root replays via the recorded launch + loader. */
         cmd->state.gfx.root = 0x801000;
         cmd->state.gfx.fs_root = 0;
         const uint64_t msaa3 = hk_upload_usc_words(cmd, hw_fs, &linked_fs);
         CHECK(msaa3 != 0 && msaa3 != msaa1);
         CHECK(cmd->ioto.programs == 2);
         CHECK(cmd->state.gfx.g17p_ioto[1].loader == msaa_loader);
      } else {
         CHECK(msaa1 == 0);
         CHECK(cmd->vk.record_result == VK_ERROR_FEATURE_NOT_PRESENT);
      }

      hk_g17p_linked_graphics_cache_finish(&f->dev, &linked_vs);
      hk_g17p_linked_graphics_cache_finish(&f->dev, &linked_fs);
   }

   /* Restore fixture state cleanly without invoking DRM BO free on stack BOs. */
   util_dynarray_fini(&cmd->large_bos);
   free(cmd->retained_set);
   cmd->retained_set = NULL;
   cmd->retained_set_capacity = 0;
   cmd->retained_set_count = 0;
   memset(cmd->recent_retained_bos, 0, sizeof(cmd->recent_retained_bos));
   memset(&cmd->ioto, 0, sizeof(cmd->ioto));
   *hw_vs = saved_vs;
   *hw_fs = saved_fs;
   f->dev.dev.chip = AGX_CHIP_G17P;
   f->dev.dev.shader_base = 0;
   f->dev.vk.physical = NULL;
   mtx_destroy(&inst.debug_utils.callbacks_mutex);
   cmd->vk.record_result = VK_SUCCESS;
   free(mesh_api);
   free(ioto_map);
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
   graphics_launch_replay_correctness(f);
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

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
#include "vk_common_entrypoints.h"
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

static void
vertex_input_elision_correctness(struct fixture *f)
{
   struct hk_cmd_buffer *cmd = &f->cmd;
   VkCommandBuffer vk_cb = hk_cmd_buffer_to_handle(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   reset(f);
   vk_dynamic_graphics_state_init(dyn);
   dyn->vi = &cmd->state.gfx._dynamic_vi;
   memset(dyn->vi, 0, sizeof(*dyn->vi));

   /* First-time empty call must not elide even though _dynamic_vi is zeroed. */
   vk_common_CmdSetVertexInputEXT(vk_cb, 0, NULL, 0, NULL);
   CHECK(BITSET_TEST(dyn->set, MESA_VK_DYNAMIC_VI));
   CHECK(BITSET_TEST(dyn->set, MESA_VK_DYNAMIC_VI_BINDINGS_VALID));
   CHECK(BITSET_TEST(dyn->set, MESA_VK_DYNAMIC_VI_BINDING_STRIDES));
   CHECK(BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI));
   CHECK(BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI_BINDING_STRIDES));
   CHECK(dyn->generation > 0);

   /* Identical empty call after clearing dirty must elide cleanly. */
   vk_dynamic_graphics_state_clear_dirty(dyn);
   BITSET_ZERO(dyn->changed_since_pipeline_bind);
   uint64_t gen = dyn->generation;
   vk_common_CmdSetVertexInputEXT(vk_cb, 0, NULL, 0, NULL);
   CHECK(dyn->generation == gen);
   CHECK(!BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI));
   CHECK(!BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI_BINDING_STRIDES));
   CHECK(!BITSET_TEST(dyn->changed_since_pipeline_bind, MESA_VK_DYNAMIC_VI));

   const VkVertexInputBindingDescription2EXT bindings[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
         .binding = 0,
         .stride = 16,
         .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
         .divisor = 1,
      },
      {
         .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
         .binding = 2,
         .stride = 32,
         .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE,
         .divisor = 4,
      },
   };
   const VkVertexInputAttributeDescription2EXT attrs[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
         .location = 0,
         .binding = 0,
         .format = VK_FORMAT_R32G32B32_SFLOAT,
         .offset = 0,
      },
      {
         .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
         .location = 3,
         .binding = 2,
         .format = VK_FORMAT_R32G32B32A32_SFLOAT,
         .offset = 16,
      },
   };

   vk_common_CmdSetVertexInputEXT(vk_cb, 2, bindings, 2, attrs);
   CHECK(dyn->generation > gen);
   CHECK(BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI));
   CHECK(BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI_BINDING_STRIDES));
   CHECK(dyn->vi->bindings_valid == (BITFIELD_BIT(0) | BITFIELD_BIT(2)));
   CHECK(dyn->vi->attributes_valid == (BITFIELD_BIT(0) | BITFIELD_BIT(3)));
   CHECK(dyn->vi_binding_strides[0] == 16 && dyn->vi_binding_strides[2] == 32);

   /* Identical state in reversed array order must also elide. */
   const VkVertexInputBindingDescription2EXT bindings_rev[2] = {
      bindings[1], bindings[0],
   };
   const VkVertexInputAttributeDescription2EXT attrs_rev[2] = {
      attrs[1], attrs[0],
   };
   vk_dynamic_graphics_state_clear_dirty(dyn);
   BITSET_ZERO(dyn->changed_since_pipeline_bind);
   gen = dyn->generation;
   vk_common_CmdSetVertexInputEXT(vk_cb, 2, bindings_rev, 2, attrs_rev);
   CHECK(dyn->generation == gen);
   CHECK(!BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI));
   CHECK(!BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI_BINDING_STRIDES));
   CHECK(!BITSET_TEST(dyn->changed_since_pipeline_bind, MESA_VK_DYNAMIC_VI));

   /* A dynamic stride override via CmdBindVertexBuffers2 must not be elided
    * when CmdSetVertexInputEXT restores the original stride. */
   const VkDeviceSize new_stride = 64;
   vk_cmd_set_vertex_binding_strides(&cmd->vk, 0, 1, &new_stride);
   CHECK(dyn->vi_binding_strides[0] == 64);
   vk_dynamic_graphics_state_clear_dirty(dyn);
   gen = dyn->generation;
   vk_common_CmdSetVertexInputEXT(vk_cb, 2, bindings, 2, attrs);
   CHECK(dyn->generation > gen);
   CHECK(dyn->vi_binding_strides[0] == 16);
   CHECK(BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI_BINDING_STRIDES));

   /* Dirty VI state (e.g. after dirty_all) must not elide even if equal. */
   vk_dynamic_graphics_state_dirty_all(dyn);
   gen = dyn->generation;
   vk_common_CmdSetVertexInputEXT(vk_cb, 2, bindings, 2, attrs);
   CHECK(dyn->generation > gen);

   /* Differing divisor, attribute offset, or binding count must invalidate. */
   vk_dynamic_graphics_state_clear_dirty(dyn);
   gen = dyn->generation;
   VkVertexInputBindingDescription2EXT diff_bindings[2] = {
      bindings[0], bindings[1],
   };
   diff_bindings[1].divisor = 2;
   vk_common_CmdSetVertexInputEXT(vk_cb, 2, diff_bindings, 2, attrs);
   CHECK(dyn->generation > gen && dyn->vi->bindings[2].divisor == 2);

   vk_dynamic_graphics_state_clear_dirty(dyn);
   gen = dyn->generation;
   VkVertexInputAttributeDescription2EXT diff_attrs[2] = {
      attrs[0], attrs[1],
   };
   diff_attrs[1].offset = 8;
   vk_common_CmdSetVertexInputEXT(vk_cb, 2, diff_bindings, 2, diff_attrs);
   CHECK(dyn->generation > gen && dyn->vi->attributes[3].offset == 8);

   vk_dynamic_graphics_state_clear_dirty(dyn);
   gen = dyn->generation;
   vk_common_CmdSetVertexInputEXT(vk_cb, 1, diff_bindings, 1, diff_attrs);
   CHECK(dyn->generation > gen);
   CHECK(dyn->vi->bindings_valid == BITFIELD_BIT(0));
   CHECK(dyn->vi->attributes_valid == BITFIELD_BIT(0));

   /* Static->dynamic VI pipeline transition: binding a static-VI pipeline
    * copies VI state into dyn and marks it dirty. Before a draw clears dirty,
    * transitioning to a dynamic-VI pipeline and calling CmdSetVertexInputEXT
    * with matching state must not elide. Once dirty is cleared, calling
    * CmdSetVertexInputEXT with the static pipeline's exact VI elides, while
    * differing VI updates and marks dirty. */
   struct vk_vertex_input_state static_vi;
   memset(&static_vi, 0, sizeof(static_vi));
   static_vi.bindings_valid = BITFIELD_BIT(0) | BITFIELD_BIT(2);
   static_vi.bindings[0] = (struct vk_vertex_binding_state){
      .stride = 16,
      .input_rate = VK_VERTEX_INPUT_RATE_VERTEX,
      .divisor = 1,
   };
   static_vi.bindings[2] = (struct vk_vertex_binding_state){
      .stride = 32,
      .input_rate = VK_VERTEX_INPUT_RATE_INSTANCE,
      .divisor = 4,
   };
   static_vi.attributes_valid = BITFIELD_BIT(0) | BITFIELD_BIT(3);
   static_vi.attributes[0] = (struct vk_vertex_attribute_state){
      .binding = 0,
      .format = VK_FORMAT_R32G32B32_SFLOAT,
      .offset = 0,
   };
   static_vi.attributes[3] = (struct vk_vertex_attribute_state){
      .binding = 2,
      .format = VK_FORMAT_R32G32B32A32_SFLOAT,
      .offset = 16,
   };

   struct vk_dynamic_graphics_state static_pipe_dyn;
   vk_dynamic_graphics_state_init(&static_pipe_dyn);
   static_pipe_dyn.vi = &static_vi;
   static_pipe_dyn.vi_bindings_valid = static_vi.bindings_valid;
   static_pipe_dyn.vi_binding_strides[0] = 16;
   static_pipe_dyn.vi_binding_strides[2] = 32;
   BITSET_SET(static_pipe_dyn.set, MESA_VK_DYNAMIC_VI);
   BITSET_SET(static_pipe_dyn.set, MESA_VK_DYNAMIC_VI_BINDINGS_VALID);
   BITSET_SET(static_pipe_dyn.set, MESA_VK_DYNAMIC_VI_BINDING_STRIDES);

   vk_dynamic_graphics_state_clear_dirty(dyn);
   vk_cmd_set_dynamic_graphics_state(&cmd->vk, &static_pipe_dyn);
   CHECK(BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI));

   /* Still dirty from static pipeline bind -> must not elide. */
   gen = dyn->generation;
   vk_common_CmdSetVertexInputEXT(vk_cb, 2, bindings, 2, attrs);
   CHECK(dyn->generation > gen);

   /* After draw clears dirty and a dynamic-VI pipeline is bound (leaves VI
    * untouched in dyn), matching CmdSetVertexInputEXT elides cleanly. */
   vk_dynamic_graphics_state_clear_dirty(dyn);
   BITSET_ZERO(dyn->changed_since_pipeline_bind);
   struct vk_dynamic_graphics_state dyn_pipe_dyn;
   vk_dynamic_graphics_state_init(&dyn_pipe_dyn);
   vk_cmd_set_dynamic_graphics_state(&cmd->vk, &dyn_pipe_dyn);
   gen = dyn->generation;
   vk_common_CmdSetVertexInputEXT(vk_cb, 2, bindings, 2, attrs);
   CHECK(dyn->generation == gen);
   CHECK(!BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI));
   CHECK(!BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI_BINDING_STRIDES));
   CHECK(!BITSET_TEST(dyn->changed_since_pipeline_bind, MESA_VK_DYNAMIC_VI));

   /* Static pipeline with a different stride followed by dynamic-VI pipeline
    * must still invalidate when CmdSetVertexInputEXT restores bindings[]. */
   static_vi.bindings[0].stride = 48;
   static_pipe_dyn.vi_binding_strides[0] = 48;
   vk_cmd_set_dynamic_graphics_state(&cmd->vk, &static_pipe_dyn);
   vk_dynamic_graphics_state_clear_dirty(dyn);
   vk_cmd_set_dynamic_graphics_state(&cmd->vk, &dyn_pipe_dyn);
   gen = dyn->generation;
   vk_common_CmdSetVertexInputEXT(vk_cb, 2, bindings, 2, attrs);
   CHECK(dyn->generation > gen);
   CHECK(dyn->vi->bindings[0].stride == 16 && dyn->vi_binding_strides[0] == 16);
   CHECK(BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI));
   CHECK(BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_VI_BINDING_STRIDES));

   reset(f);
}

static void
xfer_clean_isolation_correctness(struct fixture *f)
{
   struct hk_cmd_buffer *cmd = &f->cmd;
   VkCommandBuffer vk_cb = hk_cmd_buffer_to_handle(cmd);

   reset(f);
   memset(&cmd->sync, 0, sizeof(cmd->sync));

   /* Build a Gen9 UBO descriptor set at a non-overlapping GPU VA. */
   union {
      struct hk_descriptor_set_layout layout;
      uint8_t storage[sizeof(struct hk_descriptor_set_layout) +
                      sizeof(struct hk_descriptor_set_binding_layout)];
   } ubo_layout_mem;
   memset(&ubo_layout_mem, 0, sizeof(ubo_layout_mem));
   struct hk_descriptor_set_layout *ubo_layout = &ubo_layout_mem.layout;
   ubo_layout->vk.base.type = VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT;
   ubo_layout->descriptor_generation = 9;
   ubo_layout->binding_count = 1;
   ubo_layout->binding[0] = (struct hk_descriptor_set_binding_layout){
      .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .array_size = 1,
      .offset = 0,
      .stride = sizeof(struct hk_buffer_address),
   };

   struct hk_buffer_address ubo_desc = {
      .base_addr = 0x500000,
      .size = 4096,
   };
   struct hk_descriptor_set ubo_set = {
      .base.type = VK_OBJECT_TYPE_DESCRIPTOR_SET,
      .layout = ubo_layout,
      .mapped_ptr = &ubo_desc,
      .addr = 0x380000,
      .size = sizeof(ubo_desc),
   };
   struct vk_pipeline_layout pipe_layout = {
      .base.type = VK_OBJECT_TYPE_PIPELINE_LAYOUT,
      .set_layouts = { &ubo_layout->vk },
   };
   VkDescriptorSet vk_set = hk_descriptor_set_to_handle(&ubo_set);
   VkBindDescriptorSetsInfoKHR bind_info = {
      .sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO_KHR,
      .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
      .layout = vk_pipeline_layout_to_handle(&pipe_layout),
      .descriptorSetCount = 1,
      .pDescriptorSets = &vk_set,
   };
   hk_CmdBindDescriptorSets2KHR(vk_cb, &bind_info);

   /* Bind non-overlapping VB 0 and index buffer. */
   hk_cmd_bind_vertex_buffer(cmd, 0, (struct hk_addr_range){
      .addr = 0x600000, .range = 4096,
   });
   CHECK(cmd->state.gfx.bound_vb_mask == BITFIELD_BIT(0));
   cmd->state.gfx.index.buffer = (struct hk_addr_range){
      .addr = 0x700000, .range = 1024,
   };

   /* Note a pending staging transfer to a disjoint range [0x900000, 0x901000). */
   struct hk_cs xfer_cs = {
      .type = HK_CS_CDM,
      .sync.seq = 1,
   };
   cmd->sync.xfer_dst = (struct hk_sync_xfer_dst){
      .lo = 0x900000,
      .hi = 0x901000,
      .valid = true,
      .device_address = false,
   };
   hk_sync_note_xfer_write(cmd, &xfer_cs);
   CHECK(cmd->sync.xfer_write_count == 1);

   /* First draw populates all three xfer_clean slices with one descriptor walk. */
   hk_xfer_check_draw_slow(cmd, 0, 0);
   CHECK(cmd->state.upload_stats.xfer_desc_walks == 1);
   CHECK(cmd->sync.xfer_write_count == 1);
   CHECK(cmd->sync.xfer_clean.desc_valid);
   CHECK(cmd->sync.xfer_clean.vb_valid);
   CHECK(cmd->sync.xfer_clean.index_valid);

   /* Push-constant loop: generation advances, non_push_generation does not,
    * and hk_xfer_match_descriptors() must NOT be re-walked. */
   const uint64_t rg_before = cmd->state.gfx.descriptors.non_push_generation;
   const uint64_t gen_before = cmd->state.gfx.descriptors.generation;
   for (uint32_t i = 1; i <= 8; ++i) {
      VkPushConstantsInfoKHR pc = {
         .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO_KHR,
         .layout = vk_pipeline_layout_to_handle(&pipe_layout),
         .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
         .offset = 0,
         .size = sizeof(i),
         .pValues = &i,
      };
      hk_CmdPushConstants2KHR(vk_cb, &pc);
      hk_xfer_check_draw_slow(cmd, 0, 0);
   }
   CHECK(cmd->state.gfx.descriptors.generation == gen_before + 8);
   CHECK(cmd->state.gfx.descriptors.non_push_generation == rg_before);
   CHECK(cmd->state.upload_stats.xfer_desc_walks == 1);
   CHECK(cmd->sync.xfer_write_count == 1);

   /* Sparse VB changes must re-check only bound VBs and NEVER re-walk
    * descriptor sets. Verify bound_vb_mask sets and clears properly. */
   for (uint32_t i = 0; i < 4; ++i) {
      hk_cmd_bind_vertex_buffer(cmd, 17, (struct hk_addr_range){
         .addr = 0x620000 + i * 4096, .range = 4096,
      });
      CHECK(cmd->state.gfx.bound_vb_mask ==
            (BITFIELD_BIT(0) | BITFIELD_BIT(17)));
      hk_xfer_check_draw_slow(cmd, 0, 0);
   }
   hk_cmd_bind_vertex_buffer(cmd, 17, (struct hk_addr_range){
      .addr = 0, .range = 0,
   });
   CHECK(cmd->state.gfx.bound_vb_mask == BITFIELD_BIT(0));
   hk_xfer_check_draw_slow(cmd, 0, 0);
   CHECK(cmd->state.upload_stats.xfer_desc_walks == 1);
   CHECK(cmd->sync.xfer_write_count == 1);

   /* Index buffer change to a non-overlapping range does not re-walk descriptors,
    * while expanding range at the same base to overlap the transfer MUST order. */
   cmd->state.gfx.index.buffer = (struct hk_addr_range){
      .addr = 0x8ff000, .range = 0x1000,
   };
   hk_xfer_check_draw_slow(cmd, 0, 0);
   CHECK(cmd->state.upload_stats.xfer_desc_walks == 1);
   CHECK(cmd->sync.xfer_write_count == 1);

   cmd->state.gfx.index.buffer.range = 0x2000;
   const uint32_t ver_before_order = cmd->sync.xfer_version;
   hk_xfer_check_draw_slow(cmd, 0, 0);
   CHECK(cmd->state.upload_stats.xfer_desc_walks == 1);
   CHECK(cmd->sync.xfer_write_count == 0);
   CHECK(cmd->sync.xfer_version > ver_before_order);

   /* A pending transfer overlapping a high sparse VB slot must order properly. */
   struct hk_cs xfer_cs2 = {
      .type = HK_CS_CDM,
      .sync.seq = 2,
   };
   cmd->sync.xfer_dst = (struct hk_sync_xfer_dst){
      .lo = 0xa00000,
      .hi = 0xa01000,
      .valid = true,
      .device_address = false,
   };
   hk_sync_note_xfer_write(cmd, &xfer_cs2);
   hk_xfer_check_draw_slow(cmd, 0, 0);
   CHECK(cmd->state.upload_stats.xfer_desc_walks == 2);
   CHECK(cmd->sync.xfer_write_count == 1);

   hk_cmd_bind_vertex_buffer(cmd, 19, (struct hk_addr_range){
      .addr = 0xa00800, .range = 256,
   });
   CHECK(cmd->state.gfx.bound_vb_mask ==
         (BITFIELD_BIT(0) | BITFIELD_BIT(19)));
   hk_xfer_check_draw_slow(cmd, 0, 0);
   CHECK(cmd->state.upload_stats.xfer_desc_walks == 2);
   CHECK(cmd->sync.xfer_write_count == 0);

   /* BDA transfers set sync.xfer_any and must order even after push-constant
    * updates with unchanged descriptor/VB/index slices. */
   struct hk_cs xfer_cs3 = {
      .type = HK_CS_CDM,
      .sync.seq = 3,
   };
   cmd->sync.xfer_dst = (struct hk_sync_xfer_dst){
      .lo = 0xb00000,
      .hi = 0xb01000,
      .valid = true,
      .device_address = true,
   };
   hk_sync_note_xfer_write(cmd, &xfer_cs3);
   CHECK(cmd->sync.xfer_any == &xfer_cs3);
   hk_xfer_check_draw_slow(cmd, 0, 0);
   CHECK(cmd->sync.xfer_any == NULL);

   /* Secondary-CB graphics state invalidation clears bound_vb_mask and
    * invalidates xfer_clean so generation resets cannot falsely match. */
   struct hk_cs xfer_cs4 = {
      .type = HK_CS_CDM,
      .sync.seq = 4,
   };
   cmd->sync.xfer_dst = (struct hk_sync_xfer_dst){
      .lo = 0xc00000,
      .hi = 0xc01000,
      .valid = true,
      .device_address = false,
   };
   hk_sync_note_xfer_write(cmd, &xfer_cs4);
   hk_xfer_check_draw_slow(cmd, 0, 0);
   CHECK(cmd->sync.xfer_clean.desc_valid && cmd->sync.xfer_clean.vb_valid);
   hk_cmd_invalidate_graphics_state(cmd);
   CHECK(cmd->state.gfx.bound_vb_mask == 0);
   CHECK(!cmd->sync.xfer_clean.desc_valid);
   CHECK(!cmd->sync.xfer_clean.vb_valid);
   CHECK(!cmd->sync.xfer_clean.index_valid);

   memset(&cmd->sync, 0, sizeof(cmd->sync));
   reset(f);
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
   vertex_input_elision_correctness(f);
   xfer_clean_isolation_correctness(f);
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

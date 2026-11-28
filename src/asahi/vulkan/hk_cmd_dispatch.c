/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include <math.h>
#include "libagx/query.h"
#include "vulkan/vulkan_core.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_pack.h"
#include "agx_scratch.h"
#include "agx_tilebuffer.h"
#include "hk_buffer.h"
#include "hk_2d_stencil.h"
#include "hk_byte_stencil.h"
#include "hk_compute_opt.h"
#include "hk_acceleration_structure.h"
#include "hk_cmd_buffer.h"
#include "hk_compute_pair.h"
#include "hk_nir_match_pointwise_clamp.h"
#include "hk_nir_root_size.h"
#include "hk_nir_wgid.h"
#include "hk_pack.h"
#include "hk_g17p_storage_image.h"
#include "hk_frame_anatomy.h"
#include "hk_sampler_index.h"
#include "hk_g17p_patch.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_gen1_rt_indirect_contract.h"
#include "hk_entrypoints.h"
#include "hk_agx3_ioto_program_core.h"
#include "asahi/compiler/agx3_compute_program.h"
#include "asahi/compiler/agx3_graphics_program.h"
#include "asahi/compiler/agx3_transcode.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx3_g17p_grid_helper_core.h"
#include "asahi/lib/agx_chip_caps.h"
#include "asahi/lib/agx_scs_layout.h"
#include "asahi/lib/agx_rt_compute_gen1.h"
#include "util/log.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"
#include "pool.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

/* The G18G image-write state record (AGX_G18G_WRITE_STATE1_INIT). */
static const uint8_t hk_g18g_write_state1[32] = AGX_G18G_WRITE_STATE1_INIT;

/* G18G: most dependent compute launches that share one CDM stream (one
 * submit, one fence; hk_dispatch_g17g_native).
 *
 * 64 is where decode throughput stops rising, measured on one G18G with
 * llama-bench (7B Q4_0, -fa off -p 0 -n 32 -r 3) in 07a390b08a9: 1.83
 * (batch 8), 2.76 (16), 3.14 (32), 3.44 (64), 3.43 (128) and 3.40 (256)
 * tokens per second, with identical generated text and no G18 failure
 * lines. Prefill (pp512, pp2048 at -ub 512 and 2048) does not regress at
 * 64. The earlier ceiling of 8 (ba6b26abb75) came from a firmware stall
 * above 9 launches per stream on large flat grids (G18_RECOVERY_PB
 * restart loop -> G18_APP_JOB_FAILED EIO); it did not reproduce at batch
 * 10 to 256 on decode or at 64 on prefill, but its cause is not known.
 */
#define HK_G18_BATCH_LAUNCHES 64

/* The stream must stay under the kernel's g18_compute_stream_max (64 KiB
 * with the G18 admission patch), which ba6b26abb75 put at 1400 launches. */
static_assert(HK_G18_BATCH_LAUNCHES <= 1400,
              "G18G CDM stream exceeds the kernel's stream size limit");

/* EXP-G1b's native access::write argument is a PBE pointer followed by a
 * pointer to this control descriptor, including when it follows a sampled
 * texture in the argument table. It is not the read_write clamp scalar. */
static const uint8_t hk_g17p_native_write_control[32] = {
   0x08, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
};

/* Root, texture pointers, promoted constants and the ray-query range, plus
 * the uniform DMA spans of an application kernel (each at most a buffer part
 * and a zero-page tail; hk_g17p_dma_ranges()). */
#define HK_G17P_COMPUTE_MAX_RANGES (4u + 2u * AGX3_UNIFORM_DMA_MAX_SPANS)
#define HK_G17P_LOADER_MAX_SHAPES  4u
static_assert(HK_G17P_COMPUTE_MAX_RANGES <= HK_G17P_PATCH_MAX_RANGES,
              "patch plan capacity");

/* The uniform loader of a G17P launch, composed once per program, uniform
 * layout and address shape (hk_g17p_patch.h); each launch copies it with its
 * own range base immediates. */
struct hk_g17p_loader {
   struct hk_g17p_loader *next;

   uint64_t shader_va;
   uint64_t constant_program_va;
   const struct agx3_cprog_inputs *cprog_inputs;
   struct agx3_cdm_direct_template_sources launch_sources;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint32_t compiler_d;
   uint32_t threadgroup_memory_size_B;
   uint16_t imageblock_stride_B;
   uint32_t constant_program_size;
   uint16_t nr_gprs;
   bool compiler_d_valid;
   bool constant_program_prelude;
   enum agx_ioto_g17p_compute_ldshdr_f1 kind;
   uint8_t range_count;
   uint16_t range_start[HK_G17P_COMPUTE_MAX_RANGES];
   uint16_t range_size[HK_G17P_COMPUTE_MAX_RANGES];

   struct agx3_compute_program launch;
   uint16_t length;
   struct hk_g17p_patch_plan plan;
   uint8_t bytes[];
};

struct hk_m5_compute_loader {
   struct hk_m5_compute_loader *next;
   struct agx3_g17p_graphics_program_desc desc;
   struct agx3_ioto_uniform_range ranges[HK_G17P_COMPUTE_MAX_RANGES];
   struct hk_g17p_patch_plan plan;
   uint16_t length;
   uint8_t bytes[];
};

static void
hk_m5_loader_list_free(void **head)
{
   struct hk_m5_compute_loader *l = p_atomic_read(head);
   while (l) {
      struct hk_m5_compute_loader *next = l->next;
      free(l);
      l = next;
   }
   *head = NULL;
}

static void
hk_m5_compute_program_desc_set_device(
   const struct hk_device *dev,
   struct agx3_g17p_graphics_program_desc *desc)
{
   desc->usc_clusters = dev->dev.params.num_clusters_total;
}

static bool
hk_m5_compute_program_generate(const struct hk_device *dev,
                               struct agx3_g17p_graphics_program_desc *desc,
                               struct agx3_g17p_graphics_program *program)
{
   hk_m5_compute_program_desc_set_device(dev, desc);
   return agx3_g17g_compute_program_generate(dev->dev.chip, desc, program);
}

static bool
hk_m5_compute_launch_bind(const struct hk_device *dev,
                          struct agx3_g17p_native_compute_program_desc *desc,
                          struct agx3_compute_program *launch)
{
   desc->ordering_memory_scope = agx_cdm_ordering_memory_scope(&dev->dev);
   return agx3_m5_native_compute_program_bind(dev->dev.chip, desc, launch);
}

static bool
hk_m5_loader_matches(const struct hk_m5_compute_loader *l,
                      const struct agx3_g17p_graphics_program_desc *d)
{
   const struct agx3_g17p_graphics_program_desc *a = &l->desc;
   if (a->shader_va != d->shader_va ||
       a->shader_binary_size != d->shader_binary_size ||
       a->scratch_size != d->scratch_size || a->nr_gprs != d->nr_gprs ||
       a->compiler_d_valid != d->compiler_d_valid || a->compiler_d != d->compiler_d ||
       a->threadgroup_memory_size_B != d->threadgroup_memory_size_B ||
       a->uniform_range_count != d->uniform_range_count ||
       d->fragment || d->writes_sample_mask || d->colour_classes ||
       d->preloaded_root || d->patchable ||
       a->constant_program.code_va != d->constant_program.code_va ||
       a->constant_program.code_size != d->constant_program.code_size ||
       a->constant_program.request_prelude != d->constant_program.request_prelude ||
       a->constant_program.absolute_branch != d->constant_program.absolute_branch ||
       a->cprog_inputs != d->cprog_inputs)
      return false;
   for (unsigned i = 0; i < d->uniform_range_count; ++i)
      if (l->ranges[i].start_halfs != d->uniform_ranges[i].start_halfs ||
          l->ranges[i].size_halfs != d->uniform_ranges[i].size_halfs)
         return false;
   return true;
}

static struct hk_m5_compute_loader *
hk_m5_loader_create(struct hk_device *dev,
                     const struct agx3_g17p_graphics_program_desc *d)
{
   if (d->uniform_range_count > HK_G17P_COMPUTE_MAX_RANGES)
      return NULL;
   struct agx3_g17p_graphics_program_template *t = calloc(1, sizeof(*t));
   if (!t)
      return NULL;
   struct agx3_g17p_graphics_program_desc desc = *d;
   hk_m5_compute_program_desc_set_device(dev, &desc);
   struct hk_m5_compute_loader *l = NULL;
   if (!agx3_g17g_compute_program_template_generate(dev->dev.chip, &desc, t))
      goto out;
   l = calloc(1, sizeof(*l) + t->program.length);
   if (!l)
      goto out;
   if (!hk_g17p_patch_plan_init(&l->plan, desc.uniform_ranges,
                                desc.uniform_range_count, t->relocations,
                                t->relocation_count, t->program.length, false)) {
      free(l);
      l = NULL;
      goto out;
   }
   l->desc = desc;
   l->desc.uniform_ranges = l->ranges;
   memcpy(l->ranges, desc.uniform_ranges,
          desc.uniform_range_count * sizeof(l->ranges[0]));
   l->length = t->program.length;
   memcpy(l->bytes, t->program.bytes, l->length);
out:
   free(t);
   return l;
}

static const struct hk_m5_compute_loader *
hk_m5_loader_get(struct hk_device *dev, struct hk_linked_shader *linked,
                  const struct agx3_g17p_graphics_program_desc *d)
{
   unsigned shapes = 0;
   for (struct hk_m5_compute_loader *l =
           p_atomic_read(&linked->m5_compute_templates); l; l = l->next) {
      if (hk_m5_loader_matches(l, d)) {
         if (hk_g17p_patch_plan_fits(&l->plan, d->uniform_ranges))
            return l;
         ++shapes;
      }
   }
   /* Preserve the normal load schedule. Unusual changing address layouts
    * fall back to fresh composition rather than lengthening GPU programs. */
   if (shapes >= HK_G17P_LOADER_MAX_SHAPES)
      return NULL;
   struct hk_m5_compute_loader *created = hk_m5_loader_create(dev, d);
   if (!created)
      return NULL;
   while (true) {
      struct hk_m5_compute_loader *first =
         p_atomic_read(&linked->m5_compute_templates);
      for (struct hk_m5_compute_loader *l = first; l; l = l->next) {
         if (hk_m5_loader_matches(l, d) &&
             hk_g17p_patch_plan_fits(&l->plan, d->uniform_ranges)) {
            free(created);
            return l;
         }
      }
      created->next = first;
      if (p_atomic_cmpxchg_ptr(&linked->m5_compute_templates, first, created) == first)
         return created;
   }
}

/* Hot recording needs only the published loader bytes for tracing and the
 * compact launch record. Keep the maximum-size composer object off the draw
 * stack; one such frame crosses a host page boundary and faults on the first
 * software tessellation/geometry draw. */
struct hk_g17p_compute_program_ref {
   const uint8_t *bytes;
   uint16_t length;
   struct agx3_compute_program launch;
};

static void
hk_g17p_loader_list_free(void **head)
{
   struct hk_g17p_loader *l = p_atomic_read(head);
   while (l) {
      struct hk_g17p_loader *next = l->next;
      free(l);
      l = next;
   }
   *head = NULL;
}

void
hk_g17p_compute_program_cache_finish(struct hk_device *dev)
{
   for (unsigned i = 0; i < LIBAGX_NUM_PROGRAMS; ++i)
      hk_g17p_loader_list_free(&dev->g17p_compute_program_cache.templates[i]);
}

void
hk_g17p_linked_compute_cache_finish(struct hk_linked_shader *linked)
{
   hk_g17p_loader_list_free(&linked->g17p_compute_templates);
   hk_m5_loader_list_free(&linked->m5_compute_templates);
}

static bool
hk_g17p_loader_matches(const struct hk_g17p_loader *l,
                       const struct agx3_g17p_compute_program_desc *d)
{
   if (l->shader_va != d->shader_va || l->kind != d->ldshdr_f1 ||
       l->range_count != d->uniform_range_count ||
       l->shader_binary_size != d->shader_binary_size ||
       l->scratch_size != d->scratch_size || l->nr_gprs != d->nr_gprs ||
       l->compiler_d_valid != d->compiler_d_valid ||
       l->compiler_d != d->compiler_d ||
       l->threadgroup_memory_size_B != d->threadgroup_memory_size_B ||
       l->imageblock_stride_B != d->imageblock_stride_B ||
       l->launch_sources.ti_spill_bytes != d->launch_sources.ti_spill_bytes ||
       l->launch_sources.ti_largest_ipr_bytes !=
          d->launch_sources.ti_largest_ipr_bytes ||
       l->launch_sources.ray_traversal_depth_nonzero !=
          d->launch_sources.ray_traversal_depth_nonzero ||
       l->constant_program_va != d->constant_program.code_va ||
       l->constant_program_size != d->constant_program.code_size ||
       l->constant_program_prelude != d->constant_program.request_prelude ||
       l->cprog_inputs != d->cprog_inputs)
      return false;

   for (unsigned i = 0; i < l->range_count; ++i) {
      if (l->range_start[i] != d->uniform_ranges[i].start_halfs ||
          l->range_size[i] != d->uniform_ranges[i].size_halfs)
         return false;
   }
   return true;
}

static struct hk_g17p_loader *
hk_g17p_loader_create(struct hk_device *dev,
                      const struct agx3_g17p_compute_program_desc *desc,
                      bool independent)
{
   if (desc->uniform_range_count > HK_G17P_COMPUTE_MAX_RANGES)
      return NULL;

   struct agx_ioto_load_relocation rels[HK_G17P_PATCH_MAX_RELOCS];
   size_t rel_count = 0;
   struct agx3_g17p_compute_program_desc composed = *desc;
   /* Any aligned address inside the USC window: the bytes do not depend on
    * it, and every launch binds its own copy. */
   composed.program_va = dev->dev.shader_base + 0x10000;
   composed.relocations = rels;
   composed.relocation_capacity = ARRAY_SIZE(rels);
   composed.relocation_count = &rel_count;
   composed.patchable = independent;

   /* The composer's output object is too large for the recording stack. */
   struct agx3_g17p_compute_program *program = malloc(sizeof(*program));
   if (!program)
      return NULL;

   struct hk_g17p_loader *l = NULL;
   if (!agx3_g17p_compute_program_generate(dev->dev.chip, &composed,
                                           program))
      goto out;

   l = calloc(1, sizeof(*l) + program->length);
   if (!l)
      goto out;

   *l = (struct hk_g17p_loader){
      .shader_va = desc->shader_va,
      .constant_program_va = desc->constant_program.code_va,
      .cprog_inputs = desc->cprog_inputs,
      .launch_sources = desc->launch_sources,
      .shader_binary_size = desc->shader_binary_size,
      .scratch_size = desc->scratch_size,
      .compiler_d = desc->compiler_d,
      .threadgroup_memory_size_B = desc->threadgroup_memory_size_B,
      .imageblock_stride_B = desc->imageblock_stride_B,
      .constant_program_size = desc->constant_program.code_size,
      .nr_gprs = desc->nr_gprs,
      .compiler_d_valid = desc->compiler_d_valid,
      .constant_program_prelude = desc->constant_program.request_prelude,
      .kind = desc->ldshdr_f1,
      .range_count = desc->uniform_range_count,
      .launch = program->launch,
      .length = program->length,
   };
   memcpy(l->bytes, program->bytes, program->length);
   for (unsigned i = 0; i < l->range_count; ++i) {
      l->range_start[i] = desc->uniform_ranges[i].start_halfs;
      l->range_size[i] = desc->uniform_ranges[i].size_halfs;
   }

   if (!hk_g17p_patch_plan_init(&l->plan, desc->uniform_ranges,
                                desc->uniform_range_count, rels, rel_count,
                                program->length, independent)) {
      free(l);
      l = NULL;
   }

out:
   free(program);
   return l;
}

/* The loader for desc's program and uniform layout, composed on first use.
 * Lists are shared by every command buffer of the device. */
static const struct hk_g17p_loader *
hk_g17p_loader_get(struct hk_device *dev, void **head,
                   const struct agx3_g17p_compute_program_desc *desc)
{
   unsigned shapes = 0;
   for (struct hk_g17p_loader *l = p_atomic_read(head); l; l = l->next) {
      if (!hk_g17p_loader_matches(l, desc))
         continue;
      if (hk_g17p_patch_plan_fits(&l->plan, desc->uniform_ranges))
         return l;
      ++shapes;
   }

   /* A layout whose ranges keep moving relative to each other gets one
    * program with a base per range, which fits any addresses. */
   struct hk_g17p_loader *created =
      hk_g17p_loader_create(dev, desc, shapes >= HK_G17P_LOADER_MAX_SHAPES);
   if (!created)
      return NULL;

   while (true) {
      struct hk_g17p_loader *first = p_atomic_read(head);
      for (struct hk_g17p_loader *l = first; l; l = l->next) {
         if (hk_g17p_loader_matches(l, desc) &&
             hk_g17p_patch_plan_fits(&l->plan, desc->uniform_ranges)) {
            free(created);
            return l;
         }
      }
      created->next = first;
      if (p_atomic_cmpxchg_ptr(head, first, created) == first)
         return created;
   }
}

/* Copy the loader into the command buffer with ranges' addresses. */
static bool
hk_g17p_loader_emit(struct hk_cmd_buffer *cmd, const struct hk_g17p_loader *l,
                    const struct agx3_ioto_uniform_range *ranges,
                    uint64_t *program_va, struct hk_g17p_compute_program_ref *out)
{
   struct hk_ioto_reservation r;
   if (!hk_cmd_reserve_ioto_program(cmd, l->length, "G17P loader", &r))
      return false;

   memcpy(r.map, l->bytes, l->length);
   hk_g17p_patch_plan_apply(&l->plan, ranges, r.map);

   const uint64_t va = hk_cmd_publish_ioto_program(cmd, &r, l->length);
   if (!va)
      return false;

   *program_va = va;
   *out = (struct hk_g17p_compute_program_ref){
      .bytes = r.map,
      .length = l->length,
      .launch = l->launch,
   };
   out->launch.program_va = va;
   return true;
}

static bool
hk_g17p_precompiled_uniform_data_size(
   const struct agx_precompiled_shader *prog, size_t argument_size,
   size_t *uniform_data_size)
{
   if (!prog->info.printf_uniform) {
      *uniform_data_size = argument_size;
      return true;
   }

   if (prog->info.printf_uniform < AGX_ABI_AGX3_PRECOMP_ARG_HALFS)
      return false;

   const size_t printf_offset_B =
      (size_t)(prog->info.printf_uniform - AGX_ABI_AGX3_PRECOMP_ARG_HALFS) * 2;
   *uniform_data_size =
      MAX2(argument_size, printf_offset_B + sizeof(uint64_t));
   return true;
}

static bool
hk_g17p_compute_program_cache_prewarm_kind(
   struct hk_device *dev, enum libagx_program idx, size_t data_size,
   enum agx_ioto_g17p_compute_ldshdr_f1 kind)
{
   struct agx_precompiled_shader *prog =
      agx_get_precompiled(&dev->bg_eot, idx);
   if (!prog || prog->info.rodata.size_16)
      return prog != NULL;

   size_t uniform_data_size;
   if (!hk_g17p_precompiled_uniform_data_size(prog, data_size,
                                               &uniform_data_size))
      return false;

   uint32_t uploaded_size;
   uint16_t uniform_size_halfs;
   if (!hk_agx3_g17p_precompiled_data_layout(
          uniform_data_size, &uploaded_size, &uniform_size_halfs))
      return false;

   struct agx3_cdm_direct_template_sources launch_sources;
   if (!agx3_cdm_direct_template_sources_compute(false, false, false,
                                                  &launch_sources))
      return false;

   const uint32_t threadgroup_memory_size_B =
      prog->info.local_size
         ? MAX2(256u, ALIGN_POT(prog->info.local_size, 64u))
         : 0;
   const struct agx3_ioto_uniform_range range = {
      .start_halfs = AGX_ABI_AGX3_PRECOMP_ARG_HALFS,
      .size_halfs = uniform_size_halfs,
      .address = dev->dev.shader_base,
   };
   const struct agx3_g17p_compute_program_desc desc = {
      .shader_va = prog->ptr + prog->entry_offset,
      .uniform_ranges = &range,
      .uniform_range_count = 1,
      .shader_binary_size = prog->info.main_size,
      .scratch_size = prog->info.agx3_tps_size,
      .nr_gprs = prog->info.nr_gprs,
      .compiler_d_valid = true,
      .compiler_d = (uint32_t)prog->info.agx3_staging_halfs * 2,
      .threadgroup_memory_size_B = threadgroup_memory_size_B,
      .launch_sources = launch_sources,
      .ldshdr_f1 = kind,
   };
   return hk_g17p_loader_get(dev, &dev->g17p_compute_program_cache.templates[idx],
                             &desc) != NULL;
}

bool
hk_g17p_compute_program_cache_prewarm(struct hk_device *dev,
                                      enum libagx_program program,
                                      size_t data_size)
{
   return hk_g17p_compute_program_cache_prewarm_kind(
             dev, program, data_size,
             AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3) &&
          hk_g17p_compute_program_cache_prewarm_kind(
             dev, program, data_size,
             AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1);
}

void
hk_cmd_buffer_begin_compute(struct hk_cmd_buffer *cmd,
                            const VkCommandBufferBeginInfo *pBeginInfo)
{
}

void
hk_cmd_invalidate_compute_state(struct hk_cmd_buffer *cmd)
{
   struct hk_push_descriptor_set *push[HK_MAX_SETS];
   memcpy(push, cmd->state.cs.descriptors.push, sizeof(push));
   memset(&cmd->state.cs, 0, sizeof(cmd->state.cs));
   memcpy(cmd->state.cs.descriptors.push, push, sizeof(push));
   for (unsigned i = 0; i < HK_MAX_SETS; i++) {
      if (cmd->state.cs.descriptors.push[i])
         hk_push_descriptor_set_reset_upload(cmd->state.cs.descriptors.push[i]);
   }
}

void
hk_cmd_bind_compute_shader(struct hk_cmd_buffer *cmd,
                           struct hk_api_shader *shader)
{
   cmd->state.cs.shader = shader;
}

/* A compute shader compiled without custom-border emulation dispatches as
 * its emulating twin once the device has created a sampler that needs it. */
static void
hk_cmd_resolve_compute_border(struct hk_cmd_buffer *cmd)
{
   struct hk_api_shader *shader = cmd->state.cs.shader;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (shader && unlikely(shader->min_lod_source) &&
       p_atomic_read(&hk_device_physical(dev)->min_lod_emulation)) {
      struct hk_api_shader *twin = hk_min_lod_twin(dev, shader);
      if (twin) {
         cmd->state.cs.shader = twin;
         shader = twin;
      } else {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
   }

   if (shader && unlikely(shader->border_source) &&
       p_atomic_read(&hk_device_physical(dev)->border_emulation)) {
      struct hk_api_shader *twin = hk_border_twin(dev, shader);
      if (twin)
         cmd->state.cs.shader = twin;
      else
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
}

/* Whether a strong flush recorded for `dev` with this request has memory
 * scope: on request where the form exists (G17P and M5). */
static bool
hk_cdm_flush_has_memory_scope(const struct hk_device *dev, bool memory)
{
   return memory && hk_cdm_memory_scope_form(dev);
}

bool
hk_cdm_visibility_flush_needed(const struct hk_device *dev,
                               const struct hk_cs *cs, bool memory)
{
   if (cs->stats.cmds == 0)
      return false;
   if (cs->current != cs->strong_flush_end)
      return true;
   return hk_cdm_flush_has_memory_scope(dev, memory) &&
          cs->current != cs->memory_flush_end;
}

bool
hk_cdm_pack_visibility_flush(const struct hk_device *dev, bool memory,
                             uint8_t packed[AGX3_CDM_BARRIER_SIZE])
{
   /* The AGX3 compute encoding uses flags=5 for its strong
    * post-dispatch cache flush on every grounded G15--G17 identity.  Keep
    * Honeykrisp's deliberately conservative flush-after-every-launch
    * policy, but encode it in the target command-stream ISA. */
   const struct agx3_cdm_barrier_info info = {
      .chip = dev->dev.chip,
      .flags = 0x5,
      .wide_visibility = agx_chip_cdm_wide_visibility(dev->dev.chip),
      .flush_memory = hk_cdm_flush_has_memory_scope(dev, memory),
   };
   return agx3_cdm_barrier_pack(&info, packed);
}

static void
hk_cdm_flush_impl(struct hk_device *dev, struct hk_cs *cs, bool memory,
                  bool pure_ssbo)
{
   assert(cs->type == HK_CS_CDM);
   assert(cs->current + AGX_CDM_BARRIER_LENGTH < cs->end &&
          "caller must ensure space");

   switch (agx_chip_command_stream_generation(dev->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      cs->cdm_flush_pending = false;
      if (cs->current == cs->strong_flush_end)
         return;

      {
         /* The only Borealis AGX2 barrier-word path: pure compute-to-compute
          * SSBO dependencies inside an open stream derive 0x160 from the
          * device's measured 0x1ff default word (dropping bit 3 USC cache
          * inval and PBE/texture bits); all other barriers use the device
          * default word. Sparse resources and fullcdmbarrier upgrade to the
          * full word (0xfffff). */
         const uint32_t base_word =
            pure_ssbo ? agx2_cdm_barrier_ssbo_word(
                           dev->cdm_barrier_default_mask)
                      : dev->cdm_barrier_default_mask;
         const uint32_t mask = agx2_cdm_barrier_emit_word(
            base_word, p_atomic_read(&dev->sparse_used),
            HK_PERF(dev, FULLCDMBARRIER));

         uint32_t *word = cs->current;
         *word = mask | (AGX_CDM_BLOCK_TYPE_BARRIER << 29);
         cs->current = word + 1;
         if (!pure_ssbo || mask == dev->cdm_barrier_default_mask ||
             mask == AGX_CDM_BARRIER_FULL_WORD)
            cs->strong_flush_end = cs->current;

         /* A sparse resource created between recording and submission
          * still upgrades this record at submit. */
         if (mask != AGX_CDM_BARRIER_FULL_WORD)
            hk_cdm_note_sparse_site(cs, word);
      }
      break;

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      const bool memory_scope = hk_cdm_flush_has_memory_scope(dev, memory);

      /* A strong flush already ends the stream: every launch has one in its
       * tail, and a second barrier with no launch between them is not a
       * shape Apple's CDM emitter produces. A request for memory scope the
       * flush lacks strengthens the record in place below. */
      const bool upgrade_strong = cs->current == cs->strong_flush_end;
      if (upgrade_strong &&
          (!memory_scope || cs->current == cs->memory_flush_end))
         return;

      /* Every native 26A434 G18G capture holds only the flag-free 0x60000160
       * (2 occurrences, both launch tails).  The flags=5 strong flush is
       * grounded for G18G as a chained-launch tail (rel24b+), but a
       * standalone barrier with no launch between them is not a shape
       * Apple's CDM emitter produces. Every native launch already ends in
       * that record, so there is nothing to strengthen and no grounded
       * standalone strong record to append: leave the stream as is. This
       * does not claim write visibility beyond what the weak tail gives;
       * see the rel16 notes. */
      if (dev->dev.chip == AGX_CHIP_G18G)
         return;

      uint8_t packed[AGX3_CDM_BARRIER_SIZE];
      if (!hk_cdm_pack_visibility_flush(dev, memory, packed)) {
         mesa_loge("no grounded AGX3 CDM cache flush for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cs->cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      /* Native AGX3 launches carry a weak ordering barrier in their tail.
       * When an API operation immediately needs a cache flush, strengthen
       * that record in place. Apple's emitter does not produce adjacent weak
       * and strong barriers, and retaining the weak record buys no ordering
       * while making the control stream larger. A strong record that lacks
       * the requested memory scope is strengthened the same way. */
      if (upgrade_strong || cs->current == cs->weak_barrier_end) {
         memcpy((uint8_t *)cs->current - sizeof(packed), packed,
                sizeof(packed));
         cs->strong_flush_end = cs->current;
         cs->weak_barrier_end = NULL;
         cs->memory_flush_end = memory_scope ? cs->current : NULL;
         if (!upgrade_strong)
            cs->stats.flushes++;
         return;
      }

      const bool at_stream_start = cs->current == cs->start;
      memcpy(cs->current, packed, sizeof(packed));
      cs->current = (uint32_t *)((uint8_t *)cs->current + sizeof(packed));
      cs->strong_flush_end = cs->current;
      cs->weak_barrier_end = NULL;
      cs->memory_flush_end = memory_scope ? cs->current : NULL;
      /* A barrier that opens the stream is the post-graphics strong flush.
       * It is canonical here and only here; record it so a later inline
       * concatenation can fold it into the preceding stream's trailing
       * barrier instead of leaving the two adjacent. */
      if (at_stream_start)
         cs->leading_barrier_end = cs->current;
      break;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      mesa_loge("unknown CDM command-stream generation for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cs->cmd->vk,
                                  VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   cs->stats.flushes++;
}

static void
hk_cdm_flush(struct hk_device *dev, struct hk_cs *cs, bool memory)
{
   hk_cdm_flush_impl(dev, cs, memory, false);
}

void
hk_cdm_cache_flush(struct hk_device *dev, struct hk_cs *cs)
{
   /* The driver's own flushes keep the memory scope M5 always gave them;
    * application dependencies ask for it (hk_cdm_visibility_flush()). */
   hk_cdm_flush(dev, cs, agx_chip_g17_m5(dev->dev.chip));
}

void
hk_cdm_visibility_flush(struct hk_device *dev, struct hk_cs *cs, bool memory)
{
   hk_cdm_flush(dev, cs, memory);
}

bool
hk_cmd_alias_scope(struct hk_cmd_buffer *cmd)
{
   const struct hk_device *dev = hk_cmd_buffer_device(cmd);
   return hk_cdm_alias_scope(dev) ||
          (dev->host_alias.dynamic && cmd->state.uses_events);
}

void
hk_cdm_note_alias_site(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!dev->host_alias.dynamic || cs->stats.cmds == 0)
      return;

   /* The record just before current: the strong flush or the last
    * launch's ordering record. A marker equal to current means it is in the
    * chunk being written. */
   const bool strong = cs->current == cs->strong_flush_end;
   if (strong && cs->current == cs->memory_flush_end)
      return;

   uint32_t *word =
      (uint32_t *)((uint8_t *)cs->current - AGX3_CDM_BARRIER_SIZE);
   if ((strong || cs->current == cs->weak_barrier_end) &&
       !((uintptr_t)word & (sizeof(*word) - 1))) {
      if (cmd->alias_sites.size &&
          util_dynarray_top_ptr(&cmd->alias_sites, struct hk_alias_site)
                ->word == word)
         return;

      struct hk_alias_site *site =
         util_dynarray_grow(&cmd->alias_sites, struct hk_alias_site, 1);
      if (site) {
         *site = (struct hk_alias_site){
            .word = word,
            .value = dev->host_alias.flush_word,
         };
         cs->alias_sites++;
         /* A following application dispatch must not drop the record
          * (hk_dispatch_m5_native_impl()). */
         cs->app_dispatch_end = NULL;
         return;
      }
   }

   /* No record to write later: give the dependency memory scope now. */
   hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
   hk_cdm_visibility_flush(dev, cs, true);
}

void
hk_cdm_end_stream_flush(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* A dependency declared but not yet emitted on AGX2 lands on the
    * trailing edge of the closing compute stream. Every stream close
    * funnels here, so no caller can end the compute stream with a declared
    * dependency left unemitted. This also keeps the drain out of
    * hk_cmd_buffer.h: standalone test binaries stub this function. */
   const bool flush_pending = cs->cdm_flush_pending;
   bool barrier_pending = false;
   if (cs == cmd->current_cs.cs) {
      cmd->state.cs.cdm_stream_has_images = false;
      if (cmd->state.cs.cdm_barrier_pending) {
         cmd->state.cs.cdm_barrier_pending = false;
         cmd->state.cs.cdm_barrier_full_needed = false;
         barrier_pending = true;
      }
   }
   if (barrier_pending || flush_pending) {
      hk_ensure_cs_has_space(cmd, cs, AGX_CDM_BARRIER_LENGTH);
      hk_cdm_cache_flush(dev, cs);
   }


   /* A dependency recorded while no compute stream holding work is open
    * records no flush: the first barrier of a command buffer, or one after a
    * timestamp closed the stream. The writes of this stream may then be read
    * through another address of the same memory by the next stream, in this
    * command buffer or in a later one of the same submission. While memory
    * may be mapped at a second address, end the stream with memory scope.
    * Usually this only strengthens the last launch's ordering record in
    * place. Where only a resource created later could map it there, that
    * record is a patch site instead (hk_cdm_alias_scope()). */
   if (cs->stats.cmds == 0)
      return;

   const bool alias_scope = hk_cmd_alias_scope(cmd);
   if ((alias_scope || cs->pre_gfx_writes_preamble) &&
       hk_cdm_visibility_flush_needed(dev, cs, alias_scope)) {
      hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
      hk_cdm_visibility_flush(dev, cs, alias_scope);
   }

   if (!alias_scope)
      hk_cdm_note_alias_site(cmd, cs);
}

bool
hk_cdm_record_pipeline_barrier(struct hk_cmd_buffer *cmd,
                               const VkDependencyInfo *dep)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const bool track =
      agx_chip_command_stream_generation(dev->dev.chip) ==
         AGX_CS_GENERATION_AGX2 &&
      !HK_PERF(dev, FULLCDMBARRIER);
   const bool gfx_open = cmd->current_cs.gfx ||
                         cmd->current_cs.pre_gfx ||
                         cmd->current_cs.post_gfx;

   if (track) {
      if (!cmd->state.cs.cdm_barrier_pending) {
         cmd->state.cs.cdm_barrier_pending = true;
         cmd->state.cs.cdm_barrier_full_needed = false;
      }
      if (gfx_open || !cmd->current_cs.cs ||
          cmd->current_cs.cs->stats.cmds == 0 ||
          cmd->state.cs.cdm_stream_has_images ||
          !hk_dependency_is_pure_compute_ssbo(dep)) {
         cmd->state.cs.cdm_barrier_full_needed = true;
      }
   }

   return !track || gfx_open;
}

static void
hk_cdm_narrow_trailing_barrier(struct hk_device *dev, struct hk_cs *cs)
{
   assert(cs->type == HK_CS_CDM);
   assert(agx_chip_command_stream_generation(dev->dev.chip) ==
          AGX_CS_GENERATION_AGX2);

   const uint32_t base_word =
      agx2_cdm_barrier_ssbo_word(dev->cdm_barrier_default_mask);
   const uint32_t mask = agx2_cdm_barrier_emit_word(
      base_word, p_atomic_read(&dev->sparse_used),
      HK_PERF(dev, FULLCDMBARRIER));

   if (cs->strong_flush_end && cs->current == cs->strong_flush_end) {
      if (mask != dev->cdm_barrier_default_mask &&
          mask != AGX_CDM_BARRIER_FULL_WORD) {
         uint32_t *word = (uint32_t *)cs->current - 1;
         *word = mask | (AGX_CDM_BLOCK_TYPE_BARRIER << 29);
         cs->strong_flush_end = NULL;
      }
      return;
   }

   hk_ensure_cs_has_space(cs->cmd, cs, AGX_CDM_BARRIER_LENGTH);
   hk_cdm_flush_impl(dev, cs, false, true);
}

void
hk_dispatch_with_usc_launch(struct hk_device *dev, struct hk_cs *cs,
                            struct agx_cdm_launch_word_0_packed launch,
                            uint32_t usc, struct agx_grid grid,
                            struct agx_workgroup wg, bool allow_defer)
{
   switch (agx_chip_command_stream_generation(dev->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      break;

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300:
      /* `usc` addresses an AGX2 USC block.  AGX3 direct launches instead
       * require a dispatch-local, 64-byte-aligned Ioto program VA.  Reject
       * this legacy entry point until the caller supplies that joined object;
       * never feed an AGX2 launch to the AGX3 compute encoding. */
      mesa_loge("legacy USC compute launch rejected for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cs->cmd->vk,
                                  VK_ERROR_FEATURE_NOT_PRESENT);
      return;

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      mesa_loge("unknown compute command-stream generation for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cs->cmd->vk,
                                  VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   const bool defer =
      allow_defer && cs == cs->cmd->current_cs.cs && !cs->cmd->in_meta &&
      !HK_PERF(dev, FULLCDMBARRIER) &&
      agx_chip_has_cap(dev->dev.chip, AGX_CAP_CDM_DEFER_FLUSH);

   if (cs == cs->cmd->current_cs.cs && !HK_PERF(dev, FULLCDMBARRIER) &&
       cs->cmd->state.cs.cdm_barrier_pending) {
      const bool pure_ssbo =
         !cs->cmd->state.cs.cdm_barrier_full_needed && cs->stats.cmds > 0;
      cs->cmd->state.cs.cdm_barrier_pending = false;
      cs->cmd->state.cs.cdm_barrier_full_needed = false;
      if (pure_ssbo) {
         hk_cdm_narrow_trailing_barrier(dev, cs);
      } else {
         hk_ensure_cs_has_space(cs->cmd, cs, AGX_CDM_BARRIER_LENGTH);
         hk_cdm_cache_flush(dev, cs);
      }
   } else if (!defer && cs->cdm_flush_pending) {
      hk_ensure_cs_has_space(cs->cmd, cs, AGX_CDM_BARRIER_LENGTH);
      hk_cdm_cache_flush(dev, cs);
   }

   hk_ensure_cs_has_space(cs->cmd, cs, 0x2000 /* TODO */);
   cs->stats.cmds++;

   cs->current =
      agx_cdm_launch(cs->current, dev->dev.chip, grid, wg, launch, usc);

   if (defer) {
      cs->cdm_flush_pending = true;
   } else {
      hk_cdm_cache_flush(dev, cs);
   }
}

void
hk_dispatch_with_usc(struct hk_device *dev, struct hk_cs *cs,
                     struct agx_shader_info *info, uint32_t usc,
                     struct agx_grid grid, struct agx_workgroup local_size)
{
   struct agx_cdm_launch_word_0_packed launch;
   agx_pack(&launch, CDM_LAUNCH_WORD_0, cfg) {
      cfg.texture_state_register_count = info->texture_state_count;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(info->sampler_state_count, false);
      cfg.uniform_register_count = info->push_count;
      cfg.preshader_register_count = info->nr_preamble_gprs;
   }

   if (info->stage != MESA_SHADER_COMPUTE || info->texture_state_count != 0 ||
       info->sampler_state_count != 0 || info->uses_sampler_heap ||
       info->uses_txf) {
      cs->cmd->state.cs.cdm_barrier_full_needed = true;
      cs->cmd->state.cs.cdm_stream_has_images = true;
   }
   hk_dispatch_with_usc_launch(dev, cs, launch, usc, grid, local_size,
                               info->stage == MESA_SHADER_COMPUTE);
}

static bool
hk_g17p_compute_shader_supported(enum agx_chip chip, struct hk_shader *s,
                                 struct hk_linked_shader *linked, unsigned max_spill_B)
{
   /* poly_nir_lower_gs preserves the API geometry stage on the main kernel,
    * even though Borealis has always launched that binary through CDM. */
   return s && linked &&
          (s->b.info.stage == MESA_SHADER_COMPUTE ||
           s->info.stage == MESA_SHADER_GEOMETRY ||
           s->info.stage == MESA_SHADER_TESS_CTRL) &&
          linked->b.bo && linked->b.bo->va && linked->b.size != 0 &&
          linked->b.size <= UINT32_MAX && linked->b.nr_gprs != 0 &&
          hk_agx3_compute_cprog_supported(chip, s->info.stage, &s->info,
                                          &s->b.info, max_spill_B) &&
          MAX2(s->b.info.scratch_size, s->b.info.agx3_spectpr_scratch_size) <=
             max_spill_B &&
          MAX2(linked->b.scratch_size, linked->b.agx3_spectpr_scratch_size) <=
             max_spill_B &&
          s->b.info.agx3_staging_known &&
          s->b.info.local_size <= 32768 &&
          s->b.info.texture_state_count == 0 &&
          (s->b.info.sampler_state_count == 0 ||
           ((s->info.agx3_txf_texture_valid ||
             s->info.agx3_storage_image_valid ||
             s->info.agx3_sampled_storage_valid ||
             s->info.agx3_indexed_storage_valid ||
             s->info.agx3_indexed_sampler_valid) &&
            s->b.info.sampler_state_count == 1)) &&
          !s->b.info.uses_sampler_heap &&
          (!linked->b.uses_txf || s->info.agx3_txf_texture_valid ||
           s->info.agx3_storage_image_valid ||
           s->info.agx3_sampled_storage_valid ||
           s->info.agx3_indexed_storage_valid ||
           s->info.agx3_indexed_sampler_valid) &&
          s->info.set_count <= HK_AGX3_G17P_MAX_ROOT_SETS;
}

/* Software VS adds an indexing-parameter range to the seven hardware-VS
 * ranges. Keep its eight-range collectors and exact-program cache together;
 * the compute composer independently bounds the generated load program. */

/* Uniform ABI for compute-emulated vertex work.  Geometry variants use the
 * ordinary root-at-zero compute ABI; the software vertex variant additionally
 * consumes the same VBO, draw-parameter and vertex-parameter ranges that the
 * legacy USC builder uploads.  Ioto copies these byte-identical ranges into
 * the uniform file before launching the compute-stage binary. */
static bool
hk_g17p_compute_uniform_ranges(
   enum agx_chip chip, struct hk_cmd_buffer *cmd, struct hk_shader *s,
   uint64_t root_ptr, uint64_t texture_pointers,
   struct agx3_ioto_uniform_range ranges[HK_G17P_COMPUTE_MAX_RANGES],
   size_t *range_count);

static bool
hk_g17p_internal_graphics_uniform_ranges(
   struct hk_cmd_buffer *cmd, struct hk_shader *s,
   struct hk_linked_shader *linked, uint64_t root_ptr,
   uint64_t texture_pointers,
   struct agx3_ioto_uniform_range
      ranges[HK_G17P_COMPUTE_MAX_RANGES],
   size_t *range_count)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   /* Uniform DMA is only enabled for hardware graphics stages;
    * this internal path cannot publish its span, so fail closed. */
   if (s->b.info.agx3_uniform_dma_count)
      return false;

   if (s->info.stage != MESA_SHADER_VERTEX)
      return hk_g17p_compute_uniform_ranges(dev->dev.chip, cmd, s, root_ptr,
                                             texture_pointers, ranges,
                                             range_count);

   if (!!texture_pointers != s->info.agx3_indexed_sampler_valid)
      return false;
   const unsigned uniform_base = (s->info.agx3_indexed_sampler_valid
      ? AGX_ABI_G17P_TEXTURE_POINTER_HALFS : 0) +
      (s->info.agx3_gen1_fragment_rq ? 168 : 0);

   const unsigned nr_vbos = DIV_ROUND_UP(
      BITSET_LAST_BIT(s->info.vs.attrib_components_read), 4);
   size_t count = 0;
   if (texture_pointers) {
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0,
         .size_halfs = uniform_base,
         .address = texture_pointers,
      };
   }
   if (nr_vbos) {
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = uniform_base + AGX_ABI_VUNI_VBO_BASE(0),
         .size_halfs = 4 * nr_vbos,
         .address = root_ptr + hk_root_descriptor_offset(draw.attrib_base),
      };
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = uniform_base + AGX_ABI_VUNI_VBO_CLAMP(nr_vbos, 0),
         .size_halfs = 2 * nr_vbos,
         .address = root_ptr + hk_root_descriptor_offset(draw.attrib_clamps),
      };
   }
   if (cmd->state.gfx.draw_params) {
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = uniform_base + AGX_ABI_VUNI_FIRST_VERTEX(nr_vbos),
         .size_halfs = 4,
         .address = cmd->state.gfx.draw_params,
      };
   }
   if (cmd->state.gfx.draw_id_ptr) {
      /* A uniform range is loaded in 32-bit dwords and an odd size_halfs is
       * rejected outright, so publish the containing dword. See the same
       * range in hk_cmd_buffer.c. */
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = uniform_base + AGX_ABI_VUNI_DRAW_ID(nr_vbos),
         .size_halfs = 2,
         .address = cmd->state.gfx.draw_id_ptr,
      };
   }
   if (linked->sw_indexing) {
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = uniform_base + AGX_ABI_VUNI_VERTEX_PARAMS(nr_vbos),
         .size_halfs = 4,
         .address = root_ptr + hk_root_descriptor_offset(draw.vertex_params),
      };
   }

   const unsigned root_uniform =
      uniform_base + AGX_ABI_VUNI_COUNT_VK(nr_vbos);
   if (s->info.agx3_uses_root || s->info.set_count) {
      if (count >= HK_G17P_COMPUTE_MAX_RANGES ||
          !hk_agx3_g17p_root_range(root_uniform, s->info.set_count, root_ptr,
                                   hk_shader_root_size(s), &ranges[count]))
         return false;
      count++;
   }

   if (s->b.info.rodata.size_16) {
      const struct agx_rodata *rodata = &s->b.info.rodata;
      if (count >= HK_G17P_COMPUTE_MAX_RANGES || !s->bo || !s->bo->va ||
          !hk_agx3_g17p_rodata_range(
             rodata->base_uniform, rodata->size_16, hk_shader_code_va(s),
             rodata->offset, s->b.info.binary_size, &ranges[count]))
         return false;
      count++;
   }

   *range_count = count;
   return true;
}

/* The uniform ranges of a compute-stage launch. cmd is NULL when a loader
 * is prepared ahead of recording: the span addresses are then placeholders,
 * which only the loader's layout and address shape retain. */
static bool
hk_g17p_compute_uniform_ranges(
   enum agx_chip chip, struct hk_cmd_buffer *cmd, struct hk_shader *s,
   uint64_t root_ptr, uint64_t texture_pointers,
   struct agx3_ioto_uniform_range ranges[HK_G17P_COMPUTE_MAX_RANGES],
   size_t *range_count)
{
   if (!s || !ranges || !range_count ||
       (!!texture_pointers != (s->info.agx3_txf_texture_valid ||
                               s->info.agx3_storage_image_valid ||
                               s->info.agx3_sampled_storage_valid ||
                               s->info.agx3_indexed_storage_valid ||
                               s->info.agx3_indexed_sampler_valid)))
      return false;

   /* Ordinary compute pipelines, including meta, publish their uniform DMA
    * spans here, except on M5 (see below). Geometry, tessellation and other
    * compute-stage programs of graphics never carry spans. */
   const unsigned dma_count =
      agx_chip_g17_m5(chip) ? 0 : s->b.info.agx3_uniform_dma_count;
   if (dma_count && s->info.stage != MESA_SHADER_COMPUTE)
      return false;

   /* A shader that reads neither the root descriptor nor any descriptor set
    * needs no preamble loads at all: agx3_uses_root records whether the
    * compiler emitted load_root_agx, and the program generator composes a
    * no-resource program when range_count == 0. */
   /* M5's caller appends its direct DMA spans after these ABI/rodata
    * ranges (hk_m5_compute_uniform_dma_ranges()); other chips publish them
    * here. */
   const struct agx_rodata *rodata_probe = &s->b.info.rodata;
   if (!texture_pointers && !s->info.agx3_uses_root && s->info.set_count == 0 &&
       rodata_probe->size_16 == 0 && dma_count == 0) {
      *range_count = 0;
      return true;
   }

   const unsigned root_uniform = s->info.agx3_root_uniform ? s->info.agx3_root_uniform :
      !agx_chip_g17_m5(chip) ? 0 :
      (s->info.agx3_storage_image_valid || s->info.agx3_sampled_storage_valid ||
       s->info.agx3_indexed_storage_valid) ? AGX_G17P_STORAGE_ROOT_HALFS :
      texture_pointers ? AGX_G17P_TXF_ROOT_HALFS : 0;
   if (!hk_agx3_g17p_root_range(root_uniform, s->info.set_count, root_ptr,
                                hk_shader_root_size(s), &ranges[0]))
      return false;

   size_t count = 1;
   if (texture_pointers) {
      const unsigned texture_uniform = agx_chip_g17_m5(chip) &&
         (root_uniform == 164 || root_uniform == 176) ? root_uniform - 8 :
         root_uniform >= 168 ? 168 : 0;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = texture_uniform,
         .size_halfs = root_uniform - texture_uniform,
         .address = texture_pointers,
      };
   }
   const struct agx_rodata *rodata = &s->b.info.rodata;
   if (rodata->size_16 != 0) {
      if (!s->bo || !s->bo->va ||
          !hk_agx3_g17p_rodata_range(
             rodata->base_uniform, rodata->size_16, hk_shader_code_va(s),
             rodata->offset, s->b.info.binary_size, &ranges[count]))
         return false;
      count++;
   }

   if (dma_count) {
      if (cmd) {
         if (!hk_g17p_dma_ranges(&cmd->state.cs.descriptors, s, root_ptr,
                                 ranges, HK_G17P_COMPUTE_MAX_RANGES, &count,
                                 NULL))
            return false;
      } else {
         /* Placeholder addresses with the alignment of real ones; a buffer
          * span is assumed to be covered by its buffer. */
         for (unsigned i = 0; i < dma_count; ++i) {
            const struct agx3_uniform_dma_span *span =
               &s->b.info.agx3_uniform_dma[i];
            if (count >= HK_G17P_COMPUTE_MAX_RANGES)
               return false;
            ranges[count++] = (struct agx3_ioto_uniform_range){
               .start_halfs = span->dest,
               .size_halfs = span->halfs,
               .address = root_ptr + 0x4000 + span->offset_B,
            };
         }
      }
   }

   *range_count = count;
   return true;
}

/* The launch description of a linked compute-stage program. */
static bool
hk_g17p_linked_program_desc(struct hk_shader *s,
                            struct hk_linked_shader *linked,
                            enum agx_ioto_g17p_compute_ldshdr_f1 kind,
                            const struct agx3_ioto_uniform_range *ranges,
                            size_t range_count,
                            struct agx3_g17p_compute_program_desc *desc)
{
   struct agx3_cdm_direct_template_sources launch_sources;
   if (!agx3_cdm_direct_template_sources_compute(
          s->b.info.uses_resource_info_query, s->b.info.uses_ray_traversal,
          s->b.info.ray_traversal_depth_nonzero, &launch_sources))
      return false;

   struct agx_ioto_constant_program constant_program = {0};
   const struct agx3_cprog_inputs *cprog_inputs = NULL;
   if (s->b.info.has_preamble) {
      if (!s->bo || !s->bo->va ||
          !hk_agx3_g17p_constant_program_from_binary(
             s->code_ptr, s->code_size, s->b.info.preamble_offset,
             s->b.info.preamble_size, hk_shader_code_va(s),
             s->b.info.binary_size, &constant_program))
         return false;
      constant_program.force_external = true;
      constant_program.absolute_branch = true;
      constant_program.request_prelude = s->b.info.agx3_preamble_uniform_inputs;
      if (s->b.info.agx3_preamble_tps_inputs)
         cprog_inputs = &s->b.info.agx3_preamble_inputs;
      const unsigned cprog_frame =
         ALIGN_POT(MAX2((unsigned)s->b.info.agx3_preamble_inputs.storage_bytes,
                        s->b.info.preamble_scratch_size), 16);
      launch_sources.ti_spill_bytes =
         MAX2(launch_sources.ti_spill_bytes, cprog_frame);
      launch_sources.ti_largest_ipr_bytes =
         MAX2(launch_sources.ti_largest_ipr_bytes,
              (uint32_t)s->b.info.agx3_preamble_staging_halfs * 2u);
   }

   *desc = (struct agx3_g17p_compute_program_desc){
      .shader_va = linked->b.bo->va->addr + linked->b.entry_offset,
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
      .shader_binary_size = linked->b.size,
      .scratch_size =
         MAX2(linked->b.scratch_size, linked->b.agx3_spectpr_scratch_size),
      .nr_gprs = MAX2(linked->b.nr_gprs, s->b.info.nr_preamble_gprs),
      .compiler_d_valid = s->b.info.agx3_staging_known,
      .compiler_d = MAX2((uint32_t)s->b.info.agx3_staging_halfs * 2,
                         (uint32_t)s->b.info.agx3_preamble_staging_halfs * 2),
      .threadgroup_memory_size_B =
         s->b.info.local_size ? MAX2(256u, ALIGN_POT(s->b.info.local_size, 64u))
                              : 0,
      .imageblock_stride_B = s->b.info.imageblock_stride,
      .launch_sources = launch_sources,
      .ldshdr_f1 = kind,
      .constant_program = constant_program,
      .cprog_inputs = cprog_inputs,
   };
   return true;
}

/* Emit the loader of a launch of desc's program into the command buffer.
 * head is the loader list of the program: its linked shader's, or the
 * device's for a precompiled kernel. */
static bool
hk_g17p_build_compute_program_raw(
   struct hk_cmd_buffer *cmd, void **head,
   const struct agx3_g17p_compute_program_desc *desc,
   uint64_t *program_va_out, struct hk_g17p_compute_program_ref *out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_g17p_loader *l = hk_g17p_loader_get(dev, head, desc);
   if (!l) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   return hk_g17p_loader_emit(cmd, l, desc->uniform_ranges, program_va_out,
                              out);
}

static bool
hk_g17p_build_compute_program(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                              struct hk_shader *s,
                              struct hk_linked_shader *linked,
                              uint64_t root_ptr, uint64_t texture_pointers,
                              struct agx_workgroup local_size,
                              bool internal_graphics,
                              enum agx_ioto_g17p_compute_ldshdr_f1 kind,
                              const char *trace,
                              struct agx3_ioto_uniform_range
                                 ranges[HK_G17P_COMPUTE_MAX_RANGES],
                              size_t *range_count_out,
                              uint64_t *program_va_out,
                              struct hk_g17p_compute_program_ref *out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   size_t range_count;
   const bool ranges_ok = internal_graphics
      ? hk_g17p_internal_graphics_uniform_ranges(
           cmd, s, linked, root_ptr, texture_pointers, ranges, &range_count)
      : hk_g17p_compute_uniform_ranges(dev->dev.chip, cmd, s, root_ptr,
                                        texture_pointers, ranges, &range_count);
   if (!ranges_ok) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   if(s->b.info.uses_ray_traversal && hk_cmd_buffer_device(cmd)->gen1_native_rq) {
      struct hk_device *dev=hk_cmd_buffer_device(cmd);
      if((internal_graphics && !dev->gen1_native_graphics_rq) ||
         range_count>=HK_G17P_COMPUTE_MAX_RANGES) {
         vk_command_buffer_set_error(&cmd->vk,VK_ERROR_FEATURE_NOT_PRESENT);return false;
      }
      uint32_t context[28]={0,0,0x60000070,0x280000,0,0,0,0,0,64,1,1,
                             0,0,1,1,0,0x6f,0,0,0x2f8000,0,0,0,0,0,
                             0x00400040,0x7fff0005};
      const bool launch_shape = dev->gen1_native_graphics_rq &&
         s->b.info.workgroup_size[0] == 0 && s->b.info.workgroup_size[1] == 0 &&
         s->b.info.workgroup_size[2] == 0;
      if (!agx_rt_compute_gen1_context_local(context,
             launch_shape ? local_size.x : s->b.info.workgroup_size[0],
             launch_shape ? local_size.y : s->b.info.workgroup_size[1],
             launch_shape ? local_size.z : s->b.info.workgroup_size[2])) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }
      if (!hk_gen1_upload_rq_uniforms(cmd, context, ARRAY_SIZE(context),
                                      false, &ranges[range_count]))
         return false;
      range_count++;
      if (!cs) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }
      cs->gen1_native_rq = true;
   }
   struct agx3_g17p_compute_program_desc desc;
   if (!hk_g17p_linked_program_desc(s, linked, kind, ranges, range_count,
                                    &desc)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   /* The recorded Ioto program dereferences these executable/constant BOs
    * after vkQueueSubmit returns. Their originating pipeline need not remain
    * alive after command recording, so the command buffer owns references. */
   hk_cmd_buffer_retain_bo(cmd, linked->b.bo);
   if (s->b.info.rodata.size_16 != 0 || s->b.info.has_preamble)
      hk_cmd_buffer_retain_bo(cmd, s->bo);

   const bool ok = hk_g17p_build_compute_program_raw(
      cmd, &linked->g17p_compute_templates, &desc, program_va_out, out);
   if (ok)
      *range_count_out = range_count;
   return ok;
}

static void
hk_g17p_prewarm_linked_compute_program_kind(
   struct hk_device *dev, struct hk_cmd_buffer *cmd, struct hk_shader *s,
   struct hk_linked_shader *linked, bool internal_graphics,
   enum agx_ioto_g17p_compute_ldshdr_f1 kind)
{
   if (!agx_chip_native_agx3(dev->dev.chip) ||
       !hk_g17p_compute_shader_supported(dev->dev.chip, s, linked,
                                         AGX3_G17P_MAX_COMPUTE_SPILL_B))
      return;

   /* Use ordinary full-width, aligned addresses so the cached template has
    * the same LDIMM encoding shape as command-buffer uploads. Only the range
    * layout is retained; every real address is relocated at lookup. */
   const uint64_t dummy_root = UINT64_C(0x5a89abcde0);
   const bool uses_texture_pointers =
      s->info.agx3_txf_texture_valid ||
      s->info.agx3_storage_image_valid ||
      s->info.agx3_sampled_storage_valid ||
      s->info.agx3_indexed_storage_valid ||
      s->info.agx3_indexed_sampler_valid;
   const uint64_t dummy_texture_pointers =
      uses_texture_pointers ? dummy_root + 0x2000 : 0;

   struct agx3_ioto_uniform_range
      ranges[HK_G17P_COMPUTE_MAX_RANGES];
   size_t range_count;
   const bool vertex_graphics =
      internal_graphics && s->info.stage == MESA_SHADER_VERTEX;
   if (vertex_graphics && !cmd)
      return;

   const bool ranges_ok = vertex_graphics
      ? hk_g17p_internal_graphics_uniform_ranges(
           cmd, s, linked, dummy_root, dummy_texture_pointers, ranges,
           &range_count)
      : hk_g17p_compute_uniform_ranges(dev->dev.chip, NULL, s, dummy_root,
                                        dummy_texture_pointers, ranges,
                                        &range_count);
   struct agx3_g17p_compute_program_desc desc;
   if (!ranges_ok ||
       !hk_g17p_linked_program_desc(s, linked, kind, ranges, range_count,
                                    &desc))
      return;

   hk_g17p_loader_get(dev, &linked->g17p_compute_templates, &desc);
}

void
hk_g17p_prewarm_linked_compute_shader(struct hk_device *dev,
                                       struct hk_shader *s,
                                       struct hk_linked_shader *linked)
{
   if (!s || s->info.stage == MESA_SHADER_VERTEX)
      return;

   const enum agx_ioto_g17p_compute_ldshdr_f1 kinds[] = {
      AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
      AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1,
      AGX3_G17P_GRID_HELPER_LDSHDR_F1,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(kinds); ++i)
      hk_g17p_prewarm_linked_compute_program_kind(
         dev, NULL, s, linked, s->info.stage != MESA_SHADER_COMPUTE, kinds[i]);
}

/*
 * The helper that publishes an indirect grid's launch geometry. An
 * indirect-global grid holds threadgroup counts, which the grid helper
 * multiplies by the local size; an indirect-local grid already holds the
 * geometry, written by an earlier kernel, which the copy helper publishes.
 */
static struct hk_shader *
hk_g17p_indirect_helper(struct hk_device *dev, struct agx_grid grid,
                        uint64_t sparse_map_addr)
{
   struct hk_api_shader *helper =
      grid.mode == AGX_CDM_MODE_INDIRECT_GLOBAL
         ? (sparse_map_addr ? dev->g17p_grid_helper_sparse
                            : dev->g17p_grid_helper)
      : grid.mode == AGX_CDM_MODE_INDIRECT_LOCAL
         ? dev->g17p_geometry_copy_helper
         : NULL;

   return helper ? hk_only_variant(helper) : NULL;
}

static bool
hk_g17p_indirect_launch_supported(struct hk_device *dev, struct agx_grid grid,
                                  struct agx_workgroup local_size,
                                  uint64_t sparse_map_addr)
{
   /* Without a helper program there is nothing to publish the geometry, so
    * refuse rather than record a launch whose dimensions never arrive. */
   struct hk_shader *helper =
      hk_g17p_indirect_helper(dev, grid, sparse_map_addr);
   if (grid.ptr == 0 || (grid.ptr & 3) != 0 ||
       !hk_g17p_compute_shader_supported(dev->dev.chip, helper,
                                         helper ? helper->only_linked : NULL,
                                         agx_chip_g17_m5(dev->dev.chip)
                                            ? AGX3_G17P_MAX_SPILL_B
                                            : AGX3_G17P_MAX_COMPUTE_SPILL_B))
      return false;

   /* The local size must be launchable at all: a zero component would make
    * the helper publish a zero global size, which is indistinguishable from a
    * geometry block it never wrote.  The product itself is the application's
    * responsibility -- the helper multiplies in wrapping 32-bit, exactly as
    * the AGX2 hardware indirect mode does, and Vulkan already bounds
    * groupCount by maxComputeWorkGroupCount.  An indirect-local grid brings
    * its own local size. */
   return grid.mode == AGX_CDM_MODE_INDIRECT_LOCAL ||
          (local_size.x != 0 && local_size.y != 0 && local_size.z != 0);
}

/*
 * Append an indirect launch of a built main program to `cs`: the helper
 * publishes the grid's launch geometry, and the main record consumes it. The
 * helper program is built for the same stream as the launch.
 */
static __attribute__((hot)) bool
hk_g17p_append_indirect_launch(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs, struct agx_grid grid,
   struct agx_workgroup local_size, uint64_t sparse_map_addr,
   const struct hk_g17p_compute_program_ref *program,
   struct agx3_compute_dispatch_sink *sink)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_shader *helper_s =
      hk_g17p_indirect_helper(dev, grid, sparse_map_addr);

   /* The six-u32 block the helper publishes and the main record consumes.
    * It lives in the command buffer's own pool, so it is allocated per
    * dispatch and cannot be shared between two launches in flight. */
   const struct agx_ptr geometry =
      hk_pool_alloc(cmd, AGX3_G17P_INDIRECT_GEOMETRY_SIZE,
                    AGX3_G17P_INDIRECT_GEOMETRY_ALIGNMENT);
   if (!geometry.gpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }

   /* Zero it: an unwritten block must not read back as a launchable
    * geometry if the helper never ran. */
   memset(geometry.cpu, 0, AGX3_G17P_INDIRECT_GEOMETRY_SIZE);

   /* The helper gets its own root.  It shares the caller's descriptor
    * state -- it touches none of it -- but it needs the grid address, the
    * geometry address and the local size, and the caller's root must keep
    * describing the caller's dispatch. */
   struct hk_root_descriptor_table helper_root =
      cmd->state.cs.descriptors.root;
   helper_root.cs.group_count_addr = grid.ptr;
   helper_root.cs.grid_helper_geometry_addr = geometry.gpu;
   /* Authoritative here rather than whatever the copied root carries: an
    * internal or precompiled launch must never inherit an application
    * dispatch's residency bitmap. */
   helper_root.cs.grid_helper_sparse_map_addr = sparse_map_addr;
   helper_root.cs.grid_helper_local_size[0] = local_size.x;
   helper_root.cs.grid_helper_local_size[1] = local_size.y;
   helper_root.cs.grid_helper_local_size[2] = local_size.z;

   const struct agx_ptr helper_root_ptr =
      hk_pool_alloc(cmd, sizeof(helper_root), 8);
   if (!helper_root_ptr.gpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }
   helper_root.root_desc_addr = helper_root_ptr.gpu;
   memcpy(helper_root_ptr.cpu, &helper_root, sizeof(helper_root));

   struct agx3_ioto_uniform_range
      helper_ranges[HK_G17P_COMPUTE_MAX_RANGES];
   size_t helper_range_count;
   uint64_t helper_program_va;
   struct hk_g17p_compute_program_ref helper_program;
   if (!hk_g17p_build_compute_program(
          cmd, cs, helper_s, helper_s->only_linked,
          helper_root_ptr.gpu, 0,
          local_size, false,
          AGX3_G17P_GRID_HELPER_LDSHDR_F1, "grid-helper", helper_ranges,
          &helper_range_count, &helper_program_va, &helper_program))
      return false;

   const struct agx3_g17p_native_indirect_dispatch dispatch = {
      .helper_program = &helper_program.launch,
      .geometry_va = geometry.gpu,
   };
   if (!hk_agx3_g17p_native_indirect_compute_dispatch_append(
          &program->launch, &dispatch, sink)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   return true;
}

static bool
hk_g17p_get_precomp_program(struct hk_cmd_buffer *cmd,
                             struct agx_precompiled_shader *prog,
                             enum libagx_program idx, const void *data,
                             size_t data_size, bool indirect,
                             struct hk_g17p_compute_program_ref *program)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   size_t uniform_data_size;
   if (!hk_g17p_precompiled_uniform_data_size(prog, data_size,
                                               &uniform_data_size)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   const size_t printf_offset_B = prog->info.printf_uniform
      ? (size_t)(prog->info.printf_uniform - AGX_ABI_AGX3_PRECOMP_ARG_HALFS) * 2
      : 0;
   uint32_t uploaded_size;
   uint16_t uniform_size_halfs;
   if (!hk_agx3_g17p_precompiled_data_layout(
          uniform_data_size, &uploaded_size, &uniform_size_halfs)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   /* G17P Ioto resource loads operate on whole dwords. Packed argument
    * structs may end in a 16-bit member, so zero the unused padding half. */
   struct agx_ptr uploaded = hk_pool_alloc(cmd, uploaded_size, 4);
   if (!uploaded.cpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return false;
   }
   memcpy(uploaded.cpu, data, data_size);
   memset((uint8_t *)uploaded.cpu + data_size, 0, uploaded_size - data_size);
   if (prog->info.printf_uniform) {
      memcpy((uint8_t *)uploaded.cpu + printf_offset_B,
             &dev->dev.printf_buffer_address, sizeof(uint64_t));
   }

   struct agx3_ioto_uniform_range ranges[2] = {{
      .start_halfs = AGX_ABI_AGX3_PRECOMP_ARG_HALFS,
      .size_halfs = uniform_size_halfs,
      .address = uploaded.gpu,
   }};
   unsigned range_count = 1;
   if (prog->info.rodata.size_16) {
      ranges[range_count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = prog->info.rodata.base_uniform,
         .size_halfs = prog->info.rodata.size_16,
         .address = prog->ptr + prog->rodata_offset,
      };
   }

   struct agx3_cdm_direct_template_sources launch_sources;
   if (!agx3_cdm_direct_template_sources_compute(false, false, false,
                                                  &launch_sources)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   const struct agx3_g17p_compute_program_desc desc = {
      .shader_va = prog->ptr + prog->entry_offset,
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
      .shader_binary_size = prog->info.main_size,
      .scratch_size = prog->info.agx3_tps_size,
      .nr_gprs = prog->info.nr_gprs,
      .compiler_d_valid = true,
      .compiler_d = (uint32_t)prog->info.agx3_staging_halfs * 2,
      .threadgroup_memory_size_B =
         prog->info.local_size
            ? MAX2(256u, ALIGN_POT(prog->info.local_size, 64u))
            : 0,
      .launch_sources = launch_sources,
      .ldshdr_f1 = indirect ? AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1
                            : AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
   };

   uint64_t program_va;
   if (!hk_g17p_build_compute_program_raw(
          cmd, &dev->g17p_compute_program_cache.templates[idx], &desc,
          &program_va, program))
      return false;

   return true;
}

__attribute__((hot)) void
hk_dispatch_precomp_g17p(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                         enum agx_barrier barrier, enum libagx_program idx,
                         const void *data, size_t data_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if ((unsigned)idx >= LIBAGX_NUM_PROGRAMS) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct agx_precompiled_shader *prog =
      agx_get_precompiled(&dev->bg_eot, idx);
   if (!prog) {
      mesa_loge("no G17P precompiled program %u", (unsigned)idx);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (!agx_chip_native_agx3(dev->dev.chip) || !prog->bo ||
       !prog->info.main_size || !prog->info.nr_gprs ||
       prog->info.nr_preamble_gprs ||
       (prog->info.rodata.size_16 && !prog->rodata_offset) ||
       (prog->info.printf_uniform &&
        prog->info.printf_uniform < AGX_ABI_AGX3_PRECOMP_ARG_HALFS) ||
       !prog->info.agx3_staging_known ||
       !data || !data_size ||
       prog->entry_offset > UINT64_MAX - prog->ptr ||
       prog->b.workgroup.x == 0 || prog->b.workgroup.y == 0 ||
       prog->b.workgroup.z == 0) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   const bool indirect = agx_is_indirect(grid);
   const struct agx_workgroup local_size = {
      .x = prog->b.workgroup.x,
      .y = prog->b.workgroup.y,
      .z = prog->b.workgroup.z,
   };
   if (indirect &&
       !hk_g17p_indirect_launch_supported(dev, grid, local_size, 0)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   if (!indirect &&
       (grid.count[0] == 0 || grid.count[1] == 0 || grid.count[2] == 0))
      return;

   struct hk_cs **target = (barrier & AGX_POSTGFX)  ? &cmd->current_cs.post_gfx
                           : (barrier & AGX_PREGFX) ? &cmd->current_cs.pre_gfx
                                                    : &cmd->current_cs.cs;
   struct hk_cs *cs = hk_cmd_buffer_get_cs_general(cmd, target, true);
   if (!cs)
      return;
   if (!(barrier & (AGX_PREGFX | AGX_POSTGFX)))
      cs->render_pack_compute_internal = true;

   struct hk_g17p_compute_program_ref program;
   if (!hk_g17p_get_precomp_program(cmd, prog, idx, data, data_size,
                                    indirect, &program))
      return;

   if (indirect) {
      hk_ensure_cs_has_space(cmd, cs, 0x2000);
      struct agx3_compute_dispatch_sink sink = {
         .data = (uint8_t *)cs->current,
         .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
      };
      if (!hk_g17p_append_indirect_launch(cmd, cs, grid, local_size, 0,
                                          &program, &sink))
         return;

      cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
      /* The indirect pair ends in the deferred launch's own weak ordering
       * barrier (agx3_g17p_native_indirect_compute_dispatch_append passes
       * flags 0).  Record it exactly as the direct path does, so a following
       * hk_cdm_cache_flush() strengthens that record in place instead of
       * appending a second barrier after it: two adjacent barriers with no
       * launch between them is a shape Apple's CDM emitter never produces and
       * the G17P stream oracle rejects. */
      cs->weak_barrier_end = cs->current;
      cs->app_dispatch_end = NULL;
      /* The direct path below ends an AGX_BARRIER_ALL launch in the strong
       * flush its consumer relies on. Give the indirect launch the same:
       * strengthen its weak tail in place. */
      if (barrier & AGX_BARRIER_ALL)
         hk_cdm_cache_flush(dev, cs);
      if (unlikely(hk_cmd_buffer_device(cmd)->frame_anatomy))
         hk_frame_anatomy_note_precomp(cs, idx, grid);
      cs->stats.cmds++;
      return;
   }

   const struct agx3_compute_direct_grid direct = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {grid.count[0], grid.count[1], grid.count[2]},
      .local_size = {prog->b.workgroup.x, prog->b.workgroup.y,
                     prog->b.workgroup.z},
      .dimensions = grid.count[2] != 1 || prog->b.workgroup.z != 1
                       ? 3
                    : grid.count[1] != 1 || prog->b.workgroup.y != 1 ? 2 : 1,
   };

   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   struct agx3_compute_dispatch_sink sink = {
      .data = (uint8_t *)cs->current,
      .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
   };
   /* Internal helpers communicate through memory with their consumer. Match
    * the AGX2 flush-after-launch contract when the caller requests the common
    * all-memory barrier; HAL200 encodes Apple's witnessed strong value 5 in
    * the four-byte tail of this dispatch. */
   const uint8_t barrier_flags = (barrier & AGX_BARRIER_ALL) ? 0x5 : 0;
   if (!hk_agx3_g17p_native_compute_dispatch_append_with_barrier(
          &program.launch, &direct, barrier_flags, &sink)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   if (barrier_flags == 0x5) {
      cs->strong_flush_end = cs->current;
      cs->weak_barrier_end = NULL;
   } else {
      cs->weak_barrier_end = cs->current;
   }
   if (unlikely(hk_cmd_buffer_device(cmd)->frame_anatomy))
      hk_frame_anatomy_note_precomp(cs, idx, grid);
   cs->stats.cmds++;
}


static void
hk_m5_compute_resources(struct hk_cs *cs, uint32_t scratch_B,
                        uint32_t secondary_B, uint32_t shared_B, uint32_t threads)
{
   const struct drm_asahi_compute_program profile = {
      .primary_bytes = scratch_B, .secondary_bytes = secondary_B,
      .threadgroup_bytes = shared_B, .threads = threads,
   };
   /* The kernel folds every listed program into the submission's demand with
    * a per-field maximum, so the list is a set: identical profiles add
    * nothing. Deduplicate against the whole list, not just the last entry;
    * long streams that alternate a few dispatch shapes (software tessellation
    * prepasses, acceleration-structure builds) otherwise overflow the 16-bit
    * command size and lose the device. */
   util_dynarray_foreach(&cs->compute_resources,
                         struct drm_asahi_compute_program, entry) {
      if (!memcmp(entry, &profile, sizeof(profile)))
         return;
   }
   util_dynarray_append(&cs->compute_resources, profile);
}

void
hk_dispatch_precomp_m5(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                         enum agx_barrier barrier, enum libagx_program idx,
                         const void *data, size_t data_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!agx_chip_g17_m5(dev->dev.chip) ||
       (unsigned)idx >= LIBAGX_NUM_PROGRAMS) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct agx_precompiled_shader *prog = agx_get_precompiled(&dev->bg_eot, idx);
   if (!prog || !prog->bo || !prog->info.main_size || !prog->info.nr_gprs ||
       prog->info.nr_preamble_gprs ||
       (prog->info.agx3_tps_size && !(dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_RESOURCES)) ||
       prog->info.local_size >
          dev->vk.physical->properties.maxComputeSharedMemorySize || !prog->info.agx3_staging_known ||
       (prog->info.rodata.size_16 && !prog->rodata_offset) ||
       !data || !data_size || prog->entry_offset > UINT64_MAX - prog->ptr ||
       !prog->b.workgroup.x || !prog->b.workgroup.y || !prog->b.workgroup.z ||
       (uint64_t)prog->b.workgroup.x * prog->b.workgroup.y *
          prog->b.workgroup.z > dev->vk.physical->properties.maxComputeWorkGroupInvocations) {
      mesa_loge("M5 precompiled helper %u: unsupported resources or grid", idx);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (!agx_is_indirect(grid) &&
       (!grid.count[0] || !grid.count[1] || !grid.count[2]))
      return;

   size_t uniform_data_size;
   uint32_t uploaded_size;
   uint16_t uniform_size_halfs;
   if (!hk_g17p_precompiled_uniform_data_size(prog, data_size,
                                               &uniform_data_size) ||
       !hk_agx3_g17p_precompiled_data_layout(
          uniform_data_size, &uploaded_size, &uniform_size_halfs)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct hk_cs **target = (barrier & AGX_POSTGFX) ? &cmd->current_cs.post_gfx
                         : (barrier & AGX_PREGFX) ? &cmd->current_cs.pre_gfx
                                                 : &cmd->current_cs.cs;
   struct hk_cs *cs = hk_cmd_buffer_get_cs_general(cmd, target, true);
   if (!cs)
      return;
   if (!(barrier & (AGX_PREGFX | AGX_POSTGFX)))
      cs->render_pack_compute_internal = true;

   struct agx_ptr uploaded = hk_pool_alloc(cmd, uploaded_size, 4);
   if (!uploaded.cpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   memcpy(uploaded.cpu, data, data_size);
   memset((uint8_t *)uploaded.cpu + data_size, 0, uploaded_size - data_size);
   if (prog->info.printf_uniform) {
      const size_t offset =
         (prog->info.printf_uniform - AGX_ABI_AGX3_PRECOMP_ARG_HALFS) * 2;
      memcpy((uint8_t *)uploaded.cpu + offset, &dev->dev.printf_buffer_address,
             sizeof(uint64_t));
   }
   struct agx3_ioto_uniform_range ranges[2] = {{
      .start_halfs = AGX_ABI_AGX3_PRECOMP_ARG_HALFS,
      .size_halfs = uniform_size_halfs, .address = uploaded.gpu,
   }};
   unsigned range_count = 1;
   if (prog->info.rodata.size_16) {
      ranges[range_count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = prog->info.rodata.base_uniform,
         .size_halfs = prog->info.rodata.size_16,
         .address = prog->ptr + prog->rodata_offset,
      };
   }
   size_t capacity;
   if (!agx3_g17g_graphics_program_bound(dev->dev.chip, ranges, range_count,
                                         &capacity)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct hk_ioto_reservation reservation;
   if (!hk_cmd_reserve_ioto_program(cmd, capacity, "M5 precompiled", &reservation))
      return;
   const uint64_t shader_va = prog->ptr + prog->entry_offset;
   struct agx3_g17p_graphics_program_desc desc = {
      .program_va = reservation.va, .shader_va = shader_va,
      .uniform_ranges = ranges, .uniform_range_count = range_count,
      .shader_binary_size = prog->info.main_size,
      .scratch_size = prog->info.agx3_tps_size,
      .nr_gprs = prog->info.nr_gprs,
      .compiler_d_valid = prog->info.agx3_staging_known,
      .compiler_d = (uint32_t)prog->info.agx3_staging_halfs * 2,
      .threadgroup_memory_size_B = prog->info.local_size
         ? MAX2(256u, ALIGN_POT(prog->info.local_size, 64u)) : 0,
   };
   struct agx3_g17p_graphics_program program;
   struct agx3_g17p_native_compute_program_desc launch_desc = {
      .program_va = reservation.va, .shader_va = shader_va,
   };
   struct agx3_compute_program launch;
   if (!hk_m5_compute_program_generate(dev, &desc, &program) ||
       !agx3_cdm_direct_template_sources_compute(
          false, false, false, &launch_desc.launch_sources) ||
       !hk_m5_compute_launch_bind(dev, &launch_desc, &launch)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   memcpy(reservation.map, program.bytes, program.length);
   if (!hk_cmd_publish_ioto_program(cmd, &reservation, program.length))
      return;
   hk_cmd_buffer_retain_bo(cmd, prog->bo);
   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   struct agx3_compute_dispatch_sink sink = {
      .data = (uint8_t *)cs->current,
      .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
   };
   const uint8_t flags = (barrier & AGX_BARRIER_ALL) ? 5 : 0;
   const uint32_t local[] = {
      prog->b.workgroup.x, prog->b.workgroup.y, prog->b.workgroup.z,
   };
   const struct agx3_compute_direct_grid direct = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {grid.count[0], grid.count[1], grid.count[2]},
      .local_size = {local[0], local[1], local[2]},
      .dimensions = grid.count[2] != 1 || local[2] != 1 ? 3
                  : grid.count[1] != 1 || local[1] != 1 ? 2 : 1,
   };
   const bool appended = agx_is_indirect(grid)
      ? agx3_m5_native_indirect_dispatch_append(
           &launch, grid.ptr,
           grid.mode == AGX_CDM_MODE_INDIRECT_GLOBAL ? local : NULL, flags, &sink)
      : agx3_m5_native_compute_dispatch_append(&launch, &direct, flags, &sink);
   if (!appended) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   if (flags == 5) {
      /* M5's strong launch tail has memory scope. */
      cs->strong_flush_end = cs->current;
      cs->memory_flush_end = cs->current;
      cs->weak_barrier_end = NULL;
   } else {
      cs->weak_barrier_end = cs->current;
   }
   hk_m5_compute_resources(cs, desc.scratch_size, 0, desc.threadgroup_memory_size_B,
                           local[0] * local[1] * local[2]);
   if (unlikely(dev->frame_anatomy))
      hk_frame_anatomy_note_precomp(cs, idx, grid);
   cs->stats.cmds++;
}

bool
hk_dispatch_preamble_cs(struct hk_cmd_buffer *cmd,
                        struct hk_shader *preamble_cs, uint64_t root_ptr,
                        uint64_t out_ptr)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *linked = preamble_cs ? preamble_cs->only_linked : NULL;
   if (!agx_chip_native_or_m5(dev->dev.chip) || dev->dev.chip == AGX_CHIP_G18G ||
       !preamble_cs || !linked || !linked->b.bo || !linked->b.bo->va ||
       !linked->b.size || !linked->b.nr_gprs || !preamble_cs->b.info.agx3_staging_known ||
       preamble_cs->b.info.has_preamble ||
       (preamble_cs->b.info.rodata.size_16 &&
        (!preamble_cs->bo || !preamble_cs->bo->va)))
      return false;

   const uint32_t scratch_size =
      MAX2(linked->b.scratch_size, linked->b.agx3_spectpr_scratch_size);
   if (agx_chip_g17_m5(dev->dev.chip) && scratch_size &&
       !(dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_RESOURCES))
      return false;

   const uint32_t local_B = preamble_cs->b.info.local_size
      ? MAX2(256u, ALIGN_POT(preamble_cs->b.info.local_size, 64u))
      : 0;
   if (local_B > dev->vk.physical->properties.maxComputeSharedMemorySize)
      return false;

   struct agx_ptr uploaded = hk_pool_alloc(cmd, 2 * sizeof(uint64_t), 8);
   if (!uploaded.cpu)
      return false;
   const uint64_t args[2] = {root_ptr, out_ptr};
   memcpy(uploaded.cpu, args, sizeof(args));

   struct agx3_ioto_uniform_range ranges[2] = {{
      .start_halfs = AGX_ABI_AGX3_PRECOMP_ARG_HALFS,
      .size_halfs = 8,
      .address = uploaded.gpu,
   }};
   unsigned range_count = 1;
   if (preamble_cs->b.info.rodata.size_16) {
      const struct agx_rodata *rodata = &preamble_cs->b.info.rodata;
      if (!hk_agx3_g17p_rodata_range(rodata->base_uniform, rodata->size_16,
                                     hk_shader_code_va(preamble_cs),
                                     rodata->offset,
                                     preamble_cs->b.info.binary_size,
                                     &ranges[range_count++]))
         return false;
   }

   struct agx3_cdm_direct_template_sources launch_sources;
   if (!agx3_cdm_direct_template_sources_compute(false, false, false,
                                                  &launch_sources))
      return false;

   const uint64_t shader_va = linked->b.bo->va->addr + linked->b.entry_offset;
   const struct agx3_compute_direct_grid direct = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {1, 1, 1},
      .local_size = {1, 1, 1},
      .dimensions = 1,
   };

   struct hk_cs *cs =
      hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.pre_gfx, true);
   if (!cs)
      return false;

   if (agx_chip_native_agx3(dev->dev.chip)) {
      const struct agx3_g17p_compute_program_desc desc = {
         .shader_va = shader_va,
         .uniform_ranges = ranges,
         .uniform_range_count = range_count,
         .shader_binary_size = linked->b.size,
         .scratch_size = scratch_size,
         .nr_gprs = linked->b.nr_gprs,
         .compiler_d_valid = true,
         .compiler_d = (uint32_t)preamble_cs->b.info.agx3_staging_halfs * 2,
         .threadgroup_memory_size_B = local_B,
         .launch_sources = launch_sources,
         .ldshdr_f1 = AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
      };
      uint64_t program_va;
      struct hk_g17p_compute_program_ref program;
      if (!hk_g17p_build_compute_program_raw(
             cmd, &linked->g17p_compute_templates, &desc, &program_va,
             &program))
         return false;

      hk_cmd_buffer_retain_bo(cmd, linked->b.bo);
      if (preamble_cs->b.info.rodata.size_16)
         hk_cmd_buffer_retain_bo(cmd, preamble_cs->bo);

      hk_ensure_cs_has_space(cmd, cs, 0x2000);
      struct agx3_compute_dispatch_sink sink = {
         .data = (uint8_t *)cs->current,
         .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
      };
      if (!hk_agx3_g17p_native_compute_dispatch_append_with_barrier(
             &program.launch, &direct, 0, &sink))
         return false;

      cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
      /* Later CDM helpers can copy the address of the output into vertex
       * programs, but only VDM reads the values. Share their final flush,
       * or upgrade the stream's tail when no helper has flushed it. */
      cs->weak_barrier_end = cs->current;
      cs->pre_gfx_writes_preamble = true;
      cs->stats.cmds++;
      return true;
   }

   size_t capacity;
   if (!agx3_g17g_graphics_program_bound(dev->dev.chip, ranges, range_count,
                                         &capacity))
      return false;
   struct hk_ioto_reservation reservation;
   if (!hk_cmd_reserve_ioto_program(cmd, capacity, "M5 hoisted preamble",
                                    &reservation))
      return false;

   struct agx3_g17p_graphics_program_desc desc = {
      .program_va = reservation.va,
      .shader_va = shader_va,
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
      .shader_binary_size = linked->b.size,
      .scratch_size = scratch_size,
      .nr_gprs = linked->b.nr_gprs,
      .compiler_d_valid = true,
      .compiler_d = (uint32_t)preamble_cs->b.info.agx3_staging_halfs * 2,
      .threadgroup_memory_size_B = local_B,
   };
   struct agx3_g17p_graphics_program program;
   struct agx3_g17p_native_compute_program_desc launch_desc = {
      .program_va = reservation.va,
      .shader_va = shader_va,
      .launch_sources = launch_sources,
   };
   struct agx3_compute_program launch;
   if (!hk_m5_compute_program_generate(dev, &desc, &program) ||
       !hk_m5_compute_launch_bind(dev, &launch_desc, &launch))
      return false;

   memcpy(reservation.map, program.bytes, program.length);
   if (!hk_cmd_publish_ioto_program(cmd, &reservation, program.length))
      return false;

   hk_cmd_buffer_retain_bo(cmd, linked->b.bo);
   if (preamble_cs->b.info.rodata.size_16)
      hk_cmd_buffer_retain_bo(cmd, preamble_cs->bo);

   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   struct agx3_compute_dispatch_sink sink = {
      .data = (uint8_t *)cs->current,
      .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
   };
   if (!agx3_m5_native_compute_dispatch_append(&launch, &direct, 5, &sink))
      return false;

   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   cs->strong_flush_end = cs->current;
   cs->memory_flush_end = cs->current;
   cs->weak_barrier_end = NULL;
   hk_m5_compute_resources(cs, desc.scratch_size, 0,
                           desc.threadgroup_memory_size_B, 1);
   cs->stats.cmds++;
   return true;
}


__attribute__((hot)) void
hk_dispatch_m5_native(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                      struct hk_shader *s, struct agx_grid grid,
                      struct agx_workgroup local_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *linked = s ? s->only_linked : NULL;
   if (s && !linked && s->info.stage == MESA_SHADER_VERTEX)
      linked = cmd->state.gfx.linked[MESA_SHADER_VERTEX];
   if (!agx_chip_g17_m5(dev->dev.chip) || cs->type != HK_CS_CDM ||
       /* Static TPS addresses are 15-bit halfword offsets. */
       !hk_g17p_compute_shader_supported(dev->dev.chip, s, linked, 1u << 16) ||
       ((s->b.info.scratch_size || linked->b.scratch_size) &&
        !(dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_RESOURCES)) ||
       s->b.info.uses_ray_traversal || s->info.agx3_txf_texture_valid ||
       s->info.agx3_storage_image_valid || s->info.agx3_sampled_storage_valid ||
       s->info.agx3_indexed_storage_valid ||
       !local_size.x || !local_size.y || !local_size.z ||
       (uint64_t)local_size.x * local_size.y * local_size.z >
          dev->vk.physical->properties.maxComputeWorkGroupInvocations)
      goto refused;
   if (!agx_is_indirect(grid) &&
       (!grid.count[0] || !grid.count[1] || !grid.count[2]))
      return;

   if (s->b.info.uses_matrix16) {
      if (!(dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_MATRIX))
         goto refused;
      cs->m5_matrix_ops = true;
   }
   const bool internal = s->info.stage != MESA_SHADER_COMPUTE;
   if (internal && s->info.stage != MESA_SHADER_VERTEX && s->data_size)
      goto refused;
   if (!internal) {
      cmd->state.cs.descriptors.root.cs.constant_data_addr = s->data_size
         ? hk_pool_upload(cmd, s->data_ptr, s->data_size, HK_MIN_UBO_ALIGNMENT) : 0;
      if (s->data_size && !cmd->state.cs.descriptors.root.cs.constant_data_addr)
         return;
   }
   const uint64_t root = internal ? cmd->state.gfx.root :
      hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
   if (!root)
      return;
   const uint64_t texture_pointers = s->info.agx3_indexed_sampler_valid
      ? dev->rodata.indexed_heap_pointers : 0;
   if (s->info.agx3_indexed_sampler_valid && !texture_pointers)
      goto refused;
   struct agx3_ioto_uniform_range ranges[HK_G17P_COMPUTE_MAX_RANGES];
   size_t count = 0;
   const bool collected = internal
      ? hk_g17p_internal_graphics_uniform_ranges(cmd, s, linked, root, texture_pointers, ranges, &count)
      : hk_g17p_compute_uniform_ranges(dev->dev.chip, cmd, s, root, texture_pointers, ranges, &count);
   if (!collected)
      goto refused;
   if (!internal && s->b.info.agx3_uniform_dma_count &&
       !hk_m5_compute_uniform_dma_ranges(cmd, s, root, ranges,
                                         ARRAY_SIZE(ranges), &count))
      goto refused;
   size_t capacity;
   if (!agx3_g17g_graphics_program_bound(dev->dev.chip, ranges, count, &capacity))
      goto refused;
   if (s->b.info.has_preamble)
      capacity = AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE;
   struct hk_ioto_reservation reservation;
   if (!hk_cmd_reserve_ioto_program(cmd, capacity, "M5 compute", &reservation))
      return;
   const uint64_t shader_va = linked->b.bo->va->addr + linked->b.entry_offset;
   struct agx3_g17p_graphics_program_desc desc = {
      .program_va = reservation.va, .shader_va = shader_va,
      .uniform_ranges = ranges, .uniform_range_count = count,
      .shader_binary_size = linked->b.size,
      .nr_gprs = MAX2(linked->b.nr_gprs, s->b.info.nr_preamble_gprs),
      /* The constant program's TPS frame is the launch's thread-invariant
       * spill buffer (below), not the main program's per-thread frame. */
      .scratch_size = MAX3(s->b.info.scratch_size, linked->b.scratch_size,
                           linked->b.agx3_spectpr_scratch_size),
      .compiler_d_valid = s->b.info.agx3_staging_known,
      .compiler_d = MAX2((uint32_t)s->b.info.agx3_staging_halfs * 2,
                         (uint32_t)s->b.info.agx3_preamble_staging_halfs * 2),
      .threadgroup_memory_size_B = s->b.info.local_size
         ? MAX2(256u, ALIGN_POT(s->b.info.local_size, 64u)) : 0,
   };
   if (s->b.info.has_preamble) {
      if (internal || !s->bo || !s->bo->va ||
          !hk_agx3_g17p_constant_program_from_binary(
             s->code_ptr, s->code_size, s->b.info.preamble_offset,
             s->b.info.preamble_size, hk_shader_code_va(s),
             s->b.info.binary_size, &desc.constant_program))
         goto refused;
      desc.constant_program.force_external = true;
      desc.constant_program.absolute_branch = true;
      desc.constant_program.request_prelude = s->b.info.agx3_preamble_uniform_inputs;
      if (s->b.info.agx3_preamble_tps_inputs)
         desc.cprog_inputs = &s->b.info.agx3_preamble_inputs;
   }
   hk_m5_compute_program_desc_set_device(dev, &desc);
   struct agx3_g17p_graphics_program program;
   const struct hk_m5_compute_loader *loader =
      hk_m5_loader_get(dev, linked, &desc);
   struct agx3_g17p_native_compute_program_desc launch_desc = {
      .program_va = reservation.va, .shader_va = shader_va,
   };
   struct agx3_compute_program launch;
   if (loader) {
      program.length = loader->length;
      memcpy(program.bytes, loader->bytes, loader->length);
      hk_g17p_patch_plan_apply(&loader->plan, ranges, program.bytes);
      if (dev->dev.debug & AGX_DBG_VERIFY) {
         struct agx3_g17p_graphics_program fresh;
         if (!hk_m5_compute_program_generate(dev, &desc, &fresh) ||
             fresh.length != program.length ||
             memcmp(fresh.bytes, program.bytes, program.length)) {
            mesa_loge("M5 cached compute launch differs from fresh composition");
            goto refused;
         }
      }
   } else if (!hk_m5_compute_program_generate(dev, &desc, &program)) {
      goto refused;
   }
   if (!agx3_cdm_direct_template_sources_compute(s->b.info.uses_resource_info_query,
                                                 false, false, &launch_desc.launch_sources))
      goto refused;
   /* A constant program's TPS frame is the launch's thread-invariant spill
    * buffer (word 1 bits 3..20): the loader publishes the CProg's inputs to
    * TPS from half 8 and the CProg's RLDs read them back from that frame.
    * With a zero-sized frame the RLDs never write their destinations. Native
    * M5 launches of TPS-importing CProgs size it as the 16-byte-aligned input
    * frame (32 B for 4 and 8 input halves, 48 B for 16 halves), which is
    * agx3_cprog_inputs::storage_bytes; the CProg's own TPS spills share the
    * frame, so cover preamble_scratch_size too. */
   if (s->b.info.has_preamble) {
      const unsigned cprog_frame =
         ALIGN_POT(MAX2((unsigned)s->b.info.agx3_preamble_inputs.storage_bytes,
                        s->b.info.preamble_scratch_size), 16);
      launch_desc.launch_sources.ti_spill_bytes =
         MAX2(launch_desc.launch_sources.ti_spill_bytes, cprog_frame);
   }
   if (!hk_m5_compute_launch_bind(dev, &launch_desc, &launch))
      goto refused;
   memcpy(reservation.map, program.bytes, program.length);
   if (!hk_cmd_publish_ioto_program(cmd, &reservation, program.length))
      return;
   hk_cmd_buffer_retain_bo(cmd, linked->b.bo);
   if (s->b.info.rodata.size_16 || s->b.info.has_preamble)
      hk_cmd_buffer_retain_bo(cmd, s->bo);

   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   struct agx3_compute_dispatch_sink sink = {
      .data = (uint8_t *)cs->current,
      .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
   };
   const struct agx3_compute_direct_grid direct = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {grid.count[0], grid.count[1], grid.count[2]},
      .local_size = {local_size.x, local_size.y, local_size.z},
      .dimensions = grid.count[2] != 1 || local_size.z != 1 ? 3
                  : grid.count[1] != 1 || local_size.y != 1 ? 2 : 1,
   };
   const uint32_t local[3] = {local_size.x, local_size.y, local_size.z};
   if (cmd->m5_rt_capture) {
      /* Single-kick vkCmdTraceRays* (hk_cmd_trace_rays.c): produce this
       * kernel's direct launch plus its trailing barrier as a template the
       * select kernel copies into the stream slot, instead of appending it
       * here. Template it as a multi-dimensional grid so word 0 keeps the
       * 2D/3D bits; the select kernel clears them for a 1D launch exactly
       * as agx3_g17p_native_compute_launch_pack does, and writes the sizes. */
      const struct agx3_compute_direct_grid tmpl = {
         .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
         .size = {local[0], 2 * local[1], 2 * local[2]},
         .local_size = {local[0], local[1], local[2]},
         .dimensions = 3,
      };
      struct agx3_compute_dispatch_sink capture = {
         .data = cmd->m5_rt_capture,
         .capacity = AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE,
      };
      if (!agx3_m5_native_compute_dispatch_append(&launch, &tmpl, 0, &capture) ||
          capture.length != AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE)
         goto refused;
      cmd->m5_rt_captured = true;
      hk_m5_compute_resources(cs, desc.scratch_size, 0,
                              desc.threadgroup_memory_size_B,
                              local[0] * local[1] * local[2]);
      cs->stats.cmds++;
      return;
   }
   /* Two application dispatches in a row need no barrier between them: drop
    * the previous one's weak tail, which nothing else has followed or
    * strengthened, and let this launch's own tail end the stream instead.
    * The two then run side by side; a barrier recorded after them waits for
    * both. M5 indirect dispatches are single hardware records with the same
    * weak tail and obey the same Vulkan barrier rules, so they can overlap
    * with adjacent application dispatches too. Ray-tracing pipeline kernels
    * keep their barrier, and so does a kernel with a large per-thread scratch
    * frame: it runs slower the more of its threads are in flight, so it stays
    * alone. */
   const bool app_dispatch =
      !cmd->in_meta && !internal &&
      !cmd->state.cs.descriptors.root.cs.trace_rays_addr &&
      desc.scratch_size <= HK_M5_OVERLAP_MAX_SCRATCH_B &&
      agx_chip_elide_app_dispatch_barrier(dev->dev.chip);
   if (app_dispatch && cs->app_dispatch_end &&
       cs->current == cs->app_dispatch_end &&
       cs->current == cs->weak_barrier_end) {
      cs->current = (uint32_t *)((uint8_t *)cs->current - AGX3_CDM_BARRIER_SIZE);
      sink.data = (uint8_t *)cs->current;
      sink.capacity += AGX3_CDM_BARRIER_SIZE;
   }
   const bool appended = agx_is_indirect(grid)
      ? agx3_m5_native_indirect_dispatch_append(
           &launch, grid.ptr,
           grid.mode == AGX_CDM_MODE_INDIRECT_GLOBAL ? local : NULL, 0, &sink)
      : agx3_m5_native_compute_dispatch_append(&launch, &direct, 0, &sink);
   if (!appended)
      goto refused;
   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   cs->weak_barrier_end = cs->current;
   cs->app_dispatch_end = app_dispatch ? cs->current : NULL;
   hk_m5_compute_resources(cs, desc.scratch_size, 0,
                           desc.threadgroup_memory_size_B,
                           local[0] * local[1] * local[2]);
   cs->stats.cmds++;
   return;
refused:
   mesa_loge_once("M5 compute launch exceeds the current resource profile");
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
}

void
hk_dispatch_g17p_native(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        struct hk_shader *s, struct agx_grid grid,
                        struct agx_workgroup local_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *linked = s ? s->only_linked : NULL;
   if (s && !linked && s->info.stage == MESA_SHADER_VERTEX)
      linked = cmd->state.gfx.linked[MESA_SHADER_VERTEX];
   static_assert(HK_MAX_SETS == HK_AGX3_G17P_MAX_ROOT_SETS,
                 "G17P root range must cover every Honeykrisp set");

   if (!agx_chip_native_agx3(dev->dev.chip) || cs->type != HK_CS_CDM ||
       !hk_g17p_compute_shader_supported(dev->dev.chip, s, linked,
                                         AGX3_G17P_MAX_COMPUTE_SPILL_B)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   const bool internal_graphics = s->info.stage != MESA_SHADER_COMPUTE;
   if (s->b.info.uses_ray_traversal && dev->gen1_native_rq) {
      bool valid = !internal_graphics || dev->gen1_native_graphics_rq;
      struct hk_gen1_rt_indirect_contract tes_contract;
      if (agx_is_indirect(grid) && internal_graphics &&
          s->info.stage == MESA_SHADER_TESS_EVAL) {
         /* Poly's TES-before-GS kernel (hk_launch_gs_prerast()) always has
          * a driver-written indirect grid, since the tessellator decides its
          * size. It is an ordinary 64x1x1 graphics kernel: its query context
          * is the direct one and SCS slots are addressed by resident
          * warp/lane, independent of the grid. Application indirect query
          * dispatches keep the trace-rays contract below. */
         valid = valid && dev->gen1_rq_instances && dev->gen1_rq_candidates &&
            grid.mode == AGX_CDM_MODE_INDIRECT_LOCAL &&
            local_size.x == 64 && local_size.y == 1 && local_size.z == 1 &&
            hk_gen1_rt_indirect_contract_init(hk_gen1_ray_query_scs_layout(dev),
                                              &tes_contract);
      } else if (agx_is_indirect(grid)) {
         struct hk_gen1_rt_indirect_contract contract;
         struct agx_precompiled_shader *setup = NULL;
         /* The shared native indirect helper already publishes a private
          * six-u32 geometry block. Admit only RT's32x1x1 profile and the
          * validated Gen1 resident window under an explicit diagnostic. */
         valid = valid &&
            dev->gen1_rq_instances && dev->gen1_rq_candidates &&
            grid.mode == AGX_CDM_MODE_INDIRECT_GLOBAL &&
            local_size.x == 32 && local_size.y == 1 && local_size.z == 1 &&
            cmd->state.cs.descriptors.root.cs.trace_rays_addr != 0 &&
            cmd->state.cs.descriptors.root.cs.grid_helper_sparse_map_addr == 0 &&
            hk_gen1_rt_indirect_contract_init(hk_gen1_ray_query_scs_layout(dev), &contract);
         if (valid)
            setup = agx_get_precompiled(&dev->bg_eot, LIBAGX_TRACE_RAYS_INDIRECT_SETUP);
         valid = valid && setup && setup->info.main_size && setup->info.nr_gprs &&
            setup->info.agx3_staging_known && setup->info.push_count >= 26 &&
            setup->info.workgroup_size[0] == 1 && setup->info.workgroup_size[1] == 1 &&
            setup->info.workgroup_size[2] == 1 && setup->info.nr_preamble_gprs == 0 &&
            setup->info.preamble_size == 0 && setup->info.local_size == 0 &&
            setup->info.agx3_tps_size == 0 && setup->info.printf_uniform == 0;
      } else {
         valid = valid && grid.count[0] && grid.count[1] && grid.count[2];
      }
      if (!valid) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
   }

   /* Only an application compute dispatch can name a sparse argument buffer;
    * an internal graphics kernel's indirect grid is a driver record. */
   const uint64_t sparse_map_addr =
      internal_graphics
         ? 0
         : cmd->state.cs.descriptors.root.cs.grid_helper_sparse_map_addr;

   const bool indirect = agx_is_indirect(grid);
   if (indirect) {
      if (!hk_g17p_indirect_launch_supported(dev, grid, local_size,
                                             sparse_map_addr)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
   } else if (grid.count[0] == 0 || grid.count[1] == 0 ||
              grid.count[2] == 0) {
      return;
   }

   if (!internal_graphics && s->data_size != 0) {
      const uint64_t constant_data =
         hk_cmd_upload_constants(cmd, s, MESA_SHADER_COMPUTE);
      if (!constant_data) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }

      cmd->state.cs.descriptors.root.cs.constant_data_addr = constant_data;
   } else if (!internal_graphics) {
      cmd->state.cs.descriptors.root.cs.constant_data_addr = 0;
   }

   const uint64_t root_ptr = internal_graphics
      ? cmd->state.gfx.root
      : hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
   if (!root_ptr) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }

   uint64_t texture_pointers = 0;
   /* Poly geometry kernels use graphics descriptor roots but execute through
    * CDM. Their indexed samples consume the same device-global texture and
    * sampler heap pair as compute. Other internal-graphics resource profiles
    * still need their own root contracts. */
   if (internal_graphics &&
       (s->info.agx3_txf_texture_valid || s->info.agx3_storage_image_valid ||
        s->info.agx3_sampled_storage_valid ||
        s->info.agx3_indexed_storage_valid)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (s->info.agx3_txf_texture_valid || s->info.agx3_storage_image_valid ||
       s->info.agx3_sampled_storage_valid ||
       s->info.agx3_indexed_storage_valid ||
       s->info.agx3_indexed_sampler_valid) {
      const struct hk_descriptor_state *descriptors = internal_graphics
         ? &cmd->state.gfx.descriptors
         : &cmd->state.cs.descriptors;

      if (s->info.agx3_indexed_sampler_valid) {
         if (!dev->sampled_images.base_va ||
             !dev->samplers.table.bo || !dev->samplers.table.bo->va) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         uint64_t pointers[4] = {
            dev->sampled_images.base_va,
            dev->samplers.table.bo->va->addr,
         };
         unsigned pointer_count = 2;
         if (s->info.agx3_indexed_storage_valid) {
            const struct agx_g17p_indexed_storage_binding *binding =
               &s->info.agx3_indexed_storage;
            if (binding->pbe_set >= HK_MAX_SETS) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }
            const uint64_t set_address =
               descriptors->root.sets[binding->pbe_set];
            const uint32_t set_size =
               descriptors->set_sizes[binding->pbe_set];
            if (!set_address || binding->pbe_offset_B > set_size ||
                set_size - binding->pbe_offset_B < 32) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }
            const uint64_t write_control = hk_pool_upload(
               cmd, hk_g17p_native_write_control,
               sizeof(hk_g17p_native_write_control), 32);
            if (!write_control) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_OUT_OF_DEVICE_MEMORY);
               return;
            }
            const struct hk_agx3_g17p_indexed_storage_heap heap = {
               .texture_heap_va = pointers[0],
               .sampler_heap_va = pointers[1],
               .pbe_descriptor_set_va = set_address,
               .write_control_va = write_control,
               .pbe_offset_B = binding->pbe_offset_B,
            };
            if (!hk_agx3_g17p_indexed_storage_pointers(&heap, pointers)) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }
            pointer_count = 4;
         }
         texture_pointers = hk_pool_upload(
            cmd, pointers, pointer_count * sizeof(pointers[0]), 8);
         if (!texture_pointers) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }
      } else if (s->info.agx3_sampled_storage_valid) {
         const struct agx_g17p_sampled_storage_binding *binding =
            &s->info.agx3_sampled_storage;
         if (binding->texture_set >= HK_MAX_SETS ||
             binding->pbe_set >= HK_MAX_SETS) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         const uint64_t texture_set_address =
            descriptors->root.sets[binding->texture_set];
         const uint64_t pbe_set_address =
            descriptors->root.sets[binding->pbe_set];
         const uint32_t texture_set_size =
            descriptors->set_sizes[binding->texture_set];
         const uint32_t pbe_set_size =
            descriptors->set_sizes[binding->pbe_set];
         struct hk_descriptor_set *texture_set =
            descriptors->sets[binding->texture_set];
         if (!texture_set_address || !pbe_set_address || !texture_set ||
             !texture_set->mapped_ptr ||
             binding->texture_offset_B > texture_set_size ||
             texture_set_size - binding->texture_offset_B < 32 ||
             binding->sampler_index_offset_B > texture_set->size ||
             texture_set->size - binding->sampler_index_offset_B <
                sizeof(uint16_t) ||
             binding->pbe_offset_B > pbe_set_size ||
             pbe_set_size - binding->pbe_offset_B < 32 ||
             !dev->samplers.table.bo || !dev->samplers.table.bo->va) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }

         uint16_t sampler_index;
         memcpy(&sampler_index,
                (const uint8_t *)texture_set->mapped_ptr +
                   binding->sampler_index_offset_B,
                sizeof(sampler_index));
         sampler_index = hk_sampler_heap_index(
            texture_set->layout->native_sampler_indices, sampler_index);
         const uint64_t write_control = hk_pool_upload(
            cmd, hk_g17p_native_write_control,
            sizeof(hk_g17p_native_write_control), 32);
         if (!write_control) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }
         const struct hk_agx3_g17p_sampled_storage_heap heap = {
            .sampled = {
               .descriptor_set_va = texture_set_address,
               .sampler_heap_va = dev->samplers.table.bo->va->addr,
               .texture_offset_B = binding->texture_offset_B,
               .sampler_index = sampler_index,
               .sampler_count = dev->samplers.table.alloc,
               .sampler_element_size_B =
                  agx_chip_sampler_heap_element_size(dev->dev.chip),
            },
            .pbe_descriptor_set_va = pbe_set_address,
            .write_control_va = write_control,
            .pbe_offset_B = binding->pbe_offset_B,
         };
         uint64_t pointers[4] = {0};
         if (!hk_agx3_g17p_sampled_storage_pointers(&heap, pointers)) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         /* See hk_g17p_texture_state_pointer(). */
         pointers[0] = hk_g17p_texture_state_pointer(
            cmd, pointers[0],
            (const uint8_t *)texture_set->mapped_ptr +
               binding->texture_offset_B,
            texture_set->size > binding->texture_offset_B
               ? texture_set->size - binding->texture_offset_B
               : 0);
         if (!pointers[0]) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }

         texture_pointers = hk_pool_upload(cmd, pointers, sizeof(pointers), 8);
         if (!texture_pointers) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }
      } else {
      const uint64_t set_address = descriptors->root.sets[0];
      const uint32_t set_size = descriptors->set_sizes[0];
      const uint32_t offset = s->info.agx3_storage_image_valid
         ? s->info.agx3_storage_image.texture_offset_B
         : s->info.agx3_txf_texture.texture_offset_B;
      if (!set_address || offset > set_size ||
          set_size - offset <
             (s->info.agx3_storage_image_valid ? 64 : 32) ||
          !dev->rodata.bo ||
          !dev->rodata.bo->va) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      uint64_t pointers[4] = {0};
      if (s->info.agx3_combined_sampler_valid) {
         struct hk_descriptor_set *set = descriptors->sets[0];
         const struct agx_g17p_combined_sampler_binding *binding =
            &s->info.agx3_combined_sampler;
         if (!set || !set->mapped_ptr ||
             binding->sampler_index_offset_B > set->size ||
             set->size - binding->sampler_index_offset_B < sizeof(uint16_t)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }

         uint16_t sampler_index;
         memcpy(&sampler_index,
                (const uint8_t *)set->mapped_ptr +
                   binding->sampler_index_offset_B,
                sizeof(sampler_index));
         sampler_index = hk_sampler_heap_index(
            set->layout->native_sampler_indices, sampler_index);
         if (!dev->samplers.table.bo || !dev->samplers.table.bo->va) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         const struct hk_agx3_g17p_combined_sampler_heap heap = {
            .descriptor_set_va = set_address,
            .sampler_heap_va = dev->samplers.table.bo->va->addr,
            .texture_offset_B = binding->texture_offset_B,
            .sampler_index = sampler_index,
            .sampler_count = dev->samplers.table.alloc,
            .sampler_element_size_B =
               agx_chip_sampler_heap_element_size(dev->dev.chip),
         };
         if (!hk_agx3_g17p_combined_sampler_pointers(&heap, pointers)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         /* See hk_g17p_texture_state_pointer(). */
         pointers[0] = hk_g17p_texture_state_pointer(
            cmd, pointers[0],
            (const uint8_t *)set->mapped_ptr + binding->texture_offset_B,
            set->size > binding->texture_offset_B
               ? set->size - binding->texture_offset_B
               : 0);
         if (!pointers[0]) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }
      } else if (s->info.agx3_storage_image_valid) {
         /* Exact access controls from native G17P Tier-2 argument buffers.
          * A write-only resource is a PBE pointer followed by write-control;
          * read_write instead starts with texture/PBE and points at the
          * paired read/write controls in its third slot. */
         static const uint8_t native_read_write_control[32] = {
            0x68, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x03, 0x1b,
            0x08, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
         };
         const bool write_only = s->info.agx3_storage_image.write_only;
         const uint64_t access_control =
            hk_pool_upload(cmd, write_only ? hk_g17p_native_write_control
                                           : native_read_write_control,
                           sizeof(native_read_write_control), 32);
         if (!access_control) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }

         /* A read/write image's read takes sampler 0 from halves 4..7:
          * the texel-fetch sampler, as a read-only texel fetch does. */
         if (!hk_g17p_storage_image_pointers(
                set_address, s->info.agx3_storage_image.texture_offset_B,
                s->info.agx3_storage_image.pbe_offset_B, write_only,
                access_control, dev->rodata.bo->va->addr, pointers)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
      } else {
         /* Native G17P compute read() uses table pointers in uniform
          * halves0..7, with ordinary buffer arguments beginning at half8. */
         pointers[0] = set_address + offset;
         pointers[1] = dev->rodata.bo->va->addr;

         /* Unlike the indexed ABI, direct txf publishes the descriptor-set
          * texture word itself.  G17P ignores the low five pointer bits, so
          * snapshot an unaligned record just as the combined-sampler paths do.
          * The shader selector admits such a snapshot only when sampled-image
          * update-after-bind is disabled; an aligned live set pointer is safe
          * with either feature state. */
         if (pointers[0] & (HK_G17P_TEXTURE_STATE_ALIGN - 1)) {
            struct hk_descriptor_set *set = descriptors->sets[0];
            const void *set_map = set ? set->mapped_ptr : NULL;
            uint32_t mapped_size = set ? set->size : 0;
            if (!set && descriptors->push[0] &&
                !(descriptors->buffer_active & BITFIELD_BIT(0))) {
               set_map = descriptors->push[0]->data;
               mapped_size = sizeof(descriptors->push[0]->data);
            }

            if (dev->vk.enabled_features
                   .descriptorBindingSampledImageUpdateAfterBind ||
                !set_map || offset > mapped_size) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }

            pointers[0] = hk_g17p_texture_state_pointer(
               cmd, pointers[0], (const uint8_t *)set_map + offset,
               mapped_size - offset);
            if (!pointers[0]) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_OUT_OF_DEVICE_MEMORY);
               return;
            }
         }
      }
      const size_t pointer_count =
         s->info.agx3_storage_image_valid ? 4 : 2;
      texture_pointers =
         hk_pool_upload(cmd, pointers, pointer_count * sizeof(pointers[0]), 8);
      if (!texture_pointers) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
      }
   }

   struct agx3_ioto_uniform_range
      ranges[HK_G17P_COMPUTE_MAX_RANGES];
   size_t range_count;
   uint64_t program_va;
   struct hk_g17p_compute_program_ref program;
   if (!hk_g17p_build_compute_program(
          cmd, cs, s, linked, root_ptr, texture_pointers, local_size,
          internal_graphics,
          indirect ? AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1
                   : AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
          "main", ranges, &range_count, &program_va, &program))
      return;

   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   struct agx3_compute_dispatch_sink sink = {
      .data = (uint8_t *)cs->current,
      .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
   };

   if (indirect) {
      if (!hk_g17p_append_indirect_launch(cmd, cs, grid, local_size,
                                          sparse_map_addr, &program, &sink))
         return;

      cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
      /* The indirect pair ends in the deferred launch's own weak ordering
       * barrier (agx3_g17p_native_indirect_compute_dispatch_append passes
       * flags 0).  Record it exactly as the direct path does, so a following
       * hk_cdm_cache_flush() strengthens that record in place instead of
       * appending a second barrier after it: two adjacent barriers with no
       * launch between them is a shape Apple's CDM emitter never produces and
       * the G17P stream oracle rejects. */
      cs->weak_barrier_end = cs->current;
      cs->app_dispatch_end = NULL;
      cs->stats.cmds++;
      return;
   }

   const uint8_t dimensions =
      grid.count[2] != 1 || local_size.z != 1
         ? 3
         : grid.count[1] != 1 || local_size.y != 1 ? 2 : 1;
   const struct agx3_compute_direct_grid direct = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {grid.count[0], grid.count[1], grid.count[2]},
      .local_size = {local_size.x, local_size.y, local_size.z},
      .dimensions = dimensions,
   };

   /* Every launch already ends in a barrier record; the only open question is
    * its flags. This path passes 0, a barrier that orders the launches but
    * does not make the producer's writes visible to the consumer. Every other
    * compute path in the driver flushes: hk_dispatch_with_usc_launch() calls
    * hk_cdm_cache_flush() after each AGX2 launch, and hk_dispatch_precomp_g17p()
    * passes Apple's witnessed strong value 5 for AGX_BARRIER_ALL. So the
    * "flush after every launch" policy hk_cdm_cache_flush() documents is not
    * actually held on the one path every API dispatch and every vk_meta copy,
    * fill and buffer update on G17P goes through.
    *
    * A missing write->read dependency here is invisible until two dispatches
    * in one CDM stream are producer and consumer, which is exactly the shape
    * of the G17P meta copy paths: the compressed copies emit one dispatch per
    * block with nothing in between, and merge_control_streams() concatenates
    * adjacent CDM streams at hk_EndCommandBuffer, so independent copies share
    * a stream too.
    *
    * Whether this weak shape explains the Hyprland GMMU read fault that
    * ZINK_DEBUG=sync cures is unproven.
    */
   const uint8_t barrier_flags = 0;

   /* Two application dispatches in a row need no barrier between them: drop
    * the previous one's weak tail, which nothing else has followed or
    * strengthened, and let this launch's own tail end the stream instead.
    * Indirect launches, ray-tracing kernels, and kernels with a large
    * per-thread scratch frame keep their barrier. */
   const bool app_dispatch =
      s && s->info.stage == MESA_SHADER_COMPUTE &&
      !s->b.info.uses_ray_traversal &&
      !cmd->state.cs.descriptors.root.cs.trace_rays_addr &&
      ((!cmd->in_meta && linked &&
        MAX3(s->b.info.scratch_size, linked->b.scratch_size,
             linked->b.agx3_spectpr_scratch_size) <=
           HK_M5_OVERLAP_MAX_SCRATCH_B &&
        agx_chip_elide_app_dispatch_barrier(dev->dev.chip)) ||
       /* An acceleration-structure build orders every launch that depends on
        * another with a pipeline barrier, which strengthens the earlier
        * launch's tail; launches with none between them (one step of several
        * builds, the fills after the leaves) run side by side. */
       cmd->state.as_build_recording);
   if (app_dispatch && cs->app_dispatch_end &&
       cs->current == cs->app_dispatch_end &&
       cs->current == cs->weak_barrier_end) {
      cs->current = (uint32_t *)((uint8_t *)cs->current - AGX3_CDM_BARRIER_SIZE);
      sink.data = (uint8_t *)cs->current;
      sink.capacity += AGX3_CDM_BARRIER_SIZE;
   }

   if (!hk_agx3_g17p_native_compute_dispatch_append_with_barrier(
          &program.launch, &direct, barrier_flags, &sink)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   if (barrier_flags == 0x5) {
      cs->strong_flush_end = cs->current;
      cs->weak_barrier_end = NULL;
   } else {
      cs->weak_barrier_end = cs->current;
   }
   cs->app_dispatch_end = app_dispatch ? cs->current : NULL;
   cs->stats.cmds++;
}

void
hk_dispatch_g17g_native(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        struct hk_shader *s, struct agx_grid grid,
                        struct agx_workgroup local_size)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *linked = s ? s->only_linked : NULL;
   if (s && !linked && s->info.stage == MESA_SHADER_VERTEX)
      linked = cmd->state.gfx.linked[MESA_SHADER_VERTEX];
   const bool g18 = dev->dev.chip == AGX_CHIP_G18G;
   const bool internal = s && s->info.stage != MESA_SHADER_COMPUTE;
   const uint32_t scratch_bytes =
      (s && linked)
         ? MAX3(s->b.info.scratch_size, linked->b.scratch_size,
                MAX2(s->b.info.agx3_spectpr_scratch_size,
                     linked->b.agx3_spectpr_scratch_size))
         : 0;
   const uint32_t max_spill = g18 ? (1u << 16) : AGX3_G17P_MAX_SPILL_B;
   const bool g18_indirect = g18 && agx_is_indirect(grid);
   const bool sparse_indirect =
      g18_indirect && !internal &&
      cmd->state.cs.descriptors.root.cs.grid_helper_sparse_map_addr != 0;
   if ((!g18 && dev->dev.chip != AGX_CHIP_G17G) || cs->type != HK_CS_CDM ||
       (g18 && cs->g18_dispatch_sealed) ||
       !hk_g17p_compute_shader_supported(dev->dev.chip, s, linked, max_spill) ||
       (!g18 && (internal || s->b.info.local_size)) ||
       (internal && s->info.stage != MESA_SHADER_VERTEX && s->data_size) ||
       (!g18 && agx_is_indirect(grid)) ||
       (g18_indirect && (!grid.ptr || sparse_indirect)) ||
       s->b.info.uses_ray_traversal ||
       (g18 && (s->b.info.nr_preamble_gprs ||
                scratch_bytes > max_spill ||
                s->b.info.local_size > 32768u)) ||
       (g18 && s->info.g18_unlowered_image) ||
       s->info.agx3_txf_texture_valid ||
       (s->info.agx3_storage_image_valid && !g18) ||
       s->info.agx3_sampled_storage_valid || s->info.agx3_indexed_storage_valid ||
       (grid.mode != AGX_CDM_MODE_INDIRECT_LOCAL &&
        (!local_size.x || !local_size.y || !local_size.z ||
         (uint64_t)local_size.x * local_size.y * local_size.z > 1024))) {
      mesa_loge(
         "AGX3 native compute: unsupported shader or indirect/shared-memory launch"
         " (stage=%u indirect=%u scratch %u/%u, spectpr scratch %u, preamble GPRs %u,"
         " unlowered image access %d, shared memory %u B)",
         s ? s->info.stage : 0, agx_is_indirect(grid),
         s ? s->b.info.scratch_size : 0, linked ? linked->b.scratch_size : 0,
         s ? s->b.info.agx3_spectpr_scratch_size : 0,
         s ? s->b.info.nr_preamble_gprs : 0,
         s ? s->info.g18_unlowered_image : 0, s ? s->b.info.local_size : 0);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (!agx_is_indirect(grid) &&
       (!grid.count[0] || !grid.count[1] || !grid.count[2]))
      return;

   struct hk_descriptor_state *descriptors =
      internal ? &cmd->state.gfx.descriptors : &cmd->state.cs.descriptors;
   if (!internal) {
      descriptors->root.cs.constant_data_addr = s->data_size
         ? hk_pool_upload(cmd, s->data_ptr, s->data_size, HK_MIN_UBO_ALIGNMENT)
         : 0;
      if (s->data_size && !descriptors->root.cs.constant_data_addr) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
   }
   const uint64_t root = internal
      ? cmd->state.gfx.root
      : hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_COMPUTE);
   uint64_t texture_pointers = 0;
   if (s->info.agx3_indexed_sampler_valid) {
      /* The sampled-image heap may be a chunked table (no single BO): its
       * fixed base_va is the heap address, as on the M5 path. */
      if (!dev->sampled_images.base_va ||
          !dev->samplers.table.bo || !dev->samplers.table.bo->va) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      const uint64_t pointers[] = {
         dev->sampled_images.base_va,
         dev->samplers.table.bo->va->addr,
      };
      texture_pointers = hk_pool_upload(cmd, pointers, sizeof(pointers), 8);
      if (dev->sampled_images.bo)
         hk_cmd_buffer_retain_bo(cmd, dev->sampled_images.bo);
      hk_cmd_buffer_retain_bo(cmd, dev->samplers.table.bo);
   }
   if (g18 && s->info.agx3_storage_image_valid &&
       s->info.agx3_g18_tables_valid) {
      /* Textures + storage images: build the texture-state and sampler
       * tables from the bound sets (agx_nir_lower_g18g_combined_tables). */
      const struct agx_g18g_table_binding *t = &s->info.agx3_g18_tables;
      uint8_t tex[AGX_G18G_TABLE_MAX_TEXTURES][32];
      uint8_t smp[AGX_G18G_TABLE_MAX_SAMPLERS][32];
      memset(smp, 0, sizeof(smp));
      memcpy(smp[0], hk_g18g_write_state1, sizeof(hk_g18g_write_state1));
      bool ok =
         t->texture_count && t->texture_count <= AGX_G18G_TABLE_MAX_TEXTURES &&
         t->sampler_count < AGX_G18G_TABLE_MAX_SAMPLERS &&
         (!t->sampler_count ||
          (dev->samplers.table.map && dev->samplers.table.desc_size == 32));
      for (unsigned i = 0; ok && i < t->texture_count; ++i) {
         const struct agx_g18g_table_entry *e = &t->textures[i];
         const struct hk_descriptor_set *set =
            e->set < HK_MAX_SETS ? descriptors->sets[e->set] : NULL;
         /* The tables snapshot set memory when the dispatch is recorded:
          * refuse sets that may legally change after binding. */
         if (!set || !set->mapped_ptr || e->offset_B > set->size ||
             set->size - e->offset_B < 32 ||
             (set->layout->vk.flags &
              VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT))
            ok = false;
         else
            memcpy(tex[i], (const uint8_t *)set->mapped_ptr + e->offset_B, 32);
      }
      for (unsigned i = 0; ok && i < t->sampler_count; ++i) {
         const struct agx_g18g_table_entry *e = &t->samplers[i];
         const struct hk_descriptor_set *set =
            e->set < HK_MAX_SETS ? descriptors->sets[e->set] : NULL;
         uint16_t published;
         if (!set || !set->mapped_ptr || e->offset_B > set->size ||
             set->size - e->offset_B < sizeof(published) ||
             (set->layout->vk.flags &
              VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT)) {
            ok = false;
            break;
         }
         memcpy(&published, (const uint8_t *)set->mapped_ptr + e->offset_B,
                sizeof(published));
         if (e->clamp_0) {
            if ((int16_t)published < 0)
               continue;
            published &= HK_BORDER_IS_INT_BIT - 1;
         }
         const uint16_t native = hk_sampler_heap_index(
            set->layout->native_sampler_indices, published);
         if (native >= dev->samplers.table.alloc) {
            ok = false;
            break;
         }
         memcpy(smp[1 + i],
                (const uint8_t *)dev->samplers.table.map + native * 32u, 32);
      }
      if (!ok) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      const uint64_t tex_va =
         hk_pool_upload(cmd, tex, 32u * t->texture_count, 64);
      const uint64_t smp_va =
         hk_pool_upload(cmd, smp, 32u * (1u + t->sampler_count), 64);
      const uint64_t pointers[2] = {tex_va, smp_va};
      texture_pointers = tex_va && smp_va
                            ? hk_pool_upload(cmd, pointers, sizeof(pointers), 8)
                            : 0;
      if (!texture_pointers) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
   } else if (g18 && s->info.agx3_storage_image_valid) {
      /* One image, or several in one set: the table in the descriptor set
       * (the resolved PBE, or {texture, PBE} pairs) and the G18G
       * write-state record. */
      const unsigned table_set = s->info.agx3_storage_image.set;
      if (table_set >= HK_MAX_SETS) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      const uint64_t set_address = descriptors->root.sets[table_set];
      const uint32_t set_size = descriptors->set_sizes[table_set];
      /* Read-write: the {texture, PBE} pair as Gallium's two-entry table. */
      const uint32_t pbe_offset =
         s->info.agx3_storage_image.write_only
            ? s->info.agx3_storage_image.pbe_offset_B
            : s->info.agx3_storage_image.texture_offset_B;
      const uint32_t table_size =
         s->info.agx3_storage_image.table_size_B
            ? s->info.agx3_storage_image.table_size_B
            : (s->info.agx3_storage_image.write_only ? 32 : 64);
      if (!set_address || pbe_offset > set_size ||
          set_size - pbe_offset < table_size) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      const uint64_t write_state = hk_pool_upload(
         cmd, hk_g18g_write_state1, sizeof(hk_g18g_write_state1), 32);
      const uint64_t pointers[2] = {set_address + pbe_offset, write_state};
      texture_pointers =
         write_state ? hk_pool_upload(cmd, pointers, sizeof(pointers), 8) : 0;
      if (!texture_pointers) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
   }
   if (!root || (s->info.agx3_indexed_sampler_valid && !texture_pointers)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   struct agx3_ioto_uniform_range ranges[HK_G17P_COMPUTE_MAX_RANGES];
   size_t range_count;
   const bool ranges_ok = internal
      ? hk_g17p_internal_graphics_uniform_ranges(cmd, s, linked, root,
                                                 texture_pointers, ranges,
                                                 &range_count)
      : hk_g17p_compute_uniform_ranges(dev->dev.chip, cmd, s, root,
                                       texture_pointers, ranges, &range_count);
   if (!ranges_ok) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   /* Pool the Ioto program: the command buffer's USC pool sub-allocates
    * (bump pointer, no kernel ioctl) instead of creating a BO per dispatch.
    * The per-dispatch agx_bo_create is the dominant per-op cost (~5ms of
    * ~10ms per op in the 200-op chain); the pool makes it a pointer bump. */
   struct agx_ptr program_t = hk_pool_usc_alloc(
      cmd, AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE,
      AGX_IOTO_PROGRAM_ALIGNMENT);
   if (!program_t.cpu) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   const uint64_t shader_va = linked->b.bo->va->addr + linked->b.entry_offset;
   if (!agx_va_in_usc_window(&dev->dev, program_t.gpu) ||
       !agx_va_in_usc_window(&dev->dev, shader_va) ||
       linked->b.size > AGX_USC_WINDOW_SIZE - (shader_va - dev->dev.shader_base)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   const uint32_t threadgroup_bytes =
      s->b.info.local_size
         ? MAX2(256u, ALIGN_POT(s->b.info.local_size, 64u))
         : 0;
   struct agx3_g17p_graphics_program_desc desc = {
      .program_va = program_t.gpu, .shader_va = shader_va,
      .uniform_ranges = ranges, .uniform_range_count = range_count,
      .shader_binary_size = linked->b.size,
      .scratch_size = g18 ? scratch_bytes : 0,
      .nr_gprs = linked->b.nr_gprs,
      .compiler_d_valid = s->b.info.agx3_staging_known,
      .compiler_d = (uint32_t)s->b.info.agx3_staging_halfs * 2,
      .threadgroup_memory_size_B = threadgroup_bytes,
   };
   struct agx3_g17p_graphics_program program;
   /* The launch record is G17G's (agx3_g17g_native_compute_program_bind
    * binds it as G17G); name that chip for the shared native bind. */
   struct agx3_g17p_native_compute_program_desc launch_desc = {
      .chip = AGX_CHIP_G17G,
      .program_va = program_t.gpu, .shader_va = shader_va,
   };
   struct agx3_compute_program launch;
   bool generated;
   if (g18) {
      hk_m5_compute_program_desc_set_device(dev, &desc);
      generated = agx3_g18g_compute_program_generate(&desc, &program);
   } else {
      generated = hk_m5_compute_program_generate(dev, &desc, &program);
   }
   if (!generated ||
       !agx3_cdm_direct_template_sources_compute(s->b.info.uses_resource_info_query,
                                                 false, false, &launch_desc.launch_sources) ||
       !agx3_g17g_native_compute_program_bind(&launch_desc, &launch)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (g18) {
      struct util_dynarray code;
      util_dynarray_init(&code, NULL);
      memcpy(util_dynarray_grow_bytes(&code, 1, program.length),
             program.bytes, program.length);
      const struct agx3_transcode_ioto ioto = {
         .api_program = true,
         .shader_gprs = linked->b.nr_gprs,
      };
      const char *why = NULL;
      unsigned at = 0;
      const bool ok = agx3_transcode_agx3d_ioto(&code, 0, &ioto, &why, &at) &&
                      code.size <= sizeof(program.bytes);
      if (ok) {
         memcpy(program.bytes, code.data, code.size);
         program.length = code.size;
      }
      util_dynarray_fini(&code);
      if (!ok) {
         mesa_loge("G18G compute Ioto not re-encoded: %s at +0x%x",
                   why ? why : "too long", at);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
   }
   memcpy(program_t.cpu, program.bytes, program.length);
   hk_cmd_buffer_retain_bo(cmd, linked->b.bo);
   if (s->b.info.rodata.size_16)
      hk_cmd_buffer_retain_bo(cmd, s->bo);

   if (g18_indirect) {
      if (s->b.info.local_size)
         launch.launch_template.word0 |= UINT32_C(1) << 23;
      const uint32_t main_scratch =
         MAX2(s->b.info.scratch_size, linked->b.scratch_size);
      const uint32_t spectpr_scratch =
         MAX2(s->b.info.agx3_spectpr_scratch_size,
              linked->b.agx3_spectpr_scratch_size);
      const struct drm_asahi_g18_compute_resources target_res = {
         .version = 2,
         .flags = 1u,
         .registers_16 = linked->b.nr_gprs,
         .scratch_bytes = main_scratch,
         .spectpr_scratch_bytes = spectpr_scratch,
         .staging_16 = s->b.info.agx3_staging_halfs,
         .local_bytes = s->b.info.local_size,
         .imageblock_stride = s->b.info.imageblock_stride,
         .local_size = {local_size.x, local_size.y, local_size.z},
      };
      struct hk_cs **target =
         cs == cmd->current_cs.pre_gfx    ? &cmd->current_cs.pre_gfx
         : cs == cmd->current_cs.post_gfx ? &cmd->current_cs.post_gfx
                                          : &cmd->current_cs.cs;
      hk_g18_emit_indirect_launch(cmd, target, &launch, grid, local_size,
                                  &target_res);
      return;
   }

   /* (rel28b: the native-path homogeneity seal is REVERTED — batching mixed
 * descriptors is where the 3.4x came from; only the precomp helper is
 * sealed away now, in hk_dispatch_precomp.) */
   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   struct agx3_compute_dispatch_sink sink = {
      .data = (uint8_t *)cs->current,
      .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
   };
   const struct agx3_compute_direct_grid direct = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {grid.count[0], grid.count[1], grid.count[2]},
      .local_size = {local_size.x, local_size.y, local_size.z},
      .dimensions = grid.count[2] != 1 || local_size.z != 1 ? 3
                  : grid.count[1] != 1 || local_size.y != 1 ? 2 : 1,
   };
   /* flags=5 (strong flush) between dependent launches: the weak 0x60000160
    * tail barrier does not give write visibility on G18G (hk_cdm_cache_flush
    * notes), and flag-free chained launches hung the CDM (rel24 run 1).
    * The flags=5 chaining is grounded on hardware (rel24b+), unlike the
    * standalone flush word in hk_cdm_cache_flush, whose only grounded G18G
    * encoding is the flag-free tail. */
   if (g18) {
      if (direct.size[1] < 2 && direct.size[2] < 2)
         launch.launch_template.word0 &= UINT32_C(0xfff1ffff);
      if (s->b.info.local_size)
         launch.launch_template.word0 |= UINT32_C(1) << 23;
   }
   if (!agx3_g17g_native_compute_dispatch_append(&launch, &direct, 5, &sink)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (g18) {
      /* Dependent dispatches share one stream (the strong barrier between
       * launches orders them) and hk_EndCommandBuffer appends the final
       * terminate: N dispatches become 1 submit + 1 fence. The stream is
       * capped at HK_G18_BATCH_LAUNCHES launches (see its definition); at
       * the cap the stream terminates and seals, and the next dispatch
       * rotates to a fresh stream (dispatch_grid's g18_dispatch_sealed
       * check). */
      const uint32_t batch_limit = HK_G18_BATCH_LAUNCHES;
      /* rel31 (w7C's plan rule + w7P's validator semantics): the kernel
       * sizes the stream's compute plan (pool/SMA/counters) from these
       * resource words alone, and last-wins under-sizes a stream whose
       * big launches precede a small last launch (job 131: 1024-thread
       * launches + the small helper last = small plan = stall). Every
       * scalar field takes the running max; local dims come from the launch
       * with the LARGEST x*y*z product (a per-axis max can exceed the
       * validator's 1024-thread bound, w7P check c). */
      const uint32_t prod =
         local_size.x * local_size.y * local_size.z;
      const uint32_t prev_prod = cs->g18_compute_resources.local_size[0] *
                                 cs->g18_compute_resources.local_size[1] *
                                 cs->g18_compute_resources.local_size[2];
      const uint32_t regs = MAX2(cs->g18_compute_resources.registers_16,
                                 linked->b.nr_gprs);
      const uint32_t staging =
         MAX2(cs->g18_compute_resources.staging_16,
              s->b.info.agx3_staging_halfs);
      const uint32_t stride =
         MAX2(cs->g18_compute_resources.imageblock_stride,
              s->b.info.imageblock_stride);
      const uint32_t local_bytes =
         MAX2(cs->g18_compute_resources.local_bytes, s->b.info.local_size);
      const uint32_t main_scratch =
         MAX2(cs->g18_compute_resources.scratch_bytes,
              MAX2(s->b.info.scratch_size, linked->b.scratch_size));
      const uint32_t spectpr_scratch =
         MAX2(cs->g18_compute_resources.spectpr_scratch_bytes,
              MAX2(s->b.info.agx3_spectpr_scratch_size,
                   linked->b.agx3_spectpr_scratch_size));
      if (prod >= prev_prod || cs->stats.cmds == 0) {
         cs->g18_compute_resources = (struct drm_asahi_g18_compute_resources){
            .version = 2,
            .flags = 1u,
            .registers_16 = regs,
            .scratch_bytes = main_scratch,
            .spectpr_scratch_bytes = spectpr_scratch,
            .staging_16 = staging,
            .local_bytes = local_bytes,
            .imageblock_stride = stride,
            .local_size = {local_size.x, local_size.y, local_size.z},
         };
      } else {
         cs->g18_compute_resources.version = 2;
         cs->g18_compute_resources.registers_16 = regs;
         cs->g18_compute_resources.scratch_bytes = main_scratch;
         cs->g18_compute_resources.spectpr_scratch_bytes = spectpr_scratch;
         cs->g18_compute_resources.staging_16 = staging;
         cs->g18_compute_resources.local_bytes = local_bytes;
         cs->g18_compute_resources.imageblock_stride = stride;
      }
      /* Launch-count cap only (the rel30 code cap is retired: the gate
       * showed generation streams never approach the code boundary). */
      if (cs->stats.cmds + 1 >= batch_limit) {
         uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
         if (!agx3_cdm_stream_terminate_pack(dev->dev.chip, terminate) ||
             sink.capacity - sink.length < sizeof(terminate)) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         memcpy(sink.data + sink.length, terminate, sizeof(terminate));
         sink.length += sizeof(terminate);
         cs->g18_dispatch_sealed = true;
      }
   }
   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   cs->stats.cmds++;
}

static bool
hk_compute_shader_overlap_safe(const struct hk_shader *s,
                               const struct hk_descriptor_state *desc)
{
   const struct hk_linked_shader *linked = s ? s->only_linked : NULL;
   return s && linked && s->info.stage == MESA_SHADER_COMPUTE &&
          !s->b.info.uses_ray_traversal &&
          !s->b.info.uses_global_atomics &&
          !desc->root.cs.trace_rays_addr &&
          MAX3(s->b.info.scratch_size, linked->b.scratch_size,
               linked->b.agx3_spectpr_scratch_size) <=
             HK_M5_OVERLAP_MAX_SCRATCH_B;
}

static void
hk_record_compute_pair(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       const struct hk_api_shader *api,
                       const struct hk_shader *selected,
                       enum hk_compute_pair_alt_kind alt_kind,
                       const uint64_t pack_loop_params[HK_PACK_LOOP_PARAM_COUNT],
                       const uint32_t orig_groups[3], struct agx_grid exec_grid)
{
   const struct hk_compute_pair_info *info = api ? api->compute_pair : NULL;
   if (!info || cs->compute_pair || cs->stats.cmds != 1 || cs->stream_linked ||
       vk_command_buffer_has_error(&cmd->vk) || cmd->in_meta ||
       exec_grid.mode != AGX_CDM_MODE_DIRECT ||
       cmd->state.compute_pair_simultaneous ||
       cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
      return;

   const struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (!hk_nir_dispatch_zero_base(desc->root.cs.base_group))
      return;

   const bool is_producer = info->role == HK_COMPUTE_PAIR_DEPTHWISE ||
                            info->role == HK_COMPUTE_PAIR_GEMM;
   if (is_producer && (!api->compute_pair_producer || !desc->uploaded_root.gpu ||
                       selected->data_size != 0))
      return;

   const struct hk_nir_clamp_buffer *bindings[4];
   unsigned count;
   bool has_bias_binding = true;
   if (info->role == HK_COMPUTE_PAIR_DEPTHWISE) {
      count = 4;
      for (unsigned i = 0; i < 3; ++i)
         bindings[i] = &info->depthwise.reads[i];
      bindings[3] = &info->depthwise.output;
      has_bias_binding = info->depthwise.has_bias_buffer;
   } else if (info->role == HK_COMPUTE_PAIR_GEMM) {
      count = 4;
      for (unsigned i = 0; i < 3; ++i)
         bindings[i] = &info->gemm.reads[i];
      bindings[3] = &info->gemm.output;
      has_bias_binding = info->gemm.has_bias_buffer;
   } else {
      count = 2;
      bindings[0] = &info->clamp.input;
      bindings[1] = &info->clamp.output;
   }

   struct hk_compute_pair_dispatch record = {.info = *info};
   for (unsigned i = 0; i < count; ++i) {
      if (is_producer && i == 2 && !has_bias_binding) {
         record.bindings[i] = (struct hk_storage_backing){0};
         continue;
      }
      const unsigned set = bindings[i]->set;
      if (set >= HK_MAX_SETS ||
          ((desc->buffer_active | desc->push_active) & BITFIELD_BIT(set)))
         return;
      if (!hk_descriptor_set_storage_backing(desc->sets[set], bindings[i]->binding,
                                             &record.bindings[i])) {
         if (info->role == HK_COMPUTE_PAIR_GEMM && i == 2) {
            record.bindings[i] = (struct hk_storage_backing){0};
            continue;
         }
         return;
      }
   }

   memcpy(record.groups, orig_groups, sizeof(record.groups));
   memcpy(record.exec_threads, exec_grid.count, sizeof(record.exec_threads));
   memcpy(record.exec_local, selected->b.info.workgroup_size,
          sizeof(record.exec_local));
   record.root_va = desc->uploaded_root.gpu;
   memcpy(record.root_sets, desc->root.sets, sizeof(record.root_sets));
   memcpy(record.push, desc->root.push, sizeof(record.push));
   record.cs_group_count_addr = desc->root.cs.group_count_addr;
   memcpy(record.cs_base_group, desc->root.cs.base_group,
          sizeof(record.cs_base_group));
   record.printf_buffer_address = desc->root.printf_buffer_address;
   record.alt_kind = alt_kind;
   record.overlap_safe = hk_compute_shader_overlap_safe(selected, desc);
   if (pack_loop_params)
      memcpy(record.pack_loop_params, pack_loop_params,
             sizeof(record.pack_loop_params));

   cs->compute_pair = malloc(sizeof(record));
   if (!cs->compute_pair)
      return;
   if (is_producer)
      record.producer = hk_compute_pair_producer_ref(api->compute_pair_producer);
   *cs->compute_pair = record;
}

static bool
hk_fused_compute_uniform_ranges(
   enum agx_chip chip, const struct hk_shader *s,
   const struct hk_compute_pair_dispatch *p,
   struct agx3_ioto_uniform_range ranges[HK_G17P_COMPUTE_MAX_RANGES],
   size_t *range_count)
{
   if (!s || !p || !p->root_va || !ranges || !range_count || s->data_size != 0 ||
       s->b.info.uses_ray_traversal || s->b.info.uses_global_atomics ||
       s->info.stage != MESA_SHADER_COMPUTE ||
       s->info.agx3_txf_texture_valid || s->info.agx3_storage_image_valid ||
       s->info.agx3_sampled_storage_valid || s->info.agx3_indexed_storage_valid ||
       s->info.agx3_indexed_sampler_valid)
      return false;

   const unsigned dma_count = s->b.info.agx3_uniform_dma_count;
   const struct agx_rodata *rodata = &s->b.info.rodata;
   if (!s->info.agx3_uses_root && s->info.set_count == 0 &&
       rodata->size_16 == 0 && dma_count == 0) {
      *range_count = 0;
      return true;
   }

   const unsigned root_uniform = s->info.agx3_root_uniform;
   if (!hk_agx3_g17p_root_range(root_uniform, s->info.set_count, p->root_va,
                                hk_shader_root_size(s), &ranges[0]))
      return false;

   size_t count = 1;
   if (rodata->size_16 != 0) {
      if (!s->bo || !s->bo->va ||
          !hk_agx3_g17p_rodata_range(rodata->base_uniform, rodata->size_16,
                                     hk_shader_code_va(s), rodata->offset,
                                     s->b.info.binary_size, &ranges[count]))
         return false;
      count++;
   }

   for (unsigned i = 0; i < dma_count; ++i) {
      const struct agx3_uniform_dma_span *span = &s->b.info.agx3_uniform_dma[i];
      if (span->source == AGX3_UNIFORM_DMA_BUFFER)
         return false;
      const unsigned pointer = span->pointer;
      const int set = pointer == root_uniform ? -1
                      : pointer >= root_uniform + 4 &&
                            (pointer - root_uniform - 4) % 4 == 0 &&
                            (pointer - root_uniform - 4) / 4 < HK_MAX_SETS
                         ? (int)(pointer - root_uniform - 4) / 4
                         : -2;
      if (count >= HK_G17P_COMPUTE_MAX_RANGES || set == -2 ||
          (span->dest & 1) || (span->halfs & 1) || (span->offset_B & 3))
         return false;
      const uint64_t base = set < 0 ? p->root_va : p->root_sets[set];
      const uint64_t address = base ? base + span->offset_B : AGX_ZERO_PAGE_ADDRESS;
      if (address & 3)
         return false;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = span->dest,
         .size_halfs = span->halfs,
         .address = address,
      };
   }

   *range_count = count;
   return true;
}

static struct agx_workgroup
hk_fused_launch_local(const struct hk_shader *s)
{
   if (s->b.info.launch_size[0])
      return agx_workgroup(s->b.info.launch_size[0], s->b.info.launch_size[1],
                           s->b.info.launch_size[2]);
   return agx_workgroup(s->b.info.workgroup_size[0], s->b.info.workgroup_size[1],
                        s->b.info.workgroup_size[2]);
}

static bool
hk_fused_compute_shader_eligible_ranges(
   const struct hk_device *dev, struct hk_shader *s,
   const struct hk_compute_pair_dispatch *p,
   struct agx3_ioto_uniform_range ranges[HK_G17P_COMPUTE_MAX_RANGES],
   size_t *range_count)
{
   if (!dev || !s || !p || !agx_chip_native_or_m5(dev->dev.chip))
      return false;
   struct hk_linked_shader *linked = s->only_linked;
   const bool is_m5 = agx_chip_g17_m5(dev->dev.chip);
   const unsigned max_spill_B =
      is_m5 ? (1u << 16) : AGX3_G17P_MAX_COMPUTE_SPILL_B;
   if (!hk_g17p_compute_shader_supported(dev->dev.chip, s, linked, max_spill_B))
      return false;
   if (memcmp(s->b.info.workgroup_size, p->exec_local, sizeof(p->exec_local)))
      return false;
   if (!p->exec_threads[0] || !p->exec_threads[1] || !p->exec_threads[2])
      return false;

   const struct agx_workgroup local = hk_fused_launch_local(s);
   if (!local.x || !local.y || !local.z ||
       (uint64_t)local.x * local.y * local.z >
          dev->vk.physical->properties.maxComputeWorkGroupInvocations ||
       (p->exec_threads[0] % local.x) || (p->exec_threads[1] % local.y) ||
       (p->exec_threads[2] % local.z))
      return false;

   if (s->b.info.has_preamble && (!s->bo || !s->bo->va))
      return false;

   if (is_m5) {
      if ((s->b.info.scratch_size || linked->b.scratch_size) &&
          !(dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_RESOURCES))
         return false;
      if (s->b.info.uses_matrix16 &&
          !(dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_MATRIX))
         return false;
   }

   return hk_fused_compute_uniform_ranges(dev->dev.chip, s, p, ranges,
                                          range_count);
}

bool
hk_fused_compute_shader_eligible(const struct hk_device *dev,
                                 struct hk_shader *s,
                                 const struct hk_compute_pair_dispatch *p)
{
   struct agx3_ioto_uniform_range ranges[HK_G17P_COMPUTE_MAX_RANGES];
   size_t range_count = 0;
   return hk_fused_compute_shader_eligible_ranges(dev, s, p, ranges,
                                                  &range_count);
}

size_t
hk_fused_compute_ioto_bound(const struct hk_device *dev, struct hk_shader *s,
                            const struct hk_compute_pair_dispatch *p)
{
   const size_t fallback = ALIGN_POT(
      MAX2(HK_AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE,
           HK_AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE),
      64u);
   if (!dev || !s || !p || !agx_chip_native_or_m5(dev->dev.chip))
      return fallback;

   struct agx3_ioto_uniform_range ranges[HK_G17P_COMPUTE_MAX_RANGES];
   size_t range_count = 0;
   if (!hk_fused_compute_uniform_ranges(dev->dev.chip, s, p, ranges,
                                        &range_count))
      return fallback;

   size_t body_bound = 0, chunk_count = 0;
   if (range_count &&
       !agx3_ioto_uniform_program_body_bound(dev->dev.chip, ranges, range_count,
                                             &body_bound, &chunk_count))
      return fallback;

   /* Include state tail, profile suffix, external constant-program branch,
    * and potential independent-base relocation overhead. */
   const size_t fixed_overhead = agx_chip_g17_m5(dev->dev.chip) ? 512u : 256u;
   const size_t est = ALIGN_POT(body_bound + fixed_overhead, 64u);
   return MIN2(est, fallback);
}

bool
hk_emit_fused_compute_dispatch(struct hk_device *dev, struct hk_cs *cs,
                               struct hk_shader *s,
                               const struct hk_compute_pair_dispatch *p,
                               uint8_t *ioto_map, uint64_t ioto_va,
                               size_t ioto_cap, size_t *ioto_used_out)
{
   struct agx3_ioto_uniform_range ranges[HK_G17P_COMPUTE_MAX_RANGES];
   size_t range_count = 0;
   if (!cs || !ioto_map || !ioto_va || !ioto_used_out ||
       !hk_cs_space_fits((uint8_t *)cs->end - (uint8_t *)cs->current,
                         AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE,
                         AGX_CDM_STREAM_LINK_LENGTH) ||
       !hk_fused_compute_shader_eligible_ranges(dev, s, p, ranges,
                                                &range_count))
      return false;

   struct hk_linked_shader *linked = s->only_linked;
   const uint64_t shader_va = linked->b.bo->va->addr + linked->b.entry_offset;
   if (!agx_va_in_usc_window(&dev->dev, ioto_va) ||
       !agx_va_in_usc_window(&dev->dev, shader_va))
      return false;

   const struct agx_workgroup local = hk_fused_launch_local(s);
   const struct agx3_compute_direct_grid direct = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {p->exec_threads[0], p->exec_threads[1], p->exec_threads[2]},
      .local_size = {local.x, local.y, local.z},
      .dimensions = p->exec_threads[2] != 1 || local.z != 1 ? 3
                  : p->exec_threads[1] != 1 || local.y != 1 ? 2 : 1,
   };

   struct agx3_compute_dispatch_sink sink = {
      .data = (uint8_t *)cs->current,
      .capacity = (uint8_t *)cs->end - (uint8_t *)cs->current,
   };
   size_t ioto_used = 0;

   if (agx_chip_g17_m5(dev->dev.chip)) {
      struct agx3_g17p_graphics_program_desc desc = {
         .program_va = ioto_va,
         .shader_va = shader_va,
         .uniform_ranges = ranges,
         .uniform_range_count = range_count,
         .shader_binary_size = linked->b.size,
         .nr_gprs = MAX2(linked->b.nr_gprs, s->b.info.nr_preamble_gprs),
         .scratch_size = MAX3(s->b.info.scratch_size, linked->b.scratch_size,
                              linked->b.agx3_spectpr_scratch_size),
         .compiler_d_valid = s->b.info.agx3_staging_known,
         .compiler_d = MAX2((uint32_t)s->b.info.agx3_staging_halfs * 2,
                            (uint32_t)s->b.info.agx3_preamble_staging_halfs * 2),
         .threadgroup_memory_size_B = s->b.info.local_size
            ? MAX2(256u, ALIGN_POT(s->b.info.local_size, 64u)) : 0,
      };
      if (s->b.info.has_preamble) {
         if (!s->bo || !s->bo->va ||
             !hk_agx3_g17p_constant_program_from_binary(
                s->code_ptr, s->code_size, s->b.info.preamble_offset,
                s->b.info.preamble_size, hk_shader_code_va(s),
                s->b.info.binary_size, &desc.constant_program))
            return false;
         desc.constant_program.force_external = true;
         desc.constant_program.absolute_branch = true;
         desc.constant_program.request_prelude =
            s->b.info.agx3_preamble_uniform_inputs;
         if (s->b.info.agx3_preamble_tps_inputs)
            desc.cprog_inputs = &s->b.info.agx3_preamble_inputs;
      }

      struct agx3_g17p_native_compute_program_desc launch_desc = {
         .program_va = ioto_va,
         .shader_va = shader_va,
      };
      if (!agx3_cdm_direct_template_sources_compute(
             s->b.info.uses_resource_info_query, false, false,
             &launch_desc.launch_sources))
         return false;
      if (s->b.info.has_preamble) {
         const unsigned cprog_frame =
            ALIGN_POT(MAX2((unsigned)s->b.info.agx3_preamble_inputs.storage_bytes,
                           s->b.info.preamble_scratch_size), 16);
         launch_desc.launch_sources.ti_spill_bytes =
            MAX2(launch_desc.launch_sources.ti_spill_bytes, cprog_frame);
      }

      const struct hk_m5_compute_loader *ml = hk_m5_loader_get(dev, linked, &desc);
      if (ml) {
         if (ml->length > ioto_cap)
            return false;
         struct agx3_compute_program launch;
         if (!hk_m5_compute_launch_bind(dev, &launch_desc, &launch) ||
             !agx3_m5_native_compute_dispatch_append(&launch, &direct, 0, &sink))
            return false;
         memcpy(ioto_map, ml->bytes, ml->length);
         hk_g17p_patch_plan_apply(&ml->plan, ranges, ioto_map);
         ioto_used = ml->length;
      } else {
         struct agx3_g17p_graphics_program program;
         if (!hk_m5_compute_program_generate(dev, &desc, &program) ||
             program.length > ioto_cap)
            return false;
         struct agx3_compute_program launch;
         if (!hk_m5_compute_launch_bind(dev, &launch_desc, &launch) ||
             !agx3_m5_native_compute_dispatch_append(&launch, &direct, 0, &sink))
            return false;
         memcpy(ioto_map, program.bytes, program.length);
         ioto_used = program.length;
      }
      cs->m5_matrix_ops |= s->b.info.uses_matrix16;
      hk_m5_compute_resources(cs, desc.scratch_size, 0,
                              desc.threadgroup_memory_size_B,
                              (uint32_t)local.x * local.y * local.z);
   } else {
      struct agx3_g17p_compute_program_desc desc = {0};
      if (!agx3_cdm_direct_template_sources_compute(
             s->b.info.uses_resource_info_query, false, false,
             &desc.launch_sources))
         return false;
      if (s->b.info.has_preamble) {
         if (!s->bo || !s->bo->va ||
             !hk_agx3_g17p_constant_program_from_binary(
                s->code_ptr, s->code_size, s->b.info.preamble_offset,
                s->b.info.preamble_size, hk_shader_code_va(s),
                s->b.info.binary_size, &desc.constant_program))
            return false;
         desc.constant_program.force_external = true;
         desc.constant_program.absolute_branch = true;
         desc.constant_program.request_prelude =
            s->b.info.agx3_preamble_uniform_inputs;
         if (s->b.info.agx3_preamble_tps_inputs)
            desc.cprog_inputs = &s->b.info.agx3_preamble_inputs;
         const unsigned cprog_frame =
            ALIGN_POT(MAX2((unsigned)s->b.info.agx3_preamble_inputs.storage_bytes,
                           s->b.info.preamble_scratch_size), 16);
         desc.launch_sources.ti_spill_bytes =
            MAX2(desc.launch_sources.ti_spill_bytes, cprog_frame);
         desc.launch_sources.ti_largest_ipr_bytes =
            MAX2(desc.launch_sources.ti_largest_ipr_bytes,
                 (uint32_t)s->b.info.agx3_preamble_staging_halfs * 2u);
      }
      desc.shader_va = shader_va;
      desc.uniform_ranges = ranges;
      desc.uniform_range_count = range_count;
      desc.shader_binary_size = linked->b.size;
      desc.scratch_size =
         MAX2(linked->b.scratch_size, linked->b.agx3_spectpr_scratch_size);
      desc.nr_gprs = MAX2(linked->b.nr_gprs, s->b.info.nr_preamble_gprs);
      desc.compiler_d_valid = s->b.info.agx3_staging_known;
      desc.compiler_d = MAX2((uint32_t)s->b.info.agx3_staging_halfs * 2,
                             (uint32_t)s->b.info.agx3_preamble_staging_halfs * 2);
      desc.threadgroup_memory_size_B = s->b.info.local_size
         ? MAX2(256u, ALIGN_POT(s->b.info.local_size, 64u)) : 0;
      desc.ldshdr_f1 = AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3;

      const struct hk_g17p_loader *l =
         hk_g17p_loader_get(dev, &linked->g17p_compute_templates, &desc);
      if (!l || l->length > ioto_cap)
         return false;

      struct agx3_compute_program launch = l->launch;
      launch.program_va = ioto_va;
      if (!hk_agx3_g17p_native_compute_dispatch_append_with_barrier(
             &launch, &direct, 0, &sink))
         return false;

      memcpy(ioto_map, l->bytes, l->length);
      hk_g17p_patch_plan_apply(&l->plan, ranges, ioto_map);
      ioto_used = l->length;
   }

   cs->current = (uint32_t *)((uint8_t *)cs->current + sink.length);
   cs->weak_barrier_end = cs->current;
   cs->strong_flush_end = NULL;
   cs->memory_flush_end = NULL;
   cs->app_dispatch_end = NULL;
   cs->stats.cmds++;
   cs->stats.calls++;
   cs->pack_dispatches++;
   if (unlikely(dev->frame_anatomy)) {
      hk_frame_anatomy_note_fs(cs, s);
      const uint64_t wg = agx_workgroup_threads(local);
      const uint64_t threads =
         (uint64_t)p->exec_threads[0] * p->exec_threads[1] * p->exec_threads[2];
      hk_frame_anatomy_note_grid(cs, s, threads, wg ? threads / wg : 0, false);
   }
   *ioto_used_out = ioto_used;
   return true;
}

static bool
hk_read_push_i32(const struct hk_descriptor_state *desc, uint32_t off, int32_t *out)
{
   if (off + 4 > HK_COMPUTE_OPT_PUSH_BYTES)
      return false;
   const uint64_t mask = BITFIELD64_MASK(4) << off;
   if ((desc->identity_copy4_push_mask & mask) != mask)
      return false;
   memcpy(out, desc->root.push + off, sizeof(*out));
   return true;
}

static bool
hk_identity_copy4_required_bytes(const struct hk_descriptor_state *desc,
                                 const struct hk_identity_copy4 *info,
                                 struct agx_grid grid, const uint16_t local[3],
                                 uint64_t *bytes, uint32_t *k_out)
{
   const bool has_border = info->border_width_offset != UINT16_MAX;
   int32_t bw = 0, bh = 0, iw = 0, ih = 0, kw = 0, k = 0, m = 0, sw = 0, sh = 0, channels = 0;
   if ((has_border &&
        (!hk_read_push_i32(desc, info->border_width_offset, &bw) ||
         !hk_read_push_i32(desc, info->border_height_offset, &bh) ||
         !hk_read_push_i32(desc, info->input_height_offset, &ih))) ||
       !hk_read_push_i32(desc, info->input_width_offset, &iw) ||
       !hk_read_push_i32(desc, info->kernel_width_offset, &kw) ||
       !hk_read_push_i32(desc, info->patches_width_offset, &k) ||
       !hk_read_push_i32(desc, info->patches_height_offset, &m) ||
       !hk_read_push_i32(desc, info->stride_width_offset, &sw) ||
       !hk_read_push_i32(desc, info->stride_height_offset, &sh) ||
       !hk_read_push_i32(desc, info->channels_offset, &channels))
      return false;

   if (bw || bh || sw != 1 || sh != 1 ||
       iw <= 0 || channels <= 0 || kw != channels || k != channels ||
       k % 4 || iw % channels || m <= 0 ||
       (has_border && (ih <= 0 || (uint64_t)(iw / channels) * ih != (uint64_t)m)))
      return false;

   if (grid.count[0] != DIV_ROUND_UP((uint32_t)k, local[0]) ||
       grid.count[1] != DIV_ROUND_UP((uint32_t)m, local[1]) || grid.count[2] != 1)
      return false;

   const uint64_t extent = (uint64_t)m * k * sizeof(uint32_t);
   if (extent > (uint64_t)INT32_MAX + 1)
      return false;
   *bytes = extent;
   *k_out = (uint32_t)k;
   return true;
}

static bool
hk_intervals_disjoint(const struct hk_storage_backing *a, uint64_t a_start, uint64_t a_bytes,
                      const struct hk_storage_backing *b, uint64_t b_start, uint64_t b_bytes)
{
   if (!a_bytes || !b_bytes)
      return true;
   const uint64_t a_bo0 = a->offset + a_start, a_bo1 = a_bo0 + a_bytes;
   const uint64_t b_bo0 = b->offset + b_start, b_bo1 = b_bo0 + b_bytes;
   const uint64_t a_va0 = a->address + a_start, a_va1 = a_va0 + a_bytes;
   const uint64_t b_va0 = b->address + b_start, b_va1 = b_va0 + b_bytes;
   const bool bo_overlap =
      a->handle == b->handle && a_bo0 < b_bo1 && b_bo0 < a_bo1;
   const bool va_overlap = a_va0 < b_va1 && b_va0 < a_va1;
   return !bo_overlap && !va_overlap;
}

static bool
hk_check_storage_backing(const struct hk_descriptor_state *desc,
                         unsigned set, unsigned binding,
                         uint64_t start, uint64_t bytes, unsigned align,
                         struct hk_storage_backing *backing)
{
   if (set >= HK_MAX_SETS || ((desc->buffer_active | desc->push_active) & BITFIELD_BIT(set)))
      return false;
   if (!hk_descriptor_set_storage_backing(desc->sets[set], binding, backing))
      return false;
   if (desc->root.sets[set] != hk_descriptor_set_addr(desc->sets[set]))
      return false;
   if (!backing->address || (backing->address | backing->offset) % align)
      return false;
   const uint64_t end = start + bytes;
   if (end < bytes || backing->size < end ||
       backing->address > UINT64_MAX - end ||
       backing->offset > UINT64_MAX - end)
      return false;
   return true;
}

struct hk_copy4_dispatch_cert {
   enum hk_compute_pair_role role;
   struct hk_storage_backing bindings[3];
   union {
      struct hk_compute_pair_copy4 copy4;
      struct hk_compute_pair_pad4 pad4;
      struct hk_compute_pair_im2col im2col;
   } extra;
};

static struct hk_shader *
hk_select_im2col_vec4(struct hk_cmd_buffer *cmd, struct hk_api_shader *api,
                      struct hk_shader *normal, struct agx_grid *grid,
                      struct hk_copy4_dispatch_cert *cert_out,
                      uint64_t *loop_bound_out,
                      bool *im2col_push_valid_out,
                      uint32_t *im2col_push_off_out,
                      struct hk_im2col_push *im2col_push_out)
{
   struct hk_shader *alt = api->im2col_vec4;
   const struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_identity_copy4 *info = &api->identity_copy4_info;
   if (!alt || !info->im2col_vec4_ok ||
       info->push_bytes > HK_COMPUTE_OPT_PUSH_BYTES ||
       hk_im2col_push_offset(info->push_bytes) + sizeof(struct hk_im2col_push) > HK_MAX_PUSH_SIZE ||
       !hk_g17p_compute_shader_supported(dev->dev.chip, alt, alt->only_linked, 1u << 16) ||
       alt->b.info.workgroup_size[0] == 0 || alt->b.info.workgroup_size[1] == 0 ||
       alt->b.info.workgroup_size[2] == 0)
      return normal;

   const bool has_border = info->border_width_offset != UINT16_MAX;
   int32_t bw = 0, bh = 0, iw = 0, ih = 0, kw = 0, pw = 0, ph = 0, sw = 0, sh = 0, ch = 0;
   if ((has_border &&
        (!hk_read_push_i32(desc, info->border_width_offset, &bw) ||
         !hk_read_push_i32(desc, info->border_height_offset, &bh) ||
         !hk_read_push_i32(desc, info->input_height_offset, &ih))) ||
       !hk_read_push_i32(desc, info->input_width_offset, &iw) ||
       !hk_read_push_i32(desc, info->kernel_width_offset, &kw) ||
       !hk_read_push_i32(desc, info->patches_width_offset, &pw) ||
       !hk_read_push_i32(desc, info->patches_height_offset, &ph) ||
       !hk_read_push_i32(desc, info->stride_width_offset, &sw) ||
       !hk_read_push_i32(desc, info->stride_height_offset, &sh) ||
       !hk_read_push_i32(desc, info->channels_offset, &ch))
      return normal;
   if ((has_border && (bw < 0 || bh < 0 || ih <= 0)) ||
       iw <= 0 || kw <= 0 || pw <= 0 || ph <= 0 || sw <= 0 || sh <= 0 || ch <= 0)
      return normal;
   if ((iw % ch) != 0 || (kw % ch) != 0 || (pw % kw) != 0)
      return normal;
   const bool use_vec4 =
      ((ch % 4) == 0 && (iw % 4) == 0 && (kw % 4) == 0 && (pw % 4) == 0);
   uint64_t bordered_iw = (uint32_t)iw;
   if (has_border) {
      const bool unit = (sw == 1 && sh == 1);
      const uint64_t pad_x = (uint64_t)(uint32_t)bw * (uint32_t)ch;
      const uint64_t pad_x_tot = unit ? pad_x * 2 : pad_x;
      if (pad_x_tot > (uint64_t)INT32_MAX - (uint32_t)iw ||
          (uint32_t)bh > 0x3fffffffu || pad_x > 0x3fffffffu)
         return normal;
      bordered_iw = (uint32_t)iw + pad_x_tot;
   }
   if (bordered_iw < (uint32_t)kw)
      return normal;
   const uint64_t diff =
      bordered_iw / (uint32_t)ch - (uint32_t)kw / (uint32_t)ch;
   const uint64_t cols = diff / (uint32_t)sw + 1;
   if (cols == 0 || ((uint32_t)ph % cols) != 0)
      return normal;
   const uint64_t rows_y = (uint32_t)ph / cols;
   const uint64_t kh = (uint32_t)pw / (uint32_t)kw;
   const uint64_t step_x = (uint64_t)(uint32_t)sw * (uint32_t)ch;
   if (rows_y == 0 || kh == 0 || step_x == 0 || step_x > (uint64_t)INT32_MAX ||
       (cols - 1) > ((uint64_t)INT32_MAX - (uint32_t)kw) / step_x ||
       (rows_y - 1) > ((uint64_t)INT32_MAX - kh) / (uint32_t)sh)
      return normal;
   const uint64_t max_x = (cols - 1) * step_x + (uint32_t)kw;
   const uint64_t req_ih = (rows_y - 1) * (uint32_t)sh + kh;

   if (grid->count[2] != 1 ||
       grid->count[0] < DIV_ROUND_UP((uint32_t)pw, info->local[0]) ||
       grid->count[1] < DIV_ROUND_UP((uint32_t)ph, info->local[1]) ||
       (uint64_t)grid->count[0] * info->local[0] > (uint64_t)INT32_MAX ||
       (uint64_t)grid->count[1] * info->local[1] > (uint64_t)INT32_MAX)
      return normal;

   uint64_t src_bytes = 0;
   if (has_border) {
      src_bytes = (uint64_t)(uint32_t)iw * (uint32_t)ih * sizeof(uint32_t);
   } else {
      if (max_x > (uint64_t)(uint32_t)iw)
         return normal;
      src_bytes = req_ih * (uint32_t)iw * sizeof(uint32_t);
   }
   const uint64_t dst_bytes = (uint64_t)(uint32_t)pw * (uint32_t)ph * sizeof(uint32_t);
   if (src_bytes > (uint64_t)INT32_MAX + 1 || dst_bytes > (uint64_t)INT32_MAX + 1)
      return normal;

   const uint32_t align_B = use_vec4 ? 16 : 4;
   struct hk_storage_backing src_b, dst_b;
   if (!hk_check_storage_backing(desc, info->src_set, info->src_binding, 0, src_bytes, align_B, &src_b) ||
       !hk_check_storage_backing(desc, info->dst_set, info->dst_binding, 0, dst_bytes, align_B, &dst_b))
      return normal;

   /* Reject overlap between source and destination buffers. */
   if (!hk_intervals_disjoint(&src_b, 0, src_bytes, &dst_b, 0, dst_bytes))
      return normal;

   const uint32_t x_units = DIV_ROUND_UP((uint32_t)kw, 4u);
   const uint64_t gx = DIV_ROUND_UP(x_units, alt->b.info.workgroup_size[0]);
   const uint64_t gy = DIV_ROUND_UP(cols, alt->b.info.workgroup_size[1]);
   const uint64_t gz = DIV_ROUND_UP(DIV_ROUND_UP(rows_y, 4u), alt->b.info.workgroup_size[2]);
   if (gx > 65535 || gy > 65535 || gz > 65535)
      return normal;

   grid->count[0] = (uint32_t)gx;
   grid->count[1] = (uint32_t)gy;
   grid->count[2] = (uint32_t)gz;
   if (im2col_push_valid_out && im2col_push_off_out && im2col_push_out) {
      *im2col_push_valid_out = true;
      *im2col_push_off_out = hk_im2col_push_offset(info->push_bytes);
      *im2col_push_out = (struct hk_im2col_push){
         .cols = (uint32_t)cols,
         .rows_y = (uint32_t)rows_y,
         .kh = (uint32_t)kh,
      };
   }
   if (loop_bound_out) {
      uint64_t params[HK_PACK_LOOP_PARAM_COUNT] = {0};
      params[HK_PACK_LOOP_PARAM_IM2COL_KH] = kh;
      hk_pack_eval_effective_instrs(&alt->pack_loop_bound,
                                    alt->b.info.stats.instrs, params,
                                    loop_bound_out);
   }
   if (cert_out) {
      const bool unit = (sw == 1 && sh == 1);
      cert_out->role = HK_COMPUTE_PAIR_IM2COL;
      cert_out->bindings[0] = src_b;
      cert_out->bindings[1] = dst_b;
      cert_out->extra.im2col = (struct hk_compute_pair_im2col){
         .src_bytes = src_bytes,
         .dst_bytes = dst_bytes,
         .iw = (uint32_t)iw,
         .ih = has_border ? (uint32_t)ih : (uint32_t)req_ih,
         .kw = (uint32_t)kw,
         .cols = (uint32_t)cols,
         .step_x = (uint32_t)step_x,
         .sh = (uint32_t)sh,
         .pad_x = (has_border && unit)
                     ? (uint32_t)((uint64_t)(uint32_t)bw * (uint32_t)ch)
                     : 0u,
         .pad_y = (has_border && unit) ? (uint32_t)bh : 0u,
         .pw = (uint32_t)pw,
         .ph = (uint32_t)ph,
      };
   }
   return alt;
}

static void
hk_record_copy4_pair(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                     const struct hk_api_shader *api,
                     const struct hk_shader *selected,
                     const struct hk_copy4_dispatch_cert *cert,
                     const uint32_t orig_groups[3], struct agx_grid exec_grid)
{
   if (!cert || cert->role == HK_COMPUTE_PAIR_NONE || cs->compute_pair ||
       cs->stats.cmds != 1 || cs->stream_linked ||
       vk_command_buffer_has_error(&cmd->vk) || cmd->in_meta ||
       exec_grid.mode != AGX_CDM_MODE_DIRECT ||
       cmd->state.compute_pair_simultaneous ||
       cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
      return;

   const struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (desc->root.cs.base_group[0] || desc->root.cs.base_group[1] ||
       desc->root.cs.base_group[2])
      return;

   if (cert->role == HK_COMPUTE_PAIR_FADD &&
       (!api || !api->compute_pair_producer ||
        api->compute_pair_producer->info.role != HK_COMPUTE_PAIR_FADD ||
        !desc->uploaded_root.gpu || !selected || selected->data_size != 0))
      return;

   struct hk_compute_pair_dispatch *record = calloc(1, sizeof(*record));
   if (!record)
      return;
   record->info.role = cert->role;
   if (cert->role == HK_COMPUTE_PAIR_PAD4) {
      record->extra.pad4 = cert->extra.pad4;
      memcpy(record->bindings, cert->bindings, 3 * sizeof(record->bindings[0]));
   } else if (cert->role == HK_COMPUTE_PAIR_IM2COL) {
      record->extra.im2col = cert->extra.im2col;
      memcpy(record->bindings, cert->bindings, 2 * sizeof(record->bindings[0]));
   } else if (cert->role == HK_COMPUTE_PAIR_FADD) {
      record->info.fadd.push_bytes = api->identity_copy4_info.push_bytes;
      record->info.fadd.fp_math_ctrl = api->identity_copy4_info.fadd_fp_math_ctrl;
      record->info.fadd.float_controls_execution_mode =
         api->identity_copy4_info.fadd_float_controls_execution_mode;
      record->info.fadd.meta_valid = api->identity_copy4_info.fadd_meta_valid;
      record->extra.copy4 = cert->extra.copy4;
      memcpy(record->bindings, cert->bindings, 3 * sizeof(record->bindings[0]));
      memcpy(record->exec_local, selected->b.info.workgroup_size,
             sizeof(record->exec_local));
      record->root_va = desc->uploaded_root.gpu;
      memcpy(record->root_sets, desc->root.sets, sizeof(record->root_sets));
      memcpy(record->push, desc->root.push, sizeof(record->push));
      record->cs_group_count_addr = desc->root.cs.group_count_addr;
      memcpy(record->cs_base_group, desc->root.cs.base_group,
             sizeof(record->cs_base_group));
      record->printf_buffer_address = desc->root.printf_buffer_address;
      record->alt_kind = HK_COMPUTE_PAIR_ALT_LINEAR_FADD;
      record->producer = hk_compute_pair_producer_ref(api->compute_pair_producer);
   } else {
      record->extra.copy4 = cert->extra.copy4;
      memcpy(record->bindings, cert->bindings, 2 * sizeof(record->bindings[0]));
   }
   memcpy(record->groups, orig_groups, sizeof(record->groups));
   memcpy(record->exec_threads, exec_grid.count, sizeof(record->exec_threads));
   record->overlap_safe = hk_compute_shader_overlap_safe(selected, desc);
   cs->compute_pair = record;
}

static struct hk_shader *
hk_select_tensor_op(struct hk_cmd_buffer *cmd, struct hk_api_shader *api,
                    struct hk_shader *normal, struct agx_grid *grid)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   const struct hk_tensor_op_info *info = &api->tensor_op_info;

   if (!api->tensor_op_primary || !agx_chip_native_or_m5(dev->dev.chip) ||
       cmd->in_meta || cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY ||
       grid->mode != AGX_CDM_MODE_DIRECT ||
       !hk_nir_dispatch_zero_base(desc->root.cs.base_group) ||
       !hk_identity_copy4_supported_local(info->local) ||
       memcmp(info->local, normal->b.info.workgroup_size, sizeof(info->local)) ||
       grid->count[2] != 1)
      return normal;

   if ((uint64_t)grid->count[0] * info->local[0] > (uint64_t)INT32_MAX ||
       (uint64_t)grid->count[1] * info->local[1] > (uint64_t)INT32_MAX)
      return normal;

   struct hk_shader *alt = NULL;
   uint32_t new_grid[3] = {0, 0, 0};
   uint64_t src_bytes = 0, dst_start = 0, dst_bytes = 0;
   unsigned align = 4;

   if (info->kind == HK_TENSOR_OP_CONCAT_X) {
      int32_t dim_n = 1, dim_h = 0, dim_q = 0, dim_c = 0;
      int32_t dst_sy = 0, dst_sq = 0, dst_ox = 0;
      if ((info->dim_n_off != UINT16_MAX &&
           !hk_read_push_i32(desc, info->dim_n_off, &dim_n)) ||
          !hk_read_push_i32(desc, info->dim_h_off, &dim_h) ||
          !hk_read_push_i32(desc, info->dim_q_off, &dim_q) ||
          !hk_read_push_i32(desc, info->dim_c_off, &dim_c) ||
          !hk_read_push_i32(desc, info->dst_stride_y_off, &dst_sy) ||
          !hk_read_push_i32(desc, info->dst_stride_q_off, &dst_sq) ||
          (info->dst_off_x_off != UINT16_MAX &&
           !hk_read_push_i32(desc, info->dst_off_x_off, &dst_ox)))
         return normal;

      if (dim_n <= 0 || dim_h <= 0 || dim_q <= 0 || dim_c <= 0 ||
          dst_sy < 0 || dst_sq < 0 || dst_ox < 0)
         return normal;

      uint64_t x_lim = 0, y_lim = 0, src_elems = 0;
      if (__builtin_mul_overflow((uint64_t)(uint32_t)dim_q, (uint64_t)(uint32_t)dim_c, &x_lim) ||
          __builtin_mul_overflow((uint64_t)(uint32_t)dim_n, (uint64_t)(uint32_t)dim_h, &y_lim) ||
          x_lim > (uint64_t)INT32_MAX || y_lim > (uint64_t)INT32_MAX ||
          __builtin_mul_overflow(y_lim, x_lim, &src_elems) ||
          src_elems > (uint64_t)INT32_MAX / 4u)
         return normal;

      if (grid->count[0] < DIV_ROUND_UP((uint32_t)x_lim, info->local[0]) ||
          grid->count[1] < DIV_ROUND_UP((uint32_t)y_lim, info->local[1]))
         return normal;

      /* Require distinct destination elements so vectorized stores match scalar order. */
      uint64_t q_span = 0, y_span = 0;
      if (__builtin_mul_overflow((uint64_t)(uint32_t)(dim_q - 1), (uint64_t)(uint32_t)dst_sq, &q_span) ||
          __builtin_mul_overflow(y_lim - 1, (uint64_t)(uint32_t)dst_sy, &y_span))
         return normal;
      const bool non_overlap_qy =
         (dim_q == 1 || (uint64_t)(uint32_t)dst_sq >= (uint64_t)(uint32_t)dim_c) &&
         (y_lim == 1 || (uint64_t)(uint32_t)dst_sy >= q_span + (uint64_t)(uint32_t)dim_c);
      const bool non_overlap_yq =
         (y_lim == 1 || (uint64_t)(uint32_t)dst_sy >= (uint64_t)(uint32_t)dim_c) &&
         (dim_q == 1 || (uint64_t)(uint32_t)dst_sq >= y_span + (uint64_t)(uint32_t)dim_c);
      if (!non_overlap_qy && !non_overlap_yq)
         return normal;

      uint64_t dst_max = 0;
      if (__builtin_add_overflow(q_span, y_span, &dst_max) ||
          __builtin_add_overflow(dst_max, (uint64_t)(uint32_t)dst_ox, &dst_max) ||
          __builtin_add_overflow(dst_max, (uint64_t)(uint32_t)(dim_c - 1), &dst_max) ||
          dst_max >= (uint64_t)INT32_MAX / 4u)
         return normal;

      const bool contig = (dst_sq == dim_c);
      if (contig || dim_c == 1 || dim_c == 2 || dim_c == 4) {
         alt = api->tensor_op_primary;
         const uint32_t or_row = (uint32_t)x_lim | (uint32_t)dst_ox | (uint32_t)dst_sy;
         const uint32_t or_str = (uint32_t)dst_ox | (uint32_t)dst_sy | (uint32_t)dst_sq;
         const bool is_v4 = contig ? ((or_row & 3u) == 0) : (dim_c == 4 && (or_str & 3u) == 0);
         const bool is_v2 = contig ? ((or_row & 1u) == 0) : (dim_c == 2 && (or_str & 1u) == 0);
         const unsigned vec = is_v4 ? 4u : (is_v2 ? 2u : 1u);
         align = vec * 4u;
         const uint32_t x_cnt = contig ? (uint32_t)(x_lim / vec) : (uint32_t)dim_q;
         if (!alt->b.info.workgroup_size[0] || !alt->b.info.workgroup_size[1] ||
             alt->b.info.workgroup_size[2] != 1)
            return normal;
         new_grid[0] = DIV_ROUND_UP(x_cnt, alt->b.info.workgroup_size[0]);
         new_grid[1] = DIV_ROUND_UP((uint32_t)y_lim, alt->b.info.workgroup_size[1]);
         new_grid[2] = 1;
      } else {
         alt = api->tensor_op_secondary;
         if (!alt || !alt->b.info.workgroup_size[0] || !alt->b.info.workgroup_size[1] ||
             !alt->b.info.workgroup_size[2])
            return normal;
         const uint32_t or_all =
            (uint32_t)dim_c | (uint32_t)dst_ox | (uint32_t)dst_sy | (uint32_t)dst_sq;
         const unsigned vec = ((or_all & 3u) == 0) ? 4u : (((or_all & 1u) == 0) ? 2u : 1u);
         align = vec * 4u;
         const uint32_t c_units = (uint32_t)dim_c / vec;
         new_grid[0] = DIV_ROUND_UP(c_units, alt->b.info.workgroup_size[0]);
         new_grid[1] = DIV_ROUND_UP((uint32_t)dim_q, alt->b.info.workgroup_size[1]);
         new_grid[2] = DIV_ROUND_UP((uint32_t)y_lim, alt->b.info.workgroup_size[2]);
      }

      src_bytes = src_elems * sizeof(uint32_t);
      dst_start = (uint64_t)(uint32_t)dst_ox * sizeof(uint32_t);
      dst_bytes = (dst_max - (uint64_t)(uint32_t)dst_ox + 1u) * sizeof(uint32_t);
   } else if (info->kind == HK_TENSOR_OP_CONCAT_Y) {
      int32_t dim_n = 0, dim_h = 0, dim_w = 1, dim_c = 0;
      int32_t dst_sn = 0, dst_yo = 0;
      if (!hk_read_push_i32(desc, info->dim_n_off, &dim_n) ||
          !hk_read_push_i32(desc, info->dim_h_off, &dim_h) ||
          (info->dim_q_off != UINT16_MAX &&
           !hk_read_push_i32(desc, info->dim_q_off, &dim_w)) ||
          !hk_read_push_i32(desc, info->dim_c_off, &dim_c) ||
          !hk_read_push_i32(desc, info->dst_stride_y_off, &dst_sn) ||
          (info->dst_off_x_off != UINT16_MAX &&
           !hk_read_push_i32(desc, info->dst_off_x_off, &dst_yo)))
         return normal;

      if (dim_n <= 0 || dim_h <= 0 || dim_w <= 0 || dim_c <= 0 ||
          dst_sn < 0 || dst_yo < 0 || (dim_n > 1 && dst_sn < dim_h))
         return normal;

      uint64_t x_lim = 0, y_lim = 0, src_elems = 0;
      if (__builtin_mul_overflow((uint64_t)(uint32_t)dim_w, (uint64_t)(uint32_t)dim_c, &x_lim) ||
          __builtin_mul_overflow((uint64_t)(uint32_t)dim_n, (uint64_t)(uint32_t)dim_h, &y_lim) ||
          x_lim > (uint64_t)INT32_MAX || y_lim > (uint64_t)INT32_MAX ||
          __builtin_mul_overflow(y_lim, x_lim, &src_elems) ||
          src_elems > (uint64_t)INT32_MAX / 4u)
         return normal;

      if (grid->count[0] < DIV_ROUND_UP((uint32_t)x_lim, info->local[0]) ||
          grid->count[1] < DIV_ROUND_UP((uint32_t)y_lim, info->local[1]))
         return normal;

      uint64_t dst_row_max = 0, dst_min_el = 0, dst_end_el = 0;
      if (__builtin_mul_overflow((uint64_t)(uint32_t)(dim_n - 1), (uint64_t)(uint32_t)dst_sn, &dst_row_max) ||
          __builtin_add_overflow(dst_row_max, (uint64_t)(uint32_t)dst_yo, &dst_row_max) ||
          __builtin_add_overflow(dst_row_max, (uint64_t)(uint32_t)(dim_h - 1), &dst_row_max) ||
          __builtin_mul_overflow((uint64_t)(uint32_t)dst_yo, x_lim, &dst_min_el) ||
          __builtin_mul_overflow(dst_row_max + 1u, x_lim, &dst_end_el) ||
          dst_end_el > (uint64_t)INT32_MAX / 4u)
         return normal;

      bool use_1d = false;
      if (dim_n == 1 && ((src_elems & 3u) == 0) && ((dst_min_el & 3u) == 0) &&
          api->tensor_op_primary->b.info.workgroup_size[0] > 0 &&
          api->tensor_op_primary->b.info.workgroup_size[1] == 1 &&
          api->tensor_op_primary->b.info.workgroup_size[2] == 1) {
         const uint64_t total4 = src_elems / 4u;
         const uint64_t wg_chunk =
            (uint64_t)api->tensor_op_primary->b.info.workgroup_size[0] * 4u;
         const uint64_t gx_1d = DIV_ROUND_UP(total4, wg_chunk);
         if (gx_1d > 0 && gx_1d <= 65535u) {
            alt = api->tensor_op_primary;
            align = 16;
            new_grid[0] = (uint32_t)gx_1d;
            new_grid[1] = 1;
            new_grid[2] = 1;
            use_1d = true;
         }
      }
      if (!use_1d) {
         alt = api->tensor_op_secondary;
         if (!alt || !alt->b.info.workgroup_size[0] || !alt->b.info.workgroup_size[1] ||
             !alt->b.info.workgroup_size[2])
            return normal;
         const unsigned vec = ((x_lim & 3u) == 0) ? 4u : (((x_lim & 1u) == 0) ? 2u : 1u);
         align = vec * 4u;
         const uint32_t x_units = (uint32_t)(x_lim / vec);
         new_grid[0] = DIV_ROUND_UP(x_units, alt->b.info.workgroup_size[0]);
         new_grid[1] = DIV_ROUND_UP((uint32_t)dim_h, alt->b.info.workgroup_size[1]);
         new_grid[2] = DIV_ROUND_UP((uint32_t)dim_n, alt->b.info.workgroup_size[2]);
      }

      src_bytes = src_elems * sizeof(uint32_t);
      dst_start = dst_min_el * sizeof(uint32_t);
      dst_bytes = (dst_end_el - dst_min_el) * sizeof(uint32_t);
   } else if (info->kind == HK_TENSOR_OP_RESAMPLE_NEAREST) {
      int32_t out_w = 0, out_h = 0, ch = 0, in_w = 0;
      int32_t sx_bits = 0, sy_bits = 0;
      if (!hk_read_push_i32(desc, info->dim_q_off, &out_w) ||
          !hk_read_push_i32(desc, info->dim_h_off, &out_h) ||
          !hk_read_push_i32(desc, info->dim_c_off, &ch) ||
          !hk_read_push_i32(desc, info->dst_stride_y_off, &in_w) ||
          !hk_read_push_i32(desc, info->dst_stride_q_off, &sx_bits) ||
          !hk_read_push_i32(desc, info->scale_y_off, &sy_bits))
         return normal;

      float scale_x = 0.0f, scale_y = 0.0f;
      memcpy(&scale_x, &sx_bits, sizeof(scale_x));
      memcpy(&scale_y, &sy_bits, sizeof(scale_y));
      if (out_w <= 0 || out_h <= 0 || ch <= 0 || in_w <= 0 ||
          !isfinite(scale_x) || !isfinite(scale_y) ||
          scale_x < 0.0f || scale_y < 0.0f)
         return normal;

      const float max_sx_f = floorf((float)(out_w - 1) * scale_x);
      const float max_sy_f = floorf((float)(out_h - 1) * scale_y);
      if (!isfinite(max_sx_f) || !isfinite(max_sy_f) ||
          max_sx_f < 0.0f || max_sx_f >= (float)in_w ||
          max_sy_f < 0.0f || max_sy_f > (float)(INT32_MAX / 4))
         return normal;

      uint64_t x_lim = 0, dst_elems = 0, src_px = 0, src_elems = 0;
      const uint64_t max_sx = (uint64_t)(int32_t)max_sx_f;
      const uint64_t max_sy = (uint64_t)(int32_t)max_sy_f;
      if (__builtin_mul_overflow((uint64_t)(uint32_t)out_w, (uint64_t)(uint32_t)ch, &x_lim) ||
          x_lim > (uint64_t)INT32_MAX ||
          __builtin_mul_overflow((uint64_t)(uint32_t)out_h, x_lim, &dst_elems) ||
          dst_elems > (uint64_t)INT32_MAX / 4u ||
          __builtin_mul_overflow(max_sy, (uint64_t)(uint32_t)in_w, &src_px) ||
          __builtin_add_overflow(src_px, max_sx + 1u, &src_px) ||
          __builtin_mul_overflow(src_px, (uint64_t)(uint32_t)ch, &src_elems) ||
          src_elems > (uint64_t)INT32_MAX / 4u)
         return normal;

      if (grid->count[0] < DIV_ROUND_UP((uint32_t)x_lim, info->local[0]) ||
          grid->count[1] < DIV_ROUND_UP((uint32_t)out_h, info->local[1]))
         return normal;

      alt = api->tensor_op_primary;
      if (!alt->b.info.workgroup_size[0] || !alt->b.info.workgroup_size[1] ||
          !alt->b.info.workgroup_size[2])
         return normal;
      const unsigned vec = ((ch & 3) == 0) ? 4u : (((ch & 1) == 0) ? 2u : 1u);
      align = vec * 4u;
      const uint32_t c_units = (uint32_t)ch / vec;
      new_grid[0] = DIV_ROUND_UP(c_units, alt->b.info.workgroup_size[0]);
      new_grid[1] = DIV_ROUND_UP((uint32_t)out_w, alt->b.info.workgroup_size[1]);
      new_grid[2] = DIV_ROUND_UP((uint32_t)out_h, alt->b.info.workgroup_size[2]);

      src_bytes = src_elems * sizeof(uint32_t);
      dst_start = 0;
      dst_bytes = dst_elems * sizeof(uint32_t);
   } else {
      return normal;
   }

   if (!alt || !hk_g17p_compute_shader_supported(dev->dev.chip, alt, alt->only_linked, 1u << 16) ||
       !new_grid[0] || new_grid[0] > 65535u ||
       !new_grid[1] || new_grid[1] > 65535u ||
       !new_grid[2] || new_grid[2] > 65535u)
      return normal;

   struct hk_storage_backing src_b, dst_b;
   if (!hk_check_storage_backing(desc, info->src_set, info->src_binding, 0, src_bytes, align, &src_b) ||
       !hk_check_storage_backing(desc, info->dst_set, info->dst_binding, dst_start, dst_bytes, align, &dst_b) ||
       !hk_intervals_disjoint(&src_b, 0, src_bytes, &dst_b, dst_start, dst_bytes))
      return normal;

   grid->count[0] = new_grid[0];
   grid->count[1] = new_grid[1];
   grid->count[2] = new_grid[2];
   return alt;
}

static struct hk_shader *
hk_select_identity_copy4(struct hk_cmd_buffer *cmd, struct hk_api_shader *api,
                         struct hk_shader *normal, struct agx_grid *grid,
                         struct hk_copy4_dispatch_cert *cert_out,
                         uint64_t *loop_bound_out,
                         bool *im2col_push_valid_out,
                         uint32_t *im2col_push_off_out,
                         struct hk_im2col_push *im2col_push_out)
{
   struct hk_shader *copy = api->identity_copy4;
   const struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!copy)
      return normal;
   if (!agx_chip_native_or_m5(dev->dev.chip) ||
       cmd->in_meta || cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
      return normal;
   if (grid->mode != AGX_CDM_MODE_DIRECT || desc->root.cs.base_group[0] ||
       desc->root.cs.base_group[1] || desc->root.cs.base_group[2])
      return normal;
   if (!hk_g17p_compute_shader_supported(dev->dev.chip, copy, copy->only_linked, 1u << 16))
      return normal;

   const struct hk_identity_copy4 *info = &api->identity_copy4_info;
   if (!hk_identity_copy4_supported_local(info->local) ||
       memcmp(info->local, normal->b.info.workgroup_size, sizeof(info->local)) ||
       copy->b.info.workgroup_size[0] == 0 ||
       copy->b.info.workgroup_size[1] == 0 ||
       copy->b.info.workgroup_size[2] != 1)
      return normal;

   if (info->kind == HK_IDENTITY_COPY4_PAD4) {
      int32_t in_s[4], out_s[4];
      for (unsigned d = 0; d < 4; ++d) {
         if (!hk_read_push_i32(desc, info->in_shape_off[d], &in_s[d]) ||
             !hk_read_push_i32(desc, info->out_shape_off[d], &out_s[d]) ||
             in_s[d] <= 0 || out_s[d] <= 0)
            return normal;
      }
      if ((in_s[3] % 4) != 0 || (out_s[3] % 4) != 0)
         return normal;
      uint64_t in_elems = 1, out_elems = 1;
      for (unsigned d = 0; d < 4; ++d) {
         if (__builtin_mul_overflow(in_elems, (uint64_t)(uint32_t)in_s[d], &in_elems) ||
             __builtin_mul_overflow(out_elems, (uint64_t)(uint32_t)out_s[d], &out_elems))
            return normal;
      }
      const uint64_t in_bytes = in_elems * sizeof(uint32_t);
      const uint64_t out_bytes = out_elems * sizeof(uint32_t);
      if (in_elems > (uint64_t)INT32_MAX / 4 || out_elems > (uint64_t)INT32_MAX / 4)
         return normal;
      const uint64_t orig_x = (uint64_t)(uint32_t)out_s[2] * (uint32_t)out_s[3];
      const uint64_t orig_y = (uint64_t)(uint32_t)out_s[0] * (uint32_t)out_s[1];
      if (grid->count[2] != 1 ||
          orig_x > (uint64_t)INT32_MAX || orig_y > (uint64_t)INT32_MAX ||
          grid->count[0] < DIV_ROUND_UP((uint32_t)orig_x, info->local[0]) ||
          grid->count[1] < DIV_ROUND_UP((uint32_t)orig_y, info->local[1]) ||
          (uint64_t)grid->count[0] * info->local[0] > (uint64_t)INT32_MAX ||
          (uint64_t)grid->count[1] * info->local[1] > (uint64_t)INT32_MAX)
         return normal;
      uint32_t max_pad_idx = 0;
      for (unsigned d = 0; d < 4; ++d)
         max_pad_idx = MAX2(max_pad_idx, info->pad_idx[d]);
      const uint64_t pad_bytes = ((uint64_t)max_pad_idx + 1) * sizeof(uint32_t);
      struct hk_storage_backing src_b, pad_b, dst_b;
      if (!hk_check_storage_backing(desc, info->src_set, info->src_binding, 0, in_bytes, 16, &src_b) ||
          !hk_check_storage_backing(desc, info->pad_set, info->pad_binding, 0, pad_bytes, 4, &pad_b) ||
          !hk_check_storage_backing(desc, info->dst_set, info->dst_binding, 0, out_bytes, 16, &dst_b))
         return normal;
      if (!hk_intervals_disjoint(&src_b, 0, in_bytes, &dst_b, 0, out_bytes) ||
          !hk_intervals_disjoint(&pad_b, 0, pad_bytes, &dst_b, 0, out_bytes))
         return normal;

      const uint64_t out_x = (uint64_t)(uint32_t)out_s[2] * ((uint32_t)out_s[3] / 4);
      const uint64_t out_y = (uint64_t)(uint32_t)out_s[0] * (uint32_t)out_s[1];
      if (out_x > UINT32_MAX || out_y > UINT32_MAX)
         return normal;
      if (cert_out) {
         *cert_out = (struct hk_copy4_dispatch_cert){
            .role = HK_COMPUTE_PAIR_PAD4,
            .bindings = {src_b, pad_b, dst_b},
            .extra.pad4 = {
               .in_shape = {(uint32_t)in_s[0], (uint32_t)in_s[1],
                            (uint32_t)in_s[2], (uint32_t)in_s[3]},
               .out_shape = {(uint32_t)out_s[0], (uint32_t)out_s[1],
                             (uint32_t)out_s[2], (uint32_t)out_s[3]},
               .pad_idx = {info->pad_idx[0], info->pad_idx[1],
                           info->pad_idx[2], info->pad_idx[3]},
               .in_bytes = in_bytes,
               .out_bytes = out_bytes,
               .pad_bytes = pad_bytes,
            },
         };
      }
      grid->count[0] = DIV_ROUND_UP((uint32_t)out_x, copy->b.info.workgroup_size[0]);
      grid->count[1] = DIV_ROUND_UP((uint32_t)out_y, copy->b.info.workgroup_size[1]);
      grid->count[2] = 1;
      return copy;
   }

   if (copy->b.info.workgroup_size[1] != 1)
      return normal;

   uint64_t bytes = 0;
   uint64_t src_start = 0, dst_start = 0;
   if (info->kind == HK_IDENTITY_COPY4_IM2COL_1X1) {
      uint32_t k;
      if (!hk_identity_copy4_required_bytes(desc, info, *grid, info->local, &bytes, &k))
         return hk_select_im2col_vec4(cmd, api, normal, grid, cert_out,
                                      loop_bound_out, im2col_push_valid_out,
                                      im2col_push_off_out, im2col_push_out);
   } else if (info->kind == HK_IDENTITY_COPY4_LINEAR_COPY ||
              info->kind == HK_IDENTITY_COPY4_LINEAR_OFFSET ||
              info->kind == HK_IDENTITY_COPY4_LINEAR_FADD) {
      int32_t count = 0;
      if (!hk_read_push_i32(desc, info->count_offset, &count))
         return normal;
      if (count <= 0 || (count % 4) != 0)
         return normal;
      if (grid->count[1] != 1 || grid->count[2] != 1 ||
          grid->count[0] < DIV_ROUND_UP((uint32_t)count, info->local[0]) ||
          (uint64_t)grid->count[0] * info->local[0] > (uint64_t)INT32_MAX)
         return normal;
      bytes = (uint64_t)(uint32_t)count * sizeof(uint32_t);
      if (bytes > (uint64_t)INT32_MAX + 1)
         return normal;
      if (info->kind == HK_IDENTITY_COPY4_LINEAR_OFFSET) {
         int32_t so = 0, doff = 0;
         if (!hk_read_push_i32(desc, info->src_offset_offset, &so) ||
             !hk_read_push_i32(desc, info->dst_offset_offset, &doff))
            return normal;
         if (so < 0 || doff < 0 || (so % 4) != 0 || (doff % 4) != 0 ||
             (uint64_t)(uint32_t)so + (uint32_t)count > (uint64_t)INT32_MAX ||
             (uint64_t)(uint32_t)doff + (uint32_t)count > (uint64_t)INT32_MAX)
            return normal;
         src_start = (uint64_t)(uint32_t)so * sizeof(uint32_t);
         dst_start = (uint64_t)(uint32_t)doff * sizeof(uint32_t);
      }
   } else {
      return normal;
   }

   const unsigned nsrc = info->src_count ? info->src_count : 1;
   if (nsrc > 2)
      return normal;
   const uint8_t sets[3] = {info->src_set, info->src1_set, info->dst_set};
   const uint16_t bindings[3] = {info->src_binding, info->src1_binding, info->dst_binding};
   const uint64_t starts[3] = {src_start, src_start, dst_start};
   struct hk_storage_backing backings[3];
   for (unsigned i = 0; i < 3; i++) {
      if (i == 1 && nsrc < 2)
         continue;
      if (!hk_check_storage_backing(desc, sets[i], bindings[i], starts[i], bytes, 16, &backings[i]))
         return normal;
   }

   const bool kind_allows_inplace =
      info->kind == HK_IDENTITY_COPY4_LINEAR_COPY ||
      info->kind == HK_IDENTITY_COPY4_LINEAR_FADD;
   bool same_src0_dst = false;
   if (!hk_copy4_classify_intervals(kind_allows_inplace, nsrc, backings, src_start,
                                    dst_start, bytes, &same_src0_dst))
      return normal;

   if (cert_out && nsrc == 1) {
      *cert_out = (struct hk_copy4_dispatch_cert){
         .role = same_src0_dst ? HK_COMPUTE_PAIR_SELF_COPY : HK_COMPUTE_PAIR_COPY4,
         .bindings = {backings[0], backings[2]},
         .extra.copy4 = {
            .src_start = src_start,
            .dst_start = dst_start,
            .bytes = bytes,
         },
      };
   } else if (cert_out && info->kind == HK_IDENTITY_COPY4_LINEAR_FADD &&
              nsrc == 2 && api->compute_pair_producer &&
              api->compute_pair_producer->info.role == HK_COMPUTE_PAIR_FADD) {
      *cert_out = (struct hk_copy4_dispatch_cert){
         .role = HK_COMPUTE_PAIR_FADD,
         .bindings = {backings[0], backings[1], backings[2]},
         .extra.copy4 = {
            .src_start = src_start,
            .dst_start = dst_start,
            .bytes = bytes,
         },
      };
   }

   /* Flattened 1D vec4 grid: each invocation transfers up to 4 coalesced vec4s. */
   const uint32_t total_vec4 = (uint32_t)(bytes / 16);
   const uint32_t wg_vec4 = (uint32_t)copy->b.info.workgroup_size[0] * 4;
   grid->count[0] = DIV_ROUND_UP(total_vec4, wg_vec4);
   grid->count[1] = 1;
   grid->count[2] = 1;
   return copy;
}

static struct hk_shader *
hk_select_depthwise_vec4(struct hk_cmd_buffer *cmd, struct hk_api_shader *api,
                         struct hk_shader *normal, struct agx_grid *grid,
                         uint64_t pack_loop_params[HK_PACK_LOOP_PARAM_COUNT],
                         uint64_t *loop_bound_out)
{
   struct hk_shader *alt = api->depthwise_vec4;
   const struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!alt || !api->compute_pair ||
       api->compute_pair->role != HK_COMPUTE_PAIR_DEPTHWISE)
      return normal;
   if (!agx_chip_native_or_m5(dev->dev.chip) ||
       cmd->in_meta || cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
      return normal;
   if (grid->mode != AGX_CDM_MODE_DIRECT ||
       !hk_nir_dispatch_zero_base(desc->root.cs.base_group))
      return normal;
   if (!hk_g17p_compute_shader_supported(dev->dev.chip, alt, alt->only_linked, 1u << 16) ||
       alt->b.info.workgroup_size[0] == 0 || alt->b.info.workgroup_size[1] == 0 ||
       alt->b.info.workgroup_size[2] == 0)
      return normal;

   const struct hk_nir_depthwise_output *dw = &api->compute_pair->depthwise;
   if (!dw->vec4_valid || dw->push_bytes > HK_COMPUTE_OPT_PUSH_BYTES)
      return normal;
   if (dw->local_size[0] != normal->b.info.workgroup_size[0] ||
       dw->local_size[1] != normal->b.info.workgroup_size[1] ||
       dw->local_size[2] != normal->b.info.workgroup_size[2])
      return normal;

   int32_t iw, ih, ow, oh, kw, kh, sx, sy, dx, dy, px, py, ch, has_bias = 0;
   if (!hk_read_push_i32(desc, dw->input_width_offset_B, &iw) ||
       !hk_read_push_i32(desc, dw->input_height_offset_B, &ih) ||
       !hk_read_push_i32(desc, dw->width_offset_B, &ow) ||
       !hk_read_push_i32(desc, dw->height_offset_B, &oh) ||
       !hk_read_push_i32(desc, dw->kernel_width_offset_B, &kw) ||
       !hk_read_push_i32(desc, dw->kernel_height_offset_B, &kh) ||
       !hk_read_push_i32(desc, dw->stride_x_offset_B, &sx) ||
       !hk_read_push_i32(desc, dw->stride_y_offset_B, &sy) ||
       !hk_read_push_i32(desc, dw->dil_x_offset_B, &dx) ||
       !hk_read_push_i32(desc, dw->dil_y_offset_B, &dy) ||
       !hk_read_push_i32(desc, dw->pad_x_offset_B, &px) ||
       !hk_read_push_i32(desc, dw->pad_y_offset_B, &py) ||
       !hk_read_push_i32(desc, dw->channels_offset_B, &ch))
      return normal;
   if (dw->has_bias_buffer) {
      if (dw->has_bias_offset_B == UINT32_MAX)
         has_bias = 1;
      else if (!hk_read_push_i32(desc, dw->has_bias_offset_B, &has_bias))
         return normal;
   }

   if (iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0 || kw <= 0 || kh <= 0 ||
       sx <= 0 || sy <= 0 || dx <= 0 || dy <= 0 ||
       px < -1048576 || px > 1048576 || py < -1048576 || py > 1048576 ||
       (uint64_t)ow * (uint32_t)sx + (uint64_t)kw * (uint32_t)dx > 0x3fffffffu ||
       (uint64_t)oh * (uint32_t)sy + (uint64_t)kh * (uint32_t)dy > 0x3fffffffu ||
       ch <= 0 || (ch % 4) != 0 || (has_bias != 0 && has_bias != 1))
      return normal;

   const uint64_t orig_x = (uint64_t)(uint32_t)ow * (uint32_t)ch;
   if (orig_x > (uint64_t)INT32_MAX ||
       grid->count[2] != 1 ||
       grid->count[0] < DIV_ROUND_UP((uint32_t)orig_x, dw->local_size[0]) ||
       grid->count[1] < DIV_ROUND_UP((uint32_t)oh, dw->local_size[1]) ||
       (uint64_t)grid->count[0] * dw->local_size[0] > (uint64_t)INT32_MAX ||
       (uint64_t)grid->count[1] * dw->local_size[1] > (uint64_t)INT32_MAX)
      return normal;

   const uint64_t in_bytes = (uint64_t)(uint32_t)iw * (uint32_t)ih * (uint32_t)ch * sizeof(uint32_t);
   const uint64_t wt_bytes = (uint64_t)(uint32_t)kw * (uint32_t)kh * (uint32_t)ch * sizeof(uint32_t);
   const uint64_t bs_bytes = has_bias ? (uint64_t)(uint32_t)ch * sizeof(uint32_t) : 0;
   const uint64_t out_bytes = orig_x * (uint32_t)oh * sizeof(uint32_t);
   if (in_bytes > (uint64_t)INT32_MAX + 1 || wt_bytes > (uint64_t)INT32_MAX + 1 ||
       out_bytes > (uint64_t)INT32_MAX + 1)
      return normal;

   if ((dw->reads[0].offset_B | dw->reads[1].offset_B |
        (has_bias ? dw->reads[2].offset_B : 0) | dw->output.offset_B) % 16)
      return normal;

   struct hk_storage_backing in_b, wt_b, bs_b, out_b;
   if (!hk_check_storage_backing(desc, dw->reads[0].set, dw->reads[0].binding,
                                 dw->reads[0].offset_B, in_bytes, 16, &in_b) ||
       !hk_check_storage_backing(desc, dw->reads[1].set, dw->reads[1].binding,
                                 dw->reads[1].offset_B, wt_bytes, 16, &wt_b) ||
       !hk_check_storage_backing(desc, dw->output.set, dw->output.binding,
                                 dw->output.offset_B, out_bytes, 16, &out_b))
      return normal;
   if (has_bias &&
       !hk_check_storage_backing(desc, dw->reads[2].set, dw->reads[2].binding,
                                 dw->reads[2].offset_B, bs_bytes, 16, &bs_b))
      return normal;

   if (!hk_intervals_disjoint(&in_b, dw->reads[0].offset_B, in_bytes,
                              &out_b, dw->output.offset_B, out_bytes) ||
       !hk_intervals_disjoint(&wt_b, dw->reads[1].offset_B, wt_bytes,
                              &out_b, dw->output.offset_B, out_bytes) ||
       (has_bias &&
        !hk_intervals_disjoint(&bs_b, dw->reads[2].offset_B, bs_bytes,
                               &out_b, dw->output.offset_B, out_bytes)))
      return normal;

   const uint64_t out_pairs = DIV_ROUND_UP((uint64_t)(uint32_t)ow, 2);
   if (out_pairs > UINT32_MAX)
      return normal;
   const uint32_t gx = DIV_ROUND_UP((uint32_t)ch / 4, alt->b.info.workgroup_size[0]);
   const uint32_t gy = DIV_ROUND_UP((uint32_t)out_pairs, alt->b.info.workgroup_size[1]);
   const uint32_t gz = DIV_ROUND_UP((uint32_t)oh, alt->b.info.workgroup_size[2]);
   if (gx > 65535 || gy > 65535 || gz > 65535)
      return normal;
   grid->count[0] = gx;
   grid->count[1] = gy;
   grid->count[2] = gz;
   if (pack_loop_params) {
      pack_loop_params[HK_PACK_LOOP_PARAM_DW_KH] = (uint32_t)kh;
      pack_loop_params[HK_PACK_LOOP_PARAM_DW_KW] = (uint32_t)kw;
   }
   if (loop_bound_out) {
      uint64_t params[HK_PACK_LOOP_PARAM_COUNT] = {0};
      params[HK_PACK_LOOP_PARAM_DW_KH] = (uint32_t)kh;
      params[HK_PACK_LOOP_PARAM_DW_KW] = (uint32_t)kw;
      hk_pack_eval_effective_instrs(&alt->pack_loop_bound,
                                    alt->b.info.stats.instrs, params,
                                    loop_bound_out);
   }
   return alt;
}

static struct hk_shader *
hk_select_gemm_cmat(struct hk_cmd_buffer *cmd, struct hk_api_shader *api,
                    struct hk_shader *normal, struct agx_grid *grid,
                    enum hk_compute_pair_alt_kind *alt_kind_out,
                    uint64_t pack_loop_params[HK_PACK_LOOP_PARAM_COUNT],
                    uint64_t *loop_bound_out)
{
   const struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if ((!api->gemm_cmat16 && !api->gemm_cmat32 && !api->gemm_cmat64) ||
       !api->compute_pair ||
       api->compute_pair->role != HK_COMPUTE_PAIR_GEMM)
      return normal;
   if (!agx_chip_native_or_m5(dev->dev.chip) ||
       !agx_chip_cooperative_matrix_supported(dev->dev.chip) ||
       cmd->in_meta || cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
      return normal;
   if (grid->mode != AGX_CDM_MODE_DIRECT || desc->root.cs.base_group[0] ||
       desc->root.cs.base_group[1] || desc->root.cs.base_group[2] ||
       grid->count[2] != 1)
      return normal;

   const struct hk_nir_gemm_output *gemm = &api->compute_pair->gemm;
   if (!gemm->cmat_valid || gemm->push_bytes > HK_COMPUTE_OPT_PUSH_BYTES ||
       gemm->local_size[0] != normal->b.info.workgroup_size[0] ||
       gemm->local_size[1] != normal->b.info.workgroup_size[1] ||
       gemm->local_size[2] != normal->b.info.workgroup_size[2])
      return normal;

   int32_t m_i = 0, n_i = 0, k_i = 0, bias_mode = 2;
   if (!hk_read_push_i32(desc, gemm->n_offset_B, &n_i) ||
       !hk_read_push_i32(desc, gemm->k_offset_B, &k_i))
      return normal;
   if (n_i <= 0 || k_i <= 0)
      return normal;

   if (gemm->has_bias_mode) {
      if (!hk_read_push_i32(desc, gemm->bias_mode_offset_B, &bias_mode))
         return normal;
      if (bias_mode < 0 || bias_mode > 2 ||
          !(gemm->bias_mode_mask & (1u << bias_mode)))
         return normal;
   } else if (gemm->has_bias_buffer) {
      if (gemm->bias_mode_mask == (1u << 0))
         bias_mode = 0;
      else if (gemm->bias_mode_mask == (1u << 1))
         bias_mode = 1;
      else
         return normal;
   }

   uint64_t m_u = 0;
   if (gemm->is_guarded) {
      if (!hk_read_push_i32(desc, gemm->m_offset_B, &m_i) || m_i <= 0)
         return normal;
      m_u = (uint32_t)m_i;
   } else {
      m_u = (uint64_t)grid->count[1] * 32;
   }
   if (!hk_gemm_cmat_dispatch_dims_ok(gemm->is_guarded, m_u, (uint32_t)n_i,
                                      (uint32_t)k_i, grid->count))
      return normal;

   /* Tile-width selection (all 128t, 4 KiB shared A): see
    * hk_gemm_cmat_select_tile_n(). Exact N multiples pick the widest exact
    * tile so guarded dispatches can specialize to unguarded variants when
    * M%tile_m==0 and K%8==0; small-grid high-K shapes favor cmat16. */
   uint32_t tile_n = hk_gemm_cmat_select_tile_n(
      m_u, (uint32_t)n_i, (uint32_t)k_i,
      api->gemm_cmat16 != NULL, api->gemm_cmat32 != NULL,
      api->gemm_cmat64 != NULL);
   struct hk_shader *alt =
      tile_n == 16 ? api->gemm_cmat16 :
      tile_n == 32 ? api->gemm_cmat32 :
      tile_n == 64 ? api->gemm_cmat64 : NULL;
   if (!alt ||
       !hk_g17p_compute_shader_supported(dev->dev.chip, alt, alt->only_linked, 1u << 16))
      return normal;

   const uint64_t n_u = (uint32_t)n_i;
   const uint64_t k_u = (uint32_t)k_i;
   uint64_t a_elems = 0, b_elems = 0, c_elems = 0;
   if (__builtin_mul_overflow(m_u, k_u, &a_elems) ||
       __builtin_mul_overflow(k_u, n_u, &b_elems) ||
       __builtin_mul_overflow(m_u, n_u, &c_elems))
      return normal;
   const uint64_t a_bytes = a_elems * sizeof(uint32_t);
   const uint64_t b_bytes = b_elems * sizeof(uint32_t);
   const uint64_t c_bytes = c_elems * sizeof(uint32_t);
   const uint64_t bias_bytes =
      (bias_mode == 0) ? m_u * sizeof(uint32_t) :
      (bias_mode == 1) ? n_u * sizeof(uint32_t) : 0;
   if (a_bytes > (uint64_t)INT32_MAX + 1 ||
       b_bytes > (uint64_t)INT32_MAX + 1 ||
       c_bytes > (uint64_t)INT32_MAX + 1 ||
       bias_bytes > (uint64_t)INT32_MAX + 1)
      return normal;

   const uint32_t a_align = gemm->is_guarded ? 4 : 16;
   if ((gemm->reads[0].offset_B % a_align) || (gemm->reads[1].offset_B % 4) ||
       (bias_bytes && (gemm->reads[2].offset_B % 4)) || (gemm->output.offset_B % 4))
      return normal;

   struct hk_storage_backing a_b, b_b, bs_b, c_b;
   if (!hk_check_storage_backing(desc, gemm->reads[0].set, gemm->reads[0].binding,
                                 gemm->reads[0].offset_B, a_bytes, a_align, &a_b) ||
       !hk_check_storage_backing(desc, gemm->reads[1].set, gemm->reads[1].binding,
                                 gemm->reads[1].offset_B, b_bytes, 4, &b_b) ||
       !hk_check_storage_backing(desc, gemm->output.set, gemm->output.binding,
                                 gemm->output.offset_B, c_bytes, 4, &c_b))
      return normal;
   if (bias_bytes &&
       !hk_check_storage_backing(desc, gemm->reads[2].set, gemm->reads[2].binding,
                                 gemm->reads[2].offset_B, bias_bytes, 4, &bs_b))
      return normal;

   if (!hk_intervals_disjoint(&a_b, gemm->reads[0].offset_B, a_bytes,
                              &c_b, gemm->output.offset_B, c_bytes) ||
       !hk_intervals_disjoint(&b_b, gemm->reads[1].offset_B, b_bytes,
                              &c_b, gemm->output.offset_B, c_bytes) ||
       (bias_bytes &&
        !hk_intervals_disjoint(&bs_b, gemm->reads[2].offset_B, bias_bytes,
                               &c_b, gemm->output.offset_B, c_bytes)))
      return normal;

   enum hk_compute_pair_alt_kind chosen_kind =
      tile_n == 16 ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT16 :
      tile_n == 32 ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT32 :
                     HK_COMPUTE_PAIR_ALT_GEMM_CMAT64;

   /* When an unguarded or exact-multiple guarded GEMM dispatch has sufficient
    * grid parallelism, specialize to wider/taller unguarded cmat tiles:
    * - 64x64 (K_step=16, 4x4 accs/SG, staggered pipe_b) when M%64==0, N%64==0,
    *   K%16==0, and (M/64)*(N/64) >= 25: halves B traffic, WGs, and barriers.
    * - 64x48 (K_step=16, 4x3 accs/SG, staggered pipe_b) when M%64==0, N%48==0,
    *   N%64!=0, K%8==0, and (M/64)*(N/48) >= 25: cuts A traffic 33-50% on N=96/144.
    * - 64x24 (K_step=16, 2x3 accs/SG, pipe_b) for remaining N%24==0.
    * - Otherwise specialize exact-multiple guarded dispatches to *_UNGUARDED. */
   if (api->compute_pair_producer) {
      const bool a_align16 = (gemm->reads[0].offset_B % 16u) == 0 &&
                             ((a_b.address | a_b.offset) % 16ull) == 0;
      const bool can_64x64 =
         (m_u % 64u) == 0 && (n_u % 64u) == 0 && (k_u % 8u) == 0 &&
         k_u >= 64u && (m_u / 64u) * (n_u / 64u) >= 48u && a_align16;
      const bool can_64x48 =
         !can_64x64 && (m_u % 64u) == 0 && (n_u % 48u) == 0 &&
         n_u >= 96u && (n_u % 64u) != 0 && (k_u % 8u) == 0 &&
         (m_u / 64u) * (n_u / 48u) >= 25u && a_align16;
      /* High-K K_step=32 tiles for narrow N (16/32) or non-32-aligned M%16==0:
       * halve K-loop barriers, double/quadruple workgroup parallelism on wide
       * GPUs, and enable unguarded specialization when M%16==0 (e.g. M=400).
       * Keep 32x32 gated to small-grid M (< 48 WGs at 64x32) so wide grids do
       * not double B reads. */
      const bool can_16x32 =
         !can_64x64 && !can_64x48 &&
         (m_u % 32u) != 0 && (m_u % 16u) == 0 && (n_u % 32u) == 0 &&
         k_u >= 32u && (k_u % 8u) == 0 && a_align16;
      const bool can_16x16 =
         !can_64x64 && !can_64x48 && !can_16x32 &&
         (m_u % 32u) != 0 && (m_u % 16u) == 0 && (n_u % 16u) == 0 &&
         k_u >= 32u && (k_u % 8u) == 0 && a_align16;
      const bool can_32x32 =
         !can_64x64 && !can_64x48 && !can_16x32 && !can_16x16 &&
         (m_u % 32u) == 0 && (n_u % 32u) == 0 && (n_u % 64u) != 0 &&
         k_u >= 64u && (k_u % 32u) == 0 &&
         (m_u / 64u) * (n_u / 32u) < 48u && a_align16;
      const bool can_32x16 =
         !can_64x64 && !can_64x48 && !can_16x32 && !can_16x16 && !can_32x32 &&
         (m_u % 32u) == 0 && (n_u % 16u) == 0 && (n_u % 32u) != 0 &&
         k_u >= 32u && (k_u % 16u) == 0 &&
         ((k_u % 32u) == 0 || (m_u / 32u) <= 200u) && a_align16;
      if (can_64x64) {
         const enum hk_compute_pair_alt_kind k64 =
            gemm->is_guarded ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64_UNGUARDED
                             : HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64;
         struct hk_shader *s64 = hk_compute_pair_get_or_compile_fused(
            dev, api->compute_pair_producer, NULL, k64,
            HK_COMPUTE_PAIR_DUAL_NONE, NULL, false, NULL, NULL);
         if (s64 &&
             hk_g17p_compute_shader_supported(dev->dev.chip, s64,
                                              s64->only_linked, 1u << 16)) {
            alt = s64;
            tile_n = 64u;
            chosen_kind = k64;
         }
      } else if (can_64x48) {
         struct hk_shader *s48 = hk_compute_pair_get_or_compile_fused(
            dev, api->compute_pair_producer, NULL,
            HK_COMPUTE_PAIR_ALT_GEMM_CMAT48_UNGUARDED,
            HK_COMPUTE_PAIR_DUAL_NONE, NULL, false, NULL, NULL);
         if (s48 &&
             hk_g17p_compute_shader_supported(dev->dev.chip, s48,
                                              s48->only_linked, 1u << 16)) {
            alt = s48;
            tile_n = 48u;
            chosen_kind = HK_COMPUTE_PAIR_ALT_GEMM_CMAT48_UNGUARDED;
         }
      } else if (can_16x32 || can_16x16 || can_32x32 || can_32x16) {
         const enum hk_compute_pair_alt_kind k_small =
            can_16x32 ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT16X32_UNGUARDED :
            can_16x16 ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT16X16_UNGUARDED :
            can_32x32 ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT32X32_UNGUARDED :
                        HK_COMPUTE_PAIR_ALT_GEMM_CMAT32X16_UNGUARDED;
         struct hk_shader *s_sm = hk_compute_pair_get_or_compile_fused(
            dev, api->compute_pair_producer, NULL, k_small,
            HK_COMPUTE_PAIR_DUAL_NONE, NULL, false, NULL, NULL);
         if (s_sm &&
             hk_g17p_compute_shader_supported(dev->dev.chip, s_sm,
                                              s_sm->only_linked, 1u << 16)) {
            alt = s_sm;
            tile_n = hk_compute_pair_alt_gemm_tile_n(k_small);
            chosen_kind = k_small;
         }
      } else if ((n_u % 24u) == 0 && (n_u % 32u) != 0 && (n_u % 64u) != 0 &&
                 hk_gemm_cmat_can_specialize_unguarded(gemm->is_guarded, m_u, n_u, k_u,
                                                       24u, gemm->reads[0].offset_B,
                                                       a_b.address, a_b.offset)) {
         struct hk_shader *ug24 = hk_compute_pair_get_or_compile_fused(
            dev, api->compute_pair_producer, NULL,
            HK_COMPUTE_PAIR_ALT_GEMM_CMAT24_UNGUARDED,
            HK_COMPUTE_PAIR_DUAL_NONE, NULL, false, NULL, NULL);
         if (ug24 &&
             hk_g17p_compute_shader_supported(dev->dev.chip, ug24,
                                              ug24->only_linked, 1u << 16)) {
            alt = ug24;
            tile_n = 24u;
            chosen_kind = HK_COMPUTE_PAIR_ALT_GEMM_CMAT24_UNGUARDED;
         }
      } else if (hk_gemm_cmat_can_specialize_unguarded(
                    gemm->is_guarded, m_u, n_u, k_u, tile_n,
                    gemm->reads[0].offset_B, a_b.address, a_b.offset)) {
         enum hk_compute_pair_alt_kind ug_kind =
            tile_n == 16 ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT16_UNGUARDED :
            tile_n == 32 ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT32_UNGUARDED :
                           HK_COMPUTE_PAIR_ALT_GEMM_CMAT64_UNGUARDED;
         struct hk_shader *ug = hk_compute_pair_get_or_compile_fused(
            dev, api->compute_pair_producer, NULL, ug_kind,
            HK_COMPUTE_PAIR_DUAL_NONE, NULL, false, NULL, NULL);
         if (ug) {
            alt = ug;
            chosen_kind = ug_kind;
         }
      }
   }

   const uint32_t tile_m = hk_compute_pair_alt_gemm_tile_m(chosen_kind);
   const uint64_t gx = DIV_ROUND_UP(n_u, tile_n);
   const uint64_t gy = DIV_ROUND_UP(m_u, tile_m);
   if (gx > 65535 || gy > 65535)
      return normal;

   grid->count[0] = (uint32_t)gx;
   grid->count[1] = (uint32_t)gy;
   grid->count[2] = 1;
   if (alt_kind_out)
      *alt_kind_out = chosen_kind;
   if (pack_loop_params) {
      const uint32_t k_step = hk_compute_pair_alt_gemm_k_step(chosen_kind);
      pack_loop_params[HK_PACK_LOOP_PARAM_GEMM_K_ITERS] =
         DIV_ROUND_UP(k_u, k_step);
   }
   if (loop_bound_out && pack_loop_params)
      hk_pack_eval_effective_instrs(&alt->pack_loop_bound,
                                    alt->b.info.stats.instrs, pack_loop_params,
                                    loop_bound_out);
   return alt;
}

static void
dispatch_grid_shader(struct hk_cmd_buffer *cmd, struct hk_shader *s,
                     struct agx_grid grid, bool in_threads)
{
   if (agx_is_shader_empty(&s->b))
      return;

   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (hk_conditional_render_is_active(&cmd->state.conditional, cmd->in_meta,
                                        cmd->conditional_meta)) {
      const uint64_t patched = hk_apply_conditional_rendering(
         cmd, desc->root.cs.group_count_addr, 3, AGX_BARRIER_ALL);
      if (!patched)
         return;

      /* All API dispatch forms already publish their three group counts in
       * the root. Point both the shader-visible count and the command-stream
       * indirect launch at the predicated copy, so skipped commands also
       * contribute zero pipeline-stat invocations. */
      desc->root.cs.group_count_addr = patched;
      hk_root_mark_dirty(desc);
      grid = agx_grid_indirect(patched);
   }

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, true /* compute */);
   if (!cs)
      return;

   if (cs->compute_pair)
      hk_cs_invalidate_compute_pair(cs);

   struct hk_api_shader *api = cmd->state.cs.shader;
   const uint32_t orig_groups[3] = {grid.count[0], grid.count[1], grid.count[2]};

   struct agx_workgroup local_size =
      agx_workgroup(s->b.info.workgroup_size[0], s->b.info.workgroup_size[1],
                    s->b.info.workgroup_size[2]);

   bool invocation_stat = false;
   /* Internal compute work must not contribute to the application's active
    * pipeline query. This also honours hk_pipeline_stat_addr's !in_meta
    * contract in debug builds; graphics meta pauses its own counters. */
   if (!cmd->in_meta) {
      uint64_t stat = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT);

      if (hk_stat_enabled(stat)) {
         invocation_stat = true;
         perf_debug(cmd, "CS invocation statistic");
         uint64_t grid = cmd->state.cs.descriptors.root.cs.group_count_addr;

         libagx_increment_cs_invocations(cmd, agx_1d(1), AGX_BARRIER_ALL, grid,
                                         stat, agx_workgroup_threads(local_size));
      }
   }

   struct hk_shader *selected = s;
   struct hk_copy4_dispatch_cert copy4_cert = {.role = HK_COMPUTE_PAIR_NONE};
   enum hk_compute_pair_alt_kind selected_alt_kind = HK_COMPUTE_PAIR_ALT_NONE;
   uint64_t pack_loop_params[HK_PACK_LOOP_PARAM_COUNT] = {0};
   uint64_t loop_bound = 0;
   bool has_im2col_push = false;
   uint32_t im2col_push_off = 0;
   struct hk_im2col_push im2col_push = {0};
   if (api && api->identity_copy4 && !api->compute_pair && !in_threads && !invocation_stat) {
      selected = hk_select_identity_copy4(cmd, api, s, &grid, &copy4_cert, &loop_bound,
                                          &has_im2col_push, &im2col_push_off, &im2col_push);
      /* Record-time no-op self-copy elision: runs after
       * hk_cs_invalidate_compute_pair() and is skipped whenever pipeline stats
       * (invocation_stat), conditional rendering (indirect grid), base_group,
       * or secondary CBs are active. Any preceding/following barriers remain in
       * the stream; cs->stats.calls++ and frame-anatomy accounting are skipped
       * here by design since no CDM dispatch is emitted. */
      if (copy4_cert.role == HK_COMPUTE_PAIR_SELF_COPY && cs->stats.cmds > 0)
         return;
      local_size = agx_workgroup(selected->b.info.workgroup_size[0],
                                 selected->b.info.workgroup_size[1],
                                 selected->b.info.workgroup_size[2]);
   } else if (api && api->tensor_op_primary && !api->identity_copy4 &&
              !api->compute_pair && !in_threads && !invocation_stat) {
      selected = hk_select_tensor_op(cmd, api, s, &grid);
      local_size = agx_workgroup(selected->b.info.workgroup_size[0],
                                 selected->b.info.workgroup_size[1],
                                 selected->b.info.workgroup_size[2]);
   } else if (api && api->depthwise_vec4 && api->compute_pair &&
              api->compute_pair->role == HK_COMPUTE_PAIR_DEPTHWISE &&
              !in_threads && !invocation_stat) {
      selected = hk_select_depthwise_vec4(cmd, api, s, &grid, pack_loop_params, &loop_bound);
      if (selected == api->depthwise_vec4)
         selected_alt_kind = HK_COMPUTE_PAIR_ALT_DEPTHWISE_VEC4;
      local_size = agx_workgroup(selected->b.info.workgroup_size[0],
                                 selected->b.info.workgroup_size[1],
                                 selected->b.info.workgroup_size[2]);
   } else if (api && (api->gemm_cmat16 || api->gemm_cmat32 || api->gemm_cmat64) &&
              api->compute_pair &&
              api->compute_pair->role == HK_COMPUTE_PAIR_GEMM &&
              !in_threads && !invocation_stat) {
      selected = hk_select_gemm_cmat(cmd, api, s, &grid, &selected_alt_kind,
                                     pack_loop_params, &loop_bound);
      local_size = agx_workgroup(selected->b.info.workgroup_size[0],
                                 selected->b.info.workgroup_size[1],
                                 selected->b.info.workgroup_size[2]);
   }

   if (!agx_is_indirect(grid) && !in_threads) {
      grid.count[0] *= local_size.x;
      grid.count[1] *= local_size.y;
      grid.count[2] *= local_size.z;
   }

   /* The grid in threads is the same for any workgroup shape: a direct
    * launch of a shader whose builtins do not depend on the hardware
    * workgroup uses its smaller hardware workgroup
    * (agx_nir_lower_workgroup_split). */
   if (!agx_is_indirect(grid) && !in_threads && selected->b.info.launch_size[0]) {
      local_size = agx_workgroup(selected->b.info.launch_size[0],
                                 selected->b.info.launch_size[1],
                                 selected->b.info.launch_size[2]);
   } else if (!agx_is_indirect(grid) && !in_threads &&
              agx_chip_has_cap(hk_cmd_buffer_device(cmd)->dev.chip,
                               AGX_CAP_COALESCE_PURE_GLOBAL_WORKGROUPS) &&
              (selected->info.agx3_root_reads & HK_ROOT_PROVEN_WG_COALESCE_SAFE) &&
              !selected->b.info.uses_ray_traversal && !selected->b.info.local_size &&
              !selected->b.info.scratch_size &&
              !selected->b.info.agx3_spectpr_scratch_size &&
              !selected->b.info.uses_matrix16 &&
              !desc->root.cs.trace_rays_addr &&
              hk_nir_dispatch_zero_base(desc->root.cs.base_group)) {
      struct hk_device *dev = hk_cmd_buffer_device(cmd);
      const uint32_t eff_gprs = MAX2((uint32_t)selected->b.info.nr_gprs,
                                     (uint32_t)selected->b.info.nr_preamble_gprs);
      uint32_t max_wg =
         dev->vk.physical->properties.maxComputeWorkGroupInvocations;
      if (selected->b.info.stats.threads)
         max_wg = MIN2(max_wg, selected->b.info.stats.threads);
      if (eff_gprs > 256u)
         max_wg = 0;
      else if (eff_gprs)
         max_wg = MIN2(max_wg,
                       agx_occupancy_for_register_count(eff_gprs).max_threads);
      /* Short, low-register, loop-free pure-global shaders still hit AGX3's
       * per-WG CDM launch floor at wg=256 on massive grids; coalesce them up
       * to 512 threads (>= 1024 WGs). Account for native constant-program
       * register usage when preambles are active. General pure-global shaders
       * cap at 256 (>= 256 WGs, and always >= 4 * num_cores so wide GPUs
       * never lose core saturation); never 1024, which regresses on AGX3. */
      const uint32_t target_wg =
         (!selected->b.info.has_loops && selected->b.info.stats.instrs <= 64u &&
          eff_gprs <= 24u)
            ? 512u
            : 256u;
      const uint64_t min_wgs =
         MAX2(256ull, 4ull * agx_get_num_cores(&dev->dev));
      uint16_t reshaped[3];
      if (hk_coalesce_workgroup_size_ex(
             selected->b.info.workgroup_size, grid.count,
             dev->vk.physical->properties.maxComputeWorkGroupSize, max_wg,
             target_wg, min_wgs, reshaped) ||
          hk_split_workgroup_size_ex(
             selected->b.info.workgroup_size, grid.count,
             2ull * agx_get_num_cores(&dev->dev), reshaped)) {
         local_size = agx_workgroup(reshaped[0], reshaped[1], reshaped[2]);
      }
   }

   if (cs->g18_dispatch_sealed) {
      cs = hk_cmd_buffer_get_cs(cmd, true);
      if (!cs)
         return;
   }
   uint8_t saved_im2col_push[sizeof(struct hk_im2col_push)];
   if (has_im2col_push) {
      memcpy(saved_im2col_push, desc->root.push + im2col_push_off, sizeof(saved_im2col_push));
      memcpy(desc->root.push + im2col_push_off, &im2col_push, sizeof(im2col_push));
      hk_root_mark_dirty(desc);
   }
   hk_dispatch_with_local_size(cmd, cs, selected, grid, local_size);
   if (has_im2col_push) {
      memcpy(desc->root.push + im2col_push_off, saved_im2col_push, sizeof(saved_im2col_push));
      hk_root_mark_dirty(desc);
   }
   if (!in_threads && !invocation_stat && api && api->compute_pair &&
       (selected == s || selected_alt_kind != HK_COMPUTE_PAIR_ALT_NONE))
      hk_record_compute_pair(cmd, cs, api, selected, selected_alt_kind,
                             pack_loop_params, orig_groups, grid);
   else if (!in_threads && !invocation_stat &&
            copy4_cert.role != HK_COMPUTE_PAIR_NONE)
      hk_record_copy4_pair(cmd, cs, api, selected, &copy4_cert, orig_groups,
                           grid);
   cs->uses_global_atomics |= selected->b.info.uses_global_atomics;
   if (agx_is_indirect(grid)) {
      cs->pack_loops |= selected->b.info.has_loops;
      cs->sib_indirect = true;
   } else {
      const uint64_t threads =
         (uint64_t)grid.count[0] * grid.count[1] * grid.count[2];
      cs->sib_threads = threads > UINT64_MAX - cs->sib_threads
                           ? UINT64_MAX : cs->sib_threads + threads;

      uint64_t work = 0;
      cs->pack_loops |= hk_pack_account_dispatch_work_eff_limit(
         threads, selected->b.info.stats.instrs, selected->b.info.has_loops,
         loop_bound, hk_pack_max_static_work(hk_cmd_buffer_device(cmd)), &work);
      cs->pack_threads = threads > UINT64_MAX - cs->pack_threads
                            ? UINT64_MAX : cs->pack_threads + threads;
      cs->pack_static_work = work > UINT64_MAX - cs->pack_static_work
                                ? UINT64_MAX : cs->pack_static_work + work;
      cs->pack_dispatches++;
   }
   if (unlikely(hk_cmd_buffer_device(cmd)->frame_anatomy)) {
      hk_frame_anatomy_note_fs(cs, selected);
      const uint64_t threads = agx_is_indirect(grid) ? 0 :
         (uint64_t)grid.count[0] * grid.count[1] * grid.count[2];
      const uint64_t wgs =
         agx_is_indirect(grid) || !local_size.x || !local_size.y || !local_size.z
            ? 0
            : (uint64_t)DIV_ROUND_UP(grid.count[0], local_size.x) *
                 DIV_ROUND_UP(grid.count[1], local_size.y) *
                 DIV_ROUND_UP(grid.count[2], local_size.z);
      hk_frame_anatomy_note_grid(cs, selected, threads, wgs,
                                 agx_is_indirect(grid));
   }
   cs->stats.calls++;
   if (cmd->in_meta)
      cs->render_pack_compute_internal = true;
}

static void
dispatch_grid(struct hk_cmd_buffer *cmd, struct agx_grid grid, bool in_threads)
{
   hk_cmd_resolve_compute_border(cmd);
   struct hk_shader *s = hk_only_variant(cmd->state.cs.shader);
   dispatch_grid_shader(cmd, s, grid, in_threads);
}

static void
dispatch(struct hk_cmd_buffer *cmd, struct agx_grid grid)
{
   dispatch_grid(cmd, grid, false);
}

/*
 * Launch an exact number of invocations, allowing a partial trailing
 * workgroup.  The common acceleration-structure build shaders have no bounds
 * guard of their own and rely on this; the AGX command stream takes a thread
 * count natively, so "unaligned" costs nothing here beyond skipping the
 * workgroup-to-thread multiply that the API entry points do.
 *
 * The group_count the root still advertises is rounded up, because that is
 * what gl_NumWorkGroups and the compute-invocation statistic mean.
 */
void
hk_cmd_dispatch_unaligned(VkCommandBuffer commandBuffer, uint32_t invocations_x,
                          uint32_t invocations_y, uint32_t invocations_z)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;

   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   hk_cmd_resolve_compute_border(cmd);
   struct hk_shader *s = hk_only_variant(cmd->state.cs.shader);
   if (!s || agx_is_shader_empty(&s->b))
      return;

   desc->root.cs.base_group[0] = 0;
   desc->root.cs.base_group[1] = 0;
   desc->root.cs.base_group[2] = 0;

   VkDispatchIndirectCommand group_count = {
      .x = DIV_ROUND_UP(invocations_x, s->b.info.workgroup_size[0]),
      .y = DIV_ROUND_UP(invocations_y, s->b.info.workgroup_size[1]),
      .z = DIV_ROUND_UP(invocations_z, s->b.info.workgroup_size[2]),
   };
   desc->root.cs.group_count_addr = hk_cmd_upload_group_count(cmd, group_count);
   desc->root.cs.grid_helper_sparse_map_addr = 0;

   dispatch_grid(cmd, agx_3d(invocations_x, invocations_y, invocations_z),
                 true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t baseGroupX,
                   uint32_t baseGroupY, uint32_t baseGroupZ,
                   uint32_t groupCountX, uint32_t groupCountY,
                   uint32_t groupCountZ)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   if (!cmd->in_meta && cmd->state.cs.shader &&
       (cmd->state.cs.shader->byte_stencil ||
        cmd->state.cs.shader->stencil_2d ||
        cmd->state.cs.shader->compute_opt) &&
       !hk_conditional_render_is_active(&cmd->state.conditional, cmd->in_meta,
                                        cmd->conditional_meta) &&
       !hk_stat_enabled(hk_pipeline_stat_addr(
          cmd, VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT))) {
      const struct hk_device *dev = hk_cmd_buffer_device(cmd);
      struct hk_shader *tiled_shader = NULL;
      uint32_t tile_gx = 0, tile_gy = 0;
      if (cmd->state.cs.shader->byte_stencil &&
          hk_byte_stencil_validate_dispatch(
             dev, desc, &cmd->state.cs.shader->byte_stencil->plan, baseGroupX,
             baseGroupY, baseGroupZ, groupCountX, groupCountY, groupCountZ,
             &tile_gx, &tile_gy)) {
         tiled_shader = &cmd->state.cs.shader->byte_stencil->shader;
      } else if (cmd->state.cs.shader->stencil_2d &&
                 hk_2d_stencil_validate_dispatch(
                    dev, desc, &cmd->state.cs.shader->stencil_2d->plan,
                    baseGroupX, baseGroupY, baseGroupZ, groupCountX,
                    groupCountY, groupCountZ, &tile_gx, &tile_gy)) {
         tiled_shader = &cmd->state.cs.shader->stencil_2d->shader;
      } else if (cmd->state.cs.shader->compute_opt &&
                 hk_compute_opt_validate_dispatch(
                    dev, desc, &cmd->state.cs.shader->compute_opt->plan,
                    baseGroupX, baseGroupY, baseGroupZ, groupCountX,
                    groupCountY, groupCountZ, &tile_gx, &tile_gy)) {
         tiled_shader = &cmd->state.cs.shader->compute_opt->shader;
      }
      if (tiled_shader) {
         desc->root.cs.base_group[0] = 0;
         desc->root.cs.base_group[1] = 0;
         desc->root.cs.base_group[2] = 0;

         VkDispatchIndirectCommand tile_gc = {
            .x = tile_gx,
            .y = tile_gy,
            .z = 1,
         };
         desc->root.cs.group_count_addr =
            hk_pool_upload(cmd, &tile_gc, sizeof(tile_gc), 8);
         desc->root.cs.grid_helper_sparse_map_addr = 0;

         dispatch_grid_shader(cmd, tiled_shader, agx_3d(tile_gx, tile_gy, 1),
                              false);
         return;
      }
   }

   desc->root.cs.base_group[0] = baseGroupX;
   desc->root.cs.base_group[1] = baseGroupY;
   desc->root.cs.base_group[2] = baseGroupZ;

   /* We don't want to key the shader to whether we're indirectly dispatching,
    * so treat everything as indirect.
    */
   VkDispatchIndirectCommand group_count = {
      .x = groupCountX,
      .y = groupCountY,
      .z = groupCountZ,
   };

   desc->root.cs.group_count_addr = hk_cmd_upload_group_count(cmd, group_count);
   desc->root.cs.grid_helper_sparse_map_addr = 0;

   dispatch(cmd, agx_3d(groupCountX, groupCountY, groupCountZ));
}

/*
 * Dispatch a ray-tracing megakernel.
 *
 * VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR has its own descriptor state, but the
 * megakernel executes on the ordinary compute path, so the ray-tracing state is
 * installed into the compute slots for the duration of one dispatch and put
 * back afterwards.  Anything the dispatch lazily allocates inside the
 * descriptor state (a pushed descriptor snapshot, the constant-data address)
 * is carried back rather than dropped, or it would leak.
 */
void
hk_cmd_dispatch_rays(struct hk_cmd_buffer *cmd, struct hk_api_shader *shader,
                     uint64_t record_addr, uint64_t group_count_addr,
                     struct agx_grid grid)
{
   struct hk_descriptor_state *rt = &cmd->rt.descriptors;

   if (rt->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, rt);

   const struct hk_compute_state saved = cmd->state.cs;

   cmd->state.cs.shader = shader;
   cmd->state.cs.descriptors = *rt;

   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   desc->root.cs.base_group[0] = 0;
   desc->root.cs.base_group[1] = 0;
   desc->root.cs.base_group[2] = 0;
   desc->root.cs.group_count_addr = group_count_addr;
   desc->root.cs.grid_helper_sparse_map_addr = 0;
   desc->root.cs.trace_rays_addr = record_addr;
   hk_root_mark_dirty(desc);

   /* Conditional rendering predicates draws, dispatches and attachment
    * clears only; vkCmdTraceRays* must execute regardless of it. */
   const bool conditional_active = cmd->state.conditional.active;
   cmd->state.conditional.active = false;

   cmd->state.cs.cdm_barrier_full_needed = true;
   cmd->state.cs.cdm_stream_has_images = true;
   dispatch(cmd, grid);

   cmd->state.conditional.active = conditional_active;

   /* Carry the state back before restoring, so a pushed descriptor set the
    * flush above allocated stays owned by the ray-tracing state. */
   *rt = cmd->state.cs.descriptors;
   rt->root.cs.trace_rays_addr = 0;

   const bool cdm_barrier_pending = cmd->state.cs.cdm_barrier_pending;
   const bool cdm_barrier_full_needed = cmd->state.cs.cdm_barrier_full_needed;
   const bool cdm_stream_has_images = cmd->state.cs.cdm_stream_has_images;
   cmd->state.cs = saved;
   cmd->state.cs.cdm_barrier_pending = cdm_barrier_pending;
   cmd->state.cs.cdm_barrier_full_needed = cdm_barrier_full_needed;
   cmd->state.cs.cdm_stream_has_images = cdm_stream_has_images;
   hk_root_mark_dirty(&cmd->state.cs.descriptors);
}

void
hk_cmd_dispatch_indirect_addr(VkCommandBuffer command_buffer,
                             VkDeviceAddress address)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, command_buffer);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   assert(address && !(address & 3));
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);
   desc->root.cs.base_group[0] = 0;
   desc->root.cs.base_group[1] = 0;
   desc->root.cs.base_group[2] = 0;
   desc->root.cs.group_count_addr = address;
   desc->root.cs.grid_helper_sparse_map_addr = 0;
   dispatch(cmd, agx_grid_indirect(address));
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                       VkDeviceSize offset)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   struct hk_descriptor_state *desc = &cmd->state.cs.descriptors;
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   desc->root.cs.base_group[0] = 0;
   desc->root.cs.base_group[1] = 0;
   desc->root.cs.base_group[2] = 0;

   /* Arguments are normally read through the sparse read-only shadow, so that
    * a sparse resident buffer honours sparseResidencyNonResidentStrict: an
    * unbound page reads as zero there, and a zero grid launches nothing.
    *
    * On G17P that shadow cannot be read here.  A read through the shadow does
    * not reliably return data written through the primary mapping in the same
    * submission.  Measured by forcing hk_g17p_sparse_buffer_copy() to read
    * resident pages through the shadow as well, over five runs of
    * dEQP-VK.sparse_resources.buffer.ssbo.sparse_residency: buffer_size_2_10
    * and _2_16 failed 5/5, and _2_20 failed 1/5.  The same build passes 3/3
    * unforced, and an added strong CDM cache-flush record does not cure it.
    * The mechanism is not established; what is established is that the value
    * read back is not dependable.
    *
    * Indirect dispatch is where this was measured, not the only place it can
    * bite.  The storage-buffer lowering and the sparse buffer copy both read
    * resident pages through the primary mapping and consult the residency
    * bitmap, so neither is exposed.  hk_CmdDrawIndirect(),
    * hk_CmdDrawIndexedIndirect(), hk_CmdDrawIndirectCount() and conditional
    * rendering do still hand the shadow address to the hardware, which reads
    * the record from it, so the same unreliable read is latent on those paths.
    * They have not been probed: a draw whose indirect arguments come from an
    * unbound page is the same unbounded-work hazard this change exists to
    * remove, and is not something to demonstrate on a shared machine.
    *
    * So do what they do: name the primary mapping and hand the grid helper the
    * bitmap, which substitutes a zero grid for a non-resident page and keeps
    * the strictness promise.  A sparse buffer without residency has no bitmap;
    * every page an application may legally read from it is bound.
    */
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* A residency bitmap is only useful with the helper that reads it. That
    * helper is built, on chips with native indirect compute, for devices that
    * enabled sparseResidencyBuffer, which is the only way to create a buffer
    * carrying a bitmap. Other chips have no helper: keep reading the
    * arguments through the shadow there. */
   const bool have_sparse_helper = dev->g17p_grid_helper_sparse != NULL;
   const bool use_primary =
      agx_chip_native_agx3(dev->dev.chip) && buffer->va &&
      (!buffer->sparse_map || have_sparse_helper);

   uint64_t dispatch_addr = hk_buffer_address(buffer, offset, !use_primary);
   assert(dispatch_addr != 0);

   desc->root.cs.group_count_addr = dispatch_addr;
   desc->root.cs.grid_helper_sparse_map_addr =
      (use_primary && buffer->sparse_map) ? hk_buffer_sparse_map_addr(buffer)
                                          : 0;
   dispatch(cmd, agx_grid_indirect(dispatch_addr));
}

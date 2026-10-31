/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "libagx/query.h"
#include "vulkan/vulkan_core.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_pack.h"
#include "agx_scratch.h"
#include "agx_tilebuffer.h"
#include "hk_buffer.h"
#include "hk_acceleration_structure.h"
#include "hk_cmd_buffer.h"
#include "hk_frame_anatomy.h"
#include "hk_g17p_patch.h"
#include "hk_descriptor_set.h"
#include "hk_device.h"
#include "hk_gen1_rt_indirect_contract.h"
#include "hk_entrypoints.h"
#include "hk_agx3_ioto_program_core.h"
#include "asahi/compiler/agx3_compute_program.h"
#include "asahi/compiler/agx3_graphics_program.h"
#include "hk_physical_device.h"
#include "hk_shader.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx3_g17p_grid_helper_core.h"
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

/* Diagnostic controls, read once: these are consulted per dispatch, and an
 * uncached getenv() scans the whole environment. */
DEBUG_GET_ONCE_BOOL_OPTION(m5_launch_debug, "ASAHI_M5_LAUNCH_DEBUG", false)
DEBUG_GET_ONCE_BOOL_OPTION(m5_verify_compute_loader,
                           "AGX_M5_VERIFY_COMPUTE_LOADER", false)
DEBUG_GET_ONCE_BOOL_OPTION(m5_rq_candidate_probe, "ASAHI_M5_RQ_CANDIDATE_PROBE",
                           false)
DEBUG_GET_ONCE_BOOL_OPTION(m5_rq_candidate_api, "ASAHI_M5_RQ_CANDIDATE_API",
                           false)

static inline bool
hk_m5_rq_candidate_requested(void)
{
   return debug_get_option_m5_rq_candidate_probe() ||
          debug_get_option_m5_rq_candidate_api();
}

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
   struct hk_m5_compute_loader *l = NULL;
   if (!agx3_g17g_compute_program_template_generate(dev->dev.chip, d, t))
      goto out;
   l = calloc(1, sizeof(*l) + t->program.length);
   if (!l)
      goto out;
   if (!hk_g17p_patch_plan_init(&l->plan, d->uniform_ranges,
                                d->uniform_range_count, t->relocations,
                                t->relocation_count, t->program.length, false)) {
      free(l);
      l = NULL;
      goto out;
   }
   l->desc = *d;
   l->desc.uniform_ranges = l->ranges;
   memcpy(l->ranges, d->uniform_ranges,
          d->uniform_range_count * sizeof(l->ranges[0]));
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
 * scope: always on M5, on request on G17P. */
static bool
hk_cdm_flush_has_memory_scope(const struct hk_device *dev, bool memory)
{
   return agx_chip_g17_m5(dev->dev.chip) ||
          (memory && dev->dev.chip == AGX_CHIP_G17P);
}

bool
hk_cdm_visibility_flush_needed(const struct hk_device *dev,
                               const struct hk_cs *cs, bool memory)
{
   if (cs->stats.cmds == 0)
      return false;
   if (cs->current != cs->strong_flush_end)
      return true;
   /* M5's strong flushes already have memory scope. */
   return memory && dev->dev.chip == AGX_CHIP_G17P &&
          cs->current != cs->memory_flush_end;
}

static void
hk_cdm_flush(struct hk_device *dev, struct hk_cs *cs, bool memory)
{
   assert(cs->type == HK_CS_CDM);
   assert(cs->current + AGX_CDM_BARRIER_LENGTH < cs->end &&
          "caller must ensure space");

   switch (agx_chip_command_stream_generation(dev->dev.chip)) {
   case AGX_CS_GENERATION_AGX2:
      cs->current = agx_cdm_barrier(cs->current, dev->dev.chip);
      break;

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      const bool memory_scope = hk_cdm_flush_has_memory_scope(dev, memory);

      /* A strong flush already ends the stream: every launch has one in its
       * tail, and a second barrier with no launch between them is not a
       * shape Apple's CDM emitter produces. On M5 that flush already has
       * memory scope; on G17P a request for it strengthens the record in
       * place below. */
      const bool upgrade_strong = cs->current == cs->strong_flush_end;
      if (upgrade_strong &&
          (!memory_scope || agx_chip_g17_m5(dev->dev.chip) ||
           cs->current == cs->memory_flush_end))
         return;

      /* Apple's CommandEncodingNext emitter uses flags=5 for its strong
       * post-dispatch cache flush on every grounded G15--G17 identity.  Keep
       * Honeykrisp's deliberately conservative flush-after-every-launch
       * policy, but encode it in the target command-stream ISA. */
      const struct agx3_cdm_barrier_info info = {
         .chip = dev->dev.chip,
         .flags = 0x5,
         .wide_visibility = agx_chip_cdm_wide_visibility(dev->dev.chip),
         .flush_memory = memory_scope,
      };
      uint8_t packed[AGX3_CDM_BARRIER_SIZE];
      if (!agx3_cdm_barrier_pack(&info, packed)) {
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

void
hk_cdm_cache_flush(struct hk_device *dev, struct hk_cs *cs)
{
   hk_cdm_flush(dev, cs, false);
}

void
hk_cdm_visibility_flush(struct hk_device *dev, struct hk_cs *cs, bool memory)
{
   hk_cdm_flush(dev, cs, memory);
}

void
hk_dispatch_with_usc_launch(struct hk_device *dev, struct hk_cs *cs,
                            struct agx_cdm_launch_word_0_packed launch,
                            uint32_t usc, struct agx_grid grid,
                            struct agx_workgroup wg)
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
       * never feed an AGX2 launch to CommandEncodingNext. */
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

   hk_ensure_cs_has_space(cs->cmd, cs, 0x2000 /* TODO */);
   cs->stats.cmds++;
   cs->current =
      agx_cdm_launch(cs->current, dev->dev.chip, grid, wg, launch, usc);

   hk_cdm_cache_flush(dev, cs);
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

   hk_dispatch_with_usc_launch(dev, cs, launch, usc, grid, local_size);
}

static bool
hk_g17p_compute_shader_supported(enum agx_chip chip, struct hk_shader *s,
                                 struct hk_linked_shader *linked, unsigned max_spill_B)
{
   /* poly_nir_lower_gs preserves the API geometry stage on the main kernel,
    * even though Honeykrisp has always launched that binary through CDM. */
   return s && linked &&
          (s->b.info.stage == MESA_SHADER_COMPUTE ||
           s->info.stage == MESA_SHADER_GEOMETRY ||
           s->info.stage == MESA_SHADER_TESS_CTRL) &&
          linked->b.bo && linked->b.bo->va && linked->b.size != 0 &&
          linked->b.size <= UINT32_MAX && linked->b.nr_gprs != 0 &&
          ((!s->b.info.has_preamble && s->b.info.preamble_scratch_size == 0) ||
           /* Only an M5 compute compile forms a constant program
            * (hk_shader.c), and it carries its TPS frame in the launch. */
           (agx_chip_g17_m5(chip) && s->info.stage == MESA_SHADER_COMPUTE &&
            s->b.info.agx3_preamble_staging_known &&
            s->b.info.preamble_scratch_size <= max_spill_B &&
            s->b.info.preamble_scratch_size >=
               s->b.info.agx3_preamble_inputs.storage_bytes)) &&
          s->b.info.scratch_size <= max_spill_B &&
          linked->b.scratch_size <= max_spill_B &&
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
                                   &ranges[count]))
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
   if (!hk_agx3_g17p_root_range(root_uniform, s->info.set_count, root_ptr, &ranges[0]))
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

   *desc = (struct agx3_g17p_compute_program_desc){
      .shader_va = linked->b.bo->va->addr + linked->b.entry_offset,
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
      .shader_binary_size = linked->b.size,
      .scratch_size =
         MAX2(linked->b.scratch_size, linked->b.agx3_spectpr_scratch_size),
      .nr_gprs = linked->b.nr_gprs,
      .compiler_d_valid = s->b.info.agx3_staging_known,
      .compiler_d = (uint32_t)s->b.info.agx3_staging_halfs * 2,
      .threadgroup_memory_size_B =
         s->b.info.local_size ? MAX2(256u, ALIGN_POT(s->b.info.local_size, 64u))
                              : 0,
      .launch_sources = launch_sources,
      .ldshdr_f1 = kind,
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
      const uint64_t packed = hk_gen1_ray_query_scs_layout(dev);
      struct agx_scs_layout_gen2 scs;
      if (!agx_scs_layout_gen1_init(packed, &scs)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return false;
      }
      const uint64_t state = hk_gen1_query_state(cmd, &scs);
      struct agx_ptr u=hk_pool_alloc(cmd,512,16);
      if(!state || !u.cpu)return false;
      memset(u.cpu,0,512);memcpy(u.cpu,dev->gen1_rq_uniforms,320);
      for(unsigned i=0;i<74;i+=2) {
         uint64_t value;memcpy(&value,(uint8_t*)u.cpu+4*i,8);
         if(value>=UINT64_C(0x20d0000000) && value<UINT64_C(0x20d0060000)) {
            value+=state-UINT64_C(0x20d0000000);
            memcpy((uint8_t*)u.cpu+4*i,&value,8);
         }
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
      uint64_t table=state+0x50000,ctx=u.gpu+352;
      if (dev->gen1_rq_instances) {
         table=hk_gen1_upload_as_table(cmd);
         if (!table) return false;
         const uint32_t constants[]={0x3f7ffffe,0x3f800001,0x6000,0x8000,
            0xfe00001,0x1000,0x100,0x10002,0x1ffff,0x20000,0x400000,
            0x1f000000,0x34a00000,0x40000000,0xfffeff00};
         memcpy((uint8_t *)u.cpu+118*2,constants,sizeof(constants));
         uint32_t *uw=u.cpu;
         uw[106/2]=88;
         uw[48/2]=scs.end[6];uw[50/2]=scs.end[11];
         uw[52/2]=packed;uw[54/2]=packed>>32;
         memcpy((uint8_t *)u.cpu+32*2,&table,8);
         context[2]=packed;context[3]=packed>>32;
         context[20]=scs.end[5];

      }
      if(dev->gen1_rq_candidates) {
         const uint32_t constants[]={0x3fffffff,0x3f7ffffe,0x3f800001,
            0x6000,0x8000,0x10000,0x60400,0x3ffd0fe0,0x914348a1,
            0x100,0x10002,0x1ffff,0x20000,0xfffff,0x400000,
            0x1f000001,0x34a00000,0xfe000000,0xfe1ffffe,0xfffe0000,0xfffeff00};
         memcpy((uint8_t*)u.cpu+110*2,constants,sizeof(constants));
         ((uint32_t*)u.cpu)[104/2]=304;
      }
      memcpy(context+12,&table,8);memcpy((uint8_t*)u.cpu+352,context,sizeof(context));
      memcpy((uint8_t*)u.cpu+320,&ctx,8);memcpy((uint8_t*)u.cpu+328,&table,8);
      ranges[range_count++]=(struct agx3_ioto_uniform_range){
         .start_halfs=0,.size_halfs=168,.address=u.gpu,
      };
      hk_cmd_buffer_retain_bo(cmd,dev->gen1_rq_code);
      hk_cmd_buffer_retain_bo(cmd,dev->gen1_rq_ift);
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
   if (s->b.info.rodata.size_16 != 0)
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
   if (!agx_native_graphics(&dev->dev) ||
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
 * Append an indirect launch of a built main program: the helper publishes
 * the grid's launch geometry, and the main record consumes it.
 */
static __attribute__((hot)) bool
hk_g17p_append_indirect_launch(
   struct hk_cmd_buffer *cmd, struct agx_grid grid,
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
          cmd, cmd->current_cs.cs, helper_s, helper_s->only_linked,
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
   if (!agx_native_graphics(&dev->dev) || !prog->bo ||
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
      if (!hk_g17p_append_indirect_launch(cmd, grid, local_size, 0, &program,
                                          &sink))
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
   const struct agx3_g17p_graphics_program_desc desc = {
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
   if (!agx3_g17g_compute_program_generate(dev->dev.chip, &desc, &program) ||
       !agx3_cdm_direct_template_sources_compute(
          false, false, false, &launch_desc.launch_sources) ||
       !agx3_m5_native_compute_program_bind(dev->dev.chip, &launch_desc, &launch)) {
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
      cs->strong_flush_end = cs->current;
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


/*
 * The SCS state block the native query runtime reads (the captured user
 * image, a zero tail up to the layout's section 12, and for instance layouts
 * the ScsLayoutGen2 parameter record at +0x80020) depends only on the packed
 * layout, which is fixed for a device, and the GPU never writes it: with the
 * block bound GPU-read-only the RT gate runs without a fault, and a
 * before/after comparison over 469 traversal dispatches finds no changed byte.
 * So it is initialized once per device and shared by every traversal
 * dispatch, instead of allocating and copying 8 MiB per dispatch or per
 * vkCmdTraceRays*. ASAHI_M5_RT_PERSISTENT_STATE=0 restores the per-dispatch
 * block for A/B diagnosis.
 */
static bool
hk_m5_rt_persistent_state_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0)
      enabled = debug_get_bool_option("ASAHI_M5_RT_PERSISTENT_STATE", true);
   return enabled;
}

static struct agx_ptr
hk_m5_persistent_rt_state(struct hk_device *dev, uint64_t packed, uint64_t size,
                          const struct agx_scs_layout_gen2 *scs)
{
   struct agx_ptr state = {0};
   simple_mtx_lock(&dev->m5_as_lock);
   if (!dev->m5_rt_state_bo) {
      /* Diagnostic: bind the block GPU-read-only, so that any GPU write,
       * including a transient one, faults instead of passing silently. */
      const enum agx_bo_flags flags =
         debug_get_bool_option("ASAHI_M5_RT_STATE_READONLY_PROBE", false)
            ? AGX_BO_READONLY : 0;
      struct agx_bo *bo =
         agx_bo_create(&dev->dev, size, 0x4000, flags, "M5 SCS state");
      if (bo) {
         uint8_t *map = agx_bo_map(bo);
         memcpy(map, dev->m5_rq_user_image, 0x800000);
         if (size > 0x800000)
            memset(map + 0x800000, 0, size - 0x800000);
         if (dev->m5_rq_instances)
            agx_scs_layout_gen2_parameters(scs, map + 0x80020);
         dev->m5_rt_state_bo = bo;
         dev->m5_rt_state_packed = packed;
         dev->m5_rt_state_size = size;
      }
   }
   if (dev->m5_rt_state_bo && dev->m5_rt_state_packed == packed &&
       dev->m5_rt_state_size >= size) {
      state.cpu = agx_bo_map(dev->m5_rt_state_bo);
      state.gpu = dev->m5_rt_state_bo->va->addr;
   }
   simple_mtx_unlock(&dev->m5_as_lock);
   return state;
}

__attribute__((hot)) void
hk_dispatch_m5_native(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                      struct hk_shader *s, struct agx_grid grid,
                      struct agx_workgroup local_size)
{
   const char *reason = "shader profile";
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *linked = s ? s->only_linked : NULL;
   if (s && !linked && s->info.stage == MESA_SHADER_VERTEX)
      linked = cmd->state.gfx.linked[MESA_SHADER_VERTEX];
   if (!agx_chip_g17_m5(dev->dev.chip) || cs->type != HK_CS_CDM ||
       /* Static TPS addresses are 15-bit halfword offsets. */
       !hk_g17p_compute_shader_supported(dev->dev.chip, s, linked, 1u << 16) ||
       ((s->b.info.scratch_size || linked->b.scratch_size) &&
        !(dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_RESOURCES)) ||
       (s->b.info.uses_ray_traversal && !dev->m5_native_rq) || s->info.agx3_txf_texture_valid ||
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
      reason = "matrix kernel feature";
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
   reason = "uniform ranges";
   if (!collected)
      goto refused;
   if (!internal && s->b.info.agx3_uniform_dma_count &&
       !hk_m5_compute_uniform_dma_ranges(cmd, s, root, ranges,
                                         ARRAY_SIZE(ranges), &count))
      goto refused;
   if (s->b.info.uses_ray_traversal) {
      reason = "native query dispatch shape";
      /* Internal query kernels use the compute query ABI. A software vertex
       * variant keeps the vertex ABI (hk_g17p_internal_graphics_uniform_ranges)
       * and therefore stays refused. */
      if (internal && s->info.stage != MESA_SHADER_TESS_CTRL &&
          s->info.stage != MESA_SHADER_TESS_EVAL &&
          s->info.stage != MESA_SHADER_GEOMETRY)
         goto refused;
      /* Keep the shader's captured payload profile, but derive every allocation
       * and parameter offset from the native ScsLayoutGen2 equations. In
       * particular, the driver's payload and the saved instance levels are
       * different sections; neither is the shader's ordinary spill buffer. */
      struct agx_scs_layout_gen2 scs;
      const uint64_t packed = dev->m5_rq_instances
         ? (hk_m5_rq_candidate_requested()
                     ? UINT64_C(0x01140000600100e8) /* 276-entry IFT50 tail */
                     : UINT64_C(0x00480000600100e8)) : UINT64_C(0x0028000060000070);
      if (!agx_scs_layout_gen2_init(packed, &scs))
         goto refused;
      const uint64_t state_size = MAX2(UINT64_C(0x800000),
                                       ALIGN_POT((uint64_t)scs.end[12], 0x4000));
      /* One device-lifetime state block (hk_m5_persistent_rt_state). The
       * per-dispatch block remains for ASAHI_M5_RT_PERSISTENT_STATE=0 and as
       * the fallback if the device block cannot be created; the per-raygen
       * dispatches of one vkCmdTraceRays* share it. */
      struct agx_ptr state = hk_m5_rt_persistent_state_enabled()
         ? hk_m5_persistent_rt_state(dev, packed, state_size, &scs)
         : (struct agx_ptr){0};
      const bool persistent_state = state.cpu != NULL;
      if (persistent_state)
         hk_cmd_buffer_retain_bo(cmd, dev->m5_rt_state_bo);
      const bool reuse_state = !persistent_state && cmd->m5_rt_state.share &&
                               cmd->m5_rt_state.state.cpu &&
                               cmd->m5_rt_state.packed == packed;
      const bool fresh_state = !persistent_state && !reuse_state;
      if (!persistent_state)
         state = reuse_state ? cmd->m5_rt_state.state
                             : hk_pool_alloc(cmd, state_size, 0x4000);
      /* The per-raygen dispatches of one vkCmdTraceRays* (see
       * hk_trace_rays_select_and_dispatch) read identical traversal uniforms:
       * the same state block, AS table and context. Build them once. */
      const bool shared_uniforms = cmd->m5_rt_state.share &&
                                   cmd->m5_rt_state.uniforms.cpu &&
                                   cmd->m5_rt_state.uniforms_packed == packed &&
                                   cmd->m5_rt_state.uniforms_state == state.gpu;
      if (shared_uniforms) {
         if (!state.cpu) return;
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs=0, .size_halfs=dev->m5_rq_instances ? 168 : 156,
            .address=cmd->m5_rt_state.uniforms.gpu,
         };
      } else {
         struct agx_ptr uniforms = hk_pool_alloc(cmd, 512, 16);
         if (!state.cpu || !uniforms.cpu) return;
         uint64_t as_table;
         if (dev->m5_rq_instances) {
            as_table = hk_m5_upload_as_table(cmd);
         } else {
            /* The native profile has96 resident cliques (3072 SCS slots).
             * A slot is a hardware resident identity, not an invocation index.
             * Slot0 is reserved by the native AS selector encoding. */
            struct agx_ptr table = hk_pool_alloc(cmd, (96 * 32 + 1) * 16, 16);
            if (!table.cpu) return;
            memset(table.cpu, 0, (96 * 32 + 1) * 16);
            as_table = table.gpu;
         }
         if (!as_table) return;
         if (fresh_state) {
            memcpy(state.cpu, dev->m5_rq_user_image, 0x800000);
            if (state_size > 0x800000)
               memset((uint8_t *)state.cpu + 0x800000, 0, state_size - 0x800000);
         }
         uint32_t context[32] = {0, 0, packed, packed >> 32};
         const uint32_t offsets[] = {0x18000,0x18200,0x18400,0x18500,0x18700,0x19b00};
         for (unsigned i = 0; i < 6; ++i) {
            uint64_t pointer = state.gpu + offsets[i];
            if (i == 4)
               pointer = as_table;
            memcpy(context + 4 + 2*i, &pointer, 8);
         }
         context[20] = scs.end[5];
         context[22] = scs.end[10] - scs.end[5];
         const unsigned levels = (packed >> 16) & 0xff;
         context[23] = (levels + (levels > 1)) * 8;
         if (dev->m5_rq_instances) {
            uint64_t extra = state.gpu + 0x80050;
            memcpy(context + 26, &extra, 8);
            extra = state.gpu + 0x18900;
            memcpy(context + 30, &extra, 8);
            if (fresh_state)
               agx_scs_layout_gen2_parameters(&scs, (uint8_t *)state.cpu + 0x80020);
         }
         if (cmd->m5_rt_state.share && fresh_state) {
            cmd->m5_rt_state.state = state;
            cmd->m5_rt_state.packed = packed;
         }
         if (!reuse_state && os_get_option("ASAHI_M5_RT_STATE_DIAG"))
            hk_m5_rt_state_diag_record(cmd, state, state_size);
         memset(uniforms.cpu, 0, 512);
         memcpy(uniforms.cpu, dev->m5_rq_uniforms, sizeof(dev->m5_rq_uniforms));
         if (dev->m5_rq_instances) {
            uint64_t sizes = (uint64_t)scs.end[6] | ((uint64_t)scs.end[11] << 32);
            uint64_t layout = packed;
            uint64_t driver = scs.end[5];
            memcpy((uint8_t *)uniforms.cpu + 48*2, &sizes, 8);
            memcpy((uint8_t *)uniforms.cpu + 52*2, &layout, 8);
            memcpy((uint8_t *)uniforms.cpu + 56*2, &driver, 8);
         }
         if (hk_m5_rq_candidate_requested()) {
            /* Fresh query-once native u110..113, separately bound because the
             * captured ordinary profile uses u140 for its per-clique state. */
            uint64_t query_payload = UINT64_C(0x3fffffff00008000);
            memcpy((uint8_t *)uniforms.cpu + 152*2, &query_payload, 8);
         }
         for (unsigned i = 0; i < (dev->m5_rq_instances ? 80 : 74); i += 2) {
            uint64_t pointer; memcpy(&pointer, (uint8_t *)uniforms.cpu + i*4, 8);
            if (pointer >= UINT64_C(0x10000000000) && pointer < UINT64_C(0x10000800000)) {
               pointer += state.gpu - UINT64_C(0x10000000000);
               memcpy((uint8_t *)uniforms.cpu + i*4, &pointer, 8);
            }
         }
         if (dev->m5_rq_instances)
            memcpy((uint8_t *)uniforms.cpu + 32*2, &as_table, 8);
         unsigned context_offset = dev->m5_rq_instances ? 352 : 320;
         memcpy((uint8_t *)uniforms.cpu + context_offset, context, sizeof(context));
         uint64_t ctx = uniforms.gpu + context_offset, table = as_table;
         memcpy((uint8_t *)uniforms.cpu + (dev->m5_rq_instances ? 320 : 296), &ctx, 8);
         memcpy((uint8_t *)uniforms.cpu + (dev->m5_rq_instances ? 328 : 304), &table, 8);
         ranges[count++] = (struct agx3_ioto_uniform_range){
            .start_halfs=0, .size_halfs=dev->m5_rq_instances ? 168 : 156, .address=uniforms.gpu,
         };
         if (cmd->m5_rt_state.share) {
            cmd->m5_rt_state.uniforms = uniforms;
            cmd->m5_rt_state.uniforms_packed = packed;
            cmd->m5_rt_state.uniforms_state = state.gpu;
         }
      }
      cs->m5_native_rq = true;
      hk_cmd_buffer_retain_bo(cmd, dev->m5_rq_code);
      hk_cmd_buffer_retain_bo(cmd, dev->m5_rq_ift);
      if (dev->m5_rq_diag)
         hk_cmd_buffer_retain_bo(cmd, dev->m5_rq_diag);
   }
   reason = "program bound";
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
      reason = "native constant-program extent";
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
   struct agx3_g17p_graphics_program program;
   const struct hk_m5_compute_loader *loader =
      hk_m5_loader_get(dev, linked, &desc);
   struct agx3_g17p_native_compute_program_desc launch_desc = {
      .program_va = reservation.va, .shader_va = shader_va,
   };
   struct agx3_compute_program launch;
   reason = "program generation";
   if (loader) {
      program.length = loader->length;
      memcpy(program.bytes, loader->bytes, loader->length);
      hk_g17p_patch_plan_apply(&loader->plan, ranges, program.bytes);
      if (debug_get_option_m5_verify_compute_loader()) {
         struct agx3_g17p_graphics_program fresh;
         if (!agx3_g17g_compute_program_generate(dev->dev.chip, &desc, &fresh) ||
             fresh.length != program.length ||
             memcmp(fresh.bytes, program.bytes, program.length)) {
            reason = "cached program byte mismatch";
            goto refused;
         }
      }
   } else if (!agx3_g17g_compute_program_generate(dev->dev.chip, &desc, &program)) {
      goto refused;
   }
   /* Native instance launch words are 0x00800000/0x01000080:
    * the nested traversal path uses the 64-byte TI spill class and ID32 bit. */
   const bool native_instances = dev->m5_rq_instances && s->b.info.uses_ray_traversal;
   if (!agx3_cdm_direct_template_sources_compute(s->b.info.uses_resource_info_query || native_instances,
                                                 s->b.info.uses_ray_traversal, native_instances, &launch_desc.launch_sources))
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
   if (!agx3_m5_native_compute_program_bind(dev->dev.chip, &launch_desc, &launch))
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
   reason = "dispatch record";
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
      hk_m5_compute_resources(cs, desc.scratch_size,
                              s->b.info.uses_ray_traversal ? 128 : 0,
                              desc.threadgroup_memory_size_B,
                              local[0] * local[1] * local[2]);
      cs->stats.cmds++;
      return;
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
   hk_m5_compute_resources(cs, desc.scratch_size,
                           s->b.info.uses_ray_traversal ? 128 : 0,
                           desc.threadgroup_memory_size_B,
                           local[0] * local[1] * local[2]);
   cs->stats.cmds++;
   return;
refused:
   if (debug_get_option_m5_launch_debug()) {
      fprintf(stderr, "M5 compute refused: %s stage=%u scratch=%u/%u "
              "gprs=%u staging=%u:%u local=%u preamble=%u "
              "textures=%u samplers=%u heap=%u txf=%u sets=%u "
              "resource_flags=%u%u%u%u%u indirect=%u\n",
              reason, s ? s->info.stage : 0,
              s ? s->b.info.scratch_size : 0, linked ? linked->b.scratch_size : 0,
              linked ? linked->b.nr_gprs : 0,
              s ? s->b.info.agx3_staging_known : 0,
              s ? s->b.info.agx3_staging_halfs : 0,
              s ? s->b.info.local_size : 0, s ? s->b.info.has_preamble : 0,
              s ? s->b.info.texture_state_count : 0,
              s ? s->b.info.sampler_state_count : 0,
              s ? s->b.info.uses_sampler_heap : 0,
              linked ? linked->b.uses_txf : 0, s ? s->info.set_count : 0,
              s ? s->info.agx3_txf_texture_valid : 0,
              s ? s->info.agx3_storage_image_valid : 0,
              s ? s->info.agx3_sampled_storage_valid : 0,
              s ? s->info.agx3_indexed_storage_valid : 0,
              s ? s->info.agx3_indexed_sampler_valid : 0, agx_is_indirect(grid));
   }
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

   if (!agx_native_graphics(&dev->dev) || cs->type != HK_CS_CDM ||
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

         if (write_only) {
            pointers[0] =
               set_address + s->info.agx3_storage_image.pbe_offset_B;
            pointers[1] = access_control;
         } else {
            pointers[0] = set_address +
                          s->info.agx3_storage_image.texture_offset_B;
            pointers[1] =
               set_address + s->info.agx3_storage_image.pbe_offset_B;
            pointers[2] = access_control;
            /* The native read_write ISEL clamps each unsigned coordinate
             * against uniform half12. The consumed vector word is 0xffff. */
            pointers[3] = UINT64_C(0xffff);
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
            if (!set && descriptors->push[0]) {
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
      if (!hk_g17p_append_indirect_launch(cmd, grid, local_size,
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
    * strengthened, and let this launch's own tail end the stream instead. */
   const bool app_dispatch =
      !cmd->in_meta && s && s->info.stage == MESA_SHADER_COMPUTE &&
      agx_chip_elide_app_dispatch_barrier(dev->dev.chip);
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

static void
dispatch_grid(struct hk_cmd_buffer *cmd, struct agx_grid grid, bool in_threads)
{
   hk_cmd_resolve_compute_border(cmd);
   struct hk_shader *s = hk_only_variant(cmd->state.cs.shader);
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
      desc->root_dirty = true;
      grid = agx_grid_indirect(patched);
   }

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, true /* compute */);
   if (!cs)
      return;

   struct agx_workgroup local_size =
      agx_workgroup(s->b.info.workgroup_size[0], s->b.info.workgroup_size[1],
                    s->b.info.workgroup_size[2]);

   /* Internal compute work must not contribute to the application's active
    * pipeline query. This also honours hk_pipeline_stat_addr's !in_meta
    * contract in debug builds; graphics meta pauses its own counters. */
   if (!cmd->in_meta) {
      uint64_t stat = hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT);

      if (hk_stat_enabled(stat)) {
         perf_debug(cmd, "CS invocation statistic");
         uint64_t grid = cmd->state.cs.descriptors.root.cs.group_count_addr;

         libagx_increment_cs_invocations(cmd, agx_1d(1), AGX_BARRIER_ALL, grid,
                                         stat, agx_workgroup_threads(local_size));
      }
   }

   if (!agx_is_indirect(grid) && !in_threads) {
      grid.count[0] *= local_size.x;
      grid.count[1] *= local_size.y;
      grid.count[2] *= local_size.z;
   }

   hk_dispatch_with_local_size(cmd, cs, s, grid, local_size);
   if (unlikely(hk_cmd_buffer_device(cmd)->frame_anatomy)) {
      hk_frame_anatomy_note_fs(cs, s);
      const uint64_t wg = agx_workgroup_threads(local_size);
      const uint64_t threads = agx_is_indirect(grid) ? 0 :
         (uint64_t)grid.count[0] * grid.count[1] * grid.count[2];
      hk_frame_anatomy_note_grid(cs, s, threads, wg ? threads / wg : 0,
                                 agx_is_indirect(grid));
   }
   cs->stats.calls++;
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
   desc->root.cs.group_count_addr =
      hk_pool_upload(cmd, &group_count, sizeof(group_count), 8);
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

   desc->root.cs.group_count_addr =
      hk_pool_upload(cmd, &group_count, sizeof(group_count), 8);
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
   struct hk_descriptor_state *rt = &cmd->state.rt.descriptors;

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
   desc->root_dirty = true;

   /* Conditional rendering predicates draws, dispatches and attachment
    * clears only; vkCmdTraceRays* must execute regardless of it. */
   const bool conditional_active = cmd->state.conditional.active;
   cmd->state.conditional.active = false;

   dispatch(cmd, grid);

   cmd->state.conditional.active = conditional_active;

   /* Carry the state back before restoring, so a pushed descriptor set the
    * flush above allocated stays owned by the ray-tracing state. */
   *rt = cmd->state.cs.descriptors;
   rt->root.cs.trace_rays_addr = 0;

   cmd->state.cs = saved;
   cmd->state.cs.descriptors.root_dirty = true;
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
    * helper is built for devices that enabled sparseResidencyBuffer, which is
    * the only way to create a buffer carrying a bitmap; if it is somehow
    * absent, keep the old behaviour rather than refusing the dispatch. */
   const bool have_sparse_helper = dev->g17p_grid_helper_sparse != NULL;
   const bool use_primary =
      agx_native_graphics(&dev->dev) && buffer->va &&
      (!buffer->sparse_map || have_sparse_helper);

   uint64_t dispatch_addr = hk_buffer_address(buffer, offset, !use_primary);
   assert(dispatch_addr != 0);

   desc->root.cs.group_count_addr = dispatch_addr;
   desc->root.cs.grid_helper_sparse_map_addr =
      (use_primary && buffer->sparse_map) ? buffer->sparse_map->va->addr : 0;
   dispatch(cmd, agx_grid_indirect(dispatch_addr));
}

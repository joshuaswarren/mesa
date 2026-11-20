/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_agx3_ioto_program_core.h"

#include "asahi/compiler/agx3_graphics_program.h"
#include "asahi/compiler/agx3_compute_program.h"
#include "asahi/lib/agx_ioto_core.h"
#include "asahi/lib/agx_tilebuffer.h"
#include "asahi/compiler/agx3_binder.h"
#include "util/format/u_format.h"

#include <string.h>

bool
hk_agx3_g17p_constant_program_from_binary(
   const void *binary, size_t binary_size, size_t offset, size_t size,
   uint64_t allocation_va, size_t allocation_size,
   struct agx_ioto_constant_program *output)
{
   const uint64_t limit = UINT64_C(1) << 48;
   if (!binary || !output || !size || (offset & 1) || (size & 1) ||
       offset > binary_size || size > binary_size - offset ||
       binary_size > allocation_size || !allocation_va ||
       allocation_va >= limit || (allocation_va & 1) ||
       binary_size > limit - allocation_va)
      return false;

   *output = (struct agx_ioto_constant_program){
      .code = (const uint8_t *)binary + offset,
      .code_size = size,
      .code_va = allocation_va + offset,
      .absolute_branch = true,
   };
   return true;
}

bool
hk_agx3_ioto_program_generate(
   enum agx_chip chip, const struct hk_agx3_ioto_program_desc *desc,
   void *output, size_t capacity, size_t *written)
{
   if (!desc || !agx_ioto_target_supported(chip))
      return false;

   const struct agx_ioto_profile_control profile = {
      .chip = chip,
      .payload0 = desc->profile_payload0,
      .payload1 = desc->profile_payload1,
   };

   return agx_ioto_straight_line_program_compose(
      chip, desc->program_va, desc->body, desc->body_size, &desc->tail,
      &profile, output, capacity, written);
}

bool
hk_agx3_ioto_program_generate_no_pixrel(
   enum agx_chip chip, const struct hk_agx3_ioto_program_desc *desc,
   const struct agx_ioto_preshader_desc *preshader, void *output,
   size_t capacity, size_t *written)
{
   if (!desc || !agx_ioto_target_supported(chip))
      return false;

   const struct agx_ioto_profile_control profile = {
      .chip = chip,
      .payload0 = desc->profile_payload0,
      .payload1 = desc->profile_payload1,
   };

   return agx_ioto_program_compose_no_pixrel(
      chip, desc->program_va, desc->body, desc->body_size, &desc->tail,
      &profile, preshader, output, capacity, written);
}

bool
hk_agx3_g17p_single_layer_rendering(uint32_t view_mask, uint32_t layer_count)
{
   return view_mask == 0 && layer_count == 1;
}

bool
hk_agx3_g17p_rt0_tile_layout_supported(struct agx_tilebuffer_layout *tib)
{
   if (!tib)
      return false;

   /* Layering changes only the background/end-of-tile image coordinates and
    * descriptors.  It does not change the physical tilebuffer allocation.
    * Validate the latter through the same bounded RT0 contract, while the
    * G17P array BG/EOT programs independently carry the layer coordinate.
    */
   struct agx_tilebuffer_layout physical = *tib;
   physical.layered = false;
   /* Spilled attachments are accessed by the fragment epilog in memory.
    * Validate only the in-tile targets here, retaining allocation and offsets. */
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
      if (!physical.spilled[rt])
         continue;
      if (physical.logical_format[rt] == PIPE_FORMAT_NONE)
         return false;
      physical.logical_format[rt] = PIPE_FORMAT_NONE;
      physical.spilled[rt] = false;
   }

   /* Attachmentless rendering learns its sample count at draw time. An empty
    * pass may end before that happens. Validate its physical layout as 1x,
    * without changing the deferred sample count in the render state. The
    * physical validator still rejects any other occupied render target.
    */
   if (physical.nr_samples == 0 && physical.sample_size_B == 0 &&
       physical.logical_format[0] == PIPE_FORMAT_NONE)
      physical.nr_samples = 1;

   /* M5's native imageblock reserves one eight-byte sample for a layered
    * colourless pass (agx_build_tilebuffer_layout_for_imageblock). It holds
    * no target; check the rest of the layout as the unlayered empty one. */
   bool colourless = true;
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt)
      colourless &= physical.logical_format[rt] == PIPE_FORMAT_NONE;
   if (tib->native_imageblock && tib->layered && physical.nr_samples <= 1 &&
       physical.sample_size_B == 8 && colourless) {
      physical.sample_size_B = 0;
      if (!physical.nr_samples)
         physical.nr_samples = 1;
   }

   return agx_g17p_rt0_tile_layout_supported(&physical);
}

bool
hk_agx3_format_needs_shader_resolve(enum pipe_format format)
{
   /* A 32-bit float target's lane is stored raw, and resolving it on the
    * way out averages the bit patterns: an R32 or RG32 float view resolved
    * this way reads back values like 4e22 on an A18 Pro. Keep the samples
    * and resolve after the pass instead. */
   if (util_format_is_float(format) &&
       util_format_get_component_bits(format, UTIL_FORMAT_COLORSPACE_RGB, 0) ==
          32)
      return true;

   /* An RGB9E5 tile keeps each sample's packed word, which averages no
    * better. */
   return format == PIPE_FORMAT_R9G9B9E5_FLOAT;
}

bool
hk_agx3_g17p_direct_color_resolve_supported(
   struct agx_tilebuffer_layout *tib, unsigned color_attachment_count,
   bool has_resolve_target, bool average_resolve, bool store_source)
{
   if (!tib || hk_agx3_format_needs_shader_resolve(tib->logical_format[0]))
      return false;

   return (tib->nr_samples == 2 || tib->nr_samples == 4) &&
          color_attachment_count == 1 && has_resolve_target &&
          average_resolve && !store_source &&
          hk_agx3_g17p_rt0_tile_layout_supported(tib);
}

bool
hk_agx3_g17p_multisample_store_supported(struct agx_tilebuffer_layout *tib,
                                         unsigned color_attachment_count,
                                         bool store_source,
                                         unsigned target_samples)
{
   /* The mirror image of the direct resolve: this is the one path that keeps
    * the samples. It needs storeOp = STORE, a target whose sample count is
    * exactly the tile's, and the same bounded single-target tile layout the
    * native EOT substitution covers. */
   return tib && (tib->nr_samples == 2 || tib->nr_samples == 4) &&
          color_attachment_count == 1 && store_source &&
          target_samples == tib->nr_samples &&
          hk_agx3_g17p_rt0_tile_layout_supported(tib);
}

bool
hk_agx3_g17p_graphics_program_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *size)
{
   /* The shared composer's range budget covers the GL collector. hk's
    * collector and cache key hold at most HK_AGX3_G17P_GRAPHICS_MAX_RANGES,
    * so a larger list is refused here, fail closed. */
   if (range_count > HK_AGX3_G17P_GRAPHICS_MAX_RANGES)
      return false;
   return agx3_g17p_graphics_program_bound(chip, ranges, range_count, size);
}

static struct agx3_g17p_graphics_program_desc
hk_agx3_g17p_shared_graphics_desc(
   const struct hk_agx3_g17p_graphics_program_desc *desc)
{
   return (struct agx3_g17p_graphics_program_desc){
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .uniform_ranges = desc->uniform_ranges,
      .uniform_range_count = desc->uniform_range_count,
      .shader_binary_size = desc->shader_binary_size,
      .scratch_size = desc->scratch_size,
      .nr_gprs = desc->nr_gprs,
      .compiler_d_valid = desc->compiler_d_valid,
      .compiler_d = desc->compiler_d,
      .fragment = desc->fragment,
      .writes_sample_mask = desc->writes_sample_mask,
      .colour_classes = desc->colour_classes,
      .imageblock = desc->imageblock,
      .preloaded_root = desc->preloaded_root,
      .patchable = desc->patchable,
      .constant_program = desc->constant_program,
      .cprog_inputs = desc->cprog_inputs,
   };
}

bool
hk_agx3_g17p_basic_graphics_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_graphics_program_desc *desc,
   struct hk_agx3_g17p_graphics_program *program)
{
   if (!program || !desc)
      return false;

   const struct agx3_g17p_graphics_program_desc shared_desc =
      hk_agx3_g17p_shared_graphics_desc(desc);
   struct agx3_g17p_graphics_program shared;
   const bool generated = agx_chip_g17_m5(chip)
      ? agx3_g17g_graphics_program_generate(chip, &shared_desc, &shared)
      : agx3_g17p_graphics_program_generate(chip, &shared_desc, &shared);
   if (!generated)
      return false;
   memcpy(program->bytes, shared.bytes, shared.length);
   program->length = shared.length;
   return true;
}

bool
hk_agx3_g17p_graphics_program_template_generate(
   enum agx_chip chip,
   const struct hk_agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program_template *program)
{
   if (!desc)
      return false;
   const struct agx3_g17p_graphics_program_desc shared_desc =
      hk_agx3_g17p_shared_graphics_desc(desc);
   return agx_chip_g17_m5(chip)
      ? agx3_g17g_graphics_program_template_generate(chip, &shared_desc, program)
      : agx3_g17p_graphics_program_template_generate(chip, &shared_desc, program);
}

bool
hk_agx3_g17p_graphics_program_template_relocate(
   enum agx_chip chip,
   const struct hk_agx3_g17p_graphics_program_desc *template_desc,
   const struct hk_agx3_g17p_graphics_program_desc *target_desc,
   const struct agx3_g17p_graphics_program_template *program,
   void *output, size_t capacity, uint16_t *written)
{
   if (!template_desc || !target_desc)
      return false;
   const struct agx3_g17p_graphics_program_desc shared_template =
      hk_agx3_g17p_shared_graphics_desc(template_desc);
   const struct agx3_g17p_graphics_program_desc shared_target =
      hk_agx3_g17p_shared_graphics_desc(target_desc);
   return agx3_g17p_graphics_program_template_relocate(
      chip, &shared_template, &shared_target, program, output, capacity,
      written);
}

bool
hk_agx3_g17p_combined_sampler_pointers(
   const struct hk_agx3_g17p_combined_sampler_heap *heap,
   uint64_t pointers[2])
{
   if (!heap || !pointers || heap->descriptor_set_va == 0 ||
       heap->sampler_heap_va == 0 || (heap->texture_offset_B & 7) != 0 ||
       heap->sampler_element_size_B != 32)
      return false;

   /* The G17P software descriptor carries the raw native heap index. Entry
    * zero is reserved for null/fetch sampling; bounds remain explicit. */
   const uint32_t native_index = heap->sampler_index;
   if (native_index >= heap->sampler_count)
      return false;

   const uint64_t texture_pointer_va =
      heap->descriptor_set_va + heap->texture_offset_B;
   const uint64_t sampler_offset =
      (uint64_t)native_index * heap->sampler_element_size_B;
   const uint64_t sampler_pointer_va = heap->sampler_heap_va + sampler_offset;
   if (texture_pointer_va < heap->descriptor_set_va ||
       sampler_pointer_va < heap->sampler_heap_va ||
       texture_pointer_va > UINT64_MAX - 31 || sampler_pointer_va > UINT64_MAX - 31)
      return false;

   pointers[0] = texture_pointer_va;
   pointers[1] = sampler_pointer_va;
   return true;
}

bool
hk_agx3_g17p_sampled_storage_pointers(
   const struct hk_agx3_g17p_sampled_storage_heap *heap,
   uint64_t pointers[4])
{
   if (!heap || !pointers || !heap->pbe_descriptor_set_va ||
       !heap->write_control_va || (heap->write_control_va & 31) ||
       heap->write_control_va > UINT64_MAX - 31 ||
       (heap->pbe_offset_B & 7) != 0 ||
       heap->pbe_descriptor_set_va > UINT64_MAX - heap->pbe_offset_B ||
       heap->pbe_descriptor_set_va + heap->pbe_offset_B > UINT64_MAX - 31)
      return false;

   uint64_t staged[4] = {0};
   if (!hk_agx3_g17p_combined_sampler_pointers(&heap->sampled, staged))
      return false;

   staged[2] = heap->pbe_descriptor_set_va + heap->pbe_offset_B;
   staged[3] = heap->write_control_va;
   memcpy(pointers, staged, sizeof(staged));
   return true;
}

bool
hk_agx3_g17p_indexed_storage_pointers(
   const struct hk_agx3_g17p_indexed_storage_heap *heap,
   uint64_t pointers[4])
{
   if (!heap || !pointers || !heap->texture_heap_va ||
       !heap->sampler_heap_va || !heap->pbe_descriptor_set_va ||
       !heap->write_control_va || (heap->write_control_va & 31) ||
       heap->write_control_va > UINT64_MAX - 31 ||
       (heap->pbe_offset_B & 7) != 0 ||
       heap->pbe_descriptor_set_va > UINT64_MAX - heap->pbe_offset_B ||
       heap->pbe_descriptor_set_va + heap->pbe_offset_B > UINT64_MAX - 31)
      return false;

   const uint64_t staged[4] = {
      heap->texture_heap_va,
      heap->sampler_heap_va,
      heap->pbe_descriptor_set_va + heap->pbe_offset_B,
      heap->write_control_va,
   };
   memcpy(pointers, staged, sizeof(staged));
   return true;
}

bool
hk_agx3_g17p_root_range(uint16_t start_halfs, uint8_t set_count,
                        uint64_t root_va,
                        struct agx3_ioto_uniform_range *range)
{
   if (!range || root_va == 0 || (start_halfs & 1) != 0 ||
       set_count > HK_AGX3_G17P_MAX_ROOT_SETS)
      return false;

   *range = (struct agx3_ioto_uniform_range){
      .start_halfs = start_halfs,
      .size_halfs = 4 * (1 + set_count),
      .address = root_va,
   };
   return true;
}

bool
hk_agx3_g17p_rodata_range(uint16_t base_halfs, uint16_t size_halfs,
                          uint64_t bo_va, uint32_t offset_B,
                          uint64_t bo_size_B,
                          struct agx3_ioto_uniform_range *range)
{
   const uint32_t loaded_halfs = ((uint32_t)size_halfs + 1) & ~1u;

   if (!range || !bo_va || !size_halfs || (base_halfs & 1) ||
       loaded_halfs > 512 || base_halfs > 512 - loaded_halfs ||
       offset_B > bo_size_B || bo_va > UINT64_MAX - offset_B)
      return false;

   const uint64_t loaded_bytes = (uint64_t)loaded_halfs * 2;
   const uint64_t address = bo_va + offset_B;
   if (loaded_bytes > bo_size_B - offset_B || (address & 3))
      return false;

   *range = (struct agx3_ioto_uniform_range){
      .start_halfs = base_halfs,
      .size_halfs = loaded_halfs,
      .address = address,
   };
   return true;
}

bool
hk_agx3_g17p_fragment_blend_constants_range(
   uint16_t start_halfs, uint64_t blend_constants_va,
   struct agx3_ioto_uniform_range *range)
{
   if (!blend_constants_va || !range || (start_halfs & 1) != 0 ||
       start_halfs > 512 - 8)
      return false;

   *range = (struct agx3_ioto_uniform_range){
      .start_halfs = start_halfs,
      .size_halfs = 8,
      .address = blend_constants_va,
   };
   return true;
}

bool
hk_agx3_g17p_native_indirect_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_g17p_native_indirect_dispatch *dispatch,
   struct agx3_compute_dispatch_sink *sink)
{
   return agx3_g17p_native_indirect_compute_dispatch_append(program, dispatch,
                                                             sink);
}

/* Build a dispatch-local Ioto program from dword-aligned root, descriptor and
 * constant ranges, then bind that exact allocation to the native A18 Pro CDM
 * launch state.  Spilling and indirect dispatch have separate contracts. */
bool
hk_agx3_g17p_compute_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_compute_program_desc *desc,
   struct hk_agx3_g17p_compute_program *program)
{
   if (!program || !desc)
      return false;
   const struct agx3_g17p_compute_program_desc common_desc = {
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .uniform_ranges = desc->uniform_ranges,
      .uniform_range_count = desc->uniform_range_count,
      .shader_binary_size = desc->shader_binary_size,
      .scratch_size = desc->scratch_size,
      .nr_gprs = desc->nr_gprs,
      .compiler_d_valid = desc->compiler_d_valid,
      .compiler_d = desc->compiler_d,
      .threadgroup_memory_size_B = desc->threadgroup_memory_size_B,
      .launch_sources = desc->launch_sources,
      .ldshdr_f1 = desc->ldshdr_f1,
      .constant_program = desc->constant_program,
      .cprog_inputs = desc->cprog_inputs,
   };
   struct agx3_g17p_compute_program common;
   if (!agx3_g17p_compute_program_generate(chip, &common_desc, &common))
      return false;

   struct hk_agx3_g17p_compute_program staged = {0};
   memcpy(staged.bytes, common.bytes, common.length);
   staged.length = common.length;
   staged.launch = common.launch;
   *program = staged;
   return true;
}

bool
hk_agx3_g17p_precompiled_data_layout(size_t data_size_B,
                                     uint32_t *upload_size_B,
                                     uint16_t *uniform_size_halfs)
{
   /* The uniform-range ABI counts 16-bit halves, but G17P can only load an
    * even number of them.  Leave one even half below UINT16_MAX so rounding
    * a packed argument block up to a dword cannot wrap the field. */
   if (!upload_size_B || !uniform_size_halfs || data_size_B == 0 ||
       data_size_B > (UINT16_MAX - 1u) * 2u)
      return false;

   const uint32_t aligned = ((uint32_t)data_size_B + 3u) & ~3u;
   *upload_size_B = aligned;
   *uniform_size_halfs = aligned / 2u;
   return true;
}

bool
hk_agx3_g17p_native_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink)
{
   return agx3_g17p_native_compute_dispatch_append(program, grid, sink);
}

bool
hk_agx3_g17p_native_compute_dispatch_append_with_barrier(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid, uint8_t barrier_flags,
   struct agx3_compute_dispatch_sink *sink)
{
   return agx3_g17p_native_compute_dispatch_append_with_barrier(
      program, grid, barrier_flags, sink);
}

bool
hk_agx3_g17s_minimal_compute_program_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc, void *output,
   size_t capacity, size_t *written, struct agx3_compute_program *program)
{
   if (chip != AGX_CHIP_G17S)
      return false;

   return agx3_g17s_minimal_compute_program_prepare(
      desc, output, capacity, written, program);
}

bool
hk_agx3_g17s_minimal_compute_program_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program)
{
   return agx3_g17s_minimal_compute_program_upload_prepare(
      chip, desc, upload, written, program);
}

bool
hk_agx3_g17s_minimal_compute_program_live_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx_ioto_g17s_minimal_compute_resources *resources,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program)
{
   return agx3_g17s_minimal_compute_program_live_upload_prepare(
      chip, desc, resources, upload, written, program);
}

bool
hk_agx3_g17s_minimal_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink)
{
   return agx3_g17s_minimal_compute_dispatch_append(program, grid, sink);
}

bool
hk_agx3_g17g_graphics_program_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *size)
{
   if (range_count > HK_AGX3_G17P_GRAPHICS_MAX_RANGES)
      return false;
   return agx3_g17g_graphics_program_bound(chip, ranges, range_count, size);
}

bool
hk_agx3_g17g_basic_graphics_program_generate(
   enum agx_chip chip, const struct hk_agx3_g17p_graphics_program_desc *desc,
   struct hk_agx3_g17p_graphics_program *program)
{
   if (!agx_chip_m5_paths(chip) || !desc || !program)
      return false;
   const struct agx3_g17p_graphics_program_desc shared_desc =
      hk_agx3_g17p_shared_graphics_desc(desc);
   struct agx3_g17p_graphics_program shared;
   if (!agx3_g17g_graphics_program_generate(chip, &shared_desc, &shared))
      return false;
   memcpy(program->bytes, shared.bytes, shared.length);
   program->length = shared.length;
   return true;
}

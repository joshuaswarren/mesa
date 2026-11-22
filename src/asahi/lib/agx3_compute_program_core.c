/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_compute_program_core.h"

#include "asahi/genxml/agx3_cmdstream.h"

#include <assert.h>
#include <string.h>

_Static_assert(AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE ==
                  AGX3_CDM_DIRECT_LAUNCH_SIZE + AGX3_CDM_BARRIER_SIZE,
               "G17S dispatch size must cover launch plus barrier");
_Static_assert(AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE ==
                  AGX3_CDM_DIRECT_LAUNCH_SIZE + AGX3_CDM_BARRIER_SIZE,
               "G17P native dispatch size must cover launch plus barrier");
_Static_assert(AGX3_G17P_NATIVE_INDIRECT_LAUNCH_SIZE ==
                  AGX3_CDM_INDIRECT_LAUNCH_SIZE,
               "G17P indirect launch must be the grounded compact record");
_Static_assert(AGX3_G17P_NATIVE_INDIRECT_MAIN_SIZE ==
                  AGX3_CDM_INDIRECT_LAUNCH_SIZE + AGX3_CDM_BARRIER_SIZE,
               "G17P indirect dispatch size must cover launch plus barrier");
_Static_assert(AGX3_G17P_NATIVE_INDIRECT_COMPUTE_DISPATCH_SIZE == 72,
               "G17P indirect dispatch must cover the 44+28-byte pair");

bool
agx3_compute_program_prepare(const struct agx3_compute_program_desc *desc,
                             void *output, size_t capacity, size_t *written,
                             struct agx3_compute_program *program)
{
   if (!desc || !output || !written || !program ||
       desc->profile.chip != desc->chip ||
       !agx3_cdm_direct_launch_target_grounded(desc->chip))
      return false;

   struct agx3_cdm_direct_template launch_template;
   if (!agx3_cdm_direct_template_build_from_sources(&desc->launch_sources,
                                                    &launch_template))
      return false;

   size_t staged_written = 0;
   if (!agx_ioto_program_compose_no_pixrel(
          desc->chip, desc->program_va, desc->body, desc->body_size,
          &desc->tail, &desc->profile, &desc->preshader, output, capacity,
          &staged_written))
      return false;

   *program = (struct agx3_compute_program){
      .chip = desc->chip,
      .program_va = desc->program_va,
      .launch_template = launch_template,
   };
   *written = staged_written;
   return true;
}

bool
agx3_g17p_native_compute_program_bind(
   const struct agx3_g17p_native_compute_program_desc *desc,
   struct agx3_compute_program *program)
{
   if (!desc || !program || !agx_chip_native_agx3(desc->chip) ||
       desc->program_va == 0 || desc->shader_va == 0 ||
       desc->program_va > UINT64_C(0xffffffffffff) ||
       desc->shader_va > UINT64_C(0xffffffffffff) ||
       (desc->program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       (desc->shader_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0)
      return false;

   struct agx3_cdm_direct_template launch_template;
   if (!agx3_cdm_direct_template_build_from_sources(&desc->launch_sources,
                                                    &launch_template))
      return false;

   *program = (struct agx3_compute_program){
      .chip = desc->chip,
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .launch_template = launch_template,
      .g17p_native_compute_live_bound = true,
   };
   return true;
}

static bool
agx3_hal300_native_compute_program_bind(
   enum agx_chip chip, bool m5_live,
   const struct agx3_g17p_native_compute_program_desc *desc,
   struct agx3_compute_program *program)
{
   if (!program || !desc)
      return false;

   /* Validate via the shared native binder, then re-tag for the HAL300 chip. */
   struct agx3_g17p_native_compute_program_desc native = *desc;
   native.chip = AGX_CHIP_G17P;
   struct agx3_compute_program staged;
   if (!agx3_g17p_native_compute_program_bind(&native, &staged))
      return false;

   staged.chip = chip;
   staged.g17p_native_compute_live_bound = false;
   if (m5_live)
      staged.m5_native_compute_live_bound = true;
   else
      staged.g17g_native_compute_live_bound = true;
   *program = staged;
   return true;
}

/* The j873 G18G compute path: G18G launches G17G programs (its code is G17G
 * code re-encoded as agx3d) through these g17g helpers, which keep the
 * j873 launch and barrier forms. The G17G-family parts use the agx3_m5_*
 * helpers below. */
bool
agx3_g17g_native_compute_program_bind(
   const struct agx3_g17p_native_compute_program_desc *desc,
   struct agx3_compute_program *program)
{
   return agx3_hal300_native_compute_program_bind(AGX_CHIP_G17G, false, desc,
                                                  program);
}

bool
agx3_m5_native_compute_program_bind(
   enum agx_chip chip,
   const struct agx3_g17p_native_compute_program_desc *desc,
   struct agx3_compute_program *program)
{
   if (!agx_chip_g17_m5(chip) || !desc ||
       (desc->ordering_memory_scope &&
        !agx3_cdm_barrier_flush_memory_flags(chip, 0)))
      return false;

   if (!agx3_hal300_native_compute_program_bind(chip, true, desc, program))
      return false;

   program->ordering_memory_scope = desc->ordering_memory_scope;
   return true;
}

static bool
agx3_g17p_native_compute_launch_pack(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   uint8_t packed[AGX3_CDM_DIRECT_LAUNCH_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_CDM_DIRECT_LAUNCH_SIZE);
   if (!program || !grid ||
       !agx_chip_agx3_path(program->chip) ||
       program->program_va == 0 ||
       program->program_va > UINT64_C(0xffffffffffff) ||
       (program->program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       program->launch_template.word0 >= (UINT32_C(1) << 29) ||
       program->launch_template.word1 >= (UINT32_C(1) << 22) ||
       grid->dimensions == 0 || grid->dimensions > 3 ||
       grid->g15x_has_multiple_agcs)
      return false;

   uint32_t global_size[3];
   if (!agx3_cdm_direct_grid_resolve(grid->source, grid->size,
                                     grid->local_size, global_size))
      return false;

   for (unsigned axis = 0; axis < 3; ++axis) {
      if (global_size[axis] == 0 || grid->local_size[axis] == 0 ||
          (axis >= grid->dimensions &&
           (global_size[axis] != 1 || grid->local_size[axis] != 1)))
         return false;
   }

   /* The G15 firmware (T6030, RTKit 2419) parses the older launch layout:
    * word 1 carries VA[47:32] in its high half, word 2 VA >> 6 and word 3
    * the bare control 0x40.  There is no room for template word 1, and the
    * template's subkernel bit stays set for one-dimensional grids too. */
   if (agx_chip_native_g15(program->chip)) {
      if (program->launch_template.word1 != 0)
         return false;

      agx3_cmdstream_store_le32(packed, program->launch_template.word0);
      agx3_cmdstream_store_le32(packed + 4,
                                (uint32_t)(program->program_va >> 32) << 16);
      agx3_cmdstream_store_le32(packed + 8,
                                (uint32_t)(program->program_va >> 6));
      agx3_cmdstream_store_le32(packed + 12, UINT32_C(0x40));
      for (unsigned axis = 0; axis < 3; ++axis) {
         agx3_cmdstream_store_le32(packed + 16 + axis * 4, global_size[axis]);
         agx3_cmdstream_store_le32(packed + 28 + axis * 4,
                                   grid->local_size[axis]);
      }
      return true;
   }

   /* The output-positive A18 Pro add3 record stores the Ioto program pointer
    * as a low VA>>6 word followed by 0x40000000 | VA[47:40].  Word 1 still
    * carries the upper pointer bits and must be derived from the VA, not
    * hard-coded -- see the note on the store below. */
   uint32_t word0 = program->launch_template.word0;
   if ((agx_chip_native_hal200(program->chip) ||
        program->m5_native_compute_live_bound) &&
       global_size[1] < 2 && global_size[2] < 2)
      word0 &= UINT32_C(0xfff1ffff);
   agx3_cmdstream_store_le32(packed, word0);
   /* Word 1 bits 22..31 carry program_va[47:38]; the hardware reconstructs
    * pipeline = ((word1 >> 22) << 38) | (word@+8 << 6). */
   agx3_cmdstream_store_le32(
      packed + 4,
      (uint32_t)(((program->program_va >> 38) & 0x3ff) << 22) |
         program->launch_template.word1);
   agx3_cmdstream_store_le32(packed + 8,
                             (uint32_t)(program->program_va >> 6));
   /* Word 3 low byte carries program_va[47:40]; bit 30 is the record flag.
    * Both word 1 bits 22..31 (va[47:38]) and this byte (va[47:40]) must be
    * derived from program_va so bit 40 is never injected for low USC VAs.
    */
   agx3_cmdstream_store_le32(
      packed + 12,
      UINT32_C(0x40000000) | (uint32_t)((program->program_va >> 40) & 0xff));

   for (unsigned axis = 0; axis < 3; ++axis) {
      agx3_cmdstream_store_le32(packed + 16 + axis * 4, global_size[axis]);
      agx3_cmdstream_store_le32(packed + 28 + axis * 4,
                                grid->local_size[axis]);
   }

   return true;
}

static bool
agx3_native_compute_dispatch_append_common(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   const struct agx3_cdm_barrier_info *barrier,
   struct agx3_compute_dispatch_sink *sink)
{
   if (!grid || !sink || !sink->data || sink->length > sink->capacity ||
       AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE > sink->capacity - sink->length)
      return false;

   uint8_t staged[AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE];
   if (!agx3_g17p_native_compute_launch_pack(program, grid, staged) ||
       !agx3_cdm_barrier_pack(barrier, staged + AGX3_CDM_DIRECT_LAUNCH_SIZE))
      return false;

   memcpy(sink->data + sink->length, staged, sizeof(staged));
   sink->length += sizeof(staged);
   return true;
}

bool
agx3_g17p_native_compute_dispatch_append_with_barrier(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   uint8_t barrier_flags,
   struct agx3_compute_dispatch_sink *sink)
{
   if (!program || !agx_chip_native_agx3(program->chip) ||
       !program->g17p_native_compute_live_bound)
      return false;

   const struct agx3_cdm_barrier_info barrier = {
      .chip = program->chip,
      .flags = barrier_flags,
      .wide_visibility = barrier_flags == 0x5 &&
                         agx_chip_cdm_wide_visibility(program->chip),
   };
   return agx3_native_compute_dispatch_append_common(program, grid, &barrier,
                                                     sink);
}

bool
agx3_g17p_native_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink)
{
   return agx3_g17p_native_compute_dispatch_append_with_barrier(
      program, grid, 0, sink);
}

bool
agx3_g17g_native_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid, uint8_t barrier_flags,
   struct agx3_compute_dispatch_sink *sink)
{
   if (!program || !agx_chip_m5_paths(program->chip) ||
       !program->g17g_native_compute_live_bound)
      return false;

   const struct agx3_cdm_barrier_info barrier = {
      .chip = program->chip,
      .flags = barrier_flags,
   };
   return agx3_native_compute_dispatch_append_common(program, grid, &barrier,
                                                     sink);
}

bool
agx3_m5_native_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid, uint8_t barrier_flags,
   struct agx3_compute_dispatch_sink *sink)
{
   if (!program || !agx_chip_g17_m5(program->chip) ||
       !program->m5_native_compute_live_bound)
      return false;

   const struct agx3_cdm_barrier_info barrier = {
      .chip = program->chip,
      .flags = barrier_flags,
      .flush_memory = barrier_flags == 5 ||
                      (barrier_flags == 0 && program->ordering_memory_scope),
   };
   return agx3_native_compute_dispatch_append_common(program, grid, &barrier,
                                                     sink);
}

static void
agx3_g17p_store_native_program_pointer(uint8_t *packed, uint64_t program_va)
{
   agx3_cmdstream_store_le32(packed, (uint32_t)(program_va >> 6));
   /* Same pointer descriptor as the direct path: `low = va >> 6`,
    * `control = 0x40000000 | va[47:40]` (m1n1 `encode_shader_pointer`).  A
    * hard-coded 0x40000001 here injects 2^40 into every helper/indirect
    * program fetch whose VA has bit 40 clear -- i.e. every address in Mesa's
    * USC window -- which is the same defect the direct path took. */
   agx3_cmdstream_store_le32(
      packed + 4,
      UINT32_C(0x40000000) | (uint32_t)((program_va >> 40) & 0xff));
}

/* An indirect launch adds only bit 27 or 28 to the direct template, on both
 * G18G and M5. These templates are never 1D-masked, and the grid address
 * goes in words 4..5 of both the nine-word and the six-word forms. Words 1..3
 * are the direct record's program pointer. */
static bool
agx3_hal300_native_indirect_append(const struct agx3_compute_program *program,
                                   uint32_t mode, uint64_t grid_va,
                                   const uint32_t *local_size,
                                   uint8_t barrier_flags, bool flush_memory,
                                   struct agx3_compute_dispatch_sink *sink)
{
   const size_t launch = local_size
                            ? AGX3_G17G_NATIVE_INDIRECT_GROUPS_LAUNCH_SIZE
                            : AGX3_G17G_NATIVE_INDIRECT_LOCAL_LAUNCH_SIZE;
   uint8_t staged[AGX3_G17G_NATIVE_INDIRECT_GROUPS_LAUNCH_SIZE +
                  AGX3_CDM_BARRIER_SIZE] = {0};
   if (!program->program_va ||
       program->program_va > UINT64_C(0xffffffffffff) ||
       (program->program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       !grid_va || grid_va >= (UINT64_C(1) << 48) || (grid_va & 3) != 0 ||
       program->launch_template.word0 >= (UINT32_C(1) << 27) ||
       program->launch_template.word1 >= (UINT32_C(1) << 22) || !sink ||
       !sink->data || sink->length > sink->capacity ||
       launch + AGX3_CDM_BARRIER_SIZE > sink->capacity - sink->length)
      return false;

   agx3_cmdstream_store_le32(staged, program->launch_template.word0 | mode);
   agx3_cmdstream_store_le32(
      staged + 4, (uint32_t)(((program->program_va >> 38) & 0x3ff) << 22) |
                     program->launch_template.word1);
   agx3_g17p_store_native_program_pointer(staged + 8, program->program_va);
   agx3_cmdstream_store_le32(staged + 16, (uint32_t)(grid_va >> 32));
   agx3_cmdstream_store_le32(staged + 20, (uint32_t)grid_va);
   for (unsigned axis = 0; local_size && axis < 3; ++axis) {
      if (!local_size[axis] || local_size[axis] > 0x7ff)
         return false;
      agx3_cmdstream_store_le32(staged + 24 + axis * 4, local_size[axis]);
   }

   const struct agx3_cdm_barrier_info barrier = {
      .chip = program->chip,
      .flags = barrier_flags,
      .flush_memory = flush_memory,
   };
   if (!agx3_cdm_barrier_pack(&barrier, staged + launch))
      return false;
   memcpy(sink->data + sink->length, staged, launch + AGX3_CDM_BARRIER_SIZE);
   sink->length += launch + AGX3_CDM_BARRIER_SIZE;
   return true;
}

bool
agx3_g17g_native_indirect_compute_launch_append(
   const struct agx3_compute_program *program, uint64_t grid_va,
   uint8_t barrier_flags, struct agx3_compute_dispatch_sink *sink)
{
   if (!program || !agx_chip_m5_paths(program->chip) ||
       !program->g17g_native_compute_live_bound)
      return false;

   return agx3_hal300_native_indirect_append(
      program, AGX3_G17G_NATIVE_INDIRECT_LOCAL_MODE, grid_va, NULL,
      barrier_flags, false, sink);
}

bool
agx3_g17g_native_indirect_groups_compute_launch_append(
   const struct agx3_compute_program *program, uint64_t groups_va,
   const uint32_t local_size[3], uint8_t barrier_flags,
   struct agx3_compute_dispatch_sink *sink)
{
   if (!program || !local_size || !agx_chip_m5_paths(program->chip) ||
       !program->g17g_native_compute_live_bound)
      return false;

   return agx3_hal300_native_indirect_append(
      program, AGX3_G17G_NATIVE_INDIRECT_GROUPS_MODE, groups_va, local_size,
      barrier_flags, false, sink);
}

/* M5 indirect groups retain the direct program header and append local
 * sizes after the grid address. Indirect-local reads both counts and local
 * sizes from memory instead. */
bool
agx3_m5_native_indirect_dispatch_append(
   const struct agx3_compute_program *program, uint64_t grid_va,
   const uint32_t local_size[3], uint8_t barrier_flags,
   struct agx3_compute_dispatch_sink *sink)
{
   if (!program || !agx_chip_g17_m5(program->chip) ||
       !program->m5_native_compute_live_bound)
      return false;

   return agx3_hal300_native_indirect_append(
      program,
      local_size ? AGX3_G17G_NATIVE_INDIRECT_GROUPS_MODE
                 : AGX3_G17G_NATIVE_INDIRECT_LOCAL_MODE,
      grid_va, local_size, barrier_flags,
      barrier_flags == 5 ||
         (barrier_flags == 0 && program->ordering_memory_scope),
      sink);
}

/* The six-word indirect launch.  Words 0..3 are the same program header the
 * direct record uses; mode 2 then replaces the direct record's own three
 * global and three local size words with words 4..5, the 48-bit address of a
 * six-u32 grid the GPU reads for itself (three global THREAD counts followed
 * by three local sizes).  That is why the record is 24 bytes where the direct
 * one is 40: on G17P the launch is exactly these six words, and the
 * command stream advances by 0x18. */
static bool
agx3_g17p_native_indirect_launch_pack(
   const struct agx3_compute_program *program, uint64_t geometry_va,
   uint8_t packed[AGX3_G17P_NATIVE_INDIRECT_LAUNCH_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_G17P_NATIVE_INDIRECT_LAUNCH_SIZE);
   if (!program || !agx_chip_native_indirect_compute(program->chip) ||
       !program->g17p_native_compute_live_bound || program->program_va == 0 ||
       program->program_va > UINT64_C(0xffffffffffff) ||
       (program->program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       !geometry_va || geometry_va >= (UINT64_C(1) << 48) ||
       (geometry_va & 3) != 0 ||
       program->launch_template.word0 >= (UINT32_C(1) << 27) ||
       program->launch_template.word1 >= (UINT32_C(1) << 22))
      return false;

   agx3_cmdstream_store_le32(
      packed, program->launch_template.word0 | AGX3_G17P_NATIVE_INDIRECT_MODE);
   agx3_cmdstream_store_le32(
      packed + 4,
      (uint32_t)(((program->program_va >> 38) & 0x3ff) << 22) |
         program->launch_template.word1);
   agx3_g17p_store_native_program_pointer(packed + 8, program->program_va);
   agx3_cmdstream_store_le32(packed + 16, (uint32_t)(geometry_va >> 32));
   agx3_cmdstream_store_le32(packed + 20, (uint32_t)geometry_va);
   return true;
}

bool
agx3_g17p_native_indirect_compute_launch_append(
   const struct agx3_compute_program *program, uint64_t geometry_va,
   uint8_t barrier_flags, struct agx3_compute_dispatch_sink *sink)
{
   if (!sink || (sink->capacity != 0 && !sink->data) ||
       sink->length > sink->capacity ||
       AGX3_G17P_NATIVE_INDIRECT_MAIN_SIZE > sink->capacity - sink->length)
      return false;

   uint8_t staged[AGX3_G17P_NATIVE_INDIRECT_MAIN_SIZE];
   if (!agx3_g17p_native_indirect_launch_pack(program, geometry_va, staged))
      return false;

   /* A strong tail is a cache flush, as for the direct launch. */
   const struct agx3_cdm_barrier_info barrier = {
      .chip = program->chip,
      .flags = barrier_flags,
      .wide_visibility = barrier_flags == 0x5 &&
                         agx_chip_cdm_wide_visibility(program->chip),
   };
   if (!agx3_cdm_barrier_pack(&barrier,
                              staged + AGX3_G17P_NATIVE_INDIRECT_LAUNCH_SIZE))
      return false;

   memcpy(sink->data + sink->length, staged, sizeof(staged));
   sink->length += sizeof(staged);
   return true;
}

bool
agx3_g17p_native_indirect_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_g17p_native_indirect_dispatch *dispatch,
   struct agx3_compute_dispatch_sink *sink)
{
   if (!program || !dispatch || !dispatch->helper_program || !sink ||
       (sink->capacity != 0 && !sink->data) || sink->length > sink->capacity ||
       agx3_native_indirect_compute_dispatch_size(program->chip) >
          sink->capacity - sink->length ||
       !agx_chip_native_indirect_compute(program->chip) ||
       !program->g17p_native_compute_live_bound || program->program_va == 0 ||
       program->program_va > UINT64_C(0xffffffffffff) ||
       (program->program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       !dispatch->geometry_va || dispatch->geometry_va >= (UINT64_C(1) << 48) ||
       (dispatch->geometry_va & 3) != 0 ||
       program->launch_template.word0 >= (UINT32_C(1) << 27) ||
       dispatch->helper_program->launch_template.word0 >=
          (UINT32_C(1) << 27) ||
       program->launch_template.word1 >= (UINT32_C(1) << 22))
      return false;

   uint8_t staged[AGX3_NATIVE_INDIRECT_COMPUTE_DISPATCH_MAX_SIZE] = {0};
   struct agx3_compute_dispatch_sink helper_sink = {
      .data = staged,
      .capacity = sizeof(staged),
   };
   const struct agx3_compute_direct_grid helper_grid = {
      .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
      .size = {1, 1, 1},
      .local_size = {1, 1, 1},
      .dimensions = 1,
   };
   /* The 44-byte record is a direct dispatch, not a deferred application
    * launch. Bit 23 belongs to the program template; it is not indirect mode.
    * Run the one-invocation geometry producer with its own resource template. */
   if (!agx3_g17p_native_compute_dispatch_append(
          dispatch->helper_program, &helper_grid, &helper_sink))
      return false;

   /* Where the CDM's geometry fetch is not ordered after the helper's stores
    * by the launch tail alone, flush between the two records the way a
    * strong dependency inside the stream is published. */
   if (agx_chip_indirect_geometry_flush(program->chip)) {
      const struct agx3_cdm_barrier_info flush = {
         .chip = program->chip,
         .flags = 0x5,
         .wide_visibility = agx_chip_cdm_wide_visibility(program->chip),
      };
      if (!agx3_cdm_barrier_pack(&flush, staged + helper_sink.length))
         return false;
      helper_sink.length += AGX3_CDM_BARRIER_SIZE;
   }

   /* Exactly the standalone record, appended after the helper: there is one
    * encoder for the six words so the coupled and uncoupled forms cannot
    * drift apart. */
   if (!agx3_g17p_native_indirect_compute_launch_append(
          program, dispatch->geometry_va, 0, &helper_sink))
      return false;

   assert(helper_sink.length ==
          agx3_native_indirect_compute_dispatch_size(program->chip));
   memcpy(sink->data + sink->length, staged, helper_sink.length);
   sink->length += helper_sink.length;
   return true;
}

bool
agx3_g17s_minimal_compute_program_prepare(
   const struct agx3_g17s_minimal_compute_program_desc *desc, void *output,
   size_t capacity, size_t *written, struct agx3_compute_program *program)
{
   if (!desc || !output || !written || !program)
      return false;

   /* G17S live witness: a uniform load from a user buffer.  Exact arguments
    * are (w1..w6) = (4, 1, 0, 0x20000001, 0, 0).  The binding tag is
    * constructed from user-buffer index one; keep only caller-visible/raw
    * UniformData axes here and fail closed on every other layout. */
   if (desc->root_uniform.user_buffer_binding != 1 ||
       desc->root_uniform.byte_offset != 0 ||
       desc->root_uniform.uniform_data_destination != 4 ||
       desc->root_uniform.uniform_data_count != 1 ||
       desc->root_uniform.temporary_base != 0)
      return false;

   struct agx3_cdm_direct_template launch_template;
   if (!agx3_cdm_direct_template_build_from_sources(&desc->launch_sources,
                                                    &launch_template))
      return false;

   size_t staged_written;
   if (!agx_ioto_g17s_minimal_compute_program_compose(
          desc->program_va, desc->shader_va, output, capacity,
          &staged_written))
      return false;

   *program = (struct agx3_compute_program){
      .chip = AGX_CHIP_G17S,
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .launch_template = launch_template,
   };
   *written = staged_written;
   return true;
}

/* Narrow production condition for the independently witnessed G17S minimal
 * root-uniform program.  This does not imply chip-wide Ioto load readiness. */
bool
agx3_g17s_minimal_compute_dispatch_ready(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17S;
}

bool
agx3_g17s_minimal_compute_dispatch_append(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   struct agx3_compute_dispatch_sink *sink)
{
   if (!program || !grid || !sink ||
       (sink->capacity != 0 && !sink->data) || sink->length > sink->capacity ||
       AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE >
          sink->capacity - sink->length ||
       !agx3_g17s_minimal_compute_dispatch_ready(program->chip) ||
       !program->g17s_minimal_live_admitted)
      return false;

   uint8_t staged[AGX3_G17S_MINIMAL_COMPUTE_DISPATCH_SIZE];
   if (!agx3_compute_program_pack_direct_launch(program, grid, staged))
      return false;

   const struct agx3_cdm_barrier_info barrier = {
      .chip = AGX_CHIP_G17S,
      .flags = 5,
   };
   if (!agx3_cdm_barrier_pack(&barrier,
                              staged + AGX3_CDM_DIRECT_LAUNCH_SIZE))
      return false;

   memcpy(sink->data + sink->length, staged, sizeof(staged));
   sink->length += sizeof(staged);
   return true;
}

bool
agx3_g17s_minimal_compute_program_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program)
{
   if (chip != AGX_CHIP_G17S || !desc || !upload || !upload->cpu ||
       !written || !program ||
       ((uintptr_t)upload->cpu & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       upload->gpu_va != desc->program_va)
      return false;

   return agx3_g17s_minimal_compute_program_prepare(
      desc, upload->cpu, upload->capacity, written, program);
}

bool
agx3_g17s_minimal_compute_program_live_upload_prepare(
   enum agx_chip chip,
   const struct agx3_g17s_minimal_compute_program_desc *desc,
   const struct agx_ioto_g17s_minimal_compute_resources *resources,
   const struct agx3_compute_program_upload *upload, size_t *written,
   struct agx3_compute_program *program)
{
   if (chip != AGX_CHIP_G17S || !desc || !resources || !upload ||
       !upload->cpu || !written || !program ||
       ((uintptr_t)upload->cpu & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       upload->gpu_va != desc->program_va ||
       desc->root_uniform.user_buffer_binding != 1 ||
       desc->root_uniform.byte_offset != 0 ||
       desc->root_uniform.uniform_data_destination != 4 ||
       desc->root_uniform.uniform_data_count != 1 ||
       desc->root_uniform.temporary_base != 0)
      return false;

   struct agx3_cdm_direct_template launch_template;
   if (!agx3_cdm_direct_template_build_from_sources(&desc->launch_sources,
                                                    &launch_template))
      return false;

   size_t staged_written;
   if (!agx_ioto_g17s_minimal_compute_program_compose_resources(
          desc->program_va, desc->shader_va, resources, upload->cpu,
          upload->capacity, &staged_written))
      return false;

   *program = (struct agx3_compute_program){
      .chip = AGX_CHIP_G17S,
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .launch_template = launch_template,
      .g17s_minimal_live_admitted = true,
   };
   *written = staged_written;
   return true;
}

bool
agx3_compute_program_pack_direct_launch(
   const struct agx3_compute_program *program,
   const struct agx3_compute_direct_grid *grid,
   uint8_t packed[AGX3_CDM_DIRECT_LAUNCH_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_CDM_DIRECT_LAUNCH_SIZE);
   if (!program || !grid ||
       !agx3_cdm_direct_launch_target_grounded(program->chip) ||
       program->program_va > UINT64_C(0xffffffffffff) ||
       (program->program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0)
      return false;

   uint32_t global_size[3];
   if (!agx3_cdm_direct_grid_resolve(grid->source, grid->size, grid->local_size,
                                     global_size))
      return false;

   const struct agx3_cdm_direct_launch_info launch = {
      .chip = program->chip,
      .template_words = program->launch_template,
      .pipeline = program->program_va,
      .dimensions = grid->dimensions,
      .g15x_has_multiple_agcs = grid->g15x_has_multiple_agcs,
      .global_size = {global_size[0], global_size[1], global_size[2]},
      .local_size = {grid->local_size[0], grid->local_size[1],
                     grid->local_size[2]},
   };

   return agx3_cdm_direct_launch_pack(&launch, packed);
}

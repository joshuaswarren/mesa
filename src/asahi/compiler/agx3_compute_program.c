/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_compute_program.h"

#include "asahi/lib/agx_ioto_program_core.h"
#include "util/macros.h"

#include <string.h>

static bool
agx3_g17p_compute_state_tail(
   const struct agx3_g17p_compute_program_desc *desc,
   struct agx_ioto_state_tail_output *tail)
{
   /* SPECTPR carries the thread-private spill frame in quads. The returned
    * device atomic's 16-byte TPS word is one such frame; register spills
    * publish theirs the same way, as Apple's kernel metadata does. */
   if (!desc || !tail || !desc->shader_binary_size || !desc->nr_gprs ||
       desc->scratch_size > AGX3_G17P_MAX_COMPUTE_SPILL_B ||
       (desc->scratch_size & 3))
      return false;

   struct agx_ioto_state_tail_output staged;
   const bool tail_ok = desc->threadgroup_memory_size_B
      ? agx_ioto_g17p_runtime_compute_tail_build_shared(
           desc->shader_va, desc->ldshdr_f1,
           desc->threadgroup_memory_size_B, &staged)
      : agx_ioto_g17p_runtime_compute_tail_build(desc->shader_va,
                                                 desc->ldshdr_f1, &staged);
   if (!tail_ok)
      return false;

   /* A kernel with an imageblock describes it in SPECLM as a fragment program
    * describes its tile: without it the kernel's imageblock stores are lost
    * and its block stores write zeros. */
   if (desc->imageblock_stride_B) {
      if (desc->threadgroup_memory_size_B)
         return false;
      const struct agx_ioto_g17p_imageblock imageblock = {
         .sample_size_B = (uint8_t)ALIGN_POT(desc->imageblock_stride_B, 8),
         .nr_samples = 1,
         .tile_size_px = 32 * 32,
      };
      if (!agx_ioto_g17p_fragment_speclm_build(&imageblock, staged.bytes,
                                               AGX_IOTO_SPECLM_SIZE))
         return false;
   }

   if (!desc->compiler_d_valid ||
       !agx_ioto_g17p_compiler_spectpr_pack(
          desc->scratch_size, desc->compiler_d,
          staged.bytes + AGX_IOTO_SPECLM_SIZE + AGX_IOTO_LDSHDR_SIZE,
          AGX_IOTO_SPECTPR_SIZE))
      return false;

   *tail = staged;
   return true;
}

bool
agx3_g17p_compute_program_generate(
   enum agx_chip chip, const struct agx3_g17p_compute_program_desc *desc,
   struct agx3_g17p_compute_program *program)
{
   if (!program || !agx_chip_native_agx3(chip) || !desc ||
       (desc->uniform_range_count && !desc->uniform_ranges) ||
       (desc->constant_program.code_size & 1) ||
       (desc->cprog_inputs && desc->cprog_inputs->count &&
        (!desc->constant_program.code_size || !desc->uniform_range_count)))
      return false;

   struct agx_ioto_state_tail_output tail;
   if (!agx3_g17p_compute_state_tail(desc, &tail))
      return false;

   /* G18G composes its compute preamble into the loader (preshader); the
    * other chips keep the trusted no-preamble layouts and never take one. */
   const bool g18g = chip == AGX_CHIP_G18G;
   if (!g18g && desc->preshader)
      return false;

   struct agx3_g17p_compute_program staged = {0};
   size_t written = 0;
   if (desc->relocation_count)
      *desc->relocation_count = 0;
   if (!desc->uniform_range_count) {
      if (!(g18g ? agx_ioto_g17p_program_compose_with_preshader(
                      desc->program_va, NULL, 0, NULL, 0, &tail,
                      AGX_IOTO_G17P_PROGRAM_VERTEX_COMPUTE, 0, desc->preshader,
                      staged.bytes, sizeof(staged.bytes), &written)
                 : agx_ioto_g17p_no_resource_program_compose_trusted(
                      desc->program_va, &tail, staged.bytes,
                      sizeof(staged.bytes), &written)))
         return false;
   } else {
      uint8_t body[AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE];
      struct agx_ioto_inflight_load loads[AGX3_G17P_COMPUTE_MAX_LOAD_CHUNKS];
      struct agx_ioto_inflight_load sort[AGX3_G17P_COMPUTE_MAX_LOAD_CHUNKS];
      struct agx_ioto_load_cache_entry cache[AGX3_G17P_COMPUTE_MAX_LOAD_CHUNKS];
      size_t body_bound, chunk_count;
      if (!agx3_ioto_uniform_program_body_bound(
             chip, desc->uniform_ranges, desc->uniform_range_count,
             &body_bound, &chunk_count) ||
          body_bound > sizeof(body) || chunk_count > ARRAY_SIZE(loads))
         return false;

      const struct agx3_ioto_uniform_program_desc load_desc = {
         .program_va = desc->program_va,
         .ranges = desc->uniform_ranges,
         .range_count = desc->uniform_range_count,
         .cprog_inputs = desc->cprog_inputs,
         .patchable = desc->patchable,
      };
      const struct agx3_ioto_uniform_program_workspace workspace = {
         .body = body,
         .body_capacity = sizeof(body),
         .loads = loads,
         .load_capacity = ARRAY_SIZE(loads),
         .sort = sort,
         .sort_capacity = ARRAY_SIZE(sort),
         .cache = cache,
         .cache_capacity = ARRAY_SIZE(cache),
         .relocations = desc->relocations,
         .relocation_capacity = desc->relocation_capacity,
         .relocation_count = desc->relocation_count,
      };
      size_t body_size, copy_offset;
      if (!agx3_ioto_uniform_program_body_generate(
             chip, &load_desc, workspace, &body_size, &copy_offset) ||
          copy_offset > body_size ||
          !(g18g ? agx_ioto_g17p_program_compose_with_preshader(
                      desc->program_va, body, copy_offset, body + copy_offset,
                      body_size - copy_offset, &tail,
                      AGX_IOTO_G17P_PROGRAM_VERTEX_COMPUTE, 0, desc->preshader,
                      staged.bytes, sizeof(staged.bytes), &written)
                 : agx_ioto_g17p_resource_program_compose_trusted(
                      desc->program_va, body, copy_offset, body + copy_offset,
                      body_size - copy_offset, &tail, staged.bytes,
                      sizeof(staged.bytes), &written)))
         return false;

      if (desc->relocation_count) {
         for (size_t r = 0; r < *desc->relocation_count; ++r) {
            const struct agx_ioto_load_relocation *rel = &desc->relocations[r];
            if (rel->offset + rel->low_length + rel->high_length > copy_offset)
               return false;
         }
      }
   }

   /* Every loader ends in the absent constant-program END. A constant
    * program replaces exactly that suffix, as in the graphics loaders. */
   if (desc->constant_program.code_size) {
      static const uint8_t absent[] = {0x0e, 0, 0, 0};
      const size_t offset = written - sizeof(absent);
      struct agx_ioto_preshader_output suffix;
      if (written < sizeof(absent) ||
          memcmp(staged.bytes + offset, absent, sizeof(absent)) ||
          !agx_ioto_constant_program_emit(chip, &desc->constant_program,
                                          desc->program_va + offset,
                                          &suffix) ||
          suffix.length > sizeof(staged.bytes) - offset)
         return false;
      memcpy(staged.bytes + offset, suffix.bytes, suffix.length);
      written = offset + suffix.length;
   }

   if (written > UINT16_MAX)
      return false;

   const struct agx3_g17p_native_compute_program_desc launch_desc = {
      .chip = chip,
      .program_va = desc->program_va,
      .shader_va = desc->shader_va,
      .launch_sources = desc->launch_sources,
   };
   if (!agx3_g17p_native_compute_program_bind(&launch_desc, &staged.launch))
      return false;

   staged.length = written;
   *program = staged;
   return true;
}

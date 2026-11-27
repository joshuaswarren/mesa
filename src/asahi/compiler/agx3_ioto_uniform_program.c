/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_ioto_uniform_program.h"

#include <limits.h>
#include <string.h>

#define GPU0_CHUNK_BODY_BOUND 78u
#define GPU1_CHUNK_BODY_BOUND 64u

static bool
uniform_range_valid(const struct agx3_ioto_uniform_range *range)
{
   if (range == NULL || range->size_halfs == 0 ||
       (range->start_halfs & 1) != 0 || (range->size_halfs & 1) != 0 ||
       range->start_halfs > 512 ||
       range->size_halfs > 512 - range->start_halfs ||
       (range->address & 3) != 0)
      return false;

   const uint64_t bytes = (uint64_t)(range->size_halfs / 2) * 4;
   return bytes != 0 && range->address <= UINT64_MAX - (bytes - 1);
}

static bool
add_size(size_t *value, size_t add)
{
   if (*value > SIZE_MAX - add)
      return false;
   *value += add;
   return true;
}

bool
agx3_ioto_uniform_program_body_bound_with_inputs(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, const struct agx_ioto_cprog_inputs *inputs,
   size_t *bytes, size_t *chunks)
{
   if (bytes == NULL || chunks == NULL ||
       agx_ioto_load_target_encoding(chip) ==
          AGX_IOTO_LOAD_ENCODING_UNSUPPORTED ||
       (range_count != 0 && ranges == NULL) ||
       (inputs != NULL && (!agx_chip_native_hal200(chip) || !inputs->known)))
      return false;

   size_t staged_bytes =
      range_count == 0 &&
            agx_ioto_load_target_encoding(chip) == AGX_IOTO_LOAD_ENCODING_GPU0
         ? AGX_IOTO_LOAD_GPU0_ROUND_NOP_SIZE
         : 0;
   size_t staged_chunks = 0;
   uint64_t covered[AGX_IOTO_CPROG_UNIFORM_WORDS] = {0};
   const size_t per_chunk =
      agx_ioto_load_target_encoding(chip) == AGX_IOTO_LOAD_ENCODING_GPU0
         ? GPU0_CHUNK_BODY_BOUND
         : GPU1_CHUNK_BODY_BOUND;

   for (size_t i = 0; i < range_count; ++i) {
      if (!uniform_range_valid(&ranges[i]))
         return false;

      const size_t range_chunks = (ranges[i].size_halfs + 7u) / 8u;
      if (!add_size(&staged_chunks, range_chunks) ||
          range_chunks > SIZE_MAX / per_chunk ||
          !add_size(&staged_bytes, range_chunks * per_chunk))
         return false;

      if (inputs != NULL) {
         for (unsigned half = ranges[i].start_halfs;
              half < ranges[i].start_halfs + ranges[i].size_halfs; ++half) {
            const uint64_t bit = UINT64_C(1) << (half % 64);
            if (!(inputs->uniform_halfs[half / 64] & bit))
               continue;
            if ((covered[half / 64] & bit) ||
                !add_size(&staged_bytes, AGX_IOTO_LOAD_GPU0_COPY_MAX_SIZE))
               return false;
            covered[half / 64] |= bit;
         }
      }
   }

   if (inputs != NULL) {
      for (unsigned i = 0; i < AGX_IOTO_CPROG_UNIFORM_WORDS; ++i) {
         if (covered[i] != inputs->uniform_halfs[i])
            return false;
      }
   }

   *bytes = staged_bytes;
   *chunks = staged_chunks;
   return true;
}

bool
agx3_ioto_uniform_program_desc_body_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_program_desc *desc,
   size_t *bytes, size_t *chunks)
{
   size_t bound, count;
   if (!desc || !bytes || !chunks ||
       (desc->preshader_inputs && (desc->cprog_inputs || desc->fast_path_enabled)) ||
       !agx3_ioto_uniform_program_body_bound_with_inputs(chip, desc->ranges,
          desc->range_count, desc->preshader_inputs, &bound, &count))
      return false;
   if (desc->cprog_inputs) {
      struct agx3_cprog_inputs checked;
      const struct agx3_cprog_inputs *inputs = desc->cprog_inputs;
      if (!agx3_cprog_inputs_init(inputs->halves, &checked) ||
          checked.count != inputs->count ||
          checked.storage_bytes != inputs->storage_bytes ||
          (inputs->count &&
           ((!agx_chip_native_preamble(chip) && !agx_chip_g17_m5(chip)) ||
            desc->fast_path_enabled)))
         return false;
      uint64_t covered[AGX3_CPROG_INPUT_WORDS] = {0};
      size_t copies = 0;
      for (size_t i = 0; i < desc->range_count; ++i) {
         const struct agx3_ioto_uniform_range *range = &desc->ranges[i];
         /* Ranges are dword aligned (uniform_range_valid) and the loader
          * publishes per loaded dword (pack_cprog_copy): one 32-bit COPY for
          * a selected pair at an even compact position, else one 16-bit
          * COPY per selected half. */
         for (unsigned h = range->start_halfs;
              h < range->start_halfs + range->size_halfs; h += 2) {
            unsigned mask = 0;
            for (unsigned j = 0; j < 2; ++j) {
               const unsigned half = h + j;
               const uint64_t bit = UINT64_C(1) << (half % 64);
               if (!(inputs->halves[half / 64] & bit))
                  continue;
               if (covered[half / 64] & bit)
                  return false;
               covered[half / 64] |= bit;
               mask |= 1u << j;
            }
            if (!mask)
               continue;
            uint16_t tps;
            if (!agx3_cprog_input_tps_half(&checked, (mask & 1) ? h : h + 1,
                                           &tps))
               return false;
            copies += (mask == 3 && !(tps & 1)) ? 1 : __builtin_popcount(mask);
         }
      }
      for (unsigned w = 0; w < AGX3_CPROG_INPUT_WORDS; ++w) {
         if (covered[w] != inputs->halves[w])
            return false;
      }
      /* An extended COPY is at most 10 bytes (8 on M5). Bounding each
       * selected half by its own COPY instead roughly doubled this term and
       * refused M5 compute loaders that read a few hundred input halves
       * (e.g. ubo.*_compute) although they compose to about 4 KiB. */
      if (!add_size(&bound, 10u * copies))
         return false;
   }
   *bytes = bound;
   *chunks = count;
   return true;
}

bool
agx3_ioto_uniform_program_body_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *bytes, size_t *chunks)
{
   return agx3_ioto_uniform_program_body_bound_with_inputs(
      chip, ranges, range_count, NULL, bytes, chunks);
}

static bool
spans_overlap(const void *a, size_t a_size, const void *b, size_t b_size)
{
   if (a_size == 0 || b_size == 0)
      return false;
   if (a == NULL || b == NULL)
      return true;

   const uintptr_t a0 = (uintptr_t)a, b0 = (uintptr_t)b;
   if (a0 > UINTPTR_MAX - a_size || b0 > UINTPTR_MAX - b_size)
      return true;
   return a0 < b0 + b_size && b0 < a0 + a_size;
}

static bool
workspace_overlaps(struct agx3_ioto_uniform_program_workspace workspace)
{
   if (workspace.load_capacity >
          SIZE_MAX / sizeof(struct agx_ioto_inflight_load) ||
       workspace.sort_capacity >
          SIZE_MAX / sizeof(struct agx_ioto_inflight_load) ||
       workspace.cache_capacity >
          SIZE_MAX / sizeof(struct agx_ioto_load_cache_entry))
      return true;

   const size_t loads_size =
      workspace.load_capacity * sizeof(struct agx_ioto_inflight_load);
   const size_t sort_size =
      workspace.sort_capacity * sizeof(struct agx_ioto_inflight_load);
   const size_t cache_size =
      workspace.cache_capacity * sizeof(struct agx_ioto_load_cache_entry);

   return spans_overlap(workspace.body, workspace.body_capacity,
                        workspace.loads, loads_size) ||
          spans_overlap(workspace.body, workspace.body_capacity, workspace.sort,
                        sort_size) ||
          spans_overlap(workspace.body, workspace.body_capacity,
                        workspace.cache, cache_size) ||
          spans_overlap(workspace.loads, loads_size, workspace.sort,
                        sort_size) ||
          spans_overlap(workspace.loads, loads_size, workspace.cache,
                        cache_size) ||
          spans_overlap(workspace.sort, sort_size, workspace.cache, cache_size);
}

bool
agx3_ioto_uniform_program_body_generate(
   enum agx_chip chip, const struct agx3_ioto_uniform_program_desc *desc,
   struct agx3_ioto_uniform_program_workspace workspace, size_t *written,
   size_t *copy_offset)
{
   size_t body_bound, chunk_count;
   if (desc == NULL || written == NULL || copy_offset == NULL ||
       !agx3_ioto_uniform_program_desc_body_bound(
          chip, desc, &body_bound, &chunk_count) ||
       (body_bound > workspace.body_capacity &&
        !(desc->cprog_inputs && desc->cprog_inputs->count)) ||
       (workspace.body_capacity != 0 && workspace.body == NULL) ||
       workspace_overlaps(workspace) ||
       (workspace.relocation_capacity && !workspace.relocations) ||
       (workspace.relocations && !workspace.relocation_count))
      return false;

   (void)chunk_count;

   struct agx_ioto_load_core core;
   if (!agx_ioto_load_core_init(
          &core, chip,
          (struct agx_ioto_load_sink){workspace.body, workspace.body_capacity,
                                      0},
          workspace.loads, workspace.load_capacity, workspace.cache,
          workspace.cache_capacity, desc->initial_gpr_dwords,
          desc->fast_path_enabled,
          agx_ioto_load_target_encoding(chip) == AGX_IOTO_LOAD_ENCODING_GPU0))
      return false;

   core.relocations = workspace.relocations;
   core.relocation_capacity = workspace.relocation_capacity;
   core.long_ldimm = desc->patchable;
   if (desc->preloaded_base_valid &&
       !agx_ioto_load_core_set_fixed_base(
          &core, desc->preloaded_base_address,
          desc->preloaded_base_half))
      return false;

   const struct agx_ioto_load_workspace sort = {
      workspace.sort,
      workspace.sort_capacity,
   };
   struct agx3_ioto_load_packer packer;
   struct agx_ioto_load_emit_ops ops;
   if (desc->preshader_inputs != NULL) {
      if (!agx3_ioto_load_packer_init(&packer, chip))
         return false;
      packer.cprog_inputs = desc->preshader_inputs;
      agx3_ioto_load_packer_ops(&packer, &ops);
   }
   uint32_t cprog_dest_half = 0;

   for (size_t i = 0; i < desc->range_count; ++i) {
      const struct agx3_ioto_uniform_range *range = &desc->ranges[i];
      uint32_t dword = 0;
      if (desc->patchable)
         agx_ioto_load_core_forget_bases(&core);
      const uint32_t dword_count = range->size_halfs / 2;
      while (dword < dword_count) {
         const uint8_t count =
            (uint8_t)((dword_count - dword) < 4 ? dword_count - dword : 4);
         const uint8_t lane_mask = (uint8_t)((1u << (2u * count)) - 1u);
         if (cprog_dest_half > UINT16_MAX)
            return false;

         struct agx_ioto_load_absolute load = {
            .native_slot = range->start_halfs / 2 + dword,
            .count_dwords = count,
            .address = range->address + (uint64_t)dword * 4,
            .lane_mask = lane_mask,
            .cprog_dest_half = (uint16_t)cprog_dest_half,
         };
         if (desc->cprog_inputs) {
            for (unsigned h = 0; h < 2u * count; ++h) {
               uint16_t tps;
               if (!agx3_cprog_input_tps_half(desc->cprog_inputs,
                          range->start_halfs + 2u * dword + h, &tps))
                  continue;
               if (!load.cprog_input_mask)
                  load.cprog_input_tps_half = tps;
               load.cprog_input_mask |= 1u << h;
            }
         }
         struct agx_ioto_load_allocation allocation;
         const bool loaded = desc->preshader_inputs != NULL
            ? agx_ioto_load_absolute(&core, &load, &ops, sort, &allocation)
            : agx3_ioto_load_absolute_native(&core, &load, sort, &allocation);
         if (!loaded)
            return false;

         cprog_dest_half += 2u * (2u * count);
         dword += count;
      }
   }

   const bool finished = desc->preshader_inputs != NULL
      ? agx_ioto_load_finish_round(&core, true, &ops, sort)
      : agx3_ioto_load_finish_round_native(&core, true, sort);
   if (!finished)
      return false;

   *written = core.sink.length;
   *copy_offset = core.copy_suffix_offset;
   if (workspace.relocations)
      *workspace.relocation_count = core.relocation_count;
   return true;
}

static bool
array_overlaps(const void *array, size_t count, size_t element_size,
               const void *output, size_t output_size)
{
   return count > SIZE_MAX / element_size ||
          spans_overlap(array, count * element_size, output, output_size);
}

bool
agx3_ioto_uniform_program_generate(
   enum agx_chip chip, const struct agx3_ioto_uniform_program_desc *desc,
   struct agx3_ioto_uniform_program_workspace workspace, void *output,
   size_t capacity, size_t *written)
{
   size_t body_bound, chunk_count;
   if (desc == NULL || output == NULL || written == NULL ||
       !agx3_ioto_uniform_program_desc_body_bound(
          chip, desc, &body_bound, &chunk_count) ||
       (body_bound > workspace.body_capacity &&
        !(desc->cprog_inputs && desc->cprog_inputs->count)) ||
       (workspace.body_capacity != 0 && workspace.body == NULL) ||
       workspace_overlaps(workspace) ||
       spans_overlap(workspace.body, workspace.body_capacity, output,
                     capacity) ||
       array_overlaps(workspace.loads, workspace.load_capacity,
                      sizeof(workspace.loads[0]), output, capacity) ||
       array_overlaps(workspace.sort, workspace.sort_capacity,
                      sizeof(workspace.sort[0]), output, capacity) ||
       array_overlaps(workspace.cache, workspace.cache_capacity,
                      sizeof(workspace.cache[0]), output, capacity))
      return false;

   (void)body_bound;
   (void)chunk_count;
   size_t body_size, copy_offset;
   if (!agx3_ioto_uniform_program_body_generate(chip, desc, workspace,
                                                &body_size, &copy_offset))
      return false;
   (void)copy_offset;

   const struct agx_ioto_profile_control profile = {
      .chip = chip,
      .payload0 = desc->profile_payload0,
      .payload1 = desc->profile_payload1,
   };
   return agx_ioto_program_compose_no_pixrel(
      chip, desc->program_va, workspace.body, body_size, &desc->tail, &profile,
      &desc->preshader, output, capacity, written);
}

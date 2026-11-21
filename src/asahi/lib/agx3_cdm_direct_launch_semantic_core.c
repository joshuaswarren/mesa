/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_cdm_direct_launch_semantic_core.h"

#include "asahi/genxml/agx3_cmdstream.h"

#include <string.h>

/* The G15G, G15X, G16G, G16X, G17P, G17G and G17X families all use this
 * direct 40-byte launch record with the AGX3 compute encoding.  G16P and
 * G17A have no corresponding direct witness and deliberately remain
 * closed. */
bool
agx3_cdm_direct_launch_target_grounded(enum agx_chip chip)
{
   return agx3_cdm_launch_target_grounded(chip);
}

bool
agx3_cdm_direct_template_build(
   const struct agx3_cdm_direct_template_fields *fields,
   struct agx3_cdm_direct_template *result)
{
   if (result)
      memset(result, 0, sizeof(*result));

   if (!fields || !result || fields->word0_bits_3_14 > 0xfff ||
       (fields->word0_bits_3_14 & 1) != 0 ||
       fields->word1_bits_3_20 > 0x3ffff)
      return false;

   /* The template has the same bit layout on the G15/HAL200 and HAL300
    * sides of the command-stream boundary: the same values occupy these
    * exact template ranges on both. */
   result->word0 = ((uint32_t)fields->word0_bits_3_14 << 3) |
                   ((uint32_t)fields->ray_traversal_depth_nonzero << 23) |
                   ((uint32_t)fields->subkernel_x_target_workitems << 19);
   result->word1 = fields->word1_bits_3_20 << 3;
   return true;
}

bool
agx3_cdm_direct_template_sources_compute(
   bool uses_resource_info_query, bool uses_ray_traversal,
   bool ray_traversal_depth_nonzero,
   struct agx3_cdm_direct_template_sources *sources)
{
   if (!sources || (ray_traversal_depth_nonzero && !uses_ray_traversal))
      return false;

   uint32_t ti_spill;
   if (uses_resource_info_query && uses_ray_traversal)
      ti_spill = 0x40;
   else if (uses_ray_traversal)
      ti_spill = 0x30;
   else if (uses_resource_info_query)
      ti_spill = 0x20;
   else
      ti_spill = 0;

   *sources = (struct agx3_cdm_direct_template_sources){
      .ti_spill_bytes = ti_spill,
      .ti_largest_ipr_bytes = 0,
      .ray_traversal_depth_nonzero = ray_traversal_depth_nonzero,
   };
   return true;
}

bool
agx3_cdm_direct_template_build_from_sources(
   const struct agx3_cdm_direct_template_sources *sources,
   struct agx3_cdm_direct_template *result)
{
   if (result)
      memset(result, 0, sizeof(*result));

   if (!sources || !result)
      return false;

   /* The direct template rounds the TI spill bytes to quads in
    * launch word 1 and the TI IPR bytes to an even quad count in word 0, as
    * the vertex and fragment variants do for their stage state. */
   const uint64_t word1_value = ((uint64_t)sources->ti_spill_bytes + 3) >> 2;
   const uint64_t word0_rounded =
      ((uint64_t)sources->ti_largest_ipr_bytes + 3) >> 2;
   const uint64_t word0_value = (word0_rounded + 1) & ~UINT64_C(1);

   if (word0_value > 0xfff || word1_value > 0x3ffff)
      return false;

   const struct agx3_cdm_direct_template_fields fields = {
      .word0_bits_3_14 = (uint16_t)word0_value,
      .word1_bits_3_20 = (uint32_t)word1_value,
      .ray_traversal_depth_nonzero = sources->ray_traversal_depth_nonzero,

      /* On HAL200 and HAL300 the sub-kernel X target-workitems bit starts
       * at one in every direct template.  The launch may then clear it for
       * a one-dimensional grid. */
      .subkernel_x_target_workitems = true,
   };
   return agx3_cdm_direct_template_build(&fields, result);
}

bool
agx3_cdm_direct_template_decode(
   const struct agx3_cdm_direct_template *template_words,
   struct agx3_cdm_direct_template_fields *fields)
{
   if (fields)
      memset(fields, 0, sizeof(*fields));

   if (!template_words || !fields ||
       (template_words->word0 & ~AGX3_CDM_DIRECT_TEMPLATE_WORD0_MASK) != 0 ||
       (template_words->word1 & ~AGX3_CDM_DIRECT_TEMPLATE_WORD1_MASK) != 0)
      return false;

   fields->word0_bits_3_14 = (template_words->word0 >> 3) & 0xfff;
   fields->word1_bits_3_20 = (template_words->word1 >> 3) & 0x3ffff;
   fields->ray_traversal_depth_nonzero =
      (template_words->word0 >> 23) & 1;
   fields->subkernel_x_target_workitems =
      (template_words->word0 >> 19) & 1;
   return true;
}

static bool
agx3_cdm_direct_dimension_word(uint8_t dimensions, uint32_t *word)
{
   if (!word)
      return false;

   switch (dimensions) {
   case 1:
      *word = UINT32_C(0x28);
      return true;
   case 2:
      *word = UINT32_C(0x30);
      return true;
   case 3:
      *word = UINT32_C(0x34);
      return true;
   default:
      *word = 0;
      return false;
   }
}

static bool
agx3_cdm_direct_g15x_bit31_allowed(enum agx_chip chip)
{
   return agx_chip_command_stream_generation(chip) == AGX_CS_GENERATION_G15;
}

bool
agx3_cdm_direct_grid_resolve(enum agx3_cdm_direct_grid_source source,
                             const uint32_t input_size[3],
                             const uint32_t local_size[3],
                             uint32_t global_size[3])
{
   if (!global_size)
      return false;

   if (!input_size || !local_size) {
      memset(global_size, 0, 3 * sizeof(*global_size));
      return false;
   }

   /* Preserve both inputs before touching output so exact and partial aliasing
    * remain transactional. */
   uint32_t input[3], local[3], resolved[3] = {0};
   memcpy(input, input_size, sizeof(input));
   memcpy(local, local_size, sizeof(local));

   for (unsigned axis = 0; axis < 3; ++axis) {
      if (local[axis] > 0x7ff)
         goto fail;

      switch (source) {
      case AGX3_CDM_DIRECT_GRID_THREADGROUP_COUNTS: {
         const uint64_t product =
            (uint64_t)input[axis] * local[axis];
         if (product > UINT32_MAX)
            goto fail;
         resolved[axis] = (uint32_t)product;
         break;
      }
      case AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS:
         resolved[axis] = input[axis];
         break;
      default:
         goto fail;
      }
   }

   memcpy(global_size, resolved, sizeof(resolved));
   return true;

fail:
   memset(global_size, 0, 3 * sizeof(*global_size));
   return false;
}

static bool
agx3_cdm_direct_word3_build(enum agx_chip chip, uint8_t dimensions,
                            bool g15x_has_multiple_agcs,
                            uint32_t *word3)
{
   uint32_t value = 0;
   if (!agx3_cdm_direct_dimension_word(dimensions, &value))
      return false;

   /* Both direct-thread and direct-threadgroup entry points set bit 30 and
    * emit this same 40-byte record.  The 24-byte record selected when the
    * enqueue call carries an address is an indirect form and is not modeled
    * by this API. */
   value |= UINT32_C(1) << 30;
   if (g15x_has_multiple_agcs) {
      if (!agx3_cdm_direct_g15x_bit31_allowed(chip))
         return false;
      value |= UINT32_C(1) << 31;
   }

   *word3 = value;
   return true;
}

bool
agx3_cdm_direct_launch_pack(
   const struct agx3_cdm_direct_launch_info *info,
   uint8_t packed[AGX3_CDM_DIRECT_LAUNCH_SIZE])
{
   if (!packed)
      return false;

   memset(packed, 0, AGX3_CDM_DIRECT_LAUNCH_SIZE);
   if (!info || !agx3_cdm_direct_launch_target_grounded(info->chip))
      return false;

   struct agx3_cdm_direct_template_fields fields;
   if (!agx3_cdm_direct_template_decode(&info->template_words, &fields))
      return false;

   uint32_t word3 = 0;
   if (!agx3_cdm_direct_word3_build(info->chip, info->dimensions,
                                    info->g15x_has_multiple_agcs, &word3))
      return false;

   /* A launch clears template bits 17..19 when both higher grid
    * axes contain fewer than two threads.  Bits 17 and 18 are already outside
    * the proven template mask, leaving bit 19. */
   uint32_t emitted_word0 = info->template_words.word0;
   if (info->global_size[1] < 2 && info->global_size[2] < 2)
      emitted_word0 &= UINT32_C(0xfff1ffff);

   const struct agx3_cdm_launch_info raw = {
      .chip = info->chip,
      .word0_payload = emitted_word0,
      .word1_payload = info->template_words.word1,
      .pipeline = info->pipeline,
      .opaque_word3 = word3,
      .global_size = {info->global_size[0], info->global_size[1],
                      info->global_size[2]},
      .local_size = {info->local_size[0], info->local_size[1],
                     info->local_size[2]},
   };

   return agx3_cdm_launch_pack(&raw, packed);
}

bool
agx3_cdm_direct_launch_unpack(
   enum agx_chip chip, const void *packed, size_t available,
   struct agx3_cdm_direct_launch_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));

   if (!packed || !info ||
       !agx3_cdm_direct_launch_target_grounded(chip))
      return false;

   struct agx3_cdm_launch_info raw;
   if (!agx3_cdm_launch_unpack(chip, packed, available, &raw))
      return false;

   const struct agx3_cdm_direct_template template_words = {
      .word0 = raw.word0_payload,
      .word1 = raw.word1_payload,
   };
   struct agx3_cdm_direct_template_fields fields;
   if (!agx3_cdm_direct_template_decode(&template_words, &fields))
      return false;

   if ((raw.opaque_word3 & (UINT32_C(1) << 30)) == 0)
      return false;

   const bool g15x_has_multiple_agcs = (raw.opaque_word3 >> 31) & 1;
   const uint32_t dimension_word =
      raw.opaque_word3 & ~(UINT32_C(3) << 30);

   uint8_t dimensions = 0;
   switch (dimension_word) {
   case UINT32_C(0x28):
      dimensions = 1;
      break;
   case UINT32_C(0x30):
      dimensions = 2;
      break;
   case UINT32_C(0x34):
      dimensions = 3;
      break;
   default:
      return false;
   }

   uint32_t canonical_word3 = 0;
   if (!agx3_cdm_direct_word3_build(chip, dimensions,
                                    g15x_has_multiple_agcs,
                                    &canonical_word3) ||
       canonical_word3 != raw.opaque_word3 ||
       (raw.global_size[1] < 2 && raw.global_size[2] < 2 &&
        fields.subkernel_x_target_workitems))
      return false;

   *info = (struct agx3_cdm_direct_launch_info){
      .chip = chip,
      .template_words = template_words,
      .pipeline = raw.pipeline,
      .dimensions = dimensions,
      .g15x_has_multiple_agcs = g15x_has_multiple_agcs,
      .global_size = {raw.global_size[0], raw.global_size[1],
                      raw.global_size[2]},
      .local_size = {raw.local_size[0], raw.local_size[1],
                     raw.local_size[2]},
   };
   return true;
}

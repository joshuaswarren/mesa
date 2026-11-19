/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx3_cprog_input.h"
#include "agx3_static_spill_core.h"

#include <string.h>

bool
agx3_cprog_inputs_init(const uint64_t halves[AGX3_CPROG_INPUT_WORDS],
                      struct agx3_cprog_inputs *out)
{
   if (!halves || !out)
      return false;
   struct agx3_cprog_inputs result = {0};
   memcpy(result.halves, halves, sizeof(result.halves));
   for (unsigned i = 0; i < AGX3_CPROG_INPUT_WORDS; ++i)
      result.count += __builtin_popcountll(halves[i]);
   /* The 16-byte input-frame alignment is witnessed in native metadata.
    * Leave an absent import frame absent. */
   if (result.count)
      result.storage_bytes =
         ((AGX3_CPROG_INPUT_TPS_BASE + result.count) * 2u + 15u) & ~15u;
   *out = result;
   return true;
}

static bool
valid_inputs(const struct agx3_cprog_inputs *inputs)
{
   if (!inputs)
      return false;
   struct agx3_cprog_inputs expected;
   return agx3_cprog_inputs_init(inputs->halves, &expected) &&
          inputs->count == expected.count &&
          inputs->storage_bytes == expected.storage_bytes;
}

bool
agx3_cprog_input_tps_half(const struct agx3_cprog_inputs *inputs,
                         unsigned uniform_half, uint16_t *tps_half)
{
   if (!inputs || !tps_half || uniform_half >= AGX3_CPROG_INPUT_HALVES)
      return false;
   const unsigned word = uniform_half / 64;
   const uint64_t bit = UINT64_C(1) << (uniform_half % 64);
   if (!(inputs->halves[word] & bit))
      return false;
   unsigned rank = __builtin_popcountll(inputs->halves[word] & (bit - 1));
   for (unsigned i = 0; i < word; ++i)
      rank += __builtin_popcountll(inputs->halves[i]);
   *tps_half = AGX3_CPROG_INPUT_TPS_BASE + rank;
   return true;
}

bool
agx3_cprog_input_publish(const struct agx3_cprog_inputs *inputs,
                         unsigned uniform_half, unsigned source_half,
                         void *output, size_t capacity, size_t *written)
{
   if (!valid_inputs(inputs) || !output || !written ||
       uniform_half >= AGX3_CPROG_INPUT_HALVES - 1 || (uniform_half & 1) ||
       source_half >= 255 || (source_half & 1))
      return false;
   uint16_t destinations[2];
   bool selected[2];
   for (unsigned i = 0; i < 2; ++i)
      selected[i] = agx3_cprog_input_tps_half(
         inputs, uniform_half + i, &destinations[i]);

   unsigned mask = selected[0] | (selected[1] << 1);
   return agx3_cprog_input_publish_at(mask,
      selected[0] ? destinations[0] : selected[1] ? destinations[1] : 8,
      source_half, output, capacity, written);
}

bool
agx3_cprog_input_publish_at_encoding(
   enum agx_isa_encoding encoding, unsigned mask, unsigned tps_half,
   unsigned source_half, void *output, size_t capacity, size_t *written)
{
   if (!output || !written || mask > 3 || source_half >= 255 ||
       (source_half & 1) || tps_half < 8 || tps_half >= 520 ||
       (mask == 3 && tps_half == 519))
      return false;
   uint8_t bytes[AGX3_CPROG_INPUT_COPY_MAX_SIZE];
   size_t size = 0;
   for (unsigned i = 0; i < 2; ++i) {
      if (!(mask & (1u << i)))
         continue;
      const bool pair = i == 0 && mask == 3 && !(tps_half & 1);
      const struct agx3_static_spill_store_desc desc = {
         .tps_half = tps_half++,
         .src = source_half + i,
         .width = pair ? AGX3_MEM_W32 : AGX3_MEM_W16,
         .slot = 6,
         .src_kill = true,
      };
      uint8_t length;
      /* The native loader uses extended COPY for both widths, including
       * the otherwise compact-encodable 16-bit form. */
      if (!agx3_static_spill_encode_store_form(
             encoding, &desc, 1, bytes + size, &length))
         return false;
      size += length;
      if (pair)
         ++i;
   }
   if (size > capacity)
      return false;
   memcpy(output, bytes, size);
   *written = size;
   return true;
}

bool
agx3_cprog_input_import_encoding(
   enum agx_isa_encoding encoding, const struct agx3_cprog_inputs *inputs,
   const struct agx3_cprog_import *desc, void *output, size_t capacity,
   size_t *written)
{
   if (!valid_inputs(inputs) || !desc || !output || !written ||
       (desc->width != AGX3_MEM_W16 && desc->width != AGX3_MEM_W32))
      return false;
   uint16_t tps_half;
   if (!agx3_cprog_input_tps_half(inputs, desc->uniform_half, &tps_half))
      return false;
   if (desc->width == AGX3_MEM_W32) {
      uint16_t second;
      if (!agx3_cprog_input_tps_half(inputs, desc->uniform_half + 1, &second) ||
          second != tps_half + 1 || (tps_half & 1))
         return false;
   }
   const struct agx3_static_spill_load_desc load = {
      .dest = desc->dest_half, .tps_half = tps_half, .width = desc->width,
      .slot = desc->slot, .slot_mirror = desc->slot_mirror,
      .wait_mask = desc->wait_mask, .tps_live = desc->tps_live,
      .structural_flag = desc->structural_flag,
   };
   uint8_t bytes[AGX3_STATIC_SPILL_MAX_LENGTH], length;
   if (!agx3_static_spill_encode_load(encoding, &load, bytes, &length) ||
       length > capacity)
      return false;
   memcpy(output, bytes, length);
   *written = length;
   return true;
}

bool
agx3_cprog_input_publish_at(unsigned mask, unsigned tps_half,
                            unsigned source_half, void *output,
                            size_t capacity, size_t *written)
{
   return agx3_cprog_input_publish_at_encoding(AGX_ISA_ENCODING_AGX3B,
      mask, tps_half, source_half, output, capacity, written);
}

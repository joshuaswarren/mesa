/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_index_list_core.h"
#include "agx_bytes.h"

#include <string.h>

/*
 * Direct/indirect VDM Index List pack/unpack for G15, HAL200, and HAL300.
 * HAL300 sets header bit 27 and appends an opaque word (7). Indexed records
 * split the 48-bit index-buffer VA across header[7:0], buffer-lo, and a
 * zero-extended VA[47:40] word; only U16 and U32 index sizes are accepted.
 */

bool
agx_vdm_index_list_direct_nonindexed_pack(
   enum agx_chip chip,
   const struct agx_vdm_index_list_direct_nonindexed_info *info,
   struct agx_vdm_index_list_direct_nonindexed *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info)
      return false;

   struct agx_vdm_index_list_direct_words words;
   if (!agx_vdm_index_list_direct_pack_words(
          chip, info->primitive, false, false, 0, 0, info->count,
          info->instances, info->start, 0, &words))
      return false;

   for (unsigned i = 0; i < words.count; ++i)
      agx_store_le32(&out->bytes[i * 4], words.words[i]);
   out->length = words.count * 4;

   return true;
}

bool
agx_vdm_index_list_direct_nonindexed_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_index_list_direct_nonindexed_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));

   if (!bytes || !info || !agx_vdm_index_list_direct_target_supported_inline(chip))
      return false;

   const bool hal300 = agx_vdm_index_list_direct_is_hal300_inline(chip);
   const size_t expected_size = hal300 ? 20 : 16;
   if (size != expected_size)
      return false;

   const uint8_t *raw = bytes;
   const uint32_t header = agx_load_le32(&raw[0]);
   const uint32_t expected_header =
      hal300 ? UINT32_C(0x69c40000) : UINT32_C(0x61c40000);
   const uint8_t primitive = (uint8_t)(header >> 8);

   if ((header & UINT32_C(0xffff00ff)) != expected_header ||
       !agx_vdm_index_list_direct_primitive_supported_inline(primitive) ||
       agx_load_le32(&raw[8]) == 0 ||
       (hal300 && agx_load_le32(&raw[16]) != UINT32_C(7)))
      return false;

   info->primitive = primitive;
   info->count = agx_load_le32(&raw[4]);
   info->instances = agx_load_le32(&raw[8]);
   info->start = agx_load_le32(&raw[12]);
   return true;
}

bool
agx_vdm_index_list_direct_indexed_pack(
   enum agx_chip chip,
   const struct agx_vdm_index_list_direct_indexed_info *info,
   struct agx_vdm_index_list_direct_indexed *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info)
      return false;

   struct agx_vdm_index_list_direct_words words;
   if (!agx_vdm_index_list_direct_pack_words(
          chip, info->primitive, true, info->restart_enable,
          (uint8_t)info->index_size, info->index_buffer, info->count,
          info->instances, info->start, info->index_buffer_size_word, &words))
      return false;

   for (unsigned i = 0; i < words.count; ++i)
      agx_store_le32(&out->bytes[i * 4], words.words[i]);
   out->length = words.count * 4;
   return true;
}

bool
agx_vdm_index_list_direct_indexed_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_index_list_direct_indexed_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));

   if (!bytes || !info || !agx_vdm_index_list_direct_target_supported_inline(chip))
      return false;

   const bool hal300 = agx_vdm_index_list_direct_is_hal300_inline(chip);
   const size_t expected_size = hal300 ? 32 : 28;
   if (size != expected_size)
      return false;

   const uint8_t *raw = bytes;
   const uint32_t header = agx_load_le32(&raw[0]);
   const uint32_t expected_fixed =
      hal300 ? UINT32_C(0x69f00000) : UINT32_C(0x61f00000);
   const uint8_t primitive = (uint8_t)(header >> 8);
   const enum agx_vdm_index_list_index_size index_size =
      (enum agx_vdm_index_list_index_size)((header >> 17) & 0x7);
   const unsigned opaque_offset = 20;
   const unsigned size_offset = hal300 ? 24 : 20;
   const unsigned high_offset = hal300 ? 28 : 24;
   const uint32_t high_word = agx_load_le32(&raw[high_offset]);

   if ((header & UINT32_C(0xfff00000)) != expected_fixed ||
       !agx_vdm_index_list_direct_primitive_supported_inline(primitive) ||
       !agx_vdm_index_list_index_size_supported_inline(chip, index_size) ||
       agx_load_le32(&raw[12]) == 0 || high_word > 0xff ||
       (hal300 &&
        agx_load_le32(&raw[opaque_offset]) != UINT32_C(7)))
      return false;

   info->primitive = primitive;
   info->restart_enable = (header & (UINT32_C(1) << 16)) != 0;
   info->index_size = index_size;
   info->index_buffer =
      (uint64_t)agx_load_le32(&raw[4]) |
      ((uint64_t)(header & 0xff) << 32) | ((uint64_t)high_word << 40);
   info->count = agx_load_le32(&raw[8]);
   info->instances = agx_load_le32(&raw[12]);
   info->start = agx_load_le32(&raw[16]);
   info->index_buffer_size_word =
      agx_load_le32(&raw[size_offset]);
   return true;
}

bool
agx_vdm_index_list_indirect_nonindexed_pack(
   enum agx_chip chip,
   const struct agx_vdm_index_list_indirect_nonindexed_info *info,
   struct agx_vdm_index_list_indirect_nonindexed *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info)
      return false;

   struct agx_vdm_index_list_indirect_words words;
   if (!agx_vdm_index_list_indirect_pack_words(
          chip, info->primitive, false, false, 0, 0, info->indirect_buffer, 0,
          0, &words))
      return false;

   for (unsigned i = 0; i < words.count; ++i)
      agx_store_le32(&out->bytes[i * 4], words.words[i]);
   out->length = words.count * 4;
   return true;
}

bool
agx_vdm_index_list_indirect_nonindexed_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_index_list_indirect_nonindexed_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));

   if (!bytes || !info || !agx_vdm_index_list_direct_target_supported_inline(chip))
      return false;

   const bool hal300 = agx_vdm_index_list_direct_is_hal300_inline(chip);
   const size_t expected_size = hal300 ? 16 : 12;
   if (size != expected_size)
      return false;

   const uint8_t *raw = bytes;
   const uint32_t header = agx_load_le32(&raw[0]);
   const uint32_t expected_header =
      hal300 ? UINT32_C(0x6c040000) : UINT32_C(0x64040000);
   const uint8_t primitive = (uint8_t)(header >> 8);
   const uint32_t addr_hi = agx_load_le32(&raw[4]);
   const uint32_t addr_lo = agx_load_le32(&raw[8]);

   if ((header & UINT32_C(0xffff00ff)) != expected_header ||
       !agx_vdm_index_list_direct_primitive_supported_inline(primitive) ||
       addr_hi > 0xffff || (addr_lo & 3u) != 0 ||
       (hal300 &&
        agx_load_le32(&raw[12]) != UINT32_C(7)))
      return false;

   info->primitive = primitive;
   info->indirect_buffer =
      (uint64_t)addr_lo | ((uint64_t)addr_hi << 32);
   return true;
}

bool
agx_vdm_index_list_indirect_indexed_pack(
   enum agx_chip chip,
   const struct agx_vdm_index_list_indirect_indexed_info *info,
   struct agx_vdm_index_list_indirect_indexed *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !info)
      return false;

   struct agx_vdm_index_list_indirect_words words;
   if (!agx_vdm_index_list_indirect_pack_words(
          chip, info->primitive, true, info->restart_enable,
          (uint8_t)info->index_size, info->index_buffer, info->indirect_buffer,
          info->restart_index, info->index_buffer_size_word, &words))
      return false;

   for (unsigned i = 0; i < words.count; ++i)
      agx_store_le32(&out->bytes[i * 4], words.words[i]);
   out->length = words.count * 4;
   return true;
}

bool
agx_vdm_index_list_indirect_indexed_unpack(
   enum agx_chip chip, const void *bytes, size_t size,
   struct agx_vdm_index_list_indirect_indexed_info *info)
{
   if (info)
      memset(info, 0, sizeof(*info));

   if (!bytes || !info || !agx_vdm_index_list_direct_target_supported_inline(chip))
      return false;

   const bool hal300 = agx_vdm_index_list_direct_is_hal300_inline(chip);
   const size_t expected_size = hal300 ? 36 : 32;
   if (size != expected_size)
      return false;

   const uint8_t *raw = bytes;
   if (agx_load_le32(&raw[0]) != UINT32_C(0x40000001))
      return false;

   const uint32_t restart_index = agx_load_le32(&raw[4]);
   const uint32_t header = agx_load_le32(&raw[8]);
   const uint8_t primitive = (uint8_t)(header >> 8);
   const enum agx_vdm_index_list_index_size index_size =
      (enum agx_vdm_index_list_index_size)((header >> 17) & 0x7);
   const bool restart_enable = (header & (UINT32_C(1) << 16)) != 0;

   if (!agx_vdm_index_list_direct_primitive_supported_inline(primitive) ||
       !agx_vdm_index_list_index_size_supported_inline(chip, index_size))
      return false;

   const uint32_t index_buffer_lo = agx_load_le32(&raw[12]);
   const uint32_t addr_hi = agx_load_le32(&raw[16]);
   const uint32_t addr_lo = agx_load_le32(&raw[20]);
   const unsigned unk2_offset = 24;
   const unsigned size_offset = hal300 ? 28 : 24;
   const unsigned high_offset = hal300 ? 32 : 28;
   const uint32_t index_buffer_hi =
      agx_load_le32(&raw[high_offset]);
   const uint64_t index_buffer =
      (uint64_t)index_buffer_lo | ((uint64_t)(header & 0xff) << 32) |
      ((uint64_t)index_buffer_hi << 40);

   /* Rebuild the header from the recovered fields and require an exact match;
    * this rejects any stray fixed bit without enumerating each one. */
   const uint32_t rebuilt_header =
      (uint32_t)(3u << 29) | (hal300 ? (1u << 27) : 0u) | (1u << 26) |
      (1u << 21) | (1u << 20) | ((uint32_t)(index_size & 7u) << 17) |
      (restart_enable ? (1u << 16) : 0u) | ((uint32_t)primitive << 8) |
      (uint32_t)((index_buffer >> 32) & 0xffu);

   if (header != rebuilt_header || addr_hi > 0xffff || (addr_lo & 3u) != 0 ||
       index_buffer_hi > 0xff ||
       (hal300 &&
        agx_load_le32(&raw[unk2_offset]) != UINT32_C(7)))
      return false;

   info->primitive = primitive;
   info->restart_enable = restart_enable;
   info->index_size = index_size;
   info->index_buffer = index_buffer;
   info->indirect_buffer = (uint64_t)addr_lo | ((uint64_t)addr_hi << 32);
   info->restart_index = restart_index;
   info->index_buffer_size_word =
      agx_load_le32(&raw[size_offset]);
   return true;
}

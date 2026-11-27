/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi/lib/agx_vdm_barrier_core.h"
#include "asahi/lib/agx_vdm_decode_core.h"
#include "asahi/lib/agx_vdm_ppp_state_core.h"
#include "asahi/lib/agx_vdm_stream_link_core.h"
#include "asahi/lib/agx_vdm_stream_terminate_core.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(condition)                                                       \
   do {                                                                        \
      ++checks;                                                                \
      if (!(condition)) {                                                      \
         fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__,    \
                 #condition);                                                  \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static const enum agx_chip supported_targets[] = {
   AGX_CHIP_G15G, AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D,
   AGX_CHIP_G16G, AGX_CHIP_G16S, AGX_CHIP_G16C, AGX_CHIP_G17P,
   AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C, AGX_CHIP_G17D,
};

static const enum agx_chip unsupported_targets[] = {
   AGX_CHIP_G13G, AGX_CHIP_G13X, AGX_CHIP_G14G, AGX_CHIP_G14X,
   AGX_CHIP_G15P, AGX_CHIP_G16P, AGX_CHIP_G17A, AGX_CHIP_UNKNOWN,
};

static bool
is_hal300(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
          chip == AGX_CHIP_G17C ||
          chip == AGX_CHIP_G17D;
}

static bool
record_is_cleared(const struct agx_vdm_decoded_record *record)
{
   const uint8_t *bytes = (const uint8_t *)record;
   for (size_t i = 0; i < sizeof(*record); ++i) {
      if (bytes[i] != 0)
         return false;
   }
   return true;
}

static int
check_rejected(enum agx_chip chip, const void *bytes, size_t available)
{
   struct agx_vdm_decoded_record record;
   memset(&record, 0xff, sizeof(record));
   CHECK(!agx_vdm_decode_one(chip, bytes, available, &record));
   CHECK(record_is_cleared(&record));
   return 0;
}

static int
test_ppp_state_update(void)
{
   for (unsigned t = 0;
        t < sizeof(supported_targets) / sizeof(supported_targets[0]); ++t) {
      const enum agx_chip chip = supported_targets[t];

      for (uint32_t size_words = 0; size_words <= 0xff; ++size_words) {
         const uint64_t pointer =
            ((UINT64_C(0x81) + t) << 32) | (UINT64_C(0x12345000) + t * 4);
         struct agx_vdm_ppp_state_core packed;
         CHECK(agx_vdm_ppp_state_core_pack(chip, pointer, size_words, &packed));

         uint8_t stream[AGX_VDM_PPP_STATE_CORE_LENGTH + 4];
         memcpy(stream, packed.bytes, packed.length);
         memset(stream + packed.length, 0xa5, 4);

         struct agx_vdm_decoded_record record;
         CHECK(agx_vdm_decode_one(chip, stream, sizeof(stream), &record));
         CHECK(record.kind == AGX_VDM_RECORD_PPP_STATE_UPDATE);
         CHECK(record.length == packed.length);
         CHECK(record.chip == chip);
         CHECK(record.data.ppp_state_update.pointer == pointer);
         CHECK(record.data.ppp_state_update.size_words == size_words);
      }
   }

   return 0;
}

static int
test_barrier(void)
{
   for (unsigned t = 0;
        t < sizeof(supported_targets) / sizeof(supported_targets[0]); ++t) {
      const enum agx_chip chip = supported_targets[t];
      struct agx_vdm_barrier_core packed;
      CHECK(agx_vdm_barrier_core_pack(chip, false, &packed));

      uint8_t stream[AGX_VDM_BARRIER_CORE_LENGTH + 4];
      memcpy(stream, packed.bytes, packed.length);
      memset(stream + packed.length, 0x5a, 4);

      struct agx_vdm_decoded_record record;
      CHECK(agx_vdm_decode_one(chip, stream, sizeof(stream), &record));
      CHECK(record.kind == AGX_VDM_RECORD_BARRIER);
      CHECK(record.length == packed.length);
      CHECK(record.chip == chip);
   }

   return 0;
}

static int
test_state_update(void)
{
   for (unsigned t = 0;
        t < sizeof(supported_targets) / sizeof(supported_targets[0]); ++t) {
      const enum agx_chip chip = supported_targets[t];
      const struct agx_vdm_state_update_info info = {
         .raw_vertex_shader_word0 = UINT32_C(0x10203040) + t,
         .pipeline = UINT64_C(0xa123456780) + t * 64,
         .raw_pipeline_template = UINT32_C(0x002abcde) ^ t,
         .raw_vertex_outputs = {UINT32_C(0x55667788) + t,
                                UINT32_C(0x99aabbcc) - t},
         .raw_vertex_unknown = UINT32_C(0xddeeff00) ^ t,
      };
      struct agx_vdm_state_update packed;
      CHECK(agx_vdm_state_update_pack(chip, &info, &packed));

      uint8_t stream[AGX_VDM_STATE_UPDATE_LENGTH + 8];
      memcpy(stream, packed.bytes, packed.length);
      memset(stream + packed.length, 0x5a, 8);

      struct agx_vdm_decoded_record record;
      CHECK(agx_vdm_decode_one(chip, stream, sizeof(stream), &record));
      CHECK(record.kind == AGX_VDM_RECORD_STATE_UPDATE);
      CHECK(record.length == packed.length);
      CHECK(record.chip == chip);
      CHECK(record.data.state_update.raw_vertex_shader_word0 ==
            info.raw_vertex_shader_word0);
      CHECK(record.data.state_update.pipeline == info.pipeline);
      CHECK(record.data.state_update.raw_pipeline_template ==
            info.raw_pipeline_template);
      CHECK(memcmp(record.data.state_update.raw_vertex_outputs,
                   info.raw_vertex_outputs,
                   sizeof(info.raw_vertex_outputs)) == 0);
      CHECK(record.data.state_update.raw_vertex_unknown ==
            info.raw_vertex_unknown);
   }

   return 0;
}

static int
test_index_lists(void)
{
   static const uint8_t primitives[] = {0, 1, 3, 5, 6, 9, 11, 13, 14, 15};

   for (unsigned t = 0;
        t < sizeof(supported_targets) / sizeof(supported_targets[0]); ++t) {
      const enum agx_chip chip = supported_targets[t];

      for (unsigned p = 0; p < sizeof(primitives); ++p) {
         const struct agx_vdm_index_list_direct_nonindexed_info nonindexed = {
            .primitive = primitives[p],
            .count = UINT32_C(0x11223344) + p,
            .instances = UINT32_C(0x102) + t,
            .start = UINT32_C(0x55667788) ^ p,
         };
         struct agx_vdm_index_list_direct_nonindexed packed_nonindexed;
         CHECK(agx_vdm_index_list_direct_nonindexed_pack(chip, &nonindexed,
                                                         &packed_nonindexed));

         uint8_t
            nonindexed_stream[AGX_VDM_INDEX_LIST_DIRECT_NONINDEXED_MAX_LENGTH +
                              32];
         memcpy(nonindexed_stream, packed_nonindexed.bytes,
                packed_nonindexed.length);
         memset(nonindexed_stream + packed_nonindexed.length, 0xa5, 32);

         struct agx_vdm_decoded_record record;
         CHECK(agx_vdm_decode_one(chip, nonindexed_stream,
                                  sizeof(nonindexed_stream), &record));
         CHECK(record.kind == AGX_VDM_RECORD_INDEX_LIST_NONINDEXED);
         CHECK(record.length == packed_nonindexed.length);
         CHECK(record.chip == chip);
         CHECK(record.data.index_list_nonindexed.primitive ==
               nonindexed.primitive);
         CHECK(record.data.index_list_nonindexed.count == nonindexed.count);
         CHECK(record.data.index_list_nonindexed.instances ==
               nonindexed.instances);
         CHECK(record.data.index_list_nonindexed.start == nonindexed.start);

         for (unsigned size = AGX_VDM_INDEX_LIST_INDEX_SIZE_U16;
              size <= AGX_VDM_INDEX_LIST_INDEX_SIZE_U32; ++size) {
            const struct agx_vdm_index_list_direct_indexed_info indexed = {
               .primitive = primitives[p],
               .restart_enable = (p & 1) != 0,
               .index_size = (enum agx_vdm_index_list_index_size)size,
               .index_buffer = UINT64_C(0xab1234567800) + t * 4,
               .count = UINT32_C(0x23456789) + p,
               .instances = UINT32_C(0x203) + t,
               .start = UINT32_C(0x3456789a) ^ p,
               .index_buffer_size_word = UINT32_C(0x456789ab) + p,
            };
            struct agx_vdm_index_list_direct_indexed packed_indexed;
            CHECK(agx_vdm_index_list_direct_indexed_pack(chip, &indexed,
                                                         &packed_indexed));

            uint8_t
               indexed_stream[AGX_VDM_INDEX_LIST_DIRECT_INDEXED_MAX_LENGTH + 8];
            memcpy(indexed_stream, packed_indexed.bytes, packed_indexed.length);
            memset(indexed_stream + packed_indexed.length, 0x5a, 8);

            CHECK(agx_vdm_decode_one(chip, indexed_stream,
                                     sizeof(indexed_stream), &record));
            CHECK(record.kind == AGX_VDM_RECORD_INDEX_LIST_INDEXED);
            CHECK(record.length == packed_indexed.length);
            CHECK(record.chip == chip);
            CHECK(record.data.index_list_indexed.primitive ==
                  indexed.primitive);
            CHECK(record.data.index_list_indexed.restart_enable ==
                  indexed.restart_enable);
            CHECK(record.data.index_list_indexed.index_size ==
                  indexed.index_size);
            CHECK(record.data.index_list_indexed.index_buffer ==
                  indexed.index_buffer);
            CHECK(record.data.index_list_indexed.count == indexed.count);
            CHECK(record.data.index_list_indexed.instances ==
                  indexed.instances);
            CHECK(record.data.index_list_indexed.start == indexed.start);
            CHECK(record.data.index_list_indexed.index_buffer_size_word ==
                  indexed.index_buffer_size_word);
         }
      }
   }

   return 0;
}

static int
test_indirect_index_lists(void)
{
   static const uint8_t primitives[] = {0, 1, 3, 5, 6, 9, 11, 13, 14, 15};

   for (unsigned t = 0;
        t < sizeof(supported_targets) / sizeof(supported_targets[0]); ++t) {
      const enum agx_chip chip = supported_targets[t];
      const bool hal300 = is_hal300(chip);

      for (unsigned p = 0; p < sizeof(primitives); ++p) {
         const uint64_t indirect_buffer =
            UINT64_C(0xab1234567800) + ((uint64_t)t << 12) +
            ((uint64_t)p << 2);
         const struct agx_vdm_index_list_indirect_nonindexed_info
            nonindexed = {
               .primitive = primitives[p],
               .indirect_buffer = indirect_buffer,
            };
         struct agx_vdm_index_list_indirect_nonindexed packed_nonindexed;
         CHECK(agx_vdm_index_list_indirect_nonindexed_pack(
            chip, &nonindexed, &packed_nonindexed));
         CHECK(packed_nonindexed.length == (hal300 ? 16 : 12));

         uint8_t nonindexed_stream
            [AGX_VDM_INDEX_LIST_INDIRECT_NONINDEXED_MAX_LENGTH + 8];
         memcpy(nonindexed_stream, packed_nonindexed.bytes,
                packed_nonindexed.length);
         memset(nonindexed_stream + packed_nonindexed.length, 0xa5, 8);

         struct agx_vdm_decoded_record record;
         CHECK(agx_vdm_decode_one(chip, nonindexed_stream,
                                  sizeof(nonindexed_stream), &record));
         CHECK(record.kind ==
               AGX_VDM_RECORD_INDEX_LIST_INDIRECT_NONINDEXED);
         CHECK(record.length == packed_nonindexed.length);
         CHECK(record.chip == chip);
         CHECK(record.data.index_list_indirect_nonindexed.primitive ==
               nonindexed.primitive);
         CHECK(record.data.index_list_indirect_nonindexed.indirect_buffer ==
               nonindexed.indirect_buffer);

         for (size_t available = 0; available < packed_nonindexed.length;
              ++available) {
            CHECK(check_rejected(chip, packed_nonindexed.bytes, available) ==
                  0);
         }

         for (unsigned size = AGX_VDM_INDEX_LIST_INDEX_SIZE_U16;
              size <= AGX_VDM_INDEX_LIST_INDEX_SIZE_U32; ++size) {
            for (unsigned restart = 0; restart < 2; ++restart) {
               const struct agx_vdm_index_list_indirect_indexed_info indexed = {
                  .primitive = primitives[p],
                  .restart_enable = restart != 0,
                  .index_size =
                     (enum agx_vdm_index_list_index_size)size,
                  .index_buffer = UINT64_C(0xcd1234567800) +
                                  ((uint64_t)t << 12) +
                                  ((uint64_t)p << 4),
                  .indirect_buffer = indirect_buffer,
                  .restart_index = UINT32_C(0x10203040) ^
                                   ((uint32_t)t << 8) ^ p ^ restart,
                  .index_buffer_size_word = UINT32_C(0x50607080) + p,
               };
               struct agx_vdm_index_list_indirect_indexed packed_indexed;
               CHECK(agx_vdm_index_list_indirect_indexed_pack(
                  chip, &indexed, &packed_indexed));
               CHECK(packed_indexed.length == (hal300 ? 36 : 32));

               uint8_t indexed_stream
                  [AGX_VDM_INDEX_LIST_INDIRECT_INDEXED_MAX_LENGTH + 8];
               memcpy(indexed_stream, packed_indexed.bytes,
                      packed_indexed.length);
               memset(indexed_stream + packed_indexed.length, 0x5a, 8);

               CHECK(agx_vdm_decode_one(chip, indexed_stream,
                                        sizeof(indexed_stream), &record));
               CHECK(record.kind ==
                     AGX_VDM_RECORD_INDEX_LIST_INDIRECT_INDEXED);
               CHECK(record.length == packed_indexed.length);
               CHECK(record.chip == chip);
               CHECK(record.data.index_list_indirect_indexed.primitive ==
                     indexed.primitive);
               CHECK(record.data.index_list_indirect_indexed.restart_enable ==
                     indexed.restart_enable);
               CHECK(record.data.index_list_indirect_indexed.index_size ==
                     indexed.index_size);
               CHECK(record.data.index_list_indirect_indexed.index_buffer ==
                     indexed.index_buffer);
               CHECK(record.data.index_list_indirect_indexed.indirect_buffer ==
                     indexed.indirect_buffer);
               CHECK(record.data.index_list_indirect_indexed.restart_index ==
                     indexed.restart_index);
               CHECK(record.data.index_list_indirect_indexed
                        .index_buffer_size_word ==
                     indexed.index_buffer_size_word);

               for (size_t available = 0; available < packed_indexed.length;
                    ++available) {
                  CHECK(check_rejected(chip, packed_indexed.bytes, available) ==
                        0);
               }
            }
         }
      }
   }

   const struct agx_vdm_index_list_indirect_nonindexed_info nonindexed_info = {
      .primitive = 9,
      .indirect_buffer = UINT64_C(0x23456789abc0),
   };
   const struct agx_vdm_index_list_indirect_indexed_info indexed_info = {
      .primitive = 9,
      .restart_enable = true,
      .index_size = AGX_VDM_INDEX_LIST_INDEX_SIZE_U32,
      .index_buffer = UINT64_C(0x123456789abc),
      .indirect_buffer = UINT64_C(0x23456789abc0),
      .restart_index = UINT32_MAX,
      .index_buffer_size_word = UINT32_C(0x1020303f),
   };
   struct agx_vdm_index_list_indirect_nonindexed pre_nonindexed;
   struct agx_vdm_index_list_indirect_nonindexed hal_nonindexed;
   struct agx_vdm_index_list_indirect_indexed pre_indexed;
   struct agx_vdm_index_list_indirect_indexed hal_indexed;
   CHECK(agx_vdm_index_list_indirect_nonindexed_pack(
      AGX_CHIP_G17P, &nonindexed_info, &pre_nonindexed));
   CHECK(agx_vdm_index_list_indirect_nonindexed_pack(
      AGX_CHIP_G17S, &nonindexed_info, &hal_nonindexed));
   CHECK(agx_vdm_index_list_indirect_indexed_pack(
      AGX_CHIP_G17P, &indexed_info, &pre_indexed));
   CHECK(agx_vdm_index_list_indirect_indexed_pack(
      AGX_CHIP_G17S, &indexed_info, &hal_indexed));

   CHECK(check_rejected(AGX_CHIP_G17S, pre_nonindexed.bytes,
                        pre_nonindexed.length) == 0);
   CHECK(check_rejected(AGX_CHIP_G17P, hal_nonindexed.bytes,
                        hal_nonindexed.length) == 0);
   CHECK(check_rejected(AGX_CHIP_G17S, pre_indexed.bytes,
                        pre_indexed.length) == 0);
   CHECK(check_rejected(AGX_CHIP_G17P, hal_indexed.bytes,
                        hal_indexed.length) == 0);

   struct agx_vdm_index_list_indirect_nonindexed malformed_nonindexed =
      hal_nonindexed;
   malformed_nonindexed.bytes[2] ^= 1;
   CHECK(check_rejected(AGX_CHIP_G17S, malformed_nonindexed.bytes,
                        malformed_nonindexed.length) == 0);
   malformed_nonindexed = hal_nonindexed;
   malformed_nonindexed.bytes[12] = 6;
   CHECK(check_rejected(AGX_CHIP_G17S, malformed_nonindexed.bytes,
                        malformed_nonindexed.length) == 0);

   struct agx_vdm_index_list_indirect_indexed malformed_indexed = hal_indexed;
   malformed_indexed.bytes[11] ^= 0x20;
   CHECK(check_rejected(AGX_CHIP_G17S, malformed_indexed.bytes,
                        malformed_indexed.length) == 0);
   malformed_indexed = hal_indexed;
   malformed_indexed.bytes[24] = 6;
   CHECK(check_rejected(AGX_CHIP_G17S, malformed_indexed.bytes,
                        malformed_indexed.length) == 0);
   return 0;
}

static int
test_stream_link(void)
{
   for (unsigned t = 0;
        t < sizeof(supported_targets) / sizeof(supported_targets[0]); ++t) {
      const enum agx_chip chip = supported_targets[t];
      const uint64_t target = UINT64_C(0xcd12345678) + t;
      struct agx_vdm_stream_link_core packed;
      CHECK(agx_vdm_stream_link_core_pack(chip, target, &packed));

      uint8_t stream[AGX_VDM_STREAM_LINK_CORE_LENGTH + 4];
      memcpy(stream, packed.bytes, packed.length);
      memset(stream + packed.length, 0x5a, 4);

      struct agx_vdm_decoded_record record;
      CHECK(agx_vdm_decode_one(chip, stream, sizeof(stream), &record));
      CHECK(record.kind == AGX_VDM_RECORD_STREAM_LINK);
      CHECK(record.length == packed.length);
      CHECK(record.chip == chip);
      CHECK(record.data.stream_link.target == target);
   }

   return 0;
}

static int
test_tessellate_direct(void)
{
   for (unsigned t = 0;
        t < sizeof(supported_targets) / sizeof(supported_targets[0]); ++t) {
      const enum agx_chip chip = supported_targets[t];
      const struct agx_vdm_tessellate_direct_info info = {
         .factor_buffer = UINT64_C(0xab1234567800) + t,
         .patch_count = UINT32_C(0x10203040) + t,
         .instance_count = UINT32_C(0x50607080) - t,
         .base_patch = UINT32_C(0x90a0b0c0) ^ t,
         .base_instance = UINT32_C(0xd0e0f001) + t,
         .instance_stride_word = UINT32_C(0x70000) + t,
         .factor_buffer_size_minus_one = UINT32_C(0xfedcba98) - t,
      };
      struct agx_vdm_tessellate_direct packed;
      CHECK(agx_vdm_tessellate_direct_pack(chip, &info, &packed));

      uint8_t stream[AGX_VDM_TESSELLATE_DIRECT_LENGTH + 8];
      memcpy(stream, packed.bytes, packed.length);
      memset(stream + packed.length, 0x5a, 8);

      struct agx_vdm_decoded_record record;
      CHECK(agx_vdm_decode_one(chip, stream, sizeof(stream), &record));
      CHECK(record.kind == AGX_VDM_RECORD_TESSELLATE_DIRECT);
      CHECK(record.length == packed.length);
      CHECK(record.chip == chip);
      CHECK(record.data.tessellate_direct.factor_buffer == info.factor_buffer);
      CHECK(record.data.tessellate_direct.patch_count == info.patch_count);
      CHECK(record.data.tessellate_direct.instance_count ==
            info.instance_count);
      CHECK(record.data.tessellate_direct.base_patch == info.base_patch);
      CHECK(record.data.tessellate_direct.base_instance == info.base_instance);
      CHECK(record.data.tessellate_direct.instance_stride_word ==
            info.instance_stride_word);
      CHECK(record.data.tessellate_direct.factor_buffer_size_minus_one ==
            info.factor_buffer_size_minus_one);
   }

   /* The indirect Tessellate record shares block type 5. */
   const struct agx_vdm_tessellate_indirect_info indirect = {
      .factor_buffer = UINT64_C(0x10000018300),
      .indirect_buffer = UINT64_C(0x1000001cf00),
      .factor_buffer_size_minus_one = 5,
   };
   struct agx_vdm_tessellate_indirect packed;
   CHECK(agx_vdm_tessellate_indirect_pack(AGX_CHIP_G17P, &indirect, &packed));
   uint8_t stream[AGX_VDM_TESSELLATE_INDIRECT_CORE_LENGTH + 8];
   memcpy(stream, packed.bytes, packed.length);
   memset(stream + packed.length, 0x5a, 8);

   struct agx_vdm_decoded_record record;
   CHECK(agx_vdm_decode_one(AGX_CHIP_G17P, stream, sizeof(stream), &record));
   CHECK(record.kind == AGX_VDM_RECORD_TESSELLATE_INDIRECT);
   CHECK(record.length == AGX_VDM_TESSELLATE_INDIRECT_CORE_LENGTH);
   CHECK(memcmp(&record.data.tessellate_indirect, &indirect,
                sizeof(indirect)) == 0);
   CHECK(!agx_vdm_decode_one(AGX_CHIP_G17P, stream,
                             AGX_VDM_TESSELLATE_INDIRECT_CORE_LENGTH - 4, &record));

   return 0;
}

static int
test_stream_terminate(void)
{
   for (unsigned t = 0;
        t < sizeof(supported_targets) / sizeof(supported_targets[0]); ++t) {
      const enum agx_chip chip = supported_targets[t];
      struct agx_vdm_stream_terminate_core packed;
      CHECK(agx_vdm_stream_terminate_core_pack(chip, &packed));

      uint8_t stream[AGX_VDM_STREAM_TERMINATE_CORE_LENGTH + 4];
      memcpy(stream, packed.bytes, packed.length);
      memset(stream + packed.length, 0x5a, 4);

      struct agx_vdm_decoded_record record;
      CHECK(agx_vdm_decode_one(chip, stream, sizeof(stream), &record));
      CHECK(record.kind == AGX_VDM_RECORD_STREAM_TERMINATE);
      CHECK(record.length == packed.length);
      CHECK(record.chip == chip);
   }

   return 0;
}

static int
test_rejection(void)
{
   const enum agx_chip chip = AGX_CHIP_G17S;
   struct agx_vdm_ppp_state_core ppp;
   struct agx_vdm_barrier_core barrier;
   struct agx_vdm_state_update state;
   struct agx_vdm_index_list_direct_nonindexed nonindexed;
   struct agx_vdm_index_list_direct_indexed indexed;
   struct agx_vdm_index_list_indirect_nonindexed indirect_nonindexed;
   struct agx_vdm_index_list_indirect_indexed indirect_indexed;
   struct agx_vdm_stream_link_core link;
   struct agx_vdm_tessellate_direct tessellate;
   struct agx_vdm_stream_terminate_core terminate;
   const struct agx_vdm_state_update_info state_info = {
      .pipeline = UINT64_C(0x123456780),
   };
   const struct agx_vdm_index_list_direct_nonindexed_info nonindexed_info = {
      .primitive = 3,
      .count = 7,
      .instances = 1,
      .start = 2,
   };
   const struct agx_vdm_index_list_direct_indexed_info indexed_info = {
      .primitive = 3,
      .index_size = AGX_VDM_INDEX_LIST_INDEX_SIZE_U16,
      .index_buffer = UINT64_C(0x123456780),
      .count = 7,
      .instances = 1,
      .start = 2,
      .index_buffer_size_word = 9,
   };
   const struct agx_vdm_index_list_indirect_nonindexed_info
      indirect_nonindexed_info = {
         .primitive = 3,
         .indirect_buffer = UINT64_C(0x23456789abc0),
      };
   const struct agx_vdm_index_list_indirect_indexed_info
      indirect_indexed_info = {
         .primitive = 3,
         .restart_enable = true,
         .index_size = AGX_VDM_INDEX_LIST_INDEX_SIZE_U16,
         .index_buffer = UINT64_C(0x123456780),
         .indirect_buffer = UINT64_C(0x23456789abc0),
         .restart_index = UINT32_MAX,
         .index_buffer_size_word = 9,
      };
   const struct agx_vdm_tessellate_direct_info tessellate_info = {
      .factor_buffer = UINT64_C(0x123456780),
      .patch_count = 7,
      .instance_count = 1,
      .base_patch = 2,
      .base_instance = 3,
      .instance_stride_word = 4,
      .factor_buffer_size_minus_one = 8,
   };

   CHECK(agx_vdm_ppp_state_core_pack(chip, UINT64_C(0x123456780), 7, &ppp));
   CHECK(agx_vdm_barrier_core_pack(chip, false, &barrier));
   CHECK(agx_vdm_state_update_pack(chip, &state_info, &state));
   CHECK(agx_vdm_index_list_direct_nonindexed_pack(chip, &nonindexed_info,
                                                   &nonindexed));
   CHECK(agx_vdm_index_list_direct_indexed_pack(chip, &indexed_info, &indexed));
   CHECK(agx_vdm_index_list_indirect_nonindexed_pack(
      chip, &indirect_nonindexed_info, &indirect_nonindexed));
   CHECK(agx_vdm_index_list_indirect_indexed_pack(
      chip, &indirect_indexed_info, &indirect_indexed));
   CHECK(agx_vdm_stream_link_core_pack(chip, UINT64_C(0x123456789), &link));
   CHECK(agx_vdm_tessellate_direct_pack(chip, &tessellate_info, &tessellate));
   CHECK(agx_vdm_stream_terminate_core_pack(chip, &terminate));

   const struct {
      const uint8_t *bytes;
      size_t length;
   } records[] = {
      {ppp.bytes, ppp.length},
      {barrier.bytes, barrier.length},
      {state.bytes, state.length},
      {nonindexed.bytes, nonindexed.length},
      {indexed.bytes, indexed.length},
      {indirect_nonindexed.bytes, indirect_nonindexed.length},
      {indirect_indexed.bytes, indirect_indexed.length},
      {link.bytes, link.length},
      {tessellate.bytes, tessellate.length},
      {terminate.bytes, terminate.length},
   };

   for (unsigned r = 0; r < sizeof(records) / sizeof(records[0]); ++r) {
      for (size_t available = 0; available < records[r].length; ++available)
         CHECK(check_rejected(chip, records[r].bytes, available) == 0);

      for (unsigned t = 0;
           t < sizeof(unsupported_targets) / sizeof(unsupported_targets[0]);
           ++t) {
         CHECK(check_rejected(unsupported_targets[t], records[r].bytes,
                              records[r].length) == 0);
      }
   }

   for (uint32_t type = 1; type < 8; ++type) {
      if (type == 1 || type == 2 || type == 3 || type == 4 || type == 5 ||
          type == 6)
         continue;
      const uint8_t word[4] = {0, 0, 0, (uint8_t)(type << 5)};
      CHECK(check_rejected(chip, word, sizeof(word)) == 0);
   }

   uint8_t malformed_ppp[AGX_VDM_PPP_STATE_CORE_LENGTH];
   memcpy(malformed_ppp, ppp.bytes, sizeof(malformed_ppp));
   malformed_ppp[2] = 1;
   CHECK(check_rejected(chip, malformed_ppp, sizeof(malformed_ppp)) == 0);

   static const uint8_t aggregate_multi_agc[AGX_VDM_BARRIER_CORE_LENGTH] = {
      0x7c,
      0x09,
      0x00,
      0x21,
   };
   for (unsigned t = 0;
        t < sizeof(supported_targets) / sizeof(supported_targets[0]); ++t) {
      CHECK(check_rejected(supported_targets[t], aggregate_multi_agc,
                           sizeof(aggregate_multi_agc)) == 0);
   }

   uint8_t malformed_state[AGX_VDM_STATE_UPDATE_LENGTH];
   memcpy(malformed_state, state.bytes, sizeof(malformed_state));
   malformed_state[0] ^= 1;
   CHECK(check_rejected(chip, malformed_state, sizeof(malformed_state)) == 0);

   uint8_t malformed_link[AGX_VDM_STREAM_LINK_CORE_LENGTH];
   memcpy(malformed_link, link.bytes, sizeof(malformed_link));
   malformed_link[2] = 1;
   CHECK(check_rejected(chip, malformed_link, sizeof(malformed_link)) == 0);

   static const uint8_t
      ungrounded_return[AGX_VDM_STREAM_TERMINATE_CORE_LENGTH] = {
         0x00,
         0x00,
         0x00,
         0xe0,
      };
   for (unsigned t = 0;
        t < sizeof(supported_targets) / sizeof(supported_targets[0]); ++t) {
      CHECK(check_rejected(supported_targets[t], ungrounded_return,
                           sizeof(ungrounded_return)) == 0);
   }

   struct agx_vdm_decoded_record record;
   CHECK(!agx_vdm_decode_one(chip, ppp.bytes, ppp.length, NULL));
   CHECK(!agx_vdm_decode_one(chip, NULL, ppp.length, &record));
   return 0;
}

int
main(void)
{
   CHECK(test_ppp_state_update() == 0);
   CHECK(test_barrier() == 0);
   CHECK(test_state_update() == 0);
   CHECK(test_index_lists() == 0);
   CHECK(test_indirect_index_lists() == 0);
   CHECK(test_stream_link() == 0);
   CHECK(test_tessellate_direct() == 0);
   CHECK(test_stream_terminate() == 0);
   CHECK(test_rejection() == 0);

   printf("agx_vdm_decode_core: %u checks passed\n", checks);
   return 0;
}

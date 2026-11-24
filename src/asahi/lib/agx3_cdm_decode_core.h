/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Strict record-at-a-time decoder for the grounded AGX3 CDM subset.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

enum agx3_cdm_record_kind {
   AGX3_CDM_RECORD_UNKNOWN = 0,
   AGX3_CDM_RECORD_LAUNCH,
   AGX3_CDM_RECORD_INDIRECT_LAUNCH,
   AGX3_CDM_RECORD_INDIRECT_LAUNCH_36,
   AGX3_CDM_RECORD_STREAM_LINK,
   AGX3_CDM_RECORD_LINK_PADDING,
   AGX3_CDM_RECORD_BARRIER,
   AGX3_CDM_RECORD_SSTT_FLUSH_INVALIDATE,
   AGX3_CDM_RECORD_SUBSTREAM_BARRIER,
   AGX3_CDM_RECORD_SECONDARY_BARRIER,
   AGX3_CDM_RECORD_STREAM_TERMINATE,
};

struct agx3_cdm_decoded_record {
   enum agx3_cdm_record_kind kind;
   uint8_t length;
   enum agx_chip chip;
   union {
      struct {
         uint32_t word0_payload;
         uint32_t word1_payload;
         uint64_t pipeline;
         uint32_t opaque_word3;
         uint32_t global_size[3];
         uint32_t local_size[3];
      } launch;
      struct {
         uint32_t word0_payload;
         uint32_t word1_payload;
         uint64_t pipeline;
         uint32_t opaque_word3;
         uint16_t word4_payload;
         uint64_t parameters;
         /* Set only for the nine-word form; left unnamed because the
          * recovered corpus does not prove their meaning. */
         uint32_t trailing_words[3];
      } indirect_launch;
      struct {
         uint64_t target;
         bool with_return;
      } stream_link;
      struct {
         uint8_t count;
      } link_padding;
      struct {
         uint8_t flags;
         bool flush_memory;
      } barrier;
      struct {
         uint8_t flags;
         uint8_t substream;
      } substream_barrier;
   } data;
};

/* Decode one CDM record from the start of a stream (`available` may include
 * trailing records; `record->length` gives bytes consumed). Fails with a
 * cleared `record` on unknown types, reserved-bit errors, or truncation. */
bool agx3_cdm_decode_one(enum agx_chip chip, const void *bytes,
                         size_t available,
                         struct agx3_cdm_decoded_record *record);

#ifdef __cplusplus
}
#endif

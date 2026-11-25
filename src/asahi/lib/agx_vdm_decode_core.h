/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Strict record-at-a-time decoder for the grounded AGX3 VDM subset.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"
#include "agx_vdm_index_list_core.h"
#include "agx_vdm_state_update_core.h"
#include "agx_vdm_tessellate_core.h"
#include "agx_vdm_mesh_core.h"

#ifdef __cplusplus
extern "C" {
#endif

enum agx_vdm_record_kind {
   AGX_VDM_RECORD_UNKNOWN = 0,
   AGX_VDM_RECORD_PPP_STATE_UPDATE,
   AGX_VDM_RECORD_BARRIER,
   AGX_VDM_RECORD_STATE_UPDATE,
   AGX_VDM_RECORD_INDEX_LIST_NONINDEXED,
   AGX_VDM_RECORD_INDEX_LIST_INDEXED,
   AGX_VDM_RECORD_INDEX_LIST_INDIRECT_NONINDEXED,
   AGX_VDM_RECORD_INDEX_LIST_INDIRECT_INDEXED,
   AGX_VDM_RECORD_STREAM_LINK,
   AGX_VDM_RECORD_STREAM_TERMINATE,
   AGX_VDM_RECORD_TESSELLATE_DIRECT,
   AGX_VDM_RECORD_TESSELLATE_INDIRECT,
   AGX_VDM_RECORD_TESSELLATION_STATE,
   AGX_VDM_RECORD_MESH_STATE,
   AGX_VDM_RECORD_MESH_DISPATCH,
};

struct agx_vdm_decoded_record {
   enum agx_vdm_record_kind kind;
   uint8_t length;
   enum agx_chip chip;
   union {
      struct {
         uint64_t pointer;
         uint32_t size_words;
      } ppp_state_update;
      struct agx_vdm_state_update_info state_update;
      struct agx_vdm_index_list_direct_nonindexed_info index_list_nonindexed;
      struct agx_vdm_index_list_direct_indexed_info index_list_indexed;
      struct agx_vdm_index_list_indirect_nonindexed_info
         index_list_indirect_nonindexed;
      struct agx_vdm_index_list_indirect_indexed_info
         index_list_indirect_indexed;
      struct agx_vdm_tessellate_direct_info tessellate_direct;
      struct agx_vdm_tessellate_indirect_info tessellate_indirect;
      struct agx_vdm_tessellation_state_info tessellation_state;
      struct agx_vdm_mesh_state_info mesh_state;
      struct agx_vdm_mesh_dispatch_info mesh_dispatch;
      struct {
         uint64_t target;
      } stream_link;
   } data;
};

/* Decode one VDM record from the start of a stream (`available` may include
 * trailing records; `record->length` gives bytes consumed). Fails with a
 * cleared `record` on unknown types, reserved-bit errors, or truncation. */
bool agx_vdm_decode_one(enum agx_chip chip, const void *bytes, size_t available,
                        struct agx_vdm_decoded_record *record);

#ifdef __cplusplus
}
#endif

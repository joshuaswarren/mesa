/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_decode_core.h"
#include "agx_bytes.h"

#include "agx_vdm_barrier_core.h"
#include "agx_vdm_ppp_state_core.h"
#include "agx_vdm_stream_link_core.h"
#include "agx_vdm_stream_terminate_core.h"
#include "agx_vdm_tessellate_core.h"

#include <string.h>

bool
agx_vdm_decode_one(enum agx_chip chip, const void *bytes, size_t available,
                   struct agx_vdm_decoded_record *record)
{
   if (record)
      memset(record, 0, sizeof(*record));

   if (!bytes || !record || available < sizeof(uint32_t))
      return false;

   const uint32_t word0 = agx_load_le32(bytes);
   const uint8_t block_type = word0 >> 29;

   /* Native M5 mesh forms share block types with ordinary state/index
    * records. Match their exact captured headers before those grammars. */
   if (word0 == AGX_VDM_MESH_STATE_HEADER) {
      if (available < AGX_VDM_MESH_STATE_LENGTH ||
          !agx_vdm_mesh_state_unpack(chip, bytes, AGX_VDM_MESH_STATE_LENGTH,
                                    &record->data.mesh_state))
         return false;
      record->kind = AGX_VDM_RECORD_MESH_STATE;
      record->length = AGX_VDM_MESH_STATE_LENGTH;
      record->chip = chip;
      return true;
   }
   if ((word0 & ~UINT32_C(0xff00)) == AGX_VDM_MESH_DISPATCH_BASE) {
      if (available < AGX_VDM_MESH_DISPATCH_LENGTH ||
          !agx_vdm_mesh_dispatch_unpack(chip, bytes, AGX_VDM_MESH_DISPATCH_LENGTH,
                                       &record->data.mesh_dispatch))
         return false;
      record->kind = AGX_VDM_RECORD_MESH_DISPATCH;
      record->length = AGX_VDM_MESH_DISPATCH_LENGTH;
      record->chip = chip;
      return true;
   }

   switch (block_type) {
   case 0: {
      uint64_t pointer;
      uint32_t size_words;
      if (available < AGX_VDM_PPP_STATE_CORE_LENGTH ||
          !agx_vdm_ppp_state_core_unpack(
             chip, bytes, AGX_VDM_PPP_STATE_CORE_LENGTH, &pointer, &size_words))
         return false;

      record->kind = AGX_VDM_RECORD_PPP_STATE_UPDATE;
      record->length = AGX_VDM_PPP_STATE_CORE_LENGTH;
      record->chip = chip;
      record->data.ppp_state_update.pointer = pointer;
      record->data.ppp_state_update.size_words = size_words;
      return true;
   }

   case 1:
      if (available < AGX_VDM_BARRIER_CORE_LENGTH ||
          !agx_vdm_barrier_core_unpack(chip, bytes,
                                       AGX_VDM_BARRIER_CORE_LENGTH))
         return false;

      record->kind = AGX_VDM_RECORD_BARRIER;
      record->length = AGX_VDM_BARRIER_CORE_LENGTH;
      record->chip = chip;
      return true;

   case 2: {
      const bool hal300 = agx_chip_hal300_path(chip);
      const size_t indirect_indexed_length =
         hal300 ? AGX_VDM_INDEX_LIST_INDIRECT_INDEXED_MAX_LENGTH
                : AGX_VDM_INDEX_LIST_INDIRECT_INDEXED_MAX_LENGTH - 4;
      struct agx_vdm_state_update_info state_update;
      struct agx_vdm_index_list_indirect_indexed_info indirect_indexed;
      struct agx_vdm_tessellation_state_info tessellation_state;
      const bool is_state_update =
         available >= AGX_VDM_STATE_UPDATE_LENGTH &&
         agx_vdm_state_update_unpack(chip, bytes, AGX_VDM_STATE_UPDATE_LENGTH,
                                     &state_update);
      const bool is_indirect_indexed =
         available >= indirect_indexed_length &&
         agx_vdm_index_list_indirect_indexed_unpack(
            agx_vdm_index_list_record_chip(chip), bytes,
            indirect_indexed_length, &indirect_indexed);

      const bool is_tessellation_state =
         available >= AGX_VDM_TESSELLATION_STATE_LENGTH &&
         agx_vdm_tessellation_state_unpack(chip, bytes,
                                           AGX_VDM_TESSELLATION_STATE_LENGTH,
                                           &tessellation_state);

      /* The indexed-indirect form starts with its exact 8-byte restart State
       * Update.  Require a unique complete grammar match rather than treating
       * that prefix as a standalone state record. */
      if ((unsigned)is_state_update + (unsigned)is_indirect_indexed +
             (unsigned)is_tessellation_state !=
          1)
         return false;

      record->chip = chip;
      if (is_tessellation_state) {
         record->kind = AGX_VDM_RECORD_TESSELLATION_STATE;
         record->length = AGX_VDM_TESSELLATION_STATE_LENGTH;
         record->data.tessellation_state = tessellation_state;
      } else if (is_state_update) {
         record->kind = AGX_VDM_RECORD_STATE_UPDATE;
         record->length = AGX_VDM_STATE_UPDATE_LENGTH;
         record->data.state_update = state_update;
      } else {
         record->kind = AGX_VDM_RECORD_INDEX_LIST_INDIRECT_INDEXED;
         record->length = (uint8_t)indirect_indexed_length;
         record->data.index_list_indirect_indexed = indirect_indexed;
      }
      return true;
   }

   case 3: {
      const bool hal300 = agx_chip_hal300_path(chip);
      const size_t indexed_length = hal300 ? 32 : 28;
      const size_t nonindexed_length =
         agx_vdm_index_list_nonindexed_has_tail(chip) ? 20 : 16;
      const size_t indirect_nonindexed_length =
         hal300 ? AGX_VDM_INDEX_LIST_INDIRECT_NONINDEXED_MAX_LENGTH
                : AGX_VDM_INDEX_LIST_INDIRECT_NONINDEXED_MAX_LENGTH - 4;
      struct agx_vdm_index_list_direct_indexed_info indexed;
      struct agx_vdm_index_list_direct_nonindexed_info nonindexed;
      struct agx_vdm_index_list_indirect_nonindexed_info indirect_nonindexed;
      const bool is_indexed =
         available >= indexed_length &&
         agx_vdm_index_list_direct_indexed_unpack(
            agx_vdm_index_list_record_chip(chip), bytes, indexed_length,
            &indexed);
      const bool is_nonindexed =
         available >= nonindexed_length &&
         agx_vdm_index_list_direct_nonindexed_unpack(
            agx_vdm_index_list_record_chip(chip), bytes, nonindexed_length,
            &nonindexed);
      const bool is_indirect_nonindexed =
         available >= indirect_nonindexed_length &&
         agx_vdm_index_list_indirect_nonindexed_unpack(
            agx_vdm_index_list_record_chip(chip), bytes,
            indirect_nonindexed_length, &indirect_nonindexed);

      if ((unsigned)is_indexed + (unsigned)is_nonindexed +
             (unsigned)is_indirect_nonindexed !=
          1)
         return false;

      record->chip = chip;
      if (is_indexed) {
         record->kind = AGX_VDM_RECORD_INDEX_LIST_INDEXED;
         record->length = (uint8_t)indexed_length;
         record->data.index_list_indexed = indexed;
      } else if (is_nonindexed) {
         record->kind = AGX_VDM_RECORD_INDEX_LIST_NONINDEXED;
         record->length = (uint8_t)nonindexed_length;
         record->data.index_list_nonindexed = nonindexed;
      } else {
         record->kind = AGX_VDM_RECORD_INDEX_LIST_INDIRECT_NONINDEXED;
         record->length = (uint8_t)indirect_nonindexed_length;
         record->data.index_list_indirect_nonindexed = indirect_nonindexed;
      }
      return true;
   }

   case 4: {
      uint64_t target;
      if (available < AGX_VDM_STREAM_LINK_CORE_LENGTH ||
          !agx_vdm_stream_link_core_unpack(
             chip, bytes, AGX_VDM_STREAM_LINK_CORE_LENGTH, &target))
         return false;

      record->kind = AGX_VDM_RECORD_STREAM_LINK;
      record->length = AGX_VDM_STREAM_LINK_CORE_LENGTH;
      record->chip = chip;
      record->data.stream_link.target = target;
      return true;
   }

   case 5:
      if (available >= AGX_VDM_TESSELLATE_DIRECT_LENGTH &&
          agx_vdm_tessellate_direct_unpack(
             chip, bytes, AGX_VDM_TESSELLATE_DIRECT_LENGTH,
             &record->data.tessellate_direct)) {
         record->kind = AGX_VDM_RECORD_TESSELLATE_DIRECT;
         record->length = AGX_VDM_TESSELLATE_DIRECT_LENGTH;
      } else if (available >= AGX_VDM_TESSELLATE_INDIRECT_CORE_LENGTH &&
                 agx_vdm_tessellate_indirect_unpack(
                    chip, bytes, AGX_VDM_TESSELLATE_INDIRECT_CORE_LENGTH,
                    &record->data.tessellate_indirect)) {
         record->kind = AGX_VDM_RECORD_TESSELLATE_INDIRECT;
         record->length = AGX_VDM_TESSELLATE_INDIRECT_CORE_LENGTH;
      } else {
         return false;
      }

      record->chip = chip;
      return true;

   case 6:
      if (available < AGX_VDM_STREAM_TERMINATE_CORE_LENGTH ||
          !agx_vdm_stream_terminate_core_unpack(
             chip, bytes, AGX_VDM_STREAM_TERMINATE_CORE_LENGTH))
         return false;

      record->kind = AGX_VDM_RECORD_STREAM_TERMINATE;
      record->length = AGX_VDM_STREAM_TERMINATE_CORE_LENGTH;
      record->chip = chip;
      return true;

   case 7:
   default:
      return false;
   }
}

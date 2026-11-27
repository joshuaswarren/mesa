/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_ppp_state_core.h"
#include "agx_bytes.h"

#include <string.h>

bool
agx_vdm_ppp_state_core_supported(enum agx_chip chip)
{
   return agx_chip_has_record(chip, AGX_RECORD_VDM_PPP_STATE);
}

bool
agx_vdm_ppp_state_core_pack(enum agx_chip chip, uint64_t pointer,
                            uint32_t size_words,
                            struct agx_vdm_ppp_state_core *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !agx_vdm_ppp_state_core_supported(chip) ||
       pointer >= (UINT64_C(1) << 40) || (pointer & UINT64_C(3)) != 0 ||
       size_words > AGX_VDM_PPP_STATE_CORE_MAX_SIZE_WORDS)
      return false;

   const uint32_t word0 = (uint32_t)(pointer >> 32) | (size_words << 8);
   const uint32_t word1 = (uint32_t)pointer;

   agx_store_le32(&out->bytes[0], word0);
   agx_store_le32(&out->bytes[4], word1);
   out->length = AGX_VDM_PPP_STATE_CORE_LENGTH;
   return true;
}

bool
agx_vdm_ppp_state_core_unpack(enum agx_chip chip, const void *bytes,
                              size_t size, uint64_t *pointer,
                              uint32_t *size_words)
{
   if (pointer)
      *pointer = 0;
   if (size_words)
      *size_words = 0;

   if (!bytes || !pointer || !size_words ||
       size != AGX_VDM_PPP_STATE_CORE_LENGTH ||
       !agx_vdm_ppp_state_core_supported(chip))
      return false;

   const uint8_t *raw = bytes;
   const uint32_t word0 = agx_load_le32(&raw[0]);
   const uint32_t word1 = agx_load_le32(&raw[4]);

   if ((word0 & AGX_VDM_PPP_STATE_CORE_RESERVED_MASK) != 0 ||
       (word1 & UINT32_C(3)) != 0)
      return false;

   *pointer = ((uint64_t)(word0 & UINT32_C(0xff)) << 32) | word1;
   *size_words = (word0 >> 8) & AGX_VDM_PPP_STATE_CORE_MAX_SIZE_WORDS;
   return true;
}

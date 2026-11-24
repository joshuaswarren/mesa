/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_stream_terminate_core.h"
#include "agx_bytes.h"

#include <string.h>

/* Pack/unpack the 4-byte type-6 VDM Stream Terminate record (0xc0000000);
 * type-7 Stream Return is not accepted. */
bool
agx_vdm_stream_terminate_core_supported(enum agx_chip chip)
{
   return agx_chip_has_record(chip, AGX_RECORD_VDM_STREAM_TERMINATE);
}

bool
agx_vdm_stream_terminate_core_pack(enum agx_chip chip,
                                   struct agx_vdm_stream_terminate_core *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !agx_vdm_stream_terminate_core_supported(chip))
      return false;

   agx_store_le32(out->bytes,
                                       AGX_VDM_STREAM_TERMINATE_CORE_WORD);
   out->length = AGX_VDM_STREAM_TERMINATE_CORE_LENGTH;
   return true;
}

bool
agx_vdm_stream_terminate_core_unpack(enum agx_chip chip, const void *bytes,
                                     size_t size)
{
   if (!bytes || size != AGX_VDM_STREAM_TERMINATE_CORE_LENGTH ||
       !agx_vdm_stream_terminate_core_supported(chip))
      return false;

   return agx_load_le32(bytes) ==
          AGX_VDM_STREAM_TERMINATE_CORE_WORD;
}

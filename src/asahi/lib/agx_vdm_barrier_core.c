/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_barrier_core.h"
#include "agx_bytes.h"

#include <string.h>

/*
 * Four-byte type-1 VDM memory barrier. Pack sets USC cache invalidate (bit 3)
 * when admitted by the kernel; unpack accepts the word with or without bit 3
 * and rejects ungrounded bits 2/4.
 */
bool
agx_vdm_barrier_core_supported(enum agx_chip chip)
{
   /* G18G memory barriers use the same word. */
   return agx_chip_has_record(chip, AGX_RECORD_VDM_BARRIER);
}

bool
agx_vdm_barrier_core_pack(enum agx_chip chip, bool kernel_admits_inval,
                          struct agx_vdm_barrier_core *out)
{
   if (out)
      memset(out, 0, sizeof(*out));

   if (!out || !agx_vdm_barrier_core_supported(chip))
      return false;

   agx_store_le32(out->bytes, agx_vdm_barrier_core_word(chip, kernel_admits_inval));
   out->length = AGX_VDM_BARRIER_CORE_LENGTH;
   return true;
}

bool
agx_vdm_barrier_core_unpack(enum agx_chip chip, const void *bytes, size_t size)
{
   if (!bytes || size != AGX_VDM_BARRIER_CORE_LENGTH ||
       !agx_vdm_barrier_core_supported(chip))
      return false;

   return (agx_load_le32(bytes) & ~AGX_VDM_BARRIER_CORE_USC_INVAL) ==
          AGX_VDM_BARRIER_CORE_PLAIN;
}

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_vdm_barrier_core.h"
#include "agx_bytes.h"

#include <string.h>

/*
 * Pack/unpack only the four-byte type-1 VDM memory barrier used on
 * G15G/G15X, G16G/G16X, G17P, and G17G/G17X.  The target allowlist contains
 * only those identities; G16P and G17A remain closed.
 *
 * Pack sets the USC cache invalidate (bit 3) where the chip's kernel admits it
 * (agx_vdm_barrier_core_word); unpack accepts the word with or without it, so
 * Apple's 0x21000960 captures still decode.  The
 * aggregate G15X family can set word bits 2..4 on multi-cluster parts.  No
 * per-identity value for bits 2 and 4 is pinned, so this helper deliberately
 * rejects those alternate forms on every target.  This is a
 * representation/validation helper only and does not emit a live command,
 * submit work, or admit a device.
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

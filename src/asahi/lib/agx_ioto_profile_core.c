/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ioto_profile_core.h"
#include "agx_bytes.h"

#include <string.h>

#define AGX_IOTO_PROFILE_CONTROL_CLASSIC_OPCODE UINT16_C(0x03f7)
#define AGX_IOTO_PROFILE_CONTROL_HAL300_OPCODE  UINT16_C(0x0007)
#define AGX_IOTO_PROFILE_CONTROL_CLASSIC_PAYLOAD0 UINT16_C(0x07ff)
#define AGX_IOTO_PROFILE_CONTROL_HAL300_PAYLOAD0  UINT16_C(0xf098)
#define AGX_IOTO_PROFILE_CONTROL_HAL300_PAYLOAD1  UINT16_C(0xfc00)
#define AGX_IOTO_PROFILE_CONTROL_HAL300_WORD1_FIXED UINT16_C(0x0f00)

static const uint8_t g17s_minimal_profile_loader
   [AGX_IOTO_G17S_MINIMAL_PROFILE_LOADER_SIZE] = {
      0x1c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x34, 0x81, 0x01, 0x00, 0x20, 0x00, 0xa0, 0x02,
      0x0c, 0x80, 0x02, 0x00, 0x08, 0x00, 0x00, 0x00,
      0x27, 0x04, 0x04, 0x1a, 0x21, 0x00, 0xa3, 0x02,
      0x88, 0x80, 0x00, 0x00,
      0x1c, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x27, 0x04, 0x04, 0x00, 0x30, 0x00, 0xa4, 0x12,
   };

bool
agx_ioto_profile_control_supported(enum agx_chip chip)
{
   return agx_chip_has_record(chip, AGX_RECORD_IOTO_PROFILE_CONTROL);
}

size_t
agx_ioto_profile_control_size(enum agx_chip chip)
{
   if (!agx_ioto_profile_control_supported(chip))
      return 0;

   return agx_chip_g17_m5(chip) ? 6 : 4;
}

bool
agx_ioto_profile_control_pack(const struct agx_ioto_profile_control *profile,
                              uint8_t out[AGX_IOTO_PROFILE_CONTROL_MAX_SIZE])
{
   if (!out)
      return false;

   memset(out, 0, AGX_IOTO_PROFILE_CONTROL_MAX_SIZE);
   if (!profile || !agx_ioto_profile_control_supported(profile->chip))
      return false;

   if (agx_chip_g17_m5(profile->chip)) {
      if ((profile->payload0 & ~AGX_IOTO_PROFILE_CONTROL_HAL300_PAYLOAD0) != 0 ||
          (profile->payload1 & ~AGX_IOTO_PROFILE_CONTROL_HAL300_PAYLOAD1) != 0)
         return false;
   } else if ((profile->payload0 &
               ~AGX_IOTO_PROFILE_CONTROL_CLASSIC_PAYLOAD0) != 0 ||
              profile->payload1 != 0) {
      return false;
   }

   agx_store_le16(out, agx_chip_g17_m5(profile->chip)
                      ? AGX_IOTO_PROFILE_CONTROL_HAL300_OPCODE
                      : AGX_IOTO_PROFILE_CONTROL_CLASSIC_OPCODE);
   agx_store_le16(out + 2,
              profile->payload0 |
                 (agx_chip_g17_m5(profile->chip)
                     ? AGX_IOTO_PROFILE_CONTROL_HAL300_WORD1_FIXED
                     : 0));
   if (agx_chip_g17_m5(profile->chip))
      agx_store_le16(out + 4, profile->payload1);

   return true;
}

bool
agx_ioto_g17s_minimal_profile_loader_pack(void *output, size_t capacity)
{
   if (!output || capacity < sizeof(g17s_minimal_profile_loader))
      return false;

   memcpy(output, g17s_minimal_profile_loader,
          sizeof(g17s_minimal_profile_loader));
   return true;
}

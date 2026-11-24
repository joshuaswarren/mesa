/* SPDX-License-Identifier: MIT */
#include "agx_scs_layout.h"
#include "agx_bytes.h"

#include <string.h>

static bool
agx_scs_layout_init(uint64_t packed, struct agx_scs_layout_gen2 *layout,
                    bool gen1)
{
   if (!layout)
      return false;

   const uint64_t depth = packed & 0xffff;
   const uint64_t levels = (packed >> 16) & 0xff;
   const uint64_t cliques = (packed >> 24) & 0xff;
   const uint64_t user = (packed >> 32) & 0xffff;
   const uint64_t driver = packed >> 48;
   const uint64_t threads = cliques * 32;
   uint64_t end[13] = {0};

   /* Fixed state, then per-thread stacks, driver, level and user payloads. */
   end[1] = 0x6000;
   end[2] = end[1] + 0x2000;
   end[3] = end[2] + 0x40000;
   end[4] = end[3] + (gen1 ? 0x10000 : 0x20000);
   end[5] = end[4] + threads * depth * 8;
   end[10] = end[5] + threads * driver;
   end[6] = end[10] + (gen1 ? 0 : cliques * 256 * (levels + (levels > 1)));
   end[11] = end[6] + threads * user;
   end[12] = end[11];

   for (unsigned i = 0; i < 13; ++i) {
      if (end[i] > UINT32_MAX)
         return false;
   }

   layout->packed = packed;
   for (unsigned i = 0; i < 13; ++i)
      layout->end[i] = end[i];
   return true;
}

bool
agx_scs_layout_gen1_init(uint64_t packed, struct agx_scs_layout_gen2 *layout)
{
   return agx_scs_layout_init(packed, layout, true);
}

bool
agx_scs_layout_gen2_init(uint64_t packed, struct agx_scs_layout_gen2 *layout)
{
   return agx_scs_layout_init(packed, layout, false);
}

void
agx_scs_layout_gen2_parameters(const struct agx_scs_layout_gen2 *layout,
                              uint8_t parameters[48])
{
   /* Per-kick state starts zeroed. Payload placement remains a property of
    * the submitted shader rather than a fixed offset in this record. */
   memset(parameters, 0, 48);
   agx_store_le32(parameters + 8, layout->end[6]);
   agx_store_le32(parameters + 12, layout->end[11]);
   agx_store_le32(parameters + 16, layout->packed);
   agx_store_le32(parameters + 20, layout->packed >> 32);
   agx_store_le32(parameters + 24, layout->end[5]);
   agx_store_le32(parameters + 40, layout->end[10] - layout->end[5]);
   const unsigned levels = (layout->packed >> 16) & 0xff;
   agx_store_le32(parameters + 44, (levels + (levels > 1)) * 8);
}

void
agx_scs_layout_gen1_parameters(const struct agx_scs_layout_gen2 *layout,
                              uint8_t parameters[48])
{
   /* On G17P the SCS parameters carry the fixed Gen1 window address and zero
    * level stride, including on a payload-free layout. */
   memset(parameters, 0, 48);
   agx_store_le32(parameters + 4, 0x6f);
   agx_store_le32(parameters + 8, layout->end[6]);
   agx_store_le32(parameters + 12, layout->end[11]);
   agx_store_le32(parameters + 16, layout->packed);
   agx_store_le32(parameters + 20, layout->packed >> 32);
   agx_store_le32(parameters + 24, layout->end[5]);
   agx_store_le32(parameters + 40, layout->end[10] - layout->end[5]);
}

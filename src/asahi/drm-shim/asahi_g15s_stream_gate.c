/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "asahi_g15s_stream_gate.h"

#include <stdio.h>
#include <string.h>

#include "util/macros.h"

static bool
all_zero(const void *data, size_t size)
{
   const uint8_t *bytes = data;
   for (size_t i = 0; i < size; ++i) {
      if (bytes[i])
         return false;
   }
   return true;
}

static bool
refuse(const char *what)
{
   fprintf(stderr, "shim: G15S kernel refuses %s\n", what);
   return false;
}

/* The kernel's known payloads are Aurora's flat layouts: render is the
 * legacy command, prior 16 and ray_query 2 x 56 (448 bytes with the merged
 * UAPI's timestamp copies), compute the command with its prior tail (200). */
#define G15S_RENDER_KNOWN  sizeof(struct drm_asahi_cmd_render_with_ray_query)
#define G15S_COMPUTE_KNOWN sizeof(struct drm_asahi_cmd_compute_with_prior)

bool
asahi_g15s_render_packet_validate(const struct drm_asahi_cmd_render *render,
                                  const void *payload, size_t size,
                                  uint64_t features)
{
   uint32_t accepted = DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES |
                       DRM_ASAHI_RENDER_NO_VERTEX_CLUSTERING |
                       DRM_ASAHI_RENDER_RSRC_SPEC_HI |
                       DRM_ASAHI_RENDER_DBIAS_IS_INT;
   if (features & DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS)
      accepted |= DRM_ASAHI_RENDER_FRAGMENT_BARRIERS;

   if (size % 8)
      return refuse("a render payload that is not a multiple of 8 bytes");
   if (size > G15S_RENDER_KNOWN &&
       !all_zero((const uint8_t *)payload + G15S_RENDER_KNOWN,
                 size - G15S_RENDER_KNOWN))
      return refuse("non-zero render bytes past the known payload");
   if (render->flags & ~accepted) {
      fprintf(stderr, "shim: G15S render flags %#x, accepted %#x\n",
              render->flags, accepted);
      return refuse("a render flag");
   }
   if (render->vertex_helper.binary || render->fragment_helper.binary)
      return refuse("render helper programs");
   if (render->sampler_heap || render->sampler_count)
      return refuse("a render sampler heap");
   if (!render->isp_scissor_base)
      return refuse("a zero scissor base");
   if (render->flags & DRM_ASAHI_RENDER_RSRC_SPEC_HI) {
      if (size < 256)
         return refuse("RSRC_SPEC_HI on a render packet shorter than 256 B");
   } else if (render->bg_eot_rsrc_spec_hi ||
              render->bg_eot_partial_rsrc_spec_hi ||
              render->bg_rsrc_spec_hi || render->bg_partial_rsrc_spec_hi) {
      return refuse("resource-specifier high words without RSRC_SPEC_HI");
   }
   if (render->bg_eot_partial_rsrc_spec_hi)
      return refuse("a partial EOT resource-specifier high word");

   /* prior_vdm/prior_cdm until EXACT_PRIOR_BARRIERS, then ray_query. */
   if (size > sizeof(*render)) {
      const size_t tail = MIN2(size, G15S_RENDER_KNOWN) - sizeof(*render);
      const uint8_t *bytes = (const uint8_t *)payload + sizeof(*render);
      size_t checked = tail;
      if (features & DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS) {
         bytes += MIN2(tail, 16);
         checked -= MIN2(tail, 16);
      }
      if (!all_zero(bytes, checked))
         return refuse("prior barriers or ray-query state");
   }
   return true;
}

bool
asahi_g15s_compute_packet_validate(
   const struct drm_asahi_cmd_compute *compute, const void *payload,
   size_t size, uint64_t features)
{
   if (size % 8)
      return refuse("a compute payload that is not a multiple of 8 bytes");
   if (size > G15S_COMPUTE_KNOWN &&
       !all_zero((const uint8_t *)payload + G15S_COMPUTE_KNOWN,
                 size - G15S_COMPUTE_KNOWN))
      return refuse("non-zero compute bytes past the known payload");
   if (compute->flags)
      return refuse("compute flags");
   if (compute->helper.binary)
      return refuse("a compute helper program");
   if (compute->sampler_heap || compute->sampler_count)
      return refuse("a compute sampler heap");
   /* scs_layout, prior_vdm/prior_cdm (with EXACT_PRIOR_BARRIERS), then the
    * auxiliary allocation and intersection pool. */
   const size_t prior_start = sizeof(*compute) + 8, prior_end = prior_start + 16;
   const size_t known = MIN2(size, G15S_COMPUTE_KNOWN);
   for (size_t at = sizeof(*compute); at < known; ++at) {
      const bool prior = at >= prior_start && at < prior_end &&
                         (features & DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS);
      if (!prior && ((const uint8_t *)payload)[at])
         return refuse("compute scs/prior/auxiliary/intersection state");
   }
   return true;
}

static bool
gate_read(const struct asahi_g15s_gate_memory *memory, uint64_t va,
          uint32_t *words, unsigned count)
{
   return memory->read(memory->data, va, words, count * sizeof(uint32_t));
}

static bool
gate_reject(const char *what, uint64_t va, const uint32_t *w)
{
   fprintf(stderr,
           "shim: G15S rejecting %s record at %#llx: %08x %08x %08x %08x\n",
           what, (unsigned long long)va, w[0], w[1], w[2], w[3]);
   return false;
}

bool
asahi_g15s_vdm_gate(const struct asahi_g15s_gate_memory *memory,
                    uint64_t root)
{
   uint32_t w[4] = {0};

   if (root < UINT64_C(0x1000000000) || root >= UINT64_C(0x9000000000) ||
       (root & 3))
      return refuse("a VDM stream base outside the compact window");
   if (!gate_read(memory, root, w, 1))
      return refuse("an unmapped VDM root");
   /* As a kernel without DRM_ASAHI_FEATURE_VDM_USC_INVALIDATE, which this
    * shim does not report: only the word without the USC cache invalidate. */
   if (w[0] != 0x21000960)
      return gate_reject("VDM root", root, w);
   return true;
}

static bool
gate_program(const struct asahi_g15s_gate_memory *memory, uint64_t usc,
             uint64_t program)
{
   uint8_t block[64];

   return program >= usc && program - usc < UINT64_C(0x100000000) &&
          !(program & 63) &&
          memory->read(memory->data, program, block, sizeof(block));
}

bool
asahi_g15s_cdm_gate(const struct asahi_g15s_gate_memory *memory,
                    uint64_t usc, uint64_t va,
                    struct asahi_g15s_gate_stats *stats)
{
   unsigned links = 0;

   for (unsigned records = 0; records < 8192; ++records) {
      uint32_t w[9] = {0};

      if (!gate_read(memory, va, w, 1))
         return refuse("an unmapped CDM record");

      switch (w[0]) {
      case 0x60000160:
      case 0x60001560:
         stats->barriers++;
         va += 4;
         continue;
      case 0x600215fe:
      case 0x600201fe:
         stats->barriers++;
         stats->wide_barriers++;
         va += 4;
         continue;
      case 0x40000000:
         return true;
      }

      if ((w[0] & 0xffff0000) == 0x20000000) {
         if (!gate_read(memory, va, w, 2))
            return refuse("an unmapped CDM link");
         if (++links > 64)
            return true;
         stats->links++;
         va = ((uint64_t)(w[0] & 0xffff) << 32) | w[1];
         if (va & 3)
            return gate_reject("CDM link", va, w);
         continue;
      }

      if (w[0] == 0x00080000) {
         if (!gate_read(memory, va, w, 4))
            return refuse("an unmapped CDM launch");
         const uint64_t program =
            ((uint64_t)(w[1] >> 16) << 32) | ((uint64_t)w[2] << 6);
         if ((w[1] & 0xffff) || w[3] != 0x40 ||
             (w[2] >> 26) != ((w[1] >> 16) & 0x3f))
            return gate_reject("CDM direct launch", va, w);
         if (!gate_program(memory, usc, program))
            return gate_reject("CDM program", va, w);
         stats->launches++;
         va += 40;
         continue;
      }

      if ((w[0] & 0xff7fffff) == 0x08080000 ||
          (w[0] & 0xff7fffff) == 0x10080000) {
         const bool sizes = (w[0] & 0xff7fffff) == 0x08080000;
         if (!gate_read(memory, va, w, sizes ? 9 : 6))
            return refuse("an unmapped CDM indirect launch");
         const uint64_t program =
            ((uint64_t)(w[1] >> 22) << 38) | ((uint64_t)w[2] << 6);
         const uint64_t indirect = ((uint64_t)(w[4] & 0xffff) << 32) | w[5];
         uint32_t grid[3];
         if (w[3] != (0x40000000 | ((program >> 40) & 0xff)) || (w[4] >> 16) ||
             (indirect & 3) ||
             (sizes && (!w[6] || !w[7] || !w[8] || w[6] > 0x7ff ||
                        w[7] > 0x7ff || w[8] > 0x7ff)))
            return gate_reject("CDM indirect launch", va, w);
         if (!gate_program(memory, usc, program) ||
             !gate_read(memory, indirect, grid, 3))
            return gate_reject("CDM indirect", va, w);
         stats->indirect_launches++;
         va += sizes ? 36 : 24;
         continue;
      }

      return gate_reject("CDM", va, w);
   }

   return true;
}

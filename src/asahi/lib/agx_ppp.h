/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/genxml/agx_pack.h"
#include "agx_bo.h"
#include "agx_vdm_ppp_state_core.h"

/* Opaque structure representing a PPP update */
struct agx_ppp_update {
   uint8_t *head;
   uint64_t gpu_base;
   size_t total_size;
   uint8_t *cpu_base;
};

ALWAYS_INLINE static size_t
agx_ppp_update_size(struct AGX_PPP_HEADER *present)
{
   size_t size = AGX_PPP_HEADER_LENGTH;

#define PPP_CASE(x, y)                                                         \
   if (present->x)                                                             \
      size += AGX_##y##_LENGTH;
   PPP_CASE(fragment_control, FRAGMENT_CONTROL);
   PPP_CASE(fragment_control_2, FRAGMENT_CONTROL);
   PPP_CASE(fragment_front_face, FRAGMENT_FACE);
   PPP_CASE(fragment_front_face_2, FRAGMENT_FACE_2);
   PPP_CASE(fragment_front_stencil, FRAGMENT_STENCIL);
   PPP_CASE(fragment_back_face, FRAGMENT_FACE);
   PPP_CASE(fragment_back_face_2, FRAGMENT_FACE_2);
   PPP_CASE(fragment_back_stencil, FRAGMENT_STENCIL);
   PPP_CASE(depth_bias_scissor, DEPTH_BIAS_SCISSOR);

   if (present->region_clip)
      size += present->viewport_count * AGX_REGION_CLIP_LENGTH;

   if (present->viewport) {
      size += AGX_VIEWPORT_CONTROL_LENGTH +
              (present->viewport_count * AGX_VIEWPORT_LENGTH);
   }

   PPP_CASE(w_clamp, W_CLAMP);
   PPP_CASE(output_select, OUTPUT_SELECT);
   PPP_CASE(varying_counts_32, VARYING_COUNTS);
   PPP_CASE(varying_counts_16, VARYING_COUNTS);
   PPP_CASE(cull, CULL);
   PPP_CASE(cull_2, CULL_2);

   if (present->fragment_shader) {
      size +=
         AGX_FRAGMENT_SHADER_WORD_0_LENGTH + AGX_FRAGMENT_SHADER_WORD_1_LENGTH +
         AGX_FRAGMENT_SHADER_WORD_2_LENGTH + AGX_FRAGMENT_SHADER_WORD_3_LENGTH;
   }

   PPP_CASE(occlusion_query, FRAGMENT_OCCLUSION_QUERY);
   PPP_CASE(occlusion_query_2, FRAGMENT_OCCLUSION_QUERY_2);
   PPP_CASE(output_unknown, OUTPUT_UNKNOWN);
   PPP_CASE(output_size, OUTPUT_SIZE);
   PPP_CASE(varying_word_2, VARYING_2);
#undef PPP_CASE

   assert((size % 4) == 0 && "PPP updates are aligned");
   return size;
}

static inline bool
agx_ppp_validate(struct agx_ppp_update *ppp, size_t size)
{
#ifndef NDEBUG
   /* Assert that we don't overflow. Ideally we'd assert that types match too
    * but that's harder to do at the moment.
    */
   assert(((ppp->head - ppp->cpu_base) + size) <= ppp->total_size);
#endif

   return true;
}

#define agx_ppp_push(ppp, T, name)                                             \
   for (bool it = agx_ppp_validate((ppp), AGX_##T##_LENGTH); it;               \
        it = false, (ppp)->head += AGX_##T##_LENGTH)                           \
      agx_pack((ppp)->head, T, name)

#define agx_ppp_push_packed(ppp, src, T)                                       \
   do {                                                                        \
      agx_ppp_validate((ppp), AGX_##T##_LENGTH);                               \
      memcpy((ppp)->head, src, AGX_##T##_LENGTH);                              \
      (ppp)->head += AGX_##T##_LENGTH;                                         \
   } while (0)

/* The scratch words are uint32_t: agx_pack() stores through a uint32_t
 * pointer and agx_ppp_push_merged_blobs() ORs whole words, so a byte array
 * here is only 4-byte aligned when the compiler happens to place it so. */
#define agx_ppp_push_merged(ppp, T, name, merge)                               \
   for (uint32_t _tmp[DIV_ROUND_UP(AGX_##T##_LENGTH, 4)], it = 1; it;          \
        it = 0, agx_ppp_push_merged_blobs(ppp, AGX_##T##_LENGTH, _tmp,         \
                                          (uint32_t *)&merge))                 \
      agx_pack(_tmp, T, name)

ALWAYS_INLINE static struct agx_ppp_update
agx_new_ppp_update(struct agx_ptr out, size_t size,
                   struct AGX_PPP_HEADER *present)
{
   struct agx_ppp_update ppp = {
      .head = out.cpu,
      .gpu_base = out.gpu,
      .total_size = size,
      .cpu_base = out.cpu,
   };

   agx_ppp_push(&ppp, PPP_HEADER, cfg) {
      cfg = *present;
   }

   return ppp;
}

/*
 * Emit the VDM PPP State Update selected by the physical command-stream
 * generation.  The AGX2 path deliberately retains the generated legacy
 * packer.  Grounded AGX3 identities use the exact shared eight-byte core;
 * unknown and ungrounded identities fail closed.
 *
 * The cursor and pointed-to bytes are unchanged on failure.
 */
static inline bool
agx_ppp_emit_state(enum agx_chip chip, uint8_t **out, uint64_t gpu_base,
                   size_t size)
{
   if (!out || !*out || (size & 3) != 0 ||
       gpu_base >= (UINT64_C(1) << 40) || (gpu_base & UINT64_C(3)) != 0)
      return false;

   const size_t size_words = size / 4;
   if (size_words > AGX_VDM_PPP_STATE_CORE_MAX_SIZE_WORDS)
      return false;

   uint8_t staged[AGX_PPP_STATE_LENGTH];
   switch (agx_chip_command_stream_generation(chip)) {
   case AGX_CS_GENERATION_AGX2: {
      struct agx_ppp_state_packed packed;
      const struct AGX_PPP_STATE cfg = {
         .pointer_hi = gpu_base >> 32,
         .pointer_lo = (uint32_t)gpu_base,
         .size_words = (uint32_t)size_words,
      };
      AGX_PPP_STATE_pack(packed.opaque, &cfg);
      memcpy(staged, &packed, sizeof(staged));
      break;
   }

   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300: {
      struct agx_vdm_ppp_state_core packed;
      if (sizeof(staged) != sizeof(packed.bytes) ||
          !agx_vdm_ppp_state_core_pack(chip, gpu_base, (uint32_t)size_words,
                                       &packed))
         return false;

      memcpy(staged, packed.bytes, sizeof(staged));
      break;
   }

   case AGX_CS_GENERATION_UNKNOWN:
   default:
      return false;
   }

   memcpy(*out, staged, sizeof(staged));
   *out += AGX_PPP_STATE_LENGTH;
   return true;
}

/*
 * Whether payloads built with the legacy AGX_PPP_HEADER/push helpers may be
 * referenced by a live stream.  The recovered Gen6 G15/HAL200 layouts retain
 * that ordering.  HAL300 moves header selectors and some blocks, so its exact
 * outer pointer representation is intentionally not enough to admit a legacy
 * inner payload.
 */
static inline bool
agx_ppp_legacy_payload_supported(enum agx_chip chip)
{
   switch (agx_chip_command_stream_generation(chip)) {
   case AGX_CS_GENERATION_AGX2:
      return true;
   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
      return agx_vdm_ppp_state_core_supported(chip);
   case AGX_CS_GENERATION_HAL300:
   case AGX_CS_GENERATION_UNKNOWN:
   default:
      return false;
   }
}

/* Finalize one completely populated PPP update and append its state record. */
static inline bool
agx_ppp_fini(enum agx_chip chip, uint8_t **out, struct agx_ppp_update *ppp)
{
   if (!ppp || !ppp->head || !ppp->cpu_base ||
       (uintptr_t)ppp->head < (uintptr_t)ppp->cpu_base ||
       (uintptr_t)ppp->head - (uintptr_t)ppp->cpu_base != ppp->total_size)
      return false;

   return agx_ppp_emit_state(chip, out, ppp->gpu_base, ppp->total_size);
}

static void
agx_ppp_push_merged_blobs(struct agx_ppp_update *ppp, size_t length,
                          void *src1_, void *src2_)
{
   assert((length & 3) == 0);
   assert(((uintptr_t)src1_ & 3) == 0);
   assert(((uintptr_t)src2_ & 3) == 0);

   uint32_t *dst = (uint32_t *)ppp->head;
   uint32_t *src1 = (uint32_t *)src1_;
   uint32_t *src2 = (uint32_t *)src2_;

   for (unsigned i = 0; i < (length / 4); ++i) {
      dst[i] = src1[i] | src2[i];
   }

   ppp->head += length;
}

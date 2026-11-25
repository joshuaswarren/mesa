/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 VDM PPP State Update representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_PPP_STATE_CORE_LENGTH         8u
#define AGX_VDM_PPP_STATE_CORE_RESERVED_MASK  UINT32_C(0xffff0000)
#define AGX_VDM_PPP_STATE_CORE_MAX_SIZE_WORDS UINT32_C(0xff)

struct agx_vdm_ppp_state_core {
   uint8_t bytes[AGX_VDM_PPP_STATE_CORE_LENGTH];
   uint8_t length;
};

/*
 * Pack/unpack the 8-byte VDM PPP State Update record:
 *   word 0: pointer[39:32] in bits 0..7, size_words in bits 8..15,
 *           zero in bits 16..28, VDM block type 0 in bits 29..31
 *   word 1: 4-byte-aligned pointer[31:0]
 *
 * Admitted: G15G/X, G16G/X, G17P, G17G/X, G18G (J873 native renders);
 * closed: G16P, G17A (no test rows).
 */
bool agx_vdm_ppp_state_core_supported(enum agx_chip chip);

bool agx_vdm_ppp_state_core_pack(enum agx_chip chip, uint64_t pointer,
                                 uint32_t size_words,
                                 struct agx_vdm_ppp_state_core *out);

bool agx_vdm_ppp_state_core_unpack(enum agx_chip chip, const void *bytes,
                                   size_t size, uint64_t *pointer,
                                   uint32_t *size_words);

#ifdef __cplusplus
}
#endif

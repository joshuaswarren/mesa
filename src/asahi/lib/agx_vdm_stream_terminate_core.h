/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Bounded, target-aware AGX3 VDM Stream Terminate representation.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_STREAM_TERMINATE_CORE_LENGTH 4u
#define AGX_VDM_STREAM_TERMINATE_CORE_WORD   UINT32_C(0xc0000000)

struct agx_vdm_stream_terminate_core {
   uint8_t bytes[AGX_VDM_STREAM_TERMINATE_CORE_LENGTH];
   uint8_t length;
};

/* Pack/unpack the 4-byte type-6 VDM Stream Terminate record (0xc0000000);
 * type-7 Stream Return is not accepted. */
bool agx_vdm_stream_terminate_core_supported(enum agx_chip chip);

bool
agx_vdm_stream_terminate_core_pack(enum agx_chip chip,
                                   struct agx_vdm_stream_terminate_core *out);

bool agx_vdm_stream_terminate_core_unpack(enum agx_chip chip, const void *bytes,
                                          size_t size);

#ifdef __cplusplus
}
#endif

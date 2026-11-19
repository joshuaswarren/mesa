/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_chip.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_IOTO_PROFILE_CONTROL_MAX_SIZE 6
#define AGX_IOTO_G17S_MINIMAL_PROFILE_LOADER_SIZE 52

/*
 * Bounded canonical subset of Ioto SETPROFILECTL_0.  payload0/payload1 hold
 * already-positioned operand bits, not complete encoded halfwords.  Packing
 * rejects every bit outside the recovered field scatters instead of emitting
 * values into fixed or unattributed positions.
 */
struct agx_ioto_profile_control {
   enum agx_chip chip;
   uint16_t payload0;
   uint16_t payload1;
};

bool agx_ioto_profile_control_supported(enum agx_chip chip);

size_t agx_ioto_profile_control_size(enum agx_chip chip);

bool
agx_ioto_profile_control_pack(const struct agx_ioto_profile_control *profile,
                              uint8_t out[AGX_IOTO_PROFILE_CONTROL_MAX_SIZE]);

/* Exact minimal-profile loader emitted by G17S after the state tail.  A
 * zero-payload SETPROFILECTL is not an equivalent encoding. */
bool agx_ioto_g17s_minimal_profile_loader_pack(
   void *output, size_t capacity);

#ifdef __cplusplus
}
#endif

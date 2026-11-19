/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 ray-traversal barrier: the composer-driven encoder for the barrier
 * that brackets the G17P traversal loop (6 bytes on both AGX3b and AGX3c).
 *
 * The intersect_ray and continue_ray instructions the compiler emits are
 * encoded by agx3_rt_gen1_traversal.c and agx3_rt_m5_core.c, not here.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Output buffer size callers reserve for one encoding. */
#define AGX3_RT_CORE_MAX_LENGTH 14u

/* Both barrier scopes witnessed around the G17P traversal.  Scope depends
 * on control-flow position, not RT opcode: a continue_ray at loop re-entry
 * uses ENTRY, while a continue_ray in the spin body uses IN_LOOP.  The same
 * scopes occur before the SCS epilogue and shader end respectively.
 * No wider memory semantics are inferred.
 */
enum agx3_rt_barrier_scope {
   AGX3_RT_BARRIER_IN_LOOP = 1,
   AGX3_RT_BARRIER_ENTRY = 2,
};

bool agx3_rt_traversal_barrier(enum agx_isa_encoding enc,
                                enum agx3_rt_barrier_scope scope,
                                uint8_t out[AGX3_RT_CORE_MAX_LENGTH],
                                uint8_t *len);

#ifdef __cplusplus
}
#endif

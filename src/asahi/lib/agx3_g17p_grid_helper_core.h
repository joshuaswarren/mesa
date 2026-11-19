/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_ioto_state_core.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * G17P native indirect compute: the grid helper and its geometry contract.
 *
 * A native A18 Pro indirect dispatch is a pair of CDM records (see
 * agx3_g17p_native_indirect_compute_dispatch_append):
 *
 *   1. a 0x2c direct helper launch with one invocation, and
 *   2. a compact 0x1c application launch in indirect-local mode, naming the
 *      address of a six-u32 geometry block.
 *
 * The helper publishes geometry before the compact application record
 * consumes it. Bit 23 of the direct record is a program-template field,
 * not an indirect-mode flag. Swapping these programs runs the application
 * with the helper's inline dimensions and leaves geometry unproduced.
 *
 * GEOMETRY LAYOUT.  Six little-endian u32:
 *
 *     +0x00  global_size[0]      (threadgroups[0] * local_size[0])
 *     +0x04  global_size[1]
 *     +0x08  global_size[2]
 *     +0x0c  local_size[0]
 *     +0x10  local_size[1]
 *     +0x14  local_size[2]
 *
 * That is positionally identical to the direct launch record's own inline
 * fields (global at +0x10..+0x1b, local at +0x1c..+0x27), and it is what the
 * only known G17P grid helper actually writes.  The first three words are
 * therefore GLOBAL THREAD counts, not threadgroup counts; the helper is what
 * performs the multiply.
 */
#define AGX3_G17P_INDIRECT_GEOMETRY_WORDS 6u
#define AGX3_G17P_INDIRECT_GEOMETRY_SIZE  (AGX3_G17P_INDIRECT_GEOMETRY_WORDS * 4u)

/* Byte offsets of the two halves within the block, so a writer names the field
 * it means instead of deriving it from the total size. */
#define AGX3_G17P_INDIRECT_GEOMETRY_LOCAL_OFFSET  12u

/* agx3_g17p_native_indirect_compute_dispatch_append() splits the geometry
 * address into two u32 and requires a 4-byte-aligned address; the helper
 * writes whole u32 words, so this is the real hardware requirement. */
#define AGX3_G17P_INDIRECT_GEOMETRY_ALIGNMENT 4u

/* The LDSHDR "f1" program kinds observed in the two output-positive G17P
 * indirect witnesses.  A direct dispatch announces DIRECT_ADD3; the indirect
 * pair announces INDIRECT_ADD3 for the main program and GRID_HELPER for the
 * geometry helper program. These are witness names, not decoded semantics. */
#define AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1                                      \
   AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_INDIRECT_ADD3
#define AGX3_G17P_GRID_HELPER_LDSHDR_F1                                        \
   AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_GRID_HELPER

#ifdef __cplusplus
}
#endif

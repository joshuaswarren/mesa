/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "agx3_mem_core.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_CPROG_INPUT_HALVES 512u
#define AGX3_CPROG_INPUT_WORDS 8u
#define AGX3_CPROG_INPUT_TPS_BASE 8u
#define AGX3_CPROG_INPUT_COPY_MAX_SIZE 20u

/* Native non-fast input ABI. Selected uniform halves are compacted, in
 * ascending order, into TPS starting at half 8. This is import storage only;
 * it does not include a preamble's own spills. The native GPR fast-input
 * protocol is distinct and must not use this layout. */
struct agx3_cprog_inputs {
   uint64_t halves[AGX3_CPROG_INPUT_WORDS];
   uint16_t count;
   uint16_t storage_bytes;
};

bool agx3_cprog_inputs_init(const uint64_t halves[AGX3_CPROG_INPUT_WORDS],
                          struct agx3_cprog_inputs *out);

/* inputs must be initialized by agx3_cprog_inputs_init. Returns false for a
 * half that is not an input. Output is unchanged. */
bool agx3_cprog_input_tps_half(const struct agx3_cprog_inputs *inputs,
                             unsigned uniform_half, uint16_t *tps_half);

/* Publish selected halves of one loaded uniform dword to input TPS. The
 * preceding uniform COPY must preserve any source halves published here.
 * Native copyGprToCprog uses signal slot 6 and consumes each selected source
 * half. A pair at an odd compacted position is two 16-bit stores. Empty
 * selection emits nothing. Output and written are unchanged on failure. */
bool agx3_cprog_input_publish(const struct agx3_cprog_inputs *inputs,
                             unsigned uniform_half, unsigned source_half,
                             void *output, size_t capacity, size_t *written);

/* Chip-aware forms preserve the classic ABI and encode M5's native TPS
 * COPY/RLD forms. M5 capture uses slot6 and compact8B extended COPY. */
bool agx3_cprog_input_publish_at_encoding(
   enum agx_isa_encoding encoding, unsigned mask, unsigned tps_half,
   unsigned source_half, void *output, size_t capacity, size_t *written);

struct agx3_cprog_import {
   uint16_t uniform_half;
   uint16_t dest_half;
   enum agx3_mem_width width;
   uint8_t slot;
   uint8_t slot_mirror;
   uint16_t wait_mask;
   bool tps_live;
   /* Preserve the native structural field without attributing its currently
    * unresolved policy to CProg mode: native CProg witnesses use both values. */
   bool structural_flag;
};

/* Import one native 16/32-bit value from the same compact layout as publish.
 * A 32-bit import requires both selected halves and an aligned compact word.
 * Waits are supplied by the scoreboard, including the loader's entry slot.
 * Output and written are unchanged on any failure. */
bool agx3_cprog_input_import_encoding(
   enum agx_isa_encoding encoding, const struct agx3_cprog_inputs *inputs,
   const struct agx3_cprog_import *desc, void *output, size_t capacity,
   size_t *written);

#ifdef __cplusplus
}
#endif

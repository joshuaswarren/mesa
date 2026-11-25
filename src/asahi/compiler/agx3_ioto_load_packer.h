/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact AGX3 EslBuilder direct device-load record packer.
 *
 * This is intentionally narrower than a general shader-ISA encoder.  It maps
 * the fixed Register forms queued by Apple's G15/G16/G17P and G17G/S/C
 * EslBuilder implementations to native LDDEV/COPY records and supplies them
 * to the bounded agx_ioto_load_core scheduler.
 */

#pragma once

#include "asahi/lib/agx_ioto_load_core.h"
#include "agx3_mem_core.h"
#include "agx3_cprog_input.h"

#ifdef __cplusplus
extern "C" {
#endif

struct agx3_ioto_load_packer {
   enum agx_chip chip;
   enum agx_ioto_load_encoding encoding;
   bool copy_wait_emitted;
   /* Optional known input set, borrowed for the complete load transaction.
    * Qualified for G17P with the default allocator and round NOPs only. */
   const struct agx_ioto_cprog_inputs *cprog_inputs;
};

/* Native G17P copyGprToCprog with encoder +0x100 clear. The destination is
 * a packed byte offset, not a uniform index. Mask bits select 16-bit halves
 * of one aligned source dword. This is the last use of each selected half;
 * the emitted TPS stores kill their sources and signal scoreboard slot 6.
 * source_wait_slot is a zero-based slot (0..5), or -1 when already ready.
 * Storage allocation, entry reloads, and prelude policy belong to the caller.
 */
struct agx3_ioto_cprog_copy_desc {
   uint16_t destination_bytes;
   uint16_t source_half;
   uint8_t lane_mask;
   int8_t source_wait_slot;
};

#define AGX3_IOTO_CPROG_COPY_MAX_SIZE 20u

struct agx3_ioto_cprog_copy_output {
   uint8_t bytes[AGX3_IOTO_CPROG_COPY_MAX_SIZE];
   uint8_t length;
   uint8_t written_bytes;
};

/* Output is unchanged on failure. This encodes the native TPS subset only;
 * it does not enable preambles or qualify their implicit execution state. */
bool agx3_ioto_pack_cprog_copy(
   enum agx_chip chip, const struct agx3_ioto_cprog_copy_desc *desc,
   struct agx3_ioto_cprog_copy_output *output);

bool agx3_ioto_load_packer_init(struct agx3_ioto_load_packer *packer,
                                enum agx_chip chip);

/* True only for the exact direct-load register subset implemented here.  This
 * is not the broad production-readiness gate for arbitrary resources. */
bool agx3_ioto_load_absolute_packer_ready(enum agx_chip chip);

void agx3_ioto_load_packer_ops(struct agx3_ioto_load_packer *packer,
                               struct agx_ioto_load_emit_ops *ops);

bool
agx3_ioto_load_finish_round_native(struct agx_ioto_load_core *core,
                                   bool complete_round,
                                   struct agx_ioto_load_workspace workspace);

bool
agx3_ioto_load_absolute_native(struct agx_ioto_load_core *core,
                               const struct agx_ioto_load_absolute *load,
                               struct agx_ioto_load_workspace workspace,
                               struct agx_ioto_load_allocation *allocation);

bool agx3_ioto_load_indirect_uniform_range_cprog(
   enum agx_chip chip, uint16_t root_base_half, bool root_is_uniform,
   int16_t pointer_offset, uint16_t uniform_half, uint16_t size_halfs,
   const struct agx3_cprog_inputs *inputs, void *output, size_t capacity,
   size_t *written);

bool
agx3_ioto_load_from_scs_native(struct agx_ioto_load_core *core,
                               const struct agx_ioto_load_from_fixed *load,
                               struct agx_ioto_load_workspace workspace,
                               struct agx_ioto_load_allocation *allocation);

bool agx3_ioto_load_from_abi_buffer_native(
   struct agx_ioto_load_core *core, enum agx_ioto_abi_buffer_slot source,
   const struct agx_ioto_load_from_fixed *load,
   struct agx_ioto_load_workspace workspace,
   struct agx_ioto_load_allocation *allocation);

bool agx3_ioto_load_indirect_uniform_range(
   enum agx_chip chip, uint16_t root_base_half, int16_t pointer_offset,
   uint16_t uniform_half, uint16_t size_halfs, void *output, size_t capacity,
   size_t *written);

bool agx3_ioto_load_indirect_uniform_range_from_uniform_root(
   enum agx_chip chip, uint16_t root_base_half, int16_t pointer_offset,
   uint16_t uniform_half, uint16_t size_halfs, void *output, size_t capacity,
   size_t *written);

#ifdef __cplusplus
}
#endif

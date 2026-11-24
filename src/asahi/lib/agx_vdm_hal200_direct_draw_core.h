/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_VDM_HAL200_DRAW_STATE_POINTER_LENGTH 4u
#define AGX_VDM_HAL200_DIRECT_DRAW_BODY_LENGTH 16u
#define AGX_VDM_HAL200_DIRECT_DRAW_LENGTH \
   (AGX_VDM_HAL200_DRAW_STATE_POINTER_LENGTH + \
    AGX_VDM_HAL200_DIRECT_DRAW_BODY_LENGTH)
#define AGX_VDM_HAL200_DIRECT_DRAW_OPCODE UINT32_C(0x61c4)

struct agx_vdm_hal200_direct_draw_fields {
   uint32_t draw_state_context_offset;
   uint8_t primitive;
   uint32_t vertex_count;
   uint32_t instance_count;
   uint32_t vertex_start;
};

struct agx_vdm_hal200_direct_draw {
   uint8_t bytes[AGX_VDM_HAL200_DIRECT_DRAW_LENGTH];
   uint8_t length;
};

struct agx_vdm_hal200_direct_draw_body {
   uint8_t bytes[AGX_VDM_HAL200_DIRECT_DRAW_BODY_LENGTH];
   uint8_t length;
};

/* Pack the context-relative PPP pointer plus the direct, non-indexed draw.
 * Stream Terminate is a separate four-byte record emitted only when the
 * complete control stream is closed, not after each draw.
 */
bool agx_vdm_hal200_direct_draw_pack(
   enum agx_chip chip, const struct agx_vdm_hal200_direct_draw_fields *fields,
   struct agx_vdm_hal200_direct_draw *out);

/* Pack the native body used when PPP/render state was emitted separately. */
bool agx_vdm_hal200_direct_draw_body_pack(
   enum agx_chip chip, const struct agx_vdm_hal200_direct_draw_fields *fields,
   struct agx_vdm_hal200_direct_draw_body *out);

/* Metal emits the leading PPP selector as part of render-state publication.
 * When state is unchanged, only the native draw body is legal in the stream.
 * Copy the correct slice of an already validated packed draw atomically. */
bool agx_vdm_hal200_draw_emit(enum agx_chip chip, const void *packed,
                             size_t packed_size, bool render_state_emitted,
                             void *stream, size_t stream_capacity,
                             size_t *emitted_size);

/* Finish the preceding PPP pointer, publish native restart state, then the
 * 28-byte direct indexed record. Primitive, native index width, restart and
 * instance count remain ordinary hardware fields. Base instance is supplied
 * through the vertex uniform ABI, independently of this record. */
#define AGX_VDM_HAL200_INDEXED_DRAW_BODY_LENGTH 36u
#define AGX_VDM_HAL200_INDEXED_DRAW_LENGTH \
   (AGX_VDM_HAL200_DRAW_STATE_POINTER_LENGTH + \
    AGX_VDM_HAL200_INDEXED_DRAW_BODY_LENGTH)

struct agx_vdm_hal200_indexed_draw_fields {
   uint8_t primitive;
   uint8_t index_size_B;
   bool restart_enable;
   uint32_t restart_index;
   uint32_t draw_state_context_offset;
   uint64_t index_buffer;
   uint64_t index_buffer_range_B;
   uint32_t first_index;
   uint32_t index_count;
   uint32_t instance_count;
   int32_t base_vertex;
};

struct agx_vdm_hal200_indexed_draw {
   uint8_t bytes[AGX_VDM_HAL200_INDEXED_DRAW_LENGTH];
   uint8_t length;
};

struct agx_vdm_hal200_indexed_draw_body {
   uint8_t bytes[AGX_VDM_HAL200_INDEXED_DRAW_BODY_LENGTH];
   uint8_t length;
};

/* Checked relocation and native dword-range-minus-one conversion. */
bool agx_vdm_hal200_index_range(uint64_t address, uint64_t range_B,
                                uint32_t first_index, uint32_t count,
                                uint8_t index_size_B, uint64_t *relocated,
                                uint32_t *size_word);

bool agx_vdm_hal200_indexed_draw_pack(
   enum agx_chip chip, const struct agx_vdm_hal200_indexed_draw_fields *fields,
   struct agx_vdm_hal200_indexed_draw *out);

bool agx_vdm_hal200_indexed_draw_body_pack(
   enum agx_chip chip, const struct agx_vdm_hal200_indexed_draw_fields *fields,
   struct agx_vdm_hal200_indexed_draw_body *out);

/* Native HAL200 indirect draws use the same context-relative PPP selector as
 * direct draws, followed by the independently recovered G17P indirect Index
 * List record. */
#define AGX_VDM_HAL200_INDIRECT_NONINDEXED_DRAW_LENGTH 16u
#define AGX_VDM_HAL200_INDIRECT_INDEXED_DRAW_LENGTH 36u

struct agx_vdm_hal200_indirect_draw_fields {
   uint32_t draw_state_context_offset;
   uint8_t primitive;
   bool indexed;
   uint8_t index_size_B;
   bool restart_enable;
   uint32_t restart_index;
   /* Native absolute 48-bit address. */
   uint64_t index_buffer;
   uint64_t index_buffer_range_B;
   /* Native absolute 48-bit argument-buffer address. */
   uint64_t indirect_buffer;
};

struct agx_vdm_hal200_indirect_draw {
   uint8_t bytes[AGX_VDM_HAL200_INDIRECT_INDEXED_DRAW_LENGTH];
   uint8_t length;
};

bool agx_vdm_hal200_indirect_draw_pack(
   enum agx_chip chip, const struct agx_vdm_hal200_indirect_draw_fields *fields,
   struct agx_vdm_hal200_indirect_draw *out);

#ifdef __cplusplus
}
#endif

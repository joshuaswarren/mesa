/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Packing of the AGX3 Ioto SPECLM/LDSHDR/SPECTPR state tail of G17P compute
 * and graphics programs, by Apple's anonymous field ids.
 */

#pragma once

#include "agx_chip.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_IOTO_STATE_FIELD_COUNT 27u
#define AGX_IOTO_SPECLM_SIZE 8u
#define AGX_IOTO_LDSHDR_SIZE 10u
#define AGX_IOTO_SPECTPR_SIZE 10u
#define AGX_IOTO_STATE_TAIL_TYPE0_SIZE 28u
#define AGX_IOTO_STATE_TAIL_TYPE1_SIZE 20u
#define AGX_IOTO_STATE_TAIL_MAX_SIZE AGX_IOTO_STATE_TAIL_TYPE0_SIZE

/* Anonymous logical fields f0 through f26 from Apple's encoder graph. */
struct agx_ioto_state {
   uint64_t f[AGX_IOTO_STATE_FIELD_COUNT];
};

bool agx_ioto_state_pack_speclm(enum agx_chip chip,
                              const struct agx_ioto_state *state,
                              void *output, size_t capacity);
bool agx_ioto_state_pack_ldshdr(enum agx_chip chip,
                              const struct agx_ioto_state *state,
                              void *output, size_t capacity);

bool agx_ioto_state_pack_spectpr(enum agx_chip chip,
                                 const struct agx_ioto_state *state,
                                 void *output, size_t capacity);

enum agx_ioto_state_tail_type {
   AGX_IOTO_STATE_TAIL_TYPE0 = 0,
   AGX_IOTO_STATE_TAIL_TYPE1 = 1,
};

struct agx_ioto_state_tail {
   enum agx_ioto_state_tail_type type;
   struct agx_ioto_state speclm;
   struct agx_ioto_state ldshdr;
   struct agx_ioto_state spectpr;
};

struct agx_ioto_state_tail_output {
   size_t length;
   uint8_t bytes[AGX_IOTO_STATE_TAIL_MAX_SIZE];
};

/* Physical imageblock geometry consumed by HAL200's fragment LoadShader
 * transform.  tile_size_px is one of Mesa's three rectangular tile areas;
 * 32x16 is the only 512-pixel spelling. */
struct agx_ioto_g17p_imageblock {
   uint8_t sample_size_B;
   uint8_t nr_samples;
   uint16_t tile_size_px;
};

#define AGX_IOTO_G17P_IMAGEBLOCK_8B_1X_32X32                                \
   { .sample_size_B = 8, .nr_samples = 1, .tile_size_px = 32 * 32 }

/* Build the exact fragment SPECLM record emitted by HAL200's
 * LdShdrTransformFragment for the supplied physical imageblock.  Unlike the
 * generic anonymous-field packer, this retains the recovered dimension bits
 * which are still opaque to the field graph.  `output` is unchanged on every
 * failure. */
bool agx_ioto_g17p_fragment_speclm_build(
   const struct agx_ioto_g17p_imageblock *imageblock, void *output,
   size_t capacity);

/*
 * Type 0 packs SPECLM, LDSHDR, SPECTPR in that order.  Type 1 packs only
 * LDSHDR and SPECTPR and requires every SPECLM field to be zero.  Other type
 * values fail.  `output` is unchanged on every failure.
 */
bool agx_ioto_state_tail_pack(
   enum agx_chip chip, const struct agx_ioto_state_tail *tail,
   struct agx_ioto_state_tail_output *output);

/* Exact type-0 state tail from the live G17S minimal root-uniform/no-texture
 * compute path.  No shader-info counts are inferred into the zero payload
 * fields.  `tail` is unchanged on failure. */
bool agx_ioto_g17s_minimal_compute_tail_build(
   uint64_t shader_va, struct agx_ioto_state_tail *tail);

/* Pack the minimal G17P LDSHDR used by compiler-generated graphics and
 * compute programs directly. This is equivalent to the semantic state with
 * only f0=shader_va, f1=fetch_span and required f23=1 populated. `output` is
 * unchanged on failure. */
bool agx_ioto_g17p_compiler_ldshdr_pack(
   uint64_t shader_va, uint32_t fetch_span, void *output, size_t capacity);

/* Raw LDSHDR f1 values from three explicitly MIT-licensed, output-positive
 * G17P compute witnesses. The names identify those witnesses; they do not
 * assign inferred hardware semantics to the anonymous field. */
enum agx_ioto_g17p_compute_ldshdr_f1 {
   AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3 = 0,
   AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_GRID_HELPER = 1,
   AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_INDIRECT_ADD3 = 3,
};

/* Build the exact packed type-0 runtime compute tail shared by the witnessed
 * direct-add3, grid-helper and indirect-add3 programs. Only their observed f1
 * values are accepted. `output` is unchanged on failure. */
bool agx_ioto_g17p_runtime_compute_tail_build(
   uint64_t shader_va, enum agx_ioto_g17p_compute_ldshdr_f1 ldshdr_f1,
   struct agx_ioto_state_tail_output *output);

/* Build the same runtime compute tail with a statically allocated
 * threadgroup-memory region. The admitted byte range and 64-byte granularity
 * are the exact A18 Pro static-allocation sweep; zero remains the separate
 * no-threadgroup-memory form above. */
bool agx_ioto_g17p_runtime_compute_tail_build_shared(
   uint64_t shader_va, enum agx_ioto_g17p_compute_ldshdr_f1 ldshdr_f1,
   uint32_t threadgroup_memory_size_B,
   struct agx_ioto_state_tail_output *output);

bool agx_ioto_g17p_compiler_spectpr_build(
   uint32_t spill_bytes, uint32_t d_value, struct agx_ioto_state *state);

/* Pack those same compiler fields directly into the exact G17P SPECTPR
 * record. This is equivalent to build + generic pack without materializing a
 * 27-field semantic state. `output` is unchanged on failure. */
bool agx_ioto_g17p_compiler_spectpr_pack(
   uint32_t spill_bytes, uint32_t d_value, void *output, size_t capacity);

bool agx_ioto_state_pack_speclm(enum agx_chip chip,
                                const struct agx_ioto_state *state,
                                void *output, size_t capacity);

bool agx_ioto_state_pack_ldshdr(enum agx_chip chip,
                                const struct agx_ioto_state *state,
                                void *output, size_t capacity);

#ifdef __cplusplus
}
#endif

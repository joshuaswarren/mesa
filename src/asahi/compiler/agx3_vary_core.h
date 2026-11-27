/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 varying and interpolation encoders:
 *   iter -> f32interp (linear; perspective divides by interpolated RHW in SW),
 *   at-sample / centroid -> itrcoords + f32interp,
 *   ldcf -> ldcfr (<A,B,C> coefficient load), st_vary -> uvsw (32-bit write).
 * Scheduling/hazard marker fields are emitted zero (table-valid; unproven on
 * hardware).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_VARY_MAX_LENGTH 10u

enum agx3_vary_location {
   AGX3_VARY_CENTER = 0,
   AGX3_VARY_CENTROID = 1,
};

/* Interpolate coefficient `coef_index` into `dest`. `perspective` selects the
 * perspective-correct linear iterator (the caller pairs it with the software
 * rcp/mul sequence). */
bool agx3_vary_interp_at(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t coef_index, bool perspective,
                         enum agx3_vary_location location,
                         uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

/* Native G17P uses f63 as the shared fragment-input entry marker for a plain
 * varying-only program. Sampled and derivative-capable fragments leave it
 * clear to retain their helper-lane protocol. LDCF f48 is the alternate
 * encoding of the same marker, so a later first operation from the other
 * family must not restart the part. This is independent of f54, whose
 * optimization/liveness role remains unproven and is deliberately clear. */
/* G17P center RHW used only by the coefficient-tagged projective reciprocal.
 * Native pure smooth and smooth+position.z witnesses set f54(bit50); ordinary
 * position.w/output/arithmetic witnesses clear it. Numerical semantics beyond
 * that use domain are not established. `first` independently controls f63. */
bool agx3_vary_projective_rhw(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t coef_index, bool first,
                             uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

/* Explicit at-sample interpolation is witnessed on G17P/G17G/G18G. ITRCOORDS
 * consumes a 16-bit sample index and produces an even-aligned 32-bit
 * coordinate token. `sample_index_kill` selects the native last-use bit and
 * `wait_mask` waits for asynchronous producers of the sample index. */
bool agx3_vary_sample_coords(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t sample_index, bool sample_index_kill,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

/* Centroid interpolation computes the centroid of the covered samples
 * with ITRCOORDS from the native 16-bit SR39 coverage word, then consumes the
 * token through the same leaves as explicit sample interpolation. */
bool agx3_vary_centroid_coords(enum agx_isa_encoding enc, uint16_t dest,
                               uint16_t coverage, bool coverage_kill,
                               uint16_t wait_mask,
                               uint8_t out[AGX3_VARY_MAX_LENGTH],
                               uint8_t *len);

/* Consume an ITRCOORDS token in the sample-coordinate f32interp leaf.
 * `coords_kill` must be false while later scalar interpolations still need
 * the token and true on its final use. */
bool agx3_vary_interp_sample(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t coef_index, bool perspective,
                             uint16_t coords, bool coords_kill,
                             uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

/* Projective RHW variant of the explicit sample-coordinate leaf. */
bool agx3_vary_projective_rhw_sample(enum agx_isa_encoding enc, uint16_t dest,
                                     uint16_t coef_index, uint16_t coords,
                                     bool coords_kill,
                                     uint8_t out[AGX3_VARY_MAX_LENGTH],
                                     uint8_t *len);

bool agx3_vary_interp_at_start(enum agx_isa_encoding enc, uint16_t dest,
                               uint16_t coef_index, bool perspective,
                               enum agx3_vary_location location, bool first,
                               uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

/* Load the raw coefficient triple for `coef_index` into `dest`, on scoreboard
 * slot `slot` (1..6/1..8). */
/* Native G17P sets f48 on a plain program's first LDCF; see f63 above. */
bool agx3_vary_ldcf_at_start(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t coef_index, uint8_t slot, bool first,
                            uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

bool agx3_vary_ldcf(enum agx_isa_encoding enc, uint16_t dest,
                    uint16_t coef_index, uint8_t slot,
                    uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

/* Write `value` to unified-vertex-store index `uvs_index`, on slot `slot`.
 * `value_kill` marks the last write of that register. `wait_mask` bit i waits
 * scoreboard slot i. */
bool agx3_vary_uvsw(enum agx_isa_encoding enc, uint16_t value,
                    uint16_t uvs_index, uint8_t slot, bool value_kill,
                    uint16_t wait_mask, uint8_t out[AGX3_VARY_MAX_LENGTH],
                    uint8_t *len);

/* The same write with the unified-vertex-store index read from the 16-bit
 * register half `index`, as an unlinked vertex shader computes it. Apple's
 * AGX3b/c/d tables define this operand form; AGX3d accepts at most one wait. */
bool agx3_vary_uvsw_register_index(enum agx_isa_encoding enc, uint16_t value,
                                   uint16_t index, uint8_t slot,
                                   bool value_kill, uint16_t wait_mask,
                                   uint8_t out[AGX3_VARY_MAX_LENGTH],
                                   uint8_t *len);

/* The same write with the index read from the 16-bit uniform half
 * `uniform`, so an unlinked vertex shader's per-draw index needs no copy
 * into a register. AGX3b only. */
bool agx3_vary_uvsw_uniform_index(enum agx_isa_encoding enc, uint16_t value,
                                  uint16_t uniform, uint8_t slot,
                                  bool value_kill, uint16_t wait_mask,
                                  uint8_t out[AGX3_VARY_MAX_LENGTH],
                                  uint8_t *len);

bool agx3_vary_interp(enum agx_isa_encoding enc, uint16_t dest,
                      uint16_t coef_index, bool perspective,
                      uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len);

/* One immediate-index UVSW of a vertex program whose user varyings a linker
 * places: the program is compiled without its fragment shader, writes each
 * varying at a placeholder index (agx_unlinked_uvs_layout::relocatable), and
 * the link re-encodes the write with the index of the bound fragment shader's
 * layout. The write's other operands are kept. Such writes are always the
 * last use of their staging value. Entries are stored byte-packed in the
 * shader binary (agx_shader_info::agx3_uvs_relocs_offset), so read them with
 * memcpy.
 */
struct agx3_uvs_reloc {
   /* Of the UVSW, in bytes from the start of the main program */
   uint32_t offset;

   /* Placeholder UVS index encoded there */
   uint8_t index;

   /* Staging half read, scoreboard slot and wait mask, as encoded */
   uint8_t value;
   uint8_t slot;
   uint8_t wait;
};

#define AGX3_UVS_RELOC_SIZE 8u
#define AGX3_UVS_RELOC_MAP_SIZE 256u

/* Re-encode the `count` relocated UVSWs listed at `relocs` in `code`, the
 * `size` bytes of the program they were recorded in, each with the index
 * map[placeholder]. Every write must still hold its placeholder encoding.
 * All writes are checked before any changes, so on failure `code` is
 * unchanged. A NULL map only checks.
 */
bool agx3_vary_relocate_uvsw(enum agx_isa_encoding enc, uint8_t *code,
                             size_t size, const uint8_t *relocs,
                             unsigned count,
                             const uint8_t map[AGX3_UVS_RELOC_MAP_SIZE]);

#ifdef __cplusplus
}
#endif

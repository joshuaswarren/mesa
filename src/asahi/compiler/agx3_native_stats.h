/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#ifndef AGX3_NATIVE_STATS_H
#define AGX3_NATIVE_STATS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Counts of native instructions the AGX3 packer emits beyond the IR it is
 * given. Shader statistics count agx_instr before packing, so none of these
 * show up there otherwise.
 *
 *  wait_carriers   identity instructions emitted only to carry a scoreboard
 *                  wait (agx3_band_carrier() and the in-place identities);
 *  cf_drains       control-flow instructions before which outstanding
 *                  asynchronous results the instruction does not read were
 *                  retired: slot assignment does not carry state across a
 *                  merge or a loop (agx3_slot_cf_kind());
 *  loop_cf_drains  the subset of cf_drains inside a loop;
 *  staging_copies  copies into the staging file (texture coordinates,
 *                  image and tile payloads, vertex outputs).
 *
 * The packer resets the counts at the start of each agx3_pack_binary() call
 * and they describe the last call on this thread, as agx3_last_pack_refusal()
 * does: shaders compile concurrently on different threads.
 */
struct agx3_native_stats {
   uint32_t wait_carriers;
   uint32_t cf_drains;
   uint32_t loop_cf_drains;
   uint32_t staging_copies;
};

extern __thread struct agx3_native_stats agx3_native_stats;

#ifdef __cplusplus
}
#endif

#endif

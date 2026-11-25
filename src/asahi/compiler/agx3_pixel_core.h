/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 pixel writeout barriers (`wait_pix` -> `pixwait`, `signal_pix` ->
 * `pixrel`). Both carry the AGX2 writeout-class bitmask (1 = depth/coverage,
 * 4/8/12/... = colour classes); unrecognized masks fail closed.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* pixwait and pixrel are both six bytes on AGX3b and AGX3c. */
#define AGX3_PIXEL_MAX_LENGTH 6u

/* Complete native G17P colour-partition class corpus. */
#define AGX3_G17P_MAX_COLOUR_PARTITIONS 3u

/*
 * Open the colour-tile protocol. This is the f14=512, f15=1 pixwait form
 * that precedes the ordinary class-12 colour wait. It is byte-exactly
 * witnessed on G17P (AGX3b) and the M5 native flat-colour fragment (AGX3c).
 * AGX2 is refused. Encoding this gate does not qualify other M5 transactions.
 */
bool agx3_pixel_colour_entry(enum agx_isa_encoding enc,
                             uint8_t out[AGX3_PIXEL_MAX_LENGTH], uint8_t *len);
/*
 * The same entry gate with f34 set: 077000050000 on AGX3c. Every native G17G
 * fragment that writes depth or stencil alone at several samples opens with
 * it, and exactly those never release the depth/coverage class explicitly
 * (no pixrel 1); every other witnessed fragment uses the plain gate. What
 * the bit selects in hardware is not decoded. AGX3c only.
 */
bool agx3_pixel_colour_entry_zs_unreleased(enum agx_isa_encoding enc,
                                           uint8_t out[AGX3_PIXEL_MAX_LENGTH],
                                           uint8_t *len);

/*
 * Pixel gate for a colourless fragment's returning memory atomic: acquires
 * pixel class 4 without the colour selector (f14=4, no f34; colour-writing
 * fragments acquire partition 0's write class instead). Void atomics and plain
 * stores need neither. AGX3B only.
 */
bool agx3_pixel_atomic_acquire(enum agx_isa_encoding enc,
                               uint8_t out[AGX3_PIXEL_MAX_LENGTH],
                               uint8_t *len);
bool agx3_pixel_atomic_barrier(enum agx_isa_encoding enc,
                               uint8_t out[AGX3_PIXEL_MAX_LENGTH],
                               uint8_t *len);

/*
 * Return the read, write, and combined pixel classes for one colour-tile
 * partition.  G17P assigns two class bits per partition: RT0 uses 8/4/12,
 * RT1 uses 32/16/48, and RT2 uses 128/64/192.  These are the complete
 * partition classes witnessed by native multi-RT shaders; higher partitions
 * fail closed.
 */
bool agx3_pixel_colour_partition_classes(unsigned partition,
                                         uint32_t *read_class,
                                         uint32_t *write_class,
                                         uint32_t *combined_class);

/*
 * Wait for writeout classes in `writeout` (accepted: 1 for depth/coverage,
 * and 4, 8, 12, 16, 32, 48, 60, 64, 128, 192 for colour classes; shader-entry
 * gate f14=512/f15=1 uses agx3_pixel_colour_entry()). `wait_mask` uses 1 bit
 * per slot from slot 1 (6 slots on AGX3B, 8 on AGX3C).
 */
bool agx3_pixel_wait(enum agx_isa_encoding enc, uint32_t writeout,
                     uint16_t wait_mask, uint8_t out[AGX3_PIXEL_MAX_LENGTH],
                     uint8_t *len);

/*
 * Release writeout classes (`(last ? 0x200 : 0) | writeout`). Accepted
 * `writeout` values: {0, 1, 12, 48, 60, 192}; single-class colour masks used
 * only by `pixwait` are refused here.
 */
bool agx3_pixel_release(enum agx_isa_encoding enc, uint32_t writeout, bool last,
                        uint16_t wait_mask, uint8_t out[AGX3_PIXEL_MAX_LENGTH],
                        uint8_t *len);

#ifdef __cplusplus
}
#endif

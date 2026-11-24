/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 coverage and depth-stencil writeout encoders: `movmsk` and `depthf`
 * (6 bytes on AGX3B, 8 bytes on AGX3C).
 *
 * Both take a staged half-register operand (f9=5, f5=index, f13=kill),
 * scoreboard slot (f12), and per-slot wait mask (f17..f22 on AGX3B, f17..f24
 * on AGX3C). `depthf` adds depth/stencil enable bits f25/f26 (plus AGX3C mode
 * f35: 1=depth, 2=stencil, 3=both).
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/* movmsk and depthf are 6 bytes on AGX3b and 8 on AGX3c. */
#define AGX3_ZS_MAX_LENGTH 8u

/* Staged coverage word for unconditional kill-all-samples `movmsk`
 * (`sample_mask(target = ALL_SAMPLES, live = 0)`). */
#define AGX3_ZS_KILL_ALL_SAMPLES 0xFFFFFF00u

struct agx3_zs_desc {
   /* Half-register index of the staged writeout word. 32-bit, so even. */
   uint16_t value;
   /* The staged word is dead after this instruction. */
   bool value_kill;
   /* Scoreboard slot this instruction takes: 1..6 (AGX3b) / 1..8 (AGX3c). */
   uint8_t slot;
   /* Bit i waits scoreboard slot i+1. */
   uint16_t wait_mask;
};

/* `movmsk`: write the staged coverage word. */
bool agx3_zs_encode_sample_mask(enum agx_isa_encoding enc,
                                const struct agx3_zs_desc *d,
                                uint8_t out[AGX3_ZS_MAX_LENGTH], uint8_t *len);

/*
 * `depthf`: trigger depth/stencil writeout at the staged payload base. G17P
 * depth-only payloads contain coverage32 at value and depth32 at value + 2.
 * The caller must publish both words before this instruction. At least
 * one of `depth`/`stencil` must be set; all three combinations are witnessed.
 */
bool agx3_zs_encode_depth_stencil(enum agx_isa_encoding enc,
                                  const struct agx3_zs_desc *d, bool depth,
                                  bool stencil,
                                  uint8_t out[AGX3_ZS_MAX_LENGTH],
                                  uint8_t *len);

#ifdef __cplusplus
}
#endif

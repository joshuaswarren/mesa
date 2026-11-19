/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 tile / framebuffer encoders: st_tile -> lstcol (fragment color output)
 * and ld_tile -> lldcol (framebuffer fetch).  This core intentionally exposes
 * the directly witnessed single-sample and 2x/4x color forms. Composer-driven;
 * reproduces Apple's witnessed bytes. Blend and depth-compare are render
 * state, not ISA.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_TILE_MAX_LENGTH 14u

/* Genuine G17G builtin: base PBE pointer U0, tile offsets U4+4i,
 * one retained eight-half coordinate wire per target, descriptor indexi. */
#define AGX3_M5_MRT_EOT_MAX_LENGTH 400u
uint16_t agx3_m5_mrt_eot_nr_gprs(unsigned targets);
uint16_t agx3_m5_mrt_eot_staging_halfs(unsigned targets);
bool agx3_m5_mrt_eot_generate_sample_mask(unsigned targets, const uint8_t *selectors,
                                      unsigned multisample_mask, bool layered,
                                      uint8_t *output, size_t capacity, size_t *written);
bool agx3_m5_mrt_eot_generate_layered(unsigned targets, const uint8_t *selectors,
                                      bool multisample, bool layered,
                                      uint8_t *output, size_t capacity,
                                      size_t *written);

enum agx3_tile_width {
   AGX3_TILE_W8 = 0,
   AGX3_TILE_W16 = 1,
   AGX3_TILE_W32 = 2,
};

/*
 * The sample-mask operand of lstcol/lldcol. It is an ordinary source, not a
 * "coverage selector": the form's choice group is either an 8-bit immediate
 * (f51=1, f43) or a 16-bit register naming a half register (f51=2, f44, with
 * f53 carrying register bit 8 -- never needed below 192 -- and f55 an extra
 * bit that is set in every store witness and clear in the one load witness,
 * consistent with a last-use hint, so it is driven from the IR's kill flag
 * rather than pinned to a constant).
 *
 * Grounded against Apple's own applegpu_g17p output (MetalToolchain
 * 27.1.5194.15):
 *
 *   1x, any format          lstcol ... f51=1 f43=1
 *   2x/4x per-pixel         rld h10 <- SR39 ; lstcol ... f51=2 f44=10 f55=1
 *   4x per-sample (unrolled)  rld h16 <- SR39
 *                             bit_and h11 = h16 & 1   ... & 2, & 4, & 8
 *                             lstcol ... f44=11 / 9 / 10 / 8, f51=2, f55=1
 *   2x/4x framebuffer fetch lldcol ... f51=2 f44=19  (f55 clear: not a last use)
 *
 * The per-sample `& 1/2/4/8` sequence is what pins the meaning: the register
 * holds a plain sample bitmask in bits 0..3, and SR39's low nibble is the
 * live coverage. A per-pixel store therefore passes SR39 straight through --
 * "every sample this fragment covers" -- which is precisely what a
 * multisample store has to do. There is no all-sample immediate in any
 * witness; Apple always materialises the mask into a register at 2x and 4x.
 */
struct agx3_tile_sample {
   /* false: `value` is an 8-bit immediate sample mask.
    * true:  `value` is a half-register index holding the mask. */
   bool reg;
   /* Register form only: f55. Set in every Apple store witness, where the
    * mask register is dead afterwards; clear in the framebuffer-fetch load,
    * where it is read again. */
   bool kill;
   uint16_t value;
   /* msaa163: the caller admits the register form on an AGX3c/AGX3d store.
    * Apple's G18G 4x per-pixel fragment (and its G17G twin, byte-identical
    * here) stores `rld h10 <- SR39 ; lstcol ... f44=10 f51=2 f55=1`:
    * 270605082901ad02000071440000, native-msaa-reference-147. Only the
    * AGX_G18_MSAA4 packer sets it; loads never take it. */
   bool g17_register;
};

static inline struct agx3_tile_sample
agx3_tile_sample_imm(uint16_t mask)
{
   struct agx3_tile_sample s = {false, false, mask};
   return s;
}

static inline struct agx3_tile_sample
agx3_tile_sample_reg(uint16_t half_reg, bool kill)
{
   struct agx3_tile_sample s = {true, kill, half_reg};
   return s;
}

/* The whole-pixel form every 1x Apple fragment emits. */
#define AGX3_TILE_SAMPLE_1X agx3_tile_sample_imm(1)
/* Store `data` to the tile buffer. `chan_mask` is the component write mask
 * (R=1, G=2, B=4, A=8). `tile_off` is the byte offset of this render target
 * (sum of the per-sample sizes of the preceding RTs). */
bool agx3_tile_store(enum agx_isa_encoding enc, uint16_t data,
                     enum agx3_tile_width width, uint8_t chan_mask,
                     uint16_t tile_off, struct agx3_tile_sample sample,
                     uint16_t wait_mask, uint8_t out[AGX3_TILE_MAX_LENGTH],
                     uint8_t *len);

/* agx3_tile_store with native f13 chosen by the caller. The ordinary C
 * wrapper preserves f13=1; eight-sample fragment payloads may retain it at
 * f13=0 until their last use, as Apple's sample_id8 shader does.
 * Apple's AGX3b fragments set f13 on the leading stores of colour
 * transactions wider than five targets, see agx3_g17p_mrt_protocol(). */
bool agx3_tile_store_f13(enum agx_isa_encoding enc, uint16_t data,
                         enum agx3_tile_width width, uint8_t chan_mask,
                         uint16_t tile_off, struct agx3_tile_sample sample,
                         uint16_t wait_mask, bool f13,
                         uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len);

/* G18G only: agx3_tile_store_f13 with the G18G layer's f13 rule. f13 is
 * chosen by the caller for AGX3b and the explicit C/D register-sample path
 * (g17_register), which uses the data operand's last-use bit; legacy C/D
 * whole-pixel stores keep f13 set. */
bool agx3_tile_store_f13_g18g(enum agx_isa_encoding enc, uint16_t data,
                              enum agx3_tile_width width, uint8_t chan_mask,
                              uint16_t tile_off, struct agx3_tile_sample sample,
                              uint16_t wait_mask, bool f13,
                              uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len);

/* Load from the tile buffer into `dest` on scoreboard slot `slot`.  `first`
 * and `last` are the load-sequence boundary markers witnessed in the corpus;
 * AGX3b encodes only `first`, while AGX3c encodes both. */
bool agx3_tile_load(enum agx_isa_encoding enc, uint16_t dest,
                    enum agx3_tile_width width, uint8_t slot, uint8_t chan_mask,
                    uint16_t tile_off, struct agx3_tile_sample sample,
                    uint16_t wait_mask, bool first, bool last,
                    uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len);

/* Native1130 (Apple M6, capture 1130; run 1139): the G18 single-sample
 * mixed-format fragment forms. A whole-pixel store whose f13 is exactly
 * `last_use` (the payload's last read), and the four whole-pixel read forms of
 * the 26 fragment reads -- 8-bit lanes mask 15, 16-bit lanes mask 7 or 15, one
 * 32-bit word -- at any offset aligned to the lane width, with no incoming
 * wait. AGX3c (the G18G transcode re-encodes it) only. */
bool agx3_tile_store_lifetime(enum agx_isa_encoding enc, uint16_t data,
                              enum agx3_tile_width width, uint8_t chan_mask,
                              uint16_t tile_off, struct agx3_tile_sample sample,
                              uint16_t wait_mask, bool last_use,
                              uint8_t out[AGX3_TILE_MAX_LENGTH], uint8_t *len);
bool agx3_tile_load_g18_mixed(enum agx_isa_encoding enc, uint16_t dest,
                              enum agx3_tile_width width, uint8_t slot,
                              uint8_t chan_mask, uint16_t tile_off,
                              struct agx3_tile_sample sample, bool first,
                              bool last, uint8_t out[AGX3_TILE_MAX_LENGTH],
                              uint8_t *len);

#ifdef __cplusplus
}
#endif

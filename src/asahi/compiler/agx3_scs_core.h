/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 scalar SCS load/store encoders and decoders.
 *
 * The supported address-space(5) forms were recovered with controlled Apple
 * G17P/G17S compiler probes.  Four address kinds are represented, spanning
 * every combination Apple's G17P ray-query setup emits: direct-absolute, a
 * scalar register index plus a uniform-register base, a scalar register index
 * plus a register base, and a register base with no index -- each with a
 * signed 16-bit byte displacement.  Loads and stores may move one through four
 * 32-bit components.
 *
 * AGX3c supports W32 vectors and direct, uniform-indexed, and register-base
 * addressing witnessed in Apple's ray setup. Its destination-less traversal
 * epilogue has a separate descriptor and API below.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "agx3_mem_core.h"
#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_SCS_MAX_LENGTH 16u

enum agx3_scs_address_kind {
   /* base 0 + index 0 + byte_offset */
   AGX3_SCS_ADDRESS_DIRECT = 0,

   /* uniform[uniform_base] + gpr[index] * (1 << width) + byte_offset */
   AGX3_SCS_ADDRESS_UNIFORM_INDEXED = 1,

   /* gpr[base_register] + gpr[index] + byte_offset.  Apple's operand-kind
    * selectors read f23=2 (slot A register) and f38=2 (slot B register); the
    * base lands in f31 rather than the uniform field f32. */
   AGX3_SCS_ADDRESS_REGISTER_INDEXED = 2,

   /* gpr[base_register] + byte_offset, no register index (f23=1, f38=2). */
   AGX3_SCS_ADDRESS_REGISTER_BASE = 3,
};

struct agx3_scs_desc {
   uint16_t data; /* load destination / store value, half-register units */
   enum agx3_mem_width width;
   int16_t byte_offset;

   enum agx3_scs_address_kind address_kind;
   uint16_t index;        /* scalar GPR, half-register units */
   uint16_t uniform_base; /* scalar uniform, half-register units */

   /* Slot-B base register (Apple f31), half-register units.  Used only by the
    * two register-base kinds; must be zero otherwise.  Witnessed range is one
    * byte: the composer refuses 256 and above. */
   uint16_t base_register;

   /* Apple f68.  A four-bit field: the composer accepts every value 1..15.
    * Apple's ray-setup code uses only 1, 3 and 15, and those move one, two and
    * four 32-bit components respectively.  Zero is treated as 1 so that a
    * zero-initialised descriptor keeps the previous scalar behaviour, and the
    * encoding is byte-identical to what this core emitted before. */
   uint8_t component_mask;

   /* Apple f70.  The composer accepts exactly 0..4 and refuses 5 and above.
    * Its role is NOT established: it varies independently of `width` inside a
    * single address kind (uniform-indexed W32 stores are witnessed with both
    * f70=0 and f70=2), so it is deliberately not named a scale or a mode.
    * When `has_address_selector` is false the previous derivation from
    * `width` is kept for the two original address kinds; the two register-base
    * kinds always require an explicit value. */
   bool has_address_selector;
   uint8_t address_selector;

   uint8_t slot;       /* load only: 1..6 (b), 1..8 (c) */
   uint16_t wait_mask; /* bit i waits for scoreboard slot i + 1 */
   bool value_kill;    /* store only; witnessed only on AGX3c */

   /* AGX3c narrow loads may zero-extend to the 32-bit destination.  This is
    * exact only for uniform-indexed W8/W16 loads; every other use fails.
    */
   bool zero_extend;

   /* AGX3c ldscs f76=2.  Its placement is exact but policy is not; normal
    * lowering leaves this clear while decode/replay preserves Apple vectors.
    */
   bool c_load_marker;

   /* AGX3c f27 marks the index operand's final use. Store-side f76=1 is
    * preserved without assigning an unverified scheduling meaning to it. */
   bool index_kill;
   bool c_store_marker;

   /* Native AGX3c byte-status polling pairs f75=2 and f77=2. Keep the
    * combination explicit; its conversion semantics are not established. */
   bool c_status_load;
   bool b_status_load;

   /* Final use of the slot-B GPR base in native M5 instanced traversal. */
   bool base_kill;
};

/* Native AGX3c traversal epilogue: f9=0 and f12=0 mean there is no ordinary
 * destination or scoreboard slot. Do not lower this through a GPR load.
 * f24=1 supplies a 16-bit index; f28=1 and f80=1 are fixed native controls.
 * The operation's wider memory/coherency semantics remain to be qualified. */
struct agx3_scs_epilogue_desc {
   uint16_t index;
   uint16_t base_register;
   int16_t byte_offset;
   uint8_t component_mask;
   uint8_t address_selector;
   bool index_kill;
   bool base_kill;
   bool c_load_marker;
};

bool agx3_scs_encode_epilogue(enum agx_isa_encoding enc,
                              const struct agx3_scs_epilogue_desc *desc,
                              uint8_t out[AGX3_SCS_MAX_LENGTH], uint8_t *length);

/* AGX3b destination-less load, the form every Gen1 leaf program ends with
 * (agx3_rt_gen1_*_ops.h, and the bodies in agx3_rt_gen1_default.c). As on
 * AGX3c, f9=0 and f12=0: there is no destination register and no scoreboard
 * slot, so the access returns nothing to the shader and nothing can wait on
 * it; f28=1 and f80=1 are its fixed controls. Do not lower it through a GPR
 * load.
 *
 * The address is a 32-bit GPR base plus a signed byte offset. The leaf
 * programs use that alone (f23=1). With an index (f23=2, f24=1) a 16-bit GPR
 * is added, shifted left by the address selector as in the ordinary indexed
 * forms; that operand layout is the AGX3c epilogue's above.
 *
 * What the access does to the storage it names is not established. */
struct agx3_scs_touch_desc {
   uint16_t base_register; /* 32-bit GPR, half-register units, even */
   bool has_index;
   uint16_t index;           /* 16-bit GPR, half-register units */
   int16_t byte_offset;      /* a multiple of four */
   uint8_t component_mask;   /* 1, 3 or 15: one, two or four words */
   uint8_t address_selector; /* 2 or 4 with an index, 0 without */
   bool index_kill;
   bool base_kill;
};

bool agx3_scs_encode_touch(enum agx_isa_encoding enc,
                           const struct agx3_scs_touch_desc *desc,
                           uint8_t out[AGX3_SCS_MAX_LENGTH], uint8_t *length);

/* Accept only bytes that a descriptor reproduces exactly; in particular no
 * ordinary load or store decodes as a touch. */
bool agx3_scs_decode_touch(enum agx_isa_encoding enc, const uint8_t *bytes,
                           size_t length, struct agx3_scs_touch_desc *out);

bool agx3_scs_encode_load(enum agx_isa_encoding enc,
                          const struct agx3_scs_desc *desc,
                          uint8_t out[AGX3_SCS_MAX_LENGTH], uint8_t *length);
bool agx3_scs_encode_store(enum agx_isa_encoding enc,
                           const struct agx3_scs_desc *desc,
                           uint8_t out[AGX3_SCS_MAX_LENGTH], uint8_t *length);

/* Accept only encodings that reproduce every input byte when re-encoded. */
bool agx3_scs_decode_load(enum agx_isa_encoding enc, const uint8_t *bytes,
                         size_t length, struct agx3_scs_desc *out);
bool agx3_scs_decode_store(enum agx_isa_encoding enc, const uint8_t *bytes,
                          size_t length, struct agx3_scs_desc *out);

#ifdef __cplusplus
}
#endif

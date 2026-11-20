/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Shared "template + override" helper over the generated AGX3 encoder.
 *
 * Most AGX3 forms carry a fixed set of selector/context field values plus a
 * handful of operand fields. A binder starts from a per-form baseline (the
 * constants, recovered by decoding a witnessed instruction), overrides the
 * operand fields with the caller's values, and asks the composer to encode. The
 * composer fails closed on any contradiction, so a wrong override cannot
 * silently produce a valid-looking instruction.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "agx3_compose/agx3_compose.h"
#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

struct agx3_field_kv {
   uint16_t field;
   uint64_t value;
};

/*
 * The `ep` argument that asks for Apple's own form choice: the mnemonic's
 * encoding points by ascending size, ep breaking ties, first that encodes.
 * It is the only one AGX_ISA_ENCODING_AGX3D accepts. On AGX3B/AGX3C it is
 * Apple's choice too: it reproduces every applegpu_g17g fcmp, icmp, fseli16,
 * fseli32, iseli16, iseli32 and f32fma witness (102 distinct instructions in
 * test/native/g18g and m5-compare-select-reference).
 */
#define AGX3_BIND_APPLE_ORDER (-1)

/*
 * Encode `mnemonic.ep` for `encoding`, starting from `baseline` and applying
 * `overrides` on top (an override for a field already in the baseline wins).
 * On success writes the form's exact byte length into `*len` and the bytes into
 * `out`, and returns true. Returns false for an unknown encoding, an unknown
 * form, or any value the composer refuses.
 *
 * AGX_ISA_ENCODING_AGX3D (G18G) takes the same Apple field ids but its own
 * encoding points and lengths, so it is encoded only with
 * ep == AGX3_BIND_APPLE_ORDER, from a field vector a core encoder has
 * grounded against Apple's G18G output, and only in builds that define
 * AGX3_HAVE_AGX3D_TABLE. Any other AGX3D request returns false.
 */
/* Byte length of form `mnemonic.ep` for `encoding`, or 0 if there is none. */
unsigned agx3_bind_form_size(enum agx_isa_encoding encoding,
                             const char *mnemonic, int ep);

bool agx3_bind_compose(enum agx_isa_encoding encoding, const char *mnemonic,
                       int ep, const struct agx3_field_kv *baseline,
                       unsigned nbaseline,
                       const struct agx3_field_kv *overrides,
                       unsigned noverrides, uint8_t *out, uint8_t *len);

/*
 * The encoding point a core encoder that has grounded its field vector for
 * G18G passes: its own agx3b/agx3c `ep` unchanged, Apple's order for AGX3D.
 * Only for encoders whose AGX3D admission is argued at the call site.
 */
static inline int
agx3_bind_ep(enum agx_isa_encoding encoding, int ep)
{
   return encoding == AGX_ISA_ENCODING_AGX3D ? AGX3_BIND_APPLE_ORDER : ep;
}

#ifdef __cplusplus
}
#endif

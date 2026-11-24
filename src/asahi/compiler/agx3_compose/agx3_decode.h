/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * agx3_decode.h -- table-driven AGX3 instruction decoder, the inverse of
 * agx3_compose.
 *
 * This is agx3_table.py's Decoder in C: for one instruction's bytes it walks
 * the candidate forms' choice groups (alternatives in table order, pruned by
 * their constant bits), solves each path's field constraints, and accepts the
 * first field vector that agx3_compose() turns back into exactly those bytes.
 * A decoded vector is therefore always one the composer reproduces.
 */
#ifndef AGX3_DECODE_H
#define AGX3_DECODE_H

#include "agx3_compose.h"

/*
 * Decode `size` bytes as one instruction of table `t`. Unless `mnemonic` is
 * AGX3_MN_NONE only forms of that mnemonic are tried. On success returns the form
 * index and fills fields[]; returns -1 if no form reproduces the bytes.
 */
int32_t agx3_decode(const agx3_table *t, const uint8_t *data, unsigned size,
                    enum agx3_mnemonic mnemonic,
                    uint64_t fields[AGX3_MAX_FIELDS]);

#endif

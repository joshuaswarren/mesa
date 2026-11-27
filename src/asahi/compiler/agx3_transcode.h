/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include "util/u_dynarray.h"

/*
 * G18G (agx3d) code from the G17G (agx3c) packer: re-encode the agx3c
 * instructions in binary[start, binary->size) as agx3d, in place.
 *
 * Returns false and leaves the binary unchanged on anything not grounded;
 * *why then names the reason and *fail_offset the agx3c byte offset (from
 * start). See agx3_transcode.c for the evidence and the rules.
 */
bool agx3_transcode_agx3d(struct util_dynarray *binary, unsigned start,
                          const char **why, unsigned *fail_offset);

/* Relocate shader-part markers through the same instruction-boundary map
 * as branches. Offsets are relative to start and change only on success.
 * Cube gather operand layout requires the actual shader stage.
 * long_tile_reads: the program is a G18 single-sample mixed-format fragment
 * part (run 1139) whose whole-pixel lldcol reads take Apple's long form
 * (f79 = 2). */
bool agx3_transcode_agx3d_tile_reads(struct util_dynarray *binary,
                                     unsigned start, unsigned *offsets,
                                     unsigned count, bool fragment_stage,
                                     bool long_tile_reads, const char **why,
                                     unsigned *fail_offset);

/*
 * An Ioto (per-shader resource) program, as the M5 generators emit it, rather
 * than shader code: it may contain ldshdr, specTPR and the helper cndcall,
 * which shader code never does.
 */
struct agx3_transcode_ioto {
   /* An API program (vertex, fragment or compute) rather than one of the
    * driver's tile programs; G18G marks the former in ldshdr f21. */
   bool api_program;

   /* The launched shader's register count in 16-bit halves (the compiler's
    * nr_gprs); G18G allocates from it through specTPR f24. */
   unsigned shader_gprs;
};

bool agx3_transcode_agx3d_ioto(struct util_dynarray *binary, unsigned start,
                               const struct agx3_transcode_ioto *ioto,
                               const char **why, unsigned *fail_offset);

/* As agx3_transcode_agx3d_ioto(), relocating the instruction offsets in
 * offsets[0, count) (relative to start) as agx3_transcode_agx3d_with_offsets()
 * does; they change only on success. */
bool agx3_transcode_agx3d_ioto_with_offsets(struct util_dynarray *binary,
                                            unsigned start,
                                            const struct agx3_transcode_ioto *ioto,
                                            unsigned *offsets, unsigned count,
                                            const char **why,
                                            unsigned *fail_offset);

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_chip.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_IOTO_MAX_RECORD_SIZE 22

/* Raw AGX3 Ioto instruction encoding points (byte forms only). */
enum agx_ioto_kind {
   AGX_IOTO_KIND_INVALID = 0,
   AGX_IOTO_KIND_NOP_EP0,
   AGX_IOTO_KIND_NOP_EP1,
   AGX_IOTO_KIND_NOP_EP2,
   AGX_IOTO_KIND_NOP_EP3,
   AGX_IOTO_KIND_NOP_EP4,
   AGX_IOTO_KIND_NOP_EP5,
   AGX_IOTO_KIND_NOP_EP6,
   AGX_IOTO_KIND_NOP_EP7,
   AGX_IOTO_KIND_NOP_EP8,
   AGX_IOTO_KIND_NOP_EP9,
   AGX_IOTO_KIND_NOP_EP10,
   AGX_IOTO_KIND_LDIMM_EP0,
   AGX_IOTO_KIND_LDIMM_EP1,
   AGX_IOTO_KIND_LDIMM_EP2,
   AGX_IOTO_KIND_LDIMM_EP3,
   AGX_IOTO_KIND_COPY_EP0,
   AGX_IOTO_KIND_COPY_EP1,
   AGX_IOTO_KIND_RLD_EP0,
   AGX_IOTO_KIND_RLD_EP1,
   AGX_IOTO_KIND_SPECLM_EP0,
   AGX_IOTO_KIND_LDSHDR_EP0,
   AGX_IOTO_KIND_SPECTPR_EP0,
   AGX_IOTO_KIND_IADD_EP0,
   AGX_IOTO_KIND_B_EP0,
   AGX_IOTO_KIND_B_EP1,
   AGX_IOTO_KIND_B_EP1_COMPACT,
   AGX_IOTO_KIND_BL_EP0,
   AGX_IOTO_KIND_BL_EP1,
   AGX_IOTO_KIND_BL_EP1_COMPACT,
   AGX_IOTO_KIND_CNDCALL_EP0,
   AGX_IOTO_KIND_CNDRET_EP0,
   AGX_IOTO_KIND_LDDEV_EP0,
   AGX_IOTO_KIND_LDDEV_EP1,
   AGX_IOTO_KIND_LDDEV_EP2,
   AGX_IOTO_KIND_LDDEV_EP3,
   AGX_IOTO_KIND_LDDEV_EP4,
   AGX_IOTO_KIND_LDDEV_EP5,
   AGX_IOTO_KIND_PIXREL_EP0,
   AGX_IOTO_KIND_PIXWAIT_EP0,
   AGX_IOTO_KIND_SETPROFILECTL_EP0,
   AGX_IOTO_KIND_COUNT,
};

struct agx_ioto_record {
   enum agx_ioto_kind kind;
   uint8_t length;
   uint8_t bytes[AGX_IOTO_MAX_RECORD_SIZE];
};

/* Records which may occur while walking a complete pipeline program.  The
 * GPU0 inline LDDEV form is emitted directly by the G15/G16/G17P EslBuilder,
 * so it deliberately remains outside enum agx_ioto_kind. */
enum agx_ioto_program_record_kind {
   AGX_IOTO_PROGRAM_RECORD_INVALID = 0,
   AGX_IOTO_PROGRAM_RECORD_IOTO,
   AGX_IOTO_PROGRAM_RECORD_GPU0_INLINE_LDDEV,
   AGX_IOTO_PROGRAM_RECORD_TERMINATE,
};

#define AGX_IOTO_GPU0_INLINE_LDDEV_SIZE 14u
#define AGX_IOTO_PROGRAM_TERMINATE_SIZE 4u
#define AGX_IOTO_PROGRAM_TERMINATE_WORD UINT32_C(0x0000000e)

struct agx_ioto_program_record {
   enum agx_ioto_program_record_kind kind;
   uint8_t length;
   struct agx_ioto_record ioto;
   uint8_t bytes[AGX_IOTO_MAX_RECORD_SIZE];
};

/* Exactly the eleven byte-witnessed M3/M4/A18 Pro/M5 identities. */
bool agx_ioto_target_supported(enum agx_chip chip);

/*
 * Validate one expected record at the start of a possibly longer stream.
 * Opaque bytes are copied byte-for-byte to out and consumed is the exact
 * target-specific record length.  Both outputs are cleared on failure.
 */
bool agx_ioto_validate_kind(enum agx_chip chip,
                            enum agx_ioto_kind expected, const void *stream,
                            size_t available, struct agx_ioto_record *out,
                            size_t *consumed);

/*
 * Decode only when exactly one target-valid form matches.  A truncated prefix
 * of any other form is treated as ambiguity and rejected rather than guessed.
 * Both outputs are cleared on failure.
 */
bool agx_ioto_decode_one(enum agx_chip chip, const void *stream,
                         size_t available, struct agx_ioto_record *out,
                         size_t *consumed);

/* Decode one record in an AGX3 CDM/VDM pipeline program, including the GPU0
 * inline LDDEV form and 4-byte terminator; truncated special forms fail. */
bool agx_ioto_decode_program_record(enum agx_chip chip, const void *stream,
                                    size_t available,
                                    struct agx_ioto_program_record *out,
                                    size_t *consumed);

#ifdef __cplusplus
}
#endif

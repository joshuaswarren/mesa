/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exact bounded packing for the constant-calculation-program suffix emitted by
 * Apple's G15/G16/G17P and HAL300 state-load encoders.
 */

#pragma once

#include "agx_chip.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AGX_IOTO_PRESHADER_INLINE_LIMIT    256u
#define AGX_IOTO_PRESHADER_BRANCH_SIZE     10u
#define AGX_IOTO_PRESHADER_PRELUDE_SIZE    2u
#define AGX_IOTO_PRESHADER_ABSENT_SIZE     4u
#define AGX_IOTO_PRESHADER_MAX_SUFFIX_SIZE AGX_IOTO_PRESHADER_INLINE_LIMIT
#define AGX_IOTO_PRESHADER_NO_VA           UINT64_MAX
#define AGX_IOTO_PRESHADER_NO_BRANCH       SIZE_MAX

enum agx_ioto_preshader_mode {
   AGX_IOTO_PRESHADER_MODE_ABSENT = 0,
   AGX_IOTO_PRESHADER_MODE_INLINE,
   AGX_IOTO_PRESHADER_MODE_EXTERNAL,
};

struct agx_ioto_preshader_desc {
   /* Host-visible compiled Ioto bytes beginning at the program start. */
   const void *code;
   size_t code_size;

   /* Encoded PC-relative byte displacement modulo 2^48, or
    * AGX_IOTO_PRESHADER_NO_VA. Despite the historical field name this is
    * not an absolute target address: the caller must subtract the address
    * of the emitted B instruction from the target program address.
    * agx_ioto_preshader_emit has no source program address with which to
    * perform that relocation itself. */
   uint64_t external_va;

   /* ConstantCalculationProgramInfo bytes +0x28 and +0x29, respectively. */
   bool force_external;
   bool request_prelude;

   /* Encoder state at +0x100 (GPU0) or +0x130 (GPU1). */
   bool suppress_prelude;
};

struct agx_ioto_preshader_output {
   enum agx_ioto_preshader_mode mode;
   size_t length;

   /* Offset within bytes[] of B, or AGX_IOTO_PRESHADER_NO_BRANCH. */
   size_t branch_offset;
   uint8_t bytes[AGX_IOTO_PRESHADER_MAX_SUFFIX_SIZE];
};

/* Compiler-produced constant program. Unlike the low-level encoder descriptor
 * above, code_va is an absolute GPU address. code_size is the exact emitted
 * extent, not the distance to the next aligned shader function. A zero extent
 * means no program. The CPU code pointer is required only for inline mode. */
struct agx_ioto_constant_program {
   const void *code;
   size_t code_size;
   uint64_t code_va;
   bool force_external;
   bool request_prelude;
   bool suppress_prelude;
   /* G17P's native constant-program branch uses an absolute target (f11=1).
    * Its bytes are independent of the suffix allocation. Other families
    * currently support only the existing relative form (false). */
   bool absolute_branch;
};

/* Select and relocate a compiler-produced constant program at suffix_va,
 * the absolute address of the first suffix byte, after resource loads and
 * shader/profile state. The optional prelude shifts the B instruction's PC.
 * External programs must fit the 48-bit address space; relative branches
 * also require a signed 48-bit displacement. Addresses/extents are halfword
 * aligned; suffix_va is nonzero even for the address-independent form.
 * Output is unchanged on failure. This packs bytes only: the caller owns code
 * lifetime, resource state and launch register/occupancy accounting. */
bool agx_ioto_constant_program_emit(
   enum agx_chip chip, const struct agx_ioto_constant_program *program,
   uint64_t suffix_va, struct agx_ioto_preshader_output *output);

/*
 * Bind a nonempty compiled range to matching CPU and GPU copies of its
 * original binary. The GPU base must not refer to a separately linked main
 * shader. Both copies must contain the whole range, and its GPU extent must
 * fit the even-addressed 48-bit branch domain.
 *
 * Updates only code, code_size, and external_va. The caller's explicit
 * force/prelude options are preserved; no ABI policy is inferred here.
 * The descriptor is unchanged on failure, including a missing GPU mapping.
 * This validates bounds, not instructions, register/scratch use, or BO lifetime.
 */
bool agx_ioto_preshader_bind_range(const void *binary, size_t binary_size,
                                   uint64_t binary_va, size_t gpu_size,
                                   size_t offset, size_t size,
                                   struct agx_ioto_preshader_desc *desc);

/*
 * Pack the target-family long B or BL form.  The constant-calculation suffix
 * calls this with link=false; BL is used by the separate profile-state loader.
 * `target_va` is the 48-bit two's-complement PC-relative byte displacement,
 * represented modulo 2^48 and aligned to a halfword. `out` is unchanged on
 * failure.
 */
bool
agx_ioto_preshader_pack_branch(enum agx_chip chip, uint64_t target_va,
                               bool link,
                               uint8_t out[AGX_IOTO_PRESHADER_BRANCH_SIZE]);

/*
 * Pack a relative-form suffix after resource/state loads, LoadShader and
 * any PIXREL record. The high-level constant-program API also supports the
 * G17P renderer's native absolute branch selector:
 *
 *   inline:  compiled bytes with every trailing 06 00 NOP removed
 *   external: optional fixed LDIMM followed by the long target-family B
 *   absent:   0e 00 00 00
 *
 * Inline selection is `code_size <= 256 && !force_external`.  A zero-length
 * result after NOP trimming takes the absent path.  `output` is unchanged on
 * every failure.
 */
bool agx_ioto_preshader_emit(enum agx_chip chip,
                             const struct agx_ioto_preshader_desc *desc,
                             struct agx_ioto_preshader_output *output);

#ifdef __cplusplus
}
#endif

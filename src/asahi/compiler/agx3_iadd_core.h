/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * AGX3 `iadd` with a constant addend, an optional shift, and a narrow source.
 *
 * This is the shape all constant address arithmetic takes -- `base + i*4 + k`
 * -- so it appears in any shader with more than one memory access. It was
 * previously unencodable in the compact path and silently mis-encoded in the
 * wait-carrying one, where an immediate was deposited into a register field.
 *
 * Every field below is probe-witnessed with `metal-nt -arch applegpu_g16g`,
 * one MSL kernel per variable:
 *
 *   f2   destination register half        (32-bit)
 *   f15  slot-A immediate, 8-bit          (kind fields f23 = f24 = 1)
 *   f31  slot-B register half             (kind field f38 = 2)
 *   f39  slot-B width                     1 = 16-bit, 2 = 32-bit
 *   f43  slot-B sign-extend               `short` sets it, `ushort` does not
 *   f54  shift applied to slot B          0..3 witnessed, one kernel each
 *   f45..f50  scoreboard wait, slots 1..6
 *
 * AGX3c spells it `iadd32` with the same operand fields, its own encoding
 * constants (f44 = 3, f45 = 80), the shift moved to f55, and eight wait slots
 * at f46..f53. The same four-kernel ladder was run against
 * `-arch applegpu_g17s` to witness it.
 *
 * On AGX3b, f42 is the slot-B last-use hint for either slot-A kind. Fresh
 * G17P `o[id] = id << N` witnesses clear it because the store still needs
 * id; loaded values dead after the add set it. Drive it from IR liveness.
 * The separately witnessed AGX3c baseline remains unchanged.
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AGX3_IADD_MAX_LENGTH 14u

struct agx3_iadd_desc {
   /*
    * Slot A is normally the immediate below. Set this to add a REGISTER
    * instead -- f23/f24 move from 1 (immediate, value in f15) to 2
    * (register, value in f16), the same operand-kind convention used
    * throughout AGX3. Witnessed byte-exactly against the shifted adds Apple
    * emits for `a*b + (c << N)`, where f54 carries N.
    */
   bool src_a_is_register;
   uint16_t src_a_reg;
   /*
    * Slot A may be a single 16-bit half, zero- or sign-extended, exactly as
    * slot B may (f24 selects the width the way f39 does for slot B, and f28
    * carries slot A's sign-extension the way f43 does for slot B). This is
    * how `load_local_invocation_id`'s three 16-bit components are summed --
    * `iadd d32, h_a, h_b` -- which the register slot-A form previously could
    * not express at all, because it pinned both widths to 32-bit.
    *
    * Witnessed byte-exactly against applegpu_g17p:
    *   ushort + ushort   9f 01 56 00 02 00 04 08 17 05  f24=1 f39=1
    *   short  + short    9f 01 56 00 02 02 00 08 17 0f  f28=1 f43=1
    *   lid.x + lid.y     9f 11 54 00 03 00 04 08 17 05  f11=1 (dest cache)
    *   (that) + lid.z    9f 21 54 00 02 00 08 28 17 05  f24=2 f39=1
    */
   bool src_a_16;
   bool src_a_sign_extend;
   /*
    * Last-use (kill) hints for the two register operands, f27 for slot A and
    * f42 for slot B. These are LIVENESS, not constants -- agx3_mem_core.c
    * drives the same two fields from idx_kill/base_kill. Apple's witnesses
    * happen to carry both set because their operands are dead there; forcing
    * them on for a live register told the hardware it could be reused and
    * cost two compute probes on real hardware. Drive them from the IR.
    */
   bool src_a_kill;
   bool src_kill;
   uint16_t dest;      /* 32-bit register half, 2-aligned */
   uint8_t immediate;  /* the constant addend */
   uint16_t src;       /* register half added to it */
   bool src_16;        /* source is a single 16-bit half */
   bool sign_extend;   /* narrow source is sign- rather than zero-extended */
   uint8_t shift;      /* source shifted left by this first; 0..3 */
   /* f11: keep the result in the operand cache. A hint, not semantics --
    * Apple sets it on the first of a pair of dependent adds and clears it on
    * the second. */
   bool dest_cache;
   uint16_t wait_mask; /* bit i waits scoreboard slot i + 1 */

   /*
    * AGX3b only. `isub` is the same field layout under its own opcode, for
    * either slot-A kind; slot B is the subtrahend and keeps its width and
    * sign-extension fields:
    *
    *   a - int(short b)       1f 01 56 00 02 00 10 28 17 0d  f39=1 f43=1
    *   int(short a) - b       1f 01 56 00 02 08 00 88 17 07  f24=1 f28=1
    *   16 - int(short b)      1f 01 56 00 02 20 00 08 15 0c  f15=16
    *
    * AGX3b register slot A only: saturate (f53). With 32-bit operands the
    * slot sign fields select signed saturation, set together:
    *
    *   addsat(int, int)       9f 01 56 00 02 00 10 e8 17 0f  f28 f43 f53
    *   addsat(uint, uint)     9f 01 56 00 02 00 10 e8 17 05  f53
    *   subsat(int, int)       1f 01 56 00 02 00 10 e8 17 0f
    *   subsat(uint, uint)     1f 01 56 00 02 00 10 e8 17 05  f53
    *
    * G17G, both slots 32-bit registers, no shift. `subtract` is Apple's
    * `isub32` (root f45=100 instead of 80): slot B is subtracted. `saturate`
    * is f54=1 in place of the shift field. With saturation the 32-bit
    * operands carry their own signedness in f28 (slot A) and f43 (slot B):
    * `addsat`/`subsat` over int set both, over uint neither, and the plain
    * `a - b` sets none. Witnessed byte-exactly against applegpu_g17g
    * (gpu/m5-xfb-alu-reference.metal, -alu2-, -alu3-):
    *
    *   subsat(uint)   2f 00 04 1a 25 00 a2 02 18 81 20 00   f45=100 f54=1
    *   subsat(int)    2f 00 04 1a 25 08 a2 02 18 81 60 00   + f28=1 f43=1
    *   addsat(uint)   2f 00 04 1a 21 00 a3 02 38 81 20 00   f45=80 f54=1
    *   addsat(int)    2f 00 04 1a 21 00 a3 02 38 81 e0 00   + f28=1 f43=1
    *   a - b          2f 00 04 1a 25 00 a2 02 08 81 21 00   f45=100
    */
   bool subtract;
   bool saturate;
   bool src_a_sign_extend_32; /* f28 on a 32-bit slot A (saturating only) */
   bool sign_extend_32;       /* f43 on a 32-bit slot B (saturating only) */
   /* AGX3b register slot A only: operand-cache hints f25 (A) and f40 (B). */
   bool src_a_cache;
   bool src_cache;
};

bool agx3_iadd_encode(enum agx_isa_encoding encoding,
                      const struct agx3_iadd_desc *desc,
                      uint8_t out[AGX3_IADD_MAX_LENGTH], uint8_t *len);

/* G18G (M6) compiles as agx3c: the G17G/G18G iadd32/isub32 encoder as
 * qualified on the M6 (agx3_iadd_core.c). agx3_iadd_encode keeps the
 * G17G-family (agx3c) admission for every other GPU. */
bool agx3_iadd_encode_g18g(enum agx_isa_encoding encoding,
                           const struct agx3_iadd_desc *d,
                           uint8_t out[AGX3_IADD_MAX_LENGTH], uint8_t *len);

/* Native G17P register minus an 8-bit unsigned immediate. Subtraction is
 * not commuted: the register occupies slot A, immediate slot B. */
struct agx3_isub_imm_desc {
   uint16_t dest, src, immediate;
   bool dest_cache;
   bool src_16, sign_extend;
   bool src_cache, src_kill;
   uint16_t wait_mask;
};
bool agx3_isub_immediate_encode(enum agx_isa_encoding encoding,
                                 const struct agx3_isub_imm_desc *desc,
                                 uint8_t out[AGX3_IADD_MAX_LENGTH], uint8_t *len);
bool agx3_iadd_immediate_encode(enum agx_isa_encoding encoding,
                                 const struct agx3_isub_imm_desc *d,
                                 uint8_t out[AGX3_IADD_MAX_LENGTH], uint8_t *len);

/*
 * `imad dst32, srcA, #imm, srcC` -- multiply-accumulate with a constant
 * multiplier. This is how an index-derived address is computed
 * (`gid * stride + base`), so every SSBO access that is not at a constant
 * index goes through it.
 *
 * Witnessed the same way, with `o[i] = a[i] * K + b[i]` kernels over several
 * multipliers (129, 253, 255 -- smaller constants get strength-reduced away
 * before they reach the encoder):
 *
 *   f2   destination           f16  slot-A register
 *   f30  slot-B immediate      (kind fields f38 = f39 = 1)
 *   f46  slot-C register
 *   f54  slot-C width          1 = 16-bit, 2 = 32-bit, 3 = 64-bit
 *   f58  slot-C sign-extend    `short` sets it, `ushort` does not
 *   f10  destination width     2 = 32-bit, 3 = 64-bit register pair
 *   f11  destination cache hint
 *   f60..f65  scoreboard wait, slots 1..6
 *
 * Native reused-input variants establish per-source cache/discard hints:
 * f25/f27 (A), f40/f42 (B), and f55/f57 (C). The multiply's shift field is
 * not modelled, so a shifted imad is refused rather than encoded with the
 * shift dropped -- which is exactly the silent failure this replaces.
 */
struct agx3_imad_desc {
   uint16_t dest;      /* 32-bit register half, or the low half of a pair */
   /* Native shaderInt16 arithmetic uses destination-width selector f10=1. */
   bool dest_16;
   /*
    * 32x32 -> 64. The destination width selector f10 moves from 2 to 3 and the
    * destination becomes a register pair; each narrower operand is then widened
    * according to its own sign-extend bit. This is the only wide integer
    * multiply the hardware has: `umul_high`/`imul_high` are this instruction
    * with the low word left unread, and Apple's own `mulhi` kernels compile to
    * bytes identical to its `(ulong)a * (ulong)b` kernels.
    *
    * The pair is 32-bit aligned, not 64-bit aligned: Apple places one at half
    * 18. It occupies four halves, so the last encodable destination is 252.
    */
   bool dest_64;
   /*
    * f11, the destination's register-cache hint: set when a following ALU
    * instruction reads the result, clear when only a store does. Apple's own
    * 32-bit pure multiply carries it in one kernel and not in the other, so it
    * is per-instruction and belongs to liveness (agx_opt_register_cache), not
    * to any form's baseline.
    */
   bool dest_cache;
   uint16_t src_a;     /* multiplicand register half */
   bool src_a_kill, src_a_cache;
   bool src_a_16;      /* multiplicand is a single 16-bit half */
   /*
    * f28: widen the multiplicand by sign rather than by zero. Only meaningful
    * when the operand is narrower than the product: 16->32 and 32->64 use it,
    * while width-matched products reject it.
    */
   bool src_a_sign_extend;
   uint8_t multiplier; /* constant multiplied by src_a (immediate form) */
   /*
    * Register multiplier. When set, src_a is multiplied by the 32-bit
    * register at `multiplier_reg` instead of the `multiplier` constant, and
    * the slot-B kind moves from immediate (f38/f39 = 1, value in f30) to
    * register (f38/f39 = 2, value in f31) -- the same 1/2/3 operand-kind
    * convention used throughout AGX3. Witnessed byte-exactly against Apple's
    * own `o[i] = a[i]*b[i] + c[i]` compute kernel.
    */
   bool multiplier_is_register;
   uint16_t multiplier_reg;
   bool multiplier_kill; /* f42: last use of the multiplier register */
   bool multiplier_cache;
   bool multiplier_16;   /* multiplier register is a single 16-bit half */
   bool multiplier_sign_extend; /* f43, the slot-B counterpart of f28 */
   bool addend_is_immediate;
   /* Native slot-C unsigned8 immediate, with either multiplier source class. */
   uint16_t addend_immediate;
   uint16_t src_c;     /* addend register half */
   bool src_c_kill, src_c_cache;
   /*
    * Pure multiply, no addend. AGX3 has no shifted imad: to build
    * `a*b + (c << s)` Apple emits this form and then a shifted iadd, which is
    * exactly what the backend does for AGX3. The addend operand is absent
    * rather than zero -- a distinct encoding, witnessed against Apple's
    * `o[i] = a[i]*b[i] + (c[i] << N)` kernels, whose imadd is byte-identical
    * for N = 1, 2 and 3 because the shift never reaches this instruction.
    */
   bool no_addend;
   bool src_c_16;      /* addend is a single 16-bit half */
   bool src_c_64;      /* addend is a 64-bit register pair (slot-C width 3) */
   bool src_c_sign_extend; /* narrow addend is sign- rather than zero-extended */
   /*
    * Subtract the addend instead of adding it: `a*b - c`.  Apple spells this
    * as a separate instruction, `imsub`, which AGX3b.xml already carries with
    * a field layout identical to `imadd` -- only bit 7 of the opcode differs
    * (imadd 0x9f, imsub 0x1f).  metal-nt -arch applegpu_g17p, one variable per
    * kernel, everything else held fixed:
    *
    *   a*b + c   9f 00 56 00 02 08 00 40 d0 2f 2a 00
    *   a*b - c   1f 00 56 00 02 08 00 40 d0 2f 2a 00
    *   a*b + 1   9f 00 56 00 02 08 00 08 d0 26 0a 00
    *   a*b - 1   1f 00 56 00 02 08 00 08 d0 26 0a 00
    *
    * so the same bit serves an immediate and a register addend.  This matches
    * AGX2, where agx_pack.c allows `neg` only on IMAD's src[2] and puts it in
    * a dedicated bit (bit 27) rather than treating it as a source modifier.
    */
   bool negate_addend;
   uint16_t wait_mask; /* bit i waits scoreboard slot i + 1 */
};

bool agx3_imad_encode(enum agx_isa_encoding encoding,
                      const struct agx3_imad_desc *desc,
                      uint8_t out[AGX3_IADD_MAX_LENGTH], uint8_t *len);

/* G18G (M6) compiles as agx3c: the G17G/G18G imadd32/imadd64 encoder as
 * qualified on the M6 (agx3_iadd_core.c). agx3_imad_encode keeps the
 * G17G-family (agx3c) admission for every other GPU. */
bool agx3_imad_encode_g18g(enum agx_isa_encoding encoding,
                           const struct agx3_imad_desc *d,
                           uint8_t out[AGX3_IADD_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif

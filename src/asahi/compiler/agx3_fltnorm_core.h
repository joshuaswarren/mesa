/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The two AGX3 fragment colour-path instructions Mesa had no producer for:
 * `fltnorm` (float -> normalized lane pack) and `f32mul.ep4` (the perspective
 * divide).
 *
 * Both appear in *every* Apple fragment shader in the G17P corpus, which is
 * why they rank ahead of discard and depth: a fragment shader that writes a
 * normalized render target cannot be assembled without `fltnorm`, and one that
 * reads a perspective-correct varying cannot be assembled the way Apple does
 * without `f32mul.ep4`.
 *
 * ## fltnorm
 *
 * `fltnorm` takes two 32-bit floats and packs them, normalized, into the two
 * halves of one 16-bit register; `lstcol` then stores those 16-bit registers.
 * A four-component 8-bit target therefore costs exactly two `fltnorm`s, which
 * is what every witness shows.
 *
 * Operand model, PROVEN by construction against 19 witnesses (7 from the
 * shipped .metalar objects fmain / fdiscard / ftex / fdepth, 12 from offline
 * `applegpu-nt -arch applegpu_g17p` probes that vary component count, target
 * format, swizzle, attachment count and operand class):
 *
 *   f9  = 2, f2 = destination 16-bit register, f10 = 1 (16-bit)
 *   f23 = source-0 class: 1 immediate (AGX minifloat in f15),
 *                         2 register (f16), 3 uniform (f17); f24 = 2, 32-bit
 *   f40 = source-1 class, same encoding, value in f32 / f33 / f34; f41 = 2
 *   f25 / f27  = source-0 liveness pair, only for a register source:
 *                f27 = 1 last use, f25 = 1 still live
 *   f42 / f44  = the same pair for source 1
 *   f28 / f29  = source-0 abs / neg of a register source, f45 / f46 for
 *                source 1: abs(x) and -x into R8/RG8/RGBA8Unorm, R16Unorm,
 *                RG16Snorm and R8Snorm set them, -abs(x) sets both
 *                (blend/native/negpack.py, negpack2.py)
 *   f58 = normalized format: U8=0, S8=1, U16=2, S16=3, and the two sRGB
 *         modes below. Fresh native R16Unorm/R16Snorm pipelines establish the
 *         upper width bit.
 *   f59 = 1 two packed lanes. A single-component target instead sets f59 = 0,
 *         f40 = 0, f41 = 1 and has no second source (probe fn_r).
 *   f50 + (slot - 1) = the per-slot wait mask, as in every other AGX3b form.
 *
 * The minifloat encoding of an immediate source is Apple's usual one: 0.25 is
 * 16, 0.5 is 32, 0.75 is 40, 1.0 is 48 (probe fn_const, whose two instructions
 * carry exactly those four values).
 *
 * f68 is set when FLTNORM starts the main program, for both immediate and
 * uniform operands. It is clear after pixel operations or position arithmetic,
 * including when that later FLTNORM is the first uniform reader. Model only
 * this native instruction-position rule; the hardware meaning is not named.
 *
 * ## f32mul.ep4/ep5
 *
 * AGX3 has no fused projective iterator, so a perspective-correct varying is
 * `f32interp` of the raw value, `f32rcp` of the interpolated 1/W, and a
 * multiply. Apple spells that multiply with encoding point 4 rather than the
 * ordinary `f32mul.ep1` it uses for user multiplies -- probe fn_mul2, which
 * contains both in one shader, separates them cleanly.
 *
 * ep4 carries one extra base field, f70, and across all 20 witnesses in six
 * shaders f70 equals the varying index (`f22`) of the `f32interp` that
 * produced the multiply's raw operand -- including probe fn_swz, where the
 * shader's component order is reversed and f70 follows the varying rather than
 * the program order. INFERRED, on that correlation: f70 names the attribute
 * the divide belongs to. Nothing else about ep4 differs from ep1.
 *
 *   f9 = 2 / f2 general destination (with hint f11), or f9 = 5 / f5 staging
 *          destination on slot f12 -- ftex divides straight into the texture
 *          coordinate staging registers
 *   f35 = the reciprocal operand (f42 = 2 register, f43 = 2 32-bit), liveness
 *         pair f46 last use / f44 still live
 *   f18 = the interpolated operand (f25 = 2, f26 = 2), liveness pair
 *         f29 last use / f27 still live
 *   f80 + (slot - 1) = wait mask
 *
 * The compact ep4 form can name staging slots 1..5. Apple's dynamic argument-
 * buffer fragment path stages coordinates on slot 6 and uses the otherwise
 * identical extended ep5 form; its extra bytes carry that upper slot bit.
 *
 * ## The sRGB conversion modes
 *
 * f58 (and normflt's f40, which is the same field on the other side) reaches
 * two further values, and they are what makes an sRGB attachment free. From
 * nine fresh render pipelines built with `metal-nt -arch applegpu_g17p` and
 * rebuilt against `applegpu_g17s`:
 *
 *   f58 = 4   every lane of this pack is sRGB-encoded
 *   f58 = 6   lane 0 is sRGB-encoded, lane 1 is a linear UNORM8 lane
 *
 * A float4 into RGBA8Unorm_sRGB is therefore mode 4 on the (R,G) pack and
 * mode 6 on the (B,A) pack, because the alpha channel of an sRGB format is
 * linear. BGRA8Unorm_sRGB emits byte-identical code, so channel order is not
 * the shader's business. R8Unorm_sRGB is one lane at mode 4 and RG8Unorm_sRGB
 * two lanes at mode 4, so the mode describes the lanes rather than the
 * attachment. Fixed-function blending on an sRGB attachment adds no
 * instruction at all, and a shader with an sRGB attachment at color(0) and a
 * linear one at color(1) selects per attachment. Programmable blending shows
 * the same two modes on the `normflt` decode. f58 = 5 is unwitnessed.
 *
 * The whole difference between Apple's sRGB and linear RGBA8 fragment programs
 * is these two field values: the `lstcol` that follows is byte-identical,
 * which is the same statement as "lstcol carries a width, not a format".
 *
 * ## normflt
 *
 * The inverse, for a tilebuffer *read* of a normalized 8-bit attachment: one
 * 16-bit register in, two 32-bit floats out. Same story as the store side --
 * `lldcol` carries a width, not a format, so the unpack is explicit. Witnessed
 * by three programmable-blending probes (`float4/float2/float [[color(0)]]`)
 * on both encodings.
 *
 *   f9 = 2, f2 = the 32-bit destination pair, f10 = 2
 *   f18 = the 16-bit source, f25 = 2 register, f26 = 1 (16-bit)
 *   f41 = 1 two lanes, 0 one lane -- the same role f59 plays for fltnorm
 *   f32 + (slot - 1) = the per-slot wait mask. PROVEN: the tile load these
 *       consume publishes on slot 6 (AGX3b) or 8 (AGX3c), and exactly the
 *       corresponding bit is set on the first unpack of each pair.
 *
 * ## AGX3c
 *
 * Every operand field id is identical on AGX3c; the encoding differs only by
 * the sub-opcode constants it carries -- f48 = 3, f49 = 72 for `fltnorm`, and
 * f30 = 3, f31 = 115 for `normflt`. That was established by re-running the
 * same probe set against `applegpu_g17s`, which is the AGX3c part.
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Longest of the three forms across both encodings: AGX3b's fltnorm is 10
 * bytes and its normflt 8, while AGX3c's fltnorm is 12 and its normflt 10.
 * f32mul.ep4 is 8 on both, while the slot-6 ep5 form is 12.
 */
#define AGX3_FLTNORM_MAX_LENGTH 12u

/*
 * Which lanes of one pack are sRGB. The linear modes are chosen by `snorm` and
 * `is_16bit`; sRGB is a separate selector because the hardware distinguishes an
 * all-sRGB pair from the RGBA tail pair whose second lane is the linear alpha,
 * and neither is reachable by combining the two linear booleans.
 */
enum agx3_fltnorm_srgb {
   AGX3_FLTNORM_SRGB_NONE = 0,  /* linear; snorm/is_16bit pick the mode */
   AGX3_FLTNORM_SRGB_BOTH = 1,  /* f58 = 4 */
   AGX3_FLTNORM_SRGB_LANE0 = 2, /* f58 = 6, lane 1 stays linear UNORM8 */
};

enum agx3_fltnorm_src_kind {
   AGX3_FLTNORM_SRC_NONE = 0,
   AGX3_FLTNORM_SRC_IMMEDIATE = 1, /* AGX 8-bit minifloat */
   AGX3_FLTNORM_SRC_REGISTER = 2,
   AGX3_FLTNORM_SRC_UNIFORM = 3,
};

struct agx3_fltnorm_src {
   uint8_t kind; /* enum agx3_fltnorm_src_kind */
   uint16_t value;
   bool kill; /* register sources only: this is the last read */
   /* Register sources only: take |value| and/or negate it first. */
   bool abs, neg;
};

/* The packed word `pack` builds and `unpack` reads: pack's f92 and unpack's
 * f40. The float layouts take three lanes, and pack marks them with f93
 * (rgb9e5/native/rgb9e5.py: RGB9E5Float and RG11B10Float attachments). */
enum agx3_packed_layout {
   AGX3_PACKED_RGB10A2 = 0,
   AGX3_PACKED_RGB9E5 = 1,
   AGX3_PACKED_RG11B10 = 2,
};

/* Native RGB10A2 conversion packs four f32 registers into one32-bit word.
 * A register lane may take abs (f28 + 17i) and neg (f29 + 17i), as abs(x),
 * -x and -abs(x) into RGB10A2Unorm do (blend/native/negpack*.py). The
 * packed float layouts pack the first three lanes the same way. */
struct agx3_rgb10a2_desc {
   uint16_t dest;
   uint16_t source[4];
   uint8_t kind[4]; /* agx3_fltnorm_src_kind */
   uint16_t wait_mask;
   bool first_instruction;
   bool discard[4];
   bool cache[4];
   bool abs[4];
   bool neg[4];
   uint8_t layout; /* enum agx3_packed_layout */
};
bool agx3_rgb10a2_encode_for_encoding(enum agx_isa_encoding enc,
                        const struct agx3_rgb10a2_desc *desc,
                        uint8_t out[16], uint8_t *length);
bool agx3_rgb10a2_encode(const struct agx3_rgb10a2_desc *desc,
                        uint8_t out[14], uint8_t *length);
/* G18G (M6) compiles as agx3c: G17G/G18G RGB10A2 fragment epilogs share the
 * operand fields, with the additional f82/f83 sub-opcode pair. Packed floats
 * and part-start markers remain G17P-only there until their shader-linking
 * paths are qualified; otherwise as agx3_rgb10a2_encode_for_encoding. */
bool agx3_rgb10a2_encode_g18g(enum agx_isa_encoding enc,
                              const struct agx3_rgb10a2_desc *desc,
                              uint8_t out[16], uint8_t *length);

struct agx3_rgb10a2_unpack_desc {
   uint16_t dest, source; /* half indices: four f32 outputs, one u32 input */
   uint16_t wait_mask;
   bool discard;
   bool dest_hint; /* AGX3C f11: G17G/G18G destination lifetime hint */
   uint8_t layout; /* enum agx3_packed_layout */
};
bool agx3_rgb10a2_unpack_encode(
   enum agx_isa_encoding enc, const struct agx3_rgb10a2_unpack_desc *desc,
   uint8_t out[16], uint8_t *length);
/* G18G (M6) compiles as agx3c: native G17G/G18G RGB10A2 framebuffer reads and
 * unpack_unorm10a2_to_float, ten bytes with f27 marking a retained source.
 * AGX3B goes through agx3_rgb10a2_unpack_encode; packed-float layouts are
 * refused on agx3c/agx3d. */
bool agx3_rgb10a2_unpack_encode_g18g(
   enum agx_isa_encoding enc, const struct agx3_rgb10a2_unpack_desc *desc,
   uint8_t out[16], uint8_t *length);

struct agx3_fltnorm_desc {
   uint16_t dest;                 /* 16-bit (half) register index */
   bool dest_hint;                /* f11, unmodelled; clear in every witness */
   struct agx3_fltnorm_src src[2];
   bool two_lanes;                /* pack both sources; false = one lane */
   bool snorm;                    /* signed-normalized target */
   bool is_16bit;                 /* 16-bit normalized lane */
   uint8_t srgb;                  /* enum agx3_fltnorm_srgb */
   uint16_t wait_mask;            /* bit i waits scoreboard slot i + 1 */
   bool first_instruction;       /* G17P f68, first main instruction only */
};

bool agx3_fltnorm_encode(enum agx_isa_encoding enc,
                         const struct agx3_fltnorm_desc *d,
                         uint8_t out[AGX3_FLTNORM_MAX_LENGTH], uint8_t *len);

/* Relocate a compiler-proven part-start marker after joining shader parts.
 * This does not discover instructions by scanning arbitrary executable bytes.
 */
bool agx3_fltnorm_relocate_start(uint8_t *code, size_t size,
                                 bool has_start_marker, size_t linked_offset);

/* Finite compiler provenance for a uniform-copy-only main and an opaque
 * RGBA8 two-pack epilog. No executable byte parsing is used for fusion. */
struct agx3_colour_link_info {
   /* Producer-recorded native pixel-entry provenance for part linking.
    * straight_line admits a derivative-consumer entry gate in this part;
    * entry_unspanned proves that deleting the entry leaves every relative
    * branch displacement unchanged. */
   bool rt_query_write_acquired;
   uint8_t rt_query_wait_count;
   bool rt_query_wait_unsupported;
   uint16_t rt_query_wait_mask;
   uint32_t rt_query_wait_offset;
   bool entry_emitted;
   bool straight_line;
   bool entry_unspanned;
   uint32_t entry_offset;
   /* M5: this part ends a covered-sample depth/stencil loop with a plain
    * class-1 release at zs_loop_release_offset. When this part's entry is
    * the linked program's, agx_fast_link drops that release and marks the
    * entry with f34, the native form. */
   bool zs_loop_release;
   uint32_t zs_loop_release_offset;

   /* Colour pixel classes consumed by this part: bit n is the class of the
    * tile partitions n, n + 3, ...  The linked fragment state loader must keep
    * every consumed class acquired until the shader's terminal ST_TILE
    * releases it, and releases the others itself. */
   uint8_t colour_classes;

   bool copy_only;
   uint8_t copy_count;
   struct { uint16_t reg, uniform; } copies[4];
   bool opaque_pack;
   uint16_t store_reg;
   struct agx3_fltnorm_desc packs[2];
};

#define AGX3_COLOUR_LINK_SIZE 54u
bool agx3_fuse_colour_parts(const struct agx3_colour_link_info *main,
                             const struct agx3_colour_link_info *epilog,
                             uint8_t out[AGX3_COLOUR_LINK_SIZE]);

/*
 * Each register operand of an AGX3 float op picks between two bits: one marks
 * the read as the operand's last, the other marks the value as still live.
 * Apple also leaves both clear on occasional operands (one of the seventeen
 * perspective multiplies here does), so this is a hint rather than a
 * three-valued fact about liveness, and UNMARKED -- claiming nothing -- is the
 * conservative default.
 */
enum agx3_operand_liveness {
   AGX3_OPERAND_UNMARKED = 0,
   AGX3_OPERAND_LIVE,
   AGX3_OPERAND_KILL,
};

/* Projective reciprocal: native f32rcp has a second coefficient operand
 * (f42=6/f57=1), distinct from the ordinary single-GPR reciprocal. Its
 * coefficient occupies the ordinary source-discard bit, so only the cache
 * hint is available in this form. */
struct agx3_persp_rcp_desc {
   uint16_t dest;
   uint16_t rhw;
   uint16_t coef_index;
   bool rhw_cache;
   uint16_t wait_mask;
};

bool agx3_persp_rcp_encode(enum agx_isa_encoding enc,
                           const struct agx3_persp_rcp_desc *d,
                           uint8_t out[AGX3_FLTNORM_MAX_LENGTH], uint8_t *len);

struct agx3_persp_mul_desc {
   bool stage_dest;      /* destination is a staging register (f9 = 5) */
   uint16_t dest;        /* 32-bit register or staging index */
   uint8_t slot;         /* staging destinations only: scoreboard slot */
   uint16_t reciprocal;  /* 32-bit register holding 1/W */
   enum agx3_operand_liveness reciprocal_liveness;
   uint16_t raw;         /* 32-bit register holding the raw varying */
   enum agx3_operand_liveness raw_liveness;
   uint8_t vary_index;   /* the f32interp varying index this divide belongs to */
   bool dest_hint;       /* f11; set in every general-destination witness */
   uint16_t wait_mask;
};

bool agx3_persp_mul_encode(enum agx_isa_encoding enc,
                           const struct agx3_persp_mul_desc *d,
                           uint8_t out[AGX3_FLTNORM_MAX_LENGTH], uint8_t *len);

struct agx3_normflt_desc {
   uint16_t dest;      /* register receiving the unpacked floats */
   bool dest_16;       /* f10 = 1: unpack into f16 lanes instead of f32 */
   bool dest_hint;     /* f11; set in every witness */
   uint16_t src;       /* 16-bit register holding the packed lanes */
   bool two_lanes;     /* unpack both bytes; false = one lane */
   bool snorm;         /* signed-normalized source */
   bool is_16bit;      /* 16-bit normalized lane */
   uint8_t srgb;       /* enum agx3_fltnorm_srgb */
   uint16_t wait_mask; /* bit i waits scoreboard slot i + 1 */
};

bool agx3_normflt_encode(enum agx_isa_encoding enc,
                         const struct agx3_normflt_desc *d,
                         uint8_t out[AGX3_FLTNORM_MAX_LENGTH], uint8_t *len);

#ifdef __cplusplus
}
#endif

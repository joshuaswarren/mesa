/*
 * Copyright 2026 The Asahi Linux Contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/*
 * Apple GPU identity, instruction-set generation and binary encoding.
 *
 * These are three different questions and the driver has historically only
 * asked the first one, because every part it supported answered the other two
 * the same way. That stops being true at M3:
 *
 *   - G13 (M1) and G14 (M2) are the AGX2 instruction set. That is what
 *     agx_pack.c emits and what src/asahi/isa/AGX2.xml describes.
 *   - G15 (M3), G16 (M4), G17 (M5) and G18 (M6) are AGX3, a different
 *     instruction set. Apple's own compiler routes them through a separate
 *     LLVM target, and the community AGX2 decoder consumes 0% of their code.
 *   - AGX3 is not one encoding. Three binary encodings exist, and which one a
 *     part uses is now known for every chip this driver can enumerate.
 *
 * AGX3 now has separate decoders and encoders. Device support remains a
 * stricter question: G17P is admitted only through its exact T8140 contract,
 * while other AGX3 identities stay outside the runnable Honeykrisp path until
 * their independently-versioned pieces are connected.
 */

enum agx_chip {
   /* AGX2. `G` is a single-cluster part, `X` is multi-cluster. */
   AGX_CHIP_G13G,
   AGX_CHIP_G13X,
   AGX_CHIP_G14G,
   AGX_CHIP_G14X,

   /* G15P is the A16 phone part and still executes AGX2. The Mac G15 parts
    * execute AGX3b; G15G additionally needs explicit scheduling nops. */
   AGX_CHIP_G15P,
   AGX_CHIP_G15G,
   AGX_CHIP_G15S,
   AGX_CHIP_G15C,
   AGX_CHIP_G15D,

   /* Every G16 part executes AGX3b. */
   AGX_CHIP_G16P,
   AGX_CHIP_G16G,
   AGX_CHIP_G16S,
   AGX_CHIP_G16C,

   /* G17P/G17A remain on the HAL200/AGX3b side of the boundary. The M5
    * G/S/C parts use AGX3c.
    *
    * G17P is an instruction-set and firmware-HAL profile (AGX3b shaders,
    * HAL200 firmware interface), not one SoC. agx_chip_from_params() maps
    * three SoCs to it: the A18 Pro (T8140, one cluster), the M4 Pro (T6040)
    * and the M4 Max (T6041). They share the encodings, so encoding gates on
    * G17P are exact. They differ in cluster and core counts, caches and
    * clocks, so a performance or tuning gate on G17P is only as good as the
    * rows behind it: give it measurements on its members (at least the A18
    * Pro and the M4 Pro), or select it by the hardware property that is the
    * real reason. */
   AGX_CHIP_G17P,
   AGX_CHIP_G17A,
   AGX_CHIP_G17G,
   AGX_CHIP_G17S,
   AGX_CHIP_G17C,

   /* The M6 base part (J873, T8152). Its command-stream records are the
    * M5 HAL300 ones, but its shaders use a third AGX3 encoding. */
   AGX_CHIP_G18G,

   AGX_CHIP_UNKNOWN,
};

/* Instruction set. Apple's compiler names these agx2 and agx3, and the low
 * nibble of an applegpu Mach-O cpusubtype is exactly this value. */
enum agx_isa {
   AGX_ISA_AGX2 = 2,
   AGX_ISA_AGX3 = 3,
   AGX_ISA_UNKNOWN = 0,
};

/*
 * Binary encoding.
 *
 * One instruction set, three encodings. The first two were established by
 * decoding a 274-kernel corpus compiled for the ten AGX3 families known at the
 * time with each of the AGX3 instruction encoding tables. Every family
 * decodes 100% under exactly one table and 0-1.8% under any other -- no
 * intermediate values:
 *
 *   agx3b (variant 0): g15g, g15s, g15d, g16g, g16p, g16s, g17p
 *   agx3c (variant 1): g17g, g17s, g18p
 *
 * The third is new with the macOS 27 toolchain:
 *
 *   agx3d (variant 2): g18g, g18m, g19p
 *
 * Compiling one probe with metal-nt for every AGX3 arch gives byte-identical
 * code within that row and code unlike either row above, and 41 of 41 G18G
 * programs fail to decode under the agx3c, agx3b and AGX2 tables, where every
 * G17G one decodes cleanly under agx3c. The instruction set underneath is
 * still AGX3 (the applegpu_g18g Mach-O cpusubtype 0x1c3 has low nibble 3), and
 * roughly a fifth of the instructions keep their agx3c bytes, but the common
 * ALU, memory, control and texture forms are re-encoded with different
 * lengths. Its table is in the tree (agx3_compose/agx3d_tables.c and
 * src/asahi/isa/AGX3d.xml, variant 2), but only the
 * instruction forms a core encoder has grounded against Apple's G18G output
 * are emitted for it; see agx_chip_codegen_supported.
 *
 * An earlier partition by byte-identity read variant 0 as two groups,
 * splitting g15g off from the rest. That was an artefact: g15g uses the same
 * encoder table, and differs only in that its scheduler pads with nops. See
 * agx_chip_needs_scheduling_stalls.
 *
 * The names match src/asahi/isa/AGX3b.xml, AGX3c.xml and AGX3d.xml and the
 * disassembler entry points generated from them.
 */
enum agx_isa_encoding {
   AGX_ISA_ENCODING_AGX2,
   AGX_ISA_ENCODING_AGX3B,
   AGX_ISA_ENCODING_AGX3C,
   AGX_ISA_ENCODING_AGX3D,
   AGX_ISA_ENCODING_UNKNOWN,
};

/* Apple's independently-versioned command-stream/HAL axis.
 *
 * This names the layout of the records userspace writes -- VDM, PPP and CDM
 * words -- and not the kernel HAL that consumes them. G18G is served by
 * Apple's PI_400 kext, yet the records its Metal driver writes are the HAL300
 * ones; see agx_chip_hal300_records. */
enum agx_command_stream_generation {
   AGX_CS_GENERATION_AGX2,
   AGX_CS_GENERATION_G15,
   AGX_CS_GENERATION_HAL200,
   AGX_CS_GENERATION_HAL300,
   AGX_CS_GENERATION_UNKNOWN,
};

/* PBE/texture descriptor encoder generation. The descriptor grows from 24 to
 * 32 bytes at Gen9 and is laid out again at Gen10. */
enum agx_descriptor_generation {
   AGX_DESCRIPTOR_GENERATION_UNKNOWN = 0,
   AGX_DESCRIPTOR_GENERATION_7 = 7,
   AGX_DESCRIPTOR_GENERATION_8 = 8,
   AGX_DESCRIPTOR_GENERATION_9 = 9,
   AGX_DESCRIPTOR_GENERATION_10 = 10,
};

/* Samplers keep the Gen4 class through M4, but HAL200 moves fields within the
 * word. Gen5 adds reduction mode and a wider LOD clamp on M5, and G18G keeps
 * it. */
enum agx_sampler_generation {
   AGX_SAMPLER_GENERATION_GEN4,
   AGX_SAMPLER_GENERATION_GEN4_HAL200,
   AGX_SAMPLER_GENERATION_GEN5,
   AGX_SAMPLER_GENERATION_UNKNOWN,
};

/* M4 Pro/Max and A18 Pro carry both queueing paths; M5 and M6 are
 * SKSM-only. */
enum agx_submission_model {
   AGX_SUBMISSION_MODEL_CLASSIC,
   AGX_SUBMISSION_MODEL_DUAL,
   AGX_SUBMISSION_MODEL_SKSM,
   AGX_SUBMISSION_MODEL_UNKNOWN,
};

/* G17P keeps every colour target in the imageblock past 64 bytes a sample:
 * Apple's applegpu_g17p fragments store five RGBA32Float targets at tile
 * offsets 0..64 and eight at 0..112, and an eight-target mixed-format set at
 * 65 bytes, with no memory store, and its background builtins clear them at
 * the same offsets. */
#define AGX_G17P_MAX_BYTES_PER_SAMPLE (128)

/* G17G tile budget: a full 32 KiB, at most 2048 samples per tile. */
#define AGX_G17G_MAX_BYTES_PER_TILE   (32768)
#define AGX_G17G_MAX_SAMPLES_PER_TILE (2048)

/* On a G17P tile that no colour-writing geometry covered, the background
 * program -- clear or reload alike -- delivers only the low eight bytes of
 * each sample when the tilebuffer is multisampled. Measured on an A18 Pro:
 * with a sixteen-byte sample at 2x or 4x, such a texel reads back with its
 * upper eight bytes zero, while the same background program keeps the whole
 * sample on a tile a draw did cover -- including the pixels of that tile the
 * draw's scissor rejected. Eight-byte and narrower samples are unaffected at
 * every sample count, layer count and image size measured, and so is a
 * single-sample sixteen-byte target.
 *
 * Above this bound a load-op clear is emitted as a full-area draw instead,
 * which covers every tile and keeps the target in the tilebuffer. A reloaded
 * attachment has no such draw and spills, where the defect reaches it
 * (agx_chip_uncovered_tile_ms_reload_defect). A multiview render has a second,
 * separate defect that the full-area draw does not cover --
 * dEQP-VK.multiview.multisample.no_queries.max_multi_view_view_count, 32
 * views of RGBA32F at 4x, whose clear value is all zeroes so the background
 * clear cannot be what it observes -- and spills too.
 */
#define AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE (8)

/*
 * Whether Honeykrisp drives a chip through its native AGX3 path: AGX3b shaders
 * loaded by Ioto programs, VDM/PPP/CDM records in the HAL200 layout, Gen9
 * descriptors with heap-indexed textures and samplers, firmware-owned USC
 * private memory and no helper programs.
 *
 * Every native-path gate asks this question.  Where a chip on the path still
 * differs from G17P, a narrower capability predicate below answers for it;
 * the chip identity is never tested at the call site.
 */
#ifndef __OPENCL_VERSION__
#include <stdbool.h>
#endif

static inline bool
agx_chip_native_agx3(enum agx_chip chip)
{
   return chip == AGX_CHIP_G15G || chip == AGX_CHIP_G15S ||
          chip == AGX_CHIP_G15C || chip == AGX_CHIP_G16G ||
          chip == AGX_CHIP_G17P;
}

/* The chip whose record encodings GPU-generated (libagx) command-stream
 * records use.  Every native chip shares G17P's VDM records, so naming that
 * one family keeps the precompiled kernels independent of the native chip. */
static inline enum agx_chip
agx_chip_record_family(enum agx_chip chip)
{
   return agx_chip_native_agx3(chip) ? AGX_CHIP_G17P : chip;
}

#ifndef __OPENCL_VERSION__
#include <stdbool.h>
#endif

/*
 * Whether native captures on this chip witness the HAL300 records Mesa's M5
 * packers write: the VDM state, PPP state and Index List words, the nine PPP
 * graphics records, the viewport, W-clamp/cull2, depth/stencil and
 * depth-bias/scissor PPP records, the CDM launch and barrier, and the ZLS
 * pixel dimensions.
 *
 * G17G (M5) is where these packers were grounded. G18G (M6) is served by
 * Apple's PI_400 kext rather than a PI_300 one, but for the thirteen J873
 * native modes (runs/mini-26A428/sessions-20260925-{a,b}: color, points,
 * lines, discard, blend-add, depth-less, depth-always, stencil, msaa4, r32f,
 * r8ui, texture-sample and compute-add) Apple's G18G records and Mesa's G17G
 * packers agree byte for byte. The one layout difference is the direct
 * non-indexed Index List, where G18G emits the tailed form; see
 * agx_vdm_index_list_nonindexed_has_tail. Lines, which M5 never captured,
 * are packed for G18G alone, and one stencil face bit is modelled for
 * neither chip; see agx_ppp_hal300_graphics_core.h.
 *
 * This is a statement about the record layouts only. G17S/G17C also write
 * HAL300 records but have no native graphics capture, so the codecs that
 * accept them list them by name. It says nothing about the shader encoding
 * or the kernel submission ABI.
 */
static inline bool
agx_chip_hal300_records(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G || chip == AGX_CHIP_G18G;
}

/*
 * Whether the Gallium driver takes its M5 paths on this chip: draw admission,
 * native descriptors, the VDM/PPP streams and the M5 kernel contract. G18G
 * does: its records and texture/PBE descriptors are byte-identical to M5's
 * on every J873 native capture (agx_chip_hal300_records), its code is G17G
 * code re-encoded (agx_device_key::emit_agx3d), and the chip-specific deltas
 * (the tailed non-indexed Index List, the point-size rule) stay keyed on the
 * chip inside the record packers.
 */
static inline bool
agx_chip_m5_paths(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G || chip == AGX_CHIP_G18G;
}

/*
 * Whether an encoding is one of the AGX3 tables (agx3b, agx3c or agx3d).
 *
 * For compiler decisions about the AGX3 instruction set as such -- which IR
 * shapes the AGX3 packers can take at all -- that were written as
 * "AGX3B || AGX3C" when those were the only AGX3 tables. It says nothing about
 * which forms a table has grounded encoders for; those gates stay per
 * encoding, and an ungrounded AGX3D form is refused by its core encoder.
 */
static inline bool
agx_encoding_is_agx3(enum agx_isa_encoding encoding)
{
   return encoding == AGX_ISA_ENCODING_AGX3B ||
          encoding == AGX_ISA_ENCODING_AGX3C ||
          encoding == AGX_ISA_ENCODING_AGX3D;
}

/*
 * Whether an encoding belongs to the G17-class machine: the M5 G17G/G17S/G17C
 * parts (agx3c) and the M6 G18G (agx3d).
 *
 * Use this for machine behaviour -- scoreboard slot counts, register and
 * uniform limits, the IR lowerings Apple's G17 compiler also performs -- and
 * never for byte layout. agx3c and agx3d share the tables' anonymous field ids
 * and operand semantics (the G18 scheduling-unit list is the G17 one, MXU
 * included; __GPU_ARCH_LD_MD is identical for 41 of 41
 * G17G/G18G metal-nt pairs; G17G field vectors composed with the variant-2
 * table reproduce 1,111 of 1,547 aligned G18G instructions, the rest being
 * compiler choices), but their encoding points, lengths and bit positions
 * differ for about 80% of emitted instructions. So a gate that selects agx3c
 * field ids, encoding points, lengths or hand-written bytes stays
 * AGX_ISA_ENCODING_AGX3C, and agx3d bytes come only from agx3_compose with
 * the agx3d table (agx3_bind_compose with AGX3_BIND_APPLE_ORDER).
 */
static inline bool
agx_encoding_is_g17_class(enum agx_isa_encoding encoding)
{
   return encoding == AGX_ISA_ENCODING_AGX3C ||
          encoding == AGX_ISA_ENCODING_AGX3D;
}

#ifndef __OPENCL_VERSION__

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "drm-uapi/asahi_drm.h"

/* The AGX3 ISA family: G17P (AGX3B) and G17S (AGX3C) share instruction forms
 * and the AGX3 backend. Encoder/packer capability stays per encoding; the
 * shared compiler policy gates on the family. */
static inline bool
agx_isa_is_agx3(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C;
}

#ifdef __cplusplus
extern "C" {
#endif

/* Return the exact command-stream architecture, or UNKNOWN.  Recognition does
 * not imply a complete Honeykrisp upload-to-submit path.  Callers opening a
 * device must check agx_chip_driver_supported, not only UNKNOWN. */
enum agx_chip
agx_chip_from_params(const struct drm_asahi_params_global *params);

/* The instruction set a chip executes. */
enum agx_isa agx_chip_isa(enum agx_chip chip);

/* The binary encoding of that instruction set. Known for every chip in the
 * enum; UNKNOWN only for AGX_CHIP_UNKNOWN. */
enum agx_isa_encoding agx_chip_isa_encoding(enum agx_chip chip);

enum agx_command_stream_generation
agx_chip_command_stream_generation_reference(enum agx_chip chip);

enum agx_descriptor_generation
agx_chip_descriptor_generation_reference(enum agx_chip chip);

enum agx_sampler_generation
agx_chip_sampler_generation_reference(enum agx_chip chip);

unsigned agx_chip_sampler_heap_element_size_reference(enum agx_chip chip);

/* These are queried several times per recorded draw and dispatch. Answer them
 * with one bounded table load instead of an out-of-line call through a jump
 * table: an indirect branch per query is especially costly when the x86
 * driver build runs under FEX. */
#define AGX_CHIP_TABLE(G13G, G13X, G14G, G14X, G15P, G15G, G15S, G15C, G15D,  \
                       G16P, G16G, G16S, G16C, G17P, G17A, G17G, G17S, G17C,  \
                       G18G)                                                   \
   {                                                                           \
      G13G, G13X, G14G, G14X, G15P, G15G, G15S, G15C, G15D, G16P, G16G, G16S,  \
      G16C, G17P, G17A, G17G, G17S, G17C, G18G,                                \
   }

/* The table is positional (C++ translation units include this header and
 * reject array designators), so pin the enumerator order it relies on. */
static_assert(AGX_CHIP_G13G == 0 && AGX_CHIP_G13X == 1 && AGX_CHIP_G14G == 2 &&
                 AGX_CHIP_G14X == 3 && AGX_CHIP_G15P == 4 &&
                 AGX_CHIP_G15G == 5 && AGX_CHIP_G15S == 6 &&
                 AGX_CHIP_G15C == 7 && AGX_CHIP_G15D == 8 &&
                 AGX_CHIP_G16P == 9 && AGX_CHIP_G16G == 10 &&
                 AGX_CHIP_G16S == 11 && AGX_CHIP_G16C == 12 &&
                 AGX_CHIP_G17P == 13 && AGX_CHIP_G17A == 14 &&
                 AGX_CHIP_G17G == 15 && AGX_CHIP_G17S == 16 &&
                 AGX_CHIP_G17C == 17 && AGX_CHIP_G18G == 18 &&
                 AGX_CHIP_UNKNOWN == 19,
              "AGX_CHIP_TABLE is ordered by enum agx_chip");

static inline enum agx_command_stream_generation
agx_chip_command_stream_generation(enum agx_chip chip)
{
   static const uint8_t table[AGX_CHIP_UNKNOWN] = AGX_CHIP_TABLE(
      AGX_CS_GENERATION_AGX2, AGX_CS_GENERATION_AGX2, AGX_CS_GENERATION_AGX2,
      AGX_CS_GENERATION_AGX2,
      /* A16's command-stream class was not in the Mac corpus. */
      AGX_CS_GENERATION_UNKNOWN,
      AGX_CS_GENERATION_G15, AGX_CS_GENERATION_G15, AGX_CS_GENERATION_G15,
      AGX_CS_GENERATION_G15,
      AGX_CS_GENERATION_HAL200, AGX_CS_GENERATION_HAL200,
      AGX_CS_GENERATION_HAL200, AGX_CS_GENERATION_HAL200,
      AGX_CS_GENERATION_HAL200, AGX_CS_GENERATION_HAL200,
      AGX_CS_GENERATION_HAL300, AGX_CS_GENERATION_HAL300,
      AGX_CS_GENERATION_HAL300, AGX_CS_GENERATION_HAL300);
   return (unsigned)chip < AGX_CHIP_UNKNOWN
             ? (enum agx_command_stream_generation)table[chip]
             : AGX_CS_GENERATION_UNKNOWN;
}

static inline enum agx_descriptor_generation
agx_chip_descriptor_generation(enum agx_chip chip)
{
   static const uint8_t table[AGX_CHIP_UNKNOWN] = AGX_CHIP_TABLE(
      AGX_DESCRIPTOR_GENERATION_7, AGX_DESCRIPTOR_GENERATION_7,
      AGX_DESCRIPTOR_GENERATION_8, AGX_DESCRIPTOR_GENERATION_8,
      AGX_DESCRIPTOR_GENERATION_UNKNOWN,
      AGX_DESCRIPTOR_GENERATION_9, AGX_DESCRIPTOR_GENERATION_9,
      AGX_DESCRIPTOR_GENERATION_9, AGX_DESCRIPTOR_GENERATION_9,
      AGX_DESCRIPTOR_GENERATION_9, AGX_DESCRIPTOR_GENERATION_9,
      AGX_DESCRIPTOR_GENERATION_9, AGX_DESCRIPTOR_GENERATION_9,
      AGX_DESCRIPTOR_GENERATION_9, AGX_DESCRIPTOR_GENERATION_9,
      AGX_DESCRIPTOR_GENERATION_10, AGX_DESCRIPTOR_GENERATION_10,
      AGX_DESCRIPTOR_GENERATION_10, AGX_DESCRIPTOR_GENERATION_10);
   return (unsigned)chip < AGX_CHIP_UNKNOWN
             ? (enum agx_descriptor_generation)table[chip]
             : AGX_DESCRIPTOR_GENERATION_UNKNOWN;
}

static inline enum agx_sampler_generation
agx_chip_sampler_generation(enum agx_chip chip)
{
   static const uint8_t table[AGX_CHIP_UNKNOWN] = AGX_CHIP_TABLE(
      AGX_SAMPLER_GENERATION_GEN4, AGX_SAMPLER_GENERATION_GEN4,
      AGX_SAMPLER_GENERATION_GEN4, AGX_SAMPLER_GENERATION_GEN4,
      AGX_SAMPLER_GENERATION_UNKNOWN,
      AGX_SAMPLER_GENERATION_GEN4, AGX_SAMPLER_GENERATION_GEN4,
      AGX_SAMPLER_GENERATION_GEN4, AGX_SAMPLER_GENERATION_GEN4,
      AGX_SAMPLER_GENERATION_GEN4_HAL200, AGX_SAMPLER_GENERATION_GEN4_HAL200,
      AGX_SAMPLER_GENERATION_GEN4_HAL200, AGX_SAMPLER_GENERATION_GEN4_HAL200,
      AGX_SAMPLER_GENERATION_GEN4_HAL200, AGX_SAMPLER_GENERATION_GEN4_HAL200,
      AGX_SAMPLER_GENERATION_GEN5, AGX_SAMPLER_GENERATION_GEN5,
      AGX_SAMPLER_GENERATION_GEN5, AGX_SAMPLER_GENERATION_GEN5);
   return (unsigned)chip < AGX_CHIP_UNKNOWN
             ? (enum agx_sampler_generation)table[chip]
             : AGX_SAMPLER_GENERATION_UNKNOWN;
}

/* Native sampler heap element size. The packed state word remains eight
 * bytes, but G15 and later place it at offset zero of a 32-byte sparse heap
 * element. Zero means the heap ABI is not proven for this identity. */
static inline unsigned
agx_chip_sampler_heap_element_size(enum agx_chip chip)
{
   switch (agx_chip_command_stream_generation(chip)) {
   case AGX_CS_GENERATION_AGX2:
      return 8;
   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300:
      return 32;
   default:
      return 0;
   }
}

enum agx_submission_model
agx_chip_submission_model(enum agx_chip chip);

/*
 * Whether the native path runs on Apple's G15 command-stream generation
 * (M3, RTKit 2419) rather than HAL200.  The record layouts are shared; the
 * firmware-facing differences are the direct CDM launch pointer, the CDM
 * visibility barrier and the render-command resource specifiers.
 */
static inline bool
agx_chip_native_g15(enum agx_chip chip)
{
   return agx_chip_native_agx3(chip) &&
          agx_chip_command_stream_generation(chip) == AGX_CS_GENERATION_G15;
}

/* Whether the kernel presents the T8140 native render context: a fixed Bind0
 * and seven low context views at 0x1000000000 that every render job needs.
 * This is a HAL200 firmware construct; the G15 firmware has none. */
static inline bool
agx_chip_has_native_render_context(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17P;
}

/* Whether the native indirect compute pair (grid helper plus the 24-byte
 * mode-2 launch) is qualified.  G15 reads the same mode-2 record and the same
 * six-word geometry (three global thread counts, three local sizes); other
 * native chips refuse indirect dispatches rather than run an unqualified
 * record. */
static inline bool
agx_chip_native_indirect_compute(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17P || chip == AGX_CHIP_G15G ||
          chip == AGX_CHIP_G15S ||
          chip == AGX_CHIP_G15C || chip == AGX_CHIP_G16G;
}

/* Whether the CDM's fetch of an indirect geometry block needs a cache flush
 * after the dispatch that wrote it.  On G15 the ordinary launch tail is not
 * established for a shader-store to CDM-fetch edge, so the grid helper's tail
 * is followed by the wide flush before the mode-2 launch. */
static inline bool
agx_chip_indirect_geometry_flush(enum agx_chip chip)
{
   return agx_chip_native_g15(chip);
}

/*
 * Whether a CDM cache flush must use the wide-visibility barrier.  Apple's G15
 * encoder publishes a memory dependency between dispatches of one compute
 * stream with 0x600215fe, not the ordinary flags-5 word that suffices on
 * HAL200; the kernel's end-of-command flush does not cover dependencies
 * inside a stream.
 */
static inline bool
agx_chip_cdm_wide_visibility(enum agx_chip chip)
{
   return agx_chip_native_g15(chip);
}

/* M5 variants sharing the AGX3C instruction and HAL300 command formats. */
static inline bool
agx_chip_g17_m5(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
          chip == AGX_CHIP_G17C;
}

/* Whether consecutive application dispatches of one CDM stream may omit the
 * ordering barrier between them.  Vulkan orders dispatches only through
 * explicit barriers; every other record (internal or indirect launch, cache
 * flush, link, stream end) still follows a barrier.  On the native AGX3 CDM
 * (G15, HAL200 and M5 HAL300) the barrier holds the next launch until every
 * kernel launched before it has finished; without it the threadgroups of
 * consecutive launches run side by side. */
static inline bool
agx_chip_elide_app_dispatch_barrier(enum agx_chip chip)
{
   return agx_chip_native_agx3(chip) || agx_chip_g17_m5(chip);
}

/* Number of helper-table slots in the USC code-container lead block.  G15
 * and G16 parts have ten; with six their fragment programs of 256x256 and
 * larger walk past the lead into the guard page.  On G17P the state loader's
 * profile prefix calls the slot at 0x100 + 16 * core id (id = 16 * cluster +
 * core), so each table needs a slot per core of its cluster: six on the A18
 * Pro, ten on the M4 Pro, whose cluster 1 cores 8 and 9 otherwise run past
 * the table and never draw their tiles. */
static inline unsigned
agx_chip_usc_helper_table_slots(enum agx_chip chip, unsigned cores_per_cluster)
{
   if (chip != AGX_CHIP_G17P)
      return 10;
   return cores_per_cluster > 6 ? cores_per_cluster : 6;
}

/* Whether the USC window base page needs a return in every slot past its
 * helper tables.  On G15C the profile prefix's computed helper call can
 * index past both ten-slot tables on some shader cores; with zeroes there
 * the call never returns and runs into the unbound page after the base
 * page. */
static inline bool
agx_chip_usc_base_page_fill_returns(enum agx_chip chip)
{
   return chip == AGX_CHIP_G15C;
}

static inline bool
agx_chip_usc_base_page_fill_returns_for(enum agx_chip chip,
                                        unsigned num_clusters)
{
   return agx_chip_usc_base_page_fill_returns(chip) ||
          (agx_chip_native_agx3(chip) && num_clusters > 2);
}

/*
 * Whether the texture descriptor's last word (bytes 28-31) is ignored by the
 * texture unit and so free for the driver: Honeykrisp keeps a 2D view's
 * level-zero extent there (AGX_GEN10_IMAGE_EXTENT_OFFSET), so a constant
 * level-zero textureSize() is one load.
 *
 * True for the M5 Gen10 descriptor (G17G, G17S, G17C): the genxml layout has
 * no field there, Gallium's G17G buffer textures keep their element count in
 * it (AGX_GEN10_BUFFER_TEXTURE_SIZE_OFFSET), and G17C views carry the extent.
 * AGX2 descriptors are 24 bytes and have no such word. Nothing shows that
 * the Gen9 texture unit (G15, G16, G17P) or G18G's ignores it: no producer
 * writes a nonzero value there on either (G17P keeps its buffer metadata in
 * a separate record past the descriptor, and Gallium's Gen10 word is written
 * on M5 only), so they stay off until that is measured.
 */
static inline bool
agx_chip_texture_descriptor_spare_word(enum agx_chip chip)
{
   return agx_chip_descriptor_generation(chip) ==
             AGX_DESCRIPTOR_GENERATION_10 &&
          chip != AGX_CHIP_G18G;
}

/* Robust image accesses on native chips need explicit Vulkan view bounds:
 * native 2D accesses outside the view alias other texels of the image. G17P
 * lowers them together with its own hardware defects; G15 and G16 parts
 * bound the coordinates in NIR and take the out-of-bounds alpha from the
 * view swizzle. */
static inline bool
agx_chip_explicit_robust_image_bounds(enum agx_chip chip)
{
   return chip == AGX_CHIP_G16G || agx_chip_native_g15(chip);
}

/*
 * Whether the Apple9 imageblock applies: up to 128 bytes a sample in a full
 * 32 KiB (32768-byte) tile. Apple's Metal feature set tables give G15 (M3)
 * and G17P (A18 Pro) the same Apple9 limits.
 */
static inline bool
agx_chip_native_imageblock(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17P || chip == AGX_CHIP_G15G ||
          chip == AGX_CHIP_G15S ||
          chip == AGX_CHIP_G15C || chip == AGX_CHIP_G16G;
}

/*
 * Whether the defect applies that drops the upper bytes of a wide multisample
 * sample on a tile no geometry covered
 * (AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE). Measured on A18 Pro, and on
 * G15 (M3), where a layered 4x clear of a 16-byte format keeps only the low
 * eight bytes of each sample.
 */
static inline bool
agx_chip_uncovered_tile_ms_defect(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17P || chip == AGX_CHIP_G16G ||
          agx_chip_native_g15(chip);
}

/*
 * Whether that defect also reaches a reloaded attachment, which must then
 * spill. On M4 Pro and M4 Max (chip ids 0x6040, 0x6041) a 2x reload keeps the whole sample:
 * kept in the tilebuffer, 149 runs of 12- and 16-byte samples passed (64x64
 * to 3456x2234, 1 to 16 layers, partly drawn tiles, every empty tile
 * processed), where the same runs at 4x lost the upper bytes in 40 of 102.
 * A18 Pro, measured to lose them at 2x as well, keeps the spill.
 */
static inline bool
agx_chip_uncovered_tile_ms_reload_defect(enum agx_chip chip, uint32_t chip_id,
                                         unsigned samples)
{
   if (chip == AGX_CHIP_G17P && (chip_id == 0x6040 || chip_id == 0x6041) &&
       samples == 2)
      return false;
   return agx_chip_uncovered_tile_ms_defect(chip);
}

/* Whether a native fragment program may merge triangles across the
 * linker's other merge-disabling cases: memory writes and wide subgroup
 * operations. Apple's G17P metadata leaves the bit clear for both; on other
 * chips it is unverified, so they keep the linker's conservative bit. */
static inline bool
agx_chip_tri_merge_across_side_effects(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17P;
}

/* Whether an image load misses a store made earlier in the same invocation,
 * so Honeykrisp forwards the stored value to the load. A18 Pro's loads
 * behave this way; G15 loads observe the invocation's own stores. */
static inline bool
agx_chip_image_load_misses_own_store(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17P;
}

/* Whether native FP32 FFMA meets the exact rounding contract required by
 * VK_KHR_shader_fma. G15 and G16G remain unqualified for that contract;
 * G16G probes also fail near the minimum normal result. */
static inline bool
agx_chip_exact_fma32(enum agx_chip chip)
{
   return chip != AGX_CHIP_G16G && !agx_chip_native_g15(chip);
}

/* Whether the sampler state word carries the LOD bias.  The HAL200 sampler
 * layout has a bias field; Gen4 does not, so the shader applies the bias. */
static inline bool
agx_chip_sampler_has_lod_bias(enum agx_chip chip)
{
   return agx_chip_sampler_generation(chip) ==
          AGX_SAMPLER_GENERATION_GEN4_HAL200;
}

/*
 * Whether native shaders may carry an Ioto constant program (CProg) and a
 * shader preamble.  Only the G17P Ioto preamble sequence is qualified; other
 * native chips compile every stage with no_preamble.
 */
static inline bool
agx_chip_native_preamble(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17P;
}

/*
 * Whether a resource-info query reserves thread-invariant (TI) spill storage
 * in a native compute launch.  Apple's compiler stages a uniform query's
 * inputs for its constant program in TI storage, and the launch template
 * follows that witness (agx3_cdm_direct_template_sources_compute()).  Mesa
 * computes every query from descriptor memory, so only a constant program
 * could consume the storage: a chip without native preambles needs none, and
 * G15's direct launch has no TI field to carry it.
 */
static inline bool
agx_chip_query_ti_spill(enum agx_chip chip)
{
   return agx_chip_native_preamble(chip);
}

/*
 * Whether this driver can generate code for a chip.
 *
 * True for every encoding that has a packer: AGX2 through agx_pack.c,
 * AGX3b/AGX3c through agx3_pack_binary.c. False for AGX3d, whose table is
 * recovered and whose core encoders ground only a first instruction
 * catalogue, but for which agx3_pack_binary still refuses every shader (the
 * program-level pieces are not grounded), and for an unrecognised part, where
 * no packer can be chosen at all.
 *
 * This is a statement about the compiler, not about the driver. It says a
 * shader compiled for this chip is encoded by the right packer, and that an
 * ungrounded shader produces no binary rather than a mis-encoded one -- not
 * that the driver can submit the result. Device open must gate on
 * agx_chip_driver_supported.
 */
bool agx_chip_codegen_supported(enum agx_chip chip);

/* Whether every currently-required userspace axis is implemented. This is the
 * device-open gate; recognising a chip or having its ISA packer is not enough.
 */
bool agx_chip_driver_supported(enum agx_chip chip);

/* Validate the kernel-published identity as well as the userspace chip table.
 * G17P is the first identity whose runnable path depends on explicit USC and
 * HAL fields added to drm_asahi_params_global. */
bool agx_params_driver_supported(
   const struct drm_asahi_params_global *params);

/* Exact M6 (G18G) kernel contract admitted by default. This is not a
 * declaration of a complete or conformant device path. */
bool agx_params_g18g_bringup_supported(
   const struct drm_asahi_params_global *params);

/* Exact M5 (T8142, G17G), M5 Pro (T6050, G17S) and M5 Max (T6050, G17C)
 * kernel contracts the driver admits. This is not a declaration of a complete
 * or conformant device path. */
bool agx_params_m5_supported(
   const struct drm_asahi_params_global *params);

/* Reject the known incompatible pre-unification native kernel contract. */
bool agx_params_native_kernel_abi_supported(
   const struct drm_asahi_params_global *params);

/* Exact product name when the chip identity maps to one unambiguous SoC.
 * Variant letters are not marketing tiers: notably, G17P is A18 Pro rather
 * than an M5 variant. Returns NULL when one chip identity spans products. */
const char *agx_chip_exact_product_name(enum agx_chip chip);

bool agx_chip_uses_legacy_precompiled_programs(enum agx_chip chip);

bool agx_chip_has_hardware_ray_tracing(enum agx_chip chip);

bool agx_chip_needs_scheduling_stalls(enum agx_chip chip);

/* Every AGX3B part carries the 8x8 f16/f32 matmad backing a subgroup-scope
 * cooperative matrix. M5 has its own AGX3C encoding and matrix16 support. */
static inline bool
agx_chip_cooperative_matrix_supported(enum agx_chip chip)
{
   return agx_chip_isa_encoding(chip) == AGX_ISA_ENCODING_AGX3B ||
          agx_chip_g17_m5(chip);
}

/* Stable name for diagnostics, e.g. "G15X". Never NULL. */
const char *agx_chip_name(enum agx_chip chip);

/* Stable name for the instruction set, e.g. "AGX3". Never NULL. */
const char *agx_isa_name(enum agx_isa isa);

const char *agx_command_stream_generation_name(
   enum agx_command_stream_generation generation);

#ifdef __cplusplus
}
#endif

#endif

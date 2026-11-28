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
 * while other AGX3 identities stay outside the runnable Borealis path until
 * their independently-versioned pieces are connected.
 */

#ifndef __OPENCL_VERSION__
#include <stdbool.h>
#endif

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
 * lengths. Its table is in the tree (agx3_compose/agx3d_tables.txt and
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

/* The encoding of Ioto programs and their load, state, preshader and profile
 * records. It follows the instruction set: GPU0 for the AGX3b parts whose
 * programs are witnessed, GPU1 for the M5 parts. NONE refuses them. */
enum agx_ioto_generation {
   AGX_IOTO_GENERATION_NONE,
   AGX_IOTO_GENERATION_GPU0,
   AGX_IOTO_GENERATION_GPU1,
};

/* M4 Pro/Max and A18 Pro carry both queueing paths; M5 and M6 are
 * SKSM-only. */
enum agx_submission_model {
   AGX_SUBMISSION_MODEL_CLASSIC,
   AGX_SUBMISSION_MODEL_DUAL,
   AGX_SUBMISSION_MODEL_SKSM,
   AGX_SUBMISSION_MODEL_UNKNOWN,
};

/* Which of the driver's command paths a chip runs: the AGX2 one, the native
 * AGX3 one (agx_chip_native_agx3) or the HAL300 one of the M5 parts and G18G.
 * NONE is a recognised part that device open does not admit. */
enum agx_chip_path {
   AGX_CHIP_PATH_NONE,
   AGX_CHIP_PATH_AGX2,
   AGX_CHIP_PATH_NATIVE_AGX3,
   AGX_CHIP_PATH_HAL300,
};

/*
 * Command-stream records a chip is witnessed to take, one bit per record
 * codec. A codec refuses a chip without its bit, so a bit is set only with
 * captures or hardware results for that chip and that record. The codec's own
 * file says what the evidence is.
 */
enum agx_chip_record {
   AGX_RECORD_VDM_STATE = 1 << 0,
   AGX_RECORD_VDM_PPP_STATE = 1 << 1,
   AGX_RECORD_VDM_BARRIER = 1 << 2,
   AGX_RECORD_VDM_STREAM_LINK = 1 << 3,
   AGX_RECORD_VDM_STREAM_TERMINATE = 1 << 4,
   AGX_RECORD_VDM_TESSELLATE = 1 << 5,
   AGX_RECORD_VDM_INDEX_LIST = 1 << 6,
   AGX_RECORD_PPP_HAL300_DEPTH_STENCIL = 1 << 7,
   AGX_RECORD_PPP_HAL300_DEPTH_BIAS_SCISSOR = 1 << 8,
   AGX_RECORD_CDM_LAUNCH = 1 << 9,
   AGX_RECORD_CDM_INDIRECT_LAUNCH = 1 << 10,
   AGX_RECORD_CDM_INDIRECT_LAUNCH_36 = 1 << 11,
   AGX_RECORD_CDM_STREAM_LINK = 1 << 12,
   AGX_RECORD_CDM_LINK_PADDING = 1 << 13,
   AGX_RECORD_CDM_BARRIER = 1 << 14,
   /* G16G has it too, unlike agx_chip_cdm_wide_visibility(). */
   AGX_RECORD_CDM_BARRIER_WIDE_VISIBILITY = 1 << 15,
   AGX_RECORD_CDM_BARRIER_FLUSH_MEMORY = 1 << 16,
   AGX_RECORD_CDM_SUBSTREAM_BARRIER = 1 << 17,
   AGX_RECORD_CDM_SECONDARY_BARRIER = 1 << 18,
   AGX_RECORD_IOTO_PROFILE_CONTROL = 1 << 19,
   /* agx_chip_hal300_records */
   AGX_RECORD_HAL300_NATIVE_CAPTURE = 1 << 20,
   AGX_RECORD_PPP_VIEWPORT = 1 << 21,
   AGX_RECORD_PPP_W_CLAMP_CULL2 = 1 << 22,
   AGX_RECORD_VDM_STREAM_LINK_AGX2 = 1 << 23,
   /* The CDM ordering record (flags 0) with memory scope, 0x6000017c, which
    * agx_cdm_ordering_memory_scope() gives every M5 launch on a multi-die
    * device. Hardware results on G17D. */
   AGX_RECORD_CDM_ORDERING_FLUSH_MEMORY = 1 << 24,
};

/* AGX2 chips that use the genxml 8-byte non-returning VDM_STREAM_LINK. */
#define AGX_RECORDS_AGX2 (AGX_RECORD_VDM_STREAM_LINK_AGX2)

/* The records of every byte-witnessed G15, G16 and G17 identity. */
#define AGX_RECORDS_G15_G17                                                    \
   (AGX_RECORD_VDM_STATE | AGX_RECORD_VDM_PPP_STATE | AGX_RECORD_VDM_BARRIER | \
    AGX_RECORD_VDM_STREAM_LINK | AGX_RECORD_VDM_STREAM_TERMINATE |             \
    AGX_RECORD_VDM_TESSELLATE | AGX_RECORD_VDM_INDEX_LIST |                    \
    AGX_RECORD_PPP_VIEWPORT | AGX_RECORD_PPP_W_CLAMP_CULL2 |                   \
    AGX_RECORD_CDM_LAUNCH | AGX_RECORD_CDM_INDIRECT_LAUNCH |                   \
    AGX_RECORD_CDM_INDIRECT_LAUNCH_36 | AGX_RECORD_CDM_STREAM_LINK |           \
    AGX_RECORD_CDM_LINK_PADDING | AGX_RECORD_CDM_BARRIER |                     \
    AGX_RECORD_CDM_SUBSTREAM_BARRIER | AGX_RECORD_IOTO_PROFILE_CONTROL)

/* What the M5 parts add to them. */
#define AGX_RECORDS_M5                                                         \
   (AGX_RECORDS_G15_G17 | AGX_RECORD_PPP_HAL300_DEPTH_STENCIL |                \
    AGX_RECORD_PPP_HAL300_DEPTH_BIAS_SCISSOR |                                 \
    AGX_RECORD_CDM_BARRIER_FLUSH_MEMORY | AGX_RECORD_CDM_SECONDARY_BARRIER)

/* G18G has the records its J873 native captures contain plus the direct 8-byte
 * HAL300 VDM stream link verified on J873g hardware. The tessellate codec
 * still refuses it because no capture contains that record, and the rest stay
 * as their lists stood before this table (see each codec). */
#define AGX_RECORDS_G18G                                                       \
   (AGX_RECORD_VDM_STATE | AGX_RECORD_VDM_PPP_STATE | AGX_RECORD_VDM_BARRIER | \
    AGX_RECORD_VDM_STREAM_LINK | AGX_RECORD_VDM_STREAM_TERMINATE |             \
    AGX_RECORD_PPP_VIEWPORT | AGX_RECORD_PPP_W_CLAMP_CULL2 |                   \
    AGX_RECORD_PPP_HAL300_DEPTH_STENCIL |                                      \
    AGX_RECORD_PPP_HAL300_DEPTH_BIAS_SCISSOR | AGX_RECORD_CDM_LAUNCH |         \
    AGX_RECORD_CDM_INDIRECT_LAUNCH | AGX_RECORD_CDM_INDIRECT_LAUNCH_36 |       \
    AGX_RECORD_CDM_BARRIER | AGX_RECORD_HAL300_NATIVE_CAPTURE)

/* Per-chip behaviour that is not a generation axis. Each bit is read by the
 * one predicate below that is named after it, which documents it. */
enum agx_chip_quirk {
   AGX_QUIRK_NATIVE_RENDER_CONTEXT = 1 << 0,
   AGX_QUIRK_NATIVE_INDIRECT_COMPUTE = 1 << 1,
   AGX_QUIRK_USC_HELPER_SLOT_PER_CORE = 1 << 2,
   AGX_QUIRK_USC_BASE_PAGE_FILL_RETURNS = 1 << 3,
   AGX_QUIRK_EXPLICIT_ROBUST_IMAGE_BOUNDS = 1 << 4,
   AGX_QUIRK_NATIVE_IMAGEBLOCK = 1 << 5,
   AGX_QUIRK_UNCOVERED_TILE_MS_DEFECT = 1 << 6,
   AGX_QUIRK_TRI_MERGE_ACROSS_SIDE_EFFECTS = 1 << 7,
   AGX_QUIRK_IMAGE_LOAD_MISSES_OWN_STORE = 1 << 8,
   AGX_QUIRK_INEXACT_FMA32 = 1 << 9,
   /* Hardware-qualified HAL200/AGX3B/GPU0 Ioto CProg + preamble (G16G, G17P);
    * G15* uses RTKit 2419 (G15 CS gen, no CDM TI spill field) and is not
    * qualified here. */
   AGX_QUIRK_NATIVE_PREAMBLE = 1 << 10,
   AGX_QUIRK_SCHEDULING_STALLS = 1 << 11,
   AGX_QUIRK_M5_PATHS = 1 << 12,
   AGX_QUIRK_VDM_INDIRECT_INDEX_BOUNDS = 1 << 13,
   AGX_QUIRK_HW_TXF_BOUNDS = 1 << 14,
   AGX_QUIRK_M5_LAYER_OPERAND = 1 << 15,
   AGX_QUIRK_GL_THROUGH_ZINK = 1 << 16,
};

/* What the five native AGX3 chips share. */
#define AGX_QUIRKS_NATIVE                                                      \
   (AGX_QUIRK_NATIVE_INDIRECT_COMPUTE | AGX_QUIRK_NATIVE_IMAGEBLOCK |          \
    AGX_QUIRK_UNCOVERED_TILE_MS_DEFECT)

/* The native chips that are not G17P: G15G, G15S, G15C and G16G. */
#define AGX_QUIRKS_NATIVE_PRE_G17                                              \
   (AGX_QUIRKS_NATIVE | AGX_QUIRK_EXPLICIT_ROBUST_IMAGE_BOUNDS)

#define AGX_QUIRKS_G17P                                                        \
   (AGX_QUIRKS_NATIVE | AGX_QUIRK_NATIVE_RENDER_CONTEXT |                      \
    AGX_QUIRK_USC_HELPER_SLOT_PER_CORE |                                       \
    AGX_QUIRK_TRI_MERGE_ACROSS_SIDE_EFFECTS |                                  \
    AGX_QUIRK_IMAGE_LOAD_MISSES_OWN_STORE | AGX_QUIRK_NATIVE_PREAMBLE |        \
    AGX_QUIRK_VDM_INDIRECT_INDEX_BOUNDS)

/*
 * The chip table: one row per chip, and the only place that says what a chip
 * is. enum agx_chip and agx_chip_infos[] are both expanded from it, so they
 * cannot disagree, and every predicate below reads a field of the row.
 *
 * Columns: chip, instruction set, binary encoding, command-stream generation,
 * descriptor generation, sampler generation, Ioto generation, submission
 * model, path, witnessed records, quirks.
 *
 * AGX2: `G` is a single-cluster part, `X` is multi-cluster.
 *
 * G15P is the A16 phone part and still executes AGX2; its command-stream,
 * descriptor and sampler classes were not in the Mac driver corpus and are
 * not inferred. The Mac G15 parts execute AGX3b; G15G additionally needs
 * explicit scheduling nops. Every G16 part executes AGX3b.
 *
 * G17P/G17A remain on the HAL200/AGX3b side of the boundary. The M5 G/S/C/D
 * parts use AGX3c. G17D is the M5 Ultra: T6050 with two dies and eight
 * clusters.
 *
 * G17P is an instruction-set and firmware-HAL profile (AGX3b shaders, HAL200
 * firmware interface), not one SoC. agx_chip_from_params() maps three SoCs to
 * it: the A18 Pro (T8140, one cluster), the M4 Pro (T6040) and the M4 Max
 * (T6041). They share the encodings, so encoding gates on G17P are exact.
 * They differ in cluster and core counts, caches and clocks, so a performance
 * or tuning gate on G17P is only as good as the rows behind it: give it
 * measurements on its members (at least the A18 Pro and the M4 Pro), or
 * select it by the hardware property that is the real reason.
 *
 * G18G is the M6 base part (J873, T8152). Its shaders use a third AGX3
 * encoding, the agx3d table it shares with G18M and G19P. The other axes are
 * the M5 ones:
 *   - Records: HAL300. This is the record format, not the kernel HAL; the
 *     J873 native captures decode and repack under the HAL300 codecs (see
 *     agx_chip_hal300_records).
 *   - Descriptors: Gen10. The J873 colour render's PBE descriptor (ffe40a02
 *     00007f80 00005800 0003f010) and its texture descriptor (f6880a02
 *     0007f80f 00005800 000fc010) unpack as Gen10 and repack byte-identically.
 *   - Samplers: Gen5. The J873 texture-sample render's sampler (Metal's
 *     default descriptor with nearest filters and clamp-to-edge) is 000f0000
 *     00000780 at element 2 of a 32-byte heap. It repacks byte-identically as
 *     Gen5 and not as either Gen4 layout, whose maximum-LOD clamp of 14.0
 *     turns the 15.0 there into 000e0000.
 *   - Submission: on macOS 26A428 the G18G GPU has only SKSM channels, for
 *     3D, TA and compute work; there is no classic queue.
 *
 * No accelerator class for G15P, G16P or G17A was in the Mac corpus, so their
 * submission model is unknown.
 */
#define AGX_CHIPS(X, ...)                                                      \
   X(G13G, AGX2, AGX2, AGX2, 7, GEN4, NONE, CLASSIC, AGX2, AGX_RECORDS_AGX2,   \
     0, __VA_ARGS__)                                                           \
   X(G13X, AGX2, AGX2, AGX2, 7, GEN4, NONE, CLASSIC, AGX2, AGX_RECORDS_AGX2,   \
     0, __VA_ARGS__)                                                           \
   X(G14G, AGX2, AGX2, AGX2, 8, GEN4, NONE, CLASSIC, AGX2, AGX_RECORDS_AGX2,   \
     0, __VA_ARGS__)                                                           \
   X(G14X, AGX2, AGX2, AGX2, 8, GEN4, NONE, CLASSIC, AGX2, AGX_RECORDS_AGX2,   \
     0, __VA_ARGS__)                                                           \
   X(G15P, AGX2, AGX2, UNKNOWN, UNKNOWN, UNKNOWN, NONE, UNKNOWN, NONE, 0, 0,   \
     __VA_ARGS__)                                                              \
   X(G15G, AGX3, AGX3B, G15, 9, GEN4, GPU0, CLASSIC, NATIVE_AGX3,              \
     AGX_RECORDS_G15_G17 | AGX_RECORD_CDM_BARRIER_WIDE_VISIBILITY,             \
     AGX_QUIRKS_NATIVE_PRE_G17 | AGX_QUIRK_SCHEDULING_STALLS, __VA_ARGS__)     \
   X(G15S, AGX3, AGX3B, G15, 9, GEN4, GPU0, CLASSIC, NATIVE_AGX3,              \
     AGX_RECORDS_G15_G17 | AGX_RECORD_CDM_BARRIER_WIDE_VISIBILITY,             \
     AGX_QUIRKS_NATIVE_PRE_G17, __VA_ARGS__)                                   \
   X(G15C, AGX3, AGX3B, G15, 9, GEN4, GPU0, CLASSIC, NATIVE_AGX3,              \
     AGX_RECORDS_G15_G17 | AGX_RECORD_CDM_BARRIER_WIDE_VISIBILITY,             \
     AGX_QUIRKS_NATIVE_PRE_G17 | AGX_QUIRK_USC_BASE_PAGE_FILL_RETURNS,         \
     __VA_ARGS__)                                                              \
   X(G15D, AGX3, AGX3B, G15, 9, GEN4, GPU0, CLASSIC, NONE,                     \
     AGX_RECORDS_G15_G17, 0, __VA_ARGS__)                                      \
   X(G16P, AGX3, AGX3B, HAL200, 9, GEN4_HAL200, NONE, UNKNOWN, NONE, 0, 0,     \
     __VA_ARGS__)                                                              \
   X(G16G, AGX3, AGX3B, HAL200, 9, GEN4_HAL200, GPU0, CLASSIC, NATIVE_AGX3,    \
     AGX_RECORDS_G15_G17 | AGX_RECORD_CDM_BARRIER_WIDE_VISIBILITY,             \
     AGX_QUIRKS_NATIVE_PRE_G17 | AGX_QUIRK_NATIVE_PREAMBLE, __VA_ARGS__)       \
   X(G16S, AGX3, AGX3B, HAL200, 9, GEN4_HAL200, GPU0, DUAL, NONE,              \
     AGX_RECORDS_G15_G17, 0, __VA_ARGS__)                                      \
   X(G16C, AGX3, AGX3B, HAL200, 9, GEN4_HAL200, GPU0, DUAL, NONE,              \
     AGX_RECORDS_G15_G17, 0, __VA_ARGS__)                                      \
   X(G17P, AGX3, AGX3B, HAL200, 9, GEN4_HAL200, GPU0, DUAL, NATIVE_AGX3,       \
     AGX_RECORDS_G15_G17 | AGX_RECORD_CDM_BARRIER_FLUSH_MEMORY,                \
     AGX_QUIRKS_G17P, __VA_ARGS__)                                             \
   X(G17A, AGX3, AGX3B, HAL200, 9, GEN4_HAL200, NONE, UNKNOWN, NONE, 0, 0,     \
     __VA_ARGS__)                                                              \
   X(G17G, AGX3, AGX3C, HAL300, 10, GEN5, GPU1, SKSM, HAL300,                  \
     AGX_RECORDS_M5 | AGX_RECORD_HAL300_NATIVE_CAPTURE,                        \
     AGX_QUIRK_M5_PATHS | AGX_QUIRK_VDM_INDIRECT_INDEX_BOUNDS |                \
        AGX_QUIRK_HW_TXF_BOUNDS | AGX_QUIRK_TRI_MERGE_ACROSS_SIDE_EFFECTS,     \
     __VA_ARGS__)                                                              \
   X(G17S, AGX3, AGX3C, HAL300, 10, GEN5, GPU1, SKSM, HAL300, AGX_RECORDS_M5,  \
     AGX_QUIRK_VDM_INDIRECT_INDEX_BOUNDS | AGX_QUIRK_HW_TXF_BOUNDS |           \
        AGX_QUIRK_TRI_MERGE_ACROSS_SIDE_EFFECTS,                               \
     __VA_ARGS__)                                                              \
   X(G17C, AGX3, AGX3C, HAL300, 10, GEN5, GPU1, SKSM, HAL300, AGX_RECORDS_M5,  \
     AGX_QUIRK_VDM_INDIRECT_INDEX_BOUNDS | AGX_QUIRK_HW_TXF_BOUNDS |           \
        AGX_QUIRK_M5_LAYER_OPERAND |                                           \
        AGX_QUIRK_TRI_MERGE_ACROSS_SIDE_EFFECTS,                               \
     __VA_ARGS__)                                                              \
   X(G17D, AGX3, AGX3C, HAL300, 10, GEN5, GPU1, SKSM, HAL300,                  \
     AGX_RECORDS_M5 | AGX_RECORD_CDM_ORDERING_FLUSH_MEMORY,                    \
     AGX_QUIRK_GL_THROUGH_ZINK, __VA_ARGS__)                                   \
   X(G18G, AGX3, AGX3D, HAL300, 10, GEN5, NONE, SKSM, HAL300,                  \
     AGX_RECORDS_G18G, AGX_QUIRK_M5_PATHS, __VA_ARGS__)

enum agx_chip {
#define AGX_CHIP_ENUM(chip, ...) AGX_CHIP_##chip,
   AGX_CHIPS(AGX_CHIP_ENUM)
#undef AGX_CHIP_ENUM
   AGX_CHIP_UNKNOWN,
};

struct agx_chip_info {
   /* Stable name for diagnostics, e.g. "G15G". */
   char name[8];
   enum agx_isa isa;
   enum agx_isa_encoding encoding;
   enum agx_command_stream_generation cs_generation;
   enum agx_descriptor_generation descriptor_generation;
   enum agx_sampler_generation sampler_generation;
   enum agx_ioto_generation ioto_generation;
   enum agx_submission_model submission_model;
   enum agx_chip_path path;
   /* Mask of enum agx_chip_record. */
   unsigned records;
   /* Mask of enum agx_chip_quirk. */
   unsigned quirks;
};

#ifndef __OPENCL_VERSION__

/* Initialised by position: the C++ translation units that include this
 * header reject array and field designators. */
static const struct agx_chip_info agx_chip_infos[AGX_CHIP_UNKNOWN + 1] = {
#define AGX_CHIP_INFO(chip, isa, encoding, cs, descriptor, sampler, ioto,      \
                      submission, path, records, quirks, ...)                  \
   {#chip,                                                                     \
    AGX_ISA_##isa,                                                             \
    AGX_ISA_ENCODING_##encoding,                                               \
    AGX_CS_GENERATION_##cs,                                                    \
    AGX_DESCRIPTOR_GENERATION_##descriptor,                                    \
    AGX_SAMPLER_GENERATION_##sampler,                                          \
    AGX_IOTO_GENERATION_##ioto,                                                \
    AGX_SUBMISSION_MODEL_##submission,                                         \
    AGX_CHIP_PATH_##path,                                                      \
    records,                                                                   \
    quirks},
      AGX_CHIPS(AGX_CHIP_INFO)
#undef AGX_CHIP_INFO
      {"unknown", AGX_ISA_UNKNOWN, AGX_ISA_ENCODING_UNKNOWN,
       AGX_CS_GENERATION_UNKNOWN, AGX_DESCRIPTOR_GENERATION_UNKNOWN,
       AGX_SAMPLER_GENERATION_UNKNOWN, AGX_IOTO_GENERATION_NONE,
       AGX_SUBMISSION_MODEL_UNKNOWN, AGX_CHIP_PATH_NONE, 0, 0},
};

/* The row of a chip. AGX_CHIP_UNKNOWN and any value outside the enum get the
 * unknown row. The predicates below are queried several times per recorded
 * draw and dispatch, and this is one bounded table load: no out-of-line call
 * and no jump table, whose indirect branch is especially costly when the x86
 * driver build runs under FEX. */
static inline const struct agx_chip_info *
agx_chip_info(enum agx_chip chip)
{
   return &agx_chip_infos[(unsigned)chip < AGX_CHIP_UNKNOWN ? chip
                                                            : AGX_CHIP_UNKNOWN];
}

#endif

/*
 * The predicates the OpenCL units can call do not load from the table. The
 * precompiled libagx kernels get the chip as a run-time argument, and
 * asahi_clc does not finish compiling a kernel that indexes a table. A test
 * of the row instead expands to the comparisons of the chips whose row
 * passes it, which is what a hand-written chip list compiles to. The C units
 * use the same expansion, so tests/test-chip.c tests what the kernels run.
 * AGX_CHIPS(AGX_CHIP_ROW_TEST, chip, TEST, arg) is that expansion for the
 * rows that pass TEST(path, records, arg).
 */
#define AGX_CHIP_ROW_TEST(row, isa, encoding, cs, descriptor, sampler, ioto,   \
                          submission, path, records, quirks, chip, TEST, arg)  \
   | (((chip) == AGX_CHIP_##row) & TEST(AGX_CHIP_PATH_##path, records, arg))
#define AGX_CHIP_ROW_HAS_RECORD(path, records, record)                         \
   (((records) & (record)) != 0)
#define AGX_CHIP_ROW_ON_PATH(path, records, wanted) ((path) == (wanted))
#define AGX_CHIP_ROW_TEST_CS(row, isa, encoding, cs, descriptor, sampler,      \
                             ioto, submission, path, records, quirks, chip,    \
                             wanted_path, wanted_cs)                           \
   | (((chip) == AGX_CHIP_##row) &                                             \
      ((AGX_CHIP_PATH_##path) == (wanted_path)) &                              \
      ((AGX_CS_GENERATION_##cs) == (wanted_cs)))

/* Whether a chip is witnessed to take a record (enum agx_chip_record). */
static inline bool
agx_chip_has_record(enum agx_chip chip, unsigned record)
{
   return false AGX_CHIPS(AGX_CHIP_ROW_TEST, chip, AGX_CHIP_ROW_HAS_RECORD,
                          record);
}

/* Whether a chip is witnessed to take an 8-byte non-returning VDM stream link
 * (genxml AGX_VDM_STREAM_LINK on AGX2 or agx_vdm_stream_link_core on AGX3). */
static inline bool
agx_chip_vdm_stream_link_supported(enum agx_chip chip)
{
   return agx_chip_has_record(chip, AGX_RECORD_VDM_STREAM_LINK |
                                       AGX_RECORD_VDM_STREAM_LINK_AGX2);
}

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
static inline bool
agx_chip_native_agx3(enum agx_chip chip)
{
   return false AGX_CHIPS(AGX_CHIP_ROW_TEST, chip, AGX_CHIP_ROW_ON_PATH,
                          AGX_CHIP_PATH_NATIVE_AGX3);
}

/*
 * Whether the native AGX3 path runs on Apple's HAL200 command-stream
 * generation (G16G and G17P) rather than G15 (M3, RTKit 2419).
 *
 * HAL200 streams allow post-initial VDM State Updates to elide an unchanged
 * bind tail and allow standalone 8-byte PPP State Updates carrying only the
 * 28-byte fragment record.  Root-descriptor trimming and HAL200 libagx DGC
 * framing are also keyed on this predicate so G16G and G17P share the same
 * path while G15 stays on its G15 layout.
 */
static inline bool
agx_chip_native_hal200(enum agx_chip chip)
{
   return false AGX_CHIPS(AGX_CHIP_ROW_TEST_CS, chip,
                          AGX_CHIP_PATH_NATIVE_AGX3,
                          AGX_CS_GENERATION_HAL200);
}

/* The chip whose record encodings GPU-generated (libagx) command-stream
 * records use.  Every native chip shares G17P's VDM records, so naming that
 * one family keeps the precompiled kernels independent of the native chip. */
static inline enum agx_chip
agx_chip_record_family(enum agx_chip chip)
{
   return agx_chip_native_agx3(chip) ? AGX_CHIP_G17P : chip;
}

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
 * accept them have their own record bits. It says nothing about the shader
 * encoding or the kernel submission ABI.
 */
static inline bool
agx_chip_hal300_records(enum agx_chip chip)
{
   return agx_chip_has_record(chip, AGX_RECORD_HAL300_NATIVE_CAPTURE);
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

#include <stdint.h>

#include "drm-uapi/asahi_drm.h"

/* AGX3B (G15x/G16x/G17P/G17A) and AGX3C (G17G/S/C) share instruction forms
 * and the AGX3 backend. Intentionally excludes AGX3D (G18G). Encoder/packer
 * capability stays per encoding; shared compiler policy gates on this pair. */
static inline bool
agx_isa_is_agx3(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C;
}

/* drm_asahi_params_global::chip_id of the three SoCs that run as G17P. */
#define AGX_CHIP_ID_A18_PRO 0x8140
#define AGX_CHIP_ID_M4_PRO  0x6040
#define AGX_CHIP_ID_M4_MAX  0x6041

/* Whether the kernel runs the SoC with this chip id on its G17 runtime: AGX3
 * shaders behind the HAL200 firmware interface. */
static inline bool
agx_params_g17_runtime(const struct drm_asahi_params_global *params,
                       uint32_t chip_id)
{
   return params->chip_id == chip_id &&
          params->usc_generation == AGX_ISA_AGX3 &&
          params->gpu_hal_generation == DRM_ASAHI_GPU_HAL_200;
}

/* The chip the kernel's generation and variant name. agx_chip_from_params()
 * is this plus the checks device open needs; the command-stream decoder,
 * which does not link agx_chip.c, uses it directly. */
static inline enum agx_chip
agx_chip_from_identity(const struct drm_asahi_params_global *params)
{
   /* These are command-stream architecture identities, not an ordering.
    *
    * Generations 15 and up are recognised so diagnostics can name the part and
    * its command-stream generation. agx_chip_driver_supported decides whether
    * the driver proceeds; agx_chip_codegen_supported covers only the shader
    * packer.
    *
    * The variant character is load-bearing. Cluster count cannot distinguish
    * G17P (A18 Pro, AGX3b) from G17G (M5, AGX3c), and G15P is an AGX2 part in
    * the same numeric generation as AGX3 M3. The kernel already publishes the
    * exact P/G/S/C/D/A character, so use it directly.
    */
   switch (params->gpu_generation) {
   case 13:
      switch (params->gpu_variant) {
      case 'G': return AGX_CHIP_G13G;
      case 'S':
      case 'C':
      case 'D': return AGX_CHIP_G13X;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 14:
      switch (params->gpu_variant) {
      case 'G': return AGX_CHIP_G14G;
      case 'S':
      case 'C':
      case 'D': return AGX_CHIP_G14X;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 15:
      switch (params->gpu_variant) {
      case 'P': return AGX_CHIP_G15P;
      case 'G': return AGX_CHIP_G15G;
      case 'S': return AGX_CHIP_G15S;
      case 'C': return AGX_CHIP_G15C;
      case 'D': return AGX_CHIP_G15D;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 16:
      switch (params->gpu_variant) {
      case 'P': return AGX_CHIP_G16P;
      case 'G': return AGX_CHIP_G16G;
      case 'S':
         /* The M4 Pro (T6040) on the G17 kernel runtime takes the A18 Pro's
          * contract: its firmware has the A18 Pro firmware's interface, and
          * every G16S axis in these tables (AGX3B, HAL200, Gen9, Gen4-HAL200,
          * 32-byte sampler heap, dual submission) equals G17P's. The M4 Max
          * (T6041) below takes it for the same reason. G17P therefore names
          * the AGX3B + HAL200 profile of three SoCs, not the A18 Pro alone:
          * a performance gate on it needs rows on those parts (see enum
          * agx_chip).
          */
         if (agx_params_g17_runtime(params, AGX_CHIP_ID_M4_PRO))
            return AGX_CHIP_G17P;
         return AGX_CHIP_G16S;
      case 'C':
         if (agx_params_g17_runtime(params, AGX_CHIP_ID_M4_MAX))
            return AGX_CHIP_G17P;
         return AGX_CHIP_G16C;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 17:
      switch (params->gpu_variant) {
      case 'P': return AGX_CHIP_G17P;
      case 'A': return AGX_CHIP_G17A;
      case 'G': return AGX_CHIP_G17G;
      case 'S': return AGX_CHIP_G17S;
      case 'C': return AGX_CHIP_G17C;
      case 'D': return AGX_CHIP_G17D;
      default: return AGX_CHIP_UNKNOWN;
      }
   case 18:
      /* Only the M6 base part has been observed. G18M and G19P share its
       * shader encoding but nothing else about them is known. */
      switch (params->gpu_variant) {
      case 'G': return AGX_CHIP_G18G;
      default: return AGX_CHIP_UNKNOWN;
      }
   default:
      return AGX_CHIP_UNKNOWN;
   }
}

/* The chip a GPU device tree compatible of the form "apple,agx-g<generation>
 * <variant>" names ("apple,agx-g17d" is G17D), through agx_chip_from_identity()
 * given the generation and variant alone; UNKNOWN for any other string, such
 * as an SoC compatible ("apple,agx-t6050"). A compatible carries no SoC ID, so
 * where agx_chip_from_identity() also reads the kernel's chip ID (the M4 Pro
 * and M4 Max on the G17 runtime, G17P there) this gives the generation's own
 * row (G16S, G16C): for a chip-wide bit only, such as
 * agx_chip_gl_through_zink(), which none of those rows has. For code that
 * reads the device tree without opening the device, such as the loader.
 */
static inline enum agx_chip
agx_chip_from_compatible(const char *compatible)
{
   static const char prefix[] = "apple,agx-g";
   struct drm_asahi_params_global params = {0};
   const char *c = compatible;

   for (unsigned i = 0; i < sizeof(prefix) - 1; ++i, ++c) {
      if (!c || *c != prefix[i])
         return AGX_CHIP_UNKNOWN;
   }
   if (*c < '1' || *c > '9')
      return AGX_CHIP_UNKNOWN;
   for (; *c >= '0' && *c <= '9'; ++c) {
      params.gpu_generation = params.gpu_generation * 10 + (*c - '0');
      if (params.gpu_generation > 99)
         return AGX_CHIP_UNKNOWN;
   }
   if (*c < 'a' || *c > 'z' || c[1] != '\0')
      return AGX_CHIP_UNKNOWN;
   params.gpu_variant = *c - 'a' + 'A';
   return agx_chip_from_identity(&params);
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
static inline enum agx_isa
agx_chip_isa(enum agx_chip chip)
{
   return agx_chip_info(chip)->isa;
}

/* The binary encoding of that instruction set: which encoder table a part
 * uses. Apple's own tables decode each family 100% under exactly one of them
 * and essentially 0% under the others, so this is a measurement rather than a
 * reading of the naming scheme. Known for every chip in the enum; UNKNOWN
 * only for AGX_CHIP_UNKNOWN. */
static inline enum agx_isa_encoding
agx_chip_isa_encoding(enum agx_chip chip)
{
   return agx_chip_info(chip)->encoding;
}

static inline enum agx_command_stream_generation
agx_chip_command_stream_generation(enum agx_chip chip)
{
   return agx_chip_info(chip)->cs_generation;
}

/* The flag word the AGX2 CDM barrier record carries when every bit is set:
 * unk_0..19 plus the USC cache invalidate, the record the shared
 * agx_cdm_barrier() emitter produces. */
#define AGX_CDM_BARRIER_FULL_WORD 0xfffffu

/* Pure compute-to-compute SSBO CDM barrier word on AGX2: bits {5,6,8}
 * (cross-core L1/L2 memory coherency + execution drain) without bit 3
 * (USC cache invalidate) or PBE/texture-cache maintenance bits {0,1,2,4,7}.
 * Valid only between compute launches in the same open CDM stream when no
 * images, textures, samplers, UBOs, transfers, or host accesses intervene. */
#define AGX_CDM_BARRIER_SSBO_WORD 0x160u

/* The default flag word of the AGX2 CDM barrier record Borealis emits at a
 * tracked compute dependency (hk_cdm_flush()). The reduced word 0x1ff
 * (unk_0..8, PBE/texture-cache maintenance included) is the set measured on
 * the two AGX2 X dies; every other chip keeps the full word. The gate is a
 * chip property rather than a measured property because the reduced set is a
 * die-level drain characteristic: the two chips it was proven on are the
 * only AGX2 parts the lab holds. The word alone does not decide when a
 * barrier is emitted, and a device that created a sparse resource upgrades
 * to the full word at emit or submit (hk_cdm_flush(),
 * hk_submit_patch_sparse_sites()). */
static inline uint32_t
agx2_cdm_barrier_default_word(enum agx_chip chip, uint32_t gpu_generation)
{
   if (chip == AGX_CHIP_G13G || chip == AGX_CHIP_G13X ||
       (chip == AGX_CHIP_G14X && gpu_generation == 14))
      return 0x1ffu;
   return AGX_CDM_BARRIER_FULL_WORD;
}

/* The reduced AGX2 CDM barrier word for pure compute-to-compute storage-buffer
 * dependencies inside an open CDM stream (hk_CmdPipelineBarrier2 /
 * hk_cdm_flush()). Derived from the device's measured default word so no chip
 * list is added: parts already on the 0x1ff default word drop bit 3 (USC cache
 * invalidate) and PBE/texture-cache maintenance bits {0,1,2,4,7} to emit
 * 0x160; parts still on the full word keep their default word unchanged. */
static inline uint32_t
agx2_cdm_barrier_ssbo_word(uint32_t default_word)
{
   return default_word == 0x1ffu ? AGX_CDM_BARRIER_SSBO_WORD : default_word;
}

/* The word emitted at a tracked compute dependency (hk_cdm_flush()): the
 * full word once the device holds a sparse resource or under
 * fullcdmbarrier, the caller-selected chip word otherwise. */
static inline uint32_t
agx2_cdm_barrier_emit_word(uint32_t default_word, bool sparse_used,
                           bool force_full)
{
   return (sparse_used || force_full) ? AGX_CDM_BARRIER_FULL_WORD
                                      : default_word;
}

static inline enum agx_descriptor_generation
agx_chip_descriptor_generation(enum agx_chip chip)
{
   return agx_chip_info(chip)->descriptor_generation;
}

static inline enum agx_sampler_generation
agx_chip_sampler_generation(enum agx_chip chip)
{
   return agx_chip_info(chip)->sampler_generation;
}

/*
 * Whether an indirect indexed VDM Index List bounds its index fetch in
 * hardware the way robustBufferAccess2 requires: indices past the record's
 * index-buffer size (index_buffer_size_word, dwords minus one) read as 0.
 *
 * What is established, and by what:
 *   - The record carries that word on every AGX3 command-stream generation:
 *     the G15 and HAL200 records (agx_vdm_hal200_indirect_draw_pack) and the
 *     HAL300 one, which adds only the Unk 2 word before it
 *     (agx_vdm_index_list_indirect_pack_words). That is packet layout.
 *   - Enabled for the M5 AGX3C family (G17G, G17S, G17C; agx_chip_g17_m5()):
 *     zero substitution for out-of-range indices is measured on G17C (J716c)
 *     and G17S (J714s) by
 *     dEQP-VK.robustness.{index_access,vertex_access.*index*,robustness2}.*,
 *     dEQP-VK.draw.{*indexed*,renderpass.multi_draw.*}, and a 28-case
 *     bit-exact equivalence probe against the software rewrite (partial and
 *     full out-of-range draws, 32-bit firstIndex wrap, signed vertexOffset,
 *     multi-draw firstInstance, primitive restart, indirect count).
 *   - Enabled for G17P (HAL200): zero substitution past index_buffer_size_word
 *     is measured on A18 Pro (G17P B1) by the same 28-case bit-exact
 *     equivalence probe and all five Vulkan CTS groups above.
 *
 * Off elsewhere:
 *   - G17D (M5 Ultra) keeps the software rewrite until measured on the J775D.
 *   - G18G uses HAL300 records too, but its out-of-range hardware index
 *     behaviour is unmeasured, so it keeps the software rewrite.
 *   - G15 and G16G (G15/HAL200 records) keep the software rewrite until
 *     hardware rows (M3, M4) prove zero substitution past the size word.
 *   - AGX2 does not have it: its Index List does not read out-of-range
 *     indices as zero (libagx_draw_robust_index).
 *
 * The bound is dword granular, so callers still rewrite draws whose range
 * does not end on a dword: 8-bit indices and unaligned 16-bit ranges.
 */
static inline bool
agx_chip_vdm_indirect_index_bounds(enum agx_chip chip)
{
   return agx_chip_info(chip)->quirks & AGX_QUIRK_VDM_INDIRECT_INDEX_BOUNDS;
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

static inline enum agx_ioto_generation
agx_chip_ioto_generation(enum agx_chip chip)
{
   return agx_chip_info(chip)->ioto_generation;
}

static inline enum agx_submission_model
agx_chip_submission_model(enum agx_chip chip)
{
   return agx_chip_info(chip)->submission_model;
}

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
   return agx_chip_info(chip)->quirks & AGX_QUIRK_NATIVE_RENDER_CONTEXT;
}

/* Whether the native indirect compute pair (grid helper plus the 24-byte
 * mode-2 launch) is qualified.  G15 reads the same mode-2 record and the same
 * six-word geometry (three global thread counts, three local sizes); other
 * native chips refuse indirect dispatches rather than run an unqualified
 * record. */
static inline bool
agx_chip_native_indirect_compute(enum agx_chip chip)
{
   return agx_chip_info(chip)->quirks & AGX_QUIRK_NATIVE_INDIRECT_COMPUTE;
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
 * G15 only, unlike AGX_RECORD_CDM_BARRIER_WIDE_VISIBILITY, which G16G has.
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
   const struct agx_chip_info *info = agx_chip_info(chip);

   return info->encoding == AGX_ISA_ENCODING_AGX3C &&
          info->cs_generation == AGX_CS_GENERATION_HAL300;
}

/*
 * Whether the Gallium driver takes its M5 paths on this chip: draw admission,
 * native descriptors, the VDM/PPP streams and the M5 kernel contract. Both
 * drivers ask it for the G17G-form compute dispatch, and the Vulkan driver
 * for its G17G-form graphics program. G18G does: its records and texture/PBE
 * descriptors are byte-identical to M5's on every J873 native capture
 * (agx_chip_hal300_records), its code is G17G code re-encoded
 * (agx_device_key::emit_agx3d), and the chip-specific deltas (the tailed
 * non-indexed Index List, the point-size rule) stay keyed on the chip inside
 * the record packers.
 *
 * These are the chips of agx_chip_hal300_records() today. That one says which
 * record layouts native captures witness, this one which paths the driver
 * runs, and each has its own bit, so that neither widens with the other.
 */
static inline bool
agx_chip_m5_paths(enum agx_chip chip)
{
   return agx_chip_info(chip)->quirks & AGX_QUIRK_M5_PATHS;
}

/*
 * Whether an M5-form store to a layer of an array takes a layer operand. The
 * M5 Max (G17C) does, and its array descriptors keep the array form. The
 * other M5 parts store the layer as the z coordinate of a 3D store: the
 * shader packer (agx3_pack_binary.c), Borealis's image views and Gallium's
 * PBE read this one bit so they cannot disagree, and Borealis's store window
 * past 2048 slices (hk_nir_lower_descriptors()) follows the image view's
 * layers_as_3d choice. G18G (compiled as G17G) is outside agx_chip_g17_m5(),
 * which Borealis also requires; its layered stores are its owner's to settle.
 */
static inline bool
agx_chip_m5_layer_operand(enum agx_chip chip)
{
   return agx_chip_info(chip)->quirks & AGX_QUIRK_M5_LAYER_OPERAND;
}

/*
 * Whether GL on this chip runs through zink over Borealis rather than the
 * asahi Gallium driver, whose paths have not run there: the Gallium screen
 * refuses the chip (agx_screen_create()), and the loader picks zink for it.
 * The M5 Ultra (G17D) today.
 */
static inline bool
agx_chip_gl_through_zink(enum agx_chip chip)
{
   return agx_chip_info(chip)->quirks & AGX_QUIRK_GL_THROUGH_ZINK;
}

/*
 * Chip sets that call sites test without a capability of their own. They say
 * which chips and not why, and they are not interchangeable: three of them
 * are "runs an AGX3 path" with G18G or the M5 parts left out. Each replaced
 * the same comparisons written out at the call sites, so a site keeps the set
 * it had. A new site that knows why it asks should get a predicate named for
 * that reason instead.
 */

/* The HAL300 path: the M5 parts and G18G. */
static inline bool
agx_chip_hal300_path(enum agx_chip chip)
{
   return agx_chip_info(chip)->path == AGX_CHIP_PATH_HAL300;
}

/* Either AGX3 path: native AGX3, the M5 parts or G18G. */
static inline bool
agx_chip_agx3_path(enum agx_chip chip)
{
   const enum agx_chip_path path = agx_chip_info(chip)->path;

   return path == AGX_CHIP_PATH_NATIVE_AGX3 || path == AGX_CHIP_PATH_HAL300;
}

/* Native AGX3 or an M5 part. Not G18G. */
static inline bool
agx_chip_native_or_m5(enum agx_chip chip)
{
   return agx_chip_native_agx3(chip) || agx_chip_g17_m5(chip);
}

/* Native AGX3 or G18G. Not the M5 parts. */
static inline bool
agx_chip_native_or_g18g(enum agx_chip chip)
{
   return agx_chip_native_agx3(chip) || chip == AGX_CHIP_G18G;
}

/* G17P or an M5 part. Not the other native chips and not G18G. */
static inline bool
agx_chip_g17p_or_m5(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17P || agx_chip_g17_m5(chip);
}

/* Whether hardware/firmware orders command start stamps after a command's
 * waits. G17P and M5 (G17G/G17S/G17C) latch start stamps when firmware takes
 * the command, before barriers resolve, so Borealis places timestamps on
 * command completion (end stamps) instead. */
static inline bool
agx_chip_start_stamps_ordered(enum agx_chip chip)
{
   return !agx_chip_g17p_or_m5(chip);
}

/* Whether consecutive application dispatches of one CDM stream may omit the
 * ordering barrier between them.  Vulkan orders dispatches only through
 * explicit barriers; every other record (internal launch, non-M5 indirect
 * launch, cache flush, link, stream end) still follows a barrier.  On the
 * native AGX3 CDM (G15, HAL200 and M5 HAL300) the barrier holds the next
 * launch until every kernel launched before it has finished; without it the
 * threadgroups of consecutive launches run side by side. */
static inline bool
agx_chip_elide_app_dispatch_barrier(enum agx_chip chip)
{
   return agx_chip_native_or_m5(chip);
}

/* Whether the Borealis graphics path supplies the depth/stencil feedback-loop
 * shader twin (hk_fbl_twin()) and draw-time dynamic transition
 * (hk_select_fbl_fs()) required for VK_EXT_attachment_feedback_loop_dynamic_state.
 * Shared across native AGX3 (G15/G16/G17P) and M5 (G17G/G17S/G17C). */
static inline bool
agx_chip_has_feedback_loop_dynamic_state(enum agx_chip chip)
{
   return agx_chip_native_or_m5(chip);
}

/* Whether VK_EXT_depth_clip_control (and
 * extendedDynamicState3DepthClipNegativeOneToOne) is qualified on this chip.
 * Lowered in NIR (agx_nir_lower_clip_m1_1) via the shared root table across
 * AGX2, native AGX3, and AGX3C/HAL300 (M5: G17G/G17S/G17C). Other HAL300
 * encodings such as AGX3D (G18G) stay off until qualified on their hardware. */
static inline bool
agx_chip_has_depth_clip_control(enum agx_chip chip)
{
   return !agx_chip_hal300_path(chip) || agx_chip_g17_m5(chip);
}

/* Whether Borealis enables shader-reconstructed single-plane packed 4:2:2
 * YCbCr formats (8/10/12/16-bit views backed by plain RGBA word textures and
 * reconstructed in NIR) and 10/12-bit YCbCr plane UNORM range restoration.
 * Shared across native AGX3 (G15/G16/G17P) and M5 (G17G/G17S/G17C); AGX2 keeps
 * its existing path and other HAL300 encodings such as AGX3D (G18G) stay off
 * until qualified on their hardware. */
static inline bool
agx_chip_has_packed_ycbcr(enum agx_chip chip)
{
   return agx_chip_native_or_m5(chip);
}

/* Whether hardware ASTC HDR decode is qualified on this chip: AGX2, native
 * AGX3, and M5 (AGX3C/HAL300: G17G/G17S/G17C). Other HAL300 encodings such as
 * AGX3D (G18G) stay off until qualified on their hardware. */
static inline bool
agx_chip_has_astc_hdr(enum agx_chip chip)
{
   return !agx_chip_hal300_path(chip) || agx_chip_g17_m5(chip);
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
   if (!(agx_chip_info(chip)->quirks & AGX_QUIRK_USC_HELPER_SLOT_PER_CORE))
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
   return agx_chip_info(chip)->quirks & AGX_QUIRK_USC_BASE_PAGE_FILL_RETURNS;
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
 * True for the M5 Gen10 descriptor (G17G, G17S, G17C, G17D): the genxml
 * layout has no field there, Gallium's G17G buffer textures keep their
 * element count in it (AGX_GEN10_BUFFER_TEXTURE_SIZE_OFFSET), and G17C views
 * carry the extent.
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
   return agx_chip_info(chip)->quirks & AGX_QUIRK_EXPLICIT_ROBUST_IMAGE_BOUNDS;
}

/*
 * Whether the texture unit bounds an integer texel fetch to the Vulkan view
 * and returns the robustImageAccess2 value (zero, alpha one for a missing
 * channel) for out-of-range spatial coordinates and 16-bit array layers,
 * given a valid LOD and sample index. Shaders then only saturate wider
 * layer/LOD/sample operands before 16-bit narrowing and validate nonzero
 * LODs and multisample sample indices against view metadata
 * (AGX_NIR_TEXTURE_HW_TXF_BOUNDS), instead of predicating every fetch on
 * the descriptor's spatial size.
 *
 * Enabled for the M5 Gen10 texture unit (G17G, G17S, G17C):
 *   - G17C (J716c): established by robustness and texelFetch Vulkan CTS
 *     suites plus a 33-case out-of-range probe (LOD and sample 65536/-65536,
 *     layer 65536, negative and large x/y/z, restricted subresource views,
 *     null descriptors, float and integer missing alpha).
 *   - G17G and G17S share the same M5 Gen10 texture unit; the existing G17G
 *     measurement (be91d344340: out-of-view 2D-array layer returns zero and
 *     layer 65536 wraps to layer 0) agrees with G17C. Full probe and CTS
 *     verification on M5 Pro (G17S) / M5 (G17G) is the merge gate.
 *
 * Off elsewhere:
 *   - G17D (M5 Ultra) keeps the software predicate until measured on the
 *     J775D.
 *   - G18G uses the Gen10 descriptor layout too, but its texture lowering
 *     (agx_nir_lower_texture_for_generation) always applies the software
 *     predicate and its behaviour is not measured, so it stays off so that
 *     no caller can skip the predicate there.
 *   - Gen9 parts (G15, G16, G17P) keep the software bounds: native 2D
 *     accesses outside the view alias other texels of the image on G17P
 *     (agx_chip_explicit_robust_image_bounds and the G17P lowering).
 */
static inline bool
agx_chip_hw_texel_fetch_bounds(enum agx_chip chip)
{
   return agx_chip_info(chip)->quirks & AGX_QUIRK_HW_TXF_BOUNDS;
}

/*
 * Whether the Apple9 imageblock applies: up to 128 bytes a sample in a full
 * 32 KiB (32768-byte) tile, or 48 KiB where the kernel admits it
 * (AGX_NATIVE_TILE_BUFFER_48K_BYTES). Apple's Metal feature set tables give
 * G15 (M3) and G17P (A18 Pro) the same Apple9 limits.
 */
static inline bool
agx_chip_native_imageblock(enum agx_chip chip)
{
   return agx_chip_info(chip)->quirks & AGX_QUIRK_NATIVE_IMAGEBLOCK;
}

/* Tile buffer budget of a render on chips with the native imageblock. A
 * kernel that reports DRM_ASAHI_FEATURE_TILE_BUFFER_48K admits 48 KiB. With
 * 24-byte samples at 4x that fits 32x16 tiles instead of 16x16, so a render
 * bins half as many tiles.
 */
#define AGX_NATIVE_TILE_BUFFER_BYTES     (32768)
#define AGX_NATIVE_TILE_BUFFER_48K_BYTES (49152)

/*
 * Whether single-sample 2D colour images are losslessly compressed in the
 * Gen9 form Borealis implements for G17P. The image is GPU-tiled, and 8 bytes
 * of metadata per 16x16 tile follow it in the same allocation. The texture
 * descriptor sets bit 59 (compressed) and bit 127 (extended) and carries the
 * metadata address >> 4 in its third word. The PBE sets word 0 bit 59 and
 * word 1 bit 63, with the same third word.
 *
 * G16G: macOS 27.2 (26B123) on the M4 allocates a 4096x4096 RGBA8 texture as
 * 64 MiB of image plus 512 KiB of metadata, and its blit copy into that
 * texture reads and writes it through these exact descriptors. The capture
 * shows no compressed depth, multisampled or layered image, so those stay on
 * agx_chip_gen9_extended_compression().
 *
 * The selection is by chip because no device property reports compression.
 */
static inline bool
agx_chip_gen9_color_compression(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17P || chip == AGX_CHIP_G16G;
}

/*
 * The compressed shapes beyond single-sample colour, measured on the G17P
 * parts (A18 Pro, M4 Pro, M4 Max):
 *  - depth and stencil through the ZLS unit;
 *  - 2x and 4x attachments;
 *  - the M4 Pro's layered, cube and 3D shapes.
 */
static inline bool
agx_chip_gen9_extended_compression(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17P;
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
   return agx_chip_info(chip)->quirks & AGX_QUIRK_UNCOVERED_TILE_MS_DEFECT;
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
   if (chip == AGX_CHIP_G17P &&
       (chip_id == AGX_CHIP_ID_M4_PRO || chip_id == AGX_CHIP_ID_M4_MAX) &&
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
   return agx_chip_info(chip)->quirks & AGX_QUIRK_TRI_MERGE_ACROSS_SIDE_EFFECTS;
}

/* Whether an AGX3 chip supplies the quad ballot/shuffle encodings and
 * draw-time triangle-merge control required by VK_KHR_shader_quad_control.
 * Native AGX3B (G15/G16/G17P) and M5 AGX3C (G17G/G17S/G17C) share both;
 * AGX2 uses its legacy path and bring-up G18G stays off. */
static inline bool
agx_chip_agx3_shader_quad_control(enum agx_chip chip)
{
   return agx_chip_native_or_m5(chip);
}

/* Whether Vulkan VK_KHR_shader_quad_control / shaderQuadControl is supported
 * on this chip (AGX2 native support, or AGX3 with quad encodings and
 * draw-time triangle-merge control). */
static inline bool
agx_chip_has_shader_quad_control(enum agx_chip chip)
{
   return agx_chip_isa(chip) == AGX_ISA_AGX2 ||
          agx_chip_agx3_shader_quad_control(chip);
}

/* Whether an image load misses a store made earlier in the same invocation,
 * so Honeykrisp forwards the stored value to the load. A18 Pro's loads
 * behave this way; G15 loads observe the invocation's own stores. */
static inline bool
agx_chip_image_load_misses_own_store(enum agx_chip chip)
{
   return agx_chip_info(chip)->quirks & AGX_QUIRK_IMAGE_LOAD_MISSES_OWN_STORE;
}

/* Whether native FP32 FFMA meets the exact rounding contract required by
 * VK_KHR_shader_fma. G15 and G16G remain unqualified for that contract;
 * G16G probes also fail near the minimum normal result. */
static inline bool
agx_chip_exact_fma32(enum agx_chip chip)
{
   return !(agx_chip_info(chip)->quirks & AGX_QUIRK_INEXACT_FMA32);
}

/* Whether agx_nir_lower_denorm_preserve() guarantees FP32 DenormPreserve
 * across every advertised shader stage and encoding: native AGX3 (AGX3B) and
 * M5 (AGX3C), where agx_compile_shader_nir() runs the exact integer lowering
 * unconditionally. AGX2 and bring-up G18G stay off. */
static inline bool
agx_chip_has_fp32_denorm_preserve(enum agx_chip chip)
{
   const enum agx_isa_encoding enc = agx_chip_isa_encoding(chip);
   return (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C) &&
          agx_chip_native_or_m5(chip);
}

/* Whether the sampler state word carries the LOD bias.  The HAL200 and Gen5
 * sampler layouts have a bias field; Gen4 does not, so the shader applies it. */
static inline bool
agx_chip_sampler_has_lod_bias(enum agx_chip chip)
{
   const enum agx_sampler_generation gen = agx_chip_sampler_generation(chip);
   return gen == AGX_SAMPLER_GENERATION_GEN4_HAL200 ||
          gen == AGX_SAMPLER_GENERATION_GEN5;
}

/*
 * Whether native shaders may carry an Ioto constant program (CProg) and a
 * shader preamble.  G16G and G17P share HAL200 (cs_generation = HAL200,
 * sampler_generation = GEN4_HAL200), AGX3B, and GPU0 Ioto and are both
 * hardware-qualified for native CProg.  G15* uses the older G15 (RTKit 2419)
 * command-stream generation (its direct CDM launch has no TI spill field for
 * agx_chip_query_ti_spill(), and native preambles have not been hardware-
 * qualified on M3/J615), so G15* remains excluded and compiles every stage
 * with no_preamble.
 */
static inline bool
agx_chip_native_preamble(enum agx_chip chip)
{
   return agx_chip_info(chip)->quirks & AGX_QUIRK_NATIVE_PREAMBLE;
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

/* Exact M5 (T8142, G17G), M5 Pro (T6050, G17S), M5 Max (T6050, G17C) and M5
 * Ultra (T6050, G17D) kernel contracts the driver admits. This is not a
 * declaration of a complete or conformant device path. */
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

/* AGX2 (G13/G14 via simd_matrix_fmadd and software fallback for partial
 * subgroups; G15P is an uncharacterized A16 part), every AGX3B part
 * (8x8 matmad), and M5 (AGX3C matrix16/matmad8) support subgroup-scope
 * cooperative matrices. */
static inline bool
agx_chip_cooperative_matrix_supported(enum agx_chip chip)
{
   const enum agx_isa_encoding enc = agx_chip_isa_encoding(chip);
   return (enc == AGX_ISA_ENCODING_AGX2 && chip != AGX_CHIP_G15P) ||
          enc == AGX_ISA_ENCODING_AGX3B ||
          agx_chip_g17_m5(chip);
}

/* Hardware ray-query opcodes (intersect_ray, continue_ray, inc_stk) share one
 * instruction encoding per AGX3B / AGX3C ISA family; AGX3D (G18G) and AGX2
 * parts do not use these packers. */
static inline bool
agx_chip_has_agx3_ray_opcodes(enum agx_chip chip)
{
   return agx_isa_is_agx3(agx_chip_isa_encoding(chip));
}

/*
 * Whether Borealis may keep in-pass color input-attachment reads
 * (VK_KHR_dynamic_rendering_local_read and in-subpass self-dependencies) in
 * on-chip tile memory via the monolithic local-read pixel-class protocol.
 *
 * Selected by ISA encoding and command-stream generation rather than a chip
 * list: HAL200/AGX3B (M4 / M4 Pro / M4 Max / A18 Pro, including G16G, G16S,
 * G16C, and G17P) and HAL300/AGX3C (M5 family: G17G, G17S, G17C) share the
 * monolithic AGX3B/AGX3C pixel-class tilebuffer protocol. G15 (M3, unverified
 * command-stream generation) and AGX3D (G18G, no AGX3D local-read encoder)
 * remain on the DRAM fallback path.
 */
static inline bool
agx_chip_local_read_tilebuffer(enum agx_chip chip)
{
   const enum agx_isa_encoding enc = agx_chip_isa_encoding(chip);
   const enum agx_command_stream_generation cs =
      agx_chip_command_stream_generation(chip);

   return (enc == AGX_ISA_ENCODING_AGX3B && cs == AGX_CS_GENERATION_HAL200) ||
          (enc == AGX_ISA_ENCODING_AGX3C && cs == AGX_CS_GENERATION_HAL300);
}

/*
 * Whether the varying coefficient table selects perspective-correct
 * interpolation with shade model 5 instead of AGX2's 7.
 *
 * HAL300 parts (M5 G17G/G17S/G17C and M6 G18G) treat shade model 7 as linear,
 * so smooth perspective bindings must be rewritten to shade model 5 via
 * agx_cf_bindings_select_perspective().
 */
static inline bool
agx_chip_uses_shade_model_5(enum agx_chip chip)
{
   return agx_chip_hal300_path(chip);
}

/* Stable name for diagnostics, e.g. "G15G". Never NULL. */
static inline const char *
agx_chip_name(enum agx_chip chip)
{
   return agx_chip_info(chip)->name;
}

/* Stable name for the instruction set, e.g. "AGX3". Never NULL. */
const char *agx_isa_name(enum agx_isa isa);

const char *agx_command_stream_generation_name(
   enum agx_command_stream_generation generation);

#ifdef __cplusplus
}
#endif

#endif

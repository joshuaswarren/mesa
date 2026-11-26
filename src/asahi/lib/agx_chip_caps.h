/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_chip.h"

/*
 * Hardware capabilities, one bit per feature a driver may use.
 *
 * The chip table in agx_chip.h says what a chip is: its instruction set,
 * encodings, command-stream and descriptor generations, and command path.
 * This table says what it can do. A capability is a positive property of the
 * hardware, measured on the chips whose row carries it; a call site asks for
 * the capability it needs and never for a chip or a set of chips. Hardware
 * defects that need a workaround are not capabilities and stay in
 * enum agx_chip_quirk.
 *
 * Each capability is first introduced with exactly the chips of the
 * predicate it replaces, and tests/test-chip-caps.c checks the two agree on
 * every chip. Enabling a capability on another chip is then a change to that
 * chip's row, made together with the hardware evidence for it.
 */
enum agx_chip_cap {
   /* The 24-byte mode-2 indirect compute launch and its grid helper
    * (agx_chip_native_indirect_compute). */
   AGX_CAP_NATIVE_INDIRECT_COMPUTE = 1u << 0,

   /* The Apple9 imageblock: up to 128 bytes a sample in a 32 KiB tile
    * (agx_chip_native_imageblock). */
   AGX_CAP_NATIVE_IMAGEBLOCK = 1u << 1,

   /* Native shaders may carry an Ioto constant program and a shader preamble
    * (agx_chip_native_preamble). */
   AGX_CAP_NATIVE_PREAMBLE = 1u << 2,

   /* Consecutive application dispatches of one CDM stream may omit the
    * ordering barrier between them (agx_chip_elide_app_dispatch_barrier). */
   AGX_CAP_ELIDE_APP_DISPATCH_BARRIER = 1u << 3,

   /* An indirect indexed VDM Index List reads indices past its index-buffer
    * size word as zero (agx_chip_vdm_indirect_index_bounds). */
   AGX_CAP_VDM_INDIRECT_INDEX_BOUNDS = 1u << 4,

   /* The texture unit bounds integer texel fetches to the view and returns
    * the robustImageAccess2 value (agx_chip_hw_texel_fetch_bounds). */
   AGX_CAP_HW_TXF_BOUNDS = 1u << 5,

   /* Native fragment programs may merge triangles across memory writes and
    * wide subgroup operations (agx_chip_tri_merge_across_side_effects). */
   AGX_CAP_TRI_MERGE_ACROSS_SIDE_EFFECTS = 1u << 6,

   /* The matrix multiply-add backing a subgroup-scope cooperative matrix
    * (agx_chip_cooperative_matrix_supported). */
   AGX_CAP_COOPERATIVE_MATRIX = 1u << 7,

   /* 3D sparse image residency (sliced 64 KiB tiles) is qualified on this chip. */
   AGX_CAP_SPARSE_RESIDENCY_3D = 1u << 8,

   /* VK_EXT_descriptor_buffer is hardware-qualified on this chip. */
   AGX_CAP_DESCRIPTOR_BUFFER = 1u << 9,

   /* Borealis may coalesce small pure-global direct compute workgroups (<= 256
    * invocations, multiple of 32, proven independent of workgroup shape) into
    * larger power-of-two hardware workgroups (up to 256 threads, or 512 for
    * short low-register loop-free shaders) to avoid the AGX3 CDM per-workgroup
    * launch-rate floor on large grids. Measured and qualified on G17P
    * (HAL200) and M5 (G17G/G17S/G17C, HAL300); off on G15, G16, G18G, and
    * AGX2 until per-generation hardware A/B and CTS rows are collected. */
   AGX_CAP_COALESCE_PURE_GLOBAL_WORKGROUPS = 1u << 10,

   /* A compute or transfer barrier between work of one CDM stream that makes
    * no write available and no access visible (execution-only or
    * write-after-read) is met by the weak ordering record ending the
    * preceding launch, which holds the next launch until every earlier one
    * has finished, without the strong cache flush. Measured on G17P and M5
    * (G17G, G17S, G17C); G15 and G16G keep the flush until measured there. */
   AGX_CAP_CDM_WAR_WEAK_BARRIER = 1u << 11,

   /* Gen9 2D depth compression and compressed DRM_FORMAT_MOD_APPLE_INTERCHANGE
    * scanout (in both Gallium and Borealis) are enabled on this chip. Private
    * Gen9 colour compression is gated separately by
    * agx_chip_gen9_color_compression() (G16G and G17P). */
   AGX_CAP_GEN9_2D_COMPRESSION = 1u << 12,

   /* Register allocation tie-breaks co-read GPR pairs across the three 32-bit
    * register banks ((reg / 2) % 3) and coalesces single-use loop RMW
    * accumulator chains on non-preamble shaders. Measured and qualified on
    * G17P (HAL200) and M5 (G17G/G17S/G17C, HAL300); off on G15, G16, G18G,
    * and AGX2 until per-generation hardware A/B and CTS rows are collected. */
   AGX_CAP_MOD3_REG_BANK_RA = 1u << 13,

   /* Private Gen10 colour and depth compression. In Gallium: single-level,
    * single-layer 2D colour at 1x and 2x/4x, and Z32_FLOAT/Z16_UNORM depth at
    * 1x through the compressed Gen10 PBE, texture and ZLS forms. 2x/4x
    * Z32_FLOAT depth additionally needs the kernel to report
    * DRM_ASAHI_FEATURE_M5_DEPTH_MSAA_COMPRESSION. Measured on M5 (G17S,
    * G17C). In Borealis also single-sample colour 2D arrays, cube maps and
    * single-level 3D (1-8 byte texels), mipmapped one-byte 2D, and 16-byte
    * single-layer 2D, whose spilled attachments decompress in place
    * (libagx_agx3_decompress); measured on G17S. Off on G18G until measured
    * there. */
   AGX_CAP_GEN10_2D_COMPRESSION = 1u << 14,

   /* Sparse resource binding, aliased residency, 2D sparse images, sparse
    * buffers, and shaderResourceResidency are qualified on this chip.
    * Qualified on AGX2, G17P, and M5; off on G15, G16, and G18G (no SSTT
    * flush record or sparse texture ISA). */
   AGX_CAP_SPARSE_BINDING = 1u << 15,

   /* The render-submit ABI validates that packed ZLS depth_dimensions match
    * the render command's (width_px - 1) | ((height_px - 1) << 16) whenever
    * a depth or stencil attachment is bound, so render bounds must be at
    * least the ZLS surface dimensions even for sub-rectangle render areas. */
   AGX_CAP_ZLS_FULL_RENDER_BOUNDS = 1u << 16,

   /* The render ABI supports the native1130 mixed-format single-sample
    * fragment epilog encoding (skipping empty-mask targets and encoding
    * per-format partial channel masks natively). Verified on G18G. */
   AGX_CAP_NATIVE_MIXED_EPILOG = 1u << 17,

   /* HAL200/Gen9 lossless colour and depth/stencil compression in Borealis:
    * single-sample 1-8B colour (2D, arrays, cubemaps, single-mip 3D), 16B
    * colour (single-sample 2D, arrays, cubemaps), single-level 1x/2x/4x depth
    * and 1x stencil, shared DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED colour
    * images, in-place spill decompression via libagx_agx3_decompress, and Gen9
    * compressed compute mip blits. Enabled across native HAL200 AGX3B parts
    * (G16G and G17P), which share the Gen9 PBE/texture/ZLS/AIL metadata
    * contract; off on G15* (M3, G15/RTKit 2419 CS generation), M5 HAL300
    * (Gen10, gated separately), G18G, and AGX2. 2x/4x MSAA colour compression
    * is gated separately by AGX_CAP_HAL200_MSAA_COLOR_COMPRESSION. */
   AGX_CAP_HAL200_LOSSLESS_COMPRESSION = 1u << 18,

   /* HAL200/Gen9 lossless 2x/4x MSAA colour compression in Borealis: 1-8B
    * colour attachments (including arrays, input attachments, multiview, and
    * MRT spills decompressed in place by libagx_agx3_decompress). Enabled
    * across native HAL200 AGX3B parts (G16G and G17P), which share the Gen9
    * PBE/texture/AIL metadata contract and per-layer compressed resolve split.
    * Off on every chip without AGX_CAP_HAL200_LOSSLESS_COMPRESSION. */
   AGX_CAP_HAL200_MSAA_COLOR_COMPRESSION = 1u << 19,

   /* A signed-zero preserving fp32 fmin/fmax may be lowered to the hardware
    * min/max, with an integer min/max of the bits when the operands compare
    * equal (agx_nir_lower_fminmax_signed_zero). That needs the fp32 compares
    * to order denormals exactly whenever the shader preserves them. Measured
    * on G17P and on M5 (G17S); the M5 parts share one instruction set. */
   AGX_CAP_SIGNED_ZERO_MINMAX_SELECT = 1u << 20,

   /* Compute streams split to avoid retroactively stalling independent compute
    * work behind a render barrier or delaying graphics consumers behind
    * independent post-barrier compute, multi-chunk submissions take per-chunk
    * timeline points, and a HAL200 render whose tiling half names fewer
    * earlier renders than its fragment half falls back to a staged render.
    * Qualified on G17P and M5 (G17G, G17S, G17C); G15 and G16G (~0.3 ms
    * per-command turnaround) remain excluded until measured. */
   AGX_CAP_CDM_OVERLAP_SPLIT = 1u << 21,

   /* Command-buffer, root-descriptor, VDM-slab, Ioto-arena, and descriptor-pool
    * BOs are mapped CPU write-back (AGX_BO_WRITEBACK) when util_has_cache_ops()
    * is available, cleaned before GPU submission, enabling CPU-cached root
    * reuse, push-descriptor elision, and direct fragment-root comparisons
    * without shadow snapshots. Qualified on G17P and M5 (G17G, G17S, G17C). */
   AGX_CAP_CACHED_CMD_MEMORY = 1u << 22,
};

/* What the admitted AGX2 chips (G13G, G13X, G14G, G14X) share. */
#define AGX_CAPS_AGX2                                                          \
   (AGX_CAP_SPARSE_BINDING | AGX_CAP_SPARSE_RESIDENCY_3D |                     \
    AGX_CAP_DESCRIPTOR_BUFFER | AGX_CAP_COOPERATIVE_MATRIX)

/* What the five native AGX3 chips (G15G, G15S, G15C, G16G, G17P) share. */
#define AGX_CAPS_NATIVE                                                        \
   (AGX_CAP_NATIVE_INDIRECT_COMPUTE | AGX_CAP_NATIVE_IMAGEBLOCK |              \
    AGX_CAP_ELIDE_APP_DISPATCH_BARRIER | AGX_CAP_COOPERATIVE_MATRIX |          \
    AGX_CAP_DESCRIPTOR_BUFFER | AGX_CAP_MOD3_REG_BANK_RA |                     \
    AGX_CAP_SIGNED_ZERO_MINMAX_SELECT)

#define AGX_CAPS_G17P                                                          \
   (AGX_CAPS_NATIVE | AGX_CAP_NATIVE_PREAMBLE |                                \
    AGX_CAP_VDM_INDIRECT_INDEX_BOUNDS |                                        \
    AGX_CAP_TRI_MERGE_ACROSS_SIDE_EFFECTS | AGX_CAP_SPARSE_BINDING |           \
    AGX_CAP_SPARSE_RESIDENCY_3D | AGX_CAP_COALESCE_PURE_GLOBAL_WORKGROUPS |    \
    AGX_CAP_CDM_WAR_WEAK_BARRIER | AGX_CAP_GEN9_2D_COMPRESSION |               \
    AGX_CAP_HAL200_LOSSLESS_COMPRESSION |                                      \
    AGX_CAP_HAL200_MSAA_COLOR_COMPRESSION | AGX_CAP_CDM_OVERLAP_SPLIT |        \
    AGX_CAP_CACHED_CMD_MEMORY)

/* What the M5 parts (G17G, G17S, G17C) share. */
#define AGX_CAPS_M5                                                            \
   (AGX_CAP_ELIDE_APP_DISPATCH_BARRIER | AGX_CAP_VDM_INDIRECT_INDEX_BOUNDS |   \
    AGX_CAP_HW_TXF_BOUNDS | AGX_CAP_COOPERATIVE_MATRIX |                       \
    AGX_CAP_SPARSE_BINDING | AGX_CAP_DESCRIPTOR_BUFFER |                       \
    AGX_CAP_COALESCE_PURE_GLOBAL_WORKGROUPS | AGX_CAP_CDM_WAR_WEAK_BARRIER |   \
    AGX_CAP_GEN10_2D_COMPRESSION | AGX_CAP_MOD3_REG_BANK_RA |                  \
    AGX_CAP_SIGNED_ZERO_MINMAX_SELECT | AGX_CAP_CDM_OVERLAP_SPLIT |            \
    AGX_CAP_CACHED_CMD_MEMORY)

/*
 * One row per chip of AGX_CHIPS, in the same order: chip, capabilities.
 *
 * G15D, G16P, G16S, G16C and G17A are recognised but not admitted; they keep
 * the capabilities their encoding implies so the table answers as the
 * predicates do.
 */
#define AGX_CHIP_CAPS(X, ...)                                                  \
   X(G13G, AGX_CAPS_AGX2, __VA_ARGS__)                                         \
   X(G13X, AGX_CAPS_AGX2, __VA_ARGS__)                                         \
   X(G14G, AGX_CAPS_AGX2, __VA_ARGS__)                                         \
   X(G14X, AGX_CAPS_AGX2, __VA_ARGS__)                                         \
   X(G15P, 0, __VA_ARGS__)                                                     \
   X(G15G, AGX_CAPS_NATIVE, __VA_ARGS__)                                       \
   X(G15S, AGX_CAPS_NATIVE, __VA_ARGS__)                                       \
   X(G15C, AGX_CAPS_NATIVE, __VA_ARGS__)                                       \
   X(G15D, AGX_CAP_COOPERATIVE_MATRIX, __VA_ARGS__)                            \
   X(G16P, AGX_CAP_COOPERATIVE_MATRIX, __VA_ARGS__)                            \
   X(G16G,                                                                     \
     AGX_CAPS_NATIVE | AGX_CAP_NATIVE_PREAMBLE |                               \
        AGX_CAP_HAL200_LOSSLESS_COMPRESSION |                                  \
        AGX_CAP_HAL200_MSAA_COLOR_COMPRESSION,                                 \
     __VA_ARGS__)                                                              \
   X(G16S, AGX_CAP_COOPERATIVE_MATRIX, __VA_ARGS__)                            \
   X(G16C, AGX_CAP_COOPERATIVE_MATRIX, __VA_ARGS__)                            \
   X(G17P, AGX_CAPS_G17P, __VA_ARGS__)                                         \
   X(G17A, AGX_CAP_COOPERATIVE_MATRIX, __VA_ARGS__)                            \
   X(G17G, AGX_CAPS_M5, __VA_ARGS__)                                           \
   X(G17S, AGX_CAPS_M5, __VA_ARGS__)                                           \
   X(G17C, AGX_CAPS_M5, __VA_ARGS__)                                           \
   X(G18G, AGX_CAP_ZLS_FULL_RENDER_BOUNDS | AGX_CAP_NATIVE_MIXED_EPILOG,       \
     __VA_ARGS__)

/* Like agx_chip_has_record(), a test of the table expands to comparisons, so
 * the precompiled OpenCL kernels can call it and the C units run the same
 * expression. */
#define AGX_CHIP_CAP_ROW_TEST(row, caps, chip, cap)                            \
   | (((chip) == AGX_CHIP_##row) & (((caps) & (cap)) != 0))

/* Whether a chip has a capability (one enum agx_chip_cap bit). Unknown chips
 * and values outside the enum have none. */
static inline bool
agx_chip_has_cap(enum agx_chip chip, unsigned cap)
{
   return false AGX_CHIP_CAPS(AGX_CHIP_CAP_ROW_TEST, chip, cap);
}

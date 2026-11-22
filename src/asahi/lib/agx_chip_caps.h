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
    * (HAL200); off on M5 (G17G/G17S/G17C/G17D), G15, G16, G18G, and AGX2 until
    * per-generation hardware A/B and CTS rows are collected. */
   AGX_CAP_COALESCE_PURE_GLOBAL_WORKGROUPS = 1u << 10,

   /* A compute or transfer barrier between work of one CDM stream that makes
    * no write available and no access visible (execution-only or
    * write-after-read) is met by the weak ordering record ending the
    * preceding launch, which holds the next launch until every earlier one
    * has finished, without the strong cache flush. Measured on G17P and M5
    * (G17G, G17S, G17C); G15 and G16G keep the flush until measured there. */
   AGX_CAP_CDM_WAR_WEAK_BARRIER = 1u << 11,
};

/* What the admitted AGX2 chips (G13G, G13X, G14G, G14X) share. */
#define AGX_CAPS_AGX2                                                          \
   (AGX_CAP_SPARSE_RESIDENCY_3D | AGX_CAP_DESCRIPTOR_BUFFER |                  \
    AGX_CAP_COOPERATIVE_MATRIX)

/* What the five native AGX3 chips (G15G, G15S, G15C, G16G, G17P) share. */
#define AGX_CAPS_NATIVE                                                        \
   (AGX_CAP_NATIVE_INDIRECT_COMPUTE | AGX_CAP_NATIVE_IMAGEBLOCK |              \
    AGX_CAP_ELIDE_APP_DISPATCH_BARRIER | AGX_CAP_COOPERATIVE_MATRIX |          \
    AGX_CAP_DESCRIPTOR_BUFFER)

#define AGX_CAPS_G17P                                                          \
   (AGX_CAPS_NATIVE | AGX_CAP_NATIVE_PREAMBLE |                                \
    AGX_CAP_VDM_INDIRECT_INDEX_BOUNDS |                                        \
    AGX_CAP_TRI_MERGE_ACROSS_SIDE_EFFECTS | AGX_CAP_SPARSE_RESIDENCY_3D |      \
    AGX_CAP_COALESCE_PURE_GLOBAL_WORKGROUPS | AGX_CAP_CDM_WAR_WEAK_BARRIER)

/* What the single-die M5 parts (G17G, G17S, G17C) share. G17D keeps
 * VDM_INDIRECT_INDEX_BOUNDS and HW_TXF_BOUNDS off until measured on J775D. */
#define AGX_CAPS_M5                                                            \
   (AGX_CAP_ELIDE_APP_DISPATCH_BARRIER | AGX_CAP_VDM_INDIRECT_INDEX_BOUNDS |   \
    AGX_CAP_HW_TXF_BOUNDS | AGX_CAP_COOPERATIVE_MATRIX |                       \
    AGX_CAP_CDM_WAR_WEAK_BARRIER)

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
   X(G16G, AGX_CAPS_NATIVE | AGX_CAP_NATIVE_PREAMBLE, __VA_ARGS__)             \
   X(G16S, AGX_CAP_COOPERATIVE_MATRIX, __VA_ARGS__)                            \
   X(G16C, AGX_CAP_COOPERATIVE_MATRIX, __VA_ARGS__)                            \
   X(G17P, AGX_CAPS_G17P, __VA_ARGS__)                                         \
   X(G17A, AGX_CAP_COOPERATIVE_MATRIX, __VA_ARGS__)                            \
   X(G17G, AGX_CAPS_M5, __VA_ARGS__)                                           \
   X(G17S, AGX_CAPS_M5, __VA_ARGS__)                                           \
   X(G17C, AGX_CAPS_M5, __VA_ARGS__)                                           \
   X(G17D, AGX_CAP_ELIDE_APP_DISPATCH_BARRIER | AGX_CAP_COOPERATIVE_MATRIX,    \
     __VA_ARGS__)                                                              \
   X(G18G, 0, __VA_ARGS__)

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

/*
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

/* See compiler/README.md for the ABI */

#define AGX_ABI_VIN_ATTRIB(i)           (2 * (8 + i))
#define AGX_ABI_VIN_VERTEX_ID_ZERO_BASE (2 * 4)
#define AGX_ABI_VIN_VERTEX_ID           (2 * 5)
#define AGX_ABI_VIN_INSTANCE_ID         (2 * 6)

#define AGX_ABI_FIN_SAMPLE_MASK (2)

#define AGX_ABI_FOUT_SAMPLE_MASK   (2)
#define AGX_ABI_FOUT_Z             (4)
#define AGX_ABI_FOUT_S             (6)
#define AGX_ABI_FOUT_WRITE_SAMPLES (7)
#define AGX_ABI_FOUT_COLOUR(rt)    (2 * (4 + (4 * rt)))

#define AGX_ABI_VUNI_VBO_BASE(vbo) (4 * (vbo))
#define AGX_ABI_VUNI_VBO_CLAMP(nr_vbos, vbo)                                   \
   (AGX_ABI_VUNI_VBO_BASE(nr_vbos) + 2 * (vbo))

#define AGX_ABI_VUNI_FIRST_VERTEX(nr_vbos)                                     \
   AGX_ABI_VUNI_VBO_CLAMP(nr_vbos, nr_vbos)

#define AGX_ABI_VUNI_BASE_INSTANCE(nr_vbos)                                    \
   (AGX_ABI_VUNI_FIRST_VERTEX(nr_vbos) + 2)

#define AGX_ABI_VUNI_DRAW_ID(nr_vbos) (AGX_ABI_VUNI_FIRST_VERTEX(nr_vbos) + 4)

#define AGX_ABI_VUNI_VERTEX_PARAMS(nr_vbos)                                    \
   ALIGN_POT(AGX_ABI_VUNI_FIRST_VERTEX(nr_vbos) + 8, 4)

#define AGX_ABI_VUNI_COUNT_GL(nr_vbos, sw)                                     \
   (sw ? (AGX_ABI_VUNI_VERTEX_PARAMS(nr_vbos) + 4)                             \
       : (AGX_ABI_VUNI_BASE_INSTANCE(nr_vbos) + 2))

#define AGX_ABI_VUNI_COUNT_VK(nr_vbos)                                         \
   ALIGN_POT(AGX_ABI_VUNI_VERTEX_PARAMS(nr_vbos) + 4, 4)

/* HAL200 graphics texture instructions take a pair of 64-bit texture and
 * sampler table pointers from the first four uniform dwords. Vertex shaders
 * using texture state place the ordinary vertex ABI after this prefix. */
#define AGX_ABI_G17P_TEXTURE_POINTER_HALFS 8

/* G18 threadgroup accesses take their base from uniform half 8, which the
 * hardware preloads. Ordinary pushed pointers or preamble exports must not
 * overwrite it: stores then land nowhere and loads read zero. Keep API
 * compute uniforms above this slot, aligned for 64-bit pointers. The native
 * texture/storage prefix remains in halves 0..7. */
#define AGX_ABI_G18_SHARED_BASE_HALF 8u
#define AGX_ABI_G18_SHARED_UNIFORM_HALFS 12u

/* AGX3 threadgroup accesses take their base from uniform half 8, which
 * agx_emit_local_store/load hard-code and which the hardware preloads -- no
 * driver path ever writes it.  A shader that places its own uniforms over
 * that half therefore destroys its shared-memory base, silently: the stores
 * land nowhere and the loads read zero.
 *
 * API shaders are unaffected because their pushes are small, but precompiled
 * kernel arguments start at half 0 and run past it (libagx_unroll_restart's
 * are 26 halves), which is why a geometry shader plus primitive restart drew
 * nothing.  Precompiled AGX3 kernels therefore place their arguments above
 * both the texture-pointer prefix and this half.  Four-half aligned so 64-bit
 * arguments keep their uniform-pair alignment. */
#define AGX_ABI_AGX3_PRECOMP_ARG_HALFS 12

#define AGX_ABI_FUNI_EMRT_HEAP (0)
#define AGX_ABI_FUNI_BLEND_R   (AGX_ABI_FUNI_EMRT_HEAP + 4)
#define AGX_ABI_FUNI_BLEND_G   (AGX_ABI_FUNI_BLEND_R + 2)
#define AGX_ABI_FUNI_BLEND_B   (AGX_ABI_FUNI_BLEND_R + 4)
#define AGX_ABI_FUNI_BLEND_A   (AGX_ABI_FUNI_BLEND_R + 6)
#define AGX_ABI_FUNI_ROOT      (12)
#define AGX_ABI_FUNI_COUNT     (16)

/* This address is in our reservation, and can be
 * addressed with only small integers in the low/high. That lets us do some
 * robustness optimization even without soft fault. It spans
 * maxUniformBufferRange, so a null uniform buffer (based here) reads zero at
 * every offset a shader can legally use without per-load bounds checks.
 */
#define AGX_ZERO_PAGE_ADDRESS (((uint64_t)1) << 32)
#define AGX_ZERO_PAGE_SIZE    (65536)

#define AGX_SCRATCH_PAGE_ADDRESS (AGX_ZERO_PAGE_ADDRESS + AGX_ZERO_PAGE_SIZE)

/* Honeykrisp G17P texel-buffer metadata follows the native texture/PBE and
 * heap-index records. These are software ABI fields, not Gen9 hardware bits.
 */
#define AGX_G17P_BUFFER_TEXTURE_PBE_OFFSET 32
/* Gallium's Gen10 (G17G) buffer texture keeps its element count in the
 * descriptor's last word, which no Gen10 field occupies. */
#define AGX_GEN10_BUFFER_TEXTURE_SIZE_OFFSET 28
/* Honeykrisp's Gen10 image views keep their level-zero extent (width in the
 * low half, height in the high half) in the same word; see
 * agx_chip_texture_descriptor_spare_word(). */
#define AGX_GEN10_IMAGE_EXTENT_OFFSET 28
/* Gallium's Gen10 storage PBE keeps the image-atomic address word (the
 * G17P sideband, AGX_G17P_IMAGE_ATOMIC_SW_FROM_PBE) in its last eight bytes,
 * which no Gen10 field occupies. */
#define AGX_GEN10_IMAGE_ATOMIC_SW_OFFSET 24
/* The view's element count; the texel offset of the view's first element in
 * its native texture and PBE, which only texture and PBE coordinates add; and
 * the exact address of the view's first element, from which everything that
 * addresses the view as memory (atomics, residency, RGB32 reads) counts the
 * coordinate alone.
 */
#define AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET 72
#define AGX_G17P_BUFFER_TEXTURE_SIZE_OFFSET 72
#define AGX_G17P_BUFFER_TEXTURE_TAIL_OFFSET 76
#define AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET 80
#define AGX_G17P_BUFFER_TEXTURE_RGB32_OFFSET 88
/* Sparse residency for a texel buffer.  G17P's buffer texture descriptor is
 * never put in AGX_IMAGE_MODE_SPARSE (hk_buffer_view.c builds a plain linear
 * 2D texture), so the hardware has no residency to report for one.  A sparse
 * texel fetch therefore reads its residency from the same per-page bitmap the
 * sparse SSBO path uses, and this word carries the two things the shader needs
 * that the rest of the record does not already hold:
 *
 *   bits 0..3    log2 of the view's element size in bytes
 *   bits 4..31   the residency map's address >> AIL_PAGESIZE_LOG2
 *
 * Zero means "no sparse map" -- an ordinary buffer, or a format whose element
 * size is not a power of two -- and the fetch reports resident.
 */
#define AGX_G17P_BUFFER_TEXTURE_SPARSE_OFFSET 92
#define AGX_G17P_BUFFER_TEXTURE_SPARSE_ELSIZE_MASK 0xfu
#define AGX_G17P_BUFFER_TEXTURE_SPARSE_MAP_SHIFT 4
#define AGX_G17P_BUFFER_TEXTURE_DESCRIPTOR_SIZE 96

/* Layout of struct hk_sparse_buffer_map_header (hk_buffer.h), which both the
 * sparse SSBO lowering and the sparse texel fetch read from the shader.
 * Static-asserted against the struct there. */
#define AGX_SPARSE_BUFFER_MAP_RESOURCE_BASE_OFFSET 0
#define AGX_SPARSE_BUFFER_MAP_PAGE_COUNT_OFFSET 8
#define AGX_SPARSE_BUFFER_MAP_BITMAP_OFFSET 16

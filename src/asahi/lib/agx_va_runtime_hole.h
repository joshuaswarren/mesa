/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "agx_chip.h"
#include "util/vma.h"

#include <stdbool.h>
#include <stdint.h>

/* T8140's per-VM kernel runtime-graph range, expressed as [start, end). */
#define AGX_G17_RUNTIME_GRAPH_VA_START UINT64_C(0x7000000000)
#define AGX_G17_RUNTIME_GRAPH_VA_END   UINT64_C(0x7100000000)

/* The T8122 (M3, G15G) kernel runtime reserves, in every VM, the top half of
 * the 2^40 USC window, 0x7000000000 up to 0x8000000000, its own USC base at
 * 0x1000000000-0x1008000000 and 0x1074000000-0x1074028000. A bind that touches
 * one fails with EINVAL. These mirror Runtime::new_vm in
 * drivers/gpu/drm/asahi/m3/m3_runtime.rs as of omacom/linux-aurora 4888ad0a0f
 * (merged in be6381f9d2) by hand, until the kernel reports them. */
#define AGX_T8122_RUNTIME_WINDOW_HIGH    UINT64_C(0x10080000000)
#define AGX_T8122_RUNTIME_WINDOW_END     UINT64_C(0x10100000000)
#define AGX_T8122_RUNTIME_RESERVED_START UINT64_C(0x7000000000)
#define AGX_T8122_RUNTIME_RESERVED_END   UINT64_C(0x8000000000)

/* T8140's channel-owned AXFB aliases are installed by the kernel in
 * every render-capable client root.  This range falls inside G17P's 4-GiB USC
 * heap, so userspace must remove it from the allocator before creating any
 * shader BOs. */
#define AGX_G17P_RENDER_AUX_ALIAS_START UINT64_C(0x10002000000)
#define AGX_G17P_RENDER_AUX_ALIAS_END   UINT64_C(0x10002400000)

/* G17P's compact VDM pointer is a 39-bit offset from 0x1000000000.
 * Keep the ordinary allocation and sparse-shadow windows below this exclusive
 * limit so every VDM control stream and payload is directly representable. */
#define AGX_G17P_VDM_USER_BASE  UINT64_C(0x1000000000)
#define AGX_G17P_VDM_USER_SIZE  (UINT64_C(1) << 39)
#define AGX_G17P_VDM_USER_LIMIT                                            \
   (AGX_G17P_VDM_USER_BASE + AGX_G17P_VDM_USER_SIZE)

/*
 * The legacy USC resource records Gallium still binds for graphics carry the
 * texture and sampler table addresses as ABSOLUTE, 8-byte-scaled 36-bit
 * fields -- cmdbuf.xml "USC Texture" and "USC Sampler", both
 *
 *   <field name="Buffer" size="36" start="0:27" type="address"
 *          modifier="shr(3)"/>
 *
 * so 36 stored bits times an 8-byte scale is 39 significant address bits,
 * with no USC exec base subtracted. The generated packer enforces exactly
 * that with agx_genxml_validate_bounds(), which is fatal, so an allocation at
 * or above this limit aborts the process before any draw is submitted.
 *
 * This is narrower than the compact VDM pointer above, so on G17P it -- not
 * the VDM limit -- is what bounds the user window.
 */
#define AGX_G17P_USC_RESOURCE_FIELD_BITS  36
#define AGX_G17P_USC_RESOURCE_SCALE_SHIFT 3
#define AGX_G17P_USC_RESOURCE_LIMIT                                        \
   (UINT64_C(1) << (AGX_G17P_USC_RESOURCE_FIELD_BITS +                     \
                    AGX_G17P_USC_RESOURCE_SCALE_SHIFT))

/* The narrower of the two pointers every G17P user allocation must satisfy. */
#define AGX_G17P_USER_LIMIT                                                \
   (AGX_G17P_VDM_USER_LIMIT < AGX_G17P_USC_RESOURCE_LIMIT                  \
       ? AGX_G17P_VDM_USER_LIMIT                                           \
       : AGX_G17P_USC_RESOURCE_LIMIT)

/* Native TVB blocks are sparse 128-KiB mappings with 32-KiB page IDs
 * 0x11..0xc7. Their last range ends at compact 0x658000; the fixed scene,
 * discard and status mappings also live below this bound. Start command BOs
 * at +8 MiB, above the cold retained render pool. Additional kernel arenas are reserved
 * separately below. Bind-tail and draw-state
 * records carry unsigned 32-bit context offsets; PPP/stream pointers are wider.
 * Use that narrowest encoding's entire range, rather than limiting live command
 * BOs to the original bring-up window's 24 MiB. This reserves virtual addresses,
 * not backing memory: BO storage and page tables are allocated on demand. */
#define AGX_G17P_VDM_ALIAS_OFFSET_START UINT64_C(0x00800000)
#define AGX_G17P_VDM_ALIAS_OFFSET_END   (UINT64_C(1) << 32)
#define AGX_G17P_VDM_ALIAS_BASE                                         \
   (AGX_G17P_VDM_USER_BASE + AGX_G17P_VDM_ALIAS_OFFSET_START)
#define AGX_G17P_VDM_ALIAS_SIZE                                         \
   (AGX_G17P_VDM_ALIAS_OFFSET_END - AGX_G17P_VDM_ALIAS_OFFSET_START)

/* The kernel also owns compact render-state aliases, both growing TVB
 * arenas, and dynamic render-target backing above +32 MiB (currently ending
 * at +0x64000000). Keep the rest of the lower half reserved so cached command
 * aliases cannot run into these arenas as their live footprint grows. */
#define AGX_G17P_VDM_ALIAS_RESERVED_START \
   (AGX_G17P_VDM_USER_BASE + UINT64_C(0x02000000))
#define AGX_G17P_VDM_ALIAS_RESERVED_END \
   (AGX_G17P_VDM_USER_BASE + UINT64_C(0x80000000))
#define AGX_G17P_VDM_ALIAS_AVAILABLE_SIZE \
   (AGX_G17P_VDM_ALIAS_SIZE - \
    (AGX_G17P_VDM_ALIAS_RESERVED_END - AGX_G17P_VDM_ALIAS_RESERVED_START))

/* The exclusive top of the main heap and its sparse alias. Native AGX3 chips
 * keep the heap below the USC window, bounded by the absolute 36-bit-shr(3)
 * USC Texture/Sampler field (2^39) and, on G17P, the compact VDM aperture. On
 * T8122 (G15G) it also stops below the runtime's 0x7000000000 reservation,
 * which leaves the heap [0x4000000000, 0x5000000000) and its alias
 * [0x5000000000, 0x6000000000). Other chips keep the kernel's limit. */
static inline uint64_t
agx_va_user_limit(enum agx_chip chip, bool heap_below_usc, uint64_t kernel_start)
{
   if (!heap_below_usc)
      return kernel_start;
   return chip == AGX_CHIP_G15G ? AGX_T8122_RUNTIME_RESERVED_START
                                : AGX_G17P_USER_LIMIT;
}

/*
 * Reduce the kernel-reported user VA window to the power-of-two window the
 * driver actually allocates from, and derive the address bit that tells a
 * sparse buffer's read-only alias apart from its primary mapping.
 *
 * A sparse buffer is bound a second time at `addr + sparse_ro_offset`
 * (hk_buffer.c:95, d12_resource.c:1118).  For that to be a pure bit-set with
 * no carry -- and for the alias window to stay disjoint from the primary one
 * -- every address in [*out_start, *out_start + *out_size) must have the
 * sparse bit clear.  Since both are powers of two once rounded, that reduces
 * to *out_start > *out_size.  On success `*out_size` is both the window size
 * and the sparse read-only offset.
 *
 * Returns false if no usable split exists.
 */
bool agx_va_split_user_window(uint64_t user_start, uint64_t user_size,
                              uint64_t *out_start, uint64_t *out_size);

bool agx_va_overlaps_runtime_graph_hole(enum agx_chip chip, uint64_t addr,
                                        uint64_t size);

bool agx_g17p_vdm_alias_heap_init(struct util_vma_heap *heap);
bool agx_capture_replay_heap_init(struct util_vma_heap *main_heap,
                                  struct util_vma_heap *replay_heap,
                                  uint64_t user_size, uint64_t *out_base,
                                  uint64_t *out_size);
bool agx_g17p_vdm_alias_range_valid(uint64_t addr, uint64_t size);
bool agx_g17p_vdm_alias_compact(uint64_t addr, uint32_t *offset);

/* agx_g17p_vdm_alias_compact() for one byte at addr, for per-draw callers. */
static inline bool
agx_g17p_vdm_alias_compact_inline(uint64_t addr, uint32_t *offset)
{
   if (addr < AGX_G17P_VDM_ALIAS_BASE || addr == UINT64_MAX)
      return false;

   const uint64_t end = addr + 1;
   if (!(end <= AGX_G17P_VDM_USER_BASE + AGX_G17P_VDM_ALIAS_OFFSET_END &&
         (end <= AGX_G17P_VDM_ALIAS_RESERVED_START ||
          addr >= AGX_G17P_VDM_ALIAS_RESERVED_END)))
      return false;

   const uint64_t compact = addr - AGX_G17P_VDM_USER_BASE;
   if (compact > UINT32_MAX)
      return false;

   *offset = (uint32_t)compact;
   return true;
}

/* G17P HAL200 VDM payload fields are offsets from the render-context base. */
bool agx_g17p_vdm_payload_compact(uint64_t addr, uint64_t *compact);
static inline bool
agx_g17p_vdm_payload_expand(uint64_t compact, uint64_t size, uint64_t *addr)
{
   if (!addr || size == 0 || compact >= AGX_G17P_VDM_USER_SIZE ||
       size > AGX_G17P_VDM_USER_SIZE - compact)
      return false;

   *addr = AGX_G17P_VDM_USER_BASE + compact;
   return true;
}

bool agx_va_reserve_runtime_graph_hole(
   enum agx_chip chip, struct util_vma_heap *main_heap, uint64_t main_start,
   uint64_t main_size, uint64_t sparse_ro_offset,
   struct util_vma_heap *usc_heap, uint64_t usc_start, uint64_t usc_size);

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_va_runtime_hole.h"

#include "util/macros.h"
#include "util/u_math.h"

struct agx_va_interval {
   uint64_t start;
   uint64_t end;
};

static const struct agx_va_interval g17p_kernel_reserved[] = {
   {AGX_G17_RUNTIME_GRAPH_VA_START, AGX_G17_RUNTIME_GRAPH_VA_END},
   {AGX_G17P_RENDER_AUX_ALIAS_START, AGX_G17P_RENDER_AUX_ALIAS_END},
};

bool
agx_g17p_vdm_alias_heap_init(struct util_vma_heap *heap)
{
   if (!heap)
      return false;

   util_vma_heap_init(heap, AGX_G17P_VDM_ALIAS_BASE,
                      AGX_G17P_VDM_ALIAS_SIZE);
   heap->alloc_high = false;
   if (!util_vma_heap_alloc_addr(heap, AGX_G17P_VDM_ALIAS_RESERVED_START,
                                 AGX_G17P_VDM_ALIAS_RESERVED_END -
                                    AGX_G17P_VDM_ALIAS_RESERVED_START)) {
      util_vma_heap_finish(heap);
      return false;
   }
   if (heap->free_size != AGX_G17P_VDM_ALIAS_AVAILABLE_SIZE) {
      util_vma_heap_finish(heap);
      return false;
   }
   return true;
}

bool
agx_capture_replay_heap_init(struct util_vma_heap *main_heap,
                             struct util_vma_heap *replay_heap,
                             uint64_t user_size, uint64_t *out_base,
                             uint64_t *out_size)
{
   const uint64_t alignment = UINT64_C(1) << 30;
   const uint64_t size = MIN2(user_size / 8, UINT64_C(64) << 30) &
                         ~(alignment - 1);
   if (!main_heap || !replay_heap || size == 0)
      return false;

   const uint64_t base = util_vma_heap_alloc(main_heap, size, alignment);
   if (!base)
      return false;
   util_vma_heap_init(replay_heap, base, size);
   *out_base = base;
   *out_size = size;
   return true;
}

bool
agx_g17p_vdm_alias_range_valid(uint64_t addr, uint64_t size)
{
   if (size == 0 || addr < AGX_G17P_VDM_ALIAS_BASE ||
       size > UINT64_MAX - addr)
      return false;

   const uint64_t end = addr + size;
   return end <= AGX_G17P_VDM_USER_BASE + AGX_G17P_VDM_ALIAS_OFFSET_END &&
          (end <= AGX_G17P_VDM_ALIAS_RESERVED_START ||
           addr >= AGX_G17P_VDM_ALIAS_RESERVED_END);
}

bool
agx_g17p_vdm_alias_compact(uint64_t addr, uint32_t *offset)
{
   if (!offset || !agx_g17p_vdm_alias_range_valid(addr, 1))
      return false;

   const uint64_t compact = addr - AGX_G17P_VDM_USER_BASE;
   if (compact > UINT32_MAX)
      return false;

   *offset = (uint32_t)compact;
   return true;
}

bool
agx_g17p_vdm_payload_compact(uint64_t addr, uint64_t *compact)
{
   if (!compact || addr < AGX_G17P_VDM_USER_BASE ||
       addr >= AGX_G17P_VDM_USER_LIMIT || (addr & 3))
      return false;

   *compact = addr - AGX_G17P_VDM_USER_BASE;
   return true;
}

bool
agx_va_split_user_window(uint64_t user_start, uint64_t user_size,
                         uint64_t *out_start, uint64_t *out_size)
{
   if (!out_start || !out_size || user_start == 0 || user_size == 0)
      return false;

   /* Round the user VA window to powers-of-two... */
   user_start = util_next_power_of_two64(user_start);
   user_size = util_next_power_of_two64(user_size + 1) >> 1;

   /* ...so when we cut user size in half to emulate sparse buffers... */
   user_size /= 2;

   /*
    * ...and again as many times as it takes for the whole window to sit below
    * the bit we are about to steal.  Upstream halves at most once more, which
    * is exactly enough for AGX2: a 39-bit user top leaves a window equal to
    * the rounded start, so one extra halving lands it below.  AGX3 needs the
    * loop -- T8140's 42-bit root leaves a window eight times the start, so a
    * single fixup would still straddle the sparse bit and the read-only alias
    * of a high allocation would carry out of its own half.
    */
   while (user_size >= user_start)
      user_size /= 2;

   if (user_size == 0)
      return false;

   /* ...we can distinguish the top/bottom half by an address bit */
   if ((user_start & user_size) != 0 ||
       ((user_start + (user_size - 1)) & user_size) != 0)
      return false;

   *out_start = user_start;
   *out_size = user_size;
   return true;
}

bool
agx_va_overlaps_runtime_graph_hole(enum agx_chip chip, uint64_t addr,
                                   uint64_t size)
{
   if (!agx_chip_native_agx3(chip) || size == 0 ||
       addr >= AGX_G17_RUNTIME_GRAPH_VA_END)
      return false;

   return addr >= AGX_G17_RUNTIME_GRAPH_VA_START ||
          size > AGX_G17_RUNTIME_GRAPH_VA_START - addr;
}

static bool
range_end(uint64_t start, uint64_t size, uint64_t *end)
{
   if (size > UINT64_MAX - start)
      return false;

   *end = start + size;
   return true;
}

static void
add_intersection(struct agx_va_interval *intervals, unsigned *count,
                 uint64_t heap_start, uint64_t heap_end,
                 uint64_t reserved_start, uint64_t reserved_end)
{
   const uint64_t start = MAX2(heap_start, reserved_start);
   const uint64_t end = MIN2(heap_end, reserved_end);

   if (start < end)
      intervals[(*count)++] = (struct agx_va_interval){start, end};
}

static bool
reserve_intervals(struct util_vma_heap *heap,
                  struct agx_va_interval *intervals, unsigned count)
{
   /* The main heap may contain both a fixed kernel range and the preimage of
    * another fixed range under sparse_ro_offset.  Keep this general so adding
    * a kernel-owned range cannot make reservations order-dependent. */
   for (unsigned i = 1; i < count; ++i) {
      const struct agx_va_interval current = intervals[i];
      unsigned j = i;

      while (j > 0 && current.start < intervals[j - 1].start) {
         intervals[j] = intervals[j - 1];
         --j;
      }
      intervals[j] = current;
   }

   unsigned merged = 0;
   for (unsigned i = 0; i < count; ++i) {
      if (merged > 0 && intervals[i].start <= intervals[merged - 1].end) {
         intervals[merged - 1].end =
            MAX2(intervals[merged - 1].end, intervals[i].end);
      } else {
         intervals[merged++] = intervals[i];
      }
   }

   for (unsigned i = 0; i < merged; ++i) {
      if (!util_vma_heap_alloc_addr(heap, intervals[i].start,
                                    intervals[i].end - intervals[i].start))
         return false;
   }

   return true;
}

bool
agx_va_reserve_runtime_graph_hole(
   enum agx_chip chip, struct util_vma_heap *main_heap, uint64_t main_start,
   uint64_t main_size, uint64_t sparse_ro_offset,
   struct util_vma_heap *usc_heap, uint64_t usc_start, uint64_t usc_size)
{
   if (!agx_chip_native_agx3(chip))
      return true;

   uint64_t main_end, usc_end;
   if (!range_end(main_start, main_size, &main_end) ||
       !range_end(usc_start, usc_size, &usc_end))
      return false;

   struct agx_va_interval main_intervals[ARRAY_SIZE(g17p_kernel_reserved) * 2];
   unsigned main_count = 0;

   for (unsigned i = 0; i < ARRAY_SIZE(g17p_kernel_reserved); ++i) {
      const struct agx_va_interval reserved = g17p_kernel_reserved[i];
      add_intersection(main_intervals, &main_count, main_start, main_end,
                       reserved.start, reserved.end);

      /* Sparse buffers bind the main allocation again at sparse_ro_offset. */
      if (sparse_ro_offset < reserved.end) {
         const uint64_t shadow_start =
            sparse_ro_offset < reserved.start
               ? reserved.start - sparse_ro_offset
               : 0;
         const uint64_t shadow_end = reserved.end - sparse_ro_offset;

         add_intersection(main_intervals, &main_count, main_start, main_end,
                          shadow_start, shadow_end);
      }
   }

   if (!reserve_intervals(main_heap, main_intervals, main_count))
      return false;

   struct agx_va_interval usc_intervals[ARRAY_SIZE(g17p_kernel_reserved)];
   unsigned usc_count = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(g17p_kernel_reserved); ++i) {
      add_intersection(usc_intervals, &usc_count, usc_start, usc_end,
                       g17p_kernel_reserved[i].start,
                       g17p_kernel_reserved[i].end);
   }

   return reserve_intervals(usc_heap, usc_intervals, usc_count);
}

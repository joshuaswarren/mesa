/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "hk_compute_pair.h"

/* A byte range [start, end) of one BO, as seen through a storage backing. Two
 * descriptors alias exactly when they name the same BO handle and their ranges
 * intersect, whatever their buffer offsets or sizes. */
struct hk_compute_pair_interval {
   uint64_t start, end;
   uint32_t handle;
};

static inline bool
compute_pair_interval(const struct hk_storage_backing *backing,
                      uint64_t start, uint64_t end, struct hk_compute_pair_interval *out)
{
   if (!backing->handle || start >= end || end > backing->size ||
       backing->offset > UINT64_MAX - end)
      return false;
   *out = (struct hk_compute_pair_interval){
      backing->offset + start, backing->offset + end, backing->handle,
   };
   return true;
}

static inline bool
compute_pair_same_interval(const struct hk_compute_pair_interval *a,
                           const struct hk_compute_pair_interval *b)
{
   return a->handle == b->handle && a->start == b->start && a->end == b->end;
}

static inline bool
compute_pair_intervals_disjoint(const struct hk_compute_pair_interval *a,
                                const struct hk_compute_pair_interval *b)
{
   return a->handle != b->handle || a->end <= b->start || b->end <= a->start;
}

/* Whether the linear FADD `producer` (bindings 0 and 1 read, binding 2
 * written, all over the same element range) may also write the copy4
 * `copy4`'s destination as a second output. The copy source must be exactly
 * the producer's output, and the copy destination must not overlap the output
 * or either producer read; otherwise the two dispatches stay separate. */
static inline bool
hk_compute_pair_fadd_copy4_match(const struct hk_compute_pair_dispatch *producer,
                                 const struct hk_compute_pair_dispatch *copy4)
{
   const struct hk_compute_pair_copy4 *fp = &producer->extra.copy4;
   const struct hk_compute_pair_copy4 *cp = &copy4->extra.copy4;
   if (producer->info.role != HK_COMPUTE_PAIR_FADD ||
       producer->alt_kind != HK_COMPUTE_PAIR_ALT_LINEAR_FADD ||
       copy4->info.role != HK_COMPUTE_PAIR_COPY4 ||
       !copy4->bindings[1].address ||
       copy4->bindings[1].address > UINT64_MAX - cp->dst_start ||
       cp->src_start > UINT64_MAX - cp->bytes ||
       cp->dst_start > UINT64_MAX - cp->bytes ||
       hk_dual_write_push_offset(producer->info.fadd.push_bytes) +
          sizeof(struct hk_dual_copy_push) > HK_MAX_PUSH_SIZE ||
       ((copy4->bindings[1].address + cp->dst_start) % 16) != 0 ||
       fp->src_start > UINT64_MAX - fp->bytes ||
       fp->dst_start > UINT64_MAX - fp->bytes)
      return false;

   struct hk_compute_pair_interval output, src, dst;
   if (!compute_pair_interval(&producer->bindings[2], fp->dst_start,
                              fp->dst_start + fp->bytes, &output) ||
       !compute_pair_interval(&copy4->bindings[0], cp->src_start,
                              cp->src_start + cp->bytes, &src) ||
       !compute_pair_interval(&copy4->bindings[1], cp->dst_start,
                              cp->dst_start + cp->bytes, &dst) ||
       !compute_pair_same_interval(&output, &src) ||
       !compute_pair_intervals_disjoint(&output, &dst))
      return false;

   for (unsigned i = 0; i < 2; ++i) {
      struct hk_compute_pair_interval read;
      if (!compute_pair_interval(&producer->bindings[i], fp->src_start,
                                 fp->src_start + fp->bytes, &read) ||
          !compute_pair_intervals_disjoint(&read, &dst))
         return false;
   }
   return true;
}

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/* Sampler-private half-registers. COPY's destination and SMP's coordinate
 * operands name this file independently of the GPR file. A range remains live
 * until every asynchronous consumer of that range retires. Multiple consumers
 * may share a completion slot, or overlapping ranges may have different slots.
 */
#define AGX3_STAGING_HALFS 256u

struct agx3_staging_banks {
   uint16_t readers[AGX3_STAGING_HALFS];
   uint16_t end;
   uint8_t next_write_slot;
};

static inline void
agx3_staging_release(struct agx3_staging_banks *banks, uint16_t outstanding)
{
   for (unsigned i = 0; i < banks->end; ++i)
      banks->readers[i] &= outstanding;
}

static inline uint16_t
agx3_staging_conflicts(const struct agx3_staging_banks *banks, unsigned start,
                      unsigned span)
{
   uint16_t conflicts = 0;
   if (start >= banks->end)
      return 0;

   unsigned count = banks->end - start;
   if (span < count)
      count = span;
   for (unsigned i = start; i < start + count; ++i)
      conflicts |= banks->readers[i];
   return conflicts;
}

static inline void
agx3_staging_forget(struct agx3_staging_banks *banks, uint16_t slots)
{
   agx3_staging_release(banks, (uint16_t)~slots);
}

static inline bool
agx3_staging_hold(struct agx3_staging_banks *banks, uint16_t base,
                 uint16_t span, uint8_t slot)
{
   if (!span || span > AGX3_STAGING_HALFS ||
       base > AGX3_STAGING_HALFS - span || !slot || slot > 8)
      return false;

   if (base + span > banks->end)
      banks->end = base + span;
   for (unsigned i = base; i < base + span; ++i)
      banks->readers[i] |= 1u << (slot - 1);
   return true;
}

/* As agx3_staging_plan(), but take a free range at or above `avoid_end`
 * first: the start of the file belongs to a later fixed-location operand. */
static inline bool agx3_staging_plan(const struct agx3_staging_banks *banks,
                                     unsigned span, unsigned alignment,
                                     unsigned limit, uint16_t *base,
                                     uint16_t *waits);

static inline bool
agx3_staging_plan_above(const struct agx3_staging_banks *banks, unsigned span,
                        unsigned alignment, unsigned limit,
                        unsigned avoid_end, uint16_t *base, uint16_t *waits)
{
   if (avoid_end && alignment && !(alignment & (alignment - 1)) &&
       limit <= AGX3_STAGING_HALFS && span <= limit) {
      const unsigned first = (avoid_end + alignment - 1) & ~(alignment - 1);
      for (unsigned i = first; i + span <= limit; i += alignment) {
         if (!agx3_staging_conflicts(banks, i, span)) {
            *base = i;
            *waits = 0;
            return true;
         }
      }
   }
   return agx3_staging_plan(banks, span, alignment, limit, base, waits);
}

/* Plan an aligned allocation without changing any lifetime. Prefer a free
 * range; otherwise return the range requiring the fewest consumer slots to
 * retire. The caller must emit those waits before overwriting the range and
 * publish the new consumer with hold(). limit expresses the instruction's
 * addressable extent, not an arbitrary number of operations in flight.
 */
static inline bool
agx3_staging_plan(const struct agx3_staging_banks *banks, unsigned span,
                 unsigned alignment, unsigned limit, uint16_t *base,
                 uint16_t *waits)
{
   if (!base || !waits || !span || !alignment ||
       (alignment & (alignment - 1)) || limit > AGX3_STAGING_HALFS || span > limit)
      return false;

   unsigned best_count = 17;
   for (unsigned i = 0; i <= limit - span; i += alignment) {
      uint16_t conflict = agx3_staging_conflicts(banks, i, span);
      unsigned count = __builtin_popcount(conflict);
      if (count < best_count) {
         *base = i;
         *waits = conflict;
         best_count = count;
         if (!count)
            return true;
      }
   }
   return best_count != 17;
}

/* What emission knows about the generations in flight: the outstanding
 * slots, which of them are image writes, which write r0, which belong to
 * load-only local-read tile loads, and the staging halves their consumers
 * still read. At the arms of an if it joins as slot assignment does
 * (agx3_slot_core.h). */
struct agx3_emit_slots {
   uint16_t outstanding, image_writes, r0_inflight, local_read_load_only;
   struct agx3_staging_banks staging;
};

static inline void
agx3_emit_slots_capture(struct agx3_emit_slots *to, uint16_t outstanding,
                        uint16_t image_writes, uint16_t r0_inflight,
                        uint16_t local_read_load_only,
                        const struct agx3_staging_banks *staging)
{
   to->outstanding = outstanding;
   to->image_writes = image_writes & outstanding;
   to->r0_inflight = r0_inflight & outstanding;
   to->local_read_load_only = local_read_load_only & outstanding;
   to->staging = *staging;
   agx3_staging_release(&to->staging, outstanding);
}

static inline void
agx3_emit_slots_join(const struct agx3_emit_slots *from, uint16_t *outstanding,
                     uint16_t *image_writes, uint16_t *r0_inflight,
                     uint16_t *local_read_load_only,
                     struct agx3_staging_banks *staging)
{
   *outstanding |= from->outstanding;
   *image_writes |= from->image_writes;
   *r0_inflight |= from->r0_inflight;
   *local_read_load_only |= from->local_read_load_only;
   for (unsigned i = 0; i < from->staging.end; ++i)
      staging->readers[i] |= from->staging.readers[i];
   if (from->staging.end > staging->end)
      staging->end = from->staging.end;
}

/* A break joins what is in flight into the state its loop's pop resumes. */
static inline void
agx3_emit_slots_add(struct agx3_emit_slots *to, bool *has, uint16_t outstanding,
                    uint16_t image_writes, uint16_t r0_inflight,
                    uint16_t local_read_load_only,
                    const struct agx3_staging_banks *staging)
{
   if (!*has) {
      agx3_emit_slots_capture(to, outstanding, image_writes, r0_inflight,
                              local_read_load_only, staging);
      *has = true;
      return;
   }
   to->outstanding |= outstanding;
   to->image_writes |= image_writes & outstanding;
   to->r0_inflight |= r0_inflight & outstanding;
   to->local_read_load_only |= local_read_load_only & outstanding;
   for (unsigned i = 0; i < staging->end; ++i)
      to->staging.readers[i] |= staging->readers[i] & outstanding;
   if (staging->end > to->staging.end)
      to->staging.end = staging->end;
}

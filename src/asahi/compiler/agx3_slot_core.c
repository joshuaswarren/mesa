/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_slot_core.h"

#include <string.h>

/* Whether a register still belongs to this generation, as result or payload. */
static bool
agx3_slot_generation_live(const uint8_t *reg_slot,
                          const uint16_t *reader_slots, unsigned slot,
                          unsigned token_count)
{
   for (unsigned r = 0; r < token_count; ++r) {
      if (reg_slot[r] == slot || (reader_slots[r] & (1u << (slot - 1))))
         return true;
   }
   return false;
}

/* Most regions touch only GPRs. Do not scan the entire static TPS address
 * space every time a scoreboard generation retires. Every writer alias and
 * payload reader below comes from a source or destination in this region,
 * so tokens beyond this bound remain empty throughout the assignment.
 *
 * Leave malformed-input handling to the assignment pass, including which
 * prefix of the output it has written when it rejects an instruction. */
static unsigned
agx3_slot_token_count(const struct agx3_slot_instr *instrs, unsigned n)
{
   unsigned count = 1;
   for (unsigned i = 0; i < n; ++i) {
      const struct agx3_slot_instr *I = &instrs[i];
      if (I->nr_srcs > AGX3_SLOT_MAX_REGS || I->nr_dsts > AGX3_SLOT_MAX_REGS)
         return AGX3_SLOT_NUM_TOKENS;

      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         if (I->src[s] >= AGX3_SLOT_NUM_TOKENS)
            return AGX3_SLOT_NUM_TOKENS;
         if (I->src[s] >= count)
            count = I->src[s] + 1u;
      }
      for (unsigned d = 0; d < I->nr_dsts; ++d) {
         if (I->dst[d] >= AGX3_SLOT_NUM_TOKENS)
            return AGX3_SLOT_NUM_TOKENS;
         if (I->dst[d] >= count)
            count = I->dst[d] + 1u;
      }
   }
   return count;
}

static bool
agx3_slot_assign_initial(const struct agx3_slot_instr *instrs, unsigned n,
                            unsigned nslots, const struct agx3_slot_entry *entry,
                            unsigned entry_count,
                            const struct agx3_slot_live_range *live,
                            unsigned live_count, struct agx3_slot_result *out)
{
   if (instrs == NULL || out == NULL || nslots < 1 || nslots > 8 ||
       (entry_count && entry == NULL) || entry_count > AGX3_SLOT_NUM_TOKENS ||
       (live_count && !live))
      return false;

   unsigned token_count = agx3_slot_token_count(instrs, n);
   for (unsigned i = 0; i < entry_count; ++i) {
      if (entry[i].token >= AGX3_SLOT_NUM_TOKENS ||
          entry[i].slot < 1 || entry[i].slot > nslots)
         return false;
      if (entry[i].token >= token_count)
         token_count = entry[i].token + 1u;
   }

   for (unsigned i = 0; i < live_count; ++i) {
      const struct agx3_slot_live_range *range = &live[i];
      if (!range->count || !range->slot || range->slot > nslots ||
          range->first >= AGX3_SLOT_NUM_TOKENS ||
          range->count > AGX3_SLOT_NUM_TOKENS - range->first)
         return false;
      if (range->first + range->count > token_count)
         token_count = range->first + range->count;
   }

   /* reg_slot[r] = slot (1..nslots) of the async op whose result currently
    * lives in half-register r, or 0 if r was last written by a synchronous op
    * (or never). */
   uint8_t reg_slot[AGX3_SLOT_NUM_TOKENS];
   memset(reg_slot, 0, token_count * sizeof(reg_slot[0]));
   uint16_t pending = 0;
   for (unsigned i = 0; i < entry_count; ++i) {
      /* One token cannot name two independently pending generations. */
      if (reg_slot[entry[i].token])
         return false;
      reg_slot[entry[i].token] = entry[i].slot;
      pending |= (uint16_t)(1u << (entry[i].slot - 1));
   }
   for (unsigned i = 0; i < live_count; ++i) {
      const struct agx3_slot_live_range *range = &live[i];
      for (unsigned r = range->first; r < range->first + range->count; ++r) {
         if (reg_slot[r])
            return false;
         reg_slot[r] = range->slot;
      }
      pending |= (uint16_t)(1u << (range->slot - 1));
   }
   uint16_t entry_pending = pending;

   /* reader_slots[r] = the slots of the in-flight stores that latch
    * half-register r as payload. A store has no destination, so the result
    * bookkeeping above never covers its payload: the transform feedback copy
    * program reuses one payload quad for all sixteen captured varyings, and
    * the next varying's setup raced the outstanding store. */
   uint16_t reader_slots[AGX3_SLOT_NUM_TOKENS];
   memset(reader_slots, 0, token_count * sizeof(reader_slots[0]));
   uint16_t active_reader_slots = 0;

   /* slot_waited[s] = the slot has been synchronised since it was last
    * allocated, so later consumers need not wait it again. */
   bool slot_waited[9] = {false};

   /* slot_reissuable[s] = the slot's outstanding generation was issued by an
    * instruction that reissues live slots itself. */
   bool slot_reissuable[9] = {false};
   uint8_t slot_join_class[9] = {0};
   uint8_t slot_join_count[9] = {0};

   unsigned next = 0; /* round-robin cursor into 0..nslots-1 */

   for (unsigned i = 0; i < n; ++i) {
      const struct agx3_slot_instr *I = &instrs[i];
      if (I->nr_srcs > AGX3_SLOT_MAX_REGS || I->nr_dsts > AGX3_SLOT_MAX_REGS)
         return false;
      if (I->preferred_slot > nslots || (!I->is_async && I->preferred_slot))
         return false;
      if (I->coalesce_src > I->nr_srcs || (!I->is_async && I->coalesce_src))
         return false;
      if ((I->join_class && (!I->is_async || !I->join_limit || I->join_limit > 8)) ||
          (!I->join_class && (I->join_limit || I->prefer_join)))
         return false;
      const uint16_t valid_slots = (uint16_t)((1u << nslots) - 1u);
      if ((I->nr_srcs < 32 && (I->late_src_mask >> I->nr_srcs)) ||
          (!I->is_async && I->late_src_mask))
         return false;
      if ((I->transient_slots & ~valid_slots) ||
          (I->preferred_slot &&
           (I->transient_slots & (1u << (I->preferred_slot - 1)))))
         return false;

      uint16_t wait_mask = 0;
      if (I->drain) {
         wait_mask = pending;
         for (unsigned slot = 1; slot <= nslots; ++slot)
            if (pending & (1u << (slot - 1)))
               slot_waited[slot] = true;
      }
      for (unsigned s = 0; s < I->nr_srcs; ++s) {
         const uint16_t r = I->src[s];
         if (r >= AGX3_SLOT_NUM_TOKENS)
            return false;
         const uint8_t slot = reg_slot[r];
         if (slot != 0 && !slot_waited[slot]) {
            wait_mask |= (uint16_t)(1u << (slot - 1));
            slot_waited[slot] = true;
         }
      }

      /*
       * Write-after-write. An async result that is still in flight for this
       * register lands LATER than this instruction, so a synchronous write
       * here is silently overwritten by the load when it retires -- and any
       * consumer that waits the slot then reads the loaded word instead of
       * the computed one. Mesa's own AGX2 pass checks this case explicitly
       * (agx_insert_waits.c, "Check for write-after-write"); omitting it here
       * produced no diagnostic, just wrong values on hardware.
       *
       * This runs before the bookkeeping below, so reg_slot[] still names the
       * previous writer rather than this instruction.
       */
      for (unsigned d = 0; d < I->nr_dsts; ++d) {
         const uint16_t r = I->dst[d];
         if (r >= AGX3_SLOT_NUM_TOKENS)
            return false;
         const uint8_t slot = reg_slot[r];
         if (slot != 0 && !slot_waited[slot]) {
            wait_mask |= (uint16_t)(1u << (slot - 1));
            slot_waited[slot] = true;
         }

         /* Write-after-read: retire every async op still reading this
          * register before redefining it. */
         uint16_t readers = reader_slots[r];
         while (readers) {
            const unsigned bit = (unsigned)(readers & (~readers + 1u));
            const uint8_t reader = (uint8_t)(__builtin_ctz(bit) + 1);
            readers &= (uint16_t)~bit;
            if (!slot_waited[reader]) {
               wait_mask |= (uint16_t)bit;
               slot_waited[reader] = true;
            }
         }
      }

      /* A compound instruction may use a scoreboard slot internally, for
       * example PACK_SMPIR/COPY producers followed by a TEX which waits for
       * them. Retire a prior generation before reusing the slot, then forget
       * its aliases: the prewait has made those GPR values synchronous and
       * the internal generation is consumed before this logical instruction
       * returns. */
      for (unsigned slot = 1; slot <= nslots; ++slot) {
         const uint16_t bit = (uint16_t)(1u << (slot - 1));
         if (!(I->transient_slots & bit))
            continue;

         bool live = false;
         for (unsigned r = 0; r < token_count; ++r) {
            live |= (reader_slots[r] & bit) != 0;
            if (reg_slot[r] == slot) {
               live = true;
               reg_slot[r] = 0;
            }
         }
         if (live && !slot_waited[slot])
            wait_mask |= bit;
         slot_waited[slot] = true;
         pending &= (uint16_t)~bit;
      }

      /* Every slot this instruction waited is now retired, so the registers
       * its async ops were reading are free again. */
      if (wait_mask & active_reader_slots) {
         for (unsigned r = 0; r < token_count; ++r)
            reader_slots[r] &= (uint16_t)~wait_mask;
         active_reader_slots &= (uint16_t)~wait_mask;
      }

      if (I->is_async) {
         /*
          * A slot names one generation at a time. Issuing it while an earlier
          * generation is still outstanding for a live result leaves that
          * result's consumer waiting on the new generation instead, so it can
          * read its register before the older operation has written it. The
          * round-robin allocation therefore skips such slots, and when every
          * slot is in that state it waits the old generation out first.
          *
          * A preferred slot keeps its witnessed protocol: Apple's fragment
          * tile loads, for one, issue slot 6 again for each tile read. So
          * does an instruction that reissues live slots, such as a fragment
          * coefficient load, which has no wait field: it only skips a slot
          * whose live generation some other kind of instruction issued, and
          * when every slot is in that state it takes the next one as before.
          */
         uint8_t slot = I->preferred_slot;
         const unsigned cap =
            I->max_slot && I->max_slot < nslots ? I->max_slot : nslots;
         if (!slot) {
            if (I->coalesce_src) {
               const unsigned source = I->src[I->coalesce_src - 1];
               const uint8_t producer_slot = reg_slot[source];
               if (producer_slot && producer_slot <= cap &&
                   slot_waited[producer_slot] &&
                   !(I->transient_slots & (1u << (producer_slot - 1))))
                  slot = producer_slot;
            }
            if (I->prefer_join && !slot) {
               for (unsigned t = 0; t < nslots && !slot; ++t) {
                  const uint8_t candidate = (uint8_t)((next + t) % nslots + 1);
                  if (candidate <= cap && !slot_waited[candidate] &&
                      slot_join_class[candidate] == I->join_class &&
                      slot_join_count[candidate] < I->join_limit)
                     slot = candidate;
               }
            }
            for (unsigned t = 0; t < nslots && !slot; ++t) {
               const uint8_t candidate = (uint8_t)((next + t) % nslots + 1);
               if (candidate > cap)
                  continue;
               if (slot_waited[candidate] ||
                   !agx3_slot_generation_live(reg_slot, reader_slots, candidate,
                                             token_count) ||
                   (I->reissues_live_slot && slot_reissuable[candidate]) ||
                   (I->join_class && slot_join_class[candidate] == I->join_class &&
                    slot_join_count[candidate] < I->join_limit))
                  slot = candidate;
            }
            if (!slot) {
               slot = (uint8_t)(next + 1);
               if (slot > cap)
                  slot = 1;
               if (!I->reissues_live_slot) {
                  wait_mask |= (uint16_t)(1u << (slot - 1));
                  slot_waited[slot] = true;
               }
            }
            next = slot % nslots;
         }

         /* A preferred/reissuable native instruction may normally join an
          * in-region generation. The external loader's generation is not
          * part of that protocol: retire it before any slot reuse. */
         if ((entry_pending & (1u << (slot - 1))) && !slot_waited[slot]) {
            wait_mask |= (uint16_t)(1u << (slot - 1));
            slot_waited[slot] = true;
         }

         /* A joined queue must complete before an incompatible producer
          * replaces it. In particular, an SCS completion or a texture
          * publication must not inherit a pending device-read generation. */
         if (!slot_waited[slot] && slot_join_count[slot] &&
             ((slot_join_class[slot] != I->join_class) ||
              (I->join_class && slot_join_count[slot] >= I->join_limit))) {
            wait_mask |= (uint16_t)(1u << (slot - 1));
            slot_waited[slot] = true;
         }

         out[i].wait_mask = wait_mask;
         out[i].slot = slot;
         entry_pending &= (uint16_t)~wait_mask;
         pending &= (uint16_t)~wait_mask;
         pending |= (uint16_t)(1u << (slot - 1));

         /* Allocation can add a wait when every generation is occupied.
          * Retire its payload readers before recording this instruction. */
         if (wait_mask & active_reader_slots) {
            for (unsigned r = 0; r < token_count; ++r)
               reader_slots[r] &= (uint16_t)~wait_mask;
            active_reader_slots &= (uint16_t)~wait_mask;
         }

         /* Reusing an already-synchronized slot starts a new dependency
          * generation. Retire the old generation's register aliases before
          * publishing the new result, otherwise an unrelated later write can
          * consume the new generation's wait and hide it from its consumer. */
         const bool new_join = slot_waited[slot] ||
                               slot_join_class[slot] != I->join_class;
         if (slot_waited[slot]) {
            for (unsigned r = 0; r < token_count; ++r) {
               if (reg_slot[r] == slot)
                  reg_slot[r] = 0;
            }
         }
         slot_waited[slot] = false; /* freshly outstanding */
         slot_reissuable[slot] = I->reissues_live_slot;
         slot_join_count[slot] = I->join_class ?
            (new_join ? 1 : slot_join_count[slot] + 1) : 0;
         slot_join_class[slot] = I->join_class;
         for (unsigned d = 0; d < I->nr_dsts; ++d) {
            if (I->dst[d] >= AGX3_SLOT_NUM_TOKENS)
               return false;
            reg_slot[I->dst[d]] = slot;
         }
         if ((I->latches_srcs || I->late_src_mask) && I->nr_srcs) {
            active_reader_slots |= (uint16_t)(1u << (slot - 1));
            for (unsigned s = 0; s < I->nr_srcs; ++s) {
               if (I->latches_srcs || (I->late_src_mask & (UINT32_C(1) << s)))
                  reader_slots[I->src[s]] |= (uint16_t)(1u << (slot - 1));
            }
         }
      } else {
         out[i].wait_mask = wait_mask;
         out[i].slot = 0;
         entry_pending &= (uint16_t)~wait_mask;
         pending &= (uint16_t)~wait_mask;
         for (unsigned d = 0; d < I->nr_dsts; ++d) {
            if (I->dst[d] >= AGX3_SLOT_NUM_TOKENS)
               return false;
            reg_slot[I->dst[d]] = 0; /* synchronous result, no wait needed */
         }
      }
   }

   return true;
}

bool
agx3_slot_assign_with_entry(const struct agx3_slot_instr *instrs, unsigned n,
                            unsigned nslots, const struct agx3_slot_entry *entry,
                            unsigned entry_count, struct agx3_slot_result *out)
{
   return agx3_slot_assign_initial(instrs, n, nslots, entry, entry_count,
                                   NULL, 0, out);
}

bool
agx3_slot_assign_with_live(const struct agx3_slot_instr *instrs, unsigned n,
                           unsigned nslots,
                           const struct agx3_slot_live_range *live,
                           unsigned live_count, struct agx3_slot_result *out)
{
   return agx3_slot_assign_initial(instrs, n, nslots, NULL, 0,
                                   live, live_count, out);
}

bool
agx3_slot_assign(const struct agx3_slot_instr *instrs, unsigned n,
                 unsigned nslots, struct agx3_slot_result *out)
{
   return agx3_slot_assign_with_entry(instrs, n, nslots, NULL, 0, out);
}

void
agx3_slot_retire_at_cf(const struct agx3_slot_instr *instrs, unsigned n,
                       uint16_t entry_wait, struct agx3_slot_result *res,
                       bool *drained)
{
   uint16_t retired = 0, live = entry_wait;
   for (unsigned i = 0; i < n; ++i) {
      res[i].wait_mask &= (uint16_t)~retired;
      live &= (uint16_t)~res[i].wait_mask;

      if (res[i].slot) {
         const uint16_t bit = (uint16_t)(1u << (res[i].slot - 1));
         retired &= (uint16_t)~bit;
         live |= bit;
      }

      if (instrs[i].cf == AGX3_SLOT_CF_NONE)
         continue;

      if (drained) {
         uint16_t mask = live;
         if (instrs[i].cf == AGX3_SLOT_CF_DRAIN_EMPTY && i + 1 < n &&
             instrs[i + 1].cf != AGX3_SLOT_CF_NONE)
            mask &= (uint16_t)~res[i + 1].wait_mask;
         drained[i] = mask != 0;
      }
      retired |= live;
      live = 0;
   }
}

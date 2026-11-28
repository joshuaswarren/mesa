/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Correctness of the AGX3 scoreboard slot-assignment algorithm.
 */
#include "agx3_slot_core.h"
#include <stdio.h>
#include <string.h>

static unsigned checks, failures;
#define CHECK(x) do{ checks++; if(!(x)){ fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,#x); failures++; } }while(0)

static struct agx3_slot_instr
ld(uint16_t d){ struct agx3_slot_instr I={0}; I.is_async=true; I.nr_dsts=1; I.dst[0]=d; return I; }
static struct agx3_slot_instr
alu(uint16_t d, uint16_t a, uint16_t b){ struct agx3_slot_instr I={0}; I.nr_dsts=1; I.dst[0]=d; I.nr_srcs=2; I.src[0]=a; I.src[1]=b; return I; }
static struct agx3_slot_instr
st(uint16_t a){ struct agx3_slot_instr I={0}; I.nr_srcs=1; I.src[0]=a; return I; }
/* An asynchronous store: no destination, one payload register it reads. */
static struct agx3_slot_instr
stdev(uint16_t a){ struct agx3_slot_instr I={0}; I.is_async=true; I.latches_srcs=true; I.nr_srcs=1; I.src[0]=a; return I; }
static struct agx3_slot_instr
mov(uint16_t d){ struct agx3_slot_instr I={0}; I.nr_dsts=1; I.dst[0]=d; return I; }

static struct agx3_slot_instr
input_reload(uint16_t dest, uint16_t tps_half)
{
   struct agx3_slot_instr I = {
      .is_async = true,
      .preferred_slot = 2,
      .nr_srcs = 2,
      .nr_dsts = 4,
      .src = {AGX3_SLOT_TPS_TOKEN_BASE + tps_half,
              AGX3_SLOT_TPS_TOKEN_BASE + tps_half + 1},
      .dst = {dest, dest + 1, AGX3_SLOT_TPS_TOKEN_BASE + tps_half,
              AGX3_SLOT_TPS_TOKEN_BASE + tps_half + 1},
   };
   return I;
}

/* A fragment sample under the separate-staging policy: staging on slot 6
 * (transient), the result on slots 1..5, two results per slot. */
static struct agx3_slot_instr
staged_sample(uint16_t d)
{
   struct agx3_slot_instr I = ld(d);
   I.transient_slots = 1u << 5;
   I.max_slot = 5;
   I.join_class = 2;
   I.join_limit = 2;
   return I;
}

static void
check_sample_runs(void)
{
   struct agx3_slot_result r[16];

   /* Nine independent samples, then their sum: no sample waits an earlier
    * result. The first five take slots 1..5 and the next four join them. */
   {
      struct agx3_slot_instr p[11];
      for (unsigned i = 0; i < 9; ++i)
         p[i] = staged_sample(2 * i);
      p[9] = alu(20, 0, 16);
      p[10] = st(20);
      CHECK(agx3_slot_assign(p, 11, 6, r));
      for (unsigned i = 0; i < 9; ++i) {
         CHECK(r[i].wait_mask == 0);
         CHECK(r[i].slot >= 1 && r[i].slot <= 5);
      }
      CHECK(r[0].slot == r[5].slot);
      CHECK(r[9].wait_mask ==
            ((1u << (r[0].slot - 1)) | (1u << (r[8].slot - 1))));
   }

   /* Without the policy, the seventh of nine samples on six slots waits the
    * first sample's result before it can reuse that slot. */
   {
      struct agx3_slot_instr p[9];
      for (unsigned i = 0; i < 9; ++i)
         p[i] = ld(2 * i);
      CHECK(agx3_slot_assign(p, 9, 6, r));
      CHECK(r[6].slot == r[0].slot);
      CHECK(r[6].wait_mask == (1u << (r[0].slot - 1)));
   }

   /* A joined slot holds at most two results: an eleventh sample waits. */
   {
      struct agx3_slot_instr p[11];
      for (unsigned i = 0; i < 11; ++i)
         p[i] = staged_sample(2 * i);
      CHECK(agx3_slot_assign(p, 11, 6, r));
      for (unsigned i = 0; i < 10; ++i)
         CHECK(r[i].wait_mask == 0);
      CHECK(r[10].wait_mask != 0);
   }
}

static void
check_live_inputs(void)
{
   const uint16_t input = AGX3_SLOT_TPS_TOKEN_BASE + 8;
   struct agx3_slot_result r[8];
   const struct agx3_slot_live_range live = {input, 4, 6};

   /* A release fence includes outstanding entry producers even before any
    * instruction reads their results. Wait only the still-pending slots. */
   for (unsigned consume = 0; consume < 2; ++consume) {
      const struct agx3_slot_live_range ranges[] = {
         {input, 2, 6}, {input + 2, 2, 3},
      };
      const struct agx3_slot_instr fence = {.drain = true};
      const struct agx3_slot_instr p[] = {
         consume ? st(input) : mov(8), fence, st(input), st(input + 2), fence,
      };
      CHECK(agx3_slot_assign_with_live(p, 5, 6, ranges, 2, r));
      CHECK(r[0].wait_mask == (consume ? 32 : 0));
      CHECK(r[1].wait_mask == (consume ? 4 : 36));
      CHECK(r[2].wait_mask == 0 && r[3].wait_mask == 0 && r[4].wait_mask == 0);
   }

   /* Native G17P pointer-input entry: the loader's TPS writes signal slot 6,
    * the first RLD waits 6 and publishes on 2, and the second RLD publishes
    * on 2 without repeating the loader wait. Their GPR consumer waits 2. */
   {
      const struct agx3_slot_instr p[] = {
         input_reload(0, 8), input_reload(2, 10), alu(4, 0, 2), st(4),
      };
      CHECK(agx3_slot_assign_with_live(p, 4, 6, &live, 1, r));
      CHECK(r[0].slot == 2 && r[0].wait_mask == 32);
      CHECK(r[1].slot == 2 && r[1].wait_mask == 0);
      CHECK(r[2].slot == 0 && r[2].wait_mask == 2);
      CHECK(r[3].slot == 0 && r[3].wait_mask == 0);
   }

   /* An unrelated GPR with the same half index does not read TPS. A later
    * overwrite of any pending input must wait before it replaces the value;
    * that retires the entire incoming slot, including disjoint ranges. */
   {
      const struct agx3_slot_live_range ranges[] = {
         {input, 2, 6}, {input + 10, 2, 6},
      };
      const struct agx3_slot_instr p[] = {
         mov(8), mov(input + 1), st(input + 10), st(input),
      };
      CHECK(agx3_slot_assign_with_live(p, 4, 6, ranges, 2, r));
      CHECK(r[0].wait_mask == 0);
      CHECK(r[1].wait_mask == 32);
      CHECK(r[2].wait_mask == 0 && r[3].wait_mask == 0);
   }

   /* Retiring the loader does not make a new producer on slot 6 ready. Old
    * input aliases must not steal the new GPR result's dependency either. */
   {
      struct agx3_slot_instr fresh = ld(20);
      fresh.preferred_slot = 6;
      const struct agx3_slot_instr p[] = {
         st(input), fresh, mov(input), st(input + 3), st(20), st(20),
      };
      CHECK(agx3_slot_assign_with_live(p, 6, 6, &live, 1, r));
      CHECK(r[0].wait_mask == 32);
      CHECK(r[1].slot == 6 && r[1].wait_mask == 0);
      CHECK(r[2].wait_mask == 0 && r[3].wait_mask == 0);
      CHECK(r[4].wait_mask == 32 && r[5].wait_mask == 0);
   }

   /* An instruction with an internal slot-6 producer must retire the
    * incoming generation before reusing it, even if it reads no TPS. */
   {
      struct agx3_slot_instr compound = ld(20);
      compound.preferred_slot = 1;
      compound.transient_slots = 32;
      const struct agx3_slot_instr p[] = {compound, st(input), st(20)};
      CHECK(agx3_slot_assign_with_live(p, 3, 6, &live, 1, r));
      CHECK(r[0].slot == 1 && r[0].wait_mask == 32);
      CHECK(r[1].wait_mask == 0 && r[2].wait_mask == 1);
   }

   /* The full 512-half input domain still needs only one producer wait.
    * Probe every half independently to catch truncated imported ranges. */
   const struct agx3_slot_live_range full = {input, 512, 6};
   for (unsigned half = 0; half < 512; ++half) {
      const struct agx3_slot_instr p[] = {
         st(input + half), st(input + (511 - half)),
      };
      CHECK(agx3_slot_assign_with_live(p, 2, 6, &full, 1, r));
      CHECK(r[0].wait_mask == 32 && r[1].wait_mask == 0);
   }

   /* Invalid entry metadata is rejected before any output result changes. */
   const struct agx3_slot_live_range invalid[] = {
      {input, 0, 6}, {input, 1, 0}, {input, 1, 7},
      {AGX3_SLOT_NUM_TOKENS, 1, 6}, {AGX3_SLOT_NUM_TOKENS - 1, 2, 6},
      {UINT16_MAX, UINT16_MAX, 6},
   };
   const struct agx3_slot_instr p = st(input);
   memset(r, 0xa5, sizeof(r));
   struct agx3_slot_result before[8];
   memcpy(before, r, sizeof(r));
   for (unsigned i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
      CHECK(!agx3_slot_assign_with_live(&p, 1, 6, &invalid[i], 1, r));
      CHECK(memcmp(r, before, sizeof(r)) == 0);
   }
   const struct agx3_slot_live_range overlap[] = {
      {input, 4, 6}, {input + 3, 2, 6},
   };
   CHECK(!agx3_slot_assign_with_live(&p, 1, 6, overlap, 2, r));
   CHECK(memcmp(r, before, sizeof(r)) == 0);
   CHECK(!agx3_slot_assign_with_live(&p, 1, 6, NULL, 1, r));
   CHECK(memcmp(r, before, sizeof(r)) == 0);
}

int main(void)
{
   struct agx3_slot_result r[64];
   check_live_inputs();
   check_sample_runs();

   /* SMP latches only its staging group through completion. A descriptor
    * GPR may be reused immediately; a staging rewrite must wait for SMP. */
   {
      const unsigned staging = AGX3_SLOT_STAGING_TOKEN_BASE;
      struct agx3_slot_instr publish = ld(staging);
      publish.preferred_slot = 1;
      struct agx3_slot_instr sample = ld(32);
      sample.preferred_slot = 2;
      sample.nr_srcs = 2;
      sample.src[0] = 8;
      sample.src[1] = staging;
      sample.late_src_mask = 2;
      struct agx3_slot_instr p[] = {
         publish, sample, mov(8), mov(staging), st(32),
      };
      CHECK(agx3_slot_assign(p, 5, 6, r));
      CHECK(r[1].wait_mask == 1);
      CHECK(r[2].wait_mask == 0);
      CHECK(r[3].wait_mask == 2);
      CHECK(r[4].wait_mask == 0);
      sample.late_src_mask = 4;
      CHECK(!agx3_slot_assign(&sample, 1, 6, r));
      sample.late_src_mask = 2;
      sample.is_async = false;
      sample.preferred_slot = 0;
      CHECK(!agx3_slot_assign(&sample, 1, 6, r));
   }

   /* GPR publication inputs are retained until the COPY generation completes;
    * different staging groups do not acquire false alias dependencies. */
   {
      struct agx3_slot_instr publish = ld(AGX3_SLOT_STAGING_TOKEN_BASE);
      publish.preferred_slot = 1;
      publish.nr_srcs = 1;
      publish.src[0] = 0;
      publish.latches_srcs = true;
      struct agx3_slot_instr sample = ld(32);
      sample.preferred_slot = 2;
      sample.nr_srcs = 1;
      sample.src[0] = AGX3_SLOT_STAGING_TOKEN_BASE;
      sample.late_src_mask = 1;
      struct agx3_slot_instr p[] = {
         publish, mov(0), sample, mov(AGX3_SLOT_STAGING_TOKEN_BASE + 1),
         mov(AGX3_SLOT_STAGING_TOKEN_BASE),
      };
      CHECK(agx3_slot_assign(p, 5, 6, r));
      CHECK(r[1].wait_mask == 1);
      CHECK(r[2].wait_mask == 0);
      CHECK(r[3].wait_mask == 0);
      CHECK(r[4].wait_mask == 2);
   }

   /* Release ordering includes a load used only after the fence. Drain only
    * issued slots, and start a fresh generation for subsequent loads. */
   {
      struct agx3_slot_instr fence = {.drain = true};
      struct agx3_slot_instr p[] = {
         fence, ld(0), ld(2), st(2), fence, st(0), ld(4), fence, fence,
      };
      CHECK(agx3_slot_assign(p, 9, 6, r));
      CHECK(r[0].wait_mask == 0);
      CHECK(r[3].wait_mask == 2);
      CHECK(r[4].wait_mask == 1);
      CHECK(r[5].wait_mask == 0);
      CHECK(r[7].wait_mask == 4);
      CHECK(r[8].wait_mask == 0);
   }

   /* A resultless store is still ordered, and its old payload generations
    * must not create waits after the fence or after transient slot reuse. */
   {
      struct agx3_slot_instr fence = {.drain = true};
      struct agx3_slot_instr texture = ld(8);
      texture.preferred_slot = 3;
      texture.transient_slots = 2;
      struct agx3_slot_instr p[] = {
         stdev(0), fence, mov(0), stdev(4), texture, fence, fence,
      };
      CHECK(agx3_slot_assign(p, 7, 6, r));
      CHECK(r[1].wait_mask == 1);
      CHECK(r[2].wait_mask == 0);
      CHECK(r[4].wait_mask == 2);
      CHECK(r[5].wait_mask == 4);
      CHECK(r[6].wait_mask == 0);
   }

   /* Two loads then an add consuming both: each load gets a distinct slot, the
    * add waits both. */
   {
      struct agx3_slot_instr p[] = { ld(0), ld(2), alu(4,0,2), st(4) };
      CHECK(agx3_slot_assign(p, 4, 6, r));
      CHECK(r[0].slot==1 && r[0].wait_mask==0);
      CHECK(r[1].slot==2 && r[1].wait_mask==0);
      CHECK(r[2].slot==0 && r[2].wait_mask==0x3u);  /* wait slots 1 and 2 */
      CHECK(r[3].slot==0 && r[3].wait_mask==0);      /* add result is sync */
   }

   /* Write-after-read. An asynchronous store latches its payload when it
    * retires, so redefining that register first must wait the store's slot.
    * A store has no destination, so nothing else tracks its payload: this is
    * what corrupted the transform feedback copy program, which reuses one
    * payload quad for every captured varying. */
   {
      struct agx3_slot_instr p[] = { stdev(8), mov(8), stdev(8) };
      CHECK(agx3_slot_assign(p, 3, 6, r));
      CHECK(r[0].slot==1 && r[0].wait_mask==0);
      CHECK(r[1].slot==0 && r[1].wait_mask==0x1u); /* wait the store */
      CHECK(r[2].slot==2);
   }

   /* Having waited the slot once, a second redefinition needs no further
    * wait until a new generation is outstanding. */
   {
      struct agx3_slot_instr p[] = { stdev(8), mov(8), mov(8) };
      CHECK(agx3_slot_assign(p, 3, 6, r));
      CHECK(r[1].wait_mask==0x1u);
      CHECK(r[2].wait_mask==0);
   }

   /* A register no outstanding operation reads is free to redefine. */
   {
      struct agx3_slot_instr p[] = { stdev(8), mov(10) };
      CHECK(agx3_slot_assign(p, 2, 6, r));
      CHECK(r[1].wait_mask==0);
   }

   /* Waited once = satisfied globally: two consumers of the same load; only the
    * first waits. */
   {
      struct agx3_slot_instr p[] = { ld(0), alu(2,0,0), alu(4,0,0) };
      CHECK(agx3_slot_assign(p, 3, 6, r));
      CHECK(r[0].slot==1);
      CHECK(r[1].wait_mask==0x1u);  /* first consumer waits slot 1 */
      CHECK(r[2].wait_mask==0);    /* already satisfied */
   }

   /* Pure ALU: no slots, no waits. */
   {
      struct agx3_slot_instr p[] = { alu(0,2,4), alu(6,0,2) };
      CHECK(agx3_slot_assign(p, 2, 6, r));
      CHECK(r[0].slot==0 && r[0].wait_mask==0);
      CHECK(r[1].slot==0 && r[1].wait_mask==0);
   }

   /* Write-after-write. An ALU op that overwrites a register an in-flight load
    * is still going to land in MUST wait for that load first: otherwise the
    * load retires afterwards and silently replaces the computed value. This
    * case previously asserted the opposite (that the ALU need not wait), and
    * the resulting miscompile reached hardware -- a compute shader stored the
    * 4th word of its own descriptor load instead of the value it computed. */
   {
      struct agx3_slot_instr p[] = { ld(0), alu(0,2,4), alu(6,0,0) };
      CHECK(agx3_slot_assign(p, 3, 6, r));
      CHECK(r[0].slot==1);
      CHECK(r[1].wait_mask==0x1u); /* WAW against the pending load */
      CHECK(r[2].wait_mask==0);    /* slot 1 already retired above */
   }

   /* The exact shape that miscompiled: a 4-component descriptor load claims
    * r0..r3, only r0:r1 are used for the address, and the register allocator
    * reuses r3 for the stored value. The write to r3 must retire the load. */
   {
      struct agx3_slot_instr load = {0};
      load.is_async = true;
      load.nr_dsts = 4;
      load.dst[0] = 0; load.dst[1] = 2; load.dst[2] = 4; load.dst[3] = 6;

      struct agx3_slot_instr movimm = {0};
      movimm.nr_dsts = 1; movimm.dst[0] = 6;      /* r3 = imm */

      struct agx3_slot_instr p[] = { load, movimm, alu(6,6,4), st(6) };
      CHECK(agx3_slot_assign(p, 4, 6, r));
      CHECK(r[0].slot==1);
      CHECK(r[1].wait_mask==0x1u); /* must wait before clobbering r3 */
      CHECK(r[2].wait_mask==0);
      CHECK(r[3].wait_mask==0);
   }

   /* A write that does NOT alias the pending load still needs no wait. */
   {
      struct agx3_slot_instr p[] = { ld(0), alu(8,2,4), st(8) };
      CHECK(agx3_slot_assign(p, 3, 6, r));
      CHECK(r[0].slot==1);
      CHECK(r[1].wait_mask==0);
      CHECK(r[2].wait_mask==0);
   }

   /* Round-robin over nslots: the 7th async op on a 6-slot part wraps to
    * slot 1. Every slot still holds an outstanding live result, so it waits
    * the first load before reissuing slot 1; otherwise the first load's
    * consumer would wait the 7th load instead. */
   {
      struct agx3_slot_instr p[8];
      for (int i=0;i<7;i++) p[i]=ld((uint16_t)(i*2));
      p[7]=st(0);  /* consume the first load, slot 1 */
      CHECK(agx3_slot_assign(p, 8, 6, r));
      CHECK(r[0].slot==1 && r[1].slot==2 && r[5].slot==6 && r[6].slot==1);
      CHECK(r[6].wait_mask==0x1u);
      CHECK(r[7].wait_mask==0);
   }

   /* The wrap skips a slot whose outstanding generation is still live for a
    * later consumer, taking the next already-waited one instead. */
   {
      struct agx3_slot_instr p[9];
      for (int i=0;i<6;i++) p[i]=ld((uint16_t)(i*2));
      p[6]=st(2);  /* retire slot 2 */
      p[7]=ld(20);
      p[8]=st(0);  /* consume the first load, slot 1 */
      CHECK(agx3_slot_assign(p, 9, 6, r));
      CHECK(r[6].wait_mask==(1u << 1));
      CHECK(r[7].slot==2 && r[7].wait_mask==0);
      CHECK(r[8].wait_mask==0x1u);
   }

   /* Coefficient loads cannot wait, and Apple's own reissue slots whose
    * coefficients are still live: the seventh of seven wraps to slot 1 with
    * no wait. They still skip a slot another load's live result holds. */
   {
      struct agx3_slot_instr p[9];
      for (int i=0;i<7;i++) {
         p[i]=ld((uint16_t)(i*2));
         p[i].reissues_live_slot=true;
      }
      p[7]=st(0);
      CHECK(agx3_slot_assign(p, 8, 6, r));
      CHECK(r[6].slot==1 && r[6].wait_mask==0);

      p[0].reissues_live_slot=false;  /* slot 1 now holds a device load */
      CHECK(agx3_slot_assign(p, 8, 6, r));
      CHECK(r[6].slot==2 && r[6].wait_mask==0);

      for (int i=0;i<6;i++)
         p[i].reissues_live_slot=false;  /* no coefficient slot to reissue */
      CHECK(agx3_slot_assign(p, 8, 6, r));
      CHECK(r[6].slot==1 && r[6].wait_mask==0);
   }

   /* A reissuable instruction still takes an already-waited slot before
    * reissuing a live one, and reissues a live slot once all slots are live. */
   {
      struct agx3_slot_instr p[9];
      for (int i = 0; i < 6; ++i) {
         p[i] = ld((uint16_t)(i * 2));
         p[i].reissues_live_slot = true;
      }
      p[6] = st(2); /* retire slot 2 while slot 1 stays live */
      p[7] = ld(20);
      p[7].reissues_live_slot = true;
      p[8] = ld(22);
      p[8].reissues_live_slot = true;
      CHECK(agx3_slot_assign(p, 9, 6, r));
      CHECK(r[6].wait_mask == (1u << 1));
      CHECK(r[7].slot == 2 && r[7].wait_mask == 0);
      CHECK(r[8].slot == 3 && r[8].wait_mask == 0);
   }

   /* agx3c has 8 slots. */
   {
      struct agx3_slot_instr p[9];
      for (int i=0;i<8;i++) p[i]=ld((uint16_t)(i*2));
      p[8]=ld(100);
      CHECK(agx3_slot_assign(p, 9, 8, r));
      CHECK(r[7].slot==8 && r[8].slot==1);  /* wraps at 8 */
   }

   /* A corpus-witnessed preferred slot becomes the producer's real slot, so
    * dependency waits follow it rather than the round-robin cursor. */
   {
      struct agx3_slot_instr preferred = ld(2);
      preferred.preferred_slot = 6;
      struct agx3_slot_instr p[] = {ld(0), preferred, alu(4, 2, 2)};
      CHECK(agx3_slot_assign(p, 3, 6, r));
      CHECK(r[0].slot == 1 && r[0].wait_mask == 0);
      CHECK(r[1].slot == 6 && r[1].wait_mask == 0);
      CHECK(r[2].slot == 0 && r[2].wait_mask == 0x20);

      preferred.preferred_slot = 7;
      CHECK(!agx3_slot_assign(&preferred, 1, 6, r));
      preferred = st(0);
      preferred.preferred_slot = 1;
      CHECK(!agx3_slot_assign(&preferred, 1, 6, r));
   }

   /* Reusing an already-retired preferred slot starts a new dependency
    * generation. A stale register from the old generation must not consume
    * the new result's wait before its actual consumer. */
   {
      struct agx3_slot_instr preferred = ld(4);
      preferred.preferred_slot = 1;
      struct agx3_slot_instr p[] = {
         ld(0), alu(2, 0, 0), preferred, alu(0, 6, 8), st(4),
      };
      CHECK(agx3_slot_assign(p, 5, 6, r));
      CHECK(r[0].slot == 1 && r[1].wait_mask == 1);
      CHECK(r[2].slot == 1 && r[2].wait_mask == 0);
      CHECK(r[3].wait_mask == 0);
      CHECK(r[4].wait_mask == 1);
   }

   /* Round-robin wrap obeys the same generation rule after a prior wait. */
   {
      struct agx3_slot_instr p[] = {
         ld(0), alu(6, 0, 0), ld(2), ld(4), alu(0, 8, 10), st(4),
      };
      CHECK(agx3_slot_assign(p, 6, 2, r));
      CHECK(r[0].slot == 1 && r[1].wait_mask == 1);
      CHECK(r[2].slot == 2 && r[3].slot == 1);
      CHECK(r[4].wait_mask == 0);
      CHECK(r[5].wait_mask == 1);
   }

   /* A compound texture instruction stages an operand on slot 2, consumes
    * that generation internally, and publishes its actual result on slot 1.
    * A prior slot-2 load must be waited before staging reuses the slot; its
    * later GPR consumer is already satisfied, while the texture result still
    * carries the independent slot-1 dependency. */
   {
      struct agx3_slot_instr texture = ld(4);
      texture.preferred_slot = 1;
      texture.transient_slots = 1u << 1;
      struct agx3_slot_instr p[] = {
         ld(0), ld(2), texture, st(2), st(4),
      };
      CHECK(agx3_slot_assign(p, 5, 6, r));
      CHECK(r[0].slot == 1 && r[1].slot == 2);
      CHECK(r[2].slot == 1 && r[2].wait_mask == (1u << 1));
      CHECK(r[3].wait_mask == 0);
      CHECK(r[4].wait_mask == (1u << 0));

      texture.transient_slots = 1u << 0;
      CHECK(!agx3_slot_assign(&texture, 1, 6, r));
      texture.transient_slots = 1u << 6;
      CHECK(!agx3_slot_assign(&texture, 1, 6, r));
   }

   /* Static copy produces a TPS dependency token and a later rld consumes it.
    * The reload must wait the copy even though the two operations share no
    * architectural GPR. Its own GPR result then remains async as usual. */
   {
      struct agx3_slot_instr copy = {0}, rld = {0}, consume;
      copy.is_async = true;
      copy.nr_srcs = 2;
      copy.src[0] = 8;
      copy.src[1] = 9;
      copy.src[copy.nr_srcs++] = AGX3_SLOT_TPS_TOKEN_BASE + 18;
      copy.src[copy.nr_srcs++] = AGX3_SLOT_TPS_TOKEN_BASE + 19;
      copy.nr_dsts = 2;
      copy.dst[0] = AGX3_SLOT_TPS_TOKEN_BASE + 18;
      copy.dst[1] = AGX3_SLOT_TPS_TOKEN_BASE + 19;
      rld.is_async = true;
      rld.nr_srcs = 2;
      rld.src[0] = AGX3_SLOT_TPS_TOKEN_BASE + 18;
      rld.src[1] = AGX3_SLOT_TPS_TOKEN_BASE + 19;
      rld.nr_dsts = 2;
      rld.dst[0] = 12;
      rld.dst[1] = 13;
      rld.dst[rld.nr_dsts++] = AGX3_SLOT_TPS_TOKEN_BASE + 18;
      rld.dst[rld.nr_dsts++] = AGX3_SLOT_TPS_TOKEN_BASE + 19;
      consume = alu(16, 12, 12);
      struct agx3_slot_instr p[] = {copy, rld, consume};
      CHECK(agx3_slot_assign(p, 3, 6, r));
      CHECK(r[0].slot == 1 && r[0].wait_mask == 0);
      CHECK(r[1].slot == 2 && r[1].wait_mask == 1);
      CHECK(r[2].slot == 0 && r[2].wait_mask == 2);
   }

   /* A TPS access token is both read and rewritten, serializing WAW as well as
    * RAW hazards. The second copy waits the first before replacing its slot. */
   {
      struct agx3_slot_instr copy1 = {0}, copy2 = {0}, rld = {0};
      const uint16_t tps = AGX3_SLOT_TPS_TOKEN_BASE + 24;
      copy1.is_async = copy2.is_async = rld.is_async = true;
      copy1.nr_srcs = copy1.nr_dsts = 1;
      copy1.src[0] = copy1.dst[0] = tps;
      copy2 = copy1;
      rld.nr_srcs = rld.nr_dsts = 1;
      rld.src[0] = rld.dst[0] = tps;
      struct agx3_slot_instr p[] = {copy1, copy2, rld};
      CHECK(agx3_slot_assign(p, 3, 6, r));
      CHECK(r[0].slot == 1 && r[0].wait_mask == 0);
      CHECK(r[1].slot == 2 && r[1].wait_mask == 1);
      CHECK(r[2].slot == 3 && r[2].wait_mask == 2);
   }

   /* The top legal Mesa byte-immediate TPS token is modeled distinctly. */
   {
      struct agx3_slot_instr p[] = {
         ld(AGX3_SLOT_TPS_TOKEN_BASE + AGX3_SLOT_TPS_HALF_TOKENS - 1),
         st(AGX3_SLOT_TPS_TOKEN_BASE + AGX3_SLOT_TPS_HALF_TOKENS - 1),
      };
      CHECK(agx3_slot_assign(p, 2, 6, r));
      CHECK(r[1].wait_mask == 1);
      p[1].src[0] = AGX3_SLOT_NUM_TOKENS;
      CHECK(!agx3_slot_assign(p, 2, 6, r));
   }

   /* Gradient/minimum operands must not hide later descriptor dependencies.
    * The unrelated third load stays in flight across the sample. */
   {
      struct agx3_slot_instr sample = {0};
      sample.is_async = true;
      sample.nr_srcs = 28;
      sample.nr_dsts = 1;
      sample.dst[0] = 80;
      for (unsigned i = 0; i < sample.nr_srcs; ++i)
         sample.src[i] = 32 + i;
      sample.src[20] = 12;
      sample.src[24] = 16;
      struct agx3_slot_instr p[] = {ld(12), ld(16), ld(200), sample, st(200)};
      CHECK(agx3_slot_assign(p, 5, 6, r));
      CHECK(r[3].slot == 4 && r[3].wait_mask == 3);
      CHECK(r[4].wait_mask == 4);
      p[3].nr_srcs = AGX3_SLOT_MAX_REGS + 1;
      CHECK(!agx3_slot_assign(p, 5, 6, r));
   }

   /* Integrating store-payload tracking with generation-safe allocation:
    * a resultless store still owns its slot until its reader retires. */
   {
      struct agx3_slot_instr store = st(0);
      store.is_async = true;
      store.latches_srcs = true;
      struct agx3_slot_instr p[] = {store, ld(8), alu(0, 2, 4), st(8)};
      CHECK(agx3_slot_assign(p, 4, 1, r));
      CHECK(r[0].slot == 1 && r[0].wait_mask == 0);
      CHECK(r[1].slot == 1 && r[1].wait_mask == 1);
      CHECK(r[2].wait_mask == 0);
      CHECK(r[3].wait_mask == 1);
   }

   /* A texture's internal staging generation must also retire store readers,
    * even though no prior async instruction wrote a result on that slot. */
   {
      struct agx3_slot_instr store = st(0), texture = ld(8);
      store.is_async = true;
      store.latches_srcs = true;
      store.preferred_slot = 2;
      texture.preferred_slot = 1;
      texture.transient_slots = 1u << 1;
      struct agx3_slot_instr p[] = {store, texture, alu(0, 2, 4), st(8)};
      CHECK(agx3_slot_assign(p, 4, 6, r));
      CHECK(r[1].slot == 1 && r[1].wait_mask == 2);
      CHECK(r[2].wait_mask == 0);
      CHECK(r[3].wait_mask == 1);
   }

   /* Dependency ordering is unchanged when a token is renamed across the
    * GPR, static TPS, and special-token boundaries. In the second region the
    * highest token appears only as a payload source, never a destination. */
   {
      const uint16_t tokens[] = {
         0, AGX3_SLOT_GPR_TOKENS - 1, AGX3_SLOT_TPS_TOKEN_BASE,
         AGX3_SLOT_TEX_COORD_TOKEN - 1, AGX3_SLOT_TEX_COORD_TOKEN,
         AGX3_SLOT_DYNAMIC_TPS_TOKEN,
      };
      for (unsigned t = 0; t < sizeof(tokens) / sizeof(tokens[0]); ++t) {
         const uint16_t token = tokens[t];
         struct agx3_slot_instr producer = ld(token), store = stdev(token);
         producer.preferred_slot = 1;
         store.preferred_slot = 2;
         struct agx3_slot_instr p[] = {producer, store, mov(token)};
         CHECK(agx3_slot_assign(p, 3, 6, r));
         CHECK(r[0].slot == 1 && r[0].wait_mask == 0);
         CHECK(r[1].slot == 2 && r[1].wait_mask == 1);
         CHECK(r[2].slot == 0 && r[2].wait_mask == 2);

         struct agx3_slot_instr texture = ld(7);
         texture.preferred_slot = 1;
         texture.transient_slots = 1u << 1;
         struct agx3_slot_instr q[] = {store, texture, mov(token), st(7)};
         CHECK(agx3_slot_assign(q, 4, 6, r));
         CHECK(r[0].slot == 2 && r[0].wait_mask == 0);
         CHECK(r[1].slot == 1 && r[1].wait_mask == 2);
         CHECK(r[2].slot == 0 && r[2].wait_mask == 0);
         CHECK(r[3].slot == 0 && r[3].wait_mask == 1);
      }
   }

   /* Rejecting a malformed later instruction preserves the completed output
    * prefix, even when the bounds prepass encounters it before assignment. */
   {
      struct agx3_slot_instr p[] = {ld(0), st(AGX3_SLOT_NUM_TOKENS)};
      memset(r, 0, sizeof(r));
      r[1].slot = 8;
      r[1].wait_mask = 0xff;
      CHECK(!agx3_slot_assign(p, 2, 6, r));
      CHECK(r[0].slot == 1 && r[0].wait_mask == 0);
      CHECK(r[1].slot == 8 && r[1].wait_mask == 0xff);
      p[1].nr_srcs = AGX3_SLOT_MAX_REGS + 1;
      CHECK(!agx3_slot_assign(p, 2, 6, r));
      CHECK(r[0].slot == 1 && r[0].wait_mask == 0);
      CHECK(r[1].slot == 8 && r[1].wait_mask == 0xff);
      CHECK(agx3_slot_assign(p, 0, 6, r));
   }

   /* Native constant-program inputs arrive from an external slot-6 COPY.
    * Its generation must be consumed once, with ordinary result tracking
    * continuing independently for the RLD destinations. */
   {
      const struct agx3_slot_entry entry = {
         .token = AGX3_SLOT_CPROG_INPUT_TOKEN, .slot = 6,
      };
      struct agx3_slot_instr first = ld(0), second = ld(2);
      first.preferred_slot = 2;
      second.preferred_slot = 3;
      first.nr_srcs = second.nr_srcs = 1;
      first.src[0] = second.src[0] = entry.token;
      struct agx3_slot_instr p[] = {first, second, alu(4, 0, 2)};
      CHECK(agx3_slot_assign_with_entry(p, 3, 6, &entry, 1, r));
      CHECK(r[0].slot == 2 && r[0].wait_mask == 0x20);
      CHECK(r[1].slot == 3 && r[1].wait_mask == 0);
      CHECK(r[2].wait_mask == 0x6);

      /* Reusing slot 6 before the first import must retire the loader, even
       * though that instruction does not itself read the imported value. */
      struct agx3_slot_instr unrelated = ld(16);
      unrelated.preferred_slot = 6;
      struct agx3_slot_instr q[] = {unrelated, first, st(16)};
      CHECK(agx3_slot_assign_with_entry(q, 3, 6, &entry, 1, r));
      CHECK(r[0].slot == 6 && r[0].wait_mask == 0x20);
      CHECK(r[1].slot == 2 && r[1].wait_mask == 0);
      CHECK(r[2].wait_mask == 0x20);

      struct agx3_slot_instr drain = {.drain = true};
      struct agx3_slot_instr d[] = {drain, first};
      CHECK(agx3_slot_assign_with_entry(d, 2, 6, &entry, 1, r));
      CHECK(r[0].wait_mask == 0x20 && r[1].wait_mask == 0);

      struct agx3_slot_instr w[] = {mov(entry.token), first};
      CHECK(agx3_slot_assign_with_entry(w, 2, 6, &entry, 1, r));
      CHECK(r[0].wait_mask == 0x20 && r[1].wait_mask == 0);

      struct agx3_slot_instr transient = ld(8);
      transient.preferred_slot = 1;
      transient.transient_slots = 0x20;
      struct agx3_slot_instr t[] = {transient, first};
      CHECK(agx3_slot_assign_with_entry(t, 2, 6, &entry, 1, r));
      CHECK(r[0].wait_mask == 0x20 && r[1].wait_mask == 0);
   }

   /* Entry-only tokens contribute to the region's token bound. Multiple
    * results may belong to one initial generation; duplicate token bindings
    * and malformed entry descriptors are refused before writing outputs. */
   {
      struct agx3_slot_entry entry[] = {{AGX3_SLOT_NUM_TOKENS - 1, 1}, {7, 1}};
      struct agx3_slot_instr p[] = {ld(0), st(7)};
      CHECK(agx3_slot_assign_with_entry(p, 2, 1, entry, 2, r));
      CHECK(r[0].slot == 1 && r[0].wait_mask == 1);
      CHECK(r[1].wait_mask == 0);
      p[0].reissues_live_slot = true;
      CHECK(agx3_slot_assign_with_entry(p, 2, 1, entry, 2, r));
      CHECK(r[0].slot == 1 && r[0].wait_mask == 1);
      CHECK(r[1].wait_mask == 0);
      memset(r, 0xa5, sizeof(r));
      struct agx3_slot_result before[sizeof(r) / sizeof(r[0])];
      memcpy(before, r, sizeof(r));
      entry[1].token = entry[0].token;
      CHECK(!agx3_slot_assign_with_entry(p, 2, 6, entry, 2, r));
      CHECK(memcmp(before, r, sizeof(r)) == 0);
      entry[1].token = 7;
      entry[0].slot = 7;
      CHECK(!agx3_slot_assign_with_entry(p, 2, 6, entry, 2, r));
      entry[0].slot = 0;
      CHECK(!agx3_slot_assign_with_entry(p, 2, 6, entry, 2, r));
      entry[0].slot = 1;
      entry[0].token = AGX3_SLOT_NUM_TOKENS;
      CHECK(!agx3_slot_assign_with_entry(p, 2, 6, entry, 2, r));
      CHECK(!agx3_slot_assign_with_entry(p, 2, 6, NULL, 1, r));
      CHECK(memcmp(before, r, sizeof(r)) == 0);
   }

   /* A completed producer event can supply the consumer's physical slot.
    * Subsequent users of the original value must not wait that slot's new
    * generation; fixed-slot protocols retain their required assignment.
    */
   {
      struct agx3_slot_instr consumer = ld(8);
      consumer.nr_srcs = 1;
      consumer.src[0] = 0;
      consumer.coalesce_src = 1;
      struct agx3_slot_instr p[] = {ld(0), consumer, st(0), st(8)};
      CHECK(agx3_slot_assign(p, 4, 6, r));
      CHECK(r[0].slot == 1 && r[1].slot == 1 && r[1].wait_mask == 1);
      CHECK(r[2].wait_mask == 0 && r[3].wait_mask == 1);
      p[1].preferred_slot = 4;
      CHECK(agx3_slot_assign(p, 4, 6, r));
      CHECK(r[1].slot == 4 && r[1].wait_mask == 1 && r[3].wait_mask == 8);
      p[1].preferred_slot = 0;
      p[1].transient_slots = 1;
      CHECK(agx3_slot_assign(p, 4, 6, r));
      CHECK(r[1].slot != 1 && r[1].wait_mask == 1);
      p[1].coalesce_src = 2;
      CHECK(!agx3_slot_assign(p, 4, 6, r));
      p[1].coalesce_src = 1;
      p[1].is_async = false;
      CHECK(!agx3_slot_assign(p, 4, 6, r));
   }

   /* Control-flow drains: emission retires the load before the branch, so
    * assignment gives its consumer after the branch no wait, and the drain
    * counts. */
   {
      struct agx3_slot_instr branch = st(0);
      branch.cf = AGX3_SLOT_CF_DRAIN;
      struct agx3_slot_instr p[] = { ld(10), branch, alu(12, 10, 10) };
      bool drained[3] = {0};
      CHECK(agx3_slot_assign(p, 3, 6, r));
      CHECK(r[2].wait_mask == 0);
      agx3_slot_retire_at_cf(p, 3, 0, r, drained);
      CHECK(r[2].wait_mask == 0);
      CHECK(!drained[0] && drained[1] && !drained[2]);
   }
   /* A branch that reads the load waits for it itself: not a drain. */
   {
      struct agx3_slot_instr branch = st(10);
      branch.cf = AGX3_SLOT_CF_DRAIN;
      struct agx3_slot_instr p[] = { ld(10), branch, alu(12, 10, 10) };
      bool drained[3] = {0};
      CHECK(agx3_slot_assign(p, 3, 6, r));
      CHECK(r[1].wait_mask == (1u << (r[0].slot - 1)));
      agx3_slot_retire_at_cf(p, 3, 0, r, drained);
      CHECK(!drained[1] && r[2].wait_mask == 0);
   }
   /* A non-draining boundary (a forward if) keeps the load in flight: its
    * consumer inside the arm still waits for it. */
   {
      struct agx3_slot_instr branch = st(0);
      struct agx3_slot_instr p[] = { ld(10), branch, alu(12, 10, 10) };
      bool drained[3] = {0};
      CHECK(agx3_slot_assign(p, 3, 6, r));
      agx3_slot_retire_at_cf(p, 3, 0, r, drained);
      CHECK(r[2].wait_mask == (1u << (r[0].slot - 1)));
      CHECK(!drained[1]);
   }
   /* After a drain every slot is free: the next load takes the next slot in
    * turn without waiting, keeps its own wait, and the pre-drain value
    * needs none. */
   {
      struct agx3_slot_instr branch = st(0);
      branch.cf = AGX3_SLOT_CF_DRAIN;
      struct agx3_slot_instr p[] = { ld(10), branch, ld(12), alu(14, 12, 10) };
      CHECK(agx3_slot_assign(p, 4, 6, r));
      CHECK(r[2].wait_mask == 0 && r[2].slot != 0);
      agx3_slot_retire_at_cf(p, 4, 0, r, NULL);
      CHECK(r[3].wait_mask == (1u << (r[2].slot - 1)));
   }
   /* Six loads fill every AGX3B slot; after a drain, six more need no
    * forced retire of an earlier generation. */
   {
      struct agx3_slot_instr p[14];
      for (unsigned i = 0; i < 6; ++i)
         p[i] = ld(20 + 2 * i);
      p[6] = st(0);
      p[6].cf = AGX3_SLOT_CF_DRAIN;
      for (unsigned i = 0; i < 6; ++i)
         p[7 + i] = ld(40 + 2 * i);
      p[13] = alu(60, 40, 50);
      CHECK(agx3_slot_assign(p, 14, 6, r));
      for (unsigned i = 7; i < 13; ++i)
         CHECK(r[i].wait_mask == 0);
      CHECK(r[13].wait_mask == ((1u << (r[7].slot - 1)) | (1u << (r[12].slot - 1))));
   }

   /* Arms (enum agx3_slot_cf): a drain at the end of an arm runs under
    * the arm's exec mask, which may be empty, so it retires only what the
    * arm issued. A load from before the if is still waited for by its
    * consumers after the arm and in an else arm. */
   {
      struct agx3_slot_instr open = st(0), close = st(0);
      open.cf_push = 1;
      open.cf_arm = true;
      close.cf = AGX3_SLOT_CF_DRAIN;
      close.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         ld(10), open, alu(12, 10, 10), close, alu(14, 10, 10),
      };
      CHECK(agx3_slot_assign(p, 5, 6, r));
      CHECK(r[2].wait_mask == (1u << (r[0].slot - 1)));
      CHECK(r[4].wait_mask == (1u << (r[0].slot - 1)));
   }
   {
      struct agx3_slot_instr open = st(0), other = st(0), close = st(0);
      open.cf_push = 1;
      open.cf_arm = true;
      other.cf = AGX3_SLOT_CF_DRAIN;
      other.cf_else = true;
      close.cf = AGX3_SLOT_CF_DRAIN;
      close.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         ld(10), open, alu(12, 10, 10), other, alu(14, 10, 10), close,
      };
      CHECK(agx3_slot_assign(p, 6, 6, r));
      CHECK(r[4].wait_mask == (1u << (r[0].slot - 1)));
   }
   /* What the arm issued is retired by the drain at its end. */
   {
      struct agx3_slot_instr open = st(0), close = st(0);
      open.cf_push = 1;
      open.cf_arm = true;
      close.cf = AGX3_SLOT_CF_DRAIN;
      close.cf_pop = 1;
      struct agx3_slot_instr p[] = { open, ld(10), close, alu(12, 10, 10) };
      CHECK(agx3_slot_assign(p, 4, 6, r));
      CHECK(r[3].wait_mask == 0);
   }
   /* A non-arm level (a loop push, or an if that drains) drains as before. */
   {
      struct agx3_slot_instr open = st(0), close = st(0);
      open.cf = AGX3_SLOT_CF_DRAIN;
      open.cf_push = 1;
      close.cf = AGX3_SLOT_CF_DRAIN;
      close.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         ld(10), open, alu(12, 10, 10), close, alu(14, 10, 10),
      };
      CHECK(agx3_slot_assign(p, 5, 6, r));
      CHECK(r[2].wait_mask == 0 && r[4].wait_mask == 0);
   }
   /* Joined arms (agx3_slot_core.h): the else and pop do not drain. A load
    * in the only arm is still in flight after the pop, where its consumer
    * waits for it. */
   {
      struct agx3_slot_instr open = st(0), close = st(0);
      open.cf_push = 1;
      open.cf_arm = true;
      close.cf_pop = 1;
      struct agx3_slot_instr p[] = { open, ld(10), close, alu(12, 10, 10) };
      CHECK(agx3_slot_assign(p, 4, 6, r));
      CHECK(r[3].wait_mask == (1u << (r[1].slot - 1)));
   }
   /* Both arms load the same register. The second arm waits out the first
    * arm's generation before redefining it (a write-after-write in the joined
    * state), and after the pop the register may hold either arm's result, so
    * its consumer waits for both slots: either arm may have had no lanes, and
    * then its waits did not run. */
   {
      struct agx3_slot_instr open = st(0), other = st(0), close = st(0);
      open.cf_push = 1;
      open.cf_arm = true;
      other.cf_else = true;
      close.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         open, ld(10), other, ld(10), close, alu(12, 10, 10),
      };
      CHECK(agx3_slot_assign(p, 6, 6, r));
      CHECK(r[3].wait_mask == (1u << (r[1].slot - 1)));
      CHECK(r[5].wait_mask ==
            ((1u << (r[1].slot - 1)) | (1u << (r[3].slot - 1))));
   }
   /* The second arm starts with the first arm's generation in flight (a
    * divergent group runs both), so it takes another slot rather than reuse
    * the live one, and the consumer after the pop waits for both. */
   {
      struct agx3_slot_instr open = st(0), other = st(0), close = st(0);
      open.cf_push = 1;
      open.cf_arm = true;
      other.cf_else = true;
      close.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         open, ld(10), other, ld(20), close, alu(30, 10, 20),
      };
      CHECK(agx3_slot_assign(p, 6, 6, r));
      CHECK(r[3].wait_mask == 0);
      CHECK(r[3].slot != r[1].slot);
      CHECK(r[5].wait_mask ==
            ((1u << (r[1].slot - 1)) | (1u << (r[3].slot - 1))));
   }
   /* A synchronous write in one arm and a load in the other: the register
    * may still be the load's after the pop. */
   {
      struct agx3_slot_instr open = st(0), other = st(0), close = st(0);
      open.cf_push = 1;
      open.cf_arm = true;
      other.cf_else = true;
      close.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         open, ld(10), other, alu(10, 40, 40), close, alu(12, 10, 10),
      };
      CHECK(agx3_slot_assign(p, 6, 6, r));
      CHECK(r[5].wait_mask == (1u << (r[1].slot - 1)));
   }
   /* A load from before the if that one arm consumed is still named after
    * the pop: the other arm, or an arm with no lanes, did not retire it. */
   {
      struct agx3_slot_instr open = st(0), other = st(0), close = st(0);
      open.cf_push = 1;
      open.cf_arm = true;
      other.cf_else = true;
      close.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         ld(10), open, alu(12, 10, 10), other, alu(14, 40, 40), close,
         alu(16, 10, 10),
      };
      CHECK(agx3_slot_assign(p, 7, 6, r));
      CHECK(r[2].wait_mask == (1u << (r[0].slot - 1)));
      CHECK(r[6].wait_mask == (1u << (r[0].slot - 1)));
   }
   /* A drain after the join retires every slot either arm left. */
   {
      struct agx3_slot_instr open = st(0), other = st(0), close = st(0),
                             fence = st(0);
      open.cf_push = 1;
      open.cf_arm = true;
      other.cf_else = true;
      close.cf_pop = 1;
      fence.cf = AGX3_SLOT_CF_DRAIN;
      struct agx3_slot_instr p[] = {
         open, ld(10), other, ld(20), close, fence, alu(30, 10, 20),
      };
      CHECK(agx3_slot_assign(p, 7, 6, r));
      CHECK(r[6].wait_mask == 0);
   }
   /* An if/else inside a loop body joins the same way; the loop's continue
    * test and backedge still drain. */
   {
      struct agx3_slot_instr push = st(0), open = st(0), other = st(0),
                             close = st(0), test = st(0), back = st(0),
                             pop = st(0);
      push.cf_push = 1;
      open.cf_push = 1;
      open.cf_arm = true;
      other.cf_else = true;
      close.cf_pop = 1;
      test.cf = back.cf = pop.cf = AGX3_SLOT_CF_DRAIN;
      pop.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         push, open, ld(10), other, alu(10, 40, 40), close, alu(12, 10, 10),
         test, back, pop, alu(14, 10, 10),
      };
      CHECK(agx3_slot_assign(p, 11, 6, r));
      CHECK(r[6].wait_mask == (1u << (r[2].slot - 1)));
      CHECK(r[10].wait_mask == 0);
   }

   /* A loop body is such a level (agx3_slot_cf_kind()): entry push, body,
    * draining continue test and backedge, pop. A load from before the loop
    * stays in flight into the body, where its consumer waits for it, and is
    * outstanding again after the pop, where a consumer waits for it too:
    * every iteration but the first starts after the backedge drain, so only
    * the entry state can still hold it. */
   {
      struct agx3_slot_instr push = st(0), test = st(0), back = st(0),
                             pop = st(0);
      push.cf_push = 1;
      push.cf_arm = true;
      test.cf = back.cf = pop.cf = AGX3_SLOT_CF_DRAIN;
      pop.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         ld(10), push, alu(12, 10, 10), ld(16), test, back, pop,
         alu(14, 10, 16),
      };
      CHECK(agx3_slot_assign(p, 8, 6, r));
      CHECK(r[2].wait_mask == (1u << (r[0].slot - 1)));
      /* The body's own load is retired by its drains, not after the loop. */
      CHECK(r[7].wait_mask == (1u << (r[0].slot - 1)));
      /* While the entry load is live, the body does not reuse its slot. */
      CHECK(r[3].slot != 0);
   }
   {
      struct agx3_slot_instr push = st(0), test = st(0), back = st(0),
                             pop = st(0);
      push.cf_push = 1;
      push.cf_arm = true;
      test.cf = back.cf = pop.cf = AGX3_SLOT_CF_DRAIN;
      pop.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         ld(10), push, ld(16), alu(18, 16, 16), test, back, pop,
         alu(14, 10, 10),
      };
      CHECK(agx3_slot_assign(p, 8, 6, r));
      CHECK(r[2].slot != r[0].slot);
      CHECK(r[7].wait_mask == (1u << (r[0].slot - 1)));
   }

   /* Breaks that join (agx3_slot_core.h): a break does not drain. The lanes
    * that stay wait for the body's load where they consume it; the lanes that
    * broke never did, so the consumer after the loop waits for it too. */
   {
      struct agx3_slot_instr push = st(0), brk = st(0), back = st(0),
                             pop = st(0);
      push.cf_push = 1;
      push.cf_loop = true;
      brk.cf_break = 1;
      back.cf = AGX3_SLOT_CF_DRAIN;
      pop.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         push, ld(10), brk, alu(12, 10, 10), back, pop, alu(14, 10, 10),
      };
      bool drained[7] = {false};
      CHECK(agx3_slot_assign(p, 7, 6, r));
      CHECK(r[2].wait_mask == 0);
      CHECK(r[3].wait_mask == (1u << (r[1].slot - 1)));
      CHECK(r[6].wait_mask == (1u << (r[1].slot - 1)));
      agx3_slot_retire_at_cf(p, 7, 0, r, drained);
      for (unsigned i = 0; i < 7; ++i)
         CHECK(!drained[i]);
   }
   /* Two breaks leave one register written by different slots: the body's
    * second load waits out the first (write-after-write for the lanes that
    * stay), and after the loop the register may hold either, since lanes
    * left at both breaks. */
   {
      struct agx3_slot_instr push = st(0), brk = st(0), back = st(0),
                             pop = st(0);
      push.cf_push = 1;
      push.cf_loop = true;
      brk.cf_break = 1;
      back.cf = AGX3_SLOT_CF_DRAIN;
      pop.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         push, ld(10), brk, ld(10), brk, back, pop, alu(12, 10, 10),
      };
      CHECK(agx3_slot_assign(p, 8, 6, r));
      CHECK(r[3].wait_mask == (1u << (r[1].slot - 1)));
      CHECK(r[7].wait_mask ==
            ((1u << (r[1].slot - 1)) | (1u << (r[3].slot - 1))));
   }
   /* A break out of an if inside the loop (nest 2) joins the loop's pop, not
    * the if's. */
   {
      struct agx3_slot_instr push = st(0), open = st(0), brk = st(0),
                             close = st(0), back = st(0), pop = st(0);
      push.cf_push = 1;
      push.cf_loop = true;
      open.cf_push = 1;
      open.cf_arm = true;
      brk.cf_break = 2;
      close.cf_pop = 1;
      back.cf = AGX3_SLOT_CF_DRAIN;
      pop.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         push, ld(10), open, brk, close, back, pop, alu(12, 10, 10),
      };
      CHECK(agx3_slot_assign(p, 8, 6, r));
      CHECK(r[7].wait_mask == (1u << (r[1].slot - 1)));
   }
   /* Nested loops: the inner break takes the outer body's load and its own
    * out of the inner loop, whose consumer waits for both; once it has, the
    * outer break leaves nothing in flight. */
   {
      struct agx3_slot_instr push = st(0), brk = st(0), back = st(0),
                             pop = st(0);
      push.cf_push = 1;
      push.cf_loop = true;
      brk.cf_break = 1;
      back.cf = AGX3_SLOT_CF_DRAIN;
      pop.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         push, ld(20), push, ld(30), brk, back, pop, alu(22, 20, 30), brk,
         back, pop, alu(24, 22, 20),
      };
      CHECK(agx3_slot_assign(p, 12, 6, r));
      CHECK(r[7].wait_mask ==
            ((1u << (r[1].slot - 1)) | (1u << (r[3].slot - 1))));
      CHECK(r[11].wait_mask == 0);
   }
   /* A break that drains (its compare waits) joins nothing still in flight. */
   {
      struct agx3_slot_instr push = st(0), brk = st(0), back = st(0),
                             pop = st(0);
      push.cf_push = 1;
      push.cf_loop = true;
      brk.cf_break = 1;
      brk.cf = back.cf = AGX3_SLOT_CF_DRAIN;
      pop.cf_pop = 1;
      struct agx3_slot_instr p[] = {
         push, ld(10), brk, back, pop, alu(12, 10, 10),
      };
      CHECK(agx3_slot_assign(p, 6, 6, r));
      CHECK(r[5].wait_mask == 0);
   }
   /* A break must name an open loop level. */
   {
      struct agx3_slot_instr push = st(0), loop = st(0), brk = st(0),
                             pop = st(0);
      push.cf_push = 1;
      loop.cf_push = 1;
      loop.cf_loop = true;
      brk.cf_break = 1;
      pop.cf_pop = 1;
      struct agx3_slot_instr p[] = { push, brk, pop };
      CHECK(!agx3_slot_assign(p, 3, 6, r));
      brk.cf_break = 2;
      struct agx3_slot_instr q[] = { loop, brk, pop };
      CHECK(!agx3_slot_assign(q, 3, 6, r));
   }

   /* An arm closed together with another level, or an unbalanced pop, is
    * refused: the packer then drains everywhere. */
   {
      struct agx3_slot_instr open = st(0), close = st(0);
      open.cf_push = 1;
      open.cf_arm = true;
      close.cf = AGX3_SLOT_CF_DRAIN;
      close.cf_pop = 2;
      struct agx3_slot_instr p[] = { open, open, close };
      CHECK(!agx3_slot_assign(p, 3, 6, r));
      struct agx3_slot_instr q[] = { close };
      CHECK(!agx3_slot_assign(q, 1, 6, r));
   }

   /* Malformed inputs. */
   CHECK(!agx3_slot_assign(NULL, 1, 6, r));
   { struct agx3_slot_instr p=ld(0); CHECK(!agx3_slot_assign(&p,1,0,r)); CHECK(!agx3_slot_assign(&p,1,9,r)); }

   if (failures){ fprintf(stderr,"agx3_slot_core: %u/%u failed\n",failures,checks); return 1; }
   printf("agx3_slot_core: all %u checks passed\n", checks);
   return 0;
}

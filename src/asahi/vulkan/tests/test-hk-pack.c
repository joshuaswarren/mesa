/* SPDX-License-Identifier: MIT
 * Submit packing (hk_pack.c) with the real hk_sync recorder and resolver;
 * no GPU, DRM or Vulkan device.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/ralloc.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_compute_pair.h"
#include "hk_compute_pair_interval.h"
#include "hk_pack.h"
#include "hk_queue.h"

static unsigned tests, failures;
static void *test_context;

static void *
test_alloc(size_t size)
{
   void *ptr = rzalloc_size(test_context, size);
   if (!ptr) {
      fputs("Failed to allocate packing test fixture\n", stderr);
      exit(EXIT_FAILURE);
   }
   return ptr;
}

#define T(name, cond)                                                          \
   do {                                                                        \
      tests++;                                                                 \
      if (!(cond)) {                                                           \
         failures++;                                                           \
         printf("FAIL %s line%d\n", name, __LINE__);                           \
      } else                                                                   \
         printf("PASS %s\n", name);                                            \
   } while (0)

static const uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE] = {0xa5, 0x5a,
                                                                  0x0f, 0xf0};

/* The chip hk_pack_run_member() checks streams for, and add_plain() builds
 * them for. */
static enum agx_chip chip = AGX_CHIP_G16G;

static struct hk_device *
dev_with(enum agx_chip c, uint64_t features)
{
   struct hk_device *dev = test_alloc(sizeof(*dev));
   dev->dev.chip = c;
   dev->dev.params.features = features;
   return dev;
}

static struct hk_device *
device(void)
{
   return dev_with(chip, agx_chip_g17_m5(chip)
                            ? DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION
                            : 0);
}

static struct hk_cs *
first_cs(struct hk_cmd_buffer *c)
{
   return list_first_entry(&c->control_streams, struct hk_cs, node);
}

static struct hk_queue *
new_queue(bool exact)
{
   struct hk_queue *q = test_alloc(sizeof(struct hk_queue));
   q->sync.exact_prior = exact;
   return q;
}

static struct hk_cmd_buffer *
new_cmd(void)
{
   struct hk_cmd_buffer *c = test_alloc(sizeof(*c));
   c->vk.base.device = test_alloc(sizeof(struct hk_device));
   list_inithead(&c->control_streams);
   util_dynarray_init(&c->sync_pass_records, NULL);
   return c;
}

static struct hk_cs *
add(struct hk_cmd_buffer *c, bool compute)
{
   struct hk_cs *s = test_alloc(sizeof(*s));
   s->type = compute ? HK_CS_CDM : HK_CS_VDM;
   list_addtail(&s->node, &c->control_streams);
   hk_sync_stream_created(c, s, &s, compute);
   return s;
}

/* A compute stream that packing accepts on `chip`: `dispatches` direct
 * launches, then the ordering barrier (G16G) or the strong memory-scoped
 * flush (M5) before the terminate. */
static struct hk_cs *
add_plain(struct hk_cmd_buffer *c, unsigned dispatches)
{
   struct hk_cs *s = add(c, true);
   const size_t body = 64;
   uint8_t *bytes = test_alloc(body + sizeof(terminate) + HK_CS_SPACE_PADDING);
   memcpy(bytes + body, terminate, sizeof(terminate));
   s->start = bytes;
   s->current = bytes + body + sizeof(terminate);
   s->end = (uint8_t *)s->current + HK_CS_SPACE_PADDING;
   if (agx_chip_g17_m5(chip)) {
      s->visible_completion = true;
      s->strong_flush_end = s->memory_flush_end = bytes + body;
   } else {
      s->weak_barrier_end = bytes + body;
   }
   s->stats.cmds = s->stats.calls = dispatches;
   s->pack_dispatches = dispatches;
   s->pack_threads = 64ull * dispatches;
   s->pack_static_work = 64ull * 16 * dispatches;
   util_dynarray_init(&s->imm_writes, NULL);
   util_dynarray_init(&s->compute_resources, NULL);
   return s;
}

static struct hk_cmd_buffer *
plain(unsigned dispatches)
{
   struct hk_cmd_buffer *c = new_cmd();
   add_plain(c, dispatches);
   return c;
}

static bool
resolve(struct hk_queue *q, struct hk_cmd_buffer **cmds, unsigned count)
{
   return hk_sync_resolve_submit(q, cmds, count, false, false, 0, NULL);
}

/* Builds the packet of members cmds[0..count) the way hk_pack_runs() does,
 * without copying bytes. */
static struct hk_cmd_buffer *
pack(struct hk_cmd_buffer **cmds, unsigned count, struct hk_cs **out)
{
   struct hk_cmd_buffer *packet = new_cmd();
   struct hk_cs *cs = test_alloc(sizeof(*cs));
   cs->type = HK_CS_CDM;
   cs->cmd = packet;
   cs->sync.seq = 1;
   list_addtail(&cs->node, &packet->control_streams);
   for (unsigned m = 0; m < count; ++m)
      cs->sync.kind |= list_first_entry(&cmds[m]->control_streams,
                                        struct hk_cs, node)->sync.kind;
   uint8_t preceding = 0;
   for (unsigned m = 0; m < count; ++m) {
      const struct hk_cs *item =
         list_first_entry(&cmds[m]->control_streams, struct hk_cs, node);
      hk_pack_fold_member(packet, cs, cmds[m], item, m + 1 == count,
                          &preceding);
   }
   hk_pack_finish_sync(packet, cs);
   *out = cs;
   return packet;
}

static const struct hk_cs *
member(const struct hk_cmd_buffer *c, bool atomics)
{
   return hk_pack_run_member(device(), c, terminate, atomics);
}

static void
test_rejections(void)
{
   struct hk_cmd_buffer *c = plain(2);
   struct hk_cs *s = list_first_entry(&c->control_streams, struct hk_cs, node);
   T("plain compute stream is a member", member(c, false) == s);

#define REJECT(name, field, value)                                             \
   do {                                                                        \
      __typeof__(field) saved = field;                                         \
      field = value;                                                           \
      T(name, !member(c, true));                                               \
      field = saved;                                                           \
      T(name " (restored)", member(c, true) == s);                             \
   } while (0)

   REJECT("events", c->state.uses_events, true);
   REJECT("pending start stamp", c->sync.pending_start_ts.handle, 7);
   REJECT("start stamp", s->timestamp.start.handle, 7);
   REJECT("end stamp", s->timestamp.end.handle, 7);
   REJECT("immediate writes", s->imm_writes.size, 8);
   REJECT("indirect launch", s->sib_indirect, true);
   REJECT("internal launch", s->stats.cmds, 3);
   REJECT("uncounted launch", s->pack_dispatches, 1);
   REJECT("linked stream", s->stream_linked, true);
   REJECT("stream call", s->calls_stream, true);
   REJECT("query access", s->query_access, true);
   REJECT("pending query reset", s->query_reset_pending, true);
   REJECT("timestamp query command", s->g17p_timestamp_query, true);
   REJECT("native ray query", s->gen1_native_rq, true);
   REJECT("acceleration structure build", s->as_build, true);
   REJECT("render stream", s->type, HK_CS_VDM);
   REJECT("no tail barrier", s->weak_barrier_end, NULL);
   REJECT("wait on a stream of its own", s->sync.dep[HK_SYNC_CDM].cut, s);
   REJECT("tail cut on the render subqueue",
          c->sync.req[0][HK_SYNC_VDM].cut, s);
#undef REJECT

   s->uses_global_atomics = true;
   T("global atomics need atomics allowed", !member(c, false));
   T("global atomics with atomics allowed", member(c, true) == s);
   s->uses_global_atomics = false;

   ((uint8_t *)s->current)[-1] ^= 0xff;
   T("wrong terminate", !member(c, true));
   ((uint8_t *)s->current)[-1] ^= 0xff;

   add_plain(c, 1);
   T("two streams", !member(c, true));

   struct hk_cmd_buffer *own_cut = plain(1);
   hk_sync_barrier(own_cut, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
   T("compute tail cut naming its own stream is accepted",
     member(own_cut, false) != NULL);
}

/* M5: a member's stream ends with a strong flush, with memory scope (the
 * visible-completion tail vkEndCommandBuffer gives a last launch) or without
 * (an application barrier, or the deferred clears' flush, at the end on a
 * device whose memory has no second address). A weak barrier publishes
 * nothing and ends no member; nor does a strong flush before the end. */
static void
test_m5_tails(void)
{
   struct hk_cmd_buffer *c = plain(2);
   struct hk_cs *s = list_first_entry(&c->control_streams, struct hk_cs, node);
   struct hk_device *m5 =
      dev_with(AGX_CHIP_G17S, DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION);
   uint32_t *end = s->weak_barrier_end;

   T("M5: weak barrier tail is no member",
     !hk_pack_run_member(m5, c, terminate, false));

   s->weak_barrier_end = NULL;
   s->strong_flush_end = s->memory_flush_end = end;
   s->visible_completion = true;
   T("M5: memory-scoped visible-completion tail is a member",
     hk_pack_run_member(m5, c, terminate, false) == s);
   T("M5: no member without the visible-completion feature",
     !hk_pack_run_member(dev_with(AGX_CHIP_G17S, 0), c, terminate, false));

   s->memory_flush_end = NULL;
   s->visible_completion = false;
   T("M5: strong flush without memory scope is a member",
     hk_pack_run_member(m5, c, terminate, false) == s);
   T("M5: nor a plain strong flush without the feature",
     !hk_pack_run_member(dev_with(AGX_CHIP_G17S, 0), c, terminate, false));

   s->strong_flush_end = end - 1;
   T("M5: strong flush before the end is no member",
     !hk_pack_run_member(m5, c, terminate, false));
}

/* Plans a submission to queue `q` as hk_pack_runs() does. */
static void
plan_queue(const struct hk_queue *q, struct hk_cmd_buffer **cmds, unsigned n,
           bool atomics, struct hk_pack_run *runs, unsigned *count)
{
   struct hk_pack_member members[16];
   hk_pack_describe_submit(members, device(), &q->sync, cmds, n, terminate,
                           atomics);
   const struct hk_pack_limits limits = {.max_attachments = 16,
                                         .overhead = 64};
   unsigned packed;
   size_t total;
   *count = hk_pack_plan(members, n, &limits, runs, &packed, &total);
}

/* As hk_pack_runs() plans a submission to a legacy queue that ran `ever` on
 * its render subqueue. */
static void
plan_on(struct hk_cmd_buffer **cmds, unsigned n, uint8_t ever, bool atomics,
        struct hk_pack_run *runs, unsigned *count)
{
   struct hk_queue *q = new_queue(false);
   q->sync.ever[HK_SYNC_VDM] = ever;
   plan_queue(q, cmds, n, atomics, runs, count);
}

static void
plan(struct hk_cmd_buffer **cmds, unsigned n, struct hk_pack_run *runs,
     unsigned *count)
{
   plan_on(cmds, n, 0, false, runs, count);
}

/* The packet the planner forms of all of cmds[0..count). */
static struct hk_cmd_buffer *
pack_planned(const char *name, struct hk_cmd_buffer **cmds, unsigned count,
             struct hk_cs **out)
{
   struct hk_pack_run runs[16];
   unsigned runs_formed;
   plan(cmds, count, runs, &runs_formed);
   T(name, runs_formed == 1 && runs[0].first == 0 && runs[0].count == count);
   return pack(cmds, count, out);
}

static bool
runs_are(const struct hk_pack_run *runs, unsigned count, unsigned expected,
         const unsigned *firsts, const unsigned *lengths)
{
   if (count != expected)
      return false;
   for (unsigned r = 0; r < count; ++r) {
      if (runs[r].first != firsts[r] || runs[r].count != lengths[r])
         return false;
   }
   return true;
}

static struct hk_cmd_buffer *
looping(void)
{
   struct hk_cmd_buffer *c = plain(1);
   list_first_entry(&c->control_streams, struct hk_cs, node)->pack_loops = true;
   return c;
}

/* A member waiting for render work: a GFX -> COMPUTE barrier before it. */
static struct hk_cmd_buffer *
after_render(void)
{
   struct hk_cmd_buffer *c = new_cmd();
   hk_sync_barrier(c, HK_SYNC_GFX, HK_SYNC_COMPUTE);
   add_plain(c, 1);
   return c;
}

static void
test_plan(void)
{
   struct hk_pack_run runs[16];
   unsigned count;

   struct hk_cmd_buffer *three[] = {plain(1), plain(1), plain(1)};
   plan(three, 3, runs, &count);
   T("three plain members form one packet",
     runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){3}));

   struct hk_cmd_buffer *loops2[] = {looping(), looping()};
   plan(loops2, 2, runs, &count);
   T("two looping members stay apart", count == 0);

   struct hk_cmd_buffer *loops_mixed[] = {plain(1), looping(), looping(),
                                          plain(1)};
   plan(loops_mixed, 4, runs, &count);
   T("a packet holds at most one looping member",
     runs_are(runs, count, 2, (unsigned[]){0, 2}, (unsigned[]){2, 2}));

   struct hk_cmd_buffer *later_render[] = {plain(1), after_render()};
   plan_on(later_render, 2, HK_SYNC_GFX, false, runs, &count);
   T("a member adding a render wait starts a new packet", count == 0);
   plan(later_render, 2, runs, &count);
   T("a render wait on a queue that never rendered is no wait",
     runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){2}));

   struct hk_cmd_buffer *render_first[] = {after_render(), plain(1),
                                           after_render()};
   plan_on(render_first, 3, HK_SYNC_GFX, false, runs, &count);
   T("members waiting for no more than the first one join it",
     runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){3}));

   struct hk_cmd_buffer *tail = plain(1);
   hk_sync_barrier(tail, HK_SYNC_GFX, HK_SYNC_COMPUTE);
   struct hk_cmd_buffer *render_tail[] = {plain(1), tail, plain(1)};
   plan_on(render_tail, 3, HK_SYNC_GFX, false, runs, &count);
   T("a render wait from an earlier member's tail splits before the next",
     runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){2}));

   struct hk_cmd_buffer *xfer = new_cmd();
   xfer->sync.kind_override = HK_SYNC_XFER;
   add_plain(xfer, 1);
   xfer->sync.kind_override = 0;
   struct hk_cmd_buffer *kinds[] = {plain(1), xfer};
   plan_on(kinds, 2, HK_SYNC_GFX, false, runs, &count);
   T("after render work, a member adding a kind starts a new packet",
     count == 0);

   /* A compute-only barrier records COMPUTE against the render subqueue
    * too; with no COMPUTE work there, it is no render wait. The chain of
    * command buffers recorded as {dispatch, COMPUTE -> COMPUTE barrier} is
    * the pattern packing is for, on G16G and on M5. */
   static const struct {
      enum agx_chip chip;
      const char *name;
   } chips[] = {
      {AGX_CHIP_G16G, "G16G"},
      {AGX_CHIP_G17P, "G17P"},
      {AGX_CHIP_G17S, "G17S"},
   };
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      char name[128];
#define C(text) (snprintf(name, sizeof(name), "%s: %s", chips[c].name, text), name)
      chip = chips[c].chip;
      struct hk_cmd_buffer *chain[9];
      for (unsigned i = 0; i < 8; ++i) {
         chain[i] = plain(1);
         hk_sync_barrier(chain[i], HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
      }
      plan(chain, 8, runs, &count);
      T(C("a dispatch-then-barrier chain forms one packet"),
        runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){8}));
      plan(chain, 3, runs, &count);
      T(C("three of them form one packet"),
        runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){3}));
      struct hk_cmd_buffer *render = new_cmd();
      add(render, false);
      struct hk_cmd_buffer *after[9] = {render};
      memcpy(after + 1, chain, 8 * sizeof(*chain));
      plan(after, 9, runs, &count);
      T(C("the chain forms one packet after a render too"),
        runs_are(runs, count, 1, (unsigned[]){1}, (unsigned[]){8}));
      plan_on(chain, 8, HK_SYNC_GFX, false, runs, &count);
      T(C("the chain forms one packet on a queue that ran renders"),
        runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){8}));
      plan_on(chain, 8, HK_SYNC_COMPUTE, false, runs, &count);
      T(C("compute work on the render subqueue makes the barrier a render "
          "wait"),
        count == 0);
#undef C
   }
   chip = AGX_CHIP_G16G;

   /* A tail holds later streams of the kinds it names. A packet of compute
    * members takes none of a GFX -> GFX tail, so it forms; a GFX -> COMPUTE
    * tail splits ("a render wait from an earlier member's tail"). */
   struct hk_cmd_buffer *gfx_tail[4];
   for (unsigned i = 0; i < 4; ++i) {
      gfx_tail[i] = plain(1);
      hk_sync_barrier(gfx_tail[i], HK_SYNC_GFX, HK_SYNC_GFX);
   }
   plan_on(gfx_tail, 4, HK_SYNC_GFX, false, runs, &count);
   T("a tail for render kinds only does not split compute members",
     runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){4}));

   /* A GFX -> XFER tail on transfer members: the packet has transfer kind,
    * so it would wait for the render before the second member. */
   struct hk_cmd_buffer *xfer_pair[2];
   for (unsigned i = 0; i < 2; ++i) {
      xfer_pair[i] = new_cmd();
      xfer_pair[i]->sync.kind_override = HK_SYNC_XFER;
      add_plain(xfer_pair[i], 1);
      xfer_pair[i]->sync.kind_override = 0;
   }
   hk_sync_barrier(xfer_pair[0], HK_SYNC_GFX, HK_SYNC_XFER);
   plan_on(xfer_pair, 2, HK_SYNC_GFX, false, runs, &count);
   T("a render tail of a kind the packet has splits", count == 0);
   struct hk_cmd_buffer *compute_pair[] = {plain(1), plain(1)};
   hk_sync_barrier(compute_pair[0], HK_SYNC_GFX, HK_SYNC_XFER);
   plan_on(compute_pair, 2, HK_SYNC_GFX, false, runs, &count);
   T("a render tail of a kind the packet lacks does not split",
     runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){2}));

   struct hk_cmd_buffer *xfer_plain = new_cmd();
   xfer_plain->sync.kind_override = HK_SYNC_XFER;
   add_plain(xfer_plain, 1);
   xfer_plain->sync.kind_override = 0;
   struct hk_cmd_buffer *kinds_no_render[] = {plain(1), xfer_plain};
   plan(kinds_no_render, 2, runs, &count);
   T("kinds may differ while the render subqueue ran nothing",
     runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){2}));

   struct hk_cmd_buffer *big[] = {plain(200), plain(200)};
   plan(big, 2, runs, &count);
   T("the dispatch bound splits a run", count == 0);
}

/* The wait of `cs` on subqueue `sq` as an ordinal of the queue that had
 * `base` commands on it before this submission; 0 for none. */
static uint64_t
wait_of(const struct hk_cs *cs, unsigned sq, uint64_t base, bool exact)
{
   const uint16_t b = cs->sync.barrier[sq];
   if (b != DRM_ASAHI_BARRIER_NONE)
      return base + b;
   return exact ? cs->sync.prior[sq] : 0;
}

/* Packing follows the stream format and the kernel, not a list of measured
 * parts. */
static void
test_boundary(void)
{
   struct hk_device *dev = test_alloc(sizeof(*dev));
   static const struct {
      enum agx_chip chip;
      const char *name;
   } m5[] = {{AGX_CHIP_G17G, "G17G"}, {AGX_CHIP_G17S, "G17S"},
             {AGX_CHIP_G17C, "G17C"}, {AGX_CHIP_G17D, "G17D"}};
   char name[128];
   for (unsigned i = 0; i < ARRAY_SIZE(m5); ++i) {
      dev->dev.chip = m5[i].chip;
      dev->dev.params.features = 0;
      snprintf(name, sizeof(name), "%s without visible completion: no packing",
               m5[i].name);
      T(name, hk_pack_boundary(dev) == HK_PACK_BOUNDARY_NONE);
      dev->dev.params.features = DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION;
      snprintf(name, sizeof(name), "%s with visible completion packs",
               m5[i].name);
      T(name, hk_pack_boundary(dev) == HK_PACK_BOUNDARY_VISIBLE_COMPLETION);
   }

   dev->dev.params.features = 0;
   dev->dev.chip = AGX_CHIP_G16G;
   T("G16G: HAL200 strong flush boundary",
     hk_pack_boundary(dev) == HK_PACK_BOUNDARY_STRONG_FLUSH);
   dev->dev.chip = AGX_CHIP_G16S;
   T("G16S: the same HAL200 boundary",
     hk_pack_boundary(dev) == HK_PACK_BOUNDARY_STRONG_FLUSH);
   dev->dev.chip = AGX_CHIP_G17P;
   T("G17P: HAL200 strong flush boundary (alias-scoped at pack time)",
     hk_pack_boundary(dev) == HK_PACK_BOUNDARY_STRONG_FLUSH);
   dev->dev.chip = AGX_CHIP_G15G;
   T("G15G: strong flush boundary",
     hk_pack_boundary(dev) == HK_PACK_BOUNDARY_STRONG_FLUSH);
   dev->dev.chip = AGX_CHIP_G15S;
   T("G15S: strong flush boundary",
     hk_pack_boundary(dev) == HK_PACK_BOUNDARY_STRONG_FLUSH);
   dev->dev.chip = AGX_CHIP_G17A;
   T("G17A: HAL200, caches pending, no boundary",
     hk_pack_boundary(dev) == HK_PACK_BOUNDARY_NONE);
   dev->dev.chip = AGX_CHIP_G13X;
   T("G13X: AGX2 strong flush boundary",
     hk_pack_boundary(dev) == HK_PACK_BOUNDARY_STRONG_FLUSH);
   dev->dev.chip = AGX_CHIP_G14G;
   T("G14G: AGX2 strong flush boundary",
     hk_pack_boundary(dev) == HK_PACK_BOUNDARY_STRONG_FLUSH);

   uint8_t agx2_term[HK_PACK_CDM_TERMINATE_MAX] = {0};
   const size_t agx2_term_size =
      hk_pack_cdm_terminate(AGX_CHIP_G14G, agx2_term);
   T("AGX2 terminate packs 8 bytes",
     agx2_term_size == AGX_CDM_STREAM_TERMINATE_LENGTH);
   struct hk_cmd_buffer *agx2_cmd = new_cmd();
   struct hk_cs *agx2_cs = add(agx2_cmd, true);
   uint8_t *agx2_bytes =
      test_alloc(64 + agx2_term_size + HK_CS_SPACE_PADDING);
   memcpy(agx2_bytes + 64, agx2_term, agx2_term_size);
   agx2_cs->start = agx2_bytes;
   agx2_cs->current = agx2_bytes + 64 + agx2_term_size;
   agx2_cs->end = (uint8_t *)agx2_cs->current + HK_CS_SPACE_PADDING;
   agx2_cs->strong_flush_end = agx2_bytes + 64;
   agx2_cs->stats.cmds = agx2_cs->stats.calls = 1;
   agx2_cs->pack_dispatches = 1;
   agx2_cs->pack_threads = 64;
   agx2_cs->pack_static_work = 64 * 16;
   util_dynarray_init(&agx2_cs->imm_writes, NULL);
   util_dynarray_init(&agx2_cs->compute_resources, NULL);
   T("AGX2 G14G accepts strong-flush stream with 8-byte terminate",
     hk_pack_run_member(dev_with(AGX_CHIP_G14G, 0), agx2_cmd, agx2_term,
                        false) == agx2_cs);
   agx2_cs->strong_flush_end = NULL;
   T("AGX2 G14G rejects stream without strong-flush tail",
     !hk_pack_run_member(dev_with(AGX_CHIP_G14G, 0), agx2_cmd, agx2_term,
                         false));

   chip = AGX_CHIP_G17S;
   struct hk_cmd_buffer *c = plain(1);
   T("no member without visible completion",
     !hk_pack_run_member(dev_with(AGX_CHIP_G17S, 0), c, terminate, false));
   T("a member with visible completion",
     hk_pack_run_member(
        dev_with(AGX_CHIP_G17S, DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION),
        c, terminate, false));

   /* A stream an application barrier ends: a strong flush without memory
    * scope, so no visible completion. The member check takes it exactly
    * where hk_pack_admits_strong_tail() says, which the trace reports. */
   struct hk_cs *s = list_first_entry(&c->control_streams, struct hk_cs, node);
   s->memory_flush_end = NULL;
   s->visible_completion = false;
   T("M5: a strong-flush tail, and the trace says so",
     hk_pack_run_member(
        dev_with(AGX_CHIP_G17S, DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION),
        c, terminate, false) &&
        hk_pack_admits_strong_tail(HK_PACK_BOUNDARY_VISIBLE_COMPLETION));
   T("no strong-flush tail without a boundary",
     !hk_pack_admits_strong_tail(HK_PACK_BOUNDARY_NONE));
   chip = AGX_CHIP_G16G;
   struct hk_cmd_buffer *g = plain(1);
   s = list_first_entry(&g->control_streams, struct hk_cs, node);
   s->strong_flush_end = s->weak_barrier_end;
   s->weak_barrier_end = NULL;
   T("G16G: a strong-flush tail, and the trace says so",
     hk_pack_run_member(dev_with(AGX_CHIP_G16G, 0), g, terminate, false) &&
        hk_pack_admits_strong_tail(HK_PACK_BOUNDARY_STRONG_FLUSH));
   chip = AGX_CHIP_G17P;
   struct hk_cmd_buffer *p = plain(1);
   T("G17P: an ordering-barrier tail",
     hk_pack_run_member(dev_with(AGX_CHIP_G17P, 0), p, terminate, false));
   s = list_first_entry(&p->control_streams, struct hk_cs, node);
   s->strong_flush_end = s->weak_barrier_end;
   s->weak_barrier_end = NULL;
   T("G17P: a strong-flush tail, and the trace says so",
     hk_pack_run_member(dev_with(AGX_CHIP_G17P, 0), p, terminate, false) &&
        hk_pack_admits_strong_tail(HK_PACK_BOUNDARY_STRONG_FLUSH));
}

/* HK_PACK_ATOMICS: on by default where the kernel runs a VM's compute
 * commands one after another (G16G), opt-in elsewhere with one kernel queue;
 * an unparsable value keeps the default. */
static void
test_atomics_choice(void)
{
   T("G16G: atomics packed by default", hk_pack_atomics_choice(true, NULL, 1));
   T("G16G: HK_PACK_ATOMICS=0 keeps them apart",
     !hk_pack_atomics_choice(true, "0", 1));
   T("G16G: HK_PACK_ATOMICS=1", hk_pack_atomics_choice(true, "1", 1));
   T("G16G: an unparsable value keeps the default",
     hk_pack_atomics_choice(true, "maybe", 1));
   T("G16G: HK_PACK_ATOMICS=off keeps them apart",
     !hk_pack_atomics_choice(true, "off", 1) &&
        !hk_pack_atomics_choice(true, "OFF", 1));
   T("M5: atomics apart by default", !hk_pack_atomics_choice(false, NULL, 1));
   T("M5: HK_PACK_ATOMICS=1 with one kernel queue",
     hk_pack_atomics_choice(false, "1", 1));
   T("M5: HK_PACK_ATOMICS=1 with two kernel queues",
     !hk_pack_atomics_choice(false, "1", 2));
   T("M5: an unparsable value keeps the default",
     !hk_pack_atomics_choice(false, "maybe", 1));
   T("M5: HK_PACK_ATOMICS=on with one kernel queue",
     hk_pack_atomics_choice(false, "on", 1));
}

/* hk_sync_forget_stream() leaves a command buffer's latest render of a kind
 * on a render of another kind: the resolver counts the kind as run, and so
 * does the planner, which reads the same fields. */
static void
test_forget(bool exact)
{
   const char *mode = exact ? "exact" : "legacy";
   char name[160];
#define N(text) (snprintf(name, sizeof(name), "%s: %s", mode, text), name)
   struct hk_cmd_buffer *c[3];
   struct hk_cmd_buffer *r = new_cmd();
   r->sync.kind_override = HK_SYNC_XFER;
   add(r, false);
   r->sync.kind_override = 0;
   struct hk_cs *gone = add(r, false);
   hk_sync_forget_stream(r, gone);
   list_del(&gone->node);
   c[0] = r;
   c[1] = plain(1);
   c[2] = after_render();

   struct hk_queue *q = new_queue(exact);
   struct hk_pack_run runs[4];
   unsigned count;
   plan_queue(q, c, 3, false, runs, &count);
   T(N("forget: the member waiting for the remaining render starts a packet"),
     count == 0);
   T(N("forget: resolves"), resolve(q, c, 3));
   T(N("forget: the resolver agrees"),
     wait_of(first_cs(c[1]), HK_SYNC_VDM, 0, exact) == 0 &&
        wait_of(first_cs(c[2]), HK_SYNC_VDM, 0, exact) != 0);
#undef N
}

/* A render forgotten in an earlier submission on an exact-prior queue: the
 * queue's latest positions (last_absolute), not the kinds it ran (ever), say
 * which priors wait (from the fourth review of #488). */
static void
test_forget_history(void)
{
   struct hk_cmd_buffer *r = new_cmd();
   r->sync.kind_override = HK_SYNC_XFER;
   add(r, false);
   r->sync.kind_override = 0;
   struct hk_cs *gone = add(r, false);
   hk_sync_forget_stream(r, gone);
   list_del(&gone->node);
   struct hk_queue *q = new_queue(true);
   T("exact: earlier forget: history resolves", resolve(q, &r, 1));
   struct hk_cmd_buffer *c[2] = {plain(1), after_render()};
   struct hk_pack_run runs[4];
   unsigned count;
   plan_queue(q, c, 2, false, runs, &count);
   T("exact: a render forgotten in an earlier submission splits", count == 0);
   T("exact: earlier forget: resolves", resolve(q, c, 2));
   T("exact: earlier forget: the resolver agrees",
     wait_of(first_cs(c[0]), HK_SYNC_VDM, 0, true) == 0 &&
        wait_of(first_cs(c[1]), HK_SYNC_VDM, 0, true) != 0);
}

/* A tail of a graphics destination class the packet's stream belongs to
 * holds the members after it: a stream of every kind waits for the
 * fragment-class requirements too (hk_sync_full_kinds()). */
static void
test_full_classes(void)
{
   struct hk_cmd_buffer *all[2];
   for (unsigned i = 0; i < 2; ++i) {
      all[i] = new_cmd();
      all[i]->sync.kind_override = HK_SYNC_ALL;
      add_plain(all[i], 1);
      all[i]->sync.kind_override = 0;
   }
   hk_sync_barrier(all[0], HK_SYNC_GFX, HK_SYNC_GFX_FRAG);
   T("a fragment-class tail is a requirement of a stream of every kind",
     all[0]->sync.req[__builtin_ctz(HK_SYNC_GFX_FRAG)][HK_SYNC_VDM].prior &
        HK_SYNC_GFX);
   struct hk_pack_run runs[4];
   unsigned count;
   plan_on(all, 2, HK_SYNC_GFX, false, runs, &count);
   T("a fragment-class render tail splits members of every kind", count == 0);
   struct hk_cs *cs;
   pack(all, 2, &cs);
   T("the fold applies a fragment-class tail to the members after it",
     cs->sync.dep[HK_SYNC_VDM].prior & HK_SYNC_GFX);
}

/* M5: where memory has no second address, a command buffer recorded as
 * {dispatch, COMPUTE -> COMPUTE barrier} ends with the application's strong
 * flush without memory scope. Those streams are members too (the packet's
 * end gets the memory scope), so the chain forms one packet, also when its
 * members' tails mix both flushes. */
static void
test_m5_barrier_chain(void)
{
   struct hk_pack_run runs[16];
   unsigned count;

   chip = AGX_CHIP_G17S;
   struct hk_cmd_buffer *chain[8];
   for (unsigned i = 0; i < 8; ++i) {
      chain[i] = plain(1);
      hk_sync_barrier(chain[i], HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
      struct hk_cs *s =
         list_first_entry(&chain[i]->control_streams, struct hk_cs, node);
      s->visible_completion = false;
      s->memory_flush_end = NULL;
   }
   plan(chain, 8, runs, &count);
   T("G17S: a chain ending in barriers without memory scope forms one packet",
     runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){8}));
   plan(chain, 3, runs, &count);
   T("G17S: three of them form one packet",
     runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){3}));
   plan_on(chain, 8, HK_SYNC_GFX, false, runs, &count);
   T("G17S: the chain forms one packet on a queue that ran renders",
     runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){8}));

   struct hk_cs *scoped =
      list_first_entry(&chain[3]->control_streams, struct hk_cs, node);
   scoped->visible_completion = true;
   scoped->memory_flush_end = scoped->strong_flush_end;
   plan(chain, 8, runs, &count);
   T("G17S: plain and memory-scoped tails mixed form one packet",
     runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){8}));
   chip = AGX_CHIP_G16G;
}

static void
test_dependencies(bool exact)
{
   struct hk_cs *cs;
   const char *mode = exact ? "exact" : "legacy";
   char name[160];
#define N(text) (snprintf(name, sizeof(name), "%s: %s", mode, text), name)
   const unsigned compute = __builtin_ctz(HK_SYNC_COMPUTE);

   /* (a) A member's tail cut: a later command buffer waits for the packet.
    * Packed by the planner: a compute-only tail is no render wait. */
   struct hk_cmd_buffer *a = plain(1);
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
   struct hk_cmd_buffer *a_members[] = {a, plain(1)};
   struct hk_cmd_buffer *packet =
      pack_planned(N("(a) the planner packs a tail-cut member"), a_members, 2,
                   &cs);
   T(N("(a) tail cut becomes a cut on the packet"),
     packet->sync.req[compute][HK_SYNC_CDM].cut == cs);
   struct hk_cmd_buffer *later = plain(1);
   struct hk_cs *later_cs =
      list_first_entry(&later->control_streams, struct hk_cs, node);
   struct hk_queue *q = new_queue(exact);
   struct hk_cmd_buffer *a_submit[] = {packet, later};
   T(N("(a) resolves"), resolve(q, a_submit, 2));
   T(N("(a) a later command buffer waits for the packet"),
     cs->sync.index == 1 && wait_of(later_cs, HK_SYNC_CDM, 0, exact) == 1);

   /* The same tail through the queue's persisted tail (exact mode: the G16
    * kernel advertises exact priors). */
   if (exact) {
      a = plain(1);
      hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
      struct hk_cmd_buffer *a2_members[] = {a, plain(1)};
      packet = pack_planned(N("(a) next submission: packs"), a2_members, 2,
                            &cs);
      q = new_queue(true);
      struct hk_cmd_buffer *first[] = {packet};
      T(N("(a) next submission: packet resolves"), resolve(q, first, 1));
      const uint64_t base = q->sync.submitted[HK_SYNC_CDM];
      later = plain(1);
      later_cs = list_first_entry(&later->control_streams, struct hk_cs, node);
      struct hk_cmd_buffer *second[] = {later};
      T(N("(a) next submission: resolves"), resolve(q, second, 1));
      T(N("(a) next submission: waits for the packet through the tail"),
        base == 1 && wait_of(later_cs, HK_SYNC_CDM, base, true) == 1);
   }

   /* (b) A tail prior naming an earlier member's kind: cut on the packet. */
   struct hk_cmd_buffer *b = new_cmd();
   hk_sync_barrier(b, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
   add_plain(b, 1);
   struct hk_cmd_buffer *b_members[] = {plain(1), b};
   packet = pack_planned(N("(b) the planner packs a tail-prior member"),
                         b_members, 2, &cs);
   T(N("(b) a prior naming an earlier member becomes a cut on the packet"),
     packet->sync.req[compute][HK_SYNC_CDM].cut == cs);
   later = plain(1);
   later_cs = list_first_entry(&later->control_streams, struct hk_cs, node);
   q = new_queue(exact);
   struct hk_cmd_buffer *b_submit[] = {packet, later};
   T(N("(b) resolves"), resolve(q, b_submit, 2));
   T(N("(b) a later command buffer waits for the packet"),
     wait_of(later_cs, HK_SYNC_CDM, 0, exact) == cs->sync.index);

   /* (c) A member's render wait reaches the packet's barrier: after a
    * render, the first member waits for it and the next joins. */
   struct hk_cmd_buffer *render = new_cmd();
   add(render, false);
   struct hk_cmd_buffer *c_all[] = {render, after_render(), plain(1)};
   struct hk_pack_run runs[4];
   unsigned count;
   plan(c_all, 3, runs, &count);
   T(N("(c) the members after the render form one packet"),
     runs_are(runs, count, 1, (unsigned[]){1}, (unsigned[]){2}));
   packet = pack(c_all + 1, 2, &cs);
   T(N("(c) the packet waits for the render kinds of its members"),
     cs->sync.dep[HK_SYNC_VDM].prior & HK_SYNC_GFX);
   struct hk_cmd_buffer *c_submit[] = {render, packet};
   T(N("(c) resolves"), resolve(new_queue(exact), c_submit, 2));
   T(N("(c) the packet's render barrier names the render"),
     wait_of(cs, HK_SYNC_VDM, 0, exact) == 1);

   /* (d) A render after a packet waits for the packet. */
   struct hk_cmd_buffer *d_members[] = {plain(1), plain(1)};
   packet = pack_planned(N("(d) the planner packs two plain members"),
                         d_members, 2, &cs);
   struct hk_cmd_buffer *d_render = new_cmd();
   hk_sync_barrier(d_render, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   struct hk_cs *d_render_cs = add(d_render, false);
   struct hk_cmd_buffer *d_submit[] = {packet, d_render};
   T(N("(d) resolves"), resolve(new_queue(exact), d_submit, 2));
   T(N("(d) a render after the packet waits for the packet's index"),
     wait_of(d_render_cs, HK_SYNC_CDM, 0, exact) == cs->sync.index);

   /* The last member's tail is for later command buffers only. */
   struct hk_cmd_buffer *e_last = plain(1);
   hk_sync_barrier(e_last, HK_SYNC_GFX, HK_SYNC_COMPUTE);
   struct hk_cmd_buffer *e_members[] = {plain(1), e_last};
   packet = pack_planned(N("(e) the planner packs a render-tail last member"),
                         e_members, 2, &cs);
   T(N("the last member's tail does not hold the packet back"),
     !(cs->sync.dep[HK_SYNC_VDM].prior & HK_SYNC_GFX));
   T(N("the last member's tail still holds later command buffers"),
     packet->sync.req[compute][HK_SYNC_VDM].prior & HK_SYNC_GFX);
#undef N
}

/* The folds the planner's rules make redundant stay as defence in depth:
 * pin them on packets built directly. */
static void
test_folds(void)
{
   struct hk_cs *cs;
   struct hk_cmd_buffer *own[] = {plain(1), after_render()};
   pack(own, 2, &cs);
   T("the packet waits for every member's own render wait",
     cs->sync.dep[HK_SYNC_VDM].prior & HK_SYNC_GFX);

   struct hk_cmd_buffer *first = plain(1);
   hk_sync_barrier(first, HK_SYNC_GFX, HK_SYNC_COMPUTE);
   struct hk_cmd_buffer *tails[] = {first, plain(1)};
   pack(tails, 2, &cs);
   T("a member's tail holds the members after it",
     cs->sync.dep[HK_SYNC_VDM].prior & HK_SYNC_GFX);
}

/* Randomized: packed and unpacked submissions resolve with no wait dropped,
 * and no packet waits for more render work than its first member did. */
static uint32_t rng_state;

static uint32_t
rnd(unsigned n)
{
   rng_state = rng_state * 1664525u + 1013904223u;
   return (rng_state >> 8) % n;
}

struct scb {
   bool render;
   uint8_t kind, pre_s, pre_d, post_s, post_d;
};

static const uint8_t SRC[] = {0, 0, HK_SYNC_GFX, HK_SYNC_COMPUTE, HK_SYNC_XFER,
                              HK_SYNC_ALL};
static const uint8_t DST[] = {HK_SYNC_GFX, HK_SYNC_GFX_FRAG, HK_SYNC_COMPUTE,
                              HK_SYNC_XFER, HK_SYNC_ALL,
                              HK_SYNC_COMPUTE | HK_SYNC_XFER};
static const uint8_t KIND[] = {HK_SYNC_COMPUTE, HK_SYNC_COMPUTE,
                               HK_SYNC_COMPUTE, HK_SYNC_XFER,
                               HK_SYNC_COMPUTE | HK_SYNC_XFER, HK_SYNC_ALL};

static struct scb
rand_cb(void)
{
   struct scb d = {0};
   d.render = rnd(4) == 0;
   d.kind = KIND[rnd(ARRAY_SIZE(KIND))];
   d.pre_s = SRC[rnd(ARRAY_SIZE(SRC))];
   d.pre_d = DST[rnd(ARRAY_SIZE(DST))];
   d.post_s = SRC[rnd(ARRAY_SIZE(SRC))];
   d.post_d = DST[rnd(ARRAY_SIZE(DST))];
   return d;
}

static struct hk_cmd_buffer *
build(const struct scb *d)
{
   struct hk_cmd_buffer *c = new_cmd();
   if (d->pre_s)
      hk_sync_barrier(c, d->pre_s, d->pre_d);
   if (d->render) {
      add(c, false);
   } else {
      c->sync.kind_override = d->kind;
      add_plain(c, 1);
      c->sync.kind_override = 0;
   }
   if (d->post_s)
      hk_sync_barrier(c, d->post_s, d->post_d);
   return c;
}

static unsigned rr_checks, rr_fails, rr_packets;

static void
rr_fail(const char *what, unsigned seed, unsigned i)
{
   if (rr_fails++ < 8)
      printf("FAIL random %s seed=%u cb=%u\n", what, seed, i);
}

static void
rr_one(unsigned seed, bool exact)
{
   rng_state = seed * 2654435761u + 1;
   const unsigned n = 2 + rnd(6);
   struct scb d[8];
   for (unsigned i = 0; i < n; ++i)
      d[i] = rand_cb();
   struct scb next[2] = {rand_cb(), rand_cb()};
   next[0].render = true;
   next[1].render = false;

   struct hk_cmd_buffer *U[8], *P[8];
   for (unsigned i = 0; i < n; ++i) {
      U[i] = build(&d[i]);
      P[i] = build(&d[i]);
   }

   struct hk_pack_run runs[8];
   unsigned count;
   plan_on(P, n, 0, true, runs, &count);
   rr_packets += count;

   int run_of[8];
   for (unsigned i = 0; i < n; ++i)
      run_of[i] = -1;
   for (unsigned r = 0; r < count; ++r)
      for (unsigned m = 0; m < runs[r].count; ++m)
         run_of[runs[r].first + m] = r;

   struct hk_cmd_buffer *L[8];
   struct hk_cs *pcs[8] = {0};
   unsigned m_count = 0;
   for (unsigned i = 0; i < n;) {
      if (run_of[i] < 0) {
         L[m_count++] = P[i++];
         continue;
      }
      const struct hk_pack_run *pr = &runs[run_of[i]];
      L[m_count++] = pack(&P[pr->first], pr->count, &pcs[run_of[i]]);
      i += pr->count;
   }

   struct hk_queue *qu = new_queue(exact), *qp = new_queue(exact);
   if (!resolve(qu, U, n) || !resolve(qp, L, m_count)) {
      rr_fail("resolve failed", seed, 0);
      return;
   }

   int cdm_cb[16];
   unsigned u_cdm = 0, p_cdm = 0;
   for (unsigned i = 0; i < n; ++i) {
      struct hk_cs *u = first_cs(U[i]);
      if (u->type == HK_CS_CDM)
         cdm_cb[u->sync.index] = i, u_cdm = MAX2(u_cdm, u->sync.index);
   }
   for (unsigned e = 0; e < m_count; ++e) {
      struct hk_cs *l = first_cs(L[e]);
      if (l->type == HK_CS_CDM)
         p_cdm = MAX2(p_cdm, l->sync.index);
   }
#define PIDX(i) (run_of[i] >= 0 ? pcs[run_of[i]]->sync.index                  \
                                : first_cs(P[i])->sync.index)

   for (unsigned i = 0; i < n; ++i) {
      struct hk_cs *su = first_cs(U[i]);
      struct hk_cs *sp = run_of[i] >= 0 ? pcs[run_of[i]] : first_cs(P[i]);
      const uint64_t uv = wait_of(su, HK_SYNC_VDM, 0, exact);
      const uint64_t pv = wait_of(sp, HK_SYNC_VDM, 0, exact);
      rr_checks++;
      if (pv < uv)
         rr_fail("render wait dropped", seed, i);
      if (run_of[i] >= 0 && runs[run_of[i]].first == i && pv > uv)
         rr_fail("packet waits for more render work than its first member",
                 seed, i);
      const uint64_t uc = wait_of(su, HK_SYNC_CDM, 0, exact);
      if (uc) {
         const int target = cdm_cb[uc];
         if (run_of[i] >= 0 && run_of[target] == run_of[i])
            continue; /* ordered inside the packet */
         rr_checks++;
         if (wait_of(sp, HK_SYNC_CDM, 0, exact) < (uint64_t)PIDX(target))
            rr_fail("compute wait dropped", seed, i);
      }
   }

   if (!exact)
      return;

   /* A later submission waits through the persisted tail. */
   struct hk_cmd_buffer *U2[2] = {build(&next[0]), build(&next[1])};
   struct hk_cmd_buffer *P2[2] = {build(&next[0]), build(&next[1])};
   const uint64_t ubase[2] = {qu->sync.submitted[0], qu->sync.submitted[1]};
   const uint64_t pbase[2] = {qp->sync.submitted[0], qp->sync.submitted[1]};
   if (!resolve(qu, U2, 2) || !resolve(qp, P2, 2)) {
      rr_fail("next submission: resolve failed", seed, 0);
      return;
   }
   for (unsigned i = 0; i < 2; ++i) {
      struct hk_cs *su = first_cs(U2[i]), *sp = first_cs(P2[i]);
      rr_checks++;
      if (wait_of(sp, HK_SYNC_VDM, pbase[0], true) <
          wait_of(su, HK_SYNC_VDM, ubase[0], true))
         rr_fail("next submission: render wait dropped", seed, 100 + i);
      const uint64_t uc = wait_of(su, HK_SYNC_CDM, ubase[1], true);
      if (!uc)
         continue;
      const uint64_t want =
         uc <= u_cdm ? (uint64_t)PIDX(cdm_cb[uc]) : uc - u_cdm + p_cdm;
      rr_checks++;
      if (wait_of(sp, HK_SYNC_CDM, pbase[1], true) < want)
         rr_fail("next submission: compute wait dropped", seed, 100 + i);
   }
#undef PIDX
}

static void
test_random(void)
{
   for (unsigned seed = 0; seed < 2000; ++seed) {
      rr_one(seed, false);
      rr_one(seed, true);
      if ((seed & 7) == 7) {
         ralloc_free(test_context);
         test_context = ralloc_context(NULL);
      }
   }
   printf("random: %u packets, %u checks, %u failures\n", rr_packets,
          rr_checks, rr_fails);
   T("random submissions keep every wait", rr_fails == 0);
   T("random submissions form packets", rr_packets > 500);
}

/* Randomized with queue history: 1-2 earlier submissions with renders of
 * graphics, transfer and compute kind run unpacked on both queues, then a
 * submission is planned against the queue's state, packed and compared with
 * the same submission unpacked (from the third review of #488). */
static uint64_t
wait_hist(const struct hk_cs *cs, unsigned sq, uint64_t base, bool exact)
{
   const uint16_t b = cs->sync.barrier[sq];
   if (b != DRM_ASAHI_BARRIER_NONE)
      return exact ? base + b : (uint64_t)b + 1; /* legacy 0: earlier ioctls */
   return exact ? cs->sync.prior[sq] : 0;
}

static const uint8_t RKIND[] = {0, 0, HK_SYNC_XFER, HK_SYNC_COMPUTE};

static struct hk_cmd_buffer *
build_kind(const struct scb *d, uint8_t render_kind)
{
   struct hk_cmd_buffer *c = new_cmd();
   if (d->pre_s)
      hk_sync_barrier(c, d->pre_s, d->pre_d);
   c->sync.kind_override = d->render ? render_kind : d->kind;
   if (d->render)
      add(c, false);
   else
      add_plain(c, 1);
   c->sync.kind_override = 0;
   if (d->post_s)
      hk_sync_barrier(c, d->post_s, d->post_d);
   return c;
}

static unsigned hx_checks, hx_fails, hx_packets;

static void
hx_fail(const char *what, unsigned seed, unsigned i, bool exact)
{
   if (hx_fails++ < 8)
      printf("FAIL history %s %s seed=%u cb=%u\n", exact ? "exact" : "legacy",
             what, seed, i);
}

static void
hx_one(unsigned seed, bool exact)
{
   rng_state = seed * 2246822519u + 7;
   struct hk_queue *qu = new_queue(exact), *qp = new_queue(exact);

   const unsigned history = 1 + rnd(2);
   for (unsigned s = 0; s < history; ++s) {
      const unsigned nh = 1 + rnd(4);
      struct hk_cmd_buffer *HU[4], *HP[4];
      for (unsigned i = 0; i < nh; ++i) {
         struct scb d = rand_cb();
         d.render = rnd(2) == 0;
         const uint8_t rk = RKIND[rnd(ARRAY_SIZE(RKIND))];
         HU[i] = build_kind(&d, rk);
         HP[i] = build_kind(&d, rk);
      }
      if (!resolve(qu, HU, nh) || !resolve(qp, HP, nh)) {
         hx_fail("history resolve failed", seed, 0, exact);
         return;
      }
   }

   const unsigned n = 2 + rnd(6);
   struct hk_cmd_buffer *U[8], *P[8];
   for (unsigned i = 0; i < n; ++i) {
      const struct scb d = rand_cb();
      const uint8_t rk = RKIND[rnd(ARRAY_SIZE(RKIND))];
      U[i] = build_kind(&d, rk);
      P[i] = build_kind(&d, rk);
   }

   struct hk_pack_run runs[8];
   unsigned count;
   plan_queue(qp, P, n, true, runs, &count);
   hx_packets += count;
   int run_of[8];
   for (unsigned i = 0; i < n; ++i)
      run_of[i] = -1;
   for (unsigned r = 0; r < count; ++r)
      for (unsigned m = 0; m < runs[r].count; ++m)
         run_of[runs[r].first + m] = r;

   struct hk_cmd_buffer *L[8];
   struct hk_cs *pcs[8] = {0};
   unsigned m_count = 0;
   for (unsigned i = 0; i < n;) {
      if (run_of[i] < 0) {
         L[m_count++] = P[i++];
         continue;
      }
      const struct hk_pack_run *pr = &runs[run_of[i]];
      L[m_count++] = pack(&P[pr->first], pr->count, &pcs[run_of[i]]);
      i += pr->count;
   }

   const uint64_t base = qu->sync.submitted[HK_SYNC_VDM];
   if (!resolve(qu, U, n) || !resolve(qp, L, m_count)) {
      hx_fail("resolve failed", seed, 0, exact);
      return;
   }
   for (unsigned i = 0; i < n; ++i) {
      struct hk_cs *su = first_cs(U[i]);
      struct hk_cs *sp = run_of[i] >= 0 ? pcs[run_of[i]] : first_cs(P[i]);
      const uint64_t uv = wait_hist(su, HK_SYNC_VDM, base, exact);
      const uint64_t pv = wait_hist(sp, HK_SYNC_VDM, base, exact);
      hx_checks++;
      if (pv < uv)
         hx_fail("render wait dropped", seed, i, exact);
      if (run_of[i] >= 0 && runs[run_of[i]].first == i && pv > uv)
         hx_fail("packet waits for more render work than its first member",
                 seed, i, exact);
   }
}

static void
test_history(void)
{
   for (unsigned seed = 0; seed < 4000; ++seed) {
      hx_one(seed, false);
      hx_one(seed, true);
      if ((seed & 7) == 7) {
         ralloc_free(test_context);
         test_context = ralloc_context(NULL);
      }
   }
   printf("history: %u packets, %u checks, %u failures\n", hx_packets,
          hx_checks, hx_fails);
   T("submissions after queue history keep every wait", hx_fails == 0);
   T("submissions after queue history form packets", hx_packets > 500);
}

/* hk_sync.c references these for paths the fixtures never take. */
void
hk_dispatch_imm_writes(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd; (void)cs;
   abort();
}

/* hk_cmd_buffer_get_cs_general() serves every stream slot; fixtures never
 * open a graphics or pre-graphics stream through it. */
void
hk_tess_batch_flush(struct hk_cmd_buffer *cmd)
{
   (void)cmd;
   abort();
}

bool
hk_cs_init_graphics(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   (void)cs;
   abort();
}

/* Fixtures never copy timestamps. Emitting a copy is a bug. */
void
hk_record_stamp_copies(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       const struct hk_stamp_copy *copies, unsigned count)
{
   (void)cmd;
   (void)cs;
   (void)copies;
   (void)count;
   abort();
}

bool
hk_defer_oq_available(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd; (void)cs;
   abort();
}

void
hk_cdm_end_stream_flush(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd; (void)cs;
}

void
hk_flush_query_resets(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   if (cs->query_reset_pending)
      abort();
}

void
hk_g17p_meta_fill_addr(struct hk_cmd_buffer *cmd, struct hk_cs **target,
                       uint64_t address, uint64_t size, uint32_t value)
{
   (void)cmd; (void)target; (void)address; (void)size; (void)value;
   abort();
}

struct agx_ptr
hk_pool_alloc_from_slow(struct hk_cmd_buffer *cmd, struct hk_uploader *uploader,
                        uint32_t size, uint32_t alignment,
                        enum hk_cmd_bo_type type)
{
   (void)cmd; (void)uploader; (void)size; (void)alignment; (void)type;
   abort();
}

struct agx_ptr
hk_pool_cs_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   (void)cmd; (void)size; (void)alignment;
   abort();
}

void
hk_drm_attachment_set_init(struct hk_drm_attachment_set *set)
{
   memset(set, 0, sizeof(*set));
   set->complete = true;
}

/* Reusing one packet across submissions must clear every CDM and sync field
 * while preserving dynarray backing allocations and leaving unused graphics
 * state untouched. */
static void
test_packet_reuse(bool negctl)
{
   struct hk_device *dev = dev_with(AGX_CHIP_G17P, 0);
   struct hk_cmd_buffer *cmd = test_alloc(sizeof(*cmd));
   struct hk_cs *cs = test_alloc(sizeof(*cs));
   uint8_t buf1[256] = {0}, buf2[256] = {0};

   /* Fill unused graphics tails with a sentinel to verify hk_pack_init_packet
    * does not touch them. */
   memset(&cmd->state.gfx, 0x5a, sizeof(cmd->state.gfx));
   memset(&cs->tib, 0x5a, sizeof(*cs) - offsetof(struct hk_cs, tib));

   /* First submission populates CDM state, sync state, and both dynarrays. */
   hk_pack_init_packet(cmd, cs, dev, buf1, 0x10000, sizeof(buf1), false);
   const uint32_t dummy_res = 0x12345678u;
   const uint64_t dummy_q[2] = {0x2000, 0x40};
   util_dynarray_append(&cs->compute_resources, dummy_res);
   util_dynarray_append(&cs->query_access_ranges, dummy_q[0]);
   util_dynarray_append(&cs->query_access_ranges, dummy_q[1]);
   void *res_data = cs->compute_resources.data;
   size_t res_cap = cs->compute_resources.capacity;
   void *q_data = cs->query_access_ranges.data;
   size_t q_cap = cs->query_access_ranges.capacity;

   cs->stats.cmds = 42;
   cs->pack_threads = 9999;
   cs->uses_sampler_heap = true;
   cs->strong_flush_end = buf1 + 64;
   cs->compute_attachments.count = 3;
   cs->compute_attachments.complete = false;
   cs->sync.kind = HK_SYNC_COMPUTE;
   cs->sync.dep[HK_SYNC_VDM].prior = HK_SYNC_GFX;
   cmd->sync.req[0][HK_SYNC_CDM].prior = HK_SYNC_COMPUTE;
   cmd->sync.req[0][HK_SYNC_CDM].cut = cs;
   cmd->sync.last_any[HK_SYNC_CDM] = cs;

   /* Reinitialize the same packet for a second submission (skipped in negctl
    * mode to prove the checks below catch stale packet state). */
   if (!negctl)
      hk_pack_init_packet(cmd, cs, dev, buf2, 0x20000, 128, true);

   T("packet reuse preserves compute_resources backing",
     cs->compute_resources.data == res_data &&
        cs->compute_resources.capacity == res_cap &&
        cs->compute_resources.size == 0);
   T("packet reuse preserves query_access_ranges backing",
     cs->query_access_ranges.data == q_data &&
        cs->query_access_ranges.capacity == q_cap &&
        cs->query_access_ranges.size == 0);
   T("packet reuse clears CDM fields",
     cs->stats.cmds == 0 && cs->pack_threads == 0 && !cs->uses_sampler_heap &&
        cs->strong_flush_end == NULL && cs->compute_attachments.count == 0 &&
        cs->compute_attachments.complete && cs->sync.kind == 0 &&
        cs->sync.dep[HK_SYNC_VDM].prior == 0);
   T("packet reuse initializes stream pointers and flags",
     cs->cmd == cmd && cs->type == HK_CS_CDM && cs->start == buf2 &&
        cs->current == buf2 && cs->end == buf2 + 128 && cs->addr == 0x20000 &&
        cs->root_size == 128 && cs->sync.seq == 1 && cs->visible_completion &&
        list_is_singular(&cmd->control_streams) &&
        first_cs(cmd) == cs && cmd->vk.base.device == &dev->vk);
   T("packet reuse clears command sync state",
     cmd->sync.req[0][HK_SYNC_CDM].prior == 0 &&
        cmd->sync.req[0][HK_SYNC_CDM].cut == NULL &&
        cmd->sync.last_any[HK_SYNC_CDM] == NULL);

   bool gfx_untouched = true;
   const uint8_t *gfx_bytes = (const uint8_t *)&cmd->state.gfx;
   for (size_t i = 0; i < sizeof(cmd->state.gfx); ++i) {
      if (gfx_bytes[i] != 0x5a) {
         gfx_untouched = false;
         break;
      }
   }
   bool cs_tail_untouched = true;
   const uint8_t *tail_bytes = (const uint8_t *)&cs->tib;
   for (size_t i = 0; i < sizeof(*cs) - offsetof(struct hk_cs, tib); ++i) {
      if (tail_bytes[i] != 0x5a) {
         cs_tail_untouched = false;
         break;
      }
   }
   T("packet reuse leaves unused graphics state untouched",
     gfx_untouched && cs_tail_untouched);

   util_dynarray_fini(&cs->compute_resources);
   util_dynarray_fini(&cs->query_access_ranges);
}

static void
test_span_pool_policy(void)
{
   T("16 KiB and 64 KiB spans map to distinct O(1) buckets",
     hk_cmd_span_bucket_for_size(HK_CMD_SPAN_SIZE) !=
        hk_cmd_span_bucket_for_size(HK_CMD_STREAM_SPAN_SIZE));

   struct agx_device g17p = {.chip = AGX_CHIP_G17P, .is_virtio = false};
   struct agx_device g17p_virtio = {.chip = AGX_CHIP_G17P, .is_virtio = true};
   struct agx_device g17s = {.chip = AGX_CHIP_G17S, .is_virtio = false};
   struct agx_device g13g = {.chip = AGX_CHIP_G13G, .is_virtio = false};

   T("native AGX3 caches up to 256 main/root/ioto/vdm spans",
     hk_cmd_pool_span_cache_max(&g17p, HK_CMD_BO_MAIN) ==
        HK_G17P_CMD_POOL_MAIN_BO_MAX &&
        hk_cmd_pool_span_cache_max(&g17p, HK_CMD_BO_ROOT) ==
           HK_G17P_CMD_POOL_MAIN_BO_MAX &&
        hk_cmd_pool_span_cache_max(&g17p, HK_CMD_BO_USC) ==
           HK_CMD_POOL_BO_MAX &&
        hk_cmd_pool_ioto_span_cache_max(&g17p) ==
           HK_G17P_CMD_POOL_MAIN_BO_MAX &&
        hk_cmd_pool_vdm_span_cache_max(&g17p) ==
           HK_G17P_CMD_POOL_MAIN_BO_MAX &&
        hk_cmd_pool_span_cache_max(&g17s, HK_CMD_BO_MAIN) ==
           HK_G17P_CMD_POOL_MAIN_BO_MAX &&
        hk_cmd_pool_ioto_span_cache_max(&g17s) ==
           HK_G17P_CMD_POOL_MAIN_BO_MAX &&
        hk_cmd_pool_vdm_span_cache_max(&g17s) ==
           HK_G17P_CMD_POOL_MAIN_BO_MAX);
   T("virtio and AGX2 keep default span cache cap",
     hk_cmd_pool_span_cache_max(&g17p_virtio, HK_CMD_BO_MAIN) ==
        HK_CMD_POOL_BO_MAX &&
        hk_cmd_pool_span_cache_max(&g17p_virtio, HK_CMD_BO_ROOT) ==
           HK_CMD_POOL_BO_MAX &&
        hk_cmd_pool_ioto_span_cache_max(&g17p_virtio) == HK_CMD_POOL_BO_MAX &&
        hk_cmd_pool_vdm_span_cache_max(&g17p_virtio) == HK_CMD_POOL_BO_MAX &&
        hk_cmd_pool_span_cache_max(&g13g, HK_CMD_BO_MAIN) ==
           HK_CMD_POOL_BO_MAX &&
        hk_cmd_pool_ioto_span_cache_max(&g13g) == HK_CMD_POOL_BO_MAX &&
        hk_cmd_pool_vdm_span_cache_max(&g13g) == HK_CMD_POOL_BO_MAX);
}

void
hk_flush_deferred_clears_why(struct hk_cmd_buffer *cmd, const char *why)
{
   (void)cmd; (void)why;
   abort();
}

void
hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                    enum agx_barrier barrier, enum libagx_program idx,
                    void *data, size_t data_size)
{
   (void)cmd; (void)grid; (void)barrier; (void)idx; (void)data; (void)data_size;
   abort();
}

static void
test_compute_pair_stream_safety(void)
{
   struct hk_compute_pair_dispatch pair = {0};
   struct hk_cs base = {
      .type = HK_CS_CDM,
      .compute_pair = &pair,
      .stats = { .cmds = 1, .calls = 1 },
      .pack_dispatches = 1,
   };
   T("compute_pair clean single-dispatch",
     hk_compute_pair_stream_eligible(&base));

   struct hk_cs cs;
   cs = base; cs.compute_pair = NULL;
   T("compute_pair rejects NULL metadata", !hk_compute_pair_stream_eligible(&cs));

   /* [producer, barrier, X] or [consumer, Y]: multiple commands in one stream */
   cs = base; cs.stats.cmds = 2; cs.stats.calls = 2; cs.pack_dispatches = 2;
   T("compute_pair rejects multi-dispatch stream", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.stats.cmds = 2;
   T("compute_pair rejects extra non-call cmd", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.pack_dispatches = 2;
   T("compute_pair rejects pack_dispatches != 1", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.stream_linked = true;
   T("compute_pair rejects stream_linked", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.calls_stream = true;
   T("compute_pair rejects calls_stream", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.sib_indirect = true;
   T("compute_pair rejects sib_indirect", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.uses_global_atomics = true;
   T("compute_pair rejects global atomics", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.query_access = true;
   T("compute_pair rejects query_access", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.query_reset_pending = true;
   T("compute_pair rejects query_reset_pending", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.timestamp.start.handle = 1;
   T("compute_pair rejects start timestamp", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.timestamp.end.handle = 1;
   T("compute_pair rejects end timestamp", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.g17p_timestamp_query = true;
   T("compute_pair rejects g17p_timestamp_query", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.gen1_native_rq = true;
   T("compute_pair rejects gen1_native_rq", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.as_build = true;
   T("compute_pair rejects as_build", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.alias_sites = 1;
   T("compute_pair accepts single tail alias_site", hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.alias_sites = 2;
   T("compute_pair rejects multiple alias_sites", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.g18_dispatch_sealed = true;
   T("compute_pair rejects g18_dispatch_sealed", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.imm_writes.size = 16;
   T("compute_pair rejects imm_writes", !hk_compute_pair_stream_eligible(&cs));

   cs = base; cs.type = HK_CS_VDM;
   T("compute_pair rejects non-CDM stream", !hk_compute_pair_stream_eligible(&cs));

   /* Self-copy & shifted-overlap safety checks for identity_copy4 kinds. */
   struct hk_storage_backing b_same[3] = {
      {.handle = 7, .offset = 0x1000, .address = 0x100000},
      {0},
      {.handle = 7, .offset = 0x1000, .address = 0x100000},
   };
   bool same = false;
   /* LINEAR_OFFSET shifted overlap on same BO (so != doff) must be rejected. */
   T("copy4 rejects LINEAR_OFFSET shifted overlap",
     !hk_copy4_classify_intervals(false, 1, b_same, 0, 64, 256, &same));
   /* Same BO offset with aliased/mismatched VA must be rejected. */
   struct hk_storage_backing b_va_diff[3] = {
      {.handle = 7, .offset = 0x1000, .address = 0x100000},
      {0},
      {.handle = 7, .offset = 0x1000, .address = 0x200000},
   };
   T("copy4 rejects same-BO aliased VA self-copy",
     !hk_copy4_classify_intervals(false, 1, b_va_diff, 0, 0, 256, &same));
   /* IM2COL_1X1 in-place (stride=1, kw==channels identity) classifies as self-copy. */
   same = false;
   T("copy4 accepts IM2COL_1X1 exact in-place as self-copy",
     hk_copy4_classify_intervals(false, 1, b_same, 0, 0, 256, &same) && same);
   /* IM2COL_1X1 shifted overlap on same BO must be rejected. */
   T("copy4 rejects IM2COL_1X1 shifted overlap",
     !hk_copy4_classify_intervals(false, 1, b_same, 0, 64, 256, &same));
   /* LINEAR_FADD (nsrc==2) in-place is allowed but NEVER classified as self-copy. */
   struct hk_storage_backing b_fadd[3] = {
      {.handle = 7, .offset = 0x1000, .address = 0x100000},
      {.handle = 8, .offset = 0x2000, .address = 0x200000},
      {.handle = 7, .offset = 0x1000, .address = 0x100000},
   };
   same = true;
   T("copy4 accepts LINEAR_FADD in-place without self-copy",
     hk_copy4_classify_intervals(true, 2, b_fadd, 0, 0, 256, &same) && !same);
   /* LINEAR_FADD shifted overlap on same BO must be rejected. */
   T("copy4 rejects LINEAR_FADD shifted overlap",
     !hk_copy4_classify_intervals(true, 2, b_fadd, 0, 64, 256, &same));
}

static struct hk_storage_backing
fadd_backing(uint32_t handle, uint64_t offset, uint32_t size)
{
   return (struct hk_storage_backing){
      .address = ((uint64_t)handle << 24) + offset,
      .offset = offset,
      .size = size,
      .handle = handle,
   };
}

/* Submit-time FADD + copy4 dual-write fusion gate: the fused shader writes the
 * copy4 destination (dst2) while it still reads both FADD sources, so any
 * overlap between dst2 and the FADD output or reads must keep the two
 * dispatches separate. Overlap is decided on BO intervals, so a second
 * descriptor with a different buffer offset that reaches the same bytes
 * counts as an overlap. */
static void
test_fadd_copy4_fusion_gate(void)
{
   enum { SRC0 = 1, SRC1 = 2, OUT = 3, OTHER = 4 };
   const uint32_t bytes = 1024, size = 4096;

   struct hk_compute_pair_dispatch fadd = {0};
   fadd.info.role = HK_COMPUTE_PAIR_FADD;
   fadd.info.fadd.push_bytes = 4;
   fadd.alt_kind = HK_COMPUTE_PAIR_ALT_LINEAR_FADD;
   fadd.extra.copy4 = (struct hk_compute_pair_copy4){0, 0, bytes};
   fadd.bindings[0] = fadd_backing(SRC0, 0, size);
   fadd.bindings[1] = fadd_backing(SRC1, 0, size);
   fadd.bindings[2] = fadd_backing(OUT, 0, size);

   struct hk_compute_pair_dispatch copy = {0};
   copy.info.role = HK_COMPUTE_PAIR_COPY4;
   copy.extra.copy4 = (struct hk_compute_pair_copy4){0, 0, bytes};
   copy.bindings[0] = fadd.bindings[2];
   copy.bindings[1] = fadd_backing(OTHER, 0, size);

   struct hk_compute_pair_dispatch f, c;
   T("fadd+copy4 fuses with dst2 in another BO",
     hk_compute_pair_fadd_copy4_match(&fadd, &copy));

   /* dst2 in the output's BO, through the output's own descriptor. */
   c = copy; c.bindings[1] = fadd.bindings[2]; c.extra.copy4.dst_start = bytes;
   T("fadd+copy4 fuses with dst2 right after the output",
     hk_compute_pair_fadd_copy4_match(&fadd, &c));
   c.extra.copy4.dst_start = bytes / 2;
   T("fadd+copy4 rejects dst2 half over the output",
     !hk_compute_pair_fadd_copy4_match(&fadd, &c));
   c.extra.copy4.dst_start = bytes - 16;
   T("fadd+copy4 rejects dst2 sharing the last vec4 of the output",
     !hk_compute_pair_fadd_copy4_match(&fadd, &c));

   /* dst2 through a second descriptor on the output's BO: offsets differ but
    * the GPU ranges intersect. */
   c = copy; c.bindings[1] = fadd_backing(OUT, 768, bytes);
   T("fadd+copy4 rejects another descriptor overlapping the output",
     !hk_compute_pair_fadd_copy4_match(&fadd, &c));
   c = copy; c.bindings[1] = fadd_backing(OUT, 0, size);
   c.extra.copy4.dst_start = 0;
   T("fadd+copy4 rejects dst2 equal to the output via a second descriptor",
     !hk_compute_pair_fadd_copy4_match(&fadd, &c));
   c = copy; c.bindings[1] = fadd_backing(OUT, 512, size - 512);
   c.extra.copy4.dst_start = 1024;
   T("fadd+copy4 fuses with a second descriptor past the output",
     hk_compute_pair_fadd_copy4_match(&fadd, &c));
   c.extra.copy4.dst_start = 256;
   T("fadd+copy4 rejects a second descriptor ending inside the output",
     !hk_compute_pair_fadd_copy4_match(&fadd, &c));
   c = copy; c.bindings[1] = fadd_backing(OUT, 0, size);
   f = fadd; f.extra.copy4.dst_start = 2048;
   f.extra.copy4.src_start = 2048;
   c.extra.copy4.src_start = 2048;
   c.extra.copy4.dst_start = 1536;
   T("fadd+copy4 rejects dst2 covering the start of an offset output",
     !hk_compute_pair_fadd_copy4_match(&f, &c));

   /* dst2 over either FADD source, which the fused shader still reads. */
   c = copy; c.bindings[1] = fadd_backing(SRC1, 512, size - 512);
   T("fadd+copy4 rejects dst2 overlapping src1 via another descriptor",
     !hk_compute_pair_fadd_copy4_match(&fadd, &c));
   c = copy; c.bindings[1] = fadd_backing(SRC0, bytes - 16, size - bytes);
   T("fadd+copy4 rejects dst2 sharing the last vec4 of src0",
     !hk_compute_pair_fadd_copy4_match(&fadd, &c));
   c = copy; c.bindings[1] = fadd_backing(SRC0, bytes, size - bytes);
   T("fadd+copy4 fuses with dst2 right after src0",
     hk_compute_pair_fadd_copy4_match(&fadd, &c));

   /* In-place FADD (src0 == output). */
   f = fadd; f.bindings[0] = fadd.bindings[2];
   T("in-place fadd+copy4 fuses with disjoint dst2",
     hk_compute_pair_fadd_copy4_match(&f, &copy));
   c = copy; c.bindings[1] = fadd_backing(OUT, 512, size - 512);
   T("in-place fadd+copy4 rejects dst2 over the in-place range",
     !hk_compute_pair_fadd_copy4_match(&f, &c));

   /* The copy must read exactly what the FADD wrote. */
   c = copy; c.extra.copy4.src_start = 16;
   T("fadd+copy4 rejects a shifted copy source",
     !hk_compute_pair_fadd_copy4_match(&fadd, &c));
   c = copy; c.extra.copy4.bytes = bytes / 2;
   T("fadd+copy4 rejects a partial copy of the output",
     !hk_compute_pair_fadd_copy4_match(&fadd, &c));
   c = copy; c.bindings[0] = fadd_backing(OUT, 16, size - 16);
   T("fadd+copy4 rejects a copy source through a shifted descriptor",
     !hk_compute_pair_fadd_copy4_match(&fadd, &c));
   c = copy; c.bindings[1] = fadd_backing(OTHER, 4, size - 4);
   T("fadd+copy4 rejects an unaligned dst2",
     !hk_compute_pair_fadd_copy4_match(&fadd, &c));
   f = fadd; f.alt_kind = HK_COMPUTE_PAIR_ALT_NONE;
   T("fadd+copy4 requires the linear FADD alternate",
     !hk_compute_pair_fadd_copy4_match(&f, &copy));
}

/* A packet that carried a fused compute-pair dispatch is reused like any
 * other: the state the fused emitter sets (matrix ops, M5 resource
 * profiles) starts the next submission clear, and a packet never holds a
 * compute-pair record of its own. */
static void
test_fused_packet_reuse(bool negctl)
{
   struct hk_device *dev = dev_with(AGX_CHIP_G17P, 0);
   struct hk_cmd_buffer *cmd = test_alloc(sizeof(*cmd));
   struct hk_cs *cs = test_alloc(sizeof(*cs));
   uint8_t buf[256] = {0};
   struct hk_compute_pair_dispatch stale = {0};

   hk_pack_init_packet(cmd, cs, dev, buf, 0x10000, sizeof(buf), true);
   const uint32_t profile = 0xfeedu;
   util_dynarray_append(&cs->compute_resources, profile);
   cs->m5_matrix_ops = true;
   cs->compute_pair = &stale;

   if (!negctl)
      hk_pack_init_packet(cmd, cs, dev, buf, 0x10000, sizeof(buf), true);

   T("fused packet reuse clears matrix ops", !cs->m5_matrix_ops);
   T("fused packet reuse clears resource profiles",
     cs->compute_resources.size == 0);
   T("reused packet holds no compute-pair record", cs->compute_pair == NULL);
}

/* The buffers a packed submission pins: each once, none for NULL. */
static void
test_pack_add_bo(void)
{
   struct util_dynarray bos;
   util_dynarray_init(&bos, NULL);
   struct agx_bo *a = (struct agx_bo *)(uintptr_t)0x1000;
   struct agx_bo *b = (struct agx_bo *)(uintptr_t)0x2000;

   T("add_bo accepts NULL", hk_pack_add_bo(&bos, NULL));
   T("add_bo adds nothing for NULL",
     util_dynarray_num_elements(&bos, struct agx_bo *) == 0);
   T("add_bo appends", hk_pack_add_bo(&bos, a) && hk_pack_add_bo(&bos, b));
   T("add_bo keeps a duplicate once", hk_pack_add_bo(&bos, a) &&
     util_dynarray_num_elements(&bos, struct agx_bo *) == 2 &&
     *util_dynarray_element(&bos, struct agx_bo *, 0) == a &&
     *util_dynarray_element(&bos, struct agx_bo *, 1) == b);
   util_dynarray_fini(&bos);
}

static struct hk_pack_footprint
make_fp(uint32_t r_handle, uint64_t r0, uint64_t r1,
        uint32_t w_handle, uint64_t w0, uint64_t w1)
{
   struct hk_pack_footprint fp = {.known = true};
   if (r_handle) {
      fp.reads[0] = (struct hk_pack_interval){.start = r0, .end = r1, .handle = r_handle};
      fp.num_reads = 1;
   }
   if (w_handle) {
      fp.writes[0] = (struct hk_pack_interval){.start = w0, .end = w1, .handle = w_handle};
      fp.num_writes = 1;
   }
   return fp;
}

static void
test_hazard_tracker(void)
{
   struct hk_pack_hazard_state st = {0};

   /* Disjoint dispatches: elide weak barrier when overlap is allowed. */
   struct hk_pack_footprint a = make_fp(1, 0, 1024, 2, 0, 1024);
   struct hk_pack_footprint b = make_fp(3, 0, 1024, 4, 0, 1024);
   hk_pack_hazard_record(&st, &a, false);
   T("hazard: disjoint pair elides barrier when overlap allowed",
     hk_pack_hazard_decide(&st, &b, true) == HK_PACK_BOUNDARY_ELIDE_BARRIER &&
     st.num_dirty_writes == 1 && st.num_dirty_reads == 1);

   /* Disjoint pair with !can_elide_weak keeps weak barrier and clears dirty_reads
    * while preserving dirty_writes for subsequent RAW detection. */
   hk_pack_hazard_record(&st, &b, false);
   struct hk_pack_footprint c = make_fp(5, 0, 1024, 6, 0, 1024);
   T("hazard: !can_elide_weak keeps weak barrier and retains dirty_writes",
     hk_pack_hazard_decide(&st, &c, false) == HK_PACK_BOUNDARY_KEEP_WEAK &&
     st.num_dirty_writes == 2 && st.num_dirty_reads == 0);

   /* Delayed RAW against earlier member `a` (bo 2) triggers STRONG and resets. */
   hk_pack_hazard_record(&st, &c, false);
   struct hk_pack_footprint raw_a = make_fp(2, 512, 1024, 7, 0, 1024);
   T("hazard: delayed RAW triggers strong flush and resets state",
     hk_pack_hazard_decide(&st, &raw_a, true) == HK_PACK_BOUNDARY_STRONG &&
     st.num_dirty_writes == 0 && st.num_dirty_reads == 0 &&
     !st.write_unknown && !st.read_unknown);

   /* WAW against dirty write triggers STRONG. */
   hk_pack_hazard_record(&st, &a, false);
   struct hk_pack_footprint waw_a = make_fp(9, 0, 512, 2, 0, 256);
   T("hazard: WAW triggers strong flush",
     hk_pack_hazard_decide(&st, &waw_a, true) == HK_PACK_BOUNDARY_STRONG &&
     st.num_dirty_writes == 0);

   /* WAR against dirty read keeps weak barrier (does not need strong flush). */
   hk_pack_hazard_record(&st, &a, false);
   struct hk_pack_footprint war_a = make_fp(9, 0, 512, 1, 256, 512);
   T("hazard: WAR keeps weak barrier and retains dirty_writes",
     hk_pack_hazard_decide(&st, &war_a, true) == HK_PACK_BOUNDARY_KEEP_WEAK &&
     st.num_dirty_writes == 1 && st.num_dirty_reads == 0);

   /* Touching ranges on same handle [0,1024) vs [1024,2048) are disjoint. */
   hk_pack_hazard_reset_all(&st);
   struct hk_pack_footprint sub0 = make_fp(1, 0, 1024, 1, 1024, 2048);
   struct hk_pack_footprint sub1 = make_fp(1, 2048, 3072, 1, 3072, 4096);
   hk_pack_hazard_record(&st, &sub0, false);
   T("hazard: non-overlapping subranges on same BO elide barrier",
     hk_pack_hazard_decide(&st, &sub1, true) == HK_PACK_BOUNDARY_ELIDE_BARRIER);

   /* strong_flushed=true clears all accumulated state even for unknown fp. */
   hk_pack_hazard_record(&st, NULL, true);
   T("hazard: strong_flushed clears state",
     st.num_dirty_writes == 0 && st.num_dirty_reads == 0 &&
     !st.write_unknown && !st.read_unknown);

   /* Unknown producer fails closed to STRONG. */
   struct hk_pack_footprint unk = {.known = false};
   hk_pack_hazard_record(&st, &unk, false);
   T("hazard: unknown producer fails closed to strong",
     hk_pack_hazard_decide(&st, &b, true) == HK_PACK_BOUNDARY_STRONG &&
     !st.write_unknown && !st.read_unknown);

   /* Unknown consumer fails closed to STRONG. */
   hk_pack_hazard_record(&st, &a, false);
   T("hazard: unknown consumer fails closed to strong",
     hk_pack_hazard_decide(&st, &unk, true) == HK_PACK_BOUNDARY_STRONG &&
     st.num_dirty_writes == 0);

   /* Duplicate interval dedup avoids spurious overflow. */
   hk_pack_hazard_reset_all(&st);
   for (unsigned i = 0; i < HK_PACK_HAZARD_MAX_INTERVALS + 16; ++i)
      hk_pack_hazard_record(&st, &a, false);
   T("hazard: duplicate intervals deduplicate without overflow",
     !st.write_unknown && !st.read_unknown &&
     st.num_dirty_writes == 1 && st.num_dirty_reads == 1 &&
     hk_pack_hazard_decide(&st, &b, true) == HK_PACK_BOUNDARY_ELIDE_BARRIER);

   /* Distinct write intervals exceeding capacity fail closed to STRONG. */
   hk_pack_hazard_reset_all(&st);
   for (unsigned i = 0; i <= HK_PACK_HAZARD_MAX_INTERVALS; ++i) {
      struct hk_pack_footprint fi = make_fp(100, 0, 64, 200 + i, 0, 64);
      hk_pack_hazard_record(&st, &fi, false);
   }
   T("hazard: dirty_writes overflow fails closed to strong",
     st.write_unknown &&
     hk_pack_hazard_decide(&st, &b, true) == HK_PACK_BOUNDARY_STRONG);

   /* Distinct read intervals exceeding capacity fail closed to KEEP_WEAK. */
   hk_pack_hazard_reset_all(&st);
   for (unsigned i = 0; i <= HK_PACK_HAZARD_MAX_INTERVALS; ++i) {
      struct hk_pack_footprint fi = make_fp(300 + i, 0, 64, 400, 0, 64);
      hk_pack_hazard_record(&st, &fi, false);
   }
   T("hazard: dirty_reads overflow fails closed to keep weak",
     !st.write_unknown && st.read_unknown &&
     hk_pack_hazard_decide(&st, &b, true) == HK_PACK_BOUNDARY_KEEP_WEAK &&
     !st.read_unknown && st.num_dirty_reads == 0 && st.num_dirty_writes == 1);
}

static void
test_liveness_tracker(void)
{
   struct hk_pack_liveness_state st;
   hk_pack_liveness_reset(&st);

   const struct hk_pack_interval h1_0_64 = { .start = 0, .end = 64, .handle = 1 };
   const struct hk_pack_interval h1_64_128 = { .start = 64, .end = 128, .handle = 1 };
   const struct hk_pack_interval h1_0_128 = { .start = 0, .end = 128, .handle = 1 };
   const struct hk_pack_interval h1_16_48 = { .start = 16, .end = 48, .handle = 1 };
   const struct hk_pack_interval h1_0_16 = { .start = 0, .end = 16, .handle = 1 };
   const struct hk_pack_interval h1_48_128 = { .start = 48, .end = 128, .handle = 1 };
   const struct hk_pack_interval h2_0_64 = { .start = 0, .end = 64, .handle = 2 };

   T("liveness: empty state covers nothing", !hk_pack_liveness_covers(&st, &h1_0_64));

   hk_pack_liveness_add_overwrite(&st, &h1_0_64);
   T("liveness: exact and sub-interval covered after overwrite",
     hk_pack_liveness_covers(&st, &h1_0_64) &&
     hk_pack_liveness_covers(&st, &h1_16_48) &&
     !hk_pack_liveness_covers(&st, &h1_0_128) &&
     !hk_pack_liveness_covers(&st, &h2_0_64));

   /* Adjacent overwrite coalesces into single [0, 128) interval. */
   hk_pack_liveness_add_overwrite(&st, &h1_64_128);
   T("liveness: adjacent overwrites coalesce",
     st.num_dead == 1 && hk_pack_liveness_covers(&st, &h1_0_128));

   /* Interior read splits [0, 128) into [0, 16) and [48, 128). */
   hk_pack_liveness_kill_read(&st, &h1_16_48);
   T("liveness: interior read splits dead interval",
     st.num_dead == 2 &&
     hk_pack_liveness_covers(&st, &h1_0_16) &&
     hk_pack_liveness_covers(&st, &h1_48_128) &&
     !hk_pack_liveness_covers(&st, &h1_16_48) &&
     !hk_pack_liveness_covers(&st, &h1_0_64));

   /* Left and right trims + exact kill. */
   const struct hk_pack_interval h1_0_8 = { .start = 0, .end = 8, .handle = 1 };
   const struct hk_pack_interval h1_8_16 = { .start = 8, .end = 16, .handle = 1 };
   const struct hk_pack_interval h1_96_128 = { .start = 96, .end = 128, .handle = 1 };
   const struct hk_pack_interval h1_48_96 = { .start = 48, .end = 96, .handle = 1 };
   hk_pack_liveness_kill_read(&st, &h1_0_8);
   hk_pack_liveness_kill_read(&st, &h1_96_128);
   T("liveness: left/right trims preserve remaining dead ranges",
     hk_pack_liveness_covers(&st, &h1_8_16) &&
     !hk_pack_liveness_covers(&st, &h1_0_8) &&
     hk_pack_liveness_covers(&st, &h1_48_96) &&
     !hk_pack_liveness_covers(&st, &h1_96_128));
   hk_pack_liveness_kill_read(&st, &h1_0_128);
   T("liveness: covering read kills all overlapping intervals", st.num_dead == 0);

   /* Backward step: in-place read+write leaves interval LIVE before op. */
   struct hk_pack_footprint inplace = make_fp(1, 0, 64, 1, 0, 64);
   hk_pack_liveness_add_overwrite(&st, &h1_0_64);
   hk_pack_liveness_step_backward(&st, &inplace);
   T("liveness: in-place read+write kills deadness before op",
     !hk_pack_liveness_covers(&st, &h1_0_64));

   /* Backward step: pure overwrite marks dead, unknown op fails closed. */
   struct hk_pack_footprint wr1 = make_fp(2, 0, 64, 1, 0, 64);
   hk_pack_liveness_step_backward(&st, &wr1);
   T("liveness: pure overwrite marks target dead",
     hk_pack_liveness_covers(&st, &h1_0_64) &&
     !hk_pack_liveness_covers(&st, &h2_0_64));

   struct hk_pack_footprint unk = { .known = false };
   hk_pack_liveness_step_backward(&st, &unk);
   T("liveness: unknown footprint resets all dead intervals", st.num_dead == 0);

   /* Invalid read fails closed (resets). */
   hk_pack_liveness_add_overwrite(&st, &h1_0_64);
   const struct hk_pack_interval bad_iv = { .start = 64, .end = 64, .handle = 1 };
   hk_pack_liveness_kill_read(&st, &bad_iv);
   T("liveness: invalid read interval fails closed", st.num_dead == 0);

   /* Capacity overflow drops extra intervals soundly (never over-approximates). */
   for (unsigned i = 0; i < HK_PACK_LIVENESS_MAX_INTERVALS + 8; ++i) {
      struct hk_pack_interval iv = { .start = 0, .end = 64, .handle = 100 + i };
      hk_pack_liveness_add_overwrite(&st, &iv);
   }
   const struct hk_pack_interval overflow_iv = {
      .start = 0, .end = 64, .handle = 100 + HK_PACK_LIVENESS_MAX_INTERVALS + 4,
   };
   T("liveness: capacity overflow drops conservatively",
     st.num_dead == HK_PACK_LIVENESS_MAX_INTERVALS &&
     !hk_pack_liveness_covers(&st, &overflow_iv));
}

static void
test_loop_work_scaling(void)
{
   static const struct {
      uint64_t instrs, work;
      uint32_t divisor;
      bool fits;
      uint64_t scaled;
   } cases[] = {
      {1, 1, 3, true, 1},
      {UINT32_MAX, UINT32_MAX, UINT32_MAX, true, UINT64_C(0x1fffffffe)},
      {UINT64_C(1) << 32, UINT64_C(1) << 32, 3, true,
       UINT64_C(0xaaaaaaaaaaaaaaab)},
      {UINT64_MAX, 1, 2, true, UINT64_MAX},
      {UINT64_MAX, 1, 1, false, 0},
      {UINT64_MAX, UINT64_MAX, UINT32_MAX, false, 0},
      /* The doubling wraps and rounding carries into the high limb. */
      {UINT64_MAX, (UINT64_C(1) << 63) + 1, 3, true,
       UINT64_C(0x5555555555555555)},
      /* Rounding also wraps if the doubled numerator ends at 2^128 - 4. */
      {UINT64_MAX - 1, (UINT64_C(1) << 63) + 1, 5, true, 0},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      const uint64_t instrs = cases[i].instrs;
      const uint64_t work = cases[i].work;
      const uint32_t divisor = cases[i].divisor;
      uint64_t scaled = 0;
      const bool fits = hk_pack_scale_instrs_portable(instrs, work, divisor,
                                                     &scaled);
      T("portable loop work scaling fits and rounds correctly",
        fits == cases[i].fits && (!fits || scaled == cases[i].scaled));
#if defined(__SIZEOF_INT128__)
      const __uint128_t native =
         ((__uint128_t)2u * instrs * work + divisor - 1u) / divisor;
      T("portable loop work scaling matches native 128-bit arithmetic",
        fits == (native <= UINT64_MAX) && (!fits || scaled == native));
#endif
   }
}

static struct hk_cmd_buffer *
query_reset_cmd(unsigned dispatches, uint64_t queries)
{
   struct hk_cmd_buffer *c = new_cmd();
   c->sync.kind_override = HK_SYNC_ALL;
   struct hk_cs *s = add_plain(c, 0);
   c->sync.kind_override = 0;
   c->sync.has_gpu_query_reset = true;
   s->stats.cmds = dispatches;
   s->stats.calls = 0;
   s->stats.query_reset_dispatches = dispatches;
   s->stats.query_reset_queries = queries;
   s->pack_dispatches = 0;
   s->pack_threads = 0;
   s->pack_static_work = 0;
   s->query_access = true;
   s->query_reset_pending = false;
   s->g17p_timestamp_query = agx_chip_native_or_m5(chip);
   hk_sync_after_stream(c, s, HK_SYNC_ALL);
   return c;
}

static struct hk_cmd_buffer *
timestamp_query_reset_cmd(unsigned dispatches, uint64_t queries)
{
   struct hk_cmd_buffer *c = new_cmd();
   c->sync.kind_override = HK_SYNC_QRESET;
   struct hk_cs *s = add_plain(c, 0);
   c->sync.kind_override = 0;
   c->sync.has_gpu_query_reset = true;
   s->sync.dep[HK_SYNC_VDM].prior = HK_SYNC_QRESET_SCOPE;
   s->sync.dep[HK_SYNC_CDM].prior = HK_SYNC_QRESET_SCOPE;
   s->stats.cmds = dispatches;
   s->stats.calls = 0;
   s->stats.query_reset_dispatches = dispatches;
   s->stats.query_reset_queries = queries;
   s->pack_dispatches = 0;
   s->pack_threads = 0;
   s->pack_static_work = 0;
   s->query_access = true;
   s->query_reset_pending = false;
   s->g17p_timestamp_query = agx_chip_native_or_m5(chip);
   hk_sync_query_reset_done(c, s);
   return c;
}

static void
test_query_reset_packing(void)
{
   const enum agx_chip saved_chip = chip;
   struct hk_pack_run runs[16];
   unsigned count;

   static const struct {
      enum agx_chip chip;
      const char *name;
   } chips[] = {
      {AGX_CHIP_G16G, "G16G"},
      {AGX_CHIP_G17P, "G17P"},
      {AGX_CHIP_G17S, "G17S"},
   };

   for (unsigned ci = 0; ci < ARRAY_SIZE(chips); ++ci) {
      char name[160];
#define Q(text) (snprintf(name, sizeof(name), "%s: %s", chips[ci].name, text), name)
      chip = chips[ci].chip;

      struct hk_cmd_buffer *qr = query_reset_cmd(2, 4);
      struct hk_cs *s = first_cs(qr);
      T(Q("pure query-reset CB is a member"), member(qr, false) == s);

      s->query_reset_pending = true;
      T(Q("unflushed query-reset CB is rejected"), !member(qr, false));
      s->query_reset_pending = false;

      s->stats.cmds = 3;
      T(Q("query-reset CB with extra non-reset dispatch is rejected"),
        !member(qr, false));
      s->stats.cmds = 2;

      s->stats.query_reset_queries = 0;
      T(Q("zero-query reset CB is rejected"), !member(qr, false));
      s->stats.query_reset_queries = 4;

      s->stats.calls = 1;
      T(Q("query-reset CB with API dispatch is rejected"), !member(qr, false));
      s->stats.calls = 0;

      /* Six consecutive vkCmdResetQueryPool CBs after prior render work
       * (same shape as frame-start query resets) pack into one CDM command. */
      struct hk_cmd_buffer *six[6] = {
         query_reset_cmd(2, 2), query_reset_cmd(2, 2),
         query_reset_cmd(2, 2), query_reset_cmd(2, 2),
         query_reset_cmd(1, 7), query_reset_cmd(1, 1),
      };
      plan_on(six, 6, HK_SYNC_GFX, false, runs, &count);
      T(Q("six consecutive query-reset CBs form one packet"),
        runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){6}));

      /* Query-reset CBs never mix with plain compute CBs so plain packets
       * keep NO_QUERY_ACCESS and do not inherit HK_SYNC_ALL. */
      struct hk_cmd_buffer *qr_then_plain[] = {
         query_reset_cmd(1, 2), plain(1),
      };
      plan_on(qr_then_plain, 2, HK_SYNC_GFX, false, runs, &count);
      T(Q("query-reset CB does not pack with following plain compute CB"),
        count == 0);

      struct hk_cmd_buffer *plain_then_qr[] = {
         plain(1), query_reset_cmd(1, 2),
      };
      plan_on(plain_then_qr, 2, HK_SYNC_GFX, false, runs, &count);
      T(Q("plain compute CB does not pack with following query-reset CB"),
        count == 0);

      struct hk_cmd_buffer *two_each[] = {
         query_reset_cmd(1, 2), query_reset_cmd(1, 2),
         plain(1), plain(1),
      };
      plan_on(two_each, 4, HK_SYNC_GFX, false, runs, &count);
      T(Q("adjacent query-reset and plain compute runs pack separately"),
        runs_are(runs, count, 2, (unsigned[]){0, 2}, (unsigned[]){2, 2}));

      /* Packing folds has_gpu_query_reset and preserves HK_SYNC_ALL tail cuts. */
      struct hk_cs *pcs = NULL;
      struct hk_cmd_buffer *pkt = pack(six, 6, &pcs);
      T(Q("packed query-reset packet propagates has_gpu_query_reset"),
        pkt->sync.has_gpu_query_reset);
      bool all_cuts = true;
      u_foreach_bit(k, HK_SYNC_ALL) {
         if (pkt->sync.req[k][HK_SYNC_CDM].cut != pcs)
            all_cuts = false;
      }
      T(Q("packed query-reset packet keeps HK_SYNC_ALL CDM tail cuts"),
        all_cuts);

      /* Oversized query counts respect HK_PACK_MAX_THREADS. */
      struct hk_cmd_buffer *huge[] = {
         query_reset_cmd(1, HK_PACK_MAX_THREADS / 2 + 1),
         query_reset_cmd(1, HK_PACK_MAX_THREADS / 2 + 1),
      };
      plan(huge, 2, runs, &count);
      T(Q("query-reset thread bound splits oversized run"), count == 0);

      /* Timestamp query resets (HK_SYNC_QRESET) also pack with each other,
       * including when an interior member carries an application render->compute
       * barrier tail, while staying isolated from HK_SYNC_ALL resets and plain
       * compute CBs. */
      struct hk_cmd_buffer *ts_qr = timestamp_query_reset_cmd(2, 2);
      struct hk_cs *ts_s = first_cs(ts_qr);
      T(Q("pure timestamp query-reset CB (HK_SYNC_QRESET) is a member"),
        member(ts_qr, false) == ts_s);

      struct hk_cmd_buffer *bad_qreset = new_cmd();
      bad_qreset->sync.kind_override = HK_SYNC_QRESET;
      add_plain(bad_qreset, 1);
      bad_qreset->sync.kind_override = 0;
      T(Q("non-reset HK_SYNC_QRESET compute CB is rejected"),
        !member(bad_qreset, false));

      struct hk_cmd_buffer *six_ts[6] = {
         timestamp_query_reset_cmd(2, 2), timestamp_query_reset_cmd(2, 2),
         timestamp_query_reset_cmd(2, 2), timestamp_query_reset_cmd(2, 2),
         timestamp_query_reset_cmd(1, 7), timestamp_query_reset_cmd(1, 1),
      };
      hk_sync_barrier(six_ts[2], HK_SYNC_GFX,
                      HK_SYNC_COMPUTE | HK_SYNC_GFX_FRAG);
      plan_on(six_ts, 6, HK_SYNC_GFX, false, runs, &count);
      T(Q("six consecutive timestamp query-reset CBs form one packet"),
        runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){6}));

      struct hk_cs *ts_pcs = NULL;
      struct hk_cmd_buffer *ts_pkt = pack(six_ts, 6, &ts_pcs);
      T(Q("packed timestamp query-reset packet keeps HK_SYNC_QRESET and has_gpu_query_reset"),
        (ts_pcs->sync.kind & ~HK_SYNC_GFX_MEM) == HK_SYNC_QRESET &&
           ts_pkt->sync.has_gpu_query_reset &&
           ts_pkt->sync.last[__builtin_ctz(HK_SYNC_QRESET)][HK_SYNC_CDM] ==
              ts_pcs &&
           (ts_pkt->sync.req[__builtin_ctz(HK_SYNC_COMPUTE)][HK_SYNC_VDM].prior &
            HK_SYNC_GFX));

      struct hk_cmd_buffer *ts_and_all[] = {
         timestamp_query_reset_cmd(1, 2), query_reset_cmd(1, 2),
      };
      plan(ts_and_all, 2, runs, &count);
      T(Q("HK_SYNC_QRESET and HK_SYNC_ALL query-reset CBs do not mix"),
        count == 0);

      struct hk_cmd_buffer *ts_and_plain[] = {
         timestamp_query_reset_cmd(1, 2), plain(1),
      };
      plan(ts_and_plain, 2, runs, &count);
      T(Q("HK_SYNC_QRESET query-reset CB does not mix with plain compute"),
        count == 0);
#undef Q
   }

   chip = saved_chip;
}

int
main(int argc, char **argv)
{
   const bool negctl = argc > 1 && !strcmp(argv[1], "--negctl");
   test_context = ralloc_context(NULL);
   if (!test_context)
      return EXIT_FAILURE;
   test_loop_work_scaling();
   if (negctl) {
      test_packet_reuse(true);
      test_fused_packet_reuse(true);
      printf("%u tests, %u failures\n", tests, failures);
      ralloc_free(test_context);
      return failures ? EXIT_FAILURE : EXIT_SUCCESS;
   }
   test_boundary();
   test_atomics_choice();
   test_rejections();
   test_m5_tails();
   test_plan();
   test_m5_barrier_chain();
   test_dependencies(false);
   test_dependencies(true);
   test_folds();
   test_forget(false);
   test_forget(true);
   test_forget_history();
   test_full_classes();
   test_random();
   test_history();
   test_compute_pair_stream_safety();
   test_fadd_copy4_fusion_gate();
   test_fused_packet_reuse(false);
   test_pack_add_bo();
   test_hazard_tracker();
   test_liveness_tracker();

   /* Bounded-loop accounting, NIR-bound expression eval, and post-fusion budget checks. */
   {
      uint64_t w = 0;
      T("unbounded looping dispatch sets pack_loops",
        hk_pack_account_dispatch_work(1024, 64, true, 0, &w) && w == 65536);
      T("bounded looping dispatch scales work and clears pack_loops",
        !hk_pack_account_dispatch_work(1024, 64, true, 13, &w) && w == 1024ull * 64 * 13);
      T("bounded looping dispatch over work limit sets pack_loops and saturates",
        hk_pack_account_dispatch_work(1ull << 24, 256, true, 16, &w) && w == UINT64_MAX);
      T("overflowing bounded loop sets pack_loops and saturates",
        hk_pack_account_dispatch_work(UINT64_MAX / 2, 4, true, 4, &w) && w == UINT64_MAX);
      T("effective-instr bounded looping dispatch scales work directly",
        !hk_pack_account_dispatch_work_eff(1024, 500, true, 800, &w) && w == 1024ull * 800);
      T("effective-instr zero fails closed to pack_loops",
        hk_pack_account_dispatch_work_eff(1024, 500, true, 0, &w) && w == 1024ull * 500);

      /* Expression evaluator: max(3, kh * (1 + kw)) */
      struct hk_pack_loop_bound lb = {
         .valid = true,
         .work_valid = true,
         .count = 7,
         .work_count = 3,
         .static_nir_ops = 400,
         .ops = {
            [0] = { .kind = HK_PACK_LOOP_OP_CONST, .imm = 3 },
            [1] = { .kind = HK_PACK_LOOP_OP_PARAM, .imm = HK_PACK_LOOP_PARAM_DW_KH },
            [2] = { .kind = HK_PACK_LOOP_OP_CONST, .imm = 1 },
            [3] = { .kind = HK_PACK_LOOP_OP_PARAM, .imm = HK_PACK_LOOP_PARAM_DW_KW },
            [4] = { .kind = HK_PACK_LOOP_OP_ADD, .lhs = 2, .rhs = 3 },
            [5] = { .kind = HK_PACK_LOOP_OP_MUL, .lhs = 1, .rhs = 4 },
            [6] = { .kind = HK_PACK_LOOP_OP_MAX, .lhs = 0, .rhs = 5 },
         },
         .work_ops = {
            [0] = { .kind = HK_PACK_LOOP_OP_PARAM, .imm = HK_PACK_LOOP_PARAM_DW_KH },
            [1] = { .kind = HK_PACK_LOOP_OP_CONST, .imm = 100 },
            [2] = { .kind = HK_PACK_LOOP_OP_MUL, .lhs = 0, .rhs = 1 },
         },
      };
      uint64_t params[HK_PACK_LOOP_PARAM_COUNT] = {0};
      params[HK_PACK_LOOP_PARAM_DW_KH] = 3;
      params[HK_PACK_LOOP_PARAM_DW_KW] = 3;
      uint64_t bval = 0, eff = 0;
      T("eval nested depthwise back-edges expression",
        hk_pack_eval_loop_bound(&lb, params, &bval) && bval == 12);
      T("eval CF-weighted effective instrs tightens multi-branch bound",
        hk_pack_eval_effective_instrs(&lb, 400, params, &eff) &&
        eff == 600 && eff < 400 * bval);
      lb.work_valid = false;
      T("eval effective instrs falls back to agx_instrs * loop_bound without work_ops",
        hk_pack_eval_effective_instrs(&lb, 400, params, &eff) &&
        eff == 400 * bval);
      lb.work_valid = true;
      params[HK_PACK_LOOP_PARAM_DW_KH] = UINT64_MAX;
      params[HK_PACK_LOOP_PARAM_DW_KW] = 2;
      T("eval rejects overflow cleanly",
        !hk_pack_eval_loop_bound(&lb, params, &bval) && bval == 0);
      T("eval effective instrs rejects overflow cleanly",
        !hk_pack_eval_effective_instrs(&lb, 400, params, &eff) && eff == 0);
      lb.valid = false;
      T("eval rejects invalid loop bound",
        !hk_pack_eval_loop_bound(&lb, params, &bval) && bval == 0);
      T("eval effective instrs rejects invalid loop bound",
        !hk_pack_eval_effective_instrs(&lb, 400, params, &eff) && eff == 0);

      /* Post-fusion run re-accounting (e.g. DUAL_PAD4 zero-fill loop). */
      uint64_t nw = 0;
      unsigned nl = 0;
      T("fusion fits when post-fusion work stays within HK_PACK_MAX_STATIC_WORK",
        hk_pack_fusion_fits_run(10000, 0, 3000, 0, 4500, false, &nw, &nl) &&
        nw == 11500 && nl == 0);
      T("fusion rejected when DUAL_PAD4 zero-fill blows HK_PACK_MAX_STATIC_WORK",
        !hk_pack_fusion_fits_run(HK_PACK_MAX_STATIC_WORK - 1000, 0, 500, 0,
                                 5000, false, &nw, &nl));
      T("fusion rejected when adding a second unbounded loop to a packed run",
        !hk_pack_fusion_fits_run(10000, 1, 2000, 0, 2000, true, &nw, &nl));
      T("fusion accepted when replacing the run's only unbounded loop",
        hk_pack_fusion_fits_run(10000, 1, 2000, 1, 2500, true, &nw, &nl) &&
        nw == 10500 && nl == 1);

      /* Two bounded-loop members pack into one packet. */
      struct hk_cmd_buffer *cbs[2] = { plain(1), plain(1) };
      for (unsigned i = 0; i < 2; ++i) {
         struct hk_cs *cs = first_cs(cbs[i]);
         cs->pack_loops = hk_pack_account_dispatch_work(
            4096, 128, true, 13, &w);
         cs->pack_static_work = w;
      }
      struct hk_pack_run runs[2];
      unsigned rc = 0;
      plan_on(cbs, 2, 0, false, runs, &rc);
      T("two bounded looping members pack into one packet",
        rc == 1 && runs[0].count == 2);

      /* Core-count-scaled static-work budget (1x at <=20, 2x at 40, 4x at >=80). */
      T("work budget stays 1x for <=20 cores",
        hk_pack_max_static_work_for_cores(10) == HK_PACK_MAX_STATIC_WORK &&
        hk_pack_max_static_work_for_cores(16) == HK_PACK_MAX_STATIC_WORK &&
        hk_pack_max_static_work_for_cores(20) == HK_PACK_MAX_STATIC_WORK);
      T("work budget scales cleanly on 32/40/80+ cores",
        hk_pack_max_static_work_for_cores(32) ==
           (HK_PACK_MAX_STATIC_WORK / 4u) * 32ull / 5u &&
        hk_pack_max_static_work_for_cores(40) == HK_PACK_MAX_STATIC_WORK * 2ull &&
        hk_pack_max_static_work_for_cores(80) == HK_PACK_MAX_STATIC_WORK * 4ull &&
        hk_pack_max_static_work_for_cores(160) == HK_PACK_MAX_STATIC_WORK * 4ull);
      const uint64_t max40 = hk_pack_max_static_work_for_cores(40);
      T("scaled limit accepts bounded loop between 1x and 2x budget",
        hk_pack_account_dispatch_work_eff_limit(
           1ull << 24, 256, true, 3072, HK_PACK_MAX_STATIC_WORK, &w) &&
        !hk_pack_account_dispatch_work_eff_limit(
           1ull << 24, 256, true, 3072, max40, &w) &&
        w == (1ull << 24) * 3072ull);
      T("scaled fusion budget accepts run between 1x and 2x budget",
        !hk_pack_fusion_fits_run_limit(HK_PACK_MAX_STATIC_WORK - 1000, 0, 500, 0,
                                       5000, false, HK_PACK_MAX_STATIC_WORK,
                                       &nw, &nl) &&
        hk_pack_fusion_fits_run_limit(HK_PACK_MAX_STATIC_WORK - 1000, 0, 500, 0,
                                      5000, false, max40, &nw, &nl) &&
        nw == HK_PACK_MAX_STATIC_WORK + 3500 && nl == 0);
      /* Two 24G thread-instruction members exceed 2^35 (34.4G) on 20 cores
       * but pack into one packet under the 40-core 2^36 (68.7G) budget. */
      struct hk_cmd_buffer *wide_cbs[2] = { plain(1), plain(1) };
      for (unsigned i = 0; i < 2; ++i)
         first_cs(wide_cbs[i])->pack_static_work = UINT64_C(24) << 30;
      struct hk_pack_member wm[2];
      struct hk_device *d40 = device();
      struct hk_sync_queue sq0 = {0};
      hk_pack_describe_submit(wm, d40, &sq0, wide_cbs, 2, terminate, false);
      struct hk_pack_limits lim20 = { .max_attachments = 64, .overhead = 64 };
      struct hk_pack_limits lim40 = {
         .max_attachments = 64, .overhead = 64, .max_static_work = max40,
      };
      unsigned pm = 0;
      size_t tot = 0;
      T("24G + 24G splits under 20-core 1x budget",
        hk_pack_plan(wm, 2, &lim20, runs, &pm, &tot) == 0);
      T("24G + 24G packs under 40-core 2x budget",
        hk_pack_plan(wm, 2, &lim40, runs, &pm, &tot) == 1 && runs[0].count == 2);

      /* Observed-duration looping budget (HK_PACK_MAX_OBS_NS). */
      struct hk_pack_member obs_m[4] = {0};
      struct hk_cmd_buffer *obs_cbs[4] = {
         looping(), looping(), looping(), looping(),
      };
      struct hk_sync_queue obs_sync = {0};
      hk_pack_describe_submit(obs_m, device(), &obs_sync, obs_cbs, 4, terminate,
                              false);
      const struct hk_pack_limits obs_lim = {
         .max_attachments = UINT32_MAX,
         .overhead = 16,
      };
      unsigned obs_pm = 0;
      size_t obs_tot = 0;
      struct hk_pack_run obs_runs[4];

      for (unsigned i = 0; i < 4; ++i)
         obs_m[i].obs_ns = 10ull * 1000000; /* 10 ms each -> 3 fit in 32 ms */
      unsigned obs_rc =
         hk_pack_plan(obs_m, 4, &obs_lim, obs_runs, &obs_pm, &obs_tot);
      T("observed looping members pack up to HK_PACK_MAX_OBS_NS and split",
        obs_rc == 1 && obs_runs[0].first == 0 && obs_runs[0].count == 3);

      for (unsigned i = 0; i < 4; ++i)
         obs_m[i].obs_ns = 15ull * 1000000; /* 15 ms each -> 2+2 */
      obs_rc = hk_pack_plan(obs_m, 4, &obs_lim, obs_runs, &obs_pm, &obs_tot);
      T("observed looping members form multiple budgeted packets",
        obs_rc == 2 && obs_runs[0].count == 2 && obs_runs[1].count == 2);

      for (unsigned i = 0; i < 4; ++i)
         obs_m[i].obs_ns = HK_PACK_MAX_OBS_NS + 1;
      obs_rc = hk_pack_plan(obs_m, 4, &obs_lim, obs_runs, &obs_pm, &obs_tot);
      T("observed looping member over HK_PACK_MAX_OBS_NS does not pack",
        obs_rc == 0);

      obs_m[0].obs_ns = 5ull * 1000000;
      obs_m[1].obs_ns = 0; /* unobserved */
      obs_m[2].obs_ns = 5ull * 1000000;
      obs_m[3].obs_ns = 5ull * 1000000;
      obs_rc = hk_pack_plan(obs_m, 4, &obs_lim, obs_runs, &obs_pm, &obs_tot);
      T("unobserved looping member does not join observed looping packet",
        obs_rc == 1 && obs_runs[0].first == 2 && obs_runs[0].count == 2);
   }

   /* Observed-duration retirement and signalled_ns point tagging. */
   {
      const uint64_t ms = 1000000ull;
      struct hk_queue *q = test_alloc(sizeof(*q));

      /* 1. Progress/query path advancing signalled_value past P updates both
       * signalled_ns and signalled_ns_point so retire measures real completion
       * time (40 ms), not an earlier partial-query timestamp (0.2 ms). */
      q->pack.obs[0] = (struct hk_pack_obs_entry){
         .hash = 0x1111,
         .pending_pt = 10,
         .submit_ns = 1000 * ms,
         .cb_count = 4,
         .loop_count = 4,
         .in_flight_packed = false,
      };
      q->pack.obs_pending = 1;
      hk_queue_note_signalled_at(q, 9, 1000 * ms + 200000ull); /* +0.2 ms */
      hk_queue_note_signalled_at(q, 10, 1040 * ms);            /* +40 ms */
      /* Late older note must not regress signalled_ns or signalled_ns_point. */
      hk_queue_note_signalled_at(q, 9, 1000 * ms + 300000ull);
      T("hk_queue_note_signalled_at advances monotonically",
        q->drm.signalled_value == 10 && q->drm.signalled_ns_point == 10 &&
           q->drm.signalled_ns == 1040 * ms);

      hk_pack_obs_retire_entries(q->pack.obs, HK_PACK_OBS_SLOTS,
                                 &q->pack.obs_pending, &q->pack.obs_epoch,
                                 q->drm.signalled_value,
                                 q->drm.signalled_ns_point,
                                 q->drm.signalled_ns, 1500 * ms);
      T("obs retire uses signalled_ns when sig_point covers pending_pt",
        q->pack.obs[0].pending_pt == 0 && q->pack.obs_pending == 0 &&
           q->pack.obs[0].obs_per_loop_ns == 10 * ms &&
           q->pack.obs_epoch == 1);

      /* 2. Untimestamped advance of signalled_value (sig_point < pending_pt):
       * stale signalled_ns (0.2 ms) must NOT be used; falls back to now_ns
       * (300 ms > HK_PACK_OBS_ADMIT_NS) and rejects admission. */
      memset(q, 0, sizeof(*q));
      q->pack.obs[0] = (struct hk_pack_obs_entry){
         .hash = 0x2222,
         .pending_pt = 20,
         .submit_ns = 1000 * ms,
         .cb_count = 4,
         .loop_count = 4,
         .in_flight_packed = false,
      };
      q->pack.obs_pending = 1;
      hk_queue_note_signalled_at(q, 19, 1000 * ms + 200000ull);
      /* Simulate an untimestamped advance of signalled_value to 20. */
      p_atomic_set(&q->drm.signalled_value, 20);
      hk_pack_obs_retire_entries(q->pack.obs, HK_PACK_OBS_SLOTS,
                                 &q->pack.obs_pending, &q->pack.obs_epoch,
                                 q->drm.signalled_value,
                                 q->drm.signalled_ns_point,
                                 q->drm.signalled_ns, 1300 * ms);
      T("obs retire ignores stale signalled_ns when sig_point < pending_pt",
        q->pack.obs[0].pending_pt == 0 && q->pack.obs_pending == 0 &&
           q->pack.obs[0].obs_per_loop_ns == 0 && q->pack.obs_epoch == 0);

      /* 3. Monitored packed run: backoff above 150 ms, fail-closed above 200 ms. */
      q->pack.obs[0] = (struct hk_pack_obs_entry){
         .hash = 0x3333,
         .pending_pt = 30,
         .submit_ns = 2000 * ms,
         .obs_per_loop_ns = 10 * ms,
         .cb_count = 4,
         .loop_count = 4,
         .in_flight_packed = true,
      };
      q->pack.obs_pending = 1;
      hk_queue_note_signalled_at(q, 30, 2160 * ms); /* +160 ms -> backoff */
      hk_pack_obs_retire_entries(q->pack.obs, HK_PACK_OBS_SLOTS,
                                 &q->pack.obs_pending, &q->pack.obs_epoch,
                                 q->drm.signalled_value,
                                 q->drm.signalled_ns_point,
                                 q->drm.signalled_ns, 2200 * ms);
      T("obs retire raises per-loop budget on packed run > 150 ms",
        q->pack.obs[0].obs_per_loop_ns == 40 * ms && q->pack.obs_epoch == 1);

      q->pack.obs[0].pending_pt = 31;
      q->pack.obs[0].submit_ns = 3000 * ms;
      q->pack.obs_pending = 1;
      hk_queue_note_signalled_at(q, 31, 3250 * ms); /* +250 ms -> fail closed */
      hk_pack_obs_retire_entries(q->pack.obs, HK_PACK_OBS_SLOTS,
                                 &q->pack.obs_pending, &q->pack.obs_epoch,
                                 q->drm.signalled_value,
                                 q->drm.signalled_ns_point,
                                 q->drm.signalled_ns, 3300 * ms);
      T("obs retire resets packed run to unobserved above 200 ms",
        q->pack.obs[0].obs_per_loop_ns == 0 && q->pack.obs_epoch == 2);
   }

   test_packet_reuse(false);
   test_span_pool_policy();
   test_query_reset_packing();
   printf("%u tests, %u failures\n", tests, failures);
   ralloc_free(test_context);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

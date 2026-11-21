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
   } chips[] = {{AGX_CHIP_G16G, "G16G"}, {AGX_CHIP_G17S, "G17S"}};
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
             {AGX_CHIP_G17C, "G17C"}};
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
   T("G17P: memory-scoped flush form, no boundary",
     hk_pack_boundary(dev) == HK_PACK_BOUNDARY_NONE);
   dev->dev.chip = AGX_CHIP_G15S;
   T("G15S: not qualified, no boundary",
     hk_pack_boundary(dev) == HK_PACK_BOUNDARY_NONE);
   dev->dev.chip = AGX_CHIP_G17A;
   T("G17A: HAL200, caches pending, no boundary",
     hk_pack_boundary(dev) == HK_PACK_BOUNDARY_NONE);
   dev->dev.chip = AGX_CHIP_G14G;
   T("G14G: no boundary", hk_pack_boundary(dev) == HK_PACK_BOUNDARY_NONE);

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
   (void)set;
   abort();
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

int
main(void)
{
   test_context = ralloc_context(NULL);
   if (!test_context)
      return EXIT_FAILURE;
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
   printf("%u tests, %u failures\n", tests, failures);
   ralloc_free(test_context);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

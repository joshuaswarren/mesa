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

static struct hk_device *
g16g(void)
{
   struct hk_device *dev = test_alloc(sizeof(*dev));
   dev->dev.chip = AGX_CHIP_G16G;
   return dev;
}

static struct hk_queue *
new_queue(void)
{
   return test_alloc(sizeof(struct hk_queue));
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

/* A compute stream that packing accepts on G16G: `dispatches` direct
 * launches, a G16G ordering barrier before the terminate. */
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
   s->weak_barrier_end = bytes + body;
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
   return hk_pack_run_member(g16g(), c, terminate, atomics);
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

static void
plan(struct hk_cmd_buffer **cmds, unsigned n, struct hk_pack_run *runs,
     unsigned *count)
{
   struct hk_pack_member members[16];
   for (unsigned i = 0; i < n; ++i)
      hk_pack_describe(&members[i], cmds[i], member(cmds[i], false),
                       sizeof(terminate));
   const struct hk_pack_limits limits = {.max_attachments = 16,
                                         .overhead = 64};
   unsigned packed;
   size_t total;
   *count = hk_pack_plan(members, n, &limits, runs, &packed, &total);
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
   plan(later_render, 2, runs, &count);
   T("a member adding a render wait starts a new packet", count == 0);

   struct hk_cmd_buffer *render_first[] = {after_render(), plain(1),
                                           after_render()};
   plan(render_first, 3, runs, &count);
   T("members waiting for no more than the first one join it",
     runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){3}));

   struct hk_cmd_buffer *tail = plain(1);
   hk_sync_barrier(tail, HK_SYNC_GFX, HK_SYNC_COMPUTE);
   struct hk_cmd_buffer *render_tail[] = {plain(1), tail, plain(1)};
   plan(render_tail, 3, runs, &count);
   T("a render wait from an earlier member's tail splits before the next",
     runs_are(runs, count, 1, (unsigned[]){0}, (unsigned[]){2}));

   struct hk_cmd_buffer *xfer = new_cmd();
   xfer->sync.kind_override = HK_SYNC_XFER;
   add_plain(xfer, 1);
   xfer->sync.kind_override = 0;
   struct hk_cmd_buffer *kinds[] = {plain(1), xfer};
   plan(kinds, 2, runs, &count);
   T("a member adding a synchronization kind starts a new packet",
     count == 0);

   struct hk_cmd_buffer *big[] = {plain(200), plain(200)};
   plan(big, 2, runs, &count);
   T("the dispatch bound splits a run", count == 0);
}

static void
test_dependencies(void)
{
   struct hk_cs *cs;

   /* (a) A member's tail cut: a later command buffer waits for the packet. */
   struct hk_cmd_buffer *a = plain(1);
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
   struct hk_cmd_buffer *a_members[] = {a, plain(1)};
   struct hk_cmd_buffer *packet = pack(a_members, 2, &cs);
   T("tail cut becomes a cut on the packet",
     packet->sync.req[__builtin_ctz(HK_SYNC_COMPUTE)][HK_SYNC_CDM].cut == cs);
   struct hk_cmd_buffer *later = plain(1);
   struct hk_cs *later_cs =
      list_first_entry(&later->control_streams, struct hk_cs, node);
   struct hk_cmd_buffer *a_submit[] = {packet, later};
   T("(a) resolves", resolve(new_queue(), a_submit, 2));
   T("(a) a later command buffer waits for the packet",
     later_cs->sync.barrier[HK_SYNC_CDM] == cs->sync.index &&
        cs->sync.index == 1);

   /* (b) A tail prior naming an earlier member's kind: cut on the packet. */
   struct hk_cmd_buffer *b = new_cmd();
   hk_sync_barrier(b, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
   add_plain(b, 1);
   struct hk_cmd_buffer *b_members[] = {plain(1), b};
   packet = pack(b_members, 2, &cs);
   T("(b) a prior naming an earlier member becomes a cut on the packet",
     packet->sync.req[__builtin_ctz(HK_SYNC_COMPUTE)][HK_SYNC_CDM].cut == cs);

   /* (c) A member's render wait reaches the packet's barrier. */
   struct hk_cmd_buffer *render = new_cmd();
   add(render, false);
   struct hk_cmd_buffer *c_members[] = {after_render(), plain(1)};
   packet = pack(c_members, 2, &cs);
   T("(c) the packet waits for the render kinds of its members",
     cs->sync.dep[HK_SYNC_VDM].prior & HK_SYNC_GFX);
   struct hk_cmd_buffer *c_submit[] = {render, packet};
   T("(c) resolves", resolve(new_queue(), c_submit, 2));
   T("(c) the packet's render barrier names the render",
     cs->sync.barrier[HK_SYNC_VDM] == 1);

   /* (d) A render after a packet waits for the packet. */
   struct hk_cmd_buffer *d_members[] = {plain(1), plain(1)};
   packet = pack(d_members, 2, &cs);
   struct hk_cmd_buffer *d_render = new_cmd();
   hk_sync_barrier(d_render, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   struct hk_cs *d_render_cs = add(d_render, false);
   struct hk_cmd_buffer *d_submit[] = {packet, d_render};
   T("(d) resolves", resolve(new_queue(), d_submit, 2));
   T("(d) a render after the packet waits for the packet's index",
     d_render_cs->sync.barrier[HK_SYNC_CDM] == cs->sync.index);

   /* The last member's tail is for later command buffers only. */
   struct hk_cmd_buffer *e_last = plain(1);
   hk_sync_barrier(e_last, HK_SYNC_GFX, HK_SYNC_COMPUTE);
   struct hk_cmd_buffer *e_members[] = {plain(1), e_last};
   packet = pack(e_members, 2, &cs);
   T("the last member's tail does not hold the packet back",
     !(cs->sync.dep[HK_SYNC_VDM].prior & HK_SYNC_GFX));
   T("the last member's tail still holds later command buffers",
     packet->sync.req[__builtin_ctz(HK_SYNC_COMPUTE)][HK_SYNC_VDM].prior &
        HK_SYNC_GFX);
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
   test_rejections();
   test_plan();
   test_dependencies();
   printf("%u tests, %u failures\n", tests, failures);
   ralloc_free(test_context);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

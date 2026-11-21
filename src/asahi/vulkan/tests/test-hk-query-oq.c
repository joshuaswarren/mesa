/* SPDX-License-Identifier: MIT
 * Calls the real deferred occlusion-availability resolver (hk_query_oq.c) and
 * hk_sync on mock control streams; no GPU or DRM calls.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/ralloc.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_query_oq.h"
#include "asahi/libagx/query.h"

static unsigned tests, failures;
static void *test_context;

#define T(name, cond)                                                          \
   do {                                                                        \
      tests++;                                                                 \
      if (!(cond)) {                                                           \
         failures++;                                                           \
         printf("FAIL %s line%d\n", name, __LINE__);                           \
      } else                                                                   \
         printf("PASS %s\n", name);                                            \
   } while (0)

/* hk_dispatch_imm_writes() emits the queued stores of a kept stream. The mock
 * records which stream it was called for and how many stores it carried. */
#define MAX_DISPATCH 8
static struct {
   struct hk_cs *cs;
   unsigned writes;
} dispatched[MAX_DISPATCH];
static unsigned ndispatched;

void
hk_dispatch_imm_writes(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   if (ndispatched == MAX_DISPATCH)
      abort();
   dispatched[ndispatched].cs = cs;
   dispatched[ndispatched].writes =
      util_dynarray_num_elements(&cs->imm_writes, struct libagx_imm_write);
   ndispatched++;
}

/* These devices never map memory at a second address, so the real
 * function adds no flush either.
 */
void
hk_cdm_end_stream_flush(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   (void)cs;
}

/* hk_sync.c references these for recording paths these fixtures never take. */
void
hk_flush_query_resets(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   (void)cs;
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
   (void)cmd;
   (void)why;
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

/* The resolver reads only the chip from the device. */
static enum agx_chip test_chip = AGX_CHIP_G17P;

static struct hk_cmd_buffer *
new_cmd(void)
{
   struct hk_cmd_buffer *c = rzalloc_size(test_context, sizeof(*c));
   struct hk_device *dev = rzalloc_size(test_context, sizeof(*dev));
   if (!c || !dev)
      abort();
   dev->dev.chip = test_chip;
   c->vk.base.device = &dev->vk;
   list_inithead(&c->control_streams);
   util_dynarray_init(&c->sync_pass_records, NULL);
   ndispatched = 0;
   return c;
}

/* Streams are heap-allocated like the driver's: a dropped stream is freed by
 * hk_cs_destroy(). */
static struct hk_cs *
add(struct hk_cmd_buffer *c, bool compute)
{
   struct hk_cs *s = calloc(1, sizeof(*s));
   if (!s)
      abort();
   s->type = compute ? HK_CS_CDM : HK_CS_VDM;
   list_addtail(&s->node, &c->control_streams);
   hk_sync_stream_created(c, s, &s, compute);
   return s;
}

static unsigned
nstreams(struct hk_cmd_buffer *c)
{
   return list_length(&c->control_streams);
}

static bool
listed(struct hk_cmd_buffer *c, const struct hk_cs *cs)
{
   list_for_each_entry(struct hk_cs, it, &c->control_streams, node) {
      if (it == cs)
         return true;
   }
   return false;
}

/* A post-graphics stream holding `count` availability stores for queries
 * ended by `render`, deferred as hk_cmd_buffer_end_compute_internal() would. */
static struct hk_cs *
add_post(struct hk_cmd_buffer *c, struct hk_cs *render, unsigned count,
         uint32_t first_handle)
{
   struct hk_cs *post = add(c, true);
   post->sync.dep[HK_SYNC_VDM].cut = render;
   post->imm_writes = UTIL_DYNARRAY_INIT;
   post->oq_stamps = UTIL_DYNARRAY_INIT;
   for (unsigned i = 0; i < count; ++i) {
      const struct libagx_imm_write write = {0};
      const struct hk_oq_stamp stamp = {
         .handle = first_handle + i,
         .offset_B = 8 * (first_handle + i),
         .render_seq = render->sync.seq,
      };
      util_dynarray_append(&post->imm_writes, write);
      util_dynarray_append(&post->oq_stamps, stamp);
   }
   return post;
}

static bool
deferred(struct hk_cmd_buffer *c, struct hk_cs *post)
{
   return hk_defer_oq_available(c, post) && post->oq_avail_deferred;
}

static bool
slot_is(const struct hk_cs *render, unsigned slot, uint32_t handle)
{
   return render->oq_stamp[slot].handle == handle &&
          render->oq_stamp[slot].offset_B == 8 * handle;
}

static void
test_defer(void)
{
   struct hk_cmd_buffer *c = new_cmd();
   struct hk_cs *render = add(c, false);

   struct hk_cs *post = add_post(c, render, 2, 1);
   T("defer accepts stores that all ride on one render", deferred(c, post));

   post = add_post(c, render, 0, 1);
   T("defer refuses a stream without availability stores",
     !hk_defer_oq_available(c, post));

   post = add_post(c, render, 1, 1);
   const struct libagx_imm_write extra = {0};
   util_dynarray_append(&post->imm_writes, extra);
   T("defer refuses other immediate writes", !hk_defer_oq_available(c, post));

   post = add_post(c, render, 1, 1);
   post->stats.cmds = 1;
   T("defer refuses a stream with commands", !hk_defer_oq_available(c, post));

   post = add_post(c, render, 1, 1);
   post->timestamp.end.handle = 9;
   T("defer refuses a stream an API timestamp rides on",
     !hk_defer_oq_available(c, post));

   post = add_post(c, render, 1, 1);
   post->query_reset_pending = true;
   T("defer refuses a stream with a pending query reset",
     !hk_defer_oq_available(c, post));

   struct hk_cs *other = add(c, false);
   post = add_post(c, render, 2, 1);
   ((struct hk_oq_stamp *)post->oq_stamps.data)[1].render_seq = other->sync.seq;
   T("defer refuses stores ended by different renders",
     !hk_defer_oq_available(c, post));
}

static void
test_stamped(void)
{
   struct hk_cmd_buffer *c = new_cmd();
   struct hk_cs *render = add(c, false);
   struct hk_cs *post = add_post(c, render, 1, 5);
   T("stamped fixture defers", deferred(c, post));
   const unsigned before = nstreams(c);
   hk_resolve_deferred_oq_available(c);
   T("stamped: the render carries the store in the vertex-end slot",
     slot_is(render, 1, 5));
   T("stamped: the firmware-written fragment-end slot stays free",
     render->oq_stamp[3].handle == 0);
   T("stamped: the stream is dropped", nstreams(c) == before - 1);
   T("stamped: nothing is dispatched", ndispatched == 0);

   /* API timestamps ride on vertex start (0) and fragment end (3): the stores
    * take the free slots in order 1, 2. */
   c = new_cmd();
   render = add(c, false);
   render->timestamp.start.handle = 40;
   render->timestamp.end.handle = 41;
   post = add_post(c, render, 2, 7);
   T("stamped with API timestamps fixture defers", deferred(c, post));
   hk_resolve_deferred_oq_available(c);
   T("stamped: API timestamp slots are left alone",
     render->oq_stamp[0].handle == 0 && render->oq_stamp[3].handle == 0);
   T("stamped: stores take slots 1 then 2",
     slot_is(render, 1, 7) && slot_is(render, 2, 8));
   T("stamped: no dispatch with API timestamps", ndispatched == 0);

   /* Another occlusion store already rides on the vertex-end slot. */
   c = new_cmd();
   render = add(c, false);
   render->oq_stamp[1] = (struct agx_timestamp_req){.handle = 11, .offset_B = 88};
   post = add_post(c, render, 1, 12);
   T("stamped with an occupied slot fixture defers", deferred(c, post));
   hk_resolve_deferred_oq_available(c);
   T("stamped: the store skips the occupied slot",
     render->oq_stamp[1].handle == 11 && slot_is(render, 2, 12));
   T("stamped: dropped with an occupied slot", !listed(c, post) && ndispatched == 0);

   /* Three host-written slots carry three stores, vertex start last. */
   c = new_cmd();
   render = add(c, false);
   post = add_post(c, render, 3, 20);
   T("stamped three fixture defers", deferred(c, post));
   hk_resolve_deferred_oq_available(c);
   T("stamped: three stores take slots 1, 2 then 0",
     slot_is(render, 1, 20) && slot_is(render, 2, 21) &&
        slot_is(render, 0, 22) && render->oq_stamp[3].handle == 0 &&
        !listed(c, post) && ndispatched == 0);

   /* A fourth store does not fit: the fragment end is not a candidate even
    * when free, so the stream keeps its compute store. */
   c = new_cmd();
   render = add(c, false);
   post = add_post(c, render, 4, 30);
   T("stamped four fixture defers", deferred(c, post));
   hk_resolve_deferred_oq_available(c);
   T("stamped: four stores never use the fragment-end slot",
     render->oq_stamp[3].handle == 0 && render->oq_stamp[1].handle == 0 &&
        ndispatched == 1 && dispatched[0].cs == post &&
        dispatched[0].writes == 4);
}

static void
test_keep(void)
{
   /* The stream became the tail an API timestamp rides on while deferred. */
   struct hk_cmd_buffer *c = new_cmd();
   struct hk_cs *render = add(c, false);
   struct hk_cs *post = add_post(c, render, 1, 1);
   T("keep (timestamp) fixture defers", deferred(c, post));
   post->timestamp.end.handle = 9;
   hk_resolve_deferred_oq_available(c);
   T("keep: a later API timestamp keeps the store",
     ndispatched == 1 && dispatched[0].cs == post && listed(c, post) &&
        render->oq_stamp[3].handle == 0);

   /* A later query reset (or any compute) waits on the deferred stream. */
   c = new_cmd();
   render = add(c, false);
   post = add_post(c, render, 1, 1);
   T("keep (reset) fixture defers", deferred(c, post));
   struct hk_cs *reset = add(c, true);
   reset->sync.dep[HK_SYNC_CDM].cut = post;
   hk_resolve_deferred_oq_available(c);
   T("keep: a later query reset waiting on the stream keeps the store",
     ndispatched == 1 && dispatched[0].cs == post && render->oq_stamp[3].handle == 0);

   /* Later compute waits on the render. */
   c = new_cmd();
   render = add(c, false);
   post = add_post(c, render, 1, 1);
   T("keep (render wait) fixture defers", deferred(c, post));
   struct hk_cs *later = add(c, true);
   later->sync.dep[HK_SYNC_VDM].cut = render;
   hk_resolve_deferred_oq_available(c);
   T("keep: compute waiting on the render keeps the store",
     ndispatched == 1 && dispatched[0].cs == post);

   /* The command buffer's tail requirement names the render. */
   c = new_cmd();
   render = add(c, false);
   post = add_post(c, render, 1, 1);
   T("keep (tail) fixture defers", deferred(c, post));
   c->sync.req[0][HK_SYNC_VDM].cut = render;
   hk_resolve_deferred_oq_available(c);
   T("keep: a tail requirement on the render keeps the store",
     ndispatched == 1 && dispatched[0].cs == post);

   /* The stream does not wait on the render that ended the queries. */
   c = new_cmd();
   render = add(c, false);
   struct hk_cs *next = add(c, false);
   post = add_post(c, render, 1, 1);
   post->sync.dep[HK_SYNC_VDM].cut = next;
   T("keep (wrong render) fixture defers", deferred(c, post));
   hk_resolve_deferred_oq_available(c);
   T("keep: a stream not waiting on its render keeps the store",
     ndispatched == 1 && dispatched[0].cs == post && render->oq_stamp[3].handle == 0);
}

/* Fill every slot of a render: API timestamps on 0 and 3, other stores on 1/2. */
static void
fill_slots(struct hk_cs *render, uint32_t base)
{
   render->timestamp.start.handle = base;
   render->timestamp.end.handle = base + 1;
   render->oq_stamp[1] = (struct agx_timestamp_req){.handle = base + 2};
   render->oq_stamp[2] = (struct agx_timestamp_req){.handle = base + 3};
}

static void
test_movable(void)
{
   /* More stores than free slots on the render, and nothing later: the stream
    * is MOVABLE and, with no later movable stream, dispatched at the end. */
   struct hk_cmd_buffer *c = new_cmd();
   struct hk_cs *r1 = add(c, false);
   r1->timestamp.end.handle = 300;
   r1->oq_stamp[1] = (struct agx_timestamp_req){.handle = 301};
   struct hk_cs *p1 = add_post(c, r1, 3, 1);
   T("short-of-slots fixture defers", deferred(c, p1));
   hk_resolve_deferred_oq_available(c);
   T("movable: too few free slots dispatches the stores, slots unchanged",
     ndispatched == 1 && dispatched[0].cs == p1 && dispatched[0].writes == 3 &&
        r1->oq_stamp[2].handle == 0 && r1->oq_stamp[0].handle == 0);

   /* Two full renders in order. The later stream is MOVABLE and stays, a
    * compute stream waiting on the later render; the earlier stream joins it,
    * so one dispatch carries both stores. */
   c = new_cmd();
   r1 = add(c, false);
   fill_slots(r1, 100);
   p1 = add_post(c, r1, 1, 1);
   struct hk_cs *r2 = add(c, false);
   fill_slots(r2, 200);
   struct hk_cs *p2 = add_post(c, r2, 1, 2);
   T("in-order movable fixtures defer", deferred(c, p1) && deferred(c, p2));
   hk_resolve_deferred_oq_available(c);
   T("movable in order: the earlier stream joins the movable one",
     ndispatched == 1 && dispatched[0].cs == p2 && dispatched[0].writes == 2 &&
        !listed(c, p1) && listed(c, p2));

   /* The earlier stream waits on the later render: nothing after it waits on
    * that render, so it is MOVABLE; the later stream (waiting on the earlier
    * render) is then kept because the movable one still waits past it. */
   c = new_cmd();
   r1 = add(c, false);
   fill_slots(r1, 100);
   r2 = add(c, false);
   fill_slots(r2, 200);
   p1 = add_post(c, r2, 1, 1);
   p2 = add_post(c, r1, 1, 2);
   T("out-of-order movable fixtures defer", deferred(c, p1) && deferred(c, p2));
   hk_resolve_deferred_oq_available(c);
   T("movable out of order: the kept stream goes first, the movable one last",
     ndispatched == 2 && dispatched[0].cs == p2 && dispatched[1].cs == p1 &&
        dispatched[0].writes == 1 && dispatched[1].writes == 1);

   /* A free render whose store was deferred after a movable stream that waits
    * on a later render: the movable stream counts as compute waiting past the
    * free render, so its store is kept too (in place), the movable one last. */
   c = new_cmd();
   r1 = add(c, false);
   r2 = add(c, false);
   fill_slots(r2, 200);
   p2 = add_post(c, r2, 1, 2);
   p1 = add_post(c, r1, 1, 1);
   T("movable-then-free fixtures defer", deferred(c, p2) && deferred(c, p1));
   hk_resolve_deferred_oq_available(c);
   T("a free render's store is kept behind a movable stream's later wait",
     r1->oq_stamp[3].handle == 0 && ndispatched == 2 && dispatched[0].cs == p1 &&
        dispatched[1].cs == p2);
}

/* Resolution runs from the end of the command buffer: a later stream whose
 * render carries its stores is gone before earlier streams are judged. */
static void
test_order(void)
{
   struct hk_cmd_buffer *c = new_cmd();
   struct hk_cs *r[3], *p[3];
   for (unsigned i = 0; i < 3; ++i) {
      r[i] = add(c, false);
      p[i] = add_post(c, r[i], 1, 1 + i);
   }
   T("three passes defer", deferred(c, p[0]) && deferred(c, p[1]) && deferred(c, p[2]));
   const unsigned before = nstreams(c);
   hk_resolve_deferred_oq_available(c);
   T("order: every render carries its own store",
     slot_is(r[0], 1, 1) && slot_is(r[1], 1, 2) && slot_is(r[2], 1, 3));
   T("order: all three streams dropped, nothing dispatched",
     nstreams(c) == before - 3 && ndispatched == 0);

   /* The last render is full: its stream stays as compute waiting on that
    * render, so the earlier render may not carry its store; the earlier
    * store joins the remaining stream instead of a compute command of its
    * own. */
   c = new_cmd();
   r[0] = add(c, false);
   p[0] = add_post(c, r[0], 1, 1);
   r[1] = add(c, false);
   fill_slots(r[1], 200);
   p[1] = add_post(c, r[1], 1, 2);
   T("free-then-full fixtures defer", deferred(c, p[0]) && deferred(c, p[1]));
   hk_resolve_deferred_oq_available(c);
   T("order: the earlier store joins the remaining later stream",
     r[0]->oq_stamp[3].handle == 0 && ndispatched == 1 &&
        dispatched[0].cs == p[1] && dispatched[0].writes == 2 &&
        !listed(c, p[0]));

   /* As above, but a later reset of the first query waits on its stream: the
    * store may not move past that reset, so it stays in place. */
   c = new_cmd();
   r[0] = add(c, false);
   p[0] = add_post(c, r[0], 1, 1);
   r[1] = add(c, false);
   fill_slots(r[1], 200);
   p[1] = add_post(c, r[1], 1, 2);
   T("free-then-full-reset fixtures defer", deferred(c, p[0]) && deferred(c, p[1]));
   struct hk_cs *waiter = add(c, true);
   waiter->sync.dep[HK_SYNC_CDM].cut = p[0];
   hk_resolve_deferred_oq_available(c);
   T("order: a store something waits on does not join a later stream",
     r[0]->oq_stamp[3].handle == 0 && ndispatched == 2 &&
        dispatched[0].cs == p[0] && dispatched[1].cs == p[1] &&
        listed(c, p[0]));

   /* Many queries in each of three passes, more than any render has slots
    * for: one compute command carries every store. */
   c = new_cmd();
   for (unsigned i = 0; i < 3; ++i) {
      r[i] = add(c, false);
      p[i] = add_post(c, r[i], 8, 1 + 8 * i);
   }
   T("full-passes fixtures defer",
     deferred(c, p[0]) && deferred(c, p[1]) && deferred(c, p[2]));
   hk_resolve_deferred_oq_available(c);
   T("order: stores of passes without enough slots share one dispatch",
     ndispatched == 1 && dispatched[0].cs == p[2] && dispatched[0].writes == 24 &&
        !listed(c, p[0]) && !listed(c, p[1]) && r[0]->oq_stamp[3].handle == 0 &&
        r[1]->oq_stamp[3].handle == 0 && r[2]->oq_stamp[3].handle == 0);

   /* A later query reset waits on the last stream: both stores stay. */
   c = new_cmd();
   r[0] = add(c, false);
   p[0] = add_post(c, r[0], 1, 1);
   r[1] = add(c, false);
   p[1] = add_post(c, r[1], 1, 2);
   T("reset-after-two fixtures defer", deferred(c, p[0]) && deferred(c, p[1]));
   struct hk_cs *reset = add(c, true);
   reset->sync.dep[HK_SYNC_CDM].cut = p[1];
   hk_resolve_deferred_oq_available(c);
   T("order: a later reset keeps both stores in place",
     ndispatched == 2 && dispatched[0].cs == p[1] && dispatched[1].cs == p[0] &&
        r[0]->oq_stamp[3].handle == 0 && r[1]->oq_stamp[3].handle == 0);

   /* A later reset of the first query only: the later render still carries its
    * store, the first stays. */
   c = new_cmd();
   r[0] = add(c, false);
   p[0] = add_post(c, r[0], 1, 1);
   r[1] = add(c, false);
   p[1] = add_post(c, r[1], 1, 2);
   T("reset-first fixtures defer", deferred(c, p[0]) && deferred(c, p[1]));
   reset = add(c, true);
   reset->sync.dep[HK_SYNC_CDM].cut = p[0];
   hk_resolve_deferred_oq_available(c);
   T("order: a reset of the first query keeps only the first store",
     slot_is(r[1], 1, 2) && !listed(c, p[1]) && ndispatched == 1 &&
        dispatched[0].cs == p[0]);
}

/* The other native AGX3 chips keep recording order: while an earlier stream is
 * judged, the later deferred stream still waits on its later render. */
static void
test_order_forward(void)
{
   test_chip = AGX_CHIP_G16G;
   struct hk_cmd_buffer *c = new_cmd();
   struct hk_cs *r[3], *p[3];
   for (unsigned i = 0; i < 3; ++i) {
      r[i] = add(c, false);
      p[i] = add_post(c, r[i], 1, 1 + i);
   }
   T("forward: three passes defer", deferred(c, p[0]) && deferred(c, p[1]) && deferred(c, p[2]));
   hk_resolve_deferred_oq_available(c);
   T("forward: only the last render carries its store",
     r[0]->oq_stamp[3].handle == 0 && r[1]->oq_stamp[3].handle == 0 &&
        slot_is(r[2], 1, 3) && !listed(c, p[2]));
   T("forward: the earlier stores are dispatched in order",
     ndispatched == 2 && dispatched[0].cs == p[0] && dispatched[1].cs == p[1]);
   test_chip = AGX_CHIP_G17P;
}

/* Only the G17 runtime (G17P and M5) publishes render timestamps at
 * retirement. G15 firmware writes the vertex-end slot when tiling finishes,
 * before the fragment work writes the counts: such a stamp is not
 * availability. */
static void
test_stamp_chips(void)
{
   T("chips: G17P and M5 renders carry availability",
     hk_oq_render_stamps_supported(AGX_CHIP_G17P) &&
        hk_oq_render_stamps_supported(AGX_CHIP_G17G) &&
        hk_oq_render_stamps_supported(AGX_CHIP_G17S));
   T("chips: G15 renders keep the queued store",
     !hk_oq_render_stamps_supported(AGX_CHIP_G15G) &&
        !hk_oq_render_stamps_supported(AGX_CHIP_G15S) &&
        !hk_oq_render_stamps_supported(AGX_CHIP_G15C));
   T("chips: other native chips keep the queued store",
     !hk_oq_render_stamps_supported(AGX_CHIP_G16G));
}

static void
test_join_moves_query_access_ranges(void)
{
   /* Reverse join (hk_oq_joins path): earlier free render's store joins into
    * later full render's movable stream, preserving moved imm_write_query_ranges. */
   test_chip = AGX_CHIP_G17P;
   struct hk_cmd_buffer *c = new_cmd();
   struct hk_cs *r0 = add(c, false);
   struct hk_cs *p0 = add_post(c, r0, 1, 1);
   hk_cs_note_imm_write_query_range(p0, 42, 64, 8);
   struct hk_cs *r1 = add(c, false);
   fill_slots(r1, 200);
   struct hk_cs *p1 = add_post(c, r1, 1, 2);
   hk_cs_note_imm_write_query_range(p1, 42, 128, 8);
   T("join-rev-free: fixtures defer", deferred(c, p0) && deferred(c, p1));
   hk_resolve_deferred_oq_available(c);
   bool found_moved = false, found_local = false;
   if (ndispatched == 1 && dispatched[0].cs == p1) {
      util_dynarray_foreach(&p1->imm_write_query_ranges,
                            struct hk_query_access_range, r) {
         if (r->handle == 42 && r->start_B == 64 && r->end_B == 72)
            found_moved = true;
         if (r->handle == 42 && r->start_B == 128 && r->end_B == 136)
            found_local = true;
      }
   }
   T("join-rev-free: moved store query range survives on destination stream",
     ndispatched == 1 && dispatched[0].cs == p1 && found_moved && found_local);

   /* Reverse join (both full renders): earlier stream joins later movable one
    * and propagates imm_write_query_untracked. */
   c = new_cmd();
   r0 = add(c, false);
   fill_slots(r0, 100);
   p0 = add_post(c, r0, 1, 1);
   hk_cs_note_imm_write_query_range(p0, 55, 256, 8);
   p0->imm_write_query_untracked = true;
   r1 = add(c, false);
   fill_slots(r1, 200);
   p1 = add_post(c, r1, 1, 2);
   hk_cs_note_imm_write_query_range(p1, 55, 512, 8);
   T("join-rev-full: fixtures defer", deferred(c, p0) && deferred(c, p1));
   hk_resolve_deferred_oq_available(c);
   found_moved = found_local = false;
   if (ndispatched == 1 && dispatched[0].cs == p1) {
      util_dynarray_foreach(&p1->imm_write_query_ranges,
                            struct hk_query_access_range, r) {
         if (r->handle == 55 && r->start_B == 256 && r->end_B == 264)
            found_moved = true;
         if (r->handle == 55 && r->start_B == 512 && r->end_B == 520)
            found_local = true;
      }
   }
   T("join-rev-full: moved store query range and untracked flag survive",
     ndispatched == 1 && dispatched[0].cs == p1 && found_moved && found_local &&
        p1->imm_write_query_untracked);
}

int
main(void)
{
   test_context = ralloc_context(NULL);
   test_defer();
   test_stamped();
   test_keep();
   test_movable();
   test_order();
   test_order_forward();
   test_stamp_chips();
   test_join_moves_query_access_ranges();
   ralloc_free(test_context);
   printf("%u tests, %u failures\n", tests, failures);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

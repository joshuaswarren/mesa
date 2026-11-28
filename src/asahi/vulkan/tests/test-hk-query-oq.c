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
#include "hk_entrypoints.h"
#include "hk_event.h"
#include "hk_query_oq.h"
#include "hk_queue.h"
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
static unsigned cs_allocs;

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

/* Timestamp copies, as hk_record_stamp_copies() received them. Only the
 * event fixtures allow copies; elsewhere emitting one is a bug. A copy from
 * address zero is what faults the GPU. */
static bool allow_copies;
static unsigned recorded_copies, null_copies;

void
hk_record_stamp_copies(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       const struct hk_stamp_copy *copies, unsigned count)
{
   (void)cmd;
   (void)cs;
   if (!allow_copies)
      abort();
   for (unsigned i = 0; i < count; ++i) {
      recorded_copies++;
      if (!copies[i].src.addr || copies[i].src.event)
         null_copies++;
   }
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
   (void)cmd; (void)address; (void)size; (void)value;
   if (!target || !*target)
      abort();
   (*target)->stats.cmds++;
}

struct agx_ptr
hk_pool_alloc_from_slow(struct hk_cmd_buffer *cmd, struct hk_uploader *uploader,
                         uint32_t size, uint32_t alignment,
                         enum hk_cmd_bo_type type)
{
   (void)cmd; (void)uploader; (void)type;
   /* A standalone G17P timestamp's scratch store (stamp_carrier_launch()). */
   return (struct agx_ptr){
      .cpu = rzalloc_size(test_context, size),
      .gpu = (uint64_t)++cs_allocs * alignment,
   };
}

struct agx_ptr
hk_pool_cs_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   (void)cmd;
   return (struct agx_ptr){
      .cpu = rzalloc_size(test_context, size),
      .gpu = (uint64_t)++cs_allocs * alignment,
   };
}

void
hk_drm_attachment_set_init(struct hk_drm_attachment_set *set)
{
   (void)set;
}

void
hk_flush_deferred_clears_why(struct hk_cmd_buffer *cmd, const char *why)
{
   (void)why;
   if (cmd->state.deferred_clear_count)
      abort();
}

/* Event stream commands are represented by their dispatch count. */
void
hk_queue_write(struct hk_cmd_buffer *cmd, uint64_t address, uint32_t value,
               bool after_gfx)
{
   (void)address;
   (void)value;
   if (after_gfx || !cmd->current_cs.cs)
      abort();
   cmd->current_cs.cs->stats.cmds++;
}

void
hk_g17p_meta_wait_addr(struct hk_cmd_buffer *cmd, uint64_t address,
                      uint32_t value)
{
   hk_queue_write(cmd, address, value, false);
}

bool
brl_discard_init_prepare(struct hk_cmd_buffer *cmd, const VkDependencyInfo *info,
                         uint8_t *dst_kinds)
{
   (void)cmd;
   (void)dst_kinds;
   if (info->imageMemoryBarrierCount)
      abort();
   return false;
}

void
brl_discard_init_record(struct hk_cmd_buffer *cmd, const VkDependencyInfo *info,
                        uint8_t dst_kinds)
{
   (void)cmd;
   (void)info;
   (void)dst_kinds;
   abort();
}

void
hk_ensure_cs_has_space(struct hk_cmd_buffer *cmd, struct hk_cs *cs, size_t space)
{
   (void)cmd;
   (void)cs;
   (void)space;
   abort();
}

bool
hk_cmd_alias_scope(struct hk_cmd_buffer *cmd)
{
   (void)cmd;
   return false;
}

bool
hk_cdm_visibility_flush_needed(const struct hk_device *dev,
                               const struct hk_cs *cs, bool memory)
{
   (void)dev;
   (void)cs;
   (void)memory;
   return false;
}

void
hk_cdm_cache_flush(struct hk_device *dev, struct hk_cs *cs)
{
   (void)dev;
   (void)cs;
}

void
hk_cdm_visibility_flush(struct hk_device *dev, struct hk_cs *cs, bool memory)
{
   (void)dev;
   (void)cs;
   (void)memory;
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
   c->vk.base.type = VK_OBJECT_TYPE_COMMAND_BUFFER;
   list_inithead(&c->control_streams);
   util_dynarray_init(&c->sync_pass_records, NULL);
   ndispatched = 0;
   cs_allocs = 0;
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

/* A finalized render with queries still staged inline, as after
 * vkCmdEndRendering with no post-graphics stream. */
static struct hk_cs *
add_inline_render(struct hk_cmd_buffer *cmd, unsigned count)
{
   struct hk_cs *render = add(cmd, false);
   for (unsigned i = 0; i < count; ++i) {
      const uint32_t offset = 8 * (i + 1);
      render->oq_pending[i] = (struct hk_oq_pending){
         .avail_addr = 0x10000 + offset,
         .handle = i + 1,
         .offset_B = offset,
         .avail_size = 8,
      };
   }
   render->oq_pending_count = count;
   cmd->state.oq_pending_render = render;
   return render;
}

static struct hk_cs *
last_cs(struct hk_cmd_buffer *cmd)
{
   return list_last_entry(&cmd->control_streams, struct hk_cs, node);
}

static void
test_inline_stamps(void)
{
   const unsigned order[] = {1, 2, 0};
   for (unsigned count = 1; count <= ARRAY_SIZE(order); ++count) {
      struct hk_cmd_buffer *cmd = new_cmd();
      struct hk_cs *render = add_inline_render(cmd, count);
      hk_resolve_pending_oq_available(cmd);

      for (unsigned i = 0; i < count; ++i)
         T("inline query uses the next free stamp", slot_is(render, order[i], i + 1));
      T("inline stamping allocates no post-graphics stream",
        cs_allocs == 0 && nstreams(cmd) == 1 && ndispatched == 0 &&
           !cmd->current_cs.post_gfx);
      T("inline staging is cleared", !cmd->state.oq_pending_render &&
                                        render->oq_pending_count == 0);
   }
}

static void
test_inline_fallback(bool start_timestamp)
{
   struct hk_cmd_buffer *cmd = new_cmd();
   const unsigned count = start_timestamp ? 3 : 1;
   struct hk_cs *render = add_inline_render(cmd, count);
   if (start_timestamp)
      render->timestamp.start.handle = 9;
   else
      cmd->sync.req[util_logbase2(HK_SYNC_COMPUTE)][HK_SYNC_VDM].cut = render;

   hk_resolve_pending_oq_available(cmd);
   struct hk_cs *post = last_cs(cmd);
   T("blocked inline stamping stages a post-graphics stream",
     cs_allocs == 1 && post != render && post->oq_avail_deferred &&
        post->sync.dep[HK_SYNC_VDM].cut == render &&
        !cmd->state.oq_pending_render && render->oq_pending_count == 0);

   hk_resolve_deferred_oq_available(cmd);
   T("blocked inline stamping dispatches all stores after the render",
     ndispatched == 1 && dispatched[0].cs == post &&
        dispatched[0].writes == count && nstreams(cmd) == 2);
}

static void
test_inline_copy(void)
{
   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_cs *render1 = add_inline_render(cmd, 1);
   cmd->sync.kind_override = HK_SYNC_ALL;
   struct hk_cs *copy = hk_query_command_cs(cmd);
   if (!copy)
      abort();
   copy->stats.cmds++;
   cmd->sync.kind_override = 0;
   hk_cmd_buffer_end_compute(cmd);

   struct hk_cs *post1 = list_first_entry(&render1->node, struct hk_cs, node);
   T("query copy waits for post1", post1 != copy && post1->oq_avail_deferred &&
                                     copy->sync.dep[HK_SYNC_CDM].cut == post1);

   struct hk_cs *render2 = add(cmd, false);
   struct hk_cs *post2 = add_post(cmd, render2, 4, 2);
   T("second render's four queries are deferred", deferred(cmd, post2));
   hk_resolve_deferred_oq_available(cmd);

   T("reverse resolution preserves post1 before the copy",
     listed(cmd, post1) && post1->node.next == &copy->node &&
        copy->sync.dep[HK_SYNC_CDM].cut == post1);
   T("reverse resolution keeps the stores of each render in its own dispatch",
     ndispatched == 2 && dispatched[0].cs == post1 && dispatched[0].writes == 1 &&
        dispatched[1].cs == post2 && dispatched[1].writes == 4);
}

static void
test_inline_transition(void)
{
   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_cs *render = add_inline_render(cmd, 1);
   brl_sync_transition_prepare(cmd, HK_SYNC_GFX);
   T("layout transition waits for the render, not query availability",
     cmd->sync.brl_transition_dep[HK_SYNC_VDM].cut == render &&
        cmd->sync.brl_transition_dep[HK_SYNC_CDM].cut == NULL);

   /* Beginning a transition flushes inline availability before setting
    * brl_transition_active, so the availability stream is never tagged as a
    * transition stream and brl_sync_transition_end() does not order later
    * graphics work behind it. */
   brl_sync_transition_begin(cmd);
   struct hk_cs *post = last_cs(cmd);
   brl_sync_transition_end(cmd, HK_SYNC_GFX);
   T("transition begin flushes inline availability outside the transition",
     post != render && post->oq_avail_deferred && !post->sync.brl_transition &&
        (post->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP &&
        cmd->sync.req[util_logbase2(HK_SYNC_GFX)][HK_SYNC_CDM].cut == NULL);
}

static void
test_inline_event(bool wait)
{
   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_cs *render = add_inline_render(cmd, 1);
   struct agx_va va = {.addr = 0x20000};
   struct agx_bo bo = {.va = &va};
   struct hk_event event = {
      .base.type = VK_OBJECT_TYPE_EVENT,
      .bo = &bo,
      .addr = va.addr,
   };
   const VkEvent handle = hk_event_to_handle(&event);
   const VkMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
   };
   const VkDependencyInfo info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier,
   };
   if (wait)
      hk_CmdWaitEvents2(hk_cmd_buffer_to_handle(cmd), 1, &handle, &info);
   else
      hk_CmdResetEvent2(hk_cmd_buffer_to_handle(cmd), handle, barrier.srcStageMask);

   struct hk_cs *post = list_first_entry(&render->node, struct hk_cs, node);
   struct hk_cs *event_cs = last_cs(cmd);
   if (wait) {
      T("event wait waits for the render, not post-graphics availability",
        post != render && post != event_cs && post->oq_avail_deferred &&
           event_cs->sync.dep[HK_SYNC_VDM].cut == render &&
           event_cs->sync.dep[HK_SYNC_CDM].cut == NULL);
   } else {
      T("event write waits for post-graphics availability",
        post != render && post != event_cs && post->oq_avail_deferred &&
           event_cs->sync.dep[HK_SYNC_CDM].cut == post);
   }
}

static void
test_inline_pending_timestamp(void)
{
   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_cs *render = add_inline_render(cmd, 1);
   cmd->sync.pending_start_ts = (struct agx_timestamp_req){.handle = 9, .offset_B = 16};
   cmd->sync.pending_start_ts_unordered = true;
   hk_sync_flush_pending_timestamp(cmd);

   struct hk_cs *post = last_cs(cmd);
   T("flushed timestamp reaches the end of the post-graphics stream",
     post != render && !post->timestamp.start.handle &&
        post->timestamp.end.handle == 9 && post->timestamp.end.offset_B == 16);
   T("flushing clears the pending timestamp and its unordered flag",
     !cmd->sync.pending_start_ts.handle && !cmd->sync.pending_start_ts_unordered &&
        !cmd->sync.has_unordered_start_ts);
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
   test_chip = AGX_CHIP_G15G;
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

/* Native HAL200 (G16G/G17P) and M5 (G17G/G17S/G17C) runtimes publish render
 * timestamps at full retirement. G15 firmware writes the vertex-end slot when
 * tiling finishes, before the fragment work writes the counts: such a stamp is
 * not availability. */
static void
test_stamp_chips(void)
{
   T("chips: native HAL200, M5, and G18G renders carry availability",
     hk_oq_render_stamps_supported(AGX_CHIP_G16G) &&
        hk_oq_render_stamps_supported(AGX_CHIP_G17P) &&
        hk_oq_render_stamps_supported(AGX_CHIP_G17G) &&
        hk_oq_render_stamps_supported(AGX_CHIP_G17S) &&
        hk_oq_render_stamps_supported(AGX_CHIP_G18G));
   T("chips: G15 renders keep the queued store",
     !hk_oq_render_stamps_supported(AGX_CHIP_G15G) &&
        !hk_oq_render_stamps_supported(AGX_CHIP_G15S) &&
        !hk_oq_render_stamps_supported(AGX_CHIP_G15C));
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

static void
test_no_query_access_and_m5_reverse(void)
{
   /* M5 uses reverse resolution just like HAL200. */
   test_chip = AGX_CHIP_G17S;
   struct hk_cmd_buffer *c = new_cmd();
   struct hk_cs *r[3], *p[3];
   for (unsigned i = 0; i < 3; ++i) {
      r[i] = add(c, false);
      p[i] = add_post(c, r[i], 1, 1 + i);
   }
   T("m5-rev: three passes defer",
     deferred(c, p[0]) && deferred(c, p[1]) && deferred(c, p[2]));
   hk_resolve_deferred_oq_available(c);
   T("m5-rev: every render carries its own store",
     slot_is(r[0], 1, 1) && slot_is(r[1], 1, 2) && slot_is(r[2], 1, 3) &&
        ndispatched == 0 && !listed(c, p[0]) && !listed(c, p[1]) && !listed(c, p[2]));

   /* With COMPUTE_NO_QUERY_ACCESS, a barrier between renders or at CB tail
    * does not force a standalone LIBAGX_WRITE_U32S compute stream when the
    * render has free retirement stamp slots and the VDM dependency covers it. */
   c = new_cmd();
   hk_cmd_buffer_device(c)->dev.params.features |=
      DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
   struct hk_cs *r1 = add(c, false);
   struct hk_cs *p1 = add_post(c, r1, 1, 11);
   T("nqa-r2r: post defers", deferred(c, p1));
   struct hk_cs *r2 = add(c, false);
   r2->sync.dep[HK_SYNC_VDM].cut = r1;
   r2->sync.dep[HK_SYNC_CDM].cut = p1;
   r2->sync.dep_ta[HK_SYNC_VDM].cut = r1;
   r2->sync.dep_ta[HK_SYNC_CDM].cut = p1;
   c->sync.req[util_logbase2(HK_SYNC_GFX)][HK_SYNC_VDM].cut = r1;
   c->sync.req[util_logbase2(HK_SYNC_GFX)][HK_SYNC_CDM].cut = p1;
   hk_resolve_deferred_oq_available(c);
   T("nqa-r2r: stamps on render and drops post between renders",
     slot_is(r1, 1, 11) && !listed(c, p1) && ndispatched == 0 &&
        r2->sync.dep[HK_SYNC_VDM].cut == r1 && r2->sync.dep[HK_SYNC_CDM].cut == NULL);

   /* With COMPUTE_NO_QUERY_ACCESS, a later non-query compute waiting on render
    * still allows stamping; a later query-accessing compute keeps post. */
   c = new_cmd();
   hk_cmd_buffer_device(c)->dev.params.features |=
      DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
   r1 = add(c, false);
   p1 = add_post(c, r1, 1, 12);
   T("nqa-cmp: post defers", deferred(c, p1));
   struct hk_cs *cmp = add(c, true);
   cmp->sync.dep[HK_SYNC_VDM].cut = r1;
   cmp->sync.dep[HK_SYNC_CDM].cut = p1;
   hk_resolve_deferred_oq_available(c);
   T("nqa-cmp: non-query compute allows render stamping",
     slot_is(r1, 1, 12) && !listed(c, p1) && ndispatched == 0 &&
        cmp->sync.dep[HK_SYNC_VDM].cut == r1 && cmp->sync.dep[HK_SYNC_CDM].cut == NULL);

   c = new_cmd();
   hk_cmd_buffer_device(c)->dev.params.features |=
      DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
   r1 = add(c, false);
   p1 = add_post(c, r1, 1, 13);
   T("nqa-qcopy: post defers", deferred(c, p1));
   struct hk_cs *qcopy = add(c, true);
   qcopy->query_access = true;
   qcopy->sync.dep[HK_SYNC_VDM].cut = r1;
   qcopy->sync.dep[HK_SYNC_CDM].cut = p1;
   hk_resolve_deferred_oq_available(c);
   T("nqa-qcopy: query-accessing compute keeps post",
     r1->oq_stamp[1].handle == 0 && listed(c, p1) && ndispatched == 1 &&
        dispatched[0].cs == p1);

   /* With COMPUTE_NO_QUERY_ACCESS, inline pending OQ stamps at EndCommandBuffer
    * even when a tail barrier names the render. */
   c = new_cmd();
   hk_cmd_buffer_device(c)->dev.params.features |=
      DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
   r1 = add_inline_render(c, 1);
   c->sync.req[util_logbase2(HK_SYNC_COMPUTE)][HK_SYNC_VDM].cut = r1;
   hk_resolve_pending_oq_available(c);
   T("nqa-inline: tail barrier still stamps inline on render",
     slot_is(r1, 1, 1) && cs_allocs == 0 && !c->state.oq_pending_render);

   /* Without VDM coverage of render, a CDM-only waiter on post keeps post. */
   c = new_cmd();
   hk_cmd_buffer_device(c)->dev.params.features |=
      DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
   r1 = add(c, false);
   p1 = add_post(c, r1, 1, 14);
   T("nqa-uncovered: post defers", deferred(c, p1));
   cmp = add(c, true);
   cmp->sync.dep[HK_SYNC_CDM].cut = p1;
   hk_resolve_deferred_oq_available(c);
   T("nqa-uncovered: CDM-only waiter without VDM edge keeps post",
     r1->oq_stamp[1].handle == 0 && listed(c, p1) && ndispatched == 1);

   test_chip = AGX_CHIP_G17P;
}

static void
test_oq_post_gfx_stamp_kind(void)
{
   /* A post-graphics query availability stream gets HK_SYNC_STAMP: pipeline
    * barriers (even ALL_COMMANDS -> ALL_COMMANDS) do not stall subsequent
    * renders or computes behind it, while query commands still order after it
    * both in-CB and across CBs. */
   struct hk_cmd_buffer *cb1 = new_cmd();
   struct hk_cs *r1 = add_inline_render(cb1, 1);
   hk_sync_barrier(cb1, HK_SYNC_ALL, HK_SYNC_ALL);
   struct hk_cs *post = last_cs(cb1);
   T("post_gfx OQ availability carries HK_SYNC_STAMP only",
     post != r1 && post->oq_avail_deferred &&
        (post->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP);

   struct hk_cs *r2 = add(cb1, false);
   T("subsequent render waits on r1, not post_gfx OQ availability",
     r2->sync.dep[HK_SYNC_VDM].cut == r1 &&
        r2->sync.dep[HK_SYNC_CDM].cut == NULL &&
        r2->sync.dep_ta[HK_SYNC_CDM].cut == NULL);

   struct hk_cs *comp = add(cb1, true);
   T("subsequent compute waits on r1, not post_gfx OQ availability",
     comp->sync.dep[HK_SYNC_VDM].cut == r1 &&
        comp->sync.dep[HK_SYNC_CDM].cut == NULL);

   /* Verify directly without an intervening compute stream that an in-CB
    * query command (copy / reset) still waits on HK_SYNC_STAMP post_gfx. */
   struct hk_cmd_buffer *cb_direct = new_cmd();
   struct hk_cs *rd = add_inline_render(cb_direct, 1);
   hk_sync_barrier(cb_direct, HK_SYNC_ALL, HK_SYNC_ALL);
   struct hk_cs *post_d = last_cs(cb_direct);
   struct hk_cs *rd2 = add(cb_direct, false);
   cb_direct->sync.kind_override = HK_SYNC_ALL;
   struct hk_cs *copy_d = hk_query_command_cs(cb_direct);
   if (!copy_d)
      abort();
   copy_d->stats.cmds++;
   cb_direct->sync.kind_override = 0;
   hk_cmd_buffer_end_compute(cb_direct);
   T("in-CB query command directly waits on HK_SYNC_STAMP post_gfx",
     rd2->sync.dep[HK_SYNC_CDM].cut == NULL &&
        copy_d->sync.dep[HK_SYNC_CDM].cut == post_d &&
        rd != rd2);

   /* Cross-CB submission: cb_a ends with post_gfx OQ + ALL_COMMANDS barrier;
    * cb_b runs a render (must NOT wait on post_gfx) and cb_c runs a query
    * command (MUST wait on post_gfx). */
   struct hk_cmd_buffer *cb_a = new_cmd();
   struct hk_cs *ra = add_inline_render(cb_a, 1);
   hk_sync_barrier(cb_a, HK_SYNC_ALL, HK_SYNC_ALL);
   struct hk_cs *post_a = last_cs(cb_a);
   hk_resolve_deferred_oq_available(cb_a);

   struct hk_cmd_buffer *cb_b = new_cmd();
   struct hk_cs *rb = add(cb_b, false);

   struct hk_cmd_buffer *cb_c = new_cmd();
   cb_c->sync.kind_override = HK_SYNC_ALL;
   struct hk_cs *copy_c = hk_query_command_cs(cb_c);
   if (!copy_c)
      abort();
   copy_c->stats.cmds++;
   cb_c->sync.kind_override = 0;
   hk_cmd_buffer_end_compute(cb_c);

   struct hk_queue q = {0};
   q.sync.exact_prior = true;
   struct hk_cmd_buffer *cbs[] = {cb_a, cb_b, cb_c};
   T("cross-CB resolve succeeds",
     hk_sync_resolve_submit(&q, cbs, ARRAY_SIZE(cbs), false, false, 0, NULL));
   T("cross-CB render waits on ra, not post_a",
     rb->sync.barrier[HK_SYNC_VDM] == ra->sync.index &&
        rb->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE &&
        rb->sync.prior[HK_SYNC_CDM] == 0);
   T("cross-CB query command waits on post_a",
     copy_c->sync.barrier[HK_SYNC_CDM] == post_a->sync.index);
}

static void
test_oq_post_gfx_heap_promotion(void)
{
   /* When a render pass opens post_gfx for query availability (HK_SYNC_STAMP)
    * and a later draw in the same pass uses the geometry heap,
    * hk_sync_heap_used() promotes post_gfx to HK_SYNC_GFX so subsequent
    * barriers and heap passes wait for the heap reset. */
   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_cs *r1 = add_inline_render(cmd, 1);
   cmd->current_cs.gfx = r1;
   hk_flush_oq_pending(cmd);
   struct hk_cs *post = cmd->current_cs.post_gfx;
   T("heap promotion setup: post_gfx starts as HK_SYNC_STAMP",
     post != NULL && (post->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP);

   hk_sync_heap_used(cmd);
   T("hk_sync_heap_used promotes post_gfx to HK_SYNC_GFX | HK_SYNC_GFX_MEM",
     (post->sync.kind & HK_SYNC_GFX) != 0 &&
        (post->sync.kind & HK_SYNC_GFX_MEM) != 0 &&
        cmd->sync.last[util_logbase2(HK_SYNC_GFX)][HK_SYNC_CDM] == post);

   cmd->current_cs.gfx = NULL;
   cmd->current_cs.post_gfx = NULL;
   hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_GFX);
   struct hk_cs *r2 = add(cmd, false);
   T("subsequent render after promoted heap post_gfx waits on post_gfx",
     r2->sync.dep[HK_SYNC_CDM].cut == post);
}

static void
test_event_retirement_stamping(void)
{
   /* 1. Render followed by vkCmdSetEvent2 on G17P with COMPUTE_NO_QUERY_ACCESS:
    *    event rides on the render's vertex-end retirement slot (slot 1) with
    *    no standalone CDM stream, and a subsequent render barrier does not
    *    acquire a CDM wait. */
   {
      test_chip = AGX_CHIP_G17P;
      struct hk_cmd_buffer *cmd = new_cmd();
      hk_cmd_buffer_device(cmd)->dev.params.features |=
         DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
      struct hk_cs *r1 = add(cmd, false);
      uint64_t status = 0;
      struct agx_va va = {.addr = 0x30000};
      struct agx_bo bo = {.va = &va};
      struct hk_event event = {
         .base.type = VK_OBJECT_TYPE_EVENT,
         .bo = &bo,
         .ts_handle = 42,
         .ts_offset_B = 16,
         .addr = va.addr,
         .status = &status,
      };
      const VkMemoryBarrier2 bar = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
      };
      const VkDependencyInfo info = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .memoryBarrierCount = 1,
         .pMemoryBarriers = &bar,
      };
      hk_CmdSetEvent2(hk_cmd_buffer_to_handle(cmd), hk_event_to_handle(&event),
                      &info);

      T("stamped render SetEvent creates no extra CDM stream",
        last_cs(cmd) == r1 && r1->oq_stamp[1].handle == 42 &&
           r1->oq_stamp[1].offset_B == 16 && !r1->oq_stamp[3].handle);
      T("stamped render SetEvent records HK_SYNC_STAMP without poisoning kind",
        cmd->sync.last[util_logbase2(HK_SYNC_STAMP)][HK_SYNC_VDM] == r1 &&
           !(r1->sync.kind & HK_SYNC_STAMP));

      /* Follow with a pipeline barrier and second render: zero CDM dependency. */
      hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_GFX);
      struct hk_cs *r2 = add(cmd, false);
      T("render after stamped SetEvent has no CDM stall",
        r2->sync.dep[HK_SYNC_CDM].cut == NULL);

      /* Follow with ResetEvent(TOP_OF_PIPE): must wait on r1 and record query
       * access range covering the stamped slot. */
      hk_CmdResetEvent2(hk_cmd_buffer_to_handle(cmd),
                        hk_event_to_handle(&event),
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
      struct hk_cs *reset_cs = last_cs(cmd);
      T("ResetEvent after stamped render waits on render and notes query range",
        reset_cs != r2 && reset_cs->type == HK_CS_CDM &&
           reset_cs->sync.dep[HK_SYNC_VDM].cut == r1 &&
           reset_cs->query_access_ranges.size ==
              sizeof(struct hk_query_access_range));
   }

   /* 2. Compute followed by vkCmdSetEvent2 on G17P: event rides on compute
    *    ts.end, ends the compute stream, and does not set untracked query
    *    access on the compute stream. */
   {
      test_chip = AGX_CHIP_G17P;
      struct hk_cmd_buffer *cmd = new_cmd();
      hk_cmd_buffer_device(cmd)->dev.params.features |=
         DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
      struct hk_cs *c1 = add(cmd, true);
      c1->stats.cmds = 1;
      cmd->current_cs.cs = c1;
      uint64_t status = 0;
      struct agx_va va = {.addr = 0x40000};
      struct agx_bo bo = {.va = &va};
      struct hk_event event = {
         .base.type = VK_OBJECT_TYPE_EVENT,
         .bo = &bo,
         .ts_handle = 77,
         .ts_offset_B = 24,
         .addr = va.addr,
         .status = &status,
      };
      const VkMemoryBarrier2 bar = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      };
      const VkDependencyInfo info = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .memoryBarrierCount = 1,
         .pMemoryBarriers = &bar,
      };
      hk_CmdSetEvent2(hk_cmd_buffer_to_handle(cmd), hk_event_to_handle(&event),
                      &info);

      T("stamped compute SetEvent attaches to ts.end and closes stream",
        last_cs(cmd) == c1 && cmd->current_cs.cs == NULL &&
           c1->timestamp.end.handle == 77 && c1->timestamp.end.offset_B == 24 &&
           !c1->query_access_untracked);

      /* WaitEvents2 on the stamped event waits on c1 and notes its range. */
      const VkEvent eh = hk_event_to_handle(&event);
      hk_CmdWaitEvents2(hk_cmd_buffer_to_handle(cmd), 1, &eh, &info);
      struct hk_cs *wait_cs = last_cs(cmd);
      T("WaitEvents2 on stamped compute waits on c1 and notes query range",
        wait_cs != c1 && wait_cs->sync.dep[HK_SYNC_CDM].cut == c1 &&
           wait_cs->query_access_ranges.size ==
              sizeof(struct hk_query_access_range));
   }

   /* 3. Fallback when all three render retirement slots are already taken. */
   {
      test_chip = AGX_CHIP_G17P;
      struct hk_cmd_buffer *cmd = new_cmd();
      hk_cmd_buffer_device(cmd)->dev.params.features |=
         DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
      struct hk_cs *r1 = add(cmd, false);
      r1->oq_stamp[0] = (struct agx_timestamp_req){.handle = 1, .offset_B = 0};
      r1->oq_stamp[1] = (struct agx_timestamp_req){.handle = 1, .offset_B = 8};
      r1->oq_stamp[2] = (struct agx_timestamp_req){.handle = 1, .offset_B = 16};
      uint64_t status = 0;
      struct agx_va va = {.addr = 0x50000};
      struct agx_bo bo = {.va = &va};
      struct hk_event event = {
         .base.type = VK_OBJECT_TYPE_EVENT,
         .bo = &bo,
         .ts_handle = 99,
         .ts_offset_B = 32,
         .addr = va.addr,
         .status = &status,
      };
      const VkMemoryBarrier2 bar = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      };
      const VkDependencyInfo info = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .memoryBarrierCount = 1,
         .pMemoryBarriers = &bar,
      };
      hk_CmdSetEvent2(hk_cmd_buffer_to_handle(cmd), hk_event_to_handle(&event),
                      &info);
      struct hk_cs *ev_cs = last_cs(cmd);
      T("full render slots fall back to CDM event write without using slot 3",
        ev_cs != r1 && ev_cs->type == HK_CS_CDM && !r1->oq_stamp[3].handle &&
           ev_cs->sync.dep[HK_SYNC_VDM].cut == r1);
   }

   /* 4. M5 stamped render SetEvent + ResetEvent clears both 32-bit halves of
    *    the 64-bit event slot so upper timestamp bits cannot stick. */
   {
      test_chip = AGX_CHIP_G17G;
      struct hk_cmd_buffer *cmd = new_cmd();
      hk_cmd_buffer_device(cmd)->dev.params.features |=
         DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
      struct hk_cs *r1 = add(cmd, false);
      uint64_t status = 0;
      struct agx_va va = {.addr = 0x60000};
      struct agx_bo bo = {.va = &va};
      struct hk_event event = {
         .base.type = VK_OBJECT_TYPE_EVENT,
         .bo = &bo,
         .ts_handle = 55,
         .ts_offset_B = 8,
         .addr = va.addr,
         .status = &status,
      };
      const VkMemoryBarrier2 bar = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
      };
      const VkDependencyInfo info = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .memoryBarrierCount = 1,
         .pMemoryBarriers = &bar,
      };
      hk_CmdSetEvent2(hk_cmd_buffer_to_handle(cmd), hk_event_to_handle(&event),
                      &info);
      T("M5 stamped render SetEvent attaches to slot 1",
        last_cs(cmd) == r1 && r1->oq_stamp[1].handle == 55 &&
           r1->oq_stamp[1].offset_B == 8);

      hk_CmdResetEvent2(hk_cmd_buffer_to_handle(cmd),
                        hk_event_to_handle(&event),
                        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);
      struct hk_cs *reset_cs = last_cs(cmd);
      T("M5 ResetEvent writes both 32-bit halves and notes query range",
        reset_cs != r1 && reset_cs->type == HK_CS_CDM &&
           reset_cs->stats.cmds == 2 &&
           reset_cs->sync.dep[HK_SYNC_VDM].cut == r1 &&
           reset_cs->query_access_ranges.size ==
              sizeof(struct hk_query_access_range));
   }
}

static struct hk_event
test_event(uint64_t *status, struct agx_bo *bo, uint32_t handle,
           uint32_t offset_B)
{
   return (struct hk_event){
      .base.type = VK_OBJECT_TYPE_EVENT,
      .bo = bo,
      .ts_handle = handle,
      .ts_offset_B = offset_B,
      .addr = bo->va->addr + offset_B,
      .status = status,
   };
}

static void
set_event(struct hk_cmd_buffer *cmd, struct hk_event *event,
          VkPipelineStageFlags2 src)
{
   const VkMemoryBarrier2 bar = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = src,
   };
   const VkDependencyInfo info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &bar,
   };
   hk_CmdSetEvent2(hk_cmd_buffer_to_handle(cmd), hk_event_to_handle(event),
                   &info);
}

static struct agx_timestamp_req
ts_query(uint32_t q)
{
   return (struct agx_timestamp_req){
      .addr = 0x90000 + 8 * q, .handle = 3, .offset_B = 8 * q};
}

/* An event's retirement stamp is no timestamp query: a later
 * vkCmdWriteTimestamp2 never copies it. The copy would read address zero
 * (the event request carries no address), and a reset of the event could
 * clear the word before the copy read it. vkCmdCopyQueryPoolResults records
 * into a query stream (hk_query_command_cs()) that waits for the whole
 * prefix, query resets included, so the timestamp after the event is
 * otherwise covered by that stream's end stamp (stamp_covered()). */
static void
test_event_stamp_not_copied(void)
{
   const enum agx_chip chips[] = {AGX_CHIP_G17P, AGX_CHIP_G17G};
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned exact = 0; exact < 2; ++exact) {
         for (unsigned control = 0; control < 2; ++control) {
            test_chip = chips[c];
            struct hk_cmd_buffer *cmd = new_cmd();
            hk_cmd_buffer_device(cmd)->dev.params.features |=
               DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS |
               (exact ? DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS : 0);
            allow_copies = true;
            recorded_copies = null_copies = 0;

            /* vkCmdWriteTimestamp2(q0), vkCmdCopyQueryPoolResults(q0) */
            hk_sync_write_timestamp(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                    ts_query(0));
            cmd->sync.kind_override = HK_SYNC_ALL;
            struct hk_cs *copy = hk_query_command_cs(cmd);
            copy->stats.cmds++;
            cmd->sync.kind_override = 0;

            uint64_t status = 0;
            struct agx_va va = {.addr = 0x70000};
            struct agx_bo bo = {.va = &va};
            struct hk_event event = test_event(&status, &bo, 61, 40);
            if (control) {
               /* The same sequence with a timestamp in place of the event:
                * the next timestamp copies it, so the fixture reaches the
                * coalescing path. */
               hk_sync_write_timestamp(
                  cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, ts_query(4));
               hk_sync_write_timestamp(
                  cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, ts_query(5));
               T("control: a timestamp after a query copy coalesces",
                 copy->timestamp.end.addr == ts_query(4).addr &&
                    cmd->sync.stamp_copy_count == 1 &&
                    cmd->sync.stamp_copies[0].src.addr == ts_query(4).addr);
               hk_sync_flush_stamp_copies(cmd);
               T("control: the copy reads the query", recorded_copies == 1 &&
                                                         !null_copies);
               continue;
            }

            set_event(cmd, &event, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
            T("event rides on the query copy stream's end stamp",
              copy->timestamp.end.handle == 61 && !copy->timestamp.end.addr);

            hk_sync_write_timestamp(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                    ts_query(5));
            const bool pending_null =
               cmd->sync.stamp_copy_count &&
               !cmd->sync.stamp_copies[0].src.addr;
            hk_sync_flush_stamp_copies(cmd);
            struct hk_cs *ts = last_cs(cmd);
            T("timestamp after a stamped event never copies address zero",
              !pending_null && null_copies == 0 && recorded_copies == 0);
            T("timestamp after a stamped event gets a command of its own",
              ts != copy && ts->type == HK_CS_CDM &&
                 ts->timestamp.end.addr == ts_query(5).addr &&
                 !ts->timestamp.end.event);

            /* A second timestamp still copies the first: queries coalesce. */
            hk_sync_write_timestamp(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                    ts_query(6));
            T("a later timestamp still copies the query stamp",
              cmd->sync.stamp_copy_count == 1 &&
                 cmd->sync.stamp_copies[0].src.addr == ts_query(5).addr);
            hk_sync_flush_stamp_copies(cmd);
            T("no copy reads address zero",
              recorded_copies == 1 && !null_copies);
         }
      }
   }
   allow_copies = false;
}

/* The kernel publishes a render's timestamps in slot order. An event set
 * after a timestamp query on the same render rides only on a later slot, so
 * a host that sees the event set also sees the query. */
static void
test_event_after_render_query(void)
{
   uint64_t status = 0;
   struct agx_va va = {.addr = 0x80000};
   struct agx_bo bo = {.va = &va};

   /* Fragment-end timestamp query (slot 3), then SetEvent: no free later
    * slot, so a compute stream writes the status after the render. */
   {
      test_chip = AGX_CHIP_G17P;
      struct hk_cmd_buffer *cmd = new_cmd();
      hk_cmd_buffer_device(cmd)->dev.params.features |=
         DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
      struct hk_cs *r = add(cmd, false);
      hk_sync_write_timestamp(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                              ts_query(1));
      T("fragment-end query rides on the render",
        last_cs(cmd) == r && r->timestamp.end.addr == ts_query(1).addr);

      struct hk_event event = test_event(&status, &bo, 62, 48);
      set_event(cmd, &event, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
      struct hk_cs *ev = last_cs(cmd);
      bool in_render = false;
      for (unsigned i = 0; i < 4; ++i)
         in_render |= r->oq_stamp[i].handle == 62;
      T("event after a fragment-end query takes no render slot",
        !in_render && ev != r && ev->type == HK_CS_CDM &&
           ev->sync.dep[HK_SYNC_VDM].cut == r &&
           ev->query_access_untracked);
   }

   /* Vertex-end timestamp query (slot 1): the event takes fragment start. */
   {
      test_chip = AGX_CHIP_G17P;
      struct hk_cmd_buffer *cmd = new_cmd();
      hk_cmd_buffer_device(cmd)->dev.params.features |=
         DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
      struct hk_cs *r = add(cmd, false);
      r->timestamp.vtx_end = ts_query(2);
      struct hk_event event = test_event(&status, &bo, 63, 56);
      set_event(cmd, &event, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
      T("event after a vertex-end query takes fragment start",
        last_cs(cmd) == r && r->oq_stamp[2].handle == 63 &&
           !r->oq_stamp[0].handle && !r->oq_stamp[1].handle);
   }

   /* Occlusion availability in fragment start (slot 2): only slot 3 is
    * later, which the event never takes. */
   {
      test_chip = AGX_CHIP_G17P;
      struct hk_cmd_buffer *cmd = new_cmd();
      hk_cmd_buffer_device(cmd)->dev.params.features |=
         DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
      struct hk_cs *r = add(cmd, false);
      r->oq_stamp[2] = (struct agx_timestamp_req){.handle = 4, .offset_B = 8};
      struct hk_event event = test_event(&status, &bo, 64, 64);
      set_event(cmd, &event, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
      T("event after fragment-start availability takes no render slot",
        last_cs(cmd) != r && !r->oq_stamp[0].handle &&
           !r->oq_stamp[1].handle && !r->oq_stamp[3].handle);
   }

   /* SetEvent, a fragment-end query, SetEvent again: the second set must not
    * reuse the event's earlier slot. */
   {
      test_chip = AGX_CHIP_G17P;
      struct hk_cmd_buffer *cmd = new_cmd();
      hk_cmd_buffer_device(cmd)->dev.params.features |=
         DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
      struct hk_cs *r = add(cmd, false);
      struct hk_event event = test_event(&status, &bo, 65, 72);
      set_event(cmd, &event, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
      T("first set rides on vertex end",
        last_cs(cmd) == r && r->oq_stamp[1].handle == 65);
      hk_sync_write_timestamp(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                              ts_query(3));
      set_event(cmd, &event, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
      struct hk_cs *ev = last_cs(cmd);
      T("set again after a fragment-end query writes from compute",
        r->timestamp.end.addr == ts_query(3).addr && ev != r &&
           ev->type == HK_CS_CDM && ev->sync.dep[HK_SYNC_VDM].cut == r);
   }
}

/* vkCmdSetEvent2 is ordered after its whole first scope, earlier submissions
 * included (Vulkan 1.4, 7.4.2). On an exact-prior queue that dependency on an
 * earlier submission's compute work reaches the kernel only as the carrier's
 * prior ordinal, and a primary-queue submission waits for the sibling kernel
 * queues only when some command names such a prior (hk_drm_sibling_prefix(),
 * sib_earlier). Submission A's long dispatch ran on a sibling; submission B
 * records an independent dispatch and sets the event after it. The event rides
 * on that dispatch's end stamp, which must keep naming A's dispatch: unlike a
 * timestamp, an event carrier keeps the same-CDM dependency. */
static void
test_event_carrier_keeps_prior_submission(void)
{
   const enum agx_chip chips[] = {AGX_CHIP_G17P, AGX_CHIP_G17G};
   const unsigned compute_k = util_logbase2(HK_SYNC_COMPUTE);
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned render = 0; render < 2; ++render) {
         test_chip = chips[c];
         struct hk_cmd_buffer *cmd = new_cmd();
         hk_cmd_buffer_device(cmd)->dev.params.features |=
            DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS |
            DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
         struct hk_cs *carrier = add(cmd, !render);
         if (!render) {
            carrier->stats.cmds = 1;
            cmd->current_cs.cs = carrier;
         }

         uint64_t status = 0;
         struct agx_va va = {.addr = 0xa0000};
         struct agx_bo bo = {.va = &va};
         struct hk_event event = test_event(&status, &bo, 71, 8);
         set_event(cmd, &event,
                   render ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
                          : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
         const bool stamped =
            nstreams(cmd) == 1 &&
            (render ? carrier->oq_stamp[1].handle == 71
                    : carrier->timestamp.end.handle == 71);
         T("event rides on the dispatch or render end stamp", stamped);

         /* Submission A: ten compute commands, its dispatch the last. */
         struct hk_queue q = {0};
         q.sync.exact_prior = true;
         q.sync.submitted[HK_SYNC_CDM] = 10;
         q.sync.last_absolute[compute_k][HK_SYNC_CDM] = 10;
         struct hk_cmd_buffer *cbs[] = {cmd};
         T("event carrier submission resolves",
           hk_sync_resolve_submit(&q, cbs, 1, false, false, 0, NULL));
         T(render ? "render event carrier names the earlier dispatch"
                  : "compute event carrier names the earlier dispatch",
           carrier->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE &&
              carrier->sync.prior[HK_SYNC_CDM] == 10);
      }
   }
}

int
main(void)
{
   test_context = ralloc_context(NULL);
   test_inline_stamps();
   test_inline_fallback(true);
   test_inline_fallback(false);
   test_inline_copy();
   test_inline_transition();
   test_inline_event(false);
   test_inline_event(true);
   test_inline_pending_timestamp();
   test_defer();
   test_stamped();
   test_keep();
   test_movable();
   test_order();
   test_order_forward();
   test_stamp_chips();
   test_join_moves_query_access_ranges();
   test_no_query_access_and_m5_reverse();
   test_oq_post_gfx_stamp_kind();
   test_oq_post_gfx_heap_promotion();
   test_event_retirement_stamping();
   test_event_stamp_not_copied();
   test_event_after_render_query();
   test_event_carrier_keeps_prior_submission();
   ralloc_free(test_context);
   printf("%u tests, %u failures\n", tests, failures);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

/* SPDX-License-Identifier: MIT
 * Calls real hk_sync and production split-rebase helper; no GPU or DRM calls.
 */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/ralloc.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_drm_prior_core.h"
#include "hk_queue.h"
_Static_assert(sizeof(struct drm_asahi_cmd_render) == 320, "legacy render");
_Static_assert(sizeof(struct drm_asahi_cmd_compute) == 128,
               "legacy Mesa compute");
_Static_assert(sizeof(struct drm_asahi_cmd_render_with_prior) == 336,
               "render wire");
_Static_assert(offsetof(struct drm_asahi_cmd_render_with_prior, prior_vdm) ==
                  320,
               "render prior vdm");
_Static_assert(offsetof(struct drm_asahi_cmd_render_with_prior, prior_cdm) ==
                  328,
               "render prior cdm");
_Static_assert(sizeof(struct drm_asahi_cmd_compute_with_prior) == 200,
               "compute wire");
_Static_assert(offsetof(struct drm_asahi_cmd_compute_with_prior, scs_layout) ==
                  128,
               "kernel scs_layout gap");
_Static_assert(offsetof(struct drm_asahi_cmd_compute_with_prior, prior_vdm) ==
                  136,
               "compute prior vdm");
_Static_assert(offsetof(struct drm_asahi_cmd_compute_with_prior, prior_cdm) ==
                  144,
               "compute prior cdm");
_Static_assert(offsetof(struct drm_asahi_cmd_compute_with_prior,
                        cdm_auxiliary_allocation) == 152,
               "compute auxiliary allocation");
_Static_assert(offsetof(struct drm_asahi_cmd_compute_with_prior,
                        intersection_pool) == 160,
               "compute intersection pool");
_Static_assert(offsetof(struct drm_asahi_cmd_compute_with_prior,
                        intersection_default_control) == 192,
               "compute intersection default control");
static unsigned tests, failures;
static void *test_context;

/* Fixtures need only the recorded synchronization fields, not a Vulkan device.
 * Keep all fixture allocations under one owner, including the overflow case. */
static void *
test_alloc(size_t size)
{
   void *ptr = rzalloc_size(test_context, size);
   if (!ptr) {
      fputs("Failed to allocate synchronization test fixture\n", stderr);
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
static struct hk_queue *
new_queue(bool exact)
{
   struct hk_queue *q = test_alloc(sizeof(*q));
   q->sync.exact_prior = exact;
   return q;
}
static struct hk_cmd_buffer *
new_cmd(void)
{
   struct hk_cmd_buffer *c = test_alloc(sizeof(*c));
   /* Stream creation reads device-wide options (the frame anatomy flag):
    * give each fixture a zeroed device. */
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
static bool
submit(struct hk_queue *q, struct hk_cmd_buffer *c)
{
   struct hk_cmd_buffer *cs[] = {c};
   return hk_sync_resolve_submit(q, cs, 1, false, false, 0);
}

static struct hk_cmd_buffer *
timestamp_cmd(enum agx_chip chip, bool exact)
{
   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_device *dev = test_alloc(sizeof(*dev));
   dev->dev.chip = chip;
   dev->dev.params.features = exact ? DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS : 0;
   cmd->vk.base.device = &dev->vk;
   return cmd;
}

static void
end_timestamp(struct hk_cmd_buffer *cmd, uint32_t query)
{
   const struct agx_timestamp_req req = {
      .handle = 1, .addr = 0x10000 + 8 * query, .offset_B = 8 * query};
   hk_sync_write_timestamp(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, req);
}

/* G17P: ResetQueryPool leaves a reusable compute stream. A TOP_OF_PIPE stamp
 * after it ends that stream's command: G17P start stamps latch before the
 * command's waits (hk_sync.c, start_stamps_ordered()), so the stamp must not
 * become the start of the dispatch that reuses the stream, and ending the
 * reset's command keeps it after the reset. The dispatch gets a command of
 * its own and the bottom stamp ends it.
 */
static void
test_timestamp_reused_compute(void)
{
   for (unsigned exact = 0; exact < 2; ++exact) {
      struct hk_cmd_buffer *cmd = timestamp_cmd(AGX_CHIP_G17P, exact);
      cmd->sync.kind_override = HK_SYNC_ALL;
      struct hk_cs *reset = add(cmd, true);
      cmd->sync.kind_override = 0;
      cmd->current_cs.cs = reset;
      reset->g17p_timestamp_query = true;
      reset->stats.cmds = 1;
      hk_sync_after_stream(cmd, reset, HK_SYNC_ALL);

      const struct agx_timestamp_req start = {
         .handle = 1, .addr = 0x10000, .offset_B = 0};
      hk_sync_write_timestamp(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, start);
      T("top timestamp is no pending start", !cmd->sync.pending_start_ts.handle);
      T("top timestamp ends the reset's command",
        reset->timestamp.end.handle == start.handle &&
        reset->timestamp.end.offset_B == start.offset_B &&
        !reset->timestamp.start.handle);
      T("top timestamp closes the reset stream", !cmd->current_cs.cs);

      /* The next dispatch opens a stream of its own (the CPU fixture cannot
       * allocate one through hk_cmd_buffer_get_cs()). */
      struct hk_cs *dispatch = add(cmd, true);
      cmd->current_cs.cs = dispatch;
      T("dispatch takes no start stamp", !dispatch->timestamp.start.handle);
      dispatch->stats.cmds++;
      end_timestamp(cmd, 1);
      T("bottom timestamp ends the dispatch's command",
        dispatch->timestamp.end.handle == 1 &&
        dispatch->timestamp.end.offset_B == 8);
      T("bottom timestamp closes the timed stream", !cmd->current_cs.cs);
      T("timed reset and dispatch resolve", submit(new_queue(exact), cmd));
   }
}

/* G17P: a TOP_OF_PIPE stamp inside a render pass instance ends the render
 * (fragment end), never its vertex start: G17P start stamps latch before the
 * render's waits, so a query reset before the pass could overwrite one. */
static void
test_timestamp_top_in_pass(void)
{
   struct hk_cmd_buffer *cmd = timestamp_cmd(AGX_CHIP_G17P, true);
   hk_sync_begin_rendering(cmd, false);
   struct hk_cs *render = test_alloc(sizeof(*render));
   render->type = HK_CS_VDM;
   list_addtail(&render->node, &cmd->control_streams);
   hk_sync_stream_created(cmd, render, &cmd->current_cs.gfx, false);
   cmd->current_cs.gfx = render;
   render->stats.cmds++;
   const struct agx_timestamp_req top = {.handle = 1, .addr = 0x10000, .offset_B = 0};
   hk_sync_write_timestamp(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, top);
   T("in-pass top stamp ends the render",
     render->timestamp.end.handle == 1 && render->timestamp.end.offset_B == 0);
   T("in-pass top stamp takes no vertex start", !render->timestamp.start.handle);
}

/* Exercise actual timestamp placement, barriers and the submission resolver.
 * Existing query-command scopes are not removed by the timestamp scope's
 * redundant same-engine execution edge. No command payload or GPU is needed.
 */
static void
test_timestamp_execution_scopes(void)
{
   for (unsigned exact = 0; exact < 2; ++exact) {
      struct hk_queue *queue = new_queue(exact);
      struct hk_cmd_buffer *first = timestamp_cmd(AGX_CHIP_G17P, exact);
      add(first, true);
      end_timestamp(first, 0);
      T("first timed compute resolves", submit(queue, first));
      struct hk_cmd_buffer *second = timestamp_cmd(AGX_CHIP_G17P, exact);
      struct hk_cs *timed = add(second, true);
      end_timestamp(second, 1);
      T("next timed compute resolves", submit(queue, second));
      T("timestamp execution scope follows the exact-prior capability",
        exact ? timed->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE &&
                   timed->sync.prior[HK_SYNC_CDM] == 0
              : timed->sync.barrier[HK_SYNC_CDM] == 0);

      /* Query copy/reset commands use an ALL earlier-work scope. Record
       * that genuine scope through the production barrier entry point before
       * the consumer, then add a timestamp to it and resolve both submissions.
       */
      struct hk_cmd_buffer *query = timestamp_cmd(AGX_CHIP_G17P, exact);
      hk_sync_barrier(query, HK_SYNC_ALL, HK_SYNC_ALL);
      struct hk_cs *copy = add(query, true);
      copy->g17p_timestamp_query = true;
      end_timestamp(query, 2);
      T("query consumer plus timestamp resolves", submit(queue, query));
      T("query memory consumer retains the completed producer prefix",
        exact ? copy->sync.prior[HK_SYNC_CDM] == 2
              : copy->sync.barrier[HK_SYNC_CDM] == 0);

      struct hk_cmd_buffer *barrier = timestamp_cmd(AGX_CHIP_G17P, exact);
      hk_sync_barrier(barrier, HK_SYNC_ALL, HK_SYNC_ALL);
      T("standalone query ordering resolves", submit(queue, barrier));
      struct hk_cmd_buffer *after = timestamp_cmd(AGX_CHIP_G17P, exact);
      struct hk_cs *inherited = add(after, true);
      end_timestamp(after, 3);
      T("later timestamp with carried requirements resolves", submit(queue, after));
      T("timestamp preserves a carried query-memory requirement",
        exact ? inherited->sync.prior[HK_SYNC_CDM] == 3
              : inherited->sync.barrier[HK_SYNC_CDM] == 0);

      queue = new_queue(exact);
      first = timestamp_cmd(AGX_CHIP_G17P, exact);
      add(first, false);
      T("earlier render resolves", submit(queue, first));
      second = timestamp_cmd(AGX_CHIP_G17P, exact);
      timed = add(second, true);
      end_timestamp(second, 4);
      T("timestamp after render resolves", submit(queue, second));
      T("timestamp cross-engine scope is preserved",
        exact ? timed->sync.prior[HK_SYNC_VDM] == 1
              : timed->sync.barrier[HK_SYNC_VDM] == 0);

      queue = new_queue(exact);
      first = timestamp_cmd(AGX_CHIP_G17P, exact);
      add(first, true);
      end_timestamp(first, 5);
      T("reset producer timestamp resolves", submit(queue, first));
      struct hk_cmd_buffer *reset_cmd = timestamp_cmd(AGX_CHIP_G17P, exact);
      hk_sync_barrier(reset_cmd, HK_SYNC_ALL, HK_SYNC_ALL);
      struct hk_cs *reset = add(reset_cmd, true);
      reset->g17p_timestamp_query = true;
      hk_sync_after_stream(reset_cmd, reset, HK_SYNC_ALL);
      struct hk_cs *later = add(reset_cmd, true);
      end_timestamp(reset_cmd, 6);
      T("reset then later stamped stream resolves", submit(queue, reset_cmd));
      T("reset keeps the earlier timestamp publication requirement",
        exact ? reset->sync.prior[HK_SYNC_CDM] == 1
              : reset->sync.barrier[HK_SYNC_CDM] == 0);
      T("later stamp keeps reset stream ordering",
        later->sync.barrier[HK_SYNC_CDM] == 1);

      queue = new_queue(exact);
      first = timestamp_cmd(AGX_CHIP_G17P, exact);
      add(first, true);
      T("render timestamp compute source resolves", submit(queue, first));
      second = timestamp_cmd(AGX_CHIP_G17P, exact);
      struct hk_cs *render = add(second, false);
      end_timestamp(second, 7);
      T("render end timestamp resolves", submit(queue, second));
      T("render timestamp keeps the original compute scope",
        exact ? render->sync.prior[HK_SYNC_CDM] == 1
              : render->sync.barrier[HK_SYNC_CDM] == 0);
   }

   for (unsigned chip = 0; chip < 2; ++chip) {
      const enum agx_chip target = chip ? AGX_CHIP_G16G : AGX_CHIP_G13G;
      struct hk_queue *queue = new_queue(true);
      struct hk_cmd_buffer *first = timestamp_cmd(target, true);
      add(first, true);
      T("other target source resolves", submit(queue, first));
      struct hk_cmd_buffer *second = timestamp_cmd(target, true);
      struct hk_cs *timed = add(second, true);
      end_timestamp(second, 0);
      T("other target timestamp resolves", submit(queue, second));
      T("unqualified target retains the original timestamp prefix",
        timed->sync.prior[HK_SYNC_CDM] == 1);
   }
}
/* Fixtures never queue immediate writes. Emitting one is a bug. */
void
hk_dispatch_imm_writes(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   (void)cs;
   abort();
}

bool
hk_defer_oq_available(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   (void)cs;
   abort();
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

/* Reset-stream fixtures model the ordering, without emitting reset payloads. */
void
hk_flush_query_resets(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   if (cs->query_reset_pending)
      abort();
}

/* Timestamp recording references payload/allocation entry points for other
 * branches. These tests attach end stamps to already-created closed streams;
 * reaching a payload emission branch would invalidate that fixture contract.
 */
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

/* Record a real full-frame pass with a sparse-mask preservation footprint.
 * Layer ranges describe BG/EOT's physical access, not just enabled views. */
static struct hk_cs *
render_pass(struct hk_cmd_buffer *cmd, const void *image, uint32_t layers,
            bool preserve_layers)
{
   hk_sync_begin_rendering(cmd, false);
   hk_sync_pass_footprint(cmd, image, 0, 0, layers, 32, 32,
                         (VkRect2D){{0, 0}, {32, 32}}, preserve_layers);
   hk_sync_pass_footprints_done(cmd);
   struct hk_cs *cs = test_alloc(sizeof(*cs));
   cs->type = HK_CS_VDM;
   hk_sync_stream_created(cmd, cs, &cmd->current_cs.gfx, false);
   list_addtail(&cs->node, &cmd->control_streams);
   hk_sync_end_rendering(cmd, false);
   return cs;
}


static void
test_precise_tile_evictions(void)
{
   const void *image = (void *)0x9876;
   struct hk_cmd_buffer *full = new_cmd(), *partial = new_cmd();
   struct hk_cs *write = render_pass(full, image, 1, false);
   struct hk_cs *preserve = render_pass(partial, image, 1, true);
   struct hk_cmd_buffer *one[] = {full};
   struct hk_queue *q = new_queue(true);
   q->sync.precise_tile_evictions = true;
   bool no_query = true, no_barrier = true, resolved = true;
   for (unsigned i = 1; i <= HK_SYNC_RECENT_FOOTPRINTS + 2; ++i) {
      q->drm.timeline_value = i;
      no_query &= !hk_sync_submit_needs_completed(q, one, 1);
      resolved &= submit(q, full);
      no_barrier &= write->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE;
   }
   T("complete writes skip retirement query through ring wrap", no_query);
   T("complete writes keep overlap through ring wrap", resolved && no_barrier);
   T("complete history retained for later partial pass",
     q->sync.recent_count == HK_SYNC_RECENT_FOOTPRINTS && q->sync.evicted > 0 &&
     q->sync.last_preserve == 0 && q->sync.evicted_preserve == 0);
   struct hk_cmd_buffer *p[] = {partial};
   T("later preserving pass queries complete history",
     hk_sync_submit_needs_completed(q, p, 1));
   const uint64_t point = ++q->drm.timeline_value;
   T("later preserving pass resolves", submit(q, partial));
   T("later preserving pass waits on evicted complete writes",
     preserve->sync.barrier[HK_SYNC_VDM] == 0 && q->sync.last_preserve == point);
   T("complete pass queries preserving history",
     hk_sync_submit_needs_completed(q, one, 1));
   ++q->drm.timeline_value;
   T("complete pass resolves against preserving history", submit(q, full));
   T("complete pass waits on live preserving pass",
     write->sync.barrier[HK_SYNC_VDM] == 0);

   /* Evict the preserving entry, then retire only through that entry. Later
    * complete writes still run. They must remain independent of a full write,
    * while a new partial write must wait for those same complete writes. */
   for (unsigned i = 0; i < HK_SYNC_RECENT_FOOTPRINTS; ++i) {
      ++q->drm.timeline_value;
      resolved &= submit(q, full);
   }
   T("preserving eviction tracked separately from complete writes",
     resolved && q->sync.evicted_preserve == point && q->sync.evicted > point);
   ++q->drm.timeline_value;
   T("retired preserving pass allows complete write",
     hk_sync_resolve_submit(q, one, 1, false, false, point) &&
     write->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);
   ++q->drm.timeline_value;
   T("partial pass still waits on live evicted complete writes",
     hk_sync_resolve_submit(q, p, 1, false, false, point) &&
     preserve->sync.barrier[HK_SYNC_VDM] == 0);

   q = new_queue(true);
   q->sync.precise_tile_evictions = true;
   q->drm.timeline_value = 1;
   struct hk_cmd_buffer *pair[] = {partial, full};
   T("partial then full in one submit retains dependency",
     hk_sync_resolve_submit(q, pair, 2, false, false, 0) &&
     write->sync.barrier[HK_SYNC_VDM] == preserve->sync.index);
   q = new_queue(true);
   q->sync.precise_tile_evictions = true;
   q->drm.timeline_value = 1;
   struct hk_cmd_buffer *reverse[] = {full, partial};
   T("full then partial in one submit retains dependency",
     hk_sync_resolve_submit(q, reverse, 2, false, false, 0) &&
     preserve->sync.barrier[HK_SYNC_VDM] == write->sync.index);

   q = new_queue(true);
   q->drm.timeline_value = 1;
   T("other-device complete history resolves", submit(q, full));
   T("other devices retain retirement query policy",
     hk_sync_submit_needs_completed(q, one, 1));
}


/* Transfer one range [lo, hi) into a new compute stream, as the G17P
 * transfer entry points and hk_cmd_buffer_get_cs_general() do. */
static struct hk_cs *
xfer_write(struct hk_cmd_buffer *cmd, uint64_t lo, uint64_t hi, bool bda)
{
   cmd->sync.xfer_depth = 1;
   cmd->sync.xfer_dst = (struct hk_sync_xfer_dst){
      .lo = lo, .hi = hi, .valid = lo < hi, .device_address = bda};
   struct hk_cs *cs = add(cmd, true);
   hk_sync_note_xfer_write(cmd, cs);
   cmd->sync.xfer_depth = 0;
   cmd->sync.xfer_dst = (struct hk_sync_xfer_dst){0};
   return cs;
}

/* G17P transfer writes: only work that may access the written range is
 * ordered after the transfer, everything else keeps its overlap. */
static void
test_xfer_write_order(void)
{
   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_cs *xfer = xfer_write(cmd, 0x10000, 0x20000, false);
   /* CTS null_as: the AS-build destination is never executed. */
   hk_sync_barrier(cmd, HK_SYNC_XFER, HK_SYNC_COMPUTE);
   T("non-graphics destination keeps the note", cmd->sync.xfer_write_count == 1);
   T("an access inside the written range matches",
     hk_sync_xfer_match(cmd, 0x18000, 0x18001, NULL) == xfer);
   T("an unrelated resource does not match",
     hk_sync_xfer_match(cmd, 0x20000, 0x30000, NULL) == NULL &&
        hk_sync_xfer_match(cmd, 0x0, 0x10000, NULL) == NULL);
   hk_sync_begin_rendering(cmd, false);
   struct hk_cs *gfx = add(cmd, false);
   T("a pass that does not access it keeps the overlap",
     gfx->sync.dep[HK_SYNC_CDM].cut == NULL &&
        gfx->sync.dep_ta[HK_SYNC_CDM].cut == NULL);
   cmd->current_cs.gfx = gfx;
   struct hk_cs *pre = test_alloc(sizeof(*pre));
   pre->type = HK_CS_CDM;
   list_addtail(&pre->node, &gfx->node);
   hk_sync_stream_created(cmd, pre, &cmd->current_cs.pre_gfx, true);
   cmd->current_cs.pre_gfx = pre;
   T("pre-graphics work starts unordered", pre->sync.dep[HK_SYNC_CDM].cut == NULL);
   /* A draw binds the cleared image. */
   hk_sync_order_xfer(cmd, xfer);
   /* The render already waits on its pre-graphics stream, a later prefix. */
   T("the open render waits in both halves",
     gfx->sync.dep[HK_SYNC_CDM].cut &&
        gfx->sync.dep[HK_SYNC_CDM].cut->sync.seq >= xfer->sync.seq &&
        gfx->sync.dep_ta[HK_SYNC_CDM].cut &&
        gfx->sync.dep_ta[HK_SYNC_CDM].cut->sync.seq >= xfer->sync.seq);
   T("its pre-graphics work waits", pre->sync.dep[HK_SYNC_CDM].cut == xfer);
   T("the ordered note is dropped", !hk_sync_xfer_pending(&cmd->sync));
   cmd->current_cs.pre_gfx = NULL;
   cmd->current_cs.gfx = NULL;
   hk_sync_end_rendering(cmd, false);
   hk_sync_begin_rendering(cmd, false);
   struct hk_cs *gfx2 = add(cmd, false);
   T("later graphics stays ordered", gfx2->sync.dep[HK_SYNC_CDM].cut == xfer);
   hk_sync_end_rendering(cmd, false);
   struct hk_queue *queue = new_queue(true);
   T("ordered transfer resolves", submit(queue, cmd));
   T("pre-graphics waits for the exact transfer ordinal",
     pre->sync.barrier[HK_SYNC_CDM] == xfer->sync.index);

   /* Ordered at the next pass begin (an attachment). */
   cmd = new_cmd();
   xfer = xfer_write(cmd, 0x10000, 0x20000, false);
   hk_sync_begin_rendering(cmd, false);
   T("attachment match", hk_sync_xfer_match(cmd, 0x10000, 0x10001, NULL) == xfer);
   hk_sync_order_xfer(cmd, xfer);
   gfx = add(cmd, false);
   T("the render created after waits in both halves",
     gfx->sync.dep[HK_SYNC_CDM].cut == xfer &&
        gfx->sync.dep_ta[HK_SYNC_CDM].cut == xfer);
   T("robustness does not import earlier command buffers",
     cmd->sync.req[0][HK_SYNC_CDM].prior == 0);
   hk_sync_end_rendering(cmd, false);

   /* Dependencies the application records into graphics stages. */
   const uint8_t gfx_dsts[] = {HK_SYNC_GFX, HK_SYNC_GFX_FRAG,
                               HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG, HK_SYNC_ALL};
   for (unsigned i = 0; i < ARRAY_SIZE(gfx_dsts); ++i) {
      cmd = new_cmd();
      xfer_write(cmd, 0x10000, 0x20000, false);
      hk_sync_barrier(cmd, HK_SYNC_XFER, gfx_dsts[i]);
      T("a transfer-to-graphics dependency drops the note",
        !hk_sync_xfer_pending(&cmd->sync));
   }
   cmd = new_cmd();
   xfer_write(cmd, 0x10000, 0x20000, false);
   hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   T("a compute-to-graphics dependency does not cover a transfer",
     hk_sync_xfer_pending(&cmd->sync));
   hk_sync_barrier(cmd, HK_SYNC_ALL, HK_SYNC_ALL);
   T("ALL_COMMANDS covers it", !hk_sync_xfer_pending(&cmd->sync));
   xfer = xfer_write(cmd, 0x10000, 0x20000, false);
   T("a write after the dependency is noted again",
     hk_sync_xfer_match(cmd, 0x10000, 0x20000, NULL) == xfer);

   /* The same stream launching again for one transfer is one note. */
   cmd = new_cmd();
   xfer = xfer_write(cmd, 0x10000, 0x20000, false);
   cmd->sync.xfer_depth = 1;
   cmd->sync.xfer_dst = (struct hk_sync_xfer_dst){
      .lo = 0x20000, .hi = 0x28000, .valid = true};
   hk_sync_note_xfer_write(cmd, xfer);
   cmd->sync.xfer_depth = 0;
   T("an adjacent launch widens the note",
     cmd->sync.xfer_write_count == 1 && cmd->sync.xfer_writes[0].hi == 0x28000);

   /* What a draw cannot see matches every access. */
   cmd = new_cmd();
   xfer = xfer_write(cmd, 0x10000, 0x20000, true);
   T("a device-address buffer matches any access",
     hk_sync_xfer_match(cmd, 0x90000, 0x90001, NULL) == xfer);
   cmd = new_cmd();
   xfer = xfer_write(cmd, 0, 0, false);
   T("an unknown destination matches any access",
     hk_sync_xfer_match(cmd, 0x90000, 0x90001, NULL) == xfer);
   cmd = new_cmd();
   struct hk_cs *last = NULL;
   for (unsigned i = 0; i <= HK_SYNC_XFER_WRITES; ++i)
      last = xfer_write(cmd, 0x10000 * (i + 1), 0x10000 * (i + 1) + 0x100, false);
   T("a write past the bound matches any access",
     cmd->sync.xfer_write_count == HK_SYNC_XFER_WRITES &&
        hk_sync_xfer_match(cmd, 0x900000, 0x900001, NULL) == last);
   T("and every noted stream is matched by match_all",
     hk_sync_xfer_match_all(cmd, NULL) == last);
   /* Ordering a middle stream drops only the notes it covers. */
   struct hk_cs *mid = cmd->sync.xfer_writes[3].cs;
   hk_sync_order_xfer(cmd, mid);
   T("a prefix covers the notes at or before it",
     cmd->sync.xfer_write_count == HK_SYNC_XFER_WRITES - 4 &&
        cmd->sync.xfer_any == last);

   /* Application compute is never noted. */
   cmd = new_cmd();
   add(cmd, true);
   T("application compute is not a transfer write",
     !hk_sync_xfer_pending(&cmd->sync));

   /* Stream bookkeeping. */
   cmd = new_cmd();
   xfer = xfer_write(cmd, 0x10000, 0x20000, false);
   struct hk_cs *later = xfer_write(cmd, 0x30000, 0x40000, false);
   hk_sync_merge(cmd, xfer, later);
   T("a merged stream's note follows it",
     hk_sync_xfer_match(cmd, 0x30000, 0x30001, NULL) == xfer);
   hk_sync_forget_stream(cmd, xfer);
   T("a removed first stream's notes are dropped",
     !hk_sync_xfer_pending(&cmd->sync));
   xfer_write(cmd, 0x10000, 0x20000, true);
   hk_sync_reset(cmd);
   T("command buffer reset drops the notes", !hk_sync_xfer_pending(&cmd->sync));
}

static void
test_multiview_layer_preservation(void)
{
   const void *image = (void *)0x1234;
   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_cs *first = render_pass(cmd, image, 1, false);
   struct hk_cs *sparse = render_pass(cmd, image, 2, true);
   T("sparse layer store waits on full XY layer0 clear",
     sparse->sync.dep[HK_SYNC_VDM].cut == first);
   struct hk_queue *queue = new_queue(true);
   T("multiview same command buffer submit", submit(queue, cmd));
   T("layer preservation resolves to exact earlier render",
     sparse->sync.barrier[HK_SYNC_VDM] == first->sync.index);

   cmd = new_cmd();
   first = render_pass(cmd, image, 2, false);
   struct hk_cs *whole = render_pass(cmd, image, 2, false);
   T("complete layer writes do not add a tile dependency",
     !whole->sync.dep[HK_SYNC_VDM].cut);
   struct hk_cs *different = render_pass(cmd, (void *)0x5678, 2, true);
   T("different attachment does not add a layer dependency",
     !different->sync.dep[HK_SYNC_VDM].cut);

   queue = new_queue(true);
   struct hk_cmd_buffer *a = new_cmd(), *b = new_cmd();
   first = render_pass(a, image, 1, false);
   sparse = render_pass(b, image, 2, true);
   struct hk_cmd_buffer *pair[] = {a, b};
   T("multiview across command buffers submit",
     hk_sync_resolve_submit(queue, pair, 2, false, false, 0));
   T("across command buffers keeps layer store ordered",
     sparse->sync.barrier[HK_SYNC_VDM] == first->sync.index);

   queue = new_queue(true);
   a = new_cmd(); b = new_cmd();
   render_pass(a, image, 1, false);
   sparse = render_pass(b, image, 2, true);
   queue->drm.timeline_value = 1;
   T("multiview first queue submission", submit(queue, a));
   queue->drm.timeline_value = 2;
   struct hk_cmd_buffer *single[] = {b};
   T("multiview later queue submission",
     hk_sync_resolve_submit(queue, single, 1, false, false, 0));
   T("across queue submissions waits on prior layer store",
     sparse->sync.barrier[HK_SYNC_VDM] == 0);
}

/* The empty-render optimizer executes the API clear on the compute engine. */
static struct hk_cs *
compute_clear_pass(struct hk_cmd_buffer *cmd, const void *image,
                   uint32_t base_layer, bool end)
{
   hk_sync_begin_rendering(cmd, false);
   hk_sync_pass_footprint(cmd, image, 0, base_layer, base_layer + 1, 32, 32,
                         (VkRect2D){{0, 0}, {32, 32}}, false);
   hk_sync_pass_footprints_done(cmd);
   struct hk_cs *render = test_alloc(sizeof(*render));
   render->type = HK_CS_VDM;
   hk_sync_stream_created(cmd, render, &cmd->current_cs.gfx, false);
   list_addtail(&render->node, &cmd->control_streams);
   struct hk_cs *clear = test_alloc(sizeof(*clear));
   clear->type = HK_CS_CDM;
   clear->stats.cmds = 1;
   cmd->sync.kind_override = HK_SYNC_GFX;
   hk_sync_stream_created(cmd, clear, &cmd->current_cs.cs, true);
   list_addtail(&clear->node, &cmd->control_streams);
   cmd->sync.kind_override = 0;
   hk_sync_replace_render(cmd, render, clear);
   hk_sync_forget_stream(cmd, render);
   list_del(&render->node);
   if (end)
      hk_sync_end_rendering(cmd, false);
   return clear;
}

/* An empty suspended instance whose render was replaced by a compute clear:
 * the resuming instance continues the same render pass and loads what the
 * clear wrote, whatever its footprint. */
static void
test_resume_after_compute_clear(void)
{
   const void *image = (void *)0x4321;
   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_cs *clear = compute_clear_pass(cmd, image, 0, false);
   hk_sync_end_rendering(cmd, true);
   hk_sync_begin_rendering(cmd, true);
   hk_sync_pass_footprint(cmd, image, 0, 0, 1, 32, 32,
                          (VkRect2D){{0, 0}, {32, 32}}, false);
   hk_sync_pass_footprints_done(cmd);
   struct hk_cs *resumed = add(cmd, false);
   T("resumed instance waits on the suspended instance's compute clear",
     resumed->sync.dep[HK_SYNC_CDM].cut == clear);
   struct hk_cs *next = add(cmd, false);
   T("later streams of the resumed instance chain on it",
     next->sync.dep[HK_SYNC_VDM].cut == resumed);
   hk_sync_end_rendering(cmd, false);
   T("resume after compute clear submit", submit(new_queue(true), cmd));
   T("resumed instance resolves to the clear's compute prefix",
     resumed->sync.barrier[HK_SYNC_CDM] == clear->sync.index);

   /* A new instance is not a continuation: a whole-tile pass keeps
    * overlapping the earlier one. */
   cmd = new_cmd();
   compute_clear_pass(cmd, image, 0, true);
   hk_sync_begin_rendering(cmd, false);
   struct hk_cs *fresh = add(cmd, false);
   T("a new instance does not wait as a continuation",
     fresh->sync.dep[HK_SYNC_CDM].cut == NULL);
   hk_sync_end_rendering(cmd, false);

   /* Resumed in a later command buffer: earlier graphics work on both
    * subqueues may have produced the attachments. */
   cmd = new_cmd();
   hk_sync_begin_rendering(cmd, true);
   resumed = add(cmd, false);
   T("cross-buffer resume waits on earlier graphics on both subqueues",
     (resumed->sync.dep[HK_SYNC_VDM].prior & HK_SYNC_GFX) &&
        (resumed->sync.dep[HK_SYNC_CDM].prior & HK_SYNC_GFX));
   hk_sync_end_rendering(cmd, false);
}

static void
test_loaded_color_preservation(void)
{
   const void *image = (void *)0x1234;
   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_cs *clear = compute_clear_pass(cmd, image, 0, true);
   struct hk_cs *load = render_pass(cmd, image, 1, true);
   T("full-area color load waits on its compute clear",
     load->sync.dep[HK_SYNC_CDM].cut == clear &&
     !load->sync.dep[HK_SYNC_VDM].cut);
   struct hk_queue *queue = new_queue(true);
   T("loaded color submit", submit(queue, cmd));
   T("loaded color names the exact compute producer",
     load->sync.barrier[HK_SYNC_CDM] == clear->sync.index &&
     load->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);

   cmd = new_cmd();
   clear = compute_clear_pass(cmd, image, 0, true);
   struct hk_cs *other = render_pass(cmd, (void *)0x5678, 1, true);
   T("unrelated loaded image remains independent",
     !other->sync.dep[HK_SYNC_CDM].cut &&
     !other->sync.dep[HK_SYNC_VDM].cut);
   struct hk_cs *overwrite = render_pass(cmd, image, 1, false);
   T("complete clear or discard does not preserve prior color",
     !overwrite->sync.dep[HK_SYNC_CDM].cut &&
     !overwrite->sync.dep[HK_SYNC_VDM].cut);

   struct hk_cmd_buffer *a = new_cmd(), *b = new_cmd();
   clear = compute_clear_pass(a, image, 0, true);
   load = render_pass(b, image, 1, true);
   queue = new_queue(true);
   struct hk_cmd_buffer *pair[] = {a, b};
   T("loaded color across command buffers submit",
     hk_sync_resolve_submit(queue, pair, 2, false, false, 0));
   T("loaded color across command buffers retains compute clear",
     load->sync.barrier[HK_SYNC_CDM] == clear->sync.index &&
     load->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);

   a = new_cmd(); b = new_cmd();
   compute_clear_pass(a, image, 0, true);
   load = render_pass(b, (void *)0x5678, 1, true);
   pair[0] = a; pair[1] = b;
   queue = new_queue(true);
   T("unrelated color across command buffers submit",
     hk_sync_resolve_submit(queue, pair, 2, false, false, 0));
   T("unrelated color across command buffers stays independent",
     load->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE &&
     load->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);

   cmd = new_cmd();
   clear = compute_clear_pass(cmd, image, 1, true);
   load = render_pass(cmd, image, 1, true);
   T("loaded layer zero does not wait on disjoint layer one",
     !load->sync.dep[HK_SYNC_CDM].cut);
   load = render_pass(cmd, image, 2, true);
   T("loaded layer range waits on overlapping compute clear",
     load->sync.dep[HK_SYNC_CDM].cut == clear);

   a = new_cmd(); b = new_cmd();
   clear = compute_clear_pass(a, image, 0, true);
   load = render_pass(b, image, 1, true);
   queue = new_queue(true);
   queue->drm.timeline_value = 1;
   T("loaded color producer submission", submit(queue, a));
   queue->drm.timeline_value = 2;
   struct hk_cmd_buffer *single[] = {b};
   T("loaded color consumer submission",
     hk_sync_resolve_submit(queue, single, 1, false, false, 0));
   T("loaded color waits on prior compute history",
     load->sync.barrier[HK_SYNC_CDM] == 0 &&
     load->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);
}

static void
test_compute_clear_layer_preservation(void)
{
   const void *image = (void *)0x1234;
   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_cs *clear = compute_clear_pass(cmd, image, 0, true);
   struct hk_cs *sparse = render_pass(cmd, image, 2, true);
   T("sparse layer store waits on compute replacement clear",
     sparse->sync.dep[HK_SYNC_CDM].cut == clear);
   T("compute replacement does not fabricate render dependency",
     sparse->sync.dep[HK_SYNC_VDM].cut == NULL);
   struct hk_queue *queue = new_queue(true);
   T("compute clear same command buffer submit", submit(queue, cmd));
   T("compute clear resolves to correct engine prefix",
     sparse->sync.barrier[HK_SYNC_CDM] == clear->sync.index &&
     sparse->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);

   queue = new_queue(true);
   struct hk_cmd_buffer *a = new_cmd(), *b = new_cmd();
   clear = compute_clear_pass(a, image, 0, true);
   sparse = render_pass(b, image, 2, true);
   struct hk_cmd_buffer *pair[] = {a, b};
   T("compute clear across command buffers submit",
     hk_sync_resolve_submit(queue, pair, 2, false, false, 0));
   T("across command buffers names compute not render engine",
     sparse->sync.barrier[HK_SYNC_CDM] == clear->sync.index &&
     sparse->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);

   queue = new_queue(true);
   a = new_cmd(); b = new_cmd();
   compute_clear_pass(a, image, 0, true);
   sparse = render_pass(b, image, 2, true);
   queue->drm.timeline_value = 1;
   T("compute clear first queue submission", submit(queue, a));
   queue->drm.timeline_value = 2;
   struct hk_cmd_buffer *single[] = {b};
   T("compute clear later queue submission",
     hk_sync_resolve_submit(queue, single, 1, false, false, 0));
   T("prior compute clear hazard names only compute history",
     sparse->sync.barrier[HK_SYNC_CDM] == 0 &&
     sparse->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);
   T("completed prior clear submission", hk_sync_resolve_submit(
      queue, single, 1, false, false, 1));
   T("completed prior clear has no remaining hazard",
     sparse->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE);

   /* Separate API layers are written on separate engines; preserving both
    * needs both producers, not just whichever has the larger sequence. */
   cmd = new_cmd();
   struct hk_cs *graphics = render_pass(cmd, image, 1, false);
   clear = compute_clear_pass(cmd, image, 1, true);
   sparse = render_pass(cmd, image, 3, true);
   T("sparse third layer waits on both physical producers",
     sparse->sync.dep[HK_SYNC_VDM].cut == graphics &&
     sparse->sync.dep[HK_SYNC_CDM].cut == clear);

   cmd = new_cmd();
   clear = compute_clear_pass(cmd, image, 0, false);
   struct hk_cs *continued = test_alloc(sizeof(*continued));
   continued->type = HK_CS_VDM;
   hk_sync_stream_created(cmd, continued, &cmd->current_cs.gfx, false);
   list_addtail(&continued->node, &cmd->control_streams);
   T("render continuation waits on replaced clear",
     continued->sync.dep[HK_SYNC_CDM].cut == clear);
   hk_sync_end_rendering(cmd, false);
}

/* XFB counter copies can precede creation of their consuming render stream.
 * A self-dependency between captures must order both physical engines, and
 * must end with the pass rather than turn into a queue-wide requirement. */
static struct hk_cs *
add_slot(struct hk_cmd_buffer *cmd, struct hk_cs **slot, bool compute)
{
   struct hk_cs *cs = test_alloc(sizeof(*cs));
   cs->type = compute ? HK_CS_CDM : HK_CS_VDM;
   /* Match the real allocator when software work is introduced after the
    * consuming render has already been opened. */
   if (slot == &cmd->current_cs.pre_gfx && cmd->current_cs.gfx)
      list_addtail(&cs->node, &cmd->current_cs.gfx->node);
   else
      list_addtail(&cs->node, &cmd->control_streams);
   *slot = cs;
   hk_sync_stream_created(cmd, cs, slot, compute);
   return cs;
}

static void
test_pass_software_graphics_dependencies(void)
{
   struct hk_cmd_buffer *local = new_cmd();
   hk_sync_begin_rendering(local, false);
   struct hk_cs *render = add_slot(local, &local->current_cs.gfx, false);
   hk_sync_pass_barrier(local, HK_SYNC_GFX, HK_SYNC_GFX_FRAG);
   struct hk_cs *pre = add_slot(local, &local->current_cs.pre_gfx, true);
   T("software vertex work does not wait on fragment-only destination",
     !pre->sync.dep[HK_SYNC_VDM].cut);
   T("render still waits on its new pre-graphics producer",
     render->sync.dep[HK_SYNC_CDM].cut == pre);
   hk_sync_end_rendering(local, false);
   struct hk_queue *local_queue = new_queue(true);
   T("local barrier and later software draw resolve without cycle",
     submit(local_queue, local));

   local = new_cmd();
   struct hk_cs *external = add(local, false);
   hk_sync_barrier(local, HK_SYNC_GFX, HK_SYNC_GFX_FRAG);
   hk_sync_begin_rendering(local, false);
   render = add_slot(local, &local->current_cs.gfx, false);
   hk_sync_pass_barrier(local, HK_SYNC_GFX, HK_SYNC_GFX_FRAG);
   pre = add_slot(local, &local->current_cs.pre_gfx, true);
   T("pre-graphics producer retains external fragment requirements",
     pre->sync.dep[HK_SYNC_VDM].cut == external);
   T("local requirement does not replace producer's external source",
     pre->sync.dep[HK_SYNC_VDM].cut != render);
   hk_sync_end_rendering(local, false);
   local_queue = new_queue(true);
   T("external and local requirements resolve without cycle",
     submit(local_queue, local));

   struct hk_cmd_buffer *cmd = new_cmd();
   hk_sync_begin_rendering(cmd, false);
   struct hk_cs *first_pre =
      add_slot(cmd, &cmd->current_cs.pre_gfx, true);
   struct hk_cs *first = add_slot(cmd, &cmd->current_cs.gfx, false);
   T("render consumes pre-graphics stream opened by counter copy",
     first->sync.dep[HK_SYNC_CDM].cut == first_pre &&
     first->sync.dep_ta[HK_SYNC_CDM].cut == first_pre);

   hk_sync_pass_barrier(cmd, HK_SYNC_GFX, HK_SYNC_GFX);
   /* The API entry point ends both open streams at a memory dependency. */
   cmd->current_cs.gfx = cmd->current_cs.pre_gfx = NULL;
   struct hk_cs *resume = add_slot(cmd, &cmd->current_cs.pre_gfx, true);
   T("resumed software graphics waits for earlier capture on CDM",
     resume->sync.dep[HK_SYNC_CDM].cut == first_pre);
   T("resumed software graphics waits for earlier render on VDM",
     resume->sync.dep[HK_SYNC_VDM].cut == first);
   struct hk_cs *second = add_slot(cmd, &cmd->current_cs.gfx, false);
   T("second render consumes already-open resumed capture",
     second->sync.dep[HK_SYNC_CDM].cut == resume &&
     second->sync.dep_ta[HK_SYNC_CDM].cut == resume);
   hk_sync_end_rendering(cmd, false);
   cmd->current_cs.gfx = cmd->current_cs.pre_gfx = NULL;
   struct hk_cs *outside = add(cmd, false);
   T("self-dependency has no scope after the pass",
     !outside->sync.dep[HK_SYNC_CDM].cut &&
     !outside->sync.dep[HK_SYNC_VDM].cut &&
     !outside->sync.dep[HK_SYNC_CDM].prior &&
     !outside->sync.dep[HK_SYNC_VDM].prior);

   struct hk_queue *queue = new_queue(true);
   T("submit software graphics self-dependency", submit(queue, cmd));
   T("resumed capture resolves exact earlier CDM prefix",
     resume->sync.barrier[HK_SYNC_CDM] == first_pre->sync.index);
   T("render resolves exact resumed CDM prefix",
     second->sync.barrier[HK_SYNC_CDM] == resume->sync.index);

   cmd = new_cmd();
   add(cmd, true);
   add(cmd, false);
   hk_sync_begin_rendering(cmd, false);
   hk_sync_pass_barrier(cmd, HK_SYNC_ALL, HK_SYNC_GFX);
   struct hk_cs *unrelated = add(cmd, false);
   T("self-dependency has no source before the pass",
     !unrelated->sync.dep[HK_SYNC_CDM].cut &&
     !unrelated->sync.dep[HK_SYNC_VDM].cut &&
     !unrelated->sync.dep[HK_SYNC_CDM].prior &&
     !unrelated->sync.dep[HK_SYNC_VDM].prior);
   hk_sync_end_rendering(cmd, false);

   cmd = new_cmd();
   struct hk_cs *clear = compute_clear_pass(cmd, (void *)0x1234, 0, false);
   hk_sync_pass_barrier(cmd, HK_SYNC_GFX, HK_SYNC_GFX);
   struct hk_cs *after_clear = add(cmd, false);
   T("self-dependency follows an empty render replaced by compute clear",
     after_clear->sync.dep[HK_SYNC_CDM].cut == clear &&
     !after_clear->sync.dep[HK_SYNC_VDM].cut);
   hk_sync_end_rendering(cmd, false);
}

/* Kernels without per-command input syncs attach the ioctl's waits to the
 * first command only, so every later command is chained onto it. The first
 * command belongs to the first command buffer that emits a stream: chaining
 * onto a subqueue with no earlier command gives the kernel a barrier it
 * rejects with EINVAL. */
/* G17P, like M5: a dependency from graphics to every stage (Kyty records
 * ALL_COMMANDS -> ALL_COMMANDS for guest flushes) orders the next render's
 * vertex half only after renders that wrote memory other than their
 * attachments; its fragment half waits on everything. Other chips keep the
 * whole dependency for both halves. */
static void
test_g17p_vertex_after_memory_writers(void)
{
   for (unsigned writes = 0; writes < 2; ++writes) {
      struct hk_cmd_buffer *cmd = timestamp_cmd(AGX_CHIP_G17P, true);
      struct hk_cs *writer = add(cmd, false);
      hk_sync_note_memory_write(cmd, writer);
      hk_sync_barrier(cmd, HK_SYNC_ALL, HK_SYNC_ALL);
      struct hk_cs *plain = add(cmd, false);
      if (writes)
         hk_sync_note_memory_write(cmd, plain);
      hk_sync_barrier(cmd, HK_SYNC_ALL, HK_SYNC_ALL);
      struct hk_cs *next = add(cmd, false);
      T("G17P fragment half waits on the previous render",
        next->sync.dep[HK_SYNC_VDM].cut == plain);
      T("G17P vertex half waits on the latest memory writer",
        next->sync.dep_ta[HK_SYNC_VDM].cut == (writes ? plain : writer));
      struct hk_queue *queue = new_queue(true);
      T("G17P memory-writer submit", submit(queue, cmd));
      T("G17P resolved vertex half",
        next->sync.barrier[HK_SYNC_VDM] == 2 &&
        next->sync.barrier_ta[HK_SYNC_VDM] == (writes ? 2 : 1) &&
        next->sync.ta_split == !writes);

      /* The requirement carried into a later submission (the second
       * barrier's) narrows the same way. */
      struct hk_cmd_buffer *later = timestamp_cmd(AGX_CHIP_G17P, true);
      struct hk_cs *carried = add(later, false);
      T("G17P carried submit", submit(queue, later));
      T("G17P carried vertex half",
        carried->sync.prior[HK_SYNC_VDM] == 2 &&
        carried->sync.prior_ta[HK_SYNC_VDM] == (writes ? 2 : 1));
   }

   /* A render whose attachments can alias other memory writes memory with
    * its loads and stores alone: a draw-less clear or resolve. */
   struct hk_cmd_buffer *clear = timestamp_cmd(AGX_CHIP_G17P, true);
   clear->state.gfx.render.m5_attachments_reach_memory = true;
   struct hk_cs *cleared = add(clear, false);
   clear->state.gfx.render.m5_attachments_reach_memory = false;
   hk_sync_barrier(clear, HK_SYNC_ALL, HK_SYNC_ALL);
   struct hk_cs *after_clear = add(clear, false);
   T("G17P aliasing attachments without a draw count as a memory writer",
     (cleared->sync.kind & HK_SYNC_GFX_MEM) &&
     after_clear->sync.dep_ta[HK_SYNC_VDM].cut == cleared);

   struct hk_cmd_buffer *cmd = timestamp_cmd(AGX_CHIP_G15S, true);
   struct hk_cs *plain = add(cmd, false);
   hk_sync_barrier(cmd, HK_SYNC_ALL, HK_SYNC_ALL);
   struct hk_cs *next = add(cmd, false);
   T("other chips: vertex half keeps the whole dependency",
     next->sync.dep_ta[HK_SYNC_VDM].cut == plain);
}

/* G17P image barriers order a render's vertex half only once it draws with a
 * pre-rasterization shader that reads images (HK_SYNC_GFX_IMG). */
static void
test_image_barrier_vertex_class(void)
{
   const VkImageMemoryBarrier2 image = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
   };
   const VkMemoryBarrier2 memory = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
   };
   const VkDependencyInfo image_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &image,
   };
   const VkDependencyInfo memory_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &memory,
   };
   uint8_t src, dst;
   hk_sync_dependency_kinds(&image_dep, true, &src, &dst);
   T("image barrier names the fragment and image classes",
     src == HK_SYNC_GFX && dst == (HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG));
   hk_sync_dependency_kinds(&image_dep, false, &src, &dst);
   T("without the image class an image barrier names graphics",
     dst == HK_SYNC_GFX);
   hk_sync_dependency_kinds(&memory_dep, true, &src, &dst);
   T("memory barrier still names graphics", dst == HK_SYNC_GFX);

   struct hk_cmd_buffer *cmd = new_cmd();
   struct hk_cs *producer = add(cmd, false);
   hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG);
   struct hk_cs *plain = add(cmd, false);
   T("fragment half waits on the image producer",
     plain->sync.dep[HK_SYNC_VDM].cut == producer);
   T("vertex half without image reads does not",
     !plain->sync.dep_ta[HK_SYNC_VDM].cut);
   struct hk_cs *reader = add(cmd, false);
   hk_sync_vertex_images(cmd, reader);
   T("image-reading vertex half waits on the image producer",
     reader->sync.dep_ta[HK_SYNC_VDM].cut == producer);
   struct hk_queue *queue = new_queue(true);
   T("image class submit", submit(queue, cmd));
   T("plain render splits its vertex half off",
     plain->sync.barrier_ta[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
     plain->sync.prior_ta[HK_SYNC_VDM] == 0 &&
     (plain->sync.barrier[HK_SYNC_VDM] != DRM_ASAHI_BARRIER_NONE ||
      plain->sync.prior[HK_SYNC_VDM] != 0));
   T("image-reading render keeps its vertex half ordered",
     reader->sync.barrier_ta[HK_SYNC_VDM] ==
        reader->sync.barrier[HK_SYNC_VDM] &&
     reader->sync.prior_ta[HK_SYNC_VDM] == reader->sync.prior[HK_SYNC_VDM]);

   /* Carried into a later command buffer of the same submission. */
   struct hk_cmd_buffer *a = new_cmd(), *b = new_cmd();
   producer = add(a, false);
   hk_sync_barrier(a, HK_SYNC_GFX, HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG);
   plain = add(b, false);
   reader = add(b, false);
   hk_sync_vertex_images(b, reader);
   queue = new_queue(true);
   struct hk_cmd_buffer *pair[] = {a, b};
   T("carried image class submit",
     hk_sync_resolve_submit(queue, pair, 2, false, false, 0));
   T("carried image class leaves a plain vertex half unordered",
     plain->sync.barrier_ta[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
     plain->sync.prior_ta[HK_SYNC_VDM] == 0);
   T("carried image class orders an image-reading vertex half",
     reader->sync.barrier_ta[HK_SYNC_VDM] != DRM_ASAHI_BARRIER_NONE ||
     reader->sync.prior_ta[HK_SYNC_VDM] != 0);
}

static void
test_chain_first_skips_empty_buffers(void)
{
   for (unsigned exact = 0; exact < 2; ++exact) {
      struct hk_queue *q = new_queue(exact);
      struct hk_cmd_buffer *empty = new_cmd();
      struct hk_cmd_buffer *a = new_cmd();
      struct hk_cmd_buffer *b = new_cmd();
      struct hk_cs *compute = add(a, true);
      struct hk_cs *render = add(b, false);
      struct hk_cmd_buffer *ecr[] = {empty, a, b};
      T("chain [empty, compute, render]",
        hk_sync_resolve_submit(q, ecr, 3, false, true, 0));
      T("chain first command has no barrier",
        compute->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
           compute->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE);
      T("chain no barrier on VDM before any render",
        render->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);
      T("chain render follows the first compute",
        render->sync.barrier[HK_SYNC_CDM] == 1);
      T("chain vertex half matches",
        render->sync.barrier_ta[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
           render->sync.barrier_ta[HK_SYNC_CDM] == 1);

      q = new_queue(exact);
      empty = new_cmd();
      a = new_cmd();
      b = new_cmd();
      render = add(a, false);
      compute = add(b, true);
      struct hk_cmd_buffer *erc[] = {empty, a, b};
      T("chain [empty, render, compute]",
        hk_sync_resolve_submit(q, erc, 3, false, true, 0));
      T("chain no barrier on CDM before any compute",
        compute->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE &&
           compute->sync.barrier[HK_SYNC_VDM] == 1 &&
           render->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
           render->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE);

      /* Unchanged when the first command buffer has work. */
      q = new_queue(exact);
      a = new_cmd();
      b = new_cmd();
      compute = add(a, true);
      render = add(b, false);
      struct hk_cmd_buffer *cr[] = {a, b};
      T("chain [compute, render]",
        hk_sync_resolve_submit(q, cr, 2, false, true, 0));
      T("chain [compute, render] render follows compute",
        compute->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE &&
           render->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
           render->sync.barrier[HK_SYNC_CDM] == 1);

      /* No command at all: nothing to chain. */
      q = new_queue(exact);
      struct hk_cmd_buffer *ee[] = {new_cmd(), new_cmd()};
      T("chain without commands",
        hk_sync_resolve_submit(q, ee, 2, false, true, 0) &&
           q->sync.submitted[HK_SYNC_VDM] == 0 &&
           q->sync.submitted[HK_SYNC_CDM] == 0);
   }
}

/* Native ray tracing sends the extended compute command, which only a
 * kernel advertising both the exact-prior layout and ray query state parses.
 * Anything less keeps ray tracing off. */
static void
test_rt_kernel_gate(void)
{
   struct agx_device *dev = test_alloc(sizeof(*dev));
   dev->chip = AGX_CHIP_G17P;
   dev->params.features = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS |
                          DRM_ASAHI_FEATURE_RENDER_RAY_QUERY;
   T("rt on with exact prior and ray query", hk_g17p_rt_enabled(dev));
   dev->params.features = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS;
   T("rt off without ray query", !hk_g17p_rt_enabled(dev));
   dev->params.features = DRM_ASAHI_FEATURE_RENDER_RAY_QUERY;
   T("rt off without exact prior", !hk_g17p_rt_enabled(dev));
   dev->params.features = 0;
   T("rt off on a kernel without either", !hk_g17p_rt_enabled(dev));
   dev->chip = AGX_CHIP_G13G;
   dev->params.features = DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS |
                          DRM_ASAHI_FEATURE_RENDER_RAY_QUERY;
   T("rt off outside G17P", !hk_g17p_rt_enabled(dev));
}

/* Exercise the production payload helper against upstream's prefix contract.
 * A packet is private to an occurrence, even when its hk_cs is shared. */
static void
check_classic_packet(struct hk_cs *cs, const unsigned prefix[2],
                      const uint64_t base[2], bool exact)
{
   struct drm_asahi_cmd_header packet = agx_cmd_header(
      cs->type == HK_CS_CDM, cs->sync.barrier[0], cs->sync.barrier[1]);
   uint64_t prior[2] = {cs->sync.prior[0], cs->sync.prior[1]};
   T("classic payload prefix", hk_drm_serialize_prefix(
        &packet, prior, prefix[0], prefix[1], base, exact));
   const unsigned barriers[2] = {packet.vdm_barrier, packet.cdm_barrier};
   for (unsigned sq = 0; sq < 2; ++sq) {
      bool historical = exact && prefix[sq] == 0;
      T("classic upstream prefix", barriers[sq] ==
         (historical ? DRM_ASAHI_BARRIER_NONE : prefix[sq]));
      T("classic prior ordinal", prior[sq] == (historical ? base[sq] : 0));
   }
}

static void
test_classic_submission_order(void)
{
   for (unsigned exact = 0; exact < 2; ++exact) {
      for (unsigned groups = 1; groups <= 4; groups *= 2) {
         for (unsigned mix = 0; mix < 16; ++mix) {
            struct hk_queue *q = new_queue(exact);
            q->sync.serialize = true;
            q->sync.submitted[0] = 7;
            q->sync.submitted[1] = 11;
            struct hk_cmd_buffer *cmds[4];
            struct hk_cs *streams[4];
            for (unsigned i = 0; i < groups; ++i)
               cmds[i] = new_cmd();
            for (unsigned i = 0; i < 4; ++i)
               streams[i] = add(cmds[i * groups / 4], mix & (1u << i));
            T("classic mixed submission", hk_sync_resolve_submit(
                 q, cmds, groups, false, false, 0));
            unsigned prefix[2] = {0, 0};
            const uint64_t base[2] = {7, 11};
            for (unsigned i = 0; i < 4; ++i) {
               check_classic_packet(streams[i], prefix, base, exact);
               prefix[hk_sync_sq(streams[i])]++;
            }
            struct hk_cmd_buffer *next = new_cmd();
            struct hk_cs *read = add(next, true);
            const uint64_t next_base[2] = {7 + prefix[0], 11 + prefix[1]};
            const unsigned zero[2] = {0, 0};
            T("classic later read submission", submit(q, next));
            check_classic_packet(read, zero, next_base, exact);
         }
      }

      /* NONE event set/reset reuses one simultaneous-use recording in two
       * merged VkSubmitInfo2 entries. Include empty buffers and graphics too. */
      struct hk_queue *q = new_queue(exact);
      q->sync.serialize = true;
      q->sync.submitted[0] = 7;
      q->sync.submitted[1] = 11;
      struct hk_cmd_buffer *empty = new_cmd(), *event = new_cmd();
      struct hk_cmd_buffer *draw = new_cmd();
      add(event, true);
      add(draw, false);
      struct hk_cmd_buffer *cmds[] = {empty, event, event, draw, event};
      T("classic repeated recording", hk_sync_resolve_submit(
           q, cmds, 5, false, false, 0));
      unsigned prefix[2] = {0, 0};
      const uint64_t base[2] = {7, 11};
      for (unsigned i = 0; i < 5; ++i) {
         list_for_each_entry(struct hk_cs, cs, &cmds[i]->control_streams, node) {
            check_classic_packet(cs, prefix, base, exact);
            prefix[hk_sync_sq(cs)]++;
         }
      }
      T("classic repeated ordinals", !exact ||
         (q->sync.submitted[0] == 8 && q->sync.submitted[1] == 14));
   }

   struct drm_asahi_cmd_header packet = agx_cmd_header(true, 0, 0);
   struct drm_asahi_cmd_header saved = packet;
   uint64_t prior[2] = {3, 5}, base[2] = {7, 11};
   T("classic prefix overflow rejected", !hk_drm_serialize_prefix(
        &packet, prior, DRM_ASAHI_BARRIER_NONE, 0, base, true));
   T("classic prefix overflow no mutation", !memcmp(&packet, &saved,
      sizeof(packet)) && prior[0] == 3 && prior[1] == 5);

   /* Native independent streams keep their original graph policy. */
   struct hk_queue *q = new_queue(true);
   struct hk_cmd_buffer *cmd = new_cmd();
   add(cmd, false);
   struct hk_cs *compute = add(cmd, true);
   T("native independent submission", submit(q, cmd));
   T("native remains unordered", compute->sync.barrier[0] ==
      DRM_ASAHI_BARRIER_NONE && compute->sync.prior[0] == 0);
}

static void
test_barrier_chains_without_work(void)
{
   struct hk_cmd_buffer *cmd = new_cmd();
   cmd->sync.kind_override = HK_SYNC_XFER;
   struct hk_cs *copy = add(cmd, true);
   cmd->sync.kind_override = 0;
   hk_sync_barrier(cmd, HK_SYNC_XFER, HK_SYNC_COMPUTE);
   hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   struct hk_cs *draw = add(cmd, false);
   T("transfer-compute-graphics chain retains transfer source",
     draw->sync.dep[HK_SYNC_CDM].cut == copy);
   T("transfer-compute-graphics chain reaches vertex work",
     draw->sync.dep_ta[HK_SYNC_CDM].cut == copy);

   cmd = new_cmd();
   cmd->sync.kind_override = HK_SYNC_XFER;
   copy = add(cmd, true);
   cmd->sync.kind_override = 0;
   hk_sync_barrier(cmd, HK_SYNC_XFER, HK_SYNC_COMPUTE);
   hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_GFX);
   draw = add(cmd, false);
   T("disjoint barrier scopes do not serialize unrelated graphics",
     draw->sync.dep[HK_SYNC_CDM].cut == NULL);

   cmd = new_cmd();
   cmd->sync.kind_override = HK_SYNC_XFER;
   copy = add(cmd, true);
   cmd->sync.kind_override = 0;
   hk_sync_barrier(cmd, HK_SYNC_XFER, HK_SYNC_GFX_FRAG);
   hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_COMPUTE);
   struct hk_cs *compute = add(cmd, true);
   T("fragment destination chains into a later graphics source",
     compute->sync.dep[HK_SYNC_CDM].cut == copy);
}

/* Differential check of production resolution and history, including ring
 * wrap, retired points, mixed engines, empty buffers and multi-buffer fallback.
 * The two queues consume the same immutable recordings in the same order.
 */
static void
test_direct_single_footprints(void)
{
   const unsigned sizes[] = {0, 1, 2, HK_SYNC_RECENT_FOOTPRINTS + 1};
   for (unsigned exact = 0; exact < 2; ++exact) {
      for (unsigned serialized = 0; serialized < 2; ++serialized) {
         for (unsigned count = 0; count <= 2; ++count) {
            for (unsigned s = 0; s < ARRAY_SIZE(sizes); ++s) {
               struct hk_queue *a = new_queue(exact), *b = new_queue(exact);
               a->sync.serialize = b->sync.serialize = serialized;
               a->sync.precise_tile_evictions = b->sync.precise_tile_evictions = true;
               b->sync.direct_single_footprints = true;
               struct hk_cmd_buffer *cmds[] = {new_cmd(), new_cmd()};
               struct hk_cs **streams = test_alloc(
                  sizeof(*streams) * MAX2(1, 2 * sizes[s]));
               struct hk_sync_stream *saved = test_alloc(
                  sizeof(*saved) * MAX2(1, 2 * sizes[s]));
               unsigned n = 0;
               for (unsigned c = 0; c < count; ++c) {
                  for (unsigned p = 0; p < sizes[s]; ++p) {
                     const void *image = (void *)(uintptr_t)(0x1000 + (p % 3) * 16);
                     streams[n++] = (p & 1)
                        ? compute_clear_pass(cmds[c], image, 0, true)
                        : render_pass(cmds[c], image, 1, p % 3 == 0);
                  }
               }
               bool same = true;
               const unsigned rounds = sizes[s]
                  ? HK_SYNC_RECENT_FOOTPRINTS / sizes[s] + 3 : 3;
               for (unsigned r = 0; r < rounds; ++r) {
                  a->drm.timeline_value = b->drm.timeline_value = r + 1;
                  const uint64_t completed = r % 4 ? 0 : r;
                  const bool flush = r % 7 == 0, chain = r % 3 == 0;
                  same &= hk_sync_resolve_submit(a, cmds, count, flush, chain, completed);
                  for (unsigned j = 0; j < n; ++j)
                     saved[j] = streams[j]->sync;
                  same &= hk_sync_resolve_submit(b, cmds, count, flush, chain, completed);
                  for (unsigned j = 0; j < n; ++j) {
                     same &= saved[j].index == streams[j]->sync.index &&
                             saved[j].ta_split == streams[j]->sync.ta_split;
                     for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
                        same &= saved[j].barrier[sq] == streams[j]->sync.barrier[sq] &&
                                saved[j].prior[sq] == streams[j]->sync.prior[sq] &&
                                saved[j].barrier_ta[sq] == streams[j]->sync.barrier_ta[sq] &&
                                saved[j].prior_ta[sq] == streams[j]->sync.prior_ta[sq];
                     }
                  }
                  same &= a->sync.recent_count == b->sync.recent_count &&
                          a->sync.recent_next == b->sync.recent_next &&
                          a->sync.recent_max == b->sync.recent_max &&
                          a->sync.evicted == b->sync.evicted &&
                          a->sync.last_preserve == b->sync.last_preserve &&
                          a->sync.evicted_preserve == b->sync.evicted_preserve;
                  same &= memcmp(a->sync.ever, b->sync.ever, sizeof(a->sync.ever)) == 0 &&
                          memcmp(a->sync.tail, b->sync.tail, sizeof(a->sync.tail)) == 0 &&
                          memcmp(a->sync.last_absolute, b->sync.last_absolute,
                                 sizeof(a->sync.last_absolute)) == 0 &&
                          memcmp(a->sync.submitted, b->sync.submitted,
                                 sizeof(a->sync.submitted)) == 0;
                  for (unsigned j = 0; j < a->sync.recent_count; ++j) {
                     const struct hk_sync_recent *x = &a->sync.recent[j], *y = &b->sync.recent[j];
                     same &= x->point == y->point && x->sq == y->sq &&
                             x->fp.image == y->fp.image && x->fp.level == y->fp.level &&
                             x->fp.preserve == y->fp.preserve &&
                             x->fp.layer0 == y->fp.layer0 && x->fp.layer1 == y->fp.layer1 &&
                             x->fp.cx0 == y->fp.cx0 && x->fp.cx1 == y->fp.cx1 &&
                             x->fp.cy0 == y->fp.cy0 && x->fp.cy1 == y->fp.cy1;
                  }
               }
               T("direct single-buffer footprint resolution/history matches", same);
            }
         }
      }
   }
}

int
main(void)
{
   test_context = ralloc_context(NULL);
   test_classic_submission_order();
   if (!test_context)
      return EXIT_FAILURE;
   test_precise_tile_evictions();
   test_direct_single_footprints();
   test_xfer_write_order();
   test_barrier_chains_without_work();
   test_multiview_layer_preservation();
   test_compute_clear_layer_preservation();
   test_resume_after_compute_clear();
   test_loaded_color_preservation();
   test_pass_software_graphics_dependencies();
   test_image_barrier_vertex_class();
   test_g17p_vertex_after_memory_writers();
   test_chain_first_skips_empty_buffers();
   test_rt_kernel_gate();
   test_timestamp_execution_scopes();
   test_timestamp_reused_compute();
   test_timestamp_top_in_pass();
   {
      /* A submission longer than the 16-bit barrier field: a wait on a
       * command past it is named by its ordinal (that command went in an
       * earlier ioctl). vkd3d-proton's test_resolve_image_exhaustive_descriptors
       * records 65536 resolves in one command list. */
      struct hk_queue *lq = new_queue(true);
      struct hk_cmd_buffer *lc = new_cmd();
      const unsigned n = 70000;
      for (unsigned k = 0; k < n; ++k)
         add(lc, true);
      hk_sync_barrier(lc, HK_SYNC_COMPUTE, HK_SYNC_GFX);
      struct hk_cs *g = add(lc, false);
      T("long submission resolves", submit(lq, lc));
      T("far wait becomes an ordinal",
        g->sync.barrier[1] == DRM_ASAHI_BARRIER_NONE && g->sync.prior[1] == n);
   }

   /* Each matching destination retains the exact old source prefix, including
    * reuse of a recording and independent physical destinations on the queue. */
   struct hk_queue *q = new_queue(true);
   struct hk_cmd_buffer *a = new_cmd(), *b = new_cmd();
   add(a, true);
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   struct hk_cs *d = add(b, false);
   T("first submit", submit(q, a));
   T("later graphics submit", submit(q, b));
   T("exact old compute prefix",
     d->sync.prior[1] == 1 && d->sync.barrier[1] == DRM_ASAHI_BARRIER_NONE);
   struct hk_cmd_buffer *c = new_cmd();
   struct hk_cs *i = add(c, true);
   T("unrelated compute submit", submit(q, c));
   T("unrelated compute no graphics dependency",
     i->sync.prior[0] == 0 && i->sync.barrier[0] == DRM_ASAHI_BARRIER_NONE);
   T("repeat recorded graphics", submit(q, b));
   T("repeat still names original1 not newer2",
     d->sync.prior[1] == 1 && d->sync.barrier[1] == DRM_ASAHI_BARRIER_NONE);
   c = new_cmd();
   add(c, true);
   hk_sync_barrier(c, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   T("new tail submit", submit(q, c));
   T("repeat after new barrier", submit(q, b));
   T("new tail names compute3", d->sync.prior[1] == 3);
   struct hk_queue *isolated = new_queue(true);
   T("other queue same CB", submit(isolated, b));
   T("queue isolation no inherited dependency",
     d->sync.prior[1] == 0 && d->sync.barrier[1] == DRM_ASAHI_BARRIER_NONE);
   T("original queue same CB again", submit(q, b));
   T("no baked queue-specific prior in recording", d->sync.prior[1] == 3);
   q = new_queue(true);
   a = new_cmd();
   b = new_cmd();
   c = new_cmd();
   add(a, false);
   hk_sync_barrier(b, HK_SYNC_GFX, HK_SYNC_COMPUTE);
   d = add(c, true);
   T("barrier-only source", submit(q, a));
   T("barrier-only submit", submit(q, b));
   T("barrier-only destination", submit(q, c));
   T("barrier-only prior render1",
     d->sync.prior[0] == 1 && d->sync.barrier[0] == DRM_ASAHI_BARRIER_NONE);
   q = new_queue(true);
   a = new_cmd();
   b = new_cmd();
   add(a, true);
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   d = add(b, false);
   struct hk_cmd_buffer *pair[] = {a, b};
   T("same-submit", hk_sync_resolve_submit(q, pair, 2, false, false, 0));
   T("same-submit keeps relative1",
     d->sync.barrier[1] == 1 && d->sync.prior[1] == 0);
   q = new_queue(true);
   a = new_cmd();
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   T("empty source barrier", submit(q, a));
   b = new_cmd();
   d = add(b, false);
   T("empty source destination", submit(q, b));
   T("empty source no future dependency",
     d->sync.prior[1] == 0 && d->sync.barrier[1] == DRM_ASAHI_BARRIER_NONE);
   q = new_queue(true);
   T("synthetic SSTT only", hk_sync_resolve_submit(q, NULL, 0, true, false, 0));
   T("SSTT increments actual CDM ordinal",
     q->sync.submitted[1] == 1 && q->sync.last_absolute[1][1] == 1);
   a = new_cmd();
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   T("tail after SSTT", submit(q, a));
   b = new_cmd();
   d = add(b, false);
   T("destination after SSTT tail", submit(q, b));
   T("SSTT historical dependency1", d->sync.prior[1] == 1);
   q = new_queue(false);
   a = new_cmd();
   b = new_cmd();
   add(a, true);
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   d = add(b, false);
   T("legacy source", submit(q, a));
   T("legacy destination", submit(q, b));
   T("legacy unchanged no new history",
     d->sync.prior[1] == 0 && q->sync.submitted[1] == 0 &&
        d->sync.barrier[1] == DRM_ASAHI_BARRIER_NONE);
   q = new_queue(true);
   q->sync.submitted[1] = UINT64_MAX;
   a = new_cmd();
   d = add(a, true);
   d->sync.index = 73;
   T("ordinal overflow rejected", !submit(q, a));
   T("overflow no counter/stream mutation",
     q->sync.submitted[1] == UINT64_MAX && d->sync.index == 73);
   struct drm_asahi_cmd_render_with_prior r = {0};
   struct drm_asahi_cmd_header h = {.cmd_type = DRM_ASAHI_CMD_RENDER,
                                    .size = sizeof(r),
                                    .vdm_barrier = 2,
                                    .cdm_barrier = 5};
   uint64_t base[] = {10, 20};
   T("split rebase", hk_drm_rebase_prior(&h, &r, 3, 4, base, true));
   T("prior slice gets exact global12",
     h.vdm_barrier == DRM_ASAHI_BARRIER_NONE && r.prior_vdm == 12);
   T("same slice keeps relative1", h.cdm_barrier == 1 && r.prior_cdm == 0);
   h.vdm_barrier = 0;
   h.cdm_barrier = DRM_ASAHI_BARRIER_NONE;
   T("deliberate zero rebase", hk_drm_rebase_prior(&h, &r, 7, 8, base, true));
   T("deliberate zero preserved", h.vdm_barrier == 0 && r.prior_vdm == 12);
   struct drm_asahi_cmd_compute_with_prior co = {0};
   h = (struct drm_asahi_cmd_header){.cmd_type = DRM_ASAHI_CMD_COMPUTE,
                                     .size = sizeof(co),
                                     .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
                                     .cdm_barrier = 1};
   T("compute split layout", hk_drm_rebase_prior(&h, &co, 0, 1, base, true));
   T("compute prior21 scs remains zero",
     co.prior_cdm == 21 && co.scs_layout == 0 &&
        h.cdm_barrier == DRM_ASAHI_BARRIER_NONE);
   h.cdm_barrier = 1;
   base[1] = UINT64_MAX;
   struct drm_asahi_cmd_header before = h;
   T("split overflow rejected",
     !hk_drm_rebase_prior(&h, &co, 0, 1, base, true));
   T("split overflow no header mutation", !memcmp(&h, &before, sizeof(h)));
   h.size = 64;
   h.cdm_barrier = 1;
   base[1] = 20;
   T("legacy split", hk_drm_rebase_prior(&h, &co, 0, 1, base, false));
   T("legacy split old zero", h.cdm_barrier == 0);

   /* Sparse maintenance contributes exactly one hardware ordinal before app
    * streams. */
   q = new_queue(true);
   a = new_cmd();
   d = add(a, true);
   hk_sync_barrier(a, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   struct hk_cmd_buffer *single[] = {a};
   T("SSTT plus application",
     hk_sync_resolve_submit(q, single, 1, true, false, 0));
   T("SSTT plus app counters", q->sync.submitted[1] == 2 &&
                                  d->sync.index == 2 &&
                                  d->sync.barrier[1] == 1);
   b = new_cmd();
   d = add(b, false);
   T("SSTT app later destination", submit(q, b));
   T("SSTT app exact prefix2", d->sync.prior[1] == 2);
   /* A later barrier-only buffer with both source kinds keeps distinct engine
    * ordinals. */
   q = new_queue(true);
   a = new_cmd();
   add(a, true);
   add(a, false);
   T("both engine sources", submit(q, a));
   b = new_cmd();
   hk_sync_barrier(b, HK_SYNC_ALL, HK_SYNC_ALL);
   T("all-kind standalone barrier", submit(q, b));
   c = new_cmd();
   d = add(c, true);
   T("all-kind destination", submit(q, c));
   T("VDM CDM order",
     d->sync.prior[HK_SYNC_VDM] == 1 && d->sync.prior[HK_SYNC_CDM] == 1);
   /* A submission past the 16-bit relative fields resolves (far waits become
    * ordinals) and the queue counts every command. */
   q = new_queue(true);
   a = new_cmd();
   for (unsigned n = 0; n < 65535; n++)
      add(a, true);
   T("relative overflow resolves", submit(q, a));
   T("relative overflow counted", q->sync.submitted[1] == 65535);
   /* Legacy packet prefixes retain all original bytes, extensions are omitted.
    */
   struct drm_asahi_cmd_render lr;
   memset(&lr, 0xa5, sizeof(lr));
   struct drm_asahi_cmd_render_with_prior er = {.legacy = lr,
                                                .prior_vdm = 17,
                                                .prior_cdm = 29};
   T("legacy render bytes", memcmp(&lr, &er, sizeof(lr)) == 0);
   struct drm_asahi_cmd_compute lc;
   memset(&lc, 0x5a, sizeof(lc));
   struct drm_asahi_cmd_compute_with_prior ec = {.legacy = lc,
                                                 .prior_vdm = 17,
                                                 .prior_cdm = 29};
   T("legacy compute bytes", memcmp(&lc, &ec, sizeof(lc)) == 0);
   printf("RESULT checks=%u failures=%u\n", tests, failures);
   ralloc_free(test_context);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

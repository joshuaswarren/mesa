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
#include "agx_chip_caps.h"
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

/* On an exact-prior queue, a tile hazard against a possibly still running
 * earlier submission names everything before the submission by its starting
 * ordinal, in both halves of a render, and no relative barrier: a relative
 * zero would name everything before a later ioctl of a split submission. */
static bool
waits_on_history(const struct hk_cs *cs, unsigned sq, uint64_t start)
{
   return start != 0 &&
          cs->sync.barrier[sq] == DRM_ASAHI_BARRIER_NONE &&
          cs->sync.prior[sq] == start &&
          cs->sync.barrier_ta[sq] == DRM_ASAHI_BARRIER_NONE &&
          cs->sync.prior_ta[sq] == start;
}

/* No wait on subqueue sq in either half. */
static bool
waits_on_nothing(const struct hk_cs *cs, unsigned sq)
{
   return cs->sync.barrier[sq] == DRM_ASAHI_BARRIER_NONE &&
          cs->sync.prior[sq] == 0 &&
          cs->sync.barrier_ta[sq] == DRM_ASAHI_BARRIER_NONE &&
          cs->sync.prior_ta[sq] == 0;
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
   return hk_sync_resolve_submit(q, cs, 1, false, false, 0, NULL);
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

/* Chips whose start stamps latch before a command's waits (hk_sync.c,
 * start_stamps_ordered()): G17P and M5. */
static const enum agx_chip end_stamp_chips[] = {
   AGX_CHIP_G17P, AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C, AGX_CHIP_G17D};

/* G17P and M5: ResetQueryPool leaves a reusable compute stream. A TOP_OF_PIPE
 * stamp after it ends that stream's command: start stamps latch before the
 * command's waits, so the stamp must not become the start of the dispatch that
 * reuses the stream, and ending the reset's command keeps it after the reset.
 * The dispatch gets a command of its own and the bottom stamp ends it.
 */
static void
test_timestamp_reused_compute(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(end_stamp_chips); ++c) {
      for (unsigned exact = 0; exact < 2; ++exact) {
         struct hk_cmd_buffer *cmd = timestamp_cmd(end_stamp_chips[c], exact);
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
}

/* G17P and M5: a TOP_OF_PIPE stamp inside a render pass instance ends the
 * render (fragment end), never its vertex start: start stamps latch before the
 * render's waits, so a query reset before the pass could overwrite one. */
static void
test_timestamp_top_in_pass(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(end_stamp_chips); ++c) {
      struct hk_cmd_buffer *cmd = timestamp_cmd(end_stamp_chips[c], true);
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
}

/* G17P and M5: a leading TOP_OF_PIPE timestamp with no earlier work or barrier
 * in the command buffer rides on the first stream as an unordered start stamp
 * (saving a separate carrier command when submission has no prior barrier,
 * input wait, or GPU query reset), while a TOP_OF_PIPE after a barrier stays
 * on the end-stamp path. */
static void
test_timestamp_leading_top(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(end_stamp_chips); ++c) {
      /* Compute: leading TOP_OF_PIPE + dispatch + ALL_COMMANDS -> one stream. */
      struct hk_cmd_buffer *cmd = timestamp_cmd(end_stamp_chips[c], true);
      const struct agx_timestamp_req top = {
         .handle = 1, .addr = 0x10000, .offset_B = 0};
      hk_sync_write_timestamp(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, top);
      T("leading top stamp defers to pending start",
        cmd->sync.pending_start_ts.handle == 1 &&
        cmd->sync.pending_start_ts_unordered &&
        list_is_empty(&cmd->control_streams));

      struct hk_cs *dispatch = add(cmd, true);
      cmd->current_cs.cs = dispatch;
      dispatch->stats.cmds++;
      T("first compute stream takes unordered start stamp",
        dispatch->timestamp.start.handle == 1 &&
        dispatch->timestamp.start.offset_B == 0 &&
        dispatch->unordered_start_ts &&
        cmd->sync.has_unordered_start_ts &&
        !cmd->sync.pending_start_ts.handle &&
        !cmd->sync.pending_start_ts_unordered);

      end_timestamp(cmd, 1);
      T("leading top plus bottom stays one stream",
        list_length(&cmd->control_streams) == 1 &&
        dispatch->timestamp.end.handle == 1 &&
        dispatch->timestamp.end.offset_B == 8);

      /* Clean queue has no tail; barrier-only submit raises tail once and
       * later unbarriered submits do not raise it again; fallback carrier
       * shifts ordinal to 2. */
      struct hk_queue *q = new_queue(true);
      uint64_t zero_tail[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ] = {0};
      T("fresh queue has zero tail",
        memcmp(q->sync.tail, zero_tail, sizeof(zero_tail)) == 0);
      T("leading top resolves on clean queue", submit(q, cmd));
      T("unbarriered submit leaves tail unchanged",
        memcmp(q->sync.tail, zero_tail, sizeof(zero_tail)) == 0);

      struct hk_cmd_buffer *bar = timestamp_cmd(end_stamp_chips[c], true);
      hk_sync_barrier(bar, HK_SYNC_ALL, HK_SYNC_ALL);
      T("barrier-only submit resolves", submit(q, bar));
      T("barrier-only submit raises queue tail",
        memcmp(q->sync.tail, zero_tail, sizeof(zero_tail)) != 0);

      uint64_t saved_tail[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];
      memcpy(saved_tail, q->sync.tail, sizeof(saved_tail));
      struct hk_cmd_buffer *one[] = {cmd};
      T("fallback carrier resolves with shifted ordinal",
        hk_sync_resolve_submit(q, one, 1, true, false, 0, NULL) &&
        dispatch->sync.index == 2 &&
        dispatch->sync.barrier[HK_SYNC_CDM] == 1);
      T("unbarriered submit after tail does not raise tail further",
        memcmp(q->sync.tail, saved_tail, sizeof(saved_tail)) == 0);

      /* Render: leading TOP_OF_PIPE before BeginRendering rides on VDM start. */
      struct hk_cmd_buffer *rcmd = timestamp_cmd(end_stamp_chips[c], true);
      hk_sync_write_timestamp(rcmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, top);
      hk_sync_begin_rendering(rcmd, false);
      struct hk_cs *render = add(rcmd, false);
      T("first render takes unordered start stamp",
        render->timestamp.start.handle == 1 &&
        render->unordered_start_ts &&
        rcmd->sync.has_unordered_start_ts);

      /* A TOP_OF_PIPE recorded after a barrier in the same command buffer
       * must NOT defer to unordered_start_ts -- even when no stream has been
       * recorded yet (exercises the sync_state_has_req() branch directly). */
      struct hk_cmd_buffer *bcmd = timestamp_cmd(end_stamp_chips[c], true);
      T("fresh cmdbuf can defer leading top stamp",
        hk_sync_can_defer_unordered_start_ts(bcmd));
      hk_sync_barrier(bcmd, HK_SYNC_ALL, HK_SYNC_ALL);
      T("barrier records pending req before top stamp",
        bcmd->sync.req[0][HK_SYNC_VDM].prior != 0);
      T("barrier with no stream prevents unordered top deferral",
        !hk_sync_can_defer_unordered_start_ts(bcmd));
      struct hk_cs *prev = add(bcmd, true);
      prev->stats.cmds++;
      hk_sync_write_timestamp(bcmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, top);
      T("top stamp after barrier stays ordered end stamp",
        !bcmd->sync.pending_start_ts.handle &&
        !bcmd->sync.has_unordered_start_ts &&
        prev->timestamp.end.handle == 1);
   }
}

/* Exercise actual timestamp placement, barriers and the submission resolver.
 * Existing query-command scopes are not removed by the timestamp scope's
 * redundant same-engine execution edge. No command payload or GPU is needed.
 */
static void
test_timestamp_execution_scopes(void)
{
   T("start_stamps_ordered distinguishes G16G/AGX2 from G17P/M5",
     agx_chip_start_stamps_ordered(AGX_CHIP_G13G) &&
     agx_chip_start_stamps_ordered(AGX_CHIP_G14G) &&
     agx_chip_start_stamps_ordered(AGX_CHIP_G15G) &&
     agx_chip_start_stamps_ordered(AGX_CHIP_G16G) &&
     !agx_chip_start_stamps_ordered(AGX_CHIP_G17P) &&
     !agx_chip_start_stamps_ordered(AGX_CHIP_G17G) &&
     !agx_chip_start_stamps_ordered(AGX_CHIP_G17S) &&
     !agx_chip_start_stamps_ordered(AGX_CHIP_G17C));

   const enum agx_chip ts_chips[] = {
      AGX_CHIP_G16G, AGX_CHIP_G17P, AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };
   for (unsigned c = 0; c < ARRAY_SIZE(ts_chips); ++c) {
      const enum agx_chip chip = ts_chips[c];
      for (unsigned exact = 0; exact < 2; ++exact) {
      struct hk_queue *queue = new_queue(exact);
      struct hk_cmd_buffer *first = timestamp_cmd(chip, exact);
      add(first, true);
      end_timestamp(first, 0);
      T("first timed compute resolves", submit(queue, first));
      struct hk_cmd_buffer *second = timestamp_cmd(chip, exact);
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
      struct hk_cmd_buffer *query = timestamp_cmd(chip, exact);
      hk_sync_barrier(query, HK_SYNC_ALL, HK_SYNC_ALL);
      struct hk_cs *copy = add(query, true);
      copy->g17p_timestamp_query = true;
      end_timestamp(query, 2);
      T("query consumer plus timestamp resolves", submit(queue, query));
      T("query memory consumer retains the completed producer prefix",
        exact ? copy->sync.prior[HK_SYNC_CDM] == 2
              : copy->sync.barrier[HK_SYNC_CDM] == 0);

      struct hk_cmd_buffer *barrier = timestamp_cmd(chip, exact);
      hk_sync_barrier(barrier, HK_SYNC_ALL, HK_SYNC_ALL);
      T("standalone query ordering resolves", submit(queue, barrier));
      struct hk_cmd_buffer *after = timestamp_cmd(chip, exact);
      struct hk_cs *inherited = add(after, true);
      end_timestamp(after, 3);
      T("later timestamp with carried requirements resolves", submit(queue, after));
      T("timestamp preserves a carried query-memory requirement",
        exact ? inherited->sync.prior[HK_SYNC_CDM] == 3
              : inherited->sync.barrier[HK_SYNC_CDM] == 0);

      queue = new_queue(exact);
      first = timestamp_cmd(chip, exact);
      add(first, false);
      T("earlier render resolves", submit(queue, first));
      second = timestamp_cmd(chip, exact);
      timed = add(second, true);
      end_timestamp(second, 4);
      T("timestamp after render resolves", submit(queue, second));
      T("timestamp cross-engine scope is preserved",
        exact ? timed->sync.prior[HK_SYNC_VDM] == 1
              : timed->sync.barrier[HK_SYNC_VDM] == 0);

      queue = new_queue(exact);
      first = timestamp_cmd(chip, exact);
      add(first, true);
      end_timestamp(first, 5);
      T("reset producer timestamp resolves", submit(queue, first));
      struct hk_cmd_buffer *reset_cmd = timestamp_cmd(chip, exact);
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
      first = timestamp_cmd(chip, exact);
      add(first, true);
      T("render timestamp compute source resolves", submit(queue, first));
      second = timestamp_cmd(chip, exact);
      struct hk_cs *render = add(second, false);
      end_timestamp(second, 7);
      T("render end timestamp resolves", submit(queue, second));
      T("render timestamp keeps the original compute scope",
        exact ? render->sync.prior[HK_SYNC_CDM] == 1
              : render->sync.barrier[HK_SYNC_CDM] == 0);
   }

   }

   /* Unqualified targets (AGX2 and future G18G) retain the original same-CDM
    * timestamp prefix even if exact-prior barriers are advertised. */
   for (unsigned chip = 0; chip < 2; ++chip) {
      const enum agx_chip target = chip ? AGX_CHIP_G18G : AGX_CHIP_G13G;
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
/* A timestamp-only command (HK_SYNC_STAMP) waits like work of every class
 * but is in no application first scope: a barrier, a later command buffer
 * and a later submission order renders and dispatches after the work around
 * it, never after it. Query commands still follow it. */
static void
test_stamp_kind_isolation(void)
{
   for (unsigned exact = 0; exact < 2; ++exact) {
      struct hk_cmd_buffer *cmd = timestamp_cmd(AGX_CHIP_G17S, exact);
      struct hk_cs *work = add(cmd, true);
      hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
      cmd->sync.kind_override = HK_SYNC_STAMP;
      struct hk_cs *carrier = add(cmd, true);
      cmd->sync.kind_override = 0;
      T("timestamp stream takes the requirements of every class",
        carrier->sync.dep[HK_SYNC_CDM].cut == work &&
        (hk_sync_full_kinds(carrier->sync.kind) & HK_SYNC_ALL) == HK_SYNC_ALL);
      hk_sync_barrier(cmd, HK_SYNC_ALL, HK_SYNC_ALL);
      struct hk_cs *render = add(cmd, false);
      struct hk_cs *after = add(cmd, true);
      T("ALL_COMMANDS barrier orders later work after work, not the stamp",
        render->sync.dep[HK_SYNC_CDM].cut == work &&
        after->sync.dep[HK_SYNC_CDM].cut == work);
      struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
      hk_sync_src_dep(cmd, HK_SYNC_ALL, NULL, dep);
      T("application scope skips the stamp",
        dep[HK_SYNC_CDM].cut == after);
      hk_sync_src_dep(cmd, hk_sync_stamp_scope(HK_SYNC_GFX), NULL, dep);
      T("a timestamp's scope names the stamp",
        dep[HK_SYNC_CDM].cut == carrier);
      T("an empty timestamp scope names nothing",
        hk_sync_stamp_scope(0) == 0);

      /* Across submissions: the stamp alone on the compute subqueue. */
      struct hk_queue *q = new_queue(exact);
      struct hk_cmd_buffer *first = timestamp_cmd(AGX_CHIP_G17S, exact);
      add(first, false);
      first->sync.kind_override = HK_SYNC_STAMP;
      struct hk_cs *stamp = add(first, true);
      first->sync.kind_override = 0;
      T("stamp submission resolves", submit(q, first));
      struct hk_cmd_buffer *app = timestamp_cmd(AGX_CHIP_G17S, exact);
      hk_sync_barrier(app, HK_SYNC_ALL, HK_SYNC_ALL);
      struct hk_cs *dispatch = add(app, true);
      T("later dispatch resolves", submit(q, app));
      T("later dispatch does not wait for the stamp",
        dispatch->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE &&
        dispatch->sync.prior[HK_SYNC_CDM] == 0);
      /* And a query command right after it, on a queue of its own. */
      struct hk_queue *qq = new_queue(exact);
      struct hk_cmd_buffer *stamped = timestamp_cmd(AGX_CHIP_G17S, exact);
      add(stamped, false);
      stamped->sync.kind_override = HK_SYNC_STAMP;
      struct hk_cs *only = add(stamped, true);
      stamped->sync.kind_override = 0;
      T("second stamp submission resolves", submit(qq, stamped));
      struct hk_cmd_buffer *query = timestamp_cmd(AGX_CHIP_G17S, exact);
      hk_sync_barrier(query, HK_SYNC_QUERY_SCOPE, HK_SYNC_ALL);
      struct hk_cs *copy = add(query, true);
      T("later query command resolves", submit(qq, query));
      T("later query command waits for the stamp",
        exact ? copy->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE &&
                   copy->sync.prior[HK_SYNC_CDM] == only->sync.index
              : copy->sync.barrier[HK_SYNC_CDM] == 0);
      T("the stamp submission's compute command is the stamp",
        stamp->sync.index == 1 && only->sync.index == 1);
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
                         (VkRect2D){{0, 0}, {32, 32}}, preserve_layers, false);
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
      no_barrier &= waits_on_nothing(write, HK_SYNC_VDM);
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
   uint64_t start = q->sync.submitted[HK_SYNC_VDM];
   T("later preserving pass resolves", submit(q, partial));
   T("later preserving pass waits on evicted complete writes",
     waits_on_history(preserve, HK_SYNC_VDM, start) &&
     q->sync.last_preserve == point);
   T("complete pass queries preserving history",
     hk_sync_submit_needs_completed(q, one, 1));
   ++q->drm.timeline_value;
   start = q->sync.submitted[HK_SYNC_VDM];
   T("complete pass resolves against preserving history", submit(q, full));
   T("complete pass waits on live preserving pass",
     waits_on_history(write, HK_SYNC_VDM, start));

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
     hk_sync_resolve_submit(q, one, 1, false, false, point, NULL) &&
     waits_on_nothing(write, HK_SYNC_VDM));
   ++q->drm.timeline_value;
   start = q->sync.submitted[HK_SYNC_VDM];
   T("partial pass still waits on live evicted complete writes",
     hk_sync_resolve_submit(q, p, 1, false, false, point, NULL) &&
     waits_on_history(preserve, HK_SYNC_VDM, start));

   q = new_queue(true);
   q->sync.precise_tile_evictions = true;
   q->drm.timeline_value = 1;
   struct hk_cmd_buffer *pair[] = {partial, full};
   T("partial then full in one submit retains dependency",
     hk_sync_resolve_submit(q, pair, 2, false, false, 0, NULL) &&
     write->sync.barrier[HK_SYNC_VDM] == preserve->sync.index);
   q = new_queue(true);
   q->sync.precise_tile_evictions = true;
   q->drm.timeline_value = 1;
   struct hk_cmd_buffer *reverse[] = {full, partial};
   T("full then partial in one submit retains dependency",
     hk_sync_resolve_submit(q, reverse, 2, false, false, 0, NULL) &&
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
   T("sparse layer store vertex half waits on the clear too",
     sparse->sync.dep_ta[HK_SYNC_VDM].cut == first);
   struct hk_queue *queue = new_queue(true);
   T("multiview same command buffer submit", submit(queue, cmd));
   T("layer preservation resolves to exact earlier render",
     sparse->sync.barrier[HK_SYNC_VDM] == first->sync.index &&
     sparse->sync.barrier_ta[HK_SYNC_VDM] == first->sync.index);

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
     hk_sync_resolve_submit(queue, pair, 2, false, false, 0, NULL));
   T("across command buffers keeps layer store ordered",
     sparse->sync.barrier[HK_SYNC_VDM] == first->sync.index &&
     sparse->sync.barrier_ta[HK_SYNC_VDM] == first->sync.index);

   queue = new_queue(true);
   a = new_cmd(); b = new_cmd();
   render_pass(a, image, 1, false);
   sparse = render_pass(b, image, 2, true);
   queue->drm.timeline_value = 1;
   T("multiview first queue submission", submit(queue, a));
   queue->drm.timeline_value = 2;
   struct hk_cmd_buffer *single[] = {b};
   const uint64_t start = queue->sync.submitted[HK_SYNC_VDM];
   T("multiview later queue submission",
     hk_sync_resolve_submit(queue, single, 1, false, false, 0, NULL));
   T("across queue submissions waits on prior layer store",
     waits_on_history(sparse, HK_SYNC_VDM, start));
}

/* The empty-render optimizer executes the API clear on the compute engine. */
static struct hk_cs *
compute_clear_pass(struct hk_cmd_buffer *cmd, const void *image,
                   uint32_t base_layer, bool end)
{
   hk_sync_begin_rendering(cmd, false);
   hk_sync_pass_footprint(cmd, image, 0, base_layer, base_layer + 1, 32, 32,
                         (VkRect2D){{0, 0}, {32, 32}}, false, false);
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
                          (VkRect2D){{0, 0}, {32, 32}}, false, false);
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
     hk_sync_resolve_submit(queue, pair, 2, false, false, 0, NULL));
   T("loaded color across command buffers retains compute clear",
     load->sync.barrier[HK_SYNC_CDM] == clear->sync.index &&
     load->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);

   a = new_cmd(); b = new_cmd();
   compute_clear_pass(a, image, 0, true);
   load = render_pass(b, (void *)0x5678, 1, true);
   pair[0] = a; pair[1] = b;
   queue = new_queue(true);
   T("unrelated color across command buffers submit",
     hk_sync_resolve_submit(queue, pair, 2, false, false, 0, NULL));
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
   const uint64_t start = queue->sync.submitted[HK_SYNC_CDM];
   T("loaded color consumer submission",
     hk_sync_resolve_submit(queue, single, 1, false, false, 0, NULL));
   T("loaded color waits on prior compute history",
     waits_on_history(load, HK_SYNC_CDM, start) &&
     waits_on_nothing(load, HK_SYNC_VDM));
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
     hk_sync_resolve_submit(queue, pair, 2, false, false, 0, NULL));
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
   const uint64_t start = queue->sync.submitted[HK_SYNC_CDM];
   T("compute clear later queue submission",
     hk_sync_resolve_submit(queue, single, 1, false, false, 0, NULL));
   T("prior compute clear hazard names only compute history",
     waits_on_history(sparse, HK_SYNC_CDM, start) &&
     waits_on_nothing(sparse, HK_SYNC_VDM));
   T("completed prior clear submission", hk_sync_resolve_submit(
      queue, single, 1, false, false, 1, NULL));
   T("completed prior clear has no remaining hazard",
     waits_on_nothing(sparse, HK_SYNC_CDM));

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

/* A render in a later ioctl of a split submission, with a live tile hazard
 * against an earlier submission and a fragment-only dependency on a render
 * in an earlier ioctl of its own submission. After rebasing its staged
 * packet, its tiling prefix covers the history without naming more than its
 * whole-render prefix (the kernel rejects a tiling prefix past the whole
 * one). */
static void
test_history_tile_hazard_split(void)
{
   const void *image = (void *)0x5151, *other = (void *)0x5252,
              *third = (void *)0x5353;
   struct hk_queue *q = new_queue(true);
   struct hk_cmd_buffer *a = new_cmd(), *b = new_cmd();
   render_pass(a, image, 1, false);
   q->drm.timeline_value = 1;
   T("split history first submission", submit(q, a));

   struct hk_cs *x = render_pass(b, other, 1, false);
   hk_sync_barrier(b, HK_SYNC_GFX, HK_SYNC_GFX_FRAG);
   struct hk_cs *z = render_pass(b, third, 1, false);
   struct hk_cs *y = render_pass(b, image, 1, true);
   const uint64_t initial[2] = {q->sync.submitted[HK_SYNC_VDM],
                                q->sync.submitted[HK_SYNC_CDM]};
   q->drm.timeline_value = 2;
   struct hk_cmd_buffer *single[] = {b};
   T("split history later submission",
     hk_sync_resolve_submit(q, single, 1, false, false, 0, NULL));
   T("split history whole render waits on X and history",
     y->sync.barrier[HK_SYNC_VDM] == x->sync.index &&
     y->sync.prior[HK_SYNC_VDM] == initial[HK_SYNC_VDM]);
   T("split history tiling waits on history alone",
     y->sync.barrier_ta[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
     y->sync.prior_ta[HK_SYNC_VDM] == initial[HK_SYNC_VDM]);

   /* Y goes in a later ioctl than X and Z. */
   struct drm_asahi_cmd_render_staged staged = {0};
   staged.render.legacy.legacy.flags = DRM_ASAHI_RENDER_STAGE_PREFIXES;
   staged.render.legacy.prior_vdm = y->sync.prior[HK_SYNC_VDM];
   staged.render.legacy.prior_cdm = y->sync.prior[HK_SYNC_CDM];
   staged.stage_prefixes.ta_vdm_barrier = y->sync.barrier_ta[HK_SYNC_VDM];
   staged.stage_prefixes.ta_prior_vdm = y->sync.prior_ta[HK_SYNC_VDM];
   staged.stage_prefixes.ta_cdm_barrier = y->sync.barrier_ta[HK_SYNC_CDM];
   staged.stage_prefixes.ta_prior_cdm = y->sync.prior_ta[HK_SYNC_CDM];
   struct drm_asahi_cmd_header h = {
      .cmd_type = DRM_ASAHI_CMD_RENDER,
      .size = sizeof(staged),
      .vdm_barrier = y->sync.barrier[HK_SYNC_VDM],
      .cdm_barrier = y->sync.barrier[HK_SYNC_CDM],
   };
   const uint32_t vdm_before = z->sync.index;
   T("split history rebases",
     hk_drm_rebase_prior(&h, &staged, vdm_before, 0, initial, true));
   const uint64_t base = initial[HK_SYNC_VDM] + vdm_before;
   const uint64_t whole = hk_drm_render_prefix(
      base, h.vdm_barrier, staged.render.legacy.prior_vdm);
   const uint64_t tiling = hk_drm_render_prefix(
      base, staged.stage_prefixes.ta_vdm_barrier,
      staged.stage_prefixes.ta_prior_vdm);
   T("split history tiling covers history", tiling >= initial[HK_SYNC_VDM]);
   T("split history whole covers X",
     whole >= initial[HK_SYNC_VDM] + x->sync.index);
   T("split history tiling within whole", tiling <= whole);
}

/* A 64x64 pass instance of image over area, loading its attachment or not. */
static struct hk_cs *
load_pass(struct hk_cmd_buffer *cmd, const void *image, bool load,
          VkRect2D area)
{
   hk_sync_begin_rendering(cmd, false);
   hk_sync_pass_footprint(cmd, image, 0, 0, 1, 64, 64, area, false, load);
   hk_sync_pass_footprints_done(cmd);
   struct hk_cs *cs = test_alloc(sizeof(*cs));
   cs->type = HK_CS_VDM;
   hk_sync_stream_created(cmd, cs, &cmd->current_cs.gfx, false);
   list_addtail(&cs->node, &cmd->control_streams);
   hk_sync_end_rendering(cmd, false);
   return cs;
}

/* A pass that only loads whole tiles an earlier render stored waits for that
 * render in its fragment half alone when the kernel orders fragment work
 * after the previous render's (hk_sync_queue::load_hazard_frag), in the
 * command buffer, across command buffers and across submissions. Partly
 * covered tiles and compute producers keep both halves. */
static void
test_load_hazard_frag(void)
{
   const VkRect2D full = {{0, 0}, {64, 64}}, part = {{0, 0}, {40, 40}};
   for (unsigned frag = 0; frag < 2; ++frag) {
#define N(s) (frag ? s " (fragment-only)" : s " (both halves)")
      const void *image = (void *)0x7171;
      struct hk_cmd_buffer *cmd = new_cmd();
      hk_cmd_buffer_device(cmd)->load_hazard_frag = frag;
      struct hk_cs *a = load_pass(cmd, image, false, full);
      struct hk_cs *b = load_pass(cmd, image, true, full);
      T(N("a loading pass waits for the earlier render"),
        b->sync.dep[HK_SYNC_VDM].cut == a);
      T(N("its tiling waits for it unless only whole tiles are loaded"),
        b->sync.dep_ta[HK_SYNC_VDM].cut == (frag ? NULL : a));
      struct hk_cs *c = load_pass(cmd, image, true, part);
      T(N("a partly covered tile keeps both halves"),
        c->sync.dep[HK_SYNC_VDM].cut == b &&
           c->sync.dep_ta[HK_SYNC_VDM].cut == b);

      /* A clear the empty-render optimizer runs on compute. */
      struct hk_cmd_buffer *k = new_cmd();
      hk_cmd_buffer_device(k)->load_hazard_frag = frag;
      struct hk_cs *clear = compute_clear_pass(k, image, 0, true);
      struct hk_cs *after = load_pass(k, image, true, full);
      T(N("a compute producer keeps both halves"),
        after->sync.dep[HK_SYNC_CDM].cut == clear &&
           after->sync.dep_ta[HK_SYNC_CDM].cut == clear);

      for (unsigned exact = 0; exact < 2; ++exact) {
         /* Across command buffers of one submission. */
         struct hk_queue *q = new_queue(exact);
         q->sync.precise_tile_evictions = true;
         q->sync.load_hazard_frag = frag;
         struct hk_cmd_buffer *x = new_cmd(), *y = new_cmd();
         struct hk_cs *xa = load_pass(x, image, false, full);
         struct hk_cs *yb = load_pass(y, image, true, full);
         struct hk_cmd_buffer *both[] = {x, y};
         T(N("two command buffers resolve"),
           hk_sync_resolve_submit(q, both, 2, false, false, 0, NULL));
         T(N("the loading pass's whole render waits for the earlier one"),
           yb->sync.barrier[HK_SYNC_VDM] == xa->sync.index);
         T(N("and its tiling unless only whole tiles are loaded"),
           yb->sync.barrier_ta[HK_SYNC_VDM] ==
              (frag ? DRM_ASAHI_BARRIER_NONE : xa->sync.index));

         /* Across submissions, the earlier one possibly still running. */
         struct hk_queue *h = new_queue(exact);
         h->sync.precise_tile_evictions = true;
         h->sync.load_hazard_frag = frag;
         struct hk_cmd_buffer *first = new_cmd(), *later = new_cmd();
         load_pass(first, image, false, full);
         h->drm.timeline_value = 1;
         T(N("earlier submission"), submit(h, first));
         struct hk_cs *lb = load_pass(later, image, true, full);
         const uint64_t start = h->sync.submitted[HK_SYNC_VDM];
         h->drm.timeline_value = 2;
         T(N("later submission"), submit(h, later));
         if (exact) {
            T(N("whole render waits for the earlier submission"),
              lb->sync.prior[HK_SYNC_VDM] == start);
            T(N("tiling too unless only whole tiles are loaded"),
              lb->sync.prior_ta[HK_SYNC_VDM] == (frag ? 0 : start));
         } else {
            T(N("whole render waits for earlier ioctls"),
              lb->sync.barrier[HK_SYNC_VDM] == 0);
            T(N("tiling too unless only whole tiles are loaded"),
              lb->sync.barrier_ta[HK_SYNC_VDM] ==
                 (frag ? DRM_ASAHI_BARRIER_NONE : 0));
         }
      }
#undef N
   }
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

   /* An optimal ALIAS_BIT attachment whose memory has no buffer binding is
    * exempted from HK_SYNC_GFX_MEM while buffer_alias_mem_gen is unchanged,
    * allowing vertex/fragment overlap; if a buffer is later bound to an
    * image-bearing memory (advancing buffer_alias_mem_gen), submit-time
    * resolution upgrades it conservatively and widens intra-CB vertex waits. */
   for (unsigned stale = 0; stale < 2; ++stale) {
      struct hk_cmd_buffer *ac = timestamp_cmd(AGX_CHIP_G17P, true);
      struct hk_device *adev = hk_cmd_buffer_device(ac);
      ac->sync.has_alias_image_render = true;
      ac->sync.buffer_alias_mem_gen = adev->buffer_alias_mem_gen;
      ac->state.gfx.render.m5_attachments_alias_image = true;
      struct hk_cs *aliased = add(ac, false);
      ac->state.gfx.render.m5_attachments_alias_image = false;
      hk_sync_barrier(ac, HK_SYNC_ALL, HK_SYNC_ALL);
      struct hk_cs *after_alias = add(ac, false);
      T("alias-image render skips HK_SYNC_GFX_MEM at record time",
        !(aliased->sync.kind & HK_SYNC_GFX_MEM) &&
        aliased->sync.alias_image_render &&
        after_alias->sync.dep_ta[HK_SYNC_VDM].cut == NULL);
      if (stale)
         adev->buffer_alias_mem_gen++;
      struct hk_queue *aq = new_queue(true);
      T("alias-image render submit", submit(aq, ac));
      T("alias-image render submit-time gen check",
        ((aliased->sync.kind & HK_SYNC_GFX_MEM) != 0) == (stale != 0) &&
        after_alias->sync.barrier[HK_SYNC_VDM] == 1 &&
        after_alias->sync.barrier_ta[HK_SYNC_VDM] ==
           (stale ? 1 : DRM_ASAHI_BARRIER_NONE) &&
        after_alias->sync.ta_split == !stale);
   }

   struct hk_cmd_buffer *cmd = timestamp_cmd(AGX_CHIP_G15S, true);
   struct hk_cs *plain = add(cmd, false);
   hk_sync_barrier(cmd, HK_SYNC_ALL, HK_SYNC_ALL);
   struct hk_cs *next = add(cmd, false);
   T("other chips: vertex half keeps the whole dependency",
     next->sync.dep_ta[HK_SYNC_VDM].cut == plain);
}

/* G17P tiling order: a render whose whole prefix holds the latest earlier
 * render may leave a vertex requirement the earlier tiling kicks already
 * covered to the fragment half. Writer W (ordinal 1), plain P (2), reader R
 * (3): P's vertex half needs W, which no tiling kick covered yet; R's needs W
 * too, which P's tiling kick (submitted with its whole prefix) covers. */
static void
test_g17p_tiling_order(void)
{
   T("render prefix of the header barrier",
     hk_drm_render_prefix(10, 2, 0) == 12 &&
     hk_drm_render_prefix(10, 0, 0) == 10 &&
     hk_drm_render_prefix(10, DRM_ASAHI_BARRIER_NONE, 7) == 7 &&
     hk_drm_render_prefix(10, 1, 12) == 12 &&
     hk_drm_render_prefix(10, DRM_ASAHI_BARRIER_NONE, 0) == 0);

   uint64_t cover = 0;
   /* W: no render dependency. */
   T("no render prefix is never fragment-ordered",
     !hk_drm_g17p_tiling_ordered(0, 0, 0, 0, false, &cover) && cover == 0);
   /* P: whole 1, vertex 1. */
   T("vertex half naming an uncovered render keeps the prefix",
     !hk_drm_g17p_tiling_ordered(0, 1, 1, 1, false, &cover));
   T("a render without the flag raises the cover", cover == 1);
   /* R: whole 2, vertex 1. */
   uint64_t c = cover;
   T("vertex half covered by earlier tiling",
     hk_drm_g17p_tiling_ordered(0, 2, 2, 1, false, &c) && c == 1);
   c = cover;
   T("vertex half naming the previous render keeps the prefix",
     !hk_drm_g17p_tiling_ordered(0, 2, 2, 2, false, &c) && c == 2);
   c = cover;
   T("whole prefix without the latest earlier render keeps the prefix",
     !hk_drm_g17p_tiling_ordered(0, 2, 1, 0, false, &c) && c == 1);
   c = cover;
   T("an already flagged render leaves the cover",
     hk_drm_g17p_tiling_ordered(0, 2, 2, 2, true, &c) && c == 1);
   c = 39;
   T("ordinals across submissions",
     hk_drm_g17p_tiling_ordered(40, 0, 40, 39, false, &c) && c == 39 &&
     !hk_drm_g17p_tiling_ordered(40, 0, 40, 40, false, &c) && c == 40);
}

/* Sibling kernel queues: once siblings have run compute work, the primary
 * kernel queue's compute ordinals run ahead of the queue's own (base 20,
 * kernel queue 50). Both prefixes of a split render, the whole one and the
 * tiling half's, are rewritten the same way. */
static void
test_sibling_prefix(void)
{
   const uint64_t base[2] = {10, 20};
   const uint64_t kq_base[2] = {10, 50};
   const uint16_t none[2] = {DRM_ASAHI_BARRIER_NONE, DRM_ASAHI_BARRIER_NONE};

   bool earlier = false;
   uint64_t whole[2] = {12, 23};
   hk_drm_sibling_prefix(whole, none, base, kq_base, &earlier);
   T("a command of this submission keeps its offset",
     whole[0] == 12 && whole[1] == 53 && !earlier);

   uint64_t tiling[2] = {0, 22};
   hk_drm_sibling_prefix(tiling, none, base, kq_base, &earlier);
   T("the tiling half's prior moves with the whole prior",
     tiling[0] == 0 && tiling[1] == 52 && !earlier);

   uint64_t older[2] = {0, 15};
   hk_drm_sibling_prefix(older, none, base, kq_base, &earlier);
   T("an earlier submission's command names the kernel queue's earlier work",
     older[0] == 0 && older[1] == 50 && earlier);

   earlier = false;
   const uint16_t zero_cdm[2] = {DRM_ASAHI_BARRIER_NONE, 0};
   uint64_t header0[2] = {0, 0};
   hk_drm_sibling_prefix(header0, zero_cdm, base, kq_base, &earlier);
   T("a zero tiling barrier waits for the siblings' work",
     header0[0] == 0 && header0[1] == 0 && earlier);
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

   /* One image barrier's scopes, as a driver-performed layout transition
    * takes them (brl_sync_transition_kinds()). */
   const VkImageMemoryBarrier2 compute = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
   };
   src = dst = 0;
   brl_sync_image_barrier_kinds(&image, true, &src, &dst);
   T("one image barrier names its own classes",
     src == HK_SYNC_GFX && dst == (HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG));
   src = dst = 0;
   brl_sync_image_barrier_kinds(&compute, true, &src, &dst);
   T("a compute image barrier from the top names compute only",
     src == 0 && dst == HK_SYNC_COMPUTE);
   brl_sync_image_barrier_kinds(&image, true, &src, &dst);
   T("image barrier scopes accumulate",
     src == HK_SYNC_GFX &&
        dst == (HK_SYNC_COMPUTE | HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG));

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
     hk_sync_resolve_submit(queue, pair, 2, false, false, 0, NULL));
   T("carried image class leaves a plain vertex half unordered",
     plain->sync.barrier_ta[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
     plain->sync.prior_ta[HK_SYNC_VDM] == 0);
   T("carried image class orders an image-reading vertex half",
     reader->sync.barrier_ta[HK_SYNC_VDM] != DRM_ASAHI_BARRIER_NONE ||
     reader->sync.prior_ta[HK_SYNC_VDM] != 0);

   /* Pre-graphics work without software pre-rasterization stages skips
    * HK_SYNC_GFX_IMG unless the render pass spills colour attachments and
    * therefore decompresses them in pre_gfx. */
   cmd = new_cmd();
   producer = add(cmd, false);
   hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_GFX_IMG);
   hk_sync_begin_rendering(cmd, false);
   add_slot(cmd, &cmd->current_cs.gfx, false);
   struct hk_cs *pre_plain = add_slot(cmd, &cmd->current_cs.pre_gfx, true);
   T("non-spilling pre-graphics without pre-raster stages skips image class",
     !pre_plain->sync.dep[HK_SYNC_VDM].cut);
   hk_sync_end_rendering(cmd, false);

   cmd = new_cmd();
   producer = add(cmd, false);
   hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_GFX_IMG);
   hk_sync_begin_rendering(cmd, false);
   cmd->state.gfx.render.tilebuffer.spilled[0] = true;
   add_slot(cmd, &cmd->current_cs.gfx, false);
   struct hk_cs *pre_spill = add_slot(cmd, &cmd->current_cs.pre_gfx, true);
   T("spilling pre-graphics honours image class",
     pre_spill->sync.dep[HK_SYNC_VDM].cut == producer);
   hk_sync_end_rendering(cmd, false);
}

static void
test_vertex_image_bloom(void)
{
   /* Find three distinct keys with pairwise-disjoint 64-bit bloom bits. */
   const uint64_t k_a = UINT64_C(0x100000);
   uint64_t k_b = UINT64_C(0x200000);
   while (hk_sync_image_bloom(k_b) == hk_sync_image_bloom(k_a))
      k_b += 0x10000;
   uint64_t k_c = k_b + 0x10000;
   while (hk_sync_image_bloom(k_c) == hk_sync_image_bloom(k_a) ||
          hk_sync_image_bloom(k_c) == hk_sync_image_bloom(k_b))
      k_c += 0x10000;
   const uint64_t b_a = hk_sync_image_bloom(k_a);
   const uint64_t b_b = hk_sync_image_bloom(k_b);
   const uint64_t b_c = hk_sync_image_bloom(k_c);

   /* In-CB: disjoint attachment vs VS bloom splits TA while fragment waits. */
   struct hk_cmd_buffer *cmd = new_cmd();
   hk_sync_begin_rendering(cmd, false);
   cmd->state.gfx.render.att_bloom = b_a;
   struct hk_cs *p1 = add(cmd, false);
   hk_sync_end_rendering(cmd, false);

   hk_sync_begin_rendering(cmd, false);
   cmd->state.gfx.render.att_bloom = b_b;
   struct hk_cs *p2 = add(cmd, false);
   hk_sync_end_rendering(cmd, false);

   hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG);

   /* Disjoint reader (reads C): waits on neither P1 nor P2 in vertex half. */
   struct hk_cs *r_disjoint = add(cmd, false);
   hk_sync_vertex_images_bloom(cmd, r_disjoint, b_c);
   T("in-CB disjoint bloom splits vertex half",
     !r_disjoint->sync.dep_ta[HK_SYNC_VDM].cut &&
     r_disjoint->sync.dep[HK_SYNC_VDM].cut == p2);

   /* Reader of A: waits on P1, not P2. */
   struct hk_cs *r_match_a = add(cmd, false);
   hk_sync_vertex_images_bloom(cmd, r_match_a, b_a);
   T("in-CB bloom waits on matching earlier render only",
     r_match_a->sync.dep_ta[HK_SYNC_VDM].cut == p1 &&
     r_match_a->sync.dep[HK_SYNC_VDM].cut == p2);

   /* Adding a second draw reading B upgrades cut to P2. */
   hk_sync_vertex_images_bloom(cmd, r_match_a, b_b);
   T("in-CB second draw widens bloom cut to P2",
     r_match_a->sync.dep_ta[HK_SYNC_VDM].cut == p2);

   /* Upgrading to unsafe forces full conservative HK_SYNC_GFX_IMG wait. */
   hk_sync_vertex_images(cmd, r_disjoint);
   T("in-CB unsafe upgrade forces full wait",
     r_disjoint->sync.vtx_img_unsafe &&
     r_disjoint->sync.dep_ta[HK_SYNC_VDM].cut == p2);

   /* In-CB HK_SYNC_GFX_MEM producer always orders even disjoint bloom. */
   cmd = new_cmd();
   hk_sync_begin_rendering(cmd, false);
   cmd->state.gfx.render.att_bloom = b_a;
   struct hk_cs *pmem = add(cmd, false);
   hk_sync_note_memory_write(cmd, pmem);
   hk_sync_end_rendering(cmd, false);
   hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG);
   struct hk_cs *r_after_mem = add(cmd, false);
   hk_sync_vertex_images_bloom(cmd, r_after_mem, b_c);
   T("in-CB memory-writing render orders disjoint bloom reader",
     r_after_mem->sync.dep_ta[HK_SYNC_VDM].cut == pmem);

   /* Cross-submit with exact_prior: disjoint splits, matching hits exact ordinal. */
   struct hk_queue *q = new_queue(true);
   struct hk_cmd_buffer *s1 = new_cmd();
   hk_sync_begin_rendering(s1, false);
   s1->state.gfx.render.att_bloom = b_a;
   add(s1, false); /* ordinal 1: att A */
   hk_sync_end_rendering(s1, false);
   hk_sync_begin_rendering(s1, false);
   s1->state.gfx.render.att_bloom = b_b;
   add(s1, false); /* ordinal 2: att B */
   hk_sync_end_rendering(s1, false);
   hk_sync_barrier(s1, HK_SYNC_GFX, HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG);
   T("bloom cross-submit producer", submit(q, s1));

   struct hk_cmd_buffer *s2 = new_cmd();
   struct hk_cs *c_disjoint = add(s2, false);
   hk_sync_vertex_images_bloom(s2, c_disjoint, b_c);
   struct hk_cs *c_match_a = add(s2, false);
   hk_sync_vertex_images_bloom(s2, c_match_a, b_a);
   struct hk_cs *c_zero = add(s2, false);
   hk_sync_vertex_images_bloom(s2, c_zero, 0);
   struct hk_cs *c_unsafe = add(s2, false);
   hk_sync_vertex_images(s2, c_unsafe);
   T("bloom cross-submit consumers", submit(q, s2));

   T("cross-submit disjoint bloom skips prior renders on VDM TA",
     c_disjoint->sync.barrier_ta[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
     c_disjoint->sync.prior_ta[HK_SYNC_VDM] == 0 &&
     c_disjoint->sync.prior[HK_SYNC_VDM] == 2 &&
     c_disjoint->sync.ta_split);
   T("cross-submit matching bloom waits on exact matching ordinal 1, not 2",
     c_match_a->sync.barrier_ta[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
     c_match_a->sync.prior_ta[HK_SYNC_VDM] == 1 &&
     c_match_a->sync.prior[HK_SYNC_VDM] == 2 &&
     c_match_a->sync.ta_split);
   T("cross-submit empty bloom (null descriptors) skips prior renders",
     c_zero->sync.barrier_ta[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
     c_zero->sync.prior_ta[HK_SYNC_VDM] == 0 &&
     c_zero->sync.ta_split);
   T("cross-submit unsafe reader waits on full prior ordinal 2",
     c_unsafe->sync.prior_ta[HK_SYNC_VDM] == 2 &&
     !c_unsafe->sync.ta_split);

   /* Cross-submit ring eviction stays conservative at evicted ordinal. */
   struct hk_queue *qe = new_queue(true);
   struct hk_cmd_buffer *se = new_cmd();
   for (unsigned i = 0; i < HK_SYNC_RECENT_VDM_ATT + 4; ++i) {
      hk_sync_begin_rendering(se, false);
      se->state.gfx.render.att_bloom = b_a;
      add(se, false);
      hk_sync_end_rendering(se, false);
   }
   hk_sync_barrier(se, HK_SYNC_GFX, HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG);
   T("bloom eviction producer submit", submit(qe, se));
   struct hk_cmd_buffer *sc = new_cmd();
   struct hk_cs *c_ev = add(sc, false);
   hk_sync_vertex_images_bloom(sc, c_ev, b_c);
   T("bloom eviction consumer submit", submit(qe, sc));
   T("cross-submit ring eviction waits conservatively on evicted ordinal",
     c_ev->sync.prior_ta[HK_SYNC_VDM] == 4 &&
     c_ev->sync.prior[HK_SYNC_VDM] == HK_SYNC_RECENT_VDM_ATT + 4 &&
     c_ev->sync.ta_split);

   /* Legacy queue (exact_prior == false) falls back conservatively. */
   struct hk_queue *ql = new_queue(false);
   struct hk_cmd_buffer *la = new_cmd(), *lb = new_cmd();
   hk_sync_begin_rendering(la, false);
   la->state.gfx.render.att_bloom = b_a;
   add(la, false);
   hk_sync_end_rendering(la, false);
   hk_sync_barrier(la, HK_SYNC_GFX, HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG);
   struct hk_cs *lr = add(lb, false);
   hk_sync_vertex_images_bloom(lb, lr, b_c);
   struct hk_cmd_buffer *lpair[] = {la, lb};
   T("legacy bloom submit",
     hk_sync_resolve_submit(ql, lpair, 2, false, false, 0, NULL));
   T("legacy queue keeps conservative vertex wait",
     lr->sync.barrier_ta[HK_SYNC_VDM] == lr->sync.barrier[HK_SYNC_VDM] &&
     lr->sync.barrier_ta[HK_SYNC_VDM] != DRM_ASAHI_BARRIER_NONE);
}

/* A layout transition the driver performs as work (the metadata clear of a
 * discarded compressed storage image): brl_sync_transition_*(), recorded the
 * way brl_discard_init_prepare()/_record() and hk_CmdWaitEvents2() record it.
 */
static struct hk_cs *
transition(struct hk_cmd_buffer *cmd, const VkDependencyInfo *info,
           const VkImageMemoryBarrier2 *bar, bool acquire, bool render_pass)
{
   uint8_t src = 0, dst = 0;
   brl_sync_transition_kinds(info, bar, true, acquire, render_pass, &src,
                             &dst);
   brl_sync_transition_prepare(cmd, src);

   uint8_t all_src, all_dst;
   hk_sync_dependency_kinds(info, true, &all_src, &all_dst);
   hk_sync_barrier(cmd, all_src, all_dst);

   brl_sync_transition_begin(cmd);
   struct hk_cs *clear = add(cmd, true);
   clear->stats.cmds = 1;
   brl_sync_transition_end(cmd, dst);
   return clear;
}

static void
test_discard_transition(void)
{
   const VkImageMemoryBarrier2 to_input = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
      .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
   };
   const VkMemoryBarrier2 external = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
   };
   const VkDependencyInfo subpass = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &external,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &to_input,
   };

   /* An automatic render-pass transition (the common runtime's
    * runtime_rp_barrier): the image barrier names no source stage, the
    * VK_SUBPASS_EXTERNAL dependency comes as a memory barrier. */
   uint8_t src = 0, dst = 0;
   brl_sync_transition_kinds(&subpass, &to_input, true, false, false, &src,
                             &dst);
   T("automatic transition: the image barrier alone names no source",
     src == 0 && dst == HK_SYNC_GFX_FRAG);
   src = dst = 0;
   brl_sync_transition_kinds(&subpass, &to_input, true, false, true, &src,
                             &dst);
   T("automatic transition takes the subpass dependency's first scope",
     src == HK_SYNC_GFX && dst == HK_SYNC_GFX_FRAG);

   struct hk_cmd_buffer *cmd = timestamp_cmd(AGX_CHIP_G17S, true);
   struct hk_cs *render = add(cmd, false);
   struct hk_cs *clear = transition(cmd, &subpass, &to_input, false, true);
   T("automatic transition: the clear waits on the earlier render",
     clear->sync.dep[HK_SYNC_VDM].cut == render);
   T("the clear is transfer-class work", clear->sync.kind & HK_SYNC_XFER);
   struct hk_cs *next = add(cmd, false);
   T("the subpass's render waits on the clear",
     next->sync.dep[HK_SYNC_CDM].cut == clear);
   T("automatic transition submits", submit(new_queue(true), cmd));

   /* A discard barrier next to an unrelated one in the same dependency. */
   const VkImageMemoryBarrier2 to_compute = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   const VkMemoryBarrier2 frag_to_copy = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
   };
   const VkDependencyInfo mixed = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &frag_to_copy,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &to_compute,
   };
   cmd = timestamp_cmd(AGX_CHIP_G17S, true);
   render = add(cmd, false);
   clear = transition(cmd, &mixed, &to_compute, false, false);
   T("mixed dependency: the clear inherits no unrelated cut",
     !clear->sync.dep[HK_SYNC_VDM].cut && !clear->sync.dep[HK_SYNC_CDM].cut &&
        !(clear->sync.dep[HK_SYNC_VDM].prior & HK_SYNC_GFX) &&
        !(clear->sync.dep[HK_SYNC_CDM].prior & HK_SYNC_GFX));
   hk_sync_xfer_begin(cmd);
   struct hk_cs *copy = add(cmd, true);
   hk_sync_xfer_end(cmd);
   T("mixed dependency: a later copy keeps the application's dependency",
     copy->sync.dep[HK_SYNC_VDM].cut == render);
   T("mixed dependency: a later copy does not wait on the clear",
     copy->sync.dep[HK_SYNC_CDM].cut != clear);
   struct hk_cs *dispatch = add(cmd, true);
   T("mixed dependency: the second scope waits on the clear",
     dispatch->sync.dep[HK_SYNC_CDM].cut == clear);
   T("mixed dependency submits", submit(new_queue(true), cmd));

   /* A fragment-to-compute discard leaves no fragment requirement on later
    * transfer work. */
   const VkImageMemoryBarrier2 frag_to_compute = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   const VkDependencyInfo frag_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &frag_to_compute,
   };
   cmd = timestamp_cmd(AGX_CHIP_G17S, true);
   render = add(cmd, false);
   clear = transition(cmd, &frag_dep, &frag_to_compute, false, false);
   T("fragment-to-compute discard: the clear waits on the render",
     clear->sync.dep[HK_SYNC_VDM].cut == render);
   hk_sync_xfer_begin(cmd);
   copy = add(cmd, true);
   hk_sync_xfer_end(cmd);
   T("fragment-to-compute discard: no lasting fragment-to-transfer wait",
     !copy->sync.dep[HK_SYNC_VDM].cut && !copy->sync.dep[HK_SYNC_CDM].cut);

   /* An empty second scope. */
   const VkImageMemoryBarrier2 to_nothing = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   const VkDependencyInfo empty_dst = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &to_nothing,
   };
   const VkDependencyInfo compute_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &to_compute,
   };
   struct hk_cmd_buffer *a = timestamp_cmd(AGX_CHIP_G17S, true);
   struct hk_cs *producer = add(a, true);
   struct hk_cs *first = transition(a, &empty_dst, &to_nothing, false, false);
   T("empty second scope: the clear waits on its first scope",
     first->sync.dep[HK_SYNC_CDM].cut == producer);
   render = add(a, false);
   dispatch = add(a, true);
   T("empty second scope: later work does not wait on the clear",
     render->sync.dep[HK_SYNC_CDM].cut != first &&
        dispatch->sync.dep[HK_SYNC_CDM].cut != first);
   struct hk_cs *second =
      transition(a, &compute_dep, &to_compute, false, false);
   T("layout transitions stay in submission order",
     second->sync.dep[HK_SYNC_CDM].cut == first);
   T("a transition follows earlier command buffers' transitions",
     (second->sync.dep[HK_SYNC_CDM].prior & HK_SYNC_XFER) &&
        (second->sync.dep[HK_SYNC_VDM].prior & HK_SYNC_XFER));
   struct hk_cmd_buffer *b = timestamp_cmd(AGX_CHIP_G17S, true);
   struct hk_cs *later = transition(b, &compute_dep, &to_compute, false, false);
   struct hk_cmd_buffer *pair[] = {a, b};
   T("transitions across command buffers submit",
     hk_sync_resolve_submit(new_queue(true), pair, 2, false, false, 0, NULL));
   T("a later command buffer's transition waits on the earlier ones",
     later->sync.barrier[HK_SYNC_CDM] != DRM_ASAHI_BARRIER_NONE ||
        later->sync.prior[HK_SYNC_CDM] != 0);

   /* Queue family acquire: the source stages are ignored by default. */
   const VkImageMemoryBarrier2 acquire = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = 1,
      .dstQueueFamilyIndex = 0,
   };
   VkDependencyInfo acquire_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &acquire,
   };
   src = dst = 0;
   brl_sync_transition_kinds(&acquire_dep, &acquire, true, true, false, &src,
                             &dst);
   T("an acquire ignores its source stages",
     src == 0 && dst == HK_SYNC_COMPUTE);
   acquire_dep.dependencyFlags =
      VK_DEPENDENCY_QUEUE_FAMILY_OWNERSHIP_TRANSFER_USE_ALL_STAGES_BIT_KHR;
   src = dst = 0;
   brl_sync_transition_kinds(&acquire_dep, &acquire, true, true, false, &src,
                             &dst);
   T("an acquire with USE_ALL_STAGES keeps them",
     src == HK_SYNC_COMPUTE && dst == HK_SYNC_COMPUTE);

   /* Event waits. G17P polls the event: the transition follows the poll,
    * also when its barriers come in a later dependency (the legacy
    * vkCmdWaitEvents route) whose first scope names only the host, which
    * set the event before submitting the wait. */
   const VkImageMemoryBarrier2 from_host = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   const VkDependencyInfo host_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &from_host,
   };
   cmd = timestamp_cmd(AGX_CHIP_G17P, true);
   cmd->sync.kind_override = HK_SYNC_ALL;
   struct hk_cs *poll = add(cmd, true);
   cmd->sync.kind_override = 0;
   poll->stats.cmds = 1;
   hk_sync_after_stream(cmd, poll, 0);
   brl_sync_event_poll(cmd, poll);
   clear = transition(cmd, &host_dep, &from_host, false, false);
   T("event wait: the clear follows the poll",
     clear->sync.dep[HK_SYNC_CDM].cut == poll);
   brl_sync_event_done(cmd);
   T("the event wait ends", !cmd->sync.brl_event_poll);
   dispatch = add(cmd, true);
   T("event wait: the second scope follows the clear",
     dispatch->sync.dep[HK_SYNC_CDM].cut == clear);

   /* Without a poll, a wait on an event the host set before submitting the
    * command buffer (VUID-vkSetEvent-event-09543) needs no GPU wait for the
    * host: unrelated earlier fragment work does not hold up the clear or the
    * compute work that consumes the image. */
   cmd = timestamp_cmd(AGX_CHIP_G17S, true);
   render = add(cmd, false);
   render->stats.cmds = 1;
   clear = transition(cmd, &host_dep, &from_host, false, false);
   dispatch = add(cmd, true);
   T("host event set before submission: the clear waits on no render",
     !clear->sync.dep[HK_SYNC_VDM].cut && !clear->sync.dep[HK_SYNC_CDM].cut &&
        !(clear->sync.dep[HK_SYNC_VDM].prior & ~HK_SYNC_XFER) &&
        !(clear->sync.dep[HK_SYNC_CDM].prior & ~HK_SYNC_XFER));
   T("host event set before submission: the consumer waits on the clear only",
     dispatch->sync.dep[HK_SYNC_CDM].cut == clear &&
        !dispatch->sync.dep[HK_SYNC_VDM].cut);
   T("host event set before submission submits",
     submit(new_queue(true), cmd) &&
        clear->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
        dispatch->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);

   /* A device-set event, and a legacy wait on host and device events (the
    * common lowering's pipeline barrier names both source stages), keep the
    * device stages of the wait's first scope. */
   const VkImageMemoryBarrier2 from_compute = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   const VkDependencyInfo device_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &from_compute,
   };
   cmd = timestamp_cmd(AGX_CHIP_G17S, true);
   render = add(cmd, false);
   producer = add(cmd, true);
   clear = transition(cmd, &device_dep, &from_compute, false, false);
   T("device-set event: the clear follows the wait's first scope",
     clear->sync.dep[HK_SYNC_CDM].cut == producer &&
        !clear->sync.dep[HK_SYNC_VDM].cut);

   const VkImageMemoryBarrier2 from_host_compute = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask =
         VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   const VkDependencyInfo legacy_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &from_host_compute,
   };
   cmd = timestamp_cmd(AGX_CHIP_G17S, true);
   render = add(cmd, false);
   producer = add(cmd, true);
   clear = transition(cmd, &legacy_dep, &from_host_compute, false, false);
   T("host and device events: the clear keeps the device stages",
     clear->sync.dep[HK_SYNC_CDM].cut == producer &&
        !clear->sync.dep[HK_SYNC_VDM].cut);
}

/* A transition in a later command buffer than an unrelated
 * FRAGMENT_SHADER -> COPY dependency: it follows earlier transfer work, but
 * what that dependency asks of later transfer work stays with that work. */
static void
test_discard_transition_carry(void)
{
   const VkImageMemoryBarrier2 to_compute = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   const VkDependencyInfo discard = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &to_compute,
   };
   const VkImageMemoryBarrier2 compute_to_compute = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   const VkDependencyInfo chained = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &compute_to_compute,
   };
   const VkMemoryBarrier2 frag_to_copy = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
   };
   const VkMemoryBarrier2 frag_to_compute = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
   };
   const VkMemoryBarrier2 *const tails[] = {&frag_to_copy, &frag_to_compute};

   for (unsigned exact = 0; exact < 2; ++exact) {
      for (unsigned split = 0; split < 2; ++split) {
         /* Legacy priors carry requirements only within one submission. */
         if (split && !exact)
            continue;
         for (unsigned t = 0; t < 2; ++t) {
            /* A: earlier transfer work, an unrelated render, then a
             * dependency from fragment work to copies (t = 0), or to compute
             * work, which a discard from compute chains with (t = 1). */
            struct hk_cmd_buffer *a = timestamp_cmd(AGX_CHIP_G17S, exact);
            hk_sync_xfer_begin(a);
            struct hk_cs *copy = add(a, true);
            copy->stats.cmds = 1;
            hk_sync_xfer_end(a);
            struct hk_cs *render = add(a, false);
            render->stats.cmds = 1;
            const VkDependencyInfo tail = {
               .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
               .memoryBarrierCount = 1,
               .pMemoryBarriers = tails[t],
            };
            uint8_t src, dst;
            hk_sync_dependency_kinds(&tail, true, &src, &dst);
            hk_sync_barrier(a, src, dst);

            /* B: the discard, its consumer, and a later copy. */
            struct hk_cmd_buffer *b = timestamp_cmd(AGX_CHIP_G17S, exact);
            struct hk_cs *clear =
               t ? transition(b, &chained, &compute_to_compute, false, false)
                 : transition(b, &discard, &to_compute, false, false);
            struct hk_cs *dispatch = add(b, true);
            dispatch->stats.cmds = 1;
            hk_sync_xfer_begin(b);
            struct hk_cs *later = add(b, true);
            later->stats.cmds = 1;
            hk_sync_xfer_end(b);

            struct hk_queue *q = new_queue(exact);
            bool ok;
            if (split) {
               ok = submit(q, a) && submit(q, b);
            } else {
               struct hk_cmd_buffer *pair[] = {a, b};
               ok = hk_sync_resolve_submit(q, pair, 2, false, false, 0, NULL);
            }
            T("transition after a carried dependency submits", ok);

            /* The render is VDM command 1 and the copy CDM command 1, of
             * this ioctl or (split, exact) as an absolute position. */
            const bool waits_render =
               split ? clear->sync.prior[HK_SYNC_VDM] == 1
                     : clear->sync.barrier[HK_SYNC_VDM] == 1;
            const bool no_render =
               clear->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
               clear->sync.prior[HK_SYNC_VDM] == 0;
            if (t) {
               T("chained discard waits on the earlier render", waits_render);
            } else {
               T("discard does not take the carried transfer requirement",
                 no_render);
               T("its consumer does not wait on the render",
                 dispatch->sync.barrier[HK_SYNC_VDM] ==
                       DRM_ASAHI_BARRIER_NONE &&
                    dispatch->sync.prior[HK_SYNC_VDM] == 0);
            }
            T("discard follows earlier transfer work",
              split ? clear->sync.prior[HK_SYNC_CDM] == 1
                    : clear->sync.barrier[HK_SYNC_CDM] == copy->sync.index);
            T("a later copy keeps the carried requirement",
              t || (split ? later->sync.prior[HK_SYNC_VDM] == 1
                          : later->sync.barrier[HK_SYNC_VDM] ==
                               render->sync.index));
         }
      }
   }

   /* A transition merged into a compute stream: the merged stream takes the
    * carried requirements of compute work and of the transition's first
    * scope, still not those of transfer work. */
   struct hk_cmd_buffer *a = timestamp_cmd(AGX_CHIP_G17S, true);
   struct hk_cs *render = add(a, false);
   render->stats.cmds = 1;
   hk_sync_barrier(a, HK_SYNC_GFX, HK_SYNC_XFER);
   struct hk_cmd_buffer *b = timestamp_cmd(AGX_CHIP_G17S, true);
   struct hk_cs *dispatch = add(b, true);
   dispatch->stats.cmds = 1;
   struct hk_cs *clear = transition(b, &discard, &to_compute, false, false);
   hk_sync_merge(b, dispatch, clear);
   list_del(&clear->node);
   T("merged transition keeps an explicit carry",
     dispatch->sync.brl_transition &&
        (dispatch->sync.brl_carry & HK_SYNC_COMPUTE) &&
        !(dispatch->sync.brl_carry & HK_SYNC_XFER));
   struct hk_cmd_buffer *pair[] = {a, b};
   T("merged transition submits",
     hk_sync_resolve_submit(new_queue(true), pair, 2, false, false, 0, NULL));
   T("merged transition does not take the carried transfer requirement",
     dispatch->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
        dispatch->sync.prior[HK_SYNC_VDM] == 0);

   /* Merging a pre-graphics CDM stream that carries a geometry heap slot into
    * a preceding compute stream preserves the heap slot and slot pointer. */
   struct hk_cmd_buffer *hc = new_cmd();
   struct hk_cs *api_cs = add(hc, true);
   struct hk_cs *pre_cs = add(hc, true);
   pre_cs->sync.heap_slot = 3;
   hc->sync.last_heap_slot[2][HK_SYNC_CDM] = pre_cs;
   hk_sync_merge(hc, api_cs, pre_cs);
   list_del(&pre_cs->node);
   T("merged pre-graphics stream preserves heap_slot and last_heap_slot",
     api_cs->sync.heap_slot == 3 &&
        hc->sync.last_heap_slot[2][HK_SYNC_CDM] == api_cs);
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
        hk_sync_resolve_submit(q, ecr, 3, false, true, 0, NULL));
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
        hk_sync_resolve_submit(q, erc, 3, false, true, 0, NULL));
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
        hk_sync_resolve_submit(q, cr, 2, false, true, 0, NULL));
      T("chain [compute, render] render follows compute",
        compute->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE &&
           render->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
           render->sync.barrier[HK_SYNC_CDM] == 1);

      /* No command at all: nothing to chain. */
      q = new_queue(exact);
      struct hk_cmd_buffer *ee[] = {new_cmd(), new_cmd()};
      T("chain without commands",
        hk_sync_resolve_submit(q, ee, 2, false, true, 0, NULL) &&
           q->sync.submitted[HK_SYNC_VDM] == 0 &&
           q->sync.submitted[HK_SYNC_CDM] == 0);
   }
}

/* One simultaneous-use recording twice in a submission, {a, b, a}: each
 * occurrence resolves at its own position, so every stream waits on the one
 * just before it and never on itself or a later command. */
static void
test_repeated_recording(void)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      struct hk_queue *q = new_queue(mode & 1);
      struct hk_cmd_buffer *a = new_cmd(), *b = new_cmd();
      for (unsigned i = 0; i < 4; ++i) {
         hk_sync_barrier(i < 2 ? a : b, HK_SYNC_ALL, HK_SYNC_ALL);
         add(i < 2 ? a : b, true);
      }
      struct hk_cmd_buffer *cmds[] = {a, b, a};
      struct hk_sync_resolved out[6];
      T("repeated recording submission",
        hk_sync_resolve_submit(q, cmds, 3, false, mode & 2, 0, out));
      bool ordered = true;
      for (unsigned i = 0; i < 6; ++i) {
         ordered &= out[i].barrier[HK_SYNC_CDM] ==
                       (i ? i : DRM_ASAHI_BARRIER_NONE) &&
                    out[i].barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
                    !out[i].prior[HK_SYNC_CDM] && !out[i].prior[HK_SYNC_VDM];
      }
      T("repeated recording waits on the previous occurrence", ordered);
      T("repeated recording ordinals",
        !(mode & 1) || q->sync.submitted[HK_SYNC_CDM] == 6);
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
                 q, cmds, groups, false, false, 0, NULL));
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
           q, cmds, 5, false, false, 0, NULL));
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
                  same &= hk_sync_resolve_submit(a, cmds, count, flush, chain, completed, NULL);
                  for (unsigned j = 0; j < n; ++j)
                     saved[j] = streams[j]->sync;
                  same &= hk_sync_resolve_submit(b, cmds, count, flush, chain, completed, NULL);
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

/* A dependency into DRAW_INDIRECT names every logically later graphics stage
 * too (Vulkan 1.4, 7.1.2), so a render of direct draws waits for the compute
 * work in its vertex half as in its fragment half. The same dependency
 * recorded by an acceleration-structure build between its own dispatches
 * orders only the build's compute work; the application's dependency on the
 * build still orders later renders. */
static void
test_draw_indirect_scope(void)
{
   const uint8_t src = hk_sync_kinds_src(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
   const uint8_t dst = hk_sync_kinds_dst(VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
   T("DRAW_INDIRECT names graphics and compute work",
     (dst & HK_SYNC_GFX) && (dst & HK_SYNC_COMPUTE));

   for (unsigned build = 0; build < 2; ++build) {
      struct hk_cmd_buffer *cmd = new_cmd();
      struct hk_cs *comp = add(cmd, true);
      cmd->sync.as_build = build;
      hk_sync_barrier(cmd, src, dst);
      struct hk_cs *next = add(cmd, true);
      cmd->sync.as_build = false;

      hk_sync_begin_rendering(cmd, false);
      struct hk_cs *gfx = add(cmd, false);
      hk_sync_end_rendering(cmd, false);

      T("the dependency orders the next dispatch",
        next->sync.dep[HK_SYNC_CDM].cut == comp);
      if (!build) {
         T("a render of direct draws waits in its vertex half",
           gfx->sync.dep_ta[HK_SYNC_CDM].cut &&
              gfx->sync.dep_ta[HK_SYNC_CDM].cut->sync.seq >= comp->sync.seq);
         T("and in its fragment half",
           gfx->sync.dep[HK_SYNC_CDM].cut &&
              gfx->sync.dep[HK_SYNC_CDM].cut->sync.seq >= comp->sync.seq);
      } else {
         T("a build's own dependency leaves the render unordered",
           gfx->sync.dep_ta[HK_SYNC_CDM].cut == NULL &&
              gfx->sync.dep[HK_SYNC_CDM].cut == NULL);
      }

      /* The application's dependency after the build. */
      hk_sync_barrier(cmd, src, dst);
      hk_sync_begin_rendering(cmd, false);
      struct hk_cs *later = add(cmd, false);
      hk_sync_end_rendering(cmd, false);
      T("an application dependency after the build orders later renders",
        later->sync.dep_ta[HK_SYNC_CDM].cut == next &&
           later->sync.dep[HK_SYNC_CDM].cut == next);
   }
}

/* U1/U2: chips with AGX_CAP_CDM_OVERLAP_SPLIT (G17P and M5) keep independent
 * compute streams split so later barriers do not retroactively stall earlier
 * compute behind a render or delay a graphics consumer behind independent
 * post-barrier compute, while pure compute ALL->ALL chains stay coalesced. */
static void
test_cdm_overlap_splits(void)
{
   unsigned tested = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(end_stamp_chips); ++i) {
      const enum agx_chip chip = end_stamp_chips[i];
      if (!agx_chip_has_cap(chip, AGX_CAP_CDM_OVERLAP_SPLIT))
         continue;
      tested++;

      /* U1 same-CB: R1 -> C1(indep) -> barrier(GFX->COMPUTE) -> C2 */
      {
         struct hk_queue *q = new_queue(true);
         struct hk_cmd_buffer *cmd = timestamp_cmd(chip, true);
         struct hk_cs *r1 = add(cmd, false);
         r1->stats.cmds = 1;
         struct hk_cs *c1 = add(cmd, true);
         cmd->current_cs.cs = c1;
         c1->stats.cmds = 1;
         hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_COMPUTE);
         hk_sync_check_cs(cmd);
         T("U1 same-CB closes C1 before render-dependent C2",
           cmd->current_cs.cs == NULL);
         T("U1 same-CB C1 keeps no VDM wait",
           c1->sync.dep[HK_SYNC_VDM].cut == NULL &&
           c1->sync.dep[HK_SYNC_VDM].prior == 0);
         struct hk_cs *c2 = add(cmd, true);
         c2->stats.cmds = 1;
         T("U1 same-CB C2 waits for R1", c2->sync.dep[HK_SYNC_VDM].cut == r1);
         T("U1 same-CB merge_delays preserves split",
           hk_sync_merge_delays(cmd, c1, c2));
         T("U1 same-CB resolves", submit(q, cmd));
         T("U1 same-CB C1 has no VDM barrier, C2 waits for R1",
           c1->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
           c1->sync.prior[HK_SYNC_VDM] == 0 &&
           c2->sync.barrier[HK_SYNC_VDM] == 1);
      }

      /* U1 cross-CB prior: CB1(R1), CB2(C1 -> barrier(GFX->COMPUTE) -> C2) */
      {
         struct hk_queue *q = new_queue(true);
         struct hk_cmd_buffer *cb1 = timestamp_cmd(chip, true);
         struct hk_cs *r1 = add(cb1, false);
         r1->stats.cmds = 1;
         T("U1 cross-CB CB1 resolves", submit(q, cb1));

         struct hk_cmd_buffer *cb2 = timestamp_cmd(chip, true);
         struct hk_cs *c1 = add(cb2, true);
         cb2->current_cs.cs = c1;
         c1->stats.cmds = 1;
         hk_sync_barrier(cb2, HK_SYNC_GFX, HK_SYNC_COMPUTE);
         hk_sync_check_cs(cb2);
         T("U1 cross-CB closes C1 before prior-VDM-dependent C2",
           cb2->current_cs.cs == NULL);
         T("U1 cross-CB C1 keeps no prior VDM wait",
           c1->sync.dep[HK_SYNC_VDM].prior == 0);
         struct hk_cs *c2 = add(cb2, true);
         c2->stats.cmds = 1;
         T("U1 cross-CB C2 carries prior VDM wait",
           c2->sync.dep[HK_SYNC_VDM].prior != 0);
         T("U1 cross-CB merge_delays preserves split",
           hk_sync_merge_delays(cb2, c1, c2));
         T("U1 cross-CB CB2 resolves", submit(q, cb2));
         T("U1 cross-CB C1 prior_vdm=0, C2 prior_vdm=1",
           c1->sync.prior[HK_SYNC_VDM] == 0 &&
           c2->sync.prior[HK_SYNC_VDM] == 1);
      }

      /* U2 same-CB: C1 -> barrier(COMPUTE->GFX) -> C2(indep) -> R1 */
      {
         struct hk_queue *q = new_queue(true);
         struct hk_cmd_buffer *cmd = timestamp_cmd(chip, true);
         struct hk_cs *c1 = add(cmd, true);
         cmd->current_cs.cs = c1;
         c1->stats.cmds = 1;
         hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_GFX);
         hk_sync_check_cs(cmd);
         T("U2 same-CB closes C1 after COMPUTE->GFX barrier",
           cmd->current_cs.cs == NULL);
         struct hk_cs *c2 = add(cmd, true);
         cmd->current_cs.cs = c2;
         c2->stats.cmds = 1;
         struct hk_cs *r1 = add(cmd, false);
         r1->stats.cmds = 1;
         T("U2 same-CB R1 waits on C1 not C2",
           r1->sync.dep[HK_SYNC_CDM].cut == c1);
         T("U2 same-CB merge_delays preserves C1/C2 split",
           hk_sync_merge_delays(cmd, c1, c2));
         T("U2 same-CB resolves", submit(q, cmd));
         T("U2 same-CB R1 barrier_cdm=1 while C2 index=2",
           r1->sync.barrier[HK_SYNC_CDM] == 1 && c2->sync.index == 2);
      }

      /* U2 cross-CB: CB1(C1 -> barrier(COMPUTE->GFX) -> C2), CB2(R1) */
      {
         struct hk_queue *q = new_queue(true);
         struct hk_cmd_buffer *cb1 = timestamp_cmd(chip, true);
         struct hk_cs *c1 = add(cb1, true);
         cb1->current_cs.cs = c1;
         c1->stats.cmds = 1;
         hk_sync_barrier(cb1, HK_SYNC_COMPUTE, HK_SYNC_GFX);
         hk_sync_check_cs(cb1);
         T("U2 cross-CB closes C1", cb1->current_cs.cs == NULL);
         struct hk_cs *c2 = add(cb1, true);
         c2->stats.cmds = 1;
         T("U2 cross-CB merge_delays preserves C1/C2 split for tail GFX req",
           hk_sync_merge_delays(cb1, c1, c2));
         T("U2 cross-CB CB1 resolves", submit(q, cb1));

         struct hk_cmd_buffer *cb2 = timestamp_cmd(chip, true);
         struct hk_cs *r1 = add(cb2, false);
         r1->stats.cmds = 1;
         T("U2 cross-CB CB2 resolves", submit(q, cb2));
         T("U2 cross-CB R1 prior_cdm=1 (C1) not 2 (C2)",
           r1->sync.prior[HK_SYNC_CDM] == 1 && q->sync.submitted[HK_SYNC_CDM] == 2);
      }

      /* Pure compute chain with ALL_COMMANDS -> ALL_COMMANDS must NOT split C1
       * when no render stream waits on C1 (preserves stream coalescing & CDM
       * multi-CB packing). Once an actual render waits on C1, later compute
       * still splits so the render does not wait on it. */
      {
         struct hk_cmd_buffer *cmd = timestamp_cmd(chip, true);
         struct hk_cs *c1 = add(cmd, true);
         cmd->current_cs.cs = c1;
         c1->stats.cmds = 1;
         hk_sync_barrier(cmd, HK_SYNC_ALL, HK_SYNC_ALL);
         hk_sync_check_cs(cmd);
         T("ALL->ALL pure compute chain keeps C1 open (no spurious CDM split)",
           cmd->current_cs.cs == c1);

         struct hk_cs *r1 = add(cmd, false);
         r1->stats.cmds = 1;
         hk_sync_check_cs(cmd);
         T("ALL->ALL followed by actual render R1 still splits C1 before later compute",
           cmd->current_cs.cs == NULL);
      }

      /* U2 re-merge: C1 -> bar(C->G) -> C2 -> bar(C->G) allows merging C1+C2 */
      {
         struct hk_cmd_buffer *cmd = timestamp_cmd(chip, true);
         struct hk_cs *c1 = add(cmd, true);
         cmd->current_cs.cs = c1;
         c1->stats.cmds = 1;
         hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_GFX);
         hk_sync_check_cs(cmd);
         struct hk_cs *c2 = add(cmd, true);
         cmd->current_cs.cs = c2;
         c2->stats.cmds = 1;
         hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_GFX);
         T("U2 later COMPUTE->GFX barrier lets C1 and C2 merge",
           !hk_sync_merge_delays(cmd, c1, c2));
      }
   }

   T("at least one chip tests CDM overlap splits", tested > 0);

   /* Unmeasured chips (G15, G16G, G17D) do not enable compute overlap splits yet. */
   const enum agx_chip excluded[] = {AGX_CHIP_G15G, AGX_CHIP_G16G, AGX_CHIP_G17D};
   for (unsigned i = 0; i < ARRAY_SIZE(excluded); ++i) {
      struct hk_cmd_buffer *cmd = timestamp_cmd(excluded[i], true);
      struct hk_cs *c1 = add(cmd, true);
      cmd->current_cs.cs = c1;
      c1->stats.cmds = 1;
      hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_GFX);
      hk_sync_check_cs(cmd);
      T("unmeasured chip keeps C1 open (no extra CDM split)",
        cmd->current_cs.cs == c1);
   }
}

/* A compute barrier recorded while a compute stream is open names that stream
 * as its compute cut, and the cut covers every compute stream before it too.
 * Work after the barrier may go into the open stream only if the stream
 * already waits for the stream before it: the kernel may run earlier compute
 * streams of the queue on other firmware channels
 * (DRM_ASAHI_FEATURE_COMPUTE_MULTI_CHANNEL). */
static void
test_compute_self_cut(void)
{
   const enum agx_chip chips[] = {
      AGX_CHIP_G16G, AGX_CHIP_G17P, AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(chips); ++i) {
      for (unsigned exact = 0; exact < 2; ++exact) {
         /* Dispatch A in c1, which a timestamp or a render pass ended, then
          * unrelated dispatch X' in c2, a compute barrier, and dispatch B. */
         {
            struct hk_cmd_buffer *cmd = timestamp_cmd(chips[i], exact);
            struct hk_cs *c1 = add(cmd, true);
            c1->stats.cmds = 1;
            struct hk_cs *c2 = add(cmd, true);
            cmd->current_cs.cs = c2;
            c2->stats.cmds = 1;
            T("self cut: c2 does not wait for c1",
              c2->sync.dep[HK_SYNC_CDM].cut == NULL);
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
            hk_sync_check_cs(cmd);
            T("self cut: the barrier ends c2", cmd->current_cs.cs == NULL);
            T("self cut: c2 keeps its own waits",
              c2->sync.dep[HK_SYNC_CDM].cut == NULL);
            struct hk_cs *c3 = add(cmd, true);
            c3->stats.cmds = 1;
            T("self cut: B waits for c2 and the streams before it",
              c3->sync.dep[HK_SYNC_CDM].cut == c2);
            T("self cut: resolves", submit(new_queue(exact), cmd));
            T("self cut: B's barrier names c2",
              c2->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE &&
              c3->sync.barrier[HK_SYNC_CDM] == c2->sync.index);
         }

         /* c2 already waits for c1: B goes into c2. */
         {
            struct hk_cmd_buffer *cmd = timestamp_cmd(chips[i], exact);
            struct hk_cs *c1 = add(cmd, true);
            c1->stats.cmds = 1;
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
            struct hk_cs *c2 = add(cmd, true);
            cmd->current_cs.cs = c2;
            c2->stats.cmds = 1;
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
            hk_sync_check_cs(cmd);
            T("self cut: a stream waiting for the one before it stays open",
              cmd->current_cs.cs == c2 &&
              c2->sync.dep[HK_SYNC_CDM].cut == c1);
         }

         /* The first compute stream has nothing before it. */
         {
            struct hk_cmd_buffer *cmd = timestamp_cmd(chips[i], exact);
            struct hk_cs *c1 = add(cmd, true);
            cmd->current_cs.cs = c1;
            c1->stats.cmds = 1;
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
            hk_sync_check_cs(cmd);
            T("self cut: the first compute stream stays open",
              cmd->current_cs.cs == c1);
         }

         /* Compute, a barrier to graphics, a render, a barrier back to
          * compute: the barriers chain, so c2 waits for c1 and B goes into
          * c2. */
         {
            struct hk_cmd_buffer *cmd = timestamp_cmd(chips[i], exact);
            struct hk_cs *c1 = add(cmd, true);
            cmd->current_cs.cs = c1;
            c1->stats.cmds = 1;
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_GFX);
            hk_sync_check_cs(cmd);
            cmd->current_cs.cs = NULL;
            struct hk_cs *r1 = add(cmd, false);
            r1->stats.cmds = 1;
            hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_COMPUTE);
            struct hk_cs *c2 = add(cmd, true);
            cmd->current_cs.cs = c2;
            c2->stats.cmds = 1;
            T("self cut after a render: c2 waits for r1 and c1",
              c2->sync.dep[HK_SYNC_VDM].cut == r1 &&
              c2->sync.dep[HK_SYNC_CDM].cut == c1);
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
            hk_sync_check_cs(cmd);
            T("self cut after a render: c2 stays open",
              cmd->current_cs.cs == c2);
         }

         /* A command that needs a stream after the barrier
          * (hk_sync_get_cs_after()): c2 ends. The fixture records into an
          * errored command buffer, which opens no new stream. */
         {
            struct hk_cmd_buffer *cmd = timestamp_cmd(chips[i], exact);
            struct hk_cs *c1 = add(cmd, true);
            c1->stats.cmds = 1;
            struct hk_cs *c2 = add(cmd, true);
            cmd->current_cs.cs = c2;
            c2->stats.cmds = 1;
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
            cmd->vk.record_result = VK_ERROR_OUT_OF_HOST_MEMORY;
            struct hk_sync_dep none[HK_SYNC_NUM_SQ] = {0};
            hk_sync_get_cs_after(cmd, none);
            T("self cut: a command after the barrier ends c2",
              cmd->current_cs.cs == NULL &&
              c2->sync.dep[HK_SYNC_CDM].cut == NULL);
         }

         /* A pinned stream cannot end: it waits for c1 instead. */
         {
            struct hk_cmd_buffer *cmd = timestamp_cmd(chips[i], exact);
            struct hk_cs *c1 = add(cmd, true);
            c1->stats.cmds = 1;
            struct hk_cs *c2 = add(cmd, true);
            cmd->current_cs.cs = c2;
            c2->stats.cmds = 1;
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
            cmd->sync.pin_cs = true;
            struct hk_sync_dep none[HK_SYNC_NUM_SQ] = {0};
            T("self cut: a pinned stream is kept",
              hk_sync_get_cs_after(cmd, none) == c2);
            T("self cut: a pinned stream waits for the stream before it",
              c2->sync.dep[HK_SYNC_CDM].cut == c1);
         }

         /* A host-availability flush with c2 open. */
         {
            struct hk_cmd_buffer *cmd = timestamp_cmd(chips[i], exact);
            struct hk_cs *c1 = add(cmd, true);
            c1->stats.cmds = 1;
            struct hk_cs *c2 = add(cmd, true);
            cmd->current_cs.cs = c2;
            c2->stats.cmds = 1;
            const VkMemoryBarrier2 host = {
               .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
               .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
               .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
               .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
            };
            const VkDependencyInfo info = {
               .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
               .memoryBarrierCount = 1,
               .pMemoryBarriers = &host,
            };
            /* Recorded into c2, or deferred to the end of the submission
             * where it follows every stream. */
            struct hk_cs *flush = hk_sync_host_flush(cmd, &info);
            T("self cut: a host flush in c2 makes c2 wait for c1",
              flush == c2 ? c2->sync.dep[HK_SYNC_CDM].cut == c1
                          : !flush && (cmd->sync.host_flush_kinds &
                                       HK_SYNC_COMPUTE));
         }

         /* A pure query-memory predecessor (HK_SYNC_STAMP or HK_SYNC_QRESET)
          * is never named by an application barrier: c2 stays open across a
          * compute barrier when no earlier HK_SYNC_ALL compute stream sits
          * behind it. */
         const uint8_t non_app[] = {HK_SYNC_STAMP, HK_SYNC_QRESET};
         for (unsigned k = 0; k < ARRAY_SIZE(non_app); ++k) {
            struct hk_cmd_buffer *cmd = timestamp_cmd(chips[i], exact);
            cmd->sync.kind_override = non_app[k];
            struct hk_cs *s0 = add(cmd, true);
            cmd->sync.kind_override = 0;
            s0->stats.cmds = 1;
            struct hk_cs *c2 = add(cmd, true);
            cmd->current_cs.cs = c2;
            c2->stats.cmds = 1;
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
            hk_sync_check_cs(cmd);
            T("self cut: pure query-memory predecessor keeps c2 open",
              cmd->current_cs.cs == c2 &&
              c2->sync.dep[HK_SYNC_CDM].cut == NULL);
         }

         /* An earlier unwaited application compute stream behind a pure
          * HK_SYNC_STAMP stream still forces c2 to end across a barrier. */
         {
            struct hk_cmd_buffer *cmd = timestamp_cmd(chips[i], exact);
            struct hk_cs *c0 = add(cmd, true);
            c0->stats.cmds = 1;
            cmd->sync.kind_override = HK_SYNC_STAMP;
            struct hk_cs *s1 = add(cmd, true);
            cmd->sync.kind_override = 0;
            s1->stats.cmds = 1;
            struct hk_cs *c2 = add(cmd, true);
            cmd->current_cs.cs = c2;
            c2->stats.cmds = 1;
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
            hk_sync_check_cs(cmd);
            T("self cut: unwaited compute behind stamp stream still ends c2",
              cmd->current_cs.cs == NULL &&
              c2->sync.dep[HK_SYNC_CDM].cut == NULL);
         }

         /* Once c2 already waits for the earlier application compute stream
          * c0, an intervening pure HK_SYNC_STAMP stream s1 does not split c2
          * across a compute barrier. */
         {
            struct hk_cmd_buffer *cmd = timestamp_cmd(chips[i], exact);
            struct hk_cs *c0 = add(cmd, true);
            c0->stats.cmds = 1;
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
            cmd->sync.kind_override = HK_SYNC_STAMP;
            struct hk_cs *s1 = add(cmd, true);
            cmd->sync.kind_override = 0;
            s1->stats.cmds = 1;
            struct hk_cs *c2 = add(cmd, true);
            cmd->current_cs.cs = c2;
            c2->stats.cmds = 1;
            T("self cut: c2 waits on c0 and skips intervening stamp stream",
              c2->sync.dep[HK_SYNC_CDM].cut == c0);
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
            hk_sync_check_cs(cmd);
            T("self cut: intervening stamp stream keeps waited c2 open",
              cmd->current_cs.cs == c2 &&
              c2->sync.dep[HK_SYNC_CDM].cut == c0);
         }

         /* A query/event caller through hk_sync_get_cs_after() with a
          * dependency on an earlier HK_SYNC_STAMP stream must still end c2
          * (or make a pinned c2 wait on the stamp stream) when a compute
          * barrier produced a self cut at c2. */
         {
            struct hk_cmd_buffer *cmd = timestamp_cmd(chips[i], exact);
            cmd->sync.kind_override = HK_SYNC_STAMP;
            struct hk_cs *s0 = add(cmd, true);
            cmd->sync.kind_override = 0;
            s0->stats.cmds = 1;
            struct hk_cs *c2 = add(cmd, true);
            cmd->current_cs.cs = c2;
            c2->stats.cmds = 1;
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
            struct hk_sync_dep qdep[HK_SYNC_NUM_SQ];
            hk_sync_src_dep(cmd, HK_SYNC_QUERY_SCOPE, c2, qdep);
            cmd->vk.record_result = VK_ERROR_OUT_OF_HOST_MEMORY;
            hk_sync_get_cs_after(cmd, qdep);
            T("self cut: query scope after stamp stream still ends c2",
              cmd->current_cs.cs == NULL &&
              c2->sync.dep[HK_SYNC_CDM].cut == NULL);
         }
         {
            struct hk_cmd_buffer *cmd = timestamp_cmd(chips[i], exact);
            cmd->sync.kind_override = HK_SYNC_STAMP;
            struct hk_cs *s0 = add(cmd, true);
            cmd->sync.kind_override = 0;
            s0->stats.cmds = 1;
            struct hk_cs *c2 = add(cmd, true);
            cmd->current_cs.cs = c2;
            c2->stats.cmds = 1;
            hk_sync_barrier(cmd, HK_SYNC_COMPUTE, HK_SYNC_COMPUTE);
            struct hk_sync_dep qdep[HK_SYNC_NUM_SQ];
            hk_sync_src_dep(cmd, HK_SYNC_QUERY_SCOPE, c2, qdep);
            cmd->sync.pin_cs = true;
            T("self cut: pinned query scope after stamp stream waits on s0",
              hk_sync_get_cs_after(cmd, qdep) == c2 &&
              c2->sync.dep[HK_SYNC_CDM].cut == s0);
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
   test_draw_indirect_scope();
   test_barrier_chains_without_work();
   test_multiview_layer_preservation();
   test_compute_clear_layer_preservation();
   test_resume_after_compute_clear();
   test_loaded_color_preservation();
   test_history_tile_hazard_split();
   test_load_hazard_frag();
   test_pass_software_graphics_dependencies();
   test_image_barrier_vertex_class();
   test_discard_transition();
   test_discard_transition_carry();
   test_g17p_vertex_after_memory_writers();
   test_g17p_tiling_order();
   test_sibling_prefix();
   test_chain_first_skips_empty_buffers();
   test_repeated_recording();
   test_rt_kernel_gate();
   test_timestamp_execution_scopes();
   test_timestamp_reused_compute();
   test_timestamp_top_in_pass();
   test_timestamp_leading_top();
   test_stamp_kind_isolation();
   test_cdm_overlap_splits();
   test_compute_self_cut();
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
   T("same-submit", hk_sync_resolve_submit(q, pair, 2, false, false, 0, NULL));
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
   T("synthetic SSTT only", hk_sync_resolve_submit(q, NULL, 0, true, false, 0, NULL));
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
     hk_sync_resolve_submit(q, single, 1, true, false, 0, NULL));
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
   /* Do not merge a later independent compute stream into an earlier compute
    * stream that a subsequent graphics pass waits on (e.g. AS build after
    * deform ahead of a G-buffer pass). */
   {
      struct hk_cmd_buffer *m = new_cmd();
      hk_cmd_buffer_device(m)->dev.chip = AGX_CHIP_G17C;
      struct hk_cs *deform = add(m, true);
      deform->stats.cmds = 1;
      struct hk_cs *as_build = add(m, true);
      as_build->stats.cmds = 1;
      struct hk_cs *render = add(m, false);
      render->stats.cmds = 1;
      render->sync.dep[HK_SYNC_CDM].cut = deform;
      T("merge delays when later render waits on earlier compute",
        hk_sync_merge_delays(m, deform, as_build));
      render->sync.dep[HK_SYNC_CDM].cut = NULL;
      render->sync.dep_ta[HK_SYNC_CDM].cut = deform;
      T("merge delays when later render TA waits on earlier compute",
        hk_sync_merge_delays(m, deform, as_build));
      render->sync.dep_ta[HK_SYNC_CDM].cut = NULL;
      T("merge allowed when later render does not wait on earlier compute",
        !hk_sync_merge_delays(m, deform, as_build));
   }
   /* hk_sync_resolve_submit_ex() tail_raised_out and barrier-free empty CBs. */
   {
      struct hk_queue *eq = new_queue(true);
      struct hk_cmd_buffer *empty = new_cmd();
      T("barrier-free empty CB has epoch 0", empty->sync.epoch == 0);
      bool tail_raised = true;
      T("empty CB on idle queue",
        hk_sync_resolve_submit_ex(eq, &empty, 1, false, false, 0, NULL,
                                  &tail_raised) &&
           !tail_raised);

      /* Barrier-only CB on idle queue cannot raise tail from 0. */
      struct hk_cmd_buffer *bar = new_cmd();
      hk_sync_barrier(bar, HK_SYNC_GFX, HK_SYNC_COMPUTE);
      T("barrier-only CB sets epoch", bar->sync.epoch != 0);
      tail_raised = true;
      T("barrier-only CB on idle queue does not raise tail",
        hk_sync_resolve_submit_ex(eq, &bar, 1, false, false, 0, NULL,
                                  &tail_raised) &&
           !tail_raised);

      /* Non-empty render without trailing barrier does not raise tail. */
      struct hk_cmd_buffer *src = new_cmd();
      add(src, false);
      tail_raised = true;
      T("render without barrier does not raise tail",
        hk_sync_resolve_submit_ex(eq, &src, 1, false, false, 0, NULL,
                                  &tail_raised) &&
           !tail_raised);

      /* Barrier-free empty CB after render does not change tail or ordinals. */
      uint64_t tail_before[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];
      memcpy(tail_before, eq->sync.tail, sizeof(tail_before));
      tail_raised = true;
      T("barrier-free empty CB after render leaves tail unchanged",
        hk_sync_resolve_submit_ex(eq, &empty, 1, false, false, 0, NULL,
                                  &tail_raised) &&
           !tail_raised &&
           memcmp(tail_before, eq->sync.tail, sizeof(tail_before)) == 0 &&
           eq->sync.submitted[HK_SYNC_VDM] == 1 &&
           eq->sync.submitted[HK_SYNC_CDM] == 0);

      /* Barrier-only CB after render raises tail once, not on repeat. */
      tail_raised = false;
      T("barrier-only CB after render raises tail",
        hk_sync_resolve_submit_ex(eq, &bar, 1, false, false, 0, NULL,
                                  &tail_raised) &&
           tail_raised);
      tail_raised = true;
      T("repeated barrier-only CB with no new work does not raise tail again",
        hk_sync_resolve_submit_ex(eq, &bar, 1, false, false, 0, NULL,
                                  &tail_raised) &&
           !tail_raised);

      /* Non-empty CB with a trailing barrier raises tail. */
      struct hk_cmd_buffer *src2 = new_cmd();
      add(src2, true);
      hk_sync_barrier(src2, HK_SYNC_COMPUTE, HK_SYNC_GFX);
      tail_raised = false;
      T("compute with trailing barrier raises tail",
        hk_sync_resolve_submit_ex(eq, &src2, 1, false, false, 0, NULL,
                                  &tail_raised) &&
           tail_raised);

      /* Legacy queue (exact_prior == false) never sets tail_raised. */
      struct hk_queue *lq = new_queue(false);
      tail_raised = true;
      T("legacy queue never raises exact tail",
        hk_sync_resolve_submit_ex(lq, &src2, 1, false, false, 0, NULL,
                                  &tail_raised) &&
           !tail_raised);
   }
   /* ta_carry == carry dedup keeps identical TA fields on compute and split
    * fields on fragment-only graphics barriers. */
   {
      struct hk_queue *tq = new_queue(true);
      struct hk_cmd_buffer *s0 = new_cmd();
      add(s0, false);
      hk_sync_barrier(s0, HK_SYNC_GFX, HK_SYNC_COMPUTE | HK_SYNC_GFX_FRAG);
      T("ta dedup setup submit", submit(tq, s0));

      struct hk_cmd_buffer *s1 = new_cmd();
      struct hk_cs *cs_cdm = add(s1, true);
      struct hk_cs *cs_vdm = add(s1, false);
      T("ta dedup consumer submit", submit(tq, s1));
      T("compute dedup matches main and TA fields",
        cs_cdm->sync.prior[HK_SYNC_VDM] == 1 &&
           cs_cdm->sync.prior_ta[HK_SYNC_VDM] == 1 &&
           cs_cdm->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
           cs_cdm->sync.barrier_ta[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE);
      T("render fragment-only barrier still splits TA",
        cs_vdm->sync.prior[HK_SYNC_VDM] == 1 &&
           cs_vdm->sync.prior_ta[HK_SYNC_VDM] == 0 && cs_vdm->sync.ta_split);
   }
   /* hk_queue_note_signalled() advances monotonically without regressing. */
   {
      struct hk_queue *mq = new_queue(true);
      mq->drm.signalled_value = 0;
      hk_queue_note_signalled(mq, 5);
      T("signalled_value advances from 0 to 5",
        p_atomic_read(&mq->drm.signalled_value) == 5);
      hk_queue_note_signalled(mq, 12);
      T("signalled_value advances from 5 to 12",
        p_atomic_read(&mq->drm.signalled_value) == 12);
      hk_queue_note_signalled(mq, 7);
      T("signalled_value ignores lower point",
        p_atomic_read(&mq->drm.signalled_value) == 12);
      hk_queue_note_signalled(mq, 12);
      T("signalled_value ignores equal point",
        p_atomic_read(&mq->drm.signalled_value) == 12);
   }
   /* Same-queue semaphore waits (hk_gpu_waits.h): a wait floor makes every
    * command of the submission wait through the signalling submission's
    * ordinals, on both subqueues, whatever its own dependencies; it never
    * lowers a later in-submission wait, applies to a deferred host flush,
    * and is not carried to the next submission. */
   {
      struct hk_queue *q = new_queue(true);
      struct hk_cmd_buffer *first = new_cmd();
      add(first, false);
      add(first, true);
      add(first, false);
      T("floor: producer resolves", submit(q, first));
      T("floor: producer ordinals",
        q->sync.submitted[HK_SYNC_VDM] == 2 && q->sync.submitted[HK_SYNC_CDM] == 1);
      const uint64_t floor_vdm = q->sync.submitted[HK_SYNC_VDM];
      const uint64_t floor_cdm = q->sync.submitted[HK_SYNC_CDM];

      struct hk_cmd_buffer *mid = new_cmd();
      add(mid, true);
      T("floor: middle resolves", submit(q, mid));

      struct hk_cmd_buffer *c = new_cmd();
      struct hk_cs *r = add(c, false);
      struct hk_cs *k = add(c, true);
      hk_sync_barrier(c, HK_SYNC_COMPUTE, HK_SYNC_GFX);
      struct hk_cs *r2 = add(c, false);
      q->sync.wait_floor[HK_SYNC_VDM] = floor_vdm;
      q->sync.wait_floor[HK_SYNC_CDM] = floor_cdm;
      T("floor: consumer resolves", submit(q, c));
      memset(q->sync.wait_floor, 0, sizeof(q->sync.wait_floor));

      T("floor: render waits through the producer",
        r->sync.prior[HK_SYNC_VDM] >= floor_vdm &&
        r->sync.prior[HK_SYNC_CDM] >= floor_cdm &&
        r->sync.prior_ta[HK_SYNC_VDM] >= floor_vdm &&
        r->sync.prior_ta[HK_SYNC_CDM] >= floor_cdm);
      T("floor: compute waits through the producer",
        k->sync.prior[HK_SYNC_VDM] >= floor_vdm &&
        k->sync.prior[HK_SYNC_CDM] >= floor_cdm);
      T("floor: in-submission wait kept",
        r2->sync.barrier[HK_SYNC_CDM] == 1 &&
        r2->sync.prior[HK_SYNC_VDM] >= floor_vdm);

      /* Deferred host flush on a barrier-only submission also respects
       * wait_floor when no stream ran on either subqueue. */
      uint16_t fb[HK_SYNC_NUM_SQ];
      uint64_t fp[HK_SYNC_NUM_SQ];
      const uint32_t zero_count[HK_SYNC_NUM_SQ] = {0, 0};
      const uint64_t base_now[HK_SYNC_NUM_SQ] = {
         q->sync.submitted[HK_SYNC_VDM], q->sync.submitted[HK_SYNC_CDM],
      };
      q->sync.wait_floor[HK_SYNC_VDM] = floor_vdm;
      q->sync.wait_floor[HK_SYNC_CDM] = floor_cdm;
      T("floor: host flush prefix resolves",
        hk_sync_host_flush_prefix(&q->sync, 0, base_now, zero_count, fb, fp));
      memset(q->sync.wait_floor, 0, sizeof(q->sync.wait_floor));
      T("floor: host flush respects wait_floor",
        fb[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
        fb[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE &&
        fp[HK_SYNC_VDM] >= floor_vdm && fp[HK_SYNC_CDM] >= floor_cdm);

      struct hk_cmd_buffer *next = new_cmd();
      struct hk_cs *n = add(next, true);
      T("floor: next resolves", submit(q, next));
      T("floor: not carried to the next submission",
        n->sync.prior[HK_SYNC_VDM] == 0 && n->sync.prior[HK_SYNC_CDM] == 0 &&
        n->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
        n->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE);
   }
   test_vertex_image_bloom();
   /* Forgetting a singleton render pass (such as an absorbed deferred-clear
    * VDM render) drops its pass footprint instead of rebinding it to an
    * earlier unrelated render on the subqueue. */
   {
      const void *img_a = (const void *)0x1111;
      const void *img_b = (const void *)0x2222;
      struct hk_cmd_buffer *cmd = new_cmd();
      struct hk_cs *pass_a = render_pass(cmd, img_a, 1, false);
      struct hk_cs *clear_b = render_pass(cmd, img_b, 1, false);
      hk_sync_forget_stream(cmd, clear_b);
      list_del(&clear_b->node);

      hk_sync_begin_rendering(cmd, false);
      hk_sync_pass_footprint(cmd, img_b, 0, 0, 1, 32, 32,
                             (VkRect2D){{0, 0}, {32, 32}}, false, true);
      hk_sync_pass_footprints_done(cmd);
      struct hk_cs *draw_b = add(cmd, false);
      hk_sync_end_rendering(cmd, false);

      T("forgotten singleton clear leaves no false dependency on earlier pass",
        draw_b->sync.dep[HK_SYNC_VDM].cut == NULL &&
           draw_b->sync.dep_ta[HK_SYNC_VDM].cut == NULL &&
           pass_a->sync.seq != 0);
   }
   /* Retired exact prior pruning drops completed VDM/CDM ordinals while
    * preserving unretired dependencies and FRAGMENT_BARRIERS ta_split. */
   {
      struct hk_queue *rq = new_queue(true);
      /* Submit 1 (point 10): HK_SYNC_GFX_MEM render (ord 1) + barrier. */
      rq->drm.timeline_value = 10;
      struct hk_cmd_buffer *s1 = new_cmd();
      struct hk_cs *r_mem = add(s1, false);
      hk_sync_note_memory_write(s1, r_mem);
      hk_sync_barrier(s1, HK_SYNC_ALL, HK_SYNC_ALL);
      T("retired-prior setup s1",
        hk_sync_resolve_submit(rq, &s1, 1, false, false, 0, NULL));
      T("retired-prior reports unretired priors after s1",
        hk_sync_submit_has_unretired_priors(rq));

      /* Submit 2 (point 11): plain render (ord 2) + fragment-only barrier,
       * while point 10 is still in flight (completed = 0). */
      rq->drm.timeline_value = 11;
      struct hk_cmd_buffer *s2 = new_cmd();
      struct hk_cs *r2 = add(s2, false);
      hk_sync_barrier(s2, HK_SYNC_GFX, HK_SYNC_GFX_FRAG);
      T("retired-prior s2 unretired",
        hk_sync_resolve_submit(rq, &s2, 1, false, false, 0, NULL));
      T("retired-prior keeps unretired GFX_MEM prior_ta",
        r2->sync.prior[HK_SYNC_VDM] == 1 && r2->sync.prior_ta[HK_SYNC_VDM] == 1);

      /* Submit 3 (point 12): point 10 has retired (completed = 10), point 11
       * is still live -> prior_ta[VDM] (1) prunes to 0 while prior[VDM] (2)
       * stays live with ta_split=true! */
      rq->drm.timeline_value = 12;
      struct hk_cmd_buffer *s3 = new_cmd();
      struct hk_cs *r3 = add(s3, false);
      T("retired-prior s3 partial retirement",
        hk_sync_resolve_submit(rq, &s3, 1, false, false, 10, NULL));
      T("retired-prior prunes retired prior_ta while keeping live fragment prior",
        r3->sync.prior[HK_SYNC_VDM] == 2 && r3->sync.prior_ta[HK_SYNC_VDM] == 0 &&
           r3->sync.ta_split);

      /* Submit 4 (point 13): everything through point 12 has retired -> both
       * prior and prior_ta prune to 0 and ta_split clears. */
      rq->drm.timeline_value = 13;
      struct hk_cmd_buffer *s4 = new_cmd();
      struct hk_cs *r4 = add(s4, false);
      T("retired-prior s4 full retirement",
        hk_sync_resolve_submit(rq, &s4, 1, false, false, 12, NULL));
      T("retired-prior clears both priors and ta_split when all retired",
        r4->sync.prior[HK_SYNC_VDM] == 0 && r4->sync.prior_ta[HK_SYNC_VDM] == 0 &&
           !r4->sync.ta_split);
   }
   printf("RESULT checks=%u failures=%u\n", tests, failures);
   ralloc_free(test_context);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

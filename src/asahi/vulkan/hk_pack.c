/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_pack.h"

#include <string.h>

#include "util/bitscan.h"
#include "util/macros.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"

const struct hk_cs *
hk_pack_run_member(const struct hk_device *dev, const struct hk_cmd_buffer *cmd,
                   const uint8_t *terminate, bool atomics)
{
   /* Events can be observed before the submission completes, and a pending
    * top-of-pipe stamp would land on the next member's launch. */
   if (cmd->state.uses_events || cmd->sync.pending_start_ts.handle ||
       !list_is_singular(&cmd->control_streams))
      return NULL;

   const struct hk_cs *cs =
      list_first_entry(&cmd->control_streams, struct hk_cs, node);
   const unsigned kinds = cs->sync.kind & ~HK_SYNC_GFX_MEM;
   if (cs->type != HK_CS_CDM || !kinds ||
       ((kinds & ~(HK_SYNC_COMPUTE | HK_SYNC_XFER)) && kinds != HK_SYNC_ALL))
      return NULL;

   /* Every launch is a direct dispatch the work estimate counts. */
   if (!cs->stats.cmds || cs->stats.cmds != cs->stats.calls ||
       cs->pack_dispatches != cs->stats.calls || !cs->pack_threads)
      return NULL;

   if ((cs->uses_global_atomics && !atomics) || cs->calls_stream ||
       cs->stream_linked || cs->sib_indirect || cs->query_access ||
       cs->query_reset_pending || cs->timestamp.start.handle ||
       cs->timestamp.end.handle || cs->g17p_timestamp_query ||
       cs->gen1_native_rq || cs->as_build || cs->imm_writes.size)
      return NULL;

   const uint8_t *end = (const uint8_t *)cs->current;
   if (end - (const uint8_t *)cs->start <= AGX3_CDM_STREAM_TERMINATE_SIZE)
      return NULL;
   end -= AGX3_CDM_STREAM_TERMINATE_SIZE;
   if (memcmp(end, terminate, AGX3_CDM_STREAM_TERMINATE_SIZE))
      return NULL;

   if (agx_chip_g17_m5(dev->dev.chip)) {
      /* The stream ends with the strong memory-scoped flush. */
      if (!cs->visible_completion || cs->strong_flush_end != end ||
          cs->memory_flush_end != end)
         return NULL;
   } else {
      /* G16G: the stream ends with an ordering barrier or a strong flush,
       * which a boundary upgrades in place (hk_cdm_visibility_flush()). */
      if (cs->strong_flush_end != end && cs->weak_barrier_end != end)
         return NULL;
   }

   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      if (cs->sync.dep[sq].cut || cs->sync.dep_ta[sq].cut)
         return NULL;
      for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
         const struct hk_cs *cut = cmd->sync.req[k][sq].cut;
         if (cut && (sq != HK_SYNC_CDM || cut != cs))
            return NULL;
      }
   }

   return cs;
}

void
hk_pack_describe(struct hk_pack_member *member, const struct hk_cmd_buffer *cmd,
                 const struct hk_cs *cs, size_t terminate_size,
                 uint8_t render_kinds)
{
   memset(member, 0, sizeof(*member));
   if (!cs)
      return;

   /* A barrier records its source kinds against both subqueues
    * (hk_sync_src_dep()), so a compute-only barrier names COMPUTE work on
    * the render subqueue too. The resolver makes a wait of such a kind only
    * once the render subqueue ran work of it; neither does the planner. */
   member->cs = cs;
   member->bytes = (const uint8_t *)cs->current - (const uint8_t *)cs->start -
                   terminate_size;
   member->kinds = cs->sync.kind;
   member->render_waits = cs->sync.dep[HK_SYNC_VDM].prior & render_kinds;
   for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k)
      member->render_tail[k] =
         cmd->sync.req[k][HK_SYNC_VDM].prior & render_kinds;
   member->render_ran = render_kinds != 0;
}

uint8_t
hk_pack_render_kinds(const struct hk_cmd_buffer *cmd)
{
   uint8_t kinds = 0;
   list_for_each_entry(const struct hk_cs, cs, &cmd->control_streams, node) {
      if (hk_sync_sq(cs) == HK_SYNC_VDM)
         kinds |= cs->sync.kind;
   }
   return kinds;
}

/* The render kinds the tails `tails` make a packet of kinds `kinds` wait
 * for: hk_pack_fold_member() folds a member's tail of each kind the packet
 * has. */
static uint8_t
hk_pack_tail(const uint8_t tails[HK_SYNC_NUM_KINDS], uint8_t kinds)
{
   uint8_t tail = 0;
   u_foreach_bit(k, kinds)
      tail |= tails[k];
   return tail;
}

/* Whether `m` may start a run on its own. */
static bool
hk_pack_within_bounds(const struct hk_pack_member *m,
                      const struct hk_pack_limits *limits)
{
   const struct hk_cs *cs = m->cs;
   return cs->stats.cmds <= HK_PACK_MAX_DISPATCHES &&
          cs->pack_threads <= HK_PACK_MAX_THREADS &&
          cs->pack_static_work <= HK_PACK_MAX_STATIC_WORK &&
          m->bytes <= UINT32_MAX - limits->overhead &&
          cs->compute_resources.size <= UINT16_MAX &&
          cs->compute_attachments.count <= limits->max_attachments;
}

/*
 * A packet waits for everything any member waits for, and its members run
 * in order. For the compute subqueue that adds nothing: one queue runs its
 * compute commands in order (M5), or the kernel one command of the device at
 * a time (G16), so earlier members already followed what a later member
 * waits for. A render wait is different: a later member's wait on a render
 * would hold the earlier members back too. So every member must wait for no
 * render kind beyond those the first member waits for, counting the tails of
 * the members before it for the packet's kinds, which the packet waits for as
 * the separate commands would (hk_pack_fold_member()). Once the
 * render subqueue ran work, a member must not add synchronization kinds
 * either: the packet would take on the requirements carried in for them
 * from earlier command buffers. Otherwise the member starts a new packet.
 * This changes only how much is packed, never which waits are kept.
 */
unsigned
hk_pack_plan(const struct hk_pack_member *members, unsigned n,
             const struct hk_pack_limits *limits, struct hk_pack_run *runs,
             unsigned *packed_members, size_t *total)
{
   unsigned run_count = 0;
   struct hk_pack_run run = {0};
   uint64_t dispatches = 0, threads = 0, work = 0;
   size_t resources = 0;
   uint32_t attachments = 0;
   uint8_t kinds = 0, waits = 0, tails[HK_SYNC_NUM_KINDS] = {0};
   bool loops = false;

   *packed_members = 0;
   *total = 0;

   for (unsigned i = 0; i <= n; ++i) {
      const struct hk_pack_member *m = i < n ? &members[i] : NULL;
      const struct hk_cs *cs = m ? m->cs : NULL;

      const bool fits = cs && run.count &&
         !(m->render_ran && (m->kinds & ~kinds)) &&
         !((m->render_waits | hk_pack_tail(tails, kinds | m->kinds)) &
           ~waits) &&
         !(loops && cs->pack_loops) &&
         dispatches + cs->stats.cmds <= HK_PACK_MAX_DISPATCHES &&
         cs->pack_threads <= HK_PACK_MAX_THREADS - threads &&
         cs->pack_static_work <= HK_PACK_MAX_STATIC_WORK - work &&
         m->bytes <= UINT32_MAX - limits->overhead - run.bytes &&
         cs->compute_resources.size <= UINT16_MAX - resources &&
         cs->compute_attachments.count <=
            limits->max_attachments - attachments;
      if (fits) {
         run.count++;
         run.bytes += m->bytes;
         dispatches += cs->stats.cmds;
         threads += cs->pack_threads;
         work += cs->pack_static_work;
         resources += cs->compute_resources.size;
         attachments += cs->compute_attachments.count;
         kinds |= m->kinds;
         for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k)
            tails[k] |= m->render_tail[k];
         loops |= cs->pack_loops;
         continue;
      }

      /* Close the open run; a single member stays an ordinary command. */
      if (run.count >= 2) {
         run.offset = *total;
         run.bytes += limits->overhead;
         *total = ALIGN_POT(*total + run.bytes, 1024);
         runs[run_count++] = run;
         *packed_members += run.count;
      }
      run = (struct hk_pack_run){0};
      dispatches = threads = work = 0;
      resources = 0;
      attachments = 0;

      /* Start a new run with this member when it is within the bounds. */
      if (cs && hk_pack_within_bounds(m, limits)) {
         run = (struct hk_pack_run){.first = i, .count = 1, .bytes = m->bytes};
         dispatches = cs->stats.cmds;
         threads = cs->pack_threads;
         work = cs->pack_static_work;
         resources = cs->compute_resources.size;
         attachments = cs->compute_attachments.count;
         kinds = m->kinds;
         waits = m->render_waits;
         memcpy(tails, m->render_tail, sizeof(tails));
         loops = cs->pack_loops;
      }
   }

   return run_count;
}

void
hk_pack_fold_member(struct hk_cmd_buffer *packet, struct hk_cs *cs,
                    const struct hk_cmd_buffer *source,
                    const struct hk_cs *item, bool last,
                    uint8_t *preceding_kinds)
{
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      cs->sync.dep[sq].prior |= item->sync.dep[sq].prior;
      for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
         const struct hk_sync_dep *req = &source->sync.req[k][sq];
         packet->sync.req[k][sq].prior |= req->prior;
         /* A prior-only tail can name work already in this packet;
          * later command buffers then wait for the packet. */
         if (req->cut ||
             (sq == HK_SYNC_CDM && (req->prior & *preceding_kinds)))
            packet->sync.req[k][sq].cut = cs;
         /* The members after this one wait for its tail, as separate
          * command buffers would. */
         if (!last && (cs->sync.kind & BITFIELD_BIT(k)))
            cs->sync.dep[sq].prior |= req->prior;
      }
   }
   *preceding_kinds |= item->sync.kind;
}

void
hk_pack_finish_sync(struct hk_cmd_buffer *packet, struct hk_cs *cs)
{
   u_foreach_bit(k, cs->sync.kind)
      packet->sync.last[k][HK_SYNC_CDM] = cs;
   packet->sync.last_any[HK_SYNC_CDM] = cs;
}

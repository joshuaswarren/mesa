/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_pack.h"

#include <string.h>
#include <strings.h>

#include "util/bitscan.h"
#include "util/u_debug.h"
#include "util/macros.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"

/*
 * Selected by the stream format and the kernel, never by which parts were
 * measured:
 *
 * - M5 (HAL300) streams pack where the kernel takes their own completion
 *   flush, DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION: only then does every
 *   such stream end with a strong flush (hk_EndCommandBuffer(), or an
 *   application barrier or deferred clears at the end), and a packet's end
 *   can carry the memory-scoped one (hk_pack_runs()). This holds for every
 *   M5 part.
 * - G16 (HAL200) streams pack where the strong flush is the whole visibility
 *   record. G17P differs: its caches are tagged by virtual address and its
 *   strong flush also has a memory-scoped form, which memory reached through
 *   a second address needs (hk_cdm_alias_scope()); no boundary of this kind
 *   asks for it. G17A, the other HAL200 part, sits between G17P and M5,
 *   which both have that form; until its caches are known
 *   (hk_cdm_caches_pending()) it has no boundary either.
 * - G15: the boundary would be the wide barrier its in-stream dependencies
 *   get (agx_chip_cdm_wide_visibility()), but packing there (the kernel's
 *   command deadline, attachment limits) is not qualified. AGX2 streams have
 *   neither the terminate nor the tail records this checks.
 */
enum hk_pack_boundary
hk_pack_boundary(const struct hk_device *dev)
{
   const enum agx_chip chip = dev->dev.chip;
   if (agx_chip_g17_m5(chip)) {
      return (dev->dev.params.features &
              DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION)
                ? HK_PACK_BOUNDARY_VISIBLE_COMPLETION
                : HK_PACK_BOUNDARY_NONE;
   }

   if (agx_chip_command_stream_generation(chip) == AGX_CS_GENERATION_HAL200 &&
       !hk_cdm_memory_scope_form(dev) && !hk_cdm_caches_pending(dev))
      return HK_PACK_BOUNDARY_STRONG_FLUSH;

   return HK_PACK_BOUNDARY_NONE;
}

bool
hk_pack_admits_strong_tail(enum hk_pack_boundary boundary)
{
   return boundary != HK_PACK_BOUNDARY_NONE;
}

/* debug_parse_bool_option(), plus "on" and "off", which it does not know:
 * HK_PACK_ATOMICS=off must not leave G16G's default on. */
static bool
hk_pack_parse_bool(const char *opt, bool dfault)
{
   if (!strcasecmp(opt, "off"))
      return false;
   if (!strcasecmp(opt, "on"))
      return true;
   return debug_parse_bool_option(opt, dfault);
}

bool
hk_pack_atomics_choice(bool serial_kernel, const char *opt, unsigned queues)
{
   if (serial_kernel)
      return opt ? hk_pack_parse_bool(opt, true) : true;
   return opt && hk_pack_parse_bool(opt, false) && queues <= 1;
}

const struct hk_cs *
hk_pack_run_member(const struct hk_device *dev, const struct hk_cmd_buffer *cmd,
                   const uint8_t *terminate, bool atomics)
{
   const enum hk_pack_boundary boundary = hk_pack_boundary(dev);
   if (boundary == HK_PACK_BOUNDARY_NONE)
      return NULL;

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

   /* The stream ends with a strong flush, at a boundary that admits one
    * without memory scope (hk_pack_admits_strong_tail(), which the packing
    * trace reports, so the two cannot disagree; every boundary does). At the
    * visible-completion boundary the strong flush is a last launch's tail
    * vkEndCommandBuffer upgraded to the memory-scoped one, an application
    * barrier at the end, or the one hk_flush_deferred_clears() left after
    * the clears, standing for the barriers recorded after them. The last two
    * have memory scope only where memory may have a second address
    * (hk_cdm_alias_scope()); while host_alias.dynamic decides that,
    * vkEndCommandBuffer gives every stream with launches the memory-scoped
    * end. Each is at least the dependency the application recorded across
    * the boundary; the packet's end gets the memory scope (hk_pack_runs()),
    * and this boundary exists only where the kernel accepts such a
    * visible-completion command (hk_pack_boundary()). At the HAL200 boundary
    * the stream may also end with an ordering barrier, which the boundary
    * upgrades in place (hk_cdm_visibility_flush()). */
   const bool strong_tail =
      cs->strong_flush_end == end && hk_pack_admits_strong_tail(boundary);
   const bool ordering_tail = boundary == HK_PACK_BOUNDARY_STRONG_FLUSH &&
                              cs->weak_barrier_end == end;
   if (!strong_tail && !ordering_tail)
      return NULL;

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

static void
hk_pack_describe(struct hk_pack_member *member, const struct hk_cmd_buffer *cmd,
                 const struct hk_cs *cs, uint8_t render_kinds)
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
                   AGX3_CDM_STREAM_TERMINATE_SIZE;
   member->kinds = cs->sync.kind;
   member->render_waits = cs->sync.dep[HK_SYNC_VDM].prior & render_kinds;
   for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k)
      member->render_tail[k] =
         cmd->sync.req[k][HK_SYNC_VDM].prior & render_kinds;
   member->render_ran = render_kinds != 0;
}

/* The render kinds come from the fields the resolver reads, so the two agree
 * also where a forgotten render left a command buffer's latest render of a
 * kind on a stream of another kind (hk_sync_forget_stream()). */
unsigned
hk_pack_describe_submit(struct hk_pack_member *members,
                        const struct hk_device *dev,
                        const struct hk_sync_queue *sync,
                        struct hk_cmd_buffer *const *cmds, unsigned n,
                        const uint8_t *terminate, bool atomics)
{
   unsigned eligible = 0;
   uint8_t render_kinds = hk_sync_queue_kinds(sync, HK_SYNC_VDM);
   for (unsigned i = 0; i < n; ++i) {
      const struct hk_cs *cs =
         hk_pack_run_member(dev, cmds[i], terminate, atomics);
      hk_pack_describe(&members[i], cmds[i], cs, render_kinds);
      render_kinds |= hk_sync_cmd_kinds(cmds[i], HK_SYNC_VDM);
      eligible += cs != NULL;
   }
   return eligible;
}

/* The render kinds the tails `tails` make a packet of kinds `kinds` wait
 * for: hk_pack_fold_member() folds a member's tail of each destination class
 * the packet's stream belongs to, as the resolver applies carried
 * requirements (hk_sync_full_kinds()). */
static uint8_t
hk_pack_tail(const uint8_t tails[HK_SYNC_NUM_KINDS], uint8_t kinds)
{
   uint8_t tail = 0;
   u_foreach_bit(k, hk_sync_full_kinds(kinds))
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
 * A packet waits for everything any member waits for, and its members run in
 * order. For the compute subqueue that adds nothing: one queue runs its
 * compute commands in order (M5), and the G16 kernel every compute command of
 * a VM in order on one firmware queue, so earlier members already followed
 * what a later member waits for. A render wait is different: a later member's
 * wait on a render would hold the earlier members back too. So every member
 * must wait for no render kind beyond those the first member waits for,
 * counting the tails of the members before it for the packet's kinds, which
 * the packet waits for as the separate commands would
 * (hk_pack_fold_member()). Once the render subqueue ran work, a member must
 * not add synchronization kinds either: the packet would take on the
 * requirements carried in for them from earlier command buffers. Otherwise
 * the member starts a new packet. This changes only how much is packed, never
 * which waits are kept.
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
          * command buffers would: the resolver applies a carried
          * requirement to each destination class of a stream. */
         if (!last && (hk_sync_full_kinds(cs->sync.kind) & BITFIELD_BIT(k)))
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

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_pack.h"

#include <string.h>
#include <strings.h>

#include "util/bitscan.h"
#include "util/u_atomic.h"
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
 * - AGX2 (G13/G14) streams end every dispatch with an in-stream CDM barrier
 *   (strong_flush_end) followed by AGX2's 8-byte CDM_STREAM_TERMINATE, and
 *   publish writes on command completion. Packet duration is bounded by the
 *   planner's per-packet caps (HK_PACK_MAX_DISPATCHES, HK_PACK_MAX_THREADS,
 *   HK_PACK_MAX_STATIC_WORK, and at most one looping-shader stream per packet;
 *   see hk_pack.h) while hk_pack_attachment_limit() enforces the kernel's
 *   max_attachments UAPI limit whenever reported.
 * - G15 streams end with an ordering barrier or wide-visibility strong flush
 *   (agx_chip_cdm_wide_visibility()) and AGX3's 4-byte terminate; boundaries
 *   upgrade the trailing barrier in place, subject to the same planner work
 *   and max_attachments caps.
 * - G16 and G17P (HAL200) streams pack where the caches are qualified
 *   (!hk_cdm_caches_pending()). On G17P, whose caches are tagged by virtual
 *   address, hk_pack_runs() passes hk_cdm_alias_scope(dev) to the boundary
 *   and tail flushes so aliased mappings get the memory-scoped form. G17A,
 *   whose caches are still pending (hk_cdm_caches_pending()), keeps no
 *   boundary.
 */
static size_t
hk_pack_cdm_terminate_size(enum agx_chip chip)
{
   switch (agx_chip_command_stream_generation(chip)) {
   case AGX_CS_GENERATION_AGX2:
      return AGX_CDM_STREAM_TERMINATE_LENGTH;
   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300:
      return AGX3_CDM_STREAM_TERMINATE_SIZE;
   default:
      return 0;
   }
}

size_t
hk_pack_cdm_terminate(enum agx_chip chip,
                      uint8_t out[HK_PACK_CDM_TERMINATE_MAX])
{
   static_assert(AGX_CDM_STREAM_TERMINATE_LENGTH <= HK_PACK_CDM_TERMINATE_MAX,
                 "AGX2 CDM terminate fits in HK_PACK_CDM_TERMINATE_MAX");
   static_assert(HK_PACK_CDM_TERMINATE_MAX % sizeof(uint32_t) == 0,
                 "HK_PACK_CDM_TERMINATE_MAX is 32-bit word aligned");
   static_assert(AGX3_CDM_STREAM_TERMINATE_SIZE <= HK_PACK_CDM_TERMINATE_MAX,
                 "AGX3 CDM terminate fits in HK_PACK_CDM_TERMINATE_MAX");

   switch (agx_chip_command_stream_generation(chip)) {
   case AGX_CS_GENERATION_AGX2: {
      uint32_t words[HK_PACK_CDM_TERMINATE_MAX / sizeof(uint32_t)];
      uint32_t *end = agx_cdm_terminate(words);
      const size_t len = (size_t)((uint8_t *)end - (uint8_t *)words);
      assert(len <= HK_PACK_CDM_TERMINATE_MAX);
      memcpy(out, words, len);
      return len;
   }
   case AGX_CS_GENERATION_G15:
   case AGX_CS_GENERATION_HAL200:
   case AGX_CS_GENERATION_HAL300:
      return agx3_cdm_stream_terminate_pack(chip, out)
                ? AGX3_CDM_STREAM_TERMINATE_SIZE
                : 0;
   default:
      return 0;
   }
}

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

   switch (agx_chip_command_stream_generation(chip)) {
   case AGX_CS_GENERATION_AGX2:
   case AGX_CS_GENERATION_G15:
      return HK_PACK_BOUNDARY_STRONG_FLUSH;
   case AGX_CS_GENERATION_HAL200:
      return !hk_cdm_caches_pending(dev) ? HK_PACK_BOUNDARY_STRONG_FLUSH
                                         : HK_PACK_BOUNDARY_NONE;
   default:
      return HK_PACK_BOUNDARY_NONE;
   }
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

bool
hk_compute_pair_stream_eligible(const struct hk_cs *cs)
{
   /* alias_sites are only recorded after the first dispatch
    * (hk_cdm_note_alias_site()), so on a single-dispatch stream every alias
    * site is on the trailing barrier.  When hk_pack_runs() fuses a pair it
    * holds host_alias.patch_lock, discards both original tails, and emits a
    * fresh fused dispatch + hk_cdm_visibility_flush(..., hk_cdm_alias_scope()). */
   return cs && cs->type == HK_CS_CDM && cs->compute_pair &&
          cs->stats.cmds == 1 && cs->stats.calls == 1 &&
          cs->pack_dispatches == 1 && !cs->stream_linked &&
          !cs->calls_stream && !cs->sib_indirect &&
          !cs->uses_global_atomics && !cs->imm_writes.size &&
          !cs->query_access && !cs->query_reset_pending &&
          !cs->timestamp.start.handle && !cs->timestamp.end.handle &&
          !cs->g17p_timestamp_query && !cs->gen1_native_rq &&
          !cs->as_build && cs->alias_sites <= 1 && !cs->g18_dispatch_sealed;
}

/* Conservative instruction count per query thread for libagx_reset_query_struct
 * when bounding packed query-reset command buffers. */
#define HK_PACK_QUERY_RESET_INSTRS 32u

static inline bool
hk_pack_is_query_reset_stream(const struct hk_cs *cs)
{
   return cs->query_access && !cs->query_reset_pending &&
          cs->stats.calls == 0 && cs->stats.cmds > 0 &&
          cs->stats.query_reset_dispatches == cs->stats.cmds &&
          cs->stats.query_reset_queries > 0;
}

static inline uint64_t
hk_pack_member_threads(const struct hk_cs *cs, bool query_reset)
{
   return query_reset ? cs->stats.query_reset_queries : cs->pack_threads;
}

static inline uint64_t
hk_pack_member_static_work(const struct hk_cs *cs, bool query_reset)
{
   if (!query_reset)
      return cs->pack_static_work;

   uint64_t work = 0;
   if (__builtin_mul_overflow((uint64_t)cs->stats.query_reset_queries,
                              HK_PACK_QUERY_RESET_INSTRS, &work))
      return UINT64_MAX;
   return work;
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
   const bool query_reset = hk_pack_is_query_reset_stream(cs);
   const unsigned kinds = cs->sync.kind & ~HK_SYNC_GFX_MEM;
   if (cs->type != HK_CS_CDM || !kinds ||
       ((kinds & ~(HK_SYNC_COMPUTE | HK_SYNC_XFER)) && kinds != HK_SYNC_ALL &&
        !(query_reset && kinds == HK_SYNC_QRESET)))
      return NULL;

   if (query_reset) {
      if ((kinds != HK_SYNC_QRESET && kinds != HK_SYNC_ALL) ||
          cs->pack_dispatches || cs->pack_threads || cs->pack_loops)
         return NULL;
   } else {
      /* Every launch is a direct dispatch the work estimate counts. */
      if (!cs->stats.cmds || cs->stats.cmds != cs->stats.calls ||
          cs->pack_dispatches != cs->stats.calls || !cs->pack_threads)
         return NULL;
   }

   if ((cs->uses_global_atomics && !atomics) || cs->calls_stream ||
       cs->stream_linked || cs->sib_indirect ||
       (cs->query_access && !query_reset) ||
       cs->query_reset_pending || cs->timestamp.start.handle ||
       cs->timestamp.end.handle ||
       (cs->g17p_timestamp_query && !query_reset) ||
       cs->gen1_native_rq || cs->as_build || cs->imm_writes.size)
      return NULL;

   const size_t terminate_size = hk_pack_cdm_terminate_size(dev->dev.chip);
   const uint8_t *end = (const uint8_t *)cs->current;
   if (!terminate_size ||
       end - (const uint8_t *)cs->start <= (ptrdiff_t)terminate_size)
      return NULL;
   end -= terminate_size;
   if (memcmp(end, terminate, terminate_size))
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
                 const struct hk_cs *cs, uint8_t render_kinds,
                 size_t terminate_size)
{
   memset(member, 0, sizeof(*member));
   if (!cs)
      return;

   /* A barrier records its source kinds against both subqueues
    * (hk_sync_src_dep()), so a compute-only barrier names COMPUTE work on
    * the render subqueue too. The resolver makes a wait of such a kind only
    * once the render subqueue ran work of it; neither does the planner. */
   member->cs = cs;
   member->bytes =
      (const uint8_t *)cs->current - (const uint8_t *)cs->start - terminate_size;
   member->kinds = cs->sync.kind;
   member->render_waits = cs->sync.dep[HK_SYNC_VDM].prior & render_kinds;
   for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k)
      member->render_tail[k] =
         cmd->sync.req[k][HK_SYNC_VDM].prior & render_kinds;
   member->render_ran = render_kinds != 0;
   member->query_reset = hk_pack_is_query_reset_stream(cs);
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
   const size_t terminate_size = hk_pack_cdm_terminate_size(dev->dev.chip);
   uint8_t render_kinds = hk_sync_queue_kinds(sync, HK_SYNC_VDM);
   for (unsigned i = 0; i < n; ++i) {
      const struct hk_cs *cs =
         hk_pack_run_member(dev, cmds[i], terminate, atomics);
      hk_pack_describe(&members[i], cmds[i], cs, render_kinds, terminate_size);
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
                      const struct hk_pack_limits *limits, uint64_t max_work)
{
   const struct hk_cs *cs = m->cs;
   const uint64_t m_threads = hk_pack_member_threads(cs, m->query_reset);
   const uint64_t m_work = hk_pack_member_static_work(cs, m->query_reset);
   return cs->stats.cmds <= HK_PACK_MAX_DISPATCHES &&
          m_threads <= HK_PACK_MAX_THREADS &&
          m_work <= max_work &&
          (!cs->pack_loops || m->obs_ns <= HK_PACK_MAX_OBS_NS) &&
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
   const uint64_t max_work =
      limits->max_static_work ? limits->max_static_work
                              : HK_PACK_MAX_STATIC_WORK;
   size_t resources = 0;
   uint32_t attachments = 0;
   uint8_t kinds = 0, waits = 0, tails[HK_SYNC_NUM_KINDS] = {0};
   bool unbounded_loops = false;
   bool run_query_reset = false;
   uint64_t obs_ns = 0;

   *packed_members = 0;
   *total = 0;

   for (unsigned i = 0; i <= n; ++i) {
      const struct hk_pack_member *m = i < n ? &members[i] : NULL;
      const struct hk_cs *cs = m ? m->cs : NULL;
      const bool m_unbounded = cs && cs->pack_loops && !m->obs_ns;
      const uint64_t m_obs = (cs && cs->pack_loops) ? m->obs_ns : 0;
      const uint64_t m_threads =
         cs ? hk_pack_member_threads(cs, m->query_reset) : 0;
      const uint64_t m_work =
         cs ? hk_pack_member_static_work(cs, m->query_reset) : 0;

      const bool fits = cs && run.count &&
         m->query_reset == run_query_reset &&
         (!m->query_reset || ((m->kinds ^ kinds) & ~HK_SYNC_GFX_MEM) == 0) &&
         !(m->render_ran && (m->kinds & ~kinds)) &&
         !((m->render_waits | hk_pack_tail(tails, kinds | m->kinds)) &
           ~waits) &&
         !(unbounded_loops && cs->pack_loops) &&
         !(m_unbounded && obs_ns > 0) &&
         m_obs <= HK_PACK_MAX_OBS_NS - obs_ns &&
         dispatches + cs->stats.cmds <= HK_PACK_MAX_DISPATCHES &&
         m_threads <= HK_PACK_MAX_THREADS - threads &&
         m_work <= max_work - work &&
         m->bytes <= UINT32_MAX - limits->overhead - run.bytes &&
         cs->compute_resources.size <= UINT16_MAX - resources &&
         cs->compute_attachments.count <=
            limits->max_attachments - attachments;
      if (fits) {
         run.count++;
         run.bytes += m->bytes;
         dispatches += cs->stats.cmds;
         threads += m_threads;
         work += m_work;
         resources += cs->compute_resources.size;
         attachments += cs->compute_attachments.count;
         kinds |= m->kinds;
         for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k)
            tails[k] |= m->render_tail[k];
         unbounded_loops |= m_unbounded;
         obs_ns += m_obs;
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
      unbounded_loops = false;
      run_query_reset = false;
      obs_ns = 0;

      /* Start a new run with this member when it is within the bounds. */
      if (cs && hk_pack_within_bounds(m, limits, max_work)) {
         run = (struct hk_pack_run){.first = i, .count = 1, .bytes = m->bytes};
         dispatches = cs->stats.cmds;
         threads = m_threads;
         work = m_work;
         resources = cs->compute_resources.size;
         attachments = cs->compute_attachments.count;
         kinds = m->kinds;
         waits = m->render_waits;
         memcpy(tails, m->render_tail, sizeof(tails));
         unbounded_loops = m_unbounded;
         run_query_reset = m->query_reset;
         obs_ns = m_obs;
      }
   }

   return run_count;
}

void
hk_pack_init_packet(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                    struct hk_device *dev, uint8_t *start, uint64_t addr,
                    size_t bytes, bool visible_completion)
{
   /* Preserve dynarray backing allocations across submissions and reset only
    * the CDM prefix of hk_cs and synchronization fields of hk_cmd_buffer,
    * avoiding per-submit heap traffic and zeroing unused graphics state. */
   struct util_dynarray compute_resources = cs->compute_resources;
   struct util_dynarray query_access_ranges = cs->query_access_ranges;
   util_dynarray_clear(&compute_resources);
   util_dynarray_clear(&query_access_ranges);

   memset(cs, 0, offsetof(struct hk_cs, tib));
   cs->compute_resources = compute_resources;
   cs->query_access_ranges = query_access_ranges;

   memset(&cmd->sync, 0, sizeof(cmd->sync));
   assert(!cmd->sync_pass_records.size);
   cmd->vk.base.device = &dev->vk;
   list_inithead(&cmd->control_streams);
   list_addtail(&cs->node, &cmd->control_streams);
   cs->cmd = cmd;
   cs->type = HK_CS_CDM;
   cs->start = cs->current = start;
   cs->end = start + bytes;
   cs->addr = addr;
   cs->root_size = bytes;
   cs->sync.seq = 1;
   cs->visible_completion = visible_completion;
   hk_drm_attachment_set_init(&cs->compute_attachments);
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
   packet->sync.has_gpu_query_reset |= source->sync.has_gpu_query_reset;
   *preceding_kinds |= item->sync.kind;
}

void
hk_pack_finish_sync(struct hk_cmd_buffer *packet, struct hk_cs *cs)
{
   u_foreach_bit(k, cs->sync.kind)
      packet->sync.last[k][HK_SYNC_CDM] = cs;
   packet->sync.last_any[HK_SYNC_CDM] = cs;
}

bool
hk_pack_add_bo(struct util_dynarray *bos, struct agx_bo *bo)
{
   if (!bo)
      return true;
   util_dynarray_foreach(bos, struct agx_bo *, it) {
      if (*it == bo)
         return true;
   }
   struct agx_bo **slot = util_dynarray_grow(bos, struct agx_bo *, 1);
   if (!slot)
      return false;
   *slot = bo;
   return true;
}

uint64_t
hk_pack_max_static_work_for_cores(unsigned cores)
{
   /* HK_PACK_MAX_STATIC_WORK (2^35) was calibrated for ~16-20 core GPUs
    * (~18 ms at M4 ALU rate). Scale linearly with core count between 20 and
    * 80 cores (1x at <=20, 1.6x at 32, 2x at 40, 4x at >=80) so wall-clock
    * packet budget stays bounded while larger graphs and fused variants fit. */
   const unsigned clamped = CLAMP(cores, 20u, 80u);
   return (HK_PACK_MAX_STATIC_WORK / 4u) * (uint64_t)clamped / 5u;
}

uint64_t
hk_pack_max_static_work(const struct hk_device *dev)
{
   if (!dev)
      return HK_PACK_MAX_STATIC_WORK;
   return hk_pack_max_static_work_for_cores(agx_get_num_cores(&dev->dev));
}

bool
hk_pack_account_dispatch_work_eff_limit(uint64_t threads, uint64_t instrs,
                                        bool has_loops, uint64_t eff_instrs,
                                        uint64_t max_work, uint64_t *work_out)
{
   uint64_t work = 0;
   if (__builtin_mul_overflow(threads, instrs, &work))
      work = UINT64_MAX;

   if (!has_loops) {
      *work_out = work;
      return false;
   }

   if (eff_instrs > 0) {
      uint64_t bounded = 0;
      if (!__builtin_mul_overflow(threads, eff_instrs, &bounded) &&
          bounded <= max_work) {
         *work_out = bounded;
         return false;
      }
      *work_out = UINT64_MAX;
      return true;
   }

   *work_out = work;
   return true;
}

bool
hk_pack_account_dispatch_work_eff(uint64_t threads, uint64_t instrs,
                                  bool has_loops, uint64_t eff_instrs,
                                  uint64_t *work_out)
{
   return hk_pack_account_dispatch_work_eff_limit(
      threads, instrs, has_loops, eff_instrs, HK_PACK_MAX_STATIC_WORK,
      work_out);
}

bool
hk_pack_account_dispatch_work_limit(uint64_t threads, uint64_t instrs,
                                    bool has_loops, uint64_t loop_bound,
                                    uint64_t max_work, uint64_t *work_out)
{
   if (!has_loops || loop_bound == 0)
      return hk_pack_account_dispatch_work_eff_limit(
         threads, instrs, has_loops, 0, max_work, work_out);

   uint64_t eff_instrs = 0;
   if (__builtin_mul_overflow(instrs, loop_bound, &eff_instrs)) {
      *work_out = UINT64_MAX;
      return true;
   }
   return hk_pack_account_dispatch_work_eff_limit(
      threads, instrs, true, eff_instrs, max_work, work_out);
}

bool
hk_pack_account_dispatch_work(uint64_t threads, uint64_t instrs,
                              bool has_loops, uint64_t loop_bound,
                              uint64_t *work_out)
{
   return hk_pack_account_dispatch_work_limit(
      threads, instrs, has_loops, loop_bound, HK_PACK_MAX_STATIC_WORK,
      work_out);
}

bool
hk_pack_fusion_fits_run_limit(uint64_t run_work, unsigned run_loops,
                              uint64_t replaced_work, unsigned replaced_loops,
                              uint64_t fused_work, bool fused_loops,
                              uint64_t max_work, uint64_t *new_work_out,
                              unsigned *new_loops_out)
{
   if (replaced_work > run_work || replaced_loops > run_loops)
      return false;
   const uint64_t rem_work = run_work - replaced_work;
   const unsigned rem_loops = run_loops - replaced_loops;
   const unsigned new_loops = rem_loops + (fused_loops ? 1u : 0u);
   if (new_loops > 1u)
      return false;
   uint64_t new_work = 0;
   if (__builtin_add_overflow(rem_work, fused_work, &new_work) ||
       new_work > max_work)
      return false;
   if (new_work_out)
      *new_work_out = new_work;
   if (new_loops_out)
      *new_loops_out = new_loops;
   return true;
}

bool
hk_pack_fusion_fits_run(uint64_t run_work, unsigned run_loops,
                        uint64_t replaced_work, unsigned replaced_loops,
                        uint64_t fused_work, bool fused_loops,
                        uint64_t *new_work_out, unsigned *new_loops_out)
{
   return hk_pack_fusion_fits_run_limit(run_work, run_loops, replaced_work,
                                        replaced_loops, fused_work, fused_loops,
                                        HK_PACK_MAX_STATIC_WORK, new_work_out,
                                        new_loops_out);
}

static bool
hk_pack_interval_overlaps(const struct hk_pack_interval *a,
                          const struct hk_pack_interval *b)
{
   return a->handle == b->handle && a->start < b->end && b->start < a->end;
}

static void
hk_pack_hazard_add_interval(struct hk_pack_interval *list, uint8_t *count,
                            bool *unknown, const struct hk_pack_interval *iv)
{
   if (*unknown)
      return;
   if (!iv->handle || iv->start >= iv->end) {
      *unknown = true;
      return;
   }
   for (unsigned i = 0; i < *count; ++i) {
      if (list[i].handle != iv->handle)
         continue;
      if (list[i].start <= iv->start && list[i].end >= iv->end)
         return;
      if (iv->start <= list[i].start && iv->end >= list[i].end) {
         list[i] = *iv;
         return;
      }
   }
   if (*count >= HK_PACK_HAZARD_MAX_INTERVALS) {
      *unknown = true;
      return;
   }
   list[(*count)++] = *iv;
}

void
hk_pack_hazard_reset_all(struct hk_pack_hazard_state *st)
{
   st->num_dirty_writes = 0;
   st->num_dirty_reads = 0;
   st->write_unknown = false;
   st->read_unknown = false;
}

void
hk_pack_hazard_record(struct hk_pack_hazard_state *st,
                      const struct hk_pack_footprint *fp,
                      bool strong_flushed)
{
   if (strong_flushed) {
      hk_pack_hazard_reset_all(st);
      return;
   }
   if (!fp || !fp->known) {
      st->write_unknown = true;
      st->read_unknown = true;
      return;
   }
   for (unsigned i = 0; i < fp->num_writes; ++i)
      hk_pack_hazard_add_interval(st->dirty_writes, &st->num_dirty_writes,
                                  &st->write_unknown, &fp->writes[i]);
   for (unsigned i = 0; i < fp->num_reads; ++i)
      hk_pack_hazard_add_interval(st->dirty_reads, &st->num_dirty_reads,
                                  &st->read_unknown, &fp->reads[i]);
}

enum hk_pack_boundary_action
hk_pack_hazard_decide(struct hk_pack_hazard_state *st,
                      const struct hk_pack_footprint *next_fp,
                      bool can_elide_weak)
{
   if (!next_fp || !next_fp->known || st->write_unknown) {
      hk_pack_hazard_reset_all(st);
      return HK_PACK_BOUNDARY_STRONG;
   }

   for (unsigned w = 0; w < st->num_dirty_writes; ++w) {
      for (unsigned r = 0; r < next_fp->num_reads; ++r) {
         if (hk_pack_interval_overlaps(&st->dirty_writes[w], &next_fp->reads[r])) {
            hk_pack_hazard_reset_all(st);
            return HK_PACK_BOUNDARY_STRONG;
         }
      }
      for (unsigned nw = 0; nw < next_fp->num_writes; ++nw) {
         if (hk_pack_interval_overlaps(&st->dirty_writes[w], &next_fp->writes[nw])) {
            hk_pack_hazard_reset_all(st);
            return HK_PACK_BOUNDARY_STRONG;
         }
      }
   }

   bool war = false;
   if (!st->read_unknown) {
      for (unsigned r = 0; r < st->num_dirty_reads && !war; ++r) {
         for (unsigned nw = 0; nw < next_fp->num_writes; ++nw) {
            if (hk_pack_interval_overlaps(&st->dirty_reads[r], &next_fp->writes[nw])) {
               war = true;
               break;
            }
         }
      }
   }

   if (!can_elide_weak || st->read_unknown || war) {
      st->num_dirty_reads = 0;
      st->read_unknown = false;
      return HK_PACK_BOUNDARY_KEEP_WEAK;
   }

   return HK_PACK_BOUNDARY_ELIDE_BARRIER;
}

void
hk_pack_liveness_reset(struct hk_pack_liveness_state *st)
{
   st->num_dead = 0;
}

bool
hk_pack_liveness_covers(const struct hk_pack_liveness_state *st,
                        const struct hk_pack_interval *iv)
{
   if (!iv || !iv->handle || iv->start >= iv->end)
      return false;
   for (unsigned i = 0; i < st->num_dead; ++i) {
      if (st->dead[i].handle == iv->handle &&
          st->dead[i].start <= iv->start &&
          st->dead[i].end >= iv->end)
         return true;
   }
   return false;
}

void
hk_pack_liveness_add_overwrite(struct hk_pack_liveness_state *st,
                               const struct hk_pack_interval *iv)
{
   if (!iv || !iv->handle || iv->start >= iv->end)
      return;

   struct hk_pack_interval merged = *iv;
   for (unsigned i = 0; i < st->num_dead;) {
      if (st->dead[i].handle == merged.handle &&
          st->dead[i].start <= merged.end &&
          merged.start <= st->dead[i].end) {
         if (st->dead[i].start < merged.start)
            merged.start = st->dead[i].start;
         if (st->dead[i].end > merged.end)
            merged.end = st->dead[i].end;
         st->dead[i] = st->dead[--st->num_dead];
      } else {
         ++i;
      }
   }

   /* Dropping an interval on capacity overflow is sound (under-approximates
    * the dead set, never elides a live write). */
   if (st->num_dead < HK_PACK_LIVENESS_MAX_INTERVALS)
      st->dead[st->num_dead++] = merged;
}

void
hk_pack_liveness_kill_read(struct hk_pack_liveness_state *st,
                           const struct hk_pack_interval *iv)
{
   if (!iv || !iv->handle || iv->start >= iv->end) {
      hk_pack_liveness_reset(st);
      return;
   }

   for (unsigned i = 0; i < st->num_dead;) {
      if (!hk_pack_interval_overlaps(&st->dead[i], iv)) {
         ++i;
         continue;
      }
      struct hk_pack_interval cur = st->dead[i];
      if (iv->start <= cur.start && iv->end >= cur.end) {
         st->dead[i] = st->dead[--st->num_dead];
      } else if (iv->start <= cur.start) {
         st->dead[i].start = iv->end;
         ++i;
      } else if (iv->end >= cur.end) {
         st->dead[i].end = iv->start;
         ++i;
      } else {
         st->dead[i].end = iv->start;
         if (st->num_dead < HK_PACK_LIVENESS_MAX_INTERVALS) {
            st->dead[st->num_dead++] = (struct hk_pack_interval){
               .start = iv->end,
               .end = cur.end,
               .handle = cur.handle,
            };
         }
         ++i;
      }
   }
}

void
hk_pack_liveness_step_backward(struct hk_pack_liveness_state *st,
                               const struct hk_pack_footprint *fp)
{
   if (!fp || !fp->known) {
      hk_pack_liveness_reset(st);
      return;
   }
   for (unsigned i = 0; i < fp->num_writes; ++i)
      hk_pack_liveness_add_overwrite(st, &fp->writes[i]);
   for (unsigned i = 0; i < fp->num_reads; ++i)
      hk_pack_liveness_kill_read(st, &fp->reads[i]);
}

void
hk_pack_obs_retire_entries(struct hk_pack_obs_entry *entries,
                           unsigned slot_count, uint32_t *obs_pending,
                           uint64_t *obs_epoch, uint64_t signalled,
                           uint64_t sig_point, uint64_t sig_ns,
                           uint64_t now_ns)
{
   for (unsigned i = 0; i < slot_count; ++i) {
      struct hk_pack_obs_entry *e = &entries[i];
      if (!e->pending_pt || e->pending_pt > signalled)
         continue;

      const uint64_t done_ns =
         (e->pending_pt <= sig_point && sig_ns > e->submit_ns &&
          sig_ns <= now_ns)
            ? sig_ns
            : now_ns;
      const uint64_t elapsed_ns =
         done_ns > e->submit_ns ? done_ns - e->submit_ns : 1000ull;
      e->pending_pt = 0;
      p_atomic_dec(obs_pending);

      if (!e->in_flight_packed) {
         if (elapsed_ns <= HK_PACK_OBS_ADMIT_NS && e->loop_count >= 2) {
            const uint64_t per =
               MAX2(1000ull, (elapsed_ns + e->loop_count - 1u) / e->loop_count);
            if (per != e->obs_per_loop_ns) {
               e->obs_per_loop_ns = per;
               (*obs_epoch)++;
            }
         } else if (e->obs_per_loop_ns != 0) {
            e->obs_per_loop_ns = 0;
            (*obs_epoch)++;
         }
      } else {
         if (elapsed_ns > HK_PACK_OBS_ADMIT_NS) {
            if (e->obs_per_loop_ns != 0) {
               e->obs_per_loop_ns = 0;
               (*obs_epoch)++;
            }
         } else if (elapsed_ns > HK_PACK_OBS_BACKOFF_NS && e->loop_count >= 2) {
            const uint64_t per =
               MAX2(1000ull, (elapsed_ns + e->loop_count - 1u) / e->loop_count);
            if (per > e->obs_per_loop_ns) {
               e->obs_per_loop_ns = per;
               (*obs_epoch)++;
            }
         }
      }
   }
}

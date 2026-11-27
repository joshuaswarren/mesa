/* SPDX-License-Identifier: MIT
 *
 * Timestamp recording through the real hk_sync: which commands carry stamps,
 * which stamps are copies of earlier ones and when the copies are launched,
 * and which later streams wait for timestamp-only commands. Streams are
 * created through the production helpers on CPU memory; the copy launch
 * itself (hk_record_stamp_copies()) is logged instead of emitted. No GPU or
 * DRM calls.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/ralloc.h"
#include "util/u_math.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_queue.h"

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

static void *
test_alloc(size_t size)
{
   void *ptr = rzalloc_size(test_context, size);
   if (!ptr) {
      fputs("Failed to allocate timestamp test fixture\n", stderr);
      exit(EXIT_FAILURE);
   }
   return ptr;
}

/* Copy launches, as hk_record_stamp_copies() received them. */
#define MAX_RECORDS 16
#define MAX_RECORD_COPIES 64
static struct {
   struct hk_cs *cs;
   unsigned count;
   struct hk_stamp_copy copies[MAX_RECORD_COPIES];
} records[MAX_RECORDS];
static unsigned record_count;

void
hk_record_stamp_copies(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       const struct hk_stamp_copy *copies, unsigned count)
{
   (void)cmd;
   if (record_count >= MAX_RECORDS || count > MAX_RECORD_COPIES ||
       cs->type != HK_CS_CDM || !count)
      abort();
   records[record_count].cs = cs;
   records[record_count].count = count;
   memcpy(records[record_count].copies, copies, count * sizeof(*copies));
   record_count++;
   /* One launch, as the real helper records. */
   cs->stats.cmds++;
}

/* Streams live on CPU memory: the root of a control stream, and pool memory
 * for the native AGX3 carrier's scratch store. */
struct agx_ptr
hk_pool_cs_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   (void)cmd;
   (void)alignment;
   void *p = test_alloc(size);
   return (struct agx_ptr){.cpu = p, .gpu = (uintptr_t)p};
}

struct agx_ptr
hk_pool_alloc_from_slow(struct hk_cmd_buffer *cmd, struct hk_uploader *uploader,
                        uint32_t size, uint32_t alignment,
                        enum hk_cmd_bo_type type)
{
   (void)cmd;
   (void)uploader;
   (void)alignment;
   (void)type;
   void *p = test_alloc(size);
   return (struct agx_ptr){.cpu = p, .gpu = (uintptr_t)p};
}

void
hk_drm_attachment_set_init(struct hk_drm_attachment_set *set)
{
   memset(set, 0, sizeof(*set));
}

/* The native AGX3 carrier's executable store is a launch in the target. */
void
hk_g17p_meta_fill_addr(struct hk_cmd_buffer *cmd, struct hk_cs **target,
                       uint64_t address, uint64_t size, uint32_t value)
{
   (void)cmd;
   (void)address;
   (void)size;
   (void)value;
   (*target)->stats.cmds++;
}

/* These devices never map memory at a second address. */
void
hk_cdm_end_stream_flush(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   (void)cs;
}

/* Fixtures never queue resets, immediate writes, deferred clears or other
 * helper launches. Reaching one is a bug. */
void
hk_flush_query_resets(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   if (cs->query_reset_pending)
      abort();
}

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

bool
hk_defer_oq_available(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   (void)cs;
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
   (void)cmd;
   (void)grid;
   (void)barrier;
   (void)idx;
   (void)data;
   (void)data_size;
   abort();
}

static struct hk_queue *
new_queue(bool exact)
{
   struct hk_queue *q = test_alloc(sizeof(*q));
   q->sync.exact_prior = exact;
   return q;
}

static struct hk_cmd_buffer *
new_cmd(enum agx_chip chip, bool exact)
{
   struct hk_cmd_buffer *cmd = test_alloc(sizeof(*cmd));
   struct hk_device *dev = test_alloc(sizeof(*dev));
   dev->dev.chip = chip;
   dev->dev.params.features =
      exact ? DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS : 0;
   cmd->vk.base.device = &dev->vk;
   list_inithead(&cmd->control_streams);
   util_dynarray_init(&cmd->sync_pass_records, NULL);
   return cmd;
}

static bool
submit(struct hk_queue *q, struct hk_cmd_buffer *const *cmds, unsigned n)
{
   return hk_sync_resolve_submit(q, cmds, n, false, false, 0, NULL);
}

static struct agx_timestamp_req
query(uint32_t q)
{
   return (struct agx_timestamp_req){
      .addr = 0x10000 + 8 * q, .handle = 1, .offset_B = 8 * q};
}

static void
stamp(struct hk_cmd_buffer *cmd, VkPipelineStageFlags2 stage, uint32_t q)
{
   hk_sync_write_timestamp(cmd, stage, query(q));
}

static void
stamp_all(struct hk_cmd_buffer *cmd, uint32_t q)
{
   stamp(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, q);
}

/* A vkCmdDispatch: work in the general compute stream. */
static struct hk_cs *
dispatch(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, true);
   cs->stats.cmds++;
   return cs;
}

/* A render pass instance with one draw, still open. */
static struct hk_cs *
begin_render(struct hk_cmd_buffer *cmd)
{
   hk_sync_begin_rendering(cmd, false);
   struct hk_cs *r = test_alloc(sizeof(*r));
   r->type = HK_CS_VDM;
   list_addtail(&r->node, &cmd->control_streams);
   cmd->current_cs.gfx = r;
   hk_sync_stream_created(cmd, r, &cmd->current_cs.gfx, false);
   r->stats.cmds++;
   return r;
}

static void
end_render(struct hk_cmd_buffer *cmd)
{
   cmd->current_cs.gfx = NULL;
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);
   hk_sync_end_rendering(cmd, false);
}

static struct hk_cs *
render(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *r = begin_render(cmd);
   end_render(cmd);
   return r;
}

static bool
same_req(struct agx_timestamp_req a, struct agx_timestamp_req b)
{
   return a.addr == b.addr && a.handle == b.handle && a.offset_B == b.offset_B;
}

static bool
is_end(const struct hk_cs *cs, uint32_t q)
{
   return same_req(cs->timestamp.end, query(q));
}

static bool
copy_is(const struct hk_stamp_copy *c, uint32_t dst, struct agx_timestamp_req src)
{
   return same_req(c->dst, query(dst)) && same_req(c->src, src);
}

static unsigned
stamped_streams(struct hk_cmd_buffer *cmd)
{
   unsigned n = 0;
   list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node)
      n += cs->timestamp.start.handle || cs->timestamp.end.handle;
   return n;
}

static struct hk_cs *
last_cs(struct hk_cmd_buffer *cmd)
{
   return list_last_entry(&cmd->control_streams, struct hk_cs, node);
}

/* M5 parts, whose end stamps take every stamp, and G17P, whose carriers also
 * need an executable store. */
static const enum agx_chip chips[] = {
   AGX_CHIP_G17S, AGX_CHIP_G17C, AGX_CHIP_G17G, AGX_CHIP_G17P};

/* Whether a stamp on a compute command is ordered through the compute queue
 * alone, as stamp_wait() decides: exact-prior barriers on a native AGX3 or
 * M5 part. */
static bool
cdm_stamp_queue_ordered(enum agx_chip chip, bool exact)
{
   return agx_chip_native_or_m5(chip) && exact;
}

/* A timestamp-only command is no producer for application work: a barrier,
 * even ALL_COMMANDS, orders later renders and dispatches after the work
 * before it, not after the carrier. Within the command buffer, across
 * command buffers of a submission and across submissions. */
static void
test_carriers_order_no_application_work(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned exact = 0; exact < 2; ++exact) {
         struct hk_cmd_buffer *cmd = new_cmd(chips[c], exact);
         /* A compute-stage stamp rides on the render; an ALL_COMMANDS one
          * after it needs more than the render waited for, so it gets a
          * carrier of its own. */
         struct hk_cs *r0 = render(cmd);
         stamp(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0);
         T("compute-stage stamp ends the render", is_end(r0, 0));
         stamp_all(cmd, 1);
         struct hk_cs *k = last_cs(cmd);
         T("wider stamp gets a carrier",
           k != r0 && k->type == HK_CS_CDM && is_end(k, 1) &&
           !cmd->sync.stamp_copy_count);
         T("carrier is a timestamp stream only",
           (k->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP);
         T("carrier follows the render it is stamped after",
           k->sync.dep[HK_SYNC_VDM].cut == r0);
         T("carrier is no producer of any application class",
           !cmd->sync.last[util_logbase2(HK_SYNC_GFX)][HK_SYNC_CDM] &&
           !cmd->sync.last[util_logbase2(HK_SYNC_COMPUTE)][HK_SYNC_CDM] &&
           !cmd->sync.last[util_logbase2(HK_SYNC_XFER)][HK_SYNC_CDM] &&
           cmd->sync.last[util_logbase2(HK_SYNC_STAMP)][HK_SYNC_CDM] == k);

         hk_sync_barrier(cmd, HK_SYNC_ALL, HK_SYNC_ALL);
         struct hk_cs *r1 = render(cmd);
         T("render after an ALL_COMMANDS barrier skips the carrier",
           r1->sync.dep[HK_SYNC_CDM].cut == NULL &&
           r1->sync.dep[HK_SYNC_VDM].cut == r0);
         struct hk_cs *d = dispatch(cmd);
         T("dispatch after the barrier skips the carrier",
           d->sync.dep[HK_SYNC_CDM].cut == NULL &&
           d->sync.dep[HK_SYNC_VDM].cut == r0);

         struct hk_queue *q = new_queue(exact);
         struct hk_cmd_buffer *next = new_cmd(chips[c], exact);
         hk_sync_barrier(next, HK_SYNC_ALL, HK_SYNC_ALL);
         struct hk_cs *e = dispatch(next);
         struct hk_cmd_buffer *both[] = {cmd, next};
         T("submission with a carrier resolves", submit(q, both, 2));
         T("later command buffer's dispatch does not wait for the carrier",
           e->sync.barrier[HK_SYNC_CDM] == d->sync.index &&
           d->sync.index > k->sync.index);

         /* Across submissions, the carrier alone on the compute subqueue. */
         struct hk_queue *q2 = new_queue(exact);
         struct hk_cmd_buffer *first = new_cmd(chips[c], exact);
         struct hk_cs *fr = render(first);
         stamp(first, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0);
         stamp_all(first, 1);
         T("carrier-only compute submission resolves",
           submit(q2, &first, 1) && is_end(last_cs(first), 1) &&
           last_cs(first) != fr);
         struct hk_cmd_buffer *second = new_cmd(chips[c], exact);
         hk_sync_barrier(second, HK_SYNC_ALL, HK_SYNC_ALL);
         struct hk_cs *later = dispatch(second);
         T("later submission resolves", submit(q2, &second, 1));
         T("later submission's dispatch does not wait for the carrier",
           later->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE &&
           later->sync.prior[HK_SYNC_CDM] == 0);
      }
   }
}

/* Query commands and later timestamps still follow every carrier. */
static void
test_query_commands_follow_carriers(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned exact = 0; exact < 2; ++exact) {
         struct hk_cmd_buffer *cmd = new_cmd(chips[c], exact);
         render(cmd);
         stamp(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0);
         stamp_all(cmd, 1);
         struct hk_cs *k = last_cs(cmd);

         /* hk_query_command_cs(): the query command's scope. */
         struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
         hk_sync_src_dep(cmd, HK_SYNC_QUERY_SCOPE, NULL, dep);
         struct hk_cs *copy = hk_sync_get_cs_after(cmd, dep);
         T("query command in this command buffer follows the carrier",
           copy != k && copy->sync.dep[HK_SYNC_CDM].cut == k);

         /* A query command in a later command buffer, after one whose only
          * compute command is a carrier: in one submission, then in the
          * next. */
         for (unsigned split = 0; split < 2; ++split) {
            struct hk_cmd_buffer *first = new_cmd(chips[c], exact);
            render(first);
            stamp(first, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0);
            stamp_all(first, 1);
            struct hk_cs *fk = last_cs(first);
            struct hk_cmd_buffer *next = new_cmd(chips[c], exact);
            hk_sync_src_dep(next, HK_SYNC_QUERY_SCOPE, NULL, dep);
            struct hk_cs *reset = hk_sync_get_cs_after(next, dep);
            reset->stats.cmds++;
            struct hk_queue *q = new_queue(exact);
            struct hk_cmd_buffer *both[] = {first, next};
            if (split) {
               T("carrier submission resolves", submit(q, &first, 1));
               T("query submission resolves", submit(q, &next, 1));
               T("query command in a later submission follows the carrier",
                 exact ? reset->sync.barrier[HK_SYNC_CDM] ==
                               DRM_ASAHI_BARRIER_NONE &&
                            reset->sync.prior[HK_SYNC_CDM] == fk->sync.index
                       : reset->sync.barrier[HK_SYNC_CDM] == 0);
            } else {
               T("query submission resolves", submit(q, both, 2));
               T("query command in a later command buffer follows the carrier",
                 fk->type == HK_CS_CDM &&
                 reset->sync.barrier[HK_SYNC_CDM] == fk->sync.index);
            }
         }

         /* A carrier after a query reset follows the reset. */
         struct hk_cmd_buffer *rc = new_cmd(chips[c], exact);
         hk_sync_src_dep(rc, HK_SYNC_QUERY_SCOPE, NULL, dep);
         struct hk_cs *reset = hk_sync_get_cs_after(rc, dep);
         reset->stats.cmds++;
         hk_sync_after_stream(rc, reset, HK_SYNC_ALL);
         hk_cmd_buffer_end_compute(rc);
         struct hk_cs *r2 = render(rc);
         stamp(rc, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 2);
         stamp_all(rc, 3);
         struct hk_cs *k2 = last_cs(rc);
         T("carrier after a reset follows the reset",
           is_end(r2, 2) && k2 != r2 && is_end(k2, 3) &&
           k2->sync.dep[HK_SYNC_CDM].cut == reset);

         /* A later ALL_COMMANDS stamp follows the carrier. */
         struct hk_cmd_buffer *mono = new_cmd(chips[c], exact);
         render(mono);
         stamp(mono, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0);
         stamp_all(mono, 1);
         struct hk_cs *mk = last_cs(mono);
         struct hk_cs *r = render(mono);
         stamp_all(mono, 2);
         T("later stamp's render follows the earlier carrier",
           is_end(r, 2) && r->sync.dep[HK_SYNC_CDM].cut == mk);
         T("no copies pending", !mono->sync.stamp_copy_count);
      }
   }
}

/* Back-to-back stamps with nothing between them copy the first one. */
static void
test_consecutive_stamps_coalesce(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned exact = 0; exact < 2; ++exact) {
         record_count = 0;
         struct hk_cmd_buffer *cmd = new_cmd(chips[c], exact);
         struct hk_cs *d = dispatch(cmd);
         for (unsigned i = 0; i < 5; ++i)
            stamp_all(cmd, i);
         T("first stamp ends the dispatch", is_end(d, 0));

         /* Both exact-prior queue-ordered compute stamps and legacy prefix
          * waits coalesce all five consecutive stamps onto the dispatch with
          * no extra carrier stream. */
         T("a run of stamps records no extra carrier",
           list_length(&cmd->control_streams) == 1 &&
           stamped_streams(cmd) == 1);
         T("the rest are pending copies of the dispatch",
           cmd->sync.stamp_copy_count == 4 &&
           cmd->sync.stamp_copy_after[HK_SYNC_CDM] == d &&
           !cmd->sync.stamp_copy_after[HK_SYNC_VDM] && !record_count);
         bool sources = true;
         for (unsigned i = 0; i < 4; ++i) {
            sources &= copy_is(&cmd->sync.stamp_copies[i], 1 + i,
                               d->timestamp.end);
         }
         T("each copy reads the stamp it repeats", sources);

         hk_sync_flush_stamp_copies(cmd);
         struct hk_cs *f = last_cs(cmd);
         T("flush launches every copy once",
           record_count == 1 && records[0].cs == f &&
           records[0].count == 4 && !cmd->sync.stamp_copy_count);
         T("flush stream follows the stamp it reads",
           f->sync.dep[HK_SYNC_CDM].cut == d &&
           (f->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP);
         T("flush stream is ended", !cmd->current_cs.cs);

         struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
         hk_sync_src_dep(cmd, HK_SYNC_QUERY_SCOPE, NULL, dep);
         T("a query command follows the copies",
           dep[HK_SYNC_CDM].cut == f);
         T("coalesced recording resolves", submit(new_queue(exact), &cmd, 1));

         /* A run after a render copies the render's end stamp. */
         record_count = 0;
         struct hk_cmd_buffer *rcmd = new_cmd(chips[c], exact);
         struct hk_cs *r = render(rcmd);
         stamp_all(rcmd, 0);
         stamp_all(rcmd, 1);
         stamp(rcmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 2);
         T("render run is one stamped command",
           is_end(r, 0) && list_length(&rcmd->control_streams) == 1 &&
           rcmd->sync.stamp_copy_count == 2 &&
           rcmd->sync.stamp_copy_after[HK_SYNC_VDM] == r &&
           copy_is(&rcmd->sync.stamp_copies[0], 1, query(0)) &&
           copy_is(&rcmd->sync.stamp_copies[1], 2, query(0)));
         hk_sync_flush_stamp_copies(rcmd);
         T("render copies follow the render",
           record_count == 1 && records[0].count == 2 &&
           records[0].cs->sync.dep[HK_SYNC_VDM].cut == r);
      }
   }
}

/* Work or a wider scope between two stamps keeps them distinct. */
static void
test_stamps_with_work_stay_distinct(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned exact = 0; exact < 2; ++exact) {
         struct hk_cmd_buffer *cmd = new_cmd(chips[c], exact);
         struct hk_cs *d0 = dispatch(cmd);
         stamp_all(cmd, 0);
         struct hk_cs *d1 = dispatch(cmd);
         stamp_all(cmd, 1);
         struct hk_cs *r = render(cmd);
         stamp_all(cmd, 2);
         T("each stamp after work ends that work",
           d0 != d1 && is_end(d0, 0) && is_end(d1, 1) && is_end(r, 2) &&
           !cmd->sync.stamp_copy_count);
         T("later stamped work follows the earlier stamp",
           d1->sync.dep[HK_SYNC_CDM].cut == d0 ||
           cdm_stamp_queue_ordered(chips[c], exact));

         /* Work in a stream recorded before the latest one: the open
          * compute stream stays open across a render that does not wait
          * for it, so the next stamp must not copy the render's. */
         struct hk_cmd_buffer *open = new_cmd(chips[c], exact);
         struct hk_cs *cs = dispatch(open);
         struct hk_cs *r1 = render(open);
         T("compute stays open across the independent render",
           open->current_cs.cs == cs);
         stamp_all(open, 0);
         T("stamp ends the render, waiting for the open compute",
           is_end(r1, 0) && r1->sync.dep[HK_SYNC_CDM].cut == cs);

         /* A barrier between them that orders the second stamp after more
          * than the first waited for. */
         struct hk_cmd_buffer *bar = new_cmd(chips[c], exact);
         struct hk_cs *r2 = render(bar);
         stamp(bar, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0);
         struct hk_cs *dd = dispatch(bar);
         stamp(bar, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 1);
         hk_sync_barrier(bar, HK_SYNC_GFX, HK_SYNC_ALL);
         stamp(bar, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 2);
         struct hk_cs *k = last_cs(bar);
         T("stamp ordered after more work gets a command of its own",
           is_end(r2, 0) && is_end(dd, 1) && k != dd && is_end(k, 2) &&
           k->sync.dep[HK_SYNC_VDM].cut == r2 &&
           !bar->sync.stamp_copy_count);
         T("distinct stamps resolve", submit(new_queue(exact), &bar, 1));
      }
   }
}

/* Copies launch together: one, several, and a full batch, which the next
 * carrier takes when it follows the stamps they read. */
static void
test_batched_copies(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      record_count = 0;
      struct hk_cmd_buffer *cmd = new_cmd(chips[c], false);
      render(cmd);
      stamp(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0);
      stamp_all(cmd, 1);
      struct hk_cs *k = last_cs(cmd);
      stamp_all(cmd, 2);
      T("one pending copy", cmd->sync.stamp_copy_count == 1 &&
                            copy_is(&cmd->sync.stamp_copies[0], 2, query(1)));
      hk_sync_flush_stamp_copies(cmd);
      T("one copy, one launch",
        record_count == 1 && records[0].count == 1 &&
        records[0].cs->sync.dep[HK_SYNC_CDM].cut == k);

      /* A full batch, then a stamp with no room. */
      record_count = 0;
      struct hk_cmd_buffer *full = new_cmd(chips[c], false);
      render(full);
      stamp(full, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0);
      stamp_all(full, 1);
      struct hk_cs *k1 = last_cs(full);
      for (unsigned i = 0; i < HK_SYNC_STAMP_COPIES; ++i)
         stamp_all(full, 2 + i);
      T("a full batch is pending, nothing launched",
        full->sync.stamp_copy_count == HK_SYNC_STAMP_COPIES &&
        !record_count && last_cs(full) == k1);
      stamp_all(full, 2 + HK_SYNC_STAMP_COPIES);
      struct hk_cs *k2 = last_cs(full);
      T("the next stamp's carrier launches the batch",
        k2 != k1 && is_end(k2, 2 + HK_SYNC_STAMP_COPIES) &&
        record_count == 1 && records[0].cs == k2 &&
        records[0].count == HK_SYNC_STAMP_COPIES &&
        k2->sync.dep[HK_SYNC_CDM].cut == k1 && !full->sync.stamp_copy_count);
      bool all = true;
      for (unsigned i = 0; i < HK_SYNC_STAMP_COPIES; ++i)
         all &= copy_is(&records[0].copies[i], 2 + i, query(1));
      T("batch keeps every pair in order", all);
      T("a launch keeps the carrier a timestamp stream",
        (k2->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP);
      stamp_all(full, 3 + HK_SYNC_STAMP_COPIES);
      T("the run continues from the new carrier",
        full->sync.stamp_copy_count == 1 &&
        copy_is(&full->sync.stamp_copies[0], 3 + HK_SYNC_STAMP_COPIES,
                query(2 + HK_SYNC_STAMP_COPIES)));

      /* Copies of two runs, from different commands, in one launch. */
      record_count = 0;
      struct hk_cmd_buffer *two = new_cmd(chips[c], false);
      struct hk_cs *d0 = dispatch(two);
      stamp_all(two, 0);
      stamp_all(two, 1);
      struct hk_cs *d1 = dispatch(two);
      stamp_all(two, 2);
      stamp_all(two, 3);
      T("two runs pend two copies",
        two->sync.stamp_copy_count == 2 &&
        copy_is(&two->sync.stamp_copies[0], 1, d0->timestamp.end) &&
        copy_is(&two->sync.stamp_copies[1], 3, d1->timestamp.end) &&
        two->sync.stamp_copy_after[HK_SYNC_CDM] == d1);
      hk_sync_flush_stamp_copies(two);
      T("two runs, one launch after both",
        record_count == 1 && records[0].count == 2 &&
        records[0].cs->sync.dep[HK_SYNC_CDM].cut == d1);
   }
}

/* Later stamps inside one render pass instance copy the render's end stamp.
 * The copies pend with the command buffer's others and launch together in a
 * timestamp-only command after the render; a full batch launches first. */
static void
test_in_pass_copies(void)
{
   const unsigned counts[] = {1, HK_SYNC_STAMP_COPIES + 8};
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned ci = 0; ci < ARRAY_SIZE(counts); ++ci) {
         const unsigned n = counts[ci];
         record_count = 0;
         struct hk_cmd_buffer *cmd = new_cmd(chips[c], false);
         struct hk_cs *r = begin_render(cmd);
         stamp_all(cmd, 0);
         for (unsigned i = 1; i <= n; ++i)
            stamp_all(cmd, i);
         T("first in-pass stamp ends the render", is_end(r, 0));
         T("no post-graphics stream for in-pass copies",
           !cmd->current_cs.post_gfx);

         /* A batch that filled inside the pass launched there, after the
          * render. */
         const unsigned early = n > HK_SYNC_STAMP_COPIES;
         const unsigned rest = n - early * HK_SYNC_STAMP_COPIES;
         bool full = record_count == early;
         if (early) {
            const struct hk_cs *f = records[0].cs;
            full &= records[0].count == HK_SYNC_STAMP_COPIES &&
                    f->sync.dep[HK_SYNC_VDM].cut == r &&
                    (f->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP;
            for (unsigned i = 0; full && i < HK_SYNC_STAMP_COPIES; ++i)
               full &= copy_is(&records[0].copies[i], 1 + i, query(0));
         }
         T("a full in-pass batch launches after the render", full);
         T("later in-pass stamps pend as copies of the render's",
           cmd->sync.stamp_copy_count == rest &&
           cmd->sync.stamp_copy_after[HK_SYNC_VDM] == r &&
           copy_is(&cmd->sync.stamp_copies[rest - 1], n, query(0)));

         end_render(cmd);
         T("ending the render launches nothing", record_count == early);
         hk_sync_flush_stamp_copies(cmd);
         const struct hk_cs *f = last_cs(cmd);
         bool all = record_count == early + 1 && records[early].cs == f &&
                    records[early].count == rest;
         for (unsigned i = 0; all && i < rest; ++i) {
            all &= copy_is(&records[early].copies[i],
                           1 + early * HK_SYNC_STAMP_COPIES + i, query(0));
         }
         T("the rest launch once, in a timestamp command after the render",
           all && f->sync.dep[HK_SYNC_VDM].cut == r &&
           (f->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP);
         T("in-pass copies resolve", submit(new_queue(false), &cmd, 1));
      }
   }
}

/* vkCmdCopyQueryPoolResults, as hk_CmdCopyQueryPoolResults() records it:
 * the pending stamps and copies first, then a stream of every kind after
 * every earlier stream (hk_query_command_cs()). */
static struct hk_cs *
copy_query_results(struct hk_cmd_buffer *cmd)
{
   cmd->sync.kind_override = HK_SYNC_ALL;
   hk_sync_flush_pending_timestamp(cmd);
   hk_sync_flush_stamp_copies(cmd);
   hk_sync_end_cs_if_timestamped(cmd);
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   hk_sync_src_dep(cmd, HK_SYNC_QUERY_SCOPE, cmd->current_cs.cs, dep);
   struct hk_cs *cs = hk_sync_get_cs_after(cmd, dep);
   cs->stats.cmds++;
   cmd->sync.kind_override = 0;
   return cs;
}

/* A render whose pass instance writes several timestamps: its end stamp and
 * copies of it. */
static struct hk_cs *
profiled_render(struct hk_cmd_buffer *cmd, uint32_t q, unsigned stamps)
{
   struct hk_cs *r = begin_render(cmd);
   for (unsigned i = 0; i < stamps; ++i)
      stamp(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, q + i);
   end_render(cmd);
   return r;
}

/* Timestamp copies write query memory only. A render or dispatch recorded
 * after them waits for the work its barriers name, never for the copies,
 * whatever stages the barrier names. */
static void
test_copies_order_no_application_work(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned exact = 0; exact < 2; ++exact) {
         record_count = 0;
         struct hk_cmd_buffer *cmd = new_cmd(chips[c], exact);

         /* Render, stamps inside and after it, then an unrelated render
          * behind a barrier from every graphics stage, then another behind
          * an ALL_COMMANDS barrier. */
         struct hk_cs *r0 = profiled_render(cmd, 0, 3);
         stamp(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 3);
         T("stamps of a profiled render pend as copies",
           is_end(r0, 0) && cmd->sync.stamp_copy_count == 3 &&
           list_length(&cmd->control_streams) == 1 && !record_count);
         hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_ALL);
         struct hk_cs *r1 = profiled_render(cmd, 4, 2);
         T("render after a graphics barrier waits for the render alone",
           r1->sync.dep[HK_SYNC_VDM].cut == r0 &&
           !r1->sync.dep[HK_SYNC_CDM].cut &&
           !r1->sync.dep_ta[HK_SYNC_CDM].cut);
         hk_sync_barrier(cmd, HK_SYNC_ALL, HK_SYNC_ALL);
         struct hk_cs *r2 = render(cmd);
         T("render after an ALL_COMMANDS barrier skips the copies",
           r2->sync.dep[HK_SYNC_VDM].cut == r1 &&
           !r2->sync.dep[HK_SYNC_CDM].cut);
         hk_sync_barrier(cmd, HK_SYNC_ALL, HK_SYNC_ALL);
         struct hk_cs *d = dispatch(cmd);
         T("dispatch after an ALL_COMMANDS barrier skips the copies",
           d->sync.dep[HK_SYNC_VDM].cut == r2 &&
           !d->sync.dep[HK_SYNC_CDM].cut);
         T("only CDM work so far is application work",
           list_length(&cmd->control_streams) == 4 && !record_count &&
           cmd->sync.stamp_copy_count == 4);

         /* The end of the command buffer: one launch in a command of its
          * own, after both renders, never in the open dispatch's stream. */
         hk_sync_flush_stamp_copies(cmd);
         struct hk_cs *f = last_cs(cmd);
         T("copies launch once, in their own timestamp command",
           record_count == 1 && records[0].count == 4 && records[0].cs == f &&
           f != d && f->sync.dep[HK_SYNC_VDM].cut == r2 &&
           (f->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP);
         T("profiled renders resolve", submit(new_queue(exact), &cmd, 1));

         /* An open dispatch recorded after the profiled render, which it
          * does not wait for, keeps its own requirements when the copies
          * launch. */
         record_count = 0;
         struct hk_cmd_buffer *open = new_cmd(chips[c], exact);
         struct hk_cs *pr = profiled_render(open, 0, 2);
         struct hk_cs *od = dispatch(open);
         T("the dispatch is independent of the profiled render",
           open->current_cs.cs == od && !od->sync.dep[HK_SYNC_VDM].cut);
         hk_sync_flush_stamp_copies(open);
         struct hk_cs *of = last_cs(open);
         T("copies do not ride on the open dispatch",
           record_count == 1 && records[0].cs == of && of != od &&
           !od->sync.dep[HK_SYNC_VDM].cut &&
           of->sync.dep[HK_SYNC_VDM].cut == pr &&
           (of->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP);
         T("open dispatch and copies resolve",
           submit(new_queue(exact), &open, 1));
      }
   }
}

/* What does read the results waits for the copies: a query result copy, the
 * work a transfer barrier orders after that copy, and a host-visibility
 * flush after it. */
static void
test_copy_consumers_wait(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned exact = 0; exact < 2; ++exact) {
         record_count = 0;
         struct hk_cmd_buffer *cmd = new_cmd(chips[c], exact);
         struct hk_cs *r0 = profiled_render(cmd, 0, 3);
         struct hk_cs *qc = copy_query_results(cmd);
         T("result copy launches the pending copies first",
           record_count == 1 && records[0].count == 2 &&
           !cmd->sync.stamp_copy_count);
         struct hk_cs *f = records[0].cs;
         T("copies follow the render, the result copy follows the copies",
           f != qc && f->sync.dep[HK_SYNC_VDM].cut == r0 &&
           (f->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP &&
           qc->sync.dep[HK_SYNC_CDM].cut == f &&
           qc->sync.dep[HK_SYNC_VDM].cut == r0);

         /* TRANSFER_WRITE -> SHADER_READ: the consumer waits for the result
          * copy, and so for the copies before it. */
         hk_sync_barrier(cmd, hk_sync_kinds_src(VK_PIPELINE_STAGE_2_COPY_BIT),
                         hk_sync_kinds_dst(
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT));
         hk_cmd_buffer_end_compute(cmd);
         struct hk_cs *d = dispatch(cmd);
         T("shader after a transfer barrier waits for the result copy",
           d->sync.dep[HK_SYNC_CDM].cut == qc);
         hk_sync_barrier(cmd, hk_sync_kinds_src(VK_PIPELINE_STAGE_2_COPY_BIT),
                         hk_sync_kinds_dst(
                            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT));
         struct hk_cs *r1 = render(cmd);
         T("render after a transfer barrier waits for the result copy",
           r1->sync.dep[HK_SYNC_CDM].cut == d ||
           r1->sync.dep[HK_SYNC_CDM].cut == qc);

         /* TRANSFER_WRITE -> HOST_READ, as hk_CmdPipelineBarrier2() records
          * the host-visibility flush on M5 and G17P. */
         struct hk_cmd_buffer *host = new_cmd(chips[c], exact);
         profiled_render(host, 0, 2);
         struct hk_cs *hq = copy_query_results(host);
         const uint8_t src = hk_sync_kinds_src(VK_PIPELINE_STAGE_2_COPY_BIT);
         hk_sync_barrier(host, src,
                         hk_sync_kinds_dst(VK_PIPELINE_STAGE_2_HOST_BIT));
         struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
         hk_sync_src_dep(host, src, host->current_cs.cs, dep);
         host->sync.kind_override = HK_SYNC_ALL;
         struct hk_cs *flush = hk_sync_get_cs_after(host, dep);
         host->sync.kind_override = 0;
         T("host flush follows the result copy",
           flush && (flush == hq || flush->sync.dep[HK_SYNC_CDM].cut == hq));
         T("consumers resolve", submit(new_queue(exact), &cmd, 1) &&
                                submit(new_queue(exact), &host, 1));
      }
   }
}

/* An end timestamp attached to a render pass-merge candidate (or a covered
 * copy of it) keeps pass_merge.sync_epoch current so the next compatible
 * render pass instance can still merge into the same render stream, and a
 * merged pass that already carries an end stamp never lets an in-pass
 * TOP_OF_PIPE stamp take start on a start-stamp-ordered chip. */
static void
test_pass_merge_end_stamp(void)
{
   const enum agx_chip merge_chips[] = {
      AGX_CHIP_G16G, AGX_CHIP_G17P, AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C};
   for (unsigned c = 0; c < ARRAY_SIZE(merge_chips); ++c) {
      for (unsigned exact = 0; exact < 2; ++exact) {
         record_count = 0;
         struct hk_cmd_buffer *cmd = new_cmd(merge_chips[c], exact);
         struct hk_cs *r = render(cmd);

         cmd->pass_merge.valid = true;
         cmd->pass_merge.cs = r;
         cmd->pass_merge.next_seq = cmd->sync.next_seq;
         cmd->pass_merge.sync_epoch = cmd->sync.epoch;

         stamp_all(cmd, 0);
         T("post-pass end stamp preserves pass_merge epoch",
           is_end(r, 0) && list_length(&cmd->control_streams) == 1 &&
           cmd->pass_merge.sync_epoch == cmd->sync.epoch);

         stamp_all(cmd, 1);
         T("covered copy after end stamp preserves pass_merge epoch",
           list_length(&cmd->control_streams) == 1 &&
           cmd->sync.stamp_copy_count == 1 &&
           copy_is(&cmd->sync.stamp_copies[0], 1, query(0)) &&
           cmd->pass_merge.sync_epoch == cmd->sync.epoch);

         /* Merge the next pass instance onto the same render stream. */
         cmd->pass_merge.valid = false;
         cmd->current_cs.gfx = r;
         hk_sync_merge_rendering(cmd, r);
         stamp(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 2);
         T("in-pass TOP_OF_PIPE on end-stamped merged pass never takes start",
           !r->timestamp.start.handle && is_end(r, 0) &&
           cmd->sync.stamp_copy_count == 2 &&
           copy_is(&cmd->sync.stamp_copies[1], 2, query(0)));
         stamp_all(cmd, 3);
         T("second pass end stamp queues as a copy of the merged end stamp",
           cmd->sync.stamp_copy_count == 3 &&
           copy_is(&cmd->sync.stamp_copies[2], 3, query(0)) &&
           list_length(&cmd->control_streams) == 1 && !record_count);
         end_render(cmd);

         hk_sync_flush_stamp_copies(cmd);
         T("all merged pass copies flush in one timestamp-only stream",
           record_count == 1 && records[0].count == 3 &&
           records[0].cs->sync.dep[HK_SYNC_VDM].cut == r &&
           (records[0].cs->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP);
         T("merged end-stamped pass resolves",
           submit(new_queue(exact), &cmd, 1));
      }
   }
}

/* vkCmdResetQueryPool on a timestamp pool, as hk_CmdResetQueryPool()
 * records it: one reset launch in the stream hk_query_reset_cs() returns. */
static struct hk_cs *
reset_stamps(struct hk_cmd_buffer *cmd)
{
   cmd->sync.kind_override = HK_SYNC_QRESET;
   struct hk_cs *cs = hk_query_reset_cs(cmd);
   cs->stats.cmds++;
   cmd->sync.kind_override = 0;
   hk_sync_query_reset_done(cmd, cs);
   return cs;
}

/* The barrier field a render's vertex half was given. */
static uint16_t
vertex_barrier(const struct hk_cs *r, unsigned sq)
{
   return r->sync.ta_split ? r->sync.barrier_ta[sq] : r->sync.barrier[sq];
}

/* A timestamp query reset is ordered against the timestamps and query
 * commands that use its queries, not against application work or other
 * timestamp resets. */
static void
test_timestamp_resets_order_query_work_only(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned exact = 0; exact < 2; ++exact) {
         /* Within one command buffer. */
         struct hk_cmd_buffer *cmd = new_cmd(chips[c], exact);
         struct hk_cs *d0 = dispatch(cmd);
         struct hk_cs *z0 = reset_stamps(cmd);
         T("reset waits for earlier work",
           z0 == d0 || z0->sync.dep[HK_SYNC_CDM].cut == d0);
         T("reset is a reset stream",
           (z0->sync.kind & HK_SYNC_QRESET) &&
           cmd->sync.last[util_logbase2(HK_SYNC_QRESET)][HK_SYNC_CDM] == z0);
         hk_cmd_buffer_end_compute(cmd);
         struct hk_cs *z1 = reset_stamps(cmd);
         T("second reset does not wait for the first",
           z1 != z0 && (!z1->sync.dep[HK_SYNC_CDM].cut ||
                        z1->sync.dep[HK_SYNC_CDM].cut == d0));
         hk_cmd_buffer_end_compute(cmd);
         struct hk_cs *r = begin_render(cmd);
         T("render does not wait for the resets",
           !r->sync.dep[HK_SYNC_CDM].cut && !r->sync.dep_ta[HK_SYNC_CDM].cut);
         stamp(cmd, VK_PIPELINE_STAGE_2_NONE, 0);
         T("top-of-pipe stamp in the pass waits for the resets",
           r->sync.dep[HK_SYNC_CDM].cut == z1);
         end_render(cmd);
         struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
         hk_sync_src_dep(cmd, HK_SYNC_QUERY_SCOPE, NULL, dep);
         T("query command follows the resets",
           dep[HK_SYNC_CDM].cut && dep[HK_SYNC_CDM].cut->sync.seq >= z1->sync.seq);

         /* Reset-only command buffers ahead of a stamped render, in one
          * submission. */
         struct hk_cmd_buffer *a = new_cmd(chips[c], exact);
         struct hk_cs *za = reset_stamps(a);
         struct hk_cmd_buffer *b = new_cmd(chips[c], exact);
         struct hk_cs *zb = reset_stamps(b);
         struct hk_cmd_buffer *g = new_cmd(chips[c], exact);
         hk_sync_barrier(g, HK_SYNC_ALL, HK_SYNC_ALL);
         struct hk_cs *rg = begin_render(g);
         stamp_all(g, 1);
         end_render(g);
         struct hk_cmd_buffer *all[] = {a, b, g};
         T("reset submission resolves", submit(new_queue(exact), all, 3));
         T("resets of separate command buffers do not chain",
           za->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE &&
           zb->sync.barrier[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE);
         T("stamped render follows both resets",
           is_end(rg, 1) && rg->sync.barrier[HK_SYNC_CDM] == zb->sync.index &&
           zb->sync.index > za->sync.index);
         T("stamped render's vertex half does not wait for the resets",
           vertex_barrier(rg, HK_SYNC_CDM) == DRM_ASAHI_BARRIER_NONE);
      }
   }
}

/* A top-of-pipe stamp recorded before any stream of the command buffer rides
 * on the first stream's first end stamp instead of a command of its own. */
static void
test_leading_stamp_rides_first_stream(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned exact = 0; exact < 2; ++exact) {
         /* A barrier first, so the stamp cannot wait for submission to ride
          * on a start stamp. */
         struct hk_cmd_buffer *cmd = new_cmd(chips[c], exact);
         hk_sync_barrier(cmd, HK_SYNC_ALL, HK_SYNC_ALL);
         stamp(cmd, VK_PIPELINE_STAGE_2_NONE, 0);
         T("leading stamp waits for a stream",
           list_is_empty(&cmd->control_streams) && cmd->sync.lead_ts.handle);
         struct hk_cs *r = begin_render(cmd);
         stamp_all(cmd, 1);
         end_render(cmd);
         T("leading stamp is the render's vertex end",
           same_req(r->timestamp.vtx_end, query(0)) && is_end(r, 1) &&
           list_is_singular(&cmd->control_streams) && !cmd->sync.lead_ts.handle);

         struct hk_cmd_buffer *k = new_cmd(chips[c], exact);
         hk_sync_barrier(k, HK_SYNC_ALL, HK_SYNC_ALL);
         stamp(k, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0);
         struct hk_cs *d = dispatch(k);
         struct hk_cs *kc =
            list_first_entry(&k->control_streams, struct hk_cs, node);
         T("leading stamp before a dispatch flushes into a preceding carrier",
           kc != d && is_end(kc, 0) && !d->timestamp.end.handle &&
           (kc->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP &&
           list_length(&k->control_streams) == 2 && !k->sync.lead_ts.handle);
         stamp_all(k, 1);
         T("trailing stamp after dispatch stays distinct from leading stamp",
           is_end(kc, 0) && is_end(d, 1) && !k->sync.stamp_copy_count &&
           list_length(&k->control_streams) == 2);

         struct hk_cmd_buffer *e = new_cmd(chips[c], exact);
         hk_sync_barrier(e, HK_SYNC_ALL, HK_SYNC_ALL);
         stamp(e, VK_PIPELINE_STAGE_2_NONE, 0);
         hk_sync_flush_pending_timestamp(e);
         T("leading stamp with no stream gets its own command",
           list_is_singular(&e->control_streams) && is_end(last_cs(e), 0) &&
           (last_cs(e)->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP);

         /* The vertex half waits for timestamp resets of earlier command
          * buffers. */
         struct hk_cmd_buffer *z = new_cmd(chips[c], exact);
         struct hk_cs *zr = reset_stamps(z);
         struct hk_cmd_buffer *v = new_cmd(chips[c], exact);
         hk_sync_barrier(v, HK_SYNC_ALL, HK_SYNC_ALL);
         stamp(v, VK_PIPELINE_STAGE_2_NONE, 2);
         struct hk_cs *rv = render(v);
         struct hk_cmd_buffer *both[] = {z, v};
         T("reset and leading-stamp submission resolves",
           submit(new_queue(exact), both, 2));
         T("leading stamp's render follows the reset",
           same_req(rv->timestamp.vtx_end, query(2)) &&
           vertex_barrier(rv, HK_SYNC_CDM) == zr->sync.index);

         /* A top-of-pipe stamp has an empty first scope: its render does not
          * wait for the earlier command buffer's dispatch. A transfer-only
          * barrier keeps the stamp from riding on a start stamp. */
         struct hk_cmd_buffer *w = new_cmd(chips[c], exact);
         struct hk_cs *wd = dispatch(w);
         struct hk_cmd_buffer *t = new_cmd(chips[c], exact);
         hk_sync_barrier(t, HK_SYNC_XFER, HK_SYNC_XFER);
         stamp(t, VK_PIPELINE_STAGE_2_NONE, 3);
         struct hk_cs *rt = render(t);
         struct hk_cmd_buffer *wt[] = {w, t};
         (void)wd;
         T("top-of-pipe leading stamp submission resolves",
           submit(new_queue(exact), wt, 2));
         T("top-of-pipe leading stamp adds no wait on earlier work",
           same_req(rt->timestamp.vtx_end, query(3)) &&
           vertex_barrier(rt, HK_SYNC_CDM) == DRM_ASAHI_BARRIER_NONE);

         /* An ALL_COMMANDS stamp ahead of every stream rides on the first
          * one too, which then waits for the stamp's first scope: the
          * earlier command buffer's dispatch, as a command of its own would
          * have. */
         struct hk_cmd_buffer *a = new_cmd(chips[c], exact);
         struct hk_cs *ad = dispatch(a);
         struct hk_cmd_buffer *s = new_cmd(chips[c], exact);
         hk_sync_barrier(s, HK_SYNC_XFER, HK_SYNC_XFER);
         stamp_all(s, 4);
         T("leading ALL_COMMANDS stamp waits for a stream",
           list_is_empty(&s->control_streams) && s->sync.lead_ts.handle);
         struct hk_cs *rs = begin_render(s);
         stamp_all(s, 5);
         end_render(s);
         T("leading ALL_COMMANDS stamp is the render's vertex end",
           same_req(rs->timestamp.vtx_end, query(4)) && is_end(rs, 5) &&
           list_is_singular(&s->control_streams));
         struct hk_cmd_buffer *as[] = {a, s};
         T("leading ALL_COMMANDS stamp submission resolves",
           submit(new_queue(exact), as, 2));
         T("leading ALL_COMMANDS stamp's vertex half follows earlier work",
           vertex_barrier(rs, HK_SYNC_CDM) == ad->sync.index);

         /* On a dispatch after an earlier render, the leading ALL_COMMANDS
          * stamp flushes into a preceding carrier ordered after the render,
          * leaving the dispatch unstamped for a trailing stamp. */
         struct hk_cmd_buffer *g = new_cmd(chips[c], exact);
         struct hk_cs *gr = render(g);
         struct hk_cmd_buffer *h = new_cmd(chips[c], exact);
         stamp_all(h, 6);
         struct hk_cs *hd = dispatch(h);
         struct hk_cs *hc =
            list_first_entry(&h->control_streams, struct hk_cs, node);
         struct hk_cmd_buffer *gh[] = {g, h};
         T("leading ALL_COMMANDS stamp before a dispatch is a preceding carrier",
           hc != hd && is_end(hc, 6) && !hd->timestamp.end.handle &&
           (hc->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP &&
           list_length(&h->control_streams) == 2 &&
           submit(new_queue(exact), gh, 2));
         T("leading ALL_COMMANDS carrier follows the render",
           hc->sync.barrier[HK_SYNC_VDM] == gr->sync.index);

         /* A query command first: the stamp gets a command of its own,
          * after its first scope. */
         struct hk_cmd_buffer *m = new_cmd(chips[c], exact);
         struct hk_cs *mr = render(m);
         struct hk_cmd_buffer *n = new_cmd(chips[c], exact);
         stamp_all(n, 7);
         struct hk_cs *nq = copy_query_results(n);
         struct hk_cs *nk = list_first_entry(&n->control_streams,
                                             struct hk_cs, node);
         struct hk_cmd_buffer *mn[] = {m, n};
         T("leading ALL_COMMANDS stamp before a query command is a carrier",
           nk != nq && is_end(nk, 7) &&
           (nk->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP &&
           submit(new_queue(exact), mn, 2));
         T("leading ALL_COMMANDS carrier follows the earlier render",
           nk->sync.barrier[HK_SYNC_VDM] == mr->sync.index);

         /* A second stamp before any stream is a copy of the leading one:
          * the render carries the leading stamp after the wider of their
          * first scopes, and the copy launches after the render. */
         struct hk_cmd_buffer *p = new_cmd(chips[c], exact);
         struct hk_cs *pd = dispatch(p);
         struct hk_cmd_buffer *two = new_cmd(chips[c], exact);
         hk_sync_barrier(two, HK_SYNC_XFER, HK_SYNC_XFER);
         stamp(two, VK_PIPELINE_STAGE_2_NONE, 8);
         stamp_all(two, 9);
         T("second leading stamp is a copy of the first",
           list_is_empty(&two->control_streams) && two->sync.lead_ts.handle &&
           two->sync.stamp_copy_count == 1 &&
           copy_is(&two->sync.stamp_copies[0], 9, query(8)));
         struct hk_cs *rp = render(two);
         T("render carries the leading stamp and its copies follow it",
           same_req(rp->timestamp.vtx_end, query(8)) &&
           list_is_singular(&two->control_streams) &&
           two->sync.stamp_copy_after[HK_SYNC_VDM] == rp);
         record_count = 0;
         hk_sync_flush_stamp_copies(two);
         T("copy of the leading stamp launches after the render",
           record_count == 1 && records[0].count == 1 &&
           records[0].cs->sync.dep[HK_SYNC_VDM].cut == rp);
         struct hk_cmd_buffer *ptwo[] = {p, two};
         T("two leading stamps resolve", submit(new_queue(exact), ptwo, 2));
         T("render of two leading stamps waits for the wider first scope",
           vertex_barrier(rp, HK_SYNC_CDM) == pd->sync.index);

         /* No stream at all, flushed as vkEndCommandBuffer does (copies
          * first): the leading stamp's command, then the copy after it. */
         struct hk_cmd_buffer *q2 = new_cmd(chips[c], exact);
         hk_sync_barrier(q2, HK_SYNC_XFER, HK_SYNC_XFER);
         stamp_all(q2, 10);
         stamp_all(q2, 11);
         record_count = 0;
         hk_sync_flush_stamp_copies(q2);
         hk_sync_flush_pending_timestamp(q2);
         struct hk_cs *qk = list_first_entry(&q2->control_streams,
                                             struct hk_cs, node);
         T("leading stamps with no stream: a carrier, then the copy after it",
           list_length(&q2->control_streams) == 2 && is_end(qk, 10) &&
           record_count == 1 && records[0].cs != qk &&
           records[0].cs->sync.dep[HK_SYNC_CDM].cut == qk &&
           copy_is(&records[0].copies[0], 11, query(10)));
      }
   }
}

/* Regression test: a compute workload bracketed by leading and trailing
 * ALL_COMMANDS timestamps with interleaved graphics+compute barriers
 * (vkd3d-proton EndQuery(0) -> Dispatch -> Barrier -> Dispatch -> Barrier ->
 * EndQuery(1) -> CopyQueryPoolResults, with hoisted reset in a prior CB).
 * Both stamps must be emitted on distinct streams (start before compute, end
 * after compute), neither dropped nor collapsed to a copy. */
static void
test_leading_stamp_compute_bracket_and_barriers(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned exact = 0; exact < 2; ++exact) {
         struct hk_cmd_buffer *z = new_cmd(chips[c], exact);
         struct hk_cs *zr = reset_stamps(z);

         struct hk_cmd_buffer *cmd = new_cmd(chips[c], exact);
         stamp_all(cmd, 0);
         struct hk_cs *d0 = dispatch(cmd);
         struct hk_cs *c0 =
            list_first_entry(&cmd->control_streams, struct hk_cs, node);
         hk_sync_barrier(cmd, HK_SYNC_GFX | HK_SYNC_COMPUTE,
                         HK_SYNC_GFX | HK_SYNC_COMPUTE);
         struct hk_cs *d1 = dispatch(cmd);
         hk_sync_barrier(cmd, HK_SYNC_GFX | HK_SYNC_COMPUTE,
                         HK_SYNC_GFX | HK_SYNC_COMPUTE);
         stamp_all(cmd, 1);
         struct hk_cs *cq = copy_query_results(cmd);

         bool saw_q0 = false, saw_q1 = false;
         list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
            saw_q0 |= is_end(cs, 0);
            saw_q1 |= is_end(cs, 1);
         }
         T("bracketed compute with GFX+COMPUTE barriers emits both stamps distinctly",
           c0 != d0 && d0 == d1 && is_end(c0, 0) && is_end(d1, 1) &&
           saw_q0 && saw_q1 && c0->sync.seq < d1->sync.seq &&
           !cmd->sync.stamp_copy_count &&
           (c0->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP &&
           cq != d1 && cq->sync.dep[HK_SYNC_CDM].cut == d1);

         struct hk_cmd_buffer *pair[] = {z, cmd};
         T("bracketed compute with prior reset resolves",
           submit(new_queue(exact), pair, 2));
         T("leading compute carrier waits on prior reset and copy follows end stamp",
           c0->sync.barrier[HK_SYNC_CDM] == zr->sync.index &&
           cq->sync.barrier[HK_SYNC_CDM] == d1->sync.index);

         /* Two leading stamps before a dispatch: first flushes into a carrier
          * and the copy reads that carrier (not the dispatch), while a later
          * trailing stamp rides on the dispatch. */
         struct hk_cmd_buffer *two = new_cmd(chips[c], exact);
         hk_sync_barrier(two, HK_SYNC_XFER, HK_SYNC_XFER);
         stamp(two, VK_PIPELINE_STAGE_2_NONE, 12);
         stamp_all(two, 13);
         struct hk_cs *td = dispatch(two);
         struct hk_cs *tc =
            list_first_entry(&two->control_streams, struct hk_cs, node);
         stamp_all(two, 14);
         T("two leading stamps before dispatch attach copy to carrier, not dispatch",
           tc != td && is_end(tc, 12) && is_end(td, 14) &&
           two->sync.stamp_copy_count == 1 &&
           copy_is(&two->sync.stamp_copies[0], 13, query(12)) &&
           two->sync.stamp_copy_after[HK_SYNC_CDM] == tc);
         record_count = 0;
         hk_sync_flush_stamp_copies(two);
         T("copy of leading compute carrier launches after carrier",
           record_count == 1 && records[0].count == 1 &&
           records[0].cs->sync.dep[HK_SYNC_CDM].cut == tc &&
           submit(new_queue(exact), &two, 1));

         /* A transfer compute stream opening with lead_ts keeps the flushed
          * carrier pure HK_SYNC_STAMP without inheriting HK_SYNC_XFER. */
         struct hk_cmd_buffer *xf = new_cmd(chips[c], exact);
         stamp_all(xf, 15);
         xf->sync.xfer_depth = 1;
         struct hk_cs *xd = dispatch(xf);
         xf->sync.xfer_depth = 0;
         struct hk_cs *xc =
            list_first_entry(&xf->control_streams, struct hk_cs, node);
         T("leading stamp flushed by transfer stays HK_SYNC_STAMP",
           xc != xd && is_end(xc, 15) && !xd->timestamp.end.handle &&
           (xc->sync.kind & HK_SYNC_STREAM_KINDS) == HK_SYNC_STAMP &&
           (xd->sync.kind & HK_SYNC_XFER) &&
           submit(new_queue(exact), &xf, 1));
      }
   }
}

int
main(void)
{
   test_context = ralloc_context(NULL);
   if (!test_context)
      return EXIT_FAILURE;

   test_carriers_order_no_application_work();
   test_query_commands_follow_carriers();
   test_consecutive_stamps_coalesce();
   test_stamps_with_work_stay_distinct();
   test_batched_copies();
   test_in_pass_copies();
   test_copies_order_no_application_work();
   test_copy_consumers_wait();
   test_pass_merge_end_stamp();
   test_timestamp_resets_order_query_work_only();
   test_leading_stamp_rides_first_stream();
   test_leading_stamp_compute_bracket_and_barriers();

   printf("%u tests, %u failures\n", tests, failures);
   ralloc_free(test_context);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

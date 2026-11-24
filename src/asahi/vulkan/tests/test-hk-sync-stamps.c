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

         /* With exact priors a stamp on a compute command is ordered
          * through its compute queue alone (stamp_wait()), so the second
          * stamp takes a carrier, which the rest copy. */
         const bool own = cdm_stamp_queue_ordered(chips[c], exact);
         struct hk_cs *src = own ? last_cs(cmd) : d;
         const unsigned copies = own ? 3 : 4;
         T("a run of stamps records at most one carrier",
           list_length(&cmd->control_streams) == (own ? 2 : 1) &&
           stamped_streams(cmd) == (own ? 2 : 1));
         T("the rest are pending copies of the run's last command",
           cmd->sync.stamp_copy_count == copies &&
           cmd->sync.stamp_copy_after[HK_SYNC_CDM] == src &&
           !cmd->sync.stamp_copy_after[HK_SYNC_VDM] && !record_count);
         bool sources = true;
         for (unsigned i = 0; i < copies; ++i) {
            sources &= copy_is(&cmd->sync.stamp_copies[i], 5 - copies + i,
                               src->timestamp.end);
         }
         T("each copy reads the stamp it repeats", sources);

         hk_sync_flush_stamp_copies(cmd);
         struct hk_cs *f = last_cs(cmd);
         T("flush launches every copy once",
           record_count == 1 && records[0].cs == f &&
           records[0].count == copies && !cmd->sync.stamp_copy_count);
         T("flush stream follows the stamp it reads",
           f->sync.dep[HK_SYNC_CDM].cut == src &&
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

/* Later stamps inside one render pass instance copy the render's end stamp
 * in one launch when its post-graphics stream ends. */
static void
test_in_pass_copies(void)
{
   for (unsigned c = 0; c < ARRAY_SIZE(chips); ++c) {
      for (unsigned n = 1; n <= 40; n += 39) {
         record_count = 0;
         struct hk_cmd_buffer *cmd = new_cmd(chips[c], false);
         struct hk_cs *r = begin_render(cmd);
         stamp_all(cmd, 0);
         for (unsigned i = 1; i <= n; ++i)
            stamp_all(cmd, i);
         struct hk_cs *post = cmd->current_cs.post_gfx;
         T("first in-pass stamp ends the render", is_end(r, 0));
         T("later in-pass stamps wait in the post-graphics stream",
           post && !record_count &&
           util_dynarray_num_elements(&post->stamp_copies,
                                      struct hk_stamp_copy) == n &&
           post->sync.dep[HK_SYNC_VDM].cut == r);
         end_render(cmd);
         bool all = record_count == 1 && records[0].cs == post &&
                    records[0].count == n;
         for (unsigned i = 0; all && i < n; ++i)
            all &= copy_is(&records[0].copies[i], 1 + i, query(0));
         T("the render's copies launch once as its stream ends", all);
         T("post-graphics stream keeps no copies", !post->stamp_copies.size);
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

   printf("%u tests, %u failures\n", tests, failures);
   ralloc_free(test_context);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

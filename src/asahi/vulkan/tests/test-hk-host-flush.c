/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Host availability: which dependencies ask for the memory-scoped flush,
 * when the open compute stream takes it at the dependency, that deferring it
 * creates no stream, and what the flush that ends the submission waits for,
 * through the real hk_sync resolver; and the streams and lists of the
 * flush-only compute commands a queue builds itself. No Vulkan device, DRM or
 * GPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/ralloc.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_queue.h"
#include "hk_queue_flush.h"

static unsigned tests, failures;
static void *ctx;

#define T(name, cond)                                                          \
   do {                                                                        \
      tests++;                                                                 \
      if (!(cond)) {                                                           \
         failures++;                                                           \
         printf("FAIL %s line%d\n", name, __LINE__);                           \
      } else                                                                   \
         printf("PASS %s\n", name);                                            \
   } while (0)

#define NONE DRM_ASAHI_BARRIER_NONE

static void *
alloc(size_t size)
{
   void *ptr = rzalloc_size(ctx, size);
   if (!ptr) {
      fputs("Failed to allocate host flush test fixture\n", stderr);
      exit(EXIT_FAILURE);
   }
   return ptr;
}

static struct hk_queue *
new_queue(bool exact)
{
   struct hk_queue *q = alloc(sizeof(*q));
   q->sync.exact_prior = exact;
   return q;
}

static struct hk_cmd_buffer *
new_cmd(void)
{
   struct hk_cmd_buffer *c = alloc(sizeof(*c));
   c->vk.base.device = alloc(sizeof(struct hk_device));
   list_inithead(&c->control_streams);
   util_dynarray_init(&c->sync_pass_records, NULL);
   return c;
}

static struct hk_cs *
add(struct hk_cmd_buffer *c, bool compute)
{
   struct hk_cs *s = alloc(sizeof(*s));
   s->type = compute ? HK_CS_CDM : HK_CS_VDM;
   list_addtail(&s->node, &c->control_streams);
   hk_sync_stream_created(c, s, &s, compute);
   return s;
}

/* An open compute stream holding a dispatch or copy, as recording one
 * leaves it. */
static struct hk_cs *
open_compute(struct hk_cmd_buffer *c)
{
   struct hk_cs *s = add(c, true);
   s->stats.cmds = 1;
   c->current_cs.cs = s;
   return s;
}

static void
set_chip(struct hk_cmd_buffer *c, enum agx_chip chip)
{
   ((struct hk_device *)c->vk.base.device)->dev.chip = chip;
}

static VkMemoryBarrier2
mem(VkPipelineStageFlags2 src, VkAccessFlags2 src_access,
    VkPipelineStageFlags2 dst, VkAccessFlags2 dst_access)
{
   return (VkMemoryBarrier2){
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = src,
      .srcAccessMask = src_access,
      .dstStageMask = dst,
      .dstAccessMask = dst_access,
   };
}

static VkDependencyInfo
dep(const VkMemoryBarrier2 *b, uint32_t count)
{
   return (VkDependencyInfo){
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = count,
      .pMemoryBarriers = b,
   };
}

/* What vkCmdPipelineBarrier2 records outside a render pass on G17P and M5,
 * for the synchronization state: the stream it records the host flush in,
 * or NULL. */
static struct hk_cs *
pipeline_barrier(struct hk_cmd_buffer *cmd, const VkDependencyInfo *info)
{
   uint8_t src, dst;
   hk_sync_dependency_kinds(info, false, &src, &dst);
   hk_sync_barrier(cmd, src, dst);
   return hk_sync_host_flush(cmd, info);
}

struct flush {
   bool ok;
   uint8_t kinds;
   uint16_t barrier[HK_SYNC_NUM_SQ];
   uint64_t prior[HK_SYNC_NUM_SQ];
   uint64_t cdm_after;
};

/* The submission as queue_submit_recordings() resolves it, and the flush it
 * appends when any of its command buffers deferred one. */
static struct flush
submit(struct hk_queue *q, struct hk_cmd_buffer **cmds, unsigned n, bool chain)
{
   struct flush f = {0};
   const uint64_t base[HK_SYNC_NUM_SQ] = {q->sync.submitted[0],
                                          q->sync.submitted[1]};
   uint32_t count[HK_SYNC_NUM_SQ] = {0, 0};
   for (unsigned i = 0; i < n; ++i) {
      f.kinds |= cmds[i]->sync.host_flush_kinds;
      list_for_each_entry(struct hk_cs, cs, &cmds[i]->control_streams, node)
         count[hk_sync_sq(cs)]++;
   }
   f.ok = hk_sync_resolve_submit(q, cmds, n, false, chain, 0, NULL);
   f.barrier[0] = f.barrier[1] = NONE;
   if (f.ok && f.kinds)
      f.ok = hk_sync_host_flush_prefix(&q->sync, f.kinds, base, count,
                                       f.barrier, f.prior);
   f.cdm_after = q->sync.submitted[HK_SYNC_CDM];
   return f;
}

#define ALL_CMDS  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
#define TOP       VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
#define BOTTOM    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT
#define HOST      VK_PIPELINE_STAGE_2_HOST_BIT
#define CS        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
#define FS        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
#define COLOR     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
#define XFER      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT
#define H_R       VK_ACCESS_2_HOST_READ_BIT
#define H_W       VK_ACCESS_2_HOST_WRITE_BIT
#define M_R       VK_ACCESS_2_MEMORY_READ_BIT
#define M_W       VK_ACCESS_2_MEMORY_WRITE_BIT
#define S_R       VK_ACCESS_2_SHADER_READ_BIT
#define S_W       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
#define C_W       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
#define T_W       VK_ACCESS_2_TRANSFER_WRITE_BIT

/* Stream creation and the resolver reference recording entry points the
 * fixtures never reach: their streams carry no immediate writes, queries or
 * payloads. Reaching one is a bug. */
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

/* These devices never map memory at a second address, so the real function
 * adds no flush either. */
void
hk_cdm_end_stream_flush(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   (void)cs;
}

void
hk_flush_query_resets(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   if (cs->query_reset_pending)
      abort();
}

void
hk_tess_batch_flush(struct hk_cmd_buffer *cmd)
{
   (void)cmd;
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

struct agx_ptr
hk_pool_cs_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   (void)cmd;
   (void)size;
   (void)alignment;
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

/* Which dependencies perform a device-to-host domain operation, and after
 * which device work. */
static void
test_kinds(void)
{
   const struct {
      const char *name;
      VkPipelineStageFlags2 src, dst;
      VkAccessFlags2 src_access, dst_access;
      bool host;
      uint8_t kinds;
   } cases[] = {
      /* An empty or host-only first scope follows no device write. */
      {"NONE -> HOST_READ", VK_PIPELINE_STAGE_2_NONE, HOST, 0, H_R, true, 0},
      {"TOP -> HOST_READ", TOP, HOST, 0, H_R, true, 0},
      {"HOST_WRITE -> HOST_READ", HOST, HOST, H_W, H_R, true, 0},
      /* The domain operation also takes what earlier availability
       * operations of a chain made available: read-only or empty source
       * access still flushes once device work is in the first scope. */
      {"SHADER_READ -> HOST_READ", CS, HOST, S_R, H_R, true, HK_SYNC_COMPUTE},
      {"compute, no access -> HOST_READ", CS, HOST, 0, H_R, true,
       HK_SYNC_COMPUTE},
      {"SHADER_WRITE -> HOST_READ", CS, HOST, S_W, H_R, true,
       HK_SYNC_COMPUTE},
      {"COLOR_WRITE -> HOST_WRITE", COLOR, HOST, C_W, H_W, true, HK_SYNC_GFX},
      {"TRANSFER_WRITE -> HOST_READ", XFER, HOST, T_W, H_R, true,
       HK_SYNC_XFER},
      {"MEMORY_WRITE -> MEMORY_READ at HOST", ALL_CMDS, HOST, M_W, M_R, true,
       HK_SYNC_ALL},
      {"MEMORY_WRITE -> MEMORY_WRITE at HOST", FS, HOST, M_W, M_W, true,
       HK_SYNC_GFX},
      /* BOTTOM_OF_PIPE in the first scope is ALL_COMMANDS. */
      {"BOTTOM|HOST -> TOP|HOST, HOST_WRITE|MEMORY_READ", BOTTOM | HOST,
       TOP | HOST, H_W | M_R, H_W | M_R, true, HK_SYNC_ALL},
      {"BOTTOM -> HOST, MEMORY_READ -> HOST_WRITE", BOTTOM, HOST, M_R, H_W,
       true, HK_SYNC_ALL},
      /* No host access: an execution dependency into the host stage, the
       * host waits through a fence or event instead. */
      {"execution only into HOST", CS, HOST, 0, 0, false, 0},
      /* ALL_COMMANDS does not include the host stage. */
      {"MEMORY_READ at ALL_COMMANDS", CS, ALL_CMDS, S_W, M_R, false, 0},
      /* The other direction is not a device-to-host operation. */
      {"HOST_WRITE -> SHADER_READ", HOST, CS, H_W, S_R, false, 0},
      {"SHADER_WRITE -> SHADER_READ", CS, CS, S_W, S_R, false, 0},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      const VkMemoryBarrier2 b = mem(cases[i].src, cases[i].src_access,
                                     cases[i].dst, cases[i].dst_access);
      const VkDependencyInfo info = dep(&b, 1);
      T(cases[i].name, hk_sync_dependency_has_host_access(&info) ==
                          cases[i].host &&
                          hk_sync_host_flush_kinds(&info) == cases[i].kinds);
   }

   /* Host write to device read: no device work in the first scope, so no
    * device dependency either; the submission makes host writes before it
    * visible. Nothing changes for it. */
   {
      const VkMemoryBarrier2 b = mem(HOST, H_W, CS, S_R);
      const VkDependencyInfo info = dep(&b, 1);
      uint8_t src, dst;
      hk_sync_dependency_kinds(&info, false, &src, &dst);
      T("HOST_WRITE -> SHADER_READ: no device source", src == 0);
   }

   /* Only the barriers with a host access contribute their first scope. */
   {
      const VkMemoryBarrier2 b[] = {
         mem(COLOR, C_W, CS, S_R),
         mem(CS, S_W, HOST, H_R),
      };
      const VkDependencyInfo info = dep(b, 2);
      T("mixed dependency: host barrier's scope only",
        hk_sync_host_flush_kinds(&info) == HK_SYNC_COMPUTE);
   }

   /* Buffer and image barriers. */
   {
      const VkBufferMemoryBarrier2 bb = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
         .srcStageMask = XFER,
         .srcAccessMask = T_W,
         .dstStageMask = HOST,
         .dstAccessMask = H_R,
      };
      const VkImageMemoryBarrier2 ib = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = COLOR,
         .srcAccessMask = C_W,
         .dstStageMask = HOST,
         .dstAccessMask = H_R,
      };
      VkDependencyInfo info = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .bufferMemoryBarrierCount = 1,
         .pBufferMemoryBarriers = &bb,
      };
      T("buffer barrier TRANSFER_WRITE -> HOST_READ",
        hk_sync_dependency_has_host_access(&info) &&
           hk_sync_host_flush_kinds(&info) == HK_SYNC_XFER);
      info = (VkDependencyInfo){
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .imageMemoryBarrierCount = 1,
         .pImageMemoryBarriers = &ib,
      };
      T("image barrier COLOR_WRITE -> HOST_READ",
        hk_sync_dependency_has_host_access(&info) &&
           hk_sync_host_flush_kinds(&info) == HK_SYNC_GFX);
   }
}

/* Recording a host-availability dependency creates no stream; an event write
 * takes what is deferred. */
static void
test_recording(void)
{
   struct hk_cmd_buffer *cmd = new_cmd();
   const VkMemoryBarrier2 a = mem(BOTTOM | HOST, H_W | M_R, TOP | HOST,
                                  H_W | M_R);
   const VkMemoryBarrier2 b = mem(CS, S_W, HOST, H_R);
   const VkDependencyInfo da = dep(&a, 1), db = dep(&b, 1);
   pipeline_barrier(cmd, &da);
   pipeline_barrier(cmd, &db);
   T("barrier-only command buffer: no stream",
     list_is_empty(&cmd->control_streams));
   T("deferred kinds accumulate", cmd->sync.host_flush_kinds == HK_SYNC_ALL);
   T("event write takes the deferred kinds",
     hk_sync_take_host_flush(cmd) == HK_SYNC_ALL &&
        cmd->sync.host_flush_kinds == 0);

   cmd = new_cmd();
   const VkMemoryBarrier2 none = mem(VK_PIPELINE_STAGE_2_NONE, 0, HOST, H_R);
   const VkDependencyInfo dn = dep(&none, 1);
   pipeline_barrier(cmd, &dn);
   T("NONE -> HOST_READ defers nothing", cmd->sync.host_flush_kinds == 0 &&
                                           list_is_empty(&cmd->control_streams));
}

/* The command buffers of one frame: a render, a command buffer holding only
 * a full barrier with host bits, and the next render. The second render
 * waits on the first directly, and one flush after both ends the frame. */
static void
test_render_barrier_render(bool exact)
{
   struct hk_queue *q = new_queue(exact);
   struct hk_cmd_buffer *c0 = new_cmd(), *c1 = new_cmd(), *c2 = new_cmd();
   add(c0, false);
   const VkMemoryBarrier2 b = mem(BOTTOM | HOST, H_W | M_R, TOP | HOST,
                                  H_W | M_R);
   const VkDependencyInfo d = dep(&b, 1);
   pipeline_barrier(c1, &d);
   struct hk_cs *r1 = add(c2, false);

   struct hk_cmd_buffer *cmds[] = {c0, c1, c2};
   const struct flush f = submit(q, cmds, 3, false);
   T(exact ? "exact: frame resolves" : "legacy: frame resolves", f.ok);
   T("barrier-only command buffer sends no command",
     list_is_empty(&c1->control_streams));
   T("second render waits on the first render",
     r1->sync.barrier[HK_SYNC_VDM] == 1 &&
        r1->sync.barrier[HK_SYNC_CDM] == NONE && r1->sync.prior[1] == 0);
   T("flush follows both renders",
     f.barrier[HK_SYNC_VDM] == 2 && f.prior[HK_SYNC_VDM] == 0);
   T("flush waits on no compute",
     f.barrier[HK_SYNC_CDM] == NONE && f.prior[HK_SYNC_CDM] == 0);
   T("flush takes one compute ordinal", f.cdm_after == (exact ? 1 : 0));

   /* The flush is no later command's source. */
   struct hk_cmd_buffer *c3 = new_cmd();
   const VkMemoryBarrier2 all = mem(ALL_CMDS, M_W, ALL_CMDS, M_R);
   const VkDependencyInfo da = dep(&all, 1);
   pipeline_barrier(c3, &da);
   struct hk_cs *next = add(c3, true);
   struct hk_cmd_buffer *later[] = {c3};
   const struct flush g = submit(q, later, 1, false);
   T("next submission resolves", g.ok && g.kinds == 0);
   T("later compute does not wait on the flush",
     next->sync.barrier[HK_SYNC_CDM] == NONE &&
        next->sync.prior[HK_SYNC_CDM] == 0);
   T("later compute waits on the renders",
     exact ? next->sync.prior[HK_SYNC_VDM] == 2
           : next->sync.barrier[HK_SYNC_VDM] == 0);
}

/* A render, then a copy in the open compute stream and TRANSFER_WRITE ->
 * HOST_READ: the copy's command ends with the flush, and the submission sends
 * the render and the copy only, as before host flushes were deferred. */
static void
test_copy_host_read_in_stream(void)
{
   struct hk_queue *q = new_queue(true);
   struct hk_cmd_buffer *c = new_cmd();
   add(c, false);
   struct hk_cs *copy = open_compute(c);
   const VkMemoryBarrier2 b = mem(XFER, T_W, HOST, H_R);
   const VkDependencyInfo d = dep(&b, 1);
   T("TRANSFER_WRITE -> HOST_READ: the copy's stream takes the flush",
     pipeline_barrier(c, &d) == copy);
   T("nothing deferred", c->sync.host_flush_kinds == 0);
   T("no stream created", list_length(&c->control_streams) == 2);
   struct hk_cmd_buffer *cmds[] = {c};
   const struct flush f = submit(q, cmds, 1, false);
   T("submission sends no flush command of its own",
     f.ok && f.kinds == 0 && f.cdm_after == 1);
   T("the copy keeps its waits",
     copy->sync.barrier[HK_SYNC_VDM] == NONE &&
        copy->sync.prior[HK_SYNC_VDM] == 0);

   /* SHADER_WRITE -> HOST_READ after a dispatch in the same stream. */
   q = new_queue(true);
   c = new_cmd();
   add(c, false);
   struct hk_cs *dispatch = open_compute(c);
   const VkMemoryBarrier2 cs = mem(CS, S_W, HOST, H_R);
   const VkDependencyInfo dc = dep(&cs, 1);
   T("SHADER_WRITE -> HOST_READ: the dispatch's stream takes the flush",
     pipeline_barrier(c, &dc) == dispatch && c->sync.host_flush_kinds == 0);
}

/* Where the open compute stream cannot take the flush, it is deferred, and
 * the flush ending the submission follows the submission's last command on
 * each subqueue: behind a render it never names no render. */
static void
test_deferred_behind_render(void)
{
   const VkMemoryBarrier2 b = mem(CS, S_W, HOST, H_R);
   const VkDependencyInfo d = dep(&b, 1);

   /* A dispatch whose stream a render then closed. */
   struct hk_queue *q = new_queue(true);
   struct hk_cmd_buffer *c = new_cmd();
   open_compute(c);
   c->current_cs.cs = NULL;
   add(c, false);
   T("closed stream: deferred", !pipeline_barrier(c, &d) &&
                                   c->sync.host_flush_kinds == HK_SYNC_COMPUTE);
   struct hk_cmd_buffer *cmds[] = {c};
   const struct flush f = submit(q, cmds, 1, false);
   T("flush follows the dispatch", f.ok && f.barrier[HK_SYNC_CDM] == 1);
   T("flush follows the submission's render",
     f.barrier[HK_SYNC_VDM] == 1 && f.prior[HK_SYNC_VDM] == 0);

   /* An open stream with no work yet would become a flush-only command. */
   c = new_cmd();
   add(c, false);
   struct hk_cs *empty = add(c, true);
   c->current_cs.cs = empty;
   T("empty open stream: deferred", !pipeline_barrier(c, &d) &&
                                       c->sync.host_flush_kinds &&
                                       empty->stats.cmds == 0);

   /* A first scope whose latest stream follows the open one. */
   c = new_cmd();
   open_compute(c);
   add(c, false);
   const VkMemoryBarrier2 all = mem(ALL_CMDS, M_W, HOST, H_R);
   const VkDependencyInfo da = dep(&all, 1);
   T("render after the open stream: deferred",
     !pipeline_barrier(c, &da) && c->sync.host_flush_kinds == HK_SYNC_ALL);

   /* Inputs on the first command only: the flush follows it too. */
   q = new_queue(true);
   c = new_cmd();
   add(c, false);
   add(c, true);
   pipeline_barrier(c, &d);
   cmds[0] = c;
   const struct flush g = submit(q, cmds, 1, true);
   T("chained flush follows the first command",
     g.ok && g.barrier[HK_SYNC_VDM] == 1 && g.barrier[HK_SYNC_CDM] == 1);
}

/* A render before the open compute stream, then COLOR_WRITE -> HOST_READ:
 * the stream takes the flush and waits for the render on chips without CDM
 * overlap splitting (G15/G16), while on chips with AGX_CAP_CDM_OVERLAP_SPLIT
 * (G17P and M5) it defers instead of delaying the work the open stream already
 * holds behind an unrelated render. */
static void
test_render_scope_in_stream(void)
{
   const VkMemoryBarrier2 b = mem(COLOR, C_W, HOST, H_R);
   const VkDependencyInfo d = dep(&b, 1);

   struct hk_queue *q = new_queue(true);
   struct hk_cmd_buffer *c = new_cmd();
   set_chip(c, AGX_CHIP_G16G);
   struct hk_cs *r = add(c, false);
   struct hk_cs *s = open_compute(c);
   T("G16G: the open stream takes the flush", pipeline_barrier(c, &d) == s);
   T("and waits for the render", s->sync.dep[HK_SYNC_VDM].cut == r);
   struct hk_cmd_buffer *cmds[] = {c};
   const struct flush f = submit(q, cmds, 1, false);
   T("G16G: no flush command",
     f.ok && f.kinds == 0 && s->sync.barrier[HK_SYNC_VDM] == 1);

   const enum agx_chip split_chips[] = { AGX_CHIP_G17P, AGX_CHIP_G17G, AGX_CHIP_G17C };
   for (unsigned i = 0; i < ARRAY_SIZE(split_chips); ++i) {
      c = new_cmd();
      set_chip(c, split_chips[i]);
      add(c, false);
      open_compute(c);
      T("G17P/M5: deferred instead of delaying the stream's work",
        !pipeline_barrier(c, &d) && c->sync.host_flush_kinds == HK_SYNC_GFX);
   }
}

/* MEMORY_WRITE -> HOST_READ in a command buffer of its own, after a render
 * in an earlier submission: the flush names that render's ordinal. */
static void
test_memory_write_earlier_submission(void)
{
   struct hk_queue *q = new_queue(true);
   struct hk_cmd_buffer *c0 = new_cmd(), *c1 = new_cmd();
   add(c0, true);
   add(c0, false);
   struct hk_cmd_buffer *first[] = {c0};
   T("earlier submission", submit(q, first, 1, false).ok);

   const VkMemoryBarrier2 b = mem(ALL_CMDS, M_W, HOST, H_R);
   const VkDependencyInfo d = dep(&b, 1);
   pipeline_barrier(c1, &d);
   struct hk_cmd_buffer *second[] = {c1};
   const struct flush f = submit(q, second, 1, false);
   T("MEMORY_WRITE -> HOST_READ flushes", f.ok && f.kinds == HK_SYNC_ALL);
   T("flush names the earlier render and dispatch",
     f.barrier[HK_SYNC_VDM] == NONE && f.prior[HK_SYNC_VDM] == 1 &&
        f.barrier[HK_SYNC_CDM] == NONE && f.prior[HK_SYNC_CDM] == 1);
   T("flush ordinal after the dispatch", f.cdm_after == 2);
}

/* A render, then COLOR_WRITE -> SHADER_READ into compute at the end of the
 * submission; the next one holds only SHADER_READ -> HOST_READ from compute.
 * Their execution dependency chain hands the render's writes to the domain
 * operation, so the flush waits on the render although its own first access
 * scope has no write. */
static void
test_chain_through_read_only_source(void)
{
   struct hk_queue *q = new_queue(true);
   struct hk_cmd_buffer *a = new_cmd(), *b = new_cmd();
   add(a, false);
   const VkMemoryBarrier2 to_cs = mem(COLOR, C_W, CS, S_R);
   const VkDependencyInfo d0 = dep(&to_cs, 1);
   pipeline_barrier(a, &d0);
   struct hk_cmd_buffer *first[] = {a};
   T("render submission", submit(q, first, 1, false).ok);

   const VkMemoryBarrier2 to_host = mem(CS, S_R, HOST, H_R);
   const VkDependencyInfo d1 = dep(&to_host, 1);
   pipeline_barrier(b, &d1);
   struct hk_cmd_buffer *second[] = {b};
   const struct flush f = submit(q, second, 1, false);
   T("read-only source still flushes", f.ok && f.kinds == HK_SYNC_COMPUTE);
   T("flush follows the chained render",
     f.barrier[HK_SYNC_VDM] == NONE && f.prior[HK_SYNC_VDM] == 1);
}

/* A wait past the 16-bit field is named by its ordinal. */
static void
test_long_submission(void)
{
   struct hk_queue *q = new_queue(true);
   struct hk_cmd_buffer *c = new_cmd();
   const unsigned n = 70000;
   for (unsigned k = 0; k < n; ++k)
      add(c, true);
   const VkMemoryBarrier2 b = mem(CS, S_W, HOST, H_R);
   const VkDependencyInfo d = dep(&b, 1);
   pipeline_barrier(c, &d);
   struct hk_cmd_buffer *cmds[] = {c};
   const struct flush f = submit(q, cmds, 1, false);
   T("long submission flush by ordinal",
     f.ok && f.barrier[HK_SYNC_CDM] == NONE && f.prior[HK_SYNC_CDM] == n &&
        f.cdm_after == n + 1);
}

/* Prefix arithmetic on its own. */
static void
test_prefix(void)
{
   struct hk_queue *q = new_queue(true);
   q->sync.submitted[0] = 10;
   q->sync.submitted[1] = UINT64_MAX;
   const uint64_t base[2] = {10, UINT64_MAX};
   const uint32_t count[2] = {0, 0};
   uint16_t barrier[2];
   uint64_t prior[2];
   T("ordinal overflow rejected",
     !hk_sync_host_flush_prefix(&q->sync, HK_SYNC_COMPUTE, base, count,
                                barrier, prior));

   q = new_queue(false);
   q->sync.ever[HK_SYNC_VDM] = HK_SYNC_GFX;
   const uint32_t some[2] = {0, 3};
   const uint64_t zero[2] = {0, 0};
   T("legacy prefix", hk_sync_host_flush_prefix(&q->sync, HK_SYNC_COMPUTE,
                                                zero, some, barrier, prior) &&
                         barrier[HK_SYNC_VDM] == 0 &&
                         barrier[HK_SYNC_CDM] == 3 && !prior[0] && !prior[1]);
   q = new_queue(false);
   T("legacy prefix, nothing ran", hk_sync_host_flush_prefix(
                                      &q->sync, HK_SYNC_ALL, zero, count,
                                      barrier, prior) &&
                                      barrier[0] == NONE && barrier[1] == NONE);
}

/* The queue's own flush-only streams, in storage the BO cache returned with
 * its last user's bytes: the record and the terminator, then the
 * HK_CS_SPACE_PADDING bytes every stream keeps zero. */
static void
test_flush_stream_storage(void)
{
   static uint8_t storage[0x4000];
   memset(storage, 0xa5, sizeof(storage));

   /* The host flush record on G17P: the memory-scoped strong flush. */
   const struct agx3_cdm_barrier_info info = {
      .chip = AGX_CHIP_G17P,
      .flags = 0x5,
      .flush_memory = true,
   };
   uint8_t record[AGX3_CDM_BARRIER_SIZE];
   uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
   T("host flush record packs",
     agx3_cdm_barrier_pack(&info, record) &&
        agx3_cmdstream_load_le32(record) == UINT32_C(0x6000157c) &&
        agx3_cdm_stream_terminate_pack(AGX_CHIP_G17P, terminate));

   T("flush stream written into dirty storage",
     hk_queue_flush_stream_write(AGX_CHIP_G17P, storage, sizeof(storage),
                                 record, sizeof(record)));
   T("stream is the record, then the terminator",
     !memcmp(storage, record, sizeof(record)) &&
        !memcmp(storage + sizeof(record), terminate, sizeof(terminate)));

   const size_t end = sizeof(record) + sizeof(terminate);
   bool zero = true;
   for (size_t i = end; i < end + HK_CS_SPACE_PADDING; ++i)
      zero &= storage[i] == 0;
   T("padding after the terminator is cleared", zero);

   T("storage without room for the padding is refused",
     !hk_queue_flush_stream_write(AGX_CHIP_G17P, storage,
                                  end + HK_CS_SPACE_PADDING - 1, record,
                                  sizeof(record)));
}

/* The attachment set of each compute command of a payload, as the kernel
 * applies them: a set stays in force until the next one. */
static unsigned
compute_attachment_sets(const struct util_dynarray *payload, unsigned *sets,
                        unsigned max)
{
   unsigned current = 0, n = 0;
   for (size_t at = 0; at < payload->size;) {
      const struct drm_asahi_cmd_header *h =
         (const void *)((const uint8_t *)payload->data + at);
      if (h->cmd_type == DRM_ASAHI_SET_COMPUTE_ATTACHMENTS)
         current = h->size / sizeof(struct drm_asahi_attachment);
      else if (h->cmd_type == DRM_ASAHI_CMD_COMPUTE && n < max)
         sets[n++] = current;
      at += sizeof(*h) + h->size;
   }
   return n;
}

static void
append_compute(struct util_dynarray *payload)
{
   const struct drm_asahi_cmd_header header = {
      .cmd_type = DRM_ASAHI_CMD_COMPUTE,
      .size = sizeof(struct drm_asahi_cmd_compute),
      .vdm_barrier = NONE,
      .cdm_barrier = NONE,
   };
   const struct drm_asahi_cmd_compute cmd = {0};
   util_dynarray_append(payload, header);
   util_dynarray_append(payload, cmd);
}

/* The lists a flush-only compute command declares, by chip. */
static void
test_flush_lists(void)
{
   struct util_dynarray p;
   util_dynarray_init(&p, NULL);

   /* A dispatch with two attachment spans, then a flush-only command. */
   const struct drm_asahi_cmd_header set = {
      .cmd_type = DRM_ASAHI_SET_COMPUTE_ATTACHMENTS,
      .size = 2 * sizeof(struct drm_asahi_attachment),
      .vdm_barrier = NONE,
      .cdm_barrier = NONE,
   };
   const struct drm_asahi_attachment spans[2] = {
      {.pointer = 0x100000, .size = 0x4000},
      {.pointer = 0x200000, .size = 0x4000},
   };
   util_dynarray_append(&p, set);
   util_dynarray_append_array(&p, struct drm_asahi_attachment, spans, 2);
   append_compute(&p);
   hk_queue_append_flush_lists(AGX_CHIP_G17P, 0, &p);
   append_compute(&p);
   unsigned sets[2] = {~0u, ~0u};
   T("G17P: the flush declares an empty attachment set",
     compute_attachment_sets(&p, sets, 2) == 2 && sets[0] == 2 &&
        sets[1] == 0);

   util_dynarray_clear(&p);
   hk_queue_append_flush_lists(AGX_CHIP_G17G,
                               DRM_ASAHI_FEATURE_COMPUTE_RESOURCES, &p);
   const struct drm_asahi_cmd_header *h = p.data;
   T("M5: an empty resource list, no attachment set",
     p.size == sizeof(*h) && h->cmd_type == DRM_ASAHI_SET_COMPUTE_RESOURCES &&
        h->size == 0);

   util_dynarray_clear(&p);
   hk_queue_append_flush_lists(AGX_CHIP_G17G, 0, &p);
   T("M5 without resource lists: nothing", p.size == 0);

   util_dynarray_clear(&p);
   hk_queue_append_flush_lists(AGX_CHIP_G18G, 0, &p);
   h = p.data;
   T("G18G: an empty attachment set",
     p.size == sizeof(*h) &&
        h->cmd_type == DRM_ASAHI_SET_COMPUTE_ATTACHMENTS && h->size == 0);

   util_dynarray_clear(&p);
   hk_queue_append_flush_lists(AGX_CHIP_G13G, ~UINT64_C(0), &p);
   T("AGX2: nothing", p.size == 0);

   util_dynarray_fini(&p);
}

int
main(void)
{
   ctx = ralloc_context(NULL);
   if (!ctx)
      return EXIT_FAILURE;
   test_kinds();
   test_recording();
   test_render_barrier_render(true);
   test_render_barrier_render(false);
   test_copy_host_read_in_stream();
   test_deferred_behind_render();
   test_render_scope_in_stream();
   test_memory_write_earlier_submission();
   test_chain_through_read_only_source();
   test_long_submission();
   test_prefix();
   test_flush_stream_storage();
   test_flush_lists();
   ralloc_free(ctx);
   printf("%u tests, %u failures\n", tests, failures);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

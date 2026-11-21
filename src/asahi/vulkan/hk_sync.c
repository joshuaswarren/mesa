/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Recorded-dependency tracking for Honeykrisp control streams. The design is
 * described in hk_sync.h.
 */
#include "hk_sync.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <strings.h>

#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_queue.h"

#include "libagx_dgc.h"
#include "libagx_shaders.h"

#include "drm-uapi/asahi_drm.h"
#include "util/bitscan.h"
#include "util/list.h"
#include "util/u_debug.h"
#include "util/log.h"

static inline void
dep_max(struct hk_sync_dep *a, const struct hk_sync_dep *b)
{
   a->cut = hk_sync_later(a->cut, b->cut);
   a->prior |= b->prior;
}

static inline uint8_t
full_kinds(uint8_t kinds)
{
   return hk_sync_full_kinds(kinds);
}

/* The requirement on the next stream of the given kinds: the union over the
 * kinds, since a stream that does several kinds of work waits for all of
 * them. */
static void
req_for_kinds(const struct hk_cmd_buffer *cmd, uint8_t kinds,
              struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      dep[sq] = (struct hk_sync_dep){0};
      u_foreach_bit(k, full_kinds(kinds)) {
         dep_max(&dep[sq], &cmd->sync.req[k][sq]);
         if (cmd->sync.in_pass)
            dep_max(&dep[sq], &cmd->sync.pass_req[k][sq]);
      }
   }
}

/* The requirement on the next stream of exactly the given destination
 * classes. */
static void
req_for_mask(const struct hk_cmd_buffer *cmd, uint8_t mask,
             struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      dep[sq] = (struct hk_sync_dep){0};
      u_foreach_bit(k, mask) {
         dep_max(&dep[sq], &cmd->sync.req[k][sq]);
         if (cmd->sync.in_pass)
            dep_max(&dep[sq], &cmd->sync.pass_req[k][sq]);
      }
   }
}

/* The part of req_for_kinds() a render's vertex (tiling) half needs: not the
 * fragment-only requirements. */
static void
req_for_kinds_ta(const struct hk_cmd_buffer *cmd, uint8_t kinds,
                 struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      dep[sq] = (struct hk_sync_dep){0};
      u_foreach_bit(k, kinds) {
         dep_max(&dep[sq], &cmd->sync.req[k][sq]);
         if (cmd->sync.in_pass)
            dep_max(&dep[sq], &cmd->sync.pass_req[k][sq]);
      }
   }
}

void
hk_sync_reset(struct hk_cmd_buffer *cmd)
{
   memset(&cmd->sync, 0, sizeof(cmd->sync));
   util_dynarray_clear(&cmd->sync_pass_records);
}

#define TILE_BUCKETS 256
#define TILE_INDEX_NONE UINT32_MAX

/* Buckets only narrow candidates. The full predicate still checks image,
 * level, layers, preservation and tile overlap after hash collisions. */
static unsigned
tile_bucket(const struct hk_sync_footprint *fp)
{
   const uintptr_t image = (uintptr_t)fp->image;
   return ((image >> 4) ^ (image >> 12) ^ (fp->level * 0x9e3779b9u)) &
          (TILE_BUCKETS - 1);
}

struct recent_tile_index {
   uint16_t head[TILE_BUCKETS];
   uint16_t next[HK_SYNC_RECENT_FOOTPRINTS];
};

/* Tile hazards: see hk_sync.h. */
static bool
footprints_share_tile(const struct hk_sync_footprint *a,
                      const struct hk_sync_footprint *b)
{
   return (a->preserve || b->preserve) &&
          a->image == b->image && a->level == b->level &&
          a->layer0 < b->layer1 && b->layer0 < a->layer1 &&
          a->cx0 <= b->cx1 && b->cx0 <= a->cx1 && a->cy0 <= b->cy1 &&
          b->cy0 <= a->cy1;
}

void
hk_sync_pass_footprint(struct hk_cmd_buffer *cmd, const void *image,
                       uint32_t level, uint32_t layer0, uint32_t layer1,
                       uint32_t level_width, uint32_t level_height,
                       VkRect2D area, bool preserve_layers)
{
   struct hk_sync_state *st = &cmd->sync;
   const int64_t w = level_width, h = level_height;
   const int64_t x0 = MAX2((int64_t)area.offset.x, 0);
   const int64_t y0 = MAX2((int64_t)area.offset.y, 0);
   const int64_t x1 = MIN2((int64_t)area.offset.x + area.extent.width, w);
   const int64_t y1 = MIN2((int64_t)area.offset.y + area.extent.height, h);
   const int64_t cell = 1 << HK_SYNC_TILE_CELL_LOG2;

   if (!image || layer1 <= layer0 || x1 <= x0 || y1 <= y0)
      return;

   /* Retain complete writes too: a later sparse multiview pass may reload
    * and store these layers even though its API writes are disjoint. */
   const bool preserve = preserve_layers || x0 % cell || y0 % cell ||
                         (x1 % cell && x1 != w) ||
                         (y1 % cell && y1 != h);

   assert(st->pass_fp_count < HK_SYNC_MAX_PASS_FOOTPRINTS);
   if (st->pass_fp_count >= HK_SYNC_MAX_PASS_FOOTPRINTS)
      return;

   st->pass_fp[st->pass_fp_count++] = (struct hk_sync_footprint){
      .preserve = preserve,
      .image = image,
      .level = level,
      .layer0 = layer0,
      .layer1 = layer1,
      .cx0 = (int32_t)(x0 >> HK_SYNC_TILE_CELL_LOG2),
      .cx1 = (int32_t)((x1 - 1) >> HK_SYNC_TILE_CELL_LOG2),
      .cy0 = (int32_t)(y0 >> HK_SYNC_TILE_CELL_LOG2),
      .cy1 = (int32_t)((y1 - 1) >> HK_SYNC_TILE_CELL_LOG2),
   };
}

void
hk_sync_pass_footprints_done(struct hk_cmd_buffer *cmd)
{
   struct hk_sync_state *st = &cmd->sync;
   struct hk_cs *wait[HK_SYNC_NUM_SQ] = {0};
   util_dynarray_foreach(&cmd->sync_pass_records, struct hk_sync_pass_record, rec) {
      if (!rec->last)
         continue;
      const unsigned sq = hk_sync_sq(rec->last);
      if (wait[sq] && rec->last->sync.seq <= wait[sq]->sync.seq)
         continue;
      for (unsigned i = 0; i < st->pass_fp_count; ++i) {
         if (footprints_share_tile(&rec->fp, &st->pass_fp[i])) {
            wait[sq] = rec->last;
            break;
         }
      }
   }
   memcpy(st->pass_tile_wait, wait, sizeof(wait));
}

/* Record the footprints of the pass instance that ends, with its streams. */
static void
pass_footprints_commit(struct hk_cmd_buffer *cmd)
{
   struct hk_sync_state *st = &cmd->sync;
   struct hk_cs *first = st->pass_tile_first;
   struct hk_cs *last = st->pass_tile_last;

   if (st->pass_fp_count && first && last) {
      for (unsigned i = 0; i < st->pass_fp_count; ++i) {
         const struct hk_sync_pass_record rec = {
            .fp = st->pass_fp[i],
            .first = first,
            .last = last,
         };
         util_dynarray_append(&cmd->sync_pass_records, rec);
      }
   }

   st->pass_fp_count = 0;
   memset(st->pass_tile_wait, 0, sizeof(st->pass_tile_wait));
   st->pass_tile_first = st->pass_tile_last = NULL;
}

static void
records_replace_stream(struct hk_cmd_buffer *cmd, struct hk_cs *from,
                       struct hk_cs *to)
{
   util_dynarray_foreach(&cmd->sync_pass_records, struct hk_sync_pass_record,
                         rec) {
      if (rec->first == from)
         rec->first = to;
      if (rec->last == from)
         rec->last = to;
   }
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      if (cmd->sync.pass_tile_wait[sq] == from)
         cmd->sync.pass_tile_wait[sq] = to;
   }
   if (cmd->sync.pass_tile_first == from)
      cmd->sync.pass_tile_first = to;
   if (cmd->sync.pass_tile_last == from)
      cmd->sync.pass_tile_last = to;
   if (cmd->sync.suspended_last == from)
      cmd->sync.suspended_last = to;
}

/*
 * Stage classes. Graphics stages are the whole render pipeline; the driver
 * also runs geometry, tessellation and query bookkeeping for a render as
 * compute streams, and those are graphics work too (kind GFX on the compute
 * subqueue). Indirect parameter and conditional-rendering reads happen for
 * both draws and dispatches. Unknown bits are treated as every class.
 */
#define HK_SYNC_GFX_STAGES                                                     \
   (VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |                                     \
    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |                                    \
    VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |                      \
    VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT |                   \
    VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |                                  \
    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |                                  \
    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |                             \
    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |                              \
    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |                          \
    VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |                                     \
    VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT |                           \
    VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR |             \
    VK_PIPELINE_STAGE_2_FRAGMENT_DENSITY_PROCESS_BIT_EXT |                     \
    VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |                                  \
    VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |                                  \
    VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT |                                      \
    VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |                           \
    VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT |                        \
    VK_PIPELINE_STAGE_2_SUBPASS_SHADER_BIT_HUAWEI |                            \
    VK_PIPELINE_STAGE_2_INVOCATION_MASK_BIT_HUAWEI |                           \
    VK_PIPELINE_STAGE_2_CLUSTER_CULLING_SHADER_BIT_HUAWEI)

#define HK_SYNC_COMPUTE_STAGES                                                 \
   (VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |                                   \
    VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |                           \
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |                 \
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_COPY_BIT_KHR |                  \
    VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT)

#define HK_SYNC_BOTH_STAGES                                                    \
   (VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |                                    \
    VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT)

#define HK_SYNC_XFER_STAGES                                                    \
   (VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT |      \
    VK_PIPELINE_STAGE_2_BLIT_BIT | VK_PIPELINE_STAGE_2_RESOLVE_BIT |           \
    VK_PIPELINE_STAGE_2_CLEAR_BIT)

/* Host work is not GPU work. */
#define HK_SYNC_NO_STAGES (VK_PIPELINE_STAGE_2_HOST_BIT)

/* Graphics stages that only fragment work and attachment accesses execute
 * in: a destination scope restricted to them orders a render's fragment
 * half alone. */
#define HK_SYNC_FRAG_STAGES                                                    \
   (VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |                                  \
    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |                             \
    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |                              \
    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT)

static uint8_t
stage_kinds(VkPipelineStageFlags2 stages, bool dst)
{
   uint8_t kinds = 0;

   if (stages & VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
      return HK_SYNC_ALL;
   if (dst && (stages & HK_SYNC_GFX_STAGES) &&
       !(stages & HK_SYNC_GFX_STAGES & ~HK_SYNC_FRAG_STAGES))
      kinds |= HK_SYNC_GFX_FRAG;
   else if (stages & HK_SYNC_GFX_STAGES)
      kinds |= HK_SYNC_GFX;
   if (stages & HK_SYNC_COMPUTE_STAGES)
      kinds |= HK_SYNC_COMPUTE;
   if (stages & HK_SYNC_BOTH_STAGES)
      kinds |= HK_SYNC_GFX | HK_SYNC_COMPUTE;
   if (stages & HK_SYNC_XFER_STAGES)
      kinds |= HK_SYNC_XFER;

   const VkPipelineStageFlags2 known =
      HK_SYNC_GFX_STAGES | HK_SYNC_COMPUTE_STAGES | HK_SYNC_BOTH_STAGES |
      HK_SYNC_XFER_STAGES | HK_SYNC_NO_STAGES |
      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT |
      VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
   if (stages & ~known)
      kinds = HK_SYNC_ALL;
   if (kinds & HK_SYNC_GFX)
      kinds &= ~HK_SYNC_GFX_FRAG;

   return kinds;
}

uint8_t
hk_sync_kinds_src(VkPipelineStageFlags2 stages)
{
   /* Vulkan 1.3, 7.1.2: in the first scope BOTTOM_OF_PIPE is ALL_COMMANDS
    * and TOP_OF_PIPE is NONE. */
   if (stages & VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT)
      return HK_SYNC_ALL;
   return stage_kinds(stages & ~VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, false);
}

uint8_t
hk_sync_kinds_dst(VkPipelineStageFlags2 stages)
{
   /* And in the second scope TOP_OF_PIPE is ALL_COMMANDS, BOTTOM_OF_PIPE is
    * NONE. */
   if (stages & VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT)
      return HK_SYNC_ALL;
   return stage_kinds(stages & ~VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, true);
}

void
hk_sync_dependency_kinds(const VkDependencyInfo *info, bool image_class,
                         uint8_t *src, uint8_t *dst)
{
   *src = 0;
   *dst = 0;

   for (uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
      *src |= hk_sync_kinds_src(info->pMemoryBarriers[i].srcStageMask);
      *dst |= hk_sync_kinds_dst(info->pMemoryBarriers[i].dstStageMask);
   }

   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) {
      *src |= hk_sync_kinds_src(info->pBufferMemoryBarriers[i].srcStageMask);
      *dst |= hk_sync_kinds_dst(info->pBufferMemoryBarriers[i].dstStageMask);
   }

   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
      *src |= hk_sync_kinds_src(info->pImageMemoryBarriers[i].srcStageMask);
      /* Graphics work accesses an image in the fragment and attachment
       * stages or through a shader that reads it (HK_SYNC_GFX_IMG). */
      uint8_t d = hk_sync_kinds_dst(info->pImageMemoryBarriers[i].dstStageMask);
      if (image_class && (d & HK_SYNC_GFX))
         d = (d & ~HK_SYNC_GFX) | HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG;
      *dst |= d;
   }
}

static struct hk_cs *previous_on_sq(struct hk_cmd_buffer *cmd,
                                    struct hk_cs *cs);

void
hk_sync_src_dep(struct hk_cmd_buffer *cmd, uint8_t src_kinds,
                struct hk_cs *self, struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      dep[sq] = (struct hk_sync_dep){.prior = src_kinds};
      u_foreach_bit(k, src_kinds)
         dep[sq].cut = hk_sync_later(dep[sq].cut, cmd->sync.last[k][sq]);

      /* "Every earlier stream" for a stream that is itself the latest one
       * on the subqueue is every stream before it. */
      if (self && dep[sq].cut == self)
         dep[sq].cut = previous_on_sq(cmd, self);
   }
}

static void hk_sync_stream_wait(struct hk_cs *cs,
                                const struct hk_sync_dep dep[HK_SYNC_NUM_SQ]);

/* G17P transfer writes: see hk_sync.h. Drop the notes of streams positioned
 * at or before seq on the compute subqueue: a dependency orders them. */
static void
xfer_forget_upto(struct hk_sync_state *st, uint32_t seq)
{
   unsigned kept = 0;
   for (unsigned i = 0; i < st->xfer_write_count; ++i) {
      if (st->xfer_writes[i].cs->sync.seq > seq)
         st->xfer_writes[kept++] = st->xfer_writes[i];
   }
   if (kept != st->xfer_write_count) {
      st->xfer_write_count = kept;
      st->xfer_version++;
   }
   if (st->xfer_any && st->xfer_any->sync.seq <= seq) {
      st->xfer_any = NULL;
      st->xfer_version++;
   }
}

void
hk_sync_note_xfer_write(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_sync_state *st = &cmd->sync;
   const struct hk_sync_xfer_dst *dst = &st->xfer_dst;
   assert(cs->type == HK_CS_CDM);

   if (dst->valid && !dst->device_address) {
      /* The common case: the same transfer launching again. */
      for (unsigned i = 0; i < st->xfer_write_count; ++i) {
         struct hk_sync_xfer_write *w = &st->xfer_writes[i];
         if (w->cs == cs && dst->lo <= w->hi && w->lo <= dst->hi) {
            if (dst->lo < w->lo || dst->hi > w->hi) {
               w->lo = MIN2(w->lo, dst->lo);
               w->hi = MAX2(w->hi, dst->hi);
               st->xfer_version++;
            }
            return;
         }
      }

      if (st->xfer_write_count < HK_SYNC_XFER_WRITES) {
         st->xfer_writes[st->xfer_write_count++] = (struct hk_sync_xfer_write){
            .lo = dst->lo,
            .hi = dst->hi,
            .cs = cs,
         };
         st->xfer_version++;
         return;
      }
   }

   /* Unknown, reachable through a device address, or no room: any graphics
    * work may access it. */
   struct hk_cs *any = hk_sync_later(st->xfer_any, cs);
   if (any != st->xfer_any) {
      st->xfer_any = any;
      st->xfer_version++;
   }
}

struct hk_cs *
hk_sync_xfer_match(const struct hk_cmd_buffer *cmd, uint64_t lo, uint64_t hi,
                   struct hk_cs *best)
{
   const struct hk_sync_state *st = &cmd->sync;
   for (unsigned i = 0; i < st->xfer_write_count; ++i) {
      const struct hk_sync_xfer_write *w = &st->xfer_writes[i];
      if (lo < w->hi && w->lo < hi)
         best = hk_sync_later(best, w->cs);
   }
   return hk_sync_later(best, st->xfer_any);
}

struct hk_cs *
hk_sync_xfer_match_all(const struct hk_cmd_buffer *cmd, struct hk_cs *best)
{
   const struct hk_sync_state *st = &cmd->sync;
   for (unsigned i = 0; i < st->xfer_write_count; ++i)
      best = hk_sync_later(best, st->xfer_writes[i].cs);
   return hk_sync_later(best, st->xfer_any);
}

void
hk_sync_order_xfer(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_sync_state *st = &cmd->sync;
   assert(cs->type == HK_CS_CDM);

   /* Graphics streams created from now on, then the ones that exist: the
    * render of the open pass instance, both halves, and its pre-graphics
    * work, which the transfer prefix precedes. */
   hk_sync_after_stream(cmd, cs, HK_SYNC_GFX);
   if (st->in_pass) {
      const struct hk_sync_dep dep[HK_SYNC_NUM_SQ] = {{0}, {.cut = cs}};
      if (cmd->current_cs.gfx)
         hk_sync_stream_wait(cmd->current_cs.gfx, dep);
      if (cmd->current_cs.pre_gfx)
         hk_sync_stream_wait(cmd->current_cs.pre_gfx, dep);
   }
   xfer_forget_upto(st, cs->sync.seq);
}

static void
barrier_source_dependencies(struct hk_cmd_buffer *cmd, uint8_t src_kinds,
                            struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   hk_sync_src_dep(cmd, src_kinds, NULL, dep);

   struct hk_sync_dep pending[HK_SYNC_NUM_SQ];
   req_for_kinds(cmd, src_kinds, pending);
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
      dep_max(&dep[sq], &pending[sq]);
}

static void note_last(struct hk_cmd_buffer *cmd, struct hk_cs *cs);

/* The latest HK_SYNC_GFX_MEM stream at or before a cut: a prefix wait on it
 * covers every such stream the cut covers. */
static struct hk_cs *
memory_cut(struct hk_cs *cut)
{
   return !cut || (cut->sync.kind & HK_SYNC_GFX_MEM) ? cut
                                                     : cut->sync.prev_mem;
}

void
hk_sync_note_memory_write(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   if (cs->sync.kind & HK_SYNC_GFX_MEM)
      return;
   cs->sync.kind |= HK_SYNC_GFX_MEM;
   note_last(cmd, cs);
}

void
hk_sync_barrier(struct hk_cmd_buffer *cmd, uint8_t src_kinds,
                uint8_t dst_kinds)
{
   cmd->sync.epoch++;
   if (!src_kinds || !dst_kinds)
      return;

   /* A dependency from the transfer stage into any graphics stage orders
    * every noted transfer write as the application asked. */
   if ((src_kinds & HK_SYNC_XFER) &&
       (dst_kinds & (HK_SYNC_GFX | HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG)) &&
       hk_sync_xfer_pending(&cmd->sync))
      xfer_forget_upto(&cmd->sync, UINT32_MAX);

   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   barrier_source_dependencies(cmd, src_kinds, dep);

   /* M5 and G17P: of the renders this dependency orders pre-rasterization
    * work after, graphics or transfer, only those that wrote memory other
    * than their attachments (HK_SYNC_GFX_MEM) can hand that work anything it
    * reads without an image. Order the vertex half (HK_SYNC_GFX) after the
    * latest of them; the fragment half and image-reading vertex work keep the
    * whole dependency. The compute subqueue's requirement is unchanged: the
    * kernel orders both halves after compute work either way. */
   const uint8_t render_kinds = HK_SYNC_GFX | HK_SYNC_XFER;
   const enum agx_chip chip = hk_cmd_buffer_device(cmd)->dev.chip;
   if ((src_kinds & render_kinds) && (dst_kinds & HK_SYNC_GFX) &&
       (agx_chip_g17_m5(chip) || chip == AGX_CHIP_G17P)) {
      struct hk_sync_dep vtx = dep[HK_SYNC_VDM];
      vtx.cut = memory_cut(vtx.cut);
      if (vtx.prior & render_kinds)
         vtx.prior = (vtx.prior & ~render_kinds) | HK_SYNC_GFX_MEM;
      const unsigned gfx = util_logbase2(HK_SYNC_GFX);
      dep_max(&cmd->sync.req[gfx][HK_SYNC_VDM], &vtx);
      dep_max(&cmd->sync.req[gfx][HK_SYNC_CDM], &dep[HK_SYNC_CDM]);
      dst_kinds = (dst_kinds & ~HK_SYNC_GFX) | HK_SYNC_GFX_FRAG |
                  HK_SYNC_GFX_IMG;
   }

   u_foreach_bit(d, dst_kinds) {
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
         dep_max(&cmd->sync.req[d][sq], &dep[sq]);
   }

   cmd->sync.cs_check_pending = true;
}

void
hk_sync_pass_barrier(struct hk_cmd_buffer *cmd, uint8_t src_kinds,
                     uint8_t dst_kinds)
{
   struct hk_sync_state *st = &cmd->sync;
   assert(st->in_pass);
   st->epoch++;

   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   barrier_source_dependencies(cmd, src_kinds, dep);
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      /* Neither scope includes work before the pass instance. */
      dep[sq].prior = 0;
      if (dep[sq].cut && dep[sq].cut->sync.seq <= st->pass_start_seq)
         dep[sq].cut = NULL;
   }

   u_foreach_bit(d, dst_kinds) {
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
         dep_max(&st->pass_req[d][sq], &dep[sq]);
   }
}

static void
stream_wait(struct hk_cs *cs, const struct hk_sync_dep dep[HK_SYNC_NUM_SQ],
            bool ta)
{
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      /* A stream never waits on itself: a cut at its own position means the
       * work before it inside the stream, which the stream orders itself. */
      if (dep[sq].cut == cs) {
         cs->sync.dep[sq].prior |= dep[sq].prior;
         if (ta)
            cs->sync.dep_ta[sq].prior |= dep[sq].prior;
         continue;
      }
      dep_max(&cs->sync.dep[sq], &dep[sq]);
      if (ta)
         dep_max(&cs->sync.dep_ta[sq], &dep[sq]);
   }
}

/* Raise a stream's dependencies. */
static void
hk_sync_stream_wait(struct hk_cs *cs,
                    const struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   stream_wait(cs, dep, true);
}

/* A requirement of a render's fragment half alone. */
static void
stream_wait_frag(struct hk_cs *cs, const struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   stream_wait(cs, dep, false);
}

/* The latest stream on subqueue sq positioned before cs in the list. Stream
 * positions follow sync.seq (see hk_sync_stream_created). */
static struct hk_cs *
latest_before(struct hk_cmd_buffer *cmd, unsigned sq, const struct hk_cs *cs)
{
   struct hk_cs *best = NULL;
   list_for_each_entry(struct hk_cs, it, &cmd->control_streams, node) {
      if (hk_sync_sq(it) == sq && it->sync.seq < cs->sync.seq &&
          (!best || it->sync.seq > best->sync.seq))
         best = it;
   }
   return best;
}

static void
hk_sync_stream_wait_all(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   hk_sync_src_dep(cmd, HK_SYNC_ALL, cs, dep);

   /* Every stream positioned before cs. A stream can exist before streams
    * positioned after it -- a pass's pre-graphics stream precedes its render,
    * and hk_sync_heap_used reorders already-created streams -- and waiting
    * on those would be a dependency on later commands of the submission. */
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      if (dep[sq].cut && dep[sq].cut->sync.seq >= cs->sync.seq)
         dep[sq].cut = latest_before(cmd, sq, cs);
   }

   hk_sync_stream_wait(cs, dep);
}

void
hk_sync_vertex_images(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   /* Requirements only grow when the epoch advances. */
   if (cs->sync.vtx_images && cs->sync.vtx_images_epoch == cmd->sync.epoch)
      return;

   cs->sync.vtx_images = true;
   cs->sync.vtx_images_epoch = cmd->sync.epoch;
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   req_for_mask(cmd, HK_SYNC_GFX_IMG, dep);
   hk_sync_stream_wait(cs, dep);
}

void
hk_sync_after_stream(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                     uint8_t dst_kinds)
{
   const unsigned sq = hk_sync_sq(cs);
   const struct hk_sync_dep dep = {.cut = cs};

   u_foreach_bit(d, dst_kinds)
      dep_max(&cmd->sync.req[d][sq], &dep);

   cmd->sync.cs_check_pending = true;
   cmd->sync.epoch++;
}

static uint8_t
stream_kind(struct hk_cmd_buffer *cmd, struct hk_cs **slot, bool compute)
{
   if (cmd->sync.kind_override)
      return cmd->sync.kind_override;
   if (cmd->sync.xfer_depth)
      return HK_SYNC_XFER;
   if (!compute || slot == &cmd->current_cs.pre_gfx ||
       slot == &cmd->current_cs.post_gfx)
      return HK_SYNC_GFX;
   return HK_SYNC_COMPUTE;
}

static void
note_last(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   const unsigned sq = hk_sync_sq(cs);

   u_foreach_bit(k, cs->sync.kind)
      cmd->sync.last[k][sq] = hk_sync_later(cmd->sync.last[k][sq], cs);
   cmd->sync.last_any[sq] = hk_sync_later(cmd->sync.last_any[sq], cs);
}

void
hk_sync_stream_created(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       struct hk_cs **slot, bool compute)
{
   struct hk_sync_state *st = &cmd->sync;
   struct hk_cs *gfx = cmd->current_cs.gfx;
   const bool pre_gfx = compute && slot == &cmd->current_cs.pre_gfx && gfx &&
                        gfx != cs;
   /* Unit-test fixtures have no device. */
   const struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (unlikely(dev && dev->frame_anatomy)) {
      cs->anatomy_api_barriers = cmd->anatomy_bar.count;
      cs->anatomy_api_mem = cmd->anatomy_bar.mem;
      cs->anatomy_api_buf = cmd->anatomy_bar.buf;
      cs->anatomy_api_img = cmd->anatomy_bar.img;
      cs->anatomy_api_src = cmd->anatomy_bar.src;
      cs->anatomy_api_dst = cmd->anatomy_bar.dst;
      cs->anatomy_api_src_access = cmd->anatomy_bar.src_access;
      cs->anatomy_api_dst_access = cmd->anatomy_bar.dst_access;
      memset(&cmd->anatomy_bar, 0, sizeof(cmd->anatomy_bar));
   }
   const bool post_gfx = compute && slot == &cmd->current_cs.post_gfx;
   st->epoch++;

   /* Recording order that matches list order. A render reserves the number
    * below its own for the pre-graphics stream inserted ahead of it. */
   if (!compute) {
      st->next_seq += 2;
      cs->sync.seq = st->next_seq;
   } else if (pre_gfx) {
      cs->sync.seq = gfx->sync.seq - 1;
   } else {
      cs->sync.seq = ++st->next_seq;
   }

   cs->sync.kind = stream_kind(cmd, slot, compute);
   /* A transfer render writes its attachments, like any render; the draws
    * it records mark it if they write other memory
    * (hk_m5_note_render_resources). */
   const bool xfer_render = !compute && cs->sync.kind == HK_SYNC_XFER;
   if ((cs->sync.kind != HK_SYNC_GFX && !xfer_render) ||
       slot == &cmd->current_cs.pre_gfx || slot == &cmd->current_cs.post_gfx)
      cs->sync.kind |= HK_SYNC_GFX_MEM;
   /* Attachment stores (clears, resolves) reach memory other resources can
    * read whenever the attachments can alias them (hk_begin_rendering()),
    * whether or not the stream records a draw. */
   if (!compute && cmd->state.gfx.render.m5_attachments_reach_memory)
      cs->sync.kind |= HK_SYNC_GFX_MEM;
   cs->sync.prev_mem =
      st->last[util_logbase2(HK_SYNC_GFX_MEM)][hk_sync_sq(cs)];

   /* The recorded requirement for this kind of work. A render's vertex half
    * needs only the part whose destination is not fragment-only. */
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   /* Software vertex work is outside an in-pass fragment-only destination.
    * Waiting on the current render here would create a cycle after a kept
    * local barrier. Preserve global requirements, though: internal producers
    * such as attachment decompression must still follow external writes. */
   if (compute && slot == &cmd->current_cs.pre_gfx) {
      /* The pre-graphics work of a pipeline without software
       * pre-rasterization stages (draw parameter patching, index rewriting,
       * statistics) reads buffers in the draw-indirect and vertex-input
       * stages alone: image barriers (HK_SYNC_GFX_IMG) do not order it. */
      uint8_t classes = full_kinds(cs->sync.kind);
      if (!cmd->state.gfx.shaders[MESA_SHADER_TESS_CTRL] &&
          !cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL] &&
          !cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] &&
          !cmd->state.gfx.shaders[MESA_SHADER_TASK] &&
          !cmd->state.gfx.shaders[MESA_SHADER_MESH])
         classes &= ~HK_SYNC_GFX_IMG;
      req_for_kinds_ta(cmd, cs->sync.kind, dep);
      u_foreach_bit(k, classes) {
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
            dep_max(&dep[sq], &st->req[k][sq]);
      }
   } else {
      req_for_kinds(cmd, cs->sync.kind, dep);
   }
   if (!compute) {
      struct hk_sync_dep ta[HK_SYNC_NUM_SQ];
      req_for_kinds_ta(cmd, cs->sync.kind, ta);
      stream_wait_frag(cs, dep);
      hk_sync_stream_wait(cs, ta);
   } else {
      /* A render that continues an earlier instance across an ordered
       * dependency (hk_cs::m5_ordered) holds that instance's work, which the
       * dependency's source scope names. Its pre-graphics stream sits before
       * it and cannot wait on it. That instance wrote no memory but its
       * attachments and the continuation's own writes end the render first
       * (hk_m5_ordered_refusal()), so what the pre-graphics work reads is
       * already final; it waits on everything positioned before it. */
      if (pre_gfx && gfx->m5_ordered) {
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
            if (dep[sq].cut && dep[sq].cut->sync.seq >= cs->sync.seq)
               dep[sq].cut = latest_before(cmd, sq, cs);
         }
      }
      hk_sync_stream_wait(cs, dep);
   }

   if (st->internal_active)
      hk_sync_stream_wait(cs, st->internal);

   if (!compute) {
      /* Counter initialization/resumption can open the pre-graphics stream
       * before the draw creates its render stream. It is still a producer
       * for this render, just like a pre-graphics stream created afterwards. */
      if (cmd->current_cs.pre_gfx) {
         const struct hk_sync_dep pre[HK_SYNC_NUM_SQ] = {
            {0}, {.cut = cmd->current_cs.pre_gfx}};
         hk_sync_stream_wait(cs, pre);
      }
      /* Tile hazard: the pass instance shares a partly covered tile with an
       * earlier render that nothing else orders it after. */
      if (st->in_pass) {
         struct hk_sync_dep tile[HK_SYNC_NUM_SQ] = {0};
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
            tile[sq].cut = st->pass_tile_wait[sq];
         stream_wait_frag(cs, tile);
      }
      if (st->in_pass && (st->pass_tile_last || st->resume_pending)) {
         struct hk_sync_dep cont[HK_SYNC_NUM_SQ] = {0};
         if (st->pass_tile_last)
            cont[hk_sync_sq(st->pass_tile_last)].cut = st->pass_tile_last;
         /* An empty suspended instance may have been replaced by compute
          * clears (hk_optimize_empty_vdm()), so graphics work on either
          * subqueue can be a producer of what this instance continues. */
         if (st->resume_pending) {
            cont[HK_SYNC_VDM].prior = HK_SYNC_GFX;
            cont[HK_SYNC_CDM].prior = HK_SYNC_GFX;
         }
         hk_sync_stream_wait(cs, cont);
         st->resume_pending = false;
      }
      if (st->in_pass) {
         if (!st->pass_vdm)
            st->pass_vdm = cs;
         if (!st->pass_tile_first)
            st->pass_tile_first = cs;
         st->pass_tile_last = cs;
      }

      /* The open compute stream sits before this render. It stays open
       * across the pass unless the render waits on it: then later compute
       * must not extend what the render waits for. */
      if (slot == &cmd->current_cs.gfx && cmd->current_cs.cs &&
          cs->sync.dep[HK_SYNC_CDM].cut == cmd->current_cs.cs)
         hk_cmd_buffer_end_compute(cmd);
   } else if (pre_gfx) {
      /* The render consumes what its pre-graphics stream produces. */
      const struct hk_sync_dep pre[HK_SYNC_NUM_SQ] = {{0}, {.cut = cs}};
      hk_sync_stream_wait(gfx, pre);
   } else if (post_gfx) {
      /* Query availability and heap release after the render finishes. The
       * render may already have ended, so name the latest one. */
      const struct hk_sync_dep post[HK_SYNC_NUM_SQ] = {
         {.cut = gfx ? gfx : st->last_any[HK_SYNC_VDM]}, {0}};
      hk_sync_stream_wait(cs, post);
   }

   /* The geometry heap is one device-wide bump allocator (hk_cmd_draw.c,
    * hk_heap): a pass that uses it must not overlap any earlier stream. */
   if (st->heap_pass && (!compute || pre_gfx || post_gfx))
      hk_sync_stream_wait_all(cmd, cs);

   note_last(cmd, cs);

   if (slot == &cmd->current_cs.cs)
      st->cs_check_pending = false;

   /* A top-of-pipe timestamp waiting for a command to ride on. */
   if (st->pending_start_ts.handle) {
      cs->timestamp.start = st->pending_start_ts;
      st->pending_start_ts = (struct agx_timestamp_req){0};
   }
}

/* Whether the open general compute stream may receive work of the given
 * kinds: every stream it would have to wait on must precede it. */
static bool
cs_reusable(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t kinds)
{
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   req_for_kinds(cmd, kinds, dep);

   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      if (dep[sq].cut && dep[sq].cut->sync.seq > cs->sync.seq)
         return false;
   }

   return true;
}

/* Whether the compute stream cs already waits for the render r: directly,
 * or through the compute prefix it waits on, each stream of which completes
 * only after the renders it waits on. */
static bool
cdm_waits_for_render(struct hk_cmd_buffer *cmd, const struct hk_cs *cs,
                     const struct hk_cs *r)
{
   const struct hk_cs *v = cs->sync.dep[HK_SYNC_VDM].cut;
   if (v && v->sync.seq >= r->sync.seq)
      return true;

   const struct hk_cs *prefix = cs->sync.dep[HK_SYNC_CDM].cut;
   if (!prefix || prefix == cs)
      return false;

   list_for_each_entry(struct hk_cs, it, &cmd->control_streams, node) {
      if (it->type != HK_CS_CDM || it == cs ||
          it->sync.seq > prefix->sync.seq)
         continue;
      v = it->sync.dep[HK_SYNC_VDM].cut;
      if (v && v->sync.seq >= r->sync.seq)
         return true;
   }
   return false;
}

/* M5: whether a requirement on the render subqueue would make the work a
 * compute stream already holds wait for a render it does not wait for yet.
 * A requirement recorded after that work (a barrier between two dispatches)
 * has none of it in its second scope, so the work keeps the requirements it
 * was recorded under and only later work takes the new one. */
static bool
cdm_raise_delays_work(struct hk_cmd_buffer *cmd, const struct hk_cs *cs,
                      const struct hk_cs *r)
{
   return r && cs->stats.cmds &&
          agx_chip_g17_m5(hk_cmd_buffer_device(cmd)->dev.chip) &&
          !cdm_waits_for_render(cmd, cs, r);
}

bool
hk_sync_merge_delays(struct hk_cmd_buffer *cmd, const struct hk_cs *into,
                     const struct hk_cs *from)
{
   return cdm_raise_delays_work(cmd, into, from->sync.dep[HK_SYNC_VDM].cut);
}

void
hk_sync_check_cs(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.cs;
   if (!cs || cmd->sync.pin_cs)
      return;

   const uint8_t kinds = stream_kind(cmd, &cmd->current_cs.cs, true);

   if (!cs_reusable(cmd, cs, kinds)) {
      hk_cmd_buffer_end_compute(cmd);
      return;
   }

   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   req_for_kinds(cmd, kinds, dep);

   /* The new work goes into a stream of its own, created under the whole
    * requirement; a first scope that names compute work orders it after
    * this stream through last[]. */
   if (cdm_raise_delays_work(cmd, cs, dep[HK_SYNC_VDM].cut)) {
      hk_cmd_buffer_end_compute(cmd);
      return;
   }

   /* The stream absorbs the requirement of the work it now carries. */
   hk_sync_stream_wait(cs, dep);
   cs->sync.kind |= kinds;
   if (cs->sync.kind != HK_SYNC_GFX)
      cs->sync.kind |= HK_SYNC_GFX_MEM;
   note_last(cmd, cs);

   if ((agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev) ||
        agx_chip_g17_m5(hk_cmd_buffer_device(cmd)->dev.chip)) &&
       cmd->sync.pending_start_ts.handle && !cs->timestamp.start.handle &&
       !cs->timestamp.end.handle) {
      cs->timestamp.start = cmd->sync.pending_start_ts;
      cmd->sync.pending_start_ts = (struct agx_timestamp_req){0};
   }
}

struct hk_cs *
hk_sync_get_cs_after(struct hk_cmd_buffer *cmd,
                     const struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   struct hk_cs *cs = cmd->current_cs.cs;

   /* The stream must satisfy the caller's requirement and the recorded one
    * for the kind of work about to be recorded, so that the work itself
    * finds the stream reusable. */
   struct hk_sync_dep full[HK_SYNC_NUM_SQ];
   req_for_kinds(cmd, stream_kind(cmd, &cmd->current_cs.cs, true), full);
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
      dep_max(&full[sq], &dep[sq]);

   if (cs && !cmd->sync.pin_cs) {
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
         if (full[sq].cut && full[sq].cut->sync.seq > cs->sync.seq) {
            hk_cmd_buffer_end_compute(cmd);
            break;
         }
      }
   }

   cs = hk_cmd_buffer_get_cs(cmd, true);
   if (cs)
      hk_sync_stream_wait(cs, full);
   return cs;
}

/* A standalone G17P timestamp still needs an executable command carrier.
 * Only an otherwise empty timestamp producer uses this owned scratch store;
 * ordinary compute and barriers keep their existing streams unchanged.
 */
static void
stamp_carrier_launch(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.cs;
   if (!agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev) || !cs ||
       (!cs->timestamp.start.handle && !cs->timestamp.end.handle) ||
       cs->stats.cmds || cs->stats.flushes || cs->imm_writes.size)
      return;

   struct agx_ptr scratch = hk_pool_alloc(cmd, 16, 16);
   if (!scratch.gpu)
      return;
   hk_g17p_meta_fill_addr(cmd, &cmd->current_cs.cs, scratch.gpu, 4, 0);
}

void
hk_sync_end_cs_if_timestamped(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.cs;
   if (!cs)
      return;

   const bool g17p_start =
      (agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev) ||
       agx_chip_g17_m5(hk_cmd_buffer_device(cmd)->dev.chip)) &&
      cs->timestamp.start.handle;
   if (cs->timestamp.end.handle || g17p_start) {
      stamp_carrier_launch(cmd);
      hk_cmd_buffer_end_compute(cmd);
   }
}

/* The stream on the same subqueue that precedes cs in the list. */
static struct hk_cs *
previous_on_sq(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   for (struct list_head *n = cs->node.prev; n != &cmd->control_streams;
        n = n->prev) {
      struct hk_cs *it = list_entry(n, struct hk_cs, node);
      if (it->type == cs->type)
         return it;
   }
   return NULL;
}

static void
replace_pointer(struct hk_cs **slot, struct hk_cs *from, struct hk_cs *to)
{
   if (*slot == from)
      *slot = to;
}

/* Transfer write notes follow a stream that moves (to != NULL) or is removed
 * with nothing before it on its subqueue (to == NULL). */
static void
xfer_replace_stream(struct hk_sync_state *st, struct hk_cs *from,
                    struct hk_cs *to)
{
   unsigned kept = 0;
   for (unsigned i = 0; i < st->xfer_write_count; ++i) {
      struct hk_sync_xfer_write w = st->xfer_writes[i];
      if (w.cs == from)
         w.cs = to;
      if (w.cs)
         st->xfer_writes[kept++] = w;
   }
   st->xfer_write_count = kept;
   if (st->xfer_any == from)
      st->xfer_any = to;
   st->xfer_version++;
}

/* Every reference to cs moves to its predecessor on the subqueue: a prefix
 * that ended at cs now ends just before it. */
void
hk_sync_forget_stream(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_sync_state *st = &cmd->sync;
   struct hk_cs *prev = previous_on_sq(cmd, cs);
   const unsigned sq = hk_sync_sq(cs);
   st->epoch++;

   for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
      replace_pointer(&st->last[k][sq], cs, prev);
      replace_pointer(&st->req[k][sq].cut, cs, prev);
      replace_pointer(&st->pass_req[k][sq].cut, cs, prev);
   }
   replace_pointer(&st->last_any[sq], cs, prev);
   replace_pointer(&st->internal[sq].cut, cs, prev);
   replace_pointer(&st->pass_vdm, cs, NULL);
   xfer_replace_stream(st, cs, prev);
   if (st->pass_tile_first == cs) {
      st->pass_tile_first = st->pass_tile_last = NULL;
   } else if (st->pass_tile_last == cs &&
              (!prev || prev->sync.seq < st->pass_tile_first->sync.seq)) {
      st->pass_tile_last = st->pass_tile_first;
   }
   records_replace_stream(cmd, cs, prev);

   list_for_each_entry(struct hk_cs, it, &cmd->control_streams, node) {
      replace_pointer(&it->sync.dep[sq].cut, cs, prev);
      replace_pointer(&it->sync.dep_ta[sq].cut, cs, prev);
      replace_pointer(&it->sync.prev_mem, cs, cs->sync.prev_mem);
   }

   /* A pre-graphics or continuation stream that was waiting on nothing but
    * the removed stream inherits the removed stream's own requirement. */
   if (prev == NULL) {
      list_for_each_entry(struct hk_cs, it, &cmd->control_streams, node) {
         if (it != cs) {
            it->sync.dep[sq].prior |= cs->sync.dep[sq].prior;
            it->sync.dep_ta[sq].prior |= cs->sync.dep[sq].prior;
         }
      }
   }

   /* A stamp the stream carried is not lost: it becomes pending again. */
   if (cs->timestamp.start.handle && !st->pending_start_ts.handle)
      st->pending_start_ts = cs->timestamp.start;
}

void
hk_sync_replace_render(struct hk_cmd_buffer *cmd, struct hk_cs *from,
                        struct hk_cs *into)
{
   assert(from->type == HK_CS_VDM && into->type == HK_CS_CDM);
   assert(cmd->sync.in_pass);
   cmd->sync.epoch++;
   hk_sync_stream_wait(into, from->sync.dep);
   if (cmd->sync.pass_tile_first == from)
      cmd->sync.pass_tile_first = into;
   if (cmd->sync.pass_tile_last == from)
      cmd->sync.pass_tile_last = into;
}

void
hk_sync_merge(struct hk_cmd_buffer *cmd, struct hk_cs *into,
              struct hk_cs *from)
{
   struct hk_sync_state *st = &cmd->sync;
   const unsigned sq = hk_sync_sq(from);
   assert(sq == hk_sync_sq(into));
   st->epoch++;

   /* A cut at the destination covers its entire earlier subqueue prefix.
    * The destination itself is now ordered by the merged command stream,
    * but earlier commands still need their dependency, including any
    * timestamps the kernel publishes only after command retirement. */
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ], ta[HK_SYNC_NUM_SQ];
   memcpy(dep, from->sync.dep, sizeof(dep));
   memcpy(ta, from->sync.dep_ta, sizeof(ta));
   if (dep[sq].cut == into)
      dep[sq].cut = previous_on_sq(cmd, into);
   if (ta[sq].cut == into)
      ta[sq].cut = previous_on_sq(cmd, into);
   stream_wait_frag(into, dep);
   hk_sync_stream_wait(into, ta);
   into->sync.kind |= from->sync.kind;

   for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
      replace_pointer(&st->last[k][sq], from, into);
      replace_pointer(&st->req[k][sq].cut, from, into);
      replace_pointer(&st->pass_req[k][sq].cut, from, into);
   }
   replace_pointer(&st->last_any[sq], from, into);
   replace_pointer(&st->internal[sq].cut, from, into);
   records_replace_stream(cmd, from, into);
   xfer_replace_stream(st, from, into);

   list_for_each_entry(struct hk_cs, it, &cmd->control_streams, node) {
      if (it != into && it != from) {
         replace_pointer(&it->sync.dep[sq].cut, from, into);
         replace_pointer(&it->sync.dep_ta[sq].cut, from, into);
         replace_pointer(&it->sync.prev_mem, from, into);
      }
   }
}

/*
 * Timestamps. vkCmdWriteTimestamp2 (Vulkan 1.3, 17.5) defines an execution
 * dependency whose first scope is all earlier commands in the given stage
 * and whose second scope is the timestamp write itself. TOP_OF_PIPE in a
 * first scope is NONE, so such a stamp may be taken at any time; it becomes
 * the start stamp of the next command, except on chips whose start stamps
 * latch before a command's waits (start_stamps_ordered()), where it ends the
 * last command like any other stamp. Every other stage orders the stamp
 * after earlier work of its class, and the stamp becomes the end stamp of
 * the last command, which is then ordered after that work.
 *
 * A later stamp with a first scope must not read a lower value than an
 * earlier stamp, since the earlier write is in its scope. A pending
 * top-of-pipe stamp is therefore resolved before any dependent stamp.
 */
/* An end stamp is written when a command completes, so ordering it after
 * earlier work orders that command's completion: for a render, its fragment
 * half alone. */
static void
stamp_wait(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
           const struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   if (cs->type == HK_CS_CDM) {
      const struct hk_device *dev = hk_cmd_buffer_device(cmd);
      if (agx_native_graphics(&dev->dev) && dev->dev.chip == AGX_CHIP_G17P &&
          (dev->dev.params.features & DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS)) {
         /* The logical compute queue already orders these GPU commands by
          * their mandatory kick parents. Adding the timestamp's same-CDM
          * prefix also waits for earlier CPU timestamp publication, which
          * serializes otherwise independent timed dispatches. Keep genuine
          * query-memory requirements already present on the stream, and the
          * timestamp's cross-engine scope, unchanged. Exact priors also carry
          * standalone barrier requirements across submissions; legacy kernels
          * retain the original conservative timestamp prefix.
          */
         struct hk_sync_dep timestamp_dep[HK_SYNC_NUM_SQ];
         memcpy(timestamp_dep, dep, sizeof(timestamp_dep));
         timestamp_dep[HK_SYNC_CDM] = (struct hk_sync_dep){0};
         hk_sync_stream_wait(cs, timestamp_dep);
      } else {
         hk_sync_stream_wait(cs, dep);
      }
   } else {
      stream_wait_frag(cs, dep);
   }
}

/* G17P latches a command's start stamps when the firmware takes the command,
 * before its barriers resolve: a vkCmdResetQueryPool dispatch followed by a
 * full ALL_COMMANDS barrier still overwrites the next render's vertex start
 * stamp most of the time, and dependent start stamps read earlier than the
 * previous end. Only end stamps are ordered after a command's waits, so
 * every stamp rides on a command's end: a top-of-pipe stamp (Vulkan 1.3,
 * 17.5: no first scope) on the end of the last recorded work, which may
 * latch it at any logically later point.
 *
 * M5 latches start stamps the same way: on G17S a top-of-pipe or host stamp
 * after a full ALL_COMMANDS barrier reads a few microseconds before the end
 * stamp written ahead of the barrier. Its end stamps are ordered, so it
 * takes the same end-only path.
 */
static bool
start_stamps_ordered(const struct hk_cmd_buffer *cmd)
{
   const struct hk_device *dev = hk_cmd_buffer_device(cmd);
   return !((agx_native_graphics(&dev->dev) &&
             dev->dev.chip == AGX_CHIP_G17P) ||
            agx_chip_g17_m5(dev->dev.chip));
}

static struct hk_cs *
last_stream(struct hk_cmd_buffer *cmd)
{
   if (list_is_empty(&cmd->control_streams))
      return NULL;
   return list_last_entry(&cmd->control_streams, struct hk_cs, node);
}

static struct hk_cs *
timestamp_only_stream(struct hk_cmd_buffer *cmd,
                      const struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   cmd->sync.kind_override = HK_SYNC_ALL;
   struct hk_cs *cs = hk_sync_get_cs_after(cmd, dep);
   cmd->sync.kind_override = 0;
   return cs;
}

void
hk_sync_flush_pending_timestamp(struct hk_cmd_buffer *cmd)
{
   struct hk_sync_state *st = &cmd->sync;
   if (!st->pending_start_ts.handle)
      return;

   const struct agx_timestamp_req req = st->pending_start_ts;
   st->pending_start_ts = (struct agx_timestamp_req){0};
   st->epoch++;

   struct hk_cs *tail = last_stream(cmd);
   if (tail && !tail->timestamp.end.handle) {
      tail->timestamp.end = req;
      if (tail == cmd->current_cs.cs) {
         stamp_carrier_launch(cmd);
         hk_cmd_buffer_end_compute(cmd);
      }
      return;
   }

   /* Nothing to ride on: a command of its own, ordered after nothing. */
   const struct hk_sync_dep none[HK_SYNC_NUM_SQ] = {{0}, {0}};
   struct hk_cs *cs = timestamp_only_stream(cmd, none);
   if (!cs)
      return;
   cs->timestamp.start = req;
   stamp_carrier_launch(cmd);
   hk_cmd_buffer_end_compute(cmd);
}

void
hk_sync_write_timestamp(struct hk_cmd_buffer *cmd, VkPipelineStageFlags2 stage,
                        struct agx_timestamp_req req)
{
   struct hk_sync_state *st = &cmd->sync;
   const uint8_t src_kinds = hk_sync_kinds_src(stage);
   struct hk_cs *gfx = cmd->current_cs.gfx;
   st->epoch++;

   if (st->in_pass && gfx) {
      /* Inside a render pass instance the stamp belongs to the render.
       * Vertex start for an unordered stamp where start stamps are ordered
       * (start_stamps_ordered()), fragment end otherwise; a second end
       * stamp is copied from the first after the render. */
      if (!src_kinds && !gfx->timestamp.start.handle &&
          start_stamps_ordered(cmd)) {
         gfx->timestamp.start = req;
         return;
      }

      struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
      hk_sync_src_dep(cmd, src_kinds, gfx, dep);
      /* The render can only wait on streams positioned before it. The
       * latest stream of a kind may be this render's own post-graphics
       * stream (an in-pass stamp's copy, an occlusion query's availability
       * write), which itself waits on the render; waiting on it would form a
       * VDM<->CDM cycle. That stream runs after the render anyway, so the
       * stamp's first scope needs nothing from it. */
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
         if (dep[sq].cut && dep[sq].cut != gfx &&
             dep[sq].cut->sync.seq > gfx->sync.seq)
            dep[sq].cut = latest_before(cmd, sq, gfx);
      }
      stamp_wait(cmd, gfx, dep);

      if (gfx->timestamp.end.handle) {
         libagx_copy_timestamp(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_POSTGFX,
                               req.addr, gfx->timestamp.end.addr);
         hk_cmd_note_query_range(cmd, cmd->current_cs.post_gfx,
                                 gfx->timestamp.end.handle,
                                 gfx->timestamp.end.offset_B, sizeof(uint64_t));
         hk_cmd_note_query_range(cmd, cmd->current_cs.post_gfx, req.handle,
                                 req.offset_B, sizeof(uint64_t));
      } else {
         gfx->timestamp.end = req;
      }
      return;
   }

   if (!src_kinds && start_stamps_ordered(cmd)) {
      hk_sync_flush_pending_timestamp(cmd);
      st->pending_start_ts = req;
      /* A query reset or earlier dispatch may have left a reusable stream.
       * Consume the start stamp before recording work into it too, rather
       * than resolving it as an end stamp at the next timestamp write. */
      if (agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev) ||
          agx_chip_g17_m5(hk_cmd_buffer_device(cmd)->dev.chip))
         st->cs_check_pending = true;
      return;
   }

   hk_sync_flush_pending_timestamp(cmd);

   struct hk_cs *tail = last_stream(cmd);
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   hk_sync_src_dep(cmd, src_kinds, tail, dep);

   if (tail && !tail->timestamp.end.handle) {
      tail->timestamp.end = req;
      stamp_wait(cmd, tail, dep);

      /* Later compute would move the stamp past itself; keep the stream
       * boundary so the stamp stays at the recorded position. */
      if (tail == cmd->current_cs.cs) {
         stamp_carrier_launch(cmd);
         hk_cmd_buffer_end_compute(cmd);
      }
      return;
   }

   struct hk_cs *cs = timestamp_only_stream(cmd, dep);
   if (!cs)
      return;
   cs->timestamp.end = req;
   stamp_carrier_launch(cmd);
   hk_cmd_buffer_end_compute(cmd);
}

void
hk_sync_begin_rendering(struct hk_cmd_buffer *cmd, bool resuming)
{
   struct hk_sync_state *st = &cmd->sync;
   st->epoch++;
   st->in_pass = true;
   st->pass_start_seq = st->next_seq;
   memset(st->pass_req, 0, sizeof(st->pass_req));
   st->pass_vdm = NULL;
   st->resume_pending = resuming;
   st->heap_pass = false;
   st->pass_fp_count = 0;
   memset(st->pass_tile_wait, 0, sizeof(st->pass_tile_wait));
   st->pass_tile_first = st->pass_tile_last = NULL;

   /* A resuming instance is the same render pass instance as the suspended
    * one before it: its first stream waits on that instance's last producer,
    * which can be a compute stream on the other subqueue. */
   if (resuming)
      st->pass_tile_last = st->suspended_last;
   st->suspended_last = NULL;
}

void
hk_sync_first_render_stream(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   /* Decided when the stream was created (hk_sync_stream_created); a
    * resumed instance creates its first stream at its first draw, so the
    * decision lives there for both paths. */
   struct hk_cs *open = cmd->current_cs.cs;
   if (open && cs->sync.dep[HK_SYNC_CDM].cut == open)
      hk_cmd_buffer_end_compute(cmd);
}

void
hk_sync_end_rendering(struct hk_cmd_buffer *cmd, bool suspending)
{
   struct hk_sync_state *st = &cmd->sync;
   st->epoch++;
   st->suspended_last = suspending ? st->pass_tile_last : NULL;
   pass_footprints_commit(cmd);
   st->in_pass = false;
   memset(st->pass_req, 0, sizeof(st->pass_req));
   st->pass_vdm = NULL;
   st->resume_pending = false;
   st->heap_pass = false;
}

void
hk_sync_merge_rendering(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_sync_state *st = &cmd->sync;
   assert(!st->in_pass && cs->type == HK_CS_VDM);
   assert(cs == list_last_entry(&cmd->control_streams, struct hk_cs, node));
   st->epoch++;

   /* What a render stream created now would wait on
    * (hk_sync_stream_created), raised on the reopened one. A cut at the
    * stream itself is the work before it, which the stream orders
    * (stream_wait); no cut lies after it, since it is the latest stream. The
    * instance's own footprints were recorded when it ended, with this same
    * stream, so the tile-hazard wait it would compute is on itself. */
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ], ta[HK_SYNC_NUM_SQ];
   req_for_kinds(cmd, cs->sync.kind, dep);
   req_for_kinds_ta(cmd, cs->sync.kind, ta);
   stream_wait_frag(cs, dep);
   hk_sync_stream_wait(cs, ta);
   if (st->internal_active)
      hk_sync_stream_wait(cs, st->internal);
   /* The reopened stream's earlier draws read images in its vertex half,
    * which also runs the continuation's draws. */
   if (cs->sync.vtx_images) {
      struct hk_sync_dep img[HK_SYNC_NUM_SQ];
      req_for_mask(cmd, HK_SYNC_GFX_IMG, img);
      hk_sync_stream_wait(cs, img);
   }

   /* As hk_sync_begin_rendering() followed by the creation of cs: the
    * sequence numbers the instance starts after are those below the
    * stream's own and its pre-graphics slot. */
   st->in_pass = true;
   st->pass_start_seq = cs->sync.seq - 2;
   memset(st->pass_req, 0, sizeof(st->pass_req));
   st->pass_vdm = cs;
   st->resume_pending = false;
   st->heap_pass = false;
   st->pass_fp_count = 0;
   memset(st->pass_tile_wait, 0, sizeof(st->pass_tile_wait));
   st->pass_tile_first = st->pass_tile_last = cs;
   st->suspended_last = NULL;
}

void
hk_sync_heap_used(struct hk_cmd_buffer *cmd)
{
   struct hk_sync_state *st = &cmd->sync;
   st->heap_pass = true;
   st->epoch++;

   /* The streams of this pass already created wait like the ones still to
    * come. */
   if (cmd->current_cs.gfx)
      hk_sync_stream_wait_all(cmd, cmd->current_cs.gfx);
   if (cmd->current_cs.pre_gfx)
      hk_sync_stream_wait_all(cmd, cmd->current_cs.pre_gfx);
   if (cmd->current_cs.post_gfx)
      hk_sync_stream_wait_all(cmd, cmd->current_cs.post_gfx);
}

void
hk_sync_resolve_begin(struct hk_cmd_buffer *cmd)
{
   struct hk_sync_state *st = &cmd->sync;
   st->epoch++;
   st->internal[HK_SYNC_VDM] = (struct hk_sync_dep){
      .cut = st->last_any[HK_SYNC_VDM],
   };
   st->internal[HK_SYNC_CDM] = (struct hk_sync_dep){
      .cut = st->last_any[HK_SYNC_CDM],
   };
   st->internal_active = true;
   st->kind_override = HK_SYNC_GFX;
}

void
hk_sync_resolve_end(struct hk_cmd_buffer *cmd)
{
   struct hk_sync_state *st = &cmd->sync;
   st->epoch++;
   st->internal_active = false;
   st->internal[0] = (struct hk_sync_dep){0};
   st->internal[1] = (struct hk_sync_dep){0};
   st->kind_override = 0;
}

/*
 * Submission. Every stream of the ioctl payload receives its two barrier
 * fields. A value is an index into the commands of that subqueue in this
 * ioctl (1-based), 0 for "everything in earlier ioctls", or NONE.
 *
 * Within one command buffer the cut is exact. Across command buffers of the
 * same submission, a stream's "prior" kinds resolve against the last stream
 * of each kind the earlier command buffers contributed, and a command
 * buffer's tail requirement (its req[] state at the end of recording, which
 * is what every later command must wait on) carries into the next one.
 * Anything that reaches before this submission resolves to 0, unless this
 * queue has never run a stream of that kind on that subqueue, in which case
 * there is nothing to wait for.
 */

/* Resolved requirement: -1 none, 0 earlier ioctls, >0 an index. */
typedef int32_t hk_sync_value;

#define HK_SYNC_NONE (-1)

static inline hk_sync_value
value_max(hk_sync_value a, hk_sync_value b)
{
   return a > b ? a : b;
}

struct submit_state {
   const struct hk_queue *queue;
   /* Global index of the last stream of each kind on each subqueue among the
    * command buffers already processed. */
   hk_sync_value last[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];
   /* Requirement every later stream of each kind carries. */
   hk_sync_value carry[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];
   uint64_t carry_absolute[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];
   uint32_t count[HK_SYNC_NUM_SQ];
};

static hk_sync_value
resolve_prior(const struct submit_state *ss, uint8_t kinds, unsigned sq)
{
   hk_sync_value v = HK_SYNC_NONE;
   u_foreach_bit(k, kinds) {
      if (ss->last[k][sq] > 0)
         v = value_max(v, ss->last[k][sq]);
      else if (ss->queue->sync.ever[sq] & (1u << k))
         v = value_max(v, 0);
   }
   return v;
}

static hk_sync_value
resolve_dep(const struct submit_state *ss, const struct hk_sync_dep *dep,
            unsigned sq)
{
   hk_sync_value v = HK_SYNC_NONE;
   if (dep->cut)
      v = value_max(v, (hk_sync_value)dep->cut->sync.index);
   v = value_max(v, resolve_prior(ss, dep->prior, sq));
   return v;
}

static uint64_t
resolve_absolute(const struct submit_state *ss, const struct hk_sync_dep *dep,
                 unsigned sq)
{
   const uint64_t base = ss->queue->sync.submitted[sq];
   uint64_t value = dep->cut ? base + dep->cut->sync.index : 0;
   u_foreach_bit(k, dep->prior) {
      const uint64_t earlier = ss->last[k][sq] > 0
         ? base + ss->last[k][sq]
         : ss->queue->sync.last_absolute[k][sq];
      value = MAX2(value, earlier);
   }
   return value;
}

/* What resolve_prior() and resolve_absolute() wait for, as kinds: a kind
 * with a latest stream in an earlier command buffer of the submission
 * (ss->last, from cmd->sync.last), else one the queue ran before (legacy:
 * hk_sync_queue::ever) or holds a latest position of (exact:
 * last_absolute). */
uint8_t
hk_sync_queue_kinds(const struct hk_sync_queue *q, unsigned sq)
{
   if (!q->exact_prior)
      return q->ever[sq];

   uint8_t kinds = 0;
   for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
      if (q->last_absolute[k][sq])
         kinds |= BITFIELD_BIT(k);
   }
   return kinds;
}

uint8_t
hk_sync_cmd_kinds(const struct hk_cmd_buffer *cmd, unsigned sq)
{
   uint8_t kinds = 0;
   for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
      if (cmd->sync.last[k][sq])
         kinds |= BITFIELD_BIT(k);
   }
   return kinds;
}

/* Tile hazards across submissions (hk_sync.h). */
static bool
recent_may_run(const struct hk_sync_queue *q, uint64_t completed)
{
   return q->recent_count && q->recent_max > completed;
}

/* Complete writes only need tile-hazard ordering against a pass preserving
 * pixels/layers. Keep the full footprints: a future preserving pass must
 * still wait for complete writes that have not retired.
 */
static bool
submit_preserves_tiles(struct hk_cmd_buffer *const *cmdbufs, unsigned count)
{
   for (unsigned i = 0; i < count; ++i) {
      util_dynarray_foreach(&cmdbufs[i]->sync_pass_records,
                            struct hk_sync_pass_record, rec) {
         if (rec->fp.preserve)
            return true;
      }
   }
   return false;
}

bool
hk_sync_submit_needs_completed(const struct hk_queue *queue,
                               struct hk_cmd_buffer *const *cmdbufs,
                               unsigned count)
{
   const struct hk_sync_queue *q = &queue->sync;
   /* A serialized queue never overlaps two commands, so no earlier
    * submission can share a tile with this one while it runs. */
   if (q->serialize || !q->recent_count)
      return false;
   if (q->precise_tile_evictions && !q->last_preserve &&
       !submit_preserves_tiles(cmdbufs, count))
      return false;
   for (unsigned i = 0; i < count; ++i) {
      if (util_dynarray_num_elements(&cmdbufs[i]->sync_pass_records,
                                     struct hk_sync_pass_record))
         return true;
   }
   return false;
}

static uint8_t
recent_conflict(const struct hk_sync_queue *q, const struct hk_sync_footprint *fp,
                uint64_t completed, const struct recent_tile_index *index)
{
   if (!recent_may_run(q, completed))
      return false;
   if (q->precise_tile_evictions && !fp->preserve && !q->last_preserve)
      return false;
   /* A dropped entry may still be running: assume it shares the tile.
    * When this pass is a complete write, only an evicted preserving pass
    * can conflict. A preserving pass still considers every evicted write.
    */
   const uint64_t evicted = q->precise_tile_evictions && !fp->preserve
      ? q->evicted_preserve : q->evicted;
   if (evicted > completed)
      return BITFIELD_MASK(HK_SYNC_NUM_SQ);
   uint8_t engines = 0;
   if (index) {
      for (uint16_t i = index->head[tile_bucket(fp)]; i != UINT16_MAX;
           i = index->next[i]) {
         const uint8_t engine = BITFIELD_BIT(q->recent[i].sq);
         if (!(engines & engine) &&
             footprints_share_tile(&q->recent[i].fp, fp))
            engines |= engine;
      }
      return engines;
   }
   const uint32_t n = MIN2(q->recent_count, HK_SYNC_RECENT_FOOTPRINTS);
   for (uint32_t i = 0; i < n; ++i) {
      if (q->recent[i].point > completed &&
          footprints_share_tile(&q->recent[i].fp, fp))
         engines |= BITFIELD_BIT(q->recent[i].sq);
   }
   return engines;
}

static void
recent_add(struct hk_sync_queue *q, const struct hk_sync_footprint *fp,
           uint64_t point, enum hk_sync_sq sq)
{
   struct hk_sync_recent *slot = &q->recent[q->recent_next];
   if (q->recent_count >= HK_SYNC_RECENT_FOOTPRINTS) {
      q->evicted = MAX2(q->evicted, slot->point);
      if (q->precise_tile_evictions && slot->fp.preserve)
         q->evicted_preserve = MAX2(q->evicted_preserve, slot->point);
   }
   if (q->precise_tile_evictions && fp->preserve)
      q->last_preserve = MAX2(q->last_preserve, point);
   *slot = (struct hk_sync_recent){.fp = *fp, .point = point, .sq = sq};
   q->recent_next = (q->recent_next + 1) % HK_SYNC_RECENT_FOOTPRINTS;
   q->recent_count = MIN2(q->recent_count + 1, HK_SYNC_RECENT_FOOTPRINTS);
   q->recent_max = MAX2(q->recent_max, point);
}

/* Raise a resolved barrier field to at least v (-1 none, 0 earlier ioctls).
 * With exact priors, a position past the 16-bit field raises the absolute
 * prior instead (see resolve_one). */
static void
barrier_at_least(const struct hk_queue *queue, unsigned sq, uint16_t *barrier,
                 uint64_t *prior, hk_sync_value v)
{
   if (queue->sync.exact_prior && v >= DRM_ASAHI_BARRIER_NONE) {
      *prior = MAX2(*prior, queue->sync.submitted[sq] + (uint64_t)v);
      return;
   }
   const hk_sync_value cur =
      *barrier == DRM_ASAHI_BARRIER_NONE ? HK_SYNC_NONE : *barrier;
   if (v > cur)
      *barrier = (uint16_t)v;
}

struct submit_footprint {
   struct hk_sync_footprint fp;
   hk_sync_value index;
   enum hk_sync_sq sq;
   uint32_t next_by_image;
};

/* The barrier and prior fields of one requirement on subqueue sq, with the
 * carried requirements of the given destination classes. */
static void
resolve_one(const struct submit_state *ss, unsigned sq,
            const struct hk_sync_dep *dep, uint8_t kinds, uint16_t *barrier,
            uint64_t *prior)
{
   *prior = 0;
   if (ss->queue->sync.exact_prior) {
      uint64_t v = resolve_absolute(ss, dep, sq);
      u_foreach_bit(k, kinds)
         v = MAX2(v, ss->carry_absolute[k][sq]);
      const uint64_t base = ss->queue->sync.submitted[sq];
      /* A command too far back for the 16-bit field is in an earlier ioctl
       * of this submission (at most max_commands_per_submit() each), already
       * submitted when this one is: name it by its ordinal. */
      if (v > base && v - base < DRM_ASAHI_BARRIER_NONE) {
         *barrier = (uint16_t)(v - base);
      } else {
         *barrier = DRM_ASAHI_BARRIER_NONE;
         *prior = v;
      }
   } else {
      hk_sync_value v = resolve_dep(ss, dep, sq);
      u_foreach_bit(k, kinds)
         v = value_max(v, ss->carry[k][sq]);
      *barrier = v < 0 ? DRM_ASAHI_BARRIER_NONE : (uint16_t)v;
   }
}

void
hk_sync_load_resolved(struct hk_cs *cs, const struct hk_sync_resolved *r)
{
   memcpy(cs->sync.barrier, r->barrier, sizeof(r->barrier));
   memcpy(cs->sync.barrier_ta, r->barrier_ta, sizeof(r->barrier_ta));
   memcpy(cs->sync.prior, r->prior, sizeof(r->prior));
   memcpy(cs->sync.prior_ta, r->prior_ta, sizeof(r->prior_ta));
   cs->sync.ta_split = r->ta_split;
}

bool
hk_sync_resolve_submit(struct hk_queue *queue,
                       struct hk_cmd_buffer *const *cmdbufs, unsigned count,
                       bool first_cdm_is_flush, bool chain_first,
                       uint64_t completed, struct hk_sync_resolved *out)
{
   /* Validate all ordinal arithmetic before mutating command or queue state.
    * Historical ordinals use zero for absence, so wrapping them is invalid.
    * Positions past the 16-bit relative fields become ordinals
    * (resolve_one), so a submission may hold any number of commands. */
   if (queue->sync.exact_prior) {
      uint64_t added[HK_SYNC_NUM_SQ] = {0, first_cdm_is_flush ? 1 : 0};
      for (unsigned i = 0; i < count; ++i) {
         list_for_each_entry(struct hk_cs, cs, &cmdbufs[i]->control_streams, node)
            ++added[hk_sync_sq(cs)];
      }
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
         if (queue->sync.submitted[sq] > UINT64_MAX - added[sq])
            return false;
      }
   }

   bool indexed = !queue->sync.precise_tile_evictions ||
                  queue->sync.last_preserve ||
                  submit_preserves_tiles(cmdbufs, count);
   if (indexed) {
      indexed = false;
      for (unsigned i = 0; i < count && !indexed; ++i) {
         indexed = util_dynarray_num_elements(
            &cmdbufs[i]->sync_pass_records, struct hk_sync_pass_record) != 0;
      }
   }
   struct recent_tile_index recent_index;
   uint32_t submitted_head[TILE_BUCKETS];
   if (indexed) {
      memset(&recent_index, 0xff, sizeof(recent_index));
      memset(submitted_head, 0xff, sizeof(submitted_head));
      const struct hk_sync_queue *q = &queue->sync;
      const uint32_t n = MIN2(q->recent_count, HK_SYNC_RECENT_FOOTPRINTS);
      for (uint32_t i = 0; i < n; ++i) {
         if (q->recent[i].point <= completed)
            continue;
         const unsigned bucket = tile_bucket(&q->recent[i].fp);
         recent_index.next[i] = recent_index.head[bucket];
         recent_index.head[bucket] = i;
      }
   }

   struct submit_state ss = {.queue = queue};
   struct util_dynarray submitted;
   util_dynarray_init(&submitted, NULL);
   /* The temporary list orders different command buffers of this submit.
    * With only one, recorded dependencies already order all its passes.
    * Publish its original records to queue history after resolving hazards,
    * avoiding an allocation and a second copy without exposing this submit
    * as earlier history to itself.
    */
   const bool direct_footprints =
      queue->sync.direct_single_footprints && count == 1;

   for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
         ss.last[k][sq] = HK_SYNC_NONE;
         ss.carry[k][sq] = HK_SYNC_NONE;
         ss.carry_absolute[k][sq] = queue->sync.tail[k][sq];
      }
   }

   if (first_cdm_is_flush) {
      /* A sparse-binding flush opens the payload; it is a compute command
       * everything must follow, handled by the chain below. */
      ss.count[HK_SYNC_CDM] = 1;
      if (queue->sync.exact_prior) {
         for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k)
            ss.last[k][HK_SYNC_CDM] = 1;
      }
   }

   /* Legacy kernels attach input syncs to the first hardware command only;
    * the caller requests that chain when needed. A sparse flush must still
    * precede everything. Every
    * command that does not already follow the first one transitively gets
    * an explicit wait on it. Waits are prefixes, so following any command
    * that follows the first is enough. */
   /* The first command is the first stream of the first command buffer that
    * emits any: command buffers without streams (barrier-only or empty) send
    * nothing to the kernel. Without any command there is nothing to chain. */
   int first_sq = first_cdm_is_flush ? HK_SYNC_CDM : -1;
   for (unsigned i = 0; first_sq < 0 && i < count; ++i) {
      if (!list_is_empty(&cmdbufs[i]->control_streams))
         first_sq = hk_sync_sq(list_first_entry(&cmdbufs[i]->control_streams,
                                                struct hk_cs, node));
   }
   const bool chain = (chain_first || first_cdm_is_flush) && first_sq >= 0;

   /* covered[sq]: the lowest index on sq of a command that follows the
    * first command, 0 if none yet. A wait on prefix b of subqueue d
    * follows the first command when covered[d] is within that prefix. */
   uint32_t covered[HK_SYNC_NUM_SQ] = {0, first_cdm_is_flush ? 1 : 0};
   bool seen_first = first_cdm_is_flush;

   /* A command buffer may occur more than once: everything below that reads
    * or writes a stream's index and resolved fields finishes with one
    * occurrence, and saves it to out, before the next one starts. */
   for (unsigned i = 0; i < count; ++i) {
      struct hk_cmd_buffer *cmd = cmdbufs[i];

      /* Positions first: cuts point at streams of this command buffer. */
      list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node)
         cs->sync.index = ++ss.count[hk_sync_sq(cs)];

      list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
            /* Safety net: a wait on a stream positioned after
             * this one can never be satisfied (the kernel receives a barrier
             * past the commands submitted before it and the job hangs until
             * the firmware times it out). Clamp to the latest earlier
             * stream on that subqueue and say so. */
            struct hk_sync_dep *d = &cs->sync.dep[sq];
            if (d->cut && d->cut != cs && d->cut->sync.seq > cs->sync.seq) {
               mesa_logw_once("hk_sync: forward wait clamped (stream seq %u "
                              "waited on seq %u)", cs->sync.seq,
                              d->cut->sync.seq);
               d->cut = latest_before(cmd, sq, cs);
            }
            /* The vertex half's requirement is part of the whole one. */
            struct hk_sync_dep *t = &cs->sync.dep_ta[sq];
            if (t->cut && t->cut != cs && t->cut->sync.seq > cs->sync.seq)
               t->cut = latest_before(cmd, sq, cs);

            /* The whole render waits on the carried requirements of all its
             * destination classes, its vertex half on those of its kinds. */
            resolve_one(&ss, sq, &cs->sync.dep[sq], full_kinds(cs->sync.kind),
                        &cs->sync.barrier[sq], &cs->sync.prior[sq]);
            resolve_one(&ss, sq, &cs->sync.dep_ta[sq],
                        cs->sync.kind |
                           (cs->sync.vtx_images ? HK_SYNC_GFX_IMG : 0),
                        &cs->sync.barrier_ta[sq], &cs->sync.prior_ta[sq]);
         }
      }

      /* Tile hazards: an instance sharing a tile with an instance of an
       * earlier command buffer of this submission waits on that one, and
       * with a possibly still running earlier submission on everything
       * before this ioctl. Within the command buffer the recorded
       * dependency already orders it. */
      util_dynarray_foreach(&cmd->sync_pass_records,
                            struct hk_sync_pass_record, rec) {
         if (!rec->first || !rec->last)
            continue;
         hk_sync_value v[HK_SYNC_NUM_SQ] = {HK_SYNC_NONE, HK_SYNC_NONE};
         if (indexed) {
            for (uint32_t j = submitted_head[tile_bucket(&rec->fp)];
                 j != TILE_INDEX_NONE;) {
               const struct submit_footprint *e = util_dynarray_element(
                  &submitted, struct submit_footprint, j);
               if (e->index > v[e->sq] &&
                   footprints_share_tile(&e->fp, &rec->fp))
                  v[e->sq] = e->index;
               j = e->next_by_image;
            }
         } else {
            util_dynarray_foreach(&submitted, struct submit_footprint, e) {
               if (e->index > v[e->sq] && footprints_share_tile(&e->fp, &rec->fp))
                  v[e->sq] = e->index;
            }
         }
         const uint8_t recent = recent_conflict(
            &queue->sync, &rec->fp, completed, indexed ? &recent_index : NULL);
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
            if (v[sq] == HK_SYNC_NONE && (recent & BITFIELD_BIT(sq)))
               v[sq] = 0;
            if (v[sq] != HK_SYNC_NONE)
               barrier_at_least(queue, sq, &rec->first->sync.barrier[sq],
                                &rec->first->sync.prior[sq], v[sq]);
         }
      }
      util_dynarray_foreach(&cmd->sync_pass_records,
                            struct hk_sync_pass_record, rec) {
         if (!rec->last || direct_footprints)
            continue;
         if (indexed && util_dynarray_num_elements(
               &submitted, struct submit_footprint) >= TILE_INDEX_NONE)
            indexed = false;
         const struct submit_footprint e = {
            .fp = rec->fp,
            .index = (hk_sync_value)rec->last->sync.index,
            .sq = hk_sync_sq(rec->last),
            .next_by_image = indexed ? submitted_head[tile_bucket(&rec->fp)]
                                     : TILE_INDEX_NONE,
         };
         if (indexed)
            submitted_head[tile_bucket(&rec->fp)] = util_dynarray_num_elements(
               &submitted, struct submit_footprint);
         util_dynarray_append(&submitted, e);
      }

      /* This command buffer's tail requirement joins the carry, resolved
       * against the streams before this command buffer. */
      for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
            ss.carry[k][sq] = value_max(
               ss.carry[k][sq], resolve_dep(&ss, &cmd->sync.req[k][sq], sq));
            if (queue->sync.exact_prior)
               ss.carry_absolute[k][sq] = MAX2(ss.carry_absolute[k][sq],
                  resolve_absolute(&ss, &cmd->sync.req[k][sq], sq));
         }
      }

      for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
            struct hk_cs *last = cmd->sync.last[k][sq];
            if (last)
               ss.last[k][sq] = value_max(ss.last[k][sq], last->sync.index);
         }
      }

      list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
         if (chain) {
            const unsigned sq = hk_sync_sq(cs);
            bool covers = false;

            if (!seen_first) {
               /* This is the first command. */
               seen_first = true;
               covers = true;
            } else {
               for (unsigned d = 0; d < HK_SYNC_NUM_SQ; ++d) {
                  const uint16_t b = cs->sync.barrier[d];
                  if (b != DRM_ASAHI_BARRIER_NONE && covered[d] != 0 &&
                      covered[d] <= b)
                     covers = true;
               }
               if (!covers) {
                  /* The first command is index 1 on its subqueue. */
                  cs->sync.barrier[first_sq] = 1;
                  covers = true;
               }
            }

            if (covers && covered[sq] == 0)
               covered[sq] = cs->sync.index;
         }

         /* A render's vertex half may wait for less than its fragment half
          * only when the kernel orders the halves separately and nothing
          * above raised the whole render's fields afterwards
          * (cross-command-buffer tile hazards and the first-command chain
          * only touch barrier[]): recompute, and keep the vertex half at
          * least as strict as before. */
         cs->sync.ta_split = false;
         if (cs->type != HK_CS_CDM && (chain_first || first_cdm_is_flush)) {
            memcpy(cs->sync.barrier_ta, cs->sync.barrier,
                   sizeof(cs->sync.barrier_ta));
            memcpy(cs->sync.prior_ta, cs->sync.prior,
                   sizeof(cs->sync.prior_ta));
         } else if (cs->type != HK_CS_CDM) {
            /* M5: tiling whose pre-rasterization programs keep state in
             * thread-private storage corrupts that state when it overlaps an
             * earlier render's fragment work (Counter-Strike 2: one-frame
             * garbage geometry and blocks, only for such renders). Such a
             * render's vertex half keeps the whole render-subqueue prefix;
             * its compute prefix may still be narrower. */
            if (cs->m5_vertex_tps) {
               cs->sync.barrier_ta[HK_SYNC_VDM] = cs->sync.barrier[HK_SYNC_VDM];
               cs->sync.prior_ta[HK_SYNC_VDM] = cs->sync.prior[HK_SYNC_VDM];
            }
            for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
               cs->sync.ta_split |=
                  cs->sync.barrier_ta[sq] != cs->sync.barrier[sq] ||
                  cs->sync.prior_ta[sq] != cs->sync.prior[sq];
            }
         }

         if (out) {
            struct hk_sync_resolved *r = out++;
            memcpy(r->barrier, cs->sync.barrier, sizeof(r->barrier));
            memcpy(r->barrier_ta, cs->sync.barrier_ta, sizeof(r->barrier_ta));
            memcpy(r->prior, cs->sync.prior, sizeof(r->prior));
            memcpy(r->prior_ta, cs->sync.prior_ta, sizeof(r->prior_ta));
            r->ta_split = cs->sync.ta_split;
         }
      }
   }

   /* Every footprint of this ioctl stays live on the queue until the
    * timeline point it signals (queue_submit has already advanced it).
    * Serialized queues order the next submission after this one whole. */
   if (!queue->sync.serialize || queue->sync.stage_overlap) {
      if (direct_footprints) {
         util_dynarray_foreach(&cmdbufs[0]->sync_pass_records,
                               struct hk_sync_pass_record, rec) {
            if (rec->last)
               recent_add(&queue->sync, &rec->fp, queue->drm.timeline_value,
                          hk_sync_sq(rec->last));
         }
      } else {
         util_dynarray_foreach(&submitted, struct submit_footprint, e)
            recent_add(&queue->sync, &e->fp, queue->drm.timeline_value, e->sq);
      }
   }
   util_dynarray_fini(&submitted);

   if (queue->sync.exact_prior) {
      for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
            queue->sync.tail[k][sq] = ss.carry_absolute[k][sq];
            if (ss.last[k][sq] > 0)
               queue->sync.last_absolute[k][sq] =
                  queue->sync.submitted[sq] + ss.last[k][sq];
         }
      }
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
         queue->sync.submitted[sq] += ss.count[sq];
   }

   /* This queue has now run these kinds. */
   for (unsigned i = 0; i < count; ++i) {
      list_for_each_entry(struct hk_cs, cs, &cmdbufs[i]->control_streams,
                          node)
         queue->sync.ever[hk_sync_sq(cs)] |= cs->sync.kind;
   }
   if (first_cdm_is_flush)
      queue->sync.ever[HK_SYNC_CDM] |= HK_SYNC_ALL;
   return true;
}

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Recorded-dependency tracking for Honeykrisp control streams. The design is
 * described in hk_sync.h.
 */
#include "hk_sync.h"
#include "util/u_atomic.h"

#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <strings.h>

#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_image.h"
#include "hk_query_oq.h"
#include "hk_queue.h"

#include "agx_chip_caps.h"
#include "libagx_dgc.h"

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

/* The destination classes whose carried requirements a submission adds to a
 * stream's requirement (hk_sync_stream::brl_carry). */
static inline uint8_t
stream_carry(const struct hk_cs *cs)
{
   return cs->sync.brl_transition ? cs->sync.brl_carry
                                  : full_kinds(cs->sync.kind);
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

/* Whether two footprints that share a tile share it only through a load:
 * both cover whole tiles of every layer they name, so neither pass writes
 * back a tile it does not own whole, and the later one reads what the
 * earlier one stored. */
static bool
footprints_whole_tiles(const struct hk_sync_footprint *a,
                       const struct hk_sync_footprint *b)
{
   return !a->partial && !b->partial;
}

void
hk_sync_pass_footprint(struct hk_cmd_buffer *cmd, const void *image,
                       uint32_t level, uint32_t layer0, uint32_t layer1,
                       uint32_t level_width, uint32_t level_height,
                       VkRect2D area, bool preserve_layers, bool preserve_load)
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
   const bool partial = preserve_layers || x0 % cell || y0 % cell ||
                        (x1 % cell && x1 != w) ||
                        (y1 % cell && y1 != h);

   assert(st->pass_fp_count < HK_SYNC_MAX_PASS_FOOTPRINTS);
   if (st->pass_fp_count >= HK_SYNC_MAX_PASS_FOOTPRINTS)
      return;

   st->pass_fp[st->pass_fp_count++] = (struct hk_sync_footprint){
      .preserve = partial || preserve_load,
      .partial = partial,
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

bool
hk_sync_load_hazard_frag(const struct hk_device *dev)
{
   return dev && dev->load_hazard_frag;
}

void
hk_sync_pass_footprints_done(struct hk_cmd_buffer *cmd)
{
   struct hk_sync_state *st = &cmd->sync;
   if (!cmd->sync_pass_records.size) {
      memset(st->pass_tile_wait, 0, sizeof(st->pass_tile_wait));
      st->pass_tile_wait_frag = NULL;
      return;
   }
   const bool frag_ok = hk_sync_load_hazard_frag(hk_cmd_buffer_device(cmd));
   struct hk_cs *wait[HK_SYNC_NUM_SQ] = {0};
   struct hk_cs *frag = NULL;
   util_dynarray_foreach(&cmd->sync_pass_records, struct hk_sync_pass_record, rec) {
      if (!rec->last)
         continue;
      const unsigned sq = hk_sync_sq(rec->last);
      if (wait[sq] && rec->last->sync.seq <= wait[sq]->sync.seq)
         continue;
      /* A render that shares only whole, loaded tiles with the instance
       * needs the fragment half ordered (hk_sync_stream_created()). */
      bool shares = false, whole = true;
      for (unsigned i = 0; i < st->pass_fp_count; ++i) {
         if (footprints_share_tile(&rec->fp, &st->pass_fp[i])) {
            shares = true;
            whole &= footprints_whole_tiles(&rec->fp, &st->pass_fp[i]);
         }
      }
      if (!shares)
         continue;
      if (frag_ok && whole && rec->last->type == HK_CS_VDM)
         frag = hk_sync_later(frag, rec->last);
      else
         wait[sq] = rec->last;
   }
   memcpy(st->pass_tile_wait, wait, sizeof(wait));
   /* A wait of both halves on a later render covers it. */
   st->pass_tile_wait_frag =
      frag && (!wait[HK_SYNC_VDM] ||
               frag->sync.seq > wait[HK_SYNC_VDM]->sync.seq)
         ? frag
         : NULL;
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
   st->pass_tile_wait_frag = NULL;
   st->pass_tile_first = st->pass_tile_last = NULL;
}

static void
records_replace_stream(struct hk_cmd_buffer *cmd, struct hk_cs *from,
                       struct hk_cs *to, bool drop_singleton)
{
   util_dynarray_foreach(&cmd->sync_pass_records, struct hk_sync_pass_record,
                         rec) {
      if (drop_singleton && rec->first == from && rec->last == from) {
         rec->first = rec->last = NULL;
         continue;
      }
      if (rec->first == from)
         rec->first = to;
      if (rec->last == from)
         rec->last = to;
   }
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      if (cmd->sync.pass_tile_wait[sq] == from)
         cmd->sync.pass_tile_wait[sq] = to;
   }
   /* A render replaced by compute is no longer a render: wait on it whole. */
   if (cmd->sync.pass_tile_wait_frag == from) {
      cmd->sync.pass_tile_wait_frag = NULL;
      if (to) {
         const unsigned sq = hk_sync_sq(to);
         cmd->sync.pass_tile_wait[sq] =
            hk_sync_later(cmd->sync.pass_tile_wait[sq], to);
      }
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
brl_sync_image_barrier_kinds(const VkImageMemoryBarrier2 *barrier,
                             bool image_class, uint8_t *src, uint8_t *dst)
{
   *src |= hk_sync_kinds_src(barrier->srcStageMask);
   /* Graphics work accesses an image in the fragment and attachment
    * stages or through a shader that reads it (HK_SYNC_GFX_IMG). */
   uint8_t d = hk_sync_kinds_dst(barrier->dstStageMask);
   if (image_class && (d & HK_SYNC_GFX))
      d = (d & ~HK_SYNC_GFX) | HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG;
   *dst |= d;
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

   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i)
      brl_sync_image_barrier_kinds(&info->pImageMemoryBarriers[i], image_class,
                                   src, dst);
}

/* A host access, or MEMORY_READ or MEMORY_WRITE with the host stage in the
 * destination scope: those stand for every read or write valid in the scope,
 * the host's too. ALL_COMMANDS does not include the host stage. A host write
 * needs the earlier device writes written back first as much as a host read
 * does: a line written back after it would overwrite it. */
static bool
barrier_has_host_access(VkPipelineStageFlags2 dst_stages,
                        VkAccessFlags2 dst_access)
{
   if (dst_access & (VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_HOST_WRITE_BIT))
      return true;

   return (dst_stages & VK_PIPELINE_STAGE_2_HOST_BIT) &&
          (dst_access &
           (VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT));
}

bool
hk_sync_dependency_has_host_access(const VkDependencyInfo *info)
{
   for (uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
      const VkMemoryBarrier2 *b = &info->pMemoryBarriers[i];
      if (barrier_has_host_access(b->dstStageMask, b->dstAccessMask))
         return true;
   }

   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) {
      const VkBufferMemoryBarrier2 *b = &info->pBufferMemoryBarriers[i];
      if (barrier_has_host_access(b->dstStageMask, b->dstAccessMask))
         return true;
   }

   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
      const VkImageMemoryBarrier2 *b = &info->pImageMemoryBarriers[i];
      if (barrier_has_host_access(b->dstStageMask, b->dstAccessMask))
         return true;
   }

   return false;
}

/* Vulkan 1.4, 7.1 "Execution and Memory Dependencies": a destination access
 * scope with a host access adds a memory domain operation that makes the
 * memory available in the device domain available to the host domain. That
 * is any write an earlier availability operation in an execution dependency
 * chain reaching this one made available, not only the writes of this
 * barrier's source access scope, so a barrier with read-only or empty source
 * access still flushes. A domain operation happens after the first
 * synchronization scope, and only one with device work in it can follow
 * device writes: an empty or host-only first scope needs no flush. */
uint8_t
hk_sync_host_flush_kinds(const VkDependencyInfo *info)
{
   uint8_t kinds = 0;

   for (uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
      const VkMemoryBarrier2 *b = &info->pMemoryBarriers[i];
      if (barrier_has_host_access(b->dstStageMask, b->dstAccessMask))
         kinds |= hk_sync_kinds_src(b->srcStageMask);
   }

   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) {
      const VkBufferMemoryBarrier2 *b = &info->pBufferMemoryBarriers[i];
      if (barrier_has_host_access(b->dstStageMask, b->dstAccessMask))
         kinds |= hk_sync_kinds_src(b->srcStageMask);
   }

   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
      const VkImageMemoryBarrier2 *b = &info->pImageMemoryBarriers[i];
      if (barrier_has_host_access(b->dstStageMask, b->dstAccessMask))
         kinds |= hk_sync_kinds_src(b->srcStageMask);
   }

   return kinds;
}

/* The host observes memory only through a fence, semaphore or queue wait at
 * the end of a submission, or through an event status written before then
 * (hk_sync_take_host_flush()), so the flush can wait until the submission's
 * end. One flush there serves every such dependency of the submission, and
 * no stream is created between the work around them. */
void
hk_sync_defer_host_flush(struct hk_cmd_buffer *cmd,
                         const VkDependencyInfo *info)
{
   cmd->sync.host_flush_kinds |= hk_sync_host_flush_kinds(info);
}

uint8_t
hk_sync_take_host_flush(struct hk_cmd_buffer *cmd)
{
   const uint8_t kinds = cmd->sync.host_flush_kinds;
   cmd->sync.host_flush_kinds = 0;
   return kinds;
}

void
brl_sync_transition_kinds(const VkDependencyInfo *info,
                          const VkImageMemoryBarrier2 *barrier,
                          bool image_class, bool acquire, bool render_pass,
                          uint8_t *src, uint8_t *dst)
{
   VkImageMemoryBarrier2 b = *barrier;
   if (acquire &&
       !(info->dependencyFlags &
         VK_DEPENDENCY_QUEUE_FAMILY_OWNERSHIP_TRANSFER_USE_ALL_STAGES_BIT_KHR))
      b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
   brl_sync_image_barrier_kinds(&b, image_class, src, dst);

   /* Vulkan 1.4, 8.1 "Render Pass": an automatic layout transition happens
    * after the availability operations of the dependencies into the subpass
    * and before their visibility operations. The common runtime passes only
    * those dependencies, so their union is at most slightly wider than the
    * ones naming the attachment. */
   if (render_pass) {
      for (uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
         *src |= hk_sync_kinds_src(info->pMemoryBarriers[i].srcStageMask);
         *dst |= hk_sync_kinds_dst(info->pMemoryBarriers[i].dstStageMask);
      }
   }
}

static struct hk_cs *previous_on_sq(struct hk_cmd_buffer *cmd,
                                    struct hk_cs *cs);
static struct hk_cs *previous_on_sq_kinds(struct hk_cmd_buffer *cmd,
                                          struct hk_cs *cs, uint8_t kinds);

void
hk_sync_src_dep(struct hk_cmd_buffer *cmd, uint8_t src_kinds,
                struct hk_cs *self, struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   /* Query, timestamp, and event scopes include availability writes staged on
    * the render (HK_SYNC_STAMP). Materialize their post-graphics stream before
    * taking the CDM cut. */
   if ((src_kinds & HK_SYNC_STAMP) && unlikely(cmd->state.oq_pending_render))
      hk_flush_oq_pending(cmd);

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

static void stamp_wait(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       const struct hk_sync_dep dep[HK_SYNC_NUM_SQ]);

/* What every timestamp write waits for, whatever its stage: the earlier
 * resets of timestamp queries (HK_SYNC_QRESET), which are in no application
 * dependency. A stamp written ahead of a reset of its query would be zeroed
 * by it. */
static void
qreset_dep(struct hk_cmd_buffer *cmd, struct hk_cs *self,
           struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   hk_sync_src_dep(cmd, HK_SYNC_QRESET, self, dep);
}

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
   if (unlikely(cmd->state.oq_pending_render))
      hk_flush_oq_pending(cmd);
   cmd->sync.epoch++;

   /* The dependencies an acceleration-structure build records between its
    * own commands order only its compute and transfer work: nothing else is
    * recorded until the build returns, and later work is ordered after the
    * build by the application's own dependency on it. Their DRAW_INDIRECT
    * destination (the build's indirect dispatches read their sizes) names no
    * render, so they must not order later renders' vertex halves. */
   if (cmd->sync.as_build)
      dst_kinds &= HK_SYNC_COMPUTE | HK_SYNC_XFER;

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
       agx_chip_g17p_or_m5(chip)) {
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
   if (unlikely(cmd->state.oq_pending_render))
      hk_flush_oq_pending(cmd);
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

void
brl_sync_event_poll(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   cmd->sync.brl_event_poll = hk_sync_later(cmd->sync.brl_event_poll, cs);
}

void
brl_sync_event_done(struct hk_cmd_buffer *cmd)
{
   cmd->sync.brl_event_poll = NULL;
}

void
brl_sync_transition_prepare(struct hk_cmd_buffer *cmd, uint8_t src_kinds)
{
   struct hk_sync_state *st = &cmd->sync;
   assert(!st->brl_transition_active);
   st->epoch++;

   /* The first scope, with the requirements earlier dependencies carry into
    * it (barrier chains): of this command buffer here, of earlier ones at
    * submission (hk_sync_stream::brl_carry). A host operation in the first
    * scope has no class: Vulkan has the host signal an event before it
    * submits the wait (vkSetEvent VUID-vkSetEvent-event-09543) and finish
    * other host writes before the submission, which orders them. */
   struct hk_sync_dep *dep = st->brl_transition_dep;
   barrier_source_dependencies(cmd, src_kinds, dep);
   st->brl_transition_carry = full_kinds(src_kinds);

   /* An event wait's transition follows the wait itself. */
   dep[HK_SYNC_CDM].cut =
      hk_sync_later(dep[HK_SYNC_CDM].cut, st->brl_event_poll);

   /* Layout transitions execute in submission order on the queue: after the
    * earlier transitions of this command buffer and, since the tracking
    * knows only stage classes before it, after the transfer-class work of
    * earlier command buffers, which includes their transitions. That is
    * their work only: what earlier dependencies ask of later transfer work
    * does not apply to the transition (hk_sync_stream::brl_carry). */
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      dep[sq].cut = hk_sync_later(dep[sq].cut, st->brl_transition_last[sq]);
      dep[sq].prior |= HK_SYNC_XFER;
   }
}

void
brl_sync_transition_begin(struct hk_cmd_buffer *cmd)
{
   if (unlikely(cmd->state.oq_pending_render))
      hk_flush_oq_pending(cmd);
   struct hk_sync_state *st = &cmd->sync;
   assert(!st->brl_transition_active && !st->kind_override);
   st->brl_transition_active = true;
   st->brl_transition_seq = st->next_seq;
   st->epoch++;
}

void
brl_sync_transition_end(struct hk_cmd_buffer *cmd, uint8_t dst_kinds)
{
   struct hk_sync_state *st = &cmd->sync;
   assert(st->brl_transition_active);
   st->brl_transition_active = false;
   st->epoch++;

   /* The second scope waits on the transition's streams. An empty second
    * scope orders nothing after it but later transitions (prepare above) and
    * the end of the submission. */
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      struct hk_cs *last = st->brl_transition_last[sq];
      if (dst_kinds && last && last->sync.seq > st->brl_transition_seq)
         hk_sync_after_stream(cmd, last, dst_kinds);
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
   if (cs->sync.vtx_img_unsafe && cs->sync.vtx_images_epoch == cmd->sync.epoch)
      return;

   cs->sync.vtx_images = true;
   cs->sync.vtx_img_unsafe = true;
   cs->sync.vtx_images_epoch = cmd->sync.epoch;
   cs->sync.dep_ta[HK_SYNC_VDM].prior |= cs->sync.dep_ta_img_prior;
   cs->sync.dep_ta_img_prior = 0;
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   req_for_mask(cmd, HK_SYNC_GFX_IMG, dep);
   hk_sync_stream_wait(cs, dep);
}

void
hk_sync_vertex_images_bloom(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                            uint64_t bloom)
{
   if (cs->sync.vtx_img_unsafe) {
      hk_sync_vertex_images(cmd, cs);
      return;
   }
   if (cs->sync.vtx_images && cs->sync.vtx_images_epoch == cmd->sync.epoch &&
       !(bloom & ~cs->sync.vtx_img_bloom))
      return;

   cs->sync.vtx_images = true;
   cs->sync.vtx_images_epoch = cmd->sync.epoch;
   cs->sync.vtx_img_bloom |= bloom;

   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   req_for_mask(cmd, HK_SYNC_GFX_IMG, dep);

   /* Whole render / fragment half always keeps full HK_SYNC_GFX_IMG. */
   stream_wait_frag(cs, dep);

   /* Compute subqueue orders both halves anyway. */
   if (dep[HK_SYNC_CDM].cut == cs) {
      cs->sync.dep_ta[HK_SYNC_CDM].prior |= dep[HK_SYNC_CDM].prior;
   } else {
      dep_max(&cs->sync.dep_ta[HK_SYNC_CDM], &dep[HK_SYNC_CDM]);
   }

   /* Render subqueue: wait only on the latest in-CB render at or before the
    * image-barrier cut that wrote memory or an attachment matching our bloom. */
   struct hk_cs *vdm_cut = dep[HK_SYNC_VDM].cut;
   if (vdm_cut && vdm_cut != cs) {
      struct hk_cs *best = NULL;
      list_for_each_entry(struct hk_cs, it, &cmd->control_streams, node) {
         if (it->type != HK_CS_VDM || it == cs ||
             it->sync.seq > vdm_cut->sync.seq)
            continue;
         if (((it->sync.kind & HK_SYNC_GFX_MEM) ||
              (it->sync.att_bloom & cs->sync.vtx_img_bloom)) &&
             (!best || it->sync.seq > best->sync.seq))
            best = it;
      }
      if (best)
         cs->sync.dep_ta[HK_SYNC_VDM].cut =
            hk_sync_later(cs->sync.dep_ta[HK_SYNC_VDM].cut, best);
   }
   cs->sync.dep_ta_img_prior |= dep[HK_SYNC_VDM].prior;
}

void
hk_sync_after_stream(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                     uint8_t dst_kinds)
{
   if (unlikely(cmd->state.oq_pending_render))
      hk_flush_oq_pending(cmd);
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
   if (cmd->sync.xfer_depth || cmd->sync.brl_transition_active)
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
      cs->sync.pre_seq = ++gfx->sync.pre_count;
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
   else if (!compute && cmd->state.gfx.render.m5_attachments_alias_image)
      cs->sync.alias_image_render = true;
   if (!compute)
      cs->sync.att_bloom =
         st->in_pass ? cmd->state.gfx.render.att_bloom : UINT64_MAX;
   cs->sync.prev_mem =
      st->last[util_logbase2(HK_SYNC_GFX_MEM)][hk_sync_sq(cs)];

   /* The recorded requirement for this kind of work. A render's vertex half
    * needs only the part whose destination is not fragment-only. */
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   /* Software vertex work is outside an in-pass fragment-only destination.
    * Waiting on the current render here would create a cycle after a kept
    * local barrier. Preserve global requirements, though: internal producers
    * such as attachment decompression must still follow external writes. */
   if (st->brl_transition_active) {
      /* A driver-performed layout transition waits on its own requirement
       * only (brl_sync_transition_prepare()). */
      memcpy(dep, st->brl_transition_dep, sizeof(dep));
   } else if (compute && slot == &cmd->current_cs.pre_gfx) {
      /* The pre-graphics work of a pipeline without software
       * pre-rasterization stages (draw parameter patching, index rewriting,
       * statistics) reads buffers in the draw-indirect and vertex-input
       * stages alone: image barriers (HK_SYNC_GFX_IMG) do not order it.
       * A spilling render pass's pre-graphics decompression accesses the
       * spilled colour attachments themselves, so it must still honour
       * image barriers. */
      uint8_t classes = full_kinds(cs->sync.kind);
      if (!agx_tilebuffer_spills(&cmd->state.gfx.render.tilebuffer) &&
          !cmd->state.gfx.shaders[MESA_SHADER_TESS_CTRL] &&
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
      if (st->brl_transition_active)
         memcpy(ta, dep, sizeof(ta));
      else
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
       * earlier render that nothing else orders it after. Both halves wait:
       * a vertex half started ahead of the earlier render corrupts the
       * shared tile (seen with M5 staged renders); tiling can reach the
       * attachments before the fragment half starts, for example through a
       * partial render when the tiler heap fills. */
      if (st->in_pass) {
         struct hk_sync_dep tile[HK_SYNC_NUM_SQ] = {0};
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
            tile[sq].cut = st->pass_tile_wait[sq];
         hk_sync_stream_wait(cs, tile);

         /* Except where the instance only loads whole tiles an earlier
          * render stored: the kernel runs this render's fragment work after
          * that one's, partial renders included, and tiling never writes a
          * tile back (hk_sync_queue::load_hazard_frag). */
         const struct hk_sync_dep load[HK_SYNC_NUM_SQ] = {
            {.cut = st->pass_tile_wait_frag}, {0}};
         stream_wait_frag(cs, load);
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
    * hk_heap): a pass that uses it must not overlap any earlier stream.
    * With multi-slot heap rings (pass_heap_slot > 0), only wait on the
    * previous pass that used the same slot. */
   if (st->heap_pass && (!compute || pre_gfx || post_gfx))
      hk_sync_stream_wait_all(cmd, cs);
   if (st->pass_heap_slot && (!compute || pre_gfx || post_gfx)) {
      const uint8_t slot = st->pass_heap_slot - 1;
      cs->sync.heap_slot = st->pass_heap_slot;
      struct hk_sync_dep hdep[HK_SYNC_NUM_SQ] = {
         {.cut = st->last_heap_slot[slot][HK_SYNC_VDM]},
         {.cut = st->last_heap_slot[slot][HK_SYNC_CDM]},
      };
      hk_sync_stream_wait(cs, hdep);
      const unsigned sq = hk_sync_sq(cs);
      st->pass_heap_last[sq] = hk_sync_later(st->pass_heap_last[sq], cs);
   }

   note_last(cmd, cs);
   if (st->brl_transition_active) {
      cs->sync.brl_transition = true;
      cs->sync.brl_carry = st->brl_transition_carry;
      const unsigned sq = hk_sync_sq(cs);
      st->brl_transition_last[sq] =
         hk_sync_later(st->brl_transition_last[sq], cs);
   }

   if (slot == &cmd->current_cs.cs)
      st->cs_check_pending = false;
   else if (!compute && cmd->current_cs.cs &&
            (cs->sync.dep[HK_SYNC_CDM].cut == cmd->current_cs.cs ||
             cs->sync.dep_ta[HK_SYNC_CDM].cut == cmd->current_cs.cs))
      st->cs_check_pending = true;

   /* A top-of-pipe timestamp waiting for a command to ride on. An ordered
    * start stamp latches after the command's barriers, which then include the
    * resets of timestamp queries; an unordered one is checked against them at
    * submission (hk_queue.c). */
   if (st->pending_start_ts.handle) {
      cs->timestamp.start = st->pending_start_ts;
      cs->unordered_start_ts = st->pending_start_ts_unordered;
      if (cs->unordered_start_ts) {
         st->has_unordered_start_ts = true;
      } else {
         struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
         qreset_dep(cmd, cs, dep);
         hk_sync_stream_wait(cs, dep);
      }
      st->pending_start_ts = (struct agx_timestamp_req){0};
      st->pending_start_ts_unordered = false;
   }

   /* The command buffer's leading timestamp rides on a first render stream's
    * vertex end stamp (hk_sync_write_timestamp()), which waits for the
    * stamp's first scope. A compute stream has no separate start-side end
    * stamp, so hk_cmd_buffer_get_cs_general() flushes lead_ts into a
    * preceding timestamp-only carrier before creating one. Nothing was
    * recorded between the stamp and this stream, so the scope names the same
    * earlier work it did then. */
   if (st->lead_ts.handle) {
      assert(cs->type == HK_CS_VDM);
      assert(!cs->timestamp.vtx_end.handle);
      struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
      hk_sync_src_dep(cmd, st->lead_scope, cs, dep);
      cs->timestamp.vtx_end = st->lead_ts;
      hk_sync_stream_wait(cs, dep);
      st->lead_ts = (struct agx_timestamp_req){0};
      /* Later stamps recorded before this stream are copies of it. */
      if (st->stamp_copy_count) {
         st->stamp_copy_after[HK_SYNC_VDM] =
            hk_sync_later(st->stamp_copy_after[HK_SYNC_VDM], cs);
      }
   }
}

/* Whether the compute stream cs already waits for every earlier compute stream
 * matching kinds: for the latest one before it, which covers the rest as a
 * prefix. When kinds is HK_SYNC_STREAM_KINDS, every earlier compute stream
 * counts; application requirements (HK_SYNC_ALL) ignore pure query-memory
 * streams (HK_SYNC_STAMP, HK_SYNC_QRESET), which no application dependency
 * names. */
static bool
cdm_waits_for_prefix(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                     uint8_t kinds)
{
   const struct hk_cs *prev =
      kinds == HK_SYNC_STREAM_KINDS ? previous_on_sq(cmd, cs)
                                    : previous_on_sq_kinds(cmd, cs, kinds);
   const struct hk_cs *cut = cs->sync.dep[HK_SYNC_CDM].cut;
   return !prev || (cut && cut != cs && cut->sync.seq >= prev->sync.seq);
}

/* Whether work recorded into the open compute stream cs meets the requirement
 * dep: every stream dep names must precede cs. A cut at cs itself names the
 * earlier compute streams of prefix_kinds as well as the work inside cs. The
 * stream orders only its own work, and the kernel may run earlier compute
 * streams of the queue on other firmware channels
 * (DRM_ASAHI_FEATURE_COMPUTE_MULTI_CHANNEL), so that cut is met only if cs
 * already waits for those streams. */
static bool
cs_meets(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
         const struct hk_sync_dep dep[HK_SYNC_NUM_SQ], uint8_t prefix_kinds)
{
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      if (dep[sq].cut && dep[sq].cut->sync.seq > cs->sync.seq)
         return false;
   }

   return dep[HK_SYNC_CDM].cut != cs ||
          cdm_waits_for_prefix(cmd, cs, prefix_kinds);
}

/* Whether the open general compute stream may receive work of the given
 * kinds (cs_meets()). Recorded requirements (cmd->sync.req) come from
 * application dependencies and never name pure query-memory streams
 * (HK_SYNC_STAMP, HK_SYNC_QRESET), so application work only needs the open
 * stream to cover earlier HK_SYNC_ALL compute streams. */
static bool
cs_reusable(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t kinds)
{
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   req_for_kinds(cmd, kinds, dep);
   const uint8_t prefix_kinds =
      (kinds & (HK_SYNC_STAMP | HK_SYNC_QRESET)) ? HK_SYNC_STREAM_KINDS
                                                 : HK_SYNC_ALL;
   return cs_meets(cmd, cs, dep, prefix_kinds);
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

/* Whether the compute stream cs already waits for all prior-submission render
 * kinds in prior_mask: directly, or through the compute prefix it waits on. */
static bool
cdm_waits_for_render_prior(struct hk_cmd_buffer *cmd, const struct hk_cs *cs,
                           uint8_t prior_mask)
{
   uint8_t covered = cs->sync.dep[HK_SYNC_VDM].prior;
   if (!(prior_mask & ~covered))
      return true;

   const struct hk_cs *prefix = cs->sync.dep[HK_SYNC_CDM].cut;
   if (!prefix || prefix == cs)
      return false;

   list_for_each_entry(struct hk_cs, it, &cmd->control_streams, node) {
      if (it->type != HK_CS_CDM || it == cs ||
          it->sync.seq > prefix->sync.seq)
         continue;
      covered |= it->sync.dep[HK_SYNC_VDM].prior;
      if (!(prior_mask & ~covered))
         return true;
   }
   return false;
}

static bool
cdm_overlap_split_enabled(struct hk_cmd_buffer *cmd)
{
   return agx_chip_has_cap(hk_cmd_buffer_device(cmd)->dev.chip,
                           AGX_CAP_CDM_OVERLAP_SPLIT);
}

/* Whether a requirement on the render subqueue would make the work a compute
 * stream already holds wait for a render it does not wait for yet. A
 * requirement recorded after that work (a barrier between two dispatches) has
 * none of it in its second scope, so the work keeps the requirements it was
 * recorded under and only later work takes the new one. */
static bool
cdm_raise_delays_work(struct hk_cmd_buffer *cmd, const struct hk_cs *cs,
                      const struct hk_sync_dep *vdm, bool next_waits_on_cs)
{
   if (!cs->stats.cmds || !cdm_overlap_split_enabled(cmd))
      return false;

   if (vdm->cut && !cdm_waits_for_render(cmd, cs, vdm->cut))
      return true;
   /* A speculative prior-submission render wait (vdm->prior with no local
    * render in this CB) arises on every ALL_COMMANDS -> ALL_COMMANDS compute
    * barrier. Only split on vdm->prior when the incoming compute does not
    * already depend on cs (e.g. an explicit GFX -> COMPUTE barrier), so pure
    * compute chains using ALL_COMMANDS keep stream coalescing and CDM packing. */
   if (!next_waits_on_cs && vdm->prior &&
       !cdm_waits_for_render_prior(cmd, cs, vdm->prior))
      return true;
   return false;
}

/* Whether a render stream or pending graphics requirement waits specifically
 * on cs: appending or merging independent post-barrier compute work into cs
 * would force that graphics work to wait on the later compute too.
 *
 * When no render stream in this command buffer waits on cs yet and the next
 * compute work itself already depends on cs (e.g. an ALL_COMMANDS ->
 * ALL_COMMANDS barrier inside a pure compute chain), keep it in cs so we do
 * not shatter compute stream coalescing and multi-CB CDM packing. */
static bool
cdm_waited_by_render(struct hk_cmd_buffer *cmd, const struct hk_cs *cs,
                     bool next_waits_on_cs)
{
   if (!cs->stats.cmds || !cdm_overlap_split_enabled(cmd))
      return false;

   list_for_each_entry(struct hk_cs, it, &cmd->control_streams, node) {
      if (it->type != HK_CS_VDM)
         continue;
      if (it->sync.dep[HK_SYNC_CDM].cut == cs ||
          it->sync.dep_ta[HK_SYNC_CDM].cut == cs)
         return true;
   }

   if (next_waits_on_cs)
      return false;

   const uint8_t gfx_kinds = HK_SYNC_GFX | HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG;
   u_foreach_bit(k, gfx_kinds) {
      if (cmd->sync.req[k][HK_SYNC_CDM].cut == cs ||
          (cmd->sync.in_pass && cmd->sync.pass_req[k][HK_SYNC_CDM].cut == cs))
         return true;
   }
   return false;
}

bool
hk_sync_merge_delays(struct hk_cmd_buffer *cmd, const struct hk_cs *into,
                     const struct hk_cs *from)
{
   const bool from_waits_on_into = from->sync.dep[HK_SYNC_CDM].cut == into;
   if (cdm_raise_delays_work(cmd, into, &from->sync.dep[HK_SYNC_VDM],
                             from_waits_on_into) ||
       cdm_waited_by_render(cmd, into, from_waits_on_into))
      return true;

   /* If a later render waits on `into` (and not on `from`, which follows it),
    * merging `from` into `into` would make that render wait for `from` too,
    * serializing independent compute work such as an AS build ahead of a
    * G-buffer pass. */
   if (from->stats.cmds) {
      list_for_each_entry(struct hk_cs, it, &cmd->control_streams, node) {
         if (it->type != HK_CS_VDM || it->sync.seq <= from->sync.seq)
            continue;
         if (it->sync.dep[HK_SYNC_CDM].cut == into ||
             it->sync.dep_ta[HK_SYNC_CDM].cut == into)
            return true;
      }
   }
   return false;
}

void
hk_sync_check_cs(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.cs;
   /* A transition's stream carries its own requirement alone. */
   if (!cs || cmd->sync.pin_cs || cmd->sync.brl_transition_active)
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
    * this stream through last[]. Likewise, if a graphics barrier or render
    * stream already waits on this stream, keep independent post-barrier
    * compute separate so graphics can overlap it. */
   const bool next_waits_on_cs = dep[HK_SYNC_CDM].cut == cs;
   if (cdm_raise_delays_work(cmd, cs, &dep[HK_SYNC_VDM], next_waits_on_cs) ||
       cdm_waited_by_render(cmd, cs, next_waits_on_cs)) {
      hk_cmd_buffer_end_compute(cmd);
      return;
   }

   /* The stream absorbs the requirement of the work it now carries. */
   hk_sync_stream_wait(cs, dep);
   cs->sync.kind |= kinds;
   if (cs->sync.kind != HK_SYNC_GFX)
      cs->sync.kind |= HK_SYNC_GFX_MEM;
   note_last(cmd, cs);

   if (agx_chip_native_or_m5(hk_cmd_buffer_device(cmd)->dev.chip) &&
       cmd->sync.pending_start_ts.handle && !cs->timestamp.start.handle &&
       !cs->timestamp.end.handle) {
      cs->timestamp.start = cmd->sync.pending_start_ts;
      cs->unordered_start_ts = cmd->sync.pending_start_ts_unordered;
      if (cs->unordered_start_ts)
         cmd->sync.has_unordered_start_ts = true;
      cmd->sync.pending_start_ts = (struct agx_timestamp_req){0};
      cmd->sync.pending_start_ts_unordered = false;
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

   if (cs && !cmd->sync.pin_cs && !cs_meets(cmd, cs, full, HK_SYNC_STREAM_KINDS))
      hk_cmd_buffer_end_compute(cmd);

   cs = hk_cmd_buffer_get_cs(cmd, true);
   if (cs) {
      /* A pinned stream stays open: it waits for the streams before it. */
      if (full[HK_SYNC_CDM].cut == cs &&
          !cdm_waits_for_prefix(cmd, cs, HK_SYNC_STREAM_KINDS))
         full[HK_SYNC_CDM].cut = previous_on_sq(cmd, cs);
      hk_sync_stream_wait(cs, full);
   }
   return cs;
}

/*
 * Prepare the compute stream a query command (reset, result copy) records
 * into. Query state is updated in submission order without application
 * barriers (Vulkan 1.3, 17 "Queries": vkCmdCopyQueryPoolResults sees every
 * earlier vkCmdResetQueryPool on the queue, and vkCmdBeginQuery requires the
 * query to be unavailable, which an earlier reset in the same command buffer
 * provides). The stream therefore waits on every earlier stream, and it is
 * never the stream carrying a pending end timestamp. On G17P, both start
 * and end timestamps reach query memory at command retirement, so neither
 * producer may contain the query command or be merged back into it. The caller
 * marks later streams as ordered after it where the command's semantics require
 * that.
 */
static struct hk_cs *
query_cs(struct hk_cmd_buffer *cmd, uint8_t scope)
{
   hk_sync_flush_pending_timestamp(cmd);
   /* Timestamps written as copies of earlier ones are query results too. */
   hk_sync_flush_stamp_copies(cmd);
   hk_sync_end_cs_if_timestamped(cmd);

   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   hk_sync_src_dep(cmd, scope, cmd->current_cs.cs, dep);
   struct hk_cs *cs = hk_sync_get_cs_after(cmd, dep);
   if (cs && agx_chip_native_or_m5(hk_cmd_buffer_device(cmd)->dev.chip))
      cs->g17p_timestamp_query = true;
   return cs;
}

struct hk_cs *
hk_query_command_cs(struct hk_cmd_buffer *cmd)
{
   return query_cs(cmd, HK_SYNC_QUERY_SCOPE);
}

/* A timestamp query reset is ordered only against the commands that
 * reference the same queries (Vulkan 1.4, 17.2). Of those, timestamp writes
 * and query commands name HK_SYNC_QRESET in their first scope
 * (hk_sync_write_timestamp(), HK_SYNC_QUERY_SCOPE), so nothing else waits
 * for the reset; and the reset waits for every earlier stream but other
 * timestamp resets, which write the same values. */
struct hk_cs *
hk_query_reset_cs(struct hk_cmd_buffer *cmd)
{
   assert(cmd->sync.kind_override == HK_SYNC_QRESET);
   return query_cs(cmd, HK_SYNC_QRESET_SCOPE);
}

/* The stream holding the reset is a reset stream whatever else it holds:
 * later timestamps and query commands find it through cmd->sync.last. */
void
hk_sync_query_reset_done(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   cmd->sync.epoch++;
   if (cs->sync.kind & HK_SYNC_QRESET)
      return;
   cs->sync.kind |= HK_SYNC_QRESET;
   note_last(cmd, cs);
}

/* A host-availability flush is recorded at its dependency into the open
 * compute stream when that stream already holds work and is positioned after
 * every stream of the dependency's first scope, so recording the flush there
 * creates no stream and needs none. The stream takes the dependency's waits,
 * as it did when every such flush was recorded at its dependency, unless that
 * would make work it holds wait for a render it does not wait for yet
 * (cdm_raise_delays_work()). No kind names the flush. Otherwise the flush is
 * deferred to the end of the submission (hk_sync_defer_host_flush()). */
struct hk_cs *
hk_sync_host_flush(struct hk_cmd_buffer *cmd, const VkDependencyInfo *info)
{
   const uint8_t kinds = hk_sync_host_flush_kinds(info);
   if (!kinds)
      return NULL;

   struct hk_cs *cs = cmd->current_cs.cs;
   if (cs && cs->stats.cmds && !cmd->sync.pin_cs &&
       !cmd->sync.brl_transition_active) {
      struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
      hk_sync_src_dep(cmd, kinds, cs, dep);

      bool after = !cdm_raise_delays_work(cmd, cs, &dep[HK_SYNC_VDM], false);
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
         if (dep[sq].cut && dep[sq].cut->sync.seq > cs->sync.seq)
            after = false;
      }

      if (after) {
         hk_sync_stream_wait(cs, dep);
         return cs;
      }
   }

   cmd->sync.host_flush_kinds |= kinds;
   return NULL;
}

/* A standalone G17P timestamp still needs an executable command carrier.
 * Only an otherwise empty timestamp producer uses this owned scratch store;
 * ordinary compute and barriers keep their existing streams unchanged.
 */
static void
stamp_carrier_launch(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.cs;
   if (!agx_chip_native_or_g18g(hk_cmd_buffer_device(cmd)->dev.chip) || !cs ||
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
      agx_chip_native_or_m5(hk_cmd_buffer_device(cmd)->dev.chip) &&
      cs->timestamp.start.handle;
   if (cs->timestamp.end.handle || g17p_start) {
      stamp_carrier_launch(cmd);
      hk_cmd_buffer_end_compute(cmd);
   }
}

/* Close the open compute stream before recording long independent compute work
 * (such as acceleration-structure builds) when a graphics barrier already
 * points at it. Keeping that prefix in its own stream lets the upcoming render
 * pass wait only for the work before the barrier and overlap the new compute
 * work. */
void
hk_sync_end_cs_if_gfx_waits(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.cs;
   if (!cs || !cs->stats.cmds || cmd->sync.pin_cs ||
       cmd->sync.brl_transition_active)
      return;

   const uint8_t gfx_kinds =
      HK_SYNC_GFX | HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG;
   u_foreach_bit(k, gfx_kinds) {
      if (cmd->sync.req[k][HK_SYNC_CDM].cut == cs ||
          (cmd->sync.in_pass &&
           cmd->sync.pass_req[k][HK_SYNC_CDM].cut == cs)) {
         hk_cmd_buffer_end_compute(cmd);
         return;
      }
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

/* The latest stream of any of the given kinds on the same subqueue that
 * precedes cs in the list. */
static struct hk_cs *
previous_on_sq_kinds(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                     uint8_t kinds)
{
   for (struct list_head *n = cs->node.prev; n != &cmd->control_streams;
        n = n->prev) {
      struct hk_cs *it = list_entry(n, struct hk_cs, node);
      if (it->type == cs->type && (it->sync.kind & kinds))
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
   replace_pointer(&st->brl_transition_dep[sq].cut, cs, prev);
   replace_pointer(&st->brl_transition_last[sq], cs, prev);
   replace_pointer(&st->brl_event_poll, cs, prev);
   replace_pointer(&st->stamp_copy_after[sq], cs, prev);
   replace_pointer(&st->pass_vdm, cs, NULL);
   xfer_replace_stream(st, cs, prev);
   if (st->pass_tile_first == cs) {
      st->pass_tile_first = st->pass_tile_last = NULL;
   } else if (st->pass_tile_last == cs &&
              (!prev || prev->sync.seq < st->pass_tile_first->sync.seq)) {
      st->pass_tile_last = st->pass_tile_first;
   }
   records_replace_stream(cmd, cs, prev, true);

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
   if (cs->timestamp.start.handle && !st->pending_start_ts.handle) {
      st->pending_start_ts = cs->timestamp.start;
      st->pending_start_ts_unordered = cs->unordered_start_ts;
   }
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
hk_sync_apply_dep(struct hk_cmd_buffer *cmd, struct hk_cs *into,
                  const struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   cmd->sync.epoch++;
   hk_sync_stream_wait(into, dep);
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
   if (into->sync.brl_transition || from->sync.brl_transition) {
      into->sync.brl_carry = stream_carry(into) | stream_carry(from);
      into->sync.brl_transition = true;
   }
   into->sync.kind |= from->sync.kind;
   if (from->sync.heap_slot) {
      assert(!into->sync.heap_slot ||
             into->sync.heap_slot == from->sync.heap_slot);
      into->sync.heap_slot = from->sync.heap_slot;
   }
   into->sync.att_bloom |= from->sync.att_bloom;
   into->sync.vtx_images |= from->sync.vtx_images;
   into->sync.vtx_img_bloom |= from->sync.vtx_img_bloom;
   into->sync.dep_ta_img_prior |= from->sync.dep_ta_img_prior;
   if (into->sync.vtx_img_unsafe || from->sync.vtx_img_unsafe) {
      into->sync.vtx_img_unsafe = true;
      into->sync.dep_ta[HK_SYNC_VDM].prior |= into->sync.dep_ta_img_prior;
      into->sync.dep_ta_img_prior = 0;
   }

   for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
      replace_pointer(&st->last[k][sq], from, into);
      replace_pointer(&st->req[k][sq].cut, from, into);
      replace_pointer(&st->pass_req[k][sq].cut, from, into);
   }
   replace_pointer(&st->last_any[sq], from, into);
   replace_pointer(&st->internal[sq].cut, from, into);
   replace_pointer(&st->brl_transition_dep[sq].cut, from, into);
   replace_pointer(&st->brl_transition_last[sq], from, into);
   replace_pointer(&st->brl_event_poll, from, into);
   replace_pointer(&st->stamp_copy_after[sq], from, into);
   replace_pointer(&st->pass_heap_last[sq], from, into);
   for (unsigned hs = 0; hs < HK_HEAP_SLOTS; ++hs)
      replace_pointer(&st->last_heap_slot[hs][sq], from, into);
   records_replace_stream(cmd, from, into, false);
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
/* Whether a timestamp on a compute command is ordered against earlier compute
 * commands by the logical compute subqueue alone: exact-prior barriers on
 * native AGX3 and M5 parts. */
static inline bool
cdm_stamp_queue_ordered(const struct hk_device *dev)
{
   return agx_chip_native_or_m5(dev->dev.chip) &&
          (dev->dev.params.features & DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS);
}

/* An end stamp is written when a command completes, so ordering it after
 * earlier work orders that command's completion: for a render, its fragment
 * half alone. */
static void
stamp_wait(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
           const struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   if (cs->type == HK_CS_CDM) {
      const struct hk_device *dev = hk_cmd_buffer_device(cmd);
      if (cdm_stamp_queue_ordered(dev)) {
         /* With exact-prior barriers on non-serialized AGX3 queues (native
          * AGX3 and M5), the logical compute queue already orders GPU commands
          * by their mandatory kick parents. Adding the timestamp's same-CDM
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

/* An event set on a command's end stamp (hk_sync_stamp_event()) orders the
 * event after its whole first scope, earlier submissions included: unlike a
 * timestamp, it keeps the same-CDM dependency stamp_wait() drops on
 * exact-prior queues. A primary-queue submission learns that it must wait for
 * the sibling kernel queues only from that dependency's prior, so dropping it
 * would let the event be set before an earlier submission's compute work
 * running on a sibling has completed. A render carrier orders its fragment
 * half, after which the kernel writes every retirement slot. */
static void
event_stamp_wait(struct hk_cs *cs, const struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   if (cs->type == HK_CS_CDM)
      hk_sync_stream_wait(cs, dep);
   else
      stream_wait_frag(cs, dep);
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
   return agx_chip_start_stamps_ordered(hk_cmd_buffer_device(cmd)->dev.chip);
}

static struct hk_cs *
last_stream(struct hk_cmd_buffer *cmd)
{
   if (list_is_empty(&cmd->control_streams))
      return NULL;
   return list_last_entry(&cmd->control_streams, struct hk_cs, node);
}

/* A compute stream that carries timestamps alone, or copies them: it waits
 * like work of every class, and only query commands and later timestamps
 * wait for it (HK_SYNC_STAMP). */
static struct hk_cs *
timestamp_only_stream(struct hk_cmd_buffer *cmd,
                      const struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   const uint8_t kind = cmd->sync.kind_override;
   cmd->sync.kind_override = HK_SYNC_STAMP;
   struct hk_cs *cs = hk_sync_get_cs_after(cmd, dep);
   cmd->sync.kind_override = kind;
   return cs;
}

/* Whether the end stamp of tail, the latest stream, also gives a timestamp
 * recorded now whose own first scope is dep (hk_sync_src_dep() with tail as
 * self): a command carrying that stamp alone would wait for nothing tail
 * does not, tail itself aside, so its stamp could be latched no earlier.
 * Copying tail's value is then a valid result for it, and later than every
 * earlier stamp, which is in dep or at most tail's. */
static bool
stamp_covered(struct hk_cmd_buffer *cmd, struct hk_cs *tail,
              const struct hk_sync_dep dep[HK_SYNC_NUM_SQ])
{
   const struct hk_sync_state *st = &cmd->sync;
   if (st->in_pass || st->internal_active || st->brl_transition_active)
      return false;

   /* What hk_sync_stream_created() adds for such a command: the recorded
    * requirements of every class. */
   struct hk_sync_dep need[HK_SYNC_NUM_SQ];
   req_for_kinds(cmd, HK_SYNC_STAMP, need);

   /* And at submission the requirements earlier command buffers carry into
    * every class (stream_carry()). tail takes those of fewer classes unless
    * it waits for the whole earlier prefix, which covers them. */
   const bool carried =
      !(full_kinds(HK_SYNC_STAMP) & ~stream_carry(tail));

   /* On exact-prior native AGX3/M5 queues, stamp_wait() leaves a compute
    * stream's same-subqueue CDM wait unset because the logical compute queue
    * already orders GPU execution after every earlier CDM command. tail is the
    * latest stream in the command buffer, so its end stamp already covers the
    * compute subqueue; VDM dependencies are still checked in full. */
   const bool cdm_ordered =
      tail->type == HK_CS_CDM &&
      cdm_stamp_queue_ordered(hk_cmd_buffer_device(cmd));

   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      if (sq == HK_SYNC_CDM && cdm_ordered)
         continue;

      const struct hk_sync_dep *have = &tail->sync.dep[sq];
      dep_max(&need[sq], &dep[sq]);

      if (need[sq].prior & ~have->prior)
         return false;
      if (!carried && (have->prior & HK_SYNC_STREAM_KINDS) !=
                         HK_SYNC_STREAM_KINDS)
         return false;

      /* A wait on tail's own prefix is a wait on the streams before it. */
      struct hk_cs *cut = need[sq].cut;
      if (cut == tail)
         cut = previous_on_sq(cmd, tail);
      if (cut && (!have->cut || cut->sync.seq > have->cut->sync.seq))
         return false;
   }

   return true;
}

/* Whether cs holds nothing but timestamps and timestamp copies. */
static bool
stamp_only(const struct hk_cs *cs)
{
   return !(cs->sync.kind & HK_SYNC_ALL);
}

/* Record the pending copies in cs, a timestamp-only compute stream that
 * follows the streams they read. The launch keeps it a timestamp stream. */
static void
stamp_copies_record(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_sync_state *st = &cmd->sync;
   const uint8_t kind = st->kind_override;
   st->kind_override = HK_SYNC_STAMP;
   hk_record_stamp_copies(cmd, cs, st->stamp_copies, st->stamp_copy_count);
   st->kind_override = kind;
   st->stamp_copy_count = 0;
   memset(st->stamp_copy_after, 0, sizeof(st->stamp_copy_after));
   st->epoch++;
}

/* Whether cs, a compute stream, waits for every stream the pending copies
 * read. */
static bool
stamp_copies_follow(const struct hk_cmd_buffer *cmd, const struct hk_cs *cs)
{
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      const struct hk_cs *after = cmd->sync.stamp_copy_after[sq];
      const struct hk_cs *cut = cs->sync.dep[sq].cut;
      if (after && (!cut || cut->sync.seq < after->sync.seq))
         return false;
   }
   return true;
}

void
hk_sync_flush_stamp_copies(struct hk_cmd_buffer *cmd)
{
   struct hk_sync_state *st = &cmd->sync;
   if (!st->stamp_copy_count)
      return;

   /* Copies of a leading stamp that no render stream took yet read it after
    * the command it then gets of its own. */
   if (st->lead_ts.handle)
      hk_sync_flush_lead_timestamp(cmd);

   /* And the copies write timestamp queries: after their resets. */
   struct hk_sync_dep after[HK_SYNC_NUM_SQ];
   qreset_dep(cmd, NULL, after);
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      const struct hk_sync_dep read = {.cut = st->stamp_copy_after[sq]};
      dep_max(&after[sq], &read);
   }

   /* The copies wait for the commands whose stamps they read. Recorded in
    * an open stream that holds application work, they would make that work
    * wait for those commands too, and every later dependency on the work
    * wait for the copies. They get a command of their own. */
   struct hk_cs *open = cmd->current_cs.cs;
   if (open && !st->pin_cs && !stamp_only(open))
      hk_cmd_buffer_end_compute(cmd);

   struct hk_cs *cs = timestamp_only_stream(cmd, after);
   if (!cs)
      return;

   stamp_copies_record(cmd, cs);
   /* Ended, a later first scope naming HK_SYNC_STAMP orders after it
    * whatever stream that work lands in. */
   hk_cmd_buffer_end_compute(cmd);
}

static bool
sync_state_has_req(const struct hk_sync_state *st)
{
   for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
         if (st->req[k][sq].cut || st->req[k][sq].prior)
            return true;
      }
   }
   return false;
}

/* A leading timestamp (hk_sync_state::lead_ts) that no render stream took: a
 * command of its own, ordered after its first scope, as
 * hk_sync_write_timestamp() records one. */
void
hk_sync_flush_lead_timestamp(struct hk_cmd_buffer *cmd)
{
   struct hk_sync_state *st = &cmd->sync;
   if (!st->lead_ts.handle)
      return;

   const struct agx_timestamp_req req = st->lead_ts;
   st->lead_ts = (struct agx_timestamp_req){0};
   st->epoch++;

   /* The carrier belongs to the leading timestamp alone, not to an active
    * transfer or layout transition whose compute stream triggered the flush. */
   const bool brl = st->brl_transition_active;
   const uint32_t xfer = st->xfer_depth;
   st->brl_transition_active = false;
   st->xfer_depth = 0;

   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   hk_sync_src_dep(cmd, st->lead_scope, NULL, dep);
   struct hk_cs *cs = timestamp_only_stream(cmd, dep);
   if (cs) {
      assert(!cs->timestamp.end.handle);
      cs->timestamp.end = req;
      if (st->stamp_copy_count) {
         st->stamp_copy_after[HK_SYNC_CDM] =
            hk_sync_later(st->stamp_copy_after[HK_SYNC_CDM], cs);
      }
      stamp_carrier_launch(cmd);
      hk_cmd_buffer_end_compute(cmd);
   }

   st->brl_transition_active = brl;
   st->xfer_depth = xfer;
}

void
hk_sync_flush_pending_timestamp(struct hk_cmd_buffer *cmd)
{
   struct hk_sync_state *st = &cmd->sync;
   if (st->lead_ts.handle)
      hk_sync_flush_lead_timestamp(cmd);
   if (!st->pending_start_ts.handle)
      return;

   const struct agx_timestamp_req req = st->pending_start_ts;
   const bool unordered = st->pending_start_ts_unordered;
   st->pending_start_ts = (struct agx_timestamp_req){0};
   st->pending_start_ts_unordered = false;
   /* Creating post_gfx must not take this request as its start timestamp. */
   if (unlikely(cmd->state.oq_pending_render))
      hk_flush_oq_pending(cmd);
   st->epoch++;

   struct hk_cs *tail = last_stream(cmd);
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   if (tail && !tail->timestamp.end.handle) {
      tail->timestamp.end = req;
      qreset_dep(cmd, tail, dep);
      stamp_wait(cmd, tail, dep);
      if (tail == cmd->current_cs.cs) {
         stamp_carrier_launch(cmd);
         hk_cmd_buffer_end_compute(cmd);
      }
      return;
   }

   /* Nothing to ride on: a command of its own, ordered after nothing but
    * the resets of timestamp queries. */
   qreset_dep(cmd, NULL, dep);
   struct hk_cs *cs = timestamp_only_stream(cmd, dep);
   if (!cs)
      return;
   cs->timestamp.start = req;
   cs->unordered_start_ts = unordered;
   if (unordered)
      st->has_unordered_start_ts = true;
   stamp_carrier_launch(cmd);
   hk_cmd_buffer_end_compute(cmd);
}

bool
hk_sync_can_defer_unordered_start_ts(struct hk_cmd_buffer *cmd)
{
   const struct hk_sync_state *st = &cmd->sync;

   return !last_stream(cmd) && !st->pending_start_ts.handle &&
          !st->lead_ts.handle && !sync_state_has_req(st);
}

void
hk_sync_write_timestamp(struct hk_cmd_buffer *cmd, VkPipelineStageFlags2 stage,
                        struct agx_timestamp_req req)
{
   if (unlikely(cmd->state.oq_pending_render))
      hk_flush_oq_pending(cmd);
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
          !gfx->timestamp.end.handle && start_stamps_ordered(cmd)) {
         struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
         qreset_dep(cmd, gfx, dep);
         hk_sync_stream_wait(gfx, dep);
         gfx->timestamp.start = req;
         return;
      }

      struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
      hk_sync_src_dep(cmd, hk_sync_stamp_scope(src_kinds) | HK_SYNC_QRESET,
                      gfx, dep);
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
         /* A copy of the render's end stamp, pending with the command
          * buffer's other copies until they launch together in a
          * timestamp-only command after the render
          * (hk_sync_flush_stamp_copies()). Application work recorded after
          * the render does not wait for it. A full batch launches first. */
         if (st->stamp_copy_count == HK_SYNC_STAMP_COPIES)
            hk_sync_flush_stamp_copies(cmd);
         if (st->stamp_copy_count == HK_SYNC_STAMP_COPIES)
            return;
         assert(!gfx->timestamp.end.event);
         st->stamp_copies[st->stamp_copy_count++] = (struct hk_stamp_copy){
            .dst = req,
            .src = gfx->timestamp.end,
         };
         st->stamp_copy_after[HK_SYNC_VDM] =
            hk_sync_later(st->stamp_copy_after[HK_SYNC_VDM], gfx);
      } else {
         gfx->timestamp.end = req;
      }
      return;
   }

   if (!src_kinds) {
      if (start_stamps_ordered(cmd)) {
         hk_sync_flush_pending_timestamp(cmd);
         st->pending_start_ts = req;
         st->pending_start_ts_unordered = false;
         /* A query reset or earlier dispatch may have left a reusable stream.
          * Consume the start stamp before recording work into it too, rather
          * than resolving it as an end stamp at the next timestamp write. */
         if (agx_chip_native_or_m5(hk_cmd_buffer_device(cmd)->dev.chip))
            st->cs_check_pending = true;
         return;
      }

      /* A leading TOP_OF_PIPE timestamp with no earlier work or barrier in
       * this command buffer can ride on the first stream's start stamp when
       * submission confirms no prior queue barrier, input wait, or live
       * cross-subqueue work precedes it (hk_queue.c); otherwise submission
       * moves it to a preceding end-stamp carrier. */
      if (hk_sync_can_defer_unordered_start_ts(cmd)) {
         st->pending_start_ts = req;
         st->pending_start_ts_unordered = true;
         return;
      }
   }

   /* Another stamp while the leading one still waits for the command
    * buffer's first stream: a copy of it. The first stream carries the
    * leading stamp after the first scopes of both, a valid later latch for
    * the earlier one, and the copy launches after that stream
    * (hk_sync_flush_stamp_copies()). Neither gets a command of its own. */
   if (st->lead_ts.handle && st->stamp_copy_count < HK_SYNC_STAMP_COPIES) {
      assert(!last_stream(cmd));
      st->lead_scope |= src_kinds ? hk_sync_stamp_scope(src_kinds) | HK_SYNC_QRESET
                                  : HK_SYNC_QRESET;
      st->stamp_copies[st->stamp_copy_count++] = (struct hk_stamp_copy){
         .dst = req,
         .src = st->lead_ts,
      };
      return;
   }

   hk_sync_flush_pending_timestamp(cmd);

   const uint8_t scope = hk_sync_stamp_scope(src_kinds) | HK_SYNC_QRESET;
   struct hk_cs *tail = last_stream(cmd);
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   hk_sync_src_dep(cmd, scope, tail, dep);

   if (tail && !tail->timestamp.end.handle) {
      tail->timestamp.end = req;
      stamp_wait(cmd, tail, dep);

      /* Later compute would move the stamp past itself; keep the stream
       * boundary so the stamp stays at the recorded position. */
      if (tail == cmd->current_cs.cs) {
         stamp_carrier_launch(cmd);
         hk_cmd_buffer_end_compute(cmd);
      } else if (unlikely(cmd->pass_merge.valid) &&
                 cmd->pass_merge.cs == tail && last_stream(cmd) == tail &&
                 cmd->sync.next_seq == cmd->pass_merge.next_seq) {
         /* Attaching an end timestamp to the open pass-merge candidate
          * preserves candidacy: a merged render latches it when the merged
          * pass finishes, still after every prior command. */
         cmd->pass_merge.sync_epoch = cmd->sync.epoch;
      }
      return;
   }

   /* Nothing was recorded since tail took its end stamp and nothing this
    * stamp waits for is missing from it: copy that stamp instead of
    * recording a command for this one alone. */
   if (tail && st->stamp_copy_count < HK_SYNC_STAMP_COPIES &&
       !tail->timestamp.end.event && stamp_covered(cmd, tail, dep)) {
      const unsigned sq = hk_sync_sq(tail);
      st->stamp_copies[st->stamp_copy_count++] = (struct hk_stamp_copy){
         .dst = req,
         .src = tail->timestamp.end,
      };
      st->stamp_copy_after[sq] = hk_sync_later(st->stamp_copy_after[sq], tail);
      if (unlikely(cmd->pass_merge.valid) && cmd->pass_merge.cs == tail &&
          last_stream(cmd) == tail &&
          cmd->sync.next_seq == cmd->pass_merge.next_seq)
         cmd->pass_merge.sync_epoch = cmd->sync.epoch;
      return;
   }

   /* A stamp ahead of every stream of the command buffer, on a chip whose
    * start stamps latch before barriers resolve: a first render stream
    * carries it on its vertex end stamp and waits for its first scope
    * (hk_sync_state::lead_ts), avoiding a cross-engine carrier ahead of the
    * render. If a compute stream is opened first instead, it flushes lead_ts
    * into a preceding carrier (hk_sync_flush_lead_timestamp()). A top-of-pipe
    * stamp keeps the wait on the resets of timestamp queries alone, as its
    * first scope is empty.
    */
   if (!tail && !start_stamps_ordered(cmd)) {
      st->lead_ts = req;
      st->lead_scope = src_kinds ? scope : HK_SYNC_QRESET;
      return;
   }

   /* A command of its own, after every earlier stream in the first scope,
    * tail included: tail is no stream this stamp rides on. */
   hk_sync_end_cs_if_timestamped(cmd);
   hk_sync_src_dep(cmd, scope, NULL, dep);
   struct hk_cs *cs = timestamp_only_stream(cmd, dep);
   if (!cs)
      return;
   assert(!cs->timestamp.end.handle);
   cs->timestamp.end = req;
   /* The pending copies ride on it when it follows the stamps they read. */
   if (st->stamp_copy_count && stamp_only(cs) && stamp_copies_follow(cmd, cs))
      stamp_copies_record(cmd, cs);
   stamp_carrier_launch(cmd);
   hk_cmd_buffer_end_compute(cmd);
}

/*
 * Stamp a vkCmdSetEvent2 directly onto the last recorded stream's kernel
 * retirement timestamp on HAL200/M5 when it covers the event's first scope
 * and has a free host-written retirement slot. Avoids opening a standalone
 * CDM store stream and the cross-engine VDM->CDM->VDM stalls it causes.
 */
bool
hk_sync_stamp_event(struct hk_cmd_buffer *cmd, struct agx_timestamp_req req,
                    uint8_t src_kinds)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_sync_state *st = &cmd->sync;

   if (!dev || !req.handle || !src_kinds || st->in_pass ||
       !hk_oq_render_stamps_supported(dev->dev.chip) ||
       !(dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS) ||
       st->host_flush_kinds || cmd->state.oq_pending_render)
      return false;

   if (st->lead_ts.handle)
      return false;

   /* Flush any earlier stamp copies and pending top-of-pipe timestamps so
    * tail is positioned after them before we inspect it. */
   hk_sync_flush_stamp_copies(cmd);
   hk_sync_flush_pending_timestamp(cmd);

   struct hk_cs *tail = last_stream(cmd);
   if (!tail)
      return false;

   if (tail->type == HK_CS_VDM) {
      /* Only use the kernel retirement-written slots (vertex end, fragment
       * start, vertex start); never fragment end (hk_query_oq.h).
       *
       * The kernel publishes a render's timestamps on the host in slot order
       * (vertex start, vertex end, fragment start, fragment end). A host
       * polling vkGetEventStatus() may read any query in the event's first
       * scope once it sees the event set, so the event takes a slot after
       * every slot already holding a query of this render: a fragment-end
       * timestamp query rules out all three. The same holds when the event
       * already rides on this render and is set again. */
      const bool used[4] = {
         tail->timestamp.start.handle || tail->oq_stamp[0].handle,
         tail->timestamp.vtx_end.handle || tail->oq_stamp[1].handle,
         tail->oq_stamp[2].handle,
         tail->timestamp.end.handle || tail->oq_stamp[3].handle,
      };
      int same = -1, last_query = -1;
      for (unsigned s = 0; s < ARRAY_SIZE(used); ++s) {
         if (tail->oq_stamp[s].handle == req.handle &&
             tail->oq_stamp[s].offset_B == req.offset_B)
            same = (int)s;
         else if (used[s])
            last_query = (int)s;
      }

      int slot = -1;
      if (same >= 0) {
         slot = same > last_query ? same : -1;
      } else {
         const unsigned order[] = {1, 2, 0};
         for (unsigned i = 0; i < ARRAY_SIZE(order); ++i) {
            if ((int)order[i] > last_query && !used[order[i]]) {
               slot = (int)order[i];
               break;
            }
         }
      }
      if (slot < 0)
         return false;

      struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
      hk_sync_src_dep(cmd, hk_sync_stamp_scope(src_kinds), tail, dep);
      event_stamp_wait(tail, dep);
      tail->oq_stamp[slot] = req;
      cmd->pass_merge.valid = false;
      const unsigned stamp_k = util_logbase2(HK_SYNC_STAMP);
      st->last[stamp_k][HK_SYNC_VDM] =
         hk_sync_later(st->last[stamp_k][HK_SYNC_VDM], tail);
      st->epoch++;
      return true;
   }

   if (tail->type == HK_CS_CDM) {
      /* A compute command has a start and an end slot: the event takes the
       * end, after any start-stamp query, or shares it with an earlier set
       * of the same event. */
      if (tail->oq_avail_deferred)
         return false;
      if (tail->timestamp.end.handle &&
          (tail->timestamp.end.handle != req.handle ||
           tail->timestamp.end.offset_B != req.offset_B))
         return false;

      struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
      hk_sync_src_dep(cmd, hk_sync_stamp_scope(src_kinds), tail, dep);
      event_stamp_wait(tail, dep);
      tail->timestamp.end = req;
      const unsigned stamp_k = util_logbase2(HK_SYNC_STAMP);
      st->last[stamp_k][HK_SYNC_CDM] =
         hk_sync_later(st->last[stamp_k][HK_SYNC_CDM], tail);
      st->epoch++;
      if (tail == cmd->current_cs.cs) {
         stamp_carrier_launch(cmd);
         hk_cmd_buffer_end_compute(cmd);
      }
      return true;
   }

   return false;
}

static void
flush_pass_heap_slot(struct hk_sync_state *st)
{
   if (!st->pass_heap_slot)
      return;
   const uint8_t slot = st->pass_heap_slot - 1;
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      if (st->pass_heap_last[sq])
         st->last_heap_slot[slot][sq] =
            hk_sync_later(st->last_heap_slot[slot][sq], st->pass_heap_last[sq]);
      st->pass_heap_last[sq] = NULL;
   }
   st->pass_heap_slot = 0;
}

void
hk_sync_begin_rendering(struct hk_cmd_buffer *cmd, bool resuming)
{
   struct hk_sync_state *st = &cmd->sync;
   if (resuming && st->pending_start_ts_unordered)
      hk_sync_flush_pending_timestamp(cmd);
   flush_pass_heap_slot(st);
   st->epoch++;
   st->in_pass = true;
   st->pass_start_seq = st->next_seq;
   memset(st->pass_req, 0, sizeof(st->pass_req));
   st->pass_vdm = NULL;
   st->resume_pending = resuming;
   st->heap_pass = false;
   st->pass_fp_count = 0;
   memset(st->pass_tile_wait, 0, sizeof(st->pass_tile_wait));
   st->pass_tile_wait_frag = NULL;
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
   flush_pass_heap_slot(st);
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
   cs->sync.att_bloom |= cmd->state.gfx.render.att_bloom;
   if (cs->sync.vtx_images) {
      if (cs->sync.vtx_img_unsafe) {
         struct hk_sync_dep img[HK_SYNC_NUM_SQ];
         req_for_mask(cmd, HK_SYNC_GFX_IMG, img);
         hk_sync_stream_wait(cs, img);
         cs->sync.vtx_images_epoch = cmd->sync.epoch;
      } else {
         hk_sync_vertex_images_bloom(cmd, cs, cs->sync.vtx_img_bloom);
      }
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
   flush_pass_heap_slot(st);
   st->pass_fp_count = 0;
   memset(st->pass_tile_wait, 0, sizeof(st->pass_tile_wait));
   st->pass_tile_wait_frag = NULL;
   st->pass_tile_first = st->pass_tile_last = cs;
   st->suspended_last = NULL;
}

void
hk_sync_heap_used(struct hk_cmd_buffer *cmd)
{
   if (unlikely(cmd->state.oq_pending_render))
      hk_flush_oq_pending(cmd);
   struct hk_sync_state *st = &cmd->sync;
   st->heap_pass = true;
   st->epoch++;

   /* The streams of this pass already created wait like the ones still to
    * come. */
   if (cmd->current_cs.gfx)
      hk_sync_stream_wait_all(cmd, cmd->current_cs.gfx);
   if (cmd->current_cs.pre_gfx)
      hk_sync_stream_wait_all(cmd, cmd->current_cs.pre_gfx);
   if (cmd->current_cs.post_gfx) {
      /* A post-graphics stream opened for query availability alone carries
       * HK_SYNC_STAMP. Once this pass uses the geometry heap, hk_heap()
       * appends the heap-reset write to it, so promote it to HK_SYNC_GFX so
       * later heap passes and graphics barriers wait for the reset. */
      struct hk_cs *post = cmd->current_cs.post_gfx;
      if (!(post->sync.kind & HK_SYNC_GFX)) {
         post->sync.kind |= HK_SYNC_GFX | HK_SYNC_GFX_MEM;
         note_last(cmd, post);
      }
      hk_sync_stream_wait_all(cmd, post);
   }
}

void
hk_sync_heap_slot_used(struct hk_cmd_buffer *cmd, uint8_t slot)
{
   assert(slot < HK_HEAP_SLOTS);
   if (unlikely(cmd->state.oq_pending_render))
      hk_flush_oq_pending(cmd);
   struct hk_sync_state *st = &cmd->sync;
   flush_pass_heap_slot(st);
   st->pass_heap_slot = slot + 1;
   st->epoch++;

   struct hk_sync_dep hdep[HK_SYNC_NUM_SQ] = {
      {.cut = st->last_heap_slot[slot][HK_SYNC_VDM]},
      {.cut = st->last_heap_slot[slot][HK_SYNC_CDM]},
   };
   struct hk_cs *streams[3] = {
      cmd->current_cs.pre_gfx,
      cmd->current_cs.gfx,
      cmd->current_cs.post_gfx,
   };
   for (unsigned i = 0; i < 3; ++i) {
      struct hk_cs *s = streams[i];
      if (!s)
         continue;
      s->sync.heap_slot = slot + 1;
      hk_sync_stream_wait(s, hdep);
      const unsigned sq = hk_sync_sq(s);
      st->pass_heap_last[sq] = hk_sync_later(st->pass_heap_last[sq], s);
   }
}

void
hk_sync_resolve_begin(struct hk_cmd_buffer *cmd)
{
   if (unlikely(cmd->state.oq_pending_render))
      hk_flush_oq_pending(cmd);
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

void
hk_sync_fold_resolve(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                     const struct hk_image *dst_image, uint32_t level,
                     uint32_t layer, uint32_t width, uint32_t height)
{
   struct hk_sync_state *st = &cmd->sync;
   assert(!st->in_pass && cs->type == HK_CS_VDM);
   assert(cs == list_last_entry(&cmd->control_streams, struct hk_cs, node));
   st->epoch++;

   /* Apply pending transfer requirements to cs's fragment half. If an
    * intervening render->transfer barrier overwrote the VDM cut with cs
    * itself, conservatively preserve ordering against the latest VDM stream
    * strictly before cs so earlier passes touching dst_image stay ordered. */
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   req_for_mask(cmd, HK_SYNC_XFER, dep);
   if (dep[HK_SYNC_VDM].cut == cs)
      dep[HK_SYNC_VDM].cut = latest_before(cmd, HK_SYNC_VDM, cs);
   stream_wait_frag(cs, dep);
   if (st->internal_active)
      stream_wait_frag(cs, st->internal);

   const struct hk_sync_footprint fp = {
      .preserve = false,
      .image = &dst_image->vk,
      .level = level,
      .layer0 = layer,
      .layer1 = layer + 1,
      .cx0 = 0,
      .cx1 = (int32_t)((width - 1) >> HK_SYNC_TILE_CELL_LOG2),
      .cy0 = 0,
      .cy1 = (int32_t)((height - 1) >> HK_SYNC_TILE_CELL_LOG2),
   };
   struct hk_sync_dep tile_wait[HK_SYNC_NUM_SQ] = {{0}, {0}};
   util_dynarray_foreach(&cmd->sync_pass_records, struct hk_sync_pass_record,
                         rec) {
      if (!rec->last || rec->last == cs)
         continue;
      if (footprints_share_tile(&rec->fp, &fp)) {
         const unsigned sq = hk_sync_sq(rec->last);
         tile_wait[sq].cut = hk_sync_later(tile_wait[sq].cut, rec->last);
      }
   }
   stream_wait_frag(cs, tile_wait);

   const struct hk_sync_pass_record rec = {
      .fp = fp,
      .first = cs,
      .last = cs,
   };
   util_dynarray_append(&cmd->sync_pass_records, rec);

   const VkImageCreateFlags unsafe_flags =
      VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
      VK_IMAGE_CREATE_SPARSE_ALIASED_BIT;
   if (dst_image->plane_count == 0 || dst_image->swapchain_memory ||
       dst_image->vk.external_handle_types ||
       (dst_image->vk.create_flags & unsafe_flags) ||
       dst_image->vk.tiling == VK_IMAGE_TILING_LINEAR ||
       !dst_image->planes[0].addr) {
      cs->sync.att_bloom = UINT64_MAX;
   } else {
      cs->sync.att_bloom |= hk_sync_image_bloom(dst_image->planes[0].addr);
   }

   cs->sync.kind |= HK_SYNC_XFER;
   u_foreach_bit(k, HK_SYNC_XFER)
      st->last[k][HK_SYNC_VDM] = cs;
   st->last_any[HK_SYNC_VDM] = cs;
   if (dst_image->vk.tiling == VK_IMAGE_TILING_LINEAR)
      hk_sync_note_memory_write(cmd, cs);
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
   hk_sync_value last_heap_slot[HK_HEAP_SLOTS][HK_SYNC_NUM_SQ];
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

bool
hk_sync_host_flush_prefix(struct hk_sync_queue *q, uint8_t kinds,
                          const uint64_t base[HK_SYNC_NUM_SQ],
                          const uint32_t count[HK_SYNC_NUM_SQ],
                          uint16_t barrier[HK_SYNC_NUM_SQ],
                          uint64_t prior[HK_SYNC_NUM_SQ])
{
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      barrier[sq] = DRM_ASAHI_BARRIER_NONE;
      prior[sq] = 0;

      if (q->exact_prior) {
         /* What resolve_one() gives a stream of every kind whose first
          * scope is `kinds`, at the end of the submission: the latest
          * command of those kinds, and every requirement still carried, as
          * a stream recorded at the dependency itself would have picked up;
          * and the submission's last command on the subqueue (see
          * hk_sync.h). A command of this submission within the 16-bit field
          * is named by its position, any other by its ordinal. Waits are
          * prefixes. */
         uint64_t v = 0;
         u_foreach_bit(k, kinds)
            v = MAX2(v, q->last_absolute[k][sq]);
         for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k)
            v = MAX2(v, q->tail[k][sq]);
         v = MAX2(v, q->wait_floor[sq]);
         if (count[sq])
            v = MAX2(v, base[sq] + count[sq]);
         if (v > base[sq] && v - base[sq] < DRM_ASAHI_BARRIER_NONE)
            barrier[sq] = (uint16_t)(v - base[sq]);
         else
            prior[sq] = v;
      } else if (count[sq]) {
         /* Legacy prefixes carry no history to resolve against here: follow
          * the submission's last command on the subqueue, */
         barrier[sq] = MIN2(count[sq], DRM_ASAHI_BARRIER_NONE - 1);
      } else if (q->ever[sq]) {
         /* or with none, every command of earlier ioctls. */
         barrier[sq] = 0;
      }
   }

   if (q->exact_prior) {
      if (q->submitted[HK_SYNC_CDM] == UINT64_MAX)
         return false;
      q->submitted[HK_SYNC_CDM]++;
   }
   return true;
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

bool
hk_sync_submit_has_unretired_priors(const struct hk_queue *queue)
{
   const struct hk_sync_queue *q = &queue->sync;
   return q->exact_prior && q->retired_count > 0 &&
          (q->submitted[HK_SYNC_VDM] > q->retired_ord[HK_SYNC_VDM] ||
           q->submitted[HK_SYNC_CDM] > q->retired_ord[HK_SYNC_CDM]);
}

static void
sync_advance_retired(struct hk_sync_queue *q, uint64_t completed)
{
   const uint32_t n = MIN2(q->retired_count, HK_SYNC_RETIRED_RING);
   for (uint32_t i = 0; i < n; ++i) {
      if (q->retired[i].point && q->retired[i].point <= completed) {
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
            q->retired_ord[sq] = MAX2(q->retired_ord[sq], q->retired[i].ord[sq]);
      }
   }

   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      const uint64_t ret = q->retired_ord[sq];
      if (!ret)
         continue;
      for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
         if (q->last_absolute[k][sq] <= ret)
            q->last_absolute[k][sq] = 0;
         if (q->tail[k][sq] <= ret)
            q->tail[k][sq] = 0;
      }
   }
}

static void
sync_record_retired(struct hk_sync_queue *q, uint64_t point)
{
   if (!point)
      return;
   q->retired[q->retired_next].point = point;
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
      q->retired[q->retired_next].ord[sq] = q->submitted[sq];
   q->retired_next = (q->retired_next + 1) % HK_SYNC_RETIRED_RING;
   q->retired_count = MIN2(q->retired_count + 1, HK_SYNC_RETIRED_RING);
}

/* Whether a footprint of an earlier render on the render subqueue that
 * shares a tile with fp needs only fp's fragment half ordered after it
 * (hk_sync_queue::load_hazard_frag). */
static bool
load_hazard_frag(const struct hk_sync_queue *q, enum hk_sync_sq sq,
                 const struct hk_sync_footprint *earlier,
                 const struct hk_sync_footprint *fp)
{
   return q->load_hazard_frag && sq == HK_SYNC_VDM &&
          footprints_whole_tiles(earlier, fp);
}

/* The subqueues a possibly still running earlier submission shares a tile of
 * fp on, and in *frag_only those whose shared tiles all need only fp's
 * fragment half ordered (load_hazard_frag()). */
static uint8_t
recent_conflict(const struct hk_sync_queue *q, const struct hk_sync_footprint *fp,
                uint64_t completed, const struct recent_tile_index *index,
                uint8_t *frag_only)
{
   *frag_only = 0;
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
   uint8_t engines = 0, whole = 0;
   if (index) {
      for (uint16_t i = index->head[tile_bucket(fp)]; i != UINT16_MAX;
           i = index->next[i]) {
         const struct hk_sync_recent *r = &q->recent[i];
         const uint8_t engine = BITFIELD_BIT(r->sq);
         if ((whole & engine) || !footprints_share_tile(&r->fp, fp))
            continue;
         if (load_hazard_frag(q, r->sq, &r->fp, fp))
            engines |= engine;
         else
            whole |= engine;
      }
   } else {
      const uint32_t n = MIN2(q->recent_count, HK_SYNC_RECENT_FOOTPRINTS);
      for (uint32_t i = 0; i < n; ++i) {
         const struct hk_sync_recent *r = &q->recent[i];
         if (r->point <= completed || !footprints_share_tile(&r->fp, fp))
            continue;
         if (load_hazard_frag(q, r->sq, &r->fp, fp))
            engines |= BITFIELD_BIT(r->sq);
         else
            whole |= BITFIELD_BIT(r->sq);
      }
   }
   *frag_only = engines & ~whole;
   return engines | whole;
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

static void
vdm_att_ring_add(struct hk_sync_queue *q, uint64_t ordinal, uint64_t att_bloom)
{
   struct hk_sync_vdm_att *slot = &q->recent_vdm_att[q->recent_vdm_att_next];
   if (q->recent_vdm_att_count >= HK_SYNC_RECENT_VDM_ATT)
      q->recent_vdm_att_evicted =
         MAX2(q->recent_vdm_att_evicted, slot->ordinal);
   *slot = (struct hk_sync_vdm_att){
      .ordinal = ordinal,
      .att_bloom = att_bloom,
   };
   q->recent_vdm_att_next =
      (q->recent_vdm_att_next + 1) % HK_SYNC_RECENT_VDM_ATT;
   q->recent_vdm_att_count =
      MIN2(q->recent_vdm_att_count + 1, HK_SYNC_RECENT_VDM_ATT);
}

static uint64_t
resolve_exact_vdm_ta_bloom(const struct submit_state *ss,
                           const struct hk_cs *cs,
                           const struct hk_sync_dep *t)
{
   const struct hk_sync_queue *q = &ss->queue->sync;
   const uint64_t base = q->submitted[HK_SYNC_VDM];

   uint64_t v_base = resolve_absolute(ss, t, HK_SYNC_VDM);
   u_foreach_bit(k, cs->sync.kind)
      v_base = MAX2(v_base, ss->carry_absolute[k][HK_SYNC_VDM]);

   uint64_t v_img_cap =
      ss->carry_absolute[util_logbase2(HK_SYNC_GFX_IMG)][HK_SYNC_VDM];
   u_foreach_bit(k, cs->sync.dep_ta_img_prior) {
      const uint64_t earlier = ss->last[k][HK_SYNC_VDM] > 0
                                  ? base + ss->last[k][HK_SYNC_VDM]
                                  : q->last_absolute[k][HK_SYNC_VDM];
      v_img_cap = MAX2(v_img_cap, earlier);
   }

   uint64_t v_need = v_base;
   if (v_img_cap > v_base && cs->sync.vtx_img_bloom) {
      if (q->recent_vdm_att_evicted)
         v_need = MAX2(v_need, MIN2(v_img_cap, q->recent_vdm_att_evicted));
      const uint32_t n = MIN2(q->recent_vdm_att_count, HK_SYNC_RECENT_VDM_ATT);
      for (uint32_t j = 0; j < n; ++j) {
         const struct hk_sync_vdm_att *e = &q->recent_vdm_att[j];
         if (e->ordinal > v_need && e->ordinal <= v_img_cap &&
             (e->att_bloom & cs->sync.vtx_img_bloom))
            v_need = e->ordinal;
      }
   }

   return v_need;
}

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
      v = MAX2(v, ss->queue->sync.wait_floor[sq]);
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
hk_sync_resolve_submit_ex(struct hk_queue *queue,
                          struct hk_cmd_buffer *const *cmdbufs, unsigned count,
                          bool first_cdm_is_flush, bool chain_first,
                          uint64_t completed, struct hk_sync_resolved *out,
                          bool *tail_raised_out)
{
   if (tail_raised_out)
      *tail_raised_out = false;
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
      if (completed)
         sync_advance_retired(&queue->sync, completed);
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
   for (unsigned hs = 0; hs < HK_HEAP_SLOTS; ++hs) {
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
         ss.last_heap_slot[hs][sq] = HK_SYNC_NONE;
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
      const bool alias_mem_stale =
         cmd->sync.has_alias_image_render &&
         cmd->sync.buffer_alias_mem_gen !=
            p_atomic_read(&hk_cmd_buffer_device(cmd)->buffer_alias_mem_gen);

      /* Positions first: cuts point at streams of this command buffer. */
      list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
         cs->sync.index = ++ss.count[hk_sync_sq(cs)];
         if (unlikely(alias_mem_stale) && cs->sync.alias_image_render)
            cs->sync.kind |= HK_SYNC_GFX_MEM;
      }

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
             * destination classes, its vertex half on those of its kinds. A
             * layout transition waits on those of its first scope alone. */
            const uint8_t carry = stream_carry(cs);
            const bool exact_vdm_bloom =
               sq == HK_SYNC_VDM && queue->sync.exact_prior &&
               !cs->sync.brl_transition && cs->sync.vtx_images &&
               !cs->sync.vtx_img_unsafe;
            const uint8_t ta_carry =
               cs->sync.brl_transition
                  ? cs->sync.brl_carry
                  : cs->sync.kind |
                       (cs->sync.vtx_images ? HK_SYNC_GFX_IMG : 0);
            resolve_one(&ss, sq, d, carry, &cs->sync.barrier[sq],
                        &cs->sync.prior[sq]);
            if (exact_vdm_bloom) {
               const uint64_t v_need = resolve_exact_vdm_ta_bloom(&ss, cs, t);
               const uint64_t base = queue->sync.submitted[HK_SYNC_VDM];
               if (v_need > base && v_need - base < DRM_ASAHI_BARRIER_NONE) {
                  cs->sync.barrier_ta[HK_SYNC_VDM] = (uint16_t)(v_need - base);
                  cs->sync.prior_ta[HK_SYNC_VDM] = 0;
               } else {
                  cs->sync.barrier_ta[HK_SYNC_VDM] = DRM_ASAHI_BARRIER_NONE;
                  cs->sync.prior_ta[HK_SYNC_VDM] = v_need;
               }
            } else {
               struct hk_sync_dep t_eff = *t;
               if (sq == HK_SYNC_VDM && cs->sync.dep_ta_img_prior)
                  t_eff.prior |= cs->sync.dep_ta_img_prior;
               if (ta_carry == carry && t_eff.cut == d->cut &&
                   t_eff.prior == d->prior) {
                  cs->sync.barrier_ta[sq] = cs->sync.barrier[sq];
                  cs->sync.prior_ta[sq] = cs->sync.prior[sq];
               } else {
                  resolve_one(&ss, sq, &t_eff, ta_carry,
                              &cs->sync.barrier_ta[sq],
                              &cs->sync.prior_ta[sq]);
               }
            }
            if (unlikely(alias_mem_stale) && cs->type != HK_CS_CDM &&
                sq == HK_SYNC_VDM) {
               cs->sync.barrier_ta[sq] = cs->sync.barrier[sq];
               cs->sync.prior_ta[sq] = cs->sync.prior[sq];
            }
            if (cs->sync.heap_slot > 0) {
               const unsigned hs = cs->sync.heap_slot - 1;
               if (ss.last_heap_slot[hs][sq] > 0) {
                  const hk_sync_value hv = ss.last_heap_slot[hs][sq];
                  barrier_at_least(queue, sq, &cs->sync.barrier[sq],
                                   &cs->sync.prior[sq], hv);
                  if (cs->sync.barrier[sq] != DRM_ASAHI_BARRIER_NONE)
                     cs->sync.prior[sq] = 0;
                  barrier_at_least(queue, sq, &cs->sync.barrier_ta[sq],
                                   &cs->sync.prior_ta[sq], hv);
                  if (cs->sync.barrier_ta[sq] != DRM_ASAHI_BARRIER_NONE)
                     cs->sync.prior_ta[sq] = 0;
               } else if (queue->sync.exact_prior &&
                          queue->sync.last_heap_slot[hs][sq]) {
                  const uint64_t prev = queue->sync.last_heap_slot[hs][sq];
                  if (cs->sync.barrier[sq] == DRM_ASAHI_BARRIER_NONE)
                     cs->sync.prior[sq] = MAX2(cs->sync.prior[sq], prev);
                  if (cs->sync.barrier_ta[sq] == DRM_ASAHI_BARRIER_NONE)
                     cs->sync.prior_ta[sq] = MAX2(cs->sync.prior_ta[sq], prev);
               } else if (!queue->sync.exact_prior &&
                          (queue->sync.heap_slot_ever[sq] & BITFIELD_BIT(hs))) {
                  barrier_at_least(queue, sq, &cs->sync.barrier[sq],
                                   &cs->sync.prior[sq], 0);
                  barrier_at_least(queue, sq, &cs->sync.barrier_ta[sq],
                                   &cs->sync.prior_ta[sq], 0);
               }
            }
         }
      }

      /* Tile hazards: an instance sharing a tile with an instance of an
       * earlier command buffer of this submission waits on that one, and
       * with a possibly still running earlier submission on everything
       * before this submission (before this ioctl on queues without exact
       * prior ordinals). Within the command buffer the recorded
       * dependency already orders it. The vertex half waits too, as for the
       * in-pass hazard (hk_sync_stream_created()), unless every such tile is
       * a whole tile an earlier render stored and this one loads
       * (load_hazard_frag()): then the fragment half alone (vf). */
      util_dynarray_foreach(&cmd->sync_pass_records,
                            struct hk_sync_pass_record, rec) {
         if (!rec->first || !rec->last)
            continue;
         hk_sync_value v[HK_SYNC_NUM_SQ] = {HK_SYNC_NONE, HK_SYNC_NONE};
         hk_sync_value vf[HK_SYNC_NUM_SQ] = {HK_SYNC_NONE, HK_SYNC_NONE};
#define SUBMITTED_CONFLICT(e)                                                  \
   do {                                                                        \
      if (footprints_share_tile(&(e)->fp, &rec->fp)) {                          \
         hk_sync_value *slot =                                                 \
            load_hazard_frag(&queue->sync, (e)->sq, &(e)->fp, &rec->fp)         \
               ? &vf[(e)->sq]                                                  \
               : &v[(e)->sq];                                                  \
         *slot = value_max(*slot, (e)->index);                                 \
      }                                                                        \
   } while (0)
         if (indexed) {
            for (uint32_t j = submitted_head[tile_bucket(&rec->fp)];
                 j != TILE_INDEX_NONE;) {
               const struct submit_footprint *e = util_dynarray_element(
                  &submitted, struct submit_footprint, j);
               SUBMITTED_CONFLICT(e);
               j = e->next_by_image;
            }
         } else {
            util_dynarray_foreach(&submitted, struct submit_footprint, e)
               SUBMITTED_CONFLICT(e);
         }
#undef SUBMITTED_CONFLICT
         uint8_t recent_frag;
         const uint8_t recent =
            recent_conflict(&queue->sync, &rec->fp, completed,
                            indexed ? &recent_index : NULL, &recent_frag);
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
            struct hk_sync_stream *s = &rec->first->sync;
            const bool earlier = recent & BITFIELD_BIT(sq);
            const bool earlier_ta =
               earlier && !(recent_frag & BITFIELD_BIT(sq));
            /* A prefix in this submission also covers every earlier one. */
            hk_sync_value whole = value_max(v[sq], vf[sq]);
            hk_sync_value ta = v[sq];
            if (queue->sync.exact_prior) {
               /* Everything before this submission, by its ordinal: a zero
                * barrier would name everything before the ioctl that
                * carries the render, which for a later ioctl of a split
                * submission is more than the whole render's prefix may
                * name. */
               const uint64_t start = queue->sync.submitted[sq];
               if (whole == HK_SYNC_NONE && earlier)
                  s->prior[sq] = MAX2(s->prior[sq], start);
               if (ta == HK_SYNC_NONE && earlier_ta)
                  s->prior_ta[sq] = MAX2(s->prior_ta[sq], start);
            } else {
               if (whole == HK_SYNC_NONE && earlier)
                  whole = 0;
               if (ta == HK_SYNC_NONE && earlier_ta)
                  ta = 0;
            }
            if (whole != HK_SYNC_NONE)
               barrier_at_least(queue, sq, &s->barrier[sq], &s->prior[sq],
                                whole);
            if (ta != HK_SYNC_NONE)
               barrier_at_least(queue, sq, &s->barrier_ta[sq],
                                &s->prior_ta[sq], ta);
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
         if (cs->sync.heap_slot > 0) {
            const unsigned hs = cs->sync.heap_slot - 1;
            const unsigned sq = hk_sync_sq(cs);
            ss.last_heap_slot[hs][sq] =
               value_max(ss.last_heap_slot[hs][sq], cs->sync.index);
         }
      }
      if (unlikely(alias_mem_stale)) {
         const unsigned mem_k = util_logbase2(HK_SYNC_GFX_MEM);
         list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
            if (cs->sync.alias_image_render) {
               const unsigned sq = hk_sync_sq(cs);
               ss.last[mem_k][sq] = value_max(ss.last[mem_k][sq], cs->sync.index);
            }
         }
      }

      if (queue->sync.exact_prior) {
         const uint64_t vdm_base = queue->sync.submitted[HK_SYNC_VDM];
         list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
            if (cs->type != HK_CS_VDM)
               continue;
            const uint64_t bloom =
               (cs->sync.kind & HK_SYNC_GFX_MEM) ? UINT64_MAX
                                                 : cs->sync.att_bloom;
            vdm_att_ring_add(&queue->sync, vdm_base + cs->sync.index, bloom);
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

         /* Prune exact historical priors already retired on the queue timeline
          * before computing ta_split so stale ordinals (e.g. an old GFX_MEM
          * render) do not block fragment-only barriers or direct submission. */
         if (queue->sync.exact_prior) {
            for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
               const uint64_t ret = queue->sync.retired_ord[sq];
               if (ret) {
                  if (cs->sync.prior[sq] <= ret)
                     cs->sync.prior[sq] = 0;
                  if (cs->sync.prior_ta[sq] <= ret)
                     cs->sync.prior_ta[sq] = 0;
               }
            }
         }

         /* A render's vertex half may wait for less than its fragment half
          * only when the kernel orders the halves separately and nothing
          * above raised the whole render's fields afterwards (the
          * first-command chain only touches barrier[]; tile hazards raise
          * both halves): recompute, and keep the vertex half at least as
          * strict as before. */
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
      bool tail_raised = false;
      for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
            if (ss.carry_absolute[k][sq] != queue->sync.tail[k][sq]) {
               queue->sync.tail[k][sq] = ss.carry_absolute[k][sq];
               tail_raised = true;
            }
            if (ss.last[k][sq] > 0)
               queue->sync.last_absolute[k][sq] =
                  queue->sync.submitted[sq] + ss.last[k][sq];
         }
      }
      for (unsigned hs = 0; hs < HK_HEAP_SLOTS; ++hs) {
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
            if (ss.last_heap_slot[hs][sq] > 0)
               queue->sync.last_heap_slot[hs][sq] =
                  queue->sync.submitted[sq] + ss.last_heap_slot[hs][sq];
         }
      }
      if (tail_raised_out)
         *tail_raised_out = tail_raised;
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
         queue->sync.submitted[sq] += ss.count[sq];
      if (ss.count[HK_SYNC_VDM] || ss.count[HK_SYNC_CDM])
         sync_record_retired(&queue->sync, queue->drm.timeline_value);
   }
   for (unsigned hs = 0; hs < HK_HEAP_SLOTS; ++hs) {
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
         if (ss.last_heap_slot[hs][sq] > 0)
            queue->sync.heap_slot_ever[sq] |= BITFIELD_BIT(hs);
      }
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

bool
hk_sync_resolve_submit(struct hk_queue *queue,
                       struct hk_cmd_buffer *const *cmdbufs, unsigned count,
                       bool first_cdm_is_flush, bool chain_first,
                       uint64_t completed, struct hk_sync_resolved *out)
{
   return hk_sync_resolve_submit_ex(queue, cmdbufs, count, first_cdm_is_flush,
                                    chain_first, completed, out, NULL);
}

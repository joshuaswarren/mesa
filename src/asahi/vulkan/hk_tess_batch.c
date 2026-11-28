/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_tess_batch.h"
#include "asahi/libagx/tessellator.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_shader.h"
#include "libagx_shaders.h"

bool
hk_tess_batch_open(struct hk_cmd_buffer *cmd)
{
   struct hk_tess_batch *batch = &cmd->tess_batch;

   for (unsigned i = 0; i < HK_TESS_STAGES; ++i) {
      if (!batch->stages[i])
         batch->stages[i] = hk_cs_create(cmd, true);

      if (!batch->stages[i]) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return false;
      }
   }

   return true;
}

/* No launch followed the last drop (an empty grid): the barrier, still in
 * place behind current, comes back before anything else follows. */
static void
hk_tess_batch_restore(struct hk_tess_batch *batch, enum hk_tess_stage stage)
{
   struct hk_cs *cs = batch->stages[stage];
   if (batch->dropped[stage] &&
       (uint8_t *)cs->current + AGX3_CDM_BARRIER_SIZE == batch->dropped[stage]) {
      cs->current = batch->dropped[stage];
      cs->weak_barrier_end = cs->current;
   }
   batch->dropped[stage] = NULL;
}

struct hk_cs *
hk_tess_batch_stage(struct hk_cmd_buffer *cmd, enum hk_tess_stage stage)
{
   struct hk_tess_batch *batch = &cmd->tess_batch;
   struct hk_cs *cs = batch->stages[stage];

   hk_tess_batch_restore(batch, stage);

   /* Before the tail goes: a chunk link must still follow a barrier. */
   hk_ensure_cs_has_space(cmd, cs, 0x2000);
   if (cs->stats.cmds && cs->current == cs->weak_barrier_end) {
      batch->dropped[stage] = cs->current;
      cs->current = (uint8_t *)cs->current - AGX3_CDM_BARRIER_SIZE;
      cs->weak_barrier_end = NULL;
   }

   return cs;
}

void
hk_launch_tess_batched(struct hk_cmd_buffer *cmd, struct hk_shader *vs,
                       struct hk_shader *tcs, struct agx_grid grid_vs,
                       struct agx_grid grid_tcs, struct agx_grid grid_tess,
                       struct hk_tess_info info, uint64_t state)
{
   struct hk_tess_batch *batch = &cmd->tess_batch;
   if (!hk_tess_batch_open(cmd))
      return;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const bool m5 = agx_chip_g17_m5(dev->dev.chip);

   /* As hk_dispatch_with_local_size(), without its flush. */
   if (!agx_is_shader_empty(&vs->b)) {
      struct hk_cs *cs = hk_tess_batch_stage(cmd, HK_TESS_STAGE_VS);
      if (m5) {
         hk_dispatch_m5_native(cmd, cs, vs, grid_vs, agx_workgroup(64, 1, 1));
      } else {
         hk_reserve_scratch(cmd, cs, vs);
         hk_dispatch_g17p_native(cmd, cs, vs, grid_vs, agx_workgroup(64, 1, 1));
      }
   }
   if (!agx_is_shader_empty(&tcs->b)) {
      struct hk_cs *cs = hk_tess_batch_stage(cmd, HK_TESS_STAGE_TCS);
      const struct agx_workgroup wg =
         agx_workgroup(tcs->info.tess.tcs_output_patch_size, 1, 1);
      if (m5) {
         hk_dispatch_m5_native(cmd, cs, tcs, grid_tcs, wg);
      } else {
         hk_reserve_scratch(cmd, cs, tcs);
         hk_dispatch_g17p_native(cmd, cs, tcs, grid_tcs, wg);
      }
   }

   /* Precompiled kernels record into the pre-graphics stream: stand each
    * stage's side stream in for it. Batched draws count nothing, so the
    * prefix sum has no counters to increment. */
   struct hk_cs *pre_gfx = cmd->current_cs.pre_gfx;
   batch->recording = true;

   cmd->current_cs.pre_gfx = hk_tess_batch_stage(cmd, HK_TESS_STAGE_COUNT);
   libagx_tessellate(cmd, grid_tess, AGX_PREGFX, info.mode,
                     POLY_TESS_MODE_COUNT, state);
   cmd->current_cs.pre_gfx = hk_tess_batch_stage(cmd, HK_TESS_STAGE_PREFIX_SUM);
   libagx_prefix_sum_tess(cmd, agx_1d(1024), AGX_PREGFX, state, 0, 0, 0,
                          false);
   cmd->current_cs.pre_gfx = hk_tess_batch_stage(cmd, HK_TESS_STAGE_TESSELLATE);
   libagx_tessellate(cmd, grid_tess, AGX_PREGFX, info.mode,
                     POLY_TESS_MODE_WITH_COUNTS, state);

   cmd->current_cs.pre_gfx = pre_gfx;
   batch->recording = false;
   batch->draws++;
}

void
hk_tess_batch_flush(struct hk_cmd_buffer *cmd)
{
   struct hk_tess_batch *batch = &cmd->tess_batch;
   if (!batch->draws)
      return;

   /* Created by the first batched draw, and kept open until after this. */
   struct hk_cs *cs = cmd->current_cs.pre_gfx;
   batch->draws = 0;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   for (unsigned i = 0; i < HK_TESS_STAGES; ++i) {
      /* Every stage then ends in its last launch's weak barrier, which the
       * flush below strengthens in place. */
      hk_tess_batch_restore(batch, i);
      struct hk_cs *stage = batch->stages[i];
      batch->stages[i] = NULL;

      /* An empty stream has no barrier at its end to take over. */
      if (cs && stage->stats.cmds) {
         /* Copy a one-chunk stage into the current chunk, so its root takes
          * the next batch instead of five new roots on every flush.
          * hk_cs_merge_cdm() copies only into an unlinked stream, but the
          * copy needs no more than room in the chunk being written. */
         const bool copy = !stage->stream_linked;
         if (copy)
            hk_ensure_cs_has_space(
               cmd, cs, (uint8_t *)stage->current - (uint8_t *)stage->start);
         const bool linked = cs->stream_linked;
         cs->stream_linked &= !copy;
         hk_cs_merge_cdm(cs, stage);
         cs->stream_linked |= linked;

         /* The next stage reads what this one wrote. No chunk link may come
          * between the copy and this, which needs no room. */
         assert(cs->current == cs->weak_barrier_end);
         hk_cdm_cache_flush(dev, cs);
      }

      /* A stage the merge linked to instead is part of cs now. */
      const struct agx_ptr root = {.cpu = stage->start, .gpu = stage->addr};
      const bool reuse =
         !stage->stream_linked && (!cs || root.cpu != cs->chunk.cpu);
      hk_cs_destroy(stage);
      if (reuse)
         batch->stages[i] = hk_cs_create_at(cmd, true, root);
   }
}

void
hk_tess_batch_reset(struct hk_cmd_buffer *cmd)
{
   struct hk_tess_batch *batch = &cmd->tess_batch;

   for (unsigned i = 0; i < HK_TESS_STAGES; ++i) {
      if (batch->stages[i])
         hk_cs_destroy(batch->stages[i]);
   }

   *batch = (struct hk_tess_batch){0};
}

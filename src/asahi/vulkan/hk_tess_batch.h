/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include "asahi/lib/agx_chip.h"
#include "libagx_dgc.h"

struct hk_cmd_buffer;
struct hk_cs;
struct hk_shader;
struct hk_tess_info;

/*
 * Software tessellation runs five compute stages before the render, each
 * reading what the previous one wrote (hk_launch_tess()). Recorded draw by
 * draw, every stage needs a cache flush: a pass of a few hundred terrain
 * patches becomes thousands of serialized tiny dispatches. The draws of one
 * render do not read each other's tessellation state, so their stages can
 * run side by side: each stage is recorded into its own side stream, and
 * hk_tess_batch_flush() splices the streams into the pre-graphics stream
 * with one flush after each stage.
 *
 * Within a side stream, consecutive launches carry no barrier at all, as
 * consecutive application dispatches on G15 (hk_dispatch_g17p_native()).
 * Even the weak ordering barrier that ends every native launch waits for the
 * launch before it: with it, a pass of 1024 draws (vulkan-tess-batch) takes
 * 27 ms on the M4 Pro whether or not the stages also flush; without it, 1 ms.
 * This needs two things. The launches of one stage are independent: each
 * reads only what the previous stage of its own draw wrote, and tessellator
 * heap allocations are atomic. And the cache flush after a stage waits for
 * all of its launches, not just the last one, before the next stage starts,
 * as G15's barrier-less application dispatches assume
 * (agx_chip_elide_app_dispatch_barrier()); vulkan-tess-batch-order is the
 * hardware test for it. The G17P stream validator accepts direct launches
 * back to back and still wants a barrier before anything else.
 *
 * Any other work for the pre-graphics stream flushes the batch first
 * (hk_cmd_buffer_get_cs_general()), as does the end of the render, so that
 * work keeps its place after the tessellation of the draws before it.
 */
enum hk_tess_stage {
   HK_TESS_STAGE_VS,
   HK_TESS_STAGE_TCS,
   HK_TESS_STAGE_COUNT,
   HK_TESS_STAGE_PREFIX_SUM,
   HK_TESS_STAGE_TESSELLATE,
   HK_TESS_STAGES,
};

struct hk_tess_batch {
   struct hk_cs *stages[HK_TESS_STAGES];

   /* Draws recorded into the side streams since the last flush. */
   unsigned draws;

   /* Where the weak barrier hk_tess_batch_stage() last dropped from each
    * stage ended, so it can come back if no launch takes its place. */
   void *dropped[HK_TESS_STAGES];

   /* The pre-graphics stream is being fetched for a batched draw, which must
    * not flush the batch. */
   bool recording;
};

/*
 * Whether a draw's tessellation may join the batch. An indirect draw sizes
 * its stages on the GPU with a setup kernel of its own, and a geometry shader
 * consumes the tessellation right away. Pipeline statistics, clipper and
 * primitives-generated counters are incremented without atomics by every
 * draw's stages, so concurrent stages would lose counts.
 *
 * shortcut: G17P only, measured on the M4 Pro (T6040); the A18 Pro (T8140)
 * shares the chip id. G15/G16 native graphics take the same dispatch path
 * and could join once tested.
 */
static inline bool
hk_tess_batchable(enum agx_chip chip, struct agx_draw draw, bool geometry,
                  bool counters)
{
   return agx_chip_native_hal200(chip) && !agx_is_indirect(draw.b) &&
          !geometry && !counters;
}

/* Create the side streams that are missing. False, with the command buffer
 * error set, if they cannot be allocated. */
bool hk_tess_batch_open(struct hk_cmd_buffer *cmd);

/* The side stream of a stage, ready for the next draw's launch: the weak
 * ordering barrier of the launch before it is dropped. */
struct hk_cs *hk_tess_batch_stage(struct hk_cmd_buffer *cmd,
                                  enum hk_tess_stage stage);

/* Record a draw's tessellation stages into the side streams (from
 * hk_launch_tess()). The flushes between the stages come when the batch is
 * spliced into the pre-graphics stream. */
void hk_launch_tess_batched(struct hk_cmd_buffer *cmd, struct hk_shader *vs,
                            struct hk_shader *tcs, struct agx_grid grid_vs,
                            struct agx_grid grid_tcs, struct agx_grid grid_tess,
                            struct hk_tess_info info, uint64_t state);

/* Splice the batched stages into the pre-graphics stream. The side streams
 * it copies keep their roots for the next batch. */
void hk_tess_batch_flush(struct hk_cmd_buffer *cmd);

/* Drop the side streams of a command buffer being reset or freed. */
void hk_tess_batch_reset(struct hk_cmd_buffer *cmd);

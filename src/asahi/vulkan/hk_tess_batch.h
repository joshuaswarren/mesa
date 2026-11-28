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
 * Batch the five pre-graphics software tessellation compute stages
 * (hk_launch_tess()) across independent draws in a render pass: each stage
 * records into its own side stream, and hk_tess_batch_flush() splices the
 * stages into the pre-graphics stream with a single cache flush per stage.
 *
 * Consecutive launches within a side stream omit the trailing weak ordering
 * barrier so independent draws overlap (tessellator heap allocations are
 * atomic, and the stage-end cache flush waits for all launches before the next
 * stage starts; G17P's stream validator permits back-to-back direct launches
 * but still requires a barrier before any other work). Any other pre-graphics
 * work (hk_cmd_buffer_get_cs_general()) or render end flushes the batch first.
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
 * Direct tessellation draws without geometry shaders (which consume
 * tessellation immediately) or non-atomic pipeline/clipper/primitives-generated
 * counters may join the batch; indirect draws use a GPU setup kernel.
 * Admitted: HAL200 and M5 HAL300 (AGX3C); closed: G15 native and AGX3D
 * (untested).
 */
static inline bool
hk_tess_batchable(enum agx_chip chip, struct agx_draw draw, bool geometry,
                  bool counters)
{
   return (agx_chip_native_hal200(chip) || agx_chip_g17_m5(chip)) &&
          !agx_is_indirect(draw.b) && !geometry && !counters;
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

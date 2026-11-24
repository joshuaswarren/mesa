/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_cmd_buffer.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned flushes, reservations;

/* Exercise the production drain hook with command emission replaced by a
 * recorder. No device or BO allocation is needed to close an existing stream.
 */
void
hk_ensure_cs_has_space(struct hk_cmd_buffer *cmd, struct hk_cs *cs, size_t size)
{
   assert(cmd == cs->cmd);
   assert(size == AGX3_CDM_BARRIER_SIZE);
   reservations++;
}

void
hk_cdm_cache_flush(struct hk_device *dev, struct hk_cs *cs)
{
   (void)dev;
   assert(cs->query_reset_pending);
   assert(reservations == flushes + 1);
   flushes++;
}

/* All fixtures reuse a nonempty, already allocated general stream. These
 * paths must not allocate, change dependencies, discard a stream or emit
 * unrelated work while draining query resets. */
void
hk_flush_deferred_clears_why(struct hk_cmd_buffer *cmd, const char *why)
{
   (void)cmd;
   (void)why;
   abort();
}

void
hk_sync_check_cs(struct hk_cmd_buffer *cmd)
{
   (void)cmd;
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

bool
hk_cs_init_graphics(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   (void)cs;
   abort();
}

void
hk_drm_attachment_set_init(struct hk_drm_attachment_set *set)
{
   (void)set;
   abort();
}

void
hk_sync_stream_created(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       struct hk_cs **slot, bool compute)
{
   (void)cmd;
   (void)cs;
   (void)slot;
   (void)compute;
   abort();
}

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

void
hk_sync_forget_stream(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   (void)cs;
   abort();
}

/* These devices never map memory at a second address, so the real
 * function adds no flush either.
 */
void
hk_cdm_end_stream_flush(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   (void)cs;
}

/* No tessellation is batched here. */
void
hk_tess_batch_flush(struct hk_cmd_buffer *cmd)
{
   (void)cmd;
   abort();
}

/* Query resets are not transfers. */
void
hk_sync_note_xfer_write(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   (void)cs;
   abort();
}

int
main(void)
{
   struct hk_cmd_buffer *cmd = calloc(1, sizeof(*cmd));
   assert(cmd);
   struct hk_cs cs = {.cmd = cmd, .type = HK_CS_CDM};

   hk_flush_query_resets(cmd, &cs);
   assert(flushes == 0 && reservations == 0);

   /* The real getter defers drains during consecutive reset recording. */
   cs.query_reset_pending = true;
   cs.stats.query_reset_dispatches = 3;
   cs.stats.cmds = 3;
   cs.sync.kind = HK_SYNC_ALL;
   cmd->current_cs.cs = &cs;
   cmd->state.query_reset_recording = true;
   assert(hk_cmd_buffer_get_cs(cmd, true) == &cs);
   assert(cs.query_reset_pending && flushes == 0);

   /* The next consumer goes through that same getter and drains once. */
   cmd->state.query_reset_recording = false;
   assert(hk_cmd_buffer_get_cs(cmd, true) == &cs);
   assert(!cs.query_reset_pending);
   assert(flushes == 1 && cs.stats.query_reset_flushes == 1);
   assert(hk_cmd_buffer_get_cs(cmd, true) == &cs);
   assert(flushes == 1);

   /* Closing a stream also drains, even inside reset recording. */
   cmd->state.query_reset_recording = true;
   cs.query_reset_pending = true;
   hk_cmd_buffer_end_compute(cmd);
   assert(!cs.query_reset_pending && cmd->current_cs.cs == NULL);
   assert(flushes == 2 && cs.stats.query_reset_flushes == 2);
   assert(cs.stats.query_reset_dispatches == 3);
   hk_cmd_buffer_end_compute(cmd);
   assert(flushes == 2);
   free(cmd);
   puts(
      "query reset drains: empty, reset recording, consumer and stream end passed");
   return 0;
}

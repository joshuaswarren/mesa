/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct hk_cmd_buffer;
struct hk_cs;
struct hk_device;

/* Submit packing (hk_queue.c, hk_pack_runs()): a run of consecutive command
 * buffers that each record one plain compute stream is submitted as one
 * command. These are the parts that decide what a packet may hold and how
 * its dependencies are recorded; they touch no device or kernel state.
 *
 * A packet covers at most HK_PACK_MAX_DISPATCHES launches and the thread
 * and static work bounds; the measured 165-, 200- and 242-dispatch image and
 * compute graphs fit. Static work bounds a program without loops; a packet
 * holds at most one stream with a looping program, so it runs at most as
 * long as that stream alone plus statically bounded work.
 */
#define HK_PACK_MAX_DISPATCHES 256u
#define HK_PACK_MAX_THREADS (UINT64_C(1) << 28)
#define HK_PACK_MAX_STATIC_WORK (UINT64_C(1) << 35)

/* One command buffer of a submission, as the planner sees it. */
struct hk_pack_member {
   /* Its single compute stream, or NULL: it stays an ordinary command. */
   const struct hk_cs *cs;
   /* Stream bytes before the terminate. */
   size_t bytes;
   /* Synchronization kinds of the stream. */
   uint8_t kinds;
   /* Render (VDM) kinds the stream itself waits for, and those the command
    * buffer's tail makes later command buffers wait for. */
   uint8_t render_waits, render_tail;
};

struct hk_pack_limits {
   /* Compute attachments one command may carry. */
   uint32_t max_attachments;
   /* Bytes a packet needs beyond its members' streams. */
   size_t overhead;
};

struct hk_pack_run {
   unsigned first, count;
   size_t offset, bytes;
};

/* The single compute stream of `cmd` that a packet may copy, or NULL. */
const struct hk_cs *hk_pack_run_member(const struct hk_device *dev,
                                       const struct hk_cmd_buffer *cmd,
                                       const uint8_t *terminate, bool atomics);

/* Fills `member` for command buffer `cmd` whose stream is `cs` (or NULL). */
void hk_pack_describe(struct hk_pack_member *member,
                      const struct hk_cmd_buffer *cmd, const struct hk_cs *cs,
                      size_t terminate_size);

/* Groups members into runs of two or more: runs[] gets them in order, with
 * byte offsets in one buffer of *total bytes. Returns the run count. */
unsigned hk_pack_plan(const struct hk_pack_member *members, unsigned n,
                      const struct hk_pack_limits *limits,
                      struct hk_pack_run *runs, unsigned *packed_members,
                      size_t *total);

/* Records member `item` (the stream of `source`) in the packet's
 * synchronization: what it waits for, what it makes later command buffers
 * wait for, and, unless it is the last member, its tail as a requirement of
 * the members after it. `preceding_kinds` accumulates over the members. */
void hk_pack_fold_member(struct hk_cmd_buffer *packet, struct hk_cs *cs,
                         const struct hk_cmd_buffer *source,
                         const struct hk_cs *item, bool last,
                         uint8_t *preceding_kinds);

/* The packet's stream is its command buffer's latest of every kind. */
void hk_pack_finish_sync(struct hk_cmd_buffer *packet, struct hk_cs *cs);

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "hk_sync.h"

struct hk_cmd_buffer;
struct hk_cs;
struct hk_device;
struct hk_sync_queue;

/* Submit packing (hk_queue.c, hk_pack_runs()): a run of consecutive command
 * buffers that each record one plain compute stream is submitted as one
 * command. These are the parts that decide what a packet may hold and how
 * its dependencies are recorded; they touch no device or kernel state.
 *
 * A packet covers at most HK_PACK_MAX_DISPATCHES launches and the thread
 * and static work bounds; the measured 165-, 200- and 242-dispatch image and
 * compute graphs fit. A program with loops runs for longer than its static
 * work says, so a packet holds at most one stream with a looping program.
 * The static work bound is 2^35 thread-instructions, about 18 ms at the M4's
 * ALU issue rate; memory-bound programs run slower per instruction, so for
 * them the bound limits a packet's length only roughly.
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
   /* Render (VDM) kinds the stream itself waits for, and per stream kind k
    * those the command buffer's tail makes later streams of kind k wait for
    * (req[k][VDM]), both only of kinds that resolve to a render wait before
    * this command buffer (hk_sync_queue_kinds(), hk_sync_cmd_kinds()): other
    * kinds are no wait. */
   uint8_t render_waits;
   uint8_t render_tail[HK_SYNC_NUM_KINDS];
   /* The render subqueue ran work before this command buffer. */
   bool render_ran;
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

/* How a member's end publishes its writes to the members after it, which
 * decides whether a device packs at all. */
enum hk_pack_boundary {
   /* No boundary is defined: the device does not pack. */
   HK_PACK_BOUNDARY_NONE,
   /* The kernel takes each stream's own completion flush, the strong
    * memory-scoped flush (DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION, M5
    * streams): a member ends with a strong flush, a boundary keeps the
    * memory scope only where memory may have a second address
    * (hk_cdm_alias_scope()), and the packet's end gets it. */
   HK_PACK_BOUNDARY_VISIBLE_COMPLETION,
   /* HAL200 streams whose strong flush has no memory-scoped form (G16): the
    * kernel publishes a command's writes when it completes, and a boundary
    * makes the member's last ordering record the strong flush, the record an
    * in-stream compute dependency gets. */
   HK_PACK_BOUNDARY_STRONG_FLUSH,
};

enum hk_pack_boundary hk_pack_boundary(const struct hk_device *dev);

/* Whether a member may end with a strong flush without memory scope, as an
 * application barrier at the end of a command buffer leaves it: at every
 * boundary. The HAL200 boundary upgrades it in place; at the
 * visible-completion boundary the packet's end carries the memory-scoped
 * flush the kernel takes as its completion (hk_pack_runs()).
 * hk_pack_run_member() takes the answer from here, and the packing trace
 * reports it (strong_tails=), so tests know which barrier-ended command
 * buffers must pack. */
bool hk_pack_admits_strong_tail(enum hk_pack_boundary boundary);

/* Whether streams with global atomics may be packed, before the check of the
 * device's other kernel queues: with `serial_kernel` (G16G, whose kernel
 * runs a VM's compute commands one after another on one firmware queue) by
 * default and unless HK_PACK_ATOMICS (`opt`) says no; otherwise only when
 * `opt` says yes and the queue has a single kernel queue (`queues`). `opt`
 * takes debug_parse_bool_option()'s values plus "on" and "off"; any other
 * value keeps the default. */
bool hk_pack_atomics_choice(bool serial_kernel, const char *opt,
                            unsigned queues);

/* The single compute stream of `cmd` that a packet may copy, or NULL. */
const struct hk_cs *hk_pack_run_member(const struct hk_device *dev,
                                       const struct hk_cmd_buffer *cmd,
                                       const uint8_t *terminate, bool atomics);

/* Fills members[0..n) for the command buffers of a submission to the queue
 * whose synchronization state is `sync`, as hk_pack_plan() takes them, and
 * returns how many may be packed. */
unsigned hk_pack_describe_submit(struct hk_pack_member *members,
                                 const struct hk_device *dev,
                                 const struct hk_sync_queue *sync,
                                 struct hk_cmd_buffer *const *cmds, unsigned n,
                                 const uint8_t *terminate, bool atomics);

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

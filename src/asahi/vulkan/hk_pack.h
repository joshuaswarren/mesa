/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"
#include "util/u_dynarray.h"
#include "hk_pack_loop_bound.h"
#include "hk_sync.h"

struct agx_bo;

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
/* Observed-duration packing bounds for repeated multi-CB looping submissions.
 * Firmware compute command timeouts are ~2 s on AGX2/AGX3, so admitting only
 * whole submissions observed at <= HK_PACK_OBS_ADMIT_NS (200 ms) keeps ~10x
 * headroom below timeout even if data-dependent loop trip counts grow between
 * submits. Because any packet holds a subset of the admitted command buffers,
 * the whole-submission admission bound (200 ms) is the hard safety limit,
 * while HK_PACK_MAX_OBS_NS (32 ms) splits admitted runs using the observed
 * per-loop average and HK_PACK_OBS_BACKOFF_NS (150 ms) tightens or resets
 * monitored packed runs if duration climbs. */
#define HK_PACK_MAX_OBS_NS (UINT64_C(32) * 1000000)
#define HK_PACK_OBS_ADMIT_NS (UINT64_C(200) * 1000000)
#define HK_PACK_OBS_BACKOFF_NS (UINT64_C(150) * 1000000)
#define HK_PACK_OBS_SLOTS 8u
#define HK_PACK_CDM_TERMINATE_MAX 8u

struct hk_pack_obs_entry {
   uint64_t hash;
   uint64_t pending_pt;
   uint64_t submit_ns;
   uint64_t obs_per_loop_ns;
   uint64_t lru_tick;
   uint32_t cb_count;
   uint32_t loop_count;
   bool in_flight_packed;
};

/* Retires completed observed-duration entries using a point-tagged completion
 * timestamp: `sig_ns` is used only when `pending_pt <= sig_point` and
 * `submit_ns < sig_ns <= now_ns`; otherwise `now_ns` is used conservatively so
 * an untimestamped or stale `signalled_ns` can never under-report duration. */
void hk_pack_obs_retire_entries(struct hk_pack_obs_entry *entries,
                                unsigned slot_count, uint32_t *obs_pending,
                                uint64_t *obs_epoch, uint64_t signalled,
                                uint64_t sig_point, uint64_t sig_ns,
                                uint64_t now_ns);

/* One command buffer of a submission, as the planner sees it. */
struct hk_pack_member {
   /* Its single compute stream, or NULL: it stays an ordinary command. */
   const struct hk_cs *cs;
   /* Stream bytes before the terminate. */
   size_t bytes;
   /* Observed conservative upper-bound duration (ns) for a looping member
    * from completed queue history; 0 when unobserved (at most one per run). */
   uint64_t obs_ns;
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
   /* Pure vkCmdResetQueryPool compute stream (only packs with other
    * query-reset streams so regular compute packets keep NO_QUERY_ACCESS). */
   bool query_reset;
};

struct hk_pack_limits {
   /* Compute attachments one command may carry. */
   uint32_t max_attachments;
   /* Bytes a packet needs beyond its members' streams. */
   size_t overhead;
   /* Per-packet static thread-instruction budget; 0 uses
    * HK_PACK_MAX_STATIC_WORK. */
   uint64_t max_static_work;
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

/* Packs the CDM stream-terminate record for `chip` into `out` and returns its
 * byte length (8 for AGX2, 4 for AGX3), or 0 if unsupported. */
size_t hk_pack_cdm_terminate(enum agx_chip chip,
                             uint8_t out[HK_PACK_CDM_TERMINATE_MAX]);

/* Per-packet static work budget scaled by GPU core count (1x at <=20 cores,
 * linear up to 4x at >=80 cores) so wall-clock packet duration stays uniform
 * across small and wide GPUs. */
uint64_t hk_pack_max_static_work_for_cores(unsigned cores);
uint64_t hk_pack_max_static_work(const struct hk_device *dev);

bool hk_pack_account_dispatch_work(uint64_t threads, uint64_t instrs,
                                   bool has_loops, uint64_t loop_bound,
                                   uint64_t *work_out);

bool hk_pack_account_dispatch_work_limit(uint64_t threads, uint64_t instrs,
                                         bool has_loops, uint64_t loop_bound,
                                         uint64_t max_work, uint64_t *work_out);

bool hk_pack_account_dispatch_work_eff(uint64_t threads, uint64_t instrs,
                                       bool has_loops, uint64_t eff_instrs,
                                       uint64_t *work_out);

bool hk_pack_account_dispatch_work_eff_limit(uint64_t threads, uint64_t instrs,
                                             bool has_loops,
                                             uint64_t eff_instrs,
                                             uint64_t max_work,
                                             uint64_t *work_out);

bool hk_pack_fusion_fits_run(uint64_t run_work, unsigned run_loops,
                             uint64_t replaced_work, unsigned replaced_loops,
                             uint64_t fused_work, bool fused_loops,
                             uint64_t *new_work_out, unsigned *new_loops_out);

bool hk_pack_fusion_fits_run_limit(uint64_t run_work, unsigned run_loops,
                                   uint64_t replaced_work,
                                   unsigned replaced_loops, uint64_t fused_work,
                                   bool fused_loops, uint64_t max_work,
                                   uint64_t *new_work_out,
                                   unsigned *new_loops_out);

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

/* Whether `cs` is a clean single-dispatch compute stream eligible to
 * participate as producer or consumer in submit-time compute pair fusion. */
bool hk_compute_pair_stream_eligible(const struct hk_cs *cs);

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

/* Initializes a reusable packet's command buffer and compute control stream,
 * resetting only the compute-relevant state and preserving dynarray backing
 * storage across submissions. */
void hk_pack_init_packet(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                         struct hk_device *dev, uint8_t *start, uint64_t addr,
                         size_t bytes, bool visible_completion);

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

/* Adds `bo` once to `bos`, the buffers a packed submission keeps alive until
 * it completes (hk_cmd_pending_begin()). A NULL `bo` adds nothing. Returns
 * false if the list cannot grow. */
bool hk_pack_add_bo(struct util_dynarray *bos, struct agx_bo *bo);

#define HK_PACK_FOOTPRINT_MAX_READS 8u
#define HK_PACK_FOOTPRINT_MAX_WRITES 8u
#define HK_PACK_HAZARD_MAX_INTERVALS 64u
#define HK_PACK_LIVENESS_MAX_INTERVALS 64u

struct hk_pack_interval {
   uint64_t start;
   uint64_t end;
   uint32_t handle;
};

struct hk_pack_footprint {
   struct hk_pack_interval reads[HK_PACK_FOOTPRINT_MAX_READS];
   struct hk_pack_interval writes[HK_PACK_FOOTPRINT_MAX_WRITES];
   uint8_t num_reads;
   uint8_t num_writes;
   bool known;
};

enum hk_pack_boundary_action {
   HK_PACK_BOUNDARY_ELIDE_BARRIER,
   HK_PACK_BOUNDARY_KEEP_WEAK,
   HK_PACK_BOUNDARY_STRONG,
};

struct hk_pack_hazard_state {
   struct hk_pack_interval dirty_writes[HK_PACK_HAZARD_MAX_INTERVALS];
   struct hk_pack_interval dirty_reads[HK_PACK_HAZARD_MAX_INTERVALS];
   uint8_t num_dirty_writes;
   uint8_t num_dirty_reads;
   bool write_unknown;
   bool read_unknown;
};

void hk_pack_hazard_reset_all(struct hk_pack_hazard_state *st);
void hk_pack_hazard_record(struct hk_pack_hazard_state *st,
                           const struct hk_pack_footprint *fp,
                           bool strong_flushed);
enum hk_pack_boundary_action
hk_pack_hazard_decide(struct hk_pack_hazard_state *st,
                      const struct hk_pack_footprint *next_fp,
                      bool can_elide_weak);

/* Backward dead-interval tracker within a single packed run.
 * Every interval in `dead[]` is guaranteed to be completely overwritten by
 * later members of the same run before any read or end-of-run; coalesced
 * disjoint intervals per handle. Fails closed (empties `dead[]`) on any
 * unknown footprint or invalid read. */
struct hk_pack_liveness_state {
   struct hk_pack_interval dead[HK_PACK_LIVENESS_MAX_INTERVALS];
   uint8_t num_dead;
};

void hk_pack_liveness_reset(struct hk_pack_liveness_state *st);
bool hk_pack_liveness_covers(const struct hk_pack_liveness_state *st,
                             const struct hk_pack_interval *iv);
void hk_pack_liveness_add_overwrite(struct hk_pack_liveness_state *st,
                                    const struct hk_pack_interval *iv);
void hk_pack_liveness_kill_read(struct hk_pack_liveness_state *st,
                                const struct hk_pack_interval *iv);
void hk_pack_liveness_step_backward(struct hk_pack_liveness_state *st,
                                    const struct hk_pack_footprint *fp);

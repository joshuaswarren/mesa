/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_pack.h"
#include "hk_private.h"
#include "hk_sync.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "vk_queue.h"

struct hk_device;
struct hk_packed_submit;
struct hk_render_packet;

/* Queues advertised per family when the kernel reports independent queues
 * (DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES), and the most Vulkan queues of one
 * device that keep a kernel queue of their own (hk_queue_independent_budget()).
 */
#define HK_ADVERTISED_QUEUES 8u

/* Kernel queues one hk_queue may spread its independent work over. */
#define HK_QUEUE_SIBLINGS_MAX 4

/* Render stages with ray-query scratch, in drm_asahi_queue_create
 * render_scs_layout order. */
#define HK_RENDER_RQ_VERTEX   (1u << 0)
#define HK_RENDER_RQ_FRAGMENT (1u << 1)

struct hk_queue {
   struct vk_queue vk;

   struct {
      /* Asahi kernel queue ID, UINT32_MAX until the first GPU submission. */
      uint32_t id;

      /* The render stages (HK_RENDER_RQ_*) whose ray-query scratch the
       * kernel queue reserves, which sizes its render free list. Only a
       * queue that runs such a render gets one, and only for the stages it
       * has run ray queries in: each stage's reservation scales with the
       * GPU core count, and the kernel places every queue's free list as
       * one contiguous block in one bounded window. */
      uint8_t render_rq;

      /* Timeline syncobj backing the queue */
      uint32_t syncobj;

      /* Ring-idx used with virtgpu, equal to priority + 1 */
      uint32_t virt_ring_idx;

      /* Current maximum timeline value for the queue's syncobj. If the
       * syncobj's value equals timeline_value, then all work is complete.
       */
      uint64_t timeline_value;

      /* The highest timeline value a host wait or syncobj query has seen
       * signalled, updated monotonically via hk_queue_note_signalled(). Every
       * value at or below it is complete. */
      uint64_t signalled_value;
      /* Monotonic timestamp (ns) and highest timeline point covered by it,
       * advanced monotonically by hk_queue_note_signalled() before publishing
       * signalled_value so observed-duration retirement never uses a stale
       * timestamp from an earlier point. */
      uint64_t signalled_ns;
      uint64_t signalled_ns_point;

      /* Latest timeline point that received the waits of a submission
       * without control streams (queue_submit_empty()), 0 once it is known
       * complete. Later kernel submissions of this queue wait for it. */
      uint64_t empty_wait_point;

      /* Latest timeline point of a submission carrying commands on each
       * subqueue (VDM / CDM), cleared to 0 once observed complete. Used so a
       * leading TOP_OF_PIPE timestamp on one subqueue takes the ordered
       * carrier whenever the other subqueue still has incomplete work. */
      uint64_t last_sq_point[HK_SYNC_NUM_SQ];

      /* Latest timeline point of a submission carrying a GPU query reset
       * (vkCmdResetQueryPool), cleared to 0 once observed complete. A submit
       * that ends with compute after a reset clears the queue tail, yet its
       * reset may still be running when the next submit's leading TOP_OF_PIPE
       * start stamp would latch. */
      uint64_t last_query_reset_point;

      /* Latest timeline point covered by a queue barrier tail
       * (queue->sync.tail), cleared to 0 once observed complete. Because
       * queue->sync.tail holds monotonic ordinals that never reset to 0,
       * tracking its timeline point lets leading TOP_OF_PIPE timestamps
       * re-engage the fast path once the barrier's source work has finished. */
      uint64_t last_tail_point;
   } drm;

   /* M5 submissions whose command memory is still pinned, oldest first
    * (hk_cmd_pool.c), and the number of reaps since the oldest was last
    * seen complete. Guarded by the device's cmd_pending_lock; cmd_link is
    * in the device's cmd_pending_queues while cmd_pending is not empty. */
   struct list_head cmd_pending;
   struct list_head cmd_link;
   unsigned cmd_stalled;

   /* Device-lifetime command storage for the G17P SSTT maintenance encoder.
    * Sparse image remaps submit this record before their signal operations. */
   struct agx_bo *g17p_sstt_flush_bo;

   /* Lazy command storage for an ordered end-stamp carrier when a leading
    * TOP_OF_PIPE timestamp cannot safely ride on the first command's start. */
   struct agx_bo *ts_carrier_bo;

   /* Lazy command storage for the memory-scoped cache flush that ends a
    * submission whose command buffers deferred host availability
    * (hk_sync_defer_host_flush()). */
   struct agx_bo *host_flush_bo;

   /* Kinds of streams this queue has ever run, see hk_sync.h. */
   struct hk_sync_queue sync;
   /* Counts restarts of sync's ordinals (a new kernel queue or sibling use):
    * a binary semaphore's record of them is only valid in its generation
    * (hk_gpu_waits.h). */
   uint64_t gpu_wait_generation;

   /* Scratch for queue_submit: the resolved sync fields of each stream
    * occurrence of the submission being built (hk_sync_resolve_submit()). */
   struct util_dynarray sync_resolved;

   /* Reusable scratch for submit packing (hk_pack_runs()). */
   struct {
      struct util_dynarray members;
      struct util_dynarray runs;
      struct util_dynarray commands;
      /* Per member: the fused compute-pair plan and its elision state. */
      struct util_dynarray fused;
      struct util_dynarray plans;
      struct util_dynarray planned_suppressed;
      struct util_dynarray suppressed_arr;
      struct util_dynarray run_sup;
      /* Buffers the packets read until they complete: their streams, the
       * fused shaders' code, uniform loaders and root tables. */
      struct util_dynarray extra_bos;
      struct hk_packed_submit *packets;
      unsigned packets_cap;
      /* Observed completion durations for repeated multi-CB looping
       * submissions, used to budget packets safely below firmware timeouts. */
      struct hk_pack_obs_entry obs[HK_PACK_OBS_SLOTS];
      uint32_t obs_pending;
      uint64_t obs_epoch;
      uint64_t obs_tick;
   } pack;

   /* Reusable scratch and packets for submit-time render packing
    * (hk_render_pack_runs()). */
   struct {
      struct util_dynarray commands;
      struct util_dynarray runs;
      struct util_dynarray targets;
      struct hk_render_packet *packets;
      unsigned packets_cap;
   } rpack;

   /* Compute kick (hk_compute_kick.c). watched is set at device creation
    * for a queue whose priority class has a kick queue. The rest is
    * protected by the kicker's lock. */
   struct {
      bool watched;

      /* Timeline point of the last submission with compute work, 0 once
       * the kicker has seen it complete. */
      uint64_t pending;

      /* Kicker thread only: the queue timeline value last seen, and when
       * the next kick is due if it stays there (0: not yet seen). */
      uint64_t seen;
      int64_t due_ns;
   } kick;
   /* G17P sibling kernel queues (N = HK_G17P_SIBLING_QUEUES on G17P,
    * otherwise 1, which is off; see hk_queue.c). Submissions that depend on no earlier submission and hold only plain
    * compute streams take turns over N kernel queues, the primary (drm.id)
    * and N - 1 siblings, so consecutive ones may run side by side. Every
    * kernel queue signals the same timeline (drm.syncobj), whose points
    * complete in order, so fences, semaphores and the queue's completed
    * point keep covering all earlier work. Command ordinals are per kernel
    * queue; sync.submitted keeps counting every command of the queue.
    */
   struct {
      /* Kernel queues in the rotation, the primary included; 1 is off. */
      uint32_t count;
      /* Next kernel queue in the rotation. */
      uint32_t next;
      /* Sibling kernel queue IDs ([0] unused), UINT32_MAX until first use. */
      uint32_t id[HK_QUEUE_SIBLINGS_MAX];
      /* Commands submitted to each kernel queue, per subqueue. */
      uint64_t submitted[HK_QUEUE_SIBLINGS_MAX][HK_SYNC_NUM_SQ];
      /* A sibling has run work since the primary kernel queue started, so
       * its ordinals no longer equal sync.submitted. */
      bool used;
      /* Timeline point of the latest submission to a sibling. */
      uint64_t last_point;
      /* Timeline point of the latest packed submission containing global
       * atomics; sibling routing stays off until it completes. */
      uint64_t packed_atomic_point;
   } sib;
};

static inline struct hk_device *
hk_queue_device(struct hk_queue *queue)
{
   return (struct hk_device *)queue->vk.base.device;
}

static inline void
hk_queue_note_signalled_at(struct hk_queue *queue, uint64_t point,
                           uint64_t now_ns)
{
   uint64_t seen = p_atomic_read(&queue->drm.signalled_value);
   if (seen >= point)
      return;

   uint64_t seen_ns = p_atomic_read(&queue->drm.signalled_ns);
   while (seen_ns < now_ns) {
      const uint64_t prev =
         p_atomic_cmpxchg(&queue->drm.signalled_ns, seen_ns, now_ns);
      if (prev == seen_ns)
         break;
      seen_ns = prev;
   }

   uint64_t seen_pt = p_atomic_read(&queue->drm.signalled_ns_point);
   while (seen_pt < point) {
      const uint64_t prev =
         p_atomic_cmpxchg(&queue->drm.signalled_ns_point, seen_pt, point);
      if (prev == seen_pt)
         break;
      seen_pt = prev;
   }

   while (seen < point) {
      const uint64_t prev =
         p_atomic_cmpxchg(&queue->drm.signalled_value, seen, point);
      if (prev == seen)
         break;
      seen = prev;
   }
}

static inline void
hk_queue_note_signalled(struct hk_queue *queue, uint64_t point)
{
   if (p_atomic_read(&queue->drm.signalled_value) >= point)
      return;
   hk_queue_note_signalled_at(queue, point, (uint64_t)os_time_get_nano());
}


/* Whether a queue publishes API signals from its cumulative queue timeline
 * because independent engines or sibling kernel queues may finish out of
 * order. */
static inline bool
hk_queue_uses_cumulative_signals(const struct agx_device *dev)
{
   return dev->chip == AGX_CHIP_G17P ||
          (dev->params.features & DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES) != 0;
}

/* Whether a non-chunked submission with output signals can publish them
 * atomically inside DRM_IOCTL_ASAHI_SUBMIT via cumulative signals. */
static inline bool
hk_queue_kernel_cumulative_supported(const struct agx_device *dev,
                                     bool cumulative_signals,
                                     bool chunk_timelines,
                                     unsigned signal_count)
{
   return cumulative_signals && !chunk_timelines && signal_count > 0 &&
          (dev->params.features & DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS) != 0;
}

/* Whether a VkQueue should open sibling compute kernel queues to overlap
 * independent compute kicks within the same context. Not where the kernel
 * already spreads each queue's independent compute over several firmware
 * queues (DRM_ASAHI_FEATURE_COMPUTE_MULTI_CHANNEL): both would fan out. */
static inline bool
hk_queue_sibling_compute_supported(const struct agx_device *dev,
                                   bool exact_prior)
{
   const uint64_t required = DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS |
                             DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS;
   return !dev->is_virtio && !(dev->debug & AGX_DBG_1QUEUE) && exact_prior &&
          hk_queue_uses_cumulative_signals(dev) &&
          (dev->params.features & required) == required &&
          !(dev->params.features & DRM_ASAHI_FEATURE_COMPUTE_MULTI_CHANNEL);
}

VkResult hk_queue_init(struct hk_device *dev,
                       const VkDeviceQueueCreateInfo *pCreateInfo,
                       uint32_t index_in_family);

VkResult hk_queue_init_kick(struct hk_device *dev,
                           enum drm_asahi_priority priority,
                           struct hk_queue **out);

void hk_queue_finish(struct hk_device *dev, struct hk_queue *queue);

void hk_device_quiesce_lost_submissions(struct hk_device *dev);
void hk_device_wait_submitted(struct hk_device *dev, uint64_t timeout_ns);

/* The first command buffer of the submission whose recording failed, or NULL.
 * Such a command buffer is not executable (vkEndCommandBuffer returned its
 * error) and its streams may be incomplete. */
struct vk_command_buffer *
hk_submit_first_errored_cmd_buffer(const struct vk_queue_submit *submit);

VkResult hk_queue_submit_bind_sparse_memory(struct hk_device *device,
                                            struct vk_queue_submit *submission);

uint32_t hk_queue_independent_budget(struct hk_device *dev);

bool hk_pack_atomics_allowed(struct hk_device *dev, struct hk_queue *queue);

void hk_queue_share(struct hk_device *dev, struct hk_queue *queue);

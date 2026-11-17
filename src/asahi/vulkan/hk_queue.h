/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"
#include "hk_sync.h"
#include "vk_queue.h"

struct hk_device;

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

      /* The highest timeline value a host wait has seen signalled, for
       * occlusion query reads (hk_oq_submission_done()). Every value at or
       * below it is complete. */
      uint64_t signalled_value;
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

   /* Kinds of streams this queue has ever run, see hk_sync.h. */
   struct hk_sync_queue sync;

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
   /* G17P sibling kernel queues (ASAHI_G17P_SIBLING_QUEUES=N, default off).
    * Submissions that depend on no earlier submission and hold only plain
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
      /* Earlier submissions in flight before a sibling takes one. */
      uint32_t min_in_flight;
      /* Fewest application dispatch threads a stream needs to go to one. */
      uint64_t min_threads;
      /* Commands submitted to each kernel queue, per subqueue. */
      uint64_t submitted[HK_QUEUE_SIBLINGS_MAX][HK_SYNC_NUM_SQ];
      /* A sibling has run work since the primary kernel queue started, so
       * its ordinals no longer equal sync.submitted. */
      bool used;
      /* Timeline point of the latest submission to a sibling. */
      uint64_t last_point;
   } sib;
};

static inline struct hk_device *
hk_queue_device(struct hk_queue *queue)
{
   return (struct hk_device *)queue->vk.base.device;
}

VkResult hk_queue_init(struct hk_device *dev,
                       const VkDeviceQueueCreateInfo *pCreateInfo,
                       uint32_t index_in_family);

VkResult hk_queue_init_kick(struct hk_device *dev,
                           enum drm_asahi_priority priority,
                           struct hk_queue **out);

void hk_queue_finish(struct hk_device *dev, struct hk_queue *queue);

void hk_device_quiesce_lost_submissions(struct hk_device *dev);

VkResult hk_queue_submit_bind_sparse_memory(struct hk_device *device,
                                            struct vk_queue_submit *submission);

void hk_queue_share(struct hk_device *dev, struct hk_queue *queue);

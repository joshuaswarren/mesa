/*
 * Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "util/simple_mtx.h"
#include "vk_sync.h"

#include "hk_sync.h"

struct hk_device;
struct hk_queue;

/* Binary semaphore waits on work of the same queue, ordered on the GPU.
 *
 * A binary semaphore wait is an input sync of the kernel submission, which
 * the kernel holds on the host until the fence signals: the GPU finishes the
 * producer, the interrupt signals the fence, a scheduler worker wakes and
 * only then passes the consumer to the firmware. Layers and workloads that
 * chain submissions on one queue through binary semaphores (for instance an
 * overlay drawing after the frame and waiting on the frame's semaphore) pay
 * that host round trip on every wait.
 *
 * When the semaphore was last signalled by a submission of this same queue,
 * the wait means "after all commands this queue had submitted then". With
 * exact prior ordinals (DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS) that is a
 * prefix the kernel orders on the GPU (in firmware, for producers already
 * published). Such a wait is left out of the kernel's input syncs and becomes
 * a prior floor of every command of the waiting submission
 * (hk_sync_queue::wait_floor).
 *
 * The device remembers, per binary vk_sync, the queue and ordinals of the
 * submission that signalled it last. Every other way a payload can change or
 * leave the device forgets it: the sync type's init, signal, reset, move
 * (which carries it to the destination for non-shared payloads), imports, and
 * exports; the device's copy_sync_payloads; and empty submissions, whose
 * signals also cover their waits. Every binary wait, on any queue, consumes
 * it. Destroying a semaphore or fence forgets its payloads
 * (hk_DestroySemaphore, hk_DestroyFence); the sync type's finish stays the
 * original one, because vk_sync_type_is_drm_syncobj() compares it. Shared
 * payloads (opaque FD export or import: VK_SYNC_IS_SHARED) can change
 * elsewhere and never take this path.
 */

#define HK_GPU_WAIT_TAGS 32

struct hk_gpu_wait_tag {
   const struct vk_sync *sync;
   const struct hk_queue *queue;
   uint64_t generation;
   uint64_t submitted[HK_SYNC_NUM_SQ];
};

struct hk_gpu_waits {
   bool enabled;
   simple_mtx_t lock;
   unsigned next;
   struct hk_gpu_wait_tag tags[HK_GPU_WAIT_TAGS];
};

/* Wrap a drm_syncobj sync type's payload-changing operations. The finish
 * operation stays vk_drm_syncobj_finish, so vk_sync_type_is_drm_syncobj()
 * still holds. base keeps the original operations. */
void hk_gpu_waits_wrap_type(struct vk_sync_type *type,
                            const struct vk_sync_type *base);

void hk_gpu_waits_init(struct hk_device *dev);
void hk_gpu_waits_finish(struct hk_device *dev);

/* The device's copy_sync_payloads: forgets the signals, consumes waits, then
 * copies. */
VkResult hk_gpu_waits_copy_sync_payloads(struct vk_device *vk,
                                         uint32_t wait_count,
                                         const struct vk_sync_wait *waits,
                                         uint32_t signal_count,
                                         const struct vk_sync_signal *signals);

/* Before publishing a submission of queue: its binary signals now stand for
 * queue->sync.submitted. Without eligible, forgets them instead. */
void hk_gpu_waits_note_signals(struct hk_device *dev,
                               const struct hk_queue *queue, bool eligible,
                               uint32_t signal_count,
                               const struct vk_sync_signal *signals);

void hk_gpu_waits_forget_signals(struct hk_device *dev, uint32_t signal_count,
                                 const struct vk_sync_signal *signals);

/* Binary waits consume their records, whatever queue signalled them. */
void hk_gpu_waits_forget_waits(struct hk_device *dev, uint32_t wait_count,
                               const struct vk_sync_wait *waits);

/* A semaphore or fence is destroyed: forget its payloads. */
void hk_gpu_waits_forget_destroyed(struct hk_device *dev,
                                   struct vk_sync *permanent,
                                   struct vk_sync *temporary);

/* A binary wait of a submission to queue: if queue signalled it last, raise
 * floor to that submission's ordinals and return true; the wait then needs
 * no input sync. Any record of the sync is consumed. queue may be NULL. */
bool hk_gpu_waits_take(struct hk_device *dev, const struct hk_queue *queue,
                       const struct vk_sync *sync,
                       uint64_t floor[HK_SYNC_NUM_SQ]);

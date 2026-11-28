/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"
#include "util/macros.h"
#include "asahi/lib/agx_chip_caps.h"
#include "asahi/lib/agx_device.h"
#include "util/cache_ops.h"

#include "vk_command_pool.h"

#define HK_CMD_BO_SIZE     1024 * 128
#define HK_CMD_POOL_BO_MAX 32
#define HK_G17P_CMD_POOL_MAIN_BO_MAX 256

/* A recording with no recycled BO starts each uploader on a span of a shared
 * slab: one tile upload writes under 8 KiB in total, and a whole BO per
 * uploader costs it five VM binds, five mmaps and 640 KiB of first touches.
 * A span is 16 KiB so it keeps the alignment a BO base gives its uploader. */
#define HK_CMD_SPAN_SIZE (16 * 1024)
/* Two G17P render stream roots, or one compute root. */
#define HK_CMD_STREAM_SPAN_SIZE (64 * 1024)
#define HK_CMD_SLAB_MAX_SIZE (4 * 1024 * 1024)

enum hk_cmd_bo_type {
   HK_CMD_BO_MAIN,
   HK_CMD_BO_USC,
   HK_CMD_BO_ROOT,
   HK_CMD_BO_TYPE_COUNT,
};

enum hk_cmd_span_bucket {
   HK_CMD_SPAN_BUCKET_SMALL,
   HK_CMD_SPAN_BUCKET_STREAM,
   HK_CMD_SPAN_BUCKET_COUNT,
};

static inline enum hk_cmd_span_bucket
hk_cmd_span_bucket_for_size(uint32_t size)
{
   return size == HK_CMD_STREAM_SPAN_SIZE ? HK_CMD_SPAN_BUCKET_STREAM
                                          : HK_CMD_SPAN_BUCKET_SMALL;
}

static inline unsigned
hk_cmd_pool_span_cache_max(const struct agx_device *dev,
                           enum hk_cmd_bo_type type)
{
   return agx_chip_native_or_m5(dev->chip) && !dev->is_virtio &&
             (type == HK_CMD_BO_MAIN || type == HK_CMD_BO_ROOT)
             ? HK_G17P_CMD_POOL_MAIN_BO_MAX
             : HK_CMD_POOL_BO_MAX;
}

static inline unsigned
hk_cmd_pool_ioto_span_cache_max(const struct agx_device *dev)
{
   return agx_chip_native_or_m5(dev->chip) && !dev->is_virtio
             ? HK_G17P_CMD_POOL_MAIN_BO_MAX
             : HK_CMD_POOL_BO_MAX;
}

static inline unsigned
hk_cmd_pool_vdm_span_cache_max(const struct agx_device *dev)
{
   return hk_cmd_pool_ioto_span_cache_max(dev);
}

/* Recyclable command buffer BO, used for both push buffers and upload */
struct hk_cmd_bo {
   struct agx_bo *bo;

   void *map;

   /* What this object owns: all of bo, or span of a device slab. */
   uint64_t addr;
   uint32_t size;
   struct hk_vdm_allocation *span;

   /* Full VA of the G17P VDM alias in the architectural user window. */
   uint64_t g17p_vdm_alias;

   /* This recording allocated the BO only through uploader.streams. */
   bool stream_only;

   /** Link in a matching command-pool or command-buffer BO list. */
   struct list_head link;
};

struct hk_vdm_slab;

struct hk_vdm_allocation {
   struct list_head link;
   struct hk_vdm_slab *slab;
   uint64_t addr;
   uint64_t alias;
   uint64_t size;
};

struct agx_device;
struct agx_ptr;
struct hk_device;

/* Command-buffer memory (main/USC/root pools, VDM slabs, Ioto arenas) is
 * CPU-cached on chips with AGX_CAP_CACHED_CMD_MEMORY; recording cleans what it
 * wrote before submission (hk_cmd_buffer_publish_cached_uploads()). */
static inline bool
hk_g17p_cached_cmd_memory(const struct agx_device *dev)
{
   return agx_chip_has_cap(dev->chip, AGX_CAP_CACHED_CMD_MEMORY) &&
          util_has_cache_ops();
}

struct hk_cmd_pool;

void hk_cmd_pool_clean_vdm(const struct hk_vdm_allocation *allocation);

/* A VDM record's pointer is 40 bits and hardware reads bit 39 as the UAT
 * root select, adding 0x1000000000 when it is clear.  G17P's main heap sits
 * at 0x8000000000, so every main-heap pointer a VDM record carried would have
 * bit 39 set and be fetched from the upper root instead: the dVDM_SSD
 * unmapped read at (1<<42)|0x7ffee0400 hardware reported for the first PPP
 * state record after the root chunk (0x87ffee0400).  Native keeps these
 * payloads inside the architectural 0x1000000000 window, and so does G17P
 * recording. */
static inline bool
hk_g17p_vdm_low_window_enabled(const struct agx_device *dev)
{
   return agx_chip_native_agx3(dev->chip);
}

struct hk_cmd_pool {
   struct vk_command_pool vk;

   /** Recyclable hk_cmd_bo lists, kept separate by mapping and VA policy. */
   struct list_head free_bos[HK_CMD_BO_TYPE_COUNT];
   uint32_t num_free_bos[HK_CMD_BO_TYPE_COUNT];
   /** Spans likewise, split by size bucket for O(1) reuse without locking. */
   struct list_head free_spans[HK_CMD_BO_TYPE_COUNT][HK_CMD_SPAN_BUCKET_COUNT];
   uint32_t num_free_spans[HK_CMD_BO_TYPE_COUNT][HK_CMD_SPAN_BUCKET_COUNT];
   /** Recyclable Ioto arena spans (HK_CMD_SLAB_IOTO, HK_CMD_SPAN_SIZE). */
   struct list_head free_ioto_spans;
   uint32_t num_free_ioto_spans;
   /** Recyclable VDM spans (HK_CMD_SLAB_VDM, HK_G17P_VDM_UPLOAD_CHUNK_SIZE). */
   struct list_head free_vdm_spans;
   uint32_t num_free_vdm_spans;
   /** Recyclable control-stream descriptors with warm scissor/depth_bias storage. */
   struct list_head free_cs;
   uint32_t num_free_cs;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_cmd_pool, vk.base, VkCommandPool,
                               VK_OBJECT_TYPE_COMMAND_POOL)

static inline struct hk_device *
hk_cmd_pool_device(struct hk_cmd_pool *pool)
{
   return (struct hk_device *)pool->vk.base.device;
}

VkResult hk_cmd_pool_alloc_bo(struct hk_cmd_pool *pool,
                              enum hk_cmd_bo_type type,
                              struct hk_cmd_bo **bo_out);

void hk_cmd_pool_free_bo_list(struct hk_cmd_pool *pool,
                              enum hk_cmd_bo_type type,
                              struct list_head *bos);
VkResult hk_cmd_pool_alloc_bo_span(struct hk_cmd_pool *pool,
                                   enum hk_cmd_bo_type type, uint32_t size,
                                   struct hk_cmd_bo **bo_out);
VkResult hk_cmd_bo_ensure_vdm_alias(struct hk_cmd_pool *pool,
                                    struct hk_cmd_bo *bo);

VkResult hk_cmd_pool_alloc_span(struct hk_cmd_pool *pool,
                                enum hk_cmd_slab_class class, uint64_t size,
                                uint64_t alignment, struct agx_ptr *ptr,
                                struct hk_vdm_allocation **allocation);
VkResult hk_cmd_pool_alloc_ioto_span(struct hk_cmd_pool *pool,
                                     struct agx_ptr *ptr,
                                     struct hk_vdm_allocation **allocation);
void hk_cmd_pool_free_ioto_span(struct hk_cmd_pool *pool,
                                struct hk_vdm_allocation *allocation);
VkResult hk_cmd_pool_alloc_vdm_span(struct hk_cmd_pool *pool, uint64_t size,
                                    uint64_t alignment, struct agx_ptr *ptr,
                                    struct hk_vdm_allocation **allocation);
void hk_cmd_pool_free_vdm_span(struct hk_cmd_pool *pool,
                               struct hk_vdm_allocation *allocation);
void hk_device_free_span(struct hk_device *dev,
                         struct hk_vdm_allocation *allocation);
void hk_device_finish_vdm_slabs(struct hk_device *dev);

/* Command-memory retirement, independent of API semaphores:
 * - Driver-owned packed compute stream buffers (extra_bo) have no application
 *   handle to keep them alive, so submissions pin them until their completion
 *   syncobj retires.
 * - Application command recordings follow Vulkan lifetime rules while the
 *   device is healthy; when a device is lost (or its VM reports a kernel
 *   error), hk_cmd_pending_retain_lost() keeps their BOs and spans out of
 *   reuse lists and BO caches until device destruction. */
struct agx_bo;
struct hk_cmd_buffer;
struct hk_cmd_pending;
struct hk_queue;
void hk_cmd_pending_init(struct hk_device *dev);
VkResult hk_cmd_pending_begin(struct hk_device *dev, unsigned chunks,
                              struct agx_bo *const *extra_bos,
                              unsigned extra_bo_count,
                              struct hk_queue *reap_queue,
                              bool *reap_done_valid, uint64_t *reap_done,
                              bool already_reaped,
                              struct hk_cmd_pending **out);
uint32_t hk_cmd_pending_signal(struct hk_cmd_pending *pending);
void hk_cmd_pending_end(struct hk_device *dev, struct hk_cmd_pending *pending,
                        struct hk_queue *queue, VkResult result);
void hk_cmd_pending_retain_lost(struct hk_device *dev,
                                struct hk_cmd_buffer *cmd);
void hk_cmd_pending_reap_cached(struct hk_device *dev,
                                const struct hk_queue *cached_queue,
                                bool *cached_valid, uint64_t *cached_done);
void hk_cmd_pending_reap(struct hk_device *dev);
void hk_cmd_pending_finish_queue(struct hk_device *dev, struct hk_queue *queue);
void hk_cmd_pending_finish(struct hk_device *dev);

/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hk_private.h"
#include "vk_query_pool.h"

struct agx_bo;

struct hk_query_pool {
   struct vk_query_pool vk;

   uint64_t query_start;
   uint64_t query_stride;

   struct agx_bo *bo;
   void *bo_map;

   /* For timestamp queries, the kernel-assigned timestamp buffer handle. On
    * the native AGX3 path, occlusion pools bind their availability slots the
    * same way, so the render that ends a query can mark it available through
    * its own retirement timestamp (0 if binding failed). Unused for all other
    * query types.
    */
   uint32_t handle;

   /* log2 of the availability slot size in 32-bit words: 1 for native AGX3
    * occlusion pools (64-bit timestamp slots), 0 otherwise. */
   uint8_t avail_shift;

   unsigned oq_queries;

   /* Occlusion queries: the queue and timeline point of the latest
    * submission that ended each query, point 0 if none since the last host
    * reset. */
   struct hk_oq_submit {
      struct hk_queue *queue;
      uint64_t point;
   } *oq_submit;
};

struct hk_cmd_buffer;
struct hk_cs;
struct hk_device;
struct hk_queue;

struct hk_oq_end {
   struct hk_query_pool *pool;
   uint32_t query;
};

/* Record the queue timeline point that the submission of cmd signals. */
void hk_query_note_submit(struct hk_cmd_buffer *cmd, struct hk_queue *queue,
                          uint64_t point);

void hk_query_stamps_init(struct hk_device *dev);
void hk_query_stamps_finish(struct hk_device *dev);
void hk_query_stamps_unbind(struct hk_device *dev, uint32_t handle);
void hk_query_stamps_clear_queue(struct hk_device *dev, struct hk_queue *queue);
bool hk_query_stamps_conflict(struct hk_device *dev, struct hk_queue *queue,
                              uint64_t current_point, const struct hk_cs *cs);
void hk_query_stamps_note_cs(struct hk_device *dev, struct hk_queue *queue,
                             uint64_t point, const struct hk_cs *cs);
void hk_query_stamps_note_req(struct hk_device *dev, struct hk_queue *queue,
                              uint64_t point, uint32_t handle,
                              uint32_t offset_B);

VK_DEFINE_NONDISP_HANDLE_CASTS(hk_query_pool, vk.base, VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)

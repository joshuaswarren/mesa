/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_cmd_buffer.h"
#include "hk_cmd_pool.h"
#include "hk_device.h"
#include "hk_queue.h"
#include "asahi/lib/agx_bo.h"
#include "util/u_atomic.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned mock_syncobj_query_calls;

int
drmSyncobjQuery(int fd, uint32_t *handles, uint64_t *points,
                uint32_t handle_count)
{
   (void)fd;
   (void)handles;
   mock_syncobj_query_calls += handle_count;
   for (uint32_t i = 0; i < handle_count; ++i)
      points[i] = 0;
   return 0;
}

static void *VKAPI_CALL
test_alloc(void *user_data, size_t size, size_t alignment,
           VkSystemAllocationScope scope)
{
   (void)user_data;
   (void)alignment;
   (void)scope;
   return malloc(size);
}

static void *VKAPI_CALL
test_realloc(void *user_data, void *original, size_t size, size_t alignment,
             VkSystemAllocationScope scope)
{
   (void)user_data;
   (void)alignment;
   (void)scope;
   return realloc(original, size);
}

static void VKAPI_CALL
test_free(void *user_data, void *memory)
{
   (void)user_data;
   free(memory);
}

static const VkAllocationCallbacks test_allocator = {
   .pfnAllocation = test_alloc,
   .pfnReallocation = test_realloc,
   .pfnFree = test_free,
};

static void
init_test_device(struct hk_device *dev, enum agx_chip chip)
{
   memset(dev, 0, sizeof(*dev));
   dev->dev.chip = chip;
   dev->dev.fd = -1;
   dev->vk.alloc = test_allocator;
   simple_mtx_init(&dev->dev.bo_map_lock, mtx_plain);
   simple_mtx_init(&dev->cmd_pending_lock, mtx_plain);
   list_inithead(&dev->cmd_pending_queues);
   list_inithead(&dev->cmd_pending_failed);
   dev->cmd_pending_ready = true;
   dev->cmd_pending_active = 0;
}

static void
init_test_pool(struct hk_cmd_pool *pool, struct hk_device *dev)
{
   memset(pool, 0, sizeof(*pool));
   pool->vk.base.device = &dev->vk;
   pool->vk.alloc = test_allocator;
   for (unsigned t = 0; t < HK_CMD_BO_TYPE_COUNT; ++t) {
      list_inithead(&pool->free_bos[t]);
      list_inithead(&pool->free_spans[t]);
   }
}

static void
init_test_cmd(struct hk_cmd_buffer *cmd, struct hk_cmd_pool *pool)
{
   memset(cmd, 0, sizeof(*cmd));
   cmd->vk.pool = &pool->vk;
   list_inithead(&cmd->uploader.main.bos);
   list_inithead(&cmd->uploader.usc.bos);
   list_inithead(&cmd->uploader.root.bos);
   list_inithead(&cmd->vdm_allocations);
   util_dynarray_init(&cmd->large_bos, NULL);
}

int
main(void)
{
   /* 1. Healthy device: reap is a lock-free no-op when no packed submissions
    * are active, and retain_lost does not block command-pool BO reuse. */
   struct hk_device dev;
   struct hk_cmd_pool pool;
   struct hk_cmd_buffer cmd;
   init_test_device(&dev, AGX_CHIP_G17S);
   init_test_pool(&pool, &dev);
   init_test_cmd(&cmd, &pool);

   mock_syncobj_query_calls = 0;
   hk_cmd_pending_reap(&dev);
   assert(mock_syncobj_query_calls == 0);

   struct agx_bo backing_healthy = { .refcnt = 2, .flags = 0 };
   struct hk_cmd_bo *healthy_bo = calloc(1, sizeof(*healthy_bo));
   healthy_bo->refcount = 1;
   healthy_bo->bo = &backing_healthy;
   list_addtail(&healthy_bo->link, &cmd.uploader.main.bos);

   hk_cmd_pending_retain_lost(&dev, &cmd);
   assert(list_is_empty(&dev.cmd_pending_failed));
   hk_cmd_pool_free_bo_list(&pool, HK_CMD_BO_MAIN, &cmd.uploader.main.bos);
   assert(pool.num_free_bos[HK_CMD_BO_MAIN] == 1);
   assert(p_atomic_read(&healthy_bo->refcount) == 1);
   free(healthy_bo);

   /* 2. Lost device: retain_lost pins command BOs and large BOs out of reuse
    * lists and BO caches until hk_cmd_pending_finish() tears them down with
    * AGX_BO_NO_CACHE. */
   p_atomic_set(&dev.vk._lost.lost, 1);
   init_test_pool(&pool, &dev);
   init_test_cmd(&cmd, &pool);

   struct agx_bo backing_cmd = { .refcnt = 2, .flags = 0 };
   struct hk_cmd_bo *lost_bo = calloc(1, sizeof(*lost_bo));
   lost_bo->refcount = 1;
   lost_bo->bo = &backing_cmd;
   list_addtail(&lost_bo->link, &cmd.uploader.main.bos);

   struct agx_bo large_bo = { .refcnt = 2, .flags = 0 };
   struct agx_bo *large_ptr = &large_bo;
   util_dynarray_append(&cmd.large_bos, large_ptr);

   hk_cmd_pending_retain_lost(&dev, &cmd);
   assert(!list_is_empty(&dev.cmd_pending_failed));
   assert(p_atomic_read(&lost_bo->refcount) == 2);
   assert(p_atomic_read(&large_bo.refcnt) == 3);

   hk_cmd_pool_free_bo_list(&pool, HK_CMD_BO_MAIN, &cmd.uploader.main.bos);
   assert(pool.num_free_bos[HK_CMD_BO_MAIN] == 0);
   assert(p_atomic_read(&lost_bo->refcount) == 1);
   agx_bo_unreference(&dev.dev, &large_bo);
   assert(p_atomic_read(&large_bo.refcnt) == 2);
   util_dynarray_fini(&cmd.large_bos);

   hk_cmd_pending_finish(&dev);
   assert(list_is_empty(&dev.cmd_pending_failed));
   assert(p_atomic_read(&backing_cmd.refcnt) == 1);
   assert((backing_cmd.flags & AGX_BO_NO_CACHE) != 0);
   assert(p_atomic_read(&large_bo.refcnt) == 1);
   assert((large_bo.flags & AGX_BO_NO_CACHE) != 0);

   simple_mtx_destroy(&dev.cmd_pending_lock);
   simple_mtx_destroy(&dev.dev.bo_map_lock);

   /* 3. Kernel VM status mirror error triggers retention even before the
    * VkDevice lost bit is latched (uniformly across AGX3B and AGX3C). */
   init_test_device(&dev, AGX_CHIP_G16G);
   int32_t vm_error = -5; /* -EIO */
   dev.vm_status_mirror.word = &vm_error;
   init_test_pool(&pool, &dev);
   init_test_cmd(&cmd, &pool);

   struct agx_bo backing_mirror = { .refcnt = 2, .flags = 0 };
   struct hk_cmd_bo *mirror_bo = calloc(1, sizeof(*mirror_bo));
   mirror_bo->refcount = 1;
   mirror_bo->bo = &backing_mirror;
   list_addtail(&mirror_bo->link, &cmd.uploader.usc.bos);

   hk_cmd_pending_retain_lost(&dev, &cmd);
   assert(!list_is_empty(&dev.cmd_pending_failed));
   hk_cmd_pool_free_bo_list(&pool, HK_CMD_BO_USC, &cmd.uploader.usc.bos);
   assert(pool.num_free_bos[HK_CMD_BO_USC] == 0);
   hk_cmd_pending_finish(&dev);
   assert(p_atomic_read(&backing_mirror.refcnt) == 1);
   assert((backing_mirror.flags & AGX_BO_NO_CACHE) != 0);

   simple_mtx_destroy(&dev.cmd_pending_lock);
   simple_mtx_destroy(&dev.dev.bo_map_lock);

   puts("hk_cmd_pending: ok");
   return 0;
}

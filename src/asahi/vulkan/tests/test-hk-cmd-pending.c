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
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool fail_allocations;
static unsigned allocation_attempts;

static void *VKAPI_CALL
test_alloc(void *user_data, size_t size, size_t alignment,
           VkSystemAllocationScope scope)
{
   (void)user_data;
   (void)alignment;
   (void)scope;
   allocation_attempts++;
   return fail_allocations ? NULL : malloc(size);
}

static void *VKAPI_CALL
test_realloc(void *user_data, void *original, size_t size, size_t alignment,
             VkSystemAllocationScope scope)
{
   (void)user_data;
   (void)alignment;
   (void)scope;
   allocation_attempts++;
   return fail_allocations ? NULL : realloc(original, size);
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
   hk_cmd_pending_init(dev);
}

static void
init_test_pool(struct hk_cmd_pool *pool, struct hk_device *dev)
{
   memset(pool, 0, sizeof(*pool));
   pool->vk.base.device = &dev->vk;
   pool->vk.alloc = test_allocator;
   for (unsigned t = 0; t < HK_CMD_BO_TYPE_COUNT; ++t) {
      list_inithead(&pool->free_bos[t]);
      for (unsigned bucket = 0; bucket < HK_CMD_SPAN_BUCKET_COUNT; ++bucket)
         list_inithead(&pool->free_spans[t][bucket]);
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

static void
test_healthy_reuse(void)
{
   struct hk_device dev;
   struct hk_cmd_pool pool;
   struct hk_cmd_buffer cmd;
   init_test_device(&dev, AGX_CHIP_G17S);
   init_test_pool(&pool, &dev);
   init_test_cmd(&cmd, &pool);

   struct agx_bo backing = { .refcnt = 2 };
   struct hk_cmd_bo *bo = calloc(1, sizeof(*bo));
   assert(bo);
   bo->bo = &backing;
   list_addtail(&bo->link, &cmd.uploader.main.bos);

   /* An idle reap must also work while the current thread owns the lock. */
   simple_mtx_lock(&dev.cmd_pending_lock);
   hk_cmd_pending_reap(&dev);
   simple_mtx_unlock(&dev.cmd_pending_lock);

   hk_cmd_pending_retain_lost(&dev, &cmd);
   assert(list_is_empty(&dev.cmd_lost_bos));
   hk_cmd_pool_free_bo_list(&pool, HK_CMD_BO_MAIN, &cmd.uploader.main.bos);
   assert(pool.num_free_bos[HK_CMD_BO_MAIN] == 1);
   struct hk_cmd_bo *reused = NULL;
   assert(hk_cmd_pool_alloc_bo(&pool, HK_CMD_BO_MAIN, &reused) == VK_SUCCESS);
   assert(reused == bo);
   free(bo);
   simple_mtx_destroy(&dev.cmd_pending_lock);
   simple_mtx_destroy(&dev.dev.bo_map_lock);
}

static void
test_lost_quarantine(enum agx_chip chip, bool vm_status_loss)
{
   struct hk_device dev;
   struct hk_cmd_pool pool;
   struct hk_cmd_buffer cmd, shared;
   init_test_device(&dev, chip);
   init_test_pool(&pool, &dev);
   init_test_cmd(&cmd, &pool);
   init_test_cmd(&shared, &pool);

   int32_t vm_error = -EIO;
   if (vm_status_loss)
      dev.vm_status_mirror.word = &vm_error;
   else
      p_atomic_set(&dev.vk._lost.lost, 1);

   struct agx_bo backing = { .refcnt = 2 };
   struct hk_cmd_bo *bo = calloc(1, sizeof(*bo));
   assert(bo);
   bo->bo = &backing;
   list_addtail(&bo->link, &cmd.uploader.main.bos);

   struct hk_vdm_allocation vdm = {0}, ioto = {0};
   list_addtail(&vdm.link, &cmd.vdm_allocations);
   cmd.ioto.span = &ioto;

   struct agx_bo large = { .refcnt = 3 };
   struct agx_bo *large_ptr = &large;
   util_dynarray_append(&cmd.large_bos, large_ptr);
   util_dynarray_append(&shared.large_bos, large_ptr);

   allocation_attempts = 0;
   fail_allocations = true;
   hk_cmd_pending_retain_lost(&dev, &cmd);
   hk_cmd_pending_retain_lost(&dev, &shared);
   fail_allocations = false;
   assert(allocation_attempts == 0);

   assert(list_is_empty(&cmd.uploader.main.bos));
   assert(list_first_entry(&dev.cmd_lost_bos, struct hk_cmd_bo, link) == bo);
   hk_cmd_pool_free_bo_list(&pool, HK_CMD_BO_MAIN, &cmd.uploader.main.bos);
   assert(pool.num_free_bos[HK_CMD_BO_MAIN] == 0);
   assert(list_is_empty(&cmd.vdm_allocations));
   assert(cmd.ioto.span == NULL);
   assert(list_first_entry(&dev.cmd_lost_spans, struct hk_vdm_allocation,
                           link) == &vdm);
   assert(list_last_entry(&dev.cmd_lost_spans, struct hk_vdm_allocation,
                          link) == &ioto);

   assert(p_atomic_read(&large.refcnt) == 4);
   assert(list_is_singular(&dev.cmd_lost_large_bos));
   hk_cmd_buffer_release_bos(&cmd, &dev.dev);
   hk_cmd_buffer_release_bos(&shared, &dev.dev);
   assert(p_atomic_read(&large.refcnt) == 2);
   util_dynarray_fini(&cmd.large_bos);
   util_dynarray_fini(&shared.large_bos);

   /* Stack-only test spans do not have backing slabs to release. */
   list_del(&vdm.link);
   list_del(&ioto.link);
   hk_cmd_pending_finish(&dev);
   assert(list_is_empty(&dev.cmd_lost_bos));
   assert(list_is_empty(&dev.cmd_lost_large_bos));
   assert(!list_is_linked(&large.lru_link));
   assert(p_atomic_read(&backing.refcnt) == 1);
   assert(backing.flags & AGX_BO_NO_CACHE);
   assert(p_atomic_read(&large.refcnt) == 1);
   assert(large.flags & AGX_BO_NO_CACHE);

   simple_mtx_destroy(&dev.cmd_pending_lock);
   simple_mtx_destroy(&dev.dev.bo_map_lock);
}

int
main(void)
{
   test_healthy_reuse();
   test_lost_quarantine(AGX_CHIP_G17S, false);
   test_lost_quarantine(AGX_CHIP_G16G, true);
   puts("hk_cmd_pending: ok");
   return 0;
}

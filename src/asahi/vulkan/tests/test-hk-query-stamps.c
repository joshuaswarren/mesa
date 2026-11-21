/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_query_pool.h"
#include "hk_queue.h"
#include "vk_alloc.h"
#include "vk_instance.h"
#include "vk_physical_device.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>

#define MOCK_SYNCOBJ_MAX 128u
static uint64_t mock_syncobj_points[MOCK_SYNCOBJ_MAX];
static unsigned mock_syncobj_calls[MOCK_SYNCOBJ_MAX];
static unsigned mock_syncobj_total_calls;

int
drmSyncobjQuery(int fd, uint32_t *handles, uint64_t *points,
                uint32_t handle_count)
{
   (void)fd;
   for (uint32_t i = 0; i < handle_count; ++i) {
      mock_syncobj_total_calls++;
      if (handles[i] == 0 || handles[i] >= MOCK_SYNCOBJ_MAX)
         return -1;
      mock_syncobj_calls[handles[i]]++;
      points[i] = mock_syncobj_points[handles[i]];
   }
   return 0;
}

static void
reset_mock_syncobj(void)
{
   memset(mock_syncobj_points, 0, sizeof(mock_syncobj_points));
   memset(mock_syncobj_calls, 0, sizeof(mock_syncobj_calls));
   mock_syncobj_total_calls = 0;
}

static void
init_fake_queue(struct hk_queue *queue, uint32_t syncobj, uint64_t signalled)
{
   memset(queue, 0, sizeof(*queue));
   queue->drm.syncobj = syncobj;
   queue->drm.signalled_value = signalled;
}

static void
init_cdm_cs(struct hk_cs *cs)
{
   memset(cs, 0, sizeof(*cs));
   cs->type = HK_CS_CDM;
}

static void
init_vdm_cs(struct hk_cs *cs)
{
   memset(cs, 0, sizeof(*cs));
   cs->type = HK_CS_VDM;
}

int
main(void)
{
   struct hk_device dev;
   memset(&dev, 0, sizeof(dev));
   dev.dev.fd = -1;
   dev.dev.params.features = DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
   hk_query_stamps_init(&dev);

   struct hk_queue q1, q2;
   reset_mock_syncobj();
   init_fake_queue(&q1, 1, 10);
   init_fake_queue(&q2, 2, 20);
   mock_syncobj_points[1] = 10;
   mock_syncobj_points[2] = 20;

   /* Feature-gated off: note_cs is a no-op and conflict fails closed. */
   struct hk_device dev_off;
   memset(&dev_off, 0, sizeof(dev_off));
   dev_off.dev.fd = -1;
   hk_query_stamps_init(&dev_off);
   struct hk_cs prod_off;
   init_cdm_cs(&prod_off);
   prod_off.timestamp.end = (struct agx_timestamp_req){ .handle = 7, .offset_B = 16 };
   hk_query_stamps_note_cs(&dev_off, &q1, 11, &prod_off);
   assert(dev_off.query_stamps.pending.size == 0);
   struct hk_cs cons_off;
   init_cdm_cs(&cons_off);
   assert(hk_query_stamps_conflict(&dev_off, &q1, 12, &cons_off));
   hk_query_stamps_finish(&dev_off);

   /* Pending CDM end timestamp on handle 7, offset 16 (bytes [16, 24)), point 11. */
   struct hk_cs prod_cdm;
   init_cdm_cs(&prod_cdm);
   prod_cdm.timestamp.end = (struct agx_timestamp_req){
      .handle = 7,
      .offset_B = 16,
   };
   hk_query_stamps_note_cs(&dev, &q1, 11, &prod_cdm);

   /* Pending VDM oq_stamp on handle 8, offset 32 (bytes [32, 40)), point 21. */
   struct hk_cs prod_vdm;
   init_vdm_cs(&prod_vdm);
   prod_vdm.oq_stamp[2] = (struct agx_timestamp_req){
      .handle = 8,
      .offset_B = 32,
   };
   hk_query_stamps_note_cs(&dev, &q2, 21, &prod_vdm);

   /* Disjoint range on same handle -> no conflict. */
   struct hk_cs cons;
   init_cdm_cs(&cons);
   hk_cs_note_query_range(&cons, 7, 0, 16);
   hk_cs_note_query_range(&cons, 7, 24, 16);
   /* Recorded words decide the flag only: query_access stays "copy or
    * reset" for its other users (sibling kicks, packing). */
   assert(!cons.query_access);
   assert(!cons.query_access_untracked);
   assert(!hk_query_stamps_conflict(&dev, &q1, 12, &cons));
   util_dynarray_fini(&cons.query_access_ranges);

   /* A copy of an unbound pool (handle 0) records no word: no conflict. */
   init_cdm_cs(&cons);
   cons.query_access = true;
   hk_cs_note_query_range(&cons, 0, 0, 0);
   assert(!cons.query_access_ranges.size && !cons.query_access_untracked);
   assert(!hk_query_stamps_conflict(&dev, &q1, 12, &cons));
   util_dynarray_fini(&cons.query_access_ranges);

   /* Overlapping range on handle 7 -> conflict while point 11 > signalled 10.
    * Checking from the same in-progress submission (point 11) must return
    * conflict without issuing any drmSyncobjQuery ioctl. */
   init_cdm_cs(&cons);
   hk_cs_note_query_range(&cons, 7, 8, 16);
   mock_syncobj_total_calls = 0;
   assert(hk_query_stamps_conflict(&dev, &q1, 11, &cons));
   assert(mock_syncobj_total_calls == 0);
   assert(hk_query_stamps_conflict(&dev, &q1, 12, &cons));
   assert(mock_syncobj_calls[1] == 1);

   /* Once q1 advances past point 11, the pending entry retires and clears. */
   p_atomic_set(&q1.drm.signalled_value, 11);
   assert(!hk_query_stamps_conflict(&dev, &q1, 12, &cons));
   util_dynarray_fini(&cons.query_access_ranges);

   /* Overlapping VDM oq_stamp on handle 8 -> conflict until unbound or cleared. */
   init_cdm_cs(&cons);
   hk_cs_note_query_range(&cons, 8, 32, 8);
   assert(hk_query_stamps_conflict(&dev, &q2, 22, &cons));

   /* Unbinding handle 8 removes its pending entries. */
   hk_query_stamps_unbind(&dev, 8);
   assert(!hk_query_stamps_conflict(&dev, &q2, 22, &cons));

   /* Re-add handle 8 on q2 and clear q2. */
   hk_query_stamps_note_cs(&dev, &q2, 22, &prod_vdm);
   assert(hk_query_stamps_conflict(&dev, &q2, 23, &cons));
   hk_query_stamps_clear_queue(&dev, &q2);
   assert(!hk_query_stamps_conflict(&dev, &q2, 23, &cons));

   /* Untracked access always fails closed. */
   cons.query_access_untracked = true;
   assert(hk_query_stamps_conflict(&dev, &q2, 23, &cons));
   util_dynarray_fini(&cons.query_access_ranges);

   /* Merging CDM streams preserves query_access_ranges and untracked flag. */
   struct hk_cs a, b;
   init_cdm_cs(&a);
   init_cdm_cs(&b);
   uint8_t dummy_a[4096] = {0}, dummy_b[64] = {0};
   a.start = a.current = dummy_a;
   a.end = dummy_a + sizeof(dummy_a);
   b.start = dummy_b;
   b.current = dummy_b + 16;
   b.end = dummy_b + sizeof(dummy_b);
   hk_query_stamps_note_cs(&dev, &q2, 23, &prod_vdm);
   hk_cs_note_query_range(&b, 8, 32, 8);
   hk_cs_merge_cdm_until(&a, &b, b.current);
   assert(!a.query_access);
   assert(!a.query_access_untracked);
   assert(hk_query_stamps_conflict(&dev, &q2, 24, &a));
   util_dynarray_fini(&a.query_access_ranges);
   util_dynarray_fini(&b.query_access_ranges);
   util_dynarray_fini(&a.compute_resources);
   util_dynarray_fini(&b.compute_resources);

   /* Merging an untracked CDM stream propagates query_access_untracked. */
   init_cdm_cs(&a);
   init_cdm_cs(&b);
   a.start = a.current = dummy_a;
   a.end = dummy_a + sizeof(dummy_a);
   b.start = dummy_b;
   b.current = dummy_b + 16;
   b.end = dummy_b + sizeof(dummy_b);
   b.query_access = true;
   b.query_access_untracked = true;
   hk_cs_merge_cdm_until(&a, &b, b.current);
   assert(a.query_access);
   assert(a.query_access_untracked);
   assert(hk_query_stamps_conflict(&dev, &q2, 24, &a));
   util_dynarray_fini(&a.query_access_ranges);
   util_dynarray_fini(&b.query_access_ranges);
   util_dynarray_fini(&a.compute_resources);
   util_dynarray_fini(&b.compute_resources);

   /* Fallback bound occlusion store (imm_write_query_ranges without oq_stamps)
    * records its range even with 64-bit offsets above 4 GiB. */
   init_cdm_cs(&cons);
   hk_cs_note_imm_write_query_range(&cons, 8, 32, 8);
   hk_cs_note_imm_write_query_range(&cons, 9, UINT64_C(0x100000020), 8);
   util_dynarray_foreach(&cons.imm_write_query_ranges,
                         struct hk_query_access_range, r) {
      hk_cs_note_query_range(&cons, r->handle, r->start_B, r->end_B - r->start_B);
   }
   assert(!cons.query_access_untracked);
   assert(hk_query_stamps_conflict(&dev, &q2, 24, &cons));
   util_dynarray_fini(&cons.imm_write_query_ranges);
   util_dynarray_fini(&cons.query_access_ranges);

   /* Large burst of distinct pending stamps grows the hash index cleanly,
    * deduplicates repeat updates in place, and shrinks cleanly on unbind. */
   for (unsigned k = 0; k < 2048; ++k) {
      struct hk_cs p;
      init_cdm_cs(&p);
      p.timestamp.end = (struct agx_timestamp_req){
         .handle = 100,
         .offset_B = k * 8,
      };
      hk_query_stamps_note_cs(&dev, &q1, 100, &p);
   }
   const size_t after_burst = dev.query_stamps.pending.size;
   assert(dev.query_stamps.slot_cap >= 4096);
   assert(dev.query_stamps.prune_threshold >= 2048);
   for (unsigned k = 0; k < 2048; ++k) {
      struct hk_cs p;
      init_cdm_cs(&p);
      p.timestamp.end = (struct agx_timestamp_req){
         .handle = 100,
         .offset_B = k * 8,
      };
      hk_query_stamps_note_cs(&dev, &q1, 101, &p);
   }
   assert(dev.query_stamps.pending.size == after_burst);
   init_cdm_cs(&cons);
   hk_cs_note_query_range(&cons, 100, 2047 * 8, 8);
   assert(hk_query_stamps_conflict(&dev, &q1, 102, &cons));
   p_atomic_set(&q1.drm.signalled_value, 100);
   assert(hk_query_stamps_conflict(&dev, &q1, 102, &cons));
   p_atomic_set(&q1.drm.signalled_value, 101);
   assert(!hk_query_stamps_conflict(&dev, &q1, 102, &cons));
   util_dynarray_fini(&cons.query_access_ranges);
   hk_query_stamps_unbind(&dev, 100);

   /* More than 32 distinct queues exercise heap growth in the per-call
    * queried-queue cache: retire 2 overlapping stamps per queue across 40
    * queues via drmSyncobjQuery and verify exactly 1 ioctl per queue. */
   reset_mock_syncobj();
   struct hk_queue many_queues[40];
   for (unsigned q = 0; q < ARRAY_SIZE(many_queues); ++q) {
      const uint32_t handle = 10u + q;
      init_fake_queue(&many_queues[q], handle, 5);
      mock_syncobj_points[handle] = 5;
      for (unsigned k = 0; k < 2; ++k) {
         struct hk_cs p;
         init_cdm_cs(&p);
         p.timestamp.end = (struct agx_timestamp_req){
            .handle = 200,
            .offset_B = (q * 2 + k) * 8,
         };
         hk_query_stamps_note_cs(&dev, &many_queues[q], 6, &p);
      }
   }
   /* Now advance the mock kernel timeline on all 40 queues without updating
    * cached signalled_value, so the conflict check must query all 40 queues,
    * growing hk_query_queue_set past its 32-queue inline buffer. */
   for (unsigned q = 0; q < ARRAY_SIZE(many_queues); ++q) {
      const uint32_t handle = 10u + q;
      mock_syncobj_points[handle] = 6;
      mock_syncobj_calls[handle] = 0;
   }
   mock_syncobj_total_calls = 0;
   init_cdm_cs(&cons);
   hk_cs_note_query_range(&cons, 200, 0, ARRAY_SIZE(many_queues) * 16);
   assert(!hk_query_stamps_conflict(&dev, &many_queues[0], 7, &cons));
   assert(mock_syncobj_total_calls == ARRAY_SIZE(many_queues));
   for (unsigned q = 0; q < ARRAY_SIZE(many_queues); ++q)
      assert(mock_syncobj_calls[10u + q] == 1);
   util_dynarray_fini(&cons.query_access_ranges);
   for (unsigned q = 0; q < ARRAY_SIZE(many_queues); ++q)
      hk_query_stamps_clear_queue(&dev, &many_queues[q]);

   /* Conflict scan budget: a retired stamp followed by > 4096 live stamps
    * fails closed in O(1) bounded time without retiring in place or rebuilding
    * the hash index under the lock. Keep point 199 unsignalled while inserting
    * all stamps so insertion-time pruning cannot remove it before the checks. */
   p_atomic_set(&q1.drm.signalled_value, 198);
   mock_syncobj_points[1] = 198;
   const size_t base_budget_size = dev.query_stamps.pending.size;
   {
      struct hk_cs retired;
      init_cdm_cs(&retired);
      retired.timestamp.end = (struct agx_timestamp_req){
         .handle = 300,
         .offset_B = 0,
      };
      hk_query_stamps_note_cs(&dev, &q1, 199, &retired);
   }
   const size_t entry_size = dev.query_stamps.pending.size - base_budget_size;
   assert(entry_size > 0);
   for (unsigned k = 1; k <= 4200; ++k) {
      struct hk_cs p;
      init_cdm_cs(&p);
      p.timestamp.end = (struct agx_timestamp_req){
         .handle = 300,
         .offset_B = k * 8,
      };
      hk_query_stamps_note_cs(&dev, &q1, 200, &p);
   }
   const size_t before_budget_check = dev.query_stamps.pending.size;
   assert(before_budget_check == base_budget_size + 4201 * entry_size);
   p_atomic_set(&q1.drm.signalled_value, 199);
   mock_syncobj_points[1] = 199;
   uint32_t *const slot_index_before = dev.query_stamps.slot_index;
   init_cdm_cs(&cons);
   hk_cs_note_query_range(&cons, 300, 0, 8);
   assert(hk_query_stamps_conflict(&dev, &q1, 201, &cons));
   assert(dev.query_stamps.pending.size == before_budget_check);
   assert(dev.query_stamps.slot_index == slot_index_before);
   util_dynarray_fini(&cons.query_access_ranges);
   init_cdm_cs(&cons);
   hk_cs_note_query_range(&cons, 301, 0, 8);
   assert(hk_query_stamps_conflict(&dev, &q1, 201, &cons));
   assert(dev.query_stamps.pending.size == before_budget_check);
   assert(dev.query_stamps.slot_index == slot_index_before);
   util_dynarray_fini(&cons.query_access_ranges);
   hk_query_stamps_unbind(&dev, 300);

   /* Oversized timestamp pool (> 4 GiB) is rejected at creation time rather
    * than failing recording with VK_ERROR_FEATURE_NOT_PRESENT. */
   struct vk_instance dummy_inst;
   struct vk_physical_device dummy_phys;
   memset(&dummy_inst, 0, sizeof(dummy_inst));
   memset(&dummy_phys, 0, sizeof(dummy_phys));
   dummy_inst.base.type = VK_OBJECT_TYPE_INSTANCE;
   dummy_inst.base.client_visible = true;
   list_inithead(&dummy_inst.debug_utils.callbacks);
   mtx_init(&dummy_inst.debug_utils.callbacks_mutex, mtx_plain);
   list_inithead(&dummy_inst.debug_report.callbacks);
   mtx_init(&dummy_inst.debug_report.callbacks_mutex, mtx_plain);
   dummy_phys.instance = &dummy_inst;
   dev.vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev.vk.base.device = &dev.vk;
   dev.vk.physical = &dummy_phys;
   dev.vk.alloc = *vk_default_allocator();
   VkQueryPool huge_pool = VK_NULL_HANDLE;
   VkQueryPoolCreateInfo huge_ci = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = 0x30000000u,
   };
   assert(hk_CreateQueryPool(hk_device_to_handle(&dev), &huge_ci, NULL,
                             &huge_pool) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(huge_pool == VK_NULL_HANDLE);
   mtx_destroy(&dummy_inst.debug_report.callbacks_mutex);
   mtx_destroy(&dummy_inst.debug_utils.callbacks_mutex);

   hk_query_stamps_finish(&dev);
   puts("hk_query_stamps: ok");
   return 0;
}

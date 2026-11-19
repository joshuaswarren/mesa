/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Production AS entrypoints with host-backed sparse-page/bind fixtures.
 * A million distinct objects exercise the reserved-ID gap, free-list reuse,
 * allocation rollback and teardown without DRM or native GPU execution.
 * Structures at unaligned offsets exercise traversal-base windows.
 */
#undef NDEBUG
#include "hk_device.h"
#include "hk_buffer.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_gen1_rt_indirect_contract.h"
#include "asahi/libagx/gen1_as.h"
#include "vk_acceleration_structure.h"
#include "vk_alloc.h"
#include "vk_instance.h"
#include "util/hash_table.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static struct hk_device dev;
static struct hk_physical_device pdev;
static struct vk_instance instance;
static bool fail_insert;
static unsigned fail_page;
static unsigned pages;

struct agx_bo *__wrap_agx_bo_create(struct agx_device *, size_t, unsigned,
                                    enum agx_bo_flags, const char *);
struct agx_bo *
__wrap_agx_bo_create(struct agx_device *d, size_t size, unsigned align,
                      enum agx_bo_flags flags, const char *label)
{
   if (fail_page && ++pages == fail_page)
      return NULL;
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   assert(bo);
   bo->size = size;
   bo->_map = calloc(1, size);
   bo->_map_size = size;
   assert(bo->_map);
   return bo;
}
__typeof__(agx_bo_bind) __wrap_agx_bo_bind;
int __wrap_agx_bo_bind(struct agx_device *d, struct agx_bo *b, uint64_t addr,
                      size_t size, uint64_t offset, uint32_t flags)
{ return 0; }
static uint64_t next_va = 0x1000000000;
struct agx_va *__wrap_agx_va_alloc(struct agx_device *, uint64_t, uint64_t,
                                   enum agx_va_flags, uint64_t);
struct agx_va *
__wrap_agx_va_alloc(struct agx_device *d, uint64_t size, uint64_t align,
                    enum agx_va_flags flags, uint64_t fixed)
{
   /* Only traversal-base windows allocate address space here. */
   assert((flags & AGX_VA_LOW) && !fixed && align == 16384);
   struct agx_va *va = calloc(1, sizeof(*va));
   assert(va);
   va->addr = next_va;
   va->size_B = size;
   next_va += ALIGN_POT(size, align);
   return va;
}
void __wrap_agx_va_free(struct agx_device *, struct agx_va *, bool);
void __wrap_agx_va_free(struct agx_device *d, struct agx_va *va, bool unbind)
{
   free(va);
}
struct hash_entry *__real__mesa_hash_table_insert(struct hash_table *, const void *, void *);
struct hash_entry *__wrap__mesa_hash_table_insert(struct hash_table *, const void *, void *);
struct hash_entry *
__wrap__mesa_hash_table_insert(struct hash_table *table, const void *key, void *data)
{
   if (table == dev.gen1_as_ids && fail_insert) {
      fail_insert = false;
      return NULL;
   }
   return __real__mesa_hash_table_insert(table, key, data);
}
static unsigned id(VkAccelerationStructureKHR handle)
{
   return (uintptr_t)_mesa_hash_table_search(dev.gen1_as_ids,
      vk_acceleration_structure_from_handle(handle))->data;
}
static void init_table(struct hk_gen1_sparse_table *t, unsigned entry_bytes)
{
   static struct agx_va va = {.addr = 0x1000000000};
   *t = (struct hk_gen1_sparse_table){.va = &va, .entry_bytes = entry_bytes};
   util_dynarray_init(&t->pages, NULL);
}
static void finish_table(struct hk_gen1_sparse_table *t)
{
   util_dynarray_foreach(&t->pages, struct agx_bo *, bo) {
      free((*bo)->_map);
      free(*bo);
   }
   util_dynarray_fini(&t->pages);
}
static VkResult create_at(VkAccelerationStructureKHR *out, uint64_t address,
                          uint64_t offset)
{
   static struct hk_buffer buffer;
   buffer.vk.device_address = address;
   vk_object_base_init(&dev.vk, &buffer.vk.base, VK_OBJECT_TYPE_BUFFER);
   const VkAccelerationStructureCreateInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
      .buffer = hk_buffer_to_handle(&buffer),
      .offset = offset,
      .size = 20000,
      .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
   };
   VkResult result = hk_CreateAccelerationStructureKHR(
      hk_device_to_handle(&dev), &info, NULL, out);
   vk_object_base_finish(&buffer.vk.base);
   return result;
}
static VkResult create(VkAccelerationStructureKHR *out)
{
   return create_at(out, 0x2000000000, 0);
}
static void destroy(VkAccelerationStructureKHR as)
{
   hk_DestroyAccelerationStructureKHR(hk_device_to_handle(&dev), as, NULL);
}
static uint64_t table_entry(unsigned as_id)
{
   const unsigned per_page = 16384 / 16;
   struct agx_bo *bo = *util_dynarray_element(&dev.gen1_as_table_pages.pages,
                                              struct agx_bo *, as_id / per_page);
   return *(uint64_t *)((uint8_t *)bo->_map + (as_id % per_page) * 16);
}
static uint64_t forward(const void *node, uint64_t base)
{
   const uint32_t *w = node;
   assert(libagx_gen1_as_is_node(w[0]) && (w[14] & 0xffff) == 0x1f01);
   return base + (uint64_t)libagx_gen1_as_child_slot(w[10], w[12]) * 64;
}
/* Follows base -> shared root -> trampoline -> structure as traversal does. */
static void windows(void)
{
   enum { COUNT = 300, PAGE = 16384 };
   const uint64_t buffer = 0x2000000000, first_window = next_va;
   VkAccelerationStructureKHR handles[COUNT];
   for (unsigned i = 0; i < COUNT; ++i) {
      const uint64_t offset = 256 + (uint64_t)i * 512;
      assert(create_at(&handles[i], buffer, offset) == VK_SUCCESS);
      const unsigned w = i / 255, j = i % 255;
      const uint64_t window = first_window + (uint64_t)w * 256 * PAGE;
      const uint64_t base = table_entry(id(handles[i]));
      assert(base == window + (uint64_t)j * PAGE);
      const struct hk_gen1_as_window *win = util_dynarray_element(
         &dev.gen1_as_windows, struct hk_gen1_as_window, w);
      assert(win->va->addr == window);
      const uint8_t *roots = dev.gen1_as_window_roots->_map;
      const uint64_t trampoline = forward(roots + (uint64_t)j * PAGE, base);
      assert(trampoline == window + 255 * PAGE + j * 64);
      assert(forward((uint8_t *)win->trampolines->_map + j * 64, base) ==
             buffer + offset);
   }
   assert(util_dynarray_num_elements(&dev.gen1_as_windows,
                                     struct hk_gen1_as_window) == 2);
   /* An aligned structure is its own base; one below every window has none. */
   VkAccelerationStructureKHR aligned, below;
   assert(create_at(&aligned, buffer, 3 * PAGE) == VK_SUCCESS);
   assert(table_entry(id(aligned)) == buffer + 3 * PAGE);
   assert(create_at(&below, first_window - 4 * PAGE, 256) ==
          VK_ERROR_OUT_OF_DEVICE_MEMORY);
   destroy(aligned);
   for (unsigned i = 0; i < COUNT; ++i)
      destroy(handles[i]);
   util_dynarray_foreach(&dev.gen1_as_windows, struct hk_gen1_as_window, win) {
      assert(win->live == 0);
      for (unsigned j = 0; j < 255 * 64; ++j)
         assert(!((uint8_t *)win->trampolines->_map)[j]);
   }
   /* Freed bases are reused before any new window. */
   assert(create_at(&handles[0], buffer, 256) == VK_SUCCESS);
   assert(table_entry(id(handles[0])) % (256 * PAGE) == first_window % (256 * PAGE));
   destroy(handles[0]);
   printf("PASS: %u windowed structures over 2 windows, forwarding chain, reuse\n", COUNT);
}
static void free_fake_bo(struct agx_bo *bo)
{
   if (bo) {
      free(bo->_map);
      free(bo);
   }
}
static double now(void)
{
   struct timespec ts;clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec + ts.tv_nsec * 1e-9;
}
int main(void)
{
   vk_object_base_init(NULL, &instance.base, VK_OBJECT_TYPE_INSTANCE);
   instance.base.client_visible = true;
   list_inithead(&instance.debug_utils.callbacks);
   list_inithead(&instance.debug_report.callbacks);
   vk_object_base_init(NULL, &pdev.vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   pdev.vk.instance = &instance;
   dev.vk.physical = &pdev.vk;
   dev.vk.alloc = *vk_default_allocator();
   vk_object_base_init(&dev.vk, &dev.vk.base, VK_OBJECT_TYPE_DEVICE);
   dev.g17p_native_as = true;
   simple_mtx_init(&dev.gen1_as_lock, mtx_plain);
   util_dynarray_init(&dev.gen1_ases, NULL);
   dev.gen1_as_free_head = UINT32_MAX;
   dev.gen1_as_ids = _mesa_pointer_hash_table_create(NULL);
   init_table(&dev.gen1_as_address_pages, 8);
   init_table(&dev.gen1_as_table_pages, 16);
   init_table(&dev.gen1_as_metadata_pages, 64);
   VkAccelerationStructureKHR as;
   fail_page = 1;
   assert(create(&as) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(util_dynarray_num_elements(&dev.gen1_ases, struct hk_gen1_as_slot) == 0);
   fail_page = 0;
   fail_insert = true;
   assert(create(&as) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(dev.gen1_as_free_head == 0);
   assert(create(&as) == VK_SUCCESS && id(as) == 1);
   destroy(as);
   const unsigned counts[] = {65536, 1048577};
   for (unsigned iteration = 0; iteration < 2; ++iteration) {
      unsigned count = counts[iteration];
      VkAccelerationStructureKHR *handles = calloc(count, sizeof(*handles));
      assert(handles);
      double start = now();
      for (unsigned i = 0; i < count; ++i) {
         assert(create(&handles[i]) == VK_SUCCESS);
         unsigned expected = i + 1;
         if (expected >= HK_GEN1_AS_TABLE_AS_SLOTS)
            expected += HK_GEN1_RQ_PHYSICAL_SLOTS;
         assert(id(handles[i]) == expected);
      }
      double created = now();
      for (unsigned i = 0; i < count; ++i)
         destroy(handles[i]);
      double destroyed = now();
      unsigned bound = util_dynarray_num_elements(&dev.gen1_ases, struct hk_gen1_as_slot);
      for (unsigned i = 0; i < count; ++i)
         assert(create(&handles[i]) == VK_SUCCESS);
      assert(util_dynarray_num_elements(&dev.gen1_ases, struct hk_gen1_as_slot) == bound);
      /* A permutation crosses the reserved gap and all 16-bit boundaries. */
      for (unsigned i = 0; i < count; ++i)
         destroy(handles[(uint64_t)i * (count == 65536 ? 13 : 2) % count]);
      assert(dev.gen1_as_ids->entries == 0);
      printf("PASS: %u objects create %.3fs, ascending teardown %.3fs, complete ID reuse, permuted teardown\n",count,created-start,destroyed-created);
      free(handles);
      if (iteration == 0) {
         util_dynarray_clear(&dev.gen1_ases);
         dev.gen1_as_free_head = UINT32_MAX;
      }
   }
   windows();
   util_dynarray_foreach(&dev.gen1_as_windows, struct hk_gen1_as_window, win) {
      free_fake_bo(win->trampolines);
      free(win->va);
   }
   util_dynarray_fini(&dev.gen1_as_windows);
   free_fake_bo(dev.gen1_as_window_roots);
   finish_table(&dev.gen1_as_address_pages);
   finish_table(&dev.gen1_as_table_pages);
   finish_table(&dev.gen1_as_metadata_pages);
   _mesa_hash_table_destroy(dev.gen1_as_ids, NULL);
   util_dynarray_fini(&dev.gen1_ases);
   simple_mtx_destroy(&dev.gen1_as_lock);
   vk_object_base_finish(&dev.vk.base);
   vk_object_base_finish(&pdev.vk.base);
   vk_object_base_finish(&instance.base);
   return 0;
}

/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG

#include "hk_entrypoints.h"
#include "hk_instance.h"
#include "hk_physical_device.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MIB(x) ((uint64_t)(x) * 1024ull * 1024ull)
#define GIB(x) ((uint64_t)(x) * 1024ull * 1024ull * 1024ull)

extern const agx_device_ops_t agx_device_drm_ops;

__typeof__(drmIoctl) __wrap_drmIoctl;
__typeof__(os_get_available_system_memory) __wrap_os_get_available_system_memory;
__typeof__(os_get_total_physical_memory) __wrap_os_get_total_physical_memory;

static uint64_t fake_total_ram_B = MIB(63742);
static uint64_t fake_avail_ram_B = MIB(48000);
static bool fake_meminfo_ok = true;

static size_t fake_kernel_params_bytes = sizeof(struct drm_asahi_params_global);
static struct drm_asahi_params_global fake_kernel_params;

bool
__wrap_os_get_total_physical_memory(uint64_t *size)
{
   if (!fake_meminfo_ok)
      return false;
   *size = fake_total_ram_B;
   return true;
}

bool
__wrap_os_get_available_system_memory(uint64_t *size)
{
   if (!fake_meminfo_ok)
      return false;
   *size = fake_avail_ram_B;
   return true;
}

int
__wrap_drmIoctl(int fd, unsigned long request, void *arg)
{
   (void)fd;
   assert(request == DRM_IOCTL_ASAHI_GET_PARAMS);

   /* Mirror Linux File::get_params on _IOW DRM_IOCTL_ASAHI_GET_PARAMS: write
    * min(kernel struct size, user size) to pointer and leave gp->size
    * untouched.
    */
   struct drm_asahi_get_params *gp = arg;
   assert(gp->param_group == 0);
   size_t copy_bytes = MIN2((size_t)gp->size, fake_kernel_params_bytes);
   memcpy((void *)(uintptr_t)gp->pointer, &fake_kernel_params, copy_bytes);
   return 0;
}

/* Match drivers/gpu/drm/asahi/memacct.rs::pct_of(). */
static uint64_t
kernel_pct_of(uint64_t total, uint32_t pct)
{
   if (pct == 0 || pct >= 100)
      return 0;
   return (total / 100) * pct + ((total % 100) * pct) / 100;
}

static bool
near_f(float a, float b, float eps)
{
   return fabsf(a - b) <= eps;
}

static void
test_get_params_tail_detection(void)
{
   struct agx_device dev = {
      .fd = 42,
      .ops = agx_device_drm_ops,
   };
   const size_t full_size =
      offsetof(struct drm_asahi_params_global, client_memory_limit) +
      sizeof(dev.params.client_memory_limit);
   const size_t legacy_size =
      offsetof(struct drm_asahi_params_global, memory_limit);

   /* Legacy 600-byte kernel leaves the tail untouched and does not rewrite
    * get_param.size; agx_get_params must zero the tail and report 600 bytes.
    */
   memset(&fake_kernel_params, 0, sizeof(fake_kernel_params));
   fake_kernel_params.gpu_generation = 16;
   fake_kernel_params_bytes = legacy_size;
   ssize_t got = dev.ops.get_params(&dev, &dev.params, sizeof(dev.params));
   assert((size_t)got == legacy_size);
   assert(dev.params.memory_limit == 0);
   assert(dev.params.client_memory_limit == 0);
   assert(!(!dev.is_virtio && (size_t)got >= full_size));

   /* Current 616-byte Aurora kernel with driver limits disabled writes (0, 0)
    * over the capacity tail; agx_get_params must report all 616 bytes.
    */
   fake_kernel_params_bytes = full_size;
   fake_kernel_params.memory_limit = 0;
   fake_kernel_params.client_memory_limit = 0;
   got = dev.ops.get_params(&dev, &dev.params, sizeof(dev.params));
   assert((size_t)got == full_size);
   assert(dev.params.memory_limit == 0);
   assert(dev.params.client_memory_limit == 0);
   assert(!dev.is_virtio && (size_t)got >= full_size);

   /* Current 616-byte Aurora kernel with active device and client limits. */
   fake_kernel_params.memory_limit = kernel_pct_of(MIB(63742), 67);
   fake_kernel_params.client_memory_limit = kernel_pct_of(MIB(63742), 50);
   got = dev.ops.get_params(&dev, &dev.params, sizeof(dev.params));
   assert((size_t)got == full_size);
   assert(dev.params.memory_limit == fake_kernel_params.memory_limit);
   assert(dev.params.client_memory_limit ==
          fake_kernel_params.client_memory_limit);
}

static void
test_resolve_uma_sysmem_heap(void)
{
   float pct = -1.0f;

   /* Zero RAM fails cleanly without clobbering percent_out. */
   assert(hk_resolve_uma_sysmem_heap(0, 0.0f, true, GIB(4), GIB(4), &pct) == 0);
   assert(pct == -1.0f);

   /* Legacy kernel / virtio (has_memory_limits = false) preserves the 50%
    * fallback across UMA tiers even if struct fields are non-zero.
    */
   const uint64_t tiers[] = {
      MIB(7400), GIB(8), MIB(15800), MIB(31800), MIB(63742), GIB(128),
   };
   for (unsigned i = 0; i < ARRAY_SIZE(tiers); i++) {
      uint64_t ram = tiers[i];
      pct = 0.0f;
      uint64_t heap = hk_resolve_uma_sysmem_heap(ram, 0.0f, false, 0, 0, &pct);
      assert(heap == ROUND_DOWN_TO(ram / 2, MIB(1)));
      assert(near_f(pct, 0.5f, 0.001f));
   }
   assert(hk_resolve_uma_sysmem_heap(GIB(64), 0.0f, false, GIB(48), GIB(48),
                                     &pct) == GIB(32));
   assert(pct == 0.5f);

   /* Default Aurora kernel (mem_limit_pct = 67, client_mem_limit_pct = 50)
    * sizes the heap to the 50% per-client admission limit.
    */
   for (unsigned i = 0; i < ARRAY_SIZE(tiers); i++) {
      uint64_t ram = tiers[i];
      uint64_t dev_lim = kernel_pct_of(ram, 67);
      uint64_t cli_lim = kernel_pct_of(ram, 50);
      pct = 0.0f;
      uint64_t heap =
         hk_resolve_uma_sysmem_heap(ram, 0.0f, true, dev_lim, cli_lim, &pct);
      assert(heap == ROUND_DOWN_TO(cli_lim, MIB(1)));
      assert(heap <= cli_lim && heap <= dev_lim);
      assert(near_f(pct, 0.5f, 0.001f));
   }

   /* Raised compute limits on 16 GB, 32 GB, and 64 GB UMA Macs
    * (mem_limit_pct = 85, client_mem_limit_pct = 82 -> 82%).
    */
   const uint64_t compute_tiers[] = {MIB(15800), MIB(31800), MIB(63742)};
   for (unsigned i = 0; i < ARRAY_SIZE(compute_tiers); i++) {
      uint64_t ram = compute_tiers[i];
      uint64_t dev_lim = kernel_pct_of(ram, 85);
      uint64_t cli_lim = kernel_pct_of(ram, 82);
      pct = 0.0f;
      uint64_t heap =
         hk_resolve_uma_sysmem_heap(ram, 0.0f, true, dev_lim, cli_lim, &pct);
      assert(heap == ROUND_DOWN_TO(cli_lim, MIB(1)));
      assert(heap <= cli_lim && heap <= dev_lim);
      assert(near_f(pct, 0.82f, 0.002f));
   }

   /* Single-sided limits: client_memory_limit = 0 with memory_limit = 80%,
    * memory_limit = 0 with client_memory_limit = 82%, and client > device
    * clamped to device (90% vs 67% -> 67%).
    */
   {
      const uint64_t ram = MIB(63742);
      uint64_t lim80 = kernel_pct_of(ram, 80);
      uint64_t heap =
         hk_resolve_uma_sysmem_heap(ram, 0.0f, true, lim80, 0, &pct);
      assert(heap == ROUND_DOWN_TO(lim80, MIB(1)));
      assert(near_f(pct, 0.80f, 0.002f));

      uint64_t lim82 = kernel_pct_of(ram, 82);
      heap = hk_resolve_uma_sysmem_heap(ram, 0.0f, true, 0, lim82, &pct);
      assert(heap == ROUND_DOWN_TO(lim82, MIB(1)));
      assert(near_f(pct, 0.82f, 0.002f));

      uint64_t lim67 = kernel_pct_of(ram, 67);
      uint64_t lim90 = kernel_pct_of(ram, 90);
      heap = hk_resolve_uma_sysmem_heap(ram, 0.0f, true, lim67, lim90, &pct);
      assert(heap == ROUND_DOWN_TO(lim67, MIB(1)));
      assert(near_f(pct, 0.67f, 0.002f));
   }

   /* Disabled driver limits on a 616-byte Aurora kernel (has_memory_limits =
    * true, limits = 0, 0) keep Borealis's 50% default cap to avoid swapping
    * unless explicitly raised via driconf.
    */
   for (unsigned i = 0; i < ARRAY_SIZE(tiers); i++) {
      uint64_t ram = tiers[i];
      pct = 0.0f;
      uint64_t heap = hk_resolve_uma_sysmem_heap(ram, 0.0f, true, 0, 0, &pct);
      assert(heap == ROUND_DOWN_TO(ram / 2, MIB(1)));
      assert(near_f(pct, 0.5f, 0.001f));
   }
   {
      const uint64_t ram = MIB(63742);
      pct = 0.0f;
      uint64_t heap = hk_resolve_uma_sysmem_heap(ram, 0.0f, true, 0, 0, &pct);
      assert(heap == ROUND_DOWN_TO(ram / 2, MIB(1)));
      assert(pct == 0.5f);
   }

   /* Explicit driconf heap_memory_percent overrides:
    * - Lower than kernel limit (0.25 with 50% kernel limit) -> 25%.
    * - Higher than kernel limit (0.80 / 0.90 with 67% / 50% kernel limits) ->
    *   clamped to 50% so VkMemoryHeap::size never exceeds kernel admission.
    * - Higher overrides with disabled driver limits (0, 0) -> raises the heap
    *   above the 50% default cap (80%, 90%, 100%).
    */
   {
      const uint64_t ram = GIB(64);
      const uint64_t dev_lim = kernel_pct_of(ram, 67);
      const uint64_t cli_lim = kernel_pct_of(ram, 50);

      uint64_t heap_low =
         hk_resolve_uma_sysmem_heap(ram, 0.25f, true, dev_lim, cli_lim, &pct);
      assert(heap_low == GIB(16));
      assert(pct == 0.25f);

      uint64_t heap_clamped_80 =
         hk_resolve_uma_sysmem_heap(ram, 0.80f, true, dev_lim, cli_lim, &pct);
      assert(heap_clamped_80 == GIB(32));
      assert(pct == 0.5f);

      uint64_t heap_clamped_90 =
         hk_resolve_uma_sysmem_heap(ram, 0.90f, true, dev_lim, cli_lim, &pct);
      assert(heap_clamped_90 == GIB(32));
      assert(pct == 0.5f);

      uint64_t heap_unlim_80 =
         hk_resolve_uma_sysmem_heap(ram, 0.80f, true, 0, 0, &pct);
      assert(heap_unlim_80 ==
             ROUND_DOWN_TO((uint64_t)((double)ram * 0.80f), MIB(1)));
      assert(near_f(pct, 0.80f, 0.001f));

      uint64_t heap_unlim_90 =
         hk_resolve_uma_sysmem_heap(ram, 0.90f, true, 0, 0, &pct);
      assert(heap_unlim_90 ==
             ROUND_DOWN_TO((uint64_t)((double)ram * 0.90f), MIB(1)));
      assert(near_f(pct, 0.90f, 0.001f));

      uint64_t heap_unlim_100 =
         hk_resolve_uma_sysmem_heap(ram, 1.0f, true, 0, 0, &pct);
      assert(heap_unlim_100 == ROUND_DOWN_TO(ram, MIB(1)));
      assert(pct == 1.0f);
   }
}

static void
init_mock_pdev(struct hk_physical_device *pdev, struct hk_instance *instance,
               struct hk_heap_usage *usage)
{
   memset(pdev, 0, sizeof(*pdev));
   memset(usage, 0, sizeof(*usage));
   vk_object_base_init(NULL, &pdev->vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   pdev->vk.instance = &instance->vk;
   pdev->heap_usage = usage;
   pdev->mem_heap_count = 1;
   pdev->mem_heaps[0] = (struct hk_memory_heap){
      .used = &usage->used[0],
      .cached = &usage->cached[0],
      .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
   };
}

static void
test_memory_budget_multi_pdev(void)
{
   struct hk_instance instance = {0};
   vk_object_base_init(NULL, &instance.vk.base, VK_OBJECT_TYPE_INSTANCE);
   instance.drirc.misc.heap_memory_percent = OS_GPU_HEAP_SIZE_HEURISTIC;

   fake_total_ram_B = GIB(64);
   fake_avail_ram_B = GIB(48);
   fake_meminfo_ok = true;

   /* Two physical devices under one VkInstance: pdev_a has 82% kernel limits,
    * pdev_b is a legacy/virtio device without memory limits (50% fallback).
    * Resolving pdev_a must not mutate instance.drirc.misc.heap_memory_percent
    * or corrupt pdev_b, and VK_EXT_memory_budget must report non-zero scaled
    * budgets <= heap.size on both devices.
    */
   struct hk_physical_device pdev_a, pdev_b;
   struct hk_heap_usage usage_a, usage_b;
   init_mock_pdev(&pdev_a, &instance, &usage_a);
   init_mock_pdev(&pdev_b, &instance, &usage_b);

   pdev_a.dev.has_memory_limits = true;
   pdev_a.dev.params.memory_limit = kernel_pct_of(fake_total_ram_B, 85);
   pdev_a.dev.params.client_memory_limit = kernel_pct_of(fake_total_ram_B, 82);

   pdev_b.dev.has_memory_limits = false;

   pdev_a.mem_heaps[0].size = hk_get_sysmem_heap_size(&pdev_a);
   assert(instance.drirc.misc.heap_memory_percent ==
          OS_GPU_HEAP_SIZE_HEURISTIC);
   pdev_b.mem_heaps[0].size = hk_get_sysmem_heap_size(&pdev_b);
   assert(instance.drirc.misc.heap_memory_percent ==
          OS_GPU_HEAP_SIZE_HEURISTIC);

   assert(pdev_a.mem_heaps[0].size ==
          ROUND_DOWN_TO(pdev_a.dev.params.client_memory_limit, MIB(1)));
   assert(pdev_b.mem_heaps[0].size == GIB(32));
   assert(near_f(pdev_a.sysmem_heap_percent, 0.82f, 0.002f));
   assert(pdev_b.sysmem_heap_percent == 0.5f);

   usage_a.used[0] = GIB(2);
   usage_b.used[0] = GIB(1);

   for (unsigned iter = 0; iter < 2; iter++) {
      VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_a = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
      };
      VkPhysicalDeviceMemoryProperties2 props_a = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
         .pNext = &budget_a,
      };
      hk_GetPhysicalDeviceMemoryProperties2(
         hk_physical_device_to_handle(&pdev_a), &props_a);

      assert(budget_a.heapUsage[0] == GIB(2));
      assert(budget_a.heapBudget[0] > 0);
      assert(budget_a.heapBudget[0] <= props_a.memoryProperties.memoryHeaps[0].size);
      uint64_t expected_a = vk_physical_device_heap_budget(
         fake_avail_ram_B, 0.9f * pdev_a.sysmem_heap_percent,
         pdev_a.mem_heaps[0].size, usage_a.used[0]);
      assert(budget_a.heapBudget[0] == expected_a);

      VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_b = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
      };
      VkPhysicalDeviceMemoryProperties2 props_b = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
         .pNext = &budget_b,
      };
      hk_GetPhysicalDeviceMemoryProperties2(
         hk_physical_device_to_handle(&pdev_b), &props_b);

      assert(budget_b.heapUsage[0] == GIB(1));
      assert(budget_b.heapBudget[0] > 0);
      assert(budget_b.heapBudget[0] <= props_b.memoryProperties.memoryHeaps[0].size);
      uint64_t expected_b = vk_physical_device_heap_budget(
         fake_avail_ram_B, 0.9f * 0.5f, pdev_b.mem_heaps[0].size,
         usage_b.used[0]);
      assert(budget_b.heapBudget[0] == expected_b);
      assert(budget_a.heapBudget[0] > budget_b.heapBudget[0]);
   }

   /* HK_SYSMEM override also populates sysmem_heap_percent so the fallback
    * branch when used >= sysmem still reports a non-zero budget.
    */
   pdev_b.sysmem = GIB(8);
   pdev_b.mem_heaps[0].size = hk_get_sysmem_heap_size(&pdev_b);
   assert(pdev_b.mem_heaps[0].size == GIB(8));
   assert(pdev_b.sysmem_heap_percent == 0.125f);
   usage_b.used[0] = GIB(8);
   VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_sysmem = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
   };
   VkPhysicalDeviceMemoryProperties2 props_sysmem = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
      .pNext = &budget_sysmem,
   };
   hk_GetPhysicalDeviceMemoryProperties2(
      hk_physical_device_to_handle(&pdev_b), &props_sysmem);
   assert(budget_sysmem.heapBudget[0] == GIB(8));

   vk_object_base_finish(&pdev_b.vk.base);
   vk_object_base_finish(&pdev_a.vk.base);
   vk_object_base_finish(&instance.vk.base);
}

int
main(void)
{
   test_get_params_tail_detection();
   test_resolve_uma_sysmem_heap();
   test_memory_budget_multi_pdev();

   puts("uma heap sizing: ok");
   return 0;
}

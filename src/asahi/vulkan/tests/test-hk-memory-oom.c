/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_memory_budget.h"
#include "hk_wsi.h"
#include "vk_alloc.h"
#include "wsi_common_private.h"
#include "wsi_common_entrypoints.h"
#include "drm-uapi/drm_fourcc.h"
#include "util/os_misc.h"

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#define MIB (UINT64_C(1024) * 1024)
static const uint64_t limit = 2048 * MIB, host = 384 * MIB;
static uint64_t pinned, cached, peak, sibling_snapshot;
static unsigned bo_calls, kernel_oom_attempts, trims, budget_queries;
static unsigned passing, host_live;
static bool bind_fails, bo_fails, image_bind_fails, export_fails, limited = true;
static unsigned images_live, successes, exported_count;
static int exported_fds[4];
static struct agx_bo *live_bos[16];
static uint64_t bo_sizes[16];

__typeof__(agx_bo_bind) __wrap_agx_bo_bind;
__typeof__(agx_bo_cache_evict_all) __wrap_agx_bo_cache_evict_all;
__typeof__(agx_bo_cache_evict_all) __real_agx_bo_cache_evict_all;
__typeof__(agx_bo_export) __wrap_agx_bo_export;
__typeof__(agx_bo_import) __wrap_agx_bo_import;
__typeof__(hk_get_cgroup_memory_available) __wrap_hk_get_cgroup_memory_available;
__typeof__(__vk_errorf) __wrap___vk_errorf;
__typeof__(os_get_available_system_memory) __wrap_os_get_available_system_memory;
__typeof__(os_get_total_physical_memory) __wrap_os_get_total_physical_memory;

/* The reserve check reads real RAM in production. Fixed values keep this test
 * independent of the machine it runs on. */
static uint64_t fake_available = UINT64_C(64) * 1024 * MIB;
static uint64_t fake_total = UINT64_C(32) * 1024 * MIB;
static bool meminfo_readable = true;

bool
__wrap_os_get_available_system_memory(uint64_t *size)
{
   *size = fake_available;
   return meminfo_readable;
}

bool
__wrap_os_get_total_physical_memory(uint64_t *size)
{
   *size = fake_total;
   return true;
}

/* Mock only backend allocation/close and VM_BIND. BO cache insertion, fetch,
 * reference counting, VA release and WSI object cleanup use production code. */
static struct agx_bo *
mock_bo_alloc(struct agx_device *dev, size_t size, size_t align,
              enum agx_bo_flags flags)
{
   ++bo_calls;
   if (limited && size > limit - host - pinned - cached) {
      ++kernel_oom_attempts;
      return NULL;
   }
   if (bo_fails)
      return NULL;
   struct agx_bo *bo = calloc(1, sizeof(*bo));
   assert(bo);
   unsigned handle;
   for (handle = 1; handle < ARRAY_SIZE(live_bos); ++handle)
      if (!live_bos[handle]) break;
   assert(handle < ARRAY_SIZE(live_bos));
   live_bos[handle] = bo;
   bo_sizes[handle] = size;
   bo->dev = dev;
   bo->handle = bo->uapi_handle = handle;
   bo->prime_fd = -1;
   bo->size = size;
   bo->align = align;
   bo->flags = flags;
   bo->va = agx_va_alloc(dev, size, align, 0, 0);
   assert(bo->va);
   pinned += size;
   if (pinned > peak)
      peak = pinned;
   return bo;
}

static void
mock_bo_free(struct agx_device *dev, uint32_t handle)
{
   assert(handle && handle < ARRAY_SIZE(live_bos) && live_bos[handle]);
   assert(pinned >= bo_sizes[handle]);
   pinned -= bo_sizes[handle];
   free(live_bos[handle]);
   live_bos[handle] = NULL;
}

int
__wrap_agx_bo_bind(struct agx_device *dev, struct agx_bo *bo, uint64_t address,
                   size_t size, uint64_t offset, uint32_t flags)
{
   return bind_fails ? -1 : 0;
}

void
__wrap_agx_bo_cache_evict_all(struct agx_device *dev)
{
   ++trims;
   cached = 0;
   __real_agx_bo_cache_evict_all(dev);
}

int
__wrap_agx_bo_export(struct agx_device *dev, struct agx_bo *bo)
{
   return -1;
}

/* An imported object of import_size bytes. */
static uint64_t import_size;

struct agx_bo *
__wrap_agx_bo_import(struct agx_device *dev, int fd)
{
   struct agx_bo *bo = mock_bo_alloc(dev, import_size, AIL_PAGESIZE,
                                     AGX_BO_SHARED | AGX_BO_SHAREABLE);
   if (bo)
      p_atomic_set(&bo->refcnt, 1);
   return bo;
}

bool
__wrap_hk_get_cgroup_memory_available(const char *cgroup, const char *mountinfo,
                                     uint64_t *available)
{
   ++budget_queries;
   uint64_t used = host + pinned + cached + sibling_snapshot;
   *available = used < limit ? limit - used : 0;
   return limited;
}

VkResult
__wrap___vk_errorf(const void *object, VkResult result,
                   const char *file, int line, const char *format, ...)
{
   return result;
}

static void * VKAPI_PTR
allocate(void *data, size_t size, size_t alignment, VkSystemAllocationScope scope)
{
   if (!passing)
      return NULL;
   --passing;
   void *ptr = vk_default_allocator()->pfnAllocation(NULL, size, alignment, scope);
   if (ptr)
      ++host_live;
   return ptr;
}

static void VKAPI_PTR
release(void *data, void *ptr)
{
   if (ptr) {
      assert(host_live);
      --host_live;
      vk_default_allocator()->pfnFree(NULL, ptr);
   }
}

static VkAllocationCallbacks allocator;

struct test_image {
   uint64_t size;
};

static VkResult VKAPI_PTR
create_image(VkDevice device, const VkImageCreateInfo *info,
             const VkAllocationCallbacks *callbacks, VkImage *image)
{
   struct test_image *test = vk_alloc(callbacks, sizeof(*test), 8,
                                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!test)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   test->size = (uint64_t)info->extent.width * info->extent.height * 4;
   *image = (VkImage)(uintptr_t)test;
   ++images_live;
   return VK_SUCCESS;
}

static void VKAPI_PTR
destroy_image(VkDevice device, VkImage image,
              const VkAllocationCallbacks *callbacks)
{
   if (image) {
      assert(images_live);
      --images_live;
      vk_free(callbacks, (void *)(uintptr_t)image);
   }
}

static void VKAPI_PTR
image_requirements(VkDevice device, VkImage image, VkMemoryRequirements *req)
{
   const struct test_image *test = (void *)(uintptr_t)image;
   *req = (VkMemoryRequirements){
      .size = test->size, .alignment = 16384, .memoryTypeBits = 1,
   };
}

static VkResult VKAPI_PTR
bind_image(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize offset)
{
   return image_bind_fails ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}

static VkResult VKAPI_PTR
memory_fd(VkDevice device, const VkMemoryGetFdInfoKHR *info, int *fd)
{
   *fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
   assert(*fd >= 0);
   assert(exported_count < 4);
   exported_fds[exported_count++] = *fd;
   return export_fails ? VK_ERROR_OUT_OF_HOST_MEMORY : VK_SUCCESS;
}

static void VKAPI_PTR
format_properties(VkPhysicalDevice physical, VkFormat format, VkFormatProperties2 *props)
{
   VkDrmFormatModifierPropertiesListEXT *list = props->pNext;
   list->drmFormatModifierCount = 1;
   if (list->pDrmFormatModifierProperties) {
      list->pDrmFormatModifierProperties[0] = (VkDrmFormatModifierPropertiesEXT){
         .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
         .drmFormatModifierPlaneCount = 1,
      };
   }
}

static VkResult VKAPI_PTR
image_format_properties(VkPhysicalDevice physical,
                        const VkPhysicalDeviceImageFormatInfo2 *info,
                        VkImageFormatProperties2 *props)
{
   props->imageFormatProperties.maxExtent = (VkExtent3D){16384, 16384, 1};
   return VK_SUCCESS;
}

static VkResult VKAPI_PTR
image_modifier(VkDevice device, VkImage image, VkImageDrmFormatModifierPropertiesEXT *props)
{
   props->drmFormatModifier = DRM_FORMAT_MOD_LINEAR;
   return VK_SUCCESS;
}

static void VKAPI_PTR
image_layout(VkDevice device, VkImage image, const VkImageSubresource *subresource,
             VkSubresourceLayout *layout)
{
   const struct test_image *test = (void *)(uintptr_t)image;
   *layout = (VkSubresourceLayout){.size = test->size};
}

#define NULL_DESTROY(name, type) \
static void VKAPI_PTR name(VkDevice device, type handle, \
                           const VkAllocationCallbacks *callbacks) \
{ assert(handle == VK_NULL_HANDLE); }
NULL_DESTROY(destroy_query, VkQueryPool)
NULL_DESTROY(destroy_buffer, VkBuffer)
NULL_DESTROY(destroy_semaphore, VkSemaphore)
NULL_DESTROY(destroy_fence, VkFence)

static void
init_wsi(struct hk_device *dev)
{
   struct hk_physical_device *pdev = hk_device_physical(dev);
   struct wsi_device *wsi = &pdev->wsi_device;
   pdev->vk.wsi_device = wsi;
   *wsi = (struct wsi_device){
      .pdevice = hk_physical_device_to_handle(pdev),
      .supports_modifiers = true,
      .override_present_mode = VK_PRESENT_MODE_MAX_ENUM_KHR,
      .CreateImage = create_image, .DestroyImage = destroy_image,
      .GetImageMemoryRequirements = image_requirements,
      .AllocateMemory = hk_wsi_allocate_memory, .FreeMemory = hk_FreeMemory,
      .BindImageMemory = bind_image, .GetMemoryFdKHR = memory_fd,
      .GetPhysicalDeviceFormatProperties2 = format_properties,
      .GetPhysicalDeviceImageFormatProperties2 = image_format_properties,
      .GetImageDrmFormatModifierPropertiesEXT = image_modifier,
      .GetImageSubresourceLayout = image_layout,
      .DestroyQueryPool = destroy_query, .DestroyBuffer = destroy_buffer,
      .DestroySemaphore = destroy_semaphore, .DestroyFence = destroy_fence,
   };
   wsi->memory_props.memoryTypeCount = 1;
   wsi->memory_props.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
   assert(wsi_headless_init_wsi(wsi, vk_default_allocator(), wsi->pdevice) == VK_SUCCESS);
}

static VkResult
alloc(struct hk_device *dev, uint64_t size, VkDeviceMemory *memory)
{
   const VkMemoryAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = size,
      .memoryTypeIndex = 0,
   };
   return hk_AllocateMemory(hk_device_to_handle(dev), &info, &allocator, memory);
}

static void
swapchain_attempt(struct hk_device *dev, uint64_t image_size)
{
   exported_count = 0;
   VkIcdSurfaceHeadless surface = {.base.platform = VK_ICD_WSI_PLATFORM_HEADLESS};
   const VkSwapchainCreateInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .surface = (VkSurfaceKHR)(uintptr_t)&surface,
      .minImageCount = 4,
      .imageFormat = VK_FORMAT_R8G8B8A8_UNORM,
      .imageExtent = {image_size == 1024 * MIB ? 16384 : 8192,
                     image_size == 1024 * MIB ? 16384 : 8192},
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .presentMode = VK_PRESENT_MODE_FIFO_KHR,
   };
   VkSwapchainKHR swapchain = VK_NULL_HANDLE;
   VkResult result = wsi_CreateSwapchainKHR(hk_device_to_handle(dev), &info,
                                            &allocator, &swapchain);
   if (image_size == 1024 * MIB)
      assert(result != VK_SUCCESS);
   assert(result == VK_SUCCESS || result == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
          result == VK_ERROR_OUT_OF_HOST_MEMORY);
   if (result == VK_SUCCESS) {
      ++successes;
      wsi_DestroySwapchainKHR(hk_device_to_handle(dev), swapchain, &allocator);
   }
   assert(!pinned && !host_live && !images_live);
   assert(!dev->dev.bo_cache.size);
   for (unsigned i = 0; i < exported_count; ++i)
      assert(fcntl(exported_fds[i], F_GETFD) == -1 && errno == EBADF);
   assert(!*hk_device_physical(dev)->mem_heaps[0].used);
   assert(!kernel_oom_attempts);
}

int
main(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   assert(dev && pdev);
   dev->vk.alloc = *vk_default_allocator();
   allocator = dev->vk.alloc;
   allocator.pfnAllocation = allocate;
   allocator.pfnFree = release;
   dev->vk.physical = &pdev->vk;
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
   vk_object_base_init(NULL, &pdev->vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   dev->dev.chip = AGX_CHIP_G17P;
   static struct hk_heap_usage heap_usage;
   pdev->heap_usage = &heap_usage;
   pdev->mem_types[0].heapIndex = 0;
   pdev->mem_heaps[0].size = 8 * 1024 * MIB;
   pdev->mem_heaps[0].used = &heap_usage.used[0];
   pdev->mem_heaps[0].cached = &heap_usage.cached[0];
   simple_mtx_init(&dev->dev.bo_map_lock, mtx_plain);
   simple_mtx_init(&dev->dev.bo_cache.lock, mtx_plain);
   simple_mtx_init(&dev->dev.vma_lock, mtx_plain);
   list_inithead(&dev->dev.bo_cache.lru);
   for (unsigned i = 0; i < ARRAY_SIZE(dev->dev.bo_cache.buckets); ++i)
      list_inithead(&dev->dev.bo_cache.buckets[i]);
   util_vma_heap_init(&dev->dev.main_heap, UINT64_C(0x1000000000), UINT64_C(1) << 40);
   dev->dev.ops.bo_alloc = mock_bo_alloc;
   dev->dev.ops.bo_free = mock_bo_free;
   init_wsi(dev);

   /* The CTS tries callback limits 0..1024, including an extent requiring
    * four 1 GiB images. No attempt may enter the kernel above the hard limit,
    * and every completed image must be freed on the next image's failure. */
   for (unsigned count = 0; count <= 1024; ++count) {
      passing = count;
      swapchain_attempt(dev, 1024 * MIB);
      passing = count;
      swapchain_attempt(dev, 256 * MIB);
   }
   assert(peak == 1024 * MIB);
   assert(trims && bo_calls && successes);

   /* Preserve the current error paths after backing was allocated. */
   passing = 1024;
   bind_fails = true;
   unsigned before = bo_calls;
   swapchain_attempt(dev, 256 * MIB);
   assert(bo_calls == before + 1);
   bind_fails = false;
   bo_fails = true;
   passing = 1024;
   before = bo_calls;
   swapchain_attempt(dev, 256 * MIB);
   assert(bo_calls == before + 2);
   bo_fails = false;
   passing = 1024;
   image_bind_fails = true;
   swapchain_attempt(dev, 256 * MIB);
   assert(exported_count == 1);
   image_bind_fails = false;
   passing = 1024;
   export_fails = true;
   swapchain_attempt(dev, 256 * MIB);
   assert(exported_count == 1);
   export_fails = false;

   /* Application allocations are admitted by the allocator/kernel, even when
    * a WSI budget snapshot would refuse them. No cgroup query or trim occurs. */
   sibling_snapshot = 1536 * MIB;
   limited = true;
   passing = 4;
   VkDeviceMemory memory = 0;
   before = budget_queries;
   unsigned old_trims = trims;
   assert(alloc(dev, 1024 * MIB, &memory) == VK_SUCCESS);
   assert(budget_queries == before && trims == old_trims);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   sibling_snapshot = 0;
   assert(!pinned && !host_live);


   passing = 4;
   assert(alloc(dev, 1024 * MIB, &memory) == VK_SUCCESS);
   VkMemoryGetFdInfoKHR get_fd = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      .memory = memory,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   int fd = 88;
   assert(hk_GetMemoryFdKHR(hk_device_to_handle(dev), &get_fd, &fd) ==
          VK_ERROR_OUT_OF_HOST_MEMORY);
   assert(fd == -1);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   assert(!pinned && !host_live);

   /* The real headless path first probes dma-buf sync-file support with a
    * small export allocation lacking WSI_MEMORY_ALLOCATE_INFO_MESA. Its
    * failed export must release backing too, with no cache growth. */
   const VkExportMemoryAllocateInfo export_info = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   const VkMemoryAllocateInfo probe_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &export_info,
      .allocationSize = 4096,
   };
   passing = 4;
   assert(hk_wsi_allocate_memory(hk_device_to_handle(dev), &probe_info,
                                &allocator, &memory) == VK_SUCCESS);
   assert(hk_device_memory_from_handle(memory)->bo->flags & AGX_BO_SHAREABLE);
   get_fd.memory = memory;
   assert(hk_GetMemoryFdKHR(hk_device_to_handle(dev), &get_fd, &fd) ==
          VK_ERROR_OUT_OF_HOST_MEMORY);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   assert(!pinned && !host_live && !dev->dev.bo_cache.size);

   /* Physical heap and alignment failures must also precede BO allocation. */
   pdev->mem_heaps[0].size = 512 * MIB;
   before = bo_calls;
   assert(alloc(dev, 1024 * MIB, &memory) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(alloc(dev, UINT64_MAX, &memory) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(bo_calls == before);
   pdev->mem_heaps[0].size = 8 * 1024 * MIB;

   /* Missing/unlimited cgroups keep normal allocation behavior. */
   limited = false;
   passing = 4;
   assert(alloc(dev, 1024 * MIB, &memory) == VK_SUCCESS);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   assert(!pinned && !host_live && !kernel_oom_attempts);

   passing = 1024;
   /* The free-memory reserve is max(1 GiB, 2% of RAM), 1 GiB for 32 GiB of
    * RAM. One byte short of size plus reserve is refused before any BO is
    * allocated. Exactly enough passes. */
   fake_available = 2048 * MIB - 1;
   before = bo_calls;
   assert(alloc(dev, 1024 * MIB, &memory) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(bo_calls == before);
   fake_available = 2048 * MIB;
   assert(alloc(dev, 1024 * MIB, &memory) == VK_SUCCESS);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);

   /* Above 50 GiB of RAM the reserve is 2% of RAM. */
   fake_total = UINT64_C(128) * 1024 * MIB;
   fake_available = 1024 * MIB + fake_total / 50 - 1;
   assert(alloc(dev, 1024 * MIB, &memory) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   fake_available = 1024 * MIB + fake_total / 50;
   assert(alloc(dev, 1024 * MIB, &memory) == VK_SUCCESS);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   fake_total = UINT64_C(32) * 1024 * MIB;

   /* Allocations below 1 MiB are never refused, 1 MiB is. */
   fake_available = 0;
   assert(alloc(dev, MIB / 2, &memory) == VK_SUCCESS);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   assert(alloc(dev, MIB, &memory) == VK_ERROR_OUT_OF_DEVICE_MEMORY);

   /* An unreadable /proc/meminfo leaves only the heap limit. */
   meminfo_readable = false;
   assert(alloc(dev, 1024 * MIB, &memory) == VK_SUCCESS);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   meminfo_readable = true;

   /* A BO served from the BO cache allocates nothing, so it passes at the
    * reserve. A different size that misses the cache is refused. */
   fake_available = 64 * 1024 * MIB;
   assert(alloc(dev, 128 * MIB, &memory) == VK_SUCCESS);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   fake_available = 1024 * MIB;
   before = bo_calls;
   assert(alloc(dev, 128 * MIB, &memory) == VK_SUCCESS);
   assert(bo_calls == before);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   assert(alloc(dev, 512 * MIB, &memory) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(bo_calls == before);
   agx_bo_cache_evict_all(&dev->dev);

   /* An fd import whose object is smaller than allocationSize is refused,
    * releases the object and leaves the fd with the caller. One at least as
    * large is taken and consumes the fd. */
   VkImportMemoryFdInfoKHR import_info = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
      .fd = open("/dev/null", O_RDONLY | O_CLOEXEC),
   };
   assert(import_info.fd >= 0);
   const VkMemoryAllocateInfo import_alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &import_info,
      .allocationSize = MIB,
   };
   import_size = 16384;
   before = bo_calls;
   assert(hk_AllocateMemory(hk_device_to_handle(dev), &import_alloc, &allocator,
                            &memory) == VK_ERROR_INVALID_EXTERNAL_HANDLE);
   assert(bo_calls == before + 1 && !pinned && !host_live);
   assert(!*hk_device_physical(dev)->mem_heaps[0].used);
   assert(fcntl(import_info.fd, F_GETFD) != -1);
   import_size = MIB - 16384;
   assert(hk_AllocateMemory(hk_device_to_handle(dev), &import_alloc, &allocator,
                            &memory) == VK_ERROR_INVALID_EXTERNAL_HANDLE);
   assert(!pinned && fcntl(import_info.fd, F_GETFD) != -1);
   for (unsigned i = 0; i < 2; ++i) {
      import_size = i ? 4 * MIB : MIB;
      const int fd = import_info.fd;
      assert(hk_AllocateMemory(hk_device_to_handle(dev), &import_alloc,
                               &allocator, &memory) == VK_SUCCESS);
      assert(hk_device_memory_from_handle(memory)->bo->size == import_size);
      assert(fcntl(fd, F_GETFD) == -1 && errno == EBADF);
      hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
      import_info.fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
      assert(import_info.fd >= 0);
   }
   close(import_info.fd);

   assert(!pinned && !host_live && !kernel_oom_attempts);
   wsi_headless_finish_wsi(&pdev->wsi_device, vk_default_allocator());
   vk_object_base_finish(&dev->vk.base);
   vk_object_base_finish(&pdev->vk.base);
   util_vma_heap_finish(&dev->dev.main_heap);
   simple_mtx_destroy(&dev->dev.vma_lock);
   simple_mtx_destroy(&dev->dev.bo_cache.lock);
   simple_mtx_destroy(&dev->dev.bo_map_lock);
   free(dev);
   free(pdev);
   puts("PASS: 2050 swapchain allocation attempts, sync-file probe and fd imports, no kernel OOM entry or retained backing");
   return 0;
}

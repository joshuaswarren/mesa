/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include "hk_cmd_buffer.h"
#include "hk_cmd_pool.h"
#include "hk_descriptor_set.h"
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_entrypoints.h"
#include "hk_instance.h"
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
#include <sys/mman.h>
#include <unistd.h>

#define MIB (UINT64_C(1024) * 1024)
static const uint64_t limit = 2048 * MIB, host = 384 * MIB;
static uint64_t pinned, cached, peak, sibling_snapshot;
static unsigned bo_calls, bind_calls, kernel_oom_attempts, trims, budget_queries;
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
__typeof__(agx_bo_import_host_ptr) __wrap_agx_bo_import_host_ptr;
__typeof__(hk_get_cgroup_memory_available) __wrap_hk_get_cgroup_memory_available;
__typeof__(__vk_errorf) __wrap___vk_errorf;
__typeof__(os_get_available_system_memory) __wrap_os_get_available_system_memory;
__typeof__(os_get_total_physical_memory) __wrap_os_get_total_physical_memory;

/* The reserve check reads real RAM in production. Fixed values keep this test
 * independent of the machine it runs on. */
static uint64_t fake_available = UINT64_C(64) * 1024 * MIB;
static uint64_t fake_total = UINT64_C(32) * 1024 * MIB;
static bool meminfo_readable = true;
/* hk_AllocateMemory() reads /proc/meminfo only when the BO cache misses, and
 * never below 1 MiB. Count the reads to check that. */
static unsigned meminfo_reads;
/* Freeing a BO returns its pages to the system. */
static bool free_returns_available;

bool
__wrap_os_get_available_system_memory(uint64_t *size)
{
   ++meminfo_reads;
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
   if (free_returns_available)
      fake_available += bo_sizes[handle];
   free(live_bos[handle]);
   live_bos[handle] = NULL;
}

static void
mock_bo_mmap_fail(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr,
                  size_t length)
{
   (void)dev;
   (void)fixed_addr;
   (void)length;
   bo->_map = NULL;
   bo->_map_size = 0;
}

int
__wrap_agx_bo_bind(struct agx_device *dev, struct agx_bo *bo, uint64_t address,
                   size_t size, uint64_t offset, uint32_t flags)
{
   ++bind_calls;
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

struct agx_bo *
__wrap_agx_bo_import_host_ptr(struct agx_device *dev, void *ptr, size_t size,
                              enum agx_host_ptr_access access)
{
   enum agx_bo_flags flags = AGX_BO_HOST_PTR | AGX_BO_WRITEBACK | AGX_BO_NO_CACHE;
   if (access == AGX_HOST_PTR_READ_ONLY)
      flags |= AGX_BO_READONLY;
   struct agx_bo *bo = mock_bo_alloc(dev, size, AIL_PAGESIZE, flags);
   if (bo) {
      bo->_map = ptr;
      bo->_map_size = size;
      p_atomic_set(&bo->refcnt, 1);
   }
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
alloc_type(struct hk_device *dev, uint64_t size, uint32_t memory_type_index,
           VkDeviceMemory *memory)
{
   const VkMemoryAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = size,
      .memoryTypeIndex = memory_type_index,
   };
   return hk_AllocateMemory(hk_device_to_handle(dev), &info, &allocator, memory);
}

static VkResult
alloc(struct hk_device *dev, uint64_t size, VkDeviceMemory *memory)
{
   return alloc_type(dev, size, 0, memory);
}

static VkResult
alloc_host(struct hk_device *dev, void *ptr, uint64_t size,
           VkDeviceMemory *memory)
{
   const VkImportMemoryHostPointerInfoEXT host_info = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
      .pHostPointer = ptr,
   };
   const VkMemoryAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &host_info,
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
   struct hk_instance *instance = calloc(1, sizeof(*instance));
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   assert(instance && dev && pdev);
   pdev->vk.instance = &instance->vk;
   dev->vk.alloc = *vk_default_allocator();
   allocator = dev->vk.alloc;
   allocator.pfnAllocation = allocate;
   allocator.pfnFree = release;
   dev->vk.physical = &pdev->vk;
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
   vk_object_base_init(NULL, &pdev->vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   dev->dev.chip = AGX_CHIP_G17P;
   simple_mtx_init(&dev->host_alias.lock, mtx_plain);
   simple_mtx_init(&dev->host_alias.cmd_lock, mtx_plain);
   list_inithead(&dev->host_alias.cmd_buffers);
   u_rwlock_init(&dev->host_alias.patch_lock);
   static struct hk_heap_usage heap_usage;
   pdev->heap_usage = &heap_usage;
   pdev->mem_types[0].heapIndex = 0;
   pdev->mem_heaps[0].size = 8 * 1024 * MIB;
   pdev->mem_heaps[0].used = &heap_usage.used[0];
   pdev->mem_heaps[0].cached = &heap_usage.cached[0];
   simple_mtx_init(&dev->dev.bo_map_lock, mtx_plain);
   simple_mtx_init(&dev->dev.bo_mmap_lock, mtx_plain);
   simple_mtx_init(&dev->dev.bo_cache.lock, mtx_plain);
   simple_mtx_init(&dev->dev.vma_lock, mtx_plain);
   list_inithead(&dev->dev.bo_cache.lru);
   for (unsigned i = 0; i < ARRAY_SIZE(dev->dev.bo_cache.buckets); ++i)
      list_inithead(&dev->dev.bo_cache.buckets[i]);
   util_vma_heap_init(&dev->dev.main_heap, UINT64_C(0x1000000000), UINT64_C(1) << 40);
   dev->dev.ops.bo_alloc = mock_bo_alloc;
   dev->dev.ops.bo_free = mock_bo_free;
   dev->dev.ops.bo_mmap = mock_bo_mmap_fail;
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
   /* Verify export error handling holds on non-G17P chips and virtio too. */
   dev->dev.chip = AGX_CHIP_G17C;
   assert(hk_GetMemoryFdKHR(hk_device_to_handle(dev), &get_fd, &fd) ==
          VK_ERROR_OUT_OF_HOST_MEMORY);
   dev->dev.chip = AGX_CHIP_G13G;
   assert(hk_GetMemoryFdKHR(hk_device_to_handle(dev), &get_fd, &fd) ==
          VK_ERROR_OUT_OF_HOST_MEMORY);
   dev->dev.is_virtio = true;
   assert(hk_GetMemoryFdKHR(hk_device_to_handle(dev), &get_fd, &fd) ==
          VK_ERROR_OUT_OF_HOST_MEMORY);
   dev->dev.is_virtio = false;
   dev->dev.chip = AGX_CHIP_G17P;
   /* Reserve-unmap with a NULL BO map must succeed without calling MAP_FIXED. */
   const VkMemoryUnmapInfoKHR reserve_unmap = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO_KHR,
      .memory = memory,
      .flags = VK_MEMORY_UNMAP_RESERVE_BIT_EXT,
   };
   assert(hk_UnmapMemory2KHR(hk_device_to_handle(dev), &reserve_unmap) ==
          VK_SUCCESS);
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

   /* Allocations below 1 MiB are never refused, 1 MiB is. Below 1 MiB
    * /proc/meminfo is not read. */
   fake_available = 0;
   meminfo_reads = 0;
   assert(alloc(dev, MIB / 2, &memory) == VK_SUCCESS);
   assert(!meminfo_reads);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   /* The freed 512 KiB BO is still cached, so the refusal evicts it and
    * reads /proc/meminfo once more. */
   assert(alloc(dev, MIB, &memory) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(meminfo_reads == 2);
   assert(!dev->dev.bo_cache.size);

   /* An unreadable /proc/meminfo leaves only the heap limit. */
   meminfo_readable = false;
   assert(alloc(dev, 1024 * MIB, &memory) == VK_SUCCESS);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   meminfo_readable = true;

   /* A BO served from the BO cache allocates nothing and reuses its read-only
    * shadow mapping with zero VM_BIND calls, so it passes at the reserve. A
    * different size that misses the cache is refused. */
   fake_available = 64 * 1024 * MIB;
   unsigned binds_before = bind_calls;
   assert(alloc(dev, 128 * MIB, &memory) == VK_SUCCESS);
   assert(bind_calls == binds_before + 1);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   fake_available = 1024 * MIB;
   before = bo_calls;
   /* A cache hit does not read /proc/meminfo; a miss reads it once. */
   meminfo_reads = 0;
   binds_before = bind_calls;
   assert(alloc(dev, 128 * MIB, &memory) == VK_SUCCESS);
   assert(bo_calls == before && bind_calls == binds_before);
   assert(!meminfo_reads);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   /* A miss that does not fit evicts the cache, then asks once more. Freeing
    * the cached BO does not raise fake_available here, so it is refused. */
   assert(alloc(dev, 512 * MIB, &memory) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(bo_calls == before && bind_calls == binds_before);
   assert(meminfo_reads == 2);
   assert(!pinned && !dev->dev.bo_cache.size);
   agx_bo_cache_evict_all(&dev->dev);

   /* When freeing the idle cached BOs gives back enough memory, the retry
    * after eviction admits the allocation instead of failing it. */
   fake_available = 64 * 1024 * MIB;
   assert(alloc(dev, 128 * MIB, &memory) == VK_SUCCESS);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   assert(dev->dev.bo_cache.size == 128 * MIB);
   fake_available = 512 * MIB + 1024 * MIB - 1;
   free_returns_available = true;
   meminfo_reads = 0;
   before = bo_calls;
   assert(alloc(dev, 512 * MIB, &memory) == VK_SUCCESS);
   assert(meminfo_reads == 2);
   assert(bo_calls == before + 1);
   assert(!dev->dev.bo_cache.size && pinned == 512 * MIB);
   free_returns_available = false;
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   agx_bo_cache_evict_all(&dev->dev);

   /* Warm BO-cache reuse across 4 KiB..64 MiB must issue zero VM_BINDs,
    * isolate memory types by BO flags, retry after a bind failure, and reset
    * cleanly on cache eviction. */
   fake_available = 64 * 1024 * MIB;
   pdev->mem_types[1].heapIndex = 0;
   pdev->mem_types[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
   static const uint64_t reuse_sizes[] = {
      4096, 65536, MIB, 16 * MIB, 64 * MIB,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(reuse_sizes); ++i) {
      before = bo_calls;
      binds_before = bind_calls;
      assert(alloc(dev, reuse_sizes[i], &memory) == VK_SUCCESS);
      assert(bo_calls == before + 1 && bind_calls == binds_before + 1);
      struct agx_bo *first_bo = hk_device_memory_from_handle(memory)->bo;
      assert(first_bo->ro_shadow_bound);
      hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);

      for (unsigned iter = 0; iter < 3; ++iter) {
         assert(alloc(dev, reuse_sizes[i], &memory) == VK_SUCCESS);
         assert(hk_device_memory_from_handle(memory)->bo == first_bo);
         assert(bo_calls == before + 1 && bind_calls == binds_before + 1);
         hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
      }
   }

   /* HOST_CACHED (AGX_BO_WRITEBACK) must not reuse an uncached BO. */
   before = bo_calls;
   binds_before = bind_calls;
   assert(alloc_type(dev, 16 * MIB, 1, &memory) == VK_SUCCESS);
   assert(bo_calls == before + 1 && bind_calls == binds_before + 1);
   struct agx_bo *wb_bo = hk_device_memory_from_handle(memory)->bo;
   assert(wb_bo->flags & AGX_BO_WRITEBACK);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   assert(alloc_type(dev, 16 * MIB, 1, &memory) == VK_SUCCESS);
   assert(hk_device_memory_from_handle(memory)->bo == wb_bo);
   assert(bo_calls == before + 1 && bind_calls == binds_before + 1);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);

   /* A failed RO shadow bind must not mark the BO bound; the next allocation
    * retries and succeeds. */
   bind_fails = true;
   binds_before = bind_calls;
   assert(alloc(dev, 2 * MIB, &memory) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(bind_calls == binds_before + 1);
   bind_fails = false;
   assert(alloc(dev, 2 * MIB, &memory) == VK_SUCCESS);
   assert(bind_calls == binds_before + 2);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);

   /* Evicting the cache closes every handle; a fresh allocation binds again. */
   agx_bo_cache_evict_all(&dev->dev);
   assert(!pinned && !dev->dev.bo_cache.size);
   before = bo_calls;
   binds_before = bind_calls;
   assert(alloc(dev, 64 * MIB, &memory) == VK_SUCCESS);
   assert(bo_calls == before + 1 && bind_calls == binds_before + 1);
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   agx_bo_cache_evict_all(&dev->dev);
   assert(!pinned && !dev->dev.bo_cache.size);

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

   /* Map-OOM in hk_CreateDescriptorPool must release its BO and host object. */
   const VkDescriptorPoolSize pool_size = {
      .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      .descriptorCount = 16,
   };
   const VkDescriptorPoolCreateInfo dp_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 4,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
   };
   VkDescriptorPool dpool = VK_NULL_HANDLE;
   passing = 16;
   assert(hk_CreateDescriptorPool(hk_device_to_handle(dev), &dp_info,
                                  &allocator, &dpool) ==
          VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(dpool == VK_NULL_HANDLE);
   agx_bo_cache_evict_all(&dev->dev);
   assert(!pinned && !host_live && !dev->dev.bo_cache.size);

   /* Map-OOM in hk_pool_alloc_from_slow large-BO path must not leak or append. */
   struct hk_cmd_pool cmd_pool = {0};
   cmd_pool.vk.base.device = &dev->vk;
   struct hk_cmd_buffer cmd = {0};
   vk_object_base_init(&dev->vk, &cmd.vk.base, VK_OBJECT_TYPE_COMMAND_BUFFER);
   cmd.vk.pool = &cmd_pool.vk;
   util_dynarray_init(&cmd.large_bos, NULL);
   struct agx_ptr slow = hk_pool_alloc_from_slow(
      &cmd, &cmd.uploader.main, HK_CMD_BO_SIZE + 4096, 16, HK_CMD_BO_MAIN);
   assert(slow.cpu == NULL && slow.gpu == 0);
   assert(cmd.vk.record_result == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   assert(util_dynarray_num_elements(&cmd.large_bos, struct agx_bo *) == 0);
   util_dynarray_fini(&cmd.large_bos);
   agx_bo_cache_evict_all(&dev->dev);
   assert(!pinned && !dev->dev.bo_cache.size);

   /* Empty-region hk_CmdCopyBuffer2 and nested empty xfer ranges must not wrap
    * or widen an active transfer range to 0. */
   const VkCopyBufferInfo2 empty_copy = {
      .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
      .regionCount = 0,
   };
   hk_CmdCopyBuffer2(hk_cmd_buffer_to_handle(&cmd), &empty_copy);
   assert(!cmd.sync.xfer_depth && !cmd.sync.xfer_dst.valid);
   hk_sync_xfer_begin_dst(&cmd, 0, 0, false);
   assert(cmd.sync.xfer_depth == 1 && !cmd.sync.xfer_dst.valid);
   hk_sync_xfer_end(&cmd);
   assert(cmd.sync.xfer_depth == 0);
   hk_sync_xfer_begin_dst(&cmd, 0x20000, 0x21000, false);
   hk_sync_xfer_begin_dst(&cmd, 0, 0, true);
   assert(cmd.sync.xfer_depth == 2 && cmd.sync.xfer_dst.valid);
   assert(cmd.sync.xfer_dst.lo == 0x20000 && cmd.sync.xfer_dst.hi == 0x21000);
   assert(!cmd.sync.xfer_dst.device_address);
   hk_sync_xfer_end(&cmd);
   hk_sync_xfer_end(&cmd);
   vk_object_base_finish(&cmd.vk.base);

   /* Host-pointer imports track aliasing per import via /proc/self/maps and
    * range overlap on non-M5 chips and on M5 with host_alias.dynamic, while
    * keeping va_alias_resources == 0 for disjoint private anonymous mappings. */
   const size_t host_sz = 65536;
   void *priv_a = mmap(NULL, host_sz * 2, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   void *shared_a = mmap(NULL, host_sz, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_ANONYMOUS, -1, 0);
   assert(priv_a != MAP_FAILED && shared_a != MAP_FAILED);

   /* G17P: disjoint private-anonymous imports do not raise va_alias_resources;
    * overlapping or shared-mapping imports do, and freeing restores 0. */
   dev->dev.chip = AGX_CHIP_G17P;
   dev->host_alias.dynamic = false;
   dev->may_alias_memory = false;
   assert(!dev->va_alias_resources && !dev->alias_live &&
          !hk_cdm_alias_scope(dev));

   VkDeviceMemory hm0 = 0, hm1 = 0, hm_ov = 0, hm_sh = 0;
   passing = 32;
   assert(alloc_host(dev, priv_a, host_sz, &hm0) == VK_SUCCESS);
   assert(dev->va_alias_resources == 0 && !hk_cdm_alias_scope(dev));
   assert(alloc_host(dev, (char *)priv_a + host_sz, host_sz, &hm1) == VK_SUCCESS);
   assert(dev->va_alias_resources == 0 && !hk_cdm_alias_scope(dev));
   assert(alloc_host(dev, priv_a, host_sz, &hm_ov) == VK_SUCCESS);
   assert(dev->va_alias_resources == 2 && hk_cdm_alias_scope(dev));
   hk_FreeMemory(hk_device_to_handle(dev), hm_ov, &allocator);
   assert(dev->va_alias_resources == 0 && !hk_cdm_alias_scope(dev));
   assert(alloc_host(dev, shared_a, host_sz, &hm_sh) == VK_SUCCESS);
   assert(dev->va_alias_resources == 1 && hk_cdm_alias_scope(dev));
   hk_FreeMemory(hk_device_to_handle(dev), hm_sh, &allocator);
   assert(dev->va_alias_resources == 0 && !hk_cdm_alias_scope(dev));
   hk_FreeMemory(hk_device_to_handle(dev), hm1, &allocator);
   hk_FreeMemory(hk_device_to_handle(dev), hm0, &allocator);
   assert(dev->va_alias_resources == 0 && !pinned && !host_live);

   /* M5 (G17C) with host_alias.dynamic: va_alias_resources and alias_live stay
    * 0 for disjoint private-anonymous imports and track overlaps/shared maps
    * together. */
   dev->dev.chip = AGX_CHIP_G17C;
   dev->host_alias.dynamic = true;
   dev->may_alias_memory = false;
   passing = 32;
   assert(alloc_host(dev, priv_a, host_sz, &hm0) == VK_SUCCESS);
   assert(dev->va_alias_resources == 0 && dev->alias_live == 0 &&
          !hk_cdm_alias_scope(dev));
   assert(alloc_host(dev, priv_a, host_sz, &hm_ov) == VK_SUCCESS);
   assert(dev->va_alias_resources == 2 && dev->alias_live == 2 &&
          hk_cdm_alias_scope(dev));
   hk_FreeMemory(hk_device_to_handle(dev), hm_ov, &allocator);
   assert(dev->va_alias_resources == 0 && dev->alias_live == 0 &&
          !hk_cdm_alias_scope(dev));
   assert(alloc_host(dev, shared_a, host_sz, &hm_sh) == VK_SUCCESS);
   assert(dev->va_alias_resources == 1 && dev->alias_live == 1 &&
          hk_cdm_alias_scope(dev));
   hk_FreeMemory(hk_device_to_handle(dev), hm_sh, &allocator);
   hk_FreeMemory(hk_device_to_handle(dev), hm0, &allocator);
   assert(dev->va_alias_resources == 0 && dev->alias_live == 0 &&
          !pinned && !host_live);

   /* M5 static per-device fallback: writable host imports still count
    * unconditionally in va_alias_resources. */
   dev->host_alias.dynamic = false;
   dev->may_alias_memory = true;
   passing = 16;
   assert(alloc_host(dev, priv_a, host_sz, &hm0) == VK_SUCCESS);
   assert(dev->va_alias_resources == 1 && hk_cdm_alias_scope(dev));
   hk_FreeMemory(hk_device_to_handle(dev), hm0, &allocator);
   assert(dev->va_alias_resources == 0 && !pinned && !host_live);

   /* Reset to G17P and verify fd imports still count unconditionally. */
   dev->dev.chip = AGX_CHIP_G17P;
   dev->may_alias_memory = false;
   import_info.fd = open("/dev/null", O_RDONLY | O_CLOEXEC);
   assert(import_info.fd >= 0);
   import_size = MIB;
   passing = 16;
   assert(hk_AllocateMemory(hk_device_to_handle(dev), &import_alloc, &allocator,
                            &memory) == VK_SUCCESS);
   assert(dev->va_alias_resources == 1 && hk_cdm_alias_scope(dev));
   hk_FreeMemory(hk_device_to_handle(dev), memory, &allocator);
   assert(dev->va_alias_resources == 0 && !hk_cdm_alias_scope(dev));

   munmap(shared_a, host_sz);
   munmap(priv_a, host_sz * 2);

   assert(!pinned && !host_live && !kernel_oom_attempts);
   wsi_headless_finish_wsi(&pdev->wsi_device, vk_default_allocator());
   vk_object_base_finish(&dev->vk.base);
   vk_object_base_finish(&pdev->vk.base);
   hk_host_imports_fini(&dev->host_alias.imports);
   u_rwlock_destroy(&dev->host_alias.patch_lock);
   simple_mtx_destroy(&dev->host_alias.cmd_lock);
   simple_mtx_destroy(&dev->host_alias.lock);
   util_vma_heap_finish(&dev->dev.main_heap);
   simple_mtx_destroy(&dev->dev.vma_lock);
   simple_mtx_destroy(&dev->dev.bo_cache.lock);
   simple_mtx_destroy(&dev->dev.bo_mmap_lock);
   simple_mtx_destroy(&dev->dev.bo_map_lock);
   free(dev);
   free(pdev);
   free(instance);
   puts("PASS: 2050 swapchain allocation attempts, sync-file probe and fd imports, no kernel OOM entry or retained backing");
   return 0;
}

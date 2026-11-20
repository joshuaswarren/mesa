/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Host image copies through memory the application maps, unmaps and maps at
 * a placed address, and from two threads at once. The BO is a memfd, so what
 * a copy wrote can be read back from the file whichever mapping it went
 * through.
 */
#undef NDEBUG
#include "hk_device.h"
#include "hk_device_memory.h"
#include "hk_entrypoints.h"
#include "hk_image.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#define W    64
#define H    64
#define SIZE (4 * AIL_PAGESIZE)

static int backing = -1;
static unsigned mmaps, munmaps;
/* The agx_bo each mapping was made into, for the last two mappings. */
static struct agx_bo *mapped_bo[2];

/* While set, each mapping waits here until two threads are mapping. */
static pthread_barrier_t *mmap_rendezvous;
/* While set, each mapping takes 100 ms. */
static bool slow_mmap;

static void
mock_mmap(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr,
          size_t length)
{
   assert(!bo->_map);
   if (mmap_rendezvous)
      pthread_barrier_wait(mmap_rendezvous);
   if (slow_mmap)
      usleep(100 * 1000);

   bo->_map_size = length ?: bo->size;
   void *map = mmap(fixed_addr, bo->_map_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED | (fixed_addr ? MAP_FIXED : 0), backing, 0);
   assert(map != MAP_FAILED);
   bo->_map = map;
   mapped_bo[p_atomic_inc_return(&mmaps) & 1] = bo;
}

static void
mock_munmap(struct agx_device *dev, struct agx_bo *bo)
{
   assert(!munmap(bo->_map, agx_bo_map_size(bo)));
   p_atomic_inc(&munmaps);
}

__typeof__(__vk_errorf) __wrap___vk_errorf;
VkResult
__wrap___vk_errorf(const void *object, VkResult result, const char *file,
                   int line, const char *format, ...)
{
   return result;
}

static void
init_image(struct hk_image *image)
{
   *image = (struct hk_image){0};
   image->vk.base.type = VK_OBJECT_TYPE_IMAGE;
   image->vk.image_type = VK_IMAGE_TYPE_2D;
   image->vk.format = VK_FORMAT_R8G8B8A8_UNORM;
   image->vk.extent = (VkExtent3D){W, H, 1};
   image->vk.mip_levels = 1;
   image->vk.array_layers = 1;
   image->plane_count = 1;
   image->planes[0].layout = (struct ail_layout){
      .width_px = W,
      .height_px = H,
      .depth_px = 1,
      .sample_count_sa = 1,
      .levels = 1,
      .tiling = AIL_TILING_LINEAR,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   ail_make_miptree(&image->planes[0].layout);
   assert(image->planes[0].layout.size_B <= SIZE / 2);
}

static void
fill(uint32_t *texels, uint32_t seed)
{
   for (unsigned i = 0; i < W * H; ++i)
      texels[i] = seed * 0x9e3779b9u + i;
}

/* The image's texels as the BO holds them, read from the file. */
static void
backing_texels(const struct hk_image *image, uint64_t offset_B,
               uint32_t *texels)
{
   const struct ail_layout *layout = &image->planes[0].layout;
   const uint32_t stride_B = ail_get_linear_stride_B(layout, 0);
   for (unsigned y = 0; y < H; ++y) {
      assert(pread(backing, texels + y * W, W * 4, offset_B + y * stride_B) ==
             W * 4);
   }
}

static void
to_image(VkDevice device, struct hk_image *image, const uint32_t *texels)
{
   const VkMemoryToImageCopyEXT region = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT,
      .pHostPointer = texels,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {W, H, 1},
   };
   const VkCopyMemoryToImageInfoEXT info = {
      .sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT,
      .dstImage = hk_image_to_handle(image),
      .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .regionCount = 1,
      .pRegions = &region,
   };
   assert(hk_CopyMemoryToImageEXT(device, &info) == VK_SUCCESS);
}

static void
from_image(VkDevice device, struct hk_image *image, uint32_t *texels)
{
   const VkImageToMemoryCopyEXT region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_TO_MEMORY_COPY_EXT,
      .pHostPointer = texels,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {W, H, 1},
   };
   const VkCopyImageToMemoryInfoEXT info = {
      .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_MEMORY_INFO_EXT,
      .srcImage = hk_image_to_handle(image),
      .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .regionCount = 1,
      .pRegions = &region,
   };
   assert(hk_CopyImageToMemoryEXT(device, &info) == VK_SUCCESS);
}

static void
image_to_image(VkDevice device, struct hk_image *src, struct hk_image *dst)
{
   const VkImageCopy2 region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .extent = {W, H, 1},
   };
   const VkCopyImageToImageInfoEXT info = {
      .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_IMAGE_INFO_EXT,
      .srcImage = hk_image_to_handle(src),
      .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .dstImage = hk_image_to_handle(dst),
      .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .regionCount = 1,
      .pRegions = &region,
   };
   assert(hk_CopyImageToImageEXT(device, &info) == VK_SUCCESS);
}

static void *
map(VkDevice device, VkDeviceMemory memory, void *placed)
{
   const VkMemoryMapPlacedInfoEXT placed_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT,
      .pPlacedAddress = placed,
   };
   const VkMemoryMapInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_MAP_INFO_KHR,
      .pNext = placed ? &placed_info : NULL,
      .flags = placed ? VK_MEMORY_MAP_PLACED_BIT_EXT : 0,
      .memory = memory,
      .size = VK_WHOLE_SIZE,
   };
   void *data = NULL;
   assert(hk_MapMemory2KHR(device, &info, &data) == VK_SUCCESS);
   assert(data && (!placed || data == placed));
   return data;
}

static void
unmap(VkDevice device, VkDeviceMemory memory, bool reserve)
{
   const VkMemoryUnmapInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO_KHR,
      .flags = reserve ? VK_MEMORY_UNMAP_RESERVE_BIT_EXT : 0,
      .memory = memory,
   };
   assert(hk_UnmapMemory2KHR(device, &info) == VK_SUCCESS);
}

/* Whether [addr, addr + size) is one inaccessible mapping, per
 * /proc/self/maps: what an application's reservation must stay. */
static bool
reserved(const void *addr, size_t size)
{
   FILE *maps = fopen("/proc/self/maps", "r");
   assert(maps);
   char line[512];
   bool found = false;
   while (fgets(line, sizeof(line), maps)) {
      uintptr_t lo, hi;
      char perms[5];
      if (sscanf(line, "%lx-%lx %4s", &lo, &hi, perms) != 3)
         continue;
      if (lo <= (uintptr_t)addr && (uintptr_t)addr + size <= hi) {
         found = !strcmp(perms, "---p");
         break;
      }
   }
   fclose(maps);
   return found;
}

/* Whether [addr, addr + size) is mapped readable and writable, per
 * /proc/self/maps. */
static bool
mapped_rw(const void *addr, size_t size)
{
   FILE *maps = fopen("/proc/self/maps", "r");
   assert(maps);
   char line[512];
   bool found = false;
   while (fgets(line, sizeof(line), maps)) {
      uintptr_t lo, hi;
      char perms[5];
      if (sscanf(line, "%lx-%lx %4s", &lo, &hi, perms) != 3)
         continue;
      if (lo <= (uintptr_t)addr && (uintptr_t)addr + size <= hi) {
         found = !strncmp(perms, "rw", 2);
         break;
      }
   }
   fclose(maps);
   return found;
}

struct copy_job {
   VkDevice device;
   struct hk_image *image;
   uint32_t texels[W * H];
};

static void *
copy_thread(void *data)
{
   struct copy_job *job = data;
   to_image(job->device, job->image, job->texels);
   return NULL;
}

struct map_job {
   pthread_barrier_t *start;
   struct agx_bo *bo;
   void *map;
};

static void *
map_thread(void *data)
{
   struct map_job *job = data;
   pthread_barrier_wait(job->start);
   job->map = agx_bo_map(job->bo);
   return NULL;
}

int
main(void)
{
   backing = memfd_create("hk-host-image-copy", MFD_CLOEXEC);
   assert(backing >= 0 && !ftruncate(backing, SIZE));

   struct hk_device *dev = calloc(1, sizeof(*dev));
   assert(dev);
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
   dev->dev.ops.bo_mmap = mock_mmap;
   dev->dev.ops.bo_munmap = mock_munmap;
   dev->dev.ops.bo_mmap_independent = true;
   simple_mtx_init(&dev->dev.bo_mmap_lock, mtx_plain);
   VkDevice device = hk_device_to_handle(dev);

   struct agx_va va = {.addr = 0x100000000, .size_B = SIZE};
   struct agx_bo bo = {.dev = &dev->dev, .size = SIZE, .va = &va};
   struct hk_device_memory mem = {0};
   mem.vk.base.type = VK_OBJECT_TYPE_DEVICE_MEMORY;
   mem.vk.size = SIZE;
   mem.bo = &bo;
   VkDeviceMemory memory = hk_device_memory_to_handle(&mem);

   /* Two images in the memory, bound while it is mapped. */
   struct hk_image a, b;
   init_image(&a);
   init_image(&b);
   void *data = map(device, memory, NULL);
   memset(data, 0, SIZE);
   const VkBindImageMemoryInfo binds[2] = {
      {.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
       .image = hk_image_to_handle(&a),
       .memory = memory},
      {.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
       .image = hk_image_to_handle(&b),
       .memory = memory,
       .memoryOffset = SIZE / 2},
   };
   assert(hk_BindImageMemory2(device, 2, binds) == VK_SUCCESS);
   /* Binding maps nothing. */
   assert(!mem.copy_map);

   static uint32_t in[W * H], out[W * H];

   /* Unmapped: the first copy maps the memory for the copies, leaves the
    * application's mapping unmapped, and lands in the memory. */
   unmap(device, memory, false);
   assert(!bo._map);
   const unsigned before_copies = mmaps;
   fill(in, 1);
   to_image(device, &a, in);
   assert(mem.copy_map && !bo._map && mmaps == before_copies + 1);
   void *copy_map = mem.copy_map;
   backing_texels(&a, 0, out);
   assert(!memcmp(in, out, sizeof(in)));
   memset(out, 0, sizeof(out));
   from_image(device, &a, out);
   assert(!memcmp(in, out, sizeof(in)));
   image_to_image(device, &a, &b);
   backing_texels(&b, SIZE / 2, out);
   assert(!memcmp(in, out, sizeof(in)));
   /* One mapping serves every copy that follows. */
   from_image(device, &b, out);
   assert(mmaps == before_copies + 1 && mem.copy_map == copy_map);

   /* Mapped at a placed address: the copies keep their mapping, and what
    * they write shows through the application's. */
   void *placed =
      mmap(NULL, SIZE, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
   assert(placed != MAP_FAILED);
   data = map(device, memory, placed);
   assert(bo._map == placed);
   const unsigned placed_mmaps = mmaps;
   fill(in, 2);
   to_image(device, &a, in);
   assert(bo._map == placed && mem.map == placed && mmaps == placed_mmaps);
   assert(mem.copy_map == copy_map && mapped_rw(copy_map, SIZE));
   const uint32_t stride_B = ail_get_linear_stride_B(&a.planes[0].layout, 0);
   for (unsigned y = 0; y < H; ++y)
      assert(!memcmp((char *)placed + y * stride_B, in + y * W, W * 4));
   image_to_image(device, &a, &b);
   backing_texels(&b, SIZE / 2, out);
   assert(!memcmp(in, out, sizeof(in)));

   /* Unmapped with the range kept reserved: the copies leave the reservation
    * inaccessible and the application's mapping unmapped. */
   unmap(device, memory, true);
   assert(!bo._map && reserved(placed, SIZE));
   fill(in, 3);
   to_image(device, &a, in);
   assert(!bo._map && reserved(placed, SIZE) && mem.copy_map == copy_map);
   backing_texels(&a, 0, out);
   assert(!memcmp(in, out, sizeof(in)));

   /* Placed again, then unmapped: neither moves nor removes the copies'
    * mapping, which a copy on another thread could be using meanwhile. */
   data = map(device, memory, placed);
   assert(bo._map == placed && mapped_rw(copy_map, SIZE));
   unmap(device, memory, false);
   assert(!bo._map && mapped_rw(copy_map, SIZE));
   fill(in, 4);
   to_image(device, &a, in);
   backing_texels(&a, 0, out);
   assert(!memcmp(in, out, sizeof(in)));

   assert(munmap(copy_map, SIZE) == 0);

   /* Two threads copy into the two images of memory no copy has mapped yet,
    * and both map it at once: one mapping is kept, the other unmapped, and
    * both copies land in the memory. */
   struct hk_device_memory mem2 = {0};
   mem2.vk.base.type = VK_OBJECT_TYPE_DEVICE_MEMORY;
   mem2.vk.size = SIZE;
   mem2.bo = &bo;
   a.planes[0].mem = &mem2;
   b.planes[0].mem = &mem2;
   pthread_barrier_t rendezvous;
   assert(!pthread_barrier_init(&rendezvous, NULL, 2));
   mmap_rendezvous = &rendezvous;
   static struct copy_job jobs[2];
   pthread_t threads[2];
   const unsigned race_mmaps = mmaps, race_munmaps = munmaps;
   for (unsigned i = 0; i < 2; ++i) {
      jobs[i] = (struct copy_job){.device = device, .image = i ? &b : &a};
      fill(jobs[i].texels, 5 + i);
      assert(!pthread_create(&threads[i], NULL, copy_thread, &jobs[i]));
   }
   for (unsigned i = 0; i < 2; ++i)
      assert(!pthread_join(threads[i], NULL));
   mmap_rendezvous = NULL;
   pthread_barrier_destroy(&rendezvous);
   assert(mmaps == race_mmaps + 2 && munmaps == race_munmaps + 1);
   assert(mem2.copy_map && mapped_rw(mem2.copy_map, SIZE));
   backing_texels(&a, 0, out);
   assert(!memcmp(jobs[0].texels, out, sizeof(out)));
   backing_texels(&b, SIZE / 2, out);
   assert(!memcmp(jobs[1].texels, out, sizeof(out)));
   assert(munmap(mem2.copy_map, SIZE) == 0);

   /* Two threads map a BO's shared mapping for the first time at once, the
    * first while the second starts: it is mapped once, and both get that
    * mapping. */
   struct agx_bo bo2 = {.dev = &dev->dev, .size = SIZE, .va = &va};
   pthread_barrier_t start;
   assert(!pthread_barrier_init(&start, NULL, 2));
   static struct map_job map_jobs[2];
   slow_mmap = true;
   const unsigned shared_mmaps = mmaps;
   for (unsigned i = 0; i < 2; ++i) {
      map_jobs[i] = (struct map_job){.start = &start, .bo = &bo2};
      assert(!pthread_create(&threads[i], NULL, map_thread, &map_jobs[i]));
   }
   for (unsigned i = 0; i < 2; ++i)
      assert(!pthread_join(threads[i], NULL));
   slow_mmap = false;
   assert(mmaps == shared_mmaps + 1);
   assert(map_jobs[0].map && map_jobs[0].map == map_jobs[1].map &&
          map_jobs[0].map == bo2._map);
   assert(munmap(bo2._map, SIZE) == 0);

   /* The same on a backend that can only map a BO itself: it is mapped
    * once, in place, and both threads get that mapping. */
   dev->dev.ops.bo_mmap_independent = false;
   struct agx_bo bo3 = {.dev = &dev->dev, .size = SIZE, .va = &va};
   mapped_bo[0] = mapped_bo[1] = NULL;
   slow_mmap = true;
   const unsigned inplace_mmaps = mmaps;
   for (unsigned i = 0; i < 2; ++i) {
      map_jobs[i] = (struct map_job){.start = &start, .bo = &bo3};
      assert(!pthread_create(&threads[i], NULL, map_thread, &map_jobs[i]));
   }
   for (unsigned i = 0; i < 2; ++i)
      assert(!pthread_join(threads[i], NULL));
   slow_mmap = false;
   assert(mmaps == inplace_mmaps + 1);
   assert(mapped_bo[mmaps & 1] == &bo3);
   assert(map_jobs[0].map && map_jobs[0].map == map_jobs[1].map &&
          map_jobs[0].map == bo3._map);
   assert(munmap(bo3._map, SIZE) == 0);
   dev->dev.ops.bo_mmap_independent = true;
   pthread_barrier_destroy(&start);

   close(backing);
   simple_mtx_destroy(&dev->dev.bo_mmap_lock);
   vk_object_base_finish(&dev->vk.base);
   free(dev);
   puts("PASS: host image copies keep their own mapping through unmapped, "
        "placed and reserved mappings and two copying threads; a BO's "
        "shared mapping is made once when two threads map it first, in "
        "place on a backend that cannot map a copy");
   return 0;
}

/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2019 Collabora, Ltd.
 * Copyright 2020 Igalia S.L.
 * SPDX-License-Identifier: MIT
 */

#include "agx_device.h"
#include "agx_g17p_bind0.h"
#include "agx_linker.h"
#include "agx_m5_va.h"
#include "agx3_code_container_core.h"
#include "agx_va_runtime_hole.h"
#include <errno.h>
#include <stdlib.h>
#include <inttypes.h>
#include "clc/asahi_clc.h"
#include "drm-uapi/asahi_drm.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/timespec.h"
#include "util/u_debug.h"
#include "agx_abi.h"
#include "agx_bo.h"
#include "agx_compile.h"
#include "agx_device_virtio.h"
#include "agx_g15_wire.h"
#include "agx_scratch.h"
#include "decode.h"
#include "glsl_types.h"
#include "layout.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"

#include <fcntl.h>
#include <xf86drm.h>
#include "drm-uapi/dma-buf.h"
#include "vdrm.h"
#include "util/blob.h"
#include "util/log.h"
#include "util/mesa-blake3.h"
#include "util/os_file.h"
#include "util/os_mman.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_math.h"
#include "util/u_printf.h"
#include "git_sha1.h"
#include "nir_serialize.h"

int
asahi_simple_ioctl(struct agx_device *dev, unsigned cmd, void *req)
{
   if (dev->is_virtio)
      return agx_virtio_simple_ioctl(dev, cmd, req);
   return drmIoctl(dev->fd, cmd, req);
}

void
agx_drm_bo_munmap(struct agx_device *dev, struct agx_bo *bo)
{
   munmap(bo->_map, agx_bo_map_size(bo));
   bo->_map_size = 0;
}

void
agx_drm_bo_close(struct agx_device *dev, uint32_t handle)
{
   struct drm_gem_close args = {.handle = handle};
   drmIoctl(dev->fd, DRM_IOCTL_GEM_CLOSE, &args);
}

static int
agx_drm_bo_madvise(struct agx_device *dev, uint32_t handle, bool dontneed,
                   bool *retained)
{
   struct drm_asahi_gem_madvise args = {
      .handle = handle,
      .madv = dontneed ? DRM_ASAHI_GEM_MADV_DONTNEED
                       : DRM_ASAHI_GEM_MADV_WILLNEED,
   };

   if (drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GEM_MADVISE, &args))
      return -errno;

   *retained = args.retained != 0;
   return 0;
}

/* clang-format off */
static const struct debug_named_value agx_debug_options[] = {
   {"trace",     AGX_DBG_TRACE,    "Trace the command stream"},
   {"bodump",    AGX_DBG_BODUMP,   "Periodically dump live BOs"},
   {"no16",      AGX_DBG_NO16,     "Disable 16-bit support"},
   {"perf",      AGX_DBG_PERF,     "Print performance warnings"},
#ifndef NDEBUG
   {"dirty",     AGX_DBG_DIRTY,    "Disable dirty tracking"},
#endif
   {"precompile",AGX_DBG_PRECOMPILE,"Precompile shaders for shader-db"},
   {"nocompress",AGX_DBG_NOCOMPRESS,"Disable lossless compression"},
   {"nocluster", AGX_DBG_NOCLUSTER,"Disable vertex clustering"},
   {"sync",      AGX_DBG_SYNC,     "Synchronously wait for all submissions"},
   {"stats",     AGX_DBG_STATS,    "Show command execution statistics"},
   {"resource",  AGX_DBG_RESOURCE, "Log resource operations"},
   {"batch",     AGX_DBG_BATCH,    "Log batches"},
   {"nowc",      AGX_DBG_NOWC,     "Disable write-combining"},
   {"synctvb",   AGX_DBG_SYNCTVB,  "Synchronous TVB growth"},
   {"smalltile", AGX_DBG_SMALLTILE,"Force 16x16 tiles"},
   {"feedback",  AGX_DBG_FEEDBACK, "Debug feedback loops"},
   {"nomsaa",    AGX_DBG_NOMSAA,   "Force disable MSAA"},
   {"noshadow",  AGX_DBG_NOSHADOW, "Force disable resource shadowing"},
   {"scratch",   AGX_DBG_SCRATCH,  "Debug scratch memory usage"},
   {"1queue",    AGX_DBG_1QUEUE,   "Force usage of a single queue for multiple contexts"},
   {"nosoft",    AGX_DBG_NOSOFT,   "Disable soft fault optimizations"},
   {"nomerge",   AGX_DBG_NOMERGE,  "Disable control stream merging"},
   {"asynctrace",AGX_DBG_ASYNCTRACE,"Trace command streams without waiting"},
   {"bodumpverbose", AGX_DBG_BODUMPVERBOSE,   "Include extra info with dumps"},
   DEBUG_NAMED_VALUE_END
};
/* clang-format on */

void
agx_bo_free(struct agx_device *dev, struct agx_bo *bo)
{
   const uint64_t handle = bo->handle;

   if (bo->_map)
      dev->ops.bo_munmap(dev, bo);

   if (bo->g17p_vdm_alias) {
      /* Batches retain the BO until completion. Keep the compact alias mapped
       * while the idle BO is cached, but never recycle its VA over an old
       * mapping if an unbind fails (for example, a quarantined VM). */
      int ret = agx_bo_bind(dev, NULL, bo->g17p_vdm_alias, bo->size, 0,
                            DRM_ASAHI_BIND_UNBIND);
      if (!ret) {
         simple_mtx_lock(&dev->vma_lock);
         util_vma_heap_free(&dev->g17p_vdm_alias_heap, bo->g17p_vdm_alias,
                            bo->size);
         simple_mtx_unlock(&dev->vma_lock);
      }
   }

   /* Free the VA. No need to unmap the BO or unbind the VA, as the kernel will
    * take care of that when we close it.
    *
    * Under virtio the host closes the GEM handle, and with it the mapping,
    * only when it gets to the resource unref. Another thread can recycle the
    * VA before that, and the host then refuses the new BO's bind: its blob is
    * never created and every later map of it fails with EINVAL. Unbind first;
    * vdrm_bo_create() flushes this ahead of any blob that reuses the VA. Keep
    * (leak) the VA if the unbind cannot be queued.
    */
   if (!dev->is_virtio || !bo->va ||
       !agx_bo_bind(dev, NULL, bo->va->addr, bo->size, 0,
                    DRM_ASAHI_BIND_UNBIND))
      agx_va_free(dev, bo->va, false);

   if (bo->prime_fd != -1)
      close(bo->prime_fd);

   /* Reset the handle. This has to happen before the GEM close to avoid a race.
    */
   memset(bo, 0, sizeof(*bo));
   __sync_synchronize();

   dev->ops.bo_free(dev, handle);
}

/*
 * Name the reason a VM_BIND was refused.
 *
 * A VM whose accepted job recorded an error keeps that error for the rest of
 * its life and refuses every later map with EIO (mmu.rs
 * wait_for_user_map_admission). The kernel says so with "G17P VM map admission
 * refused", which contains neither "bind" nor an errno, so the failure reaches
 * a log as a bare EIO with no cause. Ask the VM for its sticky status and put
 * the real error next to the rejected op.
 */
static bool
agx_drm_vm_status(struct agx_device *dev, int32_t *error)
{
   if (!(dev->params.features & DRM_ASAHI_FEATURE_VM_STATUS))
      return false;

   struct drm_asahi_vm_status status = {.vm_id = dev->vm_id};
   struct drm_asahi_get_params params = {
      .param_group = DRM_ASAHI_PARAM_GROUP_VM_STATUS,
      .pointer = (uintptr_t)&status,
      .size = sizeof(status),
   };

   if (drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GET_PARAMS, &params))
      return false;

   *error = status.error;
   return true;
}

static int
agx_drm_bo_bind(struct agx_device *dev, struct drm_asahi_gem_bind_op *ops,
                uint32_t count)
{
   struct drm_asahi_vm_bind vm_bind = {
      .num_binds = count,
      .vm_id = dev->vm_id,
      .userptr = (uintptr_t)ops,
      .stride = sizeof(*ops),
   };

   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_VM_BIND, &vm_bind);
   if (ret) {
      const int saved_errno = errno;
      fprintf(stderr, "DRM_IOCTL_ASAHI_VM_BIND failed: %s (vm=%u, ops=%u)",
              strerror(saved_errno), vm_bind.vm_id, vm_bind.num_binds);
      for (uint32_t i = 0; i < count; ++i) {
         fprintf(stderr,
                 "%s{addr=%#llx range=%#llx offset=%#llx handle=%u "
                 "flags=%#x}",
                 i ? ", " : ": ", (unsigned long long)ops[i].addr,
                 (unsigned long long)ops[i].range,
                 (unsigned long long)ops[i].offset, ops[i].handle,
                 ops[i].flags);
      }
      int32_t vm_error = 0;
      if (agx_drm_vm_status(dev, &vm_error) && vm_error != 0) {
         fprintf(stderr, "; VM %u carries sticky error %d (%s) -- every later "
                         "map is refused, so this bind is a consequence of an "
                         "earlier submission failure, not of this address",
                 vm_bind.vm_id, vm_error, strerror(-vm_error));
      }
      fputc('\n', stderr);
      errno = saved_errno;
   }

   return ret;
}

uint64_t
agx_bo_get_g17p_vdm_alias(struct agx_device *dev, struct agx_bo *bo)
{
   if (!bo || !agx_chip_native_agx3(dev->chip))
      return 0;

   /* A screen may have several contexts. Serialize first publication so they
    * share one mapping, and never expose an alias before VM_BIND succeeds. */
   uint64_t alias = 0;
   for (unsigned attempt = 0; attempt < 2; ++attempt) {
      simple_mtx_lock(&dev->vma_lock);
      alias = bo->g17p_vdm_alias;
      if (!alias) {
         alias = util_vma_heap_alloc(&dev->g17p_vdm_alias_heap, bo->size,
                                     AIL_PAGESIZE);
         if (alias) {
            const uint32_t flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE;
            if (!agx_g17p_vdm_alias_range_valid(alias, bo->size) ||
                agx_bo_bind(dev, bo, alias, bo->size, 0, flags)) {
               util_vma_heap_free(&dev->g17p_vdm_alias_heap, alias, bo->size);
               simple_mtx_unlock(&dev->vma_lock);
               return 0;
            }
            bo->g17p_vdm_alias = alias;
         }
      }
      simple_mtx_unlock(&dev->vma_lock);
      if (alias || attempt)
         break;

      /* The compact address window is finite. Evict idle cached BOs before
       * refusing a new mapping, without reversing cache_lock -> vma_lock. */
      agx_bo_cache_evict_all(dev);
   }
   return alias;
}

/*
 * Convenience helper to bind a single BO regardless of kernel module.
 */
int
agx_bo_bind(struct agx_device *dev, struct agx_bo *bo, uint64_t addr,
            size_t size_B, uint64_t offset_B, uint32_t flags)
{
   assert((size_B % 16384) == 0 && "alignment required");
   assert((offset_B % 16384) == 0 && "alignment required");
   assert((addr % 16384) == 0 && "alignment required");

   if (agx_va_overlaps_runtime_graph_hole(dev->chip, addr, size_B)) {
      errno = EINVAL;
      return -1;
   }

   struct drm_asahi_gem_bind_op op = {
      .flags = flags,
      .handle = bo ? bo->uapi_handle : 0,
      .offset = offset_B,
      .range = size_B,
      .addr = addr,
   };

   return dev->ops.bo_bind(dev, &op, 1);
}

/*
 * Metal's HAL200 render context is six independently allocated resources.
 * The first five are 32 KiB and the last is 48 KiB.  Each resource has a
 * compact high mapping and an identity low alias in the render VM:
 *
 *   0x1000018000 -> 0x18000, ..., 0x1000068000 -> 0x68000
 *
 * Honeykrisp intentionally combines those resources into one fixed BO at
 * 0x1000000000, preserving the exact high addresses as BO offsets.  Bind the
 * corresponding ranges a second time so hardware-side direct-context reads
 * see the same low aperture as Metal.  The resource sizes and gaps are
 * trace-observed; do not turn this into one broad low mapping.
 */
static const struct {
   uint64_t offset;
   uint64_t size;
} agx_g17p_render_context_low_slots[] = {
   {UINT64_C(0x18000), UINT64_C(0x8000)},
   {UINT64_C(0x28000), UINT64_C(0x8000)},
   {UINT64_C(0x38000), UINT64_C(0x8000)},
   {UINT64_C(0x48000), UINT64_C(0x8000)},
   {UINT64_C(0x58000), UINT64_C(0x8000)},
   {UINT64_C(0x68000), UINT64_C(0xc000)},
};

static void
agx_g17p_unbind_render_context_low_slots(struct agx_device *dev,
                                         unsigned count)
{
   for (unsigned i = 0; i < count; ++i) {
      const uint64_t slot = agx_g17p_render_context_low_slots[i].offset;
      const uint64_t size = agx_g17p_render_context_low_slots[i].size;
      agx_bo_bind(dev, NULL, slot, size, 0, DRM_ASAHI_BIND_UNBIND);
   }
}

static bool
agx_g17p_bind_render_context_low_slots(struct agx_device *dev,
                                       struct agx_bo *context)
{
   const uint32_t flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE;

   for (unsigned i = 0;
        i < ARRAY_SIZE(agx_g17p_render_context_low_slots); ++i) {
      const uint64_t slot = agx_g17p_render_context_low_slots[i].offset;
      const uint64_t size = agx_g17p_render_context_low_slots[i].size;

      if (agx_bo_bind(dev, context, slot, size, slot, flags)) {
         fprintf(stderr,
                 "Failed to bind G17P render-context low slot %#llx\n",
                 (unsigned long long)slot);
         agx_g17p_unbind_render_context_low_slots(dev, i);
         return false;
      }
   }

   return true;
}

int
agx_bind_timestamps(struct agx_device *dev, struct agx_bo *bo, uint32_t *handle)
{
   struct drm_asahi_gem_bind_object bind = {
      .op = DRM_ASAHI_BIND_OBJECT_OP_BIND,
      .flags = DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS,
      .handle = bo->uapi_handle,
      .range = bo->size,
   };

   int ret = dev->ops.bo_bind_object(dev, &bind);
   *handle = bind.object_handle;
   return ret;
}

static struct agx_bo *
agx_bo_alloc(struct agx_device *dev, size_t size, size_t align,
             enum agx_bo_flags flags)
{
   struct agx_bo *bo;
   unsigned handle = 0;

   /* executable implies low va */
   assert(!(flags & AGX_BO_EXEC) || (flags & AGX_BO_LOW_VA));

   struct drm_asahi_gem_create gem_create = {.size = size};

   if (flags & AGX_BO_WRITEBACK)
      gem_create.flags |= DRM_ASAHI_GEM_WRITEBACK;

   if (!(flags & (AGX_BO_SHARED | AGX_BO_SHAREABLE))) {
      gem_create.flags |= DRM_ASAHI_GEM_VM_PRIVATE;
      gem_create.vm_id = dev->vm_id;
   }

   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GEM_CREATE, &gem_create);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_GEM_CREATE failed: %m\n");
      return NULL;
   }

   handle = gem_create.handle;

   simple_mtx_lock(&dev->bo_map_lock);
   bo = agx_lookup_bo(dev, handle);
   agx_include_bo_handle(dev, handle);
   simple_mtx_unlock(&dev->bo_map_lock);

   /* Fresh handle */
   assert(!memcmp(bo, &((struct agx_bo){}), sizeof(*bo)));

   bo->dev = dev;
   bo->size = gem_create.size;
   bo->align = align;
   bo->flags = flags;
   bo->handle = bo->uapi_handle = handle;
   bo->prime_fd = -1;

   enum agx_va_flags va_flags = flags & AGX_BO_LOW_VA ? AGX_VA_USC : 0;
   bo->va = agx_va_alloc(dev, size, bo->align, va_flags, 0);
   if (!bo->va) {
      fprintf(stderr, "Failed to allocate BO VMA\n");
      agx_bo_free(dev, bo);
      return NULL;
   }

   uint32_t bind = DRM_ASAHI_BIND_READ;
   if (!(flags & AGX_BO_READONLY)) {
      bind |= DRM_ASAHI_BIND_WRITE;
   }

   ret = agx_bo_bind(dev, bo, bo->va->addr, bo->size, 0, bind);
   if (ret) {
      agx_bo_free(dev, bo);
      return NULL;
   }

   return bo;
}

static void
agx_bo_mmap(struct agx_device *dev, struct agx_bo *bo, void *fixed_addr,
            size_t length)
{
   assert(bo->_map == NULL && "not double mapped");

   struct drm_asahi_gem_mmap_offset gem_mmap_offset = {.handle =
                                                          bo->uapi_handle};
   int ret, flags;

   ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET, &gem_mmap_offset);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_MMAP_BO failed: %m\n");
      assert(0);
   }

   flags = MAP_SHARED | (fixed_addr ? MAP_FIXED : 0);
   bo->_map_size = length ?: bo->size;
   bo->_map = os_mmap(fixed_addr, bo->_map_size, PROT_READ | PROT_WRITE,
                      flags, dev->fd, gem_mmap_offset.offset);
   if (bo->_map == MAP_FAILED) {
      bo->_map = NULL;
      bo->_map_size = 0;
      fprintf(stderr,
              "mmap failed: result=%p size=0x%llx fd=%i offset=0x%llx %m\n",
              bo->_map, (long long)bo->size, dev->fd,
              (long long)gem_mmap_offset.offset);
   }
}

struct agx_bo *
agx_bo_import(struct agx_device *dev, int fd)
{
   struct agx_bo *bo;
   ASSERTED int ret;
   unsigned gem_handle;

   simple_mtx_lock(&dev->bo_map_lock);

   ret = drmPrimeFDToHandle(dev->fd, fd, &gem_handle);
   if (ret) {
      fprintf(stderr, "import failed: Could not map fd %d to handle\n", fd);
      simple_mtx_unlock(&dev->bo_map_lock);
      return NULL;
   }

   bo = agx_lookup_bo(dev, gem_handle);
   agx_include_bo_handle(dev, gem_handle);

   if (!bo->size) {
      bo->dev = dev;
      bo->align = AIL_PAGESIZE;
      bo->handle = bo->uapi_handle = gem_handle;
      bo->prime_fd = -1;
      bo->size = lseek(fd, 0, SEEK_END);

      /* Sometimes this can fail and return -1. size of -1 is not
       * a nice thing for mmap to try mmap. Be more robust also
       * for zero sized maps and fail nicely too
       */
      if ((bo->size == 0) || (bo->size == (size_t)-1)) {
         goto error;
      }
      if (bo->size & (AIL_PAGESIZE - 1)) {
         fprintf(
            stderr,
            "import failed: BO is not a multiple of the page size (0x%llx bytes)\n",
            (long long)bo->size);
         goto error;
      }

      bo->flags = AGX_BO_SHARED | AGX_BO_SHAREABLE;
      bo->prime_fd = os_dupfd_cloexec(fd);
      bo->label = "Imported BO";
      if (bo->prime_fd < 0)
         goto error;

      p_atomic_set(&bo->refcnt, 1);
      bo->va = agx_va_alloc(dev, bo->size, bo->align, 0, 0);

      if (!bo->va) {
         fprintf(
            stderr,
            "import failed: Could not allocate from VMA heap (0x%llx bytes)\n",
            (long long)bo->size);
         goto error;
      }

      if (dev->is_virtio) {
         bo->uapi_handle = vdrm_handle_to_res_id(dev->vdrm, bo->handle);
      } else
      {
         bo->uapi_handle = bo->handle;
      }

      ret = agx_bo_bind(dev, bo, bo->va->addr, bo->size, 0,
                        DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE);
      if (ret) {
         fprintf(stderr, "import failed: Could not bind BO at 0x%llx\n",
                 (long long)bo->va->addr);
         goto error;
      }
   } else {
      /* bo->refcnt == 0 can happen if the BO
       * was being released but agx_bo_import() acquired the
       * lock before agx_bo_unreference(). In that case, refcnt
       * is 0 and we can't use agx_bo_reference() directly, we
       * have to re-initialize the refcnt().
       * Note that agx_bo_unreference() checks
       * refcnt value just after acquiring the lock to
       * make sure the object is not freed if agx_bo_import()
       * acquired it in the meantime.
       */
      if (p_atomic_read(&bo->refcnt) == 0)
         p_atomic_set(&bo->refcnt, 1);
      else
         agx_bo_reference(bo);

      /* If this bo came back to us via import, it had better
       * been marked shared to begin with.
       */
      assert(bo->flags & AGX_BO_SHAREABLE);
      assert(bo->flags & AGX_BO_SHARED);
      assert(bo->prime_fd != -1);
   }
   simple_mtx_unlock(&dev->bo_map_lock);

   assert(bo->dev != NULL && "post-condition");

   if (dev->debug &
       (AGX_DBG_TRACE | AGX_DBG_ASYNCTRACE)) {
      agx_bo_map(bo);
      agxdecode_track_alloc(dev->agxdecode, bo);
   }

   return bo;

error:
   /* PRIME_FD_TO_HANDLE acquired a GEM reference before any of the fallible
    * userspace setup above. Release the partially initialized entry and its
    * handle together so a malformed import cannot poison a later handle
    * reuse or leak one GEM reference per compositor retry. */
   agx_bo_free(dev, bo);
   simple_mtx_unlock(&dev->bo_map_lock);
   return NULL;
}

bool
agx_bo_make_shared(struct agx_device *dev, struct agx_bo *bo)
{
   bool success = false;
   int error = 0;

   /* Export is a one-way state transition shared by every pipe context on the
    * screen. Serialize both the PRIME fd creation and implicit-writer handoff:
    * two compositor threads must not overwrite/close each other's prime_fd or
    * publish AGX_BO_SHARED before the handoff is complete. */
   simple_mtx_lock(&dev->bo_map_lock);

   if (!(bo->flags & AGX_BO_SHAREABLE)) {
      error = EINVAL;
      goto out;
   }

   if (bo->flags & AGX_BO_SHARED) {
      success = bo->prime_fd >= 0;
      if (!success)
         error = EIO;
      goto out;
   }

   assert(bo->prime_fd == -1);

   int prime_fd = -1;
   int ret = drmPrimeHandleToFD(dev->fd, bo->handle, DRM_CLOEXEC | DRM_RDWR,
                                &prime_fd);
   if (ret || prime_fd < 0) {
      error = ret < -1 ? -ret : (errno ? errno : EIO);
      goto out;
   }

   /* Keep the transition transactional. In particular, a recoverable fd
    * exhaustion must not leave AGX_BO_SHARED set with no dma-buf fd, because
    * every later export and implicit-sync operation is allowed to retry. */
   bo->prime_fd = prime_fd;

   /* If there is a pending writer to this BO, import it into the buffer
    * for implicit sync.
    */
   uint64_t writer = p_atomic_read_relaxed(&bo->writer);
   if (writer) {
      int out_sync_fd = -1;
      ret = drmSyncobjExportSyncFile(dev->fd, agx_bo_writer_syncobj(writer),
                                     &out_sync_fd);
      if (ret || out_sync_fd < 0) {
         error = ret < -1 ? -ret : (errno ? errno : EIO);
         if (out_sync_fd >= 0)
            close(out_sync_fd);
         goto error;
      }

      ret = agx_import_sync_file(dev, bo, out_sync_fd);
      close(out_sync_fd);
      if (ret) {
         error = ret < -1 ? -ret : (errno ? errno : EIO);
         goto error;
      }
   }

   bo->flags |= AGX_BO_SHARED;
   success = true;
   goto out;

error:
   close(bo->prime_fd);
   bo->prime_fd = -1;

out:
   simple_mtx_unlock(&dev->bo_map_lock);
   if (error)
      errno = error;
   return success;
}

int
agx_bo_export(struct agx_device *dev, struct agx_bo *bo)
{
   if (!agx_bo_make_shared(dev, bo))
      return -1;

   return os_dupfd_cloexec(bo->prime_fd);
}

static int
agx_bo_bind_object(struct agx_device *dev,
                   struct drm_asahi_gem_bind_object *bind)
{
   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GEM_BIND_OBJECT, bind);
   if (ret) {
      fprintf(stderr,
              "DRM_IOCTL_ASAHI_GEM_BIND_OBJECT failed: %m (handle=%d)\n",
              bind->handle);
   }

   return ret;
}

static int
agx_bo_unbind_object(struct agx_device *dev, uint32_t object_handle)
{
   struct drm_asahi_gem_bind_object gem_bind = {
      .op = DRM_ASAHI_BIND_OBJECT_OP_UNBIND,
      .object_handle = object_handle,
   };

   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GEM_BIND_OBJECT, &gem_bind);
   if (ret) {
      fprintf(stderr,
              "DRM_IOCTL_ASAHI_GEM_BIND_OBJECT failed: %m (object_handle=%d)\n",
              object_handle);
   }

   return ret;
}

static ssize_t
agx_get_params(struct agx_device *dev, void *buf, size_t size)
{
   struct drm_asahi_get_params get_param = {
      .param_group = 0,
      .pointer = (uint64_t)(uintptr_t)buf,
      .size = size,
   };

   memset(buf, 0, size);

   int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_GET_PARAMS, &get_param);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_GET_PARAMS failed: %m\n");
      return -EINVAL;
   }

   return get_param.size;
}

static int
agx_submit(struct agx_device *dev, struct drm_asahi_submit *submit,
           struct agx_submit_virt *virt)
{
   if (dev->g15_legacy_wire) {
      /* Preserve the caller's command buffer and syncs. The kernel reads
       * this temporary command buffer synchronously during the ioctl. */
      if (!submit->cmdbuf_size || submit->cmdbuf_size > (4u << 20)) {
         errno = EINVAL;
         return -1;
      }
      void *wire = malloc(submit->cmdbuf_size);
      if (!wire) {
         errno = ENOMEM;
         return -1;
      }
      size_t size;
      bool ok = agx_g15_wire_pack((void *)(uintptr_t)submit->cmdbuf,
                                  submit->cmdbuf_size, wire,
                                  submit->cmdbuf_size, &size);
      if (!ok) {
         free(wire);
         errno = EINVAL;
         return -1;
      }
      struct drm_asahi_submit packed = *submit;
      packed.cmdbuf = (uintptr_t)wire;
      packed.cmdbuf_size = size;
      int ret = drmIoctl(dev->fd, DRM_IOCTL_ASAHI_SUBMIT, &packed);
      int saved_errno = errno;
      free(wire);
      errno = saved_errno;
      return ret;
   }
   return drmIoctl(dev->fd, DRM_IOCTL_ASAHI_SUBMIT, submit);
}

const agx_device_ops_t agx_device_drm_ops = {
   .bo_alloc = agx_bo_alloc,
   .bo_bind = agx_drm_bo_bind,
   .bo_mmap = agx_bo_mmap,
   .bo_mmap_independent = true,
   .get_params = agx_get_params,
   .submit = agx_submit,
   .bo_bind_object = agx_bo_bind_object,
   .bo_unbind_object = agx_bo_unbind_object,
   .simple_ioctl = asahi_simple_ioctl,
   .bo_munmap = agx_drm_bo_munmap,
   .bo_free = agx_drm_bo_close,
   .bo_madvise = agx_drm_bo_madvise,
};

static uint64_t
gcd(uint64_t n, uint64_t m)
{
   while (n != 0) {
      uint64_t remainder = m % n;
      m = n;
      n = remainder;
   }

   return m;
}

static void
agx_init_timestamps(struct agx_device *dev)
{
   uint64_t user_ts_gcd =
      gcd(dev->params.command_timestamp_frequency_hz, NSEC_PER_SEC);

   dev->user_timestamp_to_ns.num = NSEC_PER_SEC / user_ts_gcd;
   dev->user_timestamp_to_ns.den =
      dev->params.command_timestamp_frequency_hz / user_ts_gcd;
}

static struct agx_bo *
agx_create_fixed_bo(struct agx_device *dev, size_t size,
                    enum agx_bo_flags flags, uint64_t address,
                    uint32_t bind_flags, const char *label)
{
   struct agx_bo *bo = agx_bo_create(dev, size, 0, flags, label);
   if (bo == NULL)
      return NULL;

   if (agx_bo_bind(dev, bo, address, size, 0, bind_flags) != 0) {
      agx_bo_free(dev, bo);
      return NULL;
   }

   return bo;
}

/* The caller retains its fd/platform handle when open fails. Track only
 * completed common initialization stages; never run full close on a partial
 * device or destroy a mutex whose initialization was not reached. */
enum agx_device_init_stage {
   AGX_INIT_VIRTIO = BITFIELD_BIT(0),
   AGX_INIT_BO_MAP = BITFIELD_BIT(1),
   AGX_INIT_HEAPS = BITFIELD_BIT(2),
   AGX_INIT_ALIAS_HEAP = BITFIELD_BIT(3),
   AGX_INIT_GLSL = BITFIELD_BIT(4),
   AGX_INIT_PRINTF = BITFIELD_BIT(5),
   AGX_INIT_CONTEXT_SLOTS = BITFIELD_BIT(6),
};

static void
agx_cleanup_partial_device(struct agx_device *dev, unsigned initialized)
{
   if (initialized & AGX_INIT_GLSL) {
      if (initialized & AGX_INIT_CONTEXT_SLOTS)
         agx_g17p_unbind_render_context_low_slots(
            dev, ARRAY_SIZE(agx_g17p_render_context_low_slots));
      agx_bo_unreference(dev, dev->printf.bo);
      agx_bo_unreference(dev, dev->usc_base_bo);
      agx_bo_unreference(dev, dev->g17p_render_context_bo);
      agx_bo_unreference(dev, dev->scratch_bo);
      agx_bo_unreference(dev, dev->zero_bo);
      if (initialized & AGX_INIT_PRINTF)
         u_printf_destroy(&dev->printf);
      agx_bo_cache_evict_all(dev);
      glsl_type_singleton_decref();
   }

   if (dev->replay_heap_initialized)
      util_vma_heap_finish(&dev->replay_heap);
   if (initialized & AGX_INIT_ALIAS_HEAP)
      util_vma_heap_finish(&dev->g17p_vdm_alias_heap);
   if (initialized & AGX_INIT_HEAPS) {
      util_vma_heap_finish(&dev->usc_heap);
      util_vma_heap_finish(&dev->main_heap);
      simple_mtx_destroy(&dev->vma_lock);
   }
   if (initialized & AGX_INIT_BO_MAP) {
      util_sparse_array_finish(&dev->bo_map);
      simple_mtx_destroy(&dev->bo_cache.lock);
      simple_mtx_destroy(&dev->bo_map_lock);
      agxdecode_destroy_context(dev->agxdecode);
   }
   if (initialized & AGX_INIT_VIRTIO) {
      vdrm_device_close(dev->vdrm);
      dev->vdrm = NULL;
   }
}

bool
agx_open_device(void *memctx, struct agx_device *dev)
{
   unsigned initialized = 0;

   dev->debug =
      debug_get_flags_option("ASAHI_MESA_DEBUG", agx_debug_options, 0);

   ssize_t params_size = -1;

   /* A backend with preset operations has already identified its device. */
   if (dev->ops.get_params == NULL) {
      dev->ops = agx_device_drm_ops;

      /* DRM version check */
      drmVersionPtr version = drmGetVersion(dev->fd);
      if (!version) {
         fprintf(stderr, "cannot get version: %s", strerror(errno));
         goto fail;
      }

      if (!strcmp(version->name, "asahi")) {
         dev->is_virtio = false;
         dev->render_rsrc_spec_hi = version->version_major > 0 ||
                                   version->version_minor >= 2;
         dev->ops = agx_device_drm_ops;
      } else if (!strcmp(version->name, "virtio_gpu")) {
         dev->is_virtio = true;
         if (!agx_virtio_open_device(dev)) {
            fprintf(
               stderr,
               "Error opening virtio-gpu device for Asahi native context\n");
            drmFreeVersion(version);
            goto fail;
         }
         initialized |= AGX_INIT_VIRTIO;
      } else {
         drmFreeVersion(version);
         goto fail;
      }

      drmFreeVersion(version);
   }

   memset(&dev->params, 0, sizeof(dev->params));
   params_size = dev->ops.get_params(dev, &dev->params, sizeof(dev->params));
   if (params_size <= 0) {
      fprintf(stderr, "Failed to query AGX device parameters: %m\n");
      goto fail;
   }
   /* Capacity limits are an optional tail; keep accepting the legacy reply. */
   const size_t required_params_size =
      offsetof(struct drm_asahi_params_global, memory_limit);
   if ((size_t)params_size < required_params_size) {
      fprintf(stderr,
              "AGX parameter reply is too short (%zd, expected at least %zu)\n",
              params_size, required_params_size);
      goto fail;
   }

   dev->chip = agx_chip_from_params(&dev->params);

   uint64_t normalized_features;
   dev->g15_legacy_wire = false;
   if (!dev->is_virtio &&
       agx_g15_wire_features(&dev->params, &normalized_features)) {
      dev->g15_legacy_wire = true;
      dev->params.features = normalized_features;
   }

   /*
    * Recognition, shader codegen, descriptors and the command stream are
    * independently versioned. A part is usable only once every required axis
    * is implemented; otherwise a newly-recognised chip can silently reach an
    * encoder from the wrong generation.
    */
   const bool m5_bringup =
      debug_get_bool_option("ASAHI_M5_BRINGUP", false) &&
      agx_params_m5_bringup_supported(&dev->params);
   if (m5_bringup)
      fprintf(stderr, "Experimental M5 HAL300 integration enabled; general rendering is unqualified\n");
   if (!agx_params_driver_supported(&dev->params) && !m5_bringup) {
      if (dev->chip == AGX_CHIP_UNKNOWN) {
         fprintf(stderr,
                 "Unsupported Apple GPU architecture G%u (variant 0x%x, %u "
                 "clusters)\n",
                 dev->params.gpu_generation, dev->params.gpu_variant,
                 dev->params.num_clusters_total);
      } else if (!agx_chip_codegen_supported(dev->chip)) {
         fprintf(stderr,
                 "Apple %s executes the %s instruction set, which this "
                 "driver has no code generator for (variant 0x%x, %u "
                 "clusters)\n",
                 agx_chip_name(dev->chip),
                 agx_isa_name(agx_chip_isa(dev->chip)),
                 dev->params.gpu_variant,
                 dev->params.num_clusters_total);
      } else if (dev->chip == AGX_CHIP_G17P) {
         fprintf(stderr,
                 "Apple G17P kernel contract does not match this driver "
                 "(USC %u, HAL %u; expected USC 3, HAL 200)\n",
                 dev->params.usc_generation,
                 dev->params.gpu_hal_generation);
      } else if (dev->chip == AGX_CHIP_G15C) {
         fprintf(stderr,
                 "Apple G15C kernel parameters do not match this driver "
                 "(chip %#x, %u dies, %u clusters, USC %u, HAL %u; expected "
                 "chip 0x6031, 1 die, 4 clusters, USC 3, HAL 0)\n",
                 dev->params.chip_id, dev->params.num_dies,
                 dev->params.num_clusters_total, dev->params.usc_generation,
                 dev->params.gpu_hal_generation);
      } else {
         fprintf(stderr,
                 "Apple %s uses the %s command stream, which this driver "
                 "does not implement (variant 0x%x, %u clusters)\n",
                 agx_chip_name(dev->chip),
                 agx_command_stream_generation_name(
                    agx_chip_command_stream_generation(dev->chip)),
                 dev->params.gpu_variant,
                 dev->params.num_clusters_total);
      }

      goto fail;
   }

   if (!agx_params_native_kernel_abi_supported(&dev->params)) {
      fprintf(stderr,
              "Incompatible native AGX3 kernel contract (features %#llx); "
              "use the kernel matched to this Mesa build; see its "
              "compatibility documentation\n",
              (unsigned long long)dev->params.features);
      goto fail;
   }

   /* The M4 Pro runs as G17P too; it takes the M-series name below. */
   const char *product = dev->params.chip_id == 0x8140
                            ? agx_chip_exact_product_name(dev->chip)
                            : NULL;
   if (product) {
      /* Match the M-series format with the exact GPU codename and revision. */
      snprintf(dev->name, sizeof(dev->name), "Apple %s (%s %02X)", product,
               agx_chip_name(dev->chip), dev->params.gpu_revision + 0xA0);
   } else {
      const char *variant = " Unknown";
      switch (dev->params.gpu_variant) {
      case 'G':
         variant = "";
         break;
      case 'S':
         variant = " Pro";
         break;
      case 'C':
         variant = " Max";
         break;
      case 'D':
         variant = " Ultra";
         break;
      }
      snprintf(dev->name, sizeof(dev->name), "Apple M%d%s (G%d%c %02X)",
               dev->params.gpu_generation - 12, variant,
               dev->params.gpu_generation, dev->params.gpu_variant,
               dev->params.gpu_revision + 0xA0);
   }

   /* We need a large chunk of VA space carved out for robustness. Hardware
    * loads can shift an i32 by up to 2, for a total shift of 4. If the base
    * address is zero, 36-bits is therefore enough to trap any zero-extended
    * 32-bit index. For more generality we would need a larger carveout, but
    * this is already optimal for VBOs.
    *
    * TODO: Maybe this should be on top instead? Might be ok.
    */
   uint64_t reservation = (1ull << 36);

   /* Keep the established AGX2 printf address outside the ordinary heap.
    * G17P publishes its allocator-owned address through runtime root state,
    * but retaining the same floor keeps the robustness carveout unchanged.
    */
   assert(reservation == LIBAGX_PRINTF_BUFFER_ADDRESS);
   reservation += LIBAGX_PRINTF_BUFFER_SIZE;

   dev->guard_size = AIL_PAGESIZE;

   // Put the USC heap at the bottom of the user address space, 4GiB aligned
   dev->shader_base =
      ALIGN_POT(MAX2(dev->params.vm_start, reservation), 0x100000000ull);

   /* G17P: the USC exec window must sit at 2^40.
    *
    * HARDWARE FACT, not a preference.  With the window at the address this
    * formula derives from vm_start (0x1100000000), every Honeykrisp kick --
    * including one whose shader and preamble touch no memory at all -- ended
    * in a GMMU page fault (reason=3) with zero elapsed time, i.e. before the
    * first instruction.  With the window at 0x10000000000 the identical
    * binary executes: non-zero duration, no fault, and the shader's stores
    * land in the right buffer.  Three consecutive clients on one boot, no
    * faults.  It is also the address of the only other configuration ever
    * observed to execute on this chip, m1n1's add3 witness (CODE_IMAGE).
    *
    * Why the address matters is visible in the launch record.  The CDM
    * pipeline pointer is split across three fields: word 2 carries
    * program_va[37:6], word 1 bits 22..31 carry program_va[47:38], and word
    * 3's low byte carries program_va[47:40].  Below 2^38 the two high fields
    * are BOTH ZERO, so an entire half of the pointer encoding is never
    * exercised -- and that is exactly the regime the old base put us in.  At
    * 2^40 they carry 4 and 1, byte-identical to add3's long-standing
    * "constant" 0x01000000 and control 0x40000001.
    *
    * What is NOT the cause, each ruled out by this same experiment: the
    * offset within the window (unchanged at ~+0xfffac000 across the failing
    * and passing runs, so the window's reach is not the limit), the
    * preamble, the shader, the container, and the descriptor chain.
    *
    * The boundary IS pinned, by experiment.  A base of 0x4000000000 (2^38)
    * makes word 1 non-zero while leaving word 3's byte zero; it was run on
    * hardware and the dispatch does NOT execute (vkQueueSubmit -4, no
    * timestamps).  So "the high fields are exercised" is not the rule --
    *
    *     the program VA must have BIT 40 SET.
    *
    * Note the asymmetry this exposes: bits 38 and 39 live in the same
    * va[47:38] field as bit 40, yet setting them is not enough.  Word 3's
    * low byte carries only va[47:40], so the hardware appears to require the
    * bits that word 3 alone provides; the two fields are not the pure
    * redundancy the encoding suggests.  Anything that revisits the launch-word
    * decode, or that moves this layout again, has to explain that.
    *
    * Consequence, and it is not a conservative choice but a forced one: the
    * user heap ends up at 32GiB rather than 64GiB, because the read-only
    * sparse alias has to fit below the kernel range and the window must stay
    * a power of two.  Shrinking the kernel reservation buys nothing -- the
    * power-of-two step dominates.  32GiB of VA against <=16GB of RAM is not
    * a practical limit. */
   if (agx_chip_native_agx3(dev->chip))
      dev->shader_base = AGX_G17P_USC_WINDOW_BASE;
   else if (agx_chip_g17_m5(dev->chip))
      dev->shader_base = AGX_M5_SHADER_BASE;

   if (dev->shader_base < reservation) {
      /* Our robustness implementation requires the bottom unmapped */
      fprintf(stderr, "Unexpected address layout, can't cope\n");
      assert(0);
      goto fail;
   }

   uint64_t shader_size = 0x100000000ull;
   /* T8122: only the lower half of the 2^40 window is free. */
   if (dev->chip == AGX_CHIP_G15G)
      shader_size = AGX_T8122_RUNTIME_WINDOW_HIGH - dev->shader_base;

   /*
    * Where the user heap goes relative to the USC window.
    *
    * Everywhere else the heap follows the USC window, which is what the
    * original layout did. G17P cannot: its USC window sits at exactly 2^40,
    * so a heap placed after it starts above 2^40 -- and the VDM PPP State
    * Update carries its payload pointer in a FORTY BIT field (word 0 bits
    * 0..7 hold pointer bits 32..39; see agx_vdm_ppp_state_core.h, which reads
    * that layout off Apple's own G15/HAL200/HAL300 emitters). Every PPP
    * payload was therefore unaddressable, agx_ppp_fini() refused it, and
    * vkEndCommandBuffer returned VK_ERROR_FEATURE_NOT_PRESENT before a render
    * command buffer could ever be recorded.
    *
    * Compute never noticed because nothing on that path encodes a 40-bit
    * pointer.
    *
    * Nested G17P VDM pointers are narrower still: hardware adds the
    * 0x1000000000 render-context base to a 39-bit compact offset. Therefore
    * both the primary heap and its sparse shadow must end by 0x9000000000,
    * not merely below the 2^40 USC window. This leaves the bottom 2^36
    * robustness carveout untouched while making every ordinary VDM payload
    * directly representable, with no per-run alias option.
    */
   const bool heap_below_usc = agx_chip_native_agx3(dev->chip) ||
                               agx_chip_g17_m5(dev->chip);

   /*
    * Start the low heap as high as it can go rather than at the reservation.
    * agx_va_split_user_window() requires size < start and the sparse alias
    * needs a second window of the same size, so the usable heap is maximised
    * by start = shader_base/2: that yields size = shader_base/4 with the alias
    * ending exactly at shader_base. Starting at the reservation instead would
    * waste three quarters of the range on unused low VA.
    */
   uint64_t low_floor =
      ALIGN_POT(MAX2(dev->params.vm_start, reservation), 0x100000000ull);
   uint64_t user_start;
   if (heap_below_usc) {
      /* Bound by the NARROWER of the two G17P pointers, not by the USC
       * window: the compact VDM pointer reaches 0x9000000000, but the
       * absolute 36-bit shr(3) USC Texture/Sampler Buffer field stops at
       * 2^39. Starting halfway through that aperture is as high as the split
       * allows -- it needs size < start, and the sparse alias needs a second
       * window of the same size -- and lands the alias exactly on the limit.
       */
      user_start = MAX2(low_floor, AGX_G17P_USER_LIMIT / 2);
   } else {
      user_start = dev->shader_base + shader_size;
   }

   /* Checked rather than asserted: the base is settable at runtime and these
    * are release builds. */
   if (dev->shader_base < dev->params.vm_start ||
       (!heap_below_usc && user_start <= dev->shader_base) ||
       (heap_below_usc && user_start >= dev->shader_base) ||
       user_start >= dev->params.vm_end) {
      fprintf(stderr,
              "USC window [%#llx, %#llx) does not fit the kernel VA window "
              "[%#llx, %#llx)\n",
              (unsigned long long)dev->shader_base,
              (unsigned long long)user_start,
              (unsigned long long)dev->params.vm_start,
              (unsigned long long)dev->params.vm_end);
      goto fail;
   }

   // Put the kernel heap at the top of the address space.
   // Give it 32GB of address space, should be more than enough for any
   // reasonable use case.
   uint64_t kernel_size = MAX2(dev->params.vm_kernel_min_size, 32ull << 30);
   struct drm_asahi_vm_create vm_create = {
      .kernel_start = dev->params.vm_end - kernel_size,
      .kernel_end = dev->params.vm_end,
   };

   /* G17P is bounded by the narrower of the compact VDM aperture and the
    * absolute USC texture/sampler table pointer, both of which sit below the
    * USC window, and T8122 additionally by its runtime's reservation. Other
    * chips retain the kernel-reported user limit. */
   if (agx_chip_g17_m5(dev->chip)) {
      struct agx_m5_va_layout layout;
      if (!agx_m5_va_layout(dev->params.vm_start, dev->params.vm_end,
                            dev->params.vm_kernel_min_size, dev->shader_base,
                            &layout)) {
         fprintf(stderr, "M5 address windows do not fit the reported GPU VM\n");
         goto fail;
      }
      vm_create.kernel_start = layout.driver_start;
      vm_create.kernel_end = layout.driver_end;
      user_start = layout.data_start;
   }
   const uint64_t user_limit = agx_chip_g17_m5(dev->chip)
      ? AGX_M5_DATA_END
      : agx_va_user_limit(dev->chip, heap_below_usc, vm_create.kernel_start);
   uint64_t user_size = user_limit - user_start;

   int ret = dev->ops.simple_ioctl(dev, DRM_IOCTL_ASAHI_VM_CREATE, &vm_create);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_VM_CREATE failed: %m\n");
      goto fail;
   }

   dev->agxdecode = agxdecode_new_context(dev->shader_base);

   agx_init_timestamps(dev);

   util_sparse_array_init(&dev->bo_map, sizeof(struct agx_bo), 512);
   simple_mtx_init(&dev->bo_map_lock, mtx_plain);

   simple_mtx_init(&dev->bo_cache.lock, mtx_plain);
   list_inithead(&dev->bo_cache.lru);
   initialized |= AGX_INIT_BO_MAP;

   for (unsigned i = 0; i < ARRAY_SIZE(dev->bo_cache.buckets); ++i)
      list_inithead(&dev->bo_cache.buckets[i]);

   /* Round the user VA window to a power-of-two window whose top half can be
    * addressed by a single bit, so sparse buffers can be aliased read-only at
    * +sparse_ro_offset.  This is a hard failure rather than an assert: the
    * reduction depends on the kernel-reported window, so a geometry we cannot
    * split must be reported, not tripped over in a debug build and silently
    * miscomputed in a release one.
    */
   if (!agx_va_split_user_window(user_start, user_size, &user_start,
                                 &user_size)) {
      fprintf(stderr,
              "Cannot split the kernel user VA window [%#llx, %#llx) for "
              "sparse aliasing\n",
              (unsigned long long)dev->params.vm_start,
              (unsigned long long)dev->params.vm_end);
      goto fail;
   }

   /* Every vkAllocateMemory() binds a read-only shadow of the allocation at
    * +sparse_ro_offset, unconditionally, "in case this is used for a sparse
    * resident buffer" (hk_device_memory.c).  So the ALIAS window
    * [user_start + user_size, user_start + 2 * user_size) is live for every
    * app allocation, not just sparse ones, and the kernel rejects a bind that
    * leaves the user range or touches the kernel range (file.rs gem_bind:
    * "not contained in user range" / "intrudes in kernel range", both EINVAL).
    *
    * agx_va_split_user_window() only sizes the window against its own start;
    * it never sees kernel_start, so nothing has been checking that the alias
    * half still fits underneath it.  At the default base it fits with three
    * orders of magnitude to spare, which is why this has never fired.  Move
    * the USC window to 2^40 and the split yields a 1TiB user heap at
    * 0x20000000000 whose alias reaches 0x40000000000 -- past the top of the
    * user range and straight through the kernel's 32GiB reservation -- and
    * the FIRST app allocation fails with VK_ERROR_UNKNOWN before anything is
    * ever submitted.
    *
    * Halve until the alias fits.  This is a no-op for the default layout
    * (0x2000000000 + 2 * 0x1000000000 = 0x4000000000, far below a kernel
    * range starting at 0x3f7fff_f8000), so it does not perturb the baseline. */
   while (user_size > 0 && user_start + 2 * user_size > user_limit)
      user_size /= 2;

   if (user_size == 0) {
      fprintf(stderr,
              "User VA window at %#llx leaves no room for the sparse "
              "read-only alias below %#llx\n",
              (unsigned long long)user_start,
              (unsigned long long)user_limit);
      goto fail;
   }

   dev->sparse_ro_offset = user_size;
   dev->main_heap_base = user_start;
   dev->main_heap_size = user_size;

   simple_mtx_init(&dev->vma_lock, mtx_plain);
   util_vma_heap_init(&dev->main_heap, user_start, user_size);
   util_vma_heap_init(&dev->usc_heap, dev->shader_base, shader_size);
   initialized |= AGX_INIT_HEAPS;
   if (agx_chip_native_agx3(dev->chip) &&
       !agx_g17p_vdm_alias_heap_init(&dev->g17p_vdm_alias_heap)) {
      fprintf(stderr, "Failed to initialize G17P low VDM alias heap\n");
      goto fail;
   }
   if (agx_chip_native_agx3(dev->chip))
      initialized |= AGX_INIT_ALIAS_HEAP;
   /* The compact alias heap starts inside the T8122 runtime USC base; keep
    * out of it (the rest of that range is already outside the heap). */
   if (dev->chip == AGX_CHIP_G15G &&
       !util_vma_heap_alloc_addr(&dev->g17p_vdm_alias_heap,
                                 AGX_G17P_VDM_ALIAS_BASE,
                                 AGX_G17P_VDM_ALIAS_RESERVED_START -
                                    AGX_G17P_VDM_ALIAS_BASE)) {
      fprintf(stderr, "Failed to reserve the T8122 runtime USC base\n");
      goto fail;
   }

   if (!agx_va_reserve_runtime_graph_hole(
          dev->chip, &dev->main_heap, user_start, user_size,
          dev->sparse_ro_offset, &dev->usc_heap, dev->shader_base,
          shader_size)) {
      fprintf(stderr, "Failed to reserve firmware runtime-graph GPU VA\n");
      goto fail;
   }

   /* Keep captured addresses unavailable to ordinary allocations between
    * capture and replay. Reserve after all fixed firmware/alias holes, and
    * for native AGX3 and M5 using the common allocator; other winsyses own their VA. */
   if (agx_chip_native_agx3(dev->chip) || agx_chip_g17_m5(dev->chip)) {
      dev->replay_heap_initialized = agx_capture_replay_heap_init(
         &dev->main_heap, &dev->replay_heap, user_size, &dev->replay_start,
         &dev->replay_size);
      if (!dev->replay_heap_initialized) {
         fprintf(stderr, "Failed to reserve capture/replay GPU VA\n");
         goto fail;
      }
   }

   dev->vm_id = vm_create.vm_id;

   glsl_type_singleton_init_or_ref();
   initialized |= AGX_INIT_GLSL;

   if (agx_chip_uses_legacy_precompiled_programs(dev->chip)) {
      if (agx_gather_device_key(dev).needs_g13x_coherency == U_TRISTATE_YES)
         dev->libagx_programs = libagx_g13x;
      else
         dev->libagx_programs = libagx_g13g;
   } else if (dev->chip == AGX_CHIP_G17P) {
      dev->libagx_programs = libagx_g17p;
   } else if (dev->chip == AGX_CHIP_G15G) {
      /* Its own build: G15G needs the scheduling stalls G15S does not. */
      dev->libagx_programs = libagx_g15g;
   } else if (dev->chip == AGX_CHIP_G15S || dev->chip == AGX_CHIP_G15C) {
      /* G15C executes the same AGX3B encoding as G15S with no extra
       * scheduling rules, so it runs the G15S build of the programs. */
      dev->libagx_programs = libagx_g15s;
   } else if (dev->chip == AGX_CHIP_G16G) {
      dev->libagx_programs = libagx_g16g;
   } else if (agx_chip_g17_m5(dev->chip)) {
      dev->libagx_programs = libagx_g17g;
   } else {
      dev->libagx_programs = NULL;
   }

   /* Bind read-only zero page at 2^32. This is in our reservation, and can be
    * addressed with only small integers in the low/high. That lets us do some
    * robustness optimization even without soft fault.
    */
   {
      struct agx_bo *bo =
         agx_create_fixed_bo(dev, AGX_ZERO_PAGE_SIZE, 0, AGX_ZERO_PAGE_ADDRESS,
                             DRM_ASAHI_BIND_READ, "Zero page");
      if (bo == NULL) {
         fprintf(stderr, "Failed to bind zero page");
         goto fail;
      }

      dev->zero_bo = bo;
   }

   {
      struct agx_bo *bo = agx_create_fixed_bo(
         dev, AIL_PAGESIZE, 0, AGX_SCRATCH_PAGE_ADDRESS,
         DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE, "Scratch page");
      if (bo == NULL) {
         fprintf(stderr, "Failed to bind scratch page");
         goto fail;
      }

      dev->scratch_bo = bo;

      /* The contents of the scratch page are undefined, but making them nonzero
       * helps fuzz for bugs where we incorrectly read from the write section.
       */
      void *map = agx_bo_map(dev->scratch_bo);
      if (!map)
         goto fail;
      memset(map, 0xCA, AIL_PAGESIZE);
   }

   if (agx_chip_has_native_render_context(dev->chip)) {
      struct agx_bo *context = agx_create_fixed_bo(
         dev, AGX_G17P_RENDER_CONTEXT_SIZE,
         AGX_BO_WRITEBACK | AGX_BO_SHAREABLE,
         AGX_G17P_RENDER_CONTEXT_BASE,
         DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE,
         "G17P native render context");
      if (context == NULL) {
         fprintf(stderr, "Failed to bind G17P native render context");
         goto fail;
      }

      dev->g17p_render_context_bo = context;
      void *context_map = agx_bo_map(context);
      if (context_map == NULL) {
         fprintf(stderr, "Failed to map G17P native render context");
         goto fail;
      }
      memset(context_map, 0, AGX_G17P_RENDER_CONTEXT_SIZE);
      /* Publish fixed Bind0 exactly once, before any command buffer can
       * reference this context. Mutable PPP payloads use retained uploads. */
      if (!agx_g17p_bind0_initialize(context_map, AGX_G17P_RENDER_CONTEXT_SIZE))
         goto fail;
      agx_bo_assert_range(context, 0, AGX_G17P_RENDER_CONTEXT_SIZE);

      if (!agx_g17p_bind_render_context_low_slots(dev, context))
         goto fail;
      initialized |= AGX_INIT_CONTEXT_SLOTS;
   }

   const bool dynamic_printf = agx_chip_native_agx3(dev->chip) ||
                               agx_chip_g17_m5(dev->chip);
   struct agx_bo *bo =
      dynamic_printf
         ? agx_bo_create(dev, LIBAGX_PRINTF_BUFFER_SIZE, 0, AGX_BO_WRITEBACK,
                         "Printf/abort")
         : agx_create_fixed_bo(
              dev, LIBAGX_PRINTF_BUFFER_SIZE, AGX_BO_WRITEBACK,
              LIBAGX_PRINTF_BUFFER_ADDRESS,
              DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE, "Printf/abort");
   if (bo == NULL) {
      fprintf(stderr, "Failed to bind printf buffer");
      goto fail;
   }

   if (dynamic_printf &&
       (bo->va == NULL || dev->main_heap_size < LIBAGX_PRINTF_BUFFER_SIZE ||
        bo->va->addr < dev->main_heap_base ||
        bo->va->addr - dev->main_heap_base >
           dev->main_heap_size - LIBAGX_PRINTF_BUFFER_SIZE)) {
      fprintf(stderr, "G17P printf buffer escaped the allocator-owned heap");
      agx_bo_unreference(dev, bo);
      goto fail;
   }

   dev->printf_buffer_address =
      dynamic_printf ? bo->va->addr
                                 : LIBAGX_PRINTF_BUFFER_ADDRESS;

   dev->printf.bo = bo;
   uint32_t *printf_map = agx_bo_map(bo);
   if (!printf_map)
      goto fail;
   u_printf_init(&dev->printf, bo, printf_map);
   initialized |= AGX_INIT_PRINTF;

   /* Map the stage helper tables at the USC execution-window base. */
   if (agx_chip_native_agx3(dev->chip) || agx_chip_g17_m5(dev->chip)) {
      const size_t usc_base_size = AIL_PAGESIZE;

      simple_mtx_lock(&dev->vma_lock);
      bool reserved = util_vma_heap_alloc_addr(&dev->usc_heap, dev->shader_base,
                                               usc_base_size + dev->guard_size);
      simple_mtx_unlock(&dev->vma_lock);

      if (!reserved) {
         fprintf(stderr, "Failed to reserve the USC window base page\n");
         goto fail;
      }

      struct agx_bo *usc_base = agx_create_fixed_bo(
         dev, usc_base_size, AGX_BO_WRITEBACK, dev->shader_base,
         DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE, "G17P USC window base");
      if (usc_base == NULL) {
         fprintf(stderr, "Failed to bind the USC window base page\n");
         goto fail;
      }

      dev->usc_base_bo = usc_base;
      void *map = agx_bo_map(usc_base);
      if (map == NULL) {
         fprintf(stderr, "Failed to map the USC window base page\n");
         goto fail;
      }

      if (agx_chip_g17_m5(dev->chip)) {
         memset(map, 0, usc_base_size);
         if (!agx3_build_m5_container_lead(map, usc_base_size,
                                          dev->params.num_clusters_total))
            goto fail;
      } else {
         agx3_build_usc_base_page(map, usc_base_size, dev->chip, dev->params.num_cores_per_cluster);
      }
      agx_bo_assert_range(usc_base, 0, usc_base_size);

   }

   agx_bo_cache_start_idle_trim(dev);
   return true;

fail:
   agx_cleanup_partial_device(dev, initialized);
   return false;
}

void
agx_close_device(struct agx_device *dev)
{
   agx_bo_cache_stop_idle_trim(dev);
   agx_bo_unreference(dev, dev->printf.bo);
   agx_bo_unreference(dev, dev->usc_base_bo);
   if (dev->g17p_render_context_bo) {
      agx_g17p_unbind_render_context_low_slots(
         dev, ARRAY_SIZE(agx_g17p_render_context_low_slots));
   }
   agx_bo_unreference(dev, dev->g17p_render_context_bo);
   agx_bo_unreference(dev, dev->zero_bo);
   agx_bo_unreference(dev, dev->scratch_bo);
   u_printf_destroy(&dev->printf);
   agx_bo_cache_evict_all(dev);
   util_sparse_array_finish(&dev->bo_map);
   agxdecode_destroy_context(dev->agxdecode);

   if (dev->replay_heap_initialized)
      util_vma_heap_finish(&dev->replay_heap);
   util_vma_heap_finish(&dev->main_heap);
   util_vma_heap_finish(&dev->usc_heap);
   if (agx_chip_native_agx3(dev->chip))
      util_vma_heap_finish(&dev->g17p_vdm_alias_heap);
   glsl_type_singleton_decref();

   close(dev->fd);
   dev->fd = -1;
}

uint32_t
agx_create_command_queue(struct agx_device *dev,
                         enum drm_asahi_priority priority)
{
   const uint64_t layouts[2] = {0};
   return agx_create_command_queue_with_render_scs(dev, priority, layouts);
}

uint32_t
agx_create_command_queue_with_render_scs(struct agx_device *dev,
   enum drm_asahi_priority priority, const uint64_t layouts[2])
{
   if (dev->debug & AGX_DBG_1QUEUE) {
      // Abuse this lock for this, it's debug only anyway
      simple_mtx_lock(&dev->vma_lock);
      if (dev->queue_id) {
         /* The shared queue serves a later caller only if it was created with
          * every render SCS layout that caller needs (Vulkan's first queue
          * brings the ray-query layout; GL needs none). */
         const bool covered =
            (!layouts[0] || layouts[0] == dev->queue_layouts[0]) &&
            (!layouts[1] || layouts[1] == dev->queue_layouts[1]);
         simple_mtx_unlock(&dev->vma_lock);
         if (!covered) {
            errno = EINVAL;
            return UINT32_MAX;
         }
         return dev->queue_id;
      }
   }

   struct drm_asahi_queue_create queue_create = {
      .vm_id = dev->vm_id,
      .priority = priority,
      .usc_exec_base = dev->shader_base,
      .render_scs_layout = {layouts[0], layouts[1]},
   };

   unsigned request = DRM_IOCTL_ASAHI_QUEUE_CREATE;
   if (!layouts[0] && !layouts[1])
      request = _IOC(_IOC_DIR(request), _IOC_TYPE(request), _IOC_NR(request),
                     offsetof(struct drm_asahi_queue_create, render_scs_layout));
   int ret = dev->ops.simple_ioctl(dev, request, &queue_create);
   if (ret) {
      int saved_errno = errno;
      if (dev->debug & AGX_DBG_1QUEUE)
         simple_mtx_unlock(&dev->vma_lock);
      errno = saved_errno;
      return UINT32_MAX;
   }

   if (dev->debug & AGX_DBG_1QUEUE) {
      dev->queue_id = queue_create.queue_id;
      dev->queue_layouts[0] = layouts[0];
      dev->queue_layouts[1] = layouts[1];
      simple_mtx_unlock(&dev->vma_lock);
   }

   return queue_create.queue_id;
}

int
agx_destroy_command_queue(struct agx_device *dev, uint32_t queue_id)
{
   if (dev->debug & AGX_DBG_1QUEUE)
      return 0;

   struct drm_asahi_queue_destroy queue_destroy = {
      .queue_id = queue_id,
   };

   return dev->ops.simple_ioctl(dev, DRM_IOCTL_ASAHI_QUEUE_DESTROY,
                                &queue_destroy);
}

int
agx_import_sync_file(struct agx_device *dev, struct agx_bo *bo, int fd)
{
   struct dma_buf_import_sync_file import_sync_file_ioctl = {
      .flags = DMA_BUF_SYNC_WRITE,
      .fd = fd,
   };

   if (fd < 0 || bo->prime_fd < 0) {
      errno = EINVAL;
      return -1;
   }

   int ret = drmIoctl(bo->prime_fd, DMA_BUF_IOCTL_IMPORT_SYNC_FILE,
                      &import_sync_file_ioctl);

   return ret;
}

int
agx_export_sync_file(struct agx_device *dev, struct agx_bo *bo)
{
   struct dma_buf_export_sync_file export_sync_file_ioctl = {
      .flags = DMA_BUF_SYNC_RW,
      .fd = -1,
   };

   if (bo->prime_fd < 0) {
      errno = EINVAL;
      return -1;
   }

   int ret = drmIoctl(bo->prime_fd, DMA_BUF_IOCTL_EXPORT_SYNC_FILE,
                      &export_sync_file_ioctl);

   if (ret || export_sync_file_ioctl.fd < 0) {
      if (export_sync_file_ioctl.fd >= 0)
         close(export_sync_file_ioctl.fd);
      return -1;
   }

   return export_sync_file_ioctl.fd;
}

void
agx_debug_fault(struct agx_device *dev, uint64_t addr)
{
   simple_mtx_lock(&dev->bo_map_lock);

   struct agx_bo *best = NULL;

   for (uint32_t handle = 0; handle < dev->max_handle; handle++) {
      struct agx_bo *bo = agx_lookup_bo(dev, handle);
      if (!bo->va)
         continue;

      /* VMA heap addresses are absolute. AGX_BO_LOW_VA selects the USC
       * heap; agx_usc_addr() is the only layer that converts one to a
       * shader-base-relative packed pointer. */
      uint64_t bo_addr = bo->va->addr;

      if (!bo->size || bo_addr > addr)
         continue;

      if (!best || bo_addr > best->va->addr)
         best = bo;
   }

   if (!best) {
      mesa_logw("Address 0x%" PRIx64 " is unknown\n", addr);
   } else {
      uint64_t start = best->va->addr;
      uint64_t end = best->va->addr + best->size;
      if (addr > (end + 1024 * 1024 * 1024)) {
         /* 1GiB max as a sanity check */
         mesa_logw("Address 0x%" PRIx64 " is unknown\n", addr);
      } else if (addr > end) {
         mesa_logw("Address 0x%" PRIx64 " is 0x%" PRIx64
                   " bytes beyond an object at 0x%" PRIx64 "..0x%" PRIx64
                   " (%s)\n",
                   addr, addr - end, start, end - 1, best->label);
      } else {
         mesa_logw("Address 0x%" PRIx64 " is 0x%" PRIx64
                   " bytes into an object at 0x%" PRIx64 "..0x%" PRIx64
                   " (%s)\n",
                   addr, addr - start, start, end - 1, best->label);
      }
   }

   simple_mtx_unlock(&dev->bo_map_lock);
}

uint64_t
agx_get_gpu_timestamp(struct agx_device *dev)
{
   struct drm_asahi_get_time get_time = {.flags = 0};

   int ret = dev->ops.simple_ioctl(dev, DRM_IOCTL_ASAHI_GET_TIME, &get_time);
   if (ret) {
      fprintf(stderr, "DRM_IOCTL_ASAHI_GET_TIME failed: %m\n");
   }

   return get_time.gpu_timestamp;
}

/* (Re)define UUID_SIZE to avoid including vulkan.h (or p_defines.h) here. */
#define UUID_SIZE 16

void
agx_get_device_uuid(const struct agx_device *dev, void *uuid)
{
   blake3_hasher blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);

   /* The device UUID uniquely identifies the given device within the machine.
    * Since we never have more than one device, this doesn't need to be a real
    * UUID, so we use BLAKE3("agx" + gpu_generation + gpu_variant + gpu_revision).
    */
   static const char *device_name = "agx";
   _mesa_blake3_update(&blake3_ctx, device_name, strlen(device_name));

   _mesa_blake3_update(&blake3_ctx, &dev->params.gpu_generation,
                     sizeof(dev->params.gpu_generation));
   _mesa_blake3_update(&blake3_ctx, &dev->params.gpu_variant,
                     sizeof(dev->params.gpu_variant));
   _mesa_blake3_update(&blake3_ctx, &dev->params.gpu_revision,
                     sizeof(dev->params.gpu_revision));

   uint8_t blake3[BLAKE3_KEY_LEN];
   _mesa_blake3_final(&blake3_ctx, blake3);

   assert(BLAKE3_KEY_LEN >= UUID_SIZE);
   memcpy(uuid, blake3, UUID_SIZE);
}

void
agx_get_driver_uuid(void *uuid)
{
   const char *driver_id = PACKAGE_VERSION MESA_GIT_SHA1;

   /* The driver UUID is used for determining sharability of images and memory
    * between two Vulkan instances in separate processes, but also to
    * determining memory objects and sharability between Vulkan and OpenGL
    * driver. People who want to share memory need to also check the device
    * UUID.
    */
   blake3_hasher blake3_ctx;
   _mesa_blake3_init(&blake3_ctx);

   _mesa_blake3_update(&blake3_ctx, driver_id, strlen(driver_id));

   uint8_t blake3[BLAKE3_KEY_LEN];
   _mesa_blake3_final(&blake3_ctx, blake3);

   assert(BLAKE3_KEY_LEN >= UUID_SIZE);
   memcpy(uuid, blake3, UUID_SIZE);
}

unsigned
agx_get_num_cores(const struct agx_device *dev)
{
   unsigned n = 0;

   for (unsigned cl = 0; cl < dev->params.num_clusters_total; cl++) {
      n += util_bitcount(dev->params.core_masks[cl]);
   }

   return n;
}

struct agx_device_key
agx_gather_device_key(struct agx_device *dev)
{
   bool g13x_coh = (dev->params.gpu_generation == 13 &&
                    dev->params.num_clusters_total > 1) ||
                   dev->params.num_dies > 1;

   return (struct agx_device_key){
      .chip = dev->chip,
      .encoding = agx_chip_isa_encoding(dev->chip),
      .needs_g13x_coherency = u_tristate_make(g13x_coh),
      .soft_fault = agx_has_soft_fault(dev),
   };
}

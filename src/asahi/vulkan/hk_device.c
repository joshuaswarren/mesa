/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_device.h"
#include "hk_buffer_view.h"
#include "hk_frame_anatomy.h"
#include "hk_query_pool.h"

#include "hk_acceleration_structure.h"

#include "asahi/lib/agx_ioto_state_core.h"

#include "agx_bg_eot.h"
#include "agx_helpers.h"
#include "agx_scratch.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_compute_kick.h"
#include "hk_descriptor_table.h"
#include "hk_entrypoints.h"
#include "hk_event.h"
#include "hk_instance.h"
#include "hk_native_sampler.h"
#include "hk_ray_tracing_pipeline.h"
#include "hk_physical_device.h"
#include "hk_shader.h"

#include "asahi/genxml/agx_pack.h"
#include "asahi/lib/agx_bo.h"
#include "asahi/lib/agx_device.h"
#include "asahi/lib/agx3_g17p_grid_helper_core.h"
#include "compiler/nir/nir_builder.h"
#include "poly/geometry.h"
#include "util/hash_table.h"
#include "util/ralloc.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "vulkan/vulkan_core.h"
#include "vulkan/wsi/wsi_common.h"
#include "layout.h"
#include "vk_cmd_enqueue_entrypoints.h"
#include "vk_common_entrypoints.h"
#include "vk_debug_utils.h"
#include "vk_device.h"
#include "vk_drm_syncobj.h"
#include "vk_enum_to_str.h"
#include "vk_pipeline_cache.h"
#include "vk_shader.h"

#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <xf86drm.h>

static bool
hk_g17p_prewarm_poly_kernels(struct hk_device *dev)
{
#define PREWARM(program, args)                                                \
   do {                                                                        \
      if (agx_chip_g17_m5(dev->dev.chip)                                       \
             ? !agx_get_precompiled(&dev->bg_eot, program)                     \
             : !hk_g17p_compute_program_cache_prewarm(                         \
                  dev, program, sizeof(struct args)))                          \
         return false;                                                         \
   } while (0)

   PREWARM(LIBAGX_GS_SETUP_INDIRECT, libagx_gs_setup_indirect_args);
   PREWARM(LIBAGX_PREFIX_SUM_GEOM, libagx_prefix_sum_geom_args);
   PREWARM(LIBAGX_PREFIX_SUM_TESS_0, libagx_prefix_sum_tess_args);
   PREWARM(LIBAGX_PREFIX_SUM_TESS_1, libagx_prefix_sum_tess_args);
   PREWARM(LIBAGX_TESS_SETUP_INDIRECT, libagx_tess_setup_indirect_args);
   PREWARM(LIBAGX_TESS_ISOLINE_0, libagx_tess_isoline_args);
   PREWARM(LIBAGX_TESS_ISOLINE_1, libagx_tess_isoline_args);
   PREWARM(LIBAGX_TESS_TRI_0, libagx_tess_tri_args);
   PREWARM(LIBAGX_TESS_TRI_1, libagx_tess_tri_args);
   PREWARM(LIBAGX_TESS_QUAD_0, libagx_tess_quad_args);
   PREWARM(LIBAGX_TESS_QUAD_1, libagx_tess_quad_args);

#undef PREWARM

   /* M5 resets the geometry heap with an immediate-write kernel. */
   if (agx_chip_g17_m5(dev->dev.chip) &&
       !agx_get_precompiled(&dev->bg_eot, LIBAGX_WRITE_U32S))
      return false;

   return true;
}

/* clang-format off */
static const struct debug_named_value hk_perf_options[] = {
   {"notess",    HK_PERF_NOTESS,   "Skip draws with tessellation"},
   {"noborder",  HK_PERF_NOBORDER, "Disable custom border colour emulation"},
   {"nobarrier", HK_PERF_NOBARRIER,"Ignore pipeline barriers"},
   {"nobatch",   HK_PERF_NOBATCH,  "Disable submission batching"},
   {"norobust",  HK_PERF_NOROBUST, "Disable robustness"},
   {"weakbarrier", HK_PERF_WEAKBARRIER, "Keep ordering-only barriers between compute dispatches"},
   {"fullcdmbarrier", HK_PERF_FULLCDMBARRIER,
    "Force the full CDM barrier word on every chip"},
   DEBUG_NAMED_VALUE_END
};
/* clang-format on */

/*
 * We preupload some constants so we can cheaply reference later without extra
 * allocation and copying.
 *
 * TODO: This is small, don't waste a whole BO.
 */
static VkResult
hk_upload_rodata(struct hk_device *dev)
{
   const unsigned sampler_size =
      agx_chip_sampler_heap_element_size(dev->dev.chip);
   if (!sampler_size)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "%s has no known sampler heap element size",
                       agx_chip_name(dev->dev.chip));

   VkResult result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
   /* Background reloads use the same two native texel-fetch selectors as
    * ordinary shaders: staged multisample reads name entry one. */
   const unsigned sampler_count = agx_chip_native_agx3(dev->dev.chip) ? 2 : 1;
   const size_t rodata_size = align(sampler_count * sampler_size, sizeof(uint64_t));
   dev->rodata.bo =
      agx_bo_create(&dev->dev, rodata_size, 0, 0, "Read only data");

   if (!dev->rodata.bo)
      goto fail;

   uint8_t *map = agx_bo_map(dev->rodata.bo);
   if (!map)
      goto fail;
   uint32_t offs = 0;

   offs = align(offs, 8);
   if (agx_chip_command_stream_generation(dev->dev.chip) ==
       AGX_CS_GENERATION_AGX2) {
      agx_pack(&dev->dev.txf_sampler, USC_SAMPLER, cfg) {
         cfg.start = 0;
         cfg.count = 1;
         cfg.buffer = dev->rodata.bo->va->addr + offs;
      }
   } else {
      memset(&dev->dev.txf_sampler, 0, sizeof(dev->dev.txf_sampler));
   }

   struct hk_native_sampler_descriptor txf;
   if (!hk_pack_native_txf_sampler(dev->dev.chip, &txf)) {
      result = vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                          "%s cannot pack the texel-fetch sampler",
                          agx_chip_name(dev->dev.chip));
      goto fail;
   }
   for (unsigned i = 0; i < sampler_count; ++i)
      memcpy(map + offs + i * sampler_size, &txf, sampler_size);
   offs += sampler_count * sampler_size;

   if (agx_chip_agx3_path(dev->dev.chip)) {
      const uint64_t pointers[2] = {
         dev->sampled_images.base_va,
         dev->samplers.table.bo->va->addr,
      };

      dev->rodata.indexed_heap_pointers_bo =
         agx_bo_create(&dev->dev, sizeof(pointers), 0, 0,
                       "Indexed descriptor heap pointers");
      if (!dev->rodata.indexed_heap_pointers_bo)
         goto fail;

      void *pointers_map = agx_bo_map(dev->rodata.indexed_heap_pointers_bo);
      if (!pointers_map)
         goto fail;
      memcpy(pointers_map, pointers, sizeof(pointers));
      dev->rodata.indexed_heap_pointers =
         dev->rodata.indexed_heap_pointers_bo->va->addr;
   }

   /* The heap descriptor is NOT read-only data: hk_init_heap() writes it from
    * the CPU on the first tessellated or geometry-shaded draw, and the libagx
    * allocator then atomically bumps poly_heap::bottom from the GPU on every
    * such draw (src/poly/geometry.h poly_heap_alloc_offs). Keep it in its
    * own buffer object so the sampler table next to the texel-fetch sampler
    * never changes after device creation.
    */
   dev->rodata.heap_bo = agx_bo_create(&dev->dev,
                                       HK_HEAP_SLOTS * sizeof(struct poly_heap),
                                       0, 0, "Geometry heap descriptor");
   if (!dev->rodata.heap_bo)
      goto fail;

   void *heap_map = agx_bo_map(dev->rodata.heap_bo);
   if (!heap_map)
      goto fail;
   memset(heap_map, 0, HK_HEAP_SLOTS * sizeof(struct poly_heap));
   dev->rodata.heap = dev->rodata.heap_bo->va->addr;

   return VK_SUCCESS;

fail:
   agx_bo_unreference(&dev->dev, dev->rodata.heap_bo);
   agx_bo_unreference(&dev->dev, dev->rodata.indexed_heap_pointers_bo);
   agx_bo_unreference(&dev->dev, dev->rodata.bo);
   memset(&dev->rodata, 0, sizeof(dev->rodata));
   return result;
}

static uint32_t
internal_key_hash(const void *key_)
{
   const struct hk_internal_key *key = key_;

   return _mesa_hash_data(key, sizeof(struct hk_internal_key) + key->key_size);
}

static bool
internal_key_equal(const void *a_, const void *b_)
{
   const struct hk_internal_key *a = a_;
   const struct hk_internal_key *b = b_;

   return a->builder == b->builder && a->key_size == b->key_size &&
          a->stop == b->stop && memcmp(a->key, b->key, a->key_size) == 0;
}

static VkResult
hk_init_internal_shaders(struct hk_internal_shaders *s)
{
   s->ht = _mesa_hash_table_create(NULL, internal_key_hash, internal_key_equal);
   if (!s->ht)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   simple_mtx_init(&s->lock, mtx_plain);
   return VK_SUCCESS;
}

static void
hk_destroy_internal_shaders(struct hk_device *dev,
                            struct hk_internal_shaders *s, bool part)
{
   hash_table_foreach(s->ht, ent) {
      if (part) {
         struct agx_shader_part *part = ent->data;
         free(part->binary);

         /* The agx_shader_part itself is ralloc'd against the hash table so
          * will be freed.
          */
      } else {
         struct hk_api_shader *obj = ent->data;
         hk_api_shader_destroy(&dev->vk, &obj->vk, NULL);
      }
   }

   _mesa_hash_table_destroy(s->ht, NULL);
   simple_mtx_destroy(&s->lock);
}

DERIVE_HASH_TABLE(agx_sampler_packed);

static VkResult
hk_init_sampler_heap(struct hk_device *dev, struct hk_sampler_heap *h)
{
   const unsigned sampler_size =
      agx_chip_sampler_heap_element_size(dev->dev.chip);
   if (!sampler_size ||
       agx_chip_sampler_generation(dev->dev.chip) ==
          AGX_SAMPLER_GENERATION_UNKNOWN)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                       "%s has no known sampler generation (element size %u)",
                       agx_chip_name(dev->dev.chip), sampler_size);

   h->ht = agx_sampler_packed_table_create(NULL);
   if (!h->ht)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   VkResult result =
      hk_descriptor_table_init(dev, &h->table, sampler_size, 1024,
                               agx_chip_native_agx3(dev->dev.chip) ? 4096
                               : agx_chip_hal300_path(dev->dev.chip) ? 2048
                                                                  : 1024);

   if (result != VK_SUCCESS) {
      ralloc_free(h->ht);
      return result;
   }

   if (agx_chip_agx3_path(dev->dev.chip)) {
      /* Native indexed read() and sample() share the sampler table prefix.
       * Reserve texel-fetch sampler zero once, so mixed shaders need no
       * per-fetch pointer switch or descriptor copy. Preserve at least the
       * previous application capacity with explicit-bias companion samplers.
       * The native sampler index is 12 bits. */
      struct hk_native_sampler_descriptor txf;
      uint32_t index = UINT32_MAX;
      result = hk_pack_native_txf_sampler(dev->dev.chip, &txf)
         ? hk_descriptor_table_add(dev, &h->table, &txf, sampler_size, &index)
         : vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                     "%s cannot pack the reserved texel-fetch sampler",
                     agx_chip_name(dev->dev.chip));
      if (result != VK_SUCCESS) {
         hk_descriptor_table_finish(dev, &h->table);
         ralloc_free(h->ht);
         return result;
      }
      assert(index == 0);
   }
   simple_mtx_init(&h->lock, mtx_plain);
   return VK_SUCCESS;
}

static void
hk_destroy_sampler_heap(struct hk_device *dev, struct hk_sampler_heap *h)
{
   hk_descriptor_table_finish(dev, &h->table);
   ralloc_free(h->ht);
   simple_mtx_destroy(&h->lock);
}

static VkResult
hk_sampler_heap_add_locked(struct hk_device *dev, struct hk_sampler_heap *h,
                           const struct hk_native_sampler_descriptor *desc,
                           bool capture, struct hk_rc_sampler **out)
{
   /* Only the first eight bytes carry recovered sampler state.  The sparse
    * tail is required to be zero, so the packed word is a complete dedup key;
    * assert that invariant before an upload can silently alias future data. */
   const uint8_t *bytes = (const uint8_t *)desc;
   for (unsigned i = sizeof(struct agx_sampler_packed);
        i < h->table.desc_size; ++i) {
      assert(bytes[i] == 0);
      if (bytes[i] != 0)
         return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT,
                          "sampler descriptor byte %u is %u, not zero: the "
                          "packed dedup key would alias", i, bytes[i]);
   }

   struct agx_sampler_packed key;
   memcpy(&key, desc, sizeof(key));
   struct hash_entry *ent = _mesa_hash_table_search(h->ht, &key);
   if (ent != NULL) {
      *out = ent->data;

      assert((*out)->refcount != 0);
      (*out)->refcount++;
      (*out)->replay |= capture;

      return VK_SUCCESS;
   }

   struct hk_rc_sampler *rc = ralloc(h->ht, struct hk_rc_sampler);
   if (!rc)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   uint32_t index;
   VkResult result =
      capture ? hk_descriptor_table_add_replay(dev, &h->table, desc,
                                               h->table.desc_size, &index)
              : hk_descriptor_table_add(dev, &h->table, desc,
                                        h->table.desc_size, &index);
   if (result != VK_SUCCESS) {
      ralloc_free(rc);
      return result;
   }

   *rc = (struct hk_rc_sampler){
      .key = key,
      .refcount = 1,
      .index = index,
      .replay = capture,
   };

   _mesa_hash_table_insert(h->ht, &rc->key, rc);
   *out = rc;

   return VK_SUCCESS;
}

VkResult
hk_sampler_heap_add(struct hk_device *dev,
                    const struct hk_native_sampler_descriptor *desc,
                    struct hk_rc_sampler **out)
{
   struct hk_sampler_heap *h = &dev->samplers;

   simple_mtx_lock(&h->lock);
   VkResult result = hk_sampler_heap_add_locked(dev, h, desc, false, out);
   simple_mtx_unlock(&h->lock);

   return result;
}

/* For a sampler created to be captured (VK_EXT_descriptor_buffer). */
VkResult
hk_sampler_heap_add_capture(struct hk_device *dev,
                            const struct hk_native_sampler_descriptor *desc,
                            struct hk_rc_sampler **out)
{
   struct hk_sampler_heap *h = &dev->samplers;

   simple_mtx_lock(&h->lock);
   VkResult result = hk_sampler_heap_add_locked(dev, h, desc, true, out);
   simple_mtx_unlock(&h->lock);

   return result;
}

/* VK_EXT_descriptor_buffer replay: the hardware sampler must sit at the heap
 * index the captured descriptors name. Share an identical sampler only when it
 * is already there; otherwise take that index, failing if it is in use. A
 * sampler whose key already lives elsewhere stays out of the dedup table. */
VkResult
hk_sampler_heap_add_at(struct hk_device *dev,
                       const struct hk_native_sampler_descriptor *desc,
                       uint32_t index, struct hk_rc_sampler **out)
{
   struct hk_sampler_heap *h = &dev->samplers;
   struct agx_sampler_packed key;
   memcpy(&key, desc, sizeof(key));

   simple_mtx_lock(&h->lock);
   struct hash_entry *ent = _mesa_hash_table_search(h->ht, &key);
   if (ent != NULL && ((struct hk_rc_sampler *)ent->data)->index == index) {
      *out = ent->data;
      (*out)->refcount++;
      (*out)->replay = true;
      simple_mtx_unlock(&h->lock);
      return VK_SUCCESS;
   }

   struct hk_rc_sampler *rc = ralloc(h->ht, struct hk_rc_sampler);
   VkResult result = rc ? hk_descriptor_table_add_at(dev, &h->table, desc,
                                                     h->table.desc_size, index)
                        : VK_ERROR_OUT_OF_HOST_MEMORY;
   if (result != VK_SUCCESS) {
      ralloc_free(rc);
      simple_mtx_unlock(&h->lock);
      return result;
   }

   *rc = (struct hk_rc_sampler){
      .key = key,
      .refcount = 1,
      .index = index,
      .replay = true,
   };
   if (ent == NULL)
      _mesa_hash_table_insert(h->ht, &rc->key, rc);
   *out = rc;

   simple_mtx_unlock(&h->lock);
   return VK_SUCCESS;
}

static void
hk_sampler_heap_remove_locked(struct hk_device *dev, struct hk_sampler_heap *h,
                              struct hk_rc_sampler *rc)
{
   assert(rc->refcount != 0);
   rc->refcount--;

   if (rc->refcount == 0) {
      if (rc->replay)
         hk_descriptor_table_remove_replay(dev, &h->table, rc->index);
      else
         hk_descriptor_table_remove(dev, &h->table, rc->index);
      /* A replayed sampler may share its key with the dedup entry. */
      struct hash_entry *ent = _mesa_hash_table_search(h->ht, &rc->key);
      if (ent != NULL && ent->data == rc)
         _mesa_hash_table_remove(h->ht, ent);
      ralloc_free(rc);
   }
}

void
hk_sampler_heap_remove(struct hk_device *dev, struct hk_rc_sampler *rc)
{
   struct hk_sampler_heap *h = &dev->samplers;

   simple_mtx_lock(&h->lock);
   hk_sampler_heap_remove_locked(dev, h, rc);
   simple_mtx_unlock(&h->lock);
}

/* Register the kernel-maintained VM status mirror so the per-wait
 * device-lost check is a load instead of a GET_PARAMS ioctl. The kernel stores a job's sticky error in the word before it
 * signals that job's fence, exactly the ordering the per-wait query relies
 * on, and refuses registration whenever the query could report more than the
 * VM's own status. Any failure here leaves the query path in place.
 */
static void
hk_init_vm_status_mirror(struct hk_device *dev)
{
   /* virtio: dev->dev.fd is the virtgpu fd, asahi ioctls don't go there. */
   if (dev->dev.is_virtio ||
       !(dev->dev.params.features & DRM_ASAHI_FEATURE_VM_STATUS) ||
       !(dev->dev.params.features & DRM_ASAHI_FEATURE_VM_STATUS_MIRROR))
      return;

   struct agx_bo *bo =
      agx_bo_create(&dev->dev, 16384, 0, AGX_BO_WRITEBACK, "VM status mirror");
   if (!bo)
      return;

   int32_t *word = agx_bo_map(bo);
   if (!word) {
      agx_bo_unreference(&dev->dev, bo);
      return;
   }
   __atomic_store_n(word, 0, __ATOMIC_RELEASE);

   struct drm_asahi_vm_status_mirror mirror = {
      .vm_id = dev->dev.vm_id,
      .handle = bo->uapi_handle,
      .offset = 0,
   };
   struct drm_asahi_get_params query = {
      .param_group = DRM_ASAHI_PARAM_GROUP_VM_STATUS_MIRROR,
      .pointer = (uintptr_t)&mirror,
      .size = sizeof(mirror),
   };
   if (drmIoctl(dev->dev.fd, DRM_IOCTL_ASAHI_GET_PARAMS, &query)) {
      agx_bo_unreference(&dev->dev, bo);
      return;
   }

   dev->vm_status_mirror.bo = bo;
   dev->vm_status_mirror.word = word;
}

#define HK_CODE_CHUNK_SIZE (1024 * 1024)

struct agx_bo *
hk_code_alloc(struct hk_device *dev, size_t size, uint32_t *offset)
{
   struct hk_code_heap *heap = &dev->code_heap;
   struct agx_bo *bo = NULL;

   simple_mtx_lock(&heap->lock);
   uint32_t at = ALIGN_POT(heap->offset, HK_CODE_ALIGNMENT);
   if (!heap->chunk || size > heap->chunk->size ||
       at > heap->chunk->size - size) {
      struct agx_bo *chunk =
         agx_bo_create(&dev->dev, MAX2(HK_CODE_CHUNK_SIZE, size), 0,
                       AGX_BO_EXEC | AGX_BO_LOW_VA, "Shader code");
      if (!chunk || !agx_bo_map(chunk)) {
         agx_bo_unreference(&dev->dev, chunk);
         goto out;
      }
      agx_bo_unreference(&dev->dev, heap->chunk);
      heap->chunk = chunk;
      at = 0;
   }

   heap->offset = at + size;
   *offset = at;
   bo = heap->chunk;
   agx_bo_reference(bo);

out:
   simple_mtx_unlock(&heap->lock);
   return bo;
}

static struct agx_bo *
hk_code_alloc_linked(struct agx_device *adev, size_t size, uint32_t *offset)
{
   return hk_code_alloc(container_of(adev, struct hk_device, dev), size,
                        offset);
}

/* Release before agx_close_device(), whose BO-cache eviction then frees the
 * object; the kernel's own reference keeps its copy of the pages valid. */
static void
hk_finish_vm_status_mirror(struct hk_device *dev)
{
   if (dev->vm_status_mirror.bo)
      agx_bo_unreference(&dev->dev, dev->vm_status_mirror.bo);
   dev->vm_status_mirror.bo = NULL;
   dev->vm_status_mirror.word = NULL;
}

static VkResult
hk_check_status(struct vk_device *device)
{
   struct hk_device *dev = container_of(device, struct hk_device, vk);
   if (dev->vm_status_mirror.word) {
      /* Acquire: the kernel stored the error before signalling the fence
       * this thread has already observed through its syncobj wait. */
      int32_t error = __atomic_load_n(dev->vm_status_mirror.word, __ATOMIC_ACQUIRE);
      if (error)
         return vk_device_set_lost(&dev->vk, "GPU VM %u failed: %s",
                                   dev->dev.vm_id, strerror(-error));
   } else if (dev->dev.async_errors) {
      /* virtio submits are asynchronous: the host counts the ones its kernel
       * rejected (a lost VM rejects every later submit) instead of failing them. */
      uint32_t errors = p_atomic_read(dev->dev.async_errors);
      if (errors != dev->async_errors_base)
         return vk_device_set_lost(&dev->vk, "host rejected %u virtio commands",
                                   errors - dev->async_errors_base);
   } else if (!dev->dev.is_virtio &&
              (dev->dev.params.features & DRM_ASAHI_FEATURE_VM_STATUS)) {
      struct drm_asahi_vm_status status = {.vm_id = dev->dev.vm_id};
      struct drm_asahi_get_params query = {
         .param_group = DRM_ASAHI_PARAM_GROUP_VM_STATUS,
         .pointer = (uintptr_t)&status,
         .size = sizeof(status),
      };
      /* A DRM syncobj can signal an error fence successfully. The kernel
       * publishes this sticky VM error before signaling such a fence. */
      if (drmIoctl(dev->dev.fd, DRM_IOCTL_ASAHI_GET_PARAMS, &query))
         return vk_device_set_lost(&dev->vk, "VM status query failed: %m");
      if (status.error)
         return vk_device_set_lost(&dev->vk, "GPU VM %u failed: %s",
                                    status.vm_id, strerror(-status.error));
   }
   return vk_check_printf_status(&dev->vk, &dev->dev.printf);
}

/* The common wait entrypoints already invoke check_status. Their polling
 * counterparts only inspect the syncobj, so cover those two paths here too.
 * Older kernels perform no additional ioctl; known-lost devices short circuit.
 */
VKAPI_ATTR VkResult VKAPI_CALL
hk_GetFenceStatus(VkDevice device, VkFence fence)
{
   VkResult result = vk_common_GetFenceStatus(device, fence);
   if (result < 0)
      return result;

   VK_FROM_HANDLE(hk_device, dev, device);
   VkResult status = vk_device_check_status(&dev->vk);
   return status == VK_SUCCESS ? result : status;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetSemaphoreCounterValue(VkDevice device, VkSemaphore semaphore,
                            uint64_t *value)
{
   VkResult result = vk_common_GetSemaphoreCounterValue(device, semaphore, value);
   if (result < 0)
      return result;

   VK_FROM_HANDLE(hk_device, dev, device);
   VkResult status = vk_device_check_status(&dev->vk);
   return status == VK_SUCCESS ? result : status;
}

static VkResult
hk_get_timestamp(struct vk_device *device, uint64_t *timestamp)
{
   struct hk_device *dev = container_of(device, struct hk_device, vk);
   *timestamp = agx_get_gpu_timestamp(&dev->dev);
   return VK_SUCCESS;
}

/*
 * Residency of one byte of a sparse residency buffer, read from that buffer's
 * live bitmap exactly the way hk_g17p_sparse_buffer_copy() and
 * lower_sparse_buffer_deref() read it: a header carrying the resource base and
 * the page count, followed by one bit per AIL page.  Reading it in the shader
 * rather than at record time is what lets a sparse bind submitted after the
 * command buffer was recorded still affect this dispatch.
 *
 * Every dereference is funnelled through the zero page when the value that
 * produced it is not trustworthy, so no branch is needed and no load can fault.
 */
static nir_def *
hk_g17p_page_resident(nir_builder *b, nir_def *safe_map, nir_def *has_map,
                      nir_def *resource_base, nir_def *page_count,
                      nir_def *addr)
{
   static_assert(AIL_PAGESIZE == (1u << 14), "update shader lookup");
   nir_def *page = nir_ushr_imm(b, nir_isub(b, addr, resource_base), 14);
   nir_def *in_range = nir_ult(b, page, nir_u2u64(b, page_count));
   nir_def *valid = nir_iand(b, has_map, in_range);

   nir_def *word_addr = nir_iadd_imm(
      b,
      nir_iadd(b, safe_map,
               nir_imul_imm(b, nir_ushr_imm(b, page, 5), sizeof(uint32_t))),
      sizeof(struct hk_sparse_buffer_map_header));
   nir_def *safe_word_addr = nir_bcsel(
      b, valid, word_addr, nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));
   nir_def *word = nir_load_global(b, 1, 32, safe_word_addr,
                                   .align_mul = sizeof(uint32_t));
   /* Test the bit with a variable MASK, not `(word >> (page & 31)) & 1`.
    * nir_opt_algebraic drops the `& 31` because NIR defines a shift to mask
    * its count, agx_nir_fuse_algebraic_late then folds the result into
    * ubitfield_extract(word, page, 1), and that pass runs before
    * agx_nir_lower_algebraic_late puts AGX's shift mask back -- so the raw
    * page number would reach the hardware as the extract offset.  Same shape,
    * same reason, as lower_sparse_buffer_deref() and
    * lower_sparse_buffer_texture(); an argument buffer past 512 KB would
    * otherwise read its residency from an arbitrary bit. */
   nir_def *mask =
      nir_ishl(b, nir_imm_int(b, 1), nir_iand_imm(b, nir_u2u32(b, page), 31));

   return nir_iand(b, in_range, nir_ine_imm(b, nir_iand(b, word, mask), 0));
}

/*
 * Residency of a whole VkDispatchIndirectCommand.
 *
 * The record is twelve bytes at a four-byte-aligned offset, so it can straddle
 * a page boundary: at offset 16376 or 16380 within a 16 KiB page its tail lives
 * in the next one.  Testing only the first page there would let the z or y
 * count come from the scratch page, which is exactly the arbitrary grid this
 * lookup exists to prevent, so test the last byte as well.  Where both bytes
 * share a page the second lookup reads the same word and costs one load.
 */
static nir_def *
hk_g17p_grid_helper_resident(nir_builder *b, nir_def *root, nir_def *grid_addr)
{
   nir_def *map_addr = nir_load_global(
      b, 1, 64,
      nir_iadd_imm(b, root,
                   hk_root_descriptor_offset(cs.grid_helper_sparse_map_addr)),
      .align_mul = 8);
   nir_def *has_map = nir_ine_imm(b, map_addr, 0);
   nir_def *safe_map = nir_bcsel(b, has_map, map_addr,
                                 nir_imm_int64(b, AGX_ZERO_PAGE_ADDRESS));

   nir_def *resource_base = nir_load_global(
      b, 1, 64,
      nir_iadd_imm(b, safe_map,
                   offsetof(struct hk_sparse_buffer_map_header,
                            resource_base)),
      .align_mul = alignof(uint64_t));
   nir_def *page_count = nir_load_global(
      b, 1, 32,
      nir_iadd_imm(b, safe_map,
                   offsetof(struct hk_sparse_buffer_map_header, page_count)),
      .align_mul = alignof(uint32_t));

   nir_def *first = hk_g17p_page_resident(b, safe_map, has_map, resource_base,
                                          page_count, grid_addr);
   nir_def *last = hk_g17p_page_resident(
      b, safe_map, has_map, resource_base, page_count,
      nir_iadd_imm(b, grid_addr, sizeof(VkDispatchIndirectCommand) - 1));

   /* No bitmap means this is not a sparse residency buffer, so every byte of
    * it is there to be read. */
   return nir_ior(b, nir_inot(b, has_map), nir_iand(b, first, last));
}

/*
 * The native G17P indirect-compute grid helper.
 *
 * A native indirect dispatch is two CDM records: a direct-shaped main launch
 * with bit 23 set, whose inline dimension words the hardware ignores, and a
 * compact helper launch naming a program plus a six-u32 geometry block.  The
 * compact record carries no dimensions and no argument table, so nothing in
 * the command stream can turn a runtime VkDispatchIndirectCommand into a
 * launch geometry except a program that reads it and writes the block.
 *
 * The contract this implements is agx3_g17p_grid_helper_core.h's, which was
 * grounded byte-exactly against Apple's own compiler: geometry[0..2] are
 * GLOBAL THREAD counts (threadgroups * local size), geometry[3..5] are the
 * local size.  The kernel is written in NIR rather than reusing Apple's
 * compiled image because that image's uniform ABI is not established, while
 * Honeykrisp's own root ABI is the one every other G17P compute dispatch
 * already uses.
 *
 * Every invocation writes the same six values, so the helper needs no
 * invocation guard and therefore no system values: the compact record does not
 * say how many threads it launches, and an idempotent kernel is correct for
 * any answer.
 */
static VkResult
hk_build_g17p_grid_helper(struct hk_device *dev,
                          const VkAllocationCallbacks *pAllocator,
                          bool copy_geometry, bool sparse,
                          struct hk_api_shader **helper)
{
   assert(!(copy_geometry && sparse));

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      copy_geometry ? "G17P indirect geometry copy"
      : sparse      ? "G17P indirect grid helper (sparse)"
                    : "G17P indirect grid helper");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *root = nir_load_root_agx(&b);
   nir_def *grid_addr = nir_load_global(
      &b, 1, 64,
      nir_iadd_imm(&b, root, hk_root_descriptor_offset(cs.group_count_addr)),
      .align_mul = 8);
   nir_def *geometry_addr = nir_load_global(
      &b, 1, 64,
      nir_iadd_imm(&b, root,
                   hk_root_descriptor_offset(cs.grid_helper_geometry_addr)),
      .align_mul = 8);
   nir_def *global, *local;

   if (copy_geometry) {
      /* An indirect-local grid is already a launch geometry, written by an
       * earlier kernel in the layout the main record consumes. */
      global = nir_load_global(&b, 3, 32, grid_addr, .align_mul = 4);
      local = nir_load_global(
         &b, 3, 32,
         nir_iadd_imm(&b, grid_addr, AGX3_G17P_INDIRECT_GEOMETRY_LOCAL_OFFSET),
         .align_mul = 4);
   } else {
      local = nir_load_global(
         &b, 3, 32,
         nir_iadd_imm(&b, root,
                      hk_root_descriptor_offset(cs.grid_helper_local_size)),
         .align_mul = 4);

      /* VkDispatchIndirectCommand is exactly three u32 threadgroup counts. */
      nir_def *groups = nir_load_global(&b, 3, 32, grid_addr, .align_mul = 4);

      /* sparseResidencyNonResidentStrict is advertised, so arguments in an
       * unbound page must read as zero.  The primary mapping sends those
       * reads to the scratch page instead, so substitute the zero grid here
       * rather than relying on the read-only shadow. */
      if (sparse) {
         nir_def *resident =
            hk_g17p_grid_helper_resident(&b, root, grid_addr);
         groups = nir_bcsel(&b, resident, groups, nir_imm_zero(&b, 3, 32));
      }

      global = nir_imul(&b, groups, local);
   }

   nir_store_global(&b, global, geometry_addr, .align_mul = 4);
   nir_store_global(&b, local,
                    nir_iadd_imm(&b, geometry_addr,
                                 AGX3_G17P_INDIRECT_GEOMETRY_LOCAL_OFFSET),
                    .align_mul = 4);

   struct vk_shader_compile_info info = {
      .nir = b.shader,
      .robustness = &vk_robustness_disabled,
      .stage = MESA_SHADER_COMPUTE,
   };
   return hk_compile_shader(dev, &info, NULL, NULL, pAllocator, helper);
}

static void
hk_destroy_g17p_grid_helpers(struct hk_device *dev,
                             const VkAllocationCallbacks *pAllocator)
{
   struct hk_api_shader **helpers[] = {
      &dev->g17p_grid_helper,
      &dev->g17p_geometry_copy_helper,
      &dev->g17p_grid_helper_sparse,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(helpers); ++i) {
      if (*helpers[i]) {
         hk_api_shader_destroy(&dev->vk, &(*helpers[i])->vk, pAllocator);
         *helpers[i] = NULL;
      }
   }
}

/* The helpers are internal shaders built from fixed NIR, like the empty
 * fragment shader: a failure to compile one is a driver bug or an allocation
 * failure, never a property of the application, so it fails device creation
 * instead of leaving a device that refuses every indirect dispatch at record
 * time. */
static VkResult
hk_init_g17p_grid_helper(struct hk_device *dev,
                         const VkAllocationCallbacks *pAllocator)
{
   if (!agx_chip_native_indirect_compute(dev->dev.chip))
      return VK_SUCCESS;

   VkResult result = hk_build_g17p_grid_helper(dev, pAllocator, false, false,
                                               &dev->g17p_grid_helper);
   if (result == VK_SUCCESS)
      result = hk_build_g17p_grid_helper(dev, pAllocator, true, false,
                                         &dev->g17p_geometry_copy_helper);

   /* Only a sparse residency buffer can carry a residency bitmap, so only a
    * device that enabled the feature can ever select this helper; skip
    * compiling it otherwise. */
   if (result == VK_SUCCESS && dev->vk.enabled_features.sparseResidencyBuffer)
      result = hk_build_g17p_grid_helper(dev, pAllocator, false, true,
                                         &dev->g17p_grid_helper_sparse);

   if (result == VK_SUCCESS)
      return VK_SUCCESS;

   hk_destroy_g17p_grid_helpers(dev, pAllocator);

   /* Running out of memory is reported as such. Any other compiler result
    * (FEATURE_NOT_PRESENT, UNKNOWN) describes the helper, not anything the
    * application asked for, and is not a vkCreateDevice result an
    * application can act on. */
   if (result == VK_ERROR_OUT_OF_HOST_MEMORY ||
       result == VK_ERROR_OUT_OF_DEVICE_MEMORY)
      return result;

   return vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                    "G17P indirect grid helper failed to compile: %s",
                    vk_Result_to_str(result));
}

/* Whether memory can be mapped at a second GPU virtual address at any time
 * in the life of `dev`, from what it enabled (hk_cdm_alias_scope()):
 *
 * - sparse buffers and images, and capture/replay buffers, have an address
 *   of their own, and the memory bound to them keeps its own as well;
 * - an fd import of memory from elsewhere gets an address of its own, and
 *   another import may map the same pages (two dma-bufs over one memfd, or
 *   a dma-buf and a host import of its pages);
 * - a host import gets an address of its own, and the same pages may be
 *   imported again, or be mapped by the application at a second virtual
 *   address that another import, or memory of another kind, reaches. With
 *   hk_host_memory_import_read_only every host import is read-only for the
 *   GPU, so the GPU writes no imported page and a read through the import
 *   cannot miss a write through another address.
 *
 * Nothing else maps memory the GPU writes at two addresses: the read-only
 * shadow of every allocation is read only through sparse and capture/replay
 * buffers, and the driver's other second mappings (descriptor tables,
 * acceleration structure tables, shader code) hold memory written by the
 * CPU alone. The structures of these features may only be used when they
 * are enabled, so a device that enabled none of them never has such memory.
 *
 * On M5 every resource through which these features map memory at a second
 * address is seen as it is created, so the scope follows the live ones
 * instead (host_alias.dynamic, hk_cdm_alias_scope()): buffers and image
 * planes with an address of their own and memory mapped again for
 * descriptor buffer capture/replay (va_alias_resources), fd and dma-buf
 * imports, and host imports that may alias (hk_host_import_aliases()). An
 * acceleration structure's address is its buffer's, also on replay, so its
 * capture/replay is covered by the buffer's. An import of private anonymous
 * memory of this process (malloc, aligned_alloc, an anonymous private mmap)
 * that overlaps no other import of the device has its pages at no other GPU
 * address: they are mapped at one virtual address of the process, and the
 * GPU reaches them only through the import's own, unless the application
 * moved an imported range with mremap and imported it again
 * (hk_host_alias.h).
 */
static bool
hk_device_may_alias_memory(const struct hk_device *dev,
                           const struct hk_instance *instance)
{
   const struct vk_features *f = &dev->vk.enabled_features;
   const struct vk_device_extension_table *ext = &dev->vk.enabled_extensions;

   if (f->sparseBinding || f->sparseResidencyBuffer ||
       f->sparseResidencyImage2D || f->sparseResidencyImage3D ||
       f->sparseResidency2Samples || f->sparseResidency4Samples ||
       f->sparseResidency8Samples || f->sparseResidency16Samples ||
       f->sparseResidencyAliased)
      return true;

   if (f->bufferDeviceAddressCaptureReplay ||
       f->bufferDeviceAddressCaptureReplayEXT ||
       f->descriptorBufferCaptureReplay ||
       f->accelerationStructureCaptureReplay)
      return true;

   if (ext->KHR_external_memory_fd || ext->EXT_external_memory_dma_buf)
      return true;

   return ext->EXT_external_memory_host &&
          !instance->drirc.misc.host_memory_import_read_only;
}

/* ASAHI_M5_HOST_ALIAS_TRACE=1: name at device creation every enabled feature
 * or extension through which memory can get a second GPU address
 * (hk_device_may_alias_memory()), and the rule the device took. */
static void
hk_device_trace_alias_reasons(const struct hk_device *dev,
                              const struct hk_instance *instance)
{
   const struct vk_features *f = &dev->vk.enabled_features;
   const struct vk_device_extension_table *ext = &dev->vk.enabled_extensions;
   char reasons[512] = "";
   size_t n = 0;
#define HK_ALIAS_REASON(cond, name)                                            \
   do {                                                                        \
      if ((cond) && n < sizeof(reasons))                                       \
         n += snprintf(reasons + n, sizeof(reasons) - n, " %s", name);         \
   } while (0)
   HK_ALIAS_REASON(f->sparseBinding, "sparseBinding");
   HK_ALIAS_REASON(f->sparseResidencyBuffer, "sparseResidencyBuffer");
   HK_ALIAS_REASON(f->sparseResidencyImage2D, "sparseResidencyImage2D");
   HK_ALIAS_REASON(f->sparseResidencyImage3D, "sparseResidencyImage3D");
   HK_ALIAS_REASON(f->sparseResidency2Samples || f->sparseResidency4Samples ||
                      f->sparseResidency8Samples || f->sparseResidency16Samples,
                   "sparseResidencyNSamples");
   HK_ALIAS_REASON(f->sparseResidencyAliased, "sparseResidencyAliased");
   HK_ALIAS_REASON(f->bufferDeviceAddressCaptureReplay ||
                      f->bufferDeviceAddressCaptureReplayEXT,
                   "bufferDeviceAddressCaptureReplay");
   HK_ALIAS_REASON(f->descriptorBufferCaptureReplay,
                   "descriptorBufferCaptureReplay");
   HK_ALIAS_REASON(f->accelerationStructureCaptureReplay,
                   "accelerationStructureCaptureReplay");
   HK_ALIAS_REASON(ext->KHR_external_memory_fd, "KHR_external_memory_fd");
   HK_ALIAS_REASON(ext->EXT_external_memory_dma_buf, "EXT_external_memory_dma_buf");
   HK_ALIAS_REASON(ext->EXT_external_memory_host &&
                      !instance->drirc.misc.host_memory_import_read_only,
                   "EXT_external_memory_host(writable)");
#undef HK_ALIAS_REASON
   fprintf(stderr, "M5_HOST_ALIAS device rule=%s may_alias=%u reasons:%s\n",
           dev->host_alias.dynamic ? "per-resource"
           : dev->may_alias_memory ? "per-device"
                                   : "none",
           dev->may_alias_memory, n ? reasons : " none");
}

/* A/B and fallback: the per-device rule, as before the scope followed the
 * live resources. */
DEBUG_GET_ONCE_BOOL_OPTION(m5_host_alias_static, "ASAHI_M5_HOST_ALIAS_STATIC",
                           false)

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateDevice(VkPhysicalDevice physicalDevice,
                const VkDeviceCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator, VkDevice *pDevice)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   VkResult result = VK_ERROR_OUT_OF_HOST_MEMORY;
   struct hk_device *dev;
   struct hk_instance *instance = (struct hk_instance *)pdev->vk.instance;

   dev = vk_zalloc2(&instance->vk.alloc, pAllocator, sizeof(*dev), 8,
                    VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!dev)
      return vk_error(pdev, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_device_dispatch_table dispatch_table;

   /* For secondary command buffer support, overwrite any command entrypoints
    * in the main device-level dispatch table with
    * vk_cmd_enqueue_unless_primary_Cmd*.
    */
   vk_device_dispatch_table_from_entrypoints(
      &dispatch_table, &vk_cmd_enqueue_unless_primary_device_entrypoints, true);

   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &hk_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(&dispatch_table,
                                             &wsi_device_entrypoints, false);

   /* Populate primary cmd_dispatch table */
   vk_device_dispatch_table_from_entrypoints(&dev->cmd_dispatch,
                                             &hk_device_entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(&dev->cmd_dispatch,
                                             &wsi_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(
      &dev->cmd_dispatch, &vk_common_device_entrypoints, false);

   simple_mtx_init(&dev->code_heap.lock, mtx_plain);
   dev->dev.code_alloc = hk_code_alloc_linked;

   result = vk_device_init(&dev->vk, &pdev->vk, &dispatch_table, pCreateInfo,
                           pAllocator);
   if (result != VK_SUCCESS)
      goto fail_alloc;

   dev->vk.shader_ops = &hk_device_shader_ops;
   dev->vk.command_dispatch_table = &dev->cmd_dispatch;
   dev->may_alias_memory = hk_device_may_alias_memory(dev, instance);
   simple_mtx_init(&dev->host_alias.lock, mtx_plain);
   simple_mtx_init(&dev->host_alias.cmd_lock, mtx_plain);
   list_inithead(&dev->host_alias.cmd_buffers);
   u_rwlock_init(&dev->host_alias.patch_lock);
   dev->host_alias.trace = agx_chip_g17_m5(pdev->dev.chip) &&
      debug_get_bool_option("ASAHI_M5_HOST_ALIAS_TRACE", false);
   /* A device whose dependencies always take memory scope has none to
    * decide per resource (agx_cdm_ordering_memory_scope()). */
   if (dev->may_alias_memory && agx_chip_g17_m5(pdev->dev.chip) &&
       !agx_cdm_ordering_memory_scope(&pdev->dev) &&
       !debug_get_option_m5_host_alias_static()) {
      /* The word hk_cdm_flush() records for a memory-scoped strong flush. */
      const struct agx3_cdm_barrier_info info = {
         .chip = pdev->dev.chip,
         .flags = 0x5,
         .wide_visibility = agx_chip_cdm_wide_visibility(pdev->dev.chip),
         .flush_memory = true,
      };
      uint8_t packed[AGX3_CDM_BARRIER_SIZE];
      if (agx3_cdm_barrier_pack(&info, packed)) {
         dev->host_alias.flush_word = agx3_cmdstream_load_le32(packed);
         dev->host_alias.dynamic = true;
         dev->may_alias_memory = false;
      }
   }
   if (dev->host_alias.trace)
      hk_device_trace_alias_reasons(dev, instance);

   drmDevicePtr drm_device = NULL;
   int ret = drmGetDeviceFromDevId(pdev->render_dev, 0, &drm_device);
   if (ret != 0) {
      result = vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                         "Failed to get DRM device: %m");
      goto fail_init;
   }

   const char *path = drm_device->nodes[DRM_NODE_RENDER];
   dev->dev.fd = open(path, O_RDWR | O_CLOEXEC);
   if (dev->dev.fd < 0) {
      result = vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                         "failed to open device %s", path);
      drmFreeDevice(&drm_device);
      goto fail_init;
   }

   dev->perftest = debug_get_flags_option("HK_PERFTEST", hk_perf_options, 0);
   dev->dev.compute_timeout_ms = instance->drirc.misc.compute_timeout_ms;
   /* Bounded poll before blocking syncobj waits (see vk_device.sync_wait_poll_us).
    * Off by default; pays CPU to skip wake-up latency on short submit->signal
    * round trips. */
   dev->vk.sync_wait_poll_us =
      CLAMP(debug_get_num_option("HK_SUBMIT_POLL_US", 0), 0, 1000000);

   /* The default AGX2 CDM barrier word on this chip: the reduced 0x1ff on
    * the dies it was measured on (agx2_cdm_barrier_default_word()), the
    * full word elsewhere. fullcdmbarrier and a device that created a sparse
    * resource upgrade to the full word in hk_cdm_flush().
    */
   dev->cdm_barrier_default_mask =
      agx2_cdm_barrier_default_word(pdev->dev.chip,
                                    pdev->dev.params.gpu_generation);

   /* Native graphics compiles the emulation in only once a custom border
    * sampler exists, and most samples then pay one branch, so D3D content keeps
    * its correct borders there. */
   if (instance->drirc.misc.disable_border_emulation &&
       !agx_chip_native_agx3(pdev->dev.chip)) {
      dev->perftest |= HK_PERF_NOBORDER;
   }

   if (HK_PERF(dev, NOROBUST)) {
      dev->vk.enabled_features.robustBufferAccess = false;
      dev->vk.enabled_features.robustBufferAccess2 = false;
      dev->vk.enabled_features.robustImageAccess = false;
      dev->vk.enabled_features.robustImageAccess2 = false;
      dev->vk.enabled_features.pipelineRobustness = false;
   }

   /* All BOs use the single UMA heap. Cached BOs remain pinned after
    * vkFreeMemory, so include their bytes across every logical device. */
   assert(pdev->mem_heap_count == 1);
   dev->dev.bo_cache.size_counter = pdev->mem_heaps[0].cached;
   bool succ = agx_open_device(NULL, &dev->dev);
   drmFreeDevice(&drm_device);
   if (!succ) {
      result = vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED,
                         "Failed to get DRM device: %m");
      goto fail_fd;
   }

   /* The kernel's feature word, read when the device was opened: queues copy
    * this into their submission tracking when they are created. */
   dev->load_hazard_frag = dev->dev.params.features &
                           DRM_ASAHI_FEATURE_ORDERED_PARTIAL_RENDERS;

   /* Splitting a long submission lets the GPU start work from the first ioctl
    * while the kernel validates later ioctls, so a profile may cap batch size. */
   dev->submit_batch_size = instance->drirc.performance.submit_batch_size;

   vk_device_set_drm_fd(&dev->vk, dev->dev.fd);
   hk_gpu_waits_init(dev);
   dev->vk.copy_sync_payloads = dev->gpu_waits.enabled
      ? hk_gpu_waits_copy_sync_payloads
      : vk_drm_syncobj_copy_payloads;
   dev->vk.command_buffer_ops = &hk_cmd_buffer_ops;
   dev->vk.check_status = hk_check_status;
   dev->vk.get_timestamp = hk_get_timestamp;
   hk_init_vm_status_mirror(dev);
   if (dev->dev.async_errors)
      dev->async_errors_base = p_atomic_read(dev->dev.async_errors);

   /* This holds for current platforms. We do not currently implement
    * timestamp scaling, this would require changes in the query copy kernel
    * as well. Calibrated timestamps depends on this.
    */
   assert(dev->dev.user_timestamp_to_ns.num ==
             dev->dev.user_timestamp_to_ns.den &&
          "user timestamps are in ns");

   if ((agx_chip_g17_m5(dev->dev.chip) ||
        hk_sparse_texel_residency_from_map(&dev->dev)) &&
       dev->vk.enabled_features.sparseResidencyBuffer) {
      const uint64_t pages = dev->dev.main_heap_size / AIL_PAGESIZE;
      const uint64_t bytes = align64(16 + DIV_ROUND_UP(pages, 8), AIL_PAGESIZE);
      dev->physical_sparse_map = agx_bo_create(
         &dev->dev, bytes, AIL_PAGESIZE,
         AGX_BO_WRITEBACK | AGX_BO_ZERO_ON_REUSE,
         "Physical sparse buffer page map");
      if (!dev->physical_sparse_map || !agx_bo_map(dev->physical_sparse_map)) {
         result = vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         goto fail_dev;
      }
      memset(agx_bo_map(dev->physical_sparse_map), 0, bytes);
      if (agx_bo_sync_for_device(dev->physical_sparse_map, 0, bytes)) {
         result = vk_error(dev, VK_ERROR_DEVICE_LOST);
         goto fail_dev;
      }
   }

   /* Capture-replay buffers reserve their VA and bind the fill page until
    * memory is bound, like sparse buffers, so they need it too. */
   if (dev->vk.enabled_features.sparseBinding ||
       dev->vk.enabled_features.bufferDeviceAddressCaptureReplay) {
      dev->sparse_fill = agx_bo_create(&dev->dev, AIL_PAGESIZE, 0, 0,
                                       "Sparse fill page");
      if (!dev->sparse_fill || !agx_bo_map(dev->sparse_fill)) {
         result = vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         goto fail_dev;
      }
      memset(agx_bo_map(dev->sparse_fill), 0, AIL_PAGESIZE);
      if (agx_bo_sync_for_device(dev->sparse_fill, 0, AIL_PAGESIZE)) {
         result = vk_error(dev, VK_ERROR_DEVICE_LOST);
         goto fail_dev;
      }
   }

   result = hk_init_sampler_heap(dev, &dev->samplers);
   if (result != VK_SUCCESS)
      goto fail_dev;

   if (agx_chip_agx3_path(dev->dev.chip)) {
      /* Capacity stays HK_MAX_DESCRIPTORS, but memory is added a chunk at a
       * time as image and buffer views are created. */
      result = hk_descriptor_table_init(dev, &dev->sampled_images, 32,
                                        HK_DESCRIPTOR_TABLE_CHUNK_B / 32,
                                        HK_MAX_DESCRIPTORS);
      if (result != VK_SUCCESS)
         goto fail_samplers;

      struct hk_native_texture_descriptor null_texture;
      hk_set_null_native_texture(agx_chip_descriptor_generation(dev->dev.chip),
                                 &null_texture);
      uint32_t null_index;
      result = hk_descriptor_table_add(dev, &dev->sampled_images,
                                       &null_texture, 32, &null_index);
      if (result != VK_SUCCESS)
         goto fail_sampled_images;
      assert(null_index == 0);

      /* Reserved right after the null texture, so their indices are the
       * same on every device with the same features. */
      if (dev->vk.enabled_features.descriptorBuffer) {
         result = hk_texel_windows_init(dev);
         if (result != VK_SUCCESS)
            goto fail_sampled_images;
      }
   }

   result = hk_descriptor_table_init(
      dev, &dev->occlusion_queries, sizeof(uint64_t), AGX_MAX_OCCLUSION_QUERIES,
      AGX_MAX_OCCLUSION_QUERIES);
   if (result != VK_SUCCESS)
      goto fail_sampled_images;

   result = hk_upload_rodata(dev);
   if (result != VK_SUCCESS)
      goto fail_queries;

   /* XXX: error handling, and should this even go on the device? */
   agx_bg_eot_init(&dev->bg_eot, &dev->dev);
   if (!dev->bg_eot.ht) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail_rodata;
   }
   hk_g17p_bg_eot_dev_cache_init(dev);

   /* Geometry and tessellation are advertised core functionality on G17P and
    * M5. Upload their immutable internal kernels with the device instead of
    * making the first vkCmdDraw that needs each kernel serialize on the
    * cache lock and allocate its executable image. */
   if (agx_chip_native_or_m5(dev->dev.chip) &&
       !hk_g17p_prewarm_poly_kernels(dev)) {
      result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
      goto fail_bg_eot;
   }

   result = hk_init_internal_shaders(&dev->prolog_epilog);
   if (result != VK_SUCCESS)
      goto fail_bg_eot;

   result = hk_init_internal_shaders(&dev->kernels);
   if (result != VK_SUCCESS)
      goto fail_internal_shaders;

   for (unsigned i = 0; i < pCreateInfo->queueCreateInfoCount; i++) {
      for (unsigned q = 0; q < pCreateInfo->pQueueCreateInfos[i].queueCount;
           q++) {
         result = hk_queue_init(dev, &pCreateInfo->pQueueCreateInfos[i], q);
         if (result != VK_SUCCESS)
            goto fail_queues;
      }
   }

   struct vk_pipeline_cache_create_info cache_info = {
      .weak_ref = true,
   };
   dev->vk.mem_cache = vk_pipeline_cache_create(&dev->vk, &cache_info, NULL);
   if (dev->vk.mem_cache == NULL) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail_queues;
   }

   result = hk_device_init_accel_struct_state(dev);
   if (result != VK_SUCCESS)
      goto fail_mem_cache;

   result = hk_device_init_meta(dev);
   if (result != VK_SUCCESS)
      goto fail_accel_struct;

   /* Precompile an empty fragment shader that can be used to handle API-level
    * null fragment shaders. We do this at device-time to make binds cheap.
    * Regardless, compiling this shader should be fast.
    */
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                                  &agx_nir_options, "empty FS");
   struct vk_shader_compile_info info = {
      .nir = b.shader,
      .robustness = &vk_robustness_disabled,
      .stage = MESA_SHADER_FRAGMENT,
   };
   hk_compile_shader(dev, &info, NULL, NULL, pAllocator, &dev->null_fs);
   if (!dev->null_fs) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto fail_meta;
   }

   result = hk_init_g17p_grid_helper(dev, pAllocator);
   if (result != VK_SUCCESS)
      goto fail_null_fs;

   *pDevice = hk_device_to_handle(dev);

   hk_cmd_pending_init(dev);
   hk_query_stamps_init(dev);
   hk_events_init(dev);
   simple_mtx_init(&dev->vdm.lock, mtx_plain);
   for (unsigned class = 0; class < HK_CMD_SLAB_COUNT; ++class) {
      list_inithead(&dev->vdm.slabs[class]);
      list_inithead(&dev->vdm.full[class]);
   }
   simple_mtx_init(&dev->scratch.lock, mtx_plain);
   simple_mtx_init(&dev->heap_lock, mtx_plain);
   simple_mtx_init(&dev->sparse_buffers.lock, mtx_plain);
   list_inithead(&dev->sparse_buffers.list);
   simple_mtx_init(&dev->shader_binary_cache.lock, mtx_plain);
   agx_scratch_init(&dev->dev, &dev->scratch.vs);
   agx_scratch_init(&dev->dev, &dev->scratch.fs);
   agx_scratch_init(&dev->dev, &dev->scratch.cs);

   u_rwlock_init(&dev->external_bos.lock);
   dev->external_bos.counts = UTIL_DYNARRAY_INIT;
   dev->external_bos.list = UTIL_DYNARRAY_INIT;

   /* Opt-in profiler; never fails device creation. */
   hk_frame_anatomy_init(dev);
   hk_compute_kick_init(dev, instance->drirc.misc.compute_kick_ms);

   return VK_SUCCESS;

fail_null_fs:
   hk_api_shader_destroy(&dev->vk, &dev->null_fs->vk, pAllocator);
fail_meta:
   if (!agx_chip_g17_m5(dev->dev.chip))
      hk_gen1_rq_finish(dev);
   hk_device_finish_meta(dev);
fail_accel_struct:
   hk_device_finish_accel_struct_state(dev);
fail_mem_cache:
   vk_pipeline_cache_destroy(dev->vk.mem_cache, NULL);
fail_queues:
   /* Aliases were created after their backing queues. Finish
    * them first so a submit worker cannot refer to a freed backing queue. */
   while (!list_is_empty(&dev->vk.queues)) {
      struct vk_queue *iter =
         list_last_entry(&dev->vk.queues, struct vk_queue, link);
      struct hk_queue *queue = container_of(iter, struct hk_queue, vk);
      hk_queue_finish(dev, queue);
   }
   hk_destroy_internal_shaders(dev, &dev->kernels, false);
fail_internal_shaders:
   hk_destroy_internal_shaders(dev, &dev->prolog_epilog, true);
fail_bg_eot:
   /* Prewarming may have populated only part of this cache. */
   hk_g17p_compute_program_cache_finish(dev);
   hk_g17p_bg_eot_dev_cache_finish(dev);
   agx_bg_eot_cleanup(&dev->bg_eot);
fail_rodata:
   agx_bo_unreference(&dev->dev, dev->rodata.heap_bo);
   agx_bo_unreference(&dev->dev, dev->rodata.indexed_heap_pointers_bo);
   agx_bo_unreference(&dev->dev, dev->rodata.bo);
fail_queries:
   hk_descriptor_table_finish(dev, &dev->occlusion_queries);
fail_sampled_images:
   hk_texel_windows_finish(dev);
   if (agx_chip_agx3_path(dev->dev.chip))
      hk_descriptor_table_finish(dev, &dev->sampled_images);
fail_samplers:
   hk_destroy_sampler_heap(dev, &dev->samplers);
fail_dev:
   hk_finish_vm_status_mirror(dev);
   agx_bo_unreference(&dev->dev, dev->physical_sparse_map);
   agx_bo_unreference(&dev->dev, dev->sparse_fill);
   agx_bo_unreference(&dev->dev, dev->code_heap.chunk);
   agx_close_device(&dev->dev);
   /* agx_close_device owns the DRM fd after agx_open_device succeeds. */
   hk_gpu_waits_finish(dev);
   goto fail_init;
fail_fd:
   close(dev->dev.fd);
fail_init:
   vk_device_finish(&dev->vk);
fail_alloc:
   vk_free(&dev->vk.alloc, dev);
   return result;
}

/* At most this long, for all queues together, at vkDestroyDevice. */
#define HK_DESTROY_WAIT_NS (5ull * 1000000000ull)

VKAPI_ATTR void VKAPI_CALL
hk_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, _device);

   if (!dev)
      return;

   /* Its thread and kicks use the queues and the device's kernels. */
   hk_compute_kick_stop(dev);

   /* Freeing the device's memory below unmaps it from the GPU at once, so let
    * the GPU finish what the queues submitted first, including work the
    * application did not wait for. Bounded, so a submission that cannot
    * complete does not hang teardown. */
   hk_device_wait_submitted(dev, HK_DESTROY_WAIT_NS);

   util_dynarray_fini(&dev->external_bos.counts);
   util_dynarray_fini(&dev->external_bos.list);
   u_rwlock_destroy(&dev->external_bos.lock);

   if (!agx_chip_g17_m5(dev->dev.chip))
      hk_gen1_rq_finish(dev);
   hk_device_finish_meta(dev);
   hk_shader_binary_cache_finish(dev);
   hk_g17p_compute_program_cache_finish(dev);
   hk_device_finish_accel_struct_state(dev);
   hk_rt_device_finish(dev);
   hk_destroy_internal_shaders(dev, &dev->kernels, false);
   hk_destroy_internal_shaders(dev, &dev->prolog_epilog, true);

   vk_pipeline_cache_destroy(dev->vk.mem_cache, NULL);

   /* Retire pending profiler records while the queue syncobjs exist. */
   hk_frame_anatomy_finish(dev);

   /* Aliases were created after their backing queues. Finish
    * them first so a submit worker cannot refer to a freed backing queue. */
   while (!list_is_empty(&dev->vk.queues)) {
      struct vk_queue *iter =
         list_last_entry(&dev->vk.queues, struct vk_queue, link);
      struct hk_queue *queue = container_of(iter, struct hk_queue, vk);
      hk_queue_finish(dev, queue);
   }

   /* No submit thread is left to call into the kicker. */
   hk_compute_kick_finish(dev);

   vk_device_finish(&dev->vk);
   hk_gpu_waits_finish(dev);

   /* All submit workers/queues are gone. No command bytes can be handed to
    * a new recording during teardown. Kernel job pins retain failed work. */
   hk_cmd_pending_finish(dev);
   simple_mtx_destroy(&dev->cmd_pending_lock);
   hk_events_finish(dev);
   hk_query_stamps_finish(dev);
   hk_device_finish_vdm_slabs(dev);
   simple_mtx_destroy(&dev->vdm.lock);

   agx_scratch_fini(&dev->scratch.vs);
   agx_scratch_fini(&dev->scratch.fs);
   agx_scratch_fini(&dev->scratch.cs);
   simple_mtx_destroy(&dev->scratch.lock);
   simple_mtx_destroy(&dev->sparse_buffers.lock);

   if (dev->null_fs) {
      hk_api_shader_destroy(&dev->vk, &dev->null_fs->vk, pAllocator);
   }

   hk_destroy_g17p_grid_helpers(dev, pAllocator);

   hk_destroy_sampler_heap(dev, &dev->samplers);
   hk_texel_windows_finish(dev);
   if (agx_chip_agx3_path(dev->dev.chip))
      hk_descriptor_table_finish(dev, &dev->sampled_images);
   hk_descriptor_table_finish(dev, &dev->occlusion_queries);
   agx_bo_unreference(&dev->dev, dev->rodata.heap_bo);
   agx_bo_unreference(&dev->dev, dev->rodata.indexed_heap_pointers_bo);
   agx_bo_unreference(&dev->dev, dev->rodata.bo);
   for (unsigned s = 0; s < HK_HEAP_SLOTS; ++s)
      agx_bo_unreference(&dev->dev, dev->heap_slots[s]);
   simple_mtx_destroy(&dev->heap_lock);
   hk_g17p_bg_eot_dev_cache_finish(dev);
   agx_bg_eot_cleanup(&dev->bg_eot);
   hk_finish_vm_status_mirror(dev);
   agx_bo_unreference(&dev->dev, dev->physical_sparse_map);
   agx_bo_unreference(&dev->dev, dev->sparse_fill);
   agx_bo_unreference(&dev->dev, dev->code_heap.chunk);
   simple_mtx_destroy(&dev->code_heap.lock);
   hk_host_imports_fini(&dev->host_alias.imports);
   simple_mtx_destroy(&dev->host_alias.lock);
   simple_mtx_destroy(&dev->host_alias.cmd_lock);
   u_rwlock_destroy(&dev->host_alias.patch_lock);
   agx_close_device(&dev->dev);
   vk_free(&dev->vk.alloc, dev);
}

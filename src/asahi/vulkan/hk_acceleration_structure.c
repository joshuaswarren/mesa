/*
 * Copyright © 2021 Google
 * Copyright © 2023 Valve Corporation
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * VK_KHR_acceleration_structure for Honeykrisp.
 *
 * The heavy lifting -- Morton sorting, PLOC/HPLOC or LBVH
 * construction -- is Mesa's common builder, which produces a driver-neutral
 * binary tree in scratch memory.  All this file adds is the conversion of
 * that tree into the node format the software traversal reads (hk_bvh.h),
 * which happens in one compute dispatch per structure, plus the entry points
 * and the handful of vk_device hooks the common builder calls back into.
 */

#include "hk_acceleration_structure.h"
#include "hk_host_as.h"
#include "hk_agx3_as_format.h"
#include "hk_m5_acceleration_structure.h"
#include "asahi/compiler/agx3_rt_gen1_default.h"
#include "asahi/compiler/agx3_rt_gen1_triangle.h"
#include "asahi/compiler/agx3_rt_gen1_aabb.h"
#include "asahi/compiler/agx3_rt_gen1_instance_push.h"
#include "asahi/compiler/agx3_rt_gen1_clique.h"
#include "asahi/compiler/agx3_rt_gen1_runtime_tables.h"
#include "util/u_dl.h"
#include "util/cache_ops.h"
#include "hk_agx3_as_layout.h"
#include "util/hash_table.h"
#include "asahi/compiler/agx_nir_gen1_ray_query.h"
#include "hk_gen1_rt_indirect_contract.h"
#include "asahi/lib/agx_va_runtime_hole.h"
#include "asahi/libagx/gen1_as.h"
#include "util/u_debug.h"

#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "hk_queue.h"
#include "vk_format.h"

#include "vk_acceleration_structure.h"
#include "vk_common_entrypoints.h"
#include "vk_meta.h"

#include "bvh/hk_build_interface.h"

#include "util/u_math.h"

static const uint32_t encode_spv[] = {
#include "bvh/hk_encode.spv.h"
};

static const uint32_t write_spv[] = {
#include "bvh/hk_bvh_write.spv.h"
};

static const uint32_t vertex_table_spv[] = {
#include "bvh/hk_vertex_table.spv.h"
};

struct hk_vertex_table_layout {
   uint64_t offset, bytes;
   uint32_t geometry_count, primitives, vertices_offset;
};

struct hk_vertex_table_args {
   uint64_t leaves, table;
   uint32_t leaf_count, geometry_count, vertices_offset, leaf_stride;
};

static const uint32_t agx3_node_spv[] = {
#include "bvh/hk_agx3_gen1_leaf.spv.h"
};

static const uint32_t agx3_plan_pairs_spv[] = {
#include "bvh/hk_agx3_gen1_plan_pairs.spv.h"
};

static const uint32_t agx3_node_pairs_spv[] = {
#include "bvh/hk_agx3_gen1_leaf_pairs.spv.h"
};

static const uint32_t agx3_envelope_spv[] = {
#include "bvh/hk_agx3_gen1_envelope.spv.h"
};

static const uint32_t agx3_plan_spv[] = {
#include "bvh/hk_agx3_gen1_plan.spv.h"
};

static const uint32_t agx3_plan_fused_spv[] = {
#include "bvh/hk_agx3_gen1_plan_fused.spv.h"
};

static const uint32_t agx3_plan_fused_pairs_spv[] = {
#include "bvh/hk_agx3_gen1_plan_fused_pairs.spv.h"
};

static const uint32_t agx3_refit_spv[] = {
#include "bvh/hk_agx3_refit.spv.h"
};

/* The update phases that run per level or per job, each on its own. */
static const uint32_t agx3_refit_propagate_spv[] = {
#include "bvh/hk_agx3_refit_propagate.spv.h"
};
static const uint32_t agx3_refit_inner_spv[] = {
#include "bvh/hk_agx3_refit_inner.spv.h"
};

static const uint32_t agx3_refit_encode_spv[] = {
#include "bvh/hk_agx3_refit_encode.spv.h"
};

/* The refit encoder for triangle structures only. */
static const uint32_t agx3_refit_encode_triangles_spv[] = {
#include "bvh/hk_agx3_refit_encode_triangles.spv.h"
};

/* The node encoder for instance structures only. */
static const uint32_t agx3_node_instances_spv[] = {
#include "bvh/hk_agx3_gen1_leaf_instances.spv.h"
};

static const uint32_t agx3_refit_leaf_spv[] = {
#include "bvh/hk_agx3_refit_leaf.spv.h"
};

static void hk_flush_buffer_write_cp(VkCommandBuffer);
static VkResult hk_host_as_init(struct hk_device *dev);
static void hk_agx3_refit_save(VkCommandBuffer,
                               const struct vk_acceleration_structure_build_state *);
static void hk_agx3_refit_phase(VkCommandBuffer,
   const struct vk_acceleration_structure_build_state *, unsigned, unsigned);
static VkResult hk_cmd_refit_agx3(VkCommandBuffer, struct vk_device *,
   struct vk_meta_device *, const struct vk_acceleration_structure_build_args *,
   const struct vk_acceleration_structure_build_state *, uint32_t);
static bool hk_agx3_refit_skips_leaves(const struct hk_device *,
   const struct vk_acceleration_structure_build_state *);

/* Allow a million distinct BLASes plus a TLAS and the reserved query IDs. */
#define HK_GEN1_AS_ID_LIMIT (1u << 21)
#define HK_GEN1_SPARSE_PAGE_BYTES 0x4000u

static uint32_t
hk_gen1_as_index_to_id(uint32_t index)
{
   const uint32_t id = index + 1;
   return id < HK_GEN1_AS_TABLE_AS_SLOTS ? id : id + HK_GEN1_RQ_PHYSICAL_SLOTS;
}

static uint32_t
hk_gen1_as_id_to_index(uint32_t id)
{
   assert(id && (id < HK_GEN1_AS_TABLE_AS_SLOTS ||
                 id >= HK_GEN1_AS_TABLE_ENTRIES));
   return id - 1 - (id >= HK_GEN1_AS_TABLE_ENTRIES ?
                      HK_GEN1_RQ_PHYSICAL_SLOTS : 0);
}

/* Caller holds gen1_as_lock. No allocation, search or failure on teardown. */
static void
hk_gen1_as_free_slot(struct hk_device *dev, uint32_t index)
{
   struct hk_gen1_as_slot *slot =
      util_dynarray_element(&dev->gen1_ases, struct hk_gen1_as_slot, index);
   slot->as = NULL;
   slot->next_free = dev->gen1_as_free_head;
   dev->gen1_as_free_head = index;
}

static bool
hk_gen1_sparse_init(struct hk_device *dev, struct hk_gen1_sparse_table *t,
                    uint32_t entry_bytes, uint64_t fixed_va,
                    enum agx_bo_flags bo_flags, uint32_t bind_flags,
                    const char *label)
{
   *t = (struct hk_gen1_sparse_table){
      .entry_bytes = entry_bytes, .bo_flags = bo_flags,
      .bind_flags = bind_flags, .label = label,
   };
   util_dynarray_init(&t->pages, NULL);
   t->va = agx_va_alloc(&dev->dev, (uint64_t)HK_GEN1_AS_ID_LIMIT * entry_bytes,
                        HK_GEN1_SPARSE_PAGE_BYTES,
                        fixed_va ? AGX_VA_USC | AGX_VA_FIXED : 0, fixed_va);
   return t->va != NULL;
}

static void
hk_gen1_sparse_finish(struct hk_device *dev, struct hk_gen1_sparse_table *t)
{
   unsigned i = 0;
   util_dynarray_foreach(&t->pages, struct agx_bo *, bo) {
      if (t->va)
         agx_bo_bind(&dev->dev, NULL,
                     t->va->addr + (uint64_t)i * HK_GEN1_SPARSE_PAGE_BYTES,
                     HK_GEN1_SPARSE_PAGE_BYTES, 0, DRM_ASAHI_BIND_UNBIND);
      agx_bo_unreference(&dev->dev, *bo);
      ++i;
   }
   util_dynarray_fini(&t->pages);
   if (t->va)
      agx_va_free(&dev->dev, t->va, false);
   t->va = NULL;
}

static void *
hk_gen1_sparse_entry(struct hk_device *dev, struct hk_gen1_sparse_table *t,
                     uint32_t id)
{
   if (!t->va || id >= HK_GEN1_AS_ID_LIMIT)
      return NULL;
   const uint32_t per_page = HK_GEN1_SPARSE_PAGE_BYTES / t->entry_bytes;
   const uint32_t page = id / per_page;
   while (util_dynarray_num_elements(&t->pages, struct agx_bo *) <= page) {
      const unsigned next = util_dynarray_num_elements(&t->pages, struct agx_bo *);
      struct agx_bo *bo = agx_bo_create(&dev->dev, HK_GEN1_SPARSE_PAGE_BYTES,
                                      HK_GEN1_SPARSE_PAGE_BYTES, t->bo_flags,
                                      t->label);
      if (!bo)
         return NULL;
      memset(agx_bo_map(bo), 0, HK_GEN1_SPARSE_PAGE_BYTES);
      util_post_flush_fence();
      struct agx_bo **slot = util_dynarray_grow(&t->pages, struct agx_bo *, 1);
      if (!slot) {
         agx_bo_unreference(&dev->dev, bo);
         return NULL;
      }
      if (agx_bo_bind(&dev->dev, bo,
                      t->va->addr + (uint64_t)next * HK_GEN1_SPARSE_PAGE_BYTES,
                      HK_GEN1_SPARSE_PAGE_BYTES, 0, t->bind_flags)) {
         (void)util_dynarray_pop_ptr(&t->pages, struct agx_bo *);
         agx_bo_unreference(&dev->dev, bo);
         return NULL;
      }
      *slot = bo;
   }
   struct agx_bo *bo = *util_dynarray_element(&t->pages, struct agx_bo *, page);
   return (uint8_t *)agx_bo_map(bo) + (uint64_t)(id % per_page) * t->entry_bytes;
}

static uint32_t
hk_gen1_as_id(struct hk_device *dev, struct vk_acceleration_structure *as)
{
   uint32_t result = 0;
   simple_mtx_lock(&dev->gen1_as_lock);
   if (dev->gen1_as_ids) {
      struct hash_entry *entry = _mesa_hash_table_search(dev->gen1_as_ids, as);
      if (entry)
         result = (uint32_t)(uintptr_t)entry->data;
   }
   simple_mtx_unlock(&dev->gen1_as_lock);
   return result;
}

static uint64_t
hk_gen1_as_metadata_addr_for_id(struct hk_cmd_buffer *cmd, uint32_t id)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   void *entry = NULL;
   if (dev->gen1_as_lock_ready && id) {
      simple_mtx_lock(&dev->gen1_as_lock);
      entry = hk_gen1_sparse_entry(dev, &dev->gen1_as_metadata_pages, id);
      simple_mtx_unlock(&dev->gen1_as_lock);
   }
   if (!entry) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return 0;
   }
   return dev->gen1_as_metadata_pages.va->addr +
          (uint64_t)id * LIBAGX_GEN1_AS_METADATA_STRIDE;
}

uint64_t
hk_gen1_as_metadata_addr(struct hk_cmd_buffer *cmd,
                         struct vk_acceleration_structure *as)
{
   return hk_gen1_as_metadata_addr_for_id(cmd,
      hk_gen1_as_id(hk_cmd_buffer_device(cmd), as));
}

/* address is the structure's own address (also its Vulkan device address);
 * base is the traversal base the AS table hands to the hardware. */
static void
hk_gen1_as_publish_address(struct hk_device *dev, uint32_t id, uint64_t address,
                           uint64_t base)
{
   uint64_t *slot = hk_gen1_sparse_entry(dev, &dev->gen1_as_address_pages, id);
   uint64_t *table = hk_gen1_sparse_entry(dev, &dev->gen1_as_table_pages, id);
   if (slot)
      __atomic_store_n(slot, address, __ATOMIC_RELEASE);
   if (table)
      __atomic_store_n(table, base, __ATOMIC_RELEASE);
   util_post_flush_fence();
}

static uint64_t
hk_gen1_as_bound_va(const struct vk_acceleration_structure *as)
{
   uint64_t address = as->buffer ? as->buffer->device_address : as->addr;
   if (!address)
      return 0;
   return address + (as->buffer ? as->offset : 0);
}

/* ---------------------------------------------------------------- */
/* Traversal bases for structures that are not 16 KiB-aligned        */

/*
 * G17P reads a structure's BVH base from its AS table entry with the low 14
 * address bits ignored, starts traversal at the base's first slot and follows
 * forward 32-bit slot references from the base (libagx/gen1_as.h). Vulkan
 * places a structure at any 256-byte buffer offset, so a structure that is
 * not 16 KiB-aligned is given a base from a window:
 *
 * - a window is 256 consecutive 16 KiB pages of address space;
 * - pages 0..254 all map one device-wide object of 255 pages, whose page j
 *   holds a one-child node forwarding to slot j of page 255;
 * - page 255 maps the window's own trampoline page, whose slot j holds a
 *   one-child node forwarding to the root of the structure based at page j.
 *
 * Each such structure costs one 64-byte trampoline and 16 KiB of address
 * space, and its image stays where Vulkan placed it; its child references
 * carry the slot distance from its base (hk_gen1_as_slot_bias()). A
 * structure that is 16 KiB-aligned is its own base, as before. Windows are
 * allocated from the bottom of the address space because a structure must
 * lie above its base.
 */
#define HK_GEN1_AS_WINDOW_PAGE ((uint64_t)LIBAGX_GEN1_AS_BASE_ALIGNMENT)
#define HK_GEN1_AS_WINDOW_BASES 255u
#define HK_GEN1_AS_WINDOW_BYTES ((HK_GEN1_AS_WINDOW_BASES + 1) * HK_GEN1_AS_WINDOW_PAGE)
#define HK_GEN1_AS_PAGE_SLOTS (HK_GEN1_AS_WINDOW_PAGE / LIBAGX_GEN1_AS_SLOT_BYTES)
#define HK_GEN1_AS_REACH ((UINT64_C(1) << 32) * LIBAGX_GEN1_AS_SLOT_BYTES)

/* A one-child internal node whose box covers [-2^126, 2^126] on each axis:
 * base -2^126 (upper half 0xfe80), exponent 254 (a 2^127 grid), child cells
 * 0..1, in the encoding agx3_leaf.comp uses for internal nodes. */
static void
hk_gen1_as_forward_node(uint32_t *node, uint32_t child)
{
   const uint64_t words[8] = {
      0xa0u | (254u << 8) | (UINT64_C(1) << 36) | (UINT64_C(1) << 29) |
         (UINT64_C(1) << 22),
      0xfe80, 0xfe80, 0xfe80, (254u << 8) | 254u,
      child >> 16, child & 0xffffu, 0x1f00u | 1u,
   };
   for (unsigned i = 0; i < 8; ++i) {
      node[2 * i] = (uint32_t)words[i];
      node[2 * i + 1] = (uint32_t)(words[i] >> 32);
   }
}

static struct agx_bo *
hk_gen1_as_window_roots(struct hk_device *dev)
{
   if (dev->gen1_as_window_roots)
      return dev->gen1_as_window_roots;
   const uint64_t bytes = HK_GEN1_AS_WINDOW_BASES * HK_GEN1_AS_WINDOW_PAGE;
   struct agx_bo *bo = agx_bo_create(&dev->dev, bytes, HK_GEN1_AS_WINDOW_PAGE,
                                     0, "Gen1 AS window roots");
   if (!bo)
      return NULL;
   uint8_t *map = agx_bo_map(bo);
   memset(map, 0, bytes);
   for (uint32_t j = 0; j < HK_GEN1_AS_WINDOW_BASES; ++j)
      hk_gen1_as_forward_node((uint32_t *)(map + j * HK_GEN1_AS_WINDOW_PAGE),
         (HK_GEN1_AS_WINDOW_BASES - j) * HK_GEN1_AS_PAGE_SLOTS + j);
   util_post_flush_fence();
   dev->gen1_as_window_roots = bo;
   return bo;
}

static void
hk_gen1_as_window_destroy(struct hk_device *dev, struct agx_va *va,
                          struct agx_bo *trampolines)
{
   if (va) {
      agx_bo_bind(&dev->dev, NULL, va->addr, HK_GEN1_AS_WINDOW_BYTES, 0,
                  DRM_ASAHI_BIND_UNBIND);
      agx_va_free(&dev->dev, va, false);
   }
   agx_bo_unreference(&dev->dev, trampolines);
}

/* Caller holds gen1_as_lock. Returns the new window's index, or UINT32_MAX. */
static uint32_t
hk_gen1_as_window_create(struct hk_device *dev)
{
   struct agx_bo *roots = hk_gen1_as_window_roots(dev);
   if (!roots)
      return UINT32_MAX;
   struct agx_va *va = agx_va_alloc(&dev->dev, HK_GEN1_AS_WINDOW_BYTES,
                                    HK_GEN1_AS_WINDOW_PAGE, AGX_VA_LOW, 0);
   struct agx_bo *trampolines = va ?
      agx_bo_create(&dev->dev, HK_GEN1_AS_WINDOW_PAGE, HK_GEN1_AS_WINDOW_PAGE,
                    0, "Gen1 AS window trampolines") : NULL;
   const uint64_t roots_bytes = HK_GEN1_AS_WINDOW_BASES * HK_GEN1_AS_WINDOW_PAGE;
   if (!trampolines ||
       agx_bo_bind(&dev->dev, roots, va->addr, roots_bytes, 0,
                   DRM_ASAHI_BIND_READ) ||
       agx_bo_bind(&dev->dev, trampolines, va->addr + roots_bytes,
                   HK_GEN1_AS_WINDOW_PAGE, 0, DRM_ASAHI_BIND_READ)) {
      hk_gen1_as_window_destroy(dev, va, trampolines);
      return UINT32_MAX;
   }
   memset(agx_bo_map(trampolines), 0, HK_GEN1_AS_WINDOW_PAGE);
   util_post_flush_fence();
   struct hk_gen1_as_window *window =
      util_dynarray_grow(&dev->gen1_as_windows, struct hk_gen1_as_window, 1);
   if (!window) {
      hk_gen1_as_window_destroy(dev, va, trampolines);
      return UINT32_MAX;
   }
   *window = (struct hk_gen1_as_window){.va = va, .trampolines = trampolines};
   return util_dynarray_num_elements(&dev->gen1_as_windows,
                                     struct hk_gen1_as_window) - 1;
}

/* Lowest free base page of a window, or HK_GEN1_AS_WINDOW_BASES. */
static uint32_t
hk_gen1_as_window_free_base(const struct hk_gen1_as_window *window)
{
   for (uint32_t i = 0; i < ARRAY_SIZE(window->used); ++i) {
      uint32_t free = ~window->used[i];
      if (i == ARRAY_SIZE(window->used) - 1)
         free &= 0x7fffffffu; /* page 255 holds the trampolines */
      if (free)
         return i * 32 + ffs(free) - 1;
   }
   return HK_GEN1_AS_WINDOW_BASES;
}

/* Caller holds gen1_as_lock and slot has no window. Returns the traversal
 * base for a structure at address, or 0 if none can be assigned. */
static uint64_t
hk_gen1_as_window_assign(struct hk_device *dev, struct hk_gen1_as_slot *slot,
                         uint64_t address)
{
   if (!address || !(address % HK_GEN1_AS_WINDOW_PAGE))
      return address;
   const uint32_t count = util_dynarray_num_elements(&dev->gen1_as_windows,
                                                     struct hk_gen1_as_window);
   for (uint32_t k = 0; k <= count; ++k) {
      uint32_t index = k < count ? (dev->gen1_as_window_hint + k) % count
                                 : hk_gen1_as_window_create(dev);
      if (index == UINT32_MAX)
         return 0;
      struct hk_gen1_as_window *window = util_dynarray_element(
         &dev->gen1_as_windows, struct hk_gen1_as_window, index);
      const uint32_t j = hk_gen1_as_window_free_base(window);
      const uint64_t base = window->va->addr + j * HK_GEN1_AS_WINDOW_PAGE;
      if (j == HK_GEN1_AS_WINDOW_BASES || base >= address ||
          address - base >= HK_GEN1_AS_REACH)
         continue;
      window->used[j / 32] |= 1u << (j % 32);
      ++window->live;
      uint8_t *trampolines = agx_bo_map(window->trampolines);
      hk_gen1_as_forward_node(
         (uint32_t *)(trampolines + j * LIBAGX_GEN1_AS_SLOT_BYTES),
         (uint32_t)((address - base) / LIBAGX_GEN1_AS_SLOT_BYTES));
      util_post_flush_fence();
      slot->window = index;
      slot->window_base = j;
      dev->gen1_as_window_hint = index;
      return base;
   }
   return 0;
}

/* Caller holds gen1_as_lock. */
static void
hk_gen1_as_window_release(struct hk_device *dev, struct hk_gen1_as_slot *slot)
{
   if (slot->window != UINT32_MAX) {
      struct hk_gen1_as_window *window = util_dynarray_element(
         &dev->gen1_as_windows, struct hk_gen1_as_window, slot->window);
      const uint32_t j = slot->window_base;
      assert(window->used[j / 32] & (1u << (j % 32)));
      window->used[j / 32] &= ~(1u << (j % 32));
      --window->live;
      memset((uint8_t *)agx_bo_map(window->trampolines) +
                j * LIBAGX_GEN1_AS_SLOT_BYTES, 0, LIBAGX_GEN1_AS_SLOT_BYTES);
      util_post_flush_fence();
   }
   slot->window = UINT32_MAX;
   slot->base = 0;
}

static void
hk_gen1_as_windows_finish(struct hk_device *dev)
{
   util_dynarray_foreach(&dev->gen1_as_windows, struct hk_gen1_as_window, w)
      hk_gen1_as_window_destroy(dev, w->va, w->trampolines);
   util_dynarray_fini(&dev->gen1_as_windows);
   agx_bo_unreference(&dev->dev, dev->gen1_as_window_roots);
   dev->gen1_as_window_roots = NULL;
   dev->gen1_as_window_hint = 0;
}

/* Slots from a structure's traversal base to its image. Build and copy
 * commands encode child references with it. */
static uint32_t
hk_gen1_as_slot_bias(struct hk_device *dev, struct vk_acceleration_structure *as)
{
   uint32_t bias = 0;
   simple_mtx_lock(&dev->gen1_as_lock);
   struct hash_entry *entry = dev->gen1_as_ids ?
      _mesa_hash_table_search(dev->gen1_as_ids, as) : NULL;
   if (entry) {
      const struct hk_gen1_as_slot *slot = util_dynarray_element(
         &dev->gen1_ases, struct hk_gen1_as_slot,
         hk_gen1_as_id_to_index((uint32_t)(uintptr_t)entry->data));
      const uint64_t address = hk_gen1_as_bound_va(as);
      if (slot->base && address >= slot->base)
         bias = (uint32_t)((address - slot->base) / LIBAGX_GEN1_AS_SLOT_BYTES);
   }
   simple_mtx_unlock(&dev->gen1_as_lock);
   return bias;
}

/* Caller holds gen1_as_lock. Assigns a base if the structure has none yet
 * (its buffer had no address at creation) and publishes both addresses. */
static bool
hk_gen1_as_publish_slot(struct hk_device *dev, uint32_t index,
                        struct hk_gen1_as_slot *slot)
{
   const uint64_t address = hk_gen1_as_bound_va(slot->as);
   if (address && !slot->base) {
      slot->base = hk_gen1_as_window_assign(dev, slot, address);
      if (!slot->base)
         return false;
   }
   hk_gen1_as_publish_address(dev, hk_gen1_as_index_to_id(index), address,
                              slot->base);
   return true;
}

/* Publish the address of every acceleration structure placed in buffer, now
 * that it is bound. Returns false if one of them could not get a traversal
 * base: its table entry then does not describe it, and a ray traversal of it
 * would read whatever the entry held before, so the bind must fail. */
bool
hk_gen1_publish_buffer_as_addresses(struct hk_device *dev,
                                    const struct vk_buffer *buffer)
{
   if (!dev->g17p_native_as || !buffer->device_address)
      return true;
   bool ok = true;
   simple_mtx_lock(&dev->gen1_as_lock);
   uint32_t index = 0;
   util_dynarray_foreach(&dev->gen1_ases, struct hk_gen1_as_slot, slot) {
      if (slot->as && slot->as->buffer == buffer &&
          !hk_gen1_as_publish_slot(dev, index, slot))
         ok = false;
      ++index;
   }
   simple_mtx_unlock(&dev->gen1_as_lock);
   return ok;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateAccelerationStructureKHR(VkDevice device,
   const VkAccelerationStructureCreateInfoKHR *info,
   const VkAllocationCallbacks *alloc, VkAccelerationStructureKHR *out)
{
   VK_FROM_HANDLE(hk_device, m5_dispatch_dev, device);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5)
      return hk_m5_backend_CreateAccelerationStructureKHR(device, info, alloc, out);

   VK_FROM_HANDLE(hk_device, dev, device);
   VkAccelerationStructureCreateInfoKHR native_info;
   const VkDeviceAddress replay_address = dev->g17p_native_as ? info->deviceAddress : 0;
   if (replay_address) {
      native_info = *info;
      native_info.deviceAddress = 0;
      info = &native_info;
   }
   VkResult result = vk_common_CreateAccelerationStructureKHR(device, info, alloc, out);
   if (result != VK_SUCCESS || !dev->g17p_native_as)
      return result;
   VK_FROM_HANDLE(vk_acceleration_structure, as, *out);
   if (replay_address && vk_acceleration_structure_get_va(as) != replay_address) {
      vk_common_DestroyAccelerationStructureKHR(device, *out, alloc);
      *out = VK_NULL_HANDLE;
      return vk_error(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
   }
   bool inserted = false;
   simple_mtx_lock(&dev->gen1_as_lock);
   const bool reuse = dev->gen1_as_free_head != UINT32_MAX;
   const uint32_t index = reuse ? dev->gen1_as_free_head :
      util_dynarray_num_elements(&dev->gen1_ases, struct hk_gen1_as_slot);
   const uint32_t id = hk_gen1_as_index_to_id(index);
   if (id < HK_GEN1_AS_ID_LIMIT &&
       hk_gen1_sparse_entry(dev, &dev->gen1_as_address_pages, id) &&
       hk_gen1_sparse_entry(dev, &dev->gen1_as_table_pages, id) &&
       hk_gen1_sparse_entry(dev, &dev->gen1_as_metadata_pages, id)) {
      struct hk_gen1_as_slot *slot = reuse ?
         util_dynarray_element(&dev->gen1_ases, struct hk_gen1_as_slot, index) :
         util_dynarray_grow(&dev->gen1_ases, struct hk_gen1_as_slot, 1);
      if (slot) {
         if (reuse)
            dev->gen1_as_free_head = slot->next_free;
         *slot = (struct hk_gen1_as_slot){.as = as, .window = UINT32_MAX};
         inserted = true;
      }
   }
   if (inserted && dev->gen1_as_ids &&
       !_mesa_hash_table_insert(dev->gen1_as_ids, as, (void *)(uintptr_t)id)) {
      hk_gen1_as_free_slot(dev, index);
      inserted = false;
   }
   if (inserted &&
       !hk_gen1_as_publish_slot(dev, index, util_dynarray_element(
          &dev->gen1_ases, struct hk_gen1_as_slot, index))) {
      /* No address space below the structure for a traversal base. */
      _mesa_hash_table_remove_key(dev->gen1_as_ids, as);
      hk_gen1_as_free_slot(dev, index);
      inserted = false;
   }
   simple_mtx_unlock(&dev->gen1_as_lock);
   if (!inserted) {
      vk_common_DestroyAccelerationStructureKHR(device, *out, alloc);
      *out = VK_NULL_HANDLE;
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyAccelerationStructureKHR(VkDevice device, VkAccelerationStructureKHR handle,
                                   const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(hk_device, m5_dispatch_dev, device);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5) {
      hk_m5_backend_DestroyAccelerationStructureKHR(device, handle, alloc);
      return;
   }

   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(vk_acceleration_structure, as, handle);
   if (as && dev->g17p_native_as) {
      simple_mtx_lock(&dev->gen1_as_lock);
      if (dev->gen1_vertex_tables) {
         struct hash_entry *entry = _mesa_hash_table_search(dev->gen1_vertex_tables, as);
         if (entry) {
            free(entry->data);
            _mesa_hash_table_remove(dev->gen1_vertex_tables, entry);
         }
      }
      struct hash_entry *entry = dev->gen1_as_ids
         ? _mesa_hash_table_search(dev->gen1_as_ids, as) : NULL;
      if (entry) {
         const uint32_t id = (uint32_t)(uintptr_t)entry->data;
         const uint32_t index = hk_gen1_as_id_to_index(id);
         struct hk_gen1_as_slot *slot = util_dynarray_element(&dev->gen1_ases,
                     struct hk_gen1_as_slot, index);
         assert(slot->as == as);
         hk_gen1_as_publish_address(dev, id, 0, 0);
         hk_gen1_as_window_release(dev, slot);
         hk_gen1_as_free_slot(dev, index);
         _mesa_hash_table_remove(dev->gen1_as_ids, entry);
      }
      simple_mtx_unlock(&dev->gen1_as_lock);
   }
   vk_common_DestroyAccelerationStructureKHR(device, handle, alloc);
}

uint64_t
hk_gen1_upload_as_table(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   return dev->gen1_as_table_pages.va ? dev->gen1_as_table_pages.va->addr : 0;
}

static_assert(sizeof(struct hk_agx3_leaf_job) == 64, "native node job ABI");
static_assert(sizeof(struct hk_agx3_leaf_args) == 40, "native node push ABI");
static_assert(sizeof(struct hk_agx3_envelope_job) == 48, "native envelope job ABI");
static_assert(sizeof(struct hk_agx3_envelope_args) == 64, "native envelope push ABI");
/* The GEN1 layout: the plan state, then the node encoding dispatch arguments
 * the plan writes for the indirect encode (bvh/agx3_plan.comp), a
 * 64-byte record slot per child including bounding boxes, 16 KiB images,
 * and four 8-lane node jobs per encoding workgroup. */
static_assert(HK_AGX3_PLAN_ENCODE_DISPATCH == sizeof(struct hk_agx3_plan_state) &&
              HK_AGX3_GEN1_PLAN_TAIL_B >= HK_AGX3_PLAN_ENCODE_DISPATCH +
                                             sizeof(VkDispatchIndirectCommand),
              "GEN1 scratch: plan state, then the encode dispatch arguments");
static_assert(HK_AGX3_GEN1_AABB_RECORD_B == HK_AGX3_AS_SLOT_B,
              "GEN1 bounding-box records fill their child slot");
static_assert(HK_AGX3_GEN1_IMAGE_ALIGN == 16384, "GEN1 image alignment");
static_assert(HK_AGX3_GEN1_NODE_GROUP_LANES == 8 * HK_AGX3_NODE_JOBS_PER_GROUP,
              "GEN1 encoder workgroup");

const uint32_t *
hk_gen1_agx3_node_encoder_spirv(bool pairs, size_t *size_B)
{
   *size_B = pairs ? sizeof(agx3_node_pairs_spv) : sizeof(agx3_node_spv);
   return pairs ? agx3_node_pairs_spv : agx3_node_spv;
}

const uint32_t *
hk_gen1_agx3_refit_spirv(size_t *size_B)
{
   *size_B = sizeof(agx3_refit_spv);
   return agx3_refit_spv;
}

const uint32_t *
hk_agx3_node_encoder_spirv(enum agx_chip chip, size_t *size_B)
{
   return hk_agx3_as_format_for_chip(chip) == HK_AGX3_AS_FORMAT_M5
             ? hk_m5_agx3_node_encoder_spirv(size_B)
             : hk_gen1_agx3_node_encoder_spirv(false, size_B);
}

const uint32_t *
hk_agx3_refit_spirv(enum agx_chip chip, size_t *size_B)
{
   return hk_agx3_as_format_for_chip(chip) == HK_AGX3_AS_FORMAT_M5
             ? hk_m5_agx3_refit_spirv(size_B)
             : hk_gen1_agx3_refit_spirv(size_B);
}

static_assert(sizeof(struct hk_bvh_triangle_node) == 48, "node format");
static_assert(sizeof(struct hk_bvh_aabb_node) == 32, "node format");
static_assert(sizeof(struct hk_bvh_instance_node) == 120, "node format");
static_assert(sizeof(struct hk_bvh_box_node) == 56, "node format");
static_assert(sizeof(struct hk_bvh_box_node) == HK_BVH_NODE_PREFETCH_SIZE,
              "the traversal prefetches a whole box node before it knows the "
              "node type, so no node may be larger than one");

static VkResult
get_pipeline_spv(struct hk_device *dev, const char *name, const uint32_t *spv,
                 uint32_t spv_size, unsigned push_constant_size,
                 VkPipeline *pipeline, VkPipelineLayout *layout)
{
   size_t key_size = strlen(name);

   const VkPushConstantRange pc_range = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = push_constant_size,
   };

   VkResult result = vk_meta_get_pipeline_layout(
      &dev->vk, &dev->meta, NULL, &pc_range, name, key_size, layout);
   if (result != VK_SUCCESS)
      return result;

   VkPipeline cached = vk_meta_lookup_pipeline(&dev->meta, name, key_size);
   if (cached != VK_NULL_HANDLE) {
      *pipeline = cached;
      return VK_SUCCESS;
   }

   const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spv_size,
      .pCode = spv,
   };
   const VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage =
         {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &module_info,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = "main",
         },
      .layout = *layout,
   };

   return vk_meta_create_compute_pipeline(&dev->vk, &dev->meta, &pipeline_info,
                                          name, key_size, pipeline);
}

VkResult
hk_cmd_build_agx3_from_ir(VkCommandBuffer command_buffer,
                        const struct hk_agx3_plan_args *args,
                        VkDeviceAddress node_status,
                        VkDeviceAddress envelope_status, uint32_t slot_bias)
{
   VK_FROM_HANDLE(hk_cmd_buffer, m5_dispatch_cmd, command_buffer);
   struct hk_device *m5_dispatch_dev = hk_cmd_buffer_device(m5_dispatch_cmd);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5)
      return hk_m5_backend_cmd_build_agx3_from_ir(command_buffer, args, node_status, envelope_status);

   VK_FROM_HANDLE(hk_cmd_buffer, cmd, command_buffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (agx_chip_isa_encoding(dev->dev.chip) != AGX_ISA_ENCODING_AGX3C &&
       !dev->g17p_native_as)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct stage {
      const char *name;
      const uint32_t *code;
      size_t size;
      unsigned push_size;
   };
   /* A TLAS encodes only inner and instance nodes. */
   const bool instances = args->geometry_type == VK_GEOMETRY_TYPE_INSTANCES_KHR;
   const bool fused = args->node_bound <= HK_AGX3_PLAN_FUSED_MAX_NODES;
   const struct stage plan_stage = fused
      ? (args->pairs
            ? (struct stage){"hk-agx3-plan-fused-pairs",
                             agx3_plan_fused_pairs_spv,
                             sizeof(agx3_plan_fused_pairs_spv), sizeof(*args)}
            : (struct stage){"hk-agx3-plan-fused", agx3_plan_fused_spv,
                             sizeof(agx3_plan_fused_spv), sizeof(*args)})
      : (args->pairs
            ? (struct stage){"hk-agx3-plan-pairs", agx3_plan_pairs_spv,
                             sizeof(agx3_plan_pairs_spv), sizeof(*args)}
            : (struct stage){"hk-agx3-plan", agx3_plan_spv,
                             sizeof(agx3_plan_spv), sizeof(*args)});
   const struct stage node_encode = args->pairs
      ? (struct stage){"hk-agx3-node-encode-pairs", agx3_node_pairs_spv,
                       sizeof(agx3_node_pairs_spv),
                       sizeof(struct hk_agx3_leaf_args)}
      : instances
         ? (struct stage){"hk-agx3-node-encode-instances",
                          agx3_node_instances_spv,
                          sizeof(agx3_node_instances_spv),
                          sizeof(struct hk_agx3_leaf_args)}
         : (struct stage){"hk-agx3-node-encode", agx3_node_spv,
                          sizeof(agx3_node_spv),
                          sizeof(struct hk_agx3_leaf_args)};
   const struct stage stages[] = {
      plan_stage,
      node_encode,
      {"hk-agx3-as-envelope", agx3_envelope_spv, sizeof(agx3_envelope_spv),
       sizeof(struct hk_agx3_envelope_args)},
   };
   VkPipeline pipeline[3];
   VkPipelineLayout layout[3];
   for (unsigned i = 0; i < ARRAY_SIZE(stages); ++i) {
      VkResult result = get_pipeline_spv(
         dev, stages[i].name, stages[i].code, stages[i].size,
         stages[i].push_size, &pipeline[i], &layout[i]);
      if (result != VK_SUCCESS)
         return result;
   }

   vk_common_CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline[0]);
   const VkMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
   };
   if (fused) {
      struct hk_agx3_plan_args push = *args;
      push.phase = 0;
      vk_common_CmdPushConstants(command_buffer, layout[0],
                                 VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                                 &push);
      hk_cmd_dispatch_unaligned(command_buffer, HK_AGX3_PLAN_FUSED_GROUP_SIZE,
                                1, 1);
      vk_common_CmdPipelineBarrier(
         command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
         0, 1, &barrier, 0, NULL, 0, NULL);
   } else {
      for (unsigned phase = 0; phase < 5; ++phase) {
         struct hk_agx3_plan_args push = *args;
         push.phase = phase;
         vk_common_CmdPushConstants(command_buffer, layout[0],
                                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                                    &push);
         hk_cmd_dispatch_unaligned(command_buffer,
                                   phase == 4 ? 1 : args->node_bound,
                                   1, 1);
         vk_common_CmdPipelineBarrier(
            command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0, 1, &barrier, 0, NULL, 0, NULL);
      }
   }

   const struct hk_agx3_leaf_args encode = {
      .jobs = args->jobs, .status = node_status, .job_count = args->node_bound,
      .slot_bias = slot_bias, .counts = args->state,
      .leaf_bound = args->leaf_bound,
   };
   const struct hk_agx3_envelope_args finish = {
      .jobs = args->envelope, .status = envelope_status, .job_count = 1,
      .reserved = dev->g17p_native_as ? 2 : 3,
      .metadata = args->metadata, .ir_header = args->header,
      .retained_offset = args->retained_offset,
      .build_flags = args->build_flags, .node_bound = args->node_bound,
      .keep_retained = args->keep_retained,
      .refit = fused && instances ? 1u : 0u,
   };
   /* Large TLAS builds still rely on the envelope pass to clear the instance
    * pointer table before node encoding writes active entries. BLAS/AABB
    * builds and fused small TLAS builds (whose planner already cleared the
    * table) write disjoint regions and overlap envelope with node encoding. */
   const bool need_step_barrier = instances && !fused;
   for (unsigned step = 0; step < 2; ++step) {
      bool envelope = instances ? step == 0 : step == 1;
      unsigned stage = envelope ? 2 : 1;
      if (!envelope && fused && args->node_bound == 0)
         continue;
      vk_common_CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline[stage]);
      vk_common_CmdPushConstants(command_buffer, layout[stage],
                                 VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 envelope ? sizeof(finish) : sizeof(encode),
                                 envelope ? (const void *)&finish : (const void *)&encode);
      /* A TLAS envelope also clears a pointer per instance: spread that
       * over a thread per instance rather than one workgroup. */
      if (envelope)
         hk_cmd_dispatch_unaligned(command_buffer,
                                   instances && !fused
                                      ? CLAMP(ALIGN_POT(args->leaf_bound, 64),
                                              64, 65536)
                                      : 64,
                                   1, 1);
      else if (fused)
         hk_cmd_dispatch_unaligned(
            command_buffer,
            DIV_ROUND_UP(args->node_bound, HK_AGX3_NODE_JOBS_PER_GROUP) *
               HK_AGX3_GEN1_NODE_GROUP_LANES,
            1, 1);
      else
         hk_cmd_dispatch_indirect_addr(command_buffer,
                                       args->state + HK_AGX3_PLAN_ENCODE_DISPATCH);
      if (step == 0 && need_step_barrier)
         vk_common_CmdPipelineBarrier(command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0, 1, &barrier, 0, NULL, 0, NULL);
   }
   return VK_SUCCESS;
}

/* ---------------------------------------------------------------- */
/* Layout                                                            */

static void
hk_leaf_node_sizes(VkGeometryTypeKHR geometry_type, uint32_t *ir_size,
                   uint32_t *out_size)
{
   switch (geometry_type) {
   case VK_GEOMETRY_TYPE_TRIANGLES_KHR:
      *ir_size = sizeof(struct vk_ir_triangle_node);
      *out_size = sizeof(struct hk_bvh_triangle_node);
      return;
   case VK_GEOMETRY_TYPE_AABBS_KHR:
      *ir_size = sizeof(struct vk_ir_aabb_node);
      *out_size = sizeof(struct hk_bvh_aabb_node);
      return;
   default:
      *ir_size = sizeof(struct vk_ir_instance_node);
      *out_size = sizeof(struct hk_bvh_instance_node);
      return;
   }
}

static VkDeviceSize
hk_as_size(VkGeometryTypeKHR geometry_type, uint32_t leaf_node_count)
{
   /* The common builder always produces MAX2(leaves, 2) - 1 internal nodes,
    * so a one-primitive structure still has a root to point at. */
   uint32_t internal_node_count = MAX2(leaf_node_count, 2) - 1;

   uint32_t ir_size, out_size;
   hk_leaf_node_sizes(geometry_type, &ir_size, &out_size);

   uint32_t nodes_size = internal_node_count * sizeof(struct hk_bvh_box_node) +
                         leaf_node_count * out_size;

   /* Pad so the traversal's unconditional whole-node prefetch of the last
    * node cannot read past the allocation. */
   nodes_size = util_align_npot(nodes_size, HK_BVH_NODE_PREFETCH_SIZE);

   return sizeof(struct hk_bvh_header) + nodes_size;
}

static VkDeviceSize
hk_agx3_retained_offset(const struct vk_acceleration_structure_build_state *state)
{
   struct hk_agx3_as_layout layout;
   if (!hk_gen1_as_layout_init(state->leaf_node_count, &layout))
      return 0;
   return ALIGN_POT(layout.as_bytes +
      (vk_get_as_geometry_type(state->build_info) == VK_GEOMETRY_TYPE_INSTANCES_KHR
          ? (uint64_t)state->leaf_node_count * 80 : 0), 8);
}

static bool
hk_vertex_table_wanted(const struct hk_device *dev,
                       const struct vk_acceleration_structure_build_state *state)
{
   return dev->g17p_native_as && dev->gen1_vertex_directory &&
          dev->vk.enabled_features.rayTracingPositionFetch &&
          vk_get_as_geometry_type(state->build_info) == VK_GEOMETRY_TYPE_TRIANGLES_KHR &&
          /* Hit-object position fetch is part of invocation reorder and does
           * not require the position-fetch AS build flag. Retain the same
           * native vertex table for that feature as for explicit data access. */
          ((state->build_info->flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR) ||
           dev->vk.enabled_features.rayTracingInvocationReorder) &&
          state->build_info->geometryCount > 0;
}

static VkDeviceSize
hk_get_native_as_size(const struct vk_acceleration_structure_build_state *state)
{
   struct hk_agx3_as_layout layout;
   if (!hk_gen1_as_layout_init(state->leaf_node_count, &layout))
      return 0;
   /* The retained jobs, for as many jobs as there can be. */
   if (state->build_info->flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)
      return hk_agx3_retained_offset(state) +
             HK_AGX3_RETAINED_BYTES((uint64_t)layout.node_bound,
                                    (uint64_t)state->leaf_node_count);
   return layout.as_bytes +
      (vk_get_as_geometry_type(state->build_info) == VK_GEOMETRY_TYPE_INSTANCES_KHR
          ? (uint64_t)state->leaf_node_count * 80 : 0);
}

static bool
hk_vertex_table_layout(const struct hk_device *dev,
                       const struct vk_acceleration_structure_build_state *state,
                       struct hk_vertex_table_layout *out)
{
   if (!hk_vertex_table_wanted(dev, state))
      return false;
   const uint64_t base = hk_get_native_as_size(state);
   if (!base)
      return false;
   out->offset = ALIGN_POT(base, UINT64_C(16));
   out->geometry_count = state->build_info->geometryCount;
   out->primitives = state->leaf_node_count;
   out->vertices_offset = ALIGN_POT(16 + 4 * out->geometry_count, 16);
   out->bytes = (uint64_t)out->vertices_offset + (uint64_t)out->primitives * 36;
   return true;
}

static VkDeviceSize
hk_get_as_size(VkDevice _device,
               const struct vk_acceleration_structure_build_state *state)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   if (dev->g17p_native_as) {
      struct hk_vertex_table_layout table;
      if (hk_vertex_table_layout(dev, state, &table))
         return table.offset + table.bytes;
      return hk_get_native_as_size(state);
   }
   return hk_as_size(vk_get_as_geometry_type(state->build_info),
                     state->leaf_node_count);
}

static VkDeviceSize
hk_get_encode_scratch_size(VkDevice _device,
                           const struct vk_acceleration_structure_build_state *state)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_agx3_as_layout layout;
   return dev->g17p_native_as && hk_gen1_as_layout_init(state->leaf_node_count, &layout)
             ? layout.encode_bytes + ALIGN_POT(sizeof(struct vk_ir_header), 8) -
                  sizeof(struct vk_ir_header) : 0;
}

/* The native image lives at the Vulkan address; see libagx/gen1_as.h. */
static uint64_t
hk_as_native_va(struct vk_acceleration_structure *as)
{
   return vk_acceleration_structure_get_va(as);
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL
hk_GetAccelerationStructureDeviceAddressKHR(
   VkDevice _device, const VkAccelerationStructureDeviceAddressInfoKHR *info)
{
   VK_FROM_HANDLE(hk_device, m5_dispatch_dev, _device);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5)
      return hk_m5_backend_GetAccelerationStructureDeviceAddressKHR(_device, info);

   VK_FROM_HANDLE(vk_acceleration_structure, as, info->accelerationStructure);
   return vk_acceleration_structure_get_va(as);
}

/* ---------------------------------------------------------------- */
/* Encode                                                            */

static void
hk_encode_as(VkCommandBuffer commandBuffer,
             const struct vk_acceleration_structure_build_state *state,
             VkPipelineLayout layout)
{
   VK_FROM_HANDLE(vk_acceleration_structure, dst,
                  state->build_info->dstAccelerationStructure);

   VkGeometryTypeKHR geometry_type =
      vk_get_as_geometry_type(state->build_info);

   VkDeviceAddress scratch = state->build_info->scratchData.deviceAddress;
   uint32_t internal_bound = MAX2(state->leaf_node_count, 2) - 1;

   const struct hk_encode_args args = {
      .intermediate_bvh = scratch + state->scratch.ir_offset,
      .output_bvh = vk_acceleration_structure_get_va(dst),
      .header = scratch + state->scratch.header_offset,
      .root_offset =
         state->scratch.internal_node_offset - state->scratch.ir_offset,
      .internal_node_bound = internal_bound,
      .leaf_node_bound = state->leaf_node_count,
      .geometry_type = geometry_type,
      .as_size = hk_as_size(geometry_type, state->leaf_node_count),
   };
   vk_common_CmdPushConstants(commandBuffer, layout,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(args),
                              &args);

   /* One invocation per output node.  The shader clamps against the counts
    * the builder wrote into the header, which can be smaller than this bound
    * when leaves turn out inactive. */
   hk_cmd_dispatch_unaligned(commandBuffer,
                             internal_bound + state->leaf_node_count, 1, 1);
}

static void hk_write_buffer_cp(VkCommandBuffer commandBuffer, VkDeviceAddress addr,
                               void *data, uint32_t size);

static void
hk_vertex_table_publish(VkCommandBuffer commandBuffer, struct hk_device *dev,
                        struct vk_acceleration_structure *as,
                        const struct hk_vertex_table_layout *table)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   const uint32_t id = hk_gen1_as_id(dev, as);
   void *directory_entry = NULL;
   if (id) {
      simple_mtx_lock(&dev->gen1_as_lock);
      directory_entry = hk_gen1_sparse_entry(dev, &dev->gen1_vertex_directory_pages, id);
      simple_mtx_unlock(&dev->gen1_as_lock);
   }
   if (!directory_entry) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   const uint64_t table_va = hk_as_native_va(as) + table->offset;
   uint32_t entry[4] = {(uint32_t)table_va, (uint32_t)(table_va >> 32),
                        table->geometry_count, table->primitives};
   hk_cmd_buffer_retain_bo(cmd, dev->gen1_vertex_directory);
   hk_write_buffer_cp(commandBuffer, AGX_GEN1_VERTEX_DIRECTORY_VA + (uint64_t)id * 16,
                      entry, sizeof(entry));
   simple_mtx_lock(&dev->gen1_as_lock);
   if (dev->gen1_vertex_tables) {
      struct hash_entry *existing = _mesa_hash_table_search(dev->gen1_vertex_tables, as);
      struct hk_vertex_table_layout *copy = existing ? existing->data : malloc(sizeof(*copy));
      if (copy) {
         *copy = *table;
         if (!existing)
            _mesa_hash_table_insert(dev->gen1_vertex_tables, as, copy);
      }
   }
   simple_mtx_unlock(&dev->gen1_as_lock);
}

static void
hk_vertex_table_build(VkCommandBuffer commandBuffer,
                      const struct vk_acceleration_structure_build_state *state,
                      struct vk_acceleration_structure *dst)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_vertex_table_layout table;
   if (!hk_vertex_table_layout(dev, state, &table))
      return;
   if (table.offset + table.bytes > dst->size) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   const uint64_t table_va = hk_as_native_va(dst) + table.offset;
   uint32_t header[4] = {AGX_GEN1_VERTEX_TABLE_MAGIC, table.geometry_count,
                         table.vertices_offset, table.primitives};
   hk_write_buffer_cp(commandBuffer, table_va, header, sizeof(header));
   uint32_t bases[28];
   uint32_t running = 0, pending = 0, first = 0;
   for (uint32_t g = 0; g < table.geometry_count; ++g) {
      bases[pending++] = running;
      running += state->build_range_infos[g].primitiveCount;
      if (pending == 28 || g + 1 == table.geometry_count) {
         hk_write_buffer_cp(commandBuffer, table_va + 16 + (uint64_t)first * 4,
                            bases, pending * 4);
         first += pending;
         pending = 0;
      }
   }
   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);
   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result = get_pipeline_spv(dev, "hk-vertex-table", vertex_table_spv,
                                      sizeof(vertex_table_spv),
                                      sizeof(struct hk_vertex_table_args),
                                      &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   const struct hk_vertex_table_args args = {
      .leaves = state->build_info->scratchData.deviceAddress + state->scratch.ir_offset,
      .table = table_va,
      .leaf_count = table.primitives,
      .geometry_count = table.geometry_count,
      .vertices_offset = table.vertices_offset,
      .leaf_stride = sizeof(struct vk_ir_triangle_node),
   };
   vk_common_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   vk_common_CmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                              sizeof(args), &args);
   hk_cmd_dispatch_unaligned(commandBuffer, table.primitives, 1, 1);
   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);
   hk_vertex_table_publish(commandBuffer, dev, dst, &table);
}

static void
hk_encode(VkCommandBuffer commandBuffer, struct vk_device *device,
          struct vk_meta_device *meta,
          const struct vk_acceleration_structure_build_args *build_args,
          struct vk_acceleration_structure_build_state *states,
          uint32_t build_count, bool flushed_compute_after_init_update_scratch)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (dev->g17p_native_as) {
      vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);
      const VkResult update_result = hk_cmd_refit_agx3(commandBuffer, device, meta,
                                                       build_args, states, build_count);
      if (update_result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, update_result);
         return;
      }
      for (uint32_t i = 0; i < build_count; ++i) {
         const struct vk_acceleration_structure_build_state *state = &states[i];
         VK_FROM_HANDLE(vk_acceleration_structure, dst,
                        state->build_info->dstAccelerationStructure);
         if (state->config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
            continue;
         const bool allow_update =
            (state->build_info->flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) != 0;

         struct hk_agx3_as_layout native;
         if (!hk_gen1_as_layout_init(state->leaf_node_count, &native)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }
         const uint64_t scratch = state->build_info->scratchData.deviceAddress;
         const uint64_t encode = ALIGN_POT(scratch + state->scratch.header_offset +
                                           sizeof(struct vk_ir_header), UINT64_C(8));
         const struct hk_agx3_plan_args plan = {
            .intermediate_bvh = scratch + state->scratch.ir_offset,
            .header = scratch + state->scratch.header_offset,
            .nodes = encode + native.nodes, .jobs = encode + native.jobs,
            .envelope = encode + native.envelope, .state = encode + native.state,
            .output_bvh = hk_as_native_va(dst),
            .root_offset = state->scratch.internal_node_offset - state->scratch.ir_offset,
            .leaf_bound = state->leaf_node_count,
            .node_bound = native.node_bound, .slot_bound = native.slot_bound,
            .resource_id = hk_gen1_as_id(dev, dst),
            .geometry_type = vk_get_as_geometry_type(state->build_info),
            .retained_offset = hk_agx3_retained_offset(state),
            .build_flags = state->build_info->flags,
            /* An update re-encodes the retained jobs, which refit one
             * triangle per record, so only triangle structures that cannot
             * be updated pair triangles; and not those preferring a fast
             * build, where finding pairs costs more than it saves. */
            .pairs = !allow_update &&
                     vk_get_as_geometry_type(state->build_info) ==
                        VK_GEOMETRY_TYPE_TRIANGLES_KHR &&
                     !(state->build_info->flags &
                       VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR),
         };
         struct hk_agx3_plan_args native_plan = plan;
         native_plan.metadata = hk_gen1_as_metadata_addr_for_id(cmd, plan.resource_id);
         if (!native_plan.metadata)
            return;
         VkResult result = hk_cmd_build_agx3_from_ir(commandBuffer, &native_plan,
            encode + native.node_status, encode + native.envelope_status,
            hk_gen1_as_slot_bias(dev, dst));
         if (result != VK_SUCCESS) {
            vk_command_buffer_set_error(&cmd->vk, result);
            return;
         }
         /* Keep the jobs of a build that can be updated. */
         if (allow_update)
            hk_agx3_refit_save(commandBuffer, state);
         hk_vertex_table_build(commandBuffer, state, dst);
      }
      return;
   }
   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result =
      get_pipeline_spv(dev, "hk-bvh-encode", encode_spv, sizeof(encode_spv),
                       sizeof(struct hk_encode_args), &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   /* The common builder hands encoding the completed intermediate tree. */
   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);
   vk_common_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline);
   for (uint32_t i = 0; i < build_count; i++)
      hk_encode_as(commandBuffer, &states[i], layout);
}

/* A triangle update runs its own leaf pass (agx3_refit_leaf.comp) instead of
 * generating IR leaves: not for indirect builds, whose ranges are on the
 * device, nor with a vertex table, which is built from the IR leaves. */
static bool
hk_agx3_refit_skips_leaves(const struct hk_device *dev,
                           const struct vk_acceleration_structure_build_state *state)
{
   return vk_get_as_geometry_type(state->build_info) == VK_GEOMETRY_TYPE_TRIANGLES_KHR &&
          (state->config.build_flags & VK_BUILD_FLAG_ALWAYS_ACTIVE) &&
          !(state->config.build_flags & VK_BUILD_FLAG_HAS_QUADS) &&
          !state->indirect_addr && !hk_vertex_table_wanted(dev, state);
}

/* Levels of inner jobs an update encodes a level per dispatch; propagation
 * encodes any deeper. Each level costs a dispatch even when the tree is
 * shallower, so take about as many as the tree is likely to have: at 4K,
 * 64K and 1M triangles the fastest updates took 4-8, 8-12 and 12-16. */
static unsigned
hk_agx3_refit_levels(const struct vk_acceleration_structure_build_state *state)
{
   const unsigned bits = util_logbase2_ceil(MAX2(state->leaf_node_count, 1));
   return CLAMP((int)bits - 6, 4, (int)HK_AGX3_REFIT_LEVELS);
}

/* Run refit phase `phase`; `level` is hk_agx3_refit_args.level. */
static void
hk_agx3_refit_phase(VkCommandBuffer commandBuffer,
                    const struct vk_acceleration_structure_build_state *state,
                    unsigned phase, unsigned level)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_agx3_as_layout layout;
   if (!hk_gen1_as_layout_init(state->leaf_node_count, &layout)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   VkPipeline pipeline;
   VkPipelineLayout pipeline_layout;
   VkResult result =
      phase == HK_AGX3_REFIT_INNER
         ? get_pipeline_spv(dev, "hk-agx3-refit-inner", agx3_refit_inner_spv,
                            sizeof(agx3_refit_inner_spv),
                            sizeof(struct hk_agx3_refit_args), &pipeline,
                            &pipeline_layout)
      : phase == HK_AGX3_REFIT_PROPAGATE
         ? get_pipeline_spv(dev, "hk-agx3-refit-propagate",
                            agx3_refit_propagate_spv,
                            sizeof(agx3_refit_propagate_spv),
                            sizeof(struct hk_agx3_refit_args), &pipeline,
                            &pipeline_layout)
         : get_pipeline_spv(dev, "hk-agx3-refit", agx3_refit_spv,
                            sizeof(agx3_refit_spv),
                            sizeof(struct hk_agx3_refit_args), &pipeline,
                            &pipeline_layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   VK_FROM_HANDLE(vk_acceleration_structure, dst,
                  state->build_info->dstAccelerationStructure);
   VK_FROM_HANDLE(vk_acceleration_structure, src,
                  state->build_info->srcAccelerationStructure);
   const bool prepare = phase == HK_AGX3_REFIT_PREPARE;
   const uint64_t scratch = state->build_info->scratchData.deviceAddress;
   const uint64_t encode = ALIGN_POT(scratch + state->scratch.header_offset +
                                     sizeof(struct vk_ir_header), UINT64_C(8));
   struct hk_agx3_refit_args args = {
      .retained = hk_as_native_va(dst) + hk_agx3_retained_offset(state),
      /* The source may be a compacted copy: its metadata finds its jobs. */
      .metadata = prepare ? hk_gen1_as_metadata_addr(cmd, src) : 0,
      .intermediate_bvh = scratch + state->scratch.ir_offset,
      .header = scratch + state->scratch.header_offset,
      .plan_nodes = encode + layout.nodes, .plan_jobs = encode + layout.jobs,
      .plan_state = encode + layout.state,
      .refit = encode + layout.state, .counters = encode + layout.nodes,
      .bounds = encode + layout.jobs, .envelope = encode + layout.envelope,
      .output_bvh = hk_as_native_va(dst),
      .root_offset = state->scratch.internal_node_offset - state->scratch.ir_offset,
      .leaf_bound = state->leaf_node_count,
      .node_bound = layout.node_bound, .phase = phase,
      .resource_id = hk_gen1_as_id(dev, dst),
      .geometry_type = vk_get_as_geometry_type(state->build_info),
      .copy = prepare && state->build_info->srcAccelerationStructure !=
                         state->build_info->dstAccelerationStructure,
      .slot_bias = hk_gen1_as_slot_bias(dev, dst),
      .status = encode + layout.node_status,
      .levels = prepare && layout.node_bound ? encode + layout.refit_levels : 0,
      .level = level,
      .no_leaves = prepare && hk_agx3_refit_skips_leaves(dev, state),
   };
   /* One thread per job, per instance pointer, and per retained word that
    * an update into another structure copies. */
   uint64_t count = MAX2(layout.node_bound, state->leaf_node_count);
   if (args.copy)
      count = MAX2(count, HK_AGX3_RETAINED_BYTES((uint64_t)layout.node_bound,
                                                 (uint64_t)state->leaf_node_count) / 4);
   vk_common_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   vk_common_CmdPushConstants(commandBuffer, pipeline_layout,
      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(args), &args);
   if (phase == HK_AGX3_REFIT_PROPAGATE)
      hk_cmd_dispatch_indirect_addr(commandBuffer, args.refit +
                                    HK_AGX3_REFIT_PROPAGATE_DISPATCH);
   else if (phase == HK_AGX3_REFIT_INNER)
      hk_cmd_dispatch_indirect_addr(commandBuffer, encode + layout.refit_levels +
                                    level * 12);
   else
      hk_cmd_dispatch_unaligned(commandBuffer, count, 1, 1);
}

/* Keep the jobs of a full build that can be updated, numbered leaf first,
 * with its inner jobs listed by level. */
static void
hk_agx3_refit_save(VkCommandBuffer commandBuffer,
                   const struct vk_acceleration_structure_build_state *state)
{
   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);
   hk_agx3_refit_phase(commandBuffer, state, HK_AGX3_REFIT_SAVE, 0);
   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);
   hk_agx3_refit_phase(commandBuffer, state, HK_AGX3_REFIT_NUMBER, 0);
   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);
   hk_agx3_refit_phase(commandBuffer, state, HK_AGX3_REFIT_COUNT_LEVELS, 0);
   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);
   hk_agx3_refit_phase(commandBuffer, state, HK_AGX3_REFIT_LIST_LEVELS, 0);
}

/* The updates of one build call, step by step together: each step is a
 * launch per update with one barrier after the step, so a step's launches
 * have no barrier between them and may run side by side (see
 * hk_dispatch_g17p_native()). Per update the steps are: leaves (or, for
 * triangles that place their own, nothing), prepare from the retained jobs,
 * place the triangles, encode the leaf jobs, propagate to the inner jobs past
 * the levels, encode the inner jobs of each level, deepest first, and finish
 * the envelope. */
static VkResult
hk_cmd_refit_agx3(VkCommandBuffer commandBuffer, struct vk_device *device,
                  struct vk_meta_device *meta,
                  const struct vk_acceleration_structure_build_args *build_args,
                  const struct vk_acceleration_structure_build_state *states,
                  uint32_t build_count)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   bool any_update = false;
   for (uint32_t i = 0; i < build_count; ++i)
      any_update |= states[i].config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE;
   if (!any_update)
      return VK_SUCCESS;

   VkPipeline pipeline[4];
   VkPipelineLayout pipeline_layout[4];
   enum { ENCODE_TRIANGLES, ENCODE, ENVELOPE, PLACE };
   VkResult result =
      get_pipeline_spv(dev, "hk-agx3-refit-encode-triangles",
                       agx3_refit_encode_triangles_spv,
                       sizeof(agx3_refit_encode_triangles_spv),
                       sizeof(struct hk_agx3_refit_encode_args),
                       &pipeline[ENCODE_TRIANGLES], &pipeline_layout[ENCODE_TRIANGLES]);
   if (result == VK_SUCCESS)
      result = get_pipeline_spv(dev, "hk-agx3-refit-encode", agx3_refit_encode_spv,
         sizeof(agx3_refit_encode_spv), sizeof(struct hk_agx3_refit_encode_args),
         &pipeline[ENCODE], &pipeline_layout[ENCODE]);
   if (result == VK_SUCCESS)
      result = get_pipeline_spv(dev, "hk-agx3-as-envelope", agx3_envelope_spv,
         sizeof(agx3_envelope_spv), sizeof(struct hk_agx3_envelope_args),
         &pipeline[ENVELOPE], &pipeline_layout[ENVELOPE]);
   if (result == VK_SUCCESS)
      result = get_pipeline_spv(dev, "hk-agx3-refit-leaf", agx3_refit_leaf_spv,
         sizeof(agx3_refit_leaf_spv), sizeof(struct hk_agx3_refit_leaf_args),
         &pipeline[PLACE], &pipeline_layout[PLACE]);
   if (result != VK_SUCCESS)
      return result;

   const VkMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                       VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
   };
   bool any;
   unsigned max_levels = 0;

   /* Leaves */
   any = false;
   for (uint32_t i = 0; i < build_count; ++i) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE)
         continue;
      VK_FROM_HANDLE(vk_acceleration_structure, dst, state->build_info->dstAccelerationStructure);
      struct hk_agx3_as_layout layout;
      if (!hk_gen1_as_layout_init(state->leaf_node_count, &layout) ||
          !hk_gen1_as_metadata_addr_for_id(cmd, hk_gen1_as_id(dev, dst)))
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      max_levels = MAX2(max_levels, hk_agx3_refit_levels(state));
      if (hk_agx3_refit_skips_leaves(dev, state))
         continue;
      result = vk_cmd_build_as_leaves(commandBuffer, device, meta, build_args, state);
      if (result != VK_SUCCESS)
         return result;
      any = true;
   }
   if (any)
      vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);

   /* Prepare */
   for (uint32_t i = 0; i < build_count; ++i) {
      if (states[i].config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
         hk_agx3_refit_phase(commandBuffer, &states[i], HK_AGX3_REFIT_PREPARE,
                             hk_agx3_refit_levels(&states[i]));
   }
   vk_common_CmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
      0, 1, &barrier, 0, NULL, 0, NULL);

   /* Place the triangles: a thread per primitive of each geometry, numbered
    * as leaf generation numbers them; they take the IR leaves' place. */
   static_assert(sizeof(struct hk_agx3_refit_triangle) <= sizeof(struct vk_ir_triangle_node),
                 "placed triangles fit where the IR leaves would be");
   any = false;
   for (uint32_t i = 0; i < build_count; ++i) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE ||
          !hk_agx3_refit_skips_leaves(dev, state))
         continue;
      struct hk_agx3_as_layout layout;
      hk_gen1_as_layout_init(state->leaf_node_count, &layout);
      const uint64_t scratch = state->build_info->scratchData.deviceAddress;
      const uint64_t encode = ALIGN_POT(scratch + state->scratch.header_offset +
                                        sizeof(struct vk_ir_header), UINT64_C(8));
      const VkAccelerationStructureBuildGeometryInfoKHR *info = state->build_info;
      struct hk_agx3_refit_leaf_args leaf = {
         .refit = encode + layout.state,
         .triangles = scratch + state->scratch.ir_offset,
         .leaf_bound = state->leaf_node_count,
      };
      vk_common_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline[PLACE]);
      uint32_t first = 0;
      for (uint32_t j = 0; j < info->geometryCount; ++j) {
         const VkAccelerationStructureBuildRangeInfoKHR *range = &state->build_range_infos[j];
         if (!range->primitiveCount)
            continue;
         leaf.geometry = vk_fill_geometry_data(info->type, first, j,
            info->pGeometries ? &info->pGeometries[j] : info->ppGeometries[j], range);
         leaf.primitive_count = range->primitiveCount;
         vk_common_CmdPushConstants(commandBuffer, pipeline_layout[PLACE],
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(leaf), &leaf);
         hk_cmd_dispatch_unaligned(commandBuffer, range->primitiveCount, 1, 1);
         first += range->primitiveCount;
      }
      any = true;
   }
   if (any)
      vk_common_CmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);

   /* Encode the leaf jobs */
   for (uint32_t i = 0; i < build_count; ++i) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE)
         continue;
      VK_FROM_HANDLE(vk_acceleration_structure, dst, state->build_info->dstAccelerationStructure);
      struct hk_agx3_as_layout layout;
      hk_gen1_as_layout_init(state->leaf_node_count, &layout);
      const uint64_t scratch = state->build_info->scratchData.deviceAddress;
      const uint64_t encode = ALIGN_POT(scratch + state->scratch.header_offset +
                                        sizeof(struct vk_ir_header), UINT64_C(8));
      const bool triangles =
         vk_get_as_geometry_type(state->build_info) == VK_GEOMETRY_TYPE_TRIANGLES_KHR;
      const struct hk_agx3_refit_encode_args refit = {
         .refit = encode + layout.state, .bounds = encode + layout.jobs,
         .intermediate_bvh = scratch + state->scratch.ir_offset,
         .output_bvh = hk_as_native_va(dst), .status = encode + layout.node_status,
         .slot_bias = hk_gen1_as_slot_bias(dev, dst),
         .leaf_bound = state->leaf_node_count,
         .triangles = hk_agx3_refit_skips_leaves(dev, state)
                         ? scratch + state->scratch.ir_offset : 0,
      };
      const unsigned p = triangles ? ENCODE_TRIANGLES : ENCODE;
      vk_common_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline[p]);
      vk_common_CmdPushConstants(commandBuffer, pipeline_layout[p],
         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(refit), &refit);
      hk_cmd_dispatch_indirect_addr(commandBuffer, encode + layout.state);
   }
   vk_common_CmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);

   /* Propagate */
   for (uint32_t i = 0; i < build_count; ++i) {
      if (states[i].config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
         hk_agx3_refit_phase(commandBuffer, &states[i], HK_AGX3_REFIT_PROPAGATE,
                             hk_agx3_refit_levels(&states[i]));
   }
   vk_common_CmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);

   /* The levels, deepest first; an update takes part in its own levels. */
   for (unsigned level = max_levels; level-- > 0;) {
      any = false;
      for (uint32_t i = 0; i < build_count; ++i) {
         const struct vk_acceleration_structure_build_state *state = &states[i];
         if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE ||
             level >= hk_agx3_refit_levels(state))
            continue;
         struct hk_agx3_as_layout layout;
         hk_gen1_as_layout_init(state->leaf_node_count, &layout);
         if (!layout.node_bound)
            continue;
         hk_agx3_refit_phase(commandBuffer, state, HK_AGX3_REFIT_INNER, level);
         any = true;
      }
      if (any)
         vk_common_CmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
   }

   /* Envelopes */
   for (uint32_t i = 0; i < build_count; ++i) {
      const struct vk_acceleration_structure_build_state *state = &states[i];
      if (state->config.internal_type != VK_INTERNAL_BUILD_TYPE_UPDATE)
         continue;
      VK_FROM_HANDLE(vk_acceleration_structure, dst, state->build_info->dstAccelerationStructure);
      struct hk_agx3_as_layout layout;
      hk_gen1_as_layout_init(state->leaf_node_count, &layout);
      const uint64_t scratch = state->build_info->scratchData.deviceAddress;
      const uint64_t encode = ALIGN_POT(scratch + state->scratch.header_offset +
                                        sizeof(struct vk_ir_header), UINT64_C(8));
      const struct hk_agx3_envelope_args finish = {
         .jobs = encode + layout.envelope, .status = encode + layout.envelope_status,
         .job_count = 1, .reserved = 2,
         .metadata = hk_gen1_as_metadata_addr_for_id(cmd, hk_gen1_as_id(dev, dst)),
         .ir_header = scratch + state->scratch.header_offset,
         .retained_offset = hk_agx3_retained_offset(state),
         .build_flags = state->build_info->flags, .node_bound = layout.node_bound,
         .keep_retained = state->build_info->srcAccelerationStructure ==
                          state->build_info->dstAccelerationStructure,
         .refit = 1,
      };
      vk_common_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline[ENVELOPE]);
      vk_common_CmdPushConstants(commandBuffer, pipeline_layout[ENVELOPE],
         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(finish), &finish);
      hk_cmd_dispatch_unaligned(commandBuffer, 64, 1, 1);
   }

   for (uint32_t i = 0; i < build_count; ++i) {
      if (states[i].config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
         hk_vertex_table_build(commandBuffer, &states[i],
            vk_acceleration_structure_from_handle(states[i].build_info->dstAccelerationStructure));
   }
   return VK_SUCCESS;
}

static void
hk_native_build_config(VkDevice device,
                       struct vk_acceleration_structure_build_state *state)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   if (state->config.internal_type == VK_INTERNAL_BUILD_TYPE_PLOC &&
       dev->g17p_native_as)
      state->config.internal_type = VK_INTERNAL_BUILD_TYPE_HPLOC;
   if (state->build_info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR)
      state->config.internal_type = VK_INTERNAL_BUILD_TYPE_UPDATE;
   state->config.updateable =
      (state->build_info->flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) != 0;
}

static VkDeviceSize
hk_update_scratch_size(VkDevice device,
                       const struct vk_acceleration_structure_build_state *state)
{
   /* Leaf generation writes sort IDs, which alias native encode scratch.
    * Keep live refit parent/counter state beyond both workspaces. */
   struct hk_agx3_as_layout layout;
   if (!hk_gen1_as_layout_init(state->leaf_node_count, &layout))
      return 0;
   return (VkDeviceSize)state->scratch.size + (uint64_t)layout.node_bound * 16;
}

static void
hk_init_update_scratch(VkCommandBuffer commandBuffer,
                       const struct vk_acceleration_structure_build_state *states,
                       uint32_t count)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct vk_device *device = cmd->vk.base.device;
   /* Leaf generation counts the active leaves into a fresh header. */
   struct vk_ir_header header = {
      .min_bounds = {0x7fffffff, 0x7fffffff, 0x7fffffff},
      .max_bounds = {0x80000000, 0x80000000, 0x80000000},
      .dispatch_size_y = 1,
      .dispatch_size_z = 1,
   };
   for (unsigned i = 0; i < count; ++i)
      if (states[i].config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
         device->write_buffer_cp(commandBuffer,
                                 states[i].build_info->scratchData.deviceAddress +
                                    states[i].scratch.header_offset,
                                 &header, sizeof(header));
}

static const struct vk_acceleration_structure_build_ops hk_native_as_build_ops = {
   .get_build_config = hk_native_build_config,
   .get_as_size = hk_get_as_size,
   .get_encode_scratch_size = hk_get_encode_scratch_size,
   .get_update_scratch_size = hk_update_scratch_size,
   .init_update_scratch = hk_init_update_scratch,
   .encode = hk_encode,
};

static const struct vk_acceleration_structure_build_ops hk_as_build_ops = {
   .get_as_size = hk_get_as_size,
   .get_encode_scratch_size = hk_get_encode_scratch_size,
   .encode = hk_encode,
};

/* ---------------------------------------------------------------- */
/* vk_device hooks the common builder calls                          */

static void
hk_write_buffer_cp(VkCommandBuffer commandBuffer, VkDeviceAddress addr,
                   void *data, uint32_t size)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   struct {
      uint64_t dst;
      uint32_t count;
      uint32_t data[28];
   } args = {
      .dst = addr,
      .count = DIV_ROUND_UP(size, 4),
   };

   static_assert(sizeof(struct vk_ir_header) <= sizeof(args.data),
                 "common BVH header fits the write shader push constants");

   /* The only caller writes a vk_ir_header, which is 108 bytes.  Anything
    * larger is a change in the common builder, not a bigger buffer: refuse
    * rather than write a truncated header that would fail obscurely later. */
   assert(size <= sizeof(args.data));
   if (size > sizeof(args.data))
      return;
   memcpy(args.data, data, size);

   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result =
      get_pipeline_spv(dev, "hk-bvh-write", write_spv, sizeof(write_spv),
                       sizeof(args), &pipeline, &layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   vk_common_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline);
   vk_common_CmdPushConstants(commandBuffer, layout,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(args),
                              &args);
   hk_cmd_dispatch_unaligned(commandBuffer, args.count, 1, 1);
}

static void
hk_flush_buffer_write_cp(VkCommandBuffer commandBuffer)
{
   /* Header writes are compute dispatches and must be visible before the
    * common builder's first leaf dispatch. */
   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);
}

static void
hk_cmd_fill_buffer_addr(VkCommandBuffer commandBuffer, VkDeviceAddress addr,
                        VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   vk_meta_fill_buffer_addr(&cmd->vk, &dev->meta, addr, size, data);
}

/* ---------------------------------------------------------------- */
/* Device state                                                      */

void
hk_gen1_rq_finish(struct hk_device *dev)
{
   for(unsigned i=0;i<dev->gen1_rq_bound_maps;i++) {
      uint64_t *r=dev->gen1_rq_maps+3*i;
      agx_bo_bind(&dev->dev,NULL,r[2],r[1],0,DRM_ASAHI_BIND_UNBIND);
   }
   if(dev->gen1_rq_code_alias)agx_va_free(&dev->dev,dev->gen1_rq_code_alias,false);
   agx_bo_unreference(&dev->dev,dev->gen1_rq_code);
   agx_bo_unreference(&dev->dev,dev->gen1_rq_ift);
   agx_bo_unreference(&dev->dev, dev->gen1_rq_state);
   dev->gen1_rq_state = NULL;
   free(dev->gen1_rq_maps);free(dev->gen1_rq_user_image);
   dev->gen1_rq_maps=NULL;dev->gen1_rq_user_image=NULL;
   dev->gen1_rq_code=NULL;dev->gen1_rq_ift=NULL;
   dev->gen1_rq_code_alias=NULL;dev->gen1_rq_bound_maps=0;
   dev->gen1_native_rq=false;
   dev->gen1_vertex_directory = NULL;
   hk_gen1_sparse_finish(dev, &dev->gen1_vertex_directory_pages);
   if (dev->gen1_vertex_tables) {
      hash_table_foreach(dev->gen1_vertex_tables, entry)
         free(entry->data);
      _mesa_hash_table_destroy(dev->gen1_vertex_tables, NULL);
      dev->gen1_vertex_tables = NULL;
   }
   if (dev->host_as.pool) {
      dev->vk.dispatch_table.DestroyCommandPool(hk_device_to_handle(dev),
                                                dev->host_as.pool, NULL);
      dev->host_as.pool = VK_NULL_HANDLE;
   }
   dev->host_as.queue = NULL;
   if (dev->host_as.lock_ready) {
      simple_mtx_destroy(&dev->host_as.lock);
      dev->host_as.lock_ready = false;
   }
   if (dev->gen1_as_lock_ready) {
      util_dynarray_fini(&dev->gen1_ases);
      if (dev->gen1_as_ids) {
         _mesa_hash_table_destroy(dev->gen1_as_ids, NULL);
         dev->gen1_as_ids = NULL;
      }
      hk_gen1_as_windows_finish(dev);
      hk_gen1_sparse_finish(dev, &dev->gen1_as_table_pages);
      hk_gen1_sparse_finish(dev, &dev->gen1_as_metadata_pages);
      hk_gen1_sparse_finish(dev, &dev->gen1_as_address_pages);
      simple_mtx_destroy(&dev->gen1_as_lock);
      dev->gen1_as_lock_ready=false;
   }
   dev->gen1_as_metadata = NULL;
   dev->gen1_as_addresses = NULL;
   dev->gen1_rq_instances=false;
}

static bool
hk_gen1_rq_generated_slot_matches(const uint8_t *code,
                                  const uint64_t rows[AGX3_RT_GEN1_RUNTIME_MAP_ROWS][3],
                                  unsigned slot, const uint8_t *body, size_t body_length,
                                  const uint8_t *constants, size_t constants_length)
{
   const uint8_t *base = code + rows[slot * 3][0];
   const uint8_t *constant_base = code + rows[slot * 3 + 1][0];
   if (memcmp(base, body, body_length) || memcmp(constant_base, constants, constants_length))
      return false;
   for (size_t i = body_length; i < rows[slot * 3][1]; ++i)
      if (base[i]) return false;
   for (size_t i = constants_length; i < rows[slot * 3 + 1][1]; ++i)
      if (constant_base[i]) return false;
   return true;
}

static bool
hk_gen1_rq_init_generated(struct hk_device *dev)
{
   static uint64_t rows[AGX3_RT_GEN1_RUNTIME_MAP_ROWS][3];
   static uint8_t body[0x4000];
   uint8_t entry[AGX3_RT_GEN1_ENTRY_MAX_SIZE], constants[AGX3_RT_GEN1_DEFAULT_CONSTANTS_SIZE];
   const uint8_t *table;
   size_t body_length, entry_length, table_length;
   dev->gen1_rq_maps = calloc(630, 3 * sizeof(uint64_t));
   dev->gen1_rq_user_image = malloc(0x60000);
   if (!dev->gen1_rq_maps || !dev->gen1_rq_user_image ||
       !agx3_rt_gen1_runtime_code_map(rows) ||
       !agx3_rt_gen1_runtime_user_image(hk_gen1_ray_query_scs_layout(dev),
                                        dev->gen1_rq_user_image, 0x60000))
      goto fail;
   memcpy(dev->gen1_rq_maps, rows, sizeof(rows));
   agx3_rt_gen1_runtime_uniforms(dev->gen1_rq_uniforms);
   dev->gen1_rq_code_alias = agx_va_alloc(&dev->dev, 0x1a224000, 0x4000,
      AGX_VA_USC | AGX_VA_FIXED, UINT64_C(0x10010000000));
   if (!dev->gen1_rq_code_alias) goto fail;
   dev->gen1_rq_code = agx_bo_create(&dev->dev, 11108352, 0x4000,
      AGX_BO_EXEC | AGX_BO_LOW_VA | AGX_BO_WRITEBACK, "Gen1 ray query programs");
   dev->gen1_rq_ift = agx_bo_create(&dev->dev, 0x4000, 0x4000,
      AGX_BO_WRITEBACK, "Gen1 intersection table");
   if (!dev->gen1_rq_code || !dev->gen1_rq_ift) goto fail;
   uint8_t *code = agx_bo_map(dev->gen1_rq_code);
   uint8_t *ift = agx_bo_map(dev->gen1_rq_ift);
   if (!code || !ift) goto fail;
   memset(code, 0, 11108352);
   for (unsigned slot = 0; slot < AGX3_RT_GEN1_RUNTIME_SLOTS; ++slot) {
      if (!agx3_rt_gen1_stub_body_generate(slot, body, sizeof(body), &body_length)) goto fail;
      memcpy(code + rows[slot * 3][0], body, body_length);
      if (!agx3_rt_gen1_runtime_constants(slot, &table, &table_length) || table_length > rows[slot * 3 + 1][1])
         goto fail;
      memcpy(code + rows[slot * 3 + 1][0], table, table_length);
   }
   if (!agx3_rt_gen1_instance_push_body_generate(body, sizeof(body), &body_length)) goto fail;
   agx3_rt_gen1_instance_push_constants(&table, &table_length);
   memcpy(code + rows[AGX3_RT_GEN1_INSTANCE_PUSH_SLOT * 3][0], body, body_length);
   memcpy(code + rows[AGX3_RT_GEN1_INSTANCE_PUSH_SLOT * 3 + 1][0], table, table_length);
   for (unsigned i = 0, slot; agx3_rt_gen1_clique_slot(i, &slot); ++i) {
      if (!agx3_rt_gen1_clique_body_generate(slot, body, sizeof(body), &body_length)) goto fail;
      memcpy(code + rows[slot * 3][0], body, body_length);
   }
   if (!agx3_rt_gen1_default_body_generate(body, &body_length)) goto fail;
   agx3_rt_gen1_default_constants(constants);
   memcpy(code + rows[0][0], body, body_length);
   memcpy(code + rows[1][0], constants, sizeof(constants));
   if (!agx3_rt_gen1_triangle_body_generate(body, sizeof(body), &body_length)) goto fail;
   agx3_rt_gen1_triangle_constants(&table, &table_length);
   memcpy(code + rows[AGX3_RT_GEN1_TRIANGLE_SLOT * 3][0], body, body_length);
   memcpy(code + rows[AGX3_RT_GEN1_TRIANGLE_SLOT * 3 + 1][0], table, table_length);
   if (!agx3_rt_gen1_aabb_body_generate(body, sizeof(body), &body_length)) goto fail;
   agx3_rt_gen1_aabb_constants(&table, &table_length);
   memcpy(code + rows[AGX3_RT_GEN1_AABB_SLOT * 3][0], body, body_length);
   memcpy(code + rows[AGX3_RT_GEN1_AABB_SLOT * 3 + 1][0], table, table_length);
   agx3_rt_gen1_runtime_ift(ift);
   for (unsigned slot = 0; slot < AGX3_RT_GEN1_RUNTIME_SLOTS; ++slot) {
      uint8_t *slot_entry = code + rows[slot * 3 + 2][0];
      if (!agx3_rt_gen1_runtime_entry_generate(slot, entry, &entry_length) || entry_length > 0x4000)
         goto fail;
      memcpy(slot_entry, entry, entry_length);
   }
   if (!agx3_rt_gen1_default_body_generate(body, &body_length)) goto fail;
   agx3_rt_gen1_default_constants(constants);
   if (!hk_gen1_rq_generated_slot_matches(code, rows, 0, body, body_length, constants, sizeof(constants)))
      goto fail;
   if (!agx3_rt_gen1_triangle_body_generate(body, sizeof(body), &body_length)) goto fail;
   agx3_rt_gen1_triangle_constants(&table, &table_length);
   if (!hk_gen1_rq_generated_slot_matches(code, rows, AGX3_RT_GEN1_TRIANGLE_SLOT, body, body_length, table, table_length))
      goto fail;
   if (!agx3_rt_gen1_aabb_body_generate(body, sizeof(body), &body_length)) goto fail;
   agx3_rt_gen1_aabb_constants(&table, &table_length);
   if (!hk_gen1_rq_generated_slot_matches(code, rows, AGX3_RT_GEN1_AABB_SLOT, body, body_length, table, table_length))
      goto fail;
   if (!agx3_rt_gen1_instance_push_body_generate(body, sizeof(body), &body_length)) goto fail;
   agx3_rt_gen1_instance_push_constants(&table, &table_length);
   if (!hk_gen1_rq_generated_slot_matches(code, rows, AGX3_RT_GEN1_INSTANCE_PUSH_SLOT, body, body_length, table, table_length))
      goto fail;
   for (unsigned i = 0, slot; agx3_rt_gen1_clique_slot(i, &slot); ++i) {
      if (!agx3_rt_gen1_clique_body_generate(slot, body, sizeof(body), &body_length) ||
          !agx3_rt_gen1_runtime_constants(slot, &table, &table_length) ||
          !hk_gen1_rq_generated_slot_matches(code, rows, slot, body, body_length, table, table_length))
         goto fail;
   }
   if (util_has_cache_ops()) {
      util_flush_range(code, 11108352);
      util_flush_range(ift, 0x4000);
   }
   for (unsigned i = 0; i < 630; i++) {
      uint64_t *r = dev->gen1_rq_maps + 3 * i;
      if (agx_bo_bind(&dev->dev, dev->gen1_rq_code, r[2], r[1], r[0],
                      DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE))
         goto fail;
      dev->gen1_rq_bound_maps++;
   }
   dev->gen1_native_rq = true;
   return true;
fail:
   hk_gen1_rq_finish(dev);
   return false;
}

VkResult
hk_device_init_accel_struct_state(struct hk_device *dev)
{
   if (hk_agx3_as_format_for_chip(dev->dev.chip) == HK_AGX3_AS_FORMAT_M5)
      return hk_m5_backend_device_init_accel_struct_state(dev);

   /* The native state below is large and lives as long as the device: the
    * ray query program image with its fixed binds, the AS tables, and, through
    * gen1_native_graphics_rq/fragment_rq, a render SCS reservation in every
    * queue's kernel context that sizes that context's render free list. A
    * device enabling no ray-tracing feature cannot reach any of it, so it
    * keeps the state every device without native ray tracing has. Shader
    * cache keys tell the two kinds of device apart (HK_FEAT_RAY_TRACING). */
   dev->g17p_native_as = hk_gen1_as_enabled(&dev->dev) &&
                         hk_rt_features_enabled(&dev->vk.enabled_features);
   /* AS-only G16S devices must never allocate the generated intersectors,
    * fixed ray program binds, IFT or render/compute SCS state. The kernel
    * deliberately rejects traversal on the 25G83 firmware variant. */
   const bool native_rq = hk_g17p_rt_enabled(&dev->dev) &&
                         hk_rt_features_enabled(&dev->vk.enabled_features);
   if (native_rq && !hk_gen1_rq_init_generated(dev))
      return vk_error(dev, VK_ERROR_INITIALIZATION_FAILED);
   dev->gen1_rq_instances = dev->gen1_native_rq;
   dev->gen1_rq_candidates = dev->gen1_rq_instances;
   dev->gen1_native_fragment_rq = dev->gen1_rq_candidates &&
      (dev->dev.params.features & DRM_ASAHI_FEATURE_RENDER_RAY_QUERY);
   dev->gen1_native_graphics_rq = dev->gen1_native_fragment_rq;
   if (dev->gen1_native_rq) {
      /* Finalize after selecting the candidate profile: the initial image
       * was generated before gen1_rq_instances/candidates were enabled.
       * Neither the generated intersectors nor the query lowering stores to
       * this image. SCS stacks are allocated separately by the kernel, and
       * hk_gen1_upload_as_table() owns the writable AS query bank.
       * Preserve the full legacy extent (including zero padding) here.
       */
      struct agx_scs_layout_gen2 scs;
      ASSERTED bool valid = agx_scs_layout_gen1_init(
         hk_gen1_ray_query_scs_layout(dev), &scs);
      assert(valid);
      const uint64_t size = MAX2(UINT64_C(0x60000),
                                ALIGN_POT((uint64_t)scs.end[12], 0x4000));
      const bool cached = hk_g17p_cached_cmd_memory(&dev->dev);
      dev->gen1_rq_state = agx_bo_create(&dev->dev, size, 0x4000,
         AGX_BO_READONLY | (cached ? AGX_BO_WRITEBACK : 0),
         "Gen1 ray query user image");
      if (!dev->gen1_rq_state) {
         hk_gen1_rq_finish(dev);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
      uint8_t *state = agx_bo_map(dev->gen1_rq_state);
      if (!state) {
         hk_gen1_rq_finish(dev);
         return vk_error(dev, VK_ERROR_MEMORY_MAP_FAILED);
      }
      memcpy(state, dev->gen1_rq_user_image, 0x60000);
      if (size > 0x60000)
         memset(state + 0x60000, 0, size - 0x60000);
      agx_scs_layout_gen1_parameters(&scs, state + 0x4000);
      if (cached)
         util_flush_range(state, size);
   }
   if (dev->g17p_native_as) {
      simple_mtx_init(&dev->gen1_as_lock, mtx_plain);
      util_dynarray_init(&dev->gen1_ases, NULL);
      util_dynarray_init(&dev->gen1_as_windows, NULL);
      dev->gen1_as_free_head = UINT32_MAX;
      dev->gen1_as_lock_ready=true;
      dev->gen1_vertex_tables = _mesa_pointer_hash_table_create(NULL);
      dev->gen1_as_ids = _mesa_pointer_hash_table_create(NULL);
      if (hk_gen1_sparse_init(dev, &dev->gen1_vertex_directory_pages, 16,
                              AGX_GEN1_VERTEX_DIRECTORY_VA, 0,
                              DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE,
                              "Gen1 vertex table directory") &&
          hk_gen1_sparse_entry(dev, &dev->gen1_vertex_directory_pages,
                               HK_GEN1_AS_TABLE_ENTRIES - 1)) {
         dev->gen1_vertex_directory = *util_dynarray_element(
            &dev->gen1_vertex_directory_pages.pages, struct agx_bo *, 0);
      } else {
         mesa_loge("Gen1 vertex table directory unavailable");
         hk_gen1_sparse_finish(dev, &dev->gen1_vertex_directory_pages);
         dev->gen1_vertex_directory = NULL;
      }
      /* GPU-only live metadata: builds/copies initialize all64 bytes. */
      const bool tables_ok =
         hk_gen1_sparse_init(dev, &dev->gen1_as_metadata_pages,
                             LIBAGX_GEN1_AS_METADATA_STRIDE, 0, 0,
                             DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE,
                             "Gen1 AS live metadata") &&
         hk_gen1_sparse_init(dev, &dev->gen1_as_address_pages, sizeof(uint64_t),
                             0, 0, DRM_ASAHI_BIND_READ,
                             "Gen1 AS creation addresses") &&
         hk_gen1_sparse_init(dev, &dev->gen1_as_table_pages,
                             HK_GEN1_AS_TABLE_ENTRY_BYTES, 0, 0,
                             DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE,
                             "Gen1 AS table") &&
         hk_gen1_sparse_entry(dev, &dev->gen1_as_metadata_pages, 0) &&
         hk_gen1_sparse_entry(dev, &dev->gen1_as_address_pages, 0) &&
         hk_gen1_sparse_entry(dev, &dev->gen1_as_table_pages,
                              HK_GEN1_AS_TABLE_ENTRIES - 1);
      if (!tables_ok || !dev->gen1_as_ids) {
         hk_gen1_rq_finish(dev);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
      dev->gen1_as_metadata = *util_dynarray_element(
         &dev->gen1_as_metadata_pages.pages, struct agx_bo *, 0);
      dev->gen1_as_addresses = *util_dynarray_element(
         &dev->gen1_as_address_pages.pages, struct agx_bo *, 0);
      if (dev->vk.enabled_features.accelerationStructureHostCommands) {
         VkResult result = hk_host_as_init(dev);
         if (result != VK_SUCCESS) {
            hk_gen1_rq_finish(dev);
            return result;
         }
      }
   }
   dev->accel_struct_args = (struct vk_acceleration_structure_build_args){
      .subgroup_size = 32,
      .bvh_bounds_offset = dev->g17p_native_as ? 0x50 : offsetof(struct hk_bvh_header, bounds),
      /* This software node format does not store propagated cull flags. */
      .root_flags_offset = 0,
      .morton_sort_workgroup_size = 512,
      .morton_sort_kvs_per_thread = 2,
      .propagate_cull_flags = false,
      .has_update = dev->g17p_native_as,
      .morton_phase_dispatch = dev->g17p_native_as,
      /* On G17P the shuffle search is faster up to 4K leaves and slower
       * from 16K (gpubench rt.asbuild on the Neo). */
      .hploc_shuffle_max_leaves = dev->g17p_native_as ? 4096 : 0,
   };

   dev->vk.as_build_ops = dev->g17p_native_as ? &hk_native_as_build_ops : &hk_as_build_ops;
   dev->vk.write_buffer_cp = hk_write_buffer_cp;
   dev->vk.flush_buffer_write_cp = hk_flush_buffer_write_cp;
   dev->vk.cmd_dispatch_unaligned = hk_cmd_dispatch_unaligned;
   dev->vk.cmd_fill_buffer_addr = hk_cmd_fill_buffer_addr;

   return VK_SUCCESS;
}

/* ---------------------------------------------------------------- */
/* Entry points                                                      */

VKAPI_ATTR void VKAPI_CALL
hk_GetAccelerationStructureBuildSizesKHR(
   VkDevice _device, VkAccelerationStructureBuildTypeKHR buildType,
   const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
   const uint32_t *pMaxPrimitiveCounts,
   VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
   VK_FROM_HANDLE(hk_device, m5_dispatch_dev, _device);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5) {
      hk_m5_backend_GetAccelerationStructureBuildSizesKHR(_device, buildType, pBuildInfo, pMaxPrimitiveCounts, pSizeInfo);
      return;
   }

   VK_FROM_HANDLE(hk_device, dev, _device);

   if (dev->g17p_native_as) {
      uint64_t leaves = 0;
      for (uint32_t i = 0; i < pBuildInfo->geometryCount; ++i)
         leaves += pMaxPrimitiveCounts[i];
      if (!leaves) {
         struct hk_agx3_as_layout layout;
         hk_gen1_as_layout_init(0, &layout);
         pSizeInfo->accelerationStructureSize = layout.as_bytes;
         pSizeInfo->buildScratchSize = 256;
         pSizeInfo->updateScratchSize = 256;
         return;
      }
   }

   vk_get_as_build_sizes(_device, buildType, pBuildInfo, pMaxPrimitiveCounts,
                         pSizeInfo, &dev->accel_struct_args);
}

static void
hk_cmd_build_acceleration_structures(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos,
   const VkDeviceAddress *pIndirectDeviceAddresses,
   const uint32_t *pIndirectStrides)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   uint32_t empty_count = 0;
   if (dev->g17p_native_as) {
      for (uint32_t i = 0; i < infoCount; ++i) {
         const VkAccelerationStructureGeometryKHR *geometry =
            pInfos[i].geometryCount ? (pInfos[i].pGeometries ?
               pInfos[i].pGeometries : pInfos[i].ppGeometries[0]) : NULL;
         uint64_t primitives = 0;
         bool compatible = geometry != NULL ||
            (pInfos[i].type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR &&
             pInfos[i].geometryCount == 0);
         for (uint32_t g = 0; g < pInfos[i].geometryCount; ++g) {
            const VkAccelerationStructureGeometryKHR *next =
               pInfos[i].pGeometries ? &pInfos[i].pGeometries[g] :
                                      pInfos[i].ppGeometries[g];
            compatible &= next->geometryType == geometry->geometryType;
            primitives += ppBuildRangeInfos[i][g].primitiveCount;
         }
         struct hk_agx3_as_layout native;
         bool triangles =
            pInfos[i].type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR &&
            (!geometry || geometry->geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR);
         /* Native AS construction supports all three geometry kinds without
          * intersectors. Traversal state gates query execution, not building
          * the AABB records or the TLAS instance records it will later read. */
         bool aabbs = geometry &&
            pInfos[i].type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR &&
            geometry->geometryType == VK_GEOMETRY_TYPE_AABBS_KHR;
         bool instances = geometry &&
            pInfos[i].geometryCount == 1 &&
            pInfos[i].type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR &&
            geometry->geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR;
         if ((pInfos[i].mode != VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR &&
              pInfos[i].mode != VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) ||
             !compatible || (!triangles && !instances && !aabbs) || primitives > UINT32_MAX ||
             !hk_gen1_as_layout_init(primitives, &native)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         empty_count += primitives == 0;
      }
   }

   const uint32_t nonempty_count = infoCount - empty_count;
   VkAccelerationStructureBuildGeometryInfoKHR *nonempty_infos = NULL;
   const VkAccelerationStructureBuildRangeInfoKHR **nonempty_ranges = NULL;
   VkDeviceAddress *nonempty_indirect = NULL;
   uint32_t *nonempty_strides = NULL;
   if (empty_count && nonempty_count) {
      nonempty_infos = malloc(nonempty_count * sizeof(*nonempty_infos));
      nonempty_ranges = malloc(nonempty_count * sizeof(*nonempty_ranges));
      if (pIndirectDeviceAddresses) {
         nonempty_indirect = malloc(nonempty_count * sizeof(*nonempty_indirect));
         nonempty_strides = malloc(nonempty_count * sizeof(*nonempty_strides));
      }
      if (!nonempty_infos || !nonempty_ranges ||
          (pIndirectDeviceAddresses && (!nonempty_indirect || !nonempty_strides))) {
         free(nonempty_infos);
         free(nonempty_ranges);
         free(nonempty_indirect);
         free(nonempty_strides);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return;
      }
   }


   hk_sync_end_cs_if_gfx_waits(cmd);
   const bool was_sync_as_build = cmd->sync.as_build;
   cmd->sync.as_build = true;
   const bool was_as_build = cmd->state.as_build_recording;
   cmd->state.as_build_recording = dev->dev.chip == AGX_CHIP_G17P;
   /* Launches of separate build calls, or of a build and the application's
    * work before it, never overlap (hk_dispatch_g17p_native()). */
   if (cmd->current_cs.cs)
      cmd->current_cs.cs->app_dispatch_end = NULL;

   /* A build binds its own compute pipelines and push constants, so the
    * application's compute state has to survive it. */
   struct hk_meta_save save;
   hk_meta_begin(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);

   if (empty_count) {
      uint32_t n = 0;
      for (uint32_t i = 0; i < infoCount; ++i) {
         uint64_t primitives = 0;
         for (uint32_t g = 0; g < pInfos[i].geometryCount; ++g)
            primitives += ppBuildRangeInfos[i][g].primitiveCount;
         if (!primitives) {
            VK_FROM_HANDLE(vk_acceleration_structure, dst,
                           pInfos[i].dstAccelerationStructure);
            uint32_t id = hk_gen1_as_id(dev, dst);
            uint64_t metadata = hk_gen1_as_metadata_addr_for_id(cmd, id);
            if (!metadata)
               break;
            libagx_gen1_as_empty(cmd, agx_1d(1), AGX_BARRIER_ALL,
               hk_as_native_va(dst), metadata, id, pInfos[i].flags,
               pInfos[i].geometryCount ? vk_get_as_geometry_type(&pInfos[i]) : VK_GEOMETRY_TYPE_TRIANGLES_KHR);
         } else if (nonempty_infos) {
            if (nonempty_indirect) {
               nonempty_indirect[n] = pIndirectDeviceAddresses[i];
               nonempty_strides[n] = pIndirectStrides[i];
            }
            nonempty_infos[n] = pInfos[i];
            nonempty_ranges[n++] = ppBuildRangeInfos[i];
         }
      }
      hk_flush_buffer_write_cp(commandBuffer);
   }
   if (nonempty_count)
      vk_cmd_build_acceleration_structures_indirect(commandBuffer, &dev->vk, &dev->meta,
         nonempty_count, nonempty_infos ? nonempty_infos : pInfos,
         nonempty_ranges ? nonempty_ranges : ppBuildRangeInfos,
         nonempty_indirect ? nonempty_indirect : pIndirectDeviceAddresses,
         nonempty_strides ? nonempty_strides : pIndirectStrides,
         &dev->accel_struct_args);

   hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
   cmd->state.as_build_recording = was_as_build;
   cmd->sync.as_build = was_sync_as_build;
   if (cmd->current_cs.cs)
      cmd->current_cs.cs->app_dispatch_end = NULL;
   free(nonempty_infos);
   free(nonempty_ranges);
   free(nonempty_indirect);
   free(nonempty_strides);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(hk_cmd_buffer, m5_dispatch_cmd, commandBuffer);
   struct hk_device *m5_dispatch_dev = hk_cmd_buffer_device(m5_dispatch_cmd);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5) {
      hk_m5_backend_CmdBuildAccelerationStructuresKHR(commandBuffer, infoCount, pInfos, ppBuildRangeInfos);
      return;
   }

   hk_cmd_build_acceleration_structures(commandBuffer, infoCount, pInfos,
                                        ppBuildRangeInfos, NULL, NULL);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBuildAccelerationStructuresIndirectKHR(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkDeviceAddress *pIndirectDeviceAddresses,
   const uint32_t *pIndirectStrides,
   const uint32_t *const *ppMaxPrimitiveCounts)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   uint32_t geometry_count = 0;
   for (uint32_t i = 0; i < infoCount; ++i)
      geometry_count += pInfos[i].geometryCount;
   VkAccelerationStructureBuildRangeInfoKHR *ranges =
      calloc(MAX2(geometry_count, 1), sizeof(*ranges));
   const VkAccelerationStructureBuildRangeInfoKHR **range_ptrs =
      malloc(MAX2(infoCount, 1) * sizeof(*range_ptrs));
   if (!ranges || !range_ptrs) {
      free(ranges);
      free(range_ptrs);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   for (uint32_t i = 0, k = 0; i < infoCount; ++i) {
      range_ptrs[i] = &ranges[k];
      for (uint32_t g = 0; g < pInfos[i].geometryCount; ++g)
         ranges[k++].primitiveCount = ppMaxPrimitiveCounts[i][g];
   }
   hk_cmd_build_acceleration_structures(commandBuffer, infoCount, pInfos, range_ptrs,
                                        pIndirectDeviceAddresses, pIndirectStrides);
   free(ranges);
   free(range_ptrs);
}

VKAPI_ATTR void VKAPI_CALL
hk_GetDeviceAccelerationStructureCompatibilityKHR(
   VkDevice _device, const VkAccelerationStructureVersionInfoKHR *pVersionInfo,
   VkAccelerationStructureCompatibilityKHR *pCompatibility)
{
   VK_FROM_HANDLE(hk_device, m5_dispatch_dev, _device);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5) {
      hk_m5_backend_GetDeviceAccelerationStructureCompatibilityKHR(_device, pVersionInfo, pCompatibility);
      return;
   }

   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_physical_device *pdev = hk_device_physical(dev);

   const uint8_t format_uuid[VK_UUID_SIZE] = LIBAGX_GEN1_AS_SERIAL_FORMAT_UUID_BYTES;
   bool compat = dev->g17p_native_as &&
      memcmp(pVersionInfo->pVersionData, pdev->vk.properties.driverUUID, VK_UUID_SIZE) == 0 &&
      memcmp(pVersionInfo->pVersionData + VK_UUID_SIZE, format_uuid, VK_UUID_SIZE) == 0;
   *pCompatibility = compat
                        ? VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR
                        : VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

/*
 * Host-side builds and the copy/serialize family are unimplemented and, in
 * the case of the host builds, unadvertised: accelerationStructureHostCommands
 * stays false, so a conformant application never reaches them.  They fail
 * loudly rather than silently returning success on an untouched structure.
 */
/* Moves the image at image (serialized: inside a serialized blob) by delta
 * slots relative to its traversal base, then restores the instance-record
 * slots the node scan cannot tell apart from nodes, from source. bytes bounds
 * the image. Nothing is recorded for delta 0. */
static void
hk_gen1_as_rebase(struct hk_cmd_buffer *cmd, uint64_t image, bool serialized,
                  uint64_t bytes, uint32_t delta, uint64_t source,
                  bool source_serialized, uint64_t metadata)
{
   if (!delta)
      return;
   libagx_gen1_as_rebase(cmd, agx_1d(DIV_ROUND_UP(bytes, LIBAGX_GEN1_AS_SLOT_BYTES)),
                         AGX_BARRIER_ALL, image, serialized, delta);
   libagx_gen1_as_restore_records(cmd, agx_1d(MAX2(bytes / 128, 1)),
                                  AGX_BARRIER_ALL, image, serialized, source,
                                  source_serialized, metadata);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyAccelerationStructureKHR(
   VkCommandBuffer commandBuffer, const VkCopyAccelerationStructureInfoKHR *info)
{
   VK_FROM_HANDLE(hk_cmd_buffer, m5_dispatch_cmd, commandBuffer);
   struct hk_device *m5_dispatch_dev = hk_cmd_buffer_device(m5_dispatch_cmd);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5) {
      hk_m5_backend_CmdCopyAccelerationStructureKHR(commandBuffer, info);
      return;
   }

   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, src, info->src);
   VK_FROM_HANDLE(vk_acceleration_structure, dst, info->dst);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!dev->g17p_native_as ||
       (info->mode != VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR &&
        info->mode != VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (src->size < LIBAGX_GEN1_AS_EMPTY_BYTES ||
       DIV_ROUND_UP(src->size, 16) > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   uint32_t dst_id = hk_gen1_as_id(dev, dst);
   uint64_t dst_metadata = hk_gen1_as_metadata_addr_for_id(cmd, dst_id);
   uint64_t src_metadata = hk_gen1_as_metadata_addr(cmd, src);
   if (!dst_metadata || !src_metadata)
      return;
   libagx_gen1_as_copy(cmd, agx_1d(DIV_ROUND_UP(src->size, 16)),
                      AGX_BARRIER_ALL, hk_as_native_va(dst),
                      hk_as_native_va(src), dst_metadata, src_metadata, dst_id,
                      info->mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR);
   hk_gen1_as_rebase(cmd, hk_as_native_va(dst), false, src->size,
                     hk_gen1_as_slot_bias(dev, dst) - hk_gen1_as_slot_bias(dev, src),
                     hk_as_native_va(src), false, src_metadata);
   if (info->mode == VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR &&
       dev->gen1_vertex_tables) {
      struct hk_vertex_table_layout table;
      bool found = false;
      simple_mtx_lock(&dev->gen1_as_lock);
      struct hash_entry *entry = _mesa_hash_table_search(dev->gen1_vertex_tables, src);
      if (entry) {
         table = *(struct hk_vertex_table_layout *)entry->data;
         found = true;
      }
      simple_mtx_unlock(&dev->gen1_as_lock);
      if (found && table.offset + table.bytes <= dst->size)
         hk_vertex_table_publish(commandBuffer, dev, dst, &table);
   }
}

static uint64_t
hk_gen1_as_serial_uuid(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_physical_device *pdev = hk_device_physical(dev);
   const uint8_t format_uuid[VK_UUID_SIZE] = LIBAGX_GEN1_AS_SERIAL_FORMAT_UUID_BYTES;
   uint8_t uuid[2 * VK_UUID_SIZE];
   memcpy(uuid, pdev->vk.properties.driverUUID, VK_UUID_SIZE);
   memcpy(uuid + VK_UUID_SIZE, format_uuid, VK_UUID_SIZE);
   return hk_pool_upload(cmd, uuid, sizeof(uuid), 16);
}

static int
hk_gen1_as_address_compare(const void *a, const void *b)
{
   const struct libagx_gen1_as_address *x = a, *y = b;
   return x->address < y->address ? -1 : x->address > y->address ? 1 :
          x->resource_id < y->resource_id ? -1 : x->resource_id > y->resource_id;
}

static uint64_t
hk_gen1_as_address_map(struct hk_cmd_buffer *cmd, uint32_t *count)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   simple_mtx_lock(&dev->gen1_as_lock);
   uint32_t bound = util_dynarray_num_elements(&dev->gen1_ases,
                                              struct hk_gen1_as_slot);
   struct agx_ptr map = hk_pool_alloc(cmd,
      MAX2(bound, 1) * sizeof(struct libagx_gen1_as_address), 16);
   *count = 0;
   if (map.cpu) {
      struct libagx_gen1_as_address *entries = map.cpu;
      uint32_t index = 0;
      util_dynarray_foreach(&dev->gen1_ases, struct hk_gen1_as_slot, slot) {
         uint64_t address = slot->as ? hk_gen1_as_bound_va(slot->as) : 0;
         if (address) entries[(*count)++] = (struct libagx_gen1_as_address) {
            .address = address, .resource_id = hk_gen1_as_index_to_id(index),
         };
         ++index;
      }
      qsort(entries, *count, sizeof(*entries), hk_gen1_as_address_compare);
   }
   simple_mtx_unlock(&dev->gen1_as_lock);
   return map.cpu ? map.gpu : 0;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyAccelerationStructureToMemoryKHR(VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureToMemoryInfoKHR *info)
{
   VK_FROM_HANDLE(hk_cmd_buffer, m5_dispatch_cmd, commandBuffer);
   struct hk_device *m5_dispatch_dev = hk_cmd_buffer_device(m5_dispatch_cmd);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5) {
      hk_m5_backend_CmdCopyAccelerationStructureToMemoryKHR(commandBuffer, info);
      return;
   }

   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, src, info->src);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!dev->g17p_native_as || info->mode != VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (src->size < LIBAGX_GEN1_AS_EMPTY_BYTES ||
       DIV_ROUND_UP(src->size, 16) > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   uint64_t metadata = hk_gen1_as_metadata_addr(cmd, src);
   uint64_t uuid = hk_gen1_as_serial_uuid(cmd);
   if (!metadata || !uuid) return;
   libagx_gen1_as_serialize(cmd, agx_1d(DIV_ROUND_UP(src->size, 16)),
      AGX_BARRIER_ALL, info->dst.deviceAddress, hk_as_native_va(src),
      metadata, uuid, src->size);
   /* Serialized images are stored relative to their own first slot. */
   hk_gen1_as_rebase(cmd, info->dst.deviceAddress, true, src->size,
                     -hk_gen1_as_slot_bias(dev, src), hk_as_native_va(src),
                     false, metadata);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdCopyMemoryToAccelerationStructureKHR(VkCommandBuffer commandBuffer,
   const VkCopyMemoryToAccelerationStructureInfoKHR *info)
{
   VK_FROM_HANDLE(hk_cmd_buffer, m5_dispatch_cmd, commandBuffer);
   struct hk_device *m5_dispatch_dev = hk_cmd_buffer_device(m5_dispatch_cmd);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5) {
      hk_m5_backend_CmdCopyMemoryToAccelerationStructureKHR(commandBuffer, info);
      return;
   }

   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, dst, info->dst);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!dev->g17p_native_as || info->mode != VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   if (dst->size < LIBAGX_GEN1_AS_EMPTY_BYTES ||
       DIV_ROUND_UP(dst->size, 16) > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   uint32_t id = hk_gen1_as_id(dev, dst), map_count;
   uint64_t metadata = hk_gen1_as_metadata_addr_for_id(cmd, id);
   uint64_t uuid = hk_gen1_as_serial_uuid(cmd);
   uint64_t map = hk_gen1_as_address_map(cmd, &map_count);
   if (!metadata || !uuid || !map) return;
   uint32_t invocations = DIV_ROUND_UP(dst->size, 16);
   libagx_gen1_as_deserialize(cmd, agx_1d(invocations), AGX_BARRIER_ALL,
      hk_as_native_va(dst), info->src.deviceAddress, metadata, uuid, dst->size, id);
   simple_mtx_lock(&dev->gen1_as_lock);
   const uint32_t id_bound = hk_gen1_as_index_to_id(
      util_dynarray_num_elements(&dev->gen1_ases, struct hk_gen1_as_slot));
   simple_mtx_unlock(&dev->gen1_as_lock);
   libagx_gen1_as_deserialize_references(cmd,
      agx_1d(dst->size / 128), AGX_BARRIER_ALL,
      hk_as_native_va(dst), info->src.deviceAddress, map, map_count,
      dev->gen1_as_address_pages.va->addr, id_bound, uuid, dst->size);
   hk_gen1_as_rebase(cmd, hk_as_native_va(dst), false, dst->size,
                     hk_gen1_as_slot_bias(dev, dst), info->src.deviceAddress,
                     true, metadata);
}

#define HK_HOST_AS_QUEUE_INDEX 0xffffu

struct hk_host_as_run {
   struct hk_device *dev;
   VkCommandBuffer cmd;
   VkFence fence;
   struct util_dynarray bos;
   bool locked;
};

static VkResult
hk_host_as_init(struct hk_device *dev)
{
   struct hk_physical_device *pdev = hk_device_physical(dev);
   uint32_t family = 0;
   while (family < pdev->queue_family_count &&
          !(pdev->queue_families[family].queue_flags & VK_QUEUE_COMPUTE_BIT))
      ++family;
   if (family == pdev->queue_family_count)
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);

   simple_mtx_init(&dev->host_as.lock, mtx_plain);
   dev->host_as.lock_ready = true;

   /* hk_queue_init() takes the priority of queue index_in_family from
    * pQueuePriorities unless a global priority is chained. This queue has no
    * priority array of HK_HOST_AS_QUEUE_INDEX + 1 entries, so chain the
    * class the 1.0 priority maps to instead. */
   const float priority = 1.0f;
   const VkDeviceQueueGlobalPriorityCreateInfoKHR priority_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
      .globalPriority = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR,
   };
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .pNext = &priority_info,
      .queueFamilyIndex = family,
      .queueCount = HK_HOST_AS_QUEUE_INDEX + 1,
      .pQueuePriorities = &priority,
   };
   VkResult result = hk_queue_init(dev, &queue_info, HK_HOST_AS_QUEUE_INDEX);
   if (result != VK_SUCCESS)
      return result;
   dev->host_as.queue = container_of(
      list_last_entry(&dev->vk.queues, struct vk_queue, link), struct hk_queue, vk);

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = family,
   };
   return dev->vk.dispatch_table.CreateCommandPool(
      hk_device_to_handle(dev), &pool_info, NULL, &dev->host_as.pool);
}

static void
hk_host_as_finish(struct hk_host_as_run *run)
{
   struct hk_device *dev = run->dev;
   VkDevice device = hk_device_to_handle(dev);
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   if (run->cmd) {
      disp->FreeCommandBuffers(device, dev->host_as.pool, 1, &run->cmd);
   }
   if (run->fence)
      disp->DestroyFence(device, run->fence, NULL);
   util_dynarray_foreach(&run->bos, struct agx_bo *, bo)
      agx_bo_unreference(&dev->dev, *bo);
   util_dynarray_fini(&run->bos);
   if (run->locked) {
      run->locked = false;
      simple_mtx_unlock(&dev->host_as.lock);
   }
}

static VkResult
hk_host_as_begin(struct hk_device *dev, struct hk_host_as_run *run)
{
   VkDevice device = hk_device_to_handle(dev);
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   *run = (struct hk_host_as_run){.dev = dev};
   util_dynarray_init(&run->bos, NULL);
   if (!dev->g17p_native_as || !dev->host_as.queue || !dev->host_as.pool)
      return vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);

   /* Recording command buffers also accesses the shared pool's free BO
    * lists. Serialize the whole synchronous operation, including
    * recording and retirement, rather than just allocation and submission. */
   simple_mtx_lock(&dev->host_as.lock);
   run->locked = true;

   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkResult result = disp->CreateFence(device, &fence_info, NULL, &run->fence);
   if (result != VK_SUCCESS) {
      hk_host_as_finish(run);
      return result;
   }

   const VkCommandBufferAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = dev->host_as.pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   result = disp->AllocateCommandBuffers(device, &alloc_info, &run->cmd);
   if (result != VK_SUCCESS) {
      run->cmd = VK_NULL_HANDLE;
      hk_host_as_finish(run);
      return result;
   }

   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   result = disp->BeginCommandBuffer(run->cmd, &begin_info);
   if (result != VK_SUCCESS)
      hk_host_as_finish(run);
   return result;
}

static struct agx_bo *
hk_host_as_bo(struct hk_host_as_run *run, uint64_t size, bool cpu_mapped,
              const char *name)
{
   const enum agx_bo_flags flags =
      (cpu_mapped && util_has_cache_ops()) ? AGX_BO_WRITEBACK : 0;
   struct agx_bo *bo = agx_bo_create(&run->dev->dev, MAX2(size, 16), 0, flags,
                                     name);
   if (bo)
      util_dynarray_append(&run->bos, bo);
   return bo;
}

static VkResult
hk_host_as_execute(struct hk_host_as_run *run)
{
   struct hk_device *dev = run->dev;
   VkDevice device = hk_device_to_handle(dev);
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   VkResult result = disp->EndCommandBuffer(run->cmd);
   if (result != VK_SUCCESS)
      return result;
   const VkCommandBufferSubmitInfo cmd_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = run->cmd,
   };
   const VkSubmitInfo2 submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &cmd_info,
   };
   result = disp->QueueSubmit2(vk_queue_to_handle(&dev->host_as.queue->vk), 1,
                               &submit, run->fence);
   if (result != VK_SUCCESS)
      return result;
   return disp->WaitForFences(device, 1, &run->fence, VK_TRUE, UINT64_MAX);
}

static VkResult
hk_host_as_result(VkDeferredOperationKHR deferred, VkResult result)
{
   if (result != VK_SUCCESS)
      return result;
   return deferred != VK_NULL_HANDLE ? VK_OPERATION_NOT_DEFERRED_KHR : VK_SUCCESS;
}


VKAPI_ATTR VkResult VKAPI_CALL
hk_BuildAccelerationStructuresKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(hk_device, m5_dispatch_dev, _device);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5)
      return hk_m5_backend_BuildAccelerationStructuresKHR(_device, deferredOperation, infoCount, pInfos, ppBuildRangeInfos);

   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_host_as_run run;
   VkResult result = hk_host_as_begin(dev, &run);
   if (result != VK_SUCCESS)
      return result;

   uint32_t geometry_count = 0;
   for (uint32_t i = 0; i < infoCount; ++i)
      geometry_count += pInfos[i].geometryCount;

   VkAccelerationStructureBuildGeometryInfoKHR *infos =
      calloc(MAX2(infoCount, 1), sizeof(*infos));
   VkAccelerationStructureGeometryKHR *geometries =
      calloc(MAX2(geometry_count, 1), sizeof(*geometries));
   VkAccelerationStructureBuildRangeInfoKHR *ranges =
      calloc(MAX2(geometry_count, 1), sizeof(*ranges));
   const VkAccelerationStructureBuildRangeInfoKHR **range_ptrs =
      calloc(MAX2(infoCount, 1), sizeof(*range_ptrs));
   struct hk_host_as_staging *layouts =
      calloc(MAX2(geometry_count, 1), sizeof(*layouts));
   uint32_t *counts = calloc(MAX2(geometry_count, 1), sizeof(*counts));
   if (!infos || !geometries || !ranges || !range_ptrs || !layouts || !counts) {
      result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto out;
   }

   for (uint32_t i = 0, k = 0; i < infoCount; ++i) {
      const VkAccelerationStructureBuildGeometryInfoKHR *info = &pInfos[i];
      infos[i] = *info;
      infos[i].pGeometries = &geometries[k];
      infos[i].ppGeometries = NULL;
      range_ptrs[i] = &ranges[k];
      uint64_t bytes = 0;
      for (uint32_t g = 0; g < info->geometryCount; ++g, ++k) {
         geometries[k] = info->pGeometries ? info->pGeometries[g] : *info->ppGeometries[g];
         ranges[k] = ppBuildRangeInfos[i][g];
         counts[k] = ranges[k].primitiveCount;
         if (!hk_host_as_stage_layout(&geometries[k], &ranges[k], &layouts[k], &bytes)) {
            result = vk_error(dev, VK_ERROR_FEATURE_NOT_PRESENT);
            goto out;
         }
      }
      VkAccelerationStructureBuildSizesInfoKHR sizes = {
         .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
      };
      hk_GetAccelerationStructureBuildSizesKHR(_device,
         VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &infos[i],
         counts + (k - info->geometryCount), &sizes);
      const uint64_t scratch_bytes =
         info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR ?
            sizes.updateScratchSize : sizes.buildScratchSize;
      struct agx_bo *staging = hk_host_as_bo(&run, bytes, true, "Host AS build input");
      struct agx_bo *scratch = hk_host_as_bo(&run, scratch_bytes, false, "Host AS build scratch");
      if (!staging || !scratch) {
         result = vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         goto out;
      }
      uint8_t *map = agx_bo_map(staging);
      if (!map) {
         result = vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         goto out;
      }
      const uint32_t first = k - info->geometryCount;
      hk_host_as_stage_geometries(info->geometryCount, geometries + first,
                                  ranges + first, layouts + first, map,
                                  staging->va->addr, bytes);
      infos[i].scratchData.deviceAddress = scratch->va->addr;
   }

   hk_cmd_build_acceleration_structures(run.cmd, infoCount, infos, range_ptrs,
                                        NULL, NULL);
   result = hk_host_as_execute(&run);

out:
   free(infos);
   free(geometries);
   free(ranges);
   free(range_ptrs);
   free(layouts);
   free(counts);
   hk_host_as_finish(&run);
   return hk_host_as_result(deferredOperation, result);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyAccelerationStructureKHR(VkDevice _device,
                                VkDeferredOperationKHR deferredOperation,
                                const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(hk_device, m5_dispatch_dev, _device);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5)
      return hk_m5_backend_CopyAccelerationStructureKHR(_device, deferredOperation, pInfo);

   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_host_as_run run;
   VkResult result = hk_host_as_begin(dev, &run);
   if (result != VK_SUCCESS)
      return result;
   hk_CmdCopyAccelerationStructureKHR(run.cmd, pInfo);
   result = hk_host_as_execute(&run);
   hk_host_as_finish(&run);
   return hk_host_as_result(deferredOperation, result);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyMemoryToAccelerationStructureKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   VK_FROM_HANDLE(hk_device, m5_dispatch_dev, _device);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5)
      return hk_m5_backend_CopyMemoryToAccelerationStructureKHR(_device, deferredOperation, pInfo);

   VK_FROM_HANDLE(hk_device, dev, _device);
   uint64_t bytes;
   memcpy(&bytes, (const uint8_t *)pInfo->src.hostAddress + 32, sizeof(bytes));
   if (bytes < LIBAGX_GEN1_AS_SERIAL_HEADER_BYTES || bytes > HK_MAX_BUFFER_SIZE)
      return vk_error(dev, VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
   struct hk_host_as_run run;
   VkResult result = hk_host_as_begin(dev, &run);
   if (result != VK_SUCCESS)
      return result;
   struct agx_bo *blob = hk_host_as_bo(&run, bytes, true, "Host AS deserialize input");
   void *blob_map = blob ? agx_bo_map(blob) : NULL;
   if (!blob_map) {
      result = vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   } else {
      memcpy(blob_map, pInfo->src.hostAddress, bytes);
      if (util_has_cache_ops())
         util_flush_range(blob_map, bytes);
      const VkCopyMemoryToAccelerationStructureInfoKHR info = {
         .sType = pInfo->sType,
         .src.deviceAddress = blob->va->addr,
         .dst = pInfo->dst,
         .mode = pInfo->mode,
      };
      hk_CmdCopyMemoryToAccelerationStructureKHR(run.cmd, &info);
      result = hk_host_as_execute(&run);
   }
   hk_host_as_finish(&run);
   return hk_host_as_result(deferredOperation, result);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_CopyAccelerationStructureToMemoryKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   VK_FROM_HANDLE(hk_device, m5_dispatch_dev, _device);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5)
      return hk_m5_backend_CopyAccelerationStructureToMemoryKHR(_device, deferredOperation, pInfo);

   VK_FROM_HANDLE(hk_device, dev, _device);
   VK_FROM_HANDLE(vk_acceleration_structure, src, pInfo->src);
   struct hk_host_as_run run;
   VkResult result = hk_host_as_begin(dev, &run);
   if (result != VK_SUCCESS)
      return result;
   const uint64_t bound = src->size * 2 + 4096;
   struct agx_bo *blob = hk_host_as_bo(&run, bound, true, "Host AS serialize output");
   uint8_t *blob_map = blob ? agx_bo_map(blob) : NULL;
   if (!blob_map) {
      result = vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   } else {
      memset(blob_map, 0, bound);
      if (util_has_cache_ops())
         util_flush_inval_range(blob_map, bound);
      const VkCopyAccelerationStructureToMemoryInfoKHR info = {
         .sType = pInfo->sType,
         .src = pInfo->src,
         .dst.deviceAddress = blob->va->addr,
         .mode = pInfo->mode,
      };
      hk_CmdCopyAccelerationStructureToMemoryKHR(run.cmd, &info);
      result = hk_host_as_execute(&run);
      if (result == VK_SUCCESS) {
         if (util_has_cache_ops())
            util_flush_inval_range(blob_map, bound);
         uint64_t bytes;
         memcpy(&bytes, blob_map + 32, sizeof(bytes));
         if (bytes < LIBAGX_GEN1_AS_SERIAL_HEADER_BYTES || bytes > bound)
            result = vk_error(dev, VK_ERROR_UNKNOWN);
         else
            memcpy(pInfo->dst.hostAddress, blob_map, bytes);
      }
   }
   hk_host_as_finish(&run);
   return hk_host_as_result(deferredOperation, result);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_WriteAccelerationStructuresPropertiesKHR(
   VkDevice _device, uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType, size_t dataSize, void *pData, size_t stride)
{
   VK_FROM_HANDLE(hk_device, m5_dispatch_dev, _device);
   if (hk_agx3_as_format_for_chip(m5_dispatch_dev->dev.chip) == HK_AGX3_AS_FORMAT_M5)
      return hk_m5_backend_WriteAccelerationStructuresPropertiesKHR(_device, accelerationStructureCount, pAccelerationStructures, queryType, dataSize, pData, stride);

   VK_FROM_HANDLE(hk_device, dev, _device);
   const struct vk_device_dispatch_table *disp = &dev->vk.dispatch_table;
   if (!accelerationStructureCount)
      return VK_SUCCESS;
   if (stride < sizeof(uint64_t) || (stride & 7) ||
       dataSize < (accelerationStructureCount - 1) * stride + sizeof(uint64_t))
      return vk_error(dev, VK_ERROR_UNKNOWN);
   struct hk_host_as_run run;
   VkResult result = hk_host_as_begin(dev, &run);
   if (result != VK_SUCCESS)
      return result;
   const VkQueryPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = queryType,
      .queryCount = accelerationStructureCount,
   };
   VkQueryPool pool = VK_NULL_HANDLE;
   uint64_t *results = calloc(accelerationStructureCount, sizeof(*results));
   result = results ? disp->CreateQueryPool(_device, &pool_info, NULL, &pool)
                    : vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   if (result == VK_SUCCESS) {
      disp->CmdResetQueryPool(run.cmd, pool, 0, accelerationStructureCount);
      disp->CmdWriteAccelerationStructuresPropertiesKHR(run.cmd,
         accelerationStructureCount, pAccelerationStructures, queryType, pool, 0);
      result = hk_host_as_execute(&run);
   }
   if (result == VK_SUCCESS)
      result = disp->GetQueryPoolResults(_device, pool, 0, accelerationStructureCount,
         accelerationStructureCount * sizeof(uint64_t), results, sizeof(uint64_t),
         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
   if (result == VK_SUCCESS) {
      for (uint32_t i = 0; i < accelerationStructureCount; ++i)
         memcpy((uint8_t *)pData + i * stride, &results[i], sizeof(uint64_t));
   }
   if (pool)
      disp->DestroyQueryPool(_device, pool, NULL);
   free(results);
   hk_host_as_finish(&run);
   return result;
}

uint64_t
hk_gen1_ray_query_scs_layout(const struct hk_device *dev)
{
   if (dev->gen1_rq_candidates)
      return UINT64_C(0x0130000060010070);
   return dev->gen1_rq_instances ? UINT64_C(0x0058000060010070) :
                                  UINT64_C(0x0028000060000070);
}

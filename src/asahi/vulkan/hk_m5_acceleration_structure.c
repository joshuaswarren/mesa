/*
 * Copyright © 2021 Google
 * Copyright © 2023 Valve Corporation
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * VK_KHR_acceleration_structure for Honeykrisp.
 *
 * The heavy lifting -- Morton codes, the radix sort, PLOC/HPLOC or LBVH
 * construction -- is Mesa's common builder, which produces a driver-neutral
 * binary tree in scratch memory.  All this file adds is the conversion of
 * that tree into the node format the software traversal reads (hk_bvh.h),
 * which happens in one compute dispatch per structure, plus the entry points
 * and the handful of vk_device hooks the common builder calls back into.
 */

#include "hk_acceleration_structure.h"
#include "hk_m5_acceleration_structure.h"
#include "hk_agx3_as_layout.h"
#include "util/u_debug.h"
#include "util/cache_ops.h"
#include "util/u_atomic.h"

#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_physical_device.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"

#include "vk_acceleration_structure.h"
#include "vk_common_entrypoints.h"
#include "vk_meta.h"

#include "bvh/hk_build_interface.h"
#include "hk_agx3_as_format.h"

#include "util/u_math.h"

static const uint32_t agx3_batch_build_leaf_spv[] = {
#include "bvh/hk_agx3_batch_build_leaf.spv.h"
};

static const uint32_t encode_spv[] = {
#include "bvh/hk_encode.spv.h"
};

static const uint32_t write_spv[] = {
#include "bvh/hk_bvh_write.spv.h"
};

static const uint32_t agx3_node_spv[] = {
#include "bvh/hk_agx3_m5_legacy_leaf.spv.h"
};

static const uint32_t agx3_envelope_spv[] = {
#include "bvh/hk_agx3_m5_legacy_envelope.spv.h"
};

static const uint32_t agx3_plan_spv[] = {
#include "bvh/hk_agx3_m5_legacy_plan.spv.h"
};

static const uint32_t agx3_refit_spv[] = {
#include "bvh/hk_agx3_m5_legacy_refit.spv.h"
};

const uint32_t *
hk_m5_agx3_node_encoder_spirv(size_t *size_B)
{
   *size_B = sizeof(agx3_node_spv);
   return agx3_node_spv;
}

const uint32_t *
hk_m5_agx3_refit_spirv(size_t *size_B)
{
   *size_B = sizeof(agx3_refit_spv);
   return agx3_refit_spv;
}

static void hk_flush_buffer_write_cp(VkCommandBuffer);
static void hk_agx3_refit_phase(VkCommandBuffer,
   const struct vk_acceleration_structure_build_state *, unsigned);


/* Native tables have 16-byte entries and reserve ID zero. Keep the same
 * representable ID range as the old command-pool snapshots, without allocating
 * four GiB of physical memory. Each page is independently mapped once and lives
 * until device destruction; neither command recording nor submission copies or
 * patches the table. Updates of unrelated IDs can proceed while work is pending.
 */
#define HK_M5_AS_TABLE_PAGE_SIZE 0x4000u
#define HK_M5_AS_TABLE_SIZE (UINT64_C(1) << 32)
#define HK_M5_AS_TABLE_MAX_ID (UINT32_MAX / 16u - 1u)

/* Called under m5_as_lock. BO mappings are uncached CPU mappings, like the
 * default descriptor-pool path. Fence the published address before returning
 * from creation, including updates to a descriptor in pending work.
 */
static uint64_t *
hk_m5_as_table_entry(struct hk_device *dev, unsigned id)
{
   unsigned page = id / (HK_M5_AS_TABLE_PAGE_SIZE / 16);
   while (util_dynarray_num_elements(&dev->m5_as_table_pages,
                                     struct agx_bo *) <= page) {
      unsigned next = util_dynarray_num_elements(&dev->m5_as_table_pages,
                                                 struct agx_bo *);
      struct agx_bo *bo = agx_bo_create(&dev->dev, HK_M5_AS_TABLE_PAGE_SIZE,
                                      HK_M5_AS_TABLE_PAGE_SIZE, 0,
                                      "Native AS address table");
      if (!bo)
         return NULL;
      memset(agx_bo_map(bo), 0, HK_M5_AS_TABLE_PAGE_SIZE);
      struct agx_bo **slot = util_dynarray_grow(&dev->m5_as_table_pages,
                                               struct agx_bo *, 1);
      if (!slot) {
         agx_bo_unreference(&dev->dev, bo);
         return NULL;
      }
      const uint64_t address = dev->m5_as_table_va->addr +
                               (uint64_t)next * HK_M5_AS_TABLE_PAGE_SIZE;
      if (agx_bo_bind(&dev->dev, bo, address, HK_M5_AS_TABLE_PAGE_SIZE, 0,
                      DRM_ASAHI_BIND_READ)) {
         /* The reserved slot is uninitialized; roll it back without reading it. */
         dev->m5_as_table_pages.size -= sizeof(*slot);
         agx_bo_unreference(&dev->dev, bo);
         return NULL;
      }
      *slot = bo;
   }
   struct agx_bo *bo = util_dynarray_element(&dev->m5_as_table_pages,
                                             struct agx_bo *, page)[0];
   return (uint64_t *)agx_bo_map(bo) +
          2 * (id % (HK_M5_AS_TABLE_PAGE_SIZE / 16));
}

/* Native instance records and the AS header name entries in the same table.
 * Per-lane replacement of those entries aliases child IDs with root IDs. */
static uint32_t
hk_m5_as_id(struct hk_device *dev, struct vk_acceleration_structure *as)
{
   uint32_t result = 0, i = 1;
   simple_mtx_lock(&dev->m5_as_lock);
   util_dynarray_foreach(&dev->m5_ases, struct vk_acceleration_structure *, p) {
      if (*p == as) { result = i; break; }
      ++i;
   }
   simple_mtx_unlock(&dev->m5_as_lock);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_m5_backend_CreateAccelerationStructureKHR(VkDevice device,
   const VkAccelerationStructureCreateInfoKHR *info,
   const VkAllocationCallbacks *alloc, VkAccelerationStructureKHR *out)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VkResult result = vk_common_CreateAccelerationStructureKHR(device, info, alloc, out);
   if (result != VK_SUCCESS || !dev->m5_native_as)
      return result;
   VK_FROM_HANDLE(vk_acceleration_structure, as, *out);
   bool inserted = false;
   simple_mtx_lock(&dev->m5_as_lock);
   unsigned index = 0;
   util_dynarray_foreach(&dev->m5_ases, struct vk_acceleration_structure *, p) {
      if (!*p)
         break;
      ++index;
   }
   if (index < HK_M5_AS_TABLE_MAX_ID) {
      uint64_t *entry = hk_m5_as_table_entry(dev, index + 1);
      if (entry) {
         /* Vulkan permits AS creation before the buffer's memory binding.
          * Ordinary Honeykrisp buffers acquire their address at that binding,
          * so publish it again then rather than freezing an offset-only VA. */
         const bool pending = as->buffer->device_address == 0;
         uint32_t *pending_id = pending
            ? util_dynarray_grow(&dev->m5_pending_as_ids, uint32_t, 1) : NULL;
         if (!pending || pending_id) {
            struct vk_acceleration_structure **slot;
            if (index == util_dynarray_num_elements(&dev->m5_ases,
                                                    struct vk_acceleration_structure *))
               slot = util_dynarray_grow(&dev->m5_ases,
                                         struct vk_acceleration_structure *, 1);
            else
               slot = util_dynarray_element(&dev->m5_ases,
                                             struct vk_acceleration_structure *, index);
            if (slot) {
               *slot = as;
               if (pending) {
                  *pending_id = index + 1;
                  p_atomic_set(&dev->m5_pending_as_count,
                     util_dynarray_num_elements(&dev->m5_pending_as_ids, uint32_t));
               }
               *entry = ALIGN_POT(vk_acceleration_structure_get_va(as),
                                  UINT64_C(1024));
               util_post_flush_fence();
               inserted = true;
            } else if (pending) {
               /* No ID was stored in the reserved slot. */
               dev->m5_pending_as_ids.size -= sizeof(*pending_id);
            }
         }
      }
   }
   simple_mtx_unlock(&dev->m5_as_lock);
   if (!inserted) {
      vk_common_DestroyAccelerationStructureKHR(device, *out, alloc);
      *out = VK_NULL_HANDLE;
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_DestroyAccelerationStructureKHR(VkDevice device, VkAccelerationStructureKHR handle,
                                   const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(vk_acceleration_structure, as, handle);
   if (as && dev->m5_native_as) {
      simple_mtx_lock(&dev->m5_as_lock);
      unsigned id = 1;
      util_dynarray_foreach(&dev->m5_ases, struct vk_acceleration_structure *, p) {
         if (*p == as) {
            *p = NULL;
            util_dynarray_delete_unordered(&dev->m5_pending_as_ids, uint32_t, id);
            p_atomic_set(&dev->m5_pending_as_count,
               util_dynarray_num_elements(&dev->m5_pending_as_ids, uint32_t));
            break;
         }
         ++id;
      }
      simple_mtx_unlock(&dev->m5_as_lock);
   }
   vk_common_DestroyAccelerationStructureKHR(device, handle, alloc);
}

/* Called after a successful buffer binding. A normally pre-bound AS never
 * enters this list; its creation and every subsequent queue submission keep
 * the existing path. Only addresses which were unresolved at AS creation are
 * visited, with no shader or command-buffer mutation. */
void
hk_m5_as_buffer_bound(struct hk_device *dev, struct vk_buffer *buffer)
{
   if (!dev->m5_native_as || !p_atomic_read(&dev->m5_pending_as_count))
      return;

   bool published = false;
   simple_mtx_lock(&dev->m5_as_lock);
   for (unsigned i = 0;
        i < util_dynarray_num_elements(&dev->m5_pending_as_ids, uint32_t);) {
      uint32_t *pending = util_dynarray_element(&dev->m5_pending_as_ids,
                                               uint32_t, i);
      const uint32_t id = *pending;
      struct vk_acceleration_structure *as =
         *util_dynarray_element(&dev->m5_ases,
                                 struct vk_acceleration_structure *, id - 1);
      assert(as);
      if (as->buffer != buffer) {
         ++i;
         continue;
      }
      uint64_t *entry = hk_m5_as_table_entry(dev, id);
      assert(entry); /* The AS was published only after its page was mapped. */
      *entry = ALIGN_POT(vk_acceleration_structure_get_va(as), UINT64_C(1024));
      *pending = util_dynarray_pop(&dev->m5_pending_as_ids, uint32_t);
      published = true;
   }
   if (published)
      util_post_flush_fence();
   p_atomic_set(&dev->m5_pending_as_count,
      util_dynarray_num_elements(&dev->m5_pending_as_ids, uint32_t));
   simple_mtx_unlock(&dev->m5_as_lock);
}

static uint64_t
hk_m5_upload_as_table_count(struct hk_cmd_buffer *cmd, unsigned *table_count)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   simple_mtx_lock(&dev->m5_as_lock);
   unsigned count = util_dynarray_num_elements(&dev->m5_ases,
                                              struct vk_acceleration_structure *);
   if (table_count)
      *table_count = count;
   struct agx_ptr table = hk_pool_alloc(cmd, (count + 1) * 16, 16);
   if (table.cpu) {
      memset(table.cpu, 0, (count + 1) * 16);
      uint64_t *words = table.cpu;
      unsigned i = 1;
      util_dynarray_foreach(&dev->m5_ases, struct vk_acceleration_structure *, p) {
         if (*p)
            words[2*i] = ALIGN_POT(vk_acceleration_structure_get_va(*p), UINT64_C(1024));
         ++i;
      }
   }
   simple_mtx_unlock(&dev->m5_as_lock);
   return table.cpu ? table.gpu : 0;
}

static_assert(sizeof(struct hk_bvh_triangle_node) == 48, "node format");
static_assert(sizeof(struct hk_bvh_aabb_node) == 32, "node format");
static_assert(sizeof(struct hk_bvh_instance_node) == 120, "node format");
static_assert(sizeof(struct hk_bvh_box_node) == 56, "node format");
static_assert(sizeof(struct hk_agx3_leaf_job) == 64, "native node job ABI");
/* hk_build_interface.h is shared with the GEN1 encoders, whose leaf and plan
 * arguments end in fields (counts; pairs) that M5 leaves zero and its
 * encoders never read. */
static_assert(sizeof(struct hk_agx3_leaf_args) == 32, "native node push ABI");
static_assert(sizeof(struct hk_agx3_envelope_job) == 48, "native envelope job ABI");
static_assert(sizeof(struct hk_agx3_envelope_args) == 64, "native envelope push ABI");
static_assert(sizeof(struct hk_agx3_plan_node) == 16, "native planner node ABI");
static_assert(sizeof(struct hk_agx3_plan_state) == 32, "native planner state ABI");
static_assert(sizeof(struct hk_agx3_plan_args) == 120, "native planner push ABI");
/* The M5 layout: the plan state alone after its scratch offset, 48-byte
 * bounding-box records within the 64-byte child slots, 1 KiB images, and one
 * node per 8-lane encoding workgroup (bvh/agx3_leaf.comp). */
static_assert(HK_AGX3_M5_PLAN_TAIL_B == sizeof(struct hk_agx3_plan_state),
              "M5 scratch: no node-encoding dispatch arguments");
struct hk_agx3_m5_aabb_record {
   uint32_t marker; /* leaf type 24 */
   float min[3];
   float max[3];
   uint32_t geometry_id; /* bit 31: opaque */
   uint32_t primitive_id;
   uint32_t primitive_flags; /* bit 30: optional primitive data present */
   uint64_t primitive_data; /* relative pointer */
};
static_assert(offsetof(struct hk_agx3_m5_aabb_record, min) == 4 &&
              offsetof(struct hk_agx3_m5_aabb_record, max) == 16 &&
              offsetof(struct hk_agx3_m5_aabb_record, geometry_id) == 28 &&
              offsetof(struct hk_agx3_m5_aabb_record, primitive_id) == 32 &&
              offsetof(struct hk_agx3_m5_aabb_record, primitive_data) == 40,
              "M5 bounding-box record layout (bvh/agx3_m5_legacy_leaf.comp)");
static_assert(sizeof(struct hk_agx3_m5_aabb_record) == HK_AGX3_M5_AABB_RECORD_B &&
              HK_AGX3_M5_AABB_RECORD_B < HK_AGX3_AS_SLOT_B,
              "M5 bounding-box records pack below the slot size");
static_assert(HK_AGX3_M5_IMAGE_ALIGN == 1024, "M5 image alignment");
static_assert(HK_AGX3_M5_NODE_GROUP_LANES == 8, "M5 encoder workgroup");
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
hk_m5_backend_cmd_encode_agx3_nodes(VkCommandBuffer command_buffer,
                        const struct hk_agx3_leaf_args *args)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, command_buffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!args->job_count)
      return VK_SUCCESS;

   /* The native phase uses eight invocations per independently writable
    * node. Job planning bounds the count before it reaches this helper. */
   assert(args->job_count <= UINT32_MAX / 8);
   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result = get_pipeline_spv(
      dev, "hk-agx3-node-encode", agx3_node_spv, sizeof(agx3_node_spv),
      sizeof(*args), &pipeline, &layout);
   if (result != VK_SUCCESS)
      return result;

   vk_common_CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline);
   vk_common_CmdPushConstants(command_buffer, layout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*args), args);
   hk_cmd_dispatch_unaligned(command_buffer, args->job_count * 8, 1, 1);
   return VK_SUCCESS;
}

VkResult
hk_m5_backend_cmd_finalize_agx3_as(VkCommandBuffer command_buffer,
                       const struct hk_agx3_envelope_args *args)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, command_buffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!args->job_count)
      return VK_SUCCESS;

   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result = get_pipeline_spv(
      dev, "hk-agx3-as-envelope", agx3_envelope_spv, sizeof(agx3_envelope_spv),
      sizeof(*args), &pipeline, &layout);
   if (result != VK_SUCCESS)
      return result;

   vk_common_CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline);
   vk_common_CmdPushConstants(command_buffer, layout,
                             VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*args), args);
   hk_cmd_dispatch_unaligned(command_buffer, args->job_count, 1, 1);
   return VK_SUCCESS;
}

static VkResult
hk_cmd_build_agx3_from_ir_mode(VkCommandBuffer command_buffer,
                        const struct hk_agx3_plan_args *args,
                        VkDeviceAddress node_status,
                        VkDeviceAddress envelope_status,
                        bool single_leaf)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, command_buffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct {
      const char *name;
      const uint32_t *code;
      size_t size;
      unsigned push_size;
   } stages[] = {
      {"hk-agx3-plan", agx3_plan_spv, sizeof(agx3_plan_spv), sizeof(*args)},
      {"hk-agx3-node-encode", agx3_node_spv, sizeof(agx3_node_spv),
       sizeof(struct hk_agx3_leaf_args)},
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
   for (unsigned i = 0; i < (single_leaf ? 1 : 5); ++i) {
      unsigned phase = single_leaf ? 5 : i;
      struct hk_agx3_plan_args push = *args;
      push.phase = phase;
      vk_common_CmdPushConstants(command_buffer, layout[0],
                                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                                &push);
      hk_cmd_dispatch_unaligned(command_buffer, args->node_bound, 1, 1);
      vk_common_CmdPipelineBarrier(
         command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
         0, 1, &barrier, 0, NULL, 0, NULL);
   }

   const struct hk_agx3_leaf_args encode = {
      .jobs = args->jobs, .status = node_status, .job_count = args->node_bound,
   };
   const struct hk_agx3_envelope_args finish = {
      .jobs = args->envelope, .status = envelope_status, .job_count = 1,
   };
   const bool instances = args->geometry_type == VK_GEOMETRY_TYPE_INSTANCES_KHR;
   for (unsigned step = 0; step < 2; ++step) {
      bool envelope = instances ? step == 0 : step == 1;
      unsigned stage = envelope ? 2 : 1;
      vk_common_CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline[stage]);
      vk_common_CmdPushConstants(command_buffer, layout[stage],
                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                envelope ? sizeof(finish) : sizeof(encode),
                                envelope ? (const void *)&finish : (const void *)&encode);
      if (envelope)
         hk_cmd_dispatch_unaligned(command_buffer, 1, 1, 1);
      else
         hk_cmd_dispatch_indirect_addr(command_buffer, args->state);
      if (step == 0)
         vk_common_CmdPipelineBarrier(command_buffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0, 1, &barrier, 0, NULL, 0, NULL);
   }
   return VK_SUCCESS;
}

/* External prebuilt-IR import callers keep the full topology planner. */
VkResult
hk_m5_backend_cmd_build_agx3_from_ir(VkCommandBuffer command_buffer,
                        const struct hk_agx3_plan_args *args,
                        VkDeviceAddress node_status,
                        VkDeviceAddress envelope_status)
{
   return hk_cmd_build_agx3_from_ir_mode(command_buffer, args, node_status,
                                       envelope_status, false);
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

/* Common-builder leaf record consumed by the native planner/encoder. */
static uint32_t
hk_agx3_ir_leaf_size(VkGeometryTypeKHR geometry_type)
{
   switch (geometry_type) {
   case VK_GEOMETRY_TYPE_INSTANCES_KHR:
      return sizeof(struct vk_ir_instance_node);
   case VK_GEOMETRY_TYPE_AABBS_KHR:
      return sizeof(struct vk_ir_aabb_node);
   default:
      return sizeof(struct vk_ir_triangle_node);
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
   if (!hk_agx3_as_layout_init(state->leaf_node_count, &layout))
      return 0;
   return ALIGN_POT(layout.as_bytes +
      (vk_get_as_geometry_type(state->build_info) == VK_GEOMETRY_TYPE_INSTANCES_KHR
          ? (uint64_t)state->leaf_node_count * 80 : 0), 8);
}

static VkDeviceSize
hk_get_as_size(VkDevice _device,
               const struct vk_acceleration_structure_build_state *state)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   if (dev->m5_native_as) {
      struct hk_agx3_as_layout layout;
      if (!hk_agx3_as_layout_init(state->leaf_node_count, &layout))
         return 0;
      if (state->build_info->flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)
         return hk_agx3_retained_offset(state) + sizeof(struct vk_ir_header) +
                (uint64_t)state->internal_node_count * sizeof(struct vk_ir_box_node);
      return layout.as_bytes +
         (vk_get_as_geometry_type(state->build_info) == VK_GEOMETRY_TYPE_INSTANCES_KHR
             ? (uint64_t)state->leaf_node_count * 80 : 0);
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
   return dev->m5_native_as &&
          hk_agx3_as_layout_init(state->leaf_node_count, &layout)
             ? layout.encode_bytes : 0;
}

static uint64_t
hk_as_native_va(struct vk_acceleration_structure *as)
{
   return ALIGN_POT(vk_acceleration_structure_get_va(as), UINT64_C(1024));
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL
hk_m5_backend_GetAccelerationStructureDeviceAddressKHR(
   VkDevice _device, const VkAccelerationStructureDeviceAddressInfoKHR *info)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   VK_FROM_HANDLE(vk_acceleration_structure, as, info->accelerationStructure);
   return dev->m5_native_as ? hk_as_native_va(as)
                            : vk_acceleration_structure_get_va(as);
}

/* ---------------------------------------------------------------- */
/* Encode                                                            */

static VkResult
hk_encode_prepare(VkCommandBuffer commandBuffer,
                  const struct vk_acceleration_structure_build_state *state)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (dev->m5_native_as)
      return VK_SUCCESS;

   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result =
      get_pipeline_spv(dev, "hk-bvh-encode", encode_spv, sizeof(encode_spv),
                       sizeof(struct hk_encode_args), &pipeline, &layout);
   if (result != VK_SUCCESS)
      return result;

   vk_common_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipeline);
   return VK_SUCCESS;
}

static void
hk_encode_as(VkCommandBuffer commandBuffer,
             const struct vk_acceleration_structure_build_state *state)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, dst,
                  state->build_info->dstAccelerationStructure);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (dev->m5_native_as) {
      /* Keep only header/internal topology; updated vertices are read from
       * input buffers, with no Morton sort or tree reconstruction. */
      if ((state->build_info->flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) &&
          (state->build_info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR ||
           state->build_info->srcAccelerationStructure != state->build_info->dstAccelerationStructure)) {
         hk_agx3_refit_phase(commandBuffer, state, 0);
         hk_flush_buffer_write_cp(commandBuffer);
      }
      struct hk_agx3_as_layout layout;
      if (!hk_agx3_as_layout_init(state->leaf_node_count, &layout)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
      const uint64_t scratch = state->build_info->scratchData.deviceAddress;
      const uint64_t encode = scratch + state->scratch.header_offset +
                              sizeof(struct vk_ir_header);
      const struct hk_agx3_plan_args args = {
         .intermediate_bvh = scratch + state->scratch.ir_offset,
         .header = scratch + state->scratch.header_offset,
         .nodes = encode + layout.nodes,
         .jobs = encode + layout.jobs,
         .envelope = encode + layout.envelope,
         .state = encode + layout.state,
         .output_bvh = hk_as_native_va(dst),
         .root_offset = state->leaf_node_count *
            hk_agx3_ir_leaf_size(vk_get_as_geometry_type(state->build_info)),
         .leaf_bound = state->leaf_node_count,
         .node_bound = layout.node_bound,
         .slot_bound = layout.slot_bound,
         .resource_id = hk_m5_as_id(dev, dst),
         .geometry_type = vk_get_as_geometry_type(state->build_info),
      };
      VkResult result = hk_cmd_build_agx3_from_ir_mode(
         commandBuffer, &args, encode + layout.node_status,
         encode + layout.envelope_status,
         false);
      if (result != VK_SUCCESS)
         vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }

   VkGeometryTypeKHR geometry_type =
      vk_get_as_geometry_type(state->build_info);

   uint32_t ir_size, out_size;
   hk_leaf_node_sizes(geometry_type, &ir_size, &out_size);

   VkDeviceAddress scratch = state->build_info->scratchData.deviceAddress;
   uint32_t internal_bound = MAX2(state->leaf_node_count, 2) - 1;

   VkPipeline pipeline;
   VkPipelineLayout layout;
   if (get_pipeline_spv(dev, "hk-bvh-encode", encode_spv, sizeof(encode_spv),
                        sizeof(struct hk_encode_args), &pipeline,
                        &layout) != VK_SUCCESS)
      return;

   const struct hk_encode_args args = {
      .intermediate_bvh = scratch + state->scratch.ir_offset,
      .output_bvh = vk_acceleration_structure_get_va(dst),
      .header = scratch + state->scratch.header_offset,
      .root_offset = state->leaf_node_count * ir_size,
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

static void
hk_agx3_refit_phase(VkCommandBuffer commandBuffer,
                    const struct vk_acceleration_structure_build_state *state,
                    unsigned phase)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_agx3_as_layout layout;
   if (!hk_agx3_as_layout_init(state->leaf_node_count, &layout)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   VkPipeline pipeline;
   VkPipelineLayout pipeline_layout;
   VkResult result = get_pipeline_spv(dev, "hk-agx3-refit", agx3_refit_spv,
      sizeof(agx3_refit_spv), sizeof(struct hk_m5_refit_args), &pipeline, &pipeline_layout);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   VK_FROM_HANDLE(vk_acceleration_structure, retained_as,
      phase == 1 ? state->build_info->srcAccelerationStructure : state->build_info->dstAccelerationStructure);
   uint64_t scratch = state->build_info->scratchData.deviceAddress;
   struct hk_m5_refit_args args = {
      .retained = vk_acceleration_structure_get_va(retained_as) + hk_agx3_retained_offset(state),
      .intermediate_bvh = scratch + state->scratch.ir_offset,
      .header = scratch + state->scratch.header_offset,
      .parents = scratch + state->scratch.size,
      .root_offset = state->leaf_node_count *
         hk_agx3_ir_leaf_size(vk_get_as_geometry_type(state->build_info)),
      .leaf_bound = state->leaf_node_count,
      .node_bound = layout.node_bound, .phase = phase,
   };
   unsigned count = phase <= 1
      ? MAX2(layout.node_bound, (sizeof(struct vk_ir_header) +
               state->internal_node_count * sizeof(struct vk_ir_box_node)) / 4)
      : phase == 2 ? state->internal_node_count : state->leaf_node_count;
   vk_common_CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   vk_common_CmdPushConstants(commandBuffer, pipeline_layout,
      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(args), &args);
   hk_cmd_dispatch_unaligned(commandBuffer, count, 1, 1);
}

static void
hk_native_build_config(VkDevice device,
                       struct vk_acceleration_structure_build_state *state)
{
   if (state->build_info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR)
      state->config.internal_type = VK_INTERNAL_BUILD_TYPE_UPDATE;
   state->config.updateable =
      (state->build_info->flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) != 0;
   /* The upstream runtime always builds the internal topology; the former
    * single-leaf shortcut needed a runtime hook this base does not have. */
}

static VkDeviceSize
hk_update_scratch_size(VkDevice device,
                       const struct vk_acceleration_structure_build_state *state)
{
   /* Leaf generation writes sort IDs, which alias native encode scratch.
    * Keep live refit parent/counter state beyond both workspaces. */
   struct hk_agx3_as_layout layout;
   if (!hk_agx3_as_layout_init(state->leaf_node_count, &layout))
      return 0;
   return (VkDeviceSize)state->scratch.size + (uint64_t)layout.node_bound * 16;
}

static void
hk_init_update_scratch(VkCommandBuffer commandBuffer,
                       const struct vk_acceleration_structure_build_state *states,
                       uint32_t count)
{
   for (unsigned i = 0; i < count; ++i)
      if (states[i].config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE)
         hk_agx3_refit_phase(commandBuffer, &states[i], 1);
}

static void
hk_update_prepare(VkCommandBuffer commandBuffer,
                  const struct vk_acceleration_structure_build_state *state,
                  bool flushed_cp, bool flushed_compute)
{
   hk_flush_buffer_write_cp(commandBuffer);
}

static void
hk_update_as(VkCommandBuffer commandBuffer,
             const struct vk_acceleration_structure_build_state *state)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   hk_agx3_refit_phase(commandBuffer, state, 2);
   VkResult result = vk_cmd_build_as_leaves(commandBuffer, &dev->vk, &dev->meta,
      &dev->accel_struct_args, state);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   hk_flush_buffer_write_cp(commandBuffer);
   hk_agx3_refit_phase(commandBuffer, state, 3);
   hk_flush_buffer_write_cp(commandBuffer);
   hk_encode_as(commandBuffer, state);
}

/* Same per-AS native records, planned and encoded concurrently. */
/* The upstream runtime hands encoding every state of a batch at once, after
 * the common builder finished the intermediate trees of the non-update
 * states. Keep the M5 per-state prepare/encode/update sequence inside it. */
static void
hk_encode_dispatch(VkCommandBuffer commandBuffer, struct vk_device *device,
                   struct vk_meta_device *meta,
                   const struct vk_acceleration_structure_build_args *build_args,
                   struct vk_acceleration_structure_build_state *states,
                   uint32_t build_count, bool flushed_compute_after_init_update_scratch)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   bool prepared = false;

   /* The common builder hands encoding the completed intermediate tree. */
   vk_bvh_build_barrier_compute_to_compute(commandBuffer, false);
   for (uint32_t i = 0; i < build_count; i++) {
      if (states[i].config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE) {
         hk_update_prepare(commandBuffer, &states[i], true,
                           flushed_compute_after_init_update_scratch);
         hk_update_as(commandBuffer, &states[i]);
         continue;
      }
      if (!prepared) {
         VkResult result = hk_encode_prepare(commandBuffer, &states[i]);
         if (result != VK_SUCCESS) {
            vk_command_buffer_set_error(&cmd->vk, result);
            return;
         }
         prepared = true;
      }
      hk_encode_as(commandBuffer, &states[i]);
   }
}

struct hk_leaf_task {
   struct leaf_args leaf;
   uint32_t first;
   uint32_t count;
};

static VkResult
hk_build_leaves_batched(VkCommandBuffer commandBuffer, struct vk_device *device,
                        struct vk_meta_device *meta,
                        const struct vk_acceleration_structure_build_args *args,
                        struct vk_acceleration_structure_build_state *states,
                        uint32_t build_count, uint32_t build_flags)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   static_assert(sizeof(struct hk_leaf_task) == 88, "leaf task shader ABI");
   static_assert(offsetof(struct hk_leaf_task, first) == 80, "leaf task first ABI");
   assert(args->subgroup_size == 32);
   uint64_t task_count = 0;
   for (unsigned i = 0; i < build_count; ++i) {
      if (states[i].config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE ||
          (states[i].config.build_flags & VK_LEAF_BUILD_FLAGS) != build_flags)
         continue;
      for (unsigned j = 0; j < states[i].build_info->geometryCount; ++j) {
         uint32_t count = states[i].build_range_infos[j].primitiveCount;
         task_count += DIV_ROUND_UP((uint64_t)count, 32);
      }
   }
   if (!task_count)
      return VK_SUCCESS;
   if (task_count > SIZE_MAX / sizeof(struct hk_leaf_task))
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   if (task_count > UINT32_MAX / sizeof(struct hk_leaf_task))
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   struct hk_leaf_task *tasks = malloc(task_count * sizeof(*tasks));
   if (!tasks)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   VkPipeline pipeline;
   VkPipelineLayout layout;
   /* This callback replaces every leaf launch on this device, so the common
    * leaf key remains unique, including all original build-flag variants. */
   VkResult result = vk_get_bvh_build_pipeline_spv(device, meta,
      VK_META_OBJECT_KEY_LEAF, agx3_batch_build_leaf_spv,
      sizeof(agx3_batch_build_leaf_spv), sizeof(uint64_t), args, build_flags,
      &pipeline, false);
   if (result != VK_SUCCESS) {
      free(tasks);
      return result;
   }
   result = vk_get_bvh_build_pipeline_layout(device, meta, sizeof(uint64_t), &layout);
   if (result != VK_SUCCESS) {
      free(tasks);
      return result;
   }

   uint64_t next = 0;
   for (unsigned i = 0; i < build_count; ++i) {
      if (states[i].config.internal_type == VK_INTERNAL_BUILD_TYPE_UPDATE ||
          (states[i].config.build_flags & VK_LEAF_BUILD_FLAGS) != build_flags)
         continue;
      const VkAccelerationStructureBuildGeometryInfoKHR *info = states[i].build_info;
      uint64_t scratch = info->scratchData.deviceAddress;
      struct leaf_args leaf = {
         .bvh = scratch + states[i].scratch.ir_offset,
         .header = scratch + states[i].scratch.header_offset,
         .ids = scratch + states[i].scratch.sort_buffer_offset[0],
      };
      for (unsigned j = 0; j < info->geometryCount; ++j) {
         const VkAccelerationStructureBuildRangeInfoKHR *range = &states[i].build_range_infos[j];
         if (!range->primitiveCount)
            continue;
         const VkAccelerationStructureGeometryKHR *geom = info->pGeometries
            ? &info->pGeometries[j] : info->ppGeometries[j];
         leaf.geom_data = vk_fill_geometry_data(info->type, states[i].leaf_node_count,
                                                j, geom, range);
         for (uint64_t first = 0; first < range->primitiveCount; first += 32)
            tasks[next++] = (struct hk_leaf_task) {
               .leaf = leaf, .first = first,
               .count = MIN2(32, range->primitiveCount - first),
            };
         states[i].leaf_node_count += range->primitiveCount;
      }
   }
   assert(next == task_count);
   uint64_t gpu = hk_pool_upload(cmd, tasks, task_count * sizeof(*tasks), 8);
   free(tasks);
   if (!gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   device->dispatch_table.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   uint32_t limit = MIN2(device->physical->properties.maxComputeWorkGroupCount[0],
                         UINT32_MAX / 32);
   for (uint64_t first = 0; first < task_count;) {
      uint32_t count = MIN2(task_count - first, limit);
      uint64_t address = gpu + first * sizeof(struct hk_leaf_task);
      device->dispatch_table.CmdPushConstants(commandBuffer, layout,
         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(address), &address);
      device->cmd_dispatch_unaligned(commandBuffer, count * 32, 1, 1);
      first += count;
   }
   return VK_SUCCESS;
}

static const struct vk_acceleration_structure_build_ops hk_native_batched_leaf_build_ops = {
   .get_build_config = hk_native_build_config,
   .get_as_size = hk_get_as_size,
   .get_encode_scratch_size = hk_get_encode_scratch_size,
   .get_update_scratch_size = hk_update_scratch_size,
   .init_update_scratch = hk_init_update_scratch,
   .encode = hk_encode_dispatch,
   .build_leaves = hk_build_leaves_batched,
};

static const struct vk_acceleration_structure_build_ops hk_as_build_ops = {
   .get_as_size = hk_get_as_size,
   .get_encode_scratch_size = hk_get_encode_scratch_size,
   .encode = hk_encode_dispatch,
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

   /* IR headers and the 112-byte driver empty-AS header fit this payload.
    * Refuse larger writes rather than truncating initialized metadata. */
   assert(size <= sizeof(args.data));
   if (size > sizeof(args.data))
      return;
   memcpy(args.data, data, size);

   VkPipeline pipeline;
   VkPipelineLayout layout;
   VkResult result = get_pipeline_spv(dev, "hk-bvh-write", write_spv,
      sizeof(write_spv), sizeof(args), &pipeline, &layout);
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
   /* The common builder's initialization barrier has TOP_OF_PIPE as its
    * source: that orders a CP write but cannot order our compute replacement.
    * Publish the initialized header before build_leaves reads and atomically
    * updates its bounds/counters. Keep the dispatches in the same CDM stream. */
   const VkMemoryBarrier barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
   };
   vk_common_CmdPipelineBarrier(commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 1, &barrier, 0, NULL, 0, NULL);
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

VkResult
hk_m5_backend_device_init_accel_struct_state(struct hk_device *dev)
{
   dev->m5_native_as = agx_chip_g17_m5(dev->dev.chip);
   dev->accel_struct_args = (struct vk_acceleration_structure_build_args){
      .subgroup_size = 32,
      .bvh_bounds_offset = dev->m5_native_as ? 0x50 :
         offsetof(struct hk_bvh_header, bounds),
      /* The common builder's spec_map gives BUILD_FLAGS_ID and
       * ROOT_FLAGS_OFFSET_ID the same push-constant offset, so root_flags
       * would receive the build flags.  It is only read when cull-flag
       * propagation is on, so leave that off until that is fixed upstream. */
      .root_flags_offset = 0,
      .morton_sort_workgroup_size = 512,
      .morton_sort_kvs_per_thread = 2,
      .propagate_cull_flags = false,
   };

   dev->vk.as_build_ops = dev->m5_native_as ? &hk_native_batched_leaf_build_ops
                                            : &hk_as_build_ops;
   dev->vk.write_buffer_cp = hk_write_buffer_cp;
   dev->vk.flush_buffer_write_cp = hk_flush_buffer_write_cp;
   dev->vk.cmd_dispatch_unaligned = hk_cmd_dispatch_unaligned;
   dev->vk.cmd_fill_buffer_addr = hk_cmd_fill_buffer_addr;

   simple_mtx_init(&dev->m5_as_lock, mtx_plain);
   util_dynarray_init(&dev->m5_ases, NULL);
   util_dynarray_init(&dev->m5_as_table_pages, NULL);
   util_dynarray_init(&dev->m5_pending_as_ids, NULL);
   if (dev->m5_native_as) {
      dev->m5_as_table_va = agx_va_alloc(&dev->dev, HK_M5_AS_TABLE_SIZE,
                                       HK_M5_AS_TABLE_PAGE_SIZE, 0, 0);
      if (!dev->m5_as_table_va || !hk_m5_as_table_entry(dev, 0)) {
         hk_device_finish_accel_struct_state(dev);
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      }
   }
   return VK_SUCCESS;
}

void
hk_device_finish_accel_struct_state(struct hk_device *dev)
{
   if (!agx_chip_g17_m5(dev->dev.chip))
      return;
   bool table_unmapped = true;
   unsigned page = 0;
   util_dynarray_foreach(&dev->m5_as_table_pages, struct agx_bo *, bo) {
      if (agx_bo_bind(&dev->dev, NULL, dev->m5_as_table_va->addr +
                      (uint64_t)page * HK_M5_AS_TABLE_PAGE_SIZE,
                      HK_M5_AS_TABLE_PAGE_SIZE, 0, DRM_ASAHI_BIND_UNBIND))
         table_unmapped = false;
      agx_bo_unreference(&dev->dev, *bo);
      ++page;
   }
   /* Never recycle the reservation if an alias failed to unbind. */
   if (table_unmapped)
      agx_va_free(&dev->dev, dev->m5_as_table_va, false);
   util_dynarray_fini(&dev->m5_pending_as_ids);
   util_dynarray_fini(&dev->m5_as_table_pages);
   util_dynarray_fini(&dev->m5_ases);
   simple_mtx_destroy(&dev->m5_as_lock);
}

/* ---------------------------------------------------------------- */
/* Entry points                                                      */

VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_GetAccelerationStructureBuildSizesKHR(
   VkDevice _device, VkAccelerationStructureBuildTypeKHR buildType,
   const VkAccelerationStructureBuildGeometryInfoKHR *pBuildInfo,
   const uint32_t *pMaxPrimitiveCounts,
   VkAccelerationStructureBuildSizesInfoKHR *pSizeInfo)
{
   VK_FROM_HANDLE(hk_device, dev, _device);

   if (dev->m5_native_as) {
      uint64_t leaves = 0;
      for (uint32_t i = 0; i < pBuildInfo->geometryCount; ++i)
         leaves += pMaxPrimitiveCounts[i];
      if (!leaves) {
         struct hk_agx3_as_layout layout;
         hk_agx3_as_layout_init(0, &layout);
         pSizeInfo->accelerationStructureSize = layout.as_bytes;
         pSizeInfo->buildScratchSize = 0;
         pSizeInfo->updateScratchSize = 0;
         return;
      }
   }

   vk_get_as_build_sizes(_device, buildType, pBuildInfo, pMaxPrimitiveCounts,
                         pSizeInfo, &dev->accel_struct_args);
}

VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_CmdBuildAccelerationStructuresKHR(
   VkCommandBuffer commandBuffer, uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   uint32_t empty_count = 0;
   if (dev->m5_native_as) {
      for (uint32_t i = 0; i < infoCount; ++i) {
         const VkAccelerationStructureGeometryKHR *geometry =
            pInfos[i].geometryCount
               ? (pInfos[i].pGeometries ? pInfos[i].pGeometries :
                                        pInfos[i].ppGeometries[0]) : NULL;
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
         struct hk_agx3_as_layout layout;
         /* Bottom-level structures hold triangles or procedural AABBs; the
          * native encoder emits the hardware record for either. */
         bool triangles =
            pInfos[i].type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR &&
            (!geometry || geometry->geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR ||
             geometry->geometryType == VK_GEOMETRY_TYPE_AABBS_KHR);
         bool instances = geometry && pInfos[i].geometryCount == 1 &&
            pInfos[i].type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR &&
            geometry->geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR;
         if ((pInfos[i].mode != VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR &&
              pInfos[i].mode != VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) ||
             !compatible || (!triangles && !instances) || primitives > UINT32_MAX ||
             !hk_agx3_as_layout_init(primitives, &layout)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         empty_count += primitives == 0;
      }
   }

   const uint32_t nonempty_count = infoCount - empty_count;
   VkAccelerationStructureBuildGeometryInfoKHR *nonempty_infos = NULL;
   const VkAccelerationStructureBuildRangeInfoKHR **nonempty_ranges = NULL;
   if (empty_count && nonempty_count) {
      nonempty_infos = malloc(nonempty_count * sizeof(*nonempty_infos));
      nonempty_ranges = malloc(nonempty_count * sizeof(*nonempty_ranges));
      if (!nonempty_infos || !nonempty_ranges) {
         free(nonempty_infos);
         free(nonempty_ranges);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return;
      }
   }
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
            uint32_t header[28];
            hk_agx3_as_empty_header(header);
            hk_write_buffer_cp(commandBuffer, hk_as_native_va(dst) + 64,
                               header, sizeof(header));
         } else if (nonempty_infos) {
            nonempty_infos[n] = pInfos[i];
            nonempty_ranges[n++] = ppBuildRangeInfos[i];
         }
      }
      hk_flush_buffer_write_cp(commandBuffer);
   }
   /* Keep all remaining structures in one common build batch. */
   if (nonempty_count)
      vk_cmd_build_acceleration_structures(commandBuffer, &dev->vk, &dev->meta,
         nonempty_count, nonempty_infos ? nonempty_infos : pInfos,
         nonempty_ranges ? nonempty_ranges : ppBuildRangeInfos,
         &dev->accel_struct_args);

   hk_meta_end(cmd, &save, VK_PIPELINE_BIND_POINT_COMPUTE);
   free(nonempty_infos);
   free(nonempty_ranges);
}

VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_GetDeviceAccelerationStructureCompatibilityKHR(
   VkDevice _device, const VkAccelerationStructureVersionInfoKHR *pVersionInfo,
   VkAccelerationStructureCompatibilityKHR *pCompatibility)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   struct hk_physical_device *pdev = hk_device_physical(dev);

   bool compat = memcmp(pVersionInfo->pVersionData,
                        pdev->vk.properties.driverUUID, VK_UUID_SIZE) == 0;
   *pCompatibility = compat
                        ? VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR
                        : VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

/* The native image is smaller than its Vulkan allocation: the latter also
 * reserves the prefix needed to align the root to 1KiB. Launch for the
 * allocation's upper bound and let the GPU header bound the actual copy.
 * This keeps build -> copy -> trace entirely on the GPU, including builds
 * recorded earlier in this same command buffer. */
static uint32_t
hk_as_copy_threads(struct hk_cmd_buffer *cmd,
                   struct vk_acceleration_structure *as)
{
   const uint64_t padding = hk_as_native_va(as) -
                            vk_acceleration_structure_get_va(as);
   if (padding > as->size || as->size - padding < 192 ||
       DIV_ROUND_UP(as->size - padding, 16) > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return 0;
   }
   return DIV_ROUND_UP(as->size - padding, 16);
}

VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_CmdCopyAccelerationStructureKHR(
   VkCommandBuffer commandBuffer, const VkCopyAccelerationStructureInfoKHR *info)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, src, info->src);
   VK_FROM_HANDLE(vk_acceleration_structure, dst, info->dst);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!dev->m5_native_as ||
       (info->mode != VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR &&
        info->mode != VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   uint32_t threads = hk_as_copy_threads(cmd, src);
   if (!threads)
      return;
   libagx_agx3_as_copy(cmd, agx_1d(threads), AGX_BARRIER_ALL,
                       hk_as_native_va(dst), hk_as_native_va(src),
                       hk_m5_as_id(dev, dst));
}

VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_CmdCopyAccelerationStructureToMemoryKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyAccelerationStructureToMemoryInfoKHR *info)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, src, info->src);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!dev->m5_native_as ||
       info->mode != VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   uint32_t threads = hk_as_copy_threads(cmd, src);
   if (!threads)
      return;
   uint32_t uuid[4];
   memcpy(uuid, hk_device_physical(dev)->vk.properties.driverUUID, sizeof(uuid));
   libagx_agx3_as_serialize(cmd, agx_1d(threads), AGX_BARRIER_ALL,
                            info->dst.deviceAddress, hk_as_native_va(src),
                            uuid[0], uuid[1], uuid[2], uuid[3]);
}

VKAPI_ATTR void VKAPI_CALL
hk_m5_backend_CmdCopyMemoryToAccelerationStructureKHR(
   VkCommandBuffer commandBuffer,
   const VkCopyMemoryToAccelerationStructureInfoKHR *info)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_acceleration_structure, dst, info->dst);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!dev->m5_native_as ||
       info->mode != VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   uint32_t threads = hk_as_copy_threads(cmd, dst);
   if (!threads)
      return;
   unsigned count;
   uint64_t table = hk_m5_upload_as_table_count(cmd, &count);
   if (!table)
      return;
   libagx_agx3_as_deserialize(cmd, agx_1d(threads), AGX_BARRIER_ALL,
                              hk_as_native_va(dst), info->src.deviceAddress,
                              hk_m5_as_id(dev, dst));
   /* IDs are assigned at creation. Vulkan permits the replacement BLAS to be
    * unbuilt here, so resolve its address through the uploaded object table,
    * rather than reading a header which a later build will initialize. */
   libagx_agx3_as_relink(cmd, agx_1d(threads), AGX_BARRIER_ALL,
                         hk_as_native_va(dst), info->src.deviceAddress,
                         table, count);
}

/*
 * Host-side builds and copies are unimplemented and unadvertised:
 * accelerationStructureHostCommands
 * stays false, so a conformant application never reaches them.  They fail
 * loudly rather than silently returning success on an untouched structure.
 */
VKAPI_ATTR VkResult VKAPI_CALL
hk_m5_backend_BuildAccelerationStructuresKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   uint32_t infoCount,
   const VkAccelerationStructureBuildGeometryInfoKHR *pInfos,
   const VkAccelerationStructureBuildRangeInfoKHR *const *ppBuildRangeInfos)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_m5_backend_CopyAccelerationStructureKHR(VkDevice _device,
                                VkDeferredOperationKHR deferredOperation,
                                const VkCopyAccelerationStructureInfoKHR *pInfo)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_m5_backend_CopyMemoryToAccelerationStructureKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   const VkCopyMemoryToAccelerationStructureInfoKHR *pInfo)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_m5_backend_CopyAccelerationStructureToMemoryKHR(
   VkDevice _device, VkDeferredOperationKHR deferredOperation,
   const VkCopyAccelerationStructureToMemoryInfoKHR *pInfo)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_m5_backend_WriteAccelerationStructuresPropertiesKHR(
   VkDevice _device, uint32_t accelerationStructureCount,
   const VkAccelerationStructureKHR *pAccelerationStructures,
   VkQueryType queryType, size_t dataSize, void *pData, size_t stride)
{
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

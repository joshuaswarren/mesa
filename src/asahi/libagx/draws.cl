/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "asahi/lib/agx_abi.h"
#include "compiler/libcl/libcl_vk.h"
#include "poly/geometry.h"
#include "agx_pack.h"
#include "libagx_dgc.h"

/*
 * To implement drawIndirectCount generically, we dispatch a kernel to
 * clone-and-patch the indirect buffer, predicating out draws as appropriate.
 */
KERNEL(32)
libagx_predicate_indirect(global uint32_t *out, constant uint32_t *in,
                          constant uint32_t *draw_count, uint32_t stride_el,
                          uint indexed__2)
{
   uint draw = cl_global_id.x;
   uint words = indexed__2 ? 5 : 4;
   bool enabled = draw < *draw_count;
   out += draw * words;
   in += draw * stride_el;

   /* Copy enabled draws, zero predicated draws. */
   for (uint i = 0; i < words; ++i) {
      out[i] = enabled ? in[i] : 0;
   }
}

/*
 * Build a GPU-authored AGX3 VDM sub-stream (and per-draw Ioto vertex programs
 * when the vertex shader reads base parameters or draw ID) for multi-draw
 * indirect and indirect-count draws. Four consecutive lanes cooperate on each
 * draw record using 128-bit coalesced transfers with in-register patching; draw
 * `count` writes the returning VDM Stream Link so culled slots above `count`
 * are never parsed by the VDM.
 */
KERNEL(128)
libagx_build_indirect_vdm(constant struct libagx_indirect_vdm_desc *desc)
{
   libagx_build_indirect_vdm_sublane(desc, cl_global_id.x);
}

/*
 * Fast path for drawIndirectCount: instead of copying and predicating every
 * indirect command and letting VDM execute all maxDrawCount records, patch a
 * non-returning VDM_STREAM_LINK over draw `*draw_count`'s first 8 bytes so VDM
 * jumps straight past the remaining records.
 */
KERNEL(1)
libagx_patch_vdm_indirect_count(
   global struct libagx_vdm_truncate_state *state,
   constant struct libagx_vdm_draw_span *spans, constant uint32_t *draw_count,
   uint32_t num_spans, uint32_t max_draw_count, uint32_t link_w0,
   uint32_t link_w1)
{
   libagx_vdm_patch_indirect_count_impl(state, spans, draw_count, num_spans,
                                        max_draw_count, link_w0, link_w1);
}

/* VK_EXT_transform_feedback exposes the byte count written by streamout as a
 * draw source.  Generate the complete VkDrawIndirectCommand on the GPU so a
 * producer earlier in this command buffer never needs CPU readback. */
KERNEL(1)
libagx_draw_indirect_byte_count(global VkDrawIndirectCommand *out,
                                constant uint32_t *byte_count,
                                uint32_t counter_offset,
                                uint32_t vertex_stride,
                                uint32_t instance_count,
                                uint32_t first_instance)
{
   *out = (VkDrawIndirectCommand){
      .vertexCount =
         agx_xfb_vertex_count(*byte_count, counter_offset, vertex_stride),
      .instanceCount = instance_count,
      .firstVertex = 0,
      .firstInstance = first_instance,
   };
}

/*
 * VK_EXT_conditional_rendering predicates both graphics and compute commands
 * with one 32-bit value.  Honeykrisp lowers each affected command to an
 * indirect argument record and runs this kernel immediately before the
 * consumer.  This is deliberately separate from predicate_indirect: Vulkan
 * requires a 32-bit load (D3D12 uses 64-bit predication), permits inversion,
 * and compute dispatch records contain three rather than four/five words.
 */
/*
 * M5 task (object) draws whose grid, draw count or predicate live in GPU
 * memory. The VDM mesh dispatch record of such a draw is recorded with one
 * object workgroup; patch its object thread counts before the render runs and
 * publish whether the draw executes, which the task shader checks on entry
 * (a disabled draw keeps one workgroup, which launches no mesh children).
 */
KERNEL(1)
libagx_m5_task_dispatch(global uint32_t *record, global uint32_t *flag,
                        constant uint32_t *grid, constant uint32_t *draw_count,
                        constant uint32_t *predicate, uint32_t draw_index,
                        uint32_t inverted, uint32_t local_x, uint32_t local_y,
                        uint32_t local_z)
{
   bool enabled = grid[0] != 0 && grid[1] != 0 && grid[2] != 0;
   if (draw_count)
      enabled = enabled && draw_index < *draw_count;
   if (predicate)
      enabled = enabled && ((*predicate != 0) != (inverted != 0));

   record[0] = enabled ? grid[0] * local_x : local_x;
   record[1] = enabled ? grid[1] * local_y : local_y;
   record[2] = enabled ? grid[2] * local_z : local_z;
   *flag = enabled ? 1 : 0;
}

/*
 * Software task stage (G17P): size a task draw whose grid, draw count or
 * predicate lives in GPU memory. A disabled draw, or one whose entries do not
 * fit the heap, launches no task workgroups and so no mesh children. With a
 * heap the entries are allocated here; without one the recording allocated
 * them for its known grid.
 */
KERNEL(1)
libagx_task_setup(global struct libagx_task_record *rec,
                  constant uint32_t *grid, constant uint32_t *draw_count,
                  constant uint32_t *predicate, global struct poly_heap *heap,
                  uint32_t draw_index, uint32_t inverted)
{
   uint32_t n = grid[0] * grid[1] * grid[2];
   bool enabled = n != 0;
   if (draw_count)
      enabled = enabled && draw_index < *draw_count;
   if (predicate)
      enabled = enabled && ((*predicate != 0) != (inverted != 0));

   if (enabled && heap) {
      uint64_t size = (uint64_t)n * rec->stride_B;
      uint offs = 0;
      enabled = size <= heap->size &&
                poly_heap_try_alloc_offs(heap, (uint)size, &offs);
      if (enabled)
         rec->entries = (uint64_t)heap->base + offs;
   }

   rec->task_count = enabled ? n : 0;
   for (uint i = 0; i < 3; ++i)
      rec->dispatch[i] = enabled ? grid[i] : 0;
}

KERNEL(1)
libagx_conditional_render(global uint32_t *out, constant uint32_t *in,
                          constant uint32_t *predicate, uint32_t words,
                          uint32_t inverted)
{
   bool execute = ((*predicate != 0) != (inverted != 0));

   for (uint i = 0; i < words; ++i)
      out[i] = execute ? in[i] : 0;
}

/*
 * Materialise an adjacency decomposition as a real index buffer.
 *
 * The software-indexing path strips a draw's index/offset information and has
 * the VS prolog remap gl_VertexIndex instead.  That never works on G17P: a
 * hardware vertex shader reads its invocation index from SR 0xdd directly
 * (agx_compile.c, nir_intrinsic_load_vertex_id), so it never observes the
 * AGX_ABI_VIN_VERTEX_ID the prolog exported, and every adjacency draw fetched
 * attributes at in[i] instead of in[map(i)].
 *
 * Writing the mapped -- and, for an indexed draw, already-resolved -- indices
 * into a buffer lets the hardware input assembler perform the remap, which is
 * the one path the vertex stage does observe.  first_vertex is folded in here
 * so the resulting draw needs no index bias of its own.
 */
KERNEL(64)
libagx_write_adj_indices(global uint *out, uint64_t index_buffer,
                         uint index_buffer_range_el, uint count,
                         uint index_size_B, uint first_vertex,
                         enum mesa_prim prim)
{
   uint i = cl_global_id.x;
   if (i >= count)
      return;

   uint src = poly_map_adj_vertex(i, prim);

   out[i] = first_vertex + (index_size_B
                               ? poly_load_index(index_buffer,
                                                 index_buffer_range_el, src,
                                                 index_size_B)
                               : src);
}

/*
 * The indirect form of libagx_write_adj_indices.  The output index count is
 * not known until the GPU has read the indirect command, so the index buffer
 * is allocated from the geometry heap and named by an indexed indirect draw,
 * exactly as the primitive-restart unroll does.
 */
KERNEL(1024)
libagx_write_adj_indices_indirect(global struct poly_heap *heap,
                                  constant uint *in_draw,
                                  global uint *out_draw, uint64_t index_buffer,
                                  uint index_buffer_range_el,
                                  uint index_size_B, enum mesa_prim prim)
{
   uint tid = cl_local_id.x;

   /* VkDrawIndirectCommand and VkDrawIndexedIndirectCommand agree on their
    * first three words; only the tail differs. */
   uint count = libagx_remap_adj_count(in_draw[0], prim);
   uint instances = in_draw[1];
   uint first = in_draw[2];
   uint vertex_offset = index_size_B ? in_draw[3] : 0;
   uint first_instance = index_size_B ? in_draw[4] : in_draw[3];

   local uint offs_sh;
   local uint ok_sh;

   if (tid == 0) {
      uint offs = 0;
      uint ok = count > 0 && poly_heap_try_alloc_offs(heap, count * 4, &offs);

      offs_sh = offs;
      ok_sh = ok;

      /* An exhausted heap draws nothing rather than writing through an
       * invalid address, matching the restart unroll's behaviour. */
      out_draw[0] = ok ? count : 0;
      out_draw[1] = instances;
      out_draw[2] = ok ? (offs / 4) : 0;
      out_draw[3] = 0; /* vertexOffset, folded into the indices below */
      out_draw[4] = first_instance;
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   if (!ok_sh)
      return;

   global uint *out = (global uint *)(heap->base + offs_sh);

   for (uint i = tid; i < count; i += 1024) {
      uint src = poly_map_adj_vertex(i, prim);

      out[i] =
         index_size_B
            ? vertex_offset + poly_load_index(
                                 index_buffer + (uint64_t)first * index_size_B,
                                 index_buffer_range_el, src, index_size_B)
            : first + src;
   }
}

/*
 * Indexing/offseting is in software if necessary so we strip all
 * indexing/offset information.
 */
KERNEL(1)
libagx_draw_without_adj(global VkDrawIndirectCommand *out,
                        global VkDrawIndirectCommand *in,
                        global struct poly_vertex_params *vp,
                        uint64_t index_buffer, uint64_t index_buffer_range_el,
                        int index_size_B, enum mesa_prim prim)
{
   *out = (VkDrawIndirectCommand){
      .vertexCount = libagx_remap_adj_count(in->vertexCount, prim),
      .instanceCount = in->instanceCount,
   };

   /* TODO: Deduplicate */
   if (index_size_B) {
      uint offs = in->firstVertex;

      vp->index_buffer = poly_index_buffer(index_buffer, index_buffer_range_el,
                                           offs, index_size_B);

      vp->index_buffer_range_el =
         poly_index_buffer_range_el(index_buffer_range_el, offs);
   }
}

/*
 * AGX does not implement robustBufferAccess2 semantics for
 * index buffers, where out-of-bounds indices read as zero. When we
 * dynamically detect index buffer overread (this if-statement), we need
 * to clone the index buffer and zero-extend it to get robustness.
 *
 * We do this dynamically (generating a VDM draw to consume the result) to avoid
 * expensive allocations & memcpys in the happy path where no out-of-bounds
 * access occurs. Otherwise we could use a hardware indirect draw, rather than
 * generating VDM words directly in shader.
 *
 * A workgroup repairs one draw; the batch entry point assigns independent
 * records to workgroups, retaining a single publication barrier at the end.
 */
static void
libagx_draw_robust_index_impl(
   global uint32_t *vdm, global struct poly_heap *heap,
   global uint32_t *printf_buffer,
   constant VkDrawIndexedIndirectCommand *cmd, uint64_t in_buf_ptr,
   uint64_t in_buf_range_B, ushort restart, uint32_t restart_index,
   enum agx_chip chip, enum agx_primitive topology,
   uint32_t draw_state_context_offset,
   uint32_t render_state_emitted, uint64_t empty_draw_jump,
   enum agx_index_size index_size__3)
{
   uint tid = get_sub_group_local_id();
   bool first = tid == 0;
   enum agx_index_size index_size = index_size__3;

   struct agx_draw draw = agx_draw_indexed(
      cmd->indexCount, cmd->instanceCount, cmd->firstIndex, cmd->vertexOffset,
      cmd->firstInstance, in_buf_ptr, in_buf_range_B, index_size, restart);

   uint in_size_el =
      agx_robust_index_available(cmd->firstIndex, in_buf_range_B, index_size);
   if (in_size_el == 0) {
      /* If the entire draw is out-of-bounds, skip it. We handle this specially
       * for both performance and to avoid integer wrapping issues in the main
       * code path (where cmd->firstIndex could get treated as a negative).
       */
      draw.index_buffer = AGX_ZERO_PAGE_ADDRESS;
      draw.index_buffer_range_B = 4;
      draw.start = 0;
   } else if (cmd->instanceCount != 0 && cmd->indexCount > in_size_el) {
      uint64_t start_B = (uint64_t)draw.start << draw.index_size;
      constant void *in_buf =
         (constant void *)(draw.index_buffer + start_B);

      /* After a small number of zeroes at the end, extra zeroes cannot change
       * triangle/line rendering: they duplicate degenerate primitives. Clamp
       * the number of zeroes we need to extend with for those topologies.
       * Point lists retain every requested zero index.
       */
      draw.b.count[0] =
         agx_robust_index_clamped_count(draw.b.count[0], in_size_el, topology);

      uint out_size_el = draw.b.count[0];
      uint out_size_B = 0, alloc_size_B = 0;
      bool size_valid = agx_robust_index_shadow_sizes(
         out_size_el, index_size, &out_size_B, &alloc_size_B);

      /* Reserve only a representable span. Failed reservations leave bottom
       * unchanged, so parallel draws cannot reuse a successful allocation. */
      uint padded_offs = 0xffffffffu;
      if (first && size_valid) {
         /* Failed parallel allocations must not wrap bottom and reuse a
          * successful draw's storage. Reserve only while the whole span fits. */
         volatile global atomic_uint *bottom =
            (volatile global atomic_uint *)(&heap->bottom);
         uint offset = atomic_load(bottom);
         while (poly_heap_allocation_fits(offset, alloc_size_B, heap->size)) {
            if (atomic_compare_exchange_weak(bottom, &offset,
                                             offset + alloc_size_B)) {
               padded_offs = offset;
               break;
            }
         }
      }

      padded_offs = sub_group_broadcast(padded_offs, 0);
      bool allocated =
         size_valid &&
         poly_heap_allocation_fits(padded_offs, alloc_size_B, heap->size);
      if (!allocated) {
         /* Match nir_lower_printf's abort word. The address is a runtime
          * argument, never embedded in cached code. vk_check_printf_status()
          * turns this sticky execution failure into VK_ERROR_DEVICE_LOST.
          * Finish the VDM continuation safely instead of halting before it is
          * initialized; an empty draw alone is not successful robustness.
          */
         if (first)
            atomic_store((volatile global atomic_uint *)(printf_buffer + 1), 1);

         draw.b.count[0] = 0;
         draw.index_buffer = AGX_ZERO_PAGE_ADDRESS;
         draw.index_buffer_range_B = 4;
         draw.start = 0;
      } else {
         uint in_size_B = agx_indices_to_B(in_size_el, index_size);
         global uchar *padded = heap->base + padded_offs;
         draw.index_buffer = (uintptr_t)padded;
         draw.index_buffer_range_B = out_size_B;
         draw.start = 0;

         agx_robust_index_copy(padded, in_buf, in_size_B, out_size_B, tid);
      }
   }

   if (first) {
      agx_vdm_draw_robust_index(vdm, chip, draw, topology, restart_index,
                                draw_state_context_offset,
                                render_state_emitted != 0, empty_draw_jump);
   }
}

/* Both entrypoints publish failures through the device-owned abort buffer. */
KERNEL(32)
libagx_draw_robust_index(global uint32_t *vdm, global struct poly_heap *heap,
                         global uint32_t *printf_buffer,
                         constant VkDrawIndexedIndirectCommand *cmd,
                         uint64_t in_buf_ptr, uint64_t in_buf_range_B,
                         ushort restart, uint32_t restart_index,
                         enum agx_chip chip, enum agx_primitive topology,
                         uint32_t draw_state_context_offset,
                         uint32_t render_state_emitted,
                         uint64_t empty_draw_jump,
                         enum agx_index_size index_size__3)
{
   libagx_draw_robust_index_impl(
      vdm, heap, printf_buffer, cmd, in_buf_ptr, in_buf_range_B, restart,
      restart_index, chip, topology, draw_state_context_offset,
      render_state_emitted, empty_draw_jump, index_size__3);
}

KERNEL(32)
libagx_draw_robust_index_batch(constant struct libagx_robust_index_draw *draws,
                               global struct poly_heap *heap,
                               global uint32_t *printf_buffer,
                               uint64_t in_buf_ptr, uint64_t in_buf_range_B,
                               ushort restart, uint32_t restart_index,
                               enum agx_chip chip, enum agx_primitive topology,
                               enum agx_index_size index_size__3,
                               uint32_t batch_count)
{
   uint g = get_group_id(0);
   constant struct libagx_robust_index_draw *draw = draws + g;
   uint64_t empty_draw_jump =
      get_sub_group_local_id() == 0
         ? libagx_robust_index_coalesce_empty_jump(draws, g, batch_count, chip)
         : 0;
   libagx_draw_robust_index_impl(
      (global uint32_t *)draw->vdm, heap, printf_buffer,
      (constant VkDrawIndexedIndirectCommand *)draw->command, in_buf_ptr,
      in_buf_range_B, restart, restart_index, chip, topology,
      draw->draw_state_offset, draw->render_state_emitted, empty_draw_jump,
      index_size__3);
}

/*
 * VK_KHR_ray_tracing_pipeline indirect trace rays.
 *
 * The megakernel reads its launch size out of a VkTraceRaysIndirectCommand2KHR
 * record, and the command stream needs the same launch expressed as
 * threadgroups.  Neither is known on the host for an indirect trace, so one
 * invocation reads the application's three dimensions and publishes both.
 *
 * `out_record_dims` is the address of the `width` member of a record the driver
 * assembled on the host (vkCmdTraceRaysIndirectKHR, whose indirect buffer holds
 * only the dimensions), or zero when the application supplied the whole record
 * itself (vkCmdTraceRaysIndirect2KHR).
 */
/*
 * Per-raygen kernel selection for vkCmdTraceRays*.  A ray tracing pipeline
 * compiles one compute kernel per ray-generation group, and the group that a
 * trace runs is named only by the shader binding table record the application
 * points at -- memory whose contents the specification requires to be valid
 * at execution, not at record time (RayTracedGL1, for one, fills its SBT with
 * a GPU copy recorded in the same command buffer).  So the selection happens
 * here: read the raygen record's group handle, and publish one
 * VkDispatchIndirectCommand per kernel: the launch's threadgroup count for the
 * kernel whose raygen handle matches, zero for every other kernel.  The host
 * then records one indirect dispatch per kernel.
 *
 * `record` is the VkTraceRaysIndirectCommand2KHR every trace form publishes
 * (hk_cmd_trace_rays.c); its first quadword is raygenShaderRecordAddress and
 * the group handle's index is the first dword of that record
 * (hk_ray_tracing_core.h: hk_ray_tracing_group_handle).  `in_dims` is where the
 * three launch dimensions live right now; `out_record_dims` is where they must
 * be copied to for the kernels to read, or zero when the record already holds
 * them.  `handles` lists each kernel's raygen handle index.
 */
/* Same non-wrapping round-up as hk_rt_launch_group_count: 8x4 tiles for 2D/3D
 * launches (height > 1) with 32-lane workgroups, 32x1 strips for 1D launches. */
static inline uint2
libagx_trace_rays_group_count(uint32_t width, uint32_t height,
                              uint32_t local_size_x_log2)
{
   if (local_size_x_log2 == 5u && height > 1u) {
      return (uint2)(
         (width >> 3u) + ((width & 7u) ? 1u : 0u),
         (height >> 2u) + ((height & 3u) ? 1u : 0u)
      );
   }
   const uint32_t mask = (1u << local_size_x_log2) - 1u;
   return (uint2)(
      (width >> local_size_x_log2) + ((width & mask) ? 1u : 0u),
      height
   );
}

KERNEL(1)
libagx_trace_rays_select_setup(constant uint64_t *record,
                               global uint32_t *out_record_dims,
                               global VkDispatchIndirectCommand *out_groups,
                               constant uint32_t *in_dims,
                               constant uint32_t *handles,
                               uint32_t kernel_count,
                               uint32_t local_size_x_log2)
{
   uint32_t width = in_dims[0];
   uint32_t height = in_dims[1];
   uint32_t depth = in_dims[2];

   if (out_record_dims) {
      out_record_dims[0] = width;
      out_record_dims[1] = height;
      out_record_dims[2] = depth;
   }

   const uint2 groups_xy =
      libagx_trace_rays_group_count(width, height, local_size_x_log2);

   constant uint32_t *raygen_handle = (constant uint32_t *)(uintptr_t)record[0];
   const uint32_t selected = *raygen_handle;

   for (uint32_t k = 0; k < kernel_count; ++k) {
      const bool run = handles[k] == selected;
      out_groups[k] = (VkDispatchIndirectCommand){
         .x = run ? groups_xy.x : 0u,
         .y = run ? groups_xy.y : 0u,
         .z = run ? depth : 0u,
      };
   }
}

/*
 * Single-kick variant of libagx_trace_rays_select_setup, used on M5. Instead of K indirect dispatches, the host
 * records one Stream Link with return into `slot`, and this invocation writes
 * the selected kernel's launch there, as Metal's indirect command buffers do
 * (a GPU-written 40-byte direct launch reached through a barrier and a link).
 *
 * `templates` holds, per kernel, the 11 words the host encoded for that
 * kernel: the 40-byte direct launch (sizes are overwritten here) and its
 * trailing barrier. `slot` is {0x60000000, launch[10], barrier, return}; the
 * host wrote words 0 and 12. With no matching kernel the launch keeps kernel
 * 0's header with every size zero, the encoding Metal uses for an empty
 * command. `out_groups` still receives the VkDispatchIndirectCommand the
 * kernels' root descriptor points at.
 */
KERNEL(1)
libagx_trace_rays_select_launch(constant uint64_t *record,
                                global uint32_t *out_record_dims,
                                global VkDispatchIndirectCommand *out_groups,
                                constant uint32_t *in_dims,
                                constant uint32_t *handles,
                                constant uint32_t *templates,
                                global uint32_t *slot,
                                uint32_t kernel_count,
                                uint32_t local_size_x_log2)
{
   uint32_t width = in_dims[0];
   uint32_t height = in_dims[1];
   uint32_t depth = in_dims[2];

   if (out_record_dims) {
      out_record_dims[0] = width;
      out_record_dims[1] = height;
      out_record_dims[2] = depth;
   }

   const uint2 groups_xy =
      libagx_trace_rays_group_count(width, height, local_size_x_log2);
   const uint32_t groups_x = groups_xy.x;
   const uint32_t groups_y = groups_xy.y;

   constant uint32_t *raygen_handle = (constant uint32_t *)(uintptr_t)record[0];
   const uint32_t selected = *raygen_handle;

   uint32_t chosen = kernel_count;
   for (uint32_t k = 0; k < kernel_count; ++k) {
      if (chosen == kernel_count && handles[k] == selected)
         chosen = k;
   }

   const bool run = chosen < kernel_count && groups_x && groups_y && depth;
   *out_groups = (VkDispatchIndirectCommand){
      .x = run ? groups_x : 0u,
      .y = run ? groups_y : 0u,
      .z = run ? depth : 0u,
   };

   constant uint32_t *t = templates + 11u * (run ? chosen : 0u);
   uint32_t word0 = t[0];
   if (!run || (groups_y < 2u && depth < 2u))
      word0 &= 0xfff1ffffu;
   slot[1] = word0;
   slot[2] = t[1];
   slot[3] = t[2];
   slot[4] = t[3];
   slot[5] = run ? (groups_x << local_size_x_log2) : 0u;
   slot[6] = run ? groups_y : 0u;
   slot[7] = run ? depth : 0u;
   slot[8] = run ? t[7] : 0u;
   slot[9] = run ? t[8] : 0u;
   slot[10] = run ? t[9] : 0u;
   slot[11] = t[10];
}

KERNEL(1)
libagx_trace_rays_indirect_setup(global uint32_t *out_record_dims,
                                 global VkDispatchIndirectCommand *out_groups,
                                 constant uint32_t *in_dims,
                                 uint32_t local_size_x_log2)
{
   uint32_t width = in_dims[0];
   uint32_t height = in_dims[1];
   uint32_t depth = in_dims[2];

   if (out_record_dims) {
      out_record_dims[0] = width;
      out_record_dims[1] = height;
      out_record_dims[2] = depth;
   }

   /* The megakernel bounds-checks its own launch id, so a partial trailing
    * workgroup retires without tracing.
    *
    * The workgroup size is a power of two the driver owns, so this rounds up
    * with a shift rather than a division: an integer divide by a runtime value
    * is the one arithmetic form most likely to need an encoding this backend
    * does not have, and there is no reason to risk it for a constant. */
   /* Round up WITHOUT the (width + mask) form: that wraps for any width above
    * 0xFFFFFFE0 and silently launches zero workgroups.  maxRayDispatchInvocation-
    * Count is 2^30, so a conforming application cannot reach it -- but the
    * width comes out of a device buffer nothing validates, and a wrap here is
    * a whole trace that quietly does not happen.  The host side computes the
    * same thing the same way (hk_rt_launch_group_count), and
    * test-hk-ray-tracing-core pins the two together at the top of the range;
    * the first version of this kernel used the wrapping form and that test is
    * what caught it. */
   const uint2 groups_xy =
      libagx_trace_rays_group_count(width, height, local_size_x_log2);
   *out_groups = (VkDispatchIndirectCommand){
      .x = groups_xy.x,
      .y = groups_xy.y,
      .z = depth,
   };
}

/* Minimal 1-thread no-op helper used as the zero-grid fallback launch on G18G
 * so an indirect compute stream never submits an empty or zero-sized launch. */
KERNEL(1)
libagx_g18_noop(global uint32_t *scratch)
{
   *scratch = 0u;
}

/* Materialize a 48-byte sealed G18G direct CDM stream (40-byte direct launch +
 * 4-byte strong barrier + 4-byte terminate) from an indirect grid buffer.
 * mode_local == 0 (INDIRECT_GLOBAL): in_grid[0..2] are group counts, local size
 * comes from local_{x,y,z}; in_grid[3..5] are not read.
 * mode_local != 0 (INDIRECT_LOCAL): in_grid[0..2] are global thread counts and
 * in_grid[3..5] are local sizes.
 * If any dimension is zero, emit the pre-built 1-thread no-op launch instead. */
KERNEL(1)
libagx_g18_indirect_launch(constant uint32_t *in_grid,
                           constant uint32_t *tmpl,
                           constant uint32_t *noop_tmpl,
                           global uint32_t *out_stream,
                           uint32_t local_x,
                           uint32_t local_y,
                           uint32_t local_z,
                           uint32_t mode_local)
{
   uint32_t gx = in_grid[0];
   uint32_t gy = in_grid[1];
   uint32_t gz = in_grid[2];
   uint32_t lx = local_x;
   uint32_t ly = local_y;
   uint32_t lz = local_z;
   if (mode_local) {
      lx = in_grid[3];
      ly = in_grid[4];
      lz = in_grid[5];
   }

   uint32_t tx = mode_local ? gx : (gx * lx);
   uint32_t ty = mode_local ? gy : (gy * ly);
   uint32_t tz = mode_local ? gz : (gz * lz);
   bool run = (gx != 0u) && (gy != 0u) && (gz != 0u) &&
              (lx != 0u) && (ly != 0u) && (lz != 0u);

   constant uint32_t *src = run ? tmpl : noop_tmpl;
   uint32_t word0 = src[0];
   if (run && ty < 2u && tz < 2u)
      word0 &= 0xfff1ffffu;

   out_stream[0] = word0;
   out_stream[1] = src[1];
   out_stream[2] = src[2];
   out_stream[3] = src[3];
   out_stream[4] = run ? tx : src[4];
   out_stream[5] = run ? ty : src[5];
   out_stream[6] = run ? tz : src[6];
   out_stream[7] = run ? lx : src[7];
   out_stream[8] = run ? ly : src[8];
   out_stream[9] = run ? lz : src[9];
   out_stream[10] = src[10];
   out_stream[11] = src[11];
}

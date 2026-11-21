/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 * SPDX-License-Identifier: MIT
 */
#include "hk_queue.h"
#include "hk_native_write_class.h"
#include "hk_acceleration_structure.h"
#include "hk_buffer.h"

#include "agx_bg_eot.h"
#include "agx_bo.h"
#include "agx_device.h"
#include "agx_pack.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "agx_vdm_hal200_bind_tail_core.h"
#include "decode.h"
#include "hk_cmd_buffer.h"
#include "hk_compute_kick.h"
#include "hk_frame_anatomy.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_drm_render_core.h"
#include "hk_drm_prior_core.h"
#include "hk_execution_allowance.h"
#include "hk_pack.h"
#include "hk_image.h"
#include "hk_physical_device.h"
#include "hk_query_pool.h"
#include "hk_sparse.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <xf86drm.h>
#include "util/cache_ops.h"
#include "util/list.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/u_atomic.h"
#include "util/u_debug.h"
#include "util/u_call_once.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"

#include "hk_private.h"
#include "layout.h"
#include "vk_common_entrypoints.h"
#include "vk_drm_syncobj.h"
#include "vk_sync.h"

/* Signal scope includes all earlier commands on this Vulkan queue. A DRM
 * timeline point retains its predecessor fence chain; an individual submit's
 * aggregate fence covers only that submit. Transferring the queue point orders
 * completion without introducing a GPU execution dependency or a CPU wait.
 */
static VkResult
queue_signal_timeline(struct hk_device *dev, struct hk_queue *queue,
                      struct vk_queue_submit *submit)
{
   int fd = dev->dev.fd;

   /* Each output becomes signalled only when this queue point and its
    * predecessor chain complete.
    */
   for (unsigned i = 0; i < submit->signal_count; ++i) {
      struct vk_sync_signal *signal = &submit->signals[i];

      assert(vk_sync_type_is_drm_syncobj(signal->sync->type));
      const struct vk_drm_syncobj *syncobj =
         vk_sync_as_drm_syncobj(signal->sync);

      int ret = drmSyncobjTransfer(
         fd, syncobj->syncobj, signal->signal_value,
         queue->drm.syncobj, queue->drm.timeline_value, 0);
      if (ret)
         return vk_errorf(dev, VK_ERROR_UNKNOWN,
                          "DRM_IOCTL_SYNCOBJ_TRANSFER failed: %m");
   }

   return VK_SUCCESS;
}

/*
 * We need to specially handle submits with no control streams. The kernel
 * can't accept empty submits, but we can end up here in Vulkan for
 * synchronization purposes only. Rather than submit a no-op job (slow),
 * we simply tie the fences together.
 */
static VkResult
queue_submit_empty(struct hk_device *dev, struct hk_queue *queue,
                   struct vk_queue_submit *submit)
{
   int fd = dev->dev.fd;

   /* Transfer the waits into the queue timeline. */
   for (unsigned i = 0; i < submit->wait_count; ++i) {
      struct vk_sync_wait *wait = &submit->waits[i];

      assert(vk_sync_type_is_drm_syncobj(wait->sync->type));
      const struct vk_drm_syncobj *syncobj = vk_sync_as_drm_syncobj(wait->sync);

      int ret = drmSyncobjTransfer(
         fd, queue->drm.syncobj, ++queue->drm.timeline_value,
         syncobj->syncobj, wait->wait_value, 0);
      if (ret)
         return vk_errorf(dev, VK_ERROR_UNKNOWN,
                          "DRM_IOCTL_SYNCOBJ_TRANSFER failed: %m");
   }

   return queue_signal_timeline(dev, queue, submit);
}

/* The ray-query stage state of a G17P render whose vertex or fragment
 * programs trace rays; the stages that trace none stay zero. */
static void
hk_gen1_render_ray_query_stages(struct hk_device *dev, const struct hk_cs *cs,
                                struct drm_asahi_render_ray_query_stage rq[2])
{
   for (unsigned stage = 0; stage < 2; ++stage) {
      if (!(stage ? cs->gen1_native_fragment_rq : cs->gen1_native_vertex_rq))
         continue;
      struct drm_asahi_render_ray_query_stage *rt = &rq[stage];
      rt->scs_layout = hk_gen1_ray_query_scs_layout(dev);
      rt->intersection_pool = dev->gen1_rq_ift->va->addr;
      rt->intersection_pool_bytes = 0x4000;
      rt->intersection_first = 1;
      rt->intersection_count = 112;
      rt->intersection_default = UINT64_C(0x10010020000);
      rt->intersection_default_control = UINT64_C(0x16000000000);
   }
}

static void
asahi_fill_cdm_command(struct hk_device *dev, struct hk_cs *cs,
                       bool no_query_access,
                       struct drm_asahi_cmd_compute *cmd)
{
   /* A linked stream leaves its root through a link record, so the root
    * allocation bounds the part the kernel is given. */
   size_t len =
      cs->stream_linked ? cs->root_size : (cs->current - cs->start);

   *cmd = (struct drm_asahi_cmd_compute){
      .flags = (cs->m5_matrix_ops ? DRM_ASAHI_COMPUTE_USES_MATRIX : 0) |
               (no_query_access &&
                      (dev->dev.params.features &
                       DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS)
                   ? DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS
                   : 0) |
               (cs->visible_completion ? DRM_ASAHI_COMPUTE_VISIBLE_COMPLETION : 0),
      .cdm_ctrl_stream_base = cs->addr,
      .cdm_ctrl_stream_end = cs->addr + len,

      .ts.start.handle = cs->timestamp.start.handle,
      .ts.start.offset = cs->timestamp.start.offset_B,
      .ts.end.handle = cs->timestamp.end.handle,
      .ts.end.offset = cs->timestamp.end.offset_B,
   };

   if (cs->uses_sampler_heap && !agx_stream_sampler_state(&dev->dev)) {
      cmd->sampler_heap = dev->samplers.table.bo->va->addr;
      cmd->sampler_count = dev->samplers.table.alloc;
   }

   if (cs->scratch.cs.main || cs->scratch.cs.preamble) {
      cmd->helper.data = dev->scratch.cs.buf->va->addr;
      cmd->helper.cfg = cs->scratch.cs.preamble ? (1 << 16) : 0;
      cmd->helper.binary = agx_helper_program(&dev->bg_eot);
   }
}

/* Fill a compute command of the submission at queue->drm.timeline_value and
 * record the timestamps the kernel will publish for it. NO_QUERY_ACCESS is
 * decided by the query words the stream touches, against the stamps still
 * pending on the device (renders and compute commands earlier in this
 * submission included), not by query_access. */
static void
hk_queue_fill_cdm_command(struct hk_device *dev, struct hk_queue *queue,
                          struct hk_cs *cs, struct drm_asahi_cmd_compute *cmd)
{
   const uint64_t point = queue->drm.timeline_value;
   asahi_fill_cdm_command(dev, cs,
                          !hk_query_stamps_conflict(dev, queue, point, cs), cmd);
   hk_query_stamps_note_cs(dev, queue, point, cs);
}

/* Neither attachment storage nor empty-tile operations may observe the
 * layers eliminated below.  Draws and their XY coverage remain untouched.
 */
static bool
hk_g17p_attachmentless_render(const struct hk_device *dev,
                                    const struct hk_cs *cs)
{
   if (dev->dev.chip != AGX_CHIP_G17P ||
       cs->tib.sample_size_B != 0 || cs->cr.process_empty_tiles ||
       cs->cr.depth.buffer || cs->cr.depth.meta ||
       cs->cr.stencil.buffer || cs->cr.stencil.meta)
      return false;

   /* The tilebuffer builder normalizes an attachmentless sample count to
    * one.  Check attachment formats explicitly; nr_samples is not a tag.
    * This also excludes colour attachments spilled outside the tilebuffer.
    */
   for (unsigned i = 0; i < ARRAY_SIZE(cs->tib.logical_format); ++i) {
      if (cs->tib.logical_format[i] != PIPE_FORMAT_NONE)
         return false;
   }

   return true;
}

static void
asahi_fill_vdm_command(struct hk_device *dev, struct hk_cs *cs,
                       struct drm_asahi_cmd_render *c)
{
   memset(c, 0, sizeof(*c));

   c->vdm_ctrl_stream_base =
      hk_g17p_vdm_low_window_enabled(&dev->dev)
         ? hk_g17p_vdm_alias(cs->cmd, cs->addr)
         : cs->addr;

   agx_pack(&c->ppp_ctrl, CR_PPP_CONTROL, cfg) {
      cfg.enable_w_clamp = true;
      cfg.fixed_point_format = 1;
   }

   const bool empty_attachmentless =
      cs->stats.cmds == 0 && hk_g17p_attachmentless_render(dev, cs);
   const bool layer_zero_attachmentless =
      cs->stats.cmds > 0 && !cs->g17p_layer_zero_unknown &&
      cs->g17p_layer_zero_draws == cs->stats.cmds &&
      hk_g17p_attachmentless_render(dev, cs);
   c->width_px = empty_attachmentless ? 1 : cs->cr.width;
   c->height_px = empty_attachmentless ? 1 : cs->cr.height;

   /* Draws stay inside the render area and the tiles past its bottom-right
    * corner keep their contents, so do not walk them: user interfaces render
    * small areas of full-screen targets. */
   if (cs->area_x1 && cs->area_y1) {
      c->width_px = MIN2(c->width_px, cs->area_x1);
      c->height_px = MIN2(c->height_px, cs->area_y1);
   }

   /* An attachmentless pass takes its size from the render area, which can be
    * empty, and the kernel refuses a zero dimension. */
   c->width_px = MAX2(c->width_px, 1);
   c->height_px = MAX2(c->height_px, 1);

   c->isp_bgobjdepth = cs->cr.isp_bgobjdepth;
   c->isp_bgobjvals = cs->cr.isp_bgobjvals;

   static_assert(sizeof(c->zls_ctrl) == sizeof(cs->cr.zls_control));
   memcpy(&c->zls_ctrl, &cs->cr.zls_control, sizeof(cs->cr.zls_control));

   const bool g18g = dev->dev.chip == AGX_CHIP_G18G;
   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip) ||
       g18g) {
      c->isp_zls_pixels = agx_chip_g17_m5(dev->dev.chip) || g18g
         ? hk_drm_m5_zls_pixels(
              cs->cr.zls_width, cs->cr.zls_height, cs->cr.depth.buffer,
              cs->cr.depth.meta, cs->cr.stencil.buffer, cs->cr.stencil.meta)
         : hk_drm_g17p_zls_pixels(
         cs->cr.zls_width, cs->cr.zls_height, cs->cr.depth.buffer,
         cs->cr.depth.meta, cs->cr.stencil.buffer, cs->cr.stencil.meta);
      c->isp_bgobjdepth = hk_drm_g17p_depth_clear(
         c->isp_bgobjdepth, cs->cr.depth.buffer, cs->cr.depth.meta,
         cs->cr.stencil.buffer, cs->cr.stencil.meta);
   } else {
      agx_pack(&c->isp_zls_pixels, CR_ISP_ZLS_PIXELS, cfg) {
         cfg.x = cs->cr.zls_width;
         cfg.y = cs->cr.zls_height;
      }
   }

   c->depth.base = cs->cr.depth.buffer;
   c->depth.stride = cs->cr.depth.stride;
   c->depth.comp_base = cs->cr.depth.meta;
   c->depth.comp_stride = cs->cr.depth.meta_stride;
   c->stencil.base = cs->cr.stencil.buffer;
   c->stencil.stride = cs->cr.stencil.stride;
   c->stencil.comp_base = cs->cr.stencil.meta;
   c->stencil.comp_stride = cs->cr.stencil.meta_stride;

   if (cs->cr.dbias_is_int == U_TRISTATE_YES) {
      c->flags |= DRM_ASAHI_RENDER_DBIAS_IS_INT;
   }

   /* Native sets the fragment stage's few-primitives bit at the end of a
    * render pass whose known primitive count is at most four, and never for
    * a continued pass or one with uncounted (indirect, mesh) work. This
    * stream's bound counts every draw of the whole pass, so it proves that. */
   if ((agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
       (dev->dev.params.features & DRM_ASAHI_FEATURE_FEW_PRIMITIVES) &&
       cs->g17p_pass_begins && cs->g17p_pass_ends &&
       cs->g17p_primitive_bound <= 4) {
      c->flags |= DRM_ASAHI_RENDER_FEW_PRIMITIVES;
   }

   /* The A18 Pro loses the vertex output of the mesh workgroup in flight when
    * the tiler heap runs out mid-draw. Ask the kernel to size the heap for
    * mesh output before the render. */
   if ((dev->dev.params.features & DRM_ASAHI_FEATURE_RENDER_MESH_OUTPUT) &&
       cs->g17p_mesh_output)
      c->flags |= DRM_ASAHI_RENDER_MESH_OUTPUT;

   if (agx_chip_g17_m5(dev->dev.chip))
      c->flags |= hk_m5_shader_write_class_flags(dev->dev.params.features,
                                                cs->m5_shader_writes_memory);

   if ((dev->dev.debug & AGX_DBG_NOCLUSTER) &&
       dev->dev.chip != AGX_CHIP_G17P) {
      c->flags |= DRM_ASAHI_RENDER_NO_VERTEX_CLUSTERING;
   }

   agx_tilebuffer_set_drm_cmd(c, &cs->tib);

   /* Can be 0 for attachmentless rendering with no draws */
   c->samples = MAX2(cs->tib.nr_samples, 1);
   c->layers = (empty_attachmentless || layer_zero_attachmentless)
                  ? 1 : cs->cr.layers;

   /* Drawing max size will OOM and fail submission. But vkd3d-proton does this
    * for emulating no-attachment rendering. Clamp to something reasonable and
    * hope this is good enough in practice. This only affects a case that would
    * otherwise be guaranteed broken.
    *
    * XXX: Hack for vkd3d-proton.
    */
   if (c->layers == 2048 && c->width_px == 16384 && c->height_px == 16384) {
      mesa_log(MESA_LOG_WARN, MESA_LOG_TAG, "Clamping massive framebuffer");
      c->layers = 32;
   }

   c->ppp_multisamplectl = cs->ppp_multisamplectl;
   c->sample_size_B = cs->tib.sample_size_B;

   float tan_60 = 1.732051f;
   c->isp_merge_upper_x = fui(tan_60 / c->width_px);
   c->isp_merge_upper_y = fui(tan_60 / c->height_px);

   c->bg.usc = cs->cr.bg.main.usc | 4;
   c->eot.usc = cs->cr.eot.main.usc | 4;
   c->partial_bg.usc = cs->cr.bg.partial.usc | 4;
   c->partial_eot.usc = cs->cr.eot.partial.usc | 4;

   hk_drm_render_resource_specs_pack(
      &(struct hk_drm_render_resource_specs){
         .bg = cs->cr.bg.main.rsrc_spec,
         .eot = cs->cr.eot.main.rsrc_spec,
         .partial_bg = cs->cr.bg.partial.rsrc_spec,
         .partial_eot = cs->cr.eot.partial.rsrc_spec,
      },
      c);

   c->isp_scissor_base = cs->uploaded_scissor;
   c->isp_dbias_base = cs->uploaded_zbias;

   if (cs->uses_sampler_heap && !agx_stream_sampler_state(&dev->dev)) {
      c->sampler_heap = dev->samplers.table.bo->va->addr;
      c->sampler_count = dev->samplers.table.alloc;
   }

   c->isp_oclqry_base = dev->occlusion_queries.bo->va->addr;

   if (cs->cr.process_empty_tiles)
      c->flags |= DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES;

   /* Only a render that is its whole API pass: the streams hk splits one
    * pass into carry no barrier between them and rely on the kernel's
    * attachment ordering. */
   if (cs->cr.whole_attachments && cs->g17p_pass_begins &&
       cs->g17p_pass_ends && !cs->m5_continued_pass &&
       (dev->dev.params.features & DRM_ASAHI_FEATURE_WHOLE_ATTACHMENT_RENDERS))
      c->flags |= DRM_ASAHI_RENDER_WHOLE_ATTACHMENTS;

   if (cs->scratch.vs.main || cs->scratch.vs.preamble) {
      c->flags |= DRM_ASAHI_RENDER_VERTEX_SCRATCH;
      c->vertex_helper.data = dev->scratch.vs.buf->va->addr;
      c->vertex_helper.cfg = cs->scratch.vs.preamble ? (1 << 16) : 0;
      c->vertex_helper.binary = agx_helper_program(&dev->bg_eot);
   }

   if (cs->scratch.fs.main || cs->scratch.fs.preamble) {
      c->fragment_helper.data = dev->scratch.fs.buf->va->addr;
      c->fragment_helper.cfg = cs->scratch.fs.preamble ? (1 << 16) : 0;
      c->fragment_helper.binary = agx_helper_program(&dev->bg_eot);
   }

   if (cs->timestamp.start.handle) {
      c->ts_vtx.start.handle = cs->timestamp.start.handle;
      c->ts_vtx.start.offset = cs->timestamp.start.offset_B;
   }

   if (cs->timestamp.end.handle) {
      c->ts_frag.end.handle = cs->timestamp.end.handle;
      c->ts_frag.end.offset = cs->timestamp.end.offset_B;
   }

   /* Occlusion availability (hk_stamp_oq_available()). */
   struct drm_asahi_timestamp *slots[] = {
      &c->ts_vtx.start, &c->ts_vtx.end, &c->ts_frag.start, &c->ts_frag.end,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(slots); ++i) {
      if (cs->oq_stamp[i].handle) {
         assert(!slots[i]->handle);
         slots[i]->handle = cs->oq_stamp[i].handle;
         slots[i]->offset = cs->oq_stamp[i].offset_B;
      }
   }
}

static void
asahi_fill_sync(struct drm_asahi_sync *sync, struct vk_sync *vk_sync,
                uint64_t value)
{
   if (unlikely(!vk_sync_type_is_drm_syncobj(vk_sync->type))) {
      UNREACHABLE("Unsupported sync type");
      return;
   }

   const struct vk_drm_syncobj *syncobj = vk_sync_as_drm_syncobj(vk_sync);
   *sync = (struct drm_asahi_sync){.handle = syncobj->syncobj};

   if (vk_sync->flags & VK_SYNC_IS_TIMELINE) {
      sync->sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ;
      sync->timeline_value = value;
   } else {
      sync->sync_type = DRM_ASAHI_SYNC_SYNCOBJ;
   }
}

union drm_asahi_cmd {
   struct drm_asahi_cmd_compute compute;
   struct drm_asahi_cmd_render render;
};

static void
asahi_append_compute_attachments(struct hk_device *dev,
                                 struct util_dynarray *payload,
                                 const struct hk_cs *cs)
{
   /* Attachment spans are hints for the native AGX3 firmware interface.
    * AGX2 has never sent them, so G13 and G14 keep that submission. G18G
    * (M6) sends them as well. */
   if (!agx_native_graphics(&dev->dev) && dev->dev.chip != AGX_CHIP_G18G)
      return;

   const struct hk_drm_attachment_set *set = &cs->compute_attachments;
   const uint32_t count = MIN2(set->count, dev->dev.params.max_attachments);
   const struct drm_asahi_cmd_header header = {
      .cmd_type = DRM_ASAHI_SET_COMPUTE_ATTACHMENTS,
      .size = count * sizeof(set->list[0]),
      .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
      .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
   };
   /* Clear the preceding command's list even for an attachmentless dispatch. */
   util_dynarray_append(payload, header);
   if (count)
      util_dynarray_append_array(payload, struct drm_asahi_attachment,
                                 set->list, count);
}

static void
asahi_append_fragment_attachments(struct hk_device *dev,
                                  struct util_dynarray *payload,
                                  const struct hk_cs *cs)
{
   /* See asahi_append_compute_attachments(). */
   if (!agx_native_graphics(&dev->dev) && dev->dev.chip != AGX_CHIP_G18G)
      return;

   const struct hk_drm_attachment_set *set = &cs->fragment_attachments;
   const uint32_t count = MIN2(set->count, dev->dev.params.max_attachments);
   const struct drm_asahi_cmd_header header = {
      .cmd_type = DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS,
      .size = count * sizeof(set->list[0]),
      .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
      .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
   };

   /* Emit even an empty set. Attachment commands affect subsequent hardware
    * commands in the same submit, so an attachmentless render must clear the
    * set retained by the preceding render. */
   util_dynarray_append(payload, header);
   if (count)
      util_dynarray_append_array(payload, struct drm_asahi_attachment,
                                 set->list, count);

   if (!set->complete || count != set->count) {
      perf_debug_dev(&dev->dev,
                     "Fragment attachment hint incomplete: retained %u, emitted %u (kernel limit %u)",
                     set->count, count, dev->dev.params.max_attachments);
   }
}

/* Native AGX3 benefits from submitting ordered control streams together.
 * G13X already serializes both engines, and separate ioctls let the kernel
 * start a render while the following compute submission is prepared. This
 * matters for post-graphics query availability and its CPU round trip.
 */
static inline unsigned
max_commands_per_submit(struct hk_device *dev)
{
   return hk_drm_submit_command_limit(
      dev->dev.chip != AGX_CHIP_G13X && !HK_PERF(dev, NOBATCH),
      dev->dev.params.max_commands_per_submission);
}

static VkResult
queue_submit_single(struct hk_device *dev, struct drm_asahi_submit *submit,
                    unsigned ring_idx, struct hk_cmd_pending *pending)
{
   /* Give each ioctl an independent binary completion fence. API signals
    * may be signalled on the CPU during error unwinding; timeline lookup
    * can prune a completed error fence. Neither is a safe retirement token. */
   struct drm_asahi_submit pinned = *submit;
   if (pending) {
      const unsigned count = submit->in_sync_count + submit->out_sync_count;
      struct drm_asahi_sync *syncs =
         alloca((count + 1) * sizeof(*syncs));
      if (count)
         memcpy(syncs, (void *)(uintptr_t)submit->syncs, count * sizeof(*syncs));
      syncs[count] = (struct drm_asahi_sync) {
         .sync_type = DRM_ASAHI_SYNC_SYNCOBJ,
         .handle = hk_cmd_pending_signal(pending),
      };
      pinned.syncs = (uintptr_t)syncs;
      pinned.out_sync_count++;
      submit = &pinned;
   }
   struct agx_submit_virt virt = {.ring_idx = ring_idx};

   /* Complete the cache cleans of CPU-cached descriptor writes
    * (hk_descriptor_set_publish()) before the GPU can read them. */
   if (dev->dev.chip == AGX_CHIP_G17P && util_has_cache_ops())
      util_post_flush_fence();

   if (dev->dev.is_virtio) {
      u_rwlock_rdlock(&dev->external_bos.lock);
      virt.extres_count = util_dynarray_num_elements(
         &dev->external_bos.list, struct asahi_ccmd_submit_res);
      virt.extres = util_dynarray_begin(&dev->external_bos.list);
   }

   int ret = dev->dev.ops.submit(&dev->dev, submit, &virt);

   if (dev->dev.is_virtio)
      u_rwlock_rdunlock(&dev->external_bos.lock);

   if (ret) {
      /* ASAHI_MESA_DEBUG=trace decodes every command stream before it is
       * submitted, including one the kernel rejects. */
      return vk_device_set_lost(&dev->vk,
                                "DRM_IOCTL_ASAHI_SUBMIT failed: %m");
   }

   return VK_SUCCESS;
}

/*
 * The kernel/firmware jointly impose a limit on commands per submit ioctl, but
 * we can build up arbitrarily large command buffers. We handle this here by
 * looping the ioctl, submitting slices of the command buffers that are within
 * bounds.
 */
static VkResult
queue_submit_looped(struct hk_device *dev, struct hk_queue *queue,
                    struct drm_asahi_submit *submit, unsigned command_count,
                    unsigned ring_idx, const uint64_t prior_base[2],
                    bool chunk_timelines, uint64_t first_chunk_point,
                    struct hk_frame_anatomy_submit *anatomy,
                    struct hk_cmd_pending *pending)
{
   uint8_t *cmdbuf = (uint8_t *)(uintptr_t)submit->cmdbuf;
   uint32_t offs = 0;
   unsigned submitted_vdm = 0, submitted_cdm = 0;
   unsigned commands_remaining = command_count;
   unsigned chunk_index = 0;
   uint64_t last_attempted_point = first_chunk_point - 1;

   uint64_t out_syncs =
      submit->syncs + sizeof(struct drm_asahi_sync) * submit->in_sync_count;

   struct drm_asahi_sync *chunk_signal = (void *)(uintptr_t)out_syncs;
   if (chunk_timelines)
      assert(submit->out_sync_count == 1 &&
             chunk_signal->sync_type == DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ &&
             chunk_signal->handle == queue->drm.syncobj);

   while (commands_remaining) {
      bool first = commands_remaining == command_count;
      bool last = commands_remaining <= max_commands_per_submit(dev);

      unsigned count = MIN2(commands_remaining, max_commands_per_submit(dev));
      commands_remaining -= count;

      assert(!last || commands_remaining == 0);
      assert(count > 0);

      unsigned base_offs = offs;
      struct hk_drm_command_span span;
      if (!hk_drm_next_command_span(cmdbuf, submit->cmdbuf_size, base_offs,
                                    count, &span)) {
         if (chunk_timelines)
            queue->drm.timeline_value = last_attempted_point;
         return vk_device_set_lost(
            &dev->vk, "Invalid internal Asahi command stream framing");
      }
      offs = span.end_offset;
      unsigned cdm_count = span.compute_count;
      unsigned vdm_count = span.render_count;
      if (last && offs != submit->cmdbuf_size) {
         if (chunk_timelines)
            queue->drm.timeline_value = last_attempted_point;
         return vk_device_set_lost(
            &dev->vk, "Trailing synthetic Asahi commands without hardware");
      }

      /* We need to fix up the barriers since barriers are ioctl-relative */
      uint16_t slice_before[2] = {0, 0};
      for (unsigned scan = base_offs; scan < offs;) {
         struct drm_asahi_cmd_header *cmd = (void *)(cmdbuf + scan);
         scan += sizeof(*cmd) + cmd->size;

         if (cmd->cmd_type != DRM_ASAHI_CMD_RENDER &&
             cmd->cmd_type != DRM_ASAHI_CMD_COMPUTE)
            continue;

         /* A serialized command waits for every earlier one: the commands
          * before it in this ioctl, or with none, all earlier ioctls (0).
          * Its whole-submit index may not fit in 16 bits. */
         if (queue->sync.serialize && !queue->sync.exact_prior) {
            cmd->vdm_barrier = slice_before[0];
            cmd->cdm_barrier = slice_before[1];
            slice_before[cmd->cmd_type == DRM_ASAHI_CMD_COMPUTE]++;
            continue;
         }

         if (!hk_drm_rebase_prior(cmd, cmd + 1, submitted_vdm,
                                  submitted_cdm, prior_base,
                                  queue->sync.exact_prior)) {
            if (chunk_timelines)
               queue->drm.timeline_value = last_attempted_point;
            return vk_device_set_lost(&dev->vk,
                                      "Invalid historical Asahi prefix");
         }
      }

      /* Without per-chunk completion points, only the final ioctl contributes
       * to the output fence. It must cover the preceding chunks explicitly. */
      if (last && !first && !chunk_timelines) {
         assert(span.first_hardware_offset != SIZE_MAX);
         struct drm_asahi_cmd_header *cmd =
            (void *)(cmdbuf + span.first_hardware_offset);

         if (cmd->vdm_barrier == DRM_ASAHI_BARRIER_NONE)
            cmd->vdm_barrier = 0;

         if (cmd->cdm_barrier == DRM_ASAHI_BARRIER_NONE)
            cmd->cdm_barrier = 0;
      }

      /* Without the first-command dependency, later chunks have independent
       * roots too. Each must carry the original input syncs. Legacy kernels
       * retain the first-command chain and receive the waits only once. */
      bool has_in_syncs = first ||
         (dev->dev.params.features & DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS);
      /* Distinct timeline points join completion without making this
       * chunk execute after every command in its predecessor. */
      bool has_out_syncs = last || chunk_timelines;
      if (chunk_timelines)
         chunk_signal->timeline_value = first_chunk_point + chunk_index;

      /* Profiler: barriers of this chunk exactly as rebased above. */
      if (unlikely(anatomy))
         hk_frame_anatomy_note_barriers(anatomy, cmdbuf, base_offs, offs);

      struct drm_asahi_submit submit_ioctl = {
         .flags = submit->flags,
         .queue_id = submit->queue_id,
         .cmdbuf = submit->cmdbuf + base_offs,
         .cmdbuf_size = offs - base_offs,

         .syncs = has_in_syncs ? submit->syncs : out_syncs,
         .in_sync_count = has_in_syncs ? submit->in_sync_count : 0,
         .out_sync_count = has_out_syncs ? submit->out_sync_count : 0,
      };

      if (chunk_timelines)
         last_attempted_point = chunk_signal->timeline_value;
      VkResult result = queue_submit_single(dev, &submit_ioctl, ring_idx, pending);
      if (result != VK_SUCCESS) {
         if (chunk_timelines)
            queue->drm.timeline_value = last_attempted_point;
         return result;
      }
      chunk_index++;

      submitted_cdm += cdm_count;
      submitted_vdm += vdm_count;
   }

   return VK_SUCCESS;
}

struct hk_bind_builder {
   /* Initialized */
   struct hk_device *dev;
   struct vk_object_base *obj_base;
   struct agx_va *va;
   struct hk_buffer *buffer;
   struct hk_image *image;

   /* State */
   struct hk_device_memory *mem;
   VkDeviceSize resourceOffset;
   VkDeviceSize size;
   VkDeviceSize memoryOffset;
   VkResult result;

   /* Array of drm_asahi_gem_bind_op's */
   struct util_dynarray binds;
};

static inline struct hk_bind_builder
hk_bind_builder(struct hk_device *dev, struct vk_object_base *obj_base,
                struct agx_va *va, struct hk_buffer *buffer,
                struct hk_image *image)
{
   struct hk_bind_builder b = {
      .dev = dev,
      .obj_base = obj_base,
      .va = va,
      .buffer = buffer,
      .image = image,
   };

   b.binds = UTIL_DYNARRAY_INIT;
   return b;
}

static VkResult
hk_flush_bind(struct hk_bind_builder *b)
{
   if (b->result != VK_SUCCESS || b->size == 0) {
      return b->result;
   }

   perf_debug(b->dev, "Sparse bind");

   uint64_t va_addr = b->va->addr + b->resourceOffset;

   /* When the app wants to unbind, replace the bound pages with scratch pages
    * so we don't leave a gap.
    */
   struct drm_asahi_gem_bind_op op;
   if (!b->mem && b->buffer && hk_sparse_buffer_unmapped(&b->dev->dev)) {
      op = (struct drm_asahi_gem_bind_op){
         .flags = DRM_ASAHI_BIND_UNBIND,
         .addr = va_addr,
         .range = b->size,
      };
   } else if (!b->mem) {
      op = (struct drm_asahi_gem_bind_op){
         .handle = b->dev->sparse_fill->uapi_handle,
         .flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE |
                  agx_fill_bind_flags(&b->dev->dev),
         .addr = b->va->addr + b->resourceOffset,
         .range = b->size,
      };
   } else {
      op = (struct drm_asahi_gem_bind_op){
         .handle = b->mem->bo->uapi_handle,
         .flags = DRM_ASAHI_BIND_READ | DRM_ASAHI_BIND_WRITE,
         .addr = va_addr,
         .offset = b->memoryOffset,
         .range = b->size,
      };
   }

   util_dynarray_append(&b->binds, op);

   /* Shadow a read-only mapping to the upper half */
   op.flags &= ~DRM_ASAHI_BIND_WRITE;
   op.addr = agx_rw_addr_to_ro(&b->dev->dev, op.addr);

   if (!b->mem) {
      op.handle = b->dev->dev.zero_bo->uapi_handle;
      op.flags = DRM_ASAHI_BIND_READ | agx_fill_bind_flags(&b->dev->dev);
   }

   util_dynarray_append(&b->binds, op);

   return VK_SUCCESS;
}

static VkResult
hk_sparse_buffer_map_commit(struct hk_bind_builder *b)
{
   if (!b->buffer || (!b->buffer->sparse_map &&
                      !hk_buffer_in_physical_sparse_map(b->dev, b->buffer)))
      return VK_SUCCESS;

   struct hk_sparse_buffer_map_header *header =
      b->buffer->sparse_map ? agx_bo_map(b->buffer->sparse_map) : NULL;
   uint32_t *map = header ? (uint32_t *)(header + 1) : NULL;
   struct drm_asahi_gem_bind_op *ops = b->binds.data;
   const unsigned count = util_dynarray_num_elements(
      &b->binds, struct drm_asahi_gem_bind_op);
   assert((count % 2) == 0);

   for (unsigned i = 0; i < count; i += 2) {
      const struct drm_asahi_gem_bind_op *op = &ops[i];
      assert(op->addr >= b->va->addr);
      const uint64_t first = (op->addr - b->va->addr) / AIL_PAGESIZE;
      const uint64_t pages = DIV_ROUND_UP(op->range, AIL_PAGESIZE);
      const bool resident =
         op->handle != b->dev->sparse_fill->uapi_handle &&
         !(op->flags & DRM_ASAHI_BIND_UNBIND);

      VkResult result = hk_update_physical_sparse_map(
         b->dev, op->addr, op->range, !resident);
      if (result != VK_SUCCESS)
         return result;

      for (uint64_t page = 0; map && page < pages; ++page) {
         if (first + page >= b->buffer->sparse_map_entries)
            break;

         const uint64_t target = first + page;
         const uint32_t mask = BITFIELD_BIT(target & 31);
         if (resident)
            map[target >> 5] |= mask;
         else
            map[target >> 5] &= ~mask;
      }
   }

   if (b->buffer->sparse_map)
      agx_bo_assert_range(b->buffer->sparse_map, 0,
                          b->buffer->sparse_map->size);

   return VK_SUCCESS;
}

static void
hk_sparse_image_map_commit(struct hk_bind_builder *b)
{
   if (!b->image || !b->image->planes[0].sparse_map)
      return;

   assert(b->image->plane_count == 1 && "multiplane sparse not supported");

   uint32_t *map = agx_bo_map(b->image->planes[0].sparse_map);
   struct ail_layout *layout = &b->image->planes[0].layout;
   const uint64_t layer_stride_page =
      ail_bytes_to_pages(layout->layer_stride_B);
   struct drm_asahi_gem_bind_op *ops = b->binds.data;
   const unsigned count = util_dynarray_num_elements(
      &b->binds, struct drm_asahi_gem_bind_op);
   assert((count % 2) == 0);

   /* Each logical sparse bind is followed by its read-only shadow.  Publish
    * residency metadata only after the atomic kernel batch succeeded. */
   for (unsigned i = 0; i < count; i += 2) {
      const struct drm_asahi_gem_bind_op *op = &ops[i];
      assert(op->addr >= b->va->addr);
      const uint64_t resource_offset = op->addr - b->va->addr;
      const uint64_t size_page = ail_bytes_to_pages(op->range);
      const bool enabled =
         op->handle != b->dev->sparse_fill->uapi_handle;

      for (uint64_t offs_page = 0; offs_page < size_page; ++offs_page) {
         const uint64_t target_page =
            ail_bytes_to_pages(resource_offset) + offs_page;
         const uint64_t z = target_page / layer_stride_page;
         if (z >= layout->depth_px)
            break;

         const uint64_t page_in_layer = target_page % layer_stride_page;
         const unsigned idx =
            ail_page_to_sparse_index_el(layout, z, page_in_layer);

         agx_pack(map + idx, SPARSE_BLOCK, cfg) {
            cfg.enabled = enabled;
            cfg.unknown = enabled;
            if (enabled)
               cfg.address = op->addr + (offs_page * AIL_PAGESIZE);
         }
      }
   }
}

static VkResult
hk_bind_builder_finish(struct hk_bind_builder *b)
{
   b->result = hk_flush_bind(b);

   /* Submit everything to the kernel at once */
   if (b->result == VK_SUCCESS && b->binds.size > 0) {
      const unsigned op_count =
         util_dynarray_num_elements(&b->binds, struct drm_asahi_gem_bind_op);

      /* One op per hardware tile plus its read-only shadow, so a large
       * layered mip chain submits thousands.  The kernel copies and stages
       * the whole batch, so the op count is what decides whether this bind
       * can be satisfied when memory is tight -- print it when asked, rather
       * than leaving a VM_BIND ENOMEM with no size attached to it. */
      perf_debug_dev(&b->dev->dev, "Sparse bind batch: %u ops (%u bytes)",
                     op_count, (unsigned)b->binds.size);

      int ret = b->dev->dev.ops.bo_bind(&b->dev->dev, b->binds.data,
                                        op_count);
      if (ret) {
         const int error = errno ? errno : EIO;
         b->result = vk_errorf(
            b->dev,
            error == ENOMEM ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_ERROR_UNKNOWN,
            "Sparse VM_BIND batch failed: %s", strerror(error));
      } else {
         b->result = hk_sparse_buffer_map_commit(b);
         if (b->result == VK_SUCCESS)
            hk_sparse_image_map_commit(b);
      }
   }

   util_dynarray_fini(&b->binds);
   return b->result;
}

static void
hk_add_bind(struct hk_bind_builder *b, struct hk_device_memory *mem,
            VkDeviceSize resourceOffset, VkDeviceSize size,
            VkDeviceSize memoryOffset)
{
   /* Discard trivial binds to simplify the below logic. */
   if (size == 0)
      return;

   /* Try to merge with the previous bind */
   if (b->size && b->mem == mem &&
       resourceOffset == b->resourceOffset + b->size &&
       (!mem || memoryOffset == b->memoryOffset + b->size)) {

      b->size += size;
      return;
   }

   /* Otherwise, flush the previous bind and replace with the new one */
   hk_flush_bind(b);
   b->mem = mem;
   b->resourceOffset = resourceOffset;
   b->size = size;
   b->memoryOffset = memoryOffset;
}

static VkResult
hk_sparse_buffer_bind_memory(struct hk_device *device,
                             const VkSparseBufferMemoryBindInfo *bind)
{
   VK_FROM_HANDLE(hk_buffer, buffer, bind->buffer);

   struct hk_bind_builder b =
      hk_bind_builder(device, &buffer->vk.base, buffer->va, buffer, NULL);

   for (uint32_t i = 0; i < bind->bindCount; ++i) {
      struct hk_device_memory *cur_mem = NULL;

      if (bind->pBinds[i].memory != VK_NULL_HANDLE)
         cur_mem = hk_device_memory_from_handle(bind->pBinds[i].memory);

      hk_add_bind(&b, cur_mem, bind->pBinds[i].resourceOffset,
                  bind->pBinds[i].size, bind->pBinds[i].memoryOffset);
   }

   return hk_bind_builder_finish(&b);
}

static VkResult
hk_sparse_image_opaque_bind_memory(
   struct hk_device *device, const VkSparseImageOpaqueMemoryBindInfo *bind)
{
   VK_FROM_HANDLE(hk_image, image, bind->image);

   for (unsigned p = 0; p < image->plane_count; ++p) {
      struct hk_image_plane *plane = &image->planes[p];
      const uint64_t plane_start = plane->offset_B;
      const uint64_t plane_end = plane_start + plane->va->size_B;

      struct hk_bind_builder b =
         hk_bind_builder(device, &image->vk.base, plane->va, NULL, image);

      for (uint32_t i = 0; i < bind->bindCount; ++i) {
         struct hk_device_memory *mem = NULL;
         if (bind->pBinds[i].memory != VK_NULL_HANDLE)
            mem = hk_device_memory_from_handle(bind->pBinds[i].memory);

         VkDeviceSize resourceOffset = bind->pBinds[i].resourceOffset;

         /* Conceptually, the miptail is a single region at the end of the
          * image, possibly layered. However, due to alignment requirements we
          * need to use a non-layered miptail and internally fan out to each of
          * the layers. This is facilitated by the HK_MIP_TAIL_START_OFFSET
          * magic offset, see the comment where that is defined for more
          * detail.
          */
         if (resourceOffset >= HK_MIP_TAIL_START_OFFSET) {
            if (p != 0)
               continue;
            assert(resourceOffset == HK_MIP_TAIL_START_OFFSET &&
                   "must bind whole miptail... maybe...");

            const struct ail_layout *layout = &plane->layout;
            unsigned tail_offset_B =
               layout->level_offsets_B[layout->mip_tail_first_lod];

            for (unsigned z = 0; z < layout->depth_px; ++z) {
               uint64_t image_offs =
                  tail_offset_B + (z * layout->layer_stride_B);
               uint64_t mem_offs = bind->pBinds[i].memoryOffset +
                                   (z * layout->mip_tail_stride);

               hk_add_bind(&b, mem, image_offs, layout->mip_tail_stride,
                           mem_offs);
            }
            continue;
         }

         const uint64_t start = MAX2(resourceOffset, plane_start);
         const uint64_t end =
            MIN2(resourceOffset + bind->pBinds[i].size, plane_end);
         if (start >= end)
            continue;

         hk_add_bind(&b, mem, start - plane_start, end - start,
                     bind->pBinds[i].memoryOffset + (start - resourceOffset));
      }

      VkResult result = hk_bind_builder_finish(&b);
      if (result != VK_SUCCESS)
         return result;
   }

   return VK_SUCCESS;
}

static void
bind_hw_tile(struct hk_bind_builder *b, struct hk_device_memory *mem,
             struct ail_layout *layout, unsigned layer, unsigned level,
             VkOffset3D offset, VkExtent3D extent, struct ail_tile std_size_el,
             unsigned mem_offset, unsigned x, unsigned y, unsigned z)
{
   uint64_t bo_offset_B = hk_sparse_image_tile_offset_B(
      b->dev->dev.chip, layout, level, offset.x + x, offset.y + y,
      layer + offset.z + z);

   /* Consider the standard tiles in the bound memory to be in raster order, and
    * address accordingly in standard tiles.
    */
   unsigned mem_x_stl = x / std_size_el.width_el;
   unsigned mem_y_stl = y / std_size_el.height_el;
   unsigned extent_w_stl = DIV_ROUND_UP(extent.width, std_size_el.width_el);
   unsigned extent_y_stl = DIV_ROUND_UP(extent.height, std_size_el.height_el);
   unsigned mem_offs_stl = (extent_y_stl * extent_w_stl * z) +
                           (extent_w_stl * mem_y_stl) + mem_x_stl;

   /* There are 4 hardware tiles per standard tile, so offset
    * accordingly for each hardware tile.
    */
   unsigned mem_offset_B = mem_offset + (mem_offs_stl * 4 * AIL_PAGESIZE);

   if (x % std_size_el.width_el)
      mem_offset_B += AIL_PAGESIZE;

   if (y % std_size_el.height_el)
      mem_offset_B += (2 * AIL_PAGESIZE);

   hk_add_bind(b, mem, bo_offset_B, AIL_PAGESIZE, mem_offset_B);
}

static VkResult
hk_sparse_image_bind_memory(struct hk_device *device,
                            const VkSparseImageMemoryBindInfo *bind)
{
   VK_FROM_HANDLE(hk_image, image, bind->image);
   struct ail_layout *layout = &image->planes[0].layout;

   struct hk_bind_builder b =
      hk_bind_builder(device, &image->vk.base, image->planes[0].va, NULL,
                      image);

   for (uint32_t i = 0; i < bind->bindCount; ++i) {
      struct hk_device_memory *mem = NULL;
      if (bind->pBinds[i].memory != VK_NULL_HANDLE)
         mem = hk_device_memory_from_handle(bind->pBinds[i].memory);

      uint64_t mem_offset = bind->pBinds[i].memoryOffset;
      const uint32_t layer = bind->pBinds[i].subresource.arrayLayer;
      const uint32_t level = bind->pBinds[i].subresource.mipLevel;

      VkExtent3D bind_extent = bind->pBinds[i].extent;
      bind_extent.width = DIV_ROUND_UP(
         bind_extent.width, vk_format_get_blockwidth(image->vk.format));
      bind_extent.height = DIV_ROUND_UP(
         bind_extent.height, vk_format_get_blockheight(image->vk.format));

      VkOffset3D bind_offset = bind->pBinds[i].offset;
      bind_offset.x /= vk_format_get_blockwidth(image->vk.format);
      bind_offset.y /= vk_format_get_blockheight(image->vk.format);

      /* Hardware tiles are exactly one page (16K) */
      struct ail_tile tilesize_el = layout->tilesize_el[level];
      unsigned size_B = tilesize_el.width_el * tilesize_el.height_el *
                        ail_get_blocksize_B(layout);

      assert(size_B == AIL_PAGESIZE && "fundamental to AGX");

      /* Standard tiles are exactly 4 pages (65K), consisting of a 2x2 grid of
       * hardware tiles.
       */
      struct ail_tile std_size_el = tilesize_el;
      std_size_el.width_el *= 2;
      std_size_el.height_el *= 2;

      for (unsigned z = 0; z < bind_extent.depth; z += 1) {
         for (unsigned y = 0; y < bind_extent.height;
              y += tilesize_el.height_el) {
            for (unsigned x = 0; x < bind_extent.width;
                 x += tilesize_el.width_el) {
               bind_hw_tile(&b, mem, layout, layer, level, bind_offset,
                            bind_extent, std_size_el, mem_offset, x, y, z);
            }
         }
      }
   }

   return hk_bind_builder_finish(&b);
}

VkResult
hk_queue_submit_bind_sparse_memory(struct hk_device *device,
                                   struct vk_queue_submit *submission)
{
   assert(submission->command_buffer_count == 0);

   /* VM_BIND and residency metadata updates execute on the CPU. The common
    * queue runtime only waits for input fences to be submitted; passing those
    * fences to a later GPU submit cannot order these host updates. On M5,
    * wait for completion before changing any mapping: no GPU work there waits
    * on the host (event waits don't poll, see hk_event_wait_polls), so inputs
    * that are already submitted complete without host action. Where the GPU
    * polls for a host vkSetEvent (G17P), this wait could block the thread
    * that would set it. Bound each wait so device loss can cancel it even
    * when a semaphore will never become signalled.
    */
   while (agx_chip_g17_m5(device->dev.chip)) {
      VkResult result = vk_device_check_status(&device->vk);
      if (result != VK_SUCCESS)
         return result;

      result = vk_sync_wait_many(&device->vk, submission->wait_count,
                                 submission->waits, VK_SYNC_WAIT_COMPLETE,
                                 os_time_get_absolute_timeout(100000000));
      if (result == VK_TIMEOUT)
         continue;
      if (result != VK_SUCCESS)
         return result;
      break;
   }

   for (uint32_t i = 0; i < submission->buffer_bind_count; ++i) {
      VkResult result =
         hk_sparse_buffer_bind_memory(device, submission->buffer_binds + i);
      if (result != VK_SUCCESS)
         return result;
   }

   for (uint32_t i = 0; i < submission->image_opaque_bind_count; ++i) {
      VkResult result = hk_sparse_image_opaque_bind_memory(
         device, submission->image_opaque_binds + i);
      if (result != VK_SUCCESS)
         return result;
   }

   for (uint32_t i = 0; i < submission->image_bind_count; ++i) {
      VkResult result =
         hk_sparse_image_bind_memory(device, submission->image_binds + i);
      if (result != VK_SUCCESS)
         return result;
   }

   return VK_SUCCESS;
}

static bool
hk_sparse_submission_updates_sstt(const struct vk_queue_submit *submission)
{
   for (uint32_t i = 0; i < submission->image_opaque_bind_count; ++i) {
      VK_FROM_HANDLE(hk_image, image, submission->image_opaque_binds[i].image);
      if (submission->image_opaque_binds[i].bindCount &&
          image->planes[0].sparse_map)
         return true;
   }

   for (uint32_t i = 0; i < submission->image_bind_count; ++i) {
      VK_FROM_HANDLE(hk_image, image, submission->image_binds[i].image);
      if (submission->image_binds[i].bindCount &&
          image->planes[0].sparse_map)
         return true;
   }

   return false;
}

/* The render stages (HK_RENDER_RQ_*) in which a command buffer runs ray
 * queries, each of which needs its render ray-query scratch reservation.
 */
static unsigned
hk_cmd_buffer_needs_render_rq(struct hk_cmd_buffer *cmdbuf)
{
   unsigned stages = 0;

   list_for_each_entry(struct hk_cs, cs, &cmdbuf->control_streams, node) {
      if (cs->type != HK_CS_VDM)
         continue;
      if (cs->gen1_native_vertex_rq)
         stages |= HK_RENDER_RQ_VERTEX;
      if (cs->gen1_native_fragment_rq)
         stages |= HK_RENDER_RQ_FRAGMENT;
   }

   return stages;
}

static unsigned
hk_submit_needs_render_rq(const struct vk_queue_submit *submit)
{
   unsigned stages = 0;

   for (unsigned i = 0; i < submit->command_buffer_count; ++i) {
      stages |= hk_cmd_buffer_needs_render_rq(
         (struct hk_cmd_buffer *)submit->command_buffers[i]);
   }

   return stages;
}

/* Create the kernel queue for the first GPU submission, or replace it with
 * one that reserves render ray-query scratch when the first render using ray
 * queries in a stage the queue does not reserve yet arrives. Only the stages
 * used so far are reserved: on a 40-core G17P both stages together need a
 * contiguous ~359 MiB render free list, which the kernel's lower window may
 * not hold, while one stage needs ~203 MiB. *replaced says whether a kernel
 * queue was replaced: the caller then orders the submission after every
 * earlier one on the queue timeline, and queue-local command ordinals start
 * again.
 */
static void hk_queue_restart_sync(struct hk_queue *queue);

static VkResult
hk_queue_ensure_drm(struct hk_device *dev, struct hk_queue *queue,
                    unsigned render_rq, bool *replaced)
{
   *replaced = false;
   if (!dev->gen1_native_graphics_rq)
      render_rq &= ~HK_RENDER_RQ_VERTEX;
   if (!dev->gen1_native_fragment_rq)
      render_rq &= ~HK_RENDER_RQ_FRAGMENT;
   if (queue->drm.id != UINT32_MAX && !(render_rq & ~queue->drm.render_rq))
      return VK_SUCCESS;
   /* A replacement keeps every stage the queue reserved. */
   render_rq |= queue->drm.render_rq;

   /* Destroy the kernel queue being replaced before creating the new one,
    * once its submitted work is complete. Until then the kernel keeps its
    * compute pool and render free list, and on a 40-core G17P the lower
    * window cannot also hold the new queue's: the submission that needs the
    * new reservation would wait in the kernel for memory only this queue's
    * destruction frees. This waits at most once per stage the queue adds.
    */
   const bool had_queue = queue->drm.id != UINT32_MAX;
   if (had_queue) {
      if (drmSyncobjTimelineWait(dev->dev.fd, &queue->drm.syncobj,
                                 &queue->drm.timeline_value, 1, INT64_MAX,
                                 DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT, NULL))
         return vk_errorf(dev, VK_ERROR_DEVICE_LOST,
                          "Waiting for the replaced kernel queue failed: %s",
                          strerror(errno));
      agx_destroy_command_queue(&dev->dev, queue->drm.id);
      queue->drm.id = UINT32_MAX;
   }

   /* M5 and G17P admit GPU work at the API entrypoint, before semaphore
    * payloads move (hk_queue_admits_at_entry). Other devices retain admission
    * in the serialized driver callback. Host-only synchronization does not
    * need a kernel scheduling context. */
   const uint64_t layouts[2] = {
      (render_rq & HK_RENDER_RQ_VERTEX) ? hk_gen1_ray_query_scs_layout(dev) : 0,
      (render_rq & HK_RENDER_RQ_FRAGMENT) ? hk_gen1_ray_query_scs_layout(dev) : 0};
   const uint32_t id = agx_create_command_queue_with_render_scs(
      &dev->dev, (enum drm_asahi_priority)(queue->drm.virt_ring_idx - 1), layouts);
   if (id != UINT32_MAX) {
      *replaced = had_queue;
      queue->drm.id = id;
      queue->drm.render_rq = render_rq;
      return VK_SUCCESS;
   }

   /* The replaced queue's work is complete, so a later queue starts like a
    * first one. Keep the stages it reserved for that queue. */
   if (had_queue) {
      queue->drm.render_rq = render_rq;
      hk_queue_restart_sync(queue);
   }

   /* For an elevated class the kernel answers EACCES/EPERM when the caller
    * lacks CAP_SYS_NICE or DRM master, and EINVAL when the GPU backend does
    * not schedule that class. Both are a refused global priority, which
    * Vulkan reports as VK_ERROR_NOT_PERMITTED; the application can fall back
    * to MEDIUM. */
   const int error = errno;
   const bool elevated =
      queue->drm.virt_ring_idx - 1 > DRM_ASAHI_PRIORITY_MEDIUM;
   const VkResult result =
      error == ENOMEM ? VK_ERROR_OUT_OF_HOST_MEMORY :
      (error == ENOSPC || error == EBUSY) ? VK_ERROR_OUT_OF_DEVICE_MEMORY :
      (error == EPERM || error == EACCES) ? VK_ERROR_NOT_PERMITTED_KHR :
      (elevated && error == EINVAL) ? VK_ERROR_NOT_PERMITTED_KHR :
      VK_ERROR_INITIALIZATION_FAILED;
   return vk_errorf(dev, result, "DRM_IOCTL_ASAHI_QUEUE_CREATE failed: %s",
                    strerror(error));
}

/* Queue-local ordering state restarts with a new kernel queue. The
 * submission that starts it waits for every earlier submission, so nothing
 * recorded before it can still be pending.
 */
static void
hk_queue_restart_sync(struct hk_queue *queue)
{
   const struct hk_sync_queue old = queue->sync;
   memset(&queue->sync, 0, sizeof(queue->sync));
   queue->sync.exact_prior = old.exact_prior;
   queue->sync.serialize = old.serialize;
   queue->sync.stage_overlap = old.stage_overlap;
   queue->sync.precise_tile_evictions = old.precise_tile_evictions;
   queue->sync.direct_single_footprints = old.direct_single_footprints;
   /* The submission that starts it waits for all sibling work too. */
   memset(queue->sib.submitted[0], 0, sizeof(queue->sib.submitted[0]));
   queue->sib.used = false;
}

/* Kernel queues come from a machine-wide pool shared by every client, so
 * creating one can be refused while other clients hold the pool.
 */
static bool
hk_queue_admits_at_entry(const struct hk_device *dev)
{
   return agx_chip_g17_m5(dev->dev.chip) || dev->dev.chip == AGX_CHIP_G17P;
}

/* Admission must precede the common runtime's semaphore payload moves and
 * the first submission of a batch. Failing inside driver_submit is too late
 * to return a retryable allocation error. Keep admission lazy so unused Vulkan
 * queues do not consume the device-wide pool of kernel contexts.
 */
static VkResult
hk_queue_admit(struct hk_queue *queue, unsigned render_rq)
{
   /* Admission and callbacks must use the same backing queue and lock. */
   if (queue->vk.real_queue)
      queue = container_of(queue->vk.real_queue, struct hk_queue, vk);
   simple_mtx_lock(&queue->vk.lock);
   /* Admission only creates a missing kernel queue, with the render ray-query
    * scratch the call needs (G17P), so a first submission that needs it does
    * not create a queue only to replace it. Replacing an existing queue
    * stays in the driver callback, which orders the new queue after the old
    * one's work. M5 renders never reserve ray-query scratch.
    */
   bool replaced;
   VkResult result = hk_queue_ensure_drm(
      hk_queue_device(queue), queue,
      queue->drm.id == UINT32_MAX ? render_rq : 0, &replaced);
   assert(!replaced);
   /* Only allocation errors are retryable here. Keep the existing lost-device
    * behavior for other failures; device-creation errors such as NOT_PERMITTED
    * and INITIALIZATION_FAILED are not valid QueueSubmit return values.
    */
   if (result != VK_SUCCESS && result != VK_ERROR_OUT_OF_HOST_MEMORY &&
       result != VK_ERROR_OUT_OF_DEVICE_MEMORY)
      result = vk_queue_set_lost(&queue->vk, "Kernel queue admission failed");
   simple_mtx_unlock(&queue->vk.lock);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_QueueSubmit2(VkQueue _queue, uint32_t submitCount,
                const VkSubmitInfo2 *pSubmits, VkFence fence)
{
   VK_FROM_HANDLE(vk_queue, vk_queue, _queue);
   struct hk_queue *queue = container_of(vk_queue, struct hk_queue, vk);
   struct hk_device *dev = hk_queue_device(queue);

   if (hk_queue_admits_at_entry(dev)) {
      if (vk_device_is_lost(&dev->vk))
         return VK_ERROR_DEVICE_LOST;

      const bool rq_capable =
         dev->gen1_native_graphics_rq || dev->gen1_native_fragment_rq;
      bool needs_queue = false;
      unsigned render_rq = 0;
      for (uint32_t i = 0; i < submitCount; i++) {
         for (uint32_t j = 0; j < pSubmits[i].commandBufferInfoCount; j++) {
            VK_FROM_HANDLE(hk_cmd_buffer, cmd,
                           pSubmits[i].pCommandBufferInfos[j].commandBuffer);
            if (list_is_empty(&cmd->control_streams))
               continue;
            needs_queue = true;
            if (rq_capable)
               render_rq |= hk_cmd_buffer_needs_render_rq(cmd);
         }
      }
      if (needs_queue) {
         VkResult result = hk_queue_admit(queue, render_rq);
         if (result != VK_SUCCESS)
            return result;
      }
   }

   return vk_common_QueueSubmit2(_queue, submitCount, pSubmits, fence);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_QueueBindSparse(VkQueue _queue, uint32_t bindInfoCount,
                   const VkBindSparseInfo *pBindInfo, VkFence fence)
{
   VK_FROM_HANDLE(vk_queue, vk_queue, _queue);
   struct hk_queue *queue = container_of(vk_queue, struct hk_queue, vk);
   struct hk_device *dev = hk_queue_device(queue);

   if (hk_queue_admits_at_entry(dev)) {
      if (vk_device_is_lost(&dev->vk))
         return VK_ERROR_DEVICE_LOST;

      /* Sparse image changes may require a GPU maintenance command. Reserve
       * its queue before any bind or semaphore operation in this API call.
       */
      for (uint32_t i = 0; i < bindInfoCount; i++) {
         if (pBindInfo[i].imageOpaqueBindCount || pBindInfo[i].imageBindCount) {
            VkResult result = hk_queue_admit(queue, 0);
            if (result != VK_SUCCESS)
               return result;
            break;
         }
      }
   }

   return vk_common_QueueBindSparse(_queue, bindInfoCount, pBindInfo, fence);
}

/* Sibling kernel queues (hk_queue::sib) on G17P: kernel queues per VkQueue,
 * the primary included. macOS spreads one Metal queue over up to three
 * firmware compute queues of one context and runs two kicks at once
 * (d33c6d37328). With the two gates below, three cut M4 Pro (J616s) ALU
 * depth 4/16 from 285.7/283.5 to 275.0/274.7 us per submission
 * (f22f0908629), and the A18 Pro's bandwidth depth 4/16 by 25/13 %; neither
 * part loses at depth 1-2 or with several processes. */
#define HK_G17P_SIBLING_QUEUES 3
static_assert(HK_G17P_SIBLING_QUEUES <= HK_QUEUE_SIBLINGS_MAX,
              "sibling kernel queue IDs do not fit hk_queue::sib.id");

/* Earlier submissions still in flight before a sibling takes one
 * (hk_sibling_deep). At depth 2 both kicks run at half speed and finish
 * together, so the GPU idles for the completion-to-resubmit round trip:
 * ALU depth 2 went from 286 to 340 us with siblings and no such gate. */
#define HK_G17P_SIBLING_MIN_IN_FLIGHT 2

/* Fewest application dispatch threads a stream needs to go to a sibling
 * (hk_sibling_eligible). A 32-thread stream at depth 4 ran 13% slower on
 * siblings, the 2.1 GFLOP ALU stream faster; 65536 lies between them but
 * was not swept, so the crossover is not known more closely. */
#define HK_G17P_SIBLING_MIN_THREADS 65536

/* A submission a sibling kernel queue may run (hk_queue::sib): its streams
 * dispatch at least HK_G17P_SIBLING_MIN_THREADS threads each (or only
 * internal work), it depends on no earlier submission (every stream's prior
 * is zero or names a command of this submission, see resolve_one) and holds
 * only plain compute streams, without query access, timestamps, scratch or
 * ray queries.
 * Anything that needs the primary kernel queue's own state stays there. */
static bool
hk_sibling_eligible(struct hk_device *dev, const struct hk_queue *queue,
                    const struct vk_queue_submit *submit,
                    const struct hk_sync_resolved *resolved,
                    const uint64_t base[HK_SYNC_NUM_SQ], unsigned command_count,
                    bool kernel_cumulative)
{
   if (submit->buffer_bind_count || submit->image_bind_count ||
       submit->image_opaque_bind_count ||
       command_count > max_commands_per_submit(dev) ||
       (submit->signal_count && !kernel_cumulative) ||
       (dev->dev.debug & (AGX_DBG_SYNC | AGX_DBG_TRACE)))
      return false;

   for (unsigned i = 0; i < submit->command_buffer_count; ++i) {
      struct hk_cmd_buffer *cmdbuf =
         (struct hk_cmd_buffer *)submit->command_buffers[i];

      list_for_each_entry(struct hk_cs, cs, &cmdbuf->control_streams, node) {
         /* Overlapping kicks of a few threads costs more in ordering than it
          * gains (a 32-thread dispatch stream at depth 4 ran 13% slower). */
         if (cs->sib_threads && !cs->sib_indirect &&
             cs->sib_threads < HK_G17P_SIBLING_MIN_THREADS)
            return false;
         if (cs->type != HK_CS_CDM || cs->query_access ||
             cs->timestamp.start.handle || cs->timestamp.end.handle ||
             cs->scratch.cs.main || cs->scratch.cs.preamble ||
             cs->gen1_native_rq)
            return false;

         /* A zero barrier names every command of earlier ioctls. */
         const struct hk_sync_resolved *r = resolved++;
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
            if ((r->prior[sq] && r->prior[sq] <= base[sq]) ||
                r->barrier[sq] == 0)
               return false;
         }
      }
   }

   return true;
}

/* Whether a sibling kernel queue may take the next eligible submission: at
 * least HK_G17P_SIBLING_MIN_IN_FLIGHT earlier submissions of this queue are
 * still in flight (submitted_point is the queue timeline's last point before
 * this submission). Two kicks run at once on sibling queues, so each takes about
 * twice as long; with only one other submission in flight both then finish
 * together and the GPU idles for the whole completion-to-resubmit round trip
 * (a depth-2 compute stream ran 19% slower with siblings on the M4 Pro).
 * With work queued behind them, the overlap fills the launch and drain gaps
 * between kicks instead. */
static bool
hk_sibling_deep(struct hk_device *dev, struct hk_queue *queue,
                uint64_t submitted_point)
{
   uint64_t done = 0;

   if (drmSyncobjQuery(dev->dev.fd, &queue->drm.syncobj, &done, 1))
      return false;
   return submitted_point >= done + HK_G17P_SIBLING_MIN_IN_FLIGHT;
}

/* A stream's prior in the ordinals of the kernel queue it is sent to: a
 * command of this submission keeps its offset from the submission's first
 * ordinal; an earlier submission's command becomes every earlier command of
 * that kernel queue, and the submission also waits for the siblings' work
 * (*earlier). */
static inline uint64_t
hk_sibling_prior(uint64_t prior, uint64_t base, uint64_t kq_base, bool *earlier)
{
   if (prior == 0)
      return 0;
   if (prior > base)
      return kq_base + (prior - base);
   *earlier = true;
   return kq_base;
}

/* Submits the recordings of `submit`. With submit packing these are the
 * submission's packets and the command buffers it left alone; `original`
 * is the application's submission, and `packed_bo` holds the packets'
 * streams (hk_pack_runs()). *untouched is set when a host allocation failed
 * before any queue or synchronization state changed: the submission can
 * fail with that error and the queue stays usable. */
static VkResult
queue_submit_recordings(struct hk_device *dev, struct hk_queue *queue,
                        struct vk_queue_submit *submit,
                        const struct vk_queue_submit *original,
                        struct agx_bo *packed_bo, bool *untouched)
{
   *untouched = false;
   bool flush_sstt = false;
   const bool bound = submit->buffer_bind_count || submit->image_bind_count ||
                      submit->image_opaque_bind_count;

   /* TODO: Support asynchronous sparse queue? */
   if (bound) {

      VkResult result = hk_queue_submit_bind_sparse_memory(dev, submit);
      if (result != VK_SUCCESS)
         return result;

      flush_sstt = (dev->dev.chip == AGX_CHIP_G17P ||
                    agx_chip_g17_m5(dev->dev.chip)) &&
                   hk_sparse_submission_updates_sstt(submit);
   }

   unsigned command_count = flush_sstt ? 1 : 0;

   /* Gather the number of individual commands to submit up front */
   for (unsigned i = 0; i < submit->command_buffer_count; ++i) {
      struct hk_cmd_buffer *cmdbuf =
         (struct hk_cmd_buffer *)submit->command_buffers[i];

      command_count += list_length(&cmdbuf->control_streams);
   }

   perf_debug_dev(&dev->dev,
                  "Submitting %u control streams (%u command buffers)",
                  command_count, submit->command_buffer_count);

   if (command_count == 0) {
      /* A barrier-only command buffer still has a second scope in later
       * submissions. Unsupported kernels retain their existing path. */
      if (queue->sync.exact_prior &&
          !hk_sync_resolve_submit(
             queue, (struct hk_cmd_buffer *const *)submit->command_buffers,
             submit->command_buffer_count, false, false, 0, NULL))
         return vk_device_set_lost(&dev->vk, "Asahi command ordinal overflow");
      return queue_submit_empty(dev, queue, submit);
   }

   /* Resolved sync fields, one entry per stream occurrence: a
    * simultaneous-use command buffer may occur more than once. A sparse bind
    * that only flushes the sparse table has none, and resizing a never
    * allocated array to zero returns NULL. */
   const unsigned resolved_count = command_count - flush_sstt;
   if (resolved_count &&
       !util_dynarray_resize(&queue->sync_resolved, struct hk_sync_resolved,
                             resolved_count)) {
      *untouched = !bound;
      return vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   /* Native M5 keeps the memory of every submitted recording until the
    * submission retires. Elsewhere the application keeps its recordings
    * alive, and only the packets' buffer needs it: the BO cache would hand
    * it out again without waiting for the GPU. Pin before anything below
    * changes queue or synchronization state, so that a failed allocation
    * leaves nothing to roll back. */
   struct hk_cmd_pending *pending = NULL;
   const bool pin_recordings = agx_chip_g17_m5(dev->dev.chip);
   if ((pin_recordings || packed_bo) && !dev->dev.is_virtio) {
      const struct vk_queue_submit no_recordings = {0};
      unsigned chunks = 1 + (command_count - 1) / max_commands_per_submit(dev);
      VkResult pin_result = hk_cmd_pending_begin(
         dev, pin_recordings ? original : &no_recordings, chunks, packed_bo,
         &pending);
      if (pin_result != VK_SUCCESS) {
         *untouched = !bound;
         return vk_error(queue, pin_result);
      }
   }

   bool replaced;
   VkResult queue_result = hk_queue_ensure_drm(
      dev, queue, hk_submit_needs_render_rq(submit), &replaced);
   if (queue_result != VK_SUCCESS) {
      hk_cmd_pending_end(dev, pending, queue, queue_result);
      return queue_result;
   }

   /* The new kernel queue starts after all work of the one it replaces. */
   const uint64_t replaced_point = queue->drm.timeline_value;
   if (replaced)
      hk_queue_restart_sync(queue);

   const uint64_t prior_base[HK_SYNC_NUM_SQ] = {
      queue->sync.submitted[HK_SYNC_VDM], queue->sync.submitted[HK_SYNC_CDM],
   };

   /* Independent engines and physical render queues may finish
    * out of order. Publish API signals from the cumulative queue point,
    * preserving independent execution while covering earlier work: on G17P
    * in the ioctl when the kernel accepts a cumulative output
    * (DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS), otherwise by a transfer after
    * the ioctl. M5 always transfers, also for its chunked timelines.
    */
   const bool cumulative_signals =
      dev->dev.chip == AGX_CHIP_G17P ||
      (agx_chip_g17_m5(dev->dev.chip) &&
       (dev->dev.params.features & DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES));
   const bool kernel_cumulative = dev->dev.chip == AGX_CHIP_G17P &&
      submit->signal_count > 0 &&
      (dev->dev.params.features & DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS);
   const unsigned kernel_signal_count =
      cumulative_signals && !kernel_cumulative ? 0 : submit->signal_count;
   const bool chunk_timelines = cumulative_signals &&
      (dev->dev.params.features & DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS) &&
      agx_chip_g17_m5(dev->dev.chip) &&
      command_count > max_commands_per_submit(dev);
   const uint64_t point_count = chunk_timelines
      ? 1 + (command_count - 1) / max_commands_per_submit(dev) : 1;
   if (queue->drm.timeline_value > UINT64_MAX - point_count) {
      hk_cmd_pending_end(dev, pending, queue, VK_ERROR_DEVICE_LOST);
      return vk_device_set_lost(&dev->vk, "Asahi queue timeline overflow");
   }
   const uint64_t first_chunk_point = queue->drm.timeline_value + 1;
   queue->drm.timeline_value += point_count;
   unsigned wait_count = 0;
   struct drm_asahi_sync *syncs =
      alloca((submit->wait_count + kernel_signal_count + 3) *
             sizeof(struct drm_asahi_sync));

   if (replaced) {
      syncs[wait_count++] = (struct drm_asahi_sync){
         .sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ,
         .handle = queue->drm.syncobj,
         .timeline_value = replaced_point,
      };
   }

   for (unsigned i = 0; i < submit->wait_count; ++i) {
      /* The kernel rejects the submission if we try to wait on the same
       * timeline semaphore at multiple points.
       *
       * TODO: Can we relax the UAPI?
       *
       * XXX: This is quadratic time.
       */
      bool skip = false;
      if (submit->waits[i].sync->flags & VK_SYNC_IS_TIMELINE) {
         const uint64_t v1 = submit->waits[i].wait_value;
         for (unsigned j = 0; j < submit->wait_count; ++j) {
            const uint64_t v2 = submit->waits[j].wait_value;
            if (i != j && submit->waits[i].sync == submit->waits[j].sync &&
                (v1 < v2 || (v1 == v2 && i < j))) {
               skip = true;
               break;
            }
         }

         if (skip)
            continue;
      }

      asahi_fill_sync(&syncs[wait_count++], submit->waits[i].sync,
                      submit->waits[i].wait_value);
   }

   for (unsigned i = 0; i < kernel_signal_count; ++i) {
      asahi_fill_sync(&syncs[wait_count + i], submit->signals[i].sync,
                      submit->signals[i].signal_value);
   }

   /* Signal progress on the queue itself */
   syncs[wait_count + kernel_signal_count] = (struct drm_asahi_sync){
      .sync_type = kernel_cumulative
         ? DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ_CUMULATIVE
         : DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ,
      .handle = queue->drm.syncobj,
      .timeline_value = queue->drm.timeline_value,
   };

   for (unsigned i = 0; i < original->command_buffer_count; ++i) {
      hk_query_note_submit((struct hk_cmd_buffer *)original->command_buffers[i],
                           queue, queue->drm.timeline_value);
   }

   /* Opt-in frame anatomy profiler (HK_FRAME_ANATOMY): NULL unless enabled
    * and this submit is sampled, so the hooks below cost one NULL test. */
   struct hk_frame_anatomy_submit *anatomy =
      unlikely(dev->frame_anatomy)
         ? hk_frame_anatomy_begin_submit(dev, queue, command_count)
         : NULL;

   /* Anatomy: which waits are still unsignalled at submission; the kernel
    * holds every command of the submission until they signal. */
   if (unlikely(anatomy) && submit->wait_count) {
      unsigned timeline = 0, pending = 0, pending_timeline = 0;
      for (unsigned i = 0; i < submit->wait_count; ++i) {
         struct vk_sync *sync = submit->waits[i].sync;
         if (!vk_sync_type_is_drm_syncobj(sync->type))
            continue;
         uint32_t handle = vk_sync_as_drm_syncobj(sync)->syncobj;
         bool done;
         if (sync->flags & VK_SYNC_IS_TIMELINE) {
            uint64_t point = 0;
            timeline++;
            done = !drmSyncobjQuery(dev->dev.fd, &handle, &point, 1) &&
                   point >= submit->waits[i].wait_value;
         } else {
            done = !drmSyncobjWait(dev->dev.fd, &handle, 1, 0, 0, NULL);
         }
         if (!done) {
            pending++;
            pending_timeline += !!(sync->flags & VK_SYNC_IS_TIMELINE);
         }
      }
      hk_frame_anatomy_note(dev, "submit waits=%u timeline=%u pending=%u "
                            "pending_timeline=%u signals=%u commands=%u",
                            submit->wait_count, timeline, pending,
                            pending_timeline, submit->signal_count,
                            command_count);
   }

   /* Now setup the command structs */
   struct util_dynarray payload = UTIL_DYNARRAY_INIT;

   /* A small submit otherwise grows the heap allocation once for its
    * header and again for the command. Keep those payloads on the stack;
    * util_dynarray moves to the heap if attachments or more commands need
    * additional space. The ioctl consumes the payload before we return.
    */
   util_dynarray_init_from_stack(&payload, alloca(1024), 1024);

   unsigned nr_vdm = 0, nr_cdm = 0;

   if (flush_sstt) {
      assert(queue->g17p_sstt_flush_bo);
      if (agx_chip_g17_m5(dev->dev.chip) &&
          (dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_RESOURCES)) {
         /* This stream flushes metadata without dispatching a shader. Declare
          * its empty resource list, as for other flush-only M5 streams.
          */
         const struct drm_asahi_cmd_header resources = {
            .cmd_type = DRM_ASAHI_SET_COMPUTE_RESOURCES,
            .size = 0,
            .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
            .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
         };
         util_dynarray_append(&payload, resources);
      }
      struct drm_asahi_cmd_header header = agx_cmd_header(true, 0, 0);
      const struct drm_asahi_cmd_compute_with_prior cmd = {
         .legacy = {
            .cdm_ctrl_stream_base = queue->g17p_sstt_flush_bo->va->addr,
            .cdm_ctrl_stream_end =
               queue->g17p_sstt_flush_bo->va->addr +
               AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE +
               AGX3_CDM_STREAM_TERMINATE_SIZE,
         },
      };
      if (queue->sync.exact_prior)
         header.size = sizeof(cmd);
      util_dynarray_append(&payload, header);
      util_dynarray_append_array(&payload, uint8_t, &cmd, header.size);
      nr_cdm++;
   }

   /* Tile hazards against this queue's earlier submissions need the point
    * the queue has completed; fetched only when a partial-tile render pass
    * instance is submitted while such a footprint may still be running. On
    * failure every earlier submission is treated as still running. */
   uint64_t completed = 0;
   if (hk_sync_submit_needs_completed(
          queue, (struct hk_cmd_buffer *const *)submit->command_buffers,
          submit->command_buffer_count)) {
      uint64_t point = 0;
      if (drmSyncobjQuery(dev->dev.fd, &queue->drm.syncobj, &point, 1) == 0)
         completed = point;
   }

   /* Every command's barrier fields come from recorded dependencies. Legacy
    * kernels attach input syncs only to the first command, requiring a chain.
    * A kernel advertising per-command waits needs no such execution edge.
    * A serialized queue without exact priors replaces every field below
    * (hk_drm_serialize_prefix), and keeps no ordinals, so it skips this. */
   const bool resolve = !queue->sync.serialize || queue->sync.exact_prior ||
                        queue->sync.stage_overlap;
   const struct hk_sync_resolved *resolved =
      resolve ? queue->sync_resolved.data : NULL;
   if (resolve && !hk_sync_resolve_submit(queue,
                          (struct hk_cmd_buffer *const *)submit->command_buffers,
                          submit->command_buffer_count, flush_sstt,
                          wait_count > 0 &&
                          !(dev->dev.params.features &
                            DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS),
                          completed, queue->sync_resolved.data)) {
      util_dynarray_fini(&payload);
      if (unlikely(anatomy))
         hk_frame_anatomy_end_submit(anatomy, dev, queue, VK_ERROR_DEVICE_LOST);
      hk_cmd_pending_end(dev, pending, queue, VK_ERROR_DEVICE_LOST);
      return vk_device_set_lost(&dev->vk, "Asahi command ordinal overflow");
   }

   /* Sibling kernel queues (hk_queue::sib): an eligible submission takes the
    * next kernel queue in turn. Once a sibling has run work, the primary's
    * ordinals differ from sync.submitted: priors are then rewritten for the
    * kernel queue they go to (hk_sibling_prior), and a submission that
    * depends on an earlier one also waits for the latest sibling point. */
   unsigned sib_target = 0;
   const bool sib_remap = queue->sib.count > 1 && queue->sib.used;
   bool sib_earlier = false;
   if (queue->sib.count > 1 && !replaced && !flush_sstt && !unlikely(anatomy) &&
       !packed_bo &&
       hk_sibling_eligible(dev, queue, submit, resolved, prior_base,
                           command_count, kernel_cumulative) &&
       hk_sibling_deep(dev, queue, first_chunk_point - 1)) {
      sib_target = queue->sib.next;
      queue->sib.next = (queue->sib.next + 1) % queue->sib.count;
      if (sib_target && queue->sib.id[sib_target] == UINT32_MAX) {
         queue->sib.id[sib_target] = agx_create_command_queue(
            &dev->dev, (enum drm_asahi_priority)(queue->drm.virt_ring_idx - 1));
         if (queue->sib.id[sib_target] == UINT32_MAX) {
            /* Out of kernel queues: keep the ones already created. */
            queue->sib.count = sib_target;
            queue->sib.next = 0;
            sib_target = 0;
         }
      }
   }
   const bool sib_rewrite = sib_remap || sib_target != 0;
   uint64_t kq_base[HK_SYNC_NUM_SQ] = {prior_base[0], prior_base[1]};
   if (sib_rewrite) {
      kq_base[HK_SYNC_VDM] = queue->sib.submitted[sib_target][HK_SYNC_VDM];
      kq_base[HK_SYNC_CDM] = queue->sib.submitted[sib_target][HK_SYNC_CDM];
   }

   for (unsigned i = 0; i < submit->command_buffer_count; ++i) {
      struct hk_cmd_buffer *cmdbuf =
         (struct hk_cmd_buffer *)submit->command_buffers[i];

      if (dev->dev.debug & AGX_DBG_PERF) {
         perf_debug(cmdbuf, "Barrier renders: kept=%" PRIu64 " splits=%" PRIu64,
                    cmdbuf->state.barrier_stats.kept,
                    cmdbuf->state.barrier_stats.splits);
         for (unsigned r = 0; r < HK_BARRIER_SPLIT_COUNT; ++r) {
            perf_debug(cmdbuf, "Barrier split reason %s=%" PRIu64,
                       hk_barrier_split_reason_name(r),
                       cmdbuf->state.barrier_stats.reason[r]);
         }
      }

      list_for_each_entry(struct hk_cs, cs, &cmdbuf->control_streams, node) {
         if (resolved)
            hk_sync_load_resolved(cs, resolved++);
         struct drm_asahi_cmd_header header =
            agx_cmd_header(cs->type == HK_CS_CDM, cs->sync.barrier[HK_SYNC_VDM],
                           cs->sync.barrier[HK_SYNC_CDM]);
         uint64_t command_prior[HK_SYNC_NUM_SQ] = {
            cs->sync.prior[HK_SYNC_VDM], cs->sync.prior[HK_SYNC_CDM],
         };
         if (sib_rewrite) {
            for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
               command_prior[sq] = hk_sibling_prior(
                  command_prior[sq], prior_base[sq], kq_base[sq], &sib_earlier);
               /* A zero barrier names this kernel queue's earlier ioctls. */
               if (cs->sync.barrier[sq] == 0)
                  sib_earlier = true;
            }
         }
         /* The same recording can occur twice in a merged submit (for
          * example synchronization2.basic.event.none_set_reset). Assign
          * classic prefixes here, per emitted occurrence. Storing them in
          * hk_cs makes the later occurrence overwrite the earlier one and
          * creates a self-wait that the kernel rejects with EINVAL. */
         if (queue->sync.serialize &&
             !hk_drm_serialize_prefix(&header, command_prior, nr_vdm, nr_cdm,
                                       prior_base, queue->sync.exact_prior)) {
            util_dynarray_fini(&payload);
            hk_cmd_pending_end(dev, pending, queue, VK_ERROR_DEVICE_LOST);
            return vk_device_set_lost(&dev->vk,
                                      "Asahi serialization ordinal overflow");
         }
         assert(header.vdm_barrier == DRM_ASAHI_BARRIER_NONE ||
                header.vdm_barrier <= nr_vdm);
         assert(header.cdm_barrier == DRM_ASAHI_BARRIER_NONE ||
                header.cdm_barrier <= nr_cdm);

         if (cs->type == HK_CS_CDM) {
            perf_debug(
               cmdbuf,
               "%u: Submitting CDM with %u API calls, %u dispatches, %u flushes, %u merged",
               i, cs->stats.calls, cs->stats.cmds, cs->stats.flushes,
               cs->stats.merged);

            if (cs->stats.query_copy_dispatches ||
                cs->stats.query_reset_dispatches) {
               perf_debug(
                  cmdbuf,
                  "query copy: %" PRIu64 " dispatches, %" PRIu64
                  " queries, %" PRIu64 " counters; reset: %" PRIu64
                  " dispatches, %" PRIu64 " queries, %" PRIu64 " flushes",
                  cs->stats.query_copy_dispatches, cs->stats.query_copy_queries,
                  cs->stats.query_copy_counters, cs->stats.query_reset_dispatches,
                  cs->stats.query_reset_queries, cs->stats.query_reset_flushes);
            }

            assert(cs->stats.cmds > 0 || cs->stats.flushes > 0 ||
                   cs->timestamp.end.handle || cs->timestamp.start.handle);

            asahi_append_compute_attachments(dev, &payload, cs);
            if (agx_chip_g17_m5(dev->dev.chip) &&
                (dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_RESOURCES)) {
               /* The header size is 16 bits; never let it truncate. */
               if (cs->compute_resources.size > UINT16_MAX) {
                  util_dynarray_fini(&payload);
                  if (unlikely(anatomy))
                     hk_frame_anatomy_end_submit(anatomy, dev, queue,
                                                 VK_ERROR_DEVICE_LOST);
                  hk_cmd_pending_end(dev, pending, queue, VK_ERROR_DEVICE_LOST);
                  return vk_device_set_lost(
                     &dev->vk, "M5 compute resource list too large (%u bytes)",
                     cs->compute_resources.size);
               }
               const struct drm_asahi_cmd_header resources = {
                  .cmd_type = DRM_ASAHI_SET_COMPUTE_RESOURCES,
                  .size = cs->compute_resources.size,
                  .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
                  .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
               };
               util_dynarray_append(&payload, resources);
               util_dynarray_append_array(&payload, uint8_t,
                                         cs->compute_resources.data,
                                         cs->compute_resources.size);
            }
            struct drm_asahi_cmd_compute_with_prior cmd = {0};
            hk_queue_fill_cdm_command(dev, queue, cs, &cmd.legacy);
            cmd.legacy.flags |= hk_as_build_flags(dev->dev.chip,
                                                  dev->dev.params.features,
                                                  cs->as_build);
            if (unlikely(anatomy))
               hk_frame_anatomy_compute(anatomy, dev, cs, &cmd.legacy, i,
                                        nr_vdm + nr_cdm, payload.size);
            /* Native ray queries exist only on kernels that accept the
             * extended command (hk_g17p_rt_enabled). */
            assert(!cs->gen1_native_rq || dev->gen1_native_rq);
            if (cs->gen1_native_rq && dev->gen1_native_rq) {
               header.size=sizeof(cmd);
               cmd.scs_layout=hk_gen1_ray_query_scs_layout(dev);
               cmd.intersection_pool=dev->gen1_rq_ift->va->addr;
               cmd.intersection_pool_bytes=0x4000;
               cmd.intersection_first=1;cmd.intersection_count=112;
               cmd.intersection_default_program=UINT64_C(0x10010020000);
               cmd.intersection_default_control=UINT64_C(0x16000000000);
            }
            if (queue->sync.exact_prior) {
               header.size = sizeof(cmd);
               cmd.prior_vdm = command_prior[HK_SYNC_VDM];
               cmd.prior_cdm = command_prior[HK_SYNC_CDM];
            }
            if (dev->dev.chip == AGX_CHIP_G18G) {
               /* G18 kernels consume the versioned resource suffix on
                * every compute command (agx_batch.c's extended payload).
                * The G18 kernel never advertises
                * DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS, so
                * queue->sync.exact_prior is always false here and the
                * legacy (non-prior) command carries everything the G18
                * UAPI accepts (w7Q PR-10 finding 4). */
               struct drm_asahi_cmd_compute_g18 g18 = {
                  .base.legacy = cmd.legacy,
                  .resources = cs->g18_compute_resources,
               };
               g18.base.legacy.flags |= DRM_ASAHI_COMPUTE_G18_RESOURCES;
               g18.resources.version = 1;
               struct drm_asahi_cmd_header g18_header = header;
               g18_header.size = sizeof(g18);
               util_dynarray_append(&payload, g18_header);
               util_dynarray_append_array(&payload, uint8_t, &g18,
                                          sizeof(g18));
               nr_cdm++;
               continue;
            }
            util_dynarray_append(&payload, header);
            util_dynarray_append_array(&payload, uint8_t, &cmd, header.size);
            nr_cdm++;
         } else {
            assert(cs->type == HK_CS_VDM);
            perf_debug(cmdbuf,
                       "%u: Submitting VDM with %u API draws, %u draws, "
                       "robust_index_draws=%u robust_index_dispatches=%u "
                       "robust_index_batches=%u",
                       i, cs->stats.calls, cs->stats.cmds,
                       cs->stats.robust_index_draws,
                       cs->stats.robust_index_dispatches,
                       cs->stats.robust_index_batches);
            assert(cs->stats.cmds > 0 || cs->cr.process_empty_tiles ||
                   cs->timestamp.end.handle || cs->timestamp.start.handle);

            asahi_append_fragment_attachments(dev, &payload, cs);
            if (dev->dev.params.features & DRM_ASAHI_FEATURE_RENDER_RESOURCES) {
               /* V2 kernels take 104-byte drm_asahi_render_program_v2 records
                * through command 7; the tile record is widened to match. */
               const bool v2 = dev->dev.params.features &
                               DRM_ASAHI_FEATURE_RENDER_RESOURCES_V2;
               const struct drm_asahi_render_program_v2 tile = {
                  .legacy = cs->tile_resources};
               const size_t tile_bytes =
                  v2 ? sizeof(tile) : sizeof(cs->tile_resources);
               /* The header size is 16 bits; never let it truncate. */
               if (cs->render_resources.size + tile_bytes > UINT16_MAX) {
                  util_dynarray_fini(&payload);
                  if (unlikely(anatomy))
                     hk_frame_anatomy_end_submit(anatomy, dev, queue,
                                                 VK_ERROR_DEVICE_LOST);
                  hk_cmd_pending_end(dev, pending, queue, VK_ERROR_DEVICE_LOST);
                  return vk_device_set_lost(
                     &dev->vk, "M5 render resource list too large (%u bytes)",
                     cs->render_resources.size);
               }
               const struct drm_asahi_cmd_header resources = {
                  .cmd_type = v2 ? DRM_ASAHI_SET_RENDER_RESOURCES_V2
                                 : DRM_ASAHI_SET_RENDER_RESOURCES,
                  .size = cs->render_resources.size + tile_bytes,
                  .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
                  .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
               };
               util_dynarray_append(&payload, resources);
               util_dynarray_append_array(
                  &payload, uint8_t,
                  v2 ? (const void *)&tile : (const void *)&cs->tile_resources,
                  tile_bytes);
               util_dynarray_append_array(&payload, uint8_t,
                                         cs->render_resources.data,
                                         cs->render_resources.size);
            }
            struct drm_asahi_cmd_render_with_prior cmd = {0};
            asahi_fill_vdm_command(dev, cs, &cmd.legacy);
            hk_query_stamps_note_cs(dev, queue, queue->drm.timeline_value, cs);

            /* M5: the vertex half's own render and compute prefixes, when
             * they are narrower than the whole render's, so its tiling can
             * overlap the fragment work of the renders only its fragment
             * half depends on. Supersedes the single-subqueue forms below. */
            const bool stage_prefixes =
               agx_chip_g17_m5(dev->dev.chip) && !queue->sync.serialize &&
               queue->sync.exact_prior &&
               (dev->dev.params.features &
                DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES) &&
               cs->sync.ta_split;
            if (stage_prefixes)
               cmd.legacy.flags |= DRM_ASAHI_RENDER_STAGE_PREFIXES;

            /* G17P: a render whose vertex half depends on less compute work
             * than its fragment half (image-only producers ordered by image
             * barriers, fragment-only destinations) goes as a staged render:
             * the kernel orders its tiling kick after the narrower prefixes
             * alone, so its vertex work overlaps compute work that only its
             * fragment half waits for. Every other render keeps the forms
             * below, which the kernel can publish directly. */
            const bool g17p_staged =
               dev->dev.chip == AGX_CHIP_G17P && !queue->sync.serialize &&
               queue->sync.exact_prior &&
               (dev->dev.params.features &
                DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES_V2) &&
               cs->sync.ta_split &&
               (cs->sync.barrier_ta[HK_SYNC_CDM] !=
                   cs->sync.barrier[HK_SYNC_CDM] ||
                cs->sync.prior_ta[HK_SYNC_CDM] != cs->sync.prior[HK_SYNC_CDM]);
            if (g17p_staged) {
               cmd.legacy.flags |= DRM_ASAHI_RENDER_STAGE_PREFIXES;
               perf_debug_dev(&dev->dev, "G17P staged render");
            }

            /* The render's vertex half depends on no earlier render while its
             * fragment half does (hk_sync: every render dependency targets
             * fragment or attachment work alone): let the kernel order only
             * the fragment half after them, so this render's vertex work can
             * overlap their fragment work. Compute dependencies keep ordering
             * the vertex half either way. */
            if (!stage_prefixes && !g17p_staged && !queue->sync.serialize &&
                (dev->dev.params.features &
                 DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS) &&
                cs->sync.ta_split &&
                cs->sync.barrier_ta[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
                cs->sync.prior_ta[HK_SYNC_VDM] == 0 &&
                (cs->sync.barrier[HK_SYNC_VDM] != DRM_ASAHI_BARRIER_NONE ||
                 cs->sync.prior[HK_SYNC_VDM] != 0))
               cmd.legacy.flags |= DRM_ASAHI_RENDER_FRAGMENT_BARRIERS;

            /* Serialized queue (G13/G14): the header keeps both engines'
             * whole prefixes, but when the recorded dependencies show that
             * the vertex half reads nothing an earlier render produced, the
             * kernel may order only the fragment half after the render
             * prefix. The fragment queue keeps pass order and each
             * fragment half still waits for its own vertex half. */
            if (queue->sync.stage_overlap &&
                (cs->sync.ta_split
                    ? cs->sync.barrier_ta[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
                         cs->sync.prior_ta[HK_SYNC_VDM] == 0
                    : cs->sync.barrier[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
                         cs->sync.prior[HK_SYNC_VDM] == 0))
               cmd.legacy.flags |= DRM_ASAHI_RENDER_FRAGMENT_BARRIERS;

            /* G17P: a vertex half that still names earlier renders can use
             * the fragment-only barriers too when the order of tiling kicks
             * already covers what it names. A staged render takes neither
             * the flag nor part in ta_cover: its tiling waits only for its
             * narrower tiling prefix. */
            if (dev->dev.chip == AGX_CHIP_G17P && !g17p_staged &&
                !queue->sync.serialize && queue->sync.exact_prior &&
                (dev->dev.params.features &
                 DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS)) {
               const uint64_t base = prior_base[HK_SYNC_VDM];
               const uint64_t whole = hk_drm_render_prefix(
                  base, cs->sync.barrier[HK_SYNC_VDM],
                  cs->sync.prior[HK_SYNC_VDM]);
               const uint64_t vertex = hk_drm_render_prefix(
                  base, cs->sync.barrier_ta[HK_SYNC_VDM],
                  cs->sync.prior_ta[HK_SYNC_VDM]);
               if (hk_drm_g17p_tiling_ordered(
                      base, nr_vdm, whole, vertex,
                      cmd.legacy.flags & DRM_ASAHI_RENDER_FRAGMENT_BARRIERS,
                      &queue->sync.ta_cover))
                  cmd.legacy.flags |= DRM_ASAHI_RENDER_FRAGMENT_BARRIERS;
            }

            /* Likewise for compute: when the vertex half depends on less
             * compute work than the fragment half (image-only producers
             * ordered by image barriers, fragment-only destinations), the
             * kernel orders the tiling half after that prefix alone. */
            STATIC_ASSERT(sizeof(struct drm_asahi_cmd_render_with_stage_prior) ==
                          sizeof(struct drm_asahi_cmd_render_with_prior) + 16);
            const bool stage_cdm =
               !stage_prefixes &&
               agx_chip_g17_m5(dev->dev.chip) && queue->sync.exact_prior &&
               (dev->dev.params.features &
                DRM_ASAHI_FEATURE_COMPUTE_STAGE_BARRIERS) &&
               cs->sync.ta_split &&
               (cs->sync.barrier_ta[HK_SYNC_CDM] !=
                   cs->sync.barrier[HK_SYNC_CDM] ||
                cs->sync.prior_ta[HK_SYNC_CDM] != cs->sync.prior[HK_SYNC_CDM]);
            if (stage_cdm)
               cmd.legacy.flags |= DRM_ASAHI_RENDER_COMPUTE_STAGE_BARRIERS;
            if (unlikely(anatomy))
               hk_frame_anatomy_render(anatomy, dev, cs, &cmd.legacy, i,
                                       nr_vdm + nr_cdm, payload.size);
            header.size = hk_drm_render_command_size(
               agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip) ||
                  dev->dev.chip == AGX_CHIP_G18G,
               queue->sync.exact_prior);
            if (queue->sync.exact_prior) {
               cmd.prior_vdm = command_prior[HK_SYNC_VDM];
               cmd.prior_cdm = command_prior[HK_SYNC_CDM];
            }
            STATIC_ASSERT(sizeof(struct drm_asahi_cmd_render_with_ray_query) == 448);
            STATIC_ASSERT(sizeof(struct drm_asahi_cmd_render_staged) == 480);
            if (g17p_staged) {
               struct drm_asahi_cmd_render_staged staged = {
                  .render = {.legacy = cmd},
                  .stage_prefixes = {
                     .ta_cdm_barrier = cs->sync.barrier_ta[HK_SYNC_CDM],
                     .ta_prior_cdm = cs->sync.prior_ta[HK_SYNC_CDM],
                     .ta_vdm_barrier = cs->sync.barrier_ta[HK_SYNC_VDM],
                     .ta_prior_vdm = cs->sync.prior_ta[HK_SYNC_VDM],
                  },
               };
               hk_gen1_render_ray_query_stages(dev, cs, staged.render.ray_query);
               header.size = sizeof(staged);
               util_dynarray_append(&payload, header);
               util_dynarray_append_array(&payload, uint8_t, &staged, header.size);
            } else if (!agx_chip_g17_m5(dev->dev.chip) &&
                (cs->gen1_native_fragment_rq || cs->gen1_native_vertex_rq)) {
               struct drm_asahi_cmd_render_with_ray_query extended = {.legacy = cmd};
               hk_gen1_render_ray_query_stages(dev, cs, extended.ray_query);
               header.size = sizeof(extended);
               util_dynarray_append(&payload, header);
               util_dynarray_append_array(&payload, uint8_t, &extended, header.size);
            } else if (stage_prefixes) {
               const struct drm_asahi_cmd_render_with_stage_prefixes staged = {
                  .legacy = cmd,
                  .ta_cdm_barrier = cs->sync.barrier_ta[HK_SYNC_CDM],
                  .ta_prior_cdm = cs->sync.prior_ta[HK_SYNC_CDM],
                  .ta_vdm_barrier = cs->sync.barrier_ta[HK_SYNC_VDM],
                  .ta_prior_vdm = cs->sync.prior_ta[HK_SYNC_VDM],
               };
               header.size = sizeof(staged);
               util_dynarray_append(&payload, header);
               util_dynarray_append_array(&payload, uint8_t, &staged,
                                          header.size);
            } else if (stage_cdm) {
               const struct drm_asahi_cmd_render_with_stage_prior staged = {
                  .legacy = cmd,
                  .ta_cdm_barrier = cs->sync.barrier_ta[HK_SYNC_CDM],
                  .ta_prior_cdm = cs->sync.prior_ta[HK_SYNC_CDM],
               };
               header.size = sizeof(staged);
               util_dynarray_append(&payload, header);
               util_dynarray_append_array(&payload, uint8_t, &staged,
                                          header.size);
            } else {
               util_dynarray_append(&payload, header);
               util_dynarray_append_array(&payload, uint8_t, &cmd, header.size);
            }
            nr_vdm++;
         }
      }
   }

   if (dev->dev.debug & AGX_DBG_TRACE) {
      agxdecode_drm_cmdbuf(dev->dev.agxdecode, &dev->dev.params, &payload,
                           true);

      agxdecode_next_frame();
   }

   /* Work that depends on an earlier submission also waits for the
    * siblings' (only a primary submission can). A replaced kernel queue's
    * first submission already waits for every earlier point. */
   if (sib_remap && sib_earlier && !replaced) {
      assert(sib_target == 0);
      memmove(&syncs[wait_count + 1], &syncs[wait_count],
              (kernel_signal_count + 1) * sizeof(*syncs));
      syncs[wait_count++] = (struct drm_asahi_sync){
         .sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ,
         .handle = queue->drm.syncobj,
         .timeline_value = queue->sib.last_point,
      };
   }

   struct drm_asahi_submit submit_ioctl = {
      .flags = 0,
      .queue_id = sib_target ? queue->sib.id[sib_target] : queue->drm.id,
      .in_sync_count = wait_count,
      .out_sync_count = kernel_signal_count + 1,
      .cmdbuf_size = payload.size,
      .syncs = (uint64_t)(uintptr_t)(syncs),
      .cmdbuf = (uint64_t)(uintptr_t)(payload.data),
   };

   /* Profiler: "edges" timestamp claims happen in the final payload. */
   if (unlikely(anatomy))
      hk_frame_anatomy_prepare(anatomy, dev, payload.data, payload.size);

   VkResult result;
   if (command_count <= max_commands_per_submit(dev)) {
      if (unlikely(anatomy))
         hk_frame_anatomy_note_barriers(anatomy, payload.data, 0, payload.size);
      result =
         queue_submit_single(dev, &submit_ioctl, queue->drm.virt_ring_idx, pending);
   } else {
      result = queue_submit_looped(dev, queue, &submit_ioctl, command_count,
                                   queue->drm.virt_ring_idx, kq_base,
                                   chunk_timelines, first_chunk_point, anatomy,
                                   pending);
   }
   hk_cmd_pending_end(dev, pending, queue, result);
   if (queue->sib.count > 1 && result == VK_SUCCESS) {
      queue->sib.submitted[sib_target][HK_SYNC_VDM] += nr_vdm;
      queue->sib.submitted[sib_target][HK_SYNC_CDM] += nr_cdm;
      if (sib_target) {
         queue->sib.used = true;
         queue->sib.last_point = queue->drm.timeline_value;
      }
   }
   if (cumulative_signals && !kernel_cumulative) {
      /* The kernel can publish an error fence even when submission fails.
       * Forward it when present, without waiting for a missing queue point
       * or replacing the original submit error with a transfer error. */
      const VkResult signal_result = queue_signal_timeline(dev, queue, submit);
      if (result == VK_SUCCESS)
         result = signal_result;
   }

   util_dynarray_fini(&payload);

   if (unlikely(anatomy))
      hk_frame_anatomy_end_submit(anatomy, dev, queue, result);

   return result;
}

/* Submit packing.
 *
 * Every firmware compute command costs a fixed setup and completion on top
 * of its dispatches. On the M5 Pro (G17S) a chain of 242 one-dispatch
 * command buffers took 3.0 ms as separate commands and 0.31 ms as one; the
 * M4 (G16G) kernel took about 0.3 ms of turnaround per command before it
 * kept several of a VM in flight. Applications that record a dispatch or a
 * few per command buffer and submit many at once (ML graphs, image filters)
 * pay that cost per command buffer. A run of consecutive command buffers
 * that each record one plain compute stream is therefore submitted as one
 * command: their streams are copied back to back into a submission-owned
 * buffer.
 *
 * A member boundary publishes the member's writes as the end of its own
 * command would have published them for the members after it
 * (hk_pack_boundary()):
 *
 * - Where the kernel takes a stream's own completion flush
 *   (DRM_ASAHI_FEATURE_COMPUTE_VISIBLE_COMPLETION, M5), every such stream
 *   ends with a strong flush: the memory-scoped one, or, where memory has
 *   no second address, the application's barrier or the deferred clears'
 *   flush without memory scope (hk_pack_run_member()). A boundary keeps the
 *   strong flush and drops its memory scope, which only memory reached
 *   through a second address needs (hk_cdm_alias_scope()); while such
 *   memory may exist, it keeps or gets the scope. The packet's end gets the
 *   whole flush, upgraded in place where the last member's lacked the scope.
 * - HAL200 streams whose strong flush has no memory-scoped form (G16) end
 *   with their last launch's ordering barrier or a strong flush, and the end
 *   of the command publishes their writes. At a boundary the record becomes
 *   the strong flush, as for a compute or transfer dependency inside one
 *   stream. The last member's tail stays as recorded; the packet's own end
 *   of command follows it.
 *
 * Work in a packet waits for whatever any member waited for, and work after
 * it waits for the whole packet. On the compute subqueue that adds no
 * ordering (M5 runs one queue's compute commands in order, the G16 kernel
 * every compute command of a VM in order on one firmware queue); a member
 * that would add a render wait or synchronization kinds starts a new packet
 * instead (hk_pack_plan()). Streams whose bytes depend on where they were
 * recorded (links, calls, immediate writes) or that carry queries,
 * timestamps, events, indirect or internal launches stay separate, and so do
 * streams with global atomics where another kernel queue could run beside the
 * packet (hk_pack_atomics_allowed()). A packet stays within the launch,
 * thread and static work bounds of hk_pack.h and holds at most one stream
 * with a looping program, so at ALU rate it runs about as long as its longest
 * member could alone (hk_pack.h says how far that holds for memory-bound
 * programs); it carries no more compute attachments than the kernel takes for
 * one command. A run past a bound starts a new packet.
 *
 * It is on wherever a boundary is defined (hk_pack_boundary()), which the
 * stream format and the kernel decide. HK_PACK_CDM_SUBMIT=0/1 (or the older
 * ASAHI_M5_PACK_CDM_SUBMIT) turns it off or on there, for A/B rows.
 */
DEBUG_GET_ONCE_OPTION(hk_pack_cdm_submit_opt, "HK_PACK_CDM_SUBMIT", NULL)
DEBUG_GET_ONCE_OPTION(m5_pack_cdm_submit_opt, "ASAHI_M5_PACK_CDM_SUBMIT", NULL)

static bool
hk_pack_trace(void)
{
   static int trace = -1;
   if (trace < 0)
      trace = debug_get_bool_option("HK_PACK_CDM_TRACE", false) ||
              debug_get_bool_option("ASAHI_M5_PACK_CDM_TRACE", false);
   return trace;
}

static bool
hk_pack_runs_enabled(const struct hk_device *dev)
{
   if (dev->dev.is_virtio || (dev->dev.debug & AGX_DBG_NOMERGE) ||
       hk_pack_boundary(dev) == HK_PACK_BOUNDARY_NONE)
      return false;

   const char *opt = debug_get_option_hk_pack_cdm_submit_opt();
   if (!opt)
      opt = debug_get_option_m5_pack_cdm_submit_opt();
   return opt ? debug_parse_bool_option(opt, false) : true;
}

/* The compute attachments one packet may carry: those the kernel takes for
 * one command, so packing never drops one that a separate command would
 * have carried (asahi_append_compute_attachments()). */
static uint32_t
hk_pack_attachment_limit(const struct hk_device *dev)
{
   if ((!agx_native_graphics(&dev->dev) && dev->dev.chip != AGX_CHIP_G18G) ||
       !dev->dev.params.max_attachments)
      return UINT32_MAX;
   return dev->dev.params.max_attachments;
}

/* Streams with global atomics are packed only where no other kernel queue
 * of the device can run beside the packet. On M5, two kernel queues
 * running packed atomic chains on the same memory lost contributions (root
 * cause unknown), so such streams stay separate commands unless
 * ASAHI_M5_PACK_ATOMICS=1 (or HK_PACK_ATOMICS=1) and the device has a
 * single kernel queue, which runs its commands in order. A second kernel
 * queue created while a packet runs is not covered; unpacked atomic work
 * has the same exposure. The G16 kernel puts every compute command of a VM
 * on its one firmware compute queue and never runs one beside another VM's:
 * several commands of one VM may be in flight there, but the firmware runs
 * them one after another, never beside a packet. G16G packs them unless
 * HK_PACK_ATOMICS=0. That is a property of the G16 kernel, the only one for
 * G16G, which no uAPI feature reports yet; the chip stands for it until one
 * does (hk_pack_atomics_choice()). */
DEBUG_GET_ONCE_OPTION(hk_pack_atomics_opt, "HK_PACK_ATOMICS", NULL)
DEBUG_GET_ONCE_OPTION(m5_pack_atomics_opt, "ASAHI_M5_PACK_ATOMICS", NULL)

static bool
hk_pack_atomics_allowed(struct hk_device *dev, const struct hk_queue *queue)
{
   const char *opt = debug_get_option_hk_pack_atomics_opt();
   if (!opt)
      opt = debug_get_option_m5_pack_atomics_opt();

   if (dev->dev.chip == AGX_CHIP_G16G)
      return hk_pack_atomics_choice(true, opt, queue->sib.count);
   if (!hk_pack_atomics_choice(false, opt, queue->sib.count))
      return false;

   vk_foreach_queue(iter, &dev->vk) {
      if (vk_queue_is_emulated(iter))
         continue;
      const struct hk_queue *other = container_of(iter, struct hk_queue, vk);
      if (other != queue && other->drm.id != UINT32_MAX)
         return false;
   }
   const struct hk_queue *as = dev->host_as.queue;
   return !as || as == queue || as->drm.id == UINT32_MAX;
}

/* A submission-owned replacement for a run of command buffers. The
 * existing resolver assigns ordinals to it, so later submissions name its
 * actual kernel command. */
struct hk_packed_submit {
   struct hk_cmd_buffer cmd;
   struct hk_cs cs;
};

struct hk_packed_runs {
   struct agx_bo *bo;
   unsigned packet_count;
   struct hk_packed_submit *packets;
   struct vk_command_buffer **commands;
   unsigned command_count;
};

static void
hk_packed_runs_free(struct hk_device *dev, struct hk_packed_runs *p)
{
   for (unsigned i = 0; i < p->packet_count; ++i)
      util_dynarray_fini(&p->packets[i].cs.compute_resources);
   agx_bo_unreference(&dev->dev, p->bo);
   free(p->packets);
   free(p->commands);
}

/* Keep each distinct resource profile once: the kernel folds the list into
 * one demand with a per-field maximum (M5 only; empty elsewhere). The first
 * `unique` bytes are already distinct; entries after them are new. */
static void
hk_pack_dedup_resources(struct util_dynarray *list, size_t unique)
{
   const size_t entry = sizeof(struct drm_asahi_compute_program);
   if (list->size % entry || unique % entry)
      return;
   uint8_t *data = list->data;
   size_t kept = unique;
   for (size_t at = unique; at < list->size; at += entry) {
      bool seen = false;
      for (size_t j = 0; j < kept && !seen; j += entry)
         seen = !memcmp(data + j, data + at, entry);
      if (!seen) {
         if (kept != at)
            memmove(data + kept, data + at, entry);
         kept += entry;
      }
   }
   list->size = kept;
}

static bool
hk_pack_runs(struct hk_device *dev, const struct hk_queue *queue,
             const struct vk_queue_submit *submit, struct hk_packed_runs *out)
{
   memset(out, 0, sizeof(*out));
   const unsigned n = submit->command_buffer_count;
   if (n < 2 || submit->buffer_bind_count || submit->image_bind_count ||
       submit->image_opaque_bind_count || !hk_pack_runs_enabled(dev))
      return false;

   uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
   if (!agx3_cdm_stream_terminate_pack(dev->dev.chip, terminate))
      return false;

   const bool visible_completion =
      hk_pack_boundary(dev) == HK_PACK_BOUNDARY_VISIBLE_COMPLETION;
   const bool atomics = hk_pack_atomics_allowed(dev, queue);
   const struct hk_pack_limits limits = {
      .max_attachments = hk_pack_attachment_limit(dev),
      .overhead =
         sizeof(terminate) + HK_CS_SPACE_PADDING + AGX_CDM_STREAM_LINK_LENGTH,
   };
   struct hk_pack_member *members = malloc(n * sizeof(*members));
   struct hk_pack_run *runs = malloc(n * sizeof(*runs));
   if (!members || !runs) {
      free(members);
      free(runs);
      return false;
   }

   const unsigned eligible = hk_pack_describe_submit(
      members, dev, &queue->sync,
      (struct hk_cmd_buffer *const *)submit->command_buffers, n, terminate,
      atomics);

   unsigned packed_members;
   size_t total;
   const unsigned run_count =
      hk_pack_plan(members, n, &limits, runs, &packed_members, &total);
   free(members);

   if (!run_count) {
      if (hk_pack_trace())
         fprintf(stderr, "HK_PACK runs=0 eligible=%u of %u strong_tails=%d\n",
                 eligible, n,
                 hk_pack_admits_strong_tail(hk_pack_boundary(dev)));
      free(runs);
      return false;
   }

   out->bo = agx_bo_create(&dev->dev, total, 0, 0, "Packed compute submit");
   uint8_t *map = out->bo ? agx_bo_map(out->bo) : NULL;
   out->packets = calloc(run_count, sizeof(*out->packets));
   out->commands = malloc((n - packed_members + run_count) * sizeof(*out->commands));
   if (!map || !out->packets || !out->commands) {
      free(runs);
      hk_packed_runs_free(dev, out);
      memset(out, 0, sizeof(*out));
      return false;
   }

   unsigned r = 0;
   for (unsigned i = 0; i < n;) {
      if (r == run_count || i != runs[r].first) {
         out->commands[out->command_count++] = submit->command_buffers[i++];
         continue;
      }

      const struct hk_pack_run *pr = &runs[r];
      struct hk_packed_submit *packet = &out->packets[out->packet_count++];
      struct hk_cmd_buffer *cmd = &packet->cmd;
      struct hk_cs *cs = &packet->cs;
      cmd->vk.base.device = &dev->vk;
      list_inithead(&cmd->control_streams);
      list_addtail(&cs->node, &cmd->control_streams);
      cs->cmd = cmd;
      cs->type = HK_CS_CDM;
      cs->start = cs->current = map + pr->offset;
      cs->end = map + pr->offset + pr->bytes;
      cs->addr = out->bo->va->addr + pr->offset;
      cs->root_size = pr->bytes;
      cs->sync.seq = 1;
      cs->visible_completion = visible_completion;
      hk_drm_attachment_set_init(&cs->compute_attachments);
      util_dynarray_init(&cs->compute_resources, NULL);

      for (unsigned m = 0; m < pr->count; ++m) {
         const struct hk_cmd_buffer *source =
            (const void *)submit->command_buffers[pr->first + m];
         cs->sync.kind |= list_first_entry(&source->control_streams,
                                           struct hk_cs, node)->sync.kind;
      }

      uint8_t preceding_kinds = 0;
      for (unsigned m = 0; m < pr->count; ++m) {
         const struct hk_cmd_buffer *source =
            (const void *)submit->command_buffers[pr->first + m];
         const struct hk_cs *item = list_first_entry(&source->control_streams,
                                                     struct hk_cs, node);
         void *item_end = (uint8_t *)item->current - sizeof(terminate);
         const bool last = m + 1 == pr->count;

         /* The reserved range keeps the merge inline: the link path, which
          * allocates from the packet's (absent) pool and would point the
          * packet into the source's stream, must stay unreachable. The
          * sources are only read here; the application keeps them alive
          * until the submission completes. */
         if (!hk_cs_space_fits((uint8_t *)cs->end - (uint8_t *)cs->current,
                               (uint8_t *)item_end - (uint8_t *)item->start,
                               AGX_CDM_STREAM_LINK_LENGTH)) {
            assert(!"packet sized too small");
            free(runs);
            hk_packed_runs_free(dev, out);
            memset(out, 0, sizeof(*out));
            return false;
         }
         const size_t unique = cs->compute_resources.size;
         hk_cs_merge_cdm_until(cs, item, item_end);
         hk_pack_dedup_resources(&cs->compute_resources, unique);

         if (visible_completion) {
            /* Each source ends with a strong flush (hk_pack_run_member()).
             * The packet's end gets the memory scope, upgrading the record
             * in place if the source's lacked it, for the host and other
             * queues, which observe nothing of a packet before its
             * completion. Between members, the next one reads through the
             * same addresses, so the scope is needed only where memory can
             * also be reached through a second address
             * (hk_cdm_alias_scope()); a source without it keeps its plain
             * strong flush there. */
            if (!last && !hk_cdm_alias_scope(dev) &&
                cs->current == cs->memory_flush_end &&
                cs->current == cs->strong_flush_end) {
               uint8_t *word = (uint8_t *)cs->current - AGX3_CDM_BARRIER_SIZE;
               agx3_cmdstream_store_le32(
                  word, agx3_cmdstream_load_le32(word) &
                           ~AGX3_CDM_BARRIER_FLUSH_MEMORY);
               cs->memory_flush_end = NULL;
            }
            hk_cdm_visibility_flush(dev, cs, last || hk_cdm_alias_scope(dev));
         } else if (!last) {
            /* HAL200: the member's ordering barrier becomes the strong
             * flush, or its strong flush stays. */
            hk_cdm_visibility_flush(dev, cs, false);
         }

         hk_pack_fold_member(cmd, cs, source, item, last, &preceding_kinds);
      }

      assert(!cs->stream_linked);
      assert(visible_completion
                ? cs->memory_flush_end == cs->current
                : cs->strong_flush_end == cs->current ||
                     cs->weak_barrier_end == cs->current);
      memcpy(cs->current, terminate, sizeof(terminate));
      cs->current = (uint8_t *)cs->current + sizeof(terminate);
      memset(cs->current, 0, HK_CS_SPACE_PADDING);
      hk_pack_finish_sync(cmd, cs);

      out->commands[out->command_count++] = &cmd->vk;
      i += pr->count;
      r++;
   }

   if (hk_pack_trace())
      fprintf(stderr, "HK_PACK runs=%u members=%u eligible=%u of %u commands=%u "
              "bytes=%zu strong_tails=%d\n", run_count, packed_members,
              eligible, n, out->command_count, total,
              hk_pack_admits_strong_tail(hk_pack_boundary(dev)));
   free(runs);
   return true;
}

static VkResult
queue_submit(struct hk_device *dev, struct hk_queue *queue,
             struct vk_queue_submit *submit, bool *untouched)
{
   /* Packing copies the command buffers' streams: no patch site may be
    * written meanwhile (host_alias.patch_lock). The copies are the only CPU
    * reads of stream words at submission; the GPU reads each word whole. */
   const bool alias_dynamic = dev->host_alias.dynamic;
   if (unlikely(alias_dynamic))
      u_rwlock_rdlock(&dev->host_alias.patch_lock);
   struct hk_packed_runs runs;
   const bool have_runs = hk_pack_runs(dev, queue, submit, &runs);
   if (unlikely(alias_dynamic))
      u_rwlock_rdunlock(&dev->host_alias.patch_lock);

   if (!have_runs)
      return queue_submit_recordings(dev, queue, submit, submit, NULL,
                                     untouched);

   struct vk_queue_submit effective = *submit;
   effective.command_buffer_count = runs.command_count;
   effective.command_buffers = runs.commands;
   VkResult result = queue_submit_recordings(dev, queue, &effective, submit,
                                             runs.bo, untouched);
   hk_packed_runs_free(dev, &runs);
   return result;
}

static bool
hk_submit_has_compute(const struct vk_queue_submit *submit)
{
   for (unsigned i = 0; i < submit->command_buffer_count; ++i) {
      struct hk_cmd_buffer *cmdbuf =
         (struct hk_cmd_buffer *)submit->command_buffers[i];

      list_for_each_entry(struct hk_cs, cs, &cmdbuf->control_streams, node) {
         if (cs->type == HK_CS_CDM)
            return true;
      }
   }

   return false;
}

static VkResult
hk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct hk_queue *queue = container_of(vk_queue, struct hk_queue, vk);
   struct hk_device *dev = hk_queue_device(queue);

   if (vk_device_is_lost_no_report(&dev->vk))
      return VK_ERROR_DEVICE_LOST;

   /* While a resource may alias, the command buffers recorded before it get
    * memory scope at their patch sites (hk_cdm_alias_scope()), before
    * anything reads their streams: submit packing copies them, holding
    * host_alias.patch_lock shared (queue_submit()). */
   if (unlikely(dev->host_alias.dynamic) &&
       p_atomic_read(&dev->alias_live))
      hk_submit_patch_alias_sites(dev, submit->command_buffers,
                                  submit->command_buffer_count);

   /* Look at the command buffers before they are submitted: once the work is
    * with the kernel it can complete, and the application may then free or
    * reset them while this callback is still running on a submit thread. */
   const bool has_compute =
      queue->kick.watched && hk_submit_has_compute(submit);

   bool untouched;
   VkResult result = queue_submit(dev, queue, submit, &untouched);
   if (result != VK_SUCCESS && !untouched)
      result = vk_queue_set_lost(&queue->vk, "Submit failed");
   else if (result == VK_SUCCESS && has_compute)
      hk_compute_kick_note(dev, queue);

   if (dev->dev.debug & AGX_DBG_SYNC) {
      /* Wait for completion */
      int err = drmSyncobjTimelineWait(
         dev->dev.fd, &queue->drm.syncobj, &queue->drm.timeline_value, 1,
         INT64_MAX, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT, NULL);

      if (err) {
         result = vk_queue_set_lost(&queue->vk, "Wait failed");
      } else {
         VkResult res = dev->vk.check_status(&dev->vk);
         if (result == VK_SUCCESS)
            result = res;
      }
   }

   return result;
}

void
hk_device_quiesce_lost_submissions(struct hk_device *dev)
{
   if (!agx_chip_g17_m5(dev->dev.chip) ||
       !vk_device_is_lost_no_report(&dev->vk))
      return;

   /* GPU storage remains pinned separately. This protects host object data
    * and sync implementations referenced by deferred/threaded submissions.
    */
   vk_foreach_queue(queue, &dev->vk)
      vk_queue_wait_lost_submissions(queue);
}

/* A pending host submission also retains raw semaphore/fence sync pointers.
 * Join or cancel that host work before freeing the sync implementation. */
VKAPI_ATTR void VKAPI_CALL
hk_DestroySemaphore(VkDevice _device, VkSemaphore semaphore,
                    const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   if (semaphore != VK_NULL_HANDLE)
      hk_device_quiesce_lost_submissions(dev);
   vk_common_DestroySemaphore(_device, semaphore, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyFence(VkDevice _device, VkFence fence,
                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   if (fence != VK_NULL_HANDLE)
      hk_device_quiesce_lost_submissions(dev);
   vk_common_DestroyFence(_device, fence, pAllocator);
}

static enum drm_asahi_priority
translate_priority(VkQueueGlobalPriorityKHR prio)
{
   switch (prio) {
   case VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR:
      return DRM_ASAHI_PRIORITY_REALTIME;

   case VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR:
      return DRM_ASAHI_PRIORITY_HIGH;

   case VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR:
      return DRM_ASAHI_PRIORITY_MEDIUM;

   case VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR:
      return DRM_ASAHI_PRIORITY_LOW;

   default:
      UNREACHABLE("Invalid VkQueueGlobalPriorityKHR");
   }
}

void
hk_queue_share(struct hk_device *dev, struct hk_queue *queue)
{
   /* Keep independent Vulkan submit workers while bounding scarce kernel
    * contexts per device. Queue families differ in exposed capabilities, but
    * each kernel queue supports both engines. Priority and creation flags
    * must match; the common runtime serializes callbacks on the backing queue.
    */
   if (!agx_chip_g17_m5(dev->dev.chip))
      return;

   vk_foreach_queue(iter, &dev->vk) {
      if (iter == &queue->vk)
         break;
      if (vk_queue_is_emulated(iter))
         continue;
      struct hk_queue *real = container_of(iter, struct hk_queue, vk);
      if (real->drm.virt_ring_idx != queue->drm.virt_ring_idx ||
          real->vk.flags != queue->vk.flags)
         continue;
      queue->vk.driver_submit = NULL;
      vk_queue_set_emulated(&queue->vk, iter);
      break;
   }
}

static VkResult
hk_queue_create(struct hk_device *dev,
                const VkDeviceQueueCreateInfo *pCreateInfo,
                uint32_t index_in_family, bool kick, struct hk_queue **out)
{
   struct hk_physical_device *pdev = hk_device_physical(dev);
   VkResult result;

   assert(pCreateInfo->queueFamilyIndex < pdev->queue_family_count);

   struct hk_queue *queue = vk_zalloc(&dev->vk.alloc, sizeof(struct hk_queue),
                                      8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!queue)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   list_inithead(&queue->cmd_pending);

   const VkDeviceQueueGlobalPriorityCreateInfoKHR *priority_info =
      vk_find_struct_const(pCreateInfo->pNext,
                           DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR);
   const VkQueueGlobalPriorityKHR priority =
      priority_info ? priority_info->globalPriority
                    : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;

   /* The kernel reserves REALTIME on every GPU, so refuse it up front. HIGH
    * goes to the kernel on every chip: whether the GPU backend schedules it
    * and whether the caller may use it (CAP_SYS_NICE or DRM master) are the
    * kernel's decisions, and hk_queue_ensure_drm reports a refusal as
    * VK_ERROR_NOT_PERMITTED.
    */
   enum drm_asahi_priority drm_priority = translate_priority(priority);
   if (drm_priority == DRM_ASAHI_PRIORITY_REALTIME) {
      vk_free(&dev->vk.alloc, queue);
      return VK_ERROR_NOT_PERMITTED_KHR;
   }

   /* On M5, quantize relative priorities to the two unprivileged classes.
    * M5 only: there the Vulkan queues of one priority class share a kernel
    * queue (hk_queue_share), so the class is the only scheduling distinction
    * left between them; this mapping was qualified with the M5 snapshot.
    * Elsewhere each VkQueue owns its kernel queue and relative priorities
    * stay a hint, as the spec allows. An explicit global class takes
    * precedence. No scheduling work is needed during submission.
    */
   if (agx_chip_g17_m5(dev->dev.chip) && !priority_info &&
       pCreateInfo->pQueuePriorities[index_in_family] < 0.5f)
      drm_priority = DRM_ASAHI_PRIORITY_LOW;

   result = vk_queue_init(&queue->vk, &dev->vk, pCreateInfo, index_in_family);
   if (result != VK_SUCCESS) {
      vk_free(&dev->vk.alloc, queue);
      return result;
   }

   /* The kicker owns a kick queue: keep it out of the walks of the
    * application's queues (vkDeviceWaitIdle, lost-device quiescing, queue
    * sharing). Only the kicker thread submits to it. */
   if (kick)
      list_delinit(&queue->vk.link);

   queue->vk.driver_submit = hk_queue_submit;
   queue->sync.precise_tile_evictions = true;
   queue->sync.direct_single_footprints = true;
   queue->sync.serialize = dev->dev.params.gpu_generation >= 13 &&
                           dev->dev.params.gpu_generation <= 14;
   /* A serialized queue orders every command after both engines' complete
    * prefixes, earlier submissions included. The legacy header encodes that
    * exactly (a zero barrier names every earlier command), so exact prior
    * ordinals carry no information there. They only make the kernel resolve
    * a historical dependency per engine for every command, which costs
    * single-submission latency on G13 and G14. Keep upstream's payload. */
   queue->sync.stage_overlap =
      queue->sync.serialize &&
      (dev->dev.params.features & DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS);
   queue->sync.exact_prior =
      !queue->sync.serialize &&
      (dev->dev.params.features & DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS) != 0;

   queue->drm.id = UINT32_MAX;
   for (unsigned i = 0; i < HK_QUEUE_SIBLINGS_MAX; ++i)
      queue->sib.id[i] = UINT32_MAX;
   queue->sib.count = 1;
   if (dev->dev.chip == AGX_CHIP_G17P && !dev->dev.is_virtio &&
       queue->sync.exact_prior &&
       (dev->dev.params.features & DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS) &&
       (dev->dev.params.features & DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS)) {
      /* G17P firmware runs compute kicks of one context from sibling
       * kernel queues at once, so independent compute streams overlap
       * (see HK_G17P_SIBLING_QUEUES). */
      queue->sib.count = HK_G17P_SIBLING_QUEUES;
   }
   queue->drm.render_rq = 0;
   queue->drm.virt_ring_idx = drm_priority + 1;

   if (drmSyncobjCreate(dev->dev.fd, 0, &queue->drm.syncobj)) {
      mesa_loge("drmSyncobjCreate() failed %d\n", errno);
      vk_queue_finish(&queue->vk);
      vk_free(&dev->vk.alloc, queue);

      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "DRM_IOCTL_SYNCOBJ_CREATE failed: %m");
   }

   uint64_t initial_value = 1;
   if (drmSyncobjTimelineSignal(dev->dev.fd, &queue->drm.syncobj,
                                &initial_value, 1)) {
      hk_queue_finish(dev, queue);
      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "DRM_IOCTL_TIMELINE_SYNCOBJ_SIGNAL failed: %m");
   }

   /* Point 1 is already signalled.  Keep the software cursor in lockstep so
    * the first real submit advances to point 2 instead of reusing the
    * completed bootstrap point.  QueueWaitIdle transfers from this cursor;
    * reusing point 1 lets it return before the first submit has completed. */
   queue->drm.timeline_value = initial_value;
   queue->drm.signalled_value = initial_value;

   if ((dev->dev.chip == AGX_CHIP_G17P ||
        agx_chip_g17_m5(dev->dev.chip)) &&
       dev->vk.enabled_features.sparseBinding && !kick) {
      queue->g17p_sstt_flush_bo =
         agx_bo_create(&dev->dev, AIL_PAGESIZE, 0, 0, "SSTT flush stream");
      if (!queue->g17p_sstt_flush_bo) {
         hk_queue_finish(dev, queue);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      uint8_t *stream = agx_bo_map(queue->g17p_sstt_flush_bo);
      if (!stream ||
          !agx3_cdm_sstt_flush_invalidate_pack(dev->dev.chip, stream) ||
          !agx3_cdm_stream_terminate_pack(
             dev->dev.chip,
             stream + AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE)) {
         hk_queue_finish(dev, queue);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
   }

   /* A kick queue needs a kernel queue of its own. */
   if (!kick)
      hk_queue_share(dev, queue);

   /* Permission for an elevated class must be checked during device
    * creation, while the application can still fall back to MEDIUM. Delaying
    * it until submission would consume payloads or turn refusal into device
    * loss. Ordinary queues remain lazy to avoid reserving unused contexts.
    */
   if (!vk_queue_is_emulated(&queue->vk) && !kick &&
       drm_priority == DRM_ASAHI_PRIORITY_HIGH) {
      /* Without the render ray-query reservation; a render that needs it
       * replaces this kernel queue at submission. */
      bool replaced;
      result = hk_queue_ensure_drm(dev, queue, 0, &replaced);
      if (result != VK_SUCCESS) {
         hk_queue_finish(dev, queue);
         return result;
      }
   }

   if (out)
      *out = queue;
   return VK_SUCCESS;
}

VkResult
hk_queue_init(struct hk_device *dev, const VkDeviceQueueCreateInfo *pCreateInfo,
              uint32_t index_in_family)
{
   return hk_queue_create(dev, pCreateInfo, index_in_family, false, NULL);
}

/* A queue of the given priority class that the application cannot get:
 * its index is past any family's queue count, it is not in the device's
 * queue list, and it shares no kernel queue with the application's queues.
 * Its kernel queue is created at its first submission, like theirs. The
 * caller finishes it with hk_queue_finish. */
VkResult
hk_queue_init_kick(struct hk_device *dev, enum drm_asahi_priority priority,
                   struct hk_queue **out)
{
   const VkDeviceQueueGlobalPriorityCreateInfoKHR priority_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
      .globalPriority = priority == DRM_ASAHI_PRIORITY_LOW
                           ? VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR
                        : priority == DRM_ASAHI_PRIORITY_HIGH
                           ? VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR
                           : VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR,
   };
   const uint32_t index = UINT32_MAX - 1;
   const VkDeviceQueueCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .pNext = &priority_info,
      .queueFamilyIndex = 0,
      .queueCount = index + 1,
   };
   return hk_queue_create(dev, &info, index, true, out);
}

void
hk_queue_finish(struct hk_device *dev, struct hk_queue *queue)
{
   /* Stop/join the submit worker before releasing anything it can use. */
   vk_queue_finish(&queue->vk);
   hk_query_stamps_clear_queue(dev, queue);
   hk_cmd_pending_finish_queue(dev, queue);
   drmSyncobjDestroy(dev->dev.fd, queue->drm.syncobj);
   if (queue->drm.id != UINT32_MAX)
      agx_destroy_command_queue(&dev->dev, queue->drm.id);
   for (unsigned i = 1; i < HK_QUEUE_SIBLINGS_MAX; ++i) {
      if (queue->sib.id[i] != UINT32_MAX)
         agx_destroy_command_queue(&dev->dev, queue->sib.id[i]);
   }
   agx_bo_unreference(&dev->dev, queue->g17p_sstt_flush_bo);
   util_dynarray_fini(&queue->sync_resolved);
   vk_free(&dev->vk.alloc, queue);
}

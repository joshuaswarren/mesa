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
#include "agx_chip_caps.h"
#include "agx_device.h"
#include "agx_pack.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "agx_vdm_hal200_bind_tail_core.h"
#include "decode.h"
#include "hk_cmd_buffer.h"
#include "hk_agx3_ioto_program_core.h"
#include "hk_compute_pair.h"
#include "hk_compute_pair_interval.h"
#include "hk_compute_kick.h"
#include "hk_frame_anatomy.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_drm_render_core.h"
#include "hk_drm_prior_core.h"
#include "hk_drm_own_wait.h"
#include "hk_execution_allowance.h"
#include "hk_pack.h"
#include "hk_queue_flush.h"
#include "hk_image.h"
#include "hk_physical_device.h"
#include "hk_query_pool.h"
#include "hk_sparse.h"
#include "vk_enum_to_str.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <xf86drm.h>
#include "util/cache_ops.h"
#include "asahi/lib/agx_m5_va.h"
#include "asahi/lib/agx_vdm_stream_link_core.h"
#include "util/list.h"
#include "util/log.h"
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
#include "vk_fence.h"
#include "vk_semaphore.h"
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

   /* The waits are only in the queue timeline: no kernel job carries them,
    * and the kernel orders this queue's later work after its earlier jobs
    * alone. A wait's second scope includes every later submission of the
    * queue (vkd3d-proton implements ID3D12CommandQueue::Wait as such an
    * empty batch), so later submissions wait for this point themselves
    * (queue_submit_recordings()). */
   if (submit->wait_count)
      queue->drm.empty_wait_point = queue->drm.timeline_value;

   /* The signals also cover these waits, which no ordinal names. */
   hk_gpu_waits_forget_waits(dev, submit->wait_count, submit->waits);
   hk_gpu_waits_forget_signals(dev, submit->signal_count, submit->signals);
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
 * layers or tiles outside active draw scissors eliminated below.
 */
bool
hk_attachmentless_render(const struct hk_cs *cs)
{
   if (cs->tib.sample_size_B != 0 || cs->cr.process_empty_tiles ||
       cs->cr.depth.buffer || cs->cr.depth.meta ||
       cs->cr.stencil.buffer || cs->cr.stencil.meta)
      return false;

   /* The tilebuffer builder normalizes an attachmentless sample count to
    * one. Check attachment formats explicitly; nr_samples is not a tag.
    * This also excludes colour attachments spilled outside the tilebuffer.
    */
   for (unsigned i = 0; i < ARRAY_SIZE(cs->tib.logical_format); ++i) {
      if (cs->tib.logical_format[i] != PIPE_FORMAT_NONE)
         return false;
   }

   return true;
}

void
hk_attachmentless_vdm_bounds(const struct hk_cs *cs, uint32_t *width_px,
                             uint32_t *height_px, uint32_t *layers)
{
   const bool attachmentless = hk_attachmentless_render(cs);
   const bool empty_attachmentless = attachmentless && cs->stats.cmds == 0;
   const bool bounded_attachmentless =
      attachmentless && cs->stats.cmds > 0 &&
      cs->attachmentless_bounded_draws == cs->stats.cmds;
   const bool layer_zero_attachmentless =
      attachmentless && cs->stats.cmds > 0 && !cs->layer_zero_unknown &&
      cs->layer_zero_draws == cs->stats.cmds;

   uint32_t w, h;
   if (empty_attachmentless) {
      w = 1;
      h = 1;
   } else if (bounded_attachmentless) {
      w = MIN2(cs->cr.width, cs->attachmentless_max_x);
      h = MIN2(cs->cr.height, cs->attachmentless_max_y);
   } else {
      w = cs->cr.width;
      h = cs->cr.height;
   }

   /* Draws stay inside the render area and the tiles past its bottom-right
    * corner keep their contents, so do not walk them: user interfaces render
    * small areas of full-screen targets. */
   if (cs->area_x1 && cs->area_y1) {
      w = MIN2(w, cs->area_x1);
      h = MIN2(h, cs->area_y1);
   }

   /* An attachmentless pass takes its size from the render area or draw
    * scissors, which can be empty, and the kernel refuses a zero dimension. */
   *width_px = MAX2(w, 1);
   *height_px = MAX2(h, 1);
   *layers = (empty_attachmentless || layer_zero_attachmentless)
                ? 1 : cs->cr.layers;
}

static bool
hk_render_has_color_attachments(const struct hk_cs *cs)
{
   if (cs->tib.sample_size_B != 0)
      return true;

   for (unsigned i = 0; i < ARRAY_SIZE(cs->tib.logical_format); ++i) {
      if (cs->tib.logical_format[i] != PIPE_FORMAT_NONE)
         return true;
   }

   return false;
}

static void
asahi_fill_vdm_command(struct hk_device *dev, struct hk_cs *cs,
                       struct drm_asahi_cmd_render *c)
{
   memset(c, 0, sizeof(*c));

   c->vdm_ctrl_stream_base =
      hk_g17p_vdm_low_window_enabled(&dev->dev)
         ? hk_g17p_vdm_alias(cs->stream_owner ? cs->stream_owner : cs->cmd,
                             cs->addr)
         : cs->addr;

   agx_pack(&c->ppp_ctrl, CR_PPP_CONTROL, cfg) {
      cfg.enable_w_clamp = true;
      cfg.fixed_point_format = 1;
   }

   uint32_t width_px = 0, height_px = 0, layers = 0;
   hk_attachmentless_vdm_bounds(cs, &width_px, &height_px, &layers);
   /* Render-submit ABIs with AGX_CAP_ZLS_FULL_RENDER_BOUNDS validate
    * depth_dimensions == (width-1)|((height-1)<<16) when a depth or stencil
    * attachment is bound; keep render bounds at least as large as the ZLS
    * surface when the render area is a sub-rectangle. */
   if (agx_chip_has_cap(dev->dev.chip, AGX_CAP_ZLS_FULL_RENDER_BOUNDS) &&
       (cs->cr.depth.buffer || cs->cr.stencil.buffer)) {
      width_px = MAX2(width_px, cs->cr.zls_width);
      height_px = MAX2(height_px, cs->cr.zls_height);
   }
   c->width_px = width_px;
   c->height_px = height_px;

   c->isp_bgobjdepth = cs->cr.isp_bgobjdepth;
   c->isp_bgobjvals = cs->cr.isp_bgobjvals;

   static_assert(sizeof(c->zls_ctrl) == sizeof(cs->cr.zls_control));
   memcpy(&c->zls_ctrl, &cs->cr.zls_control, sizeof(cs->cr.zls_control));

   const bool g18g = dev->dev.chip == AGX_CHIP_G18G;

   if (agx_chip_native_or_m5(dev->dev.chip) ||
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
   if (agx_chip_native_or_m5(dev->dev.chip) &&
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

   /* On M5 (G17S), Metal sets ZLS_CONTROL bit 0 (0x00080045 for D32F,
    * 0x04080045 for D16 in isolated macOS-HV render.shadow trace t4) on
    * single-sample depth-only passes with no colour or stencil attachments
    * (tib=0B). Setting bit 0 on colour-attached passes suppresses colour
    * output, and trace t4 only covered depth-only passes without stencil, so
    * restrict it to colour-free, stencil-free single-sample depth passes
    * without shader memory writes. */
   if (agx_chip_g17_m5(dev->dev.chip) && c->samples == 1 &&
       cs->cr.depth.buffer != 0 && !cs->cr.stencil.buffer &&
       !cs->m5_shader_writes_memory &&
       !hk_render_has_color_attachments(cs)) {
      agx_unpack(NULL, &c->zls_ctrl, ZLS_CONTROL, zls);
      if (zls.z_load || zls.z_store) {
         zls.unknown_0 = true;
         agx_pack(&c->zls_ctrl, ZLS_CONTROL, cfg) {
            cfg = zls;
         }
      }
   }
   c->layers = layers;

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

   if (cs->timestamp.vtx_end.handle) {
      c->ts_vtx.end.handle = cs->timestamp.vtx_end.handle;
      c->ts_vtx.end.offset = cs->timestamp.vtx_end.offset_B;
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
   if (!agx_chip_native_or_g18g(dev->dev.chip))
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
   if (!agx_chip_native_or_g18g(dev->dev.chip))
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
   unsigned limit = hk_drm_submit_command_limit(
      dev->dev.chip != AGX_CHIP_G13X && !HK_PERF(dev, NOBATCH),
      dev->dev.params.max_commands_per_submission);
   return dev->submit_batch_size ? MIN2(limit, dev->submit_batch_size) : limit;
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
   if (hk_g17p_cached_cmd_memory(&dev->dev))
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
hk_submit_needs_render_rq(const struct hk_device *dev,
                          const struct vk_queue_submit *submit)
{
   if (!dev->gen1_native_graphics_rq && !dev->gen1_native_fragment_rq)
      return 0;

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
      hk_queue_note_signalled(queue, queue->drm.timeline_value);
      agx_destroy_command_queue(&dev->dev, queue->drm.id);
      p_atomic_set(&queue->drm.id, UINT32_MAX);
   }

   /* Native AGX3 and M5 admit GPU work at the API entrypoint, before semaphore
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
      queue->drm.render_rq = render_rq;
      p_atomic_set(&queue->drm.id, id);
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
   queue->gpu_wait_generation++;
   queue->sync.exact_prior = old.exact_prior;
   queue->sync.serialize = old.serialize;
   queue->sync.stage_overlap = old.stage_overlap;
   queue->sync.precise_tile_evictions = old.precise_tile_evictions;
   queue->sync.direct_single_footprints = old.direct_single_footprints;
   queue->sync.load_hazard_frag = old.load_hazard_frag;
   /* The submission that starts it waits for all sibling work too. */
   memset(queue->sib.submitted[0], 0, sizeof(queue->sib.submitted[0]));
   queue->sib.used = false;
   queue->sib.packed_atomic_point = 0;
}

/* Kernel queues come from a machine-wide pool shared by every client, so
 * creating one can be refused while other clients hold the pool.
 */
static bool
hk_queue_admits_at_entry(const struct hk_device *dev)
{
   return agx_chip_native_or_m5(dev->dev.chip);
}

static inline struct hk_queue *
hk_queue_backing(struct hk_queue *queue)
{
   return queue->vk.real_queue
             ? container_of(queue->vk.real_queue, struct hk_queue, vk)
             : queue;
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
   queue = hk_queue_backing(queue);
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

      /* Once the backing kernel queue exists, admission is a no-op: skip the
       * command-buffer scan and queue lock on steady-state submissions. If a
       * G17P ray-query replacement briefly clears drm.id, this falls through
       * to the locked path below. */
      if (p_atomic_read(&hk_queue_backing(queue)->drm.id) != UINT32_MAX)
         return vk_common_QueueSubmit2(_queue, submitCount, pSubmits, fence);

      const bool rq_capable =
         dev->gen1_native_graphics_rq || dev->gen1_native_fragment_rq;
      bool needs_queue = false;
      unsigned render_rq = 0;
      for (uint32_t i = 0; i < submitCount; i++) {
         for (uint32_t j = 0; j < pSubmits[i].commandBufferInfoCount; j++) {
            VK_FROM_HANDLE(hk_cmd_buffer, cmd,
                           pSubmits[i].pCommandBufferInfos[j].commandBuffer);
            /* A command buffer without streams still sends the host flush
             * it deferred (hk_sync_defer_host_flush()). */
            if (list_is_empty(&cmd->control_streams) &&
                !cmd->sync.host_flush_kinds)
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

      if (p_atomic_read(&hk_queue_backing(queue)->drm.id) != UINT32_MAX)
         return vk_common_QueueBindSparse(_queue, bindInfoCount, pBindInfo,
                                          fence);

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

/* Wait directly on the queue's DRM timeline syncobj at its latest submitted
 * point instead of creating a temporary binary syncobj, issuing an empty
 * transfer submit, waiting on the copy, and destroying it. Threaded,
 * deferred, and emulated queues fall back to the common implementation. */
VKAPI_ATTR VkResult VKAPI_CALL
hk_QueueWaitIdle(VkQueue _queue)
{
   VK_FROM_HANDLE(vk_queue, vk_queue, _queue);
   struct hk_queue *queue = container_of(vk_queue, struct hk_queue, vk);
   struct hk_device *dev = hk_queue_device(queue);

   if (vk_device_is_lost(&dev->vk))
      return VK_ERROR_DEVICE_LOST;

   if (queue->vk.submit.mode != VK_QUEUE_SUBMIT_MODE_IMMEDIATE ||
       vk_queue_is_emulated(&queue->vk) || !queue->drm.syncobj)
      return vk_common_QueueWaitIdle(_queue);

   uint64_t target = queue->drm.timeline_value;
   if (target > p_atomic_read(&queue->drm.signalled_value)) {
      int err;
      do {
         err = drmSyncobjTimelineWait(
            dev->dev.fd, &queue->drm.syncobj, &target, 1, INT64_MAX,
            DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT, NULL);
      } while (err == -EINTR || (err < 0 && errno == EINTR));

      if (err)
         return vk_queue_set_lost(
            &queue->vk, "DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT failed: %m");

      hk_queue_note_signalled(queue, target);
   }

   return vk_device_check_status(&dev->vk);
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
 * (hk_sibling_eligible). On the A18 Pro, 65536-thread streams lose on
 * siblings: gpubench compute.uber ran 25% slower (ilp, depth 4 and 16) and
 * 5% slower (divergent) than on the primary kernel queue alone, while the
 * 262144-thread ALU stream breaks even and the 4M-thread bandwidth stream
 * gains. A 32-thread stream at depth 4 also ran 13% slower. Gallium's
 * sibling routing uses the same floor. */
#define HK_G17P_SIBLING_MIN_THREADS 262144

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
/* Per-submission cache of queue->drm.syncobj's observed timeline point so a
 * single submission queries the queue timeline at most once across packing,
 * pending-reap, timestamp/wait hazards, and sibling eligibility checks. */
struct hk_queue_done_cache {
   bool valid;
   bool reaped;
   uint64_t point;
};

static bool
hk_queue_query_done(struct hk_device *dev, struct hk_queue *queue,
                    struct hk_queue_done_cache *cache, uint64_t *done_out)
{
   if (cache && cache->valid) {
      *done_out = MAX2(cache->point, p_atomic_read(&queue->drm.signalled_value));
      return true;
   }

   uint64_t done = 0;
   if (drmSyncobjQuery(dev->dev.fd, &queue->drm.syncobj, &done, 1) != 0)
      return false;

   hk_queue_note_signalled(queue, done);
   if (cache) {
      cache->point = done;
      cache->valid = true;
   }
   *done_out = done;
   return true;
}

/* Query the queue timeline's signalled point, short-circuiting through the
 * cached monotonic lower bound whenever it already covers `target`. */
static bool
hk_queue_query_progress(struct hk_device *dev, struct hk_queue *queue,
                        uint64_t target, struct hk_queue_done_cache *cache,
                        uint64_t *done_out)
{
   uint64_t done = p_atomic_read(&queue->drm.signalled_value);
   if (done >= target) {
      *done_out = done;
      return true;
   }

   return hk_queue_query_done(dev, queue, cache, done_out);
}

static bool
hk_sibling_deep(struct hk_device *dev, struct hk_queue *queue,
                uint64_t submitted_point, struct hk_queue_done_cache *cache)
{
   uint64_t done = p_atomic_read(&queue->drm.signalled_value);
   if (submitted_point <= done ||
       submitted_point - done < HK_G17P_SIBLING_MIN_IN_FLIGHT)
      return false;

   if (!hk_queue_query_done(dev, queue, cache, &done))
      return false;
   return submitted_point > done &&
          submitted_point - done >= HK_G17P_SIBLING_MIN_IN_FLIGHT;
}

/* Whether any sibling kernel queue may still be executing work. */
static bool
hk_sibling_in_flight(struct hk_device *dev, struct hk_queue *queue,
                     struct hk_queue_done_cache *cache)
{
   if (queue->sib.count <= 1 || !queue->sib.used || !queue->sib.last_point)
      return false;

   uint64_t done = 0;
   if (!hk_queue_query_progress(dev, queue, queue->sib.last_point, cache, &done))
      return true;
   return done < queue->sib.last_point;
}

/* Whether a packed submission carrying global atomics is still in flight on
 * the primary kernel queue; sibling routing stays off until it completes so
 * no second kernel queue can run beside a packed atomic chain. */
static bool
hk_sibling_packed_atomic_busy(struct hk_device *dev, struct hk_queue *queue,
                              struct hk_queue_done_cache *cache)
{
   if (!queue->sib.packed_atomic_point)
      return false;

   uint64_t done = 0;
   if (!hk_queue_query_progress(dev, queue, queue->sib.packed_atomic_point,
                                cache, &done))
      return true;
   if (done >= queue->sib.packed_atomic_point) {
      queue->sib.packed_atomic_point = 0;
      return false;
   }
   return true;
}

static struct hk_cs *
hk_submit_unordered_start_ts(const struct vk_queue_submit *submit)
{
   if (!submit->command_buffer_count)
      return NULL;

   struct hk_cmd_buffer *cmd =
      (struct hk_cmd_buffer *)submit->command_buffers[0];
   if (!cmd->sync.has_unordered_start_ts)
      return NULL;

   list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
      if (cs->unordered_start_ts && cs->timestamp.start.handle)
         return cs;
   }

   return NULL;
}

/* Strong AGX3 CDM barrier flags, grounded on every G15--G18 identity
 * (agx3_cmdstream.h); flag-free (0) barriers can hang CDM when chained. */
#define HK_TS_CARRIER_BARRIER_FLAGS 0x5

static VkResult
hk_queue_ensure_ts_carrier(struct hk_device *dev, struct hk_queue *queue)
{
   if (queue->ts_carrier_bo)
      return VK_SUCCESS;

   struct agx_bo *bo =
      agx_bo_create(&dev->dev, AIL_PAGESIZE, 0, 0, "Timestamp carrier");
   if (!bo)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   const struct agx3_cdm_barrier_info barrier = {
      .chip = dev->dev.chip,
      .flags = HK_TS_CARRIER_BARRIER_FLAGS,
   };
   uint8_t record[AGX3_CDM_BARRIER_SIZE];
   if (!agx3_cdm_barrier_pack(&barrier, record) ||
       !hk_queue_flush_stream_write(dev->dev.chip, agx_bo_map(bo), bo->size,
                                    record, sizeof(record))) {
      agx_bo_unreference(&dev->dev, bo);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   queue->ts_carrier_bo = bo;
   return VK_SUCCESS;
}

/* The stream of the compute command that ends a submission with its
 * deferred host availability: the memory-scoped strong flush, encoded as
 * hk_cdm_visibility_flush() records it in a command buffer, and the
 * terminator. */
static VkResult
hk_queue_ensure_host_flush(struct hk_device *dev, struct hk_queue *queue)
{
   if (queue->host_flush_bo)
      return VK_SUCCESS;

   struct agx_bo *bo =
      agx_bo_create(&dev->dev, AIL_PAGESIZE, 0, 0, "Host flush");
   if (!bo)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   uint8_t record[AGX3_CDM_BARRIER_SIZE];
   if (!hk_cdm_pack_visibility_flush(dev, true, record) ||
       !hk_queue_flush_stream_write(dev->dev.chip, agx_bo_map(bo), bo->size,
                                    record, sizeof(record))) {
      agx_bo_unreference(&dev->dev, bo);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   queue->host_flush_bo = bo;
   return VK_SUCCESS;
}

/* Whether the deferred host flush depends on an earlier submission, by the
 * rule hk_sibling_eligible() applies to streams. */
static bool
hk_host_flush_names_earlier(const uint16_t barrier[HK_SYNC_NUM_SQ],
                            const uint64_t prior[HK_SYNC_NUM_SQ],
                            const uint64_t base[HK_SYNC_NUM_SQ])
{
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      if ((prior[sq] && prior[sq] <= base[sq]) || barrier[sq] == 0)
         return true;
   }
   return false;
}

/* Submits the recordings of `submit`. With submit packing these are the
 * submission's packets and the command buffers it left alone; `original`
 * is the application's submission, and `extra_bos` are the buffers the
 * packets read: their streams and the fused compute pairs' code and uniform
 * loaders (hk_pack_runs()). *untouched is set when a host allocation failed
 * before any queue or synchronization state changed: the submission can
 * fail with that error and the queue stays usable. */
static VkResult
queue_submit_recordings(struct hk_device *dev, struct hk_queue *queue,
                        struct vk_queue_submit *submit,
                        const struct vk_queue_submit *original,
                        struct agx_bo *const *extra_bos,
                        unsigned extra_bo_count,
                        struct hk_queue_done_cache *done_cache,
                        bool allow_gpu_waits, bool *untouched)
{
   struct hk_queue_done_cache local_cache = {0};
   if (!done_cache)
      done_cache = &local_cache;
   *untouched = false;
   bool flush_sstt = false;
   const bool bound = submit->buffer_bind_count || submit->image_bind_count ||
                      submit->image_opaque_bind_count;

   /* TODO: Support asynchronous sparse queue? */
   if (bound) {

      VkResult result = hk_queue_submit_bind_sparse_memory(dev, submit);
      if (result != VK_SUCCESS)
         return result;

      flush_sstt = agx_chip_g17p_or_m5(dev->dev.chip) &&
                   hk_sparse_submission_updates_sstt(submit);
   }

   /* Gather recorded control streams, deferred host flushes, query resets, and
    * barrier epochs up front so empty submissions take the fast path without
    * scanning timestamps or ray-query stages. */
   uint32_t sq_count[HK_SYNC_NUM_SQ] = {0, 0};
   bool any_sync_epoch = false;
   bool has_query_reset = false;
   uint8_t host_flush_kinds = 0;
   const bool same_submit = submit == original;
   for (unsigned i = 0; i < submit->command_buffer_count; ++i) {
      struct hk_cmd_buffer *cmdbuf =
         (struct hk_cmd_buffer *)submit->command_buffers[i];

      list_for_each_entry(struct hk_cs, cs, &cmdbuf->control_streams, node)
         sq_count[hk_sync_sq(cs)]++;
      any_sync_epoch |= cmdbuf->sync.epoch != 0;
      if (same_submit) {
         has_query_reset |= cmdbuf->sync.has_gpu_query_reset;
         host_flush_kinds |= cmdbuf->sync.host_flush_kinds;
      }
   }
   if (!same_submit) {
      for (unsigned i = 0; i < original->command_buffer_count; ++i) {
         const struct hk_cmd_buffer *cmdbuf =
            (const struct hk_cmd_buffer *)original->command_buffers[i];
         has_query_reset |= cmdbuf->sync.has_gpu_query_reset;
         host_flush_kinds |= cmdbuf->sync.host_flush_kinds;
      }
   }
   const bool host_flush = host_flush_kinds != 0;
   const unsigned stream_count = sq_count[HK_SYNC_VDM] + sq_count[HK_SYNC_CDM];

   if (!flush_sstt && !host_flush && stream_count == 0) {
      /* A barrier-only command buffer still has a second scope in later
       * submissions, while a barrier-free empty submission (sync.epoch == 0)
       * cannot change queue->sync.tail and skips the resolver completely. */
      if (queue->sync.exact_prior && any_sync_epoch) {
         bool tail_raised = false;
         if (!hk_sync_resolve_submit_ex(
                queue, (struct hk_cmd_buffer *const *)submit->command_buffers,
                submit->command_buffer_count, false, false, 0, NULL,
                &tail_raised))
            return vk_device_set_lost(&dev->vk, "Asahi command ordinal overflow");
         if (tail_raised) {
            uint64_t sq_pt = MAX2(queue->drm.last_sq_point[HK_SYNC_VDM],
                                  queue->drm.last_sq_point[HK_SYNC_CDM]);
            if (queue->sib.used)
               sq_pt = MAX2(sq_pt, queue->sib.last_point);
            queue->drm.last_tail_point = MAX2(queue->drm.last_tail_point, sq_pt);
         }
      }
      return queue_submit_empty(dev, queue, submit);
   }

   struct hk_cs *uts_cs = hk_submit_unordered_start_ts(submit);
   const unsigned render_rq = hk_submit_needs_render_rq(dev, submit);
   /* VDM and CDM subqueues (and sibling kernel queues) can start commands
    * independently. If the opposite subqueue, a sibling kernel queue, an
    * earlier GPU query reset submission, or an earlier submission that raised
    * a queue barrier tail (queue->sync.tail) still has incomplete work on this
    * VkQueue, take the ordered carrier so the leading TOP_OF_PIPE timestamp
    * cannot latch ahead of it.
    *
    * Tracking last_query_reset_point closes the same-subqueue hazard where
    * submit N runs [vkCmdResetQueryPool(Q), HK_SYNC_ALL, compute X] (leaving
    * no new barrier tail after X consumes the barrier) and submit N+1 starts
    * with a leading TOP_OF_PIPE timestamp on Q before N's reset has finished
    * zeroing. Tracking last_tail_point closes barrier-tail ordering without
    * permanently disabling the fast path once the tail's source work has
    * completed (queue->sync.tail holds monotonic ordinals that never reset to
    * 0). Cross-queue resets require a semaphore wait (submit->wait_count > 0)
    * or host completion.
    *
    * Sparse binds and command buffers are never merged into one submit, so
    * uts_cs != NULL implies !flush_sstt (asserted below). */
   bool prior_busy = false;
   if (uts_cs != NULL) {
      const enum hk_sync_sq first_sq = hk_sync_sq(uts_cs);
      const enum hk_sync_sq other_sq =
         first_sq == HK_SYNC_VDM ? HK_SYNC_CDM : HK_SYNC_VDM;
      uint64_t check_point = MAX2(queue->drm.last_sq_point[other_sq],
                                  queue->drm.last_query_reset_point);
      check_point = MAX2(check_point, queue->drm.last_tail_point);
      if (queue->sib.used)
         check_point = MAX2(check_point, queue->sib.last_point);
      if (check_point > 0) {
         uint64_t done = 0;
         const bool ok = hk_queue_query_progress(dev, queue, check_point,
                                                 done_cache, &done);
         prior_busy = !ok || done < check_point;
         if (ok) {
            if (queue->drm.last_sq_point[other_sq] > 0 &&
                done >= queue->drm.last_sq_point[other_sq])
               queue->drm.last_sq_point[other_sq] = 0;
            if (queue->drm.last_query_reset_point > 0 &&
                done >= queue->drm.last_query_reset_point)
               queue->drm.last_query_reset_point = 0;
            if (queue->drm.last_tail_point > 0 &&
                done >= queue->drm.last_tail_point)
               queue->drm.last_tail_point = 0;
         }
      }
   }
   const bool ts_carrier =
      uts_cs != NULL &&
      (submit->wait_count > 0 || prior_busy ||
       queue->drm.empty_wait_point != 0 ||
       (queue->drm.id != UINT32_MAX && (render_rq & ~queue->drm.render_rq)) ||
       queue->sync.serialize || !queue->sync.exact_prior);
   const bool prepend_cdm = flush_sstt || ts_carrier;

   if (prepend_cdm)
      sq_count[HK_SYNC_CDM]++;
   const unsigned command_count =
      sq_count[HK_SYNC_VDM] + sq_count[HK_SYNC_CDM] + (host_flush ? 1 : 0);

   perf_debug_dev(&dev->dev,
                  "Submitting %u control streams (%u command buffers)",
                  command_count, submit->command_buffer_count);

   if (ts_carrier) {
      VkResult carrier_result = hk_queue_ensure_ts_carrier(dev, queue);
      if (carrier_result != VK_SUCCESS) {
         *untouched = !bound;
         return vk_error(queue, carrier_result);
      }
   }

   if (host_flush) {
      VkResult flush_result = hk_queue_ensure_host_flush(dev, queue);
      if (flush_result != VK_SUCCESS) {
         *untouched = !bound;
         return vk_error(queue, flush_result);
      }
   }

   /* Resolved sync fields, one entry per stream occurrence: a
    * simultaneous-use command buffer may occur more than once. A sparse bind
    * that only flushes the sparse table has none, and resizing a never
    * allocated array to zero returns NULL. */
   const unsigned resolved_count = command_count - prepend_cdm - host_flush;
   if (resolved_count &&
       !util_dynarray_resize(&queue->sync_resolved, struct hk_sync_resolved,
                             resolved_count)) {
      *untouched = !bound;
      return vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   /* Driver-owned buffers used by packed submissions have no application handle to
    * keep them alive: the BO cache would hand them out again without waiting
    * for the GPU. Application recordings follow Vulkan lifetime rules while
    * the device is healthy and are retained during device-loss unwinding in
    * hk_free_resettable_cmd_buffer(). Pin before anything below changes queue
    * or synchronization state, so a failed allocation leaves nothing to roll
    * back. */
   struct hk_cmd_pending *pending = NULL;
   if (extra_bo_count && !dev->dev.is_virtio) {
      unsigned chunks = 1 + (command_count - 1) / max_commands_per_submit(dev);
      VkResult pin_result = hk_cmd_pending_begin(
         dev, chunks, extra_bos, extra_bo_count, queue, &done_cache->valid,
         &done_cache->point, done_cache->reaped, &pending);
      done_cache->reaped = true;
      if (pin_result != VK_SUCCESS) {
         *untouched = !bound;
         return vk_error(queue, pin_result);
      }
   }

   bool replaced;
   VkResult queue_result =
      hk_queue_ensure_drm(dev, queue, render_rq, &replaced);
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
    * preserving independent execution while covering earlier work: in the
    * ioctl whenever the kernel accepts a cumulative output
    * (DRM_ASAHI_FEATURE_CUMULATIVE_SIGNALS) and the submission does not split
    * across per-chunk timeline points, otherwise by a transfer after the
    * ioctl.
    */
   const bool cumulative_signals = hk_queue_uses_cumulative_signals(&dev->dev);
   const bool chunk_timelines = cumulative_signals &&
      (dev->dev.params.features & DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS) &&
      agx_chip_has_cap(dev->dev.chip, AGX_CAP_CDM_OVERLAP_SPLIT) &&
      command_count > max_commands_per_submit(dev);
   const bool kernel_cumulative = hk_queue_kernel_cumulative_supported(
      &dev->dev, cumulative_signals, chunk_timelines, submit->signal_count);
   const unsigned kernel_signal_count =
      cumulative_signals && !kernel_cumulative ? 0 : submit->signal_count;
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

   /* Same-queue binary waits become prior floors (hk_gpu_waits.h). Only while
    * every earlier submission stayed on the primary kernel queue with exact
    * ordinals continuing from the signal: not across a kernel queue
    * replacement, not after a sibling kernel queue has run, not on a split
    * submission, and not behind a prepended command (sparse flush or
    * timestamp carrier). */
   const bool gpu_waits_ok = queue->sync.exact_prior && !queue->sib.used;
   const bool gpu_waits =
      gpu_waits_ok && allow_gpu_waits && !replaced && !prepend_cdm;
   uint64_t wait_floor[HK_SYNC_NUM_SQ] = {0};

   /* At most one wait on the queue's own timeline: the kernel refuses two
    * points of one timeline in a submission. own_wait is its index. */
   int own_wait = -1;
   if (replaced) {
      hk_drm_wait_own_timeline(syncs, &wait_count, 0, &own_wait,
                               queue->drm.syncobj, replaced_point);
   }

   /* Waits an earlier empty submission transferred into the queue timeline
    * (queue_submit_empty()) order this submission too, until they are known
    * complete: work sent to a sibling kernel queue or in a later chunk is not
    * ordered after the kernel job that first carried them. */
   if (queue->drm.empty_wait_point) {
      uint64_t done = 0;
      const bool queried =
         hk_queue_query_progress(dev, queue, queue->drm.empty_wait_point,
                                 done_cache, &done);
      if (hk_drm_empty_wait_pending(queue->drm.empty_wait_point, queried,
                                    done)) {
         hk_drm_wait_own_timeline(syncs, &wait_count, 0, &own_wait,
                                  queue->drm.syncobj,
                                  queue->drm.empty_wait_point);
      } else {
         queue->drm.empty_wait_point = 0;
      }
   }

   for (unsigned i = 0; i < submit->wait_count; ++i) {
      /* The kernel rejects the submission if we try to wait on the same
       * timeline semaphore at multiple points.
       *
       * TODO: Can we relax the UAPI?
       *
       * XXX: This is quadratic time.
       */
      if (submit->wait_count > 1 &&
          (submit->waits[i].sync->flags & VK_SYNC_IS_TIMELINE)) {
         bool skip = false;
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
      } else if (hk_gpu_waits_take(dev, gpu_waits ? queue : NULL,
                                   submit->waits[i].sync, wait_floor)) {
         /* Without gpu_waits this only consumes the wait's record. */
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
      hk_queue_append_flush_lists(dev->dev.chip, dev->dev.params.features,
                                  &payload);
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

   /* Tile hazards against this queue's earlier submissions and retired exact
    * prior pruning need the point the queue has completed. Start from the
    * cached monotonic lower bound and refresh via drmSyncobjQuery() when live
    * footprints or unretired exact priors exist. On failure, earlier
    * submissions remain conservatively treated as running. */
   uint64_t completed = p_atomic_read(&queue->drm.signalled_value);
   if (hk_sync_submit_needs_completed(
          queue, (struct hk_cmd_buffer *const *)submit->command_buffers,
          submit->command_buffer_count) ||
       hk_sync_submit_has_unretired_priors(queue)) {
      uint64_t point = 0;
      if (hk_queue_query_done(dev, queue, done_cache, &point))
         completed = MAX2(completed, point);
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
   const bool chain_first =
      wait_count > 0 &&
      !(dev->dev.params.features & DRM_ASAHI_FEATURE_PER_COMMAND_INPUT_SYNCS);
   bool tail_raised = false;
   memcpy(queue->sync.wait_floor, wait_floor, sizeof(wait_floor));
   const bool sync_ok = !resolve || hk_sync_resolve_submit_ex(queue,
                          (struct hk_cmd_buffer *const *)submit->command_buffers,
                          submit->command_buffer_count, prepend_cdm,
                          chain_first, completed, queue->sync_resolved.data,
                          &tail_raised);
   if (!sync_ok) {
      memset(queue->sync.wait_floor, 0, sizeof(queue->sync.wait_floor));
      hk_gpu_waits_forget_signals(dev, submit->signal_count, submit->signals);
      util_dynarray_fini(&payload);
      if (unlikely(anatomy))
         hk_frame_anatomy_end_submit(anatomy, dev, queue, VK_ERROR_DEVICE_LOST);
      hk_cmd_pending_end(dev, pending, queue, VK_ERROR_DEVICE_LOST);
      return vk_device_set_lost(&dev->vk, "Asahi command ordinal overflow");
   }

   /* The deferred host flush follows the first scopes of the dependencies
    * that asked for it, as resolved for the whole submission, and the
    * submission's last command on each subqueue, which covers the first one
    * where that carries the input syncs or a leading flush. */
   uint16_t flush_barrier[HK_SYNC_NUM_SQ] = {DRM_ASAHI_BARRIER_NONE,
                                             DRM_ASAHI_BARRIER_NONE};
   uint64_t flush_prior[HK_SYNC_NUM_SQ] = {0, 0};
   const bool flush_ok = !host_flush ||
      hk_sync_host_flush_prefix(&queue->sync, host_flush_kinds, prior_base,
                                sq_count, flush_barrier, flush_prior);
   memset(queue->sync.wait_floor, 0, sizeof(queue->sync.wait_floor));
   if (!flush_ok) {
      hk_gpu_waits_forget_signals(dev, submit->signal_count, submit->signals);
      util_dynarray_fini(&payload);
      if (unlikely(anatomy))
         hk_frame_anatomy_end_submit(anatomy, dev, queue, VK_ERROR_DEVICE_LOST);
      hk_cmd_pending_end(dev, pending, queue, VK_ERROR_DEVICE_LOST);
      return vk_device_set_lost(&dev->vk, "Asahi command ordinal overflow");
   }

   /* Sibling kernel queues (hk_queue::sib): an eligible submission takes the
    * next kernel queue in turn. Once a sibling has run work, the primary's
    * ordinals differ from sync.submitted: priors are then rewritten for the
    * kernel queue they go to (hk_drm_sibling_prior), and a submission that
    * depends on an earlier one also waits for the latest sibling point. */
   unsigned sib_target = 0;
   const bool sib_remap = queue->sib.count > 1 && queue->sib.used;
   bool sib_earlier = false;
   if (queue->sib.count > 1 && !replaced && !prepend_cdm && !unlikely(anatomy) &&
       !extra_bo_count && !wait_floor[HK_SYNC_VDM] && !wait_floor[HK_SYNC_CDM] &&
       !hk_sibling_packed_atomic_busy(dev, queue, done_cache) &&
       !(host_flush &&
         hk_host_flush_names_earlier(flush_barrier, flush_prior, prior_base)) &&
       hk_sibling_eligible(dev, queue, submit, resolved, prior_base,
                           command_count, kernel_cumulative) &&
       hk_sibling_deep(dev, queue, first_chunk_point - 1, done_cache)) {
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

   if (ts_carrier) {
      assert(queue->ts_carrier_bo);
      assert(!flush_sstt);
      hk_queue_append_flush_lists(dev->dev.chip, dev->dev.params.features,
                                  &payload);
      struct drm_asahi_cmd_header header = agx_cmd_header(
         true,
         queue->sync.exact_prior ? DRM_ASAHI_BARRIER_NONE : 0,
         queue->sync.exact_prior ? DRM_ASAHI_BARRIER_NONE : 0);
      struct drm_asahi_cmd_compute_with_prior cmd = {
         .legacy = {
            .flags = (dev->dev.params.features &
                      DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS)
                        ? DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS
                        : 0,
            .cdm_ctrl_stream_base = queue->ts_carrier_bo->va->addr,
            .cdm_ctrl_stream_end =
               queue->ts_carrier_bo->va->addr +
               AGX3_CDM_BARRIER_SIZE + AGX3_CDM_STREAM_TERMINATE_SIZE,
            .ts.end.handle = uts_cs->timestamp.start.handle,
            .ts.end.offset = uts_cs->timestamp.start.offset_B,
         },
      };
      if (queue->sync.exact_prior) {
         uint64_t prior[HK_SYNC_NUM_SQ] = {
            prior_base[HK_SYNC_VDM],
            prior_base[HK_SYNC_CDM],
         };
         if (sib_rewrite) {
            for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
               prior[sq] = hk_drm_sibling_prior(prior[sq], prior_base[sq],
                                                kq_base[sq], &sib_earlier);
            }
         }
         header.size = sizeof(cmd);
         cmd.prior_vdm = prior[HK_SYNC_VDM];
         cmd.prior_cdm = prior[HK_SYNC_CDM];
      }
      hk_query_stamps_note_req(dev, queue, queue->drm.timeline_value,
                               uts_cs->timestamp.start.handle,
                               uts_cs->timestamp.start.offset_B);
      util_dynarray_append(&payload, header);
      util_dynarray_append_array(&payload, uint8_t, &cmd, header.size);
      nr_cdm++;
   }

   bool packed_atomics = false;
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
         if (extra_bo_count && cs->uses_global_atomics)
            packed_atomics = true;
         if (resolved)
            hk_sync_load_resolved(cs, resolved++);
         struct drm_asahi_cmd_header header =
            agx_cmd_header(cs->type == HK_CS_CDM, cs->sync.barrier[HK_SYNC_VDM],
                           cs->sync.barrier[HK_SYNC_CDM]);
         uint64_t command_prior[HK_SYNC_NUM_SQ] = {
            cs->sync.prior[HK_SYNC_VDM], cs->sync.prior[HK_SYNC_CDM],
         };
         /* The tiling half's prefix of a render that splits it, which the
          * staged forms below send. It counts in the same ordinals as the
          * whole prefix, so it is rewritten for the same kernel queue. */
         uint64_t ta_command_prior[HK_SYNC_NUM_SQ] = {
            cs->sync.prior_ta[HK_SYNC_VDM], cs->sync.prior_ta[HK_SYNC_CDM],
         };
         if (sib_rewrite) {
            hk_drm_sibling_prefix(command_prior, cs->sync.barrier, prior_base,
                                  kq_base, &sib_earlier);
            if (cs->sync.ta_split)
               hk_drm_sibling_prefix(ta_command_prior, cs->sync.barrier_ta,
                                     prior_base, kq_base, &sib_earlier);
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
            if (ts_carrier && cs == uts_cs) {
               cmd.legacy.ts.start.handle = 0;
               cmd.legacy.ts.start.offset = 0;
            }
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
               if (!g18.resources.version)
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
                   cs->timestamp.end.handle || cs->timestamp.start.handle ||
                   cs->timestamp.vtx_end.handle);

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
            struct drm_asahi_cmd_render_with_prior cmd;
            asahi_fill_vdm_command(dev, cs, &cmd.legacy);
            cmd.prior_vdm = 0;
            cmd.prior_cdm = 0;
            if (ts_carrier && cs == uts_cs) {
               cmd.legacy.ts_vtx.start.handle = 0;
               cmd.legacy.ts_vtx.start.offset = 0;
            }
            hk_query_stamps_note_cs(dev, queue, queue->drm.timeline_value, cs);

            /* Prefer DRM_ASAHI_RENDER_FRAGMENT_BARRIERS over staged render
             * prefixes whenever the vertex half needs no render prefix and the
             * compute prefix is unchanged: both describe identical ordering,
             * while FRAGMENT_BARRIERS keeps the smaller render payload and
             * stays eligible for direct kernel publication. */
            const bool m5_frag_barriers_only =
               (dev->dev.params.features &
                DRM_ASAHI_FEATURE_FRAGMENT_BARRIERS) &&
               cs->sync.ta_split &&
               cs->sync.barrier_ta[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE &&
               cs->sync.prior_ta[HK_SYNC_VDM] == 0 &&
               ta_command_prior[HK_SYNC_VDM] == 0 &&
               cs->sync.barrier_ta[HK_SYNC_CDM] == cs->sync.barrier[HK_SYNC_CDM] &&
               cs->sync.prior_ta[HK_SYNC_CDM] == cs->sync.prior[HK_SYNC_CDM] &&
               ta_command_prior[HK_SYNC_CDM] == command_prior[HK_SYNC_CDM] &&
               (cs->sync.barrier[HK_SYNC_VDM] != DRM_ASAHI_BARRIER_NONE ||
                cs->sync.prior[HK_SYNC_VDM] != 0);

            /* M5: the vertex half's own render and compute prefixes, when
             * they are narrower than the whole render's, so its tiling can
             * overlap the fragment work of the renders only its fragment
             * half depends on. Supersedes the single-subqueue forms below. */
            const bool m5_staged_v2 =
               agx_chip_g17_m5(dev->dev.chip) && !queue->sync.serialize &&
               queue->sync.exact_prior && !m5_frag_barriers_only &&
               (dev->dev.params.features &
                DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES_V2) &&
               cs->sync.ta_split;
            const bool stage_prefixes =
               !m5_staged_v2 && !m5_frag_barriers_only &&
               agx_chip_g17_m5(dev->dev.chip) && !queue->sync.serialize &&
               queue->sync.exact_prior &&
               (dev->dev.params.features &
                DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES) &&
               cs->sync.ta_split;
            if (stage_prefixes || m5_staged_v2)
               cmd.legacy.flags |= DRM_ASAHI_RENDER_STAGE_PREFIXES;

            /* Native HAL200: a render whose vertex half depends on less
             * compute work than its fragment half (image-only producers
             * ordered by image barriers, fragment-only destinations) goes as
             * a staged render: the kernel orders its tiling kick after the
             * narrower prefixes alone, so its vertex work overlaps compute
             * work that only its fragment half waits for. Every other render
             * keeps the forms below, which the kernel can publish directly.
             */
            bool g17p_staged =
               agx_chip_native_hal200(dev->dev.chip) &&
               !queue->sync.serialize && queue->sync.exact_prior &&
               (dev->dev.params.features &
                DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES_V2) &&
               cs->sync.ta_split &&
               (cs->sync.barrier_ta[HK_SYNC_CDM] !=
                   cs->sync.barrier[HK_SYNC_CDM] ||
                ta_command_prior[HK_SYNC_CDM] != command_prior[HK_SYNC_CDM]);

            /* The render's vertex half depends on no earlier render while its
             * fragment half does (hk_sync: every render dependency targets
             * fragment or attachment work alone): let the kernel order only
             * the fragment half after them, so this render's vertex work can
             * overlap their fragment work. Compute dependencies keep ordering
             * the vertex half either way. */
            if (!stage_prefixes && !m5_staged_v2 && !g17p_staged && !queue->sync.serialize &&
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

            /* Native HAL200: a vertex half that still names earlier renders
             * can use the fragment-only barriers too when the order of tiling
             * kicks already covers what it names. Otherwise, when the vertex
             * half still depends on an earlier render than the fragment half,
             * send the narrower tiling prefix via a staged render instead of
             * stalling tiling on the whole prefix. A staged render takes
             * neither FRAGMENT_BARRIERS nor part in ta_cover. */
            if (agx_chip_native_hal200(dev->dev.chip) && !g17p_staged &&
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
               uint64_t next_cover = queue->sync.ta_cover;
               if (hk_drm_g17p_tiling_ordered(
                      base, nr_vdm, whole, vertex,
                      cmd.legacy.flags & DRM_ASAHI_RENDER_FRAGMENT_BARRIERS,
                      &next_cover)) {
                  cmd.legacy.flags |= DRM_ASAHI_RENDER_FRAGMENT_BARRIERS;
                  queue->sync.ta_cover = next_cover;
               } else if (agx_chip_has_cap(dev->dev.chip,
                                           AGX_CAP_CDM_OVERLAP_SPLIT) &&
                          (dev->dev.params.features &
                           DRM_ASAHI_FEATURE_RENDER_STAGE_PREFIXES_V2) &&
                          cs->sync.ta_split && vertex < whole) {
                  g17p_staged = true;
               } else {
                  queue->sync.ta_cover = next_cover;
               }
            }
            if (g17p_staged) {
               cmd.legacy.flags |= DRM_ASAHI_RENDER_STAGE_PREFIXES;
               perf_debug_dev(&dev->dev, "HAL200 staged render");
            }

            /* Likewise for compute: when the vertex half depends on less
             * compute work than the fragment half (image-only producers
             * ordered by image barriers, fragment-only destinations), the
             * kernel orders the tiling half after that prefix alone. */
            STATIC_ASSERT(sizeof(struct drm_asahi_cmd_render_with_stage_prior) ==
                          sizeof(struct drm_asahi_cmd_render_with_prior) + 16);
            const bool stage_cdm =
               !stage_prefixes && !m5_staged_v2 &&
               agx_chip_g17_m5(dev->dev.chip) && queue->sync.exact_prior &&
               (dev->dev.params.features &
                DRM_ASAHI_FEATURE_COMPUTE_STAGE_BARRIERS) &&
               cs->sync.ta_split &&
               (cs->sync.barrier_ta[HK_SYNC_CDM] !=
                   cs->sync.barrier[HK_SYNC_CDM] ||
                ta_command_prior[HK_SYNC_CDM] != command_prior[HK_SYNC_CDM]);
            if (stage_cdm)
               cmd.legacy.flags |= DRM_ASAHI_RENDER_COMPUTE_STAGE_BARRIERS;
            if (unlikely(anatomy))
               hk_frame_anatomy_render(anatomy, dev, cs, &cmd.legacy, i,
                                       nr_vdm + nr_cdm, payload.size);
            header.size = hk_drm_render_command_size(
               agx_chip_agx3_path(dev->dev.chip),
               queue->sync.exact_prior);
            if (queue->sync.exact_prior) {
               cmd.prior_vdm = command_prior[HK_SYNC_VDM];
               cmd.prior_cdm = command_prior[HK_SYNC_CDM];
            }
            STATIC_ASSERT(sizeof(struct drm_asahi_cmd_render_with_ray_query) == 448);
            STATIC_ASSERT(sizeof(struct drm_asahi_cmd_render_staged) == 480);
            if (g17p_staged || m5_staged_v2) {
               struct drm_asahi_cmd_render_staged staged = {
                  .render = {.legacy = cmd},
                  .stage_prefixes = {
                     .ta_cdm_barrier = cs->sync.barrier_ta[HK_SYNC_CDM],
                     .ta_prior_cdm = ta_command_prior[HK_SYNC_CDM],
                     .ta_vdm_barrier = cs->sync.barrier_ta[HK_SYNC_VDM],
                     .ta_prior_vdm = ta_command_prior[HK_SYNC_VDM],
                  },
               };
               if (g17p_staged)
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
                  .ta_prior_cdm = ta_command_prior[HK_SYNC_CDM],
                  .ta_vdm_barrier = cs->sync.barrier_ta[HK_SYNC_VDM],
                  .ta_prior_vdm = ta_command_prior[HK_SYNC_VDM],
               };
               header.size = sizeof(staged);
               util_dynarray_append(&payload, header);
               util_dynarray_append_array(&payload, uint8_t, &staged,
                                          header.size);
            } else if (stage_cdm) {
               const struct drm_asahi_cmd_render_with_stage_prior staged = {
                  .legacy = cmd,
                  .ta_cdm_barrier = cs->sync.barrier_ta[HK_SYNC_CDM],
                  .ta_prior_cdm = ta_command_prior[HK_SYNC_CDM],
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

   if (host_flush) {
      assert(queue->host_flush_bo);
      hk_queue_append_flush_lists(dev->dev.chip, dev->dev.params.features,
                                  &payload);
      struct drm_asahi_cmd_header header =
         agx_cmd_header(true, flush_barrier[HK_SYNC_VDM],
                        flush_barrier[HK_SYNC_CDM]);
      /* The stream ends with the memory-scoped strong flush, so where the
       * kernel takes a stream's own completion flush it takes this one, as
       * for the flush streams command buffers end (hk_EndCommandBuffer()). */
      const bool visible_completion = hk_cdm_visible_completion(dev);
      struct drm_asahi_cmd_compute_with_prior cmd = {
         .legacy = {
            .flags = ((dev->dev.params.features &
                       DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS)
                         ? DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS
                         : 0) |
                     (visible_completion ? DRM_ASAHI_COMPUTE_VISIBLE_COMPLETION
                                         : 0),
            .cdm_ctrl_stream_base = queue->host_flush_bo->va->addr,
            .cdm_ctrl_stream_end =
               queue->host_flush_bo->va->addr +
               AGX3_CDM_BARRIER_SIZE + AGX3_CDM_STREAM_TERMINATE_SIZE,
         },
      };
      uint64_t prior[HK_SYNC_NUM_SQ] = {
         flush_prior[HK_SYNC_VDM], flush_prior[HK_SYNC_CDM],
      };
      if (sib_rewrite) {
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
            prior[sq] = hk_drm_sibling_prior(prior[sq], prior_base[sq],
                                             kq_base[sq], &sib_earlier);
            if (flush_barrier[sq] == 0)
               sib_earlier = true;
         }
      }
      if (queue->sync.serialize &&
          !hk_drm_serialize_prefix(&header, prior, nr_vdm, nr_cdm, prior_base,
                                   queue->sync.exact_prior)) {
         util_dynarray_fini(&payload);
         if (unlikely(anatomy))
            hk_frame_anatomy_end_submit(anatomy, dev, queue,
                                        VK_ERROR_DEVICE_LOST);
         hk_cmd_pending_end(dev, pending, queue, VK_ERROR_DEVICE_LOST);
         return vk_device_set_lost(&dev->vk,
                                   "Asahi serialization ordinal overflow");
      }
      assert(header.vdm_barrier == DRM_ASAHI_BARRIER_NONE ||
             header.vdm_barrier <= nr_vdm);
      assert(header.cdm_barrier == DRM_ASAHI_BARRIER_NONE ||
             header.cdm_barrier <= nr_cdm);
      if (queue->sync.exact_prior) {
         header.size = sizeof(cmd);
         cmd.prior_vdm = prior[HK_SYNC_VDM];
         cmd.prior_cdm = prior[HK_SYNC_CDM];
      }
      if (unlikely(anatomy))
         hk_frame_anatomy_note(dev, "host flush kinds=0x%x barrier=%u,%u",
                               host_flush_kinds, header.vdm_barrier,
                               header.cdm_barrier);
      util_dynarray_append(&payload, header);
      util_dynarray_append_array(&payload, uint8_t, &cmd, header.size);
      nr_cdm++;
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
      /* The signals and the queue's own signal follow the waits. */
      hk_drm_wait_own_timeline(syncs, &wait_count, kernel_signal_count + 1,
                               &own_wait, queue->drm.syncobj,
                               queue->sib.last_point);
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

   /* Its binary signals now stand for this queue's ordinals. The record is
    * written before the kernel or queue_signal_timeline can make a payload
    * visible, so no waiter sees the payload without it; a failed submission
    * forgets it again. Submissions of one queue are serialized, so no wait
    * of this queue can take the record in between. */
   hk_gpu_waits_note_signals(dev, queue, gpu_waits_ok && !sib_target,
                             submit->signal_count, submit->signals);

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
   if (result == VK_SUCCESS) {
      if (nr_vdm)
         queue->drm.last_sq_point[HK_SYNC_VDM] = queue->drm.timeline_value;
      if (nr_cdm)
         queue->drm.last_sq_point[HK_SYNC_CDM] = queue->drm.timeline_value;
      if (has_query_reset)
         queue->drm.last_query_reset_point = queue->drm.timeline_value;
      if (tail_raised)
         queue->drm.last_tail_point = queue->drm.timeline_value;
   }
   if (queue->sib.count > 1 && result == VK_SUCCESS) {
      queue->sib.submitted[sib_target][HK_SYNC_VDM] += nr_vdm;
      queue->sib.submitted[sib_target][HK_SYNC_CDM] += nr_cdm;
      if (sib_target) {
         queue->sib.used = true;
         queue->sib.last_point = queue->drm.timeline_value;
         queue->gpu_wait_generation++;
      } else if (packed_atomics) {
         queue->sib.packed_atomic_point = queue->drm.timeline_value;
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

   if (result != VK_SUCCESS)
      hk_gpu_waits_forget_signals(dev, submit->signal_count, submit->signals);

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
 * - HAL200 streams whose caches are qualified (!hk_cdm_caches_pending(),
 *   G16 and G17P) end with their last launch's ordering barrier or a strong
 *   flush, and the end of the command publishes their writes. At a boundary
 *   the record becomes the strong flush, adding memory scope on G17P while
 *   aliased mappings exist (hk_cdm_alias_scope()). The last member's tail
 *   stays as recorded unless alias scope became active after recording on a
 *   part with a memory-scoped flush, in which case it is upgraded in place.
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
   /* Queue submissions on multiple queues may ask first, so keep the cached
    * answer atomic. Racing first calls compute the same value. */
   static int trace = -1;
   int cached = p_atomic_read_relaxed(&trace);
   if (unlikely(cached < 0)) {
      cached = debug_get_bool_option("HK_PACK_CDM_TRACE", false) ||
               debug_get_bool_option("ASAHI_M5_PACK_CDM_TRACE", false);
      p_atomic_set(&trace, cached);
   }
   return cached;
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
   if (!agx_chip_native_or_g18g(dev->dev.chip) ||
       !dev->dev.params.max_attachments)
      return UINT32_MAX;
   return dev->dev.params.max_attachments;
}

/* Streams with global atomics are packed only where no other kernel queue
 * of the device can run beside the packet. On M5, two kernel queues
 * running packed atomic chains on the same memory lost contributions (root
 * cause unknown), so such streams stay separate commands unless
 * ASAHI_M5_PACK_ATOMICS=1 (or HK_PACK_ATOMICS=1) and the device has a
 * single active kernel queue, which runs its commands in order. On G17P,
 * packed atomics are enabled by default (disable with HK_PACK_ATOMICS=0)
 * whenever the single-kernel-queue invariant holds for the lifetime of the
 * device: no sibling kernel queue is in flight (hk_sibling_in_flight()),
 * sibling routing stays off until any packed atomic submission completes
 * (packed_atomic_point), and no other non-emulated VkQueue, host-AS queue,
 * or compute-kick queue exists on the device. Because kernel queues are
 * admitted lazily on first submission, checking only whether another queue
 * already has a kernel queue ID leaves a race where a second queue admits
 * and submits concurrently before an in-flight packed atomic chain finishes
 * (and waiting for packed_atomic_point at admission would deadlock valid
 * wait-before-signal timeline semaphore submits). The G16 kernel puts every
 * compute command of a VM on its one firmware compute queue and never runs
 * one beside another VM's: several commands of one VM may be in flight there,
 * but the firmware runs them one after another, never beside a packet. G16G
 * packs them unless HK_PACK_ATOMICS=0. That is a property of the G16 kernel,
 * the only one for G16G, which no uAPI feature reports yet; the chip stands
 * for it until one does (hk_pack_atomics_choice()). */
DEBUG_GET_ONCE_OPTION(hk_pack_atomics_opt, "HK_PACK_ATOMICS", NULL)
DEBUG_GET_ONCE_BOOL_OPTION(hk_pack_gemm_dw_opt, "HK_PACK_GEMM_DW", true)
DEBUG_GET_ONCE_OPTION(m5_pack_atomics_opt, "ASAHI_M5_PACK_ATOMICS", NULL)

static bool
hk_pack_atomics_allowed_cached(struct hk_device *dev, struct hk_queue *queue,
                               struct hk_queue_done_cache *done_cache)
{
   const char *opt = debug_get_option_hk_pack_atomics_opt();
   if (!opt)
      opt = debug_get_option_m5_pack_atomics_opt();

   const struct hk_queue *as = dev->host_as.queue;
   if (as && as != queue && !vk_queue_is_emulated(&as->vk))
      return false;

   if (dev->dev.chip == AGX_CHIP_G16G)
      return hk_pack_atomics_choice(true, opt, queue->sib.count);

   /* Check the chip default / option before querying the syncobj for
    * sibling activity so disabled callers never pay a drmSyncobjQuery ioctl. */
   const bool default_on = dev->dev.chip == AGX_CHIP_G17P;
   if (!hk_pack_atomics_choice(default_on, opt, 1))
      return false;
   if (queue->kick.watched || hk_sibling_in_flight(dev, queue, done_cache))
      return false;

   vk_foreach_queue(iter, &dev->vk) {
      if (vk_queue_is_emulated(iter))
         continue;
      const struct hk_queue *other = container_of(iter, struct hk_queue, vk);
      if (other != queue)
         return false;
   }
   return true;
}

bool
hk_pack_atomics_allowed(struct hk_device *dev, struct hk_queue *queue)
{
   return hk_pack_atomics_allowed_cached(dev, queue, NULL);
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
   /* The fused programs' uniform loaders (hk_emit_fused_compute_dispatch()). */
   struct agx_bo *ioto_bo;
   struct agx_bo *root_bo;
   /* Every buffer the packets read, for the submission to pin
    * (queue->pack.extra_bos). */
   struct util_dynarray *extra_bos;
   unsigned packet_count;
   struct hk_packed_submit *packets;
   struct vk_command_buffer **commands;
   unsigned command_count;
   uint64_t cb_hash;
   uint32_t loop_count;
   bool obs_used;
};

static uint64_t
hk_pack_obs_hash(const struct vk_queue_submit *submit,
                 const struct hk_pack_member *members)
{
   uint64_t h = UINT64_C(14695981039346656037);
   for (unsigned i = 0; i < submit->command_buffer_count; ++i) {
      h ^= (uint64_t)(uintptr_t)submit->command_buffers[i];
      h *= UINT64_C(1099511628211);
      const struct hk_cs *cs = members[i].cs;
      if (cs) {
         h ^= cs->addr;
         h *= UINT64_C(1099511628211);
         h ^= (uint64_t)members[i].bytes;
         h *= UINT64_C(1099511628211);
         h ^= ((uint64_t)cs->stats.cmds << 1) | (uint64_t)cs->pack_loops;
         h *= UINT64_C(1099511628211);
         h ^= cs->pack_threads ^ cs->pack_static_work;
         h *= UINT64_C(1099511628211);
      } else {
         h ^= UINT64_C(0x9e3779b97f4a7c15);
         h *= UINT64_C(1099511628211);
      }
   }
   return h ? h : 1u;
}

static void
hk_pack_obs_retire(struct hk_device *dev, struct hk_queue *queue)
{
   if (!p_atomic_read(&queue->pack.obs_pending))
      return;

   uint64_t signalled = p_atomic_read(&queue->drm.signalled_value);
   bool need_query = false;
   for (unsigned i = 0; i < HK_PACK_OBS_SLOTS; ++i) {
      if (queue->pack.obs[i].pending_pt > signalled) {
         need_query = true;
         break;
      }
   }
   if (need_query && queue->drm.syncobj) {
      uint64_t pt = 0;
      if (drmSyncobjQuery(dev->dev.fd, &queue->drm.syncobj, &pt, 1) == 0 &&
          pt > signalled) {
         hk_queue_note_signalled(queue, pt);
         signalled = p_atomic_read(&queue->drm.signalled_value);
      }
   }

   const uint64_t sig_point = p_atomic_read(&queue->drm.signalled_ns_point);
   const uint64_t sig_ns = p_atomic_read(&queue->drm.signalled_ns);
   const uint64_t now_ns = (uint64_t)os_time_get_nano();
   hk_pack_obs_retire_entries(queue->pack.obs, HK_PACK_OBS_SLOTS,
                              &queue->pack.obs_pending, &queue->pack.obs_epoch,
                              signalled, sig_point, sig_ns, now_ns);
}

static uint64_t
hk_pack_obs_lookup(struct hk_queue *queue, uint64_t hash, uint32_t cb_count)
{
   for (unsigned i = 0; i < HK_PACK_OBS_SLOTS; ++i) {
      struct hk_pack_obs_entry *e = &queue->pack.obs[i];
      if (e->hash == hash && e->cb_count == cb_count) {
         e->lru_tick = ++queue->pack.obs_tick;
         return e->obs_per_loop_ns;
      }
   }
   return 0;
}

static void
hk_pack_obs_arm(struct hk_queue *queue, uint64_t hash, uint32_t cb_count,
                uint32_t loop_count, bool packed, uint64_t pt)
{
   if (!hash || loop_count < 2 || !pt)
      return;

   struct hk_pack_obs_entry *slot = NULL;
   for (unsigned i = 0; i < HK_PACK_OBS_SLOTS; ++i) {
      struct hk_pack_obs_entry *e = &queue->pack.obs[i];
      if (e->hash == hash && e->cb_count == cb_count) {
         slot = e;
         break;
      }
   }
   if (!slot) {
      for (unsigned i = 0; i < HK_PACK_OBS_SLOTS; ++i) {
         struct hk_pack_obs_entry *e = &queue->pack.obs[i];
         if (e->pending_pt)
            continue;
         if (!slot || e->lru_tick < slot->lru_tick)
            slot = e;
      }
      if (!slot)
         return;
      memset(slot, 0, sizeof(*slot));
      slot->hash = hash;
      slot->cb_count = cb_count;
   }
   if (slot->pending_pt)
      return;
   slot->loop_count = loop_count;
   slot->pending_pt = pt;
   slot->submit_ns = (uint64_t)os_time_get_nano();
   slot->in_flight_packed = packed;
   slot->lru_tick = ++queue->pack.obs_tick;
   p_atomic_inc(&queue->pack.obs_pending);
}

static void
hk_queue_pack_free_packets(struct hk_queue *queue)
{
   for (unsigned i = 0; i < queue->pack.packets_cap; ++i) {
      util_dynarray_fini(&queue->pack.packets[i].cs.compute_resources);
      util_dynarray_fini(&queue->pack.packets[i].cs.query_access_ranges);
   }
   free(queue->pack.packets);
   queue->pack.packets = NULL;
   queue->pack.packets_cap = 0;
}

static void *
hk_queue_pack_scratch(struct util_dynarray *buf, unsigned count,
                      size_t elem_size)
{
   util_dynarray_clear(buf);
   if (!util_dynarray_resize_bytes(buf, count, elem_size))
      return NULL;
   return buf->data;
}

static void *
hk_queue_pack_zscratch(struct util_dynarray *buf, unsigned count,
                       size_t elem_size)
{
   void *ptr = hk_queue_pack_scratch(buf, count, elem_size);
   if (ptr && count && elem_size)
      memset(ptr, 0, (size_t)count * elem_size);
   return ptr;
}

static bool
hk_queue_pack_ensure_packets(struct hk_queue *queue, unsigned count)
{
   if (count <= queue->pack.packets_cap)
      return true;

   unsigned new_cap = queue->pack.packets_cap ? queue->pack.packets_cap * 2 : 8;
   if (new_cap < count)
      new_cap = count;

   struct hk_packed_submit *packets = calloc(new_cap, sizeof(*packets));
   if (!packets)
      return false;

   hk_queue_pack_free_packets(queue);
   queue->pack.packets = packets;
   queue->pack.packets_cap = new_cap;
   return true;
}

static void
hk_packed_runs_free(struct hk_device *dev, struct hk_packed_runs *p)
{
   agx_bo_unreference(&dev->dev, p->bo);
   agx_bo_unreference(&dev->dev, p->ioto_bo);
   agx_bo_unreference(&dev->dev, p->root_bo);
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
compute_pair_buffers_match(const struct hk_compute_pair_dispatch *producer,
                           const struct hk_compute_pair_dispatch *clamp)
{
   const struct hk_nir_pointwise_clamp *c = &clamp->info.clamp;
   struct hk_nir_direct_shape ps = {producer->push, sizeof(producer->push)};
   struct hk_nir_direct_shape cs = {clamp->push, sizeof(clamp->push)};
   memcpy(ps.groups, producer->groups, sizeof(ps.groups));
   memcpy(cs.groups, clamp->groups, sizeof(cs.groups));

   uint64_t p_end[4], c_end[2];
   uint32_t out_off;
   uint32_t read_off[3];
   if (producer->info.role == HK_COMPUTE_PAIR_DEPTHWISE) {
      const struct hk_nir_depthwise_output *p = &producer->info.depthwise;
      if (!hk_nir_depthwise_clamp_footprint(p, c, &ps, &cs, p_end, c_end))
         return false;
      out_off = p->output.offset_B;
      for (unsigned i = 0; i < 3; ++i)
         read_off[i] = p->reads[i].offset_B;
   } else if (producer->info.role == HK_COMPUTE_PAIR_GEMM) {
      const struct hk_nir_gemm_output *p = &producer->info.gemm;
      if (!hk_nir_gemm_clamp_footprint(p, c, &ps, &cs, p_end, c_end))
         return false;
      out_off = p->output.offset_B;
      for (unsigned i = 0; i < 3; ++i)
         read_off[i] = p->reads[i].offset_B;
   } else {
      return false;
   }

   struct hk_compute_pair_interval output, input, result;
   if (!compute_pair_interval(&producer->bindings[3], out_off, p_end[3], &output) ||
       !compute_pair_interval(&clamp->bindings[0], c->input.offset_B, c_end[0], &input) ||
       !compute_pair_interval(&clamp->bindings[1], c->output.offset_B, c_end[1], &result) ||
       !compute_pair_same_interval(&output, &input) || !compute_pair_same_interval(&output, &result))
      return false;

   for (unsigned i = 0; i < 3; ++i) {
      if (i == 2 && p_end[2] == 0)
         continue; /* Optional bias buffer is absent or unread. */
      struct hk_compute_pair_interval read;
      if (!compute_pair_interval(&producer->bindings[i], read_off[i], p_end[i], &read) ||
          (read.handle == output.handle && read.start < output.end && output.start < read.end))
         return false;
   }
   return true;
}

static bool
compute_pair_submit_safe(const struct vk_queue_submit *submit)
{
   if (submit->is_protected || vk_queue_submit_has_bind(submit))
      return false;

   for (unsigned i = 0; i < submit->command_buffer_count; ++i)
      for (unsigned j = 0; j < i; ++j)
         if (submit->command_buffers[i] == submit->command_buffers[j])
            return false;
   return true;
}

static bool
compute_pair_can_replace(const struct hk_cs *producer, const struct hk_cs *consumer)
{
   if (!hk_compute_pair_stream_eligible(producer) ||
       !hk_compute_pair_stream_eligible(consumer))
      return false;

   const struct hk_compute_pair_dispatch *p = producer->compute_pair;
   const struct hk_compute_pair_dispatch *c = consumer->compute_pair;
   if (!p->producer || !p->root_va ||
       (p->info.role != HK_COMPUTE_PAIR_DEPTHWISE &&
        p->info.role != HK_COMPUTE_PAIR_GEMM) ||
       c->info.role != HK_COMPUTE_PAIR_CLAMP ||
       !hk_nir_clamp_epilogue_valid(&c->info.clamp.epilogue))
      return false;

   const uint32_t prod_fc =
      p->info.role == HK_COMPUTE_PAIR_DEPTHWISE
         ? p->info.depthwise.float_controls_execution_mode
         : p->info.gemm.float_controls_execution_mode;
   if (prod_fc != c->info.clamp.epilogue.float_controls_execution_mode)
      return false;

   return compute_pair_buffers_match(p, c);
}

static bool
compute_pair_copy4_match(const struct hk_compute_pair_dispatch *producer,
                         const struct hk_compute_pair_dispatch *copy4)
{
   if (!producer || !copy4 || !producer->producer || !producer->root_va ||
       producer->alt_kind == HK_COMPUTE_PAIR_ALT_NONE ||
       copy4->info.role != HK_COMPUTE_PAIR_COPY4)
      return false;

   const struct hk_compute_pair_copy4 *cp = &copy4->extra.copy4;
   if (!copy4->bindings[1].address ||
       copy4->bindings[1].address > UINT64_MAX - cp->dst_start ||
       cp->src_start > UINT64_MAX - cp->bytes ||
       cp->dst_start > UINT64_MAX - cp->bytes)
      return false;

   if (producer->info.role == HK_COMPUTE_PAIR_FADD) {
      const struct hk_compute_pair_copy4 *fp = &producer->extra.copy4;
      if (producer->alt_kind != HK_COMPUTE_PAIR_ALT_LINEAR_FADD ||
          hk_dual_write_push_offset(producer->info.fadd.push_bytes) +
             sizeof(struct hk_dual_copy_push) > HK_MAX_PUSH_SIZE ||
          ((copy4->bindings[1].address + cp->dst_start) % 16) != 0 ||
          fp->src_start > UINT64_MAX - fp->bytes ||
          fp->dst_start > UINT64_MAX - fp->bytes)
         return false;

      struct hk_compute_pair_interval output, src, dst;
      if (!compute_pair_interval(&producer->bindings[2], fp->dst_start,
                                 fp->dst_start + fp->bytes, &output) ||
          !compute_pair_interval(&copy4->bindings[0], cp->src_start,
                                 cp->src_start + cp->bytes, &src) ||
          !compute_pair_interval(&copy4->bindings[1], cp->dst_start,
                                 cp->dst_start + cp->bytes, &dst) ||
          !compute_pair_same_interval(&output, &src) ||
          !compute_pair_intervals_disjoint(&output, &dst))
         return false;

      for (unsigned i = 0; i < 2; ++i) {
         struct hk_compute_pair_interval read;
         if (!compute_pair_interval(&producer->bindings[i], fp->src_start,
                                    fp->src_start + fp->bytes, &read) ||
             !compute_pair_intervals_disjoint(&read, &dst))
            return false;
      }
      return true;
   }

   struct hk_nir_direct_shape ps = {producer->push, sizeof(producer->push)};
   memcpy(ps.groups, producer->groups, sizeof(ps.groups));

   uint64_t p_end[4] = {0};
   uint32_t out_off = 0, read_off[3] = {0}, push_bytes = 0, dst_align = 4;
   if (producer->info.role == HK_COMPUTE_PAIR_DEPTHWISE) {
      const struct hk_nir_depthwise_output *p = &producer->info.depthwise;
      if (producer->alt_kind != HK_COMPUTE_PAIR_ALT_DEPTHWISE_VEC4 ||
          !hk_nir_depthwise_producer_footprint(p, &ps, p_end, NULL))
         return false;
      out_off = p->output.offset_B;
      for (unsigned i = 0; i < 3; ++i)
         read_off[i] = p->reads[i].offset_B;
      push_bytes = p->push_bytes;
      dst_align = 16;
   } else if (producer->info.role == HK_COMPUTE_PAIR_GEMM) {
      const struct hk_nir_gemm_output *p = &producer->info.gemm;
      if (!hk_compute_pair_alt_is_gemm_cmat(producer->alt_kind) ||
          !hk_nir_gemm_producer_footprint(p, &ps, p_end, NULL, NULL))
         return false;
      out_off = p->output.offset_B;
      for (unsigned i = 0; i < 3; ++i)
         read_off[i] = p->reads[i].offset_B;
      push_bytes = p->push_bytes;
      dst_align = 4;
   } else {
      return false;
   }

   if (hk_dual_write_push_offset(push_bytes) + sizeof(struct hk_dual_copy_push) >
       HK_MAX_PUSH_SIZE ||
       ((copy4->bindings[1].address + cp->dst_start) % dst_align) != 0)
      return false;

   struct hk_compute_pair_interval output, src, dst;
   if (!compute_pair_interval(&producer->bindings[3], out_off, p_end[3], &output) ||
       !compute_pair_interval(&copy4->bindings[0], cp->src_start,
                              cp->src_start + cp->bytes, &src) ||
       !compute_pair_interval(&copy4->bindings[1], cp->dst_start,
                              cp->dst_start + cp->bytes, &dst) ||
       !compute_pair_same_interval(&output, &src) ||
       !compute_pair_intervals_disjoint(&output, &dst))
      return false;

   for (unsigned i = 0; i < 3; ++i) {
      if (i == 2 && p_end[2] == 0)
         continue;
      struct hk_compute_pair_interval read;
      if (!compute_pair_interval(&producer->bindings[i], read_off[i], p_end[i], &read) ||
          !compute_pair_intervals_disjoint(&read, &output) ||
          !compute_pair_intervals_disjoint(&read, &dst))
         return false;
   }
   return true;
}

static bool
compute_pair_pad4_match(const struct hk_compute_pair_dispatch *producer,
                        const struct hk_compute_pair_dispatch *pad4)
{
   if (!producer || !pad4 || !producer->producer || !producer->root_va ||
       producer->info.role != HK_COMPUTE_PAIR_GEMM ||
       !hk_compute_pair_alt_is_gemm_cmat(producer->alt_kind) ||
       pad4->info.role != HK_COMPUTE_PAIR_PAD4)
      return false;

   const struct hk_nir_gemm_output *p = &producer->info.gemm;
   if (hk_dual_write_push_offset(p->push_bytes) + sizeof(struct hk_dual_pad4_push) >
       HK_MAX_PUSH_SIZE)
      return false;

   struct hk_nir_direct_shape ps = {producer->push, sizeof(producer->push)};
   memcpy(ps.groups, producer->groups, sizeof(ps.groups));
   uint64_t p_end[4] = {0};
   uint32_t m = 0, n = 0;
   if (!hk_nir_gemm_producer_footprint(p, &ps, p_end, &m, &n))
      return false;

   const struct hk_compute_pair_pad4 *pd = &pad4->extra.pad4;
   if (pd->in_shape[0] != 1 || pd->out_shape[0] != 1 ||
       pd->in_shape[3] != pd->out_shape[3] || pd->in_shape[3] != n ||
       (uint64_t)pd->in_shape[1] * pd->in_shape[2] != m)
      return false;
   if (!pad4->bindings[1].address || (pad4->bindings[1].address % 4) != 0 ||
       !pad4->bindings[2].address || (pad4->bindings[2].address % 4) != 0)
      return false;

   struct hk_compute_pair_interval output, src, pad_b, dst;
   if (!compute_pair_interval(&producer->bindings[3], p->output.offset_B, p_end[3], &output) ||
       !compute_pair_interval(&pad4->bindings[0], 0, pd->in_bytes, &src) ||
       !compute_pair_interval(&pad4->bindings[1], 0, pd->pad_bytes, &pad_b) ||
       !compute_pair_interval(&pad4->bindings[2], 0, pd->out_bytes, &dst) ||
       !compute_pair_same_interval(&output, &src) ||
       !compute_pair_intervals_disjoint(&output, &dst) ||
       !compute_pair_intervals_disjoint(&output, &pad_b) ||
       !compute_pair_intervals_disjoint(&pad_b, &dst))
      return false;

   for (unsigned i = 0; i < 3; ++i) {
      if (i == 2 && p_end[2] == 0)
         continue;
      struct hk_compute_pair_interval read;
      if (!compute_pair_interval(&producer->bindings[i], p->reads[i].offset_B,
                                 p_end[i], &read) ||
          !compute_pair_intervals_disjoint(&read, &output) ||
          !compute_pair_intervals_disjoint(&read, &dst))
         return false;
   }
   return true;
}

static bool
compute_pair_gemm_fadd_match(const struct hk_compute_pair_dispatch *producer,
                             const struct hk_compute_pair_dispatch *fadd,
                             const struct hk_compute_pair_dispatch *copy4,
                             uint8_t *res_idx_out, bool *inplace_out)
{
   if (!producer || !fadd || !producer->producer || !producer->root_va ||
       producer->info.role != HK_COMPUTE_PAIR_GEMM ||
       !hk_compute_pair_alt_is_gemm_cmat(producer->alt_kind) ||
       fadd->info.role != HK_COMPUTE_PAIR_FADD)
      return false;

   const struct hk_compute_pair_copy4 *fc = &fadd->extra.copy4;
   if (fc->src_start != 0 || fc->dst_start != 0 || !fc->bytes ||
       !fadd->bindings[0].address || !fadd->bindings[1].address ||
       !fadd->bindings[2].address)
      return false;

   const struct hk_nir_gemm_output *p = &producer->info.gemm;
   struct hk_nir_direct_shape ps = {producer->push, sizeof(producer->push)};
   memcpy(ps.groups, producer->groups, sizeof(ps.groups));
   uint64_t p_end[4] = {0};
   if (!hk_nir_gemm_producer_footprint(p, &ps, p_end, NULL, NULL))
      return false;

   struct hk_compute_pair_interval output, src0, src1, fadd_dst;
   if (!compute_pair_interval(&producer->bindings[3], p->output.offset_B, p_end[3], &output) ||
       !compute_pair_interval(&fadd->bindings[0], 0, fc->bytes, &src0) ||
       !compute_pair_interval(&fadd->bindings[1], 0, fc->bytes, &src1) ||
       !compute_pair_interval(&fadd->bindings[2], 0, fc->bytes, &fadd_dst))
      return false;

   uint8_t res_idx = 0;
   struct hk_compute_pair_interval res;
   if (compute_pair_same_interval(&output, &src0) &&
       compute_pair_intervals_disjoint(&output, &src1)) {
      res_idx = 1;
      res = src1;
   } else if (compute_pair_same_interval(&output, &src1) &&
              compute_pair_intervals_disjoint(&output, &src0)) {
      res_idx = 0;
      res = src0;
   } else {
      return false;
   }

   const bool inplace = compute_pair_same_interval(&output, &fadd_dst);
   const bool res_inplace = compute_pair_same_interval(&res, &fadd_dst);
   if (!inplace && !res_inplace &&
       (!compute_pair_intervals_disjoint(&output, &fadd_dst) ||
        !compute_pair_intervals_disjoint(&res, &fadd_dst)))
      return false;

   if ((fadd->bindings[res_idx].address % 4) != 0 ||
       (fadd->bindings[2].address % 4) != 0)
      return false;

   const uint32_t need_extra = copy4 ? sizeof(struct hk_dual_fadd_push)
                                     : (inplace ? 8u : 16u);
   if (hk_dual_write_push_offset(p->push_bytes) + need_extra > HK_MAX_PUSH_SIZE)
      return false;

   struct hk_compute_pair_interval copy_dst = {0};
   if (copy4) {
      if (copy4->info.role != HK_COMPUTE_PAIR_COPY4)
         return false;
      const struct hk_compute_pair_copy4 *cp = &copy4->extra.copy4;
      if (!copy4->bindings[1].address ||
          copy4->bindings[1].address > UINT64_MAX - cp->dst_start ||
          cp->src_start > UINT64_MAX - cp->bytes ||
          cp->dst_start > UINT64_MAX - cp->bytes ||
          ((copy4->bindings[1].address + cp->dst_start) % 4) != 0)
         return false;
      struct hk_compute_pair_interval copy_src;
      if (!compute_pair_interval(&copy4->bindings[0], cp->src_start,
                                 cp->src_start + cp->bytes, &copy_src) ||
          !compute_pair_interval(&copy4->bindings[1], cp->dst_start,
                                 cp->dst_start + cp->bytes, &copy_dst) ||
          !compute_pair_same_interval(&fadd_dst, &copy_src) ||
          !compute_pair_intervals_disjoint(&output, &copy_dst) ||
          !compute_pair_intervals_disjoint(&res, &copy_dst) ||
          !compute_pair_intervals_disjoint(&fadd_dst, &copy_dst))
         return false;
   }

   for (unsigned i = 0; i < 3; ++i) {
      if (i == 2 && p_end[2] == 0)
         continue;
      struct hk_compute_pair_interval read;
      if (!compute_pair_interval(&producer->bindings[i], p->reads[i].offset_B,
                                 p_end[i], &read) ||
          !compute_pair_intervals_disjoint(&read, &output) ||
          !compute_pair_intervals_disjoint(&read, &fadd_dst) ||
          (copy4 && !compute_pair_intervals_disjoint(&read, &copy_dst)))
         return false;
   }

   *res_idx_out = res_idx;
   *inplace_out = inplace;
   return true;
}

static bool
compute_pair_gemm_fadd_ep_match(const struct hk_compute_pair_dispatch *producer,
                                const struct hk_compute_pair_dispatch *fadd,
                                const struct hk_compute_pair_dispatch *copy4,
                                struct hk_nir_fadd_epilogue *fadd_ep_out,
                                uint8_t *add_src_binding_out)
{
   if (!producer || !fadd || !producer->producer || !producer->root_va ||
       producer->info.role != HK_COMPUTE_PAIR_GEMM ||
       !hk_compute_pair_alt_is_gemm_cmat(producer->alt_kind) ||
       fadd->info.role != HK_COMPUTE_PAIR_FADD ||
       !fadd->info.fadd.meta_valid)
      return false;

   const struct hk_nir_gemm_output *p = &producer->info.gemm;
   if (p->float_controls_execution_mode !=
       fadd->info.fadd.float_controls_execution_mode)
      return false;

   const size_t push_need =
      copy4 ? sizeof(struct hk_dual_fadd_ep_copy_push)
            : sizeof(struct hk_dual_fadd_ep_push);
   if (hk_dual_write_push_offset(p->push_bytes) + push_need > HK_MAX_PUSH_SIZE)
      return false;

   const struct hk_compute_pair_copy4 *fp = &fadd->extra.copy4;
   if (fp->src_start > UINT64_MAX - fp->bytes ||
       fp->dst_start > UINT64_MAX - fp->bytes)
      return false;

   struct hk_nir_direct_shape ps = {producer->push, sizeof(producer->push)};
   memcpy(ps.groups, producer->groups, sizeof(ps.groups));
   uint64_t p_end[4] = {0};
   if (!hk_nir_gemm_producer_footprint(p, &ps, p_end, NULL, NULL))
      return false;

   struct hk_compute_pair_interval output, s0, s1, fadd_dst;
   if (!compute_pair_interval(&producer->bindings[3], p->output.offset_B, p_end[3], &output) ||
       !compute_pair_interval(&fadd->bindings[0], fp->src_start,
                              fp->src_start + fp->bytes, &s0) ||
       !compute_pair_interval(&fadd->bindings[1], fp->src_start,
                              fp->src_start + fp->bytes, &s1) ||
       !compute_pair_interval(&fadd->bindings[2], fp->dst_start,
                              fp->dst_start + fp->bytes, &fadd_dst))
      return false;

   const bool m0 = compute_pair_same_interval(&output, &s0);
   const bool m1 = compute_pair_same_interval(&output, &s1);
   if (m0 == m1)
      return false;

   const uint8_t add_b = m0 ? 1u : 0u;
   const struct hk_compute_pair_interval *add_src = m0 ? &s1 : &s0;
   if (!compute_pair_intervals_disjoint(&output, add_src) ||
       !compute_pair_intervals_disjoint(&output, &fadd_dst) ||
       !compute_pair_intervals_disjoint(add_src, &fadd_dst))
      return false;

   if (!fadd->bindings[add_b].address ||
       fadd->bindings[add_b].address > UINT64_MAX - fp->src_start ||
       ((fadd->bindings[add_b].address + fp->src_start) % 4) != 0 ||
       !fadd->bindings[2].address ||
       fadd->bindings[2].address > UINT64_MAX - fp->dst_start ||
       ((fadd->bindings[2].address + fp->dst_start) % 4) != 0)
      return false;

   struct hk_compute_pair_interval cdst = {0};
   if (copy4) {
      if (copy4->info.role != HK_COMPUTE_PAIR_COPY4)
         return false;
      const struct hk_compute_pair_copy4 *cp = &copy4->extra.copy4;
      if (!copy4->bindings[1].address ||
          copy4->bindings[1].address > UINT64_MAX - cp->dst_start ||
          ((copy4->bindings[1].address + cp->dst_start) % 4) != 0 ||
          cp->src_start > UINT64_MAX - cp->bytes ||
          cp->dst_start > UINT64_MAX - cp->bytes)
         return false;
      struct hk_compute_pair_interval csrc;
      if (!compute_pair_interval(&copy4->bindings[0], cp->src_start,
                                 cp->src_start + cp->bytes, &csrc) ||
          !compute_pair_interval(&copy4->bindings[1], cp->dst_start,
                                 cp->dst_start + cp->bytes, &cdst) ||
          !compute_pair_same_interval(&fadd_dst, &csrc) ||
          !compute_pair_intervals_disjoint(&output, &cdst) ||
          !compute_pair_intervals_disjoint(add_src, &cdst) ||
          !compute_pair_intervals_disjoint(&fadd_dst, &cdst))
         return false;
   }

   for (unsigned i = 0; i < 3; ++i) {
      if (i == 2 && p_end[2] == 0)
         continue;
      struct hk_compute_pair_interval read;
      if (!compute_pair_interval(&producer->bindings[i], p->reads[i].offset_B,
                                 p_end[i], &read) ||
          !compute_pair_intervals_disjoint(&read, &output) ||
          !compute_pair_intervals_disjoint(&read, &fadd_dst) ||
          (copy4 && !compute_pair_intervals_disjoint(&read, &cdst)))
         return false;
   }

   if (fadd_ep_out) {
      *fadd_ep_out = (struct hk_nir_fadd_epilogue){
         .fp_math_ctrl = fadd->info.fadd.fp_math_ctrl,
         .float_controls_execution_mode =
            fadd->info.fadd.float_controls_execution_mode,
         .prod_is_rhs = m1,
      };
   }
   if (add_src_binding_out)
      *add_src_binding_out = add_b;
   return true;
}

struct hk_compute_pair_footprint {
   struct hk_compute_pair_interval reads[3];
   struct hk_compute_pair_interval writes[1];
   uint8_t num_reads;
   uint8_t num_writes;
};

/* Return complete read/write buffer intervals for a NIR-certified dispatch.
 * Fails closed (returns false) for any uncertified role or invalid range. */
static bool
compute_pair_dispatch_footprint(const struct hk_compute_pair_dispatch *d,
                                struct hk_compute_pair_footprint *out)
{
   memset(out, 0, sizeof(*out));
   if (!d)
      return false;

   switch (d->info.role) {
   case HK_COMPUTE_PAIR_SELF_COPY:
   case HK_COMPUTE_PAIR_COPY4: {
      const struct hk_compute_pair_copy4 *cp = &d->extra.copy4;
      if (cp->src_start > UINT64_MAX - cp->bytes ||
          cp->dst_start > UINT64_MAX - cp->bytes ||
          !compute_pair_interval(&d->bindings[0], cp->src_start,
                                 cp->src_start + cp->bytes, &out->reads[0]) ||
          !compute_pair_interval(&d->bindings[1], cp->dst_start,
                                 cp->dst_start + cp->bytes, &out->writes[0]))
         return false;
      out->num_reads = 1;
      out->num_writes = 1;
      return true;
   }
   case HK_COMPUTE_PAIR_FADD: {
      const struct hk_compute_pair_copy4 *fp = &d->extra.copy4;
      if (fp->src_start > UINT64_MAX - fp->bytes ||
          fp->dst_start > UINT64_MAX - fp->bytes ||
          !compute_pair_interval(&d->bindings[0], fp->src_start,
                                 fp->src_start + fp->bytes, &out->reads[0]) ||
          !compute_pair_interval(&d->bindings[1], fp->src_start,
                                 fp->src_start + fp->bytes, &out->reads[1]) ||
          !compute_pair_interval(&d->bindings[2], fp->dst_start,
                                 fp->dst_start + fp->bytes, &out->writes[0]))
         return false;
      out->num_reads = 2;
      out->num_writes = 1;
      return true;
   }
   case HK_COMPUTE_PAIR_PAD4: {
      const struct hk_compute_pair_pad4 *pd = &d->extra.pad4;
      if (!compute_pair_interval(&d->bindings[0], 0, pd->in_bytes, &out->reads[0]) ||
          !compute_pair_interval(&d->bindings[1], 0, pd->pad_bytes, &out->reads[1]) ||
          !compute_pair_interval(&d->bindings[2], 0, pd->out_bytes, &out->writes[0]))
         return false;
      out->num_reads = 2;
      out->num_writes = 1;
      return true;
   }
   case HK_COMPUTE_PAIR_IM2COL: {
      const struct hk_compute_pair_im2col *im = &d->extra.im2col;
      if (!compute_pair_interval(&d->bindings[0], 0, im->src_bytes, &out->reads[0]) ||
          !compute_pair_interval(&d->bindings[1], 0, im->dst_bytes, &out->writes[0]))
         return false;
      out->num_reads = 1;
      out->num_writes = 1;
      return true;
   }
   case HK_COMPUTE_PAIR_CLAMP: {
      const struct hk_nir_pointwise_clamp *c = &d->info.clamp;
      struct hk_nir_direct_shape cs = {d->push, sizeof(d->push)};
      memcpy(cs.groups, d->groups, sizeof(cs.groups));
      uint64_t c_end[2] = {0};
      if (!hk_nir_pointwise_clamp_footprint(c, &cs, c_end) ||
          !compute_pair_interval(&d->bindings[0], c->input.offset_B, c_end[0],
                                 &out->reads[0]) ||
          !compute_pair_interval(&d->bindings[1], c->output.offset_B, c_end[1],
                                 &out->writes[0]))
         return false;
      out->num_reads = 1;
      out->num_writes = 1;
      return true;
   }
   case HK_COMPUTE_PAIR_DEPTHWISE:
   case HK_COMPUTE_PAIR_GEMM: {
      struct hk_nir_direct_shape ps = {d->push, sizeof(d->push)};
      memcpy(ps.groups, d->groups, sizeof(ps.groups));
      uint64_t p_end[4] = {0};
      uint32_t out_off = 0, read_off[3] = {0};
      if (d->info.role == HK_COMPUTE_PAIR_DEPTHWISE) {
         const struct hk_nir_depthwise_output *p = &d->info.depthwise;
         if (!hk_nir_depthwise_producer_footprint(p, &ps, p_end, NULL))
            return false;
         out_off = p->output.offset_B;
         for (unsigned k = 0; k < 3; ++k)
            read_off[k] = p->reads[k].offset_B;
      } else {
         const struct hk_nir_gemm_output *p = &d->info.gemm;
         if (!hk_nir_gemm_producer_footprint(p, &ps, p_end, NULL, NULL))
            return false;
         out_off = p->output.offset_B;
         for (unsigned k = 0; k < 3; ++k)
            read_off[k] = p->reads[k].offset_B;
      }
      if (!compute_pair_interval(&d->bindings[3], out_off, p_end[3], &out->writes[0]))
         return false;
      out->num_writes = 1;
      for (unsigned k = 0; k < 3; ++k) {
         if (k == 2 && p_end[2] == 0)
            continue;
         if (!compute_pair_interval(&d->bindings[k], read_off[k], p_end[k],
                                    &out->reads[out->num_reads]))
            return false;
         out->num_reads++;
      }
      return true;
   }
   default:
      return false;
   }
}

/* Scan forward within [dual_idx + 1, end) for a second COPY4 copying the same
 * producer output to a disjoint dst3, proving every intervening dispatch
 * neither writes `output` nor reads/writes `dst3`. Fails closed on any
 * uncertified dispatch or already-hoisted member. */
static bool
compute_pair_find_second_copy4(const struct hk_pack_member *members,
                               const bool *planned_suppressed,
                               unsigned prod_idx, unsigned dual_idx,
                               unsigned end, unsigned *dual2_idx_out)
{
   const struct hk_compute_pair_dispatch *p = members[prod_idx].cs->compute_pair;
   const struct hk_compute_pair_dispatch *d1 = members[dual_idx].cs->compute_pair;
   if (p->info.role != HK_COMPUTE_PAIR_DEPTHWISE &&
       p->info.role != HK_COMPUTE_PAIR_GEMM)
      return false;

   const uint32_t prod_pb =
      p->info.role == HK_COMPUTE_PAIR_DEPTHWISE ? p->info.depthwise.push_bytes
                                                : p->info.gemm.push_bytes;
   if (hk_dual_write_push_offset(prod_pb) + sizeof(struct hk_dual_copy2_push) >
       HK_MAX_PUSH_SIZE)
      return false;

   const struct hk_compute_pair_copy4 *cp1 = &d1->extra.copy4;
   struct hk_compute_pair_interval output, dst2;
   if (!compute_pair_interval(&d1->bindings[0], cp1->src_start,
                              cp1->src_start + cp1->bytes, &output) ||
       !compute_pair_interval(&d1->bindings[1], cp1->dst_start,
                              cp1->dst_start + cp1->bytes, &dst2))
      return false;

   struct hk_compute_pair_footprint inter[16];
   unsigned n_inter = 0;
   for (unsigned j = dual_idx + 1; j < end && j - prod_idx < 16; ++j) {
      if (planned_suppressed[j] || !members[j].cs ||
          !hk_compute_pair_stream_eligible(members[j].cs))
         break;

      const struct hk_compute_pair_dispatch *dj = members[j].cs->compute_pair;
      if (dj->info.role == HK_COMPUTE_PAIR_SELF_COPY)
         continue;

      if (dj->info.role == HK_COMPUTE_PAIR_COPY4 &&
          compute_pair_copy4_match(p, dj)) {
         const struct hk_compute_pair_copy4 *cp2 = &dj->extra.copy4;
         struct hk_compute_pair_interval dst3;
         if (compute_pair_interval(&dj->bindings[1], cp2->dst_start,
                                   cp2->dst_start + cp2->bytes, &dst3) &&
             compute_pair_intervals_disjoint(&dst2, &dst3)) {
            bool inter_ok = true;
            for (unsigned q = 0; q < n_inter && inter_ok; ++q) {
               for (unsigned r = 0; r < inter[q].num_reads; ++r)
                  if (!compute_pair_intervals_disjoint(&inter[q].reads[r], &dst3))
                     inter_ok = false;
               for (unsigned w = 0; w < inter[q].num_writes; ++w)
                  if (!compute_pair_intervals_disjoint(&inter[q].writes[w], &dst3))
                     inter_ok = false;
            }
            if (inter_ok) {
               *dual2_idx_out = j;
               return true;
            }
         }
      }

      if (n_inter >= ARRAY_SIZE(inter))
         break;
      if (!compute_pair_dispatch_footprint(dj, &inter[n_inter]))
         break;
      bool overwrites_out = false;
      for (unsigned w = 0; w < inter[n_inter].num_writes; ++w) {
         if (!compute_pair_intervals_disjoint(&inter[n_inter].writes[w], &output))
            overwrites_out = true;
      }
      if (overwrites_out)
         break;
      n_inter++;
   }
   return false;
}

static void
hk_pack_fold_attachments(struct hk_cs *cs, const struct hk_cs *item)
{
   for (unsigned i = 0; i < item->compute_attachments.count; ++i) {
      const struct drm_asahi_attachment *attachment =
         &item->compute_attachments.list[i];
      hk_drm_attachment_set_add(&cs->compute_attachments,
                                attachment->pointer, attachment->size);
   }
   cs->compute_attachments.complete &= item->compute_attachments.complete;
}

static bool
compute_pair_im2col_gemm_match(const struct hk_compute_pair_dispatch *im2,
                               const struct hk_compute_pair_dispatch *gemm,
                               enum hk_compute_pair_dual_write dual,
                               const struct hk_compute_pair_dispatch *d1,
                               const struct hk_compute_pair_dispatch *d2,
                               uint8_t fadd_add_src_binding)
{
   if (!im2 || !gemm || !gemm->producer || !gemm->root_va ||
       (im2->info.role != HK_COMPUTE_PAIR_IM2COL &&
        im2->info.role != HK_COMPUTE_PAIR_COPY4) ||
       gemm->info.role != HK_COMPUTE_PAIR_GEMM ||
       !hk_compute_pair_alt_is_gemm_cmat(gemm->alt_kind) ||
       hk_compute_pair_dual_is_fadd(dual))
      return false;

   if (gemm->exec_threads[0] == 0 || gemm->exec_threads[1] == 0)
      return false;

   const struct hk_nir_gemm_output *p = &gemm->info.gemm;
   if (hk_im2col_a_push_offset(p->push_bytes, dual) +
          sizeof(struct hk_im2col_a_push) > HK_MAX_PUSH_SIZE)
      return false;

   int32_t k_i = 0;
   if (p->k_offset_B + sizeof(int32_t) > sizeof(gemm->push))
      return false;
   memcpy(&k_i, gemm->push + p->k_offset_B, sizeof(k_i));
   if (k_i <= 0)
      return false;

   struct hk_nir_direct_shape ps = {gemm->push, sizeof(gemm->push)};
   memcpy(ps.groups, gemm->groups, sizeof(ps.groups));
   uint64_t p_end[4] = {0};
   uint32_t m_u = 0, n_u = 0;
   if (!hk_nir_gemm_producer_footprint(p, &ps, p_end, &m_u, &n_u))
      return false;

   struct hk_compute_pair_interval im_src, im_dst, gemm_a, gemm_b, gemm_c;
   if (im2->info.role == HK_COMPUTE_PAIR_IM2COL) {
      if (!im2->bindings[0].address || (im2->bindings[0].address % 4) != 0 ||
          !im2->bindings[1].address || (im2->bindings[1].address % 4) != 0)
         return false;
      const struct hk_compute_pair_im2col *im = &im2->extra.im2col;
      if (im->ph != m_u || im->pw != (uint32_t)k_i ||
          im->iw == 0 || im->ih == 0 || im->kw == 0 || im->cols == 0)
         return false;
      if (!compute_pair_interval(&im2->bindings[0], 0, im->src_bytes, &im_src) ||
          !compute_pair_interval(&im2->bindings[1], 0, im->dst_bytes, &im_dst))
         return false;
   } else {
      const struct hk_compute_pair_copy4 *cp = &im2->extra.copy4;
      if (((uint32_t)k_i & 3u) != 0 ||
          cp->src_start > UINT64_MAX - cp->bytes ||
          cp->dst_start > UINT64_MAX - cp->bytes ||
          cp->bytes != (uint64_t)m_u * (uint64_t)k_i * 4ull ||
          !im2->bindings[0].address ||
          ((im2->bindings[0].address + cp->src_start) & 15ull) != 0 ||
          !im2->bindings[1].address ||
          ((im2->bindings[1].address + cp->dst_start) & 15ull) != 0 ||
          !compute_pair_interval(&im2->bindings[0], cp->src_start,
                                 cp->src_start + cp->bytes, &im_src) ||
          !compute_pair_interval(&im2->bindings[1], cp->dst_start,
                                 cp->dst_start + cp->bytes, &im_dst))
         return false;
   }

   if (!compute_pair_interval(&gemm->bindings[0], p->reads[0].offset_B, p_end[0], &gemm_a) ||
       !compute_pair_interval(&gemm->bindings[1], p->reads[1].offset_B, p_end[1], &gemm_b) ||
       !compute_pair_interval(&gemm->bindings[3], p->output.offset_B, p_end[3], &gemm_c))
      return false;

   if (!compute_pair_same_interval(&im_dst, &gemm_a) ||
       !compute_pair_intervals_disjoint(&im_src, &im_dst) ||
       !compute_pair_intervals_disjoint(&im_src, &gemm_c) ||
       !compute_pair_intervals_disjoint(&im_dst, &gemm_b) ||
       !compute_pair_intervals_disjoint(&im_dst, &gemm_c))
      return false;

   if (p_end[2] > 0) {
      struct hk_compute_pair_interval gemm_bias;
      if (!compute_pair_interval(&gemm->bindings[2], p->reads[2].offset_B,
                                 p_end[2], &gemm_bias) ||
          !compute_pair_intervals_disjoint(&im_dst, &gemm_bias))
         return false;
   }

   /* Ensure im2_src is disjoint from every fused write and im2_dst (written by
    * wg_col == 0 concurrently with other workgroups) is disjoint from every
    * fused read and write. */
   if (dual == HK_COMPUTE_PAIR_DUAL_COPY || dual == HK_COMPUTE_PAIR_DUAL_COPY2) {
      if (!d1 || d1->info.role != HK_COMPUTE_PAIR_COPY4)
         return false;
      const struct hk_compute_pair_copy4 *cp1 = &d1->extra.copy4;
      struct hk_compute_pair_interval dst2;
      if (cp1->dst_start > UINT64_MAX - cp1->bytes ||
          !compute_pair_interval(&d1->bindings[1], cp1->dst_start,
                                 cp1->dst_start + cp1->bytes, &dst2) ||
          !compute_pair_intervals_disjoint(&im_src, &dst2) ||
          !compute_pair_intervals_disjoint(&im_dst, &dst2))
         return false;
      if (dual == HK_COMPUTE_PAIR_DUAL_COPY2) {
         if (!d2 || d2->info.role != HK_COMPUTE_PAIR_COPY4)
            return false;
         const struct hk_compute_pair_copy4 *cp2 = &d2->extra.copy4;
         struct hk_compute_pair_interval dst3;
         if (cp2->dst_start > UINT64_MAX - cp2->bytes ||
             !compute_pair_interval(&d2->bindings[1], cp2->dst_start,
                                    cp2->dst_start + cp2->bytes, &dst3) ||
             !compute_pair_intervals_disjoint(&im_src, &dst3) ||
             !compute_pair_intervals_disjoint(&im_dst, &dst3))
            return false;
      }
   } else if (dual == HK_COMPUTE_PAIR_DUAL_PAD4) {
      if (!d1 || d1->info.role != HK_COMPUTE_PAIR_PAD4)
         return false;
      const struct hk_compute_pair_pad4 *pd = &d1->extra.pad4;
      struct hk_compute_pair_interval pad_b, dst2;
      if (!compute_pair_interval(&d1->bindings[1], 0, pd->pad_bytes, &pad_b) ||
          !compute_pair_interval(&d1->bindings[2], 0, pd->out_bytes, &dst2) ||
          !compute_pair_intervals_disjoint(&im_dst, &pad_b) ||
          !compute_pair_intervals_disjoint(&im_src, &dst2) ||
          !compute_pair_intervals_disjoint(&im_dst, &dst2))
         return false;
   } else if (dual == HK_COMPUTE_PAIR_DUAL_FADD_EP ||
              dual == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY) {
      if (!d1 || d1->info.role != HK_COMPUTE_PAIR_FADD)
         return false;
      const struct hk_compute_pair_copy4 *fcp = &d1->extra.copy4;
      struct hk_compute_pair_interval add_src, fadd_dst;
      if (fcp->src_start > UINT64_MAX - fcp->bytes ||
          fcp->dst_start > UINT64_MAX - fcp->bytes ||
          !compute_pair_interval(&d1->bindings[fadd_add_src_binding],
                                 fcp->src_start, fcp->src_start + fcp->bytes,
                                 &add_src) ||
          !compute_pair_interval(&d1->bindings[2], fcp->dst_start,
                                 fcp->dst_start + fcp->bytes, &fadd_dst) ||
          !compute_pair_intervals_disjoint(&im_dst, &add_src) ||
          !compute_pair_intervals_disjoint(&im_src, &fadd_dst) ||
          !compute_pair_intervals_disjoint(&im_dst, &fadd_dst))
         return false;
      if (dual == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY) {
         if (!d2 || d2->info.role != HK_COMPUTE_PAIR_COPY4)
            return false;
         const struct hk_compute_pair_copy4 *cp2 = &d2->extra.copy4;
         struct hk_compute_pair_interval dst2;
         if (cp2->dst_start > UINT64_MAX - cp2->bytes ||
             !compute_pair_interval(&d2->bindings[1], cp2->dst_start,
                                    cp2->dst_start + cp2->bytes, &dst2) ||
             !compute_pair_intervals_disjoint(&im_src, &dst2) ||
             !compute_pair_intervals_disjoint(&im_dst, &dst2))
            return false;
      }
   }

   return true;
}

static bool
compute_pair_dw_gemm_match(const struct hk_compute_pair_dispatch *dw,
                           const struct hk_nir_clamp_epilogue *dw_ep,
                           const struct hk_compute_pair_dispatch *dw_copy,
                           const struct hk_compute_pair_dispatch *gemm,
                           enum hk_compute_pair_dual_write dual,
                           const struct hk_compute_pair_dispatch *d1,
                           const struct hk_compute_pair_dispatch *d2,
                           uint8_t fadd_add_src_binding,
                           struct hk_nir_dw_a_cfg *cfg_out,
                           struct hk_dw_a_push *push_out)
{
   if (!dw || !gemm || !gemm->producer || !gemm->root_va ||
       dw->info.role != HK_COMPUTE_PAIR_DEPTHWISE ||
       !dw->info.depthwise.vec4_valid ||
       gemm->info.role != HK_COMPUTE_PAIR_GEMM ||
       !hk_compute_pair_alt_is_gemm_unguarded_for(gemm->alt_kind,
                                                  gemm->info.gemm.is_guarded) ||
       hk_compute_pair_dual_is_fadd(dual))
      return false;

   /* Require single N-workgroup column (gx == 1) so each output pixel's DW
    * stencil is computed once and written directly to dst1/dst2. */
   if (gemm->exec_threads[0] == 0 || gemm->exec_threads[1] == 0 ||
       gemm->exec_local[0] == 0 ||
       gemm->exec_threads[0] != gemm->exec_local[0])
      return false;

   const struct hk_nir_depthwise_output *dwi = &dw->info.depthwise;
   const struct hk_nir_gemm_output *p = &gemm->info.gemm;
   if (dwi->float_controls_execution_mode != p->float_controls_execution_mode)
      return false;
   if (dw_ep && dw_ep->float_controls_execution_mode != p->float_controls_execution_mode)
      return false;
   if (hk_dw_a_push_offset(p->push_bytes, dual) +
          sizeof(struct hk_dw_a_push) > HK_MAX_PUSH_SIZE)
      return false;

   if (dwi->width_offset_B + 4 > sizeof(dw->push) ||
       dwi->height_offset_B + 4 > sizeof(dw->push) ||
       dwi->channels_offset_B + 4 > sizeof(dw->push) ||
       dwi->input_width_offset_B + 4 > sizeof(dw->push) ||
       dwi->input_height_offset_B + 4 > sizeof(dw->push) ||
       dwi->kernel_width_offset_B + 4 > sizeof(dw->push) ||
       dwi->kernel_height_offset_B + 4 > sizeof(dw->push) ||
       dwi->stride_x_offset_B + 4 > sizeof(dw->push) ||
       dwi->stride_y_offset_B + 4 > sizeof(dw->push) ||
       dwi->dil_x_offset_B + 4 > sizeof(dw->push) ||
       dwi->dil_y_offset_B + 4 > sizeof(dw->push) ||
       dwi->pad_x_offset_B + 4 > sizeof(dw->push) ||
       dwi->pad_y_offset_B + 4 > sizeof(dw->push))
      return false;

   int32_t out_w = 0, out_h = 0, ch = 0, in_w = 0, in_h = 0;
   int32_t kw = 0, kh = 0, sx = 0, sy = 0, dx = 0, dy = 0, px = 0, py = 0;
   memcpy(&out_w, dw->push + dwi->width_offset_B, 4);
   memcpy(&out_h, dw->push + dwi->height_offset_B, 4);
   memcpy(&ch, dw->push + dwi->channels_offset_B, 4);
   memcpy(&in_w, dw->push + dwi->input_width_offset_B, 4);
   memcpy(&in_h, dw->push + dwi->input_height_offset_B, 4);
   memcpy(&kw, dw->push + dwi->kernel_width_offset_B, 4);
   memcpy(&kh, dw->push + dwi->kernel_height_offset_B, 4);
   memcpy(&sx, dw->push + dwi->stride_x_offset_B, 4);
   memcpy(&sy, dw->push + dwi->stride_y_offset_B, 4);
   memcpy(&dx, dw->push + dwi->dil_x_offset_B, 4);
   memcpy(&dy, dw->push + dwi->dil_y_offset_B, 4);
   memcpy(&px, dw->push + dwi->pad_x_offset_B, 4);
   memcpy(&py, dw->push + dwi->pad_y_offset_B, 4);

   if (kw != 3 || kh != 3 || dx != 1 || dy != 1 ||
       (sx != 1 && sx != 2) || sy != sx ||
       in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0 ||
       in_w > 65535 || in_h > 65535 || out_w > 65535 || out_h > 65535 ||
       px < -128 || px > 127 || py < -128 || py > 127 ||
       ch <= 0 || (ch & 3) != 0)
      return false;

   bool has_bias = false;
   if (dwi->has_bias_buffer) {
      if (dwi->has_bias_offset_B == UINT32_MAX) {
         has_bias = true;
      } else {
         if (dwi->has_bias_offset_B + 4 > sizeof(dw->push))
            return false;
         int32_t hb = 0;
         memcpy(&hb, dw->push + dwi->has_bias_offset_B, 4);
         has_bias = hb != 0;
      }
   }

   const uint32_t c4 = (uint32_t)ch >> 2;
   if ((uint64_t)in_w * (uint64_t)in_h * c4 > UINT32_MAX ||
       (uint64_t)out_w * (uint64_t)out_h * c4 > UINT32_MAX ||
       9ull * c4 > UINT32_MAX)
      return false;

   struct hk_nir_direct_shape dws = {dw->push, sizeof(dw->push)};
   memcpy(dws.groups, dw->groups, sizeof(dws.groups));
   uint64_t dw_end[4] = {0}, dw_out_elems = 0;
   if (!hk_nir_depthwise_producer_footprint(dwi, &dws, dw_end, &dw_out_elems))
      return false;

   int32_t k_i = 0;
   if (p->k_offset_B + sizeof(int32_t) > sizeof(gemm->push))
      return false;
   memcpy(&k_i, gemm->push + p->k_offset_B, sizeof(k_i));
   const uint32_t k_step = hk_compute_pair_alt_gemm_k_step(gemm->alt_kind);
   const bool can_fallback_64x16 =
      gemm->alt_kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT32X16_UNGUARDED &&
      ((uint32_t)k_i % 16u) == 0 && (gemm->exec_threads[1] & 1u) == 0;
   if (k_i <= 0 || (uint32_t)k_i != (uint32_t)ch ||
       (((uint32_t)k_i % k_step) != 0 && !can_fallback_64x16))
      return false;

   struct hk_nir_direct_shape ps = {gemm->push, sizeof(gemm->push)};
   memcpy(ps.groups, gemm->groups, sizeof(ps.groups));
   uint64_t p_end[4] = {0};
   uint32_t m_u = 0, n_u = 0;
   if (!hk_nir_gemm_producer_footprint(p, &ps, p_end, &m_u, &n_u))
      return false;
   if ((uint64_t)out_w * (uint64_t)out_h != (uint64_t)m_u)
      return false;

   const uint64_t in_va = dw->bindings[0].address + dwi->reads[0].offset_B;
   const uint64_t wt_va = dw->bindings[1].address + dwi->reads[1].offset_B;
   const uint64_t bias_va =
      has_bias ? (dw->bindings[2].address + dwi->reads[2].offset_B) : 0ull;
   const uint64_t dst1_va = dw->bindings[3].address + dwi->output.offset_B;
   if (!dw->bindings[0].address || (in_va & 15ull) != 0 ||
       !dw->bindings[1].address || (wt_va & 15ull) != 0 ||
       !dw->bindings[3].address || (dst1_va & 15ull) != 0 ||
       (has_bias && (!dw->bindings[2].address || (bias_va & 15ull) != 0)))
      return false;

   uint64_t dst2_va = 0;
   struct hk_compute_pair_interval dw_dst2 = {0};
   if (dw_copy) {
      if (dw_copy->info.role != HK_COMPUTE_PAIR_COPY4 ||
          !compute_pair_copy4_match(dw, dw_copy))
         return false;
      const struct hk_compute_pair_copy4 *cp = &dw_copy->extra.copy4;
      if (cp->dst_start > UINT64_MAX - cp->bytes ||
          !dw_copy->bindings[1].address ||
          !compute_pair_interval(&dw_copy->bindings[1], cp->dst_start,
                                 cp->dst_start + cp->bytes, &dw_dst2))
         return false;
      dst2_va = dw_copy->bindings[1].address + cp->dst_start;
      if ((dst2_va & 15ull) != 0)
         return false;
   }

   struct hk_compute_pair_interval dw_in, dw_wt, dw_bias = {0}, dw_dst1;
   struct hk_compute_pair_interval gemm_a, gemm_b, gemm_bias = {0}, gemm_c;
   if (!compute_pair_interval(&dw->bindings[0], dwi->reads[0].offset_B, dw_end[0], &dw_in) ||
       !compute_pair_interval(&dw->bindings[1], dwi->reads[1].offset_B, dw_end[1], &dw_wt) ||
       !compute_pair_interval(&dw->bindings[3], dwi->output.offset_B, dw_end[3], &dw_dst1) ||
       !compute_pair_interval(&gemm->bindings[0], p->reads[0].offset_B, p_end[0], &gemm_a) ||
       !compute_pair_interval(&gemm->bindings[1], p->reads[1].offset_B, p_end[1], &gemm_b) ||
       !compute_pair_interval(&gemm->bindings[3], p->output.offset_B, p_end[3], &gemm_c))
      return false;
   if (has_bias &&
       !compute_pair_interval(&dw->bindings[2], dwi->reads[2].offset_B, dw_end[2], &dw_bias))
      return false;
   if (p_end[2] > 0 &&
       !compute_pair_interval(&gemm->bindings[2], p->reads[2].offset_B, p_end[2], &gemm_bias))
      return false;

   /* Dataflow match: GEMM's A input must be DW's output (or its fused copy4). */
   const bool matches_dst1 = compute_pair_same_interval(&dw_dst1, &gemm_a);
   const bool matches_dst2 = dw_copy && compute_pair_same_interval(&dw_dst2, &gemm_a);
   if (!matches_dst1 && !matches_dst2)
      return false;

   /* Collect all DW reads, DW writes, extra GEMM reads, and extra GEMM writes
    * to prove strict pairwise disjointness (no RAW/WAR/WAW hazards within the
    * fused dispatch). */
   const struct hk_compute_pair_interval *dw_reads[3] = { &dw_in, &dw_wt, has_bias ? &dw_bias : NULL };
   const struct hk_compute_pair_interval *dw_writes[2] = { &dw_dst1, dw_copy ? &dw_dst2 : NULL };
   struct hk_compute_pair_interval extra_reads[3];
   unsigned n_extra_reads = 0;
   extra_reads[n_extra_reads++] = gemm_b;
   if (p_end[2] > 0)
      extra_reads[n_extra_reads++] = gemm_bias;

   struct hk_compute_pair_interval extra_writes[3];
   unsigned n_extra_writes = 0;
   extra_writes[n_extra_writes++] = gemm_c;

   if (dual == HK_COMPUTE_PAIR_DUAL_COPY || dual == HK_COMPUTE_PAIR_DUAL_COPY2) {
      if (!d1 || d1->info.role != HK_COMPUTE_PAIR_COPY4)
         return false;
      const struct hk_compute_pair_copy4 *cp1 = &d1->extra.copy4;
      if (cp1->dst_start > UINT64_MAX - cp1->bytes ||
          !compute_pair_interval(&d1->bindings[1], cp1->dst_start,
                                 cp1->dst_start + cp1->bytes,
                                 &extra_writes[n_extra_writes++]))
         return false;
      if (dual == HK_COMPUTE_PAIR_DUAL_COPY2) {
         if (!d2 || d2->info.role != HK_COMPUTE_PAIR_COPY4)
            return false;
         const struct hk_compute_pair_copy4 *cp2 = &d2->extra.copy4;
         if (cp2->dst_start > UINT64_MAX - cp2->bytes ||
             !compute_pair_interval(&d2->bindings[1], cp2->dst_start,
                                    cp2->dst_start + cp2->bytes,
                                    &extra_writes[n_extra_writes++]))
            return false;
      }
   } else if (dual == HK_COMPUTE_PAIR_DUAL_PAD4) {
      if (!d1 || d1->info.role != HK_COMPUTE_PAIR_PAD4)
         return false;
      const struct hk_compute_pair_pad4 *pd = &d1->extra.pad4;
      if (!compute_pair_interval(&d1->bindings[1], 0, pd->pad_bytes,
                                 &extra_reads[n_extra_reads++]) ||
          !compute_pair_interval(&d1->bindings[2], 0, pd->out_bytes,
                                 &extra_writes[n_extra_writes++]))
         return false;
   } else if (dual == HK_COMPUTE_PAIR_DUAL_FADD_EP ||
              dual == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY) {
      if (!d1 || d1->info.role != HK_COMPUTE_PAIR_FADD)
         return false;
      const struct hk_compute_pair_copy4 *fcp = &d1->extra.copy4;
      if (fcp->src_start > UINT64_MAX - fcp->bytes ||
          fcp->dst_start > UINT64_MAX - fcp->bytes ||
          !compute_pair_interval(&d1->bindings[fadd_add_src_binding],
                                 fcp->src_start, fcp->src_start + fcp->bytes,
                                 &extra_reads[n_extra_reads++]) ||
          !compute_pair_interval(&d1->bindings[2], fcp->dst_start,
                                 fcp->dst_start + fcp->bytes,
                                 &extra_writes[n_extra_writes++]))
         return false;
      if (dual == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY) {
         if (!d2 || d2->info.role != HK_COMPUTE_PAIR_COPY4)
            return false;
         const struct hk_compute_pair_copy4 *cp2 = &d2->extra.copy4;
         if (cp2->dst_start > UINT64_MAX - cp2->bytes ||
             !compute_pair_interval(&d2->bindings[1], cp2->dst_start,
                                    cp2->dst_start + cp2->bytes,
                                    &extra_writes[n_extra_writes++]))
            return false;
      }
   }

   if (dw_copy && !compute_pair_intervals_disjoint(&dw_dst1, &dw_dst2))
      return false;
   for (unsigned r = 0; r < 3; ++r) {
      if (!dw_reads[r])
         continue;
      for (unsigned w = 0; w < 2; ++w)
         if (dw_writes[w] && !compute_pair_intervals_disjoint(dw_reads[r], dw_writes[w]))
            return false;
      for (unsigned w = 0; w < n_extra_writes; ++w)
         if (!compute_pair_intervals_disjoint(dw_reads[r], &extra_writes[w]))
            return false;
   }
   for (unsigned w = 0; w < 2; ++w) {
      if (!dw_writes[w])
         continue;
      for (unsigned r = 0; r < n_extra_reads; ++r)
         if (!compute_pair_intervals_disjoint(dw_writes[w], &extra_reads[r]))
            return false;
      for (unsigned ew = 0; ew < n_extra_writes; ++ew)
         if (!compute_pair_intervals_disjoint(dw_writes[w], &extra_writes[ew]))
            return false;
   }

   *cfg_out = (struct hk_nir_dw_a_cfg){
      .fmul_fp_ctrl = dwi->fmul_fp_ctrl,
      .fadd_fp_ctrl = dwi->fadd_fp_ctrl,
      .bias_fadd_fp_ctrl = dwi->bias_fadd_fp_ctrl,
      .has_epilogue = dw_ep != NULL,
   };
   if (dw_ep)
      cfg_out->epilogue = *dw_ep;

   const uint32_t flags =
      (has_bias ? HK_DW_A_FLAG_HAS_BIAS : 0u) |
      HK_DW_A_FLAG_HAS_DST1 |
      (dw_copy ? HK_DW_A_FLAG_HAS_DST2 : 0u) |
      (sx == 2 ? HK_DW_A_FLAG_STRIDE2 : 0u);
   *push_out = (struct hk_dw_a_push){
      .in_va_lo = (uint32_t)in_va,
      .in_va_hi = (uint32_t)(in_va >> 32),
      .wt_va_lo = (uint32_t)wt_va,
      .wt_va_hi = (uint32_t)(wt_va >> 32),
      .bias_va_lo = (uint32_t)bias_va,
      .bias_va_hi = (uint32_t)(bias_va >> 32),
      .dst1_va_lo = (uint32_t)dst1_va,
      .dst1_va_hi = (uint32_t)(dst1_va >> 32),
      .dst2_va_lo = (uint32_t)dst2_va,
      .dst2_va_hi = (uint32_t)(dst2_va >> 32),
      .in_wh = ((uint32_t)in_w & 0xffffu) | (((uint32_t)in_h & 0xffffu) << 16),
      .out_wh = ((uint32_t)out_w & 0xffffu) | (((uint32_t)out_h & 0xffffu) << 16),
      .pad_flags = ((uint32_t)(uint8_t)(int8_t)px) |
                   (((uint32_t)(uint8_t)(int8_t)py) << 8) |
                   (flags << 16),
   };
   hk_dw_a_push_init_udiv(push_out, (uint32_t)out_w);
   return true;
}

static bool
compute_pair_exp_dw_gemm_match(const struct hk_compute_pair_dispatch *exp_im2,
                               const struct hk_compute_pair_dispatch *exp,
                               const struct hk_nir_clamp_epilogue *exp_ep,
                               const struct hk_compute_pair_dispatch *dw,
                               const struct hk_compute_pair_dispatch *dw_copy,
                               const struct hk_compute_pair_dispatch *gemm,
                               enum hk_compute_pair_dual_write dual,
                               const struct hk_compute_pair_dispatch *d1,
                               const struct hk_compute_pair_dispatch *d2,
                               uint8_t fadd_add_src_binding,
                               struct hk_nir_dw_a_cfg *dw_cfg,
                               const struct hk_dw_a_push *dw_push,
                               struct hk_exp_dw_a_push *exp_push_out)
{
   if (!exp || !dw || !gemm || !dw_cfg || !dw_push || !exp_push_out ||
       exp->info.role != HK_COMPUTE_PAIR_GEMM ||
       dw->info.role != HK_COMPUTE_PAIR_DEPTHWISE ||
       gemm->info.role != HK_COMPUTE_PAIR_GEMM ||
       dual == HK_COMPUTE_PAIR_DUAL_PAD4 ||
       hk_compute_pair_dual_is_fadd(dual))
      return false;

   const uint32_t dw_flags = dw_push->pad_flags >> 16;
   const int32_t px = (int8_t)(dw_push->pad_flags & 0xffu);
   const int32_t py = (int8_t)((dw_push->pad_flags >> 8) & 0xffu);
   const uint32_t in_w = dw_push->in_wh & 0xffffu;
   const uint32_t in_h = dw_push->in_wh >> 16;
   const uint32_t out_w = dw_push->out_wh & 0xffffu;
   const uint32_t out_h = dw_push->out_wh >> 16;
   if ((dw_flags & HK_DW_A_FLAG_STRIDE2) || px != -1 || py != -1 ||
       in_w != out_w || in_h != out_h ||
       in_w < 8u || in_h < 8u || (in_w & 7u) != 0 || (in_h & 7u) != 0)
      return false;

   const struct hk_nir_gemm_output *ep_i = &exp->info.gemm;
   const struct hk_nir_gemm_output *p = &gemm->info.gemm;
   if (ep_i->float_controls_execution_mode != p->float_controls_execution_mode)
      return false;
   if (exp_ep &&
       exp_ep->float_controls_execution_mode != p->float_controls_execution_mode)
      return false;
   if (hk_exp_dw_a_push_offset(p->push_bytes, dual) +
          sizeof(struct hk_exp_dw_a_push) > HK_MAX_PUSH_SIZE)
      return false;

   int32_t k_proj = 0, k_exp = 0;
   if (p->k_offset_B + sizeof(int32_t) > sizeof(gemm->push) ||
       ep_i->k_offset_B + sizeof(int32_t) > sizeof(exp->push))
      return false;
   memcpy(&k_proj, gemm->push + p->k_offset_B, sizeof(k_proj));
   memcpy(&k_exp, exp->push + ep_i->k_offset_B, sizeof(k_exp));
   if (exp_im2) {
      if (k_proj != 32 || k_exp <= 0 || (uint32_t)k_exp > 28u)
         return false;
   } else {
      if (k_proj < 16 || (k_proj & 15) != 0 ||
          k_exp < 8 || k_exp > 24 || (k_exp & 7) != 0)
         return false;
   }

   /* Gate fusion to large expand intermediates (>= 8 MiB, above L2 across AGX3
    * chips, chosen via A/B between 12 MiB stem and 4 MiB layers) so smaller
    * cache-resident layers avoid the 16 KiB shared-memory tile overhead. */
   const uint64_t exp_c_bytes =
      (uint64_t)in_w * (uint64_t)in_h * (uint64_t)k_proj * 4ull;
   if (exp_c_bytes < (8ull << 20))
      return false;

   struct hk_nir_direct_shape es = {exp->push, sizeof(exp->push)};
   memcpy(es.groups, exp->groups, sizeof(es.groups));
   uint64_t exp_end[4] = {0};
   uint32_t m_e = 0, n_e = 0;
   if (!hk_nir_gemm_producer_footprint(ep_i, &es, exp_end, &m_e, &n_e))
      return false;
   if ((uint64_t)m_e != (uint64_t)in_w * (uint64_t)in_h ||
       n_e != (uint32_t)k_proj)
      return false;

   uint32_t im2_kh = 0;
   struct hk_compute_pair_interval im2_src = {0}, im2_dst = {0};
   if (exp_im2) {
      if (exp_im2->info.role != HK_COMPUTE_PAIR_IM2COL ||
          !compute_pair_im2col_gemm_match(exp_im2, exp, HK_COMPUTE_PAIR_DUAL_NONE,
                                          NULL, NULL, 0))
         return false;
      const struct hk_compute_pair_im2col *im = &exp_im2->extra.im2col;
      if (im->cols != in_w || im->kw == 0 || im->kw > 65535u ||
          ((uint32_t)k_exp % im->kw) != 0)
         return false;
      im2_kh = (uint32_t)k_exp / im->kw;
      if (im2_kh == 0 || im2_kh > 16u ||
          im->iw > 65535u || im->ih > 65535u ||
          im->step_x > 65535u || im->sh > 65535u ||
          im->pad_x > 32767u || im->pad_y > 32767u ||
          !exp_im2->bindings[0].address || !exp_im2->bindings[1].address ||
          !compute_pair_interval(&exp_im2->bindings[0], 0, im->src_bytes, &im2_src) ||
          !compute_pair_interval(&exp_im2->bindings[1], 0, im->dst_bytes, &im2_dst))
         return false;
   }

   bool has_exp_bias = false;
   if (ep_i->has_bias_buffer) {
      if (ep_i->has_bias_mode) {
         if (ep_i->bias_mode_offset_B + sizeof(int32_t) > sizeof(exp->push))
            return false;
         int32_t bm = 0;
         memcpy(&bm, exp->push + ep_i->bias_mode_offset_B, sizeof(bm));
         if (bm == 1 && (ep_i->bias_mode_mask & (1u << 1))) {
            has_exp_bias = true;
         } else if (bm == 0 || bm == 1) {
            return false;
         }
      } else if (ep_i->bias_mode_mask == (1u << 1)) {
         has_exp_bias = true;
      } else {
         return false;
      }
      if (has_exp_bias && exp_end[2] == 0)
         return false;
   }

   const uint64_t exp_a_va =
      exp_im2 ? exp_im2->bindings[0].address
              : (exp->bindings[0].address + ep_i->reads[0].offset_B);
   const uint64_t im2_dst_va = exp_im2 ? exp_im2->bindings[1].address : 0ull;
   const uint64_t exp_b_va = exp->bindings[1].address + ep_i->reads[1].offset_B;
   const uint64_t exp_bias_va =
      has_exp_bias ? (exp->bindings[2].address + ep_i->reads[2].offset_B) : 0ull;
   const uint64_t exp_c_va = exp->bindings[3].address + ep_i->output.offset_B;
   const uint64_t dw_in_va =
      ((uint64_t)dw_push->in_va_hi << 32) | dw_push->in_va_lo;
   if (!exp_a_va || (!exp_im2 && (exp_a_va & 15ull) != 0) ||
       (exp_im2 && ((exp_a_va & 3ull) != 0 || (im2_dst_va & 3ull) != 0)) ||
       !exp->bindings[1].address || (exp_b_va & 15ull) != 0 ||
       !exp->bindings[3].address || exp_c_va != dw_in_va ||
       (has_exp_bias && (!exp->bindings[2].address || (exp_bias_va & 15ull) != 0)))
      return false;

   struct hk_compute_pair_interval exp_a, exp_b, exp_bias = {0}, exp_c;
   if (exp_im2) {
      exp_a = im2_src;
   } else if (!compute_pair_interval(&exp->bindings[0], ep_i->reads[0].offset_B,
                                     exp_end[0], &exp_a)) {
      return false;
   }
   if (!compute_pair_interval(&exp->bindings[1], ep_i->reads[1].offset_B, exp_end[1], &exp_b) ||
       !compute_pair_interval(&exp->bindings[3], ep_i->output.offset_B, exp_end[3], &exp_c))
      return false;
   if (has_exp_bias &&
       !compute_pair_interval(&exp->bindings[2], ep_i->reads[2].offset_B, exp_end[2], &exp_bias))
      return false;

   struct hk_compute_pair_footprint dw_fp, gemm_fp;
   if (!compute_pair_dispatch_footprint(dw, &dw_fp) ||
       dw_fp.num_reads < 2 || dw_fp.num_writes != 1 ||
       !compute_pair_dispatch_footprint(gemm, &gemm_fp) ||
       gemm_fp.num_reads < 2 || gemm_fp.num_writes != 1)
      return false;
   if (!compute_pair_same_interval(&exp_c, &dw_fp.reads[0]))
      return false;

   struct hk_compute_pair_interval all_writes[8];
   unsigned n_writes = 0;
   all_writes[n_writes++] = exp_c;
   if (exp_im2)
      all_writes[n_writes++] = im2_dst;
   all_writes[n_writes++] = dw_fp.writes[0];
   if (dw_copy) {
      const struct hk_compute_pair_copy4 *cp = &dw_copy->extra.copy4;
      if (cp->dst_start > UINT64_MAX - cp->bytes ||
          !compute_pair_interval(&dw_copy->bindings[1], cp->dst_start,
                                 cp->dst_start + cp->bytes, &all_writes[n_writes++]))
         return false;
   }
   all_writes[n_writes++] = gemm_fp.writes[0];

   struct hk_compute_pair_interval other_reads[6];
   unsigned n_other_reads = 0;
   for (unsigned r = 1; r < dw_fp.num_reads; ++r)
      other_reads[n_other_reads++] = dw_fp.reads[r];
   for (unsigned r = 1; r < gemm_fp.num_reads; ++r)
      other_reads[n_other_reads++] = gemm_fp.reads[r];

   if (dual == HK_COMPUTE_PAIR_DUAL_COPY || dual == HK_COMPUTE_PAIR_DUAL_COPY2) {
      const struct hk_compute_pair_copy4 *cp1 = &d1->extra.copy4;
      if (cp1->dst_start > UINT64_MAX - cp1->bytes ||
          !compute_pair_interval(&d1->bindings[1], cp1->dst_start,
                                 cp1->dst_start + cp1->bytes, &all_writes[n_writes++]))
         return false;
      if (dual == HK_COMPUTE_PAIR_DUAL_COPY2) {
         const struct hk_compute_pair_copy4 *cp2 = &d2->extra.copy4;
         if (cp2->dst_start > UINT64_MAX - cp2->bytes ||
             !compute_pair_interval(&d2->bindings[1], cp2->dst_start,
                                    cp2->dst_start + cp2->bytes, &all_writes[n_writes++]))
            return false;
      }
   } else if (dual == HK_COMPUTE_PAIR_DUAL_FADD_EP ||
              dual == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY) {
      const struct hk_compute_pair_copy4 *fcp = &d1->extra.copy4;
      if (fcp->src_start > UINT64_MAX - fcp->bytes ||
          fcp->dst_start > UINT64_MAX - fcp->bytes ||
          !compute_pair_interval(&d1->bindings[fadd_add_src_binding],
                                 fcp->src_start, fcp->src_start + fcp->bytes,
                                 &other_reads[n_other_reads++]) ||
          !compute_pair_interval(&d1->bindings[2], fcp->dst_start,
                                 fcp->dst_start + fcp->bytes, &all_writes[n_writes++]))
         return false;
      if (dual == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY) {
         const struct hk_compute_pair_copy4 *cp2 = &d2->extra.copy4;
         if (cp2->dst_start > UINT64_MAX - cp2->bytes ||
             !compute_pair_interval(&d2->bindings[1], cp2->dst_start,
                                    cp2->dst_start + cp2->bytes, &all_writes[n_writes++]))
            return false;
      }
   }

   const struct hk_compute_pair_interval *exp_reads[3] = {
      &exp_a, &exp_b, has_exp_bias ? &exp_bias : NULL,
   };
   for (unsigned r = 0; r < 3; ++r) {
      if (!exp_reads[r])
         continue;
      for (unsigned w = 0; w < n_writes; ++w)
         if (!compute_pair_intervals_disjoint(exp_reads[r], &all_writes[w]))
            return false;
   }
   for (unsigned r = 0; r < n_other_reads; ++r) {
      if (!compute_pair_intervals_disjoint(&exp_c, &other_reads[r]))
         return false;
      if (exp_im2 && !compute_pair_intervals_disjoint(&im2_dst, &other_reads[r]))
         return false;
   }
   const unsigned n_exp_writes = exp_im2 ? 2u : 1u;
   for (unsigned ew = 0; ew < n_exp_writes; ++ew) {
      for (unsigned w = ew + 1; w < n_writes; ++w)
         if (!compute_pair_intervals_disjoint(&all_writes[ew], &all_writes[w]))
            return false;
   }

   dw_cfg->has_expand = true;
   dw_cfg->has_exp_im2col = exp_im2 != NULL;
   dw_cfg->has_exp_epilogue = exp_ep != NULL;
   if (exp_ep)
      dw_cfg->exp_epilogue = *exp_ep;

   *exp_push_out = (struct hk_exp_dw_a_push){
      .exp_a_va_lo = (uint32_t)exp_a_va,
      .exp_a_va_hi = (uint32_t)(exp_a_va >> 32),
      .exp_b_va_lo = (uint32_t)exp_b_va,
      .exp_b_va_hi = (uint32_t)(exp_b_va >> 32),
      .exp_bias_va_lo = (uint32_t)exp_bias_va,
      .exp_bias_va_hi = (uint32_t)(exp_bias_va >> 32),
      .im2_dst_va_lo = (uint32_t)im2_dst_va,
      .im2_dst_va_hi = (uint32_t)(im2_dst_va >> 32),
      .exp_k = (uint32_t)k_exp,
      .flags = has_exp_bias ? HK_EXP_DW_A_FLAG_HAS_BIAS : 0u,
   };
   if (exp_im2) {
      const struct hk_compute_pair_im2col *im = &exp_im2->extra.im2col;
      exp_push_out->im2_iw_ih = (im->iw & 0xffffu) | ((im->ih & 0xffffu) << 16);
      exp_push_out->im2_kw_kh = (im->kw & 0xffffu) | ((im2_kh & 0xffffu) << 16);
      exp_push_out->im2_step_sh = (im->step_x & 0xffffu) | ((im->sh & 0xffffu) << 16);
      exp_push_out->im2_pad_xy = (im->pad_x & 0xffffu) | ((im->pad_y & 0xffffu) << 16);
   }
   hk_exp_dw_a_push_init_udiv(exp_push_out, in_w >> 3);
   return true;
}

static bool
compute_pair_gemm_pad4_dw_match(const struct hk_compute_pair_dispatch *gemm,
                                const struct hk_nir_clamp_epilogue *gemm_ep,
                                const struct hk_compute_pair_dispatch *pad4,
                                const struct hk_compute_pair_dispatch *dw,
                                const struct hk_nir_clamp_epilogue *dw_ep,
                                const struct hk_compute_pair_dispatch *dw_copy,
                                enum hk_compute_pair_alt_kind *eff_alt_kind_out,
                                struct hk_nir_dw_a_cfg *cfg_out,
                                struct hk_dual_gemm_dw_push *push_out)
{
   if (!gemm || !pad4 || !dw || !gemm->producer || !gemm->root_va ||
       gemm->info.role != HK_COMPUTE_PAIR_GEMM ||
       !hk_compute_pair_alt_is_gemm_unguarded_for(gemm->alt_kind,
                                                  gemm->info.gemm.is_guarded) ||
       pad4->info.role != HK_COMPUTE_PAIR_PAD4 ||
       dw->info.role != HK_COMPUTE_PAIR_DEPTHWISE ||
       !dw->info.depthwise.vec4_valid)
      return false;

   if (gemm->exec_local[0] != 128 || gemm->exec_local[1] != 1 ||
       gemm->exec_local[2] != 1 ||
       gemm->exec_threads[0] == 0 || gemm->exec_threads[1] == 0)
      return false;

   const struct hk_nir_gemm_output *p = &gemm->info.gemm;
   const struct hk_nir_depthwise_output *dwi = &dw->info.depthwise;
   if (dwi->float_controls_execution_mode != p->float_controls_execution_mode)
      return false;
   if (gemm_ep &&
       gemm_ep->float_controls_execution_mode != p->float_controls_execution_mode)
      return false;
   if (dw_ep &&
       dw_ep->float_controls_execution_mode != p->float_controls_execution_mode)
      return false;
   if (hk_dual_write_push_offset(p->push_bytes) +
          sizeof(struct hk_dual_gemm_dw_push) > HK_MAX_PUSH_SIZE)
      return false;

   int32_t k_i = 0;
   if (p->k_offset_B + sizeof(int32_t) > sizeof(gemm->push))
      return false;
   memcpy(&k_i, gemm->push + p->k_offset_B, sizeof(k_i));
   if (k_i <= 0 || k_i > 32 || (k_i & 7) != 0)
      return false;

   struct hk_nir_direct_shape ps = {gemm->push, sizeof(gemm->push)};
   memcpy(ps.groups, gemm->groups, sizeof(ps.groups));
   uint64_t p_end[4] = {0};
   uint32_t m_u = 0, n_u = 0;
   if (!hk_nir_gemm_producer_footprint(p, &ps, p_end, &m_u, &n_u))
      return false;

   enum hk_compute_pair_alt_kind eff_alt = gemm->alt_kind;
   const uint32_t orig_tm = hk_compute_pair_alt_gemm_tile_m(eff_alt);
   const uint32_t orig_tn = hk_compute_pair_alt_gemm_tile_n(eff_alt);
   /* tile_n=48 allocates 15.5 KiB shared memory (2 WGs/core), so only select it
    * for single-K-step (K <= 16) fused GEMM+DW where cutting A re-reads outweighs
    * higher occupancy; larger K stays on 32/24 (3-4 WGs/core). */
   if (orig_tm != 64 || orig_tn == 64 || (orig_tn == 48 && k_i > 16)) {
      if ((n_u % 48u) == 0 && k_i <= 16) {
         eff_alt = HK_COMPUTE_PAIR_ALT_GEMM_CMAT48_UNGUARDED;
      } else if ((n_u % 32u) == 0) {
         eff_alt = gemm->info.gemm.is_guarded
                      ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT32_UNGUARDED
                      : HK_COMPUTE_PAIR_ALT_GEMM_CMAT32;
      } else if ((n_u % 24u) == 0) {
         eff_alt = HK_COMPUTE_PAIR_ALT_GEMM_CMAT24_UNGUARDED;
      } else if ((n_u % 16u) == 0) {
         eff_alt = gemm->info.gemm.is_guarded
                      ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT16_UNGUARDED
                      : HK_COMPUTE_PAIR_ALT_GEMM_CMAT16;
      } else {
         return false;
      }
   }
   const uint32_t eff_tile_n = hk_compute_pair_alt_gemm_tile_n(eff_alt);
   if (hk_compute_pair_alt_gemm_tile_m(eff_alt) != 64u ||
       (eff_tile_n != 16 && eff_tile_n != 24 && eff_tile_n != 32 && eff_tile_n != 48) ||
       (n_u % eff_tile_n) != 0 || (m_u % 64u) != 0 ||
       n_u / eff_tile_n > 65535u || m_u / 64u > 65535u)
      return false;

   const struct hk_compute_pair_pad4 *pd = &pad4->extra.pad4;
   const uint32_t in_h = pd->in_shape[1];
   const uint32_t in_w = pd->in_shape[2];
   if (pd->in_shape[0] != 1 || pd->out_shape[0] != 1 ||
       pd->in_shape[3] != pd->out_shape[3] || pd->in_shape[3] != n_u ||
       in_h < 8 || in_w < 8 || (in_h & 7u) != 0 || (in_w & 7u) != 0 ||
       in_h > 65535u || in_w > 65535u ||
       pd->out_shape[1] != in_h + 1u || pd->out_shape[2] != in_w + 1u ||
       (uint64_t)in_h * (uint64_t)in_w != (uint64_t)m_u ||
       (uint64_t)(in_h + 1u) * (uint64_t)(in_w + 1u) * (uint64_t)n_u > UINT32_MAX)
      return false;

   if (dwi->width_offset_B + 4 > sizeof(dw->push) ||
       dwi->height_offset_B + 4 > sizeof(dw->push) ||
       dwi->channels_offset_B + 4 > sizeof(dw->push) ||
       dwi->input_width_offset_B + 4 > sizeof(dw->push) ||
       dwi->input_height_offset_B + 4 > sizeof(dw->push) ||
       dwi->kernel_width_offset_B + 4 > sizeof(dw->push) ||
       dwi->kernel_height_offset_B + 4 > sizeof(dw->push) ||
       dwi->stride_x_offset_B + 4 > sizeof(dw->push) ||
       dwi->stride_y_offset_B + 4 > sizeof(dw->push) ||
       dwi->dil_x_offset_B + 4 > sizeof(dw->push) ||
       dwi->dil_y_offset_B + 4 > sizeof(dw->push) ||
       dwi->pad_x_offset_B + 4 > sizeof(dw->push) ||
       dwi->pad_y_offset_B + 4 > sizeof(dw->push))
      return false;

   int32_t dw_ow = 0, dw_oh = 0, dw_ch = 0, dw_iw = 0, dw_ih = 0;
   int32_t kw = 0, kh = 0, sx = 0, sy = 0, dx = 0, dy = 0, px = 0, py = 0;
   memcpy(&dw_ow, dw->push + dwi->width_offset_B, 4);
   memcpy(&dw_oh, dw->push + dwi->height_offset_B, 4);
   memcpy(&dw_ch, dw->push + dwi->channels_offset_B, 4);
   memcpy(&dw_iw, dw->push + dwi->input_width_offset_B, 4);
   memcpy(&dw_ih, dw->push + dwi->input_height_offset_B, 4);
   memcpy(&kw, dw->push + dwi->kernel_width_offset_B, 4);
   memcpy(&kh, dw->push + dwi->kernel_height_offset_B, 4);
   memcpy(&sx, dw->push + dwi->stride_x_offset_B, 4);
   memcpy(&sy, dw->push + dwi->stride_y_offset_B, 4);
   memcpy(&dx, dw->push + dwi->dil_x_offset_B, 4);
   memcpy(&dy, dw->push + dwi->dil_y_offset_B, 4);
   memcpy(&px, dw->push + dwi->pad_x_offset_B, 4);
   memcpy(&py, dw->push + dwi->pad_y_offset_B, 4);

   if (kw != 3 || kh != 3 || sx != 2 || sy != 2 || dx != 1 || dy != 1 ||
       px != 0 || py != 0 ||
       dw_iw != (int32_t)(in_w + 1u) || dw_ih != (int32_t)(in_h + 1u) ||
       dw_ow != (int32_t)(in_w >> 1) || dw_oh != (int32_t)(in_h >> 1) ||
       dw_ch != (int32_t)n_u || 9ull * (uint64_t)n_u > UINT32_MAX)
      return false;

   bool has_bias = false;
   if (dwi->has_bias_buffer) {
      if (dwi->has_bias_offset_B == UINT32_MAX) {
         has_bias = true;
      } else {
         if (dwi->has_bias_offset_B + 4 > sizeof(dw->push))
            return false;
         int32_t hb = 0;
         memcpy(&hb, dw->push + dwi->has_bias_offset_B, 4);
         has_bias = hb != 0;
      }
   }

   struct hk_nir_direct_shape dws = {dw->push, sizeof(dw->push)};
   memcpy(dws.groups, dw->groups, sizeof(dws.groups));
   uint64_t dw_end[4] = {0};
   if (!hk_nir_depthwise_producer_footprint(dwi, &dws, dw_end, NULL))
      return false;

   const uint64_t pad_va = pad4->bindings[1].address;
   const uint64_t pad_dst_va = pad4->bindings[2].address;
   const uint64_t dw_wt_va = dw->bindings[1].address + dwi->reads[1].offset_B;
   const uint64_t dw_bias_va =
      has_bias ? (dw->bindings[2].address + dwi->reads[2].offset_B) : 0ull;
   const uint64_t dw_dst1_va = dw->bindings[3].address + dwi->output.offset_B;
   if (!pad_va || (pad_va & 3ull) != 0 ||
       !pad_dst_va || (pad_dst_va & 3ull) != 0 ||
       !dw->bindings[1].address || (dw_wt_va & 3ull) != 0 ||
       !dw->bindings[3].address || (dw_dst1_va & 3ull) != 0 ||
       (has_bias && (!dw->bindings[2].address || (dw_bias_va & 3ull) != 0)))
      return false;

   uint64_t dw_dst2_va = 0;
   struct hk_compute_pair_interval dw_dst2 = {0};
   if (dw_copy) {
      if (dw_copy->info.role != HK_COMPUTE_PAIR_COPY4 ||
          !compute_pair_copy4_match(dw, dw_copy))
         return false;
      const struct hk_compute_pair_copy4 *cp = &dw_copy->extra.copy4;
      if (cp->dst_start > UINT64_MAX - cp->bytes ||
          !dw_copy->bindings[1].address ||
          !compute_pair_interval(&dw_copy->bindings[1], cp->dst_start,
                                 cp->dst_start + cp->bytes, &dw_dst2))
         return false;
      dw_dst2_va = dw_copy->bindings[1].address + cp->dst_start;
      if ((dw_dst2_va & 3ull) != 0)
         return false;
   }

   struct hk_compute_pair_interval gemm_a, gemm_b, gemm_bias = {0}, gemm_c;
   struct hk_compute_pair_interval pad_src, pad_b, pad_dst;
   struct hk_compute_pair_interval dw_in, dw_wt, dw_bias = {0}, dw_dst1;
   if (!compute_pair_interval(&gemm->bindings[0], p->reads[0].offset_B, p_end[0], &gemm_a) ||
       !compute_pair_interval(&gemm->bindings[1], p->reads[1].offset_B, p_end[1], &gemm_b) ||
       !compute_pair_interval(&gemm->bindings[3], p->output.offset_B, p_end[3], &gemm_c) ||
       !compute_pair_interval(&pad4->bindings[0], 0, pd->in_bytes, &pad_src) ||
       !compute_pair_interval(&pad4->bindings[1], 0, pd->pad_bytes, &pad_b) ||
       !compute_pair_interval(&pad4->bindings[2], 0, pd->out_bytes, &pad_dst) ||
       !compute_pair_interval(&dw->bindings[0], dwi->reads[0].offset_B, dw_end[0], &dw_in) ||
       !compute_pair_interval(&dw->bindings[1], dwi->reads[1].offset_B, dw_end[1], &dw_wt) ||
       !compute_pair_interval(&dw->bindings[3], dwi->output.offset_B, dw_end[3], &dw_dst1))
      return false;
   if (p_end[2] > 0 &&
       !compute_pair_interval(&gemm->bindings[2], p->reads[2].offset_B, p_end[2], &gemm_bias))
      return false;
   if (has_bias &&
       !compute_pair_interval(&dw->bindings[2], dwi->reads[2].offset_B, dw_end[2], &dw_bias))
      return false;

   /* Exact dataflow: GEMM C -> PAD4 in, PAD4 out -> DW in. */
   if (!compute_pair_same_interval(&gemm_c, &pad_src) ||
       !compute_pair_same_interval(&pad_dst, &dw_in))
      return false;

   /* Strict pairwise disjointness across all reads and writes, and among writes. */
   const struct hk_compute_pair_interval *reads[6] = {
      &gemm_a, &gemm_b, p_end[2] > 0 ? &gemm_bias : NULL,
      &pad_b, &dw_wt, has_bias ? &dw_bias : NULL,
   };
   const struct hk_compute_pair_interval *writes[4] = {
      &gemm_c, &pad_dst, &dw_dst1, dw_copy ? &dw_dst2 : NULL,
   };
   for (unsigned r = 0; r < 6; ++r) {
      if (!reads[r])
         continue;
      for (unsigned w = 0; w < 4; ++w)
         if (writes[w] && !compute_pair_intervals_disjoint(reads[r], writes[w]))
            return false;
   }
   for (unsigned w1 = 0; w1 < 4; ++w1) {
      if (!writes[w1])
         continue;
      for (unsigned w2 = w1 + 1; w2 < 4; ++w2)
         if (writes[w2] && !compute_pair_intervals_disjoint(writes[w1], writes[w2]))
            return false;
   }

   *eff_alt_kind_out = eff_alt;
   *cfg_out = (struct hk_nir_dw_a_cfg){
      .fmul_fp_ctrl = dwi->fmul_fp_ctrl,
      .fadd_fp_ctrl = dwi->fadd_fp_ctrl,
      .bias_fadd_fp_ctrl = dwi->bias_fadd_fp_ctrl,
      .has_epilogue = dw_ep != NULL,
   };
   if (dw_ep)
      cfg_out->epilogue = *dw_ep;

   *push_out = (struct hk_dual_gemm_dw_push){
      .pad_dst_va_lo = (uint32_t)pad_dst_va,
      .pad_dst_va_hi = (uint32_t)(pad_dst_va >> 32),
      .pad_va_lo = (uint32_t)pad_va,
      .pad_va_hi = (uint32_t)(pad_va >> 32),
      .dw_wt_va_lo = (uint32_t)dw_wt_va,
      .dw_wt_va_hi = (uint32_t)(dw_wt_va >> 32),
      .dw_bias_va_lo = (uint32_t)dw_bias_va,
      .dw_bias_va_hi = (uint32_t)(dw_bias_va >> 32),
      .dw_dst1_va_lo = (uint32_t)dw_dst1_va,
      .dw_dst1_va_hi = (uint32_t)(dw_dst1_va >> 32),
      .dw_dst2_va_lo = (uint32_t)dw_dst2_va,
      .dw_dst2_va_hi = (uint32_t)(dw_dst2_va >> 32),
      .in_wh = (in_w & 0xffffu) | ((in_h & 0xffffu) << 16),
      .out_wh = ((in_w + 1u) & 0xffffu) | (((in_h + 1u) & 0xffffu) << 16),
      .pad_idx = { pd->pad_idx[0], pd->pad_idx[1], pd->pad_idx[2], pd->pad_idx[3] },
      .flags = (has_bias ? HK_GEMM_DW_FLAG_HAS_BIAS : 0u) |
               (dw_copy ? HK_GEMM_DW_FLAG_HAS_DW_DST2 : 0u),
   };
   hk_dual_gemm_dw_push_init_udiv(push_out, in_w >> 3);
   return true;
}

static bool
compute_pair_gemm_dw_proj_match(const struct hk_compute_pair_dispatch *gemm,
                                const struct hk_compute_pair_dispatch *pad4,
                                const struct hk_compute_pair_dispatch *dw,
                                const struct hk_compute_pair_dispatch *dw_copy,
                                const struct hk_compute_pair_dispatch *proj,
                                const struct hk_nir_clamp_epilogue *proj_ep,
                                const struct hk_compute_pair_dispatch *proj_copy,
                                struct hk_nir_dw_a_cfg *cfg,
                                const struct hk_dual_gemm_dw_push *gdw_push,
                                struct hk_dual_gemm_dw_proj_push *proj_push_out)
{
   if (!gemm || !pad4 || !dw || !proj || !cfg || !gdw_push || !proj_push_out ||
       proj->info.role != HK_COMPUTE_PAIR_GEMM)
      return false;

   const struct hk_nir_gemm_output *p = &gemm->info.gemm;
   const struct hk_nir_gemm_output *pj = &proj->info.gemm;
   if (pj->float_controls_execution_mode != p->float_controls_execution_mode)
      return false;
   if (proj_ep &&
       proj_ep->float_controls_execution_mode != p->float_controls_execution_mode)
      return false;
   if (hk_dual_gemm_dw_proj_push_offset(p->push_bytes) +
          sizeof(struct hk_dual_gemm_dw_proj_push) > HK_MAX_PUSH_SIZE)
      return false;

   int32_t n_mid = 0, k_proj = 0;
   if (p->n_offset_B + 4 > sizeof(gemm->push) ||
       pj->k_offset_B + 4 > sizeof(proj->push))
      return false;
   memcpy(&n_mid, gemm->push + p->n_offset_B, 4);
   memcpy(&k_proj, proj->push + pj->k_offset_B, 4);
   if (n_mid <= 0 || k_proj != n_mid)
      return false;

   struct hk_nir_direct_shape pjs = {proj->push, sizeof(proj->push)};
   memcpy(pjs.groups, proj->groups, sizeof(pjs.groups));
   uint64_t pj_end[4] = {0};
   uint32_t m_pj = 0, n_pj = 0;
   if (!hk_nir_gemm_producer_footprint(pj, &pjs, pj_end, &m_pj, &n_pj))
      return false;

   const uint32_t in_w = gdw_push->in_wh & 0xffffu;
   const uint32_t in_h = gdw_push->in_wh >> 16;
   if ((uint64_t)m_pj != (uint64_t)(in_w >> 1) * (uint64_t)(in_h >> 1))
      return false;
   if (n_pj != 16u && n_pj != 24u && n_pj != 32u && n_pj != 64u)
      return false;

   bool has_bias = false;
   if (pj->has_bias_buffer) {
      if (pj->has_bias_mode) {
         if (pj->bias_mode_offset_B + 4 > sizeof(proj->push))
            return false;
         int32_t bm = 0;
         memcpy(&bm, proj->push + pj->bias_mode_offset_B, 4);
         if (bm == 1 && (pj->bias_mode_mask & (1u << 1))) {
            has_bias = true;
         } else if (bm == 0 || bm == 1) {
            return false;
         }
      } else if (pj->bias_mode_mask == (1u << 1)) {
         has_bias = true;
      } else {
         return false;
      }
      if (has_bias && pj_end[2] == 0)
         return false;
   }

   const uint64_t proj_b_va = proj->bindings[1].address + pj->reads[1].offset_B;
   const uint64_t proj_bias_va =
      has_bias ? (proj->bindings[2].address + pj->reads[2].offset_B) : 0ull;
   const uint64_t proj_c_va = proj->bindings[3].address + pj->output.offset_B;
   if (!proj->bindings[1].address || (proj_b_va & 7ull) != 0 ||
       !proj->bindings[3].address || (proj_c_va & 7ull) != 0 ||
       (has_bias && (!proj->bindings[2].address || (proj_bias_va & 7ull) != 0)))
      return false;

   uint64_t proj_copy_va = 0;
   struct hk_compute_pair_interval proj_copy_dst = {0};
   if (proj_copy) {
      if (proj_copy->info.role != HK_COMPUTE_PAIR_COPY4 ||
          !compute_pair_copy4_match(proj, proj_copy))
         return false;
      const struct hk_compute_pair_copy4 *cp = &proj_copy->extra.copy4;
      if (cp->dst_start > UINT64_MAX - cp->bytes ||
          !proj_copy->bindings[1].address ||
          !compute_pair_interval(&proj_copy->bindings[1], cp->dst_start,
                                 cp->dst_start + cp->bytes, &proj_copy_dst))
         return false;
      proj_copy_va = proj_copy->bindings[1].address + cp->dst_start;
      if ((proj_copy_va & 7ull) != 0)
         return false;
   }

   struct hk_compute_pair_interval proj_a, proj_b, proj_bias = {0}, proj_c;
   if (!compute_pair_interval(&proj->bindings[0], pj->reads[0].offset_B, pj_end[0], &proj_a) ||
       !compute_pair_interval(&proj->bindings[1], pj->reads[1].offset_B, pj_end[1], &proj_b) ||
       !compute_pair_interval(&proj->bindings[3], pj->output.offset_B, pj_end[3], &proj_c))
      return false;
   if (has_bias &&
       !compute_pair_interval(&proj->bindings[2], pj->reads[2].offset_B, pj_end[2], &proj_bias))
      return false;

   /* Exact dataflow: DW output (or its copy) -> proj A. */
   struct hk_compute_pair_footprint g_fp, dw_fp;
   if (!compute_pair_dispatch_footprint(gemm, &g_fp) ||
       !compute_pair_dispatch_footprint(dw, &dw_fp) ||
       dw_fp.num_writes != 1 || g_fp.num_writes != 1)
      return false;
   struct hk_compute_pair_interval dw_dst2 = {0};
   if (dw_copy) {
      const struct hk_compute_pair_copy4 *dcp = &dw_copy->extra.copy4;
      if (dcp->dst_start > UINT64_MAX - dcp->bytes ||
          !compute_pair_interval(&dw_copy->bindings[1], dcp->dst_start,
                                 dcp->dst_start + dcp->bytes, &dw_dst2))
         return false;
      if (!compute_pair_same_interval(&dw_dst2, &proj_a))
         return false;
   } else if (!compute_pair_same_interval(&dw_fp.writes[0], &proj_a)) {
      return false;
   }

   const struct hk_compute_pair_pad4 *pd = &pad4->extra.pad4;
   struct hk_compute_pair_interval pad_b, pad_dst;
   if (!compute_pair_interval(&pad4->bindings[1], 0, pd->pad_bytes, &pad_b) ||
       !compute_pair_interval(&pad4->bindings[2], 0, pd->out_bytes, &pad_dst))
      return false;

   /* Strict pairwise disjointness across all reads and writes. */
   const struct hk_compute_pair_interval *all_reads[8] = {0};
   unsigned nr = 0;
   for (unsigned r = 0; r < g_fp.num_reads && nr < 8; ++r)
      all_reads[nr++] = &g_fp.reads[r];
   all_reads[nr++] = &pad_b;
   for (unsigned r = 1; r < dw_fp.num_reads && nr < 8; ++r)
      all_reads[nr++] = &dw_fp.reads[r];
   all_reads[nr++] = &proj_b;
   if (has_bias)
      all_reads[nr++] = &proj_bias;

   const struct hk_compute_pair_interval *all_writes[6] = {
      &g_fp.writes[0], &pad_dst, &dw_fp.writes[0],
      dw_copy ? &dw_dst2 : NULL, &proj_c, proj_copy ? &proj_copy_dst : NULL,
   };
   for (unsigned r = 0; r < nr; ++r) {
      for (unsigned w = 0; w < 6; ++w)
         if (all_writes[w] && !compute_pair_intervals_disjoint(all_reads[r], all_writes[w]))
            return false;
   }
   for (unsigned w1 = 0; w1 < 6; ++w1) {
      if (!all_writes[w1])
         continue;
      for (unsigned w2 = w1 + 1; w2 < 6; ++w2)
         if (all_writes[w2] && !compute_pair_intervals_disjoint(all_writes[w1], all_writes[w2]))
            return false;
   }

   cfg->has_proj = true;
   cfg->proj_tile_n = (uint8_t)n_pj;
   cfg->has_proj_epilogue = proj_ep != NULL;
   if (proj_ep)
      cfg->proj_epilogue = *proj_ep;

   *proj_push_out = (struct hk_dual_gemm_dw_proj_push){
      .proj_b_va_lo = (uint32_t)proj_b_va,
      .proj_b_va_hi = (uint32_t)(proj_b_va >> 32),
      .proj_bias_va_lo = (uint32_t)proj_bias_va,
      .proj_bias_va_hi = (uint32_t)(proj_bias_va >> 32),
      .proj_c_va_lo = (uint32_t)proj_c_va,
      .proj_c_va_hi = (uint32_t)(proj_c_va >> 32),
      .proj_copy_va_lo = (uint32_t)proj_copy_va,
      .proj_copy_va_hi = (uint32_t)(proj_copy_va >> 32),
      .proj_n = n_pj,
      .proj_flags = (has_bias ? HK_GEMM_DW_PROJ_FLAG_HAS_BIAS : 0u) |
                    (proj_copy ? HK_GEMM_DW_PROJ_FLAG_HAS_COPY : 0u),
   };
   return true;
}

struct hk_pack_fuse_plan {
   struct hk_shader *shader;
   enum hk_compute_pair_dual_write dual_write;
   enum hk_compute_pair_alt_kind alt_kind_override;
   uint8_t span;
   uint8_t dual_idx_off;
   uint8_t dual2_idx_off;
   uint8_t im2col_idx_off;
   uint8_t dw_a_idx_off;
   uint8_t dw_a_copy_idx_off;
   uint8_t exp_idx_off;
   uint8_t exp_im2_idx_off;
   uint8_t gemm_dw_idx_off;
   uint8_t gemm_dw_copy_idx_off;
   uint8_t gemm_dw_proj_idx_off;
   uint8_t gemm_dw_proj_copy_idx_off;
   uint8_t fadd_add_src_binding;
   uint8_t orig_loops;
   uint8_t fused_loops;
   uint16_t ioto_bound;
   bool is_self_copy;
   bool skip_c;
   bool skip_fadd_dst;
   bool skip_im2_dst;
   bool skip_pad_dst;
   bool skip_dw_dst1;
   bool skip_dw_dst2;
   bool skip_proj_c;
   bool skip_proj_copy;
   bool skip_exp_c;
   bool skip_exp_im2_dst;
   uint64_t orig_work;
   uint64_t fused_work;
   struct hk_dw_a_push dw_a_push;
   struct hk_exp_dw_a_push exp_dw_a_push;
   struct hk_dual_gemm_dw_push gemm_dw_push;
   struct hk_dual_gemm_dw_proj_push gemm_dw_proj_push;
};

static bool
hk_pack_shader_overlap_safe(const struct hk_shader *s)
{
   const struct hk_linked_shader *linked = s ? s->only_linked : NULL;
   return s && linked && s->info.stage == MESA_SHADER_COMPUTE &&
          !s->b.info.uses_ray_traversal &&
          !s->b.info.uses_global_atomics &&
          MAX3(s->b.info.scratch_size, linked->b.scratch_size,
               linked->b.agx3_spectpr_scratch_size) <=
             HK_M5_OVERLAP_MAX_SCRATCH_B;
}

static bool
hk_cdm_downgrade_strong_to_weak(struct hk_device *dev, struct hk_cs *cs)
{
   const struct agx3_cdm_barrier_info weak_info = {
      .chip = dev->dev.chip,
      .flags = 0,
      .wide_visibility = false,
      .flush_memory = false,
   };
   uint8_t packed[AGX3_CDM_BARRIER_SIZE];
   if (!agx3_cdm_barrier_pack(&weak_info, packed))
      return false;
   memcpy((uint8_t *)cs->current - sizeof(packed), packed, sizeof(packed));
   cs->strong_flush_end = NULL;
   cs->memory_flush_end = NULL;
   cs->weak_barrier_end = cs->current;
   return true;
}

static bool
pack_footprint_add_read(struct hk_pack_footprint *fp,
                        const struct hk_compute_pair_interval *iv)
{
   if (fp->num_reads >= HK_PACK_FOOTPRINT_MAX_READS)
      return false;
   fp->reads[fp->num_reads++] = (struct hk_pack_interval){
      .start = iv->start,
      .end = iv->end,
      .handle = iv->handle,
   };
   return true;
}

static bool
pack_footprint_add_write(struct hk_pack_footprint *fp,
                         const struct hk_compute_pair_interval *iv)
{
   if (fp->num_writes >= HK_PACK_FOOTPRINT_MAX_WRITES)
      return false;
   fp->writes[fp->num_writes++] = (struct hk_pack_interval){
      .start = iv->start,
      .end = iv->end,
      .handle = iv->handle,
   };
   return true;
}

/* Complete read/write footprint for an emitted packed member (either an
 * original certified dispatch or a fused span). Fails closed (known=false) on
 * any uncertified role, global atomics, or invalid buffer range.
 *
 * When `for_liveness` is true, omitted writes (including DUAL_PAD4 `skip_c`,
 * which only writes C in the rare non-simple fallback) are excluded so they
 * are never treated as guaranteed full overwrites. When `for_liveness` is
 * false (forward hazard tracking), DUAL_PAD4 keeps C in `writes` so any
 * fallback write still receives full WAW/WAR protection. */
static void
compute_pair_emitted_footprint(const struct vk_queue_submit *submit,
                               const struct hk_pack_fuse_plan *plans,
                               unsigned idx, bool fused, bool for_liveness,
                               struct hk_pack_footprint *fp)
{
   memset(fp, 0, sizeof(*fp));
   const struct hk_cmd_buffer *source =
      (const void *)submit->command_buffers[idx];
   const struct hk_cs *item =
      list_first_entry(&source->control_streams, struct hk_cs, node);
   if (!hk_compute_pair_stream_eligible(item) || item->uses_global_atomics)
      return;

   struct hk_compute_pair_footprint base;
   if (!compute_pair_dispatch_footprint(item->compute_pair, &base))
      return;
   for (unsigned i = 0; i < base.num_reads; ++i)
      if (!pack_footprint_add_read(fp, &base.reads[i]))
         return;
   const struct hk_pack_fuse_plan *pl = &plans[idx];
   const bool omit_c =
      fused && pl->skip_c &&
      (for_liveness || pl->dual_write != HK_COMPUTE_PAIR_DUAL_PAD4);
   if (!omit_c) {
      for (unsigned i = 0; i < base.num_writes; ++i)
         if (!pack_footprint_add_write(fp, &base.writes[i]))
            return;
   }

   if (fused) {
      if (pl->im2col_idx_off) {
         const struct hk_cmd_buffer *im_src =
            (const void *)submit->command_buffers[idx - pl->im2col_idx_off];
         const struct hk_cs *im_cs =
            list_first_entry(&im_src->control_streams, struct hk_cs, node);
         const struct hk_compute_pair_dispatch *im_disp = im_cs->compute_pair;
         if (!im_disp || fp->num_reads == 0)
            return;
         struct hk_compute_pair_interval src_iv, dst_iv;
         if (im_disp->info.role == HK_COMPUTE_PAIR_IM2COL) {
            const struct hk_compute_pair_im2col *im = &im_disp->extra.im2col;
            if (!compute_pair_interval(&im_disp->bindings[0], 0, im->src_bytes, &src_iv) ||
                !compute_pair_interval(&im_disp->bindings[1], 0, im->dst_bytes, &dst_iv))
               return;
         } else if (im_disp->info.role == HK_COMPUTE_PAIR_COPY4) {
            const struct hk_compute_pair_copy4 *cp = &im_disp->extra.copy4;
            if (cp->src_start > UINT64_MAX - cp->bytes ||
                cp->dst_start > UINT64_MAX - cp->bytes ||
                !compute_pair_interval(&im_disp->bindings[0], cp->src_start,
                                       cp->src_start + cp->bytes, &src_iv) ||
                !compute_pair_interval(&im_disp->bindings[1], cp->dst_start,
                                       cp->dst_start + cp->bytes, &dst_iv))
               return;
         } else {
            return;
         }
         fp->reads[0] = (struct hk_pack_interval){
            .start = src_iv.start,
            .end = src_iv.end,
            .handle = src_iv.handle,
         };
         if (!pl->skip_im2_dst && !pack_footprint_add_write(fp, &dst_iv))
            return;
      }
      if (pl->dw_a_idx_off) {
         const struct hk_cmd_buffer *dw_src =
            (const void *)submit->command_buffers[idx - pl->dw_a_idx_off];
         const struct hk_cs *dw_cs =
            list_first_entry(&dw_src->control_streams, struct hk_cs, node);
         const struct hk_compute_pair_dispatch *dw_disp = dw_cs->compute_pair;
         if (!dw_disp || dw_disp->info.role != HK_COMPUTE_PAIR_DEPTHWISE ||
             fp->num_reads == 0)
            return;
         struct hk_compute_pair_footprint dw_fp;
         if (!compute_pair_dispatch_footprint(dw_disp, &dw_fp) ||
             dw_fp.num_reads < 2 || dw_fp.num_writes != 1)
            return;
         /* DW input replaces GEMM's A read; DW weights (+ optional bias) are added. */
         fp->reads[0] = (struct hk_pack_interval){
            .start = dw_fp.reads[0].start,
            .end = dw_fp.reads[0].end,
            .handle = dw_fp.reads[0].handle,
         };
         for (unsigned r = 1; r < dw_fp.num_reads; ++r)
            if (!pack_footprint_add_read(fp, &dw_fp.reads[r]))
               return;
         if (!pl->skip_dw_dst1 && !pack_footprint_add_write(fp, &dw_fp.writes[0]))
            return;
         if (pl->dw_a_copy_idx_off && !pl->skip_dw_dst2) {
            const struct hk_cmd_buffer *dc_src =
               (const void *)submit->command_buffers[idx - pl->dw_a_copy_idx_off];
            const struct hk_cs *dc_cs =
               list_first_entry(&dc_src->control_streams, struct hk_cs, node);
            const struct hk_compute_pair_dispatch *dc_disp = dc_cs->compute_pair;
            if (!dc_disp || dc_disp->info.role != HK_COMPUTE_PAIR_COPY4)
               return;
            const struct hk_compute_pair_copy4 *cp = &dc_disp->extra.copy4;
            struct hk_compute_pair_interval dst2_iv;
            if (cp->dst_start > UINT64_MAX - cp->bytes ||
                !compute_pair_interval(&dc_disp->bindings[1], cp->dst_start,
                                       cp->dst_start + cp->bytes, &dst2_iv) ||
                !pack_footprint_add_write(fp, &dst2_iv))
               return;
         }
         if (pl->exp_idx_off) {
            const struct hk_cmd_buffer *exp_src =
               (const void *)submit->command_buffers[idx - pl->exp_idx_off];
            const struct hk_cs *exp_cs =
               list_first_entry(&exp_src->control_streams, struct hk_cs, node);
            const struct hk_compute_pair_dispatch *exp_disp = exp_cs->compute_pair;
            if (!exp_disp || exp_disp->info.role != HK_COMPUTE_PAIR_GEMM)
               return;
            struct hk_compute_pair_footprint exp_fp;
            if (!compute_pair_dispatch_footprint(exp_disp, &exp_fp) ||
                exp_fp.num_reads < 2 || exp_fp.num_writes != 1)
               return;
            /* Expand A (or im2col src) replaces DW's input read; Expand B (+ optional bias) are added. */
            if (pl->exp_im2_idx_off) {
               const struct hk_cmd_buffer *eim_src =
                  (const void *)submit->command_buffers[idx - pl->exp_im2_idx_off];
               const struct hk_cs *eim_cs =
                  list_first_entry(&eim_src->control_streams, struct hk_cs, node);
               const struct hk_compute_pair_dispatch *eim_disp = eim_cs->compute_pair;
               if (!eim_disp || eim_disp->info.role != HK_COMPUTE_PAIR_IM2COL)
                  return;
               const struct hk_compute_pair_im2col *im = &eim_disp->extra.im2col;
               struct hk_compute_pair_interval src_iv, dst_iv;
               if (!compute_pair_interval(&eim_disp->bindings[0], 0, im->src_bytes, &src_iv) ||
                   !compute_pair_interval(&eim_disp->bindings[1], 0, im->dst_bytes, &dst_iv))
                  return;
               fp->reads[0] = (struct hk_pack_interval){
                  .start = src_iv.start,
                  .end = src_iv.end,
                  .handle = src_iv.handle,
               };
               if (!pl->skip_exp_im2_dst && !pack_footprint_add_write(fp, &dst_iv))
                  return;
            } else {
               fp->reads[0] = (struct hk_pack_interval){
                  .start = exp_fp.reads[0].start,
                  .end = exp_fp.reads[0].end,
                  .handle = exp_fp.reads[0].handle,
               };
            }
            for (unsigned r = 1; r < exp_fp.num_reads; ++r)
               if (!pack_footprint_add_read(fp, &exp_fp.reads[r]))
                  return;
            if (!pl->skip_exp_c && !pack_footprint_add_write(fp, &exp_fp.writes[0]))
               return;
         }
      }
      switch (pl->dual_write) {
      case HK_COMPUTE_PAIR_DUAL_NONE:
         /* Clamp reads and writes the producer's output interval in place. */
         break;
      case HK_COMPUTE_PAIR_DUAL_COPY:
      case HK_COMPUTE_PAIR_DUAL_COPY2: {
         const struct hk_cmd_buffer *d_src =
            (const void *)submit->command_buffers[idx + pl->dual_idx_off];
         const struct hk_cs *d_cs =
            list_first_entry(&d_src->control_streams, struct hk_cs, node);
         const struct hk_compute_pair_dispatch *d_disp = d_cs->compute_pair;
         if (!d_disp || d_disp->info.role != HK_COMPUTE_PAIR_COPY4)
            return;
         const struct hk_compute_pair_copy4 *cp = &d_disp->extra.copy4;
         struct hk_compute_pair_interval dst2;
         if (cp->dst_start > UINT64_MAX - cp->bytes ||
             !compute_pair_interval(&d_disp->bindings[1], cp->dst_start,
                                    cp->dst_start + cp->bytes, &dst2) ||
             !pack_footprint_add_write(fp, &dst2))
            return;
         if (pl->dual_write == HK_COMPUTE_PAIR_DUAL_COPY2) {
            const struct hk_cmd_buffer *d2_src =
               (const void *)submit->command_buffers[idx + pl->dual2_idx_off];
            const struct hk_cs *d2_cs =
               list_first_entry(&d2_src->control_streams, struct hk_cs, node);
            const struct hk_compute_pair_dispatch *d2_disp = d2_cs->compute_pair;
            if (!d2_disp || d2_disp->info.role != HK_COMPUTE_PAIR_COPY4)
               return;
            const struct hk_compute_pair_copy4 *cp2 = &d2_disp->extra.copy4;
            struct hk_compute_pair_interval dst3;
            if (cp2->dst_start > UINT64_MAX - cp2->bytes ||
                !compute_pair_interval(&d2_disp->bindings[1], cp2->dst_start,
                                       cp2->dst_start + cp2->bytes, &dst3) ||
                !pack_footprint_add_write(fp, &dst3))
               return;
         }
         break;
      }
      case HK_COMPUTE_PAIR_DUAL_PAD4: {
         const struct hk_cmd_buffer *d_src =
            (const void *)submit->command_buffers[idx + pl->dual_idx_off];
         const struct hk_cs *d_cs =
            list_first_entry(&d_src->control_streams, struct hk_cs, node);
         const struct hk_compute_pair_dispatch *d_disp = d_cs->compute_pair;
         if (!d_disp || d_disp->info.role != HK_COMPUTE_PAIR_PAD4)
            return;
         const struct hk_compute_pair_pad4 *pd = &d_disp->extra.pad4;
         struct hk_compute_pair_interval pad_b, dst2;
         if (!compute_pair_interval(&d_disp->bindings[1], 0, pd->pad_bytes, &pad_b) ||
             !compute_pair_interval(&d_disp->bindings[2], 0, pd->out_bytes, &dst2) ||
             !pack_footprint_add_read(fp, &pad_b) ||
             !pack_footprint_add_write(fp, &dst2))
            return;
         break;
      }
      case HK_COMPUTE_PAIR_DUAL_GEMM_DW: {
         const struct hk_cmd_buffer *p_src =
            (const void *)submit->command_buffers[idx + pl->dual_idx_off];
         const struct hk_cs *p_cs =
            list_first_entry(&p_src->control_streams, struct hk_cs, node);
         const struct hk_compute_pair_dispatch *p_disp = p_cs->compute_pair;
         if (!p_disp || p_disp->info.role != HK_COMPUTE_PAIR_PAD4 ||
             !pl->gemm_dw_idx_off)
            return;
         const struct hk_compute_pair_pad4 *pd = &p_disp->extra.pad4;
         struct hk_compute_pair_interval pad_b, pad_dst;
         if (!compute_pair_interval(&p_disp->bindings[1], 0, pd->pad_bytes, &pad_b) ||
             !compute_pair_interval(&p_disp->bindings[2], 0, pd->out_bytes, &pad_dst) ||
             !pack_footprint_add_read(fp, &pad_b) ||
             (!pl->skip_pad_dst && !pack_footprint_add_write(fp, &pad_dst)))
            return;
         const struct hk_cmd_buffer *dw_src =
            (const void *)submit->command_buffers[idx + pl->gemm_dw_idx_off];
         const struct hk_cs *dw_cs =
            list_first_entry(&dw_src->control_streams, struct hk_cs, node);
         const struct hk_compute_pair_dispatch *dw_disp = dw_cs->compute_pair;
         struct hk_compute_pair_footprint dw_fp;
         if (!dw_disp || dw_disp->info.role != HK_COMPUTE_PAIR_DEPTHWISE ||
             !compute_pair_dispatch_footprint(dw_disp, &dw_fp) ||
             dw_fp.num_reads < 2 || dw_fp.num_writes != 1)
            return;
         for (unsigned r = 1; r < dw_fp.num_reads; ++r)
            if (!pack_footprint_add_read(fp, &dw_fp.reads[r]))
               return;
         if (!pl->skip_dw_dst1 && !pack_footprint_add_write(fp, &dw_fp.writes[0]))
            return;
         if (pl->gemm_dw_copy_idx_off && !pl->skip_dw_dst2) {
            const struct hk_cmd_buffer *dc_src =
               (const void *)submit->command_buffers[idx + pl->gemm_dw_copy_idx_off];
            const struct hk_cs *dc_cs =
               list_first_entry(&dc_src->control_streams, struct hk_cs, node);
            const struct hk_compute_pair_dispatch *dc_disp = dc_cs->compute_pair;
            if (!dc_disp || dc_disp->info.role != HK_COMPUTE_PAIR_COPY4)
               return;
            const struct hk_compute_pair_copy4 *cp = &dc_disp->extra.copy4;
            struct hk_compute_pair_interval dst2_iv;
            if (cp->dst_start > UINT64_MAX - cp->bytes ||
                !compute_pair_interval(&dc_disp->bindings[1], cp->dst_start,
                                       cp->dst_start + cp->bytes, &dst2_iv) ||
                !pack_footprint_add_write(fp, &dst2_iv))
               return;
         }
         if (pl->gemm_dw_proj_idx_off) {
            const struct hk_cmd_buffer *pj_src =
               (const void *)submit->command_buffers[idx + pl->gemm_dw_proj_idx_off];
            const struct hk_cs *pj_cs =
               list_first_entry(&pj_src->control_streams, struct hk_cs, node);
            const struct hk_compute_pair_dispatch *pj_disp = pj_cs->compute_pair;
            struct hk_compute_pair_footprint pj_fp;
            if (!pj_disp || pj_disp->info.role != HK_COMPUTE_PAIR_GEMM ||
                !compute_pair_dispatch_footprint(pj_disp, &pj_fp) ||
                pj_fp.num_reads < 2 || pj_fp.num_writes != 1)
               return;
            for (unsigned r = 1; r < pj_fp.num_reads; ++r)
               if (!pack_footprint_add_read(fp, &pj_fp.reads[r]))
                  return;
            if (!pl->skip_proj_c && !pack_footprint_add_write(fp, &pj_fp.writes[0]))
               return;
            if (pl->gemm_dw_proj_copy_idx_off && !pl->skip_proj_copy) {
               const struct hk_cmd_buffer *pc_src =
                  (const void *)submit->command_buffers[idx + pl->gemm_dw_proj_copy_idx_off];
               const struct hk_cs *pc_cs =
                  list_first_entry(&pc_src->control_streams, struct hk_cs, node);
               const struct hk_compute_pair_dispatch *pc_disp = pc_cs->compute_pair;
               if (!pc_disp || pc_disp->info.role != HK_COMPUTE_PAIR_COPY4)
                  return;
               const struct hk_compute_pair_copy4 *cp = &pc_disp->extra.copy4;
               struct hk_compute_pair_interval pcp_iv;
               if (cp->dst_start > UINT64_MAX - cp->bytes ||
                   !compute_pair_interval(&pc_disp->bindings[1], cp->dst_start,
                                          cp->dst_start + cp->bytes, &pcp_iv) ||
                   !pack_footprint_add_write(fp, &pcp_iv))
                  return;
            }
         }
         break;
      }
      case HK_COMPUTE_PAIR_DUAL_FADD_INPLACE:
      case HK_COMPUTE_PAIR_DUAL_FADD_OUT:
      case HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE:
      case HK_COMPUTE_PAIR_DUAL_FADD_COPY_OUT:
      case HK_COMPUTE_PAIR_DUAL_FADD_EP:
      case HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY: {
         const struct hk_cmd_buffer *d_src =
            (const void *)submit->command_buffers[idx + pl->dual_idx_off];
         const struct hk_cs *d_cs =
            list_first_entry(&d_src->control_streams, struct hk_cs, node);
         const struct hk_compute_pair_dispatch *d_disp = d_cs->compute_pair;
         if (!d_disp || d_disp->info.role != HK_COMPUTE_PAIR_FADD)
            return;
         const struct hk_compute_pair_copy4 *fcp = &d_disp->extra.copy4;
         struct hk_compute_pair_interval add_src, fadd_dst;
         if (fcp->src_start > UINT64_MAX - fcp->bytes ||
             fcp->dst_start > UINT64_MAX - fcp->bytes ||
             !compute_pair_interval(&d_disp->bindings[pl->fadd_add_src_binding],
                                    fcp->src_start, fcp->src_start + fcp->bytes,
                                    &add_src) ||
             !compute_pair_interval(&d_disp->bindings[2], fcp->dst_start,
                                    fcp->dst_start + fcp->bytes, &fadd_dst) ||
             !pack_footprint_add_read(fp, &add_src) ||
             (!pl->skip_fadd_dst && !pack_footprint_add_write(fp, &fadd_dst)))
            return;
         if (pl->dual_write == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY ||
             pl->dual_write == HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE ||
             pl->dual_write == HK_COMPUTE_PAIR_DUAL_FADD_COPY_OUT) {
            const struct hk_cmd_buffer *d2_src =
               (const void *)submit->command_buffers[idx + pl->dual2_idx_off];
            const struct hk_cs *d2_cs =
               list_first_entry(&d2_src->control_streams, struct hk_cs, node);
            const struct hk_compute_pair_dispatch *d2_disp = d2_cs->compute_pair;
            if (!d2_disp || d2_disp->info.role != HK_COMPUTE_PAIR_COPY4)
               return;
            const struct hk_compute_pair_copy4 *cp2 = &d2_disp->extra.copy4;
            struct hk_compute_pair_interval dst2;
            if (cp2->dst_start > UINT64_MAX - cp2->bytes ||
                !compute_pair_interval(&d2_disp->bindings[1], cp2->dst_start,
                                       cp2->dst_start + cp2->bytes, &dst2) ||
                !pack_footprint_add_write(fp, &dst2))
               return;
         }
         break;
      }
      default:
         return;
      }
   }

   fp->known = true;
}

static bool
hk_pack_runs(struct hk_device *dev, struct hk_queue *queue,
             const struct vk_queue_submit *submit, bool submit_is_original,
             struct hk_packed_runs *out, struct hk_queue_done_cache *done_cache)
{
   memset(out, 0, sizeof(*out));
   const unsigned n = submit->command_buffer_count;
   if (n < 2 || submit->buffer_bind_count || submit->image_bind_count ||
       submit->image_opaque_bind_count || !hk_pack_runs_enabled(dev))
      return false;

   uint8_t terminate[HK_PACK_CDM_TERMINATE_MAX];
   const size_t terminate_size = hk_pack_cdm_terminate(dev->dev.chip, terminate);
   if (!terminate_size)
      return false;

   hk_pack_obs_retire(dev, queue);

   const bool visible_completion =
      hk_pack_boundary(dev) == HK_PACK_BOUNDARY_VISIBLE_COMPLETION;
   const bool atomics = hk_pack_atomics_allowed_cached(dev, queue, done_cache);
   const struct hk_pack_limits limits = {
      .max_attachments = hk_pack_attachment_limit(dev),
      .overhead =
         terminate_size + HK_CS_SPACE_PADDING + AGX_CDM_STREAM_LINK_LENGTH,
      .max_static_work = hk_pack_max_static_work(dev),
   };
   const bool trace = hk_pack_trace();
   const int64_t t0_ns = trace ? os_time_get_nano() : 0;

   struct hk_pack_member *members =
      hk_queue_pack_scratch(&queue->pack.members, n, sizeof(*members));
   struct hk_pack_run *runs =
      hk_queue_pack_scratch(&queue->pack.runs, n, sizeof(*runs));
   struct hk_pack_fuse_plan *plans =
      hk_queue_pack_zscratch(&queue->pack.plans, n, sizeof(*plans));
   bool *planned_suppressed =
      hk_queue_pack_zscratch(&queue->pack.planned_suppressed, n, sizeof(*planned_suppressed));
   bool *suppressed_arr =
      hk_queue_pack_zscratch(&queue->pack.suppressed_arr, n, sizeof(*suppressed_arr));
   if (!members || !runs || !plans || !planned_suppressed || !suppressed_arr)
      return false;

   util_dynarray_clear(&queue->pack.extra_bos);
   out->extra_bos = &queue->pack.extra_bos;

   const unsigned eligible = hk_pack_describe_submit(
      members, dev, &queue->sync,
      (struct hk_cmd_buffer *const *)submit->command_buffers, n, terminate,
      atomics);

   uint32_t loop_count = 0;
   for (unsigned i = 0; i < n; ++i) {
      if (members[i].cs && members[i].cs->pack_loops)
         loop_count++;
   }
   out->loop_count = loop_count;
   if (submit_is_original && !dev->dev.is_virtio && loop_count >= 2) {
      out->cb_hash = hk_pack_obs_hash(submit, members);
      const uint64_t obs_per_loop_ns =
         hk_pack_obs_lookup(queue, out->cb_hash, n);
      if (obs_per_loop_ns > 0) {
         for (unsigned i = 0; i < n; ++i) {
            if (members[i].cs && members[i].cs->pack_loops)
               members[i].obs_ns = obs_per_loop_ns;
         }
         out->obs_used = true;
      }
   }

   const bool pair_safe =
      agx_chip_native_or_m5(dev->dev.chip) && compute_pair_submit_safe(submit);
   if (pair_safe) {
      for (unsigned i = 0; i < n; ++i) {
         if (members[i].cs && hk_compute_pair_stream_eligible(members[i].cs)) {
            const struct hk_compute_pair_dispatch *p = members[i].cs->compute_pair;
            if ((p->info.role == HK_COMPUTE_PAIR_DEPTHWISE ||
                 p->info.role == HK_COMPUTE_PAIR_GEMM ||
                 p->info.role == HK_COMPUTE_PAIR_FADD) &&
                AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE > members[i].bytes) {
               members[i].bytes = AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE;
            }
         }
      }
   }

   unsigned packed_members;
   size_t total;
   const unsigned run_count =
      hk_pack_plan(members, n, &limits, runs, &packed_members, &total);

   if (!run_count) {
      if (trace)
         fprintf(stderr, "HK_PACK runs=0 eligible=%u of %u strong_tails=%d\n",
                 eligible, n,
                 hk_pack_admits_strong_tail(hk_pack_boundary(dev)));
      return false;
   }

   unsigned fused_cand_count = 0, root_cand_count = 0;
   size_t ioto_total_est = 0;
   if (pair_safe) {
      for (unsigned r = 0; r < run_count; ++r) {
         const unsigned start = runs[r].first;
         const unsigned end = start + runs[r].count;
         uint64_t run_work = 0;
         unsigned run_loops = 0;
         for (unsigned j = start; j < end; ++j) {
            if (members[j].cs) {
               run_work = members[j].cs->pack_static_work > UINT64_MAX - run_work
                             ? UINT64_MAX
                             : run_work + members[j].cs->pack_static_work;
               if (members[j].cs->pack_loops && !members[j].obs_ns)
                  run_loops++;
            }
         }
         for (unsigned i = start; i < end; ++i) {
            if (planned_suppressed[i])
               continue;
            const struct hk_cs *cs_i = members[i].cs;
            if (!cs_i || !hk_compute_pair_stream_eligible(cs_i))
               continue;
            const struct hk_compute_pair_dispatch *p = cs_i->compute_pair;
            if (p->info.role == HK_COMPUTE_PAIR_SELF_COPY) {
               plans[i].is_self_copy = true;
               continue;
            }
            if ((p->info.role != HK_COMPUTE_PAIR_DEPTHWISE &&
                 p->info.role != HK_COMPUTE_PAIR_GEMM &&
                 p->info.role != HK_COMPUTE_PAIR_FADD) ||
                !p->producer || !p->root_va)
               continue;

            unsigned k = i + 1;
            const struct hk_nir_clamp_epilogue *ep = NULL;
            if (k < end && !planned_suppressed[k] && members[k].cs &&
                compute_pair_can_replace(cs_i, members[k].cs)) {
               ep = &members[k].cs->compute_pair->info.clamp.epilogue;
               k++;
            }

            unsigned k_after_self = k;
            while (k_after_self < end && k_after_self - i < 8 &&
                   !planned_suppressed[k_after_self] &&
                   members[k_after_self].cs &&
                   hk_compute_pair_stream_eligible(members[k_after_self].cs) &&
                   members[k_after_self].cs->compute_pair->info.role ==
                      HK_COMPUTE_PAIR_SELF_COPY) {
               k_after_self++;
            }

            enum hk_compute_pair_dual_write dual = HK_COMPUTE_PAIR_DUAL_NONE;
            enum hk_compute_pair_alt_kind eff_alt_kind = p->alt_kind;
            unsigned dual_idx = 0;
            unsigned dual2_idx = 0;
            unsigned gemm_dw_idx = 0;
            unsigned gemm_dw_last_idx = 0;
            unsigned gemm_dw_pre_proj_last_idx = 0;
            unsigned gemm_dw_copy_idx = 0;
            unsigned gemm_dw_proj_idx = 0;
            unsigned gemm_dw_proj_copy_idx = 0;
            struct hk_nir_fadd_epilogue fadd_ep = {0};
            uint8_t fadd_add_src_binding = 0;
            struct hk_nir_dw_a_cfg gemm_dw_cfg = {0};
            struct hk_dual_gemm_dw_push gemm_dw_push = {0};
            struct hk_dual_gemm_dw_proj_push gemm_dw_proj_push = {0};
            if (k_after_self < end && !planned_suppressed[k_after_self] &&
                members[k_after_self].cs &&
                hk_compute_pair_stream_eligible(members[k_after_self].cs)) {
               const struct hk_compute_pair_dispatch *d =
                  members[k_after_self].cs->compute_pair;
               if (d->info.role == HK_COMPUTE_PAIR_COPY4 &&
                   compute_pair_copy4_match(p, d)) {
                  dual = HK_COMPUTE_PAIR_DUAL_COPY;
                  dual_idx = k_after_self;
                  if (compute_pair_find_second_copy4(members, planned_suppressed,
                                                     i, dual_idx, end, &dual2_idx)) {
                     dual = HK_COMPUTE_PAIR_DUAL_COPY2;
                  }
               } else if (d->info.role == HK_COMPUTE_PAIR_PAD4) {
                  bool matched_gdw = false;
                  if (debug_get_option_hk_pack_gemm_dw_opt() &&
                      p->info.role == HK_COMPUTE_PAIR_GEMM &&
                      hk_compute_pair_alt_is_gemm_unguarded_for(
                         p->alt_kind, p->info.gemm.is_guarded)) {
                     unsigned kd = k_after_self + 1;
                     while (kd < end && kd - i < 12 && !planned_suppressed[kd] &&
                            members[kd].cs &&
                            hk_compute_pair_stream_eligible(members[kd].cs) &&
                            members[kd].cs->compute_pair->info.role ==
                               HK_COMPUTE_PAIR_SELF_COPY) {
                        kd++;
                     }
                     if (kd < end && !planned_suppressed[kd] && members[kd].cs &&
                         hk_compute_pair_stream_eligible(members[kd].cs) &&
                         members[kd].cs->compute_pair->info.role ==
                            HK_COMPUTE_PAIR_DEPTHWISE) {
                        const struct hk_cs *dw_cs = members[kd].cs;
                        unsigned last_gdw = kd;
                        const struct hk_nir_clamp_epilogue *dw_ep = NULL;
                        unsigned kc = kd + 1;
                        if (kc < end && !planned_suppressed[kc] && members[kc].cs &&
                            compute_pair_can_replace(dw_cs, members[kc].cs)) {
                           dw_ep = &members[kc].cs->compute_pair->info.clamp.epilogue;
                           last_gdw = kc;
                           kc++;
                        }
                        unsigned kcp = kc;
                        while (kcp < end && kcp - i < 16 && !planned_suppressed[kcp] &&
                               members[kcp].cs &&
                               hk_compute_pair_stream_eligible(members[kcp].cs) &&
                               members[kcp].cs->compute_pair->info.role ==
                                  HK_COMPUTE_PAIR_SELF_COPY) {
                           kcp++;
                        }
                        const struct hk_compute_pair_dispatch *dw_copy_d = NULL;
                        unsigned found_dw_copy = 0;
                        if (kcp < end && !planned_suppressed[kcp] && members[kcp].cs &&
                            hk_compute_pair_stream_eligible(members[kcp].cs) &&
                            members[kcp].cs->compute_pair->info.role ==
                               HK_COMPUTE_PAIR_COPY4 &&
                            compute_pair_copy4_match(dw_cs->compute_pair,
                                                     members[kcp].cs->compute_pair)) {
                           dw_copy_d = members[kcp].cs->compute_pair;
                           found_dw_copy = kcp;
                           last_gdw = kcp;
                        }
                        enum hk_compute_pair_alt_kind gdw_alt = p->alt_kind;
                        if (compute_pair_gemm_pad4_dw_match(
                               p, ep, d, dw_cs->compute_pair, dw_ep, dw_copy_d,
                               &gdw_alt, &gemm_dw_cfg, &gemm_dw_push)) {
                           dual = HK_COMPUTE_PAIR_DUAL_GEMM_DW;
                           dual_idx = k_after_self;
                           gemm_dw_idx = kd;
                           gemm_dw_last_idx = last_gdw;
                           gemm_dw_pre_proj_last_idx = last_gdw;
                           gemm_dw_copy_idx = found_dw_copy;
                           eff_alt_kind = gdw_alt;
                           matched_gdw = true;

                           /* Look ahead for a following 1x1 Project GEMM (+ clamp + copy4)
                            * consuming DW's output to fuse the entire stride-2 block. */
                           unsigned kp = last_gdw + 1;
                           while (kp < end && kp - i < 20 && !planned_suppressed[kp] &&
                                  members[kp].cs &&
                                  hk_compute_pair_stream_eligible(members[kp].cs) &&
                                  members[kp].cs->compute_pair->info.role ==
                                     HK_COMPUTE_PAIR_SELF_COPY) {
                              kp++;
                           }
                           if (kp < end && !planned_suppressed[kp] && members[kp].cs &&
                               hk_compute_pair_stream_eligible(members[kp].cs) &&
                               members[kp].cs->compute_pair->info.role ==
                                  HK_COMPUTE_PAIR_GEMM) {
                              const struct hk_cs *pj_cs = members[kp].cs;
                              unsigned last_pj = kp;
                              const struct hk_nir_clamp_epilogue *pj_ep = NULL;
                              unsigned kpc = kp + 1;
                              if (kpc < end && !planned_suppressed[kpc] && members[kpc].cs &&
                                  compute_pair_can_replace(pj_cs, members[kpc].cs)) {
                                 pj_ep = &members[kpc].cs->compute_pair->info.clamp.epilogue;
                                 last_pj = kpc;
                                 kpc++;
                              }
                              unsigned kpcp = kpc;
                              while (kpcp < end && kpcp - i < 24 && !planned_suppressed[kpcp] &&
                                     members[kpcp].cs &&
                                     hk_compute_pair_stream_eligible(members[kpcp].cs) &&
                                     members[kpcp].cs->compute_pair->info.role ==
                                        HK_COMPUTE_PAIR_SELF_COPY) {
                                 kpcp++;
                              }
                              const struct hk_compute_pair_dispatch *pj_copy_d = NULL;
                              unsigned found_pj_copy = 0;
                              if (kpcp < end && !planned_suppressed[kpcp] && members[kpcp].cs &&
                                  hk_compute_pair_stream_eligible(members[kpcp].cs) &&
                                  members[kpcp].cs->compute_pair->info.role ==
                                     HK_COMPUTE_PAIR_COPY4 &&
                                  compute_pair_copy4_match(pj_cs->compute_pair,
                                                           members[kpcp].cs->compute_pair)) {
                                 pj_copy_d = members[kpcp].cs->compute_pair;
                                 found_pj_copy = kpcp;
                                 last_pj = kpcp;
                              }
                              if (compute_pair_gemm_dw_proj_match(
                                     p, d, dw_cs->compute_pair, dw_copy_d,
                                     pj_cs->compute_pair, pj_ep, pj_copy_d,
                                     &gemm_dw_cfg, &gemm_dw_push,
                                     &gemm_dw_proj_push)) {
                                 gemm_dw_proj_idx = kp;
                                 gemm_dw_proj_copy_idx = found_pj_copy;
                                 gemm_dw_last_idx = last_pj;
                              }
                           }
                        }
                     }
                  }
                  if (!matched_gdw && compute_pair_pad4_match(p, d)) {
                     dual = HK_COMPUTE_PAIR_DUAL_PAD4;
                     dual_idx = k_after_self;
                  }
               } else if (d->info.role == HK_COMPUTE_PAIR_FADD &&
                          compute_pair_gemm_fadd_ep_match(p, d, NULL, &fadd_ep,
                                                          &fadd_add_src_binding)) {
                  dual = HK_COMPUTE_PAIR_DUAL_FADD_EP;
                  dual_idx = k_after_self;
                  unsigned k2 = k_after_self + 1;
                  while (k2 < end && k2 - i < 12 && !planned_suppressed[k2] &&
                         members[k2].cs &&
                         hk_compute_pair_stream_eligible(members[k2].cs) &&
                         members[k2].cs->compute_pair->info.role ==
                            HK_COMPUTE_PAIR_SELF_COPY) {
                     k2++;
                  }
                  if (k2 < end && !planned_suppressed[k2] && members[k2].cs &&
                      hk_compute_pair_stream_eligible(members[k2].cs) &&
                      members[k2].cs->compute_pair->info.role ==
                         HK_COMPUTE_PAIR_COPY4 &&
                      compute_pair_gemm_fadd_ep_match(p, d,
                                                      members[k2].cs->compute_pair,
                                                      &fadd_ep,
                                                      &fadd_add_src_binding)) {
                     dual = HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY;
                     dual2_idx = k2;
                  }
               } else if (d->info.role == HK_COMPUTE_PAIR_FADD) {
                  bool inplace = false;
                  unsigned k_after_fadd = k_after_self + 1;
                  while (k_after_fadd < end && k_after_fadd - k_after_self < 8 &&
                         !planned_suppressed[k_after_fadd] &&
                         members[k_after_fadd].cs &&
                         hk_compute_pair_stream_eligible(members[k_after_fadd].cs) &&
                         members[k_after_fadd].cs->compute_pair->info.role ==
                            HK_COMPUTE_PAIR_SELF_COPY) {
                     k_after_fadd++;
                  }
                  const struct hk_compute_pair_dispatch *d2 =
                     (k_after_fadd < end && !planned_suppressed[k_after_fadd] &&
                      members[k_after_fadd].cs &&
                      hk_compute_pair_stream_eligible(members[k_after_fadd].cs) &&
                      members[k_after_fadd].cs->compute_pair->info.role ==
                         HK_COMPUTE_PAIR_COPY4)
                        ? members[k_after_fadd].cs->compute_pair
                        : NULL;
                  if (d2 && compute_pair_gemm_fadd_match(p, d, d2, &fadd_add_src_binding,
                                                         &inplace)) {
                     dual = inplace ? HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE
                                    : HK_COMPUTE_PAIR_DUAL_FADD_COPY_OUT;
                     dual_idx = k_after_self;
                     dual2_idx = k_after_fadd;
                  } else if (compute_pair_gemm_fadd_match(p, d, NULL, &fadd_add_src_binding,
                                                          &inplace)) {
                     dual = inplace ? HK_COMPUTE_PAIR_DUAL_FADD_INPLACE
                                    : HK_COMPUTE_PAIR_DUAL_FADD_OUT;
                     dual_idx = k_after_self;
                  }
               }
            }

            bool has_im2col = false;
            unsigned im2_idx = 0;
            if (p->info.role == HK_COMPUTE_PAIR_GEMM &&
                hk_compute_pair_alt_is_gemm_cmat(p->alt_kind)) {
               /* The residual-FADD post-pass does not compose with A-operand
                * fusion; prefer im2col_a and leave the FADD dispatch alone. */
               const enum hk_compute_pair_dual_write im2_dual =
                  dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW ? HK_COMPUTE_PAIR_DUAL_PAD4
                  : hk_compute_pair_dual_is_fadd(dual) ? HK_COMPUTE_PAIR_DUAL_NONE
                                                       : dual;
               unsigned prev = i;
               while (prev > start && i - prev < 8) {
                  prev--;
                  if (planned_suppressed[prev] || plans[prev].shader)
                     break;
                  if (plans[prev].is_self_copy)
                     continue;
                  if (members[prev].cs &&
                      hk_compute_pair_stream_eligible(members[prev].cs) &&
                      (members[prev].cs->compute_pair->info.role ==
                          HK_COMPUTE_PAIR_IM2COL ||
                       members[prev].cs->compute_pair->info.role ==
                          HK_COMPUTE_PAIR_COPY4)) {
                     const struct hk_compute_pair_dispatch *d1 =
                        im2_dual != HK_COMPUTE_PAIR_DUAL_NONE
                           ? members[dual_idx].cs->compute_pair
                           : NULL;
                     const struct hk_compute_pair_dispatch *d2 =
                        (im2_dual == HK_COMPUTE_PAIR_DUAL_COPY2 ||
                         im2_dual == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY)
                           ? members[dual2_idx].cs->compute_pair
                           : NULL;
                     if (compute_pair_im2col_gemm_match(
                            members[prev].cs->compute_pair, p, im2_dual, d1, d2,
                            fadd_add_src_binding)) {
                        if (dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW) {
                           if (!compute_pair_pad4_match(p, d1))
                              break;
                           dual = HK_COMPUTE_PAIR_DUAL_PAD4;
                           eff_alt_kind = p->alt_kind;
                           gemm_dw_idx = 0;
                           gemm_dw_last_idx = 0;
                           gemm_dw_copy_idx = 0;
                        } else if (hk_compute_pair_dual_is_fadd(dual)) {
                           dual = HK_COMPUTE_PAIR_DUAL_NONE;
                           dual_idx = 0;
                           dual2_idx = 0;
                           fadd_add_src_binding = 0;
                        }
                        has_im2col = true;
                        im2_idx = prev;
                        if (p->info.gemm.is_guarded &&
                            !hk_compute_pair_alt_is_gemm_unguarded(eff_alt_kind)) {
                           const uint32_t tn =
                              hk_compute_pair_alt_gemm_tile_n(eff_alt_kind);
                           const uint32_t tm =
                              hk_compute_pair_alt_gemm_tile_m(eff_alt_kind);
                           struct hk_nir_direct_shape ps = {p->push, sizeof(p->push)};
                           memcpy(ps.groups, p->groups, sizeof(ps.groups));
                           uint64_t p_end[4] = {0};
                           uint32_t m_u = 0, n_u = 0;
                           if ((tn == 16 || tn == 32 || tn == 64) &&
                               hk_nir_gemm_producer_footprint(&p->info.gemm, &ps,
                                                              p_end, &m_u, &n_u) &&
                               (m_u % tm) == 0 && (n_u % tn) == 0) {
                              eff_alt_kind =
                                 tn == 16 ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT16_UNGUARDED :
                                 tn == 32 ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT32_UNGUARDED :
                                            HK_COMPUTE_PAIR_ALT_GEMM_CMAT64_UNGUARDED;
                           }
                        }
                     }
                  }
                  break;
               }
            }

            bool has_dw_a = false;
            bool has_exp_dw_a = false;
            bool has_exp_im2 = false;
            unsigned dw_a_idx = 0;
            unsigned dw_a_copy_idx = 0;
            unsigned exp_idx = 0;
            unsigned exp_im2_idx = 0;
            enum hk_compute_pair_alt_kind non_exp_alt_kind = eff_alt_kind;
            struct hk_nir_dw_a_cfg dw_a_cfg = {0};
            struct hk_dw_a_push dw_a_push = {0};
            struct hk_exp_dw_a_push exp_dw_a_push = {0};
            if (dual != HK_COMPUTE_PAIR_DUAL_GEMM_DW && !has_im2col &&
                p->info.role == HK_COMPUTE_PAIR_GEMM &&
                hk_compute_pair_alt_is_gemm_unguarded_for(
                   p->alt_kind, p->info.gemm.is_guarded) &&
                p->exec_threads[0] > 0 && p->exec_local[0] > 0 &&
                p->exec_threads[0] == p->exec_local[0]) {
               unsigned cur = i;
               unsigned found_copy = 0, found_clamp = 0, found_dw = 0;
               bool has_copy = false, has_clamp = false, has_dw = false;
               bool chain_ok = true;
               while (cur > start && i - cur < 12) {
                  cur--;
                  if (plans[cur].is_self_copy)
                     continue;
                  if (!members[cur].cs || !hk_compute_pair_stream_eligible(members[cur].cs)) {
                     chain_ok = false;
                     break;
                  }
                  const uint32_t role = members[cur].cs->compute_pair->info.role;
                  if (role == HK_COMPUTE_PAIR_SELF_COPY)
                     continue;
                  /* Allow COPY4/CLAMP already suppressed by the preceding DW's
                   * own fused plan; strict ownership by plans[found_dw] is
                   * validated once found_dw is located. */
                  if (!has_copy && !has_clamp && role == HK_COMPUTE_PAIR_COPY4 &&
                      !plans[cur].shader) {
                     found_copy = cur;
                     has_copy = true;
                     continue;
                  }
                  if (!has_clamp && role == HK_COMPUTE_PAIR_CLAMP &&
                      !plans[cur].shader) {
                     found_clamp = cur;
                     has_clamp = true;
                     continue;
                  }
                  if (role == HK_COMPUTE_PAIR_DEPTHWISE && !planned_suppressed[cur] &&
                      (!plans[cur].shader ||
                       ((plans[cur].dual_write == HK_COMPUTE_PAIR_DUAL_NONE ||
                         plans[cur].dual_write == HK_COMPUTE_PAIR_DUAL_COPY ||
                         plans[cur].dual_write == HK_COMPUTE_PAIR_DUAL_COPY2) &&
                        !plans[cur].im2col_idx_off && !plans[cur].dw_a_idx_off &&
                        cur + plans[cur].span <= i))) {
                     found_dw = cur;
                     has_dw = true;
                  } else {
                     chain_ok = false;
                  }
                  break;
               }
               if (chain_ok && has_dw) {
                  const struct hk_pack_fuse_plan *dw_pl = &plans[found_dw];
                  for (unsigned j = found_dw + 1; j < i; ++j) {
                     if (planned_suppressed[j] &&
                         (!dw_pl->shader || j >= found_dw + dw_pl->span)) {
                        chain_ok = false;
                        break;
                     }
                  }
                  if (dw_pl->shader) {
                     if (dw_pl->dual_write == HK_COMPUTE_PAIR_DUAL_COPY ||
                         dw_pl->dual_write == HK_COMPUTE_PAIR_DUAL_COPY2) {
                        if (!has_copy || found_copy != found_dw + dw_pl->dual_idx_off)
                           chain_ok = false;
                     } else if (dw_pl->dual_write == HK_COMPUTE_PAIR_DUAL_NONE) {
                        if (dw_pl->span != 2 || !has_clamp || found_clamp != found_dw + 1)
                           chain_ok = false;
                     } else {
                        chain_ok = false;
                     }
                  }
               }
               if (chain_ok && has_dw) {
                  const struct hk_cs *dw_cs = members[found_dw].cs;
                  const struct hk_nir_clamp_epilogue *dw_ep = NULL;
                  bool sub_ok = true;
                  if (has_clamp) {
                     if (compute_pair_can_replace(dw_cs, members[found_clamp].cs))
                        dw_ep = &members[found_clamp].cs->compute_pair->info.clamp.epilogue;
                     else
                        sub_ok = false;
                  }
                  const struct hk_compute_pair_dispatch *dw_copy_d =
                     has_copy ? members[found_copy].cs->compute_pair : NULL;
                  /* As for im2col_a, prefer dw_a over the residual-FADD
                   * post-pass, which does not compose with it. */
                  const enum hk_compute_pair_dual_write dw_dual =
                     hk_compute_pair_dual_is_fadd(dual) ? HK_COMPUTE_PAIR_DUAL_NONE
                                                        : dual;
                  const struct hk_compute_pair_dispatch *d1 =
                     dw_dual != HK_COMPUTE_PAIR_DUAL_NONE
                        ? members[dual_idx].cs->compute_pair
                        : NULL;
                  const struct hk_compute_pair_dispatch *d2 =
                     (dw_dual == HK_COMPUTE_PAIR_DUAL_COPY2 ||
                      dw_dual == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY)
                        ? members[dual2_idx].cs->compute_pair
                        : NULL;
                  if (sub_ok &&
                      compute_pair_dw_gemm_match(dw_cs->compute_pair, dw_ep,
                                                 dw_copy_d, p, dw_dual, d1, d2,
                                                 fadd_add_src_binding,
                                                 &dw_a_cfg, &dw_a_push)) {
                     if (dw_dual != dual) {
                        dual = HK_COMPUTE_PAIR_DUAL_NONE;
                        dual_idx = 0;
                        dual2_idx = 0;
                        fadd_add_src_binding = 0;
                     }
                     has_dw_a = true;
                     dw_a_idx = found_dw;
                     dw_a_copy_idx = has_copy ? found_copy : 0;
                     /* 64x64 + dw_a (4x4 accs) exceeds 160 GPRs; downgrade to
                      * 32x64 (2x4 accs, K_step=32) when K is a multiple of 32. */
                     if (eff_alt_kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64 ||
                         eff_alt_kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64_UNGUARDED) {
                        int32_t k_dw = 0;
                        memcpy(&k_dw, p->push + p->info.gemm.k_offset_B, 4);
                        if (k_dw > 0 && ((uint32_t)k_dw % 32u) == 0 &&
                            p->exec_threads[1] <= 32767u) {
                           eff_alt_kind = p->info.gemm.is_guarded
                                             ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT64_UNGUARDED
                                             : HK_COMPUTE_PAIR_ALT_GEMM_CMAT64;
                        }
                     } else if (eff_alt_kind == HK_COMPUTE_PAIR_ALT_GEMM_CMAT32X16_UNGUARDED) {
                        int32_t k_dw = 0;
                        memcpy(&k_dw, p->push + p->info.gemm.k_offset_B, 4);
                        if (k_dw > 0 && ((uint32_t)k_dw % 32u) != 0 &&
                            ((uint32_t)k_dw % 16u) == 0 &&
                            (p->exec_threads[1] & 1u) == 0) {
                           eff_alt_kind = p->info.gemm.is_guarded
                                             ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT16_UNGUARDED
                                             : HK_COMPUTE_PAIR_ALT_GEMM_CMAT16;
                        }
                     }
                     non_exp_alt_kind = eff_alt_kind;

                     /* Check for a preceding im2col + expand GEMM (+ optional
                      * clamp) feeding DW input to fuse the full stem block. */
                     enum hk_compute_pair_alt_kind cand_exp_alt = p->alt_kind;
                     switch (p->alt_kind) {
                     case HK_COMPUTE_PAIR_ALT_GEMM_CMAT32X16_UNGUARDED:
                        cand_exp_alt = p->info.gemm.is_guarded
                                          ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT16_UNGUARDED
                                          : HK_COMPUTE_PAIR_ALT_GEMM_CMAT16;
                        break;
                     case HK_COMPUTE_PAIR_ALT_GEMM_CMAT32X32_UNGUARDED:
                        cand_exp_alt = p->info.gemm.is_guarded
                                          ? HK_COMPUTE_PAIR_ALT_GEMM_CMAT32_UNGUARDED
                                          : HK_COMPUTE_PAIR_ALT_GEMM_CMAT32;
                        break;
                     case HK_COMPUTE_PAIR_ALT_GEMM_CMAT64_UNGUARDED:
                        cand_exp_alt = HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64_UNGUARDED;
                        break;
                     case HK_COMPUTE_PAIR_ALT_GEMM_CMAT64:
                        cand_exp_alt = HK_COMPUTE_PAIR_ALT_GEMM_CMAT64X64;
                        break;
                     default:
                        break;
                     }
                     if (hk_compute_pair_alt_is_gemm_unguarded_for(
                            cand_exp_alt, p->info.gemm.is_guarded) &&
                         hk_compute_pair_alt_gemm_tile_m(cand_exp_alt) == 64u &&
                         p->exec_threads[0] == p->exec_local[0] &&
                         dual != HK_COMPUTE_PAIR_DUAL_PAD4) {
                        unsigned ecur = found_dw;
                        unsigned found_exp_clamp = 0, found_exp = 0, found_exp_im2 = 0;
                        bool has_exp_clamp = false, has_exp = false, has_found_im2 = false;
                        bool exp_chain_ok = true;
                        while (ecur > start && found_dw - ecur < 6 && i - ecur < 16) {
                           ecur--;
                           if (plans[ecur].is_self_copy)
                              continue;
                           if (!members[ecur].cs ||
                               !hk_compute_pair_stream_eligible(members[ecur].cs)) {
                              exp_chain_ok = false;
                              break;
                           }
                           const uint32_t erole =
                              members[ecur].cs->compute_pair->info.role;
                           if (erole == HK_COMPUTE_PAIR_SELF_COPY)
                              continue;
                           if (!has_exp_clamp && erole == HK_COMPUTE_PAIR_CLAMP &&
                               !plans[ecur].shader) {
                              found_exp_clamp = ecur;
                              has_exp_clamp = true;
                              continue;
                           }
                           if (erole == HK_COMPUTE_PAIR_GEMM &&
                               !planned_suppressed[ecur] &&
                               (!plans[ecur].shader ||
                                (plans[ecur].dual_write == HK_COMPUTE_PAIR_DUAL_NONE &&
                                 !plans[ecur].dw_a_idx_off &&
                                 !plans[ecur].exp_idx_off &&
                                 ((plans[ecur].span == 2 && has_exp_clamp &&
                                   found_exp_clamp == ecur + 1) ||
                                  (plans[ecur].span == 1 && !has_exp_clamp &&
                                   plans[ecur].im2col_idx_off))))) {
                              found_exp = ecur;
                              has_exp = true;
                              if (plans[ecur].shader && plans[ecur].im2col_idx_off) {
                                 if (ecur >= start + plans[ecur].im2col_idx_off &&
                                     i - (ecur - plans[ecur].im2col_idx_off) < 16) {
                                    found_exp_im2 = ecur - plans[ecur].im2col_idx_off;
                                    has_found_im2 = true;
                                 } else {
                                    exp_chain_ok = false;
                                 }
                              }
                           } else {
                              exp_chain_ok = false;
                           }
                           break;
                        }
                        if (exp_chain_ok && has_exp && !has_found_im2 &&
                            !plans[found_exp].shader) {
                           unsigned icur = found_exp;
                           while (icur > start && found_exp - icur < 4 && i - icur < 16) {
                              icur--;
                              if (plans[icur].is_self_copy)
                                 continue;
                              if (planned_suppressed[icur] || plans[icur].shader)
                                 break;
                              if (members[icur].cs &&
                                  hk_compute_pair_stream_eligible(members[icur].cs) &&
                                  members[icur].cs->compute_pair->info.role ==
                                     HK_COMPUTE_PAIR_IM2COL) {
                                 found_exp_im2 = icur;
                                 has_found_im2 = true;
                              }
                              break;
                           }
                        }
                        if (exp_chain_ok && has_exp) {
                           const struct hk_pack_fuse_plan *epl = &plans[found_exp];
                           const unsigned first_chk = has_found_im2 ? found_exp_im2 : found_exp;
                           for (unsigned j = first_chk + 1; j < found_dw; ++j) {
                              if (j == found_exp)
                                 continue;
                              if (planned_suppressed[j] &&
                                  (!epl->shader || j < found_exp || j >= found_exp + epl->span)) {
                                 exp_chain_ok = false;
                                 break;
                              }
                           }
                        }
                        if (exp_chain_ok && has_exp) {
                           const struct hk_cs *exp_cs = members[found_exp].cs;
                           const struct hk_compute_pair_dispatch *exp_im2_d =
                              has_found_im2 ? members[found_exp_im2].cs->compute_pair : NULL;
                           const struct hk_nir_clamp_epilogue *exp_ep = NULL;
                           bool ep_ok = true;
                           if (has_exp_clamp) {
                              if (compute_pair_can_replace(exp_cs,
                                                           members[found_exp_clamp].cs))
                                 exp_ep = &members[found_exp_clamp].cs->compute_pair->info.clamp.epilogue;
                              else
                                 ep_ok = false;
                           }
                           if (ep_ok &&
                               compute_pair_exp_dw_gemm_match(
                                  exp_im2_d, exp_cs->compute_pair, exp_ep,
                                  dw_cs->compute_pair, dw_copy_d, p, dual,
                                  d1, d2, fadd_add_src_binding,
                                  &dw_a_cfg, &dw_a_push, &exp_dw_a_push)) {
                              has_exp_dw_a = true;
                              exp_idx = found_exp;
                              has_exp_im2 = has_found_im2;
                              exp_im2_idx = has_found_im2 ? found_exp_im2 : 0;
                              eff_alt_kind = cand_exp_alt;
                           }
                        }
                     }
                  }
               }
            }

            if (!ep && dual == HK_COMPUTE_PAIR_DUAL_NONE && !has_im2col && !has_dw_a)
               continue;

            struct hk_shader *fused = NULL;
            uint64_t new_run_work = run_work;
            unsigned new_run_loops = run_loops;
            uint64_t kept_orig_work = 0, kept_fused_work = 0;
            unsigned kept_orig_loops = 0, kept_fused_loops = 0;
            unsigned span = 0;
            for (unsigned attempt = 0; attempt < 8; ++attempt) {
               if (attempt > 0) {
                  if (dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW && gemm_dw_cfg.has_proj) {
                     gemm_dw_cfg.has_proj = false;
                     gemm_dw_cfg.proj_tile_n = 0;
                     gemm_dw_cfg.has_proj_epilogue = false;
                     gemm_dw_proj_idx = 0;
                     gemm_dw_proj_copy_idx = 0;
                     gemm_dw_last_idx = gemm_dw_pre_proj_last_idx;
                  } else if (dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW) {
                     eff_alt_kind = p->alt_kind;
                     gemm_dw_idx = 0;
                     gemm_dw_last_idx = 0;
                     gemm_dw_copy_idx = 0;
                     if (compute_pair_pad4_match(p, members[dual_idx].cs->compute_pair)) {
                        dual = HK_COMPUTE_PAIR_DUAL_PAD4;
                     } else if (ep) {
                        dual = HK_COMPUTE_PAIR_DUAL_NONE;
                        dual_idx = 0;
                     } else {
                        fused = NULL;
                        break;
                     }
                  } else if (has_exp_dw_a) {
                     has_exp_dw_a = false;
                     has_exp_im2 = false;
                     exp_idx = 0;
                     exp_im2_idx = 0;
                     dw_a_cfg.has_expand = false;
                     dw_a_cfg.has_exp_im2col = false;
                     dw_a_cfg.has_exp_epilogue = false;
                     eff_alt_kind = non_exp_alt_kind;
                  } else if (dual == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY) {
                     dual = HK_COMPUTE_PAIR_DUAL_FADD_EP;
                     dual2_idx = 0;
                  } else if (dual == HK_COMPUTE_PAIR_DUAL_COPY2) {
                     dual = HK_COMPUTE_PAIR_DUAL_COPY;
                     dual2_idx = 0;
                  } else if ((ep || has_im2col || has_dw_a) &&
                             dual != HK_COMPUTE_PAIR_DUAL_NONE) {
                     dual = HK_COMPUTE_PAIR_DUAL_NONE;
                     dual_idx = 0;
                     dual2_idx = 0;
                  } else if (has_im2col && ep) {
                     has_im2col = false;
                     im2_idx = 0;
                     eff_alt_kind = p->alt_kind;
                  } else if (has_dw_a && ep) {
                     has_dw_a = false;
                     dw_a_idx = 0;
                     dw_a_copy_idx = 0;
                     eff_alt_kind = p->alt_kind;
                  } else {
                     fused = NULL;
                     break;
                  }
               }
               const struct hk_nir_fadd_epilogue *cand_fadd_ep =
                  hk_compute_pair_dual_is_fadd_ep(dual) ? &fadd_ep : NULL;
               struct hk_shader *cand = hk_compute_pair_get_or_compile_fused(
                  dev, p->producer, ep, eff_alt_kind, dual, cand_fadd_ep,
                  has_im2col, has_dw_a ? &dw_a_cfg : NULL,
                  dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW ? &gemm_dw_cfg : NULL);
               struct hk_compute_pair_dispatch eff_p = *p;
               if (eff_alt_kind != p->alt_kind ||
                   (dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW && gemm_dw_cfg.has_proj)) {
                  eff_p.alt_kind = eff_alt_kind;
                  if (dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW) {
                     const uint32_t in_w_o = gemm_dw_push.in_wh & 0xffffu;
                     const uint32_t in_h_o = gemm_dw_push.in_wh >> 16;
                     const uint32_t gdw_tn = hk_compute_pair_alt_gemm_tile_n(eff_alt_kind);
                     int32_t n_o = 0;
                     memcpy(&n_o, p->push + p->info.gemm.n_offset_B, 4);
                     eff_p.exec_threads[0] = gemm_dw_cfg.has_proj
                        ? 128u : (((uint32_t)n_o / gdw_tn) * 128u);
                     eff_p.exec_threads[1] = (in_h_o >> 3) * (in_w_o >> 3);
                     eff_p.exec_threads[2] = 1u;
                  } else if (hk_compute_pair_alt_gemm_tile_m(eff_alt_kind) == 32u &&
                             hk_compute_pair_alt_gemm_tile_m(p->alt_kind) == 64u) {
                     eff_p.exec_threads[1] = p->exec_threads[1] * 2u;
                  } else if (hk_compute_pair_alt_gemm_tile_m(eff_alt_kind) == 64u &&
                             hk_compute_pair_alt_gemm_tile_m(p->alt_kind) == 32u) {
                     eff_p.exec_threads[1] = p->exec_threads[1] / 2u;
                  }
               }
               if (has_exp_dw_a) {
                  const uint32_t in_w_e = dw_a_push.in_wh & 0xffffu;
                  const uint32_t in_h_e = dw_a_push.in_wh >> 16;
                  eff_p.exec_threads[1] = (in_h_e >> 3) * (in_w_e >> 3);
               }
               if (!cand || !hk_fused_compute_shader_eligible(dev, cand, &eff_p))
                  continue;

               const unsigned cand_span =
                  dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW ? (gemm_dw_last_idx - i + 1)
                  : (dual == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY ||
                     dual == HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE ||
                     dual == HK_COMPUTE_PAIR_DUAL_FADD_COPY_OUT ? (dual2_idx - i + 1)
                     : (dual != HK_COMPUTE_PAIR_DUAL_NONE ? (dual_idx - i + 1)
                        : (ep ? 2u : 1u)));
               uint64_t replaced_work = 0;
               unsigned replaced_loops = 0;
               bool replaced_overflow = false;
               for (unsigned j = i; j < i + cand_span; ++j) {
                  if (members[j].cs) {
                     if (__builtin_add_overflow(replaced_work,
                                                members[j].cs->pack_static_work,
                                                &replaced_work)) {
                        replaced_overflow = true;
                        break;
                     }
                     if (members[j].cs->pack_loops && !members[j].obs_ns)
                        replaced_loops++;
                  }
               }
               if (!replaced_overflow && dual == HK_COMPUTE_PAIR_DUAL_COPY2 &&
                   members[dual2_idx].cs) {
                  if (__builtin_add_overflow(replaced_work,
                                             members[dual2_idx].cs->pack_static_work,
                                             &replaced_work)) {
                     replaced_overflow = true;
                  } else if (members[dual2_idx].cs->pack_loops &&
                             !members[dual2_idx].obs_ns) {
                     replaced_loops++;
                  }
               }
               if (!replaced_overflow && has_im2col && members[im2_idx].cs) {
                  if (__builtin_add_overflow(replaced_work,
                                             members[im2_idx].cs->pack_static_work,
                                             &replaced_work)) {
                     replaced_overflow = true;
                  } else if (members[im2_idx].cs->pack_loops &&
                             !members[im2_idx].obs_ns) {
                     replaced_loops++;
                  }
               }
               uint64_t eff_run_work = run_work;
               unsigned eff_run_loops = run_loops;
               if (!replaced_overflow && has_dw_a) {
                  if (plans[dw_a_idx].shader) {
                     eff_run_work =
                        eff_run_work >= plans[dw_a_idx].fused_work
                           ? eff_run_work - plans[dw_a_idx].fused_work : 0;
                     eff_run_work =
                        plans[dw_a_idx].orig_work > UINT64_MAX - eff_run_work
                           ? UINT64_MAX : eff_run_work + plans[dw_a_idx].orig_work;
                     eff_run_loops =
                        eff_run_loops >= plans[dw_a_idx].fused_loops
                           ? eff_run_loops - plans[dw_a_idx].fused_loops : 0;
                     eff_run_loops += plans[dw_a_idx].orig_loops;
                  }
                  if (has_exp_dw_a && plans[exp_idx].shader) {
                     eff_run_work =
                        eff_run_work >= plans[exp_idx].fused_work
                           ? eff_run_work - plans[exp_idx].fused_work : 0;
                     eff_run_work =
                        plans[exp_idx].orig_work > UINT64_MAX - eff_run_work
                           ? UINT64_MAX : eff_run_work + plans[exp_idx].orig_work;
                     eff_run_loops =
                        eff_run_loops >= plans[exp_idx].fused_loops
                           ? eff_run_loops - plans[exp_idx].fused_loops : 0;
                     eff_run_loops += plans[exp_idx].orig_loops;
                  }
                  const unsigned first_pred =
                     has_exp_im2 ? exp_im2_idx : (has_exp_dw_a ? exp_idx : dw_a_idx);
                  for (unsigned j = first_pred; j < i; ++j) {
                     if (members[j].cs) {
                        if (__builtin_add_overflow(replaced_work,
                                                   members[j].cs->pack_static_work,
                                                   &replaced_work)) {
                           replaced_overflow = true;
                           break;
                        }
                        if (members[j].cs->pack_loops && !members[j].obs_ns)
                           replaced_loops++;
                     }
                  }
               }
               if (replaced_overflow)
                  continue;

               const uint64_t fused_threads =
                  (uint64_t)eff_p.exec_threads[0] *
                  (uint64_t)eff_p.exec_threads[1] *
                  (uint64_t)eff_p.exec_threads[2];
               uint64_t fused_params[HK_PACK_LOOP_PARAM_COUNT];
               memcpy(fused_params, p->pack_loop_params, sizeof(fused_params));
               if (eff_alt_kind != p->alt_kind &&
                   hk_compute_pair_alt_gemm_k_step(eff_alt_kind) !=
                      hk_compute_pair_alt_gemm_k_step(p->alt_kind)) {
                  int32_t k_o = 0;
                  memcpy(&k_o, p->push + p->info.gemm.k_offset_B, 4);
                  const uint32_t eff_ks = hk_compute_pair_alt_gemm_k_step(eff_alt_kind);
                  fused_params[HK_PACK_LOOP_PARAM_GEMM_K_ITERS] =
                     DIV_ROUND_UP((uint32_t)k_o, eff_ks);
               }
               bool params_ok = true;
               if (dual == HK_COMPUTE_PAIR_DUAL_PAD4 ||
                   dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW) {
                  const struct hk_compute_pair_pad4 *pad =
                     &members[dual_idx].cs->compute_pair->extra.pad4;
                  fused_params[HK_PACK_LOOP_PARAM_PAD_H] =
                     pad->out_shape[1] > pad->in_shape[1]
                        ? (uint64_t)(pad->out_shape[1] - pad->in_shape[1]) : 0;
                  fused_params[HK_PACK_LOOP_PARAM_PAD_W] =
                     pad->out_shape[2] > pad->in_shape[2]
                        ? (uint64_t)(pad->out_shape[2] - pad->in_shape[2]) : 0;
                  uint64_t hw = 0, out_total = 0;
                  if (fused_threads == 0 ||
                      __builtin_mul_overflow((uint64_t)pad->out_shape[1],
                                             (uint64_t)pad->out_shape[2], &hw) ||
                      __builtin_mul_overflow(hw, (uint64_t)pad->out_shape[3],
                                             &out_total)) {
                     params_ok = false;
                  } else {
                     fused_params[HK_PACK_LOOP_PARAM_PAD_ZLP_ITERS] =
                        DIV_ROUND_UP(out_total, fused_threads);
                  }
                  if (dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW) {
                     int32_t k_i = 0, n_i = 0;
                     memcpy(&k_i, p->push + p->info.gemm.k_offset_B, 4);
                     memcpy(&n_i, p->push + p->info.gemm.n_offset_B, 4);
                     fused_params[HK_PACK_LOOP_PARAM_GEMM_K_ITERS] =
                        DIV_ROUND_UP((uint64_t)k_i, 16u);
                     fused_params[HK_PACK_LOOP_PARAM_GEMM_K] =
                        DIV_ROUND_UP((uint64_t)k_i, 4u);
                     if (gemm_dw_cfg.has_proj) {
                        const uint32_t gdw_tn =
                           hk_compute_pair_alt_gemm_tile_n(eff_alt_kind);
                        fused_params[HK_PACK_LOOP_PARAM_DW_KH] =
                           (uint64_t)((uint32_t)n_i / gdw_tn);
                     }
                  }
               }
               if (has_exp_dw_a) {
                  int32_t k_o = 0;
                  memcpy(&k_o, p->push + p->info.gemm.k_offset_B, 4);
                  fused_params[HK_PACK_LOOP_PARAM_GEMM_K_ITERS] =
                     DIV_ROUND_UP((uint64_t)k_o, 16u);
                  fused_params[HK_PACK_LOOP_PARAM_GEMM_K] =
                     DIV_ROUND_UP((uint64_t)exp_dw_a_push.exp_k, 8u);
               }

               uint64_t fused_eff_instrs = 0;
               if (params_ok)
                  hk_pack_eval_effective_instrs(&cand->pack_loop_bound,
                                                cand->b.info.stats.instrs,
                                                fused_params,
                                                &fused_eff_instrs);
               uint64_t fused_work = 0;
               const bool fused_loops = hk_pack_account_dispatch_work_eff_limit(
                  fused_threads, cand->b.info.stats.instrs,
                  cand->b.info.has_loops, fused_eff_instrs,
                  limits.max_static_work, &fused_work);

               if (!hk_pack_fusion_fits_run_limit(eff_run_work, eff_run_loops,
                                                  replaced_work, replaced_loops,
                                                  fused_work, fused_loops,
                                                  limits.max_static_work,
                                                  &new_run_work, &new_run_loops))
                  continue;

               fused = cand;
               span = cand_span;
               kept_orig_work = replaced_work;
               kept_fused_work = fused_work;
               kept_orig_loops = replaced_loops;
               kept_fused_loops = fused_loops ? 1u : 0u;
               break;
            }
            if (!fused)
               continue;
            if (has_dw_a) {
               if (plans[dw_a_idx].shader) {
                  if (plans[dw_a_idx].dual_write == HK_COMPUTE_PAIR_DUAL_COPY2 &&
                      dw_a_idx + plans[dw_a_idx].dual2_idx_off < end) {
                     planned_suppressed[dw_a_idx + plans[dw_a_idx].dual2_idx_off] = false;
                  }
                  fused_cand_count--;
                  ioto_total_est -= plans[dw_a_idx].ioto_bound;
                  if (plans[dw_a_idx].dual_write != HK_COMPUTE_PAIR_DUAL_NONE)
                     root_cand_count--;
                  memset(&plans[dw_a_idx], 0, sizeof(plans[dw_a_idx]));
               }
               if (has_exp_dw_a && plans[exp_idx].shader) {
                  fused_cand_count--;
                  ioto_total_est -= plans[exp_idx].ioto_bound;
                  if (plans[exp_idx].dual_write != HK_COMPUTE_PAIR_DUAL_NONE ||
                      plans[exp_idx].im2col_idx_off != 0)
                     root_cand_count--;
                  memset(&plans[exp_idx], 0, sizeof(plans[exp_idx]));
               }
               const unsigned first_pred =
                  has_exp_im2 ? exp_im2_idx : (has_exp_dw_a ? exp_idx : dw_a_idx);
               for (unsigned j = first_pred; j < i; ++j)
                  planned_suppressed[j] = true;
            }
            const size_t cand_ioto_bound = hk_fused_compute_ioto_bound(dev, fused, p);
            run_work = new_run_work;
            run_loops = new_run_loops;
            plans[i] = (struct hk_pack_fuse_plan){
               .shader = fused,
               .dual_write = dual,
               .alt_kind_override = eff_alt_kind != p->alt_kind
                                       ? eff_alt_kind
                                       : HK_COMPUTE_PAIR_ALT_NONE,
               .span = (uint8_t)span,
               .dual_idx_off = dual != HK_COMPUTE_PAIR_DUAL_NONE
                                  ? (uint8_t)(dual_idx - i) : 0,
               .dual2_idx_off = (dual == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY ||
                                 dual == HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE ||
                                 dual == HK_COMPUTE_PAIR_DUAL_FADD_COPY_OUT ||
                                 dual == HK_COMPUTE_PAIR_DUAL_COPY2)
                                   ? (uint8_t)(dual2_idx - i) : 0,
               .im2col_idx_off = has_im2col ? (uint8_t)(i - im2_idx) : 0,
               .dw_a_idx_off = has_dw_a ? (uint8_t)(i - dw_a_idx) : 0,
               .dw_a_copy_idx_off = (has_dw_a && dw_a_copy_idx)
                                       ? (uint8_t)(i - dw_a_copy_idx) : 0,
               .exp_idx_off = has_exp_dw_a ? (uint8_t)(i - exp_idx) : 0,
               .exp_im2_idx_off = has_exp_im2 ? (uint8_t)(i - exp_im2_idx) : 0,
               .gemm_dw_idx_off = dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW
                                     ? (uint8_t)(gemm_dw_idx - i) : 0,
               .gemm_dw_copy_idx_off =
                  (dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW && gemm_dw_copy_idx)
                     ? (uint8_t)(gemm_dw_copy_idx - i) : 0,
               .gemm_dw_proj_idx_off =
                  (dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW && gemm_dw_proj_idx)
                     ? (uint8_t)(gemm_dw_proj_idx - i) : 0,
               .gemm_dw_proj_copy_idx_off =
                  (dual == HK_COMPUTE_PAIR_DUAL_GEMM_DW && gemm_dw_proj_copy_idx)
                     ? (uint8_t)(gemm_dw_proj_copy_idx - i) : 0,
               .fadd_add_src_binding = fadd_add_src_binding,
               .orig_loops = (uint8_t)MIN2(kept_orig_loops, 255u),
               .fused_loops = (uint8_t)kept_fused_loops,
               .ioto_bound = (uint16_t)MIN2(cand_ioto_bound, UINT16_MAX),
               .is_self_copy = false,
               .orig_work = kept_orig_work,
               .fused_work = kept_fused_work,
               .dw_a_push = dw_a_push,
               .exp_dw_a_push = exp_dw_a_push,
               .gemm_dw_push = gemm_dw_push,
               .gemm_dw_proj_push = gemm_dw_proj_push,
            };
            for (unsigned j = i + 1; j < i + span; ++j)
               planned_suppressed[j] = true;
            if (dual == HK_COMPUTE_PAIR_DUAL_COPY2)
               planned_suppressed[dual2_idx] = true;
            if (has_im2col)
               planned_suppressed[im2_idx] = true;
            fused_cand_count++;
            ioto_total_est += plans[i].ioto_bound;
            if (dual != HK_COMPUTE_PAIR_DUAL_NONE || has_im2col || has_dw_a)
               root_cand_count++;
            i += span - 1;
         }
      }
   }
   const int64_t t1_ns = trace ? os_time_get_nano() : 0;

   /* Reap already-retired submissions before allocating packet BOs so the
    * previous iteration's packed/Ioto/root BOs can hit the BO cache directly. */
   if (!dev->dev.is_virtio) {
      hk_cmd_pending_reap_cached(dev, queue,
                                 done_cache ? &done_cache->valid : NULL,
                                 done_cache ? &done_cache->point : NULL);
      if (done_cache)
         done_cache->reaped = true;
   }

   const bool cached = hk_g17p_cached_cmd_memory(&dev->dev);
   out->bo = agx_bo_create(&dev->dev, total, 0,
                           cached ? AGX_BO_WRITEBACK : 0,
                           "Packed compute submit");
   uint8_t *map = out->bo ? agx_bo_map(out->bo) : NULL;
   struct vk_command_buffer **commands =
      hk_queue_pack_scratch(&queue->pack.commands,
                            n - packed_members + run_count, sizeof(*commands));
   if (!map || !commands || !hk_queue_pack_ensure_packets(queue, run_count) ||
       !hk_pack_add_bo(out->extra_bos, out->bo)) {
      hk_packed_runs_free(dev, out);
      memset(out, 0, sizeof(*out));
      return false;
   }
   out->packets = queue->pack.packets;
   out->commands = commands;

   uint8_t *ioto_map = NULL;
   uint64_t ioto_va = 0;
   size_t ioto_off = 0, ioto_total = 0;
   if (fused_cand_count > 0 && ioto_total_est > 0) {
      ioto_total = ALIGN_POT(ioto_total_est, 4096u);
      out->ioto_bo = agx_bo_create(&dev->dev, ioto_total,
                                   0, AGX_BO_EXEC | AGX_BO_LOW_VA |
                                      (cached ? AGX_BO_WRITEBACK : 0),
                                   "Packed compute fused Ioto");
      ioto_map = out->ioto_bo ? agx_bo_map(out->ioto_bo) : NULL;
      ioto_va = (out->ioto_bo && out->ioto_bo->va) ? out->ioto_bo->va->addr : 0;
      if (!ioto_map || !ioto_va ||
          !agx_va_in_usc_window(&dev->dev, ioto_va) ||
          !agx_va_in_usc_window(&dev->dev, ioto_va + ioto_total - 1) ||
          !hk_pack_add_bo(out->extra_bos, out->ioto_bo)) {
         /* Fall back cleanly to unfused packing. */
         for (unsigned k = 0; k < n; ++k) {
            plans[k].shader = NULL;
            plans[k].dual_write = HK_COMPUTE_PAIR_DUAL_NONE;
            plans[k].alt_kind_override = HK_COMPUTE_PAIR_ALT_NONE;
            plans[k].span = 0;
            plans[k].im2col_idx_off = 0;
            plans[k].dw_a_idx_off = 0;
            plans[k].dw_a_copy_idx_off = 0;
            plans[k].exp_idx_off = 0;
            plans[k].exp_im2_idx_off = 0;
            plans[k].gemm_dw_idx_off = 0;
            plans[k].gemm_dw_copy_idx_off = 0;
            plans[k].gemm_dw_proj_idx_off = 0;
            plans[k].gemm_dw_proj_copy_idx_off = 0;
         }
         ioto_map = NULL;
         ioto_va = 0;
      }
   }

   const size_t root_stride = ALIGN_POT(sizeof(struct hk_root_descriptor_table), 64u);
   uint8_t *root_map = NULL;
   uint64_t root_va = 0;
   size_t root_off = 0, root_total = 0;
   if (root_cand_count > 0 && ioto_map) {
      root_total = ALIGN_POT((size_t)root_cand_count * root_stride, 4096u);
      out->root_bo = agx_bo_create(&dev->dev, root_total, 0,
                                   cached ? AGX_BO_WRITEBACK : 0,
                                   "Packed compute dual-write root");
      root_map = out->root_bo ? agx_bo_map(out->root_bo) : NULL;
      root_va = (out->root_bo && out->root_bo->va) ? out->root_bo->va->addr : 0;
      if (!root_map || !root_va || !hk_pack_add_bo(out->extra_bos, out->root_bo)) {
         for (unsigned k = 0; k < n; ++k) {
            if (plans[k].dual_write != HK_COMPUTE_PAIR_DUAL_NONE ||
                plans[k].im2col_idx_off || plans[k].dw_a_idx_off) {
               plans[k].shader = NULL;
               plans[k].dual_write = HK_COMPUTE_PAIR_DUAL_NONE;
               plans[k].alt_kind_override = HK_COMPUTE_PAIR_ALT_NONE;
               plans[k].span = 0;
               plans[k].im2col_idx_off = 0;
               plans[k].dw_a_idx_off = 0;
               plans[k].dw_a_copy_idx_off = 0;
               plans[k].exp_idx_off = 0;
               plans[k].exp_im2_idx_off = 0;
               plans[k].gemm_dw_idx_off = 0;
               plans[k].gemm_dw_copy_idx_off = 0;
            }
         }
         root_map = NULL;
         root_va = 0;
      }
   }
   const int64_t t2_ns = trace ? os_time_get_nano() : 0;
   int64_t live_ns = 0;

   unsigned r = 0, fused_pairs = 0, im2a_fused = 0, dwa_fused = 0, expdwa_fused = 0, gemm_dw_fused = 0, gdwproj_fused = 0;
   unsigned self_elided = 0;
   unsigned flushes_elided = 0, barriers_elided = 0, dead_writes_elided = 0;
   for (unsigned i = 0; i < n;) {
      if (r == run_count || i != runs[r].first) {
         out->commands[out->command_count++] = submit->command_buffers[i++];
         continue;
      }

      const struct hk_pack_run *pr = &runs[r];
      struct hk_packed_submit *packet = &out->packets[out->packet_count++];
      struct hk_cmd_buffer *cmd = &packet->cmd;
      struct hk_cs *cs = &packet->cs;
      hk_pack_init_packet(cmd, cs, dev, map + pr->offset,
                          out->bo->va->addr + pr->offset, pr->bytes,
                          visible_completion);

      bool has_non_self = false;
      for (unsigned m = 0; m < pr->count; ++m) {
         const unsigned idx = pr->first + m;
         const struct hk_cmd_buffer *source =
            (const void *)submit->command_buffers[idx];
         cs->sync.kind |= list_first_entry(&source->control_streams,
                                           struct hk_cs, node)->sync.kind;
         if (!plans[idx].is_self_copy)
            has_non_self = true;
         /* Pre-suppress preceding im2col/dw_a members fused into a later GEMM
          * so the forward emission loop skips them before reaching the GEMM. */
         const struct hk_pack_fuse_plan *pl = &plans[idx];
         if (ioto_map && root_map && pl->shader && pl->im2col_idx_off &&
             m >= pl->im2col_idx_off && m + pl->span <= pr->count &&
             (pl->dual_write != HK_COMPUTE_PAIR_DUAL_COPY2 ||
              m + pl->dual2_idx_off < pr->count)) {
            suppressed_arr[idx - pl->im2col_idx_off] = true;
         }
         if (ioto_map && root_map && pl->shader && pl->dw_a_idx_off &&
             m >= pl->dw_a_idx_off && (!pl->exp_idx_off || m >= pl->exp_idx_off) &&
             (!pl->exp_im2_idx_off || m >= pl->exp_im2_idx_off) &&
             m + pl->span <= pr->count &&
             (pl->dual_write != HK_COMPUTE_PAIR_DUAL_COPY2 ||
              m + pl->dual2_idx_off < pr->count)) {
            const unsigned first_pred =
               pl->exp_im2_idx_off ? (idx - pl->exp_im2_idx_off)
               : (pl->exp_idx_off ? (idx - pl->exp_idx_off) : (idx - pl->dw_a_idx_off));
            for (unsigned j = first_pred; j < idx; ++j)
               suppressed_arr[j] = true;
         }
      }

      /* Backward liveness pass within this packed run: mark intermediate
       * dual-write / im2col_a / dw_a output buffers as dead when provably
       * completely overwritten later in the same run before any read. Skipped
       * under alias_scope (host-visible memory aliasing) or when no BO handle
       * is written more than once in the run (provably zero dead overwrites). */
      if (!hk_cdm_alias_scope(dev) && ioto_map && root_map && pr->count > 1) {
         const int64_t lt0_ns = trace ? os_time_get_nano() : 0;
         uint32_t seen_wh[512] = {0};
         bool any_dup_write = false;
         for (unsigned m = 0; m < pr->count && !any_dup_write; ++m) {
            const unsigned idx = pr->first + m;
            if (has_non_self && plans[idx].is_self_copy)
               continue;
            const struct hk_cmd_buffer *src_m =
               (const void *)submit->command_buffers[idx];
            const struct hk_cs *cs_m =
               list_first_entry(&src_m->control_streams, struct hk_cs, node);
            const struct hk_compute_pair_dispatch *cp_m =
               hk_compute_pair_stream_eligible(cs_m) ? cs_m->compute_pair : NULL;
            if (!cp_m)
               continue;
            uint32_t wh = 0;
            switch (cp_m->info.role) {
            case HK_COMPUTE_PAIR_GEMM:
            case HK_COMPUTE_PAIR_DEPTHWISE:
               wh = cp_m->bindings[3].handle;
               break;
            case HK_COMPUTE_PAIR_COPY4:
            case HK_COMPUTE_PAIR_IM2COL:
               wh = cp_m->bindings[1].handle;
               break;
            case HK_COMPUTE_PAIR_PAD4:
            case HK_COMPUTE_PAIR_FADD:
               wh = cp_m->bindings[2].handle;
               break;
            default:
               continue;
            }
            if (!wh)
               continue;
            const uint32_t key = wh + 1u;
            unsigned slot = (wh * 2654435761u) & 511u;
            bool inserted = false;
            for (unsigned p = 0; p < 16u; ++p) {
               const uint32_t cur = seen_wh[(slot + p) & 511u];
               if (cur == 0) {
                  seen_wh[(slot + p) & 511u] = key;
                  inserted = true;
                  break;
               }
               if (cur == key) {
                  any_dup_write = true;
                  inserted = true;
                  break;
               }
            }
            if (!inserted)
               any_dup_write = true;
         }
         bool *run_sup = any_dup_write
            ? hk_queue_pack_zscratch(&queue->pack.run_sup, pr->count,
                                     sizeof(*run_sup))
            : NULL;
         if (run_sup) {
            for (unsigned m = 0; m < pr->count; ++m) {
               const unsigned idx = pr->first + m;
               if (has_non_self && plans[idx].is_self_copy)
                  run_sup[m] = true;
               const struct hk_pack_fuse_plan *pl = &plans[idx];
               const bool will_fuse =
                  pl->shader && m + pl->span <= pr->count &&
                  (!pl->im2col_idx_off || m >= pl->im2col_idx_off) &&
                  (!pl->dw_a_idx_off || m >= pl->dw_a_idx_off) &&
                  (!pl->exp_idx_off || m >= pl->exp_idx_off) &&
                  (!pl->exp_im2_idx_off || m >= pl->exp_im2_idx_off) &&
                  (pl->dual_write != HK_COMPUTE_PAIR_DUAL_COPY2 ||
                   m + pl->dual2_idx_off < pr->count);
               if (will_fuse) {
                  if (pl->im2col_idx_off)
                     run_sup[m - pl->im2col_idx_off] = true;
                  if (pl->dw_a_idx_off) {
                     const unsigned first_pred =
                        pl->exp_im2_idx_off ? (m - pl->exp_im2_idx_off)
                        : (pl->exp_idx_off ? (m - pl->exp_idx_off) : (m - pl->dw_a_idx_off));
                     for (unsigned j = first_pred; j < m; ++j)
                        run_sup[j] = true;
                  }
                  for (unsigned k = 1; k < pl->span; ++k)
                     run_sup[m + k] = true;
                  if (pl->dual_write == HK_COMPUTE_PAIR_DUAL_COPY2)
                     run_sup[m + pl->dual2_idx_off] = true;
               }
            }

            struct hk_pack_liveness_state liveness;
            hk_pack_liveness_reset(&liveness);
            for (unsigned m = pr->count; m-- > 0;) {
               if (run_sup[m])
                  continue;
               const unsigned idx = pr->first + m;
               struct hk_pack_fuse_plan *pl = &plans[idx];
               const bool will_fuse =
                  pl->shader && m + pl->span <= pr->count &&
                  (!pl->im2col_idx_off || m >= pl->im2col_idx_off) &&
                  (!pl->dw_a_idx_off || m >= pl->dw_a_idx_off) &&
                  (!pl->exp_idx_off || m >= pl->exp_idx_off) &&
                  (!pl->exp_im2_idx_off || m >= pl->exp_im2_idx_off) &&
                  (pl->dual_write != HK_COMPUTE_PAIR_DUAL_COPY2 ||
                   m + pl->dual2_idx_off < pr->count);

               if (will_fuse && liveness.num_dead > 0 &&
                   (pl->dual_write != HK_COMPUTE_PAIR_DUAL_NONE ||
                    pl->im2col_idx_off != 0 || pl->dw_a_idx_off != 0)) {
                  const struct hk_cmd_buffer *src =
                     (const void *)submit->command_buffers[idx];
                  const struct hk_cs *item =
                     list_first_entry(&src->control_streams, struct hk_cs, node);
                  struct hk_compute_pair_footprint base;
                  if (hk_compute_pair_stream_eligible(item) &&
                      !item->uses_global_atomics &&
                      compute_pair_dispatch_footprint(item->compute_pair, &base)) {
                     /* The residual-FADD post-pass writes C unconditionally. */
                     if (pl->dual_write != HK_COMPUTE_PAIR_DUAL_NONE &&
                         !hk_compute_pair_dual_is_fadd(pl->dual_write) &&
                         item->compute_pair->info.role != HK_COMPUTE_PAIR_DEPTHWISE &&
                         base.num_writes == 1) {
                        const struct hk_pack_interval c_iv = {
                           .start = base.writes[0].start,
                           .end = base.writes[0].end,
                           .handle = base.writes[0].handle,
                        };
                        if (hk_pack_liveness_covers(&liveness, &c_iv)) {
                           pl->skip_c = true;
                           dead_writes_elided++;
                        }
                     }
                     if (pl->dual_write == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY) {
                        const struct hk_cmd_buffer *d_src =
                           (const void *)submit->command_buffers[idx + pl->dual_idx_off];
                        const struct hk_cs *d_cs =
                           list_first_entry(&d_src->control_streams, struct hk_cs, node);
                        const struct hk_compute_pair_dispatch *d_disp = d_cs->compute_pair;
                        if (d_disp && d_disp->info.role == HK_COMPUTE_PAIR_FADD) {
                           const struct hk_compute_pair_copy4 *fcp = &d_disp->extra.copy4;
                           struct hk_compute_pair_interval fiv;
                           if (fcp->dst_start <= UINT64_MAX - fcp->bytes &&
                               compute_pair_interval(&d_disp->bindings[2],
                                                     fcp->dst_start,
                                                     fcp->dst_start + fcp->bytes, &fiv)) {
                              const struct hk_pack_interval pfiv = {
                                 .start = fiv.start,
                                 .end = fiv.end,
                                 .handle = fiv.handle,
                              };
                              if (hk_pack_liveness_covers(&liveness, &pfiv)) {
                                 pl->skip_fadd_dst = true;
                                 dead_writes_elided++;
                              }
                           }
                        }
                     }
                     if (pl->dual_write == HK_COMPUTE_PAIR_DUAL_GEMM_DW) {
                        const struct hk_cmd_buffer *p_src =
                           (const void *)submit->command_buffers[idx + pl->dual_idx_off];
                        const struct hk_cs *p_cs =
                           list_first_entry(&p_src->control_streams, struct hk_cs, node);
                        const struct hk_compute_pair_dispatch *p_disp = p_cs->compute_pair;
                        if (p_disp && p_disp->info.role == HK_COMPUTE_PAIR_PAD4) {
                           const struct hk_compute_pair_pad4 *pd = &p_disp->extra.pad4;
                           struct hk_compute_pair_interval piv;
                           if (compute_pair_interval(&p_disp->bindings[2], 0,
                                                     pd->out_bytes, &piv)) {
                              const struct hk_pack_interval ppiv = {
                                 .start = piv.start,
                                 .end = piv.end,
                                 .handle = piv.handle,
                              };
                              if (hk_pack_liveness_covers(&liveness, &ppiv)) {
                                 pl->skip_pad_dst = true;
                                 dead_writes_elided++;
                              }
                           }
                        }
                        if (pl->gemm_dw_idx_off) {
                           const struct hk_cmd_buffer *dw_src =
                              (const void *)submit->command_buffers[idx + pl->gemm_dw_idx_off];
                           const struct hk_cs *dw_cs =
                              list_first_entry(&dw_src->control_streams, struct hk_cs, node);
                           const struct hk_compute_pair_dispatch *dw_disp = dw_cs->compute_pair;
                           struct hk_compute_pair_footprint dw_fp;
                           if (dw_disp && dw_disp->info.role == HK_COMPUTE_PAIR_DEPTHWISE &&
                               compute_pair_dispatch_footprint(dw_disp, &dw_fp) &&
                               dw_fp.num_writes == 1) {
                              const struct hk_pack_interval d1iv = {
                                 .start = dw_fp.writes[0].start,
                                 .end = dw_fp.writes[0].end,
                                 .handle = dw_fp.writes[0].handle,
                              };
                              if (hk_pack_liveness_covers(&liveness, &d1iv)) {
                                 pl->skip_dw_dst1 = true;
                                 dead_writes_elided++;
                              }
                           }
                        }
                        if (pl->gemm_dw_copy_idx_off) {
                           const struct hk_cmd_buffer *dc_src =
                              (const void *)submit->command_buffers[idx + pl->gemm_dw_copy_idx_off];
                           const struct hk_cs *dc_cs =
                              list_first_entry(&dc_src->control_streams, struct hk_cs, node);
                           const struct hk_compute_pair_dispatch *dc_disp = dc_cs->compute_pair;
                           if (dc_disp && dc_disp->info.role == HK_COMPUTE_PAIR_COPY4) {
                              const struct hk_compute_pair_copy4 *cp = &dc_disp->extra.copy4;
                              struct hk_compute_pair_interval d2;
                              if (cp->dst_start <= UINT64_MAX - cp->bytes &&
                                  compute_pair_interval(&dc_disp->bindings[1],
                                                        cp->dst_start,
                                                        cp->dst_start + cp->bytes, &d2)) {
                                 const struct hk_pack_interval d2iv = {
                                    .start = d2.start,
                                    .end = d2.end,
                                    .handle = d2.handle,
                                 };
                                 if (hk_pack_liveness_covers(&liveness, &d2iv)) {
                                    pl->skip_dw_dst2 = true;
                                    dead_writes_elided++;
                                 }
                              }
                           }
                        }
                        if (pl->gemm_dw_proj_idx_off) {
                           const struct hk_cmd_buffer *pj_src =
                              (const void *)submit->command_buffers[idx + pl->gemm_dw_proj_idx_off];
                           const struct hk_cs *pj_cs =
                              list_first_entry(&pj_src->control_streams, struct hk_cs, node);
                           const struct hk_compute_pair_dispatch *pj_disp = pj_cs->compute_pair;
                           struct hk_compute_pair_footprint pj_fp;
                           if (pj_disp && pj_disp->info.role == HK_COMPUTE_PAIR_GEMM &&
                               compute_pair_dispatch_footprint(pj_disp, &pj_fp) &&
                               pj_fp.num_writes == 1) {
                              const struct hk_pack_interval pciv = {
                                 .start = pj_fp.writes[0].start,
                                 .end = pj_fp.writes[0].end,
                                 .handle = pj_fp.writes[0].handle,
                              };
                              if (hk_pack_liveness_covers(&liveness, &pciv)) {
                                 pl->skip_proj_c = true;
                                 dead_writes_elided++;
                              }
                           }
                           if (pl->gemm_dw_proj_copy_idx_off) {
                              const struct hk_cmd_buffer *pc_src =
                                 (const void *)submit->command_buffers[idx + pl->gemm_dw_proj_copy_idx_off];
                              const struct hk_cs *pc_cs =
                                 list_first_entry(&pc_src->control_streams, struct hk_cs, node);
                              const struct hk_compute_pair_dispatch *pc_disp = pc_cs->compute_pair;
                              if (pc_disp && pc_disp->info.role == HK_COMPUTE_PAIR_COPY4) {
                                 const struct hk_compute_pair_copy4 *cp = &pc_disp->extra.copy4;
                                 struct hk_compute_pair_interval pcp;
                                 if (cp->dst_start <= UINT64_MAX - cp->bytes &&
                                     compute_pair_interval(&pc_disp->bindings[1],
                                                           cp->dst_start,
                                                           cp->dst_start + cp->bytes, &pcp)) {
                                    const struct hk_pack_interval pcpiv = {
                                       .start = pcp.start,
                                       .end = pcp.end,
                                       .handle = pcp.handle,
                                    };
                                    if (hk_pack_liveness_covers(&liveness, &pcpiv)) {
                                       pl->skip_proj_copy = true;
                                       dead_writes_elided++;
                                    }
                                 }
                              }
                           }
                        }
                     }
                     if (pl->im2col_idx_off) {
                        const struct hk_cmd_buffer *im_src =
                           (const void *)submit->command_buffers[idx - pl->im2col_idx_off];
                        const struct hk_cs *im_cs =
                           list_first_entry(&im_src->control_streams, struct hk_cs, node);
                        const struct hk_compute_pair_dispatch *im_disp = im_cs->compute_pair;
                        struct hk_compute_pair_interval div;
                        bool have_div = false;
                        if (im_disp && im_disp->info.role == HK_COMPUTE_PAIR_IM2COL) {
                           const struct hk_compute_pair_im2col *im = &im_disp->extra.im2col;
                           have_div = compute_pair_interval(&im_disp->bindings[1], 0,
                                                            im->dst_bytes, &div);
                        } else if (im_disp && im_disp->info.role == HK_COMPUTE_PAIR_COPY4) {
                           const struct hk_compute_pair_copy4 *cp = &im_disp->extra.copy4;
                           have_div = cp->dst_start <= UINT64_MAX - cp->bytes &&
                                      compute_pair_interval(&im_disp->bindings[1],
                                                            cp->dst_start,
                                                            cp->dst_start + cp->bytes, &div);
                        }
                        if (have_div) {
                           const struct hk_pack_interval pdiv = {
                              .start = div.start,
                              .end = div.end,
                              .handle = div.handle,
                           };
                           if (hk_pack_liveness_covers(&liveness, &pdiv)) {
                              pl->skip_im2_dst = true;
                              dead_writes_elided++;
                           }
                        }
                     }
                     if (pl->dw_a_idx_off) {
                        const struct hk_cmd_buffer *dw_src =
                           (const void *)submit->command_buffers[idx - pl->dw_a_idx_off];
                        const struct hk_cs *dw_cs =
                           list_first_entry(&dw_src->control_streams, struct hk_cs, node);
                        const struct hk_compute_pair_dispatch *dw_disp = dw_cs->compute_pair;
                        struct hk_compute_pair_footprint dw_fp;
                        if (dw_disp && dw_disp->info.role == HK_COMPUTE_PAIR_DEPTHWISE &&
                            compute_pair_dispatch_footprint(dw_disp, &dw_fp) &&
                            dw_fp.num_writes == 1) {
                           const struct hk_pack_interval d1iv = {
                              .start = dw_fp.writes[0].start,
                              .end = dw_fp.writes[0].end,
                              .handle = dw_fp.writes[0].handle,
                           };
                           if (hk_pack_liveness_covers(&liveness, &d1iv)) {
                              pl->skip_dw_dst1 = true;
                              dead_writes_elided++;
                           }
                        }
                        if (pl->dw_a_copy_idx_off) {
                           const struct hk_cmd_buffer *dc_src =
                              (const void *)submit->command_buffers[idx - pl->dw_a_copy_idx_off];
                           const struct hk_cs *dc_cs =
                              list_first_entry(&dc_src->control_streams, struct hk_cs, node);
                           const struct hk_compute_pair_dispatch *dc_disp = dc_cs->compute_pair;
                           if (dc_disp && dc_disp->info.role == HK_COMPUTE_PAIR_COPY4) {
                              const struct hk_compute_pair_copy4 *cp = &dc_disp->extra.copy4;
                              struct hk_compute_pair_interval d2;
                              if (cp->dst_start <= UINT64_MAX - cp->bytes &&
                                  compute_pair_interval(&dc_disp->bindings[1],
                                                        cp->dst_start,
                                                        cp->dst_start + cp->bytes, &d2)) {
                                 const struct hk_pack_interval d2iv = {
                                    .start = d2.start,
                                    .end = d2.end,
                                    .handle = d2.handle,
                                 };
                                 if (hk_pack_liveness_covers(&liveness, &d2iv)) {
                                    pl->skip_dw_dst2 = true;
                                    dead_writes_elided++;
                                 }
                              }
                           }
                        }
                        if (pl->exp_idx_off) {
                           const struct hk_cmd_buffer *exp_src =
                              (const void *)submit->command_buffers[idx - pl->exp_idx_off];
                           const struct hk_cs *exp_cs =
                              list_first_entry(&exp_src->control_streams, struct hk_cs, node);
                           const struct hk_compute_pair_dispatch *exp_disp = exp_cs->compute_pair;
                           struct hk_compute_pair_footprint exp_fp;
                           if (exp_disp && exp_disp->info.role == HK_COMPUTE_PAIR_GEMM &&
                               compute_pair_dispatch_footprint(exp_disp, &exp_fp) &&
                               exp_fp.num_writes == 1) {
                              const struct hk_pack_interval eiv = {
                                 .start = exp_fp.writes[0].start,
                                 .end = exp_fp.writes[0].end,
                                 .handle = exp_fp.writes[0].handle,
                              };
                              if (hk_pack_liveness_covers(&liveness, &eiv)) {
                                 pl->skip_exp_c = true;
                                 dead_writes_elided++;
                              }
                           }
                        }
                        if (pl->exp_im2_idx_off) {
                           const struct hk_cmd_buffer *eim_src =
                              (const void *)submit->command_buffers[idx - pl->exp_im2_idx_off];
                           const struct hk_cs *eim_cs =
                              list_first_entry(&eim_src->control_streams, struct hk_cs, node);
                           const struct hk_compute_pair_dispatch *eim_disp = eim_cs->compute_pair;
                           struct hk_compute_pair_interval div;
                           if (eim_disp && eim_disp->info.role == HK_COMPUTE_PAIR_IM2COL &&
                               compute_pair_interval(&eim_disp->bindings[1], 0,
                                                     eim_disp->extra.im2col.dst_bytes, &div)) {
                              const struct hk_pack_interval pdiv = {
                                 .start = div.start,
                                 .end = div.end,
                                 .handle = div.handle,
                              };
                              if (hk_pack_liveness_covers(&liveness, &pdiv)) {
                                 pl->skip_exp_im2_dst = true;
                                 dead_writes_elided++;
                              }
                           }
                        }
                     }
                  }
               }

               struct hk_pack_footprint fp;
               compute_pair_emitted_footprint(submit, plans, idx, will_fuse,
                                              /*for_liveness=*/true, &fp);
               hk_pack_liveness_step_backward(&liveness, &fp);
            }
         }
         if (trace)
            live_ns += os_time_get_nano() - lt0_ns;
      }

      uint8_t preceding_kinds = 0;
      struct hk_pack_hazard_state hazard = {0};
      struct hk_pack_footprint cached_fp;
      unsigned cached_fp_idx = UINT32_MAX;
      bool cached_fp_fused = false;
      for (unsigned m = 0; m < pr->count; ++m) {
         const unsigned idx = pr->first + m;
         const struct hk_cmd_buffer *source =
            (const void *)submit->command_buffers[idx];
         const struct hk_cs *item = list_first_entry(&source->control_streams,
                                                     struct hk_cs, node);
         const bool is_suppressed_self = has_non_self && plans[idx].is_self_copy;
         const bool suppressed = suppressed_arr[idx] || is_suppressed_self;
         const bool last = m + 1 == pr->count;

         if (suppressed) {
            if (is_suppressed_self && !suppressed_arr[idx])
               self_elided++;
            hk_pack_fold_attachments(cs, item);
         } else {
            bool fused = false;
            const struct hk_pack_fuse_plan *pl = &plans[idx];
            struct hk_shader *fused_shader =
               (ioto_map && pl->shader && m + pl->span <= pr->count &&
                (!pl->im2col_idx_off || (root_map && m >= pl->im2col_idx_off)) &&
                (!pl->dw_a_idx_off || (root_map && m >= pl->dw_a_idx_off)) &&
                (!pl->exp_idx_off || (root_map && m >= pl->exp_idx_off)) &&
                (!pl->exp_im2_idx_off || (root_map && m >= pl->exp_im2_idx_off)) &&
                (pl->dual_write != HK_COMPUTE_PAIR_DUAL_COPY2 ||
                 m + pl->dual2_idx_off < pr->count))
                  ? pl->shader : NULL;
            const bool needs_root =
               pl->dual_write != HK_COMPUTE_PAIR_DUAL_NONE ||
               pl->im2col_idx_off != 0 || pl->dw_a_idx_off != 0;
            if (fused_shader && ioto_off < ioto_total &&
                (!needs_root || (root_map && root_off + root_stride <= root_total))) {
               const struct hk_compute_pair_dispatch *p_disp = item->compute_pair;
               struct hk_compute_pair_dispatch tmp_disp = *p_disp;
               bool root_ok = true;
               if (needs_root) {
                  struct hk_root_descriptor_table *rt =
                     (struct hk_root_descriptor_table *)(root_map + root_off);
                  rt->root_desc_addr = root_va + root_off;
                  memcpy(rt->sets, p_disp->root_sets, sizeof(rt->sets));
                  memset(&rt->cs, 0, sizeof(rt->cs));
                  rt->cs.group_count_addr = p_disp->cs_group_count_addr;
                  memcpy(rt->cs.base_group, p_disp->cs_base_group,
                         sizeof(rt->cs.base_group));
                  rt->printf_buffer_address = p_disp->printf_buffer_address;
                  memcpy(rt->push, p_disp->push, sizeof(rt->push));

                  const uint32_t prod_pb =
                     p_disp->info.role == HK_COMPUTE_PAIR_DEPTHWISE
                        ? p_disp->info.depthwise.push_bytes
                        : p_disp->info.role == HK_COMPUTE_PAIR_FADD
                             ? p_disp->info.fadd.push_bytes
                             : p_disp->info.gemm.push_bytes;
                  if (pl->dual_write != HK_COMPUTE_PAIR_DUAL_NONE) {
                     const struct hk_cmd_buffer *d_src =
                        (const void *)submit->command_buffers[idx + pl->dual_idx_off];
                     const struct hk_cs *d_cs =
                        list_first_entry(&d_src->control_streams, struct hk_cs, node);
                     const struct hk_compute_pair_dispatch *d_disp = d_cs->compute_pair;
                     const uint32_t dual_off = hk_dual_write_push_offset(prod_pb);
                     const uint32_t dual_flags =
                        (pl->skip_c ? HK_DUAL_WRITE_FLAG_SKIP_C : 0u) |
                        (pl->skip_fadd_dst ? HK_DUAL_WRITE_FLAG_SKIP_FADD_DST : 0u);
                     if (pl->dual_write == HK_COMPUTE_PAIR_DUAL_COPY) {
                        const uint64_t dst2_va =
                           d_disp->bindings[1].address + d_disp->extra.copy4.dst_start;
                        struct hk_dual_copy_push dp = {
                           .dst2_va_lo = (uint32_t)dst2_va,
                           .dst2_va_hi = (uint32_t)(dst2_va >> 32),
                           .flags = dual_flags,
                        };
                        memcpy(rt->push + dual_off, &dp, sizeof(dp));
                     } else if (pl->dual_write == HK_COMPUTE_PAIR_DUAL_COPY2) {
                        const struct hk_cmd_buffer *d2_src =
                           (const void *)submit->command_buffers[idx + pl->dual2_idx_off];
                        const struct hk_cs *d2_cs =
                           list_first_entry(&d2_src->control_streams, struct hk_cs, node);
                        const struct hk_compute_pair_dispatch *d2_disp = d2_cs->compute_pair;
                        const uint64_t dst2_va =
                           d_disp->bindings[1].address + d_disp->extra.copy4.dst_start;
                        const uint64_t dst3_va =
                           d2_disp->bindings[1].address + d2_disp->extra.copy4.dst_start;
                        struct hk_dual_copy2_push dp = {
                           .dst2_va_lo = (uint32_t)dst2_va,
                           .dst2_va_hi = (uint32_t)(dst2_va >> 32),
                           .dst3_va_lo = (uint32_t)dst3_va,
                           .dst3_va_hi = (uint32_t)(dst3_va >> 32),
                           .flags = dual_flags,
                        };
                        memcpy(rt->push + dual_off, &dp, sizeof(dp));
                     } else if (pl->dual_write == HK_COMPUTE_PAIR_DUAL_FADD_EP) {
                        const uint64_t add_src_va =
                           d_disp->bindings[pl->fadd_add_src_binding].address +
                           d_disp->extra.copy4.src_start;
                        const uint64_t fadd_dst_va =
                           d_disp->bindings[2].address +
                           d_disp->extra.copy4.dst_start;
                        struct hk_dual_fadd_ep_push dp = {
                           .add_src_va_lo = (uint32_t)add_src_va,
                           .add_src_va_hi = (uint32_t)(add_src_va >> 32),
                           .fadd_dst_va_lo = (uint32_t)fadd_dst_va,
                           .fadd_dst_va_hi = (uint32_t)(fadd_dst_va >> 32),
                           .flags = dual_flags,
                        };
                        memcpy(rt->push + dual_off, &dp, sizeof(dp));
                     } else if (pl->dual_write == HK_COMPUTE_PAIR_DUAL_FADD_EP_COPY) {
                        const struct hk_cmd_buffer *d2_src =
                           (const void *)submit->command_buffers[idx + pl->dual2_idx_off];
                        const struct hk_cs *d2_cs =
                           list_first_entry(&d2_src->control_streams, struct hk_cs, node);
                        const struct hk_compute_pair_dispatch *d2_disp = d2_cs->compute_pair;
                        const uint64_t add_src_va =
                           d_disp->bindings[pl->fadd_add_src_binding].address +
                           d_disp->extra.copy4.src_start;
                        const uint64_t fadd_dst_va =
                           d_disp->bindings[2].address +
                           d_disp->extra.copy4.dst_start;
                        const uint64_t dst2_va =
                           d2_disp->bindings[1].address +
                           d2_disp->extra.copy4.dst_start;
                        struct hk_dual_fadd_ep_copy_push dp = {
                           .add_src_va_lo = (uint32_t)add_src_va,
                           .add_src_va_hi = (uint32_t)(add_src_va >> 32),
                           .fadd_dst_va_lo = (uint32_t)fadd_dst_va,
                           .fadd_dst_va_hi = (uint32_t)(fadd_dst_va >> 32),
                           .dst2_va_lo = (uint32_t)dst2_va,
                           .dst2_va_hi = (uint32_t)(dst2_va >> 32),
                           .flags = dual_flags,
                        };
                        memcpy(rt->push + dual_off, &dp, sizeof(dp));
                     } else if (pl->dual_write == HK_COMPUTE_PAIR_DUAL_PAD4) {
                        const uint64_t dst2_va = d_disp->bindings[2].address;
                        const uint64_t pad_va = d_disp->bindings[1].address;
                        const struct hk_compute_pair_pad4 *pd = &d_disp->extra.pad4;
                        struct hk_dual_pad4_push dp = {
                           .dst2_va_lo = (uint32_t)dst2_va,
                           .dst2_va_hi = (uint32_t)(dst2_va >> 32),
                           .pad_va_lo = (uint32_t)pad_va,
                           .pad_va_hi = (uint32_t)(pad_va >> 32),
                           .in_h = pd->in_shape[1],
                           .in_w = pd->in_shape[2],
                           .out_h = pd->out_shape[1],
                           .out_w = pd->out_shape[2],
                           .pad_idx = {pd->pad_idx[0], pd->pad_idx[1],
                                       pd->pad_idx[2], pd->pad_idx[3]},
                           .flags = dual_flags,
                        };
                        memcpy(rt->push + dual_off, &dp, sizeof(dp));
                     } else if (pl->dual_write == HK_COMPUTE_PAIR_DUAL_GEMM_DW) {
                        struct hk_dual_gemm_dw_push dp = pl->gemm_dw_push;
                        if (pl->skip_c)
                           dp.flags |= HK_GEMM_DW_FLAG_SKIP_C;
                        if (pl->skip_pad_dst)
                           dp.flags |= HK_GEMM_DW_FLAG_SKIP_PAD_DST;
                        if (pl->skip_dw_dst1)
                           dp.flags |= HK_GEMM_DW_FLAG_SKIP_DW_DST1;
                        if (pl->skip_dw_dst2)
                           dp.flags |= HK_GEMM_DW_FLAG_SKIP_DW_DST2;
                        memcpy(rt->push + dual_off, &dp, sizeof(dp));
                        if (pl->gemm_dw_proj_idx_off) {
                           const uint32_t proj_off =
                              hk_dual_gemm_dw_proj_push_offset(prod_pb);
                           struct hk_dual_gemm_dw_proj_push pp = pl->gemm_dw_proj_push;
                           if (pl->skip_proj_c)
                              pp.proj_flags |= HK_GEMM_DW_PROJ_FLAG_SKIP_C;
                           if (pl->skip_proj_copy)
                              pp.proj_flags |= HK_GEMM_DW_PROJ_FLAG_SKIP_COPY;
                           memcpy(rt->push + proj_off, &pp, sizeof(pp));
                        }
                        if (pl->alt_kind_override != HK_COMPUTE_PAIR_ALT_NONE ||
                            pl->gemm_dw_proj_idx_off != 0) {
                           const uint32_t in_w_o = dp.in_wh & 0xffffu;
                           const uint32_t in_h_o = dp.in_wh >> 16;
                           const enum hk_compute_pair_alt_kind eff_ak =
                              pl->alt_kind_override != HK_COMPUTE_PAIR_ALT_NONE
                                 ? pl->alt_kind_override : p_disp->alt_kind;
                           const uint32_t gdw_tn =
                              hk_compute_pair_alt_gemm_tile_n(eff_ak);
                           int32_t n_o = 0;
                           memcpy(&n_o, p_disp->push + p_disp->info.gemm.n_offset_B, 4);
                           tmp_disp.alt_kind = eff_ak;
                           tmp_disp.exec_threads[0] = pl->gemm_dw_proj_idx_off
                              ? 128u : (((uint32_t)n_o / gdw_tn) * 128u);
                           tmp_disp.exec_threads[1] = (in_h_o >> 3) * (in_w_o >> 3);
                           tmp_disp.exec_threads[2] = 1u;
                        }
                     } else if (hk_compute_pair_dual_is_fadd(pl->dual_write)) {
                        /* The addresses of the output's first element in the
                         * residual and destination buffers
                         * (hk_nir_apply_gemm_fadd()). */
                        const uint64_t res_va =
                           d_disp->bindings[pl->fadd_add_src_binding].address +
                           d_disp->extra.copy4.src_start;
                        const uint64_t dst1_va =
                           d_disp->bindings[2].address + d_disp->extra.copy4.dst_start;
                        uint64_t dst2_va = 0;
                        if (pl->dual2_idx_off) {
                           const struct hk_cmd_buffer *c_src =
                              (const void *)submit->command_buffers[idx + pl->dual2_idx_off];
                           const struct hk_cs *c_cs =
                              list_first_entry(&c_src->control_streams, struct hk_cs, node);
                           const struct hk_compute_pair_dispatch *c_disp = c_cs->compute_pair;
                           dst2_va =
                              c_disp->bindings[1].address + c_disp->extra.copy4.dst_start;
                        }
                        const bool inplace =
                           pl->dual_write == HK_COMPUTE_PAIR_DUAL_FADD_INPLACE ||
                           pl->dual_write == HK_COMPUTE_PAIR_DUAL_FADD_COPY_INPLACE;
                        struct hk_dual_fadd_push dp = {
                           .res_va_lo = (uint32_t)res_va,
                           .res_va_hi = (uint32_t)(res_va >> 32),
                           .dst1_va_lo = (uint32_t)dst1_va,
                           .dst1_va_hi = (uint32_t)(dst1_va >> 32),
                           .dst2_va_lo = (uint32_t)dst2_va,
                           .dst2_va_hi = (uint32_t)(dst2_va >> 32),
                        };
                        const size_t dp_sz =
                           pl->dual2_idx_off ? sizeof(dp) : (inplace ? 8u : 16u);
                        memcpy(rt->push + dual_off, &dp, dp_sz);
                     } else {
                        root_ok = false;
                     }
                  }
                  if (pl->alt_kind_override != HK_COMPUTE_PAIR_ALT_NONE &&
                      pl->dual_write != HK_COMPUTE_PAIR_DUAL_GEMM_DW) {
                     tmp_disp.alt_kind = pl->alt_kind_override;
                     if (hk_compute_pair_alt_gemm_tile_m(pl->alt_kind_override) == 32u &&
                         hk_compute_pair_alt_gemm_tile_m(p_disp->alt_kind) == 64u) {
                        tmp_disp.exec_threads[1] = p_disp->exec_threads[1] * 2u;
                     } else if (hk_compute_pair_alt_gemm_tile_m(pl->alt_kind_override) == 64u &&
                                hk_compute_pair_alt_gemm_tile_m(p_disp->alt_kind) == 32u) {
                        tmp_disp.exec_threads[1] = p_disp->exec_threads[1] / 2u;
                     }
                  }
                  if (root_ok && pl->im2col_idx_off) {
                     const struct hk_cmd_buffer *im_src =
                        (const void *)submit->command_buffers[idx - pl->im2col_idx_off];
                     const struct hk_cs *im_cs =
                        list_first_entry(&im_src->control_streams, struct hk_cs, node);
                     const struct hk_compute_pair_dispatch *im_disp = im_cs->compute_pair;
                     const uint32_t im2_off =
                        hk_im2col_a_push_offset(prod_pb, pl->dual_write);
                     struct hk_im2col_a_push ip = {0};
                     if (im_disp->info.role == HK_COMPUTE_PAIR_IM2COL) {
                        const struct hk_compute_pair_im2col *im = &im_disp->extra.im2col;
                        const uint64_t src_va = im_disp->bindings[0].address;
                        const uint64_t dst_va =
                           pl->skip_im2_dst ? 0ull : im_disp->bindings[1].address;
                        const bool vec4_ok =
                           ((im->iw | im->kw | im->step_x | im->pad_x) & 3u) == 0 &&
                           (src_va & 15ull) == 0 &&
                           (dst_va == 0 || (dst_va & 15ull) == 0);
                        ip = (struct hk_im2col_a_push){
                           .src_va_lo = (uint32_t)src_va,
                           .src_va_hi = (uint32_t)(src_va >> 32),
                           .dst_va_lo = (uint32_t)dst_va,
                           .dst_va_hi = (uint32_t)(dst_va >> 32),
                           .iw = im->iw,
                           .ih = im->ih,
                           .kw = im->kw,
                           .cols = im->cols,
                           .step_x = im->step_x,
                           .sh = im->sh,
                           .pad_x = im->pad_x,
                           .pad_y = im->pad_y,
                           .flags = vec4_ok ? HK_IM2COL_A_FLAG_VEC4 : 0u,
                        };
                     } else {
                        const struct hk_compute_pair_copy4 *cp = &im_disp->extra.copy4;
                        const uint64_t src_va =
                           im_disp->bindings[0].address + cp->src_start;
                        const uint64_t dst_va =
                           pl->skip_im2_dst ? 0ull : (im_disp->bindings[1].address + cp->dst_start);
                        int32_t k_i = 0;
                        memcpy(&k_i, p_disp->push + p_disp->info.gemm.k_offset_B, 4);
                        const uint32_t k_u = (uint32_t)k_i;
                        const uint32_t m_u = (uint32_t)(cp->bytes / (4ull * (uint64_t)k_u));
                        ip = (struct hk_im2col_a_push){
                           .src_va_lo = (uint32_t)src_va,
                           .src_va_hi = (uint32_t)(src_va >> 32),
                           .dst_va_lo = (uint32_t)dst_va,
                           .dst_va_hi = (uint32_t)(dst_va >> 32),
                           .iw = k_u,
                           .ih = m_u,
                           .kw = k_u,
                           .cols = 1u,
                           .step_x = 0u,
                           .sh = 1u,
                           .pad_x = 0u,
                           .pad_y = 0u,
                           .flags = HK_IM2COL_A_FLAG_VEC4 | HK_IM2COL_A_FLAG_COPY1X1,
                        };
                     }
                     hk_im2col_a_push_init_udiv(&ip);
                     memcpy(rt->push + im2_off, &ip, sizeof(ip));
                  }
                  if (root_ok && pl->dw_a_idx_off) {
                     const uint32_t dw_off =
                        hk_dw_a_push_offset(prod_pb, pl->dual_write);
                     struct hk_dw_a_push dp = pl->dw_a_push;
                     if (pl->skip_dw_dst1)
                        dp.pad_flags &= ~(HK_DW_A_FLAG_HAS_DST1 << 16);
                     if (pl->skip_dw_dst2)
                        dp.pad_flags &= ~(HK_DW_A_FLAG_HAS_DST2 << 16);
                     memcpy(rt->push + dw_off, &dp, sizeof(dp));
                     if (pl->exp_idx_off) {
                        const uint32_t exp_off =
                           hk_exp_dw_a_push_offset(prod_pb, pl->dual_write);
                        struct hk_exp_dw_a_push ep_p = pl->exp_dw_a_push;
                        if (pl->skip_exp_c)
                           ep_p.flags |= HK_EXP_DW_A_FLAG_SKIP_EXP_C;
                        if (pl->skip_exp_im2_dst)
                           ep_p.flags |= HK_EXP_DW_A_FLAG_SKIP_IM2_DST;
                        memcpy(rt->push + exp_off, &ep_p, sizeof(ep_p));
                        const uint32_t in_w_e = dp.in_wh & 0xffffu;
                        const uint32_t in_h_e = dp.in_wh >> 16;
                        tmp_disp.exec_threads[1] = (in_h_e >> 3) * (in_w_e >> 3);
                     }
                  }
                  tmp_disp.root_va = root_va + root_off;
               }

               const size_t unique = cs->compute_resources.size;
               size_t ioto_used = 0;
               if (root_ok &&
                   hk_pack_add_bo(out->extra_bos, fused_shader->only_linked->b.bo) &&
                   (!(fused_shader->b.info.has_preamble ||
                      fused_shader->b.info.rodata.size_16) ||
                    hk_pack_add_bo(out->extra_bos, fused_shader->bo)) &&
                   hk_emit_fused_compute_dispatch(dev, cs, fused_shader,
                                                  &tmp_disp,
                                                  ioto_map + ioto_off,
                                                  ioto_va + ioto_off,
                                                  ioto_total - ioto_off,
                                                  &ioto_used)) {
                  ioto_off += ALIGN_POT(ioto_used, 64u);
                  if (needs_root)
                     root_off += root_stride;
                  hk_pack_fold_attachments(cs, item);
                  hk_pack_dedup_resources(&cs->compute_resources, unique);
                  for (unsigned k = 1; k < pl->span; ++k)
                     suppressed_arr[idx + k] = true;
                  if (pl->dual_write == HK_COMPUTE_PAIR_DUAL_COPY2)
                     suppressed_arr[idx + pl->dual2_idx_off] = true;
                  fused = true;
                  fused_pairs++;
                  if (pl->im2col_idx_off)
                     im2a_fused++;
                  if (pl->dw_a_idx_off)
                     dwa_fused++;
                  if (pl->exp_idx_off)
                     expdwa_fused++;
                  if (pl->dual_write == HK_COMPUTE_PAIR_DUAL_GEMM_DW) {
                     gemm_dw_fused++;
                     if (pl->gemm_dw_proj_idx_off)
                        gdwproj_fused++;
                  }
               }
            }

            if (!fused &&
                ((pl->im2col_idx_off && suppressed_arr[idx - pl->im2col_idx_off]) ||
                 (pl->dw_a_idx_off && suppressed_arr[idx - pl->dw_a_idx_off]) ||
                 (pl->exp_idx_off && suppressed_arr[idx - pl->exp_idx_off]) ||
                 (pl->exp_im2_idx_off && suppressed_arr[idx - pl->exp_im2_idx_off]))) {
               /* Preceding im2col/dw_a was already suppressed for this fused
                * GEMM; abort packing cleanly rather than silently dropping it. */
               hk_packed_runs_free(dev, out);
               memset(out, 0, sizeof(*out));
               return false;
            }

            if (!fused) {
               void *emit_end = (uint8_t *)item->current - terminate_size;

               /* The reserved range keeps the merge inline: the link path,
                * which allocates from the packet's (absent) pool and would
                * point the packet into the source's stream, must stay
                * unreachable. The sources are only read here; the application
                * keeps them alive until the submission completes. */
               if (!hk_cs_space_fits((uint8_t *)cs->end - (uint8_t *)cs->current,
                                     (uint8_t *)emit_end - (uint8_t *)item->start,
                                     AGX_CDM_STREAM_LINK_LENGTH)) {
                  assert(!"packet sized too small");
                  hk_packed_runs_free(dev, out);
                  memset(out, 0, sizeof(*out));
                  return false;
               }
               const size_t unique = cs->compute_resources.size;
               hk_cs_merge_cdm_until(cs, item, emit_end);
               hk_pack_dedup_resources(&cs->compute_resources, unique);
            }

            unsigned next_idx = UINT32_MAX;
            for (unsigned m2 = m + 1; m2 < pr->count; ++m2) {
               const unsigned idx2 = pr->first + m2;
               if (!suppressed_arr[idx2] &&
                   !(has_non_self && plans[idx2].is_self_copy)) {
                  next_idx = idx2;
                  break;
               }
            }
            const bool eff_last = next_idx == UINT32_MAX;

            if (eff_last) {
               if (visible_completion) {
                  hk_cdm_visibility_flush(dev, cs, true);
               } else if (hk_cdm_memory_scope_form(dev) &&
                          hk_cdm_alias_scope(dev) &&
                          hk_cdm_visibility_flush_needed(dev, cs, true)) {
                  hk_cdm_visibility_flush(dev, cs, true);
               }
            } else if (hk_cdm_alias_scope(dev)) {
               hk_pack_hazard_reset_all(&hazard);
               hk_cdm_visibility_flush(dev, cs, true);
            } else {
               struct hk_pack_footprint cur_fp;
               if (cached_fp_idx == idx && cached_fp_fused == fused) {
                  cur_fp = cached_fp;
               } else {
                  compute_pair_emitted_footprint(submit, plans, idx, fused,
                                                 /*for_liveness=*/false, &cur_fp);
               }
               const bool mandatory_strong =
                  (!visible_completion && cs->current == cs->memory_flush_end) ||
                  (!cur_fp.known && cs->current == cs->strong_flush_end);
               if (mandatory_strong) {
                  hk_pack_hazard_record(&hazard, &cur_fp, true);
                  if (visible_completion &&
                      cs->current == cs->memory_flush_end &&
                      cs->current == cs->strong_flush_end) {
                     uint8_t *word =
                        (uint8_t *)cs->current - AGX3_CDM_BARRIER_SIZE;
                     agx3_cmdstream_store_le32(
                        word, agx3_cmdstream_load_le32(word) &
                                 ~AGX3_CDM_BARRIER_FLUSH_MEMORY);
                     cs->memory_flush_end = NULL;
                  }
                  hk_cdm_visibility_flush(dev, cs, false);
               } else {
                  hk_pack_hazard_record(&hazard, &cur_fp, false);

                  const struct hk_cmd_buffer *next_src =
                     (const void *)submit->command_buffers[next_idx];
                  const struct hk_cs *next_cs = list_first_entry(
                     &next_src->control_streams, struct hk_cs, node);
                  const struct hk_pack_fuse_plan *npl = &plans[next_idx];
                  const bool next_planned_fused =
                     ioto_map && npl->shader &&
                     (!npl->im2col_idx_off ||
                      (root_map && (next_idx - pr->first) >= npl->im2col_idx_off)) &&
                     (!npl->dw_a_idx_off ||
                      (root_map && (next_idx - pr->first) >= npl->dw_a_idx_off)) &&
                     (!npl->exp_idx_off ||
                      (root_map && (next_idx - pr->first) >= npl->exp_idx_off)) &&
                     (!npl->exp_im2_idx_off ||
                      (root_map && (next_idx - pr->first) >= npl->exp_im2_idx_off)) &&
                     (npl->dual_write == HK_COMPUTE_PAIR_DUAL_NONE || root_map) &&
                     (next_idx - pr->first) + npl->span <= pr->count &&
                     (npl->dual_write != HK_COMPUTE_PAIR_DUAL_COPY2 ||
                      (next_idx - pr->first) + npl->dual2_idx_off < pr->count);
                  struct hk_pack_footprint next_fp;
                  compute_pair_emitted_footprint(submit, plans, next_idx,
                                                 next_planned_fused,
                                                 /*for_liveness=*/false, &next_fp);
                  cached_fp = next_fp;
                  cached_fp_idx = next_idx;
                  cached_fp_fused = next_planned_fused;

                  const bool cur_overlap_safe =
                     fused ? hk_pack_shader_overlap_safe(fused_shader)
                           : (item->compute_pair &&
                              item->compute_pair->overlap_safe);
                  const bool next_overlap_safe =
                     next_cs->compute_pair &&
                     next_cs->compute_pair->overlap_safe &&
                     (!next_planned_fused ||
                      hk_pack_shader_overlap_safe(npl->shader));
                  const bool cur_tail_barrier =
                     (cs->current == cs->weak_barrier_end ||
                      (cs->current == cs->strong_flush_end &&
                       (visible_completion ||
                        cs->current != cs->memory_flush_end))) &&
                     (size_t)((uint8_t *)cs->current - (uint8_t *)cs->start) >
                        AGX3_CDM_BARRIER_SIZE &&
                     (fused || !item->leading_barrier_end);
                  const bool can_elide_weak =
                     agx_chip_elide_app_dispatch_barrier(dev->dev.chip) &&
                     cur_tail_barrier && cur_overlap_safe &&
                     next_overlap_safe && !item->uses_global_atomics &&
                     !next_cs->uses_global_atomics &&
                     !next_cs->leading_barrier_end;

                  enum hk_pack_boundary_action action =
                     hk_pack_hazard_decide(&hazard, &next_fp, can_elide_weak);
                  if (action == HK_PACK_BOUNDARY_KEEP_WEAK &&
                      cs->current == cs->strong_flush_end &&
                      (!cur_tail_barrier ||
                       !hk_cdm_downgrade_strong_to_weak(dev, cs))) {
                     hk_pack_hazard_reset_all(&hazard);
                     action = HK_PACK_BOUNDARY_STRONG;
                  }

                  if (action == HK_PACK_BOUNDARY_ELIDE_BARRIER) {
                     cs->current = (uint32_t *)((uint8_t *)cs->current -
                                                AGX3_CDM_BARRIER_SIZE);
                     cs->weak_barrier_end = NULL;
                     cs->strong_flush_end = NULL;
                     cs->memory_flush_end = NULL;
                     flushes_elided++;
                     barriers_elided++;
                  } else if (action == HK_PACK_BOUNDARY_KEEP_WEAK) {
                     flushes_elided++;
                  } else {
                     if (visible_completion &&
                         cs->current == cs->memory_flush_end &&
                         cs->current == cs->strong_flush_end) {
                        uint8_t *word =
                           (uint8_t *)cs->current - AGX3_CDM_BARRIER_SIZE;
                        agx3_cmdstream_store_le32(
                           word, agx3_cmdstream_load_le32(word) &
                                    ~AGX3_CDM_BARRIER_FLUSH_MEMORY);
                        cs->memory_flush_end = NULL;
                     }
                     hk_cdm_visibility_flush(dev, cs, false);
                  }
               }
            }
         }

         hk_pack_fold_member(cmd, cs, source, item, last, &preceding_kinds);
      }

      assert(!cs->stream_linked);
      assert(visible_completion
                ? cs->memory_flush_end == cs->current
                : cs->strong_flush_end == cs->current ||
                     cs->weak_barrier_end == cs->current);
      memcpy(cs->current, terminate, terminate_size);
      cs->current = (uint8_t *)cs->current + terminate_size;
      memset(cs->current, 0, HK_CS_SPACE_PADDING);
      if (cached) {
         const size_t written =
            (uint8_t *)cs->current - (uint8_t *)cs->start + HK_CS_SPACE_PADDING;
         util_flush_range_no_fence(cs->start, written);
      }
      hk_pack_finish_sync(cmd, cs);

      out->commands[out->command_count++] = &cmd->vk;
      i += pr->count;
      r++;
   }

   if (cached) {
      if (ioto_map && ioto_off > 0)
         util_flush_range_no_fence(ioto_map, ioto_off);
      if (root_map && root_off > 0)
         util_flush_range_no_fence(root_map, root_off);
      util_post_flush_fence();
   }

   if (trace) {
      const int64_t t3_ns = os_time_get_nano();
      fprintf(stderr,
              "HK_PACK runs=%u members=%u fused=%u im2a=%u dwa=%u expdwa=%u gemm_dw=%u gdwproj=%u self=%u eligible=%u of %u commands=%u bytes=%zu strong_tails=%d flushes_elided=%u barriers_elided=%u dead_writes_elided=%u us=%lld/%lld/%lld/%lld ioto=%zu/%zu root=%zu/%zu\n",
              run_count, packed_members, fused_pairs, im2a_fused, dwa_fused,
              expdwa_fused, gemm_dw_fused, gdwproj_fused, self_elided,
              eligible, n, out->command_count, total,
              hk_pack_admits_strong_tail(hk_pack_boundary(dev)),
              flushes_elided, barriers_elided, dead_writes_elided,
              (long long)((t1_ns - t0_ns) / 1000),
              (long long)((t2_ns - t1_ns) / 1000),
              (long long)(live_ns / 1000),
              (long long)((t3_ns - t2_ns - live_ns) / 1000),
              ioto_off, ioto_total, root_off, root_total);
   }
   return true;
}

/* Submit-time render packing (hk_render_pack.h): a submission-owned
 * command buffer that runs a packed run of render command buffers as one
 * render command. */
#define HK_RENDER_PACK_MAX_PRE 16

struct hk_render_packet {
   struct hk_cmd_buffer cmd;
   struct hk_cs cs;
   struct hk_cs pre_cs[HK_RENDER_PACK_MAX_PRE];
   struct hk_cs post_cs[HK_HEAP_SLOTS];
};

struct hk_render_pack_run {
   unsigned first, count, real_count;
   bool ends_pass;
   /* The members whose scissor and depth-bias arrays serve every member's
    * draws (hk_render_pack_array_prefix()). */
   const struct hk_cs *scissor, *zbias;
};

static void
hk_queue_rpack_free_packets(struct hk_queue *queue)
{
   for (unsigned i = 0; i < queue->rpack.packets_cap; ++i) {
      util_dynarray_fini(&queue->rpack.packets[i].cs.render_resources);
      util_dynarray_fini(&queue->rpack.packets[i].cmd.sync_pass_records);
   }
   free(queue->rpack.packets);
   queue->rpack.packets = NULL;
   queue->rpack.packets_cap = 0;
}

static bool
hk_queue_rpack_ensure_packets(struct hk_queue *queue, unsigned count)
{
   if (count <= queue->rpack.packets_cap)
      return true;

   unsigned cap = MAX2(queue->rpack.packets_cap * 2, 4);
   cap = MAX2(cap, count);
   struct hk_render_packet *packets = calloc(cap, sizeof(*packets));
   if (!packets)
      return false;

   hk_queue_rpack_free_packets(queue);
   queue->rpack.packets = packets;
   queue->rpack.packets_cap = cap;
   return true;
}

/* The chips whose render pass continuation hk_m5_pass_merge_begin() takes,
 * HAL200 and HAL300 alike. ASAHI_MESA_DEBUG=nomerge turns it off with the
 * other stream merges. */
static bool
hk_render_pack_enabled(const struct hk_device *dev)
{
   return !dev->dev.is_virtio && !(dev->dev.debug & AGX_DBG_NOMERGE) &&
          agx_chip_native_or_m5(dev->dev.chip);
}

/* The target of a stream link to `cs`, as hk_cmd_buffer_emit_cs_jump()
 * encodes one, without changing `cmd` (a simultaneous-use command buffer may
 * be submitted from several queues at once). 0 if it has none. */
static uint64_t
hk_render_pack_link_target(struct hk_device *dev, struct hk_cmd_buffer *cmd,
                           const struct hk_cs *cs)
{
   uint64_t target = cs->addr;
   if (hk_g17p_vdm_low_window_enabled(&dev->dev)) {
      const uint64_t alias = hk_g17p_vdm_alias(cmd, cs->addr);
      target = alias > AGX_G17P_VDM_USER_BASE ? alias - AGX_G17P_VDM_USER_BASE
                                              : 0;
   } else if (agx_chip_hal300_path(dev->dev.chip) &&
              !agx_m5_vdm_compact(cs->addr, &target)) {
      return 0;
   }

   struct agx_vdm_stream_link_core link;
   return target && agx_vdm_stream_link_core_pack(dev->dev.chip, target, &link)
             ? target
             : 0;
}

/* Frame anatomy: the boundary from cmds[a]'s tail to cmds[a + 1]'s head did
 * not pack, because of `why` or (why NULL) because cmds[m] is no member. */
static void
hk_render_pack_note_boundary(struct hk_device *dev,
                             struct hk_cmd_buffer *const *cmds, unsigned a,
                             unsigned m, bool first, const char *why)
{
   struct hk_render_pack_edges *e = &cmds[a]->render_pack;
   if (likely(!dev->frame_anatomy) || !e->tail ||
       !cmds[a + 1]->render_pack.head)
      return;
   e->next_who = why ? 0 : m == a ? 1 : 2;
   e->next_why = why ? why : hk_render_pack_member_refusal_for(cmds[m], first);
   hk_frame_anatomy_note(dev, "pass-merge render pack: cb %u -> %u: %s%s",
                         a, a + 1,
                         e->next_who == 1 ? "this: "
                         : e->next_who == 2 ? "next: "
                                            : "",
                         e->next_why);
}

/* Packs runs of the submission's command buffers into render packets and
 * writes every command buffer's tail for this submission: a link to the next
 * member, or the terminate record a command buffer was recorded with. The
 * tails are written for every submission, packed or not, so a command buffer
 * an earlier submission linked never runs into a stale neighbour. On true,
 * `effective` is the submission with the packets in place of their runs. */
static bool
hk_render_pack_runs(struct hk_device *dev, struct hk_queue *queue,
                    const struct vk_queue_submit *submit,
                    struct vk_queue_submit *effective)
{
   const unsigned n = submit->command_buffer_count;
   struct hk_cmd_buffer *const *cmds =
      (struct hk_cmd_buffer *const *)submit->command_buffers;
   const bool enabled = n >= 2 && !submit->buffer_bind_count &&
                        !submit->image_bind_count &&
                        !submit->image_opaque_bind_count &&
                        hk_render_pack_enabled(dev);

   bool linked = false;
   for (unsigned i = 0; i < n && !linked; ++i)
      linked = cmds[i]->render_pack.linked != 0;
   if (!enabled && !linked)
      return false;

   uint64_t *targets = NULL;
   struct hk_render_pack_run *runs = NULL;
   unsigned run_count = 0, packed = 0;
   if (enabled) {
      targets = hk_queue_pack_zscratch(&queue->rpack.targets, n,
                                       sizeof(*targets));
      runs = hk_queue_pack_scratch(&queue->rpack.runs, n, sizeof(*runs));
   }

   for (unsigned i = 0; targets && runs && i < n;) {
      const struct hk_cs *first_pre = NULL, *first_post = NULL;
      const struct hk_cs *first =
         hk_render_pack_member_streams(cmds[i], true, &first_pre, &first_post);
      struct hk_render_pack_run run = {
         .first = i,
         .count = 1,
         .real_count = first ? 1 : 0,
         .ends_pass = false,
         .scissor = first,
         .zbias = first,
      };
      unsigned prev = i;
      enum u_tristate dbias = first ? first->cr.dbias_is_int : U_TRISTATE_UNSET;
      size_t resources = first ? first->render_resources.size : 0;
      uint8_t used_heap_slots =
         first && first->sync.heap_slot
            ? BITFIELD_BIT(first->sync.heap_slot - 1)
            : 0;
      uint8_t run_vdm_kinds = first ? first->sync.kind : 0;
      uint8_t run_vdm_tail_req[HK_SYNC_NUM_KINDS] = {0};
      uint8_t post_heap_slots =
         first_post && first_post->sync.heap_slot
            ? BITFIELD_BIT(first_post->sync.heap_slot - 1)
            : 0;
      bool run_post_heap_only =
         !first_post || hk_render_pack_post_is_heap_reset(first_post);
      uint8_t run_post_tail_req[HK_SYNC_NUM_KINDS] = {0};
      if (first) {
         for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
            if (cmds[i]->sync.req[k][HK_SYNC_VDM].cut)
               run_vdm_tail_req[k] |= first->sync.kind;
            run_vdm_tail_req[k] |= cmds[i]->sync.req[k][HK_SYNC_VDM].prior;
         }
      }
      if (first_post) {
         for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
            const struct hk_sync_dep *creq =
               &cmds[i]->sync.req[k][HK_SYNC_CDM];
            if (creq->cut == first_post ||
                (creq->prior & first_post->sync.kind))
               run_post_tail_req[k] |= first_post->sync.kind;
         }
      }
      unsigned run_pre = first_pre ? 1 : 0;
      unsigned run_post = first_post ? 1 : 0;
      struct hk_render_pack_run_state state;
      if (first)
         hk_render_pack_run_start(&state, cmds[i], first);
      unsigned stamps = first ? hk_render_pack_stamp_slots(first) : 0;
      if (unlikely(dev->frame_anatomy))
         cmds[i]->render_pack.next_why = NULL;
      if (!first && i + 1 < n)
         hk_render_pack_note_boundary(dev, cmds, i, i, true, NULL);

      while (first && i + run.count < n) {
         const unsigned next = i + run.count;
         unsigned b = next;
         unsigned step = 1;
         bool is_res = false, is_susp = false;
         if (hk_render_pack_is_empty_fixup(cmds[next], &is_res, &is_susp) &&
             is_res && cmds[prev]->render_pack.tail &&
             (cmds[prev]->render_pack.tail_pass.flags &
              VK_RENDERING_SUSPENDING_BIT) &&
             hk_render_pack_fixup_matches(&cmds[next]->render_pack.empty_pass,
                                          &cmds[prev]->render_pack.tail_pass)) {
            bool next_res = false, next_susp = false;
            if (next + 2 < n &&
                hk_render_pack_is_empty_fixup(cmds[next + 1], &next_res,
                                              &next_susp) &&
                next_susp && cmds[next + 2]->render_pack.head &&
                (cmds[next + 2]->render_pack.head_pass.flags &
                 VK_RENDERING_RESUMING_BIT) &&
                hk_render_pack_fixup_matches(
                   &cmds[next + 1]->render_pack.empty_pass,
                   &cmds[next + 2]->render_pack.head_pass)) {
               b = next + 2;
               step = 3;
            } else if (run.real_count >= 2) {
               run.count++;
               run.ends_pass = true;
               break;
            }
         }

         const struct hk_cs *bs_pre = NULL, *bs_post = NULL;
         const struct hk_cs *bs =
            hk_render_pack_member_streams(cmds[b], false, &bs_pre, &bs_post);
         if (unlikely(dev->frame_anatomy))
            cmds[b]->render_pack.next_why = NULL;
         if (!bs) {
            if (step == 1)
               hk_render_pack_note_boundary(dev, cmds, prev, b, false, NULL);
            break;
         }

         const char *why =
            hk_render_pack_boundary(&state, cmds[prev], cmds[b], bs, first);
         if (!why && bs_pre) {
            why = hk_render_pack_pre_cs_refusal(
               &state, bs_pre, run_post, post_heap_slots, run_post_heap_only,
               run_vdm_kinds, run_vdm_tail_req, run_post_tail_req);
         }
         if (!why && bs->sync.heap_slot &&
             (used_heap_slots & BITFIELD_BIT(bs->sync.heap_slot - 1)))
            why = "heap slot";
         if (!why && bs_pre && run_pre >= HK_RENDER_PACK_MAX_PRE)
            why = "pre-graphics count";
         if (!why && bs_post && run_post >= HK_HEAP_SLOTS)
            why = "post-graphics count";
         const struct util_dynarray *scissor = NULL, *zbias = NULL;
         if (!why &&
             (!hk_render_pack_array_prefix(&run.scissor->scissor,
                                           &bs->scissor, &scissor) ||
              !hk_render_pack_array_prefix(&run.zbias->depth_bias,
                                           &bs->depth_bias, &zbias)))
            why = "scissor or depth bias array";
         if (!why && bs->cr.dbias_is_int != U_TRISTATE_UNSET &&
             dbias != U_TRISTATE_UNSET && bs->cr.dbias_is_int != dbias)
            why = "depth bias format";
         /* The resource list and the tile record share a 16-bit size. */
         if (!why && resources + bs->render_resources.size >
                        UINT16_MAX - sizeof(struct drm_asahi_render_program_v2))
            why = "render resources";
         if (!why && (stamps & hk_render_pack_stamp_slots(bs)))
            why = "timestamp slot";
         const uint64_t target =
            why ? 0 : hk_render_pack_link_target(dev, cmds[b], bs);
         if (!why && !target)
            why = "stream link";
         if (why) {
            hk_render_pack_note_boundary(dev, cmds, prev, b, false, why);
            break;
         }

         targets[prev] = target;
         if (scissor == &bs->scissor)
            run.scissor = bs;
         if (zbias == &bs->depth_bias)
            run.zbias = bs;
         if (dbias == U_TRISTATE_UNSET)
            dbias = bs->cr.dbias_is_int;
         if (bs->sync.heap_slot)
            used_heap_slots |= BITFIELD_BIT(bs->sync.heap_slot - 1);
         run_vdm_kinds |= bs->sync.kind;
         for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
            if (cmds[b]->sync.req[k][HK_SYNC_VDM].cut)
               run_vdm_tail_req[k] |= bs->sync.kind;
            run_vdm_tail_req[k] |= cmds[b]->sync.req[k][HK_SYNC_VDM].prior;
         }
         if (bs_pre)
            run_pre++;
         if (bs_post) {
            if (bs_post->sync.heap_slot)
               post_heap_slots |= BITFIELD_BIT(bs_post->sync.heap_slot - 1);
            run_post_heap_only &= hk_render_pack_post_is_heap_reset(bs_post);
            run_post++;
         }
         if (run_post > 0) {
            for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
               const struct hk_sync_dep *creq =
                  &cmds[b]->sync.req[k][HK_SYNC_CDM];
               if ((bs_post && creq->cut == bs_post) ||
                   (creq->prior & (HK_SYNC_GFX | HK_SYNC_GFX_MEM)))
                  run_post_tail_req[k] |= (HK_SYNC_GFX | HK_SYNC_GFX_MEM);
            }
         }
         resources += bs->render_resources.size;
         hk_render_pack_run_join(&state, cmds[prev], cmds[b], bs);
         stamps |= hk_render_pack_stamp_slots(bs);
         prev = b;
         run.count += step;
         run.real_count++;
      }

      if (run.real_count >= 2) {
         runs[run_count++] = run;
         packed += run.count;
      }
      i += run.count;
   }

   /* The packets, before any tail is written: a failure leaves every
    * command buffer as recorded. */
   struct vk_command_buffer **commands = NULL;
   if (run_count) {
      commands = hk_queue_pack_scratch(&queue->rpack.commands,
                                       n - packed + run_count,
                                       sizeof(*commands));
      if (!commands || !hk_queue_rpack_ensure_packets(queue, run_count))
         run_count = 0;
   }

   unsigned out = 0;
   for (unsigned r = 0, i = 0; r < run_count && i <= n;) {
      if (i < runs[r].first) {
         commands[out++] = submit->command_buffers[i++];
         continue;
      }

      const struct hk_render_pack_run *run = &runs[r];
      struct hk_render_packet *p = &queue->rpack.packets[r];
      p->cmd.vk.base.device = &dev->vk;
      hk_render_pack_init(&p->cmd, &p->cs, cmds + run->first, run->count);
      uint8_t preceding = 0;
      unsigned added = 0;
      uint8_t cdm_preceding = 0;
      uint8_t pre_cdm_preceding = 0;
      uint8_t carried_req[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ] = {0};
      bool cdm_req_cut[HK_SYNC_NUM_KINDS] = {false};
      bool pre_cdm_req_cut[HK_SYNC_NUM_KINDS] = {false};
      bool pre_needs_cdm_cut[HK_RENDER_PACK_MAX_PRE] = {false};
      unsigned n_pre = 0, n_post = 0;
      bool ok = true;
      for (unsigned m = 0; ok && m < run->count; ++m) {
         const struct hk_cmd_buffer *source = cmds[run->first + m];
         if (list_is_empty(&source->control_streams))
            continue;
         const struct hk_cs *pre = NULL, *post = NULL;
         const struct hk_cs *item = hk_render_pack_find_vdm(source, &pre, &post);
         if (!item)
            item = list_first_entry(&source->control_streams, struct hk_cs, node);
         if (pre) {
            assert(n_pre < HK_RENDER_PACK_MAX_PRE);
            const unsigned pre_idx = n_pre;
            struct hk_cs *dst = &p->pre_cs[n_pre++];
            *dst = *pre;
            list_inithead(&dst->node);
            dst->cmd = &p->cmd;
            dst->stream_owner = pre->cmd;
            memset(&dst->sync, 0, sizeof(dst->sync));
            dst->sync.kind = pre->sync.kind;
            dst->sync.heap_slot = pre->sync.heap_slot;
            dst->sync.vtx_images = pre->sync.vtx_images;
            const uint8_t pre_carry = hk_sync_full_kinds(dst->sync.kind);
            for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
               dst->sync.dep[sq].prior = pre->sync.dep[sq].prior;
               dst->sync.dep_ta[sq].prior = pre->sync.dep_ta[sq].prior;
               u_foreach_bit(k, pre_carry) {
                  dst->sync.dep[sq].prior |= carried_req[k][sq];
                  dst->sync.dep_ta[sq].prior |= carried_req[k][sq];
               }
            }
            if (pre_idx > 0 &&
                ((dst->sync.dep[HK_SYNC_CDM].prior & pre_cdm_preceding) ||
                 (dst->sync.dep_ta[HK_SYNC_CDM].prior & pre_cdm_preceding)))
               pre_needs_cdm_cut[pre_idx] = true;
            u_foreach_bit(k, pre_carry) {
               if (pre_idx > 0 && pre_cdm_req_cut[k])
                  pre_needs_cdm_cut[pre_idx] = true;
            }
            pre_cdm_preceding |= pre->sync.kind;
            cdm_preceding |= pre->sync.kind;
         }
         if (post) {
            assert(n_post < HK_HEAP_SLOTS);
            struct hk_cs *dst = &p->post_cs[n_post++];
            *dst = *post;
            list_inithead(&dst->node);
            dst->cmd = &p->cmd;
            dst->stream_owner = post->cmd;
            memset(&dst->sync, 0, sizeof(dst->sync));
            dst->sync.kind = post->sync.kind;
            dst->sync.heap_slot = post->sync.heap_slot;
            for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
               dst->sync.dep[sq].prior = post->sync.dep[sq].prior;
               dst->sync.dep_ta[sq].prior = post->sync.dep_ta[sq].prior;
            }
            cdm_preceding |= post->sync.kind;
         }
         ok = hk_render_pack_add(&p->cmd, &p->cs, source, item, added == 0,
                                 added + 1 == run->real_count, &preceding);
         added++;
         for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
            const struct hk_sync_dep *creq = &source->sync.req[k][HK_SYNC_CDM];
            if (creq->cut || (creq->prior & cdm_preceding))
               cdm_req_cut[k] = true;
            if ((pre && creq->cut == pre) ||
                (creq->prior & pre_cdm_preceding))
               pre_cdm_req_cut[k] = true;
            for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
               carried_req[k][sq] |= source->sync.req[k][sq].prior;
         }
      }
      if (!ok) {
         run_count = 0;
         break;
      }
      if (run->ends_pass)
         p->cs.g17p_pass_ends = true;
      p->cs.uploaded_scissor = run->scissor->uploaded_scissor;
      p->cs.uploaded_zbias = run->zbias->uploaded_zbias;
      hk_render_pack_finish(&p->cmd, &p->cs);
      if (n_pre || n_post) {
         list_inithead(&p->cmd.control_streams);
         uint32_t seq = 0;
         for (unsigned j = 0; j < n_pre; ++j) {
            p->pre_cs[j].sync.seq = ++seq;
            if (j > 0 && pre_needs_cdm_cut[j]) {
               p->pre_cs[j].sync.dep[HK_SYNC_CDM].cut = &p->pre_cs[j - 1];
               p->pre_cs[j].sync.dep_ta[HK_SYNC_CDM].cut = &p->pre_cs[j - 1];
            }
            list_addtail(&p->pre_cs[j].node, &p->cmd.control_streams);
            u_foreach_bit(k, p->pre_cs[j].sync.kind)
               p->cmd.sync.last[k][HK_SYNC_CDM] = &p->pre_cs[j];
            p->cmd.sync.last_any[HK_SYNC_CDM] = &p->pre_cs[j];
         }
         p->cs.sync.seq = ++seq;
         p->cs.sync.heap_slot = 0;
         if (n_pre) {
            p->cs.sync.dep[HK_SYNC_CDM].cut = &p->pre_cs[n_pre - 1];
            p->cs.sync.dep_ta[HK_SYNC_CDM].cut = &p->pre_cs[n_pre - 1];
         }
         list_addtail(&p->cs.node, &p->cmd.control_streams);
         for (unsigned j = 0; j < n_post; ++j) {
            p->post_cs[j].sync.seq = ++seq;
            p->post_cs[j].sync.dep[HK_SYNC_VDM].cut = &p->cs;
            p->post_cs[j].sync.dep_ta[HK_SYNC_VDM].cut = &p->cs;
            list_addtail(&p->post_cs[j].node, &p->cmd.control_streams);
            u_foreach_bit(k, p->post_cs[j].sync.kind)
               p->cmd.sync.last[k][HK_SYNC_CDM] = &p->post_cs[j];
            p->cmd.sync.last_any[HK_SYNC_CDM] = &p->post_cs[j];
         }
         struct hk_cs *last_cdm =
            n_post ? &p->post_cs[n_post - 1] : &p->pre_cs[n_pre - 1];
         for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
            if (cdm_req_cut[k] || p->cmd.sync.req[k][HK_SYNC_CDM].cut)
               p->cmd.sync.req[k][HK_SYNC_CDM].cut = last_cdm;
         }
      }

      perf_debug_dev(&dev->dev,
                     "Packing %u render command buffers (%u draws) into one "
                     "render", run->real_count, p->cs.stats.cmds);
      commands[out++] = &p->cmd.vk;
      i = run->first + run->count;
      if (++r == run_count) {
         while (i < n)
            commands[out++] = submit->command_buffers[i++];
      }
   }

   /* The tails, for this submission. */
   if (!run_count && targets)
      memset(targets, 0, n * sizeof(*targets));
   const bool cached = hk_g17p_cached_cmd_memory(&dev->dev);
   bool written = false;
   for (unsigned i = 0; i < n; ++i) {
      struct hk_render_pack_edges *e = &cmds[i]->render_pack;
      const uint64_t target = targets ? targets[i] : 0;
      if (e->linked == target ||
          !hk_render_pack_write_tail(dev->dev.chip, e, target))
         continue;
      if (cached)
         util_flush_range_no_fence(e->terminate,
                                   AGX_VDM_STREAM_LINK_CORE_LENGTH);
      written = true;
   }
   if (written) {
      __sync_synchronize();
      if (util_has_cache_ops())
         util_post_flush_fence();
   }

   if (!run_count)
      return false;

   assert(out == n - packed + run_count);
   *effective = *submit;
   effective->command_buffers = commands;
   effective->command_buffer_count = out;
   return true;
}

static VkResult
queue_submit_segment(struct hk_device *dev, struct hk_queue *queue,
                     struct vk_queue_submit *submit, bool allow_gpu_waits,
                     bool *untouched)
{
   /* Render packing first: it also writes back the tails of command
    * buffers an earlier submission packed (hk_render_pack_runs()). */
   struct vk_queue_submit rendered;
   struct vk_queue_submit *original = submit;
   if (hk_render_pack_runs(dev, queue, submit, &rendered))
      submit = &rendered;

   /* Fast-path single-CB and sparse-bind submits before taking the alias
    * patch rwlock or calling into submit packing (which carries stack buffers
    * for multi-CB runs). */
   if (submit->command_buffer_count < 2 || submit->buffer_bind_count ||
       submit->image_bind_count || submit->image_opaque_bind_count ||
       !hk_pack_runs_enabled(dev))
      return queue_submit_recordings(dev, queue, submit, original, NULL, 0,
                                     NULL, allow_gpu_waits, untouched);

   struct hk_queue_done_cache done_cache = {0};

   /* Packing copies the command buffers' streams: no patch site may be
    * written meanwhile (host_alias.patch_lock). The copies are the only CPU
    * reads of stream words at submission; the GPU reads each word whole. */
   const bool alias_dynamic = dev->host_alias.dynamic;
   if (unlikely(alias_dynamic))
      u_rwlock_rdlock(&dev->host_alias.patch_lock);
   struct hk_packed_runs runs;
   const bool have_runs =
      hk_pack_runs(dev, queue, submit, submit == original, &runs, &done_cache);
   if (unlikely(alias_dynamic))
      u_rwlock_rdunlock(&dev->host_alias.patch_lock);

   const uint64_t prev_pt = queue->drm.timeline_value;
   if (!have_runs) {
      VkResult result = queue_submit_recordings(dev, queue, submit, original,
                                                NULL, 0, &done_cache,
                                                allow_gpu_waits, untouched);
      if (result == VK_SUCCESS && runs.loop_count >= 2 && runs.cb_hash != 0 &&
          queue->drm.timeline_value > prev_pt)
         hk_pack_obs_arm(queue, runs.cb_hash, submit->command_buffer_count,
                         runs.loop_count, false, queue->drm.timeline_value);
      return result;
   }

   struct vk_queue_submit effective = *submit;
   effective.command_buffer_count = runs.command_count;
   effective.command_buffers = runs.commands;
   VkResult result = queue_submit_recordings(dev, queue, &effective, original,
                                             util_dynarray_begin(runs.extra_bos),
                                             util_dynarray_num_elements(
                                                runs.extra_bos, struct agx_bo *),
                                             &done_cache, allow_gpu_waits,
                                             untouched);
   if (result == VK_SUCCESS && runs.loop_count >= 2 && runs.cb_hash != 0 &&
       queue->drm.timeline_value > prev_pt)
      hk_pack_obs_arm(queue, runs.cb_hash, submit->command_buffer_count,
                      runs.loop_count, runs.obs_used,
                      queue->drm.timeline_value);
   hk_packed_runs_free(dev, &runs);
   return result;
}

static VkResult
queue_submit(struct hk_device *dev, struct hk_queue *queue,
             struct vk_queue_submit *submit, bool *untouched)
{
   if (submit->command_buffer_count <= 1)
      return queue_submit_segment(dev, queue, submit, true, untouched);
   bool needs_split = false;
   for (unsigned i = 1; i < submit->command_buffer_count; ++i) {
      const struct hk_cmd_buffer *cmd =
         (const struct hk_cmd_buffer *)submit->command_buffers[i];
      if (cmd->sync.has_unordered_start_ts) {
         needs_split = true;
         break;
      }
   }
   if (!needs_split)
      return queue_submit_segment(dev, queue, submit, true, untouched);

   *untouched = true;
   unsigned start = 0;
   while (start < submit->command_buffer_count) {
      unsigned end = start + 1;
      while (end < submit->command_buffer_count) {
         const struct hk_cmd_buffer *cmd =
            (const struct hk_cmd_buffer *)submit->command_buffers[end];
         if (cmd->sync.has_unordered_start_ts)
            break;
         end++;
      }

      struct vk_queue_submit seg = *submit;
      seg.command_buffers = submit->command_buffers + start;
      seg.command_buffer_count = end - start;
      if (start > 0) {
         seg.waits = NULL;
         seg.wait_count = 0;
      }
      if (end < submit->command_buffer_count) {
         seg.signals = NULL;
         seg.signal_count = 0;
      }

      bool seg_untouched = false;
      VkResult result =
         queue_submit_segment(dev, queue, &seg, false, &seg_untouched);
      if (start > 0 || !seg_untouched)
         *untouched = false;
      if (result != VK_SUCCESS) {
         hk_gpu_waits_forget_signals(dev, submit->signal_count,
                                     submit->signals);
         return result;
      }
      start = end;
   }

   return VK_SUCCESS;
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

struct vk_command_buffer *
hk_submit_first_errored_cmd_buffer(const struct vk_queue_submit *submit)
{
   for (uint32_t i = 0; i < submit->command_buffer_count; ++i) {
      if (vk_command_buffer_has_error(submit->command_buffers[i]))
         return submit->command_buffers[i];
   }
   return NULL;
}

static VkResult
hk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct hk_queue *queue = container_of(vk_queue, struct hk_queue, vk);
   struct hk_device *dev = hk_queue_device(queue);

   if (vk_device_is_lost_no_report(&dev->vk))
      return VK_ERROR_DEVICE_LOST;

   /* Submitting a command buffer whose recording failed is invalid usage,
    * and its control streams may stop short of their end. The GPU must never
    * run them: the device is lost for this client instead.
    */
   struct vk_command_buffer *errored =
      hk_submit_first_errored_cmd_buffer(submit);
   if (unlikely(errored)) {
      return vk_queue_set_lost(&queue->vk,
                               "submitted a command buffer recorded with %s",
                               vk_Result_to_str(errored->record_result));
   }

   /* While a resource may alias, the command buffers recorded before it get
    * memory scope at their patch sites (hk_cdm_alias_scope()), before
    * anything reads their streams: submit packing copies them, holding
    * host_alias.patch_lock shared (queue_submit()). */
   if (unlikely(dev->host_alias.dynamic) &&
       p_atomic_read(&dev->alias_live))
      hk_submit_patch_alias_sites(dev, submit->command_buffers,
                                  submit->command_buffer_count);

   /* Command buffers recorded before the device held a sparse resource
    * carry the reduced AGX2 barrier word at their patch sites; write the
    * full word before the streams are packed (host_alias.patch_lock, same
    * lifetime rule as the alias sites above). */
   if (unlikely(p_atomic_read(&dev->sparse_used)))
      hk_submit_patch_sparse_sites(dev, submit->command_buffers,
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
         hk_queue_note_signalled(queue, queue->drm.timeline_value);
         VkResult res = dev->vk.check_status(&dev->vk);
         if (result == VK_SUCCESS)
            result = res;
      }
   }

   return result;
}

/* Wait, up to timeout_ns in all, for the GPU to finish everything the
 * application's queues have handed to the kernel, also when the device is
 * lost: its earlier submissions may still run. Work still queued in the
 * runtime's submit thread has not reached the kernel and is not waited for.
 */
void
hk_device_wait_submitted(struct hk_device *dev, uint64_t timeout_ns)
{
   const uint64_t deadline = os_time_get_absolute_timeout(timeout_ns);
   const int64_t abs_timeout = (int64_t)MIN2(deadline, (uint64_t)INT64_MAX);
   bool warned = false;

   list_for_each_entry(struct vk_queue, iter, &dev->vk.queues, link) {
      struct hk_queue *queue = container_of(iter, struct hk_queue, vk);
      uint64_t point = 0;

      if (!queue->drm.syncobj)
         continue;

      /* The last point the kernel accepted, which has a fence: the cursor
       * may sit past it on a point a failed submission skipped. Failed work
       * carries an error fence and does not delay this. */
      if (drmSyncobjQuery2(dev->dev.fd, &queue->drm.syncobj, &point, 1,
                           DRM_SYNCOBJ_QUERY_FLAGS_LAST_SUBMITTED) ||
          !point)
         continue;

      int ret;
      do {
         ret = drmSyncobjTimelineWait(dev->dev.fd, &queue->drm.syncobj, &point,
                                      1, abs_timeout, 0, NULL);
      } while (ret == -EINTR);
      if (ret == -ETIME && !warned) {
         mesa_logw("hk: device destroyed while its GPU work was still running");
         warned = true;
      }
   }
}

void
hk_device_quiesce_lost_submissions(struct hk_device *dev)
{
   if (!vk_device_is_lost_no_report(&dev->vk))
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
   if (semaphore != VK_NULL_HANDLE) {
      hk_device_quiesce_lost_submissions(dev);
      VK_FROM_HANDLE(vk_semaphore, sem, semaphore);
      hk_gpu_waits_forget_destroyed(dev, &sem->permanent, sem->temporary);
   }
   vk_common_DestroySemaphore(_device, semaphore, pAllocator);
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyFence(VkDevice _device, VkFence fence,
                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, _device);
   if (fence != VK_NULL_HANDLE) {
      hk_device_quiesce_lost_submissions(dev);
      VK_FROM_HANDLE(vk_fence, vkf, fence);
      hk_gpu_waits_forget_destroyed(dev, &vkf->permanent, vkf->temporary);
   }
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

/* Vulkan queues of one device that keep a kernel queue of their own
 * (hk_queue_share()).
 *
 * drm_asahi_queue_limits::max_queues is the kernel's machine-wide pool of
 * kernel queues, shared with every other client. One Vulkan queue may hold up
 * to HK_QUEUE_SIBLINGS_MAX of them, its primary and sibling kernel queues.
 * Dividing the pool by one more than that keeps a single device from claiming
 * all of it, leaving room for the device's helper queues and for other
 * clients, and no device needs more than one family advertises. Without
 * independent kernel queues, including under virtio where the limits cannot
 * be queried, the Vulkan queues of a priority class and creation flags share
 * one kernel queue.
 */
uint32_t
hk_queue_independent_budget(struct hk_device *dev)
{
   if (!(dev->dev.params.features & DRM_ASAHI_FEATURE_INDEPENDENT_QUEUES) ||
       dev->dev.is_virtio || (dev->dev.debug & AGX_DBG_1QUEUE))
      return 1;

   const uint32_t max_queues = hk_device_physical(dev)->queue_limits.max_queues;
   return CLAMP(max_queues / (1u + HK_QUEUE_SIBLINGS_MAX), 1u,
                HK_ADVERTISED_QUEUES);
}

/* Vulkan queues of the device that run on the kernel queue of `real`, other
 * than `real` itself. */
static unsigned
hk_queue_sharers(struct hk_device *dev, const struct vk_queue *real)
{
   unsigned sharers = 0;
   vk_foreach_queue(iter, &dev->vk) {
      if (iter != real && iter->real_queue == real)
         sharers++;
   }
   return sharers;
}

void
hk_queue_share(struct hk_device *dev, struct hk_queue *queue)
{
   /* Every Vulkan queue keeps an independent submit worker. The first ones,
    * up to the device's budget (hk_queue_independent_budget()), also get a
    * kernel queue of their own. A later queue shares the kernel queue of an
    * earlier queue of the same priority class and creation flags, the one
    * the fewest queues share, so shared work spreads over all of them; it
    * gets its own only when its class has none yet. Queue families differ in
    * exposed capabilities, but each kernel queue supports both engines. The
    * common runtime serializes callbacks on the backing queue.
    */
   unsigned independent = 0;
   vk_foreach_queue(iter, &dev->vk) {
      if (iter == &queue->vk)
         break;
      if (!vk_queue_is_emulated(iter))
         independent++;
   }
   if (independent < hk_queue_independent_budget(dev))
      return;

   const bool single_kernel_queue = (dev->dev.debug & AGX_DBG_1QUEUE) != 0;
   struct vk_queue *backing = NULL;
   unsigned backing_sharers = 0;
   vk_foreach_queue(iter, &dev->vk) {
      if (iter == &queue->vk)
         break;
      if (vk_queue_is_emulated(iter))
         continue;
      const struct hk_queue *real = container_of(iter, struct hk_queue, vk);
      if (!single_kernel_queue &&
          (real->drm.virt_ring_idx != queue->drm.virt_ring_idx ||
           real->vk.flags != queue->vk.flags))
         continue;
      const unsigned sharers = hk_queue_sharers(dev, iter);
      if (!backing || sharers < backing_sharers) {
         backing = iter;
         backing_sharers = sharers;
      }
   }
   if (!backing)
      return;

   queue->vk.driver_submit = NULL;
   vk_queue_set_emulated(&queue->vk, backing);
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

   /* On M5, quantize relative priorities to the two unprivileged classes;
    * this mapping was qualified with the M5 snapshot. Vulkan queues of one
    * class past the device's budget share kernel queues (hk_queue_share()),
    * and the class is then the only scheduling distinction left between
    * them. Elsewhere relative priorities stay a hint, as the spec allows. An
    * explicit global class takes precedence. No scheduling work is needed
    * during submission.
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
   queue->sync.load_hazard_frag = dev->load_hazard_frag;
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

   p_atomic_set(&queue->drm.id, UINT32_MAX);
   for (unsigned i = 0; i < HK_QUEUE_SIBLINGS_MAX; ++i)
      queue->sib.id[i] = UINT32_MAX;
   queue->sib.count = 1;
   if (hk_queue_sibling_compute_supported(&dev->dev,
                                          queue->sync.exact_prior)) {
      /* Firmware runs compute kicks of one context from sibling kernel queues
       * at once, so independent compute streams overlap
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

   if (agx_chip_g17p_or_m5(dev->dev.chip) &&
       dev->vk.enabled_features.sparseBinding && !kick) {
      queue->g17p_sstt_flush_bo =
         agx_bo_create(&dev->dev, AIL_PAGESIZE, 0, 0, "SSTT flush stream");
      if (!queue->g17p_sstt_flush_bo) {
         hk_queue_finish(dev, queue);
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }

      uint8_t record[AGX3_CDM_SSTT_FLUSH_INVALIDATE_SIZE];
      if (!agx3_cdm_sstt_flush_invalidate_pack(dev->dev.chip, record) ||
          !hk_queue_flush_stream_write(
             dev->dev.chip, agx_bo_map(queue->g17p_sstt_flush_bo),
             queue->g17p_sstt_flush_bo->size, record, sizeof(record))) {
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
   agx_bo_unreference(&dev->dev, queue->ts_carrier_bo);
   agx_bo_unreference(&dev->dev, queue->host_flush_bo);
   util_dynarray_fini(&queue->sync_resolved);
   hk_queue_pack_free_packets(queue);
   hk_queue_rpack_free_packets(queue);
   util_dynarray_fini(&queue->rpack.commands);
   util_dynarray_fini(&queue->rpack.runs);
   util_dynarray_fini(&queue->rpack.targets);
   util_dynarray_fini(&queue->pack.members);
   util_dynarray_fini(&queue->pack.runs);
   util_dynarray_fini(&queue->pack.commands);
   util_dynarray_fini(&queue->pack.fused);
   util_dynarray_fini(&queue->pack.plans);
   util_dynarray_fini(&queue->pack.planned_suppressed);
   util_dynarray_fini(&queue->pack.suppressed_arr);
   util_dynarray_fini(&queue->pack.run_sup);
   util_dynarray_fini(&queue->pack.extra_bos);
   vk_free(&dev->vk.alloc, queue);
}

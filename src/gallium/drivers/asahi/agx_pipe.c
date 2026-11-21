/*
 * Copyright 2010 Red Hat Inc.
 * Copyright 2014-2017 Broadcom
 * Copyright 2019-2020 Collabora, Ltd.
 * Copyright 2006 VMware, Inc.
 * SPDX-License-Identifier: MIT
 */
#include <errno.h>
#include <stdio.h>
#include <xf86drm.h>
#include <linux/dma-buf.h>
#include "asahi/compiler/agx_compile.h"
#include "asahi/layout/layout.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx_g17p_compression.h"
#include "asahi/lib/agx_zls_pixels.h"
#include "asahi/lib/decode.h"
#include "drm-uapi/asahi_drm.h"
#include "drm-uapi/drm_fourcc.h"
#include "frontend/winsys_handle.h"
#include "gallium/auxiliary/renderonly/renderonly.h"
#include "gallium/auxiliary/util/u_debug_cb.h"
#include "gallium/auxiliary/util/u_framebuffer.h"
#include "gallium/auxiliary/util/u_sample_positions.h"
#include "gallium/auxiliary/util/u_surface.h"
#include "gallium/auxiliary/util/u_transfer.h"
#include "gallium/auxiliary/util/u_transfer_helper.h"
#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/bitscan.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/half_float.h"
#include "util/macros.h"
#include "util/os_misc.h"
#include "util/simple_mtx.h"
#include "util/timespec.h"
#include "util/u_drm.h"
#include "util/u_gen_mipmap.h"
#include "util/u_helpers.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_process.h"
#include "util/u_resource.h"
#include "util/u_screen.h"
#include "util/u_upload_mgr.h"
#include "util/xmlconfig.h"
#include "agx_bg_eot.h"
#include "agx_bo.h"
#include "agx_device.h"
#include "agx_disk_cache.h"
#include "agx_fence.h"
#include "agx_helpers.h"
#include "agx_native_descriptor.h"
#include "agx_pack.h"
#include "agx_public.h"
#include "agx_state.h"
#include "agx_tilebuffer.h"
#include "shader_enums.h"

uint64_t agx_best_modifiers[] = {
   DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED,
   DRM_FORMAT_MOD_APPLE_GPU_TILED,
   DRM_FORMAT_MOD_LINEAR,
};

/* These limits are arbitrarily chosen and subject to change as
 * we discover more workloads with heavy shadowing.
 *
 * Maximum size of a shadowed object in bytes.
 * Hint: 1024x1024xRGBA8 = 4 MiB. Go higher for compression.
 */
#define MAX_SHADOW_BYTES (6 * 1024 * 1024)

/* Maximum cumulative size to shadow an object before we flush.
 * Allows shadowing a 4MiB + meta object 8 times with the logic
 * below (+1 shadow offset implied).
 */
#define MAX_TOTAL_SHADOW_BYTES (32 * 1024 * 1024)

void agx_init_state_functions(struct pipe_context *ctx);

/*
 * resource
 */

const static char *s_tiling[] = {
   [AIL_TILING_LINEAR] = "LINR",
   [AIL_TILING_GPU] = "GPU",
};

#define rsrc_debug(res, ...)                                                   \
   do {                                                                        \
      if (agx_device((res)->base.screen)->debug & AGX_DBG_RESOURCE)            \
         agx_msg(__VA_ARGS__);                                                 \
   } while (0)

static void
agx_resource_debug(struct agx_resource *res, const char *msg)
{
   if (!(agx_device(res->base.screen)->debug & AGX_DBG_RESOURCE))
      return;

   int ino = -1;
   if (res->bo->prime_fd >= 0) {
      struct stat sb;
      if (!fstat(res->bo->prime_fd, &sb))
         ino = sb.st_ino;
   }

   agx_msg(
      "%s%s %dx%dx%d %dL %d/%dM %dS M:%llx %s%s %s%s S:0x%llx LS:0x%llx CS:0x%llx "
      "Base=0x%llx Size=0x%llx Meta=0x%llx/0x%llx (%s) %s%s%s%s%s%sfd:%d(%d) B:%x H:%x/%x @ %p\n",
      msg ?: "", util_format_short_name(res->base.format), res->base.width0,
      res->base.height0, res->base.depth0, res->base.array_size,
      res->base.last_level, res->layout.levels, res->layout.sample_count_sa,
      (long long)res->modifier, s_tiling[res->layout.tiling],
      res->layout.compressed ? " COMP" : "",
      res->layout.mipmapped_z ? "MZ " : "",
      res->layout.page_aligned_layers ? "PL " : "",
      (long long)res->layout.linear_stride_B,
      (long long)res->layout.layer_stride_B,
      (long long)res->layout.compression_layer_stride_B,
      (long long)res->bo->va->addr, (long long)res->layout.size_B,
      res->layout.metadata_offset_B
         ? ((long long)res->bo->va->addr + res->layout.metadata_offset_B)
         : 0,
      (long long)res->layout.metadata_offset_B, res->bo->label,
      res->bo->flags & AGX_BO_SHARED ? "SH " : "",
      res->bo->flags & AGX_BO_LOW_VA ? "LO " : "",
      res->bo->flags & AGX_BO_EXEC ? "EX " : "",
      res->bo->flags & AGX_BO_WRITEBACK ? "WB " : "",
      res->bo->flags & AGX_BO_SHAREABLE ? "SA " : "",
      res->bo->flags & AGX_BO_READONLY ? "RO " : "", res->bo->prime_fd, ino,
      res->base.bind, res->bo->handle, res->bo->uapi_handle, res);
}

static void
agx_resource_setup(struct agx_device *dev, struct agx_resource *nresource)
{
   struct pipe_resource *templ = &nresource->base;

   nresource->layout = (struct ail_layout){
      .policy = ail_layout_policy_for_descriptor_generation(
         agx_chip_descriptor_generation(dev->chip)),
      .tiling = ail_drm_modifier_to_tiling(nresource->modifier),
      .compressed = ail_is_drm_modifier_compressed(nresource->modifier),
      .mipmapped_z = templ->target == PIPE_TEXTURE_3D,
      .format = templ->format,
      .width_px = templ->width0,
      .height_px = templ->height0,
      .depth_px = templ->depth0 * templ->array_size,
      .sample_count_sa = MAX2(templ->nr_samples, 1),
      .levels = templ->last_level + 1,
      .writeable_image = templ->bind & PIPE_BIND_SHADER_IMAGE,
      /* G18G keeps sample-dependent metadata tiles and compact Z/S layers. */
      .g18g = dev->chip == AGX_CHIP_G18G,

      /* Ostensibly this should be based on the bind, but Gallium bind flags are
       * notoriously unreliable. The only cost of setting this excessively is a
       * bit of extra memory use for layered textures, which isn't worth trying
       * to optimize.
       */
      .renderable = true,
   };
}

static void
agx_resource_set_damage_region(struct pipe_screen *screen,
                               struct pipe_resource *resource,
                               unsigned int nrects,
                               const struct pipe_box *rects)
{
   /* A valid Asahi render target is conservatively reloaded in full when a
    * batch starts. That is already the strongest implementation of
    * KHR_partial_update's preservation contract: pixels outside the supplied
    * region survive even when an application underestimates its damage.
    *
    * Keep the hint side-effect free until the command stream has a native
    * tile-enable map. Restricting the render area without one would skip the
    * stores needed to preserve untouched tiles across GBM buffer reuse.
    */
   (void)screen;
   (void)resource;
   (void)nrects;
   (void)rects;
}

static struct pipe_resource *
agx_resource_from_handle(struct pipe_screen *pscreen,
                         const struct pipe_resource *templat,
                         struct winsys_handle *whandle, unsigned usage)
{
   struct agx_device *dev = agx_device(pscreen);
   struct agx_resource *rsc;
   struct pipe_resource *prsc;

   assert(whandle->type == WINSYS_HANDLE_TYPE_FD);

   rsc = CALLOC_STRUCT(agx_resource);
   if (!rsc)
      return NULL;

   rsc->modifier = whandle->modifier == DRM_FORMAT_MOD_INVALID
                      ? DRM_FORMAT_MOD_LINEAR
                      : whandle->modifier;

   /* We need strides to be aligned. ail asserts this, but we want to fail
    * gracefully so the app can handle the error.
    */
   if (rsc->modifier == DRM_FORMAT_MOD_LINEAR && (whandle->stride % 16) != 0) {
      FREE(rsc);
      return NULL;
   }

   prsc = &rsc->base;

   *prsc = *templat;

   pipe_reference_init(&prsc->reference, 1);
   prsc->screen = pscreen;

   /* Set PIPE_BIND_SCANOUT for lazy on-demand creation of renderonly
    * scanout resource if agx_resource_get_handle is called for
    * WINSYS_HANDLE_TYPE_KMS on a kms-ro screen.
    */
   prsc->bind |= PIPE_BIND_SHARED | PIPE_BIND_SCANOUT;

   agx_resource_setup(dev, rsc);

   if (rsc->layout.tiling == AIL_TILING_LINEAR) {
      rsc->layout.linear_stride_B = whandle->stride;
      rsc->layout.level_offsets_B[0] = whandle->offset;
   } else if (whandle->stride != ail_get_wsi_stride_B(&rsc->layout, 0) ||
              whandle->offset != 0) {
      FREE(rsc);
      return NULL;
   }

   ail_make_miptree(&rsc->layout);

   /* Validate all modifier geometry before acquiring a BO reference.  A
    * rejected Wayland/X11 dma-buf import is routine during resize and format
    * renegotiation; importing first leaked one reference for every malformed
    * tiled stride or offset.
    */
   rsc->bo = agx_bo_import(dev, whandle->handle);
   /* Sometimes an import can fail e.g. on an invalid buffer fd, out of
    * memory space to mmap it etc.
    */
   if (!rsc->bo) {
      FREE(rsc);
      return NULL;
   }

   /* The modifier layout includes metadata and any external base offset. Do
    * not let a short dma-buf become a GPU out-of-bounds access. */
   if (rsc->layout.size_B > rsc->bo->size) {
      agx_bo_unreference(dev, rsc->bo);
      FREE(rsc);
      return NULL;
   }

   if (prsc->target == PIPE_BUFFER) {
      assert(rsc->layout.tiling == AIL_TILING_LINEAR);
      util_range_init(&rsc->valid_buffer_range);
   }

   agx_resource_debug(rsc, "Import: ");

   return prsc;
}

static bool
agx_resource_get_handle(struct pipe_screen *pscreen, struct pipe_context *ctx,
                        struct pipe_resource *pt, struct winsys_handle *handle,
                        unsigned usage)
{
   struct agx_device *dev = agx_device(pscreen);
   struct pipe_resource *cur = pt;

   /* Even though asahi doesn't support multi-planar formats, we
    * can get here through GBM, which does. Walk the list of planes
    * to find the right one.
    */
   for (int i = 0; i < handle->plane; i++) {
      cur = cur->next;
      if (!cur)
         return false;
   }

   struct agx_resource *rsrc = agx_resource(cur);

   if (handle->type == WINSYS_HANDLE_TYPE_KMS && dev->ro) {
      rsrc_debug(rsrc, "Get handle: %p (KMS RO)\n", rsrc);

      if (!rsrc->scanout && dev->ro && (rsrc->base.bind & PIPE_BIND_SCANOUT)) {
         rsrc->scanout =
            renderonly_scanout_for_resource(&rsrc->base, dev->ro, NULL);
      }

      if (!rsrc->scanout)
         return false;

      return renderonly_get_handle(rsrc->scanout, handle);
   } else if (handle->type == WINSYS_HANDLE_TYPE_KMS) {
      rsrc_debug(rsrc, "Get handle: %p (KMS)\n", rsrc);

      /* BO must be considered shared at this point. */
      if (!agx_bo_make_shared(dev, rsrc->bo))
         return false;

      handle->handle = rsrc->bo->handle;
   } else if (handle->type == WINSYS_HANDLE_TYPE_FD) {
      int fd = agx_bo_export(dev, rsrc->bo);

      if (fd < 0)
         return false;

      handle->handle = fd;
      if (dev->debug & AGX_DBG_RESOURCE) {
         struct stat sb;
         fstat(rsrc->bo->prime_fd, &sb);
         agx_msg("Get handle: %p (FD %d/%ld)\n", rsrc, fd, (long)sb.st_ino);
      }
   } else {
      /* Other handle types not supported */
      return false;
   }

   handle->stride = ail_get_wsi_stride_B(&rsrc->layout, 0);
   handle->size = rsrc->layout.size_B;
   handle->offset = rsrc->layout.level_offsets_B[0];
   handle->format = rsrc->layout.format;
   handle->modifier = rsrc->modifier;

   return true;
}

static bool
agx_resource_get_param(struct pipe_screen *pscreen, struct pipe_context *pctx,
                       struct pipe_resource *prsc, unsigned plane,
                       unsigned layer, unsigned level,
                       enum pipe_resource_param param, unsigned usage,
                       uint64_t *value)
{
   struct agx_resource *rsrc =
      (struct agx_resource *)util_resource_at_index(prsc, plane);

   switch (param) {
   case PIPE_RESOURCE_PARAM_STRIDE:
      *value = ail_get_wsi_stride_B(&rsrc->layout, level);
      return true;
   case PIPE_RESOURCE_PARAM_OFFSET:
      *value = rsrc->layout.level_offsets_B[level];
      return true;
   case PIPE_RESOURCE_PARAM_MODIFIER:
      *value = rsrc->modifier;
      return true;
   case PIPE_RESOURCE_PARAM_NPLANES:
      /* We don't support multi-planar formats, but we should still handle
       * this case for GBM shared resources.
       */
      *value = util_resource_num(prsc);
      return true;
   default:
      return false;
   }
}

static bool
agx_is_2d(enum pipe_texture_target target)
{
   return (target == PIPE_TEXTURE_2D || target == PIPE_TEXTURE_RECT);
}

static bool
agx_linear_allowed(const struct agx_resource *pres)
{
   /* Mipmapping not allowed with linear */
   if (pres->base.last_level != 0)
      return false;

   /* Depth/stencil buffers must not be linear */
   if (pres->base.bind & PIPE_BIND_DEPTH_STENCIL)
      return false;

   /* Multisampling not allowed with linear */
   if (pres->base.nr_samples > 1)
      return false;

   /* Block compression not allowed with linear */
   if (util_format_is_compressed(pres->base.format))
      return false;

   switch (pres->base.target) {
   /* Buffers are always linear, even with image atomics */
   case PIPE_BUFFER:

   /* Linear textures require specifying their strides explicitly, which only
    * works for 2D textures. Rectangle textures are a special case of 2D.
    *
    * 1D textures only exist in GLES and are lowered to 2D to bypass hardware
    * limitations.
    *
    * However, we don't want to support this case in the image atomic
    * implementation, so linear shader images are specially forbidden.
    */
   case PIPE_TEXTURE_1D:
   case PIPE_TEXTURE_1D_ARRAY:
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_RECT:
      if (pres->base.bind & PIPE_BIND_SHADER_IMAGE)
         return false;

      break;

   /* No other texture type can specify a stride */
   default:
      return false;
   }

   return true;
}

static bool
agx_twiddled_allowed(const struct agx_resource *pres)
{
   /* Certain binds force linear */
   if (pres->base.bind & (PIPE_BIND_DISPLAY_TARGET | PIPE_BIND_LINEAR))
      return false;

   /* Buffers must be linear */
   if (pres->base.target == PIPE_BUFFER)
      return false;

   /* Anything else may be twiddled */
   return true;
}

/* G17P depth: a private, single-level, single-layer, single-sample Z32F
 * depth buffer, as Honeykrisp compresses (hk_g17p_compression_depth_format()).
 * ZLS writes it compressed through the metadata agx_batch already passes, and
 * the compressed Gen9 2D texture packet reads it. On the M4 Max this is
 * byte-identical to plain depth through depth testing across passes,
 * readback, sampling, shadow comparison, blits and scissored clears. */
static bool
agx_g17p_private_depth_compression_allowed(const struct agx_device *dev,
                                           const struct pipe_resource *templ)
{
   return dev->chip == AGX_CHIP_G17P &&
          templ->target == PIPE_TEXTURE_2D && templ->depth0 == 1 &&
          templ->array_size <= 1 && templ->last_level == 0 &&
          templ->nr_samples <= 1 && templ->format == PIPE_FORMAT_Z32_FLOAT &&
          !(templ->bind & ~(PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_DEPTH_STENCIL));
}

/* G17P compresses a private, single-level, single-layer, single-sample 2D
 * colour image of agx_g17p_compression_color_format(): written by the
 * compressed Gen9 2D PBE (end-of-tile stores, blits, staging uploads) and
 * read by the compressed texture descriptor, the packets Honeykrisp
 * compresses its own G17P colour images with. Nothing decompresses such an
 * image in place: its format never spills from the tilebuffer, and it is
 * neither shared nor scanned out, so a feedback loop or a shader-image
 * binding reallocates it uncompressed (agx_decompress()), as on AGX2.
 */
static bool
agx_g17p_private_compression_allowed(const struct agx_device *dev,
                                     const struct pipe_resource *templ)
{
   return dev->chip == AGX_CHIP_G17P && templ->target == PIPE_TEXTURE_2D &&
          templ->depth0 == 1 && templ->array_size <= 1 &&
          templ->last_level == 0 && templ->nr_samples <= 1 &&
          !(templ->bind &
            ~(PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET)) &&
          agx_g17p_compression_color_format(templ->format);
}

static bool
agx_compression_allowed(const struct agx_resource *pres)
{
   const struct agx_device *dev = agx_device(pres->base.screen);
   const enum agx_descriptor_generation generation =
      agx_chip_descriptor_generation(dev->chip);

   /* Allow disabling compression for debugging */
   if (dev->debug & AGX_DBG_NOCOMPRESS) {
      rsrc_debug(pres, "No compression: disabled\n");
      return false;
   }

   /* Compression changes both the image allocation and every write
    * descriptor of it. AGX2 handles any compressed image, decompressing in
    * place where needed (agx_gallium_shared_compression_supported()). AGX3
    * has no in-place decompression, so it compresses only the G17P images
    * that never need it.
    */
   if (agx_gallium_descriptor_generation_is_agx3(generation)) {
      if (!agx_g17p_private_compression_allowed(dev, &pres->base) &&
          !agx_g17p_private_depth_compression_allowed(dev, &pres->base)) {
         rsrc_debug(pres, "No compression: outside the G17P policy\n");
         return false;
      }
   } else if (!agx_gallium_shared_compression_supported(generation)) {
      rsrc_debug(pres, "No compression: no native PBE encoding\n");
      return false;
   }

   /* Limited to renderable */
   if (pres->base.bind &
       ~(PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET |
         PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SHARED | PIPE_BIND_SCANOUT)) {
      rsrc_debug(pres, "No compression: not renderable\n");
      return false;
   }

   if (!ail_can_compress(pres->base.format, pres->base.width0,
                         pres->base.height0, MAX2(pres->base.nr_samples, 1))) {
      rsrc_debug(pres, "No compression: incompatible layout\n");
      return false;
   }

   if (pres->base.format == PIPE_FORMAT_R9G9B9E5_FLOAT) {
      rsrc_debug(pres, "No compression: RGB9E5 copies need work\n");
      return false;
   }

   return true;
}

static uint64_t
agx_select_modifier_from_list(const struct agx_resource *pres,
                              const uint64_t *modifiers, int count)
{
   if (agx_twiddled_allowed(pres) && agx_compression_allowed(pres) &&
       drm_find_modifier(DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED, modifiers,
                         count))
      return DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED;

   if (agx_twiddled_allowed(pres) &&
       drm_find_modifier(DRM_FORMAT_MOD_APPLE_GPU_TILED, modifiers, count))
      return DRM_FORMAT_MOD_APPLE_GPU_TILED;

   if (agx_linear_allowed(pres) &&
       drm_find_modifier(DRM_FORMAT_MOD_LINEAR, modifiers, count))
      return DRM_FORMAT_MOD_LINEAR;

   /* We didn't find anything */
   return DRM_FORMAT_MOD_INVALID;
}

static uint64_t
agx_select_best_modifier(const struct agx_resource *pres)
{
   /* Prefer linear for staging resources, which should be as fast as possible
    * to write from the CPU.
    */
   if (agx_linear_allowed(pres) && pres->base.usage == PIPE_USAGE_STAGING)
      return DRM_FORMAT_MOD_LINEAR;

   /* For SCANOUT or SHARED resources with no explicit modifier selection, force
    * linear since we cannot expect consumers to correctly pass through the
    * modifier (unless linear is not allowed at all).
    */
   if (agx_linear_allowed(pres) &&
       pres->base.bind & (PIPE_BIND_SCANOUT | PIPE_BIND_SHARED)) {
      return DRM_FORMAT_MOD_LINEAR;
   }

   if (agx_twiddled_allowed(pres)) {
      if (agx_compression_allowed(pres))
         return DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED;
      else
         return DRM_FORMAT_MOD_APPLE_GPU_TILED;
   }

   if (agx_linear_allowed(pres))
      return DRM_FORMAT_MOD_LINEAR;
   else
      return DRM_FORMAT_MOD_INVALID;
}

static struct pipe_resource *
agx_resource_create_with_modifiers(struct pipe_screen *screen,
                                   const struct pipe_resource *templ,
                                   const uint64_t *modifiers, int count)
{
   struct agx_device *dev = agx_device(screen);
   struct agx_resource *nresource;

   nresource = CALLOC_STRUCT(agx_resource);
   if (!nresource)
      return NULL;

   nresource->base = *templ;
   nresource->base.screen = screen;

   if (modifiers) {
      nresource->modifier =
         agx_select_modifier_from_list(nresource, modifiers, count);
   } else {
      nresource->modifier = agx_select_best_modifier(nresource);
   }

   /* There may not be a matching modifier, bail if so */
   if (nresource->modifier == DRM_FORMAT_MOD_INVALID) {
      free(nresource);
      return NULL;
   }

   /* If there's only 1 layer and there's no compression, there's no harm in
    * inferring the shader image flag. Do so to avoid reallocation in case the
    * resource is later used as an image.
    */
   if (!ail_is_drm_modifier_compressed(nresource->modifier) &&
       templ->depth0 == 1) {

      nresource->base.bind |= PIPE_BIND_SHADER_IMAGE;
   }

   nresource->mipmapped = (templ->last_level > 0);

   assert(templ->format != PIPE_FORMAT_Z24X8_UNORM &&
          templ->format != PIPE_FORMAT_Z24_UNORM_S8_UINT &&
          "u_transfer_helper should have lowered");

   agx_resource_setup(dev, nresource);

   pipe_reference_init(&nresource->base.reference, 1);

   ail_make_miptree(&nresource->layout);

   /* Fail Piglit's obnoxious allocations */
   if (nresource->layout.size_B >= (1ull << 32)) {
      free(nresource);
      return NULL;
   }

   if (templ->target == PIPE_BUFFER) {
      assert(nresource->layout.tiling == AIL_TILING_LINEAR);
      util_range_init(&nresource->valid_buffer_range);
   }

   /* Guess a label based on the bind */
   unsigned bind = templ->bind;

   const char *label = (bind & PIPE_BIND_INDEX_BUFFER)     ? "Index buffer"
                       : (bind & PIPE_BIND_SCANOUT)        ? "Scanout"
                       : (bind & PIPE_BIND_DISPLAY_TARGET) ? "Display target"
                       : (bind & PIPE_BIND_SHARED)         ? "Shared resource"
                       : (bind & PIPE_BIND_RENDER_TARGET)  ? "Render target"
                       : (bind & PIPE_BIND_DEPTH_STENCIL)
                          ? "Depth/stencil buffer"
                       : (bind & PIPE_BIND_SAMPLER_VIEW)    ? "Texture"
                       : (bind & PIPE_BIND_VERTEX_BUFFER)   ? "Vertex buffer"
                       : (bind & PIPE_BIND_CONSTANT_BUFFER) ? "Constant buffer"
                       : (bind & PIPE_BIND_GLOBAL)          ? "Global memory"
                       : (bind & PIPE_BIND_SHADER_BUFFER)   ? "Shader buffer"
                       : (bind & PIPE_BIND_SHADER_IMAGE)    ? "Shader image"
                                                            : "Other resource";

   uint32_t create_flags = 0;

   /* Default to write-combine resources, but use writeback if that is expected
    * to be beneficial.
    */
   if (nresource->base.usage == PIPE_USAGE_STAGING ||
       (nresource->base.flags & PIPE_RESOURCE_FLAG_MAP_COHERENT)) {

      create_flags |= AGX_BO_WRITEBACK;
   }

   /* Allow disabling write-combine to debug performance issues */
   if (dev->debug & AGX_DBG_NOWC) {
      create_flags |= AGX_BO_WRITEBACK;
   }

   /* Create buffers that might be shared with the SHAREABLE flag */
   if (bind & (PIPE_BIND_SCANOUT | PIPE_BIND_DISPLAY_TARGET | PIPE_BIND_SHARED))
      create_flags |= AGX_BO_SHAREABLE;

   nresource->bo =
      agx_bo_create(dev, nresource->layout.size_B, 0, create_flags, label);

   if (!nresource->bo) {
      FREE(nresource);
      return NULL;
   }

   agx_resource_debug(nresource, "New: ");
   return &nresource->base;
}

static struct pipe_resource *
agx_resource_create(struct pipe_screen *screen,
                    const struct pipe_resource *templ)
{
   return agx_resource_create_with_modifiers(screen, templ, NULL, 0);
}

static void
agx_resource_destroy(struct pipe_screen *screen, struct pipe_resource *prsrc)
{
   struct agx_resource *rsrc = (struct agx_resource *)prsrc;
   struct agx_screen *agx_screen = (struct agx_screen *)screen;

   agx_resource_debug(rsrc, "Destroy: ");

   if (prsrc->target == PIPE_BUFFER)
      util_range_destroy(&rsrc->valid_buffer_range);

   if (rsrc->scanout)
      renderonly_scanout_destroy(rsrc->scanout, agx_screen->dev.ro);

   agx_bo_unreference(&agx_screen->dev, rsrc->bo);
   FREE(rsrc);
}

void
agx_batch_track_image(struct agx_batch *batch, struct pipe_image_view *image)
{
   struct agx_resource *rsrc = agx_resource(image->resource);

   if (image->shader_access & PIPE_IMAGE_ACCESS_WRITE) {
      batch->incoherent_writes = true;

      if (rsrc->base.target == PIPE_BUFFER) {
         agx_batch_writes_range(batch, rsrc, image->u.buf.offset,
                                image->u.buf.size);
      } else {
         agx_batch_writes(batch, rsrc, image->u.tex.level);
      }
   } else {
      agx_batch_reads(batch, rsrc);
   }
}

/*
 * transfer
 */

static void
agx_transfer_flush_region(struct pipe_context *pipe,
                          struct pipe_transfer *transfer,
                          const struct pipe_box *box)
{
}

/* Reallocate the backing buffer of a resource, returns true if successful */
static bool
agx_shadow(struct agx_context *ctx, struct agx_resource *rsrc, bool needs_copy)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   struct agx_bo *old = rsrc->bo;
   size_t size = rsrc->layout.size_B;
   unsigned flags = old->flags;

   if (dev->debug & AGX_DBG_NOSHADOW)
      return false;

   /* If a resource is (or could be) shared, shadowing would desync across
    * processes. (It's also not what this path is for.)
    */
   if (flags & (AGX_BO_SHARED | AGX_BO_SHAREABLE))
      return false;

   /* Do not shadow resources that are too large */
   if (size > MAX_SHADOW_BYTES && needs_copy)
      return false;

   /* Do not shadow resources too much */
   if (rsrc->shadowed_bytes >= MAX_TOTAL_SHADOW_BYTES && needs_copy)
      return false;

   rsrc->shadowed_bytes += size;

   /* If we need to copy, we reallocate the resource with cached-coherent
    * memory. This is a heuristic: it assumes that if the app needs a shadows
    * (with a copy) now, it will again need to shadow-and-copy the same resource
    * in the future. This accelerates the later copies, since otherwise the copy
    * involves reading uncached memory.
    */
   if (needs_copy)
      flags |= AGX_BO_WRITEBACK;

   struct agx_bo *new_ = agx_bo_create(dev, size, 0, flags, old->label);

   /* If allocation failed, we can fallback on a flush gracefully*/
   if (new_ == NULL)
      return false;

   if (needs_copy) {
      perf_debug_ctx(ctx, "Shadowing %zu bytes on the CPU (%s)", size,
                     (old->flags & AGX_BO_WRITEBACK) ? "cached" : "uncached");
      agx_resource_debug(rsrc, "Shadowed: ");

      memcpy(agx_bo_map(new_), agx_bo_map(old), size);
   }

   /* Swap the pointers, dropping a reference */
   agx_bo_unreference(dev, rsrc->bo);
   rsrc->bo = new_;

   /* Reemit descriptors using this resource */
   agx_dirty_all(ctx);
   return true;
}

/*
 * Perform the required synchronization before a transfer_map operation can
 * complete. This may require syncing batches.
 */
/*
 * A writer on another context is not in this context's writer table. That is
 * the window-system case: the EGL context renders a GBM surface buffer and
 * flushes it at eglSwapBuffers, and gbm_bo_map() maps it through the GBM
 * device's own context, which then read the buffer before the GPU had
 * written it (gbm-scanout-test read black on the M5 Pro unless the client
 * called glFinish first). Wait for the syncobj the writing submission left
 * on the BO (agx_batch_submit records it for every BO a batch writes), and
 * for the implicit fences of an exported BO, which also cover writers in
 * other processes.
 */
static void
agx_sync_foreign_writer(struct agx_context *ctx, struct agx_resource *rsrc)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   struct agx_bo *bos[2] = {rsrc->bo,
                            rsrc->separate_stencil ? rsrc->separate_stencil->bo
                                                   : NULL};

   for (unsigned i = 0; i < ARRAY_SIZE(bos); ++i) {
      struct agx_bo *bo = bos[i];
      if (!bo)
         continue;

      const uint64_t writer = p_atomic_read(&bo->writer);
      if (writer && agx_bo_writer_queue(writer) != ctx->queue_id) {
         uint32_t syncobj = agx_bo_writer_syncobj(writer);
         int ret;
         do {
            ret = drmSyncobjWait(dev->fd, &syncobj, 1, INT64_MAX,
                                 DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);
         } while (ret == -EINTR);
      }

      if ((bo->flags & AGX_BO_SHARED) && bo->prime_fd >= 0) {
         struct dma_buf_sync sync = {
            .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ,
         };
         if (!drmIoctl(bo->prime_fd, DMA_BUF_IOCTL_SYNC, &sync)) {
            sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
            drmIoctl(bo->prime_fd, DMA_BUF_IOCTL_SYNC, &sync);
         }
      }
   }
}

static void
agx_prepare_for_map(struct agx_context *ctx, struct agx_resource *rsrc,
                    unsigned level,
                    unsigned usage, /* a combination of PIPE_MAP_x */
                    const struct pipe_box *box, bool staging_blit)
{
   /* GPU access does not require explicit syncs, as the batch tracking logic
    * will ensure correct ordering automatically.
    */
   if (staging_blit)
      return;

   /* Before the validity test: a write on another context need not have
    * marked the level valid here. */
   if (!(usage & PIPE_MAP_UNSYNCHRONIZED))
      agx_sync_foreign_writer(ctx, rsrc);

   /* If the level has not been written, we may freely do CPU access (writes),
    * even if other levels are being written by the GPU. This lets us write some
    * mip levels on the CPU and some on the GPU, without stalling.
    */
   if (!agx_resource_valid(rsrc, level))
      return;

   /* Upgrade DISCARD_RANGE to WHOLE_RESOURCE if the whole resource is
    * being mapped.
    */
   if ((usage & PIPE_MAP_DISCARD_RANGE) &&
       !(rsrc->base.flags & PIPE_RESOURCE_FLAG_MAP_PERSISTENT) &&
       rsrc->base.last_level == 0 &&
       util_texrange_covers_whole_level(&rsrc->base, 0, box->x, box->y, box->z,
                                        box->width, box->height, box->depth)) {

      usage |= PIPE_MAP_DISCARD_WHOLE_RESOURCE;
   }

   /* Shadowing doesn't work separate stencil or shared resources */
   if (rsrc->separate_stencil || (rsrc->bo->flags & AGX_BO_SHARED))
      usage &= ~PIPE_MAP_DISCARD_WHOLE_RESOURCE;

   /* If the access is unsynchronized, there's nothing to do */
   if (usage & PIPE_MAP_UNSYNCHRONIZED)
      return;

   /* If the range being accessed is uninitialized, we do not need to sync. */
   if (rsrc->base.target == PIPE_BUFFER && !(rsrc->bo->flags & AGX_BO_SHARED) &&
       !util_ranges_intersect(&rsrc->valid_buffer_range, box->x,
                              box->x + box->width))
      return;

   /* Everything after this needs the context, which is not safe for
    * unsynchronized transfers when we claim
    * pipe_caps.map_unsynchronized_thread_safe.
    */
   assert(!(usage & PIPE_MAP_UNSYNCHRONIZED));

   /* Reading or writing from the CPU requires syncing writers. */
   agx_sync_writer(ctx, rsrc, "Unsynchronized CPU transfer");

   /* Additionally, writing needs readers synced. */
   if (!(usage & PIPE_MAP_WRITE))
      return;

   /* If there are no readers, we're done. We check at the start to
    * avoid expensive shadowing paths or duplicated checks in this hapyp path.
    */
   if (!agx_any_batch_uses_resource(ctx, rsrc)) {
      rsrc->shadowed_bytes = 0;
      return;
   }

   /* There are readers. Try to invalidate the resource to avoid a sync */
   if ((usage & PIPE_MAP_DISCARD_WHOLE_RESOURCE) &&
       agx_shadow(ctx, rsrc, false))
      return;

   /* Or try to shadow it */
   if (!(rsrc->base.flags & PIPE_RESOURCE_FLAG_MAP_PERSISTENT) &&
       agx_shadow(ctx, rsrc, true))
      return;

   /* Otherwise, we need to sync */
   agx_sync_readers(ctx, rsrc, "Unsynchronized write");

   rsrc->shadowed_bytes = 0;
}

/*
 * Return a colour-renderable format compatible with a depth/stencil format, to
 * be used as an interchange format for depth/stencil blits. For
 * non-depth/stencil formats, returns the format itself, except when that format
 * would not round-trip so we return a compatible roundtrippable format.
 */
static enum pipe_format
agx_staging_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_Z16_UNORM:
      return PIPE_FORMAT_R16_UNORM;
   case PIPE_FORMAT_Z32_FLOAT:
      return PIPE_FORMAT_R32_FLOAT;
   case PIPE_FORMAT_S8_UINT:
      return PIPE_FORMAT_R8_UINT;
   default:
      /* Z24 and combined Z/S are lowered to one of the above formats by
       * u_transfer_helper. The caller needs to pass in the rsrc->layout.format
       * and not the rsrc->base.format to get the lowered physical format
       * (rather than the API logical format).
       */
      assert(!util_format_is_depth_or_stencil(format) &&
             "no other depth/stencil formats allowed for staging");

      /* However, snorm does not round trip, so don't use that for staging */
      return util_format_snorm_to_sint(format);
   }
}

/* Most of the time we can do CPU-side transfers, but sometimes we need to use
 * the 3D pipe for this. Let's wrap u_blitter to blit to/from staging textures.
 * Code adapted from panfrost */

static struct agx_resource *
agx_alloc_staging(struct pipe_screen *screen, struct agx_resource *rsc,
                  unsigned level, const struct pipe_box *box)
{
   struct pipe_resource tmpl = rsc->base;

   tmpl.usage = PIPE_USAGE_STAGING;
   tmpl.width0 = box->width;
   tmpl.height0 = box->height;
   tmpl.depth0 = 1;

   /* We need a linear staging resource. We have linear 2D arrays, but not
    * linear 3D or cube textures. So switch to 2D arrays if needed.
    */
   switch (tmpl.target) {
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
   case PIPE_TEXTURE_3D:
      tmpl.target = PIPE_TEXTURE_2D_ARRAY;
      tmpl.array_size = box->depth;
      break;
   default:
      assert(tmpl.array_size == 1);
      assert(box->depth == 1);
      break;
   }

   tmpl.last_level = 0;

   /* Linear is incompatible with depth/stencil, so we convert */
   tmpl.format = agx_staging_format(rsc->layout.format);
   tmpl.bind =
      PIPE_BIND_LINEAR | PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW;

   struct pipe_resource *pstaging = screen->resource_create(screen, &tmpl);
   if (!pstaging)
      return NULL;

   return agx_resource(pstaging);
}

static void
agx_blit_from_staging(struct pipe_context *pctx, struct agx_transfer *trans)
{
   struct pipe_resource *dst = trans->base.resource;
   struct pipe_blit_info blit = {0};

   blit.dst.resource = dst;
   blit.dst.format = agx_staging_format(agx_resource(dst)->layout.format);
   blit.dst.level = trans->base.level;
   blit.dst.box = trans->base.box;
   blit.src.resource = trans->staging.rsrc;
   blit.src.format = blit.dst.format;
   blit.src.level = 0;
   blit.src.box = trans->staging.box;
   blit.mask = util_format_get_mask(blit.src.format);
   blit.filter = PIPE_TEX_FILTER_NEAREST;

   agx_blit(pctx, &blit);
}

static void
agx_blit_to_staging(struct pipe_context *pctx, struct agx_transfer *trans)
{
   struct pipe_resource *src = trans->base.resource;
   struct pipe_blit_info blit = {0};

   blit.src.resource = src;
   blit.src.format = agx_staging_format(agx_resource(src)->layout.format);
   blit.src.level = trans->base.level;
   blit.src.box = trans->base.box;
   blit.dst.resource = trans->staging.rsrc;
   blit.dst.format = blit.src.format;
   blit.dst.level = 0;
   blit.dst.box = trans->staging.box;
   blit.mask = util_format_get_mask(blit.dst.format);
   blit.filter = PIPE_TEX_FILTER_NEAREST;

   agx_blit(pctx, &blit);
}

/*
 * glBufferSubData and glMapBufferRange(GL_MAP_INVALIDATE_RANGE_BIT) into a
 * buffer that unfinished batches use. Writing it on the CPU means waiting for
 * those batches, and buffers too large to shadow (a game's 128 MiB vertex
 * heap) have no other way out, so every such upload drained the GPU. Instead,
 * write a staging copy and copy it into place on the GPU at unmap, after the
 * batches already recorded and before any later one.
 */
static bool
agx_stage_buffer_write(struct agx_context *ctx, struct agx_resource *rsrc,
                       unsigned usage, const struct pipe_box *box)
{
   struct agx_device *dev = agx_device(ctx->base.screen);

   if (rsrc->base.target != PIPE_BUFFER ||
       (usage & (PIPE_MAP_WRITE | PIPE_MAP_DISCARD_RANGE)) !=
          (PIPE_MAP_WRITE | PIPE_MAP_DISCARD_RANGE) ||
       (usage & (PIPE_MAP_READ | PIPE_MAP_UNSYNCHRONIZED | PIPE_MAP_PERSISTENT |
                 PIPE_MAP_COHERENT | PIPE_MAP_DIRECTLY |
                 PIPE_MAP_DISCARD_WHOLE_RESOURCE)))
      return false;

   /* G17G copies buffers on the CPU (no precompiled kernels) */
   if (agx_chip_g17_m5(dev->chip) || (dev->debug & AGX_DBG_NOSHADOW))
      return false;

   if (rsrc->bo->flags & (AGX_BO_SHARED | AGX_BO_SHAREABLE))
      return false;

   /* A range nothing wrote yet can be written directly */
   if (!util_ranges_intersect(&rsrc->valid_buffer_range, box->x,
                              box->x + box->width))
      return false;

   /* A whole-buffer write can rename the storage instead */
   if (box->x == 0 && box->width == rsrc->base.width0 &&
       !(rsrc->base.flags & PIPE_RESOURCE_FLAG_MAP_PERSISTENT))
      return false;

   /* Where agx_prepare_for_map can shadow, keep doing so: a CPU copy of a
    * small buffer adds no GPU work, while a copy between two frames' passes
    * would stop them overlapping. Stage only what would otherwise wait.
    */
   if (rsrc->layout.size_B <= MAX_SHADOW_BYTES &&
       rsrc->shadowed_bytes < MAX_TOTAL_SHADOW_BYTES &&
       !(rsrc->base.flags & PIPE_RESOURCE_FLAG_MAP_PERSISTENT))
      return false;

   /* Batches stay listed until a later sync retires them, so this can be
    * stale. That costs a copy, while asking the kernel would cost a call on
    * every upload.
    */
   return agx_any_batch_uses_resource(ctx, rsrc);
}

static void *
agx_transfer_map(struct pipe_context *pctx, struct pipe_resource *resource,
                 unsigned level,
                 unsigned usage, /* a combination of PIPE_MAP_x */
                 const struct pipe_box *box,
                 struct pipe_transfer **out_transfer)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_resource *rsrc = agx_resource(resource);

   if (agx_stage_buffer_write(ctx, rsrc, usage, box)) {
      /* Match the destination's alignment within 16 bytes so the copy can
       * be vectorized.
       */
      unsigned misalign = box->x & 0xf;
      struct pipe_resource *staging = NULL;
      unsigned offset = 0;
      void *map = NULL;

      u_upload_alloc_ref(pctx->stream_uploader, 0, box->width + misalign, 16,
                         &offset, &staging, &map);
      if (staging) {
         util_range_add(resource, &rsrc->valid_buffer_range, box->x,
                        box->x + box->width);

         struct agx_transfer *transfer = CALLOC_STRUCT(agx_transfer);
         transfer->base.level = level;
         transfer->base.usage = usage;
         transfer->base.box = *box;
         pipe_resource_reference(&transfer->base.resource, resource);
         u_box_1d(offset + misalign, box->width, &transfer->staging.box);
         transfer->staging.rsrc = staging;
         *out_transfer = &transfer->base;

         return (uint8_t *)map + misalign;
      }
   }

   /* Can't map tiled/compressed directly */
   if ((usage & PIPE_MAP_DIRECTLY) && rsrc->modifier != DRM_FORMAT_MOD_LINEAR)
      return NULL;

   /* Can't transfer out of bounds mip levels */
   if (level >= rsrc->layout.levels)
      return NULL;

   /* For compression, we use a staging blit as we do not implement AGX
    * compression in software. In some cases, we could use this path for
    * twiddled too, but we don't have a use case for that yet.
    */
   bool staging_blit = ail_is_level_logically_compressed(&rsrc->layout, level);

   agx_prepare_for_map(ctx, rsrc, level, usage, box, staging_blit);

   struct agx_device *dev = agx_device(pctx->screen);
   const bool trace_read = agx_chip_hal300_path(dev->chip) &&
                          (dev->debug & AGX_DBG_FAULTTRACE) &&
                          (usage & PIPE_MAP_READ);
   if (trace_read) {
      fprintf(stderr, "M5_READBACK va=%#llx size=%llu format=%s modifier=%#llx valid=%u staging=%u box=%d,%d,%d,%d stride=%u\n",
              (unsigned long long)rsrc->bo->va->addr,
              (unsigned long long)rsrc->bo->size,
              util_format_name(rsrc->layout.format),
              (unsigned long long)rsrc->modifier,
              agx_resource_valid(rsrc, level), staging_blit,
              box->x, box->y, box->width, box->height,
              util_format_get_stride(rsrc->layout.format, box->width));
      const uint8_t *raw = agx_bo_map(rsrc->bo);
      if (raw) {
         size_t nonzero = 0;
         for (size_t i = 0; i < rsrc->bo->size; ++i)
            nonzero += raw[i] != 0;
         fprintf(stderr, "M5_READBACK_RAW nonzero=%zu head=", nonzero);
         for (unsigned i = 0; i < MIN2(rsrc->bo->size, 32); ++i)
            fprintf(stderr, "%02x", raw[i]);
         fputc('\n', stderr);
      }
   }

   /* Track the written buffer range */
   if (resource->target == PIPE_BUFFER) {
      /* Note the ordering: DISCARD|WRITE is valid, so clear before adding. */
      if (usage & PIPE_MAP_DISCARD_WHOLE_RESOURCE)
         util_range_set_empty(&rsrc->valid_buffer_range);
      if (usage & PIPE_MAP_WRITE) {
         util_range_add(resource, &rsrc->valid_buffer_range, box->x,
                        box->x + box->width);
      }
   }

   struct agx_transfer *transfer = CALLOC_STRUCT(agx_transfer);
   transfer->base.level = level;
   transfer->base.usage = usage;
   transfer->base.box = *box;

   pipe_resource_reference(&transfer->base.resource, resource);
   *out_transfer = &transfer->base;

   if (staging_blit) {
      /* Should never happen for buffers, and it's not safe */
      assert(resource->target != PIPE_BUFFER);

      struct agx_resource *staging =
         agx_alloc_staging(pctx->screen, rsrc, level, box);
      assert(staging);

      /* Staging resources have one LOD: level 0. Query the strides
       * on this LOD.
       */
      transfer->base.stride = ail_get_linear_stride_B(&staging->layout, 0);
      transfer->base.layer_stride = staging->layout.layer_stride_B;
      transfer->staging.rsrc = &staging->base;

      transfer->staging.box = *box;
      transfer->staging.box.x = 0;
      transfer->staging.box.y = 0;
      transfer->staging.box.z = 0;

      assert(transfer->staging.rsrc != NULL);

      if ((usage & PIPE_MAP_READ) && agx_resource_valid(rsrc, level)) {
         agx_blit_to_staging(pctx, transfer);
         agx_sync_writer(ctx, staging, "GPU read staging blit");
      }

      return agx_bo_map(staging->bo);
   }

   if (ail_is_level_twiddled_uncompressed(&rsrc->layout, level)) {
      /* Should never happen for buffers, and it's not safe */
      assert(resource->target != PIPE_BUFFER);

      transfer->base.stride =
         util_format_get_stride(rsrc->layout.format, box->width);

      transfer->base.layer_stride = util_format_get_2d_size(
         rsrc->layout.format, transfer->base.stride, box->height);

      transfer->map = calloc(transfer->base.layer_stride, box->depth);

      if ((usage & PIPE_MAP_READ) && agx_resource_valid(rsrc, level)) {
         for (unsigned z = 0; z < box->depth; ++z) {
            uint8_t *map = agx_map_texture_cpu(rsrc, level, box->z + z);
            uint8_t *dst =
               (uint8_t *)transfer->map + transfer->base.layer_stride * z;

            ail_detile(map, dst, &rsrc->layout, level, transfer->base.stride,
                       box->x, box->y, box->width, box->height);
            if (trace_read) {
               size_t nonzero = 0;
               for (size_t i = 0; i < transfer->base.layer_stride; ++i)
                  nonzero += dst[i] != 0;
               fprintf(stderr, "M5_READBACK_DETILED layer=%u nonzero=%zu head=",
                       z, nonzero);
               for (unsigned i = 0; i < MIN2(transfer->base.layer_stride, 32); ++i)
                  fprintf(stderr, "%02x", dst[i]);
               fputc('\n', stderr);
            }
         }
      }

      return transfer->map;
   } else {
      assert(rsrc->modifier == DRM_FORMAT_MOD_LINEAR);

      transfer->base.stride = ail_get_linear_stride_B(&rsrc->layout, level);
      transfer->base.layer_stride = rsrc->layout.layer_stride_B;

      /* Be conservative for direct writes */
      if ((usage & PIPE_MAP_WRITE) &&
          (usage &
           (PIPE_MAP_DIRECTLY | PIPE_MAP_PERSISTENT | PIPE_MAP_COHERENT))) {
         BITSET_SET(rsrc->data_valid, level);
      }

      uint32_t offset =
         ail_get_linear_pixel_B(&rsrc->layout, level, box->x, box->y, box->z);

      return ((uint8_t *)agx_bo_map(rsrc->bo)) + offset;
   }
}

/* texture_subdata into a twiddled level: tile straight from the caller's
 * data. The generic path maps through a calloc'd copy of the box, fills it
 * and tiles that on unmap -- for a compositor uploading a 4K SHM frame per
 * commit, the extra 33 MB copy and its page faults cost as much as the
 * tiling. Depth/stencil and MSAA keep the transfer helper's handling. */
static void
agx_texture_subdata(struct pipe_context *pctx, struct pipe_resource *prsrc,
                    unsigned level, unsigned usage, const struct pipe_box *box,
                    const void *data, unsigned stride, uintptr_t layer_stride)
{
   struct agx_resource *rsrc = agx_resource(prsrc);

   if (prsrc->target == PIPE_BUFFER || level >= rsrc->layout.levels ||
       prsrc->nr_samples > 1 ||
       util_format_is_depth_or_stencil(prsrc->format) ||
       !ail_is_level_twiddled_uncompressed(&rsrc->layout, level)) {
      u_default_texture_subdata(pctx, prsrc, level, usage, box, data, stride,
                                layer_stride);
      return;
   }

   usage |= PIPE_MAP_WRITE | PIPE_MAP_DISCARD_RANGE;
   agx_prepare_for_map(agx_context(pctx), rsrc, level, usage, box, false);

   for (unsigned z = 0; z < box->depth; ++z) {
      uint8_t *map = agx_map_texture_cpu(rsrc, level, box->z + z);

      ail_tile(map, (uint8_t *)data + layer_stride * z, &rsrc->layout, level,
               stride, box->x, box->y, box->width, box->height);
   }

   BITSET_SET(rsrc->data_valid, level);
}

static void
agx_transfer_unmap(struct pipe_context *pctx, struct pipe_transfer *transfer)
{
   /* Gallium expects writeback here, so we tile */

   struct agx_transfer *trans = agx_transfer(transfer);
   struct pipe_resource *prsrc = transfer->resource;
   struct agx_resource *rsrc = (struct agx_resource *)prsrc;

   if (trans->staging.rsrc && prsrc->target == PIPE_BUFFER) {
      /* A staged buffer write, see agx_stage_buffer_write */
      agx_copy_buffer_range(pctx, prsrc, transfer->box.x, trans->staging.rsrc,
                            trans->staging.box.x, transfer->box.width);
   } else if (trans->staging.rsrc && (transfer->usage & PIPE_MAP_WRITE)) {
      assert(prsrc->target != PIPE_BUFFER);
      agx_blit_from_staging(pctx, trans);
      agx_flush_readers(agx_context(pctx), agx_resource(trans->staging.rsrc),
                        "GPU write staging blit");
   } else if (trans->map && (transfer->usage & PIPE_MAP_WRITE)) {
      assert(
         ail_is_level_twiddled_uncompressed(&rsrc->layout, transfer->level));

      for (unsigned z = 0; z < transfer->box.depth; ++z) {
         uint8_t *map =
            agx_map_texture_cpu(rsrc, transfer->level, transfer->box.z + z);
         uint8_t *src = (uint8_t *)trans->map + transfer->layer_stride * z;

         ail_tile(map, src, &rsrc->layout, transfer->level, transfer->stride,
                  transfer->box.x, transfer->box.y, transfer->box.width,
                  transfer->box.height);
      }
   }

   /* The level we wrote is now initialized. We do this at the end so
    * blit_from_staging can avoid reloading existing contents.
    */
   if (transfer->usage & PIPE_MAP_WRITE)
      BITSET_SET(rsrc->data_valid, transfer->level);

   /* Free the transfer */
   free(trans->map);
   pipe_resource_reference(&trans->staging.rsrc, NULL);
   pipe_resource_reference(&transfer->resource, NULL);
   FREE(transfer);
}

/*
 * clear/copy
 */
static void
agx_clear(struct pipe_context *pctx, unsigned buffers,
          uint32_t color_clear_mask, uint8_t stencil_clear_mask,
          const struct pipe_scissor_state *scissor_state,
          const union pipe_color_union *color, double depth, unsigned stencil)
{
   struct agx_context *ctx = agx_context(pctx);

   /* G17G clears a layer at a time, as it draws (agx_g17g_bind_layer). */
   if (agx_chip_g17_m5(agx_device(pctx->screen)->chip) &&
       !ctx->g17g_layer_split &&
       util_framebuffer_get_num_layers(&ctx->framebuffer) > 1) {
      if (agx_render_condition_check(ctx))
         agx_g17g_layered_clear(ctx, buffers, color_clear_mask,
                                stencil_clear_mask, color, depth, stencil);
      return;
   }

   /* G17G clears a colour attachment at a time, as it draws. */
   if (agx_chip_g17_m5(agx_device(pctx->screen)->chip) &&
       ctx->framebuffer.nr_cbufs > 1) {
      if (agx_render_condition_check(ctx))
         agx_g17g_split_mrt_clear(ctx, buffers, color_clear_mask,
                                  stencil_clear_mask, color, depth, stencil);
      return;
   }

   struct agx_batch *batch = agx_get_batch(ctx);

   if (unlikely(!agx_render_condition_check(ctx)))
      return;

   unsigned fastclear = buffers & ~(batch->draw | batch->load);
   struct agx_device *dev = agx_device(pctx->screen);
   if (agx_chip_native_agx3(dev->chip)) {
      agx_batch_init_state(batch);
      const struct agx_tilebuffer_layout *tib = &batch->tilebuffer_layout;
      /* A spilling multisample tilebuffer clears every target with the
       * draw, as Honeykrisp does: a background clear of the resident targets
       * would leave the draw's epilog to preserve them by reading the tile
       * back, a multisample eMRT shape the packer refuses. On G17P such a
       * tilebuffer is always wider than the uncovered-tile limit, which
       * already sends it there. */
      const bool clear_all =
         tib->nr_samples > 1 &&
         (agx_tilebuffer_spills(tib) ||
          (agx_chip_uncovered_tile_ms_defect(dev->chip) &&
           tib->sample_size_B > AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE));
      for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
         if (tib->spilled[rt] || clear_all)
            fastclear &= ~(PIPE_CLEAR_COLOR0 << rt);
      }
   }
   unsigned slowclear = buffers & ~fastclear;

   assert(scissor_state == NULL &&
          "we don't support pipe_caps.clear_scissored");

   /* Fast clears configure the batch */
   for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
      if (!(fastclear & (PIPE_CLEAR_COLOR0 << rt)))
         continue;

      static_assert(sizeof(color->f) == 16, "mismatched structure");

      /* Clear colour must be clamped to properly handle signed ints. */
      union pipe_color_union clamped =
         util_clamp_color(batch->key.cbufs[rt].format, color);

      batch->clear_color[rt] = clamped;
      batch->uploaded_clear_color[rt] = agx_pool_upload_aligned(
         &batch->pool, clamped.f, sizeof(clamped.f), 16);
   }

   /* A batch resolving to a single-sample image reloads that image as its
    * background; a fast clear of the colour replaces it instead. Without
    * this, a clear of a multisampled window-system buffer was dropped and
    * the resolve wrote back the old image. */
   if (fastclear & PIPE_CLEAR_COLOR0)
      batch->g17p_resolve_reload = false;

   if (fastclear & PIPE_CLEAR_DEPTH)
      batch->clear_depth = depth;

   if (fastclear & PIPE_CLEAR_STENCIL)
      batch->clear_stencil = stencil;

   /* Slow clears draw a fullscreen rectangle */
   if (slowclear) {
      agx_blitter_save(ctx, ctx->blitter, ASAHI_CLEAR);
      util_blitter_clear(
         ctx->blitter, ctx->framebuffer.width, ctx->framebuffer.height,
         util_framebuffer_get_num_layers(&ctx->framebuffer), slowclear, color,
         depth, stencil,
         util_framebuffer_get_num_samples(&ctx->framebuffer) > 1);
   }

   if (fastclear)
      agx_batch_init_state(batch);

   batch->clear |= fastclear;
   batch->resolve |= buffers;
   assert((batch->draw & slowclear) == slowclear);
}

static void
transition_resource(struct pipe_context *pctx, struct agx_resource *rsrc,
                    struct pipe_resource *templ)
{
   struct agx_resource *new_res =
      agx_resource(pctx->screen->resource_create(pctx->screen, templ));

   assert(new_res);
   assert(!(rsrc->base.bind & PIPE_BIND_SHARED) && "cannot swap BOs if shared");

   int level;
   BITSET_FOREACH_SET(level, rsrc->data_valid, PIPE_MAX_TEXTURE_LEVELS) {
      /* Copy each valid level */
      struct pipe_box box;
      u_box_3d(0, 0, 0, u_minify(rsrc->layout.width_px, level),
               u_minify(rsrc->layout.height_px, level),
               util_num_layers(&rsrc->base, level), &box);

      agx_resource_copy_region(pctx, &new_res->base, level, 0, 0, 0,
                               &rsrc->base, level, &box);
   }

   /* Flush the blits out, to make sure the old resource is no longer used */
   agx_flush_writer(agx_context(pctx), new_res, "flush_resource");

   /* Copy the bind flags and swap the BOs */
   struct agx_bo *old = rsrc->bo;
   rsrc->base.bind = new_res->base.bind;
   rsrc->layout = new_res->layout;
   rsrc->modifier = new_res->modifier;
   rsrc->bo = new_res->bo;
   new_res->bo = old;

   /* Free the new resource, which now owns the old BO */
   pipe_resource_reference((struct pipe_resource **)&new_res, NULL);
}

void
agx_decompress(struct agx_context *ctx, struct agx_resource *rsrc,
               const char *reason)
{
   if (rsrc->layout.compressed) {
      perf_debug_ctx(ctx, "Decompressing resource due to %s", reason);
   } else if (!rsrc->layout.writeable_image) {
      perf_debug_ctx(ctx, "Reallocating image due to %s", reason);
   }

   struct pipe_resource templ = rsrc->base;
   assert(!(templ.bind & PIPE_BIND_SHADER_IMAGE) && "currently compressed");
   templ.bind |= PIPE_BIND_SHADER_IMAGE /* forces off compression */;
   transition_resource(&ctx->base, rsrc, &templ);
}

static void
agx_flush_resource(struct pipe_context *pctx, struct pipe_resource *pres)
{
   struct agx_resource *rsrc = agx_resource(pres);

   /* flush_resource is used to prepare resources for sharing, so if this is not
    * already a shareabe resource, make it so
    */
   struct agx_bo *old = rsrc->bo;
   if (!(old->flags & AGX_BO_SHAREABLE)) {
      assert(rsrc->layout.levels == 1 &&
             "Shared resources must not be mipmapped");
      assert(rsrc->layout.sample_count_sa == 1 &&
             "Shared resources must not be multisampled");
      assert(rsrc->bo);
      assert(!(pres->bind & PIPE_BIND_SHARED));

      struct pipe_resource templ = *pres;
      templ.bind |= PIPE_BIND_SHARED;
      transition_resource(pctx, rsrc, &templ);
   } else {
      /* Otherwise just claim it's already shared */
      pres->bind |= PIPE_BIND_SHARED;
      agx_flush_writer(agx_context(pctx), rsrc, "flush_resource");
   }
}

static unsigned
build_timestamp_offset(struct agx_batch *batch, unsigned offset)
{
   return (agx_batch_idx(batch) * sizeof(struct agx_timestamps)) + offset;
}

#define timestamp_offset(batch, offs)                                          \
   build_timestamp_offset(batch, offsetof(struct agx_timestamps, offs))

/* Native G18 ZLS arrays use either 16-KiB pages (bit 0) or 128-byte
 * cachelines. The upper field stores units minus one. In particular, a
 * 128-byte layer is encoded as zero; it is not a missing array stride.
 * Witness: layered-zls native682, including sub-page Z32 and S8 arrays. */
static uint32_t
agx_zls_layer_stride(struct agx_device *dev, const struct ail_layout *layout,
                     bool single_layer)
{
   if (dev->chip == AGX_CHIP_G18G && layout->policy == AIL_LAYOUT_POLICY_M5) {
      if (single_layer)
         return 0;

      uint64_t stride = layout->layer_stride_B;
      bool pages = util_is_aligned(stride, AIL_PAGESIZE);
      unsigned unit = pages ? AIL_PAGESIZE : AIL_CACHELINE;
      assert(stride && util_is_aligned(stride, unit));
      assert((stride / unit - 1) <= (UINT32_MAX >> 14));
      return ((stride / unit - 1) << 14) | pages;
   }

   assert((layout->depth_px == 1 ||
           util_is_aligned(layout->layer_stride_B, AIL_PAGESIZE)) &&
          "Page aligned Z/S layers");
   unsigned pages = layout->layer_stride_B / AIL_PAGESIZE;
   return ((pages - 1) << 14) | 1;
}

static void
agx_cmdbuf(struct agx_device *dev, struct drm_asahi_cmd_render *c,
           struct agx_pool *pool, struct agx_batch *batch,
           struct pipe_framebuffer_state *framebuffer, uint64_t encoder_ptr,
           uint64_t scissor_ptr, uint64_t depth_bias_ptr,
           uint64_t visibility_result_ptr, struct asahi_bg_eot pipeline_clear,
           struct asahi_bg_eot pipeline_load,
           struct asahi_bg_eot pipeline_store, bool clear_pipeline_textures,
           double clear_depth, unsigned clear_stencil,
           struct agx_tilebuffer_layout *tib)
{
   memset(c, 0, sizeof(*c));

   c->vdm_ctrl_stream_base = encoder_ptr;

   /* bit 0 specifies OpenGL clip behaviour. Since ARB_clip_control is
    * advertised, we don't set it and lower in the vertex shader.
    */
   c->ppp_ctrl = 0x202;

   c->width_px = framebuffer->width;
   c->height_px = framebuffer->height;

   c->isp_bgobjvals = 0x300;

   struct agx_resource *zres = NULL, *sres = NULL;
   struct pipe_surface *zsbuf = &framebuffer->zsbuf;

   if (framebuffer->zsbuf.texture) {
      agx_pack(&c->isp_zls_pixels, CR_ISP_ZLS_PIXELS, cfg) {
         cfg.x = c->width_px;
         cfg.y = c->height_px;
      }
   }

   if (zsbuf->texture) {
      struct agx_resource *zsres = agx_resource(zsbuf->texture);
      const struct util_format_description *desc =
         util_format_description(zsres->layout.format);

      assert(desc->format == PIPE_FORMAT_Z32_FLOAT ||
             desc->format == PIPE_FORMAT_Z16_UNORM ||
             desc->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT ||
             desc->format == PIPE_FORMAT_S8_UINT);

      if (util_format_has_depth(desc))
         zres = zsres;
      else
         sres = zsres;

      if (zsres->separate_stencil)
         sres = zsres->separate_stencil;

      unsigned level = zsbuf->level;
      unsigned first_layer = zsbuf->first_layer;
      const bool single_layer =
         MAX2(util_framebuffer_get_num_layers(framebuffer), 1) == 1;

      if (zres) {
         c->depth.base = agx_map_texture_gpu(zres, first_layer) +
                         ail_get_level_offset_B(&zres->layout, level);

         c->depth.stride = agx_zls_layer_stride(dev, &zres->layout, single_layer);

         if (zres->layout.compressed) {
            c->depth.comp_base =
               agx_map_gpu(zres) + zres->layout.metadata_offset_B +
               (first_layer * zres->layout.compression_layer_stride_B) +
               zres->layout.level_offsets_compressed_B[level];

            /* Meta stride in cache lines */
            assert(util_is_aligned(zres->layout.compression_layer_stride_B,
                                   AIL_CACHELINE) &&
                   "Cacheline aligned Z meta layers");
            unsigned stride_lines =
               zres->layout.compression_layer_stride_B / AIL_CACHELINE;
            c->depth.comp_stride = (stride_lines - 1) << 14;
         }

         if (zres->base.format == PIPE_FORMAT_Z16_UNORM) {
            const float scale = 0xffff;
            c->isp_bgobjdepth =
               (uint16_t)(SATURATE(clear_depth) * scale + 0.5f);

            c->flags |= DRM_ASAHI_RENDER_DBIAS_IS_INT;
         } else {
            c->isp_bgobjdepth = fui(clear_depth);
         }
      }

      if (sres) {
         c->stencil.base = agx_map_texture_gpu(sres, first_layer) +
                           ail_get_level_offset_B(&sres->layout, level);

         c->stencil.stride = agx_zls_layer_stride(dev, &sres->layout, single_layer);

         if (sres->layout.compressed) {
            c->stencil.comp_base =
               agx_map_gpu(sres) + sres->layout.metadata_offset_B +
               (first_layer * sres->layout.compression_layer_stride_B) +
               sres->layout.level_offsets_compressed_B[level];

            /* Meta stride in cache lines */
            assert(util_is_aligned(sres->layout.compression_layer_stride_B,
                                   AIL_CACHELINE) &&
                   "Cacheline aligned S meta layers");
            unsigned stride_lines =
               sres->layout.compression_layer_stride_B / AIL_CACHELINE;
            c->stencil.comp_stride = (stride_lines - 1) << 14;
         }

         c->isp_bgobjvals |= clear_stencil;
      }
   }

   unsigned load = batch->load & ~batch->clear;

   struct agx_zls zls = {
      .z_store = batch->resolve & PIPE_CLEAR_DEPTH,
      .s_store = batch->resolve & PIPE_CLEAR_STENCIL,
      .z_load = load & PIPE_CLEAR_DEPTH,
      .s_load = load & PIPE_CLEAR_STENCIL,
   };

   agx_pack_zls_control((struct agx_zls_control_packed *)&c->zls_ctrl,
                        zres ? &zres->layout : NULL,
                        sres ? &sres->layout : NULL, &zls);

   /* G17P treats an absent ZLS allocation as an inactive depth surface.  Its
    * native render command keeps the clear register at 1.0 in that state;
    * zero is only a real clear value when depth or stencil storage exists.
    * Honeykrisp submits M5 renders on the G17 runtime the same way
    * (hk_drm_g17p_depth_clear). */
   if ((agx_chip_native_or_g18g(dev->chip) ||
        agx_m5_render_resources(dev)) && !zres &&
       !sres)
      c->isp_bgobjdepth = fui(1.0f);

   /* Neither AGX3 generation publishes the ISP's depth/stencil dimensions the
    * way AGX2 does, and the two do not agree with each other either, so take
    * the packing from the shared witness rather than the AGX2 struct. The
    * Gallium path had been publishing G17P's 15-bit pair on G17G, which the
    * kernel's M5 depth admission refused as soon as a render actually carried
    * a depth attachment. */
   if (dev->chip == AGX_CHIP_G18G || dev->chip == AGX_CHIP_G17P || agx_chip_g17_m5(dev->chip)) {
      c->isp_zls_pixels =
         agx_zls_pixels(dev->chip, c->width_px, c->height_px, c->depth.base,
                        c->depth.comp_base, c->stencil.base,
                        c->stencil.comp_base);
   }

   if ((dev->debug & AGX_DBG_NOCLUSTER) && dev->chip != AGX_CHIP_G17P)
      c->flags |= DRM_ASAHI_RENDER_NO_VERTEX_CLUSTERING;

   c->bg.rsrc_spec = (uint32_t)pipeline_clear.rsrc_spec;
   c->eot.rsrc_spec = (uint32_t)pipeline_store.rsrc_spec;
   c->partial_bg.rsrc_spec = (uint32_t)pipeline_load.rsrc_spec;
   c->partial_eot.rsrc_spec = (uint32_t)pipeline_store.rsrc_spec;
   c->bg_rsrc_spec_hi = (uint32_t)(pipeline_clear.rsrc_spec >> 32);
   c->bg_eot_rsrc_spec_hi = (uint32_t)(pipeline_store.rsrc_spec >> 32);
   c->bg_partial_rsrc_spec_hi = (uint32_t)(pipeline_load.rsrc_spec >> 32);
   c->bg_eot_partial_rsrc_spec_hi =
      (uint32_t)(pipeline_store.rsrc_spec >> 32);
   if (c->bg_rsrc_spec_hi || c->bg_eot_rsrc_spec_hi ||
       c->bg_partial_rsrc_spec_hi || c->bg_eot_partial_rsrc_spec_hi)
      c->flags |= DRM_ASAHI_RENDER_RSRC_SPEC_HI;

   /* XXX is this correct? */
   c->bg.usc = pipeline_clear.usc | (framebuffer->nr_cbufs >= 4 ? 8 : 4);
   c->eot.usc = pipeline_store.usc | 4;
   c->partial_bg.usc = pipeline_load.usc | 4;
   c->partial_eot.usc = pipeline_store.usc | 4;

   agx_tilebuffer_set_drm_cmd(c, tib);

   /* The depth/stencil half of the render command is the part the M5 kernel
    * validates and the part no CTS failure can otherwise be read from. */
   if (dev->debug & AGX_DBG_FAULTTRACE) {
      fprintf(stderr,
              "M5_RENDER %ux%u cbufs=%u zs=%s zls=%#llx pixels=%#x "
              "clear=%#x depth=%#llx/%#llx stencil=%#llx/%#llx "
              "load=%#x clear_mask=%#x resolve=%#x\n",
              c->width_px, c->height_px, framebuffer->nr_cbufs,
              util_format_name(framebuffer->zsbuf.format),
              (unsigned long long)c->zls_ctrl, c->isp_zls_pixels,
              c->isp_bgobjdepth, (unsigned long long)c->depth.base,
              (unsigned long long)c->depth.comp_base,
              (unsigned long long)c->stencil.base,
              (unsigned long long)c->stencil.comp_base, batch->load,
              batch->clear, batch->resolve);
   }

   c->samples = tib->nr_samples;
   c->layers = MAX2(util_framebuffer_get_num_layers(framebuffer), 1);

   c->ppp_multisamplectl = batch->uniforms.ppp_multisamplectl;
   c->sample_size_B = tib->sample_size_B;

   float tan_60 = 1.732051f;
   c->isp_merge_upper_x = fui(tan_60 / framebuffer->width);
   c->isp_merge_upper_y = fui(tan_60 / framebuffer->height);

   c->isp_scissor_base = scissor_ptr;
   c->isp_dbias_base = depth_bias_ptr;
   c->isp_oclqry_base = visibility_result_ptr;

   if (batch->sampler_heap.bo &&
       !agx_stream_sampler_state(dev)) {
      c->sampler_heap = batch->sampler_heap.bo->va->addr;
      c->sampler_count = batch->sampler_heap.count;
   }

   /* If a tile is empty, we do not want to process it, as the redundant
    * roundtrip of memory-->tilebuffer-->memory wastes a tremendous amount of
    * memory bandwidth. Any draw marks a tile as non-empty, so we only need to
    * process empty tiles if the background+EOT programs have a side effect.
    * This is the case exactly when there is an attachment we are clearing (some
    * attachment A in clear and in resolve <==> non-empty intersection).
    *
    * This case matters a LOT for performance in workloads that split batches.
    */
   /* A direct resolve also writes a different image: loaded samples in
    * tiles without geometry must reach that destination. */
   if ((batch->clear & batch->resolve) ||
       (agx_chip_native_agx3(dev->chip) && batch->key.resolve &&
        (batch->load & PIPE_CLEAR_COLOR0)))
      c->flags |= DRM_ASAHI_RENDER_PROCESS_EMPTY_TILES;

   if (batch->vs_scratch) {
      c->flags |= DRM_ASAHI_RENDER_VERTEX_SCRATCH;
      c->vertex_helper.data = batch->ctx->scratch_vs.buf->va->addr;
      c->vertex_helper.cfg = batch->vs_preamble_scratch << 16;
      c->vertex_helper.binary = agx_helper_program(&batch->ctx->bg_eot);
   }
   if (batch->fs_scratch) {
      c->fragment_helper.data = batch->ctx->scratch_fs.buf->va->addr;
      c->fragment_helper.cfg = batch->fs_preamble_scratch << 16;
      c->fragment_helper.binary = agx_helper_program(&batch->ctx->bg_eot);
   }

   /* ASAHI_MESA_DEBUG=stats times every render (agx_batch_print_stats()). */
   if (batch->timestamps.size > 0 ||
       ((agx_device(batch->ctx->base.screen)->debug & AGX_DBG_STATS) &&
        batch->ctx->timestamps)) {
      c->ts_vtx.start.handle = batch->ctx->timestamp_handle;
      c->ts_frag.end.handle = batch->ctx->timestamp_handle;
      c->ts_vtx.start.offset = timestamp_offset(batch, vtx_start);
      c->ts_frag.end.offset = timestamp_offset(batch, frag_end);
   }
}

/*
 * context
 */
static void
agx_flush(struct pipe_context *pctx, struct pipe_fence_handle **fence,
          unsigned flags)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_screen *screen = agx_screen(ctx->base.screen);

   agx_flush_all(ctx, "Gallium flush");

   if ((flags & PIPE_FLUSH_END_OF_FRAME) &&
       unlikely(screen->dev.debug & AGX_DBG_PERF)) {
      perf_debug_ctx(ctx, "[Queue %u Frame %" PRIu64 "] draws=%" PRIu64
                          " vs_emits=%" PRIu64 " query_waits=%" PRIu64,
                     ctx->queue_id, ctx->perf.frame++, ctx->perf.draws,
                     ctx->perf.vs_emits, ctx->perf.query_waits);
      ctx->perf.draws = 0;
      ctx->perf.vs_emits = 0;
      ctx->perf.query_waits = 0;
   }

   if (!(flags & (PIPE_FLUSH_DEFERRED | PIPE_FLUSH_ASYNC)) &&
       ctx->flush_last_seqid) {
      /* Ensure other contexts in this screen serialize against the last
       * submission (and all prior submissions).
       */
      simple_mtx_lock(&screen->flush_seqid_lock);

      uint64_t val = p_atomic_read(&screen->flush_wait_seqid);
      if (val < ctx->flush_last_seqid)
         p_atomic_set(&screen->flush_wait_seqid, ctx->flush_last_seqid);

      /* Note: it's possible for the max() logic above to be "wrong" due
       * to a race in agx_batch_submit causing out-of-order timeline point
       * updates, making the larger value not actually a later submission.
       * However, see the comment in agx_batch.c for why this doesn't matter
       * because this corner case is handled conservatively in the kernel.
       */

      simple_mtx_unlock(&screen->flush_seqid_lock);

      /* Optimization: Avoid serializing against our own queue by
       * recording the last seen foreign seqid when flushing, and our own
       * flush seqid. If we then try to sync against our own seqid, we'll
       * instead sync against the last possible foreign one. This is *not*
       * the `val` we got above, because another context might flush with a
       * seqid between `val` and `flush_last_seqid` (which would not update
       * `flush_wait_seqid` per the logic above). This is somewhat
       * conservative: it means that if *any* foreign context flushes, then
       * on next flush of this context we will start waiting for *all*
       * prior submits on *all* contexts (even if unflushed) at that point,
       * including any local submissions prior to the latest one. That's
       * probably fine, it creates a one-time "wait for the second-previous
       * batch" wait on this queue but that still allows for at least
       * the previous batch to pipeline on the GPU and it's one-time
       * until another foreign flush happens. Phew.
       */
      if (val && val != ctx->flush_my_seqid)
         ctx->flush_other_seqid = ctx->flush_last_seqid - 1;

      ctx->flush_my_seqid = ctx->flush_last_seqid;
   }

   /* At this point all pending work has been submitted. Since jobs are
    * started and completed sequentially from a UAPI perspective, and since
    * we submit all jobs with compute+render barriers on the prior job,
    * waiting on the last submitted job is sufficient to guarantee completion
    * of all GPU work thus far, so we can create a fence out of the latest
    * syncobj.
    *
    * See this page for more info on how the GPU/UAPI queueing works:
    * https://github.com/AsahiLinux/docs/wiki/SW:AGX-driver-notes#queues
    */

   if (fence) {
      struct pipe_fence_handle *f = agx_fence_create(ctx);
      pctx->screen->fence_reference(pctx->screen, fence, NULL);
      *fence = f;
   }

   if (unlikely(screen->dev.debug & AGX_DBG_PERF)) {
      perf_debug_ctx(ctx, "Gallium flush: batches=%" PRIu64
                     " mixed=%" PRIu64 " submits=%" PRIu64
                     " submit_failures=%" PRIu64 "\n",
                     ctx->submit_perf.batches, ctx->submit_perf.mixed_batches,
                     ctx->submit_perf.submits, ctx->submit_perf.failures);
      memset(&ctx->submit_perf, 0, sizeof(ctx->submit_perf));
   }
}

static void
agx_flush_compute(struct agx_context *ctx, struct agx_batch *batch,
                  struct drm_asahi_cmd_compute *cmdbuf)
{
   /* Finalize the encoder */
   struct agx_device *dev = agx_device(ctx->base.screen);
   if (agx_chip_native_agx3(dev->chip)) {
      uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
      if (!agx3_cdm_stream_terminate_pack(dev->chip, terminate))
         UNREACHABLE("G17P CDM terminate must be grounded");
      memcpy(batch->cdm.current, terminate, sizeof(terminate));
      batch->cdm.current += sizeof(terminate);
   } else {
      agx_pack(batch->cdm.current, CDM_STREAM_TERMINATE, _)
         ;
   }

   agx_batch_add_bo(batch, batch->cdm.bo);

   if (batch->cs_scratch)
      agx_batch_add_bo(batch, ctx->scratch_cs.buf);

   *cmdbuf = (struct drm_asahi_cmd_compute){
      .cdm_ctrl_stream_base = batch->cdm.bo->va->addr,
      .cdm_ctrl_stream_end =
         batch->cdm.bo->va->addr +
         (batch->cdm.current - (uint8_t *)agx_bo_map(batch->cdm.bo)),
      .sampler_heap =
         batch->sampler_heap.bo ? batch->sampler_heap.bo->va->addr : 0,
      .sampler_count = batch->sampler_heap.count,
   };

   if (agx_stream_sampler_state(dev)) {
      cmdbuf->sampler_heap = 0;
      cmdbuf->sampler_count = 0;
   }

   if (batch->cs_scratch) {
      // The commented out lines *may* be related to subgroup-level preemption,
      // which we can't support without implementing threadgroup memory in the
      // helper. Disable them for now.

      // cmdbuf->iogpu_unk_40 = 0x1c;
      cmdbuf->helper.data = ctx->scratch_cs.buf->va->addr;
      cmdbuf->helper.cfg = batch->cs_preamble_scratch << 16;
      // cmdbuf->helper.cfg |= 0x40;
      cmdbuf->helper.binary = agx_helper_program(&batch->ctx->bg_eot);
   }

   const bool stats = (agx_device(ctx->base.screen)->debug & AGX_DBG_STATS) &&
                      ctx->timestamps;
   if (batch->timestamps.size > 0 || stats) {
      cmdbuf->ts.start.handle = ctx->timestamp_handle;
      cmdbuf->ts.end.handle = ctx->timestamp_handle;
      cmdbuf->ts.start.offset = timestamp_offset(batch, comp_start);
      cmdbuf->ts.end.offset = stats ? timestamp_offset(batch, comp_end)
                                    : timestamp_offset(batch, comp_start);
   }
}

static void
agx_flush_render(struct agx_context *ctx, struct agx_batch *batch,
                 struct drm_asahi_cmd_render *cmdbuf)
{
   struct agx_device *dev = agx_device(ctx->base.screen);

   if (batch->vs_scratch)
      agx_batch_add_bo(batch, ctx->scratch_vs.buf);
   if (batch->fs_scratch)
      agx_batch_add_bo(batch, ctx->scratch_fs.buf);

   assert(batch->initialized);

   /* Finalize the encoder */
   uint8_t stop[5 + 64] = {0x00, 0x00, 0x00, 0xc0, 0x00};
   memcpy(batch->vdm.current, stop, sizeof(stop));

   struct asahi_bg_eot pipeline_background =
      agx_build_bg_eot(batch, false, false);

   struct asahi_bg_eot pipeline_background_partial =
      agx_build_bg_eot(batch, false, true);

   struct asahi_bg_eot pipeline_store = agx_build_bg_eot(batch, true, false);

   /* The render submits the eot program for both its final and its partial
    * store (agx_cmdbuf), so these three are every tile program it runs. */
   if (agx_m5_render_resources(dev)) {
      const struct asahi_bg_eot *const tile_programs[] = {
         &pipeline_background, &pipeline_background_partial, &pipeline_store,
      };
      agx_batch_set_m5_tile_program(batch, tile_programs,
                                    ARRAY_SIZE(tile_programs));
   }

   bool clear_pipeline_textures =
      agx_tilebuffer_spills(&batch->tilebuffer_layout);

   for (unsigned i = 0; i < batch->key.nr_cbufs; ++i) {
      const struct pipe_surface *surf = &batch->key.cbufs[i];

      clear_pipeline_textures |=
         surf->texture && !(batch->clear & (PIPE_CLEAR_COLOR0 << i));
   }

   /* Scissor and depth bias arrays are staged to dynamic arrays on the CPU. At
    * submit time, they're done growing and are uploaded to GPU memory attached
    * to the batch.
    */
   uint64_t scissor = agx_pool_upload_aligned(&batch->pool, batch->scissor.data,
                                              batch->scissor.size, 64);
   uint64_t zbias = agx_pool_upload_aligned(
      &batch->pool, batch->depth_bias.data, batch->depth_bias.size, 64);

   /* BO list for a given batch consists of:
    *  - BOs for the batch's pools
    *  - BOs for the encoder
    *  - BO for internal shaders
    *  - BOs added to the batch explicitly
    */
   agx_batch_add_bo(batch, batch->vdm.bo);

   agx_cmdbuf(
      dev, cmdbuf, &batch->pool, batch, &batch->key,
      batch->vdm.g17p_vdm_alias ? batch->vdm.g17p_vdm_alias
                                : batch->vdm.bo->va->addr,
      scissor, zbias, agx_get_occlusion_heap(batch), pipeline_background,
      pipeline_background_partial, pipeline_store, clear_pipeline_textures,
      batch->clear_depth, batch->clear_stencil, &batch->tilebuffer_layout);
}

void
agx_flush_batch(struct agx_context *ctx, struct agx_batch *batch)
{
   assert(agx_batch_is_active(batch));
   assert(!agx_batch_is_submitted(batch));

   struct drm_asahi_cmd_render render;
   struct drm_asahi_cmd_compute compute;
   bool has_vdm = false, has_cdm = false;

   /* After a GPU recovery failed this context's VM, every submission fails
    * and the reset is already reported through robustness. Drop the batch
    * instead of encoding it: a batch begun after the loss can be half built
    * (Zen reached agx_build_bg_eot with a 0-sample, 0 px tile layout, which
    * the native descriptor check aborts on). Retire it as a failed submit,
    * so the BOs it writes are released from the writer table.
    */
   if (unlikely(ctx->any_faults)) {
      agx_batch_retire_faulted(ctx, batch);
      return;
   }

   if (batch->cdm.bo) {
      agx_flush_compute(ctx, batch, &compute);
      has_cdm = true;
   }

   if (batch->vdm.bo && (batch->clear || batch->initialized)) {
      agx_flush_render(ctx, batch, &render);
      has_vdm = true;
   }

   if (!has_cdm && !has_vdm) {
      agx_batch_reset(ctx, batch);
      return;
   }

   agx_batch_submit(ctx, batch, has_cdm ? &compute : NULL,
                    has_vdm ? &render : NULL);
}

static void
agx_destroy_context(struct pipe_context *pctx)
{
   struct agx_device *dev = agx_device(pctx->screen);
   struct agx_context *ctx = agx_context(pctx);
   struct agx_screen *screen = agx_screen(pctx->screen);

   /* Batch state needs to be freed on completion, and we don't want to yank
    * buffers out from in-progress GPU jobs to avoid faults, so just wait until
    * everything in progress is actually done on context destroy. This will
    * ensure everything is cleaned up properly.
    */
   agx_sync_all(ctx, "destroy context");
   agx_xfb_cpu_destroy(ctx);
   agx_swtnl_destroy(ctx);

   if (ctx->submit_quarantined)
      return;

   if (pctx->stream_uploader)
      u_upload_destroy(pctx->stream_uploader);

   if (ctx->blitter)
      util_blitter_destroy(ctx->blitter);

   util_unreference_framebuffer_state(&ctx->framebuffer);
   util_unreference_framebuffer_state(&ctx->zs_merge_real);

   agx_bg_eot_cleanup(&ctx->bg_eot);
   agx_destroy_meta_shaders(ctx);

   /* Lock around the syncobj destruction, to avoid racing
    * command submission in another context.
    **/
   u_rwlock_wrlock(&screen->destroy_lock);

   drmSyncobjDestroy(dev->fd, ctx->in_sync_obj);
   drmSyncobjDestroy(dev->fd, ctx->dummy_syncobj);
   if (ctx->in_sync_fd != -1)
      close(ctx->in_sync_fd);

   for (unsigned i = 0; i < AGX_MAX_BATCHES; ++i) {
      if (ctx->batches.slots[i].syncobj)
         drmSyncobjDestroy(dev->fd, ctx->batches.slots[i].syncobj);
   }

   if (ctx->timestamps) {
      dev->ops.bo_unbind_object(dev, ctx->timestamp_handle);
      agx_bo_unreference(dev, ctx->timestamps);
   }

   u_rwlock_wrunlock(&screen->destroy_lock);

   pipe_resource_reference(&ctx->heap, NULL);

   agx_scratch_fini(&ctx->scratch_vs);
   agx_scratch_fini(&ctx->scratch_fs);
   agx_scratch_fini(&ctx->scratch_cs);

   agx_destroy_command_queue(dev, ctx->queue_id);

   ralloc_free(ctx);
}

static void
agx_invalidate_resource(struct pipe_context *pctx,
                        struct pipe_resource *resource)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_batch *batch = agx_get_batch(ctx);

   /* Handle the glInvalidateFramebuffer case */
   if (batch->key.zsbuf.texture == resource)
      batch->resolve &= ~PIPE_CLEAR_DEPTHSTENCIL;

   for (unsigned i = 0; i < batch->key.nr_cbufs; ++i) {
      const struct pipe_surface *surf = &batch->key.cbufs[i];

      if (surf->texture == resource)
         batch->resolve &= ~(PIPE_CLEAR_COLOR0 << i);
   }
}

static enum pipe_reset_status
asahi_get_device_reset_status(struct pipe_context *pipe)
{
   struct agx_context *ctx = agx_context(pipe);

   return ctx->any_faults ? PIPE_GUILTY_CONTEXT_RESET : PIPE_NO_RESET;
}

static void
asahi_set_device_reset_callback(struct pipe_context *pipe,
                                const struct pipe_device_reset_callback *cb)
{
   struct agx_context *ctx = agx_context(pipe);

   if (cb)
      ctx->reset = *cb;
   else
      memset(&ctx->reset, 0, sizeof(ctx->reset));
}

static void
asahi_clear_buffer(struct pipe_context *pipe, struct pipe_resource *resource,
                   unsigned offset, unsigned size, const void *clear_value,
                   int clear_value_size)
{
   assert(clear_value_size > 0);
   if (clear_value_size <= 16 && util_is_power_of_two_nonzero(clear_value_size)) {
      union pipe_color_union color;
      bool aligned_16 = util_is_aligned(offset, 16) && util_is_aligned(size, 16);
      bool aligned_4 = util_is_aligned(offset, 4) && util_is_aligned(size, 4);

      /* Splat out to 128-bit */
      uint8_t *bytes = (uint8_t *)color.ui;
      memcpy(bytes, clear_value, clear_value_size);
      for (unsigned i = clear_value_size; i < 16; ++i) {
         bytes[i] = bytes[i % clear_value_size];
      }

      if (aligned_16) {
         struct agx_batch *batch = agx_get_compute_batch(agx_context(pipe));
         agx_batch_init_state(batch);
         agx_batch_writes_range(batch, agx_resource(resource), offset, size);
         libagx_fill_uint4(batch, agx_2d(size / 16, 1), AGX_BARRIER_ALL,
                           agx_map_gpu(agx_resource(resource)) + offset, 16,
                           color.ui[0], color.ui[1], color.ui[2], color.ui[3]);
         return;
      } else if (aligned_4 && clear_value_size <= 4) {
         struct agx_batch *batch = agx_get_compute_batch(agx_context(pipe));
         agx_batch_init_state(batch);
         agx_batch_writes_range(batch, agx_resource(resource), offset, size);
         libagx_fill(batch, agx_1d(size / 4), AGX_BARRIER_ALL,
                     agx_map_gpu(agx_resource(resource)) + offset, color.ui[0]);
         return;
      }
   }

   u_default_clear_buffer(pipe, resource, offset, size, clear_value,
                          clear_value_size);
}

static struct pipe_context *
agx_create_context(struct pipe_screen *screen, void *priv, unsigned flags)
{
   struct agx_context *ctx = rzalloc(NULL, struct agx_context);
   int ret;

   if (!ctx)
      return NULL;

   struct pipe_context *pctx = &ctx->base;

   pctx->screen = screen;
   pctx->priv = priv;

   util_dynarray_init(&ctx->writer, ctx);
   util_dynarray_init(&ctx->global_buffers, ctx);

   pctx->stream_uploader = u_upload_create_default(pctx);
   if (!pctx->stream_uploader) {
      ralloc_free(ctx);
      return NULL;
   }
   pctx->const_uploader = pctx->stream_uploader;

   enum drm_asahi_priority priority = DRM_ASAHI_PRIORITY_MEDIUM;

   if (flags & PIPE_CONTEXT_LOW_PRIORITY)
      priority = DRM_ASAHI_PRIORITY_LOW;
   else if (flags & (PIPE_CONTEXT_HIGH_PRIORITY | PIPE_CONTEXT_REALTIME_PRIORITY))
      priority = DRM_ASAHI_PRIORITY_HIGH;

   ctx->queue_id = agx_create_command_queue(agx_device(screen), priority);

   /* HIGH requires backend support and CAP_SYS_NICE or DRM master;
    * otherwise fall back to MEDIUM.
    */
   if (ctx->queue_id == UINT32_MAX && priority == DRM_ASAHI_PRIORITY_HIGH) {
      priority = DRM_ASAHI_PRIORITY_MEDIUM;
      ctx->queue_id = agx_create_command_queue(agx_device(screen), priority);
   }
   if (ctx->queue_id == UINT32_MAX) {
      u_upload_destroy(pctx->stream_uploader);
      ralloc_free(ctx);
      return NULL;
   }
   if (agx_device(screen)->debug & AGX_DBG_1QUEUE)
      priority = agx_device(screen)->queue_priority;

   pctx->context_priority = priority == DRM_ASAHI_PRIORITY_LOW
                               ? PIPE_CONTEXT_PRIORITY_LOW
                               : priority == DRM_ASAHI_PRIORITY_HIGH
                                    ? PIPE_CONTEXT_PRIORITY_HIGH
                                    : PIPE_CONTEXT_PRIORITY_MEDIUM;
   ctx->virt_ring_idx = priority + 1;

   pctx->destroy = agx_destroy_context;
   pctx->flush = agx_flush;
   pctx->clear = agx_clear;
   pctx->resource_copy_region = agx_resource_copy_region;
   pctx->blit = agx_blit;
   if (agx_chip_g17_m5(agx_device(screen)->chip))
      pctx->generate_mipmap = agx_generate_mipmap_g17g;
   pctx->flush_resource = agx_flush_resource;

   pctx->buffer_map = u_transfer_helper_transfer_map;
   pctx->buffer_unmap = u_transfer_helper_transfer_unmap;
   pctx->texture_map = u_transfer_helper_transfer_map;
   pctx->texture_unmap = u_transfer_helper_transfer_unmap;
   pctx->transfer_flush_region = u_transfer_helper_transfer_flush_region;

   pctx->buffer_subdata = u_default_buffer_subdata;
   pctx->clear_buffer = asahi_clear_buffer;
   pctx->texture_subdata = agx_texture_subdata;
   pctx->set_debug_callback = u_default_set_debug_callback;
   pctx->get_sample_position = u_default_get_sample_position;
   pctx->invalidate_resource = agx_invalidate_resource;
   pctx->memory_barrier = agx_memory_barrier;

   pctx->create_fence_fd = agx_create_fence_fd;
   pctx->fence_server_sync = agx_fence_server_sync;

   pctx->get_device_reset_status = asahi_get_device_reset_status;
   pctx->set_device_reset_callback = asahi_set_device_reset_callback;

   agx_init_state_functions(pctx);
   agx_init_query_functions(pctx);
   agx_init_streamout_functions(pctx);

   agx_bg_eot_init(&ctx->bg_eot, agx_device(screen));
   agx_init_meta_shaders(ctx);

   ctx->blitter = util_blitter_create(pctx);
   ctx->compute_blitter.blit_cs = asahi_blit_key_table_create(ctx);

   struct agx_device *dev = agx_device(screen);
   if (screen->caps.query_timestamp || screen->caps.query_time_elapsed) {
      size_t timestamps_size = sizeof(struct agx_timestamps) * AGX_MAX_BATCHES;

      /* The kernel requires that timestamp buffers are SHARED. */
      ctx->timestamps =
         agx_bo_create(dev, timestamps_size, 0, AGX_BO_WRITEBACK | AGX_BO_SHARED,
                       "Timestamp buffer");
      assert(ctx->timestamps);

      ret = agx_bind_timestamps(dev, ctx->timestamps, &ctx->timestamp_handle);
      assert(!ret);
   }

   /* Sync object/FD used for NATIVE_FENCE_FD. */
   ctx->in_sync_fd = -1;
   ret = drmSyncobjCreate(agx_device(screen)->fd, 0, &ctx->in_sync_obj);
   assert(!ret);

   /* Dummy sync object used before any work has been submitted. */
   ret = drmSyncobjCreate(agx_device(screen)->fd, DRM_SYNCOBJ_CREATE_SIGNALED,
                          &ctx->dummy_syncobj);
   assert(!ret);
   ctx->syncobj = ctx->dummy_syncobj;

   /* By default all samples are enabled */
   ctx->sample_mask = ~0;

   /* The sampler descriptor's level-of-detail bias is an eleven-bit field in
    * 1/64 steps, and the hardware quantises the level it finally samples at to
    * 1/64 as well. Rounding the API bias into the descriptor therefore costs a
    * step the hardware was going to take anyway, and the two compound: a
    * sampler bias of 0.629921 and a shader bias of 1.889764 should sample at
    * 2.519685 and sampled at 2.500000, two steps low, which is what
    * KHR-GL30.texture_lod_bias.texture_lod_bias_all reports. Summing the two
    * biases in floating point first leaves the hardware's single step, so
    * every chip takes nir_lower_tex's software path and no descriptor carries
    * the bias. */
   ctx->support_lod_bias = !(flags & PIPE_CONTEXT_NO_LOD_BIAS);
   ctx->robust = (flags & PIPE_CONTEXT_ROBUST_BUFFER_ACCESS);

   agx_scratch_init(agx_device(screen), &ctx->scratch_vs);
   agx_scratch_init(agx_device(screen), &ctx->scratch_fs);
   agx_scratch_init(agx_device(screen), &ctx->scratch_cs);

   return pctx;
}

static const char *
agx_get_vendor(struct pipe_screen *pscreen)
{
   return "Mesa";
}

static const char *
agx_get_device_vendor(struct pipe_screen *pscreen)
{
   return "Apple";
}

static const char *
agx_get_name(struct pipe_screen *pscreen)
{
   struct agx_device *dev = agx_device(pscreen);

   return dev->name;
}

static void
agx_query_memory_info(struct pipe_screen *pscreen,
                      struct pipe_memory_info *info)
{
   struct agx_screen *screen = agx_screen(pscreen);
   uint64_t mem_kB =
      os_get_gpu_heap_size(screen->heap_memory_percent, NULL) / 1024;

   *info = (struct pipe_memory_info){
      .total_device_memory = mem_kB,
      .avail_device_memory = mem_kB,
   };
}

static void
agx_init_shader_caps(struct pipe_screen *pscreen)
{
   bool is_no16 = agx_device(pscreen)->debug & AGX_DBG_NO16;

   for (unsigned i = 0; i <= MESA_SHADER_COMPUTE; i++) {
      struct pipe_shader_caps *caps =
         (struct pipe_shader_caps *)&pscreen->shader_caps[i];

      caps->max_instructions = caps->max_alu_instructions =
         caps->max_tex_instructions = caps->max_tex_indirections = 16384;

      caps->max_control_flow_depth = 1024;

      caps->max_inputs = i == MESA_SHADER_VERTEX ? 16 : 32;

      /* For vertex, the spec min/max is 16. We need more to handle dmat3
       * correctly, though. The full 32 is undesirable since it would require
       * shenanigans to handle.
       */
      caps->max_outputs = i == MESA_SHADER_FRAGMENT ? 8
                          : i == MESA_SHADER_VERTEX ? 24
                                                    : 32;

      caps->max_temps = 256; /* GL_MAX_PROGRAM_TEMPORARIES_ARB */

      caps->max_const_buffer0_size = 16 * 1024 * sizeof(float);

      caps->max_const_buffers = 16;

      caps->cont_supported = true;

      caps->indirect_temp_addr = true;
      caps->indirect_const_addr = true;
      caps->integers = true;

      caps->fp16 = caps->glsl_16bit_consts = caps->fp16_derivatives = !is_no16;
      /* GLSL compiler is broken. Flip this on when Panfrost does. */
      caps->int16 = false;
      /* This cap is broken, see 9a38dab2d18 ("zink: disable
       * pipe_shader_caps.fp16_const_buffers") */
      caps->fp16_const_buffers = false;
      caps->glsl_16bit_load_dst = caps->fp16;

      /* G17P's Ioto path publishes a native 32-byte per-stage sampler table.
       * Slots 15+ are selected indirectly after accounting for the reserved
       * texel-fetch entry, so native graphics and compute can expose
       * Gallium's complete API table without the legacy heap. Geometry
       * shaders use the same table through their lowered compute programs. */
      if (agx_chip_native_agx3(agx_device(pscreen)->chip) &&
          (i == MESA_SHADER_VERTEX || i == MESA_SHADER_FRAGMENT ||
           i == MESA_SHADER_GEOMETRY || i == MESA_SHADER_COMPUTE))
         caps->max_texture_samplers = PIPE_MAX_SAMPLERS;
      /* TODO: Enable when fully baked */
      else if (strcmp(util_get_process_name(), "blender") == 0)
         caps->max_texture_samplers = PIPE_MAX_SAMPLERS;
      else if (strcmp(util_get_process_name(), "run") == 0)
         caps->max_texture_samplers = PIPE_MAX_SAMPLERS;
      else if (strcasestr(util_get_process_name(), "ryujinx") != NULL)
         caps->max_texture_samplers = PIPE_MAX_SAMPLERS;
      else
         caps->max_texture_samplers = 16;

      caps->max_sampler_views = PIPE_MAX_SHADER_SAMPLER_VIEWS;

      caps->supported_irs = (1 << PIPE_SHADER_IR_NIR);

      caps->max_shader_buffers = PIPE_MAX_SHADER_BUFFERS;

      caps->max_shader_images = PIPE_MAX_SHADER_IMAGES;
   }
}

static void
agx_init_compute_caps(struct pipe_screen *pscreen)
{
   struct pipe_compute_caps *caps =
      (struct pipe_compute_caps *)&pscreen->compute_caps;
   struct agx_screen *screen = agx_screen(pscreen);
   struct agx_device *dev = agx_device(pscreen);

   caps->address_bits = 64;

   caps->grid_dimension = 3;

   caps->max_grid_size[0] = caps->max_grid_size[1] = caps->max_grid_size[2] =
      65535;

   caps->max_block_size[0] = caps->max_block_size[1] = caps->max_block_size[2] =
      1024;

   caps->max_threads_per_block = 1024;

   caps->max_global_size = caps->max_mem_alloc_size =
      os_get_gpu_heap_size(screen->heap_memory_percent, NULL);

   caps->max_local_size = 32768;

   caps->max_clock_frequency = dev->params.max_frequency_khz / 1000;

   caps->max_compute_units = agx_get_num_cores(dev);

   caps->subgroup_sizes = 32;
   caps->max_variable_threads_per_block = 1024;
   caps->max_subgroups = caps->max_variable_threads_per_block / 32;
}

/*
 * Whether the kernel binds a timestamp object: timer queries and the
 * per-context timestamp buffer need DRM_ASAHI_GEM_BIND_OBJECT with
 * DRM_ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS. Asked of the kernel with a
 * throwaway binding rather than assumed.
 */
static bool
agx_kernel_binds_timestamps(struct agx_device *dev)
{
   struct agx_bo *bo =
      agx_bo_create(dev, sizeof(struct agx_timestamps) * AGX_MAX_BATCHES, 0,
                    AGX_BO_WRITEBACK | AGX_BO_SHARED, "Timestamp probe");
   if (!bo)
      return false;

   uint32_t handle = 0;
   const bool bound = agx_bind_timestamps(dev, bo, &handle) == 0;
   if (bound)
      dev->ops.bo_unbind_object(dev, handle);
   agx_bo_unreference(dev, bo);
   return bound;
}

static void
agx_init_screen_caps(struct pipe_screen *pscreen)
{
   struct pipe_caps *caps = (struct pipe_caps *)&pscreen->caps;
   struct agx_screen *screen = agx_screen(pscreen);

   u_init_pipe_screen_caps(pscreen, 1);

   /* G17G generates mipmaps on the CPU (agx_generate_mipmap_g17g). */
   caps->generate_mipmap = agx_chip_g17_m5(agx_device(pscreen)->chip);

   caps->clip_halfz = true;
   caps->prefer_real_buffer_in_constbuf0 = true;
   caps->npot_textures = true;
   caps->shader_stencil_export = true;
   caps->mixed_color_depth_bits = true;
   caps->fragment_shader_texture_lod = true;
   caps->vertex_color_unclamped = true;
   caps->depth_clip_disable = true;
   caps->mixed_framebuffer_sizes = true;
   caps->fragment_shader_derivatives = true;
   caps->framebuffer_no_attachment = true;
   caps->shader_pack_half_float = true;
   caps->fs_fine_derivative = true;
   caps->glsl_tess_levels_as_inputs = true;
   caps->doubles = true;

   /* Every G18G stage, compute included, publishes its own native texture
    * and sampler tables (agx_upload_textures/agx_upload_samplers), so GL's
    * combined unit count is their sum, as GLES already reports. No stage
    * binds more than max_texture_samplers.
    */
   caps->combined_texture_units_all_stages =
      agx_device(pscreen)->chip == AGX_CHIP_G18G;

   caps->max_render_targets = caps->fbfetch = 8;
   caps->fbfetch_coherent = true;

   caps->max_dual_source_render_targets = 1;

   caps->occlusion_query = true;
   /* Timer queries are not optional for GL 3.3 (ARB_timer_query), which
    * gates every later desktop GL version, so on M6 (G18G), whose kernel
    * contract is still being written, and on the M5 Pro and M5 Max (G17S,
    * G17C), whose G17 runtime maps timestamp objects, follow
    * whether the kernel binds a timestamp object rather than assuming or
    * refusing it.
    */
   struct agx_device *dev = agx_device(pscreen);
   const bool timestamps =
      agx_params_g18g_bringup_supported(&dev->params) ||
            dev->chip == AGX_CHIP_G17S || dev->chip == AGX_CHIP_G17C
         ? agx_kernel_binds_timestamps(dev)
         : !agx_params_m5_supported(&dev->params);
   caps->query_timestamp = timestamps;
   caps->query_time_elapsed = timestamps;
   caps->query_so_overflow = true;
   caps->query_memory_info = true;
   caps->primitive_restart = true;
   caps->primitive_restart_fixed_index = true;
   caps->polygon_stipple = true;
   caps->anisotropic_filter = true;
   caps->native_fence_fd = true;
   caps->texture_barrier = true;

   /* Timer resolution is the length of a single tick in nanos */
   caps->timer_resolution = agx_gpu_timestamp_to_ns(agx_device(pscreen), 1);

   caps->shader_subgroup_size = 32;
   caps->shader_subgroup_supported_stages = BITFIELD_MASK(MESA_SHADER_STAGES);
   caps->shader_subgroup_supported_features =
      BITFIELD_MASK(PIPE_SHADER_SUBGROUP_FEATURE_MASK);
   caps->shader_subgroup_quad_all_stages = true;

   caps->sampler_view_target = true;
   caps->texture_swizzle = true;
   caps->blend_equation_separate = true;
   caps->indep_blend_enable = true;
   caps->indep_blend_func = true;
   caps->uma = true;
   caps->texture_float_linear = true;
   caps->texture_half_float_linear = true;
   caps->texture_mirror_clamp_to_edge = true;
   caps->shader_array_components = true;
   caps->packed_uniforms = true;
   caps->quads_follow_provoking_vertex_convention = true;
   caps->vs_instanceid = true;
   caps->vertex_element_instance_divisor = true;
   caps->conditional_render = true;
   caps->conditional_render_inverted = true;
   caps->seamless_cube_map = true;
   caps->load_constbuf = true;
   caps->seamless_cube_map_per_texture = true;
   caps->texture_buffer_objects = true;
   caps->null_textures = true;
   caps->texture_multisample = true;
   caps->image_load_formatted = true;
   caps->image_store_formatted = true;
   caps->compute = true;
   caps->int64 = true;
   caps->sample_shading = true;
   caps->start_instance = true;
   caps->draw_parameters = true;
   caps->multi_draw_indirect = true;
   caps->multi_draw_indirect_params = true;
   caps->cull_distance = true;
   caps->gl_spirv = true;
   caps->polygon_offset_clamp = true;

   /* TODO: MSRTT */
   caps->surface_sample_count = false;

   caps->cube_map_array = true;

   caps->copy_between_compressed_and_plain_formats = true;

   caps->max_stream_output_buffers = PIPE_MAX_SO_BUFFERS;

   caps->max_stream_output_separate_components =
      caps->max_stream_output_interleaved_components = PIPE_MAX_SO_OUTPUTS;

   caps->stream_output_pause_resume = true;
   caps->stream_output_interleave_buffers = true;

   caps->max_texture_array_layers = 2048;

   caps->glsl_feature_level = caps->glsl_feature_level_compatibility = 460;
   caps->essl_feature_level = 320;

   /* Settings from iris, may need tuning */
   caps->max_vertex_streams = 4;
   caps->max_geometry_output_vertices = 256;
   caps->max_geometry_total_output_components = 1024;
   caps->max_gs_invocations = 32;
   caps->constant_buffer_offset_alignment = 16;

   caps->max_texel_buffer_elements = AGX_TEXTURE_BUFFER_MAX_SIZE;

   caps->texture_buffer_offset_alignment = 64;

   caps->vertex_input_alignment = PIPE_VERTEX_INPUT_ALIGNMENT_ELEMENT;

   caps->query_pipeline_statistics_single = true;

   caps->max_texture_2d_size = 16384;
   caps->max_texture_cube_levels = 15; /* Max 16384x16384 */
   caps->max_texture_3d_levels = 12;   /* Max 2048x2048x2048 */

   caps->fs_coord_origin_upper_left = true;
   caps->fs_coord_pixel_center_integer = true;
   caps->tgsi_texcoord = true;
   caps->fs_face_is_integer_sysval = true;
   caps->fs_position_is_sysval = true;

   caps->fs_coord_origin_lower_left = false;
   caps->fs_coord_pixel_center_half_integer = false;
   caps->fs_point_is_sysval = false;

   caps->max_vertex_element_src_offset = 0xffff;

   /* On AGX3, advertising blit-based transfers routes a format-mismatched
    * glReadPixels through u_blitter.  In particular, the WSI-native
    * B8G8R8A8_UNORM -> API R8G8B8A8_UNORM conversion silently returns zero on
    * G17P even though the same drawable is byte-exact through a dma-buf CPU
    * map after both CPU upload and GPU clear.  Keep using the ordinary mapped
    * transfer path, which performs the format conversion on the CPU, until
    * the AGX3 render-blitter pixel path is hardware-grounded.
    *
    * This is correctness-critical for every default EGL surface: the DRI
    * pbuffer/window colour buffer is BGRA while GLES glReadPixels(GL_RGBA)
    * requests RGBA.  AGX2 retains its established accelerated path.
    */
   caps->texture_transfer_modes =
      agx_chip_command_stream_generation(agx_device(pscreen)->chip) ==
            AGX_CS_GENERATION_AGX2
         ? PIPE_TEXTURE_TRANSFER_BLIT
         : PIPE_TEXTURE_TRANSFER_DEFAULT;

   caps->device_type = PIPE_DEVICE_TYPE_INTEGRATED_GPU;
   caps->endianness = PIPE_ENDIAN_LITTLE;

   caps->shader_group_vote = true;
   caps->shader_ballot = true;

   caps->max_texture_gather_components = 4;
   caps->min_texture_gather_offset = -8;
   caps->max_texture_gather_offset = 7;
   caps->draw_indirect = true;
   caps->texture_query_samples = true;
   caps->texture_query_lod = true;
   caps->texture_shadow_lod = true;

   caps->max_viewports = AGX_MAX_VIEWPORTS;

   caps->video_memory =
      os_get_gpu_heap_size(screen->heap_memory_percent, NULL) >> 20;

   caps->device_reset_status_query = true;
   caps->robust_buffer_access_behavior = true;

   caps->shader_buffer_offset_alignment = 4;

   caps->max_shader_patch_varyings = 32;
   /* TODO: Probably should bump to 32? */
   caps->max_varyings = 16;

   caps->flatshade = false;
   caps->two_sided_color = false;
   caps->alpha_test = false;
   caps->clip_planes = 0;
   caps->nir_images_as_deref = false;

   caps->query_buffer_object = true;

   caps->texture_border_color_quirk =
      PIPE_QUIRK_TEXTURE_BORDER_COLOR_SWIZZLE_FREEDRENO;

   caps->supported_prim_modes = caps->supported_prim_modes_with_restart =
      BITFIELD_BIT(MESA_PRIM_POINTS) | BITFIELD_BIT(MESA_PRIM_LINES) |
      BITFIELD_BIT(MESA_PRIM_LINE_STRIP) | BITFIELD_BIT(MESA_PRIM_LINE_LOOP) |
      BITFIELD_BIT(MESA_PRIM_TRIANGLES) |
      BITFIELD_BIT(MESA_PRIM_TRIANGLE_STRIP) |
      BITFIELD_BIT(MESA_PRIM_TRIANGLE_FAN) |
      BITFIELD_BIT(MESA_PRIM_LINES_ADJACENCY) |
      BITFIELD_BIT(MESA_PRIM_LINE_STRIP_ADJACENCY) |
      BITFIELD_BIT(MESA_PRIM_TRIANGLES_ADJACENCY) |
      BITFIELD_BIT(MESA_PRIM_TRIANGLE_STRIP_ADJACENCY) |
      BITFIELD_BIT(MESA_PRIM_PATCHES);

   caps->map_unsynchronized_thread_safe = true;

   caps->vs_layer_viewport = true;
   caps->tes_layer_viewport = true;

   caps->context_priority_mask =
      PIPE_CONTEXT_PRIORITY_LOW | PIPE_CONTEXT_PRIORITY_MEDIUM;

   /* A chip/HAL match cannot establish the caller's authorization. The probe
    * uses a private queue and immediately destroys it, without submitting work.
    * Recheck creation normally: credentials/master can change after this probe.
    */
   if (agx_probe_high_priority(agx_device(pscreen)))
      caps->context_priority_mask |= PIPE_CONTEXT_PRIORITY_HIGH;

   caps->min_line_width = caps->min_line_width_aa = caps->min_point_size =
      caps->min_point_size_aa = 1;

   caps->point_size_granularity = caps->line_width_granularity = 0.1;

   caps->max_line_width = caps->max_line_width_aa =
      16.0; /* Off-by-one fixed point 4:4 encoding */

   caps->max_point_size = caps->max_point_size_aa = 511.95f;

   caps->max_texture_anisotropy = 16.0;

   caps->max_texture_lod_bias = 16.0; /* arbitrary */

   /* Up to 16 bytes are accelerated */
   caps->hw_clear_buffer_sizes = 1 | 2 | 4 | 8 | 16;
}

static bool
agx_is_format_supported(struct pipe_screen *pscreen, enum pipe_format format,
                        enum pipe_texture_target target, unsigned sample_count,
                        unsigned storage_sample_count, unsigned usage)
{
   assert(target == PIPE_BUFFER || target == PIPE_TEXTURE_1D ||
          target == PIPE_TEXTURE_1D_ARRAY || target == PIPE_TEXTURE_2D ||
          target == PIPE_TEXTURE_2D_ARRAY || target == PIPE_TEXTURE_RECT ||
          target == PIPE_TEXTURE_3D || target == PIPE_TEXTURE_CUBE ||
          target == PIPE_TEXTURE_CUBE_ARRAY);

   if (sample_count > 1 && sample_count != 4 && sample_count != 2)
      return false;

   if (sample_count > 1 && (agx_device(pscreen)->debug & AGX_DBG_NOMSAA))
      return false;

   if (MAX2(sample_count, 1) != MAX2(storage_sample_count, 1))
      return false;

   if ((usage & PIPE_BIND_VERTEX_BUFFER) && !agx_vbo_supports_format(format))
      return false;

   /* For framebuffer_no_attachments, fake support for "none" images */
   if (format == PIPE_FORMAT_NONE)
      return true;

   if (usage & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW |
                PIPE_BIND_SHADER_IMAGE)) {
      enum pipe_format tex_format = format;

      /* Mimic the fixup done in create_sampler_view and u_transfer_helper so we
       * advertise GL_OES_texture_stencil8. Alternatively, we could make mesa/st
       * less stupid?
       */
      if (tex_format == PIPE_FORMAT_X24S8_UINT)
         tex_format = PIPE_FORMAT_S8_UINT;

      struct ail_pixel_format_entry ent = ail_pixel_format[tex_format];

      if (!ail_is_valid_pixel_format(tex_format))
         return false;

      /* RGB32, luminance/alpha/intensity emulated for texture buffers only */
      if ((ent.channels == AGX_CHANNELS_R32G32B32_EMULATED ||
           util_format_is_luminance(tex_format) ||
           util_format_is_alpha(tex_format) ||
           util_format_is_luminance_alpha(tex_format) ||
           util_format_is_intensity(tex_format)) &&
          target != PIPE_BUFFER)
         return false;

      /* XXX: sort out rgb9e5 rendering */
      if ((usage & PIPE_BIND_RENDER_TARGET) &&
          (!ent.renderable || (tex_format == PIPE_FORMAT_R9G9B9E5_FLOAT)))
         return false;
   }

   if (usage & PIPE_BIND_DEPTH_STENCIL) {
      switch (format) {
      /* natively supported */
      case PIPE_FORMAT_Z16_UNORM:
      case PIPE_FORMAT_Z32_FLOAT:
      case PIPE_FORMAT_S8_UINT:

      /* lowered by u_transfer_helper to one of the above */
      case PIPE_FORMAT_Z24X8_UNORM:
      case PIPE_FORMAT_Z24_UNORM_S8_UINT:
      case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
         break;

      default:
         return false;
      }
   }

   return true;
}

static bool
agx_dmabuf_modifier_external_only(struct pipe_screen *screen, uint64_t modifier)
{
   /* AGX3 samples imported compressed images, but a shared compressed
    * render target may need in-place decompression, which has no AGX3 build
    * (agx_gallium_shared_compression_supported()). Keep the modifier
    * sample-only there.
    */
   return modifier == DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED &&
          !agx_gallium_shared_compression_supported(
             agx_chip_descriptor_generation(agx_device(screen)->chip));
}

static void
agx_query_dmabuf_modifiers(struct pipe_screen *screen, enum pipe_format format,
                           int max, uint64_t *modifiers,
                           unsigned int *external_only, int *out_count)
{
   int i;

   if (max == 0) {
      *out_count = ARRAY_SIZE(agx_best_modifiers);
      return;
   }

   for (i = 0; i < ARRAY_SIZE(agx_best_modifiers) && i < max; i++) {
      if (external_only)
         external_only[i] =
            agx_dmabuf_modifier_external_only(screen, agx_best_modifiers[i]);

      modifiers[i] = agx_best_modifiers[i];
   }

   /* Return the number of modifiers copied */
   *out_count = i;
}

static bool
agx_is_dmabuf_modifier_supported(struct pipe_screen *screen, uint64_t modifier,
                                 enum pipe_format format, bool *external_only)
{
   for (unsigned i = 0; i < ARRAY_SIZE(agx_best_modifiers); ++i) {
      if (agx_best_modifiers[i] == modifier) {
         if (external_only)
            *external_only =
               agx_dmabuf_modifier_external_only(screen, modifier);

         return true;
      }
   }

   if (external_only)
      *external_only = false;

   return false;
}

static void
agx_destroy_screen(struct pipe_screen *pscreen)
{
   struct agx_screen *screen = agx_screen(pscreen);

   /* Unknown completion cannot be repaired by tearing down userspace owners.
    * Quarantined contexts retain this screen and its device until exit.
    */
   if (p_atomic_read(&screen->submit_quarantined))
      return;

   drmSyncobjDestroy(screen->dev.fd, screen->flush_syncobj);

   if (screen->dev.ro)
      screen->dev.ro->destroy(screen->dev.ro);

   agx_bo_unreference(&screen->dev, screen->rodata);
   u_transfer_helper_destroy(pscreen->transfer_helper);
   agx_close_device(&screen->dev);
   disk_cache_destroy(screen->disk_cache);
   ralloc_free(screen);
}

static void
agx_resource_set_stencil(struct pipe_resource *prsrc,
                         struct pipe_resource *stencil)
{
   agx_resource(prsrc)->separate_stencil = agx_resource(stencil);
}

static struct pipe_resource *
agx_resource_get_stencil(struct pipe_resource *prsrc)
{
   return (struct pipe_resource *)agx_resource(prsrc)->separate_stencil;
}

static enum pipe_format
agx_resource_get_internal_format(struct pipe_resource *prsrc)
{
   return agx_resource(prsrc)->layout.format;
}

static struct disk_cache *
agx_get_disk_shader_cache(struct pipe_screen *pscreen)
{
   return agx_screen(pscreen)->disk_cache;
}

static const struct u_transfer_vtbl transfer_vtbl = {
   .resource_create = agx_resource_create,
   .resource_destroy = agx_resource_destroy,
   .transfer_map = agx_transfer_map,
   .transfer_unmap = agx_transfer_unmap,
   .transfer_flush_region = agx_transfer_flush_region,
   .get_internal_format = agx_resource_get_internal_format,
   .set_stencil = agx_resource_set_stencil,
   .get_stencil = agx_resource_get_stencil,
};

static int
agx_screen_get_fd(struct pipe_screen *pscreen)
{
   return agx_device(pscreen)->fd;
}

static uint64_t
agx_get_timestamp(struct pipe_screen *pscreen)
{
   return agx_get_gpu_timestamp(agx_device(pscreen));
}

static void
agx_screen_get_device_uuid(struct pipe_screen *pscreen, char *uuid)
{
   agx_get_device_uuid(agx_device(pscreen), uuid);
}

static void
agx_screen_get_driver_uuid(struct pipe_screen *pscreen, char *uuid)
{
   agx_get_driver_uuid(uuid);
}

static const char *
agx_get_cl_cts_version(struct pipe_screen *pscreen)
{
   struct agx_device *dev = agx_device(pscreen);

   /* https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_433
    * https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_471
    */
   if (dev->params.gpu_generation < 15)
      return "v2026-03-25-00";

   return NULL;
}

struct pipe_screen *
agx_screen_create(int fd, struct renderonly *ro,
                  const struct pipe_screen_config *config)
{
   struct agx_screen *agx_screen;
   struct pipe_screen *screen;

   agx_screen = rzalloc(NULL, struct agx_screen);
   if (!agx_screen)
      return NULL;

   screen = &agx_screen->pscreen;

   /* parse driconf configuration now for device specific overrides */
   driParseConfigFiles(config->options, config->options_info,
                       &(driConfigFileParseParams) {
                          .driverName = "aurora",
                          .driverNameAlias = "asahi",
                       });

   agx_screen->dev.fd = fd;
   agx_screen->dev.ro = ro;
   u_rwlock_init(&agx_screen->destroy_lock);

   /* Try to open an AGX device */
   if (!agx_open_device(agx_screen, &agx_screen->dev)) {
      ralloc_free(agx_screen);
      return NULL;
   }

   /* Forward no16 flag from driconf. This must happen after opening the device,
    * since agx_open_device sets debug.
    */
   if (driQueryOptionb(config->options, "asahi_no_fp16"))
      agx_screen->dev.debug |= AGX_DBG_NO16;

   /* G18G's packer has grounded half-precision forms for only part of what
    * mediump lowering produces (half FSEL/ADD/FMA with immediates or abs
    * sources), and one refused instruction drops the whole shader. mediump
    * is a precision floor: evaluate it at full precision on G18G. */
   if (agx_screen->dev.chip == AGX_CHIP_G18G)
      agx_screen->dev.debug |= AGX_DBG_NO16;

   int ret =
      drmSyncobjCreate(agx_device(screen)->fd, 0, &agx_screen->flush_syncobj);
   assert(!ret);

   simple_mtx_init(&agx_screen->flush_seqid_lock, mtx_plain);

   agx_screen->heap_memory_percent =
      driQueryOptionf(config->options, "heap_memory_percent");
   if (agx_screen->heap_memory_percent == OS_GPU_HEAP_SIZE_HEURISTIC)
      agx_screen->heap_memory_percent = 1.0f;

   screen->destroy = agx_destroy_screen;
   screen->get_screen_fd = agx_screen_get_fd;
   screen->get_name = agx_get_name;
   screen->get_vendor = agx_get_vendor;
   screen->get_device_vendor = agx_get_device_vendor;
   screen->get_device_uuid = agx_screen_get_device_uuid;
   screen->get_driver_uuid = agx_screen_get_driver_uuid;
   screen->is_format_supported = agx_is_format_supported;
   screen->query_dmabuf_modifiers = agx_query_dmabuf_modifiers;
   screen->query_memory_info = agx_query_memory_info;
   screen->is_dmabuf_modifier_supported = agx_is_dmabuf_modifier_supported;
   screen->context_create = agx_create_context;
   screen->resource_from_handle = agx_resource_from_handle;
   screen->resource_get_handle = agx_resource_get_handle;
   screen->resource_get_param = agx_resource_get_param;
   screen->resource_create_with_modifiers = agx_resource_create_with_modifiers;
   screen->set_damage_region = agx_resource_set_damage_region;
   screen->get_timestamp = agx_get_timestamp;
   screen->fence_reference = agx_fence_reference;
   screen->fence_finish = agx_fence_finish;
   screen->fence_get_fd = agx_fence_get_fd;
   screen->get_disk_shader_cache = agx_get_disk_shader_cache;
   screen->get_cl_cts_version = agx_get_cl_cts_version;

   /* G17G iterates no varying with a register coefficient index: have
    * nir_lower_io_passes lower dynamically indexed fragment inputs. */
   static nir_shader_compiler_options g17g_nir_options;
   g17g_nir_options = agx_nir_options;
   g17g_nir_options.support_indirect_inputs &=
      ~BITFIELD_BIT(MESA_SHADER_FRAGMENT);
   for (unsigned i = 0; i <= MESA_SHADER_COMPUTE; i++)
      screen->nir_options[i] = agx_chip_g17_m5(agx_screen->dev.chip)
                                  ? &g17g_nir_options
                                  : &agx_nir_options;

   screen->resource_create = u_transfer_helper_resource_create;
   screen->resource_destroy = u_transfer_helper_resource_destroy;
   screen->transfer_helper = u_transfer_helper_create(
      &transfer_vtbl,
      U_TRANSFER_HELPER_SEPARATE_Z32S8 | U_TRANSFER_HELPER_SEPARATE_STENCIL |
         U_TRANSFER_HELPER_MSAA_MAP | U_TRANSFER_HELPER_Z24_IN_Z32F);

   agx_init_shader_caps(screen);
   agx_init_compute_caps(screen);
   agx_init_screen_caps(screen);

   agx_disk_cache_init(agx_screen);

   /* TODO: Refactor readonly data? */
   {
      struct agx_bo *bo =
         agx_bo_create(&agx_screen->dev, 16384, 0, 0, "Rodata");

      agx_pack_txf_sampler((struct agx_sampler_packed *)agx_bo_map(bo));

      agx_pack(&agx_screen->dev.txf_sampler, USC_SAMPLER, cfg) {
         cfg.start = 0;
         cfg.count = 1;
         cfg.buffer = bo->va->addr;
      }

      agx_screen->rodata = bo;
   }

   return screen;
}

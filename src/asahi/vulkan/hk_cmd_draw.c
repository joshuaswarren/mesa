/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include <assert.h>
#include "agx_abi.h"
#include "agx_bg_eot.h"
#include "agx_bo.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "asahi/compiler/agx3_graphics_program.h"
#include "agx_device.h"
#include "agx_helpers.h"
#include "agx_linker.h"
#include "agx_nir_lower_vbo.h"
#include "agx_ppp.h"
#include "agx_ppp_hal200_graphics_core.h"
#include "agx_ppp_hal300_graphics_core.h"
#include "hk_ppp_state.h"
#include "agx_vdm_state_core.h"
#include "agx_ppp_hal200_raster_core.h"
#include "agx_ppp_hal300_depth_bias_scissor_core.h"
#include "agx_ppp_viewport_payload_core.h"
#include "agx_ppp_w_clamp_cull2_core.h"
#include "agx_tilebuffer.h"
#include "agx_usc.h"
#include "agx_uvs.h"
#include "agx_vdm_barrier_core.h"
#include "agx_vdm_hal200_bind_tail_core.h"
#include "agx_vdm_hal200_direct_draw_core.h"
#include "hk_buffer.h"
#include "hk_cmd_buffer.h"
#include "hk_native_vertex_template.h"
#include "hk_frame_anatomy.h"
#include "asahi/lib/agx_vdm_mesh_core.h"
#include "hk_descriptor_set.h"
#include "hk_descriptor_set_layout.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_instance.h"
#include "hk_agx3_ioto_program_core.h"
#include "hk_g17p_ppp_indices.h"
#include "hk_g17p_full_frame_scissor.h"
#include "hk_g17p_draw_state_upload.h"
#include "hk_agx3_vdm_decode.h"
#include "hk_physical_device.h"
#include "hk_private.h"
#include "asahi/lib/agx_m5_va.h"
#include "hk_shader.h"
#include "hk_native_write_class.h"
#include "hk_g17p_patch.h"
#include "util/disk_cache.h"
#include "agx_debug.h"
#include "asahi/lib/agx3_g17p_grid_helper_core.h"

#include "asahi/genxml/agx_pack.h"
#include "asahi/libagx/compression.h"
#include "asahi/libagx/libagx.h"
#include "asahi/libagx/query.h"
#include "asahi/libagx/tessellator.h"
#include "poly/geometry.h"
#include "util/blend.h"
#include "util/format/format_utils.h"
#include "util/format/u_formats.h"
#include "util/macros.h"
#include "util/os_time.h"
#include "util/ralloc.h"
#include "util/u_prim.h"
#include "vulkan/vulkan_core.h"
#include "layout.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "nir.h"
#include "nir_builder.h"
#include "util/blob.h"
#include "nir_serialize.h"
#include "nir_intrinsics.h"
#include "nir_lower_blend.h"
#include "nir_xfb_info.h"
#include "pool.h"
#include "shader_enums.h"
#include "vk_blend.h"

#include "vk_enum_to_str.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_pipeline.h"
#include "vk_render_pass.h"
#include "vk_shader.h"
#include "vk_standard_sample_locations.h"
#include "vk_util.h"
#include "util/set.h"

/* Diagnostic controls, read once: these are consulted per draw or
 * render pass, and an uncached getenv() scans the whole environment. */
DEBUG_GET_ONCE_BOOL_OPTION(m5_launch_debug, "ASAHI_M5_LAUNCH_DEBUG", false)
DEBUG_GET_ONCE_BOOL_OPTION(m5_msaa_pass_merge, "ASAHI_M5_MSAA_PASS_MERGE", true)
DEBUG_GET_ONCE_BOOL_OPTION(m5_verify_tile_loader, "AGX_M5_VERIFY_TILE_LOADER",
                           false)
DEBUG_GET_ONCE_BOOL_OPTION(mesh_native_primitive_address,
                           "ASAHI_MESH_NATIVE_PRIMITIVE_ADDRESS_DIAGNOSTIC",
                           false)
DEBUG_GET_ONCE_BOOL_OPTION(mesh_native_m5_wclamp,
                           "ASAHI_MESH_NATIVE_M5_WCLAMP_DIAGNOSTIC", false)
DEBUG_GET_ONCE_BOOL_OPTION(mesh_native_raster_enable,
                           "ASAHI_MESH_NATIVE_RASTER_ENABLE_DIAGNOSTIC", false)
DEBUG_GET_ONCE_BOOL_OPTION(mesh_pp_tail0, "ASAHI_MESH_PP_TAIL0", false)

#define IS_DIRTY(bit) BITSET_TEST(dyn->dirty, MESA_VK_DYNAMIC_##bit)

#define IS_SHADER_DIRTY(bit)                                                   \
   (cmd->state.gfx.shaders_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

#define IS_LINKED_DIRTY(bit)                                                   \
   (cmd->state.gfx.linked_dirty & BITFIELD_BIT(MESA_SHADER_##bit))

static void hk_flush_shaders(struct hk_cmd_buffer *cmd);
static struct hk_shader *hk_bound_fs(struct hk_cmd_buffer *cmd);
static uint64_t hk_heap(struct hk_cmd_buffer *cmd);
static bool hk_cmd_ensure_geometry_heap(struct hk_cmd_buffer *cmd);
static enum mesa_prim hk_gs_in_prim(struct hk_cmd_buffer *cmd);

static enum libagx_program
hk_sw_tess_program(enum tess_primitive_mode prim, enum poly_tess_mode mode)
{
   switch (prim) {
   case TESS_PRIMITIVE_QUADS:
      return LIBAGX_TESS_QUAD(mode);
   case TESS_PRIMITIVE_TRIANGLES:
      return LIBAGX_TESS_TRI(mode);
   case TESS_PRIMITIVE_ISOLINES:
      return LIBAGX_TESS_ISOLINE(mode);
   default:
      return LIBAGX_NUM_PROGRAMS;
   }
}

static uint64_t hk_upload_geometry_params(struct hk_cmd_buffer *cmd,
                                          struct agx_draw draw);
static void hk_init_tess_params(struct hk_cmd_buffer *cmd,
                                struct poly_tess_params *out);
static void hk_flush_dynamic_state(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint32_t draw_id,
   struct agx_draw draw, bool ordinary_draw_path,
   bool *g17p_native_bind_tail_emitted_out);

/* CTS coverage of indirect draws is pretty bad, so it's helpful to be able to
 * get some extra smoke testing.
 */
#define HK_TEST_INDIRECTS (0)

UNUSED static inline void
print_draw(struct agx_draw d, FILE *fp)
{
   if (agx_is_indirect(d.b))
      fprintf(fp, "indirect (buffer %" PRIx64 "):", d.b.ptr);
   else
      fprintf(fp, "direct (%ux%u):", d.b.count[0], d.b.count[1]);

   if (d.index_size)
      fprintf(fp, " index_size=%u", agx_index_size_to_B(d.index_size));
   else
      fprintf(fp, " non-indexed");

   if (d.restart)
      fprintf(fp, " restart");

   if (d.index_bias)
      fprintf(fp, " index_bias=%u", d.index_bias);

   if (d.start)
      fprintf(fp, " start=%u", d.start);

   if (d.start_instance)
      fprintf(fp, " start_instance=%u", d.start_instance);

   fprintf(fp, "\n");
}

static bool
hk_rast_discard(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   if (dyn->rs.rasterizer_discard_enable)
      return true;

   /* A non-zero rasterization stream acts as a rasterizer discard unless
    * there's a multistream geometry shader bound.
    */
   if (dyn->rs.rasterization_stream != 0) {
      struct hk_api_shader *gs = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
      return !gs || !gs->variants[HK_GS_VARIANT_COUNT].info.gs.multistream;
   }

   return false;
}

/* Report address of an active VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT query
 * on @stream, or 0. Meta draws are not counted. */
static uint64_t
hk_prims_generated_addr(struct hk_cmd_buffer *cmd, unsigned stream)
{
   return cmd->in_meta ? 0 : cmd->state.gfx.prims_generated_query[stream];
}

static bool
hk_prims_generated_active(struct hk_cmd_buffer *cmd)
{
   for (unsigned i = 0; i < ARRAY_SIZE(cmd->state.gfx.prims_generated_query); ++i) {
      if (hk_prims_generated_addr(cmd, i))
         return true;
   }
   return false;
}

static bool
hk_g17p_xfb_vertex_side_effects(struct hk_cmd_buffer *cmd,
                                struct hk_shader *hw_vs)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

   const struct hk_device *dev = hk_cmd_buffer_device(cmd);
   return (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
          gfx->xfb_enabled && gs && gs->is_passthrough &&
          hw_vs == &gs->variants[HK_GS_VARIANT_RAST];
}

static void
hk_cmd_buffer_dirty_render_pass(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* These depend on color attachment count */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ENABLES);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_WRITE_MASKS);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_CB_BLEND_ADVANCED);

   /* These depend on the depth/stencil format */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_WRITE_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_DEPTH_BOUNDS_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_DS_STENCIL_TEST_ENABLE);
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_DEPTH_BIAS_FACTORS);

   /* This may depend on render targets for ESO */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);

   /* This may depend on render targets */
   BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_COLOR_ATTACHMENT_MAP);
}

void
hk_cmd_buffer_begin_graphics(struct hk_cmd_buffer *cmd,
                             const VkCommandBufferBeginInfo *pBeginInfo)
{
   if (cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY &&
       (pBeginInfo->flags & VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT)) {
      char gcbiar_data[VK_GCBIARR_DATA_SIZE(HK_MAX_RTS)];
      const VkRenderingInfo *resume_info =
         vk_get_command_buffer_inheritance_as_rendering_resume(
            cmd->vk.level, pBeginInfo, gcbiar_data);
      if (resume_info) {
         hk_CmdBeginRendering(hk_cmd_buffer_to_handle(cmd), resume_info);
      } else {
         const VkCommandBufferInheritanceRenderingInfo *inheritance_info =
            vk_get_command_buffer_inheritance_rendering_info(cmd->vk.level,
                                                             pBeginInfo);
         assert(inheritance_info);

         struct hk_rendering_state *render = &cmd->state.gfx.render;
         render->flags = inheritance_info->flags;
         render->area = (VkRect2D){};
         render->layer_count = 0;
         render->view_mask = inheritance_info->viewMask;
         render->tilebuffer.nr_samples = inheritance_info->rasterizationSamples;

         render->color_att_count = inheritance_info->colorAttachmentCount;
         for (uint32_t i = 0; i < render->color_att_count; i++) {
            render->color_att[i].vk_format =
               inheritance_info->pColorAttachmentFormats[i];
         }
         render->depth_att.vk_format = inheritance_info->depthAttachmentFormat;
         render->stencil_att.vk_format =
            inheritance_info->stencilAttachmentFormat;

         const VkRenderingAttachmentLocationInfoKHR att_loc_info_default = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
            .colorAttachmentCount = inheritance_info->colorAttachmentCount,
         };
         const VkRenderingAttachmentLocationInfoKHR *att_loc_info =
            vk_get_command_buffer_rendering_attachment_location_info(
               cmd->vk.level, pBeginInfo);
         if (att_loc_info == NULL)
            att_loc_info = &att_loc_info_default;

         vk_cmd_set_rendering_attachment_locations(&cmd->vk, att_loc_info);

         hk_cmd_buffer_dirty_render_pass(cmd);
      }
   }

   hk_cmd_buffer_dirty_all(cmd);

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   cmd->state.gfx.descriptors.root.draw.view_index = 0;
   cmd->state.gfx.descriptors.root_dirty = true;
}

void
hk_cmd_invalidate_graphics_state(struct hk_cmd_buffer *cmd)
{
   /* From the Vulkan 1.3.275 spec:
    *
    *    "...There is one exception to this rule - if the primary command
    *    buffer is inside a render pass instance, then the render pass and
    *    subpass state is not disturbed by executing secondary command
    *    buffers."
    *
    * We need to reset everything EXCEPT the render pass state.
    *
    * Active queries are not bound state either: a query begun in the primary
    * stays active across vkCmdExecuteCommands and must also count the
    * primary's later commands, so keep every active query binding.
    */
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_rendering_state render_save = gfx->render;
   struct hk_push_descriptor_set *push[HK_MAX_SETS];
   memcpy(push, gfx->descriptors.push, sizeof(push));
   const uint64_t pipeline_stats = gfx->descriptors.root.draw.pipeline_stats;
   const VkQueryPipelineStatisticFlags pipeline_stats_flags =
      gfx->descriptors.root.draw.pipeline_stats_flags;
   const uint64_t mesh_prims_query = gfx->descriptors.root.mesh_prims_query;
   uint64_t xfb_query[ARRAY_SIZE(gfx->xfb_query)];
   uint64_t prims_generated_query[ARRAY_SIZE(gfx->prims_generated_query)];
   memcpy(xfb_query, gfx->xfb_query, sizeof(xfb_query));
   memcpy(prims_generated_query, gfx->prims_generated_query,
          sizeof(prims_generated_query));
   const enum agx_visibility_mode occlusion_mode = gfx->occlusion.mode;
   const uint16_t occlusion_index = gfx->occlusion.index;
   struct hk_query_pool *occlusion_pending_pool = gfx->occlusion.pending_pool;
   const uint32_t occlusion_pending_query = gfx->occlusion.pending_query;

   memset(gfx, 0, sizeof(*gfx));
   gfx->render = render_save;
   memcpy(gfx->descriptors.push, push, sizeof(push));
   gfx->descriptors.root.draw.pipeline_stats = pipeline_stats;
   gfx->descriptors.root.draw.pipeline_stats_flags = pipeline_stats_flags;
   gfx->descriptors.root.mesh_prims_query = mesh_prims_query;
   memcpy(gfx->xfb_query, xfb_query, sizeof(xfb_query));
   memcpy(gfx->prims_generated_query, prims_generated_query,
          sizeof(prims_generated_query));
   gfx->occlusion.mode = occlusion_mode;
   gfx->occlusion.index = occlusion_index;
   gfx->occlusion.pending_pool = occlusion_pending_pool;
   gfx->occlusion.pending_query = occlusion_pending_query;

   /* Every bound stage is gone, so vk_graphics_pipeline_cmd_bind() must not
    * elide the application's rebind of the pipeline it last saw (often the
    * one the secondary bound during replay): that left the primary drawing
    * with no vertex shader (VK_ERROR_FEATURE_NOT_PRESENT). */
   cmd->vk.last_graphics_pipeline = NULL;

   /* Dirty after clearing the cached state.  Doing this before memset loses
    * every dirty bit and lets a post-secondary rebind be elided. */
   hk_cmd_buffer_dirty_all(cmd);
}

static void
hk_attachment_init(struct hk_attachment *att,
                   const VkRenderingAttachmentInfo *info)
{
   if (info == NULL || info->imageView == VK_NULL_HANDLE) {
      *att = (struct hk_attachment){
         .iview = NULL,
      };
      return;
   }

   VK_FROM_HANDLE(hk_image_view, iview, info->imageView);
   *att = (struct hk_attachment){
      .flags = vk_get_rendering_attachment_flags(info),
      .vk_format = iview->vk.format,
      .iview = iview,
      .load_op = info->loadOp,
      .store_op = info->storeOp,
      .layout = info->imageLayout,
   };

   if (info->resolveMode != VK_RESOLVE_MODE_NONE) {
      VK_FROM_HANDLE(hk_image_view, res_iview, info->resolveImageView);
      att->resolve_mode = info->resolveMode;
      att->resolve_iview = res_iview;
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_GetRenderingAreaGranularityKHR(
   VkDevice device, const VkRenderingAreaInfoKHR *pRenderingAreaInfo,
   VkExtent2D *pGranularity)
{
   *pGranularity = (VkExtent2D){.width = 1, .height = 1};
}

static bool
is_attachment_stored(const VkRenderingAttachmentInfo *att)
{
   /* When resolving, we store the intermediate multisampled image as the
    * resolve is a separate control stream. This could be optimized.
    */
   return att->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
          att->resolveMode != VK_RESOLVE_MODE_NONE;
}

/* LDSHDR's fetch span saturates from 321 bytes of text on, so a longer
 * program, such as a multi-target mixed-format reload, loads exactly like the
 * main fragment shaders of any size. */
static bool
hk_g17p_bg_eot_shader_supported(const struct agx_bg_eot_shader *shader)
{
   return shader && shader->bo && shader->bo->va && shader->ptr != 0 &&
          shader->info.binary_size > 0 &&
          shader->info.main_offset < shader->info.binary_size &&
          shader->info.main_size <=
             shader->info.binary_size - shader->info.main_offset &&
          !shader->info.has_preamble && shader->info.rodata.size_16 == 0 &&
          shader->info.nr_gprs > 0;
}

/* Build the BG or EOT program of an AGX_G17P_MRT_MIXED layout. Only the
 * targets the operation touches are numbered, each keeping its framebuffer
 * tile offset. Reloads are numbered before clears: a reload reads its texture
 * and sampler pointers from uniform halves 0..7, which the clear value of a
 * target zero would otherwise overlap at halves 4..11. */
static struct hk_bg_eot
hk_build_g17p_mixed_bg_eot(struct hk_cmd_buffer *cmd,
                           const struct agx_bg_eot_key *framebuffer_key,
                           const VkRenderingInfo *info, bool store)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct agx_bg_eot_key key = *framebuffer_key;
   memset(key.tib.logical_format, 0, sizeof(key.tib.logical_format));
   memset(key.tib.spilled, 0, sizeof(key.tib.spilled));
   memset(key.tib._offset_B, 0, sizeof(key.tib._offset_B));
   memset(key.op, 0, sizeof(key.op));
   key.mixed_mrt = true;
   /* Multisample targets are stored unresolved: a multi-target pass
    * resolves afterwards, from the multisample images. */
   key.multisample_store = store && key.tib.nr_samples > 1;

   const VkRenderingAttachmentInfo *atts[HK_MAX_RTS];
   struct hk_image_view *views[HK_MAX_RTS];
   static const enum agx_bg_eot_op order[] = {AGX_BG_LOAD, AGX_BG_CLEAR,
                                              AGX_EOT_STORE};
   unsigned count = 0, reloads = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(order); ++i) {
      for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
         if (framebuffer_key->op[rt] != order[i])
            continue;

         struct hk_image_view *view = cmd->state.gfx.render.color_att[rt].iview;
         if (count == ARRAY_SIZE(views) || rt >= info->colorAttachmentCount ||
             !view ||
             view->vk.image->samples != MAX2(key.tib.nr_samples, 1) ||
             info->pColorAttachments[rt].imageView == VK_NULL_HANDLE) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FORMAT_NOT_SUPPORTED);
            return (struct hk_bg_eot){0};
         }

         key.tib.logical_format[count] = framebuffer_key->tib.logical_format[rt];
         key.tib._offset_B[count] = framebuffer_key->tib._offset_B[rt];
         key.op[count] = order[i];
         atts[count] = &info->pColorAttachments[rt];
         views[count++] = view;
         reloads += order[i] == AGX_BG_LOAD;
      }
   }

   const unsigned descriptor_generation =
      agx_chip_descriptor_generation(dev->dev.chip);
   if (hk_pbe_descriptor_size(descriptor_generation) !=
          AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE ||
       hk_texture_descriptor_size(descriptor_generation) !=
          AGX_TEXTURE_GEN9_LENGTH ||
       agx_chip_sampler_heap_element_size(dev->dev.chip) != 32) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }

   const bool clears = !store && count > reloads;
   if (clears)
      key.reserved_preamble = 4 + 8 * count;

   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(&dev->bg_eot, &key);
   if (!hk_g17p_bg_eot_shader_supported(shader) ||
       (!store && (shader->info.push_count != key.reserved_preamble ||
                   shader->info.uses_txf != (reloads > 0)))) {
      mesa_loge("G17P %s shader declined: supported=%u push=%u want=%u "
                "uses_txf=%u want=%u count=%u reloads=%u samples=%u "
                "layered=%u sample_size=%uB",
                store ? "EOT" : "background",
                hk_g17p_bg_eot_shader_supported(shader),
                shader ? shader->info.push_count : 0,
                key.reserved_preamble,
                shader ? shader->info.uses_txf : 0, reloads > 0, count,
                reloads, key.tib.nr_samples, key.tib.layered,
                key.tib.sample_size_B);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }

   const enum agx3_g17p_rt0_bg_op bg_op =
      count == 0         ? AGX3_G17P_RT0_BG_NONE
      : reloads == count ? AGX3_G17P_RT0_BG_RELOAD
      : reloads          ? AGX3_G17P_RT0_BG_CLEAR_RELOAD
                         : AGX3_G17P_RT0_BG_CLEAR;
   const size_t program_capacity =
      count == 0  ? AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE
      : store     ? (count > 4 ? AGX_IOTO_G17P_OCTA_EOT_PROGRAM_MAX_SIZE
                               : AGX_IOTO_G17P_QUAD_EOT_PROGRAM_MAX_SIZE)
      : count > 4 ? AGX_IOTO_G17P_OCTA_BACKGROUND_PROGRAM_MAX_SIZE
      : bg_op == AGX3_G17P_RT0_BG_CLEAR_RELOAD
         ? AGX_IOTO_G17P_CLEAR_RELOAD_PROGRAM_MAX_SIZE
         : AGX_IOTO_G17P_RT0_RELOAD_PROGRAM_MAX_SIZE;
   struct hk_ioto_reservation reservation;
   if (!hk_cmd_reserve_ioto_program(cmd, program_capacity,
                                    "G17P mixed MRT Ioto program",
                                    &reservation))
      return (struct hk_bg_eot){0};

   const struct agx_ioto_g17p_imageblock imageblock = {
      .sample_size_B = key.tib.sample_size_B,
      .nr_samples = key.tib.nr_samples,
      .tile_size_px = key.tib.tile_size,
   };
   const uint8_t *program_bytes;
   size_t program_length;
   uint64_t resource_spec;
   struct hk_agx3_g17p_graphics_program empty_program;
   struct agx3_g17p_rt0_eot_program eot_program;
   struct agx3_g17p_rt0_bg_program bg_program;
   if (count == 0) {
      const struct hk_agx3_g17p_graphics_program_desc desc = {
         .program_va = reservation.va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .compiler_d_valid = shader->info.agx3_staging_known,
         .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
         .fragment = true,
         .imageblock = imageblock,
      };
      if (!hk_agx3_g17p_basic_graphics_program_generate(
             dev->dev.chip, &desc, &empty_program) ||
          !agx3_g17p_rt0_bg_resource_spec(dev->dev.chip, bg_op,
                                          &resource_spec)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
      program_bytes = empty_program.bytes;
      program_length = empty_program.length;
   } else if (store) {
      static_assert(sizeof(struct hk_native_pbe_descriptor) ==
                       AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE,
                    "native PBE table stride");
      /* A layered pass stores through the 2D-ARRAY block image store, so its
       * targets need the array-shaped PBE descriptor. The plain BG/EOT
       * descriptor is deliberately flattened to one layer -- pack_pbe() forces
       * layers = 1 for HK_DESC_USAGE_BG_EOT -- and handing it to an array
       * store writes the wrong surface. This mirrors the single-target path.
       */
      struct hk_native_pbe_descriptor pbe_arguments[HK_MAX_RTS] = {0};
      for (unsigned rt = 0; rt < count; ++rt) {
         if (!(key.tib.layered ? views[rt]->planes[0].layered_eot_pbe_valid
                               : views[rt]->planes[0].eot_pbe_valid)) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FORMAT_NOT_SUPPORTED);
            return (struct hk_bg_eot){0};
         }
         pbe_arguments[rt] = key.tib.layered ? views[rt]->planes[0].layered_eot
                                             : views[rt]->planes[0].eot;
      }

      const uint64_t pbe_argument_va = hk_pool_upload(
         cmd, pbe_arguments, count * AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE, 32);
      if (!pbe_argument_va)
         return (struct hk_bg_eot){0};

      const struct agx3_g17p_rt0_eot_program_desc desc = {
         .program_va = reservation.va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .pbe_argument_va = pbe_argument_va,
         .imageblock_offset_B = key.tib._offset_B[0],
         .attachment_count = count,
         .imageblock_offset1_B = key.tib._offset_B[1],
         .imageblock_offset2_B = key.tib._offset_B[2],
         .imageblock_offset3_B = key.tib._offset_B[3],
         .imageblock_offset4_B = key.tib._offset_B[4],
         .imageblock_offset5_B = key.tib._offset_B[5],
         .imageblock_offset6_B = key.tib._offset_B[6],
         .imageblock_offset7_B = key.tib._offset_B[7],
         .mixed_mrt = count > 1,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_known
                             ? shader->info.agx3_staging_halfs
                             : 0,
         .imageblock = imageblock,
      };
      if (!agx3_g17p_rt0_eot_program_generate(dev->dev.chip, &desc,
                                               &eot_program)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
      program_bytes = eot_program.bytes;
      program_length = eot_program.length;
      resource_spec = AGX_IOTO_G17P_EOT_RESOURCE_SPEC;
   } else {
      struct agx3_g17p_rt0_bg_program_desc desc = {
         .program_va = reservation.va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_halfs,
         .clear_push_halfs = shader->info.push_count,
         .clear_start_halfs = 4 + 8 * reloads,
         .staging_known = shader->info.agx3_staging_known,
         .op = bg_op,
         .imageblock = imageblock,
      };

      if (clears) {
         static_assert(sizeof(atts[0]->clearValue.color) == 16, "fixed ABI");
         VkClearColorValue clear_values[HK_MAX_RTS];
         for (unsigned rt = reloads; rt < count; ++rt)
            clear_values[rt - reloads] = atts[rt]->clearValue.color;
         desc.clear_value_va = hk_pool_upload(
            cmd, clear_values, 16 * (count - reloads), 16);
         if (!desc.clear_value_va)
            return (struct hk_bg_eot){0};
      }

      if (reloads) {
         /* Target n reads native texture index 2n; the odd entries between
          * them stay zero. */
         static_assert(sizeof(struct hk_native_texture_descriptor) ==
                          AGX_TEXTURE_GEN9_LENGTH,
                       "native texture table stride");
         struct hk_native_texture_descriptor textures[2 * HK_MAX_RTS - 1] = {0};
         for (unsigned rt = 0; rt < reloads; ++rt)
            textures[2 * rt] = key.tib.layered
                                  ? views[rt]->planes[0].layered_background
                                  : views[rt]->planes[0].background;
         const uint64_t texture_va = hk_pool_upload(
            cmd, textures, (2 * reloads - 1) * AGX_TEXTURE_GEN9_LENGTH,
            AGX_TEXTURE_GEN9_LENGTH);
         const uint64_t sampler_va =
            dev->rodata.bo && dev->rodata.bo->va ? dev->rodata.bo->va->addr : 0;
         if (!texture_va || !sampler_va)
            return (struct hk_bg_eot){0};
         const uint64_t pointers[2] = {texture_va, sampler_va};
         const uint64_t table =
            hk_pool_upload(cmd, pointers, sizeof(pointers), 8);
         if (!table)
            return (struct hk_bg_eot){0};
         desc.texture_pointer_va = table;
         desc.sampler_pointer_va = table + sizeof(uint64_t);
      }

      if (!agx3_g17p_rt0_bg_program_generate(dev->dev.chip, &desc,
                                             &bg_program) ||
          !agx3_g17p_rt0_bg_resource_spec(dev->dev.chip, bg_op,
                                          &resource_spec)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
      program_bytes = bg_program.bytes;
      program_length = bg_program.length;
   }

   memcpy(reservation.map, program_bytes, program_length);
   const uint64_t published =
      hk_cmd_publish_ioto_program(cmd, &reservation, program_length);
   if (!published)
      return (struct hk_bg_eot){0};

   return (struct hk_bg_eot){
      .usc = published,
      .rsrc_spec = resource_spec,
   };
}

/* The native dual- and multi-target background and EOT programs carry one op
 * for every target: agx3_g17p_rt0_bg_program_generate() selects a single
 * RELOAD/CLEAR/CLEAR_RELOAD program, and the EOT builtin stores every target
 * it names. A layout whose live targets do not share one op -- an attachment
 * that is not stored beside one that is, or a clear beside a reload -- cannot
 * be expressed that way, which is what the multi/dual guards below refuse.
 *
 * The mixed builder can express it: it groups targets into LOAD, then CLEAR,
 * then STORE, drops the untouched ones, and carries each target's own tile
 * offset and PBE descriptor. Route there rather than refusing. */
static bool
hk_g17p_bg_eot_ops_uniform(const struct agx_bg_eot_key *key)
{
   bool seen = false;
   enum agx_bg_eot_op first = AGX_BG_EOT_NONE;

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->tib.logical_format); ++rt) {
      if (key->tib.logical_format[rt] == PIPE_FORMAT_NONE)
         continue;
      if (!seen) {
         first = key->op[rt];
         seen = true;
      } else if (key->op[rt] != first) {
         return false;
      }
   }
   return true;
}

/* Packed clear: a single-sample R8G8B8A8_UNORM RT0 clear is packed on the
 * CPU and the background program stores the word directly (COPY + LSTCOL
 * instead of two FLTNORMs + LSTCOL per pixel). Apple's background clear
 * converts per pixel (fltnorm, fltnorm, lstcol); this is one instruction per
 * pixel cheaper. */

/* The Vulkan float-to-UNORM conversion (round to nearest even, clamp, NaN to
 * zero) of one clear component. */
static uint32_t
hk_g17p_unorm8(float x)
{
   return x == x ? (uint32_t)_mesa_float_to_unorm(x, 8) : 0;
}

/* frg-codegen v4 BG/EOT device cache.
 *
 * The per-recording cache above dies with every vkResetCommandBuffer, so an
 * application that re-records each frame (the depth-2 benchmark, most games)
 * regenerates and republishes its background and end-of-tile launch programs
 * for every pass: about a third of the host-side recording cost of a light
 * pass (frg-codegen/v4/hostpath profile). For the single-attachment,
 * unspilled, non-resolving layouts the RT0 builder handles, the programs
 * depend only on the BG/EOT key, the attachment's clear value and ops, and the
 * descriptors of its image view -- all value-compared below -- plus
 * device-lifetime state (the precompiled BG/EOT shaders, rodata). Build those
 * programs and their argument blocks once into device-lifetime memory and
 * reuse them from any command buffer. Entries are immutable and never freed
 * before the device; the cache stops growing at a fixed bound. */
#define HK_G17P_BG_EOT_DEV_ENTRIES       256
#define HK_G17P_BG_EOT_DEV_CHUNK         (64 * 1024)
#define HK_G17P_BG_EOT_DEV_MAX_CHUNKS    8
#define HK_G17P_BG_EOT_DEV_PROGRAM_MAX   256
#define HK_G17P_BG_EOT_DEV_DATA_MAX      512

struct hk_g17p_bg_eot_dev_key {
   struct agx_tilebuffer_layout tib;
   uint32_t layers;
   uint32_t color_attachment_count;
   uint32_t incomplete_render_area;
   uint32_t view;
   uint32_t samples;
   uint32_t format;
   uint32_t load_op;
   uint32_t store_op;
   VkClearColorValue clear;
   struct hk_native_pbe_descriptor eot, layered_eot;
   struct hk_native_texture_descriptor background, layered_background;
   uint32_t pbe_valid;
};

struct hk_g17p_bg_eot_dev_entry {
   struct hk_g17p_bg_eot_dev_key key;
   uint64_t hash;
   struct hk_bg_eot program[4];
   uint8_t valid_variants;
};

struct hk_g17p_bg_eot_arena {
   struct agx_bo *bo;
   uint8_t *map;
   uint64_t base;
   uint32_t cursor, capacity;
};

struct hk_g17p_bg_eot_dev_cache {
   simple_mtx_t lock;
   struct hk_g17p_bg_eot_arena program, data;
   struct agx_bo *bos[2 * HK_G17P_BG_EOT_DEV_MAX_CHUNKS];
   unsigned bo_count, count;
   bool full;
   /* Open-addressed index: entry number + 1, or 0 when free. */
   uint16_t index[2 * HK_G17P_BG_EOT_DEV_ENTRIES];
   struct hk_g17p_bg_eot_dev_entry entries[HK_G17P_BG_EOT_DEV_ENTRIES];
};

void
hk_g17p_bg_eot_dev_cache_init(struct hk_device *dev)
{
   dev->g17p_bg_eot_dev_cache = NULL;
   if (!agx_native_graphics(&dev->dev))
      return;
   struct hk_g17p_bg_eot_dev_cache *dc = calloc(1, sizeof(*dc));
   if (!dc)
      return;
   simple_mtx_init(&dc->lock, mtx_plain);
   dev->g17p_bg_eot_dev_cache = dc;
}

void
hk_g17p_bg_eot_dev_cache_finish(struct hk_device *dev)
{
   struct hk_g17p_bg_eot_dev_cache *dc = dev->g17p_bg_eot_dev_cache;
   if (!dc)
      return;
   for (unsigned i = 0; i < dc->bo_count; ++i)
      agx_bo_unreference(&dev->dev, dc->bos[i]);
   simple_mtx_destroy(&dc->lock);
   free(dc);
   dev->g17p_bg_eot_dev_cache = NULL;
}

/* Make room for `need` bytes at `align` in an arena, opening a new chunk if
 * required. Programs live in the USC window exactly like the per-recording
 * Ioto arena; argument blocks in ordinary memory like hk_pool_upload(). */
static bool
hk_g17p_bg_eot_arena_room(struct hk_device *dev,
                          struct hk_g17p_bg_eot_dev_cache *dc,
                          struct hk_g17p_bg_eot_arena *a, bool exec,
                          uint32_t need, uint32_t align)
{
   if (a->map && align_uintptr(a->cursor, align) + need <= a->capacity)
      return true;
   if (dc->bo_count >= ARRAY_SIZE(dc->bos))
      return false;
   struct agx_bo *bo = agx_bo_create(
      &dev->dev, HK_G17P_BG_EOT_DEV_CHUNK,
      exec ? AGX_IOTO_PROGRAM_ALIGNMENT : 0,
      exec ? (AGX_BO_EXEC | AGX_BO_LOW_VA) : 0,
      exec ? "G17P BG/EOT device programs" : "G17P BG/EOT device arguments");
   if (!bo)
      return false;
   void *map = agx_bo_map(bo);
   if (!map || !bo->va) {
      agx_bo_unreference(&dev->dev, bo);
      return false;
   }
   dc->bos[dc->bo_count++] = bo;
   a->bo = bo;
   a->map = map;
   a->base = bo->va->addr;
   a->cursor = 0;
   a->capacity = bo->size > UINT32_MAX ? UINT32_MAX : (uint32_t)bo->size;
   return align_uintptr(0, align) + need <= a->capacity;
}

static struct agx_ptr
hk_g17p_bg_eot_arena_take(struct hk_g17p_bg_eot_arena *a, uint32_t size,
                          uint32_t align)
{
   const uint32_t offset = (uint32_t)align_uintptr(a->cursor, align);
   assert(a->map && offset + size <= a->capacity);
   a->cursor = offset + size;
   return (struct agx_ptr){.cpu = a->map + offset, .gpu = a->base + offset};
}

/* Allocation sites of the RT0 builder: the per-recording pool, or the
 * device cache when one is being filled (its room was reserved up front). */
static uint64_t
hk_g17p_bg_eot_upload(struct hk_cmd_buffer *cmd,
                      struct hk_g17p_bg_eot_dev_cache *dc, const void *data,
                      uint32_t size, uint32_t alignment)
{
   if (!dc)
      return hk_pool_upload(cmd, data, size, alignment);
   struct agx_ptr T = hk_g17p_bg_eot_arena_take(&dc->data, size, alignment);
   memcpy(T.cpu, data, size);
   return T.gpu;
}

static bool
hk_g17p_bg_eot_reserve(struct hk_cmd_buffer *cmd,
                       struct hk_g17p_bg_eot_dev_cache *dc, size_t capacity,
                       const char *label, struct hk_ioto_reservation *r)
{
   if (!dc)
      return hk_cmd_reserve_ioto_program(cmd, capacity, label, r);
   assert(capacity <= HK_G17P_BG_EOT_DEV_PROGRAM_MAX);
   struct agx_ptr T = hk_g17p_bg_eot_arena_take(
      &dc->program, (uint32_t)capacity, AGX_IOTO_PROGRAM_ALIGNMENT);
   *r = (struct hk_ioto_reservation){.map = T.cpu, .va = T.gpu};
   return true;
}

static uint64_t
hk_g17p_bg_eot_publish(struct hk_cmd_buffer *cmd,
                       struct hk_g17p_bg_eot_dev_cache *dc,
                       const struct hk_ioto_reservation *r, size_t length)
{
   if (!dc)
      return hk_cmd_publish_ioto_program(cmd, r, length);
   struct hk_g17p_bg_eot_arena *a = &dc->program;
   assert(r->va >= a->base && length > 0);
   const uint64_t offset = r->va - a->base;
   /* Give back the over-reservation, as hk_cmd_publish_ioto_program does. */
   a->cursor = (uint32_t)(offset + length);
   agx_bo_assert_range(a->bo, offset, length);
   return r->va;
}

/* Fill the value key of a render pass whose BG/EOT programs the RT0 builder
 * produces from the key, the single attachment's ops and clear value, and its
 * image view's descriptors alone; return false for anything else (MRT,
 * spills, mixed layouts, resolves), which keeps the per-recording path. */
static bool
hk_g17p_bg_eot_dev_key_init(struct hk_cmd_buffer *cmd,
                            const VkRenderingInfo *info,
                            bool incomplete_render_area,
                            struct hk_g17p_bg_eot_dev_key *dk)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   if (!dev->g17p_bg_eot_dev_cache || info->colorAttachmentCount > 1 ||
       agx_tilebuffer_spills(&render->tilebuffer) ||
       agx_g17p_classify_mrt(&render->tilebuffer) == AGX_G17P_MRT_MIXED)
      return false;

   memset(dk, 0, sizeof(*dk));
   memcpy(&dk->tib, &render->tilebuffer, sizeof(dk->tib));
   dk->layers = render->cr.layers;
   dk->color_attachment_count = info->colorAttachmentCount;
   dk->incomplete_render_area = incomplete_render_area;
   if (info->colorAttachmentCount == 1 &&
       info->pColorAttachments[0].imageView != VK_NULL_HANDLE) {
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[0];
      const struct hk_image_view *iview = render->color_att[0].iview;
      if (!iview || att->resolveMode != VK_RESOLVE_MODE_NONE ||
          att->resolveImageView != VK_NULL_HANDLE)
         return false;
      dk->view = 1;
      dk->samples = iview->vk.image->samples;
      dk->format = iview->vk.format;
      dk->load_op = att->loadOp;
      dk->store_op = att->storeOp;
      if (att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
         dk->clear = att->clearValue.color;
      dk->eot = iview->planes[0].eot;
      dk->layered_eot = iview->planes[0].layered_eot;
      dk->background = iview->planes[0].background;
      dk->layered_background = iview->planes[0].layered_background;
      dk->pbe_valid = (iview->planes[0].eot_pbe_valid ? 1 : 0) |
                      (iview->planes[0].layered_eot_pbe_valid ? 2 : 0);
   }
   return true;
}

static uint64_t
hk_g17p_bg_eot_dev_hash(const struct hk_g17p_bg_eot_dev_key *dk)
{
   static_assert(sizeof(*dk) % sizeof(uint64_t) == 0,
                 "word-addressable BG/EOT device key");
   const uint8_t *bytes = (const uint8_t *)dk;
   uint64_t hash = UINT64_C(0x9e3779b97f4a7c15);
   for (unsigned i = 0; i < sizeof(*dk) / sizeof(uint64_t); ++i) {
      uint64_t word;
      memcpy(&word, bytes + i * sizeof(word), sizeof(word));
      hash ^= word + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) +
              (hash >> 2);
   }
   return hash;
}

static struct hk_g17p_bg_eot_dev_entry *
hk_g17p_bg_eot_dev_find(struct hk_g17p_bg_eot_dev_cache *dc,
                        const struct hk_g17p_bg_eot_dev_key *dk,
                        uint64_t hash)
{
   const unsigned mask = ARRAY_SIZE(dc->index) - 1;
   static_assert((ARRAY_SIZE(((struct hk_g17p_bg_eot_dev_cache *)0)->index) &
                  (ARRAY_SIZE(((struct hk_g17p_bg_eot_dev_cache *)0)->index) -
                   1)) == 0,
                 "power-of-two index");
   for (unsigned i = hash & mask;; i = (i + 1) & mask) {
      if (!dc->index[i])
         return NULL;
      struct hk_g17p_bg_eot_dev_entry *e = &dc->entries[dc->index[i] - 1];
      if (e->hash == hash && !memcmp(&e->key, dk, sizeof(*dk)))
         return e;
   }
}

/* The index is twice the entry bound, so a free slot always exists. */
static struct hk_g17p_bg_eot_dev_entry *
hk_g17p_bg_eot_dev_insert(struct hk_g17p_bg_eot_dev_cache *dc,
                          const struct hk_g17p_bg_eot_dev_key *dk,
                          uint64_t hash)
{
   if (dc->count >= HK_G17P_BG_EOT_DEV_ENTRIES)
      return NULL;
   const unsigned mask = ARRAY_SIZE(dc->index) - 1;
   unsigned i = hash & mask;
   while (dc->index[i])
      i = (i + 1) & mask;
   struct hk_g17p_bg_eot_dev_entry *e = &dc->entries[dc->count];
   memcpy(&e->key, dk, sizeof(*dk));
   e->hash = hash;
   e->valid_variants = 0;
   memset(e->program, 0, sizeof(e->program));
   dc->index[i] = (uint16_t)(++dc->count);
   return e;
}

static struct hk_bg_eot
hk_build_g17p_bg_eot(struct hk_cmd_buffer *cmd,
                     struct agx_bg_eot_key *key,
                     const VkRenderingInfo *info, bool store,
                     bool partial_render,
                     struct hk_g17p_bg_eot_dev_cache *dc, bool *in_device)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (in_device)
      *in_device = false;
   /* Classification compacts holes internally, so it is invariant under the
    * explicit key/view compaction below. Keep the result for all subsequent
    * BG/EOT routing instead of rescanning the eight targets twice. */
   const enum agx_g17p_mrt_layout mrt = agx_g17p_classify_mrt(&key->tib);
   if (agx_tilebuffer_spills(&key->tib) ||
       mrt == AGX_G17P_MRT_MIXED ||
       !hk_g17p_bg_eot_ops_uniform(key)) {
      /* hk_g17p_bg_eot_dev_key_init() admits none of these layouts. */
      assert(!dc);
      if (dc)
         return (struct hk_bg_eot){0};
      return hk_build_g17p_mixed_bg_eot(cmd, key, info, store);
   }

   /* Vulkan keeps unused colour attachments as holes in the logical array.
    * Native BG/EOT programs number only live attachments. Compact the key and
    * matching image views together while preserving each physical tile
    * offset, then leave the render state itself in Vulkan location order. */
   struct agx_bg_eot_key compact_key = *key;
   uint8_t attachment_map[HK_MAX_RTS];
   const unsigned active_count = agx_tilebuffer_compact(
      &key->tib, &compact_key.tib, attachment_map);
   memset(compact_key.op, 0, sizeof(compact_key.op));
   struct hk_attachment color_att[HK_MAX_RTS] = {0};
   VkRenderingAttachmentInfo attachments[HK_MAX_RTS] = {0};
   for (unsigned rt = 0; rt < active_count; ++rt) {
      const unsigned logical = attachment_map[rt];
      compact_key.op[rt] = key->op[logical];
      color_att[rt] = cmd->state.gfx.render.color_att[logical];
      attachments[rt] = info->pColorAttachments[logical];
   }
   if (active_count == 0 && compact_key.tib.nr_samples == 0)
      agx_tilebuffer_set_samples(&compact_key.tib, 1);
   VkRenderingInfo compact_info = *info;
   compact_info.colorAttachmentCount = active_count;
   compact_info.pColorAttachments = attachments;
   key = &compact_key;
   info = &compact_info;
   if (!store && !partial_render && active_count == 1) {
      const VkRenderingAttachmentInfo *att = &attachments[0];
      const struct hk_rendering_state *render = &cmd->state.gfx.render;
      const bool partial_area = render->area.offset.x || render->area.offset.y ||
         render->area.extent.width < render->cr.width ||
         render->area.extent.height < render->cr.height;
      /* The EOT resolve writes full tiles. Merge the destination outside the
       * API rectangle before that store, retaining native MSAA resolution. */
      key->preserve_resolve_outside = partial_area &&
         hk_agx3_g17p_direct_color_resolve_supported(
            &key->tib, active_count,
            att->resolveImageView != VK_NULL_HANDLE,
            att->resolveMode == VK_RESOLVE_MODE_AVERAGE_BIT,
            att->storeOp == VK_ATTACHMENT_STORE_OP_STORE);
   }
   if (!hk_agx3_g17p_rt0_tile_layout_supported(&key->tib)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }

   /* As in the generic BG/EOT path, compile an empty attachmentless program
    * as 1x. This is only the local shader key: the render state's sample count
    * remains deferred until a draw supplies it.
    */
   if (!key->tib.nr_samples)
      key->tib.nr_samples = 1;

   /* Decide from the compacted attachment and its matching view. Vulkan may
    * leave holes before the sole live color attachment; selecting from
    * logical slot0 and colorAttachmentCount before compaction loses its
    * STORE requirement and selects a single-sample EOT for an MS surface.
    * Keep samples for partial renders, explicit STORE, and resolves that
    * execute after the pass. Final direct resolves remain single-sampled. */
   if (store && active_count == 1) {
      const VkRenderingAttachmentInfo *att = &attachments[0];
      const struct hk_image_view *iview = color_att[0].iview;
      const bool store_source = att->storeOp == VK_ATTACHMENT_STORE_OP_STORE;
      const bool has_resolve = att->resolveImageView != VK_NULL_HANDLE &&
                               att->resolveMode != VK_RESOLVE_MODE_NONE;
      const bool resolved_after_pass = has_resolve &&
         !hk_agx3_g17p_direct_color_resolve_supported(
            &key->tib, active_count, has_resolve,
            att->resolveMode == VK_RESOLVE_MODE_AVERAGE_BIT, store_source);
      key->multisample_store = hk_agx3_g17p_multisample_store_supported(
         &key->tib, active_count,
         partial_render || store_source || resolved_after_pass,
         iview ? iview->vk.image->samples : 0);
   }

   const bool wboit_mrt = mrt == AGX_G17P_MRT_WBOIT;
   const bool rgba8_mrt = mrt == AGX_G17P_MRT_RGBA8;
   const bool r11g11b10_mrt = mrt == AGX_G17P_MRT_R11G11B10;
   const bool dual_mrt = wboit_mrt || (rgba8_mrt && active_count == 2) ||
                         r11g11b10_mrt;
   const bool multi_mrt = mrt == AGX_G17P_MRT_RGBA32 ||
                          mrt == AGX_G17P_MRT_R32_QUAD ||
                          (rgba8_mrt && active_count > 2);
   unsigned multi_count = 0;
   if (multi_mrt) {
      while (multi_count < HK_MAX_RTS &&
             key->tib.logical_format[multi_count] != PIPE_FORMAT_NONE)
         multi_count++;
   }

   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      const enum agx_bg_eot_op op = key->op[rt];
      const bool rt0_supported = store
                                    ? (op == AGX_EOT_STORE || op == AGX_BG_EOT_NONE)
                                    : (op == AGX_BG_EOT_NONE ||
                                       op == AGX_BG_CLEAR || op == AGX_BG_LOAD);
      const bool dual_mrt_rt1 =
         rt == 1 && dual_mrt &&
         (store ? op == AGX_EOT_STORE
                : (op == AGX_BG_CLEAR || op == AGX_BG_LOAD));
      const bool multi_mrt_rt = rt < multi_count && multi_mrt &&
         (store ? op == AGX_EOT_STORE : (op == AGX_BG_CLEAR || op == AGX_BG_LOAD));
      if ((rt == 0 && !rt0_supported) ||
          (rt != 0 && op != AGX_BG_EOT_NONE && !dual_mrt_rt1 && !multi_mrt_rt)) {
         /* The refusal is about WHICH target carries WHICH operation in a
          * layout the G17P background/EOT programs cannot express, and the
          * deferred VK_ERROR_FEATURE_NOT_PRESENT surfaces much later at
          * vkEndCommandBuffer with none of that visible. Name it. */
         mesa_loge("G17P %s program declined: rt%u op=%u, ops=%u%u%u%u%u%u%u%u "
                   "mrt=%u dual=%u multi=%u/%u samples=%u layered=%u "
                   "sample_size=%uB",
                   store ? "EOT" : "background", rt, (unsigned)op,
                   (unsigned)key->op[0], (unsigned)key->op[1],
                   (unsigned)key->op[2], (unsigned)key->op[3],
                   (unsigned)key->op[4], (unsigned)key->op[5],
                   (unsigned)key->op[6], (unsigned)key->op[7], (unsigned)mrt,
                   dual_mrt, multi_mrt, multi_count, key->tib.nr_samples,
                   key->tib.layered, key->tib.sample_size_B);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
   }

   const enum agx_bg_eot_op op = key->op[0];
   const bool dual_mrt_active = dual_mrt && key->op[1] != AGX_BG_EOT_NONE;
   const bool multi_mrt_active = multi_mrt &&
      (key->op[0] != AGX_BG_EOT_NONE || key->op[1] != AGX_BG_EOT_NONE ||
       key->op[2] != AGX_BG_EOT_NONE || key->op[3] != AGX_BG_EOT_NONE);
   const unsigned attachment_count =
      multi_mrt_active ? multi_count : dual_mrt_active ? 2 : 1;
   if (multi_mrt_active) {
      for (unsigned rt = 0; rt < attachment_count; ++rt) {
         if (info->colorAttachmentCount < attachment_count ||
             key->op[rt] != op || op == AGX_BG_EOT_NONE) {
            /* DIAGNOSTIC: name the refusal the way the rt0 guard above does.
             * These two guards refuse silently, which is why this bucket had
             * no driver-side evidence at all. */
            mesa_loge("G17P %s multi-MRT guard declined: rt%u op=%u op0=%u "
                      "ops=%u%u%u%u%u%u%u%u count=%u attachments=%u mrt=%u "
                      "samples=%u sample_size=%uB",
                      store ? "EOT" : "background", rt, (unsigned)key->op[rt],
                      (unsigned)op, (unsigned)key->op[0], (unsigned)key->op[1],
                      (unsigned)key->op[2], (unsigned)key->op[3],
                      (unsigned)key->op[4], (unsigned)key->op[5],
                      (unsigned)key->op[6], (unsigned)key->op[7],
                      attachment_count, info->colorAttachmentCount,
                      (unsigned)mrt, key->tib.nr_samples,
                      key->tib.sample_size_B);
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return (struct hk_bg_eot){0};
         }
      }
   }
   if (dual_mrt_active &&
       (key->op[1] != op || info->colorAttachmentCount < 2)) {
      mesa_loge("G17P %s dual-MRT guard declined: op0=%u op1=%u "
                "attachments=%u mrt=%u samples=%u sample_size=%uB",
                store ? "EOT" : "background", (unsigned)op,
                (unsigned)key->op[1], info->colorAttachmentCount,
                (unsigned)mrt, key->tib.nr_samples, key->tib.sample_size_B);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }
   key->g17p_packed_clear =
      !store && op == AGX_BG_CLEAR && !dual_mrt_active && !multi_mrt_active &&
      agx_g17p_packed_clear_ok(&key->tib);
   if (!store && op == AGX_BG_CLEAR)
      key->reserved_preamble =
         multi_mrt_active ? 4 + 8 * attachment_count
         : rgba8_mrt && dual_mrt_active
            ? AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS
         : r11g11b10_mrt && dual_mrt_active
            ? AGX3_G17P_R11G11B10_MRT_CLEAR_PUSH_HALFS
         : wboit_mrt && dual_mrt_active
            ? AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS
            : AGX3_G17P_RT0_CLEAR_PUSH_HALFS;

   struct agx_bg_eot_shader *shader =
      agx_get_bg_eot_shader(&dev->bg_eot, key);
   const bool shader_resources_match =
      shader &&
      (store || (op == AGX_BG_EOT_NONE
                    ? shader->info.push_count == 0 && !shader->info.uses_txf
                 : agx3_g17p_rt0_bg_shader_resources_match(
                      op == AGX_BG_CLEAR ? AGX3_G17P_RT0_BG_CLEAR
                                         : AGX3_G17P_RT0_BG_RELOAD,
                      shader->info.push_count, shader->info.uses_txf)));
   if (!hk_g17p_bg_eot_shader_supported(shader) || !shader_resources_match) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }

   size_t program_capacity;
   const char *program_label;
   if (op == AGX_BG_EOT_NONE) {
      program_capacity = AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE;
      program_label = "G17P empty tile Ioto program";
   } else if (store) {
      program_capacity = multi_mrt_active ? AGX_IOTO_G17P_QUAD_EOT_PROGRAM_MAX_SIZE
                                        : AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE;
      program_label = "G17P RT0 EOT Ioto program";
   } else if (op == AGX_BG_CLEAR) {
      program_capacity =
         multi_mrt_active ? AGX_IOTO_G17P_QUAD_CLEAR_PROGRAM_MAX_SIZE
         : rgba8_mrt && dual_mrt_active
            ? AGX_IOTO_G17P_RGBA8_MRT_CLEAR_PROGRAM_SIZE
         : r11g11b10_mrt && dual_mrt_active
            ? AGX_IOTO_G17P_RGBA8_MRT_CLEAR_PROGRAM_SIZE
         : wboit_mrt && dual_mrt_active
            ? AGX_IOTO_G17P_WBOIT_MRT_CLEAR_PROGRAM_SIZE
            : AGX_IOTO_G17P_RT0_CLEAR_PROGRAM_SIZE;
      program_label = "G17P RT0 clear Ioto program";
   } else {
      program_capacity = AGX_IOTO_G17P_RT0_RELOAD_PROGRAM_MAX_SIZE;
      program_label = "G17P RT0 reload Ioto program";
   }

   struct hk_ioto_reservation program_reservation;
   if (!hk_g17p_bg_eot_reserve(cmd, dc, program_capacity, program_label,
                               &program_reservation))
      return (struct hk_bg_eot){0};

   void *program_map = program_reservation.map;
   const uint64_t program_va = program_reservation.va;

   size_t program_length;
   uint64_t resource_spec;
   if (store && op == AGX_EOT_STORE) {
      struct hk_image_view *iview = color_att[0].iview;
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[0];
      /* A direct resolve is the pass's output, so only the pass's final
       * end-of-tile program may perform one. A partial render is a mid-frame
       * flush whose tile is reloaded by the draws that follow it, and a
       * resolve is not reloadable: it averages the samples away. The
       * mid-frame store therefore writes the multisample surface unresolved,
       * which is what key->multisample_store selects below.
       */
      const bool direct_resolve =
         !partial_render &&
         hk_agx3_g17p_direct_color_resolve_supported(
            &key->tib, info->colorAttachmentCount,
            att->resolveImageView != VK_NULL_HANDLE,
            att->resolveMode == VK_RESOLVE_MODE_AVERAGE_BIT,
            att->storeOp == VK_ATTACHMENT_STORE_OP_STORE);
      struct hk_image_view *target = direct_resolve
         ? color_att[0].resolve_iview : iview;
      struct hk_image_view *iview1 = dual_mrt_active
         ? color_att[1].iview : NULL;
      const bool target_pbe_valid =
         target && (key->tib.layered
                       ? target->planes[0].layered_eot_pbe_valid
                       : target->planes[0].eot_pbe_valid);
      const bool target1_pbe_valid =
         iview1 && (key->tib.layered
                       ? iview1->planes[0].layered_eot_pbe_valid
                       : iview1->planes[0].eot_pbe_valid);
      /* A G17P end-of-tile program writes either an ordinary single-sample
       * PBE target -- resolving the tile on the way out -- or, when the key
       * asked for it, the unresolved multisample surface through the separate
       * multisample IMGWBLK builtin. Those are different programs, so the
       * target's sample count has to agree with the one the key selected.
       *
       * Without this, storeOp = STORE on a multisample attachment picks
       * target = iview (direct resolve requires storeOp != STORE), skips the
       * samples != 1 check below because that is guarded by direct_resolve,
       * and writes single-sample data into a multisample surface with no
       * diagnostic. Refuse that combination instead of storing wrong pixels.
       */
      const bool multisample_store =
         agx_native_graphics(&dev->dev) && key->multisample_store;
      if (!iview || !target ||
          (agx_native_graphics(&dev->dev) && !direct_resolve &&
           !multisample_store && target->vk.image->samples != 1) ||
          (multisample_store &&
           (direct_resolve || target != iview ||
            target->vk.image->samples != key->tib.nr_samples)) ||
          (direct_resolve &&
           (target->vk.image->samples != 1 ||
            target->vk.format != iview->vk.format)) ||
          !target_pbe_valid ||
          info->pColorAttachments[0].imageView == VK_NULL_HANDLE ||
          (dual_mrt_active &&
           (!iview1 || !target1_pbe_valid ||
            info->pColorAttachments[1].imageView == VK_NULL_HANDLE))) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FORMAT_NOT_SUPPORTED);
         return (struct hk_bg_eot){0};
      }

      const unsigned descriptor_generation =
         agx_chip_descriptor_generation(dev->dev.chip);
      if (hk_pbe_descriptor_size(descriptor_generation) !=
          AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
      static_assert(sizeof(struct hk_native_pbe_descriptor) ==
                       AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE,
                    "native PBE table stride");

      const struct hk_native_pbe_descriptor target_pbe =
         key->tib.layered ? target->planes[0].layered_eot
                          : target->planes[0].eot;
      struct hk_native_pbe_descriptor target1_pbe = {0};
      if (dual_mrt_active) {
         target1_pbe = key->tib.layered ? iview1->planes[0].layered_eot
                                        : iview1->planes[0].eot;
      }
      struct hk_native_pbe_descriptor pbe_arguments[4] = {target_pbe};
      /* The native resolve averages in linear space when its PBE requests
       * the sRGB transfer. Ordinary stores already have encoded tile bytes
       * and retain the unmodified end-of-tile descriptor. */
      if (direct_resolve && util_format_is_srgb(key->tib.logical_format[0]) &&
          !(color_att[0].flags &
            VK_RENDERING_ATTACHMENT_RESOLVE_SKIP_TRANSFER_FUNCTION_BIT_KHR))
         ((uint8_t *)&pbe_arguments[0])[15] |= 0x20;

      if (dual_mrt_active)
         pbe_arguments[1] = target1_pbe;
      if (multi_mrt_active) {
         for (unsigned rt = 1; rt < attachment_count; ++rt) {
            struct hk_image_view *view = color_att[rt].iview;
            if (!view || !view->planes[0].eot_pbe_valid ||
                view->vk.image->samples != 1 ||
                info->pColorAttachments[rt].imageView == VK_NULL_HANDLE) {
               vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FORMAT_NOT_SUPPORTED);
               return (struct hk_bg_eot){0};
            }
            pbe_arguments[rt] = view->planes[0].eot;
         }
      }
      const size_t pbe_argument_size =
         multi_mrt_active ? attachment_count * AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE
         : dual_mrt_active ? AGX_IOTO_G17P_RGBA8_MRT_PBE_ARGUMENT_SIZE
                         : AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE;
      const uint64_t pbe_argument_va =
         hk_g17p_bg_eot_upload(cmd, dc, pbe_arguments, pbe_argument_size, 32);
      if (!pbe_argument_va)
         return (struct hk_bg_eot){0};

      const struct agx3_g17p_rt0_eot_program_desc desc = {
         .program_va = program_va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .pbe_argument_va = pbe_argument_va,
         .imageblock_offset_B = agx_tilebuffer_offset_B(&key->tib, 0),
         .attachment_count = attachment_count,
         /* Layered MRT uses the shared array builtin, like Gallium. */
         .mixed_mrt = attachment_count >= 2,
         .imageblock_offset1_B =
            attachment_count > 1 ? agx_tilebuffer_offset_B(&key->tib, 1) : 0,
         .imageblock_offset2_B = attachment_count > 2 ? agx_tilebuffer_offset_B(&key->tib, 2) : 0,
         .imageblock_offset3_B = attachment_count > 3 ? agx_tilebuffer_offset_B(&key->tib, 3) : 0,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_known
                            ? shader->info.agx3_staging_halfs : 0,
         .imageblock = {
            .sample_size_B = key->tib.sample_size_B,
            .nr_samples = key->tib.nr_samples,
            .tile_size_px = key->tib.tile_size,
         },
      };
      struct agx3_g17p_rt0_eot_program program;
      if (!agx3_g17p_rt0_eot_program_generate(dev->dev.chip, &desc,
                                               &program)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      memcpy(program_map, program.bytes, program.length);
      program_length = program.length;
      resource_spec = AGX_IOTO_G17P_EOT_RESOURCE_SPEC;

      if (direct_resolve)
         cmd->state.gfx.render.g17p_direct_color_resolve = true;
   } else if (op == AGX_BG_EOT_NONE) {
      const struct hk_agx3_g17p_graphics_program_desc desc = {
         .program_va = program_va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .compiler_d_valid = shader->info.agx3_staging_known,
         .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
         .fragment = true,
         .imageblock = {
            .sample_size_B = key->tib.sample_size_B,
            .nr_samples = key->tib.nr_samples,
            .tile_size_px = key->tib.tile_size,
         },
      };
      struct hk_agx3_g17p_graphics_program program;
      if (!hk_agx3_g17p_basic_graphics_program_generate(dev->dev.chip, &desc,
                                                        &program)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      memcpy(program_map, program.bytes, program.length);
      program_length = program.length;
      if (!agx3_g17p_rt0_bg_resource_spec(
             dev->dev.chip, AGX3_G17P_RT0_BG_NONE, &resource_spec)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
   } else {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[0];
      struct hk_image_view *iview = color_att[0].iview;
      const VkRenderingAttachmentInfo *att_info1 =
         dual_mrt_active ? &info->pColorAttachments[1] : NULL;
      struct hk_image_view *iview1 =
         dual_mrt_active ? color_att[1].iview : NULL;
      if (!iview || att_info->imageView == VK_NULL_HANDLE ||
          (dual_mrt_active &&
           (!iview1 || att_info1->imageView == VK_NULL_HANDLE))) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FORMAT_NOT_SUPPORTED);
         return (struct hk_bg_eot){0};
      }
      if (multi_mrt_active) {
         for (unsigned rt = 1; rt < attachment_count; ++rt) {
            struct hk_image_view *view = color_att[rt].iview;
            if (!view || view->vk.image->samples != 1 ||
                info->pColorAttachments[rt].imageView == VK_NULL_HANDLE) {
               vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FORMAT_NOT_SUPPORTED);
               return (struct hk_bg_eot){0};
            }
         }
      }

      const unsigned descriptor_generation =
         agx_chip_descriptor_generation(dev->dev.chip);
      if (hk_texture_descriptor_size(descriptor_generation) != 32 ||
          agx_chip_sampler_heap_element_size(dev->dev.chip) != 32) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      struct agx3_g17p_rt0_bg_program_desc desc = {
         .program_va = program_va,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_halfs,
         .clear_push_halfs = shader->info.push_count,
         .staging_known = shader->info.agx3_staging_known,
         .op = op == AGX_BG_CLEAR ? AGX3_G17P_RT0_BG_CLEAR
                                  : AGX3_G17P_RT0_BG_RELOAD,
         .imageblock = {
            .sample_size_B = key->tib.sample_size_B,
            .nr_samples = key->tib.nr_samples,
            .tile_size_px = key->tib.tile_size,
         },
      };

      if (op == AGX_BG_CLEAR) {
         static_assert(sizeof(att_info->clearValue.color) == 16, "fixed ABI");
         VkClearColorValue clear_values[4] = {
            att_info->clearValue.color,
         };
         if (dual_mrt_active)
            clear_values[1] = att_info1->clearValue.color;
         if (multi_mrt_active) {
            for (unsigned rt = 1; rt < attachment_count; ++rt)
               clear_values[rt] = info->pColorAttachments[rt].clearValue.color;
         }
         if (key->g17p_packed_clear) {
            /* The background program reads one packed word at uniform 4. */
            const float *c = att_info->clearValue.color.float32;
            const uint32_t packed = hk_g17p_unorm8(c[0]) |
                                    (hk_g17p_unorm8(c[1]) << 8) |
                                    (hk_g17p_unorm8(c[2]) << 16) |
                                    (hk_g17p_unorm8(c[3]) << 24);
            memset(&clear_values[0], 0, sizeof(clear_values[0]));
            clear_values[0].uint32[0] = packed;
         }
         const size_t clear_size =
            multi_mrt_active ? 16 * attachment_count
            : rgba8_mrt && dual_mrt_active ? 32
            : r11g11b10_mrt && dual_mrt_active ? 28
            : wboit_mrt && dual_mrt_active ? 20 : 16;
         desc.clear_value_va =
            hk_g17p_bg_eot_upload(cmd, dc, clear_values, clear_size, 16);
         if (!desc.clear_value_va)
            return (struct hk_bg_eot){0};
      } else {
         /* Native texture-table entries occupy aligned 32-byte slots. RT1 is
          * native texture index2, so the exact MRT table retains index1 as a
          * zero gap. */
         static_assert(sizeof(struct hk_native_texture_descriptor) ==
                          AGX_TEXTURE_GEN9_LENGTH,
                       "native texture table stride");
         struct hk_native_texture_descriptor textures[7] = {
            key->tib.layered ? iview->planes[0].layered_background
                             : iview->planes[0].background,
         };
         if (key->preserve_resolve_outside) {
            const struct hk_image_view *resolve = color_att[0].resolve_iview;
            textures[2] = key->tib.layered ? resolve->planes[0].layered_background
                                          : resolve->planes[0].background;
            const VkRect2D area = cmd->state.gfx.render.area;
            const uint32_t bounds[4] = {
               area.offset.x, area.offset.y,
               area.offset.x + area.extent.width,
               area.offset.y + area.extent.height,
            };
            desc.resolve_bounds_va = hk_g17p_bg_eot_upload(
               cmd, dc, bounds, sizeof(bounds), 16);
            if (!desc.resolve_bounds_va)
               return (struct hk_bg_eot){0};
         }
         if (dual_mrt_active)
            textures[2] = key->tib.layered
                             ? iview1->planes[0].layered_background
                             : iview1->planes[0].background;
         if (multi_mrt_active) {
            for (unsigned rt = 1; rt < attachment_count; ++rt)
               textures[2 * rt] = color_att[rt].iview->planes[0].background;
         }
         const size_t texture_size = key->preserve_resolve_outside
            ? 3 * AGX_TEXTURE_GEN9_LENGTH : multi_mrt_active
            ? (2 * attachment_count - 1) * AGX_TEXTURE_GEN9_LENGTH : dual_mrt_active
            ? 3 * AGX_TEXTURE_GEN9_LENGTH : AGX_TEXTURE_GEN9_LENGTH;
         const uint64_t texture_va =
            hk_g17p_bg_eot_upload(cmd, dc, textures, texture_size,
                           AGX_TEXTURE_GEN9_LENGTH);
         const uint64_t sampler_va =
            dev->rodata.bo && dev->rodata.bo->va ? dev->rodata.bo->va->addr : 0;
         if (!texture_va || !sampler_va)
            return (struct hk_bg_eot){0};
         const uint64_t pointers[2] = {texture_va, sampler_va};
         const uint64_t table = hk_g17p_bg_eot_upload(cmd, dc, pointers, sizeof(pointers), 8);
         if (!table)
            return (struct hk_bg_eot){0};
         desc.texture_pointer_va = table;
         desc.sampler_pointer_va = table + sizeof(uint64_t);
      }

      struct agx3_g17p_rt0_bg_program program;
      if (!agx3_g17p_rt0_bg_program_generate(dev->dev.chip, &desc,
                                             &program)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      memcpy(program_map, program.bytes, program.length);
      program_length = program.length;
      if (!agx3_g17p_rt0_bg_resource_spec(dev->dev.chip, desc.op,
                                          &resource_spec)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }
   }

   const uint64_t published =
      hk_g17p_bg_eot_publish(cmd, dc, &program_reservation,
                             program_length);
   if (!published)
      return (struct hk_bg_eot){0};

   if (in_device)
      *in_device = dc != NULL;
   return (struct hk_bg_eot){
      .usc = published,
      .rsrc_spec = resource_spec,
   };
}

struct hk_g17p_bg_eot_cache_key {
   /* 8-byte aligned so the size is a whole number of words on 32-bit x86
    * too, where uint64_t members only need 4-byte alignment.
    */
   _Alignas(8) struct agx_tilebuffer_layout tib;
   struct {
      uint64_t image_view;
      uint64_t resolve_image_view;
      VkClearColorValue clear;
      uint32_t load_op;
      uint32_t store_op;
      uint32_t resolve_mode;
      uint32_t flags;
   } attachments[HK_MAX_RTS];
   uint32_t layers;
   uint8_t color_attachment_count;
   bool incomplete_render_area;
};

static uint64_t
hk_g17p_bg_eot_cache_fingerprint(
   const struct hk_g17p_bg_eot_cache_key *key)
{
   static_assert(sizeof(*key) % sizeof(uint64_t) == 0,
                 "word-addressable BG/EOT cache key");
   const uint8_t *bytes = (const uint8_t *)key;
   uint64_t hash = UINT64_C(0x9e3779b97f4a7c15);
   for (unsigned i = 0; i < sizeof(*key) / sizeof(uint64_t); ++i) {
      uint64_t word;
      memcpy(&word, bytes + i * sizeof(word), sizeof(word));
      hash ^= word + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) +
              (hash >> 2);
   }
   return hash;
}

static struct hk_g17p_bg_eot_cache_entry *
hk_g17p_bg_eot_cache_get(struct hk_cmd_buffer *cmd,
                         const VkRenderingInfo *info,
                         bool incomplete_render_area)
{
   static_assert(sizeof(struct hk_g17p_bg_eot_cache_key) <=
                    HK_G17P_BG_EOT_CACHE_KEY_MAX,
                 "G17P BG/EOT cache key storage");

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!agx_native_graphics(&dev->dev))
      return NULL;

   struct hk_g17p_bg_eot_cache_key cache_key = {
      .tib = cmd->state.gfx.render.tilebuffer,
      .layers = cmd->state.gfx.render.cr.layers,
      .color_attachment_count = info->colorAttachmentCount,
      .incomplete_render_area = incomplete_render_area,
   };
   for (unsigned i = 0; i < info->colorAttachmentCount; ++i) {
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[i];
      cache_key.attachments[i].image_view = (uintptr_t)att->imageView;
      cache_key.attachments[i].load_op = att->loadOp;
      cache_key.attachments[i].store_op = att->storeOp;
      cache_key.attachments[i].flags = vk_get_rendering_attachment_flags(att);
      if (att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
         cache_key.attachments[i].clear = att->clearValue.color;
      if (att->resolveMode != VK_RESOLVE_MODE_NONE) {
         cache_key.attachments[i].resolve_image_view =
            (uintptr_t)att->resolveImageView;
         cache_key.attachments[i].resolve_mode = att->resolveMode;
         cache_key.attachments[i].flags = vk_get_rendering_attachment_flags(att);
      }
   }

   const uint64_t hash = hk_g17p_bg_eot_cache_fingerprint(&cache_key);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_g17p_bg_eot_cache_entry *cached = NULL;
   for (unsigned i = 0; i < HK_G17P_BG_EOT_CACHE_ENTRIES; ++i) {
      struct hk_g17p_bg_eot_cache_entry *entry =
         &cmd->g17p_bg_eot_cache[i];
      if (!entry->valid) {
         if (!cached)
            cached = entry;
         continue;
      }
      if (entry->hash == hash && entry->key_size == sizeof(cache_key) &&
          !memcmp(entry->key, &cache_key, sizeof(cache_key))) {
         gfx->render.g17p_direct_color_resolve |=
            entry->direct_color_resolve_variants != 0;
         return entry;
      }
   }

   if (!cached)
      cached = &cmd->g17p_bg_eot_cache[
         cmd->g17p_bg_eot_cache_next++ % HK_G17P_BG_EOT_CACHE_ENTRIES];

   memcpy(cached->key, &cache_key, sizeof(cache_key));
   cached->hash = hash;
   cached->key_size = sizeof(cache_key);
   cached->valid_variants = 0;
   cached->direct_color_resolve_variants = 0;
   cached->dev_entry = NULL;
   cached->valid = true;

   /* frg-codegen v4 BG/EOT device cache: one value lookup per new
    * pass. A hit fills every variant this recording will ask for; a miss
    * attaches a fresh device entry that hk_build_bg_eot() fills. */
   struct hk_g17p_bg_eot_dev_cache *dc = dev->g17p_bg_eot_dev_cache;
   struct hk_g17p_bg_eot_dev_key dkey;
   if (dc && hk_g17p_bg_eot_dev_key_init(cmd, info, incomplete_render_area,
                                         &dkey)) {
      const uint64_t dhash = hk_g17p_bg_eot_dev_hash(&dkey);
      simple_mtx_lock(&dc->lock);
      struct hk_g17p_bg_eot_dev_entry *de =
         hk_g17p_bg_eot_dev_find(dc, &dkey, dhash);
      if (!de && !dc->full)
         de = hk_g17p_bg_eot_dev_insert(dc, &dkey, dhash);
      if (de) {
         memcpy(cached->program, de->program, sizeof(cached->program));
         cached->valid_variants = de->valid_variants;
      }
      cached->dev_entry = de;
      simple_mtx_unlock(&dc->lock);
   }
   return cached;
}

static unsigned
hk_m5_vertex_secondary_storage(const struct agx_linked_shader *linked)
{
   const unsigned staging_B = linked->g17p_compiler_d_valid
                                ? linked->g17p_compiler_d
                                : linked->nr_gprs;

   /* Native setVertexVariant adds ShaderInfo42 and ShaderInfo38 times the
    * descriptor's maxVertexCallStackDepth. Ordinary pipelines default to one;
    * the linked monolithic shader publishes the same staging extent in both
    * fields. This is the staging footprint used by SPECTPR, not UVS output.
    */
   return staging_B * 2;
}


/* The object program of a native mesh draw: the compiled API task shader when
 * one is bound (opt-in, G17S), else the mesh shader's passthrough or
 * predicated object. */
static inline struct hk_shader *
hk_native_mesh_object(struct hk_graphics_state *gfx, bool *task)
{
   if (gfx->shaders[MESA_SHADER_TASK]) {
      *task = true;
      return &gfx->shaders[MESA_SHADER_TASK]->variants[0];
   }
   *task = false;
   return &gfx->shaders[MESA_SHADER_MESH]->variants[gfx->mesh.object_params
      ? HK_MESH_VARIANT_PREDICATED_OBJECT : HK_MESH_VARIANT_RAST];
}

static void
hk_m5_note_render_resources(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!agx_chip_g17_m5(dev->dev.chip))
      return;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_shader *vs = hk_bound_hw_vs(gfx);
   struct hk_linked_shader *linked_vs = vs && vs->only_linked
      ? vs->only_linked : gfx->linked[MESA_SHADER_VERTEX];
   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   const bool mesh = gfx->shaders[MESA_SHADER_MESH] != NULL;
   struct hk_shader *fs = hk_bound_fs(cmd);
   hk_m5_accumulate_shader_writes(&cs->m5_shader_writes_memory,
                                  vs && vs->info.writes_external_memory);
   hk_m5_accumulate_shader_writes(&cs->m5_shader_writes_memory,
                                  fs && fs->info.writes_external_memory);
   if (mesh) {
      bool task;
      struct hk_shader *object = hk_native_mesh_object(gfx, &task);
      hk_m5_accumulate_shader_writes(&cs->m5_shader_writes_memory,
                                     object->info.writes_external_memory);
   }
   if (cs->m5_shader_writes_memory || gfx->xfb_enabled ||
       gfx->render.m5_attachments_reach_memory)
      hk_sync_note_memory_write(cmd, cs);
   const bool v2 = dev->dev.params.features & DRM_ASAHI_FEATURE_RENDER_RESOURCES_V2;
   struct drm_asahi_render_program_v2 programs[3] = {
      [1] = {
         .legacy = {
            .stage = 1,
            .threadgroup_bytes = cs->tile_resources.threadgroup_bytes,
            .imageblock_bytes = cs->tile_resources.imageblock_bytes,
            .multisample_bytes = cs->tile_resources.multisample_bytes,
         },
      },
   };
   if (linked_vs) {
      programs[0].legacy.primary_bytes = MAX2(linked_vs->b.scratch_size,
                                               linked_vs->b.agx3_spectpr_scratch_size);
      programs[0].legacy.secondary_bytes = hk_m5_vertex_secondary_storage(&linked_vs->b);
   }
   if (linked_fs)
      programs[1].legacy.primary_bytes = MAX2(linked_fs->b.scratch_size,
                                               linked_fs->b.agx3_spectpr_scratch_size);
   if (mesh) {
      if (!v2) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      programs[0].legacy.threadgroup_bytes = vs->b.info.local_size;
      programs[0].threads = vs->b.info.workgroup_size[0] *
         vs->b.info.workgroup_size[1] * vs->b.info.workgroup_size[2];
      programs[0].usage_flags = DRM_ASAHI_RENDER_PROGRAM_MESH;
      bool task;
      struct hk_shader *object = hk_native_mesh_object(gfx, &task);
      programs[2].legacy.threadgroup_bytes = object->b.info.local_size;
      programs[2].legacy.primary_bytes =
         MAX2(object->only_linked->b.scratch_size,
              object->only_linked->b.agx3_spectpr_scratch_size);
      programs[2].legacy.secondary_bytes =
         hk_m5_vertex_secondary_storage(&object->only_linked->b);
      programs[2].threads = task ? object->b.info.workgroup_size[0] *
                                   object->b.info.workgroup_size[1] *
                                   object->b.info.workgroup_size[2]
                                 : 3;
      programs[2].usage_flags = DRM_ASAHI_RENDER_PROGRAM_OBJECT;
      /* A task stage compiled for the native object->mesh record asks the
       * kernel for the TA mesh rendering buffer that backs it. */
      if (task && hk_m5_task_record(dev->dev.params.features))
         programs[2].usage_flags |= DRM_ASAHI_RENDER_PROGRAM_TASK;
   }
   for (unsigned stage = 0; stage < 2; ++stage) {
      if (!(stage ? cs->gen1_native_fragment_rq : cs->gen1_native_vertex_rq))
         continue;
      if (!v2) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      programs[stage].usage_flags |= DRM_ASAHI_RENDER_PROGRAM_SCS;
      programs[stage].scs_layout = UINT64_C(0x01140000600100e8);
      programs[stage].auxiliary_allocation = 14;
      programs[stage].intersection_pool = dev->m5_rq_ift->va->addr;
      programs[stage].intersection_pool_bytes = 0x8000;
      programs[stage].intersection_first = 1;
      programs[stage].intersection_count = 112;
      programs[stage].intersection_default_program = dev->m5_rq_default_program;
      programs[stage].intersection_default_control = dev->m5_rq_default_control;
   }
   const struct drm_asahi_render_program legacy[2] = {programs[0].legacy, programs[1].legacy};
   const size_t count = mesh ? 3 : 2;
   const void *values = v2 ? (const void *)programs : (const void *)legacy;
   const size_t bytes = v2 ? count * sizeof(programs[0]) : sizeof(legacy);
   static_assert(sizeof(cs->m5_render_resources_last) >= sizeof(programs),
                 "the last entry fits");
   /* Consecutive draws mostly repeat the previous entry. */
   if (cs->m5_render_resources_last_size == bytes &&
       !memcmp(cs->m5_render_resources_last, values, bytes))
      return;
   memcpy(cs->m5_render_resources_last, values, bytes);
   cs->m5_render_resources_last_size = bytes;
   for (size_t off = 0; off + bytes <= cs->render_resources.size; off += v2 ? sizeof(programs[0]) : sizeof(legacy[0])) {
      if (!memcmp((char *)cs->render_resources.data + off, values, bytes))
         return;
   }
   util_dynarray_append_array(&cs->render_resources, uint8_t, values, bytes);
}

static bool
hk_m5_has_color(const struct agx_tilebuffer_layout *tib)
{
   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      if (tib->logical_format[rt] != PIPE_FORMAT_NONE)
         return true;
   }
   return false;
}

static bool
hk_m5_empty_tile_layout_supported(const struct agx_tilebuffer_layout *tib)
{
   if (hk_m5_has_color(tib) || agx_tilebuffer_spills(tib))
      return false;
   /* A layered pass reserves one eight-byte sample, as native M5 does. */
   if (tib->nr_samples <= 1)
      return tib->sample_size_B == (tib->layered ? 8 : 0) &&
             tib->tile_size == 32 * 32;
   /* Shared tile allocation reserves one eight-byte sample even when only
    * depth/stencil is attached. Keep its allocation and sample geometry. */
   return tib->sample_size_B == 8 &&
          ((tib->nr_samples == 2 && tib->tile_size == 32 * 32) ||
           (tib->nr_samples == 4 && tib->tile_size == 32 * 16) ||
           (tib->nr_samples == 8 && tib->tile_size == 16 * 16));
}

/* Colour tile allocations an image-operation-free tile program can carry:
 * the same SpecLM imageblock geometry every M5 colour program encodes. */
static bool
hk_m5_color_imageblock_supported(const struct agx_tilebuffer_layout *tib)
{
   if (!hk_m5_has_color(tib) || tib->layered || agx_tilebuffer_spills(tib) ||
       !tib->sample_size_B || (tib->sample_size_B & 7) ||
       tib->sample_size_B > 248)
      return false;
   return (tib->nr_samples == 1 && tib->tile_size == 32 * 32) ||
          (tib->nr_samples == 2 && tib->tile_size == 32 * 32) ||
          (tib->nr_samples == 4 && tib->tile_size == 32 * 16) ||
          (tib->nr_samples == 8 && tib->tile_size == 16 * 16);
}

static bool
hk_m5_mrt_tile_layout(const struct hk_rendering_state *render,
                      struct agx_tilebuffer_layout *tile)
{
   *tile = render->tilebuffer;
   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      if (tile->spilled[rt]) {
         tile->spilled[rt] = false;
         tile->logical_format[rt] = PIPE_FORMAT_NONE;
      }
   }
   return true;
}

static bool
hk_m5_mrt_layout_supported(const struct hk_rendering_state *render)
{
   const unsigned count = render->color_att_count;
   struct agx_tilebuffer_layout tile;
   if (!hk_m5_mrt_tile_layout(render, &tile))
      return false;
   if (count <= HK_MAX_RTS && render->layer_count == 1 && !render->view_mask &&
       hk_m5_empty_tile_layout_supported(&render->tilebuffer))
      return true;
   if (count > HK_MAX_RTS ||
       (render->tilebuffer.nr_samples != 1 &&
        render->tilebuffer.nr_samples != 2 &&
        render->tilebuffer.nr_samples != 4 &&
        render->tilebuffer.nr_samples != 8) ||
       (render->tilebuffer.tile_size != 32 * 32 &&
        render->tilebuffer.tile_size != 32 * 16 &&
        render->tilebuffer.tile_size != 16 * 16) ||
       agx_tilebuffer_spills(&tile))
      return false;
   unsigned active = 0;
   for (unsigned rt = 0; rt < count; ++rt) {
      const struct hk_image_view *view = render->color_att[rt].iview;
      if (!view)
         continue;
      if (render->tilebuffer.spilled[rt]) continue;
      active++;
      const unsigned selector =
         agx_g17p_mrt_eot_selector(render->tilebuffer.logical_format[rt]);
      if (view->vk.image->samples != render->tilebuffer.nr_samples ||
          (selector != 0x15 && selector != 0x35 && selector != 0x55 &&
           selector != 0x75 && selector != 0x95))
         return false;
      const struct hk_image_view *resolve = render->color_att[rt].resolve_iview;
      if (resolve && (view->vk.image->samples <= 1 ||
                      resolve->vk.image->samples != 1 ||
                      resolve->vk.format != view->vk.format ||
                      !resolve->planes[0].eot_pbe_valid))
         return false;
   }
   if (active > 1 &&
       agx_g17p_classify_mrt(&tile) == AGX_G17P_MRT_NONE)
      return false;
   return true;
}

/* A tile shader is device-owned. Cache only its immutable launch bytes;
 * every pass still allocates and publishes its own program and descriptors.
 * Preserve the normal load schedule and fall back for non-fitting addresses. */
struct hk_m5_tile_template {
   struct hk_m5_tile_template *next;
   struct agx3_g17p_graphics_program_desc desc;
   struct agx3_ioto_uniform_range ranges[HK_G17P_PATCH_MAX_RANGES];
   struct hk_g17p_patch_plan plan;
   bool store;
   uint16_t length;
   uint8_t bytes[];
};

static bool
hk_m5_tile_matches(const struct hk_m5_tile_template *t,
                   const struct agx3_g17p_graphics_program_desc *d, bool store)
{
   const struct agx3_g17p_graphics_program_desc *a = &t->desc;
   if (t->store != store || a->shader_va != d->shader_va ||
       a->shader_binary_size != d->shader_binary_size ||
       a->scratch_size != d->scratch_size || a->nr_gprs != d->nr_gprs ||
       a->compiler_d_valid != d->compiler_d_valid || a->compiler_d != d->compiler_d ||
       a->imageblock.sample_size_B != d->imageblock.sample_size_B ||
       a->imageblock.nr_samples != d->imageblock.nr_samples ||
       a->imageblock.tile_size_px != d->imageblock.tile_size_px ||
       a->uniform_range_count != d->uniform_range_count)
      return false;
   for (unsigned i = 0; i < d->uniform_range_count; ++i)
      if (t->ranges[i].start_halfs != d->uniform_ranges[i].start_halfs ||
          t->ranges[i].size_halfs != d->uniform_ranges[i].size_halfs)
         return false;
   return hk_g17p_patch_plan_fits(&t->plan, d->uniform_ranges);
}

static bool
hk_m5_tile_program(struct hk_device *dev, struct agx_bg_eot_shader *shader,
                   const struct agx3_g17p_graphics_program_desc *d, bool store,
                   struct agx3_g17p_graphics_program *program)
{
   if (!d->fragment || d->writes_sample_mask || d->colour_classes ||
       d->preloaded_root || d->patchable || d->threadgroup_memory_size_B ||
       d->constant_program.code_size || (d->cprog_inputs && d->cprog_inputs->count) ||
       d->uniform_range_count > HK_G17P_PATCH_MAX_RANGES)
      return agx3_g17g_tile_program_generate(dev->dev.chip, d, store, program);

   simple_mtx_lock(&dev->bg_eot.lock);
   struct hk_m5_tile_template *selected = NULL;
   unsigned count = 0;
   for (struct hk_m5_tile_template *t = shader->m5_tile_templates; t; t = t->next) {
      ++count;
      if (hk_m5_tile_matches(t, d, store)) {
         selected = t;
         break;
      }
   }
   /* Bound memory even for layouts whose addresses change incompatibly. */
   if (!selected && count < 8) {
      struct agx3_g17p_graphics_program_template *fresh = calloc(1, sizeof(*fresh));
      if (fresh && agx3_g17g_tile_program_template_generate(dev->dev.chip, d, store, fresh)) {
         struct hk_m5_tile_template *t =
            rzalloc_size(shader, sizeof(*t) + fresh->program.length);
         if (t && hk_g17p_patch_plan_init(&t->plan, d->uniform_ranges,
                  d->uniform_range_count, fresh->relocations,
                  fresh->relocation_count, fresh->program.length, false)) {
            t->desc = *d;
            t->desc.uniform_ranges = t->ranges;
            t->desc.cprog_inputs = NULL;
            if (d->uniform_range_count)
               memcpy(t->ranges, d->uniform_ranges,
                      d->uniform_range_count * sizeof(t->ranges[0]));
            t->store = store;
            t->length = fresh->program.length;
            memcpy(t->bytes, fresh->program.bytes, t->length);
            t->next = shader->m5_tile_templates;
            shader->m5_tile_templates = selected = t;
         } else {
            ralloc_free(t);
         }
      }
      free(fresh);
   }
   simple_mtx_unlock(&dev->bg_eot.lock);
   if (!selected)
      return agx3_g17g_tile_program_generate(dev->dev.chip, d, store, program);

   program->length = selected->length;
   memcpy(program->bytes, selected->bytes, selected->length);
   hk_g17p_patch_plan_apply(&selected->plan, d->uniform_ranges, program->bytes);
   if (debug_get_option_m5_verify_tile_loader()) {
      struct agx3_g17p_graphics_program fresh;
      if (!agx3_g17g_tile_program_generate(dev->dev.chip, d, store, &fresh) ||
          fresh.length != program->length ||
          memcmp(fresh.bytes, program->bytes, program->length)) {
         mesa_loge("M5 cached tile launch differs from fresh composition");
         return false;
      }
   }
   return true;
}

static struct hk_bg_eot
hk_build_m5_mrt_bg_eot(struct hk_cmd_buffer *cmd, struct agx_bg_eot_key *key,
                       const VkRenderingInfo *info, bool store,
                       bool partial_render)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   const unsigned count = info->colorAttachmentCount;
   unsigned active[HK_MAX_RTS], active_count = 0;
   bool loads = false, clears = false;
   bool resolving = false;
   struct agx_bg_eot_key tile_key = *key;
   struct agx_bg_eot_shader *shader = NULL;
   if (!hk_m5_mrt_tile_layout(render, &tile_key.tib))
      goto unsupported;
   for (unsigned rt = 0; rt < count; ++rt) {
      if (!render->tilebuffer.spilled[rt]) continue;
      tile_key.op[rt] = AGX_BG_EOT_NONE;
   }
   key = &tile_key;
   if (store && !partial_render) {
      bool shader_resolve = false;
      for (unsigned rt = 0; rt < count; ++rt)
         resolving |= render->color_att[rt].resolve_iview != NULL;
      for (unsigned rt = 0; rt < count; ++rt) {
         /* The EOT resolve averages the samples of a tile. Only AVERAGE may
          * use it: SAMPLE_ZERO (every integer resolve, Vulkan 1.4 table
          * "resolve modes") averaged produces values that are neither
          * sample, so it resolves from the stored samples after the pass. */
         shader_resolve |= render->color_att[rt].resolve_iview &&
            (hk_agx3_format_needs_shader_resolve(key->tib.logical_format[rt]) ||
             info->pColorAttachments[rt].storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
             info->pColorAttachments[rt].resolveMode !=
                VK_RESOLVE_MODE_AVERAGE_BIT);
      }
      /* Raw tile words need floating-point arithmetic after the store.
       * Keep all attachments together on the shared shader resolve path. */
      resolving &= !shader_resolve && !agx_tilebuffer_spills(&render->tilebuffer);
   }
   if (debug_get_option_m5_launch_debug()) {
      for (unsigned rt = 0; rt < count; ++rt) {
         const struct hk_image_view *view = render->color_att[rt].iview;
         fprintf(stderr, "M5 MRT attachment %u: format=%u selector=%#x "
                         "samples=%u resolve=%u op=%u store=%u load=%u "
                         "offset=%u layers=%u views=%#x\n", rt,
                 key->tib.logical_format[rt],
                 agx_g17p_mrt_eot_selector(key->tib.logical_format[rt]),
                 view ? view->vk.image->samples : 0,
                 render->color_att[rt].resolve_iview != NULL, key->op[rt],
                 info->pColorAttachments[rt].storeOp,
                 info->pColorAttachments[rt].loadOp,
                 key->tib._offset_B[rt], render->layer_count, render->view_mask);
      }
   }
   if (count != render->color_att_count || !hk_m5_mrt_layout_supported(render))
      goto unsupported;
   for (unsigned rt = 0; rt < count; ++rt) {
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[rt];
      if (!render->color_att[rt].iview) {
         if (key->op[rt] != AGX_BG_EOT_NONE)
            goto unsupported;
         continue;
      }
      const bool has_resolve = att->resolveMode != VK_RESOLVE_MODE_NONE;
      /* An attachment the EOT does not store (a final EOT of a DONT_CARE or
       * NONE attachment without resolve) is simply absent from the store:
       * the M5 multi-target EOT numbers its PBEs and offset arguments by
       * stored target only. A partial EOT stores every attachment
       * (key->op), whatever its API store operation. */
      if (store && key->op[rt] == AGX_BG_EOT_NONE)
         continue;
      if (render->tilebuffer.spilled[rt]) continue;
      active[active_count++] = rt;
      /* A final resolve-only EOT writes the single-sample descriptors.
       * Partial EOTs always preserve the multisample source for reload. */
      if ((has_resolve && att->resolveMode != VK_RESOLVE_MODE_AVERAGE_BIT &&
           att->resolveMode != VK_RESOLVE_MODE_SAMPLE_ZERO_BIT) ||
          (resolving && (!has_resolve ||
                         att->storeOp == VK_ATTACHMENT_STORE_OP_STORE)))
         goto unsupported;
      if (store) {
         if (key->op[rt] != AGX_EOT_STORE)
            goto unsupported;
      } else {
         if (key->op[rt] == AGX_BG_EOT_NONE &&
             att->loadOp == VK_ATTACHMENT_LOAD_OP_DONT_CARE)
            key->op[rt] = AGX_BG_CLEAR;
         if (key->op[rt] != AGX_BG_LOAD && key->op[rt] != AGX_BG_CLEAR)
            goto unsupported;
         loads |= key->op[rt] == AGX_BG_LOAD;
         clears |= key->op[rt] == AGX_BG_CLEAR;
      }
   }
   for (unsigned rt = count; rt < HK_MAX_RTS; ++rt) {
      if (key->op[rt] != AGX_BG_EOT_NONE)
         goto unsupported;
   }
   key->mixed_mrt = true;
   key->multisample_store = store && !resolving && key->tib.nr_samples > 1;
   const unsigned clear_base = loads ? 8 : 4;
   if (clears)
      key->reserved_preamble = clear_base + 8 * count;
   shader = agx_get_bg_eot_shader(&dev->bg_eot, key);
   if (!shader || !shader->info.nr_gprs || !shader->info.agx3_staging_known ||
       shader->info.scratch_size || shader->info.has_preamble ||
       shader->info.push_count != key->reserved_preamble ||
       shader->info.uses_txf != loads)
      goto unsupported;

   struct agx3_ioto_uniform_range ranges[3];
   unsigned range_count = 0;
   if (store) {
      struct hk_native_pbe_descriptor descriptors[HK_MAX_RTS];
      uint64_t arguments[1 + HK_MAX_RTS] = {0};
      for (unsigned i = 0; i < active_count; ++i) {
         const unsigned rt = active[i];
         const struct hk_image_view *view = resolving
            ? render->color_att[rt].resolve_iview : render->color_att[rt].iview;
         if (!view || !(key->tib.layered
                           ? view->planes[0].layered_eot_pbe_valid
                           : view->planes[0].eot_pbe_valid))
            goto unsupported;
         descriptors[i] = key->tib.layered ? view->planes[0].layered_eot
                                            : view->planes[0].eot;
         if (resolving && util_format_is_srgb(key->tib.logical_format[rt]) &&
             !(render->color_att[rt].flags &
               VK_RENDERING_ATTACHMENT_RESOLVE_SKIP_TRANSFER_FUNCTION_BIT_KHR))
            ((uint8_t *)&descriptors[i])[15] |= 0x20;
      }
      uint64_t pbes = hk_pool_upload(cmd, descriptors,
                                      active_count * sizeof(descriptors[0]), 32);
      if (!pbes)
         return (struct hk_bg_eot){0};
      /* Genuine builtin ABI: U0 is the contiguous PBE table pointer;
       * U4+4rt carries each tile byte offset. IMGWBLK selects descriptorrt. */
      arguments[0] = pbes;
      for (unsigned i = 0; i < active_count; ++i) {
         const unsigned rt = active[i];
         arguments[1 + i] = agx_tilebuffer_offset_B(&key->tib, rt);
         if (debug_get_option_m5_launch_debug()) {
            fprintf(stderr, "M5 MRT EOT argument rt=%u pbe=%#" PRIx64
                            " offset=%" PRIu64 " descriptor=", rt,
                    pbes + i * sizeof(descriptors[0]), arguments[1 + i]);
            const uint8_t *bytes = (const uint8_t *)&descriptors[i];
            for (unsigned b = 0; b < sizeof(descriptors[0]); ++b)
               fprintf(stderr, "%02x", bytes[b]);
            fprintf(stderr, "\n");
         }
      }
      uint64_t table = hk_pool_upload(cmd, arguments, 8 * (1 + active_count), 16);
      if (!table)
         return (struct hk_bg_eot){0};
      ranges[range_count++] = (struct agx3_ioto_uniform_range){0, 4 * (1 + active_count), table};
   } else {
      if (loads) {
         if (!dev->rodata.bo || !dev->rodata.bo->va)
            goto unsupported;
         struct hk_native_texture_descriptor descriptors[2 * HK_MAX_RTS] = {0};
         for (unsigned rt = 0; rt < count; ++rt) {
            if (key->op[rt] == AGX_BG_LOAD)
               descriptors[2 * rt] = key->tib.layered
                  ? render->color_att[rt].iview->planes[0].layered_background
                  : render->color_att[rt].iview->planes[0].background;
         }
         uint64_t textures = hk_pool_upload(cmd, descriptors,
            2 * count * sizeof(descriptors[0]), 32);
         if (!textures)
            return (struct hk_bg_eot){0};
         const uint64_t pointers[] = {textures, dev->rodata.bo->va->addr};
         uint64_t table = hk_pool_upload(cmd, pointers, sizeof(pointers), 16);
         if (!table)
            return (struct hk_bg_eot){0};
         ranges[range_count++] = (struct agx3_ioto_uniform_range){0, 4, table};
         ranges[range_count++] = (struct agx3_ioto_uniform_range){4, 4, table + 8};
      }
      if (clears) {
         VkClearColorValue colors[HK_MAX_RTS] = {0};
         for (unsigned rt = 0; rt < count; ++rt) {
            if (key->op[rt] == AGX_BG_CLEAR &&
                info->pColorAttachments[rt].loadOp != VK_ATTACHMENT_LOAD_OP_DONT_CARE)
               colors[rt] = info->pColorAttachments[rt].clearValue.color;
         }
         uint64_t table = hk_pool_upload(cmd, colors, 16 * count, 16);
         if (!table)
            return (struct hk_bg_eot){0};
         ranges[range_count++] = (struct agx3_ioto_uniform_range){clear_base, 8 * count, table};
      }
   }
   size_t capacity;
   if (!agx3_g17g_graphics_program_bound(dev->dev.chip, ranges, range_count, &capacity))
      goto unsupported;
   struct hk_ioto_reservation reservation;
   if (!hk_cmd_reserve_ioto_program(cmd, capacity, "M5 MRT tile program", &reservation))
      return (struct hk_bg_eot){0};
   const struct agx3_g17p_graphics_program_desc desc = {
      .program_va = reservation.va,
      .shader_va = shader->ptr + shader->info.main_offset,
      .shader_binary_size = shader->info.binary_size,
      .scratch_size = shader->info.scratch_size,
      .nr_gprs = shader->info.nr_gprs,
      .compiler_d_valid = shader->info.agx3_staging_known,
      .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
      .fragment = true,
      .imageblock = {key->tib.sample_size_B, key->tib.nr_samples, key->tib.tile_size},
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
   };
   struct agx3_g17p_graphics_program program;
   if (!hk_m5_tile_program(dev, shader, &desc, store, &program) ||
       program.length > capacity)
      goto unsupported;
   if (debug_get_option_m5_launch_debug()) {
      fprintf(stderr, "M5 MRT %s count=%u sampleB=%u tile=%u gprs=%u D=%u "
                      "shader=%#" PRIx64 " mainSize=%u ioto=",
              store ? "EOT" : "BG", count, key->tib.sample_size_B,
              key->tib.tile_size, desc.nr_gprs, desc.compiler_d,
              desc.shader_va, shader->info.main_size);
      for (size_t i = 0; i < program.length; ++i)
         fprintf(stderr, "%02x", program.bytes[i]);
      fprintf(stderr, "\n");
      for (unsigned i = 0; i < range_count; ++i)
         fprintf(stderr, "M5 MRT %s range=%u start=%u size=%u address=%#" PRIx64
                         "\n", store ? "EOT" : "BG", i,
                 ranges[i].start_halfs, ranges[i].size_halfs, ranges[i].address);
      if (shader->bo && shader->bo->va &&
          desc.shader_va >= shader->bo->va->addr) {
         const uint64_t offset = desc.shader_va - shader->bo->va->addr;
         if (offset <= shader->bo->size &&
             shader->info.main_size <= shader->bo->size - offset) {
            const uint8_t *bytes = agx_bo_map(shader->bo);
            if (bytes) {
               fprintf(stderr, "M5 MRT %s shaderText=", store ? "EOT" : "BG");
               for (unsigned i = 0; i < shader->info.main_size; ++i)
                  fprintf(stderr, "%02x", bytes[offset + i]);
               fprintf(stderr, "\n");
            }
         }
      }
   }
   memcpy(reservation.map, program.bytes, program.length);
   hk_cmd_buffer_retain_bo(cmd, shader->bo);
   uint64_t published = hk_cmd_publish_ioto_program(cmd, &reservation, program.length);
   if (published && resolving)
      cmd->state.gfx.render.g17p_direct_color_resolve = true;
   return (struct hk_bg_eot){
      .usc = published,
      .primary_storage_B = desc.scratch_size,
      .staging_storage_B = desc.compiler_d,
      .rsrc_spec = store ? 0 : loads && key->tib.nr_samples > 1
         ? UINT64_C(0xffff800000000000) : UINT64_C(0xffff800000000040),
   };
unsupported:
   if (debug_get_option_m5_launch_debug())
      mesa_loge("M5 MRT admission: store=%u shader=%u gprs=%u staging=%u "
                "scratch=%u preamble=%u push=%u expected=%u txf=%u loads=%u "
                "attachments=%u samples=%u sampleB=%u tile=%u",
                store, shader != NULL, shader ? shader->info.nr_gprs : 0,
                shader ? shader->info.agx3_staging_known : 0,
                shader ? shader->info.scratch_size : 0,
                shader ? shader->info.has_preamble : 0,
                shader ? shader->info.push_count : 0, key->reserved_preamble,
                shader ? shader->info.uses_txf : 0, loads, count,
                key->tib.nr_samples, key->tib.sample_size_B, key->tib.tile_size);
   mesa_loge("M5 MRT BG/EOT exceeds supported tile-program profile");
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   return (struct hk_bg_eot){0};
}

static struct hk_bg_eot
hk_build_m5_empty_bg_eot(struct hk_cmd_buffer *cmd,
                          struct agx_bg_eot_key *key, bool store)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   /* A colour tile whose final EOT stores nothing (every attachment
    * DONT_CARE/NONE) ends with the same image-operation-free program; only
    * its launch's imageblock is the colour allocation, as G17P's empty
    * (no-resource) tile program carries key->tib. */
   const bool color = hk_m5_has_color(&key->tib);
   /* The STOP program performs no image operation; a layered pass differs
    * only in its imageblock and in render-command layer state. */
   if (color ? !store || !hk_m5_color_imageblock_supported(&key->tib)
             : !hk_m5_empty_tile_layout_supported(&render->tilebuffer))
      goto unsupported;
   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      if (key->op[rt] != AGX_BG_EOT_NONE)
         goto unsupported;
   }
   /* This program has no image operations. Reuse the same STOP shader for
    * every sample count; its launch still carries the real tile allocation. */
   struct agx_bg_eot_key shader_key = *key;
   shader_key.tib.nr_samples = 1;
   shader_key.tib.sample_size_B = 0;
   shader_key.tib.tile_size = 32 * 32;
   shader_key.tib.layered = false;
   struct agx_bg_eot_shader *shader =
      agx_get_bg_eot_shader(&dev->bg_eot, &shader_key);
   if (!shader || shader->info.binary_size != 4 ||
       shader->info.nr_gprs != 2 || !shader->info.agx3_staging_known ||
       shader->info.agx3_staging_halfs || shader->info.scratch_size ||
       shader->info.has_preamble || shader->info.push_count ||
       shader->info.uses_txf)
      goto unsupported;

   size_t capacity;
   if (!agx3_g17g_graphics_program_bound(dev->dev.chip, NULL, 0, &capacity))
      goto unsupported;
   struct hk_ioto_reservation reservation;
   if (!hk_cmd_reserve_ioto_program(cmd, capacity,
                                   "M5 empty tile program", &reservation))
      return (struct hk_bg_eot){0};
   const struct agx3_g17p_graphics_program_desc desc = {
      .program_va = reservation.va,
      .shader_va = shader->ptr,
      .shader_binary_size = shader->info.binary_size,
      .nr_gprs = shader->info.nr_gprs,
      .compiler_d_valid = true,
      .fragment = true,
      .imageblock = {key->tib.sample_size_B, MAX2(key->tib.nr_samples, 1),
                     key->tib.tile_size},
   };
   struct agx3_g17p_graphics_program program;
   if (!hk_m5_tile_program(dev, shader, &desc, store, &program) ||
       program.length > capacity)
      goto unsupported;
   memcpy(reservation.map, program.bytes, program.length);
   hk_cmd_buffer_retain_bo(cmd, shader->bo);
   return (struct hk_bg_eot){
      .usc = hk_cmd_publish_ioto_program(cmd, &reservation, program.length),
      .rsrc_spec = store ? 0 : UINT64_C(0xffff800000000040),
   };

unsupported:
   if (debug_get_option_m5_launch_debug())
      mesa_loge("M5 empty BG/EOT admission: store=%u color=%u sampleB=%u "
                "samples=%u tile=%u layered=%u layers=%u views=%#x spills=%u "
                "ops=%u%u%u%u%u%u%u%u",
                store, hk_m5_has_color(&key->tib), key->tib.sample_size_B,
                key->tib.nr_samples, key->tib.tile_size, key->tib.layered,
                cmd->state.gfx.render.layer_count,
                cmd->state.gfx.render.view_mask,
                agx_tilebuffer_spills(&key->tib), key->op[0], key->op[1],
                key->op[2], key->op[3], key->op[4], key->op[5], key->op[6],
                key->op[7]);
   mesa_loge("M5 empty BG/EOT exceeds native no-imageblock profile");
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   return (struct hk_bg_eot){0};
}

static struct hk_bg_eot
hk_build_m5_bg_eot(struct hk_cmd_buffer *cmd, struct agx_bg_eot_key *key,
                   const VkRenderingInfo *info, bool store, bool partial_render)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   if (!hk_m5_has_color(&key->tib))
      return hk_build_m5_empty_bg_eot(cmd, key, store);
   if (store) {
      bool stores = false;
      for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt)
         stores |= key->op[rt] != AGX_BG_EOT_NONE;
      if (!stores)
         return hk_build_m5_empty_bg_eot(cmd, key, store);
   }
   /* BG/EOT descriptor slots are dense, independently of fragment output
    * locations. Keep the render state in Vulkan order and compact only this
    * internal program's key and attachment selection. */
   struct agx_bg_eot_key compact_key = *key;
   uint8_t attachment_map[HK_MAX_RTS];
   const unsigned active_count = agx_tilebuffer_compact(
      &key->tib, &compact_key.tib, attachment_map);
   /* Wide multisample colors use the same typed tile program with or without
    * a final resolve. The narrow RT0 loader only covers eight-byte samples. */
   if (active_count > 1 ||
       (key->tib.nr_samples > 1 && key->tib.sample_size_B > 8) ||
       (active_count == 1 && render->color_att[attachment_map[0]].resolve_iview &&
        (agx_g17p_mrt_eot_selector(key->tib.logical_format[attachment_map[0]]) != 0x75 ||
         hk_agx3_format_needs_shader_resolve(
            key->tib.logical_format[attachment_map[0]]))))
      return hk_build_m5_mrt_bg_eot(cmd, key, info, store, partial_render);
   const unsigned logical = attachment_map[0];
   memset(compact_key.op, 0, sizeof(compact_key.op));
   compact_key.op[0] = key->op[logical];
   VkRenderingInfo compact_info = *info;
   compact_info.colorAttachmentCount = 1;
   compact_info.pColorAttachments = &info->pColorAttachments[logical];
   key = &compact_key;
   info = &compact_info;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_image_view *view = cmd->state.gfx.render.color_att[logical].iview;
   struct agx_bg_eot_shader *shader = NULL;
   const bool clear_dont_care = !store && key->op[0] == AGX_BG_EOT_NONE &&
      info->colorAttachmentCount == 1 &&
      info->pColorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   enum agx_bg_eot_op op = clear_dont_care ? AGX_BG_CLEAR : key->op[0];
   const bool multisample =
      ((key->tib.nr_samples == 2 && key->tib.tile_size == 32 * 32) ||
       (key->tib.nr_samples == 4 && key->tib.tile_size == 32 * 16) ||
       (key->tib.nr_samples == 8 && key->tib.tile_size == 16 * 16));
   const bool single_sample = key->tib.nr_samples == 1 &&
                              key->tib.tile_size == 32 * 32;
   /* Four raw 32-bit color lanes use the native 16-byte tile stride. */
   const bool sample_stride = key->tib.sample_size_B == 8 ||
      (single_sample && key->tib.sample_size_B == 16);
   const VkRenderingAttachmentInfo *att = info->colorAttachmentCount == 1
      ? &info->pColorAttachments[0] : NULL;
   struct hk_image_view *resolve =
      cmd->state.gfx.render.color_att[logical].resolve_iview;
   const bool has_resolve = att && att->resolveMode != VK_RESOLVE_MODE_NONE;
   const bool final_resolve = store && !partial_render && has_resolve;
   const bool store_source = att && att->storeOp == VK_ATTACHMENT_STORE_OP_STORE;
   if (!view || info->colorAttachmentCount != 1 || !sample_stride ||
       (has_resolve &&
        (!multisample || att->resolveMode != VK_RESOLVE_MODE_AVERAGE_BIT ||
         agx_g17p_mrt_eot_selector(key->tib.logical_format[0]) != 0x75 || !resolve ||
         resolve->vk.image->samples != 1 || resolve->vk.format != view->vk.format ||
         !(key->tib.layered ? resolve->planes[0].layered_eot_pbe_valid
                            : resolve->planes[0].eot_pbe_valid))) ||
       (store ? op != AGX_EOT_STORE : op != AGX_BG_CLEAR && op != AGX_BG_LOAD) ||
       (!single_sample && !multisample) || agx_tilebuffer_spills(&key->tib) ||
       agx_tilebuffer_offset_B(&key->tib, 0))
      goto unsupported;
   for (unsigned rt = 1; rt < HK_MAX_RTS; ++rt) {
      if (key->op[rt] != AGX_BG_EOT_NONE)
         goto unsupported;
   }
   key->op[0] = op;
   /* Partial EOTs preserve samples for reload. A final EOT can resolve the
    * tile directly, retaining the source as well only when STORE asks for it. */
   if (multisample)
      key->multisample_store = store && (!final_resolve || store_source);
   key->m5_store_and_resolve = final_resolve && store_source;
   if (op == AGX_BG_CLEAR)
      key->reserved_preamble = AGX3_G17P_RT0_CLEAR_PUSH_HALFS;
   shader = agx_get_bg_eot_shader(&dev->bg_eot, key);
   if (!shader || !shader->info.nr_gprs || !shader->info.agx3_staging_known ||
       shader->info.scratch_size || shader->info.has_preamble)
      goto unsupported;

   struct agx3_ioto_uniform_range ranges[2];
   unsigned range_count = 1;
   if (store) {
      struct hk_image_view *target = final_resolve && !store_source
         ? resolve : view;
      if (view->vk.image->samples != key->tib.nr_samples ||
          !(key->tib.layered ? target->planes[0].layered_eot_pbe_valid
                            : target->planes[0].eot_pbe_valid))
         goto unsupported;
      const struct hk_native_pbe_descriptor *descriptor = key->tib.layered
         ? &target->planes[0].layered_eot : &target->planes[0].eot;
      struct hk_native_pbe_descriptor descriptors[2] = {*descriptor};
      if (key->m5_store_and_resolve)
         descriptors[1] = key->tib.layered ? resolve->planes[0].layered_eot
                                            : resolve->planes[0].eot;
      if (final_resolve && util_format_is_srgb(key->tib.logical_format[0]) &&
          !(render->color_att[logical].flags &
            VK_RENDERING_ATTACHMENT_RESOLVE_SKIP_TRANSFER_FUNCTION_BIT_KHR))
         ((uint8_t *)&descriptors[key->m5_store_and_resolve ? 1 : 0])[15] |= 0x20;
      const unsigned descriptor_count = key->m5_store_and_resolve ? 2 : 1;
      uint64_t pbe = hk_pool_upload(cmd, descriptors,
         descriptor_count * sizeof(*descriptor), 32);
      if (!pbe)
         return (struct hk_bg_eot){0};
      const uint64_t arguments[] = {
         pbe, 0, key->tib.layered ? 0 : pbe + sizeof(*descriptor),
      };
      uint64_t table = hk_pool_upload(cmd, arguments,
         (key->m5_store_and_resolve ? 3 : 2) * sizeof(uint64_t), 16);
      if (!table)
         return (struct hk_bg_eot){0};
      ranges[0] = (struct agx3_ioto_uniform_range){
         0, key->m5_store_and_resolve ? 12 : 6, table};
   } else if (op == AGX_BG_CLEAR) {
      if (shader->info.push_count != AGX3_G17P_RT0_CLEAR_PUSH_HALFS ||
          shader->info.uses_txf)
         goto unsupported;
      const VkClearColorValue zero = {0};
      uint64_t clear = hk_pool_upload(cmd,
         clear_dont_care ? &zero : &info->pColorAttachments[0].clearValue.color,
         sizeof(zero), 16);
      if (!clear)
         return (struct hk_bg_eot){0};
      ranges[0] = (struct agx3_ioto_uniform_range){4, 8, clear};
   } else {
      if (shader->info.push_count || !shader->info.uses_txf ||
          !dev->rodata.bo || !dev->rodata.bo->va)
         goto unsupported;
      const struct hk_native_texture_descriptor *descriptor = key->tib.layered
         ? &view->planes[0].layered_background : &view->planes[0].background;
      uint64_t texture = hk_pool_upload(cmd, descriptor, sizeof(*descriptor), 32);
      if (!texture)
         return (struct hk_bg_eot){0};
      const uint64_t pointers[] = {texture, dev->rodata.bo->va->addr};
      uint64_t table = hk_pool_upload(cmd, pointers, sizeof(pointers), 8);
      if (!table)
         return (struct hk_bg_eot){0};
      ranges[0] = (struct agx3_ioto_uniform_range){0, 4, table};
      ranges[1] = (struct agx3_ioto_uniform_range){4, 4, table + 8};
      range_count = 2;
   }
   size_t capacity;
   if (!agx3_g17g_graphics_program_bound(dev->dev.chip, ranges, range_count,
                                        &capacity))
      goto unsupported;
   struct hk_ioto_reservation reservation;
   if (!hk_cmd_reserve_ioto_program(cmd, capacity, "M5 tile program", &reservation))
      return (struct hk_bg_eot){0};
   const struct agx3_g17p_graphics_program_desc desc = {
      .program_va = reservation.va,
      .shader_va = shader->ptr + shader->info.main_offset,
      .shader_binary_size = shader->info.binary_size,
      .scratch_size = shader->info.scratch_size,
      .nr_gprs = shader->info.nr_gprs,
      .compiler_d_valid = shader->info.agx3_staging_known,
      .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
      .fragment = true,
      .imageblock = {key->tib.sample_size_B, key->tib.nr_samples, key->tib.tile_size},
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
   };
   struct agx3_g17p_graphics_program program;
   if (!hk_m5_tile_program(dev, shader, &desc, store, &program) ||
       program.length > capacity)
      goto unsupported;
   memcpy(reservation.map, program.bytes, program.length);
   hk_cmd_buffer_retain_bo(cmd, shader->bo);
   uint64_t published = hk_cmd_publish_ioto_program(cmd, &reservation, program.length);
   if (published && final_resolve)
      cmd->state.gfx.render.g17p_direct_color_resolve = true;
   return (struct hk_bg_eot){
      .usc = published,
      .primary_storage_B = desc.scratch_size,
      .staging_storage_B = desc.compiler_d,
      .rsrc_spec = store ? 0 : multisample && op == AGX_BG_LOAD
         ? UINT64_C(0xffff800000000000) : UINT64_C(0xffff800000000040),
   };
unsupported:
   if (debug_get_option_m5_launch_debug()) {
      mesa_loge("M5 BG/EOT admission: store=%u partial=%u op=%u attachments=%u "
                "sampleB=%u samples=%u tile=%u layered=%u spills=%u offsetB=%u "
                "format=%u viewformat=%u pbe=%u shader=%u gprs=%u "
                "stagingKnown=%u scratch=%u preamble=%u push=%u txf=%u",
                store, partial_render, op, info->colorAttachmentCount,
                key->tib.sample_size_B, key->tib.nr_samples, key->tib.tile_size,
                key->tib.layered, agx_tilebuffer_spills(&key->tib),
                agx_tilebuffer_offset_B(&key->tib, 0), key->tib.logical_format[0],
                view ? view->vk.format : VK_FORMAT_UNDEFINED,
                view && (key->tib.layered ? view->planes[0].layered_eot_pbe_valid
                                          : view->planes[0].eot_pbe_valid),
                shader != NULL, shader ? shader->info.nr_gprs : 0,
                shader ? shader->info.agx3_staging_known : 0,
                shader ? shader->info.scratch_size : 0,
                shader ? shader->info.has_preamble : 0,
                shader ? shader->info.push_count : 0,
                shader ? shader->info.uses_txf : 0);
   }
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   return (struct hk_bg_eot){0};
}

/* anatomy_ops: NULL, or HK_MAX_RTS bytes receiving the frame anatomy
 * profiler's per-RT summary of the key this program is built from. */
static struct hk_bg_eot
hk_build_bg_eot_program(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                        bool store, bool partial_render,
                        bool incomplete_render_area,
                        struct hk_g17p_bg_eot_cache_entry *cache,
                        uint8_t *anatomy_ops)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* Construct the key */
   /* Zero the padding too: the G17P device cache compares keys bytewise. */
   struct agx_bg_eot_key key;
   memset(&key, 0, sizeof(key));
   key.tib = render->tilebuffer;
   static_assert(AGX_BG_EOT_NONE == 0, "default initializer");

   key.tib.layered = (render->cr.layers > 1);

   /* Every background program that reloads a multisample tile reloads it from
    * the multisample surface, per sample. That is the only surface that
    * carries the samples, and the samples are exactly what a reload exists to
    * restore: a resolve has averaged them away and cannot be un-averaged.
    * This holds for loadOp = LOAD, for an incomplete render area, and for the
    * mid-frame reload of a partial render alike, so single_sample_reload is
    * never the right answer here and is left false.
    *
    * Gallium does set it (agx_state.c), and it swaps in the resolve target's
    * surface to match. Honeykrisp once set the flag without swapping the
    * descriptor, which asked a single-sample texture fetch for the
    * multisample attachment -- a surface a direct resolve never writes. A
    * fresh allocation reads as zero, so every sample a partial render carried
    * across a stream boundary came back zero and the tile's whole history,
    * clear included, was lost.
    */
   key.single_sample_reload = false;


   bool needs_textures_for_spilled_rts =
      agx_tilebuffer_spills(&render->tilebuffer) && !partial_render && !store;

   for (unsigned i = 0; i < info->colorAttachmentCount; ++i) {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[i];
      if (att_info->imageView == VK_NULL_HANDLE)
         continue;

      /* Partial render programs exist only to store/load the tilebuffer to
       * main memory. When render targets are already spilled to main memory,
       * there's nothing to do.
       */
      if (key.tib.spilled[i] &&
          (partial_render || store || agx_native_graphics(&dev->dev) ||
           agx_chip_g17_m5(dev->dev.chip)))
         continue;

      if (store) {
         bool should_store = is_attachment_stored(att_info);

         /* Partial renders always need to flush to memory. */
         should_store |= partial_render;

         if (should_store)
            key.op[i] = AGX_EOT_STORE;
      } else {
         bool load = att_info->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
         bool clear = att_info->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR;

         /* The background program used for partial renders must always load
          * whatever was stored in the mid-frame end-of-tile program.
          */
         load |= partial_render;

         /* With an incomplete render area, we're forced to load back tiles and
          * then use the 3D pipe for the clear.
          */
         load |= incomplete_render_area;

         /* Don't read back spilled render targets, they're already in memory */
         load &= !key.tib.spilled[i];

         /* This is a very frustrating corner case. From the spec:
          *
          *     VK_ATTACHMENT_STORE_OP_NONE specifies the contents within the
          *     render area are not accessed by the store operation as long as
          *     no values are written to the attachment during the render pass.
          *
          * With VK_ATTACHMENT_STORE_OP_NONE, we suppress stores on the main
          * end-of-tile program. Unfortunately, that's not enough: we also need
          * to preserve the contents throughout partial renders. The easiest way
          * to do that is forcing a load in the background program, so that
          * partial stores for unused attachments will be no-op'd by writing
          * existing contents.
          *
          * Optimizing this would require nontrivial tracking. Fortunately,
          * this is all Android gunk and we don't have to care too much for
          * dekstop games. So do the simple thing.
          */
         bool no_store = (att_info->storeOp == VK_ATTACHMENT_STORE_OP_NONE);
         bool no_store_wa = no_store && !load && !clear;
         if (no_store_wa) {
            perf_debug(cmd, "STORE_OP_NONE workaround");
         }

         load |= no_store_wa;

         /* Don't apply clears for spilled render targets when we clear the
          * render area explicitly after.
          */
         if (key.tib.spilled[i] && incomplete_render_area)
            continue;

         if (load)
            key.op[i] = AGX_BG_LOAD;
         else if (clear)
            key.op[i] = AGX_BG_CLEAR;
      }
   }

   if (unlikely(anatomy_ops)) {
      /* Profiler only. The key ops below are what every chip path builds
       * from; the M5 single-RT path additionally treats a DONT_CARE load
       * with no op as a clear, visible in the log as hw '-' with API 'D'. */
      for (unsigned i = 0; i < HK_MAX_RTS; ++i) {
         const VkRenderingAttachmentInfo *att =
            i < info->colorAttachmentCount ? &info->pColorAttachments[i] : NULL;
         const bool present = att && att->imageView != VK_NULL_HANDLE;
         anatomy_ops[i] = hk_frame_anatomy_encode_op(
            present, key.op[i], present ? att->loadOp : 0,
            present ? att->storeOp : 0);
      }
   }

   if (agx_native_graphics(&dev->dev)) {
      const unsigned variant = (store ? 2 : 0) | (partial_render ? 1 : 0);
      if (cache && (cache->valid_variants & BITFIELD_BIT(variant))) {
         render->g17p_direct_color_resolve |=
            (cache->direct_color_resolve_variants &
             BITFIELD_BIT(variant)) != 0;
         return cache->program[variant];
      }

      /* frg-codegen v4: build this variant into the device-lifetime entry
       * hk_g17p_bg_eot_cache_get() attached, so every later recording of the
       * same pass reuses it (BG/EOT device cache). */
      struct hk_g17p_bg_eot_dev_cache *dc = dev->g17p_bg_eot_dev_cache;
      struct hk_g17p_bg_eot_dev_entry *de = cache ? cache->dev_entry : NULL;
      struct hk_bg_eot program = {0};
      bool built = false;
      if (dc && de) {
         simple_mtx_lock(&dc->lock);
         if (de->valid_variants & BITFIELD_BIT(variant)) {
            program = de->program[variant];
            built = true;
         } else if (hk_g17p_bg_eot_arena_room(
                       dev, dc, &dc->program, true,
                       HK_G17P_BG_EOT_DEV_PROGRAM_MAX,
                       AGX_IOTO_PROGRAM_ALIGNMENT) &&
                    hk_g17p_bg_eot_arena_room(
                       dev, dc, &dc->data, false,
                       HK_G17P_BG_EOT_DEV_DATA_MAX, 32)) {
            bool in_device = false;
            program = hk_build_g17p_bg_eot(cmd, &key, info, store,
                                           partial_render, dc, &in_device);
            built = true;
            if (program.usc && in_device &&
                !vk_command_buffer_has_error(&cmd->vk)) {
               de->program[variant] = program;
               de->valid_variants |= BITFIELD_BIT(variant);
            }
         } else {
            dc->full = true;
         }
         simple_mtx_unlock(&dc->lock);
      }
      if (!built)
         program = hk_build_g17p_bg_eot(cmd, &key, info, store,
                                        partial_render, NULL, NULL);
      if (cache && program.usc && !vk_command_buffer_has_error(&cmd->vk)) {
         cache->program[variant] = program;
         cache->valid_variants |= BITFIELD_BIT(variant);
         if (store && !partial_render && render->g17p_direct_color_resolve)
            cache->direct_color_resolve_variants |= BITFIELD_BIT(variant);
      }
      return program;
   }

   if (agx_chip_g17_m5(dev->dev.chip))
      return hk_build_m5_bg_eot(cmd, &key, info, store, partial_render);

   const unsigned descriptor_generation =
      agx_chip_descriptor_generation(dev->dev.chip);
   if (!hk_native_pbe_hardware_supported(descriptor_generation)) {
      if (needs_textures_for_spilled_rts) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return (struct hk_bg_eot){0};
      }

      for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
         if (key.op[rt] == AGX_EOT_STORE) {
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return (struct hk_bg_eot){0};
         }
      }
   }

   /* Begin building the pipeline */
   size_t usc_size = agx_usc_size(3 + HK_MAX_RTS);
   struct agx_ptr t = hk_pool_usc_alloc(cmd, usc_size, 64);
   if (!t.cpu)
      return (struct hk_bg_eot){.usc = t.gpu};

   struct agx_usc_builder b = agx_usc_builder(t.cpu, usc_size);

   bool uses_txf = false;
   unsigned uniforms = 0;
   unsigned nr_tex = 0;

   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      const VkRenderingAttachmentInfo *att_info = &info->pColorAttachments[rt];
      struct hk_image_view *iview = render->color_att[rt].iview;

      if (key.op[rt] == AGX_BG_LOAD) {
         uses_txf = true;

         const struct hk_native_texture_descriptor *desc =
            key.tib.layered ? &iview->planes[0].layered_background
                            : &iview->planes[0].background;

         agx_usc_pack(&b, TEXTURE, cfg) {
            /* Shifted to match eMRT indexing, could be optimized */
            cfg.start = rt * 2;
            cfg.count = 1;
            cfg.buffer = hk_pool_upload(
               cmd, desc,
               hk_texture_descriptor_size(descriptor_generation), 8);
         }

         nr_tex = (rt * 2) + 1;
      } else if (key.op[rt] == AGX_BG_CLEAR) {
         static_assert(sizeof(att_info->clearValue.color) == 16, "fixed ABI");
         uint64_t colour =
            hk_pool_upload(cmd, &att_info->clearValue.color, 16, 16);

         agx_usc_uniform(&b, 4 + (8 * rt), 8, colour);
         uniforms = MAX2(uniforms, 4 + (8 * rt) + 8);
      } else if (key.op[rt] == AGX_EOT_STORE) {
         const bool valid = key.tib.layered
                               ? iview->planes[0].layered_eot_pbe_valid
                               : iview->planes[0].eot_pbe_valid;
         if (!valid) {
            mesa_loge("render target %u needs an unproven native PBE axis",
                      rt);
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FORMAT_NOT_SUPPORTED);
            return (struct hk_bg_eot){0};
         }

         const struct hk_native_pbe_descriptor *desc =
            key.tib.layered ? &iview->planes[0].layered_eot
                            : &iview->planes[0].eot;

         agx_usc_pack(&b, TEXTURE, cfg) {
            cfg.start = rt;
            cfg.count = 1;
            cfg.buffer = hk_pool_upload(
               cmd, desc, hk_pbe_descriptor_size(descriptor_generation), 8);
         }

         nr_tex = rt + 1;
      }
   }

   if (needs_textures_for_spilled_rts) {
      hk_usc_upload_spilled_rt_descs(&b, cmd);
      uniforms = MAX2(uniforms, 4);
   }

   if (uses_txf) {
      agx_usc_push_packed(&b, SAMPLER, dev->dev.txf_sampler);
   }

   /* For attachmentless rendering, we don't know the sample count until
    * draw-time. But we have trivial bg/eot programs in that case too.
    */
   if (key.tib.nr_samples >= 1) {
      agx_usc_push_packed(&b, SHARED, &key.tib.usc);
   } else {
      assert(key.tib.sample_size_B == 0);
      agx_usc_shared_none(&b);

      key.tib.nr_samples = 1;
   }

   /* Get the shader */
   key.reserved_preamble = uniforms;
   /* XXX: locking? */
   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(&dev->bg_eot, &key);
   if (!shader) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return (struct hk_bg_eot){0};
   }

   agx_usc_pack(&b, SHADER, cfg) {
      cfg.code = agx_usc_addr(&dev->dev, shader->ptr);
      cfg.unk_2 = 0;
   }

   agx_usc_pack(&b, REGISTERS, cfg)
      cfg.register_count = shader->info.nr_gprs;

   if (shader->info.has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code =
            agx_usc_addr(&dev->dev, shader->ptr + shader->info.preamble_offset);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   struct hk_bg_eot ret = {.usc = t.gpu};

   agx_pack(&ret.counts, COUNTS, cfg) {
      cfg.uniform_register_count = shader->info.push_count;
      cfg.preshader_register_count = shader->info.nr_preamble_gprs;
      cfg.texture_state_register_count = nr_tex;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(uses_txf ? 1 : 0, false);
   }

   return ret;
}

static struct hk_bg_eot
hk_build_bg_eot(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                bool store, bool partial_render, bool incomplete_render_area,
                struct hk_g17p_bg_eot_cache_entry *cache)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (likely(!dev->frame_anatomy)) {
      return hk_build_bg_eot_program(cmd, info, store, partial_render,
                                     incomplete_render_area, cache, NULL);
   }

   /* Frame anatomy profiler: carry the per-RT op summary with the program,
    * so a stream reports the main or partial variant it actually uses. */
   uint8_t ops[HK_MAX_RTS] = {0};
   struct hk_bg_eot program = hk_build_bg_eot_program(
      cmd, info, store, partial_render, incomplete_render_area, cache, ops);
   memcpy(program.anatomy_ops, ops, sizeof(ops));
   return program;
}

static void
hk_merge_render_iview(struct hk_rendering_state *render,
                      struct hk_image_view *iview, bool zls)
{
   if (iview) {
      unsigned samples = iview->vk.image->samples;
      /* TODO: is this right for ycbcr? */
      unsigned level = iview->vk.base_mip_level;
      unsigned width = u_minify(iview->vk.image->extent.width, level);
      unsigned height = u_minify(iview->vk.image->extent.height, level);

      assert(render->tilebuffer.nr_samples == 0 ||
             render->tilebuffer.nr_samples == samples);
      render->tilebuffer.nr_samples = samples;

      /* TODO: Is this merging logic sound? Not sure how this is supposed to
       * work conceptually.
       */
      render->cr.width = MAX2(render->cr.width, width);
      render->cr.height = MAX2(render->cr.height, height);

      if (zls) {
         render->cr.zls_width = width;
         render->cr.zls_height = height;
      }
   }
}

static void
hk_pack_zls_control(struct agx_zls_control_packed *packed,
                    struct ail_layout *z_layout, struct ail_layout *s_layout,
                    const VkRenderingAttachmentInfo *z,
                    const VkRenderingAttachmentInfo *s,
                    bool incomplete_render_area, bool partial_render)
{
   struct agx_zls zls = {0};

   if (z) {
      /* XXX: Dropping Z stores is wrong if the render pass gets split into
       * multiple control streams (can that ever happen?) We need more ZLS
       * variants. Force || true for now.
       */
      zls.z_store = z->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    z->resolveMode != VK_RESOLVE_MODE_NONE || partial_render ||
                    true;

      zls.z_load = z->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
                   incomplete_render_area;
   }

   if (s) {
      /* TODO:
       * Fail
       * dEQP-VK.renderpass.dedicated_allocation.formats.d32_sfloat_s8_uint.input.dont_care.store.self_dep_clear_draw_use_input_aspect
       * without the force
       * .. maybe a VkRenderPass emulation bug.
       */
      zls.s_store = s->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    s->resolveMode != VK_RESOLVE_MODE_NONE || partial_render ||
                    true;

      zls.s_load = s->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || partial_render ||
                   incomplete_render_area;
   }

   agx_pack_zls_control(packed, z_layout, s_layout, &zls);
}

/* Precise depth store (AGX3).
 *
 * hk_pack_zls_control() forces the depth store on ("|| true") because a pass
 * split across several of Honeykrisp's own control streams needs every stream
 * but the last to store depth for the next one to reload. That costs one
 * full depth write per tile even for storeOp DONT_CARE: up to 4 B/px of
 * DRAM writes per pass for D32 (less where G17P compresses the depth plane,
 * hk_g17p_compression.h). Only the stream that ends the pass may drop it:
 * the first stream keeps the forced word, middle streams keep the partial
 * word, and EndRendering installs one of these two:
 *
 *  - only: the pass has a single stream: loads per loadOp, depth stores per
 *    storeOp/resolve;
 *  - last: a later stream: loads forced (the earlier stream stored), depth
 *    stores per storeOp/resolve.
 *
 * Suspending passes keep the forced store (a later command buffer reloads).
 * Firmware partial renders (TVB overflow) are unaffected: the G17 kernel
 * derives their ZLS word with both load and store forced
 * (g17_uapi.rs partial_depth_flags). Stencil follows the same rule. The
 * CTS failure in hk_pack_zls_control()'s TODO
 * (renderpass ... d32_sfloat_s8_uint.input.dont_care.store.
 * self_dep_clear_draw_use_input_aspect) is the store of a stream that does
 * not end its pass: its self-dependency splits the pass, and the next stream
 * reads stencil from memory as an input attachment. That stream keeps the
 * forced word; only the stream that ends the pass gets this one.
 *
 * Untouched depth/stencil elision (AGX3).
 *
 * Minecraft opens every full-screen pass with loadOp LOAD / storeOp STORE on
 * its D32 attachment, including passes whose pipelines neither test nor
 * write depth (GUI). Such a pass leaves the attachment's contents exactly as
 * they were, so loading and storing them (up to 2 x 14.9 MB at 2408x1506;
 * less when agx3 compresses the D32 image) is pure traffic. When the pass
 * ends with nothing
 * having touched depth/stencil, drop the attachment from every control
 * stream of the pass: no ZLS load/store, and no depth base for the kernel,
 * whose partial-render word forces Z/S load+store whenever a base is present
 * (g17_uapi.rs partial_depth_flags), which would otherwise store tile depth
 * that was never loaded. Not eligible: CLEAR load ops (the clear itself
 * changes the attachment), resolves, suspended/resumed passes and passes
 * recorded in secondaries. Depth/stencil input-attachment reads inside the
 * pass cannot see the dropped tile contents, but G17P does not expose
 * dynamicRenderingLocalReadDepthStencilAttachments, and a separate subpass
 * reads the (unchanged) image from memory. */

static bool
hk_g17p_zls_attachment_elidable(const VkRenderingAttachmentInfo *att)
{
   return !att || !att->imageView ||
          (att->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR &&
           att->resolveMode == VK_RESOLVE_MODE_NONE);
}

/* A NONE/NONE attachment that no draw touches must retain its memory
 * contents. In particular, a full-area pass must not run the forced ZLS
 * store with an uninitialized tile. Drop such an attachment after recording
 * the pass, including all of its split streams and firmware partial renders.
 * Secondary queues are replayed into the primary by hk_CmdExecuteCommands(),
 * so the same draw/clear tracking observes their depth/stencil use. */
static bool
hk_g17p_zls_attachment_none(const VkRenderingAttachmentInfo *att)
{
   return !att || !att->imageView ||
          (att->loadOp == VK_ATTACHMENT_LOAD_OP_NONE &&
           att->storeOp == VK_ATTACHMENT_STORE_OP_NONE &&
           att->resolveMode == VK_RESOLVE_MODE_NONE);
}

static bool
hk_stencil_face_writes(const struct vk_stencil_test_face_state *face)
{
   return face->write_mask &&
          (face->op.fail != VK_STENCIL_OP_KEEP ||
           face->op.pass != VK_STENCIL_OP_KEEP ||
           face->op.depth_fail != VK_STENCIL_OP_KEEP);
}

static inline void
hk_g17p_note_depth_use(struct hk_cmd_buffer *cmd)
{
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   if (likely(!render->g17p_zls_elide_mask))
      return;
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   if (dyn->ds.depth.test_enable || dyn->ds.depth.write_enable ||
       dyn->ds.depth.bounds_test.enable)
      render->g17p_zls_touched |= VK_IMAGE_ASPECT_DEPTH_BIT;
   if (dyn->ds.stencil.test_enable)
      render->g17p_zls_touched |= VK_IMAGE_ASPECT_STENCIL_BIT;

   /* Depth is written only with the depth test enabled (Vulkan 1.4, "Depth
    * Test"), stencil only with the stencil test enabled, a nonzero write
    * mask and an operation other than KEEP for a face ("Stencil Test"). */
   if (dyn->ds.depth.test_enable && dyn->ds.depth.write_enable)
      render->g17p_zls_written |= VK_IMAGE_ASPECT_DEPTH_BIT;
   if (dyn->ds.stencil.test_enable &&
       (hk_stencil_face_writes(&dyn->ds.stencil.front) ||
        hk_stencil_face_writes(&dyn->ds.stencil.back)))
      render->g17p_zls_written |= VK_IMAGE_ASPECT_STENCIL_BIT;
}

void
hk_g17p_note_depth_clear(struct hk_cmd_buffer *cmd, VkImageAspectFlags aspects)
{
   cmd->state.gfx.render.g17p_zls_touched |= aspects;
   cmd->state.gfx.render.g17p_zls_written |= aspects;
}

/* Drop each untouched plane independently. Using stencil must not let an
 * intermediate stream store uninitialized depth from a NONE/NONE attachment,
 * and using depth must not do the same to an untouched stencil attachment.
 */
static void
hk_g17p_zls_elide(struct hk_cmd_buffer *cmd, VkImageAspectFlags aspects)
{
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   if (!aspects || !render->zls_first_cs)
      return;
   bool found = false;
   list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
      if (cs == render->zls_first_cs)
         found = true;
      if (!found || cs->type != HK_CS_VDM)
         continue;
      agx_unpack(NULL, &cs->cr.zls_control, ZLS_CONTROL, control);
      /* An absent attachment must also have zero layer strides. */
      if (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) {
         memset(&cs->cr.depth, 0, sizeof(cs->cr.depth));
         control.z_load = control.z_store = false;
         control.z_load_compress = control.z_store_compress = false;
         control.z_load_tiling = control.z_store_tiling = 0;
         control.z_format = 0;
         control.z_resolve = false;
      }
      if (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) {
         memset(&cs->cr.stencil, 0, sizeof(cs->cr.stencil));
         control.s_load = control.s_store = false;
         control.s_load_compress = control.s_store_compress = false;
         control.s_load_tiling = control.s_store_tiling = 0;
         control.s_resolve = false;
      }
      agx_pack(&cs->cr.zls_control, ZLS_CONTROL, cfg) {
         cfg = control;
      }
   }
}

/* An elidable aspect that the pass tested but no draw or clear could have
 * changed holds, at the end of the pass, exactly what the pass loaded or what
 * memory already held: drop its store from every stream. Earlier streams of
 * a split pass then reload from memory that still holds the same values, and
 * a firmware partial render, which forces load and store, writes back the
 * unchanged tile. */
static void
hk_g17p_zls_elide_store(struct hk_cmd_buffer *cmd, VkImageAspectFlags aspects)
{
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   if (!aspects || !render->zls_first_cs)
      return;
   bool found = false;
   list_for_each_entry(struct hk_cs, cs, &cmd->control_streams, node) {
      if (cs == render->zls_first_cs)
         found = true;
      if (!found || cs->type != HK_CS_VDM)
         continue;
      agx_unpack(NULL, &cs->cr.zls_control, ZLS_CONTROL, control);
      if (aspects & VK_IMAGE_ASPECT_DEPTH_BIT)
         control.z_store = false;
      if (aspects & VK_IMAGE_ASPECT_STENCIL_BIT)
         control.s_store = false;
      agx_pack(&cs->cr.zls_control, ZLS_CONTROL, cfg) {
         cfg = control;
      }
   }
}

/* G17P and M5: a colour attachment that a single-stream pass (single-sample
 * on G17P) loads and stores, but that no draw or clear wrote and no fragment
 * shader read, keeps its memory contents as they are. Its end-of-tile
 * stores, partial-render ones included, are dropped, and the main background
 * program clears its tile instead of loading it: the end-of-tile program
 * then carries no operation for it, as for storeOp DONT_CARE, and the
 * background program a load-op clear, as for loadOp CLEAR. The partial-render
 * background program still reloads it, from the memory no store touched.
 * The rebuild is exact only for attachments without clear, resolve or flags,
 * so other passes keep their programs. A continuation of the pass restores
 * the full programs (hk_m5_pass_merge_begin()). */
static void
hk_g17p_elide_untouched_color(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   const bool m5 = agx_chip_g17_m5(dev->dev.chip);
   /* The rebuilt programs may drop every colour store; M5 has an empty
    * end-of-tile program only for single-layer, non-spilling colour tiles
    * (hk_build_m5_empty_bg_eot()), multisampled ones included. */
   if ((dev->dev.chip != AGX_CHIP_G17P && !m5) ||
       cs != render->zls_first_cs || cmd->in_meta || render->color_read ||
       render->view_mask || render->layer_count > 1 ||
       (m5 ? !hk_m5_color_imageblock_supported(&render->tilebuffer)
           : (render->tilebuffer.nr_samples > 1 ||
              agx_tilebuffer_spills(&render->tilebuffer) ||
              render->g17p_direct_color_resolve)) ||
       (render->flags & (VK_RENDERING_SUSPENDING_BIT |
                         VK_RENDERING_RESUMING_BIT |
                         VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT)))
      return;

   VkRenderingAttachmentInfo atts[HK_MAX_RTS], bg_atts[HK_MAX_RTS];
   uint8_t elide = 0, covered = 0;
   for (uint32_t i = 0; i < render->color_att_count; ++i) {
      const struct hk_attachment *att = &render->color_att[i];
      if (att->load_op == VK_ATTACHMENT_LOAD_OP_CLEAR ||
          att->resolve_mode != VK_RESOLVE_MODE_NONE || att->flags)
         return;

      atts[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = hk_image_view_to_handle(att->iview),
         .imageLayout = att->layout,
         .loadOp = att->load_op,
         .storeOp = att->store_op,
      };
      bg_atts[i] = atts[i];
      if (att->iview && att->load_op == VK_ATTACHMENT_LOAD_OP_LOAD &&
          att->store_op == VK_ATTACHMENT_STORE_OP_STORE &&
          !(render->color_written & BITFIELD_BIT(i))) {
         atts[i].imageView = VK_NULL_HANDLE;
         bg_atts[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
         elide |= BITFIELD_BIT(i);
      } else if (att->iview && att->load_op == VK_ATTACHMENT_LOAD_OP_LOAD &&
                 (render->color_covered & BITFIELD_BIT(i))) {
         /* The first draw overwrites every sample before anything reads
          * it (hk_g17p_note_first_draw()): start the tile cleared instead of
          * loading it. The stores, and the partial render's reload of what
          * an earlier tile pass stored, are unchanged. */
         bg_atts[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
         covered |= BITFIELD_BIT(i);
      }
   }
   if (!elide && !covered)
      return;

   const bool incomplete_render_area =
      render->area.offset.x > 0 || render->area.offset.y > 0 ||
      render->area.extent.width < render->cr.width ||
      render->area.extent.height < render->cr.height;
   const VkRenderingInfo info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = render->area,
      .layerCount = render->layer_count,
      .viewMask = render->view_mask,
      .colorAttachmentCount = render->color_att_count,
      .pColorAttachments = atts,
   };
   VkRenderingInfo bg_info = info;
   bg_info.pColorAttachments = bg_atts;
   const struct hk_bg_eot bg_main = hk_build_bg_eot(
      cmd, &bg_info, false, false, incomplete_render_area, NULL);
   const struct hk_bg_eot eot_main =
      hk_build_bg_eot(cmd, &info, true, false, incomplete_render_area, NULL);
   const struct hk_bg_eot eot_partial =
      hk_build_bg_eot(cmd, &info, true, true, incomplete_render_area, NULL);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   cs->cr.bg.main = bg_main;
   cs->cr.eot.main = eot_main;
   cs->cr.eot.partial = eot_partial;
   cs->anatomy_covered = covered;
   perf_debug(cmd, "Dropped untouched colour attachments 0x%x, loads of "
              "covered attachments 0x%x", elide, covered);
}

/* G17P: a pass whose first draw is a direct, non-indexed, single-instance draw
 * of vertices 0..2, by a vertex shader that places them on a triangle covering
 * the clip-space square (hk_vs_covers_clip_square()), under a viewport and
 * scissor that cover the render area, writes every sample of each colour
 * attachment its fragment shader writes opaquely before anything can read it:
 * no test can reject a fragment, the fragment shader cannot kill one, and the
 * attachment's whole write mask is enabled without blending or logic ops.
 * D3D translation layers load every render target they were not told to clear
 * or discard, so their full-screen post-processing passes otherwise read each
 * target only to overwrite it. hk_g17p_elide_untouched_color() turns the
 * load of those attachments into a clear. */
static void
hk_g17p_note_first_draw(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_rendering_state *render = &gfx->render;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   render->first_draw_seen = true;
   render->color_covered = 0;
   if ((dev->dev.chip != AGX_CHIP_G17P && !agx_chip_g17_m5(dev->dev.chip)) ||
       cmd->in_meta ||
       cs != render->zls_first_cs || cs->stats.cmds || cs->m5_continued_pass ||
       hk_conditional_render_is_active(&cmd->state.conditional, cmd->in_meta,
                                       cmd->conditional_meta) ||
       agx_is_indirect(draw.b) || draw.indexed || draw.b.count[0] != 3 ||
       draw.b.count[1] != 1 || draw.start || draw.start_instance ||
       render->view_mask || render->layer_count > 1 ||
       gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] || gfx->shaders[MESA_SHADER_MESH])
      return;

   const struct hk_shader *vs = hk_bound_hw_vs(gfx);
   const struct hk_shader *fs = hk_bound_fs(cmd);
   if (!vs || !vs->info.vs.covers_clip_square || !fs ||
       fs->info.fs.may_kill || fs->b.info.reads_tib)
      return;

   const unsigned samples = MAX2(dyn->ms.rasterization_samples, 1);
   const VkPrimitiveTopology topology = dyn->ia.primitive_topology;
   if ((topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST &&
        topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP &&
        topology != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) ||
       dyn->rs.rasterizer_discard_enable ||
       dyn->rs.cull_mode != VK_CULL_MODE_NONE ||
       dyn->rs.polygon_mode != VK_POLYGON_MODE_FILL ||
       dyn->ms.alpha_to_coverage_enable || dyn->cb.logic_op_enable ||
       (dyn->ms.sample_mask & BITFIELD_MASK(samples)) != BITFIELD_MASK(samples))
      return;

   if ((render->depth_att.iview &&
        ((dyn->ds.depth.test_enable &&
          dyn->ds.depth.compare_op != VK_COMPARE_OP_ALWAYS) ||
         dyn->ds.depth.bounds_test.enable)) ||
       (render->stencil_att.iview && dyn->ds.stencil.test_enable))
      return;

   const VkRect2D area = render->area;
   const float ax0 = area.offset.x, ay0 = area.offset.y;
   const float ax1 = ax0 + area.extent.width, ay1 = ay0 + area.extent.height;
   if (dyn->vp.viewport_count != 1 || dyn->vp.scissor_count != 1)
      return;
   const VkViewport *vp = &dyn->vp.viewports[0];
   if (MIN2(vp->x, vp->x + vp->width) > ax0 ||
       MAX2(vp->x, vp->x + vp->width) < ax1 ||
       MIN2(vp->y, vp->y + vp->height) > ay0 ||
       MAX2(vp->y, vp->y + vp->height) < ay1)
      return;
   const VkRect2D *sc = &dyn->vp.scissors[0];
   if (sc->offset.x > area.offset.x || sc->offset.y > area.offset.y ||
       (int64_t)sc->offset.x + sc->extent.width < ax1 ||
       (int64_t)sc->offset.y + sc->extent.height < ay1)
      return;

   uint8_t covered = 0;
   for (uint32_t i = 0; i < render->color_att_count; ++i) {
      const struct hk_attachment *att = &render->color_att[i];
      /* A disabled colour write keeps the attachment (preserve_rt in the
       * epilog), so it is not covered even when its write mask is full. */
      if (!att->iview || !(gfx->fast_link_fs.color_written & BITFIELD_BIT(i)) ||
          !(dyn->cb.color_write_enables & BITFIELD_BIT(i)) ||
          dyn->cb.attachments[i].blend_enable)
         continue;
      const unsigned channels = util_format_get_nr_components(
         vk_format_to_pipe_format(att->iview->vk.format));
      if ((dyn->cb.attachments[i].write_mask & BITFIELD_MASK(channels)) ==
          BITFIELD_MASK(channels))
         covered |= BITFIELD_BIT(i);
   }
   render->color_covered = covered;
}

static void
hk_pack_zls_control_final(struct agx_zls_control_packed *packed,
                          struct ail_layout *z_layout,
                          struct ail_layout *s_layout,
                          const VkRenderingAttachmentInfo *z,
                          const VkRenderingAttachmentInfo *s,
                          bool incomplete_render_area, bool later_stream)
{
   struct agx_zls zls = {0};

   if (z) {
      zls.z_store = z->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    z->resolveMode != VK_RESOLVE_MODE_NONE;
      zls.z_load = z->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || later_stream ||
                   incomplete_render_area;
   }

   if (s) {
      zls.s_store = s->storeOp == VK_ATTACHMENT_STORE_OP_STORE ||
                    s->resolveMode != VK_RESOLVE_MODE_NONE;
      zls.s_load = s->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD || later_stream ||
                   incomplete_render_area;
   }

   agx_pack_zls_control(packed, z_layout, s_layout, &zls);
}

/* The tilebuffer layout for colour attachments of the given formats. G17P's
 * bounded RT0 programs use non-array image operations, so it lays out a
 * single layer unless the rendering is layered. */
static struct agx_tilebuffer_layout
hk_tilebuffer_layout(const struct hk_device *dev, const VkFormat *vk_formats,
                     size_t stride, unsigned count, unsigned samples,
                     bool layered, bool g17p_wide_ms_to_memory,
                     uint8_t resident_mask)
{
   enum pipe_format formats[HK_MAX_RTS] = {0};
   for (unsigned i = 0; i < count; ++i) {
      const VkFormat format =
         *(const VkFormat *)((const uint8_t *)vk_formats + i * stride);
      formats[i] = hk_format_to_pipe_format(format);
      if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
         formats[i] = hk_padded_image_format(formats[i]);
         const enum pipe_format integer = hk_scaled_texture_format(formats[i]);
         if (integer != PIPE_FORMAT_NONE)
            formats[i] = integer;
      }
   }

   /* Storage capacity and raster samples are independent limits. In
    * particular 16x16 at 4x may occupy the full 32 KiB, while 32x32 at 4x
    * exceeds the sample budget even for an eight-byte sample. */
   const bool imageblock = agx_chip_native_imageblock(dev->dev.chip);
   const bool m5 = agx_chip_g17_m5(dev->dev.chip);
   const uint8_t max_sample_size_B =
      m5 ? AGX_MAX_RENDER_TARGETS * 16 :
      !imageblock              ? 0
      : g17p_wide_ms_to_memory ? AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE
                               : AGX_G17P_MAX_BYTES_PER_SAMPLE;
   if (m5) {
      return agx_build_tilebuffer_layout_resident(
         formats, count, samples, layered, max_sample_size_B, 65536, 2048,
         resident_mask);
   }
   return agx_build_tilebuffer_layout_with_limits(
      formats, count, samples, layered, max_sample_size_B,
      imageblock ? 32768 : 0, agx_native_graphics(&dev->dev) ? 2048 : 0);
}

/* M5 compressed multisample colour attachments stay in the tilebuffer: a
 * spilled attachment is read back during the pass through its texture view,
 * and compressed reads do not observe the pass's own memory stores.
 * Compressed multisample targets are at most eight bytes, so all eight fit
 * the 4x sample budget (64 bytes) and do not spill when allocated first. */
static uint8_t
hk_m5_resident_attachments(const struct hk_device *dev,
                           const struct hk_rendering_state *render)
{
   if (!agx_chip_g17_m5(dev->dev.chip) || render->tilebuffer.nr_samples <= 1)
      return 0;

   uint8_t mask = 0;
   for (unsigned i = 0; i < render->color_att_count; ++i) {
      const struct hk_image_view *view = render->color_att[i].iview;
      if (!view)
         continue;
      const struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);
      if (image->planes[view->planes[0].image_plane].layout.compressed)
         mask |= BITFIELD_BIT(i);
   }
   return mask;
}

/*
 * G17P transfer writes (hk_sync_xfer_write in hk_sync.h): a render pass
 * begin or draw that may access a transfer destination no dependency orders
 * before it waits for the transfer's compute prefix.
 */

/* Make every noted write up to cs visible, close cs to further work and
 * order the prefix before graphics. */
static void
hk_xfer_order(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_sync_state *st = &cmd->sync;

   struct hk_cs *flush[HK_SYNC_XFER_WRITES + 1];
   unsigned n = 0;
   for (unsigned i = 0; i <= st->xfer_write_count; ++i) {
      struct hk_cs *w =
         i < st->xfer_write_count ? st->xfer_writes[i].cs : st->xfer_any;
      if (!w || w->sync.seq > cs->sync.seq)
         continue;
      bool seen = false;
      for (unsigned j = 0; j < n; ++j)
         seen |= flush[j] == w;
      if (!seen)
         flush[n++] = w;
   }

   /* Each launch ends in a weak ordering record; strengthen the last one
    * into a cache flush, as a barrier does (hk_CmdPipelineBarrier2()). */
   const bool alias_scope = hk_cdm_alias_scope(dev);
   for (unsigned i = 0; i < n; ++i) {
      struct hk_cs *w = flush[i];
      if (hk_cdm_visibility_flush_needed(dev, w, alias_scope)) {
         hk_ensure_cs_has_space(cmd, w, AGX3_CDM_BARRIER_SIZE);
         hk_cdm_visibility_flush(dev, w, alias_scope);
      }
   }

   perf_debug(cmd, "Ordering an unsynchronized transfer before graphics");
   if (cmd->current_cs.cs == cs)
      hk_cmd_buffer_end_compute(cmd);
   hk_sync_order_xfer(cmd, cs);
}

static inline struct hk_cs *
hk_xfer_match_image(const struct hk_cmd_buffer *cmd,
                    const struct hk_image_view *iview, struct hk_cs *best)
{
   if (!iview)
      return best;
   uint64_t lo, hi;
   hk_image_va_range(container_of(iview->vk.image, struct hk_image, vk), &lo,
                     &hi);
   return lo < hi ? hk_sync_xfer_match(cmd, lo, hi, best) : best;
}

/* A render pass instance loads, stores and resolves its attachments. */
static void
hk_xfer_check_attachments(struct hk_cmd_buffer *cmd)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct hk_cs *best = NULL;

   for (unsigned i = 0; i < render->color_att_count; ++i) {
      best = hk_xfer_match_image(cmd, render->color_att[i].iview, best);
      best = hk_xfer_match_image(cmd, render->color_att[i].resolve_iview, best);
   }
   best = hk_xfer_match_image(cmd, render->depth_att.iview, best);
   best = hk_xfer_match_image(cmd, render->depth_att.resolve_iview, best);
   best = hk_xfer_match_image(cmd, render->stencil_att.iview, best);
   best = hk_xfer_match_image(cmd, render->stencil_att.resolve_iview, best);

   if (best)
      hk_xfer_order(cmd, best);
}

/* The address a Gen9 texture or PBE word at desc points at (Address, bits
 * 64..107, in units of 16 bytes). */
static inline uint64_t
hk_xfer_gen9_texture_address(const uint8_t *desc)
{
   uint64_t word;
   memcpy(&word, desc + 8, sizeof(word));
   return (word & BITFIELD64_MASK(44)) << 4;
}

static inline struct hk_cs *
hk_xfer_match_point(const struct hk_cmd_buffer *cmd, uint64_t addr,
                    struct hk_cs *best)
{
   return addr ? hk_sync_xfer_match(cmd, addr, addr + 1, best) : best;
}

/* Descriptors one check reads at most before treating the bound sets as
 * accessing everything. */
#define HK_XFER_MAX_DESCRIPTORS 1024

/* The latest noted stream the descriptors bound for graphics may access.
 * What recording cannot see (push descriptors, sets that may be updated
 * after binding, mutable or unknown types) may access everything. */
static struct hk_cs *
hk_xfer_match_descriptors(const struct hk_cmd_buffer *cmd, struct hk_cs *best)
{
   const struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   const unsigned sampled_stride = hk_sampled_image_descriptor_layout(9).stride;
   unsigned budget = HK_XFER_MAX_DESCRIPTORS;

   if (desc->push_active)
      return hk_sync_xfer_match_all(cmd, best);

   for (unsigned s = 0; s < HK_MAX_SETS; ++s) {
      const struct hk_descriptor_set *set = desc->sets[s];
      if (!set)
         continue;

      const struct hk_descriptor_set_layout *layout = set->layout;
      if (layout->descriptor_generation != 9)
         return hk_sync_xfer_match_all(cmd, best);

      for (unsigned b = 0; b < layout->binding_count; ++b) {
         const struct hk_descriptor_set_binding_layout *bl = &layout->binding[b];
         if (bl->flags & VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT)
            return hk_sync_xfer_match_all(cmd, best);

         switch (bl->type) {
         case VK_DESCRIPTOR_TYPE_SAMPLER:
         case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK:
         /* Read from the root below. */
         case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
         case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            continue;
         case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
         case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
         case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
         case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
         case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
         case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            break;
         default:
            return hk_sync_xfer_match_all(cmd, best);
         }

         if (!bl->stride || bl->offset >= set->size)
            continue;
         const uint32_t count =
            MIN2(bl->array_size, (set->size - bl->offset) / bl->stride);
         if (count > budget)
            return hk_sync_xfer_match_all(cmd, best);
         budget -= count;

         const uint8_t *base = (const uint8_t *)set->mapped_ptr + bl->offset;
         for (uint32_t e = 0; e < count; ++e) {
            const uint8_t *d = base + (size_t)e * bl->stride;
            switch (bl->type) {
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
               /* One record per plane. */
               for (unsigned p = 0; (p + 1) * sampled_stride <= bl->stride; ++p)
                  best = hk_xfer_match_point(
                     cmd, hk_xfer_gen9_texture_address(d + p * sampled_stride),
                     best);
               break;
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
               best = hk_xfer_match_point(
                  cmd, hk_xfer_gen9_texture_address(d), best);
               best = hk_xfer_match_point(
                  cmd, hk_xfer_gen9_texture_address(d + HK_TEXTURE_DESCRIPTOR_AGX3_SIZE),
                  best);
               break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
               best = hk_xfer_match_point(
                  cmd, hk_xfer_gen9_texture_address(d), best);
               if (bl->stride >= AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET +
                                    sizeof(struct hk_g17p_buffer_texture_metadata)) {
                  struct hk_g17p_buffer_texture_metadata meta;
                  memcpy(&meta, d + AGX_G17P_BUFFER_TEXTURE_METADATA_OFFSET,
                         sizeof(meta));
                  best = hk_xfer_match_point(cmd, meta.address, best);
               }
               break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
               struct hk_buffer_address addr;
               memcpy(&addr, d, sizeof(addr));
               if (addr.size)
                  best = hk_sync_xfer_match(cmd, addr.base_addr,
                                            addr.base_addr + addr.size, best);
               break;
            }
            case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR: {
               uint64_t addr;
               memcpy(&addr, d, sizeof(addr));
               best = hk_xfer_match_point(cmd, addr, best);
               break;
            }
            default:
               UNREACHABLE("filtered above");
            }
         }
      }
   }

   for (unsigned i = 0; i < desc->dynamic_buffer_end; ++i) {
      const struct hk_buffer_address *addr = &desc->root.dynamic_buffers[i];
      if (addr->size)
         best = hk_sync_xfer_match(cmd, addr->base_addr,
                                   addr->base_addr + addr->size, best);
   }

   return best;
}

/* Before a draw records anything: whether its graphics work may access a
 * noted transfer write, through the bound descriptors, vertex and index
 * buffers, [extra_lo, extra_hi) (indirect parameters) or the conditional
 * rendering predicate. Shader stores to transform feedback buffers are
 * matched against everything. */
static void
hk_xfer_check_draw_slow(struct hk_cmd_buffer *cmd, uint64_t extra_lo,
                        uint64_t extra_hi)
{
   struct hk_sync_state *st = &cmd->sync;
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_cs *best = NULL;

   if (extra_lo < extra_hi)
      best = hk_sync_xfer_match(cmd, extra_lo, extra_hi, best);
   if (cmd->state.conditional.active)
      best = hk_xfer_match_point(cmd, cmd->state.conditional.address, best);
   if (gfx->xfb_enabled)
      best = hk_sync_xfer_match_all(cmd, best);

   const bool clean =
      st->xfer_clean.valid && st->xfer_clean.version == st->xfer_version &&
      st->xfer_clean.descriptors == gfx->descriptors.generation &&
      st->xfer_clean.vertex_buffers == gfx->vertex_buffer_generation &&
      st->xfer_clean.index == gfx->index.buffer.addr;

   if (!clean) {
      struct hk_cs *state = hk_xfer_match_descriptors(cmd, NULL);
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->vb); ++i) {
         if (gfx->vb[i].range)
            state = hk_sync_xfer_match(cmd, gfx->vb[i].addr,
                                       gfx->vb[i].addr + gfx->vb[i].range,
                                       state);
      }
      if (gfx->index.buffer.range)
         state = hk_sync_xfer_match(
            cmd, gfx->index.buffer.addr,
            gfx->index.buffer.addr + gfx->index.buffer.range, state);

      st->xfer_clean.valid = state == NULL;
      st->xfer_clean.version = st->xfer_version;
      st->xfer_clean.descriptors = gfx->descriptors.generation;
      st->xfer_clean.vertex_buffers = gfx->vertex_buffer_generation;
      st->xfer_clean.index = gfx->index.buffer.addr;
      best = hk_sync_later(best, state);
   }

   if (best)
      hk_xfer_order(cmd, best);
}

static inline void
hk_xfer_check_draw(struct hk_cmd_buffer *cmd, uint64_t extra_lo,
                   uint64_t extra_hi)
{
   if (unlikely(hk_sync_xfer_pending(&cmd->sync)) && !cmd->in_meta)
      hk_xfer_check_draw_slow(cmd, extra_lo, extra_hi);
}

/*
 * AGX3 render pass continuation.
 *
 * Two consecutive render pass instances on the same attachments, where the
 * second loads what the first stored and nothing but state binding and
 * framebuffer-local pipeline barriers is recorded between them, are one tile
 * pass to the hardware. The second instance's draws then continue in the
 * first instance's render stream, which drops the first instance's store and
 * the second instance's reload of every attachment, and one command boundary.
 *
 * Within one render pass instance the attachment writes of earlier draws are
 * visible to the attachment operations of later draws, which is exactly what
 * a framebuffer-local dependency between the two instances asks for: one
 * whose second scope is the fragment-test and colour-output stages with
 * attachment accesses only. A dependency whose first scope does not include
 * the first instance's work (compute, transfer) is a requirement on the
 * merged stream, applied before it continues (hk_sync_merge_rendering()).
 * Any other graphics-to-graphics dependency without a layout transition or
 * queue family transfer, up to ALL_COMMANDS -> ALL_COMMANDS with every memory
 * access, makes the continuation "ordered": it is taken only where the merged
 * pass is memory-equivalent to the two instances by construction
 * (hk_m5_ordered_refusal(), hk_m5_ordered_draw_hazard()). Work recorded
 * between the instances, a timestamp, or a requirement that would order the
 * stream after a later one end the candidacy; those are detected through
 * the stream list, the sync epoch and the stream's own state
 * (hk_m5_pass_merge_begin()).
 */
static bool
hk_m5_pass_merge_candidate(struct hk_cmd_buffer *cmd, const struct hk_cs *cs)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   const unsigned samples = render->tilebuffer.nr_samples;
   const bool msaa = samples > 1;
   const bool msaa_enabled = msaa && dev->dev.chip == AGX_CHIP_G17S &&
      (samples == 2 || samples == 4) &&
      debug_get_option_m5_msaa_pass_merge();

   /* The instance is one ordinary render stream with draws, with no
    * pre/post-graphics streams, heap use or timestamps attached. */
   if ((!agx_chip_native_agx3(dev->dev.chip) &&
        !agx_chip_g17_m5(dev->dev.chip)) ||
       cmd->in_meta || cs->stats.cmds == 0 || cmd->sync.pass_vdm != cs ||
       cmd->current_cs.pre_gfx || cmd->current_cs.post_gfx ||
       cmd->uses_heap || cs->timestamp.start.handle ||
       cs->timestamp.end.handle) {
      if (msaa_enabled)
         hk_frame_anatomy_note(dev,
            "MSAA pass merge refused: pre=%u post=%u heap=%u draws=%u timestamp=%u",
            cmd->current_cs.pre_gfx != NULL, cmd->current_cs.post_gfx != NULL,
            cmd->uses_heap, cs->stats.cmds,
            cs->timestamp.start.handle != 0 || cs->timestamp.end.handle != 0);
      return false;
   }

   if (render->flags & (VK_RENDERING_SUSPENDING_BIT |
                        VK_RENDERING_RESUMING_BIT |
                        VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT))
      return false;

   /* In-tilebuffer rendering without resolves. Multisample continuation
    * keeps the existing per-sample partial store and reload programs. */
   if (render->view_mask || (msaa && !msaa_enabled) ||
       agx_tilebuffer_spills(&render->tilebuffer) ||
       render->g17p_direct_color_resolve)
      return false;
   /* Compressed attachments continue like uncompressed ones: the merged
    * render keeps the first instance's load and the last instance's store,
    * and a partial render stores and reloads them as a single instance's
    * would. */
   if (msaa && (render->layer_count != 1 || cs->tib.nr_samples != samples))
      return false;

   unsigned attachments = 0;
   for (unsigned i = 0; i < render->color_att_count; ++i) {
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE)
         return false;
      if (msaa && render->color_att[i].iview &&
          render->color_att[i].iview->vk.image->samples != samples)
         return false;
      attachments += render->color_att[i].iview != NULL;
   }
   if (render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE ||
       render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      return false;
   if (msaa &&
       ((render->depth_att.iview &&
         render->depth_att.iview->vk.image->samples != samples) ||
        (render->stencil_att.iview &&
         render->stencil_att.iview->vk.image->samples != samples)))
      return false;
   attachments += render->depth_att.iview != NULL;
   attachments += render->stencil_att.iview != NULL;
   return attachments > 0;
}

/* Whether the attachment of a beginning instance continues the one the
 * ended instance had in the same slot: the same view (or none in both),
 * loaded, stored the same way, unresolved, with the same flags. */
static const char *
hk_m5_attachment_continues(const VkRenderingAttachmentInfo *info,
                           const struct hk_attachment *att,
                           unsigned expected_samples)
{
   const bool present = info != NULL && info->imageView != VK_NULL_HANDLE;
   if (present != (att->iview != NULL))
      return "presence";
   if (!present)
      return NULL;

   VK_FROM_HANDLE(hk_image_view, iview, info->imageView);
   if (iview != att->iview)
      return "view";
   if (expected_samples > 1) {
      if (iview->vk.image->samples != expected_samples)
         return "sample count";
      if (info->imageLayout != att->layout)
         return "layout";
   }
   if (info->loadOp != VK_ATTACHMENT_LOAD_OP_LOAD)
      return "load op";
   if (info->storeOp != att->store_op)
      return "store op";
   if (info->resolveMode != VK_RESOLVE_MODE_NONE)
      return "resolve";
   if (vk_get_rendering_attachment_flags(info) != att->flags)
      return "flags";
   return NULL;
}

/* How one barrier of a dependency recorded between the instances relates to
 * continuing the first instance's stream with the second. */
enum hk_m5_merge_dep {
   /* It does not order the second instance's graphics work after the first
    * instance's, or only for attachment operations, which the tile pass
    * orders. */
   HK_M5_MERGE_DEP_LOCAL,
   /* It orders the second instance's work after the first instance's beyond
    * attachment operations (up to ALL_COMMANDS -> ALL_COMMANDS with every
    * memory access). One tile pass runs the second instance's vertex work
    * before the first instance's fragment work and stores the attachments
    * only at its end, so this holds only for a merge that is
    * memory-equivalent by construction, see hk_m5_ordered_refusal(). */
   HK_M5_MERGE_DEP_ORDERED,
   /* A layout transition or queue family ownership transfer between two
    * graphics scopes: never continued across. */
   HK_M5_MERGE_DEP_END,
};

static enum hk_m5_merge_dep
hk_m5_pass_merge_dependency(VkPipelineStageFlags2 src_stages,
                            VkPipelineStageFlags2 dst_stages,
                            VkAccessFlags2 dst_access, bool buffer,
                            bool transition)
{
   const VkPipelineStageFlags2 local_stages =
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
   const VkAccessFlags2 local_access =
      VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_2_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT;

   const bool from_gfx = hk_sync_kinds_src(src_stages) & HK_SYNC_GFX;
   const bool to_gfx =
      hk_sync_kinds_dst(dst_stages) & (HK_SYNC_GFX | HK_SYNC_GFX_FRAG);
   if (!from_gfx || !to_gfx)
      return HK_M5_MERGE_DEP_LOCAL;

   if (!buffer && !(dst_stages & ~local_stages) &&
       !(dst_access & ~local_access))
      return HK_M5_MERGE_DEP_LOCAL;

   return transition ? HK_M5_MERGE_DEP_END : HK_M5_MERGE_DEP_ORDERED;
}

static bool hk_m5_image_is_attachment(const struct hk_rendering_state *render,
                                      const struct vk_image *image);

static void
hk_m5_pass_merge_classify(struct hk_cmd_buffer *cmd, const char *what,
                          VkPipelineStageFlags2 src, VkPipelineStageFlags2 dst,
                          VkAccessFlags2 dst_access, bool buffer,
                          bool transition, const struct vk_image *image)
{
   struct hk_pass_merge_candidate *m = &cmd->state.pass_merge;
   if (!m->valid)
      return;

   /* hk performs no work for a layout transition: of an image that is not an
    * attachment of the candidate and cannot alias one, it only orders the
    * image's accesses, as the barrier's scopes describe. A transition of an
    * attachment, or an ownership transfer, changes what the continued
    * instance would load. */
   const enum agx_chip chip = hk_cmd_buffer_device(cmd)->dev.chip;
   if (transition && image &&
       (chip == AGX_CHIP_G17P || agx_chip_g17_m5(chip)) &&
       !hk_m5_image_is_attachment(&m->render, image) &&
       !(image->create_flags & (VK_IMAGE_CREATE_ALIAS_BIT |
                                VK_IMAGE_CREATE_SPARSE_BINDING_BIT)))
      transition = false;
   if (m->render.tilebuffer.nr_samples > 1 && transition) {
      m->valid = false;
      hk_frame_anatomy_note(hk_cmd_buffer_device(cmd),
                            "MSAA pass merge ended by a layout or ownership transition");
      return;
   }

   switch (hk_m5_pass_merge_dependency(src, dst, dst_access, buffer,
                                       transition)) {
   case HK_M5_MERGE_DEP_LOCAL:
      return;
   case HK_M5_MERGE_DEP_ORDERED:
      if (!m->ordered)
         hk_frame_anatomy_note(hk_cmd_buffer_device(cmd),
                               "pass-merge ordered: %s barrier src=0x%" PRIx64
                               " dst=0x%" PRIx64 " dst_access=0x%" PRIx64,
                               what, (uint64_t)src, (uint64_t)dst,
                               (uint64_t)dst_access);
      m->ordered = true;
      return;
   case HK_M5_MERGE_DEP_END:
      m->valid = false;
      hk_frame_anatomy_note(hk_cmd_buffer_device(cmd),
                            "pass-merge end: %s barrier src=0x%" PRIx64
                            " dst=0x%" PRIx64 " dst_access=0x%" PRIx64
                            " with a layout transition or ownership transfer",
                            what, (uint64_t)src, (uint64_t)dst,
                            (uint64_t)dst_access);
      return;
   }
}

void
hk_m5_pass_merge_barrier(struct hk_cmd_buffer *cmd,
                         const VkDependencyInfo *info)
{
   for (uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
      const VkMemoryBarrier2 *b = &info->pMemoryBarriers[i];
      hk_m5_pass_merge_classify(cmd, "memory", b->srcStageMask,
                                b->dstStageMask, b->dstAccessMask, false,
                                false, NULL);
   }
   for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) {
      const VkBufferMemoryBarrier2 *b = &info->pBufferMemoryBarriers[i];
      hk_m5_pass_merge_classify(
         cmd, "buffer", b->srcStageMask, b->dstStageMask, b->dstAccessMask,
         true, b->srcQueueFamilyIndex != b->dstQueueFamilyIndex, NULL);
   }
   for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
      const VkImageMemoryBarrier2 *b = &info->pImageMemoryBarriers[i];
      VK_FROM_HANDLE(vk_image, image, b->image);
      hk_m5_pass_merge_classify(
         cmd, "image", b->srcStageMask, b->dstStageMask, b->dstAccessMask,
         false,
         b->oldLayout != b->newLayout ||
            b->srcQueueFamilyIndex != b->dstQueueFamilyIndex,
         b->srcQueueFamilyIndex != b->dstQueueFamilyIndex ? NULL : image);
   }
}

/* Whether the image of an attachment view may hold the attachment's contents
 * through a different resource: an aliased image (only defined across
 * VK_IMAGE_CREATE_ALIAS_BIT) or sparse bindings. */
static bool
hk_m5_attachment_image_aliasable(const struct hk_image_view *iview)
{
   return iview && (iview->vk.image->create_flags &
                    (VK_IMAGE_CREATE_ALIAS_BIT |
                     VK_IMAGE_CREATE_SPARSE_BINDING_BIT));
}

/* Whether an image may be one of the attachments of the open instance. */
static bool
hk_m5_image_is_attachment(const struct hk_rendering_state *render,
                          const struct vk_image *image)
{
   for (unsigned i = 0; i < render->color_att_count; ++i) {
      if (render->color_att[i].iview &&
          render->color_att[i].iview->vk.image == image)
         return true;
   }
   return (render->depth_att.iview &&
           render->depth_att.iview->vk.image == image) ||
          (render->stencil_att.iview &&
           render->stencil_att.iview->vk.image == image);
}

/* Whether an attachment of the continuing instance may be in a rendering
 * feedback loop: its contents read through a descriptor during the pass,
 * which only works from memory the first instance has not stored yet. */
static const char *
hk_m5_ordered_attachment_refusal(const VkRenderingAttachmentInfo *info,
                                 const struct hk_attachment *att)
{
   if (!info || info->imageView == VK_NULL_HANDLE)
      return NULL;
   if (info->imageLayout != att->layout)
      return "layout";
   if (info->imageLayout == VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT)
      return "feedback loop layout";
   if (att->flags & VK_RENDERING_ATTACHMENT_INPUT_ATTACHMENT_FEEDBACK_BIT_KHR)
      return "input attachment feedback";
   const VkAttachmentFeedbackLoopInfoEXT *loop = vk_find_struct_const(
      info->pNext, ATTACHMENT_FEEDBACK_LOOP_INFO_EXT);
   if (loop && loop->feedbackLoopEnable)
      return "feedback loop";
   if (hk_m5_attachment_image_aliasable(att->iview))
      return "aliasable image";
   return NULL;
}

/* An image subresource accessed through a descriptor must be in the layout
 * the descriptor was written with, and no sampled-image, combined, storage or
 * input-attachment descriptor may name an attachment-only layout
 * (VkDescriptorImageInfo::imageLayout valid usage). */
static bool
hk_m5_layout_excludes_descriptors(VkImageLayout layout)
{
   switch (layout) {
   case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
   case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
   case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
   case VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL:
   case VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL:
      return true;
   default:
      return false;
   }
}

/* Whether no descriptor can access an attachment of the instance while it
 * renders: every attachment is in an attachment-only layout. Both instances
 * of a continuation use the same layouts (hk_m5_ordered_attachment_refusal()),
 * so this holds for the merged pass. Taken on G17P and M5. */
static bool
hk_m5_attachments_exclude_descriptors(const struct hk_cmd_buffer *cmd,
                                      const VkRenderingInfo *info)
{
   const enum agx_chip chip = hk_cmd_buffer_device(cmd)->dev.chip;
   if (chip != AGX_CHIP_G17P && !agx_chip_g17_m5(chip))
      return false;

   const VkRenderingAttachmentInfo *ds[] = {info->pDepthAttachment,
                                            info->pStencilAttachment};
   for (unsigned i = 0; i < ARRAY_SIZE(ds); ++i) {
      if (ds[i] && ds[i]->imageView &&
          !hk_m5_layout_excludes_descriptors(ds[i]->imageLayout))
         return false;
   }
   for (uint32_t i = 0; i < info->colorAttachmentCount; ++i) {
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[i];
      if (att->imageView && !hk_m5_layout_excludes_descriptors(att->imageLayout))
         return false;
   }
   return true;
}

/*
 * An ordered continuation (HK_M5_MERGE_DEP_ORDERED) is memory-equivalent to
 * the two instances when:
 *
 *  - the first instance's stream wrote no memory but its attachments: no
 *    graphics program in it writes external memory (storage buffers and
 *    images, atomics, transform feedback), and it has no pre/post-graphics
 *    streams (hk_m5_pass_merge_candidate()). The second instance's vertex
 *    work, which the tile pass runs before the first instance's fragment
 *    work, then reads what it would have read after it;
 *  - the second instance writes no memory but its attachments either, so
 *    nothing the first instance reads can change under it; every draw of the
 *    continuation is checked (hk_m5_ordered_draw_hazard()) and the stream
 *    ends before one that could;
 *  - no shader of the second instance can read the attachments' memory,
 *    which the first instance's store would have written and the merged
 *    pass writes only at its end: the attachments are in the same layouts in
 *    both instances, none is in a feedback loop, none can alias another
 *    resource, and either every attachment is in an attachment-only layout,
 *    which no descriptor may name (hk_m5_attachments_exclude_descriptors()),
 *    or no descriptor set (whose contents this command buffer cannot see) is
 *    bound for graphics; no graphics push descriptor recorded so far names an
 *    attachment image, and later ones, or later set binds without the layout
 *    proof, end the stream (hk_m5_pass_merge_note_push(),
 *    hk_m5_pass_merge_note_set_bind()).
 *
 * Attachment accesses themselves are ordered by the tile pass, as for a
 * framebuffer-local dependency.
 */
static const char *
hk_m5_ordered_refusal(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info,
                      const struct hk_rendering_state *prev,
                      const struct hk_cs *cs)
{
   const struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   if (cs->m5_shader_writes_memory)
      return "first instance writes memory";
   if (cmd->state.gfx.xfb_enabled)
      return "transform feedback";
   if (cmd->state.conditional.active)
      return "conditional rendering";

   const char *why =
      hk_m5_ordered_attachment_refusal(info->pDepthAttachment,
                                       &prev->depth_att);
   if (!why)
      why = hk_m5_ordered_attachment_refusal(info->pStencilAttachment,
                                             &prev->stencil_att);
   for (uint32_t i = 0; !why && i < info->colorAttachmentCount; ++i)
      why = hk_m5_ordered_attachment_refusal(&info->pColorAttachments[i],
                                             &prev->color_att[i]);
   if (why)
      return why;

   if (!hk_m5_attachments_exclude_descriptors(cmd, info)) {
      for (unsigned s = 0; s < HK_MAX_SETS; ++s) {
         if (desc->sets[s] && !(desc->push_active & BITFIELD_BIT(s)))
            return "descriptor set bound";
      }
   }

   if (cmd->m5_pushed_images) {
      set_foreach(cmd->m5_pushed_images, entry) {
         if (hk_m5_image_is_attachment(prev, entry->key))
            return "attachment pushed as a descriptor";
      }
   }

   return NULL;
}

void
hk_m5_pass_merge_end_ordered(struct hk_cmd_buffer *cmd, const char *reason)
{
   struct hk_cs *cs = cmd->current_cs.gfx;
   if (likely(!cs || !cs->m5_ordered))
      return;

   perf_debug(cmd, "Ending an ordered render continuation: %s", reason);
   cmd->state.m5_ordered_split = false;
   hk_cmd_buffer_end_graphics(cmd);
}

void
hk_m5_pass_merge_note_push(struct hk_cmd_buffer *cmd,
                           struct hk_image_view *iview)
{
   struct vk_image *image = iview->vk.image;
   if (!(image->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
       !(image->stencil_usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))
      return;

   if (!cmd->m5_pushed_images) {
      cmd->m5_pushed_images = _mesa_pointer_set_create(NULL);
      if (!cmd->m5_pushed_images) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return;
      }
   }
   _mesa_set_add(cmd->m5_pushed_images, image);

   struct hk_cs *cs = cmd->current_cs.gfx;
   if (unlikely(cs && cs->m5_ordered) &&
       hk_m5_image_is_attachment(&cmd->state.gfx.render, image))
      cmd->state.m5_ordered_split = true;
}

void
hk_m5_pass_merge_note_set_bind(struct hk_cmd_buffer *cmd)
{
   struct hk_cs *cs = cmd->current_cs.gfx;
   if (unlikely(cs && cs->m5_ordered && !cs->m5_layout_proof))
      cmd->state.m5_ordered_split = true;
}

/* Whether the next draw recorded into an ordered continuation could break
 * its memory equivalence (hk_m5_ordered_refusal()): it ends the stream
 * first, and the draw continues in a new stream ordered after it. */
static const char *
hk_m5_ordered_draw_hazard(struct hk_cmd_buffer *cmd)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;

   if (cmd->state.m5_ordered_split)
      return "descriptor update";
   if (gfx->xfb_enabled)
      return "transform feedback";

   /* Software and mesh stages run user programs in pre-graphics compute,
    * ahead of the whole merged pass; adjacency is materialised there too. */
   if (gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY] || gfx->shaders[MESA_SHADER_TASK] ||
       gfx->shaders[MESA_SHADER_MESH])
      return "software or mesh stages";
   if (mesa_prim_has_adjacency(vk_topology_to_mesa(
          cmd->vk.dynamic_graphics_state.ia.primitive_topology)))
      return "adjacency";

   for (unsigned stage = 0; stage < ARRAY_SIZE(gfx->shaders); ++stage) {
      struct hk_api_shader *api = gfx->shaders[stage];
      if (!api)
         continue;
      hk_foreach_variant(api, v) {
         if (v->info.writes_external_memory)
            return "shader writes memory";
      }
   }
   return NULL;
}

/* Continue the ended instance's render stream with the instance beginning
 * with info when everything above holds. Returns false, with the candidacy
 * ended, when the instance begins normally instead. */
static bool
hk_m5_pass_merge_begin(struct hk_cmd_buffer *cmd, const VkRenderingInfo *info)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_pass_merge_candidate *m = &cmd->state.pass_merge;
   struct hk_cs *cs = m->cs;
   const struct hk_rendering_state *prev = &m->render;
   struct hk_rendering_state *render = &cmd->state.gfx.render;
   const char *why = NULL;

   m->valid = false;

   /* Nothing was recorded or ordered since the instance ended: its stream is
    * still the latest, untouched, no stream was created, the open compute
    * stream (if any) did not grow, and no dependency, timestamp or pass
    * bookkeeping changed except through hk_m5_pass_merge_barrier(). */
   if (cmd->in_meta || cmd->state.deferred_clear_count ||
       cmd->current_cs.gfx || cmd->current_cs.pre_gfx ||
       cmd->current_cs.post_gfx || cmd->sync.in_pass ||
       cmd->sync.pending_start_ts.handle)
      why = "state";
   else if (list_is_empty(&cmd->control_streams) ||
            list_last_entry(&cmd->control_streams, struct hk_cs, node) != cs)
      why = "later stream";
   else if (cmd->sync.next_seq != m->next_seq ||
            cmd->sync.epoch != m->sync_epoch)
      why = "dependency";
   else if (cmd->current_cs.cs != m->open_cs ||
            (m->open_cs && m->open_cs->current != m->open_cs_current))
      why = "compute";
   else if (cs->current != m->after_current || cs->end != m->end ||
            cs->timestamp.start.handle || cs->timestamp.end.handle)
      why = "stream";
   else if (info->flags & (VK_RENDERING_SUSPENDING_BIT |
                           VK_RENDERING_RESUMING_BIT |
                           VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT))
      why = "flags";
   else if (memcmp(&info->renderArea, &prev->area, sizeof(info->renderArea)) ||
            info->layerCount != prev->layer_count ||
            info->viewMask != prev->view_mask ||
            info->colorAttachmentCount != prev->color_att_count) {
      hk_frame_anatomy_note(
         dev,
         "pass-merge end: geometry %d,%d %ux%u layers=%u views=0x%x rts=%u -> "
         "%d,%d %ux%u layers=%u views=0x%x rts=%u",
         prev->area.offset.x, prev->area.offset.y, prev->area.extent.width,
         prev->area.extent.height, prev->layer_count, prev->view_mask,
         prev->color_att_count, info->renderArea.offset.x,
         info->renderArea.offset.y, info->renderArea.extent.width,
         info->renderArea.extent.height, info->layerCount, info->viewMask,
         info->colorAttachmentCount);
      return false;
   }

   const char *slot = NULL;
   if (!why) {
      slot = "depth";
      why = hk_m5_attachment_continues(info->pDepthAttachment,
                                       &prev->depth_att,
                                       prev->tilebuffer.nr_samples);
   }
   if (!why) {
      slot = "stencil";
      why = hk_m5_attachment_continues(info->pStencilAttachment,
                                       &prev->stencil_att,
                                       prev->tilebuffer.nr_samples);
   }
   for (uint32_t i = 0; !why && i < info->colorAttachmentCount; ++i) {
      slot = "color";
      why = hk_m5_attachment_continues(&info->pColorAttachments[i],
                                       &prev->color_att[i],
                                       prev->tilebuffer.nr_samples);
   }

   if (why) {
      hk_frame_anatomy_note(dev, "pass-merge end: %s%s%s", slot ? slot : "",
                            slot ? " " : "", why);
      return false;
   }

   /* An ordered dependency, or one recorded before an earlier continuation
    * of this stream, needs the merged pass to be memory-equivalent. */
   const bool ordered = m->ordered || cs->m5_ordered;
   if (ordered) {
      why = hk_m5_ordered_refusal(cmd, info, prev, cs);
      if (why) {
         hk_frame_anatomy_note(dev, "pass-merge end: ordered, %s", why);
         return false;
      }
   }

   /* Reopen the stream where its draws ended; the terminator is emitted
    * again when the merged instance ends. Until then a continuation stream
    * may follow it, so it takes the partial end-of-tile program back, as a
    * stream has when it is created (hk_cs_init_graphics()). */
   cs->current = m->resume_current;
   cs->cr.eot.main = prev->cr.eot.partial;
   /* EndRendering may also have dropped untouched colour attachments from
    * the tile programs (hk_g17p_elide_untouched_color()). */
   cs->cr.bg = prev->cr.bg;
   cs->cr.eot.partial = prev->cr.eot.partial;
   /* EndRendering may have elided untouched depth/stencil planes or replaced
    * the forced intermediate stores with the final store operations. A later
    * draw in this continued stream can use those planes, and a split can need
    * their stores again. Restore the recording state along with the EOT. The
    * final EndRendering still elides the whole stream if it remains untouched.
    */
   cs->cr.depth = prev->cr.depth;
   cs->cr.stencil = prev->cr.stencil;
   cs->cr.zls_control = prev->cr.zls_control;
   cs->g17p_pass_ends = false;
   cs->m5_continued_pass = true;
   cs->m5_ordered = ordered;
   cs->m5_layout_proof = hk_m5_attachments_exclude_descriptors(cmd, info);
   cmd->state.m5_ordered_split = false;
   cmd->current_cs.gfx = cs;

   *render = *prev;
   render->flags = info->flags;

   hk_sync_merge_rendering(cmd, cs);
   hk_sync_first_render_stream(cmd, cs);

   /* The per-instance state hk_begin_rendering() resets. */
   const VkRenderingAttachmentLocationInfoKHR ral_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
      .colorAttachmentCount = info->colorAttachmentCount,
   };
   vk_cmd_set_rendering_attachment_locations(&cmd->vk, &ral_info);
   hk_cmd_buffer_dirty_render_pass(cmd);

   if (cmd->state.gfx.descriptors.root.draw.view_index != 0) {
      cmd->state.gfx.descriptors.root.draw.view_index = 0;
      cmd->state.gfx.descriptors.root_dirty = true;
   }

   hk_frame_anatomy_note(dev, "pass-merge continue: %ux%u draws_before=%u%s",
                         render->cr.width, render->cr.height, cs->stats.cmds,
                         ordered ? " ordered" : "");
   perf_debug(cmd, "Continuing the previous render pass instance");
   return true;
}

static void
hk_begin_rendering(VkCommandBuffer commandBuffer,
                   const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   if (unlikely(cmd->state.pass_merge.valid) &&
       hk_m5_pass_merge_begin(cmd, pRenderingInfo))
      return;

   memset(render, 0, sizeof(*render));

   hk_sync_begin_rendering(
      cmd, pRenderingInfo->flags & VK_RENDERING_RESUMING_BIT);

   render->flags = pRenderingInfo->flags;
   /* The preceding suspended instance may have written storage memory. */
   render->barrier_shader_writes =
      (pRenderingInfo->flags & VK_RENDERING_RESUMING_BIT) != 0;
   render->area = pRenderingInfo->renderArea;
   render->view_mask = pRenderingInfo->viewMask;
   render->layer_count = pRenderingInfo->layerCount;
   render->tilebuffer.nr_samples = 0;

   const uint32_t layer_count = render->view_mask
                                   ? util_last_bit(render->view_mask)
                                   : render->layer_count;

   render->color_att_count = pRenderingInfo->colorAttachmentCount;
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_attachment_init(&render->color_att[i],
                         &pRenderingInfo->pColorAttachments[i]);
   }

   hk_attachment_init(&render->depth_att, pRenderingInfo->pDepthAttachment);
   hk_attachment_init(&render->stencil_att, pRenderingInfo->pStencilAttachment);

   if (unlikely(dev->frame_anatomy)) {
      /* Frame anatomy profiler: compression state of each colour target. */
      for (uint32_t i = 0; i < render->color_att_count; i++) {
         const struct hk_image_view *view = render->color_att[i].iview;
         if (!view)
            continue;
         const struct hk_image *image =
            container_of(view->vk.image, struct hk_image, vk);
         if (image->planes[view->planes[0].image_plane].layout.compressed)
            render->cr.color_compressed_mask |= BITFIELD_BIT(i);
      }
   }

   for (uint32_t i = 0; i < render->color_att_count; i++) {
      hk_merge_render_iview(render, render->color_att[i].iview, false);
   }

   hk_merge_render_iview(
      render, render->depth_att.iview ?: render->stencil_att.iview, true);

   /* Tile hazards (hk_sync.h): every attachment this instance may store,
    * before its first control stream exists. */
   {
      const uint32_t pass_layers = render->view_mask
                                      ? util_last_bit(render->view_mask)
                                      : render->layer_count;
      const struct hk_image_view *written[2 * HK_MAX_RTS + 2];
      bool preserve_loaded_color[2 * HK_MAX_RTS + 2] = {false};
      unsigned nr_written = 0;
      for (uint32_t i = 0; i < render->color_att_count; i++) {
         /* LOAD preserves samples that later draws may not touch. Keep
          * their producer even for a whole-tile render area: an earlier
          * attachment clear may have been lowered to compute. */
         preserve_loaded_color[nr_written] =
            (dev->dev.chip == AGX_CHIP_G17P ||
             agx_chip_g17_m5(dev->dev.chip)) &&
            pRenderingInfo->pColorAttachments[i].loadOp ==
               VK_ATTACHMENT_LOAD_OP_LOAD;
         written[nr_written++] = render->color_att[i].iview;
         written[nr_written++] = render->color_att[i].resolve_iview;
      }
      written[nr_written++] = render->depth_att.iview;
      written[nr_written++] = render->stencil_att.iview;
      for (unsigned i = 0; i < nr_written; ++i) {
         const struct hk_image_view *iview = written[i];
         if (!iview)
            continue;
         const uint32_t level = iview->vk.base_mip_level;
         hk_sync_pass_footprint(
            cmd, iview->vk.image, level, iview->vk.base_array_layer,
            iview->vk.base_array_layer + MAX2(pass_layers, 1),
            u_minify(iview->vk.image->extent.width, level),
            u_minify(iview->vk.image->extent.height, level), render->area,
            preserve_loaded_color[i] ||
               (render->view_mask &&
                render->view_mask != BITFIELD64_MASK(pass_layers)));
      }
      hk_sync_pass_footprints_done(cmd);
   }

   /* Infer for attachmentless. samples is inferred at draw-time. */
   render->cr.width =
      MAX2(render->cr.width, render->area.offset.x + render->area.extent.width);

   render->cr.height = MAX2(render->cr.height,
                            render->area.offset.y + render->area.extent.height);

   /* On G17P, zero dimensions are the native inactive-ZLS encoding.  Keep
    * them zero for attachmentless and colour-only rendering; every control
    * stream, including a partial stream, inherits that state from render. */
   if (!render->cr.zls_width && !agx_native_graphics(&dev->dev) &&
       !agx_chip_g17_m5(dev->dev.chip)) {
      render->cr.zls_width = render->cr.width;
      render->cr.zls_height = render->cr.height;
   }

   render->cr.layers = layer_count;

   /* A G17P multisample tilebuffer wider than eight bytes per sample only
    * keeps a tile's whole sample where colour-writing geometry covered that
    * tile; see AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE. A load-op clear
    * gets that guarantee from the full-area clear draw emitted below, so it
    * stays in the tilebuffer. A reloaded attachment cannot: the background
    * reload is itself the truncating program, and nothing else rewrites the
    * tiles the application does not draw to. Multiview has a second, separate
    * defect that the clear draw does not cover either. Those two spill. */
   bool g17p_wide_ms_to_memory = false;
   if (agx_chip_uncovered_tile_ms_defect(dev->dev.chip) &&
       render->tilebuffer.nr_samples > 1) {
      g17p_wide_ms_to_memory = render->view_mask != 0;

      for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
         const VkRenderingAttachmentInfo *att =
            &pRenderingInfo->pColorAttachments[i];

         g17p_wide_ms_to_memory |= att->imageView != VK_NULL_HANDLE &&
                                   att->loadOp == VK_ATTACHMENT_LOAD_OP_LOAD &&
                                   is_attachment_stored(att);
      }
   }
   render->tilebuffer = hk_tilebuffer_layout(
      dev, &render->color_att[0].vk_format, sizeof(render->color_att[0]),
      render->color_att_count, render->tilebuffer.nr_samples,
      (!agx_native_graphics(&dev->dev) && !agx_chip_g17_m5(dev->dev.chip)) ||
         !hk_agx3_g17p_single_layer_rendering(render->view_mask, layer_count),
      g17p_wide_ms_to_memory, hk_m5_resident_attachments(dev, render));

   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip))
      render->g17p_rt0_layout_supported =
         hk_agx3_g17p_rt0_tile_layout_supported(&render->tilebuffer);

   if (agx_native_graphics(&dev->dev) &&
       !render->g17p_rt0_layout_supported) {
      mesa_loge_once("G17P tilebuffer layout refused: attachments=%u "
                     "samples=%u sample_size=%u tile_size=%u",
                     render->color_att_count, render->tilebuffer.nr_samples,
                     render->tilebuffer.sample_size_B,
                     render->tilebuffer.tile_size);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   const VkRenderingAttachmentLocationInfoKHR ral_info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_LOCATION_INFO_KHR,
      .colorAttachmentCount = pRenderingInfo->colorAttachmentCount,
   };
   vk_cmd_set_rendering_attachment_locations(&cmd->vk, &ral_info);

   hk_cmd_buffer_dirty_render_pass(cmd);

   /* Determine whether the render area is complete, enabling us to use a
    * fast-clear.
    *
    * TODO: If it is incomplete but tile aligned, it should be possibly to fast
    * clear with the appropriate settings. This is critical for performance.
    */
   bool incomplete_render_area =
      render->area.offset.x > 0 || render->area.offset.y > 0 ||
      render->area.extent.width < render->cr.width ||
      render->area.extent.height < render->cr.height ||
      (render->view_mask &&
       render->view_mask != BITFIELD64_MASK(render->cr.layers));

   perf_debug(cmd, "Rendering %ux%ux%u@%u %s%s", render->cr.width,
              render->cr.height, render->cr.layers,
              render->tilebuffer.nr_samples,
              render->view_mask ? " multiview" : "",
              incomplete_render_area ? " incomplete" : "");

   struct hk_g17p_bg_eot_cache_entry *bg_eot_cache =
      hk_g17p_bg_eot_cache_get(cmd, pRenderingInfo,
                               incomplete_render_area);

   render->cr.bg.main = hk_build_bg_eot(cmd, pRenderingInfo, false, false,
                                        incomplete_render_area, bg_eot_cache);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   render->cr.bg.partial =
      hk_build_bg_eot(cmd, pRenderingInfo, false, true,
                      incomplete_render_area, bg_eot_cache);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   render->cr.eot.main =
      hk_build_bg_eot(cmd, pRenderingInfo, true, false,
                      incomplete_render_area, bg_eot_cache);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   render->cr.eot.partial =
      hk_build_bg_eot(cmd, pRenderingInfo, true, true,
                      incomplete_render_area, bg_eot_cache);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   render->cr.isp_bgobjvals = 0x300;

   const VkRenderingAttachmentInfo *attach_z = pRenderingInfo->pDepthAttachment;
   const VkRenderingAttachmentInfo *attach_s =
      pRenderingInfo->pStencilAttachment;

   struct ail_layout *z_layout = NULL, *s_layout = NULL;

   if (attach_z != NULL && attach_z != VK_NULL_HANDLE && attach_z->imageView) {
      struct hk_image_view *view = render->depth_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      z_layout = &image->planes[0].layout;

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      const struct util_format_description *desc =
         util_format_description(hk_format_to_pipe_format(view->vk.format));

      assert(desc->format == PIPE_FORMAT_Z32_FLOAT ||
             desc->format == PIPE_FORMAT_Z16_UNORM ||
             desc->format == PIPE_FORMAT_Z16_UNORM_S8_UINT ||
             desc->format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT);

      render->cr.depth.buffer =
         hk_image_base_address(image, 0) +
         ail_get_layer_level_B(z_layout, first_layer, level);

      /* Main stride in pages */
      assert((z_layout->depth_px == 1 ||
              util_is_aligned(z_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned Z layers");

      unsigned stride_pages = z_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.depth.stride = ((stride_pages - 1) << 14) | 1;

      /* A compressed single-layer image is not page padded, so a small one
       * has a zero-page layer stride and the expression above underflows.
       * The stride is unused for one layer; Apple's G17P Framebuffer writes
       * 0 for a sub-cacheline layer and cacheline units below one page, a
       * form the G17P kernel's ZLS validation does not accept. Send 0. */
      if ((agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
          stride_pages == 0) {
         assert(z_layout->depth_px == 1);
         render->cr.depth.stride = 0;
      }

      assert(z_layout->tiling != AIL_TILING_LINEAR && "must tile");

      if (z_layout->compressed) {
         render->cr.depth.meta =
            hk_image_base_address(image, 0) + z_layout->metadata_offset_B +
            (first_layer * z_layout->compression_layer_stride_B) +
            z_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(util_is_aligned(z_layout->compression_layer_stride_B,
                                AIL_CACHELINE) &&
                "Cacheline aligned Z meta layers");

         unsigned stride_lines =
            z_layout->compression_layer_stride_B / AIL_CACHELINE;
         render->cr.depth.meta_stride = (stride_lines - 1) << 14;
      }

      float clear_depth = attach_z->clearValue.depthStencil.depth;

      if (z_layout->format == PIPE_FORMAT_Z16_UNORM) {
         render->cr.isp_bgobjdepth = _mesa_float_to_unorm(clear_depth, 16);
      } else {
         render->cr.isp_bgobjdepth = fui(clear_depth);
      }
   }

   if (attach_s != NULL && attach_s != VK_NULL_HANDLE && attach_s->imageView) {
      struct hk_image_view *view = render->stencil_att.iview;
      struct hk_image *image =
         container_of(view->vk.image, struct hk_image, vk);

      /* Stencil is always the last plane (possibly the only plane) */
      unsigned plane = image->plane_count - 1;
      s_layout = &image->planes[plane].layout;
      assert(s_layout->format == PIPE_FORMAT_S8_UINT);

      unsigned level = view->vk.base_mip_level;
      unsigned first_layer = view->vk.base_array_layer;

      render->cr.stencil.buffer =
         hk_image_base_address(image, plane) +
         ail_get_layer_level_B(s_layout, first_layer, level);

      /* Main stride in pages */
      assert((s_layout->depth_px == 1 ||
              util_is_aligned(s_layout->layer_stride_B, AIL_PAGESIZE)) &&
             "Page aligned S layers");
      unsigned stride_pages = s_layout->layer_stride_B / AIL_PAGESIZE;
      render->cr.stencil.stride = ((stride_pages - 1) << 14) | 1;
      if ((agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
          stride_pages == 0) {
         assert(s_layout->depth_px == 1);
         render->cr.stencil.stride = 0;
      }

      if (s_layout->compressed) {
         render->cr.stencil.meta =
            hk_image_base_address(image, plane) + s_layout->metadata_offset_B +
            (first_layer * s_layout->compression_layer_stride_B) +
            s_layout->level_offsets_compressed_B[level];

         /* Meta stride in cache lines */
         assert(util_is_aligned(s_layout->compression_layer_stride_B,
                                AIL_CACHELINE) &&
                "Cacheline aligned S meta layers");

         unsigned stride_lines =
            s_layout->compression_layer_stride_B / AIL_CACHELINE;

         render->cr.stencil.meta_stride = (stride_lines - 1) << 14;
      }

      render->cr.isp_bgobjvals |= attach_s->clearValue.depthStencil.stencil;
   }

   hk_pack_zls_control(&render->cr.zls_control, z_layout, s_layout, attach_z,
                       attach_s, incomplete_render_area, false);

   hk_pack_zls_control(&render->cr.zls_control_partial, z_layout, s_layout,
                       attach_z, attach_s, incomplete_render_area, true);

   const bool agx3 = agx_chip_native_agx3(dev->dev.chip) ||
                     agx_chip_g17_m5(dev->dev.chip);
   render->zls_precise = agx3 && z_layout != NULL;
   render->g17p_zls_touched = 0;
   render->g17p_zls_written = 0;
   render->m5_attachments_reach_memory = false;
   {
      const struct hk_image_view *views[2 * HK_MAX_RTS + 4] = {
         render->depth_att.iview, render->stencil_att.iview,
         render->depth_att.resolve_iview, render->stencil_att.resolve_iview};
      for (uint32_t i = 0; i < render->color_att_count; ++i) {
         views[4 + 2 * i] = render->color_att[i].iview;
         views[5 + 2 * i] = render->color_att[i].resolve_iview;
      }
      for (unsigned i = 0; i < 4 + 2 * render->color_att_count; ++i) {
         const struct vk_image *image = views[i] ? views[i]->vk.image : NULL;
         if (image && ((image->create_flags &
                        (VK_IMAGE_CREATE_ALIAS_BIT |
                         VK_IMAGE_CREATE_SPARSE_BINDING_BIT)) ||
                       image->tiling == VK_IMAGE_TILING_LINEAR))
            render->m5_attachments_reach_memory = true;
      }
   }
   render->g17p_zls_elide_mask = 0;
   if (agx3 && !(render->flags & (VK_RENDERING_SUSPENDING_BIT |
                                 VK_RENDERING_RESUMING_BIT))) {
      const bool secondary =
         render->flags & VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
      if (z_layout && hk_g17p_zls_attachment_elidable(attach_z) &&
          (!secondary || hk_g17p_zls_attachment_none(attach_z)))
         render->g17p_zls_elide_mask |= VK_IMAGE_ASPECT_DEPTH_BIT;
      if (s_layout && hk_g17p_zls_attachment_elidable(attach_s) &&
          (!secondary || hk_g17p_zls_attachment_none(attach_s)))
         render->g17p_zls_elide_mask |= VK_IMAGE_ASPECT_STENCIL_BIT;
   }
   if (render->zls_precise) {
      hk_pack_zls_control_final(&render->zls_control_only, z_layout,
                                s_layout, attach_z, attach_s,
                                incomplete_render_area, false);
      hk_pack_zls_control_final(&render->zls_control_last, z_layout,
                                s_layout, attach_z, attach_s,
                                incomplete_render_area, true);
   }

   /* A depth-only pass that stores depth may learn later that the store is
    * dead (hk_take_pending_z_store()); keep the word without it. */
   render->no_z_store_ok =
      render->zls_precise && attach_z && attach_z->imageView &&
      attach_z->storeOp == VK_ATTACHMENT_STORE_OP_STORE &&
      attach_z->resolveMode == VK_RESOLVE_MODE_NONE && s_layout == NULL;
   if (render->no_z_store_ok) {
      VkRenderingAttachmentInfo z_no_store = *attach_z;
      z_no_store.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      hk_pack_zls_control_final(&render->zls_control_only_no_z_store,
                                z_layout, NULL, &z_no_store, NULL,
                                incomplete_render_area, false);
   }

   /* If multiview is disabled, always read 0. If multiview is enabled,
    * hk_set_view_index will dirty the root each draw.
    */
   if (cmd->state.gfx.descriptors.root.draw.view_index != 0) {
      cmd->state.gfx.descriptors.root.draw.view_index = 0;
      cmd->state.gfx.descriptors.root_dirty = true;
   }

   if (render->flags & VK_RENDERING_RESUMING_BIT)
      return;

   /* G17P: a transfer the application did not order before graphics work
    * may have written an attachment. */
   if (unlikely(hk_sync_xfer_pending(&cmd->sync)) && !cmd->in_meta)
      hk_xfer_check_attachments(cmd);

   /* The first control stream of the render pass is special since it gets
    * the clears. Create it and swap in the clear.
    */
   assert(!cmd->current_cs.gfx && "not already in a render pass");
   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
   if (!cs)
      return;

   cs->cr.bg.main = render->cr.bg.main;
   cs->cr.zls_control = render->cr.zls_control;
   cs->g17p_pass_begins = true;
   render->zls_first_cs = cs;

   /* Reordering barrier for post-gfx, in case we had any. */
   hk_cmd_buffer_end_compute_internal(cmd, &cmd->current_cs.post_gfx);

   /* The open compute stream sits before this render in the list. It stays
    * open across the pass unless the render was recorded to wait on it: then
    * later compute must not extend what the render waits for. Compute the
    * application records after the pass with no dependency on it may
    * legally run before it (Vulkan 1.3, 7.2: without a dependency, commands
    * may execute in any order). */
   hk_sync_first_render_stream(cmd, cs);

   /* If we spill colour attachments, we need to decompress them. This happens
    * at the start of the render; it is not re-emitted when resuming
    * secondaries. It could be hoisted to the start of the command buffer but
    * we're not that clever yet.
    */
   if (agx_tilebuffer_spills(&render->tilebuffer)) {
      const unsigned descriptor_generation =
         agx_chip_descriptor_generation(dev->dev.chip);
      if (!hk_native_pbe_hardware_supported(descriptor_generation)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      perf_debug(cmd, "eMRT render pass");

      for (unsigned i = 0; i < render->color_att_count; ++i) {
         /* Resident targets retain compression. Only memory-backed targets
          * need their compressed contents converted before shader stores.
          */
         if (agx_chip_g17_m5(dev->dev.chip) && !render->tilebuffer.spilled[i])
            continue;

         struct hk_image_view *view = render->color_att[i].iview;
         if (view) {
            if (!view->planes[0].emrt_pbe_valid) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FORMAT_NOT_SUPPORTED);
               return;
            }

            struct hk_image *image =
               container_of(view->vk.image, struct hk_image, vk);

            /* TODO: YCbCr interaction? */
            uint8_t plane = 0;
            uint8_t image_plane = view->planes[plane].image_plane;
            struct ail_layout *layout = &image->planes[image_plane].layout;

            if (ail_is_level_logically_compressed(layout,
                                                  view->vk.base_mip_level)) {

               perf_debug(cmd, "Decompressing in-place");

               /* libagx_decompress has no G17P binary. G17P only compresses
                * 1-8 byte single-sample targets, which cannot spill (8 x 8 B
                * fit the 120 B/sample tilebuffer), so this is unreachable;
                * fail closed with a diagnostic rather than a NULL program. */
               if (dev->dev.chip == AGX_CHIP_G17P) {
                  mesa_loge_once("G17P: spilled render pass with a compressed "
                                 "attachment (%s) is unsupported",
                                 util_format_short_name(layout->format));
                  vk_command_buffer_set_error(&cmd->vk,
                                              VK_ERROR_FEATURE_NOT_PRESENT);
                  return;
               }

               unsigned level = view->vk.base_mip_level;
               unsigned layer = view->vk.base_array_layer;
               uint64_t base = hk_image_base_address(image, image_plane);

               struct agx_grid grid =
                  agx_3d(ail_metadata_width_tl(layout, level) * 32,
                         ail_metadata_height_tl(layout, level), layer_count);

               if (agx_chip_g17_m5(dev->dev.chip)) {
                  if (layout->format != PIPE_FORMAT_R16G16B16A16_FLOAT ||
                      (layout->sample_count_sa != 2 &&
                       layout->sample_count_sa != 4) ||
                      layout->depth_px != 1 || layout->levels != 1 ||
                      !dev->sampled_images.base_va || !dev->samplers.table.bo ||
                      view->planes[plane].emrt_read_heap_index >=
                         dev->sampled_images.alloc) {
                     vk_command_buffer_set_error(&cmd->vk,
                                                 VK_ERROR_FEATURE_NOT_PRESENT);
                     return;
                  }
                  const struct ail_tile tile = layout->tilesize_el[level];
                  libagx_m5_decompress_rgba16f(
                     cmd, grid, AGX_BARRIER_ALL | AGX_PREGFX,
                     dev->sampled_images.base_va,
                     dev->samplers.table.bo->va->addr,
                     base + layout->metadata_offset_B,
                     base + ail_get_layer_level_B(layout, layer, level),
                     view->planes[plane].emrt_read_heap_index,
                     ail_metadata_width_tl(layout, level),
                     ail_metadata_height_tl(layout, level),
                     align(layout->width_px, tile.width_el),
                     tile.width_el, tile.height_el,
                     util_logbase2(layout->sample_count_sa) - 1);
               } else {
                  struct libagx_decompress_images imgs;
                  memcpy(&imgs.compressed, &view->planes[plane].emrt_texture,
                         sizeof(imgs.compressed));
                  memcpy(&imgs.uncompressed, &view->planes[plane].emrt_pbe,
                         sizeof(imgs.uncompressed));
                  libagx_decompress(cmd, grid, AGX_BARRIER_ALL | AGX_PREGFX,
                                    layout, layer, level, base,
                                    hk_pool_upload(cmd, &imgs, sizeof(imgs), 64));
               }
            }
         }
      }
   }

   uint32_t clear_count = 0;
   VkClearAttachment clear_att[HK_MAX_RTS + 1];
   bool resolved_clear = false, resolved_clear_z = false;

   for (uint32_t i = 0; i < pRenderingInfo->colorAttachmentCount; i++) {
      const VkRenderingAttachmentInfo *att_info =
         &pRenderingInfo->pColorAttachments[i];
      if (att_info->imageView == VK_NULL_HANDLE ||
          att_info->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR)
         continue;

      clear_att[clear_count++] = (VkClearAttachment){
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .colorAttachment = i,
         .clearValue = att_info->clearValue,
      };

      render->color_att[i].clear = true;

      static_assert(sizeof(render->color_att[i].clear_colour) ==
                    sizeof(att_info->clearValue));
      memcpy(render->color_att[i].clear_colour, &att_info->clearValue,
             sizeof(att_info->clearValue));

      resolved_clear |= is_attachment_stored(att_info);
   }

   const uint32_t color_clear_count = clear_count;

   clear_att[clear_count] = (VkClearAttachment){
      .aspectMask = 0,
   };

   if (attach_z && attach_z->imageView != VK_NULL_HANDLE &&
       attach_z->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
      clear_att[clear_count].clearValue.depthStencil.depth =
         attach_z->clearValue.depthStencil.depth;

      render->depth_att.clear = true;
      render->depth_att.clear_colour[0] =
         fui(attach_z->clearValue.depthStencil.depth);

      resolved_clear_z = is_attachment_stored(attach_z);
   }

   if (attach_s != NULL && attach_s->imageView != VK_NULL_HANDLE &&
       attach_s->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear_att[clear_count].aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
      clear_att[clear_count].clearValue.depthStencil.stencil =
         attach_s->clearValue.depthStencil.stencil;

      render->stencil_att.clear = true;
      render->stencil_att.clear_colour[1] =
         attach_s->clearValue.depthStencil.stencil;

      resolved_clear |= is_attachment_stored(attach_s);
   }

   if (clear_att[clear_count].aspectMask != 0)
      clear_count++;

   const bool clear_spills =
      (agx_native_graphics(&dev->dev) ||
       agx_chip_g17_m5(dev->dev.chip)) &&
      agx_tilebuffer_spills(&render->tilebuffer);

   /* A G17P multisample tilebuffer wider than eight bytes per sample cannot
    * survive a tile that no colour-writing geometry covered: see
    * AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE. Emit the load-op clear as
    * an ordinary full-area draw instead. That covers every tile, so the
    * background program only ever runs on a tile the hardware also
    * rasterizes, which keeps the whole sample. The target stays in the
    * tilebuffer.
    *
    * A spilling multisample tilebuffer takes the same full clear on every
    * chip: a draw that clears only the spilled targets makes its epilog
    * preserve the resident ones by reading the tile back, a multisample eMRT
    * shape the packer cannot encode. (On G17P such a tilebuffer is always
    * wider than eight bytes a sample, so this adds nothing there.) */
   const bool clear_wide_multisample =
      color_clear_count > 0 && render->tilebuffer.nr_samples > 1 &&
      (clear_spills ||
       (agx_chip_uncovered_tile_ms_defect(dev->dev.chip) &&
        render->tilebuffer.sample_size_B >
           AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE));
   if (clear_spills && !incomplete_render_area && !clear_wide_multisample) {
      unsigned selected = 0;
      for (unsigned i = 0; i < clear_count; ++i) {
         if (clear_att[i].aspectMask == VK_IMAGE_ASPECT_COLOR_BIT &&
             render->tilebuffer.spilled[clear_att[i].colorAttachment])
            clear_att[selected++] = clear_att[i];
      }
      clear_count = selected;
   }
   /* Resolves of loaded samples write the destination even without geometry. */
   const bool loaded_resolve =
      render->g17p_direct_color_resolve &&
      pRenderingInfo->pColorAttachments[0].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD;
   const bool process_empty_tiles =
      resolved_clear || resolved_clear_z || loaded_resolve;
   render->process_empty_tiles_without_z = resolved_clear || loaded_resolve;
   cs->cr.process_empty_tiles = process_empty_tiles;
   if (unlikely(dev->frame_anatomy) && incomplete_render_area)
      cs->anatomy_area = render->area;
   if (clear_count > 0 &&
       (incomplete_render_area || clear_spills || clear_wide_multisample)) {
      const VkClearRect clear_rect = {
         .rect = render->area,
         .baseArrayLayer = 0,
         .layerCount = render->view_mask ? 1 : render->layer_count,
      };

      /* The clear draw covers the whole render area, so every tile it has
       * to clear is non-empty. A tile outside the area takes no draw and has
       * nothing to clear: processing it only loads and stores it back
       * unchanged (and an incomplete render area always loads). On M5 that
       * was a whole-framebuffer load/store round trip, ~50 us for a 1x
       * RGBA8 3024x1964 target, for a clear of a small HUD rectangle. Only
       * a resolve of loaded samples still writes its destination from empty
       * tiles. */
      if (agx_chip_g17_m5(dev->dev.chip)) {
         render->process_empty_tiles_without_z = loaded_resolve;
         cs->cr.process_empty_tiles = loaded_resolve;
      }

      hk_CmdClearAttachments(hk_cmd_buffer_to_handle(cmd), clear_count,
                             clear_att, 1, &clear_rect);
   } else {
      /* If a tile is empty, we do not want to process it, as the redundant
       * roundtrip of memory-->tilebuffer-->memory wastes a tremendous amount of
       * memory bandwidth. Any draw marks a tile as non-empty, so we only need
       * to process empty tiles if the background+EOT programs have a side
       * effect. This is the case exactly when there is an attachment we are
       * fast clearing and then storing.
       */
      cs->cr.process_empty_tiles = process_empty_tiles;
   }
}

static void hk_take_pending_z_store(struct hk_cmd_buffer *cmd,
                                    const VkRenderingInfo *info);

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginRendering(VkCommandBuffer commandBuffer,
                     const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   hk_take_pending_z_store(cmd, pRenderingInfo);

   if (likely(!cmd->state.deferred_clear_count)) {
      hk_begin_rendering(commandBuffer, pRenderingInfo);
      return;
   }

   /* Attachments that take a deferred clear as their load-op clear, then
    * the other deferred clears, before the pass (hk_flush_deferred_clears()). */
   VkRenderingInfo info;
   VkRenderingAttachmentInfo color[HK_MAX_RTS], depth, stencil;
   const VkRenderingInfo *taken = hk_take_deferred_clears(
      cmd, pRenderingInfo, &info, color, &depth, &stencil);
   hk_flush_deferred_clears(cmd);
   hk_begin_rendering(commandBuffer, taken);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndRendering2KHR(VkCommandBuffer commandBuffer,
                       const VkRenderingEndInfoKHR *pRenderingEndInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_rendering_state *render = &cmd->state.gfx.render;

   /* The last control stream of the render pass is special since it gets its
    * stores dropped. Swap it in.
    */
   struct hk_cs *cs = cmd->current_cs.gfx;

   struct hk_pending_z_store z_pending = {0};

   /* M5: the next instance may continue in this stream, see
    * hk_m5_pass_merge_begin(). Where the draws ended and the chunk they
    * ended in, before the terminator. */
   struct hk_pass_merge_candidate *merge = &cmd->state.pass_merge;
   merge->valid = false;
   const bool continuable = cs && hk_m5_pass_merge_candidate(cmd, cs);
   void *const resume_current = cs ? cs->current : NULL;
   void *const end_before = cs ? cs->end : NULL;

   /* An in-pass barrier may have closed the final stream already. Elide
    * untouched planes from the whole pass even then: NONE/NONE attachments
    * must not retain a forced store of uninitialized tile contents.
    */
   const VkImageAspectFlags zls_elided =
      render->g17p_zls_elide_mask & ~render->g17p_zls_touched;

   if (cs) {
      cs->cr.eot.main = render->cr.eot.main;
      cs->g17p_pass_ends = !(render->flags & VK_RENDERING_SUSPENDING_BIT);
      hk_g17p_elide_untouched_color(cmd, cs);

      /* Precise depth store: see hk_pack_zls_control_final(). A
       * suspended pass continues in a later command buffer, which reloads
       * depth, so it keeps the forced store. An elided pass has no
       * depth/stencil to store after the per-aspect elision below. */
      if (render->zls_precise &&
          !(render->flags & VK_RENDERING_SUSPENDING_BIT)) {
         cs->cr.zls_control = cs == render->zls_first_cs
                                 ? render->zls_control_only
                                 : render->zls_control_last;
      }

      /* The depth store of a single-stream pass on a whole depth-only image
       * may turn out dead; see hk_take_pending_z_store(). */
      const struct hk_image_view *z_view = render->depth_att.iview;
      const struct hk_image *z_image =
         z_view ? container_of(z_view->vk.image, struct hk_image, vk) : NULL;
      if (render->no_z_store_ok &&
          !(zls_elided & VK_IMAGE_ASPECT_DEPTH_BIT) && !cmd->in_meta &&
          cs == render->zls_first_cs && cs->stats.cmds > 0 &&
          !render->view_mask &&
          !(render->flags & (VK_RENDERING_SUSPENDING_BIT |
                             VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT)) &&
          z_image && z_image->vk.aspects == VK_IMAGE_ASPECT_DEPTH_BIT &&
          z_image->vk.image_type == VK_IMAGE_TYPE_2D &&
          z_image->vk.mip_levels == 1 && z_image->vk.array_layers == 1) {
         z_pending = (struct hk_pending_z_store){
            .cs = cs,
            .image = z_image,
            .zls_control = render->zls_control_only_no_z_store,
            .process_empty_tiles = render->process_empty_tiles_without_z,
         };
      }
   }

   /* Apply after the final control word, including already closed streams. */
   hk_g17p_zls_elide(cmd, zls_elided);
   hk_g17p_zls_elide_store(cmd, render->g17p_zls_elide_mask &
                                   render->g17p_zls_touched &
                                   ~render->g17p_zls_written);

   perf_debug(cmd, "End rendering");
   hk_cmd_buffer_end_graphics(cmd);

   if (continuable)
      merge->render = *render;

   bool need_resolve = false;

   /* Translate render state back to VK for meta */
   VkRenderingAttachmentInfo vk_color_att[HK_MAX_RTS];
   VkRenderingAttachmentFlagsInfoKHR vk_color_att_flags[HK_MAX_RTS];
   for (uint32_t i = 0; i < render->color_att_count; i++) {
      if (render->color_att[i].resolve_mode != VK_RESOLVE_MODE_NONE &&
          !render->g17p_direct_color_resolve)
         need_resolve = true;

      vk_color_att_flags[i] = (VkRenderingAttachmentFlagsInfoKHR) {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
         .flags = render->color_att[i].flags,
      };
      vk_color_att[i] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .pNext = &vk_color_att_flags[i],
         .imageView = hk_image_view_to_handle(render->color_att[i].iview),
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         .resolveMode = render->g17p_direct_color_resolve
                           ? VK_RESOLVE_MODE_NONE
                           : render->color_att[i].resolve_mode,
         .resolveImageView =
            hk_image_view_to_handle(render->color_att[i].resolve_iview),
         .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      };
   }

   const VkRenderingAttachmentFlagsInfoKHR vk_depth_att_flags = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
      .flags = render->depth_att.flags,
   };
   const VkRenderingAttachmentInfo vk_depth_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = &vk_depth_att_flags,
      .imageView = hk_image_view_to_handle(render->depth_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->depth_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->depth_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->depth_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingAttachmentFlagsInfoKHR vk_stencil_att_flags = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_FLAGS_INFO_KHR,
      .flags = render->stencil_att.flags,
   };
   const VkRenderingAttachmentInfo vk_stencil_att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .pNext = &vk_stencil_att_flags,
      .imageView = hk_image_view_to_handle(render->stencil_att.iview),
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .resolveMode = render->stencil_att.resolve_mode,
      .resolveImageView =
         hk_image_view_to_handle(render->stencil_att.resolve_iview),
      .resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL,
   };
   if (render->stencil_att.resolve_mode != VK_RESOLVE_MODE_NONE)
      need_resolve = true;

   const VkRenderingInfo vk_render = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = render->area,
      .layerCount = render->layer_count,
      .viewMask = render->view_mask,
      .colorAttachmentCount = render->color_att_count,
      .pColorAttachments = vk_color_att,
      .pDepthAttachment = &vk_depth_att,
      .pStencilAttachment = &vk_stencil_att,
   };

   if (render->flags & VK_RENDERING_SUSPENDING_BIT)
      need_resolve = false;

   memset(render, 0, sizeof(*render));

   hk_sync_end_rendering(cmd);

   /* Classic upstream submits each render after its predecessor. In
    * particular, coherent advanced blending in a later render pass must
    * load the earlier pass's stored colour without an application barrier.
    * The native dependency tracker otherwise permits full-tile passes to
    * overlap. Retain upstream ordering on AGX2, including across command
    * buffers/submissions through the tracker's prior requirements. M5
    * schedules its render passes like native AGX3. */
   if (!agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev) &&
       !agx_chip_g17_m5(hk_cmd_buffer_device(cmd)->dev.chip))
      hk_sync_barrier(cmd, HK_SYNC_GFX, HK_SYNC_GFX);

   if (continuable && !need_resolve && cs->end == end_before &&
       !vk_command_buffer_has_error(&cmd->vk)) {
      merge->cs = cs;
      merge->resume_current = resume_current;
      merge->after_current = cs->current;
      merge->end = cs->end;
      merge->next_seq = cmd->sync.next_seq;
      merge->sync_epoch = cmd->sync.epoch;
      merge->open_cs = cmd->current_cs.cs;
      merge->open_cs_current =
         cmd->current_cs.cs ? cmd->current_cs.cs->current : NULL;
      merge->ordered = false;
      merge->valid = true;
   }

   if (need_resolve) {
      perf_debug(cmd, "Resolving render pass, colour store op %u",
                 vk_color_att[0].storeOp);

      /* The resolve is a render of the driver's own that reads what this
       * pass stored: it waits on the pass, and it is graphics work for every
       * later dependency, like the resolve operation of the pass itself
       * (Vulkan 1.3, 8.4: resolves are part of the render pass instance in
       * the COLOR_ATTACHMENT_OUTPUT stage). */
      hk_sync_resolve_begin(cmd);
      hk_meta_resolve_rendering(cmd, &vk_render);
      hk_sync_resolve_end(cmd);
   }

   cmd->state.pending_z_store = z_pending;
}

/*
 * Dead depth stores (AGX3).
 *
 * Applications often store a depth attachment that is cleared again before
 * anything reads it; Minecraft stores its depth image at the end of every
 * pass and clears the whole image with vkCmdClearDepthStencilImage before
 * the next one (hk_defer_clear() kills the pending store there). A cleared and
 * stored attachment makes the pass process every tile, including tiles no
 * draw covered, only to write the clear value. When the next pass begins
 * with that whole image as its depth attachment, cleared or with undefined
 * contents, over its whole area, and nothing recorded between the two
 * passes could read the image (only buffer commands whose ranges miss it,
 * barriers, or state binding; the same commands a deferred clear survives,
 * see hk_flush_deferred_clears()), the first pass's store is unobservable:
 * drop it, and with it the empty tiles the store alone required.
 */
static void
hk_take_pending_z_store(struct hk_cmd_buffer *cmd,
                        const VkRenderingInfo *info)
{
   struct hk_pending_z_store *p = &cmd->state.pending_z_store;
   if (likely(!p->cs))
      return;

   const VkRenderingAttachmentInfo *z = info->pDepthAttachment;
   const struct hk_image *image = p->image;
   if (z && z->imageView != VK_NULL_HANDLE &&
       (z->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ||
        z->loadOp == VK_ATTACHMENT_LOAD_OP_DONT_CARE) &&
       !(info->flags & VK_RENDERING_RESUMING_BIT) && !info->viewMask &&
       info->renderArea.offset.x == 0 && info->renderArea.offset.y == 0 &&
       info->renderArea.extent.width >= image->vk.extent.width &&
       info->renderArea.extent.height >= image->vk.extent.height) {
      VK_FROM_HANDLE(hk_image_view, iview, z->imageView);
      if (iview->vk.image == &image->vk) {
         hk_kill_pending_z_store(cmd);
         return;
      }
   }

   p->cs = NULL;
}

static void
hk_init_heap(const void *data)
{
   struct hk_device *dev = (struct hk_device *)data;
   struct hk_instance *instance =
      hk_physical_device_instance(hk_device_physical(dev));

   perf_debug_dev(&dev->dev, "Allocating heap");

   /* XML drirc values bypass the option parser's environment range checks.
    * Bound them before allocation: poly_heap.size and draw ranges are u32.
    */
   STATIC_ASSERT(UINT64_C(1024) * 1024 * 1024 <= UINT32_MAX);
   unsigned mb = CLAMP(instance->drirc.misc.geometry_heap_size_mb, 16, 1024);
   size_t size = (size_t)mb * 1024 * 1024;
   struct agx_bo *heap =
      agx_bo_create(&dev->dev, size, 0, 0, "Geometry heap");
   if (!heap)
      return;

   /* The geometry state buffer is initialized here and then is treated by
    * the CPU as rodata, even though the GPU uses it for scratch internally.
    */
   struct poly_heap *map = agx_bo_map(dev->rodata.heap_bo);
   if (!map) {
      agx_bo_unreference(&dev->dev, heap);
      return;
   }

   *map = (struct poly_heap){
      .base = heap->va->addr,
      .size = size,
   };
   dev->heap = heap;
}

/*
 * Allocating the configured geometry heap requires agx_bo_create(): the kernel
 * finds and zeroes the pages. The default 128 MiB allocation measured 15.5 ms
 * on G17P. Every consumer of the
 * heap is reached from a draw, so leaving the allocation to util_call_once()
 * put those 15.5 ms inside the first vkCmdDraw() that used geometry or
 * tessellation -- a stall in the middle of a render pass, and ~150x the cost of
 * the same draw once warm (median 105 us).
 *
 * Nothing about the heap depends on the draw, only on the device, so the
 * allocation can be forced as soon as we know the client has a stage that will
 * need it. Command preparation and draw admission also check the result,
 * including heap users introduced by indirect adjacency or index robustness.
 */
VkResult
hk_device_ensure_geometry_heap(struct hk_device *dev)
{
   util_call_once_data(&dev->heap_init_once, hk_init_heap, dev);
   return dev->heap ? VK_SUCCESS : VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

static bool
hk_cmd_ensure_geometry_heap(struct hk_cmd_buffer *cmd)
{
   VkResult result = hk_device_ensure_geometry_heap(hk_cmd_buffer_device(cmd));
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return false;
   }
   return true;
}

static uint64_t
hk_heap(struct hk_cmd_buffer *cmd)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   assert(dev->heap && "heap consumers require checked preparation/admission");

   /* We need to free all allocations after each command buffer execution */
   if (!cmd->uses_heap) {
      perf_debug(cmd, "Freeing heap");
      uint64_t addr = dev->rodata.heap;

      /* One device-wide bump allocator: every pass that uses it runs after
       * every earlier stream and its free runs after the pass. */
      hk_sync_heap_used(cmd);

      /* Zeroing the allocated index frees everything */
      hk_queue_write(cmd, addr + offsetof(struct poly_heap, bottom), 0,
                     true /* after gfx */);

      cmd->uses_heap = true;
   }

   return dev->rodata.heap;
}

static uint64_t
hk_index_buffer(uint64_t index_buffer, uint size_el, uint offset_el,
                uint elsize_B)
{
   if (offset_el < size_el)
      return index_buffer + (offset_el * elsize_B);
   else
      return AGX_ZERO_PAGE_ADDRESS;
}

static uint64_t
hk_upload_vertex_params(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_vertex_params params;
   poly_vertex_params_init(&params, 0, wg_size);

   /* XXX: We should deduplicate this logic */
   bool indirect = agx_is_indirect(draw.b) || draw.restart;

   if (!indirect)
      poly_vertex_params_set_draw(&params, draw.b.count[0], draw.b.count[1]);

   if (draw.indexed) {
      unsigned index_size_B = agx_index_size_to_B(draw.index_size);
      unsigned range_el = agx_draw_index_range_el(draw);

      params.index_buffer = hk_index_buffer(agx_draw_index_buffer(draw),
                                            range_el, 0, index_size_B);

      params.index_size_B = index_size_B;
      params.index_buffer_range_el = range_el;
   }

   if (gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY]) {

      struct hk_shader *vs = hk_bound_sw_vs(gfx);
      params.outputs = vs->b.info.outputs;

      if (!indirect) {
         uint32_t verts = draw.b.count[0], instances = draw.b.count[1];
         unsigned vb_size =
            poly_tcs_in_size(verts * instances, vs->b.info.outputs);

         /* Allocate if there are any outputs, or use the null sink to trap
          * reads if there aren't. Those reads are undefined but should not
          * fault. Affects:
          *
          *    dEQP-VK.pipeline.monolithic.no_position.explicit_declarations.basic.single_view.v0_g1
          */
         params.output_buffer = vb_size ? hk_pool_alloc(cmd, vb_size, 4).gpu
                                        : AGX_SCRATCH_PAGE_ADDRESS;
      }
   }

   desc->root.draw.vertex_outputs = params.outputs;

   return hk_pool_upload(cmd, &params, sizeof(params), 8);
}

static enum mesa_prim
hk_gs_in_prim(struct hk_cmd_buffer *cmd)
{
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];

   if (tes != NULL)
      return gfx->tess.prim;
   else
      return vk_topology_to_mesa(dyn->ia.primitive_topology);
}

static enum mesa_prim
hk_rast_prim(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   if (gs != NULL) {
      return gs->variants[HK_GS_VARIANT_RAST].info.gs.mode;
   } else {
      switch (dyn->ia.primitive_topology) {
      case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
      case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
         return MESA_PRIM_LINES;
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
      case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
         return MESA_PRIM_TRIANGLES;
      default:
         return hk_gs_in_prim(cmd);
      }
   }
}

static struct hk_shader *
hk_bound_fs(struct hk_cmd_buffer *cmd)
{
   return hk_fs_variant(cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT],
                        cmd->vk.dynamic_graphics_state.ms.rasterization_samples);
}

static uint64_t
hk_upload_geometry_params(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct hk_shader *fs = hk_bound_fs(cmd);
   struct hk_shader *count = hk_count_gs_variant(gs);

   /* XXX: We should deduplicate this logic */
   bool indirect = agx_is_indirect(draw.b) ||
                   gfx->shaders[MESA_SHADER_TESS_EVAL] || draw.restart;
   enum mesa_prim mode = hk_gs_in_prim(cmd);

   if (draw.restart) {
      mode = u_decomposed_prim(mode);
   }

   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_geometry_params params;
   poly_geometry_params_init(&params, mode, wg_size);

   params.flat_outputs = fs->info.fs.interp.flat;

   if (gfx->xfb_enabled) {
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb); ++i) {
         params.xfb_base_original[i] = gfx->xfb[i].addr;
         params.xfb_size[i] = gfx->xfb[i].range;
         params.xfb_offs_ptrs[i] = gfx->xfb_offsets + i * sizeof(uint32_t);
      }
   } else {
      for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb); ++i) {
         params.xfb_offs_ptrs[i] = AGX_ZERO_PAGE_ADDRESS;
         params.xfb_base_original[i] = AGX_ZERO_PAGE_ADDRESS;
         params.xfb_size[i] = 0;
      }
   }

   for (unsigned i = 0; i < ARRAY_SIZE(gfx->xfb_query); ++i) {
      uint64_t q = gfx->xfb_query[i];

      if (q) {
         params.xfb_prims_generated_counter[i] = q;
         params.prims_generated_counter[i] = q + sizeof(uint64_t);
      } else {
         params.xfb_prims_generated_counter[i] = AGX_SCRATCH_PAGE_ADDRESS;
         params.prims_generated_counter[i] = AGX_SCRATCH_PAGE_ADDRESS;
      }

      const uint64_t generated = hk_prims_generated_addr(cmd, i);
      params.query_prims_generated_counter[i] =
         generated ? generated : AGX_SCRATCH_PAGE_ADDRESS;

      /* TODO: Optimize out? */
      params.xfb_overflow[i] = AGX_SCRATCH_PAGE_ADDRESS;
   }

   /* TODO: Optimize out? */
   params.xfb_any_overflow = AGX_SCRATCH_PAGE_ADDRESS;

   /* Calculate input primitive count for direct draws, and allocate the vertex
    * & count buffers. GPU calculates and allocates for indirect draws.
    */
   params.count_buffer_stride = count->info.gs.count_words * 4;

   if (!count->info.gs.prefix_sum && params.count_buffer_stride) {
      struct agx_ptr T = hk_pool_alloc(cmd, 16, 4);
      memset(T.cpu, 0, 16);
      params.count_buffer = T.gpu;
   }

   struct poly_gs_info *gsi = &count->info.gs;

   if (indirect) {
      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         /* Need to allocate heap if we haven't yet */
         hk_heap(cmd);

         cmd->geom_index_buffer = dev->heap->va->addr;
         cmd->geom_index_count = dev->heap->size;
      } else {
         cmd->geom_index_count =
            poly_gs_rast_vertices(gsi->shape, gsi->max_indices, 1, 0);
      }
   } else {
      poly_geometry_params_set_draw(&params, mode, gsi->shape, gsi->max_indices,
                                    draw.b.count[0], draw.b.count[1]);

      unsigned size = params.input_primitives * params.count_buffer_stride;
      if (count->info.gs.prefix_sum && size) {
         params.count_buffer = hk_pool_alloc(cmd, size, 4).gpu;
      }

      cmd->geom_index_count = params.draw.index_count;
      cmd->geom_instance_count = params.draw.instance_count;

      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         params.output_index_buffer =
            hk_pool_alloc(cmd, cmd->geom_index_count * 4, 4).gpu;

         cmd->geom_index_buffer = params.output_index_buffer;
      }
   }

   if (gsi->shape == POLY_GS_SHAPE_STATIC_INDEXED) {
      cmd->geom_index_buffer =
         hk_pool_upload(cmd, count->info.gs.topology, gsi->max_indices * 4, 4);
   }

   return hk_pool_upload(cmd, &params, sizeof(params), 8);
}

static void
hk_init_tess_params(struct hk_cmd_buffer *cmd, struct poly_tess_params *out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_shader *tcs = hk_only_variant(gfx->shaders[MESA_SHADER_TESS_CTRL]);

   enum poly_tess_partitioning partitioning =
      gfx->tess.info.spacing == TESS_SPACING_EQUAL
         ? POLY_TESS_PARTITIONING_INTEGER
      : gfx->tess.info.spacing == TESS_SPACING_FRACTIONAL_ODD
         ? POLY_TESS_PARTITIONING_FRACTIONAL_ODD
         : POLY_TESS_PARTITIONING_FRACTIONAL_EVEN;

   *out = (struct poly_tess_params){
      .heap = hk_heap(cmd),
      .tcs_stride_el = tcs->info.tess.tcs_output_stride / 4,
      .statistic = hk_pipeline_stat_addr(
         cmd,
         VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT),

      .input_patch_size = dyn->ts.patch_control_points,
      .output_patch_size = tcs->info.tess.tcs_output_patch_size,
      .tcs_patch_constants = tcs->info.tess.tcs_nr_patch_outputs,
      .tcs_per_vertex_outputs = tcs->info.tess.tcs_per_vertex_outputs,
      .partitioning = partitioning,
      .points_mode = gfx->tess.info.points,
      .isolines = gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES,
   };

   if (!out->points_mode && gfx->tess.info.mode != TESS_PRIMITIVE_ISOLINES) {
      out->ccw = gfx->tess.info.ccw;
      out->ccw ^=
         dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
   }

   /* heap is allocated by hk_heap */
   out->patch_coord_buffer = dev->heap->va->addr;
}

static void
hk_upload_tess_params(struct hk_cmd_buffer *cmd, struct poly_tess_params *out,
                      struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   hk_init_tess_params(cmd, out);

   uint32_t draw_stride_el = 5;
   size_t draw_stride_B = draw_stride_el * sizeof(uint32_t);

   if (!agx_is_indirect(draw.b)) {
      unsigned in_patches = draw.b.count[0] / out->input_patch_size;
      unsigned unrolled_patches = in_patches * draw.b.count[1];

      uint32_t alloc = 0;
      uint32_t tcs_out_offs = alloc;
      alloc += unrolled_patches * out->tcs_stride_el * sizeof(uint32_t);

      uint32_t patch_coord_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      uint32_t count_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      /* Single API draw */
      uint32_t draw_offs = alloc;
      alloc += draw_stride_B;

      struct agx_ptr blob = hk_pool_alloc(cmd, alloc, 4);
      out->tcs_buffer = blob.gpu + tcs_out_offs;
      out->patches_per_instance = in_patches;
      out->coord_allocs = blob.gpu + patch_coord_offs;
      out->nr_patches = unrolled_patches;
      out->out_draws = blob.gpu + draw_offs;
      out->counts = blob.gpu + count_offs;
   } else {
      /* Allocate 3x indirect global+local grids for VS/TCS/tess */
      uint32_t grid_stride = sizeof(uint32_t) * 6;
      gfx->tess.grids = hk_pool_alloc(cmd, grid_stride * 3, 4).gpu;

      out->out_draws = hk_pool_alloc(cmd, draw_stride_B, 4).gpu;
   }

   gfx->tess.out_draws = out->out_draws;
}

static struct hk_api_shader *
hk_build_meta_shader_locked(struct hk_device *dev, struct hk_internal_key *key,
                            hk_internal_builder_t builder, VkResult *result)
{
   /* Try to get the cached shader */
   struct hash_entry *ent = _mesa_hash_table_search(dev->kernels.ht, key);
   if (ent)
      return ent->data;

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, NULL);
   builder(&b, key->key);

   const struct vk_pipeline_robustness_state rs = {
      .images = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_DISABLED_EXT,
      .storage_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .uniform_buffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
      .vertex_inputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED_EXT,
   };

   struct vk_shader_compile_info info = {
      .stage = b.shader->info.stage,
      .nir = b.shader,
      .robustness = &rs,
   };

   hk_preprocess_nir_internal(dev->vk.physical, b.shader);

   struct hk_api_shader *s;
   *result = hk_compile_shader(dev, &info, NULL, NULL, NULL, &s);
   if (*result != VK_SUCCESS)
      return NULL;

   /* ..and cache it before we return. The key is on the stack right now, so
    * clone it before using it as a hash table key. The clone is logically owned
    * by the hash table.
    */
   size_t total_key_size = sizeof(*key) + key->key_size;
   void *cloned_key = ralloc_memdup(dev->kernels.ht, key, total_key_size);

   _mesa_hash_table_insert(dev->kernels.ht, cloned_key, s);
   return s;
}

/* On failure returns NULL and sets *result to the compile error, which can be
 * an out-of-memory error under memory pressure.
 */
struct hk_api_shader *
hk_meta_shader(struct hk_device *dev, hk_internal_builder_t builder, void *data,
               size_t data_size, VkResult *result)
{
   *result = VK_SUCCESS;
   size_t total_key_size = sizeof(struct hk_internal_key) + data_size;

   struct hk_internal_key *key = alloca(total_key_size);
   memset(key, 0, total_key_size);
   key->builder = builder;
   key->key_size = data_size;

   if (data_size)
      memcpy(key->key, data, data_size);

   simple_mtx_lock(&dev->kernels.lock);
   struct hk_api_shader *s =
      hk_build_meta_shader_locked(dev, key, builder, result);
   simple_mtx_unlock(&dev->kernels.lock);

   return s;
}

static struct agx_draw
hk_draw_as_indexed_indirect(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   assert(draw.indexed);

   if (agx_is_indirect(draw.b))
      return draw;

   VkDrawIndexedIndirectCommand desc = {
      .indexCount = draw.b.count[0],
      .instanceCount = draw.b.count[1],
      .firstIndex = draw.start,
      .vertexOffset = draw.index_bias,
      .firstInstance = draw.start_instance,
   };

   return agx_draw_indexed_indirect(
      hk_pool_upload(cmd, &desc, sizeof(desc), 4), draw.index_buffer,
      draw.index_buffer_range_B, draw.index_size, draw.restart);
}

static struct agx_draw
hk_draw_without_restart(struct hk_cmd_buffer *cmd, struct agx_draw draw,
                        uint32_t draw_count)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   perf_debug(cmd, "Unrolling primitive restart due to GS/XFB");

   /* The unroll kernel assumes an indirect draw. Synthesize one if needed */
   draw = hk_draw_as_indexed_indirect(cmd, draw);

   /* Next, we unroll the index buffer used by the indirect draw */
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

   assert(draw_count == 1 && "TODO: multidraw");

   struct libagx_unroll_restart_args ia = {
      .heap = hk_heap(cmd),
      .index_buffer = draw.index_buffer,
      .in_draw = draw.b.ptr,
      .out_draw = hk_pool_alloc(cmd, 5 * sizeof(uint32_t) * draw_count, 4).gpu,
      .max_draws = 1 /* TODO: MDI */,
      .restart_index = dyn->ia.primitive_restart_index,
      .index_buffer_size_el = agx_draw_index_range_el(draw),
      .index_size_log2 = draw.index_size,
      .flatshade_first =
         dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT,
   };

   libagx_unroll_restart_struct(cmd, agx_1d(1024 * draw_count),
                                AGX_BARRIER_ALL | AGX_PREGFX, ia,
                                poly_compact_prim(prim));

   return agx_draw_indexed_indirect(ia.out_draw, dev->heap->va->addr,
                                    dev->heap->size, draw.index_size,
                                    false /* restart */);
}

static __attribute__((hot)) struct agx_draw
hk_launch_gs_prerast(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                     struct agx_draw draw)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];
   struct agx_grid grid_vs, grid_gs;

   struct hk_shader *vs = hk_bound_sw_vs_before_gs(gfx);
   struct hk_shader *main = hk_main_gs_variant(gs);
   struct hk_shader *count = hk_count_gs_variant(gs);
   struct hk_shader *pre_gs = hk_pre_gs_variant(gs);

   uint64_t vertex_params = desc->root.draw.vertex_params;
   uint64_t geometry_params = desc->root.draw.geometry_params;
   unsigned count_words = count->info.gs.count_words;
   struct agx_workgroup wg = agx_workgroup(64, 1, 1);

   if (false /* TODO */)
      perf_debug(cmd, "Transform feedbck");
   else if (count_words)
      perf_debug(cmd, "Geometry shader with counts");
   else
      perf_debug(cmd, "Geometry shader without counts");

   enum mesa_prim mode = hk_gs_in_prim(cmd);

   if (draw.restart) {
      draw = hk_draw_without_restart(cmd, draw, 1);
      mode = u_decomposed_prim(mode);
   }

   /* Setup grids */
   if (agx_is_indirect(draw.b)) {
      struct libagx_gs_setup_indirect_args gsi = {
         .heap = hk_heap(cmd),
         .index_buffer = draw.index_buffer,
         .draw = draw.b.ptr,
         .vp = desc->root.draw.vertex_params,
         .p = desc->root.draw.geometry_params,
         .vs_outputs = vs->b.info.outputs,
         .prim = mode,
         .is_prefix_summing = count->info.gs.prefix_sum,
         .max_indices = count->info.gs.max_indices,
         .shape = count->info.gs.shape,
      };

      if (draw.indexed) {
         gsi.index_size_B = agx_index_size_to_B(draw.index_size);
         gsi.index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_gs_setup_indirect_struct(cmd, agx_1d(1),
                                      AGX_BARRIER_ALL | AGX_PREGFX, gsi);

      grid_vs = agx_grid_indirect_local(
         vertex_params + offsetof(struct poly_vertex_params, grid));

      grid_gs = agx_grid_indirect_local(
         geometry_params + offsetof(struct poly_geometry_params, grid));
   } else {
      grid_vs = grid_gs = draw.b;
      grid_gs.count[0] = u_decomposed_prims_for_vertices(mode, draw.b.count[0]);
   }

   /* A TES with native ray queries is launched with a direct grid bounding
    * the tessellator's output; the kernel skips threads past the real count
    * (hk_nir_lower_tes_variant()). See hk_tes_direct_query_threads(). */
   if (gfx->shaders[MESA_SHADER_TESS_EVAL] && gfx->tess.tes_direct_threads &&
       vs->b.info.uses_ray_traversal)
      grid_vs = agx_1d(gfx->tess.tes_direct_threads);

   /* Launch the vertex shader first */
   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
      hk_dispatch_with_local_size(cmd, cs, vs, grid_vs, wg);
   } else {
      hk_reserve_scratch(cmd, cs, vs);
      hk_dispatch_with_usc(
         dev, cs, &vs->b.info,
         hk_upload_usc_words(cmd, vs,
                             vs->info.stage == MESA_SHADER_VERTEX
                                ? gfx->linked[MESA_SHADER_VERTEX]
                                : vs->only_linked),
         grid_vs, wg);
   }

   /* Transform feedback and various queries require extra dispatching,
    * determine if we need that here.
    */
   VkQueryPipelineStatisticFlagBits gs_queries =
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT |
      VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
      VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT;

   struct hk_root_descriptor_table *root = &cmd->state.gfx.descriptors.root;
   bool xfb_or_queries =
      main->info.gs.xfb || (root->draw.pipeline_stats_flags & gs_queries) ||
      hk_prims_generated_active(cmd);

   if (xfb_or_queries) {
      /* If we need counts, launch the count shader and prefix sum the results. */
      if (count_words) {
         perf_debug(dev, "Geometry shader count");
         hk_dispatch_with_local_size(cmd, cs, count, grid_gs, wg);
      }

      if (count->info.gs.prefix_sum) {
         perf_debug(dev, "Geometry shader transform feedback prefix sum");
         libagx_prefix_sum_geom(cmd, agx_1d(1024 * count_words),
                                AGX_BARRIER_ALL | AGX_PREGFX, geometry_params);
      }

      /* Transform feedback / query program */
      perf_debug(dev, "Transform feedback / geometry query");
      hk_dispatch_with_local_size(cmd, cs, pre_gs, agx_1d(1),
                                  agx_workgroup(1, 1, 1));
   }

   /* Pre-rast geometry shader */
   hk_dispatch_with_local_size(cmd, cs, main, grid_gs, wg);

   if (poly_gs_indexed(count->info.gs.shape)) {
      enum agx_index_size index_size =
         agx_translate_index_size(poly_gs_index_size(count->info.gs.shape));

      if (agx_is_indirect(draw.b)) {
         return agx_draw_indexed_indirect(
            geometry_params + offsetof(struct poly_geometry_params, draw),
            cmd->geom_index_buffer, cmd->geom_index_count, index_size, true);
      } else {
         return agx_draw_indexed(cmd->geom_index_count,
                                 cmd->geom_instance_count, 0, 0, 0,
                                 cmd->geom_index_buffer,
                                 cmd->geom_index_count * 4, index_size, true);
      }
   } else {
      if (agx_is_indirect(draw.b)) {
         return agx_draw_indirect(geometry_params +
                                  offsetof(struct poly_geometry_params, draw));
      } else {
         return (struct agx_draw){
            .b = agx_3d(cmd->geom_index_count, cmd->geom_instance_count, 1),
         };
      }
   }
}

/*
 * On G17P, TES ray queries read back zero both from the post-tessellation
 * vertex program (an indexed indirect draw) and from Poly's TES-before-GS
 * kernel (an indirect dispatch), even with a null acceleration structure.
 * Queries issued from direct draws and dispatches (VS, TCS, GS, compute)
 * work, and no other launch property distinguishes the TES. Give the TES
 * kernel a direct grid: the tessellator decides the invocation count on the
 * GPU, but every patch emits at most HK_TES_MAX_PATCH_INDICES of them, and
 * the kernel skips invocations past the count it is given. Indirect API
 * draws keep the indirect grid.
 */
#define HK_TES_MAX_PATCH_INDICES (4u * 64u * 64u * 3u)
#define HK_TES_DIRECT_QUERY_MAX_THREADS (1u << 24)

static uint32_t
hk_tes_direct_query_threads(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_shader *tes = hk_bound_sw_vs_before_gs(gfx);
   const uint32_t patch_size =
      cmd->vk.dynamic_graphics_state.ts.patch_control_points;

   if (!tes || !tes->compiled || !tes->info.agx3_gen1_fragment_rq ||
       !hk_tes_ray_queries_in_compute(hk_cmd_buffer_device(cmd)) ||
       agx_is_indirect(draw.b) || !patch_size)
      return 0;

   const uint64_t threads = (uint64_t)(draw.b.count[0] / patch_size) *
                            draw.b.count[1] * HK_TES_MAX_PATCH_INDICES;
   return threads && threads <= HK_TES_DIRECT_QUERY_MAX_THREADS ? threads : 0;
}

static __attribute__((hot)) struct agx_draw
hk_launch_tess(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
               struct agx_draw draw, uint64_t c_prims, uint64_t c_inv,
               uint64_t prims_generated)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct agx_grid grid_vs, grid_tcs, grid_tess;

   struct hk_shader *vs = hk_bound_sw_vs(gfx);
   struct hk_shader *tcs = hk_only_variant(gfx->shaders[MESA_SHADER_TESS_CTRL]);

   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint32_t input_patch_size = dyn->ts.patch_control_points;
   uint64_t state = gfx->descriptors.root.draw.tess_params;
   struct hk_tess_info info = gfx->tess.info;

   perf_debug(cmd, "Tessellation");

   uint64_t tcs_stat = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT);

   /* Setup grids */
   if (agx_is_indirect(draw.b)) {
      perf_debug(cmd, "Indirect tessellation");

      struct libagx_tess_setup_indirect_args args = {
         .p = state,
         .grids = gfx->tess.grids,
         .indirect = draw.b.ptr,
         .vp = gfx->descriptors.root.draw.vertex_params,
         .vertex_outputs = vs->b.info.outputs,
         .tcs_statistic = hk_pipeline_stat_addr(
            cmd,
            VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT),
      };

      if (draw.indexed) {
         args.in_index_buffer = draw.index_buffer;
         args.in_index_size_B = agx_index_size_to_B(draw.index_size);
         args.in_index_buffer_range_el = agx_draw_index_range_el(draw);
      }

      libagx_tess_setup_indirect_struct(cmd, agx_1d(1),
                                        AGX_BARRIER_ALL | AGX_PREGFX, args);

      uint32_t grid_stride = sizeof(uint32_t) * 6;
      grid_vs = agx_grid_indirect_local(gfx->tess.grids + 0 * grid_stride);
      grid_tcs = agx_grid_indirect_local(gfx->tess.grids + 1 * grid_stride);
      grid_tess = agx_grid_indirect_local(gfx->tess.grids + 2 * grid_stride);
   } else {
      uint32_t patches = draw.b.count[0] / input_patch_size;
      grid_vs = grid_tcs = draw.b;

      grid_tcs.count[0] = patches * tcs->info.tess.tcs_output_patch_size;
      grid_tess = agx_1d(patches * draw.b.count[1]);

      /* TCS invocation counter increments once per-patch */
      if (hk_stat_enabled(tcs_stat)) {
         perf_debug(cmd, "Direct TCS statistic");
         libagx_increment_statistic(
            cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, tcs_stat, patches);
      }
   }

   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
      hk_dispatch_with_local_size(cmd, cs, vs, grid_vs,
                                  agx_workgroup(64, 1, 1));
      hk_dispatch_with_local_size(
         cmd, cs, tcs, grid_tcs,
         agx_workgroup(tcs->info.tess.tcs_output_patch_size, 1, 1));
   } else {
      /* Legacy launches do not reserve scratch in hk_dispatch_with_usc(). */
      hk_reserve_scratch(cmd, cs, vs);
      hk_reserve_scratch(cmd, cs, tcs);
      hk_dispatch_with_usc(
         dev, cs, &vs->b.info,
         hk_upload_usc_words(cmd, vs, gfx->linked[MESA_SHADER_VERTEX]),
         grid_vs, agx_workgroup(64, 1, 1));

      hk_dispatch_with_usc(
         dev, cs, &tcs->b.info,
         hk_upload_usc_words(cmd, tcs, tcs->only_linked), grid_tcs,
         agx_workgroup(tcs->info.tess.tcs_output_patch_size, 1, 1));
   }

   /* First generate counts, then prefix sum them, and then tessellate. */
   libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX,
                     info.mode, POLY_TESS_MODE_COUNT, state);
   libagx_prefix_sum_tess(cmd, agx_1d(1024),
                          AGX_BARRIER_ALL | AGX_PREGFX, state, c_prims,
                          c_inv, prims_generated,
                          c_prims || c_inv || prims_generated);
   libagx_tessellate(cmd, grid_tess, AGX_BARRIER_ALL | AGX_PREGFX,
                     info.mode, POLY_TESS_MODE_WITH_COUNTS, state);

   return agx_draw_indexed_indirect(gfx->tess.out_draws, dev->heap->va->addr,
                                    dev->heap->size, AGX_INDEX_SIZE_U32, false);
}

__attribute__((hot)) void
hk_cmd_bind_graphics_shader(struct hk_cmd_buffer *cmd,
                            const mesa_shader_stage stage,
                            struct hk_api_shader *shader)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   /* Null fragment shaders are annoying to handle, because we may still need an
    * actual fragment shader to attach a prolog to. Rather than adding special
    * cases, we just bind an empty fragment shader instead of NULL to make
    * everything work correctly.
    */
   if (stage == MESA_SHADER_FRAGMENT && shader == NULL) {
      shader = dev->null_fs;
   }

   /* A shader compiled without custom-border emulation is replaced by its
    * emulating twin once the device has created a sampler that needs it. */
   if (shader && unlikely(shader->border_source) &&
       p_atomic_read(&hk_device_physical(dev)->border_emulation)) {
      struct hk_api_shader *twin = hk_border_twin(dev, shader);
      if (twin)
         shader = twin;
      else
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
   }

   assert(stage < ARRAY_SIZE(cmd->state.gfx.shaders));
   if (cmd->state.gfx.shaders[stage] == shader)
      return;

   if (shader && shader->border_source)
      cmd->state.gfx.border_free_stages |= BITFIELD_BIT(stage);
   else
      cmd->state.gfx.border_free_stages &= ~BITFIELD_BIT(stage);

   /* Any direct stage replacement (shader objects, meta restore, or an
    * internal passthrough GS) invalidates the common runtime's redundant
    * pipeline-bind identity.  A complete pipeline bind sets it again only
    * after every stage and static dynamic state have been restored. */
   cmd->vk.last_graphics_pipeline = NULL;

   cmd->state.gfx.shaders[stage] = shader;

   /* Whether a bound graphics program writes memory, for render pass
    * continuation (hk_m5_ordered_refusal()). */
   bool writes = false;
   for (unsigned s = 0; s < ARRAY_SIZE(cmd->state.gfx.shaders); ++s) {
      struct hk_api_shader *api = cmd->state.gfx.shaders[s];
      if (!api)
         continue;
      hk_foreach_variant(api, v)
         writes |= v->info.writes_external_memory;
   }
   cmd->state.gfx.shaders_write_memory = writes;
   cmd->state.gfx.shaders_dirty |= BITFIELD_BIT(stage);

   if (stage == MESA_SHADER_FRAGMENT)
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);
}

static void
hk_flush_shaders(struct hk_cmd_buffer *cmd)
{
   struct hk_api_shader *fs_obj =
       cmd->state.gfx.shaders[MESA_SHADER_FRAGMENT];
   const bool sample_variant_dirty = fs_obj->g17p_sample_variants &&
      BITSET_TEST(cmd->vk.dynamic_graphics_state.dirty,
                  MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES);
   if (cmd->state.gfx.shaders_dirty == 0 && !sample_variant_dirty)
      return;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   if (sample_variant_dirty)
      gfx->shaders_dirty |= BITFIELD_BIT(MESA_SHADER_FRAGMENT);

   /* A multisample variant of a dynamic-sample-count fragment shader is
    * built on its first draw. */
   if (fs_obj->sample_source && IS_SHADER_DIRTY(FRAGMENT)) {
      VkResult result = hk_fs_build_sample_variant(
         hk_cmd_buffer_device(cmd), fs_obj,
         cmd->vk.dynamic_graphics_state.ms.rasterization_samples);
      if (result != VK_SUCCESS)
         vk_command_buffer_set_error(&cmd->vk, result);
   }

   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   /* Geometry shading overrides the restart index, reemit on rebind */
   if (IS_SHADER_DIRTY(GEOMETRY)) {
      struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
      struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

      const bool api_gs = gs && !gs->is_passthrough;
      if (desc->root.draw.api_gs != api_gs) {
         desc->root.draw.api_gs = api_gs;
         desc->root_dirty = true;
      }
      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE);
   }

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *fs = hk_bound_fs(cmd);

   /* A new VS/FS pair or sample-count variant may change interpolation. This
    * assignment does not depend on the fast linked shaders.
    */
   gfx->g17p_smooth_memo.valid = false;
   agx_assign_uvs(&gfx->linked_varyings, &hw_vs->info.uvs,
                  fs->info.fs.interp.flat,
                  fs->info.fs.interp.linear);

   /* The root holds one byte per varying slot while agx_varyings_vs::slots
    * holds unsigned ints, so the table has to be narrowed element by element;
    * a byte copy of the first VARYING_SLOT_MAX bytes of the unsigned array
    * left every slot past POS at the wrong UVS index for any vertex stage
    * that reads its varying placement from the root (shader objects,
    * graphics pipeline libraries). */
   bool uvs_index_changed = false;
   for (unsigned i = 0; i < VARYING_SLOT_MAX; ++i) {
      assert(gfx->linked_varyings.slots[i] <= UINT8_MAX);
      const uint8_t slot = gfx->linked_varyings.slots[i];
      uvs_index_changed |= desc->root.draw.uvs_index[i] != slot;
      desc->root.draw.uvs_index[i] = slot;
   }
   if (uvs_index_changed)
      desc->root_dirty = true;

   /* Shader replacement changes the logical root only if the selected
    * immutable data address actually changes.  In particular, equivalent
    * shader objects must not clone the root and invalidate both native IOTO
    * programs merely because their API handles differ. */
   const uint64_t vs_constant_data_addr = hk_cmd_upload_constants(
      cmd, hk_any_variant(gfx->shaders[MESA_SHADER_VERTEX]), MESA_SHADER_VERTEX);
   const uint64_t fs_constant_data_addr =
      hk_cmd_upload_constants(cmd, fs, MESA_SHADER_FRAGMENT);
   if (desc->root.draw.vs_constant_data_addr != vs_constant_data_addr) {
      desc->root.draw.vs_constant_data_addr = vs_constant_data_addr;
      desc->root_dirty = true;
   }
   if (desc->root.draw.fs_constant_data_addr != fs_constant_data_addr) {
      desc->root.draw.fs_constant_data_addr = fs_constant_data_addr;
      desc->root_dirty = true;
   }
}

static void build_g17p_fs_epilog(nir_builder *b, const void *key_);
static void build_fs_prolog(nir_builder *b, const void *key);

/* Prolog/epilog parts persist in the device's disk cache. The in-memory
 * table above is per process, so without it every launch would compile each
 * part again on the render thread at the first bind or draw that needed it
 * (~87 compiles for Minecraft even with a warm pipeline cache). The disk key
 * names the builder by a stable id and covers every input of the compile
 * below; the cache identity already carries the build, the GPU and the
 * codegen flags/environment. */

static uint8_t
hk_prolog_epilog_builder_id(hk_internal_builder_t builder)
{
   if (builder == agx_nir_vs_prolog_raw)
      return 1;
   if (builder == agx_nir_vs_prolog)
      return 2;
   if (builder == build_fs_prolog)
      return 3;
   if (builder == build_g17p_fs_epilog)
      return 4;
   if (builder == agx_nir_fs_epilog)
      return 5;
   return 0;
}

static bool
hk_prolog_epilog_disk_key(struct hk_device *dev, const struct hk_internal_key *key,
                          bool preprocess_nir, bool stop, unsigned cf_base,
                          const BITSET_WORD *g17p_tile_partitions,
                          unsigned nr_samples, cache_key out)
{
   struct disk_cache *cache = dev->vk.physical->disk_cache;
   const uint8_t builder = hk_prolog_epilog_builder_id(key->builder);
   if (!cache || !builder)
      return false;

   const struct agx_device_key dk = agx_gather_device_key(&dev->dev);
   struct {
      char magic[8];
      uint32_t builder, stop, preprocess, cf_base, nr_samples, key_size;
      uint32_t chip, encoding, g13x_coherency, soft_fault;
      BITSET_WORD partitions[ARRAY_SIZE(((struct agx_shader_key *)0)->g17p_tile_partitions)];
   } h;
   memset(&h, 0, sizeof(h));
   memcpy(h.magic, "hkpe-v1", 8);
   h.builder = builder;
   h.stop = stop;
   h.preprocess = preprocess_nir;
   h.cf_base = cf_base;
   h.nr_samples = nr_samples;
   h.key_size = key->key_size;
   h.chip = dk.chip;
   h.encoding = dk.encoding;
   h.g13x_coherency = dk.needs_g13x_coherency;
   h.soft_fault = dk.soft_fault;
   if (g17p_tile_partitions)
      memcpy(h.partitions, g17p_tile_partitions, sizeof(h.partitions));

   const size_t size = sizeof(h) + key->key_size;
   uint8_t *buf = alloca(size);
   memcpy(buf, &h, sizeof(h));
   memcpy(buf + sizeof(h), key->key, key->key_size);
   disk_cache_compute_key(cache, buf, size, out);
   return true;
}

/* A stored part is its agx_shader_info followed by the binary. */
static struct agx_shader_part *
hk_part_from_blob(const uint8_t *data, size_t size, void *mem_ctx)
{
   struct agx_shader_part *part = NULL;
   struct agx_shader_info info;
   if (size >= sizeof(info)) {
      memcpy(&info, data, sizeof(info));
      if (size == sizeof(info) + info.binary_size && info.binary_size) {
         part = rzalloc(mem_ctx, struct agx_shader_part);
         void *binary = part ? malloc(info.binary_size) : NULL;
         if (binary) {
            part->info = info;
            memcpy(binary, data + sizeof(info), info.binary_size);
            part->binary = binary;
         } else {
            ralloc_free(part);
            part = NULL;
         }
      }
   }
   return part;
}

static struct agx_shader_part *
hk_part_disk_get(struct hk_device *dev, const cache_key disk_key, void *mem_ctx)
{
   size_t size = 0;
   uint8_t *data = disk_cache_get(dev->vk.physical->disk_cache, disk_key, &size);
   if (!data)
      return NULL;

   struct agx_shader_part *part = hk_part_from_blob(data, size, mem_ctx);
   free(data);
   return part;
}

static void
hk_part_disk_put(struct hk_device *dev, const cache_key disk_key,
                 const struct agx_shader_part *part)
{
   const size_t size = sizeof(part->info) + part->info.binary_size;
   uint8_t *data = malloc(size);
   if (!data)
      return;
   memcpy(data, &part->info, sizeof(part->info));
   memcpy(data + sizeof(part->info), part->binary, part->info.binary_size);
   disk_cache_put(dev->vk.physical->disk_cache, disk_key, data, size, NULL);
   free(data);
}

static bool
hk_prolog_epilog_insert(struct hk_device *dev, const struct hk_internal_key *key,
                        struct agx_shader_part *part)
{
   /* The key is on the caller's stack, so clone it before using it as a hash
    * table key. The clone is logically owned by the hash table. */
   size_t total_key_size = sizeof(*key) + key->key_size;
   void *cloned_key = ralloc_memdup(dev->prolog_epilog.ht, key, total_key_size);
   if (!cloned_key)
      return false;

   _mesa_hash_table_insert(dev->prolog_epilog.ht, cloned_key, part);
   return true;
}

static unsigned
hk_g17p_count_intrinsics(nir_shader *shader, nir_intrinsic_op op)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               count++;
         }
      }
   }
   return count;
}

static struct agx_shader_part *
hk_get_prolog_epilog_locked(struct hk_device *dev, struct hk_internal_key *key,
                            hk_internal_builder_t builder, bool preprocess_nir,
                            bool stop, unsigned cf_base,
                            const BITSET_WORD *g17p_tile_partitions,
                            unsigned nr_samples)
{
   /* Try to get the cached shader */
   struct hash_entry *ent = _mesa_hash_table_search(dev->prolog_epilog.ht, key);
   if (ent)
      return ent->data;

   cache_key disk_key;
   const bool disk =
      hk_prolog_epilog_disk_key(dev, key, preprocess_nir, stop, cf_base,
                                g17p_tile_partitions, nr_samples, disk_key);
   struct agx_shader_part *cached =
      disk ? hk_part_disk_get(dev, disk_key, dev->prolog_epilog.ht) : NULL;
   if (cached) {
      if (hk_prolog_epilog_insert(dev, key, cached))
         return cached;
      free(cached->binary);
      ralloc_free(cached);
      return NULL;
   }

   nir_builder b = nir_builder_init_simple_shader(0, &agx_nir_options, NULL);
   builder(&b, key->key);

   if (preprocess_nir)
      agx_preprocess_nir(b.shader);

   /* A pixel-shaded multisample epilog that reads the tile (blending, for
    * one) is wrapped in the sample loop, which preprocessing unrolls --
    * depth/stencil write included. Each sample's
    * Z/S write then sits between the colour transactions of the samples
    * around it, a shape the G17P pixel protocol cannot express, and the
    * epilog fails to compile (any 2x/4x draw that blends and writes
    * FragDepth or FragStencilRef: VK_ERROR_FEATURE_NOT_PRESENT at
    * vkEndCommandBuffer). The same unrolling clones an API sample mask's
    * coverage update into every sample, and each clone takes class 1 again
    * after the colour classes are held, which hangs the 3D pipeline under
    * concurrency (agx_nir_hoist_g17p_zs_writes). The sample-shaded epilog
    * hoists its per-sample writes and kills into the one leading class-1
    * transaction; do the same here for every epilog left with several. */
   if (builder == build_g17p_fs_epilog &&
       !((const struct agx_fs_epilog_key *)key->key)->link.sample_shading &&
       (hk_g17p_count_intrinsics(b.shader, nir_intrinsic_store_zs_agx) > 1 ||
        hk_g17p_count_intrinsics(b.shader, nir_intrinsic_sample_mask_agx) > 1))
      NIR_PASS(_, b.shader, agx_nir_hoist_g17p_zs_writes);

   struct agx_shader_key backend_key = {
      .dev = agx_gather_device_key(&dev->dev),
      .fs.nr_samples = nr_samples,
      .secondary = true,
      .no_stop = !stop,
   };
   if (builder == build_g17p_fs_epilog)
      backend_key.g17p_emrt =
         ((const struct agx_fs_epilog_key *)key->key)->native_spills;
   if (builder == build_g17p_fs_epilog ||
       (agx_chip_g17_m5(dev->dev.chip) &&
        builder == agx_nir_fs_epilog && nr_samples > 1)) {
      /* Preprocessing can unroll any multisample epilog, not only eMRT.
       * Its induction masks then become literals. Materialize the native
       * coverage operands after that transformation, before binary packing. */
      NIR_PASS(_, b.shader, agx_nir_lower_g17p_tile_samples, nr_samples);
   }
   if (backend_key.g17p_emrt) {
      /* Only the image accesses may be conditional. Every resident tile
       * operation must execute on every path through the epilog. */
      nir_metadata_require(b.impl, nir_metadata_dominance);
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            if ((op == nir_intrinsic_store_local_pixel_agx ||
                 op == nir_intrinsic_load_local_pixel_agx) &&
                !nir_block_dominates(block, nir_impl_last_block(b.impl))) {
               backend_key.g17p_emrt = false;
               mesa_loge("G17P eMRT has conditional resident tile access");
            }
         }
      }
   }
   if (g17p_tile_partitions) {
      memcpy(backend_key.g17p_tile_partitions, g17p_tile_partitions,
             sizeof(backend_key.g17p_tile_partitions));
   }

   /* We always use dynamic sample shading in the GL driver. Indicate that. */
   if (b.shader->info.stage == MESA_SHADER_FRAGMENT) {
      backend_key.fs.cf_base = cf_base;

      if (b.shader->info.fs.uses_sample_shading)
         backend_key.fs.inside_sample_loop = true;
   }

   struct agx_shader_part *part =
      rzalloc(dev->prolog_epilog.ht, struct agx_shader_part);
   if (!part) {
      ralloc_free(b.shader);
      return NULL;
   }

   const bool compiled = agx_compile_shader_nir(b.shader, &backend_key, part);

   ralloc_free(b.shader);
   if (!compiled) {
      /* A rejected backend instruction leaves an empty part. Caching that
       * as success lets fast-link silently omit required output/blend code. */
      ralloc_free(part);
      return NULL;
   }

   /* ..and cache it before we return. */
   if (!hk_prolog_epilog_insert(dev, key, part)) {
      free(part->binary);
      ralloc_free(part);
      return NULL;
   }

   if (disk)
      hk_part_disk_put(dev, disk_key, part);
   return part;
}

static struct agx_shader_part *
hk_get_prolog_epilog(struct hk_device *dev, void *data, size_t data_size,
                     hk_internal_builder_t builder, bool preprocess_nir,
                     bool stop, unsigned cf_base,
                     const BITSET_WORD *g17p_tile_partitions,
                     unsigned nr_samples)
{
   /* Build the meta shader key */
   size_t total_key_size = sizeof(struct hk_internal_key) + data_size;

   struct hk_internal_key *key = alloca(total_key_size);
   memset(key, 0, total_key_size);
   key->builder = builder;
   key->key_size = data_size;
   key->stop = stop;

   if (data_size)
      memcpy(key->key, data, data_size);

   simple_mtx_lock(&dev->prolog_epilog.lock);

   struct agx_shader_part *part =
      hk_get_prolog_epilog_locked(dev, key, builder, preprocess_nir, stop,
                                  cf_base, g17p_tile_partitions, nr_samples);

   simple_mtx_unlock(&dev->prolog_epilog.lock);
   return part;
}

static struct hk_linked_shader *
hk_get_fast_linked_locked_vs(struct hk_device *dev, struct hk_shader *shader,
                             struct hk_fast_link_key_vs *key)
{
   struct agx_shader_part *prolog =
      hk_get_prolog_epilog(dev, &key->prolog, sizeof(key->prolog),
                           (agx_native_graphics(&dev->dev) ||
                            agx_chip_g17_m5(dev->dev.chip))
                              ? agx_nir_vs_prolog_raw : agx_nir_vs_prolog,
                           false, false, 0, 0, 0);
   if (!prolog)
      return NULL;

   struct hk_fast_link_key *key_clone =
      ralloc_memdup(shader->linked.ht, key, sizeof(*key));
   if (!key_clone)
      return NULL;

   struct hk_linked_shader *linked =
      hk_fast_link(dev, false, shader, prolog, NULL, 0);
   if (!linked) {
      ralloc_free(key_clone);
      return NULL;
   }

   /* XXX: Fix this higher up the stack */
   linked->sw_indexing = !key->prolog.hw || key->prolog.adjacency;
   linked->sw_indexing_hw_vs = key->prolog.hw && key->prolog.adjacency;
   linked->b.uses_base_param |= linked->sw_indexing;

   _mesa_hash_table_insert(shader->linked.ht, key_clone, linked);
   return linked;
}

static bool
lower_fs_root(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_root_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_load_preamble(b, 1, 64, .base = AGX_ABI_FUNI_ROOT));
   return true;
}

static void
build_fs_prolog(nir_builder *b, const void *key)
{
   agx_nir_fs_prolog(b, key);

   /* Lower load_stat_query_address_agx, needed for FS statistics */
   NIR_PASS(_, b->shader, hk_lower_uvs_index, MESA_SHADER_FRAGMENT, 0, 0);
   NIR_PASS(_, b->shader, nir_shader_intrinsics_pass, lower_fs_root,
            nir_metadata_control_flow, NULL);
}

static void
build_g17p_fs_epilog(nir_builder *b, const void *key)
{
   agx_nir_fs_epilog_g17p(b, key);
}

/*
 * Inline blend: one fragment program with the colour epilog
 * compiled into it, as Apple's compiler blends inline (native throughput
 * fragment 0x10c..0x174: LLDCOL, normflt, f32add, fltnorm, LSTCOL in the
 * main program; no export copies, no separate epilog).
 *
 * The API shader kept its NIR from just before its backend compile. Here the
 * draw's epilog is built exactly as hk_get_prolog_epilog_locked() builds it,
 * inlined after the main code, and every export_agx / load_exported_agx pair
 * becomes the exported SSA value (NIR guarantees exports sit in the main's
 * last block and loads in the epilog's first). The result compiles with the
 * main's backend key. The draw path still takes the uniform, rodata,
 * varying and preamble layout from the API shader, so the monolithic
 * program is used only if that layout is byte-for-byte the same; any
 * difference, unsupported shape or compile refusal returns NULL and the
 * split main + epilog link runs unchanged.
 */
static const char *
hk_g17p_mono_layout_mismatch(const struct agx_shader_part *a,
                             const struct agx_shader_part *m, bool own_layout)
{
#define HK_MONO_FIELD(x)                                                       \
   if (a->info.x != m->info.x)                                                 \
      return #x;
   /* A program with its own constant program launches with its own uniform
    * layout (hk_linked_shader::g17p_layout): only the resources and varyings
    * the draw still takes from the API shader must agree. */
   if (own_layout)
      goto resources;
   /* The draw loads the main's whole uniform window, so the monolithic
    * program may use a prefix of it: every uniform it reads must mean the
    * same thing. Two parts of the window are placed by the compile itself:
    * the uniform-DMA span (AGX_G17P_UNIFORM_DMA) and promoted constants
    * (rodata). The monolithic program either has the main's DMA span or
    * stays below it (loading those values itself), and either has a prefix
    * of the main's rodata at the same base or stays below it. */
   if (m->info.push_count > a->info.push_count)
      return "push_count";
   if (m->info.agx3_uniform_dma_count) {
      HK_MONO_FIELD(agx3_uniform_dma_count)
      for (unsigned i = 0; i < m->info.agx3_uniform_dma_count; ++i) {
         if (!agx3_uniform_dma_span_equal(&a->info.agx3_uniform_dma[i],
                                          &m->info.agx3_uniform_dma[i]))
            return "agx3_uniform_dma";
      }
   } else if (a->info.agx3_uniform_dma_count &&
              m->info.push_count > a->info.agx3_uniform_dma[0].dest) {
      return "push_count over dma";
   }
   if (m->info.rodata.size_16 == 0) {
      if (a->info.rodata.size_16 &&
          m->info.push_count > a->info.rodata.base_uniform)
         return "push_count over rodata";
   } else {
      HK_MONO_FIELD(rodata.base_uniform)
      if (m->info.rodata.size_16 > a->info.rodata.size_16)
         return "rodata.size_16";
      if (!a->binary || !m->binary ||
          memcmp((const uint8_t *)a->binary + a->info.rodata.offset,
                 (const uint8_t *)m->binary + m->info.rodata.offset,
                 m->info.rodata.size_16 * 2))
         return "rodata";
   }
   HK_MONO_FIELD(has_preamble)
   HK_MONO_FIELD(preamble_size)
   HK_MONO_FIELD(nr_preamble_gprs)
   HK_MONO_FIELD(agx3_preamble_uniform_inputs)
   HK_MONO_FIELD(agx3_preamble_tps_inputs)
   HK_MONO_FIELD(preamble_scratch_size)
   HK_MONO_FIELD(agx3_preamble_staging_known)
   HK_MONO_FIELD(agx3_preamble_staging_halfs)
   HK_MONO_FIELD(agx3_preamble_inputs.count)
   HK_MONO_FIELD(agx3_preamble_inputs.storage_bytes)
   if (a->info.has_preamble) {
      /* G17 graphics Ioto still launches the API shader's constant program
       * (hk_upload_usc_words_internal), while the linked main comes from m.
       * Equal sizes/register counts do not prove that their uniform values
       * agree. Keep the split link unless code and its frozen live-ins agree. */
      if (memcmp(a->info.agx3_preamble_inputs.halves,
                 m->info.agx3_preamble_inputs.halves,
                 sizeof(a->info.agx3_preamble_inputs.halves)))
         return "preamble inputs";
      if (!a->binary || !m->binary || !a->info.preamble_size ||
          a->info.preamble_offset > a->info.binary_size ||
          m->info.preamble_offset > m->info.binary_size ||
          a->info.preamble_size > a->info.binary_size - a->info.preamble_offset ||
          m->info.preamble_size > m->info.binary_size - m->info.preamble_offset)
         return "preamble extent";
      if (memcmp((const uint8_t *)a->binary + a->info.preamble_offset,
                 (const uint8_t *)m->binary + m->info.preamble_offset,
                 a->info.preamble_size))
         return "preamble code";
   }
resources:
   HK_MONO_FIELD(texture_state_count)
   HK_MONO_FIELD(sampler_state_count)
   HK_MONO_FIELD(uses_txf)
   HK_MONO_FIELD(uses_sampler_heap)
   HK_MONO_FIELD(g17p_sampled_fragment)
   HK_MONO_FIELD(uses_draw_id)
   HK_MONO_FIELD(uses_base_param)
   HK_MONO_FIELD(early_fragment_tests)
#undef HK_MONO_FIELD
   if (memcmp(&a->info.varyings.fs, &m->info.varyings.fs,
              sizeof(a->info.varyings.fs)))
      return "varyings";
   return NULL;
}

/* Replace the epilog's load_exported_agx with the main's exported values.
 * Returns false (leaving the shader unusable) for any unsupported shape. */
static bool
hk_g17p_mono_resolve_exports(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_scalar slot[256];
   uint8_t slot_bits[256] = {0};

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_export_agx) {
            nir_def *v = intr->src[0].ssa;
            if (v->bit_size != 16 && v->bit_size != 32)
               return false;
            const unsigned step = v->bit_size / 16;
            for (unsigned c = 0; c < v->num_components; ++c) {
               const unsigned h = nir_intrinsic_base(intr) + c * step;
               if (h >= ARRAY_SIZE(slot) || slot_bits[h])
                  return false;
               slot[h] = nir_get_scalar(v, c);
               slot_bits[h] = v->bit_size;
            }
            nir_instr_remove(instr);
         } else if (intr->intrinsic == nir_intrinsic_load_exported_agx) {
            nir_def *d = &intr->def;
            if (d->bit_size != 16 && d->bit_size != 32)
               return false;
            const unsigned step = d->bit_size / 16;
            nir_def *comps[NIR_MAX_VEC_COMPONENTS];
            nir_builder b = nir_builder_at(nir_before_instr(instr));
            for (unsigned c = 0; c < d->num_components; ++c) {
               const unsigned h = nir_intrinsic_base(intr) + c * step;
               if (h >= ARRAY_SIZE(slot))
                  return false;
               if (!slot_bits[h]) {
                  /* Never exported: the register was undefined too. */
                  comps[c] = nir_undef(&b, 1, d->bit_size);
               } else if (slot_bits[h] != d->bit_size) {
                  return false;
               } else {
                  comps[c] = nir_mov_scalar(&b, slot[h]);
               }
            }
            nir_def_replace(d, nir_vec(&b, comps, d->num_components));
         }
      }
   }
   return true;
}

/* Whether an inline-blend program carries its own constant program, which
 * only a program built for an API shader without one does. */
static bool
hk_g17p_mono_own_layout(const struct hk_shader *shader,
                        const struct agx_shader_part *part)
{
   return part->info.has_preamble && !shader->b.info.has_preamble;
}

/* Every G17P fragment program that touches the tile opens the pixel
 * transaction with the entry gate and ends it with a final release
 * (PIXREL with 0x200) just before END. The packer only emits that
 * protocol for shapes it has proven; a shape it does not recognise
 * would come back without it rather than failing, so check the bytes. */
static const char *
hk_g17p_mono_part_refusal(const struct hk_shader *shader,
                          const struct agx_shader_part *part)
{
   static const uint8_t gate[6] = {0x87, 0x02, 0x54, 0x00, 0x06, 0x00};
   const uint8_t *code = (const uint8_t *)part->binary + part->info.main_offset;
   const size_t n = part->info.main_size;
   bool has_gate = false;
   for (size_t o = 0; o + sizeof(gate) <= n && !has_gate; o += 2)
      has_gate = !memcmp(code + o, gate, sizeof(gate));
   const bool final_release = n >= 10 && code[n - 4] == 0x0e &&
      code[n - 10] == 0x07 && code[n - 9] == 0x02 && code[n - 8] == 0x54 &&
      (code[n - 6] & 0x02) && code[n - 5] == 0x00;
   if (!has_gate || !final_release)
      return "pixel protocol not proven";

   const bool own_layout = hk_g17p_mono_own_layout(shader, part);
   if (own_layout && !hk_g17p_native_cprog_qualified(&part->info))
      return "constant program outside the native protocol";
   return hk_g17p_mono_layout_mismatch(&shader->b, part, own_layout);
}

/* An inline-blend program is a function of the main's NIR and compile key,
 * the colour epilog key and the tile partitions (the build is part of the
 * cache's own identity). */
static bool
hk_g17p_mono_disk_key(struct hk_device *dev, const struct hk_shader *shader,
                      const struct agx_fs_epilog_key *ekey,
                      const BITSET_WORD *partitions, cache_key out)
{
   struct disk_cache *cache = dev->vk.physical->disk_cache;
   if (!cache)
      return false;

   BITSET_DECLARE(parts, AGX_G17P_MAX_BYTES_PER_SAMPLE) = {0};
   if (partitions)
      memcpy(parts, partitions, sizeof(parts));

   const char magic[8] = "hkmono3";
   const size_t rodata_size =
      shader->b.binary ? shader->b.info.rodata.size_16 * 2 : 0;
   const size_t size = sizeof(magic) + sizeof(shader->g17p_mono_key) +
                       sizeof(*ekey) + sizeof(parts) + rodata_size +
                       shader->g17p_mono_nir_size;
   uint8_t *buf = malloc(size);
   if (!buf)
      return false;

   uint8_t *p = buf;
   memcpy(p, magic, sizeof(magic));
   p += sizeof(magic);
   memcpy(p, &shader->g17p_mono_key, sizeof(shader->g17p_mono_key));
   p += sizeof(shader->g17p_mono_key);
   memcpy(p, ekey, sizeof(*ekey));
   p += sizeof(*ekey);
   memcpy(p, parts, sizeof(parts));
   p += sizeof(parts);
   if (rodata_size)
      memcpy(p, (const uint8_t *)shader->b.binary + shader->b.info.rodata.offset,
             rodata_size);
   p += rodata_size;
   memcpy(p, shader->g17p_mono_nir, shader->g17p_mono_nir_size);

   disk_cache_compute_key(cache, buf, size, out);
   free(buf);
   return true;
}

static struct agx_shader_part *
hk_g17p_inline_epilog_part(struct hk_device *dev, struct hk_shader *shader,
                           const struct agx_fs_epilog_key *ekey,
                           const BITSET_WORD *partitions)
{
   /* A refusal of these inputs is final: remember it with the programs. */
#define HK_MONO_REFUSE(why)                                                    \
   do {                                                                        \
      if (shader->info.fs.m5_inline_sample_shading)                             \
         mesa_loge("M5 inline sample shading: %s", why);                        \
      final_refusal = true;                                                    \
      goto fail;                                                               \
   } while (0)
#define HK_MONO_FAIL(why) goto fail
   static const char refused_marker[8] = "hkmonoN";

   void *mem = ralloc_context(NULL);
   struct agx_shader_part *part = NULL;
   nir_shader *nir = NULL;
   nir_builder eb = {0};
   cache_key disk_key;
   bool disk = false, final_refusal = false;

   if (!mem)
      return NULL;
   const unsigned main_samples = MAX2(shader->g17p_mono_key.fs.nr_samples, 1);
   const bool inline_samples = shader->info.fs.m5_inline_sample_shading;
   if ((ekey->native_spills && !inline_samples) ||
       (ekey->link.sample_shading && !inline_samples) ||
       MAX2(ekey->nr_samples, 1) != main_samples)
      HK_MONO_REFUSE("epilog shape");

   disk = hk_g17p_mono_disk_key(dev, shader, ekey, partitions, disk_key);
   if (disk) {
      size_t size = 0;
      uint8_t *data =
         disk_cache_get(dev->vk.physical->disk_cache, disk_key, &size);
      if (data) {
         const bool refused = size == sizeof(refused_marker) &&
                              !memcmp(data, refused_marker, size);
         part = refused ? NULL : hk_part_from_blob(data, size, NULL);
         free(data);
         if (refused)
            HK_MONO_FAIL("refused before");
         if (part) {
            struct agx_shader_part expected = shader->b;
            if (inline_samples && ekey->native_spills)
               expected.info.push_count = MAX2(expected.info.push_count,
                                               ekey->emrt_heap_base + 4);
            if (inline_samples
                   ? hk_g17p_mono_layout_mismatch(&expected, part, false)
                   : hk_g17p_mono_part_refusal(shader, part))
               HK_MONO_FAIL("cached program");
            ralloc_free(mem);
            return part;
         }
      }
   }

   struct blob_reader reader;
   blob_reader_init(&reader, shader->g17p_mono_nir, shader->g17p_mono_nir_size);
   nir = nir_deserialize(mem, &agx_nir_options, &reader);
   if (!nir || reader.overrun)
      HK_MONO_FAIL("nir");

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic ==
                nir_intrinsic_load_exported_agx)
            HK_MONO_REFUSE("main reads a prolog export");
      }
   }

   /* A main that exports only uniform values (say, a colour from push
    * constants) leaves every source term of the blend uniform: its
    * clamps and factor products are the same for every pixel. Compute them
    * once in the constant program and blend with one FMA per channel by
    * letting this program form its own
    * constant program when the API shader has none (a push-constant-only
    * fragment main forms none, see hk_compile_nir()). Only chips whose
    * native constant program is qualified form one. */
   bool uniform_exports = agx_chip_native_preamble(dev->dev.chip) &&
                          shader->g17p_mono_key.no_preamble;
   if (uniform_exports) {
      nir_divergence_analysis(nir);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_export_agx &&
                nir_instr_as_intrinsic(instr)->src[0].ssa->divergent)
               uniform_exports = false;
         }
      }
   }

   /* The epilog, exactly as the split path builds it. */
   eb = nir_builder_init_simple_shader(0, &agx_nir_options, NULL);
   if (inline_samples) {
      struct agx_fs_epilog_key pixel = *ekey;
      pixel.link.sample_shading = false;
      pixel.defer_monolithic_msaa = true;
      agx_nir_fs_epilog(&eb, &pixel);
      agx_nir_lower_sample_exports_to_end(nir);
   } else {
      build_g17p_fs_epilog(&eb, ekey);
   }
   agx_preprocess_nir(eb.shader);
   if (!inline_samples)
      NIR_PASS(_, eb.shader, agx_nir_lower_g17p_tile_samples, ekey->nr_samples);

   /* Epilog code reading uniforms (blend constants and the like) would
    * need uniforms the main's layout does not reserve. */
   nir_foreach_block(block, eb.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         const nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
         if (inline_samples && ekey->native_spills &&
             (op == nir_intrinsic_bindless_image_agx ||
              op == nir_intrinsic_bindless_image_store ||
              (op == nir_intrinsic_load_preamble &&
               nir_intrinsic_base(nir_instr_as_intrinsic(instr)) ==
                  ekey->emrt_heap_base)))
            continue;
         if (op == nir_intrinsic_load_preamble ||
             op == nir_intrinsic_load_blend_const_color_rgba ||
             op == nir_intrinsic_load_blend_const_color_r_float ||
             op == nir_intrinsic_load_blend_const_color_g_float ||
             op == nir_intrinsic_load_blend_const_color_b_float ||
             op == nir_intrinsic_load_blend_const_color_a_float ||
             op == nir_intrinsic_bindless_image_agx ||
             op == nir_intrinsic_image_store ||
             op == nir_intrinsic_bindless_image_store)
            HK_MONO_REFUSE("epilog uses uniforms or images");
      }
   }

   {
      struct hash_table *remap = _mesa_pointer_hash_table_create(mem);
      nir_builder b = nir_builder_at(nir_after_impl(impl));
      nir_inline_function_impl(&b, eb.impl, NULL, remap);
      /* The inlined definitions keep the epilog's SSA indices. */
      nir_index_ssa_defs(impl);
      nir_progress(true, impl, nir_metadata_none);
   }
   nir->info.outputs_read |= eb.shader->info.outputs_read;
   nir->info.outputs_written |= eb.shader->info.outputs_written;
   nir->info.fs.uses_fbfetch_output |= eb.shader->info.fs.uses_fbfetch_output;
   nir->info.fs.uses_sample_shading |= eb.shader->info.fs.uses_sample_shading;
   nir->info.writes_memory |= eb.shader->info.writes_memory;
   ralloc_free(eb.shader);
   eb.shader = NULL;

   if (!hk_g17p_mono_resolve_exports(nir))
      HK_MONO_REFUSE("exports");
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_dce);
   if (inline_samples) {
      struct agx_g17p_colour_layout layout;
      agx_g17p_colour_layout_init(&layout, ekey->link.loc_written,
                                  ekey->link.size_32, ekey->link.colour_comps,
                                  ekey->link.packed_colour);
      NIR_PASS(_, nir, agx_nir_lower_to_per_sample);
      NIR_PASS(_, nir, agx_nir_unroll_g17p_samples_layout,
               ekey->nr_samples, &layout);
      bool progress;
      do {
         progress = false;
         NIR_PASS(progress, nir, nir_opt_constant_folding);
         NIR_PASS(progress, nir, nir_opt_algebraic);
      } while (progress);
      NIR_PASS(_, nir, agx_nir_hoist_g17p_zs_writes);
      NIR_PASS(_, nir, agx_nir_lower_g17p_tile_samples, ekey->nr_samples);
      nir->info.fs.uses_sample_shading = false;
   }

   struct agx_shader_key key = shader->g17p_mono_key;
   if (inline_samples)
      key.fs.inside_sample_loop = false;
   key.no_stop = false;
   key.secondary = false;
   key.g17p_emrt = inline_samples && ekey->native_spills;
   if (shader->b.info.rodata.size_16 && shader->b.binary) {
      key.g17p_rodata_hint =
         (const uint16_t *)((const uint8_t *)shader->b.binary +
                            shader->b.info.rodata.offset);
      key.g17p_rodata_hint_base = shader->b.info.rodata.base_uniform;
      key.g17p_rodata_hint_size_16 = shader->b.info.rodata.size_16;
   }
   if (partitions)
      memcpy(key.g17p_tile_partitions, partitions,
             sizeof(key.g17p_tile_partitions));

   /* With uniform exports, first try the program with its own constant
    * program; if the backend or the native protocol refuses it, build the
    * program exactly as without. */
   nir_shader *without = NULL;
   if (uniform_exports) {
      without = nir_shader_clone(mem, nir);
      if (!without)
         HK_MONO_FAIL("oom");
      key.no_preamble = false;
      key.g17p_native_uniforms = true;
   }

   /* An M5 inline-samples program keeps the main's layout (its emulated
    * render target heap may extend the push window); any other program must
    * also follow the pixel protocol (hk_g17p_mono_part_refusal()). Uniform
    * exports are G17P-only (agx_chip_native_preamble()), so an inline-samples
    * program is compiled once. */
   struct agx_shader_part expected = shader->b;
   if (inline_samples && ekey->native_spills)
      expected.info.push_count = MAX2(expected.info.push_count,
                                      ekey->emrt_heap_base + 4);
   const char *why;
   for (;;) {
      part = rzalloc(NULL, struct agx_shader_part);
      if (!part)
         HK_MONO_FAIL("oom");
      /* agx_compile_shader_nir frees nothing it did not allocate. */
      const bool compiled = agx_compile_shader_nir(nir, &key, part);
      nir = NULL; /* consumed */
      why = !compiled ? "backend refused the monolithic program"
            : inline_samples ? hk_g17p_mono_layout_mismatch(&expected, part, false)
                             : hk_g17p_mono_part_refusal(shader, part);
      if (!why || !without)
         break;

      free(part->binary);
      ralloc_free(part);
      part = NULL;
      nir = without;
      without = NULL;
      key.no_preamble = shader->g17p_mono_key.no_preamble;
      key.g17p_native_uniforms = shader->g17p_mono_key.g17p_native_uniforms;
   }
   if (why)
      HK_MONO_REFUSE(why);

   if (disk)
      hk_part_disk_put(dev, disk_key, part);
   ralloc_free(mem);
   return part;

fail:
   if (disk && final_refusal)
      disk_cache_put(dev->vk.physical->disk_cache, disk_key, refused_marker,
                     sizeof(refused_marker), NULL);
   if (eb.shader)
      ralloc_free(eb.shader);
   if (part) {
      free(part->binary);
      ralloc_free(part);
   }
   ralloc_free(mem);
   return NULL;
#undef HK_MONO_REFUSE
#undef HK_MONO_FAIL
}

static struct hk_linked_shader *
hk_get_fast_linked_locked_fs(struct hk_device *dev, struct hk_shader *shader,
                             struct hk_fast_link_key_fs *key)
{
   /* TODO: prolog without fs needs to work too... */
   bool needs_prolog = key->prolog.statistics || key->prolog.run_zs_tests ||
                       key->prolog.cull_distance_size ||
                       key->prolog.api_sample_mask != 0xff;

   struct agx_shader_part *prolog = NULL;
   if (needs_prolog) {
      prolog = hk_get_prolog_epilog(dev, &key->prolog, sizeof(key->prolog),
                                    build_fs_prolog, false, false,
                                    key->prolog.cf_base, 0,
                                    key->prolog.nr_samples);
      if (!prolog) {
         mesa_loge("fragment prolog compilation failed: stats=%u cull=%u "
                   "api-mask=%#x run-zs=%u cf-base=%u",
                   key->prolog.statistics, key->prolog.cull_distance_size,
                   key->prolog.api_sample_mask, key->prolog.run_zs_tests,
                   key->prolog.cf_base);
         return NULL;
      }
   }

   /* If sample shading is used, don't stop at the epilog, there's a
    * footer that the fast linker will insert to stop.
    */
   bool epilog_stop = (key->nr_samples_shaded == 0);

   /* A G17P epilog selects each written target's pixel classes from its
    * position among all of the framebuffer's colour targets. */
   BITSET_DECLARE(g17p_tile_partitions, AGX_G17P_MAX_BYTES_PER_SAMPLE) = {0};
   if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
      const struct agx_tilebuffer_layout tib =
         key->epilog.native_imageblock
            ? agx_build_tilebuffer_layout_resident(
                 key->epilog.rt_formats, ARRAY_SIZE(key->epilog.rt_formats),
                 key->epilog.nr_samples, true, key->epilog.max_sample_size_B,
                 key->epilog.max_bytes_per_tile,
                 key->epilog.max_samples_per_tile, key->epilog.resident_mask)
            : agx_build_tilebuffer_layout_with_limits(
                 key->epilog.rt_formats, ARRAY_SIZE(key->epilog.rt_formats),
                 key->epilog.nr_samples, true, key->epilog.max_sample_size_B,
                 key->epilog.max_bytes_per_tile,
                 key->epilog.max_samples_per_tile);
      for (unsigned rt = 0; rt < ARRAY_SIZE(tib.logical_format); ++rt) {
         if (tib.logical_format[rt] != PIPE_FORMAT_NONE && !tib.spilled[rt])
            BITSET_SET(g17p_tile_partitions, tib._offset_B[rt]);
      }
      if (BITSET_COUNT(g17p_tile_partitions) < 2)
         BITSET_ZERO(g17p_tile_partitions);
   }

   /* Inline blend: try one monolithic fragment program first;
    * any refusal falls through to the split link below, unchanged. */
   if ((shader->info.fs.m5_inline_sample_shading ||
        (agx_native_graphics(&dev->dev) &&
         (agx_get_fs_structure_flags() & AGX_FS_STRUCTURE_INLINE_BLEND))) &&
       !prolog && key->nr_samples_shaded == 0 && shader->g17p_mono_nir) {
      struct agx_shader_part *mono = hk_g17p_inline_epilog_part(
         dev, shader, &key->epilog, g17p_tile_partitions);
      if (mono) {
         struct hk_fast_link_key *mono_key_clone =
            ralloc_memdup(shader->linked.ht, key, sizeof(*key));
         /* A program with its own constant program brings its own uniform
          * layout; link it as the main of a shader describing that layout. */
         struct hk_shader *layout =
            mono_key_clone && hk_g17p_mono_own_layout(shader, mono)
               ? hk_g17p_layout_shader_create(dev, shader, mono)
               : NULL;
         struct hk_linked_shader *mono_linked = NULL;
         if (layout)
            mono_linked = hk_fast_link_part(dev, true, layout, NULL, NULL,
                                            NULL, 0);
         else if (mono_key_clone && !hk_g17p_mono_own_layout(shader, mono))
            mono_linked = hk_fast_link_part(dev, true, shader, mono, NULL,
                                            NULL, 0);
         free(mono->binary);
         ralloc_free(mono);
         if (mono_linked)
            mono_linked->g17p_layout = layout;
         else if (layout) {
            free(layout->b.binary);
            agx_bo_unreference(&dev->dev, layout->bo);
            free(layout);
         }
         if (mono_linked) {
            mono_linked->g17p_blend_constant_base =
               key->epilog.blend_constant_base;
            mono_linked->g17p_emrt_heap_base = key->epilog.emrt_heap_base;
            mono_linked->g17p_prolog_uses_root = key->prolog.statistics;
            _mesa_hash_table_insert(shader->linked.ht, mono_key_clone,
                                    mono_linked);
            return mono_linked;
         }
         ralloc_free(mono_key_clone);
      }
   }

   if (shader->info.fs.m5_inline_sample_shading) {
      mesa_loge("M5: required inline sample shading could not be linked");
      return NULL;
   }

   struct agx_shader_part *epilog =
      hk_get_prolog_epilog(dev, &key->epilog, sizeof(key->epilog),
                           (agx_native_graphics(&dev->dev) ||
                            (agx_chip_g17_m5(dev->dev.chip) &&
                             (key->epilog.link.sample_shading ||
                              key->epilog.native_spills)))
                              ? build_g17p_fs_epilog
                              : agx_nir_fs_epilog,
                           true, epilog_stop, 0, g17p_tile_partitions,
                           (agx_native_graphics(&dev->dev) ||
                            agx_chip_g17_m5(dev->dev.chip))
                              ? key->epilog.nr_samples
                              : 0);
   if (!epilog) {
      uint32_t link = 0;
      memcpy(&link, &key->epilog.link, sizeof(link));
      mesa_loge("fragment epilog compilation failed: link=%#010x samples=%u "
                "shaded=%u stop=%u small=%u preserve-zero=%u g17p-tile=%u "
                "formats=%x/%x/%x/%x remap=%d/%d/%d/%d",
                link, key->epilog.nr_samples, key->nr_samples_shaded,
                epilog_stop, key->epilog.force_small_tile,
                key->epilog.preserve_zero_mask, key->epilog.g17p_tile_formats,
                key->epilog.rt_formats[0], key->epilog.rt_formats[1],
                key->epilog.rt_formats[2], key->epilog.rt_formats[3],
                key->epilog.remap[0], key->epilog.remap[1],
                key->epilog.remap[2], key->epilog.remap[3]);
      return NULL;
   }

   struct hk_fast_link_key *key_clone =
      ralloc_memdup(shader->linked.ht, key, sizeof(*key));
   if (!key_clone) {
      mesa_loge("fragment fast-link key allocation failed");
      return NULL;
   }

   struct hk_linked_shader *linked =
      hk_fast_link(dev, true, shader, prolog, epilog, key->nr_samples_shaded);
   if (!linked) {
      uint32_t link_info = 0;
      memcpy(&link_info, &key->epilog.link, sizeof(link_info));
      mesa_loge("fragment fast-link failed: link=%#010x samples=%u shaded=%u "
                "prolog=%u main=%u epilog=%u",
                link_info, key->epilog.nr_samples, key->nr_samples_shaded,
                prolog != NULL, shader->b.binary != NULL,
                epilog->binary != NULL);
      ralloc_free(key_clone);
      return NULL;
   }
   linked->g17p_blend_constant_base = key->epilog.blend_constant_base;
   linked->g17p_emrt_heap_base = key->epilog.emrt_heap_base;
   linked->g17p_prolog_uses_root = key->prolog.statistics;

   _mesa_hash_table_insert(shader->linked.ht, key_clone, linked);
   return linked;
}

/*
 * First, look for a fully linked variant. Else, build the required shader
 * parts and link.
 */
static struct hk_linked_shader *
hk_get_fast_linked(struct hk_device *dev, struct hk_shader *shader, void *key)
{
   struct hk_linked_shader *linked;
   simple_mtx_lock(&shader->linked.lock);

   if (!hk_shader_ensure_link_ht_locked(shader)) {
      simple_mtx_unlock(&shader->linked.lock);
      return NULL;
   }

   struct hash_entry *ent = _mesa_hash_table_search(shader->linked.ht, key);

   if (ent) {
      linked = ent->data;
   } else {
      if (shader->info.stage == MESA_SHADER_VERTEX)
         linked = hk_get_fast_linked_locked_vs(dev, shader, key);
      else if (shader->info.stage == MESA_SHADER_FRAGMENT)
         linked = hk_get_fast_linked_locked_fs(dev, shader, key);
      else
         UNREACHABLE("invalid stage");
   }

   simple_mtx_unlock(&shader->linked.lock);
   return linked;
}

static __attribute__((hot)) bool
hk_update_fast_linked(struct hk_cmd_buffer *cmd, struct hk_shader *shader,
                      void *key)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   mesa_shader_stage stage = shader->info.stage;

   if (stage == MESA_SHADER_VERTEX && gfx->fast_link_vs.valid &&
       gfx->fast_link_vs.shader == shader &&
       memcmp(&gfx->fast_link_vs.key, key,
              sizeof(gfx->fast_link_vs.key)) == 0 &&
       gfx->linked[stage] != NULL)
      return true;

   if (stage == MESA_SHADER_FRAGMENT && gfx->fast_link_fs.valid &&
       gfx->fast_link_fs.shader == shader &&
       memcmp(&gfx->fast_link_fs.key, key,
              sizeof(gfx->fast_link_fs.key)) == 0 &&
       gfx->linked[stage] != NULL)
      return true;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_linked_shader *new = hk_get_fast_linked(dev, shader, key);
   if (!new) {
      mesa_loge("failed to compile or link required %s shader parts",
                _mesa_shader_stage_to_string(stage));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return false;
   }

   if (gfx->linked[stage] != new) {
      gfx->linked[stage] = new;
      gfx->linked_dirty |= BITFIELD_BIT(stage);
   }

   if (stage == MESA_SHADER_VERTEX) {
      gfx->fast_link_vs.shader = shader;
      memcpy(&gfx->fast_link_vs.key, key, sizeof(gfx->fast_link_vs.key));
      gfx->fast_link_vs.valid = true;
   } else if (stage == MESA_SHADER_FRAGMENT) {
      gfx->fast_link_fs.shader = shader;
      memcpy(&gfx->fast_link_fs.key, key, sizeof(gfx->fast_link_fs.key));
      gfx->fast_link_fs.valid = true;
   }
   return true;
}

static enum agx_polygon_mode
translate_polygon_mode(VkPolygonMode vk_mode)
{
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_FILL ==
                 AGX_POLYGON_MODE_FILL);
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_LINE ==
                 AGX_POLYGON_MODE_LINE);
   static_assert((enum agx_polygon_mode)VK_POLYGON_MODE_POINT ==
                 AGX_POLYGON_MODE_POINT);

   assert(vk_mode <= VK_POLYGON_MODE_POINT);
   return (enum agx_polygon_mode)vk_mode;
}

static enum agx_zs_func
translate_compare_op(VkCompareOp vk_mode)
{
   static_assert((enum agx_zs_func)VK_COMPARE_OP_NEVER == AGX_ZS_FUNC_NEVER);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_LESS == AGX_ZS_FUNC_LESS);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_EQUAL == AGX_ZS_FUNC_EQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_LESS_OR_EQUAL ==
                 AGX_ZS_FUNC_LEQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_GREATER ==
                 AGX_ZS_FUNC_GREATER);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_NOT_EQUAL ==
                 AGX_ZS_FUNC_NOT_EQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_GREATER_OR_EQUAL ==
                 AGX_ZS_FUNC_GEQUAL);
   static_assert((enum agx_zs_func)VK_COMPARE_OP_ALWAYS == AGX_ZS_FUNC_ALWAYS);

   assert(vk_mode <= VK_COMPARE_OP_ALWAYS);
   return (enum agx_zs_func)vk_mode;
}

static enum agx_stencil_op
translate_stencil_op(VkStencilOp vk_op)
{
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_KEEP ==
                 AGX_STENCIL_OP_KEEP);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_ZERO ==
                 AGX_STENCIL_OP_ZERO);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_REPLACE ==
                 AGX_STENCIL_OP_REPLACE);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INCREMENT_AND_CLAMP ==
                 AGX_STENCIL_OP_INCR_SAT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_DECREMENT_AND_CLAMP ==
                 AGX_STENCIL_OP_DECR_SAT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INVERT ==
                 AGX_STENCIL_OP_INVERT);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_INCREMENT_AND_WRAP ==
                 AGX_STENCIL_OP_INCR_WRAP);
   static_assert((enum agx_stencil_op)VK_STENCIL_OP_DECREMENT_AND_WRAP ==
                 AGX_STENCIL_OP_DECR_WRAP);

   return (enum agx_stencil_op)vk_op;
}

static struct agx_fragment_stencil_packed
hk_pack_stencil_face(struct vk_stencil_test_face_state s, bool enabled)
{
   struct agx_fragment_stencil_packed packed;

   if (enabled) {
      agx_pack(&packed, FRAGMENT_STENCIL, cfg) {
         cfg.compare = translate_compare_op(s.op.compare);
         cfg.write_mask = s.write_mask;
         cfg.read_mask = s.compare_mask;

         cfg.depth_pass = translate_stencil_op(s.op.pass);
         cfg.depth_fail = translate_stencil_op(s.op.depth_fail);
         cfg.stencil_fail = translate_stencil_op(s.op.fail);
      }
   } else {
      agx_pack(&packed, FRAGMENT_STENCIL, cfg) {
         cfg.compare = AGX_ZS_FUNC_ALWAYS;
         cfg.write_mask = 0xFF;
         cfg.read_mask = 0xFF;

         cfg.depth_pass = AGX_STENCIL_OP_KEEP;
         cfg.depth_fail = AGX_STENCIL_OP_KEEP;
         cfg.stencil_fail = AGX_STENCIL_OP_KEEP;
      }
   }

   return packed;
}

static void
hk_ppp_push_stencil_face(struct agx_ppp_update *ppp,
                         struct vk_stencil_test_face_state s, bool enabled)
{
   struct agx_fragment_stencil_packed packed =
      hk_pack_stencil_face(s, enabled);
   agx_ppp_push_packed(ppp, &packed, FRAGMENT_STENCIL);
}

static bool
hk_stencil_test_enabled(struct hk_cmd_buffer *cmd)
{
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   return dyn->ds.stencil.test_enable &&
          render->stencil_att.vk_format != VK_FORMAT_UNDEFINED;
}

static bool
hk_emit_hal300_vp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                        enum agx_chip chip, unsigned viewport_count,
                        unsigned scissor_index, const unsigned minx[],
                        const unsigned miny[], const unsigned maxx[],
                        const unsigned maxy[], const VkViewport viewports[],
                        uint8_t **out)
{
   if (!out || !*out || viewport_count == 0 ||
       viewport_count > AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS ||
       !agx_ppp_hal300_depth_bias_scissor_core_supported(chip))
      return false;

   const size_t scissor_offset =
      (size_t)scissor_index * AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH;
   const size_t scissor_size =
      (size_t)viewport_count * AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH;
   if (scissor_offset > cs->scissor.size ||
       scissor_size > cs->scissor.size - scissor_offset)
      return false;

   const uint64_t scissor_base =
      hk_pool_upload(cmd, (const uint8_t *)cs->scissor.data + scissor_offset,
                     scissor_size, 64);

   static const uint8_t zero_depth_bias[AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH];
   const void *depth_bias = zero_depth_bias;
   if (cs->depth_bias.size >= AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH) {
      depth_bias = (const uint8_t *)cs->depth_bias.data + cs->depth_bias.size -
                   AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH;
   }

   const uint64_t depth_bias_base =
      hk_pool_upload(cmd, depth_bias,
                     AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH, 64);
   if (!scissor_base || !depth_bias_base)
      return false;

   struct agx_ppp_hal300_depth_bias_scissor_bases bases;
   const struct agx_ppp_hal300_depth_bias_scissor_bases_fields base_fields = {
      .depth_bias_base = depth_bias_base,
      .scissor_base = scissor_base,
   };
   if (!agx_ppp_hal300_depth_bias_scissor_bases_pack(chip, &base_fields,
                                                     &bases))
      return false;

   struct agx_ppp_viewport_payload_fields viewport_fields = {
      .viewport_count = viewport_count,
   };

   for (unsigned i = 0; i < viewport_count; ++i) {
      uint8_t region_clip[AGX_REGION_CLIP_LENGTH];
      agx_pack(region_clip, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx[i] / 32;
         cfg.min_y = miny[i] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx[i], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy[i], 1), 32);
      }
      memcpy(&viewport_fields.region_clip_raw[i], region_clip,
             sizeof(region_clip));

      const VkViewport *vp = &viewports[i];
      viewport_fields.viewports[i] =
         (struct agx_ppp_viewport_transform_raw){
            .translate_x = fui(vp->x + 0.5f * vp->width),
            .scale_x = fui(vp->width * 0.5f),
            .translate_y = fui(vp->y + 0.5f * vp->height),
            .scale_y = fui(vp->height * 0.5f),
            .translate_z = fui(vp->minDepth),
            .scale_z = fui(vp->maxDepth - vp->minDepth),
         };
   }

   uint8_t viewport_control[AGX_VIEWPORT_CONTROL_LENGTH];
   agx_pack(viewport_control, VIEWPORT_CONTROL, cfg)
      ;
   memcpy(&viewport_fields.viewport_control_raw, viewport_control,
          sizeof(viewport_control));

   struct agx_ppp_viewport_payload_core viewport;
   if (!agx_ppp_viewport_payload_core_pack(chip, &viewport_fields, &viewport))
      return false;

   const uint64_t bases_gpu =
      hk_pool_upload(cmd, bases.bytes, bases.length, AGX_PPP_HEADER_ALIGN);
   const uint64_t viewport_gpu =
      hk_pool_upload(cmd, viewport.bytes, viewport.length,
                     AGX_PPP_HEADER_ALIGN);
   if (!bases_gpu || !viewport_gpu)
      return false;

   /* Keep emission transactional: both exact PPP pointers are staged before
    * the caller's live control-stream cursor is advanced. HAL300's initial
    * render-state path emits the viewport update before the base update. */
   uint8_t records[2 * AGX_PPP_STATE_LENGTH];
   uint8_t *record = records;
   if (!hk_ppp_emit_state(chip, &record, viewport_gpu, viewport.length) ||
       !hk_ppp_emit_state(chip, &record, bases_gpu, bases.length) ||
       record != records + sizeof(records))
      return false;

   memcpy(*out, records, sizeof(records));
   *out += sizeof(records);
   return true;
}












static unsigned
hk_effective_viewport_count(struct hk_cmd_buffer *cmd)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   if (gfx->shaders[MESA_SHADER_MESH]) {
      const struct hk_mesh_info *mesh = &hk_bound_hw_vs(gfx)->info.mesh;
      if (!((mesh->vertex_outputs | mesh->primitive_outputs) &
            (VARYING_BIT_VIEWPORT | VARYING_BIT_LAYER)))
         return 1;
   }
   return MAX2(dyn->vp.viewport_count, 1);
}

static void
hk_flush_vp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const bool hal300 =
      agx_chip_command_stream_generation(dev->dev.chip) ==
      AGX_CS_GENERATION_HAL300;
   /* The viewport records are PPP state too: re-emit the graphics core after
    * them rather than rely on what the stream latched before. */
   cs->m5_ppp_emitted_valid = false;
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   static_assert(AGX_SCISSOR_LENGTH ==
                 AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH);

   /* We always need at least 1 viewport for the hardware. With rasterizer
    * discard the app may not supply any, but we can just program garbage.
    */
   const unsigned count = hk_effective_viewport_count(cmd);
   const unsigned api_viewports = MIN2(dyn->vp.viewport_count, count);
   const unsigned api_scissors = MIN2(dyn->vp.scissor_count, count);

   unsigned minx[HK_MAX_VIEWPORTS] = {0}, miny[HK_MAX_VIEWPORTS] = {0};
   unsigned maxx[HK_MAX_VIEWPORTS] = {0}, maxy[HK_MAX_VIEWPORTS] = {0};

   /* We implicitly scissor to the viewport. We need to do a min/max dance to
    * handle inverted viewports.
    */
   for (uint32_t i = 0; i < api_viewports; i++) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      minx[i] = MIN2(vp->x, vp->x + vp->width);
      miny[i] = MIN2(vp->y, vp->y + vp->height);
      maxx[i] = MAX2(vp->x, vp->x + vp->width);
      maxy[i] = MAX2(vp->y, vp->y + vp->height);
   }

   /* Additionally clamp to the framebuffer so we don't rasterize
    * off-screen pixels. TODO: Is this necessary? the GL driver does this but
    * it might be cargoculted at this point.
    */
   for (unsigned i = 0; i < count; ++i) {
      minx[i] = MIN2(minx[i], cmd->state.gfx.render.cr.width);
      maxx[i] = MIN2(maxx[i], cmd->state.gfx.render.cr.width);
      miny[i] = MIN2(miny[i], cmd->state.gfx.render.cr.height);
      maxy[i] = MIN2(maxy[i], cmd->state.gfx.render.cr.height);
   }

   /* We additionally apply any API scissors */
   for (unsigned i = 0; i < api_scissors; ++i) {
      const VkRect2D *s = &dyn->vp.scissors[i];

      minx[i] = MAX2(minx[i], s->offset.x);
      miny[i] = MAX2(miny[i], s->offset.y);
      maxx[i] = MIN2(maxx[i], s->offset.x + s->extent.width);
      maxy[i] = MIN2(maxy[i], s->offset.y + s->extent.height);
   }

   if (hal300) {
      /* A disjoint intersection is empty, not an inverted 16-bit rectangle. */
      for (unsigned i = 0; i < count; ++i) {
         minx[i] = MIN2(minx[i], maxx[i]);
         miny[i] = MIN2(miny[i], maxy[i]);
      }
   }

   /* Upload a hardware scissor for each viewport, whether there's a
    * corresponding API scissor or not.
    */
   unsigned index = cs->scissor.size / AGX_SCISSOR_LENGTH;
   uint8_t *scissors =
      util_dynarray_grow_bytes(&cs->scissor, count, AGX_SCISSOR_LENGTH);

   for (unsigned i = 0; i < count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      /* These settings in conjunction with the PPP control depth clip/clamp
       * settings implement depth clip/clamping. Properly setting them
       * together is required for conformant depth clip enable.
       *
       * TODO: Reverse-engineer the finer interactions here.
       */
      const float min_z = dyn->rs.depth_clamp_enable
                             ? MIN2(vp->minDepth, vp->maxDepth)
                             : 0.0f;
      const float max_z = dyn->rs.depth_clamp_enable
                             ? MAX2(vp->minDepth, vp->maxDepth)
                             : 1.0f;
      uint8_t *record = scissors + i * AGX_SCISSOR_LENGTH;

      if (hal300) {
         const struct agx_ppp_hal300_scissor_record_fields fields = {
            .max_x = maxx[i],
            .min_x = minx[i],
            .max_y = maxy[i],
            .min_y = miny[i],
            .min_z_raw = fui(min_z),
            .max_z_raw = fui(max_z),
         };
         struct agx_ppp_hal300_scissor_record exact;
         if (!agx_ppp_hal300_scissor_record_pack(dev->dev.chip, &fields,
                                                  &exact)) {
            mesa_loge("no grounded HAL300 scissor record for %s",
                      agx_chip_name(dev->dev.chip));
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         memcpy(record, exact.bytes, exact.length);
      } else {
         agx_pack(record, SCISSOR, cfg) {
            cfg.min_x = minx[i];
            cfg.min_y = miny[i];
            cfg.max_x = maxx[i];
            cfg.max_y = maxy[i];
            cfg.min_z = min_z;
            cfg.max_z = max_z;
         }
      }
   }

   if (hal300) {
      if (!hk_emit_hal300_vp_state(cmd, cs, dev->dev.chip, count, index, minx,
                                   miny, maxx, maxy, dyn->vp.viewports, out)) {
         mesa_loge("no grounded HAL300 viewport/base emitter for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      }
      return;
   }

   /* Upload state */
   struct AGX_PPP_HEADER present = {
      .depth_bias_scissor = true,
      .region_clip = true,
      .viewport = true,
      .viewport_count = count,
   };

   size_t size = agx_ppp_update_size(&present);
   struct agx_ptr T = hk_pool_vdm_alloc(cmd, size, AGX_PPP_HEADER_ALIGN);
   if (!T.cpu)
      return;

   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &present);

   agx_ppp_push(&ppp, DEPTH_BIAS_SCISSOR, cfg) {
      cfg.scissor = index;

      /* Use the current depth bias, we allocate linearly */
      unsigned count = cs->depth_bias.size / AGX_DEPTH_BIAS_LENGTH;
      cfg.depth_bias = count ? count - 1 : 0;
   };

   for (unsigned i = 0; i < count; ++i) {
      agx_ppp_push(&ppp, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx[i] / 32;
         cfg.min_y = miny[i] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx[i], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy[i], 1), 32);
      }
   }

   agx_ppp_push(&ppp, VIEWPORT_CONTROL, cfg)
      ;

   /* Upload viewports */
   for (unsigned i = 0; i < count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];

      agx_ppp_push(&ppp, VIEWPORT, cfg) {
         cfg.translate_x = vp->x + 0.5f * vp->width;
         cfg.translate_y = vp->y + 0.5f * vp->height;
         cfg.translate_z = vp->minDepth;

         cfg.scale_x = vp->width * 0.5f;
         cfg.scale_y = vp->height * 0.5f;
         cfg.scale_z = vp->maxDepth - vp->minDepth;
      }
   }

   const bool ppp_ok = agx_ppp_legacy_payload_supported(dev->dev.chip) &&
                       agx_ppp_fini(dev->dev.chip, out, &ppp);
   if (!ppp_ok) {
      mesa_loge("no grounded legacy PPP payload/finalizer for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   }
}

static enum agx_object_type
translate_object_type(enum mesa_prim topology)
{
   static_assert(MESA_PRIM_LINES < MESA_PRIM_LINE_STRIP);
   static_assert(MESA_PRIM_TRIANGLES >= MESA_PRIM_LINE_STRIP);

   if (topology == MESA_PRIM_POINTS)
      return AGX_OBJECT_TYPE_POINT_SPRITE_UV01;
   else if (topology <= MESA_PRIM_LINE_STRIP)
      return AGX_OBJECT_TYPE_LINE;
   else
      return AGX_OBJECT_TYPE_TRIANGLE;
}

static enum agx_primitive
translate_hw_primitive_topology(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return AGX_PRIMITIVE_POINTS;
   case MESA_PRIM_LINES:
      return AGX_PRIMITIVE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return AGX_PRIMITIVE_LINE_STRIP;
   case MESA_PRIM_TRIANGLES:
      return AGX_PRIMITIVE_TRIANGLES;
   case MESA_PRIM_TRIANGLE_STRIP:
      return AGX_PRIMITIVE_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return AGX_PRIMITIVE_TRIANGLE_FAN;
   default:
      UNREACHABLE("Invalid hardware primitive topology");
   }
}

static inline enum agx_vdm_vertex
translate_vdm_vertex(unsigned vtx)
{
   static_assert(AGX_VDM_VERTEX_0 == 0);
   static_assert(AGX_VDM_VERTEX_1 == 1);
   static_assert(AGX_VDM_VERTEX_2 == 2);

   assert(vtx <= 2);
   return vtx;
}

static inline enum agx_ppp_vertex
translate_ppp_vertex(unsigned vtx)
{
   static_assert(AGX_PPP_VERTEX_0 == 0 + 1);
   static_assert(AGX_PPP_VERTEX_1 == 1 + 1);
   static_assert(AGX_PPP_VERTEX_2 == 2 + 1);

   assert(vtx <= 2);
   return vtx + 1;
}

/* The value a draw's indices are compared with for primitive restart. An
 * application draw restarts at the dynamic index, the custom one from
 * VK_EXT_primitive_restart_index or the default of the bound index type.
 * Geometry shading and transform feedback instead draw an index buffer that
 * poly generates, which restarts at the all-ones value of its own index size;
 * the same fixed value is kept for draws that do not restart.
 */
static uint32_t
hk_draw_restart_index(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   if (!draw.indexed)
      return 0;

   if (!draw.restart || cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY])
      return agx_primitive_restart_index(draw.index_size);

   return cmd->vk.dynamic_graphics_state.ia.primitive_restart_index;
}

static void
hk_flush_index(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_api_shader *gs = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
   uint32_t index = cmd->vk.dynamic_graphics_state.ia.primitive_restart_index;

   if (gs) {
      enum poly_gs_shape shape =
         gs->variants[HK_GS_VARIANT_COUNT].info.gs.shape;
      index = BITFIELD_MASK(8 * poly_gs_index_size(shape));
   }

   /* VDM State updates are relatively expensive, so only emit them when the
    * restart index changes. This is simpler than accurate dirty tracking.
    */
   if (!cs->restart_index_set || cs->restart_index != index) {
      uint8_t *out = cs->current;
      agx_push(out, VDM_STATE, cfg) {
         cfg.restart_index_present = true;
      }

      agx_push(out, VDM_STATE_RESTART_INDEX, cfg) {
         cfg.value = index;
      }

      cs->current = out;
      cs->restart_index = index;
      cs->restart_index_set = true;
   }
}

static float hk_depth_bias_factor(VkFormat format, bool exact,
                                  bool force_unorm);

/* Every refusal below is a real capability gap that silently drops the draw.
 * Report the exact check that refused, once per site, so a failing
 * application names a specific unimplemented G17P bind-tail field
 * instead of a single opaque message. */
static __attribute__((cold, noinline)) bool
hk_g17p_bind_tail_refused(unsigned line)
{
   static uint32_t reported[64];
   static unsigned nr_reported;
   for (unsigned i = 0; i < nr_reported; ++i) {
      if (reported[i] == line)
         return false;
   }
   if (nr_reported < ARRAY_SIZE(reported))
      reported[nr_reported++] = line;
   mesa_loge("G17P native bind tail refused at hk_cmd_draw.c:%u", line);
   return false;
}

static bool
hk_g17p_bind_tail_state_matches(
   const struct hk_cmd_buffer *cmd,
   const struct hk_g17p_prepared_bind_tail *prepared);

static bool
hk_g17p_build_native_bind_tail(struct hk_cmd_buffer *cmd,
                               struct hk_g17p_prepared_bind_tail *prepared)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct hk_rendering_state *render = &gfx->render;
   const unsigned viewport_count = MAX2(dyn->vp.viewport_count, 1);

   /* The record is command-buffer-owned and every field consumed by a valid
    * snapshot is overwritten below.  Invalidating it is sufficient on entry;
    * clearing the complete viewport arrays here made every rebuild pay for
    * unused viewport slots as well. */
   if (prepared)
      prepared->valid = false;

   if (!agx_native_graphics(&dev->dev) || !prepared)
      return hk_g17p_bind_tail_refused(__LINE__);
   if (!hk_g17p_viewport_state_supported(dyn->vp.viewport_count,
                                         dyn->vp.scissor_count))
      return hk_g17p_bind_tail_refused(__LINE__);

   struct hk_shader *fs = hk_bound_fs(cmd);
   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   if (!fs || !linked_fs)
      return hk_g17p_bind_tail_refused(__LINE__);
   /* The launch follows the linked program's own layout, if it has one. */
   if (linked_fs->g17p_layout)
      fs = linked_fs->g17p_layout;

   const uint64_t fragment_program = hk_upload_usc_words(cmd, fs, linked_fs);
   if (!fragment_program)
      return hk_g17p_bind_tail_refused(__LINE__);
   if ((fragment_program & UINT64_C(0x3f)) != 0 ||
       fragment_program >= (UINT64_C(1) << 48))
      return hk_g17p_bind_tail_refused(__LINE__);

   /* Every input below except the fragment stage program is identical to the
    * last successful build: only the root, descriptor or vertex-buffer
    * identity moved (a descriptor push or vertex-buffer bind), and those reach
    * the tail only through fragment_program above.  Repack the fragment
    * record into a copy of the last state block. */
   if (gfx->g17p_bind_tail_memo.valid &&
       gfx->g17p_bind_tail_memo.fs == fs &&
       gfx->g17p_bind_tail_memo.hw_vs == hk_bound_hw_vs(gfx) &&
       hk_g17p_bind_tail_state_matches(cmd, prepared)) {
      struct agx_ppp_hal200_fragment_fields memo_fs_fields =
         gfx->g17p_bind_tail_memo.fs_fields;
      memo_fs_fields.stage_program = fragment_program;
      struct agx_ppp_hal200_graphics_core memo_fs_record;
      if (!agx_ppp_hal200_fragment_core_pack(dev->dev.chip, &memo_fs_fields,
                                             &memo_fs_record))
         return hk_g17p_bind_tail_refused(__LINE__);

      /* The tail's record pointers all name that block: move them to the
       * copy. */
      const uint32_t memo_state_size = gfx->g17p_bind_tail_memo.state_size;
      struct agx_ptr memo_state =
         hk_pool_vdm_alloc(cmd, memo_state_size, 64);
      if (!memo_state.cpu ||
          !agx_g17p_draw_state_range_valid(memo_state.gpu, memo_state_size))
         return hk_g17p_bind_tail_refused(__LINE__);

      memcpy(memo_state.cpu, gfx->g17p_bind_tail_memo.state_cpu,
             memo_state_size);
      memcpy(memo_state.cpu, memo_fs_record.bytes, memo_fs_record.length);

      const uint32_t delta =
         (uint32_t)memo_state.gpu - gfx->g17p_bind_tail_memo.state_compact;
      for (unsigned i = 0; i < AGX_VDM_HAL200_BIND_COUNT; ++i) {
         uint8_t *offset = &prepared->tail.bytes[4 + 8 * i];
         uint32_t value;
         memcpy(&value, offset, sizeof(value));
         value += delta;
         memcpy(offset, &value, sizeof(value));
      }
      gfx->g17p_bind_tail_memo.state_cpu = memo_state.cpu;
      gfx->g17p_bind_tail_memo.state_compact = (uint32_t)memo_state.gpu;

      prepared->dynamic_generation = dyn->generation;
      prepared->descriptor_generation = gfx->descriptors.generation;
      prepared->vertex_buffer_generation = gfx->root_vertex_buffer_generation;
      prepared->root = gfx->root;
      prepared->fragment_shader = fs;
      prepared->fragment_dma_set_base = hk_g17p_dma_set_base(cmd, fs);
      prepared->valid = true;
      return true;
   }
   /* The memo describes prepared, which the build below overwrites. */
   gfx->g17p_bind_tail_memo.valid = false;

   if (!hk_g17p_smooth_varyings_supported(cmd, linked_fs)) {
      mesa_loge_once(
         "G17P varying layout refused: bindings=%u cf=%u reads_z=%u "
         "table=%" PRIx64 " topology=%u provoking=%u",
         linked_fs->b.cf.nr_bindings, linked_fs->b.cf.nr_cf,
         linked_fs->b.cf.reads_z, gfx->varyings, gfx->topology,
         gfx->provoking);
      return hk_g17p_bind_tail_refused(__LINE__);
   }
   if ((linked_fs->b.cf.nr_bindings != 0) != (gfx->varyings != 0))
      return hk_g17p_bind_tail_refused(__LINE__);

   /* This record describes the program actually launched by the Index List.
    * Ordinary draws use the API vertex shader, while GS/XFB draws use the
    * geometry raster-copy variant. Taking the API VS unconditionally makes
    * the HAL200 output contract disagree with the live VertexProgramVariant
    * as soon as interstage optimization changes their UVS footprints. */
   const struct hk_shader *vs = hk_bound_hw_vs(gfx);
   if (!vs)
      return hk_g17p_bind_tail_refused(__LINE__);

   uint32_t output_size;
   if (!agx_ppp_hal200_output_size_pack(vs->info.uvs.size, 0, &output_size))
      return hk_g17p_bind_tail_refused(__LINE__);

   uint32_t output_select;
   if (!agx_ppp_hal200_output_select_pack(
          dev->dev.chip, vs->info.uvs.osel.opaque[0],
          linked_fs->b.cf.nr_bindings != 0, linked_fs->b.cf.reads_z,
          &output_select))
      return hk_g17p_bind_tail_refused(__LINE__);

   struct agx_ppp_hal200_graphics_core fs_record;
   struct agx_ppp_hal200_fragment_fields fs_fields = {
      /* HAL200 finalize 0x3f04e4..0x3f0508 puts iteration count at bit 16.
       * The packer merges the complete VIT address, including its high bits. */
      .resource_counts = linked_fs->b.cf.nr_bindings << 16,
      .varying_table = gfx->varyings,
      .unknown_0c = 0,
      .stage_program = fragment_program,
      .stage_program_template = 0,
      .flags = 0,
   };
   if (fs->b.info.has_preamble &&
       (!fs->b.info.agx3_preamble_staging_known ||
        !agx_ppp_hal200_fragment_ti_resources(
          dev->dev.chip, fs->b.info.preamble_scratch_size,
          fs->b.info.agx3_preamble_staging_halfs * 2u, &fs_fields)))
      return hk_g17p_bind_tail_refused(__LINE__);

   struct agx_ppp_hal200_graphics_core output_record;
   struct agx_ppp_hal200_graphics_fields output_fields = {
      /* Native vertex update 0x481be8..0x481c2c writes zero at
       * context+0x1f90 and viewport_count-1 at +0x1f94; both are zero
       * for this single-viewport layout. VIT supplies interpolation.
       * FragmentProgramVariant 0x373eb4..0x373ee0 sets output-select bit16
       * for nonempty iteration metadata, merged by 0x4824fc..0x482518. */
      .words = {output_select,
                0, 0, output_size,
                AGX_PPP_HAL200_PRIMARY_FACE_HEADER, 0},
   };
   if (!agx_ppp_hal200_viewport_count_pack(
          dev->dev.chip, viewport_count, &output_fields.words[2]))
      return hk_g17p_bind_tail_refused(__LINE__);
   if (dev->dev.chip == AGX_CHIP_G17P && vs->info.stage == MESA_SHADER_MESH &&
       vs->info.uvs.mesh_primitive_page)
      output_fields.words[1] = DIV_ROUND_UP(vs->info.mesh.primitive_stride_B, 4);

   struct agx_ppp_hal200_graphics_core primary_record;
   struct agx_ppp_hal200_graphics_fields primary_fields = {
      .words = {0, UINT32_C(0x07200f00),
                UINT32_C(0x0e000000), UINT32_C(0x07200f00),
                UINT32_C(0x0e000000), 0},
   };

   /* Each draw already owns its complete PRIMARY_FACE record. Native depth
    * control only changes these two words; retain the allocation and staging
    * layout and do not add launches, cache flushes or device-global state. */
   const bool depth_test = render->depth_att.iview && dyn->ds.depth.test_enable;
   /* A shader with side effects under VK_COMPARE_OP_NEVER emulates the test
    * by exporting NaN depth (agx_nir_lower_frag_sidefx); the hardware test
    * must then pass everything, as FRAGMENT_FACE does, or the fragments
    * never run (rasterization.frag_side_effects.*.depth_never). */
   const VkCompareOp depth_compare =
      gfx->descriptors.root.draw.force_never_in_shader
         ? VK_COMPARE_OP_ALWAYS : dyn->ds.depth.compare_op;
   if (dyn->ds.depth.bounds_test.enable ||
       !agx_ppp_hal200_depth_face_pack(
          dev->dev.chip, depth_test, dyn->ds.depth.write_enable,
          depth_compare, primary_fields.words[1],
          &primary_fields.words[1]) ||
       !agx_ppp_hal200_depth_face_pack(
          dev->dev.chip, depth_test, dyn->ds.depth.write_enable,
          depth_compare, primary_fields.words[3],
          &primary_fields.words[3]))
      return hk_g17p_bind_tail_refused(__LINE__);

   const bool stencil_test = hk_stencil_test_enabled(cmd);
   const struct vk_stencil_test_face_state stencil_faces[2] = {
      dyn->ds.stencil.front, dyn->ds.stencil.back,
   };
   for (unsigned i = 0; i < 2; ++i) {
      const struct vk_stencil_test_face_state *face = &stencil_faces[i];
      const unsigned face_word = 1 + 2 * i;
      /* Native458734/458740 updates only the low reference byte. */
      primary_fields.words[face_word] =
         (primary_fields.words[face_word] & ~0xffu) | (face->reference & 0xffu);
      primary_fields.words[face_word + 1] = agx_ppp_hal200_stencil_pack(
         stencil_test, face->op.compare, face->op.fail, face->op.depth_fail,
         face->op.pass, face->compare_mask, face->write_mask);
   }

   struct agx_ppp_hal200_graphics_core secondary_record;
   struct agx_ppp_hal200_graphics_fields secondary_fields = {
      .words = {UINT32_C(0x00000200), UINT32_C(0x07e00000),
                UINT32_C(0x07e00000), UINT32_C(0x0001ffff)},
   };

   /* UNCHANGED permits early depth rejection. DEPTHF shaders must publish
    * their actual depth contract here, including the forced-ANY contract of
    * late side effects. This is independent of pass type and triangle merging. */
   const unsigned conservative_depth =
      agx_translate_depth_layout(fs->b.info.depth_layout);
   for (unsigned face = 1; face <= 2; ++face) {
      if (!agx_ppp_hal200_fragment_face_depth_pack(
             dev->dev.chip, conservative_depth, fs->b.info.early_fragment_tests,
             fs->info.fs.writes_memory, secondary_fields.words[face],
             &secondary_fields.words[face]))
         return hk_g17p_bind_tail_refused(__LINE__);
   }

   const unsigned line_width = agx_pack_line_width(dyn->rs.line.width);
   const unsigned polygon_mode = dyn->rs.polygon_mode;
   const unsigned object_type = gfx->object_type;
   if (!agx_ppp_hal200_fragment_face_raster_pack(
          dev->dev.chip, line_width, polygon_mode, primary_fields.words[1],
          &primary_fields.words[1]) ||
       !agx_ppp_hal200_fragment_face_raster_pack(
          dev->dev.chip, line_width, polygon_mode, primary_fields.words[3],
          &primary_fields.words[3]) ||
       !agx_ppp_hal200_fragment_face_object_pack(
          dev->dev.chip, object_type, secondary_fields.words[1],
          &secondary_fields.words[1]) ||
       !agx_ppp_hal200_fragment_face_object_pack(
          dev->dev.chip, object_type, secondary_fields.words[2],
          &secondary_fields.words[2]))
      return hk_g17p_bind_tail_refused(__LINE__);

   struct agx_ppp_hal200_graphics_core cull_record;
   struct agx_ppp_hal200_graphics_fields cull_fields = {
      .words = {UINT32_C(0x00000480)},
   };
   /* Software tessellation emits its own winding; fold the domain's
    * orientation into the front face as the common CULL packet does
    * (shader_object.misc.state.*.vert_tess_frag.cull.*). */
   bool front_face_ccw = dyn->rs.front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;
   if (gfx->shaders[MESA_SHADER_TESS_CTRL] &&
       !gfx->shaders[MESA_SHADER_GEOMETRY]) {
      front_face_ccw ^= gfx->tess.info.ccw;
      front_face_ccw ^= dyn->ts.domain_origin ==
                        VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
   }
   if (!agx_ppp_hal200_cull_mode_pack(
          dev->dev.chip, dyn->rs.cull_mode, front_face_ccw,
          cull_fields.words[0], &cull_fields.words[0]))
      return hk_g17p_bind_tail_refused(__LINE__);
   if (!agx_ppp_hal200_cull_provoking_pack(dev->dev.chip, gfx->provoking,
                                         cull_fields.words[0],
                                         &cull_fields.words[0]))
      return hk_g17p_bind_tail_refused(__LINE__);
   if (!agx_ppp_hal200_depth_clip_pack(
          dev->dev.chip,
          vk_rasterization_state_depth_clip_enable(&dyn->rs),
          cull_fields.words[0], &cull_fields.words[0]))
      return hk_g17p_bind_tail_refused(__LINE__);
   if (!agx_ppp_hal200_rasterizer_discard_pack(
          dev->dev.chip, hk_rast_discard(cmd), secondary_fields.words[0],
          cull_fields.words[0], &secondary_fields.words[0],
          &cull_fields.words[0]))
      return hk_g17p_bind_tail_refused(__LINE__);
   if (!agx_ppp_hal200_primitive_msaa_pack(
          dev->dev.chip,
          object_type == AGX_OBJECT_TYPE_LINE &&
             dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR,
          cull_fields.words[0], &cull_fields.words[0]))
      return hk_g17p_bind_tail_refused(__LINE__);

   struct agx_ppp_w_clamp_cull2_core w_cull2_record;
   const bool native_mesh_primitive_address_enabled =
      dev->dev.chip == AGX_CHIP_G17P &&
      vs->info.stage == MESA_SHADER_MESH &&
      vs->info.uvs.mesh_primitive_page &&
      vs->info.mesh.primitive_stride_B != 0 &&
      debug_get_option_mesh_native_primitive_address();
   const struct agx_ppp_w_clamp_cull2_fields w_cull2_fields = {
      .w_clamp_raw = dev->dev.chip == AGX_CHIP_G17P &&
                     vs->info.stage == MESA_SHADER_MESH &&
                     debug_get_option_mesh_native_m5_wclamp()
                        ? UINT32_C(0x00840002) : 0,
      /* The generated primitive-ID coefficient needs its rasterizer input
       * enabled as well as the matching VDM producer and output slot. */
      .cull2_raw = UINT32_C(0x80) |
                   (gfx->generate_primitive_id ? UINT32_C(0x1000) : 0) |
                   (native_mesh_primitive_address_enabled ? UINT32_C(0x2000) : 0),
   };

   /* Only the first viewport_count entries are written and read. */
   struct agx_ppp_viewport_payload_fields viewport_fields;
   viewport_fields.viewport_count = viewport_count;
   viewport_fields.viewport_control_raw = 0;
   unsigned minx[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS];
   unsigned miny[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS];
   unsigned maxx[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS];
   unsigned maxy[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS];
   for (unsigned i = 0; i < viewport_count; ++i) {
      const VkViewport *vp = &dyn->vp.viewports[i];
      minx[i] = MIN2(vp->x, vp->x + vp->width);
      miny[i] = MIN2(vp->y, vp->y + vp->height);
      maxx[i] = MAX2(vp->x, vp->x + vp->width);
      maxy[i] = MAX2(vp->y, vp->y + vp->height);
      minx[i] = MIN2(minx[i], render->cr.width);
      miny[i] = MIN2(miny[i], render->cr.height);
      maxx[i] = MIN2(maxx[i], render->cr.width);
      maxy[i] = MIN2(maxy[i], render->cr.height);
      if (i < dyn->vp.scissor_count) {
         const VkRect2D *s = &dyn->vp.scissors[i];
         minx[i] = MAX2(minx[i], s->offset.x);
         miny[i] = MAX2(miny[i], s->offset.y);
         maxx[i] = MIN2(maxx[i], s->offset.x + s->extent.width);
         maxy[i] = MIN2(maxy[i], s->offset.y + s->extent.height);
      }
   }

   const VkViewport *vp = &dyn->vp.viewports[0];
   const struct hk_image_view *color = render->color_att_count == 1
                                         ? render->color_att[0].iview : NULL;
   const VkRect2D api_scissor = dyn->vp.scissor_count
                                  ? dyn->vp.scissors[0] : (VkRect2D){0};
   const struct hk_g17p_full_frame_scissor full_frame = {
      .width = render->cr.width,
      .height = render->cr.height,
      .area_x = render->area.offset.x,
      .area_y = render->area.offset.y,
      .area_width = render->area.extent.width,
      .area_height = render->area.extent.height,
      .scissor_x = api_scissor.offset.x,
      .scissor_y = api_scissor.offset.y,
      .scissor_width = api_scissor.extent.width,
      .scissor_height = api_scissor.extent.height,
      .viewport_x = vp->x,
      .viewport_y = vp->y,
      .viewport_width = vp->width,
      .viewport_height = vp->height,
      .min_depth = vp->minDepth,
      .max_depth = vp->maxDepth,
      .viewport_count = dyn->vp.viewport_count,
      .scissor_count = dyn->vp.scissor_count,
      .single_color_layer_sample =
         color && !render->depth_att.iview && !render->stencil_att.iview &&
         render->layer_count == 1 && !render->view_mask &&
         render->tilebuffer.nr_samples == 1 &&
         u_minify(color->vk.image->extent.width, color->vk.base_mip_level) ==
            render->cr.width &&
         u_minify(color->vk.image->extent.height, color->vk.base_mip_level) ==
            render->cr.height,
      .default_depth =
         vk_rasterization_state_depth_clip_enable(&dyn->rs) &&
         !dyn->rs.depth_clamp_enable && !dyn->vp.depth_clip_negative_one_to_one &&
         !dyn->rs.depth_bias.enable && !hk_rast_discard(cmd),
   };
   const bool scissor_enable = dyn->vp.viewport_count > 0 &&
                              !hk_g17p_full_frame_scissor_can_elide(&full_frame);
   const bool depth_bias_enable =
      dyn->rs.depth_bias.enable &&
      gfx->object_type == AGX_OBJECT_TYPE_TRIANGLE;
   const bool exact = dyn->rs.depth_bias.exact;
   const bool force_unorm =
      dyn->rs.depth_bias.representation ==
      VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;
   const float depth_bias =
      depth_bias_enable
         ? dyn->rs.depth_bias.constant_factor /
              hk_depth_bias_factor(render->depth_att.vk_format, exact,
                                   force_unorm)
         : 0.0f;
   struct agx_ppp_hal200_raster_state
      raster_states[AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS];
   for (unsigned i = 0; i < viewport_count; ++i) {
      const VkViewport *current_vp = &dyn->vp.viewports[i];
      const float min_z = dyn->rs.depth_clamp_enable
                             ? MIN2(current_vp->minDepth, current_vp->maxDepth)
                             : 0.0f;
      const float max_z = dyn->rs.depth_clamp_enable
                             ? MAX2(current_vp->minDepth, current_vp->maxDepth)
                             : 1.0f;
      const struct agx_ppp_hal200_raster_fields raster_fields = {
         .scissor_enable = scissor_enable,
         .depth_bias_enable = depth_bias_enable,
         .max_x = MIN2(maxx[i], UINT16_MAX),
         .min_x = MIN2(minx[i], UINT16_MAX),
         .max_y = MIN2(maxy[i], UINT16_MAX),
         .min_y = MIN2(miny[i], UINT16_MAX),
         .min_z_raw = fui(min_z),
         .max_z_raw = fui(max_z),
         .depth_bias_raw = fui(depth_bias),
         .slope_scale_raw =
            depth_bias_enable ? fui(dyn->rs.depth_bias.slope_factor) : 0,
         .clamp_raw = depth_bias_enable ? fui(dyn->rs.depth_bias.clamp) : 0,
      };
      if (!agx_ppp_hal200_raster_state_build(
             dev->dev.chip, &raster_fields, &raster_states[i]))
         return hk_g17p_bind_tail_refused(__LINE__);
   }
   const struct agx_ppp_hal200_raster_state *raster_state = &raster_states[0];

   /* Native update 0x46c6ac..0x46c6dc stores pass type at context+0x1f78.
    * The 0x46ff78 path emits it in SECONDARY header 0x02000048's first
    * payload word, masked by 0xe6600000 and ORed with 0x200. It is not the
    * older-generation FRAGMENT_CONTROL location in our PRIMARY record.
    */
   const uint32_t linked_fragment_control =
      linked_fs->b.fragment_control.opaque[0];
   const uint32_t pass_type = linked_fragment_control >> 29;
   if (pass_type != AGX_PASS_TYPE_OPAQUE &&
       pass_type != AGX_PASS_TYPE_TRANSLUCENT &&
       pass_type != AGX_PASS_TYPE_PUNCH_THROUGH &&
       pass_type != AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH)
      return hk_g17p_bind_tail_refused(__LINE__);
   /* Native458934..45896c retains both stencil-control bits in the
    * same per-draw record as depth. No extra state allocation or launch. */
   uint32_t fragment_control = raster_state->fragment_control |
      (stencil_test ? UINT32_C(0x000c0000) : 0);
   const struct agx_ppp_hal200_fragment_control_fields control_fields = {
      .linked_control = linked_fragment_control,

      /* Apple G17P shader metadata leaves bit26 clear for texture gradients,
       * storage writes, framebuffer fetch, sample-rate execution, and shadows.
       * Multisampled depth/depth-stencil variants set it.
       *
       * A shader that takes implicit quad derivatives still needs it on this
       * stack. Triangle merging shades a 2x2 quad straddling the shared edge
       * of two primitives once, with its lanes interpolated from two different
       * plane equations, so an implicit-LOD sample on that quad derives its
       * mip level across the seam.
       *
       * Measured on G17P: without this term all 30
       * dEQP-VK.texture.mipmap.{2d,3d,cubemap}.projected.* cases fail and
       * every invalid pixel lies exactly on the shared diagonal of the test's
       * two triangles (31/31 in the decoded error mask of
       * 2d.projected.nearest_linear_clamp); with it, all 30 pass and the 18
       * basic/affine siblings are unchanged. Those siblings pass either way
       * because both of their triangles carry the same affine texcoord map, so
       * the cross-seam derivative equals the correct one.
       *
       * AGX2 never had the defect: agx_compile.c derives disable_tri_merging
       * from needs_coarse_quad_helper_invocations, which nir_gather_info sets
       * for every implicit-derivative texture op; agx_linker.c ORs it across
       * the shader parts into the linked FRAGMENT_CONTROL bit 26; and
       * hk_flush_ppp_state() merges that word into the secondary control.
       * G17P never reaches it -- the draw path calls hk_flush_ppp_state() only
       * `if (!g17p_native_bind_tail_emitted)` -- so the whole control comes
       * from this record instead, and agx_ppp_hal200_fragment_controls_pack()
       * deliberately masks bit 26 out of the original and rebuilds it from the
       * three fields below. The helper-invocation term was dropped in that
       * re-derivation and nothing else carried it.
       *
       * Whether Apple's own driver disables merging here or keeps quads
       * coherent some other way is NOT established; only that this stack needs
       * the bit. Kept narrower than the linker's bit 26, which also fires on
       * memory writes and wide subgroup intrinsics -- neither needs quad
       * coherence, and Apple's metadata leaves bit 26 clear for both. */
      .shader_tri_merge_disable =
         fs->b.info.needs_quad_derivatives ||
         (render->tilebuffer.nr_samples > 1 &&
          fs->info.fs.epilog_key.write_z) ||
         /* The linker's own merge-disable bit. */
         (!agx_chip_tri_merge_across_side_effects(dev->dev.chip) &&
          (linked_fragment_control & (UINT32_C(1) << 26))),

      /* Native forces bit26 for type2 and conditionally for type3 from the
       * draw-buffer state. Keep the already-qualified conservative type3
       * policy until that draw-buffer field is decoded. */
      .coverage_tri_merge_disable =
         pass_type == AGX_PASS_TYPE_PUNCH_THROUGH ||
         pass_type == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH,

      /* encodeISPPrimitiveType selects the runtime override for points,
       * lines, and non-filled triangles, then applies it to both controls. */
      .primitive_tri_merge_disable =
         object_type != AGX_OBJECT_TYPE_TRIANGLE ||
         polygon_mode != VK_POLYGON_MODE_FILL,
   };
   if (!agx_ppp_hal200_fragment_controls_pack(
          dev->dev.chip, &control_fields, fragment_control,
          secondary_fields.words[0], &fragment_control,
          &secondary_fields.words[0]))
      return hk_g17p_bind_tail_refused(__LINE__);

   uint32_t occlusion_query;
   if (!agx_ppp_hal200_occlusion_pack(
          dev->dev.chip, gfx->occlusion.mode, gfx->occlusion.index,
          fragment_control, &fragment_control, &occlusion_query))
      return hk_g17p_bind_tail_refused(__LINE__);
   output_fields.words[5] = fragment_control;
   primary_fields.words[0] = fragment_control;
   primary_fields.words[5] = occlusion_query;
   if (!agx_ppp_hal200_fragment_core_pack(dev->dev.chip, &fs_fields,
                                          &fs_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->dev.chip, AGX_PPP_HAL200_GRAPHICS_OUTPUT, &output_fields,
          &output_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->dev.chip, AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE,
          &primary_fields, &primary_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->dev.chip, AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE,
          &secondary_fields, &secondary_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->dev.chip, AGX_PPP_HAL200_GRAPHICS_CULL, &cull_fields,
          &cull_record) ||
       !agx_ppp_w_clamp_cull2_core_pack(dev->dev.chip, &w_cull2_fields,
                                         &w_cull2_record))
      return hk_g17p_bind_tail_refused(__LINE__);

   for (unsigned i = 0; i < viewport_count; ++i) {
      static_assert(sizeof(prepared->scissors[i]) ==
                    sizeof(raster_states[i].scissor));
      static_assert(sizeof(prepared->depth_bias[i]) ==
                    sizeof(raster_states[i].depth_bias));
      memcpy(prepared->scissors[i], raster_states[i].scissor,
             sizeof(prepared->scissors[i]));
      memcpy(prepared->depth_bias[i], raster_states[i].depth_bias,
             sizeof(prepared->depth_bias[i]));
   }

   /* Preserve the physical +0x00..+0x73 layout from Metal.  The complete
    * 28-byte FS record occupies +0x00..+0x1b; in particular, +0x14 is the
    * low stage-program word and must not be overlaid by the old anonymous
    * metadata guess.  The output window at +0x1c intentionally overlaps the
    * primary header/control at +0x30/+0x34.
    */
   static_assert(AGX_PPP_HAL200_FS_LENGTH == 0x1c,
                 "HAL200 FS must end where the output record begins");
   uint8_t bind_group[0x80] = {0};
   memcpy(&bind_group[0x00], fs_record.bytes, fs_record.length);
   memcpy(&bind_group[0x1c], output_record.bytes, output_record.length);
   memcpy(&bind_group[0x30], primary_record.bytes, primary_record.length);
   memcpy(&bind_group[0x4c], secondary_record.bytes, secondary_record.length);
   memcpy(&bind_group[0x60], w_cull2_record.bytes, w_cull2_record.length);
   memcpy(&bind_group[0x6c], cull_record.bytes, cull_record.length);

   for (unsigned i = 0; i < viewport_count; ++i) {
      uint8_t region_clip[AGX_REGION_CLIP_LENGTH];
      agx_pack(region_clip, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx[i] / 32;
         cfg.min_y = miny[i] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx[i], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy[i], 1), 32);
      }
      memcpy(&viewport_fields.region_clip_raw[i], region_clip,
             sizeof(region_clip));
      const VkViewport *current_vp = &dyn->vp.viewports[i];
      viewport_fields.viewports[i] =
         (struct agx_ppp_viewport_transform_raw){
            .translate_x = fui(current_vp->x + 0.5f * current_vp->width),
            .scale_x = fui(current_vp->width * 0.5f),
            .translate_y = fui(current_vp->y + 0.5f * current_vp->height),
            .scale_y = fui(current_vp->height * 0.5f),
            .translate_z = fui(current_vp->minDepth),
            .scale_z = fui(current_vp->maxDepth - current_vp->minDepth),
         };
   }

   struct agx_ppp_viewport_payload_core viewport_record;
   if (!agx_ppp_viewport_payload_core_pack(dev->dev.chip, &viewport_fields,
                                            &viewport_record))
      return hk_g17p_bind_tail_refused(__LINE__);

   /* A VDM fetch happens after recording. Never point earlier draws at a
    * device-global record that another draw/CB can overwrite. This upload
    * and its architectural low alias live until command-buffer reset. */
   const size_t state_size = agx_g17p_bind_state_size(viewport_record.length);
   struct agx_ptr state = hk_pool_vdm_alloc(cmd, state_size, 64);
   struct hk_g17p_bind_state_upload uploaded;
   if (!state_size || !state.cpu ||
       !hk_g17p_upload_bind_state(state.cpu, state_size, state.gpu,
                                  bind_group, viewport_record.bytes,
                                  viewport_record.length,
                                  gfx->linked_varyings.counts_32.opaque[0],
                                  gfx->linked_varyings.counts_16.opaque[0],
                                  &uploaded))
      return hk_g17p_bind_tail_refused(__LINE__);

   prepared->tail = uploaded.tail;
   gfx->g17p_bind_tail_memo.state_cpu = state.cpu;
   gfx->g17p_bind_tail_memo.state_compact = (uint32_t)state.gpu;
   gfx->g17p_bind_tail_memo.state_size = state_size;
   prepared->dynamic_generation = dyn->generation;
   prepared->descriptor_generation = gfx->descriptors.generation;
   prepared->vertex_buffer_generation = gfx->root_vertex_buffer_generation;
   prepared->root = gfx->root;
   prepared->fragment_shader = fs;
   prepared->fragment_dma_set_base = hk_g17p_dma_set_base(cmd, fs);
   prepared->render_color0 = render->color_att_count == 1
      ? render->color_att[0].iview : NULL;
   prepared->render_depth = render->depth_att.iview;
   prepared->render_stencil = render->stencil_att.iview;
   prepared->render_area = render->area;
   prepared->render_depth_format = render->depth_att.vk_format;
   prepared->render_width = render->cr.width;
   prepared->render_height = render->cr.height;
   prepared->render_layer_count = render->layer_count;
   prepared->render_view_mask = render->view_mask;
   prepared->render_color_count = render->color_att_count;
   prepared->render_samples = render->tilebuffer.nr_samples;
   memcpy(prepared->shaders, gfx->shaders, sizeof(prepared->shaders));
   prepared->linked_vs = gfx->linked[MESA_SHADER_VERTEX];
   prepared->linked_fs = linked_fs;
   prepared->varyings = gfx->varyings;
   prepared->varying_counts_32 = gfx->linked_varyings.counts_32.opaque[0];
   prepared->varying_counts_16 = gfx->linked_varyings.counts_16.opaque[0];
   prepared->occlusion_index = gfx->occlusion.index;
   prepared->topology = gfx->topology;
   prepared->object_type = gfx->object_type;
   prepared->provoking = gfx->provoking;
   prepared->occlusion_mode = gfx->occlusion.mode;
   prepared->viewport_count = viewport_count;
   prepared->generate_primitive_id = gfx->generate_primitive_id;
   prepared->force_never_in_shader =
      gfx->descriptors.root.draw.force_never_in_shader;
   prepared->in_meta = cmd->in_meta;
   prepared->valid = true;

   gfx->g17p_bind_tail_memo.fs_fields = fs_fields;
   gfx->g17p_bind_tail_memo.fs = fs;
   gfx->g17p_bind_tail_memo.hw_vs = vs;
   gfx->g17p_bind_tail_memo.valid = true;

   return true;
}

/* Every builder input except the root / descriptor / vertex-buffer
 * identity, which reach the tail only through the fragment stage program. */
static bool
hk_g17p_bind_tail_state_matches(
   const struct hk_cmd_buffer *cmd,
   const struct hk_g17p_prepared_bind_tail *prepared)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;
   const struct hk_rendering_state *render = &gfx->render;

   return prepared->dynamic_generation == dyn->generation &&
          prepared->render_color0 ==
             (render->color_att_count == 1
                 ? render->color_att[0].iview : NULL) &&
          prepared->render_depth == render->depth_att.iview &&
          prepared->render_stencil == render->stencil_att.iview &&
          memcmp(&prepared->render_area, &render->area,
                 sizeof(prepared->render_area)) == 0 &&
          prepared->render_depth_format == render->depth_att.vk_format &&
          prepared->render_width == render->cr.width &&
          prepared->render_height == render->cr.height &&
          prepared->render_layer_count == render->layer_count &&
          prepared->render_view_mask == render->view_mask &&
          prepared->render_color_count == render->color_att_count &&
          prepared->render_samples == render->tilebuffer.nr_samples &&
          memcmp(prepared->shaders, gfx->shaders,
                 sizeof(prepared->shaders)) == 0 &&
          prepared->linked_vs == gfx->linked[MESA_SHADER_VERTEX] &&
          prepared->linked_fs == gfx->linked[MESA_SHADER_FRAGMENT] &&
          prepared->varyings == gfx->varyings &&
          prepared->varying_counts_32 ==
             gfx->linked_varyings.counts_32.opaque[0] &&
          prepared->varying_counts_16 ==
             gfx->linked_varyings.counts_16.opaque[0] &&
          prepared->occlusion_index == gfx->occlusion.index &&
          prepared->topology == gfx->topology &&
          prepared->object_type == gfx->object_type &&
          prepared->provoking == gfx->provoking &&
          prepared->occlusion_mode == gfx->occlusion.mode &&
          prepared->viewport_count == MAX2(dyn->vp.viewport_count, 1) &&
          prepared->generate_primitive_id == gfx->generate_primitive_id &&
          prepared->force_never_in_shader ==
             gfx->descriptors.root.draw.force_never_in_shader &&
          prepared->in_meta == cmd->in_meta;
}

static bool
hk_g17p_prepared_bind_tail_matches(
   const struct hk_cmd_buffer *cmd,
   const struct hk_g17p_prepared_bind_tail *prepared)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;

   /* The tail launches a fragment program whose UNIFORM_DMA span, when it is
    * set-sourced, names the set address the root held when the program was
    * built (hk_g17p_dma_set_base()). A push descriptor set gets its address
    * when it is flushed, which moves that address without a new descriptor
    * generation or root identity. */
   return prepared->valid &&
          prepared->descriptor_generation == gfx->descriptors.generation &&
          prepared->vertex_buffer_generation ==
             gfx->vertex_buffer_generation &&
          prepared->root == gfx->root &&
          prepared->fragment_dma_set_base ==
             hk_g17p_dma_set_base(cmd, prepared->fragment_shader) &&
          hk_g17p_bind_tail_state_matches(cmd, prepared);
}

static bool
hk_g17p_publish_native_bind_tail(
   struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out,
   const struct hk_g17p_prepared_bind_tail *prepared)
{
   if (!prepared->valid || !out || !*out)
      return false;

   const unsigned scissor_bytes =
      prepared->viewport_count * AGX_SCISSOR_LENGTH;
   const unsigned depth_bias_bytes =
      prepared->viewport_count * AGX_DEPTH_BIAS_LENGTH;
   if (!util_dynarray_ensure_cap(&cs->scissor,
                                 cs->scissor.size + scissor_bytes) ||
       !util_dynarray_ensure_cap(&cs->depth_bias,
                                 cs->depth_bias.size + depth_bias_bytes)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return false;
   }

   /* The draw-state selector names the LAST viewport_count
    * scissor and depth-bias records of this stream. When those already equal
    * this tail's records (a draw that republishes only because its vertex
    * shader reads per-draw base parameters), selecting them again is
    * identical; appending would grow the arrays by one set per draw until
    * the 16-bit selector index overflowed (>65k such draws in one stream made
    * agx_g17p_upload_draw_indices() refuse the draw and the command buffer
    * fail with VK_ERROR_FEATURE_NOT_PRESENT). */
   const bool same_tail =
      cs->scissor.size >= scissor_bytes &&
      cs->depth_bias.size >= depth_bias_bytes &&
      !memcmp((const uint8_t *)cs->scissor.data + cs->scissor.size -
                 scissor_bytes,
              prepared->scissors, scissor_bytes) &&
      !memcmp((const uint8_t *)cs->depth_bias.data + cs->depth_bias.size -
                 depth_bias_bytes,
              prepared->depth_bias, depth_bias_bytes);
   if (!same_tail) {
      uint8_t *scissors = util_dynarray_grow_bytes(
         &cs->scissor, prepared->viewport_count, AGX_SCISSOR_LENGTH);
      uint8_t *depth_bias = util_dynarray_grow_bytes(
         &cs->depth_bias, prepared->viewport_count, AGX_DEPTH_BIAS_LENGTH);
      assert(scissors && depth_bias);

      memcpy(scissors, prepared->scissors, scissor_bytes);
      memcpy(depth_bias, prepared->depth_bias, depth_bias_bytes);
   }
   memcpy(*out, prepared->tail.bytes, prepared->tail.length);
   *out += prepared->tail.length;
   return true;
}

/* known_match: the caller evaluated hk_g17p_prepared_bind_tail_matches()
 * for this draw and nothing it reads has changed since. */
static __attribute__((hot)) bool
hk_g17p_emit_native_bind_tail(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                              uint8_t **out, bool known_match)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_g17p_prepared_bind_tail *prepared =
      &gfx->g17p_bind_tail;
   if ((known_match && prepared->valid) ||
       hk_g17p_prepared_bind_tail_matches(cmd, prepared)) {
      return hk_g17p_publish_native_bind_tail(cmd, cs, out, prepared);
   }

   return hk_g17p_build_native_bind_tail(cmd, prepared) &&
          hk_g17p_publish_native_bind_tail(cmd, cs, out, prepared);
}

/* The hardware provoking-vertex selector for a topology. A point has one
 * vertex, so its first and last vertex are the same; use the selector the
 * native point pipelines carry (0) whatever the API convention is. GL's
 * last-vertex convention through Zink otherwise selected 2 for points, which
 * hk_g17p_smooth_varyings_supported() rightly does not admit for flat
 * inputs, and dEQP's fragment-shader executor draws one point per value. */
static unsigned
hk_provoking_selector(const struct vk_dynamic_graphics_state *dyn,
                      unsigned topology)
{
   if (topology == AGX_PRIMITIVE_POINTS)
      return 0;
   if (dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT)
      return 2;
   if (topology == AGX_PRIMITIVE_TRIANGLE_FAN)
      return 1;
   return 0;
}

/*
 * Return the given sample positions, packed into a 64-bit word with fixed
 * point nibbles for each x/y component of the (at most 8) samples. This is
 * suitable for programming the PPP_MULTISAMPLECTL control register.
 */
static uint64_t
hk_pack_ppp_multisamplectrl(const struct vk_sample_locations_state *sl)
{
   uint64_t ctrl = 0;

   for (int32_t i = sl->per_pixel - 1; i >= 0; i--) {
      VkSampleLocationEXT loc = sl->locations[i];

      uint32_t x = CLAMP(loc.x, 0.0f, 0.9375f) * 16.0;
      uint32_t y = CLAMP(loc.y, 0.0f, 0.9375f) * 16.0;

      assert(x <= 15);
      assert(y <= 15);

      /* Push bytes in reverse order so we can use constant shifts. */
      ctrl = (ctrl << 8) | (y << 4) | x;
   }

   return ctrl;
}

/*
 * Return the standard sample positions, prepacked as above for efficiency.
 */
uint64_t
hk_default_sample_positions(unsigned nr_samples)
{
   switch (nr_samples) {
   case 0:
   case 1:
      return 0x88;
   case 2:
      return 0x44cc;
   case 4:
      return 0xeaa26e26;
   case 8:
      return UINT64_C(0x1ffb71d3359db759);
   default:
      UNREACHABLE("Invalid sample count");
   }
}

static void
hk_flush_ppp_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t **out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *fs = hk_bound_fs(cmd);

   if (agx_chip_g17_m5(dev->dev.chip)) {
      struct hk_linked_shader *linked = gfx->linked[MESA_SHADER_FRAGMENT];
      /* With no fragment shader, attachments, or occlusion query, only
       * vertex processing is observable. Keep the vertex program live and
       * select the inert fragment state, including for transform feedback.
       */
      const bool vertex_only = gfx->shaders[MESA_SHADER_FRAGMENT] == dev->null_fs &&
         !render->tilebuffer.sample_size_B && !render->depth_att.iview &&
         !render->stencil_att.iview && !gfx->occlusion.mode;
      const bool discard = hk_rast_discard(cmd) || vertex_only;
      const VkViewport *vp = &dyn->vp.viewports[0];
      const VkRect2D *scissor = &dyn->vp.scissors[0];
      const struct agx_unlinked_uvs_layout *uvs = hw_vs ? &hw_vs->info.uvs : NULL;
      const bool psiz = uvs && uvs->group_offs[UVS_LAYER_VIEWPORT] ==
                                 uvs->group_offs[UVS_PSIZ] + 1;
      const unsigned clip_words = uvs && uvs->size >= uvs->group_offs[UVS_CLIP_DIST]
         ? uvs->size - uvs->group_offs[UVS_CLIP_DIST] : 0;
      const unsigned components = uvs ? uvs->user_size : 0;
      const bool layer_word = uvs && uvs->group_offs[UVS_CLIP_DIST] ==
                                    uvs->group_offs[UVS_LAYER_VIEWPORT] + 1;

      const bool vertex_layout = uvs && components <= 124 &&
         uvs->size <= 128 && clip_words <= 8 &&
         uvs->group_offs[UVS_POSITION] == 0 &&
         uvs->group_offs[UVS_VARYINGS] == 4 &&
         uvs->group_offs[UVS_PSIZ] == 4 + components &&
         uvs->group_offs[UVS_LAYER_VIEWPORT] == 4 + components + psiz &&
         uvs->group_offs[UVS_CLIP_DIST] == 4 + components + psiz + layer_word &&
         uvs->size == 4 + components + psiz + layer_word + clip_words &&
         uvs->osel.opaque[0] == ((psiz ? 0x40000 : 0) |
                               (layer_word ? 0x180000 : 0) |
                               ((UINT32_C(1) << clip_words) - 1));
      const struct agx_varyings_fs *cf = linked ? &linked->b.cf : NULL;
      const struct agx_cf_binding *color =
         cf && cf->nr_bindings >= 1 && cf->nr_bindings <= 3
            ? &cf->bindings[cf->nr_bindings - 1] : NULL;
      bool color_varying = vertex_layout && components <= 8 && components && color &&
         color->slot != VARYING_SLOT_POS && gfx->varyings &&
         hk_g17p_smooth_varyings_supported(cmd, linked) &&
         gfx->linked_varyings.counts_16.opaque[0] == 0 &&
         gfx->linked_varyings.counts_32.opaque[0] ==
            (components << (!color->smooth ? 8 : color->perspective ? 0 : 16));
      unsigned coefficient_count = 0;
      if (color_varying) {
         for (unsigned i = 0; i < cf->nr_bindings; ++i) {
            const struct agx_cf_binding *b = &cf->bindings[i];
            coefficient_count += b->count;
            if (b->slot == VARYING_SLOT_POS) {
               color_varying &= b->count == 1 && b->smooth && !b->perspective &&
                  ((b->offset == 3 && i == 0) ||
                   (b->offset == 2 && cf->reads_z));
            } else {
               color_varying &= b->slot < ARRAY_SIZE(uvs->components) &&
                  b->count && b->offset + b->count <= uvs->components[b->slot] &&
                  b->smooth == color->smooth && b->perspective == color->perspective;
            }
         }
         color_varying &= coefficient_count == cf->nr_cf &&
            (!color->perspective ||
             (cf->nr_bindings >= 2 &&
              cf->bindings[0].slot == VARYING_SLOT_POS &&
              cf->bindings[0].offset == 3));
      }
      /* Native HAL300 uses the same 4-byte VIT rows and interpolation-count
       * groups as Neo. Preserve the checked shared linker rather than infer
       * every binding's interpolation from the final color input. */
      bool general_varyings = vertex_layout && cf && cf->nr_bindings &&
         gfx->varyings && hk_g17p_smooth_varyings_supported(cmd, linked) &&
         gfx->linked_varyings.counts_16.opaque[0] == 0;
      if (general_varyings) {
         for (unsigned i = 0; i < cf->nr_bindings; ++i) {
            const struct agx_cf_binding *b = &cf->bindings[i];
            /* Native HAL300 VIT source2 supplies PointCoord without a UVS
             * word. Source3 supplies PrimitiveID through the shared VDM
             * generator and the PPP cull2 enable below. */
            const bool user =
               (b->slot >= VARYING_SLOT_VAR0 && b->slot <= VARYING_SLOT_VAR31) ||
               b->slot == VARYING_SLOT_LAYER || b->slot == VARYING_SLOT_VIEWPORT ||
               b->slot == VARYING_SLOT_CLIP_DIST0 ||
               b->slot == VARYING_SLOT_CLIP_DIST1 ||
               b->slot == VARYING_SLOT_CULL_PRIMITIVE ||
               b->slot == VARYING_SLOT_CULL_PRIMITIVE + 1;
            general_varyings &= b->slot == VARYING_SLOT_POS || user ||
               b->slot == VARYING_SLOT_PNTC ||
               b->slot == VARYING_SLOT_PRIMITIVE_ID;
         }
      }
      const bool no_varyings = vertex_layout && cf && !cf->nr_bindings &&
         !cf->nr_cf && !cf->reads_z && !gfx->varyings;
      const bool reads_rhw = cf && cf->nr_bindings &&
         cf->bindings[0].slot == VARYING_SLOT_POS &&
         cf->bindings[0].offset == 3;
      bool system_varyings = vertex_layout && cf && cf->nr_bindings &&
         gfx->varyings &&
         gfx->linked_varyings.counts_32.opaque[0] == 0 &&
         gfx->linked_varyings.counts_16.opaque[0] == 0;
      if (system_varyings) {
         /* The shared linker emits optional 1/W, then optional Z. These
          * system coefficients consume no user varying word; retain the
          * same native bindings when the fragment reads only FragCoord. */
         unsigned position_mask = 0;
         for (unsigned i = 0; i < cf->nr_bindings; ++i) {
            const struct agx_cf_binding *b = &cf->bindings[i];
            if (b->slot != VARYING_SLOT_POS || b->count != 1 ||
                !b->smooth || b->perspective || b->cf_base != i ||
                (b->offset != 2 && b->offset != 3) ||
                (b->offset == 3 && i != 0)) {
               system_varyings = false;
               break;
            }
            const unsigned bit = 1u << b->offset;
            if (position_mask & bit) {
               system_varyings = false;
               break;
            }
            position_mask |= bit;
         }
         system_varyings &= cf->nr_cf == cf->nr_bindings &&
            ((position_mask & (1u << 2)) != 0) == cf->reads_z;
      }
      const VkFormat depth_plane_format =
         hk_depth_stencil_plane_format(render->depth_att.vk_format, 0);
      const VkFormat stencil_plane_format = hk_depth_stencil_plane_format(
         render->stencil_att.vk_format,
         hk_split_depth_stencil(render->stencil_att.vk_format));
      const bool depth_format = !render->depth_att.iview ||
         depth_plane_format == VK_FORMAT_D32_SFLOAT ||
         depth_plane_format == VK_FORMAT_D16_UNORM;
      const bool stencil_format = !render->stencil_att.iview ||
         stencil_plane_format == VK_FORMAT_S8_UINT;
      const bool depth_active = render->depth_att.iview && dyn->ds.depth.test_enable;
      /* NEVER with late side effects is emulated by exported NaN depth. */
      const VkCompareOp depth_compare =
         gfx->descriptors.root.draw.force_never_in_shader
            ? VK_COMPARE_OP_ALWAYS : dyn->ds.depth.compare_op;
      const bool stencil_active = hk_stencil_test_enabled(cmd);
      struct vk_stencil_test_face_state front_stencil_state = dyn->ds.stencil.front;
      struct vk_stencil_test_face_state back_stencil_state = dyn->ds.stencil.back;
      // S8 compares and writes only the low eight API mask bits.
      front_stencil_state.compare_mask &= 0xffu;
      front_stencil_state.write_mask &= 0xffu;
      back_stencil_state.compare_mask &= 0xffu;
      back_stencil_state.write_mask &= 0xffu;
      const struct agx_fragment_stencil_packed front_stencil =
         hk_pack_stencil_face(front_stencil_state, stencil_active);
      const struct agx_fragment_stencil_packed back_stencil =
         hk_pack_stencil_face(back_stencil_state, stencil_active);
      const unsigned fragment_pass = linked ? linked->b.fragment_control.opaque[0] >> 29 : 0;
      const bool has_zs_attachment =
         render->depth_att.iview || render->stencil_att.iview;
      /* v70 depth928 and depth/stencil945 use the ordinary four-sample
       * 32x16 ZLS profile. The native ZLS struct/stride producer is sample
       * independent; kext selects both four/eight-sample tile geometry.
       * Native G17G DEPTHF writes select one covered sample per word,
       * preserving SR39's API-mask byte. Depth/stencil resolves run through
       * hk_meta_resolve_rendering after the multisample source is stored;
       * their presence does not change this draw's ZLS profile. */
      const bool plain_ms_zs =
         (render->tilebuffer.nr_samples == 2 ||
          render->tilebuffer.nr_samples == 4 ||
          render->tilebuffer.nr_samples == 8) && fs &&
         !gfx->descriptors.root.draw.force_never_in_shader;

      const struct vk_properties *limits = &dev->vk.physical->properties;
      const float max_viewport_height = limits->maxViewportDimensions[1];
      /* HAL300 publishes full float32 viewport transforms. Validate against
       * the device's API domain rather than the render target dimensions. */
      bool bounded_viewport = dyn->vp.viewport_count > 0 &&
         dyn->vp.viewport_count <= limits->maxViewports &&
         dyn->vp.scissor_count == dyn->vp.viewport_count;
      bool bounded_depth_range = true;
      for (unsigned i = 0; bounded_viewport && i < dyn->vp.viewport_count; ++i) {
         const VkViewport *v = &dyn->vp.viewports[i];
         bounded_viewport &=
            v->width > 0 && v->width <= limits->maxViewportDimensions[0] &&
            /* Vulkan 1.1 permits zero height. Keep the degenerate transform:
             * resize-time Zink draws still execute pre-raster shader work. */
            v->height >= -max_viewport_height &&
            v->height <= max_viewport_height &&
            v->x >= limits->viewportBoundsRange[0] &&
            v->x <= limits->viewportBoundsRange[1] &&
            v->x + v->width >= limits->viewportBoundsRange[0] &&
            v->x + v->width <= limits->viewportBoundsRange[1] &&
            v->y >= limits->viewportBoundsRange[0] &&
            v->y <= limits->viewportBoundsRange[1] &&
            v->y + v->height >= limits->viewportBoundsRange[0] &&
            v->y + v->height <= limits->viewportBoundsRange[1];
         bounded_depth_range &= v->minDepth >= 0 && v->minDepth <= 1 &&
                                v->maxDepth >= 0 && v->maxDepth <= 1;
      }

      if (!hw_vs || !fs || !linked ||
          (!no_varyings && !system_varyings && !color_varying &&
           !general_varyings) ||
          (gfx->object_type != AGX_OBJECT_TYPE_TRIANGLE &&
           gfx->object_type != AGX_OBJECT_TYPE_LINE &&
           gfx->object_type != AGX_OBJECT_TYPE_POINT_SPRITE_UV01) ||
          (render->color_att_count > 1 && !hk_m5_mrt_layout_supported(render)) ||
          !depth_format || !stencil_format ||
          (!discard && render->tilebuffer.nr_samples != 1 &&
           !((render->tilebuffer.nr_samples == 2 ||
              render->tilebuffer.nr_samples == 4 ||
              render->tilebuffer.nr_samples == 8) &&
             (render->tilebuffer.tile_size == 32 * 32 ||
              render->tilebuffer.tile_size == 32 * 16 ||
              render->tilebuffer.tile_size == 16 * 16) &&
             render->tilebuffer.nr_samples * render->tilebuffer.tile_size <=
                render->tilebuffer.max_samples_per_tile &&
             (unsigned)render->tilebuffer.sample_size_B *
                render->tilebuffer.nr_samples * render->tilebuffer.tile_size <=
                render->tilebuffer.max_bytes_per_tile &&
             (!has_zs_attachment || plain_ms_zs))) ||
          fragment_pass > AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH ||
          (!discard && (dyn->vp.depth_clip_negative_one_to_one ||
          dyn->ds.depth.bounds_test.enable ||
          !bounded_viewport || !bounded_depth_range))) {
         {
            fprintf(stderr, "M5 PPP profile: uvs=%u osel=%#x provoking=%u object=%u "
                    "depth=%u stencil=%u varyings=%#llx cf=%u/%u/%u "
                    "viewport=%g,%g,%g,%g,%g,%g negative_one=%u "
                    "scissor=%d,%d,%u,%u area=%d,%d,%u,%u\n",
                    hw_vs ? hw_vs->info.uvs.size : 0,
                    hw_vs ? hw_vs->info.uvs.osel.opaque[0] : 0,
                    gfx->provoking, gfx->object_type,
                    render->depth_att.iview != NULL, render->stencil_att.iview != NULL,
                    (unsigned long long)gfx->varyings,
                    linked ? linked->b.cf.nr_bindings : 0,
                    linked ? linked->b.cf.nr_cf : 0,
                    linked ? linked->b.cf.reads_z : 0,
                    vp->x, vp->y, vp->width, vp->height, vp->minDepth, vp->maxDepth,
                    dyn->vp.depth_clip_negative_one_to_one,
                    scissor->offset.x, scissor->offset.y,
                    scissor->extent.width, scissor->extent.height,
                    render->area.offset.x, render->area.offset.y,
                    render->area.extent.width, render->area.extent.height);
            fprintf(stderr, "M5 PPP flags: vertex_layout=%u psiz=%u bounded_viewport=%u "
                    "color=%u layers=%u viewmask=%u samples=%u pass=%u/%u "
                    "writes=%u discard=%u polygon=%u line=%g bias=%u/%g/%g/%g clamp=%u "
                    "clip=%u bounds=%u query=%u viewports=%u scissors=%u size=%u,%u\n",
                    vertex_layout, psiz, bounded_viewport,
                    render->color_att_count, render->layer_count, render->view_mask,
                    render->tilebuffer.nr_samples,
                    linked ? linked->b.fragment_control.opaque[0] >> 29 : 0,
                    AGX_PASS_TYPE_OPAQUE, fs ? fs->info.fs.writes_memory : 0,
                    hk_rast_discard(cmd), dyn->rs.polygon_mode, dyn->rs.line.width,
                    dyn->rs.depth_bias.enable, dyn->rs.depth_bias.constant_factor,
                    dyn->rs.depth_bias.slope_factor, dyn->rs.depth_bias.clamp,
                    dyn->rs.depth_clamp_enable,
                    vk_rasterization_state_depth_clip_enable(&dyn->rs),
                    dyn->ds.depth.bounds_test.enable, gfx->occlusion.mode,
                    dyn->vp.viewport_count, dyn->vp.scissor_count,
                    render->cr.width, render->cr.height);
            fprintf(stderr, "M5 PPP color_valid=%u depth_valid=%u depth_format=%u "
                            "depth_test=%u write=%u compare=%u counts=%#x/%#x\n",
                    color_varying, depth_format, render->depth_att.vk_format,
                    dyn->ds.depth.test_enable, dyn->ds.depth.write_enable,
                    dyn->ds.depth.compare_op, gfx->linked_varyings.counts_32.opaque[0],
                    gfx->linked_varyings.counts_16.opaque[0]);
         }
         mesa_loge("M5 draw exceeds the native color/depth PPP profile");
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      if (fs->b.info.has_preamble && !fs->b.info.agx3_preamble_staging_known) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      /* Software tessellation determines the generated triangle winding.
       * Match the shared CULL path when the TES feeds rasterization directly. */
      bool front_face_ccw =
         dyn->rs.front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;
      if (gfx->shaders[MESA_SHADER_TESS_CTRL] &&
          !gfx->shaders[MESA_SHADER_GEOMETRY]) {
         front_face_ccw ^= gfx->tess.info.ccw;
         front_face_ccw ^= dyn->ts.domain_origin ==
                           VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
      }
      struct agx_ppp_hal300_graphics_fields fields;
      /* Zeroed padding lets m5_ppp_pack_memo compare it bytewise. */
      memset(&fields, 0, sizeof(fields));
      /* A vertex-stage draw selects its provoking vertex through the VDM
       * vertex program template (hk_agx3_publish_vertex_state), but
       * the PPP flat-shading vertex must match it too. With it left at
       * vertex 0, the EDS tests' oversized last-vertex triangles shaded
       * from the wrong vertex on hardware while in-viewport ones were
       * right; presumably clipping takes the flat inputs of the vertices
       * it generates from the PPP selector. A native mesh draw has no VDM
       * template; the PPP selector alone selects it (hardware: EDS
       * last-vertex mesh draws pass with it and fail with the VDM
       * mesh-state word). */
      fields.flat_shading_vertex = gfx->provoking;
      fields.viewport_count = hk_effective_viewport_count(cmd);
      fields.general_varyings = general_varyings;
      fields.cf_register_count = cf->nr_cf;
      fields.varying_counts_32 = gfx->linked_varyings.counts_32.opaque[0];
      fields.varying_counts_16 = gfx->linked_varyings.counts_16.opaque[0];
      fields.output_select = uvs->osel.opaque[0] | linked->b.osel.opaque[0];
      fields.uvs_words = uvs->size;
      fields.fragment_program = hk_upload_usc_words(cmd, fs, linked);
      fields.varying_table = gfx->varyings;
      fields.cf_binding_count = cf->nr_bindings;
      fields.fragment_ti_spill_B = fs->b.info.has_preamble
            ? fs->b.info.preamble_scratch_size : 0;
      fields.fragment_ti_staging_B = fs->b.info.has_preamble
            ? fs->b.info.agx3_preamble_staging_halfs * 2u : 0;
      fields.no_color_attachment = !hk_m5_has_color(&render->tilebuffer);
      /* An attached target can have no colour transaction when the
       * shader omits it or its write mask is zero (stencil-only draws). */
      fields.no_color_writes = linked->b.g17p_colour_classes == 0;
      fields.scissor_enable = !discard;
      fields.depth_bias_enable = !discard && dyn->rs.depth_bias.enable &&
            gfx->object_type == AGX_OBJECT_TYPE_TRIANGLE &&
            (dyn->rs.depth_bias.constant_factor != 0.0f ||
             dyn->rs.depth_bias.slope_factor != 0.0f);
      fields.depth_clip_disable = !vk_rasterization_state_depth_clip_enable(&dyn->rs);
      fields.polygon_mode = translate_polygon_mode(dyn->rs.polygon_mode);
      fields.primitive_msaa = gfx->object_type == AGX_OBJECT_TYPE_LINE &&
            dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR;
      fields.line_width = agx_pack_line_width(dyn->rs.line.width);
      fields.rasterizer_discard = discard;
      fields.cull_mode = dyn->rs.cull_mode;
      fields.front_face_ccw = front_face_ccw;
      fields.point_size_output = psiz;
      fields.layer_output = layer_word;
      fields.clip_distance_words = clip_words;
      fields.needs_primitive_id = gfx->generate_primitive_id;
      fields.object_type = gfx->object_type == AGX_OBJECT_TYPE_POINT_SPRITE_UV01
                           ? AGX_PPP_HAL300_POINTS
                           : gfx->object_type == AGX_OBJECT_TYPE_LINE
                                ? AGX_PPP_HAL300_LINES
                                : AGX_PPP_HAL300_TRIANGLES;
      fields.vertex_varying_components = components;
      fields.fragment_varying_components = color_varying ? components : 0;
      fields.interpolation = !color_varying || color->perspective
                             ? AGX_PPP_HAL300_SMOOTH
                             : color->smooth ? AGX_PPP_HAL300_LINEAR
                                             : AGX_PPP_HAL300_FLAT;
      fields.fragment_pass = fragment_pass;
      /* AGX2 merges the linked FRAGMENT_CONTROL into its secondary word;
       * HAL300 needs the tag-write disable carried explicitly, or a
       * fragment that stores nothing (every channel of its format masked)
       * hides the tile beneath it. */
      fields.tag_write_disable =
            (linked->b.fragment_control.opaque[0] & BITFIELD_BIT(21)) != 0;
      fields.disable_tri_merging =
            (linked->b.fragment_control.opaque[0] & BITFIELD_BIT(26)) != 0;
      fields.fragment_writes_depth = fs->info.fs.epilog_key.write_z;
      fields.fragment_depth_mode = agx_ppp_fragment_depth_mode(
            agx_translate_depth_layout(fs->b.info.depth_layout),
            fs->b.info.early_fragment_tests, fs->info.fs.writes_memory);
      fields.fragment_reads_depth = cf->reads_z;
      fields.fragment_reads_reciprocal_w = reads_rhw;
      fields.depth_active = depth_active;
      fields.depth_compare = depth_active ? depth_compare : 0;
      fields.depth_write = depth_active && dyn->ds.depth.write_enable;
      fields.stencil_active = stencil_active;
      fields.front_stencil_reference = stencil_active
            ? dyn->ds.stencil.front.reference & 0xffu : 0;
      fields.back_stencil_reference = stencil_active
            ? dyn->ds.stencil.back.reference & 0xffu : 0;
      fields.front_stencil = front_stencil.opaque[0];
      fields.back_stencil = back_stencil.opaque[0];
      fields.visibility_mode = gfx->occlusion.mode;
      fields.visibility_index = gfx->occlusion.index;
      /* The stream already latched this state (see hk_cs). */
      if (cs->m5_ppp_emitted_valid &&
          !memcmp(&cs->m5_ppp_emitted, &fields, sizeof(fields)))
         return;

      __typeof__(gfx->m5_ppp_pack_memo) *pack = &gfx->m5_ppp_pack_memo;
      if (!pack->valid || memcmp(&pack->fields, &fields, sizeof(fields))) {
         pack->valid = false;
         if (!agx_ppp_hal300_graphics_core_pack(dev->dev.chip, &fields,
                                                &pack->state)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         pack->fields = fields;
         pack->valid = true;
      }
      const struct agx_ppp_hal300_graphics_core state = pack->state;

      /* Each record is a self-contained PPP update and the PPP keeps the
       * others, so only the records that differ from what the stream latched
       * are emitted. Every record the PPP processes stalls tiling for about
       * 50 ns, close to half a microsecond per draw for the whole core. */
      unsigned changed = 0;
      for (unsigned i = 0; i < AGX_PPP_HAL300_GRAPHICS_RECORDS; ++i) {
         if (!cs->m5_ppp_emitted_valid ||
             memcmp(cs->m5_ppp_emitted_state.bytes + state.offsets[i],
                    state.bytes + state.offsets[i], state.lengths[i]))
            changed |= BITFIELD_BIT(i);
      }

      if (changed) {
         struct agx_ptr upload =
            hk_pool_vdm_alloc(cmd, sizeof(state.bytes), 64);
         if (!upload.cpu)
            return;
         memcpy(upload.cpu, state.bytes, sizeof(state.bytes));

         uint8_t *start = *out;
         u_foreach_bit(i, changed) {
            if (!hk_ppp_emit_state(dev->dev.chip, out,
                                   upload.gpu + state.offsets[i],
                                   state.lengths[i])) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }
         }
         if (unlikely(dev->frame_anatomy))
            cs->anatomy_ppp += (*out - start) / AGX_PPP_STATE_LENGTH;
      }
      cs->m5_ppp_emitted = fields;
      cs->m5_ppp_emitted_state = state;
      cs->m5_ppp_emitted_valid = true;
      return;
   }


   bool hw_vs_dirty = IS_SHADER_DIRTY(VERTEX) || IS_SHADER_DIRTY(TESS_EVAL) ||
                      IS_SHADER_DIRTY(GEOMETRY);
   bool fs_dirty = IS_SHADER_DIRTY(FRAGMENT);

   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   bool linked_fs_dirty = IS_LINKED_DIRTY(FRAGMENT);
   /* The launch follows the linked program's own layout, if it has one. */
   if (linked_fs && linked_fs->g17p_layout)
      fs = linked_fs->g17p_layout;

   bool varyings_dirty = gfx->dirty & HK_DIRTY_VARYINGS;

   bool face_dirty =
      IS_DIRTY(DS_DEPTH_TEST_ENABLE) || IS_DIRTY(DS_DEPTH_WRITE_ENABLE) ||
      IS_DIRTY(DS_DEPTH_COMPARE_OP) || IS_DIRTY(DS_STENCIL_REFERENCE) ||
      IS_DIRTY(RS_LINE_WIDTH) || IS_DIRTY(RS_POLYGON_MODE) || fs_dirty;

   bool stencil_face_dirty =
      IS_DIRTY(DS_STENCIL_OP) || IS_DIRTY(DS_STENCIL_COMPARE_MASK) ||
      IS_DIRTY(DS_STENCIL_WRITE_MASK) || IS_DIRTY(DS_STENCIL_TEST_ENABLE);

   struct AGX_PPP_HEADER dirty = {
      .fragment_control =
         IS_DIRTY(DS_STENCIL_TEST_ENABLE) || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
         IS_DIRTY(RS_DEPTH_BIAS_ENABLE) || gfx->dirty & HK_DIRTY_OCCLUSION,

      .fragment_control_2 =
         IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) || linked_fs_dirty,

      .fragment_front_face = face_dirty,
      .fragment_front_face_2 = fs_dirty || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY),
      .fragment_front_stencil = stencil_face_dirty,
      .fragment_back_face = face_dirty,
      .fragment_back_face_2 = fs_dirty || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY),
      .fragment_back_stencil = stencil_face_dirty,
      .output_select = hw_vs_dirty || linked_fs_dirty || varyings_dirty,
      .varying_counts_32 = varyings_dirty,
      .varying_counts_16 = varyings_dirty,
      .cull = IS_DIRTY(RS_CULL_MODE) ||
              IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) ||
              IS_DIRTY(RS_FRONT_FACE) || IS_DIRTY(RS_DEPTH_CLIP_ENABLE) ||
              IS_DIRTY(RS_DEPTH_CLAMP_ENABLE) || IS_DIRTY(RS_LINE_MODE) ||
              IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
              (gfx->dirty & HK_DIRTY_PROVOKING) || IS_SHADER_DIRTY(TESS_CTRL) ||
              IS_SHADER_DIRTY(TESS_EVAL) || IS_DIRTY(TS_DOMAIN_ORIGIN),
      .cull_2 = varyings_dirty,

      .fragment_shader = fs_dirty || linked_fs_dirty || varyings_dirty ||
                         gfx->descriptors.root_dirty,

      .occlusion_query = gfx->dirty & HK_DIRTY_OCCLUSION,
      .output_size = hw_vs_dirty,
      .viewport_count = 1, /* irrelevant */
   };

   if (!agx_native_graphics(&dev->dev))
      dirty.fragment_shader &= !linked_fs->b.no_op;

   /* Calculate the update size. If it equals the header, there is nothing to
    * update so early-exit.
    */
   size_t size = agx_ppp_update_size(&dirty);
   if (size == AGX_PPP_HEADER_LENGTH)
      return;

   /* Otherwise, allocate enough space for the update and push it. */
   assert(size > AGX_PPP_HEADER_LENGTH);

   struct agx_ptr T = hk_pool_vdm_alloc(cmd, size, AGX_PPP_HEADER_ALIGN);
   if (!T.cpu)
      return;

   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &dirty);

   if (dirty.fragment_control) {
      agx_ppp_push(&ppp, FRAGMENT_CONTROL, cfg) {
         cfg.visibility_mode = gfx->occlusion.mode;
         cfg.stencil_test_enable = hk_stencil_test_enabled(cmd);

         /* TODO: Consider optimizing this? */
         cfg.two_sided_stencil = cfg.stencil_test_enable;

         cfg.depth_bias_enable = dyn->rs.depth_bias.enable &&
                                 gfx->object_type == AGX_OBJECT_TYPE_TRIANGLE;

         /* Always enable scissoring so we may scissor to the viewport (TODO:
          * optimize this out if the viewport is the default and the app does
          * not use the scissor test)
          */
         cfg.scissor_enable = true;

         /* This avoids broken derivatives along primitive edges */
         cfg.disable_tri_merging = gfx->object_type != AGX_OBJECT_TYPE_TRIANGLE;
      }
   }

   if (dirty.fragment_control_2) {
      /* Annoying, rasterizer_discard seems to be ignored (sometimes?) in the
       * main fragment control word and has to be combined into the secondary
       * word for reliable behaviour.
       */
      agx_ppp_push_merged(&ppp, FRAGMENT_CONTROL, cfg,
                          linked_fs->b.fragment_control) {

         cfg.tag_write_disable = hk_rast_discard(cmd);
      }
   }

   struct agx_fragment_face_packed fragment_face = {};
   struct agx_fragment_face_2_packed fragment_face_2 = {};

   if (dirty.fragment_front_face) {
      bool has_z = render->depth_att.vk_format != VK_FORMAT_UNDEFINED;
      bool z_test = has_z && dyn->ds.depth.test_enable;

      agx_pack(&fragment_face, FRAGMENT_FACE, cfg) {
         cfg.line_width = agx_pack_line_width(dyn->rs.line.width);
         cfg.polygon_mode = translate_polygon_mode(dyn->rs.polygon_mode);
         cfg.disable_depth_write = !(z_test && dyn->ds.depth.write_enable);

         if (z_test && !gfx->descriptors.root.draw.force_never_in_shader)
            cfg.depth_function = translate_compare_op(dyn->ds.depth.compare_op);
         else
            cfg.depth_function = AGX_ZS_FUNC_ALWAYS;
      };

      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, fragment_face) {
         cfg.stencil_reference = dyn->ds.stencil.front.reference;
      }
   }

   if (dirty.fragment_front_face_2) {
      agx_pack(&fragment_face_2, FRAGMENT_FACE_2, cfg) {
         cfg.object_type = gfx->object_type;
      }

      agx_merge(fragment_face_2, fs->frag_face, FRAGMENT_FACE_2);
      agx_ppp_push_packed(&ppp, &fragment_face_2, FRAGMENT_FACE_2);
   }

   if (dirty.fragment_front_stencil) {
      hk_ppp_push_stencil_face(&ppp, dyn->ds.stencil.front,
                               hk_stencil_test_enabled(cmd));
   }

   if (dirty.fragment_back_face) {
      assert(dirty.fragment_front_face);

      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, fragment_face) {
         cfg.stencil_reference = dyn->ds.stencil.back.reference;
      }
   }

   if (dirty.fragment_back_face_2) {
      assert(dirty.fragment_front_face_2);

      agx_ppp_push_packed(&ppp, &fragment_face_2, FRAGMENT_FACE_2);
   }

   if (dirty.fragment_back_stencil) {
      hk_ppp_push_stencil_face(&ppp, dyn->ds.stencil.back,
                               hk_stencil_test_enabled(cmd));
   }

   if (dirty.output_select) {
      struct agx_output_select_packed osel = hw_vs->info.uvs.osel;

      agx_ppp_push_merged_blobs(&ppp, AGX_OUTPUT_SELECT_LENGTH, &osel,
                                &linked_fs->b.osel);
   }

   assert(dirty.varying_counts_32 == dirty.varying_counts_16);

   if (dirty.varying_counts_32) {
      agx_ppp_push_packed(&ppp, &gfx->linked_varyings.counts_32,
                          VARYING_COUNTS);

      agx_ppp_push_packed(&ppp, &gfx->linked_varyings.counts_16,
                          VARYING_COUNTS);
   }

   if (dirty.cull) {
      agx_ppp_push(&ppp, CULL, cfg) {
         cfg.cull_front = dyn->rs.cull_mode & VK_CULL_MODE_FRONT_BIT;
         cfg.cull_back = dyn->rs.cull_mode & VK_CULL_MODE_BACK_BIT;
         cfg.front_face_ccw = dyn->rs.front_face != VK_FRONT_FACE_CLOCKWISE;

         if (gfx->shaders[MESA_SHADER_TESS_CTRL] &&
             !gfx->shaders[MESA_SHADER_GEOMETRY]) {
            cfg.front_face_ccw ^= gfx->tess.info.ccw;
            cfg.front_face_ccw ^= dyn->ts.domain_origin ==
                                  VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT;
         }

         cfg.flat_shading_vertex = translate_ppp_vertex(gfx->provoking);
         cfg.rasterizer_discard = hk_rast_discard(cmd);

         /* We do not support unrestricted depth, so clamping is inverted from
          * clipping. This implementation seems to pass CTS without unrestricted
          * depth support.
          *
          * TODO: Make sure this is right with gl_FragDepth.
          */
         cfg.depth_clip = vk_rasterization_state_depth_clip_enable(&dyn->rs);
         cfg.depth_clamp = !cfg.depth_clip;

         cfg.primitive_msaa =
            gfx->object_type == AGX_OBJECT_TYPE_LINE &&
            dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR;
      }
   }

   if (dirty.cull_2) {
      agx_ppp_push(&ppp, CULL_2, cfg) {
         cfg.needs_primitive_id = gfx->generate_primitive_id;
         cfg.clamp_w = true;
      }
   }

   if (dirty.fragment_shader) {
      /* TODO: Do less often? */
      hk_reserve_scratch(cmd, cs, fs);

      const uint64_t uploaded = hk_upload_usc_words(cmd, fs, linked_fs);
      uint32_t fragment_pipeline;
      switch (agx_chip_command_stream_generation(dev->dev.chip)) {
      case AGX_CS_GENERATION_AGX2:
         if (uploaded > UINT32_MAX) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         fragment_pipeline = uploaded;
         break;
      case AGX_CS_GENERATION_G15:
      case AGX_CS_GENERATION_HAL200:
         if (uploaded == 0 || uploaded < dev->dev.shader_base ||
             uploaded - dev->dev.shader_base > UINT32_MAX) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         fragment_pipeline = agx_usc_addr(&dev->dev, uploaded);
         break;
      case AGX_CS_GENERATION_HAL300:
      case AGX_CS_GENERATION_UNKNOWN:
      default:
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      agx_ppp_push_packed(&ppp, &linked_fs->fs_counts, FRAGMENT_SHADER_WORD_0);

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_1, cfg) {
         cfg.pipeline = fragment_pipeline;
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_2, cfg) {
         cfg.cf_bindings = gfx->varyings;
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_3, cfg)
         ;
   }

   if (dirty.occlusion_query) {
      agx_ppp_push(&ppp, FRAGMENT_OCCLUSION_QUERY, cfg) {
         cfg.index = gfx->occlusion.index;
      }
   }

   if (dirty.output_size) {
      agx_ppp_push(&ppp, OUTPUT_SIZE, cfg) {
         cfg.count = hw_vs->info.uvs.size;
      }
   }

   const bool ppp_ok = agx_ppp_legacy_payload_supported(dev->dev.chip) &&
                       agx_ppp_fini(dev->dev.chip, out, &ppp);
   if (!ppp_ok) {
      mesa_loge("no grounded legacy PPP payload/finalizer for %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   }
}

/*
 * Based somewhat on the calculation in the PowerVR driver, and mostly trial &
 * error to pass CTS. This is a mess.
 */
static float
hk_depth_bias_factor(VkFormat format, bool exact, bool force_unorm)
{
   if (hk_depth_stencil_plane_format(format, 0) == VK_FORMAT_D16_UNORM) {
      return exact ? (1 << 16) : (1 << 15);
   } else if (force_unorm) {
      return exact ? (1ull << 24) : (1ull << 23);
   } else {
      return 1.0;
   }
}

/* HAL200 texture and sampler heaps consume fragment uniform halves0..7. The
 * ordinary ABI's blend constants at halves4..11 therefore cannot coexist
 * with an indexed/combined texture shader. Shader parts can address any even
 * uniform half directly, so place the epilog-only constants after every main
 * shader range. The Ioto program publishes that exact relocated range.
 */
static bool
hk_g17p_relocated_blend_constant_base(const struct hk_shader *fs,
                                      uint16_t *base_out)
{
   if (!fs || !base_out)
      return false;

   unsigned base = MAX2(AGX_G17P_TXF_ROOT_HALFS, fs->b.info.push_count);
   if (fs->b.info.rodata.size_16) {
      const unsigned rodata_end = fs->b.info.rodata.base_uniform +
                                  fs->b.info.rodata.size_16;
      base = MAX2(base, rodata_end);
   }
   base = ALIGN_POT(base, 2);
   if (base > 512 - 8)
      return false;

   *base_out = base;
   return true;
}

/* Everything outside a fragment shader that its fast-link key depends on. */
struct hk_fs_link_inputs {
   const struct vk_dynamic_graphics_state *dyn;
   const struct agx_tilebuffer_layout *tib;
   const VkFormat *color_formats;
   unsigned color_formats_stride;
   unsigned color_att_count;
   unsigned cull_distance_size;
   bool statistics;
   bool depth_attachment;
   bool stencil_attachment;
   bool occlusion;
};

static bool
hk_g17p_read_only_early_reject_in(const struct hk_device *dev,
                                  const struct hk_shader *fs,
                                  const struct hk_fs_link_inputs *in,
                                  unsigned samples, bool prolog_discards)
{
   const struct vk_dynamic_graphics_state *dyn = in->dyn;

   /* Moving a test ahead of a discard is unobservable only when neither the
    * test nor the skipped fragment has a side effect. The all-live prolog
    * also must not inherit an API-mask or cull-distance kill. */
   return agx_native_graphics(&dev->dev) && fs->b.info.writes_sample_mask &&
          !fs->info.fs.static_tests &&
          !fs->b.info.early_fragment_tests && !fs->info.fs.writes_memory &&
          !fs->info.fs.epilog_key.write_z && !fs->info.fs.epilog_key.write_s &&
          samples == 1 && in->depth_attachment && dyn->ds.depth.test_enable &&
          !dyn->ds.depth.write_enable &&
          dyn->ds.depth.compare_op != VK_COMPARE_OP_ALWAYS &&
          !dyn->ds.depth.bounds_test.enable &&
          !(dyn->ds.stencil.test_enable && in->stencil_attachment) &&
          !in->occlusion && !in->statistics && !prolog_discards;
}

static struct hk_fs_link_inputs
hk_cmd_fs_link_inputs(struct hk_cmd_buffer *cmd, const struct hk_shader *hw_vs)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   const struct hk_rendering_state *render = &gfx->render;
   return (struct hk_fs_link_inputs){
      .dyn = &cmd->vk.dynamic_graphics_state,
      .tib = &render->tilebuffer,
      .color_formats = &render->color_att[0].vk_format,
      .color_formats_stride = sizeof(render->color_att[0]),
      .color_att_count = render->color_att_count,
      .cull_distance_size = hw_vs->info.cull_distance_array_size,
      .statistics = hk_stat_enabled(hk_pipeline_stat_addr(
         cmd, VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT)),
      .depth_attachment = render->depth_att.vk_format != VK_FORMAT_UNDEFINED,
      .stencil_attachment =
         render->stencil_att.vk_format != VK_FORMAT_UNDEFINED,
      .occlusion = gfx->occlusion.mode != AGX_VISIBILITY_MODE_NONE,
   };
}

static bool
hk_g17p_read_only_early_reject(struct hk_cmd_buffer *cmd,
                               const struct hk_shader *fs, unsigned samples,
                               bool prolog_discards, bool statistics)
{
   struct hk_fs_link_inputs in =
      hk_cmd_fs_link_inputs(cmd, hk_bound_hw_vs(&cmd->state.gfx));
   in.statistics = statistics;
   return hk_g17p_read_only_early_reject_in(hk_cmd_buffer_device(cmd), fs,
                                            &in, samples, prolog_discards);
}

void
agx_fill_velem_keys(const struct vk_vertex_input_state *vi,
                    uint64_t attribs_read, struct agx_velem_key *keys)
{
   u_foreach_bit(a, vi->attributes_valid) {
      /* An attribute the shader does not read has no slot. Its "slot" below
       * would be the count of read attributes beneath it, which for an
       * attribute above every read one is the total count -- one past the
       * key when the shader reads the maximum.
       */
      if (!(attribs_read & BITFIELD64_BIT(a)))
         continue;

      struct vk_vertex_attribute_state attr = vi->attributes[a];

      assert(vi->bindings_valid & BITFIELD_BIT(attr.binding));
      struct vk_vertex_binding_state binding = vi->bindings[attr.binding];

      /* nir_assign_io_var_locations compacts vertex inputs, eliminating
       * unused inputs. We need to do the same here to match the locations.
       */
      unsigned slot = util_bitcount64(attribs_read & BITFIELD64_MASK(a));
      assert(slot < AGX_MAX_ATTRIBS);

      keys[slot] = (struct agx_velem_key){
         .format = hk_format_to_pipe_format(attr.format),
         .stride = binding.stride,
         .divisor = binding.divisor,
         .instanced = binding.input_rate == VK_VERTEX_INPUT_RATE_INSTANCE,
      };
   }
}

static void
hk_fill_fast_link_key_vs(struct hk_cmd_buffer *cmd, struct hk_shader *sw_vs,
                         struct hk_shader *hw_vs, bool indexed,
                         enum agx_index_size index_size, bool restart,
                         struct hk_fast_link_key_vs *key)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   *key = (struct hk_fast_link_key_vs){
      .prolog.hw = (sw_vs == hw_vs),
      .prolog.uniform_base = (sw_vs->info.agx3_indexed_sampler_valid
                                 ? AGX_ABI_G17P_TEXTURE_POINTER_HALFS
                                 : 0) +
                             (sw_vs->info.agx3_gen1_fragment_rq ? 168 : 0),

      /* FIXME: handle pipeline robustness "properly" */
      .prolog.robustness.level =
         (dev->vk.enabled_features.robustBufferAccess2 ||
          dev->vk.enabled_features.pipelineRobustness)
            ? AGX_ROBUSTNESS_D3D
         : dev->vk.enabled_features.robustBufferAccess
            ? AGX_ROBUSTNESS_GL
            : AGX_ROBUSTNESS_DISABLED,
      .prolog.robustness.soft_fault = agx_has_soft_fault(&dev->dev),
      .prolog.static_vi = !sw_vs->info.vs.use_prolog,
   };

   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);
   if (mesa_prim_has_adjacency(prim)) {
      if (restart)
         prim = u_decomposed_prim(prim);

      /* G17P materialises adjacency into an index buffer before the hardware
       * vertex program. Applying the remap again in the prolog is wrong. */
      if (!agx_native_graphics(&dev->dev) && !agx_chip_g17_m5(dev->dev.chip))
         key->prolog.adjacency = prim;
   }

   if (key->prolog.adjacency || !key->prolog.hw) {
      key->prolog.sw_index_size_B =
         indexed ? agx_index_size_to_B(index_size) : 0;
   }

   static_assert(sizeof(key->prolog.component_mask) ==
                 sizeof(sw_vs->info.vs.attrib_components_read));
   BITSET_COPY(key->prolog.component_mask,
               sw_vs->info.vs.attrib_components_read);

   if (sw_vs->info.vs.use_prolog) {
      agx_fill_velem_keys(dyn->vi, sw_vs->info.vs.attribs_read,
                          key->prolog.attribs);

      u_foreach_bit(a, dyn->vi->attributes_valid) {
         if (!(sw_vs->info.vs.attribs_read & BITFIELD64_BIT(a)))
            continue;

         unsigned slot = util_bitcount64(sw_vs->info.vs.attribs_read &
                                         BITFIELD64_MASK(a));
         key->prolog.attribs[slot].stride =
            dyn->vi_binding_strides[dyn->vi->attributes[a].binding];
      }
   }
}

static bool
hk_fs_link_key_build(const struct hk_device *dev, struct hk_shader *fs,
                     const struct hk_fs_link_inputs *in,
                     struct hk_fast_link_key_fs *key, bool report_errors)
{
   const struct vk_dynamic_graphics_state *dyn = in->dyn;
   unsigned samples = MAX2(dyn->ms.rasterization_samples, 1);
   unsigned tib_sample_mask = BITFIELD_MASK(dyn->ms.rasterization_samples);
   unsigned api_sample_mask = dyn->ms.sample_mask & tib_sample_mask;
   bool has_sample_mask = api_sample_mask != tib_sample_mask;
   bool nontrivial_force_early =
      fs && (fs->b.info.early_fragment_tests &&
             (fs->b.info.writes_sample_mask || fs->info.fs.writes_memory));
   bool epilog_discards = agx_fs_epilog_may_discard(
      fs ? &fs->info.fs.epilog_key : NULL,
      dyn->ms.alpha_to_coverage_enable, nontrivial_force_early);
   bool defer_api_sample_mask = agx_native_graphics(&dev->dev) &&
                                has_sample_mask && !nontrivial_force_early;

   const bool native_samples = agx_native_graphics(&dev->dev) ||
                               agx_chip_g17_m5(dev->dev.chip);
   if (native_samples && fs->info.fs.uses_sample_shading &&
       dyn->ms.rasterization_samples > 1) {
      if (report_errors)
         mesa_loge("G17P: per-sample fragment execution needs a fixed sample "
                   "count; refusing a %u-sample draw that requests it",
                   dyn->ms.rasterization_samples);
      return false;
   }

   if (native_samples &&
       fs->info.fs.epilog_key.sample_shading &&
       fs->info.fs.g17p_samples != samples) {
      if (report_errors)
         mesa_loge("G17P: fragment unrolled for %u samples cannot run a "
                   "%u-sample draw",
                   fs->info.fs.g17p_samples, dyn->ms.rasterization_samples);
      return false;
   }

   if (native_samples && fs->b.info.writes_sample_mask &&
       fs->info.fs.g17p_samples != samples) {
      if (report_errors)
         mesa_loge("G17P: fragment coverage updates compiled for %u samples "
                   "cannot run a %u-sample draw",
                   fs->info.fs.g17p_samples, dyn->ms.rasterization_samples);
      return false;
   }

   unsigned samples_shaded = 0;
   if (fs->info.fs.epilog_key.sample_shading &&
       !native_samples)
      samples_shaded = dyn->ms.rasterization_samples;

   *key = (struct hk_fast_link_key_fs){
      .prolog.statistics = in->statistics,
      .prolog.cull_distance_size = in->cull_distance_size,
      /* AGX2 must apply the API mask before an explicit-early main part
       * counts samples, just as upstream does. Only the native and M5 paths
       * fold that mask into their forced-early coverage word
       * (agx_nir_apply_api_mask_to_force_early()). */
      .prolog.api_sample_mask =
         has_sample_mask && !defer_api_sample_mask &&
               ((!agx_native_graphics(&dev->dev) &&
                 !agx_chip_g17_m5(dev->dev.chip)) ||
                !nontrivial_force_early)
            ? api_sample_mask
            : 0xff,
      .prolog.nr_samples = native_samples ? samples : 0,
      .nr_samples_shaded = samples_shaded,
   };

   bool prolog_discards =
      (has_sample_mask && !defer_api_sample_mask) ||
      key->prolog.cull_distance_size;
   if (key->prolog.statistics || prolog_discards) {
      key->prolog.run_zs_tests = !nontrivial_force_early &&
                                 !fs->b.info.writes_sample_mask &&
                                 !epilog_discards && prolog_discards;
      if (key->prolog.cull_distance_size)
         key->prolog.cf_base = fs->b.info.varyings.fs.nr_cf;
   }

   key->epilog = (struct agx_fs_epilog_key){
      .link = fs->info.fs.epilog_key,
      .colour_unit_range = fs->info.fs.g17p_colour_unit_range,
      .nr_samples = samples,
      .blend.alpha_to_coverage = dyn->ms.alpha_to_coverage_enable,
      .blend.alpha_to_one = dyn->ms.alpha_to_one_enable,
      .blend.logicop_enable = dyn->cb.logic_op_enable,
      .blend.logicop_func = vk_logic_op_to_pipe(dyn->cb.logic_op),
      .defer_api_sample_mask = defer_api_sample_mask,
      .api_sample_mask = api_sample_mask,
   };
   if (hk_g17p_read_only_early_reject_in(dev, fs, in, samples,
                                         prolog_discards)) {
      key->prolog.run_zs_tests = true;
      key->epilog.link.already_ran_zs = true;
   }
   agx_fs_epilog_apply_device_policy(dev->dev.chip, &key->epilog);

   key->epilog.native_spills =
      (agx_native_graphics(&dev->dev) ||
       agx_chip_g17_m5(dev->dev.chip)) &&
      agx_tilebuffer_spills(in->tib);
   if (key->epilog.native_spills) {
      uint16_t base;
      if (!hk_g17p_relocated_blend_constant_base(fs, &base) ||
          ALIGN_POT(base, 4) > 512 - 12)
         return false;
      key->epilog.emrt_heap_base = ALIGN_POT(base, 4);
      key->epilog.blend_constant_base = key->epilog.emrt_heap_base + 4;
      /* The M5 image-write encoder accepts the low 256 uniform halves. The
       * resource heaps occupy 0..7 and the root starts at 12; the mandatory
       * inline program can use the reserved 8..11 gap for its spill heap. */
      if (fs->info.fs.m5_inline_sample_shading &&
          !fs->info.agx3_gen1_fragment_rq &&
          fs->info.agx3_root_uniform == AGX_ABI_FUNI_ROOT)
         key->epilog.emrt_heap_base = 8;
   }

   if (!key->epilog.native_spills &&
       (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
       hk_uses_blend_constant(&dyn->cb) &&
       (fs->info.agx3_combined_sampler_valid ||
        fs->info.agx3_indexed_sampler_valid) &&
       !hk_g17p_relocated_blend_constant_base(
          fs, &key->epilog.blend_constant_base))
      return false;

   for (unsigned rt = 0; rt < ARRAY_SIZE(dyn->cal.color_map); ++rt) {
      int map = dyn->cal.color_map[rt];
      key->epilog.remap[rt] =
         map == MESA_VK_ATTACHMENT_UNUSED ? -1 : map;
   }

   key->epilog.link.already_ran_zs |= nontrivial_force_early;
   key->epilog.native_imageblock = in->tib->native_imageblock;
   key->epilog.max_sample_size_B = in->tib->max_sample_size_B;
   key->epilog.resident_mask = in->tib->resident_mask;
   key->epilog.max_bytes_per_tile = in->tib->max_bytes_per_tile;
   key->epilog.max_samples_per_tile = in->tib->max_samples_per_tile;

   for (uint32_t i = 0; i < in->color_att_count; i++) {
      const VkFormat format = *(const VkFormat *)(
         (const uint8_t *)in->color_formats + i * in->color_formats_stride);
      key->epilog.rt_formats[i] = hk_format_to_pipe_format(format);
      if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
         const enum pipe_format logical = key->epilog.rt_formats[i];
         key->epilog.rt_formats[i] = hk_padded_image_format(logical);
         const enum pipe_format integer =
            hk_scaled_texture_format(key->epilog.rt_formats[i]);
         if (integer != PIPE_FORMAT_NONE) {
            key->epilog.rt_formats[i] = integer;
            key->epilog.scaled_formats[i] = logical;
         } else if (key->epilog.rt_formats[i] != logical) {
            key->epilog.blend_formats[i] = logical;
         }
      }

      const struct vk_color_blend_attachment_state *cb =
         &dyn->cb.attachments[i];
      bool write_enable = dyn->cb.color_write_enables & BITFIELD_BIT(i);
      unsigned write_mask = write_enable ? cb->write_mask : 0;

      if ((dev->dev.chip == AGX_CHIP_G16G ||
           agx_native_graphics(&dev->dev)) && !write_enable) {
         key->epilog.preserve_rt |= BITFIELD_BIT(i);
         key->epilog.blend.rt[i] = (struct agx_blend_rt_key){
            .colormask = BITFIELD_MASK(4),
            .mode = agx_pack_blend_standard(
               PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
               PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO),
         };
      } else if (!cb->blend_enable) {
         key->epilog.blend.rt[i] = (struct agx_blend_rt_key){
            .colormask = write_mask,
            .mode = agx_pack_blend_standard(
               PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
               PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO),
         };
      } else if (cb->color_blend_op >= VK_BLEND_OP_ZERO_EXT) {
         key->epilog.blend.rt[i] = (struct agx_blend_rt_key){
            .colormask = write_mask,
            .advanced_blend = 1,
            .mode = agx_pack_blend_advanced(
               vk_advanced_blend_op_to_pipe(cb->color_blend_op),
               vk_blend_overlap_to_pipe(cb->blend_overlap),
               cb->src_premultiplied, cb->dst_premultiplied,
               cb->clamp_results),
         };
         assert(cb->clamp_results == false);
      } else {
         key->epilog.blend.rt[i] = (struct agx_blend_rt_key){
            .colormask = write_mask,
            .mode = agx_pack_blend_standard(
               vk_blend_op_to_pipe(cb->color_blend_op),
               vk_blend_factor_to_pipe(cb->src_color_blend_factor),
               vk_blend_factor_to_pipe(cb->dst_color_blend_factor),
               vk_blend_op_to_pipe(cb->alpha_blend_op),
               vk_blend_factor_to_pipe(cb->src_alpha_blend_factor),
               vk_blend_factor_to_pipe(cb->dst_alpha_blend_factor)),
         };
      }
   }

   return true;
}

/* Colour attachments a draw with this fragment link key writes: the
 * attachment is mapped to a location the shader writes (location 0 for
 * every attachment with gl_FragColor broadcast) and its write mask is not
 * empty, or, with preserved_written (M5), the epilog preserves it. */
static uint8_t
hk_fs_color_written(const struct hk_fast_link_key_fs *key,
                    bool preserved_written)
{
   const struct agx_fs_epilog_link_info *link = &key->epilog.link;
   uint8_t mask = 0;
   for (unsigned rt = 0; rt < ARRAY_SIZE(key->epilog.remap); ++rt) {
      const int loc = key->epilog.remap[rt];
      if (loc < 0)
         continue;
      const unsigned src = link->broadcast_rt0 ? 0 : loc;
      if ((link->loc_written & BITFIELD_BIT(src)) &&
          (key->epilog.blend.rt[rt].colormask ||
           (preserved_written &&
            (key->epilog.preserve_rt & BITFIELD_BIT(rt)))))
         mask |= BITFIELD_BIT(rt);
   }
   return mask;
}

static bool
hk_fill_fast_link_key_fs(struct hk_cmd_buffer *cmd, struct hk_shader *fs,
                         struct hk_shader *hw_vs,
                         struct hk_fast_link_key_fs *key, bool report_errors)
{
   const struct hk_fs_link_inputs in = hk_cmd_fs_link_inputs(cmd, hw_vs);
   return hk_fs_link_key_build(hk_cmd_buffer_device(cmd), fs, &in, key,
                               report_errors);
}

void
hk_prelink_pipeline_fragment(struct hk_device *dev,
                             struct hk_api_shader *vs_api,
                             struct hk_api_shader *fs_api,
                             const struct vk_graphics_pipeline_state *state)
{
   static const enum mesa_vk_dynamic_graphics_state key_states[] = {
      MESA_VK_DYNAMIC_MS_RASTERIZATION_SAMPLES,
      MESA_VK_DYNAMIC_MS_SAMPLE_MASK,
      MESA_VK_DYNAMIC_MS_ALPHA_TO_COVERAGE_ENABLE,
      MESA_VK_DYNAMIC_MS_ALPHA_TO_ONE_ENABLE,
      MESA_VK_DYNAMIC_CB_LOGIC_OP_ENABLE,
      MESA_VK_DYNAMIC_CB_LOGIC_OP,
      MESA_VK_DYNAMIC_CB_ATTACHMENT_COUNT,
      MESA_VK_DYNAMIC_CB_COLOR_WRITE_ENABLES,
      MESA_VK_DYNAMIC_CB_BLEND_ENABLES,
      MESA_VK_DYNAMIC_CB_BLEND_EQUATIONS,
      MESA_VK_DYNAMIC_CB_WRITE_MASKS,
   };

   if (!state->rp || !state->ms || !state->cb)
      return;
   for (unsigned i = 0; i < ARRAY_SIZE(key_states); ++i) {
      if (BITSET_TEST(state->dynamic, key_states[i]))
         return;
   }

   struct vk_dynamic_graphics_state dyn;
   struct vk_vertex_input_state vi;
   struct vk_sample_locations_state sl;
   vk_dynamic_graphics_state_init(&dyn);
   dyn.vi = &vi;
   dyn.ms.sample_locations = &sl;
   vk_dynamic_graphics_state_fill(&dyn, state);

   for (unsigned i = 0; i < dyn.cb.attachment_count; ++i) {
      if (dyn.cb.attachments[i].color_blend_op >= VK_BLEND_OP_ZERO_EXT &&
          BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_ADVANCED))
         return;
   }

   const unsigned samples = MAX2(dyn.ms.rasterization_samples, 1);
   if (hk_fs_build_sample_variant(dev, fs_api, samples) != VK_SUCCESS)
      return;
   struct hk_shader *fs = hk_fs_variant(fs_api, samples);
   if (fs->b.info.writes_sample_mask)
      return;

   const uint32_t view_mask = state->mv ? state->mv->view_mask : 0;
   struct agx_tilebuffer_layout tib = hk_tilebuffer_layout(
      dev, state->rp->color_attachment_formats, sizeof(VkFormat),
      state->rp->color_attachment_count, samples, view_mask != 0,
      samples > 1 && view_mask != 0, 0);
   if (!hk_agx3_g17p_rt0_tile_layout_supported(&tib))
      return;

   const struct hk_fs_link_inputs in = {
      .dyn = &dyn,
      .tib = &tib,
      .color_formats = state->rp->color_attachment_formats,
      .color_formats_stride = sizeof(VkFormat),
      .color_att_count = state->rp->color_attachment_count,
      .cull_distance_size =
         vs_api->variants[HK_VS_VARIANT_HW].info.cull_distance_array_size,
   };
   struct hk_fast_link_key_fs key;
   if (hk_fs_link_key_build(dev, fs, &in, &key, false))
      hk_get_fast_linked(dev, fs, &key);
}

/*
 * Does this draw republish render state, or does it reuse what the previous
 * draw in the same stream latched?
 *
 * This is the exact condition hk_flush_dynamic_state() uses to decide whether
 * it writes anything at all, hoisted so the split policy and the emitter
 * cannot drift apart.  It must be called at the point where every producer of
 * dirtiness -- the passthrough-GS bind, hk_flush_shaders(), the push-descriptor
 * flush -- has already run, otherwise it answers for a different draw than the
 * one that gets recorded.
 */
static bool
hk_gfx_republishes_render_state(struct hk_cmd_buffer *cmd)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);

   return vk_dynamic_graphics_state_any_dirty(dyn) || gfx->dirty ||
          gfx->descriptors.root_dirty || gfx->shaders_dirty ||
          sw_vs->b.info.uses_draw_id || sw_vs->b.info.uses_base_param ||
          (gfx->linked[MESA_SHADER_VERTEX] &&
           gfx->linked[MESA_SHADER_VERTEX]->b.uses_base_param);
}

/* The AGX3 VertexProgramVariant state update for pipeline_program and, on
 * G17P, the native HAL200 bind tail: the vertex-program publication of every
 * draw that republishes render state.  Shared by hk_flush_dynamic_state_slow()
 * and hk_g17p_emit_indirect_record(), so both emit exactly the same records.
 * tail_known_match: see hk_g17p_emit_native_bind_tail().  Returns false, with
 * the command buffer error set and cs->current at the failed record, when
 * publication is refused. */
static bool
hk_agx3_publish_vertex_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                             uint8_t **out_p, struct hk_shader *hw_vs,
                             uint64_t pipeline_program, struct agx_draw draw,
                             bool ordinary_draw_path, bool tail_known_match,
                             bool *tail_emitted)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint8_t *out = *out_p;

   *tail_emitted = false;
   const struct agx_vdm_state_update_program_info *program =
      &hw_vs->agx3_vdm_state_update;
   const bool pipeline_va_valid =
      pipeline_program != 0 && pipeline_program < (UINT64_C(1) << 48) &&
      (pipeline_program & UINT64_C(0x3f)) == 0;
   const enum agx_vdm_state_update_draw_path draw_path =
      agx_is_indirect(draw.b)
         ? (draw.indexed ? AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_INDEXED
                         : AGX_VDM_STATE_UPDATE_DRAW_INDIRECT_NONINDEXED)
         : (draw.indexed ? AGX_VDM_STATE_UPDATE_DRAW_DIRECT_INDEXED
                         : AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED);
   /* RenderPipeline+0x2f0 selects the native vertex template. TES and GS
    * raster variants can write external memory just like API vertex shaders
    * and software XFB. Keep that program live when rasterization is discarded;
    * the independently packed PPP discard bits still suppress fragments. */
   const bool rasterization_enabled = hk_native_vertex_template_enabled(
      hk_rast_discard(cmd), hw_vs->info.writes_external_memory,
      hk_g17p_xfb_vertex_side_effects(cmd, hw_vs));

   /* Every encoder input except the pipeline address, which occupies only
    * the two-word pipeline pair at bytes 8..15 (agx_vdm_state_update_pack())
    * and is packed by the same function with the same template field: a
    * draw whose other inputs match the last encoding reuses it. */
   const struct hk_g17p_vdm_update_key key = {
      .hw_vs = hw_vs,
      .draw_path = draw_path,
      .rasterization_enabled = rasterization_enabled,
      .provoking = gfx->provoking,
      .primitive_id = gfx->generate_primitive_id,
      .ordinary = ordinary_draw_path,
      .selector_valid = cs->agx3_active_context_selector_valid,
      .selector = cs->agx3_raw_active_context_selector,
   };
   struct agx_vdm_state_update encoded;
   struct agx_vdm_state_vertex_pipeline_pair pair;
   const struct hk_g17p_vdm_update_key *memo = &gfx->g17p_vdm_update_memo.key;
   if (pipeline_va_valid && gfx->g17p_vdm_update_memo.valid &&
       memo->hw_vs == key.hw_vs && memo->selector == key.selector &&
       memo->draw_path == key.draw_path &&
       memo->provoking == key.provoking &&
       memo->rasterization_enabled == key.rasterization_enabled &&
       memo->primitive_id == key.primitive_id &&
       memo->ordinary == key.ordinary &&
       memo->selector_valid == key.selector_valid &&
       agx_vdm_state_vertex_pipeline_pair_pack(
          dev->dev.chip, pipeline_program,
          program->template_info.raw_state_load_count_18 << 3, &pair)) {
      static_assert(sizeof(pair.bytes) == 8, "pipeline pair is two words");
      encoded = gfx->g17p_vdm_update_memo.encoded;
      memcpy(&encoded.bytes[8], pair.bytes, sizeof(pair.bytes));
      memcpy(out, encoded.bytes, encoded.length);
      out += encoded.length;
      goto tail;
   }

   const bool complete_program =
      pipeline_va_valid &&
      hk_agx3_vdm_live_source_layout(
         program, cs->agx3_active_context_selector_valid,
         ordinary_draw_path) ==
         AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT;
   struct agx_vdm_state_update_draw_info info = {
      .source_layout = complete_program
            ? AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT
            : AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_UNSPECIFIED,
      .draw_path = draw_path,
      .rasterization_enabled = rasterization_enabled,
      /* hk already stores the actual hardware provoking vertex 0..2.
       * Apple's setter maps its API mode to this same value before
       * selecting one of the six templates. */
      .raw_provoking_selector = gfx->provoking,
      .template_info = program->template_info,
   };
   info.template_info.pipeline = pipeline_program;
   info.template_info.raw_vs_output_flag_6 = gfx->generate_primitive_id;
   info.template_info.raw_vs_output_count_1_extra =
      gfx->generate_primitive_id;
   info.template_info.raw_active_context_selector =
      cs->agx3_raw_active_context_selector;

   if (!hk_agx3_encode_vdm_state_update_draw(dev->dev.chip, &info,
                                             &encoded)) {
      const char *reason =
         !pipeline_va_valid ? "pipeline-va"
         : program->source_layout !=
              AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT
            ? "shader-layout"
         : !cs->agx3_active_context_selector_valid ? "active-context"
         : !ordinary_draw_path ? "transformed-draw"
                               : "template-fields";
      mesa_loge_once(
         "no complete AGX3 VertexProgramVariant state for %s: "
         "reason=%s pipeline=0x%" PRIx64
         " layout=%u active=%u/0x%08x ordinary=%u hw_eq_sw=%u "
         "draw=%u indexed=%u indirect=%u restart=%u raster=%u "
         "provoking=%u prim_id=%u topology=%u template="
         "%u/%u/%u/%u/%u/%u/%u/%u",
         agx_chip_name(dev->dev.chip), reason, pipeline_program,
         (unsigned)program->source_layout,
         cs->agx3_active_context_selector_valid,
         cs->agx3_raw_active_context_selector, ordinary_draw_path,
         hw_vs == hk_bound_sw_vs(gfx), (unsigned)info.draw_path, draw.indexed,
         agx_is_indirect(draw.b), draw.restart,
         info.rasterization_enabled, info.raw_provoking_selector,
         gfx->generate_primitive_id,
         (unsigned)dyn->ia.primitive_topology,
         info.template_info.raw_state_load_count_12,
         info.template_info.raw_state_load_count_18,
         info.template_info.raw_vs_output_count_0,
         info.template_info.raw_vs_output_count_2,
         info.template_info.raw_vs_output_count_1_extra,
         info.template_info.raw_vs_output_selector,
         info.template_info.raw_vs_output_flag_6,
         info.template_info.pre_hal300_flag_group_nonzero);
      vk_command_buffer_set_error(&cmd->vk,
                                  VK_ERROR_FEATURE_NOT_PRESENT);
      cs->current = out;
      return false;
   }

   /* Keep both words of the packed 48-bit vertex-pipeline pair.  The
    * Metal witness's 0x01000000/0x00004a00 pair decodes to its own
    * 0x10000128000 pipeline allocation; substituting only the second
    * word redirects Honeykrisp to an unrelated shader container.
    */

   memcpy(out, encoded.bytes, encoded.length);
   out += encoded.length;
   gfx->g17p_vdm_update_memo.key = key;
   gfx->g17p_vdm_update_memo.encoded = encoded;
   gfx->g17p_vdm_update_memo.valid = true;

tail:
   if (unlikely(dev->frame_anatomy)) {
      cs->anatomy_vdm++;
      if (pipeline_program != cs->anatomy_last_program)
         cs->anatomy_vdm_new++;
      cs->anatomy_last_program = pipeline_program;
   }
   if (agx_native_graphics(&dev->dev)) {
      if (!hk_g17p_emit_native_bind_tail(cmd, cs, &out,
                                           tail_known_match)) {
         mesa_loge("failed to emit native HAL200 bind tail for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         cs->current = out;
         return false;
      }
      *tail_emitted = true;

   }

   *out_p = out;
   return true;
}

static __attribute__((noinline)) void
hk_flush_dynamic_state_slow(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                            uint32_t draw_id, struct agx_draw draw,
                            bool ordinary_draw_path,
                            bool *g17p_native_bind_tail_emitted_out)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct hk_rendering_state *render = &cmd->state.gfx.render;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   struct hk_graphics_state *gfx = &cmd->state.gfx;

   struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);
   bool g17p_native_bind_tail_emitted = false;
   *g17p_native_bind_tail_emitted_out = false;

   /* The caller admits ordinary draws plus the bounded passthrough-GS XFB
    * route.  Reject every other software-transformed draw even if no state is
    * dirty, otherwise it could inherit a preceding draw's AGX3 state. */
   if (agx_chip_command_stream_generation(dev->dev.chip) !=
          AGX_CS_GENERATION_AGX2 &&
       !ordinary_draw_path) {
      mesa_loge("no grounded AGX3 VertexProgramVariant state for transformed "
                "draw path on %s",
                agx_chip_name(dev->dev.chip));
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }

   if (!hk_gfx_republishes_render_state(cmd))
      return;
   if (unlikely(dev->frame_anatomy))
      cs->anatomy_republish++;

   struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

   assert(cs->current + 0x1000 < cs->end && "already ensured space");
   uint8_t *out = cs->current;

   struct hk_shader *fs = hk_bound_fs(cmd);
   bool gt_dirty = IS_SHADER_DIRTY(TESS_CTRL) || IS_SHADER_DIRTY(TESS_EVAL) ||
                   IS_SHADER_DIRTY(GEOMETRY);
   const bool native_mesh = gfx->shaders[MESA_SHADER_MESH] != NULL;
   bool vgt_dirty = IS_SHADER_DIRTY(VERTEX) || IS_SHADER_DIRTY(MESH) || gt_dirty;
   bool fs_dirty = IS_SHADER_DIRTY(FRAGMENT);
   if (native_mesh && gfx->linked[MESA_SHADER_VERTEX] != hw_vs->only_linked) {
      gfx->fast_link_vs.valid = false;
      gfx->linked[MESA_SHADER_VERTEX] = hw_vs->only_linked;
      gfx->linked_dirty |= BITFIELD_BIT(MESA_SHADER_VERTEX);
   }
   if (IS_DIRTY(CB_BLEND_CONSTANTS) &&
       memcmp(desc->root.draw.blend_constant, dyn->cb.blend_constants,
              sizeof(dyn->cb.blend_constants))) {
      static_assert(sizeof(desc->root.draw.blend_constant) ==
                       sizeof(dyn->cb.blend_constants),
                    "common size");

      memcpy(desc->root.draw.blend_constant, dyn->cb.blend_constants,
             sizeof(dyn->cb.blend_constants));
      desc->root_dirty = true;
   }

   if (IS_DIRTY(MS_SAMPLE_MASK)) {
      const uint16_t sample_mask = dyn->ms.sample_mask;
      if (desc->root.draw.api_sample_mask != sample_mask) {
         desc->root.draw.api_sample_mask = sample_mask;
         desc->root_dirty = true;
      }
   }

   if (IS_DIRTY(RS_RASTERIZATION_STREAM)) {
      const uint16_t stream = dyn->rs.rasterization_stream;
      if (desc->root.draw.rasterization_stream != stream) {
         desc->root.draw.rasterization_stream = stream;
         desc->root_dirty = true;
      }

      BITSET_SET(dyn->dirty, MESA_VK_DYNAMIC_RS_RASTERIZER_DISCARD_ENABLE);
   }

   if (fs_dirty || IS_DIRTY(DS_DEPTH_TEST_ENABLE) ||
       IS_DIRTY(DS_DEPTH_COMPARE_OP)) {

      const struct hk_rendering_state *render = &cmd->state.gfx.render;
      bool has_z = render->depth_att.vk_format != VK_FORMAT_UNDEFINED;
      bool z_test = has_z && dyn->ds.depth.test_enable;

      const uint16_t force_never_in_shader =
         z_test && dyn->ds.depth.compare_op == VK_COMPARE_OP_NEVER && fs &&
         fs->info.fs.writes_memory;

      if (desc->root.draw.force_never_in_shader != force_never_in_shader) {
         desc->root.draw.force_never_in_shader = force_never_in_shader;
         desc->root_dirty = true;
      }
   }

   /* The main shader must not run tests if another linked part will. */
   bool nontrivial_force_early =
      fs && (fs->b.info.early_fragment_tests &&
             (fs->b.info.writes_sample_mask || fs->info.fs.writes_memory));

   bool epilog_discards = agx_fs_epilog_may_discard(
      fs ? &fs->info.fs.epilog_key : NULL,
      dyn->ms.alpha_to_coverage_enable, nontrivial_force_early);

   const unsigned early_reject_samples =
      MAX2(dyn->ms.rasterization_samples, 1);
   const unsigned early_reject_full_mask =
      BITFIELD_MASK(dyn->ms.rasterization_samples);
   const bool early_reject_has_api_mask =
      (dyn->ms.sample_mask & early_reject_full_mask) != early_reject_full_mask;
   const bool early_reject_defer_api_mask =
      agx_native_graphics(&dev->dev) && early_reject_has_api_mask &&
      !nontrivial_force_early;
   const bool early_reject_prolog_discards =
      (early_reject_has_api_mask && !early_reject_defer_api_mask) ||
      hw_vs->info.cull_distance_array_size;
   const bool early_reject_statistics = hk_stat_enabled(hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT));
   const bool read_only_early_reject =
      fs && hk_g17p_read_only_early_reject(
               cmd, fs, early_reject_samples, early_reject_prolog_discards,
               early_reject_statistics);

   if (fs_dirty || IS_DIRTY(MS_ALPHA_TO_COVERAGE_ENABLE) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES) || IS_DIRTY(MS_SAMPLE_MASK) ||
       IS_DIRTY(DS_DEPTH_TEST_ENABLE) || IS_DIRTY(DS_DEPTH_WRITE_ENABLE) ||
       IS_DIRTY(DS_DEPTH_COMPARE_OP) ||
       IS_DIRTY(DS_DEPTH_BOUNDS_TEST_ENABLE) ||
       IS_DIRTY(DS_STENCIL_TEST_ENABLE) ||
       (gfx->dirty & HK_DIRTY_OCCLUSION) || desc->root_dirty) {
      const uint16_t no_epilog_discard =
         !epilog_discards && !read_only_early_reject ? UINT16_MAX : 0;
      /* A main part compiled with the selector folded (AGX3 static state)
       * always tests; nothing else may then own the tests. */
      assert(!(fs && fs->info.fs.static_tests && !no_epilog_discard) &&
             "folded depth/stencil test selector disagrees with the draw");
      const uint16_t alpha_to_coverage = dyn->ms.alpha_to_coverage_enable;
      if (desc->root.draw.no_epilog_discard != no_epilog_discard ||
          desc->root.draw.alpha_to_coverage != alpha_to_coverage) {
         desc->root.draw.no_epilog_discard = no_epilog_discard;
         desc->root.draw.alpha_to_coverage = alpha_to_coverage;
         desc->root_dirty = true;
      }
   }

   if (!native_mesh && (IS_DIRTY(VI) || IS_DIRTY(VI_BINDINGS_VALID) ||
        IS_DIRTY(VI_BINDING_STRIDES) || IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) ||
        vgt_dirty || !gfx->fast_link_vs.valid)) {
      struct hk_fast_link_key_vs key;
      hk_fill_fast_link_key_vs(cmd, sw_vs, hw_vs, draw.indexed,
                               draw.index_size, draw.restart, &key);

      if (!hk_update_fast_linked(cmd, sw_vs, &key)) {
         cs->current = out;
         return;
      }

   }

   if (!native_mesh && (IS_DIRTY(VI) || IS_DIRTY(VI_BINDINGS_VALID) ||
       IS_DIRTY(VI_BINDING_STRIDES) || vgt_dirty ||
       (gfx->dirty & HK_DIRTY_VB))) {

      unsigned slot = 0;
      bool vertex_root_changed = false;
      u_foreach_bit(a, sw_vs->info.vs.attribs_read) {
         uint64_t base;
         uint32_t clamp;
         uint32_t stride;
         if (dyn->vi->attributes_valid & BITFIELD_BIT(a)) {
            struct vk_vertex_attribute_state attr = dyn->vi->attributes[a];
            struct hk_addr_range vb = gfx->vb[attr.binding];
            __typeof__(gfx->vb_format_memo[0]) *memo =
               &gfx->vb_format_memo[slot];
            if (memo->vk_format != attr.format || !memo->interchange_align) {
               memo->vk_format = attr.format;
               memo->format = hk_format_to_pipe_format(attr.format);
               memo->interchange_align = util_format_get_blocksize(
                  agx_vbo_internal_format(memo->format));
            }
            enum pipe_format fmt = memo->format;
            unsigned interchange_align = memo->interchange_align;

            clamp = agx_calculate_vbo_clamp(
               vb.addr, fmt, vb.range, dyn->vi_binding_strides[attr.binding],
               attr.offset, &base);
            stride =
               dyn->vi_binding_strides[attr.binding] / interchange_align;
         } else {
            base = AGX_ZERO_PAGE_ADDRESS;
            clamp = 0;
            stride = 0;
         }

         vertex_root_changed |= desc->root.draw.attrib_base[slot] != base ||
                                desc->root.draw.attrib_clamps[slot] != clamp ||
                                desc->root.draw.attrib_strides[slot] != stride;
         desc->root.draw.attrib_base[slot] = base;
         desc->root.draw.attrib_clamps[slot] = clamp;
         desc->root.draw.attrib_strides[slot] = stride;

         ++slot;
      }

      desc->root_dirty |= vertex_root_changed;
      gfx->root_vertex_buffer_generation = gfx->vertex_buffer_generation;
   }

   if (vgt_dirty || IS_SHADER_DIRTY(FRAGMENT) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES) || IS_DIRTY(MS_SAMPLE_MASK) ||
       IS_DIRTY(MS_ALPHA_TO_COVERAGE_ENABLE) ||
       IS_DIRTY(MS_ALPHA_TO_ONE_ENABLE) || IS_DIRTY(CB_LOGIC_OP) ||
       IS_DIRTY(CB_LOGIC_OP_ENABLE) || IS_DIRTY(CB_WRITE_MASKS) ||
       IS_DIRTY(CB_COLOR_WRITE_ENABLES) || IS_DIRTY(CB_ATTACHMENT_COUNT) ||
       IS_DIRTY(CB_BLEND_ENABLES) || IS_DIRTY(CB_BLEND_EQUATIONS) ||
       IS_DIRTY(CB_BLEND_CONSTANTS) || IS_DIRTY(CB_BLEND_ADVANCED) ||
       IS_DIRTY(DS_DEPTH_TEST_ENABLE) || IS_DIRTY(DS_DEPTH_WRITE_ENABLE) ||
       IS_DIRTY(DS_DEPTH_COMPARE_OP) ||
       IS_DIRTY(DS_DEPTH_BOUNDS_TEST_ENABLE) ||
       IS_DIRTY(DS_STENCIL_TEST_ENABLE) ||
       (gfx->dirty & HK_DIRTY_OCCLUSION) ||
       IS_DIRTY(COLOR_ATTACHMENT_MAP) ||
       /* The only root-derived input of the fragment key is whether the
        * fragment-invocation statistic is enabled. Other root changes (a
        * descriptor bind, push constants, vertex buffers) leave the key and
        * so the linked fragment program unchanged; rebuilding it for every
        * such draw dominated root-only state changes. */
       hk_stat_enabled(hk_pipeline_stat_addr(
          cmd, VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT)) !=
          gfx->fast_link_fs.key.prolog.statistics ||
       !gfx->fast_link_fs.valid) {

      struct hk_fast_link_key_fs key;
      if (!hk_fill_fast_link_key_fs(cmd, fs, hw_vs, &key, true)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         cs->current = out;
         return;
      }

      if (!hk_update_fast_linked(cmd, fs, &key)) {
         cs->current = out;
         return;
      }
      gfx->fast_link_fs.color_written =
         hk_fs_color_written(&key, agx_chip_g17_m5(dev->dev.chip));
      gfx->fast_link_fs.reads_tib = fs->b.info.reads_tib;
   }
   gfx->render.color_written |= gfx->fast_link_fs.color_written;
   gfx->render.color_read |= gfx->fast_link_fs.reads_tib;

   /* Direct draws use an immutable parameter block, reused while the values
    * match. Indirect draws always retain the execution-time GPU pointer. */
   if (gfx->linked[MESA_SHADER_VERTEX]->b.uses_base_param) {
      if (agx_is_indirect(draw.b)) {
         gfx->draw_params = draw.b.ptr;
         gfx->g17p_draw_params_host_valid = false;

         if (draw.indexed) {
            gfx->draw_params +=
               offsetof(VkDrawIndexedIndirectCommand, vertexOffset);
         } else {
            gfx->draw_params += offsetof(VkDrawIndirectCommand, firstVertex);
         }
      } else {
         hk_cmd_upload_draw_params(cmd,
                                   draw.indexed ? draw.index_bias : draw.start,
                                   draw.start_instance);
      }
   } else {
      gfx->draw_params = 0;
      gfx->g17p_draw_params_host_valid = false;
   }

   if (sw_vs->b.info.uses_draw_id) {
      hk_cmd_upload_draw_id(cmd, draw_id);
   } else {
      gfx->draw_id_ptr = 0;
      gfx->g17p_draw_id_host_valid = false;
   }

   bool topology_changed = false;
   if (IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) || gt_dirty || vgt_dirty || IS_SHADER_DIRTY(MESH)) {
      enum mesa_prim prim = native_mesh ? hw_vs->info.mesh.prim : hk_rast_prim(cmd);
      const unsigned topology = translate_hw_primitive_topology(prim);

      topology_changed = topology != gfx->topology && IS_SHADER_DIRTY(MESH);
      gfx->topology = topology;
      gfx->object_type = translate_object_type(prim);
   }

   /* The selector depends on the topology, which binding or unbinding a
    * mesh shader changes without dirtying the IA state. */
   if (IS_DIRTY(IA_PRIMITIVE_TOPOLOGY) || IS_DIRTY(RS_PROVOKING_VERTEX) ||
       topology_changed) {
      unsigned provoking = hk_provoking_selector(dyn, gfx->topology);

      if (provoking != gfx->provoking) {
         gfx->provoking = provoking;
         gfx->dirty |= HK_DIRTY_PROVOKING;

         gfx->descriptors.root.draw.provoking = provoking;
         gfx->descriptors.root_dirty = true;
      }
   }

   /* With attachmentless rendering, we don't know the sample count until draw
    * time, so we do a late tilebuffer fix up. But with rasterizer discard,
    * rasterization_samples might be 0.
    *
    * Note that we ignore dyn->ms.rasterization_samples when we do have a sample
    * count from an attachment. In Vulkan, these have to match anyway, but DX12
    * drivers are robust against this scenarios and vkd3d-proton will go out of
    * spec here. No reason we can't be robust here too.
    */
   if (dyn->ms.rasterization_samples && !gfx->render.tilebuffer.nr_samples) {
      agx_tilebuffer_set_samples(&gfx->render.tilebuffer,
                                 dyn->ms.rasterization_samples);

      /* The provisional attachmentless layout was admitted at begin time;
       * enforce the hardware contract again once the sample count is known.
       */
      if (agx_native_graphics(&dev->dev) &&
          !(gfx->render.g17p_rt0_layout_supported =
               hk_agx3_g17p_rt0_tile_layout_supported(
                  &gfx->render.tilebuffer))) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      cs->tib = gfx->render.tilebuffer;
   }

   if (IS_DIRTY(MS_SAMPLE_LOCATIONS) || IS_DIRTY(MS_SAMPLE_LOCATIONS_ENABLE) ||
       IS_DIRTY(MS_RASTERIZATION_SAMPLES)) {

      uint64_t ctrl;
      if (dyn->ms.sample_locations_enable) {
         ctrl = hk_pack_ppp_multisamplectrl(dyn->ms.sample_locations);
      } else {
         ctrl = hk_default_sample_positions(dyn->ms.rasterization_samples);
      }

      bool dont_commit = cmd->in_meta || dyn->ms.rasterization_samples == 0;

      if ( !cs->has_sample_locations) {
         cs->ppp_multisamplectl = ctrl;

         /* If we're in vk_meta, do not commit to the sample locations yet.
          * vk_meta doesn't care, but the app will!
          */
         cs->has_sample_locations |= !dont_commit;
      } else {
         assert(dont_commit || cs->ppp_multisamplectl == ctrl);
      }

      if (gfx->descriptors.root.draw.ppp_multisamplectl != (uint32_t)ctrl ||
          gfx->descriptors.root.ppp_multisamplectl_hi != (ctrl >> 32)) {
         gfx->descriptors.root.draw.ppp_multisamplectl = (uint32_t)ctrl;
         gfx->descriptors.root.ppp_multisamplectl_hi = ctrl >> 32;
         gfx->descriptors.root_dirty = true;
      }
   }

   /* Link varyings before uploading tessellation state, becuase the
    * gfx->generate_primitive_id boolean needs to be plumbed.
    */
   /* Fast-link updates above may replace either object.  Never carry the
    * admission snapshot into the authoritative path: doing so was the r246
    * correctness regression and made every later consumer see stale linkage. */
   struct hk_linked_shader *linked_vs = gfx->linked[MESA_SHADER_VERTEX];
   struct hk_linked_shader *linked_fs = gfx->linked[MESA_SHADER_FRAGMENT];
   bool linked_vs_dirty = IS_LINKED_DIRTY(VERTEX);
   bool linked_fs_dirty = IS_LINKED_DIRTY(FRAGMENT);
   const bool prepared_varyings =
      agx_native_graphics(&dev->dev) &&
      hk_g17p_prepared_bind_tail_matches(cmd, &gfx->g17p_bind_tail);
   /* Between here and the bind-tail emission below only the root can move
    * one of the tail's inputs (varyings are relinked only when this is
    * false), so the emission reuses this answer while the root is unchanged
    * instead of comparing the whole snapshot again. */
   const uint64_t prepared_varyings_root = gfx->root;

   if (!prepared_varyings &&
       ((gfx->dirty & HK_DIRTY_PROVOKING) || vgt_dirty || linked_fs_dirty)) {
      unsigned bindings = linked_fs->b.cf.nr_bindings;
      if (bindings) {
         size_t linkage_size =
            AGX_CF_BINDING_HEADER_LENGTH + (bindings * AGX_CF_BINDING_LENGTH);

         const bool native_vit = agx_native_graphics(&dev->dev) ||
                                 agx_chip_g17_m5(dev->dev.chip);
         struct agx_ptr t =
            hk_pool_usc_alloc(cmd, linkage_size, native_vit ? 64 : 16);
         if (!t.cpu)
            return;

         agx_link_varyings_vs_fs(
            t.cpu, &gfx->linked_varyings, hw_vs->info.uvs.user_size,
            &linked_fs->b.cf, gfx->provoking, 0, &gfx->generate_primitive_id);

         gfx->varyings = native_vit ? t.gpu : agx_usc_addr(&dev->dev, t.gpu);
      } else {
         gfx->varyings = 0;
      }

      gfx->dirty |= HK_DIRTY_VARYINGS;
   }

   if (
       (gfx->shaders[MESA_SHADER_TESS_EVAL] ||
        gfx->shaders[MESA_SHADER_GEOMETRY] || linked_vs->sw_indexing)) {
      const uint64_t old_outputs = desc->root.draw.vertex_outputs;
      const uint64_t vertex_params = hk_upload_vertex_params(cmd, draw);
      if (desc->root.draw.vertex_params != vertex_params ||
          desc->root.draw.vertex_outputs != old_outputs) {
         desc->root.draw.vertex_params = vertex_params;
         desc->root_dirty = true;
      }
   }

   struct agx_ptr tess_args = {0};
   if (gfx->shaders[MESA_SHADER_TESS_EVAL]) {
      tess_args = hk_pool_alloc(cmd, sizeof(struct poly_tess_params), 4);
      if (gfx->descriptors.root.draw.tess_params != tess_args.gpu) {
         gfx->descriptors.root.draw.tess_params = tess_args.gpu;
         gfx->descriptors.root_dirty = true;
      }
   }

   if (gfx->shaders[MESA_SHADER_GEOMETRY]) {
      const uint64_t geometry_params = hk_upload_geometry_params(cmd, draw);
      if (gfx->descriptors.root.draw.geometry_params != geometry_params) {
         gfx->descriptors.root.draw.geometry_params = geometry_params;
         gfx->descriptors.root_dirty = true;
      }
   }

   /* Root must be uploaded after the above, which touch the root. */
   if (gfx->descriptors.root_dirty) {
      gfx->root =
         hk_cmd_buffer_upload_root(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS);
   }

   /* The root owns only the stable pointer to this block.  Its contents are
    * per-draw state and must be populated even when the root itself is already
    * complete.  In particular, an indirect draw writes the GPU addresses of
    * the three execution-time grid records here; tying this fill to a root
    * re-upload leaves those addresses zero on the prepared-root path.
    *
    * Tess parameters consume the final root address, so allocate their stable
    * storage before root publication and fill it only after publication. */
   if (tess_args.cpu) {
      assert(gfx->root);
      hk_upload_tess_params(cmd, tess_args.cpu, draw);
   }

   /* Hardware dynamic state must be deferred until after the root and fast
    * linking, since it will use the root address and the linked shaders.
    */
   /* Any dirty G17P draw must publish HAL200 state. In particular, stencil,
    * depth and line-width changes do not necessarily dirty the VS or root.
    * Letting those changes fall through to hk_flush_ppp_state() emits the
    * legacy aggregate layout into an otherwise native stream. The initial
    * draw hid this because all shader state was dirty on stream creation. */
   if (agx_native_graphics(&dev->dev) ||
      (gfx->dirty & (HK_DIRTY_PROVOKING | HK_DIRTY_VARYINGS)) ||
      IS_DIRTY(RS_RASTERIZER_DISCARD_ENABLE) || linked_vs_dirty || vgt_dirty ||
      gfx->descriptors.root_dirty || gfx->draw_id_ptr || gfx->draw_params ||
      IS_DIRTY(RS_DEPTH_BIAS_ENABLE) || IS_DIRTY(RS_DEPTH_BIAS_FACTORS) ||
      IS_DIRTY(VP_SCISSORS) || IS_DIRTY(VP_SCISSOR_COUNT) ||
      IS_DIRTY(VP_VIEWPORTS) || IS_DIRTY(VP_VIEWPORT_COUNT) ||
      IS_DIRTY(VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) ||
      IS_DIRTY(RS_DEPTH_CLIP_ENABLE) || IS_DIRTY(RS_DEPTH_CLAMP_ENABLE)) {

      /* TODO: Do less often? */
      hk_reserve_scratch(cmd, cs, hw_vs);

      struct hk_linked_shader *linked_hw_vs =
         (hw_vs == sw_vs) ? linked_vs : hw_vs->only_linked;
      const uint64_t pipeline_program = native_mesh
         ? hk_upload_native_mesh_program(cmd, hw_vs, linked_hw_vs, false, false)
         : hk_upload_usc_words(cmd, hw_vs, linked_hw_vs);
      if (native_mesh) {
         bool task;
         struct hk_shader *object = hk_native_mesh_object(gfx, &task);
         const uint64_t object_program = hk_upload_native_mesh_program(
            cmd, object, object->only_linked, true, task);
         uint32_t mesh_dimensions[3] = {hw_vs->b.info.workgroup_size[0],
            hw_vs->b.info.workgroup_size[1],hw_vs->b.info.workgroup_size[2]};
         uint32_t mesh_threadgroup;
         if (!agx_vdm_mesh_threadgroup_pack(mesh_dimensions,&mesh_threadgroup)) {
            vk_command_buffer_set_error(&cmd->vk,VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         if (dev->dev.chip == AGX_CHIP_G17P &&
             (!hw_vs->info.mesh.max_vertices ||
              hw_vs->info.mesh.max_vertices > 256 ||
              !hw_vs->info.mesh.max_primitives ||
              hw_vs->info.mesh.max_primitives > 256 ||
              hw_vs->b.info.local_size > 32768 ||
              hw_vs->b.info.has_preamble || object->b.info.has_preamble)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         const bool native_raster_enabled =
            debug_get_option_mesh_native_raster_enable();
         struct agx_vdm_mesh_state_info state = {
            


            .raw_prefix = {dev->dev.chip == AGX_CHIP_G17P
                           ? hw_vs->info.uvs.size |
                             ((hw_vs->info.uvs.size + 1) << 8)
                           : 0x500u|hw_vs->info.uvs.size,
                           dev->dev.chip == AGX_CHIP_G17P
                              ? (native_raster_enabled && !hk_rast_discard(cmd)
                                 ? 0x10 : 0x30) |
                                (hw_vs->info.mesh.primitive_stride_B ? 0x80 : 0)
                              : 0, 0, 0},
            .object_pipeline = object_program,
            .mesh_pipeline = pipeline_program,
            .raw_object_template = 0,
            

            .raw_mesh_template = dev->dev.chip == AGX_CHIP_G17P ? 0 : 0x40,
            



            .raw_tail = {0,
               dev->dev.chip == AGX_CHIP_G17P
                  ? ((hw_vs->info.mesh.max_primitives - 1) & 0x1ff) |
                    (((hw_vs->info.mesh.max_primitives - 1) >> 4) << 16)
                  : 1,
               mesh_threadgroup,
               dev->dev.chip == AGX_CHIP_G17P
                  ? DIV_ROUND_UP(hw_vs->info.mesh.primitive_stride_B *
                                 hw_vs->info.mesh.max_primitives, 4) : 0,
               dev->dev.chip == AGX_CHIP_G17P
                  ? DIV_ROUND_UP(hw_vs->b.info.local_size, 16) : 9},
         };
         if (dev->dev.chip == AGX_CHIP_G17S) {
            if (!agx_vdm_m5_mesh_set_output_limits(
                   &state, hw_vs->info.mesh.max_vertices,
                   hw_vs->info.mesh.max_primitives, hw_vs->info.uvs.size)) {
               vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }
            /* Native object->mesh kicks (native726 mesh-object, 0x1ca48
             * bound) set tail[0]; task kicks with the native record do too.
             * Payload-free task kicks without it keep the mesh-only value. */
            if ((task && hk_m5_task_record(dev->dev.params.features)) ||
                debug_get_option_mesh_pp_tail0())
               state.raw_tail[0] = 1;
         }
         if (!agx_vdm_mesh_state_pack(dev->dev.chip, &state, out)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         out += AGX_VDM_MESH_STATE_LENGTH;
         if (dev->dev.chip == AGX_CHIP_G17P) {
            if (!hk_g17p_emit_native_bind_tail(cmd, cs, &out, false)) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               cs->current = out;
               return;
            }
            g17p_native_bind_tail_emitted = true;
            *g17p_native_bind_tail_emitted_out = true;
         }
      } else switch (agx_chip_command_stream_generation(dev->dev.chip)) {
      case AGX_CS_GENERATION_AGX2:
         if (pipeline_program > UINT32_MAX) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            cs->current = out;
            return;
         }

         agx_push(out, VDM_STATE, cfg) {
            cfg.vertex_shader_word_0_present = true;
            cfg.vertex_shader_word_1_present = true;
            cfg.vertex_outputs_present = true;
            cfg.vertex_unknown_present = true;
         }

         agx_push_packed(out, hw_vs->counts, VDM_STATE_VERTEX_SHADER_WORD_0);

         agx_push(out, VDM_STATE_VERTEX_SHADER_WORD_1, cfg) {
            cfg.pipeline = pipeline_program;
         }

         agx_push_packed(out, hw_vs->info.uvs.vdm,
                         VDM_STATE_VERTEX_OUTPUTS);

         agx_push(out, VDM_STATE_VERTEX_UNKNOWN, cfg) {
            cfg.flat_shading_control = translate_vdm_vertex(gfx->provoking);
            cfg.unknown_4 = cfg.unknown_5 = hk_rast_discard(cmd);
            cfg.generate_primitive_id = gfx->generate_primitive_id;
         }

         /* Pad up to a multiple of 8 bytes */
         memset(out, 0, 4);
         out += 4;
         break;

      case AGX_CS_GENERATION_G15:
      case AGX_CS_GENERATION_HAL200:
      case AGX_CS_GENERATION_HAL300: {
         bool tail_emitted = false;
         if (!hk_agx3_publish_vertex_state(
                cmd, cs, &out, hw_vs, pipeline_program, draw,
                ordinary_draw_path,
                prepared_varyings && gfx->root == prepared_varyings_root,
                &tail_emitted))
            return;
         if (tail_emitted) {
            g17p_native_bind_tail_emitted = true;
            *g17p_native_bind_tail_emitted_out = true;
         }
         break;
      }

      default:
         mesa_loge("unknown command-stream generation for %s",
                   agx_chip_name(dev->dev.chip));
         vk_command_buffer_set_error(&cmd->vk,
                                     VK_ERROR_FEATURE_NOT_PRESENT);
         cs->current = out;
         return;
      }
   }

   /* A pure cull/winding update needs new immutable fragment state, but
    * leaves the vertex program and its VDM state unchanged. */
   if (agx_native_graphics(&dev->dev) && !g17p_native_bind_tail_emitted &&
       (IS_DIRTY(RS_CULL_MODE) || IS_DIRTY(RS_FRONT_FACE))) {
      if (!hk_g17p_emit_native_bind_tail(cmd, cs, &out, false)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         cs->current = out;
         return;
      }
      g17p_native_bind_tail_emitted = true;
      *g17p_native_bind_tail_emitted_out = true;
   }

   if (!g17p_native_bind_tail_emitted && IS_DIRTY(RS_DEPTH_BIAS_FACTORS)) {
      bool exact = dyn->rs.depth_bias.exact;
      bool force_unorm =
         dyn->rs.depth_bias.representation ==
         VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT;

      const float slope_scale = dyn->rs.depth_bias.slope_factor;
      const float clamp = dyn->rs.depth_bias.clamp;
      const float depth_bias =
         dyn->rs.depth_bias.constant_factor /
         hk_depth_bias_factor(render->depth_att.vk_format, exact, force_unorm);
      uint8_t record[AGX_DEPTH_BIAS_LENGTH];

      static_assert(AGX_DEPTH_BIAS_LENGTH ==
                    AGX_PPP_HAL300_DEPTH_BIAS_RECORD_LENGTH);
      if (agx_chip_command_stream_generation(dev->dev.chip) ==
          AGX_CS_GENERATION_HAL300) {
         const struct agx_ppp_hal300_depth_bias_record_fields fields = {
            .depth_bias_raw = fui(depth_bias),
            .slope_scale_raw = fui(slope_scale),
            .clamp_raw = fui(clamp),
         };
         struct agx_ppp_hal300_depth_bias_record exact_record;
         if (!agx_ppp_hal300_depth_bias_record_pack(
                dev->dev.chip, &fields, &exact_record)) {
            mesa_loge("no grounded HAL300 depth-bias record for %s",
                      agx_chip_name(dev->dev.chip));
            vk_command_buffer_set_error(&cmd->vk,
                                        VK_ERROR_FEATURE_NOT_PRESENT);
            cs->current = out;
            return;
         }
         memcpy(record, exact_record.bytes, exact_record.length);
      } else {
         agx_pack(record, DEPTH_BIAS, cfg) {
            cfg.depth_bias = depth_bias;
            cfg.slope_scale = slope_scale;
            cfg.clamp = clamp;
         }
      }

      void *ptr =
         util_dynarray_grow_bytes(&cs->depth_bias, 1, sizeof(record));
      memcpy(ptr, record, sizeof(record));
   }

   /* Hardware viewport/scissor state is entangled with depth bias. */
   if (!g17p_native_bind_tail_emitted &&
       (IS_DIRTY(RS_DEPTH_BIAS_FACTORS) || IS_DIRTY(VP_SCISSORS) ||
       IS_DIRTY(VP_SCISSOR_COUNT) || IS_DIRTY(VP_VIEWPORTS) ||
       IS_DIRTY(VP_VIEWPORT_COUNT) ||
       IS_DIRTY(VP_DEPTH_CLIP_NEGATIVE_ONE_TO_ONE) ||
       IS_DIRTY(RS_DEPTH_CLIP_ENABLE) || IS_DIRTY(RS_DEPTH_CLAMP_ENABLE))) {

      hk_flush_vp_state(cmd, cs, &out);
      if (vk_command_buffer_has_error(&cmd->vk)) {
         cs->current = out;
         return;
      }
   }

   if (!g17p_native_bind_tail_emitted)
      hk_flush_ppp_state(cmd, cs, &out);

   cs->current = out;

   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   vk_dynamic_graphics_state_clear_dirty(dyn);
   gfx->shaders_dirty = 0;
   gfx->linked_dirty = 0;
   gfx->dirty = 0;
   gfx->descriptors.root_dirty = false;

}

static __attribute__((hot)) void
hk_flush_dynamic_state(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       uint32_t draw_id, struct agx_draw draw,
                       bool ordinary_draw_path,
                       bool *g17p_native_bind_tail_emitted_out)
{
   hk_flush_dynamic_state_slow(
      cmd, cs, draw_id, draw, ordinary_draw_path,
      g17p_native_bind_tail_emitted_out);
}

static bool
hk_needs_index_robustness(struct hk_cmd_buffer *cmd, struct agx_draw *draw,
                          const struct agx_draw *conditional_direct)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   if (!draw->indexed)
      return false;

   /* Geometry or tessellation use robust software index buffer fetch anyway */
   if (gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL])
      return false;

   /* Soft fault does not cover the hardware index buffer fetch, use the zero
    * page instead.
    */
   if (draw->index_buffer_range_B == 0 ||
       ((agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
        draw->index_buffer == AGX_ZERO_PAGE_ADDRESS &&
        draw->index_buffer_range_B == 4)) {
      draw->index_buffer = AGX_ZERO_PAGE_ADDRESS;
      draw->index_buffer_range_B = 4;
      draw->start = 0;
      return false;
   }

   if (!(dev->vk.enabled_features.robustBufferAccess ||
         dev->vk.enabled_features.robustBufferAccess2 ||
         dev->vk.enabled_features.pipelineRobustness))
      return false;

   /* A conditionally lowered direct draw has an immutable indirect record:
    * apply the direct draw's exact range test instead of conservatively
    * treating it as application-authored indirect data. */
   if (agx_is_indirect(draw->b))
      return conditional_direct
                ? agx_direct_draw_overreads_indices(*conditional_direct)
                : true;

   return agx_direct_draw_overreads_indices(*draw);
}

static void
hk_nir_passthrough_gs_native(struct nir_builder *b, const void *key)
{
   poly_nir_passthrough_gs(b, key);
   b->shader->info.internal = false;
}

static void
hk_handle_passthrough_gs(struct hk_cmd_buffer *cmd, struct agx_draw draw)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *gs = gfx->shaders[MESA_SHADER_GEOMETRY];

   /* The TES decides as well, and the primitive the passthrough consumes
    * comes from the merged TCS/TES state. */
   if (!IS_SHADER_DIRTY(VERTEX) && !IS_SHADER_DIRTY(GEOMETRY) &&
       (hk_cmd_buffer_device(cmd)->dev.chip != AGX_CHIP_G17P ||
        (!IS_SHADER_DIRTY(TESS_CTRL) && !IS_SHADER_DIRTY(TESS_EVAL))))
      return;

   /* If there's an application geometry shader, there's nothing to un/bind */
   if (gs && !gs->is_passthrough)
      return;

   /* Determine if we need a geometry shader to emulate XFB, or to rasterize
    * a TES whose native ray queries run in its compute variant (the queries
    * and the application's stores then execute before rasterization, and
    * the passthrough's raster program replays outputs only). */
   struct hk_shader *last_sw = hk_bound_sw_vs_before_gs(gfx);
   uint32_t xfb_outputs = last_sw->info.xfb_info.output_count;
   const bool tes_rq = gfx->shaders[MESA_SHADER_TESS_EVAL] &&
                       last_sw->compiled &&
                       last_sw->info.agx3_gen1_fragment_rq &&
                       hk_tes_ray_queries_in_compute(
                          hk_cmd_buffer_device(cmd));
   bool needs_gs = xfb_outputs || tes_rq;

   /* If we don't need a GS but we do have a passthrough, unbind it */
   if (!needs_gs) {
      if (gs != NULL) {
         assert(gs->is_passthrough);
         hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_GEOMETRY, NULL);
      }
      return;
   }

   /* Else, we need to bind a passthrough GS */
   size_t key_size = poly_passthrough_gs_key_size(xfb_outputs);
   struct poly_passthrough_gs_key *key = alloca(key_size);

   *key = (struct poly_passthrough_gs_key){
      .prim = u_decomposed_prim(hk_gs_in_prim(cmd)),
      .outputs = last_sw->b.info.outputs,
      .clip_distance_array_size = last_sw->info.clip_distance_array_size,
      .cull_distance_array_size = last_sw->info.cull_distance_array_size,
   };

   for (uint32_t i = 0; i < NUM_TOTAL_VARYING_SLOTS; i++) {
      unsigned num_comps = 0;
      switch (i) {
         case VARYING_SLOT_TESS_LEVEL_OUTER:
         case VARYING_SLOT_TESS_LEVEL_INNER:
         case VARYING_SLOT_BOUNDING_BOX1:
            num_comps = 4;
            break;
         default:
            num_comps = nir_slot_num_components(i, MESA_SHADER_VERTEX);
            break;
      }
      key->output_components[i] = num_comps ?: 4;

      /* We don't care about VS varying types in AGX, just set everything to
       * uint32 to improve cache hits. */
      key->output_types[i] = nir_type_uint32;
   }

   if (xfb_outputs) {
      typed_memcpy(key->xfb_stride, last_sw->info.xfb_stride,
                   ARRAY_SIZE(key->xfb_stride));

      memcpy(&key->xfb_info, &last_sw->info.xfb_info,
             nir_xfb_info_size(xfb_outputs));
   }

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   perf_debug(cmd, "Binding passthrough GS for%s\n", xfb_outputs ? " XFB" : "");

   VkResult result;
   gs = hk_meta_shader(dev, hk_nir_passthrough_gs_native, key, key_size,
                       &result);
   if (!gs) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   gs->is_passthrough = true;
   hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_GEOMETRY, gs);
}

static __attribute__((hot)) struct hk_cs *
hk_flush_gfx_state(struct hk_cmd_buffer *cmd, uint32_t draw_id,
                   struct agx_draw *draw, bool *needs_idx_robust,
                   bool *g17p_native_bind_tail_emitted,
                   const struct agx_draw *conditional_direct)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;

   /* Every stage hk_flush_shaders() reads has to actually be bound.  Binding
    * VK_NULL_HANDLE for a stage is how CmdBindShadersEXT *unbinds* it, and it
    * is also what a client gets when we refuse to compile that stage's shader
    * -- so this is the point where a compiler refusal would otherwise walk
    * into the draw path.
    *
    * The fragment stage already has an answer: hk_cmd_bind_graphics_shader()
    * substitutes dev->null_fs for a null bind.  Apply it to a stage that was
    * never bound at all, so the substitution does not depend on the client
    * having called CmdBindShadersEXT for it.
    *
    * The vertex stage has no such substitute, and hk_bound_hw_vs() indexes
    * ->variants[] off the bound API shader: with none bound that is a wild
    * pointer a few bytes above zero, which hk_flush_shaders() dereferences
    * for its UVS layout and then publishes to the VDM.  Fail the draw closed
    * instead.
    */
   if (gfx->shaders[MESA_SHADER_FRAGMENT] == NULL)
      hk_cmd_bind_graphics_shader(cmd, MESA_SHADER_FRAGMENT, NULL);

   /* The device may have created its first sampler that needs custom-border
    * emulation since these shaders were bound. */
   if (unlikely(gfx->border_free_stages) &&
       p_atomic_read(&hk_device_physical(dev)->border_emulation)) {
      u_foreach_bit(stage, gfx->border_free_stages)
         hk_cmd_bind_graphics_shader(cmd, stage, gfx->shaders[stage]);
   }

   if (gfx->shaders[MESA_SHADER_VERTEX] == NULL &&
       gfx->shaders[MESA_SHADER_MESH] == NULL) {
      mesa_loge("draw with no vertex shader bound; dropping it");
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return NULL;
   }

   if (unlikely(gfx->occlusion.pending_pool) &&
       !hk_ensure_occlusion_query_for_draw(cmd))
      return NULL;

   /* An ordered render continuation (hk_m5_pass_merge_begin()) ends before a
    * draw that could break its memory equivalence; the draw goes to a
    * continuation stream ordered after the whole merged pass. Meta draws
    * inside the pass (attachment clears) neither write memory nor read the
    * attachments through it. */
   if (unlikely(cmd->current_cs.gfx && cmd->current_cs.gfx->m5_ordered) &&
       !cmd->in_meta) {
      const char *hazard = hk_m5_ordered_draw_hazard(cmd);
      if (hazard)
         hk_m5_pass_merge_end_ordered(cmd, hazard);
   }

   struct hk_cs *cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   if (cs)
      cs->m5_shader_writes_memory |= gfx->shaders_write_memory;

   if (!cs)
      return NULL;

   /* Annoyingly,
    * VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT is
    * render pass state on Imaginapple but draw state in Vulkan. In practice,
    * Proton never changes it within a render pass, but we technically need to
    * handle the switch regardless. Do so early since `cs` will be invalidated
    * if we need to split the render pass to switch representation mid-frame.
    *
    * Note we only do this dance with depth bias is actually enabled to avoid
    * senseless control stream splits with DXVK.
    */
   /* A split later in this function may need to reapply unchanged API state
    * to a fresh render. */
   const bool dbias_selected = dyn->rs.depth_bias.enable;
   const bool dbias_is_int =
      dbias_selected &&
      ((dyn->rs.depth_bias.representation ==
        VK_DEPTH_BIAS_REPRESENTATION_LEAST_REPRESENTABLE_VALUE_FORCE_UNORM_EXT) ||
       (hk_depth_stencil_plane_format(gfx->render.depth_att.vk_format, 0) ==
        VK_FORMAT_D16_UNORM));

   if (dbias_selected) {
      /* Attempt to set dbias_is_int per the draw requirement. If this fails,
       * flush the control stream and set it on the new control stream.
       */
      bool succ = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
      if (!succ) {
         perf_debug(cmd, "Splitting control stream due to depth bias");

         hk_cmd_buffer_end_graphics(cmd);
         cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
         if (!cs)
            return NULL;

         succ = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
         assert(succ && "can always set tri-state on a new control stream");
      }
   }

   /* A render pass instance continued in the previous instance's stream
    * (hk_m5_pass_merge_begin()) may commit other sample locations than that
    * stream already holds; a stream holds one set. Such a draw goes to a
    * continuation stream, as it would have in the instance's own stream. */
   if (unlikely(cs->m5_continued_pass) && cs->has_sample_locations &&
       !cmd->in_meta && dyn->ms.rasterization_samples != 0) {
      const uint64_t ctrl =
         dyn->ms.sample_locations_enable
            ? hk_pack_ppp_multisamplectrl(dyn->ms.sample_locations)
            : hk_default_sample_positions(dyn->ms.rasterization_samples);
      if (cs->ppp_multisamplectl != ctrl) {
         perf_debug(cmd, "Splitting continued render for sample locations");

         hk_cmd_buffer_end_graphics(cmd);
         cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
         if (!cs)
            return NULL;
         if (dbias_selected) {
            ASSERTED bool succ =
               u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
            assert(succ && "can always set tri-state on a new control stream");
         }
      }
   }

   hk_ensure_cs_has_space(cmd, cs, HK_GFX_DRAW_RESERVATION);
   if (vk_command_buffer_has_error(&cmd->vk))
      return NULL;

#ifndef NDEBUG
   if (unlikely(dev->dev.debug & AGX_DBG_DIRTY)) {
      hk_cmd_buffer_dirty_all(cmd);
   }
#endif

   /* Merge tess info before GS construction since that depends on
    * gfx->tess.prim
    */
   if ((IS_SHADER_DIRTY(TESS_CTRL) || IS_SHADER_DIRTY(TESS_EVAL)) &&
       gfx->shaders[MESA_SHADER_TESS_CTRL]) {
      struct hk_api_shader *tcs = gfx->shaders[MESA_SHADER_TESS_CTRL];
      struct hk_api_shader *tes = gfx->shaders[MESA_SHADER_TESS_EVAL];
      struct hk_shader *tese = hk_any_variant(tes);
      struct hk_shader *tesc = hk_only_variant(tcs);

      gfx->tess.info =
         hk_tess_info_merge(tese->info.tess.info, tesc->info.tess.info);

      /* Determine primitive based on the merged state */
      if (gfx->tess.info.points) {
         gfx->tess.prim = MESA_PRIM_POINTS;
      } else if (gfx->tess.info.mode == TESS_PRIMITIVE_ISOLINES) {
         gfx->tess.prim = MESA_PRIM_LINES;
      } else {
         gfx->tess.prim = MESA_PRIM_TRIANGLES;
      }
   }

   /* TODO: Try to reduce draw overhead of this */
   if (!gfx->shaders[MESA_SHADER_MESH])
      hk_handle_passthrough_gs(cmd, *draw);
   if (vk_command_buffer_has_error(&cmd->vk))
      return NULL;

   hk_flush_shaders(cmd);

   const bool geom = gfx->shaders[MESA_SHADER_GEOMETRY];
   const bool tess = gfx->shaders[MESA_SHADER_TESS_EVAL];
   bool adj = mesa_prim_has_adjacency(
      vk_topology_to_mesa(dyn->ia.primitive_topology));
   adj &= !geom;
   *needs_idx_robust =
      hk_needs_index_robustness(cmd, draw, conditional_direct);
   /* Reject heap allocation failure before parameter upload or lowering can
    * dereference the heap. Direct adjacency without restart uses no heap. */
   const bool needs_heap = geom || tess || (*needs_idx_robust && !adj) ||
      (adj && (draw->restart ||
               ((agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
                agx_is_indirect(draw->b))));
   if (needs_heap && !hk_cmd_ensure_geometry_heap(cmd))
      return NULL;

   /* The G17P robust-index producer emits the same native HAL200 indexed
    * body as an ordinary draw, only after a pre-graphics compute pass has
    * bounded the index range.  It therefore needs the ordinary render-state
    * publication path as well; the generated body carries the matching
    * context-relative selector when that publication is fresh. */
   const bool g17p_robust_index_path =
      (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
      *needs_idx_robust;
   const bool ordinary_draw_path =
      !geom && !tess && !adj &&
      (!*needs_idx_robust || g17p_robust_index_path);
   /* Geometry and tessellation draw their compute-generated vertices through
    * the bound hardware vertex program: poly's geometry raster variant, or
    * the evaluation shader after the software or hardware tessellator. It
    * publishes its render state exactly like an ordinary draw's. */
   /* An adjacency topology with no application geometry shader keeps the
    * hardware vertex program and only makes it fetch its own indices, so it
    * publishes render state exactly as a geometry or tessellation draw does.
    * Excluding it left hk_flush_dynamic_state() refusing the draw, and every
    * adjacency topology failed at vkEndCommandBuffer with
    * VK_ERROR_FEATURE_NOT_PRESENT. */
   const bool g17p_transformed_path =
      (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
      (geom || tess || adj);
   if (desc->push_dirty)
      hk_cmd_buffer_flush_push_descriptors(cmd, desc);

   /* HAL200's indexed draw packet carries its own native restart-state
    * record. Geometry/XFB lowers even a non-indexed API draw to that packet,
    * so emitting the legacy state here as well produces two consecutive
    * restart records before the synthetic indexed draw. */
   if ((draw->restart || geom) && !agx_native_graphics(&dev->dev) &&
       !agx_chip_g17_m5(dev->dev.chip))
      hk_flush_index(cmd, cs);

   /* Everything that can dirty render state for this draw has now run, so
    * this is the first point at which "does this draw republish state?" is
    * answerable, and -- because hk_flush_index() is not on the G17P path --
    * the last point at which nothing of this draw has been written to the
    * stream yet.  Both halves are required: the republication is the hazard,
    * and a split has to happen before the record it protects. */
   const bool republishes = hk_gfx_republishes_render_state(cmd);

   /* The HAL200 draw-state selector holds 16-bit scissor and
    * depth-bias indices. A stream whose arrays are about to exceed that
    * (only reachable with >65k distinct scissor/depth-bias states) is ended
    * before the republication instead of failing the draw. */
   const unsigned ppp_margin = AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS + 1;
   const bool ppp_index_limit =
      agx_native_graphics(&dev->dev) &&
      (cs->scissor.size / AGX_SCISSOR_LENGTH + ppp_margin > UINT16_MAX ||
       cs->depth_bias.size / AGX_DEPTH_BIAS_LENGTH + ppp_margin > UINT16_MAX);

   if (republishes && cs->stats.cmds > 0 && ppp_index_limit) {
      hk_cmd_buffer_end_graphics(cmd);
      cs = hk_cmd_buffer_get_cs(cmd, false /* compute */);
      if (!cs)
         return NULL;

      if (dbias_selected) {
         ASSERTED bool succ = u_tristate_set(&cs->cr.dbias_is_int, dbias_is_int);
         assert(succ && "can always set tri-state on a new control stream");
      }

      hk_ensure_cs_has_space(cmd, cs, HK_GFX_DRAW_RESERVATION);
   }

   hk_flush_dynamic_state(cmd, cs, draw_id, *draw,
                          ordinary_draw_path || g17p_transformed_path,
                          g17p_native_bind_tail_emitted);
   cmd->state.gfx.render.barrier_shader_writes |=
      gfx->shaders_write_memory || gfx->xfb_enabled;
   return cs;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindIndexBuffer2KHR(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, VkDeviceSize size,
                          VkIndexType indexType)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   vk_cmd_set_index_buffer_type(&cmd->vk, indexType);

   cmd->state.gfx.index = (struct hk_index_buffer_state){
      .buffer = hk_buffer_addr_range(buffer, offset, size, true),
      .size = agx_translate_index_size(vk_index_type_to_bytes(indexType)),
   };
}

void
hk_cmd_bind_vertex_buffer(struct hk_cmd_buffer *cmd, uint32_t vb_idx,
                          struct hk_addr_range addr_range)
{
   /* An identical rebind leaves every consumer's input unchanged. Dirtiness
    * already pending (e.g. after state invalidation) is kept as it is. */
   if (cmd->state.gfx.vb[vb_idx].addr == addr_range.addr &&
       cmd->state.gfx.vb[vb_idx].range == addr_range.range)
      return;

   cmd->state.gfx.vb[vb_idx] = addr_range;
   cmd->state.gfx.dirty |= HK_DIRTY_VB;
   cmd->state.gfx.vertex_buffer_generation++;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindVertexBuffers2(VkCommandBuffer commandBuffer, uint32_t firstBinding,
                         uint32_t bindingCount, const VkBuffer *pBuffers,
                         const VkDeviceSize *pOffsets,
                         const VkDeviceSize *pSizes,
                         const VkDeviceSize *pStrides)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   if (pStrides) {
      vk_cmd_set_vertex_binding_strides(&cmd->vk, firstBinding, bindingCount,
                                        pStrides);
   }

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(hk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;

      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;
      const struct hk_addr_range addr_range =
         hk_buffer_addr_range(buffer, pOffsets[i], size, true);

      hk_cmd_bind_vertex_buffer(cmd, idx, addr_range);
   }
}

static bool
hk_set_view_index(struct hk_cmd_buffer *cmd, uint32_t view_idx)
{
   if (cmd->state.gfx.render.view_mask) {
      cmd->state.gfx.descriptors.root.draw.view_index = view_idx;
      cmd->state.gfx.descriptors.root_dirty = true;
   }

   return true;
}

/*
 * Iterator macro to duplicate a draw for each enabled view (when multiview is
 * enabled, else always view 0). Along with hk_lower_multiview, this forms the
 * world's worst multiview lowering.
 */
#define hk_foreach_view(cmd)                                                   \
   u_foreach_bit(view_idx, cmd->state.gfx.render.view_mask ?: 1)               \
      if (hk_set_view_index(cmd, view_idx))

static void
hk_ia_update(struct hk_cmd_buffer *cmd, struct agx_draw draw,
             uint64_t ia_vertices, uint64_t ia_prims, uint64_t vs_invocations,
             uint64_t c_prims, uint64_t c_inv)
{
   perf_debug(cmd, "Input assembly counters");

   uint64_t draw_ptr;
   if (agx_is_indirect(draw.b)) {
      draw_ptr = draw.b.ptr;
   } else {
      uint32_t desc[] = {draw.b.count[0], draw.b.count[1], 0};
      draw_ptr = hk_pool_upload(cmd, &desc, sizeof(desc), 4);
   }

   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

   bool geom = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
   bool tess = cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];

   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* Clipper counters depend on geom/tess outputs and must be written with the
    * geom/tess output. They are updated as IA counters only when geom/tess is
    * not used. G17P represents a counter that is not written by the writable
    * scratch page, like a disabled one.
    */
   if (geom || tess) {
      c_prims = agx_native_graphics(&dev->dev) ? AGX_SCRATCH_PAGE_ADDRESS : 0;
      c_inv = c_prims;
   }

   uint32_t patch_size = dyn->ts.patch_control_points;

   if (draw.restart) {
      uint32_t index_size_B = agx_index_size_to_B(draw.index_size);
      /* The range helper removes firstIndex for direct draws. Adjust its
       * matching base too; indirect draws retain their original base/range
       * and read firstIndex from the application draw descriptor.
       */
      uint64_t index_buffer = agx_draw_index_buffer(draw);

      if (agx_native_graphics(&dev->dev)) {
         /* Disabled statistics are represented by the writable scratch page,
          * so all five addresses are valid. Use the native specialization
          * that does not need to reduce nullable 64-bit pointers. */
         libagx_increment_ia_restart_nonnull(
            cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX, ia_vertices,
            ia_prims, vs_invocations, c_prims, c_inv, draw_ptr,
            index_buffer, agx_draw_index_range_el(draw),
            dyn->ia.primitive_restart_index, index_size_B, prim, patch_size);
      } else {
         libagx_increment_ia_restart(
            cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX, ia_vertices,
            ia_prims, vs_invocations, c_prims, c_inv, draw_ptr,
            index_buffer, agx_draw_index_range_el(draw),
            dyn->ia.primitive_restart_index, index_size_B, prim, patch_size);
      }
   } else {
      libagx_increment_ia(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                          ia_vertices, ia_prims, vs_invocations, c_prims, c_inv,
                          draw_ptr, prim, patch_size);
   }
}

/* The draw-state selector record names the stream's latest viewport_count
 * scissor and depth-bias records. A draw record carries its context offset
 * only right after a fresh bind tail; consecutive tails that select the same
 * records share one upload. */
static bool
hk_g17p_draw_state_offset(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                          uint32_t *offset)
{
   const unsigned viewport_count =
      MAX2(cmd->vk.dynamic_graphics_state.vp.viewport_count, 1);

   if (cs->g17p_draw_state.offset &&
       cs->g17p_draw_state.scissor_size == cs->scissor.size &&
       cs->g17p_draw_state.depth_bias_size == cs->depth_bias.size &&
       cs->g17p_draw_state.viewport_count == viewport_count) {
      *offset = cs->g17p_draw_state.offset;
      return true;
   }

   struct agx_ptr state = hk_pool_vdm_alloc(cmd, 8, 4);
   if (!state.cpu ||
       !hk_g17p_upload_draw_indices(state.cpu, 8, state.gpu,
                                    cs->scissor.size, cs->depth_bias.size,
                                    viewport_count, offset))
      return false;

   cs->g17p_draw_state.offset = *offset;
   cs->g17p_draw_state.scissor_size = cs->scissor.size;
   cs->g17p_draw_state.depth_bias_size = cs->depth_bias.size;
   cs->g17p_draw_state.viewport_count = viewport_count;
   return true;
}

/* The primitives of the HAL200 direct Index List tables: 0, 1, 3, 5, 6, 9,
 * 11, 13, 14 and 15. */
static inline bool
hk_g17p_draw_primitive_supported(enum agx_primitive topology)
{
   return topology < 16 && (UINT32_C(0xea6b) >> topology) & 1;
}

/* The G17P draw record after any state publication: the context-relative
 * draw-state selector record when a fresh bind tail precedes it, then the
 * HAL200 draw body (indirect, indexed or direct).  Shared by hk_draw_impl()
 * and hk_g17p_emit_indirect_record().  Returns false with the command buffer
 * error set when the draw cannot be encoded. */
static bool
hk_g17p_emit_draw(struct hk_cmd_buffer *cmd, struct hk_cs *cs, uint8_t *out,
                  struct agx_draw draw, enum agx_primitive topology,
                  bool tail_emitted)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* The selector completes a fresh bind tail. Without one the draw record
    * starts at its body; a bare selector would be parsed as a header. */
   uint32_t draw_state_offset = 0;
   if (tail_emitted && !hk_g17p_draw_state_offset(cmd, cs, &draw_state_offset))
      goto refused;

   if (agx_is_indirect(draw.b)) {
      const struct agx_vdm_hal200_indirect_draw_fields fields = {
         /* Not emitted without a fresh tail, but packed regardless. */
         .draw_state_context_offset = tail_emitted ? draw_state_offset : 4,
         .primitive = (uint8_t)topology,
         .indexed = draw.indexed,
         .index_size_B =
            draw.indexed ? agx_index_size_to_B(draw.index_size) : 0,
         .restart_enable = draw.restart,
         /* Geometry/XFB lowers a non-indexed API draw to a synthetic
          * indexed draw, which restarts at its own fixed value. */
         .restart_index = hk_draw_restart_index(cmd, draw),
         .index_buffer = draw.index_buffer,
         .index_buffer_range_B = draw.index_buffer_range_B,
         /* Unlike PPP and stream-link pointers, the Index List
          * indirect-argument pointer is a native absolute 48-bit VA.
          * Compacting it by the render-context base makes hardware
          * read an unmapped address and silently skip the draw. */
         .indirect_buffer = draw.b.ptr,
      };
      struct agx_vdm_hal200_indirect_draw encoded;
      size_t emitted;
      if (!agx_vdm_hal200_indirect_draw_pack(dev->dev.chip, &fields,
                                              &encoded) ||
          !agx_vdm_hal200_draw_emit(dev->dev.chip, encoded.bytes,
                                    encoded.length, tail_emitted, out,
                                    (uint8_t *)cs->end - out, &emitted))
         goto refused;
      cs->current = out + emitted;
      return true;
   }

   if (!hk_g17p_draw_primitive_supported(topology) || !draw.b.count[1])
      goto refused;

   uint32_t w[10];
   unsigned n = 0;
   if (tail_emitted)
      w[n++] = draw_state_offset;

   if (draw.indexed) {
      /* Native direct indexed 0x47e57c..0x47e678: a restart State Update
       * precedes every indexed draw (restart disabled or not), the index
       * buffer is relocated to the first index, and its remaining length is
       * rounded up to dwords minus one. The whole requested span must lie
       * inside the bound range, except the one-dword zero page that stands in
       * for a null index buffer. */
      const unsigned index_size_B = agx_index_size_to_B(draw.index_size);
      const uint64_t range_B = draw.index_buffer_range_B;
      const uint64_t offset_B = (uint64_t)draw.start * index_size_B;
      const uint64_t count_B = (uint64_t)draw.b.count[0] * index_size_B;
      const bool zero_indices = draw.index_buffer == AGX_ZERO_PAGE_ADDRESS &&
                                range_B == 4 && draw.start == 0;
      const uint64_t limit = UINT64_C(1) << 48;
      if (draw.index_size > AGX_INDEX_SIZE_U32 || !draw.index_buffer ||
          (draw.index_buffer & (index_size_B - 1)) ||
          draw.index_buffer >= limit || !range_B || range_B > UINT32_MAX ||
          range_B > limit - draw.index_buffer || !draw.b.count[0] ||
          offset_B >= range_B ||
          (!zero_indices && count_B > range_B - offset_B))
         goto refused;

      const uint64_t ib = draw.index_buffer + offset_B;
      w[n++] = UINT32_C(0x40000001);
      w[n++] = hk_draw_restart_index(cmd, draw);
      w[n++] = UINT32_C(0x61f00000) | (uint32_t)((ib >> 32) & 0xff) |
               ((uint32_t)topology << 8) |
               ((uint32_t)draw.index_size << 17) |
               (draw.restart ? UINT32_C(1) << 16 : 0);
      w[n++] = (uint32_t)ib;
      w[n++] = draw.b.count[0];
      w[n++] = draw.b.count[1];
      w[n++] = (uint32_t)(int32_t)draw.index_bias;
      w[n++] = (uint32_t)(((range_B - offset_B + (ib & 3) + 3) >> 2) - 1);
      w[n++] = (uint32_t)((ib >> 40) & 0xff);
   } else {
      if (!draw.b.count[0])
         goto refused;

      w[n++] = (AGX_VDM_HAL200_DIRECT_DRAW_OPCODE << 16) |
               ((uint32_t)topology << 8);
      w[n++] = draw.b.count[0];
      w[n++] = draw.b.count[1];
      w[n++] = draw.start;
   }

   if ((size_t)((uint8_t *)cs->end - out) < n * 4)
      goto refused;

   memcpy(out, w, n * 4);
   cs->current = out + n * 4;
   return true;

refused:
   vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   return false;
}

/* Capture the selected last producer at each successful emission, never
 * infer a stream's layer range from the command buffer's final bindings.
 */
static void
hk_g17p_note_layer_zero_draw(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   if (!agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev) ||
       cmd->state.gfx.render.view_mask ||
       cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] ||
       cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL]) {
      cs->g17p_layer_zero_unknown = true;
      return;
   }

   const struct hk_shader *vs = hk_bound_hw_vs(&cmd->state.gfx);
   if (vs && vs->info.g17p_default_layer_zero)
      cs->g17p_layer_zero_draws++;
   else
      cs->g17p_layer_zero_unknown = true;
}

/* An upper bound of the primitives one view of an API draw submits, or
 * UINT64_MAX when recording cannot bound it: an indirect count, or geometry
 * and tessellation stages, whose amplification is not bounded cheaply.
 * Primitive restart only lowers the true count; adjacency topologies (which
 * without a geometry shader rasterize their base primitives) are counted as
 * one primitive per vertex. */
static inline uint64_t
hk_draw_primitive_bound(const struct hk_cmd_buffer *cmd,
                        const struct agx_draw *draw)
{
   const struct hk_graphics_state *gfx = &cmd->state.gfx;
   if (agx_is_indirect(draw->b) || gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL])
      return UINT64_MAX;

   const uint64_t n = draw->b.count[0];
   uint64_t primitives;
   switch (cmd->vk.dynamic_graphics_state.ia.primitive_topology) {
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      primitives = n / 2;
      break;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      primitives = n >= 1 ? n - 1 : 0;
      break;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      primitives = n / 3;
      break;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      primitives = n >= 2 ? n - 2 : 0;
      break;
   case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST:
      return UINT64_MAX;
   default:
      primitives = n;
      break;
   }

   /* Both factors are below 2^32, so the product fits. */
   return primitives * draw->b.count[1];
}

/* Scoped to one API multi-draw. There is no pending repair across API
 * commands, barriers, or render stream changes. */
struct hk_robust_index_batch {
   struct agx_ptr jobs;
   uint32_t count;
   struct hk_cs *cs;
   struct agx_draw draw;
   uint32_t restart_index;
};

static void
hk_flush_robust_index_batch(struct hk_cmd_buffer *cmd,
                            struct hk_robust_index_batch *batch)
{
   if (!batch->count || vk_command_buffer_has_error(&cmd->vk))
      return;

   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   const struct agx_draw *draw = &batch->draw;
   libagx_draw_robust_index_batch(cmd, agx_1d(32 * batch->count),
                                  AGX_BARRIER_ALL | AGX_PREGFX, batch->jobs.gpu,
                                  hk_heap(cmd),
                                  dev->dev.printf_buffer_address,
                                  draw->index_buffer,
                                  draw->index_buffer_range_B, draw->restart,
                                  batch->restart_index,
                                  agx_chip_record_family(dev->dev.chip),
                                  cmd->state.gfx.topology, draw->index_size);
   batch->cs->stats.robust_index_dispatches++;
   batch->cs->stats.robust_index_batches++;
   batch->count = 0;
}

static void
hk_emit_robust_index(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                     struct agx_draw draw, enum agx_primitive topology,
                     bool tail_emitted, struct hk_robust_index_batch *batch)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   uint32_t draw_state_offset = 0;
   if (agx_native_graphics(&dev->dev) && tail_emitted) {
      struct agx_ptr draw_state = hk_pool_vdm_alloc(cmd, 8, 4);
      if (!draw_state.cpu ||
          !hk_g17p_upload_draw_indices(
             draw_state.cpu, 8, draw_state.gpu, cs->scissor.size,
             cs->depth_bias.size,
             MAX2(cmd->vk.dynamic_graphics_state.vp.viewport_count, 1),
             &draw_state_offset)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
   }

   size_t size_B = libagx_draw_robust_index_vdm_size(
      agx_chip_record_family(dev->dev.chip), tail_emitted);
   uint64_t target = hk_cs_alloc_for_indirect(cs, size_B);

   uint64_t empty_draw_jump = 0;
   if (agx_native_graphics(&dev->dev)) {
      const uint64_t compact = hk_g17p_vdm_compact(cmd, target + size_B);
      if (!compact || compact >= (UINT64_C(1) << 40)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
      empty_draw_jump = compact;
   } else if (agx_chip_g17_m5(dev->dev.chip)) {
      if (!agx_m5_vdm_compact(target + size_B, &empty_draw_jump)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
         return;
      }
   }

   const struct libagx_robust_index_draw job = {
      .vdm = target,
      .command = draw.b.ptr,
      .empty_draw_jump = empty_draw_jump,
      .draw_state_offset = draw_state_offset,
      .render_state_emitted = tail_emitted,
   };
   const uint32_t restart_index = hk_draw_restart_index(cmd, draw);
   cs->stats.robust_index_draws++;
   if (batch) {
      assert(batch->count == 0 || batch->cs == cs);
      ((struct libagx_robust_index_draw *)batch->jobs.cpu)[batch->count++] =
         job;
      batch->cs = cs;
      batch->draw = draw;
      batch->restart_index = restart_index;
   } else {
      libagx_draw_robust_index(
         cmd, agx_1d(32), AGX_BARRIER_ALL | AGX_PREGFX, target, hk_heap(cmd),
         dev->dev.printf_buffer_address, draw.b.ptr, draw.index_buffer,
         draw.index_buffer_range_B, draw.restart, restart_index,
         agx_chip_record_family(dev->dev.chip), topology, draw_state_offset,
         tail_emitted, empty_draw_jump, draw.index_size);
      cs->stats.robust_index_dispatches++;
   }
}

/* G17P and M5: a draw whose pre-rasterization shaders may read images
 * orders its render's vertex half after the image barriers before it
 * (HK_SYNC_GFX_IMG). Software tessellation, geometry, task and mesh stages
 * are assumed to read.
 *
 * An image barrier's memory dependency covers only its image, but its
 * execution dependency covers all work in its stages, so vertex work that
 * writes memory (stores, atomics, transform feedback) is ordered after it
 * too: a write after an earlier pass's read of another resource may rely on
 * that execution dependency alone. */
static inline void
hk_note_pre_raster_images(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   if (!hk_sync_image_class(cmd) ||
       (cs->sync.vtx_images && cs->sync.vtx_images_epoch == cmd->sync.epoch))
      return;

   const struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
   if (!hw_vs || hw_vs->info.reads_images ||
       hw_vs->info.writes_external_memory || gfx->xfb_enabled ||
       gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       gfx->shaders[MESA_SHADER_GEOMETRY] || gfx->shaders[MESA_SHADER_TASK] ||
       gfx->shaders[MESA_SHADER_MESH])
      hk_sync_vertex_images(cmd, cs);
}

static __attribute__((hot)) void
hk_draw_impl(struct hk_cmd_buffer *cmd, uint16_t draw_id, struct agx_draw draw_,
             struct hk_robust_index_batch *batch)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   /* Filter trivial draws so we don't need to worry about null index buffers */
   if (!agx_is_indirect(draw_.b) &&
       (draw_.b.count[0] == 0 || draw_.b.count[1] == 0))
      return;

   /* G17P: before anything is recorded for the draw, order a transfer the
    * application did not order before graphics work if the draw may access
    * what it wrote (hk_xfer_check_draw_slow()). */
   if (agx_is_indirect(draw_.b))
      hk_xfer_check_draw(cmd, draw_.b.ptr,
                         draw_.b.ptr + (draw_.indexed ? 5 : 4) * 4);
   else
      hk_xfer_check_draw(cmd, 0, 0);

   hk_g17p_note_depth_use(cmd);

   draw_.restart = dyn->ia.primitive_restart_enable && draw_.indexed;
   draw_.index_size = cmd->state.gfx.index.size;
   struct hk_device *dev = hk_cmd_buffer_device(cmd);

   /* Null index bindings execute the requested vertices, all with index zero.
    * Normalize before native admission and conditional draw lowering, as the
    * older path does in hk_needs_index_robustness. The one-dword range bounds
    * hardware fetches to the zero page; it is not a count limit.
    */
   if ((agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) &&
       draw_.indexed && draw_.index_buffer_range_B == 0) {
      draw_.index_buffer = AGX_ZERO_PAGE_ADDRESS;
      draw_.index_buffer_range_B = 4;
      draw_.start = 0;
   }
   const struct agx_draw api_draw = draw_;
   bool conditional_direct_indexed = false;
   const uint64_t primitive_bound =
      (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip))
         ? hk_draw_primitive_bound(cmd, &api_draw) : 0;

   /* There is no native conditional-render packet on AGX. Lower the command
    * to the API's indirect record and let an ordered compute kernel either
    * copy it or zero its counts from the execution-time predicate. This also
    * predicates IA/pipeline-stat accounting and every later geometry lowering
    * because they all consume the patched record. */
   if (hk_conditional_render_is_active(&cmd->state.conditional, cmd->in_meta,
                                        cmd->conditional_meta)) {
      uint64_t source = draw_.b.ptr;
      const uint32_t words = draw_.indexed ? 5 : 4;
      conditional_direct_indexed =
         draw_.indexed && !agx_is_indirect(draw_.b);

      if (!agx_is_indirect(draw_.b)) {
         uint32_t record[HK_CONDITIONAL_RENDER_MAX_WORDS];
         const uint32_t built = hk_conditional_render_direct_draw_record(
            record, draw_.indexed, draw_.b.count[0], draw_.b.count[1],
            draw_.start, (int32_t)draw_.index_bias, draw_.start_instance);
         assert(built == words);
         source = hk_pool_upload(cmd, record, words * sizeof(uint32_t), 4);
      }

      const uint64_t patched = hk_apply_conditional_rendering(
         cmd, source, words, AGX_BARRIER_ALL | AGX_PREGFX);
      if (!patched)
         return;

      draw_ = draw_.indexed
                 ? agx_draw_indexed_indirect(
                      patched, draw_.index_buffer,
                      draw_.index_buffer_range_B, draw_.index_size,
                      draw_.restart)
                 : agx_draw_indirect(patched);
   }

   uint64_t stat_ia_verts = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT);

   uint64_t stat_ia_prims = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT);

   uint64_t stat_vs_inv = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT);

   uint64_t stat_c_inv = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT);

   uint64_t stat_c_prims = hk_pipeline_stat_addr(
      cmd, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT);

   bool ia_stats = hk_stat_enabled(stat_ia_verts) ||
                   hk_stat_enabled(stat_ia_prims) ||
                   hk_stat_enabled(stat_vs_inv) ||
                   hk_stat_enabled(stat_c_inv) || hk_stat_enabled(stat_c_prims);
   /* Validate the native index domain before robustness or geometry lowering
    * can introduce a pre-render compute dispatch.  Robust draws keep their
    * application range for the ordered GPU-side bounds pass, which shadows the
    * in-range indices and pads the overread with zero indices; only draws
    * that reach the native Index List unmodified prove their complete span
    * here. */
   /* Geometry and software tessellation run the vertex shader in compute,
    * which fetches indices robustly like on every other generation. */
   const bool software_index_fetch =
      cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY] ||
      cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];
   if (agx_native_graphics(&dev->dev) && draw_.indexed &&
       !software_index_fetch) {
      uint64_t address;
      uint32_t size_word;
      const bool valid_index_size =
         draw_.index_size == AGX_INDEX_SIZE_U8 ||
         draw_.index_size == AGX_INDEX_SIZE_U16 ||
         draw_.index_size == AGX_INDEX_SIZE_U32;
      struct agx_draw robust_draw = draw_;
      const bool robust_index = hk_needs_index_robustness(
         cmd, &robust_draw, conditional_direct_indexed ? &api_draw : NULL);
      const bool conditional_direct_in_bounds =
         !conditional_direct_indexed || robust_index ||
         (valid_index_size &&
          agx_vdm_hal200_index_range(
             api_draw.index_buffer, api_draw.index_buffer_range_B,
             api_draw.start, api_draw.b.count[0],
             agx_index_size_to_B(api_draw.index_size), &address, &size_word));
      /* A tessellated draw's vertex shader fetches its indices in compute,
       * but they still have to be in range. */
      if (!valid_index_size ||
          (agx_is_indirect(draw_.b)
              ? (!draw_.b.ptr || !draw_.index_buffer ||
                 draw_.index_buffer_range_B == 0 ||
                 !conditional_direct_in_bounds)
              : !robust_index &&
                   !agx_vdm_hal200_index_range(
                      draw_.index_buffer, draw_.index_buffer_range_B,
                      draw_.start, draw_.b.count[0],
                      agx_index_size_to_B(draw_.index_size), &address,
                      &size_word))) {
         mesa_loge("native G17P indexed draw requires a valid "
                   "uint8/uint16/uint32 range");
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
   }

   hk_foreach_view(cmd) {
      struct agx_draw draw = draw_;
      bool needs_idx_robust = false;
      bool g17p_native_bind_tail_emitted = false;
      struct hk_cs *cs =
         hk_flush_gfx_state(cmd, draw_id, &draw, &needs_idx_robust,
                            &g17p_native_bind_tail_emitted,
                            conditional_direct_indexed ? &api_draw : NULL);
      /* If we failed to allocate a control stream, we've already lost the
       * device. Just drop the draw so we don't crash.
       */
      if (!cs)
         return;
      if (vk_command_buffer_has_error(&cmd->vk))
         return;
      if (unlikely(!cmd->state.gfx.render.first_draw_seen))
         hk_g17p_note_first_draw(cmd, cs, draw_);

      /* Each view's draw counts: multiview replays the draw per view. */
      cs->g17p_primitive_bound =
         primitive_bound > UINT64_MAX - cs->g17p_primitive_bound
            ? UINT64_MAX
            : cs->g17p_primitive_bound + primitive_bound;

      struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
      bool geom = cmd->state.gfx.shaders[MESA_SHADER_GEOMETRY];
      bool tess = cmd->state.gfx.shaders[MESA_SHADER_TESS_EVAL];
      bool adj = mesa_prim_has_adjacency(
         vk_topology_to_mesa(dyn->ia.primitive_topology));
      adj &= !geom;
      needs_idx_robust &= !adj;

      struct hk_cs *ccs = NULL;
      uint8_t *out = cs->current;
      assert(cs->current + 0x1000 < cs->end);

      if (tess && HK_PERF(dev, NOTESS))
         continue;

      cs->stats.calls++;
      hk_upload_stat(cmd, draws, 1);

      if (geom || tess) {
         ccs =
            hk_cmd_buffer_get_cs_general(cmd, &cmd->current_cs.pre_gfx, true);
         if (!ccs)
            return;
      }

      if (ia_stats) {
         hk_ia_update(cmd, draw, stat_ia_verts, stat_ia_prims, stat_vs_inv,
                      stat_c_prims, stat_c_inv);
      }

      /* Without geometry or tessellation the primitives generated are the
       * assembled primitives, counted like the clipper statistic; the
       * tessellator and the geometry pipeline count their own outputs. */
      const uint64_t prims_generated = hk_prims_generated_addr(cmd, 0);
      if (prims_generated && !geom && !tess) {
         hk_ia_update(cmd, draw, AGX_SCRATCH_PAGE_ADDRESS,
                      AGX_SCRATCH_PAGE_ADDRESS, AGX_SCRATCH_PAGE_ADDRESS,
                      AGX_SCRATCH_PAGE_ADDRESS, prims_generated);
      }

      if (tess) {
         cmd->state.gfx.tess.tes_direct_threads =
            hk_tes_direct_query_threads(cmd, draw);
         draw = hk_launch_tess(cmd, ccs, draw, geom ? 0 : stat_c_prims,
                               geom ? 0 : stat_c_inv,
                               geom ? 0 : prims_generated);
      }

      if (geom)
         draw = hk_launch_gs_prerast(cmd, ccs, draw);

      if (adj) {
         assert(!geom && "geometry shaders handle adj directly");
         enum mesa_prim prim = vk_topology_to_mesa(dyn->ia.primitive_topology);

         if (draw.restart) {
            draw = hk_draw_without_restart(cmd, draw, 1);
            prim = u_decomposed_prim(prim);
         }

         if (agx_is_indirect(draw.b)) {
            const size_t size = sizeof(VkDrawIndexedIndirectCommand);
            static_assert(sizeof(VkDrawIndexedIndirectCommand) >
                             sizeof(VkDrawIndirectCommand),
                          "allocation size is conservative");

            uint64_t out_draw = hk_pool_alloc(cmd, size, 4).gpu;
            struct hk_device *dev = hk_cmd_buffer_device(cmd);

            if (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip)) {
               /* As in the direct case above, a G17P hardware vertex shader
                * never sees the prolog's remap, so the mapping has to reach
                * the input assembler as real indices.  The output count is
                * only known once the GPU has read the indirect command, so
                * the kernel allocates the index buffer from the heap and
                * names it in the indexed indirect draw it writes. */

               libagx_write_adj_indices_indirect(
                  cmd, agx_1d(1024), AGX_BARRIER_ALL | AGX_PREGFX,
                  hk_heap(cmd), draw.b.ptr, out_draw,
                  draw.indexed ? draw.index_buffer : 0,
                  draw.indexed ? agx_draw_index_range_el(draw) : 0,
                  draw.indexed ? agx_index_size_to_B(draw.index_size) : 0,
                  prim);

               draw = agx_draw_indexed_indirect(out_draw, dev->heap->va->addr,
                                                dev->heap->size,
                                                AGX_INDEX_SIZE_U32, false);
            } else {
               struct hk_descriptor_state *desc = &cmd->state.gfx.descriptors;

               libagx_draw_without_adj(
                  cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, out_draw,
                  draw.b.ptr, desc->root.draw.vertex_params, draw.index_buffer,
                  draw.indexed ? agx_draw_index_range_el(draw) : 0,
                  draw.indexed ? agx_index_size_to_B(draw.index_size) : 0,
                  prim);

               draw = agx_draw_indirect(out_draw);
            }
         } else {
            unsigned count = libagx_remap_adj_count(draw.b.count[0], prim);
            struct hk_device *dev = hk_cmd_buffer_device(cmd);

            /* Software indexing hands the remap to the VS prolog, which a G17P
             * hardware vertex shader never observes: it reads its invocation
             * index straight from SR 0xdd rather than the
             * AGX_ABI_VIN_VERTEX_ID the prolog exports, so every adjacency
             * draw fetched in[i] where it owed in[map(i)].  Materialise the
             * mapping as an index buffer instead and let the input assembler
             * apply it, which is the one path the vertex stage does see. */
            if (count > 0 &&
                (agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip))) {
               unsigned ib_B = count * sizeof(uint32_t);
               uint64_t ib = hk_pool_alloc(cmd, ib_B, 4).gpu;

               /* Take the index buffer straight off the draw rather than out
                * of poly_vertex_params, which is only uploaded when the linked
                * vertex shader asked for software indexing. */
               libagx_write_adj_indices(
                  cmd, agx_1d(count), AGX_BARRIER_ALL | AGX_PREGFX, ib,
                  draw.indexed ? agx_draw_index_buffer(draw) : 0,
                  draw.indexed ? agx_draw_index_range_el(draw) : 0, count,
                  draw.indexed ? agx_index_size_to_B(draw.index_size) : 0,
                  draw.indexed ? draw.index_bias : draw.start, prim);

               draw = agx_draw_indexed(count, draw.b.count[1], 0, 0,
                                       draw.start_instance, ib, ib_B,
                                       AGX_INDEX_SIZE_U32, false);
            } else {
               draw = (struct agx_draw){
                  .b = agx_3d(count, draw.b.count[1], 1),
               };
            }
         }
      }

      /* Geometry, tessellation and adjacency lowering can turn a nonempty API
       * draw into a direct raster draw with no complete output primitives.
       * There is no draw packet to encode in that case. The pre-raster work,
       * including any shader side effects, has already been recorded above. */
      if (!agx_is_indirect(draw.b) &&
          (draw.b.count[0] == 0 || draw.b.count[1] == 0)) {
         /* The bind tail was already published before lowering determined
          * there were no complete primitives. Its selector word must still
          * be present, even though no Index List follows it. */
         if (agx_native_graphics(&dev->dev) && g17p_native_bind_tail_emitted) {
            struct agx_ptr state = hk_pool_vdm_alloc(cmd, 8, 4);
            uint32_t offset;
            if (!state.cpu || !hk_g17p_upload_draw_indices(
                   state.cpu, 8, state.gpu, cs->scissor.size,
                   cs->depth_bias.size, MAX2(dyn->vp.viewport_count, 1),
                   &offset)) {
               vk_command_buffer_set_error(&cmd->vk,
                                           VK_ERROR_FEATURE_NOT_PRESENT);
               return;
            }
            memcpy(out, &offset, sizeof(offset));
            out += sizeof(offset);
            cs->current = out;
         }
         continue;
      }

      /* hk_g17p_emit_draw() validates a direct draw as it encodes it. */
      enum agx_primitive topology = cmd->state.gfx.topology;
      if ((!agx_native_graphics(&dev->dev) || agx_is_indirect(draw.b)) &&
          !agx_vdm_draw_supported(dev->dev.chip, draw, topology)) {
         if (debug_get_option_m5_launch_debug())
            mesa_loge("draw admission: indexed=%u indirect=%u size=%u ib=%#llx range=%u arg=%#llx topology=%u",
                      draw.indexed, agx_is_indirect(draw.b), draw.index_size,
                      (unsigned long long)draw.index_buffer, draw.index_buffer_range_B,
                      (unsigned long long)draw.b.ptr, topology);
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }

      if (needs_idx_robust) {
         assert(!geom && !tess && !adj);
         perf_debug(cmd, "lowering robust index buffer");

         cs->current = out;

         draw = hk_draw_as_indexed_indirect(cmd, draw);
         if (!agx_vdm_draw_supported(dev->dev.chip, draw, topology)) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }

         hk_emit_robust_index(cmd, cs, draw, topology,
                              g17p_native_bind_tail_emitted, batch);
      } else {
         if (agx_native_graphics(&dev->dev)) {
            if (!hk_g17p_emit_draw(cmd, cs, out, draw, topology,
                                   g17p_native_bind_tail_emitted))
               return;
         } else {
            /* HAL300 direct Index Lists consume persistent restart state;
             * the indirect packet carries its own state update. Geometry
             * may have synthesized an index width unrelated to the API IB. */
            if (agx_chip_g17_m5(dev->dev.chip) && draw.indexed &&
                draw.restart && !agx_is_indirect(draw.b)) {
               const uint32_t index =
                  agx_primitive_restart_index(draw.index_size);
               if (cs->restart_index != index) {
                  agx_push(out, VDM_STATE, cfg) {
                     cfg.restart_index_present = true;
                  }
                  agx_push(out, VDM_STATE_RESTART_INDEX, cfg) {
                     cfg.value = index;
                  }
                  cs->restart_index = index;
               }
            }
            cs->current = (void *)agx_vdm_draw((uint32_t *)out,
                                               dev->dev.chip, draw, topology,
                                               hk_draw_restart_index(cmd, draw));
         }
      }

      hk_g17p_note_layer_zero_draw(cmd, cs);
      hk_note_pre_raster_images(cmd, cs);
      hk_m5_note_render_resources(cmd, cs);
      if (unlikely(dev->frame_anatomy)) {
         hk_frame_anatomy_note_fs(cs, hk_bound_fs(cmd));
         hk_frame_anatomy_note_vs(cs, hk_bound_hw_vs(&cmd->state.gfx));
         hk_frame_anatomy_note_m5_fragment(cs);
      }
      cs->stats.cmds++;
   }
}

static __attribute__((hot)) void
hk_draw(struct hk_cmd_buffer *cmd, uint16_t draw_id, struct agx_draw draw)
{
   hk_draw_impl(cmd, draw_id, draw, NULL);
}

static void
hk_draw_mesh_native(struct hk_cmd_buffer *cmd, uint64_t grid,
                    const uint32_t *cpu_grid, uint32_t draw_id,
                    uint64_t count_address)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_api_shader *task_api = gfx->shaders[MESA_SHADER_TASK];
   if (!gfx->shaders[MESA_SHADER_MESH] ||
       (task_api && (dev->dev.chip != AGX_CHIP_G17S || !hk_m5_task_enabled())) ||
       (dev->dev.chip == AGX_CHIP_G17P
          ? hk_mesh_mode() != HK_MESH_MODE_NATIVE_PROBE
          : (dev->dev.chip != AGX_CHIP_G17S ||
             !(dev->dev.params.features & DRM_ASAHI_FEATURE_RENDER_RESOURCES_V2)))) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   gfx->mesh.grid = grid;
   hk_xfer_check_draw(cmd, grid, grid + 3 * sizeof(uint32_t));
   if (count_address)
      hk_xfer_check_draw(cmd, count_address, count_address + sizeof(uint32_t));
   

   const bool conditional = hk_conditional_render_is_active(
      &cmd->state.conditional, cmd->in_meta, cmd->conditional_meta);
   const struct hk_mesh_info *mesh_info =
      &hk_main_mesh_variant(gfx->shaders[MESA_SHADER_MESH])->info.mesh;
   const struct hk_shader *task = task_api ? &task_api->variants[0] : NULL;
   /* A task draw whose grid, draw count or predicate lives in GPU memory is
    * recorded with one object workgroup; libagx_m5_task_dispatch patches the
    * object thread counts before the render and publishes whether the draw
    * executes (hk_m5_task_predicate). */
   const bool task_gpu_dispatch = task_api &&
      (!cpu_grid || count_address || conditional);
   uint64_t task_flag = 0;
   if (task) {
      if (!task->only_linked) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      if (task_gpu_dispatch) {
         struct agx_ptr flag = hk_pool_alloc(cmd, 4, 4);
         if (!flag.cpu) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
            return;
         }
         task_flag = flag.gpu;
         for (unsigned i = 0; i < 3; ++i)
            gfx->mesh.task_grid[i] = 1;
      } else {
         memcpy(gfx->mesh.task_grid, cpu_grid, sizeof(gfx->mesh.task_grid));
      }
   }
   gfx->mesh.grid = grid;
   /* Only a stage that reads gl_DrawID or gl_NumWorkGroups needs the root
    * republished per draw; others keep the bound root. With a task stage,
    * the API grid belongs to the task shader. */
   struct hk_descriptor_state *desc = &gfx->descriptors;
   if (task && desc->root.mesh_task_flag != task_flag) {
      desc->root.mesh_task_flag = task_flag;
      desc->root_dirty = true;
   }
   if ((mesh_info->reads_num_workgroups ||
        (task && task->info.mesh.reads_num_workgroups)) &&
       desc->root.mesh_grid_addr != grid) {
      desc->root.mesh_grid_addr = grid;
      desc->root_dirty = true;
   }
   if ((mesh_info->reads_draw_id || (task && task->info.mesh.reads_draw_id)) &&
       desc->root.mesh_draw_id != draw_id) {
      desc->root.mesh_draw_id = draw_id;
      desc->root_dirty = true;
   }
   gfx->mesh.object_params = 0;
   if (!task && (dev->dev.chip == AGX_CHIP_G17P || count_address || conditional)) {
      const struct {
         uint64_t grid, count, predicate;
         uint32_t draw_id, inverted;
      } params = {grid, count_address,
         conditional ? cmd->state.conditional.address : 0,
         draw_id, conditional && cmd->state.conditional.inverted};
      gfx->mesh.object_params = hk_pool_upload(cmd, &params, sizeof(params), 8);
   }
   gfx->shaders_dirty |= BITFIELD_BIT(MESA_SHADER_MESH);
   hk_g17p_note_depth_use(cmd);
   struct agx_draw dummy = {.b = agx_1d(3)};
   bool robust = false, tail = false;
   struct hk_cs *cs = hk_flush_gfx_state(cmd, draw_id, &dummy, &robust,
                                        &tail, NULL);
   if (!cs || vk_command_buffer_has_error(&cmd->vk))
      return;

   cs->g17p_primitive_bound = UINT64_MAX;

   if (tail) {
      uint32_t draw_state_offset;
      if (!hk_g17p_draw_state_offset(cmd, cs, &draw_state_offset)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
         return;
      }
      memcpy(cs->current, &draw_state_offset, sizeof(draw_state_offset));
      cs->current += sizeof(draw_state_offset);
   }
   /* The record is an Index List block; its Primitive field must name the
    * mesh output topology. With the triangle value, point and line meshes
    * of 22 or more primitives (more than 64 bytes at three index bytes per
    * primitive) lost primitives on hardware. */
   struct agx_vdm_mesh_dispatch_info dispatch = {
      .threads = {3, 1, 1}, .local_size = {3, 1, 1},
      .primitive = mesh_info->prim == MESA_PRIM_POINTS ? 0
                 : mesh_info->prim == MESA_PRIM_LINES  ? 1 : 6,
   };
   if (task) {
      for (unsigned i = 0; i < 3; ++i) {
         const uint64_t threads =
            (uint64_t)gfx->mesh.task_grid[i] * task->b.info.workgroup_size[i];
         if (!threads || threads > UINT32_MAX) {
            vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
            return;
         }
         dispatch.threads[i] = threads;
         dispatch.local_size[i] = task->b.info.workgroup_size[i];
      }
   }
   /* The record's object thread counts follow its header word. */
   const uint64_t dispatch_threads = hk_cs_current_addr(cs) + 4;
   if (!agx_vdm_mesh_dispatch_pack(dev->dev.chip, &dispatch, cs->current)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   cs->current += AGX_VDM_MESH_DISPATCH_LENGTH;
   if (task_gpu_dispatch) {
      libagx_m5_task_dispatch(cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX,
                              dispatch_threads, task_flag, grid, count_address,
                              conditional ? cmd->state.conditional.address : 0,
                              draw_id,
                              conditional && cmd->state.conditional.inverted,
                              task->b.info.workgroup_size[0],
                              task->b.info.workgroup_size[1],
                              task->b.info.workgroup_size[2]);
   }
   hk_m5_note_render_resources(cmd, cs);
   cs->stats.calls++;
   hk_upload_stat(cmd, draws, 1);
   hk_note_pre_raster_images(cmd, cs);
   cs->stats.cmds++;
   vk_dynamic_graphics_state_clear_dirty(&cmd->vk.dynamic_graphics_state);
   gfx->dirty = gfx->shaders_dirty = gfx->linked_dirty = 0;
   gfx->descriptors.root_dirty = false;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMeshTasksEXT(VkCommandBuffer commandBuffer, uint32_t groupCountX,
                      uint32_t groupCountY, uint32_t groupCountZ)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   if (!groupCountX || !groupCountY || !groupCountZ)
      return;
   const uint32_t grid[4] = {groupCountX, groupCountY, groupCountZ, 0};
   hk_draw_mesh_native(cmd, hk_pool_upload(cmd, grid, sizeof(grid), 16), grid,
                       0, 0);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMeshTasksIndirectEXT(VkCommandBuffer commandBuffer, VkBuffer buffer,
                              VkDeviceSize offset, uint32_t drawCount,
                              uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, src, buffer);
   for (unsigned i = 0; i < drawCount; ++i)
      hk_draw_mesh_native(cmd, hk_buffer_address_rw(src, offset + (uint64_t)i * stride),
                          NULL, i, 0);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMeshTasksIndirectCountEXT(VkCommandBuffer commandBuffer, VkBuffer buffer,
                                   VkDeviceSize offset, VkBuffer countBuffer,
                                   VkDeviceSize countBufferOffset,
                                   uint32_t maxDrawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, src, buffer);
   VK_FROM_HANDLE(hk_buffer, count, countBuffer);
   const uint64_t count_address = hk_buffer_address_rw(count, countBufferOffset);
   for (unsigned i = 0; i < maxDrawCount; ++i)
      hk_draw_mesh_native(cmd, hk_buffer_address_rw(src, offset + (uint64_t)i * stride),
                          NULL, i, count_address);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount,
           uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct agx_draw draw;

   if (HK_TEST_INDIRECTS) {
      uint32_t data[] = {
         vertexCount,
         instanceCount,
         firstVertex,
         firstInstance,
      };

      draw = agx_draw_indirect(hk_pool_upload(cmd, data, sizeof(data), 4));
   } else {
      draw = (struct agx_draw){
         .b = agx_3d(vertexCount, instanceCount, 1),
         .start = firstVertex,
         .start_instance = firstInstance,
      };
   }

   hk_draw(cmd, 0, draw);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                   const VkMultiDrawInfoEXT *pVertexInfo,
                   uint32_t instanceCount, uint32_t firstInstance,
                   uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   for (unsigned i = 0; i < drawCount; ++i) {
      struct agx_draw draw = {
         .b = agx_3d(pVertexInfo->vertexCount, instanceCount, 1),
         .start = pVertexInfo->firstVertex,
         .start_instance = firstInstance,
      };

      hk_draw(cmd, i, draw);
      pVertexInfo = ((void *)pVertexInfo) + stride;
   }
}

static void
hk_draw_indexed(VkCommandBuffer commandBuffer, uint16_t draw_id,
                uint32_t indexCount, uint32_t instanceCount,
                uint32_t firstIndex, int32_t vertexOffset,
                uint32_t firstInstance)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct agx_draw draw;
   struct hk_addr_range buf = cmd->state.gfx.index.buffer;

   if (HK_TEST_INDIRECTS && draw_id == 0) {
      uint32_t data[] = {
         indexCount, instanceCount, firstIndex, vertexOffset, firstInstance,
      };
      uint64_t addr = hk_pool_upload(cmd, data, sizeof(data), 4);

      draw = agx_draw_indexed_indirect(addr, buf.addr, buf.range, 0, 0);
   } else {
      draw =
         agx_draw_indexed(indexCount, instanceCount, firstIndex, vertexOffset,
                          firstInstance, buf.addr, buf.range, 0, 0);
   }

   hk_draw(cmd, draw_id, draw);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount,
                  uint32_t instanceCount, uint32_t firstIndex,
                  int32_t vertexOffset, uint32_t firstInstance)
{
   hk_draw_indexed(commandBuffer, 0, indexCount, instanceCount, firstIndex,
                   vertexOffset, firstInstance);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawMultiIndexedEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                          const VkMultiDrawIndexedInfoEXT *pIndexInfo,
                          uint32_t instanceCount, uint32_t firstInstance,
                          uint32_t stride, const int32_t *pVertexOffset)
{
   for (unsigned i = 0; i < drawCount; ++i) {
      const int32_t vertex_offset =
         pVertexOffset != NULL ? *pVertexOffset : pIndexInfo->vertexOffset;

      hk_draw_indexed(commandBuffer, i, pIndexInfo->indexCount, instanceCount,
                      pIndexInfo->firstIndex, vertex_offset, firstInstance);

      pIndexInfo = ((void *)pIndexInfo) + stride;
   }
}

/* hk_draw_impl()'s normalisation of an API indirect draw on G17P. */
static inline void
hk_g17p_normalize_indirect_draw(const struct hk_cmd_buffer *cmd,
                                struct agx_draw *draw)
{
   const struct vk_dynamic_graphics_state *dyn =
      &cmd->vk.dynamic_graphics_state;

   draw->restart = dyn->ia.primitive_restart_enable && draw->indexed;
   draw->index_size = cmd->state.gfx.index.size;
   if (draw->indexed && draw->index_buffer_range_B == 0) {
      draw->index_buffer = AGX_ZERO_PAGE_ADDRESS;
      draw->index_buffer_range_B = 4;
      draw->start = 0;
   }
}

/*
 * Records of an indirect (multi-)draw on G17P or M5 whose draws would find nothing
 * to do in hk_draw_impl() / hk_flush_gfx_state() /
 * hk_flush_dynamic_state_slow() except their own publication: no dirty
 * dynamic, shader, link, root, descriptor or vertex-buffer state, and none of
 * the transformed, conditional, statistics, multiview, split or trace paths.
 * Ordinary emission also excludes robustness. In that state the general path
 * reduces to exactly the sequence
 * in hk_g17p_emit_indirect_record() -- stream space, the record's draw
 * parameters, the vertex-program publication (hk_agx3_publish_vertex_state())
 * when the vertex shader reads draw parameters, then the draw record
 * (hk_g17p_emit_draw()) -- so ordinary records are emitted directly, byte for
 * byte what the general path emits. With allow_robust, the first record has
 * already queued a repair job for this API multi-draw. The same proof admits
 * later records' state publication followed by a reserved repair slot instead
 * (hk_emit_robust_index); no other compute work or stream split intervenes.
 *
 * On M5 the general path also derives the HAL300 graphics core, which is then
 * the one the stream already latched (hk_cs.m5_ppp_emitted), so it emits
 * nothing there either.
 *
 * The check reads no per-record value except the indirect address, and
 * emitting a record changes none of what it reads, so a multi-draw checks it
 * once (draw is already normalised).  It has no side effects.
 */
static bool
hk_g17p_indirect_records_eligible(struct hk_cmd_buffer *cmd,
                                  const struct agx_draw *draw,
                                  bool allow_robust)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   assert(agx_is_indirect(draw->b));

   /* Nothing dirty: every state-derivation block of the general path is
    * then a no-op, and hk_flush_shaders() returns before doing anything. */
   if (gfx->shaders_dirty || gfx->linked_dirty || gfx->dirty ||
       desc->root_dirty || desc->push_dirty ||
       vk_dynamic_graphics_state_any_dirty(dyn) ||
       !gfx->fast_link_vs.valid || !gfx->fast_link_fs.valid)
      return false;

   struct hk_api_shader *fs_obj = gfx->shaders[MESA_SHADER_FRAGMENT];
   struct hk_linked_shader *linked_vs = gfx->linked[MESA_SHADER_VERTEX];
   if (!fs_obj || !gfx->shaders[MESA_SHADER_VERTEX] || !linked_vs ||
       !gfx->linked[MESA_SHADER_FRAGMENT] || linked_vs->sw_indexing ||
       gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] || gfx->render.view_mask ||
       mesa_prim_has_adjacency(
          vk_topology_to_mesa(dyn->ia.primitive_topology)))
      return false;

   if (hk_conditional_render_is_active(&cmd->state.conditional, cmd->in_meta,
                                       cmd->conditional_meta) ||
       (desc->root.draw.pipeline_stats_flags &
        (VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT |
         VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
         VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
         VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
         VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT)) ||
       hk_prims_generated_active(cmd) || dyn->rs.depth_bias.enable ||
       (dyn->ms.rasterization_samples && !gfx->render.tilebuffer.nr_samples))
      return false;

   /* hk_draw_impl()'s index validation, and the general path's refusals. */
   if (draw->indexed) {
      struct agx_draw robust_draw = *draw;
      if ((draw->index_size != AGX_INDEX_SIZE_U8 &&
           draw->index_size != AGX_INDEX_SIZE_U16 &&
           draw->index_size != AGX_INDEX_SIZE_U32) ||
          !draw->index_buffer ||
          (!allow_robust && hk_needs_index_robustness(cmd, &robust_draw, NULL)))
         return false;
   }

   return agx_vdm_draw_supported(dev->dev.chip, *draw, gfx->topology);
}

/* Emit one eligible record (see hk_g17p_indirect_records_eligible()). */
static void
hk_g17p_emit_indirect_record(struct hk_cmd_buffer *cmd, uint16_t draw_id,
                             struct agx_draw draw, bool republishes,
                             bool continuing,
                             struct hk_robust_index_batch *batch)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct hk_descriptor_state *desc = &gfx->descriptors;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_cs *cs = cmd->current_cs.gfx;

   /* An eligible multi-draw may emit even its first record here, so this
    * direct path must record depth/stencil use exactly as hk_draw_impl()
    * does for ZLS elision. */
   hk_g17p_note_depth_use(cmd);

   /* The general path from here on, with its no-op branches removed. */
   hk_ensure_cs_has_space(cmd, cs, HK_GFX_DRAW_RESERVATION);
   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   /* An indirect count is not known when the draw is recorded. */
   cs->g17p_primitive_bound = UINT64_MAX;

   bool tail_emitted = false;
   if (republishes) {
      struct hk_linked_shader *linked_vs = gfx->linked[MESA_SHADER_VERTEX];
      struct hk_shader *hw_vs = hk_bound_hw_vs(gfx);
      struct hk_shader *sw_vs = hk_bound_sw_vs(gfx);

      uint8_t *out = cs->current;

      if (linked_vs->b.uses_base_param) {
         gfx->draw_params =
            draw.b.ptr +
            (draw.indexed ? offsetof(VkDrawIndexedIndirectCommand, vertexOffset)
                          : offsetof(VkDrawIndirectCommand, firstVertex));
      } else {
         gfx->draw_params = 0;
      }
      gfx->g17p_draw_params_host_valid = false;

      if (sw_vs->b.info.uses_draw_id) {
         hk_cmd_upload_draw_id(cmd, draw_id);
      } else {
         gfx->draw_id_ptr = 0;
         gfx->g17p_draw_id_host_valid = false;
      }

      /* Only the native bind tail consumes this (hk_agx3_publish_vertex_state),
       * as in hk_flush_dynamic_state_slow(). */
      const bool prepared_varyings =
         agx_native_graphics(&dev->dev) &&
         hk_g17p_prepared_bind_tail_matches(cmd, &gfx->g17p_bind_tail);
      const uint64_t prepared_varyings_root = gfx->root;

      hk_reserve_scratch(cmd, cs, hw_vs);
      struct hk_linked_shader *linked_hw_vs =
         (hw_vs == sw_vs) ? linked_vs : hw_vs->only_linked;
      /* A record after the first of a multi-draw republishes the previous
       * record's program with its own draw parameters. */
      const uint64_t pipeline_program =
         continuing && (agx_chip_native_agx3(dev->dev.chip) ||
                        agx_chip_g17_m5(dev->dev.chip))
            ? hk_m5_multi_draw_vertex_program(cmd, hw_vs, linked_hw_vs)
            : hk_upload_usc_words(cmd, hw_vs, linked_hw_vs);

      if (!hk_agx3_publish_vertex_state(
             cmd, cs, &out, hw_vs, pipeline_program, draw, true,
             prepared_varyings && gfx->root == prepared_varyings_root,
             &tail_emitted))
         return;

      cs->current = out;
      if (vk_command_buffer_has_error(&cmd->vk))
         return;

      vk_dynamic_graphics_state_clear_dirty(dyn);
      gfx->shaders_dirty = 0;
      gfx->linked_dirty = 0;
      gfx->dirty = 0;
      desc->root_dirty = false;
   }

   if (vk_command_buffer_has_error(&cmd->vk))
      return;

   cs->stats.calls++;
   hk_upload_stat(cmd, draws, 1);
   if (batch) {
      hk_emit_robust_index(cmd, cs, draw, gfx->topology, tail_emitted, batch);
   } else if (agx_native_graphics(&hk_cmd_buffer_device(cmd)->dev)) {
      if (!hk_g17p_emit_draw(cmd, cs, cs->current, draw, gfx->topology,
                             tail_emitted))
         return;
   } else {
      /* As hk_draw_impl() emits an indirect HAL300 draw: the packet carries
       * its own restart state. */
      cs->current = (void *)agx_vdm_draw((uint32_t *)cs->current,
                                         hk_cmd_buffer_device(cmd)->dev.chip,
                                         draw, gfx->topology,
                                         hk_draw_restart_index(cmd, draw));
   }
   hk_g17p_note_layer_zero_draw(cmd, cs);
   hk_note_pre_raster_images(cmd, cs);
   hk_m5_note_render_resources(cmd, cs);
   if (unlikely(dev->frame_anatomy)) {
      hk_frame_anatomy_note_fs(cs, hk_bound_fs(cmd));
      hk_frame_anatomy_note_vs(cs, hk_bound_hw_vs(gfx));
      hk_frame_anatomy_note_m5_fragment(cs);
   }
   cs->stats.cmds++;
}

/* Record one indirect record: directly when the multi-draw is eligible
 * (checked once, on the first record that is), otherwise through hk_draw(). */
static inline void
hk_draw_indirect_record(struct hk_cmd_buffer *cmd, uint16_t draw_id,
                        struct agx_draw draw, int *eligible)
{
   if (*eligible != 0) {
      struct agx_draw normalized = draw;
      hk_g17p_normalize_indirect_draw(cmd, &normalized);
      /* The previous record of this multi-draw was emitted directly, with
       * nothing recorded since. */
      const bool continuing = *eligible > 0;
      if (*eligible < 0) {
         *eligible =
            hk_g17p_indirect_records_eligible(cmd, &normalized, false) ? 1 : -1;
      }
      /* A zero indirect address is refused by the general path. */
      if (*eligible > 0 && normalized.b.ptr &&
          !vk_command_buffer_has_error(&cmd->vk)) {
         hk_g17p_emit_indirect_record(cmd, draw_id, normalized,
                                      hk_gfx_republishes_render_state(cmd),
                                      continuing, NULL);
         return;
      }
   }
   hk_draw(cmd, draw_id, draw);
}

static void
hk_draw_indirect_inner(VkCommandBuffer commandBuffer, uint64_t base,
                       uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndirect-drawCount-00476
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to sizeof(VkDrawIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndirectCommand));
   }

   const struct hk_device *dev = hk_cmd_buffer_device(cmd);
   int eligible =
      agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip) ? -1 : 0;
   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      uint64_t addr = base + stride * draw_id;
      hk_draw_indirect_record(cmd, draw_id, agx_draw_indirect(addr),
                              &eligible);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                   VkDeviceSize offset, uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   hk_draw_indirect_inner(commandBuffer, hk_buffer_address_ro(buffer, offset),
                          drawCount, stride);
}

/* The first record derives/flushes state through the general path. Once it
 * has reached robustness, no helper follows it. Later records can join only
 * when the existing indirect-record proof rules out all other compute work
 * and stream splits. Otherwise publish the first repair before falling back.
 * This also covers DrawIndexedIndirectCount after its count predication. */
static bool
hk_g17p_draw_robust_multi(struct hk_cmd_buffer *cmd, uint64_t buffer,
                          uint32_t draw_count, uint32_t stride)
{
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   if (!agx_native_graphics(&dev->dev) || draw_count <= 1)
      return false;

   struct hk_graphics_state *gfx = &cmd->state.gfx;
   struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct hk_addr_range buf = gfx->index.buffer;
   struct agx_draw first =
      agx_draw_indexed_indirect(buffer, buf.addr, buf.range, 0, 0);
   hk_g17p_normalize_indirect_draw(cmd, &first);

   if (gfx->render.view_mask || dyn->rs.depth_bias.enable ||
       gfx->shaders[MESA_SHADER_GEOMETRY] ||
       gfx->shaders[MESA_SHADER_TESS_CTRL] ||
       gfx->shaders[MESA_SHADER_TESS_EVAL] ||
       mesa_prim_has_adjacency(
          vk_topology_to_mesa(dyn->ia.primitive_topology)) ||
       hk_conditional_render_is_active(&cmd->state.conditional, cmd->in_meta,
                                       cmd->conditional_meta) ||
       gfx->descriptors.root.draw.pipeline_stats_flags ||
       !hk_needs_index_robustness(cmd, &first, NULL))
      return false;

   /* maxDrawIndirectCount is UINT16_MAX; the grid has exactly one 32-lane
    * workgroup per record and neither the byte allocation nor grid wraps. */
   assert(draw_count <= UINT16_MAX);
   struct hk_robust_index_batch batch = {
      .jobs = hk_pool_alloc(
         cmd, (size_t)draw_count * sizeof(struct libagx_robust_index_draw), 8),
   };
   if (!batch.jobs.cpu)
      return true;

   hk_draw_impl(cmd, 0, first, &batch);
   if (vk_command_buffer_has_error(&cmd->vk))
      return true;

   const bool eligible =
      batch.count == 1 && hk_g17p_indirect_records_eligible(cmd, &first, true);
   if (!eligible)
      hk_flush_robust_index_batch(cmd, &batch);

   for (uint32_t i = 1; i < draw_count; ++i) {
      struct agx_draw draw = first;
      draw.b.ptr = buffer + (uint64_t)stride * i;
      if (eligible) {
         hk_g17p_emit_indirect_record(
            cmd, i, draw, hk_gfx_republishes_render_state(cmd), i > 1, &batch);
      } else {
         hk_draw(cmd, i, draw);
      }
      if (vk_command_buffer_has_error(&cmd->vk))
         return true;
   }

   hk_flush_robust_index_batch(cmd, &batch);
   return true;
}

static void
hk_draw_indexed_indirect_inner(VkCommandBuffer commandBuffer, uint64_t buffer,
                               uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   /* From the Vulkan 1.3.238 spec:
    *
    *    VUID-vkCmdDrawIndexedIndirect-drawCount-00528
    *
    *    "If drawCount is greater than 1, stride must be a multiple of 4 and
    *    must be greater than or equal to
    * sizeof(VkDrawIndexedIndirectCommand)"
    *
    * and
    *
    *    "If drawCount is less than or equal to one, stride is ignored."
    */
   if (drawCount > 1) {
      assert(stride % 4 == 0);
      assert(stride >= sizeof(VkDrawIndexedIndirectCommand));
   }

   if (hk_g17p_draw_robust_multi(cmd, buffer, drawCount, stride))
      return;

   const struct hk_device *dev = hk_cmd_buffer_device(cmd);
   int eligible =
      agx_native_graphics(&dev->dev) || agx_chip_g17_m5(dev->dev.chip) ? -1 : 0;
   for (unsigned draw_id = 0; draw_id < drawCount; ++draw_id) {
      uint64_t addr = buffer + stride * draw_id;
      struct hk_addr_range buf = cmd->state.gfx.index.buffer;

      hk_draw_indirect_record(
         cmd, draw_id,
         agx_draw_indexed_indirect(addr, buf.addr, buf.range, 0, 0),
         &eligible);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                          VkDeviceSize offset, uint32_t drawCount,
                          uint32_t stride)
{
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);

   hk_draw_indexed_indirect_inner(
      commandBuffer, hk_buffer_address_ro(buffer, offset), drawCount, stride);
}

/*
 * To implement drawIndirectCount generically, we dispatch a compute kernel to
 * patch the indirect buffer and then we dispatch the predicated maxDrawCount
 * indirect draws.
 */
static void
hk_draw_indirect_count(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                       VkDeviceSize offset, VkBuffer countBuffer,
                       VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                       uint32_t stride, bool indexed)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, buffer, _buffer);
   VK_FROM_HANDLE(hk_buffer, count_buffer, countBuffer);

   perf_debug(cmd, "Draw indirect count");
   hk_m5_pass_merge_end_ordered(cmd, "indirect count draw");

   assert((stride % 4) == 0 && "aligned");

   /* The Vulkan commands are defined as no-ops for a zero maximum. Avoid a
    * zero-sized pool allocation and a zero-grid internal dispatch. */
   if (maxDrawCount == 0)
      return;

   size_t out_stride = sizeof(uint32_t) * (indexed ? 5 : 4);
   uint64_t patched = hk_pool_alloc(cmd, out_stride * maxDrawCount, 4).gpu;
   uint64_t in = hk_buffer_address_ro(buffer, offset);
   uint64_t count_addr = hk_buffer_address_ro(count_buffer, countBufferOffset);

   /* The predicate below reads both buffers before the draw is recorded. */
   hk_xfer_check_draw(cmd, in, in + (uint64_t)stride * (maxDrawCount - 1) +
                                  out_stride);
   hk_xfer_check_draw(cmd, count_addr, count_addr + sizeof(uint32_t));

   libagx_predicate_indirect(cmd, agx_1d(maxDrawCount),
                             AGX_BARRIER_ALL | AGX_PREGFX, patched, in,
                             count_addr, stride / 4, indexed);

   if (indexed) {
      hk_draw_indexed_indirect_inner(commandBuffer, patched, maxDrawCount,
                                     out_stride);
   } else {
      hk_draw_indirect_inner(commandBuffer, patched, maxDrawCount, out_stride);
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                        VkDeviceSize offset, VkBuffer countBuffer,
                        VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                        uint32_t stride)
{
   hk_draw_indirect_count(commandBuffer, _buffer, offset, countBuffer,
                          countBufferOffset, maxDrawCount, stride, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                               VkDeviceSize offset, VkBuffer countBuffer,
                               VkDeviceSize countBufferOffset,
                               uint32_t maxDrawCount, uint32_t stride)
{
   hk_draw_indirect_count(commandBuffer, _buffer, offset, countBuffer,
                          countBufferOffset, maxDrawCount, stride, true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdDrawIndirectByteCountEXT(VkCommandBuffer commandBuffer,
                               uint32_t instanceCount, uint32_t firstInstance,
                               VkBuffer counterBuffer,
                               VkDeviceSize counterBufferOffset,
                               uint32_t counterOffset, uint32_t vertexStride)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(hk_buffer, counter, counterBuffer);

   perf_debug(cmd, "Draw indirect byte count");
   hk_m5_pass_merge_end_ordered(cmd, "indirect byte count draw");

   const uint64_t draw =
      hk_pool_alloc(cmd, sizeof(VkDrawIndirectCommand), 4).gpu;
   const uint64_t count =
      hk_buffer_address_ro(counter, counterBufferOffset);

   libagx_draw_indirect_byte_count(
      cmd, agx_1d(1), AGX_BARRIER_ALL | AGX_PREGFX, draw, count,
      counterOffset, vertexStride, instanceCount, firstInstance);

   hk_draw_indirect_inner(commandBuffer, draw, 1,
                          sizeof(VkDrawIndirectCommand));
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBindTransformFeedbackBuffersEXT(VkCommandBuffer commandBuffer,
                                      uint32_t firstBinding,
                                      uint32_t bindingCount,
                                      const VkBuffer *pBuffers,
                                      const VkDeviceSize *pOffsets,
                                      const VkDeviceSize *pSizes)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   for (uint32_t i = 0; i < bindingCount; i++) {
      VK_FROM_HANDLE(hk_buffer, buffer, pBuffers[i]);
      uint32_t idx = firstBinding + i;
      uint64_t size = pSizes ? pSizes[i] : VK_WHOLE_SIZE;

      gfx->xfb[idx] = hk_buffer_addr_range(buffer, pOffsets[i], size, false);
   }
}

static void
hk_begin_end_xfb(VkCommandBuffer commandBuffer, uint32_t firstCounterBuffer,
                 uint32_t counterBufferCount, const VkBuffer *pCounterBuffers,
                 const VkDeviceSize *pCounterBufferOffsets, bool begin)

{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_graphics_state *gfx = &cmd->state.gfx;

   /* The counter copies write application buffers ahead of the render. */
   hk_m5_pass_merge_end_ordered(cmd, "transform feedback");
   gfx->xfb_enabled = begin;

   /* If we haven't reserved XFB offsets yet for the command buffer, do so. */
   if (!gfx->xfb_offsets) {
      gfx->xfb_offsets = hk_pool_alloc(cmd, 4 * sizeof(uint32_t), 4).gpu;
   }

   struct libagx_xfb_counter_copy params = {};
   unsigned copies = 0;

   /* For CmdBeginTransformFeedbackEXT, we need to initialize everything */
   if (begin) {
      for (copies = 0; copies < 4; ++copies) {
         params.dest[copies] = gfx->xfb_offsets + copies * sizeof(uint32_t);
      }
   }

   /* pCounterBuffers may be NULL with a non-zero count: no counter buffers. */
   for (unsigned i = 0; pCounterBuffers && i < counterBufferCount; ++i) {
      if (pCounterBuffers[i] == VK_NULL_HANDLE)
         continue;

      VK_FROM_HANDLE(hk_buffer, buffer, pCounterBuffers[i]);

      uint64_t offset = pCounterBufferOffsets ? pCounterBufferOffsets[i] : 0;
      uint64_t cb_addr = hk_buffer_address_rw(buffer, offset);
      uint32_t cmd_idx = firstCounterBuffer + i;

      if (begin) {
         params.src[cmd_idx] = cb_addr;
      } else {
         params.dest[copies] = cb_addr;
         params.src[copies] = gfx->xfb_offsets + cmd_idx * sizeof(uint32_t);
         ++copies;
      }
   }

   if (begin)
      copies = 4;

   if (copies > 0) {
      perf_debug(cmd, "XFB counter copy");

      libagx_copy_xfb_counters(cmd, agx_1d(copies),
                               AGX_BARRIER_ALL | AGX_PREGFX,
                               hk_pool_upload(cmd, &params, sizeof(params), 8));
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                                uint32_t firstCounterBuffer,
                                uint32_t counterBufferCount,
                                const VkBuffer *pCounterBuffers,
                                const VkDeviceSize *pCounterBufferOffsets)
{
   hk_begin_end_xfb(commandBuffer, firstCounterBuffer, counterBufferCount,
                    pCounterBuffers, pCounterBufferOffsets, true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndTransformFeedbackEXT(VkCommandBuffer commandBuffer,
                              uint32_t firstCounterBuffer,
                              uint32_t counterBufferCount,
                              const VkBuffer *pCounterBuffers,
                              const VkDeviceSize *pCounterBufferOffsets)
{
   hk_begin_end_xfb(commandBuffer, firstCounterBuffer, counterBufferCount,
                    pCounterBuffers, pCounterBufferOffsets, false);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdBeginConditionalRenderingEXT(
   VkCommandBuffer commandBuffer,
   const VkConditionalRenderingBeginInfoEXT *pConditionalRenderingBegin)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   /* Predicated draws patch their records in pre-graphics compute. */
   hk_m5_pass_merge_end_ordered(cmd, "conditional rendering");
   hk_flush_deferred_clears(cmd);
   VK_FROM_HANDLE(hk_buffer, buffer, pConditionalRenderingBegin->buffer);
   struct hk_conditional_render_state *conditional =
      &cmd->state.conditional;

   assert(!conditional->active);
   conditional->address =
      hk_buffer_address_ro(buffer, pConditionalRenderingBegin->offset);
   conditional->active = true;
   conditional->inverted =
      pConditionalRenderingBegin->flags &
      VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT;

   if (!conditional->address)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdEndConditionalRenderingEXT(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   struct hk_conditional_render_state *conditional =
      &cmd->state.conditional;

   assert(conditional->active);
   const bool inherited = conditional->inherited;
   *conditional = (struct hk_conditional_render_state){
      .inherited = inherited,
   };
}

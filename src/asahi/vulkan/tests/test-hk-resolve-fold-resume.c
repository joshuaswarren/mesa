/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* CPU-only unit tests for AGX3 trailing MSAA resolve folding FS-barrier
 * hazard detection and in-CB suspended/resumed direct-resolve gating through
 * libhk. */
#undef NDEBUG
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "asahi/lib/agx_tilebuffer.h"
#include "hk_agx3_ioto_program_core.h"
#include "hk_cmd_buffer.h"
#include "hk_cmd_pool.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_image_view.h"
#include "hk_pass_merge.h"

static unsigned checks;

#define CHECK(cond)                                                            \
   do {                                                                        \
      checks++;                                                                \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
         return 1;                                                             \
      }                                                                        \
   } while (0)

struct test_fixture {
   struct hk_device dev;
   struct hk_cmd_pool pool;
   struct hk_cmd_buffer cmd;
   struct hk_cs cs;
   uint8_t stream[0x1000];
};

static void
fixture_init(struct test_fixture *f)
{
   memset(f, 0, sizeof(*f));
   f->dev.vk.base.type = VK_OBJECT_TYPE_DEVICE;
   f->dev.dev.chip = AGX_CHIP_G17P;
   f->pool.vk.base.type = VK_OBJECT_TYPE_COMMAND_POOL;
   f->pool.vk.base.device = &f->dev.vk;
   f->cmd.vk.base.type = VK_OBJECT_TYPE_COMMAND_BUFFER;
   f->cmd.vk.base.device = &f->dev.vk;
   f->cmd.vk.pool = &f->pool.vk;
   f->cmd.vk.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   list_inithead(&f->cmd.control_streams);
   util_dynarray_init(&f->cmd.sync_pass_records, NULL);
   util_dynarray_init(&f->cmd.sparse_sites, NULL);

   f->cs.cmd = &f->cmd;
   f->cs.type = HK_CS_VDM;
   f->cs.start = f->stream;
   f->cs.current = f->stream + 64;
   f->cs.end = f->stream + sizeof(f->stream);
   f->cs.cr.width = 128;
   f->cs.cr.height = 128;
   f->cs.cr.layers = 1;
   f->cs.stats.cmds = 1;
   list_addtail(&f->cs.node, &f->cmd.control_streams);
}

static void
fixture_fini(struct test_fixture *f)
{
   util_dynarray_fini(&f->cmd.sync_pass_records);
   util_dynarray_fini(&f->cmd.sparse_sites);
}

static void
init_2d_image(struct hk_image *img, VkFormat format,
              VkSampleCountFlagBits samples, uint32_t w, uint32_t h)
{
   memset(img, 0, sizeof(*img));
   img->vk.base.type = VK_OBJECT_TYPE_IMAGE;
   img->vk.image_type = VK_IMAGE_TYPE_2D;
   img->vk.format = format;
   img->vk.samples = samples;
   img->vk.extent = (VkExtent3D){w, h, 1};
   img->vk.mip_levels = 1;
   img->vk.array_layers = 1;
   img->plane_count = 1;
   img->planes[0].addr = 0x100000;
   img->planes[0].layout.size_B = 0x10000;
}

static void
arm_pending_resolve(struct test_fixture *f, struct hk_image *src,
                    bool fs_reads_images)
{
   f->cs.fs_reads_images = fs_reads_images;
   f->cmd.sync.in_pass = false;
   f->cmd.sync.next_seq = 7;
   f->cmd.state.pending_msaa_resolve = (struct hk_pending_msaa_resolve){
      .cs = &f->cs,
      .src_image = src,
      .src_format = src->vk.format,
      .src_load_op = VK_ATTACHMENT_LOAD_OP_LOAD,
      .src_mip_level = 0,
      .src_base_layer = 0,
      .next_seq = f->cmd.sync.next_seq,
      .fs_barrier_seen = false,
   };
}

static VkResolveImageInfo2
make_resolve_info(struct hk_image *src, struct hk_image *dst,
                  VkImageResolve2 *region)
{
   *region = (VkImageResolve2){
      .sType = VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2,
      .srcSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = 0,
         .baseArrayLayer = 0,
         .layerCount = 1,
      },
      .dstSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = 0,
         .baseArrayLayer = 0,
         .layerCount = 1,
      },
      .extent = {128, 128, 1},
   };
   return (VkResolveImageInfo2){
      .sType = VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2,
      .srcImage = hk_image_to_handle(src),
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .dstImage = hk_image_to_handle(dst),
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount = 1,
      .pRegions = region,
   };
}

static int
test_resolve_fold_fs_barriers(void)
{
   struct hk_image src, dst;
   init_2d_image(&src, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_4_BIT, 128, 128);
   init_2d_image(&dst, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, 128, 128);

   /* Ryan's WAR scenario: 4x pass with FS sampling dst, followed by a
    * COLOR_ATTACHMENT_OUTPUT -> TRANSFER image barrier transitioning src to
    * TRANSFER_SRC and dst SHADER_READ_ONLY -> TRANSFER_DST, then full-image
    * vkCmdResolveImage2. The barrier orders FS reads, so folding must refuse. */
   {
      struct test_fixture f;
      fixture_init(&f);
      arm_pending_resolve(&f, &src, true);

      VkImageMemoryBarrier2 img_bars[2] = {
         {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_RESOLVE_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .image = hk_image_to_handle(&src),
         },
         {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstStageMask = VK_PIPELINE_STAGE_2_RESOLVE_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = hk_image_to_handle(&dst),
         },
      };
      VkDependencyInfo dep = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .imageMemoryBarrierCount = 2,
         .pImageMemoryBarriers = img_bars,
      };
      hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(&f.cmd), &dep);
      CHECK(f.cmd.state.pending_msaa_resolve.fs_barrier_seen);
      CHECK(hk_resolve_fold_fs_hazard(
         f.cs.fs_reads_images, f.cmd.state.pending_msaa_resolve.fs_barrier_seen));

      VkImageResolve2 region;
      VkResolveImageInfo2 resolve = make_resolve_info(&src, &dst, &region);
      CHECK(!hk_try_fold_msaa_resolve(&f.cmd, &resolve));
      CHECK(f.cmd.state.pending_msaa_resolve.cs == NULL);
      fixture_fini(&f);
   }

   /* Every source stage at or after the fragment shader (and global stages)
    * must mark fs_barrier_seen across image, memory, and buffer barriers. */
   static const VkPipelineStageFlags2 fs_ordering_stages[] = {
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
      VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(fs_ordering_stages); ++i) {
      const VkPipelineStageFlags2 st = fs_ordering_stages[i];

      /* Memory barrier */
      {
         struct test_fixture f;
         fixture_init(&f);
         arm_pending_resolve(&f, &src, true);
         VkMemoryBarrier2 mb = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = st,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
         };
         VkDependencyInfo dep = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &mb,
         };
         hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(&f.cmd), &dep);
         CHECK(f.cmd.state.pending_msaa_resolve.fs_barrier_seen);
         VkImageResolve2 region;
         VkResolveImageInfo2 resolve = make_resolve_info(&src, &dst, &region);
         CHECK(!hk_try_fold_msaa_resolve(&f.cmd, &resolve));
         fixture_fini(&f);
      }

      /* Buffer barrier */
      {
         struct test_fixture f;
         fixture_init(&f);
         arm_pending_resolve(&f, &src, true);
         VkBufferMemoryBarrier2 bb = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = st,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
         };
         VkDependencyInfo dep = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &bb,
         };
         hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(&f.cmd), &dep);
         CHECK(f.cmd.state.pending_msaa_resolve.fs_barrier_seen);
         fixture_fini(&f);
      }

      /* Image barrier */
      {
         struct test_fixture f;
         fixture_init(&f);
         arm_pending_resolve(&f, &src, true);
         VkImageMemoryBarrier2 ib = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = st,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .image = hk_image_to_handle(&src),
         };
         VkDependencyInfo dep = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &ib,
         };
         hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(&f.cmd), &dep);
         CHECK(f.cmd.state.pending_msaa_resolve.fs_barrier_seen);
         fixture_fini(&f);
      }
   }

   /* Multi-barrier dependency where only the second barrier orders FS. */
   {
      struct test_fixture f;
      fixture_init(&f);
      arm_pending_resolve(&f, &src, true);
      VkImageMemoryBarrier2 img_bars[2] = {
         {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .image = hk_image_to_handle(&src),
         },
         {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .image = hk_image_to_handle(&dst),
         },
      };
      VkDependencyInfo dep = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .imageMemoryBarrierCount = 2,
         .pImageMemoryBarriers = img_bars,
      };
      hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(&f.cmd), &dep);
      CHECK(f.cmd.state.pending_msaa_resolve.fs_barrier_seen);
      fixture_fini(&f);
   }

   /* Earlier / non-graphics source stages must NOT mark fs_barrier_seen. */
   static const VkPipelineStageFlags2 non_fs_stages[] = {
      VK_PIPELINE_STAGE_2_NONE,
      VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
      VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      VK_PIPELINE_STAGE_2_COPY_BIT,
      VK_PIPELINE_STAGE_2_RESOLVE_BIT,
      VK_PIPELINE_STAGE_2_HOST_BIT,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(non_fs_stages); ++i) {
      struct test_fixture f;
      fixture_init(&f);
      arm_pending_resolve(&f, &src, true);
      VkMemoryBarrier2 mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = non_fs_stages[i],
         .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
      };
      VkDependencyInfo dep = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .memoryBarrierCount = 1,
         .pMemoryBarriers = &mb,
      };
      hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(&f.cmd), &dep);
      CHECK(!f.cmd.state.pending_msaa_resolve.fs_barrier_seen);
      CHECK(!hk_resolve_fold_fs_hazard(
         f.cs.fs_reads_images, f.cmd.state.pending_msaa_resolve.fs_barrier_seen));
      fixture_fini(&f);
   }

   /* Pass whose FS reads no images (e.g. procedural MSAA scene) still sets
    * fs_barrier_seen on COLOR_ATTACHMENT_OUTPUT, but does not trigger the FS
    * WAR hazard so EOT resolve folding stays eligible. */
   {
      struct test_fixture f;
      fixture_init(&f);
      arm_pending_resolve(&f, &src, false);
      VkImageMemoryBarrier2 ib = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .image = hk_image_to_handle(&src),
      };
      VkDependencyInfo dep = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .imageMemoryBarrierCount = 1,
         .pImageMemoryBarriers = &ib,
      };
      hk_CmdPipelineBarrier2(hk_cmd_buffer_to_handle(&f.cmd), &dep);
      CHECK(f.cmd.state.pending_msaa_resolve.fs_barrier_seen);
      CHECK(!hk_resolve_fold_fs_hazard(
         f.cs.fs_reads_images, f.cmd.state.pending_msaa_resolve.fs_barrier_seen));
      fixture_fini(&f);
   }

   return 0;
}

static void
setup_suspended_pass(struct test_fixture *f, VkRenderingFlagBits prev_flags,
                     struct hk_image_view *rt0_view,
                     struct hk_image_view *rt0_resolve,
                     struct hk_image_view *rt1_view,
                     VkAttachmentStoreOp rt0_store_op)
{
   struct hk_pass_merge_candidate *m = &f->cmd.suspended_pass;
   memset(m, 0, sizeof(*m));
   m->valid = true;
   m->cs = &f->cs;
   m->resume_current = f->stream + 32;
   m->after_current = f->cs.current;
   m->end = f->cs.end;
   m->next_seq = f->cmd.sync.next_seq;
   m->sync_epoch = f->cmd.sync.epoch;

   struct hk_rendering_state *prev = &m->render;
   prev->flags = prev_flags;
   prev->area = (VkRect2D){.offset = {0, 0}, .extent = {128, 128}};
   prev->layer_count = 1;
   prev->view_mask = 0;
   prev->color_att_count = 2;
   prev->g17p_direct_color_resolve = false;
   prev->zls_first_cs = &f->cs;
   prev->color_written = BITFIELD_BIT(0); /* RT0 written, RT1 untouched */

   prev->color_att[0] = (struct hk_attachment){
      .vk_format = VK_FORMAT_R8G8B8A8_UNORM,
      .iview = rt0_view,
      .load_op = VK_ATTACHMENT_LOAD_OP_LOAD,
      .store_op = rt0_store_op,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .resolve_mode = VK_RESOLVE_MODE_AVERAGE_BIT,
      .resolve_iview = rt0_resolve,
   };
   prev->color_att[1] = (struct hk_attachment){
      .vk_format = VK_FORMAT_R8G8B8A8_UNORM,
      .iview = rt1_view,
      .load_op = VK_ATTACHMENT_LOAD_OP_LOAD,
      .store_op = VK_ATTACHMENT_STORE_OP_STORE,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .resolve_mode = VK_RESOLVE_MODE_NONE,
   };

   enum pipe_format fmts[2] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   prev->tilebuffer = agx_build_tilebuffer_layout(fmts, 2, 4, false);
   prev->cr.width = 128;
   prev->cr.height = 128;
   prev->cr.layers = 1;
   prev->cr.whole_attachments = true;
   f->cs.cr.process_empty_tiles = false;
}

static int
test_in_cb_resume_forbids_direct_resolve(void)
{
   struct hk_image rt0_img, rt0_res_img, rt1_img;
   init_2d_image(&rt0_img, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_4_BIT, 128, 128);
   init_2d_image(&rt0_res_img, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, 128, 128);
   init_2d_image(&rt1_img, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_4_BIT, 128, 128);

   struct hk_image_view rt0_view = {0}, rt0_resolve = {0}, rt1_view = {0};
   rt0_view.vk.base.type = VK_OBJECT_TYPE_IMAGE_VIEW;
   rt0_view.vk.image = &rt0_img.vk;
   rt0_view.vk.format = VK_FORMAT_R8G8B8A8_UNORM;
   rt0_view.planes[0].eot_pbe_valid = true;
   rt0_resolve.vk.base.type = VK_OBJECT_TYPE_IMAGE_VIEW;
   rt0_resolve.vk.image = &rt0_res_img.vk;
   rt0_resolve.vk.format = VK_FORMAT_R8G8B8A8_UNORM;
   rt0_resolve.planes[0].eot_pbe_valid = true;
   rt1_view.vk.base.type = VK_OBJECT_TYPE_IMAGE_VIEW;
   rt1_view.vk.image = &rt1_img.vk;
   rt1_view.vk.format = VK_FORMAT_R8G8B8A8_UNORM;
   rt1_view.planes[0].eot_pbe_valid = true;

   /* Ryan's in-CB resume scenario: BeginRendering(SUSPENDING) with RT0 4x
    * LOAD + AVERAGE resolve and untouched RT1 4x LOAD/STORE; EndRendering;
    * BeginRendering(RESUMING) same info. Reopening drops SUSPENDING and never
    * carries RESUMING, so resumed_in_cb must keep direct-resolve forbidden
    * during end-of-pass elision EOT rebuilds. */
   {
      struct test_fixture f;
      fixture_init(&f);
      setup_suspended_pass(&f, VK_RENDERING_SUSPENDING_BIT, &rt0_view,
                           &rt0_resolve, &rt1_view,
                           VK_ATTACHMENT_STORE_OP_DONT_CARE);

      VkRenderingAttachmentInfo color_atts[2] = {
         {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = hk_image_view_to_handle(&rt0_view),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT,
            .resolveImageView = hk_image_view_to_handle(&rt0_resolve),
            .resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         },
         {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = hk_image_view_to_handle(&rt1_view),
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         },
      };
      VkRenderingInfo resume_info = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .flags = VK_RENDERING_RESUMING_BIT,
         .renderArea = {.offset = {0, 0}, .extent = {128, 128}},
         .layerCount = 1,
         .colorAttachmentCount = 2,
         .pColorAttachments = color_atts,
      };

      hk_CmdBeginRendering(hk_cmd_buffer_to_handle(&f.cmd), &resume_info);

      /* Confirm hk_agx3_resume_suspended_pass() actually reopened f.cs. */
      CHECK(f.cmd.current_cs.gfx == &f.cs);
      CHECK(f.cs.current == f.stream + 32);
      CHECK(f.cmd.state.gfx.render.flags == 0);
      CHECK(f.cmd.state.gfx.render.resumed_in_cb);
      CHECK(!f.cmd.state.gfx.render.g17p_direct_color_resolve);
      CHECK(!f.cs.cr.process_empty_tiles);

      /* Direct resolve must remain forbidden despite flags == 0. */
      const bool suspend_resume = hk_render_forbids_direct_resolve(
         f.cmd.state.gfx.render.flags, f.cmd.state.gfx.render.resumed_in_cb);
      CHECK(suspend_resume);
      CHECK(!hk_render_forbids_direct_resolve(0, false));

      /* Feed the gate into the mixed direct-resolve admission check (as
       * hk_build_g17p_mixed_bg_eot() does when untouched RT1 is elided):
       * allowed for a non-resumed pass, refused for this resumed pass. */
      struct hk_agx3_g17p_mixed_resolve_target target = {
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .view_format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .has_resolve_target = true,
         .average_resolve = true,
         .store_source = false,
         .resolve_pbe_valid = true,
         .resolve_format_matches = true,
         .resolve_samples = 1,
      };
      CHECK(hk_agx3_g17p_mixed_direct_color_resolve_supported(
         &f.cmd.state.gfx.render.tilebuffer, &target, 1, false, false,
         false /* non-resumed */, false, 3));
      CHECK(!hk_agx3_g17p_mixed_direct_color_resolve_supported(
         &f.cmd.state.gfx.render.tilebuffer, &target, 1, false, false,
         suspend_resume, false, 3));

      /* Single-RT store_and_resolve (RT0 LOAD/STORE + AVERAGE resolve) must
       * also stay blocked when gated by !suspend_resume. */
      enum pipe_format single_fmt[1] = {PIPE_FORMAT_R8G8B8A8_UNORM};
      struct agx_tilebuffer_layout single_tib =
         agx_build_tilebuffer_layout(single_fmt, 1, 4, false);
      const struct hk_agx3_g17p_store_and_resolve_target fused = {
         .color_attachment_count = 1,
         .final_store = true,
         .whole_attachments = true,
         .store_source = true,
         .has_resolve_target = true,
         .average_resolve = true,
         .source_samples = 4,
         .resolve_samples = 1,
         .resolve_format_matches = true,
         .source_pbe_valid = true,
         .resolve_pbe_valid = true,
      };
      CHECK(hk_agx3_g17p_store_and_resolve_supported(&single_tib, &fused));
      CHECK(!(!suspend_resume &&
              hk_agx3_g17p_store_and_resolve_supported(&single_tib, &fused)));

      fixture_fini(&f);
   }

   /* Chained resume (SUSPENDING -> RESUMING|SUSPENDING -> RESUMING) and
    * secondary flag preservation keep resumed_in_cb and forbid direct resolve. */
   {
      const VkRenderingFlagBits mid = hk_agx3_resumed_render_flags(
         VK_RENDERING_SUSPENDING_BIT,
         VK_RENDERING_RESUMING_BIT | VK_RENDERING_SUSPENDING_BIT);
      CHECK(mid == VK_RENDERING_SUSPENDING_BIT);
      CHECK(hk_render_forbids_direct_resolve(mid, true));

      const VkRenderingFlagBits fin =
         hk_agx3_resumed_render_flags(mid, VK_RENDERING_RESUMING_BIT);
      CHECK(fin == 0);
      CHECK(hk_render_forbids_direct_resolve(fin, true));

      const VkRenderingFlagBits mid_from_ext = hk_agx3_resumed_render_flags(
         VK_RENDERING_RESUMING_BIT | VK_RENDERING_SUSPENDING_BIT,
         VK_RENDERING_RESUMING_BIT);
      CHECK(mid_from_ext == VK_RENDERING_RESUMING_BIT);
      CHECK(hk_render_forbids_direct_resolve(mid_from_ext, true));

      const VkRenderingFlagBits with_sec = hk_agx3_resumed_render_flags(
         VK_RENDERING_SUSPENDING_BIT,
         VK_RENDERING_RESUMING_BIT |
            VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT);
      CHECK(with_sec == VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT);
      CHECK(hk_render_forbids_direct_resolve(with_sec, true));
      CHECK(!hk_render_forbids_direct_resolve(with_sec, false));
   }

   return 0;
}

int
main(void)
{
   if (test_resolve_fold_fs_barriers())
      return 1;
   if (test_in_cb_resume_forbids_direct_resolve())
      return 1;

   printf("hk_resolve_fold_resume: %u checks passed\n", checks);
   return 0;
}

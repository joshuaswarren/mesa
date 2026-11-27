/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 * Exercises the production classifier without a Vulkan device or GPU.
 */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include "hk_barrier.h"

#define BIT(r) (1u << HK_BARRIER_SPLIT_##r)
#define FRAG   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
#define COLOR  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
#define EARLY  VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
#define LATE   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
#define VERT   VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
#define CW     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
#define CR     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT
#define ZW     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
#define ZR     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
#define SR     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
#define SW     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT

static unsigned tests, failures;

static void
check(const char *name, uint32_t actual, uint32_t expected)
{
   tests++;
   if (actual != expected) {
      fprintf(stderr, "%s: got 0x%x, expected 0x%x\n", name, actual, expected);
      failures++;
   }
}

static VkMemoryBarrier2
memory(VkPipelineStageFlags2 src, VkAccessFlags2 src_access,
       VkPipelineStageFlags2 dst, VkAccessFlags2 dst_access)
{
   return (VkMemoryBarrier2){
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = src,
      .srcAccessMask = src_access,
      .dstStageMask = dst,
      .dstAccessMask = dst_access,
   };
}

static void
test_scopes(void)
{
   const struct {
      const char *name;
      VkPipelineStageFlags2 src, dst;
      VkAccessFlags2 src_access, dst_access;
      VkDependencyFlags flags;
      uint32_t reasons;
   } cases[] = {
      {"color WAW", COLOR, COLOR, CW, CW, VK_DEPENDENCY_BY_REGION_BIT, 0},
      {"blending RAW", COLOR, COLOR, CW, CR, VK_DEPENDENCY_BY_REGION_BIT, 0},
      {"late depth to early test", LATE, EARLY, ZW, ZR,
       VK_DEPENDENCY_BY_REGION_BIT, 0},
      {"late depth WAW", LATE, LATE, ZW, ZW, VK_DEPENDENCY_BY_REGION_BIT, 0},
      {"input reads need stored color", COLOR, FRAG, CW,
       VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT,
       BIT(TEXTURE)},
      {"depth input also uses texture", LATE, FRAG, ZW,
       VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT,
       BIT(TEXTURE)},
      {"feedback flag does not enable tile reads", COLOR, FRAG, CW, SR,
       VK_DEPENDENCY_BY_REGION_BIT | VK_DEPENDENCY_FEEDBACK_LOOP_BIT_EXT,
       BIT(TEXTURE)},
      {"generic shader read also uses memory", COLOR, FRAG, CW,
       VK_ACCESS_2_SHADER_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT, BIT(TEXTURE)},
      {"global attachment dependency", COLOR, COLOR, CW, CR, 0,
       BIT(GLOBAL) | BIT(TEXTURE)},
      {"region execution", FRAG, FRAG, 0, 0, VK_DEPENDENCY_BY_REGION_BIT, 0},
      {"global execution", FRAG, FRAG, 0, 0, 0, BIT(GLOBAL)},
      {"native vertex execution needs fence", VERT, VERT, 0, 0, 0, BIT(VERTEX)},
      {"vertex to fragment execution", VERT, FRAG, 0, 0, 0, 0},
      {"fragment to vertex cannot share render", FRAG, VERT, 0, 0,
       VK_DEPENDENCY_BY_REGION_BIT, BIT(GLOBAL) | BIT(VERTEX)},
      {"unwritten uniform memory", VERT, FRAG, VK_ACCESS_2_UNIFORM_READ_BIT,
       VK_ACCESS_2_UNIFORM_READ_BIT, 0, 0},
      {"visibility can expose earlier attachment availability", FRAG, FRAG, SW,
       SR, VK_DEPENDENCY_BY_REGION_BIT, BIT(TEXTURE)},
      {"read-only source can relay local attachment availability", COLOR, FRAG,
       CR, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT,
       BIT(TEXTURE)},
      {"empty access source can expose available attachments", COLOR, FRAG, 0,
       SR, VK_DEPENDENCY_BY_REGION_BIT, BIT(TEXTURE)},
      {"generic write without shader stores only attachments", COLOR, COLOR,
       VK_ACCESS_2_MEMORY_WRITE_BIT, CR, VK_DEPENDENCY_BY_REGION_BIT, 0},
      {"availability-only attachment barrier", COLOR, FRAG, CW, 0,
       VK_DEPENDENCY_BY_REGION_BIT, BIT(TEXTURE)},
      {"transfer write", VK_PIPELINE_STAGE_2_COPY_BIT, FRAG,
       VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_UNIFORM_READ_BIT, 0,
       BIT(COMPUTE_TRANSFER)},
      {"compute storage write", VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VERT,
       SW, VK_ACCESS_2_UNIFORM_READ_BIT, 0,
       BIT(COMPUTE_TRANSFER) | BIT(VERTEX)},
      {"top source is empty", VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, FRAG, 0,
       VK_ACCESS_2_UNIFORM_READ_BIT, 0, 0},
      {"bottom destination is empty", FRAG,
       VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0, 0, 0, 0},
      {"none scopes", 0, 0, 0, 0, 0, 0},
      {"unknown stage", UINT64_C(1) << 63, VERT, 0, 0, 0,
       BIT(UNKNOWN) | BIT(VERTEX)},
      {"unknown access", VERT, FRAG, UINT64_C(1) << 63,
       VK_ACCESS_2_UNIFORM_READ_BIT, 0, BIT(UNKNOWN)},
      {"unknown destination access", VERT, FRAG, 0, UINT64_C(1) << 63, 0,
       BIT(UNKNOWN)},
   };

   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      VkMemoryBarrier2 b = memory(cases[i].src, cases[i].src_access,
                                  cases[i].dst, cases[i].dst_access);
      VkDependencyInfo info = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .dependencyFlags = cases[i].flags,
         .memoryBarrierCount = 1,
         .pMemoryBarriers = &b,
      };
      VkImage attachment = (VkImage)(uintptr_t)1;
      struct hk_barrier_state state = {
         .attachment_count = 1,
         .attachments = &attachment,
      };
      check(cases[i].name, hk_barrier_classify(&info, &state),
            cases[i].reasons);
   }
}

static void
test_sides_and_writes(void)
{
   /* Merely opening a query/GS stream must not turn a framebuffer-local
    * barrier into a store/reload. Vertex destinations must still split. */
   for (unsigned sides = 0; sides < 4; ++sides) {
      VkImage attachment = (VkImage)(uintptr_t)1;
      struct hk_barrier_state state = {
         .attachment_count = 1,
         .attachments = &attachment,
         .pre_graphics = (sides & 1) != 0,
         .post_graphics = (sides & 2) != 0,
      };
      VkMemoryBarrier2 b = memory(COLOR, CW, COLOR, CR);
      VkDependencyInfo info = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
         .memoryBarrierCount = 1,
         .pMemoryBarriers = &b,
      };
      check("local attachment with side streams",
            hk_barrier_classify(&info, &state), 0);
      b = memory(FRAG, 0, FRAG, 0);
      check("local execution with side streams",
            hk_barrier_classify(&info, &state), 0);
      b = memory(VERT, 0, FRAG, 0);
      check("software vertex already precedes fragments",
            hk_barrier_classify(&info, &state), 0);
      b = memory(VERT, 0, VERT, 0);
      check("software vertex destination", hk_barrier_classify(&info, &state),
            BIT(VERTEX) | (sides ? BIT(SIDE_STREAM) : 0));
      state.attachment_count = 0;
      b = memory(FRAG, SW, FRAG, SR);
      check("side streams may write storage",
            hk_barrier_classify(&info, &state), sides ? BIT(SHADER) : 0);
      state.shader_writes = true;
      check("storage memory is not ordered by region",
            hk_barrier_classify(&info, &state), BIT(SHADER));
   }

   VkImage attachment = (VkImage)(uintptr_t)1;
   struct hk_barrier_state state = {
      .tilebuffer_spilled = true,
      .attachment_count = 1,
      .attachments = &attachment,
   };
   VkMemoryBarrier2 b = memory(COLOR, CW, COLOR, CR);
   VkDependencyInfo info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &b,
   };
   check("spilled attachment cannot use tile proof",
         hk_barrier_classify(&info, &state), BIT(TEXTURE));
   state = (struct hk_barrier_state){.attachments_alias = true};
   check("aliased attachment cannot use tile proof",
         hk_barrier_classify(&info, &state), BIT(TEXTURE));
   state = (struct hk_barrier_state){.shader_writes = true};
   b = memory(VERT, SW, FRAG, SR);
   check("earlier physical stream's shader stores remain relevant",
         hk_barrier_classify(&info, &state), BIT(SHADER));
   b = memory(FRAG, VK_ACCESS_2_SHADER_READ_BIT, FRAG, SR);
   check("visibility-only operation after earlier shader availability",
         hk_barrier_classify(&info, &state), BIT(SHADER));
   b.srcAccessMask = 0;
   check("empty access source still needs shader visibility",
         hk_barrier_classify(&info, &state), BIT(SHADER));
   state.shader_writes = false;
   check("no attachments or shader writes to publish",
         hk_barrier_classify(&info, &state), 0);

   state = (struct hk_barrier_state){
      .attachment_count = 1,
      .attachments = &attachment,
   };
   b = memory(COLOR, CW, COLOR, CR);
   check("availability chain begins in tile memory",
         hk_barrier_classify(&info, &state), 0);
   b = memory(COLOR, CR, FRAG, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   check("availability chain ends with a texture-visible store",
         hk_barrier_classify(&info, &state), BIT(TEXTURE));
}

static void
test_read_only_attachments(void)
{
   const struct {
      const char *name;
      VkImageLayout layout;
      bool depth_writes, stencil_writes;
   } layouts[] = {
      {"generic read-only", VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL, false, false},
      {"depth/stencil read-only",
       VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL, false, false},
      {"depth read-only", VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, false, true},
      {"stencil read-only", VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL, true,
       false},
      {"depth RO stencil RW",
       VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL, false, true},
      {"depth RW stencil RO",
       VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL, true, false},
      {"general", VK_IMAGE_LAYOUT_GENERAL, true, true},
      {"attachment", VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL, true, true},
      {"feedback", VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT, true,
       true},
      {"unknown", VK_IMAGE_LAYOUT_MAX_ENUM, true, true},
   };
   for (unsigned i = 0; i < sizeof(layouts) / sizeof(layouts[0]); ++i) {
      char label[128];
      snprintf(label, sizeof(label), "%s depth", layouts[i].name);
      check(label,
            hk_barrier_attachment_may_write(
               layouts[i].layout, VK_IMAGE_ASPECT_DEPTH_BIT,
               VK_ATTACHMENT_LOAD_OP_LOAD, VK_RESOLVE_MODE_NONE),
            layouts[i].depth_writes);
      snprintf(label, sizeof(label), "%s stencil", layouts[i].name);
      check(label,
            hk_barrier_attachment_may_write(
               layouts[i].layout, VK_IMAGE_ASPECT_STENCIL_BIT,
               VK_ATTACHMENT_LOAD_OP_LOAD, VK_RESOLVE_MODE_NONE),
            layouts[i].stencil_writes);
      check("clear is not an unwritten attachment",
            hk_barrier_attachment_may_write(
               layouts[i].layout, VK_IMAGE_ASPECT_DEPTH_BIT,
               VK_ATTACHMENT_LOAD_OP_CLEAR, VK_RESOLVE_MODE_NONE),
            1);
   }
   check("non-load remains conservative",
         hk_barrier_attachment_may_write(
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            VK_IMAGE_ASPECT_DEPTH_BIT, VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_RESOLVE_MODE_NONE),
         1);
   check("color is not inferred read-only",
         hk_barrier_attachment_may_write(
            VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT,
            VK_ATTACHMENT_LOAD_OP_LOAD, VK_RESOLVE_MODE_NONE),
         1);

   check("read-only resolve remains conservative",
         hk_barrier_attachment_may_write(
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            VK_IMAGE_ASPECT_DEPTH_BIT, VK_ATTACHMENT_LOAD_OP_LOAD,
            VK_RESOLVE_MODE_SAMPLE_ZERO_BIT),
         1);

   VkImage image = (VkImage)(uintptr_t)1;
   VkImageMemoryBarrier2 b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = EARLY,
      .srcAccessMask = ZR,
      .dstStageMask = FRAG,
      .dstAccessMask = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT,
      .image = image,
      .oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
   };
   VkDependencyInfo info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &b,
   };
   struct hk_barrier_state state = {.attachments = &image};
   state.attachment_count = hk_barrier_attachment_may_write(
      b.oldLayout, VK_IMAGE_ASPECT_DEPTH_BIT, VK_ATTACHMENT_LOAD_OP_LOAD,
      VK_RESOLVE_MODE_NONE);
   check("loaded read-only depth can stay texture-backed without a store",
         hk_barrier_classify(&info, &state), 0);
   state.shader_writes = true;
   check("read-only attachment does not hide shader writes",
         hk_barrier_classify(&info, &state), BIT(SHADER));
   state.shader_writes = false;
   b.oldLayout = b.newLayout =
      VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL;
   state.attachment_count = hk_barrier_attachment_may_write(
      b.oldLayout, VK_IMAGE_ASPECT_STENCIL_BIT, VK_ATTACHMENT_LOAD_OP_LOAD,
      VK_RESOLVE_MODE_NONE);
   check("writable stencil keeps whole-image dependency conservative",
         hk_barrier_classify(&info, &state), BIT(TEXTURE));
}

static void
test_resources_and_fallbacks(void)
{
   check("ordinary optimal attachment admits disjoint proof",
         hk_barrier_image_may_alias(0, VK_IMAGE_TILING_OPTIMAL, 0), 0);
   check("explicit alias flag",
         hk_barrier_image_may_alias(VK_IMAGE_CREATE_ALIAS_BIT,
                                    VK_IMAGE_TILING_OPTIMAL, 0),
         1);
   check("sparse alias possibility",
         hk_barrier_image_may_alias(VK_IMAGE_CREATE_SPARSE_BINDING_BIT,
                                    VK_IMAGE_TILING_OPTIMAL, 0),
         1);
   check("linear alias does not need ALIAS flag",
         hk_barrier_image_may_alias(0, VK_IMAGE_TILING_LINEAR, 0), 1);
   check(
      "modifier may be linear",
      hk_barrier_image_may_alias(0, VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT, 0),
      1);
   check(
      "external memory implies alias",
      hk_barrier_image_may_alias(0, VK_IMAGE_TILING_OPTIMAL,
                                 VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT),
      1);

   VkImage attachment = (VkImage)(uintptr_t)1;
   struct hk_barrier_state state = {
      .attachment_count = 1,
      .attachments = &attachment,
   };
   VkImageMemoryBarrier2 image = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = COLOR,
      .srcAccessMask = CW,
      .dstStageMask = FRAG,
      .dstAccessMask = SR,
      .image = attachment,
      .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
   };
   VkDependencyInfo info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &image,
   };
   check("same image texture feedback", hk_barrier_classify(&info, &state),
         BIT(TEXTURE));
   image.image = (VkImage)(uintptr_t)2;
   check("unrelated image is unwritten", hk_barrier_classify(&info, &state), 0);
   state.attachments_alias = true;
   check("alias/sparse attachment disables disjoint proof",
         hk_barrier_classify(&info, &state), BIT(TEXTURE));
   state.attachments_alias = false;
   image.srcStageMask = FRAG;
   image.srcAccessMask = SW;
   state.shader_writes = true;
   check("arbitrary image may have storage writes",
         hk_barrier_classify(&info, &state), BIT(SHADER));
   state.shader_writes = false;
   image.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   check("layout transition needs boundary", hk_barrier_classify(&info, &state),
         BIT(LAYOUT));
   image.oldLayout = image.newLayout = VK_IMAGE_LAYOUT_UNDEFINED;
   check("equal UNDEFINED is not a transition",
         hk_barrier_classify(&info, &state), 0);
   image.dstQueueFamilyIndex = 0;
   check("ownership transfer", hk_barrier_classify(&info, &state),
         BIT(OWNERSHIP));
   image.pNext = &info;
   check("unknown image extension retains ownership cause",
         hk_barrier_classify(&info, &state), BIT(OWNERSHIP) | BIT(UNKNOWN));

   VkBufferMemoryBarrier2 buffer = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VERT,
      .srcAccessMask = SW,
      .dstStageMask = FRAG,
      .dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
   };
   info.imageMemoryBarrierCount = 0;
   info.bufferMemoryBarrierCount = 1;
   info.pBufferMemoryBarriers = &buffer;
   check("buffer unwritten by read-only shaders",
         hk_barrier_classify(&info, &state), 0);
   state.shader_writes = true;
   check("buffer shader writes", hk_barrier_classify(&info, &state),
         BIT(SHADER));
   state.shader_writes = false;
   buffer.dstQueueFamilyIndex = 0;
   check("buffer ownership", hk_barrier_classify(&info, &state),
         BIT(OWNERSHIP));
   buffer.pNext = &info;
   check("buffer unknown extension", hk_barrier_classify(&info, &state),
         BIT(OWNERSHIP) | BIT(UNKNOWN));

   info = (VkDependencyInfo){.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
   check("empty dependency", hk_barrier_classify(&info, &state), 0);
   info.dependencyFlags = 1u << 31;
   check("unknown dependency flag", hk_barrier_classify(&info, &state),
         BIT(UNKNOWN));
   info.dependencyFlags = 0;
   info.pNext = &image;
   check("unknown dependency extension", hk_barrier_classify(&info, &state),
         BIT(UNKNOWN));

   VkMemoryBarrier2 barriers[] = {
      memory(COLOR, CW, COLOR, CR),
      memory(COLOR, CW, FRAG, SR),
      memory(FRAG, SW, FRAG, SR),
   };
   info = (VkDependencyInfo){
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
      .memoryBarrierCount = 3,
      .pMemoryBarriers = barriers,
   };
   state.shader_writes = true;
   check("mixed barriers retain every cause",
         hk_barrier_classify(&info, &state), BIT(TEXTURE) | BIT(SHADER));
   barriers[0] = memory(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, 0,
                        VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, 0);
   info.memoryBarrierCount = 1;
   check("ALL_GRAPHICS includes fragment-to-vertex",
         hk_barrier_classify(&info, &state), BIT(GLOBAL) | BIT(VERTEX));
   barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
   check("ALL_COMMANDS conservatively expands",
         hk_barrier_classify(&info, &state),
         BIT(GLOBAL) | BIT(COMPUTE_TRANSFER) | BIT(VERTEX));
   barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
   check("bottom source is ALL_COMMANDS", hk_barrier_classify(&info, &state),
         BIT(GLOBAL) | BIT(COMPUTE_TRANSFER) | BIT(VERTEX));
   barriers[0] = memory(VERT, 0, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0);
   check("top destination is ALL_COMMANDS", hk_barrier_classify(&info, &state),
         BIT(COMPUTE_TRANSFER) | BIT(VERTEX));
   barriers[0] = memory(VERT, 0, FRAG, 0);
   barriers[0].pNext = &image;
   check("memory access extension is not silently ignored",
         hk_barrier_classify(&info, &state), BIT(UNKNOWN));

   for (unsigned r = 0; r < HK_BARRIER_SPLIT_COUNT; ++r)
      check("every split reason has a name",
            hk_barrier_split_reason_name(r) != NULL, 1);
}

static void
test_local_read(void)
{
   VkImage color_img = (VkImage)(uintptr_t)10;
   VkImage depth_img = (VkImage)(uintptr_t)11;
   VkImage atts[2] = {color_img, depth_img};

   struct hk_barrier_state color_only = {
      .color_input_local_read = true,
      .depth_stencil_writes = false,
      .color_attachment_count = 1,
      .attachment_count = 1,
      .attachments = atts,
   };
   struct hk_barrier_state color_and_ds = {
      .color_input_local_read = true,
      .depth_stencil_writes = true,
      .color_attachment_count = 1,
      .attachment_count = 2,
      .attachments = atts,
   };

   VkMemoryBarrier2 b =
      memory(COLOR, CW, FRAG, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   VkDependencyInfo info = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &b,
   };

   check("local_read color CW -> input read kept",
         hk_barrier_classify(&info, &color_only), 0);
   check("local_read color CW -> input read kept even with writable D/S",
         hk_barrier_classify(&info, &color_and_ds), 0);

   b = memory(COLOR, CW, FRAG,
              VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT | CR);
   check("local_read color CW -> input read + CR kept",
         hk_barrier_classify(&info, &color_and_ds), 0);

   /* Non-BY_REGION still splits globally + texture. */
   info.dependencyFlags = 0;
   b = memory(COLOR, CW, FRAG, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   check("local_read without BY_REGION splits",
         hk_barrier_classify(&info, &color_only), BIT(GLOBAL) | BIT(TEXTURE));
   info.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

   /* Depth/stencil write -> input read must split. */
   b = memory(LATE, ZW, FRAG, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   check("local_read depth ZW -> input read splits",
         hk_barrier_classify(&info, &color_and_ds), BIT(TEXTURE));
   b = memory(LATE, 0, FRAG, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   check("local_read depth stage -> input read splits when D/S writable",
         hk_barrier_classify(&info, &color_and_ds), BIT(TEXTURE));
   b = memory(COLOR, VK_ACCESS_2_MEMORY_WRITE_BIT, FRAG,
              VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   check("local_read MEMORY_WRITE splits when D/S writable",
         hk_barrier_classify(&info, &color_and_ds), BIT(TEXTURE));
   check("local_read MEMORY_WRITE kept when color-only",
         hk_barrier_classify(&info, &color_only), 0);

   /* Availability chain: CW -> CR then CR -> input read stays in tile. */
   b = memory(COLOR, CR, FRAG, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   check("local_read availability chain CR -> input read kept",
         hk_barrier_classify(&info, &color_and_ds), 0);

   /* Sampled / generic shader reads still split. */
   b = memory(COLOR, CW, FRAG, SR);
   check("local_read sampled read still splits",
         hk_barrier_classify(&info, &color_only), BIT(TEXTURE));
   b = memory(COLOR, CW, FRAG, VK_ACCESS_2_SHADER_READ_BIT);
   check("local_read generic shader read still splits",
         hk_barrier_classify(&info, &color_only), BIT(TEXTURE));

   /* Tilebuffer spill and aliasing disable local_read. */
   struct hk_barrier_state spilled = color_only;
   spilled.tilebuffer_spilled = true;
   b = memory(COLOR, CW, FRAG, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   check("local_read spilled splits",
         hk_barrier_classify(&info, &spilled), BIT(TEXTURE));
   struct hk_barrier_state aliased = color_only;
   aliased.attachments_alias = true;
   check("local_read aliased splits",
         hk_barrier_classify(&info, &aliased), BIT(TEXTURE));

   /* Image barriers: color attachment kept, D/S attachment splits. */
   VkImageMemoryBarrier2 img = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = COLOR,
      .srcAccessMask = CW,
      .dstStageMask = FRAG,
      .dstAccessMask = VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT,
      .image = color_img,
      .oldLayout = VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ,
      .newLayout = VK_IMAGE_LAYOUT_RENDERING_LOCAL_READ,
      .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
   };
   info.memoryBarrierCount = 0;
   info.imageMemoryBarrierCount = 1;
   info.pImageMemoryBarriers = &img;
   check("local_read color image barrier kept",
         hk_barrier_classify(&info, &color_and_ds), 0);
   img.image = depth_img;
   img.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
   check("local_read depth image barrier splits",
         hk_barrier_classify(&info, &color_and_ds), BIT(TEXTURE));

   /* used_local_read is set ONLY when the local-read rule actually kept the
    * barrier (reasons == 0 and a written color attachment was admitted via
    * color_input_local_read). */
   bool used_lr = false;
   info.imageMemoryBarrierCount = 0;
   info.memoryBarrierCount = 1;
   b = memory(COLOR, CW, FRAG, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   info.pMemoryBarriers = &b;
   check("used_local_read: kept color CW -> input read returns 0",
         hk_barrier_classify_ex(&info, &color_only, &used_lr), 0);
   check("used_local_read: kept color CW -> input read sets flag", used_lr, 1);

   /* Kept barrier without INPUT_ATTACHMENT_READ must NOT set used_local_read. */
   b = memory(COLOR, CW, COLOR, CR);
   used_lr = true;
   check("used_local_read: color CW -> CR returns 0",
         hk_barrier_classify_ex(&info, &color_only, &used_lr), 0);
   check("used_local_read: color CW -> CR clears flag", used_lr, 0);

   /* Split barrier with INPUT_ATTACHMENT_READ must NOT set used_local_read. */
   b = memory(LATE, ZW, FRAG, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   used_lr = true;
   check("used_local_read: depth ZW -> input read splits",
         hk_barrier_classify_ex(&info, &color_and_ds, &used_lr), BIT(TEXTURE));
   check("used_local_read: split barrier clears flag", used_lr, 0);

   /* Kept barrier with INPUT_ATTACHMENT_READ when color_input_local_read is
    * false and no attachment writes exist must NOT set used_local_read. */
   struct hk_barrier_state no_writes = {
      .color_input_local_read = false,
      .attachment_count = 0,
   };
   b = memory(COLOR, 0, FRAG, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   used_lr = true;
   check("used_local_read: no writable attachments keeps barrier",
         hk_barrier_classify_ex(&info, &no_writes, &used_lr), 0);
   check("used_local_read: no writable attachments clears flag", used_lr, 0);

   /* Image barrier on an unrelated image with INPUT_ATTACHMENT_READ must NOT
    * set used_local_read even when color_input_local_read is true. */
   img.image = (VkImage)(uintptr_t)99;
   img.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
   info.memoryBarrierCount = 0;
   info.imageMemoryBarrierCount = 1;
   info.pImageMemoryBarriers = &img;
   used_lr = true;
   check("used_local_read: unrelated image barrier returns 0",
         hk_barrier_classify_ex(&info, &color_only, &used_lr), 0);
   check("used_local_read: unrelated image barrier clears flag", used_lr, 0);

   /* Deferred draw-time fallback split transition:
    * - A kept local-read barrier arms kept_local_read_barrier.
    * - Non-input-attachment draws, meta draws, and on-chip local-read draws
    *   keep the pass open.
    * - The first texture-path input-attachment draw (e.g. depth input after a
    *   memory barrier, different view of the same image, or constant-sample
    *   MSAA read) triggers the deferred split and clears the armed flag.
    * - An unrelated kept barrier never arms the flag, so a subsequent
    *   texture-path input-attachment draw does not split needlessly. */
   info.imageMemoryBarrierCount = 0;
   info.memoryBarrierCount = 1;
   b = memory(COLOR, CW, FRAG, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   info.pMemoryBarriers = &b;
   bool kept_local_read_barrier = false;
   used_lr = false;
   if (hk_barrier_classify_ex(&info, &color_only, &used_lr) == 0 && used_lr)
      kept_local_read_barrier = true;
   check("deferred split: barrier arms kept_local_read_barrier",
         kept_local_read_barrier, 1);
   check("deferred split: draw without input attachments keeps pass",
         hk_barrier_local_read_draw_needs_split(kept_local_read_barrier, false,
                                                false, false),
         0);
   check("deferred split: on-chip local_read draw keeps pass",
         hk_barrier_local_read_draw_needs_split(kept_local_read_barrier, false,
                                                true, true),
         0);
   check("deferred split: meta draw keeps pass",
         hk_barrier_local_read_draw_needs_split(kept_local_read_barrier, true,
                                                true, false),
         0);
   check("deferred split: texture-path input-attachment draw splits",
         hk_barrier_local_read_draw_needs_split(kept_local_read_barrier, false,
                                                true, false),
         1);
   /* Ending graphics clears kept_local_read_barrier. */
   kept_local_read_barrier = false;
   check("deferred split: after split, subsequent fallback draw does not split",
         hk_barrier_local_read_draw_needs_split(kept_local_read_barrier, false,
                                                true, false),
         0);

   /* Unrelated kept barrier with INPUT_ATTACHMENT_READ does not arm a split. */
   b = memory(COLOR, 0, FRAG, VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
   used_lr = false;
   if (hk_barrier_classify_ex(&info, &no_writes, &used_lr) == 0 && used_lr)
      kept_local_read_barrier = true;
   check("deferred split: unrelated barrier does not arm split",
         hk_barrier_local_read_draw_needs_split(kept_local_read_barrier, false,
                                                true, false),
         0);
}


static void
test_pure_compute_ssbo(void)
{
   const VkPipelineStageFlags2 cs = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
   const VkAccessFlags2 sw = VK_ACCESS_2_SHADER_WRITE_BIT;
   const VkAccessFlags2 srw =
      VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;

   VkMemoryBarrier2 mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = cs,
      .srcAccessMask = sw,
      .dstStageMask = cs,
      .dstAccessMask = srw,
   };
   VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &mb,
   };
   check("pure_ssbo: compute memory barrier admitted",
         hk_dependency_is_pure_compute_ssbo(&dep), 1);

   mb.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
   check("pure_ssbo: memory barrier with NONE srcStage rejected",
         hk_dependency_is_pure_compute_ssbo(&dep), 0);
   mb.srcStageMask = cs;
   mb.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
   check("pure_ssbo: memory barrier with NONE dstStage rejected",
         hk_dependency_is_pure_compute_ssbo(&dep), 0);
   mb.dstStageMask = cs;
   mb.dstAccessMask = VK_ACCESS_2_UNIFORM_READ_BIT;
   check("pure_ssbo: uniform read access rejected",
         hk_dependency_is_pure_compute_ssbo(&dep), 0);
   mb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
   check("pure_ssbo: sampled read access rejected",
         hk_dependency_is_pure_compute_ssbo(&dep), 0);
   mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
   check("pure_ssbo: broad MEMORY_READ rejected",
         hk_dependency_is_pure_compute_ssbo(&dep), 0);

   VkBufferMemoryBarrier2 bb = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = cs,
      .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
      .dstStageMask = cs,
      .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
   };
   dep = (VkDependencyInfo){
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &bb,
   };
   check("pure_ssbo: local buffer barrier (IGNORED) admitted",
         hk_dependency_is_pure_compute_ssbo(&dep), 1);

   bb.srcQueueFamilyIndex = 0;
   bb.dstQueueFamilyIndex = 0;
   check("pure_ssbo: local buffer barrier (same queue) admitted",
         hk_dependency_is_pure_compute_ssbo(&dep), 1);

   /* Ownership release to VK_QUEUE_FAMILY_EXTERNAL (dstScope NONE). */
   bb.srcQueueFamilyIndex = 0;
   bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
   bb.srcStageMask = cs;
   bb.srcAccessMask = sw;
   bb.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
   bb.dstAccessMask = VK_ACCESS_2_NONE;
   check("pure_ssbo: ownership release to EXTERNAL rejected",
         hk_dependency_is_pure_compute_ssbo(&dep), 0);

   /* Ownership acquire from VK_QUEUE_FAMILY_EXTERNAL (srcScope NONE). */
   bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
   bb.dstQueueFamilyIndex = 0;
   bb.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
   bb.srcAccessMask = VK_ACCESS_2_NONE;
   bb.dstStageMask = cs;
   bb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
   check("pure_ssbo: ownership acquire from EXTERNAL rejected",
         hk_dependency_is_pure_compute_ssbo(&dep), 0);

   /* Cross-queue transfer even with COMPUTE on both scopes. */
   bb.srcQueueFamilyIndex = 0;
   bb.dstQueueFamilyIndex = 1;
   bb.srcStageMask = cs;
   bb.srcAccessMask = sw;
   bb.dstStageMask = cs;
   bb.dstAccessMask = srw;
   check("pure_ssbo: cross-queue buffer transfer rejected",
         hk_dependency_is_pure_compute_ssbo(&dep), 0);

   /* EXTERNAL == EXTERNAL or FOREIGN == FOREIGN rejected. */
   bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
   bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
   check("pure_ssbo: EXTERNAL queue family rejected",
         hk_dependency_is_pure_compute_ssbo(&dep), 0);

   bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
   bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
   check("pure_ssbo: FOREIGN queue family rejected",
         hk_dependency_is_pure_compute_ssbo(&dep), 0);

   /* Image barrier alongside pure buffer barrier rejected. */
   bb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
   VkImageMemoryBarrier2 ib = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = cs,
      .srcAccessMask = sw,
      .dstStageMask = cs,
      .dstAccessMask = srw,
   };
   dep.imageMemoryBarrierCount = 1;
   dep.pImageMemoryBarriers = &ib;
   check("pure_ssbo: image barrier present rejected",
         hk_dependency_is_pure_compute_ssbo(&dep), 0);
}

int
main(void)
{
   test_scopes();
   test_sides_and_writes();
   test_read_only_attachments();
   test_resources_and_fallbacks();
   test_local_read();
   test_pure_compute_ssbo();
   printf("hk_barrier: %u checks, %u failures\n", tests, failures);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

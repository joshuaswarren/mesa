/* SPDX-License-Identifier: MIT
 * Submit-time render packing (hk_render_pack.c) with the real hk_sync
 * recorder and resolver; no GPU, DRM or Vulkan device.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/ralloc.h"
#include "util/set.h"
#include "util/u_math.h"
#include "asahi/lib/agx_vdm_stream_link_core.h"
#include "asahi/lib/agx_vdm_stream_terminate_core.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_queue.h"
#include "hk_render_pack.h"

static unsigned tests, failures;
static void *test_context;

static void *
test_alloc(size_t size)
{
   void *ptr = rzalloc_size(test_context, size);
   if (!ptr) {
      fputs("Failed to allocate render packing test fixture\n", stderr);
      exit(EXIT_FAILURE);
   }
   return ptr;
}

#define T(name, cond)                                                          \
   do {                                                                        \
      tests++;                                                                 \
      if (!(cond)) {                                                           \
         failures++;                                                           \
         printf("FAIL %s line%d\n", name, __LINE__);                           \
      } else                                                                   \
         printf("PASS %s\n", name);                                            \
   } while (0)

/* Every chip the packing runs on encodes the stream link and terminate the
 * same way; pick one of each command-stream generation. */
static enum agx_chip chip = AGX_CHIP_G17S;

static struct hk_queue *
new_queue(bool exact)
{
   struct hk_queue *q = test_alloc(sizeof(struct hk_queue));
   q->sync.exact_prior = exact;
   return q;
}

static struct hk_cmd_buffer *
new_cmd(void)
{
   struct hk_cmd_buffer *c = test_alloc(sizeof(*c));
   struct hk_device *dev = test_alloc(sizeof(struct hk_device));
   dev->dev.chip = chip;
   c->vk.base.device = &dev->vk;
   c->vk.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
   list_inithead(&c->control_streams);
   util_dynarray_init(&c->sync_pass_records, NULL);
   return c;
}

static struct hk_cs *
first_cs(const struct hk_cmd_buffer *c)
{
   return list_first_entry(&c->control_streams, struct hk_cs, node);
}

static struct hk_cs *
add_stream(struct hk_cmd_buffer *c, bool compute)
{
   struct hk_cs *s = test_alloc(sizeof(*s));
   s->type = compute ? HK_CS_CDM : HK_CS_VDM;
   s->cmd = c;
   list_addtail(&s->node, &c->control_streams);
   hk_sync_stream_created(c, s, &s, compute);
   util_dynarray_init(&s->scissor, NULL);
   util_dynarray_init(&s->depth_bias, NULL);
   util_dynarray_init(&s->render_resources, NULL);
   util_dynarray_init(&s->imm_writes, NULL);
   util_dynarray_init(&s->oq_stamps, NULL);
   return s;
}

static uint64_t next_addr = UINT64_C(0x2000000000);

/* The image views and images the passes name; compared, never read. */
static struct hk_image_view *views[4];
static uint8_t images[4][64];

static void
set_att(struct hk_render_pack_att *a, unsigned view)
{
   a->iview = views[view];
   a->image = (const struct vk_image *)images[view];
   a->load_op = VK_ATTACHMENT_LOAD_OP_LOAD;
   a->store_op = VK_ATTACHMENT_STORE_OP_STORE;
   a->layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   a->resolve_mode = VK_RESOLVE_MODE_NONE;
   a->samples = 1;
}

/* GTA V's G-buffer split: five colour targets and depth, all LOAD/STORE. */
static void
gbuffer(struct hk_render_pack_pass *p)
{
   memset(p, 0, sizeof(*p));
   for (unsigned i = 0; i < 3; ++i)
      set_att(&p->att[i], i);
   set_att(&p->att[HK_RENDER_PACK_DEPTH], 3);
   p->att[HK_RENDER_PACK_DEPTH].layout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
   p->area = (VkRect2D){.extent = {3024, 1964}};
   p->layer_count = 1;
   p->color_count = 3;
   p->layout_proof = true;
}

static const uint64_t bg = UINT64_C(0x1000040), eot = UINT64_C(0x1000080);

/* A command buffer recording one G-buffer split as hk_EndCommandBuffer()
 * leaves it: one render stream that begins and ends its pass, with head and
 * tail edges and the terminate record the submission may link over. */
static struct hk_cmd_buffer *
split(unsigned draws)
{
   struct hk_cmd_buffer *c = new_cmd();
   struct hk_cs *s = add_stream(c, false);
   s->stats.cmds = s->stats.calls = draws;
   s->g17p_pass_begins = s->g17p_pass_ends = true;
   s->tib.nr_samples = 1;
   s->tib.sample_size_B = 24;
   s->tib.tile_size = 32 * 32;
   s->cr.width = 3024;
   s->cr.height = 1964;
   s->cr.layers = 1;
   s->cr.depth.buffer = UINT64_C(0x3000000);
   s->cr.bg.main.usc = s->cr.bg.partial.usc = bg;
   s->cr.eot.main.usc = s->cr.eot.partial.usc = eot;
   s->area_x1 = 3024;
   s->area_y1 = 1964;
   s->tile_resources.stage = 1;

   struct agx_vdm_stream_terminate_core t;
   agx_vdm_stream_terminate_core_pack(chip, &t);
   uint8_t *bytes = test_alloc(64 + HK_CS_SPACE_PADDING);
   memcpy(bytes + 60, t.bytes, t.length);
   s->start = bytes;
   s->current = bytes + 60 + t.length;
   s->end = bytes + 64 + HK_CS_SPACE_PADDING;
   s->addr = next_addr;
   next_addr += 0x10000;

   struct hk_render_pack_edges *e = &c->render_pack;
   e->head = e->tail = true;
   e->head_dep = e->tail_dep = HK_M5_MERGE_DEP_LOCAL;
   gbuffer(&e->head_pass);
   gbuffer(&e->tail_pass);
   e->terminate = bytes + 60;
   return c;
}

static bool
is_terminate(const uint8_t *at)
{
   struct agx_vdm_stream_terminate_core t;
   agx_vdm_stream_terminate_core_pack(chip, &t);
   return !memcmp(at, t.bytes, t.length);
}

static bool
is_link_to(const uint8_t *at, uint64_t target)
{
   uint64_t got;
   return agx_vdm_stream_link_core_unpack(chip, at,
                                          AGX_VDM_STREAM_LINK_CORE_LENGTH,
                                          &got) &&
          got == target;
}

static void
test_non_local_src(void)
{
   const VkPipelineStageFlags2 color =
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
   const VkPipelineStageFlags2 attach =
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | color;
   const VkAccessFlags2 color_write = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;

   T("FS -> color output classifies LOCAL but non-local src",
     hk_m5_pass_merge_dependency(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, color,
                                 color_write, false, false) ==
           HK_M5_MERGE_DEP_LOCAL &&
        hk_m5_pass_merge_non_local_src(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                       color));
   T("ALL_GRAPHICS -> color output is non-local src",
     hk_m5_pass_merge_dependency(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, color,
                                 color_write, false, false) ==
           HK_M5_MERGE_DEP_LOCAL &&
        hk_m5_pass_merge_non_local_src(VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
                                       color));
   T("BOTTOM_OF_PIPE -> color output is non-local src",
     hk_m5_pass_merge_non_local_src(VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                    color));
   T("color|FS -> color output is non-local src",
     hk_m5_pass_merge_non_local_src(
        color | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, color));
   T("color -> color output is attachment-local src",
     !hk_m5_pass_merge_non_local_src(color, color));
   T("attachment stages -> color output is attachment-local src",
     !hk_m5_pass_merge_non_local_src(attach, color));
   T("compute -> color output is outside graphics src",
     !hk_m5_pass_merge_non_local_src(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                     color));
   T("TOP_OF_PIPE -> color output is empty src",
     !hk_m5_pass_merge_non_local_src(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                     color));
   T("FS -> compute does not reach graphics dst",
     !hk_m5_pass_merge_non_local_src(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT));
}

static void
test_continues(void)
{
   struct hk_render_pack_pass a, b;
#define RESET() (gbuffer(&a), gbuffer(&b))
   RESET();
   T("identical G-buffer edges continue",
     !hk_render_pack_continues(&a, &b, false, false, false));
   T("identical G-buffer edges continue across an ordered dependency",
     !hk_render_pack_continues(&a, &b, true, false, true));

   RESET();
   b.att[1].iview = views[3];
   T("another view refuses", hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.att[HK_RENDER_PACK_STENCIL] = a.att[HK_RENDER_PACK_DEPTH];
   T("an attachment only B has refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.att[0].load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
   T("B clearing refuses", hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.att[HK_RENDER_PACK_DEPTH].load_op = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
   T("B discarding depth refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.att[2].store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   T("colour STORE -> DONT_CARE continues",
     !hk_render_pack_continues(&a, &b, false, false, false) &&
        !hk_render_pack_continues(&a, &b, true, false, true) &&
        !hk_render_pack_continues(&a, &b, false, true, false));
   RESET();
   b.att[2].store_op = VK_ATTACHMENT_STORE_OP_NONE;
   T("colour STORE -> NONE refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   a.att[2].store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   b.att[2].store_op = VK_ATTACHMENT_STORE_OP_STORE;
   T("colour DONT_CARE -> STORE refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.att[HK_RENDER_PACK_DEPTH].store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   T("depth STORE -> DONT_CARE refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.att[0].resolve_mode = VK_RESOLVE_MODE_AVERAGE_BIT;
   b.att[0].store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   T("final-member colour resolve continues locally",
     !hk_render_pack_continues(&a, &b, false, false, false));
   b.att[0].flags =
      VK_RENDERING_ATTACHMENT_RESOLVE_SKIP_TRANSFER_FUNCTION_BIT_KHR;
   T("final-member resolve skip-transfer flag continues",
     !hk_render_pack_continues(&a, &b, false, false, false));
   T("final-member colour resolve refuses across an ordered dependency",
     hk_render_pack_continues(&a, &b, true, false, true));
   T("final-member colour resolve refuses after a non-local graphics source",
     hk_render_pack_continues(&a, &b, false, true, false));
   RESET();
   b.att[HK_RENDER_PACK_DEPTH].resolve_mode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
   T("depth resolve refuses", hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   a.att[0].resolve_mode = VK_RESOLVE_MODE_AVERAGE_BIT;
   T("continuing after a resolve refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   b.att[0].resolve_mode = VK_RESOLVE_MODE_AVERAGE_BIT;
   T("two resolves refuse", hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.area.extent.width = 2268;
   T("another render area refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.layer_count = 2;
   T("another layer count refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.color_count = 2;
   b.att[2] = (struct hk_render_pack_att){0};
   T("another colour count refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   a.view_mask = b.view_mask = 3;
   T("multiview refuses", hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.flags = VK_RENDERING_RESUMING_BIT;
   T("a resuming instance refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   a.flags = VK_RENDERING_SUSPENDING_BIT;
   T("a suspending instance refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   a.flags = VK_RENDERING_SUSPENDING_BIT;
   b.flags = VK_RENDERING_RESUMING_BIT | VK_RENDERING_SUSPENDING_BIT;
   T("a resuming instance continues a suspending one",
     !hk_render_pack_continues(&a, &b, false, false, false));
   b.att[0].load_op = VK_ATTACHMENT_LOAD_OP_CLEAR;
   T("and ignores loadOp when resuming",
     !hk_render_pack_continues(&a, &b, false, false, false));
   b.att[0].store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   T("a resuming+suspending instance refuses STORE -> DONT_CARE",
     hk_render_pack_continues(&a, &b, false, false, false));
   b.att[0].resolve_mode = VK_RESOLVE_MODE_AVERAGE_BIT;
   T("a resuming+suspending instance refuses resolve",
     hk_render_pack_continues(&a, &b, false, false, false));
   b.flags = VK_RENDERING_RESUMING_BIT;
   T("a final resuming instance allows colour resolve locally",
     !hk_render_pack_continues(&a, &b, false, false, false));
   T("a final resuming instance refuses resolve after non-local src",
     hk_render_pack_continues(&a, &b, false, true, false));
   RESET();
   a.flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
   b.flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
   T("inline-replayed secondary passes continue",
     !hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   for (unsigned s = 0; s < HK_RENDER_PACK_SLOTS; ++s)
      a.att[s].samples = b.att[s].samples = a.att[s].iview ? 4 : 0;
   b.att[0].layout = VK_IMAGE_LAYOUT_GENERAL;
   T("multisampled attachments in another layout refuse",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.att[0].layout = VK_IMAGE_LAYOUT_GENERAL;
   T("attachments in another layout refuse",
     hk_render_pack_continues(&a, &b, false, false, false) &&
        hk_render_pack_continues(&a, &b, true, false, true));
   RESET();
   b.att[1].layout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
   a.att[1].layout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
   T("a feedback loop layout refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.att[1].feedback_loop = true;
   T("an enabled feedback loop refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   a.att[0].flags = b.att[0].flags =
      VK_RENDERING_ATTACHMENT_INPUT_ATTACHMENT_FEEDBACK_BIT_KHR;
   T("input attachment feedback refuses",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.att[0].flags = VK_RENDERING_ATTACHMENT_INPUT_ATTACHMENT_FEEDBACK_BIT_KHR;
   T("other attachment flags refuse",
     hk_render_pack_continues(&a, &b, false, false, false));
   RESET();
   b.att[2].aliasable = true;
   T("an aliasable image continues after a framebuffer-local dependency",
     !hk_render_pack_continues(&a, &b, false, false, false));
   b.layout_proof = true;
   T("but refuses an ordered attachment-visible one",
     hk_render_pack_continues(&a, &b, true, false, true));
   T("aliasable image continues across ordered non-attachment-visible barrier",
     !hk_render_pack_continues(&a, &b, true, false, false));
   RESET();
   b.layout_proof = false;
   T("without attachment-only layouts, a local dependency continues",
     !hk_render_pack_continues(&a, &b, false, false, false));
   T("without attachment-only layouts, an ordered att-visible one does not",
     hk_render_pack_continues(&a, &b, true, false, true));
   T("without attachment-only layouts, non-att-visible ordered continues",
     !hk_render_pack_continues(&a, &b, true, false, false));
   RESET();
   memset(&a.att, 0, sizeof(a.att));
   memset(&b.att, 0, sizeof(b.att));
   a.color_count = b.color_count = 0;
   T("attachmentless passes refuse",
     hk_render_pack_continues(&a, &b, false, false, false));
#undef RESET
}

static void
test_streams_match(void)
{
   struct hk_cmd_buffer *ca = split(10), *cb = split(20);
   struct hk_cs *a = first_cs(ca), *b = first_cs(cb);
   T("identical render registers match", !hk_render_pack_streams_match(a, b));

   b->tib.sample_size_B = 16;
   T("another tilebuffer refuses", hk_render_pack_streams_match(a, b));
   b->tib = a->tib;

   b->cr.bg.main.usc = bg + 0x40;
   b->cr.eot.main.usc = eot + 0x40;
   b->cr.eot.partial.usc = eot + 0x80;
   T("the same programs recorded in another command buffer match",
     !hk_render_pack_streams_match(a, b));
   b->render_pack_elided = 0x2;
   T("an attachment elided in one stream only refuses",
     hk_render_pack_streams_match(a, b));
   a->render_pack_elided = 0x2;
   T("the same attachment elided in both matches",
     !hk_render_pack_streams_match(a, b));
   a->render_pack_elided = b->render_pack_elided = 0;
   b->cr.bg.main = a->cr.bg.main;
   b->cr.eot.main = a->cr.eot.main;
   b->cr.eot.partial = a->cr.eot.partial;
   b->cr.bg.partial.rsrc_spec = 1;
   T("tile programs with other resources refuse",
     hk_render_pack_streams_match(a, b));
   b->cr.bg.partial = a->cr.bg.partial;
   b->cr.zls_control.opaque[0] ^= 1;
   T("other depth/stencil load or store refuses",
     hk_render_pack_streams_match(a, b));
   b->cr.zls_control = a->cr.zls_control;
   agx_pack(&a->cr.zls_control, ZLS_CONTROL, cfg) {
      cfg.z_load = false;
      cfg.z_store = true;
      cfg.z_store_tiling = AGX_ZLS_TILING_TWIDDLED;
      cfg.z_store_compress = true;
   }
   agx_pack(&b->cr.zls_control, ZLS_CONTROL, cfg) {
      cfg.z_load = true;
      cfg.z_load_tiling = AGX_ZLS_TILING_TWIDDLED;
      cfg.z_load_compress = true;
      cfg.z_store = true;
      cfg.z_store_tiling = AGX_ZLS_TILING_TWIDDLED;
      cfg.z_store_compress = true;
   }
   a->cr.process_empty_tiles = true;
   b->cr.process_empty_tiles = false;
   a->cr.bg.main.rsrc_spec = 0x111;
   b->cr.bg.main.rsrc_spec = 0x222;
   T("clear-then-load continues across ZLS, empty tiles, and main BG",
     !hk_render_pack_streams_match(a, b));
   a->cr.process_empty_tiles = false;
   b->cr.process_empty_tiles = true;
   T("load-then-clear empty-tiles mismatch refuses",
     hk_render_pack_streams_match(a, b));
   b->cr.process_empty_tiles = false;
   agx_pack(&a->cr.zls_control, ZLS_CONTROL, cfg) {
      cfg.z_load = true;
      cfg.z_store = true;
   }
   agx_pack(&b->cr.zls_control, ZLS_CONTROL, cfg) {
      cfg.z_load = false;
      cfg.z_store = true;
   }
   T("load-then-clear ZLS mismatch refuses",
     hk_render_pack_streams_match(a, b));
   a->cr.bg.main.rsrc_spec = b->cr.bg.main.rsrc_spec = 0;
   memset(&a->cr.zls_control, 0, sizeof(a->cr.zls_control));
   b->cr.zls_control = a->cr.zls_control;
   b->cr.depth.buffer += 0x1000;
   T("another depth buffer refuses", hk_render_pack_streams_match(a, b));
   b->cr.depth = a->cr.depth;

   b->cr.isp_bgobjdepth = fui(0.25f);
   T("another depth clear value refuses where depth is not loaded",
     hk_render_pack_streams_match(a, b));
   agx_pack(&a->cr.zls_control, ZLS_CONTROL, cfg) {
      cfg.z_load = cfg.z_store = true;
   }
   b->cr.zls_control = a->cr.zls_control;
   T("but matches where depth is loaded", !hk_render_pack_streams_match(a, b));
   b->cr.stencil.buffer = a->cr.stencil.buffer = UINT64_C(0x4000000);
   T("and refuses again where stencil is not",
     hk_render_pack_streams_match(a, b));
   memset(&a->cr.stencil, 0, sizeof(a->cr.stencil));
   memset(&b->cr.stencil, 0, sizeof(b->cr.stencil));
   b->cr.isp_bgobjdepth = a->cr.isp_bgobjdepth;
   memset(&a->cr.zls_control, 0, sizeof(a->cr.zls_control));
   b->cr.zls_control = a->cr.zls_control;

   a->cr.bg.main.anatomy_ops[0] = 7;
   a->cr.color_compressed_mask = 3;
   T("profiler-only fields do not matter", !hk_render_pack_streams_match(a, b));

   b->cr.dbias_is_int = U_TRISTATE_YES;
   T("depth bias format set in one only matches",
     !hk_render_pack_streams_match(a, b));
   a->cr.dbias_is_int = U_TRISTATE_NO;
   T("conflicting depth bias formats refuse",
     hk_render_pack_streams_match(a, b));
   a->cr.dbias_is_int = b->cr.dbias_is_int;

   b->has_sample_locations = true;
   T("other sample positions refuse", hk_render_pack_streams_match(a, b));
   b->has_sample_locations = false;
   b->area_x1 = 2268;
   T("another render area refuses", hk_render_pack_streams_match(a, b));
   b->area_x1 = a->area_x1;
   b->fragment_attachments.count = 1;
   T("another attachment set refuses", hk_render_pack_streams_match(a, b));
   b->fragment_attachments.count = 0;
   b->tile_resources.primary_bytes = 64;
   T("other tile resources refuse", hk_render_pack_streams_match(a, b));
}

static void
test_arrays(void)
{
   const uint32_t s = 0x11223344, t = 0x55667788;
   struct util_dynarray one, two, other, none;
   util_dynarray_init(&one, NULL);
   util_dynarray_init(&two, NULL);
   util_dynarray_init(&other, NULL);
   util_dynarray_init(&none, NULL);
   util_dynarray_append(&one, s);
   util_dynarray_append(&two, s);
   util_dynarray_append(&two, s);
   util_dynarray_append(&other, t);

   const struct util_dynarray *longer = NULL;
   T("a prefix fits", hk_render_pack_array_prefix(&one, &two, &longer) &&
                         longer == &two);
   T("either way round", hk_render_pack_array_prefix(&two, &one, &longer) &&
                            longer == &two);
   T("an empty array fits any", hk_render_pack_array_prefix(&none, &other,
                                                            &longer) &&
                                   longer == &other);
   T("different entries refuse",
     !hk_render_pack_array_prefix(&one, &other, &longer));
   util_dynarray_fini(&one);
   util_dynarray_fini(&two);
   util_dynarray_fini(&other);
}

static void
test_member(void)
{
   struct hk_cmd_buffer *c = split(4);
   T("a single-render command buffer is a member", hk_render_pack_member(c));

   T("a member has no refusal", !hk_render_pack_member_refusal(c));

   c = split(4);
   first_cs(c)->g17p_pass_ends = false;
   T("a split pass without SUSPENDING_BIT is not a member",
     !hk_render_pack_member(c));
   c->render_pack.tail_pass.flags = VK_RENDERING_SUSPENDING_BIT;
   T("a suspending pass is a member", hk_render_pack_member(c));
   first_cs(c)->g17p_pass_begins = false;
   T("without RESUMING_BIT it refuses", !hk_render_pack_member(c));
   c->render_pack.head_pass.flags = VK_RENDERING_RESUMING_BIT;
   T("a resuming+suspending pass is a member", hk_render_pack_member(c));

   struct hk_cmd_buffer *fixup = new_cmd();
   bool is_res = false, is_susp = false;
   T("plain empty cmd is not a suspend/resume fixup",
     !hk_render_pack_is_empty_fixup(fixup, &is_res, &is_susp));
   fixup->render_pack.empty_resume = true;
   gbuffer(&fixup->render_pack.empty_pass);
   T("empty resume fixup is recognized",
     hk_render_pack_is_empty_fixup(fixup, &is_res, &is_susp) && is_res && !is_susp);
   T("matching fixup pass matches member",
     hk_render_pack_fixup_matches(&fixup->render_pack.empty_pass,
                                  &c->render_pack.tail_pass));
   fixup->render_pack.empty_pass.att[0].iview = views[3];
   T("mismatched fixup view refuses",
     !hk_render_pack_fixup_matches(&fixup->render_pack.empty_pass,
                                   &c->render_pack.tail_pass));
   fixup->sync.req[0][HK_SYNC_VDM].prior = HK_SYNC_GFX;
   T("empty fixup with barrier requirement refuses",
     !hk_render_pack_is_empty_fixup(fixup, &is_res, &is_susp));

   c = split(4);
   first_cs(c)->stats.cmds = 0;
   T("a render without draws is not", !hk_render_pack_member(c));
   T("and says so",
     !strcmp(hk_render_pack_member_refusal(c) ?: "", "no draws"));

   c = split(4);
   add_stream(c, true);
   T("a command buffer with two streams is not", !hk_render_pack_member(c));

   c = new_cmd();
   add_stream(c, true)->stats.cmds = 1;
   T("a compute command buffer is not", !hk_render_pack_member(c));

   c = split(4);
   first_cs(c)->timestamp.end.handle = 1;
   T("a render with an end stamp is a member", hk_render_pack_member(c));
   T("taking the fragment end",
     hk_render_pack_stamp_slots(first_cs(c)) == 2);

   c = split(4);
   first_cs(c)->timestamp.vtx_end.handle = 1;
   T("a render with a vertex-end stamp is a member",
     hk_render_pack_member(c) && hk_render_pack_stamp_slots(first_cs(c)) == 1);

   c = split(4);
   first_cs(c)->timestamp.start.handle = 1;
   T("a render with a start stamp is not", !hk_render_pack_member(c));
   T("and says so",
     !strcmp(hk_render_pack_member_refusal(c) ?: "", "timestamp"));

   c = split(4);
   first_cs(c)->oq_stamp[3].handle = 1;
   T("a render making a query available is not", !hk_render_pack_member(c));
   T("and says so",
     !strcmp(hk_render_pack_member_refusal(c) ?: "", "occlusion query"));

   c = split(4);
   c->state.uses_events = true;
   T("a command buffer with events is not", !hk_render_pack_member(c));

   c = split(4);
   c->sync.pending_start_ts.handle = 1;
   T("a pending start stamp is not", !hk_render_pack_member(c));

   c = split(4);
   first_cs(c)->g17p_pass_ends = false;
   T("a render that does not end its pass is not", !hk_render_pack_member(c));

   c = split(4);
   c->vk.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
   T("a secondary command buffer is not", !hk_render_pack_member(c));
}

/* hk_render_pack_boundary() for a run of A alone, as the submission starts
 * one at A. */
static const char *
boundary(const struct hk_cmd_buffer *a, const struct hk_cmd_buffer *b)
{
   struct hk_render_pack_run_state run;
   hk_render_pack_run_start(&run, a, first_cs(a));
   return hk_render_pack_boundary(&run, a, b, first_cs(b), first_cs(a));
}

static void
test_boundary(void)
{
   struct hk_cmd_buffer *a = split(10), *b = split(20);
   struct hk_cs *as = first_cs(a), *bs = first_cs(b);
   T("adjacent splits pack", !boundary(a, b));

   b->render_pack.head = false;
   T("no head refuses", boundary(a, b));
   b->render_pack.head = true;
   a->render_pack.tail = false;
   T("no tail refuses", boundary(a, b));
   a->render_pack.tail = true;

   a->usage_flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
   T("a simultaneous-use command buffer never links",
     boundary(a, b));
   a->usage_flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
   T("a one-time command buffer links",
     !boundary(a, b));
   a->usage_flags = 0;

   b->render_pack.head_dep = HK_M5_MERGE_DEP_END;
   T("a layout transition between them refuses",
     boundary(a, b));

   b->render_pack.head_dep = HK_M5_MERGE_DEP_ORDERED;
   T("an ordered dependency between memory-free streams packs",
     !boundary(a, b));
   as->m5_shader_writes_memory = true;
   T("not when A's shaders write memory",
     boundary(a, b));
   as->m5_shader_writes_memory = false;
   bs->m5_shader_writes_memory = true;
   T("not when B's shaders write memory",
     boundary(a, b));
   bs->m5_shader_writes_memory = false;
   bs->render_pack_unsafe = true;
   T("not when B draws with a hazard",
     boundary(a, b));
   bs->render_pack_unsafe = false;
   b->render_pack.head_pass.layout_proof = false;
   T("not when B's attachments may be descriptors",
     boundary(a, b));
   b->render_pack.head_pass.layout_proof = true;
   b->m5_pushed_images = _mesa_pointer_set_create(test_context);
   _mesa_set_add(b->m5_pushed_images, images[0]);
   T("not when B pushed an attachment as a descriptor",
     boundary(a, b));
   b->render_pack.head_att_not_visible = true;
   b->render_pack.head_pass.layout_proof = false;
   T("non-attachment ordered barrier packs without layout proof or push refusal",
     !boundary(a, b));
   /* Two-barrier availability chain: a LOCAL barrier in A's tail made
    * attachment writes available, then B's head chains an ORDERED barrier
    * with srcAccessMask == 0 (head_att_not_visible == true). Must enforce
    * layout_proof, aliasable, and pushed_attachment. */
   a->render_pack.tail_att_available = true;
   T("tail att_available + head ordered (srcAccess=0) refuses without layout proof",
     boundary(a, b));
   b->render_pack.head_pass.layout_proof = true;
   T("tail att_available + head ordered (srcAccess=0) refuses pushed attachment",
     boundary(a, b));
   b->m5_pushed_images = NULL;
   b->render_pack.head_pass.att[0].aliasable = true;
   T("tail att_available + head ordered (srcAccess=0) refuses aliasable image",
     boundary(a, b));
   b->render_pack.head_pass.att[0].aliasable = false;
   T("tail att_available + head ordered (srcAccess=0) packs when layout proof holds",
     !boundary(a, b));
   a->render_pack.tail_att_available = false;

   /* Same chain from an earlier in-buffer LOCAL availability barrier on A. */
   as->m5_att_available = true;
   b->render_pack.head_pass.layout_proof = false;
   T("in-buffer m5_att_available + head ordered (srcAccess=0) refuses without layout proof",
     boundary(a, b));
   as->m5_att_available = false;

   /* Three-member chain: A->B carries LOCAL att_available, then B->C adds an
    * ORDERED barrier with srcAccessMask == 0 (head_att_not_visible == true). */
   struct hk_cmd_buffer *m0 = split(1), *m1 = split(2), *m2 = split(3);
   m0->render_pack.tail_att_available = true;
   m2->render_pack.head_dep = HK_M5_MERGE_DEP_ORDERED;
   m2->render_pack.head_att_not_visible = true;
   m2->render_pack.head_pass.layout_proof = false;
   struct hk_render_pack_run_state chain;
   hk_render_pack_run_start(&chain, m0, first_cs(m0));
   T("three-member chain: A->B local att_available packs",
     !hk_render_pack_boundary(&chain, m0, m1, first_cs(m1), first_cs(m0)));
   hk_render_pack_run_join(&chain, m0, m1, first_cs(m1));
   T("three-member chain: run carries att_available into B->C ordered (srcAccess=0)",
     chain.att_available &&
     hk_render_pack_boundary(&chain, m1, m2, first_cs(m2), first_cs(m0)) != NULL);
   m2->render_pack.head_pass.layout_proof = true;
   T("three-member chain: B->C packs once layout_proof holds",
     !hk_render_pack_boundary(&chain, m1, m2, first_cs(m2), first_cs(m0)));

   b->render_pack.head_pass.layout_proof = true;
   b->render_pack.head_att_not_visible = false;
   b->m5_pushed_images = NULL;

   b->render_pack.head_dep = HK_M5_MERGE_DEP_LOCAL;
   as->m5_shader_writes_memory = bs->m5_shader_writes_memory = true;
   T("a local dependency packs even when the shaders write memory",
     !boundary(a, b));
   as->m5_ordered = true;
   T("an ordered in-buffer continuation of A makes it ordered",
     boundary(a, b));
   as->m5_ordered = false;
   as->m5_shader_writes_memory = bs->m5_shader_writes_memory = false;

   bs->render_pack_elided = 1;
   T("streams programming another tile pass refuse",
     boundary(a, b));
   bs->render_pack_elided = 0;

   /* Final-member direct colour resolve with STORE -> DONT_CARE. */
   as->cr.whole_attachments = bs->cr.whole_attachments = true;
   as->cr.process_empty_tiles = false;
   bs->cr.process_empty_tiles = true;
   bs->cr.eot.main.primary_storage_B = 64;
   bs->tile_resources.primary_bytes = 64;
   as->fragment_attachments.complete = bs->fragment_attachments.complete = true;
   as->fragment_attachments.count = 1;
   as->fragment_attachments.list[0] =
      (struct drm_asahi_attachment){.pointer = 0x10000, .size = 0x4000};
   bs->fragment_attachments.count = 2;
   bs->fragment_attachments.list[0] = as->fragment_attachments.list[0];
   bs->fragment_attachments.list[1] =
      (struct drm_asahi_attachment){.pointer = 0x20000, .size = 0x1000};
   b->render_pack.head_pass.att[0].store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   b->render_pack.head_pass.att[0].resolve_mode = VK_RESOLVE_MODE_AVERAGE_BIT;
   b->render_pack.tail = false;
   T("final-member colour resolve boundary packs over whole attachments",
     !boundary(a, b));
   as->cr.whole_attachments = bs->cr.whole_attachments = false;
   T("final-member colour resolve refuses partial render areas",
     boundary(a, b));
   as->cr.whole_attachments = bs->cr.whole_attachments = true;
   b->render_pack.head_dep = HK_M5_MERGE_DEP_ORDERED;
   T("final-member colour resolve refuses ordered dependencies",
     boundary(a, b));
   b->render_pack.head_dep = HK_M5_MERGE_DEP_LOCAL;
   b->render_pack.head_non_local_src = true;
   T("final-member colour resolve refuses head non-local graphics barrier",
     boundary(a, b));
   b->render_pack.head_non_local_src = false;
   a->render_pack.tail_non_local_src = true;
   T("final-member colour resolve refuses tail non-local graphics barrier",
     boundary(a, b));
   a->render_pack.tail_non_local_src = false;
   as->m5_non_local_src = true;
   T("final-member colour resolve refuses prior in-buffer non-local barrier",
     boundary(a, b));
   as->m5_non_local_src = false;
   struct hk_cmd_buffer *c_after = split(5);
   T("no member continues after a resolving tail", boundary(b, c_after));

   /* Near-full attachment set with multiple resolve targets and poisoned
    * unused list slots: dry-run boundary() and real hk_render_pack_add() must
    * compute the exact same coalesced union via hk_drm_attachment_set_add(). */
   struct hk_cmd_buffer *fa = split(10), *fb = split(20);
   struct hk_cs *fas = first_cs(fa), *fbs = first_cs(fb);
   fas->cr.whole_attachments = fbs->cr.whole_attachments = true;
   fb->render_pack.head_pass.att[0].store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   fb->render_pack.head_pass.att[0].resolve_mode = VK_RESOLVE_MODE_AVERAGE_BIT;
   fb->render_pack.tail = false;
   hk_drm_attachment_set_init(&fas->fragment_attachments);
   hk_drm_attachment_set_init(&fbs->fragment_attachments);
   for (unsigned i = 0; i < HK_DRM_MAX_FRAGMENT_ATTACHMENTS - 2; ++i) {
      const uint64_t addr = UINT64_C(0x100000) + (uint64_t)i * 0x10000;
      hk_drm_attachment_set_add(&fas->fragment_attachments, addr, 0x4000);
   }
   /* Poison unused slots in A with the 3rd resolve target to verify dry-run
    * never scans unwritten slots beyond A's count. */
   const uint64_t r0 = UINT64_C(0x500000);
   const uint64_t r1 = UINT64_C(0x600000);
   const uint64_t r2 = UINT64_C(0x700000);
   fas->fragment_attachments.list[HK_DRM_MAX_FRAGMENT_ATTACHMENTS - 2] =
      (struct drm_asahi_attachment){.pointer = r2, .size = 0x4000};
   fas->fragment_attachments.list[HK_DRM_MAX_FRAGMENT_ATTACHMENTS - 1] =
      (struct drm_asahi_attachment){.pointer = r1, .size = 0x4000};

   /* B has 2 new disjoint resolve targets plus 1 span overlapping A's first
    * attachment (3 entries in B; 18 + 3 > 20 without coalescing, but 20 with
    * hk_drm_attachment_set_add() overlap coalescing). */
   hk_drm_attachment_set_add(&fbs->fragment_attachments, r0, 0x4000);
   hk_drm_attachment_set_add(&fbs->fragment_attachments, r1, 0x4000);
   hk_drm_attachment_set_add(&fbs->fragment_attachments, UINT64_C(0x102000),
                             0x4000);
   T("near-full attachment set with coalescing resolve targets packs",
     !boundary(fa, fb));
   struct hk_cmd_buffer *fpkt = new_cmd();
   struct hk_cs *fcs = test_alloc(sizeof(*fcs));
   util_dynarray_init(&fcs->render_resources, NULL);
   struct hk_cmd_buffer *fab[] = {fa, fb};
   hk_render_pack_init(fpkt, fcs, fab, 2);
   uint8_t fprec = 0;
   T("near-full attachment set folds cleanly at submit",
     hk_render_pack_add(fpkt, fcs, fa, fas, true, false, &fprec) &&
        hk_render_pack_add(fpkt, fcs, fb, fbs, false, true, &fprec) &&
        fcs->fragment_attachments.count == HK_DRM_MAX_FRAGMENT_ATTACHMENTS &&
        fcs->fragment_attachments.complete);

   /* Adding a 3rd disjoint resolve target overflows even though A's unused
    * slot was poisoned with r2. */
   hk_drm_attachment_set_add(&fbs->fragment_attachments, r2, 0x4000);
   T("near-full attachment set overflowing with 3rd resolve target refuses",
     boundary(fa, fb));
}

/* The tails a submission writes, and that every later submission writes
 * afresh: a packed run submitted twice runs the same links; submitted
 * unpacked or in another order, the recorded terminates come back. */
static void
test_tails(void)
{
   struct hk_cmd_buffer *c[3] = {split(4), split(5), split(6)};
   uint8_t *at[3];
   uint8_t recorded[3][AGX_VDM_STREAM_LINK_CORE_LENGTH];
   for (unsigned i = 0; i < 3; ++i) {
      at[i] = c[i]->render_pack.terminate;
      memcpy(recorded[i], at[i], sizeof(recorded[i]));
   }

   const uint64_t to1 = first_cs(c[1])->addr, to2 = first_cs(c[2])->addr;
   for (unsigned replay = 0; replay < 2; ++replay) {
      const bool w0 = hk_render_pack_write_tail(chip, &c[0]->render_pack, to1);
      const bool w1 = hk_render_pack_write_tail(chip, &c[1]->render_pack, to2);
      const bool w2 = hk_render_pack_write_tail(chip, &c[2]->render_pack, 0);
      T(replay ? "the replayed packed run rewrites nothing"
               : "the packed run links its members",
        w0 == !replay && w1 == !replay && !w2);
      T("member 0 links to member 1", is_link_to(at[0], to1));
      T("member 1 links to member 2", is_link_to(at[1], to2));
      T("the last member keeps its terminate", is_terminate(at[2]));
   }

   /* Submitted alone: back to the recording. */
   for (unsigned i = 0; i < 3; ++i)
      hk_render_pack_write_tail(chip, &c[i]->render_pack, 0);
   for (unsigned i = 0; i < 3; ++i) {
      T("an unpacked submission restores the recorded terminate",
        is_terminate(at[i]) && !memcmp(at[i], recorded[i], 4) &&
           !c[i]->render_pack.linked);
   }

   /* Packed again, in another order: [2, 0]. */
   const uint64_t to0 = first_cs(c[0])->addr;
   hk_render_pack_write_tail(chip, &c[2]->render_pack, to0);
   hk_render_pack_write_tail(chip, &c[0]->render_pack, 0);
   T("another order links its own neighbours",
     is_link_to(at[2], to0) && is_terminate(at[0]) && is_terminate(at[1]));
   hk_render_pack_write_tail(chip, &c[2]->render_pack, 0);
   T("and comes back", is_terminate(at[2]));

   /* A tail is written only where the command buffer recorded one. */
   struct hk_render_pack_edges none = {0};
   T("no terminate, nothing written",
     !hk_render_pack_write_tail(chip, &none, to1));
}

/* An ordered dependency anywhere in a run needs the memory-equivalence proof
 * over every member, not only over the two command buffers around it: in one
 * tile pass a later member's vertex work runs ahead of every earlier
 * member's fragment work. */
static void
test_run_hazard(void)
{
   /* [CB0, CB1, CB2]: CB0's shaders write a storage buffer, CB0 -> CB1 is
    * framebuffer-local, CB1 -> CB2 orders fragment writes before vertex
    * reads. */
   struct hk_cmd_buffer *c0 = split(10), *c1 = split(10), *c2 = split(10);
   first_cs(c0)->m5_shader_writes_memory = true;
   c2->render_pack.head_dep = HK_M5_MERGE_DEP_ORDERED;
   struct hk_render_pack_run_state run;
   hk_render_pack_run_start(&run, c0, first_cs(c0));
   T("CB0 and CB1 pack over a framebuffer-local dependency",
     !hk_render_pack_boundary(&run, c0, c1, first_cs(c1), first_cs(c0)));
   hk_render_pack_run_join(&run, c0, c1, first_cs(c1));
   const char *why =
      hk_render_pack_boundary(&run, c1, c2, first_cs(c2), first_cs(c0));
   T("CB2 does not join a run whose CB0 writes memory",
     why && !strcmp(why, "ordered, writes memory"));
   struct hk_render_pack_run_state from_c1;
   hk_render_pack_run_start(&from_c1, c1, first_cs(c1));
   T("which CB1 and CB2 alone do not show",
     !hk_render_pack_boundary(&from_c1, c1, c2, first_cs(c2), first_cs(c1)));

   /* The ordered dependency first, the writer after it. */
   c0 = split(10), c1 = split(10), c2 = split(10);
   c1->render_pack.head_dep = HK_M5_MERGE_DEP_ORDERED;
   first_cs(c2)->m5_shader_writes_memory = true;
   hk_render_pack_run_start(&run, c0, first_cs(c0));
   T("an ordered run of memory-free members packs",
     !hk_render_pack_boundary(&run, c0, c1, first_cs(c1), first_cs(c0)));
   hk_render_pack_run_join(&run, c0, c1, first_cs(c1));
   T("a writer does not join a run with an ordered dependency",
     hk_render_pack_boundary(&run, c1, c2, first_cs(c2), first_cs(c0)));

   /* An ordered continuation inside the joining member reaches back to every
    * earlier member, and a hazard draw of an earlier member counts. */
   c0 = split(10), c1 = split(10);
   first_cs(c0)->m5_shader_writes_memory = true;
   first_cs(c1)->m5_ordered = true;
   hk_render_pack_run_start(&run, c0, first_cs(c0));
   T("B's own ordered continuation makes the boundary ordered",
     hk_render_pack_boundary(&run, c0, c1, first_cs(c1), first_cs(c0)));
   c0 = split(10), c1 = split(10), c2 = split(10);
   first_cs(c0)->render_pack_unsafe = true;
   c2->render_pack.head_dep = HK_M5_MERGE_DEP_ORDERED;
   hk_render_pack_run_start(&run, c0, first_cs(c0));
   hk_render_pack_run_join(&run, c0, c1, first_cs(c1));
   T("CB0's hazard draw refuses an ordered CB2",
     hk_render_pack_boundary(&run, c1, c2, first_cs(c2), first_cs(c0)));

   /* [CB0, CB1, CB2(resolve)]: CB0 -> CB1 has a FRAGMENT_SHADER ->
    * COLOR_ATTACHMENT_OUTPUT barrier (head_non_local_src), so CB0 and CB1
    * still pack, but a later resolving CB2 must not join the run because CB0's
    * fragment shaders may still read or write CB2's resolve target. */
   c0 = split(10), c1 = split(10), c2 = split(10);
   first_cs(c0)->cr.whole_attachments = first_cs(c1)->cr.whole_attachments =
      first_cs(c2)->cr.whole_attachments = true;
   c1->render_pack.head_non_local_src = true;
   c2->render_pack.head_pass.att[0].store_op = VK_ATTACHMENT_STORE_OP_DONT_CARE;
   c2->render_pack.head_pass.att[0].resolve_mode = VK_RESOLVE_MODE_AVERAGE_BIT;
   c2->render_pack.tail = false;
   hk_render_pack_run_start(&run, c0, first_cs(c0));
   T("CB0 and CB1 pack across a non-local graphics source barrier",
     !hk_render_pack_boundary(&run, c0, c1, first_cs(c1), first_cs(c0)));
   hk_render_pack_run_join(&run, c0, c1, first_cs(c1));
   T("a resolving CB2 does not join a run with an earlier non-local barrier",
     hk_render_pack_boundary(&run, c1, c2, first_cs(c2), first_cs(c0)));
}

/* Builds the packet of members cmds[0..count) the way hk_render_pack_runs()
 * does. */
static struct hk_cmd_buffer *
pack(struct hk_cmd_buffer **cmds, unsigned count, struct hk_cs **out)
{
   struct hk_cmd_buffer *packet = new_cmd();
   struct hk_cs *cs = test_alloc(sizeof(*cs));
   util_dynarray_init(&cs->render_resources, NULL);
   hk_render_pack_init(packet, cs, cmds, count);
   uint8_t preceding = 0;
   for (unsigned m = 0; m < count; ++m)
      hk_render_pack_add(packet, cs, cmds[m], first_cs(cmds[m]), m == 0,
                         m + 1 == count, &preceding);
   hk_render_pack_finish(packet, cs);
   *out = cs;
   return packet;
}

static uint64_t
wait_of(const struct hk_cs *cs, unsigned sq, uint64_t base, bool exact)
{
   const uint16_t b = cs->sync.barrier[sq];
   if (b != DRM_ASAHI_BARRIER_NONE)
      return base + b;
   return exact ? cs->sync.prior[sq] : 0;
}

static void
test_fold(bool exact)
{
#define N(s) (exact ? s " (exact)" : s " (legacy)")
   /* compute C; split A; split B waiting for C (COMPUTE -> GFX before it);
    * then compute D after a GFX -> COMPUTE barrier. */
   struct hk_cmd_buffer *c = new_cmd();
   add_stream(c, true)->stats.cmds = 1;
   struct hk_cmd_buffer *a = split(10);
   struct hk_cmd_buffer *b = new_cmd();
   hk_sync_barrier(b, HK_SYNC_COMPUTE, HK_SYNC_GFX);
   struct hk_cs *bs = add_stream(b, false);
   bs->stats.cmds = 20;
   bs->g17p_pass_begins = bs->g17p_pass_ends = true;
   struct hk_cmd_buffer *d = new_cmd();
   hk_sync_barrier(d, HK_SYNC_GFX, HK_SYNC_COMPUTE);
   struct hk_cs *ds = add_stream(d, true);
   ds->stats.cmds = 1;

   struct hk_cmd_buffer *ab[] = {a, b};
   struct hk_cs *cs;
   struct hk_cmd_buffer *packet = pack(ab, 2, &cs);
   T(N("the packet runs both members' draws"),
     cs->stats.cmds == 30 && cs->render_pack_members == 2);
   T(N("the packet's stream memory is the first member's recording"),
     cs->cmd == packet && cs->stream_owner == a &&
     cs->addr == first_cs(a)->addr);
   T(N("the packet waits for compute work as B did"),
     cs->sync.dep[HK_SYNC_CDM].prior & HK_SYNC_COMPUTE);
   T(N("the packet is one tile pass of two instances"),
     cs->m5_continued_pass && cs->g17p_pass_begins && cs->g17p_pass_ends);
   T(N("the packet stores with the last member's end-of-tile program"),
     cs->cr.eot.main.usc == bs->cr.eot.main.usc &&
        cs->cr.bg.main.usc == first_cs(a)->cr.bg.main.usc);

   struct hk_cmd_buffer *submit[] = {c, packet, d};
   T(N("resolves"), hk_sync_resolve_submit(new_queue(exact), submit, 3, false,
                                           false, 0, NULL));
   T(N("the packet waits for the compute stream before it"),
     wait_of(cs, HK_SYNC_CDM, 0, exact) == first_cs(c)->sync.index);
   T(N("later work waits for the packet"),
     wait_of(ds, HK_SYNC_VDM, 0, exact) == cs->sync.index);

   /* A barrier after A in A's command buffer (its tail) holds the members
    * after A and later command buffers. */
   struct hk_cmd_buffer *a2 = split(10);
   hk_sync_barrier(a2, HK_SYNC_GFX, HK_SYNC_GFX);
   struct hk_cmd_buffer *b2 = split(20);
   struct hk_cmd_buffer *ab2[] = {a2, b2};
   packet = pack(ab2, 2, &cs);
   /* A render that writes no memory is cut for the fragment classes only
    * (HK_SYNC_GFX_MEM); every class the member's tail cuts, the packet's
    * does. */
   unsigned cuts = 0;
   bool same = true;
   for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
      const bool cut = a2->sync.req[k][HK_SYNC_VDM].cut != NULL;
      cuts += cut;
      same &= (packet->sync.req[k][HK_SYNC_VDM].cut == cs) == cut;
   }
   T(N("a member's tail naming it makes later work wait for the packet"),
     cuts && same);

   /* Tile footprints move to the packet. */
   struct hk_cmd_buffer *a3 = split(1), *b3 = split(1);
   struct hk_sync_pass_record rec = {.first = first_cs(a3),
                                     .last = first_cs(a3)};
   util_dynarray_append(&a3->sync_pass_records, rec);
   rec.first = rec.last = first_cs(b3);
   util_dynarray_append(&b3->sync_pass_records, rec);
   struct hk_cmd_buffer *ab3[] = {a3, b3};
   packet = pack(ab3, 2, &cs);
   bool moved = util_dynarray_num_elements(&packet->sync_pass_records,
                                           struct hk_sync_pass_record) == 2;
   util_dynarray_foreach(&packet->sync_pass_records,
                         struct hk_sync_pass_record, r)
      moved &= r->first == cs && r->last == cs;
   T(N("the members' tile footprints are the packet's"), moved);

   /* [A, B]: A's tail orders compute work before image reads of
    * pre-rasterization shaders (COMPUTE -> VERTEX_SHADER on an image), and
    * only B's vertex shader reads images. Unpacked, the resolver applies
    * A's carried requirement to B's vertex half; packed, the packet's vertex
    * half must carry it too, though A, folded first, does not read images
    * there. */
   struct hk_cmd_buffer *a4 = split(10);
   hk_sync_barrier(a4, HK_SYNC_COMPUTE, HK_SYNC_GFX_IMG);
   struct hk_cmd_buffer *b4 = split(10);
   first_cs(b4)->sync.vtx_images = true;
   struct hk_cmd_buffer *ab4[] = {a4, b4};
   packet = pack(ab4, 2, &cs);
   T(N("a later member's image reads carry an earlier tail to the vertex half"),
     cs->sync.vtx_images &&
        (cs->sync.dep_ta[HK_SYNC_CDM].prior & HK_SYNC_COMPUTE));
   /* Final-member resolve fold: merges process_empty_tiles, EOT program,
    * tile resources, and fragment attachment sets. */
   struct hk_cmd_buffer *a5 = split(4), *b5 = split(6);
   struct hk_cs *as5 = first_cs(a5), *bs5 = first_cs(b5);
   as5->cr.process_empty_tiles = false;
   bs5->cr.process_empty_tiles = true;
   bs5->cr.eot.main.usc = eot + 0x100;
   bs5->cr.eot.main.primary_storage_B = 96;
   bs5->tile_resources.primary_bytes = 96;
   bs5->tile_resources.secondary_bytes = 32;
   as5->fragment_attachments.complete = bs5->fragment_attachments.complete = true;
   as5->fragment_attachments.count = 1;
   as5->fragment_attachments.list[0] =
      (struct drm_asahi_attachment){.pointer = 0x10000, .size = 0x4000};
   bs5->fragment_attachments.count = 2;
   bs5->fragment_attachments.list[0] = as5->fragment_attachments.list[0];
   bs5->fragment_attachments.list[1] =
      (struct drm_asahi_attachment){.pointer = 0x20000, .size = 0x1000};
   struct hk_cmd_buffer *ab5[] = {a5, b5};
   packet = pack(ab5, 2, &cs);
   T(N("final-member resolve fold updates EOT, empty-tile flag, tile resources and attachments"),
     cs->cr.eot.main.usc == eot + 0x100 &&
        cs->cr.eot.main.primary_storage_B == 96 &&
        cs->cr.process_empty_tiles &&
        cs->tile_resources.primary_bytes == 96 &&
        cs->tile_resources.secondary_bytes == 32 &&
        cs->fragment_attachments.count == 2 &&
        cs->fragment_attachments.list[1].pointer == 0x20000 &&
        cs->fragment_attachments.complete);

   /* A leading top-of-pipe stamp on the first member's vertex end and an
    * end stamp on the last member's fragment end ride on the packet. */
   struct hk_cmd_buffer *a6 = split(10), *b6 = split(10);
   first_cs(a6)->timestamp.vtx_end =
      (struct agx_timestamp_req){.handle = 7, .offset_B = 8};
   first_cs(b6)->timestamp.end =
      (struct agx_timestamp_req){.handle = 7, .offset_B = 16};
   struct hk_cmd_buffer *ab6[] = {a6, b6};
   packet = pack(ab6, 2, &cs);
   T(N("the packet carries the members' vertex-end and fragment-end stamps"),
     cs->timestamp.vtx_end.offset_B == 8 && cs->timestamp.end.offset_B == 16 &&
        !cs->timestamp.start.handle);
#undef N
}

/* hk_sync.c references these for paths the fixtures never take. */
void
hk_dispatch_imm_writes(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd; (void)cs;
   abort();
}

void
hk_tess_batch_flush(struct hk_cmd_buffer *cmd)
{
   (void)cmd;
   abort();
}

bool
hk_cs_init_graphics(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd; (void)cs;
   abort();
}

void
hk_record_stamp_copies(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                       const struct hk_stamp_copy *copies, unsigned count)
{
   (void)cmd; (void)cs; (void)copies; (void)count;
   abort();
}

bool
hk_defer_oq_available(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd; (void)cs;
   abort();
}

void
hk_cdm_end_stream_flush(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd; (void)cs;
}

void
hk_flush_query_resets(struct hk_cmd_buffer *cmd, struct hk_cs *cs)
{
   (void)cmd;
   if (cs->query_reset_pending)
      abort();
}

void
hk_g17p_meta_fill_addr(struct hk_cmd_buffer *cmd, struct hk_cs **target,
                       uint64_t address, uint64_t size, uint32_t value)
{
   (void)cmd; (void)target; (void)address; (void)size; (void)value;
   abort();
}

struct agx_ptr
hk_pool_alloc_from_slow(struct hk_cmd_buffer *cmd, struct hk_uploader *uploader,
                        uint32_t size, uint32_t alignment,
                        enum hk_cmd_bo_type type)
{
   (void)cmd; (void)uploader; (void)size; (void)alignment; (void)type;
   abort();
}

struct agx_ptr
hk_pool_cs_alloc(struct hk_cmd_buffer *cmd, uint32_t size, uint32_t alignment)
{
   (void)cmd; (void)size; (void)alignment;
   abort();
}

void
hk_flush_deferred_clears_why(struct hk_cmd_buffer *cmd, const char *why)
{
   (void)cmd; (void)why;
   abort();
}

void
hk_dispatch_precomp(struct hk_cmd_buffer *cmd, struct agx_grid grid,
                    enum agx_barrier barrier, enum libagx_program idx,
                    void *data, size_t data_size)
{
   (void)cmd; (void)grid; (void)barrier; (void)idx; (void)data; (void)data_size;
   abort();
}

static void
test_pregfx_and_union_elision(void)
{
   /* 1. Driver-only [pre_gfx, VDM, post_gfx] CB on a ring heap slot. */
   struct hk_cmd_buffer *c = split(4);
   struct hk_cs *vdm = first_cs(c);
   struct hk_cs *pre = test_alloc(sizeof(*pre));
   struct hk_cs *post = test_alloc(sizeof(*post));
   pre->type = HK_CS_CDM;
   pre->stats.cmds = 1;
   pre->stats.calls = 0;
   pre->sync.kind = HK_SYNC_GFX | HK_SYNC_GFX_MEM;
   pre->sync.heap_slot = 2;
   vdm->sync.heap_slot = 2;
   vdm->sync.dep[HK_SYNC_CDM].cut = pre;
   vdm->sync.dep_ta[HK_SYNC_CDM].cut = pre;
   post->type = HK_CS_CDM;
   post->stats.cmds = 1;
   post->stats.calls = 0;
   post->sync.kind = HK_SYNC_GFX | HK_SYNC_GFX_MEM;
   post->sync.heap_slot = 2;
   post->sync.dep[HK_SYNC_VDM].cut = vdm;
   post->sync.dep_ta[HK_SYNC_VDM].cut = vdm;
   list_inithead(&c->control_streams);
   list_addtail(&pre->node, &c->control_streams);
   list_addtail(&vdm->node, &c->control_streams);
   list_addtail(&post->node, &c->control_streams);

   const struct hk_cs *got_pre = NULL, *got_post = NULL;
   const struct hk_cs *got_vdm =
      hk_render_pack_member_streams(c, false, &got_pre, &got_post);
   T("driver-only [pre, VDM, post] with matching ring slot is a member",
     got_vdm == vdm && got_pre == pre && got_post == post);

   pre->stats.calls = 1;
   pre->sync.kind = HK_SYNC_COMPUTE | HK_SYNC_GFX | HK_SYNC_GFX_MEM;
   T("API compute dispatch before VDM is admitted",
     hk_render_pack_member(c) == vdm);
   pre->render_pack_compute_internal = true;
   T("meta/internal compute dispatch before VDM is refused on later member",
     hk_render_pack_member(c) == NULL);
   T("meta/internal compute dispatch before VDM is admitted on first member",
     hk_render_pack_member_streams(c, true, &got_pre, &got_post) == vdm &&
        got_pre == pre && got_post == post);
   pre->sync.kind = HK_SYNC_XFER | HK_SYNC_COMPUTE | HK_SYNC_GFX | HK_SYNC_GFX_MEM;
   T("transfer+compute before VDM is refused on later member",
     hk_render_pack_member(c) == NULL);
   T("transfer+compute before VDM is admitted on first member",
     hk_render_pack_member_streams(c, true, &got_pre, &got_post) == vdm);
   pre->timestamp.end.handle = 9;
   T("first member pre with timestamp is still refused",
     hk_render_pack_member_streams(c, true, &got_pre, &got_post) == NULL);
   pre->timestamp.end.handle = 0;
   pre->render_pack_compute_internal = false;
   pre->sync.kind = HK_SYNC_GFX | HK_SYNC_GFX_MEM;
   pre->stats.calls = 0;

   /* Hoisting refusal gate: must refuse hoisting pre_cs past an earlier
    * member's dependent/non-heap post_cs, same-slot heap reset, memory
    * writes, or VDM->CDM ordering dependencies, while allowing driver
    * pre-graphics on a disjoint heap slot past a pure heap-slot reset. */
   struct hk_render_pack_run_state hoist = {0};
   uint8_t tail_req[HK_SYNC_NUM_KINDS] = {0};
   uint8_t post_tail_req[HK_SYNC_NUM_KINDS] = {0};
   const uint8_t slot1_mask = BITFIELD_BIT(0);
   const uint8_t slot2_mask = BITFIELD_BIT(1);
   T("pure heap-reset post_cs recognized",
     hk_render_pack_post_is_heap_reset(post));
   post->query_access = true;
   T("query-access post_cs is not a pure heap reset",
     !hk_render_pack_post_is_heap_reset(post));
   post->query_access = false;

   pre->stats.calls = 1;
   pre->sync.kind = HK_SYNC_COMPUTE;
   pre->sync.heap_slot = 0;
   T("clean API pre_cs hoists when run_post == 0",
     hk_render_pack_pre_cs_refusal(&hoist, pre, 0, 0, true, HK_SYNC_GFX,
                                   tail_req, post_tail_req) == NULL);
   T("API pre_cs refused when earlier member has pure heap post_cs",
     hk_render_pack_pre_cs_refusal(&hoist, pre, 1, slot1_mask, true,
                                   HK_SYNC_GFX, tail_req,
                                   post_tail_req) != NULL);

   pre->stats.calls = 0;
   pre->sync.kind = HK_SYNC_GFX | HK_SYNC_GFX_MEM;
   pre->sync.heap_slot = 2;
   T("driver pre_cs on disjoint heap slot hoists past pure heap-reset post_cs",
     hk_render_pack_pre_cs_refusal(&hoist, pre, 1, slot1_mask, true,
                                   HK_SYNC_GFX, tail_req,
                                   post_tail_req) == NULL);
   T("driver pre_cs on same heap slot refused after heap-reset post_cs",
     hk_render_pack_pre_cs_refusal(&hoist, pre, 1, slot2_mask, true,
                                   HK_SYNC_GFX, tail_req,
                                   post_tail_req) != NULL);
   T("driver pre_cs refused after non-heap-reset post_cs",
     hk_render_pack_pre_cs_refusal(&hoist, pre, 1, slot1_mask, false,
                                   HK_SYNC_GFX, tail_req,
                                   post_tail_req) != NULL);
   pre->sync.dep[HK_SYNC_CDM].prior = HK_SYNC_GFX;
   T("driver pre_cs with CDM prior dependency refused after post_cs",
     hk_render_pack_pre_cs_refusal(&hoist, pre, 1, slot1_mask, true,
                                   HK_SYNC_GFX, tail_req,
                                   post_tail_req) != NULL);
   pre->sync.dep[HK_SYNC_CDM].prior = 0;
   post_tail_req[util_logbase2(HK_SYNC_GFX)] = HK_SYNC_GFX;
   T("driver pre_cs refused when earlier tail barrier orders post_cs before pre_cs",
     hk_render_pack_pre_cs_refusal(&hoist, pre, 1, slot1_mask, true,
                                   HK_SYNC_GFX, tail_req,
                                   post_tail_req) != NULL);
   post_tail_req[util_logbase2(HK_SYNC_GFX)] = 0;
   pre->sync.kind = HK_SYNC_XFER;
   T("non-GFX driver pre_cs refused after post_cs",
     hk_render_pack_pre_cs_refusal(&hoist, pre, 1, slot1_mask, true,
                                   HK_SYNC_GFX, tail_req,
                                   post_tail_req) != NULL);
   pre->sync.kind = HK_SYNC_GFX | HK_SYNC_GFX_MEM;
   T("driver pre_cs allowed when run_post == 0",
     hk_render_pack_pre_cs_refusal(&hoist, pre, 0, 0, true, HK_SYNC_GFX,
                                   tail_req, post_tail_req) == NULL);

   pre->stats.calls = 1;
   pre->sync.kind = HK_SYNC_COMPUTE;
   pre->sync.heap_slot = 0;
   hoist.writes_memory = true;
   T("API pre_cs refused after memory-writing render",
     hk_render_pack_pre_cs_refusal(&hoist, pre, 0, 0, true, HK_SYNC_GFX,
                                   tail_req, post_tail_req) != NULL);
   hoist.writes_memory = false;
   pre->sync.dep[HK_SYNC_VDM].prior = HK_SYNC_GFX;
   T("API pre_cs refused when depending on earlier render",
     hk_render_pack_pre_cs_refusal(&hoist, pre, 0, 0, true, HK_SYNC_GFX,
                                   tail_req, post_tail_req) != NULL);
   pre->sync.dep[HK_SYNC_VDM].prior = 0;
   tail_req[util_logbase2(HK_SYNC_COMPUTE)] = HK_SYNC_GFX;
   T("API pre_cs refused when earlier tail barrier orders VDM->CDM",
     hk_render_pack_pre_cs_refusal(&hoist, pre, 0, 0, true, HK_SYNC_GFX,
                                   tail_req, post_tail_req) != NULL);
   tail_req[util_logbase2(HK_SYNC_COMPUTE)] = 0;
   pre->sync.kind = HK_SYNC_GFX | HK_SYNC_GFX_MEM;
   pre->sync.heap_slot = 2;
   pre->stats.calls = 0;

   post->stats.calls = 1;
   T("API compute dispatch after VDM is refused",
     hk_render_pack_member(c) == NULL);
   post->stats.calls = 0;

   post->sync.heap_slot = 3;
   T("post-graphics with mismatched heap slot is refused",
     hk_render_pack_member(c) == NULL);
   post->sync.heap_slot = 2;

   /* 2. Union untouched-RT elision + union ZLS store elision. */
   struct hk_cmd_buffer *a = split(2), *b = split(3);
   struct hk_cs *as = first_cs(a), *bs = first_cs(b);
   as->cr.bg.main.rsrc_spec = 0x111;
   as->cr.eot.main.rsrc_spec = 0x222;
   as->cr.eot.partial.rsrc_spec = 0x333;
   bs->cr.bg.main.rsrc_spec = 0x444; /* elided program */
   bs->cr.eot.main.rsrc_spec = 0x555; /* elided program */
   bs->cr.eot.partial.rsrc_spec = 0x666;
   bs->render_pack_elided = 0x2;
   bs->render_pack_has_full_programs = true;
   bs->render_pack_full_bg_main = as->cr.bg.main;
   bs->render_pack_full_eot_main = as->cr.eot.main;
   bs->render_pack_full_eot_partial = as->cr.eot.partial;

   agx_pack(&as->cr.zls_control, ZLS_CONTROL, cfg) {
      cfg.z_load = true;
      cfg.z_store = true;
   }
   agx_pack(&bs->cr.zls_control, ZLS_CONTROL, cfg) {
      cfg.z_load = true;
      cfg.z_store = false;
   }
   bs->render_pack_zls_elided_store = VK_IMAGE_ASPECT_DEPTH_BIT;

   T("differing elision with preserved full programs and elided z_store matches",
     !hk_render_pack_streams_match(as, bs));

   struct hk_cmd_buffer pkt = {0};
   struct hk_cs pcs = {0};
   struct hk_cmd_buffer *members[2] = {b, a};
   hk_render_pack_init(&pkt, &pcs, members, 2);
   uint8_t preceding = 0;
   T("packet add b then a succeeds",
     hk_render_pack_add(&pkt, &pcs, b, bs, true, false, &preceding) &&
        hk_render_pack_add(&pkt, &pcs, a, as, false, true, &preceding));
   hk_render_pack_finish(&pkt, &pcs);
   agx_unpack(NULL, &pcs.cr.zls_control, ZLS_CONTROL, zout);
   T("packet restores full bg/eot and unions z_store across members",
     pcs.render_pack_elided == 0 && pcs.cr.bg.main.rsrc_spec == 0x111 &&
        pcs.cr.eot.partial.rsrc_spec == 0x333 &&
        pcs.cr.eot.main.rsrc_spec == 0x222 && zout.z_store);

   /* Reverse order: last member is the elided one -> must restore full eot.main. */
   struct hk_cmd_buffer *rev[2] = {a, b};
   hk_render_pack_init(&pkt, &pcs, rev, 2);
   preceding = 0;
   T("packet add a then b succeeds",
     hk_render_pack_add(&pkt, &pcs, a, as, true, false, &preceding) &&
        hk_render_pack_add(&pkt, &pcs, b, bs, false, true, &preceding));
   hk_render_pack_finish(&pkt, &pcs);
   agx_unpack(NULL, &pcs.cr.zls_control, ZLS_CONTROL, zout2);
   T("packet restores last member's full eot.main and preserves z_store",
     pcs.render_pack_elided == 0 && pcs.cr.eot.main.rsrc_spec == 0x222 &&
        zout2.z_store);

   util_dynarray_fini(&pcs.render_resources);
   util_dynarray_fini(&pkt.sync_pass_records);
}

static void
test_vtx_img_bloom(void)
{
   struct hk_cmd_buffer *m0 = split(10), *m1 = split(20);
   struct hk_cs *c0 = first_cs(m0), *c1 = first_cs(m1);
   const uint64_t b_a = hk_sync_image_bloom(0x100000);
   uint64_t k_b = 0x200000;
   while (hk_sync_image_bloom(k_b) == b_a)
      k_b += 0x10000;
   const uint64_t b_b = hk_sync_image_bloom(k_b);

   c0->sync.att_bloom = b_a;
   c1->sync.att_bloom = b_b;
   c0->sync.vtx_images = true;
   c0->sync.vtx_img_bloom = b_a;
   c0->sync.dep_ta_img_prior = HK_SYNC_GFX;
   c1->sync.vtx_images = true;
   c1->sync.vtx_img_bloom = b_b;
   m0->sync.req[util_logbase2(HK_SYNC_GFX_IMG)][HK_SYNC_VDM].prior = HK_SYNC_XFER;

   struct hk_cmd_buffer *packet = new_cmd();
   struct hk_cs *pcs = test_alloc(sizeof(*pcs));
   struct hk_cmd_buffer *members[] = {m0, m1};
   hk_render_pack_init(packet, pcs, members, 2);
   uint8_t preceding = 0;
   T("bloom pack add m0",
     hk_render_pack_add(packet, pcs, m0, c0, true, false, &preceding));
   T("bloom pack add m1",
     hk_render_pack_add(packet, pcs, m1, c1, false, true, &preceding));
   hk_render_pack_finish(packet, pcs);

   T("packed stream unions att_bloom and vtx_img_bloom safely",
     pcs->sync.vtx_images && !pcs->sync.vtx_img_unsafe &&
     pcs->sync.att_bloom == (b_a | b_b) &&
     pcs->sync.vtx_img_bloom == (b_a | b_b) &&
     (pcs->sync.dep_ta_img_prior & (HK_SYNC_GFX | HK_SYNC_XFER)) ==
        (HK_SYNC_GFX | HK_SYNC_XFER) &&
     !(pcs->sync.dep_ta[HK_SYNC_VDM].prior & HK_SYNC_XFER));

   /* One unsafe member upgrades the packet to conservative vtx_img_unsafe. */
   c1->sync.vtx_img_unsafe = true;
   hk_render_pack_init(packet, pcs, members, 2);
   preceding = 0;
   T("unsafe bloom pack add m0",
     hk_render_pack_add(packet, pcs, m0, c0, true, false, &preceding));
   T("unsafe bloom pack add m1",
     hk_render_pack_add(packet, pcs, m1, c1, false, true, &preceding));
   hk_render_pack_finish(packet, pcs);
   T("unsafe member routes image req into dep_ta[VDM]",
     pcs->sync.vtx_img_unsafe &&
     (pcs->sync.dep_ta[HK_SYNC_VDM].prior & HK_SYNC_XFER));
}

int
main(void)
{
   test_context = ralloc_context(NULL);
   for (unsigned i = 0; i < ARRAY_SIZE(views); ++i)
      views[i] = test_alloc(64);

   static const enum agx_chip chips[] = {AGX_CHIP_G17S, AGX_CHIP_G16G};
   for (unsigned i = 0; i < ARRAY_SIZE(chips); ++i) {
      chip = chips[i];
      printf("# %s\n", agx_chip_name(chip));
      test_non_local_src();
      test_continues();
      test_streams_match();
      test_arrays();
      test_member();
      test_boundary();
      test_run_hazard();
      test_tails();
      test_fold(false);
      test_fold(true);
      test_pregfx_and_union_elision();
      test_vtx_img_bloom();
   }

   ralloc_free(test_context);
   printf("%u tests, %u failures\n", tests, failures);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

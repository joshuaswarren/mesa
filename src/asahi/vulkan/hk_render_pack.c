/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_render_pack.h"

#include <string.h>

#include "asahi/lib/agx_vdm_stream_link_core.h"
#include "asahi/lib/agx_vdm_stream_terminate_core.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "util/set.h"
#include "hk_cmd_buffer.h"

static_assert(HK_RENDER_PACK_COLORS == HK_MAX_RTS,
              "one render pack slot per colour attachment");
static_assert(AGX_VDM_STREAM_TERMINATE_CORE_LENGTH <=
                 AGX_VDM_STREAM_LINK_CORE_LENGTH,
              "a stream link covers the terminate record it replaces");

const char *
hk_render_pack_continues(const struct hk_render_pack_pass *a,
                         const struct hk_render_pack_pass *b, bool ordered)
{
   const VkRenderingFlags open = VK_RENDERING_SUSPENDING_BIT |
                                 VK_RENDERING_RESUMING_BIT |
                                 VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT;
   if ((a->flags | b->flags) & open)
      return "flags";
   if (memcmp(&a->area, &b->area, sizeof(a->area)) ||
       a->layer_count != b->layer_count || a->view_mask != b->view_mask ||
       a->color_count != b->color_count)
      return "geometry";
   /* The in-buffer continuation keeps multiview passes apart too
    * (hk_m5_pass_merge_candidate()). */
   if (a->view_mask)
      return "multiview";

   unsigned attachments = 0;
   for (unsigned i = 0; i < HK_RENDER_PACK_SLOTS; ++i) {
      if (i >= a->color_count && i < HK_RENDER_PACK_COLORS)
         continue;

      const struct hk_render_pack_att *x = &a->att[i];
      const struct hk_render_pack_att *y = &b->att[i];
      if ((x->iview != NULL) != (y->iview != NULL))
         return "presence";
      if (!x->iview)
         continue;

      attachments++;
      if (x->iview != y->iview)
         return "view";
      if (x->samples != y->samples)
         return "sample count";
      /* B continues what A left in the tiles: it must load it, and store it
       * as A would have. */
      if (y->load_op != VK_ATTACHMENT_LOAD_OP_LOAD)
         return "load op";
      if (y->store_op != x->store_op)
         return "store op";
      if (x->resolve_mode != VK_RESOLVE_MODE_NONE ||
          y->resolve_mode != VK_RESOLVE_MODE_NONE)
         return "resolve";
      if (x->flags != y->flags)
         return "flags";
      /* The same layout in both: a layout transition between them that
       * recorded no work left the memory as A stored it
       * (hk_render_pack_classify()). */
      if (x->layout != y->layout)
         return "layout";
      /* Nothing in B may read the attachment's memory, which A's store would
       * have written and the merged pass writes only at its end. Feedback
       * loops and aliases are the ways it could without a descriptor. An
       * alias needs a dependency that makes A's attachment writes visible to
       * B's shaders, which only an ordered one does: after a
       * framebuffer-local one, a read through an alias races with A's
       * writes whether or not the instances pack. The in-buffer continuation
       * draws the same line (hk_m5_ordered_attachment_refusal()). */
      if (x->layout == VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT ||
          y->layout == VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT)
         return "feedback loop layout";
      if ((x->flags | y->flags) &
          VK_RENDERING_ATTACHMENT_INPUT_ATTACHMENT_FEEDBACK_BIT_KHR)
         return "input attachment feedback";
      if (x->feedback_loop || y->feedback_loop)
         return "feedback loop";
      if (ordered && (x->aliasable || y->aliasable))
         return "ordered with an aliasable image";
   }
   if (!attachments)
      return "no attachments";

   /* An ordered dependency makes A's attachment writes visible to B's
    * shaders; the merged pass has not stored them yet. Within one command
    * buffer the continuation may instead show that no descriptor set is
    * bound, which another command buffer's recording cannot. */
   if (ordered && !b->layout_proof)
      return "ordered without attachment-only layouts";
   return NULL;
}

/* The resources two tile programs take. The programs themselves are named
 * by address, and on M5 every command buffer records its own copies
 * (hk_cmd_reserve_ioto_program()), so equal programs of two command buffers
 * have different addresses; hk_render_pack_streams_match() proves them the
 * same programs from what built them instead. */
static bool
bg_eot_resources_equal(const struct hk_bg_eot *a, const struct hk_bg_eot *b)
{
   return a->primary_storage_B == b->primary_storage_B &&
          a->staging_storage_B == b->staging_storage_B &&
          a->rsrc_spec == b->rsrc_spec;
}

/* Whether the background object's depth and stencil (the attachments' clear
 * values) never reach `r`'s tiles: every depth or stencil buffer it has is
 * loaded instead. vkd3d-proton passes a clear value with every attachment,
 * loaded or not, so two command buffers continuing one pass may carry
 * different ones. */
static bool
background_object_unused(const struct hk_render_registers *r)
{
   agx_unpack(NULL, &r->zls_control, ZLS_CONTROL, zls);
   return (!r->depth.buffer || zls.z_load) &&
          (!r->stencil.buffer || zls.s_load);
}

/* Which of the render registers two streams program differently, or NULL.
 * `b` follows `a` in the tile pass, which begins with `a`'s background. */
static const char *
registers_differ(const struct hk_render_registers *a,
                 const struct hk_render_registers *b)
{
   if (a->width != b->width || a->height != b->height ||
       a->layers != b->layers || a->zls_width != b->zls_width ||
       a->zls_height != b->zls_height ||
       a->depth_dimensions != b->depth_dimensions)
      return "dimensions";
   if ((a->isp_bgobjdepth != b->isp_bgobjdepth ||
        a->isp_bgobjvals != b->isp_bgobjvals) &&
       !background_object_unused(b))
      return "background object";
   if (memcmp(&a->zls_control, &b->zls_control, sizeof(a->zls_control)) ||
       memcmp(&a->zls_control_partial, &b->zls_control_partial,
              sizeof(a->zls_control_partial)))
      return "depth/stencil control";
   if (a->process_empty_tiles != b->process_empty_tiles ||
       a->whole_attachments != b->whole_attachments)
      return "tile flags";

   /* A stream that drew nothing with depth bias leaves it unset. */
   if (a->dbias_is_int != U_TRISTATE_UNSET &&
       b->dbias_is_int != U_TRISTATE_UNSET &&
       a->dbias_is_int != b->dbias_is_int)
      return "depth bias format";

   if (a->depth.dimensions != b->depth.dimensions ||
       a->depth.buffer != b->depth.buffer || a->depth.meta != b->depth.meta ||
       a->depth.stride != b->depth.stride ||
       a->depth.meta_stride != b->depth.meta_stride ||
       a->stencil.buffer != b->stencil.buffer ||
       a->stencil.meta != b->stencil.meta ||
       a->stencil.stride != b->stencil.stride ||
       a->stencil.meta_stride != b->stencil.meta_stride)
      return "depth/stencil buffers";

   if (!bg_eot_resources_equal(&a->bg.main, &b->bg.main) ||
       !bg_eot_resources_equal(&a->eot.main, &b->eot.main) ||
       !bg_eot_resources_equal(&a->bg.partial, &b->bg.partial) ||
       !bg_eot_resources_equal(&a->eot.partial, &b->eot.partial))
      return "tile program resources";
   return NULL;
}

static bool
attachment_sets_equal(const struct hk_drm_attachment_set *a,
                      const struct hk_drm_attachment_set *b)
{
   return a->count == b->count && a->complete == b->complete &&
          !memcmp(a->list, b->list, a->count * sizeof(a->list[0]));
}

const char *
hk_render_pack_streams_match(const struct hk_cs *a, const struct hk_cs *b)
{
   if (memcmp(&a->tib, &b->tib, sizeof(a->tib)))
      return "tilebuffer";
   const char *why = registers_differ(&a->cr, &b->cr);
   if (why)
      return why;
   /* The tile programs: the merged command runs A's background programs, B's
    * end-of-tile program and A's partial-render programs. All are built from
    * the attachments, their operations and the tilebuffer layout, which
    * hk_render_pack_continues() and the checks above require the same (B
    * loads every attachment, so A's own load semantics stand), and from
    * nothing else but the untouched-attachment elision of the stream's last
    * instance, which rebuilds the end-of-tile programs without the stores
    * (and the background program without the loads) of attachments no draw
    * wrote. With equal elision both streams' programs store the same
    * attachments: one that only A writes, or only B, is elided in neither
    * (it would be in both or in neither had neither written it), so no write
    * is lost and no load skipped. A first draw that covers an attachment
    * changes only the background program, A's own. */
   if (a->render_pack_elided != b->render_pack_elided)
      return "elided attachments";
   if (a->area_x1 != b->area_x1 || a->area_y1 != b->area_y1)
      return "render area";
   /* The render command sets the sample positions once. */
   if (a->has_sample_locations != b->has_sample_locations ||
       a->ppp_multisamplectl != b->ppp_multisamplectl)
      return "sample locations";
   if (!attachment_sets_equal(&a->fragment_attachments,
                              &b->fragment_attachments))
      return "attachment set";
   if (memcmp(&a->tile_resources, &b->tile_resources,
              sizeof(a->tile_resources)))
      return "tile resources";
   return NULL;
}

/* Whether `cmd` pushed one of the attachments of `pass` as a descriptor. */
static bool
pushes_attachment(const struct hk_cmd_buffer *cmd,
                  const struct hk_render_pack_pass *pass)
{
   if (!cmd->m5_pushed_images)
      return false;
   set_foreach(cmd->m5_pushed_images, entry) {
      for (unsigned i = 0; i < HK_RENDER_PACK_SLOTS; ++i) {
         if (pass->att[i].image == entry->key)
            return true;
      }
   }
   return false;
}

void
hk_render_pack_run_start(struct hk_render_pack_run_state *run,
                         const struct hk_cmd_buffer *cmd,
                         const struct hk_cs *cs)
{
   *run = (struct hk_render_pack_run_state){
      .ordered = cs->m5_ordered,
      .writes_memory = cs->m5_shader_writes_memory,
      .unsafe = cs->render_pack_unsafe,
      .pushed_attachment =
         pushes_attachment(cmd, &cmd->render_pack.tail_pass),
   };
}

/* Whether the run is ordered once `b` follows `a`: a dependency recorded
 * between them, or one inside either stream (as hk_m5_pass_merge_begin()
 * counts one recorded before an earlier continuation of a stream; its first
 * scope reaches every earlier member, its second every later one), or one
 * earlier in the run. */
static bool
run_ordered(const struct hk_render_pack_run_state *run,
            const struct hk_cmd_buffer *a, const struct hk_cmd_buffer *b,
            const struct hk_cs *bs)
{
   const uint8_t dep = MAX2(a->render_pack.tail_dep, b->render_pack.head_dep);
   return run->ordered || dep == HK_M5_MERGE_DEP_ORDERED || bs->m5_ordered;
}

const char *
hk_render_pack_boundary(const struct hk_render_pack_run_state *run,
                        const struct hk_cmd_buffer *a,
                        const struct hk_cmd_buffer *b, const struct hk_cs *bs,
                        const struct hk_cs *first)
{
   const struct hk_render_pack_edges *ea = &a->render_pack;
   const struct hk_render_pack_edges *eb = &b->render_pack;
   if (!ea->tail)
      return "no tail";
   if (!eb->head)
      return "no head";
   if (!hk_render_pack_tail_writable(a))
      return "tail not writable";

   const uint8_t dep = MAX2(ea->tail_dep, eb->head_dep);
   if (dep >= HK_M5_MERGE_DEP_END)
      return "dependency";

   const bool ordered = run_ordered(run, a, b, bs);
   const char *why = hk_render_pack_continues(&ea->tail_pass, &eb->head_pass,
                                              ordered);
   if (why)
      return why;

   /* hk_m5_ordered_refusal() and hk_m5_ordered_draw_hazard(), over every
    * member's whole stream: none writes memory but its attachments, none of
    * their draws has a stage that runs ahead of the tile pass, and no
    * descriptor a member pushed names an attachment (descriptor sets cannot,
    * hk_render_pack_continues() requires attachment-only layouts). */
   if (ordered) {
      if (run->writes_memory || bs->m5_shader_writes_memory)
         return "ordered, writes memory";
      if (run->unsafe || bs->render_pack_unsafe)
         return "ordered, draw hazard";
      if (run->pushed_attachment || pushes_attachment(b, &eb->head_pass))
         return "ordered, attachment pushed as a descriptor";
   }

   return hk_render_pack_streams_match(first, bs);
}

void
hk_render_pack_run_join(struct hk_render_pack_run_state *run,
                        const struct hk_cmd_buffer *a,
                        const struct hk_cmd_buffer *b, const struct hk_cs *bs)
{
   run->ordered = run_ordered(run, a, b, bs);
   run->writes_memory |= bs->m5_shader_writes_memory;
   run->unsafe |= bs->render_pack_unsafe;
   run->pushed_attachment |= pushes_attachment(b, &b->render_pack.head_pass);
}

bool
hk_render_pack_array_prefix(const struct util_dynarray *a,
                            const struct util_dynarray *b,
                            const struct util_dynarray **longer)
{
   const struct util_dynarray *lo = a->size <= b->size ? a : b;
   const struct util_dynarray *hi = lo == a ? b : a;
   if (lo->size && memcmp(lo->data, hi->data, lo->size))
      return false;
   *longer = hi;
   return true;
}

const char *
hk_render_pack_member_refusal(const struct hk_cmd_buffer *cmd)
{
   if (cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
      return "secondary";
   if (!list_is_singular(&cmd->control_streams))
      return list_is_empty(&cmd->control_streams) ? "no stream" : "streams";

   /* Events can be observed before the submission completes; a pending or
    * unordered start stamp, a heap or a query reset carries work of its own
    * at submission. */
   if (cmd->state.uses_events)
      return "events";
   if (cmd->uses_heap)
      return "geometry heap";
   if (cmd->sync.pending_start_ts.handle || cmd->sync.has_unordered_start_ts)
      return "start stamp";
   if (cmd->sync.has_gpu_query_reset)
      return "query reset";

   const struct hk_cs *cs =
      list_first_entry(&cmd->control_streams, struct hk_cs, node);
   if (cs->type != HK_CS_VDM)
      return "compute";
   if (!cs->stats.cmds)
      return "no draws";
   if (!cs->g17p_pass_begins || !cs->g17p_pass_ends)
      return "pass edges";

   /* No query or timestamp rides on the render, nothing calls another
    * stream, and nothing patches the stream at submission. */
   if (cs->timestamp.start.handle || cs->timestamp.end.handle ||
       cs->unordered_start_ts || cs->g17p_timestamp_query)
      return "timestamp";
   if (cs->oq_pending_count || cs->oq_stamps.size)
      return "occlusion query";
   for (unsigned i = 0; i < ARRAY_SIZE(cs->oq_stamp); ++i) {
      if (cs->oq_stamp[i].handle)
         return "occlusion query";
   }
   if (cs->imm_writes.size || cs->query_access || cs->query_reset_pending)
      return "query";
   if (cs->calls_stream)
      return "stream call";
   if (cs->gen1_native_rq || cs->gen1_native_fragment_rq ||
       cs->gen1_native_vertex_rq)
      return "ray query";
   if (cs->alias_sites || cs->sparse_sites)
      return "submission patch";
   if (cs->as_build)
      return "acceleration structure";
   if (cs->sync.brl_transition)
      return "layout transition";

   /* The only stream has no in-buffer cut, and the tail requirements name
    * nothing but it. */
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      if (cs->sync.dep[sq].cut || cs->sync.dep_ta[sq].cut)
         return "in-buffer dependency";
      for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
         const struct hk_cs *cut = cmd->sync.req[k][sq].cut;
         if (cut && cut != cs)
            return "tail requirement";
      }
   }
   util_dynarray_foreach(&cmd->sync_pass_records, struct hk_sync_pass_record,
                         rec) {
      if ((rec->first && rec->first != cs) || (rec->last && rec->last != cs))
         return "pass record";
   }
   return NULL;
}

const struct hk_cs *
hk_render_pack_member(const struct hk_cmd_buffer *cmd)
{
   return hk_render_pack_member_refusal(cmd)
             ? NULL
             : list_first_entry(&cmd->control_streams, struct hk_cs, node);
}

bool
hk_render_pack_tail_writable(const struct hk_cmd_buffer *cmd)
{
   return !(cmd->usage_flags & VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT) &&
          cmd->render_pack.terminate != NULL;
}

bool
hk_render_pack_write_tail(enum agx_chip chip, struct hk_render_pack_edges *e,
                          uint64_t target)
{
   if (!e->terminate || e->linked == target)
      return false;

   if (target) {
      struct agx_vdm_stream_link_core link;
      if (!agx_vdm_stream_link_core_pack(chip, target, &link))
         return false;
      memcpy(e->terminate, link.bytes, link.length);
   } else {
      struct agx_vdm_stream_terminate_core terminate;
      if (!agx_vdm_stream_terminate_core_pack(chip, &terminate))
         return false;
      memcpy(e->terminate, terminate.bytes, terminate.length);
   }
   e->linked = target;
   return true;
}

void
hk_render_pack_init(struct hk_cmd_buffer *packet, struct hk_cs *cs,
                    struct hk_cmd_buffer *const *members, unsigned count)
{
   const struct hk_cs *first =
      list_first_entry(&members[0]->control_streams, struct hk_cs, node);
   uint8_t kinds = 0;
   bool vtx_images = false;
   for (unsigned m = 0; m < count; ++m) {
      const struct hk_cs *item =
         list_first_entry(&members[m]->control_streams, struct hk_cs, node);
      kinds |= item->sync.kind;
      vtx_images |= item->sync.vtx_images;
   }

   /* The packet keeps its own resource list across submissions; the rest of
    * the stream is the first member's, which the packet only reads. */
   struct util_dynarray resources = cs->render_resources;
   util_dynarray_clear(&resources);
   memcpy(cs, first, sizeof(*cs));
   cs->render_resources = resources;

   list_inithead(&cs->node);
   cs->cmd = packet;
   /* The stream memory, and the low VDM aliases HAL200 names it through
    * (hk_g17p_vdm_alias()), belong to the first member's recording. */
   cs->stream_owner = first->cmd;
   memset(&cs->sync, 0, sizeof(cs->sync));
   cs->sync.seq = 1;
   cs->sync.kind = kinds;
   /* Whether a carried requirement reaches the vertex half depends on every
    * member, a later one included: a member's tail requirement on images
    * applies to the vertex half of every member after it that reads images
    * there, as the resolver applies it to that command buffer's own render
    * (hk_render_pack_add()). */
   cs->sync.vtx_images = vtx_images;

   /* Summed or joined over the members by hk_render_pack_add(). */
   memset(&cs->stats, 0, sizeof(cs->stats));
   memset(&cs->scratch, 0, sizeof(cs->scratch));
   cs->g17p_primitive_bound = 0;
   cs->layer_zero_draws = 0;
   cs->layer_zero_unknown = false;
   cs->m5_shader_writes_memory = false;
   cs->m5_vertex_tps = false;
   cs->uses_sampler_heap = false;
   cs->g17p_mesh_output = false;
   cs->render_pack_unsafe = false;
   cs->render_pack_members = 0;
   memset(cs->anatomy_fs, 0, sizeof(cs->anatomy_fs));
   memset(cs->anatomy_vs, 0, sizeof(cs->anatomy_vs));
   cs->anatomy_fs_other = cs->anatomy_vs_other = 0;
   cs->anatomy_republish = cs->anatomy_vdm = cs->anatomy_vdm_new = 0;
   cs->anatomy_ppp = 0;
   memset(cs->anatomy_pass, 0, sizeof(cs->anatomy_pass));
   cs->anatomy_nomerge = cs->anatomy_zwrite = cs->anatomy_late_z = 0;

   /* One tile pass of several API instances, as an in-buffer continuation
    * makes it (hk_m5_pass_merge_begin()): no whole-pass render flags. */
   cs->m5_continued_pass = true;

   memset(&packet->sync, 0, sizeof(packet->sync));
   util_dynarray_clear(&packet->sync_pass_records);
   list_inithead(&packet->control_streams);
   list_addtail(&cs->node, &packet->control_streams);
}

static uint64_t
saturating_add(uint64_t a, uint64_t b)
{
   return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

/* Frame anatomy only: joins a member's per-program draw counts. */
static void
add_anatomy(struct hk_cs *cs, const struct hk_cs *item)
{
   for (unsigned i = 0; i < HK_FA_FS_MAX; ++i) {
      if (!item->anatomy_fs[i].fs)
         continue;
      unsigned j = 0;
      while (j < HK_FA_FS_MAX && cs->anatomy_fs[j].fs &&
             cs->anatomy_fs[j].fs != item->anatomy_fs[i].fs)
         j++;
      if (j == HK_FA_FS_MAX) {
         cs->anatomy_fs_other += item->anatomy_fs[i].draws;
         continue;
      }
      cs->anatomy_fs[j].fs = item->anatomy_fs[i].fs;
      cs->anatomy_fs[j].draws += item->anatomy_fs[i].draws;
   }
   for (unsigned i = 0; i < HK_FA_FS_MAX; ++i) {
      if (!item->anatomy_vs[i].vs)
         continue;
      unsigned j = 0;
      while (j < HK_FA_FS_MAX && cs->anatomy_vs[j].vs &&
             cs->anatomy_vs[j].vs != item->anatomy_vs[i].vs)
         j++;
      if (j == HK_FA_FS_MAX) {
         cs->anatomy_vs_other += item->anatomy_vs[i].draws;
         continue;
      }
      cs->anatomy_vs[j].vs = item->anatomy_vs[i].vs;
      cs->anatomy_vs[j].draws += item->anatomy_vs[i].draws;
   }
   cs->anatomy_fs_other += item->anatomy_fs_other;
   cs->anatomy_vs_other += item->anatomy_vs_other;
   cs->anatomy_republish += item->anatomy_republish;
   cs->anatomy_vdm += item->anatomy_vdm;
   cs->anatomy_vdm_new += item->anatomy_vdm_new;
   cs->anatomy_ppp += item->anatomy_ppp;
   for (unsigned i = 0; i < ARRAY_SIZE(cs->anatomy_pass); ++i)
      cs->anatomy_pass[i] += item->anatomy_pass[i];
   cs->anatomy_nomerge += item->anatomy_nomerge;
   cs->anatomy_zwrite += item->anatomy_zwrite;
   cs->anatomy_late_z += item->anatomy_late_z;
}

bool
hk_render_pack_add(struct hk_cmd_buffer *packet, struct hk_cs *cs,
                   const struct hk_cmd_buffer *source, const struct hk_cs *item,
                   bool first, bool last, uint8_t *preceding_kinds)
{
   cs->render_pack_members++;
   cs->stats.calls += item->stats.calls;
   cs->stats.cmds += item->stats.cmds;
   cs->stats.flushes += item->stats.flushes;
   cs->stats.merged += item->stats.merged + !first;
   cs->stats.robust_index_draws += item->stats.robust_index_draws;
   cs->stats.robust_index_dispatches += item->stats.robust_index_dispatches;
   cs->stats.robust_index_batches += item->stats.robust_index_batches;
   cs->g17p_primitive_bound =
      saturating_add(cs->g17p_primitive_bound, item->g17p_primitive_bound);
   cs->layer_zero_draws += item->layer_zero_draws;
   cs->layer_zero_unknown |= item->layer_zero_unknown;

   cs->m5_shader_writes_memory |= item->m5_shader_writes_memory;
   cs->m5_vertex_tps |= item->m5_vertex_tps;
   cs->uses_sampler_heap |= item->uses_sampler_heap;
   cs->g17p_mesh_output |= item->g17p_mesh_output;
   cs->render_pack_unsafe |= item->render_pack_unsafe;
   cs->scratch.vs.main |= item->scratch.vs.main;
   cs->scratch.vs.preamble |= item->scratch.vs.preamble;
   cs->scratch.fs.main |= item->scratch.fs.main;
   cs->scratch.fs.preamble |= item->scratch.fs.preamble;
   if (cs->cr.dbias_is_int == U_TRISTATE_UNSET)
      cs->cr.dbias_is_int = item->cr.dbias_is_int;
   if (last) {
      /* The last instance stores the attachments, as the final instance of
       * an in-buffer continuation does. */
      cs->cr.eot.main = item->cr.eot.main;
      cs->g17p_pass_ends = item->g17p_pass_ends;
   }
   add_anatomy(cs, item);

   /* The kernel folds the list into one demand with a per-field maximum, so
    * the union of the members' lists is the merged render's. */
   if (item->render_resources.size) {
      void *dst = util_dynarray_grow_bytes(&cs->render_resources, 1,
                                           item->render_resources.size);
      if (!dst)
         return false;
      memcpy(dst, item->render_resources.data, item->render_resources.size);
   }

   /* A packet waits for everything any member waits for. The members after
    * one wait for its tail, as separate command buffers would: the resolver
    * applies a carried requirement to each destination class of a stream,
    * the whole render's (hk_sync_full_kinds()) and its vertex half's (its
    * kinds and, if any member draws with image-reading pre-rasterization
    * shaders, the image class: hk_render_pack_init()). A tail that names a
    * member makes later command buffers wait for the packet; so does a
    * prior-only render tail of a kind an earlier member already ran. */
   const uint8_t carry = hk_sync_full_kinds(cs->sync.kind);
   const uint8_t ta_carry =
      cs->sync.kind | (cs->sync.vtx_images ? HK_SYNC_GFX_IMG : 0);
   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      cs->sync.dep[sq].prior |= item->sync.dep[sq].prior;
      cs->sync.dep_ta[sq].prior |= item->sync.dep_ta[sq].prior;
      for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
         const struct hk_sync_dep *req = &source->sync.req[k][sq];
         packet->sync.req[k][sq].prior |= req->prior;
         if (req->cut ||
             (sq == HK_SYNC_VDM && (req->prior & *preceding_kinds)))
            packet->sync.req[k][sq].cut = cs;
         if (!last && (carry & BITFIELD_BIT(k)))
            cs->sync.dep[sq].prior |= req->prior;
         if (!last && (ta_carry & BITFIELD_BIT(k)))
            cs->sync.dep_ta[sq].prior |= req->prior;
      }
   }
   *preceding_kinds |= item->sync.kind;

   /* Tile footprints: a later command buffer that shares a tile with any
    * member waits for the packet. Hazards between the members are inside
    * the one tile pass. */
   util_dynarray_foreach(&source->sync_pass_records,
                         struct hk_sync_pass_record, rec) {
      const struct hk_sync_pass_record copy = {
         .fp = rec->fp,
         .first = rec->first ? cs : NULL,
         .last = rec->last ? cs : NULL,
      };
      struct hk_sync_pass_record *dst =
         util_dynarray_grow(&packet->sync_pass_records,
                            struct hk_sync_pass_record, 1);
      if (!dst)
         return false;
      *dst = copy;
   }
   return true;
}

void
hk_render_pack_finish(struct hk_cmd_buffer *packet, struct hk_cs *cs)
{
   u_foreach_bit(k, cs->sync.kind)
      packet->sync.last[k][HK_SYNC_VDM] = cs;
   packet->sync.last_any[HK_SYNC_VDM] = cs;
}

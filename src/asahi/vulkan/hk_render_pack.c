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
                         const struct hk_render_pack_pass *b, bool ordered,
                         bool non_local_src, bool att_visible)
{
   const bool a_susp = (a->flags & VK_RENDERING_SUSPENDING_BIT) != 0;
   const bool b_res = (b->flags & VK_RENDERING_RESUMING_BIT) != 0;
   const bool b_susp = (b->flags & VK_RENDERING_SUSPENDING_BIT) != 0;
   if (a_susp != b_res)
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
      /* B continues what A left in the tiles: it must load it (a resuming
       * instance always continues A's tiles regardless of loadOp), and either
       * keep A's store op or discard a stored colour attachment (STORE ->
       * DONT_CARE) when B is not suspending. A must not have resolved; B may
       * terminate the merged pass with a colour resolve across a
       * framebuffer-local dependency. */
      if (!b_res && y->load_op != VK_ATTACHMENT_LOAD_OP_LOAD)
         return "load op";
      const bool is_color = i < a->color_count;
      const bool store_drop =
         is_color && !b_susp &&
         x->store_op == VK_ATTACHMENT_STORE_OP_STORE &&
         y->store_op == VK_ATTACHMENT_STORE_OP_DONT_CARE;
      if (y->store_op != x->store_op && !store_drop)
         return "store op";
      if (x->resolve_mode != VK_RESOLVE_MODE_NONE)
         return "resolve";
      const bool y_resolves = y->resolve_mode != VK_RESOLVE_MODE_NONE;
      if (y_resolves && (!is_color || ordered || non_local_src || b_susp))
         return "resolve";
      const uint32_t y_flags =
         y_resolves
            ? (y->flags &
               ~VK_RENDERING_ATTACHMENT_RESOLVE_SKIP_TRANSFER_FUNCTION_BIT_KHR)
            : y->flags;
      if (x->flags != y_flags)
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
      if (ordered && att_visible && (x->aliasable || y->aliasable))
         return "ordered with an aliasable image";
   }
   if (!attachments)
      return "no attachments";

   /* An ordered dependency that makes A's attachment writes visible to B's
    * shaders requires attachment-only layouts so no descriptor can read A's
    * unstored attachments. Within one command buffer the continuation may
    * instead show that no descriptor set is bound, which another command
    * buffer's recording cannot. */
   if (ordered && att_visible && !b->layout_proof)
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

/* Whether `b`'s ZLS control is compatible with continuing `a`'s: `a` may clear
 * depth/stencil (z_load/s_load == 0) while `b` loads what `a` left in the
 * tiles, and either stream may have dropped a dead z_store/s_store that
 * hk_render_pack_add() unions back. Every other ZLS bit (formats, tiling,
 * compression, resolve) must match. */
static bool
zls_control_compatible(const struct hk_cs *a, const struct hk_cs *b)
{
   if (!memcmp(&a->cr.zls_control, &b->cr.zls_control,
               sizeof(a->cr.zls_control)))
      return true;

   agx_unpack(NULL, &a->cr.zls_control, ZLS_CONTROL, za);
   agx_unpack(NULL, &b->cr.zls_control, ZLS_CONTROL, zb);
   if ((za.z_load && !zb.z_load) || (za.s_load && !zb.s_load))
      return false;

   if (!za.z_load && zb.z_load) {
      za.z_load = true;
      za.z_load_tiling = zb.z_load_tiling;
      za.z_load_compress = zb.z_load_compress;
   }
   if (!za.s_load && zb.s_load) {
      za.s_load = true;
      za.s_load_tiling = zb.s_load_tiling;
      za.s_load_compress = zb.s_load_compress;
   }

   const uint8_t elided =
      a->render_pack_zls_elided_store | b->render_pack_zls_elided_store;
   if (elided & VK_IMAGE_ASPECT_DEPTH_BIT)
      za.z_store = zb.z_store = true;
   if (elided & VK_IMAGE_ASPECT_STENCIL_BIT)
      za.s_store = zb.s_store = true;

   struct agx_zls_control_packed pa, pb;
   agx_pack(&pa, ZLS_CONTROL, cfg) {
      cfg = za;
   }
   agx_pack(&pb, ZLS_CONTROL, cfg) {
      cfg = zb;
   }
   return !memcmp(&pa, &pb, sizeof(pa));
}

static const struct hk_bg_eot *
full_eot_main(const struct hk_cs *cs)
{
   return cs->render_pack_has_full_programs ? &cs->render_pack_full_eot_main
                                            : &cs->cr.eot.main;
}

static const struct hk_bg_eot *
full_eot_partial(const struct hk_cs *cs)
{
   return cs->render_pack_has_full_programs ? &cs->render_pack_full_eot_partial
                                            : &cs->cr.eot.partial;
}
/* Which of the render registers two streams program differently, or NULL.
 * `b` follows `a` in the tile pass, which begins with `a`'s background. When
 * `eot_changes` is set, `b` supplies a different final end-of-tile program
 * (a colour STORE -> DONT_CARE discard or a final direct resolve), so its
 * `eot.main` resources and `process_empty_tiles` may differ from `a`'s. */
static const char *
registers_differ(const struct hk_cs *cs_a, const struct hk_cs *cs_b,
                 bool eot_changes)
{
   const struct hk_render_registers *a = &cs_a->cr;
   const struct hk_render_registers *b = &cs_b->cr;

   if (a->width != b->width || a->height != b->height ||
       a->layers != b->layers || a->zls_width != b->zls_width ||
       a->zls_height != b->zls_height ||
       a->depth_dimensions != b->depth_dimensions)
      return "dimensions";
   if ((a->isp_bgobjdepth != b->isp_bgobjdepth ||
        a->isp_bgobjvals != b->isp_bgobjvals) &&
       !background_object_unused(b))
      return "background object";
   if (!zls_control_compatible(cs_a, cs_b) ||
       memcmp(&a->zls_control_partial, &b->zls_control_partial,
              sizeof(a->zls_control_partial)))
      return "depth/stencil control";
   if ((!eot_changes && b->process_empty_tiles && !a->process_empty_tiles) ||
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

   /* The merged packet runs `a`'s main background program (which may clear
    * while `b` loads), `a`'s partial programs, and `b`'s main end-of-tile
    * (which may differ when `eot_changes` is set). */
   if ((!eot_changes &&
        !bg_eot_resources_equal(full_eot_main(cs_a), full_eot_main(cs_b))) ||
       !bg_eot_resources_equal(&a->bg.partial, &b->bg.partial) ||
       !bg_eot_resources_equal(full_eot_partial(cs_a), full_eot_partial(cs_b)))
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

static bool
attachment_set_merge(struct hk_drm_attachment_set *dst,
                     const struct hk_drm_attachment_set *src, bool apply)
{
   struct hk_drm_attachment_set merged;
   hk_drm_attachment_set_init(&merged);
   merged.complete = dst->complete && src->complete;
   for (unsigned i = 0; i < dst->count; ++i) {
      if (!hk_drm_attachment_set_add(&merged, dst->list[i].pointer,
                                     dst->list[i].size))
         return false;
   }
   for (unsigned i = 0; i < src->count; ++i) {
      if (!hk_drm_attachment_set_add(&merged, src->list[i].pointer,
                                     src->list[i].size))
         return false;
   }
   if (apply)
      *dst = merged;
   return true;
}

static const char *
render_pack_streams_match(const struct hk_cs *a, const struct hk_cs *b,
                          bool eot_changes, bool resolving)
{
   if (memcmp(&a->tib, &b->tib, sizeof(a->tib)))
      return "tilebuffer";
   if (resolving && (!a->cr.whole_attachments || !b->cr.whole_attachments))
      return "partial render area resolve";
   const char *why = registers_differ(a, b, eot_changes);
   if (why)
      return why;
   /* The tile programs: when both streams elided the same untouched colour
    * attachments, the merged pass keeps the elided programs. When their
    * elision masks differ and each elided stream preserved its full
    * (unelided) tile programs before hk_g17p_elide_untouched_color() rewrote
    * them, hk_render_pack_init()/hk_render_pack_add() restore the full
    * programs so no attachment written by either member loses its load or
    * store. */
   if (a->render_pack_elided != b->render_pack_elided &&
       ((a->render_pack_elided && !a->render_pack_has_full_programs) ||
        (b->render_pack_elided && !b->render_pack_has_full_programs)))
      return "elided attachments";
   if (a->area_x1 != b->area_x1 || a->area_y1 != b->area_y1)
      return "render area";
   /* The render command sets the sample positions once. */
   if (a->has_sample_locations != b->has_sample_locations ||
       a->ppp_multisamplectl != b->ppp_multisamplectl)
      return "sample locations";
   if (resolving) {
      struct hk_drm_attachment_set merged = a->fragment_attachments;
      if (!attachment_set_merge(&merged, &b->fragment_attachments, false))
         return "attachment set";
   } else if (!attachment_sets_equal(&a->fragment_attachments,
                                     &b->fragment_attachments)) {
      return "attachment set";
   }
   if (eot_changes) {
      if (a->tile_resources.stage != b->tile_resources.stage ||
          a->tile_resources.imageblock_bytes !=
             b->tile_resources.imageblock_bytes ||
          a->tile_resources.multisample_bytes !=
             b->tile_resources.multisample_bytes)
         return "tile resources";
   } else if (memcmp(&a->tile_resources, &b->tile_resources,
                     sizeof(a->tile_resources))) {
      return "tile resources";
   }
   return NULL;
}

const char *
hk_render_pack_streams_match(const struct hk_cs *a, const struct hk_cs *b)
{
   return render_pack_streams_match(a, b, false, false);
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
      .non_local_src = cs->m5_non_local_src,
      .att_available = cs->m5_ordered || cs->m5_att_available,
      .att_visible = cs->m5_ordered,
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

/* Whether any barrier between or inside the run's members had a graphics first
 * scope outside the attachment stages (hk_m5_pass_merge_non_local_src()). */
static bool
run_non_local_src(const struct hk_render_pack_run_state *run,
                  const struct hk_cmd_buffer *a, const struct hk_cmd_buffer *b,
                  const struct hk_cs *bs)
{
   return run->non_local_src || a->render_pack.tail_non_local_src ||
          b->render_pack.head_non_local_src || bs->m5_non_local_src;
}

/* Whether any barrier between or inside the run's members made attachment
 * writes available (regardless of whether that barrier was local or ordered).
 * A later ordered barrier with srcAccessMask == 0 chains visibility onto it. */
static bool
run_att_available(const struct hk_render_pack_run_state *run,
                  const struct hk_cmd_buffer *a, const struct hk_cmd_buffer *b,
                  const struct hk_cs *bs)
{
   return run->att_available || a->render_pack.tail_att_available ||
          b->render_pack.head_att_available || bs->m5_ordered ||
          bs->m5_att_available;
}

static bool
run_att_visible(const struct hk_render_pack_run_state *run,
                const struct hk_cmd_buffer *a, const struct hk_cmd_buffer *b,
                const struct hk_cs *bs)
{
   const bool tail_vis =
      a->render_pack.tail_dep == HK_M5_MERGE_DEP_ORDERED &&
      !a->render_pack.tail_att_not_visible;
   const bool head_vis =
      b->render_pack.head_dep == HK_M5_MERGE_DEP_ORDERED &&
      !b->render_pack.head_att_not_visible;
   const bool chained =
      run_ordered(run, a, b, bs) && run_att_available(run, a, b, bs);
   return run->att_visible || tail_vis || head_vis || chained || bs->m5_ordered;
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
   const bool non_local_src = run_non_local_src(run, a, b, bs);
   const bool att_visible = run_att_visible(run, a, b, bs);
   const char *why = hk_render_pack_continues(&ea->tail_pass, &eb->head_pass,
                                              ordered, non_local_src,
                                              att_visible);
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
      if (att_visible &&
          (run->pushed_attachment || pushes_attachment(b, &eb->head_pass)))
         return "ordered, attachment pushed as a descriptor";
   }

   bool eot_changes = false, resolving = false;
   for (unsigned i = 0; i < eb->head_pass.color_count; ++i) {
      const struct hk_render_pack_att *y = &eb->head_pass.att[i];
      if (!y->iview)
         continue;
      if (y->resolve_mode != VK_RESOLVE_MODE_NONE)
         resolving = eot_changes = true;
      if (y->store_op == VK_ATTACHMENT_STORE_OP_DONT_CARE)
         eot_changes = true;
   }

   return render_pack_streams_match(first, bs, eot_changes, resolving);
}

bool
hk_render_pack_post_is_heap_reset(const struct hk_cs *post)
{
   return post && post->type == HK_CS_CDM && post->stats.calls == 0 &&
          post->sync.heap_slot != 0 &&
          post->sync.kind == (HK_SYNC_GFX | HK_SYNC_GFX_MEM) &&
          !post->query_access && !post->query_access_ranges.size &&
          !post->imm_write_query_ranges.size && !post->query_reset_pending &&
          !post->oq_pending_count && !post->oq_stamps.size;
}

const char *
hk_render_pack_pre_cs_refusal(
   const struct hk_render_pack_run_state *state, const struct hk_cs *bs_pre,
   unsigned run_post, uint8_t post_heap_slots, bool post_heap_only,
   uint8_t run_vdm_kinds, const uint8_t run_vdm_tail_req[HK_SYNC_NUM_KINDS],
   const uint8_t run_post_tail_req[HK_SYNC_NUM_KINDS])
{
   if (!bs_pre)
      return NULL;

   /* Every hoisted pre_cs runs before the packed render stream while every
    * earlier member's post_cs runs after it. Hoisting past an earlier post_cs
    * is only safe for pure driver pre-graphics when every earlier post_cs in
    * the run is a geometry heap-slot reset on a strictly disjoint heap slot
    * and no recorded dependency orders any earlier post_cs before bs_pre. */
   if (run_post > 0) {
      if (bs_pre->stats.calls != 0)
         return "api compute after post-graphics";
      const uint8_t post_kinds = HK_SYNC_GFX | HK_SYNC_GFX_MEM;
      if (!post_heap_only || bs_pre->sync.kind != post_kinds)
         return "pre-graphics after post-graphics";
      if (bs_pre->sync.heap_slot &&
          (post_heap_slots & BITFIELD_BIT(bs_pre->sync.heap_slot - 1)))
         return "pre-graphics after post-graphics";
      if ((bs_pre->sync.dep[HK_SYNC_CDM].prior & post_kinds) ||
          (bs_pre->sync.dep_ta[HK_SYNC_CDM].prior & post_kinds))
         return "pre-graphics after post-graphics";
      const uint8_t pre_full = hk_sync_full_kinds(bs_pre->sync.kind);
      u_foreach_bit(k, pre_full) {
         if (run_post_tail_req[k] & post_kinds)
            return "pre-graphics after post-graphics";
      }
   }

   if (bs_pre->stats.calls == 0)
      return NULL;

   /* Hoisting an API compute stream ahead of earlier render members in this
    * packed run requires that earlier members only wrote attachments (no
    * buffer/SSBO writes) and that no barrier ordered those earlier renders
    * before this compute stream. */
   if (state->writes_memory)
      return "api compute after memory write";
   if ((bs_pre->sync.dep[HK_SYNC_VDM].prior & run_vdm_kinds) ||
       (bs_pre->sync.dep_ta[HK_SYNC_VDM].prior & run_vdm_kinds))
      return "api compute depends on render";

   const uint8_t pre_full = hk_sync_full_kinds(bs_pre->sync.kind);
   u_foreach_bit(k, pre_full) {
      if (run_vdm_tail_req[k] & run_vdm_kinds)
         return "api compute ordered after render";
   }
   return NULL;
}

void
hk_render_pack_run_join(struct hk_render_pack_run_state *run,
                        const struct hk_cmd_buffer *a,
                        const struct hk_cmd_buffer *b, const struct hk_cs *bs)
{
   run->ordered = run_ordered(run, a, b, bs);
   run->non_local_src = run_non_local_src(run, a, b, bs);
   run->att_available = run_att_available(run, a, b, bs);
   run->att_visible = run_att_visible(run, a, b, bs);
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

const struct hk_cs *
hk_render_pack_find_vdm(const struct hk_cmd_buffer *cmd,
                        const struct hk_cs **pre_out,
                        const struct hk_cs **post_out)
{
   if (pre_out)
      *pre_out = NULL;
   if (post_out)
      *post_out = NULL;

   if (list_is_empty(&cmd->control_streams))
      return NULL;

   if (list_is_singular(&cmd->control_streams)) {
      const struct hk_cs *cs =
         list_first_entry(&cmd->control_streams, struct hk_cs, node);
      return cs->type == HK_CS_VDM ? cs : NULL;
   }

   const struct hk_cs *pre = NULL, *vdm = NULL, *post = NULL;
   for (const struct list_head *pos = cmd->control_streams.next;
        pos != &cmd->control_streams; pos = pos->next) {
      const struct hk_cs *s = list_entry(pos, const struct hk_cs, node);
      if (s->type == HK_CS_VDM) {
         if (vdm)
            return NULL;
         vdm = s;
      } else if (s->type == HK_CS_CDM) {
         if (!vdm) {
            if (pre)
               return NULL;
            pre = s;
         } else {
            if (post)
               return NULL;
            post = s;
         }
      } else {
         return NULL;
      }
   }

   if (!vdm)
      return NULL;
   if (pre_out)
      *pre_out = pre;
   if (post_out)
      *post_out = post;
   return vdm;
}

static const char *
hk_render_pack_aux_cdm_refusal(const struct hk_cs *cs, bool is_post,
                               bool is_first)
{
   if (cs->type != HK_CS_CDM)
      return "aux not compute";
   if (is_post) {
      if (cs->stats.calls != 0)
         return "api compute";
      if (!cs->stats.cmds)
         return "empty post-graphics";
      if (cs->sync.kind != (HK_SYNC_GFX | HK_SYNC_GFX_MEM))
         return "aux sync kind";
   } else if (cs->stats.calls != 0) {
      if (!is_first && cs->render_pack_compute_internal)
         return "api compute internal";
      const uint8_t allowed =
         HK_SYNC_COMPUTE | HK_SYNC_GFX | HK_SYNC_GFX_MEM |
         (is_first ? HK_SYNC_XFER : 0);
      if (!cs->sync.kind || (cs->sync.kind & ~allowed))
         return "aux sync kind";
   } else if (is_first && cs->stats.cmds) {
      const uint8_t allowed =
         HK_SYNC_XFER | HK_SYNC_COMPUTE | HK_SYNC_GFX | HK_SYNC_GFX_MEM;
      if (!cs->sync.kind || (cs->sync.kind & ~allowed))
         return "aux sync kind";
   } else if (cs->sync.kind != (HK_SYNC_GFX | HK_SYNC_GFX_MEM)) {
      return "aux sync kind";
   }
   if (cs->timestamp.start.handle || cs->timestamp.end.handle ||
       cs->unordered_start_ts || cs->g17p_timestamp_query)
      return "aux timestamp";
   if (cs->oq_pending_count || cs->oq_stamps.size)
      return "aux occlusion query";
   for (unsigned i = 0; i < ARRAY_SIZE(cs->oq_stamp); ++i) {
      if (cs->oq_stamp[i].handle)
         return "aux occlusion query";
   }
   if ((!is_post && cs->imm_writes.size) || cs->imm_write_query_ranges.size ||
       cs->query_access || cs->query_access_ranges.size ||
       cs->query_reset_pending)
      return "aux query";
   if (cs->calls_stream || cs->g18_dispatch_sealed)
      return "aux stream call";
   if (cs->gen1_native_rq || cs->gen1_native_fragment_rq ||
       cs->gen1_native_vertex_rq)
      return "aux ray query";
   if (cs->alias_sites || cs->sparse_sites)
      return "aux submission patch";
   if (cs->as_build)
      return "aux acceleration structure";
   if (cs->sync.brl_transition)
      return "aux layout transition";
   return NULL;
}

static const char *
hk_render_pack_member_refusal_streams(const struct hk_cmd_buffer *cmd,
                                      bool first, const struct hk_cs **vdm_out,
                                      const struct hk_cs **pre_out,
                                      const struct hk_cs **post_out)
{
   if (vdm_out)
      *vdm_out = NULL;
   if (pre_out)
      *pre_out = NULL;
   if (post_out)
      *post_out = NULL;

   if (cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
      return "secondary";

   const struct hk_cs *pre = NULL, *cs = NULL, *post = NULL;
   if (list_is_empty(&cmd->control_streams))
      return "no stream";
   if (list_is_singular(&cmd->control_streams)) {
      cs = list_first_entry(&cmd->control_streams, struct hk_cs, node);
      if (cs->type != HK_CS_VDM)
         return "compute";
   } else {
      cs = hk_render_pack_find_vdm(cmd, &pre, &post);
      if (!cs)
         return "streams";
   }

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

   if (!cs->stats.cmds)
      return "no draws";
   const bool pass_begins_ok =
      cs->g17p_pass_begins ||
      (cmd->render_pack.head &&
       (cmd->render_pack.head_pass.flags & VK_RENDERING_RESUMING_BIT));
   const bool pass_ends_ok =
      cs->g17p_pass_ends ||
      (cmd->render_pack.tail &&
       (cmd->render_pack.tail_pass.flags & VK_RENDERING_SUSPENDING_BIT));
   if (!pass_begins_ok || !pass_ends_ok)
      return "pass edges";

   /* No query or start timestamp rides on the render, nothing calls another
    * stream, and nothing patches the stream at submission. A vertex-end or
    * fragment-end timestamp may ride on it: the packet carries it on the
    * same stamp of the merged render, which latches no earlier
    * (hk_render_pack_add()). */
   if (cs->timestamp.start.handle || cs->unordered_start_ts ||
       cs->g17p_timestamp_query)
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

   if (pre) {
      const char *why = hk_render_pack_aux_cdm_refusal(pre, false, first);
      if (why)
         return why;
      if (pre->sync.heap_slot && pre->sync.heap_slot != cs->sync.heap_slot)
         return "pre heap slot";
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
         if (pre->sync.dep[sq].cut || pre->sync.dep_ta[sq].cut)
            return "pre in-buffer dependency";
      }
   }
   if (post) {
      const char *why = hk_render_pack_aux_cdm_refusal(post, true, false);
      if (why)
         return why;
      if (!post->sync.heap_slot || post->sync.heap_slot != cs->sync.heap_slot)
         return "post heap slot";
      if ((post->sync.dep[HK_SYNC_VDM].cut &&
           post->sync.dep[HK_SYNC_VDM].cut != cs) ||
          (post->sync.dep_ta[HK_SYNC_VDM].cut &&
           post->sync.dep_ta[HK_SYNC_VDM].cut != cs) ||
          post->sync.dep[HK_SYNC_CDM].cut ||
          post->sync.dep_ta[HK_SYNC_CDM].cut)
         return "post in-buffer dependency";
   }

   /* The render stream has no in-buffer cut except its own pre-graphics CDM,
    * and tail requirements name nothing outside {pre, cs, post}. */
   if (cs->sync.dep[HK_SYNC_VDM].cut || cs->sync.dep_ta[HK_SYNC_VDM].cut)
      return "in-buffer dependency";
   if ((cs->sync.dep[HK_SYNC_CDM].cut &&
        cs->sync.dep[HK_SYNC_CDM].cut != pre) ||
       (cs->sync.dep_ta[HK_SYNC_CDM].cut &&
        cs->sync.dep_ta[HK_SYNC_CDM].cut != pre))
      return "in-buffer dependency";
   for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
      const struct hk_cs *vcut = cmd->sync.req[k][HK_SYNC_VDM].cut;
      if (vcut && vcut != cs)
         return "tail requirement";
      const struct hk_cs *ccut = cmd->sync.req[k][HK_SYNC_CDM].cut;
      if (ccut && ccut != pre && ccut != post)
         return "tail requirement";
   }
   util_dynarray_foreach(&cmd->sync_pass_records, struct hk_sync_pass_record,
                         rec) {
      if ((rec->first && rec->first != cs) || (rec->last && rec->last != cs))
         return "pass record";
   }

   if (vdm_out)
      *vdm_out = cs;
   if (pre_out)
      *pre_out = pre;
   if (post_out)
      *post_out = post;
   return NULL;
}

const char *
hk_render_pack_member_refusal_for(const struct hk_cmd_buffer *cmd, bool first)
{
   return hk_render_pack_member_refusal_streams(cmd, first, NULL, NULL, NULL);
}

const char *
hk_render_pack_member_refusal(const struct hk_cmd_buffer *cmd)
{
   return hk_render_pack_member_refusal_for(cmd, false);
}

const struct hk_cs *
hk_render_pack_member_streams(const struct hk_cmd_buffer *cmd, bool first,
                              const struct hk_cs **pre_out,
                              const struct hk_cs **post_out)
{
   const struct hk_cs *vdm = NULL;
   if (hk_render_pack_member_refusal_streams(cmd, first, &vdm, pre_out,
                                             post_out))
      return NULL;
   return vdm;
}

/* The packet stamps the members' timestamps take: bit 0 the vertex end,
 * bit 1 the fragment end. */
unsigned
hk_render_pack_stamp_slots(const struct hk_cs *cs)
{
   return (cs->timestamp.vtx_end.handle ? 1 : 0) |
          (cs->timestamp.end.handle ? 2 : 0);
}
const struct hk_cs *
hk_render_pack_member(const struct hk_cmd_buffer *cmd)
{
   return hk_render_pack_member_streams(cmd, false, NULL, NULL);
}

bool
hk_render_pack_is_empty_fixup(const struct hk_cmd_buffer *cmd,
                              bool *is_resume, bool *is_suspend)
{
   if (is_resume)
      *is_resume = false;
   if (is_suspend)
      *is_suspend = false;

   if (cmd->vk.level != VK_COMMAND_BUFFER_LEVEL_PRIMARY ||
       !list_is_empty(&cmd->control_streams) || cmd->state.uses_events ||
       cmd->uses_heap || cmd->sync.pending_start_ts.handle ||
       cmd->sync.has_unordered_start_ts || cmd->sync.has_gpu_query_reset ||
       cmd->sync.stamp_copy_count || cmd->state.deferred_clear_count ||
       cmd->sync_pass_records.size != 0)
      return false;

   const bool res = cmd->render_pack.empty_resume;
   const bool susp = cmd->render_pack.empty_suspend;
   if (res == susp)
      return false;

   for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq) {
      for (unsigned k = 0; k < HK_SYNC_NUM_KINDS; ++k) {
         if (cmd->sync.req[k][sq].cut || cmd->sync.req[k][sq].prior)
            return false;
      }
   }

   if (is_resume)
      *is_resume = res;
   if (is_suspend)
      *is_suspend = susp;
   return true;
}

bool
hk_render_pack_fixup_matches(const struct hk_render_pack_pass *fixup,
                             const struct hk_render_pack_pass *other)
{
   if (memcmp(&fixup->area, &other->area, sizeof(fixup->area)) ||
       fixup->layer_count != other->layer_count ||
       fixup->view_mask != other->view_mask ||
       fixup->color_count != other->color_count || fixup->view_mask)
      return false;

   unsigned attachments = 0;
   for (unsigned i = 0; i < HK_RENDER_PACK_SLOTS; ++i) {
      if (i >= fixup->color_count && i < HK_RENDER_PACK_COLORS)
         continue;
      const struct hk_render_pack_att *x = &fixup->att[i];
      const struct hk_render_pack_att *y = &other->att[i];
      if ((x->iview != NULL) != (y->iview != NULL))
         return false;
      if (!x->iview)
         continue;
      attachments++;
      if (x->iview != y->iview || x->samples != y->samples ||
          x->store_op != y->store_op || x->resolve_mode != VK_RESOLVE_MODE_NONE ||
          y->resolve_mode != VK_RESOLVE_MODE_NONE || x->flags != y->flags ||
          x->layout != y->layout)
         return false;
   }
   return attachments > 0;
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
   const struct hk_cs *first = hk_render_pack_find_vdm(members[0], NULL, NULL);
   if (!first)
      first = list_first_entry(&members[0]->control_streams, struct hk_cs, node);
   uint8_t kinds = 0;
   bool vtx_images = false;
   bool elided_diff = false;
   bool vtx_img_unsafe = false;
   uint64_t att_bloom = 0;
   uint64_t vtx_img_bloom = 0;
   for (unsigned m = 0; m < count; ++m) {
      if (list_is_empty(&members[m]->control_streams))
         continue;
      const struct hk_cs *item = hk_render_pack_find_vdm(members[m], NULL, NULL);
      if (!item)
         item = list_first_entry(&members[m]->control_streams, struct hk_cs, node);
      kinds |= item->sync.kind;
      vtx_images |= item->sync.vtx_images;
      if (item->render_pack_elided != first->render_pack_elided)
         elided_diff = true;
      vtx_img_unsafe |= item->sync.vtx_img_unsafe;
      att_bloom |= item->sync.att_bloom;
      vtx_img_bloom |= item->sync.vtx_img_bloom;
   }

   /* The packet keeps its own resource list across submissions; the rest of
    * the stream is the first member's, which the packet only reads. */
   struct util_dynarray resources = cs->render_resources;
   util_dynarray_clear(&resources);
   memcpy(cs, first, sizeof(*cs));
   cs->render_resources = resources;
   if (elided_diff) {
      if (first->render_pack_elided) {
         cs->cr.bg.main = first->render_pack_full_bg_main;
         cs->cr.eot.partial = first->render_pack_full_eot_partial;
      }
      cs->render_pack_elided = 0;
   }

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
   cs->sync.vtx_img_unsafe = vtx_img_unsafe;
   cs->sync.att_bloom = att_bloom;
   cs->sync.vtx_img_bloom = vtx_img_bloom;

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
   cs->cr.process_empty_tiles |= item->cr.process_empty_tiles;
   cs->tile_resources.primary_bytes =
      MAX2(cs->tile_resources.primary_bytes, item->tile_resources.primary_bytes);
   cs->tile_resources.secondary_bytes =
      MAX2(cs->tile_resources.secondary_bytes,
           item->tile_resources.secondary_bytes);
   if (memcmp(&cs->cr.zls_control, &item->cr.zls_control,
              sizeof(cs->cr.zls_control))) {
      agx_unpack(NULL, &cs->cr.zls_control, ZLS_CONTROL, za);
      agx_unpack(NULL, &item->cr.zls_control, ZLS_CONTROL, zb);
      za.z_store |= zb.z_store;
      za.s_store |= zb.s_store;
      agx_pack(&cs->cr.zls_control, ZLS_CONTROL, cfg) {
         cfg = za;
      }
   }
   if (last) {
      /* The last instance stores or resolves the attachments, as the final
       * instance of an in-buffer continuation does. When members differed in
       * untouched-RT elision (hk_render_pack_init() cleared
       * cs->render_pack_elided), restore the unelided end-of-tile program so
       * no earlier member's write is lost. */
      cs->cr.eot.main =
         item->render_pack_elided != cs->render_pack_elided
            ? item->render_pack_full_eot_main
            : item->cr.eot.main;
      cs->g17p_pass_ends = item->g17p_pass_ends;
      if (!attachment_set_merge(&cs->fragment_attachments,
                                &item->fragment_attachments, true))
         return false;
   }

   /* A member's vertex-end or fragment-end timestamp becomes the merged
    * render's: its vertex half ends after the member's vertex work, its
    * fragment half after the member's fragment work, and the packet waits
    * for everything the member waited for. A run holds at most one of each
    * (hk_render_pack_stamp_slots()). */
   if (item->timestamp.vtx_end.handle)
      cs->timestamp.vtx_end = item->timestamp.vtx_end;
   if (item->timestamp.end.handle)
      cs->timestamp.end = item->timestamp.end;
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
   const bool exact_bloom = cs->sync.vtx_images && !cs->sync.vtx_img_unsafe;
   const uint8_t ta_carry =
      cs->sync.kind | (cs->sync.vtx_images ? HK_SYNC_GFX_IMG : 0);
   cs->sync.dep_ta_img_prior |= item->sync.dep_ta_img_prior;
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
         if (!last && (ta_carry & BITFIELD_BIT(k))) {
            if (sq == HK_SYNC_VDM && exact_bloom &&
                BITFIELD_BIT(k) == HK_SYNC_GFX_IMG)
               cs->sync.dep_ta_img_prior |= req->prior;
            else
               cs->sync.dep_ta[sq].prior |= req->prior;
         }
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

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"
#include "util/u_dynarray.h"
#include "vulkan/vulkan_core.h"
#include "hk_pass_merge.h"
#include "hk_sync.h"

struct hk_cmd_buffer;
struct hk_cs;
struct hk_image_view;
struct vk_image;

/*
 * Submit-time render packing: consecutive command buffers of one submission
 * that each hold a single render stream, where every stream continues the
 * render pass instance the one before it ended, run as one tile pass.
 *
 * Engines that record one render pass in several command buffers (vkd3d's
 * command lists, GTA V's G-buffer pass recorded as ~21 of them) otherwise pay
 * a store and a reload of every attachment at each command buffer boundary.
 * hk_m5_pass_merge_begin() already continues an instance in the stream of the
 * one before it inside a command buffer; this is the same continuation across
 * command buffers, decided when both are known, at queue submission:
 *
 *  - Recording notes how each command buffer begins and ends
 *    (struct hk_render_pack_edges): whether its first render pass instance
 *    began before anything but pipeline barriers was recorded (the head), and
 *    whether nothing but pipeline barriers followed its last one, which is
 *    still a continuation candidate (the tail, cmd->pass_merge at
 *    vkEndCommandBuffer). The barriers are classified exactly as the in-buffer
 *    continuation classifies them (hk_m5_pass_merge_dependency()).
 *
 *  - At submission, adjacent members A and B pack when B's head continues A's
 *    tail as hk_m5_pass_merge_begin() requires of two instances in one
 *    command buffer (hk_render_pack_continues()): the same attachment views,
 *    B loads every attachment and stores it as A does, no resolves or
 *    feedback loops, the same render area, layers and samples. Their streams
 *    must also describe the same tile pass (hk_render_pack_streams_match()):
 *    identical render registers, which include the background and end-of-tile
 *    programs and the depth/stencil buffers, the same tilebuffer layout,
 *    sample positions and attachment set, and scissor and depth-bias arrays
 *    one of which is a prefix of the other, so the longer one serves the
 *    draws of both. A dependency that is ordered (not framebuffer-local)
 *    additionally needs the memory-equivalence proof of hk_m5_ordered_refusal()
 *    over both whole streams.
 *
 *  - A packed run is submitted as one render command: a submission-owned
 *    command buffer (the packet) whose stream starts at the first member's
 *    and whose synchronization folds the members' (hk_render_pack_add()),
 *    so the resolver gives it the waits of every member and later work waits
 *    for it where it waited for any member. The members' streams are chained
 *    by writing a stream link over each terminate record but the last
 *    (hk_render_pack_write_tail()).
 *
 * A recorded stream is never edited but for that one record, and only for
 * the submission that packs it: every submission writes each member's tail
 * afresh, as a link to the next member or as the terminate record it was
 * recorded with, before the kernel sees it. That is safe because a command
 * buffer recorded without SIMULTANEOUS_USE cannot be pending in two
 * submissions (VUID-vkQueueSubmit2-commandBuffer-03874), so no GPU reads the
 * record while it changes; a simultaneous-use command buffer is only ever
 * the last member, whose stream is not written.
 */

/* Colour attachments, then depth and stencil. */
#define HK_RENDER_PACK_COLORS     8
#define HK_RENDER_PACK_DEPTH      (HK_RENDER_PACK_COLORS)
#define HK_RENDER_PACK_STENCIL    (HK_RENDER_PACK_COLORS + 1)
#define HK_RENDER_PACK_SLOTS      (HK_RENDER_PACK_COLORS + 2)

/* Layout transitions of images that may or may not be attachments of the
 * first instance a head remembers until that instance begins. */
#define HK_RENDER_PACK_TRANSITIONS 4

/* One attachment slot of a render pass instance, as it began (head) or as it
 * ended (tail). */
struct hk_render_pack_att {
   const struct hk_image_view *iview;
   const struct vk_image *image;
   uint32_t load_op, store_op, layout, resolve_mode, flags;
   uint8_t samples;
   /* VkAttachmentFeedbackLoopInfoEXT::feedbackLoopEnable (head only). */
   bool feedback_loop;
   /* The image may hold its contents through another resource
    * (VK_IMAGE_CREATE_ALIAS_BIT, sparse binding). */
   bool aliasable;
};

struct hk_render_pack_pass {
   struct hk_render_pack_att att[HK_RENDER_PACK_SLOTS];
   VkRect2D area;
   uint32_t layer_count, view_mask, flags;
   uint8_t color_count;
   /* Every attachment is in an attachment-only layout, which no descriptor
    * may name (hk_m5_attachments_exclude_descriptors()). */
   bool layout_proof;
};

/* Per command buffer (hk_cmd_buffer::render_pack). */
struct hk_render_pack_edges {
   /* Head, while recording: nothing but pipeline barriers since
    * vkBeginCommandBuffer, as of sync epoch head_epoch, and their class
    * (enum hk_m5_merge_dep). Their layout transitions are remembered until
    * the first instance's attachments are known (transitions[]), and so is
    * whether one would end a multisampled instance (head_msaa_end). */
   bool head_open;
   bool head_msaa_end;
   bool head_non_local_src;
   bool head_att_available;
   bool head_att_not_visible;
   bool transitions_overflow;
   bool head_undef_alias_overflow;
   uint8_t head_dep;
   uint8_t transition_count;
   uint8_t head_undef_alias_mask;
   uint32_t head_epoch;
   const struct vk_image *transitions[HK_RENDER_PACK_TRANSITIONS];

   /* The first render pass instance began the first stream with nothing but
    * those barriers before it: head_pass is how it began, head_dep their
    * class against its attachments. */
   bool head;
   struct hk_render_pack_pass head_pass;
   /* The stream that instance left when it ended, and its recording number
    * (streams are recycled, so an address alone could be a later one). */
   bool head_ended;
   const struct hk_cs *head_stream;
   uint32_t head_seq;

   /* Tail, while recording: the last render pass instance ended as a
    * continuation candidate (hk_m5_pass_merge_candidate()) in tail_stream,
    * which then ended at tail_current in the chunk ending at tail_end, and
    * nothing but pipeline barriers of class tail_dep followed, as of
    * tail_next_seq and sync epoch tail_epoch. */
   bool tail_open;
   bool tail_msaa;
   bool tail_non_local_src;
   bool tail_att_available;
   bool tail_att_not_visible;
   uint8_t tail_dep;
   uint32_t tail_epoch, tail_next_seq, tail_seq;
   const struct hk_cs *tail_stream;
   const void *tail_current, *tail_end;

   /* The last instance ended the only stream with nothing but those
    * barriers after it: tail_pass is how it ended. */
   bool tail;
   struct hk_render_pack_pass tail_pass;

   /* Empty suspend/resume fixup command buffer: no control streams, barriers,
    * events, timestamps, or queries, holding only an empty RESUMING (closing)
    * or empty SUSPENDING (opening) render pass. */
   bool empty_resume;
   bool empty_suspend;
   struct hk_render_pack_pass empty_pass;

   /* Frame anatomy only: why the command buffer has no head or tail, and,
    * as of its latest submission with frame anatomy on, why its tail did
    * not pack with the next command buffer's head (next_who: 0 the boundary,
    * 1 this command buffer is no member, 2 the next one is not). */
   const char *head_why, *tail_why, *next_why;
   uint8_t next_who;

   /* The terminate record of the command buffer's only render stream, and
    * the stream address a submission wrote a link to over it (0: the
    * terminate is in place). */
   uint8_t *terminate;
   uint64_t linked;
};

/* Whether `b`, beginning, continues `a` as it ended, as two instances of one
 * command buffer continue (hk_m5_pass_merge_begin()). `ordered`: a recorded
 * dependency between them orders beyond attachment operations.
 * `non_local_src`: a recorded barrier between them (or inside an earlier
 * member of the run) has a graphics first scope outside the attachment stages
 * (hk_m5_pass_merge_non_local_src()), forbidding a final-pass resolve.
 * Returns NULL or why not. */
const char *hk_render_pack_continues(const struct hk_render_pack_pass *a,
                                     const struct hk_render_pack_pass *b,
                                     bool ordered, bool non_local_src,
                                     bool att_visible);

/* What the members a packed run holds so far bring to its next boundary. In
 * one tile pass every member's vertex work may run ahead of every earlier
 * member's fragment work, so an ordered dependency anywhere in the run needs
 * the memory-equivalence proof over all of them, as the merged stream of an
 * in-buffer continuation carries it (hk_m5_pass_merge_begin()). */
struct hk_render_pack_run_state {
   /* An ordered dependency inside the run, between members or inside one. */
   bool ordered;
   /* A barrier inside the run (between members or inside one) has a graphics
    * first scope outside the attachment stages, forbidding a final resolve. */
   bool non_local_src;
   /* A barrier in the run made attachment writes available; any ordered
    * barrier after it (even with srcAccessMask == 0) chains visibility. */
   bool att_available;
   /* An ordered dependency in the run makes attachment writes visible to
    * later shaders (requires attachment-only layouts and non-aliasable images). */
   bool att_visible;
   /* A member's shaders write memory other than its attachments. */
   bool writes_memory;
   /* A member draws with a stage that runs ahead of the tile pass
    * (hk_cs::render_pack_unsafe). */
   bool unsafe;
   /* A member pushed one of its attachments as a descriptor. */
   bool pushed_attachment;
};

/* The run of one member, `cmd` (stream `cs`). */
void hk_render_pack_run_start(struct hk_render_pack_run_state *run,
                              const struct hk_cmd_buffer *cmd,
                              const struct hk_cs *cs);

/* Whether member `b` (stream `bs`) may follow member `a` at the end of
 * `run`, whose first member's stream is `first`: their edges continue
 * (hk_render_pack_continues()), an ordered dependency in the run has the
 * memory-equivalence proof over every member, and `bs` programs the tile pass
 * `first` does (hk_render_pack_streams_match()). Returns NULL or why not. */
const char *hk_render_pack_boundary(const struct hk_render_pack_run_state *run,
                                    const struct hk_cmd_buffer *a,
                                    const struct hk_cmd_buffer *b,
                                    const struct hk_cs *bs,
                                    const struct hk_cs *first);

/* Whether `post` is a pure driver geometry heap-slot reset stream (no API
 * compute, no query/timestamp/OQ work, and bound to a non-zero heap slot). */
bool hk_render_pack_post_is_heap_reset(const struct hk_cs *post);

/* Whether a later member's leading CDM stream `bs_pre` may be hoisted ahead of
 * earlier members already in `state`. Returns NULL if safe, or a refusal reason. */
const char *hk_render_pack_pre_cs_refusal(
   const struct hk_render_pack_run_state *state, const struct hk_cs *bs_pre,
   unsigned run_post, uint8_t post_heap_slots, bool post_heap_only,
   uint8_t run_vdm_kinds, const uint8_t run_vdm_tail_req[HK_SYNC_NUM_KINDS],
   const uint8_t run_post_tail_req[HK_SYNC_NUM_KINDS]);

/* `b` (stream `bs`) joins `run` after `a`, its boundary accepted. */
void hk_render_pack_run_join(struct hk_render_pack_run_state *run,
                             const struct hk_cmd_buffer *a,
                             const struct hk_cmd_buffer *b,
                             const struct hk_cs *bs);

/* Whether two render streams program the same tile pass, so `b`'s draws may
 * follow `a`'s in one render command. Returns NULL or why not. */
const char *hk_render_pack_streams_match(const struct hk_cs *a,
                                         const struct hk_cs *b);

/* Whether one of two staged arrays (scissor or depth bias) is a prefix of the
 * other, so the longer serves the draws of both. *longer gets that one. */
bool hk_render_pack_array_prefix(const struct util_dynarray *a,
                                 const struct util_dynarray *b,
                                 const struct util_dynarray **longer);

/* Locates the unique VDM stream of `cmd`, optionally bracketed by a single
 * pre-graphics CDM stream before it and/or a single post-graphics CDM stream
 * after it. Does not inspect sync/query validity; used by packet init and
 * end-of-recording edge setup. */
const struct hk_cs *hk_render_pack_find_vdm(const struct hk_cmd_buffer *cmd,
                                            const struct hk_cs **pre_out,
                                            const struct hk_cs **post_out);

/* Like hk_render_pack_member(), also returning optional pre/post CDM streams
 * belonging to the same pass. When `first` is true, `cmd` starts a packed run
 * so its leading pre-render CDM stream stays in submission order and may carry
 * meta/transfer work; later members (`first == false`) must hoist their
 * pre-render compute ahead of earlier render passes. */
const struct hk_cs *hk_render_pack_member_streams(
   const struct hk_cmd_buffer *cmd, bool first, const struct hk_cs **pre_out,
   const struct hk_cs **post_out);

/* The only stream of `cmd` if it may be a member of a packed run (a render
 * stream with draws and no query, start timestamp, event or in-buffer
 * cut), or NULL. */
const struct hk_cs *hk_render_pack_member(const struct hk_cmd_buffer *cmd);

/* The merged render's timestamps a member's take (bit 0 vertex end, bit 1
 * fragment end): a run holds at most one member timestamp of each. */
unsigned hk_render_pack_stamp_slots(const struct hk_cs *cs);

/* Why `cmd` is no member (hk_render_pack_member()), or NULL. */
const char *hk_render_pack_member_refusal(const struct hk_cmd_buffer *cmd);
const char *hk_render_pack_member_refusal_for(const struct hk_cmd_buffer *cmd,
                                              bool first);

/* Whether `cmd` is an empty suspend/resume fixup command buffer with no
 * streams or side effects. */
bool hk_render_pack_is_empty_fixup(const struct hk_cmd_buffer *cmd,
                                   bool *is_resume, bool *is_suspend);

/* Whether an empty suspend/resume fixup pass matches `other`'s attachments
 * and geometry. */
bool hk_render_pack_fixup_matches(const struct hk_render_pack_pass *fixup,
                                  const struct hk_render_pack_pass *other);

/* Whether `cmd`'s tail may be written: recorded without SIMULTANEOUS_USE,
 * with room for a stream link at its terminate record. */
bool hk_render_pack_tail_writable(const struct hk_cmd_buffer *cmd);

/* Writes the tail of `e`: a stream link to `target` (the next member's
 * stream, as its VDM link names it) or, with target 0, the terminate record.
 * Returns whether anything changed. */
bool hk_render_pack_write_tail(enum agx_chip chip,
                               struct hk_render_pack_edges *e,
                               uint64_t target);

/* Packet assembly for the `count` members `members`: `cs` becomes a copy of
 * the first member's stream owned by `packet` (the packet only reads what the
 * copy shares with the member), of every member's synchronization kinds and
 * vertex-half image reads, and with the synchronization of nothing yet; the
 * members are then added in order and the packet finished. The packet keeps
 * its render resource list and tile footprints across submissions. */
void hk_render_pack_init(struct hk_cmd_buffer *packet, struct hk_cs *cs,
                         struct hk_cmd_buffer *const *members,
                         unsigned count);

/* Adds member `item` (the stream of `source`): its draws, flags and
 * resources, its synchronization (as hk_pack_fold_member() folds a compute
 * member's) and its tile footprints. */
bool hk_render_pack_add(struct hk_cmd_buffer *packet, struct hk_cs *cs,
                        const struct hk_cmd_buffer *source,
                        const struct hk_cs *item, bool first, bool last,
                        uint8_t *preceding_kinds);

/* The packet's stream is its command buffer's latest of every kind it is. */
void hk_render_pack_finish(struct hk_cmd_buffer *packet, struct hk_cs *cs);

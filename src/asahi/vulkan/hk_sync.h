/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Recorded-dependency tracking for Honeykrisp control streams.
 *
 * A command buffer is a list of control streams, each submitted as one
 * render (VDM) or compute (CDM) command. The kernel orders commands through
 * two per-command fields, vdm_barrier and cdm_barrier, each naming a prefix
 * of the respective subqueue that must complete before the command runs
 * (include/drm-uapi/asahi_drm.h, drm_asahi_cmd_header). The value NONE means
 * the command waits on nothing on that subqueue, not even earlier submits.
 *
 * Vulkan orders work through explicit dependencies: pipeline barriers,
 * events, subpass dependencies (which the common render-pass code lowers to
 * pipeline barriers) and semaphores. Every one of them is prefix-shaped: the
 * first synchronization scope of vkCmdPipelineBarrier2 is "all commands
 * earlier in submission order" restricted to the source stages, and the
 * second is "all commands later in submission order" restricted to the
 * destination stages (Vulkan 1.3, 7.6 "Pipeline Barriers"). The kernel's
 * prefix encoding therefore loses nothing against the barrier model on one
 * queue; what it cannot express is stage granularity (a barrier from vertex
 * work into fragment work still waits on and gates whole commands), and it
 * cannot distinguish two commands in the same prefix. Both would need UAPI
 * extensions.
 *
 * This module records, per stream, the latest earlier stream on each
 * subqueue that a recorded dependency orders it after ("cut"), plus which
 * kinds of pre-command-buffer work it must wait on ("prior"). Kinds
 * approximate the Vulkan stage classes at the granularity the hardware has:
 * graphics, compute and transfer. Streams the driver adds on its own
 * (queries, timestamps, events, heap management) carry the implicit orderings
 * Vulkan requires of them; a stream a meta operation records inherits the
 * application dependency it stands in for because it is created under the
 * same recorded state.
 *
 * At submission the cuts become exact prefix indices, cross-command-buffer
 * dependencies are resolved against the other command buffers in the same
 * submission. With EXACT_PRIOR_BARRIERS, dependencies reaching before this
 * submission name exact queue-local hardware ordinals; destination requirements
 * persist across submissions. Older kernels retain the "all earlier submits"
 * representation and existing per-submit requirement lifetime.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "vulkan/vulkan_core.h"

struct hk_cmd_buffer;
struct hk_cs;
struct hk_device;
struct hk_queue;

/* Which class of Vulkan pipeline stages a stream's work belongs to. Streams
 * the driver adds for its own bookkeeping are HK_SYNC_ALL: every source scope
 * covers them and they wait as their own semantics require.
 */
enum hk_sync_kind {
   HK_SYNC_GFX = 1 << 0,
   HK_SYNC_COMPUTE = 1 << 1,
   HK_SYNC_XFER = 1 << 2,
   HK_SYNC_ALL = HK_SYNC_GFX | HK_SYNC_COMPUTE | HK_SYNC_XFER,

   /* Destination scopes only, never a stream's kind: graphics work in the
    * fragment and attachment stages alone. A render's fragment half waits on
    * requirements of both graphics classes, its vertex (tiling) half only on
    * HK_SYNC_GFX ones. Every other graphics destination, and every stream
    * kind, counts as HK_SYNC_GFX. */
   HK_SYNC_GFX_FRAG = 1 << 3,

   /* Destination scopes only: pre-rasterization shaders that read images.
    * An image barrier can only order the image's own accesses, and a
    * pre-rasterization stage accesses an image only through a shader that
    * reads one. On G17P and M5, image barriers therefore name this class
    * (and the fragment class) where other dependencies name HK_SYNC_GFX: a
    * render's fragment half always waits on it, its vertex half only once it
    * draws with a pre-rasterization shader that reads images
    * (hk_sync_vertex_images). Other chips never record it. */
   HK_SYNC_GFX_IMG = 1 << 4,

   /* Stream kinds only, never a destination scope: work that can write
    * memory a later pre-rasterization stage reads other than through an
    * image. Graphics work writes such memory only from shaders (storage,
    * atomics), through transform feedback, through attachments that can
    * alias other resources, and in its pre- and post-graphics streams; a
    * render without any of those writes only its attachments, which a later
    * pre-rasterization stage can read only through an image (HK_SYNC_GFX_IMG).
    * Every stream of another kind is counted too. On M5 and G17P a
    * dependency from graphics work to pre-rasterization stages orders a
    * render's vertex half only after the latest such stream on the render
    * subqueue (hk_sync_barrier()). */
   HK_SYNC_GFX_MEM = 1 << 5,
};

#define HK_SYNC_NUM_KINDS 6

/* Subqueue index: 0 render (VDM), 1 compute (CDM). */
enum hk_sync_sq {
   HK_SYNC_VDM = 0,
   HK_SYNC_CDM = 1,
};

#define HK_SYNC_NUM_SQ 2

/* A requirement on one subqueue: the latest stream recorded in this command
 * buffer that must complete first (NULL: none in this command buffer), and
 * the kinds of streams recorded before this command buffer that must
 * complete first.
 */
struct hk_sync_dep {
   struct hk_cs *cut;
   uint8_t prior;
};

/*
 * Represents a firmware timestamp request.  Handle is a kernel timestamp object
 * handle, not a GEM handle.
 *
 * The kernel/firmware uses the handle/offset_B to write. We use the address to
 * read the results back. We could deduplicate this, but this is convenient.
 */
struct agx_timestamp_req {
   uint64_t addr;
   uint32_t handle;
   uint32_t offset_B;
};

/*
 * Tile hazards. A render pass instance's load, store and resolve operations
 * only affect pixels inside its render area (Vulkan 1.4, "Render Pass Load
 * Operations" / "Render Pass Store Operations"), so two instances that write
 * pixel-disjoint parts of one attachment need no application barrier. The
 * hardware loads and stores whole tiles, however: a tile the render area only
 * partly covers is reloaded from memory and stored back whole. Two instances
 * that share such a tile race when they run concurrently -- the later one can
 * reload the tile before the earlier one stored it and then write the stale
 * pixels back -- and with per-command barriers nothing else orders them.
 *
 * Sparse multiview masks have the same hazard in the layer dimension: BG/EOT
 * reloads and stores every layer below the highest enabled view, preserving
 * inactive layers. A full X/Y area does not make those writes independent.
 * Record every attachment footprint, including complete writes that a later
 * pass may preserve. Order overlapping footprints when either pass preserves
 * pixels or layers. Complete independent writes retain no extra dependency.
 * Footprints use 32x32-pixel cells, the largest tile (agx_select_tile_size); smaller tiles
 * nest inside them, so the test is conservative for every layout.
 *
 * The rule applies within a command buffer at recording, across the command
 * buffers of one submission and against this queue's earlier, possibly still
 * running submissions at submit time. Other queues and other processes are
 * not covered (see the g1fix notes).
 */
#define HK_SYNC_TILE_CELL_LOG2 5
#define HK_SYNC_MAX_PASS_FOOTPRINTS 20
#define HK_SYNC_RECENT_FOOTPRINTS 256

struct hk_sync_footprint {
   /* The hardware preserves pixels or inactive layers outside the API writes. */
   bool preserve;
   /* Identity of the written image; compared, never dereferenced. */
   const void *image;
   uint32_t level;
   /* Layers [layer0, layer1). */
   uint32_t layer0, layer1;
   /* Inclusive 32x32 cell bounds. */
   int32_t cx0, cx1, cy0, cy1;
};

/* A pass footprint and its first and last attachment-producing streams. */
struct hk_sync_pass_record {
   struct hk_sync_footprint fp;
   struct hk_cs *first, *last;
};

/* A footprint of an earlier submission on this queue and the queue timeline
 * point that submission signals. */
struct hk_sync_recent {
   struct hk_sync_footprint fp;
   uint64_t point;
   enum hk_sync_sq sq;
};

/*
 * G17P transfer writes that nothing orders before graphics work yet.
 *
 * Vulkan does not order a transfer before a later render pass without a
 * dependency, and G17P overlaps the two when there is none. Some
 * applications (and the CTS ray_query null_as cases) clear a resource, name
 * only a non-graphics stage in the following barrier, then use the resource
 * in a draw. Be stricter there, but only where the pass can actually access
 * what the transfer wrote: each transfer recorded outside a pass notes its
 * destination VA range and compute stream, a dependency from the transfer
 * stage into graphics drops the notes, and a render pass begin (attachments)
 * or draw (descriptors, vertex, index and indirect buffers) that may access a
 * noted range orders that stream's prefix before its graphics work. Work that
 * does not touch a noted range keeps the full overlap.
 *
 * Shaders can reach a buffer through its device address, and a transfer
 * whose destination is unknown may write anything: such notes, and writes
 * that no longer fit in the set, match every pass (xfer_any). Only one
 * command buffer is tracked, and aliasing between different resources is
 * covered only where their VA ranges overlap.
 */
#define HK_SYNC_XFER_WRITES 8

struct hk_sync_xfer_write {
   /* GPU VA range [lo, hi) the stream writes. */
   uint64_t lo, hi;
   struct hk_cs *cs;
};

/* The destination of the transfer command being recorded. */
struct hk_sync_xfer_dst {
   uint64_t lo, hi;
   bool valid;
   bool device_address;
};

/* Per-stream state, embedded in hk_cs. */
struct hk_sync_stream {
   /* Recording order. Equal to list order for every pair the tracking
    * compares: a pre-graphics stream takes the number reserved just below
    * its render so it sorts before it, where the list places it. */
   uint32_t seq;

   /* The latest HK_SYNC_GFX_MEM stream on this stream's subqueue recorded
    * before it. */
   struct hk_cs *prev_mem;

   /* Set at submission: 1-based position among the commands of this
    * subqueue in the ioctl being built. */
   uint32_t index;

   /* Resolved barrier fields for the ioctl being built. */
   uint16_t barrier[HK_SYNC_NUM_SQ];

   /* Exact historical source prefixes, resolved afresh at each submission. */
   uint64_t prior[HK_SYNC_NUM_SQ];

   uint8_t kind;

   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];

   /* For a render: what its vertex (tiling) half needs, a subset of dep
    * without the fragment-only requirements (HK_SYNC_GFX_FRAG) and tile
    * hazards. Resolved like dep into barrier_ta/prior_ta; ta_split when
    * they differ from barrier/prior. */
   struct hk_sync_dep dep_ta[HK_SYNC_NUM_SQ];
   uint16_t barrier_ta[HK_SYNC_NUM_SQ];
   uint64_t prior_ta[HK_SYNC_NUM_SQ];
   bool ta_split;

   /* For a render: it draws with a pre-rasterization shader that reads
    * images, so its vertex half also waits on HK_SYNC_GFX_IMG, as of the
    * dependency epoch vtx_images_epoch. */
   bool vtx_images;
   uint32_t vtx_images_epoch;
};

/* Per-command-buffer state. */
struct hk_sync_state {
   uint32_t next_seq;

   /* Latest stream of each kind on each subqueue, and the latest stream of
    * any kind. */
   struct hk_cs *last[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];
   struct hk_cs *last_any[HK_SYNC_NUM_SQ];

   /* What the next stream of each kind must wait on. Monotonic: a barrier's
    * second scope covers every later command. This is also the command
    * buffer's tail requirement, applied to later command buffers of the same
    * submission at queue submit. */
   struct hk_sync_dep req[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];

   /* A self-dependency has both scopes inside the current pass instance.
    * Software graphics stages execute on CDM, so it must order those streams
    * too, without becoming a requirement on work after the pass. */
   struct hk_sync_dep pass_req[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];
   uint32_t pass_start_seq;

   /* Dependencies added to every stream created while set. Used for the
    * driver's own passes that continue a render (resolves). */
   struct hk_sync_dep internal[HK_SYNC_NUM_SQ];
   bool internal_active;

   /* Render pass instance bookkeeping. */
   bool in_pass;
   bool resume_pending;
   bool heap_pass;
   struct hk_cs *pass_vdm;

   /* The open general compute stream must be re-validated against the
    * requirements before it receives more work. */
   bool cs_check_pending;

   /* Incremented by every operation that changes recorded dependencies or
    * stream bookkeeping. A reader that cached a decision about the tracked
    * state compares it to know whether anything happened since. */
   uint32_t epoch;

   /* A meta helper selected an exact stream; do not replace it. */
   bool pin_cs;

   /* Kind for streams created now. 0 infers from the recording context. */
   uint8_t kind_override;

   /* Depth of transfer entry points on the stack. */
   uint16_t xfer_depth;

   /* G17P transfer writes not yet ordered before graphics (see
    * hk_sync_xfer_write): the destination of the transfer being recorded,
    * the noted writes, and the latest stream with a write any graphics work
    * may access. xfer_version changes whenever the notes do. */
   struct hk_sync_xfer_dst xfer_dst;
   struct hk_sync_xfer_write xfer_writes[HK_SYNC_XFER_WRITES];
   uint8_t xfer_write_count;
   struct hk_cs *xfer_any;
   uint32_t xfer_version;

   /* The draw state the last draw-time check found no noted write in: valid
    * while xfer_version, the descriptor and vertex buffer generations and the
    * index buffer are unchanged (hk_cmd_draw.c). */
   struct {
      bool valid;
      uint32_t version;
      uint64_t descriptors, vertex_buffers, index;
   } xfer_clean;

   /* A top-of-pipe timestamp waiting for the next stream to carry it as its
    * start timestamp. */
   struct agx_timestamp_req pending_start_ts;

   /* Tile hazards: attachment footprints of the open pass instance and
    * the latest earlier render stream it shares a tile with. The command
    * buffer keeps the recorded instances (hk_cmd_buffer::sync_pass_records). */
   struct hk_sync_footprint pass_fp[HK_SYNC_MAX_PASS_FOOTPRINTS];
   uint8_t pass_fp_count;
   struct hk_cs *pass_tile_wait[HK_SYNC_NUM_SQ];
   /* Attachment producers, including a compute replacement of a clear. */
   struct hk_cs *pass_tile_first, *pass_tile_last;
   /* The last attachment producer of a suspended pass instance, which the
    * resuming instance continues from. */
   struct hk_cs *suspended_last;
};

/* Per-queue state: kinds of streams ever submitted on each subqueue. A
 * dependency on earlier submits of a kind that never ran is no dependency. */
struct hk_sync_queue {
   uint8_t ever[HK_SYNC_NUM_SQ];
   bool exact_prior;
   /* G13/G14 retain upstream's order after both preceding engine prefixes. */
   bool serialize;
   /* Serialized queue that still records dependencies so a render whose
    * vertex half depends on no earlier render can order only its fragment
    * half after the serialized prefix (HK_G13_STAGE_OVERLAP=0 disables). */
   bool stage_overlap;
   uint64_t submitted[HK_SYNC_NUM_SQ];
   uint64_t last_absolute[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];
   uint64_t tail[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];

   /* Tile hazards across submissions: a ring of the footprints recently
    * submitted on this queue. An entry is live until the queue timeline
    * reaches its point. `evicted` is the latest point of an entry the ring
    * dropped, which keeps the check conservative while that point is live. */
   struct hk_sync_recent recent[HK_SYNC_RECENT_FOOTPRINTS];
   uint32_t recent_next, recent_count;
   uint64_t recent_max, evicted;
   /* G16S: complete writes need no preservation query against other
    * complete writes, including entries evicted from the bounded ring.
    * Retain all writes for a future partial-tile or sparse-layer pass.
    */
   bool precise_tile_evictions;
   /* A single command buffer already owns the complete footprint list. */
   bool direct_single_footprints;
   uint64_t last_preserve, evicted_preserve;
};

void hk_sync_reset(struct hk_cmd_buffer *cmd);

/* Stage masks to stream kinds, for a first synchronization scope. */
uint8_t hk_sync_kinds_src(VkPipelineStageFlags2 stages);
/* And for a second synchronization scope. */
uint8_t hk_sync_kinds_dst(VkPipelineStageFlags2 stages);

/* Union of the scopes of every barrier in a dependency. With image_class,
 * an image barrier whose second scope names graphics work names
 * HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG instead of HK_SYNC_GFX. */
void hk_sync_dependency_kinds(const VkDependencyInfo *info, bool image_class,
                              uint8_t *src, uint8_t *dst);

/* Record an execution dependency: all earlier streams of the source kinds
 * complete before any later stream of the destination kinds starts. */
void hk_sync_barrier(struct hk_cmd_buffer *cmd, uint8_t src_kinds,
                     uint8_t dst_kinds);
void hk_sync_pass_barrier(struct hk_cmd_buffer *cmd, uint8_t src_kinds,
                         uint8_t dst_kinds);

/* The requirement that "all earlier streams of these kinds complete", for
 * the stream self when it is one the requirement will be applied to (it is
 * then every stream before self), or for a stream still to be created. */
void hk_sync_src_dep(struct hk_cmd_buffer *cmd, uint8_t src_kinds,
                     struct hk_cs *self, struct hk_sync_dep dep[HK_SYNC_NUM_SQ]);

/* The render stream draws with a pre-rasterization shader that reads images:
 * its vertex half joins the image-barrier requirements (HK_SYNC_GFX_IMG). */
void hk_sync_vertex_images(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
void hk_sync_note_memory_write(struct hk_cmd_buffer *cmd, struct hk_cs *cs);

/* Every later stream of the given kinds waits on cs. */
void hk_sync_after_stream(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                          uint8_t dst_kinds);

/* Hooks from the control-stream machinery. */
void hk_sync_stream_created(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                            struct hk_cs **slot, bool compute);
void hk_sync_check_cs(struct hk_cmd_buffer *cmd);

/* Stage class of work entering the open general compute stream. Keep this
 * equivalent to stream_kind() for &cmd->current_cs.cs. */
#define hk_sync_general_cs_kind(cmd)                                           \
   ((cmd)->sync.kind_override ? (cmd)->sync.kind_override                       \
    : (cmd)->sync.xfer_depth  ? HK_SYNC_XFER                                    \
                              : HK_SYNC_COMPUTE)
void hk_sync_forget_stream(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
void hk_sync_merge(struct hk_cmd_buffer *cmd, struct hk_cs *into,
                   struct hk_cs *from);

/* M5: merging compute stream from into the adjacent stream into would make
 * the work into already holds wait for a render it does not wait for. */
bool hk_sync_merge_delays(struct hk_cmd_buffer *cmd, const struct hk_cs *into,
                          const struct hk_cs *from);

/* Obtain the general compute stream, ended and recreated if it is not
 * positioned after dep. The returned stream carries dep. */
struct hk_cs *hk_sync_get_cs_after(struct hk_cmd_buffer *cmd,
                                   const struct hk_sync_dep dep[HK_SYNC_NUM_SQ]);

/* End the general compute stream if it carries an end timestamp, so a query
 * command that follows cannot run before the stamp is taken. */
void hk_sync_end_cs_if_timestamped(struct hk_cmd_buffer *cmd);

/* Timestamps. */
void hk_sync_write_timestamp(struct hk_cmd_buffer *cmd,
                             VkPipelineStageFlags2 stage,
                             struct agx_timestamp_req req);
void hk_sync_flush_pending_timestamp(struct hk_cmd_buffer *cmd);

/* G17P transfer writes (see hk_sync_xfer_write). Note that the transfer
 * being recorded writes into cs. */
void hk_sync_note_xfer_write(struct hk_cmd_buffer *cmd, struct hk_cs *cs);

static inline bool
hk_sync_xfer_pending(const struct hk_sync_state *st)
{
   return st->xfer_write_count || st->xfer_any;
}

/* The latest stream of best and the noted streams that write [lo, hi). */
struct hk_cs *hk_sync_xfer_match(const struct hk_cmd_buffer *cmd,
                                 uint64_t lo, uint64_t hi, struct hk_cs *best);

/* The latest stream of best and every noted stream: graphics work that may
 * access memory this command buffer cannot see. */
struct hk_cs *hk_sync_xfer_match_all(const struct hk_cmd_buffer *cmd,
                                     struct hk_cs *best);

/* Order the prefix of the compute subqueue up to cs before every graphics
 * stream from now on, including the open render and its pre-graphics stream.
 * The caller has made the prefix's writes visible and closed cs to further
 * work. Drops the notes the prefix covers. */
void hk_sync_order_xfer(struct hk_cmd_buffer *cmd, struct hk_cs *cs);

/* Render pass instance boundaries. */
void hk_sync_begin_rendering(struct hk_cmd_buffer *cmd, bool resuming);
void hk_sync_first_render_stream(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
void hk_sync_end_rendering(struct hk_cmd_buffer *cmd, bool suspending);

/* A new render pass instance continues in cs, the render stream of the
 * instance that just ended, instead of a stream of its own: the stream takes
 * on the requirements a render stream created now would, and the pass
 * bookkeeping starts as if cs had just been created for the instance. The
 * caller guarantees cs is the latest stream of the command buffer. */
void hk_sync_merge_rendering(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
void hk_sync_heap_used(struct hk_cmd_buffer *cmd);

/* Tile hazards: a written attachment of the pass instance being begun, then
 * the end of the list, before the instance creates its first stream. */
void hk_sync_pass_footprint(struct hk_cmd_buffer *cmd, const void *image,
                            uint32_t level, uint32_t layer0, uint32_t layer1,
                            uint32_t level_width, uint32_t level_height,
                            VkRect2D area, bool preserve_layers);
void hk_sync_pass_footprints_done(struct hk_cmd_buffer *cmd);
void hk_sync_replace_render(struct hk_cmd_buffer *cmd, struct hk_cs *from,
                            struct hk_cs *into);
/* Whether a submission records tile-hazard footprints, and whether the
 * queue may still be running one, so the caller fetches the completed
 * timeline point only when it can matter. */
bool hk_sync_submit_needs_completed(const struct hk_queue *queue,
                                    struct hk_cmd_buffer *const *cmdbufs,
                                    unsigned count);

/* The driver's own resolve pass after a render: waits on the render. */
void hk_sync_resolve_begin(struct hk_cmd_buffer *cmd);
void hk_sync_resolve_end(struct hk_cmd_buffer *cmd);

/* Transfer entry points bracket their recording so the streams they create
 * belong to the transfer stage class. Defined in hk_cmd_buffer.h. */

/* Submission: resolve every stream's barrier fields for one ioctl payload.
 * cmdbufs are the command buffers in submission order; first_cdm_is_flush
 * says the payload opens with a driver-added compute command that every
 * command must run after; chain_first says the commands must transitively
 * follow the first hardware command (the kernel attaches the ioctl's input
 * syncs only to it).
 */
bool hk_sync_resolve_submit(struct hk_queue *queue,
                            struct hk_cmd_buffer *const *cmdbufs,
                            unsigned count, bool first_cdm_is_flush,
                            bool chain_first, uint64_t completed);

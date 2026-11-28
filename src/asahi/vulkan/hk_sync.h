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
 * covers them and they wait as their own semantics require. Streams that only
 * write timestamps are HK_SYNC_STAMP instead.
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

   /* Stream kinds and first scopes only, never named by an application
    * dependency: a command the driver records to carry a timestamp alone, to
    * copy timestamps (hk_sync_write_timestamp()), or to publish post-graphics
    * query availability (hk_flush_oq_pending(),
    * hk_queue_query_available_write()). It writes nothing but query memory,
    * which only query commands, events, and later timestamps observe, so only
    * their first scopes name it (HK_SYNC_QUERY_SCOPE, hk_sync_stamp_scope());
    * a pipeline barrier, even an ALL_COMMANDS one, does not order later work
    * after it. It waits like work of every class (hk_sync_full_kinds()). */
   HK_SYNC_STAMP = 1 << 6,

   /* Stream kinds and first scopes only, never named by an application
    * dependency: a command that resets timestamp queries
    * (hk_CmdResetQueryPool()). Like HK_SYNC_STAMP it writes nothing but
    * query memory, and only later query commands and timestamps are in its
    * second scope (Vulkan 1.4, 17.2: the reset is ordered against the
    * commands that reference the same queries). Unlike a timestamp write it
    * is not in the first scope of another reset: two resets write the same
    * values, so their order does not matter. It waits like work of every
    * class. */
   HK_SYNC_QRESET = 1 << 7,
};

#define HK_SYNC_NUM_KINDS 8

/* Every kind a stream can be of: a first scope naming all of them is every
 * earlier stream on a subqueue. */
#define HK_SYNC_STREAM_KINDS (HK_SYNC_ALL | HK_SYNC_STAMP | HK_SYNC_QRESET)

/* The first scope of a query command (reset, result copy, property write):
 * every earlier stream, timestamp carriers included. */
#define HK_SYNC_QUERY_SCOPE HK_SYNC_STREAM_KINDS

/* The first scope of a timestamp query reset: every earlier stream but other
 * timestamp query resets. */
#define HK_SYNC_QRESET_SCOPE (HK_SYNC_STREAM_KINDS & ~HK_SYNC_QRESET)

/* The first scope of a timestamp or event of the given source kinds: an
 * earlier timestamp write is in it unless it is empty (TOP_OF_PIPE). */
static inline uint8_t
hk_sync_stamp_scope(uint8_t src_kinds)
{
   return src_kinds ? src_kinds | HK_SYNC_STAMP : 0;
}

/* The destination classes a stream of the given kinds belongs to: graphics
 * work is in every graphics class (HK_SYNC_GFX_FRAG, HK_SYNC_GFX_IMG), and a
 * timestamp stream in every class. A stream waits for the requirements
 * carried in for each of them. */
static inline uint8_t
hk_sync_full_kinds(uint8_t kinds)
{
   if (kinds & (HK_SYNC_STAMP | HK_SYNC_QRESET))
      kinds |= HK_SYNC_ALL;
   return kinds & HK_SYNC_GFX
             ? kinds | HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG
             : kinds;
}

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
   /* A VkEvent status word set by the retirement timestamp
    * (hk_sync_stamp_event()), not a timestamp query: it has no address to
    * read, and a reset of the event can clear it before a copy would read it,
    * so it is never the source of a timestamp copy. */
   bool event;
};

/* A timestamp write satisfied by copying an earlier stamp: the query it
 * writes and the stamp it reads. */
struct hk_stamp_copy {
   struct agx_timestamp_req dst;
   struct agx_timestamp_req src;
};

/* Copies one recording waits with before it records them in one launch. */
#define HK_SYNC_STAMP_COPIES 64

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
   /* It does so for the area or the layers: a partly covered tile, or layers
    * a sparse view mask skips. A footprint that preserves only because it
    * loads its attachment covers whole tiles of every layer it names. */
   bool partial;
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

#define HK_SYNC_RECENT_VDM_ATT 64

static inline uint64_t
hk_sync_image_bloom(uint64_t key)
{
   if (!key)
      return 0;
   uint64_t h = key;
   h ^= h >> 33;
   h *= UINT64_C(0xff51afd7ed558ccd);
   h ^= h >> 33;
   return UINT64_C(1) << (h & 63);
}

struct hk_sync_vdm_att {
   uint64_t ordinal;
   uint64_t att_bloom;
};

/* Per-stream state, embedded in hk_cs. */
struct hk_sync_stream {
   /* Recording order. Equal to list order for every pair the tracking
    * compares: a pre-graphics stream takes the number reserved just below
    * its render so it sorts before it, where the list places it. Multiple
    * pre-graphics streams ahead of the same render break ties by pre_seq. */
   uint32_t seq;
   uint32_t pre_seq;
   uint32_t pre_count;

   /* The latest HK_SYNC_GFX_MEM stream on this stream's subqueue recorded
    * before it. */
   struct hk_cs *prev_mem;

   /* Set at submission: 1-based position among the commands of this
    * subqueue in the ioctl being built. Like the resolved fields below, it
    * describes one occurrence of the stream: a simultaneous-use command
    * buffer may occur more than once in a submission
    * (hk_sync_resolve_submit()). */
   uint32_t index;

   /* Resolved barrier fields for the ioctl being built. */
   uint16_t barrier[HK_SYNC_NUM_SQ];

   /* Exact historical source prefixes, resolved afresh at each submission. */
   uint64_t prior[HK_SYNC_NUM_SQ];

   uint8_t kind;
   /* Render stream had an optimal VK_IMAGE_CREATE_ALIAS_BIT attachment whose
    * memory had no buffer binding at record time. Upgraded to HK_SYNC_GFX_MEM
    * at submit if hk_device::buffer_alias_mem_gen advanced since recording. */
   bool alias_image_render;

   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];

   /* For a render: what its vertex (tiling) half needs, a subset of dep
    * without the fragment-only requirements (HK_SYNC_GFX_FRAG). Tile
    * hazards stay in it: starting tiling ahead of them corrupts the shared
    * tile. Resolved like dep into barrier_ta/prior_ta; ta_split when they
    * differ from barrier/prior. */
   struct hk_sync_dep dep_ta[HK_SYNC_NUM_SQ];
   uint16_t barrier_ta[HK_SYNC_NUM_SQ];
   uint64_t prior_ta[HK_SYNC_NUM_SQ];
   bool ta_split;

   /* For a render: it draws with a pre-rasterization shader that reads
    * images, so its vertex half also waits on HK_SYNC_GFX_IMG, as of the
    * dependency epoch vtx_images_epoch. When !vtx_img_unsafe, vtx_img_bloom
    * holds the bloom filter of image plane GPU VAs read by vertex shaders
    * so VDM waits only on renders whose att_bloom intersects it. */
   bool vtx_images;
   bool vtx_img_unsafe;
   uint8_t dep_ta_img_prior;
   uint32_t vtx_images_epoch;
   uint64_t att_bloom;
   uint64_t vtx_img_bloom;

   /* A layout transition the driver records as work
    * (brl_sync_transition_begin()), or a stream one was merged into: a
    * submission adds to its requirement the requirements earlier command
    * buffers carry into brl_carry, the classes of the transition's first
    * scope (barrier chains), instead of those of the stream's own classes.
    * Its transfer class orders it after earlier transfer work, not after
    * what earlier dependencies ask of later transfer work. */
   bool brl_transition;
   uint8_t brl_carry;
   /* 1..HK_HEAP_SLOTS when this stream belongs to a geometry-heap slot pass. */
   uint8_t heap_slot;
};

#define HK_HEAP_SLOTS 8u

/* The resolved fields of one occurrence of a stream in a submission. */
struct hk_sync_resolved {
   uint16_t barrier[HK_SYNC_NUM_SQ];
   uint16_t barrier_ta[HK_SYNC_NUM_SQ];
   uint64_t prior[HK_SYNC_NUM_SQ];
   uint64_t prior_ta[HK_SYNC_NUM_SQ];
   bool ta_split;
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

   /* Draw-state slices the last draw-time check found no noted write in:
    * each slice stays valid independently while xfer_version and that
    * slice's resource identity are unchanged (hk_cmd_draw.c). */
   struct {
      uint32_t desc_version;
      uint32_t vb_version;
      uint32_t index_version;
      bool desc_valid;
      bool vb_valid;
      bool index_valid;
      uint64_t descriptors;
      uint64_t vertex_buffers;
      uint64_t index_addr;
      uint64_t index_range;
   } xfer_clean;

   /* A top-of-pipe timestamp waiting for the next stream to carry it as its
    * start timestamp. pending_start_ts_unordered marks a leading stamp on a
    * chip whose start stamps latch before barriers resolve (G17P, M5):
    * queue submission keeps it on the command's start when nothing precedes
    * it, and otherwise moves it to an ordered end-stamp carrier. */
   struct agx_timestamp_req pending_start_ts;
   bool pending_start_ts_unordered;

   /* On chips whose start stamps latch before barriers resolve: a
    * timestamp recorded before any stream of the command buffer, waiting for
    * a first render stream to carry it on its vertex end stamp (vtx_end),
    * which is latched after the render's barriers and remains distinct from
    * fragment end (end). That render waits for lead_scope, the kinds of the
    * stamp's first scope as hk_sync_src_dep() takes them: the resets of
    * timestamp queries alone for a top-of-pipe stamp, whose first scope is
    * empty; the earlier work of its stages and the earlier timestamps for
    * any other. A compute stream has no separate start-side end stamp, so
    * opening one first flushes lead_ts into a preceding timestamp-only
    * carrier (hk_sync_flush_lead_timestamp()). */
   struct agx_timestamp_req lead_ts;
   uint8_t lead_scope;
   bool has_unordered_start_ts;
   bool has_gpu_query_reset;

   /* Recording the commands of an acceleration-structure build
    * (hk_sync_barrier()). */
   bool as_build;

   /* Timestamps written as copies of an earlier end stamp
    * (hk_sync_write_timestamp()), and the latest stream on each subqueue
    * whose stamp they read. They are recorded in one launch in a
    * timestamp-only compute stream ordered after those streams: the next
    * timestamp carrier that follows them anyway, or a command of their own
    * before the next query command or event write, at the end of the
    * command buffer, or when the batch is full
    * (hk_sync_flush_stamp_copies()). */
   struct hk_stamp_copy stamp_copies[HK_SYNC_STAMP_COPIES];
   uint8_t stamp_copy_count;
   struct hk_cs *stamp_copy_after[HK_SYNC_NUM_SQ];

   /* Source kinds of the host-availability dependencies recorded and not yet
    * flushed (hk_sync_defer_host_flush()). Submission flushes them once,
    * after this submission's work (hk_sync_host_flush_prefix()). */
   uint8_t host_flush_kinds;

   /* Tile hazards: attachment footprints of the open pass instance and
    * the latest earlier render stream it shares a tile with. The command
    * buffer keeps the recorded instances (hk_cmd_buffer::sync_pass_records). */
   struct hk_sync_footprint pass_fp[HK_SYNC_MAX_PASS_FOOTPRINTS];
   uint8_t pass_fp_count;
   struct hk_cs *pass_tile_wait[HK_SYNC_NUM_SQ];
   /* The latest earlier render that only the fragment half waits for: it
    * shares only whole tiles with the instance, which loads them
    * (hk_sync_queue::load_hazard_frag). */
   struct hk_cs *pass_tile_wait_frag;
   /* Attachment producers, including a compute replacement of a clear. */
   struct hk_cs *pass_tile_first, *pass_tile_last;
   /* The last attachment producer of a suspended pass instance, which the
    * resuming instance continues from. */
   struct hk_cs *suspended_last;

   /* Layout transitions the driver performs as work (brl_sync_transition_*):
    * the requirement of the transition being recorded, whether its streams
    * are being recorded, next_seq when they started, and the latest stream
    * of any such transition on each subqueue. brl_transition_carry is
    * hk_sync_stream::brl_carry for its streams. */
   struct hk_sync_dep brl_transition_dep[HK_SYNC_NUM_SQ];
   uint8_t brl_transition_carry;
   bool brl_transition_active;
   uint32_t brl_transition_seq;
   struct hk_cs *brl_transition_last[HK_SYNC_NUM_SQ];

   /* The event wait being recorded (brl_sync_event_poll()): its latest poll
    * stream. */
   struct hk_cs *brl_event_poll;
   /* Per-slot geometry heap tracking (1..HK_HEAP_SLOTS, 0 = none). */
   uint8_t pass_heap_slot;
   struct hk_cs *pass_heap_last[HK_SYNC_NUM_SQ];
   struct hk_cs *last_heap_slot[HK_HEAP_SLOTS][HK_SYNC_NUM_SQ];

   /* Snapshot of hk_device::buffer_alias_mem_gen taken before the first
    * alias-image-only render in this command buffer was exempted from
    * HK_SYNC_GFX_MEM. */
   bool has_alias_image_render;
   uint32_t buffer_alias_mem_gen;
};

/* Per-queue state: kinds of streams ever submitted on each subqueue. A
 * dependency on earlier submits of a kind that never ran is no dependency. */
struct hk_sync_queue {
   uint8_t ever[HK_SYNC_NUM_SQ];
   uint8_t heap_slot_ever[HK_SYNC_NUM_SQ];
   bool exact_prior;
   /* G13/G14 retain upstream's order after both preceding engine prefixes. */
   bool serialize;
   /* Serialized queue that still records dependencies so a render whose
    * vertex half depends on no earlier render can order only its fragment
    * half after the serialized prefix. */
   bool stage_overlap;
   uint64_t submitted[HK_SYNC_NUM_SQ];
   /* Exact priors: every command of the submission being resolved waits
    * through these ordinals (same-queue semaphore waits, hk_gpu_waits.h).
    * Set by the submitter around hk_sync_resolve_submit_ex() and
    * hk_sync_host_flush_prefix(), zero otherwise. */
   uint64_t wait_floor[HK_SYNC_NUM_SQ];
   uint64_t last_absolute[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];
   uint64_t tail[HK_SYNC_NUM_KINDS][HK_SYNC_NUM_SQ];
   uint64_t last_heap_slot[HK_HEAP_SLOTS][HK_SYNC_NUM_SQ];
   /* G17P: the largest whole render prefix of a render submitted without
    * DRM_ASAHI_RENDER_FRAGMENT_BARRIERS (hk_drm_g17p_tiling_ordered()). */
   uint64_t ta_cover;

   /* Tile hazards across submissions: a ring of the footprints recently
    * submitted on this queue. An entry is live until the queue timeline
    * reaches its point. `evicted` is the latest point of an entry the ring
    * dropped, which keeps the check conservative while that point is live. */
   struct hk_sync_recent recent[HK_SYNC_RECENT_FOOTPRINTS];
   uint32_t recent_next, recent_count;
   uint64_t recent_max, evicted;
   /* Complete writes need no preservation query against other complete
    * writes, including entries evicted from the bounded ring: only a
    * preserving footprint shares a tile (footprints_share_tile()). Retain
    * all writes for a future partial-tile or sparse-layer pass. Always set
    * by hk_queue_create(); tests clear it to cover the conservative path.
    */
   bool precise_tile_evictions;
   /* A single command buffer already owns the complete footprint list. */
   bool direct_single_footprints;
   /* The kernel runs a render's fragment work after the previous render's on
    * this queue, partial renders included, so a render that only loads whole
    * tiles an earlier render stores waits for it in its fragment half alone
    * (hk_sync.h, tile hazards). */
   bool load_hazard_frag;
   uint64_t last_preserve, evicted_preserve;
   /* Recent VDM renders' attachment blooms for exact-image HK_SYNC_GFX_IMG
    * narrowing across submissions. */
   struct hk_sync_vdm_att recent_vdm_att[HK_SYNC_RECENT_VDM_ATT];
   uint32_t recent_vdm_att_next, recent_vdm_att_count;
   uint64_t recent_vdm_att_evicted;

   /* Retired-ordinal map for exact priors: each entry pairs a queue timeline
    * point with the submitted VDM/CDM ordinals through that submission. Once
    * `completed >= point`, every ordinal up to `ord[sq]` has retired and can
    * be pruned to 0. */
#define HK_SYNC_RETIRED_RING 64
   struct {
      uint64_t point;
      uint64_t ord[HK_SYNC_NUM_SQ];
   } retired[HK_SYNC_RETIRED_RING];
   uint32_t retired_next, retired_count;
   uint64_t retired_ord[HK_SYNC_NUM_SQ];
};

void hk_sync_reset(struct hk_cmd_buffer *cmd);

/* Stage masks to stream kinds, for a first synchronization scope. */
uint8_t hk_sync_kinds_src(VkPipelineStageFlags2 stages);
/* And for a second synchronization scope. */
uint8_t hk_sync_kinds_dst(VkPipelineStageFlags2 stages);

/* Adds the scopes of one image barrier to src and dst, as
 * hk_sync_dependency_kinds() counts them. */
void brl_sync_image_barrier_kinds(const VkImageMemoryBarrier2 *barrier,
                                  bool image_class, uint8_t *src,
                                  uint8_t *dst);

/*
 * Layout transitions the driver performs as work. Vulkan orders an image
 * layout transition after the first synchronization scope of its barrier and
 * before the second (Vulkan 1.4, 7.7.4 "Image Layout Transitions"), and in
 * submission order with every other layout transition on the queue. A
 * transition the driver records as streams (brl_discard_init_record()) follows
 * exactly that:
 *
 *    brl_sync_transition_kinds()   for each image barrier it performs
 *    brl_sync_transition_prepare() before the dependency's hk_sync_barrier(),
 *                                  so other barriers of the dependency do not
 *                                  widen the first scope
 *    brl_sync_transition_begin()   streams created from here on wait on that
 *                                  requirement alone, not on what the recorded
 *                                  dependencies ask of transfer-class work,
 *                                  in this command buffer or, at submission,
 *                                  in earlier ones (hk_sync_stream::brl_carry)
 *    brl_sync_transition_end()     later work of the second scope waits on the
 *                                  transition's streams, nothing else does
 *
 * The streams are of the transfer class, so a later first scope naming the
 * transfer stages also covers them.
 */

/* Adds the scopes one image barrier gives the transition it performs. A queue
 * family acquire ignores its source stages unless the dependency has
 * VK_DEPENDENCY_QUEUE_FAMILY_OWNERSHIP_TRANSFER_USE_ALL_STAGES_BIT_KHR: the
 * release and the semaphore between them order it. An automatic layout
 * transition of a render pass (render_pass: the common runtime's
 * runtime_rp_barrier) is ordered by the subpass dependencies, which arrive as
 * the dependency's memory barriers while the image barrier names no source
 * stage; their scopes are added too. */
void brl_sync_transition_kinds(const VkDependencyInfo *info,
                               const VkImageMemoryBarrier2 *barrier,
                               bool image_class, bool acquire,
                               bool render_pass, uint8_t *src, uint8_t *dst);

/* The event wait being recorded: transitions it performs also follow its poll
 * stream cs (G17P, hk_event_wait_polls()). brl_sync_event_done() ends the
 * wait. */
void brl_sync_event_poll(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
void brl_sync_event_done(struct hk_cmd_buffer *cmd);

void brl_sync_transition_prepare(struct hk_cmd_buffer *cmd, uint8_t src_kinds);
void brl_sync_transition_begin(struct hk_cmd_buffer *cmd);
void brl_sync_transition_end(struct hk_cmd_buffer *cmd, uint8_t dst_kinds);

/* Union of the scopes of every barrier in a dependency. With image_class,
 * an image barrier whose second scope names graphics work names
 * HK_SYNC_GFX_FRAG | HK_SYNC_GFX_IMG instead of HK_SYNC_GFX. */
void hk_sync_dependency_kinds(const VkDependencyInfo *info, bool image_class,
                              uint8_t *src, uint8_t *dst);

/* Whether a dependency performs a device-to-host memory domain operation: a
 * barrier names a host access, or MEMORY_READ or MEMORY_WRITE with the host
 * stage in its destination scope. */
bool hk_sync_dependency_has_host_access(const VkDependencyInfo *info);

/* The first synchronization scopes, as kinds, of the barriers of a dependency
 * that perform a device-to-host domain operation. Zero when none does, or
 * when their first scopes hold no device work. */
uint8_t hk_sync_host_flush_kinds(const VkDependencyInfo *info);

/* Defer the host availability a dependency asks for to the submission:
 * nothing is recorded, so a barrier-only command buffer stays without
 * streams. */
void hk_sync_defer_host_flush(struct hk_cmd_buffer *cmd,
                              const VkDependencyInfo *info);

/* The host availability a pipeline barrier asks for: the open compute stream
 * to record the memory-scoped flush in, when that stream holds work and
 * already follows the dependency's first scope (it then also waits for what
 * that scope names), or NULL, with the flush deferred to the submission
 * (hk_sync_defer_host_flush()) or none needed. */
struct hk_cs *hk_sync_host_flush(struct hk_cmd_buffer *cmd,
                                 const VkDependencyInfo *info);

/* Take the deferred host-availability kinds, for an operation the host can
 * observe before the submission completes (an event write). */
uint8_t hk_sync_take_host_flush(struct hk_cmd_buffer *cmd);

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

/* A compute stream for a query command, ordered after earlier query work. */
struct hk_cs *hk_query_command_cs(struct hk_cmd_buffer *cmd);

/* The render stream draws with a pre-rasterization shader that reads images:
 * its vertex half joins the image-barrier requirements (HK_SYNC_GFX_IMG). */
void hk_sync_vertex_images(struct hk_cmd_buffer *cmd, struct hk_cs *cs);
void hk_sync_vertex_images_bloom(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                                 uint64_t bloom);
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
   ((cmd)->sync.kind_override ? (cmd)->sync.kind_override                      \
    : ((cmd)->sync.xfer_depth || (cmd)->sync.brl_transition_active)            \
       ? HK_SYNC_XFER                                                          \
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
void hk_sync_end_cs_if_gfx_waits(struct hk_cmd_buffer *cmd);

/* Timestamps. */
void hk_sync_write_timestamp(struct hk_cmd_buffer *cmd,
                             VkPipelineStageFlags2 stage,
                             struct agx_timestamp_req req);
bool hk_sync_stamp_event(struct hk_cmd_buffer *cmd,
                         struct agx_timestamp_req req, uint8_t src_kinds);
void hk_sync_flush_lead_timestamp(struct hk_cmd_buffer *cmd);
void hk_sync_flush_pending_timestamp(struct hk_cmd_buffer *cmd);

/* Record the pending timestamp copies (hk_sync_state::stamp_copies) in a
 * timestamp-only compute stream positioned after the streams they read, and
 * end it. Every later first scope that names HK_SYNC_STAMP orders after
 * them; no application dependency does. */
void hk_sync_flush_stamp_copies(struct hk_cmd_buffer *cmd);

/* The compute stream a timestamp query reset records into: after every
 * earlier stream but other such resets (HK_SYNC_QRESET_SCOPE). Record with
 * hk_sync_state::kind_override set to HK_SYNC_QRESET, then call
 * hk_sync_query_reset_done() with the stream that holds the reset. */
struct hk_cs *hk_query_reset_cs(struct hk_cmd_buffer *cmd);
void hk_sync_query_reset_done(struct hk_cmd_buffer *cmd, struct hk_cs *cs);

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
void hk_sync_heap_slot_used(struct hk_cmd_buffer *cmd, uint8_t slot);

/* Tile hazards: a written attachment of the pass instance being begun, then
 * the end of the list, before the instance creates its first stream. */
void hk_sync_pass_footprint(struct hk_cmd_buffer *cmd, const void *image,
                            uint32_t level, uint32_t layer0, uint32_t layer1,
                            uint32_t level_width, uint32_t level_height,
                            VkRect2D area, bool preserve_layers,
                            bool preserve_load);
void hk_sync_pass_footprints_done(struct hk_cmd_buffer *cmd);

/* hk_device::load_hazard_frag, false without a device (unit fixtures). */
bool hk_sync_load_hazard_frag(const struct hk_device *dev);
void hk_sync_replace_render(struct hk_cmd_buffer *cmd, struct hk_cs *from,
                            struct hk_cs *into);
void hk_sync_apply_dep(struct hk_cmd_buffer *cmd, struct hk_cs *into,
                       const struct hk_sync_dep dep[HK_SYNC_NUM_SQ]);
/* Whether a submission records tile-hazard footprints, and whether the
 * queue may still be running one, so the caller fetches the completed
 * timeline point only when it can matter. */
bool hk_sync_submit_needs_completed(const struct hk_queue *queue,
                                    struct hk_cmd_buffer *const *cmdbufs,
                                    unsigned count);
bool hk_sync_submit_has_unretired_priors(const struct hk_queue *queue);

/* The driver's own resolve pass after a render: waits on the render. */
void hk_sync_resolve_begin(struct hk_cmd_buffer *cmd);
void hk_sync_resolve_end(struct hk_cmd_buffer *cmd);

struct hk_image;
/* Fold a trailing vkCmdResolveImage2 into cs's end-of-tile store_and_resolve:
 * applies pending transfer waits to cs's fragment half, records dst_image's
 * tile footprint and attachment bloom, and marks cs as transfer work for
 * subsequent dependencies. */
void hk_sync_fold_resolve(struct hk_cmd_buffer *cmd, struct hk_cs *cs,
                          const struct hk_image *dst_image, uint32_t level,
                          uint32_t layer, uint32_t width, uint32_t height);

/* Transfer entry points bracket their recording so the streams they create
 * belong to the transfer stage class. Defined in hk_cmd_buffer.h. */

/* Submission: resolve every stream's barrier fields for one ioctl payload.
 * cmdbufs are the command buffers in submission order; first_cdm_is_flush
 * says the payload opens with a driver-added compute command that every
 * command must run after; chain_first says the commands must transitively
 * follow the first hardware command (the kernel attaches the ioctl's input
 * syncs only to it).
 *
 * A command buffer may occur more than once in cmdbufs, and its streams then
 * resolve differently at each occurrence. Unless out is NULL, it receives
 * one entry per stream occurrence in submission order; the caller loads each
 * into its stream (hk_sync_load_resolved()) before emitting it. The streams
 * keep the fields of their last occurrence.
 */
bool hk_sync_resolve_submit_ex(struct hk_queue *queue,
                               struct hk_cmd_buffer *const *cmdbufs,
                               unsigned count, bool first_cdm_is_flush,
                               bool chain_first, uint64_t completed,
                               struct hk_sync_resolved *out,
                               bool *tail_raised_out);
bool hk_sync_resolve_submit(struct hk_queue *queue,
                            struct hk_cmd_buffer *const *cmdbufs,
                            unsigned count, bool first_cdm_is_flush,
                            bool chain_first, uint64_t completed,
                            struct hk_sync_resolved *out);

/* The kinds whose priors on subqueue `sq` hk_sync_resolve_submit() resolves
 * to a wait, at the first command buffer of the next submission on `q`: a
 * prior of any other kind is no wait. Each command buffer adds
 * hk_sync_cmd_kinds() for those after it in the submission. A driver-added
 * flush opening the payload (first_cdm_is_flush) is not counted. */
uint8_t hk_sync_queue_kinds(const struct hk_sync_queue *q, unsigned sq);

/* The barrier fields of the compute command that ends a submission with the
 * host flush its command buffers deferred (hk_sync_defer_host_flush()),
 * called after hk_sync_resolve_submit(). With exact priors it follows the
 * latest command of each of `kinds` and every requirement still carried, on
 * both subqueues and earlier submissions included, and the submission's last
 * command on each subqueue it has commands on, as legacy prefixes do; with
 * none there, legacy prefixes follow every earlier command. A flush that
 * trails render-dependent compute work then names a render too, so the
 * kernel takes it behind a render dependency like that work instead of as a
 * command with no dependency. base holds the
 * ordinals before the submission, count its commands on each subqueue. With
 * exact priors it also takes the flush's ordinal; false if that overflows.
 * The flush is no later command's source: no kind names it. */
bool hk_sync_host_flush_prefix(struct hk_sync_queue *q, uint8_t kinds,
                               const uint64_t base[HK_SYNC_NUM_SQ],
                               const uint32_t count[HK_SYNC_NUM_SQ],
                               uint16_t barrier[HK_SYNC_NUM_SQ],
                               uint64_t prior[HK_SYNC_NUM_SQ]);
uint8_t hk_sync_cmd_kinds(const struct hk_cmd_buffer *cmd, unsigned sq);
void hk_sync_load_resolved(struct hk_cs *cs, const struct hk_sync_resolved *r);

bool hk_sync_can_defer_unordered_start_ts(struct hk_cmd_buffer *cmd);

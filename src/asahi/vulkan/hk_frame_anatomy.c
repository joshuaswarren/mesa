/*
 * Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 *
 * Opt-in frame anatomy profiler. See hk_frame_anatomy.h.
 *
 * Timestamps ride on the kernel's native per-command timestamp writes
 * (drm_asahi_cmd_render::ts_vtx/ts_frag, drm_asahi_cmd_compute::ts), which
 * the firmware fills from its pass records; the profiler owns a dedicated
 * timestamp object and only claims slots no query pool already uses. Results
 * are read back lazily, under a non-blocking syncobj query, at the next
 * sampled submit, and with a bounded wait at device destruction.
 *
 * The uapi in include/drm-uapi/asahi_drm.h exposes no TVB overflow or
 * partial-render counter; the kernel keeps the firmware's per-command
 * tvb_overflow_count and its TVB growth messages to itself (TVBStats debug
 * flag, dmesg). So no such indicator is logged here; "pass" shows only the
 * userspace-visible split of an API render pass into several streams.
 */
#include "hk_frame_anatomy.h"
#include "nir.h"

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>

#include "agx_bo.h"
#include "agx_chip.h"
#include "agx_device.h"
#include "agx_pack.h"
#include "asahi/libagx/query.h"
#include "drm-uapi/asahi_drm.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_queue.h"
#include "util/format/u_format.h"
#include "util/hash_table.h"
#include "util/log.h"
#include "util/macros.h"
#include "util/os_misc.h"
#include "util/os_time.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "util/u_dynarray.h"

#define HK_FA_TEXT_MAX       1024
#define HK_FA_STAMPS         4
#define HK_FA_TS_BO_SIZE     (256 * 1024)
#define HK_FA_RING_RECS      (HK_FA_TS_BO_SIZE / (HK_FA_STAMPS * sizeof(uint64_t)))
#define HK_FA_FINISH_WAIT_NS (2 * 1000000000ll)

enum hk_fa_ts_mode {
   HK_FA_TS_ALL,
   HK_FA_TS_EDGES,
   HK_FA_TS_NONE,
};

struct hk_frame_anatomy_rec {
   uint32_t syncobj;
   uint64_t point;
   uint32_t slot;        /* ring record index, UINT32_MAX when none was free */
   uint32_t header_off;  /* payload offset of the drm_asahi_cmd_header */
   uint8_t own;          /* bit i: stamp i belongs to the profiler */
   uint8_t query;        /* bit i: stamp i belongs to a query pool */
   uint32_t query_handle[HK_FA_STAMPS], query_offset[HK_FA_STAMPS];
   uint8_t nr_stamps;    /* 4 render, 2 compute */
   char kind;            /* 'R' render, 'C' compute */
   bool barriers_known;
   uint16_t vdm_barrier, cdm_barrier;
   uint64_t prior_vdm, prior_cdm;
   char text[HK_FA_TEXT_MAX];
};

struct hk_frame_anatomy_submit {
   struct util_dynarray recs; /* struct hk_frame_anatomy_rec */
   uint64_t seq;
   int64_t cpu_ns;
   unsigned command_count;
   unsigned dropped;
};

struct hk_frame_anatomy {
   FILE *file;
   bool own_file;
   simple_mtx_t lock;
   uint64_t every;
   enum hk_fa_ts_mode ts_mode;
   uint64_t submit_seq;
   uint64_t sampled;

   struct agx_bo *ts_bo;
   uint32_t ts_handle;
   uint64_t *ts_map;

   /* Records in flight, FIFO; their slots form a contiguous ring span. */
   uint32_t ring_head, ring_count;
   struct util_dynarray pending; /* struct hk_frame_anatomy_rec */

   /* Free-form notes written so far and dropped at the limit
    * (hk_frame_anatomy_note); the limit is HK_FRAME_ANATOMY_NOTES. */
   uint32_t notes, notes_dropped, note_limit;

   /* HK_FRAME_ANATOMY_DUMP=<dir>: each sampled fragment program's binary is
    * written once to <dir>/fs-<hash>.bin. */
   const char *dump_dir;
   uint32_t dumped[1024];
   unsigned nr_dumped;

   bool merge_notes;

   /* Bound timestamp query pools (hk_frame_anatomy_bind_pool), so a stamp a
    * query owns can still be read back once its command retired. */
   struct util_dynarray pools; /* struct hk_fa_pool */
};

struct hk_fa_pool {
   uint32_t handle;
   const uint64_t *map;
   uint64_t size;
};

/* A query pool's copy of stamp i, 0 if the pool is gone. Called locked. */
static uint64_t
hk_fa_query_stamp(struct hk_frame_anatomy *fa,
                  const struct hk_frame_anatomy_rec *rec, unsigned i)
{
   util_dynarray_foreach(&fa->pools, struct hk_fa_pool, pool) {
      if (pool->handle == rec->query_handle[i] &&
          rec->query_offset[i] + sizeof(uint64_t) <= pool->size)
         return p_atomic_read(&pool->map[rec->query_offset[i] / 8]);
   }
   return 0;
}

void
hk_frame_anatomy_note(struct hk_device *dev, const char *fmt, ...)
{
   struct hk_frame_anatomy *fa = dev->frame_anatomy;
   if (likely(!fa))
      return;

   /* Render pass continuation notes arrive for nearly every pass; they are
    * written only with HK_FRAME_ANATOMY_MERGE_NOTES so other notes fit. */
   if (!fa->merge_notes && (!strncmp(fmt, "pass-merge", 10) ||
                            !strncmp(fmt, "MSAA pass merge", 15)))
      return;

   simple_mtx_lock(&fa->lock);
   if (fa->notes < fa->note_limit) {
      va_list ap;
      va_start(ap, fmt);
      fputs("N ", fa->file);
      vfprintf(fa->file, fmt, ap);
      va_end(ap);
      if (++fa->notes == fa->note_limit)
         fputs(" (note limit reached)", fa->file);
      fputc('\n', fa->file);
      fflush(fa->file);
   } else {
      fa->notes_dropped++;
   }
   simple_mtx_unlock(&fa->lock);
}

uint8_t
hk_frame_anatomy_encode_op(bool present, unsigned hw_op,
                           VkAttachmentLoadOp load, VkAttachmentStoreOp store)
{
   unsigned l = load == VK_ATTACHMENT_LOAD_OP_LOAD        ? 0
                : load == VK_ATTACHMENT_LOAD_OP_CLEAR     ? 1
                : load == VK_ATTACHMENT_LOAD_OP_DONT_CARE ? 2
                                                          : 3;
   unsigned s = store == VK_ATTACHMENT_STORE_OP_STORE       ? 0
                : store == VK_ATTACHMENT_STORE_OP_DONT_CARE ? 1
                                                            : 2;
   return (present ? HK_FA_OP_PRESENT : 0) | (hw_op & 3) | (l << 2) | (s << 4);
}

static inline uint64_t
hk_fa_ns(struct hk_device *dev, uint64_t stamp)
{
   return stamp ? agx_gpu_timestamp_to_ns(&dev->dev, stamp) : 0;
}

static void
hk_fa_emit(struct hk_frame_anatomy *fa, struct hk_device *dev,
           const struct hk_frame_anatomy_rec *rec, bool complete)
{
   uint64_t t[HK_FA_STAMPS] = {0};
   uint8_t recovered = 0;
   if (rec->slot != UINT32_MAX) {
      for (unsigned i = 0; i < HK_FA_STAMPS; ++i) {
         if (rec->own & BITFIELD_BIT(i))
            t[i] = hk_fa_ns(dev, fa->ts_map[rec->slot * HK_FA_STAMPS + i]);
      }
   }

   /* A stamp a query owns is the same retirement time the kernel writes for
    * the query. The application may already have reset or reused the slot,
    * so keep it only when it lies within a second after the stamp before.
    */
   for (unsigned i = 0; complete && i < rec->nr_stamps; ++i) {
      if (!(rec->query & BITFIELD_BIT(i)))
         continue;
      uint64_t v = hk_fa_ns(dev, hk_fa_query_stamp(fa, rec, i));
      uint64_t prev = 0;
      for (unsigned j = 0; j < i; ++j)
         prev = t[j] ? t[j] : prev;
      if (v && prev && v >= prev && v - prev < 1000000000ull) {
         t[i] = v;
         recovered |= BITFIELD_BIT(i);
      }
   }

   fputs(rec->text, fa->file);

   /* Barriers exactly as submitted: NONE prints as '-', else the relative
    * command index inside the ioctl; exact-prior ordinals when nonzero. */
   if (rec->barriers_known) {
      fputs(" bar=", fa->file);
      if (rec->vdm_barrier == DRM_ASAHI_BARRIER_NONE)
         fputs("v-", fa->file);
      else
         fprintf(fa->file, "v%u", rec->vdm_barrier);
      if (rec->cdm_barrier == DRM_ASAHI_BARRIER_NONE)
         fputs(",c-", fa->file);
      else
         fprintf(fa->file, ",c%u", rec->cdm_barrier);
      if (rec->prior_vdm || rec->prior_cdm)
         fprintf(fa->file, " prior=v%" PRIu64 ",c%" PRIu64, rec->prior_vdm,
                 rec->prior_cdm);
   } else {
      fputs(" bar=?", fa->file);
   }

   /* Which stamps the command carries: p profiler, Q query pool (value
    * read back), q query pool (unreadable), - none. */
   fputs(" ts=", fa->file);
   for (unsigned i = 0; i < rec->nr_stamps; ++i) {
      fputc((rec->own & BITFIELD_BIT(i))       ? 'p'
            : (recovered & BITFIELD_BIT(i))    ? 'Q'
            : (rec->query & BITFIELD_BIT(i))   ? 'q'
                                               : '-',
            fa->file);
   }

   if (rec->kind == 'R') {
      /* t0 vertex start, t1 vertex end, t2 fragment start, t3 fragment end */
      fprintf(fa->file, " ts_ns=%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64,
              t[0], t[1], t[2], t[3]);
      if (t[0] && t[1])
         fprintf(fa->file, " vtx_us=%.1f", (t[1] - t[0]) / 1000.0);
      if (t[1] && t[2])
         fprintf(fa->file, " gap_us=%.1f",
                 ((int64_t)t[2] - (int64_t)t[1]) / 1000.0);
      if (t[2] && t[3])
         fprintf(fa->file, " frg_us=%.1f", (t[3] - t[2]) / 1000.0);
      if (t[0] && t[3])
         fprintf(fa->file, " total_us=%.1f", (t[3] - t[0]) / 1000.0);
   } else {
      fprintf(fa->file, " ts_ns=%" PRIu64 ",%" PRIu64, t[0], t[1]);
      if (t[0] && t[1])
         fprintf(fa->file, " dur_us=%.1f", (t[1] - t[0]) / 1000.0);
   }

   if (rec->slot == UINT32_MAX)
      fputs(" (no timestamp slot)", fa->file);
   else if (!rec->point)
      fputs(" (submit failed)", fa->file);
   else if (!complete)
      fputs(" (not retired at device destruction)", fa->file);
   fputc('\n', fa->file);
}

/* Retire completed records in FIFO order. With wait (device teardown only),
 * block up to HK_FA_FINISH_WAIT_NS per queue point and then emit every
 * record, flagging those whose point never completed. */
static void
hk_fa_flush(struct hk_frame_anatomy *fa, struct hk_device *dev, bool wait)
{
   const unsigned n =
      util_dynarray_num_elements(&fa->pending, struct hk_frame_anatomy_rec);
   struct hk_frame_anatomy_rec *recs = fa->pending.data;
   uint32_t cached_syncobj = 0;
   uint64_t cached_value = 0;
   bool have_cached = false;
   unsigned done = 0;

   for (unsigned i = 0; i < n; ++i) {
      const struct hk_frame_anatomy_rec *rec = &recs[i];
      bool complete = true;
      if (rec->point) {
         if (!have_cached || cached_syncobj != rec->syncobj ||
             cached_value < rec->point) {
            uint32_t syncobj = rec->syncobj;
            if (wait) {
               uint64_t point = rec->point;
               drmSyncobjTimelineWait(dev->dev.fd, &syncobj, &point, 1,
                                      os_time_get_nano() + HK_FA_FINISH_WAIT_NS,
                                      0, NULL);
            }
            uint64_t value = 0;
            if (drmSyncobjQuery(dev->dev.fd, &syncobj, &value, 1))
               value = 0;
            cached_syncobj = rec->syncobj;
            cached_value = value;
            have_cached = true;
         }
         if (cached_value < rec->point) {
            if (!wait)
               break;
            complete = false;
         }
      }
      hk_fa_emit(fa, dev, rec, complete);
      done++;
   }

   if (done) {
      memmove(recs, recs + done, (n - done) * sizeof(*recs));
      fa->pending.size -= done * sizeof(*recs);
      /* Every record holds exactly one ring slot. */
      fa->ring_head = (fa->ring_head + done) % HK_FA_RING_RECS;
      fa->ring_count -= done;
      fflush(fa->file);
   }
}

/* Caller holds fa->lock. */
static bool
hk_fa_alloc_slot(struct hk_frame_anatomy *fa, uint32_t *slot)
{
   if (fa->ring_count >= HK_FA_RING_RECS)
      return false;
   *slot = (fa->ring_head + fa->ring_count) % HK_FA_RING_RECS;
   fa->ring_count++;
   memset(&fa->ts_map[*slot * HK_FA_STAMPS], 0, HK_FA_STAMPS * sizeof(uint64_t));
   return true;
}

void
hk_frame_anatomy_init(struct hk_device *dev)
{
   dev->frame_anatomy = NULL;

   const char *path = os_get_option("HK_FRAME_ANATOMY");
   if (!path || !*path)
      return;

   struct hk_frame_anatomy *fa = calloc(1, sizeof(*fa));
   if (!fa)
      return;

   int64_t every = debug_get_num_option("HK_FRAME_ANATOMY_EVERY", 200);
   fa->every = every > 0 ? (uint64_t)every : 1;
   int64_t notes = debug_get_num_option("HK_FRAME_ANATOMY_NOTES",
                                        HK_FA_NOTE_LIMIT);
   fa->note_limit = notes > 0 ? (uint32_t)MIN2(notes, INT32_MAX) : 0;

   fa->merge_notes = debug_get_bool_option("HK_FRAME_ANATOMY_MERGE_NOTES", false);
   fa->dump_dir = os_get_option("HK_FRAME_ANATOMY_DUMP");
   if (fa->dump_dir && !*fa->dump_dir)
      fa->dump_dir = NULL;

   const char *ts = os_get_option("HK_FRAME_ANATOMY_TS");
   fa->ts_mode = HK_FA_TS_ALL;
   if (ts && !strcmp(ts, "edges"))
      fa->ts_mode = HK_FA_TS_EDGES;
   else if (ts && !strcmp(ts, "none"))
      fa->ts_mode = HK_FA_TS_NONE;
   else if (ts && *ts && strcmp(ts, "all"))
      mesa_logw("HK_FRAME_ANATOMY_TS=%s unknown; using all", ts);

   if (!strcmp(path, "-") || !strcmp(path, "stderr")) {
      fa->file = stderr;
   } else {
      fa->file = fopen(path, "a");
      fa->own_file = true;
   }
   if (!fa->file) {
      mesa_logw("HK_FRAME_ANATOMY: cannot open %s: %s", path, strerror(errno));
      free(fa);
      return;
   }

   /* The kernel requires timestamp buffers to be SHARED (hk_query_pool.c). */
   fa->ts_bo = agx_bo_create(&dev->dev, HK_FA_TS_BO_SIZE, 0,
                             AGX_BO_WRITEBACK | AGX_BO_SHARED,
                             "Frame anatomy timestamps");
   if (fa->ts_bo)
      fa->ts_map = agx_bo_map(fa->ts_bo);
   if (!fa->ts_map ||
       agx_bind_timestamps(&dev->dev, fa->ts_bo, &fa->ts_handle) ||
       !fa->ts_handle) {
      mesa_logw("HK_FRAME_ANATOMY: timestamp buffer setup failed; disabled");
      if (fa->ts_bo)
         agx_bo_unreference(&dev->dev, fa->ts_bo);
      if (fa->own_file)
         fclose(fa->file);
      free(fa);
      return;
   }
   memset(fa->ts_map, 0, HK_FA_TS_BO_SIZE);

   simple_mtx_init(&fa->lock, mtx_plain);
   util_dynarray_init(&fa->pending, NULL);
   util_dynarray_init(&fa->pools, NULL);

   static const char *const mode_names[] = {"all", "edges", "none"};
   fprintf(fa->file,
           "# hk frame anatomy pid=%d chip=%s(0x%x) ts_hz=%" PRIu64
           " every=%" PRIu64 " ts=%s ring_recs=%u features=0x%" PRIx64 "\n",
           (int)getpid(), agx_chip_name(dev->dev.chip), dev->dev.chip,
           (uint64_t)dev->dev.params.command_timestamp_frequency_hz,
           fa->every, mode_names[fa->ts_mode], (unsigned)HK_FA_RING_RECS,
           (uint64_t)dev->dev.params.features);
   if (fa->ts_mode == HK_FA_TS_ALL)
      fprintf(fa->file,
              "# ts=all: cross-engine steps of sampled submits wait on the "
              "host, so cross-engine gaps and durations in sampled submits are "
              "inflated; use HK_FRAME_ANATOMY_TS=edges or none for timing\n");
   fprintf(fa->file,
           "# S: sampled submit. R: render stream, C: compute stream; i: "
           "hardware command index in the submit.\n"
           "# rtN=<format>:<bg><eot>:<api load><api store>:<c|u>[/spill]  bg: "
           "-none C clear L load; eot: -none S store; api load: L C D N; api "
           "store: S D N; c compressed u uncompressed; '?' unknown\n"
           "# z/s=<fmt>:<L|-><S|->:<c|u>  (ZLS control as submitted)  "
           "pass=<B|.><E|.> B: stream begins the API pass, E: ends it; '..' "
           "is a continuation; xcb=<n>: n command buffers packed into this "
           "render at submission; rp=<H|-><T|->: the recording's head and tail "
           "for that packing, rp_head/rp_tail: why not, rp_next: why its tail "
           "did not pack with the next command buffer's head\n"
           "# bar=v<n>,c<n>: header vdm/cdm barrier as submitted ('-' = NONE, "
           "n = relative command count in the ioctl); prior=: exact-prior "
           "ordinals\n"
           "# ts=: per stamp p profiler / q query pool / - none; ts_ns: render "
           "vtx start,vtx end,frag start,frag end; compute start,end; 0 = not "
           "written\n"
           "# no TVB/partial-render indicator exists in the asahi uapi\n");
   fflush(fa->file);

   dev->frame_anatomy = fa;
}

void
hk_frame_anatomy_finish(struct hk_device *dev)
{
   struct hk_frame_anatomy *fa = dev->frame_anatomy;
   if (!fa)
      return;
   dev->frame_anatomy = NULL;

   simple_mtx_lock(&fa->lock);
   hk_fa_flush(fa, dev, true);
   const unsigned left =
      util_dynarray_num_elements(&fa->pending, struct hk_frame_anatomy_rec);
   fprintf(fa->file,
           "# end submits=%" PRIu64 " sampled=%" PRIu64
           " unretired=%u notes=%u dropped=%u\n",
           fa->submit_seq, fa->sampled, left, fa->notes, fa->notes_dropped);
   fflush(fa->file);
   simple_mtx_unlock(&fa->lock);

   if (dev->dev.ops.bo_unbind_object)
      dev->dev.ops.bo_unbind_object(&dev->dev, fa->ts_handle);
   agx_bo_unreference(&dev->dev, fa->ts_bo);
   util_dynarray_fini(&fa->pending);
   util_dynarray_fini(&fa->pools);
   simple_mtx_destroy(&fa->lock);
   if (fa->own_file)
      fclose(fa->file);
   free(fa);
}

struct hk_frame_anatomy_submit *
hk_frame_anatomy_begin_submit(struct hk_device *dev, struct hk_queue *queue,
                              unsigned command_count)
{
   struct hk_frame_anatomy *fa = dev->frame_anatomy;

   simple_mtx_lock(&fa->lock);
   const uint64_t seq = ++fa->submit_seq;
   const bool sample = ((seq - 1) % fa->every) == 0;
   if (sample) {
      fa->sampled++;
      hk_fa_flush(fa, dev, false);
   }
   simple_mtx_unlock(&fa->lock);

   if (!sample)
      return NULL;

   struct hk_frame_anatomy_submit *s = calloc(1, sizeof(*s));
   if (!s)
      return NULL;
   util_dynarray_init(&s->recs, NULL);
   s->seq = seq;
   s->cpu_ns = os_time_get_nano();
   s->command_count = command_count;
   return s;
}

/* Every record takes a ring slot (so FIFO retirement stays one slot per
 * record); only the stamps actually claimed are ever read. */
static struct hk_frame_anatomy_rec *
hk_fa_new_rec(struct hk_frame_anatomy_submit *s, struct hk_device *dev,
              char kind, size_t header_offset)
{
   struct hk_frame_anatomy *fa = dev->frame_anatomy;
   struct hk_frame_anatomy_rec *rec =
      util_dynarray_grow(&s->recs, struct hk_frame_anatomy_rec, 1);
   if (!rec)
      return NULL;
   memset(rec, 0, sizeof(*rec));
   rec->kind = kind;
   rec->nr_stamps = kind == 'R' ? 4 : 2;
   rec->slot = UINT32_MAX;
   rec->header_off = header_offset;

   simple_mtx_lock(&fa->lock);
   if (!hk_fa_alloc_slot(fa, &rec->slot)) {
      rec->slot = UINT32_MAX;
      s->dropped++;
   }
   simple_mtx_unlock(&fa->lock);
   return rec;
}

static void
hk_fa_note_query(struct hk_frame_anatomy_rec *rec, unsigned i,
                 const struct drm_asahi_timestamp *ts)
{
   if (ts->handle) {
      rec->query |= BITFIELD_BIT(i);
      rec->query_handle[i] = ts->handle;
      rec->query_offset[i] = ts->offset;
   }
}

/* Claim stamp i of the record for the kernel to write unless a query owns it. */
static void
hk_fa_claim(struct hk_frame_anatomy *fa, struct hk_frame_anatomy_rec *rec,
            unsigned i, struct drm_asahi_timestamp *ts)
{
   if (ts->handle) {
      if (ts->handle != fa->ts_handle)
         hk_fa_note_query(rec, i, ts);
      return;
   }
   if (rec->slot == UINT32_MAX)
      return;
   ts->handle = fa->ts_handle;
   ts->offset = (rec->slot * HK_FA_STAMPS + i) * sizeof(uint64_t);
   rec->own |= BITFIELD_BIT(i);
}

static void
hk_fa_claim_render(struct hk_frame_anatomy *fa,
                   struct hk_frame_anatomy_rec *rec,
                   struct drm_asahi_cmd_render *c)
{
   hk_fa_claim(fa, rec, 0, &c->ts_vtx.start);
   hk_fa_claim(fa, rec, 1, &c->ts_vtx.end);
   hk_fa_claim(fa, rec, 2, &c->ts_frag.start);
   hk_fa_claim(fa, rec, 3, &c->ts_frag.end);
}

static void
hk_fa_claim_compute(struct hk_frame_anatomy *fa,
                    struct hk_frame_anatomy_rec *rec,
                    struct drm_asahi_cmd_compute *c)
{
   hk_fa_claim(fa, rec, 0, &c->ts.start);
   hk_fa_claim(fa, rec, 1, &c->ts.end);
}

#define HK_FA_APPEND(...)                                                     \
   do {                                                                       \
      if (len < sizeof(rec->text)) {                                          \
         int _r = snprintf(rec->text + len, sizeof(rec->text) - len,          \
                           __VA_ARGS__);                                      \
         if (_r > 0)                                                          \
            len += MIN2((size_t)_r, sizeof(rec->text) - 1 - len);             \
      }                                                                       \
   } while (0)

static char
hk_fa_hw_char(unsigned op)
{
   static const char c[4] = {'-', 'C', 'L', 'S'};
   return c[op & 3];
}

void
hk_frame_anatomy_dump_nir(struct hk_device *dev, const struct hk_shader *shader,
                          struct nir_shader *nir, unsigned stage)
{
   struct hk_frame_anatomy *fa = dev->frame_anatomy;
   if (!fa || !fa->dump_dir || !shader->code_size)
      return;

   /* Named by the same binary hash as the record's fs=, vs= or cs=. */
   const char *prefix = stage == MESA_SHADER_FRAGMENT ? "fs"
                        : stage == MESA_SHADER_COMPUTE ? "cs"
                        : stage == MESA_SHADER_MESH    ? "ms"
                        : stage == MESA_SHADER_VERTEX  ? "vs"
                                                       : NULL;
   if (!prefix)
      return;

   char path[512];
   snprintf(path, sizeof(path), "%s/%s-%08x.nir", fa->dump_dir, prefix,
            _mesa_hash_data(shader->code_ptr, shader->code_size));
   if (access(path, F_OK) == 0)
      return;
   FILE *f = fopen(path, "w");
   if (f) {
      nir_print_shader(nir, f);
      fclose(f);
   }
}

static void
hk_fa_dump_fs(struct hk_frame_anatomy *fa, uint32_t hash,
              const struct hk_shader *fs, const char *prefix)
{
   if (!fa->dump_dir || !fs->code_size)
      return;

   simple_mtx_lock(&fa->lock);
   bool dump = fa->nr_dumped < ARRAY_SIZE(fa->dumped);
   for (unsigned i = 0; dump && i < fa->nr_dumped; ++i)
      dump = fa->dumped[i] != hash;
   if (dump)
      fa->dumped[fa->nr_dumped++] = hash;
   simple_mtx_unlock(&fa->lock);
   if (!dump)
      return;

   char path[512];
   snprintf(path, sizeof(path), "%s/%s-%08x.bin", fa->dump_dir, prefix, hash);
   FILE *f = fopen(path, "wb");
   if (f) {
      fwrite(fs->code_ptr, 1, fs->code_size, f);
      fclose(f);
   }
}

void
hk_frame_anatomy_render(struct hk_frame_anatomy_submit *s,
                        struct hk_device *dev, const struct hk_cs *cs,
                        struct drm_asahi_cmd_render *c, unsigned cmdbuf_index,
                        unsigned command_index, size_t header_offset)
{
   struct hk_frame_anatomy *fa = dev->frame_anatomy;
   struct hk_frame_anatomy_rec *rec = hk_fa_new_rec(s, dev, 'R', header_offset);
   if (!rec)
      return;

   if (fa->ts_mode == HK_FA_TS_ALL) {
      hk_fa_claim_render(fa, rec, c);
   } else {
      hk_fa_note_query(rec, 0, &c->ts_vtx.start);
      hk_fa_note_query(rec, 1, &c->ts_vtx.end);
      hk_fa_note_query(rec, 2, &c->ts_frag.start);
      hk_fa_note_query(rec, 3, &c->ts_frag.end);
   }

   size_t len = 0;
   HK_FA_APPEND("R s=%" PRIu64 " cb=%u i=%u fb=%ux%ux%u ms=%u tile=%ux%u "
                "tib=%uB draws=%u api=%u pass=%c%c flags=0x%x",
                s->seq, cmdbuf_index, command_index, c->width_px,
                c->height_px, c->layers, c->samples, c->utile_width_px,
                c->utile_height_px, c->sample_size_B, cs->stats.cmds,
                cs->stats.calls, cs->g17p_pass_begins ? 'B' : '.',
                cs->g17p_pass_ends ? 'E' : '.', c->flags);
   if (cs->render_pack_members > 1) {
      HK_FA_APPEND(" xcb=%u", cs->render_pack_members);
   } else if (cs->cmd) {
      /* Submit-time render packing edges of the recording (hk_render_pack.h). */
      const struct hk_render_pack_edges *e = &cs->cmd->render_pack;
      HK_FA_APPEND(" rp=%c%c", e->head ? 'H' : '-', e->tail ? 'T' : '-');
      if (!e->head && e->head_why)
         HK_FA_APPEND(" rp_head=\"%s\"", e->head_why);
      if (!e->tail && e->tail_why)
         HK_FA_APPEND(" rp_tail=\"%s\"", e->tail_why);
      if (e->tail && e->next_why)
         HK_FA_APPEND(" rp_next=\"%s%s\"",
                      e->next_who == 1   ? "this: "
                      : e->next_who == 2 ? "next: "
                                         : "",
                      e->next_why);
   }

   /* Depth/stencil exactly as the ZLS unit is programmed for this command. */
   agx_unpack(NULL, &cs->cr.zls_control, ZLS_CONTROL, zls);
   if (cs->cr.depth.buffer) {
      HK_FA_APPEND(" z=%s:%c%c:%c",
                   zls.z_format == AGX_ZLS_FORMAT_32F ? "z32f" : "z16",
                   zls.z_load ? 'L' : '-', zls.z_store ? 'S' : '-',
                   cs->cr.depth.meta ? 'c' : 'u');
   }
   if (cs->cr.stencil.buffer) {
      HK_FA_APPEND(" s=s8:%c%c:%c", zls.s_load ? 'L' : '-',
                   zls.s_store ? 'S' : '-', cs->cr.stencil.meta ? 'c' : 'u');
   }
   if (cs->cr.depth.buffer || cs->cr.stencil.buffer)
      HK_FA_APPEND(" zls=%ux%u", cs->cr.zls_width, cs->cr.zls_height);

   for (unsigned rt = 0; rt < HK_MAX_RTS; ++rt) {
      const enum pipe_format fmt = cs->tib.logical_format[rt];
      const uint8_t bg = cs->cr.bg.main.anatomy_ops[rt];
      const uint8_t eot = cs->cr.eot.main.anatomy_ops[rt];
      if (fmt == PIPE_FORMAT_NONE && !(bg & HK_FA_OP_PRESENT))
         continue;

      static const char load_c[4] = {'L', 'C', 'D', 'N'};
      static const char store_c[4] = {'S', 'D', 'N', '?'};
      HK_FA_APPEND(" rt%u=%s:", rt,
                   fmt == PIPE_FORMAT_NONE ? "none" : util_format_short_name(fmt));
      if (bg & HK_FA_OP_PRESENT) {
         HK_FA_APPEND("%c%c:%c%c", hk_fa_hw_char(HK_FA_OP_HW(bg)),
                      hk_fa_hw_char(HK_FA_OP_HW(eot)),
                      load_c[HK_FA_OP_LOAD(bg)], store_c[HK_FA_OP_STORE(bg)]);
      } else {
         HK_FA_APPEND("??:??");
      }
      HK_FA_APPEND(":%c%s",
                   (cs->cr.color_compressed_mask & BITFIELD_BIT(rt)) ? 'c' : 'u',
                   cs->tib.spilled[rt] ? "/spill" : "");
   }

   if (cs->anatomy_area.extent.width)
      HK_FA_APPEND(" area=%ux%u+%d+%d", cs->anatomy_area.extent.width,
                   cs->anatomy_area.extent.height, cs->anatomy_area.offset.x,
                   cs->anatomy_area.offset.y);
   HK_FA_APPEND(" pet=%u scratch=%c%c", cs->cr.process_empty_tiles ? 1 : 0,
                (c->flags & DRM_ASAHI_RENDER_VERTEX_SCRATCH) ? 'v' : '-',
                c->fragment_helper.data ? 'f' : '-');
   if (cs->g17p_primitive_bound != UINT64_MAX)
      HK_FA_APPEND(" prims<=%" PRIu64, cs->g17p_primitive_bound);

   /* Fragment programs: hash of the main binary, draws, then the compiler's
    * instruction count, registers, threads, spills+fills, scratch bytes and
    * sample-rate shading. */
   for (unsigned i = 0; i < HK_FA_FS_MAX && cs->anatomy_fs[i].fs; ++i) {
      const struct hk_shader *fs = cs->anatomy_fs[i].fs;
      const struct agx2_stats *st = &fs->b.info.stats;
      const uint32_t hash =
         fs->code_size ? _mesa_hash_data(fs->code_ptr, fs->code_size) : 0;
      hk_fa_dump_fs(fa, hash, fs, "fs");
      HK_FA_APPEND(" fs=%08x:d%u:i%u:g%u:t%u:sp%u:sc%u:ss%u", hash,
                   cs->anatomy_fs[i].draws, st->instrs, st->gprs, st->threads,
                   st->spills + st->fills, st->scratch,
                   fs->info.fs.uses_sample_shading);
   }
   if (cs->anatomy_fs_other)
      HK_FA_APPEND(" fs_other=%u", cs->anatomy_fs_other);
   /* Vertex programs: hash, draws, instructions, registers, threads,
    * spills+fills. */
   for (unsigned i = 0; i < HK_FA_FS_MAX && cs->anatomy_vs[i].vs; ++i) {
      const struct hk_shader *vs = cs->anatomy_vs[i].vs;
      const struct agx2_stats *st = &vs->b.info.stats;
      const uint32_t hash =
         vs->code_size ? _mesa_hash_data(vs->code_ptr, vs->code_size) : 0;
      hk_fa_dump_fs(fa, hash, vs, "vs");
      HK_FA_APPEND(" vs=%08x:d%u:i%u:g%u:t%u:sp%u", hash,
                   cs->anatomy_vs[i].draws, st->instrs, st->gprs, st->threads,
                   st->spills + st->fills);
   }
   if (cs->anatomy_vs_other)
      HK_FA_APPEND(" vs_other=%u", cs->anatomy_vs_other);
   if (cs->anatomy_covered)
      HK_FA_APPEND(" cov=%x", cs->anatomy_covered);
   if (cs->anatomy_republish)
      HK_FA_APPEND(" st=r%u:v%u:n%u:p%u", cs->anatomy_republish,
                   cs->anatomy_vdm, cs->anatomy_vdm_new, cs->anatomy_ppp);
   if (cs->anatomy_pass[0] | cs->anatomy_pass[1] | cs->anatomy_pass[2] |
       cs->anatomy_pass[3])
      HK_FA_APPEND(" pt=o%u:t%u:p%u:tp%u:nm%u:zw%u:lz%u", cs->anatomy_pass[0],
                   cs->anatomy_pass[1], cs->anatomy_pass[2],
                   cs->anatomy_pass[3], cs->anatomy_nomerge,
                   cs->anatomy_zwrite, cs->anatomy_late_z);
   if (cs->anatomy_api_barriers)
      HK_FA_APPEND(" apibar=%u:%" PRIx64 ":%" PRIx64 ":%" PRIx64 ":%" PRIx64,
                   cs->anatomy_api_barriers, cs->anatomy_api_src,
                   cs->anatomy_api_dst, cs->anatomy_api_src_access,
                   cs->anatomy_api_dst_access);
   if (cs->anatomy_api_barriers)
      HK_FA_APPEND(" apibar_n=m%u:b%u:i%u", cs->anatomy_api_mem,
                   cs->anatomy_api_buf, cs->anatomy_api_img);

   /* The vertex (tiling) half's own requirement, resolved for this ioctl
    * like bar=/prior=, the stream's sync kinds and whether they differ. */
   HK_FA_APPEND(" tak=%x/%x", cs->sync.dep_ta[HK_SYNC_VDM].prior,
                cs->sync.dep_ta[HK_SYNC_CDM].prior);
   HK_FA_APPEND(" ta=%c:v%d,c%d:pv%" PRIu64 ",pc%" PRIu64 " kind=0x%x",
                cs->sync.ta_split ? 'S' : '=',
                cs->sync.barrier_ta[HK_SYNC_VDM] == DRM_ASAHI_BARRIER_NONE
                   ? -1 : (int)cs->sync.barrier_ta[HK_SYNC_VDM],
                cs->sync.barrier_ta[HK_SYNC_CDM] == DRM_ASAHI_BARRIER_NONE
                   ? -1 : (int)cs->sync.barrier_ta[HK_SYNC_CDM],
                cs->sync.prior_ta[HK_SYNC_VDM], cs->sync.prior_ta[HK_SYNC_CDM],
                cs->sync.kind);
}

void
hk_frame_anatomy_note_fs(struct hk_cs *cs, const struct hk_shader *fs)
{
   if (!fs)
      return;

   /* An inline blend can replace the API fragment main with a monolithic
    * program owning its constant-program layout. Counting the API main here
    * reports only its export copies and hides the blend, registers and
    * scratch actually launched. The linked shader owns this layout for the
    * same lifetime as the API shader, and only devices that launch it set
    * it, as the uniform upload (hk_cmd_buffer.c) assumes. Compute streams
    * keep recording their compute shader. This hook is reached only with
    * frame anatomy enabled.
    */
   if (cs->type == HK_CS_VDM) {
      const struct hk_linked_shader *linked =
         cs->cmd->state.gfx.linked[MESA_SHADER_FRAGMENT];
      if (linked && linked->g17p_layout)
         fs = linked->g17p_layout;
   }

   for (unsigned i = 0; i < HK_FA_FS_MAX; ++i) {
      if (cs->anatomy_fs[i].fs == fs || !cs->anatomy_fs[i].fs) {
         cs->anatomy_fs[i].fs = fs;
         cs->anatomy_fs[i].draws++;
         return;
      }
   }
   cs->anatomy_fs_other++;
}

void
hk_frame_anatomy_note_grid(struct hk_cs *cs, const struct hk_shader *shader,
                           uint64_t threads, uint64_t groups, bool indirect)
{
   for (unsigned i = 0; i < HK_FA_FS_MAX; ++i) {
      if (cs->anatomy_fs[i].fs == shader) {
         if (indirect)
            cs->anatomy_fs[i].indirect++;
         cs->anatomy_fs[i].threads += threads;
         cs->anatomy_fs[i].groups += groups;
         return;
      }
   }
}

void
hk_frame_anatomy_note_m5_fragment(struct hk_cs *cs)
{
   if (!cs->m5_ppp_emitted_valid)
      return;
   const struct agx_ppp_hal300_graphics_fields *f = &cs->m5_ppp_emitted;
   cs->anatomy_pass[f->fragment_pass & 3]++;
   cs->anatomy_nomerge += f->disable_tri_merging;
   cs->anatomy_zwrite += f->depth_active && f->depth_write;
   cs->anatomy_late_z += f->fragment_depth_mode != 3;
}

void
hk_frame_anatomy_note_vs(struct hk_cs *cs, const struct hk_shader *vs)
{
   if (!vs)
      return;
   for (unsigned i = 0; i < HK_FA_FS_MAX; ++i) {
      if (cs->anatomy_vs[i].vs == vs || !cs->anatomy_vs[i].vs) {
         cs->anatomy_vs[i].vs = vs;
         cs->anatomy_vs[i].draws++;
         return;
      }
   }
   cs->anatomy_vs_other++;
}

void
hk_frame_anatomy_note_precomp(struct hk_cs *cs, unsigned program,
                              struct agx_grid grid)
{
   const uint64_t threads =
      agx_is_indirect(grid)
         ? UINT64_MAX
         : (uint64_t)grid.count[0] * grid.count[1] * grid.count[2];
   for (unsigned i = 0; i < HK_FA_FS_MAX; ++i) {
      if (!cs->anatomy_precomp[i].dispatches ||
          cs->anatomy_precomp[i].program == program) {
         cs->anatomy_precomp[i].program = program;
         cs->anatomy_precomp[i].dispatches++;
         if (threads == UINT64_MAX ||
             cs->anatomy_precomp[i].threads == UINT64_MAX)
            cs->anatomy_precomp[i].threads = UINT64_MAX;
         else
            cs->anatomy_precomp[i].threads += threads;
         return;
      }
   }
   cs->anatomy_precomp_other++;
}

void
hk_frame_anatomy_compute(struct hk_frame_anatomy_submit *s,
                         struct hk_device *dev, const struct hk_cs *cs,
                         struct drm_asahi_cmd_compute *c, unsigned cmdbuf_index,
                         unsigned command_index, size_t header_offset)
{
   struct hk_frame_anatomy *fa = dev->frame_anatomy;
   struct hk_frame_anatomy_rec *rec = hk_fa_new_rec(s, dev, 'C', header_offset);
   if (!rec)
      return;

   if (fa->ts_mode == HK_FA_TS_ALL) {
      hk_fa_claim_compute(fa, rec, c);
   } else {
      hk_fa_note_query(rec, 0, &c->ts.start);
      hk_fa_note_query(rec, 1, &c->ts.end);
   }

   size_t len = 0;
   HK_FA_APPEND("C s=%" PRIu64 " cb=%u i=%u dispatches=%u api=%u flushes=%u "
                "merged=%u imm=%u rq=%u matrix=%u scratch=%c",
                s->seq, cmdbuf_index, command_index, cs->stats.cmds,
                cs->stats.calls, cs->stats.flushes, cs->stats.merged,
                (unsigned)util_dynarray_num_elements(&cs->imm_writes,
                                                     struct libagx_imm_write),
                cs->gen1_native_rq ? 1 : 0,
                cs->m5_matrix_ops ? 1 : 0, c->helper.data ? 'c' : '-');
   /* Compute programs, as for fragment programs in render records. */
   for (unsigned i = 0; i < HK_FA_FS_MAX && cs->anatomy_fs[i].fs; ++i) {
      const struct hk_shader *prog = cs->anatomy_fs[i].fs;
      const struct agx2_stats *st = &prog->b.info.stats;
      const uint32_t hash = prog->code_size
         ? _mesa_hash_data(prog->code_ptr, prog->code_size) : 0;
      hk_fa_dump_fs(fa, hash, prog, "cs");
      HK_FA_APPEND(" cs=%08x:d%u:i%u:g%u:t%u:sp%u:sc%u:n%" PRIu64 ":w%" PRIu64
                   ":x%u", hash,
                   cs->anatomy_fs[i].draws, st->instrs, st->gprs, st->threads,
                   st->spills + st->fills, st->scratch,
                   cs->anatomy_fs[i].threads, cs->anatomy_fs[i].groups,
                   cs->anatomy_fs[i].indirect);
   }
   if (cs->anatomy_fs_other)
      HK_FA_APPEND(" cs_other=%u", cs->anatomy_fs_other);
   if (cs->anatomy_api_barriers)
      HK_FA_APPEND(" apibar=%u:%" PRIx64 ":%" PRIx64 ":%" PRIx64 ":%" PRIx64,
                   cs->anatomy_api_barriers, cs->anatomy_api_src,
                   cs->anatomy_api_dst, cs->anatomy_api_src_access,
                   cs->anatomy_api_dst_access);
   if (cs->anatomy_api_barriers)
      HK_FA_APPEND(" apibar_n=m%u:b%u:i%u", cs->anatomy_api_mem,
                   cs->anatomy_api_buf, cs->anatomy_api_img);
   for (unsigned i = 0; i < HK_FA_FS_MAX && cs->anatomy_precomp[i].dispatches;
        ++i) {
      if (cs->anatomy_precomp[i].threads == UINT64_MAX)
         HK_FA_APPEND(" pre=%u:d%u:indirect", cs->anatomy_precomp[i].program,
                      cs->anatomy_precomp[i].dispatches);
      else
         HK_FA_APPEND(" pre=%u:d%u:t%" PRIu64, cs->anatomy_precomp[i].program,
                      cs->anatomy_precomp[i].dispatches,
                      cs->anatomy_precomp[i].threads);
   }
   if (cs->anatomy_precomp_other)
      HK_FA_APPEND(" pre_other=%u", cs->anatomy_precomp_other);
}

/* The command struct a record's header introduces, if the payload holds it. */
static void *
hk_fa_payload_cmd(const struct hk_frame_anatomy_rec *rec, void *payload,
                  size_t payload_size, struct drm_asahi_cmd_header **hdr)
{
   if ((size_t)rec->header_off + sizeof(**hdr) > payload_size)
      return NULL;
   struct drm_asahi_cmd_header *h =
      (struct drm_asahi_cmd_header *)((uint8_t *)payload + rec->header_off);
   const uint16_t expect =
      rec->kind == 'R' ? DRM_ASAHI_CMD_RENDER : DRM_ASAHI_CMD_COMPUTE;
   if (h->cmd_type != expect ||
       (size_t)rec->header_off + sizeof(*h) + h->size > payload_size)
      return NULL;
   const size_t need = rec->kind == 'R' ? sizeof(struct drm_asahi_cmd_render)
                                        : sizeof(struct drm_asahi_cmd_compute);
   if (h->size < need)
      return NULL;
   if (hdr)
      *hdr = h;
   return h + 1;
}

void
hk_frame_anatomy_prepare(struct hk_frame_anatomy_submit *s,
                         struct hk_device *dev, void *payload,
                         size_t payload_size)
{
   struct hk_frame_anatomy *fa = dev->frame_anatomy;
   if (fa->ts_mode != HK_FA_TS_EDGES)
      return;

   const unsigned n =
      util_dynarray_num_elements(&s->recs, struct hk_frame_anatomy_rec);
   if (!n)
      return;

   /* First and last recorded hardware command of the submit (one command
    * when n == 1). Claiming happens in the payload copy the kernel reads. */
   const unsigned edges[2] = {0, n - 1};
   for (unsigned e = 0; e < (n == 1 ? 1u : 2u); ++e) {
      struct hk_frame_anatomy_rec *rec =
         util_dynarray_element(&s->recs, struct hk_frame_anatomy_rec, edges[e]);
      void *cmd = hk_fa_payload_cmd(rec, payload, payload_size, NULL);
      if (!cmd)
         continue;
      if (rec->kind == 'R')
         hk_fa_claim_render(fa, rec, cmd);
      else
         hk_fa_claim_compute(fa, rec, cmd);
   }
}

void
hk_frame_anatomy_note_barriers(struct hk_frame_anatomy_submit *s,
                               const void *payload, size_t begin, size_t end)
{
   util_dynarray_foreach(&s->recs, struct hk_frame_anatomy_rec, rec) {
      if (rec->header_off < begin || rec->header_off >= end)
         continue;
      struct drm_asahi_cmd_header *h = NULL;
      const void *cmd = hk_fa_payload_cmd(rec, (void *)payload, end, &h);
      if (!cmd)
         continue;
      rec->vdm_barrier = h->vdm_barrier;
      rec->cdm_barrier = h->cdm_barrier;
      rec->barriers_known = true;
      rec->prior_vdm = rec->prior_cdm = 0;
      if (rec->kind == 'R' &&
          h->size >= sizeof(struct drm_asahi_cmd_render_with_prior)) {
         const struct drm_asahi_cmd_render_with_prior *r = cmd;
         rec->prior_vdm = r->prior_vdm;
         rec->prior_cdm = r->prior_cdm;
      } else if (rec->kind == 'C' &&
                 h->size >= offsetof(struct drm_asahi_cmd_compute_with_prior,
                                     prior_cdm) +
                               sizeof(uint64_t)) {
         const struct drm_asahi_cmd_compute_with_prior *k = cmd;
         rec->prior_vdm = k->prior_vdm;
         rec->prior_cdm = k->prior_cdm;
      }
   }
}

void
hk_frame_anatomy_end_submit(struct hk_frame_anatomy_submit *s,
                            struct hk_device *dev, struct hk_queue *queue,
                            VkResult result)
{
   struct hk_frame_anatomy *fa = dev->frame_anatomy;
   const unsigned n =
      util_dynarray_num_elements(&s->recs, struct hk_frame_anatomy_rec);

   simple_mtx_lock(&fa->lock);
   fprintf(fa->file,
           "S s=%" PRIu64 " q=%u pt=%" PRIu64 " cmds=%u recs=%u dropped=%u "
           "cpu_ns=%" PRId64 " result=%d\n",
           s->seq, queue->drm.syncobj, queue->drm.timeline_value,
           s->command_count, n, s->dropped, s->cpu_ns, (int)result);

   util_dynarray_foreach(&s->recs, struct hk_frame_anatomy_rec, rec) {
      rec->syncobj = queue->drm.syncobj;
      rec->point = result == VK_SUCCESS ? queue->drm.timeline_value : 0;
      util_dynarray_append(&fa->pending, *rec);
   }
   if (result != VK_SUCCESS)
      hk_fa_flush(fa, dev, false);
   simple_mtx_unlock(&fa->lock);

   util_dynarray_fini(&s->recs);
   free(s);
}

void
hk_frame_anatomy_bind_pool(struct hk_device *dev, uint32_t handle,
                           const void *map, uint64_t size)
{
   struct hk_frame_anatomy *fa = dev->frame_anatomy;
   if (!fa || !handle || !map)
      return;
   struct hk_fa_pool pool = {.handle = handle, .map = map, .size = size};
   simple_mtx_lock(&fa->lock);
   util_dynarray_append(&fa->pools, pool);
   simple_mtx_unlock(&fa->lock);
}

void
hk_frame_anatomy_unbind_pool(struct hk_device *dev, uint32_t handle)
{
   struct hk_frame_anatomy *fa = dev->frame_anatomy;
   if (!fa || !handle)
      return;
   simple_mtx_lock(&fa->lock);
   util_dynarray_foreach(&fa->pools, struct hk_fa_pool, pool) {
      if (pool->handle == handle) {
         *pool = util_dynarray_pop(&fa->pools, struct hk_fa_pool);
         break;
      }
   }
   simple_mtx_unlock(&fa->lock);
}

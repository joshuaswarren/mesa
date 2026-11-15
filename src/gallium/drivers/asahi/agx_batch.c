/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2019-2020 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <xf86drm.h>
#include "asahi/lib/agx_device_virtio.h"
#include "asahi/lib/agx_va_runtime_hole.h"
#include "asahi/lib/decode.h"
#include "drm-uapi/dma-buf.h"
#include "poly/tessellator.h"
#include "util/bitset.h"
#include "util/libsync.h"
#include "util/u_dynarray.h"
#include "util/u_range.h"
#include "agx_device.h"
#include "agx_state.h"
#include "vdrm.h"

#define foreach_active(ctx, idx)                                               \
   BITSET_FOREACH_SET(idx, ctx->batches.active, AGX_MAX_BATCHES)

/* Intermediate and final G18 submissions carry their own dispatch metadata. */
static void
agx_append_split_compute(struct util_dynarray *cmdbuf,
                         struct drm_asahi_cmd_header header,
                         struct drm_asahi_cmd_compute command, bool g18,
                         const struct drm_asahi_g18_compute_resources *resources)
{
   if (g18) {
      /* The complete compute command (its prior, SCS, auxiliary and
       * intersection fields zero) plus the suffix. */
      struct drm_asahi_cmd_compute_g18 extended = {
         .base.legacy = command,
         .resources = *resources,
      };
      STATIC_ASSERT(offsetof(struct drm_asahi_cmd_compute_g18, resources) ==
                    sizeof(struct drm_asahi_cmd_compute_with_prior));
      extended.base.legacy.flags |= DRM_ASAHI_COMPUTE_G18_RESOURCES;
      header.size = sizeof(extended);
      if (agx_env_flag(AGX_ENV_G18_RESOURCE_TRACE)) {
         const uint8_t *bytes = (const void *)&extended;
         fprintf(stderr, "G18_COMPUTE_RESOURCE_PAYLOAD ");
         for (unsigned i = 0; i < sizeof(extended); ++i)
            fprintf(stderr, "%02x", bytes[i]);
         fprintf(stderr, "\n");
      }
      util_dynarray_append(cmdbuf, header);
      util_dynarray_append(cmdbuf, extended);
   } else {
      util_dynarray_append(cmdbuf, header);
      util_dynarray_append(cmdbuf, command);
   }
}

#define foreach_submitted(ctx, idx)                                            \
   BITSET_FOREACH_SET(idx, ctx->batches.submitted, AGX_MAX_BATCHES)

#define batch_debug(batch, fmt, ...)                                           \
   do {                                                                        \
      if (unlikely(agx_device(batch->ctx->base.screen)->debug &                \
                   AGX_DBG_BATCH))                                             \
         agx_msg("[Queue %u Batch %u] " fmt "\n", batch->ctx->queue_id,        \
                 agx_batch_idx(batch), ##__VA_ARGS__);                         \
   } while (0)

/* The J873 BO-ownership trace; other chips keep agx3c's batch log. */
#define g18_batch_debug(batch, fmt, ...)                                       \
   do {                                                                        \
      if (agx_device(batch->ctx->base.screen)->chip == AGX_CHIP_G18G)          \
         batch_debug(batch, fmt, ##__VA_ARGS__);                               \
   } while (0)

bool
agx_batch_is_active(struct agx_batch *batch)
{
   return BITSET_TEST(batch->ctx->batches.active, agx_batch_idx(batch));
}

bool
agx_batch_is_submitted(struct agx_batch *batch)
{
   return BITSET_TEST(batch->ctx->batches.submitted, agx_batch_idx(batch));
}

static void
agx_batch_mark_active(struct agx_batch *batch)
{
   unsigned batch_idx = agx_batch_idx(batch);

   batch_debug(batch, "ACTIVE");

   assert(!BITSET_TEST(batch->ctx->batches.submitted, batch_idx));
   assert(!BITSET_TEST(batch->ctx->batches.active, batch_idx));
   BITSET_SET(batch->ctx->batches.active, batch_idx);
}

static void
agx_batch_mark_submitted(struct agx_batch *batch)
{
   unsigned batch_idx = agx_batch_idx(batch);

   batch_debug(batch, "SUBMIT");

   assert(BITSET_TEST(batch->ctx->batches.active, batch_idx));
   assert(!BITSET_TEST(batch->ctx->batches.submitted, batch_idx));
   BITSET_CLEAR(batch->ctx->batches.active, batch_idx);
   BITSET_SET(batch->ctx->batches.submitted, batch_idx);
}

static void
agx_batch_mark_complete(struct agx_batch *batch)
{
   unsigned batch_idx = agx_batch_idx(batch);

   batch_debug(batch, "COMPLETE");

   assert(!BITSET_TEST(batch->ctx->batches.active, batch_idx));
   assert(BITSET_TEST(batch->ctx->batches.submitted, batch_idx));
   BITSET_CLEAR(batch->ctx->batches.submitted, batch_idx);
}

struct agx_encoder
agx_encoder_allocate(struct agx_batch *batch, struct agx_device *dev)
{
   struct agx_bo *bo = agx_bo_create(dev, 0x80000, 0, 0, "Encoder");
   uint8_t *map = agx_bo_map(bo);
   return (struct agx_encoder){.bo = bo, .current = map, .end = map + bo->size};
}

struct agx_g17p_vdm_alias {
   struct agx_bo *bo;
   uint64_t address;
};

uint64_t
agx_batch_get_g17p_vdm_alias(struct agx_batch *batch, struct agx_bo *bo,
                             uint64_t address)
{
   struct agx_device *dev = agx_device(batch->ctx->base.screen);
   if (!agx_native_graphics(dev))
      return address;

   if (!bo || address < bo->va->addr || address - bo->va->addr >= bo->size)
      return 0;

   uint64_t alias = 0;
   util_dynarray_foreach(&batch->g17p_vdm_aliases,
                         struct agx_g17p_vdm_alias, entry) {
      if (entry->bo == bo) {
         alias = entry->address;
         break;
      }
   }

   if (!alias) {
      alias = agx_bo_get_g17p_vdm_alias(dev, bo);
      if (!alias)
         return 0;

      const struct agx_g17p_vdm_alias entry = {bo, alias};
      util_dynarray_append(&batch->g17p_vdm_aliases, entry);
   }

   return alias + (address - bo->va->addr);
}

static void
agx_encoder_bind_vdm_alias(struct agx_device *dev, struct agx_encoder *encoder)
{
   if (!agx_native_graphics(dev))
      return;

   encoder->g17p_vdm_alias = agx_bo_get_g17p_vdm_alias(dev, encoder->bo);
   assert(encoder->g17p_vdm_alias);
}

static void
agx_encoder_cleanup(struct agx_device *dev, struct agx_encoder *encoder)
{
   agx_bo_unreference(dev, encoder->bo);
}

/*
 * M5 program storage on the G17 kernel runtime (agx_m5_render_resources).
 *
 * The kernel sizes each render's and dispatch's firmware pools from the
 * programs listed ahead of it in the submission, folding the list with a
 * per-field maximum after evaluating each entry on its own. A list is
 * therefore a set: identical entries add nothing, so keep each distinct one
 * once (a batch of a thousand draws of one program sends one record). The
 * records follow Honeykrisp's: hk_m5_note_render_resources(),
 * hk_cmd_buffer.c's tile record and hk_m5_compute_resources().
 */
void
agx_batch_note_m5_render_program(
   struct agx_batch *batch, const struct drm_asahi_render_program_v2 *program)
{
   util_dynarray_foreach(&batch->m5_render_programs,
                         struct drm_asahi_render_program_v2, entry) {
      if (!memcmp(entry, program, sizeof(*program)))
         return;
   }
   util_dynarray_append(&batch->m5_render_programs, *program);
}

void
agx_batch_note_m5_compute_program(
   struct agx_batch *batch, const struct drm_asahi_compute_program *program)
{
   util_dynarray_foreach(&batch->m5_compute_programs,
                         struct drm_asahi_compute_program, entry) {
      if (!memcmp(entry, program, sizeof(*program)))
         return;
   }
   util_dynarray_append(&batch->m5_compute_programs, *program);
}

/* The tile record: a fragment-stage entry carrying the render's imageblock
 * and the largest storage of its background and end-of-tile programs. The
 * multisample bytes are the native per-sample-count footprint, as
 * hk_cmd_buffer.c computes it. */
void
agx_batch_set_m5_tile_program(struct agx_batch *batch,
                              const struct asahi_bg_eot *const *programs,
                              unsigned count)
{
   const struct agx_tilebuffer_layout *tib = &batch->tilebuffer_layout;
   const uint32_t samples = MAX2(tib->nr_samples, 1);
   const uint32_t sample_bytes = tib->tile_size * tib->sample_size_B;

   batch->m5_tile_program = (struct drm_asahi_render_program){
      .stage = 1,
      .imageblock_bytes = sample_bytes * samples,
      .multisample_bytes = samples == 2   ? sample_bytes / 4
                           : samples == 4 ? sample_bytes
                           : samples == 8 ? 4 * sample_bytes
                                          : 0,
   };
   for (unsigned i = 0; i < count; ++i) {
      batch->m5_tile_program.primary_bytes =
         MAX2(batch->m5_tile_program.primary_bytes,
              programs[i]->primary_storage_B);
      batch->m5_tile_program.secondary_bytes =
         MAX2(batch->m5_tile_program.secondary_bytes,
              programs[i]->staging_storage_B);
   }
}

/* DRM_ASAHI_SET_COMPUTE_RESOURCES for the compute commands that follow, when
 * the kernel takes one. An empty list sends nothing: a dispatch without
 * thread-private storage needs no plan. Returns false if the list cannot be
 * encoded (the header size is 16 bits). */
static bool
agx_append_m5_compute_resources(struct agx_device *dev,
                                struct agx_batch *batch,
                                struct util_dynarray *cmdbuf)
{
   if (!(dev->params.features & DRM_ASAHI_FEATURE_COMPUTE_RESOURCES) ||
       !batch->m5_compute_programs.size)
      return true;
   if (batch->m5_compute_programs.size > UINT16_MAX)
      return false;

   const struct drm_asahi_cmd_header header = {
      .cmd_type = DRM_ASAHI_SET_COMPUTE_RESOURCES,
      .size = batch->m5_compute_programs.size,
      .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
      .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
   };
   util_dynarray_append(cmdbuf, header);
   util_dynarray_append_array(cmdbuf, uint8_t, batch->m5_compute_programs.data,
                              batch->m5_compute_programs.size);
   return true;
}

/* Compute commands for the recorded dispatches first..end-1 of the batch's
 * count, one command each (see agx_batch_submit()). Each waits on the one
 * before it in its submission, the first on the queue's earlier submissions
 * (barrier 0). The batch's compute timestamps bracket all of its commands,
 * across submissions. */
static void
agx_append_m5_compute_commands(struct agx_batch *batch,
                               const struct drm_asahi_cmd_compute *compute,
                               unsigned first, unsigned end, unsigned count,
                               struct util_dynarray *cmdbuf)
{
   const struct agx_cdm_dispatch *cdm_dispatch = batch->cdm_dispatch.data;
   const uint64_t cdm_base = batch->cdm.bo->va->addr;

   for (unsigned i = first; i < end; ++i) {
      struct drm_asahi_cmd_header header = agx_cmd_header(true, 0, i - first);
      struct drm_asahi_cmd_compute one = *compute;
      one.cdm_ctrl_stream_base = cdm_base + cdm_dispatch[i].start;
      one.cdm_ctrl_stream_end = cdm_base + cdm_dispatch[i].end;
      if (i != 0)
         memset(&one.ts.start, 0, sizeof(one.ts.start));
      if (i + 1 != count)
         memset(&one.ts.end, 0, sizeof(one.ts.end));

      util_dynarray_append(cmdbuf, header);
      util_dynarray_append(cmdbuf, one);
   }
}

/* DRM_ASAHI_SET_RENDER_RESOURCES(_V2) for the render that follows: the tile
 * record, then every linked program the render's draws ran. V2 kernels take
 * the 104-byte records through command 7; on others each record is cut to its
 * leading drm_asahi_render_program, which is all a Gallium render fills (no
 * mesh, object or ray-query stage). Returns false if the list cannot be
 * encoded. */
static bool
agx_append_m5_render_resources(struct agx_device *dev, struct agx_batch *batch,
                               struct util_dynarray *cmdbuf)
{
   const bool v2 =
      dev->params.features & DRM_ASAHI_FEATURE_RENDER_RESOURCES_V2;
   const size_t stride = v2 ? sizeof(struct drm_asahi_render_program_v2)
                            : sizeof(struct drm_asahi_render_program);
   const unsigned count =
      1 + util_dynarray_num_elements(&batch->m5_render_programs,
                                     struct drm_asahi_render_program_v2);
   if (count * stride > UINT16_MAX)
      return false;

   const struct drm_asahi_cmd_header header = {
      .cmd_type = v2 ? DRM_ASAHI_SET_RENDER_RESOURCES_V2
                     : DRM_ASAHI_SET_RENDER_RESOURCES,
      .size = count * stride,
      .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
      .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
   };
   util_dynarray_append(cmdbuf, header);

   const struct drm_asahi_render_program_v2 tile = {
      .legacy = batch->m5_tile_program,
   };
   util_dynarray_append_array(cmdbuf, uint8_t, &tile, stride);
   util_dynarray_foreach(&batch->m5_render_programs,
                         struct drm_asahi_render_program_v2, entry) {
      util_dynarray_append_array(cmdbuf, uint8_t, entry, stride);
   }
   return true;
}

static void
agx_batch_init(struct agx_context *ctx,
               const struct pipe_framebuffer_state *key,
               struct agx_batch *batch)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   struct agx_screen *screen = agx_screen(ctx->base.screen);

   batch->ctx = ctx;
   util_copy_framebuffer_state(&batch->key, key);
   /* A framebuffer resolve (a multisampled window-system buffer) keeps its
    * multisample image as well: GL reads it back (glReadPixels, copies,
    * blits) and later batches load it. Reloading the resolved image into
    * every sample instead loses the samples, and a reader of the
    * multisample image saw whatever was there before. Only the DRI seed
    * copy (agx_blit) reloads from the single-sample image. */
   batch->g17p_resolve_reload = false;
   batch->g17p_resolve_preserve = key->resolve != NULL;
   batch->seqnum = ++ctx->batches.seqnum;
   batch->g17p_fragment_program = 0;
   batch->vs_root_valid = false;
   batch->g17p_emrt_heap = 0;
   batch->g17p_fragment_shader = NULL;
   batch->g17p_fragment_linked = NULL;

   agx_bo_reference(screen->rodata);
   agx_pool_init(&batch->pool, dev, "Batch pool", 0, true);
   agx_pool_init(&batch->pipeline_pool, dev, "Batch executable low VA pool",
                 AGX_BO_LOW_VA | AGX_BO_EXEC, true);

   /* Reuse storage across batches; both allocations are context-owned. */
   if (!batch->bo_list.set) {
      batch->bo_list.set = _mesa_pointer_set_create(ctx);
      util_dynarray_init(&batch->bo_list.list, ctx);
   } else {
      _mesa_set_clear(batch->bo_list.set, NULL);
      util_dynarray_clear(&batch->bo_list.list);
   }

   if (agx_batch_is_compute(batch)) {
      batch->cdm = agx_encoder_allocate(batch, dev);
      memset(&batch->vdm, 0, sizeof(batch->vdm));
   } else {
      batch->vdm = agx_encoder_allocate(batch, dev);
      agx_encoder_bind_vdm_alias(dev, &batch->vdm);
      memset(&batch->cdm, 0, sizeof(batch->cdm));
   }

   util_dynarray_init(&batch->cdm_dispatch, ctx);
   util_dynarray_init(&batch->scissor, ctx);
   util_dynarray_init(&batch->depth_bias, ctx);
   util_dynarray_init(&batch->timestamps, ctx);
   util_dynarray_init(&batch->g17p_vdm_aliases, ctx);
   util_dynarray_init(&batch->faulttrace_cmdbuf, ctx);
   util_dynarray_init(&batch->m5_render_programs, ctx);
   util_dynarray_init(&batch->m5_compute_programs, ctx);
   memset(&batch->m5_tile_program, 0, sizeof(batch->m5_tile_program));
   batch->m5_shader_writes = false;
   batch->faulttrace_seqid = 0;

   batch->clear = 0;
   batch->draw = 0;
   batch->load = 0;
   batch->resolve = 0;
   batch->feedback = 0;
   memset(batch->uploaded_clear_color, 0, sizeof(batch->uploaded_clear_color));
   memset(batch->clear_color, 0, sizeof(batch->clear_color));
   batch->clear_depth = 0;
   batch->clear_stencil = 0;
   batch->varyings = 0;
   batch->heap = 0;
   batch->initialized = false;
   batch->native_rejected = false;
   batch->compute_dropped = false;
   batch->draws_refused = 0;
   batch->draws = 0;
   batch->g17g_prims = 0;
   batch->g17g_small_budget = false;
   batch->incoherent_writes = false;
   agx_bo_unreference(dev, batch->sampler_heap.bo);
   batch->sampler_heap.bo = NULL;
   batch->sampler_heap.count = 0;
   batch->vs_scratch = false;
   batch->fs_scratch = false;
   batch->cs_scratch = false;
   memset(&batch->g18, 0, sizeof(batch->g18));
   batch->vs_preamble_scratch = 0;
   batch->fs_preamble_scratch = 0;
   batch->cs_preamble_scratch = 0;

   /* May get read before write, need to initialize to 0 to avoid GPU-side UAF
    * conditions.
    */
   batch->uniforms.tables[AGX_SYSVAL_TABLE_PARAMS] = 0;

   /* We need to emit prim state at the start. Max collides with all. */
   batch->reduced_prim = MESA_PRIM_COUNT;

   if (!batch->syncobj) {
      int ret = drmSyncobjCreate(dev->fd, 0, &batch->syncobj);
      assert(!ret && batch->syncobj);
   }

   agx_batch_mark_active(batch);
}

static struct agx_timestamps *
agx_batch_timestamps(struct agx_batch *batch)
{
   struct agx_timestamps *ts = agx_bo_map(batch->ctx->timestamps);
   return ts + agx_batch_idx(batch);
}

static void
agx_batch_print_stats(struct agx_device *dev, struct agx_batch *batch)
{
   unsigned batch_idx = agx_batch_idx(batch);

   if (u_printf_check_abort(stdout, &dev->printf)) {
      fprintf(stderr, "GPU abort");
      abort();
   }

   if (likely(!(dev->debug & AGX_DBG_STATS)) || !batch->ctx->timestamps)
      return;

   struct agx_timestamps *ts = agx_batch_timestamps(batch);

   if (batch->cdm.bo) {
      float time = (ts->comp_end - ts->comp_start) /
                   (float)dev->params.command_timestamp_frequency_hz;

      mesa_logw("[Batch %d] Compute: %.06f\n", batch_idx, time);
   }

   if (batch->vdm.bo) {
      float time_vtx = (ts->vtx_end - ts->vtx_start) /
                       (float)dev->params.command_timestamp_frequency_hz;
      float time_frag = (ts->frag_end - ts->frag_start) /
                        (float)dev->params.command_timestamp_frequency_hz;
      mesa_logw("[Batch %d] vtx %.06f frag %.06f\n", batch_idx, time_vtx,
                time_frag);
   }
}

static void
agx_batch_cleanup(struct agx_context *ctx, struct agx_batch *batch, bool reset)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   struct agx_screen *screen = agx_screen(ctx->base.screen);

   assert(batch->ctx == ctx);
   assert(!batch->submit_quarantined);
   assert(agx_batch_is_submitted(batch));

   assert(ctx->batch != batch);

   uint64_t begin_ts = ~0, end_ts = 0;
   if (batch->timestamps.size) {
      struct agx_timestamps *ts = agx_batch_timestamps(batch);

      if (batch->cdm.bo && !batch->compute_dropped) {
         begin_ts = MIN2(begin_ts, ts->comp_start);
         end_ts = MAX2(end_ts, ts->comp_end);
      }

      if (batch->vdm.bo) {
         begin_ts = MIN2(begin_ts, ts->vtx_start);
         end_ts = MAX2(end_ts, ts->frag_end);
      }
   }

   agx_finish_batch_queries(batch, begin_ts, end_ts);

   /* Diagnostic CPU observation after retirement distinguishes an empty
    * multisample store from a later texture-fetch or resolve problem. */
   if (!reset && (dev->debug & AGX_DBG_ASYNCTRACE) &&
       batch->key.nr_cbufs == 1 && batch->key.cbufs[0].texture &&
       batch->key.cbufs[0].format == PIPE_FORMAT_R8_UNORM &&
       util_res_sample_count(batch->key.cbufs[0].texture) > 1) {
      struct agx_resource *rt = agx_resource(batch->key.cbufs[0].texture);
      const size_t size = MIN2(rt->layout.size_B, rt->bo->size);
      const uint8_t *map = size <= 4 * 1024 * 1024 ? agx_bo_map(rt->bo) : NULL;
      if (map) {
         size_t nonzero = 0, full = 0;
         size_t first = size, last = 0, quarters[4] = {0}, residues[4] = {0};
         for (size_t i = 0; i < size; ++i) {
            nonzero += map[i] != 0;
            full += map[i] == 0xff;
            if (map[i]) {
               first = MIN2(first, i);
               last = i;
               quarters[MIN2((i * 4) / size, 3)]++;
               residues[i % 4]++;
            }
         }
         agx_msg("MSAA image at retire: base=0x%llx size=0x%zx "
                 "nonzero=%zu full=%zu samples=%u first=0x%zx last=0x%zx "
                 "quarters=%zu,%zu,%zu,%zu residues4=%zu,%zu,%zu,%zu\n",
                 (unsigned long long)agx_map_gpu(rt), size, nonzero, full,
                 util_res_sample_count(&rt->base), first, last,
                 quarters[0], quarters[1], quarters[2], quarters[3],
                 residues[0], residues[1], residues[2], residues[3]);
      }
   }

   if (reset) {
      AGX_BATCH_FOREACH_BO(batch, entry) {
         struct agx_bo *bo = *entry;
         unsigned handle = bo->handle;
         /* We should write no buffers if this is an empty batch */
         assert(agx_writer_get(ctx, handle) != batch);

         agx_bo_unreference(dev, bo);
      }
   } else {
      AGX_BATCH_FOREACH_BO(batch, entry) {
         struct agx_bo *bo = *entry;
         unsigned handle = bo->handle;

         /* There is no more writer on this context for anything we wrote */
         struct agx_batch *writer = agx_writer_get(ctx, handle);

         if (writer == batch) {
            batch_debug(batch, "Releases BO @ 0x%" PRIx64, bo->va->addr);
            agx_writer_remove(ctx, handle);
         }

         p_atomic_cmpxchg(&bo->writer,
                          agx_bo_writer(ctx->queue_id, batch->syncobj), 0);

         agx_bo_unreference(dev, bo);
      }
   }

   agx_bo_unreference(dev, screen->rodata);
   agx_encoder_cleanup(dev, &batch->vdm);
   agx_encoder_cleanup(dev, &batch->cdm);
   agx_pool_cleanup(&batch->pool);
   agx_pool_cleanup(&batch->pipeline_pool);

   util_dynarray_clear(&batch->bo_list.list);
   util_dynarray_fini(&batch->cdm_dispatch);
   util_dynarray_fini(&batch->scissor);
   util_dynarray_fini(&batch->depth_bias);
   util_dynarray_fini(&batch->timestamps);
   util_dynarray_fini(&batch->g17p_vdm_aliases);
   util_dynarray_fini(&batch->faulttrace_cmdbuf);
   util_dynarray_fini(&batch->m5_render_programs);
   util_dynarray_fini(&batch->m5_compute_programs);

   if (!(dev->debug & (AGX_DBG_TRACE | AGX_DBG_SYNC))) {
      agx_batch_print_stats(dev, batch);
   }

   util_unreference_framebuffer_state(&batch->key);
   agx_batch_mark_complete(batch);
}

int
agx_cleanup_batches(struct agx_context *ctx)
{
   struct agx_device *dev = agx_device(ctx->base.screen);

   unsigned i;
   unsigned count = 0;
   struct agx_batch *batches[AGX_MAX_BATCHES];
   uint32_t syncobjs[AGX_MAX_BATCHES];
   uint32_t first = 0;

   foreach_submitted(ctx, i) {
      if (ctx->batches.slots[i].submit_quarantined)
         continue;

      batches[count] = &ctx->batches.slots[i];
      syncobjs[count++] = ctx->batches.slots[i].syncobj;
   }

   if (!count)
      return -1;

   int ret = drmSyncobjWait(dev->fd, syncobjs, count, 0, 0, &first);
   assert(!ret || ret == -ETIME);
   if (ret)
      return -1;

   assert(first < AGX_MAX_BATCHES);
   agx_batch_cleanup(ctx, batches[first], false);
   return agx_batch_idx(batches[first]);
}

static struct agx_batch *
agx_get_batch_for_framebuffer(struct agx_context *ctx,
                              const struct pipe_framebuffer_state *state)
{
   /* Look if we have a matching batch */
   unsigned i;
   foreach_active(ctx, i) {
      struct agx_batch *candidate = &ctx->batches.slots[i];

      if (util_framebuffer_state_equal(&candidate->key, state)) {
         /* We found a match, increase the seqnum for the LRU
          * eviction logic.
          */
         candidate->seqnum = ++ctx->batches.seqnum;
         return candidate;
      }
   }

   /* Look for a free batch */
   for (i = 0; i < AGX_MAX_BATCHES; ++i) {
      if (!BITSET_TEST(ctx->batches.active, i) &&
          !BITSET_TEST(ctx->batches.submitted, i)) {
         struct agx_batch *batch = &ctx->batches.slots[i];
         agx_batch_init(ctx, state, batch);
         return batch;
      }
   }

   /* Try to clean up one batch */
   int freed = agx_cleanup_batches(ctx);
   if (freed >= 0) {
      struct agx_batch *batch = &ctx->batches.slots[freed];
      agx_batch_init(ctx, state, batch);
      return batch;
   }

   /* Else, evict something */
   struct agx_batch *batch = NULL;
   bool submitted = false;
   for (i = 0; i < AGX_MAX_BATCHES; ++i) {
      struct agx_batch *candidate = &ctx->batches.slots[i];
      bool cand_submitted = BITSET_TEST(ctx->batches.submitted, i);
      if (candidate->submit_quarantined)
         continue;

      /* Prefer submitted batches first */
      if (!cand_submitted && submitted)
         continue;

      if (!batch || batch->seqnum > candidate->seqnum) {
         batch = candidate;
         submitted = cand_submitted;
      }
   }
   assert(batch);

   agx_sync_batch_for_reason(ctx, batch, "Too many batches");

   /* Batch is now free */
   agx_batch_init(ctx, state, batch);
   return batch;
}

struct agx_batch *
agx_get_batch(struct agx_context *ctx)
{
   if (!ctx->batch || agx_batch_is_compute(ctx->batch)) {
      ctx->batch = agx_get_batch_for_framebuffer(ctx, &ctx->framebuffer);
      agx_dirty_all(ctx);
   }

   assert(util_framebuffer_state_equal(&ctx->framebuffer, &ctx->batch->key));
   return ctx->batch;
}

struct agx_batch *
agx_get_compute_batch(struct agx_context *ctx)
{
   agx_dirty_all(ctx);

   struct pipe_framebuffer_state key = {.width = AGX_COMPUTE_BATCH_WIDTH};
   ctx->batch = agx_get_batch_for_framebuffer(ctx, &key);
   return ctx->batch;
}

void
agx_flush_all(struct agx_context *ctx, const char *reason)
{
   unsigned idx;
   foreach_active(ctx, idx) {
      if (reason)
         perf_debug_ctx(ctx, "Flushing due to: %s\n", reason);

      agx_flush_batch(ctx, &ctx->batches.slots[idx]);
   }
}

void
agx_flush_batch_for_reason(struct agx_context *ctx, struct agx_batch *batch,
                           const char *reason)
{
   if (reason)
      perf_debug_ctx(ctx, "Flushing due to: %s\n", reason);

   if (agx_batch_is_active(batch))
      agx_flush_batch(ctx, batch);
}

static void
agx_flush_readers_except(struct agx_context *ctx, struct agx_resource *rsrc,
                         struct agx_batch *except, const char *reason,
                         bool sync)
{
   unsigned idx;

   /* Flush everything to the hardware first */
   foreach_active(ctx, idx) {
      struct agx_batch *batch = &ctx->batches.slots[idx];

      if (batch == except)
         continue;

      if (agx_batch_uses_bo(batch, rsrc->bo)) {
         perf_debug_ctx(ctx, "Flush reader due to: %s\n", reason);
         agx_flush_batch(ctx, batch);
      }
   }

   /* Then wait on everything if necessary */
   if (sync) {
      foreach_submitted(ctx, idx) {
         struct agx_batch *batch = &ctx->batches.slots[idx];

         if (batch == except)
            continue;

         if (agx_batch_uses_bo(batch, rsrc->bo)) {
            perf_debug_ctx(ctx, "Sync reader due to: %s\n", reason);
            agx_sync_batch(ctx, batch);
         }
      }
   }
}

static void
agx_flush_writer_except(struct agx_context *ctx, struct agx_resource *rsrc,
                        struct agx_batch *except, const char *reason, bool sync)
{
   struct agx_batch *writer = agx_writer_get(ctx, rsrc->bo->handle);

   if (writer && writer != except &&
       (agx_batch_is_active(writer) || agx_batch_is_submitted(writer))) {
      if (agx_batch_is_active(writer) || sync) {
         perf_debug_ctx(ctx, "%s writer due to: %s\n", sync ? "Sync" : "Flush",
                        reason);
      }
      if (agx_batch_is_active(writer))
         agx_flush_batch(ctx, writer);
      /* Check for submitted state, because if the batch was a no-op it'll
       * already be cleaned up */
      if (sync && agx_batch_is_submitted(writer))
         agx_sync_batch(ctx, writer);
   }
}

bool
agx_any_batch_uses_resource(struct agx_context *ctx, struct agx_resource *rsrc)
{
   unsigned idx;
   foreach_active(ctx, idx) {
      struct agx_batch *batch = &ctx->batches.slots[idx];

      if (agx_batch_uses_bo(batch, rsrc->bo))
         return true;
   }

   foreach_submitted(ctx, idx) {
      struct agx_batch *batch = &ctx->batches.slots[idx];

      if (agx_batch_uses_bo(batch, rsrc->bo))
         return true;
   }

   return false;
}

void
agx_flush_readers(struct agx_context *ctx, struct agx_resource *rsrc,
                  const char *reason)
{
   agx_flush_readers_except(ctx, rsrc, NULL, reason, false);
}

void
agx_sync_readers(struct agx_context *ctx, struct agx_resource *rsrc,
                 const char *reason)
{
   agx_flush_readers_except(ctx, rsrc, NULL, reason, true);
}

void
agx_flush_writer(struct agx_context *ctx, struct agx_resource *rsrc,
                 const char *reason)
{
   agx_flush_writer_except(ctx, rsrc, NULL, reason, false);
}

void
agx_sync_writer(struct agx_context *ctx, struct agx_resource *rsrc,
                const char *reason)
{
   agx_flush_writer_except(ctx, rsrc, NULL, reason, true);
}

void
agx_batch_reads(struct agx_batch *batch, struct agx_resource *rsrc)
{
   agx_batch_add_bo(batch, rsrc->bo);

   if (rsrc->separate_stencil)
      agx_batch_add_bo(batch, rsrc->separate_stencil->bo);

   /* Don't hazard track fake resources internally created for meta */
   if (!rsrc->base.screen)
      return;

   batch_debug(batch, "Reads BO @ 0x%" PRIx64 " (%s)", rsrc->bo->va->addr,
               rsrc->bo->label ? rsrc->bo->label : "unlabelled");

   /* Hazard: read-after-write. A separate stencil plane is its own resource
    * with its own writer (a stencil-only blit renders to it alone), and the
    * BO was just added to this batch, so its writer is a hazard too.
    */
   agx_flush_writer_except(batch->ctx, rsrc, batch, "Read from another batch",
                            false);

   /* A combined depth/stencil read also retains the separate stencil BO.
    * Its writer may differ after a plane-by-plane blit, so resolve that
    * dependency before submitting the reader as well. */
   if (rsrc->separate_stencil)
      agx_flush_writer_except(batch->ctx, rsrc->separate_stencil, batch,
                              "Read stencil from another batch", false);
}

static void
agx_batch_writes_internal(struct agx_batch *batch, struct agx_resource *rsrc,
                          unsigned level)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_batch *writer = agx_writer_get(ctx, rsrc->bo->handle);

   assert(batch->initialized);

   agx_flush_readers_except(ctx, rsrc, batch, "Write from other batch", false);

   BITSET_SET(rsrc->data_valid, level);

   /* Nothing to do if we're already writing */
   if (writer == batch)
      return;

   /* Hazard: writer-after-write, write-after-read */
   if (writer)
      agx_flush_writer(ctx, rsrc, "Multiple writers");

   /* Write is strictly stronger than a read */
   agx_batch_reads(batch, rsrc);

   writer = agx_writer_get(ctx, rsrc->bo->handle);
   assert(!writer || agx_batch_is_submitted(writer));

   /* We are now the new writer. Disregard the previous writer -- anything that
    * needs to wait for the writer going forward needs to wait for us.
    */
   agx_writer_remove(ctx, rsrc->bo->handle);
   agx_writer_add(ctx, agx_batch_idx(batch), rsrc->bo->handle);
   assert(agx_batch_is_active(batch));
   batch_debug(batch, "Becomes writer of BO @ 0x%" PRIx64 " (%s)",
               rsrc->bo->va->addr,
               rsrc->bo->label ? rsrc->bo->label : "unlabelled");
}

void
agx_batch_writes(struct agx_batch *batch, struct agx_resource *rsrc,
                 unsigned level)
{
   agx_batch_writes_internal(batch, rsrc, level);

   if (rsrc->base.target == PIPE_BUFFER) {
      /* Assume BOs written by the GPU are fully valid */
      rsrc->valid_buffer_range.start = 0;
      rsrc->valid_buffer_range.end = ~0;
   }
}

void
agx_batch_writes_range(struct agx_batch *batch, struct agx_resource *rsrc,
                       unsigned offset, unsigned size)
{
   assert(rsrc->base.target == PIPE_BUFFER);
   agx_batch_writes_internal(batch, rsrc, 0);
   util_range_add(&rsrc->base, &rsrc->valid_buffer_range, offset,
                  offset + size);
}

static bool
agx_get_in_sync(struct agx_context *ctx, uint32_t *syncobj)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   *syncobj = 0;

   if (ctx->in_sync_fd >= 0) {
      int ret =
         drmSyncobjImportSyncFile(dev->fd, ctx->in_sync_obj, ctx->in_sync_fd);

      /* A native-fence import can fail under transient kernel allocation
       * pressure. Waiting on the original sync file preserves the acquire
       * ordering without losing the submission or poisoning the context. */
      if (ret && agx_fallback_wait(ctx->in_sync_fd, "acquire import failed")) {
         agx_msg("native acquire fence import and fallback wait failed\n");
         close(ctx->in_sync_fd);
         ctx->in_sync_fd = -1;
         return false;
      }

      close(ctx->in_sync_fd);
      ctx->in_sync_fd = -1;

      if (!ret)
         *syncobj = ctx->in_sync_obj;
   }

   return true;
}

static void
agx_add_sync(struct drm_asahi_sync *syncs, unsigned *count, uint32_t handle)
{
   if (!handle)
      return;

   syncs[(*count)++] = (struct drm_asahi_sync){
      .sync_type = DRM_ASAHI_SYNC_SYNCOBJ,
      .handle = handle,
   };
}

#define MAX_ATTACHMENTS 16

struct attachments {
   struct drm_asahi_attachment list[MAX_ATTACHMENTS];
   size_t count;
};

struct agx_shared_bo_sync {
   struct agx_bo *bo;
   uint32_t in_syncobj;
};

static bool
agx_wait_shared_bo(struct agx_bo *bo)
{
   struct dma_buf_sync sync = {
      .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW,
   };

   if (bo->prime_fd < 0 || drmIoctl(bo->prime_fd, DMA_BUF_IOCTL_SYNC, &sync))
      return false;

   sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;
   return drmIoctl(bo->prime_fd, DMA_BUF_IOCTL_SYNC, &sync) == 0;
}

/* A fresh private output distinguishes publication by this ioctl from a fence
 * left in the reusable batch syncobj by an earlier generation. The supported
 * kernel publishes an aggregate before enqueue, or a completed error fence if
 * no work was accepted. Earlier failures can leave the output without a fence.
 */
static bool
agx_wait_failed_submit(struct agx_device *dev, uint32_t syncobj, bool accepted)
{
   if (!syncobj)
      return true;

   if (!accepted) {
      /* With a valid private handle, count=1, flags=0 and timeout=0,
       * SYNCOBJ_WAIT returns EINVAL only for a missing fence. In particular,
       * do not use sync-file export: it can also return EINVAL on allocation
       * failure even when a fence exists.
       */
      int ret = drmSyncobjWait(dev->fd, &syncobj, 1, 0, 0, NULL);
      if (ret == 0 || ret == -EINVAL)
         return true;
      if (ret != -ETIME)
         return false;
   }

   int ret;
   do {
      ret = drmSyncobjWait(dev->fd, &syncobj, 1, INT64_MAX,
                           DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);
   } while (ret == -EINTR);

   return ret == 0;
}

static void
agx_retire_failed_submit(struct agx_context *ctx, struct agx_batch *batch,
                         struct drm_asahi_sync *syncs,
                         struct agx_shared_bo_sync *shared_bos,
                         unsigned shared_bo_count, void *extres,
                         struct util_dynarray *cmdbuf, int error,
                         uint32_t accepted_syncobj, uint32_t error_syncobj)
{
   struct agx_device *dev = agx_device(ctx->base.screen);

   /* Qualification must stop before the client can queue more work after a
    * recovered submission error. Keep this independent of bounded tracing. */
   const char *stop = getenv("G18_STOP_ON_SUBMIT_ERROR");
   if (dev->chip == AGX_CHIP_G18G && stop && !strcmp(stop, "1")) {
      fprintf(stderr, "G18_SUBMIT_STOP first_submit_error=%d\n", error);
      fflush(stderr);
      _Exit(86);
   }

   for (unsigned i = 0; i < shared_bo_count; ++i) {
      if (shared_bos[i].in_syncobj)
         drmSyncobjDestroy(dev->fd, shared_bos[i].in_syncobj);
   }

   free(syncs);
   free(shared_bos);
   free(extres);
   util_dynarray_fini(cmdbuf);

   /* The kernel refuses a submission outside its admitted profile with
    * ENOTSUPP (524, a value libc does not name). Nothing was queued and no
    * fence will signal, so there is no completion to await and no fault to
    * record: drop the batch and keep the context usable.
    */
   const bool refused = (error == 524);

   /* The G17 runtime validates a whole submission before queueing any of it
    * and refuses one with EINVAL (a lost VM fails with EIO). Earlier split
    * submissions of the batch still run, so their wait below stays. */
   const bool g17 = agx_m5_render_resources(dev);
   const bool g17_refused = g17 && error == EINVAL;

   /* An ioctl error does not cancel accepted commands. Wait for both the
    * failed ioctl's aggregate (if published) and any earlier split compute.
    * Keep both handles alive until completion is established.
    */
   bool complete = refused || agx_wait_failed_submit(dev, error_syncobj, false);
   if (!refused)
      complete &= agx_wait_failed_submit(dev, accepted_syncobj, true);

   /* Never put resources in the BO cache on an inconclusive wait. Keep the
    * entire batch, its fences, context and screen until process exit. This
    * deliberately trades memory on a lost context for avoiding GPU use after
    * reuse, including on context/screen destruction. No command is replayed.
    */
   if (ctx->batch == batch)
      ctx->batch = NULL;
   agx_batch_mark_submitted(batch);
   if (complete) {
      if (accepted_syncobj)
         drmSyncobjDestroy(dev->fd, accepted_syncobj);
      if (error_syncobj)
         drmSyncobjDestroy(dev->fd, error_syncobj);
      /* A G17 refusal signals an error fence into every output, and
       * ctx->syncobj may name this one. A later wait on it would fail as a
       * cancelled dependency, so replace it with a clean signalled fence:
       * the slot's earlier work has completed and this batch ran nothing. */
      if (g17 && (refused || g17_refused))
         drmSyncobjSignal(dev->fd, &batch->syncobj, 1);
      agx_batch_cleanup(ctx, batch, false);
   } else {
      batch->submit_quarantined = true;
      batch->failed_syncobjs[0] = accepted_syncobj;
      batch->failed_syncobjs[1] = error_syncobj;
      ctx->submit_quarantined = true;
      p_atomic_set(&agx_screen(ctx->base.screen)->submit_quarantined, true);
      agx_msg("submit completion unknown; quarantining context resources\n");
   }

   /* A refused batch is lost on its own; the context stays usable. */
   if (!refused && !(g17_refused && complete)) {
      const bool first_fault = !ctx->any_faults;
      ctx->any_faults = true;
      if (first_fault && ctx->reset.reset)
         ctx->reset.reset(ctx->reset.data, PIPE_GUILTY_CONTEXT_RESET);
   }

   errno = error;
}

/* Drop a batch on a context that has faulted, without encoding it. This is
 * the retirement agx_batch_submit uses for the same case: the batch may write
 * BOs, so it needs the full cleanup that releases its writer entries, which
 * agx_batch_reset (for empty batches) does not do.
 */
void
agx_batch_retire_faulted(struct agx_context *ctx, struct agx_batch *batch)
{
   struct util_dynarray cmdbuf = UTIL_DYNARRAY_INIT;

   assert(ctx->any_faults);
   agx_retire_failed_submit(ctx, batch, NULL, NULL, 0, NULL, &cmdbuf, EIO, 0,
                            0);
}

static void
asahi_add_attachment(struct attachments *att, struct agx_resource *rsrc)
{
   assert(att->count < MAX_ATTACHMENTS);

   att->list[att->count++] = (struct drm_asahi_attachment){
      .size = rsrc->layout.size_B - rsrc->layout.level_offsets_B[0],
      .pointer = agx_map_gpu(rsrc),
   };
}

static void
agx_cmdtrace_dump_cmdbuf(struct agx_device *dev, struct agx_batch *batch,
                         struct util_dynarray *cmdbuf, const char *role,
                         uint64_t seqid, int submit_errno)
{
   fprintf(stderr,
           "asahi: cmdtrace role=%s queue=%u batch=%u seqid=%" PRIu64
           " syncobj=%u draws=%u errno=%d cmdbuf-bytes=%u\n",
           role, batch->ctx->queue_id, agx_batch_idx(batch), seqid,
           batch->syncobj, batch->draws, submit_errno, cmdbuf->size);

   if (!cmdbuf->size)
      return;

   /* G17P VDM encodes compact aliases rather than the allocator-owned BO
    * addresses tracked by agxdecode. Add decoder-only views for the lifetime
    * of this dump so it can follow the encoder and every aliased PPP payload
    * without changing any GPU mapping. */
   const unsigned payload_alias_count = util_dynarray_num_elements(
      &batch->g17p_vdm_aliases, struct agx_g17p_vdm_alias);
   const unsigned alias_count =
      (batch->vdm.g17p_vdm_alias != 0) + payload_alias_count;
   struct agx_bo *alias_bos = calloc(alias_count, sizeof(*alias_bos));
   struct agx_va *alias_vas = calloc(alias_count, sizeof(*alias_vas));
   unsigned alias_index = 0;

   if (alias_bos && alias_vas && batch->vdm.g17p_vdm_alias) {
      alias_bos[alias_index] = *batch->vdm.bo;
      alias_vas[alias_index] = (struct agx_va){
         .addr = batch->vdm.g17p_vdm_alias,
         .size_B = batch->vdm.bo->size,
      };
      alias_bos[alias_index].handle = UINT32_MAX - alias_index;
      alias_bos[alias_index].va = &alias_vas[alias_index];
      agxdecode_track_alloc(dev->agxdecode, &alias_bos[alias_index++]);
   }

   if (alias_bos && alias_vas) {
      util_dynarray_foreach(&batch->g17p_vdm_aliases,
                            struct agx_g17p_vdm_alias, entry) {
         alias_bos[alias_index] = *entry->bo;
         alias_vas[alias_index] = (struct agx_va){
            .addr = entry->address,
            .size_B = entry->bo->size,
         };
         alias_bos[alias_index].handle = UINT32_MAX - alias_index;
         alias_bos[alias_index].va = &alias_vas[alias_index];
         agxdecode_track_alloc(dev->agxdecode, &alias_bos[alias_index++]);
      }
   }

   agxdecode_drm_cmdbuf(dev->agxdecode, &dev->params, cmdbuf, true);
   agxdecode_next_frame();

   while (alias_index)
      agxdecode_track_free(dev->agxdecode, &alias_bos[--alias_index]);
   free(alias_vas);
   free(alias_bos);
}

static bool
agx_trace_bo_contains(struct agx_bo *bo, uint64_t address, size_t size)
{
   return bo && bo->va && address >= bo->va->addr &&
          address - bo->va->addr <= bo->size &&
          size <= bo->size - (address - bo->va->addr);
}

/* The batch/pools retain these BO references through the split submit loop. */
static struct agx_bo *
agx_trace_data_bo(struct agx_batch *batch, uint64_t address, size_t size)
{
   AGX_BATCH_FOREACH_BO(batch, entry) {
      struct agx_bo *bo = *entry;
      if (agx_trace_bo_contains(bo, address, size))
         return bo;
   }
   struct agx_pool *pools[] = {&batch->pool, &batch->pipeline_pool};
   for (unsigned i = 0; i < ARRAY_SIZE(pools); ++i) {
      util_dynarray_foreach(&pools[i]->bos, struct agx_bo *, bo) {
         if (agx_trace_bo_contains(*bo, address, size))
            return *bo;
      }
   }
   return NULL;
}

static bool
agx_trace_split_enabled(struct agx_device *dev)
{
   const char *value = getenv("G18_TRACE_SPLIT");
   return dev->chip == AGX_CHIP_G18G && value && !strcmp(value, "1");
}

/* G18's synchronous split-submit path has retired the input dependencies and
 * every earlier dispatch before calling this. Indirect dimensions may have
 * been produced by those jobs, so resolving them at command encoding is too
 * early. An empty dispatch must not enter the hardware completion protocol. */
static int
agx_g18_indirect_empty(struct agx_batch *batch,
                       const struct agx_cdm_dispatch *dispatch, bool *empty)
{
   *empty = false;
   if (dispatch->trace_mode == AGX_CDM_MODE_DIRECT)
      return 0;

   const bool local = dispatch->trace_mode == AGX_CDM_MODE_INDIRECT_LOCAL;
   if (!local && dispatch->trace_mode != AGX_CDM_MODE_INDIRECT_GLOBAL)
      return EINVAL;

   const size_t size = (local ? 6 : 3) * sizeof(uint32_t);
   const uint64_t address = dispatch->trace_indirect;
   struct agx_bo *bo = agx_trace_data_bo(batch, address, size);
   if (!address || (address & 3) || !bo)
      return EINVAL;

   const uint8_t *map = agx_bo_map(bo);
   if (!map)
      return ENOMEM;

   const bool shared = bo->flags & AGX_BO_SHARED;
   struct dma_buf_sync sync = {.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ};
   if (shared && (bo->prime_fd < 0 ||
                  drmIoctl(bo->prime_fd, DMA_BUF_IOCTL_SYNC, &sync)))
      return bo->prime_fd < 0 ? EINVAL : (errno ? errno : EIO);

   uint32_t dims[6] = {0};
   __sync_synchronize();
   memcpy(dims, map + (address - bo->va->addr), size);
   __sync_synchronize();

   if (shared) {
      sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ;
      if (drmIoctl(bo->prime_fd, DMA_BUF_IOCTL_SYNC, &sync))
         return errno ? errno : EIO;
   }

   /* A malformed local size is not valid empty work. */
   if (local && (!dims[3] || !dims[4] || !dims[5]))
      return EINVAL;

   *empty = !dims[0] || !dims[1] || !dims[2];
   if (*empty && agx_trace_split_enabled(agx_device(batch->ctx->base.screen)))
      fprintf(stderr, "G18_EMPTY_INDIRECT mode=%u grid=%u,%u,%u local=%u,%u,%u\n",
              dispatch->trace_mode, dims[0], dims[1], dims[2],
              dims[3], dims[4], dims[5]);
   return 0;
}

static bool
agx_trace_producer_data(struct agx_batch *batch, unsigned capture,
                         const char *label, uint64_t address, size_t size,
                         void *copy)
{
   uint8_t first[512], second[512];
   struct agx_bo *bo = size && size <= sizeof(first)
                         ? agx_trace_data_bo(batch, address, size) : NULL;
   if (!address || !bo || (bo->flags & (AGX_BO_SHARED | AGX_BO_WRITEBACK))) {
      fprintf(stderr,
              "G18_PRODUCER_UNAVAILABLE capture=%u label=%s address=%#" PRIx64
              " bytes=%zu owned=%u flags=%#x\n",
              capture, label, address, size, bo != NULL, bo ? bo->flags : 0);
      return false;
   }
   const uint8_t *map = agx_bo_map(bo);
   if (!map) {
      fprintf(stderr, "G18_PRODUCER_UNAVAILABLE capture=%u label=%s map_failed=1\n",
              capture, label);
      return false;
   }
   const uint8_t *source = map + (address - bo->va->addr);
   __sync_synchronize();
   memcpy(first, source, size);
   __sync_synchronize();
   memcpy(second, source, size);
   bool stable = !memcmp(first, second, size);
   fprintf(stderr,
           "G18_PRODUCER_DATA capture=%u label=%s address=%#" PRIx64
           " bytes=%zu stable=%u hex=",
           capture, label, address, size, stable);
   for (size_t i = 0; i < size; ++i)
      fprintf(stderr, "%02x", first[i]);
   fprintf(stderr, "\n");
   if (stable && copy)
      memcpy(copy, first, size);
   return stable;
}

static void
agx_trace_producer_state(struct agx_batch *batch,
                         const struct agx_cdm_dispatch *dispatch,
                         unsigned capture)
{
   if (!dispatch->trace_tess_params)
      return;

   struct poly_tess_params tess;
   if (!agx_trace_producer_data(batch, capture, "tess_params",
                                dispatch->trace_tess_params, sizeof(tess), &tess))
      return;

   fprintf(stderr,
           "G18_PRODUCER_TESS capture=%u patches=%u input_patch=%u output_patch=%u "
           "tcs_stride_el=%u isolines=%u points=%u\n",
           capture, tess.nr_patches, tess.input_patch_size, tess.output_patch_size,
           tess.tcs_stride_el, tess.isolines, tess.points_mode);
   agx_trace_producer_data(batch, capture, "heap", tess.heap,
                            sizeof(struct poly_heap), NULL);
   agx_trace_producer_data(batch, capture, "indirect_draw", tess.out_draws,
                            5 * sizeof(uint32_t), NULL);
   if (tess.nr_patches) {
      unsigned words = MIN2(tess.nr_patches, 64);
      agx_trace_producer_data(batch, capture, "counts_prefix", tess.counts,
                               words * sizeof(uint32_t), NULL);
      agx_trace_producer_data(batch, capture, "coord_allocs_prefix", tess.coord_allocs,
                               words * sizeof(uint32_t), NULL);
      words = MIN2(tess.tcs_stride_el, 64);
      if (words)
         agx_trace_producer_data(batch, capture, "first_tcs_patch_prefix", tess.tcs_buffer,
                                  words * sizeof(uint32_t), NULL);
   }
   agx_trace_producer_data(batch, capture, "vertex_params", dispatch->trace_vertex_params,
                            sizeof(struct poly_vertex_params), NULL);
   agx_trace_producer_data(batch, capture, "geometry_params", dispatch->trace_geometry_params,
                            sizeof(struct poly_geometry_params), NULL);
}

static void
agx_trace_split_dispatch(struct agx_batch *batch,
                         const struct agx_cdm_dispatch *dispatch,
                         unsigned index, unsigned count, uint64_t seqid,
                         unsigned in_sync_count)
{
   static unsigned captures;
   unsigned capture = p_atomic_inc_return(&captures);
   const uint8_t *map = agx_bo_map(batch->cdm.bo);
   if (capture > 128 || !map || dispatch->end < dispatch->start ||
       dispatch->end > batch->cdm.bo->size ||
       dispatch->end - dispatch->start > 512) {
      fprintf(stderr, "G18_SPLIT_TRACE_REFUSED capture=%u extent=%u..%u\n",
              capture, dispatch->start, dispatch->end);
      fflush(stderr);
      _Exit(87);
   }
   fprintf(stderr,
           "G18_SPLIT_TRACE capture=%u seq=%" PRIu64 " batch=%u dispatch=%u/%u "
           "prior_retired=%u in_syncs=%u mode=%u global=%u,%u,%u local=%u,%u,%u "
           "indirect=%#" PRIx64 " shader=%#" PRIx64 " program=%u texture=%u bytes=%u\n",
           capture, seqid, agx_batch_idx(batch), index + 1, count,
           index, in_sync_count, dispatch->trace_mode,
           dispatch->trace_global[0], dispatch->trace_global[1], dispatch->trace_global[2],
           dispatch->trace_local[0], dispatch->trace_local[1], dispatch->trace_local[2],
           dispatch->trace_indirect, dispatch->trace_shader,
           dispatch->trace_program, dispatch->texture_state, dispatch->end - dispatch->start);
   fprintf(stderr, "G18_SPLIT_BYTES capture=%u", capture);
   for (uint32_t i = dispatch->start; i < dispatch->end; ++i)
      fprintf(stderr, "%s%02x", i == dispatch->start ? " " : "", map[i]);
   fprintf(stderr, "\n");
   if (dispatch->trace_indirect) {
      struct agx_bo *bo = agx_trace_data_bo(batch, dispatch->trace_indirect,
                                           6 * sizeof(uint32_t));
      if (!bo || (bo->flags & (AGX_BO_SHARED | AGX_BO_WRITEBACK))) {
         fprintf(stderr, "G18_SPLIT_GRID_UNAVAILABLE capture=%u owned=%u flags=%#x\n",
                 capture, bo != NULL, bo ? bo->flags : 0);
      } else {
         const uint8_t *base = agx_bo_map(bo);
         if (!base) {
            fprintf(stderr, "G18_SPLIT_GRID_UNAVAILABLE capture=%u map_failed=1\n", capture);
         } else {
            uint32_t words[6], repeated[6];
            const uint8_t *grid = base + (dispatch->trace_indirect - bo->va->addr);
            __sync_synchronize();
            memcpy(words, grid, sizeof(words));
            __sync_synchronize();
            memcpy(repeated, grid, sizeof(repeated));
            fprintf(stderr,
                    "G18_SPLIT_GRID capture=%u handle=%u base=%#" PRIx64 " flags=%#x stable=%u words=",
                    capture, bo->handle, bo->va->addr, bo->flags,
                    !memcmp(words, repeated, sizeof(words)));
            for (unsigned i = 0; i < 6; ++i)
               fprintf(stderr, "%s%08x", i ? "," : "", words[i]);
            fprintf(stderr, "\n");
         }
      }
   }
   agx_trace_producer_state(batch, dispatch, capture);
   fflush(stderr);
}

void
agx_batch_submit(struct agx_context *ctx, struct agx_batch *batch,
                 struct drm_asahi_cmd_compute *compute,
                 struct drm_asahi_cmd_render *render)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   struct agx_screen *screen = agx_screen(ctx->base.screen);
   const struct agx_cdm_dispatch *cdm_dispatch = batch->cdm_dispatch.data;
   const unsigned cdm_dispatch_count =
      util_dynarray_num_elements(&batch->cdm_dispatch, struct agx_cdm_dispatch);

   /* Nothing from a batch that requested state without a proven native
    * encoding may reach the hardware. Retire it here without a context
    * reset: the draws simply never executed, which the application (and a
    * conformance run) observes as wrong output for exactly those commands.
    */
   /* A G18 render whose programs carry a TPS frame needs both API stages in
    * its resource suffix; the kernel sizes the shared TA/3D pool from it. */
   if (render && batch->g18.scratch && !batch->native_rejected &&
       (!batch->g18.resources.vertex_registers_16 ||
        !batch->g18.resources.fragment_registers_16)) {
      mesa_loge("asahi: a G18 render with a TPS frame has no recorded %s "
                "program on %s", batch->g18.resources.vertex_registers_16
                ? "fragment" : "vertex", agx_chip_name(dev->chip));
      batch->native_rejected = true;
   }

   /* G18G takes a per-dispatch split (see the prepass loop below), but its
    * hardware has run compute-only jobs (the J873 compute-add bring-up), so
    * it is not held to the M5 compute-before-render profile. A compute command with no
    * recorded dispatch carries no work -- the dispatch that opened the CDM
    * stream was refused -- and a stream without a launch is what M5 firmware
    * linked and never retired, so do not send it; the render still goes. */
   if (compute && !batch->native_rejected && dev->chip == AGX_CHIP_G18G &&
       !cdm_dispatch_count) {
      compute = NULL;
      if (!render) {
         mesa_loge("asahi: a G18G compute batch without a dispatch carries "
                   "no work");
         batch->native_rejected = true;
      }
   }

   if (batch->native_rejected) {
      batch_debug(batch, "DROP");
      mesa_loge("asahi: dropping work: batch %u used state with no proven "
                "native encoding on %s; nothing was submitted",
                agx_batch_idx(batch), agx_chip_name(dev->chip));
      if (ctx->batch == batch)
         ctx->batch = NULL;
      agx_batch_mark_submitted(batch);
      agx_batch_cleanup(ctx, batch, false);
      return;
   }

   /* Every draw in this batch was refused by the G17G profile, so its VDM
    * stream received none of them and the render has nothing to rasterize.
    * Submitting it anyway fails UAPI translation with a silent EINVAL, which
    * costs the compute prepass its result as well -- and a transform-feedback
    * capture is produced entirely by that prepass. Drop the render and keep
    * the dispatches. A batch that simply has no draws (an ordinary clear)
    * has no refusals and still submits its render. */
   if (render && batch->draws_refused && !batch->draws &&
       agx_chip_isa_encoding(dev->chip) == AGX_ISA_ENCODING_AGX3C) {
      mesa_loge("asahi: dropping a render whose %u draws were all outside the "
                "native profile on %s; the batch's compute work still runs",
                batch->draws_refused, agx_chip_name(dev->chip));
      render = NULL;
      /* With no compute either, nothing is left to submit: an empty
       * command buffer is an EINVAL from the M5 kernel. */
      if (!compute)
         batch->native_rejected = true;
   }

   /* Nothing from a batch that requested state without a proven native
    * encoding may reach the hardware. Retire it here without a context
    * reset: the draws simply never executed, which the application (and a
    * conformance run) observes as wrong output for exactly those commands.
    */
   /* A compute batch on AGX3c is submitted one dispatch at a time (see the
    * prepass loop below). A batch that carries compute work with no recorded
    * dispatches has nothing the profile covers, so drop it rather than send a
    * stream firmware will link and never retire -- which times out after
    * forty seconds and quarantines the VM, taking the session with it. */
   if (compute && !batch->native_rejected &&
       dev->chip == AGX_CHIP_G17G &&
       !cdm_dispatch_count) {
      mesa_loge("asahi: a compute batch with %u single-dispatch commands and "
                "%s render has no proven native encoding on %s",
                cdm_dispatch_count, render ? "a" : "no",
                agx_chip_name(dev->chip));
      batch->native_rejected = true;
   }

   /* M5 records each dispatch as its own command, so a compute batch with
    * none (its shader was empty) has no work, and the G17 runtime refuses a
    * submission without a command. Leave compute out; a render still goes.
    * With no render, retire the batch: nothing waits on its fences (it took
    * no timeline point and is neither ctx->syncobj nor a BO's writer), and a
    * pending server wait stays queued for the next submission. */
   if (compute && !batch->native_rejected && agx_chip_g17_m5(dev->chip) &&
       !cdm_dispatch_count) {
      compute = NULL;
      batch->compute_dropped = true;
      if (!render) {
         batch_debug(batch, "EMPTY");
         if (ctx->batch == batch)
            ctx->batch = NULL;
         /* No GPU timestamps were written for it. */
         util_dynarray_clear(&batch->timestamps);
         agx_batch_mark_submitted(batch);
         agx_batch_cleanup(ctx, batch, false);
         return;
      }
   }

   /* The G17 runtime refuses a render without its program storage list, and
    * the list's 16-bit command size bounds it. A list that does not fit is
    * a batch this driver cannot describe to the kernel. */
   if (!batch->native_rejected && agx_m5_render_resources(dev)) {
      const size_t render_stride =
         (dev->params.features & DRM_ASAHI_FEATURE_RENDER_RESOURCES_V2)
            ? sizeof(struct drm_asahi_render_program_v2)
            : sizeof(struct drm_asahi_render_program);
      const size_t render_bytes =
         (1 + util_dynarray_num_elements(&batch->m5_render_programs,
                                         struct drm_asahi_render_program_v2)) *
         render_stride;
      if ((render && render_bytes > UINT16_MAX) ||
          (compute && batch->m5_compute_programs.size > UINT16_MAX)) {
         mesa_loge("asahi: the program storage list of batch %u does not fit "
                   "one command (%zu render, %u compute bytes)",
                   agx_batch_idx(batch), render ? render_bytes : 0,
                   compute ? batch->m5_compute_programs.size : 0);
         batch->native_rejected = true;
      }
   }

   if (batch->native_rejected) {
      batch_debug(batch, "DROP");
      mesa_loge("asahi: dropping work: batch %u used state with no proven "
                "native encoding on %s; nothing was submitted",
                agx_batch_idx(batch), agx_chip_name(dev->chip));
      if (ctx->batch == batch)
         ctx->batch = NULL;
      agx_batch_mark_submitted(batch);
      agx_batch_cleanup(ctx, batch, false);
      return;
   }

   /* At most every BO contributes an input wait, followed by the explicit and
    * inter-context input fences. The submission then appends two output
    * fences and a private publication fence, all needing full records.
    */
   const bool split_compute = dev->chip == AGX_CHIP_G17P && compute && render &&
                             dev->params.max_commands_per_submission < 2;
   unsigned bo_count = util_dynarray_num_elements(&batch->bo_list.list,
                                                  struct agx_bo *);
   unsigned sync_capacity = bo_count + 5 + split_compute;
   unsigned in_sync_count = 0;
   unsigned shared_bo_count = 0;
   struct drm_asahi_sync *syncs =
      malloc(sync_capacity * sizeof(*syncs));
   struct agx_shared_bo_sync *shared_bos =
      malloc(MAX2(bo_count, 1) * sizeof(*shared_bos));
   struct util_dynarray cmdbuf = UTIL_DYNARRAY_INIT;

   if (!syncs || !shared_bos || ctx->any_faults) {
      agx_retire_failed_submit(ctx, batch, syncs, shared_bos, 0, NULL,
                               &cmdbuf, ctx->any_faults ? EIO : ENOMEM, 0, 0);
      return;
   }

   uint64_t wait_seqid = p_atomic_read(&screen->flush_wait_seqid);

   struct agx_submit_virt virt = {.ring_idx = ctx->virt_ring_idx};

   /* Elide syncing against our own queue */
   if (wait_seqid && wait_seqid == ctx->flush_my_seqid) {
      batch_debug(batch,
                  "Wait sync point %" PRIu64 " is ours, waiting on %" PRIu64
                  " instead",
                  wait_seqid, ctx->flush_other_seqid);
      wait_seqid = ctx->flush_other_seqid;
   }

   uint64_t seqid = p_atomic_inc_return(&screen->flush_cur_seqid);
   assert(seqid > wait_seqid);

   batch_debug(batch, "Sync point is %" PRIu64, seqid);

   /* This lock protects against a subtle race scenario:
    * - Context 1 submits and registers itself as writer for a BO
    * - Context 2 runs the below loop, and finds the writer syncobj
    * - Context 1 is destroyed,
    *     - flushing all batches, unregistering itself as a writer, and
    *     - Destroying syncobjs for all batches
    * - Context 2 submits, with a now invalid syncobj ID
    *
    * Since batch syncobjs are only destroyed on context destruction, we can
    * protect against this scenario with a screen-wide rwlock to ensure that
    * the syncobj destroy code cannot run concurrently with any other
    * submission. If a submit runs before the wrlock is taken, the syncobjs
    * must still exist (even if the batch was flushed and no longer a writer).
    * If it runs after the wrlock is released, then by definition the
    * just-destroyed syncobjs cannot be writers for any BO at that point.
    *
    * A screen-wide (not device-wide) rwlock is sufficient because by definition
    * resources can only be implicitly shared within a screen. Any shared
    * resources across screens must have been imported and will go through the
    * AGX_BO_SHARED path instead, which has no race (but is slower).
    */
   u_rwlock_rdlock(&screen->destroy_lock);

   AGX_BATCH_FOREACH_BO(batch, entry) {
      struct agx_bo *bo = *entry;

      if (bo->flags & AGX_BO_SHARED) {
         batch_debug(batch, "Waits on shared BO @ 0x%" PRIx64, bo->va->addr);

         /* Get a sync file fd from the buffer */
         int in_sync_fd = agx_export_sync_file(dev, bo);
         uint32_t sync_handle = 0;
         int ret = in_sync_fd >= 0
                      ? drmSyncobjCreate(dev->fd, 0, &sync_handle)
                      : -1;

         if (!ret) {
            ret = drmSyncobjImportSyncFile(dev->fd, sync_handle, in_sync_fd);
            if (ret) {
               drmSyncobjDestroy(dev->fd, sync_handle);
               sync_handle = 0;
            }
         }

         /* Preserve correctness if a sync-file fd or temporary syncobj cannot
          * be allocated. The dma-buf ioctl waits all prior implicit access;
          * if we already have the sync file, waiting it avoids cache-domain
          * transitions. */
         if (in_sync_fd < 0 ? !agx_wait_shared_bo(bo)
                            : (ret && agx_fallback_wait(in_sync_fd, "shared BO import failed"))) {
            const int sync_errno = errno ? errno : EIO;
            if (in_sync_fd >= 0)
               close(in_sync_fd);
            u_rwlock_rdunlock(&screen->destroy_lock);
            agx_retire_failed_submit(ctx, batch, syncs, shared_bos,
                                     shared_bo_count, virt.extres, &cmdbuf,
                                     sync_errno, 0, 0);
            return;
         }

         if (in_sync_fd >= 0)
            close(in_sync_fd);

         if (sync_handle)
            agx_add_sync(syncs, &in_sync_count, sync_handle);

         /* And keep track of the BO for cloning the out_sync */
         shared_bos[shared_bo_count++] = (struct agx_shared_bo_sync){
            .bo = bo,
            .in_syncobj = sync_handle,
         };
         if (dev->is_virtio)
            virt.extres_count++;
      } else {
         /* Deal with BOs which are not externally shared, but which have been
          * written from another context within the same screen. We also need to
          * wait on these using their syncobj.
          */
         uint64_t writer = p_atomic_read_relaxed(&bo->writer);
         uint32_t queue_id = agx_bo_writer_queue(writer);
         if (writer && queue_id != ctx->queue_id) {
            batch_debug(
               batch, "Waits on inter-context BO @ 0x%" PRIx64 " from queue %u",
               bo->va->addr, queue_id);

            agx_add_sync(syncs, &in_sync_count, agx_bo_writer_syncobj(writer));
         }
      }
   }

   if (dev->is_virtio && virt.extres_count) {
      virt.extres =
         malloc(virt.extres_count * sizeof(struct asahi_ccmd_submit_res));
      if (!virt.extres) {
         u_rwlock_rdunlock(&screen->destroy_lock);
         agx_retire_failed_submit(ctx, batch, syncs, shared_bos,
                                  shared_bo_count, NULL, &cmdbuf, ENOMEM, 0, 0);
         return;
      }

      for (unsigned i = 0; i < virt.extres_count; i++) {
         virt.extres[i].res_id = shared_bos[i].bo->uapi_handle;
         virt.extres[i].flags = ASAHI_EXTRES_READ | ASAHI_EXTRES_WRITE;
      }
   }

   if (dev->debug & AGX_DBG_SCRATCH) {
      if (compute)
         agx_scratch_debug_pre(&ctx->scratch_cs);
      if (render) {
         agx_scratch_debug_pre(&ctx->scratch_vs);
         agx_scratch_debug_pre(&ctx->scratch_fs);
      }
   }

   /* Add an explicit fence from gallium, if any */
   uint32_t explicit_syncobj = 0;
   if (!agx_get_in_sync(ctx, &explicit_syncobj)) {
      const int sync_errno = errno ? errno : EIO;
      u_rwlock_rdunlock(&screen->destroy_lock);
      agx_retire_failed_submit(ctx, batch, syncs, shared_bos,
                               shared_bo_count, virt.extres, &cmdbuf,
                               sync_errno, 0, 0);
      return;
   }
   agx_add_sync(syncs, &in_sync_count, explicit_syncobj);

   /* Add an implicit cross-context flush sync point, if any */
   if (wait_seqid) {
      batch_debug(batch, "Waits on inter-context sync point %" PRIu64,
                  wait_seqid);
      syncs[in_sync_count++] = (struct drm_asahi_sync){
         .sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ,
         .handle = screen->flush_syncobj,
         .timeline_value = wait_seqid,
      };
   }

   /* Subtle concurrency note: Since we assign seqids atomically and do
    * not lock submission across contexts, it is possible for two threads
    * to submit timeline syncobj updates out of order. As far as I can
    * tell, this case is handled in the kernel conservatively: it triggers
    * a fence context bump and effectively "splits" the timeline at the
    * larger point, causing future lookups for earlier points to return a
    * later point, waiting more. The signaling code still makes sure all
    * prior fences have to be signaled before considering a given point
    * signaled, regardless of order. That's good enough for us.
    *
    * (Note: this case breaks drm_syncobj_query_ioctl and for this reason
    * triggers a DRM_DEBUG message on submission, but we don't use that
    * so we don't care.)
    *
    * This case can be tested by setting seqid = 1 unconditionally here,
    * causing every single syncobj update to reuse the same timeline point.
    * Everything still works (but over-synchronizes because this effectively
    * serializes all submissions once any context flushes once).
    */
   /* Current G17P kernels accept mixed command arrays. Keep the old path
    * only for kernels advertising a single hardware command per ioctl.
    */
   uint32_t compute_syncobj = 0;
   uint32_t submit_syncobj = 0;
   int create_ret = drmSyncobjCreate(dev->fd, 0, &submit_syncobj);
   if (create_ret) {
      const int error = create_ret < -1 ? -create_ret : (errno ? errno : EIO);
      u_rwlock_rdunlock(&screen->destroy_lock);
      agx_retire_failed_submit(ctx, batch, syncs, shared_bos,
                               shared_bo_count, virt.extres, &cmdbuf,
                               error, 0, 0);
      return;
   }
   if (unlikely(dev->debug & AGX_DBG_PERF)) {
      ctx->submit_perf.batches++;
      ctx->submit_perf.mixed_batches += compute && render;
   }
   if (split_compute) {
      struct util_dynarray compute_cmdbuf = UTIL_DYNARRAY_INIT;
      struct drm_asahi_sync *compute_syncs =
         malloc((in_sync_count + 1) * sizeof(*compute_syncs));
      int ret = compute_syncs ? drmSyncobjCreate(dev->fd, 0, &compute_syncobj)
                              : -1;

      if (!ret) {
         memcpy(compute_syncs, syncs, in_sync_count * sizeof(*syncs));
         compute_syncs[in_sync_count] = (struct drm_asahi_sync){
            .sync_type = DRM_ASAHI_SYNC_SYNCOBJ,
            .handle = compute_syncobj,
         };

         struct drm_asahi_cmd_header header = agx_cmd_header(true, 0, 0);
         util_dynarray_append(&compute_cmdbuf, header);
         util_dynarray_append(&compute_cmdbuf, *compute);

         struct drm_asahi_submit compute_submit = {
            .flags = 0,
            .queue_id = ctx->queue_id,
            .in_sync_count = in_sync_count,
            .out_sync_count = 1,
            .syncs = (uint64_t)(uintptr_t)(compute_syncs),
            .cmdbuf = (uint64_t)(uintptr_t)(compute_cmdbuf.data),
            .cmdbuf_size = compute_cmdbuf.size,
         };

         if (unlikely(dev->debug & AGX_DBG_PERF))
            ctx->submit_perf.submits++;
         ret = dev->ops.submit(dev, &compute_submit, &virt);
         if (unlikely((dev->debug & AGX_DBG_PERF) && ret))
            ctx->submit_perf.failures++;
      }

      const int compute_errno =
         ret < -1 ? -ret : (ret ? (errno ? errno : EIO) : 0);

      if (!ret && (dev->debug & AGX_DBG_ASYNCTRACE))
         agx_cmdtrace_dump_cmdbuf(dev, batch, &compute_cmdbuf,
                                  "accepted-compute", seqid, 0);

      free(compute_syncs);
      util_dynarray_fini(&compute_cmdbuf);

      if (ret) {
         fprintf(stderr, "DRM_IOCTL_ASAHI_SUBMIT compute failed: %s\n",
                 strerror(compute_errno));
         drmSyncobjDestroy(dev->fd, submit_syncobj);
         u_rwlock_rdunlock(&screen->destroy_lock);
         agx_retire_failed_submit(ctx, batch, syncs, shared_bos,
                                  shared_bo_count, virt.extres, &cmdbuf,
                                  compute_errno, 0, compute_syncobj);
         return;
      }

      batch_debug(batch, "Render waits on its own compute submission");
      agx_add_sync(syncs, &in_sync_count, compute_syncobj);
   }

   struct drm_asahi_sync *out_syncs = syncs + in_sync_count;
   assert(in_sync_count + 3 <= sync_capacity);

   out_syncs[0] = (struct drm_asahi_sync){
      .sync_type = DRM_ASAHI_SYNC_SYNCOBJ,
      .handle = batch->syncobj,
   };

   out_syncs[1] = (struct drm_asahi_sync){
      .sync_type = DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ,
      .handle = screen->flush_syncobj,
      .timeline_value = seqid,
   };

   out_syncs[2] = (struct drm_asahi_sync){
      .sync_type = DRM_ASAHI_SYNC_SYNCOBJ,
      .handle = submit_syncobj,
   };

   /* A G17 refusal fills every output with an error fence, and another
    * context's timeline wait can find it. Add the screen point only once the
    * kernel has accepted the batch. */
   const bool late_point = agx_m5_render_resources(dev);
   if (late_point)
      out_syncs[1] = out_syncs[2];

   /* G18G's bring-up submit path, like the M5 kernel it follows, admits
    * exactly one hardware command per submission, so every batch that needs
    * a compute prepass before its render -- which is every geometry-shader
    * and transform-feedback draw -- would be refused outright with ENOTSUPP
    * and drop its work. Send the prepass as its own submission instead. That
    * is ordered by construction rather than by a barrier: the runtime
    * installs, dispatches and waits for the compute command to retire inside
    * the ioctl, so it has completed before the render submission is even
    * issued. Only the last submission carries the batch's output syncs.
    *
    * One submission per dispatch. When a render follows, it is the tail
    * submission and owns the output syncs; with no render the final dispatch
    * takes that role, so the loop below stops one short.
    *
    * The G17 runtime that runs G17S and G17C has none of that restriction:
    * it takes a mixed command array and orders it with barriers, so the
    * recorded dispatches go as separate commands of this one submission. So
    * does G17G where it runs on that runtime (agx_m5_render_resources); only
    * on the M5 runtime of older kernels does it take one dispatch per
    * submission. */
   const bool m5_compute_commands = agx_m5_render_resources(dev) && compute;
   const bool m5_split_prepass =
      (dev->chip == AGX_CHIP_G17G || dev->chip == AGX_CHIP_G18G) && compute &&
      !m5_compute_commands;
   const unsigned m5_split_count =
      !m5_split_prepass  ? 0
      : render           ? cdm_dispatch_count
      : cdm_dispatch_count ? cdm_dispatch_count - 1
                                  : 0;

   /* Compute commands in this submission, which the render waits on. */
   unsigned compute_commands = 0;

   /* Submit! */
   if (compute && !split_compute && m5_compute_commands) {
      /* The G17 runtime (M5 Pro, M5 Max, and M5 on it) takes a mixed
       * command array, so each recorded dispatch is its own compute command
       * here, the single-dispatch stream the Gallium M5 compute path builds
       * and the M5 runtime takes one ioctl at a time. Command i waits on the
       * i before it and on all earlier work; the storage list precedes them
       * all.
       *
       * The kernel takes at most max_commands_per_submission render and
       * compute commands in one submission (64) and refuses a longer one
       * with EINVAL, and a batch records a command per dispatch. The
       * leading dispatches then go in submissions of their own, as many as
       * fit in each, each with the storage list. Barrier 0 on their first
       * command keeps the chain in order, and this submission keeps the
       * last dispatches and the render. */
      const unsigned limit = MAX2(dev->params.max_commands_per_submission, 2);
      const unsigned tail = limit - (render ? 1 : 0);
      unsigned first = 0;

      while (cdm_dispatch_count - first > tail) {
         const unsigned n = MIN2(limit, cdm_dispatch_count - first - tail);
         struct util_dynarray chunk = UTIL_DYNARRAY_INIT;
         struct drm_asahi_sync *chunk_syncs =
            malloc((in_sync_count + 1) * sizeof(*chunk_syncs));
         int ret = -1;

         if (chunk_syncs &&
             (compute_syncobj ||
              !drmSyncobjCreate(dev->fd, 0, &compute_syncobj))) {
            memcpy(chunk_syncs, syncs, in_sync_count * sizeof(*syncs));
            chunk_syncs[in_sync_count] = (struct drm_asahi_sync){
               .sync_type = DRM_ASAHI_SYNC_SYNCOBJ,
               .handle = compute_syncobj,
            };

            ASSERTED bool encoded =
               agx_append_m5_compute_resources(dev, batch, &chunk);
            assert(encoded);
            agx_append_m5_compute_commands(batch, compute, first, first + n,
                                           cdm_dispatch_count, &chunk);

            struct drm_asahi_submit chunk_submit = {
               .flags = 0,
               .queue_id = ctx->queue_id,
               .in_sync_count = in_sync_count,
               .out_sync_count = 1,
               .syncs = (uint64_t)(uintptr_t)chunk_syncs,
               .cmdbuf = (uint64_t)(uintptr_t)chunk.data,
               .cmdbuf_size = chunk.size,
            };

            if (unlikely(dev->debug & AGX_DBG_PERF))
               ctx->submit_perf.submits++;
            ret = dev->ops.submit(dev, &chunk_submit, &virt);
            if (unlikely((dev->debug & AGX_DBG_PERF) && ret))
               ctx->submit_perf.failures++;
         }

         const int chunk_errno =
            ret < -1 ? -ret : (ret ? (errno ? errno : EIO) : 0);
         free(chunk_syncs);
         util_dynarray_fini(&chunk);

         if (ret) {
            fprintf(stderr, "DRM_IOCTL_ASAHI_SUBMIT compute failed: %s\n",
                    strerror(chunk_errno));
            drmSyncobjDestroy(dev->fd, submit_syncobj);
            u_rwlock_rdunlock(&screen->destroy_lock);
            /* Earlier chunks are in flight: their fence is the accepted
             * one. Without them the syncobj holds no fence. */
            agx_retire_failed_submit(ctx, batch, syncs, shared_bos,
                                     shared_bo_count, virt.extres, &cmdbuf,
                                     chunk_errno, first ? compute_syncobj : 0,
                                     first ? 0 : compute_syncobj);
            return;
         }
         first += n;
      }

      ASSERTED bool encoded =
         agx_append_m5_compute_resources(dev, batch, &cmdbuf);
      assert(encoded);
      agx_append_m5_compute_commands(batch, compute, first, cdm_dispatch_count,
                                     cdm_dispatch_count, &cmdbuf);
      compute_commands = cdm_dispatch_count - first;
   } else if (compute && !split_compute) {
      /* Barrier on previous submission */
      struct drm_asahi_cmd_header header = agx_cmd_header(true, 0, 0);

      util_dynarray_append(&cmdbuf, header);
      util_dynarray_append(&cmdbuf, *compute);
      compute_commands = 1;

      if (m5_split_prepass) {
         /* The M5 compute profile is grounded for a single-dispatch control
          * stream; a geometry or transform-feedback prepass emits several.
          * Report the stream extent so a hang can be told apart from a
          * refusal without a kernel rebuild. */
         if (agx_env_flag(AGX_ENV_NEO_DRAW_DIAGNOSTICS)) {
            fprintf(stderr,
                    "NEO_PREPASS queue=%u batch=%u stream=%" PRIx64 "..%" PRIx64
                    " stream_bytes=%" PRIu64 "\n",
                    ctx->queue_id, agx_batch_idx(batch),
                    (uint64_t)compute->cdm_ctrl_stream_base,
                    (uint64_t)compute->cdm_ctrl_stream_end,
                    (uint64_t)(compute->cdm_ctrl_stream_end -
                               compute->cdm_ctrl_stream_base));
            /* Dump the words so the stream can be decoded against genxml
             * without another hardware cycle. */
            const uint32_t *w = agx_bo_map(batch->cdm.bo);
            const size_t n = (compute->cdm_ctrl_stream_end -
                              compute->cdm_ctrl_stream_base) / 4;
            fprintf(stderr, "NEO_PREPASS_WORDS");
            for (size_t i = 0; i < n && i < 64; ++i)
               fprintf(stderr, " %08x", w[i]);
            fprintf(stderr, "\n");
         }
         /* One command per dispatch. agx_launch_internal() terminated each
          * one where it ended and recorded its extent, so each command is
          * the single-dispatch stream the profile covers. They are ordered
          * by construction: the runtime polls for retirement inside the
          * ioctl, so dispatch N has completed before N+1 is issued. */
         const uint64_t cdm_base = batch->cdm.bo->va->addr;
         /* G18G only: a dispatch that bound texture or image state asks the
          * kernel for the native texture compute mode. M5's kernel refuses
          * the bit, and no other chip takes this split. */
         const bool g18g = dev->chip == AGX_CHIP_G18G;
         /* Wait through the kernel's normal input-fence path before reading
          * indirect counts, even if the first or every dispatch is empty.
          * This submission has no output fence and touches no GPU queue. */
         bool has_indirect = false;
         for (unsigned i = 0; g18g && i < cdm_dispatch_count; ++i)
            has_indirect |= cdm_dispatch[i].trace_mode != AGX_CDM_MODE_DIRECT;
         if (has_indirect && in_sync_count) {
            struct drm_asahi_submit wait = {
               .queue_id = ctx->queue_id,
               .in_sync_count = in_sync_count,
               .syncs = (uint64_t)(uintptr_t)syncs,
            };
            const int ret = dev->ops.submit(dev, &wait, &virt);
            if (ret) {
               const int error = ret < -1 ? -ret : (errno ? errno : EIO);
               drmSyncobjDestroy(dev->fd, submit_syncobj);
               u_rwlock_rdunlock(&screen->destroy_lock);
               agx_retire_failed_submit(ctx, batch, syncs, shared_bos,
                                        shared_bo_count, virt.extres, &cmdbuf,
                                        error, 0, 0);
               return;
            }
            in_sync_count = 0;
         }
         for (unsigned i = 0; i < m5_split_count; ++i) {
            bool empty = false;
            const int error = g18g
               ? agx_g18_indirect_empty(batch, &cdm_dispatch[i], &empty) : 0;
            if (error) {
               fprintf(stderr, "asahi: cannot resolve indirect dispatch %u: %s\n",
                       i + 1, strerror(error));
               drmSyncobjDestroy(dev->fd, submit_syncobj);
               u_rwlock_rdunlock(&screen->destroy_lock);
               agx_retire_failed_submit(ctx, batch, syncs, shared_bos,
                                        shared_bo_count, virt.extres, &cmdbuf,
                                        error, 0, 0);
               return;
            }
            if (empty)
               continue;
            struct drm_asahi_cmd_compute one = *compute;
            one.cdm_ctrl_stream_base = cdm_base + cdm_dispatch[i].start;
            one.cdm_ctrl_stream_end = cdm_base + cdm_dispatch[i].end;
            if (g18g && cdm_dispatch[i].texture_state)
               one.flags |= DRM_ASAHI_COMPUTE_USES_TEXTURE_STATE;

            util_dynarray_clear(&cmdbuf);
            agx_append_split_compute(&cmdbuf, header, one, g18g,
                                      &cdm_dispatch[i].resources);

            struct drm_asahi_submit prepass = {
               .flags = 0,
               .queue_id = ctx->queue_id,
               .in_sync_count = i == 0 ? in_sync_count : 0,
               .out_sync_count = 0,
               .syncs = (uint64_t)(uintptr_t)(syncs),
               .cmdbuf = (uint64_t)(uintptr_t)(cmdbuf.data),
               .cmdbuf_size = cmdbuf.size,
            };
            if (agx_trace_split_enabled(dev))
               agx_trace_split_dispatch(batch, &cdm_dispatch[i], i,
                                        m5_split_count, seqid, prepass.in_sync_count);
            int prepass_ret = dev->ops.submit(dev, &prepass, &virt);
            if (prepass_ret) {
               const int prepass_errno = prepass_ret < -1 ? -prepass_ret
                                         : (errno ? errno : EIO);
               fprintf(stderr,
                       "DRM_IOCTL_ASAHI_SUBMIT compute dispatch %u/%u failed: %s\n",
                       i + 1, m5_split_count,
                       strerror(prepass_errno));
               if (agx_trace_split_enabled(dev)) {
                  fprintf(stderr, "G18_SPLIT_STOP first_submit_error=%d dispatch=%u/%u\n",
                          prepass_errno, i + 1, m5_split_count);
                  fflush(stderr);
                  _Exit(86);
               }
               drmSyncobjDestroy(dev->fd, submit_syncobj);
               u_rwlock_rdunlock(&screen->destroy_lock);
               agx_retire_failed_submit(ctx, batch, syncs, shared_bos,
                                        shared_bo_count, virt.extres, &cmdbuf,
                                        prepass_errno, 0, 0);
               return;
            }
         }
         /* The tail below is a submission of its own: it waits for nothing
          * (every earlier dispatch has retired) and owns the output syncs.
          * With no render it carries the final dispatch. */
         util_dynarray_clear(&cmdbuf);
         in_sync_count = 0;

         if (!render && cdm_dispatch_count) {
            const unsigned last = cdm_dispatch_count - 1;
            bool empty = false;
            const int error = g18g
               ? agx_g18_indirect_empty(batch, &cdm_dispatch[last], &empty) : 0;
            if (error) {
               fprintf(stderr, "asahi: cannot resolve indirect tail: %s\n", strerror(error));
               drmSyncobjDestroy(dev->fd, submit_syncobj);
               u_rwlock_rdunlock(&screen->destroy_lock);
               agx_retire_failed_submit(ctx, batch, syncs, shared_bos,
                                        shared_bo_count, virt.extres, &cmdbuf,
                                        error, 0, 0);
               return;
            }
            struct drm_asahi_cmd_compute one = *compute;
            one.cdm_ctrl_stream_base = cdm_base + cdm_dispatch[last].start;
            one.cdm_ctrl_stream_end = cdm_base + cdm_dispatch[last].end;
            if (g18g && cdm_dispatch[last].texture_state)
               one.flags |= DRM_ASAHI_COMPUTE_USES_TEXTURE_STATE;

            /* An all-empty batch still submits below: its normal kernel
             * output publication preserves binary and timeline ordering. */
            if (!empty) {
               agx_append_split_compute(&cmdbuf, header, one, g18g,
                                         &cdm_dispatch[last].resources);
            }
         }
      }
   }

   if (render) {
      struct attachments att = {.count = 0};
      struct pipe_framebuffer_state *fb = &batch->key;

      for (unsigned i = 0; i < fb->nr_cbufs; ++i) {
         if (fb->cbufs[i].texture)
            asahi_add_attachment(&att, agx_resource(fb->cbufs[i].texture));
      }

      if (fb->resolve)
         asahi_add_attachment(&att, agx_resource(fb->resolve));

      if (fb->zsbuf.texture) {
         struct agx_resource *rsrc = agx_resource(fb->zsbuf.texture);
         asahi_add_attachment(&att, rsrc);

         if (rsrc->separate_stencil)
            asahi_add_attachment(&att, rsrc->separate_stencil);
      }

      if (att.count) {
         struct drm_asahi_cmd_header header = {
            .cmd_type = DRM_ASAHI_SET_FRAGMENT_ATTACHMENTS,
            .size = sizeof(att.list[0]) * att.count,
            .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
            .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
         };

         util_dynarray_append(&cmdbuf, header);
         util_dynarray_append_array(&cmdbuf, struct drm_asahi_attachment,
                                    att.list, att.count);
      }

      if (agx_m5_render_resources(dev)) {
         ASSERTED bool encoded =
            agx_append_m5_render_resources(dev, batch, &cmdbuf);
         assert(encoded);

         if (batch->m5_shader_writes &&
             (dev->params.features & DRM_ASAHI_FEATURE_RENDER_SHADER_WRITES))
            render->flags |= DRM_ASAHI_RENDER_SHADER_WRITES;
      }

      /* Compute starts with barriers on both prior subqueues. Waiting on
       * its prefix (every compute command of this submission) therefore
       * orders render after both the compute results and all prior work,
       * including ioctl input syncs on older kernels that attach them only to
       * the first hardware command (hk's first_sq rule). Software attachment
       * and resource commands do not advance the hardware prefixes.
       * Every dispatch of a G18G split prepass went out as its own
       * submission above, and naming one here would be a barrier on a
       * command this submission does not contain.
       */
      const bool compute_here =
         compute && !split_compute && !m5_split_prepass && compute_commands;
      struct drm_asahi_cmd_header header = agx_cmd_header(
         false, compute_here ? DRM_ASAHI_BARRIER_NONE : 0,
         compute_here ? compute_commands : 0);
      /* G18G takes the full drm_asahi_cmd_render. */
      header.size = dev->chip == AGX_CHIP_G18G
                       ? sizeof(struct drm_asahi_cmd_render)
                       : agx_render_command_size(dev);

      if (batch->g18.scratch) {
         /* The complete render command (prior and ray-query fields zero =
          * absent) plus the suffix; see DRM_ASAHI_RENDER_G18_RESOURCES. */
         struct drm_asahi_cmd_render_g18 extended = {
            .legacy.legacy.legacy = *render,
            .resources = batch->g18.resources,
         };
         STATIC_ASSERT(offsetof(struct drm_asahi_cmd_render_g18, resources) ==
                       sizeof(struct drm_asahi_cmd_render_with_ray_query));
         extended.legacy.legacy.legacy.flags |= DRM_ASAHI_RENDER_G18_RESOURCES;
         extended.resources.version = 1;
         extended.resources.flags = 3;
         header.size = sizeof(extended);
         util_dynarray_append(&cmdbuf, header);
         util_dynarray_append(&cmdbuf, extended);
      } else {
         util_dynarray_append(&cmdbuf, header);
         util_dynarray_append_array(&cmdbuf, uint8_t, render, header.size);
      }
   }

   struct drm_asahi_submit submit = {
      .flags = 0,
      .queue_id = ctx->queue_id,
      .in_sync_count = in_sync_count,
      .out_sync_count = late_point ? 2 : 3,
      .syncs = (uint64_t)(uintptr_t)(m5_split_prepass ? out_syncs : syncs),
      .cmdbuf = (uint64_t)(uintptr_t)(cmdbuf.data),
      .cmdbuf_size = cmdbuf.size,
   };

   if (agx_env_flag(AGX_ENV_NEO_DRAW_DIAGNOSTICS))
      fprintf(stderr, "NEO_SUBMIT queue=%u batch=%u seq=%llu draws=%u render=%u compute=%u bytes=%zu\n",
              ctx->queue_id, agx_batch_idx(batch), (unsigned long long)seqid,
              batch->draws, !!render, !!compute, (size_t)cmdbuf.size);

   if (unlikely(dev->debug & AGX_DBG_PERF))
      ctx->submit_perf.submits++;
   if (dev->debug & AGX_DBG_FAULTTRACE) {
      util_dynarray_append_array(&batch->faulttrace_cmdbuf, uint8_t,
                                 cmdbuf.data, cmdbuf.size);
      batch->faulttrace_seqid = seqid;
   }
   int ret = dev->ops.submit(dev, &submit, &virt);
   if (unlikely((dev->debug & AGX_DBG_PERF) && ret))
      ctx->submit_perf.failures++;
   const int submit_errno =
      ret < -1 ? -ret : (ret ? (errno ? errno : EIO) : 0);
   if (agx_env_flag(AGX_ENV_NEO_DRAW_DIAGNOSTICS))
      fprintf(stderr, "NEO_SUBMIT_RESULT queue=%u batch=%u seq=%llu ret=%d errno=%d\n",
              ctx->queue_id, agx_batch_idx(batch), (unsigned long long)seqid,
              ret, submit_errno);

   u_rwlock_rdunlock(&screen->destroy_lock);

   if (!ret && (dev->debug & AGX_DBG_ASYNCTRACE))
      agx_cmdtrace_dump_cmdbuf(dev, batch, &cmdbuf, "accepted-current",
                               seqid, 0);

   if (ret) {
      if (compute && !split_compute && !(m5_split_prepass && render)) {
         fprintf(stderr, "DRM_IOCTL_ASAHI_SUBMIT compute failed: %s\n",
                 strerror(submit_errno));
      }

      if (render) {
         struct drm_asahi_cmd_render *c = render;
         fprintf(
            stderr,
            "DRM_IOCTL_ASAHI_SUBMIT render failed: %s (%dx%d tile %dx%d layers %d samples %d)\n",
            strerror(submit_errno),
            c->width_px, c->height_px, c->utile_width_px, c->utile_height_px,
            c->layers, c->samples);
      }

      /* Unlike `trace`, faulttrace does not wait after or decode successful
       * submissions, so it preserves the compositor's queue chronology. BO
       * tracking was armed at device open and each submitted batch retained
       * its small DRM command wrapper. A wait-queue-timeline failure is
       * observed by the current (rejected) submit but was caused by an earlier
       * in-flight submit, so decode those candidates before the observer. */
      if (dev->debug & AGX_DBG_FAULTTRACE) {
         unsigned i;
         foreach_submitted(ctx, i) {
            struct agx_batch *candidate = &ctx->batches.slots[i];
            agx_cmdtrace_dump_cmdbuf(
               dev, candidate, &candidate->faulttrace_cmdbuf,
               "in-flight-prior", candidate->faulttrace_seqid, submit_errno);
         }

         agx_cmdtrace_dump_cmdbuf(dev, batch, &batch->faulttrace_cmdbuf,
                                  "rejected-current", seqid, submit_errno);
      } else if (dev->debug & AGX_DBG_ASYNCTRACE) {
         agx_cmdtrace_dump_cmdbuf(dev, batch, &cmdbuf, "rejected-current",
                                  seqid, submit_errno);
      }

      agx_retire_failed_submit(ctx, batch, syncs, shared_bos,
                               shared_bo_count, virt.extres, &cmdbuf,
                               submit_errno, compute_syncobj, submit_syncobj);
      return;
   }

   /* Successful enqueue retains the fences in the kernel. */
   if (compute_syncobj)
      drmSyncobjDestroy(dev->fd, compute_syncobj);
   drmSyncobjDestroy(dev->fd, submit_syncobj);

   /* Add the screen point held back above. Failing that, finish the batch on
    * the CPU and keep the older point: later work is ordered after it anyway. */
   bool point_published = true;
   if (late_point && drmSyncobjTransfer(dev->fd, screen->flush_syncobj, seqid,
                                        batch->syncobj, 0, 0)) {
      agx_msg("timeline point transfer failed; waiting for the batch\n");
      drmSyncobjWait(dev->fd, &batch->syncobj, 1, INT64_MAX,
                     DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);
      point_published = false;
   }

   /* Now stash our batch fence into any shared BOs. */
   if (shared_bo_count) {
      /* Convert our handle to a sync file */
      int out_sync_fd = -1;
      int ret = drmSyncobjExportSyncFile(dev->fd, batch->syncobj, &out_sync_fd);
      bool transfer_failed = ret || out_sync_fd < 0;

      for (unsigned i = 0; i < shared_bo_count; i++) {
         batch_debug(batch, "Signals shared BO @ 0x%" PRIx64,
                     shared_bos[i].bo->va->addr);

         /* Free the in_sync handle we just acquired */
         if (shared_bos[i].in_syncobj) {
            ret = drmSyncobjDestroy(dev->fd, shared_bos[i].in_syncobj);
            if (ret)
               transfer_failed = true;
            shared_bos[i].in_syncobj = 0;
         }

         /* And then import the out_sync sync file into it */
         if (out_sync_fd >= 0 &&
             agx_import_sync_file(dev, shared_bos[i].bo, out_sync_fd))
            transfer_failed = true;
      }

      if (out_sync_fd >= 0)
         close(out_sync_fd);

      /* Once submitted, a missing implicit release fence cannot be repaired
       * by dropping the batch. Wait for completion before returning so every
       * dma-buf is safe for its next owner even when sync-file transfer runs
       * out of kernel resources. */
      if (transfer_failed) {
         ret = drmSyncobjWait(dev->fd, &batch->syncobj, 1, INT64_MAX,
                              DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);
         if (ret) {
            agx_msg("shared BO release-fence fallback wait failed: %s\n",
                    strerror(ret < -1 ? -ret : (errno ? errno : EIO)));
            const bool first_fault = !ctx->any_faults;
            ctx->any_faults = true;
            if (first_fault && ctx->reset.reset)
               ctx->reset.reset(ctx->reset.data, PIPE_GUILTY_CONTEXT_RESET);
         }
      }
   }

   /* Record the syncobj on each BO we write, so it can be added post-facto as a
    * fence if the BO is exported later...
    */
   AGX_BATCH_FOREACH_BO(batch, entry) {
      struct agx_bo *bo = *entry;
      struct agx_batch *writer = agx_writer_get(ctx, bo->handle);

      if (!writer)
         continue;

      /* Skip BOs that are written by submitted batches, they're not ours */
      if (agx_batch_is_submitted(writer))
         continue;

      /* But any BOs written by active batches are ours */
      if (unlikely(writer != batch) && (dev->chip == AGX_CHIP_G18G || dev->chip == AGX_CHIP_G17G)) {
         mesa_loge("asahi: batch %u lists BO %u (%s, %zu bytes @ 0x%" PRIx64
                   ") whose active writer is batch %u",
                   agx_batch_idx(batch), bo->handle,
                   bo->label ? bo->label : "unlabelled", bo->size,
                   bo->va->addr, agx_batch_idx(writer));
      }
      assert(writer == batch && "exclusive writer");
      p_atomic_set(&bo->writer, agx_bo_writer(ctx->queue_id, batch->syncobj));
      batch_debug(batch, "Writes to BO @ 0x%" PRIx64, bo->va->addr);
   }

   free(syncs);
   free(shared_bos);

   if (dev->debug & (AGX_DBG_TRACE | AGX_DBG_SYNC | AGX_DBG_SCRATCH)) {
      if (dev->debug & AGX_DBG_TRACE) {
         agxdecode_drm_cmdbuf(dev->agxdecode, &dev->params, &cmdbuf, true);
         agxdecode_next_frame();
      }

      /* Wait so we can get errors reported back */
      int ret = drmSyncobjWait(dev->fd, &batch->syncobj, 1, INT64_MAX, 0, NULL);
      assert(!ret);

      agx_batch_print_stats(dev, batch);

      if (dev->debug & AGX_DBG_SCRATCH) {
         if (compute) {
            fprintf(stderr, "CS scratch:\n");
            agx_scratch_debug_post(&ctx->scratch_cs);
         }
         if (render) {
            fprintf(stderr, "VS scratch:\n");
            agx_scratch_debug_post(&ctx->scratch_vs);
            fprintf(stderr, "FS scratch:\n");
            agx_scratch_debug_post(&ctx->scratch_fs);
         }
      }
   }

   util_dynarray_fini(&cmdbuf);
   agx_batch_mark_submitted(batch);

   free(virt.extres);

   /* Record the last syncobj for fence creation */
   ctx->syncobj = batch->syncobj;

   /* Update the last seqid in the context (must only happen if the submit
    * succeeded, otherwise the timeline point would not be valid).
    */
   if (point_published)
      ctx->flush_last_seqid = seqid;

   if (ctx->batch == batch)
      ctx->batch = NULL;

   /* Try to clean up up to two batches, to keep memory usage down */
   if (agx_cleanup_batches(ctx) >= 0)
      agx_cleanup_batches(ctx);
}

bool
agx_poll_batch(struct agx_context *ctx, struct agx_batch *batch)
{
   /* Polling must never submit an active batch. A query may have writers in
    * several slots, so retire only the completed submitted writer here. */
   if (agx_batch_is_active(batch) || batch->submit_quarantined)
      return false;

   if (!agx_batch_is_submitted(batch))
      return true;

   struct agx_device *dev = agx_device(ctx->base.screen);
   int ret = drmSyncobjWait(dev->fd, &batch->syncobj, 1, 0, 0, NULL);
   assert(!ret || ret == -ETIME);
   if (ret)
      return false;

   agx_batch_cleanup(ctx, batch, false);
   return true;
}

void
agx_sync_batch(struct agx_context *ctx, struct agx_batch *batch)
{
   struct agx_device *dev = agx_device(ctx->base.screen);

   if (agx_batch_is_active(batch))
      agx_flush_batch(ctx, batch);

   /* Empty batch case, already cleaned up */
   if (!agx_batch_is_submitted(batch))
      return;

   if (batch->submit_quarantined)
      return;

   assert(batch->syncobj);
   int ret = drmSyncobjWait(dev->fd, &batch->syncobj, 1, INT64_MAX, 0, NULL);
   assert(!ret);
   agx_batch_cleanup(ctx, batch, false);
}

void
agx_sync_batch_for_reason(struct agx_context *ctx, struct agx_batch *batch,
                          const char *reason)
{
   if (reason)
      perf_debug_ctx(ctx, "Syncing due to: %s\n", reason);

   agx_sync_batch(ctx, batch);
}

void
agx_sync_all(struct agx_context *ctx, const char *reason)
{
   if (reason)
      perf_debug_ctx(ctx, "Syncing all due to: %s\n", reason);

   unsigned idx;
   foreach_active(ctx, idx) {
      agx_flush_batch(ctx, &ctx->batches.slots[idx]);
   }

   foreach_submitted(ctx, idx) {
      agx_sync_batch(ctx, &ctx->batches.slots[idx]);
   }
}

void
agx_batch_reset(struct agx_context *ctx, struct agx_batch *batch)
{
   batch_debug(batch, "RESET");

   assert(!batch->initialized);

   /* Reset an empty batch. Like submit, but does nothing. */
   agx_batch_mark_submitted(batch);

   if (ctx->batch == batch)
      ctx->batch = NULL;

   agx_batch_cleanup(ctx, batch, true);
}

/*
 * Timestamp queries record the time after all current work is finished,
 * which we handle as the time after all current batches finish (since we're a
 * tiler and would rather not split the batch). So add a query to all active
 * batches.
 */
void
agx_add_timestamp_end_query(struct agx_context *ctx, struct agx_query *q)
{
   unsigned idx;
   foreach_active(ctx, idx) {
      agx_batch_add_timestamp_query(&ctx->batches.slots[idx], q);
   }
}

/*
 * To implement a memory barrier conservatively, flush any batch that contains
 * an incoherent memory write (requiring a memory barrier to synchronize). This
 * could be further optimized.
 */
void
agx_memory_barrier(struct pipe_context *pctx, unsigned flags)
{
   struct agx_context *ctx = agx_context(pctx);

   unsigned i;
   foreach_active(ctx, i) {
      struct agx_batch *batch = &ctx->batches.slots[i];

      if (batch->incoherent_writes)
         agx_flush_batch_for_reason(ctx, batch, "Memory barrier");
   }
}

/*
 * Copyright 2022 Alyssa Rosenzweig
 * Copyright 2019-2020 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <stdint.h>
#include <xf86drm.h>
#include "pipe/p_defines.h"
#include "util/bitset.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_dump.h"
#include "util/u_inlines.h"
#include "agx_abi.h"
#include "agx_bo.h"
#include "agx_device.h"
#include "agx_state.h"
#include "libagx.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"

static bool
is_occlusion(struct agx_query *query)
{
   switch (query->type) {
   case PIPE_QUERY_OCCLUSION_COUNTER:
   case PIPE_QUERY_OCCLUSION_PREDICATE:
   case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
      return true;
   default:
      return false;
   }
}

static bool
is_timer(struct agx_query *query)
{
   switch (query->type) {
   case PIPE_QUERY_TIMESTAMP:
   case PIPE_QUERY_TIME_ELAPSED:
      return true;
   default:
      return false;
   }
}

struct agx_oq_heap {
   /* The GPU allocation itself */
   struct agx_device *dev;
   struct agx_bo *bo;

   /* Bitset of query indices that are in use */
   BITSET_DECLARE(available, AGX_MAX_OCCLUSION_QUERIES);
};

static struct agx_oq_heap *
agx_alloc_oq_heap(struct agx_context *ctx)
{
   struct agx_oq_heap *heap = CALLOC_STRUCT(agx_oq_heap);
   if (!heap)
      return NULL;

   heap->dev = agx_device(ctx->base.screen);
   heap->bo =
      agx_bo_create(heap->dev, AGX_MAX_OCCLUSION_QUERIES * sizeof(uint64_t), 0,
                    AGX_BO_WRITEBACK, "Occlusion query heap");
   if (!heap->bo) {
      FREE(heap);
      return NULL;
   }

   /* At the start, everything is available */
   BITSET_ONES(heap->available);

   return heap;
}

void
agx_destroy_oq_heap_ctx(struct agx_context *ctx)
{
   if (!ctx->oq)
      return;

   agx_bo_unreference(ctx->oq->dev, ctx->oq->bo);
   FREE(ctx->oq);
   ctx->oq = NULL;
}

static struct agx_oq_heap *
agx_get_oq_heap_locked(struct agx_context *ctx)
{
   if (!ctx->oq)
      ctx->oq = agx_alloc_oq_heap(ctx);

   return ctx->oq;
}

static struct agx_ptr
agx_alloc_oq(struct agx_context *ctx)
{
   simple_mtx_lock(&ctx->oq_lock);
   struct agx_oq_heap *heap = agx_get_oq_heap_locked(ctx);
   if (!heap) {
      simple_mtx_unlock(&ctx->oq_lock);
      return (struct agx_ptr){NULL, 0};
   }

   /* Find first available */
   int ffs = BITSET_FFS(heap->available);
   if (!ffs) {
      simple_mtx_unlock(&ctx->oq_lock);
      return (struct agx_ptr){NULL, 0};
   }

   /* Allocate it */
   unsigned index = ffs - 1;
   BITSET_CLEAR(heap->available, index);

   unsigned offset = index * sizeof(uint64_t);
   struct agx_ptr ptr = {
      (uint8_t *)agx_bo_map(heap->bo) + offset,
      heap->bo->va->addr + offset,
   };
   simple_mtx_unlock(&ctx->oq_lock);
   return ptr;
}

static unsigned
agx_oq_index(struct agx_context *ctx, struct agx_query *q)
{
   assert(is_occlusion(q));

   return (q->ptr.gpu - ctx->oq->bo->va->addr) / sizeof(uint64_t);
}

static void
agx_free_oq(struct agx_context *ctx, struct agx_query *q)
{
   simple_mtx_lock(&ctx->oq_lock);
   struct agx_oq_heap *heap = agx_get_oq_heap_locked(ctx);
   unsigned index = agx_oq_index(ctx, q);

   assert(index < AGX_MAX_OCCLUSION_QUERIES);
   assert(!BITSET_TEST(heap->available, index));

   BITSET_SET(heap->available, index);
   simple_mtx_unlock(&ctx->oq_lock);
}

uint64_t
agx_get_occlusion_heap(struct agx_batch *batch)
{
   simple_mtx_lock(&batch->ctx->oq_lock);
   struct agx_oq_heap *oq = batch->ctx->oq;
   struct agx_bo *bo = oq ? oq->bo : NULL;
   simple_mtx_unlock(&batch->ctx->oq_lock);

   if (!bo)
      return 0;

   if (agx_batch_uses_bo(batch, bo))
      return bo->va->addr;
   else
      return 0;
}

static struct pipe_query *
agx_create_query(struct pipe_context *ctx, unsigned query_type, unsigned index)
{
   if ((query_type == PIPE_QUERY_TIMESTAMP && !ctx->screen->caps.query_timestamp) ||
       (query_type == PIPE_QUERY_TIME_ELAPSED && !ctx->screen->caps.query_time_elapsed))
      return NULL;

   struct agx_query *query = calloc(1, sizeof(struct agx_query));

   query->type = query_type;
   query->index = index;

   /* Set all writer generations to a sentinel that will always compare as
    * false, since nothing writes to no queries.
    */
   for (unsigned i = 0; i < ARRAY_SIZE(query->writer_generation); ++i) {
      query->writer_generation[i] = UINT64_MAX;
   }

   if (is_occlusion(query)) {
      query->ptr = agx_alloc_oq(agx_context(ctx));
   } else {
      /* TODO: a BO for the query is wasteful, but we benefit from BO list
       * tracking / reference counting to deal with lifetimes.
       */
      query->bo = agx_bo_create(agx_device(ctx->screen), sizeof(uint64_t) * 2,
                                0, AGX_BO_WRITEBACK, "Query");
      query->ptr = (struct agx_ptr){
         .gpu = query->bo->va->addr,
         .cpu = agx_bo_map(query->bo),
      };
   }

   if (!query->ptr.gpu) {
      free(query);
      return NULL;
   }

   return (struct pipe_query *)query;
}

static void
flush_query_writers(struct agx_context *ctx, struct agx_query *query,
                    const char *reason)
{
   STATIC_ASSERT(ARRAY_SIZE(ctx->batches.generation) == AGX_MAX_BATCHES);
   STATIC_ASSERT(ARRAY_SIZE(ctx->batches.slots) == AGX_MAX_BATCHES);
   STATIC_ASSERT(ARRAY_SIZE(query->writer_generation) == AGX_MAX_BATCHES);

   for (unsigned i = 0; i < AGX_MAX_BATCHES; ++i) {
      if (query->writer_generation[i] == ctx->batches.generation[i])
         agx_flush_batch_for_reason(ctx, &ctx->batches.slots[i], reason);
   }
}

static void
sync_query_writers(struct agx_context *ctx, struct agx_query *query,
                   const char *reason)
{
   for (unsigned i = 0; i < AGX_MAX_BATCHES; ++i) {
      if (query->writer_generation[i] == ctx->batches.generation[i] &&
          !agx_poll_batch(ctx, &ctx->batches.slots[i])) {
         if (unlikely(agx_device(ctx->base.screen)->debug & AGX_DBG_PERF))
            ctx->perf.query_waits++;
         agx_sync_batch_for_reason(ctx, &ctx->batches.slots[i], reason);
      }
   }
}

static bool
is_query_busy(struct agx_context *ctx, struct agx_query *query, bool submit)
{
   bool busy = false;
   bool need_submit = false;
   for (unsigned i = 0; i < AGX_MAX_BATCHES; ++i) {
      if (query->writer_generation[i] == ctx->batches.generation[i] &&
          !agx_poll_batch(ctx, &ctx->batches.slots[i])) {
         if (!submit)
            return true;

         /* GL availability polling must eventually finish without an app
          * flush. Submit only still-active or queued writers, once; never wait
          * for them here. Already submitted writers need only the zero-timeout
          * poll above. Return not-ready even if submission retires inline. */
         busy = true;
         if (agx_batch_is_active(&ctx->batches.slots[i]))
            agx_flush_batch_for_reason(ctx, &ctx->batches.slots[i],
                                       "Query result availability");
         if (agx_batch_is_queued(&ctx->batches.slots[i]))
            need_submit = true;
      }
   }

   if (need_submit)
      agx_submit_pending(ctx);

   return busy;
}

static void
agx_destroy_query(struct pipe_context *pctx, struct pipe_query *pquery)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_query *query = (struct agx_query *)pquery;
   struct agx_device *dev = agx_device(pctx->screen);

   /* We don't reference count the occlusion query allocations, so we need to
    * sync writers when destroying so we can freely write from the CPU after
    * it's destroyed, since the driver will assume an available query is idle.
    *
    * For other queries, the BO itself is reference counted after the pipe_query
    * is destroyed so we don't need to flush.
    */
   if (is_occlusion(query)) {
      sync_query_writers(ctx, query, "Occlusion query destroy");
      agx_free_oq(ctx, query);
   } else {
      agx_bo_unreference(dev, query->bo);
   }

   free(pquery);
}

static bool
agx_begin_query(struct pipe_context *pctx, struct pipe_query *pquery)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_query *query = (struct agx_query *)pquery;

   ctx->dirty |= AGX_DIRTY_QUERY;

   switch (query->type) {
   case PIPE_QUERY_OCCLUSION_COUNTER:
   case PIPE_QUERY_OCCLUSION_PREDICATE:
   case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
      ctx->occlusion_query = query;
      break;

   case PIPE_QUERY_PRIMITIVES_GENERATED:
      ctx->prims_generated[query->index] = query;
      break;

   case PIPE_QUERY_PRIMITIVES_EMITTED:
      ctx->tf_prims_generated[query->index] = query;
      break;

   case PIPE_QUERY_SO_OVERFLOW_PREDICATE:
      ctx->tf_overflow[query->index] = query;
      break;

   case PIPE_QUERY_SO_OVERFLOW_ANY_PREDICATE:
      ctx->tf_any_overflow = query;
      break;

   case PIPE_QUERY_TIME_ELAPSED:
      ctx->time_elapsed = query;
      break;

   case PIPE_QUERY_TIMESTAMP:
      /* No-op */
      break;

   case PIPE_QUERY_PIPELINE_STATISTICS_SINGLE:
      assert(query->index < ARRAY_SIZE(ctx->pipeline_statistics));
      ctx->pipeline_statistics[query->index] = query;
      break;

   default:
      return false;
   }

   /* begin_query zeroes, sync so we can do that write from the CPU */
   sync_query_writers(ctx, query, "Query overwritten");

   uint64_t *ptr = query->ptr.cpu;
   ptr[0] = 0;

   if (query->type == PIPE_QUERY_TIME_ELAPSED) {
      /* Timestamp begin in second record, the timestamp end in the first */
      ptr[1] = UINT64_MAX;
   }

   return true;
}

static bool
agx_end_query(struct pipe_context *pctx, struct pipe_query *pquery)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_device *dev = agx_device(pctx->screen);
   struct agx_query *query = (struct agx_query *)pquery;

   ctx->dirty |= AGX_DIRTY_QUERY;

   switch (query->type) {
   case PIPE_QUERY_OCCLUSION_COUNTER:
   case PIPE_QUERY_OCCLUSION_PREDICATE:
   case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
      ctx->occlusion_query = NULL;
      return true;
   case PIPE_QUERY_PRIMITIVES_GENERATED:
      ctx->prims_generated[query->index] = NULL;
      return true;
   case PIPE_QUERY_PRIMITIVES_EMITTED:
      ctx->tf_prims_generated[query->index] = NULL;
      return true;
   case PIPE_QUERY_SO_OVERFLOW_PREDICATE:
      ctx->tf_overflow[query->index] = NULL;
      return true;
   case PIPE_QUERY_SO_OVERFLOW_ANY_PREDICATE:
      ctx->tf_any_overflow = NULL;
      return true;
   case PIPE_QUERY_TIME_ELAPSED:
      ctx->time_elapsed = NULL;
      return true;
   case PIPE_QUERY_PIPELINE_STATISTICS_SINGLE:
      assert(query->index < ARRAY_SIZE(ctx->pipeline_statistics));
      ctx->pipeline_statistics[query->index] = NULL;
      return true;
   case PIPE_QUERY_TIMESTAMP: {
      /* Timestamp logically written now, set up batches to MAX their finish
       * time in. If there are no batches, it's just the current time stamp.
       */
      agx_add_timestamp_end_query(ctx, query);

      uint64_t *value = query->ptr.cpu;
      *value = agx_get_gpu_timestamp(dev);

      return true;
   }
   default:
      return false;
   }
}

enum query_copy_type {
   QUERY_COPY_NORMAL,
   QUERY_COPY_BOOL32,
   QUERY_COPY_BOOL64,
   QUERY_COPY_TIMESTAMP,
   QUERY_COPY_TIME_ELAPSED,
};

static enum query_copy_type
classify_query_type(enum pipe_query_type type)
{
   switch (type) {
   case PIPE_QUERY_OCCLUSION_PREDICATE:
   case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
      return QUERY_COPY_BOOL32;

   case PIPE_QUERY_SO_OVERFLOW_PREDICATE:
   case PIPE_QUERY_SO_OVERFLOW_ANY_PREDICATE:
      return QUERY_COPY_BOOL64;

   case PIPE_QUERY_TIMESTAMP:
      return QUERY_COPY_TIMESTAMP;

   case PIPE_QUERY_TIME_ELAPSED:
      return QUERY_COPY_TIME_ELAPSED;

   default:
      return QUERY_COPY_NORMAL;
   }
}

static void
agx_query_format_result(struct agx_device *dev, enum pipe_query_type type,
                        uint64_t value, uint64_t begin_val,
                        union pipe_query_result *vresult)
{
   switch (classify_query_type(type)) {
   case QUERY_COPY_BOOL32:
      vresult->u64 = ((uint32_t)value) != 0;
      return;

   case QUERY_COPY_BOOL64:
      vresult->u64 = value != 0;
      return;

   case QUERY_COPY_NORMAL:
      vresult->u64 = value;
      return;

   case QUERY_COPY_TIMESTAMP:
      vresult->u64 = agx_gpu_timestamp_to_ns(dev, value);
      return;

   case QUERY_COPY_TIME_ELAPSED:
      /* end - begin */
      vresult->u64 = agx_gpu_timestamp_to_ns(dev, value - begin_val);
      return;

   default:
      UNREACHABLE("Other queries not yet supported");
   }
}

static bool
agx_get_query_result_inner(struct pipe_context *pctx, struct pipe_query *pquery,
                           bool wait, bool submit,
                           union pipe_query_result *vresult)
{
   struct agx_query *query = (struct agx_query *)pquery;
   struct agx_context *ctx = agx_context(pctx);
   struct agx_device *dev = agx_device(pctx->screen);

   if (wait)
      sync_query_writers(ctx, query, "Reading query results");
   else if (is_query_busy(ctx, query, submit))
      return false;

   uint64_t *ptr = query->ptr.cpu;
   agx_query_format_result(dev, query->type, ptr[0],
                           is_timer(query) ? ptr[1] : 0, vresult);
   return true;
}

static bool
agx_get_query_result_flushed(struct agx_context *ctx, struct agx_query *query,
                             bool wait, union pipe_query_result *vresult)
{
   struct agx_device *dev = agx_device(ctx->base.screen);

   for (;;) {
      uint32_t wait_syncobj = 0;
      bool need_wait = false;
      uint64_t extra_end_ts = 0;
      uint64_t extra_begin_ts = UINT64_MAX;

      simple_mtx_lock(&ctx->query_lock);

      for (unsigned i = 0; i < AGX_MAX_BATCHES; ++i) {
         if (query->writer_generation[i] !=
             ctx->batches.query_state[i].generation)
            continue;

         if (!ctx->batches.query_state[i].submitted) {
            /* A flushed query's writers must already be submitted. Fall back
             * safely if an unexpected unsubmitted writer is observed.
             */
            assert(!"flushed query still has unsubmitted writer");
            simple_mtx_unlock(&ctx->query_lock);
            threaded_context_unwrap_sync(&ctx->tc->base);
            return agx_get_query_result_inner(&ctx->base,
                                              (struct pipe_query *)query, wait,
                                              true, vresult);
         }

         if (ctx->batches.query_state[i].quarantined) {
            if (!wait) {
               simple_mtx_unlock(&ctx->query_lock);
               return false;
            }
            continue;
         }

         uint32_t syncobj = ctx->batches.query_state[i].syncobj;
         int ret = drmSyncobjWait(dev->fd, &syncobj, 1, 0, 0, NULL);
         assert(!ret || ret == -ETIME);
         if (ret) {
            if (!wait) {
               simple_mtx_unlock(&ctx->query_lock);
               return false;
            }
            if (unlikely(dev->debug & AGX_DBG_PERF))
               p_atomic_inc(&ctx->perf.query_waits);
            wait_syncobj = syncobj;
            need_wait = true;
            break;
         }

         /* Batch i finished on the GPU before the driver thread retired it.
          * Fold its timestamps locally while holding query_lock so the slot
          * cannot be retired and reused underneath us.
          */
         if (is_timer(query) && ctx->timestamps) {
            struct agx_timestamps *ts =
               ((struct agx_timestamps *)agx_bo_map(ctx->timestamps)) + i;
            if (ctx->batches.query_state[i].has_cdm) {
               extra_begin_ts = MIN2(extra_begin_ts, ts->comp_start);
               extra_end_ts = MAX2(extra_end_ts, ts->comp_end);
            }
            if (ctx->batches.query_state[i].has_vdm) {
               extra_begin_ts = MIN2(extra_begin_ts, ts->vtx_start);
               extra_end_ts = MAX2(extra_end_ts, ts->frag_end);
            }
         }
      }

      if (!need_wait) {
         uint64_t *ptr = query->ptr.cpu;
         uint64_t value = ptr[0];
         uint64_t begin_val = is_timer(query) ? ptr[1] : 0;

         if (is_timer(query)) {
            value = MAX2(value, extra_end_ts);
            begin_val = MIN2(begin_val, extra_begin_ts);
         }

         simple_mtx_unlock(&ctx->query_lock);
         agx_query_format_result(dev, query->type, value, begin_val, vresult);
         return true;
      }

      /* Snapshot the pending syncobj before dropping query_lock so a concurrent
       * driver-thread retirement and reuse of slot i cannot replace the fence.
       */
      uint32_t tmp_syncobj = 0;
      if (drmSyncobjCreate(dev->fd, 0, &tmp_syncobj) == 0) {
         if (drmSyncobjTransfer(dev->fd, tmp_syncobj, 0, wait_syncobj, 0, 0) !=
             0) {
            drmSyncobjDestroy(dev->fd, tmp_syncobj);
            tmp_syncobj = 0;
         }
      }
      simple_mtx_unlock(&ctx->query_lock);

      uint32_t target = tmp_syncobj ? tmp_syncobj : wait_syncobj;
      int ret;
      do {
         ret = drmSyncobjWait(dev->fd, &target, 1, INT64_MAX, 0, NULL);
      } while (ret == -EINTR);
      assert(!ret);

      if (tmp_syncobj)
         drmSyncobjDestroy(dev->fd, tmp_syncobj);
   }
}

static bool
agx_get_query_result(struct pipe_context *pctx, struct pipe_query *pquery,
                     bool wait, union pipe_query_result *vresult)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_query *query = (struct agx_query *)pquery;

   /* When threaded_query::flushed is true, u_threaded_context calls
    * get_query_result directly from the frontend thread without syncing.
    * Poll/wait against the locked batch query_state snapshot instead of
    * draining the TC queue.
    */
   if (ctx->tc && query->base.flushed)
      return agx_get_query_result_flushed(ctx, query, wait, vresult);

   return agx_get_query_result_inner(pctx, pquery, wait, true, vresult);
}

static unsigned
result_type_size(enum pipe_query_value_type result_type)
{
   return (result_type <= PIPE_QUERY_TYPE_U32) ? 4 : 8;
}

static void
agx_get_query_result_resource_cpu(struct agx_context *ctx,
                                  struct agx_query *query,
                                  enum pipe_query_flags flags,
                                  enum pipe_query_value_type result_type,
                                  int index, struct pipe_resource *resource,
                                  unsigned offset)
{
   union pipe_query_result result;
   if (index < 0) {
      /* availability */
      result.u64 = !is_query_busy(ctx, query, true);
   } else {
      if (!agx_get_query_result_inner(&ctx->base, (void *)query,
                                      flags & PIPE_QUERY_WAIT, true, &result))
         return;
   }

   /* Clamp to type, arb_query_buffer_object-qbo tests */
   if (result_type == PIPE_QUERY_TYPE_U32) {
      result.u32 = MIN2(result.u64, u_uintN_max(32));
   } else if (result_type == PIPE_QUERY_TYPE_I32) {
      int64_t x = result.u64;
      x = MAX2(MIN2(x, u_intN_max(32)), u_intN_min(32));
      result.u32 = x;
   }

   tc_buffer_write(&ctx->base, resource, offset,
                   result_type_size(result_type), &result.u64);
}

static bool
agx_get_query_result_resource_gpu(struct agx_context *ctx,
                                  struct agx_query *query,
                                  enum pipe_query_flags flags,
                                  enum pipe_query_value_type result_type,
                                  int index, struct pipe_resource *prsrc,
                                  unsigned offset)
{
   /* Handle availability queries on CPU */
   if (index < 0)
      return false;

   /* TODO: timer queries on GPU */
   if (query->type == PIPE_QUERY_TIMESTAMP ||
       query->type == PIPE_QUERY_TIME_ELAPSED)
      return false;

   flush_query_writers(ctx, query, util_str_query_type(query->type, true));

   struct agx_resource *rsrc = agx_resource(prsrc);
   enum query_copy_type copy_type = classify_query_type(query->type);

   struct agx_batch *batch = agx_get_compute_batch(ctx);
   batch->query_access = true;
   agx_batch_init_state(batch);
   agx_dirty_all(ctx);

   /* Set params */
   agx_batch_writes_range(batch, rsrc, offset, result_type_size(result_type));

   unsigned bool_size = copy_type == QUERY_COPY_BOOL64   ? 8
                        : copy_type == QUERY_COPY_BOOL32 ? 4
                                                         : 0;

   libagx_copy_query_gl(batch, agx_1d(1), AGX_BARRIER_ALL, query->ptr.gpu,
                        agx_map_gpu(rsrc) + offset, result_type, bool_size);
   return true;
}

static void
agx_get_query_result_resource(struct pipe_context *pipe, struct pipe_query *q,
                              enum pipe_query_flags flags,
                              enum pipe_query_value_type result_type, int index,
                              struct pipe_resource *resource, unsigned offset)
{
   struct agx_query *query = (struct agx_query *)q;
   struct agx_context *ctx = agx_context(pipe);

   /* Try to copy on the GPU */
   if (!agx_get_query_result_resource_gpu(ctx, query, flags, result_type, index,
                                          resource, offset)) {

      /* Else, fallback to CPU */
      agx_get_query_result_resource_cpu(ctx, query, flags, result_type, index,
                                        resource, offset);
   }
}

static void
agx_set_active_query_state(struct pipe_context *pipe, bool enable)
{
   struct agx_context *ctx = agx_context(pipe);

   ctx->active_queries = enable;
   ctx->dirty |= AGX_DIRTY_QUERY;
}

static void
agx_add_query_to_batch(struct agx_batch *batch, struct agx_query *query)
{
   unsigned idx = agx_batch_idx(batch);
   struct agx_bo *bo = is_occlusion(query) ? batch->ctx->oq->bo : query->bo;

   agx_batch_add_bo(batch, bo);
   batch->barrier_untracked = true;
   query->writer_generation[idx] = batch->ctx->batches.generation[idx];
}

void
agx_batch_add_timestamp_query(struct agx_batch *batch, struct agx_query *q)
{
   if (q) {
      agx_add_query_to_batch(batch, q);
      util_dynarray_append(&batch->timestamps, q->ptr);
   }
}

uint16_t
agx_get_oq_index(struct agx_batch *batch, struct agx_query *query)
{
   agx_add_query_to_batch(batch, query);
   return agx_oq_index(batch->ctx, query);
}

uint64_t
agx_get_query_address(struct agx_batch *batch, struct agx_query *query)
{
   if (query) {
      agx_add_query_to_batch(batch, query);
      return query->ptr.gpu;
   } else {
      return AGX_SCRATCH_PAGE_ADDRESS;
   }
}

void
agx_finish_batch_queries(struct agx_batch *batch, uint64_t begin_ts,
                         uint64_t end_ts)
{
   struct agx_context *ctx = batch->ctx;
   unsigned idx = agx_batch_idx(batch);

   simple_mtx_lock(&ctx->query_lock);

   /* Write out timestamps before advancing the batch generation so app-thread
    * readers observing the incremented generation see the final values.
    */
   util_dynarray_foreach(&batch->timestamps, struct agx_ptr, it) {
      uint64_t *ptr = it->cpu;

      ptr[0] = MAX2(ptr[0], end_ts);
      ptr[1] = MIN2(ptr[1], begin_ts);
   }

   /* Remove the batch as writer from all queries by incrementing the generation
    * of the batch.
    */
   ctx->batches.generation[idx]++;
   ctx->batches.query_state[idx].generation = ctx->batches.generation[idx];
   ctx->batches.query_state[idx].submitted = false;
   ctx->batches.query_state[idx].quarantined = false;
   ctx->batches.query_state[idx].has_cdm = false;
   ctx->batches.query_state[idx].has_vdm = false;

   simple_mtx_unlock(&ctx->query_lock);
}

void
agx_query_increment_cpu(struct agx_context *ctx, struct agx_query *query,
                        uint64_t increment)
{
   if (!query)
      return;

   sync_query_writers(ctx, query, "CPU query increment");

   uint64_t *value = query->ptr.cpu;
   *value += increment;
}

static void
agx_render_condition(struct pipe_context *pipe, struct pipe_query *query,
                     bool condition, enum pipe_render_cond_flag mode)
{
   struct agx_context *ctx = agx_context(pipe);

   ctx->cond_query = query;
   ctx->cond_cond = condition;
   ctx->cond_mode = mode;
}

bool
agx_render_condition_check_inner(struct agx_context *ctx)
{
   assert(ctx->cond_query != NULL && "precondition");

   perf_debug_ctx(ctx, "Implementing conditional rendering on the CPU");

   union pipe_query_result res = {0};
   bool wait = ctx->cond_mode != PIPE_RENDER_COND_NO_WAIT &&
               ctx->cond_mode != PIPE_RENDER_COND_BY_REGION_NO_WAIT;

   /* A NO_WAIT draw can render conservatively without splitting its batch.
    * Unlike an explicit availability request, it need not force progress. */
   struct pipe_query *pq = (struct pipe_query *)ctx->cond_query;

   if (agx_get_query_result_inner(&ctx->base, pq, wait, false, &res))
      return res.u64 != ctx->cond_cond;

   return true;
}

void
agx_init_query_functions(struct pipe_context *pctx)
{
   pctx->create_query = agx_create_query;
   pctx->destroy_query = agx_destroy_query;
   pctx->begin_query = agx_begin_query;
   pctx->end_query = agx_end_query;
   pctx->get_query_result = agx_get_query_result;
   pctx->get_query_result_resource = agx_get_query_result_resource;
   pctx->set_active_query_state = agx_set_active_query_state;
   pctx->render_condition = agx_render_condition;

   /* By default queries are active */
   agx_context(pctx)->active_queries = true;
}

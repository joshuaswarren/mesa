/*
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2024 Valve Corporation
 * Copyright 2022 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl.h"
#include "compiler/libcl/libcl_vk.h"
#include "query.h"
#include "query_copy.h"

static inline global uint64_t *
query_report(global uint64_t *results, global uint16_t *oq_index,
             uint reports_per_query, uint query)
{
   /* For occlusion queries, results[] points to the device global heap. We
    * need to remap indices according to the query pool's allocation.
    */
   uint result_index = oq_index ? oq_index[query] : query;
   if (oq_index && result_index == LIBAGX_QUERY_INDEX_UNUSED)
      return 0;

   return results + (result_index * reports_per_query);
}

KERNEL(32)
libagx_copy_query(global uint32_t *availability, global uint64_t *results,
                  global uint16_t *oq_index, uint64_t dst_addr,
                  uint64_t dst_stride, uint32_t first_query,
                  VkQueryResultFlagBits flags, uint16_t reports_per_query,
                  uint16_t availability_shift)
{
   uint i = cl_global_id.x;
   uint64_t dst = dst_addr + (((uint64_t)i) * dst_stride);
   uint32_t query = first_query + i;

   bool available;
   if (availability)
      available = libagx_query_available(availability, availability_shift,
                                         query);
   else
      available = (results[query] != LIBAGX_QUERY_UNAVAILABLE);

   global uint64_t *report =
      query_report(results, oq_index, reports_per_query, query);

   if (available || (flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
      for (unsigned i = 0; i < reports_per_query; ++i) {
         vk_write_query(dst, i, flags, report ? report[i] : 0);
      }
   }

   if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
      vk_write_query(dst, reports_per_query, flags, available);
   }
}

/* G17P uses a single range dispatch, including WAIT polling. Keep the AGX2
 * entry point above unchanged: no timestamp-copy UAPI is involved here. */
KERNEL(32)
libagx_copy_query_g17p(global uint32_t *availability, global uint64_t *results,
                       global uint16_t *oq_index, uint64_t dst_addr,
                       uint64_t dst_stride, uint32_t first_query,
                       VkQueryResultFlagBits flags, uint16_t reports_per_query,
                       uint16_t availability_shift)
{
   libagx_copy_query_g17p_result(availability, results, oq_index, dst_addr,
                                dst_stride, first_query, flags,
                                reports_per_query, availability_shift,
                                cl_global_id.x);
}

/**
 * Goes through a series of consecutive query indices in the given pool,
 * setting all element values to 0 and emitting them as available. With
 * availability_shift 1, each availability slot is 64 bits wide.
 */
KERNEL(32)
libagx_reset_query(global uint32_t *availability, global uint64_t *results,
                   global uint16_t *oq_index, uint32_t first_query,
                   uint16_t reports_per_query, int set_available,
                   uint16_t availability_shift)
{
   uint32_t query = first_query + cl_global_id.x;

   uint64_t value = 0;
   if (availability) {
      uint32_t word = query << availability_shift;
      availability[word] = set_available;
      if (availability_shift)
         availability[word + 1] = 0;
   } else {
      value = set_available ? 0 : LIBAGX_QUERY_UNAVAILABLE;
   }

   global uint64_t *report =
      query_report(results, oq_index, reports_per_query, query);

   /* XXX: is this supposed to happen on the begin? */
   if (report) {
      for (unsigned j = 0; j < reports_per_query; ++j)
         report[j] = value;
   }
}

/* TODO: Share with Gallium... */
enum pipe_query_value_type {
   PIPE_QUERY_TYPE_I32,
   PIPE_QUERY_TYPE_U32,
   PIPE_QUERY_TYPE_I64,
   PIPE_QUERY_TYPE_U64,
};

KERNEL(1)
libagx_copy_query_gl(global uint64_t *query, global uint64_t *dest,
                     ushort value_type, ushort bool_size)
{
   uint64_t value = *query;

   if (bool_size == 4) {
      value = (uint32_t)value;
   }

   if (bool_size) {
      value = value != 0;
   }

   if (value_type <= PIPE_QUERY_TYPE_U32) {
      global uint32_t *dest32 = (global uint32_t *)dest;
      bool u32 = (value_type == PIPE_QUERY_TYPE_U32);

      *dest32 = u32 ? convert_uint_sat(value) : convert_int_sat((int64_t)value);
   } else {
      *dest = value;
   }
}

KERNEL(4)
libagx_copy_xfb_counters(constant struct libagx_xfb_counter_copy *push)
{
   unsigned i = cl_local_id.x;

   *(push->dest[i]) = push->src[i] ? *(push->src[i]) : 0;
}

KERNEL(1)
libagx_increment_statistic(global uint32_t *statistic, uint32_t delta)
{
   *statistic += delta;
}

KERNEL(1)
libagx_increment_cs_invocations(global uint *grid, global uint32_t *statistic,
                                uint32_t local_size_threads)
{
   *statistic += local_size_threads * grid[0] * grid[1] * grid[2];
}

/*
 * G17P launches an indirect dispatch from a six-word grid -- three GLOBAL
 * thread counts then the three local sizes -- not from the API's three
 * workgroup counts. Expand one into the other, exactly as the tessellator's
 * setup kernel does for its own indirect launches, so the indirect path uses
 * the same grounded launch record as the direct one.
 */
KERNEL(1)
libagx_setup_indirect_dispatch(global uint *api_grid, global uint *grid,
                               uint32_t local_x, uint32_t local_y,
                               uint32_t local_z)
{
   grid[0] = api_grid[0] * local_x;
   grid[1] = api_grid[1] * local_y;
   grid[2] = api_grid[2] * local_z;
   grid[3] = local_x;
   grid[4] = local_y;
   grid[5] = local_z;
}

KERNEL(32)
libagx_write_u32s(constant struct libagx_imm_write *p)
{
   uint id = cl_global_id.x;
   *(p[id].address) = p[id].value;
}

/*
 * Copy earlier timestamps into later queries, one pair per thread, so a batch
 * of copies is one launch and one flush.
 *
 * We set the source as volatile since the caching situation around timestamps
 * is a bit unclear. It might not be necessary but - absent hardware/firmware
 * documentation - this gives me peace of mind.
 */
KERNEL(32)
libagx_copy_timestamps(constant struct libagx_timestamp_copy *p)
{
   uint id = cl_global_id.x;
   volatile global uint64_t *src = p[id].src;
   *(p[id].dest) = *src;
}

KERNEL(1)
libagx_write_u32(global uint32_t *address, uint32_t value)
{
   *address = value;
}

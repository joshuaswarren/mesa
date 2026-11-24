/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The shader-less compute commands a queue builds itself, outside command
 * buffers: the sparse-table flush, the timestamp carrier and the host flush
 * that ends a submission (hk_queue.c).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/lib/agx_chip.h"
#include "drm-uapi/asahi_drm.h"
#include "util/u_dynarray.h"
#include "hk_cs_root_size.h"

/* Writes a stream of one record and the terminator at the start of its
 * storage, a BO the queue keeps. The hardware reads past the end of a stream,
 * which is why every stream keeps HK_CS_SPACE_PADDING bytes after it
 * (hk_ensure_cs_has_space()). agx_bo_create() can return a BO from the BO
 * cache with the bytes its last user wrote, so those bytes are cleared, as
 * submit packing clears them after every stream it terminates
 * (hk_pack_runs()). The stream ends at record_size +
 * AGX3_CDM_STREAM_TERMINATE_SIZE. */
static inline bool
hk_queue_flush_stream_write(enum agx_chip chip, uint8_t *storage, size_t size,
                            const uint8_t *record, size_t record_size)
{
   const size_t length = record_size + AGX3_CDM_STREAM_TERMINATE_SIZE;
   if (!storage || !record || length > size ||
       HK_CS_SPACE_PADDING > size - length)
      return false;

   memcpy(storage, record, record_size);
   if (!agx3_cdm_stream_terminate_pack(chip, storage + record_size))
      return false;

   memset(storage + length, 0, HK_CS_SPACE_PADDING);
   return true;
}

/* The lists a shader-less compute command declares before itself: an empty
 * compute resource list on M5 kernels that take one, and an empty attachment
 * set where compute commands carry attachment hints
 * (asahi_append_compute_attachments()). An attachment set applies to every
 * later compute command of the submission until another replaces it, so
 * without one the command would take the set of the dispatch before it. */
static inline void
hk_queue_append_flush_lists(enum agx_chip chip, uint64_t features,
                            struct util_dynarray *payload)
{
   const struct drm_asahi_cmd_header empty = {
      .size = 0,
      .vdm_barrier = DRM_ASAHI_BARRIER_NONE,
      .cdm_barrier = DRM_ASAHI_BARRIER_NONE,
   };

   if (agx_chip_g17_m5(chip) &&
       (features & DRM_ASAHI_FEATURE_COMPUTE_RESOURCES)) {
      struct drm_asahi_cmd_header resources = empty;
      resources.cmd_type = DRM_ASAHI_SET_COMPUTE_RESOURCES;
      util_dynarray_append(payload, resources);
   }

   if (agx_chip_native_or_g18g(chip)) {
      struct drm_asahi_cmd_header attachments = empty;
      attachments.cmd_type = DRM_ASAHI_SET_COMPUTE_ATTACHMENTS;
      util_dynarray_append(payload, attachments);
   }
}

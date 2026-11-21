/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS flag queue_submit_recordings() emits
 * for compute commands, through its own fill path (hk_queue_fill_cdm_command)
 * and its render stamp note.
 */
#undef NDEBUG
#include <assert.h>
#include <string.h>
#include "../hk_queue.c"

/* No queue point is reached unless the test says so (signalled_value). */
int
drmSyncobjQuery(int fd, uint32_t *handles, uint64_t *points,
                uint32_t handle_count)
{
   return -1;
}

static struct hk_device dev;
static struct hk_queue queue;

static void
init_cs(struct hk_cs *cs, enum hk_cs_type type)
{
   memset(cs, 0, sizeof(*cs));
   cs->type = type;
}

/* As the render branch of queue_submit_recordings() records its stamps. */
static void
submit_render(struct hk_cs *cs)
{
   hk_query_stamps_note_cs(&dev, &queue, queue.drm.timeline_value, cs);
}

static bool
submit_compute(struct hk_cs *cs)
{
   struct drm_asahi_cmd_compute cmd;
   hk_queue_fill_cdm_command(&dev, &queue, cs, &cmd);
   return cmd.flags & DRM_ASAHI_COMPUTE_NO_QUERY_ACCESS;
}

/* vkCmdCopyQueryPoolResults() of a timestamp query at offset_B. */
static void
init_copy(struct hk_cs *cs, uint32_t handle, uint64_t offset_B)
{
   init_cs(cs, HK_CS_CDM);
   cs->query_access = true;
   hk_cs_note_query_range(cs, handle, offset_B, sizeof(uint64_t));
}

int
main(void)
{
   dev.dev.fd = -1;
   dev.dev.params.features = DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS;
   hk_query_stamps_init(&dev);
   queue.drm.syncobj = 1;
   queue.drm.signalled_value = 4;
   queue.drm.timeline_value = 5;

   /* One submission: a render stamps query 8 (handle 7, offset 64), then a
    * compute stream copies that query. The copy must wait. */
   struct hk_cs render, copy, other, plain;
   init_cs(&render, HK_CS_VDM);
   render.timestamp.end =
      (struct agx_timestamp_req){.handle = 7, .offset_B = 64};
   submit_render(&render);

   init_copy(&copy, 7, 64);
   assert(!submit_compute(&copy));
   assert(copy.query_access);

   /* A copy of another query of the same pool, in the same submission. */
   init_copy(&other, 7, 72);
   assert(submit_compute(&other));

   /* A stream touching no query word. */
   init_cs(&plain, HK_CS_CDM);
   assert(submit_compute(&plain));

   /* A compute stamp earlier in the submission conflicts too, and the
    * producer itself (which touches no recorded word) keeps the flag. */
   struct hk_cs producer, reset;
   init_cs(&producer, HK_CS_CDM);
   producer.timestamp.end =
      (struct agx_timestamp_req){.handle = 8, .offset_B = 0};
   assert(submit_compute(&producer));
   init_copy(&reset, 8, 0);
   assert(!submit_compute(&reset));

   /* The next submission: point 5 is not reached yet, then it is. */
   queue.drm.timeline_value = 6;
   assert(!submit_compute(&copy));
   p_atomic_set(&queue.drm.signalled_value, 5);
   assert(submit_compute(&copy));
   assert(submit_compute(&reset));

   /* An availability store alone records its word without query_access,
    * which stays "copy or reset" for sibling kicks and packing. */
   struct hk_cs avail;
   init_cs(&avail, HK_CS_CDM);
   hk_cs_note_query_range(&avail, 7, 128, 4);
   assert(!avail.query_access && avail.query_access_ranges.size);
   assert(submit_compute(&avail));

   /* Recording paths note words only where the kernel accepts the flag. */
   struct hk_cmd_buffer cmd;
   memset(&cmd, 0, sizeof(cmd));
   cmd.vk.base.device = &dev.vk;
   struct hk_cs rec;
   init_cs(&rec, HK_CS_CDM);
   hk_cmd_note_query_range(&cmd, &rec, 7, 64, 8);
   hk_cmd_note_imm_write_query_range(&cmd, &rec, 7, 72, 8);
   assert(rec.query_access_ranges.size && rec.imm_write_query_ranges.size);
   util_dynarray_fini(&rec.query_access_ranges);
   util_dynarray_fini(&rec.imm_write_query_ranges);

   /* Without the kernel feature nothing is recorded, query_access keeps
    * marking copies and resets, and the flag is never emitted. */
   dev.dev.params.features = 0;
   init_cs(&rec, HK_CS_CDM);
   rec.query_access = true;
   hk_cmd_note_query_range(&cmd, &rec, 7, 64, 8);
   hk_cmd_note_imm_write_query_range(&cmd, &rec, 7, 72, 8);
   assert(!rec.query_access_ranges.data && !rec.imm_write_query_ranges.data);
   assert(!rec.query_access_untracked && !rec.imm_write_query_untracked);
   assert(rec.query_access);
   assert(!submit_compute(&plain));
   assert(!submit_compute(&other));

   util_dynarray_fini(&copy.query_access_ranges);
   util_dynarray_fini(&other.query_access_ranges);
   util_dynarray_fini(&reset.query_access_ranges);
   util_dynarray_fini(&avail.query_access_ranges);
   hk_query_stamps_finish(&dev);
   puts("hk_query_no_access: ok");
   return 0;
}

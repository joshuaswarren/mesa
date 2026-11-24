/*
 * Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 *
 * Opt-in frame-anatomy profiler: logs one line per submitted control stream
 * (render or compute) with framebuffer/load-store state, submitted barriers,
 * and firmware GPU timestamps.
 *
 * Controlled by HK_FRAME_ANATOMY=<path|"-" for stderr>,
 * HK_FRAME_ANATOMY_EVERY=<N> (default 200), and
 * HK_FRAME_ANATOMY_TS=<"all"|"edges"|"none"> (default "all").
 *
 * Timing caveats:
 * - "all" makes each cross-engine step wait on the host first (on M5 the
 *   kernel turns every timestamped command into a host-prefix producer;
 *   measured 0.53 -> 3.41 ms/frame on M4 Max); use "all" for intra-engine
 *   shape and "edges"/"none" for end-to-end timing.
 * - Start timestamps precede barrier resolution, so durations include waits.
 * - Dense sampling adds overhead (HK_FRAME_ANATOMY_EVERY=1 cost ~23% fps on
 *   G17P); keep HK_FRAME_ANATOMY_EVERY sparse.
 * When unset, hk_device::frame_anatomy is NULL and hooks are single NULL checks.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include "util/macros.h"
#include "vulkan/vulkan_core.h"

struct hk_device;
struct hk_queue;
struct hk_cs;
struct hk_shader;
struct drm_asahi_cmd_render;
struct drm_asahi_cmd_compute;
struct hk_frame_anatomy_submit;

/* Per-render-target load/store summary carried by struct hk_bg_eot, recorded
 * only while the profiler is enabled. Bit 7 marks an attachment with an image
 * view; bits 0-1 hold the op the program actually performs (0 none, 1 clear,
 * 2 load, 3 store); bits 2-3 hold the API loadOp (0 LOAD, 1 CLEAR, 2
 * DONT_CARE, 3 NONE); bits 4-5 hold the API storeOp (0 STORE, 1 DONT_CARE,
 * 2 NONE). */
#define HK_FA_OP_PRESENT 0x80u
#define HK_FA_OP_HW(x) ((x) & 3u)
#define HK_FA_OP_LOAD(x) (((x) >> 2) & 3u)
#define HK_FA_OP_STORE(x) (((x) >> 4) & 3u)

uint8_t hk_frame_anatomy_encode_op(bool present, unsigned hw_op,
                                   VkAttachmentLoadOp load,
                                   VkAttachmentStoreOp store);

void hk_frame_anatomy_init(struct hk_device *dev);
void hk_frame_anatomy_finish(struct hk_device *dev);

/* One free-form "N <text>" line about a recording-time decision (not tied to
 * a sampled submit). Nothing unless the profiler is enabled; at most
 * HK_FRAME_ANATOMY_NOTES (default HK_FA_NOTE_LIMIT) lines per device, so a
 * hot path cannot flood the log; the end line counts the dropped ones. */
#define HK_FA_NOTE_LIMIT 4096
void hk_frame_anatomy_note(struct hk_device *dev, const char *fmt, ...)
   PRINTFLIKE(2, 3);

/* Count one draw of a render stream with fragment program fs (NULL for no
 * fragment shader). Called only while the profiler is enabled. */
void hk_frame_anatomy_note_fs(struct hk_cs *cs, const struct hk_shader *fs);
void hk_frame_anatomy_note_vs(struct hk_cs *cs, const struct hk_shader *vs);
void hk_frame_anatomy_note_m5_fragment(struct hk_cs *cs);
void hk_frame_anatomy_note_grid(struct hk_cs *cs, const struct hk_shader *cs_shader,
                                 uint64_t threads, uint64_t groups, bool indirect);

/* HK_FRAME_ANATOMY_DUMP only: write the final NIR of a compiled fragment,
 * compute, vertex or mesh program to <dir>/{fs,cs,vs,ms}-<binary hash>.nir. */
struct nir_shader;
void hk_frame_anatomy_dump_nir(struct hk_device *dev,
                               const struct hk_shader *shader,
                               struct nir_shader *nir, unsigned stage);

/* Count one dispatch of precompiled internal program `program`. */
struct agx_grid;
void hk_frame_anatomy_note_precomp(struct hk_cs *cs, unsigned program,
                                   struct agx_grid grid);

/* Returns NULL unless this submit is sampled. */
struct hk_frame_anatomy_submit *
hk_frame_anatomy_begin_submit(struct hk_device *dev, struct hk_queue *queue,
                              unsigned command_count);

/* Record one hardware command of a sampled submit. header_offset is the
 * payload offset at which the command's drm_asahi_cmd_header is appended
 * next. In "all" mode this claims the command's free timestamp slots (never
 * one a query already owns). */
void hk_frame_anatomy_render(struct hk_frame_anatomy_submit *s,
                             struct hk_device *dev, const struct hk_cs *cs,
                             struct drm_asahi_cmd_render *c,
                             unsigned cmdbuf_index, unsigned command_index,
                             size_t header_offset);
void hk_frame_anatomy_compute(struct hk_frame_anatomy_submit *s,
                              struct hk_device *dev, const struct hk_cs *cs,
                              struct drm_asahi_cmd_compute *c,
                              unsigned cmdbuf_index, unsigned command_index,
                              size_t header_offset);

/* Payload complete, before any ioctl: "edges" mode claims the timestamp
 * slots of the first and last recorded command in place. */
void hk_frame_anatomy_prepare(struct hk_frame_anatomy_submit *s,
                              struct hk_device *dev, void *payload,
                              size_t payload_size);

/* Capture each command's header barriers (and exact-prior ordinals) from the
 * payload bytes [begin, end) exactly as they are about to reach the kernel. */
void hk_frame_anatomy_note_barriers(struct hk_frame_anatomy_submit *s,
                                    const void *payload, size_t begin,
                                    size_t end);

/* Timestamp query pools bound to the kernel, whose stamps the profiler reads
 * back for commands that end on a query instead of a profiler stamp. */
void hk_frame_anatomy_bind_pool(struct hk_device *dev, uint32_t handle,
                                const void *map, uint64_t size);
void hk_frame_anatomy_unbind_pool(struct hk_device *dev, uint32_t handle);

/* Queue the records for readback once queue->drm.timeline_value completes;
 * a failed result retires them immediately without timestamps. */
void hk_frame_anatomy_end_submit(struct hk_frame_anatomy_submit *s,
                                 struct hk_device *dev, struct hk_queue *queue,
                                 VkResult result);

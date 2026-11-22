/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct drm_asahi_cmd_render;
struct drm_asahi_cmd_compute;

struct asahi_g17p_stream_memory {
   void *data;
   bool (*read)(void *data, uint64_t address, void *dst, size_t size);
   bool (*owned)(void *data, uint64_t address, size_t size);
   bool (*span)(void *data, uint64_t address, size_t *available);
   bool verbose;
   /*
    * Follow the surface address inside a resource pair's texture descriptor.
    *
    * OFF BY DEFAULT, and it must stay that way. The payload a resource pair is
    * read out of also carries push constants, attribute divisors and indirect
    * arguments; decoding those sixteen bytes as a pointer pair and then
    * decoding what they point at as a texture descriptor produces a plausible
    * looking unowned surface on healthy streams. Measured on this tree's own
    * asahi suite: 69 reports across 180 passing tests, concentrated in the
    * meta blit/copy paths and in compute-only conditional-rendering cases that
    * bind no texture at all. A check that cannot tell a pointer from an
    * integer must not speak by default.
    *
    * 1 reports, 2 rejects. Point it at one reproduction, read the decode, and
    * judge it by hand.
    */
   unsigned strict_descriptors;
};

struct asahi_g17p_stream_stats {
   unsigned draws;
   unsigned indexed_draws;
   unsigned tessellated_draws;
   unsigned state_updates;
   unsigned elided_bind_tails;
   unsigned incremental_fs_ppp_updates;
   /* Every VDM memory-barrier record in the stream, and the subset of them
    * that separate one draw body from a later one.  hk_cs_init_graphics()
    * always opens a stream with one barrier before any draw exists, so only
    * the second number describes inter-draw ordering. */
   unsigned barriers;
   unsigned inter_draw_barriers;
   /* State republications that follow a draw without a barrier in between.
    * This is the hazard the inter-draw barrier exists to close, so a test can
    * assert it is zero rather than only that some barriers were emitted. */
   unsigned unbarriered_state_republications;
   unsigned links;
   unsigned viewport_count;
   unsigned max_instance_count;
   uint32_t primitive_mask;
   uint8_t index_size_mask;
   bool clear_only;
   bool body_only;
   bool restart_enabled;
   bool restart_disabled;
   bool indirect_nonindexed;
   bool indirect_indexed;
   bool varying_fs;
   bool no_varying_fs;
   bool viewport_changed;
   bool multi_viewport;
   bool two_viewport;
   bool four_viewport;
   bool scissor_changed;
   bool cull_enabled;
   bool depth_enabled;
   bool stencil_enabled;
   unsigned stencil_state_updates;
   bool blend_enabled;
   bool rasterizer_discard;
   bool primary_tri_merge_disable;
   bool secondary_tri_merge_disable;
   bool layer_output;
   uint64_t fragment_code;
};

struct asahi_g17p_stream_config {
   uint64_t root;
   uint64_t scissor_base;
   uint64_t depth_bias_base;
};

struct asahi_g17p_compute_stats {
   unsigned launches;
   /* Direct launches right after a direct launch, with no barrier between. */
   unsigned unbarriered_launches;
   unsigned two_dimensional_launches;
   /* Barriers paired with launches; ordered pre-launch flushes are separate. */
   unsigned barriers;
   unsigned prologue_barriers;
   unsigned strong_barriers;
   unsigned sstt_flushes;
   bool terminated;
};

bool asahi_g17p_stream_validate(
   const struct asahi_g17p_stream_memory *memory,
   const struct asahi_g17p_stream_config *config,
   struct asahi_g17p_stream_stats *stats);

bool asahi_g17p_render_aux_validate(
   const struct asahi_g17p_stream_memory *memory,
   const struct drm_asahi_cmd_render *render);

bool asahi_g17p_compute_validate(
   const struct asahi_g17p_stream_memory *memory,
   const struct drm_asahi_cmd_compute *compute,
   struct asahi_g17p_compute_stats *stats);

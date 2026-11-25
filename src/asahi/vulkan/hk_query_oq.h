/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "asahi/lib/agx_chip.h"

/*
 * Native AGX3 occlusion availability.
 *
 * On G17P and M5, the kernel writes the vertex-start, vertex-end, and
 * fragment-start render timestamp slots at render retirement (after cache
 * maintenance makes occlusion counts visible and before signaling). Fragment
 * end is written by firmware and may land before counts are visible (a G17P
 * depth-only pass read 0 from a stamped query that held the full count after
 * submission completion), so it is never used for availability. When a render
 * has free retirement slots and no subsequent compute work in the command
 * buffer depends on it, availability is stamped via the render instead of
 * emitting a post-graphics compute store.
 */
/* Supported only on G17P and M5, where the kernel writes these slots at
 * retirement; G15 is excluded because its firmware writes the vertex-end slot
 * when tiling finishes, before fragment occlusion counts land. */
bool hk_oq_render_stamps_supported(enum agx_chip chip);

struct hk_oq_stamp {
   uint32_t handle;
   uint32_t offset_B;
   uint32_t render_seq;
};

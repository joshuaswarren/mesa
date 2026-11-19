/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>

#include "util/macros.h"

#ifdef __cplusplus
extern "C" {
#endif

/* clang-format off */
enum agx_compiler_dbg {
   /* bit 0 unused */
   AGX_DBG_SHADERS     = BITFIELD_BIT(1),
   AGX_DBG_SHADERDB    = BITFIELD_BIT(2),
   AGX_DBG_VERBOSE     = BITFIELD_BIT(3),
   AGX_DBG_INTERNAL    = BITFIELD_BIT(4),
   AGX_DBG_NOVALIDATE  = BITFIELD_BIT(5),
   AGX_DBG_NOOPT       = BITFIELD_BIT(6),
   AGX_DBG_WAIT        = BITFIELD_BIT(7),
   AGX_DBG_NOPREAMBLE  = BITFIELD_BIT(8),
   AGX_DBG_DEMAND      = BITFIELD_BIT(9),
   AGX_DBG_NOSCHED     = BITFIELD_BIT(10),
   AGX_DBG_SPILL       = BITFIELD_BIT(11),
   AGX_DBG_NOPROMOTE   = BITFIELD_BIT(12),
   AGX_DBG_LATECOV     = BITFIELD_BIT(13),
   AGX_DBG_REFUSE      = BITFIELD_BIT(14),
   AGX_DBG_BINARIES    = BITFIELD_BIT(15),
};
/* clang-format on */

uint64_t agx_get_compiler_debug(void);

/* Why the last agx_compile_shader_nir() on this thread returned false
 * ("" when it succeeded), for the driver's "cannot encode" error. */
const char *agx_last_compile_refusal(void);

/* The G18G paths below used to be environment-selectable for A/B bring-up
 * and are now unconditional on G18G. Only agx3_pack_binary.c still calls
 * these; fold them into its call sites and delete them. Each one names the
 * path it gated:
 *
 *  - msaa4: G18G 2x/4x multisampling (tile layouts, sample-mask and tile
 *    lowering, epilogs, background/end-of-tile programs);
 *  - native1109: single-sample RG32 target pairs and partial RGBA8 tile
 *    stores, and a single non-zero layer of a multisample D32 array;
 *  - native1121: 2x/4x alpha-to-coverage and sample-shaded framebuffer-fetch
 *    draws and their tile and sample-mask forms;
 *  - native1128: 2x/4x per-sample discard (separate sample-mask kill and
 *    update words);
 *  - native1130 / native_mixed: single-sample mixed-format tiles laid out in
 *    attachment-index order with per-format tile stores.
 */
static inline bool agx_g18_msaa4_enabled(void) { return true; }
static inline bool agx_g18_native1109_enabled(void) { return true; }
static inline bool agx_g18_native1121_enabled(void) { return true; }
static inline bool agx_g18_native1128_enabled(void) { return true; }
static inline bool agx_g18_native1130_enabled(void) { return true; }
static inline bool agx_g18_native_mixed_enabled(void) { return true; }

/* Fragment-structure code generation choices that remain selectable. */
enum agx_fs_structure_flag {
   /* Compile the colour epilog into the fragment shader at link time, as
    * Apple's compiler blends inline. Honeykrisp keeps the main shader's NIR
    * and backend key; at fast-link it appends the draw's epilog, turns every
    * export / load_exported pair into SSA values and compiles one fragment
    * program. It falls back to the split main + epilog whenever the
    * monolithic compile would change any uniform, rodata, varying or
    * preamble layout the draw path takes from the API shader, or fails to
    * compile. */
   AGX_FS_STRUCTURE_INLINE_BLEND = BITFIELD_BIT(4),
};

uint32_t agx_get_fs_structure_flags(void);

#ifdef __cplusplus
} /* extern C */
#endif

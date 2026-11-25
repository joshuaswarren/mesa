/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "asahi/compiler/agx_compile.h"

/* Whether two fragment parts carry the same constant program. Equal sizes
 * and register counts do not prove that their uniform values agree, so the
 * code and its frozen live-ins must agree too. Returns the first difference,
 * or NULL. */
static inline const char *
hk_g17p_cprog_mismatch(const struct agx_shader_part *a,
                       const struct agx_shader_part *m)
{
#define HK_CPROG_FIELD(x)                                                      \
   if (a->info.x != m->info.x)                                                 \
      return #x;
   HK_CPROG_FIELD(has_preamble)
   HK_CPROG_FIELD(preamble_size)
   HK_CPROG_FIELD(nr_preamble_gprs)
   HK_CPROG_FIELD(agx3_preamble_uniform_inputs)
   HK_CPROG_FIELD(agx3_preamble_tps_inputs)
   HK_CPROG_FIELD(preamble_scratch_size)
   HK_CPROG_FIELD(agx3_preamble_staging_known)
   HK_CPROG_FIELD(agx3_preamble_staging_halfs)
   HK_CPROG_FIELD(agx3_preamble_inputs.count)
   HK_CPROG_FIELD(agx3_preamble_inputs.storage_bytes)
#undef HK_CPROG_FIELD
   if (!a->info.has_preamble)
      return NULL;
   if (memcmp(a->info.agx3_preamble_inputs.halves,
              m->info.agx3_preamble_inputs.halves,
              sizeof(a->info.agx3_preamble_inputs.halves)))
      return "preamble inputs";
   if (!a->binary || !m->binary || !a->info.preamble_size ||
       a->info.preamble_offset > a->info.binary_size ||
       m->info.preamble_offset > m->info.binary_size ||
       a->info.preamble_size > a->info.binary_size - a->info.preamble_offset ||
       m->info.preamble_size > m->info.binary_size - m->info.preamble_offset)
      return "preamble extent";
   if (memcmp((const uint8_t *)a->binary + a->info.preamble_offset,
              (const uint8_t *)m->binary + m->info.preamble_offset,
              a->info.preamble_size))
      return "preamble code";
   return NULL;
}

/* Whether an inline-blend fragment program (main and colour epilog compiled
 * as one) carries a constant program of its own rather than the API
 * shader's: either the API shader has none, or the program's is a different
 * one because it also took uniform work of the inlined epilog.
 *
 * A colour read from a uniform buffer (zink's default uniform block) is in
 * the API shader's constant program already. Inlining the blend leaves its
 * clamps, 1 - alpha and source products uniform too, and the backend moves
 * them into the program's constant program. Such a program launches with
 * its own layout (hk_linked_shader::g17p_layout): those terms run once per
 * draw on the GPU, reading the buffer when the draw executes, instead of
 * once per pixel. A program whose constant program is the API shader's keeps
 * the API shader's layout. */
static inline bool
hk_g17p_inline_blend_own_layout(const struct agx_shader_part *api,
                                const struct agx_shader_part *part)
{
   /* The draw launches the program's constant program from its own binary,
    * so that program must lie inside it. */
   const struct agx_shader_info *info = &part->info;
   if (!info->has_preamble || !part->binary || !info->preamble_size ||
       info->preamble_offset > info->binary_size ||
       info->preamble_size > info->binary_size - info->preamble_offset)
      return false;

   return !api->info.has_preamble || hk_g17p_cprog_mismatch(api, part);
}

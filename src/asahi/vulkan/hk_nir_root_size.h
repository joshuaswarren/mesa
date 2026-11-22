/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "nir.h"
#include "util/macros.h"

/* Required prefix of the root table. Unknown/escaping accesses require full_size. */
unsigned hk_nir_root_size(const nir_shader *nir, unsigned full_size);

/* Compute the trimmed root descriptor upload size for a shader stage.
 * Uniform lowering loads the root pointer and set bases separately, so retain
 * that prefix.  Hardware VS and FS prologs/epilogs access only the draw arm
 * preceding push constants, so flooring graphics stages at push_offset
 * preserves every driver-synthesized root load while trimming unused
 * push-constant, dynamic-buffer, and trailing tail bytes. */
static inline uint32_t
hk_shader_root_descriptor_size(const nir_shader *nir,
                               mesa_shader_stage sw_stage,
                               unsigned set_count, unsigned sets_offset,
                               unsigned push_offset, unsigned full_size)
{
   unsigned bytes = hk_nir_root_size(nir, full_size);
   bytes = MAX2(bytes, sets_offset + set_count * (unsigned)sizeof(uint64_t));
   if (sw_stage != MESA_SHADER_COMPUTE)
      bytes = MAX2(bytes, push_offset);
   return MIN2(ALIGN_POT(bytes, 16), full_size);
}

/* Active graphics root prefix: maximum of the bound VS and optional FS
 * footprints, falling back to full_size whenever either stage lacks a proven
 * trimmed size. */
static inline size_t
hk_gfx_root_size_from_stages(uint32_t vs_size, bool has_fs, uint32_t fs_size,
                             size_t full_size)
{
   if (!vs_size)
      return full_size;

   size_t size = vs_size;
   if (has_fs) {
      if (!fs_size)
         return full_size;
      size = MAX2(size, (size_t)fs_size);
   }

   return MIN2(size, full_size);
}

/* Force a root re-upload when a newly bound shader needs more of the root
 * table than the last upload published. */
static inline bool
hk_root_upload_needs_growth(uint32_t uploaded_root_size,
                            size_t required_root_size)
{
   return uploaded_root_size < required_root_size;
}

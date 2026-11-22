/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * A compute workgroup of at most 32 invocations is one AGX subgroup, so its
 * workgroup scope and subgroup scope name the same invocations. Narrow its
 * workgroup execution and workgroup memory scopes to subgroup scope on
 * barriers that order only shared memory (or nothing): the backend then
 * emits the SIMD-group shared-memory fence instead of the threadgroup
 * rendezvous. Barriers that order device memory or images keep their
 * scopes. Metal's compiler removes these barriers from 32-wide variants of
 * the same kernels on M5 (Geekbench's particle kernel).
 */

#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"

static bool
narrow(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_barrier ||
       (nir_intrinsic_memory_modes(intr) & ~nir_var_mem_shared))
      return false;

   bool progress = false;
   if (nir_intrinsic_execution_scope(intr) == SCOPE_WORKGROUP) {
      nir_intrinsic_set_execution_scope(intr, SCOPE_SUBGROUP);
      progress = true;
   }
   if (nir_intrinsic_memory_scope(intr) == SCOPE_WORKGROUP) {
      nir_intrinsic_set_memory_scope(intr, SCOPE_SUBGROUP);
      progress = true;
   }
   return progress;
}

bool
agx_nir_narrow_single_subgroup_barriers(nir_shader *shader)
{
   if (shader->info.stage != MESA_SHADER_COMPUTE ||
       shader->info.workgroup_size_variable)
      return false;

   const unsigned invocations = shader->info.workgroup_size[0] *
                                shader->info.workgroup_size[1] *
                                shader->info.workgroup_size[2];
   if (!invocations || invocations > 32)
      return false;

   return nir_shader_intrinsics_pass(shader, narrow, nir_metadata_all, NULL);
}

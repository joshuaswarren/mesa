/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Verify that every Gen1 acceleration-structure precompiled kernel (including
 * rebase and instance-record restore used by compaction/clone/serialization)
 * is compiled for both G17P and G16G with valid launch metadata and zero
 * scratch spills.
 */
#include "asahi/compiler/agx_compile.h"
#include "libagx.h"
#include "libagx_shaders.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>

static void
check_kernel(const uint32_t *bin, const char *name, uint32_t wg_x)
{
   const struct agx_precompiled_kernel_info *info = (const void *)bin;
   assert(info && name);
   assert(info->main_size && info->nr_gprs);
   assert(info->main_offset <= info->binary_size);
   assert(info->main_size <= info->binary_size - info->main_offset);
   assert(info->agx3_staging_known);
   assert(info->nr_preamble_gprs == 0 && info->preamble_size == 0);
   assert(info->workgroup_size[0] == wg_x);
   assert(info->workgroup_size[1] == 1 && info->workgroup_size[2] == 1);
   assert(info->printf_uniform == 0);
   assert(info->scratch_size == 0);
}

int
main(void)
{
   static const struct {
      enum libagx_program prog;
      const char *name;
      uint32_t wg_x;
   } kernels[] = {
      {LIBAGX_GEN1_AS_PROPERTY, "libagx_gen1_as_property", 1},
      {LIBAGX_GEN1_AS_COPY, "libagx_gen1_as_copy", 32},
      {LIBAGX_GEN1_AS_REBASE, "libagx_gen1_as_rebase", 32},
      {LIBAGX_GEN1_AS_RESTORE_RECORDS, "libagx_gen1_as_restore_records", 32},
      {LIBAGX_GEN1_AS_SERIALIZE, "libagx_gen1_as_serialize", 32},
      {LIBAGX_GEN1_AS_DESERIALIZE, "libagx_gen1_as_deserialize", 32},
      {LIBAGX_GEN1_AS_DESERIALIZE_REFERENCES, "libagx_gen1_as_deserialize_references", 32},
      {LIBAGX_GEN1_AS_EMPTY, "libagx_gen1_as_empty", 1},
   };

   for (unsigned i = 0; i < sizeof(kernels) / sizeof(kernels[0]); ++i) {
      check_kernel(libagx_g17p[kernels[i].prog], kernels[i].name, kernels[i].wg_x);
      check_kernel(libagx_g16g[kernels[i].prog], kernels[i].name, kernels[i].wg_x);
   }

   printf("hk_gen1_as_precomp: ok (%zu kernels x G17P/G16G)\n",
          sizeof(kernels) / sizeof(kernels[0]));
   return 0;
}

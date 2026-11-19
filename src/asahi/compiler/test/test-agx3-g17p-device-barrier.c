/*
 * Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 */

#include "agx3_pack.h"
#include "agx_compile.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* How many device fences (AGX3_BARRIER_SCOPE_DEVICE_FENCE) a compute shader
 * that writes global memory, runs a workgroup barrier with `memory_scope` over
 * global memory and reads global memory compiles to on `chip`. */
static int
device_fences(enum agx_chip chip, mesa_scope memory_scope)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "workgroup barrier");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *id = nir_u2u64(&b, nir_load_local_invocation_index(&b));
   nir_def *addr = nir_iadd(&b, base, nir_ishl_imm(&b, id, 2));
   nir_store_global(&b, nir_load_local_invocation_index(&b), addr,
                    .align_mul = 4);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = memory_scope,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
   nir_def *v = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, addr, 4096),
                                .align_mul = 4);
   nir_store_global(&b, v, nir_iadd_imm(&b, addr, 8192), .align_mul = 4);
   agx_preprocess_nir_native(b.shader, chip);

   struct agx_shader_key key = {
      .dev.chip = chip,
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .reserved_preamble = 8,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   int count = -1;
   struct agx3_encoded_scoped_barrier fence;
   if (agx_compile_shader_nir(b.shader, &key, &part) && part.binary &&
       agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3B,
                                AGX3_BARRIER_SCOPE_DEVICE_FENCE, &fence)) {
      const uint8_t *code = part.binary;
      count = 0;
      for (unsigned i = 0; i + fence.length <= part.info.binary_size; ++i)
         count += !memcmp(code + i, fence.bytes, fence.length);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return count;
}

/* A workgroup barrier at device scope must make the workgroup's global writes
 * visible to other workgroups. On G17P the rendezvous's device-memory
 * operation alone does not (cross-workgroup test of aurora-silicon/mesa#268:
 * 72-95% of words read stale), so a device fence follows it, as on every
 * AGX3B chip (G16G here). Workgroup memory scope keeps the rendezvous alone. */
int
main(void)
{
   const int g17p_device = device_fences(AGX_CHIP_G17P, SCOPE_DEVICE);
   const int g17p_queue = device_fences(AGX_CHIP_G17P, SCOPE_QUEUE_FAMILY);
   const int g17p_workgroup = device_fences(AGX_CHIP_G17P, SCOPE_WORKGROUP);
   const int g16g_device = device_fences(AGX_CHIP_G16G, SCOPE_DEVICE);

   if (g17p_device != 1 || g17p_queue != 1 || g17p_workgroup != 0 ||
       g16g_device != 1) {
      fprintf(stderr,
              "device fences: G17P device %d, queue family %d, workgroup %d; "
              "G16G device %d\n",
              g17p_device, g17p_queue, g17p_workgroup, g16g_device);
      return 1;
   }

   puts("agx3_g17p_device_barrier: G17P fences device memory after a "
        "device-scope workgroup barrier");
   return 0;
}

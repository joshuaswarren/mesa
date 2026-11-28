/*
 * Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 */

#include "agx3_compose/agx3_decode.h"
#include "agx3_pack.h"
#include "agx_compile.h"
#include "asahi/isa/disasm.h"
#include "asahi/lib/agx_chip_caps.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct barrier_compile_stats {
   int fences;
   int stdev_total;
   int stdev_coherent;
};

/* Compile a compute shader that performs a non-coherent global store, a
 * workgroup barrier with `memory_scope` over global memory, a global load, and
 * a second global store, returning the device fence count and stdev f75
 * (coherent) breakdown. */
static struct barrier_compile_stats
compile_and_inspect(enum agx_chip chip, mesa_scope memory_scope,
                    bool mark_store_coherent)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "workgroup barrier");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *id = nir_u2u64(&b, nir_load_local_invocation_index(&b));
   nir_def *addr = nir_iadd(&b, base, nir_ishl_imm(&b, id, 2));
   const enum gl_access_qualifier acc =
      mark_store_coherent ? ACCESS_COHERENT : 0;
   nir_store_global(&b, nir_load_local_invocation_index(&b), addr,
                    .align_mul = 4, .access = acc);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = memory_scope,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
   nir_def *v = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, addr, 4096),
                                .align_mul = 4);
   nir_store_global(&b, v, nir_iadd_imm(&b, addr, 8192), .align_mul = 4,
                    .access = acc);
   agx_preprocess_nir_native(b.shader, chip);

   struct agx_shader_key key = {
      .dev.chip = chip,
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .reserved_preamble = 8,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   struct barrier_compile_stats out = {-1, -1, -1};
   struct agx3_encoded_scoped_barrier fence;
   if (agx_compile_shader_nir(b.shader, &key, &part) && part.binary &&
       agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3B,
                                AGX3_BARRIER_SCOPE_DEVICE_FENCE, &fence)) {
      const uint8_t *code = part.binary;
      out.fences = 0;
      out.stdev_total = 0;
      out.stdev_coherent = 0;
      for (unsigned i = 0; i + fence.length <= part.info.binary_size; ++i)
         out.fences += !memcmp(code + i, fence.bytes, fence.length);

      for (size_t off = 0; off < part.info.binary_size;) {
         const signed n = agx3b_instr_length(code + off,
                                             part.info.binary_size - off,
                                             NULL);
         if (n <= 0) {
            out.stdev_total = -1;
            break;
         }
         uint64_t fields[AGX3_MAX_FIELDS];
         if (agx3_decode(&agx3b_table, code + off, (unsigned)n, AGX3_MN_stdev,
                         fields) >= 0) {
            out.stdev_total++;
            if (fields[75] == 1)
               out.stdev_coherent++;
         }
         off += (size_t)n;
      }
   }
   free(part.binary);
   ralloc_free(b.shader);
   return out;
}

/* A workgroup barrier at device scope must make the workgroup's global writes
 * visible to other workgroups. On G17P the rendezvous's device-memory
 * operation alone does not (cross-workgroup test of aurora-silicon/mesa#268:
 * 72-95% of words read stale), so a device fence follows it, as on every
 * AGX3B chip (G16G here). Workgroup memory scope keeps the rendezvous alone.
 *
 * In addition, chips carrying AGX_CAP_COMPUTE_COHERENT_STORE (G17P) promote
 * non-coherent compute device stores to stdev f75=1 while keeping the device
 * fence intact, whereas chips without the cap (G16G) emit f75=0 unless NIR
 * explicitly requests ACCESS_COHERENT. */
int
main(void)
{
   const struct barrier_compile_stats g17p_device =
      compile_and_inspect(AGX_CHIP_G17P, SCOPE_DEVICE, false);
   const struct barrier_compile_stats g17p_queue =
      compile_and_inspect(AGX_CHIP_G17P, SCOPE_QUEUE_FAMILY, false);
   const struct barrier_compile_stats g17p_workgroup =
      compile_and_inspect(AGX_CHIP_G17P, SCOPE_WORKGROUP, false);
   const struct barrier_compile_stats g16g_device =
      compile_and_inspect(AGX_CHIP_G16G, SCOPE_DEVICE, false);
   const struct barrier_compile_stats g16g_coherent =
      compile_and_inspect(AGX_CHIP_G16G, SCOPE_DEVICE, true);

   if (g17p_device.fences != 1 || g17p_queue.fences != 1 ||
       g17p_workgroup.fences != 0 || g16g_device.fences != 1 ||
       g16g_coherent.fences != 1) {
      fprintf(stderr,
              "device fences: G17P device %d, queue family %d, workgroup %d; "
              "G16G device %d, G16G coherent %d\n",
              g17p_device.fences, g17p_queue.fences, g17p_workgroup.fences,
              g16g_device.fences, g16g_coherent.fences);
      return 1;
   }

   if (!agx_chip_has_cap(AGX_CHIP_G17P, AGX_CAP_COMPUTE_COHERENT_STORE) ||
       agx_chip_has_cap(AGX_CHIP_G16G, AGX_CAP_COMPUTE_COHERENT_STORE) ||
       g17p_device.stdev_total < 2 ||
       g17p_device.stdev_coherent != g17p_device.stdev_total ||
       g16g_device.stdev_total < 2 || g16g_device.stdev_coherent != 0 ||
       g16g_coherent.stdev_total < 2 ||
       g16g_coherent.stdev_coherent != g16g_coherent.stdev_total) {
      fprintf(stderr,
              "stdev coherent policy: G17P %d/%d, G16G non-coherent %d/%d, "
              "G16G coherent %d/%d\n",
              g17p_device.stdev_coherent, g17p_device.stdev_total,
              g16g_device.stdev_coherent, g16g_device.stdev_total,
              g16g_coherent.stdev_coherent, g16g_coherent.stdev_total);
      return 1;
   }

   puts("agx3_g17p_device_barrier: G17P fences device memory after a "
        "device-scope workgroup barrier and packs compute stdev coherent "
        "policy via AGX_CAP_COMPUTE_COHERENT_STORE");
   return 0;
}

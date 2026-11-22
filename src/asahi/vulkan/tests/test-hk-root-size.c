/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_nir_root_size.h"
#include "nir_builder.h"
#undef NDEBUG
#include <assert.h>
#include <stdio.h>

static nir_builder
builder(void)
{
   return nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "root-size");
}

static void
load(nir_builder *b, nir_def *addr, unsigned components, unsigned bits)
{
   nir_load_global_constant(b, components, bits, addr, .align_mul = bits / 8);
}

int main(void)
{
   const unsigned full = 8192;
   nir_builder b = builder();
   assert(hk_nir_root_size(b.shader, full) == 0);
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 832), 4, 32);
   assert(hk_nir_root_size(b.shader, full) == 848);
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 40), 1, 64);
   assert(hk_nir_root_size(b.shader, full) == 848);
   ralloc_free(b.shader);

   b = builder();
   nir_def *addr = nir_iadd_imm(&b, nir_load_root_agx(&b), 32);
   load(&b, nir_iadd_imm(&b, addr, 64), 2, 32);
   assert(hk_nir_root_size(b.shader, full) == 104);
   ralloc_free(b.shader);

   b = builder();
   addr = nir_iadd(&b, nir_load_root_agx(&b), nir_u2u64(&b, nir_load_local_invocation_index(&b)));
   load(&b, addr, 1, 32);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);

   b = builder();
   nir_store_global(&b, nir_imm_int(&b, 0), nir_load_root_agx(&b), .align_mul = 4, .write_mask = 1);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);

   for (int offset = -1; offset <= (int)full; offset += full + 1) {
      b = builder();
      load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), offset), 4, 32);
      assert(hk_nir_root_size(b.shader, full) == full);
      ralloc_free(b.shader);
   }
   b = builder();
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), full - 4), 4, 32);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);

   b = builder();
   addr = nir_load_root_agx(&b);
   for (unsigned i = 0; i < 10; ++i)
      addr = nir_iadd_imm(&b, addr, 8);
   load(&b, addr, 1, 32);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);
   b = builder();
   nir_def *reflected = nir_load_global_constant(&b, 1, 64,
      nir_load_root_agx(&b), .align_mul = 8);
   load(&b, nir_iadd_imm(&b, reflected, 4096), 1, 32);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);

   /* Test hk_shader_root_descriptor_size(): compute trims to the highest
    * read / set-base prefix, whereas graphics VS/FS are floored at push_offset
    * so driver-synthesized prolog/epilog loads in the draw arm stay covered. */
   const unsigned sets_off = 32;
   const unsigned push_off = 512;
   const mesa_shader_stage gfx_stages[] = {
      MESA_SHADER_VERTEX,
      MESA_SHADER_FRAGMENT,
   };

   /* Compute with 1 set and no root loads stays below push_off. */
   b = builder();
   assert(hk_shader_root_descriptor_size(b.shader, MESA_SHADER_COMPUTE, 1,
                                         sets_off, push_off, full) == 48);
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 60), 1, 32);
   assert(hk_shader_root_descriptor_size(b.shader, MESA_SHADER_COMPUTE, 1,
                                         sets_off, push_off, full) == 64);
   ralloc_free(b.shader);

   for (unsigned i = 0; i < ARRAY_SIZE(gfx_stages); ++i) {
      /* Empty shader / low root load is floored at push_offset (aligned to 16). */
      b = nir_builder_init_simple_shader(gfx_stages[i], NULL, "gfx-root-floor");
      assert(hk_shader_root_descriptor_size(b.shader, gfx_stages[i], 2,
                                            sets_off, push_off, full) ==
             push_off);
      load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 128), 4, 32);
      assert(hk_shader_root_descriptor_size(b.shader, gfx_stages[i], 2,
                                            sets_off, push_off, full) ==
             push_off);
      /* Constant push-constant read past push_off extends and aligns to 16B. */
      load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), push_off + 20), 1, 32);
      assert(hk_shader_root_descriptor_size(b.shader, gfx_stages[i], 2,
                                            sets_off, push_off, full) ==
             push_off + 32);
      ralloc_free(b.shader);

      /* Dynamic indexing stays full-sized. */
      b = nir_builder_init_simple_shader(gfx_stages[i], NULL, "gfx-root-dyn");
      addr = nir_iadd(&b, nir_load_root_agx(&b),
                      nir_u2u64(&b, nir_load_vertex_id(&b)));
      load(&b, addr, 1, 32);
      assert(hk_shader_root_descriptor_size(b.shader, gfx_stages[i], 2,
                                            sets_off, push_off, full) == full);
      ralloc_free(b.shader);
   }

   /* Test hk_gfx_root_size_from_stages() (VS/FS max, VS-only, and fallbacks). */
   assert(hk_gfx_root_size_from_stages(512, true, 576, full) == 576);
   assert(hk_gfx_root_size_from_stages(640, true, 512, full) == 640);
   assert(hk_gfx_root_size_from_stages(512, false, 0, full) == 512);
   assert(hk_gfx_root_size_from_stages(0, true, 512, full) == full);
   assert(hk_gfx_root_size_from_stages(512, true, 0, full) == full);
   assert(hk_gfx_root_size_from_stages(full + 64, false, 0, full) == full);

   /* Test hk_root_upload_needs_growth() re-upload trigger. */
   assert(hk_root_upload_needs_growth(0, 512));
   assert(hk_root_upload_needs_growth(512, 576));
   assert(!hk_root_upload_needs_growth(576, 576));
   assert(!hk_root_upload_needs_growth(640, 512));

   puts("root footprint: compute and graphics floor, stage sizing, growth trigger, dynamic, escape, overflow, depth and reflection controls passed");
   return 0;
}

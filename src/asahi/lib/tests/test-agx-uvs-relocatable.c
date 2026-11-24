/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx3_vary_core.h"
#include "agx_compile.h"
#include "agx_uvs.h"
#include "nir_builder.h"
#include <stdio.h>

/* A vertex shader compiled without its fragment shader writes every varying
 * at a constant placeholder UVS index, including an array it indexes at run
 * time, and linking places each varying in the range of the fragment shader's
 * qualifier. The same holds with the layer written as a varying and its
 * fixed-function copy sent to zero by a root table flag.
 */

static nir_shader *
build_vs(bool known_qualifiers, bool layer)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                  &agx_nir_options, "uvs");
   nir_def *zero = nir_imm_int(&b, 0);
   nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), zero,
                    .write_mask = 0xf, .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_POS,
                                     .num_slots = 1});
   nir_store_output(&b, nir_imm_float(&b, 1), zero, .write_mask = 1,
                    .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_VAR0,
                                     .num_slots = 1});
   /* VAR1..VAR3, indexed by the vertex ID */
   nir_store_output(&b, nir_imm_vec2(&b, 2, 3),
                    nir_umod_imm(&b, nir_load_vertex_id(&b), 3),
                    .write_mask = 0x3, .src_type = nir_type_float32,
                    .io_semantics = {.location = VARYING_SLOT_VAR1,
                                     .num_slots = 3});
   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_RANGE(VARYING_SLOT_VAR0, 4);
   if (layer) {
      nir_store_output(&b, nir_load_instance_id(&b), zero, .write_mask = 1,
                       .src_type = nir_type_uint32,
                       .io_semantics = {.location = VARYING_SLOT_LAYER,
                                        .num_slots = 1});
      b.shader->info.outputs_written |= VARYING_BIT_LAYER;
   }
   b.shader->info.known_interpolation_qualifiers = known_qualifiers;
   return b.shader;
}

/* NIR_PASS validates the lowered shader in debug builds: every placeholder
 * store must be dominated by its index. */
static bool
check_relocatable(bool layer)
{
   nir_shader *s = build_vs(false, layer);
   struct agx_unlinked_uvs_layout layout = {0};
   agx_nir_lower_uvs(s, &layout, layer, true, layer ? 8 : -1);
   NIR_PASS(_, s, nir_opt_constant_folding);
   NIR_PASS(_, s, nir_opt_copy_prop);

   bool ok = layout.relocatable && layout.components[VARYING_SLOT_VAR0] == 1;
   for (unsigned i = 1; i <= 3; ++i)
      ok &= layout.components[VARYING_SLOT_VAR0 + i] == 2;

   unsigned stores = 0, layer_zero_flags = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            ok &= intr->intrinsic != nir_intrinsic_load_uvs_index_agx;
            if (intr->intrinsic == nir_intrinsic_store_uvs_agx) {
               ok &= nir_src_is_const(intr->src[1]);
               stores++;
            }
            layer_zero_flags +=
               intr->intrinsic == nir_intrinsic_load_global_constant;
         }
      }
   }
   /* Position, VAR0, and both components of each array element, then the
    * layer varying and the packed layer/viewport word. */
   ok &= stores == 4 + 1 + 3 * 2 + (layer ? 2 : 0);
   ok &= layer_zero_flags == layer;
   if (layer) {
      ralloc_free(s);
      return ok;
   }

   /* VAR0 flat and VAR2 noperspective: each moves to its range, after the
    * smooth words, and the map sends its placeholder there. */
   const uint64_t flat = VARYING_BIT_VAR(0), linear = VARYING_BIT_VAR(2);
   struct agx_varyings_vs placeholder, linked;
   uint8_t map[AGX3_UVS_RELOC_MAP_SIZE];
   agx_assign_uvs(&placeholder, &layout, 0, 0);
   agx_assign_uvs(&linked, &layout, flat, linear);
   agx_uvs_relocation_map(&layout, flat, linear, map);
   const unsigned smooth_end = layout.group_offs[UVS_VARYINGS] + 2 + 2;
   ok &= linked.slots[VARYING_SLOT_VAR0] == smooth_end;
   ok &= linked.slots[VARYING_SLOT_VAR2] == smooth_end + 1;
   for (unsigned i = 0; i <= 3; ++i) {
      const unsigned loc = VARYING_SLOT_VAR0 + i;
      for (unsigned c = 0; c < layout.components[loc]; ++c)
         ok &= map[placeholder.slots[loc] + c] == linked.slots[loc] + c;
   }

   ralloc_free(s);
   return ok;
}

/* A shader that knows its fragment shader's qualifiers is not relocated. */
static bool
check_linked(void)
{
   nir_shader *s = build_vs(true, false);
   struct agx_unlinked_uvs_layout layout = {0};
   agx_nir_lower_uvs(s, &layout, false, true, -1);
   const bool ok = !layout.relocatable;
   ralloc_free(s);
   return ok;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool ok = true;
   if (!check_relocatable(false)) {
      fprintf(stderr, "relocatable layout: FAIL\n");
      ok = false;
   }
   if (!check_relocatable(true)) {
      fprintf(stderr, "relocatable layout, layer zeroed: FAIL\n");
      ok = false;
   }
   if (!check_linked()) {
      fprintf(stderr, "linked layout: FAIL\n");
      ok = false;
   }
   glsl_type_singleton_decref();
   return ok ? 0 : 1;
}

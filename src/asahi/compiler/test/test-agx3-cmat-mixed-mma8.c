/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The 8x8 matmad form has a single input-type field, so the packer
 * refuses a matmad8 whose A and B types differ. A shader that multiplies
 * an f16 A tile by an f32 B tile (typical of f16 activations against
 * f32 dequantized weights) must be legalized in the cooperative-matrix
 * lowering by widening both operands to f32, which is exact for f16 and
 * bf16, before the instruction is emitted.
 *
 * An f16 accumulator is widened into the f32 matmad and rounded back after
 * each product; the chained f16-accumulator case checks that the rounding
 * survives optimization.
 */

#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_nir.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const struct glsl_type *
cmat_type(enum glsl_base_type elem, enum glsl_cmat_use use)
{
   const struct glsl_cmat_description desc = {
      .rows = 8,
      .cols = 8,
      .element_type = (uint16_t)elem,
      .scope = (uint8_t)SCOPE_SUBGROUP,
      .use = (uint8_t)use,
   };
   return glsl_cmat_type(&desc);
}

static nir_def *
elem_imm(nir_builder *b, enum glsl_base_type type, double value)
{
   unsigned bits = glsl_base_type_bit_size(type);
   return nir_imm_floatN_t(b, value, bits ? bits : 32);
}

static bool
scan_muladd_types(nir_shader *nir, bool expect_f32)
{
   bool saw = false;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_cmat_muladd_agx)
               continue;
            saw = true;
            const nir_alu_type a = nir_intrinsic_src_type(intr);
            const nir_alu_type b = nir_intrinsic_src_type2(intr);
            if (a != b) {
               fprintf(stderr,
                       "cmat_muladd_agx still has mixed types %u / %u\n",
                       (unsigned)a, (unsigned)b);
               return false;
            }
            if (expect_f32 && a != nir_type_float32) {
               fprintf(stderr,
                       "mixed 8x8 product was not widened to f32 (type %u)\n",
                       (unsigned)a);
               return false;
            }
         }
      }
   }
   if (!saw) {
      fprintf(stderr, "no cmat_muladd_agx after lowering\n");
      return false;
   }
   return true;
}

/* Whether every product's accumulator comes through an f16 rounding */
static bool
products_rounded(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic !=
                   nir_intrinsic_cmat_muladd_agx)
               continue;
            nir_scalar c = nir_scalar_resolved(
               nir_instr_as_intrinsic(instr)->src[2].ssa, 0);
            if (nir_scalar_is_alu(c) &&
                nir_scalar_alu_op(c) == nir_op_f2f32 &&
                nir_src_bit_size(nir_def_as_alu(c.def)->src[0].src) == 16)
               continue;
            if (nir_scalar_is_const(c))
               continue;
            return false;
         }
      }
   }
   return true;
}

static bool
compile_mma8(enum glsl_base_type type_a, enum glsl_base_type type_b,
             enum glsl_base_type type_c, enum agx_isa_encoding encoding,
             const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "8x8 mixed cmat %s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 4;
   b.shader->info.cs.has_cooperative_matrix = true;

   nir_variable *va =
      nir_local_variable_create(b.impl, cmat_type(type_a, GLSL_CMAT_USE_A), "a");
   nir_variable *vb =
      nir_local_variable_create(b.impl, cmat_type(type_b, GLSL_CMAT_USE_B), "b");
   nir_variable *vc = nir_local_variable_create(
      b.impl, cmat_type(type_c, GLSL_CMAT_USE_ACCUMULATOR), "c");
   nir_variable *vd = nir_local_variable_create(
      b.impl, cmat_type(type_c, GLSL_CMAT_USE_ACCUMULATOR), "d");

   nir_deref_instr *da = nir_build_deref_var(&b, va);
   nir_deref_instr *db = nir_build_deref_var(&b, vb);
   nir_deref_instr *dc = nir_build_deref_var(&b, vc);
   nir_deref_instr *dd = nir_build_deref_var(&b, vd);

   nir_cmat_construct(&b, &da->def, elem_imm(&b, type_a, 1.0));
   nir_cmat_construct(&b, &db->def, elem_imm(&b, type_b, 1.0));
   nir_cmat_construct(&b, &dc->def, elem_imm(&b, type_c, 0.0));
   nir_cmat_muladd(&b, &dd->def, &da->def, &db->def, &dc->def);
   /* A second product on the first one's result */
   nir_cmat_muladd(&b, &dc->def, &da->def, &db->def, &dd->def);
   nir_cmat_copy(&b, &dd->def, &dc->def);
   nir_def *elt = nir_cmat_extract(&b, glsl_base_type_bit_size(type_c),
                                   &dd->def, nir_imm_int(&b, 0));
   nir_store_shared(&b, elt, nir_imm_int(&b, 0), .align_mul = 4);

   if (!agx_nir_lower_cmat(b.shader)) {
      fprintf(stderr, "%s: cooperative-matrix lowering did not run\n", name);
      ralloc_free(b.shader);
      return false;
   }
   if (!scan_muladd_types(b.shader, type_a != type_b)) {
      fprintf(stderr, "%s: lowering left an unencodable product\n", name);
      ralloc_free(b.shader);
      return false;
   }

   agx_preprocess_nir(b.shader);

   /* A 16-bit accumulator is the type the shader asked for: each product
    * must still be rounded to it, not carried in f32 to the next one. */
   if (type_c == GLSL_TYPE_FLOAT16 && !products_rounded(b.shader)) {
      fprintf(stderr, "%s: f16 accumulator carried in f32 between products\n",
              name);
      ralloc_free(b.shader);
      return false;
   }

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};
   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "%s: AGX3 compile refused the legalized shader\n", name);
      ralloc_free(b.shader);
      return false;
   }
   const bool ok = part.binary != NULL && part.info.binary_size > 0;
   if (!ok)
      fprintf(stderr, "%s: compiled with an empty binary\n", name);

   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool pass = true;
   const struct {
      enum glsl_base_type a, b, c;
      enum agx_isa_encoding enc;
      const char *name;
   } cases[] = {
      {GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT, GLSL_TYPE_FLOAT,
       AGX_ISA_ENCODING_AGX3B, "f16 x f32 on AGX3B"},
      {GLSL_TYPE_FLOAT, GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT,
       AGX_ISA_ENCODING_AGX3B, "f32 x f16 on AGX3B"},
      {GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT,
       AGX_ISA_ENCODING_AGX3B, "f16 x f16 on AGX3B"},
      {GLSL_TYPE_FLOAT, GLSL_TYPE_FLOAT, GLSL_TYPE_FLOAT,
       AGX_ISA_ENCODING_AGX3B, "f32 x f32 on AGX3B"},
      {GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT, GLSL_TYPE_FLOAT,
       AGX_ISA_ENCODING_AGX3C, "f16 x f32 on AGX3C"},
      {GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT16,
       AGX_ISA_ENCODING_AGX3B, "f16 x f16 + f16 on AGX3B"},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      if (!compile_mma8(cases[i].a, cases[i].b, cases[i].c, cases[i].enc, cases[i].name)) {
         fprintf(stderr, "FAIL %s\n", cases[i].name);
         pass = false;
      }
   }

   glsl_type_singleton_decref();
   return pass ? 0 : 1;
}

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

#include "agx3_compose/agx3_decode.h"
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

/* Whether every product takes and returns f16 itself (f16matmad8x8) */
static bool
products_native_f16(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_cmat_muladd_agx &&
                nir_instr_as_intrinsic(instr)->def.bit_size != 16)
               return false;
         }
      }
   }
   return true;
}

static bool
compile_mma8(enum glsl_base_type type_a, enum glsl_base_type type_b,
             enum glsl_base_type type_c, enum agx_isa_encoding encoding,
             bool native, const char *name)
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

   if (!agx_nir_lower_cmat(b.shader, native, false)) {
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
   if (native && !products_native_f16(b.shader)) {
      fprintf(stderr, "%s: f16 accumulator not taken natively\n", name);
      ralloc_free(b.shader);
      return false;
   }
   if (type_c == GLSL_TYPE_FLOAT16 && !native && !products_rounded(b.shader)) {
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

static bool
compile_mma8_loop_ra(enum agx_isa_encoding encoding, const char *name)
{
   /* Verify that a multi-accumulator 8x8 matmad8 loop with unrolled K-slices
    * coalesces accumulator phi webs into aligned pairs in place without
    * spilling or emitting per-accumulator copy churn across the loop body. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "8x8 mma8 loop ra %s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 4096;
   b.shader->info.cs.has_cooperative_matrix = true;

   nir_def *lane = nir_load_subgroup_invocation(&b);
   nir_def *zero_i = nir_imm_int(&b, 0);
   nir_def *zero_f = nir_imm_float(&b, 0.0f);
   nir_def *zero_v2 = nir_vec2(&b, zero_f, zero_f);

   /* Prefetched vec4 tiles carried across the loop header ahead of acc_phi
    * (matching a software-pipelined GEMM K-loop): storing them into shared
    * memory at the top of the loop kills 16 halves below acc_phi, exposing a
    * low-register hole before matmad8 runs. */
   nir_def *va0_init =
      nir_load_shared(&b, 4, 32, nir_ishl_imm(&b, lane, 4), .align_mul = 16);
   nir_def *va1_init = nir_load_shared(
      &b, 4, 32, nir_ishl_imm(&b, nir_iadd_imm(&b, lane, 32), 4),
      .align_mul = 16);

   nir_loop *lp = nir_push_loop(&b);
   lp->control = nir_loop_control_dont_unroll;
   nir_block *pre = nir_cf_node_as_block(nir_cf_node_prev(&lp->cf_node));
   nir_block *hdr = nir_loop_first_block(lp);

   nir_phi_instr *k_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&k_phi->instr, &k_phi->def, 1, 32);
   nir_phi_instr_add_src(k_phi, pre, zero_i);

   nir_phi_instr *b_idx_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&b_idx_phi->instr, &b_idx_phi->def, 1, 32);
   nir_phi_instr_add_src(b_idx_phi, pre, lane);

   nir_phi_instr *va0_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&va0_phi->instr, &va0_phi->def, 4, 32);
   nir_phi_instr_add_src(va0_phi, pre, va0_init);

   nir_phi_instr *va1_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&va1_phi->instr, &va1_phi->def, 4, 32);
   nir_phi_instr_add_src(va1_phi, pre, va1_init);

   nir_phi_instr *acc_phi[4][2];
   for (unsigned m = 0; m < 4; ++m) {
      for (unsigned n = 0; n < 2; ++n) {
         acc_phi[m][n] = nir_phi_instr_create(b.shader);
         nir_def_init(&acc_phi[m][n]->instr, &acc_phi[m][n]->def, 2, 32);
         nir_phi_instr_add_src(acc_phi[m][n], pre, zero_v2);
      }
   }

   nir_def *k = &k_phi->def;
   nir_def *b_idx = &b_idx_phi->def;
   nir_if *done = nir_push_if(&b, nir_ige_imm(&b, k, 16));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, done);

   /* Consume prefetched vec4 tiles before matmad8 so their registers die. */
   nir_store_shared(&b, &va0_phi->def, nir_ishl_imm(&b, lane, 4),
                    .align_mul = 16, .write_mask = 0xf);
   nir_store_shared(&b, &va1_phi->def,
                    nir_ishl_imm(&b, nir_iadd_imm(&b, lane, 32), 4),
                    .align_mul = 16, .write_mask = 0xf);

   nir_def *cur_acc[4][2];
   for (unsigned m = 0; m < 4; ++m)
      for (unsigned n = 0; n < 2; ++n)
         cur_acc[m][n] = &acc_phi[m][n]->def;

   for (unsigned ks = 0; ks < 2; ++ks) {
      nir_def *b_tile[2];
      for (unsigned n = 0; n < 2; ++n) {
         nir_def *off = nir_ishl_imm(
            &b, nir_iadd_imm(&b, b_idx, (ks * 6 + n) * 32), 3);
         b_tile[n] = nir_load_shared(&b, 2, 32, off, .align_mul = 8);
      }

      for (unsigned m = 0; m < 4; ++m) {
         nir_def *off = nir_ishl_imm(
            &b, nir_iadd_imm(&b, nir_iadd(&b, lane, k), (ks * 6 + 2 + m) * 32),
            3);
         nir_def *a_tile = nir_load_shared(&b, 2, 32, off, .align_mul = 8);
         for (unsigned n = 0; n < 2; ++n) {
            cur_acc[m][n] = nir_cmat_muladd_agx(
               &b, a_tile, b_tile[n], cur_acc[m][n],
               .src_type = nir_type_float32, .src_type2 = nir_type_float32);
         }
      }
   }

   nir_def *k_next = nir_iadd_imm(&b, k, 1);
   nir_def *va0_next = nir_load_shared(
      &b, 4, 32, nir_ishl_imm(&b, nir_iadd(&b, lane, k_next), 4),
      .align_mul = 16);
   nir_def *va1_next = nir_load_shared(
      &b, 4, 32,
      nir_ishl_imm(&b, nir_iadd_imm(&b, nir_iadd(&b, lane, k_next), 32), 4),
      .align_mul = 16);

   nir_block *back = nir_loop_last_block(lp);
   for (unsigned m = 0; m < 4; ++m) {
      for (unsigned n = 0; n < 2; ++n) {
         nir_phi_instr_add_src(acc_phi[m][n], back, cur_acc[m][n]);
         nir_instr_insert(nir_before_block(hdr), &acc_phi[m][n]->instr);
      }
   }
   nir_phi_instr_add_src(va1_phi, back, va1_next);
   nir_instr_insert(nir_before_block(hdr), &va1_phi->instr);
   nir_phi_instr_add_src(va0_phi, back, va0_next);
   nir_instr_insert(nir_before_block(hdr), &va0_phi->instr);
   nir_phi_instr_add_src(b_idx_phi, back, nir_iadd_imm(&b, b_idx, 16));
   nir_instr_insert(nir_before_block(hdr), &b_idx_phi->instr);
   nir_phi_instr_add_src(k_phi, back, k_next);
   nir_instr_insert(nir_before_block(hdr), &k_phi->instr);
   nir_pop_loop(&b, lp);

   /* Post-loop conditional tail merging acc_phi (like a GEMM K-tail if),
    * ensuring acc_phi also feeds a later phi rather than only the loop's
    * matmad8 collect. */
   nir_def *tail_acc[4][2];
   nir_if *tail_if = nir_push_if(&b, nir_ieq_imm(&b, nir_iand_imm(&b, lane, 1), 0));
   {
      nir_def *a_t = nir_load_shared(&b, 2, 32, nir_ishl_imm(&b, lane, 3), .align_mul = 8);
      nir_def *b_t = nir_load_shared(&b, 2, 32, nir_ishl_imm(&b, nir_iadd_imm(&b, lane, 32), 3), .align_mul = 8);
      for (unsigned m = 0; m < 4; ++m) {
         for (unsigned n = 0; n < 2; ++n) {
            tail_acc[m][n] = nir_cmat_muladd_agx(
               &b, a_t, b_t, &acc_phi[m][n]->def,
               .src_type = nir_type_float32, .src_type2 = nir_type_float32);
         }
      }
   }
   nir_push_else(&b, tail_if);
   nir_pop_if(&b, tail_if);

   nir_def *out_acc[4][2];
   for (unsigned m = 0; m < 4; ++m)
      for (unsigned n = 0; n < 2; ++n)
         out_acc[m][n] = nir_if_phi(&b, tail_acc[m][n], &acc_phi[m][n]->def);

   for (unsigned m = 0; m < 4; ++m) {
      for (unsigned n = 0; n < 2; ++n) {
         nir_def *off = nir_ishl_imm(&b, nir_iadd_imm(&b, lane, (m * 2 + n) * 32), 3);
         nir_store_shared(&b, out_acc[m][n], off, .align_mul = 8, .write_mask = 3);
      }
   }

   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};
   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "%s: AGX3 compile failed\n", name);
      ralloc_free(b.shader);
      return false;
   }
   /* Without cross-block accumulator collect sinking, loop-phi copyprop, and
    * in-place matmad8_agx3 coalescing, RA emits 48+ accumulator mov1 copies
    * (>=193 instrs, >=92 GPRs); with all fixes, all 16 loop + 8 tail matmad8
    * instructions coalesce in place (~151-153 instrs, 88 GPRs). */
   const bool ok = part.binary != NULL && part.info.binary_size > 0 &&
                   part.info.scratch_size == 0 && part.info.nr_gprs <= 90 &&
                   part.info.stats.instrs <= 165;
   fprintf(ok ? stdout : stderr,
           "%s: instrs=%u binary_size=%u scratch=%u gprs=%u\n", name,
           part.info.stats.instrs, part.info.binary_size,
           part.info.scratch_size, part.info.nr_gprs);

   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static bool
compile_zero_accum_case(enum agx_isa_encoding encoding, bool is_gemm16,
                        bool neg_zero, bool expect_zero_accum,
                        unsigned *instrs_out, const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 4096;

   nir_def *lane = nir_load_subgroup_invocation(&b);
   const unsigned ncomp = is_gemm16 ? 8 : 2;
   const uint32_t zero_bits = neg_zero ? 0x80000000u : 0x00000000u;

   nir_def *a_mat, *b_mat;
   if (is_gemm16) {
      nir_def *a_lo = nir_load_shared(&b, 4, 16, nir_ishl_imm(&b, lane, 4),
                                      .align_mul = 16);
      nir_def *a_hi = nir_load_shared(
         &b, 4, 16, nir_iadd_imm(&b, nir_ishl_imm(&b, lane, 4), 8),
         .align_mul = 8);
      nir_def *b_lo = nir_load_shared(
         &b, 4, 16, nir_iadd_imm(&b, nir_ishl_imm(&b, lane, 4), 512),
         .align_mul = 16);
      nir_def *b_hi = nir_load_shared(
         &b, 4, 16, nir_iadd_imm(&b, nir_ishl_imm(&b, lane, 4), 520),
         .align_mul = 8);
      nir_def *ac[8] = {
         nir_channel(&b, a_lo, 0), nir_channel(&b, a_lo, 1),
         nir_channel(&b, a_lo, 2), nir_channel(&b, a_lo, 3),
         nir_channel(&b, a_hi, 0), nir_channel(&b, a_hi, 1),
         nir_channel(&b, a_hi, 2), nir_channel(&b, a_hi, 3),
      };
      nir_def *bc[8] = {
         nir_channel(&b, b_lo, 0), nir_channel(&b, b_lo, 1),
         nir_channel(&b, b_lo, 2), nir_channel(&b, b_lo, 3),
         nir_channel(&b, b_hi, 0), nir_channel(&b, b_hi, 1),
         nir_channel(&b, b_hi, 2), nir_channel(&b, b_hi, 3),
      };
      a_mat = nir_vec(&b, ac, 8);
      b_mat = nir_vec(&b, bc, 8);
   } else {
      a_mat = nir_load_shared(&b, 2, 32, nir_ishl_imm(&b, lane, 3),
                              .align_mul = 8);
      b_mat = nir_load_shared(
         &b, 2, 32, nir_iadd_imm(&b, nir_ishl_imm(&b, lane, 3), 256),
         .align_mul = 8);
   }

   nir_const_value cv[8] = {0};
   for (unsigned i = 0; i < ncomp; ++i)
      cv[i].u32 = zero_bits;
   nir_def *c_mat = nir_build_imm(&b, ncomp, 32, cv);

   nir_def *res = nir_cmat_muladd_agx(
      &b, a_mat, b_mat, c_mat,
      .src_type = is_gemm16 ? nir_type_float16 : nir_type_float32,
      .src_type2 = is_gemm16 ? nir_type_float16 : nir_type_float32);

   if (is_gemm16) {
      nir_def *r_lo = nir_vec4(&b, nir_channel(&b, res, 0),
                               nir_channel(&b, res, 1),
                               nir_channel(&b, res, 2),
                               nir_channel(&b, res, 3));
      nir_def *r_hi = nir_vec4(&b, nir_channel(&b, res, 4),
                               nir_channel(&b, res, 5),
                               nir_channel(&b, res, 6),
                               nir_channel(&b, res, 7));
      nir_store_shared(&b, r_lo,
                       nir_iadd_imm(&b, nir_ishl_imm(&b, lane, 5), 1024),
                       .align_mul = 16, .write_mask = 0xf);
      nir_store_shared(&b, r_hi,
                       nir_iadd_imm(&b, nir_ishl_imm(&b, lane, 5), 1040),
                       .align_mul = 16, .write_mask = 0xf);
   } else {
      nir_store_shared(&b, res,
                       nir_iadd_imm(&b, nir_ishl_imm(&b, lane, 3), 1024),
                       .align_mul = 8, .write_mask = 0x3);
   }

   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};
   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "%s: AGX3 compile failed\n", name);
      ralloc_free(b.shader);
      return false;
   }

   const agx3_table *tbl =
      encoding == AGX_ISA_ENCODING_AGX3C ? &agx3c_table : &agx3b_table;
   const enum agx3_mnemonic target_mn =
      is_gemm16 ? AGX3_MN_fGeMM16x16x16 : AGX3_MN_f32matmad8x8;
   unsigned matches = 0;
   bool field_ok = true;
   uint64_t f[AGX3_MAX_FIELDS];
   const uint8_t *bin = (const uint8_t *)part.binary;
   for (unsigned off = 0; off + 2 <= part.info.binary_size;) {
      unsigned step = 0;
      for (unsigned sz = 2; sz <= 14 && off + sz <= part.info.binary_size;
           sz += 2) {
         int32_t form = agx3_decode(tbl, bin + off, sz,
                                    AGX3_MN_NONE, f);
         if (form >= 0) {
            step = sz;
            if (tbl->forms[form].mnemonic == target_mn) {
               matches++;
               if (is_gemm16) {
                  if (f[7] != (uint64_t)expect_zero_accum) {
                     fprintf(stderr,
                             "%s: gemm16 zero_accumulator=%" PRIu64
                             " expected %u\n",
                             name, f[7], (unsigned)expect_zero_accum);
                     field_ok = false;
                  }
               } else {
                  const uint64_t exp_c_kind = expect_zero_accum ? 1u : 2u;
                  if (f[59] != exp_c_kind ||
                      (expect_zero_accum && f[52] != 0u)) {
                     fprintf(stderr,
                             "%s: mma8 c_kind=%" PRIu64 " c=%" PRIu64
                             " expected kind=%" PRIu64 "\n",
                             name, f[59], f[52], exp_c_kind);
                     field_ok = false;
                  }
               }
            }
            break;
         }
      }
      if (!step)
         break;
      off += step;
   }

   if (instrs_out)
      *instrs_out = part.info.stats.instrs;
   const bool ok = part.binary != NULL && part.info.binary_size > 0 &&
                   matches == 1 && field_ok;
   fprintf(ok ? stdout : stderr,
           "%s: instrs=%u binary_size=%u matches=%u zero_accum_ok=%u\n",
           name, part.info.stats.instrs, part.info.binary_size,
           matches, (unsigned)field_ok);

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
      bool native;
      const char *name;
   } cases[] = {
      {GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT, GLSL_TYPE_FLOAT,
       AGX_ISA_ENCODING_AGX3B, false, "f16 x f32 on AGX3B"},
      {GLSL_TYPE_FLOAT, GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT,
       AGX_ISA_ENCODING_AGX3B, false, "f32 x f16 on AGX3B"},
      {GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT,
       AGX_ISA_ENCODING_AGX3B, false, "f16 x f16 on AGX3B"},
      {GLSL_TYPE_FLOAT, GLSL_TYPE_FLOAT, GLSL_TYPE_FLOAT,
       AGX_ISA_ENCODING_AGX3B, false, "f32 x f32 on AGX3B"},
      {GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT, GLSL_TYPE_FLOAT,
       AGX_ISA_ENCODING_AGX3C, false, "f16 x f32 on AGX3C"},
      {GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT16,
       AGX_ISA_ENCODING_AGX3B, false, "f16 x f16 + f16 on AGX3B, f32 matmad"},
      {GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT16,
       AGX_ISA_ENCODING_AGX3B, true, "f16 x f16 + f16 on AGX3B, f16 matmad"},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      if (!compile_mma8(cases[i].a, cases[i].b, cases[i].c, cases[i].enc,
                        cases[i].native, cases[i].name)) {
         fprintf(stderr, "FAIL %s\n", cases[i].name);
         pass = false;
      }
   }

   if (!compile_mma8_loop_ra(AGX_ISA_ENCODING_AGX3B, "8-acc mma8 loop on AGX3B") ||
       !compile_mma8_loop_ra(AGX_ISA_ENCODING_AGX3C, "8-acc mma8 loop on AGX3C"))
      pass = false;

   unsigned g16_pos = 0, g16_neg = 0, m8_pos = 0, m8_neg = 0;
   if (!compile_zero_accum_case(AGX_ISA_ENCODING_AGX3C, true, false, true,
                                &g16_pos, "gemm16 +0.0 on AGX3C") ||
       !compile_zero_accum_case(AGX_ISA_ENCODING_AGX3C, true, true, false,
                                &g16_neg, "gemm16 -0.0 on AGX3C") ||
       !compile_zero_accum_case(AGX_ISA_ENCODING_AGX3C, false, false, true,
                                &m8_pos, "mma8 +0.0 on AGX3C") ||
       !compile_zero_accum_case(AGX_ISA_ENCODING_AGX3C, false, true, false,
                                &m8_neg, "mma8 -0.0 on AGX3C") ||
       !compile_zero_accum_case(AGX_ISA_ENCODING_AGX3B, false, false, false,
                                NULL, "mma8 +0.0 untouched on AGX3B"))
      pass = false;
   if (g16_pos >= g16_neg || m8_pos >= m8_neg) {
      fprintf(stderr,
              "FAIL zero_accum did not reduce instructions: "
              "gemm16 %u vs %u, mma8 %u vs %u\n",
              g16_pos, g16_neg, m8_pos, m8_neg);
      pass = false;
   }

   glsl_type_singleton_decref();
   return pass ? 0 : 1;
}

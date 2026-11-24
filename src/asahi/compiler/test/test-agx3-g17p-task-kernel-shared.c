/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "agx_compiler.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool
rewrite_task_payload_to_shared(nir_builder *b, nir_intrinsic_instr *intr,
                               void *unused)
{
   (void)unused;
   if (intr->intrinsic != nir_intrinsic_store_task_payload)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_store_shared(b, intr->src[0].ssa, intr->src[1].ssa,
                    .base = nir_intrinsic_base(intr),
                    .write_mask = nir_intrinsic_write_mask(intr),
                    .align_mul = nir_intrinsic_align_mul(intr),
                    .align_offset = nir_intrinsic_align_offset(intr));
   nir_instr_remove(&intr->instr);
   return true;
}

/* Match u2u16(imadshl_agx(tid, expected_factor, expected_addend, 0)). */
static bool
offset_is_u2u16_of_imad(nir_def *offset, uint64_t expected_factor,
                        uint64_t expected_addend)
{
   nir_scalar narrow = nir_scalar_resolved(offset, 0);
   if (narrow.def->bit_size != 16 || !nir_scalar_is_alu(narrow) ||
       nir_scalar_alu_op(narrow) != nir_op_u2u16)
      return false;

   nir_scalar mad = nir_scalar_chase_alu_src(narrow, 0);
   if (mad.def->bit_size != 32 || !nir_scalar_is_alu(mad) ||
       nir_scalar_alu_op(mad) != nir_op_imadshl_agx)
      return false;

   nir_scalar s0 = nir_scalar_chase_alu_src(mad, 0);
   nir_scalar s1 = nir_scalar_chase_alu_src(mad, 1);
   nir_scalar s2 = nir_scalar_chase_alu_src(mad, 2);
   nir_scalar s3 = nir_scalar_chase_alu_src(mad, 3);
   const bool factor_ok =
      (nir_scalar_is_const(s0) && nir_scalar_as_uint(s0) == expected_factor) ||
      (nir_scalar_is_const(s1) && nir_scalar_as_uint(s1) == expected_factor);
   const bool addend_ok =
      nir_scalar_is_const(s2) && nir_scalar_as_uint(s2) == expected_addend &&
      nir_scalar_is_const(s3) && nir_scalar_as_uint(s3) == 0;
   return factor_ok && addend_ok;
}

static bool
offset_is_u2u16_of_nonconst_iadd32(nir_def *offset)
{
   nir_scalar narrow = nir_scalar_resolved(offset, 0);
   if (narrow.def->bit_size != 16 || !nir_scalar_is_alu(narrow) ||
       nir_scalar_alu_op(narrow) != nir_op_u2u16)
      return false;

   nir_scalar sum = nir_scalar_chase_alu_src(narrow, 0);
   if (sum.def->bit_size != 32 || !nir_scalar_is_alu(sum) ||
       nir_scalar_alu_op(sum) != nir_op_iadd)
      return false;

   return !nir_scalar_is_const(nir_scalar_chase_alu_src(sum, 0)) &&
          !nir_scalar_is_const(nir_scalar_chase_alu_src(sum, 1));
}

static bool
offset_is_u2u16_of_ushr32(nir_def *offset, uint64_t expected_shift)
{
   nir_scalar narrow = nir_scalar_resolved(offset, 0);
   if (narrow.def->bit_size != 16 || !nir_scalar_is_alu(narrow) ||
       nir_scalar_alu_op(narrow) != nir_op_u2u16)
      return false;

   nir_scalar sh = nir_scalar_chase_alu_src(narrow, 0);
   if (sh.def->bit_size != 32 || !nir_scalar_is_alu(sh) ||
       nir_scalar_alu_op(sh) != nir_op_ushr)
      return false;

   nir_scalar amount = nir_scalar_chase_alu_src(sh, 1);
   return nir_scalar_is_const(amount) &&
          nir_scalar_as_uint(amount) == expected_shift;
}

static unsigned
count_alu_bitsize(nir_shader *shader, nir_op op, unsigned bit_size)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            const nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == op && (bit_size == 0 || alu->def.bit_size == bit_size))
               count++;
         }
      }
   }
   return count;
}

/* Regression test for G17P task-to-compute lowering:
 * Borealis preprocesses a MESA_SHADER_TASK shader once before lowering task
 * payload accesses (which sets shared_element_index_agx = true), then turns
 * the shader into a MESA_SHADER_COMPUTE kernel with byte-addressed shared
 * payload stores/loads and preprocesses it again.
 *
 * Using 8-byte-aligned byte offsets (tid * 56 + 32) and (tid * 56 + 36)
 * distinguishes all four outcomes:
 * - unvectorized: 2 stores + 2 loads instead of 1 vec2 store + 1 vec2 load
 * - 0 conversions: 32-bit offset fails AGX3 packing (or scale 56, addend 32)
 * - 1 conversion (expected): (tid * 56 + 32) / 4 = tid * 14 + 8 ->
 *   u2u16(imadshl_agx(tid, 14, 8, 0)) with 0 ushr instructions
 * - 2 conversions: 14 is not divisible by 4, emitting a ushr
 */
static bool
check_g17p_task_kernel_shared(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_TASK, &agx_nir_options, "g17p task kernel shared");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *byte_base = nir_imul_imm(&b, tid, 56);
   nir_store_task_payload(&b, nir_iadd_imm(&b, tid, 1),
                          nir_iadd_imm(&b, byte_base, 32), .align_mul = 8,
                          .align_offset = 0);
   nir_store_task_payload(&b, nir_iadd_imm(&b, tid, 2),
                          nir_iadd_imm(&b, byte_base, 36), .align_mul = 8,
                          .align_offset = 4);

   /* First preprocess as MESA_SHADER_TASK (no shared memory accesses yet). */
   agx_preprocess_nir_g17p(b.shader);
   if (!b.shader->info.shared_element_index_agx) {
      fprintf(stderr, "TASK preprocess did not set shared_element_index_agx\n");
      ralloc_free(b.shader);
      return false;
   }

   /* Lower task payload into shared memory and turn the shader into a compute
    * kernel, matching hk_g17p_lower_task_kernel(). */
   nir_shader_intrinsics_pass(b.shader, rewrite_task_payload_to_shared,
                              nir_metadata_control_flow, NULL);

   b.cursor = nir_after_impl(nir_shader_get_entrypoint(b.shader));
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);

   tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *off0 = nir_imul_imm(&b, tid, 56);
   nir_def *v0 = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, off0, 32),
                                 .align_mul = 8, .align_offset = 0);
   nir_def *v1 = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, off0, 36),
                                 .align_mul = 8, .align_offset = 4);
   nir_store_global(&b, nir_iadd(&b, v0, v1),
                    nir_iadd(&b, nir_imm_int64(&b, UINT64_C(0x2000000000)),
                             nir_u2u64(&b, nir_imul_imm(&b, tid, 4))),
                    .align_mul = 4);

   b.shader->info.shared_size = 1856;
   b.shader->info.stage = MESA_SHADER_COMPUTE;
   memset(&b.shader->info.cs, 0, sizeof(b.shader->info.cs));

   /* Second preprocess as MESA_SHADER_COMPUTE must clear ownership for
    * deferred vectorization + element-index lowering. */
   agx_preprocess_nir_g17p(b.shader);
   if (b.shader->info.shared_element_index_agx) {
      fprintf(stderr, "COMPUTE preprocess failed to defer shared lowering\n");
      ralloc_free(b.shader);
      return false;
   }

   struct agx_shader_key key = {
      .dev.chip = AGX_CHIP_G17P,
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part);

   unsigned store_count = 0, load_count = 0;
   bool store_ok = false, load_ok = false;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_shared) {
               store_count++;
               store_ok = intr->num_components == 2 &&
                          nir_src_bit_size(intr->src[0]) == 32 &&
                          nir_intrinsic_write_mask(intr) == 0x3 &&
                          nir_intrinsic_base(intr) == 0 &&
                          offset_is_u2u16_of_imad(intr->src[1].ssa, 14, 8);
            } else if (intr->intrinsic == nir_intrinsic_load_shared) {
               load_count++;
               load_ok = intr->def.num_components == 2 &&
                         intr->def.bit_size == 32 &&
                         nir_intrinsic_base(intr) == 0 &&
                         offset_is_u2u16_of_imad(intr->src[0].ssa, 14, 8);
            }
         }
      }
   }

   const unsigned ushr_count = count_alu_bitsize(b.shader, nir_op_ushr, 0);
   const bool valid = ok && part.binary != NULL && part.info.binary_size > 0 &&
                      part.info.local_size == 1856 &&
                      b.shader->info.shared_element_index_agx &&
                      store_count == 1 && load_count == 1 &&
                      store_ok && load_ok && ushr_count == 0;

   if (!valid) {
      fprintf(stderr,
              "G17P task kernel shared check failed: ok=%d stores=%u (ok=%d) "
              "loads=%u (ok=%d) ushr=%u\n",
              ok, store_count, store_ok, load_count, load_ok, ushr_count);
      nir_print_shader(b.shader, stderr);
   }

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

/* AGX2 (M1/M2) never sets shared_element_index_agx and must keep its scalar
 * byte-addressed shared lowering without enabling nir_var_mem_shared
 * vectorization by default. */
static bool
check_agx2_compute_shared_not_vectorized(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "agx2 compute shared scalar");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 544;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *off = nir_imul_imm(&b, tid, 16);
   nir_store_shared(&b, nir_iadd_imm(&b, tid, 1), nir_iadd_imm(&b, off, 32),
                    .align_mul = 16, .align_offset = 0);
   nir_store_shared(&b, nir_iadd_imm(&b, tid, 2), nir_iadd_imm(&b, off, 36),
                    .align_mul = 16, .align_offset = 4);

   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);

   nir_def *v0 = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, off, 32),
                                 .align_mul = 16, .align_offset = 0);
   nir_def *v1 = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, off, 36),
                                 .align_mul = 16, .align_offset = 4);
   nir_store_global(&b, nir_iadd(&b, v0, v1),
                    nir_iadd(&b, nir_imm_int64(&b, UINT64_C(0x2000000000)),
                             nir_u2u64(&b, nir_imul_imm(&b, tid, 4))),
                    .align_mul = 4);

   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {
      .dev.chip = AGX_CHIP_G13G,
      .dev.encoding = AGX_ISA_ENCODING_AGX2,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part);

   unsigned store_count = 0, load_count = 0;
   bool all_vec2 = true;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_shared) {
               store_count++;
               all_vec2 &= intr->num_components == 2 &&
                           intr->src[1].ssa->bit_size == 16;
            } else if (intr->intrinsic == nir_intrinsic_load_shared) {
               load_count++;
               all_vec2 &= intr->def.num_components == 2 &&
                           intr->src[0].ssa->bit_size == 16;
            }
         }
      }
   }

   const bool valid = ok && part.binary != NULL && part.info.binary_size > 0 &&
                      !b.shader->info.shared_element_index_agx &&
                      store_count == 1 && load_count == 1 && all_vec2;
   if (!valid) {
      fprintf(stderr,
              "AGX2 compute shared check failed: ok=%d elem=%d stores=%u "
              "loads=%u vec2=%d\n",
              ok, b.shader->info.shared_element_index_agx, store_count,
              load_count, all_vec2);
      nir_print_shader(b.shader, stderr);
   }

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

/* Frontends that preprocess with g17p == false (agx_preprocess_nir() for
 * Gallium/libagx/internal helpers, and agx_preprocess_nir_g18g() for G18G)
 * narrow shared offsets to u2u16(off32) during preprocess. Verify AGX3 compute
 * lowering chases through u2u16 in 32 bits without emitting a 16-bit ushr and
 * preserves agx_nir_opt_shared_byte_offsets() constant folding into .base on
 * both exact-divided sums and shifted byte offsets. */
static bool
check_agx3_generic_preprocess_shared_byte_offsets(enum agx_chip chip)
{
   /* G18G compiles as G17G and re-encodes as agx3d (agx_gather_device_key) */
   const bool g18g = chip == AGX_CHIP_G18G;
   const enum agx_isa_encoding encoding =
      agx_chip_isa_encoding(g18g ? AGX_CHIP_G17G : chip);

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      g18g ? "g18g compute shared offsets" : "agx3 generic preprocess shared");
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 4;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 4352;

   nir_def *lid = nir_load_local_invocation_id(&b);
   nir_def *lx = nir_channel(&b, lid, 0);
   nir_def *ly = nir_channel(&b, lid, 1);
   nir_def *exact_bytes =
      nir_iadd_imm(&b, nir_iadd(&b, nir_ishl_imm(&b, lx, 2),
                                nir_ishl_imm(&b, ly, 2)),
                   32);
   nir_store_shared(&b, lx, exact_bytes, .align_mul = 4);

   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_shared);

   nir_def *out_base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *dyn_byte =
      nir_iand_imm(&b, nir_load_preamble(&b, 1, 32, .base = 4), 4095);
   nir_def *val = nir_load_shared(&b, 1, 32, nir_iadd_imm(&b, dyn_byte, 12),
                                  .align_mul = 4);
   nir_store_global(&b, val,
                    nir_iadd(&b, out_base,
                             nir_u2u64(&b, nir_imul_imm(&b, lx, 4))),
                    .align_mul = 4);

   if (g18g)
      agx_preprocess_nir_g18g(b.shader);
   else
      agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {
      .dev.chip = chip,
      .dev.encoding = encoding,
      .dev.emit_agx3d = g18g,
      .reserved_preamble = 8,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part);

   bool store_ok = false, load_ok = false;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_shared) {
               store_ok = nir_intrinsic_base(intr) == 32 &&
                          offset_is_u2u16_of_nonconst_iadd32(intr->src[1].ssa);
            } else if (intr->intrinsic == nir_intrinsic_load_shared) {
               load_ok = nir_intrinsic_base(intr) == 12 &&
                         offset_is_u2u16_of_ushr32(intr->src[0].ssa, 2);
            }
         }
      }
   }

   const unsigned ushr16 = count_alu_bitsize(b.shader, nir_op_ushr, 16);
   const bool valid = ok && part.binary != NULL && part.info.binary_size > 0 &&
                      b.shader->info.shared_element_index_agx &&
                      store_ok && load_ok && ushr16 == 0;
   if (!valid) {
      fprintf(stderr,
              "AGX3 generic preprocess check (chip=%d) failed: ok=%d "
              "store_ok=%d load_ok=%d ushr16=%u\n",
              chip, ok, store_ok, load_ok, ushr16);
      nir_print_shader(b.shader, stderr);
   }

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

int
main(void)
{
   if (!check_g17p_task_kernel_shared() ||
       !check_agx2_compute_shared_not_vectorized() ||
       !check_agx3_generic_preprocess_shared_byte_offsets(AGX_CHIP_G17P) ||
       !check_agx3_generic_preprocess_shared_byte_offsets(AGX_CHIP_G17G) ||
       !check_agx3_generic_preprocess_shared_byte_offsets(AGX_CHIP_G18G))
      return 1;

   puts("g17p task kernel shared: ok");
   return 0;
}

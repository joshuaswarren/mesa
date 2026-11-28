/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Regression tests for hk_lower_aligned_ubo_nir():
 * - 32-bit / 64-bit / 16-bit dynamic array index wraparound across nonzero
 *   struct offsets, arbitrary strides, and nested arrays must select
 *   AGX_ZERO_PAGE_ADDRESS instead of wrapping into bounds.
 * - Constant-bound inline uniform blocks (both <= 64 KiB and > 64 KiB) must
 *   stay on explicit I/O so byte-exact bounds checks are preserved.
 */
#undef NDEBUG
#include "hk_shader.h"
#include "asahi/compiler/agx_compile.h"
#include "asahi/lib/agx_abi.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);          \
         abort();                                                              \
      }                                                                        \
   } while (0)

static const uint64_t TEST_BASE = 0x1000000000ull;

static nir_def *
make_dyn_u32(nir_builder *b, unsigned offset)
{
   return nir_load_global_constant(b, 1, 32,
                                   nir_imm_int64(b, 0x2000000000ull + offset),
                                   .align_mul = 4);
}

static nir_def *
make_ubo_desc(nir_builder *b, bool const_bound, uint32_t bound_val)
{
   nir_def *base_lo = make_dyn_u32(b, 0);
   nir_def *base_hi = make_dyn_u32(b, 4);
   nir_def *bound =
      const_bound ? nir_imm_int(b, bound_val) : make_dyn_u32(b, 8);
   return nir_vec4(b, base_lo, base_hi, bound, nir_imm_int(b, 0));
}

static void
substitute_load(nir_shader *nir, uint64_t addr, uint64_t val, unsigned bits)
{
   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_global_constant ||
                !nir_src_is_const(intr->src[0]) ||
                nir_src_as_uint(intr->src[0]) != addr)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_def *c = nir_imm_intN_t(&b, val, bits);
            nir_def_replace(&intr->def, c);
         }
      }
   }
}

/* Clone a lowered shader, substitute runtime inputs for base/bound/indices,
 * constant-fold, and return the resolved load_global_constant address. */
static uint64_t
eval_lowered_addr(const nir_shader *lowered, uint64_t base, uint32_t bound,
                  uint64_t idx0, unsigned idx0_bits, uint64_t idx1,
                  unsigned idx1_bits)
{
   nir_shader *s = nir_shader_clone(NULL, lowered);
   substitute_load(s, 0x2000000000ull + 0, (uint32_t)base, 32);
   substitute_load(s, 0x2000000000ull + 4, (uint32_t)(base >> 32), 32);
   substitute_load(s, 0x2000000000ull + 8, bound, 32);
   if (idx0_bits)
      substitute_load(s, 0x2000000000ull + 16, idx0, idx0_bits);
   if (idx1_bits)
      substitute_load(s, 0x2000000000ull + 24, idx1, idx1_bits);

   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, s, nir_opt_constant_folding);
      NIR_PASS(progress, s, nir_opt_algebraic);
      NIR_PASS(progress, s, nir_opt_dce);
   } while (progress);
   nir_validate_shader(s, "eval_lowered_addr");

   uint64_t found_addr = UINT64_MAX;
   unsigned count = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_global_constant) {
               CHECK(nir_src_is_const(intr->src[0]));
               found_addr = nir_src_as_uint(intr->src[0]);
               count++;
            }
         }
      }
   }
   CHECK(count == 1);
   ralloc_free(s);
   return found_addr;
}

static unsigned
count_load_derefs(nir_shader *nir)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_load_deref)
               n++;
         }
      }
   }
   return n;
}

/* Dynamically indexed inline uniform blocks (<= 64 KiB and > 64 KiB) carry
 * constant bounds and must stay on explicit I/O rather than 64-byte record
 * rounding. */
static void
test_inline_uniform_block_skipped(uint32_t binding_size)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "inline");
   const struct glsl_type *arr =
      glsl_array_type(glsl_uint_type(), 16, 4);
   nir_def *desc = make_ubo_desc(&b, true, binding_size);
   nir_deref_instr *cast =
      nir_build_deref_cast(&b, desc, nir_var_mem_ubo, arr, 4);
   cast->cast.align_mul = 4;
   cast->cast.align_offset = 0;
   nir_deref_instr *elem =
      nir_build_deref_array(&b, cast, make_dyn_u32(&b, 16));
   nir_def *val = nir_load_deref(&b, elem);
   nir_store_global(&b, val, nir_imm_int64(&b, 0x3000000000ull),
                    .align_mul = 4, .write_mask = 1);

   CHECK(!hk_lower_aligned_ubo_nir(b.shader, 0));
   CHECK(count_load_derefs(b.shader) == 1);
   nir_validate_shader(b.shader, "inline_skipped");
   ralloc_free(b.shader);
}

/* Nonzero struct offset + 16-byte stride vec4 array: 32-bit wraparound
 * (0x10000000 * 16 == 0 mod 2^32), negative index (-1), and exact 64 KiB
 * boundary must all route to AGX_ZERO_PAGE_ADDRESS. */
static void
test_struct_offset_vec4_wraparound(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "struct_vec4");
   const struct glsl_type *arr = glsl_array_type(glsl_vec4_type(), 64, 16);
   struct glsl_struct_field fields[2] = {
      {.type = glsl_vec4_type(), .name = "pad", .offset = 0},
      {.type = arr, .name = "data", .offset = 16},
   };
   const struct glsl_type *blk =
      glsl_interface_type(fields, 2, GLSL_INTERFACE_PACKING_STD140, false, "B");

   nir_def *desc = make_ubo_desc(&b, false, 0);
   nir_deref_instr *cast =
      nir_build_deref_cast(&b, desc, nir_var_mem_ubo, blk, 0);
   cast->cast.align_mul = 64;
   cast->cast.align_offset = 0;
   nir_deref_instr *field = nir_build_deref_struct(&b, cast, 1);
   nir_deref_instr *elem =
      nir_build_deref_array(&b, field, make_dyn_u32(&b, 16));
   nir_def *val = nir_load_deref(&b, elem);
   nir_store_global(&b, val, nir_imm_int64(&b, 0x3000000000ull),
                    .align_mul = 16, .write_mask = 0xf);

   CHECK(hk_lower_aligned_ubo_nir(b.shader, 0));
   CHECK(count_load_derefs(b.shader) == 0);
   nir_validate_shader(b.shader, "struct_vec4");

   /* In-bounds: idx=0 -> base + 16 */
   CHECK(eval_lowered_addr(b.shader, TEST_BASE, 256, 0, 32, 0, 0) ==
         TEST_BASE + 16);
   /* In-bounds at max valid slot (16 + 4094 * 16 = 65520 < 65536) */
   CHECK(eval_lowered_addr(b.shader, TEST_BASE, 65536, 4094, 32, 0, 0) ==
         TEST_BASE + 65520);
   /* Exact 64 KiB boundary (16 + 4095 * 16 = 65536) -> zero page */
   CHECK(eval_lowered_addr(b.shader, TEST_BASE, 65536, 4095, 32, 0, 0) ==
         AGX_ZERO_PAGE_ADDRESS);
   /* 32-bit wraparound: 0x10000000 * 16 == 0 mod 2^32 -> zero page */
   uint64_t wrap_addr =
      eval_lowered_addr(b.shader, TEST_BASE, 256, 0x10000000u, 32, 0, 0);
   CHECK(wrap_addr >= AGX_ZERO_PAGE_ADDRESS &&
         wrap_addr < AGX_ZERO_PAGE_ADDRESS + AGX_ZERO_PAGE_SIZE);
   /* Negative 32-bit index (-1) -> zero page */
   uint64_t neg_addr =
      eval_lowered_addr(b.shader, TEST_BASE, 256, 0xffffffffu, 32, 0, 0);
   CHECK(neg_addr >= AGX_ZERO_PAGE_ADDRESS &&
         neg_addr < AGX_ZERO_PAGE_ADDRESS + AGX_ZERO_PAGE_SIZE);

   ralloc_free(b.shader);
}

/* 4-byte scalar array (align_mul == 4, < 16-byte access): 0x40000000 * 4 wraps
 * to 0 mod 2^32 and must select AGX_ZERO_PAGE_ADDRESS. */
static void
test_u32_stride4_wraparound(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "u32_s4");
   const struct glsl_type *arr = glsl_array_type(glsl_uint_type(), 64, 4);
   nir_def *desc = make_ubo_desc(&b, false, 0);
   nir_deref_instr *cast =
      nir_build_deref_cast(&b, desc, nir_var_mem_ubo, arr, 4);
   cast->cast.align_mul = 4;
   cast->cast.align_offset = 0;
   nir_deref_instr *elem =
      nir_build_deref_array(&b, cast, make_dyn_u32(&b, 16));
   nir_def *val = nir_load_deref(&b, elem);
   nir_store_global(&b, val, nir_imm_int64(&b, 0x3000000000ull),
                    .align_mul = 4, .write_mask = 1);

   CHECK(hk_lower_aligned_ubo_nir(b.shader, 0));
   CHECK(count_load_derefs(b.shader) == 0);
   nir_validate_shader(b.shader, "u32_s4");

   CHECK(eval_lowered_addr(b.shader, TEST_BASE, 256, 3, 32, 0, 0) ==
         TEST_BASE + 12);
   uint64_t wrap_addr =
      eval_lowered_addr(b.shader, TEST_BASE, 256, 0x40000000u, 32, 0, 0);
   CHECK(wrap_addr >= AGX_ZERO_PAGE_ADDRESS &&
         wrap_addr < AGX_ZERO_PAGE_ADDRESS + AGX_ZERO_PAGE_SIZE);
   uint64_t oob_addr =
      eval_lowered_addr(b.shader, TEST_BASE, 64, 16, 32, 0, 0);
   CHECK(oob_addr >= AGX_ZERO_PAGE_ADDRESS &&
         oob_addr < AGX_ZERO_PAGE_ADDRESS + AGX_ZERO_PAGE_SIZE);

   ralloc_free(b.shader);
}

/* Nested 2D array: outer wraparound and combined sum >= 64 KiB (even with a
 * > 64 KiB descriptor bound) must select AGX_ZERO_PAGE_ADDRESS. */
static void
test_nested_2d_array(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "nested2d");
   const struct glsl_type *inner = glsl_array_type(glsl_vec4_type(), 4, 16);
   const struct glsl_type *outer = glsl_array_type(inner, 16, 64);

   nir_def *desc = make_ubo_desc(&b, false, 0);
   nir_deref_instr *cast =
      nir_build_deref_cast(&b, desc, nir_var_mem_ubo, outer, 64);
   cast->cast.align_mul = 64;
   cast->cast.align_offset = 0;
   nir_deref_instr *row =
      nir_build_deref_array(&b, cast, make_dyn_u32(&b, 16));
   nir_deref_instr *elem =
      nir_build_deref_array(&b, row, make_dyn_u32(&b, 24));
   nir_def *val = nir_load_deref(&b, elem);
   nir_store_global(&b, val, nir_imm_int64(&b, 0x3000000000ull),
                    .align_mul = 16, .write_mask = 0xf);

   CHECK(hk_lower_aligned_ubo_nir(b.shader, 0));
   CHECK(count_load_derefs(b.shader) == 0);
   nir_validate_shader(b.shader, "nested2d");

   /* In-bounds: 2 * 64 + 1 * 16 = 144 */
   CHECK(eval_lowered_addr(b.shader, TEST_BASE, 256, 2, 32, 1, 32) ==
         TEST_BASE + 144);
   /* Outer wraps mod 2^32: 0x04000000 * 64 == 0 mod 2^32 -> zero page */
   uint64_t wrap_outer =
      eval_lowered_addr(b.shader, TEST_BASE, 256, 0x04000000u, 32, 0, 32);
   CHECK(wrap_outer >= AGX_ZERO_PAGE_ADDRESS &&
         wrap_outer < AGX_ZERO_PAGE_ADDRESS + AGX_ZERO_PAGE_SIZE);
   /* Both indices <= per-step max, sum = 1023*64 + 4095*16 = 130992 >= 65536 */
   uint64_t sum_oob =
      eval_lowered_addr(b.shader, TEST_BASE, 131072, 1023, 32, 4095, 32);
   CHECK(sum_oob >= AGX_ZERO_PAGE_ADDRESS &&
         sum_oob < AGX_ZERO_PAGE_ADDRESS + AGX_ZERO_PAGE_SIZE);

   ralloc_free(b.shader);
}

/* Struct field near 64 KiB + large stride: only idx == 0 can remain in bounds;
 * idx == 1 and wraparound indices must select AGX_ZERO_PAGE_ADDRESS. */
static void
test_high_offset_and_large_stride(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "hi_off");
   const struct glsl_type *arr = glsl_array_type(glsl_vec4_type(), 4, 32);
   struct glsl_struct_field fields[2] = {
      {.type = glsl_vec4_type(), .name = "pad", .offset = 0},
      {.type = arr, .name = "tail", .offset = 65520},
   };
   const struct glsl_type *blk =
      glsl_interface_type(fields, 2, GLSL_INTERFACE_PACKING_STD140, false, "H");

   nir_def *desc = make_ubo_desc(&b, false, 0);
   nir_deref_instr *cast =
      nir_build_deref_cast(&b, desc, nir_var_mem_ubo, blk, 0);
   cast->cast.align_mul = 64;
   cast->cast.align_offset = 0;
   nir_deref_instr *field = nir_build_deref_struct(&b, cast, 1);
   nir_deref_instr *elem =
      nir_build_deref_array(&b, field, make_dyn_u32(&b, 16));
   nir_def *val = nir_load_deref(&b, elem);
   nir_store_global(&b, val, nir_imm_int64(&b, 0x3000000000ull),
                    .align_mul = 16, .write_mask = 0xf);

   CHECK(hk_lower_aligned_ubo_nir(b.shader, 0));
   CHECK(eval_lowered_addr(b.shader, TEST_BASE, 65536, 0, 32, 0, 0) ==
         TEST_BASE + 65520);
   uint64_t oob1 =
      eval_lowered_addr(b.shader, TEST_BASE, 65536, 1, 32, 0, 0);
   CHECK(oob1 >= AGX_ZERO_PAGE_ADDRESS &&
         oob1 < AGX_ZERO_PAGE_ADDRESS + AGX_ZERO_PAGE_SIZE);
   uint64_t wrap =
      eval_lowered_addr(b.shader, TEST_BASE, 65536, 0x08000000u, 32, 0, 0);
   CHECK(wrap >= AGX_ZERO_PAGE_ADDRESS &&
         wrap < AGX_ZERO_PAGE_ADDRESS + AGX_ZERO_PAGE_SIZE);
   ralloc_free(b.shader);
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   test_inline_uniform_block_skipped(20);
   test_inline_uniform_block_skipped(64);
   test_inline_uniform_block_skipped(AGX_ZERO_PAGE_SIZE);
   test_inline_uniform_block_skipped(AGX_ZERO_PAGE_SIZE * 2);
   test_struct_offset_vec4_wraparound();
   test_u32_stride4_wraparound();
   test_nested_2d_array();
   test_high_offset_and_large_stride();
   glsl_type_singleton_decref();
   printf("PASS: hk_aligned_ubo\n");
   return 0;
}

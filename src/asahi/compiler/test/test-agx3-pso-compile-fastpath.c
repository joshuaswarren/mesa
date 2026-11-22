/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agx_compile.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "util/blob.h"
#include "util/hash_table.h"

static struct agx_shader_key
g17p_key(bool no_preamble)
{
   return (struct agx_shader_key){
      .dev.chip = AGX_CHIP_G17P,
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .dev.soft_fault = true,
      .reserved_preamble = 8,
      .no_preamble = no_preamble,
      .g17p_native_uniforms = !no_preamble,
      .promote_constants = true,
   };
}

static unsigned
type_size_vec4(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/* Exercises the real post-lower path: preprocess once, lower shader I/O +
 * push constants (the exact passes allowed before agx_preprocess_nir_post_lower),
 * and verify bit-exact equivalence against a full second preprocess including
 * G17P signed-zero fmax lowering. */
static bool
test_post_lower_equivalence(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "post_lower_equiv");

   nir_variable *in_var =
      nir_variable_create(b.shader, nir_var_shader_in, glsl_vec4_type(), "in0");
   in_var->data.location = VARYING_SLOT_VAR0;
   in_var->data.interpolation = INTERP_MODE_SMOOTH;

   nir_variable *out_var =
      nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "out0");
   out_var->data.location = FRAG_RESULT_DATA0;

   nir_def *in = nir_load_var(&b, in_var);
   nir_def *pc_scale = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0),
                                              .base = 0, .range = 16);
   nir_def *scaled = nir_fmul(&b, in, pc_scale);
   nir_def *clamped = nir_fmax(&b, scaled, nir_imm_float(&b, 0.0f));
   nir_def_as_alu(clamped)->fp_math_ctrl = nir_fp_preserve_signed_zero;
   nir_store_var(&b, out_var, clamped, 0xf);

   agx_preprocess_nir_native(b.shader, AGX_CHIP_G17P);

   /* Run the exact frontend I/O lowering allowed between preprocess calls. */
   nir_lower_io(b.shader, nir_var_shader_in | nir_var_shader_out,
                type_size_vec4,
                nir_lower_io_lower_64bit_to_32 |
                   nir_lower_io_use_interpolated_input_intrinsics);
   nir_opt_copy_prop(b.shader);
   nir_opt_constant_folding(b.shader);
   nir_opt_dce(b.shader);

   nir_shader *ref = nir_shader_clone(NULL, b.shader);
   agx_preprocess_nir_native(ref, AGX_CHIP_G17P);
   agx_preprocess_nir_post_lower(b.shader, AGX_CHIP_G17P);

   /* Replace load_push_constant with preamble loads so the backend can compile
    * directly without Vulkan descriptor lowering. */
   nir_shader *shaders[2] = {ref, b.shader};
   for (unsigned s = 0; s < 2; ++s) {
      nir_function_impl *impl = nir_shader_get_entrypoint(shaders[s]);
      nir_builder rb = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_push_constant) {
               rb.cursor = nir_before_instr(instr);
               nir_def *repl = nir_load_preamble(&rb, 1, 32, .base = 0);
               nir_def_replace(&intr->def, repl);
            } else if (intr->intrinsic == nir_intrinsic_store_output) {
               rb.cursor = nir_before_instr(instr);
               nir_def *val = intr->src[0].ssa;
               for (unsigned c = 0; c < val->num_components; ++c)
                  nir_export_agx(&rb, nir_channel(&rb, val, c), .base = c * 2);
               nir_instr_remove(instr);
            }
         }
      }
      agx_nir_lower_interpolation(shaders[s], false);
   }

   struct agx_shader_key key_ref = g17p_key(false);
   struct agx_shader_key key_fast = g17p_key(false);
   struct agx_shader_part part_ref = {0}, part_fast = {0};

   bool ok = agx_compile_shader_nir(ref, &key_ref, &part_ref) &&
             agx_compile_shader_nir(b.shader, &key_fast, &part_fast) &&
             part_ref.info.binary_size > 0 &&
             part_ref.info.binary_size == part_fast.info.binary_size &&
             memcmp(part_ref.binary, part_fast.binary,
                    part_ref.info.binary_size) == 0;
   if (!ok)
      fprintf(stderr, "post_lower_equivalence failed: ref=%u fast=%u\n",
              part_ref.info.binary_size, part_fast.info.binary_size);

   free(part_ref.binary);
   free(part_fast.binary);
   ralloc_free(ref);
   ralloc_free(b.shader);
   return ok;
}

static bool
test_post_lower_temp_var_fallback(bool shader_temp)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      shader_temp ? "post_lower_shader_temp_fallback"
                  : "post_lower_local_temp_fallback");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   agx_preprocess_nir_native(b.shader, AGX_CHIP_G17P);

   /* Inject a shader_temp or function_temp variable after first preprocess to
    * exercise agx_preprocess_nir_post_lower()'s full fallback path. */
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   b.cursor = nir_after_impl(impl);
   nir_variable *tmp = shader_temp
      ? nir_variable_create(b.shader, nir_var_shader_temp, glsl_uint_type(), "st")
      : nir_local_variable_create(impl, glsl_uint_type(), "lt");
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *id = nir_load_local_invocation_index(&b);
   nir_store_var(&b, tmp, nir_iadd_imm(&b, id, 7), 1);
   nir_def *v = nir_load_var(&b, tmp);
   nir_store_global(&b, v, nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, id), 2)),
                    .align_mul = 4);

   agx_preprocess_nir_post_lower(b.shader, AGX_CHIP_G17P);

   struct agx_shader_key key = g17p_key(true);
   struct agx_shader_part part = {0};
   bool ok = exec_list_is_empty(&impl->locals) &&
             agx_compile_shader_nir(b.shader, &key, &part) &&
             part.info.binary_size > 0;
   if (!ok)
      fprintf(stderr, "post_lower_temp_var_fallback(shader_temp=%d) failed\n",
              shader_temp);

   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/* Regression test: Gallium's sysval lowering rewrites uses of load_clip_z_coeff
 * and leaves bindless load_sysval_agx dead in NIR. Compiling such a shader
 * without 8-bit bit-size work must still DCE dead intrinsics before
 * nir_divergence_analysis(). */
static bool
test_dead_sysval_before_divergence(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "dead_sysval_dce");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *id = nir_load_local_invocation_index(&b);
   nir_store_global(&b, id, nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, id), 2)),
                    .align_mul = 4);
   agx_preprocess_nir_native(b.shader, AGX_CHIP_G17P);

   /* Inject dead intrinsics after preprocess, mirroring agx_nir_lower_sysvals. */
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);
   b.cursor = nir_before_impl(impl);
   (void)nir_load_clip_z_coeff(&b);
   (void)nir_load_sysval_agx(&b, 1, 64, .desc_set = 7, .binding = 0,
                             .flags = -1);

   struct agx_shader_key key = g17p_key(true);
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(b.shader, &key, &part) &&
             part.info.binary_size > 0;
   if (!ok)
      fprintf(stderr, "dead_sysval_before_divergence failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static bool
test_bit_size_guard(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "bit_size_8bit");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *id = nir_load_local_invocation_index(&b);
   nir_def *addr = nir_iadd(&b, base, nir_u2u64(&b, id));
   nir_def *x8 = nir_load_global(&b, 1, 8, addr, .align_mul = 1);
   nir_def *y8 = nir_iadd(&b, x8, nir_imm_intN_t(&b, 3, 8));
   nir_def *cmp = nir_ult(&b, y8, nir_imm_intN_t(&b, 100, 8));
   nir_def *rev = nir_bitfield_reverse(&b, nir_u2u16(&b, y8));
   nir_def *res = nir_bcsel(&b, cmp, nir_u2u32(&b, rev), nir_imm_int(&b, 0));
   nir_store_global(&b, res, nir_iadd_imm(&b, base, 1024), .align_mul = 4);

   agx_preprocess_nir_native(b.shader, AGX_CHIP_G17P);

   struct agx_shader_key key = g17p_key(true);
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(b.shader, &key, &part) &&
             part.info.binary_size > 0;
   if (!ok)
      fprintf(stderr, "bit_size_guard (8-bit + bitfield_reverse) failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static nir_shader *
build_test_fs_main(bool uniform_only)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      uniform_only ? "fs_preopt_uniform" : "fs_preopt_main");
   nir_def *u_base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *scale =
      nir_load_global_constant(&b, 1, 32, nir_iadd_imm(&b, u_base, 16),
                               .align_mul = 4);
   nir_def *bias =
      nir_load_global_constant(&b, 1, 32, nir_iadd_imm(&b, u_base, 20),
                               .align_mul = 4);
   nir_def *in = NULL;
   if (!uniform_only) {
      nir_def *bary = nir_load_barycentric_pixel(
         &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
      in = nir_load_interpolated_input(
         &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   }
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *val = uniform_only
         ? nir_fadd(&b, nir_fmul(&b, scale, nir_imm_float(&b, 0.25f * (c + 1))),
                    bias)
         : nir_fadd(&b, nir_fmul(&b, nir_channel(&b, in, c), scale), bias);
      nir_export_agx(&b, val, .base = c * 2);
   }
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_preprocess_nir_native(b.shader, AGX_CHIP_G17P);
   NIR_PASS(_, b.shader, agx_nir_lower_interpolation, false);
   return b.shader;
}

static nir_shader *
build_test_epilog(bool blended)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      blended ? "fs_epilog_blend" : "fs_epilog_opaque");
   nir_def *ch[4];
   for (unsigned c = 0; c < 4; ++c)
      ch[c] = nir_load_exported_agx(&b, 1, 32, .base = c * 2);
   nir_def *col = nir_vec(&b, ch, 4);
   if (blended) {
      nir_def *dst = nir_load_local_pixel_agx(
         &b, 4, 32, nir_imm_intN_t(&b, 0xffff, 16), .base = 0,
         .format = PIPE_FORMAT_R8_UNORM);
      nir_def *a = nir_fmin(&b, nir_fmax(&b, ch[3], nir_imm_float(&b, 0.0f)),
                            nir_imm_float(&b, 1.0f));
      nir_def *inv_a = nir_fsub(&b, nir_imm_float(&b, 1.0f), a);
      col = nir_fadd(&b, nir_fmul(&b, col, a), nir_fmul(&b, dst, inv_a));
   }
   nir_store_local_pixel_agx(&b, col, nir_imm_intN_t(&b, 0xffff, 16),
                             nir_undef(&b, 2, 16), .base = 0,
                             .write_mask = 0xf,
                             .format = PIPE_FORMAT_R8_UNORM);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static void
inline_and_resolve(nir_shader *main_nir, nir_shader *epilog_nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(main_nir);
   nir_function_impl *eimpl = nir_shader_get_entrypoint(epilog_nir);
   struct hash_table *remap = _mesa_pointer_hash_table_create(NULL);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   nir_inline_function_impl(&b, eimpl, NULL, remap);
   _mesa_hash_table_destroy(remap, NULL);
   nir_index_ssa_defs(impl);
   nir_progress(true, impl, nir_metadata_none);

   nir_def *exported[8] = {NULL};
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_export_agx) {
            exported[nir_intrinsic_base(intr)] = intr->src[0].ssa;
            nir_instr_remove(instr);
         } else if (intr->intrinsic == nir_intrinsic_load_exported_agx) {
            nir_def_rewrite_uses(&intr->def,
                                 exported[nir_intrinsic_base(intr)]);
            nir_instr_remove(instr);
         }
      }
   }
   nir_opt_copy_prop(main_nir);
   nir_opt_constant_folding(main_nir);
   nir_opt_dce(main_nir);
}

static bool
test_preoptimized_fs_inline(bool blended, bool uniform_exports)
{
   nir_shader *src = build_test_fs_main(uniform_exports);

   /* Reference: inline into raw NIR and run full agx_compile_shader_nir(). */
   nir_shader *ref_nir = nir_shader_clone(NULL, src);
   nir_shader *ref_epilog = build_test_epilog(blended);
   inline_and_resolve(ref_nir, ref_epilog);
   ralloc_free(ref_epilog);

   struct agx_shader_key ref_key = g17p_key(false);
   struct agx_shader_part ref_part = {0};
   bool ok = agx_compile_shader_nir(ref_nir, &ref_key, &ref_part);
   ralloc_free(ref_nir);

   /* Fast path: capture pre-preamble NIR during standalone FS compile (without
    * preamble when testing uniform_exports), then deserialize, inline epilog,
    * and compile via preoptimized entrypoint with preamble enabled. */
   struct blob mono_blob;
   blob_init(&mono_blob);
   struct agx_shader_key standalone_key = g17p_key(uniform_exports);
   standalone_key.no_stop = true;
   struct agx_shader_part standalone_part = {0};
   ok &= agx_compile_shader_nir_capture(src, &standalone_key, &standalone_part,
                                        &mono_blob);
   ralloc_free(src);

   struct blob_reader reader;
   blob_reader_init(&reader, mono_blob.data, mono_blob.size);
   nir_shader *fast_nir = nir_deserialize(NULL, &agx_nir_options, &reader);
   ok &= fast_nir != NULL && !reader.overrun;

   struct agx_shader_part fast_part = {0};
   if (ok) {
      struct agx_shader_key fast_key = g17p_key(false);
      nir_shader *fast_epilog = build_test_epilog(blended);
      agx_prepare_preoptimized_epilog_nir(fast_epilog, &fast_key);
      inline_and_resolve(fast_nir, fast_epilog);
      ralloc_free(fast_epilog);
      ok &= agx_compile_shader_nir_preoptimized(fast_nir, &fast_key, &fast_part,
                                                blended, uniform_exports);
      ok &= ref_part.info.binary_size > 0 &&
            ref_part.info.binary_size == fast_part.info.binary_size &&
            memcmp(ref_part.binary, fast_part.binary,
                   ref_part.info.binary_size) == 0;
   }

   if (!ok) {
      fprintf(stderr,
              "preoptimized_fs_inline(blended=%d, uniform_exports=%d) failed: "
              "ref=%u fast=%u\n",
              blended, uniform_exports, ref_part.info.binary_size,
              fast_part.info.binary_size);
   }

   blob_finish(&mono_blob);
   free(ref_part.binary);
   free(standalone_part.binary);
   free(fast_part.binary);
   ralloc_free(fast_nir);
   return ok;
}

static bool
test_preoptimized_fs_inline_undef_clamp(void)
{
   /* Regression test: single-channel FS exporting undef on G/B/A paired with
    * an integer clamping epilog (umin(ch, 255)). nir_opt_undef must fold
    * umin(undef, 255) before preamble extraction so undef channels are not
    * hoisted into preamble stores. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "fs_main_undef_r8u");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *in = nir_load_interpolated_input(
      &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 0,
      .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   nir_def *r = nir_f2u32(&b, in);
   nir_def *u = nir_undef(&b, 1, 32);
   nir_export_agx(&b, r, .base = 0);
   nir_export_agx(&b, u, .base = 2);
   nir_export_agx(&b, u, .base = 4);
   nir_export_agx(&b, u, .base = 6);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_preprocess_nir_native(b.shader, AGX_CHIP_G17P);
   NIR_PASS(_, b.shader, agx_nir_lower_interpolation, false);
   nir_shader *src = b.shader;

   nir_builder eb = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "fs_epilog_r8u_clamp");
   nir_def *ch[4];
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *v = nir_load_exported_agx(&eb, 1, 32, .base = c * 2);
      ch[c] = nir_u2u16(&eb, nir_umin(&eb, v, nir_imm_int(&eb, 255)));
   }
   nir_store_local_pixel_agx(&eb, nir_vec(&eb, ch, 4),
                             nir_imm_intN_t(&eb, 0xffff, 16),
                             nir_undef(&eb, 2, 16), .base = 0,
                             .write_mask = 0xf,
                             .format = PIPE_FORMAT_R8_UINT);
   agx_preprocess_nir(eb.shader);
   nir_shader *epilog_proto = eb.shader;

   nir_shader *ref_nir = nir_shader_clone(NULL, src);
   nir_shader *ref_epilog = nir_shader_clone(NULL, epilog_proto);
   inline_and_resolve(ref_nir, ref_epilog);
   ralloc_free(ref_epilog);

   struct agx_shader_key ref_key = g17p_key(false);
   struct agx_shader_part ref_part = {0};
   bool ok = agx_compile_shader_nir(ref_nir, &ref_key, &ref_part);
   ralloc_free(ref_nir);

   struct blob mono_blob;
   blob_init(&mono_blob);
   struct agx_shader_key standalone_key = g17p_key(false);
   standalone_key.no_stop = true;
   struct agx_shader_part standalone_part = {0};
   ok &= agx_compile_shader_nir_capture(src, &standalone_key, &standalone_part,
                                        &mono_blob);
   ralloc_free(src);

   struct blob_reader reader;
   blob_reader_init(&reader, mono_blob.data, mono_blob.size);
   nir_shader *fast_nir = nir_deserialize(NULL, &agx_nir_options, &reader);
   ok &= fast_nir != NULL && !reader.overrun;

   struct agx_shader_part fast_part = {0};
   if (ok) {
      struct agx_shader_key fast_key = g17p_key(false);
      nir_shader *fast_epilog = nir_shader_clone(NULL, epilog_proto);
      agx_prepare_preoptimized_epilog_nir(fast_epilog, &fast_key);
      inline_and_resolve(fast_nir, fast_epilog);
      ralloc_free(fast_epilog);
      ok &= agx_compile_shader_nir_preoptimized(fast_nir, &fast_key, &fast_part,
                                                true, false);
      ok &= ref_part.info.binary_size > 0 &&
            ref_part.info.push_count == fast_part.info.push_count &&
            ref_part.info.binary_size == fast_part.info.binary_size &&
            memcmp(ref_part.binary, fast_part.binary,
                   ref_part.info.binary_size) == 0;
   }

   if (!ok) {
      fprintf(stderr,
              "preoptimized_fs_inline_undef_clamp failed: ref=(%u B, push=%u) "
              "fast=(%u B, push=%u)\n",
              ref_part.info.binary_size, ref_part.info.push_count,
              fast_part.info.binary_size, fast_part.info.push_count);
   }

   blob_finish(&mono_blob);
   ralloc_free(epilog_proto);
   free(ref_part.binary);
   free(standalone_part.binary);
   free(fast_part.binary);
   ralloc_free(fast_nir);
   return ok;
}

int
main(void)
{
   bool ok = true;
   ok &= test_post_lower_equivalence();
   ok &= test_post_lower_temp_var_fallback(false);
   ok &= test_post_lower_temp_var_fallback(true);
   ok &= test_dead_sysval_before_divergence();
   ok &= test_bit_size_guard();
   ok &= test_preoptimized_fs_inline(false, false);
   ok &= test_preoptimized_fs_inline(true, false);
   ok &= test_preoptimized_fs_inline(false, true);
   ok &= test_preoptimized_fs_inline(true, true);
   ok &= test_preoptimized_fs_inline_undef_clamp();
   if (!ok)
      return 1;
   puts("agx3_pso_compile_fastpath: PASS");
   return 0;
}

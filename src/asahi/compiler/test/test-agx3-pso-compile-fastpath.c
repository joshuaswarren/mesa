/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agx_compile.h"
#include "agx_nir.h"
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

/* Verify that native 1x32 nir_intrinsic_ballot (emitted by subgroup vote
 * lowering during the first preprocess) passes agx_assert_post_lower_invariants
 * on the post-lower fast path instead of triggering a false subgroup assert. */
static bool
test_post_lower_native_ballot(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "post_lower_native_ballot");

   nir_variable *in_var =
      nir_variable_create(b.shader, nir_var_shader_in, glsl_uint_type(), "in0");
   in_var->data.location = VARYING_SLOT_VAR0;
   in_var->data.interpolation = INTERP_MODE_FLAT;

   nir_variable *out_var =
      nir_variable_create(b.shader, nir_var_shader_out, glsl_uint_type(), "out0");
   out_var->data.location = FRAG_RESULT_DATA0;

   nir_def *cond = nir_ine_imm(&b, nir_load_var(&b, in_var), 0);
   nir_def *any = nir_vote_any(&b, 1, cond);
   nir_store_var(&b, out_var, nir_b2i32(&b, any), 0x1);

   agx_preprocess_nir_native(b.shader, AGX_CHIP_G17P);
   agx_preprocess_nir_post_lower(b.shader, AGX_CHIP_G17P);

   bool saw_ballot = false;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_ballot)
               saw_ballot = true;
         }
      }
   }
   ralloc_free(b.shader);
   return saw_ballot;
}

/* Verify that M5 compute shaders keep native 1x32 shuffle_xor (including with
 * a dynamic mask that is lowered later by agx_nir_lower_dynamic_shuffle_xor())
 * and non-ftz32 32-bit float min/max reduce across agx_preprocess_nir_post_lower. */
static bool
test_post_lower_m5_compute_shuffle_xor(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "post_lower_m5_compute_shuf_xor");

   nir_def *addr = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 0),
                                          .base = 0, .range = 16);
   nir_def *dyn_mask = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8),
                                              .base = 8, .range = 4);
   nir_def *lid = nir_load_local_invocation_index(&b);
   nir_def *lane_addr = nir_iadd(&b, addr, nir_u2u64(&b, nir_imul_imm(&b, lid, 4)));
   nir_def *val = nir_load_global(&b, 1, 32, lane_addr);
   nir_def *shuf = nir_shuffle_xor(&b, val, dyn_mask);
   nir_def *fmin = nir_reduce(&b, shuf, .reduction_op = nir_op_fmin,
                              .cluster_size = 0);
   nir_store_global(&b, fmin, lane_addr);

   agx_preprocess_nir_m5(b.shader);
   agx_preprocess_nir_post_lower(b.shader, AGX_CHIP_G17S);

   bool saw_shuf_xor = false;
   bool saw_fmin_reduce = false;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_shuffle_xor)
               saw_shuf_xor = true;
            if (intr->intrinsic == nir_intrinsic_reduce &&
                nir_intrinsic_reduction_op(intr) == nir_op_fmin)
               saw_fmin_reduce = true;
         }
      }
   }
   ralloc_free(b.shader);
   if (!saw_shuf_xor || !saw_fmin_reduce) {
      fprintf(stderr,
              "FAIL: m5 compute post_lower saw_shuf_xor=%d saw_fmin_reduce=%d\n",
              saw_shuf_xor, saw_fmin_reduce);
      return false;
   }
   return true;
}

/* Verify that G18G quad ops and clustered reductions are lowered by the first
 * preprocess (leaving native 1x32 ballot/read_invocation) and pass
 * agx_preprocess_nir_post_lower() without asserting. */
static bool
test_post_lower_g18g_quad_clustered(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "post_lower_g18g_quad_clustered");

   nir_def *addr = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 0),
                                          .base = 0, .range = 16);
   nir_def *lid = nir_load_local_invocation_index(&b);
   nir_def *lane_addr = nir_iadd(&b, addr, nir_u2u64(&b, nir_imul_imm(&b, lid, 4)));
   nir_def *val = nir_load_global(&b, 1, 32, lane_addr);
   nir_def *q = nir_quad_swap_horizontal(&b, val);
   nir_def *c = nir_reduce(&b, q, .reduction_op = nir_op_iadd,
                           .cluster_size = 4);
   nir_store_global(&b, c, lane_addr);

   agx_preprocess_nir_g18g(b.shader);
   agx_preprocess_nir_post_lower(b.shader, AGX_CHIP_G18G);

   bool saw_quad = false;
   bool saw_clustered = false;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_quad_swap_horizontal)
               saw_quad = true;
            if (intr->intrinsic == nir_intrinsic_reduce &&
                nir_intrinsic_cluster_size(intr) == 4)
               saw_clustered = true;
         }
      }
   }
   ralloc_free(b.shader);
   if (saw_quad || saw_clustered) {
      fprintf(stderr,
              "FAIL: g18g post_lower saw_quad=%d saw_clustered=%d\n",
              saw_quad, saw_clustered);
      return false;
   }
   return true;
}

#ifndef NDEBUG
/* Self-check that agx_nir_subgroup_intrinsic_needs_lowering() agrees with
 * whether agx_nir_lower_subgroups_for_target() actually makes progress across
 * every subgroup intrinsic, shape, and target profile. */
static bool
check_one_subgroup_case(const char *name, mesa_shader_stage stage, bool ftz32,
                        bool g17p, bool m5, bool g18g,
                        nir_intrinsic_instr *(*emit)(nir_builder *b,
                                                     nir_def *addr,
                                                     bool ftz32))
{
   nir_builder b =
      nir_builder_init_simple_shader(stage, &agx_nir_options, "%s", name);
   if (ftz32)
      b.shader->info.float_controls_execution_mode |=
         FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;

   nir_def *addr = nir_load_push_constant(&b, 1, 64, nir_imm_int(&b, 0),
                                          .base = 0, .range = 16);
   nir_intrinsic_instr *intr = emit(&b, addr, ftz32);
   nir_def *out = &intr->def;
   if (out->bit_size == 1)
      out = nir_b2i32(&b, out);
   nir_store_global(&b, out, addr);

   bool pred = agx_nir_subgroup_intrinsic_needs_lowering(b.shader, intr, g17p,
                                                         m5, g18g);
   bool progress =
      agx_nir_lower_subgroups_for_target(b.shader, g17p, m5, g18g);

   ralloc_free(b.shader);
   if (pred != progress) {
      fprintf(stderr,
              "FAIL: subgroup predicate mismatch on %s (stage=%d ftz32=%d "
              "g17p=%d m5=%d g18g=%d): pred=%d progress=%d\n",
              name, stage, ftz32, g17p, m5, g18g, pred, progress);
      return false;
   }
   return true;
}

#define DEF_EMIT(fname, body)                                                  \
   static nir_intrinsic_instr *emit_##fname(nir_builder *b, nir_def *addr,     \
                                            bool ftz32)                        \
   {                                                                           \
      (void)ftz32;                                                             \
      nir_def *v32 = nir_load_global(b, 1, 32, addr);                          \
      nir_def *v32x2 = nir_load_global(b, 2, 32, addr);                        \
      nir_def *v16 = nir_load_global(b, 1, 16, addr);                          \
      nir_def *v64 = nir_load_global(b, 1, 64, addr);                          \
      nir_def *cond = nir_ine_imm(b, v32, 0);                                  \
      nir_def *idx = nir_load_global(b, 1, 32, nir_iadd_imm(b, addr, 16));     \
      nir_def *m4 = nir_load_global(b, 4, 32, addr);                           \
      nir_def *f32 = ftz32 ? nir_fcanonicalize(b, v32) : v32;                  \
      (void)v32; (void)v32x2; (void)v16; (void)v64; (void)cond;                \
      (void)idx; (void)m4; (void)f32;                                          \
      nir_def *res = (body);                                                   \
      return nir_def_as_intrinsic(res);                                        \
   }

DEF_EMIT(ballot_1x32, nir_ballot(b, 1, 32, cond))
DEF_EMIT(ballot_4x32, nir_ballot(b, 4, 32, cond))
DEF_EMIT(ballot_1x64, nir_ballot(b, 1, 64, cond))
DEF_EMIT(elect, nir_elect(b, 1))
DEF_EMIT(first_inv, nir_first_invocation(b))
DEF_EMIT(last_inv, nir_last_invocation(b))
DEF_EMIT(vote_any, nir_vote_any(b, 1, cond))
DEF_EMIT(vote_all, nir_vote_all(b, 1, cond))
DEF_EMIT(vote_ieq, nir_vote_ieq(b, 1, v32))
DEF_EMIT(vote_feq, nir_vote_feq(b, 1, v32))
DEF_EMIT(quad_vote_any, nir_quad_vote_any(b, 1, cond))
DEF_EMIT(quad_vote_all, nir_quad_vote_all(b, 1, cond))
DEF_EMIT(subgroup_size, nir_load_subgroup_size(b))
DEF_EMIT(num_subgroups, nir_load_num_subgroups(b))
DEF_EMIT(eq_mask, nir_load_subgroup_eq_mask(b, 4, 32))
DEF_EMIT(ge_mask, nir_load_subgroup_ge_mask(b, 4, 32))
DEF_EMIT(gt_mask, nir_load_subgroup_gt_mask(b, 4, 32))
DEF_EMIT(le_mask, nir_load_subgroup_le_mask(b, 4, 32))
DEF_EMIT(lt_mask, nir_load_subgroup_lt_mask(b, 4, 32))
DEF_EMIT(inv_ballot, nir_inverse_ballot(b, m4))
DEF_EMIT(ballot_bfe, nir_ballot_bitfield_extract(b, m4, idx))
DEF_EMIT(ballot_cnt, nir_ballot_bit_count_reduce(b, m4))
DEF_EMIT(ballot_lsb, nir_ballot_find_lsb(b, m4))
DEF_EMIT(ballot_msb, nir_ballot_find_msb(b, m4))
DEF_EMIT(ballot_exc, nir_ballot_bit_count_exclusive(b, m4))
DEF_EMIT(ballot_inc, nir_ballot_bit_count_inclusive(b, m4))
DEF_EMIT(read_inv_s, nir_read_invocation(b, v32, idx))
DEF_EMIT(read_inv_v2, nir_read_invocation(b, v32x2, idx))
DEF_EMIT(read_first, nir_read_first_invocation(b, v32))
DEF_EMIT(quad_bcast_s, nir_quad_broadcast(b, v32, nir_imm_int(b, 1)))
DEF_EMIT(quad_bcast_v2, nir_quad_broadcast(b, v32x2, nir_imm_int(b, 1)))
DEF_EMIT(quad_h_s, nir_quad_swap_horizontal(b, v32))
DEF_EMIT(quad_h_v2, nir_quad_swap_horizontal(b, v32x2))
DEF_EMIT(quad_v_s, nir_quad_swap_vertical(b, v32))
DEF_EMIT(quad_v_v2, nir_quad_swap_vertical(b, v32x2))
DEF_EMIT(quad_d_s, nir_quad_swap_diagonal(b, v32))
DEF_EMIT(quad_d_v2, nir_quad_swap_diagonal(b, v32x2))
DEF_EMIT(shuffle, nir_shuffle(b, v32, idx))
DEF_EMIT(shuf_xor_32c, nir_shuffle_xor(b, v32, nir_imm_int(b, 1)))
DEF_EMIT(shuf_xor_32d, nir_shuffle_xor(b, v32, idx))
DEF_EMIT(shuf_xor_16, nir_shuffle_xor(b, v16, nir_imm_int(b, 1)))
DEF_EMIT(shuf_xor_v2, nir_shuffle_xor(b, v32x2, nir_imm_int(b, 1)))
DEF_EMIT(shuf_up, nir_shuffle_up(b, v32, idx))
DEF_EMIT(shuf_down, nir_shuffle_down(b, v32, idx))
DEF_EMIT(rotate, nir_rotate(b, v32, idx, .cluster_size = 0))
DEF_EMIT(red_iadd_32, nir_reduce(b, v32, .reduction_op = nir_op_iadd, .cluster_size = 0))
DEF_EMIT(red_iadd_64, nir_reduce(b, v64, .reduction_op = nir_op_iadd, .cluster_size = 0))
DEF_EMIT(red_iadd_v2, nir_reduce(b, v32x2, .reduction_op = nir_op_iadd, .cluster_size = 0))
DEF_EMIT(red_iadd_c2, nir_reduce(b, v32, .reduction_op = nir_op_iadd, .cluster_size = 2))
DEF_EMIT(red_iadd_c4, nir_reduce(b, v32, .reduction_op = nir_op_iadd, .cluster_size = 4))
DEF_EMIT(red_iadd_c32, nir_reduce(b, v32, .reduction_op = nir_op_iadd, .cluster_size = 32))
DEF_EMIT(red_imin_32, nir_reduce(b, v32, .reduction_op = nir_op_imin, .cluster_size = 0))
DEF_EMIT(red_imin_64, nir_reduce(b, v64, .reduction_op = nir_op_imin, .cluster_size = 0))
DEF_EMIT(red_imul_32, nir_reduce(b, v32, .reduction_op = nir_op_imul, .cluster_size = 0))
DEF_EMIT(red_fmin_32, nir_reduce(b, f32, .reduction_op = nir_op_fmin, .cluster_size = 0))
DEF_EMIT(red_fmin_32_c4, nir_reduce(b, f32, .reduction_op = nir_op_fmin, .cluster_size = 4))
DEF_EMIT(red_fmax_32_c32, nir_reduce(b, f32, .reduction_op = nir_op_fmax, .cluster_size = 32))
DEF_EMIT(red_fmin_16_c4, nir_reduce(b, v16, .reduction_op = nir_op_fmin, .cluster_size = 4))
DEF_EMIT(red_fadd_32, nir_reduce(b, v32, .reduction_op = nir_op_fadd, .cluster_size = 0))
DEF_EMIT(red_fadd_64, nir_reduce(b, v64, .reduction_op = nir_op_fadd, .cluster_size = 0))
DEF_EMIT(inc_iadd_32, nir_inclusive_scan(b, v32, .reduction_op = nir_op_iadd))
DEF_EMIT(inc_fmin_32, nir_inclusive_scan(b, f32, .reduction_op = nir_op_fmin))
DEF_EMIT(exc_iadd_32, nir_exclusive_scan(b, v32, .reduction_op = nir_op_iadd))
DEF_EMIT(exc_iadd_64, nir_exclusive_scan(b, v64, .reduction_op = nir_op_iadd))
DEF_EMIT(exc_imin_64, nir_exclusive_scan(b, v64, .reduction_op = nir_op_imin))
DEF_EMIT(exc_imul_32, nir_exclusive_scan(b, v32, .reduction_op = nir_op_imul))
DEF_EMIT(exc_fmin_32, nir_exclusive_scan(b, f32, .reduction_op = nir_op_fmin))

static bool
test_subgroup_predicate_matches_lowering(void)
{
   static const struct {
      const char *name;
      nir_intrinsic_instr *(*emit)(nir_builder *b, nir_def *addr, bool ftz32);
   } cases[] = {
      {"ballot_1x32", emit_ballot_1x32},
      {"ballot_4x32", emit_ballot_4x32},
      {"ballot_1x64", emit_ballot_1x64},
      {"elect", emit_elect},
      {"first_inv", emit_first_inv},
      {"last_inv", emit_last_inv},
      {"vote_any", emit_vote_any},
      {"vote_all", emit_vote_all},
      {"vote_ieq", emit_vote_ieq},
      {"vote_feq", emit_vote_feq},
      {"quad_vote_any", emit_quad_vote_any},
      {"quad_vote_all", emit_quad_vote_all},
      {"subgroup_size", emit_subgroup_size},
      {"num_subgroups", emit_num_subgroups},
      {"eq_mask", emit_eq_mask},
      {"ge_mask", emit_ge_mask},
      {"gt_mask", emit_gt_mask},
      {"le_mask", emit_le_mask},
      {"lt_mask", emit_lt_mask},
      {"inv_ballot", emit_inv_ballot},
      {"ballot_bfe", emit_ballot_bfe},
      {"ballot_cnt", emit_ballot_cnt},
      {"ballot_lsb", emit_ballot_lsb},
      {"ballot_msb", emit_ballot_msb},
      {"ballot_exc", emit_ballot_exc},
      {"ballot_inc", emit_ballot_inc},
      {"read_inv_s", emit_read_inv_s},
      {"read_inv_v2", emit_read_inv_v2},
      {"read_first", emit_read_first},
      {"quad_bcast_s", emit_quad_bcast_s},
      {"quad_bcast_v2", emit_quad_bcast_v2},
      {"quad_h_s", emit_quad_h_s},
      {"quad_h_v2", emit_quad_h_v2},
      {"quad_v_s", emit_quad_v_s},
      {"quad_v_v2", emit_quad_v_v2},
      {"quad_d_s", emit_quad_d_s},
      {"quad_d_v2", emit_quad_d_v2},
      {"shuffle", emit_shuffle},
      {"shuf_xor_32c", emit_shuf_xor_32c},
      {"shuf_xor_32d", emit_shuf_xor_32d},
      {"shuf_xor_16", emit_shuf_xor_16},
      {"shuf_xor_v2", emit_shuf_xor_v2},
      {"shuf_up", emit_shuf_up},
      {"shuf_down", emit_shuf_down},
      {"rotate", emit_rotate},
      {"red_iadd_32", emit_red_iadd_32},
      {"red_iadd_64", emit_red_iadd_64},
      {"red_iadd_v2", emit_red_iadd_v2},
      {"red_iadd_c2", emit_red_iadd_c2},
      {"red_iadd_c4", emit_red_iadd_c4},
      {"red_iadd_c32", emit_red_iadd_c32},
      {"red_imin_32", emit_red_imin_32},
      {"red_imin_64", emit_red_imin_64},
      {"red_imul_32", emit_red_imul_32},
      {"red_fmin_32", emit_red_fmin_32},
      {"red_fmin_32_c4", emit_red_fmin_32_c4},
      {"red_fmax_32_c32", emit_red_fmax_32_c32},
      {"red_fmin_16_c4", emit_red_fmin_16_c4},
      {"red_fadd_32", emit_red_fadd_32},
      {"red_fadd_64", emit_red_fadd_64},
      {"inc_iadd_32", emit_inc_iadd_32},
      {"inc_fmin_32", emit_inc_fmin_32},
      {"exc_iadd_32", emit_exc_iadd_32},
      {"exc_iadd_64", emit_exc_iadd_64},
      {"exc_imin_64", emit_exc_imin_64},
      {"exc_imul_32", emit_exc_imul_32},
      {"exc_fmin_32", emit_exc_fmin_32},
   };

   static const struct {
      mesa_shader_stage stage;
      bool ftz32;
      bool g17p;
      bool m5;
      bool g18g;
   } profiles[] = {
      {MESA_SHADER_COMPUTE, false, false, false, false}, /* AGX2 */
      {MESA_SHADER_COMPUTE, false, true, false, false},  /* G17P */
      {MESA_SHADER_COMPUTE, false, false, true, false},  /* M5 compute */
      {MESA_SHADER_COMPUTE, true, false, true, false},   /* M5 compute ftz32 */
      {MESA_SHADER_FRAGMENT, false, false, true, false}, /* M5 fragment */
      {MESA_SHADER_COMPUTE, false, false, true, true},   /* G18G compute */
   };

   bool ok = true;
   for (unsigned p = 0; p < ARRAY_SIZE(profiles); ++p) {
      for (unsigned c = 0; c < ARRAY_SIZE(cases); ++c) {
         ok &= check_one_subgroup_case(cases[c].name, profiles[p].stage,
                                       profiles[p].ftz32, profiles[p].g17p,
                                       profiles[p].m5, profiles[p].g18g,
                                       cases[c].emit);
      }
   }
   return ok;
}
#endif

static bool
block_in_loop(nir_block *block, nir_loop *loop)
{
   for (nir_cf_node *n = &block->cf_node; n; n = n->parent) {
      if (n == &loop->cf_node)
         return true;
   }
   return false;
}

static bool
ungroup_loop_invariant_add(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool rewrote = false;

   nir_foreach_block(block, impl) {
      nir_loop *loop = NULL;
      for (nir_cf_node *n = &block->cf_node; n; n = n->parent) {
         if (n->type == nir_cf_node_loop) {
            loop = nir_cf_node_as_loop(n);
            break;
         }
      }
      if (!loop)
         continue;

      /* Match the reassociated 4-term chain ((inv + v1) + v2) where
       * inv = u0 + u1 has two loop-invariant terms and v1, v2 are loop-varying,
       * and rewrite it back to (((v1 + u0) + v2) + u1). Because length == 4 and
       * the chain is non-local, nir_opt_reassociate's CSE heuristic is skipped
       * and only nir_reassociate_loop_invariant can regroup (u0 + u1). */
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_alu)
            continue;
         nir_alu_instr *outer = nir_instr_as_alu(instr);
         if (outer->op != nir_op_iadd || outer->def.bit_size != 32)
            continue;

         for (unsigned os = 0; os < 2; ++os) {
            nir_instr *mid_instr = nir_def_instr(outer->src[os].src.ssa);
            nir_def *v2 = outer->src[1 - os].src.ssa;
            if (mid_instr->type != nir_instr_type_alu ||
                !block_in_loop(nir_def_instr(v2)->block, loop))
               continue;
            nir_alu_instr *mid = nir_instr_as_alu(mid_instr);
            if (mid->op != nir_op_iadd || mid->def.bit_size != 32)
               continue;

            for (unsigned ms = 0; ms < 2; ++ms) {
               nir_instr *inv_instr = nir_def_instr(mid->src[ms].src.ssa);
               nir_def *v1 = mid->src[1 - ms].src.ssa;
               if (inv_instr->type != nir_instr_type_alu ||
                   !block_in_loop(nir_def_instr(v1)->block, loop))
                  continue;
               nir_alu_instr *inv = nir_instr_as_alu(inv_instr);
               if (inv->op != nir_op_iadd || inv->def.bit_size != 32)
                  continue;

               nir_def *u0 = inv->src[0].src.ssa;
               nir_def *u1 = inv->src[1].src.ssa;
               if (block_in_loop(nir_def_instr(u0)->block, loop) ||
                   block_in_loop(nir_def_instr(u1)->block, loop))
                  continue;

               nir_builder b = nir_builder_at(nir_before_instr(&outer->instr));
               nir_def *ungrouped = nir_iadd(
                  &b, nir_iadd(&b, nir_iadd(&b, v1, u0), v2), u1);
               nir_def_rewrite_uses(&outer->def, ungrouped);
               nir_instr_remove(&outer->instr);
               if (nir_def_is_unused(&mid->def))
                  nir_instr_remove(&mid->instr);
               if (nir_def_is_unused(&inv->def))
                  nir_instr_remove(&inv->instr);
               rewrote = true;
               break;
            }
            if (rewrote)
               break;
         }
         if (rewrote)
            break;
      }
      if (rewrote)
         break;
   }

   if (rewrote)
      nir_progress(true, impl, nir_metadata_control_flow);
   return rewrote;
}

static bool
test_preoptimized_fs_inline_loop_invariant_reassoc(bool blended,
                                                   bool uniform_exports)
{
   /* Regression test: serialize a pre-preamble SSA fragment shader whose loop
    * still has an ungrouped invariant chain ((i + u0) + u1) with byte_indices
    * enabled (G17P). Full compilation passes nir_reassociate_loop_invariant to
    * group (u0 + u1) and hoist it into the preamble; the pre-preamble fast
    * path must pass the same flag so fast and full outputs match byte-for-byte
    * (reverting the fast-path byte_indices gate causes a mismatch here). */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      uniform_exports ? "fs_loop_invar_uniform" : "fs_loop_invar_main");
   nir_function_impl *impl = nir_shader_get_entrypoint(b.shader);

   nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *count = nir_load_global_constant(
      &b, 1, 32, addr, .align_mul = 4);
   nir_def *u0 = nir_load_global_constant(
      &b, 1, 32, nir_iadd_imm(&b, addr, 16), .align_mul = 4);
   nir_def *u1 = nir_load_global_constant(
      &b, 1, 32, nir_iadd_imm(&b, addr, 32), .align_mul = 4);

   nir_def *in = NULL;
   if (!uniform_exports) {
      nir_def *bary = nir_load_barycentric_pixel(
         &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
      in = nir_load_interpolated_input(
         &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0,
         .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
   }

   nir_variable *idx_var =
      nir_local_variable_create(impl, glsl_uint_type(), "idx");
   nir_variable *acc_var =
      nir_local_variable_create(impl, glsl_uint_type(), "acc");
   nir_store_var(&b, idx_var, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

   nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, idx_var);
      nir_push_if(&b, nir_uge(&b, i, count));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);

      nir_def *a = nir_load_var(&b, acc_var);
      nir_def *v2 = nir_ixor(&b, a, i);
      nir_def *x = nir_iadd(&b, nir_iadd(&b, nir_iadd(&b, i, u0), v2), u1);
      nir_store_var(&b, acc_var, nir_ixor(&b, a, x), 1);
      nir_store_var(&b, idx_var, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, NULL);

   nir_def *sum = nir_u2f32(&b, nir_load_var(&b, acc_var));
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *val = uniform_exports
         ? nir_fmul_imm(&b, sum, 0.25f * (c + 1))
         : nir_fadd(&b, nir_channel(&b, in, c), sum);
      nir_export_agx(&b, val, .base = c * 2);
   }
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_preprocess_nir_native(b.shader, AGX_CHIP_G17P);
   NIR_PASS(_, b.shader, agx_nir_lower_interpolation, false);
   nir_shader *src = b.shader;

   nir_shader *ref_nir = nir_shader_clone(NULL, src);
   nir_shader *ref_epilog = build_test_epilog(blended);
   inline_and_resolve(ref_nir, ref_epilog);
   ralloc_free(ref_epilog);

   struct agx_shader_key ref_key = g17p_key(false);
   struct agx_shader_part ref_part = {0};
   bool ok = agx_compile_shader_nir(ref_nir, &ref_key, &ref_part);
   ralloc_free(ref_nir);

   /* Capture pre-preamble NIR, ungroup the loop-invariant add chain, and
    * re-serialize so mono_blob carries an ungrouped ((i + u0) + u1) chain.
    * Only agx_compile_shader_nir_preoptimized()'s fast-path reassociation can
    * re-group (u0 + u1) and match the full compile. */
   struct blob captured_blob;
   blob_init(&captured_blob);
   struct agx_shader_key standalone_key = g17p_key(uniform_exports);
   standalone_key.no_stop = true;
   struct agx_shader_part standalone_part = {0};
   ok &= agx_compile_shader_nir_capture(src, &standalone_key, &standalone_part,
                                        &captured_blob);
   ralloc_free(src);

   struct blob_reader cap_reader;
   blob_reader_init(&cap_reader, captured_blob.data, captured_blob.size);
   nir_shader *ungrouped_nir =
      nir_deserialize(NULL, &agx_nir_options, &cap_reader);
   ok &= ungrouped_nir != NULL && !cap_reader.overrun;
   if (ok)
      ok &= ungroup_loop_invariant_add(ungrouped_nir);

   struct blob mono_blob;
   blob_init(&mono_blob);
   if (ok)
      nir_serialize(&mono_blob, ungrouped_nir, false);
   ralloc_free(ungrouped_nir);
   blob_finish(&captured_blob);

   nir_shader *fast_nir = NULL;
   if (ok) {
      struct blob_reader reader;
      blob_reader_init(&reader, mono_blob.data, mono_blob.size);
      fast_nir = nir_deserialize(NULL, &agx_nir_options, &reader);
      ok &= fast_nir != NULL && !reader.overrun;
   }

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
            ref_part.info.push_count == fast_part.info.push_count &&
            ref_part.info.binary_size == fast_part.info.binary_size &&
            memcmp(ref_part.binary, fast_part.binary,
                   ref_part.info.binary_size) == 0;
   }

   if (!ok) {
      fprintf(stderr,
              "preoptimized_fs_inline_loop_invariant_reassoc(blended=%d, "
              "uniform_exports=%d) failed: ref=(%u B, push=%u) fast=(%u B, "
              "push=%u)\n",
              blended, uniform_exports, ref_part.info.binary_size,
              ref_part.info.push_count, fast_part.info.binary_size,
              fast_part.info.push_count);
   }

   blob_finish(&mono_blob);
   free(ref_part.binary);
   free(standalone_part.binary);
   free(fast_part.binary);
   ralloc_free(fast_nir);
   return ok;
}

/* Verify that capturing post-agx_optimize_nir NIR via
 * agx_compile_shader_nir_capture_ex() and compiling a trivial non-ALU epilog
 * via agx_compile_shader_nir_postopt() produces bit-exact machine code
 * matching agx_compile_shader_nir_preoptimized(). */
static bool
test_postopt_fs_inline_equivalence(void)
{
   nir_shader *src = build_test_fs_main(false);
   nir_shader *ref_src = nir_shader_clone(NULL, src);
   const nir_shader_compiler_options *src_options = src->options;

   struct agx_shader_key cap_key = g17p_key(true);
   struct agx_shader_part standalone_part = {0};
   struct blob pre_preamble_blob;
   blob_init(&pre_preamble_blob);
   nir_shader *post_opt_nir = NULL;

   bool ok = agx_compile_shader_nir_capture_ex(
      src, &cap_key, &standalone_part, &pre_preamble_blob, &post_opt_nir);
   ralloc_free(src);
   ok &= post_opt_nir != NULL && pre_preamble_blob.size > 0;
   /* The capture outlives the compile: it must not keep the compiler's
    * stack-local per-encoding options. */
   if (post_opt_nir && post_opt_nir->options != src_options) {
      fprintf(stderr, "postopt capture kept transient compiler options\n");
      ok = false;
   }

   struct agx_shader_part ref_part = {0}, fast_part = {0};
   if (ok) {
      struct agx_shader_key ref_key = g17p_key(true);
      nir_shader *ref_epilog = build_test_epilog(false);
      inline_and_resolve(ref_src, ref_epilog);
      ralloc_free(ref_epilog);
      ok &= agx_compile_shader_nir(ref_src, &ref_key, &ref_part);
   }
   ralloc_free(ref_src);

   if (ok) {
      struct agx_shader_key fast_key = g17p_key(true);
      nir_shader *fast_epilog = build_test_epilog(false);
      inline_and_resolve(post_opt_nir, fast_epilog);
      ralloc_free(fast_epilog);
      ok &= agx_compile_shader_nir_postopt(post_opt_nir, &fast_key, &fast_part);
      ok &= ref_part.info.binary_size > 0 &&
            ref_part.info.binary_size == fast_part.info.binary_size &&
            memcmp(ref_part.binary, fast_part.binary,
                   ref_part.info.binary_size) == 0;
   }

   if (!ok) {
      fprintf(stderr,
              "postopt_fs_inline_equivalence failed: ref=%u fast=%u\n",
              ref_part.info.binary_size, fast_part.info.binary_size);
   }

   blob_finish(&pre_preamble_blob);
   free(standalone_part.binary);
   free(ref_part.binary);
   free(fast_part.binary);
   ralloc_free(post_opt_nir);
   return ok;
}

/* Verify that nir_opt_fp_math_ctrl early-out still marks IO loads
 * no_signed_zero and still clears preserve_signed_zero on unused-sign ops. */
static bool
test_fp_math_ctrl_fast_paths(void)
{
   /* Case 1: shader with no fp_math_ctrl flags and no IO loads. */
   nir_builder b1 = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "fp_ctrl_none");
   nir_def *base = nir_load_preamble(&b1, 1, 64, .base = 0);
   nir_def *id = nir_load_local_invocation_index(&b1);
   nir_def *v = nir_fadd(&b1, nir_u2f32(&b1, id), nir_imm_float(&b1, 1.0f));
   nir_store_global(&b1, v, base, .align_mul = 4);
   bool ok = !nir_opt_fp_math_ctrl(b1.shader);
   ralloc_free(b1.shader);

   /* Case 2: preserve_signed_zero on fadd whose consumer does not observe -0. */
   nir_builder b2 = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "fp_ctrl_sz_clear");
   nir_def *in = nir_load_input(&b2, 1, 32, nir_imm_int(&b2, 0),
                                .base = 0,
                                .io_semantics = {.location = VARYING_SLOT_VAR0,
                                                 .num_slots = 1});
   nir_def *sum = nir_fadd(&b2, in, nir_imm_float(&b2, 2.0f));
   nir_def_as_alu(sum)->fp_math_ctrl = nir_fp_preserve_signed_zero;
   nir_def *abs_sum = nir_fabs(&b2, sum);
   nir_def_as_alu(abs_sum)->fp_math_ctrl = nir_fp_preserve_signed_zero;
   nir_export_agx(&b2, abs_sum, .base = 0);

   ok &= nir_opt_fp_math_ctrl(b2.shader);
   ok &= !(nir_def_as_alu(sum)->fp_math_ctrl & nir_fp_preserve_signed_zero);
   ok &= nir_intrinsic_io_semantics(nir_def_as_intrinsic(in)).no_signed_zero;
   ralloc_free(b2.shader);

   if (!ok)
      fprintf(stderr, "fp_math_ctrl_fast_paths failed\n");
   return ok;
}

int
main(void)
{
   bool ok = true;
   ok &= test_post_lower_equivalence();
   ok &= test_post_lower_native_ballot();
   ok &= test_post_lower_m5_compute_shuffle_xor();
   ok &= test_post_lower_g18g_quad_clustered();
#ifndef NDEBUG
   ok &= test_subgroup_predicate_matches_lowering();
#endif
   ok &= test_post_lower_temp_var_fallback(false);
   ok &= test_post_lower_temp_var_fallback(true);
   ok &= test_dead_sysval_before_divergence();
   ok &= test_bit_size_guard();
   ok &= test_preoptimized_fs_inline(false, false);
   ok &= test_preoptimized_fs_inline(true, false);
   ok &= test_preoptimized_fs_inline(false, true);
   ok &= test_preoptimized_fs_inline(true, true);
   ok &= test_preoptimized_fs_inline_undef_clamp();
   ok &= test_preoptimized_fs_inline_loop_invariant_reassoc(true, false);
   ok &= test_preoptimized_fs_inline_loop_invariant_reassoc(false, true);
   ok &= test_preoptimized_fs_inline_loop_invariant_reassoc(true, true);
   ok &= test_postopt_fs_inline_equivalence();
   ok &= test_fp_math_ctrl_fast_paths();
   if (!ok)
      return 1;
   puts("agx3_pso_compile_fastpath: PASS");
   return 0;
}

/*
 * Copyright 2023 Alyssa Rosenzweig
 * Copyright 2020 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "asahi_clc.h"
#include "asahi/compiler/agx_compile.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/glsl_types.h"
#include "compiler/spirv/nir_spirv.h"
#include "nir.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_precompiled.h"
#include "shader_enums.h"

#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "util/macros.h"
#include "util/set.h"
#include <sys/mman.h>

const char *targets[] = {"g13g", "g13x", "g17p", "g15s", "g15g", "g16g",
                         "g17g", "g18g"};

/* G18G compiles every qualified kernel, like the other targets: a refusal,
 * or a kernel needing scratch (its compute profile,
 * agx3_g18g_compute_program_generate, takes no TPS frame for libagx), fails
 * the build. A kernel the G18G encoders cannot take yet may only be left out
 * by name here; agx_launch_precomp then refuses its NULL entry at launch.
 * Keep this list empty unless a qualified kernel is new to G18G. */
static const char *const g18g_omittable_programs[] = {
   NULL,
};

static bool
target_omits_refused_program(const char *target, const char *program)
{
   if (strcmp(target, "g18g"))
      return false;

   for (const char *const *p = g18g_omittable_programs; *p; ++p) {
      if (!strcmp(*p, program))
         return true;
   }
   return false;
}

#define foreach_target(target)                                                 \
   for (const char **target = &targets[0];                                     \
        target < &targets[ARRAY_SIZE(targets)]; ++target)

static enum agx_chip
target_chip(const char *target)
{
   if (!strcmp(target, "g17p"))
      return AGX_CHIP_G17P;
   else if (!strcmp(target, "g17g"))
      return AGX_CHIP_G17G;
   else if (!strcmp(target, "g15s"))
      return AGX_CHIP_G15S;
   else if (!strcmp(target, "g15g"))
      return AGX_CHIP_G15G;
   else if (!strcmp(target, "g16g"))
      return AGX_CHIP_G16G;
   /* G18G compiles as G17G (agx3c) and re-encodes its code as agx3d, exactly
    * as the driver's own compiles do (agx_device_key::emit_agx3d). */
   else if (!strcmp(target, "g18g"))
      return AGX_CHIP_G17G;
   else if (!strcmp(target, "g13x"))
      return AGX_CHIP_G13X;
   else
      return AGX_CHIP_G13G;
}

/* Whether a target runs on the native AGX3 path rather than AGX2. */
static bool
target_is_native_agx3(const char *target)
{
   return agx_chip_native_agx3(target_chip(target));
}

/* Whether a target executes AGX3 code: the native AGX3B path or the M5 AGX3C
 * one. */
static bool
target_is_agx3(const char *target)
{
   const enum agx_chip chip = target_chip(target);
   return agx_chip_native_agx3(chip) || agx_chip_g17_m5(chip);
}

static const struct spirv_to_nir_options spirv_options = {
   .environment = NIR_SPIRV_OPENCL,
   .shared_addr_format = nir_address_format_62bit_generic,
   .global_addr_format = nir_address_format_62bit_generic,
   .temp_addr_format = nir_address_format_62bit_generic,
   .constant_addr_format = nir_address_format_64bit_global,
   .create_library = true,
   .printf = true,
};

/* Standard optimization loop */
static void
optimize(nir_shader *nir)
{
   bool progress;
   do {
      progress = false;

      NIR_PASS(progress, nir, nir_split_var_copies);
      NIR_PASS(progress, nir, nir_split_struct_vars, nir_var_function_temp);
      NIR_PASS(progress, nir, nir_lower_var_copies);
      NIR_PASS(progress, nir, nir_lower_vars_to_ssa);

      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_lower_all_phis_to_scalar);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);

      nir_opt_peephole_select_options peephole_select_options = {
         .limit = 64,
         .expensive_alu_ok = true,
      };
      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &peephole_select_options);
      NIR_PASS(progress, nir, nir_opt_phi_precision);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);

      NIR_PASS(progress, nir, nir_opt_deref);
      NIR_PASS(progress, nir, nir_opt_copy_prop_vars);
      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_lower_undef_to_zero, NULL);

      NIR_PASS(progress, nir, nir_opt_shrink_vectors, true);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);

   } while (progress);
}

static nir_shader *
compile(void *memctx, const uint32_t *spirv, size_t spirv_size)
{
   const nir_shader_compiler_options *nir_options = &agx_nir_options;

   assert(spirv_size % 4 == 0);
   nir_shader *nir =
      spirv_to_nir(spirv, spirv_size / 4, NULL, MESA_SHADER_KERNEL, "library",
                   &spirv_options, nir_options);
   nir_validate_shader(nir, "after spirv_to_nir");
   ralloc_steal(memctx, nir);

   nir_fixup_is_exported(nir);

   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_lower_calls_to_builtins);

   nir_lower_compute_system_values_options cs = {.global_id_is_32bit = true};
   NIR_PASS(_, nir, nir_lower_compute_system_values, &cs);

   NIR_PASS(_, nir, nir_lower_printf,
            &(const struct nir_lower_printf_options){
               .hash_format_strings = true,
            });

   /* We have to lower away local constant initializers right before we
    * inline functions.  That way they get properly initialized at the top
    * of the function and not at the top of its caller.
    */
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_remove_non_exported(nir);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_deref);

   /* We can't deal with constant data, get rid of it */
   nir_lower_constant_to_temp(nir);

   /* We can go ahead and lower the rest of the constant initializers.  We do
    * this here so that nir_remove_dead_variables and split_per_member_structs
    * below see the corresponding stores.
    */
   NIR_PASS(_, nir, nir_lower_variable_initializers, ~0);

   /* LLVM loves take advantage of the fact that vec3s in OpenCL are 16B
    * aligned and so it can just read/write them as vec4s.  This results in a
    * LOT of vec4->vec3 casts on loads and stores.  One solution to this
    * problem is to get rid of all vec3 variables.
    */
   NIR_PASS(_, nir, nir_lower_vec3_to_vec4,
            nir_var_shader_temp | nir_var_function_temp | nir_var_mem_shared |
               nir_var_mem_global | nir_var_mem_constant);

   /* We assign explicit types early so that the optimizer can take advantage
    * of that information and hopefully get rid of some of our memcpys.
    */
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
            nir_var_uniform | nir_var_shader_temp | nir_var_function_temp |
               nir_var_mem_shared | nir_var_mem_global,
            glsl_get_cl_type_size_align);

   optimize(nir);

   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_all, NULL);

   /* Lower again, this time after dead-variables to get more compact variable
    * layouts.
    */
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types,
            nir_var_shader_temp | nir_var_function_temp | nir_var_mem_shared |
               nir_var_mem_global | nir_var_mem_constant,
            glsl_get_cl_type_size_align);
   assert(nir->constant_data_size == 0);

   NIR_PASS(_, nir, nir_lower_memcpy);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_constant,
            nir_address_format_64bit_global);

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_uniform,
            nir_address_format_32bit_offset_as_64bit);

   /* Note: we cannot lower explicit I/O here, because we need derefs in tact
    * for function calls into the library to work.
    */

   NIR_PASS(_, nir, nir_lower_convert_alu_types, NULL);
   NIR_PASS(_, nir, nir_opt_if, 0);
   NIR_PASS(_, nir, nir_opt_idiv_const, 16);

   NIR_PASS(_, nir, agx_nir_lower_texture_early, false /* support_lod_bias */);
   NIR_PASS(_, nir, agx_nir_lower_texture);
   NIR_PASS(_, nir, agx_nir_lower_multisampled_image_store);

   optimize(nir);

   return nir;
}

static void
print_shader(FILE *fp, const char *name, const char *suffix, uint32_t variant,
             struct agx_shader_part *p, uint16_t printf_uniform)
{
   struct agx_precompiled_kernel_info info = agx_compact_kernel_info(&p->info);
   info.printf_uniform = printf_uniform;
   size_t sz_B = sizeof(info) + p->info.binary_size;
   size_t sz_el = DIV_ROUND_UP(sz_B, 4);
   uint32_t *mem = calloc(sz_el, 4);

   memcpy(mem, &info, sizeof(info));
   memcpy((uint8_t *)mem + sizeof(info), p->binary, p->info.binary_size);

   nir_precomp_print_blob(fp, name, suffix, variant, mem, sz_B, true);
   free(mem);
}

static bool
uses_printf_address(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic == nir_intrinsic_load_printf_buffer_address)
      *(bool *)data = true;

   return false;
}

/* AGX3 has no fixed printf buffer address, so a kernel reads it from a 64-bit
 * uniform after its arguments, which the driver fills in at dispatch.
 */
static bool
lower_printf_address_to_uniform(nir_builder *b, nir_intrinsic_instr *intr,
                                void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_printf_buffer_address)
      return false;

   const unsigned *base_halfs = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_load_preamble(b, 1, 64, .base = *base_halfs));
   return true;
}

static bool
gather_atomic_info(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   bool *any = data;

   switch (intr->intrinsic) {
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_agx:
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_global_atomic_swap:
   case nir_intrinsic_global_atomic_swap_agx:
   case nir_intrinsic_deref_atomic_swap:
      *any = true;
      return false;
   default:
      return false;
   }
}

/* G13X variants are only compiled when atomics are used */
static const char *
remap_variant(nir_function *func, unsigned variant, const char *target)
{
   /* pass_flags is only 32-bit */
   assert(variant < 32 && "maximum # of variants");

   bool has_atomic = func->pass_flags & BITFIELD_BIT(variant);

   if (!has_atomic && !strcmp(target, "g13x"))
      return "g13g";
   else
      return target;
}

static bool
target_supports_program(const char *target, const nir_function *func)
{
   if (!strcmp(func->name, "libagx_m5_decompress_rgba16f"))
      return !strcmp(target, "g17g");
   if (!strcmp(func->name, "libagx_g17p_decompress_16b"))
      return !strcmp(target, "g17p") || !strcmp(target, "g17g");

   if (!strcmp(target, "g17g") &&
       !strncmp(func->name, "libagx_d3d12_", strlen("libagx_d3d12_")))
      return false;
   /* The range copy uses the native query command's producer ordering and
    * volatile WAIT loop. Never select this program for the AGX2/G13 path. */
   if (!strcmp(func->name, "libagx_copy_query_g17p"))
      return target_is_native_agx3(target);

   /* AGX3 internal programs need both native machine code and the HAL200
    * Ioto/CDM launch contract. Keep unqualified slots NULL and fail closed.
    * Fast clear is required by the qualified G17P compressed color shapes;
    * compressed render-pass clears dispatch this program for metadata/body.
    *
    * G18G launches these through the G17G-family compute profile; it takes
    * the qualified list below, without fast clear and with the hardware
    * tessellator. target_is_agx3() covers it: its target chip is G17G.
    *
    * Tessellation runs as libagx_tess_isoline/tri/quad (libagx_tessellate is
    * only the tessellator.h macro choosing one), qualified below for every
    * AGX3 target. libagx_decompress is NOT qualified: it fails to compile
    * even for strict g17p (spilling decoder), so it stays AGX2-only. */
   return !target_is_agx3(target) ||
          /* M5 native acceleration-structure copy, compaction,
           * serialization and property kernels (copy.cl, C87 GPU-side AS
           * operations). Native AS images exist only on G17S. */
          (!strcmp(target, "g17g") &&
           !strncmp(func->name, "libagx_agx3_as_", strlen("libagx_agx3_as_"))) ||
          !strcmp(func->name, "libagx_fill_uint4") ||
          !strcmp(func->name, "libagx_gen1_as_property") ||
          !strcmp(func->name, "libagx_gen1_as_copy") ||
          /* Native G17P AS copies relocate child slots and then restore the
           * instance-record slots which can resemble node markers. */
          (!strcmp(target, "g17p") &&
           (!strcmp(func->name, "libagx_gen1_as_rebase") ||
            !strcmp(func->name, "libagx_gen1_as_restore_records"))) ||
          !strcmp(func->name, "libagx_gen1_as_serialize") ||
          !strcmp(func->name, "libagx_gen1_as_deserialize") ||
          !strcmp(func->name, "libagx_gen1_as_deserialize_references") ||
          !strcmp(func->name, "libagx_gen1_as_empty") ||
          (strcmp(target, "g18g") &&
           !strcmp(func->name, "libagx_fast_clear")) ||
          (!strcmp(target, "g17p") &&
           !strcmp(func->name, "libagx_reset_compression_metadata")) ||
          !strcmp(func->name, "libagx_copy_image") ||
          !strcmp(func->name, "libagx_mipmap_unorm8x4") ||
          !strcmp(func->name, "libagx_mipmap_unorm8x4_2level") ||
          !strcmp(func->name, "libagx_mipmap_unorm8x4_odd") ||
          !strcmp(func->name, "libagx_resolve_unorm8x4") ||
          !strcmp(func->name, "libagx_predicate_indirect") ||
          /* VK_KHR_ray_tracing_pipeline's indirect trace-rays setup.  One
           * invocation, three constant loads, three global stores and a shift
           * -- no atomics, no barriers, no printf, no scratch.  Its ARITHMETIC
           * has a host oracle (test-hk-ray-tracing-core's launch
           * decomposition, which the direct path uses for the same numbers);
           * its MACHINE CODE has never executed, exactly like the rest of the
           * ray-tracing-pipeline path it serves, which is why that whole path
           * is behind HK_G17P_UNVERIFIED_FEATURES.  Do not read this entry as
           * a claim that it runs. */
          !strcmp(func->name, "libagx_trace_rays_indirect_setup") ||
          /* Its per-raygen-kernel sibling: the same one-invocation shape
           * (constant loads of the record, the raygen handle and the launch
           * dimensions; one VkDispatchIndirectCommand store per kernel; the
           * same shift/compare arithmetic, with hk_rt_launch_group_count as
           * the host oracle).  No atomics, barriers, printf or scratch.
           * Qualifying its compilation here is not an execution claim; the
           * hardware test of the per-raygen dispatch covers it. */
          !strcmp(func->name, "libagx_trace_rays_select_setup") ||
          /* Its single-kick sibling (opt-in): same reads, plus 11 global
           * stores of a launch template into a stream slot. */
          !strcmp(func->name, "libagx_trace_rays_select_launch") ||
          !strcmp(func->name, "libagx_draw_indirect_byte_count") ||
          !strcmp(func->name, "libagx_conditional_render") ||
          /* M5 task draws with a GPU-side grid, count or predicate: one
           * invocation of constant loads, compares and four global stores
           * into the mesh dispatch record and the draw's execute word. */
          (!strcmp(target, "g17g") &&
           !strcmp(func->name, "libagx_m5_task_dispatch")) ||
          /* G17P software task draws: one invocation of constant loads, a
           * heap allocation and global stores into the task record. */
          (!strcmp(target, "g17p") &&
           !strcmp(func->name, "libagx_task_setup")) ||
          !strcmp(func->name, "libagx_draw_robust_index") ||
          !strcmp(func->name, "libagx_draw_robust_index_batch") ||
          !strcmp(func->name, "libagx_unroll_restart") ||
          !strcmp(func->name, "libagx_gs_setup_indirect") ||
          !strcmp(func->name, "libagx_prefix_sum_geom") ||
          /* Query batches use the same native pointer loads/global stores as
           * the existing copy-query helpers; qualify their batch launch too. */
          !strcmp(func->name, "libagx_write_u32") ||
          !strcmp(func->name, "libagx_write_u32s") ||
          !strcmp(func->name, "libagx_reset_query") ||
          !strcmp(func->name, "libagx_copy_uchar") ||
          !strcmp(func->name, "libagx_copy_uint4") ||
          !strcmp(func->name, "libagx_fill") ||
          !strcmp(func->name, "libagx_copy_xfb_counters") ||
          !strcmp(func->name, "libagx_increment_ia") ||
          !strcmp(func->name, "libagx_increment_ia_restart") ||
          !strcmp(func->name, "libagx_increment_ia_restart_nonnull") ||
          !strcmp(func->name, "libagx_copy_query") ||
          !strcmp(func->name, "libagx_copy_query_gl") ||
          !strcmp(func->name, "libagx_increment_cs_invocations") ||
          !strcmp(func->name, "libagx_setup_indirect_dispatch") ||
          !strcmp(func->name, "libagx_copy_timestamp") ||
          !strcmp(func->name, "libagx_increment_statistic") ||
          /* Software tessellation */
          !strcmp(func->name, "libagx_tess_setup_indirect") ||
          !strcmp(func->name, "libagx_tess_isoline") ||
          !strcmp(func->name, "libagx_tess_tri") ||
          !strcmp(func->name, "libagx_tess_quad") ||
          !strcmp(func->name, "libagx_prefix_sum_tess") ||
          !strcmp(func->name, "libagx_draw_without_adj") ||
          /* Materialises an adjacency decomposition as an index buffer, which
           * is the only remap a G17P hardware vertex shader observes: it reads
           * its invocation index from SR 0xdd, never the prolog's exported
           * AGX_ABI_VIN_VERTEX_ID.  One invocation per output index, one call
           * to poly_map_adj_vertex(), at most one index-buffer load and one
           * global store -- no atomics, barriers, printf or scratch, and the
           * same poly_load_index() the already-qualified
           * libagx_draw_without_adj uses.  Its ARITHMETIC has a host oracle:
           * test-poly-adj-map checks poly_map_adj_vertex() against the
           * per-topology sequences for all four adjacency primitives. */
          !strcmp(func->name, "libagx_write_adj_indices") ||
          /* Its indirect form.  Same mapping and the same index load, plus a
           * heap allocation and one barrier -- both already qualified here by
           * libagx_unroll_restart, which allocates from the same heap with the
           * same printf-free poly_heap_try_alloc_offs(). */
          !strcmp(func->name, "libagx_write_adj_indices_indirect");
}

/* AGX3 has no grounded INTL instruction encoding. Expand the 16-bit Morton
 * interleave used by libagx_fast_clear into ordinary integer operations
 * before instruction selection, without changing AGX2's native path. */
static nir_def *
g17p_part1by1(nir_builder *b, nir_def *src)
{
   nir_def *v = nir_u2u32(b, src);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 8)), 0x00ff00ff);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 4)), 0x0f0f0f0f);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 2)), 0x33333333);
   return nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 1)), 0x55555555);
}

static bool
lower_g17p_interleave(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if (alu->op != nir_op_interleave)
      return false;

   assert(alu->def.bit_size == 32);
   assert(nir_src_bit_size(alu->src[0].src) == 16);
   assert(nir_src_bit_size(alu->src[1].src) == 16);
   b->cursor = nir_before_instr(&alu->instr);
   nir_def *x = g17p_part1by1(b, nir_ssa_for_alu_src(b, alu, 0));
   nir_def *y = g17p_part1by1(b, nir_ssa_for_alu_src(b, alu, 1));
   nir_def_replace(&alu->def, nir_ior(b, x, nir_ishl_imm(b, y, 1)));
   return true;
}

/* Like nir_precomp_print_binary_map, but programs the target does not support
 * are left NULL, as they have no binary. So are the variants in `omitted`,
 * named as their binaries would have been.
 */
static void
print_supported_binary_map(FILE *fp, const nir_shader *nir, const char *target,
                           const struct set *omitted)
{
   fprintf(fp, "const uint32_t *libagx_%s[LIBAGX_NUM_PROGRAMS] = {\n", target);

   nir_foreach_entrypoint(func, nir) {
      if (!target_supports_program(target, func))
         continue;

      for (unsigned v = 0; v < nir_precomp_nr_variants(func); ++v) {
         char name[256];
         snprintf(name, sizeof(name), "%s_%u_%s", func->name, v,
                  remap_variant(func, v, target));
         if (_mesa_set_search(omitted, name))
            continue;

         fprintf(fp, "    [");
         nir_precomp_print_enum_variant_value(fp, func, v);
         fprintf(fp, "] = %s_%u_%s,\n", func->name, v,
                 remap_variant(func, v, target));
      }
   }

   fprintf(fp, "};\n\n");
}

/* Lift every kernel-argument preamble load above the halves the AGX3
 * hardware ABI owns -- see AGX_ABI_AGX3_PRECOMP_ARG_HALFS.  Run per target,
 * on that target's clone, so AGX2 keeps its arguments at half 0. */
static bool
rebase_preamble_for_agx3(nir_builder *b, nir_intrinsic_instr *intr,
                         UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_preamble)
      return false;

   nir_intrinsic_set_base(intr,
                          nir_intrinsic_base(intr) +
                             AGX_ABI_AGX3_PRECOMP_ARG_HALFS);
   return true;
}

/* This helper consumes native heap indices, not descriptor byte offsets. */
static bool
lower_m5_decompress_load(nir_builder *b, nir_intrinsic_instr *intr,
                          UNUSED void *data)
{
   b->cursor = nir_before_instr(&intr->instr);
   if (intr->intrinsic == nir_intrinsic_bindless_image_agx) {
      nir_src_rewrite(&intr->src[0], nir_u2u64(b, intr->src[0].ssa));
      return true;
   }
   return false;
}

static nir_def *
load_kernel_input(nir_builder *b, unsigned num_components, unsigned bit_size,
                  unsigned offset_B)
{
   assert((offset_B & 1) == 0 && "half-aligned");
   return nir_load_preamble(b, num_components, bit_size, .base = offset_B / 2);
}

int
main(int argc, char **argv)
{
   /* Without this the compiler's own AGX_MESA_DEBUG flags stay clear here, so
    * a kernel that fails to compile reports only its name and no reason.
    * Declared locally: agx_compiler.h needs generated headers this tool does
    * not include. */
   {
      extern int agx_compiler_debug;
      extern uint64_t agx_get_compiler_debug(void);
      agx_compiler_debug = (int)agx_get_compiler_debug();
   }
   if (argc != 4) {
      fprintf(stderr, "Usage: %s [input spir-v] [output header] [output C]\n",
              argv[0]);
      return 1;
   }

   const char *infile = argv[1];
   const char *outh_file = argv[2];
   const char *outc_file = argv[3];

   void *mem_ctx = ralloc_context(NULL);

   int fd = open(infile, O_RDONLY);
   if (fd < 0) {
      fprintf(stderr, "Failed to open %s\n", infile);
      ralloc_free(mem_ctx);
      return 1;
   }

   off_t spirv_len = lseek(fd, 0, SEEK_END);
   const void *spirv_map = mmap(NULL, spirv_len, PROT_READ, MAP_PRIVATE, fd, 0);
   close(fd);
   if (spirv_map == MAP_FAILED) {
      fprintf(stderr, "Failed to mmap the file: errno=%d, %s\n", errno,
              strerror(errno));
      ralloc_free(mem_ctx);
      return 1;
   }

   FILE *fp_h = fopen(outh_file, "w");
   FILE *fp_c = fopen(outc_file, "w");
   glsl_type_singleton_init_or_ref();

   nir_precomp_print_header(fp_c, fp_h, "The Asahi Linux Contributors",
                            "libagx_shaders.h");

   nir_shader *nir = compile(mem_ctx, spirv_map, spirv_len);

   /* load_preamble works at 16-bit granularity */
   struct nir_precomp_opts opt = {.arg_align_B = 2};

   /* Variants target_omits_refused_program() let a target go without. */
   struct set *omitted =
      _mesa_set_create(mem_ctx, _mesa_hash_string, _mesa_key_string_equal);
   unsigned nr_compiled = 0;

   nir_foreach_entrypoint(libfunc, nir) {
      libfunc->pass_flags = 0;
      struct nir_precomp_layout layout =
         nir_precomp_derive_layout(&opt, libfunc);
      unsigned nr_vars = nir_precomp_nr_variants(libfunc);

      nir_precomp_print_layout_struct(fp_h, &opt, libfunc);

      for (unsigned v = 0; v < nr_vars; ++v) {
         nir_shader *s = nir_precompiled_build_variant(
            libfunc, MESA_SHADER_COMPUTE, v, &agx_nir_options, &opt,
            load_kernel_input);

         nir_link_shader_functions(s, nir);
         NIR_PASS(_, s, nir_inline_functions);
         nir_remove_non_entrypoints(s);
         NIR_PASS(_, s, nir_opt_deref);
         NIR_PASS(_, s, nir_lower_vars_to_ssa);
         NIR_PASS(_, s, nir_remove_dead_derefs);
         NIR_PASS(_, s, nir_remove_dead_variables,
                  nir_var_function_temp | nir_var_shader_temp, NULL);
         NIR_PASS(_, s, nir_lower_vars_to_explicit_types,
                  nir_var_shader_temp | nir_var_function_temp,
                  glsl_get_cl_type_size_align);

         NIR_PASS(_, s, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
                  glsl_get_cl_type_size_align);

         NIR_PASS(_, s, nir_lower_explicit_io, nir_var_mem_shared,
                  nir_address_format_62bit_generic);

         agx_preprocess_nir(s);

         NIR_PASS(_, s, nir_opt_deref);
         NIR_PASS(_, s, nir_lower_vars_to_ssa);
         NIR_PASS(_, s, nir_lower_explicit_io,
                  nir_var_shader_temp | nir_var_function_temp |
                     nir_var_mem_shared | nir_var_mem_global,
                  nir_address_format_62bit_generic);

         bool has_atomic = false;
         nir_shader_intrinsics_pass(s, gather_atomic_info, nir_metadata_all,
                                    &has_atomic);
         if (has_atomic) {
            libfunc->pass_flags |= BITFIELD_BIT(v);
         }

         foreach_target(target)
         {
            if (!target_supports_program(*target, libfunc))
               continue;

            /* Skip unused variants */
            if (strcmp(*target, remap_variant(libfunc, v, *target)))
               continue;

            struct agx_shader_part compiled;
            bool is_helper = !strcmp(libfunc->name, "libagx_helper");
            struct agx_shader_key key = {
               .promote_constants = !is_helper,

               /* Most of the internal programs don't use textures at all, but
                * the few that do are exclusively bindless, so we want to
                * promote their access.
                */
               .promote_textures = true,

               .reserved_preamble = layout.size_B / 2,
               .is_helper = is_helper,
               .no_preamble = target_is_agx3(*target),

               /* AGX3 has a smaller register file than AGX2, so the largest
                * kernels spill to thread-private storage like API shaders.
                * G18G's profile cannot launch a spilling kernel, but letting
                * the allocator spill turns "does not fit" into a scratch size
                * checked below rather than an unspillable-shader assert.
                */
               .has_scratch = target_is_agx3(*target),
            };

            key.dev.encoding = !strcmp(*target, "g17g")
                                  ? AGX_ISA_ENCODING_AGX3C
                                  : target_is_native_agx3(*target)
                                  ? AGX_ISA_ENCODING_AGX3B
                                  : AGX_ISA_ENCODING_AGX2;
            key.dev.chip = target_chip(*target);

            if (!strcmp(*target, "g18g")) {
               /* The device key agx_gather_device_key() builds for G18G. */
               key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
               key.dev.chip = AGX_CHIP_G17G;
               key.dev.emit_agx3d = true;
            }

            if (has_atomic) {
               key.dev.needs_g13x_coherency =
                  u_tristate_make(!strcmp(*target, "g13x"));
            }

            nir_shader *clone = nir_shader_clone(NULL, s);

            if (key.dev.encoding != AGX_ISA_ENCODING_AGX2) {
               NIR_PASS(_, clone, nir_shader_intrinsics_pass,
                        rebase_preamble_for_agx3, nir_metadata_all, NULL);
               NIR_PASS(_, clone, nir_shader_alu_pass,
                        lower_g17p_interleave, nir_metadata_control_flow,
                        NULL);
               key.reserved_preamble += AGX_ABI_AGX3_PRECOMP_ARG_HALFS;
            }

            if (!strcmp(libfunc->name, "libagx_m5_decompress_rgba16f") ||
                !strcmp(libfunc->name, "libagx_g17p_decompress_16b")) {
               key.g17p_texture_uniform_base = AGX_ABI_AGX3_PRECOMP_ARG_HALFS;
               NIR_PASS(_, clone, nir_shader_intrinsics_pass,
                        lower_m5_decompress_load, nir_metadata_none, NULL);
            }

            uint16_t printf_uniform = 0;
            if (target_is_agx3(*target)) {
               /* Both frontends publish the allocator-owned printf buffer.
                * Keep that runtime ABI instead of dropping kernel assertions
                * as upstream did before this branch supplied the address. */
               bool uses_printf = false;
               nir_shader_intrinsics_pass(clone, uses_printf_address,
                                          nir_metadata_all, &uses_printf);

               if (uses_printf) {
                  /* A 64-bit uniform address must start on a whole dword
                   * of halves.
                   */
                  unsigned base_halfs = AGX_ABI_AGX3_PRECOMP_ARG_HALFS +
                     ALIGN_POT(layout.size_B, 8) / 2;
                  printf_uniform = base_halfs;
                  key.reserved_preamble = base_halfs + 4;

                  NIR_PASS(_, clone, nir_shader_intrinsics_pass,
                           lower_printf_address_to_uniform,
                           nir_metadata_control_flow, &base_halfs);
               }
            }
            extern const char *agx_last_compile_refusal(void);
            bool compiled_ok = agx_compile_shader_nir(clone, &key, &compiled);
            const char *refusal =
               compiled_ok ? NULL : agx_last_compile_refusal();
            if (compiled_ok && !strcmp(*target, "g18g") &&
                compiled.info.scratch_size) {
               refusal = ralloc_asprintf(mem_ctx, "needs %u B of scratch",
                                         compiled.info.scratch_size);
               free(compiled.binary);
               compiled_ok = false;
            }
            if (!compiled_ok) {
               if (!target_omits_refused_program(*target, libfunc->name)) {
                  fprintf(stderr, "Failed to compile %s for %s: %s\n",
                          libfunc->name, *target, refusal);
                  ralloc_free(clone);
                  return 1;
               }
               fprintf(stderr, "asahi_clc: %s variant %u left NULL for %s: %s\n",
                       libfunc->name, v, *target,
                       refusal ? refusal : "refused");
               _mesa_set_add(omitted,
                             ralloc_asprintf(mem_ctx, "%s_%u_%s",
                                             libfunc->name, v, *target));
               ralloc_free(clone);
               continue;
            }
            if (!strcmp(*target, "g18g"))
               nr_compiled++;
            print_shader(fp_c, libfunc->name, *target, v, &compiled,
                         printf_uniform);
            free(compiled.binary);
            ralloc_free(clone);

            assert((compiled.info.scratch_size == 0 ||
                    target_is_agx3(*target)) &&
                   "only AGX3 internal shaders spill");

            assert(compiled.info.preamble_scratch_size == 0 &&
                   "internal shader preambles do not spill");
         }

         ralloc_free(s);
      }
   }

   nir_precomp_print_program_enum(fp_h, nir, "libagx");
   nir_precomp_print_dispatch_macros(fp_h, &opt, nir);

   /* For each target, generate a table mapping programs to binaries */
   foreach_target(target)
   {
      nir_precomp_print_extern_binary_map(fp_h, "libagx", *target);
      print_supported_binary_map(fp_c, nir, *target, omitted);
   }

   /* Say which qualified kernels a lenient target went without, so a newly
    * grounded encoding shows up as a shorter list. */
   if (omitted->entries) {
      fprintf(stderr, "asahi_clc: %u of %u qualified kernel variants are "
              "left NULL for G18G (no encoding or launch shape yet):",
              omitted->entries, omitted->entries + nr_compiled);
      set_foreach(omitted, entry)
         fprintf(stderr, " %s", (const char *)entry->key);
      fprintf(stderr, "\n");
   }

   glsl_type_singleton_decref();
   fclose(fp_c);
   fclose(fp_h);
   ralloc_free(mem_ctx);
   return 0;
}

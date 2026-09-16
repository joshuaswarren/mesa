/*
 * Offline Honeykrisp shader-compile diff harness.
 *
 * Compiles a Vulkan compute SPIR-V module through the same NIR pipeline the
 * Honeykrisp driver uses (minus descriptor-set plumbing, which is replaced
 * with constant global addresses), emits the AGX binary, and dumps NIR +
 * disassembly for bit-level comparison across compilation units.
 *
 * Not part of Mesa's build. Compile against the native build tree:
 *   clang trig_dump.c -o trig_dump -I<src> -I<build>/src -I<build>/src/compiler/nir \
 *     -I<src>/src/compiler -I<src>/src/asahi -I<src>/src/asahi/compiler \
 *     -I<src>/src/gallium/include -I<src>/src/util \
 *     <libs...>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nir.h"
#include "nir_builder.h"
#include "nir_spirv.h"
#include "compiler/shader_enums.h"
#include "agx_compile.h"
#include "agx_nir.h"
#include "disasm.h"

/* Fake device key for an M1 (G13). soft_fault selectable by env. */
static struct agx_device_key
harness_device_key(void)
{
   const char *sf = getenv("TRIG_SOFT_FAULT");
   return (struct agx_device_key){
      .needs_g13x_coherency = false,
      .soft_fault = sf && sf[0] == '1',
   };
}

/* HK lowers push constants to global loads through the root pointer; we
 * substitute a constant root. SSBOs become global loads at per-binding
 * constant addresses. The fp chain sees identical ALU.
 */
static int mark_n;
#define MARK() do { if (getenv("TRIG_MARKS")) fprintf(stderr, "ok %d\n", mark_n++); } while(0)
/* Exported from nir_deref.c; no public header declares it. */
nir_def *nir_build_deref_offset(nir_builder *b, nir_deref_instr *deref,
                                glsl_type_size_align_func size_align);

#define FAKE_ROOT  0x100000000ull
#define FAKE_SSBO(b) (0x20000000ull + 0x100000ull * (b))

/* Push constants surface as entry-point params; SSBO descriptors surface as
 * vulkan_resource_index/load_vulkan_descriptor after explicit-io lowering.
 * Both become constant-address global loads, exactly the shape hk produces
 * from its root/descriptor tables.
 */
static bool
fake_lower_intrinsics(nir_builder *b, nir_intrinsic_instr *intr, void *_)
{
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *val = NULL;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_param: {
      unsigned idx = nir_intrinsic_param_idx(intr);
      nir_def *addr = nir_imm_int64(b, FAKE_ROOT + 0x1000ull + 0x100ull * idx);
      val = nir_load_global(b, intr->def.num_components, intr->def.bit_size,
                            addr, 16);
      break;
   }
   case nir_intrinsic_load_push_constant: {
      unsigned base = nir_intrinsic_base(intr);
      nir_def *addr =
         nir_iadd_imm(b, nir_u2u64(b, intr->src[0].ssa), FAKE_ROOT + base);
      val = nir_load_global(b, intr->def.num_components, intr->def.bit_size,
                            addr, 16);
      break;
   }
   case nir_intrinsic_load_num_workgroups: {
      nir_def *addr = nir_imm_int64(b, FAKE_ROOT + 0x800ull);
      val = nir_load_global(b, 3, 32, addr, 4);
      break;
   }
   case nir_intrinsic_load_base_workgroup_id:
      val = nir_imm_zero(b, intr->def.num_components, intr->def.bit_size);
      break;
   case nir_intrinsic_load_vulkan_descriptor: {
      nir_intrinsic_instr *idx = nir_src_as_intrinsic(intr->src[0]);
      unsigned set = idx ? nir_intrinsic_desc_set(idx) : 0;
      unsigned binding = idx ? nir_intrinsic_binding(idx) : 0;
      val = nir_imm_int64(b, FAKE_SSBO((unsigned long long)set * 64ull +
                                       (unsigned long long)binding));
      break;
   }
   default:
      return false;
   }

   nir_def_rewrite_uses(&intr->def, val);
   nir_instr_remove(&intr->instr);
   nir_progress(true, b->impl, nir_metadata_none);
   return true;
}

/* Deref-based SSBO access: root at a nir_var_mem_ssbo variable. Replace the
 * whole chain with a constant per-binding address + the deref-computed offset.
 */
static bool
fake_lower_ssbo_deref(nir_builder *b, nir_intrinsic_instr *intr, void *_)
{
   if (intr->intrinsic != nir_intrinsic_load_deref &&
       intr->intrinsic != nir_intrinsic_store_deref)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   if (!deref)
      return false;
   nir_variable *var = nir_deref_instr_get_variable(deref);
   nir_def *base = NULL;
   if (var && (var->data.mode & (nir_var_mem_ssbo | nir_var_mem_push_const))) {
      b->cursor = nir_before_instr(&intr->instr);
      base = (var->data.mode & nir_var_mem_push_const)
                ? nir_imm_int64(b, FAKE_ROOT + 0x1000ull)
                : nir_imm_int64(b, FAKE_SSBO(var->data.binding));
   } else if (!var) {
      /* vtn Vulkan buffer chain: deref_cast over a vulkan_resource_index */
      nir_deref_instr *d = deref;
      nir_intrinsic_instr *idx = NULL;
      while (d != NULL && d->deref_type != nir_deref_type_var &&
             d->deref_type != nir_deref_type_cast) {
         nir_deref_instr *pd = nir_src_as_deref(d->parent);
         if (pd == NULL)
            break;
         d = pd;
      }
      /* The root pointer may be the resource index itself or my constant
       * load_global that already replaced a load_vulkan_descriptor. */
      if (d != NULL && d->deref_type == nir_deref_type_cast) {
         nir_load_const_instr *cl = nir_src_as_load_const(d->parent);
         if (cl != NULL) {
            /* vtn folded the block address to a plain constant */
            b->cursor = nir_before_instr(&intr->instr);
            base = nir_imm_int64(b, cl->value[0].u64);
         }
         nir_intrinsic_instr *r = nir_src_as_intrinsic(d->parent);
         if (base != NULL) {
            /* already resolved */
         } else if (r != NULL && r->intrinsic == nir_intrinsic_vulkan_resource_index) {
            b->cursor = nir_before_instr(&intr->instr);
            unsigned set = nir_intrinsic_desc_set(r);
            unsigned binding = nir_intrinsic_binding(r);
            base = nir_imm_int64(b, FAKE_SSBO((unsigned long long)set * 64ull +
                                              (unsigned long long)binding));
         } else if (r != NULL && r->intrinsic == nir_intrinsic_load_global) {
            /* chase the address constant */
            nir_load_const_instr *lc = nir_src_as_load_const(r->src[0]);
            if (lc != NULL) {
               uint64_t v = lc->value[0].u64;
               b->cursor = nir_before_instr(&intr->instr);
               base = nir_imm_int64(b, v);
            }
         }
      }
      if (base == NULL)
         return false;
   } else {
      return false;
   }
   nir_def *addr = base;
   if (var) {
      addr = nir_explicit_io_address_from_deref(
         b, deref, base, nir_address_format_64bit_global);
   } else {
      nir_def *off = nir_build_deref_offset(
         b, deref, glsl_get_natural_size_align_bytes);
      addr = nir_iadd(b, base, nir_u2u64(b, off));
   }

   if (intr->intrinsic == nir_intrinsic_load_deref) {
      nir_def *val = nir_load_global(b, intr->def.num_components,
                                     intr->def.bit_size, addr, 16);
      nir_def_rewrite_uses(&intr->def, val);
      nir_instr_remove(&intr->instr);
   } else {
      nir_build_store_global(b, intr->src[1].ssa, addr, .align_mul = 16);
      nir_instr_remove(&intr->instr);
   }
   nir_progress(true, b->impl, nir_metadata_none);
   return true;
}

static void
dump_nir(nir_shader *nir, const char *tag)
{
   const char *dir = getenv("TRIG_DUMP_DIR");
   if (!dir)
      return;
   char path[512];
   snprintf(path, sizeof(path), "%s/%s.nir", dir, tag);
   FILE *fp = fopen(path, "w");
   if (fp) {
      nir_print_shader(nir, fp);
      fclose(fp);
   }
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s <module.spv>\n", argv[0]);
      return 2;
   }

   glsl_type_singleton_init_or_ref();

   FILE *f = fopen(argv[1], "rb");
   if (!f) {
      perror("open spv");
      return 1;
   }
   fseek(f, 0, SEEK_END);
   long len = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *words = malloc(len);
   if (fread(words, 1, len, f) != (size_t)len) {
      perror("read");
      return 1;
   }
   fclose(f);
   size_t word_count = len / 4;

   struct spirv_to_nir_options spirv_opts = {
      .ssbo_addr_format = nir_address_format_64bit_global,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .ubo_addr_format = nir_address_format_64bit_global,
      .shared_addr_format = nir_address_format_32bit_offset,
      .min_ssbo_alignment = 16,
      .min_ubo_alignment = 16,
   };

   nir_shader *nir = spirv_to_nir(words, word_count, NULL,
                                  MESA_SHADER_COMPUTE, "main", &spirv_opts,
                                  &agx_nir_options);
   if (!nir) {
      fprintf(stderr, "spirv_to_nir failed\n");
      return 1;
   }

      NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_inline_functions);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_deref);
   nir_remove_non_entrypoints(nir);

/* ---- hk_preprocess_nir_internal (compute subset) ---- */
   MARK(); NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   bool progress;
   do {
      progress = false;
      MARK(); NIR_PASS(progress, nir, nir_lower_vars_to_ssa);
      MARK(); NIR_PASS(progress, nir, nir_opt_copy_prop);
      MARK(); NIR_PASS(progress, nir, nir_opt_dce);
      MARK(); NIR_PASS(progress, nir, nir_opt_constant_folding);
      MARK(); NIR_PASS(progress, nir, nir_opt_loop);
      MARK(); NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);
   MARK(); NIR_PASS(_, nir, nir_lower_system_values);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   MARK(); NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries,
            nir_shader_get_entrypoint(nir), nir_var_shader_out);
   MARK(); NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   MARK(); NIR_PASS(_, nir, nir_split_var_copies);
   MARK(); NIR_PASS(_, nir, nir_split_struct_vars, nir_var_function_temp);
   agx_preprocess_nir(nir);
   MARK(); NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   MARK(); NIR_PASS(_, nir, nir_lower_var_copies);

   nir_lower_compute_system_values_options csv_options = {
      .has_base_workgroup_id = true,
   };
   MARK(); NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);

   /* ---- hk_lower_nir (compute subset, descriptor-free) ---- */
   const nir_opt_access_options access_options = {.is_vulkan = true};
   MARK(); NIR_PASS(_, nir, nir_opt_access, &access_options);

   /* Push-constant block variables -> load_push_constant intrinsics */
   MARK(); NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);

   MARK(); NIR_PASS(_, nir, nir_shader_intrinsics_pass, fake_lower_intrinsics,
            nir_metadata_none, NULL);

   MARK(); NIR_PASS(_, nir, nir_shader_intrinsics_pass, fake_lower_ssbo_deref,
            nir_metadata_none, NULL);

   /* Shared memory: same lowering shape as hk (explicit io, zero-init) */
   MARK(); NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
            glsl_get_natural_size_align_bytes);
   MARK(); NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
            nir_address_format_32bit_offset);

   if (nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
      nir->info.shared_size = ALIGN_POT(nir->info.shared_size, 16);
      MARK(); NIR_PASS(_, nir, nir_zero_initialize_shared_memory,
               nir->info.shared_size, 16);
      MARK(); NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
   }

   dump_nir(nir, "after_fakelower");

   bool p2;
   do {
      p2 = false;
      MARK(); NIR_PASS(p2, nir, nir_opt_constant_folding);
      MARK(); NIR_PASS(p2, nir, nir_opt_algebraic);
      MARK(); NIR_PASS(p2, nir, nir_opt_copy_prop);
      MARK(); NIR_PASS(p2, nir, nir_opt_dce);
   } while (p2);

   /* hk moves ssbo/ubo loads; we moved to global loads */
   nir_move_options move_opts = nir_move_load_ssbo | nir_move_load_ubo;
   MARK(); NIR_PASS(_, nir, nir_opt_sink, move_opts);
   MARK(); NIR_PASS(_, nir, nir_opt_move, move_opts);

   /* hk tail: peephole select + opt_if, then compile. Dump NIR before this
    * final stretch: the debug-build metadata validator refuses to print
    * after passes that ran with nir_metadata_none.
    */
   dump_nir(nir, "before_compile");
   agx_preprocess_nir(nir);
   nir_opt_peephole_select_options peephole_select_options = {
      .limit = 0,
      .discard_ok = true,
   };
   MARK(); NIR_PASS(_, nir, nir_opt_peephole_select, &peephole_select_options);
   MARK(); NIR_PASS(_, nir, nir_opt_if,
            nir_opt_if_optimize_phi_true_false | nir_opt_if_avoid_64bit_phis);

   struct agx_shader_key key = {
      .reserved_preamble = 4,
      .dev = harness_device_key(),
      .has_scratch = true,
      .promote_constants = true,
      .promote_textures = true,
   };

   struct agx_shader_part part;
   memset(&part, 0, sizeof(part));
   agx_compile_shader_nir(nir, &key, &part);

   const char *dir = getenv("TRIG_DUMP_DIR");
   if (dir) {
      char path[512];
      snprintf(path, sizeof(path), "%s/isa.txt", dir);
      FILE *fp = fopen(path, "w");
      if (fp) {
         agx2_disassemble(part.binary, part.info.binary_size, fp);
         fclose(fp);
      }
      snprintf(path, sizeof(path), "%s/stats.txt", dir);
      fp = fopen(path, "w");
      if (fp) {
         fprintf(fp, "binary_size %u has_preamble %d nr_gprs %u "
                     "nr_preamble_gprs %u instrs %u alu %u code_size %u\n",
                 part.info.binary_size, part.info.has_preamble,
                 part.info.nr_gprs, part.info.nr_preamble_gprs,
                 part.info.stats.instrs, part.info.stats.alu,
                 part.info.stats.code_size);
         fclose(fp);
      }
   }

   printf("%s: compiled, %u bytes\n", argv[1], part.info.binary_size);
   return 0;
}

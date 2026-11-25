/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_abi.h"
#include "agx_compile.h"
#include "agx_linker.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>

/* agx_nir_inline_fs_epilog(): an epilog appended to its main reads exactly
 * the values the main exported, and every shape the split link would hand
 * over differently is refused. */

#define COLOUR0     AGX_ABI_FOUT_COLOUR(0)
#define COLOUR_UNIF 32
#define TEST_EXPORT 64

static nir_builder
fs_builder(const char *name)
{
   return nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,
                                         &agx_nir_options, "%s", name);
}

enum main_shape {
   MAIN_PLAIN,
   MAIN_EXPORT_IN_BRANCH,
   MAIN_READS_EXPORT,
   MAIN_OVERLAPPING_EXPORTS,
};

/* A main exporting `comps` 32-bit colour components loaded from uniforms. */
static nir_shader *
build_main(unsigned comps, enum main_shape shape)
{
   nir_builder b = fs_builder("main");
   nir_def *colour = nir_load_preamble(&b, comps, 32, .base = COLOUR_UNIF);

   if (shape == MAIN_READS_EXPORT) {
      nir_def *mask =
         nir_load_exported_agx(&b, 1, 16, .base = AGX_ABI_FIN_SAMPLE_MASK);
      nir_export_agx(&b, mask, .base = AGX_ABI_FOUT_SAMPLE_MASK);
   }

   if (shape == MAIN_EXPORT_IN_BRANCH) {
      nir_def *cond = nir_ieq_imm(&b, nir_load_preamble(&b, 1, 32, .base = 40),
                                  0);
      nir_if *nif = nir_push_if(&b, cond);
      nir_export_agx(&b, colour, .base = COLOUR0);
      nir_pop_if(&b, nif);
   } else {
      nir_export_agx(&b, colour, .base = COLOUR0);
   }

   if (shape == MAIN_OVERLAPPING_EXPORTS)
      nir_export_agx(&b, nir_channel(&b, colour, 0), .base = COLOUR0 + 2);

   return b.shader;
}

/* An epilog reading `comps` components of `bits` at half `base` and handing
 * them on in an export of its own, which the tests read back. */
static nir_shader *
build_epilog(unsigned base, unsigned comps, unsigned bits)
{
   nir_builder b = fs_builder("epilog");
   nir_def *value = nir_load_exported_agx(&b, comps, bits, .base = base);
   nir_export_agx(&b, value, .base = TEST_EXPORT);
   b.shader->info.fs.uses_fbfetch_output = true;
   return b.shader;
}

static unsigned
count_intrinsics(nir_shader *s, nir_intrinsic_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            n += instr->type == nir_instr_type_intrinsic &&
                 nir_instr_as_intrinsic(instr)->intrinsic == op;
         }
      }
   }
   return n;
}

static nir_intrinsic_instr *
find_export(nir_shader *s, unsigned base)
{
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_export_agx &&
                nir_intrinsic_base(intr) == base)
               return intr;
         }
      }
   }
   return NULL;
}

/* Whether component `c` of the test export is component `c` of the main's
 * colour uniform, or undefined. */
static bool
reads_main_colour(nir_intrinsic_instr *x, unsigned c, bool undefined)
{
   nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(x->src[0].ssa, c));
   nir_instr *parent = nir_def_instr(s.def);

   if (undefined)
      return parent->type == nir_instr_type_undef;

   if (parent->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *load = nir_instr_as_intrinsic(parent);
   return load->intrinsic == nir_intrinsic_load_preamble &&
          nir_intrinsic_base(load) == COLOUR_UNIF && s.comp == c;
}

static bool
inline_and_check(const char *name, unsigned main_comps, enum main_shape shape,
                 unsigned base, unsigned comps, unsigned bits,
                 bool expect_inlined)
{
   nir_shader *main_fs = build_main(main_comps, shape);
   nir_shader *epilog = build_epilog(base, comps, bits);
   bool ok = agx_nir_inline_fs_epilog(main_fs, epilog) == expect_inlined;

   if (ok && expect_inlined) {
      nir_validate_shader(main_fs, name);

      nir_intrinsic_instr *x = find_export(main_fs, TEST_EXPORT);
      ok = x && x->src[0].ssa->num_components == comps &&
           count_intrinsics(main_fs, nir_intrinsic_load_exported_agx) == 0 &&
           count_intrinsics(main_fs, nir_intrinsic_export_agx) == 1 &&
           main_fs->info.fs.uses_fbfetch_output;

      for (unsigned c = 0; ok && c < comps; ++c)
         ok = reads_main_colour(x, c, c >= main_comps);
   }

   if (!ok)
      fprintf(stderr, "%s failed\n", name);

   ralloc_free(main_fs);
   ralloc_free(epilog);
   return ok;
}

int
main(void)
{
   bool ok = true;

   /* The colours the main exported reach the epilog's reads unchanged. */
   ok &= inline_and_check("colour hand-over", 4, MAIN_PLAIN, COLOUR0, 4, 32,
                          true);

   /* A colour component the main never wrote is undefined, as in the API. */
   ok &= inline_and_check("unwritten colour component", 3, MAIN_PLAIN, COLOUR0,
                          4, 32, true);

   /* A register the split epilog would read from the hardware or the prolog
    * has no value to hand over. */
   ok &= inline_and_check("unexported sample mask", 4, MAIN_PLAIN,
                          AGX_ABI_FOUT_SAMPLE_MASK, 1, 16, false);

   /* A narrower read of an exported register is a different value. */
   ok &= inline_and_check("width mismatch", 4, MAIN_PLAIN, COLOUR0, 4, 16,
                          false);

   /* A main that reads exported registers must start its shader part. */
   ok &= inline_and_check("main reads exports", 4, MAIN_READS_EXPORT, COLOUR0,
                          4, 32, false);

   /* Exports must be in the main's last block and must not overlap. */
   ok &= inline_and_check("export in a branch", 4, MAIN_EXPORT_IN_BRANCH,
                          COLOUR0, 4, 32, false);
   ok &= inline_and_check("overlapping exports", 4, MAIN_OVERLAPPING_EXPORTS,
                          COLOUR0, 4, 32, false);

   return ok ? 0 : 1;
}

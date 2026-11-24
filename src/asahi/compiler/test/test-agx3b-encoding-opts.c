/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* NIR lowerings that depend on the instruction set follow the AGX3B encoding,
 * not the G17P part: M3 (G15) and M4 (G16G) compile a 64-bit division with the
 * normalized loop (agx_nir_lower_divmod64) and relax fp32 min/max flushes the
 * way the A18 Pro does, while AGX2 keeps the unrolled division. */

#include "agx_compile.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>

static unsigned failures;

#define CHECK(cond, ...)                                                       \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL line %d: ", __LINE__);                          \
         fprintf(stderr, __VA_ARGS__);                                         \
         fprintf(stderr, "\n");                                                \
         failures++;                                                           \
      }                                                                        \
   } while (0)

static nir_builder
compute_shader(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

/* Instructions of the compiled main program, 0 if it did not compile. */
static unsigned
compile(nir_shader *s, enum agx_isa_encoding encoding, enum agx_chip chip)
{
   agx_preprocess_nir(s);
   struct agx_shader_key key = {
      .dev.encoding = encoding,
      .dev.chip = chip,
      .reserved_preamble = 8,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   unsigned instrs = 0;
   if (agx_compile_shader_nir(s, &key, &part) && part.binary)
      instrs = part.info.stats.instrs;
   free(part.binary);
   ralloc_free(s);
   return instrs;
}

static unsigned
udiv64(enum agx_isa_encoding encoding, enum agx_chip chip)
{
   nir_builder b = compute_shader("64-bit division");
   nir_def *input = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *output = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *x = nir_load_global(&b, 1, 64, input, .align_mul = 8);
   nir_def *y =
      nir_load_global(&b, 1, 64, nir_iadd_imm(&b, input, 8), .align_mul = 8);
   nir_store_global(&b, nir_udiv(&b, x, y), output, .align_mul = 8);
   return compile(b.shader, encoding, chip);
}

static unsigned
fmin32(enum agx_chip chip, bool ftz)
{
   nir_builder b = compute_shader("fp32 min of loaded values");
   if (ftz) {
      b.shader->info.float_controls_execution_mode =
         FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32;
   }
   nir_def *input = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *output = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *x = nir_load_global(&b, 1, 32, input, .align_mul = 4);
   nir_def *y =
      nir_load_global(&b, 1, 32, nir_iadd_imm(&b, input, 4), .align_mul = 4);
   nir_store_global(&b, nir_fmin(&b, x, y), output, .align_mul = 4);
   return compile(b.shader, AGX_ISA_ENCODING_AGX3B, chip);
}

int
main(void)
{
   const unsigned g17p = udiv64(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P);
   const unsigned g16g = udiv64(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G);
   const unsigned g15g = udiv64(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G15G);
   const unsigned agx2 = udiv64(AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G);

   CHECK(g17p && g16g && g15g && agx2, "division did not compile: %u %u %u %u",
         g17p, g16g, g15g, agx2);
   /* The same lowering on every AGX3B part; the backend still differs in a
    * few witnessed forms, so allow a little slack, far below the unrolled
    * division's size. */
   CHECK(g16g <= g17p + g17p / 8 && g15g <= g17p + g17p / 8,
         "AGX3B parts lower the division differently: G17P %u, G16G %u, G15G %u",
         g17p, g16g, g15g);
   /* Negative control: AGX2 keeps nir_lower_int64's unrolled division. */
   CHECK(agx2 > 2 * g17p, "AGX2 division unexpectedly small: %u vs G17P %u",
         agx2, g17p);

   /* Without a flush-to-zero execution mode a min/max result may stay
    * denormal; with one, the flush stays. */
   const unsigned relaxed = fmin32(AGX_CHIP_G16G, false);
   const unsigned flushed = fmin32(AGX_CHIP_G16G, true);
   CHECK(relaxed && flushed, "fmin did not compile: %u %u", relaxed, flushed);
   CHECK(flushed > relaxed,
         "G16G fmin flush not relaxed (%u) or not kept under FTZ (%u)", relaxed,
         flushed);
   CHECK(fmin32(AGX_CHIP_G17P, false) == relaxed,
         "G17P and G16G relax the fmin flush differently");

   printf("RESULT G17P=%u G16G=%u G15G=%u AGX2=%u fmin relaxed=%u ftz=%u "
          "failures=%u\n",
          g17p, g16g, g15g, agx2, relaxed, flushed, failures);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>

enum convert {
   /* OpUConvert of a byte to 32 bits, then OpConvertSToF. */
   SIGNED_OF_ZERO_EXTENDED_BYTE,
   /* The unsigned conversion of the byte itself. */
   UNSIGNED_BYTE,
   /* The unsigned conversion of a 16-bit load: one native conversion. */
   UNSIGNED_HALF,
};

/* The size of a compute shader that loads one integer per invocation,
 * converts it to float and stores the float, compiled for `chip`. */
static int
program_size(enum agx_chip chip, enum convert convert)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "byte to float");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *id = nir_u2u64(&b, nir_load_local_invocation_index(&b));
   nir_def *value;
   if (convert == UNSIGNED_HALF) {
      nir_def *x = nir_load_global(&b, 1, 16, nir_iadd(&b, base,
                                                       nir_ishl_imm(&b, id, 1)),
                                   .align_mul = 2);
      value = nir_u2f32(&b, x);
   } else {
      nir_def *x = nir_load_global(&b, 1, 8, nir_iadd(&b, base, id),
                                   .align_mul = 1);
      value = convert == UNSIGNED_BYTE ? nir_u2f32(&b, x)
                                       : nir_i2f32(&b, nir_u2u32(&b, x));
   }
   nir_def *out = nir_iadd(&b, nir_iadd_imm(&b, base, 4096),
                           nir_ishl_imm(&b, id, 2));
   nir_store_global(&b, value, out, .align_mul = 4);
   agx_preprocess_nir_native(b.shader, chip);

   struct agx_shader_key key = {
      .dev.chip = chip,
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .reserved_preamble = 8,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   int size = -1;
   if (agx_compile_shader_nir(b.shader, &key, &part) && part.binary)
      size = (int)part.info.binary_size;
   free(part.binary);
   ralloc_free(b.shader);
   return size;
}

/* A zero-extended byte converts with the native unsigned-half conversion of
 * its half register, as a 16-bit value does: neither the signed conversion of
 * its 32-bit zero extension nor the unsigned byte conversion costs more than
 * that one instruction. */
int
main(void)
{
   static const enum agx_chip chips[] = {AGX_CHIP_G17P, AGX_CHIP_G16G};
   int failures = 0;
   for (unsigned c = 0; c < sizeof(chips) / sizeof(chips[0]); ++c) {
      const int half = program_size(chips[c], UNSIGNED_HALF);
      const int byte = program_size(chips[c], UNSIGNED_BYTE);
      const int widened = program_size(chips[c], SIGNED_OF_ZERO_EXTENDED_BYTE);
      if (half <= 0 || byte != half || widened != half) {
         fprintf(stderr,
                 "chip %u: 16-bit u2f32 %d B, byte u2f32 %d B, "
                 "i2f32(u2u32(byte)) %d B\n",
                 (unsigned)chips[c], half, byte, widened);
         failures++;
      }
   }
   if (failures)
      return 1;

   puts("agx_byte_to_float: a zero-extended byte converts as a half does");
   return 0;
}

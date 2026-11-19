/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "agx3_cf_core.h"
#include "agx_compiler.h"
#include "nir_builder.h"
#include "util/half_float.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool
check_empty_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t expected[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "empty AGX3 compute");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "empty AGX3 compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid = part.binary != NULL &&
                      part.info.binary_size == sizeof(expected) &&
                      part.info.main_offset == 0 &&
                      part.info.main_size == sizeof(expected) &&
                      part.info.stats.instrs == 1 &&
                      part.info.workgroup_size[0] == 1 &&
                      part.info.workgroup_size[1] == 1 &&
                      part.info.workgroup_size[2] == 1 &&
                      memcmp(part.binary, expected, sizeof(expected)) == 0;

   if (!valid)
      fprintf(stderr, "empty AGX3 compute shader metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_shared_store_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 shared store");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 4;
   nir_store_shared(&b, nir_imm_int(&b, 0x12345678), nir_imm_int(&b, 0),
                    .align_mul = 4);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 shared-store compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs > 1 && part.info.local_size == 4 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   if (!valid)
      fprintf(stderr, "AGX3 shared-store metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_g17p_shared_byte_vectors(void)
{
   for (unsigned channels = 2; channels <= 4; ++channels) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "packed shared bytes");
      b.shader->info.workgroup_size[0] = 1;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
      b.shader->info.shared_size = 16;
      nir_def *values[4];
      for (unsigned c = 0; c < channels; ++c)
         values[c] = nir_imm_intN_t(&b, 0x35 + c * 0x31, 8);
      nir_store_shared(&b, nir_vec(&b, values, channels), nir_imm_int(&b, 1),
                       .align_mul = 1, .write_mask = BITFIELD_MASK(channels),
                       .access = ACCESS_VOLATILE);
      nir_def *loaded = nir_load_shared(
         &b, channels, 8, nir_imm_int(&b, 1), .align_mul = 1,
         .access = ACCESS_VOLATILE);
      nir_store_global(&b, nir_u2u32(&b, loaded),
                       nir_imm_int64(&b, UINT64_C(0x2000000000)),
                       .align_mul = 4);
      agx_preprocess_nir_g17p(b.shader);
      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .no_preamble = true,
      };
      struct agx_shader_part part = {0};
      const bool ok = agx_compile_shader_nir(b.shader, &key, &part);
      free(part.binary);
      ralloc_free(b.shader);
      if (!ok) {
         fprintf(stderr, "shared byte vector rejected: %u channels\n", channels);
         return false;
      }
   }
   return true;
}

static bool
check_unlowered_printf_address_rejected(enum agx_isa_encoding encoding)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "unlowered printf address");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *printf_address = nir_load_printf_buffer_address(&b, 64);
   nir_store_global(&b, printf_address,
                    nir_imm_int64(&b, UINT64_C(0x2000000000)),
                    .align_mul = 8);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part;
   memset(&part, 0xa5, sizeof(part));

   if (agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 unlowered printf address was accepted\n");
      free(part.binary);
      ralloc_free(b.shader);
      return false;
   }

   const uint8_t zero[sizeof(part)] = {0};
   const bool cleared = memcmp(&part, zero, sizeof(part)) == 0;
   if (!cleared)
      fprintf(stderr, "AGX3 printf rejection did not clear compiler output\n");

   ralloc_free(b.shader);
   return cleared;
}

static bool
check_agx2_legacy_printf_address_accepted(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "legacy printf address");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *printf_address = nir_load_printf_buffer_address(&b, 64);
   nir_store_global(&b, printf_address,
                    nir_imm_int64(&b, UINT64_C(0x2000000000)),
                    .align_mul = 8);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX2;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   if (!compiled)
      fprintf(stderr, "AGX2 legacy printf address was rejected\n");

   free(part.binary);
   ralloc_free(b.shader);
   return compiled;
}

static bool
check_shared_rmw_compute(enum agx_isa_encoding encoding)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 shared RMW");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 32 * sizeof(uint32_t);

   nir_def *offset = nir_imul_imm(
      &b, nir_load_local_invocation_index(&b), sizeof(uint32_t));
   nir_def *value = nir_load_shared(&b, 1, 32, offset, .align_mul = 4);
   nir_store_shared(&b, nir_iadd_imm(&b, value, 1), offset, .align_mul = 4);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   /*
    * This shader's address arithmetic is `iadd dst32, #0, zext(src16) << 2`.
    * It briefly regressed to a refusal during the immediate/shift audit:
    * before that it "compiled", but the constant was deposited into a
    * register field and the shift was dropped, so it computed a wrong address
    * every time. Both fields are now probe-witnessed and encoded
    * (agx3_iadd_core.h), so it compiles again -- correctly this time.
    */
   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 shared-RMW compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > 4 &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs > 3 &&
      part.info.local_size == 32 * sizeof(uint32_t);

   if (!valid)
      fprintf(stderr, "AGX3 shared-RMW metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_g17p_saturating_iadd_immediate_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "G17P saturating integer add with immediate");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 2 * sizeof(uint32_t);

   nir_def *value = nir_load_shared(&b, 1, 32, nir_imm_int(&b, 0),
                                    .align_mul = 4);
   nir_def *bounded = nir_uadd_sat(&b, value, nir_imm_int(&b, 4));
   nir_store_shared(&b, bounded, nir_imm_int(&b, sizeof(uint32_t)),
                    .align_mul = 4);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr,
              "G17P saturating integer add with immediate was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid = part.binary != NULL && part.info.binary_size > 4 &&
                      part.info.main_offset == 0 &&
                      part.info.main_size == part.info.binary_size &&
                      part.info.stats.instrs > 3 &&
                      part.info.local_size == 2 * sizeof(uint32_t);

   if (!valid)
      fprintf(stderr,
              "G17P saturating integer add immediate metadata mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_local_id_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 local invocation ID");
   b.shader->info.workgroup_size[0] = 4;
   b.shader->info.workgroup_size[1] = 2;
   b.shader->info.workgroup_size[2] = 2;
   b.shader->info.shared_size = sizeof(uint32_t);

   nir_def *id = nir_load_local_invocation_id(&b);
   nir_def *sum = nir_iadd(
      &b, nir_iadd(&b, nir_channel(&b, id, 0), nir_channel(&b, id, 1)),
      nir_channel(&b, id, 2));
   nir_store_shared(&b, sum, nir_imm_int(&b, 0), .align_mul = 4);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 local-ID compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 7 && part.info.local_size == sizeof(uint32_t) &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   if (!valid)
      fprintf(stderr, "AGX3 local-ID metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_subgroup_id_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 subgroup IDs");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = sizeof(uint32_t);

   nir_def *sum = nir_iadd(&b, nir_load_subgroup_invocation(&b),
                           nir_load_subgroup_id(&b));
   nir_store_shared(&b, sum, nir_imm_int(&b, 0), .align_mul = 4);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 subgroup-ID compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 5 && part.info.local_size == sizeof(uint32_t) &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   if (!valid)
      fprintf(stderr, "AGX3 subgroup-ID metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_subgroup_multiply_compute(void)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P subgroup multiply");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *input_address = nir_imm_int64(&b, 0x1000);
   nir_def *output_address = nir_imm_int64(&b, 0x2000);
   nir_def *value = nir_load_global(
      &b, 1, 32, input_address, .align_mul = sizeof(uint32_t));
   nir_def *product =
      nir_reduce(&b, value, .reduction_op = nir_op_imul);
   nir_store_global(&b, product, output_address,
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "G17P subgroup-multiply shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs > 5 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;
   if (!valid)
      fprintf(stderr, "G17P subgroup-multiply metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_device_ballot_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 device-load ballot");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *input_address = nir_imm_int64(&b, 0x1000);
   nir_def *output_address = nir_imm_int64(&b, 0x2000);
   nir_def *value = nir_load_global(
      &b, 1, 32, input_address, .align_mul = sizeof(uint32_t));
   nir_def *mask = nir_ballot(&b, 1, 32, nir_ine_imm(&b, value, 0));
   nir_store_global(&b, mask, output_address,
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};
   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 device-load ballot shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 5 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;
   if (!valid)
      fprintf(stderr, "AGX3 device-load ballot metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_g17p_subgroup_vote_compute(void)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P subgroup vote");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *lane = nir_load_subgroup_invocation(&b);
   nir_def *ge32 = nir_uge(&b, lane, nir_imm_int(&b, 32));
   nir_def *eq0 = nir_ieq_imm(&b, lane, 0);
   nir_def *all_mask = nir_ballot(&b, 1, 32, ge32);
   nir_def *any_mask = nir_ballot(&b, 1, 32, eq0);
   nir_store_global(&b, all_mask, nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   nir_store_global(&b, any_mask, nir_imm_int64(&b, 0x2004),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "G17P subgroup-vote shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 5 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;
   if (!valid)
      fprintf(stderr, "G17P subgroup-vote metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_g17p_signed_immediate_ballot_compiles(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "G17P ballot of a signed-immediate compare");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *lane = nir_load_subgroup_invocation(&b);
   nir_def *mask = nir_ballot(&b, 1, 32,
                              nir_ilt(&b, lane, nir_imm_int(&b, 7)));
   nir_store_global(&b, mask, nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   free(part.binary);
   ralloc_free(b.shader);

   /* The AGX3B ballot has no signed-immediate compare form other than
    * against zero. The optimizer ballots the opposite compare through the
    * register form instead (agx3b_ballot_compare_encodable()), and the packer
    * still refuses any unwitnessed form, so the shader compiles. */
   if (!compiled)
      fprintf(stderr, "signed-immediate ballot failed to compile\n");
   return compiled;
}

static bool
check_g17p_subgroup_quad_compute(bool valid_lane)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P subgroup quad");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *lane = nir_load_subgroup_invocation(&b);
   nir_def *horizontal = nir_quad_swap_horizontal(&b, lane);
   nir_def *vertical = nir_quad_swap_vertical(&b, lane);
   /*
    * The lane is f29 on the quad forms exactly as on the SIMD ones, and
    * Apple's own compiler encodes quad_broadcast(a, 4) as f29 = 4 -- a lane
    * past the end of a quad is undefined behaviour in the source language,
    * not something the ISA refuses. What the field genuinely cannot hold is
    * a lane above seven bits, because the eighth spills into the source
    * register, so that is what this checks.
    */
   nir_def *broadcast =
      nir_quad_broadcast(&b, lane, nir_imm_int(&b, valid_lane ? 3 : 0x80));
   nir_def *sum = nir_iadd(&b, nir_iadd(&b, horizontal, vertical), broadcast);
   nir_store_global(&b, sum, nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);

   const bool valid = valid_lane
                         ? compiled && part.binary != NULL &&
                              part.info.binary_size > sizeof(end) &&
                              part.info.main_offset == 0 &&
                              part.info.main_size == part.info.binary_size &&
                              part.info.stats.instrs >= 6 &&
                              memcmp((const uint8_t *)part.binary +
                                        part.info.binary_size - sizeof(end),
                                     end, sizeof(end)) == 0
                         : !compiled;
   if (!valid)
      fprintf(stderr, "G17P subgroup-quad %s lane %s\n",
              valid_lane ? "valid" : "out-of-range",
              compiled ? "was accepted" : "was rejected");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

/*
 * A quad broadcast whose lane is a REGISTER rather than an immediate has no
 * 16-bit encoding: agx3_subgroup_encode_shuffle() narrows the width pair only
 * on the immediate-lane word, and no Apple witness exists for a narrow
 * register-lane quad shuffle.  Every
 * dEQP-VK.subgroups.quad.compute.subgroupquadbroadcast_nonconst_* case on a
 * scalar 8-bit, 16-bit or boolean type reached that shape -- 8-bit values and
 * booleans live in 16-bit registers -- and the refusal failed the WHOLE
 * shader:
 *
 *   agx3_pack_binary.c: no grounded AGX3 encoding for quad_shuffle
 *     [d16:reg13 s0=16:reg13 s1=32:reg2 ...]
 *   hk: vkCreateComputePipelines refused pipeline 0 of 1: VK_ERROR_UNKNOWN
 *
 * measured on Neo (G17P, A18 Pro).  The backend widens that shape to the
 * 32-bit register-lane form, which is the same one nir_lower_subgroups
 * already produces for the narrow VECTOR types that passed all along.  Both
 * widths have to compile.
 */
static bool
check_g17p_quad_broadcast_dynamic_lane(unsigned bits)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P dynamic quad broadcast");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *value = nir_load_global(&b, 1, bits, nir_imm_int64(&b, 0x1000),
                                    .align_mul = sizeof(uint32_t));
   nir_def *lane = nir_iand_imm(
      &b, nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x1004),
                          .align_mul = sizeof(uint32_t)),
      3);
   nir_def *broadcast = nir_quad_broadcast(&b, value, lane);
   nir_store_global(&b, nir_u2u32(&b, broadcast), nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   free(part.binary);
   ralloc_free(b.shader);

   if (!compiled)
      fprintf(stderr, "G17P %u-bit quad broadcast from a register lane was "
                      "refused\n", bits);
   return compiled;
}

static bool
check_device_shuffle_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 device-load shuffle");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *value = nir_load_global(
      &b, 1, 32, nir_imm_int64(&b, 0x1000),
      .align_mul = sizeof(uint32_t));
   nir_def *lane = nir_iand_imm(
      &b, nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x1004),
                          .align_mul = sizeof(uint32_t)),
      31);
   nir_def *shuffled = nir_read_invocation(&b, value, lane);
   nir_store_global(&b, shuffled, nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};
   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 device-load shuffle shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 7 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;
   if (!valid)
      fprintf(stderr, "AGX3 device-load shuffle metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

/* popcount(ballot(true)) reads the active thread count on AGX2 and stays a
 * ballot on G17P, where that register is only witnessed inside Apple's
 * helper-guarded uniform atomics. */
static bool
check_active_count_lowering(bool g17p)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "active subgroup count");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *mask = nir_ballot(&b, 1, 32, nir_imm_true(&b));
   nir_store_global(&b, nir_bit_count(&b, mask), nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   if (g17p)
      agx_preprocess_nir_g17p(b.shader);
   else
      agx_preprocess_nir(b.shader);

   bool count = false, ballot = false;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            const nir_intrinsic_op op =
               nir_instr_as_intrinsic(instr)->intrinsic;
            count |= op == nir_intrinsic_load_active_subgroup_count_agx;
            ballot |= op == nir_intrinsic_ballot;
         }
      }
   }

   const bool valid = g17p ? (ballot && !count) : (count && !ballot);
   if (!valid)
      fprintf(stderr, "active subgroup count lowering wrong for %s\n",
              g17p ? "G17P" : "AGX2");
   ralloc_free(b.shader);
   return valid;
}

static bool
check_elect_broadcast_first_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "AGX3 divergent elect and broadcast-first");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *value = nir_load_global(
      &b, 1, 32, nir_imm_int64(&b, 0x1000),
      .align_mul = sizeof(uint32_t));
   nir_push_if(&b, nir_ine_imm(&b, value, 0));
   nir_def *first = nir_read_first_invocation(&b, value);
   nir_def *elected = nir_b2i32(&b, nir_elect(&b, 1));
   nir_store_global(&b, nir_iadd(&b, first, elected),
                    nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   nir_pop_if(&b, NULL);
   agx_preprocess_nir(b.shader);

   /* Both must key off the active thread index. A subgroup reduction also
    * sees lanes masked off by the enclosing control flow, so reducing the
    * invocation IDs elects the wrong lane there. */
   bool lowered = true;
   bool active_index = false;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            const nir_intrinsic_op op =
               nir_instr_as_intrinsic(instr)->intrinsic;
            if (op == nir_intrinsic_elect ||
                op == nir_intrinsic_read_first_invocation ||
                op == nir_intrinsic_reduce)
               lowered = false;
            if (op == nir_intrinsic_load_active_subgroup_invocation_agx)
               active_index = true;
         }
      }
   }
   lowered &= active_index;

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};
   const bool compiled =
      lowered && agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid =
      compiled && part.binary != NULL && part.info.binary_size > sizeof(end) &&
      (encoding == AGX_ISA_ENCODING_AGX2 ||
       memcmp((const uint8_t *)part.binary + part.info.binary_size -
                 sizeof(end),
              end, sizeof(end)) == 0);
   if (!valid)
      fprintf(stderr,
              "elect/broadcast-first lowering failed for encoding %u\n",
              encoding);

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_device_rmw_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 device RMW");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *address = nir_imm_int64(&b, 0x1000);
   nir_def *value =
      nir_load_global(&b, 1, 32, address, .align_mul = sizeof(uint32_t));
   nir_store_global(&b, nir_iadd_imm(&b, value, 1), address,
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 device-RMW compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 4 &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   if (!valid)
      fprintf(stderr, "AGX3 device-RMW metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_device_atomic_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};
   static const struct {
      nir_atomic_op op;
      const char *name;
   } cases[] = {
      {nir_atomic_op_iadd, "add"},   {nir_atomic_op_imin, "imin"},
      {nir_atomic_op_umin, "umin"}, {nir_atomic_op_imax, "imax"},
      {nir_atomic_op_umax, "umax"}, {nir_atomic_op_xchg, "xchg"},
      {nir_atomic_op_cmpxchg, "cmpxchg"},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 device atomic family");
      b.shader->info.workgroup_size[0] = 1;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      nir_def *address = nir_imm_int64(&b, 0x1000);
      nir_def *value_address = nir_imm_int64(&b, 0x2000);
      nir_def *value = nir_load_global(&b, 1, 32, value_address,
                                       .align_mul = sizeof(uint32_t));
      nir_def *old;
      if (cases[i].op == nir_atomic_op_cmpxchg) {
         nir_def *replacement = nir_load_global(
            &b, 1, 32,
            nir_iadd_imm(&b, value_address, sizeof(uint32_t)),
            .align_mul = sizeof(uint32_t));
         old = nir_global_atomic_swap(&b, 32, address, value, replacement,
                                      .atomic_op = nir_atomic_op_cmpxchg);
      } else {
         old = nir_global_atomic(&b, 32, address, value,
                                 .atomic_op = cases[i].op);
      }

      /* Keep both the returned-old value and the input live.  AGX3b must copy
       * a live scalar input to a private register before using its in-place
       * return form; cmpxchg must preserve [replacement, compare] ordering.
       */
      nir_store_global(&b, old, nir_imm_int64(&b, 0x3000),
                       .align_mul = sizeof(uint32_t));
      nir_store_global(&b, value, nir_imm_int64(&b, 0x3004),
                       .align_mul = sizeof(uint32_t));
      agx_preprocess_nir(b.shader);

      struct agx_shader_key key = {0};
      key.dev.encoding = encoding;
      struct agx_shader_part part = {0};

      if (!agx_compile_shader_nir(b.shader, &key, &part)) {
         fprintf(stderr, "AGX3 device atomic %s was rejected\n",
                 cases[i].name);
         ralloc_free(b.shader);
         return false;
      }

      const bool valid =
         part.binary != NULL && part.info.binary_size > sizeof(end) &&
         part.info.main_offset == 0 &&
         part.info.main_size == part.info.binary_size &&
         part.info.stats.instrs >= 6 &&
         memcmp((const uint8_t *)part.binary +
                   part.info.binary_size - sizeof(end),
                end, sizeof(end)) == 0;

      if (!valid)
         fprintf(stderr, "AGX3 device atomic %s metadata/binary mismatch\n",
                 cases[i].name);

      free(part.binary);
      ralloc_free(b.shader);
      if (!valid)
         return false;
   }

   return true;
}

static bool
check_uniform_atomic_tcs(enum agx_isa_encoding encoding)
{
   static const struct {
      nir_atomic_op op;
      unsigned value;
   } cases[] = {
      {nir_atomic_op_iadd, 2}, {nir_atomic_op_imin, 3},
      {nir_atomic_op_imax, 4}, {nir_atomic_op_iand, 0},
      {nir_atomic_op_ior, 7},  {nir_atomic_op_ixor, 4},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_TESS_CTRL, &agx_nir_options, "uniform TCS atomic");
      b.shader->info.workgroup_size[0] = 1;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      /* Software TCS keeps its API stage while running as a kernel. Unlike
       * compute with a 1x1x1 workgroup, its uniform-address atomics go through
       * nir_opt_uniform_atomics, including constant exclusive scans. Images
       * have already been lowered to global addresses at this point. */
      nir_def *old = nir_global_atomic(
         &b, 32, nir_imm_int64(&b, 0x1000), nir_imm_int(&b, cases[i].value),
         .atomic_op = cases[i].op);
      nir_store_global(&b, old, nir_imm_int64(&b, 0x2000), .align_mul = 4);
      agx_preprocess_nir(b.shader);

      struct agx_shader_key key = {0};
      key.dev.encoding = encoding;
      struct agx_shader_part part = {0};
      bool valid = agx_compile_shader_nir(b.shader, &key, &part);
      valid &= part.binary != NULL && part.info.main_size > 0;
      if (!valid)
         fprintf(stderr, "uniform TCS atomic op %u rejected (encoding %u)\n",
                 cases[i].op, encoding);
      free(part.binary);
      ralloc_free(b.shader);
      if (!valid)
         return false;
   }

   return true;
}

static bool
check_shared_atomic_compute(enum agx_isa_encoding encoding)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 shared atomic");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = sizeof(uint32_t);

   nir_def *old = nir_shared_atomic(&b, 32, nir_imm_int(&b, 0),
                                    nir_imm_int(&b, 1),
                                    .atomic_op = nir_atomic_op_iadd);
   nir_store_shared(&b, old, nir_imm_int(&b, 0),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "AGX3 shared-atomic compute shader was rejected\n");
      ralloc_free(b.shader);
      return false;
   }

   const bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      part.info.main_offset == 0 &&
      part.info.main_size == part.info.binary_size &&
      part.info.stats.instrs >= 4 &&
      part.info.local_size == sizeof(uint32_t) &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   if (!valid)
      fprintf(stderr, "AGX3 shared-atomic metadata/binary mismatch\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

/*
 * Mirrors the G17P milestone compute probe shader in the shape the AGX3
 * backend actually receives it, i.e. after Honeykrisp lowering:
 *
 *   - the deliberately-oversized 65-entry constant array becomes
 *     nir->constant_data plus a load_constant, and hk_nir_lower_descriptors.c
 *     (lower_load_constant, :66-93) rewrites that into a speculatable global
 *     load through the root descriptor's cs.constant_data_addr;
 *   - the SSBO store becomes a global store through a descriptor-set pointer;
 *   - hk_shader.c lower_uniforms (:1026-1057) turns load_root_agx and
 *     load_descriptor_set_agx into load_preamble uniform reads.
 *
 * No other AGX3 test compiles a compute shader carrying NIR constant data,
 * and hk_dispatch_g17p_native() has no fallback: if this shape does not
 * compile, or if the promoted-constant range that falls out of it is not
 * 32-bit aligned in both base and size, hk_g17p_compute_uniform_ranges()
 * (hk_cmd_dispatch.c:168-196) rejects the dispatch with
 * VK_ERROR_FEATURE_NOT_PRESENT.  Both facts are asserted here.
 */
static bool
check_constant_data_compute(enum agx_isa_encoding encoding, bool soft_fault)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};

   /* Byte offset of cs.constant_data_addr inside the Honeykrisp root
    * descriptor table.  The exact value is immaterial to the encoding; what
    * matters is that the address is loaded out of the root rather than being
    * an immediate.
    */
   const unsigned root_constant_data_offset = 16;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 constant data");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   /* Exactly the payload nir_opt_large_constants leaves behind for
    * the milestone probe's deltas[65].
    */
   uint32_t deltas[65];
   for (unsigned i = 0; i < ARRAY_SIZE(deltas); ++i)
      deltas[i] = 3 + i;

   b.shader->constant_data = ralloc_size(b.shader, sizeof(deltas));
   memcpy(b.shader->constant_data, deltas, sizeof(deltas));
   b.shader->constant_data_size = sizeof(deltas);

   nir_def *root = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *set0 = nir_load_preamble(&b, 1, 64, .base = 4);

   nir_def *index = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *offset = nir_imul_imm(&b, index, sizeof(uint32_t));

   nir_def *constant_data_addr = nir_load_global_constant(
      &b, 1, 64, nir_iadd_imm(&b, root, root_constant_data_offset),
      .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
   nir_def *delta = nir_load_global_constant(
      &b, 1, 32, nir_iadd(&b, constant_data_addr, nir_u2u64(&b, offset)),
      .align_mul = 4, .access = ACCESS_CAN_SPECULATE);

   nir_def *output = nir_load_global_constant(&b, 1, 64, set0, .align_mul = 8,
                                              .access = ACCESS_CAN_SPECULATE);
   nir_def *value =
      nir_iadd(&b, nir_iadd_imm(&b, index, 0x12340000), delta);

   nir_store_global(&b, value,
                    nir_iadd(&b, output, nir_u2u64(&b, offset)),
                    .align_mul = 4);

   agx_preprocess_nir(b.shader);

   /* Exactly the backend key hk_compile_shader() builds for a G17P compute
    * shader with one descriptor set (hk_shader.c:1220-1250): root == 0 for
    * compute, so reserved_preamble == 4 * (1 + set_count), and G17P forces
    * no_preamble because hk_g17p_compute_shader_supported() refuses a
    * preamble.  soft_fault is exercised both
    * ways: the kernel advertises DRM_ASAHI_FEATURE_SOFT_FAULTS whenever its
    * fault_control module parameter is 0xb, which is the default, but
    * AGX_MESA_DEBUG=nosoft and a hard-fault boot both clear it, and it changes
    * how the speculatable descriptor loads above are lowered.
    */
   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   key.dev.soft_fault = soft_fault;
   key.reserved_preamble = 4 * (1 + 1);
   key.has_scratch = true;
   key.promote_constants = true;
   key.promote_textures = true;
   key.no_preamble = true;
   struct agx_shader_part part = {0};

   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr,
              "AGX3 constant-data compute shader was rejected "
              "(soft_fault %u)\n",
              soft_fault);
      ralloc_free(b.shader);
      return false;
   }

   /* main_offset is deliberately not required to be zero: when the backend
    * emits a rodata blob it is uploaded *before* the code and the shader body
    * is then padded up to AGX_CODE_ALIGN (agx_compile.c:3684-3697), so a
    * constant-carrying shader legitimately starts past the top of its binary.
    */
   bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      (uint64_t)part.info.main_offset + part.info.main_size <=
         part.info.binary_size &&
      part.info.main_size > sizeof(end) && part.info.stats.instrs >= 6 &&
      part.info.nr_gprs != 0 && part.info.local_size == 0 &&
      part.info.scratch_size == 0 && part.info.preamble_scratch_size == 0 &&
      !part.info.has_preamble && part.info.texture_state_count == 0 &&
      part.info.sampler_state_count == 0 && !part.info.uses_sampler_heap &&
      !part.info.uses_txf;

   if (!valid)
      fprintf(stderr,
              "AGX3 constant-data shader does not satisfy "
              "hk_g17p_compute_shader_supported(): binary_size %u "
              "main_offset %u main_size %u instrs %u local_size %u "
              "scratch %u preamble_scratch %u has_preamble %u nr_gprs %u "
              "push_count %u tex %u smp %u heap %u txf %u "
              "rodata{base %u size_16 %u offset %u}\n",
              part.info.binary_size, part.info.main_offset,
              part.info.main_size, part.info.stats.instrs,
              part.info.local_size, part.info.scratch_size,
              part.info.preamble_scratch_size, part.info.has_preamble,
              part.info.nr_gprs, part.info.push_count,
              part.info.texture_state_count, part.info.sampler_state_count,
              part.info.uses_sampler_heap, part.info.uses_txf,
              part.info.rodata.base_uniform, part.info.rodata.size_16,
              part.info.rodata.offset);

   /* The rodata blob, when the backend emits one, is appended after main, so
    * main_size no longer covers the whole binary.  Whatever the split, the
    * shader body must still terminate.
    */
   const struct agx_rodata *rodata = &part.info.rodata;
   if (valid && rodata->size_16 != 0) {
      if (rodata->offset > part.info.binary_size ||
          (uint64_t)rodata->size_16 * 2 >
             (uint64_t)part.info.binary_size - rodata->offset) {
         fprintf(stderr, "AGX3 constant-data rodata blob is out of bounds\n");
         valid = false;
      }

      /* G17P's dword loader covers an odd final half with the binary's zeroed
       * alignment padding, but its destination base remains dword aligned. */
      const uint32_t loaded_halfs = ((uint32_t)rodata->size_16 + 1) & ~1u;
      if (valid && ((rodata->base_uniform & 1) != 0 || loaded_halfs > 512 ||
                    rodata->base_uniform > 512 - loaded_halfs ||
                    (uint64_t)loaded_halfs * 2 >
                       (uint64_t)part.info.binary_size - rodata->offset)) {
         fprintf(stderr,
                 "AGX3 constant-data rodata range (base %u, size_16 %u) is "
                 "not bindable by hk_g17p_compute_uniform_ranges()\n",
                 rodata->base_uniform, rodata->size_16);
         valid = false;
      }

      /* The rodata range must sit above the root/descriptor-set uniforms that
       * hk_agx3_g17p_root_range() already claims, or the two Ioto uniform
       * ranges would overlap.
       */
      if (valid && rodata->base_uniform < 4 * (1 + 1)) {
         fprintf(stderr,
                 "AGX3 constant-data rodata base %u collides with the root "
                 "uniform range\n",
                 rodata->base_uniform);
         valid = false;
      }
   }

   if (valid &&
       memcmp((const uint8_t *)part.binary + part.info.main_offset +
                 part.info.main_size - sizeof(end),
              end, sizeof(end)) != 0) {
      fprintf(stderr, "AGX3 constant-data shader is not terminated\n");
      valid = false;
   }

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

/* The generated preamble writes a 64-bit address back to two consecutive
 * uniform dwords. Each dword uses Apple's byte-exact reg-to-uniform copy form;
 * this checks the complete compiler path which Gallium's blitter VS needs. */
static bool
check_preamble_compute(enum agx_isa_encoding encoding, bool enable,
                       unsigned uniform_loads)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 generated preamble");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *root = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *output = nir_load_global_constant(&b, 1, 64, root, .align_mul = 8,
                                              .access = ACCESS_CAN_SPECULATE);
   nir_def *index = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);

   /* Keep a nontrivial uniform calculation live through a varying store.
    * This exercises a real compiler-generated external preamble, rather than
    * padding the short fixture to cross the native 256-byte inline limit. */
   nir_def *value = index;
   if (uniform_loads) {
      nir_def *sum = nir_imm_int(&b, 0);
      for (unsigned i = 0; i < uniform_loads; ++i) {
         nir_def *input = nir_load_global_constant(
            &b, 1, 32, nir_iadd_imm(&b, root, 8 + i * sizeof(uint32_t)),
            .align_mul = 4, .access = ACCESS_CAN_SPECULATE);
         sum = nir_ixor(&b, nir_imul_imm(&b, sum, 33), input);
      }
      value = nir_ixor(&b, sum, index);
   }

   nir_store_global(
      &b, value,
      nir_iadd(&b, output,
               nir_u2u64(&b, nir_imul_imm(&b, index, sizeof(uint32_t)))),
      .align_mul = 4);

   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   key.reserved_preamble = 4 * (1 + 1);
   key.no_preamble = !enable;
   key.has_scratch = true;
   key.promote_constants = true;
   key.promote_textures = true;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool supported = encoding == AGX_ISA_ENCODING_AGX3B ||
                          encoding == AGX_ISA_ENCODING_AGX2 ||
                          encoding == AGX_ISA_ENCODING_AGX3C;
   bool valid = supported ? compiled && part.binary != NULL
                          : !compiled && part.binary == NULL;
   if (supported && valid) {
      const struct agx_shader_info *info = &part.info;
      valid = info->has_preamble == enable &&
              (info->preamble_size != 0) == enable &&
              (info->nr_preamble_gprs != 0) == enable &&
              info->preamble_offset <= info->binary_size &&
              info->preamble_size <= info->binary_size - info->preamble_offset &&
              info->main_offset <= info->binary_size &&
              info->main_size <= info->binary_size - info->main_offset;

      if (encoding == AGX_ISA_ENCODING_AGX3B) {
         valid &= info->agx3_staging_known &&
                  info->agx3_preamble_staging_known == enable;
         /* Constant loads return directly to GPRs. These preambles therefore
          * need temporary registers but no IPR staging; a known zero must
          * remain distinct from missing metadata and from the GPR count. */
         valid &= info->agx3_preamble_staging_halfs == 0;
      } else if (encoding == AGX_ISA_ENCODING_AGX2) {
         valid &= !info->agx3_preamble_staging_known &&
                  info->agx3_preamble_staging_halfs == 0;
      }

      if (valid && enable) {
         valid = info->preamble_offset + info->preamble_size <= info->main_offset ||
                 info->main_offset + info->main_size <= info->preamble_offset;
         if (uniform_loads)
            valid &= info->preamble_size > 256 && info->scratch_size == 0 &&
                     info->preamble_scratch_size == 0;
      }

      /* G17P's constant-program suffix must copy the executable through END,
       * without including inter-function padding or adjacent constant data. */
      if (valid && enable && encoding == AGX_ISA_ENCODING_AGX3B) {
         static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};
         valid = info->preamble_size >= sizeof(end) &&
                 memcmp((const uint8_t *)part.binary + info->preamble_offset +
                           info->preamble_size - sizeof(end),
                        end, sizeof(end)) == 0;
      }

      if (valid) {
         const struct agx_precompiled_kernel_info compact =
            agx_compact_kernel_info(&part.info);
         valid = compact.preamble_offset == info->preamble_offset &&
                 compact.preamble_size == info->preamble_size &&
                 compact.agx3_preamble_staging_halfs ==
                    info->agx3_preamble_staging_halfs &&
                 compact.agx3_preamble_staging_known ==
                    info->agx3_preamble_staging_known;
      }
   }
   if (supported && (uniform_loads || !valid)) {
      fprintf(stderr,
              "preamble workload encoding=%u enabled=%u inputs=%u main=%u "
              "preamble=%u gprs=%u preamble_gprs=%u staging_halfs=%u "
              "preamble_staging_halfs=%u valid=%u\n",
              encoding, enable, uniform_loads, part.info.main_size,
              part.info.preamble_size, part.info.nr_gprs,
              part.info.nr_preamble_gprs, part.info.agx3_staging_halfs,
              part.info.agx3_preamble_staging_halfs, valid);
   }
   if (!valid)
      fprintf(stderr, "AGX3 generated 64-bit preamble write failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

/* A preamble has uniform inputs, but no per-invocation entry-register ABI. */
enum preamble_input_test {
   PREAMBLE_UNIFORM_INPUT,
   PREAMBLE_ENTRY_GPR,
   MAIN_ENTRY_GPR,
   PREAMBLE_SPARSE_UNIFORMS,
   PREAMBLE_CONSTANT_ONLY,
};

static bool
check_preamble_entry_gpr(enum agx_isa_encoding encoding, bool g18g,
                         enum preamble_input_test input)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "preamble input ABI");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *root = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *store_value = nir_load_preamble(&b, 1, 32, .base = 4);
   if (input == MAIN_ENTRY_GPR)
      store_value = nir_ixor(&b, store_value, nir_load_vertex_id_zero_base(&b));
   nir_store_global(&b, store_value, root, .align_mul = 4);

   nir_builder p = nir_builder_create(nir_shader_get_preamble(b.shader));
   p.cursor = nir_after_impl(p.impl);
   nir_def *value;
   if (input == PREAMBLE_ENTRY_GPR) {
      /* Software vertex invocation data arrives in a fixed GPR. It is not
       * a draw/dispatch-uniform input and must not be used by a preamble. */
      value = nir_load_vertex_id_zero_base(&p);
   } else if (input == PREAMBLE_CONSTANT_ONLY) {
      value = nir_imm_int(&p, 47);
      nir_load_preamble(&p, 1, 64, .base = 196);
   } else if (input == PREAMBLE_SPARSE_UNIFORMS) {
      nir_def *a = nir_load_global_constant(
         &p, 1, 32, nir_load_preamble(&p, 1, 64, .base = 8),
         .align_mul = 4, .access = ACCESS_CAN_SPECULATE);
      nir_def *b = nir_load_global_constant(
         &p, 1, 32, nir_load_preamble(&p, 1, 64, .base = 508),
         .align_mul = 4, .access = ACCESS_CAN_SPECULATE);
      value = nir_ixor(&p, a, b);
      value = nir_ixor(&p, value, nir_load_preamble(&p, 1, 32, .base = 64));
      value = nir_ixor(&p, value, nir_u2u32(&p, nir_load_preamble(&p, 1, 16, .base = 127)));
      nir_load_preamble(&p, 1, 64, .base = 196);
   } else {
      value = nir_load_global_constant(
         &p, 1, 32, nir_load_preamble(&p, 1, 64, .base = 0),
         .align_mul = 4, .access = ACCESS_CAN_SPECULATE);
   }
   nir_store_preamble(&p, value, .base = 4,
                      .preamble_class = nir_preamble_class_general);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = encoding;
   key.dev.emit_agx3d = g18g;
   key.reserved_preamble = 8;
   key.no_preamble = true; /* Preserve the explicit preamble without adding one. */
   key.has_scratch = true;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   bool valid = input == PREAMBLE_ENTRY_GPR
                         ? !compiled && !part.binary &&
                              !part.info.has_preamble && !part.info.binary_size
                         : compiled && part.binary && part.info.has_preamble;
   fprintf(stderr, "preamble input encoding=%u mode=%u compiled=%u valid=%u\n",
           encoding, input, compiled, valid);
   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

/*
 * hk_nir_lower_descriptors.c lower_load_constant (:66-93), reproduced so this
 * test can run the same ordering Honeykrisp does without linking libhk.
 */
static bool
lower_load_constant_like_hk(nir_builder *b, nir_intrinsic_instr *intr,
                            UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_constant)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   nir_def *root = nir_load_preamble(b, 1, 64, .base = 0);
   nir_def *blob = nir_load_global_constant(b, 1, 64, nir_iadd_imm(b, root, 16),
                                            .align_mul = 8,
                                            .access = ACCESS_CAN_SPECULATE);
   nir_def *offset =
      nir_iadd_imm(b, intr->src[0].ssa, nir_intrinsic_base(intr));
   nir_def *data_def = nir_load_global_constant(
      b, intr->def.num_components, intr->def.bit_size,
      nir_iadd(b, blob, nir_u2u64(b, offset)),
      .align_mul = nir_intrinsic_align_mul(intr),
      .access = ACCESS_CAN_SPECULATE);

   nir_def_rewrite_uses(&intr->def, data_def);
   return true;
}

/*
 * The milestone probe's 65-entry deltas[] in the shape glslang and
 * nir_lower_variable_initializers hand to the driver: a function_temp array
 * filled by constant stores in one block, then read at a dynamic index.
 * 65 * 4 == 260 bytes, deliberately just over the 256-byte threshold of the
 * nir_lower_vars_to_scratch call inside agx_preprocess_nir
 * (agx_compile.c:3852) -- which is the whole point of this test.
 */
static nir_builder
build_constant_array_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 constant array");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   const unsigned n = 65;
   nir_variable *var = nir_local_variable_create(
      b.impl, glsl_array_type(glsl_uint_type(), n, 4), "deltas");
   nir_deref_instr *array = nir_build_deref_var(&b, var);

   for (unsigned i = 0; i < n; ++i) {
      nir_store_deref(&b, nir_build_deref_array_imm(&b, array, i),
                      nir_imm_int(&b, 3 + i), 0x1);
   }

   nir_def *index = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *delta =
      nir_load_deref(&b, nir_build_deref_array(&b, array, index));
   nir_def *output = nir_load_global_constant(
      &b, 1, 64, nir_load_preamble(&b, 1, 64, .base = 4), .align_mul = 8,
      .access = ACCESS_CAN_SPECULATE);

   nir_store_global(
      &b, nir_iadd(&b, nir_iadd_imm(&b, index, 0x12340000), delta),
      nir_iadd(&b, output,
               nir_u2u64(&b, nir_imul_imm(&b, index, sizeof(uint32_t)))),
      .align_mul = 4);

   return b;
}

static void
g17p_backend_key(enum agx_isa_encoding encoding, struct agx_shader_key *key)
{
   *key = (struct agx_shader_key){0};
   key->dev.encoding = encoding;
   key->dev.chip = encoding == AGX_ISA_ENCODING_AGX3B ? AGX_CHIP_G17P
                                                       : AGX_CHIP_G13G;
   key->reserved_preamble = 4 * (1 + 1);
   key->has_scratch = true;
   key->promote_constants = true;
   key->promote_textures = true;
   key->no_preamble = true;
}

/*
 * The pass-ordering contract for G17P constant data, pinned in both
 * directions.
 *
 * Honeykrisp must run nir_opt_large_constants BEFORE agx_preprocess_nir
 * (hk_shader.c hk_preprocess_nir_internal).  Run it after and the array is
 * already scratch, nir_opt_large_constants has no function_temp variable left
 * to move, and the shader reaches the backend as stack_store/stack_load.
 * AGX3B now has a grounded dynamic/static TPS path for that fallback.
 *
 * `ordered` reproduces the correct order and must compile with no scratch and
 * a non-empty constant blob. `!ordered` pins the bounded AGX3B scratch launch.
 */
static bool
check_constant_array_ordering(enum agx_isa_encoding encoding, bool ordered)
{
   nir_builder b = build_constant_array_compute();

   if (ordered)
      NIR_PASS(_, b.shader, nir_opt_large_constants, NULL, 32);

   agx_preprocess_nir(b.shader);

   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
            lower_load_constant_like_hk, nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);

   const bool has_constants = b.shader->constant_data_size != 0;
   const bool has_scratch = b.shader->scratch_size != 0;

   struct agx_shader_key key;
   g17p_backend_key(encoding, &key);
   struct agx_shader_part part;
   memset(&part, 0xa5, sizeof(part));

   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);

   bool valid;
   if (ordered) {
      valid = has_constants && !has_scratch && compiled &&
              part.info.scratch_size == 0 &&
              part.info.preamble_scratch_size == 0 && !part.info.has_preamble &&
              part.binary != NULL && part.info.binary_size != 0;

      if (!valid) {
         fprintf(stderr,
                 "AGX3 constant array did not survive as constant data: "
                 "constant_data_size %u scratch %u compiled %d "
                 "info.scratch %u\n",
                 b.shader->constant_data_size, b.shader->scratch_size, compiled,
                 compiled ? part.info.scratch_size : 0);
      }
   } else {
      /* Without the pass the array must land in scratch... */
      valid = has_scratch && !has_constants;
      if (!valid) {
         fprintf(stderr,
                 "AGX3 constant array test is no longer exercising the "
                 "scratch path (constant_data_size %u scratch %u); retune the "
                 "array size against agx_compile.c's nir_lower_vars_to_scratch "
                 "threshold\n",
                 b.shader->constant_data_size, b.shader->scratch_size);
      }

      if (encoding == AGX_ISA_ENCODING_AGX3B) {
         /* The grounded AGX3B TPS backend must preserve it in a launchable,
          * bounded private frame. hk_g17p_compute_shader_supported() already
          * admits this same 4096-byte bound. */
         if (valid)
            valid = compiled && part.binary != NULL &&
                    part.info.binary_size > 4 && part.info.main_size > 4 &&
                    !part.info.has_preamble &&
                    part.info.preamble_scratch_size == 0 &&
                    part.info.scratch_size > 0 &&
                    part.info.scratch_size <= 4096 &&
                    part.info.agx3_staging_known;

         if (!valid)
            fprintf(stderr,
                    "AGX3B scratch fallback is not a bounded launch: "
                    "compiled %u binary %u main %u scratch %u preamble %u "
                    "staging %u\n",
                    compiled, compiled ? part.info.binary_size : 0,
                    compiled ? part.info.main_size : 0,
                    compiled ? part.info.scratch_size : 0,
                    compiled ? part.info.preamble_scratch_size : 0,
                    compiled ? part.info.agx3_staging_known : 0);
      }
   }

   if (compiled)
      free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

/*
 * AGX3 ALU operands are register-or-immediate only; the uniform file is
 * reachable only from the memory-op bases, the texture heap and the subgroup
 * sources.  agx_instr_accepts_uniform() is what keeps a promoted constant out
 * of an AGX3 ALU source, and getting it wrong costs a whole-shader refusal
 * with `no grounded AGX3 encoding for iadd`.
 */
static bool
check_mixed_fma_neg_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P mixed FMA fneg");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 4;

   nir_def *lane = nir_u2f32(&b, nir_load_local_invocation_index(&b));
   nir_def *half = nir_f2f16(&b, nir_fadd_imm(&b, lane, 0.5f));
   nir_def *half32 = nir_f2f32(&b, half);
   nir_def *result = nir_ffma(&b, nir_fneg(&b, lane), half32,
                              nir_imm_float(&b, 1.0f));
   nir_store_shared(&b, result, nir_imm_int(&b, 0),
                    .align_mul = 4);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary != NULL &&
                      part.info.binary_size > 4 && part.info.stats.instrs >= 7;
   if (!valid)
      fprintf(stderr, "G17P mixed-source negative FMA lowering failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_add_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P shaderFloat16 add");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *input = nir_load_global(
      &b, 2, 32, nir_imm_int64(&b, 0x1000),
      .align_mul = sizeof(uint32_t));
   nir_def *left = nir_f2f16(&b, nir_channel(&b, input, 0));
   nir_def *right = nir_f2f16(&b, nir_channel(&b, input, 1));
   nir_def *sum = nir_fadd(&b, left, right);
   nir_store_global(&b, nir_f2f32(&b, sum), nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary != NULL &&
                      part.info.binary_size > 4 && part.info.stats.instrs >= 7;
   if (!valid)
      fprintf(stderr, "G17P shaderFloat16 add lowering failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_fma_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P shaderFloat16 FMA");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *input = nir_load_global(
      &b, 3, 32, nir_imm_int64(&b, 0x1000),
      .align_mul = sizeof(uint32_t));
   nir_def *left = nir_f2f16(&b, nir_channel(&b, input, 0));
   nir_def *right = nir_f2f16(&b, nir_channel(&b, input, 1));
   nir_def *addend = nir_f2f16(&b, nir_channel(&b, input, 2));
   nir_def *result = nir_ffma(&b, left, right, addend);
   nir_store_global(&b, nir_f2f32(&b, result), nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary != NULL &&
                      part.info.binary_size > 4 && part.info.stats.instrs >= 8;
   if (!valid)
      fprintf(stderr, "G17P shaderFloat16 FMA lowering failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_pk_immediate_fma_sequence(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P CTS pk_immediate FMA sequence");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *zero = nir_imm_floatN_t(&b, 0.0, 16);
   nir_def *one = nir_imm_floatN_t(&b, 1.0, 16);
   nir_def *two = nir_imm_floatN_t(&b, 2.0, 16);
   nir_def *a = nir_load_global(&b, 2, 16, nir_imm_int64(&b, 0x1000),
                                .align_mul = sizeof(uint32_t));
   nir_def *multipliers[] = {
      nir_vec2(&b, two, two),
      nir_vec2(&b, zero, two),
      nir_vec2(&b, two, zero),
      nir_vec2(&b, two, one),
   };

   for (unsigned i = 0; i < ARRAY_SIZE(multipliers); ++i) {
      nir_def *address = nir_imm_int64(&b, 0x2000 + i * sizeof(uint32_t));
      nir_def *addend = nir_load_global(&b, 2, 16, address,
                                        .align_mul = sizeof(uint32_t));
      nir_store_global(&b, nir_ffma(&b, multipliers[i], a, addend), address,
                       .align_mul = sizeof(uint32_t));
   }

   nir_store_global(&b, a, nir_imm_int64(&b, 0x3000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary != NULL &&
                      part.info.binary_size > 4 && part.info.stats.instrs >= 12;
   if (!valid)
      fprintf(stderr, "G17P CTS pk_immediate FMA sequence failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_fms_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P shaderFloat16 FMS");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *input = nir_load_global(
      &b, 3, 32, nir_imm_int64(&b, 0x1000),
      .align_mul = sizeof(uint32_t));
   nir_def *left = nir_f2f16(&b, nir_channel(&b, input, 0));
   nir_def *right = nir_f2f16(&b, nir_channel(&b, input, 1));
   nir_def *subtrahend = nir_f2f16(&b, nir_channel(&b, input, 2));
   nir_def *result = nir_ffma(&b, left, right, nir_fneg(&b, subtrahend));
   nir_store_global(&b, nir_f2f32(&b, result), nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary != NULL &&
                      part.info.binary_size > 4 && part.info.stats.instrs >= 8;
   if (!valid)
      fprintf(stderr, "G17P shaderFloat16 FMS lowering failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_abs_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P shaderFloat16 abs");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *input = nir_load_global(
      &b, 1, 32, nir_imm_int64(&b, 0x1000),
      .align_mul = sizeof(uint32_t));
   nir_def *half = nir_f2f16(&b, input);
   nir_def *absolute = nir_fabs(&b, half);
   /* Keep a native-half use alongside the widening use. This is the
    * frexp/modf shape CTS found: before the G17P fabs lowering, the half use
    * reached the packer as an unencodable fadd d16, abs(src16), -0. */
   nir_store_global(&b, absolute, nir_imm_int64(&b, 0x3000),
                    .align_mul = sizeof(uint16_t));
   nir_store_global(&b, nir_f2f32(&b, absolute), nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary != NULL &&
                      part.info.binary_size > 4 && part.info.stats.instrs >= 7;
   if (!valid)
      fprintf(stderr, "G17P shaderFloat16 abs lowering failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_min_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P shaderFloat16 minimum");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_global(&b, 2, 32, nir_imm_int64(&b, 0x1000),
                                    .align_mul = sizeof(uint32_t));
   nir_def *a = nir_f2f16(&b, nir_channel(&b, input, 0));
   nir_def *c = nir_f2f16(&b, nir_channel(&b, input, 1));
   nir_def *minimum = nir_fmin(&b, a, c);
   nir_store_global(&b, nir_f2f32(&b, minimum), nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary && part.info.stats.instrs >= 7;
   if (!valid)
      fprintf(stderr, "G17P shaderFloat16 minimum lowering failed\n");
   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_neg_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P shaderFloat16 negate");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *input = nir_load_global(
      &b, 1, 32, nir_imm_int64(&b, 0x1000),
      .align_mul = sizeof(uint32_t));
   nir_def *half = nir_f2f16(&b, input);
   nir_def *negated = nir_fneg(&b, half);
   nir_store_global(&b, nir_f2f32(&b, negated), nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary != NULL &&
                      part.info.binary_size > 4 && part.info.stats.instrs >= 5;
   if (!valid)
      fprintf(stderr, "G17P shaderFloat16 negate lowering failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_max_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P shaderFloat16 maximum");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_global(&b, 2, 32, nir_imm_int64(&b, 0x1000),
                                    .align_mul = sizeof(uint32_t));
   nir_def *a = nir_f2f16(&b, nir_channel(&b, input, 0));
   nir_def *c = nir_f2f16(&b, nir_channel(&b, input, 1));
   nir_def *maximum = nir_fmax(&b, a, c);
   nir_store_global(&b, nir_f2f32(&b, maximum), nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0}; key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary && part.info.stats.instrs >= 7;
   if (!valid) fprintf(stderr, "G17P shaderFloat16 maximum lowering failed\n");
   free(part.binary); ralloc_free(b.shader); return valid;
}

static bool
check_half_clamp_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P shaderFloat16 clamp");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_global(&b, 3, 32, nir_imm_int64(&b, 0x1000),
                                    .align_mul = sizeof(uint32_t));
   nir_def *x = nir_f2f16(&b, nir_channel(&b, input, 0));
   nir_def *low = nir_f2f16(&b, nir_channel(&b, input, 1));
   nir_def *high = nir_f2f16(&b, nir_channel(&b, input, 2));
   nir_def *clamped = nir_fclamp(&b, x, low, high);
   nir_store_global(&b, nir_f2f32(&b, clamped), nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0}; key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary && part.info.stats.instrs >= 8;
   if (!valid) fprintf(stderr, "G17P shaderFloat16 clamp lowering failed\n");
   free(part.binary); ralloc_free(b.shader); return valid;
}

static bool
check_half_divide_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P shaderFloat16 divide");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_global(&b, 2, 32, nir_imm_int64(&b, 0x1000),
                                    .align_mul = sizeof(uint32_t));
   nir_def *a = nir_f2f16(&b, nir_channel(&b, input, 0));
   nir_def *c = nir_f2f16(&b, nir_channel(&b, input, 1));
   nir_def *quotient = nir_fdiv(&b, a, c);
   nir_store_global(&b, nir_f2f32(&b, quotient), nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0}; key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary && part.info.stats.instrs >= 8;
   if (!valid) fprintf(stderr, "G17P shaderFloat16 divide lowering failed\n");
   free(part.binary); ralloc_free(b.shader); return valid;
}

static bool
check_half_compare_select_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "G17P shaderFloat16 comparison select");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_global(&b, 4, 32, nir_imm_int64(&b, 0x1000),
                                    .align_mul = sizeof(uint32_t));
   nir_def *a = nir_f2f16(&b, nir_channel(&b, input, 0));
   nir_def *c = nir_f2f16(&b, nir_channel(&b, input, 1));
   nir_def *if_true = nir_f2f16(&b, nir_channel(&b, input, 2));
   nir_def *if_false = nir_f2f16(&b, nir_channel(&b, input, 3));
   /* frexp compares against an absolute half before selecting a mantissa. */
   nir_def *selected =
      nir_bcsel(&b, nir_flt(&b, a, nir_fabs(&b, c)), if_true, if_false);
   nir_store_global(&b, nir_f2f32(&b, selected),
                    nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary && part.info.stats.instrs >= 8;
   if (!valid)
      fprintf(stderr, "G17P shaderFloat16 comparison select lowering failed\n");
   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_isnan_isinf_compute(void)
{
   for (unsigned bits = 0; bits <= UINT16_MAX; ++bits) {
      const uint16_t magnitude = bits & UINT16_C(0x7fff);
      const float value = _mesa_half_to_float(bits);
      if ((magnitude > UINT16_C(0x7c00)) != !!isnan(value) ||
          (magnitude == UINT16_C(0x7c00)) != !!isinf(value)) {
         fprintf(stderr,
                 "binary16 classification mismatch for bit pattern %#06x\n",
                 bits);
         return false;
      }
   }

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "G17P shaderFloat16 isnan/isinf");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *input = nir_load_global(&b, 1, 32,
                                    nir_imm_int64(&b, 0x1000),
                                    .align_mul = sizeof(uint32_t));
   nir_def *half = nir_f2f16(&b, input);

   const unsigned fp_math_ctrl = b.fp_math_ctrl;
   b.fp_math_ctrl |= nir_fp_preserve_nan | nir_fp_preserve_inf;
   nir_def *is_nan = nir_fneu(&b, half, half);
   nir_def *is_inf = nir_feq(&b, nir_fabs(&b, half),
                             nir_imm_floatN_t(&b, INFINITY, 16));
   b.fp_math_ctrl = fp_math_ctrl;

   nir_def *result = nir_iadd(&b, nir_b2i32(&b, is_nan),
                              nir_b2i32(&b, is_inf));
   nir_store_global(&b, result, nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir_g17p(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary != NULL &&
                      part.info.binary_size > 4 && part.info.stats.instrs >= 7;
   if (!valid)
      fprintf(stderr, "G17P shaderFloat16 isnan/isinf lowering failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_f32_abs_zero_compare_select_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "G17P f32 absolute zero comparison select");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_global(&b, 3, 32,
                                    nir_imm_int64(&b, 0x1000),
                                    .align_mul = sizeof(uint32_t));
   nir_def *absolute = nir_fabs(&b, nir_channel(&b, input, 0));
   nir_def *selected = nir_bcsel(
      &b, nir_flt(&b, nir_imm_float(&b, 0.0f), absolute),
      nir_channel(&b, input, 1), nir_channel(&b, input, 2));
   nir_store_global(&b, selected, nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary != NULL &&
                      part.info.binary_size > 4 && part.info.stats.instrs >= 7;
   if (!valid)
      fprintf(stderr, "G17P f32 absolute zero comparison select failed\n");

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_sqrt_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P shaderFloat16 square root");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_global(&b, 1, 32,
                                    nir_imm_int64(&b, 0x1000),
                                    .align_mul = sizeof(uint32_t));
   nir_def *half = nir_f2f16(&b, input);
   nir_def *root = nir_fsqrt(&b, half);
   nir_store_global(&b, nir_f2f32(&b, root),
                    nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary && part.info.stats.instrs >= 8;
   if (!valid)
      fprintf(stderr, "G17P shaderFloat16 square-root lowering failed\n");
   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_rsqrt_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "G17P shaderFloat16 reciprocal square root");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_global(&b, 1, 32,
                                    nir_imm_int64(&b, 0x1000),
                                    .align_mul = sizeof(uint32_t));
   nir_def *half = nir_f2f16(&b, input);
   nir_def *root = nir_frsq(&b, half);
   nir_store_global(&b, nir_f2f32(&b, root),
                    nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary && part.info.stats.instrs >= 7;
   if (!valid)
      fprintf(stderr, "G17P shaderFloat16 reciprocal-square-root lowering failed\n");
   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_exp2_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "G17P shaderFloat16 base-2 exponential");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_global(&b, 1, 32,
                                    nir_imm_int64(&b, 0x1000),
                                    .align_mul = sizeof(uint32_t));
   nir_def *half = nir_f2f16(&b, input);
   nir_def *result = nir_fexp2(&b, half);
   nir_store_global(&b, nir_f2f32(&b, result),
                    nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary && part.info.stats.instrs >= 7;
   if (!valid)
      fprintf(stderr, "G17P shaderFloat16 exp2 lowering failed\n");
   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_log2_compute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "G17P shaderFloat16 base-2 logarithm");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_global(&b, 1, 32,
                                    nir_imm_int64(&b, 0x1000),
                                    .align_mul = sizeof(uint32_t));
   nir_def *half = nir_f2f16(&b, input);
   nir_def *result = nir_flog2(&b, half);
   nir_store_global(&b, nir_f2f32(&b, result),
                    nir_imm_int64(&b, 0x2000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary && part.info.stats.instrs >= 7;
   if (!valid)
      fprintf(stderr, "G17P shaderFloat16 log2 lowering failed\n");
   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_half_floor_compute(void)
{
   nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &agx_nir_options,"G17P shaderFloat16 floor");
   b.shader->info.workgroup_size[0]=1; b.shader->info.workgroup_size[1]=1;
   b.shader->info.workgroup_size[2]=1;
   nir_def *input=nir_load_global(&b,1,32,nir_imm_int64(&b,0x1000),
                                   .align_mul=sizeof(uint32_t));
   /* modf lowers through floor(abs(x)) for its magnitude path. */
   nir_def *result=nir_ffloor(&b,nir_fabs(&b,nir_f2f16(&b,input)));
   nir_store_global(&b,nir_f2f32(&b,result),nir_imm_int64(&b,0x2000),
                    .align_mul=sizeof(uint32_t));
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key={0}; key.dev.encoding=AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part={0};
   const bool compiled=agx_compile_shader_nir(b.shader,&key,&part);
   const bool valid=compiled&&part.binary&&part.info.stats.instrs>=7;
   if(!valid) fprintf(stderr,"G17P shaderFloat16 floor lowering failed\n");
   free(part.binary); ralloc_free(b.shader); return valid;
}

static bool
check_half_ceil_compute(void)
{
   nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
      &agx_nir_options,"G17P shaderFloat16 ceil");
   b.shader->info.workgroup_size[0]=1; b.shader->info.workgroup_size[1]=1;
   b.shader->info.workgroup_size[2]=1;
   nir_def *input=nir_load_global(&b,1,32,nir_imm_int64(&b,0x1000),
                                   .align_mul=sizeof(uint32_t));
   nir_def *result=nir_fceil(&b,nir_f2f16(&b,input));
   nir_store_global(&b,nir_f2f32(&b,result),nir_imm_int64(&b,0x2000),
                    .align_mul=sizeof(uint32_t));
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key={0}; key.dev.encoding=AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part={0};
   const bool compiled=agx_compile_shader_nir(b.shader,&key,&part);
   const bool valid=compiled&&part.binary&&part.info.stats.instrs>=7;
   if(!valid) fprintf(stderr,"G17P shaderFloat16 ceil lowering failed\n");
   free(part.binary); ralloc_free(b.shader); return valid;
}

/*
 * The acceleration-structure builder's radix-sort lookback loop breaks out
 * from inside nested ifs. That shape used to refuse -- "no grounded AGX3
 * encoding for break" -- because the packer only knew CNDLT, which spends
 * exactly the loop's own two exec-mask levels. CNDCF carries the level count,
 * so the break now encodes at any depth.
 *
 * Each condition reads a distinct device address, so nothing can be folded or
 * proven, and each level stores before the next if opens, so the ifs cannot
 * collapse into one.
 */
static bool
check_deep_loop_break_compute(unsigned enclosing_ifs)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};
   nir_if *opened[8];

   if (enclosing_ifs > ARRAY_SIZE(opened))
      return false;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P break under %u ifs",
      enclosing_ifs);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_loop *loop = nir_push_loop(&b);
   {
      /* The loop's own exit test: a depth-1 break, which stays on CNDLT. */
      nir_def *guard = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x1000),
                                       .align_mul = sizeof(uint32_t));
      nir_break_if(&b, nir_ieq_imm(&b, guard, 0));

      for (unsigned i = 0; i < enclosing_ifs; ++i) {
         nir_def *v =
            nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0x1008 + 8 * i),
                            .align_mul = sizeof(uint32_t));
         opened[i] = nir_push_if(&b, nir_ine_imm(&b, v, 0));
         nir_store_global(&b, nir_imm_int(&b, i + 1),
                          nir_imm_int64(&b, 0x2000 + 8 * i),
                          .align_mul = sizeof(uint32_t));
      }
      nir_jump(&b, nir_jump_break);
      for (unsigned i = enclosing_ifs; i-- > 0;)
         nir_pop_if(&b, opened[i]);
   }
   nir_pop_loop(&b, loop);

   nir_store_global(&b, nir_imm_int(&b, 0xfeed), nir_imm_int64(&b, 0x3000),
                    .align_mul = sizeof(uint32_t));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   struct agx_shader_part part = {0};
   if (!agx_compile_shader_nir(b.shader, &key, &part)) {
      fprintf(stderr, "G17P loop break under %u ifs was rejected\n",
              enclosing_ifs);
      ralloc_free(b.shader);
      return false;
   }

   bool valid =
      part.binary != NULL && part.info.binary_size > sizeof(end) &&
      memcmp((const uint8_t *)part.binary + part.info.binary_size - sizeof(end),
             end, sizeof(end)) == 0;

   /* Folding an enclosing if into the break subtracts at most one level, so
    * two or more enclosing ifs cannot land back on CNDLT: the wide form has
    * to be there, with a count somewhere in [3, ifs + 2]. */
   if (valid && enclosing_ifs >= 2) {
      bool found = false;
      for (unsigned count = 3; count <= enclosing_ifs + 2 && !found; ++count) {
         for (unsigned invert = 0; invert < 2 && !found; ++invert) {
            uint8_t want[AGX3_CF_MAX_LENGTH];
            uint8_t len = 0;
            if (!agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3B, 1, invert != 0, count,
                               want, &len))
               continue;
            for (unsigned at = 0; at + len <= part.info.binary_size; at += 2) {
               if (!memcmp((const uint8_t *)part.binary + at, want, len)) {
                  found = true;
                  break;
               }
            }
         }
      }
      if (!found) {
         fprintf(stderr,
                 "G17P loop break under %u ifs emitted no CNDCF\n",
                 enclosing_ifs);
         valid = false;
      }
   }

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_uniform_alu_policy(void)
{
   const struct {
      enum agx_isa_encoding encoding;
      bool alu;
   } cases[] = {
      {AGX_ISA_ENCODING_AGX2, true},
      {AGX_ISA_ENCODING_AGX3B, false},
      {AGX_ISA_ENCODING_AGX3C, false},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      const enum agx_isa_encoding e = cases[i].encoding;

      if (agx_instr_accepts_uniform(e, AGX_OPCODE_IADD, 0, 4, AGX_SIZE_32) !=
             cases[i].alu ||
          agx_instr_accepts_uniform(e, AGX_OPCODE_IMAD, 1, 4, AGX_SIZE_32) !=
             cases[i].alu) {
         fprintf(stderr, "AGX ALU uniform policy wrong for encoding %u\n", e);
         return false;
      }

      /* 64-bit ALU never takes a uniform on any encoding. */
      if (agx_instr_accepts_uniform(e, AGX_OPCODE_IADD, 0, 4, AGX_SIZE_64)) {
         fprintf(stderr, "64-bit ALU uniform accepted for encoding %u\n", e);
         return false;
      }

      /* Memory bases keep their uniform, on every encoding: the AGX3 packer
       * depends on it (agx3_pack_binary.c:961 requires a uniform base). */
      if (!agx_instr_accepts_uniform(e, AGX_OPCODE_DEVICE_LOAD, 0, 4,
                                     AGX_SIZE_64) ||
          !agx_instr_accepts_uniform(e, AGX_OPCODE_DEVICE_STORE, 1, 4,
                                     AGX_SIZE_64)) {
         fprintf(stderr, "memory-base uniform refused for encoding %u\n", e);
         return false;
      }
   }

   return true;
}

int
main(int argc, char **argv)
{
   if (argc == 2 && !strcmp(argv[1], "--subgroup-multiply-only")) {
      const bool valid = check_subgroup_multiply_compute();
      fprintf(stderr, "G17P subgroup multiply compute: %s\n",
               valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }

   if (argc == 2 && !strcmp(argv[1], "--half-abs-only")) {
      const bool valid = check_half_abs_compute();
      fprintf(stderr, "G17P shaderFloat16 abs: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }

   if (argc == 2 && !strcmp(argv[1], "--half-neg-only")) {
      const bool valid = check_half_neg_compute();
      fprintf(stderr, "G17P shaderFloat16 negate: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }

   if (argc == 2 && !strcmp(argv[1], "--half-min-only")) {
      const bool valid = check_half_min_compute();
      fprintf(stderr, "G17P shaderFloat16 minimum: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }

   if (argc == 2 && !strcmp(argv[1], "--half-max-only")) {
      const bool valid = check_half_max_compute();
      fprintf(stderr, "G17P shaderFloat16 maximum: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }

   if (argc == 2 && !strcmp(argv[1], "--half-clamp-only")) {
      const bool valid = check_half_clamp_compute();
      fprintf(stderr, "G17P shaderFloat16 clamp: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }

   if (argc == 2 && !strcmp(argv[1], "--half-divide-only")) {
      const bool valid = check_half_divide_compute();
      fprintf(stderr, "G17P shaderFloat16 divide: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--half-fma-only")) {
      const bool valid = check_half_fma_compute();
      fprintf(stderr, "G17P shaderFloat16 FMA: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--pk-immediate-fma-only")) {
      const bool valid = check_pk_immediate_fma_sequence();
      fprintf(stderr, "G17P CTS pk_immediate FMA sequence: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--half-fms-only")) {
      const bool valid = check_half_fms_compute();
      fprintf(stderr, "G17P shaderFloat16 FMS: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }

   if (argc == 2 && !strcmp(argv[1], "--half-add-only")) {
      const bool valid = check_half_add_compute();
      fprintf(stderr, "G17P shaderFloat16 add: %s\n",
               valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }

   if (argc == 2 && !strcmp(argv[1], "--device-ballot-only")) {
      const bool valid =
         check_device_ballot_compute(AGX_ISA_ENCODING_AGX3B) &&
         check_device_ballot_compute(AGX_ISA_ENCODING_AGX3C);
      fprintf(stderr, "AGX3 device-load ballot compute: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--subgroup-vote-only")) {
      const bool valid = check_g17p_subgroup_vote_compute() &&
                         check_g17p_signed_immediate_ballot_compiles();
      fprintf(stderr, "G17P subgroup vote: %s\n", valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--subgroup-quad-only")) {
      const bool valid = check_g17p_subgroup_quad_compute(true) &&
                         check_g17p_subgroup_quad_compute(false) &&
                         check_g17p_quad_broadcast_dynamic_lane(16) &&
                         check_g17p_quad_broadcast_dynamic_lane(32);
      fprintf(stderr, "G17P subgroup quad: %s\n", valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--device-shuffle-only")) {
      const bool valid =
         check_device_shuffle_compute(AGX_ISA_ENCODING_AGX3B) &&
         check_device_shuffle_compute(AGX_ISA_ENCODING_AGX3C);
      fprintf(stderr, "AGX3 device-load shuffle compute: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--elect-only")) {
      const bool valid =
         check_elect_broadcast_first_compute(AGX_ISA_ENCODING_AGX2) &&
         check_elect_broadcast_first_compute(AGX_ISA_ENCODING_AGX3B);
      fprintf(stderr, "elect/broadcast-first lowering: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }

   if (argc == 2 && !strcmp(argv[1], "--half-compare-only")) {
      const bool valid = check_half_compare_select_compute();
      fprintf(stderr, "G17P shaderFloat16 comparison select: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--half-classify-only")) {
      const bool valid = check_half_isnan_isinf_compute();
      fprintf(stderr, "G17P shaderFloat16 isnan/isinf: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--f32-abs-select-only")) {
      const bool valid = check_f32_abs_zero_compare_select_compute();
      fprintf(stderr, "G17P f32 absolute zero comparison select: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--half-sqrt-only")) {
      const bool valid = check_half_sqrt_compute();
      fprintf(stderr, "G17P shaderFloat16 square root: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--half-rsqrt-only")) {
      const bool valid = check_half_rsqrt_compute();
      fprintf(stderr, "G17P shaderFloat16 reciprocal square root: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--half-exp2-only")) {
      const bool valid = check_half_exp2_compute();
      fprintf(stderr, "G17P shaderFloat16 exp2: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--half-log2-only")) {
      const bool valid = check_half_log2_compute();
      fprintf(stderr, "G17P shaderFloat16 log2: %s\n",
              valid ? "PASS" : "FAIL");
      return valid ? 0 : 1;
   }
   if(argc==2&&!strcmp(argv[1],"--half-floor-only")) {
      const bool valid=check_half_floor_compute();
      fprintf(stderr,"G17P shaderFloat16 floor: %s\n",valid?"PASS":"FAIL");
      return valid?0:1;
   }
   if(argc==2&&!strcmp(argv[1],"--half-ceil-only")) {
      const bool valid=check_half_ceil_compute();
      fprintf(stderr,"G17P shaderFloat16 ceil: %s\n",valid?"PASS":"FAIL");
      return valid?0:1;
   }
   const enum agx_isa_encoding encodings[] = {
      AGX_ISA_ENCODING_AGX3B,
      AGX_ISA_ENCODING_AGX3C,
   };

   /* glsl_array_type() in the constant-array case needs the type singleton. */
   glsl_type_singleton_init_or_ref();

   if (argc == 2 && strcmp(argv[1], "--device-atomics-only") == 0) {
      for (unsigned i = 0; i < ARRAY_SIZE(encodings); ++i) {
         if (!check_device_atomic_compute(encodings[i]))
            return 1;
         if (!check_uniform_atomic_tcs(encodings[i]))
            return 1;
      }
      return 0;
   }
   if (argc != 1) {
      fprintf(stderr, "usage: %s [--device-atomics-only]\n", argv[0]);
      return 1;
   }

   if (!check_uniform_alu_policy())
      return 1;
   for (unsigned ifs = 0; ifs <= 6; ++ifs) {
      if (!check_deep_loop_break_compute(ifs))
         return 1;
   }
   if (!check_agx2_legacy_printf_address_accepted())
      return 1;
   if (!check_mixed_fma_neg_compute())
      return 1;
   if (!check_g17p_saturating_iadd_immediate_compute())
      return 1;
   if (!check_subgroup_multiply_compute())
      return 1;
   if (!check_half_add_compute())
      return 1;
   if (!check_half_fma_compute())
      return 1;
   if (!check_half_fms_compute())
      return 1;
   if (!check_half_abs_compute())
      return 1;
   if (!check_half_neg_compute())
      return 1;
   if (!check_half_min_compute())
      return 1;
   if (!check_half_max_compute())
      return 1;
   if (!check_half_clamp_compute())
      return 1;
   if (!check_half_divide_compute())
      return 1;
   if (!check_half_compare_select_compute())
      return 1;
   if (!check_half_isnan_isinf_compute())
      return 1;
   if (!check_f32_abs_zero_compare_select_compute())
      return 1;
   if (!check_half_sqrt_compute())
      return 1;
   if (!check_half_rsqrt_compute())
      return 1;
   if (!check_half_exp2_compute())
      return 1;
   if (!check_half_log2_compute())
      return 1;
   if (!check_half_floor_compute())
      return 1;
   if (!check_half_ceil_compute())
      return 1;
   if (!check_elect_broadcast_first_compute(AGX_ISA_ENCODING_AGX2))
      return 1;
   if (!check_active_count_lowering(false) ||
       !check_active_count_lowering(true))
      return 1;
   if (!check_g17p_subgroup_vote_compute() ||
       !check_g17p_signed_immediate_ballot_compiles())
      return 1;
   if (!check_g17p_subgroup_quad_compute(true) ||
       !check_g17p_subgroup_quad_compute(false))
      return 1;
   if (!check_g17p_quad_broadcast_dynamic_lane(16) ||
       !check_g17p_quad_broadcast_dynamic_lane(32))
      return 1;

   if (!check_g17p_shared_byte_vectors())
      return 1;

   for (unsigned i = 0; i < sizeof(encodings) / sizeof(encodings[0]); ++i) {
      if (!check_empty_compute(encodings[i]))
         return 1;
      if (!check_unlowered_printf_address_rejected(encodings[i]))
         return 1;
      if (!check_shared_store_compute(encodings[i]))
         return 1;
      if (!check_shared_rmw_compute(encodings[i]))
         return 1;
      if (!check_local_id_compute(encodings[i]))
         return 1;
      if (!check_subgroup_id_compute(encodings[i]))
         return 1;
      if (!check_device_ballot_compute(encodings[i]))
         return 1;
      if (!check_device_shuffle_compute(encodings[i]))
         return 1;
      if (encodings[i] == AGX_ISA_ENCODING_AGX3B &&
          !check_elect_broadcast_first_compute(encodings[i]))
         return 1;
      if (!check_device_rmw_compute(encodings[i]))
         return 1;
      if (!check_device_atomic_compute(encodings[i]))
         return 1;
      if (!check_uniform_atomic_tcs(encodings[i]))
         return 1;
      if (!check_shared_atomic_compute(encodings[i]))
         return 1;
      if (!check_constant_data_compute(encodings[i], false) ||
          !check_constant_data_compute(encodings[i], true))
         return 1;
      if (!check_preamble_compute(encodings[i], true, 0))
         return 1;
      if (encodings[i] == AGX_ISA_ENCODING_AGX3B &&
          (!check_preamble_compute(encodings[i], false, 0) ||
           !check_preamble_compute(AGX_ISA_ENCODING_AGX2, false, 0) ||
           !check_preamble_compute(AGX_ISA_ENCODING_AGX2, true, 0) ||
           !check_preamble_compute(encodings[i], false, 32) ||
           !check_preamble_compute(encodings[i], true, 32) ||
           !check_preamble_compute(AGX_ISA_ENCODING_AGX2, false, 32) ||
           !check_preamble_compute(AGX_ISA_ENCODING_AGX2, true, 32)))
         return 1;
      if (encodings[i] == AGX_ISA_ENCODING_AGX3B) {
         for (unsigned input = PREAMBLE_UNIFORM_INPUT; input <= PREAMBLE_CONSTANT_ONLY;
              ++input) {
            if (input == PREAMBLE_ENTRY_GPR)
               continue;
            if (!check_preamble_entry_gpr(encodings[i], false, input) ||
                !check_preamble_entry_gpr(AGX_ISA_ENCODING_AGX2, false, input))
               return 1;
         }
      } else if (!check_preamble_entry_gpr(encodings[i], true,
                                           PREAMBLE_ENTRY_GPR)) {
         return 1;
      }
      if (!check_constant_array_ordering(encodings[i], true) ||
          !check_constant_array_ordering(encodings[i], false))
         return 1;

      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "unlowered ray query");
      b.shader->info.io_lowered = true;
      b.shader->info.ray_queries = 1;

      struct agx_shader_key key = {0};
      key.dev.encoding = encodings[i];
      struct agx_shader_part part;
      memset(&part, 0xa5, sizeof(part));

      if (agx_compile_shader_nir(b.shader, &key, &part)) {
         fprintf(stderr, "AGX3 unlowered ray query was accepted\n");
         ralloc_free(b.shader);
         return 1;
      }

      const uint8_t zero[sizeof(part)] = {0};
      if (memcmp(&part, zero, sizeof(part)) != 0) {
         fprintf(stderr, "AGX3 rejection did not clear compiler output\n");
         ralloc_free(b.shader);
         return 1;
      }

      ralloc_free(b.shader);
   }

   glsl_type_singleton_decref();

   puts("agx3_compile: compute SR/shared-memory/constant-data paths emitted; "
        "AGX3B scratch fallback bounded; unlowered ray queries and printf "
        "addresses rejected");
   return 0;
}

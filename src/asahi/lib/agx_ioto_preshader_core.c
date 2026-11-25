/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ioto_preshader_core.h"

#include "agx_ioto_core.h"

#include <string.h>

enum agx_ioto_preshader_encoding {
   AGX_IOTO_PRESHADER_ENCODING_GPU0,
   AGX_IOTO_PRESHADER_ENCODING_GPU1,
   AGX_IOTO_PRESHADER_ENCODING_UNSUPPORTED,
};

static enum agx_ioto_preshader_encoding
agx_ioto_preshader_target_encoding(enum agx_chip chip)
{
   /* This is an ISA-generation property, not a runtime record property.  Do
    * not rediscover it through two decoder table walks for every graphics
    * launcher branch.  The exhaustive preshader tests still validate every
    * listed chip's result through the independent record decoder. */
   switch (agx_chip_ioto_generation(chip)) {
   case AGX_IOTO_GENERATION_GPU0:
      return AGX_IOTO_PRESHADER_ENCODING_GPU0;
   case AGX_IOTO_GENERATION_GPU1:
      return AGX_IOTO_PRESHADER_ENCODING_GPU1;
   default:
      return AGX_IOTO_PRESHADER_ENCODING_UNSUPPORTED;
   }
}

bool
agx_ioto_preshader_bind_range(const void *binary, size_t binary_size,
                              uint64_t binary_va, size_t gpu_size,
                              size_t offset, size_t size,
                              struct agx_ioto_preshader_desc *desc)
{
   const uint64_t va_limit = UINT64_C(1) << 48;
   if (!binary || !desc || !size || ((offset | size | binary_va) & 1) ||
       offset > binary_size || size > binary_size - offset ||
       offset > gpu_size || size > gpu_size - offset ||
       binary_va >= va_limit || offset >= va_limit - binary_va)
      return false;

   const uint64_t address = binary_va + offset;
   if (size > va_limit - address)
      return false;

   desc->code = (const uint8_t *)binary + offset;
   desc->code_size = size;
   desc->external_va = address;
   return true;
}

static void
agx_ioto_preshader_set_bit(uint8_t *bytes, unsigned bit, bool value)
{
   const uint8_t mask = (uint8_t)(1u << (bit & 7));

   if (value)
      bytes[bit >> 3] |= mask;
   else
      bytes[bit >> 3] &= (uint8_t)~mask;
}

bool
agx_ioto_preshader_pack_branch(enum agx_chip chip, uint64_t target_va,
                               bool link,
                               uint8_t out[AGX_IOTO_PRESHADER_BRANCH_SIZE])
{
   const enum agx_ioto_preshader_encoding encoding =
      agx_ioto_preshader_target_encoding(chip);
   if (encoding == AGX_IOTO_PRESHADER_ENCODING_UNSUPPORTED)
      return false;

   if (!out || target_va >= (UINT64_C(1) << 48) || (target_va & 1) != 0)
      return false;

   uint8_t staged[AGX_IOTO_PRESHADER_BRANCH_SIZE] = {0};
   const uint64_t half_target = target_va >> 1;
   enum agx_ioto_kind kind;

   if (encoding == AGX_IOTO_PRESHADER_ENCODING_GPU0) {
      staged[0] = link ? 0x8f : 0x0f;
      /* Zero-valued f12/f13/f14 pass through the classic [1, 2, 0] LUT. */
      staged[2] = 0x54;
      /* GPU0's complete 47-bit halfword target is contiguous at bits25..71.
       * Deposit the six bytes directly: compact graphics launchers build one
       * branch per draw, and the former bit loop plus decoder validation was
       * a measurable hot path.  The preshader test exhausts the input bits
       * and still decodes every result through the independent oracle. */
      const uint64_t shifted = half_target << 1;
      for (unsigned byte = 0; byte < 6; ++byte)
         staged[3 + byte] = shifted >> (8 * byte);

      memcpy(out, staged, sizeof(staged));
      return true;
   } else {
      static const uint8_t target_bits[47] = {
         7,  12, 13, 14, 17, 19, 21, 22, 23, 8,  9,  10, 11, 28, 32, 33,
         34, 35, 36, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63,
         64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
      };

      staged[0] = link ? 0x7e : 0x5e;
      staged[2] = 0x11;
      staged[3] = 0x0e;
      kind = link ? AGX_IOTO_KIND_BL_EP1 : AGX_IOTO_KIND_B_EP1;

      for (unsigned source = 0; source < 47; ++source)
         agx_ioto_preshader_set_bit(staged, target_bits[source],
                                    (half_target >> source) & 1);
   }

   struct agx_ioto_record record;
   size_t consumed = 0;
   if (!agx_ioto_validate_kind(chip, kind, staged, sizeof(staged), &record,
                               &consumed) ||
       consumed != sizeof(staged) || record.length != sizeof(staged))
      return false;

   memcpy(out, record.bytes, sizeof(staged));
   return true;
}

static bool
agx_ioto_preshader_append_absent(struct agx_ioto_preshader_output *output)
{
   static const uint8_t absent[AGX_IOTO_PRESHADER_ABSENT_SIZE] = {
      0x0e,
      0x00,
      0x00,
      0x00,
   };

   output->mode = AGX_IOTO_PRESHADER_MODE_ABSENT;
   output->length = sizeof(absent);
   output->branch_offset = AGX_IOTO_PRESHADER_NO_BRANCH;
   memcpy(output->bytes, absent, sizeof(absent));
   return true;
}

bool
agx_ioto_preshader_emit(enum agx_chip chip,
                        const struct agx_ioto_preshader_desc *desc,
                        struct agx_ioto_preshader_output *output)
{
   if (!desc || !output ||
       agx_ioto_preshader_target_encoding(chip) ==
          AGX_IOTO_PRESHADER_ENCODING_UNSUPPORTED)
      return false;

   struct agx_ioto_preshader_output staged = {
      .branch_offset = AGX_IOTO_PRESHADER_NO_BRANCH,
   };

   if (!desc->force_external &&
       desc->code_size <= AGX_IOTO_PRESHADER_INLINE_LIMIT) {
      if ((desc->code_size != 0 && !desc->code) || (desc->code_size & 1) != 0)
         return false;

      const uint8_t *code = desc->code;
      size_t trimmed = desc->code_size;
      while (trimmed >= 2 && code[trimmed - 2] == 0x06 &&
             code[trimmed - 1] == 0x00)
         trimmed -= 2;

      if (trimmed == 0) {
         agx_ioto_preshader_append_absent(&staged);
      } else {
         staged.mode = AGX_IOTO_PRESHADER_MODE_INLINE;
         staged.length = trimmed;
         memcpy(staged.bytes, code, trimmed);
      }

      *output = staged;
      return true;
   }

   if (desc->external_va == AGX_IOTO_PRESHADER_NO_VA) {
      agx_ioto_preshader_append_absent(&staged);
      *output = staged;
      return true;
   }

   staged.mode = AGX_IOTO_PRESHADER_MODE_EXTERNAL;
   if (desc->request_prelude && !desc->suppress_prelude) {
      const bool gpu1 = agx_ioto_preshader_target_encoding(chip) ==
                        AGX_IOTO_PRESHADER_ENCODING_GPU1;
      staged.bytes[0] = gpu1 ? 0x04 : 0x0c;
      staged.bytes[1] = 0x10;

      struct agx_ioto_record record;
      size_t consumed = 0;
      if (!agx_ioto_validate_kind(chip, AGX_IOTO_KIND_LDIMM_EP0, staged.bytes,
                                  AGX_IOTO_PRESHADER_PRELUDE_SIZE, &record,
                                  &consumed) ||
          consumed != AGX_IOTO_PRESHADER_PRELUDE_SIZE)
         return false;

      staged.length = AGX_IOTO_PRESHADER_PRELUDE_SIZE;
   }

   staged.branch_offset = staged.length;
   if (!agx_ioto_preshader_pack_branch(chip, desc->external_va, false,
                                       staged.bytes + staged.length))
      return false;

   staged.length += AGX_IOTO_PRESHADER_BRANCH_SIZE;
   *output = staged;
   return true;
}

bool
agx_ioto_constant_program_emit(
   enum agx_chip chip, const struct agx_ioto_constant_program *program,
   uint64_t suffix_va, struct agx_ioto_preshader_output *output)
{
   const uint64_t address_limit = UINT64_C(1) << 48;
   if (!program || !output || !suffix_va || suffix_va >= address_limit ||
       (suffix_va & 1) || (program->code_size & 1))
      return false;

   struct agx_ioto_preshader_desc desc = {
      .code = program->code,
      .code_size = program->code_size,
      .external_va = AGX_IOTO_PRESHADER_NO_VA,
      .force_external = program->force_external,
      .request_prelude = program->request_prelude,
      .suppress_prelude = program->suppress_prelude,
   };

   if (program->code_size &&
       (program->force_external ||
        program->code_size > AGX_IOTO_PRESHADER_INLINE_LIMIT)) {
      if (!program->code_va || program->code_va >= address_limit ||
          (program->code_va & 1) ||
          program->code_size > address_limit - program->code_va)
         return false;

      const unsigned branch_offset =
         program->request_prelude && !program->suppress_prelude
            ? AGX_IOTO_PRESHADER_PRELUDE_SIZE : 0;
      if (branch_offset + AGX_IOTO_PRESHADER_BRANCH_SIZE >
          address_limit - suffix_va)
         return false;

      if (program->absolute_branch) {
         if (!agx_chip_native_preamble(chip) && !agx_chip_g17_m5(chip))
            return false;
         desc.external_va = program->code_va;
      } else {
         const int64_t displacement =
            (int64_t)program->code_va - (int64_t)(suffix_va + branch_offset);
         if (displacement < -(INT64_C(1) << 47) ||
             displacement >= (INT64_C(1) << 47))
            return false;

         desc.external_va = (uint64_t)displacement & (address_limit - 1);
      }
   }

   struct agx_ioto_preshader_output staged;
   if (!agx_ioto_preshader_emit(chip, &desc, &staged) ||
       staged.length > address_limit - suffix_va)
      return false;

   if (staged.mode == AGX_IOTO_PRESHADER_MODE_EXTERNAL &&
       program->absolute_branch) {
      /* On G17P the absolute code_va sits at object bits 1..48 with object bit
       * 0 set. The B instruction carries that selector at instruction bit 24
       * (f11); target bits occupy 25..71. Relocating the program changes the
       * code VA without subtracting a B PC. */
      if (!agx_chip_g17_m5(chip))
         staged.bytes[staged.branch_offset + 3] |= 1;
      else
         /* Captured G17S BLAS loader+512: B_EP1 f11 is bit61. */
         staged.bytes[staged.branch_offset + 7] |= 0x20;
   }

   *output = staged;
   return true;
}

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ioto_state_core.h"

#include "agx_bytes.h"
#include "agx_ioto_core.h"

#include <stdlib.h>
#include <string.h>

enum agx_ioto_state_encoding {
   AGX_IOTO_STATE_ENCODING_GPU0,
   AGX_IOTO_STATE_ENCODING_GPU1,
   AGX_IOTO_STATE_ENCODING_UNSUPPORTED,
};

static const uint8_t agx_ioto_e3[3] = {1, 2, 0};
static const uint8_t agx_ioto_e4[4] = {2, 0, 1, 3};

#define FM(field) (UINT32_C(1) << (field))

static enum agx_ioto_state_encoding
agx_ioto_state_target_encoding(enum agx_chip chip)
{
   switch (agx_chip_ioto_generation(chip)) {
   case AGX_IOTO_GENERATION_GPU0:
      return AGX_IOTO_STATE_ENCODING_GPU0;
   case AGX_IOTO_GENERATION_GPU1:
      return AGX_IOTO_STATE_ENCODING_GPU1;
   default:
      return AGX_IOTO_STATE_ENCODING_UNSUPPORTED;
   }
}

static bool
agx_ioto_state_fields_zero_except(const struct agx_ioto_state *state,
                                  uint32_t allowed)
{
   for (unsigned i = 0; i < AGX_IOTO_STATE_FIELD_COUNT; ++i) {
      if ((allowed & FM(i)) == 0 && state->f[i] != 0)
         return false;
   }
   return true;
}

static bool
agx_ioto_state_fields_at_most(const struct agx_ioto_state *state,
                              unsigned first, unsigned last, uint64_t maximum)
{
   for (unsigned i = first; i <= last; ++i) {
      if (state->f[i] > maximum)
         return false;
   }
   return true;
}

static bool
agx_ioto_state_is_zero(const struct agx_ioto_state *state)
{
   return agx_ioto_state_fields_zero_except(state, 0);
}

static bool
agx_ioto_state_validate_speclm(enum agx_ioto_state_encoding encoding,
                               const struct agx_ioto_state *state)
{
   if (encoding == AGX_IOTO_STATE_ENCODING_GPU0) {
      const uint32_t allowed =
         FM(2) | FM(3) | FM(4) | FM(5) | FM(6) | FM(7) | FM(10) |
         FM(11) | FM(12) | FM(18) | FM(21) | FM(22) | FM(24) | FM(25) |
         FM(26);
      return agx_ioto_state_fields_zero_except(state, allowed) &&
             agx_ioto_state_fields_at_most(state, 2, 7, 1) &&
             agx_ioto_state_fields_at_most(state, 10, 12, 2) &&
             state->f[18] <= UINT64_C(0xffff) &&
             (state->f[18] & UINT64_C(0x3f)) == 0 && state->f[21] <= 3 &&
             state->f[22] <= UINT64_C(0xf8) &&
             (state->f[22] & UINT64_C(7)) == 0 && state->f[24] <= 1 &&
             state->f[25] <= 1 && state->f[26] == 0;
   }

   if (encoding == AGX_IOTO_STATE_ENCODING_GPU1) {
      const uint32_t allowed =
         FM(0) | FM(1) | FM(2) | FM(3) | FM(4) | FM(5) | FM(6) | FM(7) |
         FM(8) | FM(9) | FM(18) | FM(21) | FM(22) | FM(24) | FM(25) |
         FM(26);
      return agx_ioto_state_fields_zero_except(state, allowed) &&
             state->f[0] == 3 && state->f[1] == 165 &&
             agx_ioto_state_fields_at_most(state, 2, 9, 1) &&
             state->f[18] <= UINT64_C(0xffff) && state->f[21] <= 3 &&
             state->f[22] <= UINT64_C(0xff) && state->f[24] <= 1 &&
             state->f[25] <= 1 && state->f[26] <= 1;
   }

   return false;
}

static bool
agx_ioto_state_validate_ldshdr(enum agx_ioto_state_encoding encoding,
                               const struct agx_ioto_state *state)
{
   if (encoding == AGX_IOTO_STATE_ENCODING_GPU0) {
      const uint32_t allowed =
         FM(0) | FM(1) | FM(7) | FM(8) | FM(9) | FM(10) | FM(11) |
         FM(12) | FM(15) | FM(16) | FM(17) | FM(23) | FM(24);
      return agx_ioto_state_fields_zero_except(state, allowed) &&
             state->f[0] <= UINT64_C(0xffffffffffff) &&
             (state->f[0] & UINT64_C(0x3f)) == 0 && state->f[1] <= 3 &&
             agx_ioto_state_fields_at_most(state, 7, 12, 1) &&
             agx_ioto_state_fields_at_most(state, 15, 17, 2) &&
             state->f[23] == 1 && state->f[24] == 0;
   }

   if (encoding == AGX_IOTO_STATE_ENCODING_GPU1) {
      const uint32_t allowed =
         FM(0) | FM(1) | FM(3) | FM(5) | FM(6) | FM(7) | FM(8) | FM(9) |
         FM(10) | FM(11) | FM(12) | FM(13) | FM(14) | FM(23) | FM(24);
      return agx_ioto_state_fields_zero_except(state, allowed) &&
             state->f[0] <= UINT64_C(0xffffffffffff) &&
             (state->f[0] & UINT64_C(0x3f)) == 0 && state->f[1] <= 3 &&
             state->f[3] <= 1 && state->f[5] == 3 && state->f[6] == 107 &&
             agx_ioto_state_fields_at_most(state, 7, 14, 1) &&
             state->f[23] <= 1 && state->f[24] <= 1;
   }

   return false;
}

static bool
agx_ioto_state_validate_spectpr(enum agx_ioto_state_encoding encoding,
                                const struct agx_ioto_state *state)
{
   if (encoding == AGX_IOTO_STATE_ENCODING_GPU0) {
      const uint32_t allowed =
         FM(2) | FM(3) | FM(4) | FM(5) | FM(6) | FM(7) | FM(10) |
         FM(11) | FM(12) | FM(13) | FM(14) | FM(20) | FM(21) | FM(22) |
         FM(23);
      return agx_ioto_state_fields_zero_except(state, allowed) &&
             agx_ioto_state_fields_at_most(state, 2, 7, 1) &&
             state->f[10] <= UINT64_C(0x7f) && state->f[11] <= 1 &&
             agx_ioto_state_fields_at_most(state, 12, 14, 2) &&
             state->f[20] <= UINT64_C(0xffff) &&
             state->f[21] <= UINT64_C(0xffff) &&
             state->f[22] <= UINT64_C(0xffe) &&
             (state->f[22] & UINT64_C(1)) == 0 && state->f[23] == 0;
   }

   if (encoding == AGX_IOTO_STATE_ENCODING_GPU1) {
      const uint32_t allowed =
         FM(0) | FM(1) | FM(2) | FM(3) | FM(4) | FM(5) | FM(6) | FM(7) |
         FM(8) | FM(9) | FM(10) | FM(11) | FM(20) | FM(21) | FM(22) |
         FM(23);
      return agx_ioto_state_fields_zero_except(state, allowed) &&
             state->f[0] == 3 && state->f[1] == 166 &&
             agx_ioto_state_fields_at_most(state, 2, 9, 1) &&
             state->f[10] <= UINT64_C(0x7f) && state->f[11] <= 1 &&
             state->f[20] <= UINT64_C(0xffff) &&
             state->f[21] <= UINT64_C(0xffff) &&
             state->f[22] <= UINT64_C(0xffe) &&
             (state->f[22] & UINT64_C(1)) == 0 && state->f[23] <= 1;
   }

   return false;
}

static void
agx_ioto_state_scatter(uint16_t words[5], uint64_t value, unsigned word,
                       unsigned destination_shift, unsigned width,
                       unsigned source_shift)
{
   const uint64_t mask = (UINT64_C(1) << width) - 1;
   words[word] |=
      (uint16_t)(((value >> source_shift) & mask) << destination_shift);
}

static bool
agx_ioto_state_encode_speclm(enum agx_ioto_state_encoding encoding,
                             const struct agx_ioto_state *state,
                             uint16_t words[5])
{
   if (!agx_ioto_state_validate_speclm(encoding, state))
      return false;

   if (encoding == AGX_IOTO_STATE_ENCODING_GPU0) {
      words[0] = UINT16_C(0x0077);
      for (unsigned f = 2; f <= 6; ++f)
         agx_ioto_state_scatter(words, state->f[f], 0, 9 + f, 1, 0);
      agx_ioto_state_scatter(words, state->f[7], 1, 0, 1, 0);
      agx_ioto_state_scatter(words, agx_ioto_e3[state->f[12]], 1, 1, 2, 0);
      agx_ioto_state_scatter(words, agx_ioto_e3[state->f[11]], 1, 3, 2, 0);
      agx_ioto_state_scatter(words, agx_ioto_e3[state->f[10]], 1, 5, 2, 0);
      agx_ioto_state_scatter(words, agx_ioto_e4[state->f[21]], 1, 7, 2, 0);
      agx_ioto_state_scatter(words, state->f[22], 1, 9, 5, 3);
      agx_ioto_state_scatter(words, state->f[25], 2, 6, 1, 0);
      agx_ioto_state_scatter(words, state->f[24], 2, 7, 1, 0);
      agx_ioto_state_scatter(words, state->f[18], 2, 8, 8, 6);
      agx_ioto_state_scatter(words, state->f[18], 3, 0, 2, 14);
      return true;
   }

   words[0] = UINT16_C(0x0007);
   words[1] = UINT16_C(0x0c00);
   agx_ioto_state_scatter(words, state->f[24], 0, 3, 1, 0);
   agx_ioto_state_scatter(words, state->f[25], 0, 4, 1, 0);
   agx_ioto_state_scatter(words, state->f[18], 0, 5, 11, 0);
   agx_ioto_state_scatter(words, state->f[18], 1, 3, 5, 11);
   for (unsigned f = 2; f <= 7; ++f)
      agx_ioto_state_scatter(words, state->f[f], 2, 8 + f, 1, 0);
   agx_ioto_state_scatter(words, state->f[26], 3, 0, 1, 0);
   agx_ioto_state_scatter(words, state->f[8], 3, 1, 1, 0);
   agx_ioto_state_scatter(words, state->f[9], 3, 2, 1, 0);
   agx_ioto_state_scatter(words, state->f[21], 3, 6, 2, 0);
   agx_ioto_state_scatter(words, state->f[22], 3, 8, 8, 0);
   return true;
}

static bool
agx_ioto_state_encode_ldshdr(enum agx_ioto_state_encoding encoding,
                             const struct agx_ioto_state *state,
                             uint16_t words[5])
{
   if (!agx_ioto_state_validate_ldshdr(encoding, state))
      return false;

   if (encoding == AGX_IOTO_STATE_ENCODING_GPU0) {
      words[0] = UINT16_C(0x0177);
      for (unsigned f = 7; f <= 11; ++f)
         agx_ioto_state_scatter(words, state->f[f], 0, 4 + f, 1, 0);
      agx_ioto_state_scatter(words, state->f[12], 1, 0, 1, 0);
      agx_ioto_state_scatter(words, agx_ioto_e3[state->f[17]], 1, 1, 2, 0);
      agx_ioto_state_scatter(words, agx_ioto_e3[state->f[16]], 1, 3, 2, 0);
      agx_ioto_state_scatter(words, agx_ioto_e3[state->f[15]], 1, 5, 2, 0);
      agx_ioto_state_scatter(words, state->f[0], 1, 7, 9, 6);
      agx_ioto_state_scatter(words, state->f[0], 2, 0, 16, 15);
      agx_ioto_state_scatter(words, state->f[0], 3, 0, 16, 31);
      agx_ioto_state_scatter(words, state->f[0], 4, 0, 1, 47);
      agx_ioto_state_scatter(words, agx_ioto_e4[state->f[1]], 4, 1, 2, 0);
      return true;
   }

   words[0] = UINT16_C(0x0007);
   words[1] = UINT16_C(0x0e00);
   agx_ioto_state_scatter(words, state->f[0], 0, 6, 10, 6);
   agx_ioto_state_scatter(words, state->f[1], 1, 3, 2, 0);
   agx_ioto_state_scatter(words, state->f[23], 1, 6, 1, 0);
   agx_ioto_state_scatter(words, state->f[3], 1, 12, 1, 0);
   agx_ioto_state_scatter(words, state->f[13], 1, 13, 1, 0);
   agx_ioto_state_scatter(words, state->f[14], 1, 14, 1, 0);
   agx_ioto_state_scatter(words, state->f[24], 1, 15, 1, 0);
   for (unsigned f = 7; f <= 12; ++f)
      agx_ioto_state_scatter(words, state->f[f], 2, 3 + f, 1, 0);
   agx_ioto_state_scatter(words, state->f[0], 3, 0, 16, 16);
   agx_ioto_state_scatter(words, state->f[0], 4, 0, 16, 32);
   return true;
}

static bool
agx_ioto_state_encode_spectpr(enum agx_ioto_state_encoding encoding,
                              const struct agx_ioto_state *state,
                              uint16_t words[5])
{
   if (!agx_ioto_state_validate_spectpr(encoding, state))
      return false;

   if (encoding == AGX_IOTO_STATE_ENCODING_GPU0) {
      words[0] = UINT16_C(0x00f7);
      for (unsigned f = 2; f <= 6; ++f)
         agx_ioto_state_scatter(words, state->f[f], 0, 9 + f, 1, 0);
      agx_ioto_state_scatter(words, state->f[7], 1, 0, 1, 0);
      agx_ioto_state_scatter(words, agx_ioto_e3[state->f[14]], 1, 1, 2, 0);
      agx_ioto_state_scatter(words, agx_ioto_e3[state->f[13]], 1, 3, 2, 0);
      agx_ioto_state_scatter(words, agx_ioto_e3[state->f[12]], 1, 5, 2, 0);
      agx_ioto_state_scatter(words, state->f[10], 1, 7, 7, 0);
      agx_ioto_state_scatter(words, state->f[11], 1, 14, 1, 0);
      agx_ioto_state_scatter(words, state->f[22], 1, 15, 1, 1);
      agx_ioto_state_scatter(words, state->f[22], 2, 0, 10, 2);
      agx_ioto_state_scatter(words, state->f[20], 2, 10, 6, 0);
      agx_ioto_state_scatter(words, state->f[20], 3, 0, 10, 6);
      agx_ioto_state_scatter(words, state->f[21], 3, 10, 6, 0);
      agx_ioto_state_scatter(words, state->f[21], 4, 0, 10, 6);
      return true;
   }

   words[0] = UINT16_C(0x0007);
   words[1] = UINT16_C(0x0d00);
   agx_ioto_state_scatter(words, state->f[10], 0, 3, 2, 0);
   agx_ioto_state_scatter(words, state->f[22], 0, 5, 11, 1);
   agx_ioto_state_scatter(words, state->f[10], 1, 3, 5, 2);
   agx_ioto_state_scatter(words, state->f[11], 1, 12, 1, 0);
   agx_ioto_state_scatter(words, state->f[8], 1, 13, 1, 0);
   agx_ioto_state_scatter(words, state->f[9], 1, 14, 1, 0);
   agx_ioto_state_scatter(words, state->f[23], 1, 15, 1, 0);
   for (unsigned f = 2; f <= 7; ++f)
      agx_ioto_state_scatter(words, state->f[f], 2, 8 + f, 1, 0);
   agx_ioto_state_scatter(words, state->f[20], 3, 0, 16, 0);
   agx_ioto_state_scatter(words, state->f[21], 4, 0, 16, 0);
   return true;
}

typedef bool (*agx_ioto_state_encoder)(enum agx_ioto_state_encoding,
                                       const struct agx_ioto_state *,
                                       uint16_t[5]);

static bool
agx_ioto_state_pack_record(enum agx_chip chip,
                           const struct agx_ioto_state *state, void *output,
                           size_t capacity, enum agx_ioto_kind kind,
                           size_t size, agx_ioto_state_encoder encode)
{
   if (!state || !output || capacity < size)
      return false;

   const enum agx_ioto_state_encoding encoding =
      agx_ioto_state_target_encoding(chip);
   if (encoding == AGX_IOTO_STATE_ENCODING_UNSUPPORTED)
      return false;

   uint16_t words[5] = {0};
   if (!encode(encoding, state, words))
      return false;

   uint8_t staged[AGX_IOTO_LDSHDR_SIZE] = {0};
   for (size_t i = 0; i < size / 2; ++i) {
      staged[i * 2] = (uint8_t)words[i];
      staged[i * 2 + 1] = (uint8_t)(words[i] >> 8);
   }

   /* encode() validates all fields and writes the fixed opcode bits. */
   (void)kind;
   memcpy(output, staged, size);
   return true;
}

/* Each packer is transactional: output is unchanged on every failure. */
bool
agx_ioto_state_pack_speclm(enum agx_chip chip,
                           const struct agx_ioto_state *state, void *output,
                           size_t capacity)
{
   return agx_ioto_state_pack_record(
      chip, state, output, capacity, AGX_IOTO_KIND_SPECLM_EP0,
      AGX_IOTO_SPECLM_SIZE, agx_ioto_state_encode_speclm);
}

bool
agx_ioto_state_pack_ldshdr(enum agx_chip chip,
                           const struct agx_ioto_state *state, void *output,
                           size_t capacity)
{
   return agx_ioto_state_pack_record(
      chip, state, output, capacity, AGX_IOTO_KIND_LDSHDR_EP0,
      AGX_IOTO_LDSHDR_SIZE, agx_ioto_state_encode_ldshdr);
}

bool
agx_ioto_state_pack_spectpr(enum agx_chip chip,
                            const struct agx_ioto_state *state, void *output,
                            size_t capacity)
{
   return agx_ioto_state_pack_record(
      chip, state, output, capacity, AGX_IOTO_KIND_SPECTPR_EP0,
      AGX_IOTO_SPECTPR_SIZE, agx_ioto_state_encode_spectpr);
}

/* The two-bit sample selector inside the SPECLM word. On G17P the imageblock
 * description carries log2(samples) beside the eight-byte stride, and the
 * SPECLM instruction holds that code in word 1 bits 7..8 through the table
 * {0 -> 2, 1 -> 0, 2 -> 1}. The one-sample value 2 matches every captured
 * one-sample record.
 */
static unsigned
agx_ioto_g17p_speclm_sample_bits(unsigned samples)
{
   return samples == 1 ? 0x0100u : samples == 4 ? 0x0080u : 0x0000u;
}

bool
agx_ioto_g17p_fragment_speclm_build(
   const struct agx_ioto_g17p_imageblock *imageblock, void *output,
   size_t capacity)
{
   if (!imageblock || !output || capacity < AGX_IOTO_SPECLM_SIZE)
      return false;

   const uint32_t samples = imageblock->nr_samples;
   const uint32_t sample_size_B = imageblock->sample_size_B;
   if ((imageblock->tile_size_px != 32 * 32 &&
        imageblock->tile_size_px != 32 * 16 &&
        imageblock->tile_size_px != 16 * 16) ||
       (!sample_size_B && samples != 0 && samples != 1 && samples != 2 &&
        samples != 4) ||
       (sample_size_B &&
        ((samples != 1 && samples != 2 && samples != 4) ||
         (sample_size_B & 7) || (sample_size_B >> 3) > 31 ||
         sample_size_B * samples * imageblock->tile_size_px >
            AGX_NATIVE_TILE_BUFFER_48K_BYTES)))
      return false;

   /* SPECLM encodes the sample stride in 8-byte units (5-bit field at bit 35,
    * max 248 B/sample) and log2(samples) in word 1 bits 7..8. The tile cannot
    * exceed the largest tile buffer a kernel admits (48 KiB); the render's
    * tilebuffer layout keeps it within what this kernel admits. Patch only
    * those fields onto the per-tile-geometry seeds,
    * preserving the opaque geometry bits in word 2[5:0]. */
   static const uint8_t imageblock_32x32[AGX_IOTO_SPECLM_SIZE] = {
      0x77, 0x00, 0x2a, 0x43, 0x4b, 0x00, 0x00, 0x00,
   };
   static const uint8_t imageblock_32x16[AGX_IOTO_SPECLM_SIZE] = {
      0x77, 0x00, 0x2a, 0x03, 0x43, 0x00, 0x00, 0x00,
   };
   static const uint8_t imageblock_16x16[AGX_IOTO_SPECLM_SIZE] = {
      0x77, 0x00, 0x2a, 0x03, 0x41, 0x00, 0x00, 0x00,
   };
   static const uint8_t no_imageblock[AGX_IOTO_SPECLM_SIZE] = {
      0x77, 0x00, 0x2a, 0x41, 0x00, 0x00, 0x00, 0x00,
   };

   uint8_t staged[AGX_IOTO_SPECLM_SIZE];
   if (!sample_size_B) {
      memcpy(staged, no_imageblock, sizeof(staged));
   } else {
      const uint8_t *seed =
         imageblock->tile_size_px == 32 * 32 ? imageblock_32x32
         : imageblock->tile_size_px == 32 * 16 ? imageblock_32x16
                                                : imageblock_16x16;
      memcpy(staged, seed, sizeof(staged));

      uint16_t word1;
      memcpy(&word1, staged + 2, sizeof(word1));
      word1 &= ~UINT16_C(0x3f80);
      word1 |= (uint16_t)((sample_size_B >> 3) << 9);
      word1 |= (uint16_t)agx_ioto_g17p_speclm_sample_bits(samples);
      memcpy(staged + 2, &word1, sizeof(word1));
   }

   memcpy(output, staged, sizeof(staged));
   return true;
}

bool
agx_ioto_state_tail_pack(enum agx_chip chip,
                         const struct agx_ioto_state_tail *tail,
                         struct agx_ioto_state_tail_output *output)
{
   if (!tail || !output)
      return false;

   struct agx_ioto_state_tail_output staged = {0};
   size_t offset = 0;

   switch (tail->type) {
   case AGX_IOTO_STATE_TAIL_TYPE0:
      if (!agx_ioto_state_pack_speclm(
             chip, &tail->speclm, staged.bytes + offset,
             sizeof(staged.bytes) - offset))
         return false;
      offset += AGX_IOTO_SPECLM_SIZE;
      break;
   case AGX_IOTO_STATE_TAIL_TYPE1:
      if (!agx_ioto_state_is_zero(&tail->speclm))
         return false;
      break;
   default:
      return false;
   }

   if (!agx_ioto_state_pack_ldshdr(chip, &tail->ldshdr,
                                    staged.bytes + offset,
                                    sizeof(staged.bytes) - offset))
      return false;
   offset += AGX_IOTO_LDSHDR_SIZE;

   if (!agx_ioto_state_pack_spectpr(chip, &tail->spectpr,
                                    staged.bytes + offset,
                                    sizeof(staged.bytes) - offset))
      return false;
   offset += AGX_IOTO_SPECTPR_SIZE;

   staged.length = offset;
   *output = staged;
   return true;
}

bool
agx_ioto_g17s_minimal_compute_tail_build(
   uint64_t shader_va, struct agx_ioto_state_tail *tail)
{
   if (!tail)
      return false;

   struct agx_ioto_state_tail staged = {
      .type = AGX_IOTO_STATE_TAIL_TYPE0,
   };
   staged.speclm.f[0] = 3;
   staged.speclm.f[1] = 165;
   staged.ldshdr.f[0] = shader_va;
   staged.ldshdr.f[5] = 3;
   staged.ldshdr.f[6] = 107;
   staged.ldshdr.f[23] = 1;
   staged.spectpr.f[0] = 3;
   staged.spectpr.f[1] = 166;

   struct agx_ioto_state_tail_output packed;
   if (!agx_ioto_state_tail_pack(AGX_CHIP_G17S, &staged, &packed))
      return false;

   *tail = staged;
   return true;
}

bool
agx_ioto_g17p_compiler_ldshdr_pack(
   uint64_t shader_va, uint32_t fetch_span, void *output, size_t capacity)
{
   if (!output || capacity < AGX_IOTO_LDSHDR_SIZE || fetch_span > 3 ||
       shader_va > UINT64_C(0xffffffffffff) ||
       (shader_va & UINT64_C(0x3f)) != 0)
      return false;

   /* Specialize agx_ioto_state_encode_ldshdr(GPU0) for the compiler state,
    * whose only nonzero semantic fields are f0/f1 and required f23. */
   const uint16_t words[5] = {
      UINT16_C(0x0177),
      UINT16_C(0x002a) | (uint16_t)(((shader_va >> 6) & 0x1ff) << 7),
      (uint16_t)(shader_va >> 15),
      (uint16_t)(shader_va >> 31),
      (uint16_t)(((shader_va >> 47) & 1) |
                 ((uint16_t)agx_ioto_e4[fetch_span] << 1)),
   };
   uint8_t *dst = output;
   for (unsigned i = 0; i < 5; ++i)
      agx_store_le16(&dst[2 * i], words[i]);

   return true;
}

bool
agx_ioto_g17p_runtime_compute_tail_build(
   uint64_t shader_va, enum agx_ioto_g17p_compute_ldshdr_f1 ldshdr_f1,
   struct agx_ioto_state_tail_output *output)
{
   if (!output)
      return false;

   switch (ldshdr_f1) {
   case AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3:
   case AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_GRID_HELPER:
   case AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_INDIRECT_ADD3:
      break;
   default:
      return false;
   }

   struct agx_ioto_state_tail tail = {
      .type = AGX_IOTO_STATE_TAIL_TYPE0,
   };
   tail.ldshdr.f[0] = shader_va;
   tail.ldshdr.f[1] = ldshdr_f1;
   tail.ldshdr.f[23] = 1;

   struct agx_ioto_state_tail_output staged;
   if (!agx_ioto_state_tail_pack(AGX_CHIP_G17P, &tail, &staged) ||
       staged.length != AGX_IOTO_STATE_TAIL_TYPE0_SIZE)
      return false;

   /* All three runtime witnesses set SPECLM word 1 bit 14. The recovered
    * anonymous-field encoder graph does not yet name this bit. */
   staged.bytes[3] |= UINT8_C(0x40);

   *output = staged;
   return true;
}

bool
agx_ioto_g17p_runtime_compute_tail_build_shared(
   uint64_t shader_va, enum agx_ioto_g17p_compute_ldshdr_f1 ldshdr_f1,
   uint32_t threadgroup_memory_size_B,
   struct agx_ioto_state_tail_output *output)
{
   if (!output || threadgroup_memory_size_B < 256 ||
       threadgroup_memory_size_B > 32768 ||
       (threadgroup_memory_size_B & 63) != 0)
      return false;

   struct agx_ioto_state_tail_output staged;
   if (!agx_ioto_g17p_runtime_compute_tail_build(shader_va, ldshdr_f1,
                                                  &staged))
      return false;

   /* Encode `(bytes << 2) | 0x80` in the second SPECLM dword via byte-size
    * f18 and enable f24 (supports up to 32 KiB threadgroup memory). */
   struct agx_ioto_state speclm = {0};
   speclm.f[18] = threadgroup_memory_size_B;
   speclm.f[24] = 1;
   if (!agx_ioto_state_pack_speclm(AGX_CHIP_G17P, &speclm, staged.bytes,
                                    AGX_IOTO_SPECLM_SIZE))
      return false;
   staged.bytes[3] |= UINT8_C(0x40);

   *output = staged;
   return true;
}

static bool
agx_ioto_g17p_compiler_spectpr_values(uint32_t spill_bytes,
                                      uint32_t d_value,
                                      uint16_t *spill_quads,
                                      uint16_t *d_rounded)
{
   if (!spill_quads || !d_rounded ||
       spill_bytes > (uint32_t)UINT16_MAX * 4 ||
       d_value > UINT32_MAX - 3)
      return false;

   *spill_quads = (uint16_t)((spill_bytes + 3) >> 2);
   *d_rounded =
      (uint16_t)(((((uint64_t)d_value + 3) >> 2) + 1) & UINT64_C(0xffe));
   return true;
}

bool
agx_ioto_g17p_compiler_spectpr_build(
   uint32_t spill_bytes, uint32_t d_value, struct agx_ioto_state *state)
{
   uint16_t spill_quads, d_rounded;
   if (!state || !agx_ioto_g17p_compiler_spectpr_values(
                    spill_bytes, d_value, &spill_quads, &d_rounded))
      return false;

   struct agx_ioto_state staged = {0};
   staged.f[20] = spill_quads;
   staged.f[21] = spill_quads;
   staged.f[22] = d_rounded;

   *state = staged;
   return true;
}

bool
agx_ioto_g17p_compiler_spectpr_pack(
   uint32_t spill_bytes, uint32_t d_value, void *output, size_t capacity)
{
   uint16_t spill_quads, d_rounded;
   if (!output || capacity < AGX_IOTO_SPECTPR_SIZE ||
       !agx_ioto_g17p_compiler_spectpr_values(
          spill_bytes, d_value, &spill_quads, &d_rounded))
      return false;

   /* Specialize agx_ioto_state_encode_spectpr(GPU0) for the compiler state,
    * whose only nonzero semantic fields are f20/f21/f22. */
   const uint16_t words[5] = {
      UINT16_C(0x00f7),
      UINT16_C(0x002a) | (uint16_t)((d_rounded & 2) << 14),
      (uint16_t)(((d_rounded >> 2) & 0x3ff) |
                 ((spill_quads & 0x3f) << 10)),
      (uint16_t)(((spill_quads >> 6) & 0x3ff) |
                 ((spill_quads & 0x3f) << 10)),
      (uint16_t)((spill_quads >> 6) & 0x3ff),
   };
   uint8_t *dst = output;
   for (unsigned i = 0; i < 5; ++i)
      agx_store_le16(&dst[2 * i], words[i]);

   return true;
}

#undef FM

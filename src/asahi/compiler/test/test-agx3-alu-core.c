/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The composer-driven wait-carrying ALU encoder must reproduce the compact hand
 * packers exactly when no wait is needed, and depositing a wait must change the
 * bytes (and keep the length) without disturbing the operands.
 */
#include "agx3_alu_core.h"
#include "agx3_falu_mod_core.h"
#include "agx3_pack.h"
#include "agx3_alu_core_vectors.c"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static void
test_native_immediate_shifts32(void)
{
   /* Neo's runtime-compiled Metal archives, all independently GPU-checked.
    * Counts optimized to other operations by Apple have no SLL/SRL fixture. */
   static const struct {
      bool left;
      uint16_t dest, src;
      unsigned shift;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      uint8_t apple[12];
   } vectors[] = {
      {true, 0, 0, 10, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x28, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_10-compute-6528.fields */
      {true, 0, 0, 11, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x2c, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_11-compute-6528.fields */
      {true, 0, 0, 12, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x30, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_12-compute-6528.fields */
      {true, 0, 0, 13, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x34, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_13-compute-6528.fields */
      {true, 0, 0, 14, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x38, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_14-compute-6528.fields */
      {true, 0, 0, 15, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x3c, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_15-compute-6528.fields */
      {true, 0, 0, 17, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x44, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_17-compute-6528.fields */
      {true, 0, 0, 18, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x48, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_18-compute-6528.fields */
      {true, 0, 0, 19, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x4c, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_19-compute-6528.fields */
      {true, 0, 0, 20, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x50, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_20-compute-6528.fields */
      {true, 0, 0, 21, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x54, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_21-compute-6528.fields */
      {true, 0, 0, 22, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x58, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_22-compute-6528.fields */
      {true, 0, 0, 23, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x5c, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_23-compute-6528.fields */
      {true, 0, 0, 24, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x60, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_24-compute-6528.fields */
      {true, 0, 0, 25, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x64, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_25-compute-6528.fields */
      {true, 0, 0, 26, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x68, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_26-compute-6528.fields */
      {true, 0, 0, 27, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x6c, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_27-compute-6528.fields */
      {true, 0, 0, 28, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x70, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_28-compute-6528.fields */
      {true, 0, 0, 29, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x74, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_29-compute-6528.fields */
      {true, 0, 0, 30, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x78, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_30-compute-6528.fields */
      {true, 0, 0, 31, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x7c, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_31-compute-6528.fields */
      {true, 0, 0, 9, 1, false, 32, {0x27, 0x00, 0x56, 0x00, 0x02, 0x00, 0x24, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* sll_9-compute-6528.fields */
      {false, 0, 0, 1, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x04, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_1-compute-6512.fields */
      {false, 0, 0, 10, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x28, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_10-compute-6528.fields */
      {false, 0, 0, 11, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x2c, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_11-compute-6528.fields */
      {false, 0, 0, 12, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x30, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_12-compute-6528.fields */
      {false, 0, 0, 13, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x34, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_13-compute-6528.fields */
      {false, 0, 0, 14, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x38, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_14-compute-6528.fields */
      {false, 0, 0, 15, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x3c, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_15-compute-6528.fields */
      {false, 0, 0, 17, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x44, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_17-compute-6528.fields */
      {false, 0, 0, 18, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x48, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_18-compute-6528.fields */
      {false, 0, 0, 19, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x4c, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_19-compute-6528.fields */
      {false, 0, 0, 2, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x08, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_2-compute-6512.fields */
      {false, 0, 0, 20, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x50, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_20-compute-6528.fields */
      {false, 0, 0, 21, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x54, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_21-compute-6528.fields */
      {false, 0, 0, 22, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x58, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_22-compute-6528.fields */
      {false, 0, 0, 23, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x5c, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_23-compute-6528.fields */
      {false, 0, 0, 24, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x60, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_24-compute-6528.fields */
      {false, 0, 0, 25, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x64, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_25-compute-6528.fields */
      {false, 0, 0, 26, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x68, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_26-compute-6528.fields */
      {false, 0, 0, 27, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x6c, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_27-compute-6528.fields */
      {false, 0, 0, 28, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x70, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_28-compute-6528.fields */
      {false, 0, 0, 29, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x74, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_29-compute-6528.fields */
      {false, 0, 0, 3, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x0c, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_3-compute-6528.fields */
      {false, 0, 0, 30, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x78, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_30-compute-6528.fields */
      {false, 0, 0, 4, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x10, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_4-compute-6512.fields */
      {false, 0, 0, 5, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x14, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_5-compute-6528.fields */
      {false, 0, 0, 6, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x18, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_6-compute-6528.fields */
      {false, 0, 0, 7, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x1c, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_7-compute-6512.fields */
      {false, 0, 0, 8, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x20, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_8-compute-6512.fields */
      {false, 0, 0, 9, 1, false, 32, {0xa7, 0x00, 0x56, 0x00, 0x02, 0x00, 0x24, 0x00, 0xf0, 0x11, 0x01, 0x00}}, /* srl_9-compute-6528.fields */
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      if (!agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX3B, vectors[i].left,
                               vectors[i].dest, vectors[i].src, vectors[i].shift,
                               vectors[i].hint, vectors[i].dest_cache,
                               vectors[i].wait, bytes, &length) || length != 12 ||
          memcmp(bytes, vectors[i].apple, 12)) {
         fprintf(stderr, "native immediate shift vector %u mismatch\n", i);
         abort();
      }
   }
   const unsigned bad_shifts[] = {0, 32, 255};
   for (unsigned i = 0; i < 3; ++i)
      if (agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX3B, true, 0, 2,
                             bad_shifts[i], AGX3_ALU_HINT_NONE, false, 0,
                             bytes, &length)) abort();
   const unsigned bad_registers[] = {1, 191, 192, 254, 256};
   for (unsigned i = 0; i < 5; ++i) {
      if (agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX3B, true, bad_registers[i],
                             0, 13, AGX3_ALU_HINT_NONE, false, 0,
                             bytes, &length)) abort();
      if (agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX3B, false, 0,
                             bad_registers[i], 17, AGX3_ALU_HINT_NONE, false, 0,
                             bytes, &length)) abort();
   }
   if (agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX2, true, 0, 2, 13,
                          AGX3_ALU_HINT_NONE, false, 0, bytes, &length) ||
       agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX3B, true, 0, 2, 13,
                          AGX3_ALU_HINT_NONE, false, 64, bytes, &length) ||
       agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX3B, true, 0, 2, 13,
                          AGX3_ALU_HINT_NONE, false, 0, NULL, &length) ||
       agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX3B, true, 0, 2, 13,
                          AGX3_ALU_HINT_NONE, false, 0, bytes, NULL)) abort();
   const uint16_t regs[] = {0, 2, 62, 64, 126, 128, 190};
   const unsigned shifts[] = {1, 5, 13, 17, 31};
   const uint16_t waits[] = {0, 1, 2, 4, 8, 16, 32, 63};
   unsigned legal = 0;
   for (unsigned left = 0; left < 2; ++left)
      for (unsigned d = 0; d < 7; ++d)
         for (unsigned v = 0; v < 7; ++v)
            for (unsigned h = 0; h < 3; ++h)
               for (unsigned cache = 0; cache < 2; ++cache)
                  for (unsigned w = 0; w < 8; ++w)
                     for (unsigned k = 0; k < 5; ++k) {
                        if (!agx3_alu_shift_imm32(AGX_ISA_ENCODING_AGX3B, left,
                                                 regs[d], regs[v], shifts[k], h,
                                                 cache, waits[w], bytes, &length) ||
                            length != 12) abort();
                        ++legal;
                     }
   printf("native immediate shifts: 51 byte-exact Metal vectors, %u legal register/lifetime/wait shapes and invalid-input guards passed\n", legal);
}

static void
test_native_popcount(void)
{
   /* applegpu_g17p popcount of ushort and uint loads and of computed values,
    * including a still-live SIMD ballot mask and wait slots 1, 2 and 6. */
   static const struct {
      uint16_t dest, src;
      bool half;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      uint8_t apple[8];
   } vectors[] = {
      {2, 2, true, AGX3_ALU_HINT_DISCARD, false, 1u << 5,
       {0x27, 0x05, 0x56, 0x02, 0x02, 0x04, 0x50, 0x04}},
      {2, 2, true, AGX3_ALU_HINT_DISCARD, false, 0,
       {0x27, 0x05, 0x54, 0x02, 0x02, 0x04, 0x50, 0x04}},
      {0, 0, true, AGX3_ALU_HINT_DISCARD, false, 1u << 5,
       {0x27, 0x05, 0x56, 0x00, 0x02, 0x00, 0x50, 0x04}},
      {2, 2, true, AGX3_ALU_HINT_DISCARD, false, 1u << 0,
       {0x27, 0x15, 0x54, 0x02, 0x02, 0x04, 0x50, 0x04}},
      {4, 1, true, AGX3_ALU_HINT_DISCARD, false, 1u << 1,
       {0x27, 0x25, 0x54, 0x04, 0x02, 0x02, 0x50, 0x04}},
      {0, 1, true, AGX3_ALU_HINT_CACHE, false, 1u << 5,
       {0x27, 0x05, 0x56, 0x00, 0x02, 0x02, 0x42, 0x04}},
      {0, 0, false, AGX3_ALU_HINT_DISCARD, false, 1u << 5,
       {0x27, 0x05, 0x56, 0x00, 0x02, 0x00, 0x5c, 0x04}},
      {2, 2, false, AGX3_ALU_HINT_DISCARD, false, 1u << 0,
       {0x27, 0x15, 0x54, 0x02, 0x02, 0x04, 0x5c, 0x04}},
      {4, 4, false, AGX3_ALU_HINT_DISCARD, false, 1u << 1,
       {0x27, 0x25, 0x54, 0x04, 0x02, 0x08, 0x5c, 0x04}},
      {4, 0, false, AGX3_ALU_HINT_CACHE, true, 0,
       {0x27, 0x05, 0x54, 0x04, 0x03, 0x00, 0x4e, 0x04}},
      /* The source stored after the count, and a cached count. */
      {3, 2, true, AGX3_ALU_HINT_NONE, false, 1u << 5,
       {0x27, 0x05, 0x56, 0x03, 0x02, 0x04, 0x40, 0x04}},
      {4, 0, false, AGX3_ALU_HINT_NONE, false, 1u << 5,
       {0x27, 0x05, 0x56, 0x04, 0x02, 0x00, 0x4c, 0x04}},
      {3, 2, true, AGX3_ALU_HINT_DISCARD, true, 1u << 5,
       {0x27, 0x05, 0x56, 0x03, 0x03, 0x04, 0x50, 0x04}},
      /* A chain of 70 to 78 live 64-bit values pushes the counted value
       * into h128 and above: the source's eighth bit is its own. */
      {128, 128, false, AGX3_ALU_HINT_DISCARD, true, 0,
       {0x27, 0x05, 0x54, 0x80, 0x03, 0x00, 0x5d, 0x04}},
      {136, 136, false, AGX3_ALU_HINT_DISCARD, true, 0,
       {0x27, 0x05, 0x54, 0x88, 0x03, 0x10, 0x5d, 0x04}},
      {144, 0, false, AGX3_ALU_HINT_DISCARD, true, 0,
       {0x27, 0x05, 0x54, 0x90, 0x03, 0x00, 0x5c, 0x04}},
      {152, 152, false, AGX3_ALU_HINT_DISCARD, true, 0,
       {0x27, 0x05, 0x54, 0x98, 0x03, 0x30, 0x5d, 0x04}},
      {160, 160, false, AGX3_ALU_HINT_DISCARD, true, 0,
       {0x27, 0x05, 0x54, 0xa0, 0x03, 0x40, 0x5d, 0x04}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_popcount(AGX_ISA_ENCODING_AGX3B, vectors[i].dest,
                               vectors[i].src, vectors[i].half,
                               vectors[i].hint, vectors[i].dest_cache,
                               vectors[i].wait, bytes, &len));
      assert(len == 8 && !memcmp(bytes, vectors[i].apple, len));
   }

   assert(!agx3_alu_popcount(AGX_ISA_ENCODING_AGX2, 0, 0, false,
                             AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(!agx3_alu_popcount(AGX_ISA_ENCODING_AGX3B, 0, 0, false,
                             AGX3_ALU_HINT_DISCARD, false, 1u << 6, bytes,
                             &len));
}

static void
test_gpu1_popcount(void)
{
   /* Apple applegpu_g17g, uint4/ushort4 popcount and popcount(v) + v. */
   static const struct {
      uint16_t dest, src, wait;
      bool half, live;
      uint8_t bytes[10];
   } witnesses[] = {
      {0,0,128,false,false,{0x2f,0x00,0x00,0x80,0x32,0x30,0xaa,0x02,0x04,0x00}},
      {2,2,0,false,false,{0x37,0x02,0x00,0x80,0x32,0x30,0xaa,0x02,0x04,0x00}},
      {4,4,0,false,false,{0x27,0x00,0x00,0x80,0x32,0x70,0xaa,0x0a,0x04,0x00}},
      {6,6,0,false,false,{0x37,0x02,0x00,0x80,0x32,0x70,0xaa,0x0a,0x04,0x00}},
      {0,0,128,true,false,{0x2f,0x00,0x00,0x00,0x32,0x20,0xaa,0x02,0x04,0x00}},
      {1,1,0,true,false,{0x27,0x01,0x00,0x01,0x32,0x20,0xaa,0x02,0x04,0x00}},
      {2,2,0,true,false,{0x37,0x02,0x00,0x00,0x32,0x20,0xaa,0x02,0x04,0x00}},
      {3,3,0,true,false,{0x37,0x03,0x00,0x01,0x32,0x20,0xaa,0x02,0x04,0x00}},
      {14,0,128,false,true,{0x3f,0x00,0x00,0x80,0x3a,0x30,0xaa,0x1a,0x02,0x00}},
      {12,4,0,false,true,{0x27,0x00,0x00,0x80,0x3a,0x70,0xaa,0x1a,0x02,0x00}},
      {16,2,0,false,true,{0xa7,0x02,0x00,0x80,0x3a,0x30,0xaa,0x02,0x02,0x00}},
      {10,6,0,false,true,{0x37,0x02,0x00,0x80,0x3a,0x70,0xaa,0x12,0x02,0x00}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(witnesses) / sizeof(witnesses[0]); ++i) {
      assert(agx3_alu_popcount(AGX_ISA_ENCODING_AGX3C,
         witnesses[i].dest, witnesses[i].src, witnesses[i].half,
         witnesses[i].live ? AGX3_ALU_HINT_CACHE : AGX3_ALU_HINT_DISCARD,
         witnesses[i].live, witnesses[i].wait, bytes, &len));
      assert(len == 10 && !memcmp(bytes, witnesses[i].bytes, len));
   }
   assert(!agx3_alu_popcount(AGX_ISA_ENCODING_AGX3C, 1, 0, false,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(!agx3_alu_popcount(AGX_ISA_ENCODING_AGX3C, 0, 256, false,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   /* Apple applegpu_g17g popcount(a[id]) + popcount(b[id * 3]): the count
    * of the slot-8 load waits in f38 and that of the slot-1 load in f31
    * (test/native/g17g/m5-libagx-reference.metal popcnt_two_loads). */
   static const uint8_t slot8[] = {0x2f,0x00,0x00,0x80,0x3a,0x30,0xaa,0x02,0x04,0x00};
   static const uint8_t slot1[] = {0x27,0x04,0x00,0x80,0x3a,0x70,0xaa,0x0a,0x04,0x00};
   assert(agx3_alu_popcount(AGX_ISA_ENCODING_AGX3C, 0, 0, false,
      AGX3_ALU_HINT_DISCARD, true, 128, bytes, &len));
   assert(len == 10 && !memcmp(bytes, slot8, len));
   assert(agx3_alu_popcount(AGX_ISA_ENCODING_AGX3C, 4, 4, false,
      AGX3_ALU_HINT_DISCARD, true, 1, bytes, &len));
   assert(len == 10 && !memcmp(bytes, slot1, len));
   /* No other slot is witnessed. */
   assert(!agx3_alu_popcount_g18g(AGX_ISA_ENCODING_AGX3C, 0, 0, false,
      AGX3_ALU_HINT_DISCARD, false, 2, bytes, &len));
}

static void
test_native_ft1(void)
{
   static const struct {
      uint16_t dest, src, wait;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint8_t bytes[8];
   } witnesses[] = {
      {2, 0, 1, AGX3_ALU_HINT_NONE, false,
       {0xa7, 0x05, 0x56, 0x02, 0x02, 0x00, 0x4c, 0x04}},
      {0, 0, 0, AGX3_ALU_HINT_CACHE, true,
       {0xa7, 0x05, 0x54, 0x00, 0x03, 0x00, 0x5c, 0x04}},
      {6, 0, 1, AGX3_ALU_HINT_DISCARD, true,
       {0xa7, 0x05, 0x56, 0x06, 0x03, 0x00, 0x4e, 0x04}},
      {6, 4, 1, AGX3_ALU_HINT_DISCARD, true,
       {0xa7, 0x05, 0x56, 0x06, 0x03, 0x04, 0x4e, 0x04}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(witnesses) / sizeof(witnesses[0]); ++i) {
      assert(agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3B, witnesses[i].dest,
                              witnesses[i].src,
                              witnesses[i].hint, witnesses[i].dest_cache,
                              witnesses[i].wait, bytes, &len));
      assert(len == 8 && !memcmp(bytes, witnesses[i].bytes, len));
   }

   /* This exact post-RA shape is the vkQuake update_lightmap use. */
   assert(agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3B, 26, 26,
                           AGX3_ALU_HINT_CACHE, true, 0, bytes, &len));
   assert(len == 8 &&
          !memcmp(bytes,
                  (uint8_t[]){0xa7, 0x05, 0x54, 0x1a,
                              0x03, 0x1a, 0x5c, 0x04},
                  8));

   assert(!agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3B, 27, 26,
                            AGX3_ALU_HINT_CACHE, true, 0, bytes, &len));
   assert(!agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3B, 26, 26,
                            AGX3_ALU_HINT_CACHE, true, 64, bytes, &len));
   assert(!agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3B, 26, 27,
                            AGX3_ALU_HINT_CACHE, true, 0, bytes, &len));
}

/*
 * Full-precision reciprocal and reciprocal square root.
 *
 * Only the half forms were wired up, so a 32-bit float divide or a
 * normalize() -- which NIR lowers to frcp / frsq on 32-bit values -- refused
 * the whole shader out of agx3_pack_binary. Both vectors are Apple's own,
 * from `metal-nt -N compute.mtlp -arch applegpu_g17p`:
 *
 *   o[i] = 1.0f / a[i]    af 00 56 00 02 00 10 48 20 00
 *   o[i] = rsqrt(a[i])    af 01 56 00 02 00 b0 40 00 00
 *
 * In both, the loaded source is dead (f29 = 1) and the instruction waits the
 * load's scoreboard slot 6.
 */
static void
test_native_f32_rcp_rsqrt(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   static const uint8_t rcp[] = {0xaf, 0x00, 0x56, 0x00, 0x02,
                                 0x00, 0x10, 0x48, 0x20, 0x00};
   assert(agx3_alu_f32rcp(AGX_ISA_ENCODING_AGX3B, 0, 0,
                          AGX3_ALU_HINT_DISCARD, false, 1u << 5, bytes, &len));
   assert(len == sizeof(rcp) && !memcmp(bytes, rcp, len));

   static const uint8_t rsq[] = {0xaf, 0x01, 0x56, 0x00, 0x02,
                                 0x00, 0xb0, 0x40, 0x00, 0x00};
   assert(agx3_alu_f32rsqrt(AGX_ISA_ENCODING_AGX3B, 0, 0,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, bytes,
                            &len));
   assert(len == sizeof(rsq) && !memcmp(bytes, rsq, len));

   /* The half and the 32-bit form are distinct encodings, not aliases. */
   uint8_t half[AGX3_ALU_MAX_LENGTH], hlen;
   assert(agx3_alu_hrcp(AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_DISCARD,
                        false, 1u << 5, half, &hlen));
   assert(hlen != len || memcmp(half, rcp, len) != 0);

   /* A 32-bit operand pair is aligned. */
   assert(!agx3_alu_f32rcp(AGX_ISA_ENCODING_AGX3B, 1, 0, AGX3_ALU_HINT_NONE,
                           false, 0, bytes, &len));
   assert(!agx3_alu_f32rsqrt(AGX_ISA_ENCODING_AGX3B, 0, 1, AGX3_ALU_HINT_NONE,
                             false, 0, bytes, &len));

   /* G17G's reciprocal is Apple's own, from the cube major axis of
    * gpu/m5-shadow-array-cube-reference.metal: the same form with the Gen10
    * root pair and eight producer slots. The reciprocal is the one piece of
    * cube coordinate arithmetic that is an ordinary ALU instruction, so it
    * lives here rather than beside CEM. */
   static const uint8_t gpu1_rcp[] = {0x27, 0x00, 0x04, 0x18, 0x2a,
                                      0x20, 0xa1, 0x02, 0x30, 0x00};
   assert(agx3_alu_f32rcp(AGX_ISA_ENCODING_AGX3C, 0, 0,
                          AGX3_ALU_HINT_DISCARD, true, 0, bytes, &len));
   assert(len == sizeof(gpu1_rcp) && !memcmp(bytes, gpu1_rcp, len));
   assert(agx3_alu_f32rcp(AGX_ISA_ENCODING_AGX3C, 0, 0, AGX3_ALU_HINT_NONE,
                          false, 1u << 7, bytes, &len));
   assert(!agx3_alu_f32rcp(AGX_ISA_ENCODING_AGX3C, 0, 0, AGX3_ALU_HINT_NONE,
                           false, 1u << 8, bytes, &len));
   assert(!agx3_alu_f32rcp(AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_NONE,
                           false, 1u << 6, bytes, &len));

   /* G17G's mixed constant/register 16-bit select, from Apple's own vector
    * select() kernels (gpu/m5-mixed-select3-reference.metal). Neither the
    * two-constant fseli16 form nor the two-register fsel32 one can express
    * `cond ? 1 : x`, which is what a lowered boolean blend produces. */
   static const uint8_t mixsel_lt[] = {0x22, 0x00, 0x05, 0x0a, 0x23, 0x00,
                                       0xa0, 0x02, 0x14, 0x80, 0x4e, 0x60,
                                       0x21, 0x04, 0x80, 0x00};
   static const uint8_t mixsel_ge[] = {0x22, 0x00, 0x05, 0x0a, 0x23, 0x00,
                                       0xa0, 0x02, 0x14, 0x80, 0x4e, 0x24,
                                       0x21, 0x04, 0x80, 0x00};
   assert(agx3_alu_fcmp_select16_const_reg_gpu1(
      0, 0, AGX3_ALU_HINT_DISCARD, 0, 1, 10, 1, 128, bytes, &len));
   assert(len == sizeof(mixsel_lt) && !memcmp(bytes, mixsel_lt, len));
   assert(agx3_alu_fcmp_select16_const_reg_gpu1(
      0, 0, AGX3_ALU_HINT_DISCARD, 0, 1, 10, 5, 128, bytes, &len));
   assert(len == sizeof(mixsel_ge) && !memcmp(bytes, mixsel_ge, len));
   /* The selected constant is three bits and the register is aligned; a
    * condition outside the witnessed five, and a wait on any slot but
    * eight, stay closed. */
   assert(!agx3_alu_fcmp_select16_const_reg_gpu1(
      0, 0, AGX3_ALU_HINT_DISCARD, 0, 8, 10, 1, 128, bytes, &len));
   assert(!agx3_alu_fcmp_select16_const_reg_gpu1(
      0, 0, AGX3_ALU_HINT_DISCARD, 0, 1, 11, 1, 128, bytes, &len));
   assert(!agx3_alu_fcmp_select16_const_reg_gpu1(
      0, 0, AGX3_ALU_HINT_DISCARD, 0, 1, 10, 3, 128, bytes, &len));
   assert(!agx3_alu_fcmp_select16_const_reg_gpu1(
      0, 0, AGX3_ALU_HINT_DISCARD, 0, 1, 10, 1, 32, bytes, &len));

   /* The general 16-bit select, against Apple's applegpu_g17g bytes from
    * test/native/g17g/m5-compare-select-reference.metal (glm_fsel16_regs,
    * sel_f16_imm, glm_fsel16_eqzero_imm, sel_f16_imm_zero), and the two
    * mixed constant/register witnesses above, which it must also spell. */
   const struct {
      struct agx3_fsel32_gpu1_desc d;
      uint8_t bytes[16];
      uint8_t len;
   } fsel16[] = {
      {{.dest = 2, .cmp0 = 0, .cmp0_hint = AGX3_ALU_HINT_DISCARD, .cmp1 = 2,
        .cmp1_hint = AGX3_ALU_HINT_DISCARD, .sel = {4, 5},
        .sel_hint = {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD},
        .condition = 1},
       {0x12, 0x01, 0x1d, 0x03, 0x82, 0x04, 0x03, 0x00, 0x80, 0x05}, 10},
      {{.dest = 0, .cmp0 = 0, .cmp0_hint = AGX3_ALU_HINT_DISCARD, .cmp1 = 2,
        .cmp1_hint = AGX3_ALU_HINT_DISCARD, .sel_immediate = {true, true},
        .sel = {255, 0}, .condition = 1},
       {0x02, 0x01, 0x1d, 0x03, 0xa2, 0xff, 0x03, 0x00, 0x20, 0x80}, 10},
      {{.dest = 0, .cmp0 = 0, .cmp0_hint = AGX3_ALU_HINT_DISCARD,
        .cmp1_immediate = true, .cmp1 = 0, .sel_immediate = {true, true},
        .sel = {255, 0}, .condition = 0},
       {0x02, 0x01, 0x0d, 0x81, 0xa6, 0xff, 0x00, 0x02, 0x20, 0x80}, 10},
      {{.dest = 0, .cmp0 = 0, .cmp0_hint = AGX3_ALU_HINT_DISCARD,
        .cmp1_immediate = true, .cmp1 = 0, .sel_immediate = {true, true},
        .sel = {255, 0}, .condition = 1},
       {0x02, 0x01, 0x0d, 0x81, 0xa2, 0xff, 0x03, 0x02, 0x20, 0x80}, 10},
      {{.dest = 0, .cmp0 = 0, .cmp0_hint = AGX3_ALU_HINT_DISCARD,
        .cmp1_immediate = true, .cmp1 = 0, .sel_immediate = {true, false},
        .sel = {1, 10}, .sel_hint = {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD},
        .condition = 1, .wait_mask = 128},
       {0}, 0 /* mixsel_lt */},
      {{.dest = 0, .cmp0 = 0, .cmp0_hint = AGX3_ALU_HINT_DISCARD,
        .cmp1_immediate = true, .cmp1 = 0, .sel_immediate = {true, false},
        .sel = {1, 10}, .sel_hint = {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD},
        .condition = 5, .wait_mask = 128},
       {0}, 0 /* mixsel_ge */},
   };
   for (unsigned i = 0; i < sizeof(fsel16) / sizeof(fsel16[0]); ++i) {
      const uint8_t *want = fsel16[i].bytes;
      unsigned want_len = fsel16[i].len;
      if (i == 4) {
         want = mixsel_lt;
         want_len = sizeof(mixsel_lt);
      } else if (i == 5) {
         want = mixsel_ge;
         want_len = sizeof(mixsel_ge);
      }
      assert(agx3_alu_fcmp_select16_gpu1(&fsel16[i].d, bytes, &len));
      assert(len == want_len && !memcmp(bytes, want, len));
   }
   /* Comparands are 32-bit registers; a hinted immediate, a condition
    * outside the five and a ninth wait slot stay closed. */
   struct agx3_fsel32_gpu1_desc bad = fsel16[0].d;
   bad.cmp1 = 3;
   assert(!agx3_alu_fcmp_select16_gpu1(&bad, bytes, &len));
   bad = fsel16[1].d;
   bad.sel_hint[0] = AGX3_ALU_HINT_CACHE;
   assert(!agx3_alu_fcmp_select16_gpu1(&bad, bytes, &len));
   bad = fsel16[1].d;
   bad.condition = 3;
   assert(!agx3_alu_fcmp_select16_gpu1(&bad, bytes, &len));
   bad = fsel16[1].d;
   bad.wait_mask = 256;
   assert(!agx3_alu_fcmp_select16_gpu1(&bad, bytes, &len));
}

/*
 * Full-precision base-2 logarithm and exponential.
 *
 * Only the half forms were wired up, so every 32-bit log2 or exp2 refused the
 * whole shader -- and with them every pow(), which NIR lowers to
 * fexp2(fmul(y, flog2(x))). An sRGB fragment epilog is made of exactly that,
 * which is why `asahi-g17p-srgb-epilog` could not compile and why a G17P
 * fragment shader writing an sRGB swapchain was refused outright.
 *
 * Ten vectors, all Apple's own from `metal-nt -arch applegpu_g17p` over three
 * probe sets. They vary the operand registers, the source lifetime, the
 * destination cache bit and the wait mask independently, so the table pins the
 * fields rather than merely covering the instruction.
 */
static void
test_native_f32_log2_exp2(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   static const struct {
      const char *what;
      bool exp2;
      uint16_t dest, src;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      uint8_t bytes[10];
   } vectors[] = {
      {"log2(a[i])", false, 0, 0, AGX3_ALU_HINT_DISCARD, false, 1u << 5,
       {0x2f, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"exp2(a[i])", true, 0, 0, AGX3_ALU_HINT_DISCARD, false, 1u << 5,
       {0xaf, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"log2 kept live for two readers", false, 4, 0, AGX3_ALU_HINT_CACHE,
       true, 1u << 5,
       {0x2f, 0x02, 0x56, 0x04, 0x03, 0x00, 0x92, 0x40, 0x00, 0x00}},
      {"exp2 of a multiply, pow()", true, 8, 6, AGX3_ALU_HINT_DISCARD, true, 0,
       {0xaf, 0x02, 0x54, 0x08, 0x03, 0x0c, 0xb0, 0x40, 0x00, 0x00}},
      {"exp2 in an sRGB encode", true, 4, 4, AGX3_ALU_HINT_DISCARD, true, 0,
       {0xaf, 0x02, 0x54, 0x04, 0x03, 0x08, 0xb0, 0x40, 0x00, 0x00}},
      /* exp2(log2(x) * y) spelled out: the shape NIR lowers fpow to, and the
       * one an sRGB epilog is built from. */
      {"log2 of a hand-written pow", false, 0, 0, AGX3_ALU_HINT_DISCARD, true,
       1u << 5,
       {0x2f, 0x02, 0x56, 0x00, 0x03, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"exp2 of a hand-written pow", true, 0, 0, AGX3_ALU_HINT_DISCARD, true, 0,
       {0xaf, 0x02, 0x54, 0x00, 0x03, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"exp2 of powr()", true, 0, 0, AGX3_ALU_HINT_DISCARD, false, 0,
       {0xaf, 0x02, 0x54, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"log2 with no wait", false, 0, 0, AGX3_ALU_HINT_DISCARD, false, 0,
       {0x2f, 0x02, 0x54, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"log2 of powr()", false, 4, 4, AGX3_ALU_HINT_DISCARD, true, 1u << 5,
       {0x2f, 0x02, 0x56, 0x04, 0x03, 0x08, 0xb0, 0x40, 0x00, 0x00}},
   };

   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert((vectors[i].exp2 ? agx3_alu_f32exp2 : agx3_alu_f32log2)(
         AGX_ISA_ENCODING_AGX3B, vectors[i].dest, vectors[i].src,
         vectors[i].hint, vectors[i].dest_cache, vectors[i].wait, bytes,
         &len));
      assert(len == sizeof(vectors[i].bytes));
      assert(!memcmp(bytes, vectors[i].bytes, len));
   }

   /* The half and the 32-bit form are distinct encodings, not aliases: the
    * two width selectors are the whole difference. */
   uint8_t half[AGX3_ALU_MAX_LENGTH], hlen;
   assert(agx3_alu_hlog2(AGX_ISA_ENCODING_AGX3B, 2, 2, AGX3_ALU_HINT_DISCARD,
                         false, 1u << 5, half, &hlen));
   assert(agx3_alu_f32log2(AGX_ISA_ENCODING_AGX3B, 2, 2, AGX3_ALU_HINT_DISCARD,
                           false, 1u << 5, bytes, &len));
   assert(hlen == len && memcmp(half, bytes, len) != 0);

   /* A 32-bit operand pair is aligned and the wait mask has six slots. */
   assert(!agx3_alu_f32log2(AGX_ISA_ENCODING_AGX3B, 1, 0, AGX3_ALU_HINT_NONE,
                            false, 0, bytes, &len));
   assert(!agx3_alu_f32exp2(AGX_ISA_ENCODING_AGX3B, 0, 1, AGX3_ALU_HINT_NONE,
                            false, 0, bytes, &len));
   assert(!agx3_alu_f32exp2(AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_NONE,
                            false, 1u << 6, bytes, &len));
}

/*
 * The source modifiers this whole form family carries: f31 negates the
 * source, f30 takes its absolute value. Every vector is Apple's own
 * applegpu_g17p machine code for the named expression, and every one differs
 * from the unmodified form in exactly those bits.
 *
 * exp(-x) is vkQuake's fog term. NIR folds the negation into the source, so
 * the packer saw `exp2 d32, r.neg` and refused every fragment shader in the
 * game.
 */
static void
test_native_f32_log2_exp2_modifiers(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   static const struct {
      const char *name;
      bool exp2;
      bool neg, abs;
      uint8_t bytes[10];
   } vectors[] = {
      {"exp2(a[i])", true, false, false,
       {0xaf, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"exp2(-a[i])", true, true, false,
       {0xaf, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x01, 0x00}},
      {"exp2(fabs(a[i]))", true, false, true,
       {0xaf, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0xc0, 0x00, 0x00}},
      {"exp2(-fabs(a[i]))", true, true, true,
       {0xaf, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0xc0, 0x01, 0x00}},
      {"log2(a[i])", false, false, false,
       {0x2f, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x00, 0x00}},
      {"log2(-a[i])", false, true, false,
       {0x2f, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0x40, 0x01, 0x00}},
      {"log2(fabs(a[i]))", false, false, true,
       {0x2f, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0xc0, 0x00, 0x00}},
      {"log2(-fabs(a[i]))", false, true, true,
       {0x2f, 0x02, 0x56, 0x00, 0x02, 0x00, 0xb0, 0xc0, 0x01, 0x00}},
   };

   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert((vectors[i].exp2 ? agx3_alu_f32exp2_mod : agx3_alu_f32log2_mod)(
         AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_DISCARD, false, 1u << 5,
         vectors[i].neg, vectors[i].abs, bytes, &len));
      assert(len == sizeof(vectors[i].bytes));
      assert(!memcmp(bytes, vectors[i].bytes, len));
   }

   /* The unmodified entry point and the modifier one agree when no modifier
    * is asked for, so nothing silently changed for existing callers. */
   uint8_t plain[AGX3_ALU_MAX_LENGTH], plen;
   assert(agx3_alu_f32exp2(AGX_ISA_ENCODING_AGX3B, 4, 6, AGX3_ALU_HINT_CACHE,
                           true, 1u << 2, plain, &plen));
   assert(agx3_alu_f32exp2_mod(AGX_ISA_ENCODING_AGX3B, 4, 6,
                               AGX3_ALU_HINT_CACHE, true, 1u << 2, false,
                               false, bytes, &len));
   assert(plen == len && !memcmp(plain, bytes, len));

   /* A negated source is a different instruction, not a no-op. */
   assert(agx3_alu_f32exp2_mod(AGX_ISA_ENCODING_AGX3B, 4, 6,
                               AGX3_ALU_HINT_CACHE, true, 1u << 2, true,
                               false, bytes, &len));
   assert(plen == len && memcmp(plain, bytes, len) != 0);
}

/*
 * Full-precision floor and ceiling. Apple's f32rint with f43 selecting the
 * mode and the width selectors raised from the half form. floor() on a float
 * is vkQuake's quantized lightmap coordinate; only the half form was wired
 * up, so world.frag refused.
 */
static void
test_native_f32_floor_ceil(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   /* o[i] = floor(a[i]) -- source dead, waiting the load's slot 6. */
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 0, 0,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, false,
                            false, bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x00\x02\x00\xb0\x40\x02\x00", 10));

   /* o[i] = ceil(a[i]) -- the same instruction with f43 = 2. */
   assert(agx3_alu_f32ceil(AGX_ISA_ENCODING_AGX3B, 0, 0,
                           AGX3_ALU_HINT_DISCARD, false, 1u << 5, false, false,
                           bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x00\x02\x00\xb0\x40\x04\x00", 10));

   /* o[i] = floor(-a[i]) -- the family's f31 negate. */
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 0, 0,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, true, false,
                            bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x00\x02\x00\xb0\x40\x03\x00", 10));

   /* float x = a[i]; o[i] = floor(x) + x -- the source is retained, and the
    * destination is cached for the add. This is the CACHE hint the half
    * encoder still refuses for want of a witness. */
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 4, 0, AGX3_ALU_HINT_CACHE,
                            true, 1u << 5, false, false, bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x04\x03\x00\x92\x40\x02\x00", 10));

   /* o[i] = floor(a[i]) * 4 + floor(b[i]) -- the second floor waits slot 1,
    * so the wait mask reaches further than the half form's slot 6. */
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 4, 4,
                            AGX3_ALU_HINT_DISCARD, true, 1u << 0, false, false,
                            bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x10\x54\x04\x03\x08\xb0\x40\x02\x00", 10));

   /* o[i] = ceil(fabs(a[i])) and floor(fabs(a[i])) -- the family's f30. */
   assert(agx3_alu_f32ceil(AGX_ISA_ENCODING_AGX3B, 0, 0,
                           AGX3_ALU_HINT_DISCARD, false, 1u << 5, false, true,
                           bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x00\x02\x00\xb0\xc0\x04\x00", 10));
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 0, 0,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, false, true,
                            bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x00\x02\x00\xb0\xc0\x02\x00", 10));

   /* float x = a[i]; o[i] = ceil(x) + x, and the second of two ceils. */
   assert(agx3_alu_f32ceil(AGX_ISA_ENCODING_AGX3B, 4, 0, AGX3_ALU_HINT_CACHE,
                           true, 1u << 5, false, false, bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x00\x56\x04\x03\x00\x92\x40\x04\x00", 10));
   assert(agx3_alu_f32ceil(AGX_ISA_ENCODING_AGX3B, 4, 4,
                           AGX3_ALU_HINT_DISCARD, true, 1u << 0, false, false,
                           bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes,
                  "\x2f\x10\x54\x04\x03\x08\xb0\x40\x04\x00", 10));

   /* The lifetime and wait fields are modelled, not pinned. The shared funop
    * baseline (agx3_funop_core.c) folds f29 and f40 into its form constant,
    * so it cannot express a live source or an absent slot-6 wait; these
    * encoders can, which is why floor and ceil do not fall through to it. */
   uint8_t live[AGX3_ALU_MAX_LENGTH], llen;
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 0, 0,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, false,
                            false, live, &llen));
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_NONE,
                            false, 0, false, false, bytes, &len));
   assert(llen == len && memcmp(live, bytes, len) != 0);

   /* The half and 32-bit forms are distinct encodings, not aliases. */
   uint8_t half[AGX3_ALU_MAX_LENGTH], hlen;
   assert(agx3_alu_hfloor(AGX_ISA_ENCODING_AGX3B, 2, 2, AGX3_ALU_HINT_DISCARD,
                          false, 1u << 5, half, &hlen));
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 2, 2,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, false,
                            false, bytes, &len));
   assert(hlen == len && memcmp(half, bytes, len) != 0);

   /* A 32-bit operand pair is aligned and the wait mask has six slots. */
   assert(!agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 1, 0, AGX3_ALU_HINT_NONE,
                             false, 0, false, false, bytes, &len));
   assert(!agx3_alu_f32ceil(AGX_ISA_ENCODING_AGX3B, 0, 1, AGX3_ALU_HINT_NONE,
                            false, 0, false, false, bytes, &len));
   assert(!agx3_alu_f32ceil(AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_NONE,
                            false, 1u << 6, false, false, bytes, &len));
}

/*
 * Full-precision round-to-nearest-even: the same f32rint form with f43
 * absent, the first line of the witness table beside the declaration.
 *
 * The select core encodes this operation too, but its encoding point holds
 * the source in a six-bit field and refuses anything above r63. Register
 * allocation reaches r64 in any shader with pressure, and the refusal
 * rejects the whole shader, so this form has to cover the register file.
 */
static void
test_native_f32_roundeven(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   /* metal-nt -arch applegpu_g17p, o[i] = rint(a[i]): the floor/ceil witness
    * with the mode field absent. */
   assert(agx3_alu_f32roundeven(AGX_ISA_ENCODING_AGX3B, 0, 0,
                                AGX3_ALU_HINT_DISCARD, false, 1u << 5, false,
                                false, bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes, "\x2f\x00\x56\x00\x02\x00\xb0\x40\x00\x00", 10));

   /* It differs from its siblings only in f43, so the mode really is the one
    * field that separates them. */
   uint8_t other[AGX3_ALU_MAX_LENGTH], olen;
   assert(agx3_alu_f32floor(AGX_ISA_ENCODING_AGX3B, 0, 0,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, false,
                            false, other, &olen));
   assert(olen == len && memcmp(other, bytes, 8) == 0 && other[8] != bytes[8]);
   assert(agx3_alu_f32ceil(AGX_ISA_ENCODING_AGX3B, 0, 0,
                           AGX3_ALU_HINT_DISCARD, false, 1u << 5, false, false,
                           other, &olen));
   assert(olen == len && memcmp(other, bytes, 8) == 0 && other[8] != bytes[8]);

   /* The family's modifiers, cache/lifetime hints and wait slots all reach
    * this mode, exactly as they reach floor and ceil. */
   assert(agx3_alu_f32roundeven(AGX_ISA_ENCODING_AGX3B, 4, 0,
                                AGX3_ALU_HINT_CACHE, true, 1u << 5, false,
                                false, bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes, "\x2f\x00\x56\x04\x03\x00\x92\x40\x00\x00", 10));
   assert(agx3_alu_f32roundeven(AGX_ISA_ENCODING_AGX3B, 4, 4,
                                AGX3_ALU_HINT_DISCARD, true, 1u << 0, false,
                                false, bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes, "\x2f\x10\x54\x04\x03\x08\xb0\x40\x00\x00", 10));
   assert(agx3_alu_f32roundeven(AGX_ISA_ENCODING_AGX3B, 0, 0,
                                AGX3_ALU_HINT_DISCARD, false, 1u << 5, false,
                                true, bytes, &len));
   assert(len == 10);
   assert(!memcmp(bytes, "\x2f\x00\x56\x00\x02\x00\xb0\xc0\x00\x00", 10));

   /* The point of the form. Every aligned 32-bit register pair this family
    * addresses encodes here, with no gaps -- 96 registers, r0 through r95,
    * the same reach agx3_alu_f32floor() and agx3_alu_f32ceil() already have
    * and the same the 32-bit move form has. The select core's encoding point
    * stops at r63 in both operands. */
   for (uint16_t d = 0; d <= 190; d += 2) {
      for (uint16_t s = 0; s <= 190; s += 2) {
         assert(agx3_alu_f32roundeven(AGX_ISA_ENCODING_AGX3B, d, s,
                                      AGX3_ALU_HINT_DISCARD, false, 1u << 5,
                                      false, false, bytes, &len));
         assert(len == 10);
      }
   }
   assert(!agx3_alu_f32roundeven(AGX_ISA_ENCODING_AGX3B, 192, 0,
                                 AGX3_ALU_HINT_DISCARD, false, 0, false, false,
                                 bytes, &len));
   assert(!agx3_alu_f32roundeven(AGX_ISA_ENCODING_AGX3B, 0, 192,
                                 AGX3_ALU_HINT_DISCARD, false, 0, false, false,
                                 bytes, &len));

   /* The exact shape the G17P robustness2 shaders refused: dest r59, source
    * r64. (test-agx3-pack.c pins the select core's refusal of it.) */
   assert(agx3_alu_f32roundeven(AGX_ISA_ENCODING_AGX3B, 118, 128,
                                AGX3_ALU_HINT_DISCARD, false, 0, false, false,
                                bytes, &len));
   assert(len == 10);

   /* Same guards as the rest of the family. */
   assert(!agx3_alu_f32roundeven(AGX_ISA_ENCODING_AGX3B, 1, 0,
                                 AGX3_ALU_HINT_NONE, false, 0, false, false,
                                 bytes, &len));
   assert(!agx3_alu_f32roundeven(AGX_ISA_ENCODING_AGX3B, 0, 1,
                                 AGX3_ALU_HINT_NONE, false, 0, false, false,
                                 bytes, &len));
   assert(!agx3_alu_f32roundeven(AGX_ISA_ENCODING_AGX3B, 0, 0,
                                 AGX3_ALU_HINT_NONE, false, 1u << 6, false,
                                 false, bytes, &len));
}

static void
test_native_bit_nor(void)
{
   /* metal-nt -arch applegpu_g17p, ~(x | y) of two buffer loads. */
   const struct {
      uint16_t src[2];
      enum agx3_alu_hint hints[2];
      uint8_t bytes[10];
   } cases[] = {
      {{2, 0}, {AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE},
       {0x2b, 0x83, 0x06, 0x81, 0x01, 0x00, 0x00, 0x80, 0x00, 0x00}},
      {{2, 0}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE},
       {0x2b, 0x03, 0x0e, 0x01, 0x01, 0x00, 0x00, 0x80, 0x00, 0x00}},
      {{0, 2}, {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD},
       {0x2b, 0x01, 0x16, 0x03, 0x01, 0x00, 0x00, 0x80, 0x00, 0x00}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      assert(agx3_alu_bit_nor(AGX_ISA_ENCODING_AGX3B, 4, cases[i].src,
                              cases[i].hints, false, 32, bytes, &len));
      assert(len == 10 && !memcmp(bytes, cases[i].bytes, 10));
   }
   puts("Native bit_nor: three exact live/dead witnesses PASS");
}

/* blend/native/bitops.py, R32Uint and RGBA32Uint logic ops. */
static void
test_native_bit_nand_xnor_orc(void)
{
   const enum agx3_alu_hint dead[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
   };
   const struct {
      bool (*encode)(enum agx_isa_encoding, uint16_t, const uint16_t[2],
                     const enum agx3_alu_hint[2], bool, uint16_t,
                     uint8_t[AGX3_ALU_MAX_LENGTH], uint8_t *);
      uint16_t dest, src[2], wait;
      uint8_t bytes[10];
   } cases[] = {
      {agx3_alu_bit_nand, 0, {2, 0}, 32,
       {0x0b, 0x03, 0x1e, 0x01, 0x03, 0x08, 0x00, 0x80, 0x00, 0x00}},
      {agx3_alu_bit_nand, 2, {2, 10}, 0,
       {0x1b, 0x03, 0x1e, 0x0b, 0x03, 0x08, 0x00, 0x00, 0x00, 0x00}},
      {agx3_alu_bit_xnor, 0, {2, 0}, 32,
       {0x0b, 0x03, 0x1f, 0x01, 0x01, 0x00, 0x00, 0x80, 0x00, 0x00}},
      {agx3_alu_bit_xnor, 6, {6, 8}, 0,
       {0x3b, 0x07, 0x1f, 0x09, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00}},
      {agx3_alu_bit_orc, 0, {0, 2}, 32,
       {0x0b, 0x01, 0x1f, 0x03, 0x03, 0x00, 0x00, 0x80, 0x00, 0x00}},
      {agx3_alu_bit_orc, 0, {2, 0}, 32,
       {0x0b, 0x03, 0x1f, 0x01, 0x03, 0x00, 0x00, 0x80, 0x00, 0x00}},
      {agx3_alu_bit_orc, 4, {12, 4}, 0,
       {0x2b, 0x0d, 0x1f, 0x05, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      assert(cases[i].encode(AGX_ISA_ENCODING_AGX3B, cases[i].dest,
                             cases[i].src, dead, false, cases[i].wait, bytes,
                             &len));
      assert(len == 10 && !memcmp(bytes, cases[i].bytes, 10));
   }
   puts("Native bit_nand/bit_xnor/bit_orc: seven exact witnesses PASS");
}

static void
test_band_identity_wait(void)
{
   /* The r0 identity that retires scoreboard slots, waiting on slot 1. The
    * composer table's bit_and specimen spells the same instruction with f27,
    * its first source's last use, set; the carrier keeps r0 live instead and
    * keeps the specimen's slot-6 wait. */
   static const uint8_t live[] = {0x0b, 0x01, 0x07, 0x01, 0x00,
                                  0x20, 0x00, 0x80, 0x00, 0x00};
   static const uint8_t dead[] = {0x0b, 0x01, 0x0f, 0x01, 0x00,
                                  0x20, 0x00, 0x80, 0x00, 0x00};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_band_identity_wait(AGX_ISA_ENCODING_AGX3B, 0, 1, bytes,
                                      &len));
   assert(len == sizeof(live) && !memcmp(bytes, live, sizeof(live)));

   const uint16_t src[] = {0, 0};
   const enum agx3_alu_hint last_use[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE,
   };
   assert(agx3_alu_bit_reg(AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND, 0, src,
                           last_use, 1 | (1u << 5), bytes, &len));
   assert(len == sizeof(dead) && !memcmp(bytes, dead, sizeof(dead)));
   assert(agx3_alu_compose(AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BAND, 0, 0, 0, 0,
                           0, 1, bytes, &len));
   assert(len == sizeof(dead) && !memcmp(bytes, dead, sizeof(dead)));
   puts("BAND identity wait: keeps its register live PASS");
}

static void
test_form_lifetime_baselines(void)
{
   const enum agx_isa_encoding b = AGX_ISA_ENCODING_AGX3B;
   /* Specimens whose baseline ends a source's lifetime. */
   assert(agx3_alu_form_ends_source(b, AGX3_ALU_IADD, 0));
   assert(!agx3_alu_form_ends_source(b, AGX3_ALU_IADD, 1));
   assert(agx3_alu_form_ends_source(b, AGX3_ALU_BAND, 0));
   assert(agx3_alu_form_ends_source(b, AGX3_ALU_FMUL, 0));
   assert(agx3_alu_form_ends_source(b, AGX3_ALU_FMUL, 1));
   assert(agx3_alu_form_ends_source(b, AGX3_ALU_FMA, 2));
   assert(agx3_alu_form_ends_source(b, AGX3_ALU_F2I, 0));
   /* ... and ones that keep their sources. */
   assert(!agx3_alu_form_ends_source(b, AGX3_ALU_ISUB, 0));
   assert(!agx3_alu_form_ends_source(b, AGX3_ALU_FADD, 0));
   assert(!agx3_alu_form_ends_source(b, AGX3_ALU_SLL, 0));
   assert(!agx3_alu_form_ends_source(b, AGX3_ALU_LSR, 1));
   assert(!agx3_alu_form_ends_source(b, AGX3_ALU_LDIMM, 0));
   assert(!agx3_alu_form_ends_source(AGX_ISA_ENCODING_AGX3C, AGX3_ALU_IADD, 0));
   puts("ALU form lifetime baselines PASS");
}

static void
test_native_bit_andc(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   const uint16_t dead_src[] = {0, 2};
   const enum agx3_alu_hint dead[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE,
   };
   assert(agx3_alu_bit_andc(AGX_ISA_ENCODING_AGX3B, 0, dead_src, dead,
                            false, 32, bytes, &len));
   assert(len == 10 &&
          !memcmp(bytes,
                  (uint8_t[]){0x0b, 0x01, 0x0e, 0x03, 0x02,
                              0x00, 0x00, 0x80, 0x00, 0x00},
                  10));

   const uint16_t live_src[] = {2, 4};
   const enum agx3_alu_hint live[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_bit_andc(AGX_ISA_ENCODING_AGX3B, 0, live_src, live,
                            false, 32, bytes, &len));
   assert(len == 10 &&
          !memcmp(bytes,
                  (uint8_t[]){0x0b, 0x03, 0x16, 0x05, 0x02,
                              0x00, 0x00, 0x80, 0x00, 0x00},
                  10));

   /* Exact vkQuake update_lightmap shape: r10 = $r10 & ~$r3. */
   const uint16_t quake_src[] = {20, 6};
   const enum agx3_alu_hint cached[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
   };
   assert(agx3_alu_bit_andc(AGX_ISA_ENCODING_AGX3B, 20, quake_src, cached,
                            false, 0, bytes, &len));
   assert(len == 10 &&
          !memcmp(bytes,
                  (uint8_t[]){0xab, 0x95, 0x06, 0x87, 0x02,
                              0x00, 0x00, 0x00, 0x00, 0x00},
                  10));
}

static void
test_native_minifloat_half_select(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_fcmp_minifloat_select_half(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 176, 10, 0,
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, 0, false, 32,
      bytes, &len));
   assert(len == 10 &&
          !memcmp(bytes,
                  (uint8_t[]){0x02, 0x01, 0x19, 0xb1, 0x86,
                              0x0a, 0x00, 0xc2, 0x20, 0x80},
                  10));

   /* A non-zero selected integer takes the f3 destination placement: Apple's
    * `x + ((a == -1.0f) ? h : 5)`, 12033db1860500222085, with its slot 5 wait
    * (f102) cleared. */
   assert(agx3_alu_fcmp_minifloat_select_half(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 176, 5, 5,
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, 0, true, 0, bytes, &len));
   assert(len == 10 &&
          !memcmp(bytes,
                  (uint8_t[]){0x12, 0x03, 0x3d, 0xb1, 0x86,
                              0x05, 0x00, 0x02, 0x20, 0x85},
                  10));

   /* Exact Quake form after normalizing its commutative #176,r15 compare. */
   assert(agx3_alu_fcmp_minifloat_select_half(
      AGX_ISA_ENCODING_AGX3B, 6, 30, 176, 6, 0,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, 2, true, 0, bytes, &len));
   assert(len == 10 &&
          !memcmp(bytes,
                  (uint8_t[]){0x32, 0x1f, 0x31, 0xb1, 0x02,
                              0x06, 0x02, 0x02, 0x20, 0x80},
                  10));
}

static void
check_minifloat_bits(bool multiply, unsigned dest, unsigned src, unsigned imm,
                     bool kill, unsigned wait)
{
   static const unsigned wait_decode[8] = {0, 1, 2, 4, 8, 16, 32, 3};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   const bool encoded = multiply
      ? agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, dest, src, imm,
                               kill, wait, bytes, &len)
      : agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, dest, src, imm,
                               false, kill, wait, bytes, &len);
   assert(encoded);
   assert(len == 6);
   uint64_t bits = 0;
   for (unsigned i = 0; i < len; ++i)
      bits |= (uint64_t)bytes[i] << (8 * i);
   assert(((bits >> 16) & 7) == (multiply ? 5u : 4u));

   /* Independent physical bit extraction; no composer/decoder round trip. */
   assert((((bits >> 34) & 1) | (((bits >> 4) & 15) << 1) |
           (((bits >> 22) & 3) << 5) | (((bits >> 44) & 1) << 7)) == dest);
   assert((((bits >> 36) & 1) | (((bits >> 25) & 63) << 1) |
           (((bits >> 42) & 1) << 7)) == src);
   assert((((bits >> 35) & 1) | (((bits >> 9) & 63) << 1) |
           (((bits >> 19) & 1) << 7)) == imm);
   assert(((bits >> 20) & 1) == kill);
   assert(wait_decode[(bits >> 45) & 7] == wait);
   assert(((bits >> 3) & 1) && ((bits >> 8) & 1) &&
          ((bits >> 24) & 1)); /* f32 widths */
   assert(((bits >> 15) & 1) && ((bits >> 39) & 1)); /* minifloat */
   assert(!(bits & ((UINT64_C(1) << 37) | (UINT64_C(1) << 38) |
                    (UINT64_C(1) << 41) | (UINT64_C(1) << 43))));
}

static void
test_fadd_minifloat(void)
{
   /* Fresh Apple -fno-fast-math witnesses: real loaded float producer,
    * neighboring immediates, negative immediate, and positively reused input.
    */
   static const struct {
      unsigned dest, src, imm, dest_cache, hint, wait;
      uint8_t bytes[6];
   } witness[] = {
      {0, 0, 0x20, 0, AGX3_ALU_HINT_DISCARD, 0,
       {0x09, 0xa1, 0x14, 0x01, 0x80, 0x00}},
      {2, 2, 0x20, 0, AGX3_ALU_HINT_DISCARD, 0,
       {0x19, 0xa1, 0x14, 0x03, 0x80, 0x00}},
      {0, 0, 0x20, 0, AGX3_ALU_HINT_DISCARD, 32,
       {0x09, 0xa1, 0x14, 0x01, 0x80, 0xc0}},
      {0, 0, 0x1f, 0, AGX3_ALU_HINT_DISCARD, 32,
       {0x09, 0x9f, 0x14, 0x01, 0x88, 0xc0}},
      {0, 0, 0x21, 0, AGX3_ALU_HINT_DISCARD, 32,
       {0x09, 0xa1, 0x14, 0x01, 0x88, 0xc0}},
      {0, 0, 0xa0, 0, AGX3_ALU_HINT_DISCARD, 32,
       {0x09, 0xa1, 0x1c, 0x01, 0x80, 0xc0}},
      {0, 2, 0x20, 0, AGX3_ALU_HINT_NONE, 32,
       {0x09, 0xa1, 0x04, 0x03, 0x80, 0xc0}},
      /* applegpu_g17p pair.fragment.bin SHA256
       * d38342dd219c2e9b5d9ee2ffc44beb185929fdaf994ac67a64099127ab3fc428:
       * a loaded float is retained across x - 0.5, and the result is reused.
       * Apple keeps both lifetimes and the producer wait on this one ADD. */
      {0, 4, 0xa0, 1, AGX3_ALU_HINT_CACHE, 32,
       {0x09, 0xa1, 0x2c, 0x85, 0x80, 0xc0}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   for (unsigned i = 0; i < sizeof(witness) / sizeof(witness[0]); ++i) {
      assert(agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B,
                                    witness[i].dest, witness[i].src,
                                    witness[i].imm, witness[i].dest_cache,
                                    witness[i].hint,
                                    witness[i].wait, bytes, &len));
      assert(len == 6 && !memcmp(bytes, witness[i].bytes, 6));
   }
   const unsigned waits[] = {0, 1, 2, 4, 8, 16, 32, 3};
   for (unsigned dest = 0; dest < 256; dest += 2)
      for (unsigned src = 0; src < 256; src += 2)
         for (unsigned w = 0; w < 8; ++w)
            for (unsigned kill = 0; kill < 2; ++kill)
               check_minifloat_bits(false, dest, src, 0x20, kill, waits[w]);
   for (unsigned imm = 0; imm < 256; ++imm)
      for (unsigned w = 0; w < 8; ++w)
         check_minifloat_bits(false, 254, 128, imm, true, waits[w]);

   /* Pixel-centre arithmetic and adjacent minifloats are exact across the
    * complete U16 pixel coordinate range. This checks the mathematical
    * contract, not a claim of executing the instruction on GPU hardware.
    */
   const unsigned numerator[] = {31, 32, 34};
   for (unsigned i = 0; i < 3; ++i) {
      for (uint32_t u = 0; u <= 65535; ++u) {
         uint32_t n = u * 64 + numerator[i];
         unsigned bit = 0;
         while ((n >> (bit + 1)) != 0)
            ++bit;
         uint32_t want = ((121 + bit) << 23) |
                         ((n - (1u << bit)) << (23 - bit));
         float f = (float)u + (float)numerator[i] / 64.0f;
         uint32_t got;
         memcpy(&got, &f, sizeof(got));
         assert(got == want);
      }
   }
   /* Normal IEEE nearest-even addition, including ties where a truncating
    * interpretation would differ. The -fno-fast-math native probes select
    * this ordinary f32add; no rounding/conversion modifier is synthesized.
    */
   const uint32_t tie_input[] = {0x4b000000, 0x4b000001, 0x4b7fffff};
   const uint32_t tie_result[] = {0x4b000000, 0x4b000002, 0x4b800000};
   for (unsigned i = 0; i < 3; ++i) {
      float input;
      memcpy(&input, &tie_input[i], sizeof(input));
      volatile float live = input;
      float result = live + 0.5f;
      uint32_t bits;
      memcpy(&bits, &result, sizeof(bits));
      assert(bits == tie_result[i]);
   }
   for (unsigned wait = 0; wait < 64; ++wait) {
      bool supported = wait == 0 || wait == 3 || !(wait & (wait - 1));
      memset(bytes, 0xa5, sizeof(bytes));
      len = 99;
      assert(agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 2, 0x20,
                                    false, AGX3_ALU_HINT_DISCARD, wait,
                                    bytes, &len) == supported);
      if (!supported)
         assert(len == 99 && bytes[0] == 0xa5);
   }
   /* Independently inspected AGX3c compact ADD retains these exact operand
    * and wait bits, while requiring its own dispatch/opcode fields. */
   const uint8_t m5_pixel_centre[] = {0x19,0xa1,0x14,0x07,0x80,0x00};
   assert(agx3_alu_fadd_minifloat_g18g(AGX_ISA_ENCODING_AGX3C, 2, 6, 32,
                                 false, AGX3_ALU_HINT_DISCARD, 0, bytes, &len));
   assert(len == sizeof(m5_pixel_centre) && !memcmp(bytes,m5_pixel_centre,len));
   for (unsigned i = 0; i < sizeof(witness) / sizeof(witness[0]); ++i) {
      assert(agx3_alu_fadd_minifloat_g18g(AGX_ISA_ENCODING_AGX3C,
                                    witness[i].dest, witness[i].src,
                                    witness[i].imm, witness[i].dest_cache,
                                    witness[i].hint, witness[i].wait, bytes, &len));
      assert(len == 6 && !memcmp(bytes, witness[i].bytes, 6));
   }
   for (unsigned wait = 0; wait < 256; ++wait) {
      bool supported = wait < 64 && (wait == 0 || wait == 3 || !(wait & (wait - 1)));
      assert(agx3_alu_fadd_minifloat_g18g(AGX_ISA_ENCODING_AGX3C, 0, 2, 32,
                                    false, AGX3_ALU_HINT_DISCARD, wait,
                                    bytes, &len) == supported);
   }
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 1, 0, 32,
                                  false, AGX3_ALU_HINT_NONE, 0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 1, 32,
                                  false, AGX3_ALU_HINT_NONE, 0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 256, 0, 32,
                                  false, AGX3_ALU_HINT_NONE, 0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 256, 32,
                                  false, AGX3_ALU_HINT_NONE, 0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 256,
                                  false, AGX3_ALU_HINT_NONE, 0, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 32,
                                  false, AGX3_ALU_HINT_NONE, 64, bytes, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 32,
                                  false, AGX3_ALU_HINT_NONE, 0, NULL, &len));
   assert(!agx3_alu_fadd_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 32,
                                  false, AGX3_ALU_HINT_NONE, 0, bytes, NULL));
   puts("G17P fadd minifloat: 8 Apple witnesses, register/wait/liveness bits,"
        " all256 immediates, all65536 pixel centres and adjacent values PASS");
}

static void
test_fmul_minifloat(void)
{
   static const struct {
      unsigned dest, src, imm, kill, wait;
      uint8_t bytes[6];
   } witness[] = {
      {0, 0, 0x02, 1, 0,  {0x09, 0x83, 0x15, 0x01, 0x80, 0x00}},
      {2, 2, 0x02, 1, 0,  {0x19, 0x83, 0x15, 0x03, 0x80, 0x00}},
      {0, 0, 0x02, 1, 32, {0x09, 0x83, 0x15, 0x01, 0x80, 0xc0}},
      {0, 0, 0x01, 1, 32, {0x09, 0x81, 0x15, 0x01, 0x88, 0xc0}},
      {0, 0, 0x03, 1, 32, {0x09, 0x83, 0x15, 0x01, 0x88, 0xc0}},
      {0, 0, 0x82, 1, 32, {0x09, 0x83, 0x1d, 0x01, 0x80, 0xc0}},
      {0, 2, 0x02, 0, 32, {0x09, 0x83, 0x05, 0x03, 0x80, 0xc0}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   for (unsigned i = 0; i < sizeof(witness) / sizeof(witness[0]); ++i) {
      assert(agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B,
                                    witness[i].dest, witness[i].src,
                                    witness[i].imm, witness[i].kill,
                                    witness[i].wait, bytes, &len));
      assert(len == 6 && !memcmp(bytes, witness[i].bytes, 6));
   }
   const unsigned waits[] = {0, 1, 2, 4, 8, 16, 32, 3};
   for (unsigned dest = 0; dest < 256; dest += 2)
      for (unsigned src = 0; src < 256; src += 2)
         for (unsigned w = 0; w < 8; ++w)
            for (unsigned kill = 0; kill < 2; ++kill)
               check_minifloat_bits(true, dest, src, 0x02, kill, waits[w]);
   for (unsigned imm = 0; imm < 256; ++imm)
      for (unsigned w = 0; w < 8; ++w)
         check_minifloat_bits(true, 254, 128, imm, true, waits[w]);

   /* Complete pixel-centre range times1/64,1/32,3/64. The result is the
    * exact rational (2*u+1)*scale/128, independently normalized below.
    */
   for (unsigned scale = 1; scale <= 3; ++scale) {
      for (uint32_t u = 0; u <= 65535; ++u) {
         uint32_t n = (2 * u + 1) * scale;
         unsigned bit = 0;
         while ((n >> (bit + 1)) != 0)
            ++bit;
         uint32_t want = ((120 + bit) << 23) |
                         ((n - (1u << bit)) << (23 - bit));
         float f = ((float)u + 0.5f) * ((float)scale / 64.0f);
         uint32_t got;
         memcpy(&got, &f, sizeof(got));
         assert(got == want);
      }
   }
   const uint32_t tie_input[] = {0x3f800001, 0x3f800003};
   const uint32_t tie_result[] = {0x3fc00002, 0x3fc00004};
   for (unsigned i = 0; i < 2; ++i) {
      float input;
      memcpy(&input, &tie_input[i], sizeof(input));
      volatile float live = input;
      float result = live * 1.5f;
      uint32_t bits;
      memcpy(&bits, &result, sizeof(bits));
      assert(bits == tie_result[i]);
   }
   for (unsigned wait = 0; wait < 64; ++wait) {
      bool supported = wait == 0 || wait == 3 || !(wait & (wait - 1));
      memset(bytes, 0xa5, sizeof(bytes));
      len = 99;
      assert(agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 2, 2, true,
                                    wait, bytes, &len) == supported);
      if (!supported)
         assert(len == 99 && bytes[0] == 0xa5);
   }
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX2, 0, 0, 2, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 1, 0, 2, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 1, 2, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 256, 0, 2, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 256, 2, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 256, 0,
                                  0, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 2, 0,
                                  64, bytes, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 2, 0,
                                  0, NULL, &len));
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3B, 0, 0, 2, 0,
                                  0, bytes, NULL));
   puts("G17P fmul minifloat: 7 Apple witnesses, register/wait/liveness bits,"
        " all256 immediates, all65536 scaled pixel centres PASS");
}

static void
test_fmul_minifloat_modifiers(void)
{
   unsigned rejected = 0;
   for (unsigned imm = 0; imm <= UINT8_MAX; ++imm) {
      for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache) {
         for (unsigned hint = AGX3_ALU_HINT_NONE;
              hint <= AGX3_ALU_HINT_CACHE; ++hint) {
            for (unsigned wait = 0; wait < 64; ++wait) {
               uint8_t absolute[AGX3_ALU_MAX_LENGTH];
               uint8_t absolute_len = 0xa5;
               memset(absolute, 0xa5, sizeof(absolute));
               assert(!agx3_alu_fmul_minifloat_mod(
                  AGX_ISA_ENCODING_AGX3B, 254, 128, imm, dest_cache,
                  true, hint, wait, absolute, &absolute_len));
               assert(absolute_len == 0xa5 && absolute[0] == 0xa5);
               rejected++;
            }
         }
      }
   }
   printf("G17P absolute minifloat FMUL: %u ungrounded fused encodings "
          "rejected PASS\n", rejected);
}

static void test_fmul_half_minifloat_to_f32(void)
{
   static const uint8_t native[] = {0x09, 0xf1, 0x35, 0x00, 0x80, 0x00};
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_fmul_half_minifloat_to_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 112, AGX3_ALU_HINT_NONE, 0, out,
      &len));
   assert(len == sizeof(native) && !memcmp(out, native, len));
   assert(!agx3_alu_fmul_half_minifloat_to_f32(
      AGX_ISA_ENCODING_AGX3C, 2, 0, 112, AGX3_ALU_HINT_DISCARD, 0, out,
      &len));
   assert(!agx3_alu_fmul_half_minifloat_to_f32(
      AGX_ISA_ENCODING_AGX3B, 3, 0, 112, AGX3_ALU_HINT_DISCARD, 0, out,
      &len));
   assert(!agx3_alu_fmul_half_minifloat_to_f32(
      AGX_ISA_ENCODING_AGX3B, 2, 0, 112, AGX3_ALU_HINT_CACHE, 0, out,
      &len));
   assert(!agx3_alu_fmul_half_minifloat_to_f32(
      AGX_ISA_ENCODING_AGX3B, 2, 0, 112, AGX3_ALU_HINT_DISCARD, 1, out,
      &len));
   puts("Native half times minifloat to f32: exact Apple vector and "
        "refusals PASS");
}

static void test_u16_to_f32(void)
{
   /* Literal applegpu-nt32023 G17P ushort/ushort2/reused-ushort witnesses.
    * Each source comes from an actual16-bit LDDEV, then a float32 STDEV.
    */
   static const struct { unsigned d,s,kill,wait; uint8_t bytes[8]; } v[] = {
      {0,0,1,32,{0xa7,0x07,0x56,0x00,0x02,0x00,0xa4,0x20}},
      {0,2,1,32,{0xa7,0x07,0x56,0x00,0x02,0x04,0xa4,0x20}},
      {2,3,1,0, {0xa7,0x07,0x54,0x02,0x02,0x06,0xa4,0x20}},
      {2,4,0,32,{0xa7,0x07,0x56,0x02,0x02,0x08,0x84,0x20}},
      /* hiregs.metal / hiregs16.metal (96 live conversions): destinations
       * and sources above half 127 use the same form, the source's high
       * bit in byte 6. */
      {132,88,1,0,{0xa7,0x07,0x54,0x84,0x02,0xb0,0xa4,0x20}},
      {134,88,1,0,{0xa7,0x07,0x54,0x86,0x02,0xb0,0xa4,0x20}},
      {70,162,1,0,{0xa7,0x07,0x54,0x46,0x02,0x44,0xa5,0x20}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   for(unsigned i=0;i<sizeof(v)/sizeof(v[0]);++i) {
      assert(agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,v[i].d,v[i].s,
                                v[i].kill,v[i].wait,bytes,&len));
      assert(len==8 && !memcmp(bytes,v[i].bytes,8));
   }
   /* Independent literal bit decode: register halves, unsigned16 width,
    * source kill and all six wait bits. Adjacent source half is not consumed.
    */
   for(unsigned d=0;d<192;d+=2) for(unsigned s=0;s<192;++s)
      for(unsigned k=0;k<2;++k) for(unsigned w=0;w<64;++w) {
         assert(agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,d,s,k,w,bytes,&len));
         assert(len==8 && bytes[3]==d && bytes[5]==((s&0x7f)<<1) &&
                (bytes[6]&1)==(s>>7));
         assert((bytes[6]&0x18)==0); /* source-width LUT[1]=0, U16 */
         assert(((bytes[6]>>5)&1)==k);
         assert((bytes[7]&0x40)==0); /* unsigned */
         assert((unsigned)((bytes[1]>>4)|((bytes[2]&3)<<4))==w);
      }
   /* Every U16 has an exact IEEE754f32 representation (<=16 significantbits).
    * Compare independent integer normalization to the C conversion over the
    * full domain; RTE/RTZ have no rounding divergence in this exact range.
    */
   for(uint32_t u=0;u<=65535;++u) {
      uint32_t expected=0;
      if(u) {
         unsigned bit=0;while((u>>(bit+1))!=0)++bit;
         expected=((127+bit)<<23)|((u-(1u<<bit))<<(23-bit));
      }
      float f=(float)(uint16_t)u;uint32_t actual;memcpy(&actual,&f,4);
      assert(actual==expected);
   }
   /* Independent AGX3c table compositions for the Zink copy shader's cast,
    * an odd source half, and signed conversion waiting on producer slot8.
    * These are encoding regressions, not hardware numerical witnesses. */
   const uint8_t m5_cast[] = {0x37,0x02,0x00,0x02,0x22,0x80,0xaf,0x0a,0x20,0x00};
   const uint8_t m5_odd[] = {0x27,0x03,0x00,0x02,0x22,0x80,0xaf,0x82,0x20,0x00};
   const uint8_t m5_signed[] = {0x3f,0x03,0x00,0x02,0x22,0x80,0xaf,0x02,0x60,0x00};
   assert(agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3C,6,2,
                             AGX3_ALU_HINT_DISCARD,0,bytes,&len));
   assert(len == sizeof(m5_cast) && !memcmp(bytes,m5_cast,len));
   assert(agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3C,0,3,
                             AGX3_ALU_HINT_DISCARD,32,bytes,&len));
   assert(len == sizeof(m5_odd) && !memcmp(bytes,m5_odd,len));
   assert(agx3_alu_s16_to_f32(AGX_ISA_ENCODING_AGX3C,2,3,
                             AGX3_ALU_HINT_DISCARD,128,bytes,&len));
   assert(len == sizeof(m5_signed) && !memcmp(bytes,m5_signed,len));
   /* G18G llama.cpp mul_mat_vec_q6_k_f32_f32 (M6, 2026-11-03): the register
    * allocator placed the u16->f32 conversion at half 136 with a discarded
    * source at half 42, and the >=128 guard refused to encode it, failing
    * vkCreateComputePipelines for the whole shader. The checked-in agx3c
    * int2flt form binds every dest and source half through 254 (8-bit
    * fields, probe against the form itself), so these vectors pin the
    * encoding at the form level. No live G18G capture at half >=128 yet:
    * a capture with an int2flt above half 127, or the hardware run, is the
    * confirming evidence. */
   const uint8_t y_g5_hiregs[] =
      {0x27,0x02,0x08,0x62,0x22,0x80,0xaf,0x12,0x20,0x00};
   const uint8_t y_g5_hiregs_signed[] =
      {0x27,0x03,0x08,0x62,0x22,0x80,0xaf,0x12,0x60,0x00};
   assert(agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3C,136,42,
                             AGX3_ALU_HINT_DISCARD,0,bytes,&len));
   assert(len == sizeof(y_g5_hiregs) && !memcmp(bytes,y_g5_hiregs,len));
   assert(agx3_alu_s16_to_f32(AGX_ISA_ENCODING_AGX3C,136,43,
                             AGX3_ALU_HINT_DISCARD,0,bytes,&len));
   assert(len == sizeof(y_g5_hiregs_signed) &&
          !memcmp(bytes,y_g5_hiregs_signed,len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3C,0,0,0,256,bytes,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX2,0,0,0,0,bytes,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,1,0,0,0,bytes,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,192,0,0,0,bytes,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,0,192,0,0,bytes,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,0,0,0,64,bytes,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,0,0,0,0,NULL,&len));
   assert(!agx3_alu_u16_to_f32(AGX_ISA_ENCODING_AGX3B,0,0,0,0,bytes,NULL));
   puts("G17P U16->F32: Apple witnesses,1048576 register/kill/wait encodings, all65536 exact numeric values PASS");
}
static void exact(const uint8_t *bytes, uint8_t length, const char *hex)
{
   if (strlen(hex) != (size_t)length*2) {
      fprintf(stderr, "length mismatch: expected %s (%zu bytes), got ",
              hex, strlen(hex)/2);
      for (unsigned j=0;j<length;++j) fprintf(stderr, "%02x", bytes[j]);
      fputc('\n', stderr);
   }
   assert(strlen(hex) == (size_t)length*2);
   for (unsigned i=0;i<length;++i) {
      unsigned value;
      assert(sscanf(hex+2*i,"%2x",&value)==1);
      if (bytes[i] != value) {
         fprintf(stderr, "expected %s, got ", hex);
         for (unsigned j=0;j<length;++j) fprintf(stderr, "%02x", bytes[j]);
         fputc('\n', stderr);
         assert(bytes[i] == value);
      }
   }
}

static void
test_modified_minifloat_add(void)
{
   /* Fresh Apple Metal 32023.917, applegpu_g17p: sat_only, complement,
    * sat_add, sat_neg, complement_reuse, sat_reuse, sat_pair, cache_add.
    */
   const struct {
      unsigned dest, src, imm;
      bool neg, sat;
      enum agx3_alu_hint hint;
      unsigned wait;
      const char *hex;
   } cases[] = {
      {0, 0, 0x80, false, true, AGX3_ALU_HINT_DISCARD, 32, "09011c8101020082"},
      {0, 0, 0x30, true, false, AGX3_ALU_HINT_DISCARD, 32, "09b1140180c8"},
      {0, 0, 0x20, false, true, AGX3_ALU_HINT_DISCARD, 32, "09a1140181000082"},
      {0, 0, 0x30, true, true, AGX3_ALU_HINT_DISCARD, 32, "09b1140181080082"},
      {0, 2, 0x30, true, false, AGX3_ALU_HINT_NONE, 32, "09b1040380c8"},
      {0, 2, 0x80, false, true, AGX3_ALU_HINT_NONE, 32, "0903148101020082"},
      {2, 2, 0x80, false, true, AGX3_ALU_HINT_DISCARD, 0, "19031c8101020002"},
      {0, 6, 0x80, false, true, AGX3_ALU_HINT_CACHE, 32, "0987148101020082"},
      {2, 6, 0x30, true, false, AGX3_ALU_HINT_CACHE, 0, "19b104878008"},
   };
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      assert(agx3_alu_fadd_minifloat_mod(
         AGX_ISA_ENCODING_AGX3B, cases[i].dest, cases[i].src, cases[i].imm,
         cases[i].neg, cases[i].sat, cases[i].hint, cases[i].wait, out, &len));
      exact(out, len, cases[i].hex);
   }

   /* Fresh Metal 32023.917 G17P fragments for 1.0f - abs(x), with x dead
    * and then retained for another output channel. */
   assert(agx3_alu_fadd_one_absneg(AGX_ISA_ENCODING_AGX3B, 0, 0, true,
                                   AGX3_ALU_HINT_DISCARD, 0, out, &len));
   exact(out, len, "09b13401820800000200");
   assert(agx3_alu_fadd_one_absneg(AGX_ISA_ENCODING_AGX3B, 2, 0, true,
                                   AGX3_ALU_HINT_CACHE, 0, out, &len));
   exact(out, len, "19b12481820800000200");
   assert(!agx3_alu_fadd_one_absneg(AGX_ISA_ENCODING_AGX3C, 0, 0, false,
                                    AGX3_ALU_HINT_NONE, 0, out, &len));
   assert(agx3_alu_fadd_abs_negzero(AGX_ISA_ENCODING_AGX3B, 8, 2, true,
                                    AGX3_ALU_HINT_CACHE, 1, out, &len));
   exact(out, len, "49812c83822000000200");
   assert(!agx3_alu_fadd_abs_negzero(AGX_ISA_ENCODING_AGX3C, 8, 2, true,
                                     AGX3_ALU_HINT_CACHE, 1, out, &len));

   static const struct {
      uint16_t imm;
      bool neg, sat;
      uint16_t wait;
      const char *hex;
   } abs_add[] = {
      {0x20, false, false, 32, "09a11401820000800200"},
      {0x38, false, false, 32, "09b91401820000800200"},
      {0x40, false, false, 32, "09c11401820000800200"},
      {0x20, true,  false, 32, "09a11401820800800200"},
      {0x20, false, true,  32, "09a11401820000820200"},
   };
   for (unsigned i = 0; i < sizeof(abs_add) / sizeof(abs_add[0]); ++i) {
      assert(agx3_alu_fadd_abs_minifloat(
         AGX_ISA_ENCODING_AGX3B, 0, 0, abs_add[i].imm, false,
         abs_add[i].neg, abs_add[i].sat, AGX3_ALU_HINT_DISCARD,
         abs_add[i].wait,
         out, &len));
      exact(out, len, abs_add[i].hex);
   }

   unsigned abs_shapes = 0;
   for (unsigned imm = 0; imm <= UINT8_MAX; ++imm)
      for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache)
         for (unsigned neg = 0; neg < 2; ++neg)
            for (unsigned sat = 0; sat < 2; ++sat)
               for (unsigned hint = AGX3_ALU_HINT_NONE;
                    hint <= AGX3_ALU_HINT_CACHE; ++hint)
                  for (unsigned wait = 0; wait < 64; ++wait) {
                     assert(agx3_alu_fadd_abs_minifloat(
                        AGX_ISA_ENCODING_AGX3B, 254, 128, imm,
                        dest_cache, neg, sat, hint, wait, out, &len));
                     assert(len == 10);
                     abs_shapes++;
                  }
   memset(out, 0xa5, sizeof(out));
   len = 0xa5;
   assert(!agx3_alu_fadd_abs_minifloat(
      AGX_ISA_ENCODING_AGX3C, 0, 0, 0x20, false, false, false,
      AGX3_ALU_HINT_NONE, 0, out, &len));
   assert(len == 0xa5 && out[0] == 0xa5);
   assert(!agx3_alu_fadd_abs_minifloat(
      AGX_ISA_ENCODING_AGX3B, 1, 0, 0x20, false, false, false,
      AGX3_ALU_HINT_NONE, 0, out, &len));
   assert(!agx3_alu_fadd_abs_minifloat(
      AGX_ISA_ENCODING_AGX3B, 0, 1, 0x20, false, false, false,
      AGX3_ALU_HINT_NONE, 0, out, &len));
   assert(!agx3_alu_fadd_abs_minifloat(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0x100, false, false, false,
      AGX3_ALU_HINT_NONE, 0, out, &len));

   const unsigned regs[] = {0, 2, 30, 62, 126, 128, 254};
   const unsigned immediates[] = {0, 0x20, 0x30, 0x80, 0xb0, 0xff};
   unsigned shapes = 0;
   for (unsigned d = 0; d < sizeof(regs) / sizeof(regs[0]); ++d)
      for (unsigned r = 0; r < sizeof(regs) / sizeof(regs[0]); ++r)
         for (unsigned k = 0; k < sizeof(immediates) / sizeof(immediates[0]); ++k)
            for (unsigned neg = 0; neg < 2; ++neg)
               for (unsigned hint = 0; hint <= AGX3_ALU_HINT_CACHE; ++hint)
                  for (unsigned wait = 0; wait < 64; ++wait) {
                     const unsigned imm = immediates[k];
                     const bool swap = imm == 0x80 && !neg;
                     assert(agx3_alu_fadd_minifloat_mod(
                        AGX_ISA_ENCODING_AGX3B, regs[d], regs[r], imm, neg,
                        true, hint, wait, out, &len));
                     assert(len == 8);
                     uint64_t bits = 0;
                     for (unsigned i = 0; i < len; ++i)
                        bits |= (uint64_t)out[i] << (i * 8);
                     /* Physical bit extraction independent of the composer. */
                     assert((((bits >> 34) & 1) | (((bits >> 4) & 15) << 1) |
                             (((bits >> 22) & 3) << 5) |
                             (((bits >> 44) & 1) << 7)) == regs[d]);
                     const unsigned low = swap ? 35 : 36;
                     const unsigned middle = swap ? 9 : 25;
                     const unsigned high = swap ? 40 : 42;
                     assert((((bits >> low) & 1) |
                             (((bits >> middle) & 63) << 1) |
                             (((bits >> high) & 1) << 7)) == regs[r]);
                     assert((((bits >> (swap ? 36 : 35)) & 1) |
                             (((bits >> (swap ? 25 : 9)) & 63) << 1) |
                             (((bits >> (swap ? 20 : 19)) & 1) << 7)) == imm);
                     assert(((bits >> (swap ? 15 : 31)) & 1) ==
                            (hint == AGX3_ALU_HINT_CACHE));
                     assert(((bits >> (swap ? 19 : 20)) & 1) ==
                            (hint == AGX3_ALU_HINT_DISCARD));
                     assert(((bits >> 43) & 1) == neg);
                     assert((bits >> 57) & 1);
                     assert((((bits >> 45) & 7) | (((bits >> 61) & 7) << 3)) == wait);
                     ++shapes;
                  }
   const uint8_t gpu1_sat[] = {0x09,0x01,0x14,0x81,0x01,0x02,0x00,0x02};
   assert(agx3_alu_fadd_minifloat_mod_g18g(AGX_ISA_ENCODING_AGX3C, 0, 0, 0x80,
                                    false, true, 0, 0, out, &len));
   assert(len == sizeof(gpu1_sat) && !memcmp(out, gpu1_sat, len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 1, 0, 0x80,
                                     false, true, 0, 0, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 0, 1, 0x80,
                                     false, true, 0, 0, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 256, 0, 0x80,
                                     false, true, 0, 0, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 0, 256, 0x80,
                                     false, true, 0, 0, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 0, 0, 256,
                                     false, true, 0, 0, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 0, 0, 0x80,
                                     false, true, 3, 0, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 0, 0, 0x80,
                                     false, true, 0, 64, out, &len));
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 0, 0, 0x30,
                                     true, false, 0, 63, out, &len));
   printf("Modified minifloat add:14 native witnesses +%u regular and %u "
          "absolute physical forms passed\n", shapes, abs_shapes);
}

static void test_half_uv(void)
{
   uint8_t out[16], size;
   const struct {unsigned d,bits; const char *hex;} moves[] = {
      {2,0x1800,"14800200000c0000"}, {2,0x1801,"14810200000c0000"},
      {2,0x9800,"14800200000c0400"}, {3,0x1800,"14800600000c0000"},
      {4,0x1800,"24800200000c0000"},
   };
   for (unsigned i=0;i<sizeof(moves)/sizeof(moves[0]);++i) {
      assert(agx3_alu_mov16(AGX_ISA_ENCODING_AGX3B,moves[i].d,moves[i].bits,0,out,&size));
      exact(out,size,moves[i].hex);
   }
   const struct {unsigned d,f,h; enum agx3_alu_hint a,b; bool cache; unsigned wait; const char *hex;} muls[] = {
      {0,0,4,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,32,"09011d0400c0"},
      {0,2,0,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_DISCARD,false,1,"090315000020"},
      {0,0,10,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,32,"09011d0a00c0"},
      {2,2,11,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,0,"19031d0a1000"},
      {0,0,10,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,false,32,"09010d8a00c0"},
      {2,2,10,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,0,"1903190a"},
      {0,0,2,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,true,32,"09013d0200c0"},
   };
   for (unsigned i=0;i<sizeof(muls)/sizeof(muls[0]);++i) {
      assert(agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,muls[i].d,muls[i].f,muls[i].h,muls[i].a,muls[i].b,
                   muls[i].cache,muls[i].wait,out,&size));
      exact(out,size,muls[i].hex);
   }
   assert(!agx3_alu_mov16(AGX_ISA_ENCODING_AGX3B,256,0x1800,0,out,&size));
   assert(!agx3_alu_mov16(AGX_ISA_ENCODING_AGX3B,2,0x10000,0,out,&size));
   assert(!agx3_alu_mov16(AGX_ISA_ENCODING_AGX3B,2,0x1800,1,out,&size));

   /* G17G half forms, Apple applegpu_g17g (gpu/m5-shift-add16-reference.metal
    * and the depth/stencil writers of gpu/m5-fragment-zs-constant-
    * reference.metal / gpu/m5-fragment-depth-reference.metal). */
   const struct {unsigned d; bool cache; unsigned bits; const char *hex;} g17g_moves[] = {
      {1,true,0xff00,"048026001c0c0700"}, {1,true,0xff5a,"04da26001c0c0700"},
      {1,true,0xff01,"048126001c0c0700"},
   };
   for (unsigned i=0;i<sizeof(g17g_moves)/sizeof(g17g_moves[0]);++i) {
      assert(agx3_alu_mov16_gpu1(g17g_moves[i].d,g17g_moves[i].cache,g17g_moves[i].bits,0,out,&size));
      exact(out,size,g17g_moves[i].hex);
   }
   assert(!agx3_alu_mov16_gpu1(256,false,0x1800,0,out,&size));
   assert(!agx3_alu_mov16_gpu1(2,false,0x10000,0,out,&size));
   assert(!agx3_alu_mov16_gpu1(2,false,0x1800,0x100,out,&size));
   const struct {
      unsigned d; bool cache, imm; unsigned s0; enum agx3_alu_hint h0;
      unsigned s1; enum agx3_alu_hint h1; unsigned shift, wait; const char *hex;
   } g17g_adds[] = {
      /* ushort a + b, a + (b << 4), a + (b << 2), v + (v << 4), 0 + (a << 4) */
      {2,false,false,3,AGX3_ALU_HINT_DISCARD,2,AGX3_ALU_HINT_DISCARD,0,0x80,"3f0304182100a302a0002100"},
      {2,false,false,2,AGX3_ALU_HINT_DISCARD,3,AGX3_ALU_HINT_DISCARD,4,0x80,"3f0204182100a302e0002000"},
      {2,false,false,2,AGX3_ALU_HINT_DISCARD,3,AGX3_ALU_HINT_DISCARD,2,0x80,"3f0204182100a302e0002500"},
      {2,false,false,2,AGX3_ALU_HINT_DISCARD,2,AGX3_ALU_HINT_DISCARD,4,0x80,"3f0204182100a302a0002000"},
      {2,false,true,0,AGX3_ALU_HINT_NONE,2,AGX3_ALU_HINT_DISCARD,4,0x80,"3f0004182500a302a0002000"},
      /* the coverage word of a depth writer: r2h = 0 + (r2l << 4) waiting
       * the SR read on slot 1, then r0l = r2l + (r2h << 4) */
      {5,true,true,0,AGX3_ALU_HINT_NONE,4,AGX3_ALU_HINT_CACHE,4,0x1,"270404192d00a30a20011000"},
      {0,true,false,4,AGX3_ALU_HINT_DISCARD,5,AGX3_ALU_HINT_DISCARD,4,0,"270004382900a30260012000"},
   };
   for (unsigned i=0;i<sizeof(g17g_adds)/sizeof(g17g_adds[0]);++i) {
      assert(agx3_alu_iadd16_gpu1(g17g_adds[i].d,g17g_adds[i].cache,g17g_adds[i].imm,g17g_adds[i].s0,
                                  g17g_adds[i].h0,g17g_adds[i].s1,g17g_adds[i].h1,g17g_adds[i].shift,
                                  g17g_adds[i].wait,out,&size));
      exact(out,size,g17g_adds[i].hex);
   }
   assert(!agx3_alu_iadd16_gpu1(2,false,false,2,AGX3_ALU_HINT_NONE,2,AGX3_ALU_HINT_NONE,5,0,out,&size));
   assert(!agx3_alu_iadd16_gpu1(256,false,false,2,AGX3_ALU_HINT_NONE,2,AGX3_ALU_HINT_NONE,0,0,out,&size));

   /* G17G integer-to-float, Apple applegpu_g17g
    * (gpu/m5-xfb-alu7-reference.metal): float(uint/int) and half(uint/int),
    * dead source waiting for slot 8. */
   {
      static const uint8_t u32f[] = {0x2f,0x00,0x00,0x02,0x22,0x80,0xaf,0x02,0x28,0x00};
      static const uint8_t i32f[] = {0x2f,0x00,0x00,0x02,0x22,0x80,0xaf,0x02,0x68,0x00};
      static const uint8_t u32h[] = {0x3f,0x02,0x00,0x00,0x22,0x80,0xaf,0x02,0x28,0x00};
      static const uint8_t i32h[] = {0x3f,0x02,0x00,0x00,0x22,0x80,0xaf,0x02,0x68,0x00};
      assert(agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3C,0,0,false,AGX3_ALU_HINT_DISCARD,false,0x80,out,&size));
      assert(size==sizeof(u32f) && !memcmp(out,u32f,size));
      assert(agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3C,0,0,true,AGX3_ALU_HINT_DISCARD,false,0x80,out,&size));
      assert(size==sizeof(i32f) && !memcmp(out,i32f,size));
      assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3C,2,2,false,AGX3_ALU_HINT_DISCARD,false,0x80,out,&size));
      assert(size==sizeof(u32h) && !memcmp(out,u32h,size));
      assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3C,2,2,true,AGX3_ALU_HINT_DISCARD,false,0x80,out,&size));
      assert(size==sizeof(i32h) && !memcmp(out,i32h,size));
      assert(!agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3C,0,0,false,AGX3_ALU_HINT_DISCARD,false,0x100,out,&size));
   }

   /* G17G half and-not, Apple applegpu_g17g ctz kernel
    * (gpu/m5-xfb-alu3-reference.metal): bit_andc h3 (cached) = h3 (dead)
    * & ~h2 (cached). */
   {
      static const uint8_t andc16[] = {0x33,0x03,0x06,0x01,0x3a,0x20,0xa2,0x02,0x84,0x00};
      assert(agx3_alu_bit_andc16_gpu1(3,true,3,AGX3_ALU_HINT_DISCARD,2,AGX3_ALU_HINT_CACHE,0,out,&size));
      assert(size==sizeof(andc16) && !memcmp(out,andc16,size));
      assert(!agx3_alu_bit_andc16_gpu1(3,true,3,AGX3_ALU_HINT_DISCARD,2,AGX3_ALU_HINT_CACHE,0x100,out,&size));

      /* The word form, Apple applegpu_g17g
       * (test/native/g17g/m5-libagx-reference.metal andc32_*):
       * r0 = r0 & ~r1 both dead; r2 = r0 & ~r1 both cached; r2 = r1 & ~r0
       * with r1 dead. */
      static const struct {
         uint16_t dest, keep, clear;
         enum agx3_alu_hint keep_hint, clear_hint;
         uint8_t bytes[10];
      } andc32[] = {
         {0, 0, 2, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
          {0x23,0x00,0x06,0x80,0x32,0x30,0xa4,0x02,0x84,0x80}},
         {4, 0, 2, AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
          {0x23,0x00,0x06,0x80,0x32,0x30,0xa2,0x0a,0x82,0x80}},
         {4, 2, 0, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE,
          {0x23,0x02,0x06,0x80,0x32,0x30,0xa0,0x0a,0x04,0x80}},
      };
      for (unsigned i = 0; i < 3; ++i) {
         assert(agx3_alu_bit_andc32_gpu1(andc32[i].dest,false,andc32[i].keep,
            andc32[i].keep_hint,andc32[i].clear,andc32[i].clear_hint,0,out,&size));
         assert(size==10 && !memcmp(out,andc32[i].bytes,10));
      }
      /* A word operand is an aligned pair. */
      assert(!agx3_alu_bit_andc32_gpu1(0,false,1,AGX3_ALU_HINT_DISCARD,2,AGX3_ALU_HINT_DISCARD,0,out,&size));

      /* Bit reverse, Apple applegpu_g17g reverse_bits(uint)
       * (test/native/g17g/m5-gl4-reference.metal rbit32_*): in place and
       * dead, into a cached r1 from a cached r0, and waiting for slot 8. */
      static const uint8_t rev_dead[] = {0x27,0x00,0x00,0x1a,0x26,0x00,0xa9,0x02,0x08,0x00};
      static const uint8_t rev_live[] = {0x37,0x80,0x00,0x1a,0x2a,0x00,0xa9,0x02,0x08,0x00};
      static const uint8_t rev_wait[] = {0x2f,0x00,0x00,0x1a,0x26,0x00,0xa9,0x02,0x08,0x00};
      assert(agx3_alu_rev_gpu1(0,0,AGX3_ALU_HINT_DISCARD,false,0,out,&size));
      assert(size==10 && !memcmp(out,rev_dead,10));
      assert(agx3_alu_rev_gpu1(2,0,AGX3_ALU_HINT_CACHE,true,0,out,&size));
      assert(size==10 && !memcmp(out,rev_live,10));
      assert(agx3_alu_rev_gpu1(0,0,AGX3_ALU_HINT_DISCARD,false,128,out,&size));
      assert(size==10 && !memcmp(out,rev_wait,10));
      /* No other wait slot is witnessed. */
      assert(!agx3_alu_rev_gpu1(0,0,AGX3_ALU_HINT_DISCARD,false,1,out,&size));

      /* Half AND/OR/XOR over registers (gpu/m5-xfb-alu9-reference.metal):
       * h2 = h3 op h2, both sources dead, waiting for slot 8. */
      static const struct { enum agx3_alu_op op; uint8_t bytes[10]; } r16[] = {
         {AGX3_ALU_BAND, {0x3b,0x03,0x07,0x00,0x30,0x20,0xa4,0x02,0x84,0x00}},
         {AGX3_ALU_BOR,  {0x3b,0x03,0x07,0x00,0x32,0x28,0xa4,0x02,0x84,0x00}},
         {AGX3_ALU_BXOR, {0x3b,0x03,0x06,0x00,0x32,0x28,0xa4,0x02,0x84,0x00}},
      };
      for (unsigned i = 0; i < 3; ++i) {
         assert(agx3_alu_bit_reg16_gpu1(r16[i].op,2,false,3,AGX3_ALU_HINT_DISCARD,2,AGX3_ALU_HINT_DISCARD,0x80,out,&size));
         assert(size==10 && !memcmp(out,r16[i].bytes,10));
      }
      assert(!agx3_alu_bit_reg16_gpu1(AGX3_ALU_BNOT,2,false,3,AGX3_ALU_HINT_DISCARD,2,AGX3_ALU_HINT_DISCARD,0x80,out,&size));
   }

   /* G17G find-top-one, Apple applegpu_g17g clz kernels
    * (gpu/m5-xfb-alu3-reference.metal). */
   {
      static const uint8_t ft1_u32[] = {0x2f,0x80,0x00,0x1a,0x2a,0x00,0xab,0x0a,0x08,0x00};
      static const uint8_t ft1_u16[] = {0x3f,0x83,0x00,0x18,0x2a,0x00,0xab,0x02,0x00,0x00};
      assert(agx3_alu_ft1_gpu1(4,0,false,AGX3_ALU_HINT_CACHE,true,0x80,out,&size));
      assert(size==sizeof(ft1_u32) && !memcmp(out,ft1_u32,size));
      assert(agx3_alu_ft1_gpu1(2,3,true,AGX3_ALU_HINT_CACHE,true,0x80,out,&size));
      assert(size==sizeof(ft1_u16) && !memcmp(out,ft1_u16,size));
      assert(!agx3_alu_ft1_gpu1(4,1,false,AGX3_ALU_HINT_CACHE,true,0x80,out,&size));
      assert(!agx3_alu_ft1_gpu1(4,0,false,AGX3_ALU_HINT_CACHE,true,0x40,out,&size));
   }

   /* G17G float compare to a 16-bit boolean (gpu/m5-xfb-alu-reference.metal,
    * -alu3-, -alu4-): minifloat and register comparands, lt and ge, the
    * selected constants 1/0, 3/7 and 0/1. */
   {
      static const struct {
         bool reg; uint16_t src1; bool kill; uint8_t t, f; unsigned cond;
         uint8_t bytes[16];
      } sel[] = {
         {false, 0xb0, false, 1, 0, 1,
          {0x32,0x02,0x05,0x0a,0x23,0x00,0xa0,0x02,0x14,0xac,0x4e,0x58,0x01,0x00,0x80,0x00}},
         {false, 0x40, false, 1, 0, 5,
          {0x32,0x02,0x05,0x0a,0x23,0x00,0xa0,0x02,0x14,0x90,0x4e,0x1c,0x01,0x00,0x80,0x00}},
         {false, 0x00, false, 1, 0, 1,
          {0x32,0x02,0x05,0x0a,0x23,0x00,0xa0,0x02,0x14,0x80,0x4e,0x58,0x01,0x00,0x80,0x00}},
         {false, 0x30, false, 3, 7, 1,
          {0x32,0x02,0x05,0x0a,0x23,0x00,0xa0,0x02,0x14,0x8c,0x4e,0x58,0x27,0x02,0x80,0x00}},
         {false, 0x30, false, 0, 1, 1,
          {0x32,0x02,0x05,0x0a,0x23,0x00,0xa0,0x02,0x14,0x8c,0x4a,0x58,0x05,0x00,0x80,0x00}},
         {true, 4, true, 1, 0, 1,
          {0x32,0x02,0x05,0x0a,0x23,0x00,0xa0,0x02,0x14,0x01,0x65,0x58,0x01,0x00,0x80,0x00}},
         {false, 0x30, false, 1, 0, 0,
          {0x32,0x02,0x05,0x0a,0x23,0x00,0xa0,0x02,0x14,0x8c,0x4e,0x18,0x01,0x00,0x80,0x00}},
         {false, 0x30, false, 0, 1, 0,
          {0x32,0x02,0x05,0x0a,0x23,0x00,0xa0,0x02,0x14,0x8c,0x4a,0x18,0x05,0x00,0x80,0x00}},
         {false, 0x30, false, 1, 0, 2,
          {0x32,0x02,0x45,0x0a,0x23,0x00,0xa0,0x02,0x14,0x8c,0x4e,0x18,0x01,0x00,0x80,0x00}},
         {false, 0x30, false, 1, 0, 6,
          {0x32,0x02,0x45,0x0a,0x23,0x00,0xa0,0x02,0x14,0x8c,0x4e,0x1c,0x01,0x00,0x80,0x00}},
      };
      for (unsigned i = 0; i < sizeof(sel)/sizeof(sel[0]); ++i) {
         assert(agx3_alu_fcmp_bool16_gpu1(2,2,AGX3_ALU_HINT_DISCARD,sel[i].reg,sel[i].src1,sel[i].kill,
                                          sel[i].t,sel[i].f,sel[i].cond,0x80,out,&size));
         assert(size==16 && !memcmp(out,sel[i].bytes,16));
      }
      assert(!agx3_alu_fcmp_bool16_gpu1(2,2,AGX3_ALU_HINT_DISCARD,false,0xb0,false,1,0,3,0x80,out,&size));
      assert(!agx3_alu_fcmp_bool16_gpu1(2,2,AGX3_ALU_HINT_DISCARD,false,0xb0,false,8,0,1,0x80,out,&size));
      assert(!agx3_alu_fcmp_bool16_gpu1(2,3,AGX3_ALU_HINT_DISCARD,false,0xb0,false,1,0,1,0x80,out,&size));
      assert(!agx3_alu_fcmp_bool16_gpu1(2,2,AGX3_ALU_HINT_DISCARD,false,0xb0,false,1,0,1,0x01,out,&size));
   }
   /* r0h = r0l & 255 in Apple's varying-stencil fragment */
   assert(agx3_alu_bit_imm_gpu1(AGX3_ALU_BAND,1,true,0,255,AGX3_ALU_HINT_DISCARD,0,out,&size));
   exact(out,size,"230007013820a302c4bf");
   assert(!agx3_alu_bit_imm_gpu1(AGX3_ALU_BOR,1,true,0,255,AGX3_ALU_HINT_DISCARD,0,out,&size));
   assert(!agx3_alu_bit_imm_gpu1(AGX3_ALU_BAND,256,true,0,255,AGX3_ALU_HINT_DISCARD,0,out,&size));
   assert(!agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,1,0,4,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,0,out,&size));
   assert(!agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,0,1,4,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,0,out,&size));
   assert(!agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,0,0,256,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,0,out,&size));
   assert(!agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,0,0,4,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,false,64,out,&size));
   unsigned shapes=0;
   for (unsigned d=0;d<256;d+=2)
      for (unsigned h=0;h<256;++h)
         for (unsigned wait=0;wait<=32;wait=wait ? wait*2 : 1) {
            assert(agx3_alu_fmul_mixed(AGX_ISA_ENCODING_AGX3B,d,(d+2)%256,h,AGX3_ALU_HINT_DISCARD, h&1 ? AGX3_ALU_HINT_DISCARD : AGX3_ALU_HINT_CACHE,
                         !!(h & 2),wait,out,&size));
            assert(size==4 || size==6);
            ++shapes;
         }
   puts("Half UV native candidate PASS:5 LDIMM16 +7 mixed-multiply byte-exact Apple witnesses; odd halves, source/result lifetimes and wait6/no-wait/compact variants; invalid widths/ranges/waits rejected");
   printf("Validated%u register/half/wait shapes through actual composer; offline production encoder coverage\n",shapes);
}

static void
test_native_fmul_abs_mixed(void)
{
   static const struct {
      uint16_t dest, f32, f16, wait;
      enum agx3_alu_hint hint32, hint16;
      bool dest_cache;
      const char *bytes;
   } vectors[] = {
      /* Both loaded inputs die, waiting on their producer slot. */
      {0, 0, 4, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "09011d040200008001000000"},
      /* The same native operation without a producer wait. */
      {0, 0, 4, 0, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "09011d040200000001000000"},
      /* Retain the f32 input for a later store. */
      {0, 2, 0, 32, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD, false,
       "090315000200008001000000"},
      /* Cache the f32 input shared by two channel multiplies. */
      {0, 2, 6, 32, AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_DISCARD, false,
       "098315060200008001000000"},
      /* The second channel proves that scalar-half register 7 stays odd. */
      {2, 2, 7, 0, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "19031d061200000001000000"},
      /* Reuse both the result and scalar-half source. */
      {0, 0, 2, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_CACHE, true,
       "09012d820200008001000000"},
      /* A separately scheduled conversion pins producer slot 0. */
      {0, 0, 8, 1, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "09011d080220000001000000"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_fmul_abs_mixed(
         AGX_ISA_ENCODING_AGX3B, vectors[i].dest, vectors[i].f32,
         vectors[i].f16, vectors[i].hint32, vectors[i].hint16,
         vectors[i].dest_cache, vectors[i].wait, bytes, &len));
      exact(bytes, len, vectors[i].bytes);
   }

   /* Every lifetime and producer-slot combination is an independent field on
    * this Apple endpoint. Exercise the full cross-product at the high bank
    * boundary, including a retained (non-cache, non-discard) half source. */
   for (unsigned hint32 = 0; hint32 <= AGX3_ALU_HINT_CACHE; ++hint32) {
      for (unsigned hint16 = 0; hint16 <= AGX3_ALU_HINT_CACHE; ++hint16) {
         for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache) {
            for (unsigned wait = 0; wait < 64; ++wait) {
               assert(agx3_alu_fmul_abs_mixed(
                  AGX_ISA_ENCODING_AGX3B, 254, 252, 255,
                  (enum agx3_alu_hint)hint32, (enum agx3_alu_hint)hint16,
                  dest_cache, wait, bytes, &len));
               assert(len == 12);
            }
         }
      }
   }

   memset(bytes, 0xa5, sizeof(bytes));
   len = 0xa5;
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3C, 0, 0, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3B, 1, 0, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3B, 0, 1, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 256, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 4, (enum agx3_alu_hint)3,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 4, AGX3_ALU_HINT_DISCARD,
      (enum agx3_alu_hint)3, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_abs_mixed(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 64, bytes, &len));
   assert(len == 0xa5);
   for (unsigned i = 0; i < sizeof(bytes); ++i)
      assert(bytes[i] == 0xa5);

   puts("Native abs-f32 times scalar-half: seven exact Apple vectors, "
        "1152 lifetime/wait forms and strict bounds PASS");
}

static void
test_bitwise_lifetimes(void)
{
   /* Apple g17p native kernels reuse the second operand, then release it;
    * keep_both stores both unmodified operands after the AND result. */
   const struct {
      enum agx3_alu_op op;
      uint16_t dest, src[2];
      enum agx3_alu_hint hints[2];
      unsigned wait;
      const char *hex;
   } cases[] = {
      {AGX3_ALU_BAND, 0, {0, 2}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_CACHE}, 32, "0b010f83000000800000"},
      {AGX3_ALU_BAND, 2, {6, 2}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD}, 1, "1b071f03002000000000"},
      {AGX3_ALU_BOR, 0, {0, 2}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_CACHE}, 32, "0b010f83020800800000"},
      {AGX3_ALU_BOR, 2, {6, 2}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD}, 1, "1b071f03022800000000"},
      {AGX3_ALU_BXOR, 0, {0, 2}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_CACHE}, 32, "0b010e83020800800000"},
      {AGX3_ALU_BXOR, 2, {6, 2}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD}, 1, "1b071e03022800000000"},
      {AGX3_ALU_BAND, 6, {2, 0}, {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE}, 32, "3b030701000000800000"},
   };
   uint8_t out[AGX3_ALU_MAX_LENGTH], size;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      assert(agx3_alu_bit_reg(AGX_ISA_ENCODING_AGX3B, cases[i].op,
                            cases[i].dest, cases[i].src, cases[i].hints,
                            cases[i].wait, out, &size));
      exact(out, size, cases[i].hex);
   }
   puts("Bitwise lifetimes: seven exact native AND/OR/XOR live/dead witnesses PASS");
}

static void
test_integer_lifetimes(void)
{
   const struct {
      enum agx3_alu_op op;
      uint16_t dest, src[2];
      enum agx3_alu_hint hints[2];
      const char *hex;
   } cases[] = {
      {AGX3_ALU_IADD, 0, {4, 0}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD}, "9f015600020800a81705"},
      {AGX3_ALU_IADD, 0, {4, 2}, {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE}, "9f015600020808a81105"},
      {AGX3_ALU_ISUB, 0, {0, 4}, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD}, "1f015600020010a81705"},
      {AGX3_ALU_ISUB, 0, {2, 4}, {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE}, "1f015600020410a81105"},
   };
   uint8_t out[AGX3_ALU_MAX_LENGTH], size;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      assert(agx3_alu_integer_reg(AGX_ISA_ENCODING_AGX3B, cases[i].op,
                                cases[i].dest, cases[i].src, cases[i].hints,
                                32, out, &size));
      exact(out, size, cases[i].hex);
   }
   puts("Integer lifetimes: four exact native IADD/ISUB live/dead witnesses PASS");
}

static void
test_float_compare_zero(void)
{
   const struct {
      unsigned dest, src[3], hints[3], cond, wait;
      bool half;
      const char *hex;
   } cases[] = {
      {0,{10,0,2},{1,1,1},0,32,true,"020a0f81840200c2"},
      {0,{2,4,6},{2,0,0},0,32,true,"02820781060400c20006"},
      {0,{10,0,2},{1,1,1},1,32,true,"020a0f81800203c2"},
      {0,{0,2,4},{1,1,1},0,32,false,"02010f81860200c28004"},
      {0,{6,2,4},{1,2,2},0,32,true,"02060f81068200c20084"},
      {2,{7,4,2},{1,1,1},0,0,true,"12060f8185040802"},
   };
   for (unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);++i) {
      uint8_t out[AGX3_ALU_MAX_LENGTH], len;
      uint16_t src[3]; enum agx3_alu_hint hints[3];
      for(unsigned j=0;j<3;++j) {src[j]=cases[i].src[j];hints[j]=cases[i].hints[j];}
      assert(agx3_alu_fcmp_select_zero(AGX_ISA_ENCODING_AGX3B,cases[i].dest,
             src,cases[i].half,hints,cases[i].cond,false,cases[i].wait,out,&len));
      exact(out,len,cases[i].hex);
   }
   /* Rejection leaves the output untouched. Half comparisons may be odd,
    * but full float operands/results must be aligned, and unsupported wait
    * sets must be retired explicitly by the caller. */
   for (unsigned bad=0;bad<8;++bad) {
      uint16_t src[] = {1,4,2};
      enum agx3_alu_hint hints[] = {AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE};
      uint8_t out[AGX3_ALU_MAX_LENGTH],len=0xa5;
      memset(out,0xa5,sizeof(out));
      unsigned dest=70,cond=0,wait=0;
      bool half=true;
      if(bad==0)dest=71;
      if(bad==1)src[1]=5;
      if(bad==2)src[2]=256;
      if(bad==3)half=false;
      if(bad==4)hints[0]=3;
      if(bad==5)wait=1;
      if(bad==6)cond=7;
      if(bad==7)dest=256;
      assert(!agx3_alu_fcmp_select_zero(AGX_ISA_ENCODING_AGX3B,dest,src,half,
                                      hints,cond,false,wait,out,&len));
      assert(len==0xa5);
      for(unsigned j=0;j<sizeof(out);++j)assert(out[j]==0xa5);
   }
   puts("FSEL zero: six exact mixed-width native lifetime witnesses PASS");
}

/*
 * A HALF-WIDTH COMPARATOR on the general compare-select.
 *
 * agx3_alu_fcmp_select() is the endpoint that takes a register in a select
 * slot (as opposed to two selected immediates). It pinned both comparators to
 * 32 bits -- their width fields f43/f60 were hardcoded to 2 and their register
 * numbers were required to be even -- so a highp value compared against a
 * mediump one refused the whole shader closed.
 *
 * That is not an exotic shape. It is what three.js's point-light shadow
 * fragment compiles to: cubeToUV()'s `absV.z >= almostOne` chain compares a
 * 32-bit component against a half-precision texel-size term and selects
 * between an immediate and a register. Under the G17P drm-shim the packer
 * refused
 *
 *    agx3_pack_binary.c:8001: no grounded AGX3 encoding for fcmpsel
 *      [d16:reg34 s0=32:reg34.discard s1=16:reg37.cache s2=16:imm1
 *       s3=16:reg50.discard cond=5 chan=1 mask=0x0 fmt=0 shift=0]
 *
 * and vkCreateGraphicsPipelines returned VK_ERROR_FEATURE_NOT_PRESENT.
 *
 * The width fields are Apple's, not inferred. Two kernels built with
 * `metal-nt -arch applegpu_g17p`, identical but for the second comparator's
 * width (`(x <= y) ? 1.0h : z` with y a float, then y a half):
 *
 *    both comparators 32-bit   12 03 19 05 26 b0 03 c0 80 06
 *    comparator 1 16-bit       12 03 19 04 26 b0 03 c0 80 05
 *
 * One bit moves for the width -- byte 3 bit 0, which is exactly where this
 * encoder deposits f60 -- and the comparator register keeps the same value
 * (4) across the pair. The other changed byte is the selected register, which
 * Apple's allocator placed differently; note it lands on an ODD half (5),
 * which the old even-only rule also refused.
 */
static void
test_fcmp_select_half_comparator(void)
{
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;

   /* The refused three.js instruction, verbatim. */
   struct agx3_fsel_desc d = {
      .dest = 34,
      .dest_16 = true,
      .cmp0 = 34,
      .cmp0_hint = AGX3_ALU_HINT_DISCARD,
      .cmp0_16 = false,
      .cmp1 = 37,
      .cmp1_hint = AGX3_ALU_HINT_CACHE,
      .cmp1_16 = true,
      .sel0_kind = AGX3_FSEL_SEL_IMMEDIATE,
      .sel0 = 1,
      .sel1_kind = AGX3_FSEL_SEL_REGISTER,
      .sel1 = 50,
      .sel_16 = true,
      .condition = 5,
   };
   assert(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   /* Corrected synthetic register-operand assertion. The old 0x20 selected
    * uniform half 50 rather than the requested GPR half 50. */
   exact(out, len, "12234da4268112000032");

   /* The same instruction with a 32-bit comparator differs only in the width
    * field and in the register number -- the length and every other byte are
    * unchanged, so widening is not silently selecting a different endpoint. */
   uint8_t wide[AGX3_ALU_MAX_LENGTH], wide_len;
   struct agx3_fsel_desc d32 = d;
   d32.cmp1_16 = false;
   d32.cmp1 = 36;
   assert(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &d32, wide, &wide_len));
   exact(wide, wide_len, "12234da5268102000032");
   assert(wide_len == len);

   /* Holding the register number and moving only the width isolates the bit:
    * byte 3, bit 0, the same bit Apple's two kernels differ in. */
   struct agx3_fsel_desc even16 = d;
   even16.cmp1 = 36;
   assert(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &even16, out, &len));
   exact(out, len, "12234da4268102000032");
   {
      unsigned differing = 0;
      for (unsigned i = 0; i < len; ++i)
         differing += (unsigned)__builtin_popcount((unsigned)(out[i] ^ wide[i]));
      assert(differing == 1);
      assert((out[3] ^ wide[3]) == 0x01);
   }

   /* The first comparator takes the same treatment, odd half included. */
   struct agx3_fsel_desc first = d;
   first.cmp0_16 = true;
   first.cmp0 = 35;
   first.cmp1_16 = false;
   first.cmp1 = 36;
   assert(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &first, out, &len));
   assert(len == wide_len && memcmp(out, wide, len) != 0);

   /* Fresh native all-half comparison with a register selected arm. */
   struct agx3_fsel_desc both = d;
   both.cmp0_16 = true;
   both.cmp0 = 34;
   assert(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &both, out, &len));

   struct agx3_fsel_desc native = {
      .dest = 0,
      .dest_16 = false,
      .cmp0 = 4,
      .cmp0_hint = AGX3_ALU_HINT_DISCARD,
      .cmp0_16 = true,
      .cmp1 = 5,
      .cmp1_hint = AGX3_ALU_HINT_DISCARD,
      .cmp1_16 = true,
      .sel0_kind = AGX3_FSEL_SEL_IMMEDIATE,
      .sel0 = 32,
      .sel1_kind = AGX3_FSEL_SEL_REGISTER,
      .sel1 = 0,
      .sel1_hint = AGX3_ALU_HINT_DISCARD,
      .sel_16 = false,
      .condition = 1,
      .wait_mask = 32,
   };
   assert(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &native, out, &len));
   exact(out, len, "02041f0421a013c0");
   native.sel0_kind = AGX3_FSEL_SEL_REGISTER;
   native.sel0 = 0;
   native.sel0_hint = AGX3_ALU_HINT_DISCARD;
   native.sel1_kind = AGX3_FSEL_SEL_IMMEDIATE;
   native.sel1 = 32;
   native.sel1_hint = AGX3_ALU_HINT_NONE;
   assert(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &native, out, &len));
   exact(out, len, "02041f0420a013c0");

   /* Fail-closed rules that must survive. A 32-bit comparator is a register
    * pair and still may not start on an odd half, and an immediate compare
    * slot always declares width 2. */

   struct agx3_fsel_desc odd32 = d;
   odd32.cmp1_16 = false;
   assert(!agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &odd32, out, &len));

   struct agx3_fsel_desc odd_first = d;
   odd_first.cmp0 = 35;
   assert(!agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &odd_first, out, &len));

   struct agx3_fsel_desc imm = d;
   imm.cmp1_is_immediate = true;
   imm.cmp1 = 0;
   imm.cmp1_16 = true;
   uint8_t imm_bytes[AGX3_ALU_MAX_LENGTH] = {0}, imm_len = 0;
   const bool imm_ok =
      agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &imm, imm_bytes, &imm_len);
   struct agx3_fsel_desc imm32 = imm;
   imm32.cmp1_16 = false;
   uint8_t imm32_bytes[AGX3_ALU_MAX_LENGTH] = {0}, imm32_len = 0;
   const bool imm32_ok = agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &imm32,
                                              imm32_bytes, &imm32_len);
   assert(imm_ok == imm32_ok);
   assert(imm_len == imm32_len &&
          memcmp(imm_bytes, imm32_bytes, imm_len) == 0);

   puts("FSEL half comparator: three.js point-shadow shape encodes PASS");
}

static void
test_native_fsel_selected_register_kind(void)
{
   /* Fresh G17P select-kind.metal, fabs(f32) < f16 ? a : b.
    * GPR compute text SHA 0029c35e1a30be4d161f3bb941b56a93e5f63749391afe31bee9d80094a855db.
    * Uniform text SHA 53ac8904298508f51d69ff53f46687581b5e7b32d9f1ce55981ebffba50d7903.
    * Both native instructions are at text offset 0x58. */
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   struct agx3_fsel_desc d = {
      .dest = 0, .cmp0 = 0, .cmp0_abs = true,
      .cmp0_hint = AGX3_ALU_HINT_DISCARD,
      .cmp1 = 2, .cmp1_16 = true, .cmp1_hint = AGX3_ALU_HINT_DISCARD,
      .sel0_kind = AGX3_FSEL_SEL_REGISTER, .sel0 = 4,
      .sel1_kind = AGX3_FSEL_SEL_REGISTER, .sel1 = 6,
      .sel0_hint = AGX3_ALU_HINT_DISCARD,
      .sel1_hint = AGX3_ALU_HINT_DISCARD,
      .condition = 1,
   };
   assert(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   exact(bytes, len, "02011f0283040300800600000100");

   d.sel0_kind = d.sel1_kind = AGX3_FSEL_SEL_UNIFORM;
   d.sel0 = 12;
   d.sel1 = 14;
   /* Uniforms cannot carry GPR-cache hints. */
   assert(!agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.sel0_hint = d.sel1_hint = AGX3_ALU_HINT_NONE;
   assert(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   exact(bytes, len, "02011f02230c0300200e00000100");
   d.sel0_kind = 4;
   assert(!agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   puts("FSEL selected GPR versus uniform native contrast PASS");
}

static void
test_float_compare_select(void)
{
   const uint16_t src[] = {0, 2};
   const enum agx3_alu_hint dead[] = {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
   const enum agx3_alu_hint live[] = {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,1,255,0,true,false,32,out,&len));
   exact(out,len,"02011d03a2ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,8,src,live,1,255,0,true,false,32,out,&len));
   exact(out,len,"42010503a2ff03c02080");
   const uint16_t mixed_src[] = {8, 10, 6, 4};
   const enum agx3_alu_hint mixed_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_fcmp_select_f32_f16(
      AGX_ISA_ENCODING_AGX3B, 4, mixed_src, mixed_hints, 0, true, 0,
      out, &len));
   exact(out, len, "2289278a05860000");
   const uint16_t half_half_src[] = {8, 9, 0, 4};
   const enum agx3_alu_hint half_half_hints[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
   };
   const bool no_abs[] = {false, false};
   assert(agx3_alu_fcmp_select_f16_f16_f32(
      AGX_ISA_ENCODING_AGX3B, 0, half_half_src, half_half_hints, no_abs,
      1 /* lt */, false, 32, out, &len));
   exact(out, len, "02081f08800413c0");
   const bool first_abs[] = {true, false};
   assert(agx3_alu_fcmp_select_f16_f16_f32(
      AGX_ISA_ENCODING_AGX3B, 0, half_half_src, half_half_hints, first_abs,
      1 /* lt */, false, 32, out, &len));
   exact(out, len, "02081f0883001300800400800100");
   const bool second_abs[] = {false, true};
   assert(agx3_alu_fcmp_select_f16_f16_f32(
      AGX_ISA_ENCODING_AGX3B, 0, half_half_src, half_half_hints, second_abs,
      1 /* lt */, false, 32, out, &len));
   exact(out, len, "02081f0883001300800400800200");
   const bool both_abs[] = {true, true};
   assert(agx3_alu_fcmp_select_f16_f16_f32(
      AGX_ISA_ENCODING_AGX3B, 0, half_half_src, half_half_hints, both_abs,
      1 /* lt */, false, 32, out, &len));
   exact(out, len, "02081f0883001300800400800300");
   assert(!agx3_alu_fcmp_select_f16_f16_f32(
      AGX_ISA_ENCODING_AGX3C, 0, half_half_src, half_half_hints, first_abs,
      1 /* lt */, false, 32, out, &len));
   const uint16_t lightmap_src[] = {46, 52, 26, 36};
   const enum agx3_alu_hint lightmap_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE,
   };
   assert(agx3_alu_fcmp_select_f32_f16(
      AGX_ISA_ENCODING_AGX3B, 36, lightmap_src, lightmap_hints,
      1, true, 0, out, &len));
   exact(out, len, "22af67b4029a03000024");
   /* The condition is orthogonal: gt, ge and le encode the same way, and
    * only the condition bytes move. Apple's own six-kernel sweep is in
    * test_native_f32_minifloat_select. The NaN-ignoring ltn/gtn codes were
    * not witnessed for this form and stay refused. */
   {
      uint8_t widened[AGX3_ALU_MAX_LENGTH];
      uint8_t widened_len;
      static const struct {
         unsigned condition;
         const char *bytes;
      } conditions[] = {
         {2 /* gt */, "22af67b4029a02000024"},
         {5 /* ge */, "22af67b4069a02000024"},
         {6 /* le */, "22af67b4069a03000024"},
      };
      for (unsigned i = 0; i < 3; ++i) {
         assert(agx3_alu_fcmp_select_f32_f16(
            AGX_ISA_ENCODING_AGX3B, 36, lightmap_src, lightmap_hints,
            conditions[i].condition, true, 0, widened, &widened_len));
         exact(widened, widened_len, conditions[i].bytes);
      }
      assert(!agx3_alu_fcmp_select_f32_f16(
         AGX_ISA_ENCODING_AGX3B, 36, lightmap_src, lightmap_hints, 3, true, 0,
         widened, &widened_len));
      assert(!agx3_alu_fcmp_select_f32_f16(
         AGX_ISA_ENCODING_AGX3B, 36, lightmap_src, lightmap_hints, 7, true, 0,
         widened, &widened_len));
   }
   assert(!agx3_alu_fcmp_select_f32_f16(
      AGX_ISA_ENCODING_AGX3C, 4, mixed_src, mixed_hints, 0, true, 0,
      out, &len));
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,5,255,0,true,false,32,out,&len));
   exact(out,len,"02011d03a6ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,8,src,live,5,255,0,true,false,32,out,&len));
   exact(out,len,"42010503a6ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,2,255,0,true,false,32,out,&len));
   exact(out,len,"02011d03a2ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,8,src,live,2,255,0,true,false,32,out,&len));
   exact(out,len,"42010503a2ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,6,255,0,true,false,32,out,&len));
   exact(out,len,"02011d03a6ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,8,src,live,6,255,0,true,false,32,out,&len));
   exact(out,len,"42010503a6ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,0,255,0,true,false,32,out,&len));
   exact(out,len,"02011d03a6ff00c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,8,src,live,0,255,0,true,false,32,out,&len));
   exact(out,len,"42010503a6ff00c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,0,0,255,true,false,32,out,&len));
   exact(out,len,"02011d03268000c0a0ff");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,8,src,live,0,0,255,true,false,32,out,&len));
   exact(out,len,"42010503268000c0a0ff");
   const uint16_t alpha_src[] = {6, 10};
   const enum agx3_alu_hint alpha_hint[] = {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD};
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,11,alpha_src,alpha_hint,1,0,1,true,true,0,out,&len));
   exact(out,len,"5a07350b228003002081");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,1,255,0,false,false,32,out,&len));
   exact(out,len,"02011f03a2ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,6,src,live,1,255,0,false,false,32,out,&len));
   exact(out,len,"32010703a2ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,5,255,0,false,false,32,out,&len));
   exact(out,len,"02011f03a6ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,6,src,live,5,255,0,false,false,32,out,&len));
   exact(out,len,"32010703a6ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,2,255,0,false,false,32,out,&len));
   exact(out,len,"02011f03a2ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,6,src,live,2,255,0,false,false,32,out,&len));
   exact(out,len,"32010703a2ff02c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,6,255,0,false,false,32,out,&len));
   exact(out,len,"02011f03a6ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,6,src,live,6,255,0,false,false,32,out,&len));
   exact(out,len,"32010703a6ff03c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,0,255,0,false,false,32,out,&len));
   exact(out,len,"02011f03a6ff00c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,6,src,live,0,255,0,false,false,32,out,&len));
   exact(out,len,"32010703a6ff00c02080");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,0,src,dead,0,0,255,false,false,32,out,&len));
   exact(out,len,"02011f03268000c0a0ff");
   assert(agx3_alu_fcmp_select_imm(AGX_ISA_ENCODING_AGX3B,6,src,live,0,0,255,false,false,32,out,&len));
   exact(out,len,"32010703268000c0a0ff");
   puts("Float select:25 native comparison/alpha dead-live vectors PASS");
}

static void test_int_to_float_liveness(void)
{
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3B,0,0,false,AGX3_ALU_HINT_DISCARD,false,32,out,&len));
   exact(out,len,"a70756000200ac20");
   assert(agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3B,0,2,false,AGX3_ALU_HINT_NONE,false,32,out,&len));
   exact(out,len,"a707560002048c20");
   assert(agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3B,0,0,true,AGX3_ALU_HINT_DISCARD,false,32,out,&len));
   exact(out,len,"a70756000200ac60");
   assert(agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3B,0,2,true,AGX3_ALU_HINT_NONE,false,32,out,&len));
   exact(out,len,"a707560002048c60");
   puts("Integer-to-float: four native signed/unsigned live/dead vectors PASS");
}

static void test_float_to_int32_liveness(void)
{
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_float_to_int32(
      AGX_ISA_ENCODING_AGX3B, 4, 2, true, AGX3_ALU_HINT_CACHE, true, 0,
      out, &len));
   exact(out, len, "27075404030496480300");
   assert(agx3_alu_float_to_int32(
      AGX_ISA_ENCODING_AGX3B, 190, 190, false, AGX3_ALU_HINT_DISCARD, true, 0,
      out, &len));
   exact(out, len, "270754be037cb5080200");
   assert(!agx3_alu_float_to_int32(
      AGX_ISA_ENCODING_AGX3B, 256, 0, false, AGX3_ALU_HINT_NONE, false, 0,
      out, &len));
   assert(!agx3_alu_float_to_int32(
      AGX_ISA_ENCODING_AGX3B, 0, 256, false, AGX3_ALU_HINT_NONE, false, 0,
      out, &len));
   assert(!agx3_alu_float_to_int32(
      AGX_ISA_ENCODING_AGX3B, 4, 3, true, AGX3_ALU_HINT_NONE, false, 0,
      out, &len));
   puts("Float-to-int32: native live-source cache and high-register vectors PASS");
}

static void test_i32_to_f16(void)
{
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 0, 2, false,
                              AGX3_ALU_HINT_NONE, true, 1, out, &len));
   exact(out, len, "a717540003048820");
   assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 2, 0, false,
                              AGX3_ALU_HINT_NONE, false, 0, out, &len));
   exact(out, len, "a707540202008820");
   assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 0, 0, false,
                              AGX3_ALU_HINT_DISCARD, false, 0, out, &len));
   exact(out, len, "a70754000200a820");
   assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 0, 0, true,
                              AGX3_ALU_HINT_DISCARD, false, 0, out, &len));
   exact(out, len, "a70754000200a860");
   assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 2, 0, true,
                              AGX3_ALU_HINT_NONE, true, 1, out, &len));
   exact(out, len, "a717540203008860");
   /* G17G is witnessed above; a ninth wait slot is still out of range. */
   assert(!agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3C, 0, 2, false,
                               AGX3_ALU_HINT_NONE, false, 0x100, out, &len));
   assert(!agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 0, 3, false,
                               AGX3_ALU_HINT_NONE, false, 0, out, &len));
   /* A CACHED source used to be refused; f27 carries it, the way f29 carries
    * discard.  Apple's own live-source kernel, recovered by composing with
    * the registers held equal rather than by diffing two kernels whose
    * register allocation differs: */
   assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 4, 0, true,
                              AGX3_ALU_HINT_CACHE, false, 0x20, out, &len));
   exact(out, len, "a707560402008a60");
   /* and its dead-source twin, which our field set already reproduced: */
   assert(agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 2, 2, true,
                              AGX3_ALU_HINT_DISCARD, false, 0x20, out, &len));
   exact(out, len, "a70756020204a860");
   assert(!agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3B, 0, 2, false,
                               AGX3_ALU_HINT_NONE, false, 0x40, out, &len));
   puts("Native i32-to-f16: exact Apple vectors incl. cached source, and "
        "refusals PASS");
}

static void
test_f16_to_int32(void)
{
   static const struct {
      bool sign, cache;
      uint16_t dest, src, wait;
      enum agx3_alu_hint hint;
      uint8_t bytes[10];
   } native[] = {
      {true, false, 0, 0, 32, AGX3_ALU_HINT_DISCARD,
       {0x27, 0x07, 0x56, 0x00, 0x02, 0x00, 0xac, 0x48, 0x03, 0x00}},
      {false, false, 0, 0, 32, AGX3_ALU_HINT_DISCARD,
       {0x27, 0x07, 0x56, 0x00, 0x02, 0x00, 0xac, 0x08, 0x02, 0x00}},
      {true, false, 0, 0, 0, AGX3_ALU_HINT_DISCARD,
       {0x27, 0x07, 0x54, 0x00, 0x02, 0x00, 0xac, 0x48, 0x03, 0x00}},
      {true, false, 2, 4, 32, AGX3_ALU_HINT_NONE,
       {0x27, 0x07, 0x56, 0x02, 0x02, 0x08, 0x8c, 0x48, 0x03, 0x00}},
      {true, true, 0, 0, 32, AGX3_ALU_HINT_DISCARD,
       {0x27, 0x07, 0x56, 0x00, 0x03, 0x00, 0xac, 0x48, 0x03, 0x00}},
      {true, true, 0, 6, 32, AGX3_ALU_HINT_NONE,
       {0x27, 0x07, 0x56, 0x00, 0x03, 0x0c, 0x8c, 0x48, 0x03, 0x00}},
      {true, false, 4, 1, 32, AGX3_ALU_HINT_DISCARD,
       {0x27, 0x07, 0x56, 0x04, 0x02, 0x02, 0xac, 0x48, 0x03, 0x00}},
      {true, false, 0, 0, 1, AGX3_ALU_HINT_DISCARD,
       {0x27, 0x17, 0x54, 0x00, 0x02, 0x00, 0xac, 0x48, 0x03, 0x00}},
   };

   uint8_t out[AGX3_ALU_MAX_LENGTH];
   uint8_t len = 0;
   for (unsigned i = 0; i < sizeof(native) / sizeof(native[0]); ++i) {
      assert(agx3_alu_f16_to_int32(
         AGX_ISA_ENCODING_AGX3B, native[i].dest, native[i].src,
         native[i].sign, native[i].hint, native[i].cache,
         native[i].wait, out, &len));
      assert(len == sizeof(native[i].bytes));
      assert(!memcmp(out, native[i].bytes, len));
   }

   /* Exercise the complete admitted Cartesian product, including the upper
    * register boundary and every producer wait combination. */
   for (unsigned sign = 0; sign < 2; ++sign)
      for (unsigned hint = AGX3_ALU_HINT_NONE;
           hint <= AGX3_ALU_HINT_DISCARD; ++hint)
         for (unsigned cache = 0; cache < 2; ++cache)
            for (unsigned wait = 0; wait < 64; ++wait) {
               assert(agx3_alu_f16_to_int32(
                  AGX_ISA_ENCODING_AGX3B, 126, 127, sign, hint, cache,
                  wait, out, &len));
               assert(len == 10);
            }

   const uint8_t sentinel[AGX3_ALU_MAX_LENGTH] = {
      0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
      0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5, 0xa5,
   };
#define REJECT(...)                                                            \
   do {                                                                        \
      memcpy(out, sentinel, sizeof(out));                                      \
      len = 0x5a;                                                              \
      assert(!agx3_alu_f16_to_int32(__VA_ARGS__));                             \
      assert(!memcmp(out, sentinel, sizeof(out)) && len == 0x5a);              \
   } while (0)
   REJECT(AGX_ISA_ENCODING_AGX3B, 128, 0, true,
          AGX3_ALU_HINT_NONE, false, 0, out, &len);
   REJECT(AGX_ISA_ENCODING_AGX3B, 0, 128, true,
          AGX3_ALU_HINT_NONE, false, 0, out, &len);
   REJECT(AGX_ISA_ENCODING_AGX3B, 1, 0, true,
          AGX3_ALU_HINT_NONE, false, 0, out, &len);
   REJECT(AGX_ISA_ENCODING_AGX3B, 0, 0, true,
          AGX3_ALU_HINT_CACHE, false, 0, out, &len);
   REJECT(AGX_ISA_ENCODING_AGX3B, 0, 0, true,
          (enum agx3_alu_hint)3, false, 0, out, &len);
   REJECT(AGX_ISA_ENCODING_AGX3B, 0, 0, true,
          AGX3_ALU_HINT_NONE, false, 64, out, &len);
   assert(!agx3_alu_f16_to_int32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, true, AGX3_ALU_HINT_NONE,
      false, 0, NULL, &len));
   assert(!agx3_alu_f16_to_int32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, true, AGX3_ALU_HINT_NONE,
      false, 0, out, NULL));
#undef REJECT

   puts("F16-to-int32: eight exact native vectors, 512 legal forms and "
        "strict refusals PASS");
}

static void
test_native_fma_minifloat(void)
{
   /* Fresh applegpu_g17g fma(x, k, +/-y), including both live operands. */
   static const struct {
      uint16_t dest, src, add;
      uint8_t imm, wait;
      bool dc, neg;
      enum agx3_alu_hint sh, ah;
      const char *hex;
   } g17g[] = {
      {0,0,8,32,128,0,0,1,1,"2900070a2020a3023088980008080080"},
      {2,2,10,32,0,0,0,1,1,"19030ea1a00a"},
      {12,4,12,32,128,1,0,2,1,"2980071a2820a31a108898000c080080"},
      {14,6,14,32,0,1,0,2,1,"798726a1a00e"},
      {12,12,4,32,128,1,0,1,2,"2900073a2820a31a3088980004040080"},
      {14,14,6,32,0,1,0,1,2,"790f2ea12086"},
      {0,0,8,32,128,0,1,1,1,"2900070a2020a30230889800080800c0"},
      {2,2,10,32,0,0,1,1,1,"19030ea1910a0202"},
      {0,0,8,64,128,0,0,1,1,"2900070a2020a3023090980008080080"},
      {2,2,10,64,0,0,0,1,1,"19030ec1a00a"},
   };
   for (unsigned i = 0; i < sizeof(g17g) / sizeof(g17g[0]); ++i) {
      uint8_t out[16], len;
      assert(agx3_alu_fma_minifloat_mod(
         AGX_ISA_ENCODING_AGX3C, g17g[i].dest, g17g[i].src, g17g[i].imm,
         g17g[i].add, g17g[i].neg, false, g17g[i].sh, g17g[i].ah, g17g[i].dc,
         g17g[i].wait, out, &len));
      exact(out, len, g17g[i].hex);
   }
   /* M4 applegpu_g16g offline compiler: saturate(fma(-8, src.x, src.y)).
    * Native text+0x52, multiplier 0xe0, result saturation field14. */
   {
      uint8_t out[16], len;
      assert(agx3_alu_fma_minifloat_mod(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 224, 2, false, true,
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false, 0x20, out, &len));
      exact(out, len, "09011ee1820202020082");
   }
   {
      static const struct {
         uint16_t dest, src, addend;
         uint8_t immediate;
         enum agx3_alu_hint src_hint, addend_hint;
         const char *bytes;
      } negative_addend[] = {
         {0, 0, 2, 64, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
          "09010ec1910202c2"},
         {0, 2, 4, 64, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
          "090306c1110402c2"},
         {0, 0, 2, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
          "09010ea1910202c2"},
         {0, 2, 4, 32, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
          "090306a1110402c2"},
      };
      for (unsigned i = 0;
           i < sizeof(negative_addend) / sizeof(negative_addend[0]); ++i) {
         uint8_t out[16], len;
         assert(agx3_alu_fma_minifloat_mod(
            AGX_ISA_ENCODING_AGX3B, negative_addend[i].dest,
            negative_addend[i].src, negative_addend[i].immediate,
            negative_addend[i].addend, true, false, negative_addend[i].src_hint,
            negative_addend[i].addend_hint, false, 0, out, &len));
         exact(out, len, negative_addend[i].bytes);
      }
   }
   {const uint8_t native[]={0x49,0x11,0x2e,0xc1,0x81,0x08,0x02,0x42};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,8,16,64,8,1,1,1,2,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x59,0x13,0x2e,0xc1,0xa0,0x0a};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,10,18,64,10,1,1,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x09,0x01,0x2e,0xd1,0x81,0x08,0x02,0x22};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,0,0,80,8,1,1,1,1,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x19,0x03,0x2e,0xd1,0xa0,0x0a};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,2,2,80,10,1,1,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x19,0x15,0x2e,0xc1,0xa0,0x0c};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,2,20,64,12,1,1,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x49,0x17,0x2e,0xc1,0xa0,0x0e};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,8,22,64,14,1,1,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x19,0x05,0x2e,0xd1,0xa0,0x02};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,2,4,80,2,1,1,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x29,0x07,0x2e,0xd1,0xa0,0x08};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,4,6,80,8,1,1,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x49,0x09,0x2e,0xc1,0x01,0x80,0x02,0x42};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,8,8,64,0,1,2,1,2,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x59,0x0b,0x2e,0xc1,0x20,0x82};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,10,10,64,2,1,2,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x19,0x0d,0x2e,0xc1,0x20,0x84};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,2,12,64,4,1,2,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x49,0x0f,0x2e,0xc1,0x20,0x86};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,8,14,64,6,1,2,1,0,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x09,0x01,0x0e,0xc1,0x81,0x04,0x02,0xc2};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,0,0,64,4,1,1,0,32,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   {const uint8_t native[]={0x09,0x83,0x06,0xd1,0x01,0x86,0x02,0xc2};uint8_t out[16],len;
    assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,0,2,80,6,2,2,0,32,out,&len));
    assert(len==sizeof(native) && !memcmp(out,native,len));}
   for(unsigned wait=0;wait<64;wait++) {
      uint8_t out[16],len;
      assert(agx3_alu_fma_minifloat(AGX_ISA_ENCODING_AGX3B,4,24,64,4,1,1,false,wait,out,&len));
      assert(len==6 || len==8 || len==10);
      if(len==10) {
         /* Extended FMA physical wait bits: slots1..3 at61..63 and
          * slots4..6 at77..79, independent of the composer. */
         const unsigned decoded=(out[7]>>5)|((out[9]>>5)<<3);
         assert(decoded==wait);
      }
   }
   puts("native FMA minifloat:14 byte vectors and64 wait combinations PASS");
}

static void test_native_u16_arithmetic(void)
{
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   const uint16_t xy[] = {3, 4};
   const enum agx3_alu_hint cache[] = {AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE};
   const enum agx3_alu_hint dead[] = {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
   assert(agx3_alu_bit_reg16(AGX_ISA_ENCODING_AGX3B,AGX3_ALU_BAND,2,xy,cache,32,out,&len));
   exact(out,len,"13820784080000800000");
   assert(agx3_alu_bit_reg16(AGX_ISA_ENCODING_AGX3B,AGX3_ALU_BOR,2,xy,cache,32,out,&len));
   exact(out,len,"138207840a0800800000");
   assert(agx3_alu_bit_reg16(AGX_ISA_ENCODING_AGX3B,AGX3_ALU_BXOR,3,xy,dead,0,out,&len));
   exact(out,len,"13021e040e0800000000");
   const uint16_t t[] = {16, 16}, n[] = {11, 10};
   assert(agx3_alu_shift_add16(AGX_ISA_ENCODING_AGX3B,10,t,dead,4,true,2,out,&len));
   exact(out,len,"9f21540a032040001605");
   assert(agx3_alu_shift_add16(AGX_ISA_ENCODING_AGX3B,10,n,dead,4,true,0,out,&len));
   exact(out,len,"9f01540a031628001605");
   const uint16_t sll_dead_src[] = {0, 1};
   const bool sll_half_src[] = {true, true};
   assert(agx3_alu_sll_reg(AGX_ISA_ENCODING_AGX3B, 0, true, true,
                           sll_dead_src, sll_half_src, dead, 2, out, &len));
   exact(out, len, "272054000300040080130500");
   const uint16_t sll_mixed_src[] = {2, 0};
   const bool sll_mixed_width[] = {true, false};
   const enum agx3_alu_hint sll_mixed_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_sll_reg(AGX_ISA_ENCODING_AGX3B, 0, false, true,
                           sll_mixed_src, sll_mixed_width, sll_mixed_hints,
                           2, out, &len));
   exact(out, len, "2720540003040200d0120500");
   const enum agx3_alu_hint sll_bad_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
   };
   assert(!agx3_alu_sll_reg(AGX_ISA_ENCODING_AGX3B, 0, true, false,
                            sll_mixed_src, sll_mixed_width, sll_bad_hints,
                            0, out, &len));
   /* applegpu_g17p `ushort(1) << ushort(n & 15u)` over uint counts. */
   assert(agx3_alu_sll_one_reg32(AGX_ISA_ENCODING_AGX3B, 2, 2,
                                 AGX3_ALU_HINT_DISCARD, true, 0,
                                 out, &len));
   exact(out, len, "270054020302080080120400");
   assert(agx3_alu_sll_one_reg32(AGX_ISA_ENCODING_AGX3B, 4, 0,
                                 AGX3_ALU_HINT_NONE, false, 0,
                                 out, &len));
   exact(out, len, "270054040202000080100400");
   assert(agx3_alu_sll_one_reg32(AGX_ISA_ENCODING_AGX3B, 0, 0,
                                 AGX3_ALU_HINT_DISCARD, false, 0,
                                 out, &len));
   exact(out, len, "270054000202000080120400");
   assert(agx3_alu_sll_one_reg32(AGX_ISA_ENCODING_AGX3B, 0, 2,
                                 AGX3_ALU_HINT_DISCARD, false, 1,
                                 out, &len));
   exact(out, len, "271054000202080080120400");
   assert(!agx3_alu_sll_one_reg32(AGX_ISA_ENCODING_AGX3B, 0, 3,
                                  AGX3_ALU_HINT_DISCARD, false, 1,
                                  out, &len));
   assert(agx3_alu_srl_reg(AGX_ISA_ENCODING_AGX3B, 0, false, false,
                           sll_mixed_src, sll_mixed_width, sll_mixed_hints,
                           2, out, &len));
   exact(out, len, "a720540002040200d0120500");
   assert(!agx3_alu_srl_reg(AGX_ISA_ENCODING_AGX3B, 0, true, false,
                            sll_mixed_src, sll_mixed_width, sll_bad_hints,
                            0, out, &len));
   const uint16_t asr_src[] = {4, 2};
   assert(agx3_alu_asr_reg16(AGX_ISA_ENCODING_AGX3B, 2, asr_src[0],
                             asr_src[1], dead, false, 1, out, &len));
   exact(out, len, "a7115402020808e0e200");
   /* applegpu_g17p `short(x >> (n & 15u))` with x or n still stored
    * afterwards, x cached for a later xor, and a cached result. */
   {
      const struct {
         uint16_t dest, value, count;
         enum agx3_alu_hint hints[2];
         bool dest_cache;
         const char *apple;
      } live[] = {
         {2, 4, 2, {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD}, false,
          "a7115402020808a0e200"},
         {4, 4, 0, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE}, false,
          "a711540402080060e200"},
         {2, 5, 2, {AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_DISCARD}, false,
          "a7115402020a0aa0e200"},
         {3, 5, 2, {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD}, true,
          "a7115403030a08e0e200"},
      };
      for (unsigned i = 0; i < sizeof(live) / sizeof(live[0]); ++i) {
         assert(agx3_alu_asr_reg16(AGX_ISA_ENCODING_AGX3B, live[i].dest,
                                   live[i].value, live[i].count,
                                   live[i].hints, live[i].dest_cache, 1,
                                   out, &len));
         exact(out, len, live[i].apple);
      }
   }
   assert(!agx3_alu_asr_reg16(AGX_ISA_ENCODING_AGX3B, 2, asr_src[0], 3,
                              dead, false, 1, out, &len));
   assert(!agx3_alu_asr_reg16(AGX_ISA_ENCODING_AGX3C, 2, asr_src[0],
                              asr_src[1], dead, false, 1, out, &len));
   const uint16_t mixed_src[] = {2, 4};
   const bool mixed_width[] = {false, true};
   assert(agx3_alu_fcmp_select_imm_width(AGX_ISA_ENCODING_AGX3B,2,mixed_src,mixed_width,dead,1,255,0,true,false,32,out,&len));
   exact(out,len,"12031d04a2ff03c02080");
   const uint16_t half_src[] = {0, 1};
   const bool half_width[] = {true, true};
   assert(agx3_alu_fcmp_select_imm_width(
      AGX_ISA_ENCODING_AGX3B, 0, half_src, half_width, dead, 1, 2, 0,
      false, false, 32, out, &len));
   exact(out, len, "02001f00228213c02080");
   assert(agx3_alu_fcmp_select_imm_width(
      AGX_ISA_ENCODING_AGX3B, 0, half_src, half_width, dead, 1, 7, 3,
      false, false, 32, out, &len));
   exact(out, len, "02001f00228713c02083");
   const uint16_t neg_src[] = {8, 0};
   const bool full_width[] = {false, false};
   assert(agx3_alu_fcmp_select_imm_width_neg_second(
      AGX_ISA_ENCODING_AGX3B, 0, neg_src, full_width, dead,
      2, 255, 0, true, false, 32, out, &len));
   exact(out, len, "02091d01a2ff02c82080");
   /* f65 is an ordinary field, not a property of one witnessed shape: it
    * composes with every relation and with a 32-bit result. These two used
    * to be refusals here, on the strength of nothing having been measured
    * rather than of anything having been ruled out. */
   assert(agx3_alu_fcmp_select_imm_width_neg_second(
      AGX_ISA_ENCODING_AGX3B, 0, neg_src, full_width, dead,
      1, 255, 0, true, false, 32, out, &len));
   exact(out, len, "02091d01a2ff03c82080");
   assert(agx3_alu_fcmp_select_imm_width_neg_second(
      AGX_ISA_ENCODING_AGX3B, 0, neg_src, full_width, dead,
      2, 255, 0, false, false, 32, out, &len));
   exact(out, len, "02091f01a2ff02c82080");
   puts("Native u16 ALU and mixed-width FSEL: six native byte vectors PASS");
}

/* applegpu_g17p `~ushort` and `~short` in fragments and kernels whose result
 * is stored once or read again, whose input is dead, cached or retained, and
 * which wait on slots 1, 2 and 6. */
static void
test_native_bit_not16(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   const enum agx3_alu_hint D = AGX3_ALU_HINT_DISCARD;
   const enum agx3_alu_hint C = AGX3_ALU_HINT_CACHE;
   const enum agx3_alu_hint N = AGX3_ALU_HINT_NONE;
   const struct {
      uint16_t dest, src;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *expected;
   } witnesses[] = {
      {0, 0, D, true, 0, "03002e00010800000000"},
      {0, 0, D, true, 1u << 0, "03002e00012800000000"},
      {1, 0, C, true, 1u << 0, "03802600052800000000"},
      {1, 2, C, true, 0, "03822600050800000000"},
      {3, 0, C, true, 0, "13802600050800000000"},
      {2, 2, D, true, 1u << 5, "13022e00010800800000"},
      {2, 2, D, false, 1u << 5, "13020e00010800800000"},
      {3, 3, D, false, 0, "13020e000d0800000000"},
      {2, 2, D, false, 0, "13020e00010800000000"},
      {3, 2, C, true, 1u << 5, "13822600050800800000"},
      {0, 2, C, false, 1u << 5, "03820600010800800000"},
      {3, 2, N, false, 1u << 5, "13020600050800800000"},
      {1, 1, D, false, 1u << 0, "03000e000d2800000000"},
      {2, 2, D, false, 1u << 1, "13020e00014800000000"},
   };
   for (unsigned i = 0; i < sizeof(witnesses) / sizeof(witnesses[0]); ++i) {
      assert(agx3_alu_bit_not_reg16(AGX_ISA_ENCODING_AGX3B, witnesses[i].dest,
                                    witnesses[i].src, witnesses[i].hint,
                                    witnesses[i].dest_cache,
                                    witnesses[i].wait, bytes, &len));
      exact(bytes, len, witnesses[i].expected);
   }
   assert(!agx3_alu_bit_not_reg16(AGX_ISA_ENCODING_AGX3B, 2, 2, D, false,
                                  1u << 6, bytes, &len));
   assert(!agx3_alu_bit_not_reg16(AGX_ISA_ENCODING_AGX3C, 0, 0, N, false, 0,
                                  bytes, &len));
}

/* applegpu_g17p `short(a - b)` and `short(-a)` with dead, cached and retained
 * inputs, a cached result, and wait slots 1, 2 and 6. */
static void
test_native_sub16(void)
{
   const enum agx_isa_encoding B = AGX_ISA_ENCODING_AGX3B;
   const enum agx3_alu_hint D = AGX3_ALU_HINT_DISCARD;
   const enum agx3_alu_hint C = AGX3_ALU_HINT_CACHE;
   const enum agx3_alu_hint N = AGX3_ALU_HINT_NONE;
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;

   assert(agx3_alu_sub16(B, 2, (uint16_t[]){2, 3},
                         (enum agx3_alu_hint[]){D, D}, false, 1u << 5,
                         out, &len));
   exact(out, len, "1f01560202040c001705");
   assert(agx3_alu_sub16(B, 0, (uint16_t[]){2, 3},
                         (enum agx3_alu_hint[]){C, C}, false, 1u << 5,
                         out, &len));
   exact(out, len, "1f01560002040e041105");
   assert(agx3_alu_sub16(B, 4, (uint16_t[]){3, 2},
                         (enum agx3_alu_hint[]){N, N}, false, 1u << 5,
                         out, &len));
   exact(out, len, "1f015604020608001105");
   assert(agx3_alu_sub16(B, 0, (uint16_t[]){3, 1},
                         (enum agx3_alu_hint[]){C, C}, false, 1u << 5,
                         out, &len));
   exact(out, len, "1f015600020606041105");
   assert(agx3_alu_sub16(B, 1, (uint16_t[]){1, 2},
                         (enum agx3_alu_hint[]){D, C}, false, 1u << 0,
                         out, &len));
   exact(out, len, "1f115401020208041305");
   assert(agx3_alu_sub16(B, 2, (uint16_t[]){2, 3},
                         (enum agx3_alu_hint[]){D, D}, false, 0, out, &len));
   exact(out, len, "1f01540202040c001705");
   assert(agx3_alu_sub16(B, 3, (uint16_t[]){2, 3},
                         (enum agx3_alu_hint[]){D, D}, true, 1u << 5,
                         out, &len));
   exact(out, len, "1f01560303040c001705");
   assert(!agx3_alu_sub16(B, 192, (uint16_t[]){2, 3},
                          (enum agx3_alu_hint[]){D, D}, false, 0, out, &len));
   assert(!agx3_alu_sub16(B, 2, (uint16_t[]){2, 192},
                          (enum agx3_alu_hint[]){D, D}, false, 0, out, &len));

   assert(agx3_alu_neg16(B, 2, 2, D, false, 1u << 5, out, &len));
   exact(out, len, "1f015602020008001504");
   assert(agx3_alu_neg16(B, 2, 3, C, false, 1u << 5, out, &len));
   exact(out, len, "1f01560202000c041104");
   assert(agx3_alu_neg16(B, 3, 2, N, false, 1u << 5, out, &len));
   exact(out, len, "1f015603020008001104");
   assert(agx3_alu_neg16(B, 0, 0, D, false, 1u << 5, out, &len));
   exact(out, len, "1f015600020000001504");
   assert(agx3_alu_neg16(B, 1, 1, D, false, 1u << 0, out, &len));
   exact(out, len, "1f115401020004001504");
   assert(agx3_alu_neg16(B, 2, 2, D, false, 1u << 1, out, &len));
   exact(out, len, "1f215402020008001504");
   assert(agx3_alu_neg16(B, 3, 2, D, true, 1u << 5, out, &len));
   exact(out, len, "1f015603030008001504");
}

/* applegpu_g17p `addsat` of shorts and ushorts with dead, cached and
 * retained inputs, a cached result, and wait slots 1 and 6. */
static void
test_native_add_sat16(void)
{
   const enum agx3_alu_hint D = AGX3_ALU_HINT_DISCARD;
   const enum agx3_alu_hint C = AGX3_ALU_HINT_CACHE;
   const enum agx3_alu_hint N = AGX3_ALU_HINT_NONE;
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   const struct {
      uint16_t dest, src[2];
      enum agx3_alu_hint hints[2];
      bool is_signed, dest_cache;
      uint16_t wait;
      const char *apple;
   } vectors[] = {
      {2, {2, 3}, {D, D}, true, false, 1u << 5, "9f01560202040c40170f"},
      {2, {3, 2}, {D, D}, true, false, 0, "9f01540202060840170f"},
      {0, {2, 3}, {C, C}, true, false, 1u << 5, "9f01560002040e44110f"},
      {4, {3, 2}, {N, N}, true, false, 1u << 5, "9f01560402060840110f"},
      {3, {2, 3}, {D, D}, true, true, 1u << 5, "9f01560303040c40170f"},
      {0, {3, 1}, {C, C}, true, false, 1u << 5, "9f01560002060644110f"},
      {1, {1, 2}, {D, C}, true, false, 1u << 0, "9f11540102020844130f"},
      {2, {2, 3}, {D, D}, false, false, 1u << 5, "9f01560202040c401705"},
      {0, {2, 3}, {C, C}, false, false, 1u << 5, "9f01560002040e441105"},
      {4, {3, 2}, {N, N}, false, false, 1u << 5, "9f015604020608401105"},
      {3, {2, 3}, {D, D}, false, true, 1u << 5, "9f01560303040c401705"},
      {0, {3, 1}, {C, C}, false, false, 1u << 5, "9f015600020606441105"},
      {1, {1, 2}, {D, C}, false, false, 1u << 0, "9f115401020208441305"},
      {2, {2, 3}, {D, D}, false, false, 0, "9f01540202040c401705"},
   };
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_add_sat16(AGX_ISA_ENCODING_AGX3B, vectors[i].dest,
                                vectors[i].src, vectors[i].hints,
                                vectors[i].is_signed, vectors[i].dest_cache,
                                vectors[i].wait, out, &len));
      exact(out, len, vectors[i].apple);
   }
   assert(!agx3_alu_add_sat16(AGX_ISA_ENCODING_AGX3B, 192, (uint16_t[]){2, 3},
                              (enum agx3_alu_hint[]){D, D}, true, false, 0,
                              out, &len));
   assert(!agx3_alu_add_sat16(AGX_ISA_ENCODING_AGX3C, 2, (uint16_t[]){2, 3},
                              (enum agx3_alu_hint[]){D, D}, true, false, 0,
                              out, &len));
}

/* applegpu_g17p `half(char)`, `float(char)`, `float(uchar)`, `half(short)`
 * and `half(ushort)`, including a byte read from a wider half register. */
static void
test_native_narrow_int_to_float(void)
{
   const enum agx_isa_encoding B = AGX_ISA_ENCODING_AGX3B;
   const enum agx3_alu_hint D = AGX3_ALU_HINT_DISCARD;
   const enum agx3_alu_hint C = AGX3_ALU_HINT_CACHE;
   const enum agx3_alu_hint N = AGX3_ALU_HINT_NONE;
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   const struct {
      uint16_t dest;
      bool dest_16;
      uint16_t src;
      bool src_8, is_signed;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *apple;
   } vectors[] = {
      {2, true, 2, true, true, D, false, 1u << 5, "a70756020204b070"},
      {2, true, 2, true, true, D, false, 0, "a70754020204b070"},
      {2, true, 4, true, true, C, false, 0, "a707540202089270"},
      {3, true, 2, true, true, N, false, 0, "a707540302049070"},
      {3, true, 2, true, true, D, true, 0, "a70754030304b070"},
      {0, true, 0, true, true, D, false, 1u << 5, "a70756000200b070"},
      {1, true, 1, true, true, D, false, 1u << 0, "a71754010202b070"},
      {2, true, 2, true, true, D, false, 1u << 1, "a72754020204b070"},
      {0, false, 0, true, true, D, false, 1u << 5, "a70756000200b470"},
      {0, false, 0, true, true, D, false, 0, "a70754000200b470"},
      {2, false, 7, true, true, C, true, 0, "a7075402030e9670"},
      {2, true, 2, false, true, D, false, 1u << 5, "a70756020204a060"},
      {3, true, 3, false, true, D, false, 0, "a70754030206a060"},
      {3, true, 2, false, true, D, true, 1u << 5, "a70756030304a060"},
      {1, true, 1, false, true, D, false, 1u << 0, "a71754010202a060"},
      {2, true, 2, false, false, D, false, 1u << 5, "a70756020204a020"},
      {2, true, 3, false, false, C, false, 1u << 5, "a707560202068220"},
      {3, true, 3, false, false, D, false, 0, "a70754030206a020"},
      {0, false, 0, true, false, D, false, 0, "a70754000200b430"},
      {2, false, 4, true, false, N, false, 0, "a707540202089430"},
   };
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_narrow_int_to_float(
         B, vectors[i].dest, vectors[i].dest_16, vectors[i].src,
         vectors[i].src_8, vectors[i].is_signed, vectors[i].hint,
         vectors[i].dest_cache, vectors[i].wait, out, &len));
      exact(out, len, vectors[i].apple);
   }
   /* No unsigned byte to half form, odd float32 destination, or high
    * register. */
   assert(!agx3_alu_narrow_int_to_float(B, 2, true, 2, true, false, D, false,
                                        0, out, &len));
   assert(!agx3_alu_narrow_int_to_float(B, 3, false, 2, true, true, D, false,
                                        0, out, &len));
   assert(!agx3_alu_narrow_int_to_float(B, 2, true, 192, false, true, D,
                                        false, 0, out, &len));
}

static void
test_native_u16_shift_right_immediate(void)
{
   static const struct {
      uint16_t dest, src, wait;
      enum agx3_alu_hint hint;
      bool dest_cache;
      const char *bytes;
   } witnesses[] = {
      /* Fresh applegpu_g17p [[sample_mask]] extraction. */
      {0, 0, 1, AGX3_ALU_HINT_DISCARD, true,
       "a710540003001000c0110100"},
      /* Apple's constant sample-mask fragment path. */
      {1, 0, 1, AGX3_ALU_HINT_CACHE, true,
       "a710540103001200c0100100"},
      /* Apple's per-sample fragment path. */
      {15, 14, 2, AGX3_ALU_HINT_CACHE, true,
       "a720540f031c1200c0100100"},
   };
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;

   for (unsigned i = 0; i < sizeof(witnesses) / sizeof(witnesses[0]); ++i) {
      assert(agx3_alu_srl_imm16(
         AGX_ISA_ENCODING_AGX3B, witnesses[i].dest, witnesses[i].src, 4,
         witnesses[i].hint, witnesses[i].dest_cache, witnesses[i].wait, out,
         &len));
      exact(out, len, witnesses[i].bytes);
   }

   assert(!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3B, 0, 0, 0,
                              AGX3_ALU_HINT_NONE, false, 0, out, &len));
   assert(!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3B, 0, 0, 32,
                              AGX3_ALU_HINT_NONE, false, 0, out, &len));
   assert(!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3B, 256, 0, 4,
                              AGX3_ALU_HINT_NONE, false, 0, out, &len));
   assert(!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3B, 0, 256, 4,
                              AGX3_ALU_HINT_NONE, false, 0, out, &len));
   assert(!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3B, 0, 0, 4,
                              (enum agx3_alu_hint)3, false, 0, out, &len));
   assert(!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3B, 0, 0, 4,
                              AGX3_ALU_HINT_NONE, false, 1u << 6, out,
                              &len));
   puts("Native u16 immediate SRL: three exact Apple vectors and refusals PASS");
}

static void
test_blur_half_arithmetic(void)
{
   uint8_t out[AGX3_ALU_MAX_LENGTH],len;
   assert(agx3_alu_s16_to_f32(AGX_ISA_ENCODING_AGX3B,0,0,AGX3_ALU_HINT_DISCARD,32,out,&len));
   exact(out,len,"a70756000200a460");
   assert(agx3_alu_s16_to_f32(AGX_ISA_ENCODING_AGX3B,2,4,AGX3_ALU_HINT_NONE,32,out,&len));
   exact(out,len,"a707560202088460");
   assert(agx3_alu_s16_to_f32(AGX_ISA_ENCODING_AGX3B,0,2,AGX3_ALU_HINT_DISCARD,32,out,&len));
   exact(out,len,"a70756000204a460");
   assert(agx3_alu_s16_to_f32(AGX_ISA_ENCODING_AGX3B,2,3,AGX3_ALU_HINT_DISCARD,0,out,&len));
   exact(out,len,"a70754020206a460");
   assert(agx3_alu_add_imm16(AGX_ISA_ENCODING_AGX3B,2,2,2,AGX3_ALU_HINT_DISCARD,false,32,out,&len));
   exact(out,len,"9f015602020408001504");
   assert(agx3_alu_add_imm16(AGX_ISA_ENCODING_AGX3B,2,3,2,AGX3_ALU_HINT_NONE,false,32,out,&len));
   exact(out,len,"9f01560202040c001104");
   assert(agx3_alu_add_imm16(AGX_ISA_ENCODING_AGX3B,3,3,17,AGX3_ALU_HINT_DISCARD,false,0,out,&len));
   exact(out,len,"9f01540302220c001504");
   assert(agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,2,2,1,
                            AGX3_ALU_HINT_DISCARD,false,32,out,&len));
   exact(out,len,"1f015602020404001301");
   assert(agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,2,3,2,
                            AGX3_ALU_HINT_CACHE,false,32,out,&len));
   exact(out,len,"1f015602020608001101");
   assert(agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,3,3,17,
                            AGX3_ALU_HINT_DISCARD,false,0,out,&len));
   exact(out,len,"1f015403020644001301");
   assert(agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,2,2,255,
                            AGX3_ALU_HINT_DISCARD,true,0,out,&len));
   exact(out,len,"1f0154020304fc031301");
   assert(!agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3C,2,2,2,
                             AGX3_ALU_HINT_DISCARD,false,0,out,&len));
   assert(!agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,2,2,0,
                             AGX3_ALU_HINT_DISCARD,false,0,out,&len));
   assert(agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,2,3,2,
                            AGX3_ALU_HINT_NONE,false,32,out,&len));
   exact(out,len,"1f015602020608001101");
   assert(!agx3_alu_sub_imm16(AGX_ISA_ENCODING_AGX3B,2,2,2,
                             AGX3_ALU_HINT_DISCARD,false,1,out,&len));
   assert(agx3_alu_f32_to_s16(AGX_ISA_ENCODING_AGX3B,2,2,
                              AGX3_ALU_HINT_DISCARD,false,32,out,&len));
   exact(out,len,"270756020204b0c80200");
   assert(agx3_alu_f32_to_s16(AGX_ISA_ENCODING_AGX3B,1,2,
                              AGX3_ALU_HINT_NONE,false,0,out,&len));
   exact(out,len,"27075401020490c80200");
   assert(agx3_alu_f32_to_s16(AGX_ISA_ENCODING_AGX3B,2,0,
                              AGX3_ALU_HINT_CACHE,true,0,out,&len));
   exact(out,len,"27075402030090c80200");
   assert(!agx3_alu_f32_to_s16(AGX_ISA_ENCODING_AGX3B,2,1,
                               AGX3_ALU_HINT_NONE,false,0,out,&len));
   puts("Blur half arithmetic: fourteen exact native conversion/add/sub witnesses PASS");
}

static void test_native_extract32(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],length;
   const unsigned shifts[]={1,4,16,28,31};
   const char *golden[5][2]={
      {"2701560002040400f0150900","2701560002080410f0100900"},
      {"2701560002041000f0150900","2701560002081010f0100900"},
      {"2701560002044000f0150900","2701560002084010f0100900"},
      {"2701560002047000f0150900","2701560002087010f0100900"},
      {"2701560002047c00f0150900","2701560002087c10f0100900"},
   };
   for(unsigned n=0;n<5;n++)for(unsigned live=0;live<2;live++) {
      const uint16_t src[2]={live?2:0,live?4:2};
      const enum agx3_alu_hint hints[2]={live?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD,live?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD};
      assert(agx3_alu_extract_imm32(AGX_ISA_ENCODING_AGX3B,0,src,shifts[n],hints,false,32,bytes,&length));
      exact(bytes,length,golden[n][live]);
   }
   puts("Native EXTRACT32:10 exact Apple immediate/live/dead vectors PASS");
   /* Independent M5 table vector from Zink's descriptor-address extract:
    * discard low half-pair8, retain high half-pair10, wait for slot7. */
   const uint16_t m5_src[2] = {8,10};
   const enum agx3_alu_hint m5_hints[2] = {AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE};
   assert(agx3_alu_extract_imm32(AGX_ISA_ENCODING_AGX3C,8,m5_src,2,m5_hints,
                                false,64,bytes,&length));
   exact(bytes,length,"2782401222a0a21280805c000808");
   assert(!agx3_alu_extract_imm32(AGX_ISA_ENCODING_AGX3C,8,m5_src,2,m5_hints,
                                 false,256,bytes,&length));
   const enum agx3_alu_hint field_hints[2] = {AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD};
   assert(agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3C,20,10,20,true,14,field_hints,
                         0,bytes,&length));
   exact(bytes,length,"a782005a2200a10a0885203400e0");
   assert(!agx3_alu_bfeil(AGX_ISA_ENCODING_AGX3C,20,10,20,true,32,field_hints,
                          0,bytes,&length));
}

static void test_native_asr32(void)
{
   const unsigned shifts[]={1,4,28};
   const char *golden[3][2]={
      {"a7015600020004786200","a7015600020404386200"},
      {"a7015600020010786200","a7015600020410386200"},
      {"a7015600020070786200","a7015600020470386200"},
   };
   for(unsigned n=0;n<3;n++)for(unsigned live=0;live<2;live++) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],length;
      assert(agx3_alu_asr_imm32(AGX_ISA_ENCODING_AGX3B,0,live?2:0,shifts[n],
                                live?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD,
                                false,32,bytes,&length));
      exact(bytes,length,golden[n][live]);
   }
   puts("Native ASR32:6 exact Apple immediate/live/dead vectors PASS");
   /* Independent M5 native-composer vectors. The first discarded-source
    * case is Xorg glamor's vertex-ID shift before integer-to-float conversion.
    * Also cover retained sources, cached destinations, waits and high regs. */
   const struct {
      uint16_t dest, source;
      unsigned shift;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *bytes;
   } m5[] = {
      {0,0,1,AGX3_ALU_HINT_DISCARD,false,0,"2700001a2600a30278804000"},
      {0,0,31,AGX3_ALU_HINT_NONE,false,0,"2700001a2200a302f8874000"},
      {4,6,7,AGX3_ALU_HINT_NONE,true,32,"2702003a2a00a38af8814000"},
      {190,188,16,AGX3_ALU_HINT_CACHE,true,63,"b7fc08fa2a80a3ba3a844000"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
   for (unsigned n = 0; n < sizeof(m5) / sizeof(m5[0]); ++n) {
      assert(agx3_alu_asr_imm32(AGX_ISA_ENCODING_AGX3C,
                                m5[n].dest,m5[n].source,m5[n].shift,m5[n].hint,
                                m5[n].dest_cache,m5[n].wait,bytes,&length));
      exact(bytes,length,m5[n].bytes);
   }
   assert(!agx3_alu_asr_imm32(AGX_ISA_ENCODING_AGX3C,0,0,0,
                              AGX3_ALU_HINT_NONE,false,0,bytes,&length));
   assert(!agx3_alu_asr_imm32(AGX_ISA_ENCODING_AGX3C,0,0,32,
                              AGX3_ALU_HINT_NONE,false,0,bytes,&length));
   assert(!agx3_alu_asr_imm32(AGX_ISA_ENCODING_AGX3C,1,0,1,
                              AGX3_ALU_HINT_NONE,false,0,bytes,&length));
   assert(!agx3_alu_asr_imm32(AGX_ISA_ENCODING_AGX3C,0,0,1,
                              AGX3_ALU_HINT_NONE,false,64,bytes,&length));
   puts("M5 ASR32:4 native vectors and invalid operands PASS");
}

static void test_native_asr16(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
   assert(agx3_alu_asr_imm16(AGX_ISA_ENCODING_AGX3B,2,2,1,
                              AGX3_ALU_HINT_DISCARD,false,32,bytes,&length));
   exact(bytes,length,"a7015602020404606200");
   assert(!agx3_alu_asr_imm16(AGX_ISA_ENCODING_AGX3B,2,2,16,
                               AGX3_ALU_HINT_DISCARD,false,32,bytes,&length));
   puts("Native ASR16 exact Apple immediate/dead vector PASS");
}

static void test_native_narrow_multiply(void)
{
   const uint16_t src[]={0,2};
   for (unsigned live=0;live<2;live++) {
      enum agx3_alu_hint hint=live?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD;
      const enum agx3_alu_hint hints[]={hint,hint};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_fmul_narrow(AGX_ISA_ENCODING_AGX3B,live?6:0,src,hints,
                                 false,32,bytes,&len));
      exact(bytes,len,live?"3101050300c0":"01011d0300c0");
   }
   const uint16_t odd_src[]={4,6};
   const enum agx3_alu_hint odd_hints[]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   assert(agx3_alu_fmul_narrow(AGX_ISA_ENCODING_AGX3B,1,odd_src,odd_hints,
                              false,0,bytes,&len));
   exact(bytes,len,"01051d070400");
   puts("Native narrow multiply: three Apple live/dead/odd-half vectors PASS");

   /* Independently compiled with metal-nt -arch applegpu_g17g, Metal 26.0,
    * -fno-fast-math: half4(input[i] * scale[i]). Compact instructions at
    * text+0x68/0x6e/0x72; the first component uses the slot-8 long form. */
   const struct {
      uint16_t dest, src[2], wait;
      const char *bytes;
   } m5[] = {
      {1,{2,10},0,"01031d0b0400"},
      {2,{4,12},0,"1105190d"},
      {3,{6,14},0,"11071d0f0400"},
      {0,{0,8},128,"290005182280a0021c0220000010"},
   };
   for (unsigned i=0;i<sizeof(m5)/sizeof(m5[0]);i++) {
      assert(agx3_alu_fmul_narrow(AGX_ISA_ENCODING_AGX3C,m5[i].dest,
                                  m5[i].src,odd_hints,false,m5[i].wait,
                                  bytes,&len));
      exact(bytes,len,m5[i].bytes);
   }
   assert(!agx3_alu_fmul_narrow(AGX_ISA_ENCODING_AGX3C,256,src,odd_hints,
                                 false,0,bytes,&len));
   assert(!agx3_alu_fmul_narrow(AGX_ISA_ENCODING_AGX3C,0,src,odd_hints,
                                 false,256,bytes,&len));
   puts("M5 narrow multiply: four native half4 vectors and bounds PASS");
}

static void test_native_half_multiply(void)
{
   for (unsigned live=0;live<2;live++) {
      const uint16_t src[]={live?0:2,live?1:3};
      enum agx3_alu_hint hint=live?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD;
      const enum agx3_alu_hint hints[]={hint,hint};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_hmul(AGX_ISA_ENCODING_AGX3B,live?4:2,src,hints,
                           false,32,bytes,&len));
      exact(bytes,len,live?"2000050100c0":"10021d0300c0");
   }
   const uint16_t src[]={3,2};
   const enum agx3_alu_hint hints[]={
      AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD};
   const bool both_abs[]={true,true};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   assert(agx3_alu_hmul_abs(AGX_ISA_ENCODING_AGX3B,2,src,hints,both_abs,
                            false,false,32,bytes,&len));
   exact(bytes,len,"10031d02020000800300");

   /* saturate(|a| * |b|) is the same ep3 form with only f14 added. */
   for (unsigned e = 0; e < 2; ++e) {
      const enum agx_isa_encoding enc =
         e ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      uint8_t plain[AGX3_ALU_MAX_LENGTH], plain_len;
      uint8_t sat[AGX3_ALU_MAX_LENGTH], sat_len;
      assert(agx3_alu_hmul_abs(enc,2,src,hints,both_abs,false,false,32,
                               plain,&plain_len));
      assert(agx3_alu_hmul_abs(enc,2,src,hints,both_abs,true,false,32,
                               sat,&sat_len));
      assert(plain_len == 10 && sat_len == plain_len);
      unsigned differing_bits = 0;
      for (unsigned i = 0; i < sat_len; ++i) {
         for (unsigned x = plain[i] ^ sat[i]; x; x &= x - 1)
            ++differing_bits;
      }
      assert(differing_bits == 1);
   }
   puts("Native half multiply: live/dead and absolute Apple vectors PASS");
}

static void test_wait_code_forms_fail_closed(void)
{
   /* Two wait ENCODINGS exist and agx3_alu_compose() must not mix them up.
    *
    * The ten-byte integer forms carry a six-bit MASK: iadd's f45..f50 land in
    * byte1 bits 4..7 and byte2 bits 0..1, so a pinned slot plus a requested
    * slot is simply "wait both" -- an over-wait, conservative and safe.
    *
    * The compact float forms instead carry a three-bit CODE naming ONE slot,
    * values 1..6, in byte5 bits 5..7. f32add's f52..f57 deposit
    * 0x20/0x40/0x60/0x80/0xa0/0xc0 there, and each of those six was witnessed
    * from Apple in a single kernel that issues six device loads and consumes
    * them in reverse: every consumer's code matched the f12 signal slot of the
    * lddev that produced its source. agx3_alu_table.h pins f52 (code 1) in the
    * f32add baseline as if it were a form constant, so asking for slot 2 used
    * to deposit both and emit 09 01 04 05 00 e0 -- code 7, a value Apple never
    * emits and which names no slot. That is a wrong value in the instruction
    * stream, so it must refuse; agx3_emit_alu() then retires the slot with a
    * synchronous prelude and the shader still compiles. */
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   const enum agx_isa_encoding B = AGX_ISA_ENCODING_AGX3B;

   /* Slot 1 is the pinned code, so it still encodes, byte-for-byte as before. */
   assert(agx3_alu_compose(B, AGX3_ALU_FADD, 0, 4, 0, 0, 0, 0, bytes, &len));
   exact(bytes, len, "090104050020");
   assert(agx3_alu_compose(B, AGX3_ALU_FADD, 0, 4, 0, 0, 0, 1u << 0, bytes, &len));
   exact(bytes, len, "090104050020");

   /* Slot 2 must NOT come back as code 7, and neither must any mask that
    * would merge two codes. */
   assert(!agx3_alu_compose(B, AGX3_ALU_FADD, 0, 4, 0, 0, 0, 1u << 1, bytes, &len));
   assert(!agx3_alu_compose(B, AGX3_ALU_FADD, 0, 4, 0, 0, 0, 0x3u, bytes, &len));

   /* The mask forms are untouched: pinned slot 6 plus the requested slot. */
   assert(agx3_alu_compose(B, AGX3_ALU_IADD, 0, 4, 0, 0, 0, 1u << 1, bytes, &len));
   exact(bytes, len, "9f215600020800a81305");
   assert(agx3_alu_compose(B, AGX3_ALU_IADD, 0, 4, 0, 0, 0, 0x3u, bytes, &len));
   exact(bytes, len, "9f315600020800a81305");

   /* inv/BNOT deposits every slot now that its wait list exists; before, all
    * 64 masks produced the same unwaited-except-slot-6 bytes. */
   assert(agx3_alu_compose(B, AGX3_ALU_BNOT, 0, 4, 0, 0, 0, 1u << 0, bytes, &len));
   exact(bytes, len, "0b052600012800800000");
   assert(agx3_alu_compose(B, AGX3_ALU_BNOT, 0, 4, 0, 0, 0, 1u << 3, bytes, &len));
   exact(bytes, len, "0b052600010800a00000");
   assert(agx3_alu_compose(B, AGX3_ALU_BNOT, 0, 4, 0, 0, 0,
                           (1u << 0) | (1u << 3), bytes, &len));
   exact(bytes, len, "0b052600012800a00000");

   puts("Wait-code forms fail closed: f32add slot2/slot1+2 refused instead of "
        "emitting code 7; mask forms and inv unchanged PASS");
}

static void test_native_half_saturating_arithmetic(void)
{
   /* applegpu_g17p, macOS 26 Metal toolchain. Each saturating vector is the
    * unsaturated one with G17P's f14 set, which moves it from the six-byte
    * ep1 point to the eight-byte ep2 point; the two field vectors differ in
    * f14 alone. The live/dead pairs additionally pin which lifetime bit
    * belongs to which source: f28 is source 0 (f17), f45 is source 1 (f34).
    *   saturate(a[i] * b[i])          10 03 1d 02 01 00 00 82
    *   saturate(a[i] + b[i])          10 03 1c 02 01 00 00 82
    *   saturate(x * y), y still live  18 03 0d 02 01 00 00 82
    *   saturate(x * y), x still live  18 02 15 03 01 00 00 82
    *   saturate(x + y), y still live  18 03 0c 02 01 00 00 82
    *   saturate(x + y), x still live  18 02 14 03 01 00 00 82 */
   static const struct {
      bool multiply;
      uint16_t dest, src0, src1;
      enum agx3_alu_hint hint0, hint1;
      const char *bytes;
   } vectors[] = {
      {true,  2, 3, 2, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
       "10031d0201000082"},
      {false, 2, 3, 2, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
       "10031c0201000082"},
      {true,  3, 3, 2, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE,
       "18030d0201000082"},
      {true,  3, 2, 3, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD,
       "1802150301000082"},
      {false, 3, 3, 2, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE,
       "18030c0201000082"},
      {false, 3, 2, 3, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD,
       "1802140301000082"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(*vectors); ++i) {
      const uint16_t src[2] = {vectors[i].src0, vectors[i].src1};
      const enum agx3_alu_hint hints[2] = {vectors[i].hint0, vectors[i].hint1};
      assert((vectors[i].multiply ? agx3_alu_hmul_saturate : agx3_alu_hadd_saturate)(
                AGX_ISA_ENCODING_AGX3B, vectors[i].dest, src, hints, false, 32,
                bytes, &len));
      exact(bytes, len, vectors[i].bytes);
   }

   /* The unsaturated encoders must not have moved: f14 = 0 is byte-identical
    * to omitting the field at every encoding point. */
   {
      const uint16_t src[2] = {3, 2};
      const enum agx3_alu_hint hints[2] = {AGX3_ALU_HINT_DISCARD,
                                           AGX3_ALU_HINT_DISCARD};
      assert(agx3_alu_hmul(AGX_ISA_ENCODING_AGX3B, 2, src, hints, false, 32,
                           bytes, &len));
      exact(bytes, len, "10031d0200c0");
      assert(agx3_alu_hadd(AGX_ISA_ENCODING_AGX3B, 2, src, hints, false, 32,
                           bytes, &len));
      exact(bytes, len, "10031c0200c0");

      /* Reject dependency slots outside each generation's scoreboard. */
      assert(!agx3_alu_hmul_saturate(AGX_ISA_ENCODING_AGX3C, 2, src, hints,
                                     false, 0x100, bytes, &len));
      assert(!agx3_alu_hadd_saturate(AGX_ISA_ENCODING_AGX3B, 2, src, hints,
                                     false, 0x40, bytes, &len));
   }
   puts("Native saturating half multiply/add: six exact Apple vectors plus "
        "unsaturated parity and bounds PASS");
}

static void test_native_half_multiply_minifloat(void)
{
   static const struct {
      uint16_t dest, src, immediate, wait;
      enum agx3_alu_hint hint;
      bool dest_cache;
      const char *bytes;
   } vectors[] = {
      /* Fresh applegpu_g17p Metal output: *2, *4, *0.5 and *-2. */
      {2, 2, 0x40, 32, AGX3_ALU_HINT_DISCARD, false, "10c0150280c0"},
      {2, 2, 0x40, 0, AGX3_ALU_HINT_DISCARD, false, "10c015028000"},
      {2, 2, 0x50, 32, AGX3_ALU_HINT_DISCARD, false, "10d0150280c0"},
      {2, 2, 0x20, 32, AGX3_ALU_HINT_DISCARD, false, "10a0150280c0"},
      {2, 2, 0xc0, 32, AGX3_ALU_HINT_DISCARD, false, "10c01d0280c0"},
      /* Odd retained source and cached destination are independent axes. */
      {2, 3, 0x40, 32, AGX3_ALU_HINT_NONE, false, "10c0050380c0"},
      {3, 2, 0x50, 32, AGX3_ALU_HINT_NONE, false, "18d0050280c0"},
      {2, 2, 0x50, 32, AGX3_ALU_HINT_DISCARD, true, "10d0350280c0"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_hmul_minifloat(
         AGX_ISA_ENCODING_AGX3B, vectors[i].dest, vectors[i].src,
         vectors[i].immediate, vectors[i].hint, vectors[i].dest_cache,
         vectors[i].wait, bytes, &len));
      exact(bytes, len, vectors[i].bytes);
   }

   assert(!agx3_alu_hmul_minifloat(
      AGX_ISA_ENCODING_AGX3C, 2, 2, 0x40, AGX3_ALU_HINT_DISCARD, false, 0x100,
      bytes, &len));
   assert(!agx3_alu_hmul_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 0x100, AGX3_ALU_HINT_DISCARD, false, 32,
      bytes, &len));
   assert(!agx3_alu_hmul_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 0x40, AGX3_ALU_HINT_DISCARD, false, 64,
      bytes, &len));
   puts("Native half/minifloat multiply: eight exact Apple vectors and bounds PASS");
}

static void test_native_minifloat_mixed_select(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   {
      const uint16_t selected[] = {2, 0};
      const bool immediate[] = {false, true};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0x30, selected, immediate, hints,
         1, false, false, 32, bytes, &len));
      exact(bytes, len, "02010bb1020203c22080");
   }
   {
      const uint16_t selected[] = {4, 0};
      const bool immediate[] = {false, true};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0x30, selected, immediate, hints,
         1, false, true, 32, bytes, &len));
      exact(bytes, len, "02012bb1020403c22080");
   }
   {
      const uint16_t selected[] = {0, 2};
      const bool immediate[] = {true, false};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0x40, selected, immediate, hints,
         1, false, false, 32, bytes, &len));
      exact(bytes, len, "02010bc1228003c20002");
   }
   {
      /* Fresh G17P vector witness for `a >= 0.25 ? v : 0`. */
      const uint16_t selected[] = {0, 0};
      const bool immediate[] = {false, true};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
         AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 8, 0x10, selected, immediate, hints,
         5, false, false, 32, bytes, &len));
      exact(bytes, len, "02090b91248002c2");
   }
   {
      /* Exact native `a >= 15.5 ? v : 0` alpha-coverage boundary. */
      const uint16_t selected[] = {0, 0};
      const bool immediate[] = {false, true};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
         AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 4, 0x6f, selected, immediate, hints,
         5, false, false, 32, bytes, &len));
      exact(bytes, len, "02050bef248012c2");
   }
   {
      const uint16_t selected[] = {255, 0};
      const bool immediate[] = {true, true};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 2, 2, 0x30, selected, immediate, hints,
         0, true, false, 32, bytes, &len));
      exact(bytes, len, "12030db1a6ff00c22080");
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 2, 2, 0x30, selected, immediate, hints,
         1, true, false, 32, bytes, &len));
      exact(bytes, len, "12030db1a2ff03c22080");
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 3, 2, 0x30, selected, immediate, hints,
         0, true, false, 32, bytes, &len));
      exact(bytes, len, "1a030db1a6ff00c22080");
   }

   /* Fresh applegpu_g17p GTK box-shadow witnesses. Apple canonicalizes
    * `-x < -0` to `x > +0`; the latter is byte-identical to a source written
    * directly as `x > 0`. The remaining vectors independently cover every
    * ordered condition, signed minifloats, both lifetime hints, both half
    * destination endpoints, arbitrary selected bytes and texture/load waits.
    */
   {
      static const struct {
         uint16_t dest, compare;
         uint8_t threshold, if_true, if_false, condition;
         enum agx3_alu_hint hint;
         bool dest_cache;
         uint16_t wait;
         const char *bytes;
      } native[] = {
         {2, 2, 0x00, 1, 0, 0, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030d81268100c22080"},
         {2, 2, 0x00, 1, 0, 1, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030d81228103c22080"},
         {2, 2, 0x00, 1, 0, 2, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030d81228102c22080"},
         {2, 2, 0x00, 1, 0, 5, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030d81268102c22080"},
         {2, 2, 0x00, 1, 0, 6, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030d81268103c22080"},
         {2, 2, 0xb0, 1, 0, 2, AGX3_ALU_HINT_DISCARD, false, 32,
          "12031db1228102c22080"},
         {2, 2, 0x20, 1, 0, 6, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030da1268103c22080"},
         {2, 2, 0x30, 1, 0, 2, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030db1228102c22080"},
         {2, 2, 0xb0, 1, 0, 1, AGX3_ALU_HINT_DISCARD, false, 32,
          "12031db1228103c22080"},
         {4, 0, 0x00, 1, 0, 2, AGX3_ALU_HINT_NONE, false, 32,
          "22010581228102c22080"},
         {2, 4, 0x00, 1, 0, 2, AGX3_ALU_HINT_CACHE, false, 32,
          "12850581228102c22080"},
         {4, 0, 0x00, 1, 0, 2, AGX3_ALU_HINT_CACHE, false, 0,
          "22810581228102022080"},
         {3, 2, 0x00, 1, 0, 2, AGX3_ALU_HINT_DISCARD, false, 32,
          "1a030d81228102c22080"},
         {2, 2, 0x00, 5, 7, 2, AGX3_ALU_HINT_DISCARD, false, 32,
          "12030d81228502c22087"},
         {2, 2, 0x00, 1, 0, 2, AGX3_ALU_HINT_DISCARD, false, 1,
          "12030d81228102222080"},
      };
      const bool immediate[] = {true, true};
      for (unsigned i = 0; i < sizeof(native) / sizeof(native[0]); ++i) {
         const uint16_t selected[] = {
            native[i].if_true, native[i].if_false};
         const enum agx3_alu_hint hints[] = {
            native[i].hint, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
         assert(agx3_alu_fcmp_minifloat_select_mixed(
            AGX_ISA_ENCODING_AGX3B, native[i].dest, native[i].compare,
            native[i].threshold, selected, immediate, hints,
            native[i].condition, true, native[i].dest_cache, native[i].wait,
            bytes, &len));
         exact(bytes, len, native[i].bytes);
      }

      /* Apple canonicalizes `1 < -x ? 0 : y` to `x < -1 ? 0 : y`. */
      const uint16_t signed_selected[] = {0, 4};
      const bool signed_immediate[] = {true, false};
      const enum agx3_alu_hint signed_hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE,
         AGX3_ALU_HINT_DISCARD};
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0xb0, signed_selected,
         signed_immediate, signed_hints, 1, false, false, 32, bytes, &len));
      exact(bytes, len, "02011bb1228003c28004");
   }

   /* Fresh full-result sweep from the same Apple build. Each condition is
    * pinned with floating zero on either side of the selected register. */
   {
      static const struct {
         uint8_t condition;
         bool zero_true;
         const char *bytes;
      } native[] = {
         {0, true,  "02010bb1268000c28004"},
         {1, true,  "02010bb1228003c28004"},
         {2, true,  "02010bb1228002c28004"},
         {5, true,  "02010bb1268002c28004"},
         {6, true,  "02010bb1268003c28004"},
         {0, false, "02010bb1860400c22080"},
         {1, false, "02010bb1820403c22080"},
         {2, false, "02010bb1820402c22080"},
         {5, false, "02010bb1860402c22080"},
         {6, false, "02010bb1860403c22080"},
      };
      for (unsigned i = 0; i < sizeof(native) / sizeof(native[0]); ++i) {
         const uint16_t selected[] = {
            native[i].zero_true ? 0 : 4,
            native[i].zero_true ? 4 : 0,
         };
         const bool immediate[] = {
            native[i].zero_true, !native[i].zero_true,
         };
         const enum agx3_alu_hint hints[] = {
            AGX3_ALU_HINT_DISCARD,
            native[i].zero_true ? AGX3_ALU_HINT_NONE
                                : AGX3_ALU_HINT_DISCARD,
            native[i].zero_true ? AGX3_ALU_HINT_DISCARD
                                : AGX3_ALU_HINT_NONE,
         };
         assert(agx3_alu_fcmp_minifloat_select_mixed(
            AGX_ISA_ENCODING_AGX3B, 0, 0, 0x30, selected, immediate,
            hints, native[i].condition, false, false, 32, bytes, &len));
         exact(bytes, len, native[i].bytes);
      }
   }

   /* Complete admitted half-result field product. */
   {
      static const unsigned conditions[] = {0, 1, 2, 5, 6};
      static const unsigned waits[] = {0, 1, 2, 4, 8, 16, 32};
      const bool immediate[] = {true, true};
      const uint16_t selected[] = {255, 7};
      for (unsigned c = 0; c < sizeof(conditions) / sizeof(conditions[0]); ++c)
         for (unsigned hint = 0; hint <= AGX3_ALU_HINT_CACHE; ++hint)
            for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache)
               for (unsigned dest_odd = 0; dest_odd < 2; ++dest_odd)
                  for (unsigned w = 0; w < sizeof(waits) / sizeof(waits[0]);
                       ++w) {
                     const enum agx3_alu_hint hints[] = {
                        (enum agx3_alu_hint)hint, AGX3_ALU_HINT_NONE,
                        AGX3_ALU_HINT_NONE};
                     assert(agx3_alu_fcmp_minifloat_select_mixed(
                        AGX_ISA_ENCODING_AGX3B, dest_odd ? 255 : 254, 254,
                        0x80, selected, immediate, hints, conditions[c], true,
                        dest_cache, waits[w], bytes, &len));
                     assert(len == 10);
                  }

      const enum agx3_alu_hint dead[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
      for (unsigned value = 0; value < 256; ++value) {
         const uint16_t span[] = {value, 255 - value};
         assert(agx3_alu_fcmp_minifloat_select_mixed(
            AGX_ISA_ENCODING_AGX3B, value, value & 0xfe, value, span,
            immediate, dead, 2, true, value & 1, 1u << (value % 6),
            bytes, &len));
      }
   }

   /* Complete full-result condition/orientation/lifetime/wait product. */
   {
      static const unsigned conditions[] = {0, 1, 2, 5, 6};
      static const unsigned waits[] = {0, 32};
      for (unsigned c = 0; c < sizeof(conditions) / sizeof(conditions[0]); ++c)
         for (unsigned zero_true = 0; zero_true < 2; ++zero_true)
            for (unsigned compare_hint = 0;
                 compare_hint <= AGX3_ALU_HINT_CACHE; ++compare_hint)
               for (unsigned selected_hint = 0;
                    selected_hint <= AGX3_ALU_HINT_CACHE; ++selected_hint)
                  for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache)
                     for (unsigned w = 0;
                          w < sizeof(waits) / sizeof(waits[0]); ++w) {
                        const uint16_t selected[] = {
                           zero_true ? 0 : 252,
                           zero_true ? 252 : 0,
                        };
                        const bool immediate[] = {
                           (bool)zero_true, !zero_true,
                        };
                        const enum agx3_alu_hint hints[] = {
                           (enum agx3_alu_hint)compare_hint,
                           zero_true
                              ? AGX3_ALU_HINT_NONE
                              : (enum agx3_alu_hint)selected_hint,
                           zero_true
                              ? (enum agx3_alu_hint)selected_hint
                              : AGX3_ALU_HINT_NONE,
                        };
                        assert(agx3_alu_fcmp_minifloat_select_mixed(
                           AGX_ISA_ENCODING_AGX3B, 254, 254, 0x80,
                           selected, immediate, hints, conditions[c], false,
                           dest_cache, waits[w], bytes, &len));
                     }
   }

   {
      const uint16_t gtk_selected[] = {1, 0};
      const bool gtk_immediate[] = {true, true};
      const enum agx3_alu_hint gtk_hints[] = {
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 44, 44, 0, gtk_selected, gtk_immediate,
         gtk_hints, 2, true, false, 0, bytes, &len));
      exact(bytes, len, "622d4581228102022080");
   }
   {
      const uint16_t full_selected[] = {0, 0};
      const bool full_immediate[] = {false, true};
      const enum agx3_alu_hint full_hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
         AGX3_ALU_HINT_NONE};
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 14, 2, 0xef, full_selected,
         full_immediate, full_hints, 6, false, false, 0, bytes, &len));
      exact(bytes, len, "72031bef860013022080");
   }
   {
      const uint16_t linear_selected[] = {0, 10};
      const bool linear_immediate[] = {true, false};
      const enum agx3_alu_hint linear_hints[] = {
         AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 10, 22, 0x30, linear_selected,
         linear_immediate, linear_hints, 5, false, false, 0, bytes, &len));
      exact(bytes, len, "529703b126800202000a");
   }
   {
      const uint16_t conic_selected[] = {0, 24};
      const bool conic_immediate[] = {true, false};
      const enum agx3_alu_hint conic_hints[] = {
         AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE,
         AGX3_ALU_HINT_DISCARD};
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 24, 12, 0x30, conic_selected,
         conic_immediate, conic_hints, 6, false, false, 0, bytes, &len));
      exact(bytes, len, "c28d03b1268003028018");
   }

   const uint16_t selected[] = {2, 0};
   const bool immediate[] = {false, true};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   assert(!agx3_alu_fcmp_minifloat_select_mixed(
      AGX_ISA_ENCODING_AGX3C, 0, 0, 0x30, selected, immediate, hints, 1,
      false, false, 0, bytes, &len));
   assert(agx3_alu_fcmp_minifloat_select_mixed(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0x30, selected, immediate, hints, 2,
      false, false, 0, bytes, &len));
   {
      const uint16_t nonzero[] = {7, 2};
      const bool zero_first[] = {true, false};
      assert(!agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0x30, nonzero, zero_first, hints, 2,
         false, false, 0, bytes, &len));
   }
   {
      const uint16_t selected_half[] = {1, 0};
      const bool immediate_half[] = {true, true};
      const enum agx3_alu_hint half_hints[] = {
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
      for (unsigned condition = 3; condition <= 7; ++condition) {
         if (condition == 5 || condition == 6)
            continue;
         assert(!agx3_alu_fcmp_minifloat_select_mixed(
            AGX_ISA_ENCODING_AGX3B, 0, 0, 0, selected_half,
            immediate_half, half_hints, condition, true, false, 0, bytes,
            &len));
      }
      assert(!agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0, selected_half, immediate_half,
         half_hints, 2, true, false, 3, bytes, &len));
      assert(!agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 0, selected_half, immediate_half,
         half_hints, 2, true, false, 64, bytes, &len));
      assert(!agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 0, 1, 0, selected_half, immediate_half,
         half_hints, 2, true, false, 0, bytes, &len));
      assert(!agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 256, 0, 0, selected_half,
         immediate_half, half_hints, 2, true, false, 0, bytes,
         &len));
   }
   /* The three measured Hyprland post-RA physical-register shapes. */
   {
      const uint16_t blur_selected[] = {24, 0};
      const bool blur_immediate[] = {false, true};
      const enum agx3_alu_hint blur_hints[] = {
         AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 24, 20, 0x30, blur_selected,
         blur_immediate, blur_hints, 1, false, true, 0, bytes, &len));
   }
   {
      const uint16_t ext_selected[] = {255, 0};
      const bool ext_immediate[] = {true, true};
      const enum agx3_alu_hint ext_hints[] = {
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 1, 14, 0x30, ext_selected,
         ext_immediate, ext_hints, 0, true, true, 0, bytes, &len));
   }
   {
      const uint16_t glitch_selected[] = {0, 2};
      const bool glitch_immediate[] = {true, false};
      const enum agx3_alu_hint glitch_hints[] = {
         AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD,
      };
      assert(agx3_alu_fcmp_minifloat_select_mixed(
         AGX_ISA_ENCODING_AGX3B, 2, 22, 0x40, glitch_selected,
         glitch_immediate, glitch_hints, 1, false, true, 0, bytes, &len));
   }
   puts("Native minifloat mixed select: 34 exact Apple vectors, 676 half + "
        "360 full legal forms and strict bounds PASS");
}

static void test_native_half_minifloat_select_f32(void)
{
   static const struct {
      unsigned condition;
      const char *bytes;
   } vectors[] = {
      {6, "02000fa1060203c20004"},
      {1, "02000fa1020203c20004"},
      {2, "02000fa1020202c20004"},
      {5, "02000fa1060202c20004"},
      {0, "02000fa1060200c20004"},
   };
   const uint16_t selected[] = {2, 4};
   const bool selected_immediate[] = {false, false};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_fcmp_half_minifloat_select_f32(
         AGX_ISA_ENCODING_AGX3B, 0, 0, 32, selected, selected_immediate, hints,
         vectors[i].condition, false, 32, bytes, &len));
      exact(bytes, len, vectors[i].bytes);
   }

   const uint16_t register_zero[] = {2, 0};
   const bool register_zero_immediate[] = {false, true};
   assert(agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0, register_zero,
      register_zero_immediate, hints, 1, false, 32, bytes, &len));
   exact(bytes, len, "02000b81020203c22080");
   const uint16_t zero_register[] = {0, 2};
   const bool zero_register_immediate[] = {true, false};
   assert(agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0, zero_register,
      zero_register_immediate, hints, 1, false, 32, bytes, &len));
   exact(bytes, len, "02000b81228003c20002");

   /* A non-zero integer keeps the f3 placement: Apple's
    * `x + ((a < 0.5h) ? v : 5u)` and `? 5u : v` as the untied ep27 form. */
   const uint16_t register_five[] = {4, 5};
   const bool register_five_immediate[] = {false, true};
   const enum agx3_alu_hint five_hints[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE,
   };
   assert(agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 4, 6, 32, register_five,
      register_five_immediate, five_hints, 1, true, 0, bytes, &len));
   exact(bytes, len, "22062fa1820403022085");
   const uint16_t five_register[] = {5, 4};
   const bool five_register_immediate[] = {true, false};
   const enum agx3_alu_hint five_second_hints[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 4, 6, 32, five_register,
      five_register_immediate, five_second_hints, 1, true, 0, bytes, &len));
   exact(bytes, len, "22062fa1228503028004");

   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3C, 0, 0, 32, selected, selected_immediate, hints,
      6, false, 32, bytes, &len));
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 1, 0, 32, selected, selected_immediate, hints,
      6, false, 32, bytes, &len));
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 256, 32, selected, selected_immediate, hints,
      6, false, 32, bytes, &len));
   const uint16_t odd_selected[] = {1, 4};
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 32, odd_selected, selected_immediate,
      hints, 6, false, 32, bytes, &len));
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 32, selected, selected_immediate, hints,
      3, false, 32, bytes, &len));
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 32, selected, selected_immediate, hints,
      6, false, 1, bytes, &len));
   const enum agx3_alu_hint bad_hints[] = {
      (enum agx3_alu_hint)3, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 32, selected, selected_immediate,
      bad_hints, 6, false, 32, bytes, &len));
   const bool both_immediate[] = {true, true};
   assert(!agx3_alu_fcmp_half_minifloat_select_f32(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 32, selected, both_immediate, hints, 6,
      false, 32, bytes, &len));
   puts("Native half/minifloat full FSEL: seven exact Apple vectors and bounds PASS");
}

static void test_native_fadd_abs_mixed_half(void)
{
   static const struct {
      uint16_t dest, f32, f16, wait;
      enum agx3_alu_hint hint32, hint16;
      bool dest_cache;
      const char *bytes;
   } vectors[] = {
      {0, 0, 4, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "09011c04020000800100"},
      {0, 0, 4, 1, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_CACHE, false,
       "09010c84022000000100"},
      {0, 2, 0, 32, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD, false,
       "09031400020000800100"},
      {0, 0, 2, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, true,
       "09013c02020000800100"},
      {0, 2, 3, 1, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, false,
       "09030402122000000100"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_fadd_abs_mixed_half(
         AGX_ISA_ENCODING_AGX3B, vectors[i].dest, vectors[i].f32,
         vectors[i].f16, vectors[i].hint32, vectors[i].hint16,
         vectors[i].dest_cache, vectors[i].wait, bytes, &len));
      exact(bytes, len, vectors[i].bytes);
   }

   assert(agx3_alu_fadd_abs_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 8, 8, 28, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_CACHE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_abs_mixed_half(
      AGX_ISA_ENCODING_AGX3C, 8, 8, 28, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_CACHE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_abs_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 9, 8, 28, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_CACHE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_abs_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 8, 8, 28, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_CACHE, false, 64, bytes, &len));
   puts("Native abs-f32 plus half add: five exact Apple vectors and bounds PASS");
}

static void test_native_fadd_mixed_half(void)
{
   static const struct {
      uint16_t dest, f32, f16, wait;
      enum agx3_alu_hint hint32, hint16;
      bool dest_cache;
      const char *bytes;
   } vectors[] = {
      /* add_dead */
      {0, 0, 4, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "09011c0400c0"},
      /* add_half_live */
      {0, 0, 8, 1, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "09011c080020"},
      /* add_float_live */
      {0, 2, 0, 32, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD, false,
       "0903140000c0"},
      /* add_half_pair, second channel: physical half 7 is intentionally odd. */
      {2, 2, 7, 0, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false,
       "19031c061000"},
      /* add_reuse_odd: the odd half and destination are both retained. */
      {0, 0, 5, 32, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_CACHE, true,
       "09012c8410c0"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_fadd_mixed_half(
         AGX_ISA_ENCODING_AGX3B, vectors[i].dest, vectors[i].f32,
         vectors[i].f16, vectors[i].hint32, vectors[i].hint16,
         vectors[i].dest_cache, vectors[i].wait, bytes, &len));
      exact(bytes, len, vectors[i].bytes);
   }

   assert(!agx3_alu_fadd_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 1, 0, 0, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 0, 1, 0, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 256, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0, (enum agx3_alu_hint)3,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0, AGX3_ALU_HINT_NONE,
      (enum agx3_alu_hint)3, false, 0, bytes, &len));
   assert(!agx3_alu_fadd_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 0, 0, 0, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, false, 64, bytes, &len));
   puts("Native f32 plus half add: five exact Apple vectors and bounds PASS");
}

static void test_native_half_add(void)
{
   {
      const uint16_t src[] = {2, 3};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_hadd(AGX_ISA_ENCODING_AGX3B, 2, src, hints, false, 32,
                           bytes, &len));
      exact(bytes, len, "10021c0300c0");
   }
   {
      const uint16_t src[] = {2, 3};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
      };
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_hadd(AGX_ISA_ENCODING_AGX3B, 1, src, hints, true, 32,
                           bytes, &len));
      exact(bytes, len, "0882248300c0");
   }

   const uint16_t src[] = {2, 3};
   const enum agx3_alu_hint hints[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   assert(!agx3_alu_hadd(AGX_ISA_ENCODING_AGX3C, 0, src, hints, false, 0x100,
                         bytes, &len));
   assert(!agx3_alu_hadd(AGX_ISA_ENCODING_AGX3B, 256, src, hints, false, 0,
                         bytes, &len));
   assert(!agx3_alu_hadd(AGX_ISA_ENCODING_AGX3B, 0, src, hints, false, 0x40,
                         bytes, &len));
   assert(agx3_alu_hadd_neg_negzero(
      AGX_ISA_ENCODING_AGX3B, 2, 2, AGX3_ALU_HINT_DISCARD, false, 32,
      bytes, &len));
   exact(bytes, len, "10021c8001020280");
   puts("Native half add: three exact Apple vectors and bounds PASS");
}

static void test_native_half_saturate(void)
{
   static const struct {
      uint16_t dest, src, wait;
      const char *bytes;
   } witnesses[] = {
      {0, 0, 1, "00001c8001220002"},
      {1, 1, 0, "08011c8001020002"},
      {2, 2, 0, "10021c8001020002"},
      {3, 3, 0, "18031c8001020002"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(witnesses) / sizeof(witnesses[0]); ++i) {
      assert(agx3_alu_hadd_saturate_negzero(
         AGX_ISA_ENCODING_AGX3B, witnesses[i].dest, witnesses[i].src,
         AGX3_ALU_HINT_DISCARD, false, witnesses[i].wait, bytes, &len));
      exact(bytes, len, witnesses[i].bytes);
   }

   /* Original applegpu_g17g half4 saturate bytes: dead and live inputs. */
   static const struct {
      uint16_t dest, src, wait;
      enum agx3_alu_hint hint;
      const char *bytes;
   } m5[] = {
      {0, 0, 128, AGX3_ALU_HINT_DISCARD, "280004022220a0028416"},
      {1, 1, 0, AGX3_ALU_HINT_DISCARD, "08011c8001020002"},
      {2, 2, 0, AGX3_ALU_HINT_DISCARD, "10021c8001020002"},
      {3, 3, 0, AGX3_ALU_HINT_DISCARD, "18031c8001020002"},
      {0, 4, 128, AGX3_ALU_HINT_NONE, "2800040a2220a0028016"},
      {1, 5, 0, AGX3_ALU_HINT_NONE, "0805148001020002"},
      {2, 6, 0, AGX3_ALU_HINT_NONE, "1006148001020002"},
      {3, 7, 0, AGX3_ALU_HINT_NONE, "1807148001020002"},
   };
   for (unsigned i = 0; i < sizeof(m5) / sizeof(m5[0]); ++i) {
      assert(agx3_alu_hadd_saturate_negzero(
         AGX_ISA_ENCODING_AGX3C, m5[i].dest, m5[i].src, m5[i].hint,
         false, m5[i].wait, bytes, &len));
      exact(bytes, len, m5[i].bytes);
   }
   assert(!agx3_alu_hadd_saturate_negzero(
      AGX_ISA_ENCODING_AGX3C, 0, 0, AGX3_ALU_HINT_NONE, false, 0x100,
      bytes, &len));
   assert(!agx3_alu_hadd_saturate_negzero(
      AGX_ISA_ENCODING_AGX3B, 256, 0, AGX3_ALU_HINT_NONE, false, 0, bytes,
      &len));
   assert(!agx3_alu_hadd_saturate_negzero(
      AGX_ISA_ENCODING_AGX3B, 0, 256, AGX3_ALU_HINT_NONE, false, 0, bytes,
      &len));
   assert(!agx3_alu_hadd_saturate_negzero(
      AGX_ISA_ENCODING_AGX3B, 0, 0, AGX3_ALU_HINT_NONE, false, 0x40, bytes,
      &len));
   puts("Native half saturate: twelve exact Apple vectors and bounds PASS");
}

static void test_native_half_inverse(void)
{
   static const struct {
      uint16_t dest, src, wait;
      enum agx3_alu_hint hint;
      const char *bytes;
   } witnesses[] = {
      {0, 0, 1, AGX3_ALU_HINT_DISCARD, "00b034008028"},
      {1, 1, 0, AGX3_ALU_HINT_DISCARD, "08b034018008"},
      {2, 2, 0, AGX3_ALU_HINT_DISCARD, "10b034028008"},
      {3, 3, 0, AGX3_ALU_HINT_DISCARD, "18b034038008"},
      {0, 1, 1, AGX3_ALU_HINT_CACHE, "00b024818028"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(witnesses) / sizeof(witnesses[0]); ++i) {
      assert(agx3_alu_hadd_one_neg(
         AGX_ISA_ENCODING_AGX3B, witnesses[i].dest, witnesses[i].src,
         witnesses[i].hint, true, witnesses[i].wait, bytes, &len));
      exact(bytes, len, witnesses[i].bytes);
   }

   /* Original applegpu_g17g 1.0h - half4 bytes, dead and live inputs. */
   static const struct {
      uint16_t dest, src, wait;
      enum agx3_alu_hint hint;
      const char *bytes;
   } m5[] = {
      {0, 0, 128, AGX3_ALU_HINT_DISCARD, "280024622620a8020008"},
      {1, 1, 0, AGX3_ALU_HINT_DISCARD, "08b014018008"},
      {2, 2, 0, AGX3_ALU_HINT_DISCARD, "10b014028008"},
      {3, 3, 0, AGX3_ALU_HINT_DISCARD, "18b014038008"},
      {0, 4, 128, AGX3_ALU_HINT_NONE, "280024622620a8020200"},
      {1, 5, 0, AGX3_ALU_HINT_NONE, "08b004058008"},
      {2, 6, 0, AGX3_ALU_HINT_NONE, "10b004068008"},
      {3, 7, 0, AGX3_ALU_HINT_NONE, "18b004078008"},
   };
   for (unsigned i = 0; i < sizeof(m5) / sizeof(m5[0]); ++i) {
      assert(agx3_alu_hadd_one_neg(
         AGX_ISA_ENCODING_AGX3C, m5[i].dest, m5[i].src, m5[i].hint,
         false, m5[i].wait, bytes, &len));
      exact(bytes, len, m5[i].bytes);
   }
   assert(!agx3_alu_hadd_one_neg(AGX_ISA_ENCODING_AGX3C, 0, 0,
                                  AGX3_ALU_HINT_NONE, false, 0x100, bytes, &len));
   assert(!agx3_alu_hadd_one_neg(AGX_ISA_ENCODING_AGX3B, 256, 0,
                                  AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_hadd_one_neg(AGX_ISA_ENCODING_AGX3B, 0, 256,
                                  AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_hadd_one_neg(AGX_ISA_ENCODING_AGX3B, 0, 0,
                                  AGX3_ALU_HINT_NONE, false, 0x40, bytes,
                                  &len));
   puts("Native half inverse: thirteen exact Apple vectors and bounds PASS");
}

static void test_native_half_fma(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   const uint16_t live_src[] = {2, 3, 4};
   const enum agx3_alu_hint live_hints[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
   };
   assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 1, live_src, live_hints,
                        false, false, true, 32, bytes, &len));
   exact(bytes, len, "08822683018400c0");

   const uint16_t dead_src[] = {0, 1, 2};
   const enum agx3_alu_hint dead_hints[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 0, dead_src, dead_hints,
                        false, false, false, 32, bytes, &len));
   exact(bytes, len, "00001e01810200c0");
   assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 1, live_src, live_hints,
                        false, true, true, 32, bytes, &len));
   exact(bytes, len, "08822683118400c0");
   assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 0, dead_src, dead_hints,
                        false, true, false, 32, bytes, &len));
   exact(bytes, len, "00001e01910200c0");
   assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 1, live_src, live_hints,
                        true, false, true, 32, bytes, &len));
   exact(bytes, len, "08822683018400c8");
   const uint16_t neg0_src[] = {3, 2, 4};
   assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 1, neg0_src, live_hints,
                        true, false, true, 32, bytes, &len));
   exact(bytes, len, "08832682018400c8");
   assert(!agx3_alu_hfma(AGX_ISA_ENCODING_AGX3C, 0, dead_src, dead_hints,
                         false, false, false, 0x100, bytes, &len));
   assert(!agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 256, dead_src, dead_hints,
                         false, false, false, 0, bytes, &len));
   assert(!agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 0, dead_src, dead_hints,
                          false, false, false, 0x40, bytes, &len));

   const uint8_t wait_tail[] = {0x00, 0x20, 0x40, 0x60,
                                 0x80, 0xa0, 0xc0};
   for (unsigned slot = 0; slot <= 6; ++slot) {
      const uint16_t wait = slot ? (1u << (slot - 1)) : 0;
      assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 0, dead_src, dead_hints,
                           false, false, false, wait, bytes, &len));
      assert(len == 8 && bytes[7] == wait_tail[slot]);
   }
   assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3B, 0, dead_src, dead_hints,
                        false, false, false, 3, bytes, &len));
   assert(len == 10 && ((bytes[7] >> 5) | ((bytes[9] >> 5) << 3)) == 3);
   puts("Native half FMA/FMS: six complete Apple modifier/lifetime vectors, "
        "six direct wait slots, no-wait, and refusals PASS");
}

static void test_m5_native_half_arithmetic(void)
{
   /* Original applegpu_g17g half4 arithmetic. Long forms wait on slot 8;
    * short forms exercise odd registers and both live/dead source lifetimes. */
   static const struct {
      bool add, saturate;
      uint16_t dest, src[2], wait;
      enum agx3_alu_hint hint;
      const char *bytes;
   } vectors[] = {
      {false, false, 0, {0, 4}, 128, AGX3_ALU_HINT_DISCARD, "280005022220a4020401"},
      {false, false, 1, {1, 5}, 0, AGX3_ALU_HINT_DISCARD, "08011905"},
      {false, false, 2, {2, 6}, 0, AGX3_ALU_HINT_DISCARD, "10021906"},
      {false, false, 3, {3, 7}, 0, AGX3_ALU_HINT_DISCARD, "18031907"},
      {false, false, 8, {4, 0}, 128, AGX3_ALU_HINT_NONE, "2800050a2220a0120000"},
      {false, false, 9, {5, 1}, 0, AGX3_ALU_HINT_NONE, "48050101"},
      {false, false, 10, {6, 2}, 0, AGX3_ALU_HINT_NONE, "50060102"},
      {false, false, 11, {7, 3}, 0, AGX3_ALU_HINT_NONE, "58070103"},
      {true, false, 0, {0, 4}, 128, AGX3_ALU_HINT_DISCARD, "280004022220a0020608"},
      {true, false, 1, {1, 5}, 0, AGX3_ALU_HINT_DISCARD, "08011805"},
      {true, false, 2, {2, 6}, 0, AGX3_ALU_HINT_DISCARD, "10021806"},
      {true, false, 3, {3, 7}, 0, AGX3_ALU_HINT_DISCARD, "18031807"},
      {false, true, 0, {0, 4}, 128, AGX3_ALU_HINT_DISCARD, "280005022220a4020601"},
      {false, true, 1, {1, 5}, 0, AGX3_ALU_HINT_DISCARD, "08011d0501000002"},
      {false, true, 2, {2, 6}, 0, AGX3_ALU_HINT_DISCARD, "10021d0601000002"},
      {false, true, 3, {3, 7}, 0, AGX3_ALU_HINT_DISCARD, "18031d0701000002"},
      {true, true, 0, {0, 4}, 128, AGX3_ALU_HINT_DISCARD, "280004022220a0020618"},
      {true, true, 1, {1, 5}, 0, AGX3_ALU_HINT_DISCARD, "08011c0501000002"},
      {true, true, 2, {2, 6}, 0, AGX3_ALU_HINT_DISCARD, "10021c0601000002"},
      {true, true, 3, {3, 7}, 0, AGX3_ALU_HINT_DISCARD, "18031c0701000002"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      const enum agx3_alu_hint hints[] = {vectors[i].hint, vectors[i].hint};
      bool (*encode)(enum agx_isa_encoding, uint16_t, const uint16_t *,
                     const enum agx3_alu_hint *, bool, uint16_t, uint8_t *, uint8_t *) =
         vectors[i].add ? (vectors[i].saturate ? agx3_alu_hadd_saturate : agx3_alu_hadd)
                        : (vectors[i].saturate ? agx3_alu_hmul_saturate : agx3_alu_hmul);
      assert(encode(AGX_ISA_ENCODING_AGX3C, vectors[i].dest, vectors[i].src,
                    hints, false, vectors[i].wait, bytes, &len));
      exact(bytes, len, vectors[i].bytes);
   }
   const char *fma[] = {"280006022320a4020c018000", "08011e058009",
                        "10021e06800a", "18031e07800b"};
   for (unsigned i = 0; i < 4; ++i) {
      const uint16_t src[] = {i, i + 4, i + 8};
      const enum agx3_alu_hint hints[] = {AGX3_ALU_HINT_DISCARD,
                                         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
      assert(agx3_alu_hfma(AGX_ISA_ENCODING_AGX3C, i, src, hints,
                           false, false, false, i ? 0 : 128, bytes, &len));
      exact(bytes, len, fma[i]);
   }
   puts("M5 half arithmetic: twenty-four original Apple instructions PASS");
}

static void test_m5_native_half_minmax(void)
{
   static const struct {
      bool maximum, live;
      uint16_t dest, x, y, wait;
      const char *bytes;
   } vectors[] = {
      {false, false, 0, 0, 4, 128, "2200450a2300a0020c81a06000028000"},
      {false, false, 1, 1, 5, 0, "0a001c041900"},
      {false, false, 2, 2, 6, 0, "12021c060100"},
      {false, false, 3, 3, 7, 0, "1a021c061900"},
      {true, false, 0, 0, 4, 128, "2200050a2300a0020c81a06400028000"},
      {true, false, 1, 1, 5, 0, "0a001c041800"},
      {true, false, 2, 2, 6, 0, "12021c060000"},
      {true, false, 3, 3, 7, 0, "1a021c061800"},
      {false, true, 8, 4, 0, 128, "2200452a2300a0120a80004000008000"},
      {false, true, 9, 5, 1, 0, "4a0404001900"},
      {false, true, 10, 6, 2, 0, "520604020100"},
      {false, true, 11, 7, 3, 0, "5a0604021900"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      const uint16_t src[] = {vectors[i].x, vectors[i].y, vectors[i].x, vectors[i].y};
      enum agx3_alu_hint hint = vectors[i].live ? AGX3_ALU_HINT_NONE : AGX3_ALU_HINT_DISCARD;
      const enum agx3_alu_hint hints[] = {hint, hint, hint, hint};
      assert((vectors[i].maximum ? agx3_alu_fmax_half : agx3_alu_fmin_half)(
         AGX_ISA_ENCODING_AGX3C, vectors[i].dest, src, hints, false,
         vectors[i].wait, bytes, &len));
      exact(bytes, len, vectors[i].bytes);
   }
   puts("M5 half min/max: twelve original Apple instructions PASS");
}

static void test_m5_native_half_immediates(void)
{
   static const struct {
      bool multiply, live;
      uint16_t dest, src, immediate, wait;
      const char *bytes;
   } vectors[] = {
      {false, false, 0, 0, 0x30, 128, "280004622620a8020008"},
      {false, false, 1, 1, 0x30, 0, "08b014018000"},
      {false, false, 2, 2, 0x30, 0, "10b014028000"},
      {false, false, 3, 3, 0x30, 0, "18b014038000"},
      {false, true, 0, 4, 0x30, 128, "280004622620a8020200"},
      {false, true, 1, 5, 0x30, 0, "08b004058000"},
      {false, true, 2, 6, 0x30, 0, "10b004068000"},
      {false, true, 3, 7, 0x30, 0, "18b004078000"},
      {true, false, 0, 0, 0x40, 128, "280005822620ac020000"},
      {true, false, 1, 1, 0x40, 0, "08c015018000"},
      {true, false, 2, 2, 0x40, 0, "10c015028000"},
      {true, false, 3, 3, 0x40, 0, "18c015038000"},
      {true, true, 0, 4, 0x40, 128, "280005822620a8020001"},
      {true, true, 1, 5, 0x40, 0, "08c005058000"},
      {true, true, 2, 6, 0x40, 0, "10c005068000"},
      {true, true, 3, 7, 0x40, 0, "18c005078000"},
      {true, false, 0, 0, 0x20, 128, "280005422620ac020000"},
      {true, false, 1, 1, 0x38, 0, "08b815018000"},
      {true, false, 2, 2, 0x44, 0, "10c415028000"},
      {true, false, 3, 3, 0x4c, 0, "18cc15038000"},
      {false, false, 0, 0, 0x40, 128, "280004822620a8020008"},
      {false, false, 1, 1, 0xb0, 0, "08b01c018000"},
      {false, false, 2, 2, 0x20, 0, "10a014028000"},
      {false, false, 3, 3, 0x38, 0, "18b814038000"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      enum agx3_alu_hint hint = vectors[i].live ? AGX3_ALU_HINT_NONE : AGX3_ALU_HINT_DISCARD;
      bool encoded = vectors[i].multiply
         ? agx3_alu_hmul_minifloat(AGX_ISA_ENCODING_AGX3C, vectors[i].dest,
                                    vectors[i].src, vectors[i].immediate, hint,
                                    false, vectors[i].wait, bytes, &len)
         : agx3_alu_hadd_minifloat(AGX_ISA_ENCODING_AGX3C, vectors[i].dest,
                                    vectors[i].src, vectors[i].immediate, false,
                                    false, hint, false, vectors[i].wait, bytes,
                                    &len);
      assert(encoded);
      exact(bytes, len, vectors[i].bytes);
   }
   assert(!agx3_alu_hadd_minifloat(AGX_ISA_ENCODING_AGX3C, 0, 0, 256,
                                   false, false, AGX3_ALU_HINT_NONE, false, 0,
                                   bytes, &len));
   assert(!agx3_alu_hadd_minifloat(AGX_ISA_ENCODING_AGX3C, 0, 256, 0x30,
                                   false, false, AGX3_ALU_HINT_NONE, false, 0,
                                   bytes, &len));
   assert(!agx3_alu_hadd_minifloat(AGX_ISA_ENCODING_AGX3C, 0, 0, 0x30,
                                   false, false, AGX3_ALU_HINT_NONE, false, 0x100,
                                   bytes, &len));
   puts("M5 half immediates: twenty-four original Apple instructions PASS");
}

static void test_m5_native_half_negation(void)
{
   /* Original strict Metal x-y, y-x, -x-y and saturate(x-y), dead/live. */
   const char *native[8][4] = {
      {"280024022220a0020608", "08011c050008", "10021c060008", "18031c070008"},
      {"2800240a2220a0020408", "08051c010008", "10061c020008", "18071c030008"},
      {"280024022220a0020688", "08011c0501080200", "10021c0601080200", "18031c0701080200"},
      {"280024022220a0020618", "08011c0501080002", "10021c0601080002", "18031c0701080002"},
      {"280024022220a0120200", "480104050008", "500204060008", "580304070008"},
      {"2800240a2220a0120000", "480504010008", "500604020008", "580704030008"},
      {"280024022220a0120280", "4801040501080200", "5002040601080200", "5803040701080200"},
      {"280024022220a0120210", "4801040501080002", "5002040601080002", "5803040701080002"},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned form = 0; form < 8; ++form) {
      const enum agx3_alu_hint hint = form >= 4 ? AGX3_ALU_HINT_NONE : AGX3_ALU_HINT_DISCARD;
      const enum agx3_alu_hint hints[] = {hint, hint};
      for (unsigned lane = 0; lane < 4; ++lane) {
         const bool reverse = form % 4 == 1;
         const uint16_t src[] = {lane + (reverse ? 4 : 0), lane + (reverse ? 0 : 4)};
         assert(agx3_alu_hadd_neg(AGX_ISA_ENCODING_AGX3C,
            lane + (form >= 4 ? 8 : 0), src, hints, form % 4 == 2 ? 3 : 2,
            form % 4 == 3, false, lane == 0 ? 128 : 0, bytes, &len));
         exact(bytes, len, native[form][lane]);
      }
   }
   const uint16_t src[] = {1, 5};
   const enum agx3_alu_hint hints[] = {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
   assert(!agx3_alu_hadd_neg(AGX_ISA_ENCODING_AGX3B, 1, src, hints,
                              2, false, false, 0, bytes, &len));
   assert(!agx3_alu_hadd_neg(AGX_ISA_ENCODING_AGX3C, 1, src, hints,
                              4, false, false, 0, bytes, &len));
   assert(!agx3_alu_hadd_neg(AGX_ISA_ENCODING_AGX3C, 1, src, hints,
                              2, false, false, 256, bytes, &len));
   puts("M5 half negation: thirty-two original Apple instructions PASS");
}

static void test_native_half_fma_minifloat_addend(void)
{
   /* applegpu-nt 32023.917, apple-g17p, fma(a, b, half(1.5/2.5)). */
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   const uint16_t src[] = {2, 3};
   const enum agx3_alu_hint dead_hints[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
   };

   assert(agx3_alu_hfma_addend_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, src, 56, dead_hints, false, 32, bytes,
      &len));
   exact(bytes, len, "10021e0321b800c0");
   assert(agx3_alu_hfma_addend_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, src, 68, dead_hints, false, 32, bytes,
      &len));
   exact(bytes, len, "10021e0321c400c0");

   assert(!agx3_alu_hfma_addend_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, src, 256, dead_hints, false, 32, bytes,
      &len));
   assert(!agx3_alu_hfma_addend_minifloat(
      AGX_ISA_ENCODING_AGX3B, 2, src, 56, dead_hints, false, 64, bytes,
      &len));
   for (unsigned wait = 0; wait < 64; ++wait) {
      assert(agx3_alu_hfma_addend_minifloat(
         AGX_ISA_ENCODING_AGX3B, 6, src, 128, dead_hints, false, wait,
         bytes, &len));
      if (wait && (wait & (wait - 1))) {
         assert(len == 10);
         assert(((bytes[7] >> 5) | ((bytes[9] >> 5) << 3)) == wait);
      }
   }
   puts("Native half FMA minifloat addend: two exact Apple vectors, all 64 "
        "wait masks, and bounds PASS");
}

static void test_native_widen_half(void)
{
   const char *golden[]={"09021c8100c2","19021c810802","0984148100c2","198414810802"};
   for (unsigned i=0;i<4;i++) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_widen_half(AGX_ISA_ENCODING_AGX3B,(i&1)*2,i+2,
         i<2?AGX3_ALU_HINT_DISCARD:AGX3_ALU_HINT_CACHE,false,false,false,
         (i&1)?0:32,bytes,&len));
      exact(bytes,len,golden[i]);
   }
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   assert(agx3_alu_widen_half(AGX_ISA_ENCODING_AGX3B,0,4,
      AGX3_ALU_HINT_CACHE,true,false,false,0,bytes,&len));
   exact(bytes,len,"09841481020200800100");
   assert(agx3_alu_widen_half(AGX_ISA_ENCODING_AGX3B,0,0,
      AGX3_ALU_HINT_DISCARD,true,false,false,0,bytes,&len));
   exact(bytes,len,"09001c81020200800100");
   assert(agx3_alu_widen_half(AGX_ISA_ENCODING_AGX3B,0,4,
      AGX3_ALU_HINT_CACHE,false,true,false,0,bytes,&len));
   exact(bytes,len,"0984148101020280");
   assert(agx3_alu_widen_half(AGX_ISA_ENCODING_AGX3B,0,0,
      AGX3_ALU_HINT_DISCARD,false,true,false,0,bytes,&len));
   exact(bytes,len,"09001c8101020280");
   assert(!agx3_alu_widen_half(AGX_ISA_ENCODING_AGX3B,0,0,
      AGX3_ALU_HINT_DISCARD,true,true,false,0,bytes,&len));
   puts("Native half widening/abs/neg: eight Apple vectors plus combined-modifier refusal PASS");
}

static void test_native_sqrt_step(void)
{
   for (unsigned live=0;live<2;live++) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_sqrt_step(AGX_ISA_ENCODING_AGX3B,live?0:2,live?2:0,
                               AGX3_ALU_HINT_CACHE,true,32,bytes,&len));
      exact(bytes,len,live?"2f015600030492400000":"2f015602030092400000");
   }
   puts("Native sqrt step: two complete Apple live/dead square-root kernels PASS");
}

static void test_native_half_sqrt(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   assert(agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 2, 4,
                                  AGX3_ALU_HINT_CACHE, true, 32,
                                  bytes, &len));
   exact(bytes, len, "2f01560203088a400000");
   assert(agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 2, 4,
                                  AGX3_ALU_HINT_CACHE, true, 0,
                                  bytes, &len));
   exact(bytes, len, "2f01540203088a400000");
   assert(agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 4, 3,
                                  AGX3_ALU_HINT_CACHE, true, 0,
                                  bytes, &len));
   exact(bytes, len, "2f01540403068a400000");

   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   exact(bytes, len, "11030904");
   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   exact(bytes, len, "11031904");
   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 3, 4, 3, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   exact(bytes, len, "11051d021400");
   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 4, 2, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, true, 0, bytes, &len));
   exact(bytes, len, "11053902");
   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 8, 4, 0, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
   exact(bytes, len, "41051900");
   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 8, 4, 0, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, false, 1, bytes, &len));
   exact(bytes, len, "41051d000020");
   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 4, 0, 4, AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_DISCARD, false, 32, bytes, &len));
   exact(bytes, len, "2181150400c0");
   assert(agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 4, 3, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_NONE, false, 32, bytes, &len));
   exact(bytes, len, "11050d0210c0");

   static const uint16_t half_sqrt_src[] = {14, 8, 10, 6};
   static const char *half_sqrt_hex[] = {
      "010f1900", "01091d001400", "110b1902", "11071d021400",
   };
   for (unsigned i = 0; i < 4; ++i) {
      assert(agx3_alu_fmul_mixed_half(
         AGX_ISA_ENCODING_AGX3C, i, half_sqrt_src[i], i,
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, false, 0, bytes, &len));
      exact(bytes, len, half_sqrt_hex[i]);
   }

   memset(bytes, 0xa5, sizeof(bytes));
   assert(!agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 3, 4,
                                   AGX3_ALU_HINT_CACHE, true, 0,
                                   bytes, &len));
   assert(!agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 2, 4,
                                   AGX3_ALU_HINT_DISCARD, true, 0,
                                   bytes, &len));
   assert(!agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 2, 4,
                                   AGX3_ALU_HINT_CACHE, false, 0,
                                   bytes, &len));
   assert(agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3B, 2, 4,
                                  AGX3_ALU_HINT_CACHE, true, 1,
                                  bytes, &len));
   exact(bytes, len, "2f11540203088a400000");
   memset(bytes, 0xa5, sizeof(bytes));
   assert(!agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 3, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 4, (enum agx3_alu_hint)3,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 4, AGX3_ALU_HINT_DISCARD,
      (enum agx3_alu_hint)3, false, 0, bytes, &len));
   assert(!agx3_alu_fmul_mixed_half(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 4, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_NONE, false, 64, bytes, &len));
   for (unsigned i = 0; i < sizeof(bytes); ++i)
      assert(bytes[i] == 0xa5);

   puts("Native half sqrt: nine exact Apple-derived step/multiply vectors and narrow refusal guards PASS");
}

static void test_native_select_reg_imm(void)
{
   const uint16_t src[4][4]={{0,2,10,1},{0,2,7,1},{0,2,10,1},{4,6,2,1}};
   const char *golden[]={"02011d03860a00c02081","32011d03060700c02081",
      "02011f03860a00c02081","02051f07060200c02081"};
   const bool no_abs[]={false,false};
   const bool no_neg[]={false,false};
   const bool imm[]={false,true};
   for(unsigned i=0;i<4;i++) {
      const enum agx3_alu_hint hints[]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,
         (i&1)?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_NONE};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,i==1?6:0,src[i],false,no_abs,no_neg,imm,
         hints,0,i<2,false,32,bytes,&len));
      exact(bytes,len,golden[i]);
   }
   puts("Native retained-register/immediate FSEL: four exact16/32-bit live/dead vectors PASS");
   for(unsigned live=0;live<2;live++) {
      const uint16_t inputs[]={live?2:0,0,1,0};
      const bool immediate[]={true,true};
      const enum agx3_alu_hint hints[]={live?AGX3_ALU_HINT_NONE:AGX3_ALU_HINT_DISCARD,
         AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,0,inputs,true,no_abs,no_neg,immediate,
         hints,2,true,false,32,bytes,&len));
      exact(bytes,len,live?"02030581228102c22080":"02010d81228102c22080");
   }
   puts("Native zero-comparison boolean FSEL: two exact live/dead vectors PASS");
   /* Fresh complete Apple kernels select two loaded values, then optionally
    * store both operands again. Cover DISCARD, CACHE and retained NONE hints. */
   const uint16_t registers[4][4]={{4,6,8,9},{4,6,2,3},{4,6,8,10},{6,8,2,4}};
   const char *reg_golden[]={"02051d07820803c08009","0a053d07028203c00083",
      "02051f07820803c0800a","02071f09020203c00004"};
   const bool reg_immediate[]={false,false};
   for(unsigned i=0;i<4;i++) {
      const enum agx3_alu_hint live_hint=i==1?AGX3_ALU_HINT_CACHE:AGX3_ALU_HINT_NONE;
      const enum agx3_alu_hint hints[]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,
         (i&1)?live_hint:AGX3_ALU_HINT_DISCARD,
         (i&1)?live_hint:AGX3_ALU_HINT_DISCARD};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,i==1?1:0,registers[i],
         false,no_abs,no_neg,reg_immediate,hints,1,i<2,i==1,32,bytes,&len));
      exact(bytes,len,reg_golden[i]);
   }
   puts("Native register-register FSEL: four exact 16/32-bit live/dead vectors PASS");
   const uint16_t zero_registers[4][4]={{0,0,8,9},{0,0,2,3},{14,0,4,6},{0,0,2,4}};
   const char *zero_golden[]={"02010d81820803c28009","0a012d81028203c20083",
      "020f0f81820403c28006","02010f81020203c20004"};
   for(unsigned i=0;i<4;i++) {
      const enum agx3_alu_hint live_hint=i==1?AGX3_ALU_HINT_CACHE:AGX3_ALU_HINT_NONE;
      const enum agx3_alu_hint hints[]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_NONE,
         (i&1)?live_hint:AGX3_ALU_HINT_DISCARD,
         (i&1)?live_hint:AGX3_ALU_HINT_DISCARD};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,i==1?1:0,
         zero_registers[i],true,no_abs,no_neg,reg_immediate,hints,1,i<2,i==1,32,bytes,&len));
      exact(bytes,len,zero_golden[i]);
   }
   puts("Native zero-compare register FSEL: four exact 16/32-bit live/dead vectors PASS");

   /* A negated SECOND comparator, at both result widths. Apple's own
    * applegpu_g17p output for `select(x, y, a < -b)` against the same kernel
    * written `a < b`:
    *
    *   32-bit result   a <  b   02011f03820a03c08006
    *                   a < -b   02011f03820a03c88006
    *   16-bit result   a <  b   02011d03820b0320800a
    *                   a < -b   02011d03820b0328800a
    *
    * One field apart (f65) in the same encoding point, so the negate does not
    * escalate the form the way the absolute modifiers do. `-a < b` came back
    * as 02031d01820b0228800a: f65 again, with the operands swapped and the
    * relation flipped, so there is no bit on the first comparator and that
    * side must stay refused. */
   {
      const bool neg_second[] = {false, true};
      const bool neg_first[] = {true, false};
      const bool no_neg[] = {false, false};
      const uint16_t neg_src32[] = {0, 2, 10, 6};
      const uint16_t neg_src16[] = {0, 2, 11, 10};
      const enum agx3_alu_hint dead[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
      uint8_t nbytes[AGX3_ALU_MAX_LENGTH];
      uint8_t nlen;

      assert(agx3_alu_fcmp_select_reg_imm(
         AGX_ISA_ENCODING_AGX3B, 0, neg_src32, false, no_abs, no_neg,
         reg_immediate, dead, 1, false, false, 32, nbytes, &nlen));
      exact(nbytes, nlen, "02011f03820a03c08006");
      assert(agx3_alu_fcmp_select_reg_imm(
         AGX_ISA_ENCODING_AGX3B, 0, neg_src32, false, no_abs, neg_second,
         reg_immediate, dead, 1, false, false, 32, nbytes, &nlen));
      exact(nbytes, nlen, "02011f03820a03c88006");

      /* The 16-bit pair is checked as a DELTA. Apple's kernel waits on a
       * different scoreboard slot than this encoder can spell (f102 rather
       * than the f107 every checked-in 16-bit vector carries), so its bytes
       * are not reproducible here operand-for-operand -- but the negate's
       * effect is: Apple's pair moves byte 6 from 0x20 to 0x28, and so does
       * this encoder's, from 0xc0 to 0xc8. Same bit, same byte, same form. */
      uint8_t plain16[AGX3_ALU_MAX_LENGTH];
      uint8_t plain16_len;
      assert(agx3_alu_fcmp_select_reg_imm(
         AGX_ISA_ENCODING_AGX3B, 0, neg_src16, false, no_abs, no_neg,
         reg_immediate, dead, 1, true, false, 32, plain16, &plain16_len));
      exact(plain16, plain16_len, "02011d03820b03c0800a");
      assert(agx3_alu_fcmp_select_reg_imm(
         AGX_ISA_ENCODING_AGX3B, 0, neg_src16, false, no_abs, neg_second,
         reg_immediate, dead, 1, true, false, 32, nbytes, &nlen));
      exact(nbytes, nlen, "02011d03820b03c8800a");
      assert(nlen == plain16_len);
      for (unsigned i = 0; i < nlen; ++i)
         assert(nbytes[i] == (i == 7 ? (uint8_t)(plain16[i] | 0x08)
                                     : plain16[i]));

      /*
       * Three combinations stood refused here as unwitnessed. Each now has a
       * witness, so each flips -- and none flips because it started failing.
       *
       * 1. A negated FIRST comparator. The comment above this block already
       *    records Apple's `-a < b` as 02031d01820b0228800a, "f65 again, with
       *    the operands swapped and the relation flipped": the rewrite was
       *    observed and then declined. Performing it must give byte-for-byte
       *    what the caller would get by doing the swap itself, so assert the
       *    identity rather than a copied literal.
       */
      const uint16_t neg_swapped32[] = {2, 0, 10, 6};
      const enum agx3_alu_hint dead_swapped[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
      uint8_t by_hand[AGX3_ALU_MAX_LENGTH];
      uint8_t by_hand_len;
      assert(agx3_alu_fcmp_select_reg_imm(
         AGX_ISA_ENCODING_AGX3B, 0, neg_src32, false, no_abs, neg_first,
         reg_immediate, dead, 1, false, false, 32, nbytes, &nlen));
      assert(agx3_alu_fcmp_select_reg_imm(
         AGX_ISA_ENCODING_AGX3B, 0, neg_swapped32, false, no_abs, neg_second,
         reg_immediate, dead_swapped, 2, false, false, 32, by_hand,
         &by_hand_len));
      assert(nlen == by_hand_len && memcmp(nbytes, by_hand, nlen) == 0);

      /*
       * 2. The negate mixed with an ABSOLUTE comparator. Fresh applegpu_g17p
       *    kernels selecting integers, one field apart, dest r1l with cache
       *    and comparisons r1 and r2 with last use:
       *
       *      x + ((fabs(a) < -c) ? 3 : 1)   12033d0523830308208100800100
       *      x + ((-a < fabs(c)) ? 3 : 1)   12053d0323830208208100800100
       *
       *    f47 and f65 on one fourteen-byte instruction, which retires the
       *    note that said they cannot compose. The second row is the swap with
       *    an abs riding along -- Apple exchanges f35/f52 (2/4 -> 4/2), flips
       *    f110 lt -> gt, and carries the abs with the operand it belonged to,
       *    which is exactly what this encoder does.
       */
      const bool one_abs[] = {true, false};
      const bool second_abs[] = {false, true};
      const bool imm_selects[] = {true, true};
      const uint16_t abs_neg_src[] = {2, 4, 3, 1};
      const enum agx3_alu_hint abs_neg_hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};
      assert(agx3_alu_fcmp_select_reg_imm(
         AGX_ISA_ENCODING_AGX3B, 2, abs_neg_src, false, one_abs, neg_second,
         imm_selects, abs_neg_hints, 1, true, true, 32, nbytes, &nlen));
      exact(nbytes, nlen, "12033d0523830308208100800100");
      assert(agx3_alu_fcmp_select_reg_imm(
         AGX_ISA_ENCODING_AGX3B, 2, abs_neg_src, false, second_abs, neg_first,
         imm_selects, abs_neg_hints, 1, true, true, 32, nbytes, &nlen));
      exact(nbytes, nlen, "12053d0323830208208100800100");

      /* The same pair with a full-width result, on its own endpoint. That
       * kernel's comparators land in r2 and r3. */
      const uint16_t abs_neg_src32[] = {4, 6, 3, 1};
      assert(agx3_alu_fcmp_select_reg_imm(
         AGX_ISA_ENCODING_AGX3B, 4, abs_neg_src32, false, one_abs, neg_second,
         imm_selects, abs_neg_hints, 1, false, true, 32, nbytes, &nlen));
      exact(nbytes, nlen, "22053f0723830308208100800100");

      /*
       * 3. A negate against the exact immediate ZERO. -0.0 and +0.0 compare
       *    equal, so the negate cannot change the result and must not change
       *    the bytes. Apple agrees on the other side of the same identity:
       *    `-a < 0.0f` and `a > 0.0f` are one instruction, 1203098122b002c22080
       *    both. Assert the identity, which is what the fold claims.
       */
      /* compare_zero pins the second comparator to the literal 0, and an
       * immediate slot carries no lifetime hint. */
      const uint16_t zero_src32[] = {0, 0, 10, 6};
      const enum agx3_alu_hint zero_hints[] = {
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE,
         AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
      uint8_t zero_plain[AGX3_ALU_MAX_LENGTH];
      uint8_t zero_plain_len;
      assert(agx3_alu_fcmp_select_reg_imm(
         AGX_ISA_ENCODING_AGX3B, 0, zero_src32, true, no_abs, no_neg,
         reg_immediate, zero_hints, 1, false, false, 32, zero_plain,
         &zero_plain_len));
      assert(agx3_alu_fcmp_select_reg_imm(
         AGX_ISA_ENCODING_AGX3B, 0, zero_src32, true, no_abs, neg_second,
         reg_immediate, zero_hints, 1, false, false, 32, nbytes, &nlen));
      assert(nlen == zero_plain_len &&
             memcmp(nbytes, zero_plain, nlen) == 0);

      /* Still refused, and still unwitnessed: a negate on a SELECTED value.
       * Those slots carry a value rather than an arithmetic operand and have
       * no modifier field at all -- the packer refuses them before reaching
       * this encoder, and nothing here invents one. */
   }
   puts("Native FSEL comparator negation: nine exact vectors and two rewrite identities PASS");

   /* Own Metal kernels isolate fabs on each comparator and retain all three
    * selected-operand forms used by update_lightmap. */
   const bool both_abs[]={true,true};
   const uint16_t abs_reg0[]={0,2,4,0};
   const bool reg0_imm[]={false,true};
   const enum agx3_alu_hint no_hints[]={AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_reg0,
      false,both_abs,no_neg,reg0_imm,no_hints,0,false,false,32,bytes,&len));
   exact(bytes,len,"3201030307040000208000800300");

   const bool left_abs[]={true,false},right_abs[]={false,true};
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_reg0,
      false,left_abs,no_neg,reg0_imm,no_hints,0,false,false,32,bytes,&len));
   exact(bytes,len,"3201030307040000208000800100");
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_reg0,
      false,right_abs,no_neg,reg0_imm,no_hints,0,false,false,32,bytes,&len));
   exact(bytes,len,"3201030307040000208000800200");

   const uint16_t abs_regreg[]={0,2,4,6};
   const enum agx3_alu_hint regreg_hints[]={AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_DISCARD};
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_regreg,
      false,both_abs,no_neg,reg_immediate,regreg_hints,0,false,false,32,bytes,&len));
   exact(bytes,len,"3201070307040000800600800300");

   const uint16_t abs_zeroreg[]={0,2,0,4};
   const bool zeroreg_imm[]={true,false};
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_zeroreg,
      false,both_abs,no_neg,zeroreg_imm,no_hints,0,false,false,32,bytes,&len));
   exact(bytes,len,"3201030327800000000400800300");

   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_reg0,
      false,both_abs,no_neg,reg0_imm,no_hints,1,false,false,32,bytes,&len));
   exact(bytes,len,"3201030303040300208000800300");
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_regreg,
      false,both_abs,no_neg,reg_immediate,regreg_hints,1,false,false,32,bytes,&len));
   exact(bytes,len,"3201070303040300800600800300");
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_zeroreg,
      false,both_abs,no_neg,zeroreg_imm,no_hints,1,false,false,32,bytes,&len));
   exact(bytes,len,"3201030323800300000400800300");
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3B,6,abs_reg0,
      false,both_abs,no_neg,reg0_imm,no_hints,2,false,false,32,bytes,&len));
   exact(bytes,len,"3201030303040200208000800300");

   /* Hyprland's colour conversion compares a full value against one
    * absolute full value.  Fresh direct-value Metal kernels cover every
    * ordered condition on the same long endpoint. */
   const uint16_t abs_second_src[]={2,4,6,0};
   const bool abs_second[]={false,true};
   const enum agx3_alu_hint abs_second_hints[]={AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_NONE,AGX3_ALU_HINT_DISCARD};
   static const struct {
      unsigned condition;
      const char *bytes;
   } abs_ordered[] = {
      {5,"0203070507060200800000800200"},
      {6,"0203070507060300800000800200"},
      {1,"0203070503060300800000800200"},
      {2,"0203070503060200800000800200"},
      {0,"0203070507060000800000800200"},
   };
   for (unsigned i=0;i<sizeof(abs_ordered)/sizeof(abs_ordered[0]);++i) {
      assert(agx3_alu_fcmp_select_reg_imm(
         AGX_ISA_ENCODING_AGX3B,0,abs_second_src,false,abs_second,no_neg,
         reg_immediate,abs_second_hints,abs_ordered[i].condition,false,false,
         32,bytes,&len));
      exact(bytes,len,abs_ordered[i].bytes);
   }

   const uint16_t abs_first_src[]={4,2,6,0};
   const bool abs_first[]={true,false};
   assert(agx3_alu_fcmp_select_reg_imm(
      AGX_ISA_ENCODING_AGX3B,0,abs_first_src,false,abs_first,no_neg,reg_immediate,
      abs_second_hints,6,false,false,32,bytes,&len));
   exact(bytes,len,"0205070307060300800000800100");
   /* Ordered dual-absolute selects use the general endpoint. */
   assert(agx3_alu_fcmp_select_reg_imm(
      AGX_ISA_ENCODING_AGX3B,0,abs_second_src,false,both_abs,no_neg,reg_immediate,
      abs_second_hints,5,false,false,32,bytes,&len));
   puts("Native absolute-comparison FSEL: fifteen exact G17P vectors PASS");
}

/*
 * A NEGATED comparison source on `fcmpsel`.
 *
 * f65 names the second comparator and only that one. Apple has no bit for a
 * negated FIRST source: it spells `-a > b` as `b < -a`, exchanging the
 * operands and reversing the relation. Every vector below is
 * metal-nt -arch applegpu_g17p, one kernel per row, the reproducer's own
 * shape -- a 32-bit result selecting 1 and 0 from two loaded floats.
 *
 * The last two rows are what make the rewrite a fact rather than a reading.
 * `a < -b` sets f65 with the operands left alone, so f65 is slot 1's bit; and
 * `-a < -b` comes back byte-identical to `a > b`, with no f65 at all, so
 * negating both sides is the relation flip on its own.
 */
static void test_native_select_negated_comparator(void)
{
   static const struct {
      const char *what;
      bool neg0, neg1;
      unsigned condition;
      const char *bytes;
   } rows[] = {
      {" a >  b", false, false, 2 /* gt */, "02011d03228102c02080"},
      {"-a >  b", true,  false, 2 /* gt */, "02031d01228103c82080"},
      {"-a <  b", true,  false, 1 /* lt */, "02031d01228102c82080"},
      {"-a >= b", true,  false, 5 /* ge */, "02031d01268103c82080"},
      {"-a <= b", true,  false, 6 /* le */, "02031d01268102c82080"},
      {"-a == b", true,  false, 0 /* eq */, "02031d01268100c82080"},
      {" a < -b", false, true,  1 /* lt */, "02011d03228103c82080"},
      {"-a < -b", true,  true,  1 /* lt */, "02011d03228102c02080"},
   };

   const uint16_t src[] = {0, 2};
   const bool src_16[] = {false, false};
   const enum agx3_alu_hint hints[] = {AGX3_ALU_HINT_DISCARD,
                                       AGX3_ALU_HINT_DISCARD};
   for (unsigned i = 0; i < sizeof(rows)/sizeof(rows[0]); ++i) {
      const bool neg[] = {rows[i].neg0, rows[i].neg1};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_fcmp_select_imm_width_neg(
         AGX_ISA_ENCODING_AGX3B, 0, src, src_16, hints, neg,
         rows[i].condition, 1, 0, true, false, 32, bytes, &len));
      exact(bytes, len, rows[i].bytes);
   }

   /* Unmodified sources reach the same bytes through the plain encoder, so
    * the negate path adds no encoding of its own for them. */
   const bool no_neg[] = {false, false};
   uint8_t through_neg[AGX3_ALU_MAX_LENGTH], neg_len;
   uint8_t plain[AGX3_ALU_MAX_LENGTH], plain_len;
   assert(agx3_alu_fcmp_select_imm_width_neg(
      AGX_ISA_ENCODING_AGX3B, 0, src, src_16, hints, no_neg, 2 /* gt */, 1,
      0, true, false, 32, through_neg, &neg_len));
   assert(agx3_alu_fcmp_select_imm_width(
      AGX_ISA_ENCODING_AGX3B, 0, src, src_16, hints, 2 /* gt */, 1, 0, true,
      false, 32, plain, &plain_len));
   assert(neg_len == plain_len && !memcmp(through_neg, plain, neg_len));

   /* The NaN-ordering relations have no witness on this form and the caller
    * rewrites them before it gets here, so they still fail closed. */
   const bool neg_first[] = {true, false};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   assert(!agx3_alu_fcmp_select_imm_width_neg(
      AGX_ISA_ENCODING_AGX3B, 0, src, src_16, hints, neg_first, 3 /* ltn */,
      1, 0, true, false, 32, bytes, &len));

   puts("Negated FSEL comparator: eight exact G17P vectors PASS");
}

/*
 * RTZ float to a 16-bit integer, from a HALF source.
 *
 * The source width is its own field (f24), independent of the result width
 * (f10) and of the signed/unsigned mode (f41). Each pair below changes one
 * of those three and nothing else, applegpu_g17p:
 */
static void test_native_float_to_int16(void)
{
   static const struct {
      const char *what;
      bool src_16, is_signed;
      uint16_t dest, src, wait;
      enum agx3_alu_hint hint;
      const char *bytes;
   } rows[] = {
      {"short(float)",  false, true,  2, 2, 32, AGX3_ALU_HINT_DISCARD,
       "270756020204b0c80200"},
      {"short(half)",   true,  true,  2, 2, 32, AGX3_ALU_HINT_DISCARD,
       "270756020204a8c80200"},
      {"ushort(half)",  true,  false, 2, 2, 32, AGX3_ALU_HINT_DISCARD,
       "270756020204a8080300"},
      {"ushort(float)", false, false, 2, 2, 32, AGX3_ALU_HINT_DISCARD,
       "270756020204b0080300"},
      /* No wait: the clamp-then-convert shape, whose input is already live. */
      {"short(half) nowait", true, true, 2, 2, 0, AGX3_ALU_HINT_DISCARD,
       "270754020204a8c80200"},
      /* A half source may sit in an ODD half; a 32-bit one may not. */
      {"short(half odd)", true, true, 3, 3, 0, AGX3_ALU_HINT_DISCARD,
       "270754030206a8c80200"},
   };

   for (unsigned i = 0; i < sizeof(rows)/sizeof(rows[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_float_to_int16(AGX_ISA_ENCODING_AGX3B, rows[i].dest,
                                     rows[i].src, rows[i].src_16,
                                     rows[i].is_signed, rows[i].hint, false,
                                     rows[i].wait, bytes, &len));
      exact(bytes, len, rows[i].bytes);
   }

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   assert(!agx3_alu_float_to_int16(AGX_ISA_ENCODING_AGX3B, 2, 3, false, true,
                                   AGX3_ALU_HINT_NONE, false, 0, bytes, &len));

   puts("Half-source float-to-short: six exact G17P vectors PASS");
}

static void test_native_half_min(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   const uint16_t internal_src[]={2,4,2,4};
   const enum agx3_alu_hint cached[]={AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE};
   assert(agx3_alu_fmin_half(AGX_ISA_ENCODING_AGX3B,0,internal_src,cached,
                            true,32,bytes,&len));
   exact(bytes,len,"0282248401c0");
   const uint16_t dead_src[]={2,3,2,3};
   const enum agx3_alu_hint dead[]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD};
   assert(agx3_alu_fmin_half(AGX_ISA_ENCODING_AGX3B,2,dead_src,dead,
                            false,32,bytes,&len));
   exact(bytes,len,"12021c0211c0");
   const uint16_t wait1_src[]={1,8,1,8};
   assert(agx3_alu_fmin_half(AGX_ISA_ENCODING_AGX3B,1,wait1_src,dead,
                            false,1,bytes,&len));
   exact(bytes,len,"0a001c080920");
   puts("Native half minimum: exact internal-clamp/dead Apple FSEL vectors PASS");
}

static void test_native_half_compare_select(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   const uint16_t src[] = {2, 3, 4, 5};
   const enum agx3_alu_hint cached[] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_DISCARD,
   };
   /* The native vector starts with 0x0a because it writes odd half 1 while
    * caching the result. Field12 carries cache; byte0 bit3 remains the low
    * destination bit. */
   assert(agx3_alu_fcmp_select_half(AGX_ISA_ENCODING_AGX3B, 1, src,
                                    cached, 1, true, 0, bytes, &len));
   exact(bytes, len, "0a822582028413c08005");
   assert(agx3_alu_fcmp_select_half(AGX_ISA_ENCODING_AGX3B, 0, src,
                                    cached, 1, true, 0, bytes, &len));
   exact(bytes, len, "02822582028413c08005");

   const uint16_t dead_src[] = {0, 1, 2, 3};
   const enum agx3_alu_hint dead[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_fcmp_select_half(AGX_ISA_ENCODING_AGX3B, 0, dead_src,
                                    dead, 1, false, 0, bytes, &len));
   exact(bytes, len, "02001d00820213c08003");

   /* Reject unwitnessed shapes without touching the caller's storage. */
   for (unsigned bad = 0; bad < 7; ++bad) {
      uint16_t bad_src[] = {0, 1, 2, 3};
      enum agx3_alu_hint bad_hints[] = {
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      };
      enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
      unsigned dest = 0, cond = 1, wait = 0;
      if (bad == 0) enc = AGX_ISA_ENCODING_AGX3C;
      if (bad == 1) dest = 256;
      if (bad == 2) bad_src[0] = 256;
      if (bad == 3) bad_src[3] = 256;
      if (bad == 4) bad_hints[2] = 3;
      if (bad == 5) cond = 4;
      if (bad == 6) wait = 1;
      memset(bytes, 0xa5, sizeof(bytes));
      len = 0xa5;
      assert(!agx3_alu_fcmp_select_half(enc, dest, bad_src, bad_hints,
                                        cond, false, wait, bytes, &len));
      assert(len == 0xa5);
      for (unsigned i = 0; i < sizeof(bytes); ++i)
         assert(bytes[i] == 0xa5);
   }
   puts("Native scalar half comparison/select: two exact Apple vectors and fail-closed cases PASS");
}

static void test_native_half_max(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   const uint16_t src[]={2,3,2,3};
   const enum agx3_alu_hint cached[]={AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE};
   assert(agx3_alu_fmax_half(AGX_ISA_ENCODING_AGX3B,0,src,cached,
                            true,32,bytes,&len));
   exact(bytes,len,"0282248210c0");
   const enum agx3_alu_hint dead[]={AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD};
   assert(agx3_alu_fmax_half(AGX_ISA_ENCODING_AGX3B,2,src,dead,
                            false,32,bytes,&len));
   exact(bytes,len,"12021c0210c0");
   const uint16_t wait2_src[]={2,7,2,7};
   assert(agx3_alu_fmax_half(AGX_ISA_ENCODING_AGX3B,2,wait2_src,dead,
                            false,2,bytes,&len));
   exact(bytes,len,"12021c061040");
   assert(!agx3_alu_fmax_half(AGX_ISA_ENCODING_AGX3C,0,src,cached,
                             true,0x100,bytes,&len));
   assert(!agx3_alu_fmax_half(AGX_ISA_ENCODING_AGX3B,256,src,cached,
                             true,32,bytes,&len));
   puts("Native half maximum: exact internal-clamp/dead Apple FSEL vectors and fail-closed bounds PASS");
}

/*
 * Apple's own applegpu_g17p bytes for the 32-bit tied minimum/maximum, from
 * `metal-nt`.  These are the shapes NIR produces for normalize(), clamp(),
 * saturate() and discard, none of which had an encoding before.
 */
static void test_native_f32_minmax(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   /* o[i] = fmax(a[i], b[i]); loaded operands dead, waiting the load's slot. */
   const uint16_t rr[] = {2, 4, 2, 4};
   const enum agx3_alu_hint dead[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
   };
   assert(agx3_alu_fmax_f32(AGX_ISA_ENCODING_AGX3B, 2, rr, dead, false, 32,
                            bytes, &len));
   exact(bytes, len, "12031e0500c0");
   assert(agx3_alu_fmin_f32(AGX_ISA_ENCODING_AGX3B, 2, rr, dead, false, 32,
                            bytes, &len));
   exact(bytes, len, "12031e0501c0");

   /* The same maximum with the second operand carried as a minifloat.  The
    * immediate byte is a plain agx_minifloat_encode() of the constant. */
   static const struct {
      uint8_t minifloat;
      const char *bytes;
   } imm[] = {
      {0x00, "12030a8100c2"},   /*  0.0f      */
      {0x10, "12030a9100c2"},   /*  0.25f     */
      {0x20, "12030aa100c2"},   /*  0.5f      */
      {0x28, "12030aa900c2"},   /*  0.75f     */
      {0x30, "12030ab100c2"},   /*  1.0f      */
      {0x34, "12030ab500c2"},   /*  1.25f     */
      {0x38, "12030ab900c2"},   /*  1.5f      */
      {0x40, "12030ac100c2"},   /*  2.0f      */
      {0x48, "12030ac900c2"},   /*  3.0f      */
      {0x50, "12030ad100c2"},   /*  4.0f      */
      {0x60, "12030ae100c2"},   /*  8.0f      */
      {0x70, "12030af100c2"},   /* 16.0f      */
      {0x08, "12030a8900c2"},   /*  0.125f    */
      {0x01, "12030a8110c2"},   /*  0.015625f */
      {0x7f, "12030aff10c2"},   /* 31.0f      */
      {0x3f, "12030abf10c2"},   /*  1.9375f   */
      {0xa0, "12031aa100c2"},   /* -0.5f      */
      {0xb0, "12031ab100c2"},   /* -1.0f      */
   };
   for (unsigned i = 0; i < sizeof(imm) / sizeof(imm[0]); ++i) {
      assert(agx3_alu_fmax_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2,
                                   imm[i].minifloat, AGX3_ALU_HINT_DISCARD,
                                   false, 32, bytes, &len));
      exact(bytes, len, imm[i].bytes);
   }
   assert(agx3_alu_fmin_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x00,
                                AGX3_ALU_HINT_DISCARD, false, 32, bytes, &len));
   exact(bytes, len, "12030a8101c2");
   assert(agx3_alu_fmin_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x30,
                                AGX3_ALU_HINT_DISCARD, false, 32, bytes, &len));
   exact(bytes, len, "12030ab101c2");

   /* Eight simultaneously-live maxima against zero: Apple's own register and
    * lifetime assignment, from one kernel. */
   static const struct {
      uint16_t dest, src;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *bytes;
   } live[] = {
      { 8, 16, AGX3_ALU_HINT_CACHE,   true, 32, "4291228100c2"},
      {12, 18, AGX3_ALU_HINT_CACHE,   true,  0, "629322810002"},
      {12, 20, AGX3_ALU_HINT_CACHE,   true,  0, "629522810002"},
      {14, 22, AGX3_ALU_HINT_CACHE,   true,  0, "729722810002"},
      { 0,  0, AGX3_ALU_HINT_DISCARD, true,  0, "02012a810002"},
      {28,  2, AGX3_ALU_HINT_DISCARD, true,  0, "e2032a810002"},
      { 4,  4, AGX3_ALU_HINT_DISCARD, true,  0, "22052a810002"},
      { 6,  6, AGX3_ALU_HINT_DISCARD, true,  0, "32072a810002"},
   };
   for (unsigned i = 0; i < sizeof(live) / sizeof(live[0]); ++i) {
      assert(agx3_alu_fmax_f32_imm(AGX_ISA_ENCODING_AGX3B, live[i].dest,
                                   live[i].src, 0x00, live[i].hint,
                                   live[i].dest_cache, live[i].wait, bytes,
                                   &len));
      exact(bytes, len, live[i].bytes);
   }

   /* A maximum whose operand came from a slot-1 load, also Apple's. */
   assert(agx3_alu_fmax_f32(AGX_ISA_ENCODING_AGX3B, 2, rr, dead, false, 1,
                            bytes, &len));
   exact(bytes, len, "12031e050020");

   /* Fail closed: wrong encoding, out-of-range registers, a lifetime hint on
    * an immediate, and a wait spanning two slots. */
   const enum agx3_alu_hint none[] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   /* M5 uses a different four-operand fseli32 field layout. Zero can be
    * selected directly; nonzero floating immediates remain rejected. */
   assert(agx3_alu_fmax_f32_imm(AGX_ISA_ENCODING_AGX3C,2,2,0,
                                AGX3_ALU_HINT_DISCARD,false,0,bytes,&len));
   exact(bytes,len,"3202070a2300a00214808a4c0200");
   const uint16_t m5_rr[4] = {0,2,0,2};
   assert(agx3_alu_fmin_f32(AGX_ISA_ENCODING_AGX3C,2,m5_rr,none,false,129,bytes,&len));
   exact(bytes,len,"3204470b2300a002900001400040");
   assert(!agx3_alu_fmax_f32_imm(AGX_ISA_ENCODING_AGX3C,2,2,0x30,
                                 AGX3_ALU_HINT_NONE,false,0,bytes,&len));
   assert(!agx3_alu_fmax_f32(AGX_ISA_ENCODING_AGX2, 2, rr, none, false, 0,
                             bytes, &len));
   assert(!agx3_alu_fmax_f32(AGX_ISA_ENCODING_AGX3B, 256, rr, none, false, 0,
                             bytes, &len));
   assert(!agx3_alu_fmax_f32(AGX_ISA_ENCODING_AGX3B, 2, rr, none, false, 33,
                             bytes, &len));
   assert(!agx3_alu_fmax_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x00,
                                 AGX3_ALU_HINT_NONE, false, 3, bytes, &len));
   puts("Native f32 minimum/maximum: exact Apple register, minifloat and "
        "lifetime vectors plus fail-closed bounds PASS");
}

/*
 * A 32-bit float compared against a minifloat immediate, selecting two 32-bit
 * registers -- what an ordinary `x < k ? a : b` becomes.
 *
 * Apple's own bytes for `(a[i] OP 0.5f) ? c[i] : d[i]`, from
 * `metal-nt -arch applegpu_g17p`.  Six kernels differing only in the operator;
 * only the condition changes, and `!=` is `==` with the selected operands
 * exchanged:
 *
 *   <    12 03 0f a1 22 06 03 02 20 0a
 *   >    12 03 0f a1 22 06 02 02 20 0a
 *   >=   12 03 0f a1 26 06 02 02 20 0a
 *   <=   12 03 0f a1 26 06 03 02 20 0a
 *   ==   12 03 0f a1 26 06 00 02 20 0a
 *   !=   12 03 0f a1 26 08 00 02 20 06
 *
 * Those witnesses route their selected operands through Apple's kind-3 slots;
 * the encoder emits ordinary registers, which changes only the operand bits.
 * The condition lives in byte 4 bit 2 and byte 6, and the test pins those
 * against the witnesses directly as well as pinning the whole instruction.
 */
static void test_native_f32_minifloat_select(void)
{
   static const struct {
      unsigned condition;
      const char *bytes;
      uint8_t apple_byte4, apple_byte6;
   } cases[] = {
      {0 /* eq */, "12030fa106060002000a", 0x26, 0x00},
      {1 /* lt */, "12030fa102060302000a", 0x22, 0x03},
      {2 /* gt */, "12030fa102060202000a", 0x22, 0x02},
      {5 /* ge */, "12030fa106060202000a", 0x26, 0x02},
      {6 /* le */, "12030fa106060302000a", 0x26, 0x03},
   };
   const enum agx3_alu_hint hints[4] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };

   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20,
                                          false, 6, 10, hints, cases[i].condition,
                                          false, 0, bytes, &len));
      exact(bytes, len, cases[i].bytes);

      /* The condition bits are Apple's, not ours. */
      assert((bytes[4] & 0x04) == (cases[i].apple_byte4 & 0x04));
      assert(bytes[6] == cases[i].apple_byte6);
   }

   /* Fresh current-toolchain applegpu_g17p kernels for `0.5 OP fabs(x)`.
    * Apple reverses the operands, places fabs(x) in source0, and uses the
    * long endpoint whose field47 is the absolute modifier. */
   static const struct {
      unsigned condition;
      const char *bytes;
   } absolute[] = {
      {6 /* le */, "020307a107040302000600800100"},
      {5 /* ge */, "020307a107040202000600800100"},
      {2 /* gt */, "020307a103040202000600800100"},
      {1 /* lt */, "020307a103040302000600800100"},
      {0 /* eq */, "020307a107040002000600800100"},
   };
   const enum agx3_alu_hint absolute_hints[4] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   for (unsigned i = 0; i < sizeof(absolute) / sizeof(absolute[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_fcmp_select_f32_imm(
         AGX_ISA_ENCODING_AGX3B, 0, 2, 0x20, true, 4, 6,
         absolute_hints, absolute[i].condition, false, 32, bytes, &len));
      exact(bytes, len, absolute[i].bytes);
   }

   /* Apple spells != as EQ with the selected values exchanged here too. */
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   assert(agx3_alu_fcmp_select_f32_imm(
      AGX_ISA_ENCODING_AGX3B, 0, 2, 0x20, true, 6, 4,
      absolute_hints, 0, false, 32, bytes, &len));
   exact(bytes, len, "020307a107060002000400800100");

   /* The minifloat spans the same field as every other fsel immediate. */
   assert(agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x00, false, 6,
                                       10, hints, 1, false, 0, bytes, &len));
   exact(bytes, len, "12030f8102060302000a");
   assert(agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0xff, false, 6,
                                       10, hints, 1, false, 0, bytes, &len));
   exact(bytes, len, "12031fff02061302000a");

   /* Waits, lifetimes and the cached destination stay orthogonal. */
   assert(agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, 6,
                                       10, hints, 2, true, 32, bytes, &len));
   exact(bytes, len, "12032fa1020602c2000a");

   /* Fail closed: the wrong encoding, an unwitnessed condition, a wait
    * spanning two slots, an odd register, and a lifetime on the immediate. */
   assert(!agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3C, 2, 2, 0x20, false, 6,
                                        10, hints, 1, false, 0, bytes, &len));
   assert(!agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, 6,
                                        10, hints, 3, false, 0, bytes, &len));
   assert(!agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, 6,
                                        10, hints, 7, false, 0, bytes, &len));
   assert(!agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, 6,
                                        10, hints, 1, false, 3, bytes, &len));
   assert(!agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 3, 2, 0x20, false, 6,
                                        10, hints, 1, false, 0, bytes, &len));
   const enum agx3_alu_hint bad[4] = {
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   assert(!agx3_alu_fcmp_select_f32_imm(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, 6,
                                        10, bad, 1, false, 0, bytes, &len));
   assert(!agx3_alu_fcmp_select_f32_imm(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, true, 6, 10,
      absolute_hints, 1, false, 1, bytes, &len));
   assert(agx3_alu_fcmp_select_f32_imm(
      AGX_ISA_ENCODING_AGX3B, 254, 252, 0xff, true, 250, 248,
      absolute_hints, 1, true, 0, bytes, &len));
   assert(!agx3_alu_fcmp_select_f32_imm(
      AGX_ISA_ENCODING_AGX3B, 254, 253, 0xff, true, 250, 248,
      absolute_hints, 1, true, 0, bytes, &len));
   const enum agx3_alu_hint gtk_hints[4] = {
      AGX3_ALU_HINT_CACHE, AGX3_ALU_HINT_NONE,
      AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
   };
   assert(agx3_alu_fcmp_select_f32_imm(
      AGX_ISA_ENCODING_AGX3B, 20, 12, 0x20, true, 30, 20,
      gtk_hints, 6, false, 0, bytes, &len));
   exact(bytes, len, "a28d07a1071e0302001400000100");
   puts("Native f32 minifloat select: Apple's plain and absolute conditions, "
        "immediate span, wait/lifetime orthogonality and fail-closed PASS");
}

static void test_native_half_clamp(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
   const uint16_t max_src[]={2,3,2,3};
   const enum agx3_alu_hint max_hints[]={AGX3_ALU_HINT_CACHE,
      AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE};
   assert(agx3_alu_fmax_half(AGX_ISA_ENCODING_AGX3B,0,max_src,max_hints,
                            true,32,bytes,&len));
   exact(bytes,len,"0282248210c0");

   const uint16_t min_src[]={0,4,0,4};
   const enum agx3_alu_hint min_hints[]={AGX3_ALU_HINT_DISCARD,
      AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE};
   assert(agx3_alu_fmin_half(AGX_ISA_ENCODING_AGX3B,0,min_src,min_hints,
                            false,0,bytes,&len));
   exact(bytes,len,"02000c840100");
   puts("Native half clamp: exact internal Apple FSEL plus canonical no-wait tail PASS");
}

static void
test_native_reciprocal_widths(void)
{
   static const struct {
      uint16_t dest, src, wait;
      bool dest_half, src_half, src_abs, src_neg, dest_cache;
      enum agx3_alu_hint hint;
      const char *bytes;
   } cases[] = {
      /* The four independent destination/source width combinations. */
      {2, 2, 32, true, true, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af005602020410463000"},
      {0, 0, 32, false, true, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af005600020010442000"},
      {2, 2, 32, true, false, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af0056020204104a3000"},
      {0, 0, 32, false, false, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af005600020010482000"},

      /* Source arithmetic modifiers on GTK's float-from-half form. */
      {0, 0, 32, false, true, true, false, false,
       AGX3_ALU_HINT_DISCARD, "af005600020010446000"},
      {0, 0, 32, false, true, false, true, false,
       AGX3_ALU_HINT_DISCARD, "af00560002001044a000"},
      {0, 0, 32, false, true, true, true, false,
       AGX3_ALU_HINT_DISCARD, "af00560002001044e000"},

      /* Source/result lifetime fields are independent. */
      {0, 2, 32, false, true, false, false, false,
       AGX3_ALU_HINT_CACHE, "af005600020402442000"},
      {0, 0, 32, false, true, false, false, true,
       AGX3_ALU_HINT_DISCARD, "af005600030010442000"},
      {0, 2, 32, false, true, false, false, true,
       AGX3_ALU_HINT_CACHE, "af005600030402442000"},

      /* The exact shape three.js was refused on: a full-precision reciprocal
       * of an absolute value, cached, with a destination above the old 128
       * gate.  Bytes from Apple's own composer tables for f32rcp ep0. */
      {140, 86, 0, false, false, true, false, false,
       AGX3_ALU_HINT_CACHE, "af00548c02ac02486000"},

      /* Odd half input, no dependency, and texture producer slot zero. */
      {0, 3, 32, false, true, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af005600020610442000"},
      {0, 0, 0, false, true, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af005400020010442000"},
      {0, 0, 1, false, true, false, false, false,
       AGX3_ALU_HINT_DISCARD, "af105400020010442000"},
   };

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      assert(agx3_alu_rcp(
         AGX_ISA_ENCODING_AGX3B, cases[i].dest, cases[i].src,
         cases[i].dest_half, cases[i].src_half, cases[i].src_abs,
         cases[i].src_neg, cases[i].hint, cases[i].dest_cache,
         cases[i].wait, bytes, &len));
      exact(bytes, len, cases[i].bytes);
   }

   unsigned forms = 0;
   for (unsigned dest_half = 0; dest_half < 2; ++dest_half) {
      for (unsigned src_half = 0; src_half < 2; ++src_half) {
         for (unsigned modifiers = 0; modifiers < 4; ++modifiers) {
            for (unsigned hint = 0; hint <= AGX3_ALU_HINT_CACHE; ++hint) {
               for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache) {
                  for (unsigned wait = 0; wait < 64; ++wait) {
                     const bool ok = agx3_alu_rcp(
                        AGX_ISA_ENCODING_AGX3B, dest_half ? 127 : 126,
                        src_half ? 127 : 126, dest_half, src_half,
                        modifiers & 1, modifiers & 2,
                        (enum agx3_alu_hint)hint, dest_cache, wait, bytes,
                        &len);
                     if (!ok)
                        fprintf(stderr,
                                "RCP compose failed: dh=%u sh=%u mod=%u "
                                "hint=%u dc=%u wait=%u\n",
                                dest_half, src_half, modifiers, hint,
                                dest_cache, wait);
                     assert(ok);
                     assert(len == 10);
                     ++forms;
                  }
               }
            }
         }
      }
   }
   assert(forms == 6144);

   /* G17G composes the same cross-product over eight producer slots. */
   unsigned gpu1_forms = 0;
   for (unsigned dest_half = 0; dest_half < 2; ++dest_half) {
      for (unsigned src_half = 0; src_half < 2; ++src_half) {
         for (unsigned modifiers = 0; modifiers < 4; ++modifiers) {
            for (unsigned hint = 0; hint <= AGX3_ALU_HINT_CACHE; ++hint) {
               for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache) {
                  for (unsigned wait = 0; wait < 256; ++wait) {
                     assert(agx3_alu_rcp(
                        AGX_ISA_ENCODING_AGX3C, dest_half ? 127 : 126,
                        src_half ? 127 : 126, dest_half, src_half,
                        modifiers & 1, modifiers & 2,
                        (enum agx3_alu_hint)hint, dest_cache, wait, bytes,
                        &len));
                     assert(len == 10);
                     ++gpu1_forms;
                  }
               }
            }
         }
      }
   }
   assert(gpu1_forms == 24576);

   /* The register fields accept [0, 191]; 192 is where Apple's tables stop,
    * and where the refusal now is.  This positive case has to run before the
    * refusal sweep below, which checks that a refusal leaves `len` alone. */
   assert(agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 190, 190, true, true, false, false,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(len == 10);

   memset(bytes, 0xa5, sizeof(bytes));
   len = 0xa5;
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 192, 0, true, true, false, false,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 0, 192, true, true, false, false,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 1, 0, false, true, false, false,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 0, 1, true, false, false, false,
      AGX3_ALU_HINT_NONE, false, 0, bytes, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 0, 0, true, true, false, false,
      (enum agx3_alu_hint)3, false, 0, bytes, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 0, 0, true, true, false, false,
      AGX3_ALU_HINT_NONE, false, 64, bytes, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 0, 0, true, true, false, false,
      AGX3_ALU_HINT_NONE, false, 0, NULL, &len));
   assert(!agx3_alu_rcp(
      AGX_ISA_ENCODING_AGX3B, 0, 0, true, true, false, false,
      AGX3_ALU_HINT_NONE, false, 0, bytes, NULL));
   assert(len == 0xa5);
   for (unsigned i = 0; i < sizeof(bytes); ++i)
      assert(bytes[i] == 0xa5);

   puts("Native reciprocal widths/modifiers: thirteen exact Apple vectors, "
        "6144 field combinations and strict bounds PASS");
}

static void test_native_half_rsqrt(void)
{
   static const struct {
      uint16_t dest, src;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *bytes;
   } cases[] = {
      {3,2,AGX3_ALU_HINT_NONE,    false,32,"af01560302048c600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false,32,"af0156020204ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false, 0,"af0154020204ac600000"},
      {3,3,AGX3_ALU_HINT_DISCARD, false, 0,"af0154030206ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, true, 32,"af0156020304ac600000"},
      {1,1,AGX3_ALU_HINT_DISCARD, false, 1,"af1154010202ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false, 2,"af2154020204ac600000"},
      {3,3,AGX3_ALU_HINT_DISCARD, false, 4,"af4154030206ac600000"},
      {6,6,AGX3_ALU_HINT_DISCARD, false, 8,"af815406020cac600000"},
      {7,7,AGX3_ALU_HINT_DISCARD, false,16,"af015507020eac600000"},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_hrsqrt(AGX_ISA_ENCODING_AGX3B, cases[i].dest,
                             cases[i].src, cases[i].hint,
                             cases[i].dest_cache, cases[i].wait,
                             bytes, &len));
      exact(bytes, len, cases[i].bytes);
   }

   for (unsigned bad = 0; bad < 4; ++bad) {
      uint16_t dest = 2, src = 2, wait = 0;
      enum agx3_alu_hint hint = AGX3_ALU_HINT_DISCARD;
      if (bad == 0) dest = 256;
      if (bad == 1) src = 256;
      if (bad == 2) hint = (enum agx3_alu_hint)3;
      if (bad == 3) wait = 64;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0xa5;
      memset(bytes, 0xa5, sizeof(bytes));
      assert(!agx3_alu_hrsqrt(AGX_ISA_ENCODING_AGX3B, dest, src, hint, false,
                              wait, bytes, &len));
      assert(len == 0xa5);
      for (unsigned j = 0; j < sizeof(bytes); ++j)
         assert(bytes[j] == 0xa5);
   }
   puts("Native half reciprocal sqrt: exact Apple wait-bank vectors and bounds PASS");
}

static void test_native_half_exp2(void)
{
   static const struct {
      uint16_t dest, src;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *bytes;
      bool src_neg;
   } cases[] = {
      {3,2,AGX3_ALU_HINT_NONE,    false,32,"af02560302048c600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false,32,"af0256020204ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false, 0,"af0254020204ac600000"},
      {3,3,AGX3_ALU_HINT_DISCARD, false, 0,"af0254030206ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, true, 32,"af0256020304ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false,32,"af0256020204ac600100", true},
      {1,1,AGX3_ALU_HINT_DISCARD, false, 1,"af1254010202ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false, 2,"af2254020204ac600000"},
      {3,3,AGX3_ALU_HINT_DISCARD, false, 4,"af4254030206ac600000"},
      {6,6,AGX3_ALU_HINT_DISCARD, false, 8,"af825406020cac600000"},
      {7,7,AGX3_ALU_HINT_DISCARD, false,16,"af025507020eac600000"},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_hexp2_neg(AGX_ISA_ENCODING_AGX3B, cases[i].dest,
                                cases[i].src, cases[i].hint,
                                cases[i].dest_cache, cases[i].wait,
                                cases[i].src_neg, bytes, &len));
      exact(bytes, len, cases[i].bytes);
   }

   for (unsigned bad = 0; bad < 4; ++bad) {
      uint16_t dest = 2, src = 2, wait = 0;
      enum agx3_alu_hint hint = AGX3_ALU_HINT_DISCARD;
      if (bad == 0) dest = 256;
      if (bad == 1) src = 256;
      if (bad == 2) hint = (enum agx3_alu_hint)3;
      if (bad == 3) wait = 64;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0xa5;
      memset(bytes, 0xa5, sizeof(bytes));
      assert(!agx3_alu_hexp2(AGX_ISA_ENCODING_AGX3B, dest, src, hint, false,
                             wait, bytes, &len));
      assert(len == 0xa5);
      for (unsigned j = 0; j < sizeof(bytes); ++j)
         assert(bytes[j] == 0xa5);
   }
   puts("Native half exp2: exact Apple wait-bank vectors and bounds PASS");
}

static void test_native_half_log2(void)
{
   static const struct {
      uint16_t dest, src;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *bytes;
   } cases[] = {
      {3,2,AGX3_ALU_HINT_NONE,    false,32,"2f02560302048c600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false,32,"2f0256020204ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false, 0,"2f0254020204ac600000"},
      {3,3,AGX3_ALU_HINT_DISCARD, false, 0,"2f0254030206ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, true, 32,"2f0256020304ac600000"},
      {1,1,AGX3_ALU_HINT_DISCARD, false, 1,"2f1254010202ac600000"},
      {2,2,AGX3_ALU_HINT_DISCARD, false, 2,"2f2254020204ac600000"},
      {3,3,AGX3_ALU_HINT_DISCARD, false, 4,"2f4254030206ac600000"},
      {6,6,AGX3_ALU_HINT_DISCARD, false, 8,"2f825406020cac600000"},
      {7,7,AGX3_ALU_HINT_DISCARD, false,16,"2f025507020eac600000"},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_hlog2(AGX_ISA_ENCODING_AGX3B, cases[i].dest,
                            cases[i].src, cases[i].hint,
                            cases[i].dest_cache, cases[i].wait,
                            bytes, &len));
      exact(bytes, len, cases[i].bytes);
   }

   for (unsigned bad = 0; bad < 4; ++bad) {
      uint16_t dest = 2, src = 2, wait = 0;
      enum agx3_alu_hint hint = AGX3_ALU_HINT_DISCARD;
      if (bad == 0) dest = 256;
      if (bad == 1) src = 256;
      if (bad == 2) hint = (enum agx3_alu_hint)3;
      if (bad == 3) wait = 64;
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0xa5;
      memset(bytes, 0xa5, sizeof(bytes));
      assert(!agx3_alu_hlog2(AGX_ISA_ENCODING_AGX3B, dest, src, hint, false,
                             wait, bytes, &len));
      assert(len == 0xa5);
      for (unsigned j = 0; j < sizeof(bytes); ++j)
         assert(bytes[j] == 0xa5);
   }
   puts("Native half log2: exact Apple wait-bank vectors and bounds PASS");
}

static void test_native_mixed_width_log2(void)
{
   /* Fresh current-Xcode applegpu_g17p kernels. f10 and f26 independently
    * select the destination and source width. The matrix also pins an odd
    * half source, both lifetimes, result caching, modifiers, and wait/no-wait
    * without relying on a synthetic field composition. */
   static const struct {
      uint16_t dest, src;
      bool dest_half, src_half, src_abs, src_neg;
      enum agx3_alu_hint hint;
      bool dest_cache;
      uint16_t wait;
      const char *bytes;
   } cases[] = {
      {0,0,true, true, false,false,AGX3_ALU_HINT_DISCARD,true, 32,
       "2f0256000300ac600000"},
      {0,2,true, true, false,false,AGX3_ALU_HINT_CACHE,  true, 32,
       "2f02560003048e600000"},
      {0,0,false,true, false,false,AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256000200a8400000"},
      {0,2,false,true, false,false,AGX3_ALU_HINT_CACHE,  false,32,
       "2f02560002048a400000"},
      {0,3,false,true, false,false,AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256000206a8400000"},
      {0,0,false,true, false,false,AGX3_ALU_HINT_DISCARD,false,0,
       "2f0254000200a8400000"},
      {0,0,false,true, true, false,AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256000200a8c00000"},
      {0,0,false,true, false,true, AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256000200a8400100"},
      {0,0,false,true, true, true, AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256000200a8c00100"},
      {0,0,false,true, false,false,AGX3_ALU_HINT_DISCARD,true, 32,
       "2f0256000300a8400000"},
      {2,2,true, false,false,false,AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256020204b4600000"},
      {2,4,true, false,false,false,AGX3_ALU_HINT_CACHE,  false,32,
       "2f025602020896600000"},
      {2,2,true, false,true, false,AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256020204b4e00000"},
      {2,2,true, false,false,true, AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256020204b4600100"},
      {2,2,true, false,true, true, AGX3_ALU_HINT_DISCARD,false,32,
       "2f0256020204b4e00100"},
   };

   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
      assert(agx3_alu_log2(
         AGX_ISA_ENCODING_AGX3B, cases[i].dest, cases[i].src,
         cases[i].dest_half, cases[i].src_half, cases[i].src_abs,
         cases[i].src_neg, cases[i].hint, cases[i].dest_cache,
         cases[i].wait, bytes, &len));
      exact(bytes, len, cases[i].bytes);
   }

   /* The recovered operand grammar names halves 0..191. A full scalar
    * consumes an aligned pair; a half may independently select the final odd
    * half. */
   for (unsigned dest_half = 0; dest_half < 2; ++dest_half) {
      for (unsigned src_half = 0; src_half < 2; ++src_half) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         assert(agx3_alu_log2(
            AGX_ISA_ENCODING_AGX3B, dest_half ? 191 : 190,
            src_half ? 191 : 190, dest_half, src_half, false, false,
            AGX3_ALU_HINT_NONE, false, 0,
            bytes, &len));
         assert(len == 10);
      }
   }

   struct {
      enum agx_isa_encoding enc;
      uint16_t dest, src;
      bool dest_half, src_half;
      enum agx3_alu_hint hint;
      uint16_t wait;
   } bad[] = {
      {AGX_ISA_ENCODING_AGX3B, 256, 0,   true, true, AGX3_ALU_HINT_NONE,0},
      {AGX_ISA_ENCODING_AGX3B, 0,   256, false,true, AGX3_ALU_HINT_NONE,0},
      {AGX_ISA_ENCODING_AGX3B, 192, 0,   true, true, AGX3_ALU_HINT_NONE,0},
      {AGX_ISA_ENCODING_AGX3B, 0,   192, false,true, AGX3_ALU_HINT_NONE,0},
      {AGX_ISA_ENCODING_AGX3B, 255, 0,   false,true, AGX3_ALU_HINT_NONE,0},
      {AGX_ISA_ENCODING_AGX3B, 0,   255, false,false,AGX3_ALU_HINT_NONE,0},
      {AGX_ISA_ENCODING_AGX3B, 0,   0,   false,true,
       (enum agx3_alu_hint)3, 0},
      {AGX_ISA_ENCODING_AGX3B, 0,   0,   false,false,AGX3_ALU_HINT_NONE,64},
   };
   for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0xa5;
      memset(bytes, 0xa5, sizeof(bytes));
      assert(!agx3_alu_log2(
         bad[i].enc, bad[i].dest, bad[i].src, bad[i].dest_half,
         bad[i].src_half, false, false, bad[i].hint, false, bad[i].wait,
         bytes, &len));
      assert(len == 0xa5);
      for (unsigned j = 0; j < sizeof(bytes); ++j)
         assert(bytes[j] == 0xa5);
   }

   puts("Native mixed-width log2: fifteen exact Apple vectors and strict width/register guards PASS");
}

static void test_native_half_floor(void)
{
   static const struct { uint16_t d,s; enum agx3_alu_hint h; bool dc;
      uint16_t w; const char *b; } cases[] = {
      {3,2,AGX3_ALU_HINT_NONE,false,32,"2f00560302048c600200"},
      {2,2,AGX3_ALU_HINT_DISCARD,false,32,"2f0056020204ac600200"},
      {2,2,AGX3_ALU_HINT_DISCARD,false,0,"2f0054020204ac600200"},
      {3,3,AGX3_ALU_HINT_DISCARD,false,0,"2f0054030206ac600200"},
      {2,2,AGX3_ALU_HINT_DISCARD,true,32,"2f0056020304ac600200"},
   };
   for (unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_hfloor(AGX_ISA_ENCODING_AGX3B,cases[i].d,cases[i].s,
                             cases[i].h,cases[i].dc,cases[i].w,bytes,&len));
      exact(bytes,len,cases[i].b);
   }
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len=0xa5;
   memset(bytes,0xa5,sizeof(bytes));
   assert(!agx3_alu_hfloor(AGX_ISA_ENCODING_AGX3C,2,2,
                           AGX3_ALU_HINT_DISCARD,false,0,bytes,&len));
   assert(agx3_alu_hfloor(AGX_ISA_ENCODING_AGX3B,2,2,
                          AGX3_ALU_HINT_CACHE,false,0,bytes,&len));
   exact(bytes,len,"2f00540202048e600200");
   assert(agx3_alu_hfloor(AGX_ISA_ENCODING_AGX3B,2,2,
                          AGX3_ALU_HINT_DISCARD,false,1,bytes,&len));
   exact(bytes,len,"2f1054020204ac600200");
   assert(agx3_alu_htrunc(AGX_ISA_ENCODING_AGX3B,2,2,
                          AGX3_ALU_HINT_DISCARD,false,1,bytes,&len));
   exact(bytes,len,"2f1054020204ac600600");
   assert(agx3_alu_hroundeven(AGX_ISA_ENCODING_AGX3B,2,2,
                              AGX3_ALU_HINT_DISCARD,false,1,bytes,&len));
   exact(bytes,len,"2f1054020204ac600000");
   len=0xa5;
   memset(bytes,0xa5,sizeof(bytes));
   assert(!agx3_alu_hfloor(AGX_ISA_ENCODING_AGX3B,2,2,
                           AGX3_ALU_HINT_DISCARD,false,64,bytes,&len));
   assert(len==0xa5);
   for(unsigned i=0;i<sizeof(bytes);++i) assert(bytes[i]==0xa5);
   puts("Native half rounding modes: exact Apple cache/wait vectors and bounds PASS");
}

static void test_native_half_ceil(void)
{
   static const struct { uint16_t d,s; enum agx3_alu_hint h; bool dc;
      uint16_t w; const char *b; } cases[] = {
      {3,2,AGX3_ALU_HINT_NONE,false,32,"2f00560302048c600400"},
      {2,2,AGX3_ALU_HINT_DISCARD,false,32,"2f0056020204ac600400"},
      {2,2,AGX3_ALU_HINT_DISCARD,false,0,"2f0054020204ac600400"},
      {3,3,AGX3_ALU_HINT_DISCARD,false,0,"2f0054030206ac600400"},
      {2,2,AGX3_ALU_HINT_DISCARD,true,32,"2f0056020304ac600400"},
   };
   for (unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_hceil(AGX_ISA_ENCODING_AGX3B,cases[i].d,cases[i].s,
                            cases[i].h,cases[i].dc,cases[i].w,bytes,&len));
      exact(bytes,len,cases[i].b);
   }
   uint8_t bytes[AGX3_ALU_MAX_LENGTH],len=0xa5;
   memset(bytes,0xa5,sizeof(bytes));
   assert(!agx3_alu_hceil(AGX_ISA_ENCODING_AGX3C,2,2,
                          AGX3_ALU_HINT_DISCARD,false,0,bytes,&len));
   assert(agx3_alu_hceil(AGX_ISA_ENCODING_AGX3B,2,2,
                         AGX3_ALU_HINT_CACHE,false,0,bytes,&len));
   exact(bytes,len,"2f00540202048e600400");
   assert(agx3_alu_hceil(AGX_ISA_ENCODING_AGX3B,2,2,
                         AGX3_ALU_HINT_DISCARD,false,1,bytes,&len));
   exact(bytes,len,"2f1054020204ac600400");
   len=0xa5;
   memset(bytes,0xa5,sizeof(bytes));
   assert(!agx3_alu_hceil(AGX_ISA_ENCODING_AGX3B,2,2,
                          AGX3_ALU_HINT_DISCARD,false,64,bytes,&len));
   assert(len==0xa5);
   for(unsigned i=0;i<sizeof(bytes);++i) assert(bytes[i]==0xa5);
   puts("Native half ceil: exact Apple cache/wait vectors and bounds PASS");
}

static void test_native_high_float_registers(void)
{
   /* Original native80-value kernels force retained ALU values into the upper
    * register bank. These are complete-instruction byte vectors, not patched
    * low-register examples; they retain native hints and scoreboard waits. */
   static const struct {
      enum agx3_alu_op op; uint16_t dest,src[3];
      enum agx3_alu_hint hints[3]; uint16_t wait; const char *bytes;
   } vectors[]={
      {AGX3_ALU_FMUL,168,{0,2,0},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},4,"498345810070"},
      {AGX3_ALU_FMUL,172,{4,6,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},0,"698755050010"},
      {AGX3_ALU_FMUL,174,{6,160,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},0,"79a155070011"},
      {AGX3_ALU_FMUL,142,{142,128,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},0,"7981150f0015"},
      {AGX3_ALU_FMUL,110,{110,152,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},0,"7999d56f0001"},
      {AGX3_ALU_FMUL,70,{70,56,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},0,"39b99147"},
      {AGX3_ALU_FMUL,38,{38,24,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},0,"39995127"},
      {AGX3_ALU_FMUL,4,{184,186,0},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_NONE},0,"293b1d390005"},
      {AGX3_ALU_FMA,168,{2,0,4},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_CACHE},4,"4981468301840270"},
      {AGX3_ALU_FMA,172,{6,4,152},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE},0,"69054e8741980210"},
      {AGX3_ALU_FMA,174,{152,6,154},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE},0,"79074e99419a0214"},
      {AGX3_ALU_FMA,150,{128,150,130},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE},0,"b9170e8141820215"},
      {AGX3_ALU_FMA,118,{160,86,162},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE},0,"b957cea141a20204"},
      {AGX3_ALU_FMA,70,{48,70,50},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE},0,"39478eb100b2"},
      {AGX3_ALU_FMA,38,{104,24,106},{AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_CACHE},0,"39194ee900ea"},
      {AGX3_ALU_FMA,2,{102,100,104},{AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD},0,"19651e678068"},
   };
   for(unsigned i=0;i<sizeof(vectors)/sizeof(vectors[0]);i++) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH],len;
      assert(agx3_alu_float_reg(AGX_ISA_ENCODING_AGX3B,vectors[i].op,vectors[i].dest,
         vectors[i].src,vectors[i].hints,vectors[i].wait,bytes,&len));
      exact(bytes,len,vectors[i].bytes);
   }
   puts("Native full-bank FMUL/FMA:16 exact high-register/lifetime/wait vectors PASS");
}

static void test_gpu1_modified_add(void)
{
   const uint8_t expected[4][8] = {
      {0x09,0x05,0x1c,0x81,0x01,0x02,0x00,0x02},
      {0xf9,0xbd,0x54,0x81,0x01,0x33,0x00,0x82},
      {0x09,0xa1,0x14,0x05,0x81,0x00,0x00,0x02},
      {0x09,0xb1,0x14,0x05,0x80,0x08},
   };
   const uint16_t immediates[] = {0x80, 0x80, 0x20, 0x30};
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < 4; ++i) {
      assert(agx3_alu_fadd_minifloat_mod_g18g(AGX_ISA_ENCODING_AGX3C,
         i == 1 ? 190 : 0, i == 1 ? 188 : 4, immediates[i],
         i == 3, i != 3, i == 1 ? AGX3_ALU_HINT_CACHE : AGX3_ALU_HINT_DISCARD,
         i == 1 ? 0x21 : 0, out, &len));
      assert(len == (i == 3 ? 6 : 8) && !memcmp(out, expected[i], len));
   }
   assert(!agx3_alu_fadd_minifloat_mod(AGX_ISA_ENCODING_AGX3C,
      0, 4, 0x80, false, true, AGX3_ALU_HINT_NONE, 0x40, out, &len));
}

static void test_gpu1_float_compare_select(void)
{
   /* Apple applegpu_g17g 32-bit float selects, one kernel per row
    * (gpu/m5-xfb-alu7-, -alu8-, -alu10-, -alu11-reference.metal). The
    * comparand `a` is r2 or r4, the selected registers v/w are r0/r2; every
    * source is dead after the select and the load it waits on signals slot
    * 8. `fseli32` selects two registers, `fself32` a register and a
    * minifloat constant. Two constants are refused: Apple emits `fself16`
    * plus a widen for `select(0.0f, 1.0f, ...)`. */
   enum { D = AGX3_ALU_HINT_DISCARD, N = AGX3_ALU_HINT_NONE };
   static const struct {
      const char *kernel;
      struct agx3_fsel32_gpu1_desc d;
      const char *bytes;
   } vectors[] = {
      /* fseli32: select(w, v, a OP k) and select(w, v, a OP b). */
      {"alu8 fsel_reg_reg  a < -1.0", {.cmp0=8,.cmp0_hint=D,.cmp1_immediate=true,
        .cmp1=176,.sel={0,4},.sel_hint={D,D},.condition=1,.wait_mask=0x80},
       "2200074a2300a00214ac8a602040"},
      {"alu10 fsel_lt_two  a < 2.0", {.cmp0=8,.cmp0_hint=D,.cmp1_immediate=true,
        .cmp1=64,.sel={0,4},.sel_hint={D,D},.condition=1,.wait_mask=0x80},
       "2200074a2300a00214908a602040"},
      {"alu10 fsel_lt_half  a < 0.5", {.cmp0=8,.cmp0_hint=D,.cmp1_immediate=true,
        .cmp1=32,.sel={0,4},.sel_hint={D,D},.condition=1,.wait_mask=0x80},
       "2200074a2300a00214888a602040"},
      {"alu11 fsel_lt_zero  a < 0.0", {.cmp0=8,.cmp0_hint=D,.cmp1_immediate=true,
        .cmp1=0,.sel={0,4},.sel_hint={D,D},.condition=1,.wait_mask=0x80},
       "2200074a2300a00214808a602040"},
      {"alu10 fsel_ge_one", {.cmp0=8,.cmp0_hint=D,.cmp1_immediate=true,
        .cmp1=48,.sel={0,4},.sel_hint={D,D},.condition=5,.wait_mask=0x80},
       "2200074a2300a002148c8a242040"},
      {"alu10 fsel_eq_one", {.cmp0=8,.cmp0_hint=D,.cmp1_immediate=true,
        .cmp1=48,.sel={0,4},.sel_hint={D,D},.condition=0,.wait_mask=0x80},
       "2200074a2300a002148c8a202040"},
      {"alu10 fsel_gt_one", {.cmp0=8,.cmp0_hint=D,.cmp1_immediate=true,
        .cmp1=48,.sel={0,4},.sel_hint={D,D},.condition=2,.wait_mask=0x80},
       "2200474a2300a002148c8a202040"},
      {"alu10 fsel_le_one", {.cmp0=8,.cmp0_hint=D,.cmp1_immediate=true,
        .cmp1=48,.sel={0,4},.sel_hint={D,D},.condition=6,.wait_mask=0x80},
       "2200474a2300a002148c8a242040"},
      /* a != 1.0 is equality with the two registers exchanged. */
      {"alu10 fsel_ne_one", {.cmp0=8,.cmp0_hint=D,.cmp1_immediate=true,
        .cmp1=48,.sel={4,0},.sel_hint={D,D},.condition=0,.wait_mask=0x80},
       "2200074a2300a002168c8a200040"},
      {"alu10 fsel_reg_lt_reg", {.cmp0=8,.cmp0_hint=D,.cmp1=6,.cmp1_hint=D,
        .sel={0,4},.sel_hint={D,D},.condition=1,.wait_mask=0x80},
       "2200074a2300a0029401a1602040"},
      {"alu11 fsel_reg_ge_reg", {.cmp0=8,.cmp0_hint=D,.cmp1=6,.cmp1_hint=D,
        .sel={0,4},.sel_hint={D,D},.condition=5,.wait_mask=0x80},
       "2200074a2300a0029401a1242040"},
      {"alu11 fsel_reg_eq_reg", {.cmp0=8,.cmp0_hint=D,.cmp1=6,.cmp1_hint=D,
        .sel={0,4},.sel_hint={D,D},.condition=0,.wait_mask=0x80},
       "2200074a2300a0029401a1202040"},
      /* Apple's [[depth(any)]] fragment reads the result again: f11. */
      {"fragment-depth select, result cached", {.dest_cache=true,.cmp0=4,
        .cmp0_hint=D,.cmp1=0,.cmp1_hint=D,.sel={0,4},.sel_hint={D,D},
        .condition=1,.wait_mask=0x80},
       "2200072a2b00a0021400a1602040"},
      /* fself32: a minifloat on either side of the select. */
      {"alu8 fsel_zero_reg  select(v, 0.0, a < -1.0)", {.cmp0=4,.cmp0_hint=D,
        .cmp1_immediate=true,.cmp1=176,.sel_immediate={true,false},.sel={0,0},
        .sel_hint={N,D},.condition=1,.wait_mask=0x80},
       "2200032a2300a00214ac4a200000c000"},
      {"alu10 fself_one_reg  select(v, 1.0, ...)", {.cmp0=4,.cmp0_hint=D,
        .cmp1_immediate=true,.cmp1=176,.sel_immediate={true,false},.sel={48,0},
        .sel_hint={N,D},.condition=1,.wait_mask=0x80},
       "2200032a2300a00234ac4a201000c000"},
      {"alu10 fself_two_reg  select(v, 2.0, ...)", {.cmp0=4,.cmp0_hint=D,
        .cmp1_immediate=true,.cmp1=176,.sel_immediate={true,false},.sel={64,0},
        .sel_hint={N,D},.condition=1,.wait_mask=0x80},
       "2200032a2300a00214ac4a204000c000"},
      {"alu10 fself_reg_one  select(1.0, v, ...)", {.cmp0=4,.cmp0_hint=D,
        .cmp1_immediate=true,.cmp1=176,.sel_immediate={false,true},.sel={0,48},
        .sel_hint={D,N},.condition=1,.wait_mask=0x80},
       "2200032a2300a00214ac8a080006c000"},
      {"alu10 fself_zero_reg_zerocmp  a < 0.0", {.cmp0=4,.cmp0_hint=D,
        .cmp1_immediate=true,.cmp1=0,.sel_immediate={true,false},.sel={0,0},
        .sel_hint={N,D},.condition=1,.wait_mask=0x80},
       "2200032a2300a00214804a200000c000"},
      {"alu11 fself_ge", {.cmp0=4,.cmp0_hint=D,.cmp1_immediate=true,.cmp1=48,
        .sel_immediate={true,false},.sel={0,0},.sel_hint={N,D},.condition=5,
        .wait_mask=0x80},
       "2200032a2300a002148c4a2000808000"},
      {"alu11 fself_eq", {.cmp0=4,.cmp0_hint=D,.cmp1_immediate=true,.cmp1=48,
        .sel_immediate={true,false},.sel={0,0},.sel_hint={N,D},.condition=0,
        .wait_mask=0x80},
       "2200032a2300a002148c4a2000008000"},
      {"alu11 fself_gt", {.cmp0=4,.cmp0_hint=D,.cmp1_immediate=true,.cmp1=48,
        .sel_immediate={true,false},.sel={0,0},.sel_hint={N,D},.condition=2,
        .wait_mask=0x80},
       "2200432a2300a002148c4a2000008000"},
      {"alu11 fself_le", {.cmp0=4,.cmp0_hint=D,.cmp1_immediate=true,.cmp1=48,
        .sel_immediate={true,false},.sel={0,0},.sel_hint={N,D},.condition=6,
        .wait_mask=0x80},
       "2200432a2300a002148c4a2000808000"},
      {"alu11 fself_ne  = eq exchanged", {.cmp0=4,.cmp0_hint=D,
        .cmp1_immediate=true,.cmp1=48,.sel_immediate={false,true},.sel={0,0},
        .sel_hint={D,N},.condition=0,.wait_mask=0x80},
       "2200032a2300a002148c8a0800008000"},
      {"alu11 fself_zero_reg_immcmp  select(0.0, v, a < 1.0)", {.cmp0=4,
        .cmp0_hint=D,.cmp1_immediate=true,.cmp1=48,.sel_immediate={false,true},
        .sel={0,0},.sel_hint={D,N},.condition=1,.wait_mask=0x80},
       "2200032a2300a002148c8a080000c000"},
      {"alu10 fself_reg_zero_regcmp  select(0.0, v, a < b)", {.cmp0=4,
        .cmp0_hint=D,.cmp1=8,.cmp1_hint=D,.sel_immediate={false,true},.sel={0,0},
        .sel_hint={D,N},.condition=1,.wait_mask=0x80},
       "2200032a2300a0021402a1080000c000"},
      {"alu10 fself_two_reg_regcmp  select(v, 2.0, a < b)", {.cmp0=4,
        .cmp0_hint=D,.cmp1=8,.cmp1_hint=D,.sel_immediate={true,false},.sel={64,0},
        .sel_hint={N,D},.condition=1,.wait_mask=0x80},
       "2200032a2300a002140261204000c000"},
   };
   uint8_t out[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_fcmp_select32_gpu1(&vectors[i].d, out, &len));
      exact(out, len, vectors[i].bytes);
   }

   /* The general select's GPU1 branch lands on the same forms: gears'
    * `0 < x ? y : 0` is the register/zero shape above with its comparison
    * reversed. Mesa's selected immediates are integers, so only zero
    * reaches the minifloat-reading `fself32`. */
   const bool no_abs[2] = {false, false};
   const uint16_t gears[4] = {4, 0, 0, 0};
   const bool gears_imm[2] = {true, false};
   const enum agx3_alu_hint gears_hints[4] = {D, N, N, D};
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3C, 0, gears, true,
      no_abs, no_abs, gears_imm, gears_hints, 1, false, false, 0x80, out, &len));
   exact(out, len, "2200032a2300a00214804a200000c000");
   /* A 16-bit result selecting a register or zero is the general fseli16:
    * Apple's `(x > 0.0f) ? s : ushort(0)` (m5-compare-select-reference,
    * sel_f16_reg_zero). */
   uint16_t src[4] = {0, 0, 8, 0};
   const bool immediate[2] = {false, true};
   const enum agx3_alu_hint reg_zero_hints[4] = {D, N, D, N};
   assert(agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3C,
      0, src, true, no_abs, no_abs, immediate, reg_zero_hints, 2, true, false,
      0, out, &len));
   exact(out, len, "02010d81820802022080");
   const enum agx3_alu_hint hints[4] = {0};
   src[3] = 1; /* A minifloat payload is not an integer result immediate. */
   assert(!agx3_alu_fcmp_select_reg_imm(AGX_ISA_ENCODING_AGX3C,
      14, src, true, no_abs, no_abs, immediate, hints, 2, false, false, 0, out, &len));

   /* Rejection boundaries of the core itself. */
   struct agx3_fsel32_gpu1_desc bad = vectors[13].d;
   bad.sel_immediate[1] = true;
   assert(!agx3_alu_fcmp_select32_gpu1(&bad, out, &len)); /* two constants */
   bad = vectors[13].d; bad.condition = 3;
   assert(!agx3_alu_fcmp_select32_gpu1(&bad, out, &len));
   bad = vectors[13].d; bad.condition = 7; /* the NaN-true relations */
   assert(!agx3_alu_fcmp_select32_gpu1(&bad, out, &len));
   bad = vectors[13].d; bad.dest = 1;
   assert(!agx3_alu_fcmp_select32_gpu1(&bad, out, &len));
   bad = vectors[13].d; bad.cmp0 = 3;
   assert(!agx3_alu_fcmp_select32_gpu1(&bad, out, &len));
   bad = vectors[13].d; bad.cmp1 = 256;
   assert(!agx3_alu_fcmp_select32_gpu1(&bad, out, &len));
   bad = vectors[13].d; bad.sel[0] = 256;
   assert(!agx3_alu_fcmp_select32_gpu1(&bad, out, &len));
   bad = vectors[13].d; bad.sel_hint[0] = D;
   assert(!agx3_alu_fcmp_select32_gpu1(&bad, out, &len)); /* hint on a constant */
   bad = vectors[13].d; bad.wait_mask = 0x100;
   assert(!agx3_alu_fcmp_select32_gpu1(&bad, out, &len));
   bad = vectors[0].d; bad.sel[1] = 5;
   assert(!agx3_alu_fcmp_select32_gpu1(&bad, out, &len)); /* odd register */
   assert(!agx3_alu_fcmp_select32_gpu1(NULL, out, &len));
   puts("GPU1 fseli32/fself32: 26 Apple applegpu_g17g witnesses byte-exact; "
        "two constants, unwitnessed relations, odd registers, out-of-range "
        "constants, hinted constants and slot-9 waits refused PASS");
}

static void
test_gpu1_minifloat_scale(void)
{
   /* Apple G17G float4 scale, with dead and retained input vectors. */
   static const struct {
      uint16_t dest, src, immediate, wait;
      bool discard;
      uint8_t length, bytes[14];
   } vectors[] = {
      {0,0,0x20,128,true,14,
       {0x29,0x00,0x05,0x10,0x22,0x80,0xac,0x02,0x0a,0x00,0x20,0x00,0x02,0x10}},
      {2,2,0x38,0,true,6,{0x19,0xb9,0x15,0x03,0x80,0x00}},
      {4,4,0x40,0,true,6,{0x29,0xc1,0x15,0x05,0x80,0x00}},
      {6,6,0x02,0,true,6,{0x39,0x83,0x15,0x07,0x80,0x00}},
      {8,0,0x20,128,false,14,
       {0x29,0x00,0x05,0x10,0x22,0x80,0xac,0x12,0x0a,0x00,0x00,0x00,0x02,0x10}},
      {10,2,0x38,0,false,6,{0x59,0xb9,0x05,0x03,0x80,0x00}},
      {12,4,0x40,0,false,6,{0x69,0xc1,0x05,0x05,0x80,0x00}},
      {14,6,0x02,0,false,6,{0x79,0x83,0x05,0x07,0x80,0x00}},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      assert(agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3C,
         vectors[i].dest, vectors[i].src, vectors[i].immediate,
         vectors[i].discard ? AGX3_ALU_HINT_DISCARD : AGX3_ALU_HINT_NONE,
         vectors[i].wait, bytes, &len));
      assert(len == vectors[i].length && !memcmp(bytes, vectors[i].bytes, len));
   }
   assert(!agx3_alu_fmul_minifloat(AGX_ISA_ENCODING_AGX3C, 0, 0, 32,
      AGX3_ALU_HINT_DISCARD, 256, bytes, &len));
}



/*
 * Source modifiers on the reciprocal-square-root family.
 *
 * `inversesqrt(abs(x))` and `sqrt(abs(x))` -- the standard safe forms of both
 * idioms -- had no encoding at all: f30 was simply not modelled on these two
 * forms, though f32log2/f32exp2/f32rcp have carried it for a while.
 *
 * Apple's own bytes, `metal-nt -arch applegpu_g17p`, one kernel per modifier,
 * each a single-field difference from the plain control. Every vector was
 * recovered by re-composing through this encoder, so these are facts about
 * the encoder rather than readings of a disassembler.
 */
static void
test_native_f32_rsqrt_modifiers(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   /* o[i] = rsqrt(a[i]) -- the control the three below differ from. */
   static const uint8_t plain[] = {0xaf, 0x01, 0x56, 0x00, 0x02,
                                   0x00, 0xb0, 0x40, 0x00, 0x00};
   assert(agx3_alu_f32rsqrt_mod(AGX_ISA_ENCODING_AGX3B, 0, 0,
                                AGX3_ALU_HINT_DISCARD, false, 1u << 5, false,
                                false, bytes, &len));
   assert(len == sizeof(plain) && !memcmp(bytes, plain, len));

   /* o[i] = rsqrt(-a[i])  -- f31 alone. */
   static const uint8_t neg[] = {0xaf, 0x01, 0x56, 0x00, 0x02,
                                 0x00, 0xb0, 0x40, 0x01, 0x00};
   assert(agx3_alu_f32rsqrt_mod(AGX_ISA_ENCODING_AGX3B, 0, 0,
                                AGX3_ALU_HINT_DISCARD, false, 1u << 5, true,
                                false, bytes, &len));
   assert(len == sizeof(neg) && !memcmp(bytes, neg, len));

   /* o[i] = rsqrt(fabs(a[i])) -- f30 alone. */
   static const uint8_t abs_[] = {0xaf, 0x01, 0x56, 0x00, 0x02,
                                  0x00, 0xb0, 0xc0, 0x00, 0x00};
   assert(agx3_alu_f32rsqrt_mod(AGX_ISA_ENCODING_AGX3B, 0, 0,
                                AGX3_ALU_HINT_DISCARD, false, 1u << 5, false,
                                true, bytes, &len));
   assert(len == sizeof(abs_) && !memcmp(bytes, abs_, len));

   /* o[i] = rsqrt(-fabs(a[i])) -- both, which is what makes them separable. */
   static const uint8_t both[] = {0xaf, 0x01, 0x56, 0x00, 0x02,
                                  0x00, 0xb0, 0xc0, 0x01, 0x00};
   assert(agx3_alu_f32rsqrt_mod(AGX_ISA_ENCODING_AGX3B, 0, 0,
                                AGX3_ALU_HINT_DISCARD, false, 1u << 5, true,
                                true, bytes, &len));
   assert(len == sizeof(both) && !memcmp(bytes, both, len));

   /* The unmodified wrapper must keep emitting the control exactly. */
   assert(agx3_alu_f32rsqrt(AGX_ISA_ENCODING_AGX3B, 0, 0,
                            AGX3_ALU_HINT_DISCARD, false, 1u << 5, bytes,
                            &len));
   assert(len == sizeof(plain) && !memcmp(bytes, plain, len));
}

/*
 * The square-root step carries the same absolute-value bit. Apple compiles
 * sqrt() to this step followed by a multiply with the original input, so
 * `sqrt(abs(x))` sets f30 here and nowhere else.
 *
 *   o[i] = sqrt(a[i])        2f 01 56 04 03 00 92 40 00 00
 *   o[i] = sqrt(fabs(a[i]))  2f 01 56 04 03 00 92 c0 00 00
 */
static void
test_native_sqrt_step_absolute(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   static const uint8_t plain[] = {0x2f, 0x01, 0x56, 0x04, 0x03,
                                   0x00, 0x92, 0x40, 0x00, 0x00};
   assert(agx3_alu_sqrt_step_mod(AGX_ISA_ENCODING_AGX3B, 4, 0,
                                 AGX3_ALU_HINT_CACHE, true, 1u << 5, false,
                                 bytes, &len));
   assert(len == sizeof(plain) && !memcmp(bytes, plain, len));

   static const uint8_t abs_[] = {0x2f, 0x01, 0x56, 0x04, 0x03,
                                  0x00, 0x92, 0xc0, 0x00, 0x00};
   assert(agx3_alu_sqrt_step_mod(AGX_ISA_ENCODING_AGX3B, 4, 0,
                                 AGX3_ALU_HINT_CACHE, true, 1u << 5, true,
                                 bytes, &len));
   assert(len == sizeof(abs_) && !memcmp(bytes, abs_, len));

   /* The plain wrapper is unchanged. */
   assert(agx3_alu_sqrt_step(AGX_ISA_ENCODING_AGX3B, 4, 0,
                             AGX3_ALU_HINT_CACHE, true, 1u << 5, bytes, &len));
   assert(len == sizeof(plain) && !memcmp(bytes, plain, len));
}

/*
 * The multiply's destination width (f10) is selected independently of its
 * source width (f26), so a 32-bit value scaled by a minifloat lands directly
 * in a half register. That is the instruction `packHalf2x16(vec2(x * k, ...))`
 * produces once the f2f16 folds into the multiply, and it had no encoding.
 *
 *   o[i] = a[i] * 0.5f        (float o)  19 a1 15 03 80 c0   f10 = 2
 *   o[i] = half(a[i] * 0.5f)  (half  o)  11 a1 15 03 80 c0   f10 = 1
 *   o[i] = half(a[i] * 2.0f)             11 c1 15 03 80 c0   f17 = 64
 */
static void
test_native_sqrt_step_negation(void)
{
   /* Native G17P metal32023.850 probes: plain / neg / absneg. */
   static const uint8_t apple[3][10] = {
      {0x2f,0x01,0x56,0x04,0x03,0x00,0x92,0x40,0x00,0x00},
      {0x2f,0x01,0x56,0x04,0x03,0x00,0x92,0x40,0x01,0x00},
      {0x2f,0x01,0x56,0x04,0x03,0x00,0x92,0xc0,0x01,0x00},
   };
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   for (unsigned i = 0; i < 3; ++i) {
      assert(agx3_alu_sqrt_step_source_modifiers(
         AGX_ISA_ENCODING_AGX3B, 4, 0, AGX3_ALU_HINT_CACHE, true, 1u << 5,
         i == 2, i != 0, bytes, &len));
      assert(len == 10 && memcmp(bytes, apple[i], 10) == 0);
   }
   assert(!agx3_alu_sqrt_step_source_modifiers(
      AGX_ISA_ENCODING_AGX3B, 3, 0, AGX3_ALU_HINT_CACHE, true, 0,
      false, true, bytes, &len));
   assert(!agx3_alu_sqrt_step_source_modifiers(
      AGX_ISA_ENCODING_AGX3B, 4, 0, AGX3_ALU_HINT_CACHE, true, 0x40,
      false, true, bytes, &len));
}

static void
test_native_fmul_minifloat_narrow(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;

   /* 32-bit destination: the control, and byte-identical to what the
    * unmodified entry point already emitted. */
   static const uint8_t wide[] = {0x19, 0xa1, 0x15, 0x03, 0x80, 0xc0};
   assert(agx3_alu_fmul_minifloat_narrow(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, false, false, false,
      AGX3_ALU_HINT_DISCARD, 1u << 5, bytes, &len));
   assert(len == sizeof(wide) && !memcmp(bytes, wide, len));
   assert(agx3_alu_fmul_minifloat_mod(AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20,
                                      false, false, AGX3_ALU_HINT_DISCARD,
                                      1u << 5, bytes, &len));
   assert(len == sizeof(wide) && !memcmp(bytes, wide, len));

   /* Half destination: one field different. */
   static const uint8_t narrow[] = {0x11, 0xa1, 0x15, 0x03, 0x80, 0xc0};
   assert(agx3_alu_fmul_minifloat_narrow(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, false, true, false,
      AGX3_ALU_HINT_DISCARD, 1u << 5, bytes, &len));
   assert(len == sizeof(narrow) && !memcmp(bytes, narrow, len));

   /* The immediate still sweeps as the minifloat operand. */
   static const uint8_t narrow2[] = {0x11, 0xc1, 0x15, 0x03, 0x80, 0xc0};
   assert(agx3_alu_fmul_minifloat_narrow(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 0x40, false, false, true, false,
      AGX3_ALU_HINT_DISCARD, 1u << 5, bytes, &len));
   assert(len == sizeof(narrow2) && !memcmp(bytes, narrow2, len));

   /* A half destination may be an odd register half; a 32-bit one may not. */
   assert(agx3_alu_fmul_minifloat_narrow(
      AGX_ISA_ENCODING_AGX3B, 3, 2, 0x20, false, false, true, false,
      AGX3_ALU_HINT_DISCARD, 1u << 5, bytes, &len));
   assert(!agx3_alu_fmul_minifloat_narrow(
      AGX_ISA_ENCODING_AGX3B, 3, 2, 0x20, false, false, false, false,
      AGX3_ALU_HINT_DISCARD, 1u << 5, bytes, &len));

   /* Neither destination width has a grounded fused ABS form. */
   assert(!agx3_alu_fmul_minifloat_narrow(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, true, true, false,
      AGX3_ALU_HINT_DISCARD, 1u << 5, bytes, &len));
   assert(!agx3_alu_fmul_minifloat_narrow(
      AGX_ISA_ENCODING_AGX3B, 2, 2, 0x20, false, true, false, false,
      AGX3_ALU_HINT_DISCARD, 1u << 5, bytes, &len));

   /* Apple applegpu_g17g: half4(float4_input * 0.5f). The first operation
    * waits for its load; later channels narrow directly to even/odd halves. */
   static const struct {
      uint16_t dest, src, wait;
      uint8_t size, bytes[14];
   } gpu1[] = {
      {0,0,128,14,{0x29,0x00,0x05,0x18,0x22,0x80,0xac,0x02,0x0a,0x00,0x20,0x00,0x02,0x10}},
      {1,2,0,6,{0x01,0xa1,0x15,0x03,0x84,0x00}},
      {2,4,0,6,{0x11,0xa1,0x15,0x05,0x80,0x00}},
      {3,6,0,6,{0x11,0xa1,0x15,0x07,0x84,0x00}},
   };
   for (unsigned i = 0; i < sizeof(gpu1) / sizeof(gpu1[0]); ++i) {
      assert(agx3_alu_fmul_minifloat_narrow(
         AGX_ISA_ENCODING_AGX3C, gpu1[i].dest, gpu1[i].src, 0x20,
         false, false, true, false, AGX3_ALU_HINT_DISCARD, gpu1[i].wait, bytes, &len));
      assert(len == gpu1[i].size && !memcmp(bytes, gpu1[i].bytes, len));
   }
}


/*
 * The subtract's slot-B shift (f54).
 *
 * `x * -7` is strength-reduced to `x - (x << 3)`, so a shifted integer
 * subtract is an ordinary shape rather than an exotic one, and every such
 * multiply failed the whole shader closed while it had no encoding. It is
 * NOT a degenerate self-subtraction that should fold to zero: with the shift
 * it computes -7x.
 *
 * Apple, one kernel per shift, `o[i] = a[i] - (c[i] << N)`.
 */
static void
test_native_isub_shift(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   const uint16_t src[2] = {0, 4};
   const enum agx3_alu_hint hints[2] = {AGX3_ALU_HINT_DISCARD,
                                        AGX3_ALU_HINT_DISCARD};

   static const uint8_t sh[5][10] = {
      {0x1f, 0x01, 0x56, 0x00, 0x02, 0x00, 0x10, 0xa8, 0x17, 0x05},
      {0x1f, 0x01, 0x56, 0x00, 0x02, 0x00, 0x10, 0xe8, 0x16, 0x05},
      {0x1f, 0x01, 0x56, 0x00, 0x02, 0x00, 0x10, 0xa8, 0x16, 0x15},
      {0x1f, 0x01, 0x56, 0x00, 0x02, 0x00, 0x10, 0xe8, 0x16, 0x15},
      {0x1f, 0x01, 0x56, 0x00, 0x02, 0x00, 0x10, 0xa8, 0x16, 0x05},
   };
   for (uint8_t n = 0; n <= 4; ++n) {
      assert(agx3_alu_integer_reg_shift(AGX_ISA_ENCODING_AGX3B,
                                        AGX3_ALU_ISUB, 0, src, hints, n,
                                        1u << 5, bytes, &len));
      assert(len == 10 && !memcmp(bytes, sh[n], 10));
   }

   /* The unshifted wrapper keeps emitting exactly the N = 0 bytes. */
   assert(agx3_alu_integer_reg(AGX_ISA_ENCODING_AGX3B, AGX3_ALU_ISUB, 0, src,
                               hints, 1u << 5, bytes, &len));
   assert(len == 10 && !memcmp(bytes, sh[0], 10));

   /* Apple stops emitting this form at N = 5, so 5 is refused rather than
    * extrapolated, and the ADD's shifted form belongs to agx3_iadd_encode. */
   assert(!agx3_alu_integer_reg_shift(AGX_ISA_ENCODING_AGX3B, AGX3_ALU_ISUB,
                                      0, src, hints, 5, 1u << 5, bytes,
                                      &len));
   assert(!agx3_alu_integer_reg_shift(AGX_ISA_ENCODING_AGX3B, AGX3_ALU_IADD,
                                      0, src, hints, 1, 1u << 5, bytes,
                                      &len));
}


/*
 * G17G's reg/reg integer add and bitops carry their own lifetimes.
 *
 * Reaching these through the generic select template instead pinned the
 * specimen's first-operand discard and its wait on slot 8 as if both were
 * form constants, so a register read a second time after the add read dead.
 * That template is `agx3_pack_iadd_u32_reg_reg`'s Apple G17G/G17S replay,
 * and this encoder reproduces it byte for byte when it is given that
 * specimen's own lifetimes and wait -- which is what says the only bits that
 * moved are the ones describing them. The corpus
 * (`agx3_compose_corpus_vectors.c`) carries Apple iadd32 and bitop
 * instructions with each combination of f27/f42 set and clear.
 */
static void
test_gpu1_integer_reg_lifetimes(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   const uint16_t src[2] = {0, 4};

   struct agx3_encoded_iadd_u32_reg_reg canonical;
   assert(agx3_pack_iadd_u32_reg_reg(AGX_ISA_ENCODING_AGX3C, 4, 0, 4,
                                     &canonical));

   const enum agx3_alu_hint specimen[2] = {AGX3_ALU_HINT_DISCARD,
                                           AGX3_ALU_HINT_NONE};
   assert(agx3_alu_integer_reg(AGX_ISA_ENCODING_AGX3C, AGX3_ALU_IADD, 4, src,
                               specimen, 1u << 7, bytes, &len));
   assert(len == canonical.length &&
          !memcmp(bytes, canonical.bytes, canonical.length));

   /* Every other combination differs from it, and from each other. */
   uint8_t seen[9][AGX3_ALU_MAX_LENGTH];
   unsigned n = 0;
   for (unsigned h0 = 0; h0 <= AGX3_ALU_HINT_CACHE; ++h0) {
      for (unsigned h1 = 0; h1 <= AGX3_ALU_HINT_CACHE; ++h1) {
         const enum agx3_alu_hint hints[2] = {(enum agx3_alu_hint)h0,
                                              (enum agx3_alu_hint)h1};
         assert(agx3_alu_integer_reg(AGX_ISA_ENCODING_AGX3C, AGX3_ALU_IADD, 4,
                                     src, hints, 0, bytes, &len));
         assert(len == canonical.length);
         /* No combination reproduces the pinned wait. */
         assert(memcmp(bytes, canonical.bytes, len) != 0);
         for (unsigned i = 0; i < n; ++i)
            assert(memcmp(bytes, seen[i], len) != 0);
         memcpy(seen[n++], bytes, len);
      }
   }

   for (unsigned h0 = 0; h0 <= AGX3_ALU_HINT_CACHE; ++h0) {
      const enum agx3_alu_hint hints[2] = {(enum agx3_alu_hint)h0,
                                           AGX3_ALU_HINT_NONE};
      uint8_t band[AGX3_ALU_MAX_LENGTH];
      uint8_t band_len = 0;
      assert(agx3_alu_bit_reg(AGX_ISA_ENCODING_AGX3C, AGX3_ALU_BAND, 4, src,
                              hints, 0, band, &band_len));
      if (h0 > 0)
         assert(memcmp(band, seen[0], band_len) != 0);
   }

   /* G17G's shifted subtract is unwitnessed. */
   const enum agx3_alu_hint dd[2] = {AGX3_ALU_HINT_DISCARD,
                                     AGX3_ALU_HINT_DISCARD};
   assert(!agx3_alu_integer_reg_shift(AGX_ISA_ENCODING_AGX3C, AGX3_ALU_ISUB,
                                      0, src, dd, 1, 0, bytes, &len));
}

/*
 * The compare-select's absolute-value sources.
 *
 * ABS is not a modifier bit on the ten-byte fsel: Apple moves to a
 * fourteen-byte endpoint (ep20..27 -> ep28..35) carrying f47 and f64, the two
 * sources' bits, separable because fabs(a) < fabs(b) sets both. FCMPSEL's
 * selected immediates are integers, which that endpoint takes in the same
 * f3/f11/f12 destination block as the short form. Every vector below is
 * Apple's own, one kernel per field, `x + (cmp ? 3u : 1u)` unless noted.
 */
static void
fsel_abs_is(uint16_t dest, bool dest_16, uint16_t src0, uint16_t src1,
            bool src0_16, bool src1_16, bool dest_cache, unsigned condition,
            uint8_t if_true, uint8_t if_false, const bool src_abs[2],
            const uint8_t *expect)
{
   const uint16_t src[2] = {src0, src1};
   const bool width[2] = {src0_16, src1_16};
   const enum agx3_alu_hint dd[2] = {AGX3_ALU_HINT_DISCARD,
                                     AGX3_ALU_HINT_DISCARD};
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   assert(agx3_alu_fcmp_select_imm_width_abs(
      AGX_ISA_ENCODING_AGX3B, dest, src, width, dd, condition, if_true,
      if_false, dest_16, dest_cache, 32, src_abs, bytes, &len));
   assert(len == 14 && !memcmp(bytes, expect, 14));
}

static void
test_native_fcmp_select_absolute(void)
{
   const bool abs0[2] = {true, false};
   const bool abs1[2] = {false, true};
   const bool abs01[2] = {true, true};

   /* (fabs(a) < b) ? 3u : 1u over float32 -- the baseline. */
   static const uint8_t base[]  = {0x22,0x05,0x3f,0x07,0x23,0x83,0x03,
                                   0x00,0x20,0x81,0x00,0x80,0x01,0x00};
   /* condition alone */
   static const uint8_t c_gt[]  = {0x22,0x05,0x3f,0x07,0x23,0x83,0x02,
                                   0x00,0x20,0x81,0x00,0x80,0x01,0x00};
   static const uint8_t c_eq[]  = {0x22,0x05,0x3f,0x07,0x27,0x83,0x00,
                                   0x00,0x20,0x81,0x00,0x80,0x01,0x00};
   static const uint8_t c_ge[]  = {0x22,0x05,0x3f,0x07,0x27,0x83,0x02,
                                   0x00,0x20,0x81,0x00,0x80,0x01,0x00};
   static const uint8_t c_le[]  = {0x22,0x05,0x3f,0x07,0x27,0x83,0x03,
                                   0x00,0x20,0x81,0x00,0x80,0x01,0x00};
   /* both selects: ? 0u : 7u */
   static const uint8_t s_07[]  = {0x22,0x05,0x3f,0x07,0x23,0x80,0x03,
                                   0x00,0x20,0x87,0x00,0x80,0x01,0x00};
   /* which source carries ABS */
   static const uint8_t a_1[]   = {0x22,0x05,0x3f,0x07,0x23,0x83,0x03,
                                   0x00,0x20,0x81,0x00,0x80,0x02,0x00};
   static const uint8_t a_01[]  = {0x22,0x05,0x3f,0x07,0x23,0x83,0x03,
                                   0x00,0x20,0x81,0x00,0x80,0x03,0x00};
   /* 16-bit result: dest r1l, comparisons r1 and r2; and ? 1u : 0u */
   static const uint8_t d16[]   = {0x12,0x03,0x3d,0x05,0x23,0x83,0x03,
                                   0x00,0x20,0x81,0x00,0x80,0x01,0x00};
   static const uint8_t d16_10[] = {0x22,0x05,0x3d,0x07,0x23,0x81,0x03,
                                    0x00,0x20,0x80,0x00,0x80,0x01,0x00};
   /* half comparisons: half/half with a 16-bit result on half 3, half/half
    * fabs(b) with a 32-bit result, float/half, and the plain uint result of
    * (fabs(a) < b) ? 1u : 0u, which Apple narrows to a cache-less half. */
   static const uint8_t h_d16[] = {0x1a,0x02,0x3d,0x04,0x23,0x83,0x0b,
                                   0x00,0x20,0x81,0x00,0x80,0x01,0x00};
   static const uint8_t h_a_1[] = {0x22,0x04,0x3f,0x04,0x23,0x83,0x13,
                                   0x00,0x20,0x81,0x00,0x80,0x02,0x00};
   static const uint8_t fh[]    = {0x22,0x05,0x3f,0x06,0x23,0x83,0x03,
                                   0x00,0x20,0x81,0x00,0x80,0x01,0x00};
   static const uint8_t h_10[]  = {0x12,0x02,0x1d,0x02,0x23,0x81,0x13,
                                   0x00,0x20,0x80,0x00,0x80,0x01,0x00};

   fsel_abs_is(4, false, 4, 6, false, false, true, 1, 3, 1, abs0, base);
   fsel_abs_is(4, false, 4, 6, false, false, true, 2, 3, 1, abs0, c_gt);
   fsel_abs_is(4, false, 4, 6, false, false, true, 0, 3, 1, abs0, c_eq);
   fsel_abs_is(4, false, 4, 6, false, false, true, 5, 3, 1, abs0, c_ge);
   fsel_abs_is(4, false, 4, 6, false, false, true, 6, 3, 1, abs0, c_le);
   fsel_abs_is(4, false, 4, 6, false, false, true, 1, 0, 7, abs0, s_07);
   fsel_abs_is(4, false, 4, 6, false, false, true, 1, 3, 1, abs1, a_1);
   fsel_abs_is(4, false, 4, 6, false, false, true, 1, 3, 1, abs01, a_01);
   fsel_abs_is(2, true, 2, 4, false, false, true, 1, 3, 1, abs0, d16);
   fsel_abs_is(4, true, 4, 6, false, false, true, 1, 1, 0, abs0, d16_10);
   fsel_abs_is(3, true, 3, 4, true, true, true, 1, 3, 1, abs0, h_d16);
   fsel_abs_is(4, false, 4, 5, true, true, true, 1, 3, 1, abs1, h_a_1);
   fsel_abs_is(4, false, 4, 6, false, true, true, 1, 3, 1, abs0, fh);
   fsel_abs_is(2, true, 2, 3, true, true, false, 1, 1, 0, abs0, h_10);

   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
   const uint16_t src[2] = {2, 4};
   const bool w32[2] = {false, false};
   const enum agx3_alu_hint dd[2] = {AGX3_ALU_HINT_DISCARD,
                                     AGX3_ALU_HINT_DISCARD};

   /* Without an absolute source this is the SHORT endpoint, and emitting the
    * long form would change bytes that already work: refuse instead. */
   const bool none[2] = {false, false};
   assert(!agx3_alu_fcmp_select_imm_width_abs(
      AGX_ISA_ENCODING_AGX3B, 2, src, w32, dd, 1, 48, 0, true, false, 32,
      none, bytes, &len));

   /* The short endpoint is untouched: it still encodes, and still at ten
    * bytes rather than being pulled onto the long form. Its exact bytes are
    * owned by the tests that already cover it -- this only pins that adding
    * the absolute-value endpoint did not divert it. */
   len = 0;
   assert(agx3_alu_fcmp_select_imm_width(AGX_ISA_ENCODING_AGX3B, 2, src, w32,
                                         dd, 1, 48, 0, true, false, 32,
                                         bytes, &len));
   assert(len == 10);

   /* AGX3c has no witness for any of this. */
   len = 0;
   assert(!agx3_alu_fcmp_select_imm_width_abs(
      AGX_ISA_ENCODING_AGX3C, 2, src, w32, dd, 1, 48, 0, true, false, 32,
      abs0, bytes, &len));
}

/*
 * The three axes a three.js frame was refused on: an operand's WIDTH, an
 * operand's KIND and a source MODIFIER are ordinary fields of one encoding
 * point and compose freely. Every vector below is Apple's own, from
 * metal-nt -arch applegpu_g17p, recovered through tools/decomp.c.
 */
static void test_threejs_width_kind_modifier_axes(void)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH];
   uint8_t len = 0;

   /* f32add over a float/half pair. `float a` in one slot, `half b` in the
    * other; f43 and f26 are the two slots' widths, f47/f48 and f30/f31 their
    * modifiers, and f14 saturates whatever the widths are. */
   {
      static const struct {
         const char *why;
         bool s0_16, s1_16, s0_abs, s0_neg, s1_abs, s1_neg, sat;
         uint16_t s0, s1;
         const char *want;
      } rows[] = {
         {"a - b",        true,  false, false, true,  false, false, false,
          4, 0, "09011c0400c8"},
         {"b - a",        false, true,  false, true,  false, false, false,
          0, 4, "09041c0100c8"},
         {"a + fabs(b)",  true,  false, true,  false, false, false, false,
          4, 0, "09011c04020000800200"},
         {"fabs(a) + b",  true,  false, false, false, true,  false, false,
          4, 0, "09011c04020000800100"},
         {"b - fabs(a)",  false, true,  true,  true,  false, false, false,
          0, 4, "09041c01020800800200"},
         {"-a - b",       false, true,  false, true,  false, true,  false,
          0, 4, "09041c0101080280"},
         {"saturate(a+b)", true, false, false, false, false, false, true,
          4, 0, "09011c0401000082"},
         {"saturate(b-a)", false, true, false, true,  false, false, true,
          0, 4, "09041c0101080082"},
      };
      for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
         struct agx3_falu_mods m = {0};
         struct agx3_falu_runtime rt = {0};
         m.src0_16 = rows[i].s0_16;
         m.src1_16 = rows[i].s1_16;
         m.src0_abs = rows[i].s0_abs;
         m.src0_neg = rows[i].s0_neg;
         m.src1_abs = rows[i].s1_abs;
         m.src1_neg = rows[i].s1_neg;
         m.saturate = rows[i].sat;
         rt.discard[0] = rt.discard[1] = true;
         rt.wait_mask = 0x20;
         assert(agx3_falu_add_mod_runtime(enc, 0, rows[i].s0, rows[i].s1, &m,
                                          &rt, bytes, &len));
         exact(bytes, len, rows[i].want);
      }
   }

   /* The same pair on f32mul, and the ffma addend's own width field f60. */
   {
      struct agx3_falu_mods m = {0};
      struct agx3_falu_runtime rt = {0};
      m.src0_neg = true;
      m.src1_16 = true;
      rt.discard[0] = rt.discard[1] = true;
      rt.wait_mask = 0x20;
      assert(agx3_falu_mul_mod_runtime(enc, 0, 0, 4, &m, &rt, bytes, &len));
      exact(bytes, len, "09041d0100c8");

      struct agx3_falu_runtime frt = {0};
      frt.discard[0] = frt.discard[1] = frt.discard[2] = true;
      frt.wait_mask = 0x20;

      struct agx3_falu_mods f = {0};
      f.addend_16 = true;
      assert(agx3_falu_fma_mod_runtime(enc, 0, 4, 0, 8, &f, &frt, bytes,
                                       &len));
      exact(bytes, len, "09011e05810800c0");
      f.addend_neg = true;
      assert(agx3_falu_fma_mod_runtime(enc, 0, 4, 0, 8, &f, &frt, bytes,
                                       &len));
      exact(bytes, len, "09011e05910800c0");

      struct agx3_falu_mods g = {0};
      g.src1_16 = true;
      assert(agx3_falu_fma_mod_runtime(enc, 0, 0, 8, 4, &g, &frt, bytes,
                                       &len));
      exact(bytes, len, "09081e01810402c0");
   }

   /* fsel: an absolute comparator against a minifloat IMMEDIATE. Against the
    * register row it changes only f59, the slot's kind, and the operand
    * field f52 -> f51. Refusing every non-zero immediate recorded what had
    * been measured, not what the field can hold.
    *
    * The bytes below are the ep33 placement of the destination block, which
    * is the one Apple's own corpus uses (four vectors in
    * test/agx3_compose_corpus_vectors.c) and the one this encoder reaches
    * first. Apple's fresh kernels selecting 1.0h take the ep28 placement
    * instead -- 1203190523b00300208000800100 and
    * 120309a123b00302208000800100 -- which reads the selects as minifloats;
    * this ep33 placement reads the same select fields as integers. */
   {
      struct agx3_fsel_desc d;
      memset(&d, 0, sizeof(d));
      d.dest = 2;
      d.dest_16 = true;
      d.cmp0 = 2;
      d.cmp0_hint = AGX3_ALU_HINT_DISCARD;
      d.cmp0_abs = true;
      d.sel0_kind = AGX3_FSEL_SEL_IMMEDIATE;
      d.sel0 = 48;
      d.sel1_kind = AGX3_FSEL_SEL_IMMEDIATE;
      d.sel1 = 0;
      d.sel_16 = true;
      d.condition = 1;
      d.wait_mask = 32;

      struct agx3_fsel_desc reg = d;
      reg.cmp1 = 4;
      reg.cmp1_hint = AGX3_ALU_HINT_DISCARD;
      assert(agx3_alu_fcmp_select(enc, &reg, bytes, &len));
      exact(bytes, len, "12031d0523b00300208000800100");

      struct agx3_fsel_desc imm = d;
      imm.cmp1_is_immediate = true;
      imm.cmp1 = 32;
      assert(agx3_alu_fcmp_select(enc, &imm, bytes, &len));
      exact(bytes, len, "12030da123b00302208000800100");

      /* A minifloat carries its sign in bit 7, so `-a < k` is `a > -k`:
       * constant folding, not a second encoding. */
      struct agx3_fsel_desc neg = imm;
      neg.cmp0_neg = true;
      assert(agx3_alu_fcmp_select(enc, &neg, bytes, &len));
      exact(bytes, len, "12031da123b00202208000800100");
   }

   /* fsel: the negate algebra on the register-select form. Both comparators
    * negated is the relation flip alone, which means the two descriptors must
    * produce the same bytes. */
   {
      uint16_t src[4] = {8, 16, 6, 0};
      const bool abs[2] = {false, false};
      const bool imm[2] = {false, true};
      const enum agx3_alu_hint hints[4] = {
         AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE,
         AGX3_ALU_HINT_NONE};
      const bool both[2] = {true, true};
      const bool none[2] = {false, false};
      const bool second[2] = {false, true};
      uint8_t flipped[AGX3_ALU_MAX_LENGTH];
      uint8_t flipped_len = 0;

      assert(agx3_alu_fcmp_select_reg_imm(enc, 2, src, false, abs, both, imm,
                                          hints, 5, true, false, 0, bytes,
                                          &len));
      assert(agx3_alu_fcmp_select_reg_imm(enc, 2, src, false, abs, none, imm,
                                          hints, 6, true, false, 0, flipped,
                                          &flipped_len));
      assert(len == flipped_len && memcmp(bytes, flipped, len) == 0);

      /* One negate is f65, with the operands and the relation left alone:
       * exactly one byte differs from the plain select. */
      uint8_t plain[AGX3_ALU_MAX_LENGTH];
      uint8_t plain_len = 0;
      assert(agx3_alu_fcmp_select_reg_imm(enc, 2, src, false, abs, second, imm,
                                          hints, 5, true, false, 0, bytes,
                                          &len));
      assert(agx3_alu_fcmp_select_reg_imm(enc, 2, src, false, abs, none, imm,
                                          hints, 5, true, false, 0, plain,
                                          &plain_len));
      assert(len == plain_len);
      unsigned differing = 0;
      for (unsigned i = 0; i < len; ++i)
         differing += bytes[i] != plain[i];
      assert(differing == 1);
   }

   /* bitop against an immediate: the operand kind is f37 on the register
    * form's own encoding point, with the constant in f29. A narrow result
    * narrows f10 and f24 and NOT f38, because an immediate operand declares
    * width 2 whatever the destination is. */
   {
      static const struct {
         enum agx3_alu_op op;
         bool narrow;
         uint16_t dest, src;
         uint8_t imm;
         const char *want;
      } rows[] = {
         {AGX3_ALU_BAND, false, 0, 0, 0xff, "0b011fff100200800000"},
         {AGX3_ALU_BOR,  false, 0, 0, 0xff, "0b011fff120a00800000"},
         {AGX3_ALU_BXOR, false, 0, 0, 0xff, "0b011eff120a00800000"},
         {AGX3_ALU_BAND, false, 0, 0, 0x7f, "0b010fff100200800000"},
         {AGX3_ALU_BXOR, false, 0, 0, 0x01, "0b010e81120a00800000"},
         {AGX3_ALU_BAND, true,  2, 2, 0xff, "13021fff100200800000"},
         {AGX3_ALU_BOR,  true,  2, 2, 0x80, "13021f81020a00800000"},
      };
      for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
         assert(agx3_alu_bit_imm(enc, rows[i].op, rows[i].narrow, rows[i].dest,
                                 rows[i].src, rows[i].imm,
                                 AGX3_ALU_HINT_DISCARD, 0x20, bytes, &len));
         exact(bytes, len, rows[i].want);
      }
      /* A truth table with no witnessed immediate form still fails closed. */
      assert(!agx3_alu_bit_imm(enc, AGX3_ALU_BNOT, false, 0, 0, 0xff,
                               AGX3_ALU_HINT_NONE, 0, bytes, &len));
   }

   puts("three.js width/kind/modifier axes: 23 exact G17P vectors PASS");
}


static void test_native_fmax_abs(void)
{
   uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
   uint16_t src[] = {10, 0};
   enum agx3_alu_hint hints[] = {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
   assert(agx3_alu_fmax_abs_f32(AGX_ISA_ENCODING_AGX3B, 0, src, hints,
                               true, bytes, &len));
   exact(bytes, len, "020b3b01830a0000880000000200");
   assert(!agx3_alu_fmax_abs_f32(AGX_ISA_ENCODING_AGX3B, 1, src, hints,
                                true, bytes, &len));
   src[1] = 1;
   assert(!agx3_alu_fmax_abs_f32(AGX_ISA_ENCODING_AGX3B, 0, src, hints,
                                true, bytes, &len));
   src[1] = 256;
   assert(!agx3_alu_fmax_abs_f32(AGX_ISA_ENCODING_AGX3B, 0, src, hints,
                                true, bytes, &len));
   src[1] = 0;
   hints[0] = (enum agx3_alu_hint)99;
   assert(!agx3_alu_fmax_abs_f32(AGX_ISA_ENCODING_AGX3B, 0, src, hints,
                                true, bytes, &len));
}

int main(void){
   test_native_fmax_abs();
   test_gpu1_minifloat_scale();
   test_gpu1_popcount();
   test_gpu1_modified_add();
   test_gpu1_float_compare_select();
   test_native_immediate_shifts32();
   test_native_popcount();
   test_native_minifloat_half_select();
   test_native_bit_not16();
   test_native_bit_andc();
   test_band_identity_wait();
   test_form_lifetime_baselines();
   test_native_bit_nand_xnor_orc();
   test_native_bit_nor();
   test_native_ft1();
   test_native_f32_rcp_rsqrt();
   test_native_f32_rsqrt_modifiers();
   test_native_sqrt_step_absolute();
   test_native_sqrt_step_negation();
   test_native_fmul_minifloat_narrow();
   test_native_isub_shift();
   test_gpu1_integer_reg_lifetimes();
   test_native_fcmp_select_absolute();
   test_native_f32_log2_exp2();
   test_native_f32_log2_exp2_modifiers();
   test_native_f32_floor_ceil();
   test_native_f32_roundeven();
   test_native_high_float_registers();
   test_native_widen_half();
   test_native_half_add();
   test_native_half_saturate();
   test_native_half_inverse();
   test_native_half_multiply();
   test_native_half_multiply_minifloat();
   test_native_half_saturating_arithmetic();
   test_wait_code_forms_fail_closed();
   test_native_minifloat_mixed_select();
   test_native_half_minifloat_select_f32();
   test_native_fadd_mixed_half();
   test_native_fadd_abs_mixed_half();
   test_native_half_fma();
   test_m5_native_half_arithmetic();
   test_m5_native_half_minmax();
   test_m5_native_half_immediates();
   test_m5_native_half_negation();
   test_native_half_fma_minifloat_addend();
   test_native_narrow_multiply();
   test_native_sqrt_step();
   test_native_half_sqrt();
   test_native_select_reg_imm();
   test_native_half_compare_select();
   test_native_half_min();
   test_native_half_max();
   test_native_f32_minmax();
   test_native_f32_minifloat_select();
   test_native_half_clamp();
   test_native_reciprocal_widths();
   test_native_half_rsqrt();
   test_native_half_exp2();
   test_native_half_log2();
   test_native_mixed_width_log2();
   test_native_half_floor();
   test_native_half_ceil();
   test_native_asr32();
   test_native_asr16();
   test_native_extract32();
   test_blur_half_arithmetic();
   test_native_u16_arithmetic();
   test_native_sub16();
   test_native_add_sat16();
   test_native_narrow_int_to_float();
   test_native_select_negated_comparator();
   test_native_float_to_int16();
   test_native_u16_shift_right_immediate();
   test_native_fma_minifloat();
   test_int_to_float_liveness();
   test_float_to_int32_liveness();
   test_i32_to_f16();
   test_f16_to_int32();
   test_float_compare_select();
   test_fcmp_select_half_comparator();
   test_native_fsel_selected_register_kind();
   test_threejs_width_kind_modifier_axes();
   test_float_compare_zero();
   test_integer_lifetimes();
   test_bitwise_lifetimes();
   test_modified_minifloat_add();
   test_half_uv();
   test_native_fmul_abs_mixed();
   test_fmul_minifloat();
   test_fmul_minifloat_modifiers();
   test_fmul_half_minifloat_to_f32();
   test_fadd_minifloat();
   test_u16_to_f32();
   unsigned pass=0,fail=0,unsupported=0;
   for(size_t i=0;i<avn;i++){
      const struct av*v=&avs[i];
      /* fmin/fmax use a tied-operand form this encoder does not model. */
      if(!agx3_alu_wait_supported((enum agx3_alu_op)v->op)){ unsupported++; continue; }
      enum agx_isa_encoding e=v->enc?AGX_ISA_ENCODING_AGX3C:AGX_ISA_ENCODING_AGX3B;
      uint8_t out[AGX3_ALU_MAX_LENGTH]; uint8_t len=0;
      bool ok=agx3_alu_compose(e,(enum agx3_alu_op)v->op,2,4,6,8,0x1234u,0,out,&len);
      if(!ok||len!=v->n||memcmp(out,v->b,v->n)){ if(fail<12) fprintf(stderr,"FAIL op=%d enc=%d\n",v->op,v->enc); fail++; } else pass++;
   }
   /* Wait deposit: same op/operands, one slot waited -> same length, different bytes. */
   for(int e=0;e<2;e++){
      enum agx_isa_encoding E=e?AGX_ISA_ENCODING_AGX3C:AGX_ISA_ENCODING_AGX3B;
      uint8_t a[16],b[16]; uint8_t la,lb;
      if(agx3_alu_compose(E,AGX3_ALU_IADD,2,4,6,0,0,0,a,&la) &&
         agx3_alu_compose(E,AGX3_ALU_IADD,2,4,6,0,0,0x1,b,&lb)){
         pass++;
         if(la!=lb||memcmp(a,b,la)==0){ fprintf(stderr,"FAIL wait deposit enc=%d\n",e); fail++; }
      }

      /* A wait-only prelude uses ldimm on a fresh destination or bit_and on an
       * in-place accumulator. Cover all six G17P slots and the same common
       * subset on AGX3c.
       */
      const unsigned slots=6;
      for(unsigned slot=0;slot<slots;slot++){
         if(!agx3_alu_compose(E,AGX3_ALU_LDIMM,2,0,0,0,0,
                              1u<<slot,b,&lb)){
            fprintf(stderr,"FAIL ldimm wait slot=%u enc=%d\n",slot,e);
            fail++;
         } else pass++;
         if(!agx3_alu_compose(E,AGX3_ALU_FMUL,2,4,6,0,0,
                              1u<<slot,b,&lb)){
            unsupported++;
         } else pass++;
         if(!agx3_alu_compose(E,AGX3_ALU_BAND,2,2,2,0,0,
                              1u<<slot,b,&lb)){
            fprintf(stderr,"FAIL band wait slot=%u enc=%d\n",slot,e);
            fail++;
         } else pass++;
      }
   }
   if(fail){ fprintf(stderr,"agx3_alu_core: %u failed (%u passed, %u unsupported)\n",fail,pass,unsupported); return 1; }
   printf("agx3_alu_core: %u checks passed (%u unsupported forms skipped)\n",pass,unsupported);
   return 0;
}

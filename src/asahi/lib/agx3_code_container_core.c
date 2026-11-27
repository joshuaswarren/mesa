/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_code_container_core.h"

#include <string.h>

bool
agx3_build_g18g_container_lead(void *image, size_t capacity)
{
   if (!image || capacity < AGX3_G18G_CONTAINER_LEAD_SIZE)
      return false;

   /* Native G18 CODE leads: four identical tables, six helpers followed
    * by two terminals. The shader setup selects 0x200 for fragment/tile
    * programs or 0x400 for vertex programs and adds its SR14 offset.
    * These are AGX3D instructions, not the M5 table's AGX3C bytecode. */
   uint8_t *bytes = image;
   memset(bytes, 0, AGX3_G18G_CONTAINER_LEAD_SIZE);
   bytes[0] = 0x40;
   bytes[1] = 0x06;
   for (size_t i = 0x40; i < AGX3_G18G_CONTAINER_LEAD_SIZE; i += 2)
      bytes[i] = 0x08;

   static const uint8_t terminal[] = {0x00, 0xd1, 0x22, 0, 0, 0, 0, 0x74};
   for (unsigned table = 2; table <= 5; ++table) {
      uint8_t *base = bytes + table * 0x100;
      for (unsigned i = 0; i < 6; ++i) {
         uint8_t *record = base + i * 16;
         memset(record, 0, 10);
         record[1] = 0x0e;
         record[2] = (6 - i) << 4;
      }
      memcpy(base + 0x60, terminal, sizeof(terminal));
      memcpy(base + 0x70, terminal, sizeof(terminal));
   }
   return true;
}

bool
agx3_build_m5_container_lead(void *image, size_t capacity, unsigned clusters)
{
   if (!clusters || clusters > (UINT32_MAX - 0x40u) / 0x300u)
      return false;
   const size_t size = (size_t)clusters * 0x300u + 0x40u;
   if (!image || capacity < size)
      return false;

   /* Both native M5 tables contain ten helper records, then two identical
    * terminals. The GPU0 table has only six helpers and different opcodes. */
   static const uint32_t helpers[] = {
      0x0e33005e, 0x0e31405e, 0x0e31005e, 0x0e1b405e, 0x0e1b005e,
      0x0e19405e, 0x0e19005e, 0x0e13405e, 0x0e13005e, 0x0e11405e,
   };
   static const uint8_t terminal[] = {
      0x07,0x00,0x10,0x0f,0x00,0x00,0x27,0x00,
      0x04,0x00,0x20,0x00,0xa5,0x0a,0x06,0x00,
   };
   uint8_t *bytes = image;
   memset(bytes, 0, size);
   bytes[0] = size;
   bytes[1] = size >> 8;
   for (size_t i = 0x40; i < size; i += 2)
      bytes[i] = 0x06;
   /* One helper table per cluster for each of the two stage groups, from
    * the first group's offset (agx3_m5_container_group_offset()). */
   const uint32_t first = agx3_m5_container_group_offset(clusters, false);
   for (unsigned table = 0; table < 2 * clusters; ++table) {
      uint8_t *base = bytes + first + table * 0x100;
      for (unsigned i = 0; i < sizeof(helpers) / sizeof(helpers[0]); ++i) {
         uint8_t *record = base + i * 16;
         memset(record, 0, 10);
         for (unsigned j = 0; j < 4; ++j)
            record[j] = helpers[i] >> (8 * j);
      }
      memcpy(base + 0xa0, terminal, sizeof(terminal));
      memcpy(base + 0xb0, terminal, sizeof(terminal));
   }
   return true;
}

size_t
agx3_code_block_size(size_t main_span)
{
   if (main_span == 0 || main_span > UINT32_MAX - 0xbf)
      return 0;

   return AGX3_CODE_BLOCK_ENTRY_OFFSET + ((main_span + 0x3f) & ~(size_t)0x3f);
}

bool
agx3_build_code_block(void *image, size_t capacity, const void *main,
                      size_t main_size, size_t main_span,
                      const uint8_t *constant_program,
                      bool fragment)
{
   const size_t block_size = agx3_code_block_size(main_span);
   if (!image || !main || !main_size || main_size > main_span || !block_size ||
       capacity < block_size)
      return false;

   uint8_t *bytes = image;
   memset(bytes, 0, block_size);
   for (unsigned i = 0; i < sizeof(uint32_t); ++i)
      bytes[i] = (uint8_t)(block_size >> (8 * i));

   uint8_t *constant = bytes + AGX3_CODE_BLOCK_HEADER_SIZE;
   if (constant_program)
      memcpy(constant, constant_program, AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE);
   else
      agx3_build_container_constant_program(constant, fragment, false);

   memcpy(bytes + AGX3_CODE_BLOCK_ENTRY_OFFSET, main, main_size);
   return true;
}

void
agx3_build_container_constant_program(
   uint8_t image[AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE], bool fragment,
   bool sampled_fragment)
{
   static const uint8_t fragment_stop[] = {
      0x0e, 0x00, 0x00, 0x00,
   };
   static const uint8_t sampled_fragment_program[] = {
      0x03, 0x00, 0x07, 0x00, 0x02, 0x00, 0x00,
      0x00, 0x20, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   static const uint8_t vertex_compute[] = {
      0x03, 0x00, 0x07, 0x00, 0x02, 0x00, 0x00,
      0x00, 0x60, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   const uint8_t *program =
      !fragment ? vertex_compute
                : sampled_fragment ? sampled_fragment_program : fragment_stop;
   const size_t program_size =
      !fragment ? sizeof(vertex_compute)
                : sampled_fragment ? sizeof(sampled_fragment_program)
                                   : sizeof(fragment_stop);

   /* Pad in instruction pairs, keeping both byte stores within the image. */
   for (size_t pair = program_size / 2;
        pair < AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE / 2; ++pair) {
      image[2 * pair] = 0x06;
      image[2 * pair + 1] = 0x00;
   }
   memcpy(image, program, program_size);
}

void
agx3_build_container_stop_program_agx3d(
   uint8_t image[AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE])
{
   /* G18G (agx3d): `end` is 08 04 and `nop` 08 00. Every applegpu_g18g
    * fragment witness without a constant program of its own
    * (test/native/g18g oracle.*.fragment.g18g.bin) carries exactly this:
    * end, then 31 nops. The M5 paths only ever use the stop program. */
   image[0] = 0x08;
   image[1] = 0x04;
   for (size_t offset = 2; offset < AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE;
        offset += 2) {
      image[offset] = 0x08;
      image[offset + 1] = 0x00;
   }
}

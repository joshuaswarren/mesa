/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include "hk_shader_upload_core.h"
#include <assert.h>
#include <stdio.h>

#define ALIGN 256u

int
main(void)
{
   /* A 400 KB main followed by its preamble: upload from the preamble on. */
   assert(hk_shader_upload_start(420476, true, 1569 * ALIGN, 0, 0, ALIGN) ==
          1569 * ALIGN);

   /* An unaligned preamble keeps its alignment: start at the aligned byte
    * before it. */
   assert(hk_shader_upload_start(136354, true, 509 * ALIGN + 64, 0, 0, ALIGN) ==
          509 * ALIGN);

   /* Rodata precedes the main, so a shader with rodata keeps the whole
    * binary. */
   assert(hk_shader_upload_start(5000, true, 4096, 0, 12, ALIGN) == 0);
   assert(hk_shader_upload_start(5000, false, 0, 0, 12, ALIGN) == 0);

   /* A preamble without rodata at the start of the binary (empty main). */
   assert(hk_shader_upload_start(640, true, 0, 0, 0, ALIGN) == 0);

   /* Rodata alone, past an empty first block. */
   assert(hk_shader_upload_start(1024, false, 0, 520, 4, ALIGN) == 512);

   /* A preamble offset of zero still counts when has_preamble is set; an
    * offset without has_preamble or rodata without size does not. */
   assert(hk_shader_upload_start(2048, false, 0, 0, 0, ALIGN) == 2048);
   assert(hk_shader_upload_start(2048, true, 1792, 256, 0, ALIGN) == 1792);

   /* Whatever the layout, the kept range covers each needed region and starts
    * aligned. */
   for (uint32_t pre = 0; pre < 4096; pre += 2) {
      for (uint32_t ro = 0; ro < 4096; ro += 4) {
         const uint32_t size = 8192;
         const uint32_t s = hk_shader_upload_start(size, true, pre, ro, 2, ALIGN);
         assert(s % ALIGN == 0 && s <= pre && s <= ro && s < size);
         const uint32_t t = hk_shader_upload_start(size, true, pre, ro, 0, ALIGN);
         assert(t % ALIGN == 0 && t <= pre && pre - t < ALIGN);
      }
   }

   printf("hk_shader_upload: ok\n");
   return 0;
}

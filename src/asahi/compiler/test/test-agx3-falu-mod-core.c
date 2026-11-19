/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The f32 modifier encoders must reproduce the composer's bytes -- the same
 * recipe (baseline + operands + modifiers) the RE proved byte-exact against
 * Apple -- across the abs/neg/saturate combinations of add, mul and fma.
 */

#include "agx3_falu_mod_core.h"
#include "agx3_falu_mod_core_vectors.c"

#include <stdio.h>
#include <string.h>

static unsigned
narrow_runtime_vectors(void)
{
   const struct {
      uint16_t dest, src;
      bool discard, dest_cache, saturate;
      uint8_t wait, size, bytes[8];
   } vectors[] = {
      {0,0,true,false,false,32,6,{0x01,0x01,0x1c,0x81,0,0xc2}},
      {1,2,true,false,false,0,6,{0x01,0x03,0x1c,0x81,0x04,0x02}},
      {3,6,true,false,false,0,6,{0x11,0x07,0x1c,0x81,0x04,0x02}},
      {0,2,false,true,false,32,6,{0x01,0x03,0x34,0x81,0,0xc2}},
      {2,2,true,false,true,32,8,{0x11,0x03,0x1c,0x81,0x01,0x02,0,0x82}},
   };
   unsigned fail = 0;
   for (unsigned i=0; i<sizeof(vectors)/sizeof(vectors[0]); ++i) {
      struct agx3_falu_runtime rt = {
         .discard={vectors[i].discard}, .dest_cache=vectors[i].dest_cache,
         .wait_mask=vectors[i].wait,
      };
      uint8_t bytes[16], length;
      if (!agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B, vectors[i].dest,
             vectors[i].src, vectors[i].saturate, &rt, bytes, &length) ||
          length!=vectors[i].size || memcmp(bytes,vectors[i].bytes,length)) {
         fprintf(stderr,"native narrowing vector %u failed\n",i);
         fail++;
      }
   }
   struct agx3_falu_runtime rt = {.cache={true},.wait_mask=63};
   uint8_t bytes[16], length=0;
   fail += !agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B,255,254,false,&rt,bytes,&length);
   fail += length>10;
   fail += agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B,256,254,false,&rt,bytes,&length);
   fail += agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B,0,255,false,&rt,bytes,&length);
   rt.discard[0]=true;
   fail += agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3B,0,2,false,&rt,bytes,&length);
   printf("native f32-to-f16: odd/even halves, live/dead input, saturate, waits: %u failures\n",fail);

   /* metal-nt applegpu_g17g, -fno-fast-math: half4(float4), and
    * half4(saturate(float4)). The first lane consumes load dependency 8. */
   const struct {
      uint16_t dest, src;
      bool saturate;
      uint8_t wait, size, bytes[12];
   } m5[] = {
      {0,0,false,128,12,{0x29,0x00,0x04,0x08,0x22,0x30,0xa0,0x02,0x84,0x06,0x42,0x00}},
      {1,2,false,0,6,{0x01,0x03,0x1c,0x81,0x04,0x02}},
      {2,4,false,0,6,{0x11,0x05,0x1c,0x81,0x00,0x02}},
      {3,6,false,0,6,{0x11,0x07,0x1c,0x81,0x04,0x02}},
      {0,0,true,128,12,{0x29,0x00,0x04,0x08,0x22,0x30,0xa0,0x02,0x84,0x16,0x42,0x00}},
      {1,2,true,0,8,{0x01,0x03,0x1c,0x81,0x05,0x02,0x00,0x02}},
   };
   for (unsigned i=0;i<sizeof(m5)/sizeof(m5[0]);i++) {
      struct agx3_falu_runtime runtime = {
         .discard={true}, .wait_mask=m5[i].wait,
      };
      if (!agx3_falu_narrow_f32(AGX_ISA_ENCODING_AGX3C,m5[i].dest,
             m5[i].src,m5[i].saturate,&runtime,bytes,&length) ||
          length != m5[i].size || memcmp(bytes,m5[i].bytes,length)) {
         fprintf(stderr,"M5 native narrowing vector %u failed\n",i);
         for (unsigned j=0;j<length;j++) fprintf(stderr,"%02x",bytes[j]);
         fprintf(stderr,"\n");
         fail++;
      }
   }
   printf("M5 native f32-to-f16: six Apple half/saturate/slot-8 vectors: %u failures\n",fail);

   /* blend/native/negpack3.py: half4(1.0 - x) and half4(-x - 1.0). */
   const struct {
      uint16_t dest, src;
      uint8_t minifloat, bytes[6];
   } sub[] = {
      {0, 0, 48, {0x01,0xb1,0x14,0x01,0x80,0x08}},
      {1, 2, 48, {0x01,0xb1,0x14,0x03,0x84,0x08}},
      {0, 0, 176, {0x01,0xb1,0x1c,0x01,0x80,0x08}},
      {1, 2, 176, {0x01,0xb1,0x1c,0x03,0x84,0x08}},
   };
   unsigned sub_fail = 0;
   for (unsigned i = 0; i < sizeof(sub) / sizeof(sub[0]); ++i) {
      struct agx3_falu_runtime srt = {.discard = {true}};
      uint8_t sbytes[16], slength;
      if (!agx3_falu_narrow_imm_sub_f32(AGX_ISA_ENCODING_AGX3B, sub[i].dest,
                                        sub[i].minifloat, sub[i].src, &srt,
                                        sbytes, &slength) ||
          slength != 6 || memcmp(sbytes, sub[i].bytes, 6)) {
         fprintf(stderr, "native narrowing subtract vector %u failed\n", i);
         sub_fail++;
      }
   }
   struct agx3_falu_runtime srt = {.cache = {true}, .discard = {true}};
   sub_fail += agx3_falu_narrow_imm_sub_f32(AGX_ISA_ENCODING_AGX3B, 0, 0x80,
                                            0, &srt, bytes, &length);
   sub_fail += agx3_falu_narrow_imm_sub_f32(AGX_ISA_ENCODING_AGX3B, 0, 0x80,
                                            1, &rt, bytes, &length);
   printf("native minifloat-minus-f32 to f16: %u failures\n", sub_fail);
   return fail + sub_fail;
}

static unsigned
add_runtime_vectors(void)
{
   struct add_vector {
      uint16_t src0, src1;
      struct agx3_falu_mods mods;
      struct agx3_falu_runtime runtime;
      unsigned length;
      uint8_t bytes[10];
   };
   const struct add_vector vectors[] = {
      /* Original x-y ordering commutes to Apple's shorter negate-first form. */
      {0, 2, {.src1_neg=true}, {.discard={true,true}, .wait_mask=32}, 6,
       {0x09,0x01,0x1c,0x03,0,0xc8}},
      {2, 0, {.src0_neg=true}, {.discard={true,true}, .wait_mask=32}, 6,
       {0x09,0x01,0x1c,0x03,0,0xc8}},
      {4, 2, {.src0_neg=true}, {.wait_mask=32}, 6,
       {0x09,0x03,0x04,0x05,0,0xc8}},
      {2, 0, {.src0_abs=true, .src0_neg=true, .src1_abs=true},
       {.discard={true,true}, .wait_mask=32}, 10,
       {0x09,0x01,0x1c,0x03,0x02,0x08,0,0x80,0x03,0}},
      {2, 0, {.src0_neg=true, .saturate=true},
       {.discard={true,true}, .wait_mask=32}, 8,
       {0x09,0x01,0x1c,0x03,0x01,0x08,0,0x82}},
      {2, 0, {.src0_neg=true, .src1_neg=true},
       {.discard={true,true}, .wait_mask=32}, 8,
       {0x09,0x01,0x1c,0x03,0x01,0x08,0x02,0x80}},
   };
   unsigned fail = 0;
   for (unsigned i = 0; i < sizeof(vectors)/sizeof(vectors[0]); ++i) {
      const struct add_vector *v = &vectors[i];
      uint8_t out[16], length;
      if (!agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0,
             v->src0, v->src1, &v->mods, &v->runtime, out, &length) ||
          length != v->length || memcmp(out, v->bytes, length)) {
         fprintf(stderr, "native modified ADD vector %u failed\n", i);
         fail++;
      }
   }
   /* G17G immediate-operand adds, Apple applegpu_g17g
    * (gpu/m5-xfb-alu4-reference.metal): `a + 2.0` and `fabs(a) + 0.5`, the
    * dead source waiting for the load in slot 8. */
   {
      static const struct {
         uint8_t minifloat; bool abs, neg; uint8_t bytes[12];
      } g17g[] = {
         {0x40, false, false, {0x29,0x00,0x04,0x0a,0x22,0x20,0xac,0x02,0x01,0x08,0x11,0x00}},
         {0x20, true,  false, {0x29,0x00,0x04,0x4a,0x22,0x20,0xac,0x02,0x01,0x08,0x10,0x40}},
         {0x20, false, true,  {0x29,0x00,0x24,0x4a,0x22,0x20,0xac,0x02,0x01,0x08,0x10,0x00}},
         {0x20, true,  true,  {0x29,0x00,0x24,0x4a,0x22,0x20,0xac,0x02,0x01,0x08,0x10,0x40}},
      };
      for (unsigned i = 0; i < sizeof(g17g)/sizeof(g17g[0]); ++i) {
         struct agx3_falu_mods im = {.src0_abs = g17g[i].abs, .src0_neg = g17g[i].neg};
         struct agx3_falu_runtime irt = {.discard = {true}, .wait_mask = 0x80};
         uint8_t o[16], l = 0;
         if (!agx3_falu_add_imm_runtime(0, 0, g17g[i].minifloat, &im, &irt,
                                        o, &l) ||
             l != 12 || memcmp(o, g17g[i].bytes, 12)) {
            fprintf(stderr, "G17G immediate ADD vector %u failed\n", i);
            fail++;
         }
      }
      /* G17G immediate-operand multiplies (gpu/m5-xfb-alu4-reference.metal,
       * -alu6-): `a * 2.0`, `fabs(a) * 2.0` and `-fabs(a) * 2.0`, the last
       * folded into the constant's sign. */
      static const struct {
         uint8_t minifloat; bool abs; uint8_t bytes[14];
      } g17g_mul[] = {
         {0x40, false, {0x29,0x00,0x05,0x10,0x22,0x80,0xac,0x02,0x0b,0x00,0x20,0x00,0x00,0x10}},
         {0x40, true,  {0x29,0x00,0x05,0x10,0x22,0x80,0xac,0x02,0x0b,0x00,0x20,0x08,0x00,0x10}},
         {0xc0, true,  {0x29,0x00,0x05,0x10,0x22,0x80,0xad,0x02,0x0b,0x00,0x20,0x08,0x00,0x10}},
      };
      for (unsigned i = 0; i < sizeof(g17g_mul)/sizeof(g17g_mul[0]); ++i) {
         struct agx3_falu_mods mm = {.src0_abs = g17g_mul[i].abs};
         struct agx3_falu_runtime mrt = {.discard = {true}, .wait_mask = 0x80};
         uint8_t o[16], l = 0;
         if (!agx3_falu_mul_imm_runtime(0, 0, g17g_mul[i].minifloat, &mm, &mrt,
                                        o, &l) ||
             l != 14 || memcmp(o, g17g_mul[i].bytes, 14)) {
            fprintf(stderr, "G17G immediate MUL vector %u failed\n", i);
            fail++;
         }
      }
      {
         struct agx3_falu_mods mm = {.src0_neg = true};
         struct agx3_falu_runtime mrt = {.wait_mask = 0x80};
         uint8_t o[16], l = 0;
         fail += agx3_falu_mul_imm_runtime(0, 0, 0x40, &mm, &mrt, o, &l);
      }

      struct agx3_falu_mods im = {.saturate = true};
      struct agx3_falu_runtime irt = {.wait_mask = 0x80};
      uint8_t o[16], l = 0;
      fail += agx3_falu_add_imm_runtime(0, 0, 0x80, &im, &irt, o, &l);
      im = (struct agx3_falu_mods){0};
      fail += agx3_falu_add_imm_runtime(1, 0, 0x80, &im, &irt, o, &l);
   }

   struct agx3_falu_mods m = {.src0_neg=true, .src1_abs=true};
   struct agx3_falu_runtime rt = {.wait_mask=63};
   uint8_t out[16], length = 0;
   /* Long forms must carry multiple dependency slots without a separate
    * arithmetic operation or discarding either live input. */
   fail += !agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 254, 252, 250,
                                       &m, &rt, out, &length);
   fail += length > 10;
   memset(out, 0xa5, sizeof(out)); length = 0xa5;
   rt.cache[0] = rt.discard[0] = true;
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 4,
                                      &m, &rt, out, &length);
   rt.cache[0] = false; rt.wait_mask = 64;
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 4,
                                      &m, &rt, out, &length);
   rt.wait_mask = 0;
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 255, 4,
                                      &m, &rt, out, &length);
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX2, 0, 2, 4,
                                      &m, &rt, out, &length);
   fail += length != 0xa5;
   for (unsigned i=0; i<sizeof(out); ++i) fail += out[i] != 0xa5;
   printf("native modified ADD: five Apple vectors plus commuted subtraction, live operands, waits and bounds: %u failures\n",fail);
   return fail;
}

static unsigned
m5_runtime_vectors(void)
{
   /* First three vectors are native G17G Metal kernels retaining the input
    * operands in their output. They therefore wait for slot8 without
    * discarding the sources. Remaining vectors exercise independent hints
    * and all eight waits against the G17G instruction composer. */
   const struct {
      unsigned op;
      uint16_t dest, src0, src1, addend;
      struct agx3_falu_mods mods;
      struct agx3_falu_runtime runtime;
      uint8_t length, bytes[16];
   } vectors[] = {
      {0,0,4,2,0,{.src0_neg=true},{.wait_mask=128},12,
       {0x29,0x02,0x24,0x0a,0x22,0x20,0xa0,0x02,0x02,0x00,0x12,0x00}},
      {1,0,2,4,0,{.src0_neg=true,.src1_abs=true,.saturate=true},
       {.wait_mask=128},14,
       {0x29,0x00,0x25,0x30,0x22,0x80,0xa0,0x02,0x98,0x00,0x04,0x02,0x00,0x10}},
      {2,0,4,2,6,{.src0_abs=true,.addend_neg=true},{.wait_mask=128},16,
       {0x29,0x02,0x07,0x0a,0x20,0x20,0xa3,0x02,0x18,0x01,0x80,0x08,0x06,0x00,0x00,0xc0}},
      {0,2,32,8,0,{.src1_neg=true},{.cache={false,true}},6,
       {0x19,0x21,0x04,0x89,0x00,0x08}},
      {0,2,32,8,0,{.src1_neg=true},{.cache={false,true},.wait_mask=255},12,
       {0x39,0xfc,0x44,0x2a,0x22,0x20,0xa0,0x82,0x20,0x00,0x12,0x04}},
      {1,8,32,6,0,{.src0_neg=true},
       {.cache={true},.discard={false,true},.wait_mask=255},14,
       {0x29,0x7e,0x65,0x30,0x22,0x80,0xa0,0x92,0x1c,0x08,0x10,0x00,0x00,0x10}},
      {2,8,32,6,2,{.src0_abs=true,.addend_neg=true},
       {.cache={true,false,true},.discard={false,true},.wait_mask=255},16,
       {0x29,0x7e,0x47,0x1a,0x20,0x20,0xa3,0x92,0x38,0x08,0x90,0x08,0x02,0x04,0x00,0xc0}},
   };
   unsigned fail = 0;
   for (unsigned i = 0; i < sizeof(vectors)/sizeof(vectors[0]); ++i) {
      uint8_t out[16], length = 0;
      bool ok;
      if (vectors[i].op == 0)
         ok = agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3C,
            vectors[i].dest, vectors[i].src0, vectors[i].src1,
            &vectors[i].mods, &vectors[i].runtime, out, &length);
      else if (vectors[i].op == 1)
         ok = agx3_falu_mul_mod_runtime(AGX_ISA_ENCODING_AGX3C,
            vectors[i].dest, vectors[i].src0, vectors[i].src1,
            &vectors[i].mods, &vectors[i].runtime, out, &length);
      else
         ok = agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3C,
            vectors[i].dest, vectors[i].src0, vectors[i].src1, vectors[i].addend,
            &vectors[i].mods, &vectors[i].runtime, out, &length);
      if (!ok || length != vectors[i].length ||
          memcmp(out, vectors[i].bytes, length)) {
         fprintf(stderr, "M5 float runtime vector %u failed\n", i);
         fail++;
      }
   }
   struct agx3_falu_mods mods = {.src1_neg=true};
   struct agx3_falu_runtime rt = {.cache={true},.discard={true}};
   uint8_t out[16], length = 0xa5;
   memset(out, 0xa5, sizeof(out));
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3C,0,2,4,&mods,&rt,out,&length);
   fail += agx3_falu_mul_mod_runtime(AGX_ISA_ENCODING_AGX3C,0,2,4,&mods,&rt,out,&length);
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3C,0,2,4,6,&mods,&rt,out,&length);
   rt.cache[0] = false;
   fail += agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3C,0,3,4,&mods,&rt,out,&length);
   fail += agx3_falu_mul_mod_runtime(AGX_ISA_ENCODING_AGX3C,256,2,4,&mods,&rt,out,&length);
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3C,0,2,4,7,&mods,&rt,out,&length);
   fail += length != 0xa5;
   for (unsigned i = 0; i < sizeof(out); ++i) fail += out[i] != 0xa5;
   printf("M5 float runtime: native ADD/MUL/FMA, live inputs, hints and eight waits: %u failures\n", fail);
   return fail;
}

static unsigned
runtime_vectors(void)
{
   unsigned fail = 0;
   {
      const uint8_t expected[] = {0x09, 0x01, 0x1e, 0x03, 0x91, 0x04, 0x02, 0xc0};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 0, .src1_neg = 0, .addend_abs = 0, .addend_neg = 1, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 0 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x03, 0x06, 0x05, 0x11, 0x06, 0x02, 0xc0};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 0, .src1_neg = 0, .addend_abs = 0, .addend_neg = 1, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {0, 0, 0}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 4, 2, 6,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 1 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x01, 0x1e, 0x03, 0x83, 0x04, 0x02, 0x00, 0x00, 0x80, 0x01, 0x00};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 1, .src1_neg = 0, .addend_abs = 0, .addend_neg = 0, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 2 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x03, 0x1e, 0x01, 0x81, 0x04, 0x02, 0xc8};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 1, .src1_abs = 0, .src1_neg = 0, .addend_abs = 0, .addend_neg = 0, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 0, 2, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 3 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x01, 0x1e, 0x03, 0x89, 0x04, 0x02, 0xc0};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 0, .src1_neg = 0, .addend_abs = 1, .addend_neg = 0, .saturate = 0};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 4 failed\n");
         fail++;
      }
   }
   {
      const uint8_t expected[] = {0x09, 0x01, 0x1e, 0x03, 0x82, 0x04, 0x02, 0x00, 0x00, 0x82};
      const struct agx3_falu_mods m = {.src0_abs = 0, .src0_neg = 0, .src1_abs = 0, .src1_neg = 0, .addend_abs = 0, .addend_neg = 0, .saturate = 1};
      const struct agx3_falu_runtime rt = {.cache = {0, 0, 0}, .discard = {1, 1, 1}, .dest_cache = 0, .wait_mask = 32};
      uint8_t out[16], len;
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 4,
                                      &m, &rt, out, &len) ||
          len != sizeof(expected) || memcmp(out, expected, len)) {
         fprintf(stderr, "native modified FMA vector 5 failed\n");
         fail++;
      }
   }
   /* The witnessed abs source forces the long form. Independently inspect
    * all six physical wait bits and source hints, not the encoder's recipe. */
   for (unsigned waits = 0; waits < 64; ++waits) {
      const struct agx3_falu_mods m = {.src1_abs = true};
      struct agx3_falu_runtime rt = {.wait_mask = waits};
      uint8_t out[16], len;
      const unsigned wait_bits[] = {61, 62, 63, 77, 78, 79};
      if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 4,
                                    &m, &rt, out, &len) || len != 12) {
         fail++;
         continue;
      }
      for (unsigned i = 0; i < 6; ++i)
         fail += ((out[wait_bits[i] / 8] >> (wait_bits[i] % 8)) & 1) !=
                 ((waits >> i) & 1);
      const unsigned cache_bits[] = {31, 15, 47};
      const unsigned discard_bits[] = {20, 19, 39};
      for (unsigned hints = 0; hints < 27; ++hints) {
         unsigned v = hints;
         for (unsigned i = 0; i < 3; ++i, v /= 3) {
            rt.cache[i] = (v % 3) == 1;
            rt.discard[i] = (v % 3) == 2;
         }
         if (!agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 4,
                                       &m, &rt, out, &len) || len != 12) {
            fail++;
            continue;
         }
         for (unsigned i = 0; i < 3; ++i) {
            fail += ((out[cache_bits[i] / 8] >> (cache_bits[i] % 8)) & 1)
                    != rt.cache[i];
            fail += ((out[discard_bits[i] / 8] >> (discard_bits[i] % 8)) & 1)
                    != rt.discard[i];
         }
      }
   }
   struct agx3_falu_mods m = {.addend_neg = true};
   struct agx3_falu_runtime rt = {.wait_mask = 64};
   uint8_t out[16], len = 0xa5;
   memset(out, 0xa5, sizeof(out));
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 4, 6,
                                    &m, &rt, out, &len);
   rt.wait_mask = 0;
   rt.cache[0] = rt.discard[0] = true;
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 4, 6,
                                    &m, &rt, out, &len);
   rt.cache[0] = false;
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX3B, 0, 2, 4, 255,
                                    &m, &rt, out, &len);
   fail += agx3_falu_fma_mod_runtime(AGX_ISA_ENCODING_AGX2, 0, 2, 4, 6,
                                    &m, &rt, out, &len);
   fail += len != 0xa5;
   for (unsigned i = 0; i < sizeof(out); ++i)
      fail += out[i] != 0xa5;
   printf("native modified FMA: six Apple byte vectors, 64 wait masks, "
          "27 independent lifetime combinations: %u failures\n", fail);
   return fail;
}

static unsigned
mul_runtime_vectors(void)
{
   const struct {
      uint16_t dst, s0, s1;
      struct agx3_falu_mods m;
      struct agx3_falu_runtime rt;
      unsigned size;
      uint8_t bytes[16];
   } vectors[] = {
      {0,6,4,{.src0_abs=0,.src0_neg=0,.src1_abs=0,.src1_neg=0,.saturate=1}, {.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},8,{0x09,0x05,0x1d,0x07,0x01,0x00,0x00,0x82}},
      {0,4,2,{.src0_abs=0,.src0_neg=0,.src1_abs=0,.src1_neg=0,.saturate=1}, {.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x05,0x05,0x01,0x00,0x00,0x82}},
      {0,6,4,{.src0_abs=0,.src0_neg=0,.src1_abs=1,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},12,{0x09,0x05,0x1d,0x07,0x02,0x00,0x00,0x80,0x01,0x00,0x00,0x00}},
      {0,4,2,{.src0_abs=0,.src0_neg=0,.src1_abs=1,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},12,{0x09,0x03,0x05,0x05,0x02,0x00,0x00,0x80,0x01,0x00,0x00,0x00}},
      {0,4,6,{.src0_abs=0,.src0_neg=1,.src1_abs=0,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},6,{0x09,0x07,0x1d,0x05,0x00,0xc8}},
      {0,2,4,{.src0_abs=0,.src0_neg=1,.src1_abs=0,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},6,{0x09,0x05,0x05,0x03,0x00,0xc8}},
      {0,4,6,{.src0_abs=1,.src0_neg=1,.src1_abs=1,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},12,{0x09,0x07,0x1d,0x05,0x02,0x08,0x00,0x80,0x03,0x00,0x00,0x00}},
      {0,2,4,{.src0_abs=1,.src0_neg=1,.src1_abs=1,.src1_neg=0,.saturate=0}, {.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},12,{0x09,0x05,0x05,0x03,0x02,0x08,0x00,0x80,0x03,0x00,0x00,0x00}},
   };
   unsigned fail = 0;
   for (unsigned i = 0; i < sizeof(vectors)/sizeof(vectors[0]); ++i) {
      uint8_t out[16], size;
      if (!agx3_falu_mul_mod_runtime(AGX_ISA_ENCODING_AGX3B,
              vectors[i].dst, vectors[i].s0, vectors[i].s1,
              &vectors[i].m, &vectors[i].rt, out, &size) ||
          size != vectors[i].size || memcmp(out, vectors[i].bytes, size)) {
         fprintf(stderr, "native modified FMUL vector %u failed size=%u: ", i, size);
         for (unsigned j=0;j<size;++j) fprintf(stderr,"%02x",out[j]);
         fprintf(stderr,"\n");
         fail++;
      }
   }
   printf("native modified FMUL: eight Apple live/dead byte vectors: %u failures\n", fail);
   return fail;
}

static unsigned
fma_addend_vectors(void)
{
   const struct {
      uint16_t dst,s0,s1; uint8_t imm;
      struct agx3_falu_runtime rt;
      unsigned size; uint8_t bytes[16];
   } vectors[] = {
      {0,6,4,48,{.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},8,{0x09,0x05,0x1e,0x07,0x21,0xb0,0x02,0xc0}},
      {0,4,2,48,{.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0x05,0x21,0xb0,0x02,0xc0}},
      {0,6,4,176,{.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},8,{0x09,0x05,0x1e,0x07,0xa1,0xb0,0x02,0xc0}},
      {0,4,2,176,{.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0x05,0xa1,0xb0,0x02,0xc0}},
      {0,6,4,32,{.cache={0,0},.discard={1,1},.dest_cache=0,.wait_mask=32},8,{0x09,0x05,0x1e,0x07,0x21,0xa0,0x02,0xc0}},
      {0,4,2,32,{.cache={0,0},.discard={0,0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0x05,0x21,0xa0,0x02,0xc0}},
   };
   unsigned fail=0;
   for(unsigned i=0;i<sizeof(vectors)/sizeof(vectors[0]);++i){
      uint8_t out[16],size=0;
      if(!agx3_falu_fma_addend_imm(AGX_ISA_ENCODING_AGX3B,vectors[i].dst,
             vectors[i].s0,vectors[i].s1,vectors[i].imm,&vectors[i].rt,out,&size)||
         size!=vectors[i].size||memcmp(out,vectors[i].bytes,size)){
         fprintf(stderr,"FMA minifloat addend vector %u failed\n",i);fail++;
      }
   }
   {
      const struct {
         uint16_t dst, src0, src1;
         struct agx3_falu_runtime rt;
         uint8_t bytes[8];
      } negative_multiplier[] = {
         {0, 0, 2, {.discard={1,1}},
          {0x09,0x03,0x1e,0x01,0x21,0x98,0x02,0xc8}},
         {0, 2, 4, {.cache={0,0}, .discard={0,0}},
          {0x09,0x05,0x06,0x03,0x21,0x98,0x02,0xc8}},
      };
      for (unsigned i = 0;
           i < sizeof(negative_multiplier) / sizeof(negative_multiplier[0]);
           ++i) {
         uint8_t out[16], size = 0;
         if (!agx3_falu_fma_addend_imm_mod(
                AGX_ISA_ENCODING_AGX3B, negative_multiplier[i].dst,
                negative_multiplier[i].src0, negative_multiplier[i].src1,
                24, false, true, false, false,
                &negative_multiplier[i].rt, out, &size) ||
             size != sizeof(negative_multiplier[i].bytes) ||
             memcmp(out, negative_multiplier[i].bytes, size)) {
            fprintf(stderr,
                    "FMA minifloat addend negated-multiplier vector %u failed\n",
                    i);
            for (unsigned j = 0; j < size; ++j)
               fprintf(stderr, "%02x", out[j]);
            fprintf(stderr, " size=%u\n", size);
            fail++;
         }
      }
   }
   {
      /* applegpu-nt v32023.255, -arch applegpu_g17p:
       * fma(a, fabs(b), 1.0f).  This is the exact modifier-bearing
       * minifloat-addend form emitted by production GTK shaders. */
      static const uint8_t native[] = {
         0x09,0x01,0x1e,0x03,0x23,0xb0,0x02,0x00,0x00,0x80,0x02,0x00,
      };
      const struct agx3_falu_runtime rt = {.discard={true,true}};
      uint8_t out[16], size = 0;
      if (!agx3_falu_fma_addend_imm_mod(
             AGX_ISA_ENCODING_AGX3B, 0, 2, 0, 48,
             true, false, false, false, &rt, out, &size) ||
          size != sizeof(native) || memcmp(out, native, size)) {
         fprintf(stderr, "FMA absolute multiplier/immediate addend failed\n");
         for (unsigned i = 0; i < size; ++i)
            fprintf(stderr, "%02x", out[i]);
         fprintf(stderr, " size=%u\n", size);
         fail++;
      }
   }
   printf("native FMA minifloat addend: nine Apple live/dead/modifier vectors: "
          "%u failures\n", fail);
   return fail;
}

static unsigned
fma_two_vectors(void)
{
   const struct {
      uint16_t dst,src; uint8_t mul,add;
      struct agx3_falu_runtime rt;
      unsigned size; uint8_t bytes[16];
   } vectors[] = {
      {0,0,64,176,{.cache={0},.discard={1},.dest_cache=0,.wait_mask=32},8,{0x09,0x01,0x0e,0xc1,0xa1,0xb0,0x02,0xc2}},
      {0,2,64,176,{.cache={0},.discard={0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0xc1,0xa1,0xb0,0x02,0xc2}},
      {0,0,32,16,{.cache={0},.discard={1},.dest_cache=0,.wait_mask=32},8,{0x09,0x01,0x0e,0xa1,0x21,0x90,0x02,0xc2}},
      {0,2,32,16,{.cache={0},.discard={0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0xa1,0x21,0x90,0x02,0xc2}},
      {0,0,56,160,{.cache={0},.discard={1},.dest_cache=0,.wait_mask=32},8,{0x09,0x01,0x0e,0xb9,0xa1,0xa0,0x02,0xc2}},
      {0,2,56,160,{.cache={0},.discard={0},.dest_cache=0,.wait_mask=32},8,{0x09,0x03,0x06,0xb9,0xa1,0xa0,0x02,0xc2}},
   };
   unsigned fail=0;
   for(unsigned i=0;i<sizeof(vectors)/sizeof(vectors[0]);++i){
      uint8_t out[16],size=0;
      if(!agx3_falu_fma_two_imm(AGX_ISA_ENCODING_AGX3B,vectors[i].dst,
             vectors[i].src,vectors[i].mul,vectors[i].add,&vectors[i].rt,out,&size)||
         size!=vectors[i].size||memcmp(out,vectors[i].bytes,size)){
         fprintf(stderr,"FMA two-immediate vector %u failed\n",i);fail++;
      }
   }
   printf("native FMA two minifloats: six Apple live/dead vectors: %u failures\n",fail);
   return fail;
}

/*
 * FMA with a minifloat MULTIPLIER and modifier-bearing register operands.
 * Apple's own applegpu_g17p output for the five cells; each kernel is
 * `o[i] = fma(<k>, <b>, <c>)` over device loads, so both register operands
 * are dead after the FMA (f29/f63) and it waits on the loads' slot (f102).
 *
 * The third vector is the one glmark2's terrain noise kernel needs: an
 * immediate multiplier standing on the same encoding point as two absolute
 * register operands.  The fourth records that a negated immediate multiplier
 * is the minifloat's sign bit, not a modifier bit.
 */
static unsigned
fma_mul_imm_vectors(void)
{
   const struct agx3_falu_runtime rt = {
      .cache = {false, false, false},
      .discard = {false, true, true},
      .dest_cache = false,
      .wait_mask = 0x20,
   };
   const struct {
      const char *name;
      uint8_t mul;
      uint16_t src1, addend;
      struct agx3_falu_mods m;
      unsigned size;
      uint8_t bytes[16];
   } vectors[] = {
      {"fma(0.5, b, c)", 32, 0, 4, {0}, 8,
       {0x09,0x01,0x0e,0xa1,0x81,0x04,0x02,0xc2}},
      {"fma(0.5, fabs(b), c)", 32, 0, 4, {.src1_abs = true}, 12,
       {0x09,0x01,0x0e,0xa1,0x83,0x04,0x02,0x02,0x00,0x80,0x01,0x00}},
      {"fma(0.5, fabs(b), fabs(c))", 32, 0, 4,
       {.src1_abs = true, .addend_abs = true}, 12,
       {0x09,0x01,0x0e,0xa1,0x8b,0x04,0x02,0x02,0x00,0x80,0x01,0x00}},
      {"fma(-0.5, b, c)", 160, 0, 4, {0}, 8,
       {0x09,0x01,0x1e,0xa1,0x81,0x04,0x02,0xc2}},
      {"fma(0.5, fabs(b), -fabs(c))", 32, 0, 4,
       {.src1_abs = true, .addend_abs = true, .addend_neg = true}, 12,
       {0x09,0x01,0x0e,0xa1,0x9b,0x04,0x02,0x02,0x00,0x80,0x01,0x00}},

      /* The remaining cells of the kind x width x modifier x saturate grid on
       * this form, witnessed rather than assumed: orthogonality is grounded
       * for f32add/f32mul/f32fma/fsel and the bitwise ops, not for the ISA at
       * large, so each cell that the encoder admits gets its own kernel
       * (fmaimm2.metal).  f14 saturates beside the immediate; f26 narrows the
       * multiplicand and f60 the addend, each beside its own abs bit. */
      {"saturate(fma(0.5, fabs(b), fabs(c)))", 32, 0, 4,
       {.src1_abs = true, .addend_abs = true, .saturate = true}, 12,
       {0x09,0x01,0x0e,0xa1,0x8b,0x04,0x02,0x02,0x00,0x82,0x01,0x00}},
      {"saturate(fma(0.5, b, c))", 32, 0, 4, {.saturate = true}, 10,
       {0x09,0x01,0x0e,0xa1,0x82,0x04,0x02,0x02,0x00,0x82}},
      {"fma(0.5, half b, c)", 32, 4, 0, {.src1_16 = true}, 8,
       {0x09,0x04,0x0e,0xa1,0x81,0x00,0x02,0xc2}},
      {"fma(0.5, b, half c)", 32, 0, 4, {.addend_16 = true}, 8,
       {0x09,0x01,0x0e,0xa1,0x81,0x04,0x00,0xc2}},
      {"fma(0.5, fabs(half b), fabs(c))", 32, 4, 0,
       {.src1_16 = true, .src1_abs = true, .addend_abs = true}, 12,
       {0x09,0x04,0x0e,0xa1,0x8b,0x00,0x02,0x02,0x00,0x80,0x01,0x00}},

      /* Apple has no f31 in this form. Asked for a negated multiplicand it
       * folds the sign into the minifloat and leaves f30/f64 alone, which is
       * exact because k * -b == -k * b -- so these two are written with the
       * sign already folded, as the packer folds it. */
      {"fma(0.5, -fabs(b), c) folded", 160, 0, 4, {.src1_abs = true}, 12,
       {0x09,0x01,0x1e,0xa1,0x83,0x04,0x02,0x02,0x00,0x80,0x01,0x00}},
      {"fma(0.5, -b, fabs(c)) folded", 160, 0, 4, {.addend_abs = true}, 8,
       {0x09,0x01,0x1e,0xa1,0x89,0x04,0x02,0xc2}},
   };
   unsigned fail = 0;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], size = 0;
      if (!agx3_falu_fma_mul_imm_mod(AGX_ISA_ENCODING_AGX3B, 0, vectors[i].mul,
                                     vectors[i].src1, vectors[i].addend,
                                     &vectors[i].m, &rt, out, &size) ||
          size != vectors[i].size ||
          memcmp(out, vectors[i].bytes, size) != 0) {
         fprintf(stderr, "FMA minifloat multiplier %s failed: ",
                 vectors[i].name);
         for (unsigned j = 0; j < size; ++j)
            fprintf(stderr, "%02x", out[j]);
         fprintf(stderr, " size=%u\n", size);
         fail++;
      }
   }

   /* There is no f35 register slot in this form, so a multiplier-side
    * modifier or lifetime hint has nowhere to go and must be refused rather
    * than dropped: the caller has to fold it into the minifloat.  The
    * multiplicand's negate is refused for the second reason above -- f31 is
    * unwitnessed beside an immediate multiplier, so it fails closed instead
    * of being emitted on the strength of the other slots' behaviour. */
   {
      const struct agx3_falu_mods bad[] = {
         {.src0_neg = true}, {.src0_abs = true}, {.src0_16 = true},
         {.src1_neg = true},
      };
      for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
         uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], size = 0;
         if (agx3_falu_fma_mul_imm_mod(AGX_ISA_ENCODING_AGX3B, 0, 32, 0, 4,
                                       &bad[i], &rt, out, &size)) {
            fprintf(stderr,
                    "FMA minifloat multiplier accepted unwitnessed mods %u\n", i);
            fail++;
         }
      }
   }

   printf("native FMA minifloat multiplier: twelve Apple modifier/width/"
          "saturate/sign-fold vectors: "
          "%u failures\n", fail);
   return fail;
}

/* The same immediate-multiplier cell on G17G: Apple's applegpu_g17g bytes
 * from test/native/g17g/m5-compare-select-reference.metal. The last one is
 * glamor's reflect-repeat fma(-2.0, x, |y|) with the absolute addend kept in
 * the instruction. */
static unsigned
m5_fma_mul_imm_vectors(void)
{
   const struct agx3_falu_runtime rt = {
      .discard = {false, true, true},
   };
   const struct {
      const char *name;
      uint8_t mul;
      uint16_t src1, addend;
      struct agx3_falu_mods m;
      uint8_t size, bytes[8];
   } vectors[] = {
      {"glm_fma_neg_imm_reg fma(-2.0, x, y)", 192, 0, 2, {0}, 6,
       {0x09, 0x01, 0x1e, 0xc1, 0xa0, 0x02}},
      {"glm_fma_imm_mul fma(x, 2.0, y)", 64, 0, 2, {0}, 6,
       {0x09, 0x01, 0x0e, 0xc1, 0xa0, 0x02}},
      {"glm_fma_negimm_absload fma(-2.0, x, |y|)", 192, 0, 4,
       {.addend_abs = true}, 8,
       {0x09, 0x01, 0x1e, 0xc1, 0x89, 0x04, 0x02, 0x02}},
   };
   unsigned fail = 0;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], size = 0;
      if (!agx3_falu_fma_mul_imm_mod(AGX_ISA_ENCODING_AGX3C, 0, vectors[i].mul,
                                     vectors[i].src1, vectors[i].addend,
                                     &vectors[i].m, &rt, out, &size) ||
          size != vectors[i].size ||
          memcmp(out, vectors[i].bytes, size) != 0) {
         fprintf(stderr, "G17G FMA minifloat multiplier %s failed\n",
                 vectors[i].name);
         fail++;
      }
   }
   /* A negated register multiplicand is not witnessed on G17G. */
   const struct agx3_falu_mods bad[] = {
      {.src1_neg = true},
   };
   for (unsigned i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
      uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], size = 0;
      if (agx3_falu_fma_mul_imm_mod(AGX_ISA_ENCODING_AGX3C, 0, 192, 0, 4,
                                    &bad[i], &rt, out, &size)) {
         fprintf(stderr, "G17G FMA minifloat multiplier accepted %u\n", i);
         fail++;
      }
   }
   printf("G17G FMA minifloat multiplier: three Apple vectors: %u failures\n",
          fail);
   return fail;
}

static unsigned
m5_fma_immediate_vectors(void)
{
   /* Independent metal-nt applegpu_g17g output from eleven addend/minifloat,
    * live-input, ABS and NEG programs. Expected bytes are Apple's output. */
   const struct {
      bool two;
      uint16_t dest, src0, src1;
      uint8_t immediate;
      bool abs0, neg0, abs1, neg1;
      struct agx3_falu_runtime runtime;
      uint8_t size, bytes[16];
   } vectors[] = {
      {false, 0, 8, 0, 32, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=128},
       16, {0x29,0x00,0x07,0x0a,0x20,0x20,0xa3,0x02,0x38,0x02,0x60,0x00,0x20,0x06,0x00,0x80}},
      {false, 2, 10, 2, 32, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       8, {0x19,0x03,0x1e,0x0b,0x21,0xa0,0x02,0x00}},
      {false, 4, 12, 4, 32, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       8, {0x29,0x05,0x1e,0x0d,0x21,0xa0,0x02,0x00}},
      {false, 6, 14, 6, 32, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       8, {0x39,0x07,0x1e,0x0f,0x21,0xa0,0x02,0x00}},
      {false, 12, 12, 4, 32, false, false, false, false,
       {.dest_cache=true, .cache={false,true}, .discard={true,false}, .wait_mask=128},
       16, {0x29,0x80,0x07,0x1a,0x28,0x20,0xa3,0x1a,0x18,0x03,0x60,0x00,0x20,0x06,0x00,0x80}},
      {false, 14, 14, 6, 32, false, false, false, false,
       {.dest_cache=true, .cache={false,true}, .discard={true,false}, .wait_mask=0},
       8, {0x79,0x87,0x36,0x0f,0x21,0xa0,0x02,0x00}},
      {false, 8, 8, 0, 32, false, false, false, false,
       {.dest_cache=true, .cache={false,true}, .discard={true,false}, .wait_mask=0},
       8, {0x49,0x81,0x36,0x09,0x21,0xa0,0x02,0x00}},
      {false, 10, 10, 2, 32, false, false, false, false,
       {.dest_cache=true, .cache={false,true}, .discard={true,false}, .wait_mask=0},
       8, {0x59,0x83,0x36,0x0b,0x21,0xa0,0x02,0x00}},
      {false, 12, 4, 12, 32, false, false, false, false,
       {.dest_cache=true, .cache={true,false}, .discard={false,true}, .wait_mask=128},
       16, {0x29,0x00,0x07,0x3a,0x28,0x20,0xa3,0x1a,0x38,0x01,0x50,0x00,0x20,0x06,0x00,0x80}},
      {false, 14, 6, 14, 32, false, false, false, false,
       {.dest_cache=true, .cache={true,false}, .discard={false,true}, .wait_mask=0},
       8, {0x79,0x0f,0x2e,0x87,0x21,0xa0,0x02,0x00}},
      {false, 8, 0, 8, 32, false, false, false, false,
       {.dest_cache=true, .cache={true,false}, .discard={false,true}, .wait_mask=0},
       8, {0x49,0x09,0x2e,0x81,0x21,0xa0,0x02,0x00}},
      {false, 10, 2, 10, 32, false, false, false, false,
       {.dest_cache=true, .cache={true,false}, .discard={false,true}, .wait_mask=0},
       8, {0x59,0x0b,0x2e,0x83,0x21,0xa0,0x02,0x00}},
      {false, 0, 8, 0, 160, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=128},
       16, {0x29,0x00,0x07,0x0a,0x20,0x20,0xa3,0x02,0x38,0x02,0x62,0x00,0x20,0x06,0x00,0x80}},
      {false, 2, 10, 2, 160, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       8, {0x19,0x03,0x1e,0x0b,0xa1,0xa0,0x02,0x00}},
      {false, 4, 12, 4, 160, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       8, {0x29,0x05,0x1e,0x0d,0xa1,0xa0,0x02,0x00}},
      {false, 6, 14, 6, 160, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       8, {0x39,0x07,0x1e,0x0f,0xa1,0xa0,0x02,0x00}},
      {false, 0, 0, 8, 32, false, true, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=128},
       16, {0x29,0x00,0x27,0x2a,0x20,0x20,0xa3,0x02,0x38,0x00,0x60,0x00,0x20,0x06,0x00,0x80}},
      {false, 2, 2, 10, 32, false, true, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       8, {0x19,0x0b,0x1e,0x03,0x21,0xa0,0x02,0x08}},
      {false, 4, 4, 12, 32, false, true, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       8, {0x29,0x0d,0x1e,0x05,0x21,0xa0,0x02,0x08}},
      {false, 6, 6, 14, 32, false, true, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       8, {0x39,0x0f,0x1e,0x07,0x21,0xa0,0x02,0x08}},
      {false, 0, 8, 0, 32, false, false, true, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=128},
       16, {0x29,0x00,0x07,0x0a,0x20,0x20,0xa3,0x02,0x38,0x02,0x60,0x00,0x60,0x06,0x00,0x80}},
      {false, 2, 10, 2, 32, false, false, true, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       12, {0x19,0x03,0x1e,0x0b,0x23,0xa0,0x02,0x00,0x00,0x00,0x01,0x00}},
      {false, 4, 12, 4, 32, false, false, true, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       12, {0x29,0x05,0x1e,0x0d,0x23,0xa0,0x02,0x00,0x00,0x00,0x01,0x00}},
      {false, 6, 14, 6, 32, false, false, true, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       12, {0x39,0x07,0x1e,0x0f,0x23,0xa0,0x02,0x00,0x00,0x00,0x01,0x00}},
      {false, 0, 8, 0, 32, false, true, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=128},
       16, {0x29,0x00,0x27,0x0a,0x20,0x20,0xa3,0x02,0x38,0x02,0x60,0x00,0x20,0x06,0x00,0x80}},
      {false, 2, 10, 2, 32, false, true, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       8, {0x19,0x03,0x1e,0x0b,0x21,0xa0,0x02,0x08}},
      {false, 4, 12, 4, 32, false, true, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       8, {0x29,0x05,0x1e,0x0d,0x21,0xa0,0x02,0x08}},
      {false, 6, 14, 6, 32, false, true, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       8, {0x39,0x07,0x1e,0x0f,0x21,0xa0,0x02,0x08}},
      {false, 0, 8, 0, 32, true, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=128},
       16, {0x29,0x00,0x07,0x0a,0x20,0x20,0xa3,0x02,0x38,0x02,0x60,0x08,0x20,0x06,0x00,0x80}},
      {false, 2, 10, 2, 32, true, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       12, {0x19,0x03,0x1e,0x0b,0x23,0xa0,0x02,0x00,0x00,0x00,0x02,0x00}},
      {false, 4, 12, 4, 32, true, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       12, {0x29,0x05,0x1e,0x0d,0x23,0xa0,0x02,0x00,0x00,0x00,0x02,0x00}},
      {false, 6, 14, 6, 32, true, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,true}, .wait_mask=0},
       12, {0x39,0x07,0x1e,0x0f,0x23,0xa0,0x02,0x00,0x00,0x00,0x02,0x00}},
      {true, 0, 0, 32, 16, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,false}, .wait_mask=128},
       16, {0x29,0x00,0x07,0x0a,0x20,0x20,0xa3,0x02,0x30,0x88,0x58,0x00,0x10,0x06,0x00,0x80}},
      {true, 2, 2, 32, 16, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,false}, .wait_mask=0},
       8, {0x19,0x03,0x0e,0xa1,0x21,0x90,0x02,0x02}},
      {true, 4, 4, 32, 16, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,false}, .wait_mask=0},
       8, {0x29,0x05,0x0e,0xa1,0x21,0x90,0x02,0x02}},
      {true, 6, 6, 32, 16, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,false}, .wait_mask=0},
       8, {0x39,0x07,0x0e,0xa1,0x21,0x90,0x02,0x02}},
      {true, 14, 0, 32, 16, false, false, false, false,
       {.dest_cache=true, .cache={true,false}, .discard={false,false}, .wait_mask=128},
       16, {0x39,0x80,0x07,0x0a,0x28,0x20,0xa3,0x1a,0x10,0x88,0x58,0x00,0x10,0x06,0x00,0x80}},
      {true, 10, 2, 32, 16, false, false, false, false,
       {.dest_cache=true, .cache={true,false}, .discard={false,false}, .wait_mask=0},
       8, {0x59,0x83,0x26,0xa1,0x21,0x90,0x02,0x02}},
      {true, 12, 4, 32, 16, false, false, false, false,
       {.dest_cache=true, .cache={true,false}, .discard={false,false}, .wait_mask=0},
       8, {0x69,0x85,0x26,0xa1,0x21,0x90,0x02,0x02}},
      {true, 10, 6, 32, 16, false, false, false, false,
       {.dest_cache=true, .cache={true,false}, .discard={false,false}, .wait_mask=0},
       8, {0x59,0x87,0x26,0xa1,0x21,0x90,0x02,0x02}},
      {true, 0, 0, 160, 144, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,false}, .wait_mask=128},
       16, {0x29,0x00,0x07,0x0a,0x20,0x20,0xa3,0x02,0x30,0xa8,0x5a,0x00,0x10,0x06,0x00,0x80}},
      {true, 2, 2, 160, 144, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,false}, .wait_mask=0},
       8, {0x19,0x03,0x1e,0xa1,0xa1,0x90,0x02,0x02}},
      {true, 4, 4, 160, 144, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,false}, .wait_mask=0},
       8, {0x29,0x05,0x1e,0xa1,0xa1,0x90,0x02,0x02}},
      {true, 6, 6, 160, 144, false, false, false, false,
       {.dest_cache=false, .cache={false,false}, .discard={true,false}, .wait_mask=0},
       8, {0x39,0x07,0x1e,0xa1,0xa1,0x90,0x02,0x02}},
   };
   unsigned fail = 0;
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], size = 0;
      bool ok = vectors[i].two
         ? agx3_falu_fma_two_imm(AGX_ISA_ENCODING_AGX3C, vectors[i].dest,
              vectors[i].src0, vectors[i].src1, vectors[i].immediate,
              &vectors[i].runtime, out, &size)
         : agx3_falu_fma_addend_imm_mod(AGX_ISA_ENCODING_AGX3C, vectors[i].dest,
              vectors[i].src0, vectors[i].src1, vectors[i].immediate,
              vectors[i].abs0, vectors[i].neg0, vectors[i].abs1, vectors[i].neg1,
              &vectors[i].runtime, out, &size);
      if (!ok || size != vectors[i].size || memcmp(out, vectors[i].bytes, size)) {
         fprintf(stderr, "M5 native FMA immediate vector %u failed\n", i);
         fail++;
      }
   }
   struct agx3_falu_runtime bad = {0};
   uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], size = 0;
   fail += agx3_falu_fma_two_imm(AGX_ISA_ENCODING_AGX3C, 1, 2, 32, 16, &bad, out, &size);
   fail += agx3_falu_fma_addend_imm(AGX_ISA_ENCODING_AGX3C, 0, 256, 4, 32, &bad, out, &size);
   bad.wait_mask = 0;
   bad.cache[0] = bad.discard[0] = true;
   fail += agx3_falu_fma_two_imm(AGX_ISA_ENCODING_AGX3C, 0, 2, 32, 16, &bad, out, &size);
   fail += agx3_falu_fma_addend_imm(AGX_ISA_ENCODING_AGX3C, 0, 2, 4, 32, &bad, out, &size);
   printf("M5 FMA immediates: 44 Apple vectors and invalid registers/lifetimes: %u failures\n", fail);
   return fail;
}


int
main(void)
{
   unsigned pass = 0, fail = runtime_vectors() + add_runtime_vectors() + narrow_runtime_vectors() + mul_runtime_vectors() + fma_addend_vectors() + fma_two_vectors() + fma_mul_imm_vectors() + m5_runtime_vectors();
   fail += m5_fma_immediate_vectors();
   fail += m5_fma_mul_imm_vectors();
   for (size_t i = 0; i < mod_vec_count; ++i) {
      const struct mod_vec *v = &mod_vecs[i];
      const enum agx_isa_encoding e =
         v->enc == 0 ? AGX_ISA_ENCODING_AGX3B : AGX_ISA_ENCODING_AGX3C;
      const struct agx3_falu_mods m = {
         .src0_abs = v->s0a, .src0_neg = v->s0n, .src1_abs = v->s1a,
         .src1_neg = v->s1n, .addend_abs = v->ada, .addend_neg = v->adn,
         .saturate = v->sat,
      };
      uint8_t out[AGX3_FALU_MOD_MAX_LENGTH];
      uint8_t len = 0;
      bool ok;
      if (!strcmp(v->mnem, "f32add"))
         ok = agx3_falu_add_mod(e, v->dest, v->s0, v->s1, &m, out, &len);
      else if (!strcmp(v->mnem, "f32mul"))
         ok = agx3_falu_mul_mod(e, v->dest, v->s0, v->s1, &m, out, &len);
      else
         ok = agx3_falu_fma_mod(e, v->dest, v->s0, v->s1, v->ad, &m, out, &len);

      if (!ok || len != v->nbytes || memcmp(out, v->bytes, v->nbytes) != 0) {
         if (fail < 10)
            fprintf(stderr, "FAIL %s enc=%d\n", v->mnem, v->enc);
         fail++;
      } else {
         pass++;
      }
   }
   if (fail) {
      fprintf(stderr, "agx3_falu_mod_core: %u of %zu failed\n", fail,
              mod_vec_count);
      return 1;
   }
   printf("agx3_falu_mod_core: all %u modifier forms reproduced\n", pass);
   return 0;
}

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Every AGX3 control-flow encoder must reproduce Apple's witnessed bytes, and
 * the branch fixup must reconstruct real displaced branches from a reserved
 * template. Replays composer-generated vectors covering the full condition,
 * slot, push/pop and displacement domains.
 */

#include "agx3_cf_core.h"
#include "agx3_binder.h"
#include "agx3_cf_core_vectors.c"

#include "util/macros.h"

#include <stdio.h>
#include <string.h>

static unsigned checks, failures;

/* Compare a byte run against a lowercase hex literal. */
static bool
hex_is(const uint8_t *bytes, uint8_t len, const char *hex)
{
   if (strlen(hex) != (size_t)len * 2)
      return false;
   for (unsigned i = 0; i < len; ++i) {
      unsigned value;
      if (sscanf(hex + 2 * i, "%2x", &value) != 1 || bytes[i] != value)
         return false;
   }
   return true;
}

#define REJECT(expr)                                                           \
   do {                                                                        \
      checks++;                                                                \
      if ((expr)) {                                                            \
         fprintf(stderr, "FAIL accepted ungrounded form: %s\n", #expr);        \
         failures++;                                                           \
      }                                                                        \
   } while (0)

static void
expect(const struct cf_vec *v, const uint8_t *got, uint8_t len, bool ok)
{
   checks++;
   if (!ok || len != v->nbytes || memcmp(got, v->bytes, v->nbytes) != 0) {
      if (failures < 10)
         fprintf(stderr, "FAIL %s enc=%d args=%d,%d,%d,%d\n", v->kind, v->enc,
                 v->a0, v->a1, v->a2, v->a3);
      failures++;
   }
}

static void
test_half_comparisons(void)
{
   const struct {
      uint16_t src[2];
      struct agx3_cf_source_hints hints[2];
      enum agx3_cf_icond cond;
      uint8_t len, bytes[10];
   } native[] = {
      {{2,3},{{false,true},{false,true}},AGX3_CF_ICOND_SLT,10,
       {0x0a,0x02,0x3b,0x02,0x02,0x00,0x17,0xc8,0x02,0x00}},
      {{2,3},{{false,true},{false,true}},AGX3_CF_ICOND_ULT,6,
       {0x0a,0x02,0x3a,0x02,0x15,0xc0}},
      {{2,3},{{false,false},{false,false}},AGX3_CF_ICOND_SLT,10,
       {0x0a,0x02,0x23,0x02,0x02,0x00,0x17,0xc8,0x02,0x00}},
      {{3,2},{{false,true},{false,false}},AGX3_CF_ICOND_SLT,10,
       {0x0a,0x02,0x2b,0x02,0x02,0x00,0x0f,0xc8,0x02,0x00}},
   };
   for(unsigned i=0;i<sizeof(native)/sizeof(native[0]);++i) {
      uint8_t out[AGX3_CF_MAX_LENGTH],len=0;
      bool ok=agx3_cf_icmp_reg16(AGX_ISA_ENCODING_AGX3B,native[i].cond,0,
                               native[i].src,native[i].hints,32,out,&len);
      ++checks;
      if(!ok || len!=native[i].len || memcmp(out,native[i].bytes,len)) {
         fprintf(stderr,"half compare %u failed len%u: ",i,len);
         for(unsigned j=0;j<len;++j)fprintf(stderr,"%02x",out[j]);
         fprintf(stderr,"\n"); ++failures;
      }
   }
   for(unsigned bad=0;bad<6;++bad) {
      uint16_t src[]={18,1};
      struct agx3_cf_source_hints hints[2]={{false,false},{false,false}};
      uint8_t out[AGX3_CF_MAX_LENGTH],len=0xa5;
      memset(out,0xa5,sizeof(out));
      unsigned slot=0,wait=0;
      enum agx3_cf_icond cond=AGX3_CF_ICOND_SLT;
      if(bad==0)src[0]=256;
      if(bad==1)src[1]=256;
      if(bad==2)hints[0].cache=hints[0].discard=true;
      if(bad==3)slot=7;
      if(bad==4)wait=3;
      if(bad==5)cond=3;
      REJECT(agx3_cf_icmp_reg16(AGX_ISA_ENCODING_AGX3B,cond,slot,src,hints,wait,out,&len));
      ++checks;
      if(len!=0xa5)++failures;
      for(unsigned j=0;j<sizeof(out);++j)if(out[j]!=0xa5)++failures;
   }
}

/* G17G float branches against Apple's applegpu_g17g bytes
 * (test/native/g17g/m5-compare-select-reference.metal cmp_fimm_*,
 * cmp_freg16): a compared register not read again, slot 0. */
static void
test_g17g_float_compares(void)
{
   const struct agx3_cf_source_hints dead = {false, true};
   const struct {
      const char *name;
      enum agx3_cf_fcond cond;
      bool or_nan, half;
      uint8_t minifloat, len, bytes[10];
   } imm[] = {
      {"x < 0.5", AGX3_CF_FCOND_LT, false, false, 32, 6,
       {0x0a, 0x01, 0x2a, 0xa1, 0x03, 0x02}},
      {"x > 1.0", AGX3_CF_FCOND_GT, false, false, 48, 6,
       {0x0a, 0x01, 0x2a, 0xb1, 0x02, 0x02}},
      {"x < -1.0", AGX3_CF_FCOND_LT, false, false, 176, 6,
       {0x0a, 0x01, 0x3a, 0xb1, 0x03, 0x02}},
      {"half x < 1.0h", AGX3_CF_FCOND_LT, false, true, 48, 6,
       {0x0a, 0x00, 0x2a, 0xb1, 0x03, 0x02}},
      {"x >= 0.0 as lt unordered", AGX3_CF_FCOND_LT, true, false, 0, 10,
       {0x1a, 0x01, 0x2b, 0x81, 0x06, 0x00, 0x02, 0x02, 0x00, 0x00}},
      {"x <= 2.0 as gt unordered", AGX3_CF_FCOND_GT, true, false, 64, 10,
       {0x1a, 0x01, 0x2b, 0xc1, 0x06, 0x00, 0x03, 0x02, 0x00, 0x00}},
      {"x == 0.0", AGX3_CF_FCOND_EQ, false, false, 0, 10,
       {0x0a, 0x01, 0x2b, 0x81, 0x06, 0x00, 0x00, 0x02, 0x00, 0x00}},
   };
   for (unsigned i = 0; i < sizeof(imm) / sizeof(imm[0]); ++i) {
      uint8_t out[AGX3_CF_MAX_LENGTH], len = 0;
      const bool ok = agx3_cf_fcmp_zero_imm(
         AGX_ISA_ENCODING_AGX3C, imm[i].cond, imm[i].or_nan, 0, 0,
         imm[i].half, imm[i].minifloat, dead, 0, out, &len);
      ++checks;
      if (!ok || len != imm[i].len || memcmp(out, imm[i].bytes, len)) {
         fprintf(stderr, "G17G fcmp %s failed\n", imm[i].name);
         ++failures;
      }
   }

   const uint16_t src[] = {0, 1};
   const struct agx3_cf_source_hints hints[] = {dead, dead};
   static const uint8_t half_lt[] = {0x0a, 0x00, 0x3a, 0x00, 0x13, 0x00};
   uint8_t out[AGX3_CF_MAX_LENGTH], len = 0;
   const bool ok = agx3_cf_fcmp_reg16(AGX_ISA_ENCODING_AGX3C,
                                      AGX3_CF_FCOND_LT, false, 0, src, hints,
                                      0, out, &len);
   ++checks;
   if (!ok || len != sizeof(half_lt) || memcmp(out, half_lt, len)) {
      fprintf(stderr, "G17G half fcmp failed\n");
      ++failures;
   }

   /* A ninth wait slot and an unordered equality stay closed. */
   REJECT(agx3_cf_fcmp_zero_imm(AGX_ISA_ENCODING_AGX3C, AGX3_CF_FCOND_LT,
                                false, 0, 0, false, 32, dead, 0x100, out,
                                &len));
   REJECT(agx3_cf_fcmp_zero_imm(AGX_ISA_ENCODING_AGX3C, AGX3_CF_FCOND_EQ,
                                true, 0, 0, false, 32, dead, 0, out, &len));
}

static void
test_half_zero_lifetime(void)
{
   const uint8_t native[3][10] = {
      {0x0a,0x02,0x2b,0x80,0x02,0x00,0x07,0xc2,0x02,0x00},
      {0x0a,0x02,0x23,0x80,0x02,0x00,0x07,0xc2,0x02,0x00},
      {0x0a,0x02,0x23,0x80,0x06,0x00,0x07,0xc2,0x02,0x00},
   };
   for(unsigned i=0;i<3;++i) {
      uint8_t out[AGX3_CF_MAX_LENGTH],len=0;
      const struct agx3_cf_source_hints hint={false,i==0};
      bool ok=agx3_cf_icmp_zero16_runtime(AGX_ISA_ENCODING_AGX3B,
                 i==2 ? AGX3_CF_ICOND_EQ : AGX3_CF_ICOND_SLT,0,2,hint,32,out,&len);
      ++checks;
      if(!ok || len!=10 || memcmp(out,native[i],10))++failures;
   }
}

/*
 * Apple's own bytes for a 16-bit compare against a non-zero constant, from
 * .tmp/g17p-hyprland-cf-evidence/icmp16imm.metal via
 * `metal-nt -arch applegpu_g17p`. Every kernel differs from the zero control
 * in one place only -- the constant -- and the recovered field is the same
 * f16 the 32-bit source form already uses.
 */
static void
test_half_immediate_compare(void)
{
   const struct {
      unsigned cond;
      uint8_t imm;
      uint8_t bytes[10];
   } v[] = {
      /* short x < imm */
      {AGX3_CF_ICOND_SLT, 0,
       {0x0a, 0x04, 0x23, 0x80, 0x02, 0x00, 0x07, 0xc2, 0x02, 0x00}},
      {AGX3_CF_ICOND_SLT, 1,
       {0x0a, 0x04, 0x23, 0x80, 0x02, 0x00, 0x17, 0xc2, 0x02, 0x00}},
      {AGX3_CF_ICOND_SLT, 2,
       {0x0a, 0x04, 0x23, 0x82, 0x02, 0x00, 0x07, 0xc2, 0x02, 0x00}},
      {AGX3_CF_ICOND_SLT, 3,
       {0x0a, 0x04, 0x23, 0x82, 0x02, 0x00, 0x17, 0xc2, 0x02, 0x00}},
      {AGX3_CF_ICOND_SLT, 4,
       {0x0a, 0x04, 0x23, 0x84, 0x02, 0x00, 0x07, 0xc2, 0x02, 0x00}},
      {AGX3_CF_ICOND_SLT, 17,
       {0x0a, 0x04, 0x23, 0x90, 0x02, 0x00, 0x17, 0xc2, 0x02, 0x00}},
      {AGX3_CF_ICOND_SLT, 127,
       {0x0a, 0x04, 0x23, 0xfe, 0x02, 0x00, 0x17, 0xc2, 0x02, 0x00}},
      {AGX3_CF_ICOND_SLT, 255,
       {0x0a, 0x04, 0x33, 0xfe, 0x02, 0x00, 0x17, 0xc2, 0x02, 0x00}},
      /* short x > 5, short x == 7, ushort x == 2 */
      {AGX3_CF_ICOND_SGT, 5,
       {0x0a, 0x04, 0x23, 0x84, 0x02, 0x00, 0x16, 0xc2, 0x02, 0x00}},
      {AGX3_CF_ICOND_EQ, 7,
       {0x0a, 0x04, 0x23, 0x86, 0x06, 0x00, 0x17, 0xc2, 0x02, 0x00}},
      {AGX3_CF_ICOND_EQ, 2,
       {0x0a, 0x04, 0x23, 0x82, 0x06, 0x00, 0x07, 0xc2, 0x02, 0x00}},
   };

   for (unsigned i = 0; i < sizeof(v) / sizeof(v[0]); i++) {
      uint8_t bytes[AGX3_CF_MAX_LENGTH], length = 0;
      bool ok = agx3_cf_icmp_imm16_runtime(
         AGX_ISA_ENCODING_AGX3B, v[i].cond, 0, 4, v[i].imm,
         (struct agx3_cf_source_hints){false, false}, 32, bytes, &length);
      checks++;
      if (!ok || length != 10 || memcmp(bytes, v[i].bytes, 10)) {
         fprintf(stderr, "FAIL native half-immediate branch vector%u\n", i);
         failures++;
      }
   }

   /* Independent AGX3c table vectors: a live odd half tested against zero,
    * including the signed predicate used by border-sampler selection. */
   const enum agx3_cf_icond m5_cond[] = {
      AGX3_CF_ICOND_EQ, AGX3_CF_ICOND_SLT, AGX3_CF_ICOND_SGT,
   };
   const uint8_t m5_bytes[][10] = {
      {0x0a,0x04,0x23,0x80,0x06,0x00,0x0f,0x02,0x02,0x00},
      {0x0a,0x04,0x23,0x80,0x02,0x00,0x0f,0x02,0x02,0x00},
      {0x0a,0x04,0x23,0x80,0x02,0x00,0x0e,0x02,0x02,0x00},
   };
   for (unsigned i=0;i<3;i++) {
      uint8_t bytes[AGX3_CF_MAX_LENGTH], length=0;
      checks++;
      if (!agx3_cf_icmp_imm16_runtime_g18g(AGX_ISA_ENCODING_AGX3C,m5_cond[i],
             0,5,0,(struct agx3_cf_source_hints){false,false},0,bytes,&length) ||
          length != 10 || memcmp(bytes,m5_bytes[i],10))
         failures++;
   }

   /* Fresh witnesses from
    * .tmp/cts-pipeline/witness/cf-uimm16-20260914/uimm16.metal. The dead and
    * live `ushort x < 4` kernels produce the same compact compare, while the
    * immediate and relation fields vary independently. */
   {
      uint8_t bytes[AGX3_CF_MAX_LENGTH], length = 0;
      const struct agx3_cf_source_hints hints[] = {
         {false, true}, {true, false},
      };
      const struct {
         enum agx3_cf_icond cond;
         uint8_t imm;
         const char *hex;
      } compact[] = {
         {AGX3_CF_ICOND_ULT, 4, "0a04228405c2"},
         {AGX3_CF_ICOND_ULT, 32, "0a0422a005c2"},
         {AGX3_CF_ICOND_UGT, 4, "0a04228404c2"},
      };
      for (unsigned i = 0; i < ARRAY_SIZE(compact); ++i) {
         for (unsigned h = 0; h < ARRAY_SIZE(hints); ++h) {
            const bool ok = agx3_cf_icmp_imm16_runtime(
               AGX_ISA_ENCODING_AGX3B, compact[i].cond, 0, 4,
               compact[i].imm, hints[h], 32, bytes, &length);
            checks++;
            if (!ok || !hex_is(bytes, length, compact[i].hex))
               failures++;
         }
      }
      /* Zero keeps working for every relation: that is the existing form. */
      checks++;
      if (!agx3_cf_icmp_imm16_runtime(AGX_ISA_ENCODING_AGX3B,
                                      AGX3_CF_ICOND_ULT, 0, 4, 0, hints[0], 32,
                                      bytes, &length))
         failures++;
   }
}

static void
test_float_zero_compare(void)
{
   /* Complete native branches preserve X for their body or discard it when
    * it is used only by the condition. GE/LE additionally exercise or-NaN. */
   const struct {unsigned cond;bool cache,or_nan;unsigned length;uint8_t bytes[10];} v[]={
      {0,true,false,10,{0x0a,0x83,0x23,0x81,0x06,0x00,0x00,0xc2,0x00,0x00}},
      {1,true,false,6,{0x0a,0x83,0x22,0x81,0x03,0xc2}},
      {2,true,false,6,{0x0a,0x83,0x22,0x81,0x02,0xc2}},
      {0,false,false,10,{0x0a,0x03,0x2b,0x81,0x06,0x00,0x00,0xc2,0x00,0x00}},
      {1,false,false,6,{0x0a,0x03,0x2a,0x81,0x03,0xc2}},
      {2,false,false,6,{0x0a,0x03,0x2a,0x81,0x02,0xc2}},
      {1,true,true,10,{0x1a,0x83,0x23,0x81,0x06,0x00,0x02,0xc2,0x00,0x00}},
      {2,true,true,10,{0x1a,0x83,0x23,0x81,0x06,0x00,0x03,0xc2,0x00,0x00}},
   };
   for(unsigned i=0;i<sizeof(v)/sizeof(v[0]);i++) {
      uint8_t bytes[AGX3_CF_MAX_LENGTH],length=0;
      bool ok=agx3_cf_fcmp_zero(AGX_ISA_ENCODING_AGX3B,v[i].cond,v[i].or_nan,0,2,false,
         (struct agx3_cf_source_hints){v[i].cache,!v[i].cache},32,bytes,&length);
      checks++;
      if(!ok || length!=v[i].length || memcmp(bytes,v[i].bytes,length)) {
         fprintf(stderr,"FAIL native float-zero branch vector%u\n",i);failures++;
      }
   }
}

/* metal-nt -arch applegpu_g17p: `if (x < y)` and friends over two loaded
 * halves, each operand either read again after the branch or not. */
static void
test_half_register_compare(void)
{
   static const struct {
      enum agx3_cf_fcond cond;
      bool or_nan;
      uint16_t src[2];
      bool live[2];
      const char *bytes;
   } rows[] = {
      {AGX3_CF_FCOND_LT, false, {4, 2}, {false, false}, "0a043a0203c0"},
      {AGX3_CF_FCOND_LT, false, {4, 2}, {true, false},  "0a84320203c0"},
      {AGX3_CF_FCOND_LT, false, {5, 4}, {false, true},  "0a042a840bc0"},
      {AGX3_CF_FCOND_LT, false, {5, 4}, {true, true},   "0a8422840bc0"},
      {AGX3_CF_FCOND_GT, false, {4, 2}, {false, false}, "0a043a0202c0"},
      {AGX3_CF_FCOND_GT, false, {5, 4}, {true, true},   "0a8422840ac0"},
      {AGX3_CF_FCOND_EQ, false, {4, 2}, {false, false}, "0a043b02060000c00000"},
      {AGX3_CF_FCOND_EQ, false, {5, 4}, {true, true},   "0a842384060008c00000"},
      /* x >= y: lt or unordered, inverted by the branch */
      {AGX3_CF_FCOND_LT, true,  {4, 2}, {false, false}, "1a043b02060002c00000"},
      {AGX3_CF_FCOND_LT, true,  {5, 4}, {false, true},  "1a042b8406000ac00000"},
      /* x <= y */
      {AGX3_CF_FCOND_GT, true,  {4, 2}, {true, false},  "1a843302060003c00000"},
      {AGX3_CF_FCOND_GT, true,  {5, 4}, {true, true},   "1a84238406000bc00000"},
   };

   for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
      const struct agx3_cf_source_hints hints[] = {
         {rows[i].live[0], !rows[i].live[0]},
         {rows[i].live[1], !rows[i].live[1]}};
      uint8_t out[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      /* Each operand came from a load on scoreboard slot 6. */
      const bool ok = agx3_cf_fcmp_reg16(AGX_ISA_ENCODING_AGX3B, rows[i].cond,
                                         rows[i].or_nan, 0, rows[i].src, hints,
                                         1u << 5, out, &len);
      checks++;
      if (!ok || !hex_is(out, len, rows[i].bytes)) {
         fprintf(stderr, "FAIL native half register branch %u\n", i);
         failures++;
      }
   }

   /* The same f28/f30 layout on the f32 form: y live, then y dead. */
   static const struct {
      bool live1;
      uint16_t src[2];
      const char *bytes;
   } f32_rows[] = {
      {true,  {4, 2}, "0a052a8303c0"},
      {false, {2, 4}, "0a033a0503c0"},
   };
   for (unsigned i = 0; i < 2; ++i) {
      const struct agx3_field_kv base[] = {
         {9, 2}, {10, 2}, {26, 2}, {27, 2}};
      const struct agx3_field_kv ov[] = {
         {2, f32_rows[i].src[0]}, {13, 1}, {19, f32_rows[i].src[1]},
         {28, f32_rows[i].live1}, {30, !f32_rows[i].live1}, {43, 1},
         {46, AGX3_CF_FCOND_LT}};
      uint8_t out[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      const bool ok = agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, "fcmp", 0,
                                        base, 4, ov, 7, out, &len);
      checks++;
      if (!ok || !hex_is(out, len, f32_rows[i].bytes)) {
         fprintf(stderr, "FAIL f32 second-source lifetime %u\n", i);
         failures++;
      }
   }
}

static void
test_float_negated_register_compare(void)
{
   const uint16_t src[] = {0, 2};
   const bool left[] = {true, false};
   const bool right[] = {false, true};
   /* The second source's cache is f28 and its discard f30. Both specimens
    * keep the compare's left operand live and let its right operand die; the
    * negated-left one swaps the operands, and their lifetimes, first. */
   const struct agx3_cf_source_hints discard[] = {
      {true,false},{false,true}};
   const struct agx3_cf_source_hints cache[] = {
      {true,false},{false,true}};
   const uint8_t native_left[] = {0x0a,0x03,0x2a,0x81,0x02,0x28};
   const uint8_t native_right[] = {0x0a,0x81,0x32,0x03,0x03,0x28};
   uint8_t out[AGX3_CF_MAX_LENGTH];
   uint8_t len = 0;
   bool ok = agx3_cf_fcmp_reg_neg(AGX_ISA_ENCODING_AGX3B,
      AGX3_CF_FCOND_LT, false, 0, src, left, discard, out, &len);
   checks++;
   if (!ok || len != sizeof(native_left) || memcmp(out,native_left,len))
      failures++;
   ok = agx3_cf_fcmp_reg_neg(AGX_ISA_ENCODING_AGX3B,
      AGX3_CF_FCOND_LT, false, 0, src, right, cache, out, &len);
   checks++;
   if (!ok || len != sizeof(native_right) || memcmp(out,native_right,len))
      failures++;
}

/*
 * An ABSOLUTE source on the control-flow float compare.
 *
 * ABS is not a bit on the six-byte form: Apple moves to a fourteen-byte
 * encoding point, ep0 -> ep2, and f14 and f31 are the two sources' bits.
 * Every other field keeps its number, so the long form differs from the short
 * one by the endpoint and the two bits alone.
 *
 * metal-nt -arch applegpu_g17p, one kernel per row. The branch body is a
 * device atomic: Metal if-converts a plain conditional assignment into an
 * `fsel` and then the experiment answers a different question entirely.
 */
static void
test_float_absolute_register_compare(void)
{
   static const struct {
      const char *what;
      bool abs0, abs1;
      enum agx3_cf_fcond cond;
      const char *bytes;
   } rows[] = {
      {"fabs(a) <  b",       true,  false, AGX3_CF_FCOND_LT,
       "0a013b0303000300000000800100"},
      {"a < fabs(b)",        false, true,  AGX3_CF_FCOND_LT,
       "0a013b0303000300000000800200"},
      {"fabs(a) < fabs(b)",  true,  true,  AGX3_CF_FCOND_LT,
       "0a013b0303000300000000800300"},
      {"fabs(a) > fabs(b)",  true,  true,  AGX3_CF_FCOND_GT,
       "0a013b0303000200000000800300"},
      {"fabs(a) == fabs(b)", true,  true,  AGX3_CF_FCOND_EQ,
       "0a013b0307000000000000800300"},
   };

   const uint16_t src[] = {0, 2};
   const struct agx3_cf_source_hints hints[] = {{false, true}, {false, true}};
   for (unsigned i = 0; i < sizeof(rows)/sizeof(rows[0]); ++i) {
      const bool abs[] = {rows[i].abs0, rows[i].abs1};
      uint8_t out[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      const bool ok = agx3_cf_fcmp_reg_abs(AGX_ISA_ENCODING_AGX3B,
         rows[i].cond, false, 0, src, abs, hints, 1u << 5, out, &len);
      checks++;
      if (!ok || len != 14 || !hex_is(out, len, rows[i].bytes)) {
         fprintf(stderr, "FAIL native abs float branch %s\n", rows[i].what);
         failures++;
      }
   }

   /* Without an absolute source this is the short endpoint; emitting the long
    * form for it would change bytes that already work. */
   uint8_t out[AGX3_CF_MAX_LENGTH];
   uint8_t len = 0;
   const bool none[] = {false, false};
   const bool both[] = {true, true};
   checks++;
   if (agx3_cf_fcmp_reg_abs(AGX_ISA_ENCODING_AGX3B, AGX3_CF_FCOND_LT, false,
                            0, src, none, hints, 0, out, &len))
      failures++;
   /* or_nan, AGX3c and an odd half all stay refused. */
   checks++;
   if (agx3_cf_fcmp_reg_abs(AGX_ISA_ENCODING_AGX3B, AGX3_CF_FCOND_LT, true,
                            0, src, both, hints, 0, out, &len))
      failures++;
   checks++;
   if (agx3_cf_fcmp_reg_abs(AGX_ISA_ENCODING_AGX3C, AGX3_CF_FCOND_LT, false,
                            0, src, both, hints, 0, out, &len))
      failures++;
   const uint16_t odd[] = {1, 2};
   checks++;
   if (agx3_cf_fcmp_reg_abs(AGX_ISA_ENCODING_AGX3B, AGX3_CF_FCOND_LT, false,
                            0, odd, both, hints, 0, out, &len))
      failures++;
}

static void
test_half_float_zero_compare(void)
{
   const uint8_t gold[2][3][10]={
      {{0x0a,0x82,0x23,0x81,0x06,0,0,0xc2,0,0},
       {0x0a,0x82,0x22,0x81,0x03,0xc2},
       {0x0a,0x82,0x22,0x81,0x02,0xc2}},
      {{0x0a,0x82,0x23,0x81,0x06,0,0x08,0xc2,0,0},
       {0x0a,0x82,0x22,0x81,0x0b,0xc2},
       {0x0a,0x82,0x22,0x81,0x0a,0xc2}},
   };
   for(unsigned odd=0;odd<2;odd++)for(unsigned condition=0;condition<3;condition++) {
      uint8_t bytes[AGX3_CF_MAX_LENGTH],length=0;
      bool ok=agx3_cf_fcmp_zero(AGX_ISA_ENCODING_AGX3B,condition,false,0,2+odd,true,
         (struct agx3_cf_source_hints){true,false},32,bytes,&length);
      checks++;
      if(!ok || length!=(condition?6:10) || memcmp(bytes,gold[odd][condition],length)) {
         fprintf(stderr,"FAIL native half float-zero branch odd%u condition%u\n",odd,condition);failures++;
      }
   }
   {
      const uint8_t gold[] = {
         0x0a, 0x08, 0x23, 0x81, 0x06, 0x00, 0x08, 0x02, 0x00, 0x00,
      };
      uint8_t bytes[AGX3_CF_MAX_LENGTH], length = 0;
      const bool ok = agx3_cf_fcmp_zero_imm(
         AGX_ISA_ENCODING_AGX3B, AGX3_CF_FCOND_EQ, false, 0, 9, true, 0,
         (struct agx3_cf_source_hints){false, false}, 0, bytes, &length);
      checks++;
      if (!ok || length != sizeof(gold) || memcmp(bytes, gold, length)) {
         fprintf(stderr, "FAIL native retained half float-zero branch\n");
         failures++;
      }
   }
}

/*
 * Branching on a float against a NON-ZERO constant.
 *
 * f18 is the compare's second operand and was pinned to zero, so only
 * `x < 0.0` encoded. `x < 1.0` is ordinary GLSL and was 136 of the 146
 * instructions still refused in the Vulkan compute block after the
 * threadgroup fix. Exactly the defect the integer side already carried:
 * agx3_cf_icmp_zero16_runtime() never set f16.
 *
 * Apple, one kernel per variable. The branch body has to be a loop: a
 * trivial if/else is if-converted to a select and emits no fcmp at all,
 * which is why several hand-written probes for this shape found nothing.
 */
static void
test_float_minifloat_compare(void)
{
   uint8_t out[AGX3_CF_MAX_LENGTH];
   uint8_t len;
   const struct agx3_cf_source_hints dead = {false, true};

   struct {
      const char *name;
      enum agx3_cf_fcond cond;
      uint8_t imm;
      bool src16;
      uint8_t bytes[6];
   } v[] = {
      {"v < 0.0f", AGX3_CF_FCOND_LT,  0, false, {0x0a,0x03,0x2a,0x81,0x03,0xc2}},
      {"v < 0.5f", AGX3_CF_FCOND_LT, 32, false, {0x0a,0x03,0x2a,0xa1,0x03,0xc2}},
      {"v < 1.0f", AGX3_CF_FCOND_LT, 48, false, {0x0a,0x03,0x2a,0xb1,0x03,0xc2}},
      {"v < 2.0f", AGX3_CF_FCOND_LT, 64, false, {0x0a,0x03,0x2a,0xc1,0x03,0xc2}},
      {"v > 1.0f", AGX3_CF_FCOND_GT, 48, false, {0x0a,0x03,0x2a,0xb1,0x02,0xc2}},
      {"half v < 1.0h", AGX3_CF_FCOND_LT, 48, true,
       {0x0a,0x02,0x2a,0xb1,0x03,0xc2}},
   };

   for (unsigned i = 0; i < sizeof(v) / sizeof(v[0]); ++i) {
      len = 0;
      const bool ok = agx3_cf_fcmp_zero_imm(AGX_ISA_ENCODING_AGX3B, v[i].cond,
                                            false, 0, 2, v[i].src16, v[i].imm,
                                            dead, 32, out, &len);
      checks++;
      if (!ok || len != 6 || memcmp(out, v[i].bytes, 6)) {
         fprintf(stderr, "FAIL float-minifloat branch: %s\n", v[i].name);
         failures++;
      }
   }

   /* The zero wrapper must still emit exactly the bytes it always did. */
   len = 0;
   const bool zero_ok =
      agx3_cf_fcmp_zero(AGX_ISA_ENCODING_AGX3B, AGX3_CF_FCOND_LT, false, 0, 2,
                        false, dead, 32, out, &len);
   checks++;
   if (!zero_ok || len != 6 || memcmp(out, v[0].bytes, 6)) {
      fprintf(stderr, "FAIL float-minifloat: zero wrapper changed bytes\n");
      failures++;
   }
}

/*
 * A loop break that also escapes enclosing ifs. Apple's own compiler spells
 * it CNDCF with a level count, where CNDLT is the compact count-2 form.
 *
 * Every row below is Apple's byte-exact `metal-nt -arch applegpu_g17p` output
 * for MSL that breaks out of a `for` loop from inside N nested ifs, each
 * condition reading a distinct element of a `device const uint *` so nothing
 * can be folded, and each level storing before the next opens
 * (.tmp/g17p-deep-break-evidence/deepbreak.metal, probe2.metal).
 */
static void
test_deep_break_levels(void)
{
   const struct {
      unsigned slot, count;
      bool invert;
      const char *hex;
      const char *witness;
   } native[] = {
      {1, 3, false, "8f0554030001", "break inside 2 nested ifs"},
      {1, 4, false, "8f0554040001", "break inside 3 nested ifs"},
      {1, 5, false, "8f0554050001", "break inside 4 nested ifs"},
      {1, 6, false, "8f0554060001", "break inside 5 nested ifs"},
      {1, 7, false, "8f0554070001", "break inside 6 nested ifs"},
      {1, 8, false, "8f0554080001", "break inside 7 nested ifs"},
      {1, 3, true, "8f0554030009", "break from an else arm (invert)"},
      {2, 3, false, "8f0554030002", "break out of an INNER loop"},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(native); ++i) {
      uint8_t out[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      checks++;
      if (!agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3B, native[i].slot,
                         native[i].invert, native[i].count, out, &len) ||
          !hex_is(out, len, native[i].hex)) {
         fprintf(stderr, "FAIL cndcf slot=%u invert=%u count=%u (%s)\n",
                 native[i].slot, (unsigned)native[i].invert, native[i].count,
                 native[i].witness);
         failures++;
      }
   }

   /* A one-level exit is not a loop break: a loop owns two levels, so the
    * shallowest break is count 2 and that is CNDLT's encoding point. Flag
    * slots stay inside the witnessed 0..5 range. */
   {
      uint8_t out[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      REJECT(agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3B, 1, false, 0, out, &len));
      REJECT(agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3B, 1, false, 1, out, &len));
      REJECT(agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3B, 6, false, 3, out, &len));
      REJECT(agx3_cf_cndcf(AGX_ISA_ENCODING_AGX2, 1, false, 3, out, &len));
   }

   /* The AGX3c constants are structural (cndlt/cndst siblings); assert only
    * that the operation exists and that its count reaches the same depths,
    * since no AGX3c capture covers it. */
   for (unsigned count = 3; count <= 8; ++count) {
      uint8_t out[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      checks++;
      if (!agx3_cf_cndcf(AGX_ISA_ENCODING_AGX3C, 1, false, count, out, &len)) {
         fprintf(stderr, "FAIL agx3c cndcf count=%u\n", count);
         failures++;
      }
   }
}

/* v52b: Apple's G17P loop-test compare of a register against a uniform. */
static void
test_uniform_compare(void)
{
   uint8_t out[AGX3_CF_MAX_LENGTH], len = 0;
   ++checks;
   if (!agx3_cf_icmp_reg_uniform32(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ,
                                   0, 2, 6,
                                   (struct agx3_cf_source_hints){true, false},
                                   out, &len) ||
       !hex_is(out, len, "0a832307060007020000")) {
      fprintf(stderr, "uniform compare (cached counter) mismatch\n");
      ++failures;
   }
   ++checks;
   if (!agx3_cf_icmp_reg_uniform32(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ,
                                   0, 2, 10,
                                   (struct agx3_cf_source_hints){false, false},
                                   out, &len) ||
       !hex_is(out, len, "0a03230b060007020000")) {
      fprintf(stderr, "uniform compare (uncached counter) mismatch\n");
      ++failures;
   }
   const struct agx3_cf_source_hints none = {false, false};
   REJECT(agx3_cf_icmp_reg_uniform32(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ,
                                     0, 3, 6, none, out, &len));
   REJECT(agx3_cf_icmp_reg_uniform32(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ,
                                     0, 2, 7, none, out, &len));
   REJECT(agx3_cf_icmp_reg_uniform32(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ,
                                     0, 2, 256, none, out, &len));
   REJECT(agx3_cf_icmp_reg_uniform32(AGX_ISA_ENCODING_AGX3B,
                                     (enum agx3_cf_icond)3, 0, 2, 6, none,
                                     out, &len));
   REJECT(agx3_cf_icmp_reg_uniform32(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ,
                                     6, 2, 6, none, out, &len));
   REJECT(agx3_cf_icmp_reg_uniform32(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ,
                                     0, 2, 6,
                                     (struct agx3_cf_source_hints){true, true},
                                     out, &len));
}


/* frg-codegen: Apple's G17P loop guard compares the trip-count uniform with
 * an immediate directly, without copying the uniform into a GPR first. */
static void
test_uniform_imm_compare(void)
{
   uint8_t out[AGX3_CF_MAX_LENGTH], len = 0;
   ++checks;
   if (!agx3_cf_icmp_uniform_imm(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ, 0,
                                 6, 0, out, &len) ||
       !hex_is(out, len, "0a072380060087020000")) {
      fprintf(stderr, "uniform/immediate compare (loop guard) mismatch\n");
      ++failures;
   }
   /* The register form of the same compare differs only in the operand
    * class and index: f9 = 2 / f2 = register instead of f9 = 3 / f3. */
   uint8_t reg[AGX3_CF_MAX_LENGTH], reg_len = 0;
   ++checks;
   if (!agx3_cf_icmp_imm(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ, 0, 4, 0,
                         reg, &reg_len) ||
       !hex_is(reg, reg_len, "0a052380060007020000")) {
      fprintf(stderr, "register/immediate compare control mismatch\n");
      ++failures;
   }
   /* Other uniforms, relations, immediates and slots compose (the packer
    * relies on the whole even-index domain below 256). */
   static const uint16_t uniforms[] = {0, 2, 18, 64, 128, 254};
   static const enum agx3_cf_icond conds[] = {
      AGX3_CF_ICOND_EQ, AGX3_CF_ICOND_ULT, AGX3_CF_ICOND_UGT,
      AGX3_CF_ICOND_SLT, AGX3_CF_ICOND_SGT};
   for (unsigned u = 0; u < ARRAY_SIZE(uniforms); ++u) {
      for (unsigned c = 0; c < ARRAY_SIZE(conds); ++c) {
         ++checks;
         uint8_t a[AGX3_CF_MAX_LENGTH], b[AGX3_CF_MAX_LENGTH], la = 0, lb = 0;
         if (!agx3_cf_icmp_uniform_imm(AGX_ISA_ENCODING_AGX3B, conds[c], 1,
                                       uniforms[u], 7, a, &la) ||
             !agx3_cf_icmp_uniform_imm(AGX_ISA_ENCODING_AGX3B, conds[c], 1,
                                       uniforms[u] == 254 ? 0 : uniforms[u] + 2,
                                       7, b, &lb) ||
             la != 10 || lb != 10 || !memcmp(a, b, la)) {
            fprintf(stderr, "uniform/immediate compare u%u cond %u\n",
                    uniforms[u], conds[c]);
            ++failures;
         }
      }
   }
   REJECT(agx3_cf_icmp_uniform_imm(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ,
                                   0, 7, 0, out, &len));
   REJECT(agx3_cf_icmp_uniform_imm(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ,
                                   0, 256, 0, out, &len));
   REJECT(agx3_cf_icmp_uniform_imm(AGX_ISA_ENCODING_AGX3B,
                                   (enum agx3_cf_icond)3, 0, 6, 0, out, &len));
   REJECT(agx3_cf_icmp_uniform_imm(AGX_ISA_ENCODING_AGX3B, AGX3_CF_ICOND_EQ,
                                   6, 6, 0, out, &len));
}


int
main(void)
{
   test_uniform_compare();
   test_uniform_imm_compare();
   test_float_minifloat_compare();
   test_float_zero_compare();
   test_half_register_compare();
   test_float_negated_register_compare();
   test_float_absolute_register_compare();
   test_half_float_zero_compare();
   test_half_comparisons();
   test_g17g_float_compares();
   test_half_zero_lifetime();
   test_half_immediate_compare();
   test_deep_break_levels();
   for (size_t i = 0; i < cf_vec_count; ++i) {
      const struct cf_vec *v = &cf_vecs[i];
      const enum agx_isa_encoding e =
         v->enc == 0 ? AGX_ISA_ENCODING_AGX3B : AGX_ISA_ENCODING_AGX3C;
      uint8_t out[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      bool ok = false;

      if (!strcmp(v->kind, "icmp_rr"))
         ok = agx3_cf_icmp_reg(e, v->a0, v->a1, v->a2, v->a3, out, &len);
      else if (!strcmp(v->kind, "icmp_imm"))
         ok =
            agx3_cf_icmp_imm(e, v->a0, v->a1, v->a2, (uint8_t)v->a3, out, &len);
      else if (!strcmp(v->kind, "fcmp_rr"))
         ok = agx3_cf_fcmp_reg(e, v->a0, v->a1, v->a2, v->a3, /*src1*/ 4, out,
                               &len);
      else if (!strcmp(v->kind, "cndst"))
         ok = agx3_cf_cndst(e, v->a0, v->a1, v->a2, out, &len);
      else if (!strcmp(v->kind, "cndlt"))
         ok = agx3_cf_cndlt(e, v->a0, v->a1, out, &len);
      else if (!strcmp(v->kind, "cndef"))
         ok = agx3_cf_cndef(e, out, &len);
      else if (!strcmp(v->kind, "cndend"))
         ok = agx3_cf_cndend(e, v->a0, out, &len);
      else if (!strcmp(v->kind, "branch"))
         ok = agx3_cf_branch(e, v->a0, out, &len);
      else if (!strcmp(v->kind, "branch_disp")) {
         ok = agx3_cf_branch(e, v->a0, out, &len) &&
              agx3_cf_patch_branch(e, v->a0, out, v->a1);
      }
      expect(v, out, len, ok);
   }

   /* The public codec is itself fail-closed, not just its production caller:
    * Apple's integer condition 3 remains unknown, float condition 3 is
    * represented only through LT+or_nan, and no other raw enum values are
    * admitted. */
   {
      uint8_t out[AGX3_CF_MAX_LENGTH];
      uint8_t len = 0;
      REJECT(agx3_cf_icmp_reg(AGX_ISA_ENCODING_AGX3B, (enum agx3_cf_icond)3, 0,
                              0, 2, out, &len));
      REJECT(agx3_cf_icmp_imm(AGX_ISA_ENCODING_AGX3C, (enum agx3_cf_icond)7, 0,
                              0, 1, out, &len));
      REJECT(agx3_cf_fcmp_reg(AGX_ISA_ENCODING_AGX3B, (enum agx3_cf_fcond)3,
                              false, 0, 0, 2, out, &len));
      REJECT(agx3_cf_fcmp_reg(AGX_ISA_ENCODING_AGX3C, (enum agx3_cf_fcond)7,
                              false, 0, 0, 2, out, &len));
   }

   if (failures) {
      fprintf(stderr, "agx3_cf_core: %u of %u checks failed\n", failures,
              checks);
      return 1;
   }
   printf("agx3_cf_core: all %u checks passed\n", checks);
   return 0;
}

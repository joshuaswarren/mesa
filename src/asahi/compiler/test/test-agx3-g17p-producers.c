/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Byte-exact replay of Apple's own G17P (A18 Pro, AGX3b) encodings for the
 * producers added in the 2026-08-31 writeout/colour-path session:
 * `fltnorm`, the perspective `f32mul.ep4`, the staging-file writers
 * (`copy`, `ldimm.ep2`, `clr`), the 16-bit shift-add `iadd`, the SR 2 read,
 * the whole coverage-word sequence, and the immediate-lane `simdbcast`.
 *
 * Every expected byte string below is a literal Apple instruction, from one of
 * two sources, both offline:
 *
 *  - the native `applegpu` slices of apple-g17p-isa.metalar, i.e. the shaders
 *    Apple's own compiler produced for this part (fmain, fdiscard, ftex,
 *    fdepth, vmain and its four pipeline variants, cbits, csimd, catomic);
 *  - `applegpu-nt -platform_version macos 26.0 26.0 -arch applegpu_g17p`,
 *    which reproduces all thirteen of those shaders byte-for-byte from source
 *    and therefore serves as a probeable oracle. The comments name each
 *    probe kernel (fn_*, c_*).
 *
 * If any of these stops matching, the backend is emitting something Apple's
 * hardware was never asked to execute.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "agx3_fltnorm_core.h"
#include "agx3_sr_core.h"
#include "agx3_stage_core.h"
#include "agx3_subgroup_core.h"
#include "agx3_tex_core.h"
#include "agx3_writeout_core.h"
#include "agx3_zs_core.h"

static unsigned checks, failures;

static void
check(const char *name, const char *expect_hex, const uint8_t *got,
      unsigned len, bool ok)
{
   char buf[256];
   checks++;
   for (unsigned i = 0; i < len && i < 64; ++i)
      sprintf(buf + 2 * i, "%02x", got[i]);
   buf[2 * len] = 0;
   if (ok && strcmp(buf, expect_hex) == 0)
      return;
   failures++;
   fprintf(stderr, "%-34s want %s got %s\n", name, expect_hex,
           ok ? buf : "(refused)");
}

#define B AGX_ISA_ENCODING_AGX3B


/* ---------------------------------------------------------------- stage */
struct ldimm_v { const char *n, *hex; uint16_t dest; uint32_t imm; uint8_t slot; };
static const struct ldimm_v ldimm_vs[] = {
   {"vmain ldimm.ep2 1.0f",  "0c80233e0000000c4020", 0,  0x3F800000, 2},
   {"vmain ldimm.ep2 0.5f",  "4c80233e000000084080", 8,  0x3F000000, 5},
   {"vmain ldimm.ep2 0.2f",  "5ccd233e1204660240a0", 10, 0x3E4CCCCD, 6},
   {"vmain.v1 ldimm.ep2",    "3c80233e0000000c4060", 6,  0x3F800000, 4},
   {"vmain.v2 ldimm.ep2",    "5ccd233e120466024080", 10, 0x3E4CCCCD, 5},
   {"d_always kill-all",     "0c8023fe1c0cff0f4000", 0,  0xFFFFFF00, 1},
};

struct clr_v { const char *n, *hex; uint16_t dest; uint8_t slot; bool stage; };
static const struct clr_v clr_vs[] = {
   {"vmain clr stage[2] s1", "1b002004", 2, 1, true},
   {"vmain.v1 clr stage[0]", "0b002004", 0, 1, true},
   {"c_zero clr r0",         "0b000000", 0, 0, false},
   {"c_zero clr r2",         "1b000000", 2, 0, false},
   {"c_zero clr r4",         "2b000000", 4, 0, false},
   {"c_zero clr r6",         "3b000000", 6, 0, false},
   
};

static void
test_stage(void)
{
   uint8_t o[AGX3_STAGE_MAX_LENGTH];
   uint8_t l;
   for (unsigned i = 0; i < sizeof(ldimm_vs) / sizeof(*ldimm_vs); ++i) {
      const struct ldimm_v *v = &ldimm_vs[i];
      bool ok = agx3_stage_ldimm(B, v->dest, v->imm, false, v->slot, 0, o, &l);
      check(v->n, v->hex, o, l, ok);
   }
   for (unsigned i = 0; i < sizeof(clr_vs) / sizeof(*clr_vs); ++i) {
      const struct clr_v *v = &clr_vs[i];
      bool ok = v->stage ? agx3_stage_clr(B, v->dest, false, v->slot, o, &l)
                         : agx3_clr(B, v->dest, false, false, o, &l);
      check(v->n, v->hex, o, l, ok);
   }
   /* the two staging copies Apple emits in the writeout path */
   bool sok = agx3_stage_copy(B, 0, 4, 1, 0, o, &l);
   check("fdiscard copy stage[0]<-r4", "0b042904", o, l, sok);
   sok = agx3_stage_copy_kill(B, 0, 0, true, 1, 0, o, &l);
   check("fdepth copy stage[0]<-r0 kill", "0b012f00420000001000", o, l, sok);
   sok = agx3_clr(B, 8, true, true, o, &l);
   check("cbits clr h8 hint", "43002000", o, l, sok);

   /* Independent AGX3c corpus vectors by812, by814, by876 and by1419. */
   sok = agx3_clr(AGX_ISA_ENCODING_AGX3C, 2, true, false, o, &l);
   check("M5-family clr h2", "13000000", o, l, sok);
   sok = agx3_clr(AGX_ISA_ENCODING_AGX3C, 17, true, true, o, &l);
   check("M5-family clr h17 hint", "83002001", o, l, sok);
   sok = agx3_clr(AGX_ISA_ENCODING_AGX3C, 26, false, false, o, &l);
   check("M5-family clr r13", "db000000", o, l, sok);
   sok = agx3_clr(AGX_ISA_ENCODING_AGX3C, 12, false, true, o, &l);
   check("M5-family clr r6 hint", "6b002000", o, l, sok);
}

/* -------------------------------------------------------------- fltnorm */
struct fn_v {
   const char *n, *hex;
   uint16_t dest;
   uint8_t k0; uint16_t v0; bool kill0;
   uint8_t k1; uint16_t v1; bool kill1;
   bool two, snorm;
   uint16_t wait;
   uint8_t srgb;
};
static const struct fn_v fn_vs[] = {
   /* mnemonic witnesses from the shipped .metalar objects */
   {"fmain fltnorm h0",   "97045400020020d045c2", 0, 2,0,true, 2,8,true,  true,false,0},
   {"fmain fltnorm h1",   "97045401020410d045c2", 1, 2,2,true, 2,4,true,  true,false,0},
   {"fdiscard fltnorm h1","97045401020808d045c2", 1, 2,4,true, 2,2,true,  true,false,0},
   {"ftex fltnorm h1",    "97045401020818d045c2", 1, 2,4,true, 2,6,true,  true,false,0},
   {"ftex fltnorm h0 w1", "97145400020008d045c2", 0, 2,0,true, 2,2,true,  true,false,1},
   {"fdepth fltnorm h4",  "97045404020c125045c2", 4, 2,6,false,2,4,true,  true,false,0},
   {"fdepth fltnorm h5",  "97045405021028d045c2", 5, 2,8,true, 2,10,true, true,false,0},
   /* offline probes */
   {"fn_rg fltnorm h0",   "97045400020008d045c2", 0, 2,0,true, 2,2,true,  true,false,0},
   {"fn_swz fltnorm h0",  "97045400021000d045c2", 0, 2,8,true, 2,0,true,  true,false,0},
   {"fn_swz fltnorm h1",  "97045401020808d045c2", 1, 2,4,true, 2,2,true,  true,false,0},
   {"fn_2ct fltnorm h2",  "97045402020c125444c2", 2, 2,6,false,2,4,false, true,false,0},
   {"fn_2ct fltnorm h3",  "970454030210025444c2", 3, 2,8,false,2,0,false, true,false,0},
   {"fn_snorm fltnorm h0","97045400020020d04562", 0, 2,0,true, 2,8,true,  true,true, 0},
   {"fn_snorm fltnorm h1","97045401020410d04562", 1, 2,2,true, 2,4,true,  true,true, 0},
   {"fn_flat fltnorm h4", "97045404020008d045c2", 4, 2,0,true, 2,2,true,  true,false,0},
   {"fn_flat fltnorm h5", "97045405020008d045c2", 5, 2,0,true, 2,2,true,  true,false,0},
   {"fn_const fltnorm h1","970454010250c05004c8", 1, 1,40,false,1,48,false,true,false,0},
   {"fn_buf fltnorm h1",  "97045401021028d005c8", 1, 3,8,false,3,10,false,true,false,0},
   /* single-lane (R8Unorm) */
   {"fn_r fltnorm h0",    "970454000200009044c0", 0, 2,0,true, 0,0,false, false,false,0},
   /*
    * sRGB attachments. Each of these is the same instruction as the linear
    * witness above it with only the conversion mode changed: RGBA8Unorm_sRGB
    * emits mode 4 on the (R,G) pack and mode 6 on the (B,A) pack, because the
    * alpha channel of an sRGB format is linear, and R8Unorm_sRGB is a single
    * lane at mode 4.
    */
   {"s_rgba fltnorm h0",  "97045400020020d04522", 0, 2,0,true, 2,8,true,  true,false,0, 1},
   {"s_rgba fltnorm h1",  "97045401020410d045a2", 1, 2,2,true, 2,4,true,  true,false,0, 2},
   {"s_r8 fltnorm h0",    "97045400020000904420", 0, 2,0,true, 0,0,false, false,false,0, 1},
};

/* blend/native/negpack.py and negpack2.py: -x, abs(x) and -abs(x) into
 * normalized attachments. */
struct fn_mod_v {
   const char *n, *hex;
   bool two, snorm, is_16bit;
   bool abs[2], neg[2];
};
static const struct fn_mod_v fn_mod_vs[] = {
   {"neg r8unorm",     "970454000200009044c1", false, false, false, {0}, {1}},
   {"neg rg8unorm",    "97045400020008d045d3", true, false, false, {0}, {1, 1}},
   {"abs rg8unorm",    "97045400020008d0c5c6", true, false, false, {1, 1}, {0}},
   {"nabs rg8unorm",   "97045400020008d0c5d7", true, false, false, {1, 1}, {1, 1}},
   {"neg r16unorm",    "97045400020000904481", false, false, true, {0}, {1}},
   {"neg rg16snorm",   "97045400020008d04553", true, true, true, {0}, {1, 1}},
   {"neg r8snorm",     "97045400020000904461", false, true, false, {0}, {1}},
};

static void
test_fltnorm_modifiers(void)
{
   uint8_t o[AGX3_FLTNORM_MAX_LENGTH];
   uint8_t l;
   for (unsigned i = 0; i < sizeof(fn_mod_vs) / sizeof(*fn_mod_vs); ++i) {
      const struct fn_mod_v *v = &fn_mod_vs[i];
      struct agx3_fltnorm_desc d = {
         .src = {{AGX3_FLTNORM_SRC_REGISTER, 0, true, v->abs[0], v->neg[0]},
                 {v->two ? AGX3_FLTNORM_SRC_REGISTER : AGX3_FLTNORM_SRC_NONE,
                  2, v->two, v->abs[1], v->neg[1]}},
         .two_lanes = v->two,
         .snorm = v->snorm,
         .is_16bit = v->is_16bit,
      };
      bool ok = agx3_fltnorm_encode(B, &d, o, &l);
      check(v->n, v->hex, o, l, ok);
   }

   /* Only register operands carry the modifiers. */
   struct agx3_fltnorm_desc imm = {
      .src = {{AGX3_FLTNORM_SRC_IMMEDIATE, 48, false, false, true}},
   };
   checks++;
   if (agx3_fltnorm_encode(B, &imm, o, &l)) {
      fprintf(stderr, "negated immediate fltnorm accepted\n");
      failures++;
   }
}

static void
test_fltnorm(void)
{
   uint8_t o[AGX3_FLTNORM_MAX_LENGTH];
   uint8_t l;
   for (unsigned i = 0; i < sizeof(fn_vs) / sizeof(*fn_vs); ++i) {
      const struct fn_v *v = &fn_vs[i];
      struct agx3_fltnorm_desc d = {
         .dest = v->dest,
         .src = {{v->k0, v->v0, v->kill0}, {v->k1, v->v1, v->kill1}},
         .two_lanes = v->two,
         .snorm = v->snorm,
         .wait_mask = v->wait,
         .srgb = v->srgb,
      };
      bool ok = agx3_fltnorm_encode(B, &d, o, &l);
      check(v->n, v->hex, o, l, ok);
   }
}

/* --------------------------------------------------------- perspective */
struct pm_v {
   const char *n, *hex;
   bool stage; uint16_t dest; uint8_t slot;
   uint16_t rcp; int rcp_l;
   uint16_t raw; int raw_l;
   uint8_t vary;
};
static const struct pm_v pm_vs[] = {
   {"fmain pmul r0",    "09012f8500010000", false, 0, 0, 4,1, 0,2, 1},
   {"fmain pmul r8",    "49092f8500020000", false, 8, 0, 4,1, 8,2, 2},
   {"fmain pmul r2",    "19032f8500030000", false, 2, 0, 4,1, 2,2, 3},
   {"fmain pmul r4",    "29073f0500040000", false, 4, 0, 4,2,  6,2, 4},
   {"fdiscard pmul r0", "09012f0300010000", false, 0, 0, 2,0, 0,2, 1},
   {"fdiscard pmul r8", "490b2f8300020000", false, 8, 0, 2,1, 10,2,2},
   {"fdiscard pmul r4", "29052f8300030000", false, 4, 0, 2,1, 4,2, 3},
   {"fdiscard pmul r2", "19073f0300040000", false, 2, 0, 2,2,  6,2, 4},
   {"fdepth pmul r6",   "39072f8b00010000", false, 6, 0, 10,1,6,2, 1},
   {"fdepth pmul r4",   "290f2f8b00020000", false, 4, 0, 10,1,14,2,2},
   {"fdepth pmul r8",   "49092f8b00030000", false, 8, 0, 10,1,8,2, 3},
   {"fdepth pmul r10",  "590d3f0b00040000", false, 10,0, 10,2, 12,2,4},
   {"ftex pmul stage0",  "09052f8100014000", true, 0, 1, 0,1, 4,2, 1},
   {"ftex pmul stage2",  "19033f0100024000", true, 2, 1, 0,2,  2,2, 2},
   {"fn_rg pmul r0",    "09012f8300010000", false, 0, 0, 2,1, 0,2, 1},
   {"fn_rg pmul r2",    "19053f0300020000", false, 2, 0, 2,2,  4,2, 2},
   {"fn_r pmul r0",     "09033f0100010000", false, 0, 0, 0,2,  2,2, 1},
};

static void
test_persp(void)
{
   uint8_t o[AGX3_FLTNORM_MAX_LENGTH];
   uint8_t l;
   for (unsigned i = 0; i < sizeof(pm_vs) / sizeof(*pm_vs); ++i) {
      const struct pm_v *v = &pm_vs[i];
      struct agx3_persp_mul_desc d = {
         .stage_dest = v->stage,
         .dest = v->dest,
         .slot = v->slot,
         .reciprocal = v->rcp,
         .reciprocal_liveness = v->rcp_l,
         .raw = v->raw,
         .raw_liveness = v->raw_l,
         .vary_index = v->vary,
         .dest_hint = !v->stage,
      };
      bool ok = agx3_persp_mul_encode(B, &d, o, &l);
      check(v->n, v->hex, o, l, ok);
   }
}

/* ------------------------------------------------------------- writeout */
struct sa_v {
   const char *n, *hex;
   uint16_t dest; bool hint;
   bool imm; uint16_t src0; bool src0_kill;
   uint16_t src1; bool src1_kill; uint8_t shift; uint16_t wait;
};
static const struct sa_v sa_vs[] = {
   {"fdiscard iadd h4=h4+(h4<<4)", "9f115404030810001605", 4, true, false, 4, false, 4, true, 4, 1},
   {"fdiscard iadd h4=h5+(h4<<4)", "9f015404030a10001605", 4, true, false, 5, false, 4, true, 4, 0},
   {"fdepth iadd h10=h16+(h16<<4)","9f11540a022040001605", 10,false,false, 16,false,16, true, 4, 1},
   {"fdepth iadd h0=15+(h10<<4)",  "9f015400031e28001404", 0, true, true, 15, false, 10, true, 4, 0},
   {"c_sha16 h2=h2+(h2<<4)",       "9f015602030408001605", 2, true, false, 2, false, 2, true, 4, 0x20},
   {"c_sha16 h2=15+(h2<<4)",       "9f015402021e08001404", 2, false, true, 15, false, 2, true, 4, 0},
   {"shl7 first, dead source",      "9f115400030000401414", 0, true, true, 0, false, 0, true, 3, 1},
   {"shl7 first, live source",      "9f115401030000401014", 1, true, true, 0, false, 0, false, 3, 1},
   {"shl7 second",                  "9f015400030000001404", 0, true, true, 0, false, 0, true, 4, 0},
   {"ushort shift1 native",         "9f015602020008401404", 2, false, true, 0, false, 2, true, 1, 0x20},
};

static void
test_writeout(void)
{
   uint8_t o[AGX3_WRITEOUT_MAX_LENGTH];
   uint8_t l;
   for (unsigned i = 0; i < sizeof(sa_vs) / sizeof(*sa_vs); ++i) {
      const struct sa_v *v = &sa_vs[i];
      struct agx3_shift_add_desc d = {
         .dest = v->dest,
         .dest_hint = v->hint,
         .src0_immediate = v->imm,
         .src0 = v->src0,
         .src0_kill = v->src0_kill,
         .src1 = v->src1,
         .src1_kill = v->src1_kill,
         .shift = v->shift,
         .wait_mask = v->wait,
      };
      bool ok = agx3_shift_add16_encode(B, &d, o, &l);
      check(v->n, v->hex, o, l, ok);
   }

   /* SR 2 read (the coverage-word seed) and SR 39 (incoming coverage). */
   uint8_t so[AGX3_SR_MAX_LENGTH];
   uint8_t sl;
   struct agx3_sr_desc sd = {.dest = 4, .width = AGX3_SR_W16, .sr = 2, .slot = 1};
   bool srok = agx3_sr_encode(B, &sd, so, &sl);
   check("fdiscard rld h4<-SR2", "24841006", so, sl, srok);
   struct agx3_sr_desc sd2 = {.dest = 16, .width = AGX3_SR_W16, .sr = 2, .slot = 1};
   srok = agx3_sr_encode(B, &sd2, so, &sl);
   check("fdepth rld h16<-SR2", "84841006", so, sl, srok);

   /* the whole 1x coverage-word sequence, as fdiscard spells it */
   uint8_t seq[AGX3_WRITEOUT_SEQ_MAX];
   unsigned n = 0;
   struct agx3_coverage_desc cd = {
      .sr_temp = 16,
      .mid = 10,
      .word = 0,
      .stage = 0,
      .nibble_immediate = true,
      .nibble = 15,
      .slot = 1,
      .sr_slot = 1,
      .word_kill = true,
      .word_hint = true,
   };
   bool ok = agx3_coverage_word(B, &cd, seq, sizeof(seq), &n);
   char buf[256];
   for (unsigned i = 0; i < n; ++i)
      sprintf(buf + 2 * i, "%02x", seq[i]);
   buf[2 * n] = 0;
   check("fdepth coverage word (4 instrs)",
         "84841006"
         "9f11540a022040001605"
         "9f015400031e28001404"
         "0b012f00420000001000",
         seq, n, ok);
}

/* ------------------------------------------------------------- subgroup */
static void
test_subgroup(void)
{
   uint8_t o[AGX3_SUBGROUP_MAX_LENGTH];
   uint8_t l;
   struct agx3_subgroup_shuffle_desc d = {
      .dest = 4,
      .src = 4,
      .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE,
      .lane = 3,
      .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
      .dest_hint = true,
   };
   bool ok = agx3_subgroup_encode_shuffle(B, &d, o, &l);
   check("csimd simdbcast lane3", "470454040308062c0400", o, l, ok);

   struct agx3_subgroup_shuffle_desc d2 = {
      .dest = 0, .src = 0,
      .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE, .lane = 3,
      .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
      .wait_mask = 0x20,
   };
   ok = agx3_subgroup_encode_shuffle(B, &d2, o, &l);
   check("c_bcast simdbcast lane3", "470456000200062c0400", o, l, ok);

   struct agx3_subgroup_shuffle_desc d3 = {
      .dest = 0, .src = 0,
      .lane_kind = AGX3_SUBGROUP_LANE_IMMEDIATE, .lane = 5,
      .op = AGX3_SUBGROUP_SHUFFLE_BROADCAST,
      .wait_mask = 0x20,
   };
   ok = agx3_subgroup_encode_shuffle(B, &d3, o, &l);
   check("c_bcastf simdbcast lane5", "4704560002000a2c0400", o, l, ok);

   struct agx3_subgroup_reduce_desc r = {
      .dest = 0, .src = 0, .op = AGX3_SUBGROUP_REDUCE_IADD, .wait_mask = 0x20,
   };
   ok = agx3_subgroup_encode_reduce(B, &r, o, &l);
   check("c_rediadd sRediadd", "bf01560002001403", o, l, ok);

   struct agx3_subgroup_reduce_desc r2 = {
      .dest = 0, .src = 0, .op = AGX3_SUBGROUP_REDUCE_FADD, .wait_mask = 0x20,
   };
   ok = agx3_subgroup_encode_reduce(B, &r2, o, &l);
   check("c_redadd sRedf32add", "3f06560002001412", o, l, ok);

   struct agx3_subgroup_reduce_desc r3 = {
      .dest = 0, .src = 0, .op = AGX3_SUBGROUP_REDUCE_FMAX, .wait_mask = 0x20,
   };
   ok = agx3_subgroup_encode_reduce(B, &r3, o, &l);
   check("c_redmax sRedfmax", "bf07560002000812", o, l, ok);

   /* corpus witnesses: dest != src, source stays live */
   struct agx3_subgroup_reduce_desc r4 = {
      .dest = 0, .src = 4, .op = AGX3_SUBGROUP_REDUCE_FADD, .dest_hint = true,
      .src_live = true, .wait_mask = 0x20,
   };
   ok = agx3_subgroup_encode_reduce(B, &r4, o, &l);
   check("csimd sRedf32add", "3f06560003081602", o, l, ok);

   struct agx3_subgroup_reduce_desc r5 = {
      .dest = 8, .src = 4, .op = AGX3_SUBGROUP_REDUCE_FMAX, .dest_hint = true,
      .src_live = true,
   };
   ok = agx3_subgroup_encode_reduce(B, &r5, o, &l);
   check("csimd sRedfmax", "bf07540803080a02", o, l, ok);

   struct agx3_subgroup_reduce_desc r6 = {
      .dest = 0, .src = 0, .op = AGX3_SUBGROUP_REDUCE_IADD, .wait_mask = 0x1,
   };
   ok = agx3_subgroup_encode_reduce(B, &r6, o, &l);
   check("catomic sRediadd", "bf11540002001403", o, l, ok);

   struct agx3_subgroup_reduce_desc r7 = {
      .dest = 0, .src = 0, .op = AGX3_SUBGROUP_REDUCE_SMIN, .wait_mask = 0x20,
   };
   ok = agx3_subgroup_encode_reduce(B, &r7, o, &l);
   check("sr_min_s32 sRedimin", "3f02560002001407", o, l, ok);

   r7.op = AGX3_SUBGROUP_REDUCE_SMAX;
   ok = agx3_subgroup_encode_reduce(B, &r7, o, &l);
   check("sr_max_s32 sRedimax", "bf02560002001407", o, l, ok);
}

/* ------------------------------------------------------------------ zs */
static void
test_zs(void)
{
   uint8_t o[AGX3_ZS_MAX_LENGTH];
   uint8_t l;
   struct agx3_zs_desc d = {.value = 0, .value_kill = true, .slot = 1,
                            .wait_mask = 1};
   bool ok = agx3_zs_encode_sample_mask(B, &d, o, &l);
   check("fdiscard movmsk", "571454000001", o, l, ok);
   ok = agx3_zs_encode_depth_stencil(B, &d, true, false, o, &l);
   check("fdepth depthf", "d71454000003", o, l, ok);
}



/* ------------------------------------------------------- AGX3c / normflt */
struct fnc_v {
   const char *n, *hex;
   uint16_t dest;
   uint8_t k0; uint16_t v0; bool kill0;
   uint8_t k1; uint16_t v1; bool kill1;
   bool two, snorm;
   uint16_t wait;
   uint8_t srgb;
};
/*
 * The same operand map on AGX3c, established by re-running the g17p probe set
 * against applegpu_g17s. Only the sub-opcode constants differ.
 */
static const struct fnc_v fnc_vs[] = {
   {"g17s fn_rgba fltnorm h0", "270000022182a90230028140", 0, 2,0,true, 2,8,true,  true,false,0},
   {"g17s fn_rgba fltnorm h1", "27020003a182a90230008140", 1, 2,2,true, 2,4,true,  true,false,0},
   {"g17s fn_r fltnorm h0",    "270000022181a90230008a40", 0, 2,0,true, 0,0,false, false,false,0},
   {"g17s fn_const fltnorm h1","270000532581a902098c8240", 1, 1,40,false,1,48,false,true,false,0},
   {"g17s fn_snorm fltnorm h0","270000022182a90230024140", 0, 2,0,true, 2,8,true,  true,true, 0},
   {"g17s fn_buf fltnorm h1",  "270000132581a90290028140", 1, 3,8,false,3,10,false,true,false,0},
   {"g17s ftex fltnorm h0 w1", "270400022182a902b0008140", 0, 2,0,true, 2,2,true,  true,false,1},
   {"g17s fb4 fltnorm h4",     "270000022182a90ab0008140", 4, 2,0,true, 2,2,true,  true,false,0},
   /* The sRGB modes are the same two field values on this encoding: these are
    * byte-for-byte the first two vectors above with only the mode changed. */
   {"g17s s_rgba fltnorm h0",  "270000022182a90230024100", 0, 2,0,true, 2,8,true,  true,false,0, 1},
   {"g17s s_rgba fltnorm h1",  "27020003a182a9023000c100", 1, 2,2,true, 2,4,true,  true,false,0, 2},
};

struct nf_v {
   const char *n, *hex;
   int enc; /* 0 = AGX3b, 1 = AGX3c */
   uint16_t dest, src;
   bool two;
   uint16_t wait;
   uint8_t srgb;
   bool no_hint;
};
static const struct nf_v nf_vs[] = {
   {"fb4 normflt r0<-h4 w6", "17045600010814ea", 0, 0, 4, true,  0x20},
   {"fb4 normflt r0<-h5",    "17045400010a14ea", 0, 0, 5, true,  0},
   {"fb2 normflt r0<-h0 w6", "17045600010014ea", 0, 0, 0, true,  0x20},
   {"fb1 normflt r0<-h0 w6", "170456000100146a", 0, 0, 0, false, 0x20},
   {"g17s fb4 normflt w8",   "2f00003a2d00a802d000", 1, 0, 4, true,  0x80},
   {"g17s fb4 normflt r0<-h5","2701003a2d00a802d000", 1, 0, 5, true,  0},
   {"g17s fb1 normflt w8",   "2f00001a2d00a802c000", 1, 0, 0, false, 0x80},
   /*
    * The decode half of the same story: programmable blending on an sRGB
    * attachment reads it back with the same two modes. The two linear
    * witnesses below are the same instructions from the same probe against a
    * linear attachment, so the pair isolates the mode field.
    */
   {"l_pblend normflt r4<-h0", "17045604010014ea", 0, 4, 0, true, 0x20, 0, false},
   {"l_pblend normflt r0<-h1", "17045400000214ea", 0, 0, 1, true, 0,    0, true},
   {"s_pblend normflt r4<-h0", "170456040100149a", 0, 4, 0, true, 0x20, 1, false},
   {"s_pblend normflt r0<-h1", "17045400000214da", 0, 0, 1, true, 0,    2, true},
};

static void
test_agx3c_and_normflt(void)
{
   uint8_t o[AGX3_FLTNORM_MAX_LENGTH];
   uint8_t l;
   for (unsigned i = 0; i < sizeof(fnc_vs) / sizeof(*fnc_vs); ++i) {
      const struct fnc_v *v = &fnc_vs[i];
      struct agx3_fltnorm_desc d = {
         .dest = v->dest,
         .src = {{v->k0, v->v0, v->kill0}, {v->k1, v->v1, v->kill1}},
         .two_lanes = v->two,
         .snorm = v->snorm,
         .wait_mask = v->wait,
         .srgb = v->srgb,
      };
      const bool ok = agx3_fltnorm_encode(AGX_ISA_ENCODING_AGX3C, &d, o, &l);
      check(v->n, v->hex, o, l, ok);
   }

   for (unsigned i = 0; i < sizeof(nf_vs) / sizeof(*nf_vs); ++i) {
      const struct nf_v *v = &nf_vs[i];
      const struct agx3_normflt_desc d = {
         .dest = v->dest,
         .dest_hint = !v->no_hint,
         .src = v->src,
         .two_lanes = v->two,
         .wait_mask = v->wait,
         .srgb = v->srgb,
      };
      const bool ok =
         agx3_normflt_encode(v->enc ? AGX_ISA_ENCODING_AGX3C
                                    : AGX_ISA_ENCODING_AGX3B,
                             &d, o, &l);
      check(v->n, v->hex, o, l, ok);
   }
}

static void
test_refusals(void)
{
   uint8_t o[AGX3_FLTNORM_MAX_LENGTH];
   uint8_t l;
   struct agx3_fltnorm_desc d = {
      .dest = 0,
      .src = {{AGX3_FLTNORM_SRC_REGISTER, 0, true},
              {AGX3_FLTNORM_SRC_REGISTER, 2, true}},
      .two_lanes = true,
   };
   checks++;
   struct agx3_fltnorm_desc odd = d;
   odd.src[0].value = 1; /* a 32-bit source must be even */
   if (agx3_fltnorm_encode(B, &odd, o, &l)) {
      fprintf(stderr, "fltnorm accepted an odd 32-bit source\n");
      failures++;
   }

   checks++;
   struct agx3_fltnorm_desc mism = d;
   mism.two_lanes = false; /* two_lanes must agree with src[1] */
   if (agx3_fltnorm_encode(B, &mism, o, &l)) {
      fprintf(stderr, "fltnorm accepted a lane-count mismatch\n");
      failures++;
   }

   /* The sRGB modes replace the linear format field rather than joining it,
    * and mode 6 exists only to name lane 1 as the linear one, so it has no
    * meaning without a second lane. */
   static const struct {
      const char *why;
      uint8_t srgb;
      bool snorm, is_16bit, one_lane;
   } bad_srgb[] = {
      {"sRGB combined with snorm", AGX3_FLTNORM_SRGB_BOTH, true, false, false},
      {"sRGB combined with 16-bit", AGX3_FLTNORM_SRGB_BOTH, false, true, false},
      {"sRGB lane0 mode on a single lane", AGX3_FLTNORM_SRGB_LANE0, false,
       false, true},
      {"an out-of-range sRGB selector", 3, false, false, false},
   };
   for (unsigned i = 0; i < sizeof(bad_srgb) / sizeof(*bad_srgb); ++i) {
      checks++;
      struct agx3_fltnorm_desc bad = d;
      bad.srgb = bad_srgb[i].srgb;
      bad.snorm = bad_srgb[i].snorm;
      bad.is_16bit = bad_srgb[i].is_16bit;
      if (bad_srgb[i].one_lane) {
         bad.two_lanes = false;
         bad.src[1].kind = AGX3_FLTNORM_SRC_NONE;
      }
      if (agx3_fltnorm_encode(B, &bad, o, &l)) {
         fprintf(stderr, "fltnorm accepted %s\n", bad_srgb[i].why);
         failures++;
      }

      checks++;
      struct agx3_normflt_desc nbad = {
         .dest = 0, .dest_hint = true, .src = 0,
         .two_lanes = !bad_srgb[i].one_lane,
         .srgb = bad_srgb[i].srgb,
         .snorm = bad_srgb[i].snorm,
         .is_16bit = bad_srgb[i].is_16bit,
      };
      if (agx3_normflt_encode(B, &nbad, o, &l)) {
         fprintf(stderr, "normflt accepted %s\n", bad_srgb[i].why);
         failures++;
      }
   }

   checks++;
   struct agx3_shift_add_desc sa = {.dest = 0, .src0 = 0, .src1 = 0, .shift = 5};
   uint8_t wo[AGX3_WRITEOUT_MAX_LENGTH];
   if (agx3_shift_add16_encode(B, &sa, wo, &l)) {
      fprintf(stderr, "iadd accepted an unwitnessed shift\n");
      failures++;
   }

   checks++;
   uint8_t so[AGX3_STAGE_MAX_LENGTH];
   if (agx3_stage_ldimm(AGX_ISA_ENCODING_AGX3C, 0, 0, false, 1, 0, so, &l)) {
      fprintf(stderr, "staging ldimm accepted on AGX3c\n");
      failures++;
   }

   checks++;
   if (agx3_stage_ldimm(B, 0, 0, false, 7, 0, so, &l)) {
      fprintf(stderr, "staging ldimm accepted slot 7 on AGX3b\n");
      failures++;
   }

}

int
main(void)
{
   test_stage();
   test_fltnorm();
   test_fltnorm_modifiers();
   test_persp();
   test_writeout();
   test_subgroup();
   test_zs();
   test_agx3c_and_normflt();
   test_refusals();

   if (failures) {
      fprintf(stderr, "agx3_g17p_producers: %u/%u checks failed\n", failures,
              checks);
      return 1;
   }
   printf("agx3_g17p_producers: all %u checks passed\n", checks);
   return 0;
}

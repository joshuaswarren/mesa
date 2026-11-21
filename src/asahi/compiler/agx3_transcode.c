/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G18G (agx3d) code from the G17G (agx3c) packer.
 *
 * G18G executes the G17G instruction set under a third AGX3 encoding. The
 * field ids of the AGX3 instruction encoding tables are shared across
 * encodings, and on the 27 G17G/G18G pairs of Apple-compiled witness texts
 * (test/native/g18g) the two are one machine:
 *
 *  - every G17G instruction decodes, and all but the texture samples
 *    (1750 of 1755) encode under agx3d with the same field vector;
 *  - where Apple's G17G and G18G compilers chose the same field vector for an
 *    aligned instruction, composing the G17G vector with the agx3d table in
 *    Apple's form order reproduces Apple's G18G bytes (146 of 146 distinct);
 *  - scoreboard slots and waits are the same fields with the same meaning,
 *    and agx3d encodes every single-slot and two-slot wait agx3c can (tested
 *    on f32add, f32mul and f32fma).
 *
 * So the G17G packer's output is re-encoded instruction by instruction:
 * segment with the agx3c length rule, decode the field vector with
 * agx3_decode(), apply the G18G field rules below, and compose it with the
 * agx3d table. Rules, all measured on the witness texts:
 *
 *  - lddev: G18G sets f87 = 2 whenever f75 is nonzero and leaves it 0 when
 *    f75 is 0 (16 of 16 distinct G18G lddev shapes); G17G never sets it.
 *  - smp, from the five G17G/G18G sample pairs (a fragment sample, level-zero
 *    array samples, and array texel reads with uniform and register indices):
 *    G18G always sets f57 = 1 and f158 = 4; f56 = 2 where f77 is 0 and
 *    f19 = 4 where f77 is 2; and where G17G sets f69 it sets f62 instead.
 *    Mesa's tile reload has f62 = f69 = 0, a pair no witness shows; it is
 *    kept 0 on the reading that f69 folds into f62.
 *  - imgwci, from Apple's image_write(texture2d<uint, access::write>) --
 *    the kernel native run112 kick 19 ran on the J873 -- compiled by
 *    metal-nt 27.1.266.1 for both machines. The G18G text
 *    is byte-identical to kick 19's code on the device:
 *
 *      G17G 250411042420a0020080008c002000000100  X@0 Y@2 data@4, f5 0
 *      G18G 160340c00b002100000000000060          data@0 X@2 Y@4, f5 2
 *
 *    The two vectors are the same but for f5 and five G18G-only fields.
 *    G17G implies Y and data from its bank (f5 + 2, f5 + 4); G18G names each
 *    operand: f5 the X half, f116 the Y half, f118 the data half. Checked on
 *    six more metal-nt G18G writes whose staging Apple laid out differently:
 *    RGBA data 0..6/X 8/Y 10, RG data 0,2/X 4/Y 6, constant coordinates
 *    X 0/data 2/Y 4, a 2D array X 6/Y 0/data 4, and two writes sharing
 *    X 0/Y 2 with data 4 and data 6.
 *    f117 equals f13 in every witness (both 0 only on the first of the two
 *    writes that share coordinates, so read as the staged operands'
 *    release); f119 is 1 in every witness and f120 is 4, which every form's
 *    predicates require. Mesa's G18G packer stages the one-component layout
 *    Apple's G18G compiler uses (data just below X, Y just above) and names
 *    X in f5 (agx3_tex_encode_image_write_g18g), so the rule below only
 *    adds f116 = f5 + 2, f118 = f5 - 2 and the constants, and composes
 *    kick 19's bytes exactly for X@2.
 *
 *    The four-component writes are the same rule with the data eight halves
 *    below X: Apple's G18G compiler lays a lone RGBA write out as data
 *    0..6, X 8, Y 10 (m6-image-write iw_cs_rgba32f, m6-image-write2
 *    iw2_{f4,u4,i4}_t*), and Mesa's vector for that layout composes exactly
 *    Apple's 160940c017002100008010000020 (float, slot 1) and
 *    160940c017002300008000000020 (integer, staging slot 3). Its other
 *    fields pass through, each within the values the G18G witnesses show:
 *    the result slot f12 1..4 and one staging wait f59..f63 (m6-image-write4
 *    iw4_four), f70, texture3d f72 3 and the 2D array's f73 2 (whose z or
 *    layer is implicit at Y + 2: iw4_f4_3d, iw4_f4_array, iw_cs_r32i_array),
 *    and the heap index f91 (0..3 in iw4_four) at heap half f79 0. Apple's
 *    3D and array vectors differ from Mesa's only in f116/f118, where its
 *    allocator put Y below X.
 *
 * Ioto resource programs (agx3_transcode_agx3d_ioto), measured on Apple's
 * G18G Ioto programs in the J873 native captures (every mode of sessions
 * 20260925-a, -b, -g1 and -g2, and compute-add):
 *
 *  - ldshdr: agx3d has no f23 (G17G sets 1); f21 = 2 on every API program
 *    (vertex, fragment, compute) and 0 on the driver's tile programs;
 *  - cndcall: the helper call keeps its target f16 = 27 and drops f15;
 *  - specTPR: agx3d adds f24, the launched shader's register allocation:
 *    the power of two at or above its 32-bit register count, at least 8
 *    (every fragment 8, vertex 8 or 16, the end-of-tile shader 16 for r12).
 *
 * Control flow, measured on the control-flow witnesses (probe
 * m5-control-flow-reference, sixteen kernels of ifs, elses, loops, breaks,
 * continues and switches that Apple cannot if-convert):
 *
 *  - Flag slots move up by one. Every G17G compare that writes slot 0 or 1
 *    (icmp f31, fcmp f35) writes slot 1 or 2 on G18G, and every exec-mask op
 *    that reads it (cndst/cndlt/cndcf f1) reads slot 1 or 2; G17G's
 *    always-true slot 6 (the loop-entry cndst and cndef f1 = 6) is G18G's
 *    slot 0 with the invert bit f0 flipped. No G18G compare ever writes slot
 *    0. With that rule 114 of the 118 aligned cndst/cndlt/cndcf/cndef/cndend
 *    reproduce Apple's G18G bytes -- the other four are cndsts whose invert
 *    Apple chose together with the opposite compare condition -- and so does
 *    every compare whose sources Apple chose alike.
 *  - Slots 2..5 have no G18G witness. The G17G packer emits every compare
 *    immediately before the exec-mask op that consumes it, so no predicate is
 *    live across another instruction and the slot is only a name; such a
 *    program folds them to slot 1. That is checked here, over the whole
 *    program, rather than assumed.
 *  - b/bn keep their field vector apart from the displacement f0: a 48-bit
 *    byte offset from the branch's own first byte on both encodings (35 of
 *    35 aligned branch targets land on the same instruction). Apple encodes
 *    every G18G branch in the ten-byte form (ep 5, all 39 branches), which is
 *    also the G17G branch length, so re-encoding the other instructions moves
 *    the targets but never a branch's own size: the displacements are
 *    recomputed once every agx3c offset has its agx3d offset.
 *
 * Refused (the shader then fails, it is never mis-encoded):
 *
 *  - the control flow no G17G/G18G pair shows: bl, brk, cndret, dbgrtn,
 *    yield, and cndcall outside Ioto programs;
 *  - a branch whose target is not an instruction boundary of the program;
 *  - smp with an f77 other than 0 or 2 (the only values the witnesses show);
 *  - any imgwci outside the forms agx3_tex_encode_image_write_g18g() emits:
 *    a nonzero heap half f79, a register or uniform image index, no wait or
 *    more than one, f114 splat (5) or RGB-and-one (2), the cube texture
 *    type f72 4, and a one-component write other than kick 19's 2D integer
 *    one.
 *
 * Not modelled: Apple's G18G vertex programs add a bit_andc/iself32/bit_and/
 * bit_or epilogue on float varyings that G17G does not. Its purpose is not
 * known, so it is not emitted; hardware will say whether it matters.
 */
#include "agx3_transcode.h"
#include "agx3_tex_core.h"

#include <stdlib.h>
#include <string.h>

#include "util/macros.h"
#include "agx3_compose/agx3_compose.h"
#include "agx3_compose/agx3_decode.h"
#include "asahi/isa/disasm.h"

static const char *const refused_control_flow[] = {
   "bl", "brk", "cndcall", "cndret", "dbgrtn", "yield",
};

/* The G17G flag slot that always reads true, and G18G's always-false one. */
#define G17G_TRUE_SLOT  6
#define G18G_FALSE_SLOT 0

/* agx3d's ten-byte b/bn, the only branch form in Apple's G18G code. */
#define G18G_BRANCH_EP   5
#define G18G_BRANCH_SIZE 10

#define DISPLACEMENT_MASK ((UINT64_C(1) << 48) - 1)

static bool
is_control_flow(const char *mnemonic)
{
   for (unsigned i = 0; i < ARRAY_SIZE(refused_control_flow); ++i) {
      if (!strcmp(mnemonic, refused_control_flow[i]))
         return true;
   }
   return false;
}

/* The disassembler's display name, without its padding. */
static void
trim_display(const char *display, char *out, size_t size)
{
   size_t n = 0;
   while (display[n] && display[n] != ' ' && n + 1 < size) {
      out[n] = display[n];
      n++;
   }
   out[n] = '\0';
}

static unsigned
spectpr_allocation(unsigned gprs16)
{
   const unsigned regs = (gprs16 + 1) / 2;
   unsigned alloc = 8;
   while (alloc < regs)
      alloc *= 2;
   return alloc;
}

static bool
apply_ioto_rules(const char *mnemonic, uint64_t fields[AGX3_MAX_FIELDS],
                 const struct agx3_transcode_ioto *ioto, const char **why)
{
   if (!strcmp(mnemonic, "ldshdr")) {
      fields[23] = 0;
      fields[21] = ioto->api_program ? 2 : 0;
   } else if (!strcmp(mnemonic, "specTPR")) {
      if (!ioto->shader_gprs || ioto->shader_gprs > 256) {
         *why = "specTPR without a register count";
         return false;
      }
      fields[24] = spectpr_allocation(ioto->shader_gprs);
   } else if (!strcmp(mnemonic, "cndcall")) {
      /* Only the witnessed helper call: target 27. */
      if (fields[16] != 27 || fields[17] != 1 || fields[25] != 1) {
         *why = "cndcall other than the witnessed Ioto helper call";
         return false;
      }
      fields[15] = 0;
   }
   return true;
}

/* The constant part of every agx3c imgwci agx3_tex_encode_image_write_g18g()
 * emits: Apple's G17G image_write vector (see the header) with the staged
 * operands' release f13 set. */
static const struct {
   uint8_t field, value;
} g18g_imgwci_fields[] = {
   {9, 5},   {10, 2},  {13, 1},  {24, 1},  {38, 1},  {51, 1},  {52, 1},
   {56, 2},  {58, 0x57}, {71, 2}, {74, 1}, {85, 3},  {86, 3}, {113, 1},
};

/* The last staging slot an imgwci waits on and the last it signals in
 * Apple's G18G code (f59..f63 and f12 1..4 across m6-image-write{,2,3,4}). */
#define G18G_IMGWCI_MAX_WAIT_SLOT   5
#define G18G_IMGWCI_MAX_RESULT_SLOT 4
/* Mesa's image heap indices: an immediate index below the sixteen texture
 * state registers (agx_nir_lower_bindings). */
#define G18G_IMGWCI_MAX_INDEX 15

static bool
apply_imgwci_rule(uint64_t fields[AGX3_MAX_FIELDS], const char **why)
{
   uint64_t expected[AGX3_MAX_FIELDS] = {0};
   for (unsigned i = 0; i < ARRAY_SIZE(g18g_imgwci_fields); ++i)
      expected[g18g_imgwci_fields[i].field] = g18g_imgwci_fields[i].value;

   /* The operand fields the G18G witnesses vary, each within the values
    * they show: one staging wait, the result slot, the data type, the
    * texture type (2D, 2D array, 3D), the heap base and index form, and
    * one or four staged components (f114 4 only for the 2D integer write of
    * kick 19). */
   unsigned wait = 0, waits = 0;
   for (unsigned s = 1; s <= 8; ++s) {
      if (fields[58 + s]) {
         wait = s;
         waits++;
      }
   }
   const uint64_t x = fields[5];
   const bool array = fields[73] == 2;
   const bool three_d = fields[72] == 3;
   const bool one = fields[114] == 4;
   const uint64_t data_halves = one ? 2 : 8;
   /* Native459 keeps descriptor selection independent of the data type:
    * f91 immediate, f92 GPR, f93 uniform. Only the GPR form may kill its
    * index. Unselected index fields and all unknown fields remain zero. */
   const bool immediate = fields[99] == 1 && fields[100] == 1 &&
      fields[91] <= G18G_IMGWCI_MAX_INDEX && !fields[92] && !fields[93] &&
      !fields[103];
   const bool reg = fields[99] == 2 && fields[100] == 2 &&
      !fields[91] && !(fields[92] & 1) && fields[92] <= 126 &&
      !fields[93] && fields[103] <= 1;
   const bool uniform = fields[99] == 3 && fields[100] == 2 &&
      !fields[91] && !fields[92] && !(fields[93] & 1) &&
      fields[93] <= 252 && !fields[103];
   bool witnessed =
      waits == 1 && fields[58 + wait] == 1 &&
      wait <= G18G_IMGWCI_MAX_WAIT_SLOT && fields[12] >= 1 &&
      fields[12] <= G18G_IMGWCI_MAX_RESULT_SLOT && fields[70] <= 1 &&
      (fields[72] == 1 || three_d) && (fields[73] == 0 || array) &&
      !(three_d && array) && !(fields[79] & 3) && fields[79] <= 252 &&
      (immediate || reg || uniform) &&
      (fields[114] == 0 ||
       (one && fields[72] == 1 && !array && fields[70] == 0)) &&
      !(x & 1) && x >= data_halves &&
      x + (array || three_d ? 4 : 2) <= 126;
   for (unsigned i = 0; i < AGX3_MAX_FIELDS && witnessed; ++i) {
      witnessed = i == 5 || i == 12 || i == 58 + wait || i == 70 ||
                  i == 72 || i == 73 || i == 79 || i == 91 || i == 92 ||
                  i == 93 || i == 99 || i == 100 || i == 103 || i == 114 ||
                  fields[i] == expected[i];
   }
   if (!witnessed) {
      *why = "image write outside the witnessed G18G imgwci forms";
      return false;
   }

   /* G18G names Y and the data; a layer or z is implicit at Y + 2 in every
    * witness (iw_cs_rgba32ui_3d, iw4_f4_3d, iw_cs_r32i_array, iw4_f4_array),
    * which is where the packer stages it. */
   fields[116] = x + 2;
   fields[117] = fields[13];
   fields[118] = x - data_halves;
   fields[119] = 1;
   fields[120] = 4;
   return true;
}

static bool
apply_g18g_rules(const char *mnemonic, uint64_t fields[AGX3_MAX_FIELDS],
                 const struct agx3_transcode_ioto *ioto,
                 const bool *fragment_stage, const char **why)
{
   const bool ioto_only = !strcmp(mnemonic, "ldshdr") ||
                          !strcmp(mnemonic, "specTPR") ||
                          !strcmp(mnemonic, "specLM") ||
                          !strcmp(mnemonic, "cndcall");
   if (ioto_only) {
      if (!ioto) {
         *why = "Ioto-only instruction in shader code";
         return false;
      }
      return apply_ioto_rules(mnemonic, fields, ioto, why);
   }
   if (is_control_flow(mnemonic)) {
      *why = "control flow (branch relocation is not grounded for G18G)";
      return false;
   }
   /* Native G18 PACK_SMPIR takes a 32-bit multisample index. The G17
    * intermediate only represents this typed source at 16 bits. Bank 2
    * identifies the G18 publisher, whose caller zero-extends packed NIR
    * samples into a separate full word before reaching this translation. */
   if (!strcmp(mnemonic, "pack_smpir") && fields[9] == 5 && fields[5] == 2 &&
       fields[74] == 9 && fields[75] == 12 && fields[76] == 1 &&
       fields[23] == 2 && fields[24] == 1)
      fields[24] = 2;
   if (!strcmp(mnemonic, "smp")) {
      if (fields[77] != 0 && fields[77] != 2) {
         *why = "texture sample whose f77 no G18G witness shows";
         return false;
      }
      fields[57] = 1;
      fields[158] = 4;
      /* Native371: every G18 gather adds f155=4, independently of its
       * component and stage; ordinary sample controls leave it clear. */
      if (fields[144])
         fields[155] = 4;
      if (fields[77] == 0) {
         /* G18 names the second coordinate explicitly. The original
          * fullscreen texture witness stages X/Y at 0/2 (f19/f56=0/2),
          * while the native three-comparison witness stages them at 4/0
          * (f19/f56=4/0). G17's ordinary 2D operand is a contiguous pair;
          * its rotated staging blocks must retain their Y half here.
          * 3D is the same: G18G reads Z at Y + 4 and a staged level or bias
          * at Y + 2 whatever X is, and the G18G stagers place both to match
          * (agx3_tex_core.h, AGX3_TEX_G18G_LEVEL_WORD). Array and cube
          * operands have separate layouts. */
         fields[56] = fields[76] == 1 || fields[76] == 3 ? fields[19] + 2 : 2;
      } else
         fields[19] = 4;
       /* Native949 query3D publishes X/Z separately from Y. Unlike an
        * ordinary 3D sample, G18's query names X at8 and Y at0. */
       if (fields[155] == 2 && fields[74] == 0 && fields[76] == 3 &&
           fields[139] == 0) {
          fields[19] = 8;
          fields[56] = 0;
       }
      /* A comparison (f143) names its reference half on G18G -- f58, with
       * f59 set -- where G17G reads it implicitly at X + 4. applegpu_g18g
       * depth2d sample_compare (m6-shadow-lod): X 0, Y 2, reference 4 with
       * f58 = 4; with a level or bias (f139 = 4 or 3) that word is Y + 2, so
       * the reference moves up: X 4, Y 0, level 2, reference 6, f58 = 6. The
       * G18G stagers keep a 2D reference at X + 4, or X + 6 with a staged
       * level or bias. Only the 2D non-array shape is witnessed. */
      if (fields[143] && fields[77] == 0 && fields[76] == 1) {
         fields[58] = fields[19] +
                      (fields[139] == 3 || fields[139] == 4 ? 6 : 4);
         fields[59] = 1;
      }
      /* Native G18 depth2d_array samples use X/Y = 4/0 and the packed
       * layer/LOD word at 2. Name the separately staged reference at 6,
       * including immediate texel offsets and per-lane bias/level. */
      if (fields[143] && fields[77] == 2 && fields[76] == 1) {
         fields[56] = 0;
         fields[58] = 6;
         fields[59] = 1;
      }
      /* Native403 runtime gathers share X/Y=4/0, offset/layer=2 and
       * comparison=6 across 2D/arrays and vertex/fragment/compute. */
      if (fields[144] && fields[141] == 2 && fields[76] == 1) {
         fields[19] = 4;
         fields[56] = 0;
         if (fields[143]) {
            fields[58] = 6;
            fields[59] = 1;
         }
      }
      /* Native408's plain/constant-offset array gathers retain Y and the
       * comparison reference through nonterminal samples. Their final SMP
       * carries f150=2 and discards those operands. Runtime offset/layer
       * publications retain their independently witnessed lifetime above. */
      if (fields[144] && fields[77] == 2 && fields[76] == 1 &&
          fields[141] <= 1) {
         fields[57] = fields[150] == 2;
         if (fields[143])
            fields[59] = fields[150] == 2;
      }
      /* Native G18 cube comparisons consume CEM X/Y at 6/0 and reference
       * 4; cube arrays use X/Y 4/0 and reference 6. Both publications put
       * face/layer/LOD at half 2. The G18-specific stagers use those banks. */
      if (fields[143] && fields[76] == 4) {
         fields[19] = fields[77] == 2 ? 4 : 6;
         fields[56] = 0;
         fields[58] = fields[77] == 2 ? 6 : 4;
         fields[59] = 1;
      }
      /* Native607 explicit levels and native610 fragment implicit/bias
       * colour cubes use CEM X/Y=4/0 and the face/level word at 2. */
      if (fields[76] == 4 && fields[78] == 2 && fields[77] == 0 &&
          !fields[143] && !fields[144] &&
          (fields[139] == 0 || fields[139] == 2 ||
           fields[139] == 3 || fields[139] == 4)) {
         fields[19] = 4;
         fields[56] = 0;
      }
      /* Native413: cube-gather banks depend on the actual stage, not the
       * final-fragment marker, which may be absent on earlier samples. */
      if (fields[144] && fields[76] == 4) {
         if (!fragment_stage) {
            *why = "cube gather requires shader-stage context";
            return false;
         }
         const bool depth_fragment = fields[143] && *fragment_stage;
         fields[19] = depth_fragment ? 6 : 4;
         fields[56] = 0;
         if (fields[143]) {
            fields[58] = depth_fragment ? 4 : 6;
            fields[59] = 1;
         }
      }
      /* G18 multisample read: native X/Y = 4/0, packed sample/layer = 2. */
      if (fields[76] == 2 && fields[78] == 1) {
         fields[19] = 4;
         fields[56] = 0;
      }
      /* G18 2D/array explicit gradients: X/Y 16/0, derivatives 4/6/8/10.
       * Both stages and immediate offsets use this same native layout. */
      if (fields[76] == 1 && fields[139] == 5 && !fields[143]) {
         fields[19] = AGX3_TEX_G18G_GRADIENT_X;
         fields[56] = 0;
      }
      fields[62] |= fields[69];
      fields[69] = 0;
   }
   if (!strcmp(mnemonic, "imgwci") && !apply_imgwci_rule(fields, why))
      return false;
   if (!strcmp(mnemonic, "lddev"))
      fields[87] = fields[75] ? 2 : 0;
   /* Native454/567's register-sample RGBA32, RGBA16Float and BGRA8 reloads
    * use the G18 long LLDCol form with f79=2 at every payload width.
    * Sample, slot and payload fields retain their G17G meaning. */
   /* Matched native scalar framebuffer fetches also carry f79=2 at 1x.
    * The immediate has already been normalized from Mesa's 0xff to 1. */
   const bool scalar_pixel_load = fields[51] == 1 && fields[43] == 1 &&
                                  fields[10] == 2 && fields[26] == 1 &&
                                  !(fields[57] % 4);
   /* Native844 pairs: whole-pixel loads at RT0 retain their width/mask
    * fields but use the G18 long form. Do not extend this to other offsets
    * or width/mask combinations without a matching native profile.
    * Native844 brackets that long form with pixwait 0x200 ... pixrel 0x20c;
    * Mesa's G18 fragment programs do not. Fault1080/1084: this unbracketed
    * long RGBA8 (8-bit, mask 0xf) load is the only GPU-visible change in a
    * render that recovers on M6; its short form passed physically through
    * 723. Keep RGBA8 on that short form until the bracket is emitted and
    * physically validated. */
   const bool native844_pixel_load =
      fields[51] == 1 && fields[43] == 1 && fields[57] == 0 &&
      ((fields[10] == 0 && fields[26] == 3) ||
       (fields[10] == 1 && (fields[26] == 1 || fields[26] == 3 ||
                            fields[26] == 7 || fields[26] == 15)) ||
       (fields[10] == 2 && (fields[26] == 1 || fields[26] == 3)));
   if (!strcmp(mnemonic, "lldcol") &&
       ((fields[51] == 2 && fields[10] <= 2 && fields[26] == 15) ||
        scalar_pixel_load || native844_pixel_load))
      fields[79] = 2;
   if (!strcmp(mnemonic, "ldcfr")) {
      /* Native G18 no-perspective interpolate_at_offset keeps the raw
       * three-coefficient selector f38=7 and adds f49=2. The flat-input
       * witness also has f49=2, but uses f38=5: that flat-only selector
       * must not replace Mesa's raw <A,B,C> load (used by cull distance).
       * Keep all operand, result-slot and scheduling fields unchanged.
       */
      fields[49] = 2;
   }
   return true;
}

/* The field naming the flag slot a compare writes or an exec-mask op reads,
 * or -1. cndend names none. */
static int
flag_slot_field(const char *mnemonic)
{
   if (!strcmp(mnemonic, "icmp"))
      return 31;
   if (!strcmp(mnemonic, "fcmp"))
      return 35;
   if (!strcmp(mnemonic, "cndst") || !strcmp(mnemonic, "cndlt") ||
       !strcmp(mnemonic, "cndcf") || !strcmp(mnemonic, "cndef"))
      return 1;
   return -1;
}

static bool
is_compare(const char *mnemonic)
{
   return !strcmp(mnemonic, "icmp") || !strcmp(mnemonic, "fcmp");
}

/* Whole-program flag-slot state: which predicate the previous instruction
 * wrote, whether every predicate so far was consumed by the instruction right
 * after its compare, and whether a slot without a G18G witness was folded. */
struct flag_state {
   int pending;
   bool adjacent;
   bool folded;
};

static bool
apply_flag_slot_rule(const char *mnemonic, uint64_t fields[AGX3_MAX_FIELDS],
                     struct flag_state *state, const char **why)
{
   const int field = flag_slot_field(mnemonic);
   const bool compare = is_compare(mnemonic);
   const uint64_t slot = field >= 0 ? fields[field] : 0;
   const bool reads_flag = field >= 0 && !compare && slot != G17G_TRUE_SLOT;

   /* A predicate must be read by the instruction right after its compare,
    * and a read must come right after the compare, for the slot to be only
    * a name. */
   if (state->pending >= 0 && !(reads_flag && slot == (uint64_t)state->pending))
      state->adjacent = false;
   if (reads_flag && state->pending != (int)slot)
      state->adjacent = false;
   state->pending = compare ? (int)slot : -1;

   if (field < 0)
      return true;

   if (slot == G17G_TRUE_SLOT) {
      /* Only the unconditional loop-entry push and the else flip read the
       * always-true slot, in G17G and G18G code alike. */
      if (strcmp(mnemonic, "cndst") && strcmp(mnemonic, "cndef")) {
         *why = "always-true flag slot outside cndst/cndef";
         return false;
      }
      fields[1] = G18G_FALSE_SLOT;
      fields[0] ^= 1;
      return true;
   }
   if (slot > 5) {
      *why = "flag slot above 5";
      return false;
   }
   if (slot <= 1) {
      fields[field] = slot + 1;
   } else {
      fields[field] = 1;
      state->folded = true;
   }
   return true;
}

static int64_t
sign_extend_48(uint64_t v)
{
   return (int64_t)(v << 16) >> 16;
}

/* One re-encoded b/bn waiting for its displacement. */
struct pending_branch {
   unsigned out_offset;
   unsigned in_offset;
   int64_t in_target;
   int32_t form;
   uint64_t fields[AGX3_MAX_FIELDS];
};

/* agx3c offset -> agx3d offset of one instruction boundary. */
struct offset_pair {
   unsigned in, out;
};

static bool
relocate_branches(struct util_dynarray *out, const struct util_dynarray *map,
                  const struct util_dynarray *branches, const char **why,
                  unsigned *fail_offset)
{
   const struct offset_pair *pairs = map->data;
   const unsigned count = util_dynarray_num_elements(map, struct offset_pair);

   util_dynarray_foreach(branches, struct pending_branch, br) {
      *fail_offset = br->in_offset;
      /* The map is in ascending agx3c order; find the target boundary. */
      unsigned lo = 0, hi = count;
      while (lo < hi) {
         const unsigned mid = lo + (hi - lo) / 2;
         if ((int64_t)pairs[mid].in < br->in_target)
            lo = mid + 1;
         else
            hi = mid;
      }
      if (br->in_target < 0 || lo == count ||
          (int64_t)pairs[lo].in != br->in_target) {
         *why = "branch target is not an instruction boundary of the program";
         return false;
      }

      const int64_t displacement =
         (int64_t)pairs[lo].out - (int64_t)br->out_offset;
      uint64_t fields[AGX3_MAX_FIELDS];
      memcpy(fields, br->fields, sizeof(fields));
      fields[0] = (uint64_t)displacement & DISPLACEMENT_MASK;
      uint8_t bytes[2 * AGX3_MAX_WORDS];
      if (!agx3_compose(&agx3d_table, br->form, fields, bytes)) {
         *why = "branch displacement has no agx3d encoding";
         return false;
      }
      memcpy((uint8_t *)out->data + br->out_offset, bytes, G18G_BRANCH_SIZE);
   }
   return true;
}

static bool
transcode(struct util_dynarray *binary, unsigned start,
          const struct agx3_transcode_ioto *ioto,
          unsigned *offsets, unsigned offset_count,
          const bool *fragment_stage, bool long_tile_reads, const char **why,
          unsigned *fail_offset)
{
   const char *unused_why;
   unsigned unused_offset;
   if (!why)
      why = &unused_why;
   if (!fail_offset)
      fail_offset = &unused_offset;
   *why = NULL;
   *fail_offset = 0;

   if (start > binary->size) {
      *why = "start past the end of the binary";
      return false;
   }

   const unsigned len = binary->size - start;
   const uint8_t *in = (const uint8_t *)binary->data + start;
   struct util_dynarray out, map, branches;
   util_dynarray_init(&out, NULL);
   util_dynarray_init(&map, NULL);
   util_dynarray_init(&branches, NULL);
   struct flag_state flags = {.pending = -1, .adjacent = true};
   bool helper_base_next = false;

   unsigned i = 0;
   while (i < len) {
      const char *display = NULL;
      const signed n = agx3c_instr_length(in + i, len - i, &display);
      *fail_offset = i;
      if (n <= 0) {
         *why = "not an agx3c instruction";
         goto fail;
      }

      char mnemonic[32];
      trim_display(display, mnemonic, sizeof(mnemonic));

      uint64_t fields[AGX3_MAX_FIELDS];
      int32_t form =
         agx3_decode(&agx3c_table, in + i, (unsigned)n, mnemonic, fields);
      /* As the witness-vector generator does: the display name is only a
       * preference, some forms print under another name. */
      if (form < 0)
         form = agx3_decode(&agx3c_table, in + i, (unsigned)n, NULL, fields);
      if (form < 0) {
         *why = "agx3c bytes do not decode to a field vector";
         goto fail;
      }
      /* Encode by the table's own name for the decoded form. */
      const char *name = agx3c_table.forms[form].mnemonic;
      if (helper_base_next) {
         /* The SR14 setup sequence indexes the USC-base helper table.
          * G18's AGX3D tables start at 0x200/0x400, twice the G17 bases.
          * Match this sequence rather than rewriting arbitrary immediates. */
         if (strcmp(name, "ldimm") || fields[2] != 0 ||
             (fields[14] != 0x100 && fields[14] != 0x200)) {
            *why = "unrecognized Ioto SR14 helper base";
            goto fail;
         }
         fields[14] *= 2;
      }
      helper_base_next = ioto && !strcmp(name, "rld") && fields[21] == 14;
      if (!apply_g18g_rules(name, fields, ioto, fragment_stage, why) ||
          !apply_flag_slot_rule(name, fields, &flags, why))
         goto fail;
      /* Native1130 (run 1139): 22 of the 26 whole-pixel reads of Apple's
       * single-sample mixed-format fragments take the long LLDCol form with
       * f79 = 2, inside the fragment's 0x200 entry and final release; the
       * packer validated every read of such a program. */
      if (long_tile_reads && !strcmp(name, "lldcol") && fields[51] == 1 &&
          fields[43] == 1)
         fields[79] = 2;

      util_dynarray_append(&map, ((struct offset_pair){i, out.size}));

      uint8_t bytes[2 * AGX3_MAX_WORDS];
      int32_t dform;
      const bool branch = !strcmp(name, "b") || !strcmp(name, "bn");
      if (branch) {
         /* Encode with a zero displacement now, the real one once every
          * target has its agx3d offset. */
         struct pending_branch br = {
            .out_offset = out.size,
            .in_offset = i,
            .in_target = (int64_t)i + sign_extend_48(fields[0]),
            .form = agx3_form_index(&agx3d_table, name, G18G_BRANCH_EP),
         };
         fields[0] = 0;
         memcpy(br.fields, fields, sizeof(br.fields));
         dform = br.form;
         if (dform < 0 || agx3d_table.forms[dform].size != G18G_BRANCH_SIZE ||
             !agx3_compose(&agx3d_table, dform, fields, bytes)) {
            *why = "branch has no ten-byte agx3d encoding";
            goto fail;
         }
         util_dynarray_append(&branches, br);
      } else {
         dform = agx3_compose_apple_order(&agx3d_table, name, fields, bytes);
         if (dform < 0) {
            *why = "field vector has no agx3d encoding";
            goto fail;
         }
      }
      const unsigned size = agx3d_table.forms[dform].size;
      if (agx3d_instr_length(bytes, size, NULL) != (signed)size) {
         *why = "agx3d encoding disagrees with the agx3d length rule";
         goto fail;
      }
      memcpy(util_dynarray_grow_bytes(&out, 1, size), bytes, size);
      i += n;
   }

   /* A branch may also land on the end of the program. */
   util_dynarray_append(&map, ((struct offset_pair){len, out.size}));
   *fail_offset = len;
   if (flags.pending >= 0)
      flags.adjacent = false;
   if (flags.folded && !flags.adjacent) {
      *why = "flag slot above 1 while a predicate lives across instructions";
      goto fail;
   }
   if (!relocate_branches(&out, &map, &branches, why, fail_offset))
      goto fail;

   /* Validate every marker before changing any caller-owned offsets. The
    * packer's colour entry is used to remove duplicate linked entries and
    * must refer to the re-encoded instruction, not its old AGX3C address. */
   for (unsigned j = 0; j < offset_count; ++j) {
      bool found = false;
      util_dynarray_foreach(&map, struct offset_pair, pair)
         found |= pair->in == offsets[j];
      if (!found) {
         *why = "shader-part marker is not an instruction boundary";
         *fail_offset = offsets[j];
         goto fail;
      }
   }
   for (unsigned j = 0; j < offset_count; ++j) {
      util_dynarray_foreach(&map, struct offset_pair, pair) {
         if (pair->in == offsets[j]) {
            offsets[j] = pair->out;
            break;
         }
      }
   }

   binary->size = start;
   memcpy(util_dynarray_grow_bytes(binary, 1, out.size), out.data, out.size);
   util_dynarray_fini(&out);
   util_dynarray_fini(&map);
   util_dynarray_fini(&branches);
   return true;

fail:
   util_dynarray_fini(&out);
   util_dynarray_fini(&map);
   util_dynarray_fini(&branches);
   return false;
}

bool
agx3_transcode_agx3d(struct util_dynarray *binary, unsigned start,
                     const char **why, unsigned *fail_offset)
{
   return transcode(binary, start, NULL, NULL, 0, NULL, false, why,
                    fail_offset);
}

bool
agx3_transcode_agx3d_tile_reads(struct util_dynarray *binary, unsigned start,
                                unsigned *offsets, unsigned count,
                                bool fragment_stage, bool long_tile_reads,
                                const char **why, unsigned *fail_offset)
{
   return transcode(binary, start, NULL, offsets, count, &fragment_stage,
                    long_tile_reads, why, fail_offset);
}

bool
agx3_transcode_agx3d_ioto(struct util_dynarray *binary, unsigned start,
                          const struct agx3_transcode_ioto *ioto,
                          const char **why, unsigned *fail_offset)
{
   return ioto && transcode(binary, start, ioto, NULL, 0, NULL, false, why,
                            fail_offset);
}

bool
agx3_transcode_agx3d_ioto_with_offsets(struct util_dynarray *binary,
                                       unsigned start,
                                       const struct agx3_transcode_ioto *ioto,
                                       unsigned *offsets, unsigned count,
                                       const char **why, unsigned *fail_offset)
{
   return ioto && transcode(binary, start, ioto, offsets, count, NULL, false,
                            why, fail_offset);
}

/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * End-to-end compile coverage for the AGX3 graphics stages.
 *
 * The AGX3 compute paths are covered by test-agx3-compile-fail-closed.c. This
 * file is the vertex/fragment counterpart, starting from a flat-shaded
 * triangle -- the smallest program that is still a real draw.
 *
 * The NIR here is written at the *post-lowering* level, i.e. what the backend
 * actually sees. agx_nir_lower_uvs() and agx_nir_lower_tilebuffer() live in
 * src/asahi/lib, which is built after the compiler and therefore cannot be
 * linked from a compiler test; store_uvs_agx / store_local_pixel_agx below are
 * exactly the intrinsics those two passes emit (agx_nir_lower_uvs.c:108,
 * agx_nir_lower_tilebuffer.c:75).
 *
 * Cases marked "expect reject" are gaps that remain fail-closed in
 * agx3_pack_binary(). They are asserted as rejections so that grounding one
 * later is a deliberate change to this table, not a silent behaviour drift.
 */

#include "agx_compile.h"
#include "agx_debug.h"
#include "poly/nir/poly_nir.h"
#include "nir_xfb_info.h"
#include "agx3_alu_core.h"
#include "agx3_copy_core.h"
#include "agx3_funop_core.h"
#include "agx3_isel_core.h"
#include "agx3_sr_core.h"
#include "agx3_vary_core.h"
#include "agx3_stage_core.h"
#include "agx3_zs_core.h"
#include "agx3_tex_core.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "asahi/lib/agx_abi.h"
#include "nir_builder.h"
#include "nir_builtin_builder.h"
#include "asahi/libagx/libagx.h"
#include "util/format/u_formats.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint8_t agx3_end[] = {0x0e, 0x00, 0x00, 0x00};

static bool g17p_sample_mask_input_protocol(
   const struct agx_shader_part *part);

/* Apple G17P's byte-exact immediate colour packing followed by the deferred
 * colour-write fence and normalized RGBA8 store. */
static const uint8_t g17p_flat_colour[] = {
   0x97, 0x0c, 0x54, 0x00, 0x02, 0x20, 0x80, 0x50, 0x04, 0xc8,
   0x97, 0x04, 0x54, 0x01, 0x02, 0x50, 0xc0, 0x50, 0x04, 0xc8,
   0x87, 0x02, 0x54, 0x00, 0x06, 0x00, /* deferred colour entry */
   0x87, 0x02, 0x54, 0x0c, 0x08, 0x00, /* colour wait */
   0xe7, 0x06, 0x54, 0x00, 0x00, 0x00, 0x01, 0x4e, 0x00, 0x00, 0x00, 0x00,
   0x07, 0x02, 0x54, 0x0c, 0x02, 0x00, /* final colour release */
   0x0e, 0x00, 0x00, 0x00,
};

/* Apple's complete G17P float2-position VS. UVSW operands name the separate
 * staging file, not general registers. The dataflow check below requires a
 * real producer in that file before accepting each position write. */
static const uint8_t apple_g17p_float2_vs[] = {
   0x0c, 0xdd, 0x10, 0x06,
   0x67, 0x10, 0x44, 0x00, 0x00, 0x80, 0x20, 0x00, 0x59, 0x01, 0x00, 0x40,
   0x48, 0x00,
   0x3b, 0x00, 0x29, 0xc4,
   0x2b, 0x03, 0x2f, 0x00, 0x42, 0x00, 0x00, 0x04, 0x00, 0x00,
   0x1b, 0x00, 0x26, 0x00, 0x40, 0x00, 0x00, 0x08, 0x00, 0x00,
   0x0c, 0x80, 0x23, 0x3e, 0x00, 0x00, 0x00, 0x0c, 0x40, 0x60,
   0x57, 0x16, 0x54, 0x06, 0x00, 0x40, 0x48, 0x00,
   0x57, 0x26, 0x54, 0x04, 0x20, 0x40, 0x49, 0x00,
   0x57, 0x46, 0x54, 0x02, 0x40, 0x40, 0x4a, 0x00,
   0x57, 0x86, 0x54, 0x00, 0x60, 0x40, 0x4b, 0x00,
   0x0e, 0x00, 0x00, 0x00,
};

static const uint8_t mesa_unstaged_g17p_float2_vs[] = {
   0x0c, 0xdd, 0x10, 0x06,
   0x67, 0x10, 0x44, 0x00, 0x00, 0x80, 0x20, 0x00, 0x59, 0x01, 0x00, 0x40,
   0x48, 0x00,
   0x57, 0x16, 0x54, 0x00, 0x00, 0x40, 0x48, 0x00,
   0x57, 0x26, 0x54, 0x02, 0x20, 0x40, 0x49, 0x00,
   0x0b, 0x00, 0x00, 0x00,
   0x57, 0x46, 0x54, 0x00, 0x40, 0x40, 0x4a, 0x00,
   0x0c, 0x80, 0x02, 0x3e, 0x00, 0x00, 0x00, 0x0c,
   0x57, 0x86, 0x54, 0x00, 0x60, 0x40, 0x4b, 0x00,
   0x0e, 0x00, 0x00, 0x00,
};

static const uint8_t mesa_g17p_float2_vs[] = {
   0x0c,0xdd,0x10,0x06,0x67,0x10,0x44,0x00,0x00,0x80,0x20,0x00,0x59,0x01,0x00,0x40,
   0x48,0x00,0x0b,0x00,0x21,0xc4,
   0x1b,0x03,0x27,0x00,0x42,0x00,0x00,0x04,0x00,0x00,
   0x0b,0x00,0x00,0x00,
   0x2b,0x01,0x27,0x00,0x42,0x00,0x00,0x08,0x00,0x00,
   0x0c,0x80,0x02,0x3e,0x00,0x00,0x00,0x0c,
   0x3b,0x01,0x27,0x00,0x42,0x00,0x00,0x0c,0x00,0x00,
   0x57,0x16,0x54,0x00,0x00,0x40,0x48,0x00,
   0x57,0x26,0x54,0x02,0x20,0x40,0x49,0x00,
   0x57,0x46,0x54,0x04,0x40,0x40,0x4a,0x00,
   0x57,0x86,0x54,0x06,0x60,0x40,0x4b,0x00,0x0e,0x00,0x00,0x00,
};

/* Exact post-lowering R16G16_SFLOAT vertex path.  The two six-byte f32add
 * instructions at +0x12/+0x18 are the corpus-grounded half-to-float widens;
 * both preserve their raw LDDEV halves through separate UVS exports. */
static const uint8_t mesa_g17p_half2_varying_vs[] = {
   0x0c,0xdd,0x10,0x06,0x67,0x10,0x54,0x02,0x01,0x00,0x20,0x00,0x49,0x01,0x00,0x40,
   0x44,0x00,0x29,0x02,0x1c,0x81,0x00,0xc2,0x39,0x02,0x1c,0x81,0x08,0x02,0x67,0x00,
   0x44,0x08,0x00,0x00,0x20,0x00,0x57,0x00,0x00,0x40,0x40,0x00,0x0b,0x08,0x21,0x44,
   0x1b,0x0b,0x27,0x00,0x42,0x00,0x00,0x04,0x00,0x00,0x2b,0x0d,0x27,0x00,0x42,0x00,
   0x00,0x08,0x00,0x00,0x3b,0x0f,0x27,0x00,0x42,0x00,0x00,0x0c,0x00,0x00,0x4b,0x05,
   0x27,0x00,0x42,0x00,0x00,0x10,0x00,0x00,0x5b,0x07,0x27,0x00,0x42,0x00,0x00,0x14,
   0x00,0x00,0x57,0x16,0x54,0x00,0x00,0x40,0x48,0x00,0x57,0x26,0x54,0x02,0x20,0x40,
   0x49,0x00,0x57,0x46,0x54,0x04,0x40,0x40,0x4a,0x00,0x57,0x86,0x54,0x06,0x60,0x40,
   0x4b,0x00,0x57,0x06,0x55,0x08,0x80,0x40,0x4c,0x00,0x57,0x06,0x56,0x0a,0xa0,0x40,
   0x4d,0x00,0x0e,0x00,0x00,0x00,
};

static bool
g17p_vs_position_dataflow(const uint8_t *binary, size_t size)
{
   if (size < 22 || memcmp(binary, apple_g17p_float2_vs, 18))
      return false;

   /* Symbolic values: 1=x, 2=y, 3=zero, 4=one. Decode only the grounded
    * producer forms occurring in these two isolated float2 shaders. Unlike
    * the former normalizer, this cannot mistake an unwritten staging slot
    * for a general register with the same numeric index. */
   uint8_t gpr[256] = {1, 0, 2}, staging[256] = {0};
   uint8_t staging_slot[256] = {0};
   bool attribute_ready = false;
   unsigned writes = 0, offset = 18;
   while (offset + 4 <= size) {
      const uint8_t *p = binary + offset;
      unsigned length = 0;
      const unsigned dest = (p[0] >> 4) * 2;
      if (p[0] == 0x0e && p[1] == 0 && p[2] == 0 && p[3] == 0)
         return writes == 4 && offset + 4 == size;
      if ((p[0] & 0xf) == 0xb &&
          ((p[2] & 0xf7) == 0x21 || (p[2] & 0xf7) == 0x27)) {
         length = (p[2] & 7) == 1 ? 4 : 10;
         if (length > size - offset)
            return false;
         /* Compact COPY has an enumerated one-hot wait and implicit slot1;
          * extended COPY has six independent wait bits and explicit f12. */
         const unsigned wait_index = p[3] >> 5;
         const unsigned wait = length == 4
                                  ? (wait_index ? 1u << (wait_index - 1) : 0)
                                  : (p[5] >> 5) | ((p[7] >> 5) << 3);
         attribute_ready |= (wait & 0x20) != 0;
         const unsigned src = p[1] & 0xfe;
         if ((gpr[src] == 1 || gpr[src] == 2) && !attribute_ready)
            return false;
         staging[dest] = gpr[src];
         staging_slot[dest] = length == 4 ? 1 : 1 + ((p[7] >> 2) & 7);
      } else if ((p[0] & 0xf) == 0xb && p[1] == 0 && p[2] == 0 &&
                 p[3] == 0) {
         length = 4;
         gpr[dest] = 3;
      } else if ((p[0] & 0xf) == 0xb && p[1] == 0 && p[2] == 0x26) {
         length = 10;
         if (length > size - offset)
            return false;
         staging[dest] = 3;
         staging_slot[dest] = 1 + ((p[7] >> 2) & 7);
      } else if ((p[0] & 0xf) == 0xc && p[1] == 0x80 &&
                 (p[2] == 2 || p[2] == 0x23) && p[3] == 0x3e) {
         length = p[2] == 2 ? 8 : 10;
         if (length > size - offset)
            return false;
         if (p[2] == 2)
            gpr[dest] = 4;
         else {
            staging[dest] = 4;
            staging_slot[dest] = 1 + (p[9] >> 5);
         }
      } else if (p[0] == 0x57 && p[2] == 0x54 && offset + 8 <= size) {
         length = 8;
         const unsigned component = p[4] >> 5;
         /* The matched native shape produces all four before first UVSW. */
         if (!staging[0] || !staging[2] || !staging[4] || !staging[6])
            return false;
         if (component != writes || staging[p[3]] != component + 1 ||
             (p[1] >> 4) != (1u << writes) ||
             staging_slot[p[3]] != 1 + (p[6] & 7) ||
             staging_slot[p[3]] != writes + 1)
            return false;
         writes++;
      } else {
         return false;
      }
      if (length > size - offset)
         return false;
      offset += length;
   }
   return false;
}

static bool
g17p_vs_matches_apple(const uint8_t *binary, size_t size)
{
   uint8_t bad_wait[sizeof(mesa_g17p_float2_vs)];
   memcpy(bad_wait, mesa_g17p_float2_vs, sizeof(bad_wait));
   bad_wait[21] = 4; /* staging COPY no longer waits for attribute LDDEV */
   uint8_t bad_bank[sizeof(mesa_g17p_float2_vs)];
   memcpy(bad_bank, mesa_g17p_float2_vs, sizeof(bad_bank));
   bad_bank[20] &= ~0x20; /* destination no longer selects staging */
   return !g17p_vs_position_dataflow(bad_wait, sizeof(bad_wait)) &&
          !g17p_vs_position_dataflow(bad_bank, sizeof(bad_bank)) &&
          g17p_vs_position_dataflow(apple_g17p_float2_vs,
                                    sizeof(apple_g17p_float2_vs)) &&
          !g17p_vs_position_dataflow(mesa_unstaged_g17p_float2_vs,
                                     sizeof(mesa_unstaged_g17p_float2_vs)) &&
          g17p_vs_position_dataflow(binary, size);
}

/* In 71 fully decoded AGX3b vertex programs, f79 selects byte2=0x44 on the
 * final LDDEV (and only the final LDDEV), while the first LDDEV always uses
 * scoreboard slot 6. Locate the compiler's 14-byte device-load forms and pin
 * those two corpus invariants without constraining allocation-only fields. */
static bool
g17p_vertex_load_policy(const uint8_t *binary, size_t size,
                        unsigned expected_loads)
{
   const uint8_t *loads[8];
   unsigned count = 0;

   for (unsigned i = 0; i + 14 <= size; ++i) {
      const uint8_t *p = binary + i;
      if (p[0] != 0x67 || (p[2] != 0x44 && p[2] != 0x54) ||
          p[6] != 0x20 || p[11] != 0x40 || p[13] != 0)
         continue;
      if (count == ARRAY_SIZE(loads))
         return false;
      loads[count++] = p;
      i += 13;
   }

   if (count != expected_loads)
      return false;

   for (unsigned i = 0; i < count; ++i) {
      const bool terminal = loads[i][2] == 0x44;
      if (terminal != (i + 1 == count))
         return false;
   }

   const unsigned first_slot =
      1 + (loads[0][8] >> 6) + 4 * (loads[0][9] & 1);
   return first_slot == 6;
}

/* AIL_ISA_FORMAT_U8NORM: the tilebuffer format for an RGBA8_UNORM target. */
#define AGX3_TEST_TIB_FORMAT PIPE_FORMAT_R8_UNORM

static const char *
enc_name(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3B ? "AGX3b" : "AGX3c";
}

/*
 * Minimal vertex shader: gl_Position = vec4(float(gl_VertexID), 0, 0, 1),
 * written to the UVS the way agx_nir_lower_uvs lowers a store_output of
 * VARYING_SLOT_POS with a single-varying layout (base index 0).
 */
static nir_shader *
build_vs_flat(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 flat triangle VS");

   nir_def *id = nir_u2f32(&b, nir_load_vertex_id(&b));
   nir_def *pos[4] = {
      id,
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 1.0f),
   };

   for (unsigned c = 0; c < 4; ++c)
      nir_store_uvs_agx(&b, pos[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written = VARYING_BIT_POS;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static nir_shader *
build_vs_g16_imad_negative_byte(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G16 register multiply minus byte");
   nir_def *a = nir_load_vertex_id(&b);
   nir_def *c = nir_load_instance_id(&b);
   nir_def *product = nir_imul(&b, a, c);
   nir_def *pos[4] = {
      nir_i2f32(&b, nir_iadd_imm(&b, product, -1)),
      nir_i2f32(&b, nir_iadd_imm(&b,
         nir_imul(&b, a, nir_iadd_imm(&b, c, 1)), -255)),
      nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f),
   };
   for (unsigned i = 0; i < 4; ++i)
      nir_store_uvs_agx(&b, pos[i], nir_imm_intN_t(&b, i, 16));
   b.shader->info.outputs_written = VARYING_BIT_POS;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static nir_shader *
build_vs_g17p_div3_mul_high(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 div3 mul-high VS");
   nir_def *id = nir_load_vertex_id(&b);
   nir_def *high =
      nir_imul_high(&b, id, nir_imm_int(&b, UINT32_C(0x55555556)));
   nir_def *pos[4] = {
      nir_i2f32(&b, high),
      nir_u2f32(&b, id),
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 1.0f),
   };

   for (unsigned c = 0; c < 4; ++c)
      nir_store_uvs_agx(&b, pos[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written = VARYING_BIT_POS;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* glmark2's ideas vertex shader computes its point size with a constant-first
 * comparison: 100.0 < size ? scaled_size : 0.0. Keep the minifloat compare
 * order and mixed register/immediate selection in an end-to-end shader. */
static nir_shader *
build_vs_glmark2_ideas(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 glmark2 ideas VS");

   nir_def *size = nir_u2f32(&b, nir_load_vertex_id(&b));
   nir_def *scaled = nir_fmul_imm(&b, nir_fadd_imm(&b, size, -10.0f), 0.5f);
   nir_def *point_size = nir_bcsel(
      &b, nir_flt(&b, nir_imm_float(&b, 100.0f), size), scaled,
      nir_imm_float(&b, 0.0f));
   nir_def *outputs[] = {
      size, nir_imm_float(&b, 0.0f), nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 1.0f), point_size,
   };

   for (unsigned c = 0; c < ARRAY_SIZE(outputs); ++c)
      nir_store_uvs_agx(&b, outputs[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written = VARYING_BIT_POS | VARYING_BIT_PSIZ;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* Instance ID uses its own G17P special register. Keep a complete shader case
 * so the end-to-end compiler, not only the standalone rld encoder, proves the
 * hardware-vertex lowering. */
static nir_shader *
build_vs_instance_id(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 instance ID VS");

   nir_def *id = nir_u2f32(&b, nir_load_instance_id(&b));
   nir_def *pos[4] = {
      id,
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 1.0f),
   };

   for (unsigned c = 0; c < 4; ++c)
      nir_store_uvs_agx(&b, pos[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written = VARYING_BIT_POS;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/*
 * Vertex shader with a real attribute fetch: agx_nir_lower_vbo turns
 * load_input into the load_agx used here, so this is the shape a DRAW with a
 * vertex buffer actually reaches the backend with.
 */
static nir_shader *
build_vs_attribute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 attribute VS");

   nir_def *base = nir_imm_int64(&b, 0x1000);
   nir_def *index = nir_load_vertex_id(&b);
   nir_def *attr = nir_load_agx(&b, 4, 32, base, index, .access = 0, .base = 0,
                                .format = PIPE_FORMAT_R32_UINT,
                                .sign_extend = false);

   for (unsigned c = 0; c < 4; ++c) {
      nir_store_uvs_agx(&b, nir_channel(&b, attr, c),
                        nir_imm_intN_t(&b, c, 16));
   }

   b.shader->info.outputs_written = VARYING_BIT_POS;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* The live HAL200 vertex path fills uniform halfs 0..3 with the first VBO
 * address. This covers the same device load with that Ioto-populated base. */
static nir_shader *
build_vs_uniform_attribute(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 uniform-base attribute VS");

   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *index = nir_load_vertex_id(&b);
   nir_def *attr = nir_load_agx(&b, 4, 32, base, index, .access = 0, .base = 0,
                                .format = PIPE_FORMAT_R32_UINT,
                                .sign_extend = false);

   for (unsigned c = 0; c < 4; ++c) {
      nir_store_uvs_agx(&b, nir_u2f32(&b, nir_channel(&b, attr, c)),
                        nir_imm_intN_t(&b, c, 16));
   }

   b.shader->info.outputs_written = VARYING_BIT_POS;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* The exact live G17P render-only shape: one float2 VBO attribute indexed by
 * vertex_id, expanded to a four-component position with z=0 and w=1. */
static nir_shader *
build_vs_g17p_float2_position(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G17P float2 position VS");

   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *index = nir_load_vertex_id(&b);
   nir_def *attr = nir_load_agx(&b, 2, 32, base, index, .access = 0, .base = 1,
                                .format = PIPE_FORMAT_R32_UINT,
                                .sign_extend = false);
   nir_def *position[] = {
      nir_channel(&b, attr, 0),
      nir_channel(&b, attr, 1),
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 1.0f),
   };

   for (unsigned c = 0; c < ARRAY_SIZE(position); ++c)
      nir_store_uvs_agx(&b, position[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written = VARYING_BIT_POS;
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static nir_shader *
build_vs_g17p_single_instanced_varying(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options,
      "G17P single instanced load with varying VS");

   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *attr = nir_load_agx(&b, 2, 32, base, nir_load_instance_id(&b),
                                .access = 0, .base = 1,
                                .format = PIPE_FORMAT_R32_UINT,
                                .sign_extend = false);
   nir_def *values[] = {
      nir_channel(&b, attr, 0), nir_channel(&b, attr, 1),
      nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f),
      nir_channel(&b, attr, 0), nir_channel(&b, attr, 1),
   };

   for (unsigned c = 0; c < ARRAY_SIZE(values); ++c)
      nir_store_uvs_agx(&b, values[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static nir_shader *
build_vs_g17p_two_loads(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G17P two-load position VS");

   nir_def *index = nir_load_vertex_id(&b);
   nir_def *position = nir_load_agx(
      &b, 2, 32, nir_load_preamble(&b, 1, 64, .base = 0), index,
      .access = 0, .base = 1, .format = PIPE_FORMAT_R32_UINT,
      .sign_extend = false);
   nir_def *varying = nir_load_agx(
      &b, 2, 32, nir_load_preamble(&b, 1, 64, .base = 4), index,
      .access = 0, .base = 1, .format = PIPE_FORMAT_R32_UINT,
      .sign_extend = false);
   nir_def *values[] = {
      nir_channel(&b, position, 0), nir_channel(&b, position, 1),
      nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f),
      nir_channel(&b, varying, 0), nir_channel(&b, varying, 1),
   };

   for (unsigned c = 0; c < ARRAY_SIZE(values); ++c)
      nir_store_uvs_agx(&b, values[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* Post-VBO-lowering shape for a separate R16G16_SFLOAT attribute.  The raw
 * 32-bit LDDEV result carries two IEEE binary16 values; each is widened by
 * the native mixed-width f32add-with-negative-zero form before UVS export. */
static nir_shader *
build_vs_g17p_half2_varying(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G17P half2 varying VS");

   nir_def *vertex = nir_load_vertex_id(&b);
   nir_def *packed = nir_load_agx(
      &b, 1, 32, nir_load_preamble(&b, 1, 64, .base = 4), vertex,
      .access = 0, .base = 2 << AGX_NIR_MEMORY_PACKED_COUNT_SHIFT,
      .format = PIPE_FORMAT_R16_UINT, .sign_extend = false);
   nir_def *halfs[] = {
      nir_unpack_32_2x16_split_x(&b, packed),
      nir_unpack_32_2x16_split_y(&b, packed),
   };

   for (unsigned c = 0; c < ARRAY_SIZE(halfs); ++c)
      nir_store_uvs_agx(&b, nir_f2f32(&b, halfs[c]),
                        nir_imm_intN_t(&b, 4 + c, 16));

   nir_def *position = nir_load_agx(
      &b, 4, 32, nir_load_preamble(&b, 1, 64, .base = 0), vertex,
      .access = 0, .base = 2, .format = PIPE_FORMAT_R32_UINT,
      .sign_extend = false);
   for (unsigned c = 0; c < 4; ++c)
      nir_store_uvs_agx(&b, nir_channel(&b, position, c),
                        nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* Standard vkcube's vertex interface: position plus a vec4 texture coordinate
 * and a vec3 object-space position, occupying UVS words 4..10. */
static nir_shader *
build_vs_vkcube_varyings(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 vkcube varying VS");

   nir_def *id = nir_u2f32(&b, nir_load_vertex_id(&b));
   nir_def *values[11] = {
      id,
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 1.0f),
      id,
      nir_imm_float(&b, 0.25f),
      nir_imm_float(&b, 0.5f),
      nir_imm_float(&b, 1.0f),
      id,
      nir_imm_float(&b, 0.0f),
      nir_imm_float(&b, 1.0f),
   };

   for (unsigned c = 0; c < ARRAY_SIZE(values); ++c)
      nir_store_uvs_agx(&b, values[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0) |
      BITFIELD64_BIT(VARYING_SLOT_VAR1);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* Apple G17P accepts position plus eight float4 user locations: its vs36
 * pipeline writes UVS words 0..35 and __GPU_LD_MD field0 is exactly 36.
 * Keep the complete backend writeout path covered at that native limit. */
static nir_shader *
build_vs_g17p_36_outputs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G17P 36-word vertex output VS");
   nir_def *vertex = nir_load_vertex_id(&b);

   for (unsigned group = 0; group < 9; ++group) {
      nir_def *values = nir_load_agx(
         &b, 4, 32, nir_imm_int64(&b, UINT64_C(0x1000) + group * 16),
         vertex, .access = 0, .base = 0, .format = PIPE_FORMAT_R32_UINT,
         .sign_extend = false);
      for (unsigned c = 0; c < 4; ++c) {
         nir_store_uvs_agx(&b, nir_channel(&b, values, c),
                           nir_imm_intN_t(&b, group * 4 + c, 16));
      }
   }

   b.shader->info.outputs_written =
      VARYING_BIT_POS |
      (BITFIELD64_MASK(8) << VARYING_SLOT_VAR0);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/* The bounded live vkcube vertex shape: position and texture coordinates come
 * from two VBO CProg bases, while set 0 binding 0 supplies a column-major MVP
 * matrix through the descriptor-set pointer following the root pointer.
 */
static nir_shader *
build_vs_vkcube_vbo_ubo(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 vkcube VBO plus UBO VS");

   const unsigned nr_vbos = 2;
   const unsigned root = AGX_ABI_VUNI_COUNT_VK(nr_vbos);
   nir_def *vertex = nir_load_vertex_id(&b);
   nir_def *position = nir_load_agx(
      &b, 4, 32,
      nir_load_preamble(&b, 1, 64, .base = AGX_ABI_VUNI_VBO_BASE(0)), vertex,
      .access = 0, .base = 0, .format = PIPE_FORMAT_R32_UINT,
      .sign_extend = false);
   nir_def *texcoord = nir_load_agx(
      &b, 4, 32,
      nir_load_preamble(&b, 1, 64, .base = AGX_ABI_VUNI_VBO_BASE(1)), vertex,
      .access = 0, .base = 0, .format = PIPE_FORMAT_R32_UINT,
      .sign_extend = false);

   nir_def *set = nir_load_preamble(&b, 1, 64, .base = root + 4);
   nir_def *descriptor = nir_load_global_constant(
      &b, 4, 32, set, .align_mul = 16, .access = ACCESS_CAN_SPECULATE);
   nir_def *ubo = nir_pack_64_2x32_split(
      &b, nir_channel(&b, descriptor, 0), nir_channel(&b, descriptor, 1));

   nir_def *column[4];
   for (unsigned c = 0; c < ARRAY_SIZE(column); ++c) {
      column[c] = nir_load_global_constant(
         &b, 4, 32, nir_iadd_imm(&b, ubo, c * 16), .align_mul = 16,
         .access = ACCESS_CAN_SPECULATE);
   }

   nir_def *transformed[4];
   for (unsigned row = 0; row < ARRAY_SIZE(transformed); ++row) {
      transformed[row] = nir_ffma(
         &b, nir_channel(&b, position, 0), nir_channel(&b, column[0], row),
         nir_ffma(
            &b, nir_channel(&b, position, 1),
            nir_channel(&b, column[1], row),
            nir_ffma(&b, nir_channel(&b, position, 2),
                     nir_channel(&b, column[2], row),
                     nir_fmul(&b, nir_channel(&b, position, 3),
                              nir_channel(&b, column[3], row)))));
   }

   for (unsigned c = 0; c < 4; ++c) {
      nir_store_uvs_agx(&b, transformed[c], nir_imm_intN_t(&b, c, 16));
      nir_store_uvs_agx(&b, nir_channel(&b, texcoord, c),
                        nir_imm_intN_t(&b, 4 + c, 16));
   }
   for (unsigned c = 0; c < 3; ++c)
      nir_store_uvs_agx(&b, transformed[c], nir_imm_intN_t(&b, 8 + c, 16));

   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0) |
      BITFIELD64_BIT(VARYING_SLOT_VAR1);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

/*
 * The FS lowering passes introduce driver-owned system values that every
 * driver resolves itself (d12_dxil_lower_root_mask / hk_lower_uvs_index and
 * friends). Resolve them to the monolithic-draw constants so the test
 * exercises the backend rather than a driver ABI.
 */
static bool
lower_driver_sysvals(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *_)
{
   uint16_t value;

   switch (intr->intrinsic) {
   case nir_intrinsic_load_api_sample_mask_agx:
      value = 0xFF;
      break;
   case nir_intrinsic_load_shader_part_tests_zs_agx:
      value = 0;
      break;
   case nir_intrinsic_load_active_samples_agx:
      value = 0xFF;
      break;
   default:
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_intN_t(b, value, 16));
   return true;
}

static void
fs_lower(nir_shader *s)
{
   NIR_PASS(_, s, agx_nir_lower_discard_zs_emit);
   NIR_PASS(_, s, agx_nir_lower_sample_mask);
   NIR_PASS(_, s, agx_nir_lower_interpolation, false);
   NIR_PASS(_, s, nir_shader_intrinsics_pass, lower_driver_sysvals,
            nir_metadata_control_flow, NULL);
}

static void
fs_store_colour(nir_builder *b, nir_def *colour)
{
   nir_store_local_pixel_agx(b, colour, nir_imm_intN_t(b, 0xFF, 16),
                             nir_undef(b, 2, 16), .base = 0, .write_mask = 0xf,
                             .format = AGX3_TEST_TIB_FORMAT);
}

static nir_shader *
fs_finish(nir_builder *b)
{
   b->shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_preprocess_nir(b->shader);

   fs_lower(b->shader);
   return b->shader;
}

static unsigned
count_sample_mask(nir_shader *s)
{
   unsigned count = 0;

   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_sample_mask_agx)
               count++;
         }
      }
   }

   return count;
}

static bool
test_g17p_frag_sidefx_late_depth_once(void)
{
   for (unsigned early = 0; early < 2; ++early) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P late depth feedback");
      b.shader->info.internal = true;
      fs_store_colour(&b, nir_load_preamble(&b, 4, 32, .base = 4));
      b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
      b.shader->info.writes_memory = true;
      b.shader->info.fs.early_fragment_tests = early;
      for (unsigned pass = 0; pass < 2; ++pass) {
         agx_preprocess_nir_g17p(b.shader);
         unsigned depth_writes = 0;
         nir_foreach_function_impl(impl, b.shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  if (instr->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic == nir_intrinsic_store_output &&
                      nir_intrinsic_io_semantics(intr).location == FRAG_RESULT_DEPTH) {
                     depth_writes++;
                     if (nir_src_bit_size(intr->src[0]) != 32) {
                        ralloc_free(b.shader);
                        return false;
                     }
                  }
               }
            }
         }
         if (depth_writes != !early || count_sample_mask(b.shader) != 0) {
            ralloc_free(b.shader);
            return false;
         }
      }
      ralloc_free(b.shader);
   }
   return true;
}

static bool
test_g17p_background_clear(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P RT0 background clear");
   b.shader->info.internal = true;
   fs_store_colour(&b, nir_load_preamble(&b, 4, 32, .base = 4));
   nir_shader *nir = fs_finish(&b);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.ignore_tib_dependencies = true,
      .reserved_preamble = 12,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(nir, &key, &part);
   const bool ok = compiled && part.binary != NULL &&
                   part.info.binary_size > sizeof(agx3_end) &&
                   part.info.push_count == key.reserved_preamble &&
                   !part.info.has_preamble &&
                   memcmp((const uint8_t *)part.binary +
                             part.info.binary_size - sizeof(agx3_end),
                          agx3_end, sizeof(agx3_end)) == 0;
   if (!ok)
      fprintf(stderr, "AGX3b background-clear compiler mismatch\n");

   free(part.binary);
   ralloc_free(nir);
   return ok;
}

static nir_shader *build_fs_alpha_discard_word(void);
static nir_shader *build_fs_kill_then_alpha_discard_word(void);
static nir_shader *build_fs_constant_target_alpha_discard(uint16_t target);

/* At one sample both non-constant coverage sources
 * are reduced to bit 0 (FLAGS = 3); at four samples the whole masks are kept
 * (FLAGS = 0). The first, no-tests update of the kill-then-discard shape
 * keeps its constant live source, so only its target is reduced (FLAGS = 1).
 * The binaries' coverage protocol is checked by the fs-*-discard-word cases. */
static bool
g17p_discard_word_case(nir_shader *(*build)(void), unsigned nr_samples,
                       const unsigned *expected_flags, unsigned count)
{
   nir_shader *nir = build();
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
      .fs.nr_samples = nr_samples,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(nir, &key, &part);

   unsigned seen = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_sample_mask_agx)
               continue;
            const unsigned want =
               (seen < count && nr_samples <= 1) ? expected_flags[seen] : 0;
            ok &= nir_intrinsic_flags(intr) == want;
            ++seen;
         }
      }
   }
   ok &= seen == count;

   if (!ok)
      fprintf(stderr, "AGX3b discard word mismatch (%ux, %u updates)\n",
              MAX2(nr_samples, 1), seen);
   free(part.binary);
   ralloc_free(nir);
   return ok;
}

static bool
test_g17p_discard_word(void)
{
   static const unsigned one[] = {3}, two[] = {1, 3};
   if (!g17p_discard_word_case(build_fs_alpha_discard_word, 1, one, 1) ||
       !g17p_discard_word_case(build_fs_kill_then_alpha_discard_word, 1, two, 2))
      return false;

   /* Constant target 0xffff at one sample passes immediate 1 directly and
    * zero-extends the coverage word instead of materializing 0xffff and
    * multiplying by a dynamic target. Verify it compiles cleanly and emits
    * fewer instructions than the dynamic-target form, while target 0 and
    * multisample forms still compile. */
   nir_shader *const_nir = build_fs_constant_target_alpha_discard(0xffff);
   nir_shader *dyn_nir = build_fs_alpha_discard_word();
   nir_shader *zero_nir = build_fs_constant_target_alpha_discard(0);
   nir_shader *msaa_nir = build_fs_constant_target_alpha_discard(0xffff);
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
      .fs.nr_samples = 1,
   };
   struct agx_shader_key msaa_key = key;
   msaa_key.fs.nr_samples = 4;

   struct agx_shader_part const_part = {0}, dyn_part = {0}, zero_part = {0},
      msaa_part = {0};
   bool ok = agx_compile_shader_nir(const_nir, &key, &const_part) &&
             agx_compile_shader_nir(dyn_nir, &key, &dyn_part) &&
             agx_compile_shader_nir(zero_nir, &key, &zero_part) &&
             agx_compile_shader_nir(msaa_nir, &msaa_key, &msaa_part) &&
             const_part.info.stats.instrs + 3 <= dyn_part.info.stats.instrs &&
             const_part.info.binary_size < dyn_part.info.binary_size;
   if (!ok) {
      fprintf(stderr,
              "AGX3b constant-target discard mismatch: const=%u instrs/%u B "
              "dyn=%u instrs/%u B\n",
              const_part.info.stats.instrs, const_part.info.binary_size,
              dyn_part.info.stats.instrs, dyn_part.info.binary_size);
   }
   free(const_part.binary);
   free(dyn_part.binary);
   free(zero_part.binary);
   free(msaa_part.binary);
   ralloc_free(const_nir);
   ralloc_free(dyn_nir);
   ralloc_free(zero_nir);
   ralloc_free(msaa_nir);
   return ok;
}

static bool
test_g17p_background_f16_clear_narrowing(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P f16 background clear");
   b.shader->info.internal = true;
   nir_store_local_pixel_agx(
      &b, nir_load_preamble(&b, 4, 32, .base = 4),
      nir_imm_intN_t(&b, 0xff, 16), nir_undef(&b, 2, 16), .base = 0,
      .write_mask = 0xf, .format = PIPE_FORMAT_R16_FLOAT);
   nir_shader *nir = fs_finish(&b);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.ignore_tib_dependencies = true,
      .reserved_preamble = 12,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(nir, &key, &part);
   static const uint8_t expected[] =
      "\x0b\x04\x01\x08\x01\x01\x1c\x81\x00\x02"
      "\x1b\x06\x01\x08\x01\x03\x1c\x81\x04\x02"
      "\x1b\x08\x01\x08\x11\x03\x1c\x81\x00\x02"
      "\x2b\x0a\x01\x08\x11\x05\x1c\x81\x04\x02"
      "\xe7\x06\x54\x00\x00\x00\x01\x0e\x00\x00\x00\x00"
      "\x0e\x00\x00\x00";
   bool ok = compiled && part.binary != NULL &&
             part.info.binary_size == sizeof(expected) - 1 &&
             !memcmp(part.binary, expected, sizeof(expected) - 1);
   if (!ok)
      fprintf(stderr, "AGX3b f16 background-clear narrowing mismatch\n");

   free(part.binary);
   ralloc_free(nir);

   return ok;
}

static nir_shader *
build_half_normalized_epilog(enum pipe_format format)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P half ABI normalized epilog");
   nir_def *colour = nir_load_exported_agx(
      &b, 4, 16, .base = AGX_ABI_FOUT_COLOUR(0));
   nir_store_local_pixel_agx(
      &b, colour, nir_imm_intN_t(&b, 0xff, 16), nir_undef(&b, 2, 16),
      .base = 0, .write_mask = 0xf, .format = format);
   return fs_finish(&b);
}

static bool
test_g17p_half_norm8_epilog(void)
{
   static const uint8_t expected[] = {
      0x09,0x0a,0x1c,0x81,0x08,0x02,
      0x19,0x0a,0x1c,0x81,0x00,0x02,
      0x97,0x04,0x54,0x01,0x02,0x04,0x02,0x54,0x44,0xc2,
      0x19,0x08,0x1c,0x81,0x08,0x02,
      0x29,0x08,0x1c,0x81,0x00,0x02,
      0x97,0x04,0x54,0x00,0x02,0x08,0x0a,0x54,0x44,0xc2,
      0x87,0x02,0x54,0x00,0x06,0x00,
      0x87,0x02,0x54,0x0c,0x08,0x00,
      0xe7,0x06,0x54,0x00,0x00,0x00,0x01,0x4e,0x00,0x00,0x00,0x00,
      0x07,0x02,0x54,0x0c,0x02,0x00,
      0x0e,0x00,0x00,0x00,
   };
   nir_shader *nir = build_half_normalized_epilog(PIPE_FORMAT_R8_UNORM);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(nir, &key, &part);
   bool ok = compiled && part.binary != NULL &&
             part.info.binary_size == sizeof(expected) &&
             memcmp(part.binary, expected, sizeof(expected)) == 0;

   free(part.binary);
   ralloc_free(nir);
   return ok;
}

static bool
test_g17p_half_rgb10_epilog(void)
{
   /* A mediump texture result reaches the RGB30 desktop epilog through
    * four half registers. It must compile instead of silently dropping
    * the textured draw at the tile-store encoder. */
   nir_shader *nir =
      build_half_normalized_epilog(PIPE_FORMAT_R10G10B10A2_UNORM);
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .secondary = true,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(nir, &key, &part);
   ok &= part.binary != NULL && part.info.binary_size > 0;
   free(part.binary);
   ralloc_free(nir);
   return ok;
}

static bool
test_g17p_background_reload_staging(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P RT0 reload staging");
   b.shader->info.internal = true;
   nir_def *coord = nir_u2u32(&b, nir_load_pixel_coord(&b));
   nir_def *tex = nir_txf(&b, coord, .lod = nir_imm_int(&b, 0),
                          .dim = GLSL_SAMPLER_DIM_2D,
                          .dest_type = nir_type_float32,
                          .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP);
   fs_store_colour(&b, tex);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   nir_shader *nir = fs_finish(&b);
   NIR_PASS(_, nir, agx_nir_lower_texture);
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.ignore_tib_dependencies = true,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(nir, &key, &part);
   ok &= part.info.agx3_staging_known && part.info.agx3_staging_halfs == 4 &&
         part.info.uses_txf && part.info.push_count == 0 &&
         part.info.scratch_size == 0;
   unsigned found = 0;
   for (unsigned i = 8; part.binary && i + 14 <= part.info.binary_size; i++) {
      struct agx3_tex_fetch_desc fetch;
      if (!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B,
                                 (const uint8_t *)part.binary + i, 14, &fetch))
         continue;
      /* With the default AGX_G17P_SR32_PIXEL_COORD the coordinates are the
       * zero-extended 32-bit SR80/SR81 reads in r0/r1 (halves 0 and 2), staged
       * directly: stage 0 <- r0 waiting for SR80, a second stage 0 <- r0 that
       * only carries the wait for SR81 (and r0's discard), then stage 2 <- r1.
       * The repeated stage 0 copy is redundant but harmless (same value). */
      uint8_t x[AGX3_TEX_COPY_MAX_LENGTH], w[AGX3_TEX_COPY_MAX_LENGTH],
         y[AGX3_TEX_COPY_MAX_LENGTH];
      uint8_t nx, nw, ny;
      ok &= fetch.coord == 0 && fetch.wait_mask == (1u << (fetch.slot - 1)) &&
            agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, 0, 0, fetch.slot,
                                      0x1, false, false, x, &nx) &&
            agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, 0, 0, fetch.slot,
                                      0x2, false, true, w, &nw) &&
            agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, 2, 2, fetch.slot,
                                      0, false, true, y, &ny) &&
            i >= nx + nw + ny &&
            !memcmp((const uint8_t *)part.binary + i - nx - nw - ny, x, nx) &&
            !memcmp((const uint8_t *)part.binary + i - nw - ny, w, nw) &&
            !memcmp((const uint8_t *)part.binary + i - ny, y, ny);
      found++;
   }
   ok &= found == 1;
   if (!ok)
      fprintf(stderr, "G17P reload staging mismatch: fetches=%u known=%u halves=%u\n",
              found, part.info.agx3_staging_known, part.info.agx3_staging_halfs);
   free(part.binary);
   ralloc_free(nir);
   return ok;
}

/*
 * Minimal fragment shader: colour = vec4(0.25, 0.5, 0.75, 1.0) stored to
 * render target 0, as agx_nir_lower_tilebuffer lowers a store_output of
 * FRAG_RESULT_DATA0 into a single sample-replicating store_local_pixel_agx.
 */
static nir_shader *
build_fs_flat_colour(float red, float green, float blue)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 flat triangle FS");

   fs_store_colour(&b, nir_vec4(&b, nir_imm_float(&b, red),
                                nir_imm_float(&b, green),
                                nir_imm_float(&b, blue),
                                nir_imm_float(&b, 1.0f)));
   return fs_finish(&b);
}

static nir_shader *
build_fs_flat(void)
{
   return build_fs_flat_colour(0.25f, 0.5f, 0.75f);
}

/* Export a different colour for a hardware test. The NIR compiler supplies
 * every instruction; no captured shader bytes are used by this path. */
static bool
export_g17g_flat_variant(const char *path)
{
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3C,
      .no_preamble = true,
   };
   nir_shader *nir = build_fs_flat_colour(0.75f, 0.25f, 0.5f);
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(nir, &key, &part);
   ok = ok && part.binary && part.info.binary_size == 60 &&
        part.info.main_offset == 0 && !part.info.has_preamble &&
        !part.info.push_count && !part.info.scratch_size &&
        part.info.nr_gprs == 2 && part.info.agx3_staging_known &&
        part.info.agx3_staging_halfs == 0;
   if (ok) {
      FILE *out = fopen(path, "wbx");
      if (!out) {
         ok = false;
      } else {
         ok = fwrite(part.binary, 1, part.info.binary_size, out) == part.info.binary_size;
         if (fclose(out) != 0)
            ok = false;
      }
   }
   fprintf(stderr, "M5_GENERATED_FRAGMENT %s bytes=%u gprs=%u staging=%u/%u rgb=191,64,128\n",
           ok ? "EXPORTED" : "FAILED", part.info.binary_size, part.info.nr_gprs,
           part.info.agx3_staging_known, part.info.agx3_staging_halfs);
   free(part.binary);
   ralloc_free(nir);
   return ok;
}

/* Fragment shader reading a perspective-interpolated vec4 varying. */
static nir_shader *
build_fs_interpolated(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 interpolated FS");

   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *colour = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);

   fs_store_colour(&b, colour);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

static nir_shader *
build_fs_quad_control_mode(bool require_full_quads, bool quad_derivatives)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P quad-control mode FS");

   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *colour = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);

   fs_store_colour(&b, colour);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   b.shader->info.fs.require_full_quads = require_full_quads;
   b.shader->info.fs.quad_derivatives = quad_derivatives;
   agx_preprocess_nir_g17p(b.shader);
   fs_lower(b.shader);
   return b.shader;
}

/* Quad-control execution modes must select the helper-capable G17P fragment
 * protocol even after optimization removes every explicit quad instruction.
 * Native derivative and quad-shuffle fragments both start with the exact
 * unmarked F32INTERP below and otherwise have identical ProgramVariant
 * metadata. A plain interpolation is the marked negative control. */
static bool
test_g17p_quad_control_modes(void)
{
   struct agx_shader_part part[3] = {0};
   const bool full[] = {false, true, false};
   const bool derivative[] = {false, false, true};
   bool semantic[3] = {false};
   bool ok = true;

   for (unsigned i = 0; i < ARRAY_SIZE(part); ++i) {
      nir_shader *nir = build_fs_quad_control_mode(full[i], derivative[i]);
      semantic[i] = nir->info.fs.needs_coarse_quad_helper_invocations &&
                    nir->info.fs.needs_full_quad_helper_invocations;
      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .no_preamble = true,
      };
      ok &= agx_compile_shader_nir(nir, &key, &part[i]) && part[i].binary;
      ralloc_free(nir);
   }

   static const uint8_t marked_interp[] = {
      0x2f, 0x0d, 0x54, 0x00, 0x03, 0x00, 0x04, 0x02, 0x10, 0x00,
   };
   static const uint8_t helper_interp[] = {
      0x2f, 0x05, 0x54, 0x00, 0x03, 0x00, 0x04, 0x02, 0x10, 0x00,
   };

   if (ok) {
      ok &= !semantic[0] && semantic[1] && semantic[2] &&
            !part[0].info.disable_tri_merging &&
            part[1].info.disable_tri_merging &&
            part[2].info.disable_tri_merging &&
            part[0].info.binary_size == part[1].info.binary_size &&
            part[1].info.binary_size == part[2].info.binary_size &&
            part[0].info.binary_size >= sizeof(marked_interp) &&
            !memcmp(part[0].binary, marked_interp, sizeof(marked_interp)) &&
            !memcmp(part[1].binary, helper_interp, sizeof(helper_interp)) &&
            !memcmp(part[2].binary, helper_interp, sizeof(helper_interp)) &&
            !memcmp(part[1].binary, part[2].binary,
                    part[1].info.binary_size);

      /* The entry marker is the only shader-text difference from the plain
       * control. No undocumented quad-control opcode or metadata surrogate is
       * permitted to creep into this protocol. */
      for (unsigned byte = 0; ok && byte < part[0].info.binary_size; ++byte) {
         const uint8_t plain = ((const uint8_t *)part[0].binary)[byte];
         const uint8_t helper = ((const uint8_t *)part[1].binary)[byte];
         ok &= (byte == 1 ? plain == (uint8_t)(helper | 0x08)
                          : plain == helper);
      }
   }

   if (!ok)
      fputs("G17P quad-control helper protocol mismatch\n", stderr);
   for (unsigned i = 0; i < ARRAY_SIZE(part); ++i)
      free(part[i].binary);
   return ok;
}

static nir_shader *
build_fs_divergent_quad_loop(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P divergent quad-control loop FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *varying = nir_load_interpolated_input(
      &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);
   nir_def *pixel = nir_u2u32(&b, nir_load_pixel_coord(&b));
   nir_def *x = nir_channel(&b, pixel, 0);
   nir_def *count = nir_iadd_imm(&b, nir_iand_imm(&b, x, 1), 1);

   nir_variable *counter =
      nir_local_variable_create(b.impl, glsl_uint_type(), "counter");
   nir_variable *quad_sum =
      nir_local_variable_create(b.impl, glsl_uint_type(), "quad_sum");
   nir_variable *derivative_sum =
      nir_local_variable_create(b.impl, glsl_float_type(), "derivative_sum");
   nir_store_var(&b, counter, nir_imm_int(&b, 0), 0x1);
   nir_store_var(&b, quad_sum, nir_imm_int(&b, 0), 0x1);
   nir_store_var(&b, derivative_sum, nir_imm_float(&b, 0.0f), 0x1);

   nir_loop *loop = nir_push_loop(&b);
   nir_def *i = nir_load_var(&b, counter);
   nir_break_if(&b, nir_uge(&b, i, count));
   nir_store_var(&b, quad_sum,
                 nir_iadd(&b, nir_load_var(&b, quad_sum),
                          nir_quad_swap_horizontal(&b, x)),
                 0x1);
   nir_store_var(&b, derivative_sum,
                 nir_fadd(&b, nir_load_var(&b, derivative_sum),
                          nir_ddx(&b, varying)),
                 0x1);
   nir_store_var(&b, counter, nir_iadd_imm(&b, i, 1), 0x1);
   nir_pop_loop(&b, loop);

   nir_def *q = nir_u2f32(&b, nir_load_var(&b, quad_sum));
   nir_def *d = nir_load_var(&b, derivative_sum);
   fs_store_colour(&b, nir_vec4(&b, q, d, nir_u2f32(&b, count),
                                nir_imm_float(&b, 1.0f)));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   b.shader->info.fs.require_full_quads = true;
   b.shader->info.fs.quad_derivatives = true;
   agx_preprocess_nir_g17p(b.shader);
   fs_lower(b.shader);
   return b.shader;
}

static bool
test_g17p_divergent_quad_loop(void)
{
   /* Complete post-RA G17P shader. Its unmarked F32INTERP entry, immediate
    * QUADBFLY, DFDX, ICMP+CNDLT, backward B and CNDEND(2) forms match the
    * fresh applegpu_g17p loop witnesses byte-for-byte at each corresponding
    * instruction family. Keep the compiler's surrounding lifetimes exact.
    *
    * Default codegen: AGX_G17P_SR32_PIXEL_COORD reads the pixel x coordinate
    * as the zero-extended 32-bit SR80 (no 16->32 copy/clr pair),
    * AGX_G17P_INT_DEST_CACHE sets the iadd destination-cache hint, and the
    * C3 ILP schedule issues the loop body's independent DFDX and counter
    * increment ahead of the quad add. The pre-RA SSA of this shader is
    * identical to the pass-off build's (only the SR80 zero extension
    * differs); loop control (CNDLT, backward B, CNDEND) is unchanged.
    *
    * AGX_G17P_BITOP_IMMEDIATES, default on since 2b2e972d60b ("Mesa W",
    * hardware-accepted), folds the one `ldimm` + register bitop of the loop
    * setup into the immediate-operand bitop (234 -> 232 bytes). With that
    * flag off the previous 234-byte pin still matches exactly.
    *
    * The loop body is divergent, so five of its register sources carry
    * last-use hints from physical liveness (agx_opt_register_cache), one bit
    * each; without divergent last uses the bytes are the earlier pin. */
   static const uint8_t expected[] = {
      0x2f,0x05,0x54,0x02,0x03,0x00,0x04,0x02,0x10,0x00,
      0xaf,0x00,0x54,0x02,0x03,0x04,0x00,0x48,0x20,0x01,
      0x2f,0x05,0x54,0x04,0x03,0x02,0x00,0x02,0x10,0x00,
      0x19,0x05,0x3f,0x03,0x00,0x01,0x00,0x00,0x2c,0xa0,
      0x10,0x06,0x3b,0x05,0x07,0x81,0x10,0x22,0x00,0x00,
      0x00,0x00,0x9f,0x01,0x54,0x06,0x03,0x02,0x18,0x88,
      0x15,0x04,0x4b,0x00,0x00,0x00,0x5b,0x00,0x00,0x00,
      0x6b,0x00,0x00,0x00,0x0f,0x05,0x54,0x19,0x0f,0x05,
      0x54,0x1a,0x0a,0x0d,0x23,0x07,0x02,0x00,0x05,0x00,
      0x00,0x00,0x8f,0x04,0x54,0x02,0xc7,0x00,0x54,0x0e,
      0x03,0x08,0x02,0x2c,0x02,0x00,0x37,0x05,0x54,0x10,
      0x03,0x04,0x92,0x40,0x00,0x00,0x9f,0x01,0x54,0x0c,
      0x03,0x02,0x30,0x88,0x15,0x04,0x9f,0x01,0x54,0x0a,
      0x03,0x14,0x38,0xa8,0x15,0x05,0x49,0x11,0x08,0x09,
      0x0f,0x00,0x54,0xc6,0xff,0xff,0xff,0xff,0xff,0x00,
      0x0f,0x06,0x04,0x02,0x00,0x00,0x0f,0x06,0x04,0x01,
      0x00,0x00,0xa7,0x07,0x54,0x02,0x02,0x14,0xac,0x20,
      0xa7,0x07,0x54,0x04,0x02,0x0c,0xac,0x20,0x87,0x02,
      0x54,0x00,0x06,0x00,0x87,0x02,0x54,0x0c,0x08,0x00,
      0x97,0x04,0x54,0x02,0x02,0x04,0x22,0x54,0x44,0xc2,
      0x97,0x04,0x54,0x03,0x02,0x08,0xc2,0x50,0x44,0xc8,
      0xe7,0x06,0x54,0x02,0x00,0x00,0x01,0x4e,0x00,0x00,
      0x00,0x00,0x07,0x02,0x54,0x0c,0x02,0x00,0x0e,0x00,
      0x00,0x00,
   };
   nir_shader *nir = build_fs_divergent_quad_loop();
   struct agx_shader_key key = {
      .dev.chip = AGX_CHIP_G17P,
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(nir, &key, &part) && part.binary &&
             part.info.disable_tri_merging &&
             part.info.binary_size == sizeof(expected) &&
             !memcmp(part.binary, expected, sizeof(expected));
   if (getenv("AGX3_DUMP_BINARY") && part.binary) {
      fprintf(stderr, "G17P divergent quad loop: %u bytes:",
              part.info.binary_size);
      for (unsigned i = 0; i < part.info.binary_size; ++i)
         fprintf(stderr, " %02x", ((const uint8_t *)part.binary)[i]);
      fputc('\n', stderr);
   }
   if (!ok)
      fputs("G17P divergent quad loop compiler mismatch\n", stderr);
   free(part.binary);
   ralloc_free(nir);
   return ok;
}

static nir_shader *
build_fs_alpha_coverage_boundary(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P alpha-to-coverage pixel boundary FS");
   nir_def *zero = nir_imm_float(&b, 0.0f);
   nir_def *one = nir_imm_float(&b, 1.0f);
   nir_def *pixel = nir_u2u32(&b, nir_load_pixel_coord(&b));
   nir_def *frag_x = nir_fadd_imm(
      &b, nir_u2f32(&b, nir_channel(&b, pixel, 0)), 0.5f);
   nir_def *alpha = nir_bcsel(
      &b, nir_flt_imm(&b, frag_x, 16.0f),
      zero, one);
   nir_def *colour = nir_vec4(&b, one, zero, zero, alpha);
   for (unsigned c = 0; c < 4; ++c)
      nir_export_agx(&b, nir_channel(&b, colour, c),
                     .base = AGX_ABI_FOUT_COLOUR(0) + 2 * c);

   /* Match the Vulkan frontend ordering: FragCoord.xy becomes the native
    * pixel-coordinate system value plus its half-pixel center before the AGX
    * optimizer folds the boundary to the exactly representable 15.5. */
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_preprocess_nir_g17p(b.shader);
   fs_lower(b.shader);
   return b.shader;
}

static bool
test_g17p_alpha_coverage_boundary(void)
{
   /* Exact post-RA select from the unchanged Vulkan fragment:
    *   gl_FragCoord.x < 16.0 ? 0.0 : 1.0
    * Pixel-center lowering reverses that to x >= 15.5. Physical r7 cannot
    * use native ep10, so the composer must choose this ten-byte ep22 form. */
   static const uint8_t expected_fsel[] = {
      0x72, 0x01, 0x0b, 0xef, 0x06,
      0x08, 0x12, 0x02, 0x20, 0x80,
   };
   nir_shader *nir = build_fs_alpha_coverage_boundary();
   struct agx_shader_key key = {
      .dev.chip = AGX_CHIP_G17P,
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(nir, &key, &part) && part.binary;
   bool found = false;
   for (unsigned i = 0;
        ok && i + sizeof(expected_fsel) <= part.info.binary_size; ++i) {
      if (!memcmp((const uint8_t *)part.binary + i, expected_fsel,
                  sizeof(expected_fsel))) {
         found = true;
         break;
      }
   }
   ok &= found;
   if (!ok && part.binary) {
      fprintf(stderr, "G17P alpha-coverage boundary (%u bytes):",
              part.info.binary_size);
      for (unsigned i = 0; i < part.info.binary_size; ++i)
         fprintf(stderr, " %02x", ((const uint8_t *)part.binary)[i]);
      fputc('\n', stderr);
   }
   free(part.binary);
   ralloc_free(nir);
   return ok;
}

/* A real post-lowering interpolateAtSample shape. Deriving the runtime sample
 * index from the pixel coordinate keeps it dynamic without leaving an
 * unlowered load_sample_id intrinsic for the backend. */
static nir_shader *
build_fs_interpolated_at_sample(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &agx_nir_options,
                                     "G17P explicit at-sample interpolated FS");

   nir_def *pixel = nir_u2u32(&b, nir_load_pixel_coord(&b));
   nir_def *sample = nir_iand_imm(&b, nir_channel(&b, pixel, 0), 3);
   nir_def *bary = nir_load_barycentric_at_sample(
      &b, 32, sample, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *colour = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);

   fs_store_colour(&b, colour);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* Vulkan `centroid in vec4`, the CTS multiple_interpolation shape. */
static nir_shader *
build_fs_interpolated_centroid(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &agx_nir_options,
                                     "G17P centroid interpolated FS");

   nir_def *bary = nir_load_barycentric_centroid(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *colour = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);

   fs_store_colour(&b, colour);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* Retain the two coverage updates in the faulting Hyprland fragment: a
 * dynamic no-tests discard followed by the final depth/stencil test. */
static nir_shader *
build_fs_discard_then_tests(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P discard then tests FS");
   nir_def *target = nir_iand_imm(
      &b, nir_load_preamble(&b, 1, 16, .base = 0), 1);
   nir_sample_mask_agx(&b, target, nir_imm_intN_t(&b, 0, 16));
   nir_sample_mask_agx(&b, nir_imm_intN_t(&b, 0xff, 16),
                       nir_imm_intN_t(&b, 0xff, 16));
   fs_store_colour(&b, nir_imm_vec4(&b, 0.25f, 0.5f, 0.75f, 1.0f));
   return fs_finish(&b);
}

/* Minecraft's `if (colour.a == 0.0) discard;` as Honeykrisp's deferred
 * lowering hands it to the backend: the dynamic tests_zs mask ORed into the
 * killed set, and its complement as the live set. AGX_G17P_DISCARD_WORD
 * reduces both to bit 0 at one sample; the single testing update must keep
 * its class-1 transaction. */
static nir_shader *
build_fs_alpha_discard_word(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P alpha discard word FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *colour = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);
   nir_def *c = nir_feq(&b, nir_channel(&b, colour, 3), nir_imm_float(&b, 0));
   nir_def *killed = nir_bcsel(&b, c, nir_imm_intN_t(&b, 0xff, 16),
                               nir_imm_intN_t(&b, 0, 16));
   nir_def *tests = nir_load_preamble(&b, 1, 16, .base = 0);
   nir_sample_mask_agx(&b, nir_ior(&b, tests, killed),
                       nir_ixor(&b, killed, nir_imm_intN_t(&b, 0xff, 16)));
   fs_store_colour(&b, colour);
   b.shader->info.fs.uses_discard = true;
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

static nir_shader *
build_fs_constant_target_alpha_discard(uint16_t target)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P constant-target alpha discard FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *colour = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);
   nir_def *c = nir_feq(&b, nir_channel(&b, colour, 3), nir_imm_float(&b, 0));
   nir_def *killed = nir_bcsel(&b, c, nir_imm_intN_t(&b, 0xff, 16),
                               nir_imm_intN_t(&b, 0, 16));
   nir_sample_mask_agx(&b, nir_imm_intN_t(&b, target, 16),
                       nir_ixor(&b, killed, nir_imm_intN_t(&b, 0xff, 16)));
   fs_store_colour(&b, colour);
   b.shader->info.fs.uses_discard = true;
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* A dynamic no-tests kill (killed | 0 target, constant live 0) followed by
 * the testing update with a dynamic live set: only the second owns class 1,
 * and reducing the first's target must not turn it into a testing update. */
static nir_shader *
build_fs_kill_then_alpha_discard_word(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P kill then alpha discard word FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *colour = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);
   nir_def *first = nir_flt(&b, nir_channel(&b, colour, 0),
                            nir_imm_float(&b, 0.5));
   nir_sample_mask_agx(&b, nir_bcsel(&b, first, nir_imm_intN_t(&b, 0xff, 16),
                                     nir_imm_intN_t(&b, 0, 16)),
                       nir_imm_intN_t(&b, 0, 16));
   nir_def *c = nir_feq(&b, nir_channel(&b, colour, 3), nir_imm_float(&b, 0));
   nir_def *killed = nir_bcsel(&b, c, nir_imm_intN_t(&b, 0xff, 16),
                               nir_imm_intN_t(&b, 0, 16));
   nir_def *tests = nir_load_preamble(&b, 1, 16, .base = 0);
   nir_sample_mask_agx(&b, nir_ior(&b, tests, killed),
                       nir_ixor(&b, killed, nir_imm_intN_t(&b, 0xff, 16)));
   fs_store_colour(&b, colour);
   b.shader->info.fs.uses_discard = true;
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* gl_SampleMask = mask, as lowered: the samples outside the mask are demoted
 * without tests, then the remainder runs the depth/stencil tests. */
static nir_shader *
build_fs_sample_mask_write(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P sample mask write FS");
   nir_def *mask = nir_load_preamble(&b, 1, 16, .base = 0);
   nir_sample_mask_agx(&b, nir_inot(&b, mask), nir_imm_intN_t(&b, 0, 16));
   nir_sample_mask_agx(&b, nir_imm_intN_t(&b, 0xff, 16),
                       nir_imm_intN_t(&b, 0xff, 16));
   fs_store_colour(&b, nir_imm_vec4(&b, 0.25f, 0.5f, 0.75f, 1.0f));
   return fs_finish(&b);
}

/* Fragment shader with a discard: lowered to a sample_mask write. */
static nir_shader *
build_fs_discard(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 discard FS");

   nir_demote(&b);
   b.shader->info.fs.uses_discard = true;
   fs_store_colour(&b, nir_vec4(&b, nir_imm_float(&b, 0.25f),
                                nir_imm_float(&b, 0.5f),
                                nir_imm_float(&b, 0.75f),
                                nir_imm_float(&b, 1.0f)));
   return fs_finish(&b);
}

/* A multisample shader reading the API-visible incoming coverage mask. */
static nir_shader *
build_fs_sample_mask_in(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 sample-mask input FS");
   nir_def *mask = nir_load_sample_mask_in(&b);
   nir_def *value = nir_u2f32(&b, mask);
   fs_store_colour(&b, nir_vec4(&b, value, value, value,
                                nir_imm_float(&b, 1.0f)));
   return fs_finish(&b);
}

/* A two-sided material choosing a value from gl_FrontFacing. */
static nir_shader *
build_fs_front_facing(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 front-facing FS");
   nir_def *front = nir_load_front_face(&b, 1);
   nir_def *bits = nir_bcsel(&b, front, nir_imm_int(&b, 17),
                             nir_imm_int(&b, 3));
   nir_def *value = nir_u2f32(&b, bits);
   fs_store_colour(&b, nir_vec4(&b, value, value, value,
                                nir_imm_float(&b, 1.0f)));
   return fs_finish(&b);
}

/* Fragment shader writing gl_FragDepth: lowered to zs_emit. */
static nir_shader *
build_fs_depth(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 depth-write FS");

   fs_store_colour(&b, nir_vec4(&b, nir_imm_float(&b, 0.25f),
                                nir_imm_float(&b, 0.5f),
                                nir_imm_float(&b, 0.75f),
                                nir_imm_float(&b, 1.0f)));
   nir_store_output(&b, nir_imm_float(&b, 0.5f), nir_imm_int(&b, 0),
                    .base = 0, .write_mask = 0x1, .src_type = nir_type_float32,
                    .io_semantics.location = FRAG_RESULT_DEPTH,
                    .io_semantics.num_slots = 1);

   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0) |
                                    BITFIELD64_BIT(FRAG_RESULT_DEPTH);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   return b.shader;
}

/* A depth write to samples known only when the fragment runs. */
static nir_shader *
build_fs_depth_dynamic_samples(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 dynamic-sample depth FS");

   fs_store_colour(&b, nir_imm_vec4(&b, 0.25f, 0.5f, 0.75f, 1.0f));
   nir_store_zs_agx(&b, nir_load_preamble(&b, 1, 16, .base = 0),
                    nir_imm_float(&b, 0.5f), nir_undef(&b, 1, 16), .base = 1);

   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0) |
                                    BITFIELD64_BIT(FRAG_RESULT_DEPTH);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   return b.shader;
}

static nir_shader *
build_fs_depth_only_value(float depth)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 depth-only FS");
   nir_store_output(&b, nir_imm_float(&b, depth), nir_imm_int(&b, 0),
                    .base = 0, .write_mask = 0x1,
                    .src_type = nir_type_float32,
                    .io_semantics.location = FRAG_RESULT_DEPTH,
                    .io_semantics.num_slots = 1);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DEPTH);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   return b.shader;
}

static nir_shader *
build_fs_depth_only(void)
{
   return build_fs_depth_only_value(0.5f);
}

static nir_shader *
build_fs_depth_precision(void)
{
   /* This differs from 0.5f in float32 but rounds to it in float16. */
   return build_fs_depth_only_value(0.50001f);
}

/* Same depth path with a genuine run-time f32 source. Keep its precision
 * and register lifetime through the complete staged depth payload. */
static nir_shader *
build_fs_depth_dynamic(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 dynamic depth-write FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *depth = nir_load_interpolated_input(
      &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);
   fs_store_colour(&b, nir_vec4(&b, nir_imm_float(&b, 0.25f),
                                nir_imm_float(&b, 0.5f),
                                nir_imm_float(&b, 0.75f),
                                nir_imm_float(&b, 1.0f)));
   nir_store_output(&b, depth, nir_imm_int(&b, 0), .base = 0,
                    .write_mask = 0x1, .src_type = nir_type_float32,
                    .io_semantics.location = FRAG_RESULT_DEPTH,
                    .io_semantics.num_slots = 1);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0) |
                                    BITFIELD64_BIT(FRAG_RESULT_DEPTH);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   return b.shader;
}

/* gl_FragStencilRefARB from a run-time value, optionally with gl_FragDepth,
 * as the attachment feedback loop tests write it. */
static nir_shader *
build_fs_stencil_value(bool depth)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 stencil-write FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *value = nir_load_interpolated_input(
      &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);
   fs_store_colour(&b, nir_vec4(&b, nir_imm_float(&b, 0.25f),
                                nir_imm_float(&b, 0.5f),
                                nir_imm_float(&b, 0.75f),
                                nir_imm_float(&b, 1.0f)));
   nir_store_output(&b, nir_f2i32(&b, value), nir_imm_int(&b, 0), .base = 0,
                    .write_mask = 0x1, .src_type = nir_type_int32,
                    .io_semantics.location = FRAG_RESULT_STENCIL,
                    .io_semantics.num_slots = 1);
   if (depth) {
      nir_store_output(&b, value, nir_imm_int(&b, 0), .base = 0,
                       .write_mask = 0x1, .src_type = nir_type_float32,
                       .io_semantics.location = FRAG_RESULT_DEPTH,
                       .io_semantics.num_slots = 1);
   }
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written =
      BITFIELD64_BIT(FRAG_RESULT_DATA0) | BITFIELD64_BIT(FRAG_RESULT_STENCIL) |
      (depth ? BITFIELD64_BIT(FRAG_RESULT_DEPTH) : 0);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   return b.shader;
}

static nir_shader *
build_fs_stencil(void)
{
   return build_fs_stencil_value(false);
}

static nir_shader *
build_fs_depth_stencil(void)
{
   return build_fs_stencil_value(true);
}

/*
 * Fragment shader that reads the tilebuffer before writing it -- the shape a
 * blend or a framebuffer fetch takes, and the shape Apple's own `w_pass8`
 * probe uses to pair a tile load's scoreboard slot with the store that
 * consumes it.
 */
static nir_shader *
build_fs_tilebuffer_load(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 tilebuffer-load FS");

   nir_def *old = nir_load_local_pixel_agx(
      &b, 4, 32, nir_imm_intN_t(&b, 0xFF, 16), .base = 0,
      .format = AGX3_TEST_TIB_FORMAT);
   fs_store_colour(&b, old);
   return fs_finish(&b);
}

/* Exercise the complete single-sample fragment-I/O family reached by
 * Gallium's native tilebuffer lowering.  The individual AGX3 encoders have
 * byte-exact Apple witnesses; this matrix proves that production NIR selects
 * those encoders with the right register width and channel packing instead of
 * merely testing hand-built backend IR.
 */
static unsigned
g17p_count_bytes(const struct agx_shader_part *part, const uint8_t *needle,
                 unsigned length)
{
   unsigned count = 0;
   for (unsigned off = 0; part->binary && off + length <= part->info.binary_size;
        ++off)
      count += !memcmp((const uint8_t *)part->binary + off, needle, length);
   return count;
}

static bool
test_agx3_native_relative_shuffles(void)
{
   static const nir_intrinsic_op ops[] = {
      nir_intrinsic_shuffle_xor,
      nir_intrinsic_shuffle_up,
      nir_intrinsic_shuffle_down,
   };

   for (unsigned oi = 0; oi < ARRAY_SIZE(ops); ++oi) {
      /* Constant deltas in 1..31 stay native on G17P and M5 compute and compile
       * end-to-end on AGX3B and AGX3C. */
      static const uint32_t valid_deltas[] = {1, 4, 16, 31};
      for (unsigned di = 0; di < ARRAY_SIZE(valid_deltas); ++di) {
         for (unsigned m5 = 0; m5 < 2; ++m5) {
            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_COMPUTE, &agx_nir_options,
               "native relative shuffle");
            b.shader->info.workgroup_size[0] = 32;
            b.shader->info.workgroup_size[1] = 1;
            b.shader->info.workgroup_size[2] = 1;
            nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);
            nir_def *val = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
            nir_def *delta = nir_imm_int(&b, valid_deltas[di]);
            nir_def *res = ops[oi] == nir_intrinsic_shuffle_xor
                              ? nir_shuffle_xor(&b, val, delta)
                           : ops[oi] == nir_intrinsic_shuffle_up
                              ? nir_shuffle_up(&b, val, delta)
                              : nir_shuffle_down(&b, val, delta);
            nir_store_global(&b, res, nir_iadd_imm(&b, addr, 64),
                             .align_mul = 4);

            if (m5)
               agx_nir_lower_subgroups_m5(b.shader, true, true);
            else
               agx_nir_lower_subgroups_g17p(b.shader);

            unsigned kept = 0;
            nir_foreach_function_impl(impl, b.shader) {
               nir_foreach_block(block, impl) {
                  nir_foreach_instr(instr, block) {
                     kept += instr->type == nir_instr_type_intrinsic &&
                             nir_instr_as_intrinsic(instr)->intrinsic == ops[oi];
                  }
               }
            }
            if (kept != 1) {
               fprintf(stderr,
                       "relative shuffle op=%u delta=%u m5=%u kept=%u\n", oi,
                       valid_deltas[di], m5, kept);
               ralloc_free(b.shader);
               return false;
            }

            b.shader->info.io_lowered = true;
            struct agx_shader_key key = {
               .dev.encoding =
                  m5 ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B,
               .no_preamble = true,
            };
            struct agx_shader_part part = {0};
            const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
            free(part.binary);
            ralloc_free(b.shader);
            if (!compiled) {
               fprintf(stderr,
                       "relative shuffle compile failed op=%u delta=%u m5=%u\n",
                       oi, valid_deltas[di], m5);
               return false;
            }
         }
      }

      /* Dynamic, zero, and out-of-range deltas must be lowered before backend
       * emission. */
      static const int invalid_deltas[] = {-1, 0, 32};
      for (unsigned di = 0; di < ARRAY_SIZE(invalid_deltas); ++di) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options,
            "lowered relative shuffle");
         nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);
         nir_def *val = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
         nir_def *delta = invalid_deltas[di] < 0
                             ? nir_load_global(&b, 1, 32,
                                               nir_iadd_imm(&b, addr, 4),
                                               .align_mul = 4)
                             : nir_imm_int(&b, invalid_deltas[di]);
         nir_def *res = ops[oi] == nir_intrinsic_shuffle_xor
                           ? nir_shuffle_xor(&b, val, delta)
                        : ops[oi] == nir_intrinsic_shuffle_up
                           ? nir_shuffle_up(&b, val, delta)
                           : nir_shuffle_down(&b, val, delta);
         nir_store_global(&b, res, nir_iadd_imm(&b, addr, 64), .align_mul = 4);

         agx_nir_lower_subgroups_g17p(b.shader);
         agx_nir_lower_dynamic_shuffle_xor(b.shader);

         unsigned remaining = 0;
         nir_foreach_function_impl(impl, b.shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  remaining += instr->type == nir_instr_type_intrinsic &&
                               nir_instr_as_intrinsic(instr)->intrinsic ==
                                  ops[oi];
               }
            }
         }
         ralloc_free(b.shader);
         if (remaining != 0) {
            fprintf(stderr,
                    "invalid relative shuffle not lowered op=%u delta=%d\n", oi,
                    invalid_deltas[di]);
            return false;
         }
      }
   }
   return true;
}

static bool
test_g17p_clustered_minmax_width(void)
{
   for (unsigned bits = 16; bits <= 64; bits *= 2) {
      for (unsigned max = 0; max < 2; ++max) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "clustered minmax width");
         nir_reduce(&b, nir_undef(&b, 1, bits), .cluster_size = 4,
                    .reduction_op = max ? nir_op_fmax : nir_op_fmin);
         agx_nir_lower_subgroups_g17p(b.shader);
         unsigned remaining = 0;
         nir_foreach_function_impl(impl, b.shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  remaining += instr->type == nir_instr_type_intrinsic &&
                     nir_instr_as_intrinsic(instr)->intrinsic ==
                        nir_intrinsic_reduce;
               }
            }
         }
         ralloc_free(b.shader);
         if (remaining != (bits == 32)) {
            fprintf(stderr, "clustered f%s %u-bit lowering mismatch\n",
                    max ? "max" : "min", bits);
            return false;
         }
      }
   }
   return true;
}

static bool
test_g17p_pbe_to_texture_fence(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P PBE-to-texture fence");
   nir_fence_pbe_to_tex_agx(&b);
   b.shader->info.io_lowered = true;

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   static const uint8_t wait[] = {
      0x07, 0x14, 0x54, 0x50, 0x06, 0x00,
      0x07, 0x04, 0x54, 0xd0, 0x06, 0x00,
   };
   static const uint8_t execution[] = {
      0x07, 0x04, 0x54, 0x51, 0x0e, 0x00,
      0x07, 0x04, 0x54, 0xd1, 0x0e, 0x00,
   };
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool ok = compiled &&
                   g17p_count_bytes(&part, wait, sizeof(wait)) == 1 &&
                   g17p_count_bytes(&part, execution, sizeof(execution)) == 0;

   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/* An epilog that reads a target only for channels it does not store, and one
 * that stores a target back unchanged. Apple's store-only fragments take
 * class 12 at once, and a fragment writing no channel takes no colour class
 * (blend/native/tilemask.py wm_*, blend/native/wm/none.py). */
static bool
test_g17p_unread_colour_waits(void)
{
   static const uint8_t wait8[] = {0x87, 0x02, 0x54, 0x08, 0x08, 0x00};
   static const uint8_t wait12[] = {0x87, 0x02, 0x54, 0x0c, 0x08, 0x00};
   static const uint8_t release12[] = {0x07, 0x02, 0x54, 0x0c, 0x02, 0x00};
   bool ok = true;

   for (unsigned unchanged = 0; unchanged < 2; ++unchanged) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P unread colour FS");
      nir_def *old = nir_load_local_pixel_agx(
         &b, 4, 16, nir_imm_intN_t(&b, 0xff, 16), .base = 0,
         .format = PIPE_FORMAT_R16_UINT);
      nir_def *value =
         unchanged ? old
                   : nir_vec4(&b, nir_channel(&b, old, 0),
                              nir_imm_intN_t(&b, 1, 16),
                              nir_imm_intN_t(&b, 2, 16),
                              nir_channel(&b, old, 3));
      nir_store_local_pixel_agx(
         &b, value, nir_imm_intN_t(&b, 0xff, 16), nir_undef(&b, 2, 16),
         .base = 0, .write_mask = unchanged ? 0xf : 0x6,
         .format = PIPE_FORMAT_R16_UINT);
      nir_shader *nir = fs_finish(&b);

      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .no_preamble = true,
         .secondary = true,
      };
      struct agx_shader_part part = {0};
      bool compiled = agx_compile_shader_nir(nir, &key, &part);
      compiled = compiled && g17p_count_bytes(&part, wait8, sizeof(wait8)) == 0;
      if (unchanged) {
         compiled = compiled &&
                    g17p_count_bytes(&part, wait12, sizeof(wait12)) == 0 &&
                    g17p_count_bytes(&part, release12, sizeof(release12)) == 0;
      } else {
         compiled = compiled &&
                    g17p_count_bytes(&part, wait12, sizeof(wait12)) == 1 &&
                    g17p_count_bytes(&part, release12, sizeof(release12)) == 1;
      }
      if (!compiled) {
         fprintf(stderr, "G17P %s colour epilog waits mismatch\n",
                 unchanged ? "unchanged" : "partial");
         ok = false;
      }
      free(part.binary);
      ralloc_free(nir);
   }

   return ok;
}

/* A logical 32-bit integer tile read must agree with explicitly loading the
 * physical halfwords and widening them. LLDCOL itself does not widen. */
static bool
test_g17p_halfword_tile_widening(void)
{
   for (unsigned channels = 1; channels <= 4; ++channels) {
      struct agx_shader_part parts[2] = {0};
      bool ok = true;
      for (unsigned explicit_widen = 0; explicit_widen < 2; ++explicit_widen) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "halfword tile widening");
         nir_def *value = nir_load_local_pixel_agx(
            &b, channels, explicit_widen ? 16 : 32,
            nir_imm_intN_t(&b, 0xff, 16), .base = 0,
            .format = PIPE_FORMAT_R16_UINT);
         value = nir_u2u32(&b, value);
         nir_store_local_pixel_agx(
            &b, nir_inot(&b, value), nir_imm_intN_t(&b, 0xff, 16),
            nir_undef(&b, 2, 16), .base = 0,
            .write_mask = BITFIELD_MASK(channels),
            .format = PIPE_FORMAT_R32_UINT);
         /* Tile I/O is already lowered, as in a specialized blend epilog. */
         nir_shader *nir = b.shader;
         nir->info.io_lowered = true;
         nir->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
         struct agx_shader_key key = {
            .dev.encoding = AGX_ISA_ENCODING_AGX3B,
            .no_preamble = true,
         };
         ok &= agx_compile_shader_nir(nir, &key, &parts[explicit_widen]);
         ralloc_free(nir);
      }
      ok = ok && parts[0].info.binary_size == parts[1].info.binary_size &&
           !memcmp(parts[0].binary, parts[1].binary, parts[0].info.binary_size);
      free(parts[0].binary);
      free(parts[1].binary);
      if (!ok) {
         fprintf(stderr, "halfword tile widening mismatch: %u channels\n",
                 channels);
         return false;
      }
   }
   return true;
}

/* Three blended RGBA32 targets need 24 payload halves live at their stores.
 * This was once beyond the tile-store bank and refused; 46eeb53ec49 ("asahi:
 * complete native G17P three-target blend and subset epilogs") implemented it
 * against Apple's witnessed register bank, and agx3_tile_store_bank() now
 * returns 32 for this shape -- see its comment recording that Apple's own
 * blending fragments store three RGBA32Float targets from halves0..23. So the
 * shader must now COMPILE, and register allocation must find an assignment.
 *
 * NOTE: this no longer gives agx3_tile_store_payloads_fit() negative coverage.
 * Re-establishing that needs a shape whose demand genuinely exceeds the bank
 * the function computes for it, grounded in an Apple witness rather than
 * assumed -- asserting a refusal for a shape the hardware supports would be
 * worse than having no negative case. */
static bool
test_g17p_blended_payloads_beyond_bank(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P three blended RGBA32 FS");
   nir_def *values[3];
   for (unsigned rt = 0; rt < 3; ++rt) {
      nir_def *old = nir_load_local_pixel_agx(
         &b, 4, 32, nir_imm_intN_t(&b, 0xff, 16), .base = 16 * rt,
         .format = PIPE_FORMAT_R32_UINT);
      values[rt] = nir_inot(&b, old);
   }
   for (unsigned rt = 0; rt < 3; ++rt) {
      nir_store_local_pixel_agx(
         &b, values[rt], nir_imm_intN_t(&b, 0xff, 16), nir_undef(&b, 2, 16),
         .base = 16 * rt, .write_mask = 0xf, .format = PIPE_FORMAT_R32_UINT);
   }
   nir_shader *nir = fs_finish(&b);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
      .secondary = true,
   };
   struct agx_shader_part part;
   memset(&part, 0xa5, sizeof(part));
   const bool ok = agx_compile_shader_nir(nir, &key, &part) &&
                   part.binary != NULL && part.info.binary_size > 0;
   if (!ok)
      fprintf(stderr,
              "G17P three-target RGBA32 blend was not compiled\n");
   if (part.binary)
      free(part.binary);
   ralloc_free(nir);
   return ok;
}

/* Tile-load match with byte 3, the destination register, ignored: the load
 * form (format, slot 6 scoreboard, mask) is the contract. The destination
 * depends on scheduling: AGX_G17P_COLOR_ILP, default on since 2b2e972d60b
 * ("Mesa W", hardware-accepted), issues the f16 negation constants first,
 * moving R16_FLOAT2/3 tile loads off r0. */
static bool
tile_load_present(const struct agx_shader_part *part, const uint8_t *load,
                  size_t size)
{
   for (unsigned offset = 0; offset + size <= part->info.binary_size;
        ++offset) {
      const uint8_t *at = (const uint8_t *)part->binary + offset;
      if (!memcmp(at, load, 3) && !memcmp(at + 4, load + 4, size - 4))
         return true;
   }
   return false;
}

static bool
test_g17p_fragment_io_matrix(void)
{
   static const uint8_t pair_loads[][12] = {
      /* Production follows Apple's fixed G17P tile-read scoreboard slot 6. */
      {0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0xd2,0x02,0x00,0x00,0x00},
      {0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0x92,0x02,0x00,0x00,0x00},
      {0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0xb2,0x02,0x00,0x00,0x00},
   };
   static const uint8_t rgb565_load[] = {
      0x67,0x0e,0x54,0x00,0x00,0x00,0x01,0x9a,0x02,0x00,0x00,0x00,
   };
   static const struct {
      enum pipe_format format;
      uint8_t components;
      bool expect_ok;
   } cases[] = {
      {PIPE_FORMAT_R8_UNORM, 1, true},
      {PIPE_FORMAT_R8_UNORM, 2, true},
      {PIPE_FORMAT_R8_UNORM, 4, true},
      {PIPE_FORMAT_R8_SNORM, 1, true},
      {PIPE_FORMAT_R8_SNORM, 2, true},
      {PIPE_FORMAT_R8_SNORM, 4, true},
      {PIPE_FORMAT_R16_FLOAT, 1, true},
      {PIPE_FORMAT_R16_FLOAT, 2, true},
      {PIPE_FORMAT_R16_FLOAT, 3, true},
      {PIPE_FORMAT_R16_FLOAT, 4, true},
      {PIPE_FORMAT_R16_UNORM, 1, true},
      {PIPE_FORMAT_R16_UNORM, 2, true},
      {PIPE_FORMAT_R16_UNORM, 4, true},
      {PIPE_FORMAT_R16_SNORM, 1, true},
      {PIPE_FORMAT_R16_SNORM, 2, true},
      {PIPE_FORMAT_R16_SNORM, 4, true},
      {PIPE_FORMAT_R16_UINT, 1, true},
      {PIPE_FORMAT_R16_UINT, 2, true},
      {PIPE_FORMAT_R16_UINT, 4, true},
      {PIPE_FORMAT_R32_UINT, 1, true},
      {PIPE_FORMAT_R32_UINT, 2, true},
      {PIPE_FORMAT_R32_UINT, 4, true},
      {PIPE_FORMAT_R10G10B10A2_UNORM, 4, true},
   };

   bool ok = true;
   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options,
         "G17P fragment I/O %s", util_format_name(cases[i].format));
      nir_def *old = nir_load_local_pixel_agx(
         &b, cases[i].components, 32, nir_imm_intN_t(&b, 0xff, 16),
         .base = 0, .format = cases[i].format);
      /* Write back a new value: storing the loaded one is no store at all. */
      nir_def *value = util_format_is_pure_integer(cases[i].format)
                          ? nir_inot(&b, old)
                          : nir_fneg(&b, old);
      nir_store_local_pixel_agx(
         &b, value, nir_imm_intN_t(&b, 0xff, 16), nir_undef(&b, 2, 16),
         .base = 0, .write_mask = BITFIELD_MASK(cases[i].components),
         .format = cases[i].format);
      nir_shader *nir = fs_finish(&b);
      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .no_preamble = true,
      };
      struct agx_shader_part part;
      memset(&part, 0xa5, sizeof(part));
      const bool compiled = agx_compile_shader_nir(nir, &key, &part);
      const struct agx_shader_part cleared = {0};
      const bool terminated = compiled && part.binary &&
         part.info.binary_size > sizeof(agx3_end) &&
         !memcmp((const uint8_t *)part.binary + part.info.binary_size -
                    sizeof(agx3_end),
                 agx3_end, sizeof(agx3_end));
      bool exact_pair_load = true;
      if (compiled && cases[i].components == 2) {
         const unsigned width =
            cases[i].format == PIPE_FORMAT_R8_UNORM ||
                  cases[i].format == PIPE_FORMAT_R8_SNORM
               ? 0
            : cases[i].format == PIPE_FORMAT_R32_UINT
               ? 2
               : 1;
         exact_pair_load = tile_load_present(&part, pair_loads[width],
                                             sizeof(pair_loads[width]));
      }
      if (compiled && cases[i].format == PIPE_FORMAT_R16_FLOAT &&
          cases[i].components == 3) {
         exact_pair_load =
            tile_load_present(&part, rgb565_load, sizeof(rgb565_load));
      }
      if (compiled != cases[i].expect_ok ||
          (cases[i].expect_ok && !terminated) ||
          (cases[i].expect_ok && !exact_pair_load) ||
          (!cases[i].expect_ok && memcmp(&part, &cleared, sizeof(part)))) {
         fprintf(stderr, "G17P fragment I/O %s%u %s unexpectedly\n",
                 util_format_name(cases[i].format), cases[i].components,
                 compiled ? "compiled" : "rejected");
         if (compiled && !exact_pair_load) {
            fputs("  binary:", stderr);
            for (unsigned j = 0; j < part.info.binary_size; ++j)
               fprintf(stderr, " %02x", ((const uint8_t *)part.binary)[j]);
            fputc('\n', stderr);
         }
         ok = false;
      }
      free(part.binary);
      ralloc_free(nir);
   }

   return ok;
}

/* Fragment shader matching vkcube's statically indexed set-0 combined image
 * sampler after Honeykrisp descriptor lowering. The Gen9 texture starts at
 * byte 64, followed by the native sampler word at byte 96.
 */
static nir_shader *
build_fs_texture_bias_offset_root(uint32_t bias_offset_B, bool use_root)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 textured FS");

   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *uv = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);

   nir_def *timage = nir_bindless_image_agx(&b, nir_imm_int(&b, 64),
                                            .desc_set = 0);
   nir_def *tsampler = nir_bindless_sampler_agx(
      &b, nir_imm_int(&b, 96), nir_imm_intN_t(&b, 45, 16), .desc_set = 0);

   /* Honeykrisp lowers the API sampler bias to a descriptor load. HAL200
    * carries that bias in the native sampler word, while the software field is
    * always zero, so the bounded lowering removes this source again.
    */
   nir_def *set = nir_load_descriptor_set_agx(&b, 0);
   nir_def *bias_offset = nir_u2u64(&b, nir_imm_int(&b, bias_offset_B));
   nir_def *bias = nir_load_global_constant(
      &b, 1, 16, nir_iadd(&b, set, bias_offset), .align_mul = 4,
      .align_offset = 0, .access = ACCESS_CAN_SPECULATE);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_txb;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, uv);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, timage);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, tsampler);
   tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_bias, bias);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   nir_def *colour = &tex->def;
   if (use_root) {
      nir_def *constant = nir_load_global_constant(
         &b, 1, 32, nir_load_root_agx(&b), .align_mul = 4,
         .access = ACCESS_CAN_SPECULATE);
      colour = nir_fmul(&b, colour, constant);
   }
   fs_store_colour(&b, colour);

   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   return b.shader;
}

static nir_shader *
build_fs_texture_bias_offset(uint32_t bias_offset_B)
{
   return build_fs_texture_bias_offset_root(bias_offset_B, false);
}

static bool
test_combined_sampler_root(void)
{
   nir_shader *shader = build_fs_texture_bias_offset_root(108, true);
   struct agx_g17p_combined_sampler_binding binding;
   bool ok = agx_nir_lower_g17p_combined_sampler(shader, &binding, true);
   unsigned roots = 0, descriptors = 0, samples = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               samples += tex->op == nir_texop_tex && tex->texture_index == 0 &&
                          tex->sampler_index == 0;
            } else if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
               roots += op == nir_intrinsic_load_root_agx;
               descriptors += op == nir_intrinsic_load_descriptor_set_agx ||
                              op == nir_intrinsic_bindless_image_agx ||
                              op == nir_intrinsic_bindless_sampler_agx;
            }
         }
      }
   }
   ok &= roots == 1 && descriptors == 0 && samples == 1 &&
         binding.texture_offset_B == 64 && binding.sampler_index_offset_B == 104;
   ralloc_free(shader);
   return ok;
}

static bool
test_g17g_combined_sampler_scope(void)
{
   /* Positive layout plus independent rejection axes: API LOD, descriptor
    * set, sampler offset, implicit derivatives, and integer interpretation. */
   for (unsigned variant = 0; variant < 6; variant++) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "G17G copy sampler scope");
      nir_def *image = nir_bindless_image_agx(
         &b, nir_imm_int(&b, 64), .desc_set = variant == 2 ? 1 : 0);
      nir_def *sampler = nir_bindless_sampler_agx(
         &b, nir_imm_int(&b, variant == 3 ? 104 : 96),
         nir_imm_intN_t(&b, 3, 16), .desc_set = 0);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, variant == 4 ? 3 : 4);
      tex->op = variant == 4 ? nir_texop_tex : nir_texop_txl;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->dest_type = variant == 5 ? nir_type_uint32 : nir_type_float32;
      tex->coord_components = 2;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                       nir_imm_vec2(&b, 0.5, 0.5));
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
      tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
      if (variant != 4)
         tex->src[3] = nir_tex_src_for_ssa(
            nir_tex_src_lod, nir_imm_float(&b, variant == 1 ? 1.0 : 0.0));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      fs_store_colour(&b, &tex->def);
      struct agx_g17p_combined_sampler_binding binding = {0};
      const bool lowered = agx_nir_lower_g17g_combined_sampler(b.shader, &binding);
      const bool ok = lowered == (variant == 0) &&
         (!lowered || (binding.texture_offset_B == 64 &&
                       binding.sampler_index_offset_B == 104));
      ralloc_free(b.shader);
      if (!ok)
         return false;
   }
   return true;
}

static bool
replace_query_descriptor_load(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_global_constant ||
       !nir_src_is_const(intr->src[0]))
      return false;
   uint64_t address = nir_src_as_uint(intr->src[0]);
   const unsigned size = intr->def.bit_size / 8;
   if (address < 0x1000 || address - 0x1000 +
       size * intr->def.num_components > 32 || size > 8)
      return false;
   nir_const_value values[NIR_MAX_VEC_COMPONENTS] = {0};
   for (unsigned i = 0; i < intr->def.num_components; i++) {
      uint64_t value = 0;
      memcpy(&value, (uint8_t *)data + address - 0x1000 + i * size, size);
      values[i] = nir_const_value_for_uint(value, intr->def.bit_size);
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_build_imm(
      b, intr->def.num_components, intr->def.bit_size, values));
   return true;
}

static bool
test_g17g_texture_queries(void)
{
   for (unsigned variant = 0; variant < 4; variant++) {
      /* Fixed Gen10 descriptor bits deliberately cross the 14-bit Gen9
       * dimension boundary. The expected API values are independent of the
       * generated query code. Test tiled/linear arrays, null, and LOD. */
      uint64_t desc[4] = {
         ((uint64_t)20000 << 28) | ((uint64_t)18002 << 43) |
            ((uint64_t)3 << 58) | (variant == 1 ? 0 : 2u << 4),
         (UINT64_C(0x200000000) >> 4) | (UINT64_C(6) << 46),
         (UINT64_C(1) << 44) | (UINT64_C(3) << 48) | 10,
         0,
      };
      if (variant == 2)
         desc[1] = (AGX_ZERO_PAGE_ADDRESS >> 4) | (UINT64_C(6) << 46);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "G17G descriptor queries");
      nir_def *ptr = nir_imm_int64(&b, 0x1000);
      nir_def *size = libagx_txs_gen10(
         &b, ptr, nir_imm_intN_t(&b, variant == 3 ? 2 : 0, 16),
         nir_imm_int(&b, 3), nir_imm_false(&b), nir_imm_true(&b),
         nir_imm_false(&b), nir_imm_true(&b));
      nir_def *samples = libagx_texture_samples_gen10(&b, ptr);
      nir_def *levels = libagx_texture_levels_gen10(&b, ptr);
      nir_def *result = nir_vec4(&b, nir_channel(&b, size, 0),
         nir_channel(&b, size, 1), nir_channel(&b, size, 2),
         nir_ior(&b, nir_ishl_imm(&b, samples, 8), levels));
      nir_store_output(&b, result, nir_imm_int(&b, 0),
                       .write_mask = 15, .src_type = nir_type_uint32,
                       .io_semantics.location = FRAG_RESULT_DATA0,
                       .io_semantics.num_slots = 1);
      for (unsigned i = 0; i < 4; i++) {
         agx_preprocess_nir(b.shader);
         NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
                  replace_query_descriptor_load, nir_metadata_control_flow,
                  desc);
      }
      agx_preprocess_nir(b.shader);
      bool found = false, ok = true;
      const uint32_t expected[4] = {
         variant == 2 ? 0 : variant == 3 ? 2500 : 10000,
         variant == 2 ? 0 : variant == 3 ? 2250 : 9001,
         variant == 2 ? 0 : variant == 1 ? 11 : 7,
         variant == 2 ? 0 : 0x103,
      };
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_store_output)
                  continue;
               found = true;
               for (unsigned c = 0; c < intr->num_components; c++) {
                  const unsigned component = nir_intrinsic_component(intr) + c;
                  ok &= nir_src_is_const(intr->src[0]) && component < 4 &&
                     nir_src_comp_as_uint(intr->src[0], c) == expected[component];
               }
            }
         }
      }
      if (!found || !ok)
         nir_print_shader(b.shader, stderr);
      ralloc_free(b.shader);
      if (!found || !ok)
         return false;
   }
   return true;
}

static bool
test_indexed_sampler_sets_stage(mesa_shader_stage stage)
{
   nir_builder b = nir_builder_init_simple_shader(
      stage, &agx_nir_options, "indexed sample sets");
   nir_def *root = nir_load_root_agx(&b);
   nir_def *element = nir_load_global_constant(&b, 1, 32, root, .align_mul=4);
   nir_def *offset = nir_imul_imm(&b, element, 72);
   nir_def *result = nir_imm_vec4(&b, 0, 0, 0, 0);
   const unsigned sets[] = {0, 3, 7};
   for (unsigned i = 0; i < ARRAY_SIZE(sets); ++i) {
      nir_def *address = nir_load_descriptor_set_agx(&b, sets[i]);
      address = nir_iadd(&b, address, nir_u2u64(&b, offset));
      nir_def *sampler_index = nir_load_global_constant(
         &b, 1, 16, nir_iadd_imm(&b, address, 40), .align_mul=2);
      nir_def *image = nir_bindless_image_agx(&b, offset, .desc_set=sets[i]);
      nir_def *sampler = nir_bindless_sampler_agx(
         &b, nir_iadd_imm(&b, offset, 32), sampler_index, .desc_set=sets[i]);
      const bool explicit_lod = stage != MESA_SHADER_FRAGMENT;
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, explicit_lod ? 4 : 3);
      tex->op = explicit_lod ? nir_texop_txl : nir_texop_tex;
      tex->sampler_dim=GLSL_SAMPLER_DIM_2D;
      tex->dest_type=nir_type_float32; tex->is_array=i==2;
      tex->coord_components=tex->is_array?3:2;
      tex->src[0]=nir_tex_src_for_ssa(
         nir_tex_src_coord,
         tex->is_array
            ? nir_vec3(&b, nir_imm_float(&b, .25), nir_imm_float(&b, .75),
                       nir_imm_int(&b, 1))
            : nir_imm_vec2(&b, .25, .75));
      tex->src[1]=nir_tex_src_for_ssa(nir_tex_src_texture_handle,image);
      tex->src[2]=nir_tex_src_for_ssa(nir_tex_src_sampler_handle,sampler);
      if (explicit_lod)
         tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_lod,
                                          nir_imm_float(&b, 0.0f));
      nir_def_init(&tex->instr,&tex->def,4,32);
      nir_builder_instr_insert(&b,&tex->instr);
      result=nir_fadd(&b,result,&tex->def);
   }
   nir_store_output(&b,result,nir_imm_int(&b,0),.write_mask=15,
                    .src_type=nir_type_float32);
   bool ok=agx_nir_lower_g17p_indexed_samplers(b.shader, true);
   unsigned samples=0, arrays=0, descriptors=0, roots=0, image_loads=0,
            sampler_loads=0, set_mask=0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr,block) {
         if (instr->type==nir_instr_type_tex) {
            nir_tex_instr *tex=nir_instr_as_tex(instr);
            int ti=nir_tex_instr_src_index(tex,nir_tex_src_texture_offset);
            int si=nir_tex_instr_src_index(tex,nir_tex_src_sampler_handle);
            ok &= ti>=0 && si>=0;
            if (ti>=0 && si>=0)
               ok &= nir_src_bit_size(tex->src[ti].src)==64 &&
                     nir_src_bit_size(tex->src[si].src)==64;
            samples++;
            arrays += tex->is_array;
         } else if (instr->type==nir_instr_type_intrinsic) {
            nir_intrinsic_instr *in=nir_instr_as_intrinsic(instr);
            roots += in->intrinsic==nir_intrinsic_load_root_agx;
            descriptors += in->intrinsic==nir_intrinsic_bindless_image_agx ||
                           in->intrinsic==nir_intrinsic_bindless_sampler_agx;
            if (in->intrinsic==nir_intrinsic_load_descriptor_set_agx)
               set_mask |= 1u << nir_intrinsic_desc_set(in);
            if (in->intrinsic==nir_intrinsic_load_global_constant) {
               image_loads += in->def.bit_size==32;
               sampler_loads += in->def.bit_size==16;
            }
         }
      }
   }
   ok &= samples==3 && arrays==1 && descriptors==0 && roots==1 && image_loads==4 &&
         sampler_loads==3 && set_mask==((1u<<0)|(1u<<3)|(1u<<7));
   ralloc_free(b.shader);
   return ok;
}

static bool
test_indexed_sampler_sets(void)
{
   /* Geometry runs through the same compute texture heap ABI as compute
    * shaders. A fresh indexed vertex Metal witness grounds the explicit-level
    * register texture/sampler instruction used after this NIR rewrite; Metal
    * has no geometry stage to target directly. */
   return test_indexed_sampler_sets_stage(MESA_SHADER_FRAGMENT) &&
          test_indexed_sampler_sets_stage(MESA_SHADER_GEOMETRY) &&
          test_indexed_sampler_sets_stage(MESA_SHADER_TESS_CTRL) &&
          test_indexed_sampler_sets_stage(MESA_SHADER_TESS_EVAL);
}

static bool
test_indexed_storage_image_read(void)
{
   /* mode 0 is the fully specified positive case.  1 and 3 used to be
    * negatives because g17p_indexed_image_fetch_shape() admitted only
    * PIPE_FORMAT_R8G8B8A8_UNORM and only non-array 2D; neither was a
    * constraint.  agx3_emit_texture_fetch() never reads I->format, and
    * metal-nt -arch applegpu_g17p emits byte-identical code for
    * texture2d<float>, texture2d<uint> and texture2d<int> read(), so the
    * component type is in the descriptor.  3D and 2D array are dimensions the
    * same encoder already takes. Storage cube and cube-array descriptors are
    * also 2D arrays, matching agx_emit_image_load()'s existing dimension
    * transform. Mode 2 is a coherent, volatile read, which takes the same
    * native fetch. Mode 7 keeps the negative coverage on a runtime LOD
    * operand. Mode 8 is the 1D-array shape of the meta image-copy kernel:
    * lower_1d_image() maps it onto the 2D-array fetch before emit. Mode 9 is
    * the nir_opt_access NON_READABLE stamp the meta copy puts on loads. */
   for (unsigned mode = 0; mode < 10; ++mode) {
      const bool expect_lowered = mode != 7;
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options,
         "indexed read-only storage image");
      b.shader->info.workgroup_size[0] = 1;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      nir_def *handle = nir_bindless_image_agx(
         &b, nir_imm_int(&b, 128), .desc_set = 4);
      /* Keep the original descriptor handle live as the real robustness
       * lowering does. The hardware image read gets a distinct heap-indexed
       * handle; this crawl must continue to use the descriptor set. */
      nir_def *descriptor = nir_load_from_texture_handle_agx(&b, handle);
      nir_def *width = nir_load_global_constant(
         &b, 1, 32, descriptor, .align_mul = 4);
      nir_def *zero = nir_imm_int(&b, 0);
      nir_def *coord = nir_vec4(&b, nir_iand(&b, width, zero), zero,
                                zero, zero);
      nir_def *value = nir_bindless_image_load(
         &b, mode == 0 ? 1 : 4, 32, handle, coord,
         mode == 7 ? nir_channel(&b, coord, 1) : zero,
         nir_imm_intN_t(&b, 0, 16),
         .image_dim = mode == 3     ? GLSL_SAMPLER_DIM_3D
                      : mode == 8   ? GLSL_SAMPLER_DIM_1D
                      : mode >= 5   ? GLSL_SAMPLER_DIM_CUBE
                                    : GLSL_SAMPLER_DIM_2D,
         .image_array = mode == 4 || mode == 6 || mode == 8,
         .format = mode == 1 ? PIPE_FORMAT_R8G8B8A8_UINT
                             : PIPE_FORMAT_R8G8B8A8_UNORM,
         .access = ACCESS_RESTRICT |
                   (mode == 2 ? ACCESS_COHERENT | ACCESS_VOLATILE
                    : mode == 9 ? ACCESS_NON_READABLE
                                : ACCESS_NON_WRITEABLE));
      nir_export_agx(&b, nir_channel(&b, value, 0), .base = 0);

      const bool lowered = agx_nir_lower_g17p_indexed_samplers(b.shader, true);
      unsigned original_handles = 0, native_handles = 0;
      unsigned heap_index_loads = 0, set4_loads = 0;
      nir_intrinsic_instr *read = NULL;
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_bindless_image_agx) {
                  if (nir_intrinsic_desc_set(intr) == 4)
                     original_handles++;
                  else if (nir_intrinsic_desc_set(intr) == 0)
                     native_handles++;
               } else if (intr->intrinsic ==
                          nir_intrinsic_load_descriptor_set_agx) {
                  set4_loads += nir_intrinsic_desc_set(intr) == 4;
               } else if (intr->intrinsic ==
                             nir_intrinsic_load_global_constant &&
                          intr->def.bit_size == 32) {
                  heap_index_loads++;
               } else if (intr->intrinsic ==
                          nir_intrinsic_bindless_image_load) {
                  read = intr;
               }
            }
         }
      }

      bool ok;
      if (mode == 0) {
         nir_intrinsic_instr *native =
            read ? nir_src_as_intrinsic(read->src[0]) : NULL;
         nir_alu_instr *extend = native ?
            nir_def_as_alu_or_null(native->src[0].ssa) : NULL;
         nir_intrinsic_instr *index =
            extend && extend->op == nir_op_u2u64 ?
               nir_src_as_intrinsic(extend->src[0].src) : NULL;
         ok = lowered && read && native &&
              native->intrinsic == nir_intrinsic_bindless_image_agx &&
              nir_intrinsic_desc_set(native) == 0 &&
              native->src[0].ssa->bit_size == 64 && index &&
              index->intrinsic == nir_intrinsic_load_global_constant &&
              index->def.bit_size == 32 && original_handles == 1 &&
              native_handles == 1 && heap_index_loads == 2 && set4_loads == 1;
      } else if (expect_lowered) {
         ok = lowered && read && original_handles == 1 &&
              native_handles == 1 && set4_loads == 1;
      } else {
         ok = !lowered && original_handles == 1 && native_handles == 0;
      }
      ralloc_free(b.shader);
      if (!ok)
         return false;
   }

   return true;
}

static bool
lower_test_descriptor_set(nir_builder *b, nir_intrinsic_instr *intr,
                          UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_descriptor_set_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_load_preamble(b, 1, 64,
                                     .base = 4 * nir_intrinsic_desc_set(intr)));
   return true;
}

static bool
lower_test_g17p_descriptor_set(nir_builder *b, nir_intrinsic_instr *intr,
                               UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_descriptor_set_agx)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def,
                   nir_load_preamble(b, 1, 64,
                                     .base = AGX_ABI_FUNI_ROOT + 4 +
                                             4 * nir_intrinsic_desc_set(intr)));
   return true;
}

static bool
contains_integer_lod_stage(const struct agx_shader_part *part)
{
   const uint8_t *binary = part->binary;
   uint8_t candidate[AGX3_TEX_MAX_LENGTH], length;

   for (unsigned src = 0; src < 256; src += 2) {
      for (unsigned slot = 1; slot <= 6; ++slot) {
         for (unsigned wait = 0; wait < 64; ++wait) {
            for (unsigned hint = 0; hint < 3; ++hint) {
               if (!agx3_tex_stage_typed_lod(
                      AGX_ISA_ENCODING_AGX3B, src, slot, wait,
                      hint == 1, hint == 2, AGX3_TEX_LEVEL_INTEGER,
                      candidate, &length))
                  continue;

               for (unsigned offset = 0;
                    offset + length <= part->info.binary_size; ++offset) {
                  if (!memcmp(binary + offset, candidate, length))
                     return true;
               }
            }
         }
      }
   }

   return false;
}

/* Keep the exact constant-level route tied to the full compiler path. NIR
 * legalizes texture LODs to 16 bits for AGX2, while native G17P read() takes a
 * uint. A literal must use Apple's direct f29 level field even when constant
 * promotion is enabled, rather than manufacturing a per-lane dependency. */
static bool
test_g17p_fetch_lod_compile(unsigned dynamic_bits)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P constant and runtime texel-fetch levels");
   nir_def *image =
      nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 0);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_backend1, nir_imm_ivec2(&b, 3, 4));
   tex->src[1] =
      nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   nir_def *level = nir_imm_intN_t(&b, 2, 16);
   if (dynamic_bits) {
      level = nir_load_preamble(&b, 1, 32, .base = 12);
      if (dynamic_bits == 16)
         level = nir_u2u16(&b, level);
   }
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, level);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   fs_store_colour(&b, &tex->def);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   bool ok = agx_nir_lower_g17p_indexed_samplers(b.shader, true);
   const int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   ok &= lod >= 0 && nir_src_bit_size(tex->src[lod].src) == 32;
   if (dynamic_bits)
      ok &= !nir_src_is_const(tex->src[lod].src);
   else
      ok &= nir_src_is_const(tex->src[lod].src) &&
            nir_src_as_uint(tex->src[lod].src) == 2;

   if (!ok) {
      ralloc_free(b.shader);
      return false;
   }

   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
            lower_test_descriptor_set, nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
      .no_stop = true,
      .promote_constants = true,
      .reserved_preamble = 16,
   };
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &key, &part) && part.binary &&
         (contains_integer_lod_stage(&part) == (dynamic_bits != 0));
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static bool
test_g17p_storage_read_zero_lod_compile(void)
{
   for (unsigned mode = 0; mode < 3; ++mode) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "G17P storage image level zero");
      b.shader->info.workgroup_size[0] = 1;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
      nir_def *gid = nir_load_global_invocation_id(&b, 32);
      nir_def *zero = nir_imm_int(&b, 0);
      nir_def *coord = nir_vec4(&b, nir_channel(&b, gid, 0),
                               nir_channel(&b, gid, 1), zero, zero);
      const enum pipe_format format = mode == 2 ? PIPE_FORMAT_R8G8B8A8_UINT
                                                : PIPE_FORMAT_R32_UINT;
      nir_def *handle = mode == 0 ? nir_bindless_image_agx(
         &b, nir_imm_int(&b, 64), .desc_set = 0) : zero;
      nir_def *value = mode == 0 ? nir_bindless_image_load(
         &b, 4, 32, handle, coord, zero, nir_imm_intN_t(&b, 0, 16),
         .image_dim = GLSL_SAMPLER_DIM_2D, .format = format,
         .access = ACCESS_NON_WRITEABLE) : nir_image_load(
         &b, 4, 32, handle, coord, zero, nir_imm_intN_t(&b, 0, 16),
         .image_dim = GLSL_SAMPLER_DIM_2D, .format = format,
         .access = ACCESS_NON_WRITEABLE);
      nir_export_agx(&b, nir_channel(&b, value, 0), .base = 0);

      bool ok = mode != 0 || agx_nir_lower_g17p_indexed_samplers(b.shader, true);
      NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
               lower_test_descriptor_set, nir_metadata_control_flow, NULL);
      agx_preprocess_nir(b.shader);
      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .no_preamble = true,
         .promote_constants = true,
         .reserved_preamble = 16,
      };
      struct agx_shader_part part = {0};
      ok &= agx_compile_shader_nir(b.shader, &key, &part) && part.binary &&
            part.info.uses_txf && !contains_integer_lod_stage(&part);
      free(part.binary);
      ralloc_free(b.shader);
      if (!ok)
         return false;
   }
   return true;
}

struct formatless_load_fixture {
   uint8_t texture[32];
};

static bool
fold_formatless_descriptor(nir_builder *b, nir_intrinsic_instr *intr,
                           void *data)
{
   const struct formatless_load_fixture *f = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = NULL;
   if (intr->intrinsic == nir_intrinsic_load_from_texture_handle_agx) {
      value = nir_imm_int64(b, 0x1000);
   } else if (intr->intrinsic == nir_intrinsic_load_global_constant &&
              nir_src_is_const(intr->src[0]) &&
              intr->def.num_components == 1 && intr->def.bit_size == 32) {
      const uint64_t offset = nir_src_as_uint(intr->src[0]) - 0x1000;
      if (offset > sizeof(f->texture) - 4)
         return false;
      uint32_t word;
      memcpy(&word, f->texture + offset, sizeof(word));
      value = nir_imm_int(b, word);
   }
   if (!value)
      return false;
   nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

/* A formatless load has no format to fold its robustImageAccess2 value from.
 * With view-bounds robustness, G17P reads texel 0 for an out-of-bounds
 * coordinate and selects zero with the view's missing-alpha one afterwards:
 * (0, 0, 0, 1) for an R32F view, (0, 0, 0, 0) for an RGBA32F view. A formatted
 * load, or one without robustness, is left alone. */
static bool
test_g17p_formatless_robust_image_load(void)
{
   for (unsigned robust = 0; robust < 2; ++robust) {
      for (unsigned formatted = 0; formatted < 2; ++formatted) {
         for (unsigned alpha_one = 0; alpha_one < 2; ++alpha_one) {
            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_COMPUTE, &agx_nir_options,
               "G17P formatless robust image load");
            nir_def *handle =
               nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 0);
            /* x = 7 is outside the 4x4 image below. */
            nir_def *load = nir_bindless_image_load(
               &b, 4, 32, handle, nir_imm_ivec4(&b, 7, 3, 0, 0),
               nir_imm_int(&b, 0), nir_imm_intN_t(&b, 0, 16),
               .image_dim = GLSL_SAMPLER_DIM_2D,
               .format = formatted ? PIPE_FORMAT_R32_FLOAT : PIPE_FORMAT_NONE,
               .dest_type = nir_type_float32);
            nir_store_ssbo(&b, load, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                           .write_mask = 15, .align_mul = 16);
            b.shader->info.num_ssbos = 1;
            agx_nir_lower_texture_with_buffer_metadata(b.shader, 9, robust, 0, false);

            /* The surviving read is clamped by a select on the predicate. */
            bool clamped = false;
            nir_foreach_function_impl(impl, b.shader) {
               nir_foreach_block(block, impl) {
                  nir_foreach_instr(instr, block) {
                     if (instr->type != nir_instr_type_intrinsic ||
                         nir_instr_as_intrinsic(instr)->intrinsic !=
                            nir_intrinsic_bindless_image_load)
                        continue;
                     nir_instr *coord =
                        nir_def_instr(nir_instr_as_intrinsic(instr)->src[1].ssa);
                     clamped = coord->type == nir_instr_type_alu &&
                               nir_instr_as_alu(coord)->op == nir_op_bcsel;
                  }
               }
            }

            /* Gen9 texture: 2D, 4x4, swizzle A = 1 (Channel 1) or A. */
            struct formatless_load_fixture f = {0};
            const uint64_t word0 = 2 | (2 << 4) |
                                   ((alpha_one ? 4ull : 3ull) << 25) |
                                   (3ull << 28) | (3ull << 42);
            const uint64_t word1 = UINT64_C(0x123456780) >> 4;
            memcpy(f.texture, &word0, 8);
            memcpy(f.texture + 8, &word1, 8);
            bool progress;
            do {
               progress = nir_shader_intrinsics_pass(
                  b.shader, fold_formatless_descriptor,
                  nir_metadata_control_flow, &f);
               progress |= nir_opt_constant_folding(b.shader);
               progress |= nir_opt_algebraic(b.shader);
               progress |= nir_opt_dead_cf(b.shader);
               progress |= nir_opt_remove_phis(b.shader);
               progress |= nir_opt_dce(b.shader);
               progress |= nir_opt_copy_prop(b.shader);
            } while (progress);

            bool constant = false;
            float stored[4] = {0};
            nir_foreach_function_impl(impl, b.shader) {
               nir_foreach_block(block, impl) {
                  nir_foreach_instr(instr, block) {
                     if (instr->type != nir_instr_type_intrinsic)
                        continue;
                     nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                     if (intr->intrinsic != nir_intrinsic_store_ssbo)
                        continue;
                     constant = nir_src_is_const(intr->src[0]);
                     for (unsigned c = 0; constant && c < 4; ++c)
                        stored[c] = nir_src_comp_as_float(intr->src[0], c);
                  }
               }
            }
            ralloc_free(b.shader);

            const bool lowered = robust && !formatted;
            if (lowered != constant || lowered != clamped)
               return false;
            if (lowered &&
                (stored[0] != 0.0f || stored[1] != 0.0f || stored[2] != 0.0f ||
                 stored[3] != (alpha_one ? 1.0f : 0.0f)))
               return false;
         }
      }
   }
   return true;
}

/* An invalid texelFetch on G17P likewise returns zero with the view's
 * missing-alpha one, not zero alone. */
static bool
test_g17p_robust_texel_fetch_value(void)
{
   for (unsigned alpha_one = 0; alpha_one < 2; ++alpha_one) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "G17P robust texel fetch");
      nir_def *handle =
         nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 0);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
      tex->op = nir_texop_txf;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->dest_type = nir_type_int32;
      tex->coord_components = 2;
      /* x = 7 is outside the 4x4 texture below. */
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                      nir_imm_ivec2(&b, 7, 3));
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
      tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(&b, 0));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      nir_store_ssbo(&b, &tex->def, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                     .write_mask = 15, .align_mul = 16);
      b.shader->info.num_ssbos = 1;
      agx_nir_lower_texture_with_buffer_metadata(b.shader, 9, true, 0, false);

      struct formatless_load_fixture f = {0};
      const uint64_t word0 = 2 | (2 << 4) |
                             ((alpha_one ? 4ull : 3ull) << 25) |
                             (3ull << 28) | (3ull << 42);
      const uint64_t word1 = UINT64_C(0x123456780) >> 4;
      memcpy(f.texture, &word0, 8);
      memcpy(f.texture + 8, &word1, 8);
      bool progress;
      do {
         progress = nir_shader_intrinsics_pass(
            b.shader, fold_formatless_descriptor, nir_metadata_control_flow,
            &f);
         progress |= nir_opt_constant_folding(b.shader);
         progress |= nir_opt_algebraic(b.shader);
         progress |= nir_opt_dead_cf(b.shader);
         progress |= nir_opt_remove_phis(b.shader);
         progress |= nir_opt_dce(b.shader);
         progress |= nir_opt_copy_prop(b.shader);
      } while (progress);

      bool ok = false;
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_store_ssbo ||
                   !nir_src_is_const(intr->src[0]))
                  continue;
               ok = nir_src_comp_as_uint(intr->src[0], 0) == 0 &&
                    nir_src_comp_as_uint(intr->src[0], 1) == 0 &&
                    nir_src_comp_as_uint(intr->src[0], 2) == 0 &&
                    nir_src_comp_as_uint(intr->src[0], 3) == alpha_one;
            }
         }
      }
      ralloc_free(b.shader);
      if (!ok)
         return false;
   }
   return true;
}

struct buffer_metadata_fixture {
   uint32_t elements, tail, coord;
   bool rgb32, saw_data;
   unsigned loads;
};

static bool
fold_buffer_descriptor(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct buffer_metadata_fixture *f = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = NULL;
   if (intr->intrinsic == nir_intrinsic_load_from_texture_handle_agx) {
      value = nir_imm_int64(b, 0x1000);
   } else if (intr->intrinsic == nir_intrinsic_load_global_constant &&
              nir_src_is_const(intr->src[0])) {
      const uint64_t address = nir_src_as_uint(intr->src[0]);
      switch (address) {
      case 0x1000 + AGX_G17P_BUFFER_TEXTURE_SIZE_OFFSET:
         value = nir_imm_int(b, f->elements);
         break;
      case 0x1000 + AGX_G17P_BUFFER_TEXTURE_TAIL_OFFSET:
      case 0x1000 + AGX_G17P_BUFFER_TEXTURE_TAIL_OFFSET -
              AGX_G17P_BUFFER_TEXTURE_PBE_OFFSET:
         value = nir_imm_int(b, f->tail);
         break;
      case 0x1000 + AGX_G17P_BUFFER_TEXTURE_RGB32_OFFSET:
         value = nir_imm_int(b, f->rgb32);
         break;
      case 0x1000 + AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET:
      case 0x1000 + AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET -
              AGX_G17P_BUFFER_TEXTURE_PBE_OFFSET:
         value = nir_imm_int64(b, UINT64_C(0x123000000000));
         break;
      default:
         if (!f->rgb32 || intr->def.num_components != 3 ||
             address != UINT64_C(0x123000000000) + (uint64_t)f->coord * 12)
            return false;
         value = nir_imm_ivec3(b, 11, 101, 83);
         f->saw_data = true;
         break;
      }
      f->loads++;
   }
   if (!value)
      return false;
   nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

static bool
test_g17p_buffer_metadata(void)
{
   const struct {
      uint32_t elements, tail, coord, x, y;
      bool rgb32;
   } cases[] = {
      {64, 0, 6, 6, 0},
      {64, 3, 63, 66, 0},
      {16384, 1, 16383, 0, 1},
      {64, 0, 64, 16383, 262143},
      {0, 0, 6, 16383, 262143},
      {64, 3, 6, 0, 0, true},
   };
   for (unsigned c = 0; c < ARRAY_SIZE(cases); ++c) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "G17P buffer metadata");
      nir_def *handle = nir_bindless_image_agx(
         &b, nir_imm_int(&b, 0), .desc_set = 0);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
      tex->op = nir_texop_txf;
      tex->sampler_dim = GLSL_SAMPLER_DIM_BUF;
      tex->dest_type = nir_type_uint32;
      tex->coord_components = 1;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                      nir_imm_int(&b, cases[c].coord));
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
      tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(&b, 0));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      nir_store_ssbo(&b, &tex->def, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                     .write_mask = 15, .align_mul = 16);
      b.shader->info.num_ssbos = 1;
      agx_nir_lower_texture_with_buffer_metadata(b.shader, 9, false, 0, false);

      struct buffer_metadata_fixture f = {
         .elements = cases[c].elements, .tail = cases[c].tail,
         .coord = cases[c].coord, .rgb32 = cases[c].rgb32,
      };
      bool progress;
      do {
         progress = nir_shader_intrinsics_pass(b.shader,
            fold_buffer_descriptor, nir_metadata_control_flow, &f);
         progress |= nir_opt_constant_folding(b.shader);
         progress |= nir_opt_dead_cf(b.shader);
         progress |= nir_opt_dce(b.shader);
         progress |= nir_opt_copy_prop(b.shader);
      } while (progress);

      bool ok = false;
      if (cases[c].rgb32) {
         unsigned textures = 0;
         nir_foreach_function_impl(impl, b.shader) {
            nir_foreach_block(block, impl) {
               nir_foreach_instr(instr, block) {
                  textures += instr->type == nir_instr_type_tex;
                  if (instr->type != nir_instr_type_intrinsic)
                     continue;
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic == nir_intrinsic_store_ssbo &&
                      nir_src_is_const(intr->src[0])) {
                     ok = nir_src_comp_as_uint(intr->src[0], 0) == 11 &&
                          nir_src_comp_as_uint(intr->src[0], 1) == 101 &&
                          nir_src_comp_as_uint(intr->src[0], 2) == 83 &&
                          nir_src_comp_as_uint(intr->src[0], 3) == 1;
                  }
               }
            }
         }
         ok &= f.saw_data && textures == 0;
      } else {
         const int coord = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
         ok = f.loads >= 3 && tex->sampler_dim == GLSL_SAMPLER_DIM_2D &&
              coord >= 0 && nir_src_is_const(tex->src[coord].src) &&
              nir_src_comp_as_uint(tex->src[coord].src, 0) == cases[c].x &&
              nir_src_comp_as_uint(tex->src[coord].src, 1) == cases[c].y;
      }
      ralloc_free(b.shader);
      if (!ok) {
         fprintf(stderr, "buffer metadata case %u failed\n", c);
         return false;
      }
   }
   return true;
}

/* A buffer atomic addresses its view from the descriptor's exact address. The
 * tail offset counts the view's elements in its texture, which need not be the
 * atomic's: a 32-bit atomic through an R16 view in a texel window. */
static bool
test_g17p_buffer_atomic_address(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P buffer atomic address");
   nir_def *pbe = nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 0);
   nir_def *address = nir_bindless_image_texel_address(
      &b, 64, pbe, nir_imm_ivec4(&b, 5, 0, 0, 0), nir_imm_int(&b, 0),
      .image_dim = GLSL_SAMPLER_DIM_BUF, .image_array = false,
      .format = PIPE_FORMAT_R32_UINT);
   nir_store_ssbo(&b, address, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                  .write_mask = 1, .align_mul = 8);
   b.shader->info.num_ssbos = 1;
   agx_nir_lower_texture_with_buffer_metadata(b.shader, 9, false, 0, false);

   struct buffer_metadata_fixture f = {.elements = 64, .tail = 1 << 20};
   bool progress;
   do {
      progress = nir_shader_intrinsics_pass(b.shader, fold_buffer_descriptor,
                                            nir_metadata_control_flow, &f);
      progress |= nir_opt_constant_folding(b.shader);
      progress |= nir_opt_dce(b.shader);
      progress |= nir_opt_copy_prop(b.shader);
   } while (progress);

   bool ok = false;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_store_ssbo &&
                nir_src_is_const(intr->src[0]))
               ok = nir_src_as_uint(intr->src[0]) ==
                    UINT64_C(0x123000000000) + 5 * 4;
         }
      }
   }
   ralloc_free(b.shader);
   if (!ok)
      fprintf(stderr, "buffer atomic address counts the tail offset\n");
   return ok;
}

static bool
test_g17p_buffer_image_metadata(void)
{
   for (unsigned tail = 0; tail < 4; ++tail) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "G17P imageBuffer metadata");
      nir_def *zero = nir_imm_int(&b, 0);
      nir_def *handle = nir_bindless_image_agx(&b, zero, .desc_set = 0);
      nir_def *coord = nir_imm_ivec4(&b, 16383, 0, 0, 0);
      nir_def *value = nir_bindless_image_load(
         &b, 4, 32, handle, coord, zero, zero,
         .image_dim = GLSL_SAMPLER_DIM_BUF,
         .format = PIPE_FORMAT_R8G8B8A8_UINT,
         .access = ACCESS_NON_WRITEABLE);
      nir_intrinsic_instr *load = nir_def_as_intrinsic(value);
      nir_def *size = nir_bindless_image_size(
         &b, 1, 32, handle, zero, .image_dim = GLSL_SAMPLER_DIM_BUF);
      nir_export_agx(&b, value, .base = 0);
      nir_export_agx(&b, size, .base = 8);
      /* A store's handle is the PBE, AGX_G17P_BUFFER_TEXTURE_PBE_OFFSET into
       * the descriptor; the fixture folds either handle to the same address,
       * so the tail must come from the PBE-relative metadata offset. */
      nir_def *pbe = nir_bindless_image_agx(
         &b, nir_imm_int(&b, AGX_G17P_BUFFER_TEXTURE_PBE_OFFSET),
         .desc_set = 0);
      nir_intrinsic_instr *store = nir_bindless_image_store(
         &b, pbe, coord, zero, value, zero,
         .image_dim = GLSL_SAMPLER_DIM_BUF,
         .format = PIPE_FORMAT_R8G8B8A8_UINT, .src_type = nir_type_uint32,
         .access = ACCESS_NON_READABLE);
      agx_nir_lower_texture_with_buffer_metadata(b.shader, 9, false, 0, false);
      struct buffer_metadata_fixture f = {.elements = 16384, .tail = tail};
      bool progress;
      do {
         progress = nir_shader_intrinsics_pass(b.shader,
            fold_buffer_descriptor, nir_metadata_control_flow, &f);
         progress |= nir_opt_constant_folding(b.shader);
         progress |= nir_opt_dce(b.shader);
      } while (progress);
      bool size_ok = false;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_export_agx &&
                nir_intrinsic_base(intr) == 8)
               size_ok = nir_src_is_const(intr->src[0]) &&
                         nir_src_as_uint(intr->src[0]) == 16384;
         }
      }
      nir_scalar x = nir_scalar_resolved(load->src[1].ssa, 0);
      nir_scalar y = nir_scalar_resolved(load->src[1].ssa, 1);
      nir_scalar store_x = nir_scalar_resolved(store->src[1].ssa, 0);
      nir_scalar store_y = nir_scalar_resolved(store->src[1].ssa, 1);
      bool ok = size_ok && f.loads == 3 &&
         nir_intrinsic_image_dim(load) == GLSL_SAMPLER_DIM_2D &&
         nir_intrinsic_image_dim(store) == GLSL_SAMPLER_DIM_2D &&
         nir_scalar_is_const(x) && nir_scalar_is_const(y) &&
         nir_scalar_as_uint(x) == ((16383 + tail) & 16383) &&
         nir_scalar_as_uint(y) == ((16383 + tail) >> 14) &&
         nir_scalar_is_const(store_x) && nir_scalar_is_const(store_y) &&
         nir_scalar_as_uint(store_x) == nir_scalar_as_uint(x) &&
         nir_scalar_as_uint(store_y) == nir_scalar_as_uint(y);
      if (!ok)
         fprintf(stderr, "imageBuffer tail %u: size=%u loads=%u\n",
                 tail, size_ok, f.loads);
      ralloc_free(b.shader);
      if (!ok)
         return false;
   }
   return true;
}

static bool
test_g17p_query_lod_compile(void)
{
   static const struct {
      enum glsl_sampler_dim dim;
      bool array;
      unsigned coords;
   } shapes[] = {
      {GLSL_SAMPLER_DIM_2D, false, 2},
      {GLSL_SAMPLER_DIM_3D, false, 3},
      {GLSL_SAMPLER_DIM_CUBE, false, 3},
      {GLSL_SAMPLER_DIM_2D, true, 2},
      {GLSL_SAMPLER_DIM_CUBE, true, 3},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(shapes); ++i) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P LOD query");
      nir_def *bary = nir_load_barycentric_pixel(
         &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
      nir_def *coord = nir_load_interpolated_input(
         &b, shapes[i].coords, 32, bary, nir_imm_int(&b, 0),
         .io_semantics.location = VARYING_SLOT_VAR0,
         .io_semantics.num_slots = 1);
      nir_def *image =
         nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 0);
      nir_def *set = nir_load_descriptor_set_agx(&b, 0);
      nir_def *heap_index = nir_load_global_constant(
         &b, 1, 16, nir_iadd_imm(&b, set, 40), .align_mul = 2,
         .access = ACCESS_CAN_SPECULATE);
      nir_def *sampler = nir_bindless_sampler_agx(
         &b, nir_imm_int(&b, 32), heap_index, .desc_set = 0);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
      tex->op = nir_texop_lod;
      tex->sampler_dim = shapes[i].dim;
      tex->is_array = shapes[i].array;
      tex->dest_type = nir_type_float32;
      tex->coord_components = shapes[i].coords;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
      tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
      nir_def_init(&tex->instr, &tex->def, 2, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      fs_store_colour(&b, nir_vec4(&b, nir_channel(&b, &tex->def, 0),
                                   nir_channel(&b, &tex->def, 1),
                                   nir_imm_float(&b, 0), nir_imm_float(&b, 1)));
      b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
      b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
      NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
               AGX_DESCRIPTOR_GENERATION_9);
      struct agx_g17p_combined_sampler_binding binding;
      bool ok = tex->coord_components == shapes[i].coords &&
                agx_nir_lower_g17p_combined_sampler(b.shader, &binding, true);
      agx_preprocess_nir(b.shader);
      fs_lower(b.shader);
      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .no_preamble = true, .reserved_preamble = 24,
      };
      struct agx_shader_part part = {0};
      ok &= agx_compile_shader_nir(b.shader, &key, &part) &&
            part.binary && !part.info.scratch_size &&
            part.info.agx3_staging_known &&
            part.info.agx3_staging_halfs ==
               (shapes[i].dim == GLSL_SAMPLER_DIM_2D ? 4 : 8);
      free(part.binary);
      ralloc_free(b.shader);
      if (!ok) {
         fprintf(stderr, "LOD query compile failed for shape %u\n", i);
         return false;
      }
   }
   return true;
}

/* Dump the driver's cube-array depth-comparison sequence (varying direction,
 * layer and reference, immediate descriptors) so it can be diffed against
 * Apple's `depthcube_array.sample_compare(s, dir, layer, ref)` fragment.
 * KHR-GL30.ext_texture_shadow_lod.texture.samplercubearrayshadow_fragment
 * compiles but renders a mismatch; this is the byte-level evidence. */
/* The transform feedback path compiles the vertex shader in its compute
 * form: poly_nir_lower_vs_before_gs() turns each output store into a global
 * store at the vertex parameter buffer, the stage becomes COMPUTE and the
 * ordinary vertex prolog exports the zero-based vertex ID. Mirror that
 * pipeline (the driver's sysval lowering maps the parameter buffer to a
 * root uniform, done here as a preamble load) and report what the AGX3
 * packer refuses: KHR-GL30.transform_feedback.capture_vertex_* fail to link
 * with "cannot encode the default vertex shader variant". */
static bool
lower_vertex_param_buffer(nir_builder *b, nir_intrinsic_instr *intr,
                          void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_vertex_param_buffer_poly)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_load_preamble(b, 1, 64, .base = 24));
   return true;
}

static bool
test_g17p_compute_form_vertex_shader(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G17P compute-form VS");
   nir_def *id = nir_u2f32(&b, nir_load_vertex_id(&b));
   nir_def *iid = nir_u2f32(&b, nir_load_instance_id(&b));
   nir_def *pos = nir_vec4(&b, id, iid, nir_imm_float(&b, 0.0f),
                           nir_imm_float(&b, 1.0f));
   nir_def *var = nir_vec4(&b, nir_fmul_imm(&b, id, 0.5f), iid,
                           nir_imm_float(&b, 0.25f), nir_imm_float(&b, 0.75f));
   nir_store_output(&b, pos, nir_imm_int(&b, 0), .base = 0, .write_mask = 15,
                    .io_semantics.location = VARYING_SLOT_POS,
                    .io_semantics.num_slots = 1);
   nir_store_output(&b, var, nir_imm_int(&b, 0), .base = 1, .write_mask = 15,
                    .io_semantics.location = VARYING_SLOT_VAR0,
                    .io_semantics.num_slots = 1);
   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.io_lowered = true;

   NIR_PASS(_, b.shader, poly_nir_lower_vs_before_gs);
   b.shader->info.stage = MESA_SHADER_COMPUTE;
   memset(&b.shader->info.cs, 0, sizeof(b.shader->info.cs));
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass, lower_vertex_param_buffer,
            nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part) &&
                         part.binary != NULL;
   if (getenv("AGX3_DUMP_BINARY")) {
      fprintf(stderr, "G17P compute-form VS: compiled=%u size=%u bytes:",
              compiled, part.info.binary_size);
      for (unsigned i = 0; compiled && i < part.info.binary_size; ++i)
         fprintf(stderr, " %02x", ((const uint8_t *)part.binary)[i]);
      fputc('\n', stderr);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return compiled;
}

/* The driver's transform feedback pipeline: the vertex program becomes a
 * passthrough geometry shader (nir_create_passthrough_gs), which poly lowers
 * into a count program, a pre-GS program, the main geometry program and the
 * rasterization copy program. The first three run as compute kernels on
 * G17P; every KHR-GL30.transform_feedback case reported "cannot encode this
 * GS shader". Reproduce the compute kernels here with the driver's sysval
 * mapping (root uniforms, done as preamble loads) so the refused instruction
 * shapes are visible without hardware. */
static bool
lower_poly_sysvals(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   unsigned base;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_vertex_param_buffer_poly:
      base = 24;
      break;
   case nir_intrinsic_load_geometry_param_buffer_poly:
      base = 28;
      break;
   case nir_intrinsic_load_vs_outputs_poly:
      base = 32;
      break;
   case nir_intrinsic_load_stat_query_address_poly:
      base = 36 + 4 * nir_intrinsic_base(intr);
      break;
   default:
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_load_preamble(b, 1, 64, .base = base));
   return true;
}

static unsigned
xfb_test_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static bool
compile_xfb_compute_program(nir_shader *nir, const char *what)
{
   if (!nir)
      return true;
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_poly_sysvals,
            nir_metadata_control_flow, NULL);
   agx_preprocess_nir(nir);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(nir, &key, &part) &&
                         part.binary != NULL;
   if (getenv("AGX3_DUMP_BINARY") || !compiled)
      fprintf(stderr, "G17P xfb %s program: compiled=%u size=%u\n", what,
              compiled, part.info.binary_size);
   free(part.binary);
   return compiled;
}

static bool
test_g17p_xfb_geometry_programs(void)
{
   /* The passthrough builder and nir_lower_io consult GLSL types. */
   glsl_type_singleton_init_or_ref();
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "xfb source VS");
   nir_variable *pos = nir_variable_create(b.shader, nir_var_shader_out,
                                           glsl_vec4_type(), "gl_Position");
   pos->data.location = VARYING_SLOT_POS;
   nir_variable *var = nir_variable_create(b.shader, nir_var_shader_out,
                                           glsl_vec4_type(), "xfb_var");
   var->data.location = VARYING_SLOT_VAR0;
   var->data.driver_location = 1;
   nir_def *id = nir_u2f32(&b, nir_load_vertex_id(&b));
   nir_store_var(&b, pos,
                 nir_vec4(&b, id, nir_imm_float(&b, 0.0f),
                          nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f)),
                 0xf);
   nir_store_var(&b, var,
                 nir_vec4(&b, nir_fmul_imm(&b, id, 0.5f),
                          nir_imm_float(&b, 0.25f), nir_imm_float(&b, 0.5f),
                          nir_imm_float(&b, 0.75f)),
                 0xf);
   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);

   nir_xfb_info *xfb = rzalloc_size(b.shader, nir_xfb_info_size(1));
   xfb->buffers_written = 1;
   xfb->streams_written = 1;
   xfb->buffers[0].stride = 16;
   xfb->buffers[0].varying_count = 1;
   xfb->output_count = 1;
   xfb->outputs[0].buffer = 0;
   xfb->outputs[0].offset = 0;
   xfb->outputs[0].location = VARYING_SLOT_VAR0;
   xfb->outputs[0].component_mask = 0xf;
   b.shader->xfb_info = xfb;
   b.shader->info.xfb_stride[0] = 4;

   nir_shader *gs = nir_create_passthrough_gs(
      &agx_nir_options, b.shader, MESA_PRIM_TRIANGLES, MESA_PRIM_TRIANGLES,
      false, false, false);
   NIR_PASS(_, gs, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            xfb_test_type_size, nir_lower_io_lower_64bit_to_32);
   nir_shader_gather_info(gs, nir_shader_get_entrypoint(gs));

   nir_shader *gs_count = NULL, *gs_copy = NULL, *pre_gs = NULL;
   struct poly_gs_info info = {0};
   NIR_PASS(_, gs, poly_nir_lower_gs, &gs_count, &gs_copy, &pre_gs, &info);
   if (getenv("AGX3_DUMP_BINARY"))
      fprintf(stderr,
              "G17P xfb lowering: xfb=%u prefix_sum=%u count_words=%u "
              "count=%p pre=%p copy=%p\n",
              info.xfb, info.prefix_sum, info.count_words, (void *)gs_count,
              (void *)pre_gs, (void *)gs_copy);

   bool ok = compile_xfb_compute_program(gs_count, "count");
   ok &= compile_xfb_compute_program(pre_gs, "pre");
   ok &= compile_xfb_compute_program(gs, "main");
   ralloc_free(b.shader);
   ralloc_free(gs);
   ralloc_free(gs_count);
   ralloc_free(pre_gs);
   ralloc_free(gs_copy);
   glsl_type_singleton_decref();
   return ok;
}

/* KHR-GL30.ext_texture_shadow_lod.texturelod.samplercubearrayshadow_fragment
 * as the CTS writes it: `textureLod(u_sampler, v_texCoord, v_lodShadowRef.y,
 * v_lodShadowRef.x)` with the coordinate, the reference and the level all
 * interpolated, and the sampler at API slot 1. */
static bool
test_g17p_cube_array_shadow_level_varyings(bool bias)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      bias ? "G17P cube-array shadow bias varyings"
           : "G17P cube-array shadow level varyings");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *coord = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *lod_ref = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 1, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR1,
      .io_semantics.num_slots = 1);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = bias ? nir_texop_txb : nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
   tex->is_array = true;
   tex->is_shadow = true;
   tex->is_new_style_shadow = true;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 4;
   tex->texture_index = 0;
   tex->sampler_index = 1;
   if (!getenv("AGX3_CUBE_ARRAY_CLAMP"))
      tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                     nir_channel(&b, lod_ref, 1));
   tex->src[2] = nir_tex_src_for_ssa(bias ? nir_tex_src_bias : nir_tex_src_lod,
                                     nir_channel(&b, lod_ref, 0));
   nir_def_init(&tex->instr, &tex->def, 1, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   fs_store_colour(&b, nir_vec4(&b, &tex->def, nir_imm_float(&b, 0.0f),
                                nir_imm_float(&b, 0.0f),
                                nir_imm_float(&b, 1.0f)));
   b.shader->info.inputs_read =
      BITFIELD64_BIT(VARYING_SLOT_VAR0) | BITFIELD64_BIT(VARYING_SLOT_VAR1);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_early, false);
   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part) &&
                         part.binary != NULL;
   if (getenv("AGX3_DUMP_BINARY") || !compiled) {
      fprintf(stderr, "G17P cube-array shadow %s varyings: compiled=%u size=%u bytes:",
              bias ? "bias" : "level", compiled, part.info.binary_size);
      for (unsigned i = 0; compiled && i < part.info.binary_size; ++i)
         fprintf(stderr, " %02x", ((const uint8_t *)part.binary)[i]);
      fputc('\n', stderr);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return compiled;
}

/*
 * KHR-GL30.ext_texture_shadow_lod's sampler2DArrayShadow fragments:
 * textureLod() and texture(..., bias) with the level and the bias arriving
 * in registers. The comparison stages its level or bias beside the layer it
 * already publishes; admitting only the cube form refuses these and the
 * programs fail to link, with no offline gate the wiser.
 */
static bool
test_g17p_explicit_texture_transactions(bool red_only)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "explicit native texture transactions");
   nir_def *bary = nir_load_barycentric_pixel(&b, 32,
                                             .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *input = nir_load_interpolated_input(
      &b, 3, 32, bary, nir_imm_int(&b, 0), .base = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *lod = nir_channel(&b, input, 2);
   nir_def *sum = nir_imm_vec4(&b, 0, 0, 0, 0);
   for (unsigned i = 0; i < 10; ++i) {
      nir_def *uv = nir_vec2(&b, nir_fadd_imm(&b, nir_channel(&b, input, 0),
                                           i * 0.03125f),
                              nir_channel(&b, input, 1));
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
      tex->op = nir_texop_txl;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->dest_type = nir_type_float32;
      tex->coord_components = 2;
      tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, uv);
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      nir_def *value = &tex->def;
      if (red_only) {
         nir_def *red = nir_channel(&b, value, 0);
         value = nir_vec4(&b, red, red, red, red);
      }
      sum = nir_fadd(&b, sum, value);
   }
   fs_store_colour(&b, sum);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   NIR_PASS(_, b.shader, agx_nir_lower_texture_early, false);
   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(b.shader, &key, &part) && part.binary;
   if (getenv("AGX3_DUMP_BINARY")) {
      fprintf(stderr, "explicit-staging-nir-%s: compiled=%u bytes=%u gprs=%u "
                      "staging=%u scratch=%u\n",
              red_only ? "red" : "rgba", ok, part.info.binary_size,
              part.info.nr_gprs, part.info.agx3_staging_halfs,
              part.info.scratch_size);
      for (unsigned i = 0; part.binary && i < part.info.binary_size; ++i)
         fprintf(stderr, "%02x", ((uint8_t *)part.binary)[i]);
      fputc('\n', stderr);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static bool
test_g17p_array_shadow_register_lod(bool bias)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      bias ? "G17P array shadow bias" : "G17P array shadow level");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *coord = nir_load_interpolated_input(
      &b, 3, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *lod_ref = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 1, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR1,
      .io_semantics.num_slots = 1);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = bias ? nir_texop_txb : nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_array = true;
   tex->is_shadow = true;
   tex->is_new_style_shadow = true;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 3;
   tex->texture_index = 0;
   tex->sampler_index = 1;
   tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                     nir_channel(&b, lod_ref, 1));
   tex->src[2] = nir_tex_src_for_ssa(bias ? nir_tex_src_bias : nir_tex_src_lod,
                                     nir_channel(&b, lod_ref, 0));
   nir_def_init(&tex->instr, &tex->def, 1, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   fs_store_colour(&b, nir_vec4(&b, &tex->def, nir_imm_float(&b, 0.0f),
                                nir_imm_float(&b, 0.0f),
                                nir_imm_float(&b, 1.0f)));
   b.shader->info.inputs_read =
      BITFIELD64_BIT(VARYING_SLOT_VAR0) | BITFIELD64_BIT(VARYING_SLOT_VAR1);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   NIR_PASS(_, b.shader, agx_nir_lower_texture_early, false);
   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(b.shader, &key, &part) &&
                   part.binary != NULL && part.info.binary_size > 0;
   if (!ok) {
      fprintf(stderr, "G17P array shadow %s did not compile\n",
              bias ? "bias" : "level");
   }
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/*
 * A 2D-array comparison with a constant level or bias: SPIR-V's
 * OpImageSampleDrefExplicitLod with a constant Lod or Bias on an arrayed
 * image (vkd3d's SampleCmpLevel), the same shape as GLSL's textureLod() and
 * biased texture() on sampler1DArrayShadow. G17P must admit the constant in
 * the combined-sampler pass and keep it in a register (the immediate shares
 * SMP's operand with the staged layer), or the whole shader declines and the
 * pipeline fails. Without g17p the pass keeps declining it.
 */
static bool
test_g17p_array_shadow_constant_lod(bool bias)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      bias ? "G17P array shadow constant bias" : "G17P array shadow constant level");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *coord = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *set = nir_load_descriptor_set_agx(&b, 0);
   nir_def *image =
      nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 0);
   nir_def *heap_index = nir_load_global_constant(
      &b, 1, 16, nir_iadd_imm(&b, set, 40), .align_mul = 2,
      .access = ACCESS_CAN_SPECULATE);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_imm_int(&b, 32), heap_index, .desc_set = 0);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 5);
   tex->op = bias ? nir_texop_txb : nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_array = true;
   tex->is_shadow = true;
   tex->is_new_style_shadow = true;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 3;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                     nir_trim_vector(&b, coord, 3));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                     nir_channel(&b, coord, 3));
   tex->src[2] = nir_tex_src_for_ssa(bias ? nir_tex_src_bias : nir_tex_src_lod,
                                     nir_imm_float(&b, bias ? 0.5f : 2.0f));
   tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   tex->src[4] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   nir_def_init(&tex->instr, &tex->def, 1, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   fs_store_colour(&b, nir_replicate(&b, &tex->def, 4));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   struct agx_g17p_combined_sampler_binding binding;
   const bool declined_elsewhere =
      !agx_nir_g17p_combined_sampler_candidate(b.shader, false);
   const bool admitted =
      agx_nir_lower_g17p_combined_sampler(b.shader, &binding, true);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   struct agx_shader_key key = {0};
   key.dev.chip = AGX_CHIP_G17P;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   key.reserved_preamble = 24;
   struct agx_shader_part part = {0};
   const bool compiled = admitted &&
                         agx_compile_shader_nir(b.shader, &key, &part) &&
                         part.binary != NULL && part.info.binary_size > 0;
   if (!compiled || !declined_elsewhere) {
      fprintf(stderr,
              "G17P array shadow constant %s: declined_elsewhere=%u "
              "admitted=%u compiled=%u\n",
              bias ? "bias" : "level", declined_elsewhere, admitted, compiled);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return compiled && declined_elsewhere;
}

static bool
test_g17p_cube_array_shadow_dump(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P cube-array shadow dump");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *coord = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *ref = nir_load_interpolated_input(
      &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 1, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR1,
      .io_semantics.num_slots = 1);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
   tex->is_array = true;
   tex->is_shadow = true;
   tex->is_new_style_shadow = true;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 4;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   /* The staging sequence under test is independent of the robustness
    * prologue, which needs a bindless descriptor pointer. */
   tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator, ref);
   nir_def_init(&tex->instr, &tex->def, 1, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   fs_store_colour(&b, nir_replicate(&b, &tex->def, 4));
   b.shader->info.inputs_read =
      BITFIELD64_BIT(VARYING_SLOT_VAR0) | BITFIELD64_BIT(VARYING_SLOT_VAR1);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_early, false);
   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part) &&
                         part.binary != NULL;
   if (getenv("AGX3_DUMP_BINARY")) {
      fprintf(stderr, "G17P cube-array shadow: compiled=%u size=%u staging=%u/%u bytes:",
              compiled, part.info.binary_size, part.info.agx3_staging_known,
              part.info.agx3_staging_halfs);
      for (unsigned i = 0; compiled && i < part.info.binary_size; ++i)
         fprintf(stderr, " %02x", ((const uint8_t *)part.binary)[i]);
      fputc('\n', stderr);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return compiled;
}

static bool
test_g17p_native_cube_array_sample(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P native cube-array sample");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *coord = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *set = nir_load_descriptor_set_agx(&b, 0);
   nir_def *image =
      nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 0);
   nir_def *heap_index = nir_load_global_constant(
      &b, 1, 16, nir_iadd_imm(&b, set, 40), .align_mul = 2,
      .access = ACCESS_CAN_SPECULATE);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_imm_int(&b, 32), heap_index, .desc_set = 0);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
   tex->is_array = true;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 4;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   fs_store_colour(&b, &tex->def);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   int ci = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
   struct agx_g17p_combined_sampler_binding binding;
   bool shape = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE && tex->is_array &&
                ci >= 0 && nir_src_num_components(tex->src[ci].src) == 4;
   bool ok = shape &&
             agx_nir_lower_g17p_combined_sampler(b.shader, &binding, true) &&
             binding.texture_offset_B == 0 &&
             binding.sampler_index_offset_B == 40;

   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   key.reserved_preamble = 24;
   struct agx_shader_part part = {0};
   bool compiled = ok && agx_compile_shader_nir(b.shader, &key, &part) &&
                   part.binary != NULL;
   ok &= compiled;

   bool native_smp = false;
   /* The combined HK ABI stages the post-RA layer with CEM's face. Its
    * cube/array selectors match Apple's GPR-layer cube-array witness
    * (f77=2, no direct layer), while f83/f112 retain the driver's independently
    * qualified 0/4 profile. The scoreboard allocator may reuse any retired
    * slot; require a matching signalling/wait pair instead of freezing slot 2.
    */
   for (unsigned slot = 1; slot <= 6; ++slot) {
      const struct agx3_tex_sample_desc sample = {
         .width = AGX3_TEX_W32, .mask = 15, .lod = AGX3_TEX_LOD_AUTO,
         .slot = slot, .wait_mask = 1u << (slot - 1),
         .coord_kill = true, .final = true, .array = true, .cube = true,
         .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
      };
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], length;
      if (!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &sample, bytes, &length)) {
         ok = false;
         break;
      }
      for (unsigned off = 0;
           part.binary && off + length <= part.info.binary_size; ++off)
         native_smp |= !memcmp((uint8_t *)part.binary + off, bytes, length);
   }
   ok &= native_smp && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8;
   if (!ok || getenv("AGX3_DUMP_BINARY")) {
      fprintf(stderr,
              "cube-array: shape=%u compiled=%u smp=%u staging=%u/%u size=%u bytes:",
              shape, compiled, native_smp, part.info.agx3_staging_known,
              part.info.agx3_staging_halfs, part.info.binary_size);
      for (unsigned i = 0; compiled && i < part.info.binary_size; ++i)
         fprintf(stderr, " %02x", ((const uint8_t *)part.binary)[i]);
      fputc('\n', stderr);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static nir_tex_instr *
custom_border_sample(nir_builder *b, nir_def *image, nir_def *sampler,
                     nir_def *coord, unsigned flags)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->backend_flags = flags;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return tex;
}

static bool
g17p_register_sample_group_is_global(const struct agx_shader_part *part,
                                     unsigned expected)
{
   const uint8_t *code = part->binary;
   unsigned samples = 0, unterminated = 0;

   for (unsigned off = 0; off + 20 <= part->info.binary_size; ++off) {
      const uint8_t *p = code + off;

      /* SMP.EP4 with register texture and sampler operands. The descriptor
       * fields vary with RA, while these selector bytes are fixed by Apple's
       * argument-buffer witnesses. f156 is bit 7 of byte 3 in this form. */
      if (p[2] != 0x0c || p[9] != 0x02 || p[10] != 0x90 ||
          p[11] != 0x01 || (p[12] != 0x01 && p[12] != 0x12) ||
          p[13] != 0x00 ||
          !(p[15] & 0x20))
         continue;

      samples++;
      unterminated += !(p[3] & 0x80);
      off += 19;
   }

   const unsigned expected_unterminated = expected > 1 ? expected - 1 : 0;
   if (samples != expected || unterminated != expected_unterminated) {
      fprintf(stderr,
              "indexed SMP.EP4 count/unterminated %u/%u, expected %u/%u\n",
              samples, unterminated, expected, expected_unterminated);
      return false;
   }
   return true;
}

static bool
test_custom_border_samplers(bool dynamic, unsigned invalid)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "custom border indexed samples");
   nir_def *set = nir_load_descriptor_set_agx(&b, 24);
   nir_def *ordinary_index = dynamic ? nir_load_global_constant(
      &b, 1, 16, nir_iadd_imm(&b, set, 40), .align_mul = 2)
      : nir_imm_intN_t(&b, 80, 16);
   nir_def *clamp_index = dynamic ? nir_load_global_constant(
      &b, 1, 16, nir_iadd_imm(&b, set, 42), .align_mul = 2)
      : nir_imm_intN_t(&b, 84, 16);
   nir_def *image = nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 24);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_imm_int(&b, 32), ordinary_index, .desc_set = 24);
   nir_def *coord = nir_imm_vec2(&b, -.125, .75);
   nir_tex_instr *ordinary = custom_border_sample(&b, image, sampler, coord, 0);
   nir_def *condition = nir_ige_imm(&b, clamp_index, 0);
   nir_if *branch = nir_push_if(&b, condition);
   nir_tex_instr *clamp = custom_border_sample(
      &b, image, clamp_index, coord,
      invalid == 2 ? 0 :
      invalid ? AGX_TEXTURE_FLAG_NO_CLAMP : AGX_TEXTURE_FLAG_CLAMP_TO_0);
   nir_def *border = nir_load_global_constant(
      &b, 4, 32, nir_iadd_imm(&b, set, 52), .align_mul = 4);
   nir_def *mixed = nir_flrp(&b, &clamp->def, &ordinary->def, border);
   nir_pop_if(&b, branch);
   nir_def *result = nir_if_phi(&b, mixed, &ordinary->def);
   for (unsigned c = 0; c < 4; ++c)
      nir_export_agx(&b, nir_channel(&b, result, c),
                     .base = AGX_ABI_FOUT_COLOUR(0) + 2 * c);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   bool lowered = agx_nir_lower_g17p_indexed_samplers(b.shader, true);
   bool ok = lowered == (invalid == 0);
   if (invalid) {
      /* An unsupported third shape must not partially convert the first. */
      ok &= nir_tex_instr_src_index(ordinary, nir_tex_src_texture_handle) >= 0;
      ok &= nir_tex_instr_src_index(clamp, nir_tex_src_texture_handle) >= 0;
   } else {
      ok &= branch->condition.ssa == condition;
      ok &= nir_def_instr(mixed)->block == clamp->instr.block;
      ok &= clamp->instr.block != ordinary->instr.block;
      nir_alu_instr *lerp = nir_instr_as_alu(nir_def_instr(mixed));
      ok &= lerp->op == nir_op_flrp && lerp->src[0].src.ssa == &clamp->def &&
            lerp->src[1].src.ssa == &ordinary->def &&
            lerp->src[2].src.ssa == border;
      nir_tex_instr *samples[] = {ordinary, clamp};
      nir_def *original[] = {ordinary_index, clamp_index};
      for (unsigned i = 0; i < 2; ++i) {
         nir_tex_instr *tex = samples[i];
         int si = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
         int ti = nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
         ok &= si >= 0 && ti >= 0;
         if (si < 0 || ti < 0)
            continue;
         ok &= nir_src_bit_size(tex->src[si].src) == 64 &&
               nir_src_bit_size(tex->src[ti].src) == 64;
         nir_alu_instr *wide = nir_def_as_alu_or_null(tex->src[si].src.ssa);
         /* G17P descriptors carry native heap indices without AGX2's bias. */
         ok &= wide && wide->op == nir_op_u2u64 &&
               wide->src[0].src.ssa == original[i];
      }
      if (!dynamic) {
         NIR_PASS(_, b.shader, nir_opt_constant_folding);
         for (unsigned i = 0; i < 2; ++i) {
            int si = nir_tex_instr_src_index(samples[i], nir_tex_src_sampler_handle);
            ok &= nir_src_is_const(samples[i]->src[si].src) &&
                  nir_src_as_uint(samples[i]->src[si].src) == (i ? 84 : 80);
         }
      }

      if (dynamic) {
         /* This is the indexed two-sample/control-flow shape used by desktop
          * compositors. Keep the check through RA and binary scheduling: both
          * descriptor loads must retire before SMP consumes their register
          * indices, preserving both native sampler-heap indices. */
         struct agx_shader_key key = {
            .dev.encoding = AGX_ISA_ENCODING_AGX3B,
            .no_preamble = true,
            .reserved_preamble = 128,
         };
         fs_lower(b.shader);
         NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
                  lower_test_descriptor_set, nir_metadata_control_flow, NULL);
         agx_preprocess_nir_g17p(b.shader);
         struct agx_shader_part part = {0};
         const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
         ok &= compiled && part.binary != NULL &&
               part.info.g17p_sampled_fragment &&
               g17p_register_sample_group_is_global(&part, 2);
         free(part.binary);
      }
   }
   ralloc_free(b.shader);
   return ok;
}

static bool
test_indexed_sampler_native_indices(void)
{
   static const uint16_t indices[] = {0, 1, 27, 28, 29, 43, 44, UINT16_MAX};

   for (unsigned i = 0; i < ARRAY_SIZE(indices); ++i) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options,
         "G17P indexed sampler native index");
      nir_def *image =
         nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 0);
      nir_def *sampler = nir_bindless_sampler_agx(
         &b, nir_imm_int(&b, 32), nir_imm_intN_t(&b, indices[i], 16),
         .desc_set = 0);
      nir_tex_instr *tex = custom_border_sample(
         &b, image, sampler, nir_imm_vec2(&b, 0.25f, 0.75f), 0);
      nir_export_agx(&b, nir_channel(&b, &tex->def, 0),
                     .base = AGX_ABI_FOUT_COLOUR(0));

      bool ok = agx_nir_lower_g17p_indexed_samplers(b.shader, true);
      const int si = nir_tex_instr_src_index(
         tex, nir_tex_src_sampler_handle);
      const int ti = nir_tex_instr_src_index(
         tex, nir_tex_src_texture_offset);
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      const uint32_t expected = indices[i];
      ok &= si >= 0 && ti >= 0 &&
            nir_src_is_const(tex->src[si].src) &&
            nir_src_as_uint(tex->src[si].src) == expected &&
            nir_src_bit_size(tex->src[si].src) == 64 &&
            nir_src_bit_size(tex->src[ti].src) == 64;
      ralloc_free(b.shader);
      if (!ok)
         return false;
   }

   return true;
}

static nir_tex_instr *
indexed_sample(nir_builder *b, nir_def *set, uint32_t descriptor_offset_B,
               enum glsl_sampler_dim dim, bool array, nir_def *coord)
{
   nir_def *image = nir_bindless_image_agx(
      b, nir_imm_int(b, descriptor_offset_B), .desc_set = 0);
   nir_def *sampler_index = nir_load_global_constant(
      b, 1, 16,
      nir_iadd_imm(b, set, descriptor_offset_B + 40), .align_mul = 2);
   nir_def *sampler = nir_bindless_sampler_agx(
      b, nir_imm_int(b, descriptor_offset_B + 32), sampler_index,
      .desc_set = 0);
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = dim;
   tex->is_array = array;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 3;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return tex;
}

static bool
test_indexed_array_and_3d_compile(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P indexed compositor array and 3D samples");
   nir_def *set = nir_load_descriptor_set_agx(&b, 0);
   nir_tex_instr *array = indexed_sample(
      &b, set, 0, GLSL_SAMPLER_DIM_2D, true,
      nir_vec3(&b, nir_imm_float(&b, 0.25f), nir_imm_float(&b, 0.75f),
               nir_imm_float(&b, 1.0f)));
   nir_tex_instr *three_d = indexed_sample(
      &b, set, 72, GLSL_SAMPLER_DIM_3D, false,
      nir_imm_vec3(&b, 0.125f, 0.5f, 0.875f));
   nir_def *result = nir_fadd(&b, &array->def, &three_d->def);
   for (unsigned c = 0; c < 4; ++c)
      nir_export_agx(&b, nir_channel(&b, result, c),
                     .base = AGX_ABI_FOUT_COLOUR(0) + 2 * c);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   bool ok = agx_nir_lower_g17p_indexed_samplers(b.shader, true);
   nir_tex_instr *samples[] = {array, three_d};
   for (unsigned i = 0; i < ARRAY_SIZE(samples); ++i) {
      const int ti = nir_tex_instr_src_index(
         samples[i], nir_tex_src_texture_offset);
      const int si = nir_tex_instr_src_index(
         samples[i], nir_tex_src_sampler_handle);
      ok &= ti >= 0 && si >= 0 &&
            nir_src_bit_size(samples[i]->src[ti].src) == 64 &&
            nir_src_bit_size(samples[i]->src[si].src) == 64;
   }

   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
            lower_test_g17p_descriptor_set, nir_metadata_control_flow, NULL);
   agx_preprocess_nir_g17p(b.shader);
   fs_lower(b.shader);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
      .no_stop = true,
      .reserved_preamble = AGX_ABI_FUNI_ROOT + 8,
   };
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &key, &part) && part.binary != NULL &&
         part.info.g17p_sampled_fragment && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 &&
         g17p_register_sample_group_is_global(&part, 2);
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static bool
test_indexed_fetch_sets(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "mixed indexed sample and array fetch sets");
   nir_def *root = nir_load_root_agx(&b);
   nir_def *element = nir_load_global_constant(&b, 1, 32, root, .align_mul=4);
   nir_def *offset = nir_imul_imm(&b, element, 72);
   nir_def *result = nir_imm_vec4(&b, 0, 0, 0, 0);
   const unsigned sets[] = {0, 3, 7};
   for (unsigned i = 0; i < ARRAY_SIZE(sets); ++i) {
      nir_def *address = nir_load_descriptor_set_agx(&b, sets[i]);
      address = nir_iadd(&b, address, nir_u2u64(&b, offset));
      nir_def *sampler_index = nir_load_global_constant(
         &b, 1, 16, nir_iadd_imm(&b, address, 40), .align_mul=2);
      nir_def *image = nir_bindless_image_agx(&b, offset, .desc_set=sets[i]);
      nir_def *sampler = nir_bindless_sampler_agx(
         &b, nir_iadd_imm(&b, offset, 32), sampler_index, .desc_set=sets[i]);
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
      tex->op=i ? nir_texop_txf : nir_texop_tex;
      tex->sampler_dim=GLSL_SAMPLER_DIM_2D;
      tex->dest_type=nir_type_float32;
      tex->is_array=i==2;
      tex->coord_components=i==1?1:(tex->is_array?3:2);
      tex->src[0]=nir_tex_src_for_ssa(i==1?nir_tex_src_backend1:nir_tex_src_coord,
         i ? (i==2 ? nir_imm_ivec3(&b,3,4,1) : nir_imm_ivec2(&b,3,4))
           : nir_imm_vec2(&b,.25,.75));
      tex->src[1]=nir_tex_src_for_ssa(nir_tex_src_texture_handle,image);
      tex->src[2]=nir_tex_src_for_ssa(i?nir_tex_src_lod:nir_tex_src_sampler_handle,
                                    i?nir_imm_intN_t(&b,i==1?2:63,16):sampler);
      nir_def_init(&tex->instr,&tex->def,4,32);
      nir_builder_instr_insert(&b,&tex->instr);
      result=nir_fadd(&b,result,&tex->def);
   }
   nir_store_output(&b,result,nir_imm_int(&b,0),.write_mask=15,
                    .src_type=nir_type_float32);
   bool ok=agx_nir_lower_g17p_indexed_samplers(b.shader, true);
   unsigned samples=0, fetches=0, arrays=0, descriptors=0, roots=0, image_loads=0, sampler_loads=0, set_mask=0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr,block) {
         if (instr->type==nir_instr_type_tex) {
            nir_tex_instr *tex=nir_instr_as_tex(instr);
            int ti=nir_tex_instr_src_index(tex,nir_tex_src_texture_offset);
            int si=nir_tex_instr_src_index(tex,nir_tex_src_sampler_handle);
            ok &= ti>=0;
            if(ti>=0)ok &= nir_src_bit_size(tex->src[ti].src)==64;
            if(tex->op==nir_texop_tex) {
               ok &= si>=0;
               if(si>=0)ok &= nir_src_bit_size(tex->src[si].src)==64;
               samples++;
            } else {
               ok &= si<0 && tex->op==nir_texop_txf;
               int li=nir_tex_instr_src_index(tex,nir_tex_src_lod);
               ok &= li>=0 && nir_src_is_const(tex->src[li].src) &&
                     nir_src_bit_size(tex->src[li].src)==32 &&
                     nir_src_as_uint(tex->src[li].src)==
                        (tex->is_array ? 63 : 2);
               fetches++;
               arrays+=tex->is_array;
            }
         } else if (instr->type==nir_instr_type_intrinsic) {
            nir_intrinsic_instr *in=nir_instr_as_intrinsic(instr);
            roots += in->intrinsic==nir_intrinsic_load_root_agx;
            descriptors += in->intrinsic==nir_intrinsic_bindless_image_agx ||
                           in->intrinsic==nir_intrinsic_bindless_sampler_agx;
            if (in->intrinsic==nir_intrinsic_load_descriptor_set_agx)
               set_mask |= 1u << nir_intrinsic_desc_set(in);
            if (in->intrinsic==nir_intrinsic_load_global_constant) {
               image_loads += in->def.bit_size==32;
               sampler_loads += in->def.bit_size==16;
            }
         }
      }
   }
   ok &= samples==1 && fetches==2 && arrays==1 && descriptors==0 && roots==1 && image_loads==4 &&
         sampler_loads==1 && set_mask==((1u<<0)|(1u<<3)|(1u<<7));
   ralloc_free(b.shader);
   return ok;
}

static bool
test_indexed_dynamic_fetch_lod(void)
{
   const mesa_shader_stage stages[] = {
      MESA_SHADER_VERTEX, MESA_SHADER_FRAGMENT, MESA_SHADER_COMPUTE,
   };
   bool all_ok = true;

   for (unsigned stage = 0; stage < ARRAY_SIZE(stages); ++stage) {
      for (unsigned dimension = 0; dimension < 3; ++dimension) {
         const bool array = dimension == 1;
         const bool three_d = dimension == 2;

         for (unsigned bits = 16; bits <= 32; bits += 16) {
            nir_builder b = nir_builder_init_simple_shader(
               stages[stage], &agx_nir_options, "indexed dynamic fetch LOD");
            b.shader->info.workgroup_size[0] = 1;
            b.shader->info.workgroup_size[1] = 1;
            b.shader->info.workgroup_size[2] = 1;
            nir_def *lod = nir_load_global_constant(
               &b, 1, bits, nir_imm_int64(&b, 0x10000), .align_mul = 4);
            nir_def *handle = nir_bindless_image_agx(
               &b, nir_imm_int(&b, 72), .desc_set = 3);
            nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
            tex->op = nir_texop_txf;
            tex->sampler_dim =
               three_d ? GLSL_SAMPLER_DIM_3D : GLSL_SAMPLER_DIM_2D;
            tex->is_array = array;
            tex->coord_components = array || three_d ? 3 : 2;
            tex->dest_type = nir_type_float32;
            tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
            tex->src[0] = nir_tex_src_for_ssa(
               nir_tex_src_backend1,
               array || three_d ? nir_imm_ivec3(&b, 3, 4, 1)
                                : nir_imm_ivec2(&b, 3, 4));
            tex->src[1] =
               nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
            tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
            nir_def_init(&tex->instr, &tex->def, 4, 32);
            nir_builder_instr_insert(&b, &tex->instr);

            if (stages[stage] == MESA_SHADER_VERTEX) {
               for (unsigned c = 0; c < 4; ++c)
                  nir_store_uvs_agx(&b, nir_channel(&b, &tex->def, c),
                                    nir_imm_intN_t(&b, c, 16));
               b.shader->info.outputs_written = VARYING_BIT_POS;
            } else if (stages[stage] == MESA_SHADER_FRAGMENT) {
               fs_store_colour(&b, &tex->def);
            } else {
               nir_store_global(&b, &tex->def, nir_imm_int64(&b, 0x20000),
                                .align_mul = 16, .write_mask = 15);
            }

            bool ok = agx_nir_lower_g17p_indexed_samplers(b.shader, true);
            const int li =
               nir_tex_instr_src_index(tex, nir_tex_src_lod);
            const int ti =
               nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
            ok &= li >= 0 && ti >= 0;
            if (ok) {
               nir_def *lowered_lod = tex->src[li].src.ssa;
               ok &= lowered_lod->bit_size == 32 &&
                     !nir_src_is_const(tex->src[li].src) &&
                     nir_src_bit_size(tex->src[ti].src) == 64;
               if (bits == 32) {
                  ok &= lowered_lod == lod;
               } else {
                  nir_alu_instr *convert = nir_src_as_alu(tex->src[li].src);
                  ok &= convert && convert->op == nir_op_u2u32 &&
                        convert->src[0].src.ssa == lod;
               }
            }

            if (ok) {
               NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
                        lower_test_descriptor_set, nir_metadata_control_flow,
                        NULL);
               agx_preprocess_nir_g17p(b.shader);
               struct agx_shader_key key = {
                  .dev.encoding = AGX_ISA_ENCODING_AGX3B,
                  .no_preamble = true,
               };
               struct agx_shader_part part = {0};
               ok = agx_compile_shader_nir(b.shader, &key, &part) &&
                    part.binary;
               free(part.binary);
            }

            if (!ok)
               fprintf(stderr,
                       "dynamic indexed txf stage=%u array=%u three_d=%u "
                       "bits=%u "
                       "failed\n",
                       stages[stage], array, three_d, bits);
            all_ok &= ok;
            ralloc_free(b.shader);
         }
      }
   }

   return all_ok;
}

/* vkCmdResolveImage lowers each multisample read to txf_ms after packing the
 * sample and array layer into backend coordinate 2. It has no API sampler;
 * native sampler zero is the device-wide texel-fetch sampler. */
static bool
test_indexed_multisample_fetch(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "indexed multisample resolve fetch");
   nir_def *image =
      nir_bindless_image_agx(&b, nir_imm_int(&b, 144), .desc_set = 2);
   nir_def *coord = nir_imm_ivec3(&b, 5, 7, (1u << 16) | 2u);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
   tex->op = nir_texop_txf_ms;
   tex->sampler_dim = GLSL_SAMPLER_DIM_MS;
   tex->is_array = true;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 3;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1, coord);
   tex->src[1] =
      nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_store_output(&b, &tex->def, nir_imm_int(&b, 0), .write_mask = 15,
                    .src_type = nir_type_float32);

   bool ok = agx_nir_lower_g17p_indexed_samplers(b.shader, true);
   const int texture =
      nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
   ok &= texture >= 0 &&
         nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) < 0 &&
         nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) < 0 &&
         nir_tex_instr_src_index(tex, nir_tex_src_ms_index) < 0;
   if (texture >= 0)
      ok &= nir_src_bit_size(tex->src[texture].src) == 64;

   unsigned bindless = 0, heap_indices = 0, set_mask = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         bindless += intr->intrinsic == nir_intrinsic_bindless_image_agx ||
                     intr->intrinsic == nir_intrinsic_bindless_sampler_agx;
         if (intr->intrinsic == nir_intrinsic_load_descriptor_set_agx)
            set_mask |= 1u << nir_intrinsic_desc_set(intr);
         heap_indices += intr->intrinsic ==
                            nir_intrinsic_load_global_constant &&
                         intr->def.num_components == 1 &&
                         intr->def.bit_size == 32;
      }
   }

   ok &= bindless == 0 && heap_indices == 1 && set_mask == (1u << 2);
   ralloc_free(b.shader);
   return ok;
}

static nir_shader *
build_fs_texture(void)
{
   return build_fs_texture_bias_offset(108);
}

static nir_shader *
build_fs_texture_typed(nir_alu_type dest_type)
{
   nir_shader *shader = build_fs_texture();
   unsigned samples = 0;

   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;

            nir_instr_as_tex(instr)->dest_type = dest_type;
            samples++;
         }
      }
   }

   assert(samples == 1);
   return shader;
}

static nir_shader *
build_fs_texture_uint(void)
{
   return build_fs_texture_typed(nir_type_uint32);
}

static nir_shader *
build_fs_texture_int(void)
{
   return build_fs_texture_typed(nir_type_int32);
}

static bool
test_combined_sampler_integer_scope(void)
{
   nir_shader *shader = build_fs_texture_typed(nir_type_uint16);
   struct agx_g17p_combined_sampler_binding binding = {
      .texture_offset_B = UINT32_MAX,
      .sampler_index_offset_B = UINT32_MAX,
   };

   /* Integer sampled results are a native 32-bit register shape. Keep a
    * narrower result fail-closed so widening this admission cannot silently
    * discard an unwitnessed result-width encoding. */
   const bool ok = !agx_nir_lower_g17p_combined_sampler(shader, &binding, true) &&
                   binding.texture_offset_B == UINT32_MAX &&
                   binding.sampler_index_offset_B == UINT32_MAX;
   ralloc_free(shader);
   return ok;
}

static bool
test_combined_sampler_scope(void)
{
   nir_shader *nir = build_fs_texture_bias_offset(112);
   struct agx_g17p_combined_sampler_binding binding = {
      .texture_offset_B = UINT32_MAX,
      .sampler_index_offset_B = UINT32_MAX,
   };

   const bool lowered =
      agx_nir_lower_g17p_combined_sampler(nir, &binding, true);
   const bool ok = !lowered && binding.texture_offset_B == UINT32_MAX &&
                   binding.sampler_index_offset_B == UINT32_MAX;
   ralloc_free(nir);
   return ok;
}

/* The same shader with a real API LOD bias instead of the driver's software
 * one. `texture(tex, uv, k)` is a txb whose bias is the shader's own value,
 * not a descriptor field, so the lowering must keep both the source and the
 * txb: the hardware sampler applies its own bias and the instruction applies
 * this one. */
static nir_shader *
build_fs_texture_api_bias(bool gradient, bool cube)
{
   assert(!(gradient && cube));

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 biased FS");

   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *uv = nir_load_interpolated_input(
      &b, cube ? 3 : 2, 32, bary, nir_imm_int(&b, 0), .base = 0,
      .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);

   nir_def *timage = nir_bindless_image_agx(&b, nir_imm_int(&b, 64),
                                            .desc_set = 0);
   nir_def *tsampler = nir_bindless_sampler_agx(
      &b, nir_imm_int(&b, 96), nir_imm_intN_t(&b, 45, 16), .desc_set = 0);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, gradient ? 5 : 4);
   tex->op = gradient ? nir_texop_txd : nir_texop_txb;
   tex->sampler_dim = cube ? GLSL_SAMPLER_DIM_CUBE : GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = cube ? 3 : 2;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, uv);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, timage);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, tsampler);
   if (gradient) {
      /* Any live 32-bit vec2 pair stands in for the screen-space
       * derivatives; this test is about the lowering keeping them. */
      tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_ddx,
                                        nir_fmul_imm(&b, uv, 0.5));
      tex->src[4] = nir_tex_src_for_ssa(nir_tex_src_ddy,
                                        nir_fmul_imm(&b, uv, 0.25));
   } else {
      tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_bias,
                                        nir_imm_floatN_t(&b, 0.5, 16));
   }
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   fs_store_colour(&b, &tex->def);

   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   return b.shader;
}

/* The lowering keeps the API bias and the explicit gradients, binds the
 * combined sampler, and leaves no descriptor crawl behind. */
static bool
test_combined_sampler_biased_and_gradient(bool gradient, bool cube)
{
   nir_shader *shader = build_fs_texture_api_bias(gradient, cube);
   struct agx_g17p_combined_sampler_binding binding = {
      .texture_offset_B = UINT32_MAX,
      .sampler_index_offset_B = UINT32_MAX,
   };
   bool ok = agx_nir_lower_g17p_combined_sampler(shader, &binding, true);
   ok &= binding.texture_offset_B == 64 &&
         binding.sampler_index_offset_B == 104;

   unsigned samples = 0, descriptors = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               const int bias =
                  nir_tex_instr_src_index(tex, nir_tex_src_bias);
               const int ddx = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
               const int ddy = nir_tex_instr_src_index(tex, nir_tex_src_ddy);
               samples += tex->texture_index == 0 && tex->sampler_index == 0 &&
                          tex->sampler_dim ==
                             (cube ? GLSL_SAMPLER_DIM_CUBE
                                   : GLSL_SAMPLER_DIM_2D) &&
                          (gradient ? (tex->op == nir_texop_txd && ddx >= 0 &&
                                       ddy >= 0)
                                    : (tex->op == nir_texop_txb && bias >= 0));
            } else if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
               descriptors += op == nir_intrinsic_load_descriptor_set_agx ||
                              op == nir_intrinsic_bindless_image_agx ||
                              op == nir_intrinsic_bindless_sampler_agx;
            }
         }
      }
   }

   ok &= samples == 1 && descriptors == 0;
   ralloc_free(shader);
   return ok;
}

static nir_shader *
build_cs_bindless_txf(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 static bindless TXF CS");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *coord = nir_load_global_invocation_id(&b, 32);
   nir_def *handle = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 64), .desc_set = 0);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_array = true;
   tex->dest_type = nir_type_uint32;
   tex->coord_components = 3;
   tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(&b, 0));
   tex->src[2] =
      nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   /* Keep the fetch live without introducing an unrelated memory operation. */
   nir_def *sink = nir_channel(&b, &tex->def, 0);
   nir_export_agx(&b, sink, .base = 0);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static bool
test_txf_lowering(void)
{
   nir_shader *nir = build_cs_bindless_txf();
   struct agx_g17p_txf_binding binding = {.texture_offset_B = UINT32_MAX};

   const bool lowered = agx_nir_lower_g17p_txf(nir, &binding);
   bool ok = lowered && binding.texture_offset_B == 64;
   if (ok) {
      nir_tex_instr *fetch = NULL;
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_tex)
                  fetch = nir_instr_as_tex(instr);
            }
         }
      }

      ok = fetch && fetch->texture_index == 0 &&
           nir_tex_instr_src_index(fetch, nir_tex_src_texture_handle) < 0;
   }

   ralloc_free(nir);
   return ok;
}

/* texelFetchOffset is core WebGL2/GLES3.  AGX texture_load has no witnessed
 * integer-offset form, while adding the compile-time offset to the integer
 * coordinate is exactly equivalent and is already provided by nir_lower_tex.
 * Keep that lowering before descriptor binding so source 5 never reaches the
 * AGX3 packer as an unencodable texture offset. */
static bool
test_txf_offset_lowering(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 texelFetchOffset FS");

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                      nir_imm_ivec2(&b, 1, 1));
   tex->src[1] =
      nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(&b, 0));
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_offset,
                                      nir_imm_ivec2(&b, 1, -1));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   fs_store_colour(&b, &tex->def);

   bool progress = agx_nir_lower_g17p_texture_early(b.shader, false, false);
   NIR_PASS(_, b.shader, nir_opt_constant_folding);

   const int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const bool ok = progress && coord >= 0 &&
                   nir_tex_instr_src_index(tex, nir_tex_src_offset) < 0 &&
                   nir_src_is_const(tex->src[coord].src) &&
                   nir_src_comp_as_int(tex->src[coord].src, 0) == 2 &&
                   nir_src_comp_as_int(tex->src[coord].src, 1) == 0;
   ralloc_free(b.shader);
   return ok;
}

static bool
test_g17p_3d_runtime_offset_lowering(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 runtime 3D offset FS");
   nir_def *offset = nir_load_push_constant(
      &b, 3, 32, nir_imm_int(&b, 0), .base = 0, .range = 12);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_3D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 3;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_imm_vec3(&b, 0.25f, 0.5f, 0.75f));
   tex->src[1] =
      nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 1.0f));
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_offset, offset);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   fs_store_colour(&b, &tex->def);

   const bool progress =
      agx_nir_lower_g17p_texture_early(b.shader, false, false);
   const int coord =
      nir_tex_instr_src_index(tex, nir_tex_src_coord);
   unsigned size_queries = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex &&
             nir_instr_as_tex(instr)->op == nir_texop_txs)
            size_queries++;
      }
   }

   const bool ok = progress && coord >= 0 &&
                   nir_tex_instr_src_index(tex, nir_tex_src_offset) < 0 &&
                   !nir_src_is_const(tex->src[coord].src) &&
                   size_queries == 1;
   ralloc_free(b.shader);
   return ok;
}

static bool
test_g17p_projected_3d_runtime_offset_lowering(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options,
      "AGX3 projected runtime 3D offset VS");
   nir_def *offset = nir_load_push_constant(
      &b, 3, 32, nir_imm_int(&b, 0), .base = 0, .range = 12);
   nir_def *projector = nir_load_push_constant(
      &b, 1, 32, nir_imm_int(&b, 0), .base = 12, .range = 4);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 5);
   tex->op = nir_texop_txd;
   tex->sampler_dim = GLSL_SAMPLER_DIM_3D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 3;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_vec3(&b, 0.9f, 1.05f, -0.08f));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_projector, projector);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_offset, offset);
   tex->src[3] = nir_tex_src_for_ssa(
      nir_tex_src_ddx, nir_imm_vec3(&b, 0.2f, 0.0f, 0.0f));
   tex->src[4] = nir_tex_src_for_ssa(
      nir_tex_src_ddy, nir_imm_vec3(&b, 0.0f, 0.0f, 0.0f));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   const bool progress =
      agx_nir_lower_g17p_texture_early(b.shader, false, false);
   const int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   unsigned size_queries = 0, reciprocals = 0, fused_adds = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex &&
             nir_instr_as_tex(instr)->op == nir_texop_txs)
            size_queries++;
         else if (instr->type == nir_instr_type_alu) {
            const nir_op op = nir_instr_as_alu(instr)->op;
            reciprocals += op == nir_op_frcp;
            fused_adds += op == nir_op_ffma;
         }
      }
   }

   const bool ok = progress && coord >= 0 &&
                   nir_tex_instr_src_index(tex, nir_tex_src_offset) < 0 &&
                   nir_tex_instr_src_index(tex, nir_tex_src_projector) < 0 &&
                   !nir_src_is_const(tex->src[coord].src) &&
                   size_queries == 1 && reciprocals >= 2 && fused_adds >= 1;
   ralloc_free(b.shader);
   return ok;
}

/* The homogeneous offset composition is specific to the 3D fallback. Keep
 * projected 2D runtime offsets on the native offset path while the common
 * early lowering still consumes their projector. */
static bool
test_g17p_projected_2d_runtime_offset_scope(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "AGX3 projected runtime 2D offset FS");
   nir_def *coord = nir_load_push_constant(
      &b, 2, 32, nir_imm_int(&b, 0), .base = 0, .range = 8);
   nir_def *projector = nir_load_push_constant(
      &b, 1, 32, nir_imm_int(&b, 0), .base = 8, .range = 4);
   nir_def *offset = nir_load_push_constant(
      &b, 2, 32, nir_imm_int(&b, 0), .base = 12, .range = 8);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 5);
   tex->op = nir_texop_txd;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_int32;
   tex->coord_components = 2;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_projector, projector);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_offset, offset);
   tex->src[3] = nir_tex_src_for_ssa(
      nir_tex_src_ddx, nir_imm_vec2(&b, 0.2f, 0.0f));
   tex->src[4] = nir_tex_src_for_ssa(
      nir_tex_src_ddy, nir_imm_vec2(&b, 0.0f, 0.0f));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   const bool progress =
      agx_nir_lower_g17p_texture_early(b.shader, false, false);
   const int lowered_coord =
      nir_tex_instr_src_index(tex, nir_tex_src_coord);
   unsigned size_queries = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex &&
             nir_instr_as_tex(instr)->op == nir_texop_txs)
            size_queries++;
      }
   }

   const bool ok = progress && lowered_coord >= 0 &&
                   nir_tex_instr_src_index(tex, nir_tex_src_projector) < 0 &&
                   nir_tex_instr_src_index(tex, nir_tex_src_offset) >= 0 &&
                   !nir_src_is_const(tex->src[lowered_coord].src) &&
                   size_queries == 0;
   ralloc_free(b.shader);
   return ok;
}

static bool
test_agx3d_texture_early_lowering(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "AGX3D early texture lowering");

   nir_def *coord2 = nir_load_input(&b, 2, 32, nir_imm_int(&b, 0));
   nir_def *coord3 = nir_load_input(&b, 3, 32, nir_imm_int(&b, 1));
   nir_def *lod = nir_load_input(&b, 1, 32, nir_imm_int(&b, 2));
   nir_def *min_lod = nir_load_input(&b, 1, 32, nir_imm_int(&b, 3));
   nir_def *offset = nir_load_constant_agx(
      &b, 2, 32, nir_imm_int64(&b, 0), nir_imm_int(&b, 0),
      .format = PIPE_FORMAT_R32_UINT, .sign_extend = true);

   /* 1. txb with min_lod -> txl without bias or min_lod. */
   nir_tex_instr *txb = nir_tex_instr_create(b.shader, 3);
   txb->op = nir_texop_txb;
   txb->sampler_dim = GLSL_SAMPLER_DIM_2D;
   txb->dest_type = nir_type_float32;
   txb->coord_components = 2;
   txb->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord2);
   txb->src[1] = nir_tex_src_for_ssa(nir_tex_src_bias, lod);
   txb->src[2] = nir_tex_src_for_ssa(nir_tex_src_min_lod, min_lod);
   nir_def_init(&txb->instr, &txb->def, 4, 32);
   nir_builder_instr_insert(&b, &txb->instr);

   /* 2. 3D txd -> txl. */
   nir_tex_instr *txd = nir_tex_instr_create(b.shader, 3);
   txd->op = nir_texop_txd;
   txd->sampler_dim = GLSL_SAMPLER_DIM_3D;
   txd->dest_type = nir_type_float32;
   txd->coord_components = 3;
   txd->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord3);
   txd->src[1] = nir_tex_src_for_ssa(nir_tex_src_ddx, nir_imm_vec3(&b, 0.1f, 0.0f, 0.0f));
   txd->src[2] = nir_tex_src_for_ssa(nir_tex_src_ddy, nir_imm_vec3(&b, 0.0f, 0.1f, 0.0f));
   nir_def_init(&txd->instr, &txd->def, 4, 32);
   nir_builder_instr_insert(&b, &txd->instr);

   /* 3. 2D_ARRAY txl with runtime offset -> offset lowered with txs + query_levels. */
   nir_tex_instr *txl = nir_tex_instr_create(b.shader, 3);
   txl->op = nir_texop_txl;
   txl->sampler_dim = GLSL_SAMPLER_DIM_2D;
   txl->is_array = true;
   txl->dest_type = nir_type_float32;
   txl->coord_components = 3;
   txl->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord3);
   txl->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
   txl->src[2] = nir_tex_src_for_ssa(nir_tex_src_offset, offset);
   nir_def_init(&txl->instr, &txl->def, 4, 32);
   nir_builder_instr_insert(&b, &txl->instr);

   /* 4. tg4 with runtime offset -> offset preserved for native PACK_SMPIR. */
   nir_tex_instr *tg4 = nir_tex_instr_create(b.shader, 2);
   tg4->op = nir_texop_tg4;
   tg4->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tg4->dest_type = nir_type_float32;
   tg4->coord_components = 2;
   tg4->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord2);
   tg4->src[1] = nir_tex_src_for_ssa(nir_tex_src_offset, offset);
   nir_def_init(&tg4->instr, &tg4->def, 4, 32);
   nir_builder_instr_insert(&b, &tg4->instr);

   const bool progress = agx_nir_lower_agx3d_texture_early(b.shader, false, true);
   nir_validate_shader(b.shader, "after agx_nir_lower_agx3d_texture_early");

   unsigned level_queries = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex &&
             nir_instr_as_tex(instr)->op == nir_texop_query_levels)
            level_queries++;
      }
   }

   const bool ok =
      progress &&
      txb->op == nir_texop_txl &&
      nir_tex_instr_src_index(txb, nir_tex_src_min_lod) < 0 &&
      nir_tex_instr_src_index(txb, nir_tex_src_bias) < 0 &&
      nir_tex_instr_src_index(txb, nir_tex_src_lod) >= 0 &&
      txd->op == nir_texop_txl &&
      nir_tex_instr_src_index(txd, nir_tex_src_ddx) < 0 &&
      nir_tex_instr_src_index(txd, nir_tex_src_lod) >= 0 &&
      nir_tex_instr_src_index(txl, nir_tex_src_offset) < 0 &&
      level_queries == 1 &&
      nir_tex_instr_src_index(tg4, nir_tex_src_offset) >= 0;
   ralloc_free(b.shader);
   return ok;
}

/* A runtime offset loaded as one vector has no allocated_vec entry. Combining
 * it with a shadow comparator used to call agx_emit_extract() anyway and abort
 * while compiling the pcOffset sampler2DShadow bias fragment. Keep the same
 * source shape in a complete G17P fragment compile. */
static bool
test_g17p_shadow_runtime_offset_compile(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P shadow bias with runtime offset");
   nir_def *offset = nir_load_constant_agx(
      &b, 2, 32, nir_imm_int64(&b, 0), nir_imm_int(&b, 0),
      .format = PIPE_FORMAT_R32_UINT, .sign_extend = true);
   nir_def *image =
      nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 0);
   nir_def *set = nir_load_descriptor_set_agx(&b, 0);
   nir_def *heap_index = nir_load_global_constant(
      &b, 1, 16, nir_iadd_imm(&b, set, 40), .align_mul = 2,
      .access = ACCESS_CAN_SPECULATE);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_imm_int(&b, 32), heap_index, .desc_set = 0);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 6);
   tex->op = nir_texop_txb;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_shadow = true;
   tex->is_new_style_shadow = true;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_vec2(&b, 0.25f, 0.75f));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_comparator, nir_imm_float(&b, 0.5f));
   tex->src[2] =
      nir_tex_src_for_ssa(nir_tex_src_bias, nir_imm_float(&b, 0.25f));
   tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_offset, offset);
   tex->src[4] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   tex->src[5] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   nir_def_init(&tex->instr, &tex->def, 1, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *value = nir_channel(&b, &tex->def, 0);
   fs_store_colour(&b, nir_vec4(&b, value, value, value,
                                nir_imm_float(&b, 1.0f)));
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   const int backend2 =
      nir_tex_instr_src_index(tex, nir_tex_src_backend2);
   struct agx_g17p_combined_sampler_binding binding;
   bool ok = backend2 >= 0 &&
             nir_src_num_components(tex->src[backend2].src) == 2 &&
             agx_nir_lower_g17p_combined_sampler(b.shader, &binding, true);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   key.reserved_preamble = 24;
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &key, &part) &&
         part.binary != NULL && part.info.binary_size > 0;
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static nir_shader *
build_cs_storage_image(unsigned mode)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX3 bounded storage image CS");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   const enum pipe_format format =
      mode >= 6 ? PIPE_FORMAT_R32_UINT
      : mode == 1 ? PIPE_FORMAT_R16G16B16A16_UINT
                : PIPE_FORMAT_R8G8B8A8_UINT;
   const enum glsl_sampler_dim dim =
      mode == 2 ? GLSL_SAMPLER_DIM_3D
      : mode == 3 ? GLSL_SAMPLER_DIM_MS
                  : GLSL_SAMPLER_DIM_2D;
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *coord = nir_vec4(&b, nir_channel(&b, gid, 0),
                             nir_channel(&b, gid, 1), zero, zero);
   nir_def *texture = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 64), .desc_set = 0);
   nir_def *pbe = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 96), .desc_set = 0);
   if (mode == 4) {
      nir_def *descriptor = nir_load_from_texture_handle_agx(&b, texture);
      nir_def *width = nir_load_global_constant(
         &b, 1, 32, descriptor, .align_mul = 4);
      coord = nir_vec4(&b, nir_iand(&b, nir_channel(&b, gid, 0), width),
                       nir_channel(&b, gid, 1), zero, zero);
   }
   if (mode == 5)
      nir_push_if(&b, nir_ult_imm(&b, nir_channel(&b, gid, 0), 4));
   nir_def *value = mode >= 6
      ? nir_imm_ivec4(&b, 9, 8, 7, 6)
      : nir_bindless_image_load(
           &b, 4, 32, texture, coord, zero, zero, .image_dim = dim,
           .image_array = false, .format = format, .access = ACCESS_RESTRICT);
   if (mode == 5) {
      nir_store_scratch(&b, value, zero, .write_mask = 0xf, .align_mul = 16);
      nir_pop_if(&b, NULL);
      nir_push_if(&b, nir_ult_imm(&b, nir_channel(&b, gid, 1), 4));
   }
   if (mode == 8) {
      nir_barrier(&b, .memory_scope = SCOPE_DEVICE,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_image | nir_var_mem_ssbo);
   }
   /* A memory-model write_after_read shader whose guard lives in workgroup
    * memory names only image and shared memory in its release fence. The
    * state-0 route cannot order the image either way. */
   if (mode == 9) {
      nir_barrier(&b, .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_RELEASE,
                  .memory_modes = nir_var_image | nir_var_mem_shared);
   }
   /* An image-only barrier is the same defect with nothing else named. */
   if (mode == 10) {
      nir_barrier(&b, .memory_scope = SCOPE_DEVICE,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_image);
   }
   nir_bindless_image_store(
      &b, pbe, coord, zero,
      mode == 5 ? nir_imm_ivec4(&b, 9, 8, 7, 6) : value, zero,
      .image_dim = dim, .image_array = false, .format = format,
      .access = ACCESS_RESTRICT | (mode == 7 ? ACCESS_COHERENT : 0));
   if (mode == 7) {
      nir_bindless_image_atomic(
         &b, 32, pbe, coord, zero, nir_imm_int(&b, 1),
         .atomic_op = nir_atomic_op_iadd, .image_dim = dim,
         .image_array = false, .format = format,
         .access = ACCESS_RESTRICT | ACCESS_COHERENT);
   }
   if (mode == 5)
      nir_pop_if(&b, NULL);
   return b.shader;
}

static bool
test_storage_image_lowering(void)
{
   for (unsigned mode = 0; mode < 8; ++mode) {
      nir_shader *nir = build_cs_storage_image(mode);
      struct agx_g17p_storage_image_binding binding = {
         .texture_offset_B = UINT32_MAX,
         .pbe_offset_B = UINT32_MAX,
      };
      const bool lowered = agx_nir_lower_g17p_storage_image(nir, &binding);
      unsigned bindless_handles = 0, bindless_ops = 0;
      unsigned ordinary_loads = 0, ordinary_stores = 0, atomics = 0;
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               const nir_intrinsic_op op =
                  nir_instr_as_intrinsic(instr)->intrinsic;
               bindless_handles += op == nir_intrinsic_bindless_image_agx;
               bindless_ops += op == nir_intrinsic_bindless_image_load ||
                               op == nir_intrinsic_bindless_image_store;
               atomics += op == nir_intrinsic_bindless_image_atomic ||
                          op == nir_intrinsic_bindless_image_atomic_swap;
               ordinary_loads += op == nir_intrinsic_image_load;
               ordinary_stores += op == nir_intrinsic_image_store;
            }
         }
      }

      const bool ok = mode == 0 || mode == 4 || mode == 5 || mode == 6 ||
                      mode == 7
         ? lowered && binding.texture_offset_B == 64 &&
              binding.pbe_offset_B == 96 &&
              binding.write_only == (mode == 6) &&
              bindless_handles == (mode == 4 || mode == 7 ? 1 : 0) &&
              bindless_ops == 0 && ordinary_loads == (mode >= 6 ? 0 : 1) &&
              ordinary_stores == 1 && atomics == (mode == 7 ? 1 : 0)
         : !lowered && binding.texture_offset_B == UINT32_MAX &&
              binding.pbe_offset_B == UINT32_MAX && bindless_handles == 2 &&
              bindless_ops == 2 && ordinary_loads == 0 &&
              ordinary_stores == 0 && atomics == 0;
      ralloc_free(nir);
      if (!ok)
         return false;
   }

   /* A barrier that names the image at all must stay on the general indexed
    * storage path: measured on G17P, a memory barrier does not order the
    * native state-0 image accesses, whatever else the barrier names. Mode 8
    * pairs the image with buffer memory, mode 9 with workgroup memory -- the
    * dEQP-VK.memory_model.write_after_read shape -- and mode 10 names the
    * image alone. */
   for (unsigned mode = 8; mode <= 10; ++mode) {
      nir_shader *nir = build_cs_storage_image(mode);
      struct agx_g17p_storage_image_binding binding = {
         .texture_offset_B = UINT32_MAX,
         .pbe_offset_B = UINT32_MAX,
      };
      const bool lowered = agx_nir_lower_g17p_storage_image(nir, &binding);
      const bool ok = !lowered && binding.texture_offset_B == UINT32_MAX &&
                      binding.pbe_offset_B == UINT32_MAX;
      ralloc_free(nir);
      if (!ok)
         return false;
   }

   nir_shader *nir;
   struct agx_g17p_storage_image_binding binding;

   /* Buffer-image lowering preserves its descriptor class while mapping the
    * coordinate to 2D.  A Gen9 linear buffer descriptor is valid through the
    * indexed texture/PBE route, but not through the direct state-0 image-view
    * pair selected here. */
   nir = build_cs_storage_image(0);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_bindless_image_load ||
                intr->intrinsic == nir_intrinsic_bindless_image_store) {
               nir_intrinsic_set_access(
                  intr, nir_intrinsic_access(intr) | ACCESS_BUFFER_AGX);
            }
         }
      }
   }
   binding = (struct agx_g17p_storage_image_binding){
      .texture_offset_B = UINT32_MAX,
      .pbe_offset_B = UINT32_MAX,
   };
   const bool buffer_lowered =
      agx_nir_lower_g17p_storage_image(nir, &binding);
   const bool buffer_ok =
      !buffer_lowered && binding.texture_offset_B == UINT32_MAX &&
      binding.pbe_offset_B == UINT32_MAX;
   ralloc_free(nir);
   if (!buffer_ok)
      return false;

   return true;
}

static nir_shader *
build_gallium_storage_write(enum pipe_format format, unsigned state,
                            enum gl_access_qualifier access)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "AGX3 Gallium write-only scalar storage image");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   BITSET_SET(b.shader->info.images_used, 0);

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *zero16 = nir_imm_intN_t(&b, 0, 16);
   nir_def *coord = nir_imm_ivec4(&b, 0, 0, 0, 0);
   nir_image_store(&b, nir_imm_intN_t(&b, state, 16), coord, zero,
                   nir_imm_ivec4(&b, 9, 0, 0, 0), zero16,
                   .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
                   .format = format, .access = access,
                   .src_type = format == PIPE_FORMAT_R32_SINT
                                  ? nir_type_int32
                                  : nir_type_uint32);
   return b.shader;
}

static bool
test_gallium_storage_write_lowering(void)
{
   const enum pipe_format formats[] = {
      PIPE_FORMAT_R32_UINT,
      PIPE_FORMAT_R32_SINT,
   };

   for (unsigned i = 0; i < ARRAY_SIZE(formats); ++i) {
      nir_shader *nir = build_gallium_storage_write(
         formats[i], 1, ACCESS_RESTRICT);
      agx_preprocess_nir(nir);
      bool ok = agx_nir_lower_g17p_gallium_storage_write(nir, 1);
      unsigned stores = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_image_store) {
               stores++;
               ok &= nir_src_is_const(intr->src[0]) &&
                     nir_src_as_uint(intr->src[0]) == 0;
            }
         }
      }

      struct agx_shader_key key = {0};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.no_preamble = true;
      key.reserved_preamble = AGX_G17P_STORAGE_ROOT_HALFS;
      struct agx_shader_part part = {0};
      ok &= stores == 1 && agx_compile_shader_nir(nir, &key, &part) &&
            part.binary != NULL && part.info.binary_size > 0;
      free(part.binary);
      ralloc_free(nir);
      if (!ok)
         return false;
   }

   nir_shader *coherent = build_gallium_storage_write(
      PIPE_FORMAT_R32_SINT, 1, ACCESS_RESTRICT | ACCESS_COHERENT);
   nir_shader *wrong_state = build_gallium_storage_write(
      PIPE_FORMAT_R32_SINT, 2, ACCESS_RESTRICT);
   const bool rejected =
      !agx_nir_lower_g17p_gallium_storage_write(coherent, 1) &&
      !agx_nir_lower_g17p_gallium_storage_write(wrong_state, 1);
   ralloc_free(coherent);
   ralloc_free(wrong_state);
   return rejected;
}

static nir_shader *
build_cs_2d_image_access(bool store)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      store ? "AGX3 robust 2D image store" : "AGX3 robust 2D image load");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_variable *image = nir_variable_create(
      b.shader, nir_var_image,
      glsl_image_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_UINT), "image");
   image->data.image.format = PIPE_FORMAT_R8G8B8A8_UINT;
   nir_def *deref = &nir_build_deref_var(&b, image)->def;
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *coord = nir_vec4(&b, nir_channel(&b, gid, 0),
                             nir_channel(&b, gid, 1), zero, zero);

   if (store) {
      nir_image_deref_store(
         &b, deref, coord, zero, nir_imm_ivec4(&b, 1, 2, 3, 4), zero,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
         .format = PIPE_FORMAT_R8G8B8A8_UINT, .access = ACCESS_RESTRICT,
         .src_type = nir_type_uint32);
   } else {
      nir_def *value = nir_image_deref_load(
         &b, 4, 32, deref, coord, zero, zero,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
         .format = PIPE_FORMAT_R8G8B8A8_UINT, .access = ACCESS_RESTRICT,
         .dest_type = nir_type_uint32);

      /* Keep the robust result live so the explicit zero select is part of
       * the test's observable graph. */
      nir_store_scratch(&b, value, nir_imm_int(&b, 0), .write_mask = 0xf,
                        .align_mul = 16);
   }

   return b.shader;
}

static nir_shader *
build_cs_2d_texel_fetch(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "AGX3 robust sampled 2D texel fetch");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *handle = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 64), .desc_set = 0);
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_uint32;
   tex->coord_components = 2;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_trim_vector(&b, gid, 2));
   tex->src[1] =
      nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
   tex->src[2] =
      nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(&b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_store_scratch(&b, &tex->def, nir_imm_int(&b, 0), .write_mask = 0xf,
                     .align_mul = 16);
   return b.shader;
}

static bool
test_g17p_2d_robustness_scope(void)
{
   for (unsigned store = 0; store < 2; ++store) {
      nir_shader *nir = build_cs_2d_image_access(store);
      agx_nir_lower_texture_early_with_2d_robustness(nir, true, true);

      unsigned loads = 0, stores = 0, sizes = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_image_deref_load:
               loads++;
               break;
            case nir_intrinsic_image_deref_store:
               stores++;
               break;
            case nir_intrinsic_image_deref_size:
               sizes++;
               break;
            default:
               break;
            }
         }
      }

      const bool ok = store ? loads == 0 && stores == 1 && sizes == 0
                            : loads == 1 && stores == 0 && sizes == 1;
      ralloc_free(nir);
      if (!ok)
         return false;
   }

   nir_shader *nir = build_cs_2d_texel_fetch();
   agx_nir_lower_texture_for_generation(nir, 9);
   unsigned fetches = 0, selects = 0, descriptor_loads = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex) {
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (tex->op == nir_texop_txf) {
               const int coord =
                  nir_tex_instr_src_index(tex, nir_tex_src_backend1);
               fetches += coord >= 0 &&
                          tex->src[coord].src.ssa->num_components == 2;
            }
         } else if (instr->type == nir_instr_type_alu) {
            selects += nir_instr_as_alu(instr)->op == nir_op_bcsel;
         } else if (instr->type == nir_instr_type_intrinsic) {
            descriptor_loads +=
               nir_instr_as_intrinsic(instr)->intrinsic ==
               nir_intrinsic_load_global_constant;
         }
      }
   }
   const bool sampled_ok = fetches == 1 && selects >= 2 && descriptor_loads > 0;
   ralloc_free(nir);
   if (!sampled_ok)
      return false;

   return true;
}

static nir_shader *
build_cs_texel_fetch_case(bool is_ms, bool is_array, int32_t const_lod_or_sample,
                          bool runtime_lod_or_sample, unsigned backend_flags)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "AGX3 texel fetch hw bounds");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *handle = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 64), .desc_set = 0);
   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = is_ms ? nir_texop_txf_ms : nir_texop_txf;
   tex->sampler_dim = is_ms ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D;
   tex->is_array = is_array;
   tex->dest_type = nir_type_uint32;
   tex->coord_components = is_array ? 3 : 2;
   tex->backend_flags = backend_flags;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_trim_vector(&b, gid, tex->coord_components));
   tex->src[1] =
      nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
   tex->src[2] = nir_tex_src_for_ssa(
      is_ms ? nir_tex_src_ms_index : nir_tex_src_lod,
      runtime_lod_or_sample ? nir_channel(&b, gid, 2)
                            : nir_imm_int(&b, const_lod_or_sample));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_store_scratch(&b, &tex->def, nir_imm_int(&b, 0), .write_mask = 0xf,
                     .align_mul = 16);
   return b.shader;
}

static bool
ssa_uses_umin_65535(nir_def *def, unsigned depth)
{
   if (!def || depth > 8 || nir_def_instr(def)->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(nir_def_instr(def));
   if (alu->op == nir_op_umin) {
      for (unsigned s = 0; s < 2; ++s) {
         if (nir_src_is_const(alu->src[s].src) &&
             nir_src_as_uint(alu->src[s].src) == 65535)
            return true;
      }
   }
   for (unsigned s = 0; s < nir_op_infos[alu->op].num_inputs; ++s) {
      if (ssa_uses_umin_65535(alu->src[s].src.ssa, depth + 1))
         return true;
   }
   return false;
}

static bool
test_g17c_txf_layer_saturation(void)
{
   /* 0: 2D array lod=0 with HW_TXF_BOUNDS -> layer saturated into backend1, no desc loads/bcsel.
    * 1: 2D array lod=0 with NO_CLAMP -> no saturation.
    * 2: 2D array lod=0 WITHOUT HW_TXF_BOUNDS (plain gen10) -> no saturation (scoped to the flag).
    * 3: 2D array runtime lod with HW_TXF_BOUNDS -> levels check + bcsel + safe lod.
    * 4: 2D array const lod=65536 with HW_TXF_BOUNDS -> does NOT truncate to 0, emits levels check + bcsel.
    * 5: 2D array const lod=-65536 with HW_TXF_BOUNDS -> does NOT truncate to 0, emits levels check + bcsel.
    * 6: 2D_MS runtime sample with HW_TXF_BOUNDS -> samples check + bcsel + umin 65535.
    * 7: 2D_MS_ARRAY runtime sample with HW_TXF_BOUNDS -> high-half layer in backend1 saturated + samples check.
    * 8: 2D_MS const sample=65536 with HW_TXF_BOUNDS -> saturated to 65535 (not 0), emits samples check + bcsel. */
   for (unsigned mode = 0; mode < 9; ++mode) {
      const bool is_ms = mode >= 6;
      const bool is_array = mode != 6 && mode != 8;
      const bool runtime = mode == 3 || mode == 6 || mode == 7;
      const int32_t const_val =
         (mode == 4 || mode == 8) ? 65536 : (mode == 5 ? -65536 : 0);
      const unsigned flags = mode == 1 ? AGX_TEXTURE_FLAG_NO_CLAMP : 0;
      const unsigned gen =
         mode == 2 ? 10 : (10 | AGX_NIR_TEXTURE_HW_TXF_BOUNDS);

      nir_shader *nir =
         build_cs_texel_fetch_case(is_ms, is_array, const_val, runtime, flags);
      agx_nir_lower_texture_with_buffer_metadata(nir, gen, false, 0, false);

      unsigned fetches = 0, umin_65535 = 0, bcsels = 0, desc_loads = 0;
      bool backend1_layer_saturated = false;
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               int b1 = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
               if ((tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms) &&
                   b1 >= 0) {
                  fetches++;
                  nir_def *coord = tex->src[b1].src.ssa;
                  if (is_array &&
                      nir_def_instr(coord)->type == nir_instr_type_alu) {
                     nir_alu_instr *vec = nir_instr_as_alu(nir_def_instr(coord));
                     nir_def *last = vec->src[vec->def.num_components - 1].src.ssa;
                     if (is_ms &&
                         nir_def_instr(last)->type == nir_instr_type_alu) {
                        nir_alu_instr *pack =
                           nir_instr_as_alu(nir_def_instr(last));
                        if (pack->op == nir_op_pack_32_2x16_split)
                           backend1_layer_saturated =
                              ssa_uses_umin_65535(pack->src[1].src.ssa, 0);
                     } else {
                        backend1_layer_saturated = ssa_uses_umin_65535(last, 0);
                     }
                  }
               }
            } else if (instr->type == nir_instr_type_alu) {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (alu->op == nir_op_bcsel) {
                  bcsels++;
               } else if (alu->op == nir_op_umin) {
                  for (unsigned s = 0; s < 2; ++s) {
                     if (nir_src_is_const(alu->src[s].src) &&
                         nir_src_as_uint(alu->src[s].src) == 65535)
                        umin_65535++;
                  }
               }
            } else if (instr->type == nir_instr_type_intrinsic) {
               if (nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_load_global_constant)
                  desc_loads++;
            }
         }
      }

      const bool ok =
         mode == 0 ? (fetches == 1 && umin_65535 == 1 && backend1_layer_saturated &&
                      bcsels == 0 && desc_loads == 0)
         : mode == 1 ? (fetches == 1 && umin_65535 == 0 && !backend1_layer_saturated &&
                        bcsels == 0 && desc_loads == 0)
         : mode == 2 ? (fetches == 1 && umin_65535 == 0 && !backend1_layer_saturated &&
                        bcsels == 0 && desc_loads == 0)
         : mode == 3 ? (fetches == 1 && umin_65535 >= 2 && backend1_layer_saturated &&
                        bcsels >= 2 && desc_loads > 0)
         : (mode == 4 || mode == 5)
                     ? (fetches == 1 && umin_65535 == 1 && backend1_layer_saturated &&
                        bcsels >= 2 && desc_loads > 0)
         : mode == 6 ? (fetches == 1 && umin_65535 == 1 && bcsels >= 2 && desc_loads > 0)
         : mode == 7 ? (fetches == 1 && umin_65535 == 2 && backend1_layer_saturated &&
                        bcsels >= 2 && desc_loads > 0)
                     : (fetches == 1 && bcsels >= 2 && desc_loads > 0);
      ralloc_free(nir);
      if (!ok)
         return false;
   }

   return true;
}

static nir_shader *
build_cs_sampled_storage(unsigned mode)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "AGX3 sampled texture to storage image CS");
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 8;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *coord2 = nir_vec2(&b, nir_u2f32(&b, nir_channel(&b, gid, 0)),
                              nir_u2f32(&b, nir_channel(&b, gid, 1)));
   nir_def *texture = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 0), .desc_set = 0);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_imm_int(&b, 32), nir_imm_intN_t(&b, 28, 16), .desc_set = 0);
   nir_tex_instr *sample = nir_tex_instr_create(b.shader, 4);
   sample->op = nir_texop_txl;
   sample->sampler_dim = GLSL_SAMPLER_DIM_2D;
   sample->dest_type = nir_type_float32;
   sample->coord_components = 2;
   sample->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord2);
   sample->src[1] =
      nir_tex_src_for_ssa(nir_tex_src_texture_handle, texture);
   sample->src[2] =
      nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   sample->src[3] =
      nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 0.0f));
   nir_def_init(&sample->instr, &sample->def, 4, 32);
   nir_builder_instr_insert(&b, &sample->instr);

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *coord4 = nir_vec4(&b, nir_channel(&b, gid, 0),
                              nir_channel(&b, gid, 1), zero, zero);
   nir_def *pbe = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 32), .desc_set = 1);
   nir_def *value = mode == 2
      ? nir_fadd_imm(&b, &sample->def, 1.0f)
      : &sample->def;
   nir_bindless_image_store(
      &b, pbe, coord4, zero, value, zero,
      .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = false,
      .format = mode == 1 ? PIPE_FORMAT_R8G8B8A8_UINT
                          : PIPE_FORMAT_R8G8B8A8_UNORM,
      .access = ACCESS_RESTRICT);
   return b.shader;
}

static bool
test_sampled_storage_lowering(void)
{
   for (unsigned mode = 0; mode < 3; ++mode) {
      nir_shader *nir = build_cs_sampled_storage(mode);
      struct agx_g17p_sampled_storage_binding binding = {
         .texture_offset_B = UINT32_MAX,
         .sampler_index_offset_B = UINT32_MAX,
         .pbe_offset_B = UINT32_MAX,
         .texture_set = UINT8_MAX,
         .pbe_set = UINT8_MAX,
      };
      const bool lowered =
         agx_nir_lower_g17p_sampled_storage(nir, &binding);
      unsigned bindless_handles = 0, bindless_stores = 0;
      unsigned ordinary_stores = 0, samples = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex) {
               nir_tex_instr *tex = nir_instr_as_tex(instr);
               samples++;
               if (lowered &&
                   (nir_tex_instr_src_index(tex,
                                            nir_tex_src_texture_handle) >= 0 ||
                    nir_tex_instr_src_index(tex,
                                            nir_tex_src_sampler_handle) >= 0))
                  samples += 100;
            } else if (instr->type == nir_instr_type_intrinsic) {
               const nir_intrinsic_op op =
                  nir_instr_as_intrinsic(instr)->intrinsic;
               bindless_handles += op == nir_intrinsic_bindless_image_agx ||
                                   op == nir_intrinsic_bindless_sampler_agx;
               bindless_stores += op == nir_intrinsic_bindless_image_store;
               ordinary_stores += op == nir_intrinsic_image_store;
            }
         }
      }

      const bool ok = mode == 0
         ? lowered && binding.texture_set == 0 && binding.pbe_set == 1 &&
              binding.texture_offset_B == 0 &&
              binding.sampler_index_offset_B == 40 &&
              binding.pbe_offset_B == 32 && bindless_handles == 0 &&
              bindless_stores == 0 && ordinary_stores == 1 && samples == 1
         : !lowered && binding.texture_offset_B == UINT32_MAX &&
              binding.sampler_index_offset_B == UINT32_MAX &&
              binding.pbe_offset_B == UINT32_MAX &&
              binding.texture_set == UINT8_MAX &&
              binding.pbe_set == UINT8_MAX && bindless_handles == 3 &&
              bindless_stores == 1 && ordinary_stores == 0 && samples == 1;
      ralloc_free(nir);
      if (!ok)
         return false;
   }
   return true;
}

static bool
test_indexed_compute_sample(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "AGX3 indexed compute sample");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *element = nir_channel(&b, gid, 0);
   nir_def *offset = nir_imul_imm(&b, element, 72);
   nir_def *set = nir_load_descriptor_set_agx(&b, 4);
   nir_def *entry = nir_iadd(&b, set, nir_u2u64(&b, offset));
   nir_def *sampler_index = nir_load_global_constant(
      &b, 1, 16, nir_iadd_imm(&b, entry, 40), .align_mul = 2);
   nir_def *image = nir_bindless_image_agx(&b, offset, .desc_set = 4);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_iadd_imm(&b, offset, 32), sampler_index, .desc_set = 4);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_vec2(&b, nir_u2f32(&b, element), nir_imm_float(&b, 0.5f)));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   tex->src[3] =
      nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 0.0f));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_export_agx(&b, nir_channel(&b, &tex->def, 0), .base = 0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   bool ok = agx_nir_lower_g17p_indexed_samplers(b.shader, true);
   const int ti = nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
   const int si = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   ok &= ti >= 0 && si >= 0 &&
         nir_src_bit_size(tex->src[ti].src) == 64 &&
         nir_src_bit_size(tex->src[si].src) == 64;

   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
            lower_test_descriptor_set, nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   key.reserved_preamble = 32;
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &key, &part) && part.binary != NULL;
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

/* dEQP-VK.glsl.texture_functions.texture.samplercubearrayshadow_compute: a
 * compute shader with a derivative group, whose texture() on a cube-array
 * comparison sampler agx_nir_lower_compute_derivatives() turns into a txl with
 * a per-lane level. The indexed lowering must give it native texture and
 * sampler state on AGX3B as well: kept as a software heap handle, the sample
 * has no AGX3B encoding and the pipeline fails. */
static bool
test_indexed_compute_cube_array_shadow_level(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "AGX3 indexed compute cube-array shadow level");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *element = nir_channel(&b, gid, 0);
   nir_def *set = nir_load_descriptor_set_agx(&b, 4);
   nir_def *sampler_index = nir_load_global_constant(
      &b, 1, 16, nir_iadd_imm(&b, set, 40), .align_mul = 2);
   nir_def *image = nir_bindless_image_agx(&b, nir_imm_int(&b, 0), .desc_set = 4);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_imm_int(&b, 32), sampler_index, .desc_set = 4);

   nir_def *x = nir_u2f32(&b, element);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 5);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
   tex->is_array = true;
   tex->is_shadow = true;
   tex->is_new_style_shadow = true;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 4;
   tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_vec4(&b, x, nir_imm_float(&b, 0.5f), nir_imm_float(&b, 1.0f),
               nir_imm_float(&b, 1.0f)));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                     nir_fmul_imm(&b, x, 0.125f));
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod,
                                     nir_fmul_imm(&b, x, 0.25f));
   tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   tex->src[4] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   nir_def_init(&tex->instr, &tex->def, 1, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_export_agx(&b, &tex->def, .base = 0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   bool ok = agx_nir_lower_g17p_indexed_samplers(b.shader, true);
   ok &= nir_tex_instr_src_index(tex, nir_tex_src_texture_handle) < 0 &&
         nir_tex_instr_src_index(tex, nir_tex_src_texture_offset) >= 0 &&
         nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle) >= 0;

   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
            lower_test_descriptor_set, nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   key.reserved_preamble = 32;
   struct agx_shader_part part = {0};
   const bool compiled = ok && agx_compile_shader_nir(b.shader, &key, &part) &&
                         part.binary != NULL;
   if (!compiled)
      fprintf(stderr, "indexed compute cube-array shadow level: lowered=%u compiled=%u\n",
              ok, compiled);
   free(part.binary);
   ralloc_free(b.shader);
   return compiled;
}

static bool
test_indexed_compute_storage(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "AGX3 indexed compute sample and formatted store");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *element = nir_channel(&b, gid, 0);
   nir_def *offset = nir_imul_imm(&b, element, 72);
   nir_def *set = nir_load_descriptor_set_agx(&b, 4);
   nir_def *entry = nir_iadd(&b, set, nir_u2u64(&b, offset));
   nir_def *sampler_index = nir_load_global_constant(
      &b, 1, 16, nir_iadd_imm(&b, entry, 40), .align_mul = 2);
   nir_def *image = nir_bindless_image_agx(&b, offset, .desc_set = 4);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_iadd_imm(&b, offset, 32), sampler_index, .desc_set = 4);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_vec2(&b, nir_u2f32(&b, element), nir_imm_float(&b, 0.5f)));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   tex->src[3] =
      nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 0.0f));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *coord = nir_vec4(&b, element, zero, zero, zero);
   nir_def *pbe = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 176), .desc_set = 0);
   nir_def *data = nir_fadd_imm(&b, &tex->def, 0.25f);
   nir_bindless_image_store(
      &b, pbe, coord, zero, data, zero, .image_dim = GLSL_SAMPLER_DIM_2D,
      .image_array = false, .format = PIPE_FORMAT_R10G10B10A2_UNORM,
      .access = ACCESS_RESTRICT);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   struct agx_g17p_indexed_storage_binding binding = {
      .pbe_offset_B = UINT32_MAX,
      .pbe_set = UINT8_MAX,
   };
   bool ok = agx_nir_lower_g17p_indexed_samplers(b.shader, true) &&
             agx_nir_lower_g17p_indexed_storage(b.shader, &binding) &&
             binding.pbe_set == 0 && binding.pbe_offset_B == 176;

   unsigned bindless_stores = 0, ordinary_stores = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         const nir_intrinsic_op op =
            nir_instr_as_intrinsic(instr)->intrinsic;
         bindless_stores += op == nir_intrinsic_bindless_image_store;
         ordinary_stores += op == nir_intrinsic_image_store;
      }
   }
   ok &= bindless_stores == 0 && ordinary_stores == 1;

   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
            lower_test_descriptor_set, nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   key.reserved_preamble = 80;
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &key, &part) && part.binary != NULL;
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static bool
test_buffer_fetch_r32_store(void)
{
   /* CTS buffer_view_memory_test_*_compute uses a textureBuffer followed by
    * a scalar R32 image store after nir_opt_shrink_stores. Exercise the full
    * metadata, indexed texture, PBE binding and native integer-state1 path. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P buffer fetch to R32 store");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *index = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *handle = nir_bindless_image_agx(&b, zero, .desc_set = 0);
   nir_def *value = nir_build_tex(
      &b, nir_texop_txf, .texture_handle = handle, .coord = index,
      .lod = zero, .dim = GLSL_SAMPLER_DIM_BUF, .dest_type = nir_type_uint32);
   nir_def *pbe = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 128), .desc_set = 0);
   nir_bindless_image_store(
      &b, pbe, nir_vec4(&b, index, index, zero, zero), zero,
      nir_channel(&b, value, 0), zero, .image_dim = GLSL_SAMPLER_DIM_2D,
      .format = PIPE_FORMAT_R32_UINT, .src_type = nir_type_uint32,
      .access = ACCESS_NON_READABLE);
   agx_nir_lower_texture_with_buffer_metadata(b.shader, 9, false, 0, false);
   struct buffer_metadata_fixture fixture = {.elements = 64};
   bool progress;
   do {
      progress = nir_shader_intrinsics_pass(b.shader,
         fold_buffer_descriptor, nir_metadata_control_flow, &fixture);
      progress |= nir_opt_constant_folding(b.shader);
      progress |= nir_opt_dead_cf(b.shader);
      progress |= nir_opt_dce(b.shader);
   } while (progress);
   agx_preprocess_nir_g17p(b.shader);
   struct agx_g17p_indexed_storage_binding binding = {0};
   bool ok = agx_nir_lower_g17p_indexed_samplers(b.shader, true) &&
             agx_nir_lower_g17p_indexed_storage(b.shader, &binding) &&
             binding.pbe_offset_B == 128 && binding.pbe_set == 0;
   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
            lower_test_descriptor_set, nir_metadata_control_flow, NULL);
   agx_preprocess_nir_g17p(b.shader);
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
      .promote_constants = true,
      .reserved_preamble = 32,
   };
   struct agx_shader_part part = {0};
   ok &= agx_compile_shader_nir(b.shader, &key, &part) && part.binary;
   const uint8_t native[] = {
      0xd7, 0x10, 0x54, 0x00, 0x40, 0x40, 0x00, 0x00,
      0x00, 0x04, 0x00, 0x00, 0x88, 0x1a, 0x09, 0x00,
   };
   bool saw_native_store = false;
   for (size_t i = 0; part.binary && i + sizeof(native) <= part.info.binary_size; ++i)
      saw_native_store |= memcmp((uint8_t *)part.binary + i,
                                 native, sizeof(native)) == 0;
   if (!ok || !saw_native_store)
      fprintf(stderr, "buffer R32 store: compile/binding=%u native=%u size=%u\n",
              ok, saw_native_store, part.info.binary_size);
   ok &= saw_native_store && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 20;
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static bool
test_buffer_fetch_unorm_store_fallback(void)
{
   /* The native indexed state-1 write following a lowered buffer fetch
    * resolves an empty normalized state on G17P.  Preserve the descriptor-
    * backed bindless write while still lowering the fetch to the native
    * indexed texture heap. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "G17P buffer fetch to normalized store fallback");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *index = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *zero = nir_imm_int(&b, 0);
   nir_def *handle = nir_bindless_image_agx(&b, zero, .desc_set = 0);
   nir_def *value = nir_build_tex(
      &b, nir_texop_txf, .texture_handle = handle, .coord = index,
      .lod = zero, .dim = GLSL_SAMPLER_DIM_BUF, .dest_type = nir_type_float32);
   nir_def *pbe = nir_bindless_image_agx(
      &b, nir_imm_int(&b, 128), .desc_set = 0);
   nir_bindless_image_store(
      &b, pbe, nir_vec4(&b, index, index, zero, zero), zero, value, zero,
      .image_dim = GLSL_SAMPLER_DIM_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM, .src_type = nir_type_float32,
      .access = ACCESS_NON_READABLE);

   agx_nir_lower_texture_with_buffer_metadata(b.shader, 9, false, 0, false);
   struct buffer_metadata_fixture fixture = {.elements = 64};
   bool progress;
   do {
      progress = nir_shader_intrinsics_pass(
         b.shader, fold_buffer_descriptor, nir_metadata_control_flow,
         &fixture);
      progress |= nir_opt_constant_folding(b.shader);
      progress |= nir_opt_dead_cf(b.shader);
      progress |= nir_opt_dce(b.shader);
   } while (progress);
   agx_preprocess_nir_g17p(b.shader);

   struct agx_g17p_indexed_storage_binding binding = {0};
   bool ok = agx_nir_lower_g17p_indexed_samplers(b.shader, true) &&
             !agx_nir_lower_g17p_indexed_storage(b.shader, &binding);
   unsigned bindless_stores = 0, ordinary_stores = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         const nir_intrinsic_op op =
            nir_instr_as_intrinsic(instr)->intrinsic;
         bindless_stores += op == nir_intrinsic_bindless_image_store;
         ordinary_stores += op == nir_intrinsic_image_store;
      }
   }
   ok &= bindless_stores == 1 && ordinary_stores == 0;
   ralloc_free(b.shader);
   return ok;
}

/* Apple's builtin end-of-tile factory, asked for a 2D-multisample attachment
 * (texture type 4) instead of a 2D one (texture type 2), emits a second
 * program per selector. The pair differs in exactly three bytes and two
 * encoder fields -- PACK_IMGWIR field 5 (4 -> 6) and IMGWBLK field 30
 * (1 -> 2) -- and no instruction is added, because the block store covers
 * every sample of the tile. That is the whole of "G17P can store an
 * unresolved multisample surface"; see isa/README-metal-witness.md for how
 * the witnesses were generated and round-tripped.
 */
static bool
test_g17p_multisample_eot_block_store(void)
{
   /* Native U8NORM key 0x10020300 with texture type 4: selector 0x75 -> 0x72. */
   static const uint8_t expected_u8norm[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x06, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0x28, 0x72, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   /* Native I32 key with texture type 4: selector 0x35 -> 0x32. */
   static const uint8_t expected_i32[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x06, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0x28, 0x32, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   /* The resolving forms of the same two selectors, unchanged. */
   static const uint8_t expected_u8norm_resolved[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x75, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   static const uint8_t expected_i32_resolved[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x35, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   /* Compile one end-of-tile store. dim selects the IMGWBLK dimension the
    * shader asks for; the key states which Apple builtin the caller means.
    * Returning the emitted bytes lets the caller assert both the match and
    * the refusals. */
   struct eot_case {
      bool key_multisample;
      enum pipe_format format;
      uint8_t words;
      bool i32;
      bool i32_four_words;
      bool i32_signed_four_words;
      const uint8_t *expected;
      size_t expected_size;
   };
   const struct eot_case cases[] = {
      /* The two witnessed selectors, unresolved. */
      {true, PIPE_FORMAT_R8_UNORM, 0, false, false, false, expected_u8norm,
       sizeof(expected_u8norm)},
      {true, PIPE_FORMAT_R32_UINT, 1, true, false, false, expected_i32,
       sizeof(expected_i32)},
      /* The same graphs without the key must still emit the resolving
       * programs, byte for byte. Their only difference from the pair above is
       * the three witnessed bytes, so this is what proves the key -- not some
       * incidental compiler state -- is what selects between them. */
      {false, PIPE_FORMAT_R8_UNORM, 0, false, false, false,
       expected_u8norm_resolved, sizeof(expected_u8norm_resolved)},
      {false, PIPE_FORMAT_R32_UINT, 1, true, false, false,
       expected_i32_resolved, sizeof(expected_i32_resolved)},
      /* Internal formats 10, 11 and 17 (signed, unsigned, float) give the
       * same texture-type-4 builtin with four components, as with type 2. */
      {true, PIPE_FORMAT_R32_UINT, 4, true, false, true, expected_i32,
       sizeof(expected_i32)},
   };

   bool ok = true;
   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      const struct eot_case *c = &cases[i];
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "G17P MSAA RT0 EOT");
      nir_def *coords =
         nir_trim_vector(&b, nir_load_global_invocation_id(&b, 32), 2);
      nir_image_store_block_agx(&b, nir_imm_intN_t(&b, 0, 16),
                                nir_imm_int(&b, 0), coords, .format = c->format,
                                .image_dim = GLSL_SAMPLER_DIM_2D,
                                .explicit_coord = true);
      agx_preprocess_nir(b.shader);

      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .no_preamble = true,
         .g17p_rt0_eot_multisample = c->key_multisample,
         .g17p_rt0_eot_words = c->words,
         .g17p_rt0_eot_i32 = c->i32,
         .g17p_rt0_eot_i32_four_words = c->i32_four_words,
         .g17p_rt0_eot_i32_signed_four_words = c->i32_signed_four_words,
      };
      struct agx_shader_part part = {0};
      const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);

      bool case_ok;
      if (c->expected) {
         case_ok = compiled && part.binary &&
                   part.info.binary_size == c->expected_size &&
                   part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
                   part.info.agx3_staging_halfs == 8 &&
                   memcmp(part.binary, c->expected, c->expected_size) == 0;
      } else {
         case_ok = !compiled;
      }

      if (!case_ok) {
         fprintf(stderr,
                 "G17P multisample EOT case %u: compiled=%u size=%u "
                 "expected=%s\n",
                 i, compiled, compiled ? part.info.binary_size : 0,
                 c->expected ? "bytes" : "refusal");
         if (compiled && part.binary) {
            for (unsigned j = 0; j < part.info.binary_size; ++j)
               fprintf(stderr, "%02x", ((const uint8_t *)part.binary)[j]);
            fputc('\n', stderr);
         }
         ok = false;
      }

      free(part.binary);
      ralloc_free(b.shader);
   }

   return ok;
}

static bool
test_g17p_eot_block_store(void)
{
   /* Genuine applegpu-nt builtin EOT, key0x10020300, RTmask1. Native fragment
    * metadata requests tableVA64 at u0 and imageblock byteoffset32 at u4;
    * common0=3 GPRdwords, fields38/42 D=16. Not a compute-wrapper prefix. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P RT0 EOT");
   nir_def *coords =
      nir_trim_vector(&b, nir_load_global_invocation_id(&b, 32), 2);
   nir_image_store_block_agx(
      &b, nir_imm_intN_t(&b, 0, 16), nir_imm_int(&b, 0), coords,
      .format = PIPE_FORMAT_R8_UNORM, .image_dim = GLSL_SAMPLER_DIM_2D,
      .explicit_coord = true);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   static const uint8_t expected[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x75, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };

   bool ok = compiled && part.info.binary_size == sizeof(expected) &&
             part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
             part.info.agx3_staging_halfs == 8 &&
             memcmp(part.binary, expected, sizeof(expected)) == 0;

   ok &= sizeof(expected) == 68 && part.info.main_offset == 0 &&
         part.info.main_size == 68;
   if (!ok) {
      fprintf(stderr,
              "AGX3b EOT block-store compiler mismatch: compiled=%u size=%u "
              "gprs=%u\n",
              compiled, part.info.binary_size, part.info.nr_gprs);
      if (compiled) {
         for (unsigned i = 0; i < part.info.binary_size; ++i)
            fprintf(stderr, "%02x", ((const uint8_t *)part.binary)[i]);
         fputc('\n', stderr);
      }
   }

   free(part.binary);
   ralloc_free(b.shader);

   /* R5G6B5 keeps three 16-bit float lanes (two 32-bit words), as Apple's
    * B5G6R5Unorm fragment does, so the store selects the 16-bit lane class
    * 0x15. PBE owns the 5:6:5 conversion. */
   nir_builder rgb565 = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P RGB565 RT0 EOT");
   coords = nir_trim_vector(
      &rgb565, nir_load_global_invocation_id(&rgb565, 32), 2);
   nir_image_store_block_agx(
      &rgb565, nir_imm_intN_t(&rgb565, 0, 16), nir_imm_int(&rgb565, 0),
      coords, .format = PIPE_FORMAT_R16_FLOAT,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   agx_preprocess_nir(rgb565.shader);
   memset(&part, 0, sizeof(part));
   key.g17p_rt0_eot_words = 2;
   const bool rgb565_compiled =
      agx_compile_shader_nir(rgb565.shader, &key, &part);
   static const uint8_t expected_rgb565[] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06,
      0x0c, 0x9d, 0x10, 0x26, 0x24, 0xa9, 0x10, 0x26,
      0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00,
      0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00,
      0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x15, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   ok &= rgb565_compiled &&
         part.info.binary_size == sizeof(expected_rgb565) &&
         part.info.main_offset == 0 &&
         part.info.main_size == sizeof(expected_rgb565) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 &&
         memcmp(part.binary, expected_rgb565, sizeof(expected_rgb565)) == 0;
   if (!rgb565_compiled)
      fprintf(stderr, "G17P RGB565 EOT did not compile\n");
   if (rgb565_compiled)
      free(part.binary);
   ralloc_free(rgb565.shader);
   key.g17p_rt0_eot_words = 0;

   /* Native mixed RGBA16F/R8 EOT at offsets0/8 and sample stride16.
    * Both physical partition quotients are zero; the partitions0/1 input
    * that produced the longer cross-partition program was an input error,
    * not a second layout. The independent native oracle has
    * SHA256 b3191cbbd673db5fb00d3e744f60ed59befad9bc19a2bf30aa9420effa63df12.
    * The driver reaches it the same way, naming each target's selector by
    * its logical format (agx_bg_eot.c), because a physical store format does
    * not name one selector: I32 alone is derived for three. */
   nir_builder mrt = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P WBOIT MRT EOT");
   coords = nir_trim_vector(&mrt, nir_load_global_invocation_id(&mrt, 32), 2);
   nir_image_store_block_agx(
      &mrt, nir_imm_intN_t(&mrt, 0, 16), nir_imm_int(&mrt, 0), coords,
      .format = PIPE_FORMAT_R16_FLOAT,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   nir_image_store_block_agx(
      &mrt, nir_imm_intN_t(&mrt, 1, 16), nir_imm_int(&mrt, 8), coords,
      .format = PIPE_FORMAT_R8_UNORM, .image_dim = GLSL_SAMPLER_DIM_2D,
      .explicit_coord = true);
   agx_preprocess_nir(mrt.shader);
   memset(&part, 0, sizeof(part));
   key.g17p_mrt_eot_count = 2;
   key.g17p_mrt_eot_selectors[0] = 0x15;
   key.g17p_mrt_eot_selectors[1] = 0x75;
   const bool mrt_compiled = agx_compile_shader_nir(mrt.shader, &key, &part);
   static const uint8_t expected_mrt[] = {
      0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,0x0c,0x9d,0x10,0x26,
      0x24,0xa9,0x10,0x26,0x9f,0x10,0x54,0x02,0x03,0x04,0x14,0x00,
      0x50,0x26,0x0a,0x00,0x9f,0x20,0x54,0x00,0x03,0x00,0x10,0x00,
      0x50,0x26,0x0a,0x00,0x97,0x05,0x54,0x04,0x00,0x08,0x00,0x00,
      0xd0,0x40,0x10,0x00,0x0b,0x82,0x21,0x04,0x1b,0x80,0x21,0x04,
      0x97,0x05,0x54,0x0c,0x00,0x10,0x00,0x00,0xd0,0x44,0x10,0x00,
      0x4b,0x03,0x2f,0x00,0x42,0x00,0x00,0x04,0x00,0x00,0x5b,0x01,
      0x2f,0x00,0x42,0x00,0x00,0x04,0x00,0x00,0x57,0x10,0x54,0x00,
      0x00,0x00,0x00,0x80,0xa8,0x15,0x00,0x00,0x57,0x20,0x54,0x08,
      0x00,0x10,0x00,0x90,0xa8,0x75,0x00,0x00,0x0e,0x00,0x00,0x00,
   };
   ok &= mrt_compiled && part.info.binary_size == sizeof(expected_mrt) &&
         part.info.main_offset == 0 && part.info.main_size == sizeof(expected_mrt) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 16 &&
         memcmp(part.binary, expected_mrt, sizeof(expected_mrt)) == 0;
   if (mrt_compiled)
      free(part.binary);
   ralloc_free(mrt.shader);

   nir_builder rgba8_mrt = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P RGBA8 MRT EOT");
   coords = nir_trim_vector(
      &rgba8_mrt, nir_load_global_invocation_id(&rgba8_mrt, 32), 2);
   nir_image_store_block_agx(
      &rgba8_mrt, nir_imm_intN_t(&rgba8_mrt, 0, 16),
      nir_imm_int(&rgba8_mrt, 0), coords, .format = PIPE_FORMAT_R8_UNORM,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   nir_image_store_block_agx(
      &rgba8_mrt, nir_imm_intN_t(&rgba8_mrt, 1, 16),
      nir_imm_int(&rgba8_mrt, 4), coords, .format = PIPE_FORMAT_R8_UNORM,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   agx_preprocess_nir(rgba8_mrt.shader);
   memset(&part, 0, sizeof(part));
   key.g17p_mrt_eot_selectors[0] = 0x75;
   const bool rgba8_mrt_compiled =
      agx_compile_shader_nir(rgba8_mrt.shader, &key, &part);
   static const uint8_t expected_rgba8_mrt[] = {
      0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,
      0x0c,0x9d,0x10,0x26,0x24,0xa9,0x10,0x26,
      0x9f,0x10,0x54,0x02,0x03,0x04,0x14,0x00,
      0x50,0x26,0x0a,0x00,0x9f,0x20,0x54,0x00,
      0x03,0x00,0x10,0x00,0x50,0x26,0x0a,0x00,
      0x97,0x05,0x54,0x04,0x00,0x08,0x00,0x00,
      0xd0,0x40,0x10,0x00,0x0b,0x82,0x21,0x04,
      0x1b,0x80,0x21,0x04,0x97,0x05,0x54,0x0c,
      0x00,0x10,0x00,0x00,0xd0,0x44,0x10,0x00,
      0x4b,0x03,0x2f,0x00,0x42,0x00,0x00,0x04,
      0x00,0x00,0x5b,0x01,0x2f,0x00,0x42,0x00,
      0x00,0x04,0x00,0x00,0x57,0x10,0x54,0x00,
      0x00,0x00,0x00,0x80,0xa8,0x75,0x00,0x00,
      0x57,0x20,0x54,0x08,0x00,0x10,0x00,0x90,
      0xa8,0x75,0x00,0x00,0x0e,0x00,0x00,0x00,
   };
   ok &= rgba8_mrt_compiled &&
         part.info.binary_size == sizeof(expected_rgba8_mrt) &&
         part.info.main_offset == 0 &&
         part.info.main_size == sizeof(expected_rgba8_mrt) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 16 &&
         memcmp(part.binary, expected_rgba8_mrt,
                sizeof(expected_rgba8_mrt)) == 0;
   if (rgba8_mrt_compiled)
      free(part.binary);
   ralloc_free(rgba8_mrt.shader);

   /* The key-selected mixed builtin ends here: the raw32 shapes below select
    * their program by format, and an unselected one must stay refused. */
   key.g17p_mrt_eot_count = 0;
   key.g17p_mrt_eot_selectors[0] = 0;
   key.g17p_mrt_eot_selectors[1] = 0;

   /* SuperTuxKart's deferred targets are two packed B10G11R11 images. The
    * tilebuffer carries each as one raw I32 word in partitions 0 and 4. */
   nir_builder r11_mrt = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P R11G11B10 MRT EOT");
   coords = nir_trim_vector(
      &r11_mrt, nir_load_global_invocation_id(&r11_mrt, 32), 2);
   nir_image_store_block_agx(
      &r11_mrt, nir_imm_intN_t(&r11_mrt, 0, 16),
      nir_imm_int(&r11_mrt, 0), coords, .format = PIPE_FORMAT_R32_UINT,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   nir_image_store_block_agx(
      &r11_mrt, nir_imm_intN_t(&r11_mrt, 1, 16),
      nir_imm_int(&r11_mrt, 4), coords, .format = PIPE_FORMAT_R32_UINT,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   agx_preprocess_nir(r11_mrt.shader);

   nir_shader *r11_ungated_shader = nir_shader_clone(NULL, r11_mrt.shader);
   memset(&part, 0xa5, sizeof(part));
   key.g17p_rt0_eot_i32 = true;
   key.g17p_rt0_eot_words = 1;
   const bool r11_ungated =
      agx_compile_shader_nir(r11_ungated_shader, &key, &part);
   const struct agx_shader_part empty_part = {0};
   ok &= !r11_ungated &&
         memcmp(&part, &empty_part, sizeof(part)) == 0;
   ralloc_free(r11_ungated_shader);

   key.g17p_r11g11b10_mrt_eot = true;
   memset(&part, 0, sizeof(part));
   const bool r11_mrt_compiled =
      agx_compile_shader_nir(r11_mrt.shader, &key, &part);
   uint8_t expected_r11_mrt[sizeof(expected_rgba8_mrt)];
   memcpy(expected_r11_mrt, expected_rgba8_mrt,
          sizeof(expected_rgba8_mrt));
   expected_r11_mrt[101] = 0x35;
   expected_r11_mrt[113] = 0x35;
   ok &= r11_mrt_compiled &&
         part.info.binary_size == sizeof(expected_r11_mrt) &&
         part.info.main_offset == 0 &&
         part.info.main_size == sizeof(expected_r11_mrt) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 16 &&
         memcmp(part.binary, expected_r11_mrt,
                sizeof(expected_r11_mrt)) == 0;
   if (r11_mrt_compiled)
      free(part.binary);
   ralloc_free(r11_mrt.shader);

   /* The gate does not admit a different partition or an unrelated mixed
    * raw/normalized pair; both failures leave no partial command text. */
   nir_builder wrong_r11_mrt = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "unsupported G17P raw MRT");
   coords = nir_trim_vector(
      &wrong_r11_mrt, nir_load_global_invocation_id(&wrong_r11_mrt, 32), 2);
   nir_image_store_block_agx(
      &wrong_r11_mrt, nir_imm_intN_t(&wrong_r11_mrt, 0, 16),
      nir_imm_int(&wrong_r11_mrt, 0), coords,
      .format = PIPE_FORMAT_R32_UINT, .image_dim = GLSL_SAMPLER_DIM_2D,
      .explicit_coord = true);
   nir_image_store_block_agx(
      &wrong_r11_mrt, nir_imm_intN_t(&wrong_r11_mrt, 1, 16),
      nir_imm_int(&wrong_r11_mrt, 8), coords,
      .format = PIPE_FORMAT_R32_UINT, .image_dim = GLSL_SAMPLER_DIM_2D,
      .explicit_coord = true);
   agx_preprocess_nir(wrong_r11_mrt.shader);
   memset(&part, 0xa5, sizeof(part));
   const bool wrong_r11_compiled =
      agx_compile_shader_nir(wrong_r11_mrt.shader, &key, &part);
   ok &= !wrong_r11_compiled &&
         memcmp(&part, &empty_part, sizeof(part)) == 0;
   if (wrong_r11_compiled)
      free(part.binary);
   ralloc_free(wrong_r11_mrt.shader);
   key.g17p_r11g11b10_mrt_eot = false;
   key.g17p_rt0_eot_i32 = false;
   key.g17p_rt0_eot_words = 0;

   /* Genuine applegpu-nt builtin EOT key0x04021100: scalar I32 tile store. */
   nir_builder i32 = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P I32 RT0 EOT");
   coords =
      nir_trim_vector(&i32, nir_load_global_invocation_id(&i32, 32), 2);
   nir_image_store_block_agx(
      &i32, nir_imm_intN_t(&i32, 0, 16), nir_imm_int(&i32, 0), coords,
      .format = PIPE_FORMAT_R32_UINT, .image_dim = GLSL_SAMPLER_DIM_2D,
      .explicit_coord = true);
   agx_preprocess_nir(i32.shader);
   memset(&part, 0, sizeof(part));
   key.g17p_rt0_eot_i32 = true;
   key.g17p_rt0_eot_words = 1;
   const bool i32_compiled = agx_compile_shader_nir(i32.shader, &key, &part);
   static const uint8_t expected_i32[] = {
      0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,0x0c,0x9d,0x10,0x26,
      0x24,0xa9,0x10,0x26,0x9f,0x10,0x54,0x00,0x00,0x04,0x14,0x00,
      0x50,0x06,0x0b,0x00,0x9f,0x20,0x54,0x02,0x00,0x00,0x10,0x00,
      0x50,0x06,0x0b,0x00,0x97,0x05,0x54,0x04,0x00,0x08,0x00,0x00,
      0xd0,0x40,0x10,0x00,0x57,0x10,0x54,0x00,0x00,0x00,0x00,0x80,
      0xa8,0x35,0x00,0x00,0x0e,0x00,0x00,0x00,
   };
   ok &= i32_compiled && part.info.binary_size == sizeof(expected_i32) &&
         part.info.main_offset == 0 &&
         part.info.main_size == sizeof(expected_i32) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 &&
         memcmp(part.binary, expected_i32, sizeof(expected_i32)) == 0;
   if (i32_compiled)
      free(part.binary);
   ralloc_free(i32.shader);

   /* Fresh Apple key0x10020b00: four I32 words deliberately return to the
    * scalar 0x35 selector instead of RG32_UINT's two-word 0x75 selector. */
   nir_builder i32x4 = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P RGBA32 UINT RT0 EOT");
   coords =
      nir_trim_vector(&i32x4, nir_load_global_invocation_id(&i32x4, 32), 2);
   nir_image_store_block_agx(
      &i32x4, nir_imm_intN_t(&i32x4, 0, 16), nir_imm_int(&i32x4, 0), coords,
      .format = PIPE_FORMAT_R32_UINT, .image_dim = GLSL_SAMPLER_DIM_2D,
      .explicit_coord = true);
   agx_preprocess_nir(i32x4.shader);
   memset(&part, 0, sizeof(part));
   key.g17p_rt0_eot_i32_four_words = true;
   key.g17p_rt0_eot_words = 4;
   const bool i32x4_compiled =
      agx_compile_shader_nir(i32x4.shader, &key, &part);
   ok &= i32x4_compiled && part.info.binary_size == sizeof(expected_i32) &&
         part.info.main_offset == 0 &&
         part.info.main_size == sizeof(expected_i32) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 &&
         memcmp(part.binary, expected_i32, sizeof(expected_i32)) == 0;
   if (i32x4_compiled)
      free(part.binary);
   ralloc_free(i32x4.shader);

   /* Fresh Apple key0x10021100: RGBA32 SFLOAT is the same exact four-word
    * I32 tile store and 0x35 program; the FLOAT conversion lives in PBE. */
   nir_builder i32x4_float = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P RGBA32 SFLOAT RT0 EOT");
   coords = nir_trim_vector(
      &i32x4_float, nir_load_global_invocation_id(&i32x4_float, 32), 2);
   nir_image_store_block_agx(
      &i32x4_float, nir_imm_intN_t(&i32x4_float, 0, 16),
      nir_imm_int(&i32x4_float, 0), coords, .format = PIPE_FORMAT_R32_UINT,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   agx_preprocess_nir(i32x4_float.shader);
   memset(&part, 0, sizeof(part));
   const bool i32x4_float_compiled =
      agx_compile_shader_nir(i32x4_float.shader, &key, &part);
   ok &= i32x4_float_compiled &&
         part.info.binary_size == sizeof(expected_i32) &&
         part.info.main_offset == 0 &&
         part.info.main_size == sizeof(expected_i32) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 &&
         memcmp(part.binary, expected_i32, sizeof(expected_i32)) == 0;
   if (i32x4_float_compiled)
      free(part.binary);
   ralloc_free(i32x4_float.shader);

   /* Fresh format10, components4, texture-type2 is byte-identical to UINT
    * and FLOAT. The old 84-byte fixture belongs to texture-type3, an array
    * store that reads a layer register even for a non-layered attachment. */
   nir_builder i32x4_signed = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P RGBA32 SINT RT0 EOT");
   coords = nir_trim_vector(
      &i32x4_signed, nir_load_global_invocation_id(&i32x4_signed, 32), 2);
   nir_image_store_block_agx(
      &i32x4_signed, nir_imm_intN_t(&i32x4_signed, 0, 16),
      nir_imm_int(&i32x4_signed, 0), coords, .format = PIPE_FORMAT_R32_UINT,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   agx_preprocess_nir(i32x4_signed.shader);
   memset(&part, 0, sizeof(part));
   key.g17p_rt0_eot_i32_four_words = false;
   key.g17p_rt0_eot_i32_signed_four_words = true;
   const bool i32x4_signed_compiled =
      agx_compile_shader_nir(i32x4_signed.shader, &key, &part);
   ok &= i32x4_signed_compiled &&
         part.info.binary_size == sizeof(expected_i32) &&
         part.info.main_offset == 0 &&
         part.info.main_size == sizeof(expected_i32) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 &&
         memcmp(part.binary, expected_i32, sizeof(expected_i32)) == 0;
   if (i32x4_signed_compiled)
      free(part.binary);
   ralloc_free(i32x4_signed.shader);
   key.g17p_rt0_eot_i32_signed_four_words = false;
   key.g17p_rt0_eot_i32_four_words = true;

   nir_builder bad_i32x4 = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "unsupported G17P RGBA32 EOT");
   coords = nir_trim_vector(
      &bad_i32x4, nir_load_global_invocation_id(&bad_i32x4, 32), 2);
   nir_image_store_block_agx(
      &bad_i32x4, nir_imm_intN_t(&bad_i32x4, 0, 16),
      nir_imm_int(&bad_i32x4, 0), coords, .format = PIPE_FORMAT_R32_UINT,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   agx_preprocess_nir(bad_i32x4.shader);
   memset(&part, 0xa5, sizeof(part));
   key.g17p_rt0_eot_words = 3;
   const bool bad_i32x4_compiled =
      agx_compile_shader_nir(bad_i32x4.shader, &key, &part);
   const struct agx_shader_part cleared_i32x4 = {0};
   if (bad_i32x4_compiled ||
       memcmp(&part, &cleared_i32x4, sizeof(part)) != 0) {
      fprintf(stderr, "unsupported G17P four-word I32 shape did not close\n");
      ok = false;
   }
   if (bad_i32x4_compiled)
      free(part.binary);
   ralloc_free(bad_i32x4.shader);
   key.g17p_rt0_eot_i32_four_words = false;
   key.g17p_rt0_eot_i32 = false;
   key.g17p_rt0_eot_words = 0;

   /* Fresh Apple builtin key0x10020900: an RGBA16_UINT attachment lowers to
    * exactly two I16 imageblock words and selects the native 0x15 EOT. */
   nir_builder i16 = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P RGBA16 UINT RT0 EOT");
   coords =
      nir_trim_vector(&i16, nir_load_global_invocation_id(&i16, 32), 2);
   nir_image_store_block_agx(
      &i16, nir_imm_intN_t(&i16, 0, 16), nir_imm_int(&i16, 0), coords,
      .format = PIPE_FORMAT_R16_UINT, .image_dim = GLSL_SAMPLER_DIM_2D,
      .explicit_coord = true);
   agx_preprocess_nir(i16.shader);
   memset(&part, 0, sizeof(part));
   key.g17p_rt0_eot_words = 2;
   const bool i16_compiled = agx_compile_shader_nir(i16.shader, &key, &part);
   static const uint8_t expected_i16[] = {
      0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,0x0c,0x9d,0x10,0x26,
      0x24,0xa9,0x10,0x26,0x9f,0x10,0x54,0x00,0x00,0x04,0x14,0x00,
      0x50,0x06,0x0b,0x00,0x9f,0x20,0x54,0x02,0x00,0x00,0x10,0x00,
      0x50,0x06,0x0b,0x00,0x97,0x05,0x54,0x04,0x00,0x08,0x00,0x00,
      0xd0,0x40,0x10,0x00,0x57,0x10,0x54,0x00,0x00,0x00,0x00,0x80,
      0xa8,0x15,0x00,0x00,0x0e,0x00,0x00,0x00,
   };
   ok &= i16_compiled && part.info.binary_size == sizeof(expected_i16) &&
         part.info.main_offset == 0 &&
         part.info.main_size == sizeof(expected_i16) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 &&
         memcmp(part.binary, expected_i16, sizeof(expected_i16)) == 0;
   if (i16_compiled)
      free(part.binary);
   ralloc_free(i16.shader);

   /* Packed A2B10G10R10_UINT key0x10021200 is the grounded one-word I16
    * sibling of the two-word RGBA16_UINT store. */
   nir_builder i16_one = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P RGB10A2 UINT RT0 EOT");
   coords = nir_trim_vector(
      &i16_one, nir_load_global_invocation_id(&i16_one, 32), 2);
   nir_image_store_block_agx(
      &i16_one, nir_imm_intN_t(&i16_one, 0, 16), nir_imm_int(&i16_one, 0),
      coords, .format = PIPE_FORMAT_R16_UINT,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   agx_preprocess_nir(i16_one.shader);
   memset(&part, 0, sizeof(part));
   key.g17p_rt0_eot_words = 1;
   const bool i16_one_compiled =
      agx_compile_shader_nir(i16_one.shader, &key, &part);
   ok &= i16_one_compiled && part.info.binary_size == sizeof(expected_i16) &&
         part.info.main_offset == 0 &&
         part.info.main_size == sizeof(expected_i16) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 &&
         memcmp(part.binary, expected_i16, sizeof(expected_i16)) == 0;
   if (i16_one_compiled)
      free(part.binary);
   ralloc_free(i16_one.shader);
   key.g17p_rt0_eot_words = 2;

   /* Signed RGBA16 uses the unsigned two-word program. Apple's 84-byte
    * program for 16-bit internal formats is the 2D-array builtin. */
   nir_builder i16_signed = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "G17P RGBA16 SINT RT0 EOT");
   coords = nir_trim_vector(
      &i16_signed, nir_load_global_invocation_id(&i16_signed, 32), 2);
   nir_image_store_block_agx(
      &i16_signed, nir_imm_intN_t(&i16_signed, 0, 16),
      nir_imm_int(&i16_signed, 0), coords, .format = PIPE_FORMAT_R16_UINT,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   agx_preprocess_nir(i16_signed.shader);
   memset(&part, 0, sizeof(part));
   const bool i16_signed_compiled =
      agx_compile_shader_nir(i16_signed.shader, &key, &part);
   ok &= i16_signed_compiled &&
         part.info.binary_size == sizeof(expected_i16) &&
         part.info.main_offset == 0 &&
         part.info.main_size == sizeof(expected_i16) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 &&
         memcmp(part.binary, expected_i16, sizeof(expected_i16)) == 0;
   if (i16_signed_compiled)
      free(part.binary);
   ralloc_free(i16_signed.shader);

   /* Do not let an ungrounded I16 word count inherit the same selector. */
   nir_builder bad_i16 = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "unsupported G17P I16 EOT");
   coords = nir_trim_vector(
      &bad_i16, nir_load_global_invocation_id(&bad_i16, 32), 2);
   nir_image_store_block_agx(
      &bad_i16, nir_imm_intN_t(&bad_i16, 0, 16), nir_imm_int(&bad_i16, 0),
      coords, .format = PIPE_FORMAT_R16_UINT,
      .image_dim = GLSL_SAMPLER_DIM_2D, .explicit_coord = true);
   agx_preprocess_nir(bad_i16.shader);
   memset(&part, 0xa5, sizeof(part));
   key.g17p_rt0_eot_words = 3;
   const bool bad_i16_compiled =
      agx_compile_shader_nir(bad_i16.shader, &key, &part);
   const struct agx_shader_part cleared_i16 = {0};
   if (bad_i16_compiled ||
       memcmp(&part, &cleared_i16, sizeof(part)) != 0) {
      fprintf(stderr, "unsupported G17P I16 word count did not fail closed\n");
      ok = false;
   }
   if (bad_i16_compiled)
      free(part.binary);
   ralloc_free(bad_i16.shader);
   key.g17p_rt0_eot_words = 0;

   /* A different store format must not inherit RT0's byte string. */
   nir_builder bad = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "unsupported G17P EOT");
   coords = nir_trim_vector(&bad, nir_load_global_invocation_id(&bad, 32), 2);
   nir_image_store_block_agx(
      &bad, nir_imm_intN_t(&bad, 0, 16), nir_imm_int(&bad, 0), coords,
      .format = PIPE_FORMAT_R32_UINT, .image_dim = GLSL_SAMPLER_DIM_2D,
      .explicit_coord = true);
   agx_preprocess_nir(bad.shader);
   memset(&part, 0xa5, sizeof(part));
   const bool bad_compiled = agx_compile_shader_nir(bad.shader, &key, &part);
   const bool rejected = !bad_compiled;
   const struct agx_shader_part cleared = {0};
   if (!rejected || memcmp(&part, &cleared, sizeof(part)) != 0) {
      fprintf(stderr, "unsupported G17P EOT format did not fail closed\n");
      ok = false;
   }
   if (bad_compiled)
      free(part.binary);
   ralloc_free(bad.shader);
   return ok;
}

/* Apple's builtin end-of-tile factory (metal-nt -arch applegpu_g17p, BTNF
 * texture type 3) emits the same 84-byte program for a 2D-array attachment of
 * every format, with the IMGWBLK lane-class selector at byte 77. Texture type
 * 8, a 2D multisample array, clears the selector's low nibble. */
static bool
test_g17p_layered_eot_block_store(void)
{
   static const struct {
      enum pipe_format physical;
      uint8_t words;
      bool i32;
      bool multisample;
      uint8_t selector;
   } cases[] = {
      {PIPE_FORMAT_R8_UNORM, 0, false, false, 0x75},
      {PIPE_FORMAT_R8_UNORM, 0, false, true, 0x70},
      {PIPE_FORMAT_R8_UINT, 1, false, false, 0x55},
      {PIPE_FORMAT_R16_FLOAT, 2, false, false, 0x15},
      {PIPE_FORMAT_R16_UNORM, 1, false, true, 0x90},
      {PIPE_FORMAT_R32_UINT, 1, true, false, 0x35},
   };
   static const uint8_t expected[] = {
      0x1c,0x9c,0x10,0x06,0x34,0xa8,0x10,0x06,
      0x0c,0x9d,0x10,0x26,0x24,0xa9,0x14,0x26,
      0x24,0xc3,0x10,0x46,
      0x9f,0x10,0x54,0x00,0x00,0x04,0x18,0x00,0x50,0x06,0x0b,0x00,
      0x9f,0x20,0x54,0x02,0x00,0x00,0x14,0x00,0x50,0x06,0x0b,0x00,
      0x97,0x45,0x54,0x04,0x00,0x08,0x10,0x00,0xf0,0xc0,0x00,0x00,
      0x97,0x05,0x54,0x06,0x00,0x08,0x00,0x00,0xd0,0x40,0x10,0x00,
      0x57,0x10,0x54,0x00,0x00,0x00,0x00,0x80,0x28,0x75,0x00,0x00,
      0x0e,0x00,0x00,0x00,
   };
   bool ok = sizeof(expected) == 84;

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "G17P layered RT0 EOT");
      nir_def *xy =
         nir_trim_vector(&b, nir_load_global_invocation_id(&b, 32), 2);
      nir_def *coords =
         nir_vec3(&b, nir_channel(&b, xy, 0), nir_channel(&b, xy, 1),
                  nir_u2u32(&b, nir_load_layer_id(&b)));
      nir_image_store_block_agx(
         &b, nir_imm_intN_t(&b, 0, 16), nir_imm_int(&b, 0), coords,
         .format = cases[i].physical, .image_dim = GLSL_SAMPLER_DIM_2D,
         .image_array = true, .explicit_coord = true);
      agx_preprocess_nir(b.shader);

      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .no_preamble = true,
         .g17p_rt0_eot_words = cases[i].words,
         .g17p_rt0_eot_i32 = cases[i].i32,
         .g17p_rt0_eot_multisample = cases[i].multisample,
      };
      struct agx_shader_part part = {0};
      const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
      uint8_t want[sizeof(expected)];
      memcpy(want, expected, sizeof(want));
      want[77] = cases[i].selector;

      const bool case_ok =
         compiled && part.info.binary_size == sizeof(want) &&
         part.info.nr_gprs == 8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 && part.info.main_offset == 0 &&
         part.info.main_size == 84 &&
         memcmp(part.binary, want, sizeof(want)) == 0;
      if (!case_ok) {
         fprintf(stderr,
                 "AGX3b layered EOT block-store compiler mismatch: format=%u "
                 "compiled=%u size=%u gprs=%u\n",
                 cases[i].physical, compiled,
                 compiled ? part.info.binary_size : 0,
                 compiled ? part.info.nr_gprs : 0);
         if (compiled) {
            for (unsigned j = 0; j < part.info.binary_size; ++j)
               fprintf(stderr, "%02x", ((const uint8_t *)part.binary)[j]);
            fputc('\n', stderr);
         }
      }
      ok &= case_ok;

      if (compiled)
         free(part.binary);
      ralloc_free(b.shader);
   }
   return ok;
}

/*
 * Fragment shader sampling texture 0 through sampler 0 with implicit LOD.
 * Keep both index ABIs explicit: ordinary bound sampling reserves sampler 0
 * for texel fetch, while G17P's compact pointer tables select native slot 0.
 * The compact reference is constructed directly at the post-lowering level;
 * it must not run the combined-sampler pass that it independently checks.
 */
static nir_shader *
build_fs_texture_index_abi(bool compact_native, bool gen9_bound)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 textured-triangle FS");

   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *uv = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   tex->src[0] = nir_tex_src_for_ssa(compact_native ? nir_tex_src_backend1
                                                 : nir_tex_src_coord,
                                     uv);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   fs_store_colour(&b, &tex->def);

   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_early, false);
   if (gen9_bound)
      NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
               AGX_DESCRIPTOR_GENERATION_9);
   else
      NIR_PASS(_, b.shader, agx_nir_lower_texture);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   return b.shader;
}

/* u_blitter's same-size depth blit is a fragment texel fetch whose level is
 * the interpolated texcoord.w, a per-lane register. Keep that shape in the
 * compiler oracle so its native staging sequence can be compared byte for
 * byte against Apple's `t.read(uint2(uv), uint(lod))` fragments. */
static nir_shader *
build_fs_texel_fetch_register_lod(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 texel fetch register LOD FS");

   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *in = nir_load_interpolated_input(
      &b, 3, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   /* The staging sequence under test is independent of the robustness
    * prologue, which needs a bindless descriptor pointer. */
   tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_f2i32(&b, nir_trim_vector(&b, in, 2)));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_lod, nir_f2i32(&b, nir_channel(&b, in, 2)));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   fs_store_colour(&b, &tex->def);

   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);

   NIR_PASS(_, b.shader, agx_nir_lower_texture_early, false);
   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   return b.shader;
}

static nir_shader *
build_fs_texture_imm(void)
{
   return build_fs_texture_index_abi(false, false);
}

/* Hyprland's ordinary surface path may sample the client texture and then a
 * second atlas/decoration texture with the same interpolated UV. Keep that
 * real two-AUTO fragment shape in the compiler oracle: native G17P clears the
 * sample-group marker on the first SMP and emits it only on the final SMP. */
/*
 * One mediump sampler2DShadow sampled three times at the same coordinates
 * with three references, which is KHR-GL30.texture_repeat_mode's depth
 * shader. Each sample owns a staging group, and its comparison reference
 * belongs to that group at bank + 4. Staged at a fixed half 4 instead, all
 * three samples read whichever reference was written last and every depth
 * repeat-mode case fails on hardware while the shader still compiles.
 */
static bool
test_g17p_three_shadow_sample_references(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &agx_nir_options,
                                     "G17P three shadow references");
   nir_def *bary =
      nir_load_barycentric_pixel(&b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *uv = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);
   static const float refs[3] = {0.3f, 0.5f, 0.8f};
   nir_def *chan[3];
   for (unsigned i = 0; i < 3; ++i) {
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
      tex->op = nir_texop_tex;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->dest_type = nir_type_float32;
      tex->coord_components = 2;
      tex->is_shadow = true;
      tex->is_new_style_shadow = true;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1, uv);
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                        nir_imm_float(&b, refs[i]));
      nir_def_init(&tex->instr, &tex->def, 1, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      chan[i] = &tex->def;
   }
   fs_store_colour(&b, nir_vec4(&b, chan[0], chan[1], chan[2],
                                nir_imm_float(&b, 1.0f)));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   b.shader->info.fs.needs_coarse_quad_helper_invocations = true;
   NIR_PASS(_, b.shader, agx_nir_lower_texture_early, false);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(b.shader, &key, &part) && part.binary;

   /* Three distinct reference COPYs, one per staging group. Byte 0 carries
    * the destination half as (half / 2) << 4, so halves 4, 12 and 20 are
    * 0x2b, 0x6b and 0xab. Byte 2 is the COPY's operand form -- 0x29 compact
    * or 0x2f register-source, both of which the references use depending on
    * where the allocator put each value. Coordinates occupy the even halves
    * of each group and never carry these destination bytes. Before the fix
    * the second and third samples both wrote 0x2b, so one reference served
    * all three compares. */
   unsigned found = 0;
   for (unsigned group = 0; ok && group < 3; ++group) {
      const uint8_t dest_byte = (uint8_t)(((8 * group + 4) / 2) << 4 | 0x0b);
      for (unsigned i = 0; part.binary && i + 3 <= part.info.binary_size; ++i) {
         const uint8_t *at = (const uint8_t *)part.binary + i;
         if (at[0] == dest_byte && (at[2] == 0x29 || at[2] == 0x2f)) {
            found++;
            break;
         }
      }
   }
   ok &= found == 3 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 22;
   if (!ok) {
      fprintf(stderr,
              "G17P shadow references not per-group: found %u of 3, "
              "staging %u halfs\n",
              found, part.info.agx3_staging_halfs);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return ok;
}

static nir_shader *
build_fs_two_texture_samples(void)
{
   nir_builder b =
      nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &agx_nir_options,
                                     "AGX3 Hyprland two-texture surface FS");
   nir_def *bary =
      nir_load_barycentric_pixel(&b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *uv = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);
   nir_def *colour[2];

   for (unsigned i = 0; i < 2; ++i) {
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
      tex->op = nir_texop_tex;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->dest_type = nir_type_float32;
      tex->coord_components = 2;
      tex->texture_index = i;
      tex->sampler_index = i;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1, uv);
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      colour[i] = &tex->def;
   }

   fs_store_colour(&b, nir_fadd(&b, colour[0], colour[1]));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   b.shader->info.fs.needs_coarse_quad_helper_invocations = true;

   NIR_PASS(_, b.shader, agx_nir_lower_texture_early, false);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   return b.shader;
}

/* A18 Pro's v_texture vertex pipeline samples texture0/sampler0 at explicit
 * level zero and emits the same 14-byte SMP as the shared t_base witness:
 *
 *    05800cb8b0000900000010000100
 *
 * Keep a real vertex-stage compile in the corpus so txl(0) cannot regress to
 * the fragment-only implicit-derivative selector. */
static nir_shader *
build_vs_texture_level(bool register_lod)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "AGX3 vertex texture level-zero VS");

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_vec2(&b, 0.5f, 0.5f));
   nir_def *lod = register_lod
                    ? nir_u2f32(&b, nir_iand_imm(
                         &b, nir_load_vertex_id(&b), 3))
                    : nir_imm_float(&b, 0.0f);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   for (unsigned c = 0; c < 4; ++c) {
      nir_store_uvs_agx(&b, c == 3 ? nir_imm_float(&b, 1.0f)
                                   : nir_imm_float(&b, 0.0f),
                        nir_imm_intN_t(&b, c, 16));
      nir_store_uvs_agx(&b, nir_channel(&b, &tex->def, c),
                        nir_imm_intN_t(&b, 4 + c, 16));
   }

   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);
   NIR_PASS(_, b.shader, agx_nir_lower_texture_early, false);
   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_9);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static nir_shader *
build_vs_texture_level_zero(void)
{
   return build_vs_texture_level(false);
}

/* Apple's G17P vertex level(float(vertex_id & 3)) witness stages the
 * per-vertex LOD with PACK_SMPIR before SMP. Keep the LOD out of the constant
 * preamble so this exercises the register operand accepted by the encoder. */
static nir_shader *
build_vs_texture_register_lod(void)
{
   return build_vs_texture_level(true);
}

/* Firefox WebRender uses texelFetch from a dynamically indexed descriptor in
 * vertex shaders. Lower the Vulkan descriptor to the native G17P heap index,
 * then compile the exact explicit-level read plus terminal UVS outputs. */
static nir_shader *
build_vs_indexed_texture_fetch(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options,
      "AGX3 indexed vertex texel fetch");
   nir_def *handle =
      nir_bindless_image_agx(&b, nir_imm_int(&b, 72), .desc_set = 3);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_backend1, nir_imm_ivec2(&b, 0, 0));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_lod, nir_imm_intN_t(&b, 0, 16));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   for (unsigned c = 0; c < 4; ++c) {
      nir_store_uvs_agx(&b, c == 3 ? nir_imm_float(&b, 1.0f)
                                   : nir_imm_float(&b, 0.0f),
                        nir_imm_intN_t(&b, c, 16));
      nir_store_uvs_agx(&b, nir_channel(&b, &tex->def, c),
                        nir_imm_intN_t(&b, 4 + c, 16));
   }
   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);

   if (!agx_nir_lower_g17p_indexed_samplers(b.shader, true)) {
      ralloc_free(b.shader);
      return NULL;
   }
   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
            lower_test_descriptor_set, nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static nir_shader *
build_vs_indexed_texture_sample(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options,
      "AGX3 indexed vertex texture sample");
   nir_def *offset = nir_imm_int(&b, 72);
   nir_def *image = nir_bindless_image_agx(&b, offset, .desc_set = 3);
   nir_def *sampler = nir_bindless_sampler_agx(
      &b, nir_imm_int(&b, 104), nir_imm_intN_t(&b, 0, 16), .desc_set = 3);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_backend1, nir_imm_vec2(&b, 0.5f, 0.5f));
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, image);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
   tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_lod,
                                     nir_imm_float(&b, 0.0f));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   for (unsigned c = 0; c < 4; ++c) {
      nir_store_uvs_agx(&b, c == 3 ? nir_imm_float(&b, 1.0f)
                                   : nir_imm_float(&b, 0.0f),
                        nir_imm_intN_t(&b, c, 16));
      nir_store_uvs_agx(&b, nir_channel(&b, &tex->def, c),
                        nir_imm_intN_t(&b, 4 + c, 16));
   }
   b.shader->info.outputs_written =
      VARYING_BIT_POS | BITFIELD64_BIT(VARYING_SLOT_VAR0);

   if (!agx_nir_lower_g17p_indexed_samplers(b.shader, true)) {
      ralloc_free(b.shader);
      return NULL;
   }
   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass,
            lower_test_descriptor_set, nir_metadata_control_flow, NULL);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static nir_shader *
build_fs_texture_compact_native(void)
{
   return build_fs_texture_index_abi(true, false);
}

static nir_shader *
build_fs_texture_3d_level(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 texture3d level FS");
   nir_def *coord = nir_imm_vec3(&b, 0.5f, 0.5f, 0.625f);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 2);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_3D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 3;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod,
                                     nir_imm_float(&b, 2.0f));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   fs_store_colour(&b, &tex->def);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   return b.shader;
}

static nir_shader *
build_fs_texture_3d_implicit(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "AGX3 Hyprland ICC texture3d implicit FS");
   nir_def *coord = nir_imm_vec3(&b, 0.5f, 0.5f, 0.625f);
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_3D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 3;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1, coord);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   fs_store_colour(&b, &tex->def);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   return b.shader;
}

static bool
has_compact_native_sample(const nir_shader *shader)
{
   unsigned samples = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic) {
               const nir_intrinsic_op op =
                  nir_instr_as_intrinsic(instr)->intrinsic;
               if (op == nir_intrinsic_bindless_image_agx ||
                   op == nir_intrinsic_bindless_sampler_agx ||
                   op == nir_intrinsic_load_descriptor_set_agx ||
                   op == nir_intrinsic_load_root_agx)
                  return false;
            }

            if (instr->type != nir_instr_type_tex)
               continue;

            const nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (++samples != 1 || tex->op != nir_texop_tex ||
                tex->texture_index != 0 || tex->sampler_index != 0 ||
                tex->num_srcs != 1 ||
                tex->src[0].src_type != nir_tex_src_backend1 ||
                tex->sampler_dim != GLSL_SAMPLER_DIM_2D || tex->is_array ||
                tex->is_shadow || tex->is_sparse || tex->backend_flags != 0 ||
                tex->coord_components != 2 ||
                nir_src_num_components(tex->src[0].src) != 2 ||
                nir_src_bit_size(tex->src[0].src) != 32 ||
                (tex->dest_type != nir_type_float32 &&
                 tex->dest_type != nir_type_uint32 &&
                 tex->dest_type != nir_type_int32) ||
                tex->def.num_components != 4 || tex->def.bit_size != 32)
               return false;
         }
      }
   }
   return samples == 1;
}

static bool
test_texture_index_abis(void)
{
   nir_shader *legacy = build_fs_texture_imm();
   nir_shader *g17p_bound = build_fs_texture_index_abi(false, true);
   nir_shader *native = build_fs_texture_compact_native();
   unsigned bound_samples = 0;
   bool bound_ok = true;

   nir_shader *bound[] = {legacy, g17p_bound};
   for (unsigned i = 0; i < ARRAY_SIZE(bound); ++i) {
      nir_foreach_function_impl(impl, bound[i]) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_tex)
                  continue;

               const nir_tex_instr *tex = nir_instr_as_tex(instr);
               const int sampler =
                  nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
               ++bound_samples;
               bound_ok &= sampler >= 0 &&
                           nir_src_is_const(tex->src[sampler].src) &&
                           nir_src_as_uint(tex->src[sampler].src) == 1;
            }
         }
      }
   }

   bool ok = bound_ok && bound_samples == ARRAY_SIZE(bound) &&
             !has_compact_native_sample(legacy) &&
             !has_compact_native_sample(g17p_bound) &&
             has_compact_native_sample(native);

   /* The old specialized result had no handles, but still selected sampler
    * 1. Reject that exact regression, independently of the source count. */
   nir_foreach_function_impl(impl, native) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               nir_instr_as_tex(instr)->sampler_index = 1;
         }
      }
   }
   ok &= !has_compact_native_sample(native);
   ralloc_free(legacy);
   ralloc_free(g17p_bound);
   ralloc_free(native);
   return ok;
}

/* Descriptor set 0 is placed in fragment uniform halfs 16..19. Load one
 * vec4 directly from that descriptor resource and write it to RT0. */
static nir_shader *
build_fs_descriptor_resource(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 descriptor resource FS");

   nir_def *set = nir_load_preamble(&b, 1, 64, .base = 16);
   nir_def *raw = nir_load_agx(&b, 4, 32, set, nir_imm_int(&b, 0),
                               .access = ACCESS_CAN_SPECULATE, .base = 0,
                               .format = PIPE_FORMAT_R32_UINT,
                               .sign_extend = false);
   fs_store_colour(&b, nir_u2f32(&b, raw));
   return fs_finish(&b);
}

/* The bounded buffer-to-image meta fragment shape: use the root uniform to
 * fetch a staging-buffer address, index it by pixel coordinate, unpack one
 * RGBA8 texel, and write RT0.
 */
static nir_shader *
build_fs_meta_rgba_upload(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 RGBA upload FS");

   nir_def *root =
      nir_load_preamble(&b, 1, 64, .base = AGX_ABI_FUNI_ROOT);
   nir_def *source = nir_load_global_constant(
      &b, 1, 64, root, .align_mul = 8, .access = ACCESS_CAN_SPECULATE);
   nir_def *xy = nir_u2u32(&b, nir_load_pixel_coord(&b));
   nir_def *pixel = nir_iadd(
      &b, nir_channel(&b, xy, 0),
      nir_imul_imm(&b, nir_channel(&b, xy, 1), 256));
   nir_def *address =
      nir_iadd(&b, source, nir_u2u64(&b, nir_imul_imm(&b, pixel, 4)));
   nir_def *packed =
      nir_load_global(&b, 1, 32, address, .align_mul = 4);
   nir_def *channel[4];
   for (unsigned i = 0; i < ARRAY_SIZE(channel); ++i) {
      const unsigned shift = i * 8;
      nir_def *masked =
         nir_iand(&b, packed, nir_imm_int(&b, 0xffu << shift));
      channel[i] = nir_fmul_imm(
         &b, nir_u2f32(&b, masked),
         1.0f / (255.0f * (float)(1u << shift)));
   }
   nir_def *colour =
      nir_vec4(&b, channel[0], channel[1], channel[2], channel[3]);

   fs_store_colour(&b, colour);
   return fs_finish(&b);
}

/*
 * Fragment shader exercising the funop family: floor/exp2/log2 (f32rint and
 * friends) plus the fragment-only gradient dfdx.
 */
static nir_shader *
build_fs_native_trig_precision(bool half)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P native trig FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *x = nir_load_interpolated_input(
      &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);
   if (half)
      x = nir_f2f16(&b, x);
   nir_def *sine = nir_fsin(&b, x);
   nir_def *cosine = nir_fcos(&b, x);
   if (half) {
      sine = nir_f2f32(&b, sine);
      cosine = nir_f2f32(&b, cosine);
   }
   fs_store_colour(&b, nir_vec4(&b, sine, cosine,
                               nir_fmul(&b, sine, nir_frcp(&b, cosine)),
                               nir_imm_float(&b, 1)));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_preprocess_nir_g17p(b.shader);
   unsigned fmas = 0, sincs = 0, selects = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_ffma) {
               ++fmas;
               assert(nir_alu_instr_is_exact(alu));
            }
            sincs += alu->op == nir_op_fsinc_agx3;
            selects += alu->op == nir_op_bittest_select_agx3;
         }
      }
   }
   assert(fmas == 4 && sincs == 2 && selects == 4);
   fs_lower(b.shader);
   return b.shader;
}

static nir_shader *
build_fs_native_trig(void)
{
   return build_fs_native_trig_precision(false);
}

static nir_shader *
build_fs_native_trig_f16(void)
{
   return build_fs_native_trig_precision(true);
}

static nir_shader *
build_fs_funops(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 funop FS");

   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *v = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);

   nir_def *x = nir_channel(&b, v, 0);
   nir_def *y = nir_channel(&b, v, 1);

   fs_store_colour(&b, nir_vec4(&b, nir_ffloor(&b, x), nir_fexp2(&b, y),
                                nir_flog2(&b, nir_fabs(&b, x)),
                                nir_ddx(&b, y)));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* Quickshell's rounded-rectangle fragment path computes abs(delta) * 2.0
 * before clamping against a live radius. Keep that application-derived
 * modifier/immediate shape in the full graphics compiler, not just the
 * low-level encoder test. */
static nir_shader *
build_fs_quickshell_abs_scale(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 Quickshell abs-scale FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *v = nir_load_interpolated_input(
      &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *scaled = nir_fmul_imm(&b, nir_fabs(&b, v), 2.0f);
   nir_def *clamped = nir_fmin(&b, scaled, nir_imm_float(&b, 0.67f));
   fs_store_colour(&b, nir_vec4(&b, clamped, scaled, v,
                                nir_imm_float(&b, 1.0f)));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* Hyprland's colour-management matrix accumulates a mediump term into a
 * highp result. The backend must widen the half input before the plain f32
 * add instead of asking the packer for an ungrounded mixed-width FADD. */
static nir_shader *
build_fs_hyprland_cm_mixed_add(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 Hyprland CM mixed-add FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *input = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *x = nir_channel(&b, input, 0);
   nir_def *y = nir_channel(&b, input, 2);
   nir_def *half = nir_f2f16(&b, nir_channel(&b, input, 1));
   nir_def *sum = nir_fadd(&b, x, nir_f2f32(&b, half));
   nir_def *mixed_compare = nir_bcsel(
      &b, nir_fge(&b, half, nir_imm_floatN_t(&b, 0.5, 16)), x, y);
   nir_def *absolute_compare = nir_bcsel(
      &b, nir_fge(&b, x, nir_fabs(&b, y)),
      nir_channel(&b, input, 1), nir_channel(&b, input, 3));
   nir_def *register_or_zero = nir_bcsel(
      &b, nir_flt(&b, half, nir_imm_floatN_t(&b, 0.0, 16)), x,
      nir_imm_float(&b, 0.0));
   fs_store_colour(
      &b, nir_vec4(&b, sum, mixed_compare, absolute_compare,
                   register_or_zero));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* Hyprland's optional discard path turns a highp alpha comparison against
 * 1.0 into a mediump 1/0 mask. Materialize the compare constant so the
 * grounded register-comparison/byte-selection endpoint is used. */
static nir_shader *
build_fs_hyprland_discard_predicate(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "AGX3 Hyprland discard predicate FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *alpha = nir_load_interpolated_input(
      &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_discard_if(&b, nir_feq(&b, alpha, nir_imm_float(&b, 1.0)));
   fs_store_colour(&b, nir_vec4(&b, alpha, alpha, alpha, alpha));
   b.shader->info.fs.uses_discard = true;
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* Zink's dynamic fragment path uses a minifloat multiplier and an absolute
 * addend. Cover both multiplier positions and abs/-abs while retaining a
 * fused FMA rather than splitting its arithmetic into two rounded steps. */
static nir_shader *
build_fs_zink_minifloat_abs_fma(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "AGX3 Zink minifloat absolute-addend FMA FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *input = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *x = nir_channel(&b, input, 0);
   nir_def *y = nir_channel(&b, input, 1);
   nir_def *absolute = nir_fabs(&b, y);
   nir_def *constant = nir_imm_float(&b, -2.0f);
   nir_def *a = nir_ffma(&b, constant, x, absolute);
   nir_def *n = nir_ffma(&b, x, constant, nir_fneg(&b, absolute));
   fs_store_colour(&b, nir_vec4(&b, a, n, x, nir_imm_float(&b, 1.0f)));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* Chromium/Skia's rounded-rect fragment compares a negated highp value for
 * equality and selects two unrelated highp values. The backend must
 * materialize fneg before RA and reach the grounded unmodified FSEL. */
static nir_shader *
build_fs_chromium_negated_equality_select(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "AGX3 Chromium negated equality select FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *input = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *x = nir_channel(&b, input, 0);
   nir_def *y = nir_channel(&b, input, 1);
   nir_def *t = nir_channel(&b, input, 2);
   nir_def *f = nir_channel(&b, input, 3);
   nir_def *selected = nir_bcsel(&b, nir_feq(&b, nir_fneg(&b, x), y), t, f);
   fs_store_colour(&b, nir_vec4(&b, selected, x, y,
                                nir_imm_float(&b, 1.0f)));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* Chromium's three.js skinning example keeps a real branch around a
 * minifloat threshold.  The backend must materialize the nonzero immediate
 * before register allocation so the fused IF_FCMP reaches the grounded
 * register/register control-flow comparison. */
static nir_shader *
build_fs_chromium_minifloat_branch(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "AGX3 Chromium minifloat branch FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *input = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);

   nir_def *x = nir_channel(&b, input, 0);
   nir_push_if(&b, nir_flt(&b, nir_imm_float(&b, 0.5f), x));
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1, input);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *then_value = &tex->def;
   nir_push_else(&b, NULL);
   nir_def *else_value = nir_vec4(&b, x, nir_channel(&b, input, 1), x,
                                  nir_imm_float(&b, 1.0f));
   nir_pop_if(&b, NULL);

   nir_def *selected = nir_if_phi(&b, then_value, else_value);
   fs_store_colour(&b, selected);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   NIR_PASS(_, b.shader, agx_nir_lower_texture_early, false);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   return b.shader;
}

/* dEQP's varying-struct comparison keeps a real branch on x >= abs(y).
 * G17P's direct ABS branch lacks the unordered selector required to express
 * ordered GE, so the compiler must materialize fabs and use the ordinary
 * register/register control-flow compare. */
static nir_shader *
build_fs_ordered_abs_branch(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "AGX3 ordered absolute-value branch FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *input = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *x = nir_channel(&b, input, 0);
   nir_def *y = nir_channel(&b, input, 1);

   nir_push_if(&b, nir_fge(&b, x, nir_fabs(&b, y)));
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->dest_type = nir_type_float32;
   tex->coord_components = 2;
   tex->texture_index = 0;
   tex->sampler_index = 0;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1, input);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *then_value = &tex->def;
   nir_push_else(&b, NULL);
   nir_def *else_value =
      nir_vec4(&b, x, y, x, nir_imm_float(&b, 1.0f));
   nir_pop_if(&b, NULL);

   fs_store_colour(&b, nir_if_phi(&b, then_value, else_value));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   NIR_PASS(_, b.shader, agx_nir_lower_texture_early, false);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   agx_preprocess_nir(b.shader);
   fs_lower(b.shader);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   return b.shader;
}

/* Hyprland clamps the motion-blur loop bound through a float comparison that
 * selects a raw 16-bit one or a retained half register. Both constants must
 * be materialized before using the general half-register FSEL. */
static nir_shader *
build_fs_hyprland_motion_bound(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "AGX3 Hyprland motion bound FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *input = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *value = nir_channel(&b, input, 0);
   nir_def *half = nir_f2f16(&b, nir_channel(&b, input, 1));
   nir_def *bound = nir_bcsel(
      &b, nir_flt(&b, nir_imm_float(&b, 1.0), value),
      nir_imm_intN_t(&b, 1, 16), half);
   nir_def *wide = nir_f2f32(&b, bound);
   fs_store_colour(&b, nir_vec4(&b, wide, value, wide, value));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* glmark2's buffer scene combines both half-ALU shapes that used to fail
 * closed: exp2 of a negated mediump value followed by x * 0.5h + 0.5h. */
static nir_shader *
build_fs_glmark2_buffer(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 glmark2 buffer FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *v32 = nir_load_interpolated_input(
      &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *v = nir_f2f16(&b, v32);
   nir_def *square = nir_fmul(&b, v, v);
   nir_def *fog = nir_fexp2(&b, nir_fneg(&b, square));
   nir_def *mapped =
      nir_ffma(&b, fog, nir_imm_floatN_t(&b, 0.5, 16),
               nir_imm_floatN_t(&b, 0.5, 16));
   fs_store_colour(&b, nir_vec4(&b, fog, mapped, mapped, mapped));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* glmark2's control-flow scenes retain a mediump input across floor() and
 * then subtract the rounded value. This exercises both the optional cached
 * floor source and the exact half-subtract legalization used by those scenes. */
static nir_shader *
build_fs_glmark2_control(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 glmark2 control FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *input = nir_load_interpolated_input(
      &b, 3, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *v = nir_f2f16(&b, nir_channel(&b, input, 0));
   nir_def *rounded = nir_ffloor(&b, v);
   nir_def *fraction = nir_fsub(&b, v, rounded);
   nir_def *choice = nir_bcsel(
      &b, nir_fge(&b, v, nir_imm_floatN_t(&b, 0.5, 16)),
      nir_f2f16(&b, nir_channel(&b, input, 1)),
      nir_f2f16(&b, nir_channel(&b, input, 2)));
   fs_store_colour(&b, nir_vec4(&b, v, rounded, fraction, choice));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* glmark2's desktop blur offsets a retained mediump coordinate by a negative
 * minifloat. The general half add must materialize that constant without
 * disturbing the two independently grounded immediate-special cases. */
static nir_shader *
build_fs_glmark2_desktop(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 glmark2 desktop FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *v32 = nir_load_interpolated_input(
      &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *v = nir_f2f16(&b, v32);
   nir_def *offset =
      nir_fadd(&b, v, nir_imm_floatN_t(&b, -0.15625, 16));
   fs_store_colour(&b, nir_vec4(&b, v, offset, offset, offset));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* glmark2's bump fragment clamps a mediump lighting term with max(x, 0.0h).
 * The tied immediate must become one shared half register so the recovered
 * native scalar-half maximum remains applicable after register allocation. */
static nir_shader *
build_fs_glmark2_bump(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 glmark2 bump FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *v32 = nir_load_interpolated_input(
      &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *v = nir_f2f16(&b, v32);
   nir_def *clamped = nir_fmax(&b, v, nir_imm_floatN_t(&b, 0.0, 16));
   fs_store_colour(&b, nir_vec4(&b, v, clamped, clamped, clamped));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* ideas' logo fragment converts a mediump equality test into a half 1/0
 * value. This exercises the explicit half-to-f32 comparator normalization
 * feeding G17P's byte-constant boolean FSEL. */
static nir_shader *
build_fs_glmark2_ideas(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 glmark2 ideas FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *v32 = nir_load_interpolated_input(
      &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *v = nir_f2f16(&b, v32);
   nir_def *mask = nir_bcsel(
      &b, nir_feq(&b, v, nir_imm_floatN_t(&b, 0.0, 16)),
      nir_imm_floatN_t(&b, 1.0, 16), nir_imm_floatN_t(&b, 0.0, 16));
   fs_store_colour(&b, nir_vec4(&b, v, mask, mask, mask));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* refract computes 1.0h - x*x as a fused half operation. The negated
 * multiplier requires the general product-negate FMA, with its immediate
 * addend materialized as an exact half register. */
static nir_shader *
build_fs_glmark2_refract(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 glmark2 refract FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *v32 = nir_load_interpolated_input(
      &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *v = nir_f2f16(&b, v32);
   nir_def *one_minus_square = nir_ffma(
      &b, nir_fneg(&b, v), v, nir_imm_floatN_t(&b, 1.0, 16));
   fs_store_colour(
      &b, nir_vec4(&b, v, one_minus_square, one_minus_square,
                   one_minus_square));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* terrain's blur and procedural-noise variants form mediump offsets such as
 * 0.5h - x. Constant materialization must expose this as the grounded
 * register FMA subtraction form before register allocation. */
static nir_shader *
build_fs_glmark2_terrain_offset(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 glmark2 terrain offset FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *v32 = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *v = nir_f2f16(&b, nir_channel(&b, v32, 0));
   nir_def *w = nir_f2f16(&b, nir_channel(&b, v32, 1));
   nir_def *offset = nir_fadd(
      &b, nir_imm_floatN_t(&b, 0.5, 16), nir_fneg(&b, v));
   nir_def *absolute_fma = nir_ffma(
      &b, v, nir_fabs(&b, w), nir_imm_floatN_t(&b, 0.25, 16));
   nir_def *negative_sum =
      nir_fadd(&b, nir_fneg(&b, v), nir_fneg(&b, w));
   nir_def *threshold = nir_bcsel(
      &b, nir_fge(&b, v, w), negative_sum, nir_imm_floatN_t(&b, 0.0, 16));
   fs_store_colour(
      &b, nir_vec4(&b, offset, absolute_fma, negative_sum, threshold));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

/* terrain's procedural-noise kernel forms `1.0 - abs(x) - abs(y)` at full
 * precision, which reaches the backend as an f32 FMA whose multiplier is a
 * minifloat immediate and whose other two operands are absolute registers --
 * an immediate operand and a source modifier on one encoding point at once.
 * That combination used to have no home in either the immediate packer or the
 * modifier packer, so it refused the whole fragment shader; see
 * agx3_falu_fma_mul_imm_mod() for Apple's bytes. */
static nir_shader *
build_fs_glmark2_terrain_noise(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "AGX3 glmark2 terrain noise FS");
   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *v32 = nir_load_interpolated_input(
      &b, 2, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0,
      .io_semantics.num_slots = 1);
   nir_def *x = nir_channel(&b, v32, 0);
   nir_def *y = nir_channel(&b, v32, 1);

   /* h = 1.0 - |x| - |y| */
   nir_def *h = nir_fsub(&b, nir_fsub(&b, nir_imm_float(&b, 1.0),
                                      nir_fabs(&b, x)),
                         nir_fabs(&b, y));
   /* half * |x| + |y|: the immediate multiplier standing beside two absolute
    * register operands, exactly as the refusal printed it. */
   nir_def *scaled = nir_ffma(&b, nir_imm_float(&b, 0.5), nir_fabs(&b, x),
                              nir_fabs(&b, y));
   /* a negated immediate multiplier folds into the minifloat's sign bit */
   nir_def *negated = nir_ffma(&b, nir_imm_float(&b, -0.5), x,
                               nir_fabs(&b, y));
   /* and the addend's own negation rides the f65 bit */
   nir_def *sub = nir_ffma(&b, nir_imm_float(&b, 0.25), nir_fabs(&b, x),
                           nir_fneg(&b, nir_fabs(&b, y)));

   fs_store_colour(&b, nir_vec4(&b, h, scaled, negated, sub));
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   return fs_finish(&b);
}

struct agx3_graphics_case {
   const char *name;
   nir_shader *(*build)(void);
   bool expect_ok;
   bool lower_combined_sampler;
   uint16_t reserved_preamble;
   uint16_t minimum_binary_size;
   uint8_t expected_leading_sr;
   const uint8_t *expected_g17p_binary;
   size_t expected_g17p_binary_size;
   bool compare_apple_g17p_vs;
   uint8_t expected_g17p_loads;
   bool g17p_only;
   uint8_t expected_g17p_depth;
   uint8_t expected_g17p_stencil;
   bool expected_g17p_implicit_coord_group;
   bool expected_g17p_sampled_fragment;
   bool expected_g17p_sample_mask_input;
   bool expected_g17p_explicit_sample;
   bool expected_g17p_centroid;
   bool expected_g17p_sample_group;
   bool expected_g17p_balanced_coverage;
   uint8_t expected_g17p_sincs;
   uint8_t expected_g17p_sample_zs_writes;
   uint8_t nr_samples;
};

static bool
binary_contains(const struct agx_shader_part *part, const uint8_t *needle,
                unsigned length)
{
   for (unsigned off = 0; off + length <= part->info.binary_size; ++off) {
      if (!memcmp((const uint8_t *)part->binary + off, needle, length))
         return true;
   }

   return false;
}

static bool
g17p_sample_group_protocol(const struct agx_shader_part *part)
{
   bool first = false, incorrectly_terminated_first = false, final = false;
   uint8_t candidate[AGX3_TEX_MAX_LENGTH], length;

   for (unsigned dest = 0; dest < 128; dest += 2) {
      for (unsigned slot = 1; slot <= 6; ++slot) {
         for (unsigned wait_case = 0; wait_case <= 6; ++wait_case) {
            for (unsigned kill = 0; kill < 2; ++kill) {
               struct agx3_tex_sample_desc d = {
                  .dest = dest,
                  .coord = 0,
                  .width = AGX3_TEX_W32,
                  .mask = 15,
                  .lod = AGX3_TEX_LOD_AUTO,
                  .slot = slot,
                  .wait_mask = wait_case ? 1u << (wait_case - 1) : 0,
                  .coord_kill = kill,
                  .sample_group_nonterminal = true,
               };
               if (!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d,
                                           candidate, &length))
                  return false;
               first |= binary_contains(part, candidate, length);

               d.sample_group_nonterminal = false;
               if (!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d,
                                           candidate, &length))
                  return false;
               incorrectly_terminated_first |=
                  binary_contains(part, candidate, length);

               /* Both samples consume the same published coordinates.
                * The packer shares that immutable publication in bank 0;
                * no second staging write needs an independent bank. */
               d.texture = 1;
               d.sampler = 1;
               d.coord = 0;
               d.final = true;
               d.sample_group_end = true;
               if (!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d,
                                           candidate, &length))
                  return false;
               final |= binary_contains(part, candidate, length);
            }
         }
      }
   }

   if (!first || !final || incorrectly_terminated_first)
      fprintf(stderr, "SMP group: first=%u final=%u premature=%u\n",
              first, final, incorrectly_terminated_first);
   return first && final && !incorrectly_terminated_first;
}

static bool
g17p_implicit_coord_group(const struct agx_shader_part *part)
{
   /* Apple's G17P implicit sample orders its front end as RHW, reciprocal,
    * both user interpolants, both direct coordinate publications, PIXWAIT.
    * Its first F32INTERP leaves f63 clear for the sampled-fragment helper
    * protocol. Physical user registers are allocator-specific; this exact
    * Mesa prefix pins both that policy and the native transaction boundary. */
   static const uint8_t expected[] = {
      0x2f,0x05,0x54,0x00,0x03,0x00,0x04,0x02,0x10,0x00,
      0xaf,0x00,0x54,0x00,0x03,0x00,0x00,0x48,0x20,0x01,
      0x2f,0x05,0x54,0x02,0x03,0x02,0x00,0x02,0x10,0x00,
      0x2f,0x05,0x54,0x04,0x03,0x04,0x00,0x02,0x10,0x00,
      0x09,0x03,0x2f,0x01,0x00,0x01,0x40,0x00,
      0x19,0x05,0x3f,0x01,0x00,0x02,0x40,0x00,
      0x87,0x02,0x54,0x00,0x06,0x00,
   };

   return part->info.binary_size >= sizeof(expected) &&
          !memcmp(part->binary, expected, sizeof(expected));
}

static bool
g17p_explicit_sample_protocol(const struct agx_shader_part *part)
{
   const uint8_t *code = part->binary;
   const unsigned size = part->info.binary_size;

   for (unsigned i = 0; i + 8 <= size; ++i) {
      /* G17P ITRCOORDS.sample: fixed opcode bytes, no input-start marker,
       * and f38=2 in bit57. Its f2 destination is byte3 for the allocator
       * range exercised here. */
      if (code[i] != 0xaf || (code[i + 1] & 0x0f) != 0x04 ||
          code[i + 2] != 0x54 || !(code[i + 7] & 0x02))
         continue;

      const unsigned coords = code[i + 3];
      unsigned interps = 0;
      bool projective_rhw = false;
      bool terminal = false;
      for (unsigned j = i + 8; j + 10 <= size; ++j) {
         if (code[j] != 0x2f || code[j + 2] != 0x54 ||
             ((code[j + 6] & 0x7f) << 1) != coords)
            continue;

         const unsigned coordinate_liveness = code[j + 8] & 0x28;
         if (coordinate_liveness != 0x08 && coordinate_liveness != 0x20)
            continue;

         interps++;
         projective_rhw |= (code[j + 8] & 1) != 0;
         terminal |= coordinate_liveness == 0x20;
      }

      if (interps >= 2 && projective_rhw && terminal)
         return true;
   }

   return false;
}

static bool
g17p_centroid_protocol(const struct agx_shader_part *part)
{
   const uint8_t *code = part->binary;
   const unsigned size = part->info.binary_size;

   /* Apple's G17P centroid fragments open with the entry-marked W16 SR39
    * read in slot 1; ITRCOORDS f38=1 consumes it as its terminal source,
    * waiting on that slot, and the RHW and every channel read the token. */
   for (unsigned sr_dest = 0; sr_dest < 256; ++sr_dest) {
      const struct agx3_sr_desc d = {
         .dest = sr_dest,
         .width = AGX3_SR_W16,
         .sr = 39,
         .slot = 1,
         .first = true,
      };
      uint8_t sr[AGX3_SR_MAX_LENGTH];
      uint8_t sr_length = 0;
      if (!agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &d, sr, &sr_length) ||
          sr_length + 8 > size || memcmp(code, sr, sr_length))
         continue;

      const unsigned coords = code[sr_length + 3];
      uint8_t itrcoords[AGX3_VARY_MAX_LENGTH];
      uint8_t itrcoords_length = 0;
      if (!agx3_vary_centroid_coords(AGX_ISA_ENCODING_AGX3B, coords, sr_dest,
                                     true, 1, itrcoords, &itrcoords_length) ||
          memcmp(code + sr_length, itrcoords, itrcoords_length))
         return false;

      unsigned interps = 0;
      bool projective_rhw = false;
      bool terminal = false;
      for (unsigned j = sr_length + itrcoords_length; j + 10 <= size; ++j) {
         if (code[j] != 0x2f || code[j + 2] != 0x54 ||
             ((code[j + 6] & 0x7f) << 1) != coords)
            continue;

         const unsigned coordinate_liveness = code[j + 8] & 0x28;
         if (coordinate_liveness != 0x08 && coordinate_liveness != 0x20)
            continue;

         interps++;
         projective_rhw |= (code[j + 8] & 1) != 0;
         terminal |= coordinate_liveness == 0x20;
      }

      return interps == 5 && projective_rhw && terminal;
   }

   return false;
}

static bool
g17p_sample_mask_input_protocol(const struct agx_shader_part *part)
{
   const uint8_t *binary = part->binary;
   uint8_t sr_bytes[AGX3_SR_MAX_LENGTH];
   uint8_t sr_length = 0;

   /* Apple's [[sample_mask]] input is the 16-bit SR39 RLD followed by an
    * immediate ushort SRL extracting bits 4..7. The RLD carries the entry
    * marker when it opens the program. Search across allocation and
    * liveness choices, but require the SRL to consume the RLD register and
    * wait on its exact async slot. */
   for (unsigned first = 0; first <= 1; ++first) {
      for (unsigned slot = 1; slot <= 6; ++slot) {
         for (unsigned sr_dest = 0; sr_dest < 256; ++sr_dest) {
            const struct agx3_sr_desc d = {
               .dest = sr_dest,
               .width = AGX3_SR_W16,
               .sr = 39,
               .slot = slot,
               .first = first,
            };
            if (!agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &d, sr_bytes,
                                &sr_length))
               continue;

            for (unsigned sr_off = 0;
                 sr_off + sr_length <= part->info.binary_size; ++sr_off) {
               if (memcmp(binary + sr_off, sr_bytes, sr_length))
                  continue;

               for (unsigned shift_dest = 0; shift_dest < 256; ++shift_dest) {
                  for (unsigned hint = AGX3_ALU_HINT_NONE;
                       hint <= AGX3_ALU_HINT_CACHE; ++hint) {
                     for (unsigned dest_cache = 0; dest_cache <= 1;
                          ++dest_cache) {
                        uint8_t shift_bytes[AGX3_ALU_MAX_LENGTH];
                        uint8_t shift_length = 0;
                        if (!agx3_alu_srl_imm16(
                               AGX_ISA_ENCODING_AGX3B, shift_dest, sr_dest, 4,
                               hint, dest_cache, 1u << (slot - 1), shift_bytes,
                               &shift_length))
                           continue;

                        for (unsigned shift_off = sr_off + sr_length;
                             shift_off + shift_length <= part->info.binary_size;
                             ++shift_off) {
                           if (!memcmp(binary + shift_off, shift_bytes,
                                       shift_length))
                              return true;
                        }
                     }
                  }
               }
            }
         }
      }
   }

   return false;
}

static bool
g17p_depth_protocol(const struct agx_shader_part *part, bool final)
{
   if (!part->info.agx3_staging_known || part->info.agx3_staging_halfs != 4)
      return false;
   const uint8_t acquire[] = {0x87, 0x02, 0x54, 0x01, 0, 0};
   const uint8_t release[] = {0x07, 0x02, 0x54, 0x01, final ? 0x02 : 0, 0};
   /* Two independent f32 staging writes, then one class-1 transaction.
    * The source run is allocator-owned; do not pin a register allocation.
    * A part that ends the program with no colour store makes the release
    * final: metal-nt -arch applegpu_g17p depth-only fragments (the
    * g17p-depth-witness probe, and a [[depth(any)]] function with no colour
    * attachments) end in 870254010000 d71454000003 070254010200, while a
    * fragment returning a colour releases class 1 plainly before its store. */
   for (unsigned slot = 1; slot <= 6; ++slot) {
      for (unsigned src = 0; src < 126; src += 2) {
         for (unsigned discard = 0; discard < 2; ++discard) {
            uint8_t sequence[64], length;
            unsigned size = 0;
            if (!agx3_stage_copy_liveness(AGX_ISA_ENCODING_AGX3B, 0, src,
                                       true, discard, slot, 0, sequence, &length))
               continue;
            size += length;
            if (!agx3_stage_copy_liveness(AGX_ISA_ENCODING_AGX3B, 2, src + 2,
                                       true, discard, slot, 0, sequence + size, &length))
               continue;
            size += length;
            memcpy(sequence + size, acquire, sizeof(acquire));
            size += sizeof(acquire);
            struct agx3_zs_desc desc = {
               .value_kill = true, .slot = slot, .wait_mask = 1u << (slot - 1),
            };
            if (!agx3_zs_encode_depth_stencil(AGX_ISA_ENCODING_AGX3B, &desc,
                                             true, false, sequence + size, &length))
               continue;
            size += length;
            memcpy(sequence + size, release, sizeof(release));
            size += sizeof(release);
            for (unsigned off = 0; off + size <= part->info.binary_size; ++off) {
               if (!memcmp((const uint8_t *)part->binary + off, sequence, size))
                  return true;
            }
         }
      }
   }
   return false;
}

/* Compile one vertex shader for one encoding and report whether it was
 * accepted, dumping the bytes under AGX3_DUMP_BINARY. `expected` pins the
 * whole binary as hex when given: these are this compiler's own bytes, so
 * that a change in what the shapes below emit is a deliberate edit here;
 * the instruction forms inside are the Apple-witnessed ones asserted by the
 * core tests. */
static bool
compile_vs_g17g(const char *name, nir_shader *nir, enum agx_isa_encoding enc,
                bool expect_ok, const char *expected)
{
   struct agx_shader_key key = {
      .dev.encoding = enc,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(nir, &key, &part);
   bool pass = ok == expect_ok;
   if (ok && expected) {
      const size_t expected_size = strlen(expected) / 2;
      pass = pass && part.binary && part.info.binary_size == expected_size;
      for (unsigned i = 0; pass && i < expected_size; ++i) {
         unsigned byte;
         pass = sscanf(expected + 2 * i, "%2x", &byte) == 1 &&
                ((const uint8_t *)part.binary)[i] == byte;
      }
   }
   if (!pass || getenv("AGX3_DUMP_BINARY")) {
      fprintf(stderr, "%s (%s): %s size=%u bytes=", name,
              enc == AGX_ISA_ENCODING_AGX3C ? "agx3c" : "agx3b",
              ok ? "compiled" : "refused", part.info.binary_size);
      for (unsigned i = 0; part.binary && i < part.info.binary_size; ++i)
         fprintf(stderr, "%02x", ((const uint8_t *)part.binary)[i]);
      fprintf(stderr, "\n");
   }
   free(part.binary);
   ralloc_free(nir);
   return pass;
}

/* The fragment twin of compile_vs_g17g. A fragment program reaching the
 * packer has already been through agx_nir_lower_texture and the tilebuffer
 * lowering, so it needs the tilebuffer dependency relaxation the background
 * programs use; nothing else differs. */
static bool
compile_fs_g17g(const char *name, nir_shader *nir, bool expect_ok,
                struct agx_shader_info *info_out)
{
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3C,
      .fs.ignore_tib_dependencies = true,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(nir, &key, &part);
   if (info_out)
      *info_out = part.info;
   if (ok != expect_ok || getenv("AGX3_DUMP_BINARY")) {
      fprintf(stderr, "%s (agx3c): %s size=%u halfs=%u known=%u bytes=", name,
              ok ? "compiled" : "refused", part.info.binary_size,
              part.info.agx3_staging_halfs, part.info.agx3_staging_known);
      for (unsigned i = 0; part.binary && i < part.info.binary_size; ++i)
         fprintf(stderr, "%02x", ((const uint8_t *)part.binary)[i]);
      fprintf(stderr, "\n");
   }
   free(part.binary);
   ralloc_free(nir);
   return ok == expect_ok;
}

/*
 * A 2D-array texel fetch. This is what u_blitter emits for a level-preserving
 * array copy, and what the KHR-GL30.ext_texture_shadow_lod fragments carry
 * alongside their array shadow lookups -- the 2D form of the fetch whose
 * staging bank already had to be widened once for the runtime level.
 * Apple's own G17G array fetch stages x into staging
 * half 0 and y into half 2, then names the layer's *general* register in the
 * sample (f31), so the layer never occupies a staging half.
 */
static nir_shader *
build_fs_g17g_array_fetch(bool runtime_lod)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17G 2D array texel fetch FS");
   b.shader->info.internal = true;

   nir_def *pixel = nir_u2u32(&b, nir_load_pixel_coord(&b));
   nir_def *x = nir_channel(&b, pixel, 0);
   nir_def *y = nir_channel(&b, pixel, 1);
   nir_def *coord = nir_vec3(&b, x, y, nir_ushr_imm(&b, x, 3));
   nir_def *lod =
      runtime_lod ? nir_ushr_imm(&b, y, 4) : nir_imm_int(&b, 0);
   nir_def *texel =
      nir_txf(&b, coord, .lod = lod, .dim = GLSL_SAMPLER_DIM_2D,
              .is_array = true, .dest_type = nir_type_float32,
              .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP);

   fs_store_colour(&b, texel);
   NIR_PASS(_, b.shader, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_10);
   nir_shader *nir = fs_finish(&b);
   NIR_PASS(_, nir, agx_nir_lower_texture_for_generation,
            AGX_DESCRIPTOR_GENERATION_10);
   return nir;
}

static bool
test_g17g_array_fetch(void)
{
   struct agx_shader_info constant = {0}, runtime = {0};
   bool ok = compile_fs_g17g("g17g array fetch, level 0",
                             build_fs_g17g_array_fetch(false), true, &constant);
   ok &= compile_fs_g17g("g17g array fetch, runtime level",
                         build_fs_g17g_array_fetch(true), true, &runtime);
   /* Both shapes stage two coordinate words; the runtime level publishes its
    * word into bank 6 of the same block, so neither may claim fewer than the
    * eight halves the sampler reads. */
   ok &= constant.agx3_staging_known && constant.agx3_staging_halfs >= 8;
   ok &= runtime.agx3_staging_known && runtime.agx3_staging_halfs >= 8;
   return ok;
}

/*
 * A cube shadow lookup with an explicit level, in the two shapes the CTS and
 * the hardware probe respectively use. KHR-GL30.ext_texture_shadow_lod's
 * fragment is
 *
 *   textureLod(samplerCubeShadow, v_texCoord, v_lodShadowRef.x)
 *
 * -- the direction, the comparison reference and the level are all
 * interpolated, so CEM, the reference copy and the level staging each run per
 * lane. The probe that passes on hardware takes all three from uniforms. Emit
 * both and read the difference.
 */
static nir_shader *
build_fs_g17g_cube_shadow_lod(bool per_lane, bool cube_array)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17G cube shadow textureLod");
   b.shader->info.internal = true;

   nir_def *bary = nir_load_barycentric_pixel(
      &b, 32, .interp_mode = INTERP_MODE_SMOOTH);
   nir_def *varying = nir_load_interpolated_input(
      &b, 4, 32, bary, nir_imm_int(&b, 0), .base = 0, .component = 0,
      .io_semantics.location = VARYING_SLOT_VAR0, .io_semantics.num_slots = 1);
   nir_def *coord = nir_trim_vector(&b, varying, cube_array ? 4 : 3);
   nir_def *reference = cube_array
      ? nir_load_preamble(&b, 1, 32, .base = 4)
      : nir_channel(&b, varying, 3);
   nir_def *lod;
   if (per_lane) {
      nir_def *second = nir_load_interpolated_input(
         &b, 1, 32, bary, nir_imm_int(&b, 0), .base = 1, .component = 0,
         .io_semantics.location = VARYING_SLOT_VAR1,
         .io_semantics.num_slots = 1);
      lod = second;
   } else {
      lod = nir_load_preamble(&b, 1, 32, .base = 0);
   }

   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
   tex->is_array = cube_array;
   tex->is_shadow = true;
   tex->is_new_style_shadow = true;
   tex->dest_type = nir_type_float32;
   tex->coord_components = cube_array ? 4 : 3;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator, reference);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
   nir_def_init(&tex->instr, &tex->def, 1, 32);
   nir_builder_instr_insert(&b, &tex->instr);

   nir_def *value = nir_vec4(&b, &tex->def, &tex->def, &tex->def, &tex->def);
   fs_store_colour(&b, value);
   b.shader->info.inputs_read =
      BITFIELD64_BIT(VARYING_SLOT_VAR0) |
      (per_lane ? BITFIELD64_BIT(VARYING_SLOT_VAR1) : 0);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   nir_shader *nir = fs_finish(&b);
   NIR_PASS(_, nir, agx_nir_lower_texture);
   return nir;
}

static bool
test_g17g_cube_shadow_lod(void)
{
   struct agx_shader_info uniform = {0}, per_lane = {0};
   struct agx_shader_info array = {0};
   bool ok = compile_fs_g17g("g17g cube shadow, uniform level",
                             build_fs_g17g_cube_shadow_lod(false, false), true,
                             &uniform);
   ok &= compile_fs_g17g("g17g cube shadow, per-lane level",
                         build_fs_g17g_cube_shadow_lod(true, false), true,
                         &per_lane);
   /* A cube array would need the bindless descriptor handle the driver
    * supplies, which this synthetic NIR has no way to provide
    * (texture_descriptor_ptr asserts on it); its encoding is pinned by the
    * byte-exact vectors in test-agx3-tex-core.c and by the CTS cases. */
   (void)array;
   return ok;
}

/*
 * The shape a transform-feedback vertex shader reduces to: an integer vertex
 * id converted to float, added to a constant, and stored as both a captured
 * varying and the position. On hardware this refuses with
 *
 *   no grounded AGX3 encoding for fadd
 *     [d32:reg2 s0=32:reg0.cache s1=16:reg2.discard chan=1]
 *
 * -- a 32-bit add with a 16-bit operand, which agx3_emit_alu() has no form
 * for.
 *
 * This reproduces on the host, so the gap can be closed without a hardware
 * cycle: the 16-bit operand is the `u2f32` of the vertex id meeting a 32-bit
 * constant, and nothing in the driver's VS lowering chain is needed to
 * provoke it. Asserted as a rejection per this file's convention, so that
 * grounding it later is a deliberate change to this table rather than a
 * silent drift.
 *
 * Fix it by widening rather than by inventing an encoding: AGX3 has no
 * witnessed 32-bit fadd with a 16-bit operand, and `upconvert_uvs_value()`
 * in agx_nir_lower_uvs.c already sets the precedent of converting before the
 * store. Blocks every KHR-GL30.transform_feedback case.
 */
static nir_shader *
build_vs_g17g_xfb_add(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "G17G xfb add VS");

   nir_def *v = nir_u2f32(&b, nir_load_vertex_id(&b));
   nir_def *captured = nir_fadd(&b, v, nir_imm_float(&b, 100.0f));

   nir_store_uvs_agx(&b, captured, nir_imm_intN_t(&b, 4, 16));
   nir_def *pos[4] = {nir_imm_float(&b, 0.0f), nir_imm_float(&b, 0.0f),
                      nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f)};
   for (unsigned c = 0; c < 4; ++c)
      nir_store_uvs_agx(&b, pos[c], nir_imm_intN_t(&b, c, 16));

   b.shader->info.outputs_written = VARYING_BIT_POS | VARYING_BIT_VAR(0);
   agx_preprocess_nir(b.shader);
   return b.shader;
}

static bool
test_g17g_xfb_add(void)
{
   /* Compiles once agx_opt_compact_constants() stops narrowing a constant
    * that feeds a full-width AGX3 float op. */
   return compile_vs_g17g("g17g-xfb-add", build_vs_g17g_xfb_add(),
                          AGX_ISA_ENCODING_AGX3C, true, NULL);
}

/*
 * Apple's 2x/4x fragments writing [[depth(any)]] or [[stencil]] per
 * [[sample_id]] (metal-nt applegpu_g17p) take the class-1 wait once, write
 * each sample's word, and its depth, staged after the previous sample's, and
 * release after the last write. A whole-pixel depth or stencil alone is
 * written the same way for every sample; a whole-pixel depth and stencil is
 * written once.
 */
static bool
g17p_sample_zs_protocol(const struct agx_shader_part *part, unsigned writes,
                        bool depth, bool stencil)
{
   static const uint8_t acquire[] = {0x87, 0x02, 0x54, 0x01, 0, 0};
   static const uint8_t release[] = {0x07, 0x02, 0x54, 0x01, 0, 0};
   const uint8_t *text = part->binary;
   const unsigned step = depth ? 4 : 2;

   if (!part->info.agx3_staging_known ||
       part->info.agx3_staging_halfs < step * writes)
      return false;

   unsigned acquires = 0, releases = 0, acquire_at = 0, release_at = 0;
   for (unsigned off = 0; off + sizeof(acquire) <= part->info.binary_size;
        ++off) {
      if (!memcmp(text + off, acquire, sizeof(acquire))) {
         acquires++;
         acquire_at = off;
      }
      if (!memcmp(text + off, release, sizeof(release))) {
         releases++;
         release_at = off;
      }
   }
   if (acquires != 1 || releases != 1)
      return false;

   unsigned last = acquire_at, found_writes = 0;
   for (unsigned off = acquire_at; off < release_at; ++off) {
      for (unsigned slot = 1; slot <= 6; ++slot) {
         for (unsigned kill = 0; kill < 2; ++kill) {
            for (unsigned staging = 0; staging <= step * 4; staging += 2) {
               struct agx3_zs_desc desc = {
                  .value = staging, .value_kill = kill, .slot = slot,
                  .wait_mask = 1u << (slot - 1),
               };
               uint8_t exact[AGX3_ZS_MAX_LENGTH], length;
               if (!agx3_zs_encode_depth_stencil(AGX_ISA_ENCODING_AGX3B,
                                                &desc, depth, stencil, exact,
                                                &length) ||
                   off + length > release_at ||
                   memcmp(text + off, exact, length))
                  continue;
               if (off < last || staging != step * found_writes)
                  return false;
               last = off + length;
               found_writes++;
            }
         }
      }
   }

   return found_writes == writes;
}

/*
 * Apple's G17P [[stencil]] fragments (metal-nt applegpu_g17p, render
 * pipelines with stencil8 and depth32float_stencil8 attachments) copy the
 * 16-bit reference into the upper half of the coverage register, stage that
 * dword, and write it out in the class-1 transaction:
 *
 *   stencil only   (__text SHA-256 7a34f819...b3b18117)
 *     13042109 copy r1h <- u4     0b022904 copy s0 <- r1 (f27)
 *     870254010000 pixwait 1      d71454000001 depthf (f26)
 *   depth+stencil  (__text SHA-256 e09ec09c...8035a44e)
 *     03142109 copy r0h <- u20    0b012f00420000001000 copy s0 <- r0 (f13)
 *     870254010000 pixwait 1      d71454000007 depthf (f25, f26)
 *
 * Mesa's reference comes from a register rather than a uniform, and depth is
 * staged from a register as in the depth-only protocol above.
 */
static bool
g17p_stencil_protocol(const struct agx_shader_part *part, bool depth)
{
   if (!part->info.agx3_staging_known ||
       part->info.agx3_staging_halfs != (depth ? 4 : 2))
      return false;
   const uint8_t acquire[] = {0x87, 0x02, 0x54, 0x01, 0, 0};
   const uint8_t release[] = {0x07, 0x02, 0x54, 0x01, 0, 0};
   bool upper_half = false, writeout = false;
   for (unsigned word = 0; word < 126; word += 2) {
      for (unsigned src = 0; src < 128; ++src) {
         for (unsigned dest_cache = 0; dest_cache < 2; ++dest_cache) {
            for (unsigned cache = 0; cache < 2; ++cache) {
               for (unsigned discard = 0; discard < 2 - cache; ++discard) {
                  uint8_t copy[AGX3_COPY_MAX_LENGTH], length;
                  if (agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3B, word + 1,
                                             src, dest_cache, cache, discard, 0,
                                             copy, &length))
                     upper_half |= binary_contains(part, copy, length);
               }
            }
         }
      }
      for (unsigned slot = 1; slot <= 6; ++slot) {
         for (unsigned discard = 0; discard < 2; ++discard) {
            uint8_t sequence[64], length;
            unsigned size = 0;
            if (!agx3_stage_copy_liveness(AGX_ISA_ENCODING_AGX3B, 0, word,
                                          depth, discard, slot, 0, sequence,
                                          &length))
               continue;
            size += length;
            if (depth) {
               if (!agx3_stage_copy_liveness(AGX_ISA_ENCODING_AGX3B, 2,
                                             word + 2, true, discard, slot, 0,
                                             sequence + size, &length))
                  continue;
               size += length;
            }
            memcpy(sequence + size, acquire, sizeof(acquire));
            size += sizeof(acquire);
            struct agx3_zs_desc desc = {
               .value_kill = true, .slot = slot, .wait_mask = 1u << (slot - 1),
            };
            if (!agx3_zs_encode_depth_stencil(AGX_ISA_ENCODING_AGX3B, &desc,
                                             depth, true, sequence + size,
                                             &length))
               continue;
            size += length;
            memcpy(sequence + size, release, sizeof(release));
            size += sizeof(release);
            writeout |= binary_contains(part, sequence, size);
         }
      }
   }
   return upper_half && writeout;
}

static const struct agx3_graphics_case cases[] = {
   {"vs-register-multiply-minus-byte", build_vs_g16_imad_negative_byte, true,
    .g17p_only = true},
   {"fs-zink-minifloat-abs-fma", build_fs_zink_minifloat_abs_fma,
    true, .g17p_only = true},
   {"vs-flat-triangle", build_vs_flat, true, .expected_leading_sr = 0xdd},
   {"vs-g17p-div3-mul-high", build_vs_g17p_div3_mul_high, true,
    .expected_leading_sr = 0xdd, .g17p_only = true},
   {"vs-glmark2-ideas", build_vs_glmark2_ideas, true,
    .g17p_only = true},
   {"vs-instance-id", build_vs_instance_id, true,
    .expected_leading_sr = 0xd8},
   {"vs-attribute-fetch", build_vs_attribute, true,
    .expected_leading_sr = 0xdd},
   {"vs-uniform-attribute-fetch", build_vs_uniform_attribute, true},
   {"vs-g17p-float2-position", build_vs_g17p_float2_position, true,
    .expected_g17p_binary = mesa_g17p_float2_vs,
    .expected_g17p_binary_size = sizeof(mesa_g17p_float2_vs),
    .compare_apple_g17p_vs = true, .expected_g17p_loads = 1},
   {"vs-g17p-single-instanced-varying", build_vs_g17p_single_instanced_varying,
    true, .expected_g17p_loads = 1},
   {"vs-g17p-two-loads", build_vs_g17p_two_loads, true,
    .expected_g17p_loads = 2},
   {"vs-g17p-half2-varying", build_vs_g17p_half2_varying, true,
    .expected_g17p_binary = mesa_g17p_half2_varying_vs,
    .expected_g17p_binary_size = sizeof(mesa_g17p_half2_varying_vs),
    .expected_g17p_loads = 2, .g17p_only = true},
   {"vs-vkcube-varyings", build_vs_vkcube_varyings, true},
   {"vs-g17p-36-outputs", build_vs_g17p_36_outputs, true,
    .g17p_only = true},
   {"vs-vkcube-vbo-ubo", build_vs_vkcube_vbo_ubo, true, false,
    AGX_ABI_VUNI_COUNT_VK(2) + 8, 257},
   {"vs-texture-level-zero", build_vs_texture_level_zero, true,
    .g17p_only = true},
   {"vs-texture-register-lod", build_vs_texture_register_lod, true,
    .g17p_only = true},
   {"vs-indexed-texture-fetch", build_vs_indexed_texture_fetch, true,
    .g17p_only = true, .reserved_preamble = 64},
   {"vs-indexed-texture-sample", build_vs_indexed_texture_sample, true,
    .g17p_only = true, .reserved_preamble = 64},
   {"fs-flat-colour", build_fs_flat, true,
    .expected_g17p_binary = g17p_flat_colour,
    .expected_g17p_binary_size = sizeof(g17p_flat_colour)},
   {"fs-interpolated-varying", build_fs_interpolated, true},
   {"fs-interpolated-at-sample", build_fs_interpolated_at_sample, true,
    .g17p_only = true, .expected_g17p_explicit_sample = true},
   {"fs-interpolated-centroid", build_fs_interpolated_centroid, true,
    .g17p_only = true, .expected_g17p_centroid = true},
   {"fs-tilebuffer-load", build_fs_tilebuffer_load, true},
   {"fs-textured-triangle", build_fs_texture_imm, true,
    .expected_g17p_sampled_fragment = true},
   {"fs-texel-fetch-register-lod", build_fs_texel_fetch_register_lod, true,
    .g17p_only = true},
   {"fs-two-texture-sample-group", build_fs_two_texture_samples, true,
    .g17p_only = true, .expected_g17p_sampled_fragment = true,
    .expected_g17p_sample_group = true},
   {"fs-vkcube-combined-sampler", build_fs_texture, true, true,
    .expected_g17p_implicit_coord_group = true,
    .expected_g17p_sampled_fragment = true},
   {"fs-uint-combined-sampler", build_fs_texture_uint, true, true,
    .g17p_only = true, .expected_g17p_implicit_coord_group = true,
    .expected_g17p_sampled_fragment = true},
   {"fs-int-combined-sampler", build_fs_texture_int, true, true,
    .g17p_only = true, .expected_g17p_implicit_coord_group = true,
    .expected_g17p_sampled_fragment = true},
   {"fs-texture3d-level", build_fs_texture_3d_level, true,
    .g17p_only = true, .expected_g17p_sampled_fragment = true},
   {"fs-texture3d-implicit", build_fs_texture_3d_implicit, true,
    .g17p_only = true, .expected_g17p_sampled_fragment = true},
   {"fs-descriptor-resource", build_fs_descriptor_resource, true},
   {"fs-meta-rgba-upload", build_fs_meta_rgba_upload, true, false,
    AGX_ABI_FUNI_ROOT + 4, .g17p_only = true},
   {"fs-transcendentals", build_fs_funops, true},
   {"fs-native-trig", build_fs_native_trig, true, .g17p_only = true,
    .expected_g17p_sincs = 2},
   {"fs-native-trig-f16", build_fs_native_trig_f16, true, .g17p_only = true,
    .expected_g17p_sincs = 2},
   {"fs-quickshell-abs-scale", build_fs_quickshell_abs_scale, true,
    .g17p_only = true},
   {"fs-hyprland-cm-mixed-add", build_fs_hyprland_cm_mixed_add, true,
    .g17p_only = true},
   {"fs-hyprland-discard-predicate", build_fs_hyprland_discard_predicate,
    true, .g17p_only = true},
   {"fs-chromium-negated-equality-select",
    build_fs_chromium_negated_equality_select, true, .g17p_only = true},
   {"fs-chromium-minifloat-branch", build_fs_chromium_minifloat_branch,
    true, .g17p_only = true, .expected_g17p_sampled_fragment = true},
   {"fs-ordered-abs-branch", build_fs_ordered_abs_branch, true,
    .g17p_only = true, .expected_g17p_sampled_fragment = true},
   {"fs-hyprland-motion-bound", build_fs_hyprland_motion_bound, true,
    .g17p_only = true},
   {"fs-glmark2-buffer", build_fs_glmark2_buffer, true,
    .g17p_only = true},
   {"fs-glmark2-control", build_fs_glmark2_control, true,
    .g17p_only = true},
   {"fs-glmark2-desktop", build_fs_glmark2_desktop, true,
    .g17p_only = true},
   {"fs-glmark2-bump", build_fs_glmark2_bump, true,
    .g17p_only = true},
   {"fs-glmark2-ideas", build_fs_glmark2_ideas, true,
    .g17p_only = true},
   {"fs-glmark2-refract", build_fs_glmark2_refract, true,
    .g17p_only = true},
   {"fs-glmark2-terrain-offset", build_fs_glmark2_terrain_offset, true,
    .g17p_only = true},
   {"fs-glmark2-terrain-noise", build_fs_glmark2_terrain_noise, true,
    .g17p_only = true},

   {"fs-discard", build_fs_discard, true, .g17p_only = true},
   {"fs-discard-then-tests", build_fs_discard_then_tests, true,
    .g17p_only = true, .expected_g17p_balanced_coverage = true},
   /* At several samples the coverage word is built from SR39's low byte. */
   {"fs-discard-then-tests-2x", build_fs_discard_then_tests, true,
    .g17p_only = true, .expected_g17p_sample_mask_input = true,
    .expected_g17p_balanced_coverage = true, .nr_samples = 2},
   {"fs-sample-mask-write", build_fs_sample_mask_write, true,
    .g17p_only = true, .expected_g17p_balanced_coverage = true},
   {"fs-alpha-discard-word", build_fs_alpha_discard_word, true,
    .g17p_only = true, .expected_g17p_balanced_coverage = true},
   /* Four samples keep whole masks (the SR39 path); only the select uses
    * the widening multiply. */
   {"fs-alpha-discard-word-4x", build_fs_alpha_discard_word, true,
    .g17p_only = true, .expected_g17p_balanced_coverage = true,
    .nr_samples = 4},
   {"fs-kill-then-alpha-discard-word", build_fs_kill_then_alpha_discard_word,
    true, .g17p_only = true, .expected_g17p_balanced_coverage = true},
   {"fs-sample-mask-write-4x", build_fs_sample_mask_write, true,
    .g17p_only = true, .expected_g17p_sample_mask_input = true,
    .expected_g17p_balanced_coverage = true, .nr_samples = 4},
   {"fs-sample-mask-in", build_fs_sample_mask_in, true, .g17p_only = true,
    .expected_g17p_sample_mask_input = true},
   {"fs-front-facing", build_fs_front_facing, true, .g17p_only = true},
   {"fs-depth-write", build_fs_depth, true, .g17p_only = true,
    .expected_g17p_depth = true},
   {"fs-depth-only", build_fs_depth_only, true, .g17p_only = true,
    .expected_g17p_depth = true},
   {"fs-depth-precision", build_fs_depth_precision, true, .g17p_only = true,
    .expected_g17p_depth = 3},
   {"fs-depth-write-dynamic", build_fs_depth_dynamic, true,
    .g17p_only = true, .expected_g17p_depth = 2},
   {"fs-stencil-write", build_fs_stencil, true, .g17p_only = true,
    .expected_g17p_stencil = 1},
   {"fs-depth-stencil-write", build_fs_depth_stencil, true,
    .g17p_only = true, .expected_g17p_stencil = 2},
   /* A multisample depth or stencil export writes each sample separately;
    * both together are written once for every covered sample. */
   {"fs-depth-write-4x", build_fs_depth, true, .g17p_only = true,
    .expected_g17p_sample_mask_input = true,
    .expected_g17p_sample_zs_writes = 4, .nr_samples = 4},
   {"fs-stencil-write-2x", build_fs_stencil, true, .g17p_only = true,
    .expected_g17p_sample_mask_input = true,
    .expected_g17p_sample_zs_writes = 2, .nr_samples = 2},
   {"fs-depth-stencil-write-4x", build_fs_depth_stencil, true,
    .g17p_only = true, .expected_g17p_sample_zs_writes = 1, .nr_samples = 4},
   /* A write to samples known only at run time takes the same per-sample
    * words, with the run-time predicate ANDed into each coverage nibble. */
   {"fs-depth-dynamic-samples-4x", build_fs_depth_dynamic_samples, true,
    .g17p_only = true, .expected_g17p_sample_mask_input = true,
    .expected_g17p_sample_zs_writes = 4, .nr_samples = 4},
};

static bool
run_case(const struct agx3_graphics_case *c, enum agx_isa_encoding enc)
{
   nir_shader *nir = c->build();
   const bool writes_colour =
      nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DATA0);

   if (c->lower_combined_sampler) {
      struct agx_g17p_combined_sampler_binding binding;
      if (!agx_nir_lower_g17p_combined_sampler(nir, &binding, true) ||
          binding.texture_offset_B != 64 ||
          binding.sampler_index_offset_B != 104 ||
          !has_compact_native_sample(nir)) {
         fprintf(stderr, "%s %s: combined descriptor lowering mismatch\n",
                 enc_name(enc), c->name);
         ralloc_free(nir);
         return false;
      }
   }

   struct agx_shader_key key = {0};
   key.dev.encoding = enc;
   key.no_preamble = true;
   key.reserved_preamble = c->reserved_preamble;
   key.fs.nr_samples = c->nr_samples;

   struct agx_shader_part part;
   memset(&part, 0xa5, sizeof(part));

   const bool ok = agx_compile_shader_nir(nir, &key, &part);
   bool result;

   if (!ok) {
      const uint8_t zero[sizeof(part)] = {0};
      if (memcmp(&part, zero, sizeof(part)) != 0) {
         fprintf(stderr, "%s %s: rejection did not clear compiler output\n",
                 enc_name(enc), c->name);
         ralloc_free(nir);
         return false;
      }

      result = !c->expect_ok;
      if (!result)
         fprintf(stderr, "%s %s: REJECTED by agx3_pack_binary\n", enc_name(enc),
                 c->name);
   } else {
      result = c->expect_ok && part.binary != NULL &&
               part.info.binary_size > sizeof(agx3_end) &&
               part.info.main_offset == 0 &&
               memcmp((const uint8_t *)part.binary + part.info.binary_size -
                         sizeof(agx3_end),
                      agx3_end, sizeof(agx3_end)) == 0;

      if (result && c->reserved_preamble != 0) {
         result = !part.info.has_preamble &&
                  part.info.push_count == c->reserved_preamble;
      }

      if (result && c->minimum_binary_size != 0)
         result = part.info.binary_size >= c->minimum_binary_size;

      if (result && c->expected_leading_sr != 0) {
         const uint8_t expected[] = {
            0x0c, c->expected_leading_sr, 0x10, 0x06,
         };
         result = part.info.binary_size >= sizeof(expected) &&
                  memcmp(part.binary, expected, sizeof(expected)) == 0;
         if (!result) {
            fprintf(stderr, "%s %s: missing leading SR %#x rld\n",
                    enc_name(enc), c->name, c->expected_leading_sr);
         }
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->expected_g17p_binary != NULL) {
         const bool resources_unchanged =
            !part.info.has_preamble && part.info.push_count == 0 &&
            part.info.scratch_size == 0 && !part.info.uses_txf &&
            !part.info.uses_sampler_heap &&
            part.info.texture_state_count == 0 &&
            part.info.sampler_state_count == 0;
         result = resources_unchanged &&
                  part.info.binary_size == c->expected_g17p_binary_size &&
                  memcmp(part.binary, c->expected_g17p_binary,
                         c->expected_g17p_binary_size) == 0;
         if (!result) {
            fprintf(stderr,
                    "%s %s: G17P binary/resource metadata mismatch\n",
                    enc_name(enc), c->name);
            fprintf(stderr, "actual main (%u bytes, staging %u halfs): ",
                    part.info.binary_size, part.info.agx3_staging_halfs);
            for (unsigned j = 0; j < part.info.binary_size; ++j)
               fprintf(stderr, "%02x", ((const uint8_t *)part.binary)[j]);
            fprintf(stderr, "\n");
         }
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->compare_apple_g17p_vs) {
         result = part.info.agx3_staging_known && part.info.agx3_staging_halfs == 8 &&
                  g17p_vs_matches_apple(part.binary, part.info.binary_size);
         if (!result) {
            fprintf(stderr, "%s %s: Apple-88 VS protocol mismatch\n",
                    enc_name(enc), c->name);
         }
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->expected_g17p_loads != 0) {
         result = g17p_vertex_load_policy(part.binary, part.info.binary_size,
                                           c->expected_g17p_loads);
         if (!result) {
            fprintf(stderr, "%s %s: vertex LDDEV policy mismatch\n",
                    enc_name(enc), c->name);
         }
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->expected_g17p_sincs != 0) {
         unsigned found = 0;
         const uint8_t *text = part.binary;
         for (unsigned off = 0; off + 10 <= part.info.binary_size; off += 2) {
            if (text[off] != 0x2f || text[off + 1] != 0x03)
               continue;
            bool match = false;
            for (unsigned dest = 0; dest < 128 && !match; dest += 2) {
               for (unsigned src = 0; src < 128 && !match; src += 2) {
                  uint8_t exact[AGX3_FUNOP_MAX_LENGTH], len = 0;
                  match = agx3_funop_encode(enc, AGX3_FUNOP_SINC, dest, src,
                                            false, false, false, 0,
                                            exact, &len) && len == 10 &&
                          memcmp(text + off, exact, len) == 0;
               }
            }
            found += match;
         }
         result = found == c->expected_g17p_sincs;
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B && c->nr_samples > 1 &&
          c->expected_g17p_sample_zs_writes) {
         const bool depth_only = c->build == build_fs_depth ||
                                 c->build == build_fs_depth_dynamic_samples;
         result = g17p_sample_zs_protocol(
            &part, c->expected_g17p_sample_zs_writes,
            c->build != build_fs_stencil, !depth_only);
         if (!result)
            fprintf(stderr, "%s %s: per-sample depth/stencil protocol "
                            "mismatch\n",
                    enc_name(enc), c->name);
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->expected_g17p_depth) {
         result = g17p_depth_protocol(&part, !writes_colour);
         if (!result)
            fprintf(stderr, "%s %s: depth staging protocol mismatch\n",
                    enc_name(enc), c->name);
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->expected_g17p_stencil) {
         result = g17p_stencil_protocol(&part, c->expected_g17p_stencil == 2);
         if (!result)
            fprintf(stderr, "%s %s: stencil staging protocol mismatch\n",
                    enc_name(enc), c->name);
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->expected_g17p_implicit_coord_group) {
         result = g17p_implicit_coord_group(&part);
         if (!result)
            fprintf(stderr, "%s %s: implicit coordinate group mismatch\n",
                    enc_name(enc), c->name);
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          part.info.g17p_sampled_fragment !=
             c->expected_g17p_sampled_fragment) {
         result = false;
         fprintf(stderr, "%s %s: sampled-fragment constant-program flag "
                         "mismatch\n",
                 enc_name(enc), c->name);
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->expected_g17p_sample_mask_input) {
         result = g17p_sample_mask_input_protocol(&part);
         if (!result)
            fprintf(stderr,
                    "%s %s: missing native SR39 W16 read/shift protocol\n",
                    enc_name(enc), c->name);
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->expected_g17p_explicit_sample) {
         result = g17p_explicit_sample_protocol(&part);
         if (!result)
            fprintf(stderr,
                    "%s %s: missing native ITRCOORDS/f32interp "
                    "sample token chain\n",
                    enc_name(enc), c->name);
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->expected_g17p_centroid) {
         result = g17p_centroid_protocol(&part);
         if (!result)
            fprintf(stderr,
                    "%s %s: missing native SR39/ITRCOORDS/f32interp "
                    "centroid token chain\n",
                    enc_name(enc), c->name);
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->expected_g17p_sample_group) {
         result = g17p_sample_group_protocol(&part);
         if (!result)
            fprintf(stderr,
                    "%s %s: fragment SMP group did not terminate only on "
                    "its final sample\n",
                    enc_name(enc), c->name);
      }

      if (result && enc == AGX_ISA_ENCODING_AGX3B &&
          c->expected_g17p_balanced_coverage) {
         /* The actual discard lowering emits a no-tests update followed by
          * a testing update. Only the latter owns a class-1 transaction. */
         const uint8_t acquire[] = {0x87, 0x02, 0x54, 0x01, 0x00, 0x00};
         const uint8_t release[] = {0x07, 0x02, 0x54, 0x01, 0x00, 0x00};
         unsigned acquired = 0, released = 0;
         for (unsigned i = 0; i + sizeof(acquire) <= part.info.binary_size;
              ++i) {
            const uint8_t *bytes = (const uint8_t *)part.binary + i;
            acquired += !memcmp(bytes, acquire, sizeof(acquire));
            released += !memcmp(bytes, release, sizeof(release));
         }
         result = acquired == 1 && released == 1;
         if (!result)
            fprintf(stderr, "%s %s: coverage acquired %u, released %u\n",
                    enc_name(enc), c->name, acquired, released);
      }

      if (result && c->lower_combined_sampler) {
         nir_shader *reference_nir = build_fs_texture_compact_native();
         struct agx_shader_part reference = {0};
         const bool reference_ok =
            has_compact_native_sample(reference_nir) &&
            agx_compile_shader_nir(reference_nir, &key, &reference);
         const bool binary_matches =
            reference_ok &&
            reference.info.binary_size == part.info.binary_size &&
            memcmp(reference.binary, part.binary, part.info.binary_size) == 0;
         result = binary_matches && !part.info.has_preamble &&
                  !part.info.uses_sampler_heap &&
                  part.info.texture_state_count == 0 &&
                  part.info.sampler_state_count == 0;
         if (!result) {
            fprintf(stderr,
                    "%s %s: specialized=%u reference=%u binary_match=%u "
                    "preamble=%u heap=%u ts=%u ss=%u\n",
                    enc_name(enc), c->name, part.info.binary_size,
                    reference.info.binary_size, binary_matches,
                    part.info.has_preamble, part.info.uses_sampler_heap,
                    part.info.texture_state_count,
                    part.info.sampler_state_count);
         }
         free(reference.binary);
         ralloc_free(reference_nir);
      }

      if (!result) {
         fprintf(stderr, "%s %s: %s\n", enc_name(enc), c->name,
                 c->expect_ok ? "accepted but binary is malformed"
                              : "unexpectedly accepted");
      } else if (getenv("AGX3_DUMP_BINARY")) {
         fprintf(stderr, "%s %s: %u bytes:", enc_name(enc), c->name,
                 part.info.binary_size);
         for (unsigned i = 0; i < part.info.binary_size; ++i)
            fprintf(stderr, " %02x", ((const uint8_t *)part.binary)[i]);
         fprintf(stderr, "\n");
      }

      free(part.binary);
   }

   ralloc_free(nir);
   return result;
}

static bool
test_g17p_half2_varying(void)
{
   const struct agx3_graphics_case c = {
      "vs-g17p-half2-varying", build_vs_g17p_half2_varying, true,
      .expected_g17p_binary = mesa_g17p_half2_varying_vs,
      .expected_g17p_binary_size = sizeof(mesa_g17p_half2_varying_vs),
      .expected_g17p_loads = 2,
   };
   return run_case(&c, AGX_ISA_ENCODING_AGX3B);
}

static bool
test_g17p_gallium_vertex_param_abi(void)
{
   /* LDDEV uniform bases are 64-bit pointers and therefore 4-half aligned.
    * Odd VBO counts used to place the software-vertex parameter pointer at
    * half 14, which made Gallium's internal blitter VS fail compilation.
    */
   for (unsigned nr_vbos = 0; nr_vbos < 8; ++nr_vbos) {
      const unsigned base = AGX_ABI_VUNI_VERTEX_PARAMS(nr_vbos);
      if ((base & 3) || AGX_ABI_VUNI_COUNT_GL(nr_vbos, true) < base + 4)
         return false;
   }

   return AGX_ABI_VUNI_VERTEX_PARAMS(1) == 16 &&
          AGX_ABI_VUNI_VERTEX_PARAMS(2) == 20;
}

static bool
test_agx3_heavy_branch_loop_unroll_limit(void)
{
   /* 1. Loop-free shader reports !agx_nir_has_loop(). */
   nir_builder b0 = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "loop-free-check");
   nir_def *gid0 = nir_channel(&b0, nir_load_global_invocation_id(&b0, 32), 0);
   nir_store_global(&b0, nir_iadd_imm(&b0, gid0, 1), nir_imm_int64(&b0, 0x1000));
   if (agx_nir_has_loop(b0.shader)) {
      ralloc_free(b0.shader);
      return false;
   }
   ralloc_free(b0.shader);

   /* 2. Multi-branch pure-ALU vector loop (pso_uber shape: 12 iterations,
    * 2+ nested ifs, >512 scalarized ops total, zero memory loads) must be
    * marked nir_loop_control_dont_unroll so pre-scalarization unrolling does
    * not explode compile time. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "heavy-branch-unroll-limit");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc =
      nir_local_variable_create(b.impl, glsl_vector_type(GLSL_TYPE_UINT, 4), "v");
   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_vec4(&b, id, nir_iadd_imm(&b, id, 1),
                                   nir_iadd_imm(&b, id, 2),
                                   nir_iadd_imm(&b, id, 3)), 0xf);
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_push_if(&b, nir_uge_imm(&b, i, 12));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);

      nir_def *sel = nir_iand_imm(&b, nir_iadd(&b, id, i), 3);
      nir_def *v = nir_load_var(&b, acc);
      nir_push_if(&b, nir_ieq_imm(&b, sel, 0));
      {
         nir_def *t = v;
         for (unsigned k = 0; k < 8; ++k)
            t = nir_ixor(&b, nir_imul_imm(&b, t, 16777619u + k),
                         nir_ushr_imm(&b, t, 3));
         nir_store_var(&b, acc, t, 0xf);
      }
      nir_push_else(&b, NULL);
      {
         nir_push_if(&b, nir_ieq_imm(&b, sel, 1));
         {
            nir_def *t = v;
            for (unsigned k = 0; k < 8; ++k)
               t = nir_iadd(&b, nir_imul_imm(&b, t, 747796405u + k),
                            nir_ishl_imm(&b, t, 5));
            nir_store_var(&b, acc, t, 0xf);
         }
         nir_pop_if(&b, NULL);
      }
      nir_pop_if(&b, NULL);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   nir_def *fin = nir_load_var(&b, acc);
   nir_store_global(&b, nir_channel(&b, fin, 0), nir_imm_int64(&b, 0x2000));
   nir_lower_vars_to_ssa(b.shader);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_remove_dead_variables(b.shader, nir_var_function_temp, NULL);

   if (!agx_nir_has_loop(b.shader)) {
      ralloc_free(b.shader);
      return false;
   }
   agx_nir_limit_heavy_branch_unroll(b.shader);
   bool marked_dont_unroll = (loop->control == nir_loop_control_dont_unroll);
   ralloc_free(b.shader);
   if (!marked_dont_unroll)
      return false;

   /* 3. Negative control: a short 4-iteration loop must remain unrollable. */
   nir_builder b2 = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "short-loop-unroll-ok");
   nir_variable *iv2 = nir_local_variable_create(b2.impl, glsl_uint_type(), "i");
   nir_variable *acc2 = nir_local_variable_create(b2.impl, glsl_uint_type(), "a");
   nir_store_var(&b2, iv2, nir_imm_int(&b2, 0), 1);
   nir_store_var(&b2, acc2, nir_imm_int(&b2, 1), 1);
   nir_loop *loop2 = nir_push_loop(&b2);
   {
      nir_def *i = nir_load_var(&b2, iv2);
      nir_push_if(&b2, nir_uge_imm(&b2, i, 4));
      nir_jump(&b2, nir_jump_break);
      nir_pop_if(&b2, NULL);
      nir_store_var(&b2, acc2, nir_iadd(&b2, nir_load_var(&b2, acc2), i), 1);
      nir_store_var(&b2, iv2, nir_iadd_imm(&b2, i, 1), 1);
   }
   nir_pop_loop(&b2, loop2);
   nir_store_global(&b2, nir_load_var(&b2, acc2), nir_imm_int64(&b2, 0x3000));
   nir_lower_vars_to_ssa(b2.shader);
   nir_opt_copy_prop(b2.shader);
   nir_opt_dce(b2.shader);
   nir_remove_dead_variables(b2.shader, nir_var_function_temp, NULL);
   agx_nir_limit_heavy_branch_unroll(b2.shader);
   bool kept_unrollable = (loop2->control == nir_loop_control_none);
   ralloc_free(b2.shader);
   return kept_unrollable;
}

static bool
test_agx3_tiny_push_vertex_preamble_inlined(void)
{
   /* 1. Tiny push-only vertex shader (bubo.vert shape: loads 4-byte draw_id
    * from root+768, computes tile x/y and NDC offset in ~8 fast 1-cycle ALU
    * ops). Uniform DMA must replace the root load, inline the preamble away
    * (has_preamble == false), and record !agx3_main_uses_root_uniform. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "agx3-tiny-push-vs");
   b.shader->info.io_lowered = true;
   b.shader->info.outputs_written = VARYING_BIT_POS;

   nir_def *root = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *draw_id = nir_load_global_constant(
      &b, 1, 32, nir_iadd_imm(&b, root, 768), .align_mul = 4);
   nir_def *tx = nir_iand_imm(&b, draw_id, 63);
   nir_def *ty = nir_ushr_imm(&b, draw_id, 6);
   nir_def *fx = nir_ffma(&b, nir_u2f32(&b, tx), nir_imm_float(&b, 0.03125f),
                          nir_imm_float(&b, -1.0f));
   nir_def *fy = nir_ffma(&b, nir_u2f32(&b, ty), nir_imm_float(&b, 0.03125f),
                          nir_imm_float(&b, -1.0f));
   /* A hardware vertex shader reads its index as load_vertex_id;
    * load_vertex_id_zero_base is the software (compute) VS form. */
   nir_def *vid = nir_load_vertex_id(&b);
   nir_def *vx = nir_fadd(&b, fx, nir_u2f32(&b, nir_iand_imm(&b, vid, 1)));
   nir_def *vy = nir_fadd(&b, fy, nir_u2f32(&b, nir_ushr_imm(&b, vid, 1)));
   nir_def *pos[4] = {vx, vy, nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f)};
   for (unsigned c = 0; c < 4; ++c)
      nir_store_uvs_agx(&b, pos[c], nir_imm_intN_t(&b, c, 16));
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.reserved_preamble = 4;
   key.agx3_native_cprog = true;
   key.g17p_native_uniforms = true;
   key.g17p_uniform_dma = true;
   key.g17p_dma_root = 0;

   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(b.shader, &key, &part);
   bool tiny_ok = ok && !part.info.has_preamble &&
                  part.info.agx3_uniform_dma_count == 1 &&
                  part.info.agx3_uniform_dma[0].offset_B == 768 &&
                  part.info.agx3_uniform_dma[0].halfs == 2 &&
                  part.info.agx3_root_uniform_usage_valid &&
                  !part.info.agx3_main_uses_root_uniform &&
                  !part.info.agx3_preamble_uses_root_uniform;
   free(part.binary);
   ralloc_free(b.shader);
   if (!tiny_ok)
      return false;

   /* 2. Negative control: a vertex preamble doing slow ops (frcp) or heavy
    * multi-word root math (>16 bytes loaded) must NOT be inlined away. */
   nir_builder b2 = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "agx3-heavy-root-vs");
   b2.shader->info.io_lowered = true;
   b2.shader->info.outputs_written = VARYING_BIT_POS;

   nir_def *root2 = nir_load_preamble(&b2, 1, 64, .base = 0);
   nir_def *acc = nir_imm_float(&b2, 1.0f);
   for (unsigned k = 0; k < 8; ++k) {
      nir_def *w = nir_load_global_constant(
         &b2, 1, 32, nir_iadd_imm(&b2, root2, 768 + k * 4), .align_mul = 4);
      acc = nir_ffma(&b2, acc, w, nir_frcp(&b2, nir_fadd_imm(&b2, w, 2.0f)));
   }
   nir_def *vid2 = nir_load_vertex_id(&b2);
   nir_def *vx2 = nir_fadd(&b2, acc, nir_u2f32(&b2, vid2));
   nir_def *pos2[4] = {vx2, acc, nir_imm_float(&b2, 0.0f),
                       nir_imm_float(&b2, 1.0f)};
   for (unsigned c = 0; c < 4; ++c)
      nir_store_uvs_agx(&b2, pos2[c], nir_imm_intN_t(&b2, c, 16));
   agx_preprocess_nir(b2.shader);

   struct agx_shader_part part2 = {0};
   bool ok2 = agx_compile_shader_nir(b2.shader, &key, &part2);
   bool heavy_kept = ok2 && part2.info.has_preamble &&
                     part2.info.agx3_root_uniform_usage_valid &&
                     !part2.info.agx3_main_uses_root_uniform &&
                     part2.info.agx3_preamble_uses_root_uniform;
   free(part2.binary);
   ralloc_free(b2.shader);
   return heavy_kept;
}

/* A vertex shader with nothing to hoist has no constant program. Looking
 * for a hoistable vertex preamble must not create an empty one. */
static bool
test_agx3_vertex_without_preamble_has_none(void)
{
   nir_shader *nir = build_vs_flat();

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.reserved_preamble = 4;
   key.agx3_native_cprog = true;
   key.g17p_native_uniforms = true;

   struct agx_shader_part part = {0};
   bool ok = agx_compile_shader_nir(nir, &key, &part);
   ok = ok && !part.info.has_preamble && part.info.nr_preamble_gprs == 0 &&
        part.info.preamble_size == 0;
   free(part.binary);
   ralloc_free(nir);
   return ok;
}

/* The driver keeps a vertex shader's descriptor-set pointers only when its
 * uniform-half record names one (hk_agx3_g17p_graphics_live_set_count()).
 * Firefox's vertex texel fetch through set 3 reads that set's pointer, halves
 * 12..15 under lower_test_descriptor_set(), and not set 1's, halves 4..7.
 */
static bool
test_agx3_vertex_set_pointer_uniform_halfs(void)
{
   nir_shader *nir = build_vs_indexed_texture_fetch();
   if (!nir)
      return false;

   struct agx_shader_key key = {0};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   key.reserved_preamble = 64;

   struct agx_shader_part part = {0};
   const bool ok = agx_compile_shader_nir(nir, &key, &part);
   const uint64_t halfs = part.info.agx3_uniform_halfs[0];
   const uint64_t set1 = BITFIELD64_RANGE(4, 4), set3 = BITFIELD64_RANGE(12, 4);
   const bool named = ok && part.info.agx3_uniform_halfs_valid &&
                      (halfs & set3) == set3 && !(halfs & set1);
   if (!named) {
      fprintf(stderr, "vertex set-pointer halves: ok=%d valid=%d "
              "halfs0=0x%llx\n", ok, part.info.agx3_uniform_halfs_valid,
              (unsigned long long)halfs);
   }
   free(part.binary);
   ralloc_free(nir);
   return named;
}

/* Unpacking a 32-bit word into four bytes, clamping to 255, narrowing to u16,
 * and storing to an R8G8B8A8_UINT tilebuffer (vk_meta's compressed
 * buffer-to-image upload shape) must fold away the unpack/clamp/repack chain
 * while leaving non-identity byte permutations on the explicit packing path. */
static bool
test_agx3_packed_byte_word_tile_store(void)
{
   const enum agx_isa_encoding encs[] = {
      AGX_ISA_ENCODING_AGX3B,
      AGX_ISA_ENCODING_AGX3C,
   };

   for (unsigned e = 0; e < ARRAY_SIZE(encs); ++e) {
      unsigned sizes[4] = {0, 0, 0, 0};
      for (unsigned mode = 0; mode < 4; ++mode) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_FRAGMENT, &agx_nir_options, "packed-i8-tile");
         b.shader->info.io_lowered = true;
         nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);
         nir_def *word = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
         nir_def *ch[4];
         if (mode == 2) {
            /* Exact half-bitfield-extract lowered shape produced for vk_meta
             * R32_UINT -> R8G8B8A8_UINT uploads (shared shift + unpack_y). */
            nir_def *q = nir_ushr_imm(&b, word, 8);
            nir_def *raw[4] = {
               nir_iand_imm(&b, word, 0xff),
               nir_u2u32(&b, nir_iand_imm(&b, nir_unpack_32_2x16_split_x(&b, q), 0xff)),
               nir_u2u32(&b, nir_iand_imm(&b, nir_unpack_32_2x16_split_y(&b, word), 0xff)),
               nir_u2u32(&b, nir_unpack_32_2x16_split_y(&b, q)),
            };
            for (unsigned i = 0; i < 4; ++i)
               ch[i] = nir_u2u16(&b, nir_umin(&b, raw[i], nir_imm_int(&b, 255)));
         } else if (mode == 3) {
            /* Exact nir_format_unpack_int() shift-left/shift-right shape
             * emitted by vk_meta_copy_buffer_to_image for R32_UINT ->
             * R8G8B8A8_UINT compressed uploads. */
            for (unsigned i = 0; i < 4; ++i) {
               nir_def *ext = nir_ushr_imm(&b, nir_ishl_imm(&b, word, 24 - i * 8), 24);
               nir_def *clamped = nir_umin(&b, ext, nir_imm_int(&b, 255));
               ch[i] = nir_u2u16(&b, clamped);
            }
         } else {
            for (unsigned i = 0; i < 4; ++i) {
               unsigned src_byte = mode ? (3 - i) : i;
               nir_def *ext = nir_ubitfield_extract_imm(&b, word, src_byte * 8, 8);
               nir_def *clamped = nir_umin(&b, ext, nir_imm_int(&b, 255));
               ch[i] = nir_u2u16(&b, clamped);
            }
         }
         nir_store_local_pixel_agx(
            &b, nir_vec(&b, ch, 4), nir_imm_intN_t(&b, 0xff, 16),
            nir_undef(&b, 2, 16), .base = 0, .write_mask = 0xf,
            .format = PIPE_FORMAT_R8_UINT, .target = 0);

         struct agx_shader_key key = {0};
         key.dev.encoding = encs[e];
         key.no_preamble = true;
         key.reserved_preamble = 8;
         key.fs.nr_samples = 1;

         struct agx_shader_part part = {0};
         const bool ok = agx_compile_shader_nir(b.shader, &key, &part);
         sizes[mode] = part.info.binary_size;
         free(part.binary);
         ralloc_free(b.shader);
         if (!ok || !sizes[mode]) {
            fprintf(stderr, "packed byte-word tile store failed enc=%u mode=%u\n",
                    e, mode);
            return false;
         }
      }
      if (sizes[0] + 48 > sizes[1] || sizes[2] + 48 > sizes[1] ||
          sizes[3] + 48 > sizes[1]) {
         fprintf(stderr,
                 "packed byte-word tile store did not fold enc=%u: id=%u half=%u shift=%u perm=%u\n",
                 e, sizes[0], sizes[2], sizes[3], sizes[1]);
         return false;
      }
   }
   return true;
}

int
main(int argc, char **argv)
{
   if (argc == 3 && !strcmp(argv[1], "--m5-export-flat-variant"))
      return export_g17g_flat_variant(argv[2]) ? 0 : 1;
   if (argc == 2 && !strcmp(argv[1], "--zink-minifloat-abs-fma-only")) {
      const struct agx3_graphics_case c = {
         "fs-zink-minifloat-abs-fma", build_fs_zink_minifloat_abs_fma,
         true, .g17p_only = true,
      };
      const bool ok = run_case(&c, AGX_ISA_ENCODING_AGX3B);
      fprintf(stderr, "G17P Zink minifloat absolute-addend FMA: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--vertex-texture-lod-only")) {
      const struct agx3_graphics_case c = {
         "vs-texture-register-lod", build_vs_texture_register_lod, true,
         .g17p_only = true,
      };
      const bool ok = run_case(&c, AGX_ISA_ENCODING_AGX3B);
      fprintf(stderr, "G17P vertex register texture LOD: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--zink-minifloat-abs-fma-only")) {
      const struct agx3_graphics_case c = {
         "fs-zink-minifloat-abs-fma", build_fs_zink_minifloat_abs_fma,
         true, .g17p_only = true,
      };
      const bool ok = run_case(&c, AGX_ISA_ENCODING_AGX3B);
      fprintf(stderr, "G17P Zink minifloat absolute-addend FMA: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--alpha-coverage-only")) {
      const bool ok = test_g17p_alpha_coverage_boundary();
      fprintf(stderr, "G17P alpha-coverage boundary select: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--quad-control-only")) {
      const bool ok = test_g17p_quad_control_modes() &&
                      test_g17p_divergent_quad_loop();
      fprintf(stderr, "G17P shader quad-control protocol: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--gallium-vp-abi-only")) {
      const bool ok = test_g17p_gallium_vertex_param_abi();
      fprintf(stderr, "G17P Gallium vertex-parameter ABI alignment: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--half-vbo-only")) {
      const bool ok = test_g17p_half2_varying();
      fprintf(stderr, "G17P R16G16_SFLOAT vertex attribute: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--eot-only")) {
      const bool ok = test_g17p_eot_block_store();
      printf("G17P RT0 and WBOIT MRT EOT: %s\n", ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--msaa-eot-only")) {
      const bool ok = test_g17p_multisample_eot_block_store();
      printf("G17P multisample RT0 EOT: %s\n", ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--custom-border-only")) {
      bool ok = test_custom_border_samplers(false, false) &&
                test_custom_border_samplers(true, false) &&
                test_custom_border_samplers(true, 1) &&
                test_custom_border_samplers(true, 2) &&
                test_indexed_sampler_native_indices() &&
                test_indexed_sampler_sets() && test_indexed_fetch_sets();
      printf("custom border runtime branch, exact indices, mixed reads: %s\n",
             ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--indexed-fetch-only")) {
      const bool ok=test_indexed_dynamic_fetch_lod() &&
                    test_indexed_fetch_sets() &&
                    test_indexed_multisample_fetch() &&
                    test_g17p_fetch_lod_compile(0) &&
                    test_g17p_fetch_lod_compile(16) &&
                    test_g17p_fetch_lod_compile(32);
      printf("mixed indexed texture reads, multisample resolve, sets and arrays: %s\n",ok?"PASS":"FAIL");
      return ok?0:1;
   }
   if (argc == 2 && !strcmp(argv[1], "--indexed-samplers-only")) {
      const bool ok = test_indexed_sampler_sets() &&
                      test_indexed_sampler_native_indices() &&
                      test_indexed_array_and_3d_compile() &&
                      test_indexed_compute_sample() &&
                      test_indexed_storage_image_read() &&
                      test_g17p_storage_read_zero_lod_compile();
      printf("indexed graphics/compute sampler and image reads: %s\n",
             ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--storage-read-lod-only")) {
      const bool ok = test_g17p_storage_read_zero_lod_compile();
      printf("storage-image level-zero compilation: %s\n", ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--indexed-storage-only")) {
      const bool ok = test_indexed_compute_storage();
      printf("indexed compute samples plus RGB10 storage: %s\n",
             ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--sampler-root-only")) {
      const bool ok = test_g17g_combined_sampler_scope() &&
                      test_combined_sampler_root() &&
                      test_combined_sampler_scope() &&
                      test_combined_sampler_integer_scope();
      fprintf(stderr, "native sampler plus push-root lowering: %s\n", ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--g17g-texture-queries")) {
      const bool ok = test_g17g_texture_queries();
      fprintf(stderr, "G17G descriptor query values: %s\n", ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--bias-gradient-only")) {
      const bool ok = test_combined_sampler_biased_and_gradient(false, false) &&
                      test_combined_sampler_biased_and_gradient(true, false) &&
                      test_combined_sampler_biased_and_gradient(false, true) &&
                      test_combined_sampler_scope();
      fprintf(stderr, "API LOD bias and explicit gradient lowering: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--reload-staging-only")) {
      const bool ok = test_g17p_background_reload_staging();
      fprintf(stderr, "G17P background reload staging: %s\n", ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--f16-clear-only")) {
      const bool ok = test_g17p_background_f16_clear_narrowing();
      fprintf(stderr, "G17P f16 background clear narrowing: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--half-norm8-epilog-only")) {
      const bool ok = test_g17p_half_norm8_epilog();
      fprintf(stderr, "G17P half ABI normalized epilog: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--half-rgb10-epilog-only")) {
      const bool ok = test_g17p_half_rgb10_epilog();
      fprintf(stderr, "G17P half ABI RGB10 epilog: %s\n", ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--fragment-io-matrix-only")) {
      const bool ok = test_g17p_fragment_io_matrix();
      fprintf(stderr, "G17P Gallium fragment I/O matrix: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--storage-image-lowering-only")) {
      const bool ok = test_storage_image_lowering();
      fprintf(stderr, "G17P bounded RGBA8UI storage-image lowering: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--sampled-storage-lowering-only")) {
      const bool ok = test_sampled_storage_lowering();
      fprintf(stderr, "G17P sampled RGBA8 storage-image lowering: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--buffer-metadata-only")) {
      const bool ok = test_g17p_buffer_metadata() &&
                      test_g17p_buffer_atomic_address() &&
                      test_g17p_buffer_image_metadata() &&
                      test_buffer_fetch_r32_store() &&
                      test_buffer_fetch_unorm_store_fallback();
      fprintf(stderr, "G17P buffer metadata lowering: %s\n", ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--cube-sample-lowering-only")) {
      const bool ok = test_g17p_native_cube_array_sample();
      fprintf(stderr, "G17P native cube-array sample: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--2d-robustness-only")) {
      const bool ok = test_g17p_2d_robustness_scope();
      fprintf(stderr, "G17P 2D image robustness scope: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--txf-offset-only")) {
      const bool ok = test_txf_offset_lowering() &&
                      test_g17p_3d_runtime_offset_lowering() &&
                      test_g17p_projected_3d_runtime_offset_lowering() &&
                      test_g17p_projected_2d_runtime_offset_scope();
      fprintf(stderr, "G17P texture offset coordinate lowering: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 &&
       !strcmp(argv[1], "--chromium-negated-equality-only")) {
      const struct agx3_graphics_case c = {
         "fs-chromium-negated-equality-select",
         build_fs_chromium_negated_equality_select,
         true,
         .g17p_only = true,
      };
      const bool ok = run_case(&c, AGX_ISA_ENCODING_AGX3B);
      fprintf(stderr, "G17P Chromium negated equality select: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   if (argc == 2 && !strcmp(argv[1], "--chromium-minifloat-branch-only")) {
      const struct agx3_graphics_case c = {
         "fs-chromium-minifloat-branch",
         build_fs_chromium_minifloat_branch,
         true,
         .g17p_only = true,
         .expected_g17p_sampled_fragment = true,
      };
      const bool ok = run_case(&c, AGX_ISA_ENCODING_AGX3B);
      fprintf(stderr, "G17P Chromium minifloat branch: %s\n",
              ok ? "PASS" : "FAIL");
      return ok ? 0 : 1;
   }
   const enum agx_isa_encoding encodings[] = {
      AGX_ISA_ENCODING_AGX3B,
      AGX_ISA_ENCODING_AGX3C,
   };
   bool all_ok = true;

   /* Name whichever check failed. An && chain short-circuits, so the ones
    * after the first failure never ran, and none of these reported anything
    * on their own -- between them a failure was a bare exit status with an
    * empty log, which is indistinguishable from the test not having run. */
#define CHECK_NAMED(label, expr)                                               \
   do {                                                                        \
      if (!(expr)) {                                                           \
         fputs(label " failed\n", stderr);                                     \
         all_ok = false;                                                       \
      }                                                                        \
   } while (0)

   CHECK_NAMED("custom_border_samplers(plain)",
               test_custom_border_samplers(false, 0));
   CHECK_NAMED("custom_border_samplers(custom)",
               test_custom_border_samplers(true, 0));
   CHECK_NAMED("custom_border_samplers(custom,1)",
               test_custom_border_samplers(true, 1));
   CHECK_NAMED("custom_border_samplers(custom,2)",
               test_custom_border_samplers(true, 2));
   CHECK_NAMED("indexed_sampler_rebase_boundaries",
               test_indexed_sampler_native_indices());
   CHECK_NAMED("indexed_array_and_3d_compile",
               test_indexed_array_and_3d_compile());
   CHECK_NAMED("g17p_quad_control_modes", test_g17p_quad_control_modes());
   CHECK_NAMED("g17p_divergent_quad_loop", test_g17p_divergent_quad_loop());
   CHECK_NAMED("g17p_alpha_coverage_boundary",
               test_g17p_alpha_coverage_boundary());
#undef CHECK_NAMED

   if (!test_g17p_gallium_vertex_param_abi()) {
      fputs("Gallium vertex-parameter ABI is not LDDEV aligned\n", stderr);
      all_ok = false;
   }

   if (!test_texture_index_abis()) {
      fputs("legacy and compact-native sampler index ABIs conflated\n", stderr);
      all_ok = false;
   }

   if (!test_g17g_texture_queries() || !test_g17g_combined_sampler_scope() ||
       !test_combined_sampler_scope() || !test_combined_sampler_root() ||
       !test_combined_sampler_integer_scope()) {
      fputs("combined sampler lowering consumed a non-sampler bias\n", stderr);
      all_ok = false;
   }

   if (!test_combined_sampler_biased_and_gradient(false, false) ||
       !test_combined_sampler_biased_and_gradient(true, false) ||
       !test_combined_sampler_biased_and_gradient(false, true)) {
      fputs("combined sampler lowering disowned a 2D/cube API bias or gradient\n",
             stderr);
      all_ok = false;
   }

   if (!test_txf_lowering()) {
      fputs("static TXF descriptor lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_txf_offset_lowering()) {
      fputs("texelFetchOffset coordinate lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_g17p_3d_runtime_offset_lowering()) {
      fputs("G17P 3D runtime offset was not lowered to coordinates\n", stderr);
      all_ok = false;
   }

   if (!test_g17p_projected_3d_runtime_offset_lowering()) {
      fputs("G17P projected 3D runtime offset was not composed before divide\n",
            stderr);
      all_ok = false;
   }

   if (!test_g17p_projected_2d_runtime_offset_scope()) {
      fputs("G17P projected 2D runtime offset escaped its native path\n",
            stderr);
      all_ok = false;
   }

   if (!test_g17p_shadow_runtime_offset_compile()) {
      fputs("G17P shadow runtime offset compile failed\n", stderr);
      all_ok = false;
   }

   if (!test_indexed_multisample_fetch()) {
      fputs("indexed multisample resolve lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_indexed_dynamic_fetch_lod()) {
      fputs("dynamic indexed texel-fetch LOD lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_g17p_fetch_lod_compile(0) ||
       !test_g17p_fetch_lod_compile(16) ||
       !test_g17p_fetch_lod_compile(32)) {
      fputs("G17P nonzero texel-fetch LOD lost its uint staging\n", stderr);
      all_ok = false;
   }

   if (!test_indexed_compute_sample()) {
      fputs("indexed compute sampler lowering mismatch\n", stderr);
      all_ok = false;
   }
   if (!test_indexed_compute_cube_array_shadow_level()) {
      fputs("indexed compute cube-array shadow level refused\n", stderr);
      all_ok = false;
   }

   if (!test_indexed_storage_image_read()) {
      fputs("indexed storage-image read lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_g17p_storage_read_zero_lod_compile()) {
      fputs("storage-image constant level zero used runtime LOD staging\n", stderr);
      all_ok = false;
   }

   if (!test_indexed_compute_storage()) {
      fputs("indexed compute storage lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_storage_image_lowering()) {
      fputs("bounded storage-image lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_gallium_storage_write_lowering()) {
      fputs("Gallium scalar storage-write lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_sampled_storage_lowering()) {
      fputs("sampled storage-image lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_g17p_buffer_metadata() || !test_g17p_buffer_atomic_address() ||
       !test_g17p_buffer_image_metadata() || !test_buffer_fetch_r32_store() ||
       !test_buffer_fetch_unorm_store_fallback()) {
      fputs("buffer metadata lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_g17p_formatless_robust_image_load()) {
      fputs("formatless robust image load lowering mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_g17p_robust_texel_fetch_value()) {
      fputs("robust texel fetch value mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_g17p_query_lod_compile()) {
      fputs("native LOD query compile mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_g17p_cube_array_shadow_dump()) {
      fputs("cube-array shadow dump compile failed\n", stderr);
      return 1;
   }
   if (!test_g17p_explicit_texture_transactions(false) ||
       !test_g17p_explicit_texture_transactions(true)) {
      fputs("explicit texture transaction compile failed\n", stderr);
      return 1;
   }
   if (!test_g17p_compute_form_vertex_shader()) {
      fputs("compute-form vertex shader compile failed\n", stderr);
      return 1;
   }
   if (!test_g17p_cube_array_shadow_level_varyings(false) ||
       !test_g17p_cube_array_shadow_level_varyings(true)) {
      fprintf(stderr, "agx3_graphics: cube-array shadow level/bias from varyings refused\n");
      return 1;
   }

   if (!test_g17p_xfb_geometry_programs()) {
      fputs("transform feedback geometry programs compile failed\n", stderr);
      return 1;
   }
   if (!test_g17p_native_cube_array_sample()) {
      fputs("native cube-array sample mismatch\n", stderr);
      all_ok = false;
   }

   if (!test_g17p_2d_robustness_scope()) {
      fputs("2D image robustness scope mismatch\n", stderr);
      all_ok = false;
   }

   /* Each of these reports on its own only when it can say something useful
    * about *how* it differed. Name the one that failed regardless, so a
    * failure is never a bare exit status with an empty log. */
#define RUN(fn)                                                                \
   do {                                                                        \
      if (!fn()) {                                                             \
         fputs(#fn " failed\n", stderr);                                       \
         all_ok = false;                                                       \
      }                                                                        \
   } while (0)

   RUN(test_g17g_array_fetch);
   RUN(test_g17g_cube_shadow_lod);
   RUN(test_g17g_xfb_add);
   RUN(test_agx3_vertex_without_preamble_has_none);
   RUN(test_g17p_eot_block_store);
   RUN(test_indexed_sampler_sets);
   RUN(test_g17p_layered_eot_block_store);
   RUN(test_g17p_multisample_eot_block_store);
   RUN(test_g17p_background_reload_staging);
   RUN(test_g17p_background_clear);
   RUN(test_g17p_discard_word);
   RUN(test_g17p_background_f16_clear_narrowing);
   RUN(test_g17p_half_norm8_epilog);
   RUN(test_g17p_half_rgb10_epilog);
   RUN(test_g17p_three_shadow_sample_references);
   if (!test_g17p_array_shadow_register_lod(false) ||
       !test_g17p_array_shadow_register_lod(true)) {
      fputs("test_g17p_array_shadow_register_lod failed\n", stderr);
      all_ok = false;
   }
   if (!test_g17p_array_shadow_constant_lod(false) ||
       !test_g17p_array_shadow_constant_lod(true)) {
      fputs("test_g17p_array_shadow_constant_lod failed\n", stderr);
      all_ok = false;
   }
   RUN(test_g17p_fragment_io_matrix);
   RUN(test_g17p_unread_colour_waits);
   RUN(test_g17p_halfword_tile_widening);
   RUN(test_g17p_blended_payloads_beyond_bank);
   RUN(test_g17p_pbe_to_texture_fence);
   RUN(test_g17p_clustered_minmax_width);
   RUN(test_agx3_native_relative_shuffles);
   RUN(test_g17p_half_rgb10_epilog);
   RUN(test_g17p_frag_sidefx_late_depth_once);
   RUN(test_agx3_heavy_branch_loop_unroll_limit);
   RUN(test_g17c_txf_layer_saturation);
   RUN(test_agx3_tiny_push_vertex_preamble_inlined);
   RUN(test_agx3_vertex_set_pointer_uniform_halfs);
   RUN(test_agx3_packed_byte_word_tile_store);
   RUN(test_agx3d_texture_early_lowering);
#undef RUN

   for (unsigned e = 0; e < ARRAY_SIZE(encodings); ++e) {
      for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
         if (cases[i].g17p_only && encodings[e] != AGX_ISA_ENCODING_AGX3B)
            continue;
         if (!run_case(&cases[i], encodings[e])) {
            fprintf(stderr, "%s %s: case failed\n", enc_name(encodings[e]),
                    cases[i].name);
            all_ok = false;
         }
      }
   }

   if (!all_ok)
      return 1;

   puts("agx3_graphics: vertex and fragment shaders compile");
   return 0;
}

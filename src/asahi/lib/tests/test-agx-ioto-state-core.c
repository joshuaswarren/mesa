/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Differential and fail-closed contract tests for the direct-write G17P
 * compiler LDSHDR and SPECTPR packers in agx_ioto_state_core.c, and for the
 * fragment SPECLM imageblock record.
 */

#include "../agx_ioto_core.h"
#include "../agx_ioto_state_core.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint64_t valid_encodes;
static uint64_t buffer_preserving_rejections;
static uint64_t null_rejections;

#define CHECK(x)                                                               \
   do {                                                                        \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                \
         return 1;                                                             \
      }                                                                        \
   } while (0)

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

static void
fill_canary(uint8_t *buf, size_t len, uint32_t salt)
{
   for (size_t i = 0; i < len; ++i)
      buf[i] = (uint8_t)(0xa5u ^ (uint8_t)(i * 29u + salt * 17u));
}

/* Independent reference implementing the pre-cleanup staged little-endian
 * serialization for agx_ioto_g17p_compiler_ldshdr_pack(). */
static bool
ref_g17p_compiler_ldshdr_pack(uint64_t shader_va, uint32_t fetch_span,
                              void *output, size_t capacity)
{
   static const uint8_t e4[4] = {2, 0, 1, 3};

   if (!output || capacity < AGX_IOTO_LDSHDR_SIZE || fetch_span > 3 ||
       shader_va > UINT64_C(0xffffffffffff) ||
       (shader_va & UINT64_C(0x3f)) != 0)
      return false;

   const uint16_t words[5] = {
      UINT16_C(0x0177),
      UINT16_C(0x002a) | (uint16_t)(((shader_va >> 6) & 0x1ff) << 7),
      (uint16_t)(shader_va >> 15),
      (uint16_t)(shader_va >> 31),
      (uint16_t)(((shader_va >> 47) & 1) | ((uint16_t)e4[fetch_span] << 1)),
   };
   uint8_t staged[AGX_IOTO_LDSHDR_SIZE];
   for (unsigned i = 0; i < 5; ++i) {
      staged[2 * i] = (uint8_t)words[i];
      staged[2 * i + 1] = (uint8_t)(words[i] >> 8);
   }

   memcpy(output, staged, sizeof(staged));
   return true;
}

/* Independent reference implementing the pre-cleanup staged little-endian
 * serialization for agx_ioto_g17p_compiler_spectpr_pack(). */
static bool
ref_g17p_compiler_spectpr_pack(uint32_t spill_bytes, uint32_t d_value,
                               void *output, size_t capacity)
{
   if (!output || capacity < AGX_IOTO_SPECTPR_SIZE ||
       spill_bytes > (uint32_t)UINT16_MAX * 4 || d_value > UINT32_MAX - 3)
      return false;

   const uint16_t spill_quads = (uint16_t)((spill_bytes + 3) >> 2);
   const uint16_t d_rounded =
      (uint16_t)(((((uint64_t)d_value + 3) >> 2) + 1) & UINT64_C(0xffe));

   const uint16_t words[5] = {
      UINT16_C(0x00f7),
      UINT16_C(0x002a) | (uint16_t)((d_rounded & 2) << 14),
      (uint16_t)(((d_rounded >> 2) & 0x3ff) | ((spill_quads & 0x3f) << 10)),
      (uint16_t)(((spill_quads >> 6) & 0x3ff) | ((spill_quads & 0x3f) << 10)),
      (uint16_t)((spill_quads >> 6) & 0x3ff),
   };
   uint8_t staged[AGX_IOTO_SPECTPR_SIZE];
   for (unsigned i = 0; i < 5; ++i) {
      staged[2 * i] = (uint8_t)words[i];
      staged[2 * i + 1] = (uint8_t)(words[i] >> 8);
   }

   memcpy(output, staged, sizeof(staged));
   return true;
}

static int
check_ldshdr_valid_case(uint64_t shader_va, uint32_t fetch_span,
                        size_t capacity, size_t offset, uint32_t salt)
{
   uint8_t direct_buf[48];
   uint8_t ref_buf[48];
   uint8_t generic_buf[48];
   uint8_t canary[48];

   CHECK(offset + capacity <= sizeof(direct_buf));
   fill_canary(canary, sizeof(canary), salt);
   memcpy(direct_buf, canary, sizeof(canary));
   memcpy(ref_buf, canary, sizeof(canary));
   memcpy(generic_buf, canary, sizeof(canary));

   CHECK(agx_ioto_g17p_compiler_ldshdr_pack(shader_va, fetch_span,
                                            direct_buf + offset, capacity));
   CHECK(ref_g17p_compiler_ldshdr_pack(shader_va, fetch_span, ref_buf + offset,
                                       capacity));

   struct agx_ioto_state state = {0};
   state.f[0] = shader_va;
   state.f[1] = fetch_span;
   state.f[23] = 1;
   CHECK(agx_ioto_state_pack_ldshdr(AGX_CHIP_G17P, &state, generic_buf + offset,
                                    capacity));

   CHECK(memcmp(direct_buf, ref_buf, sizeof(direct_buf)) == 0);
   CHECK(memcmp(direct_buf, generic_buf, sizeof(direct_buf)) == 0);
   CHECK(memcmp(direct_buf, canary, offset) == 0);
   CHECK(memcmp(direct_buf + offset + AGX_IOTO_LDSHDR_SIZE,
                canary + offset + AGX_IOTO_LDSHDR_SIZE,
                sizeof(direct_buf) - offset - AGX_IOTO_LDSHDR_SIZE) == 0);

   struct agx_ioto_record decoded;
   size_t consumed = 0;
   CHECK(agx_ioto_decode_one(AGX_CHIP_G17P, direct_buf + offset,
                             AGX_IOTO_LDSHDR_SIZE, &decoded, &consumed));
   CHECK(decoded.kind == AGX_IOTO_KIND_LDSHDR_EP0 &&
         consumed == AGX_IOTO_LDSHDR_SIZE);

   ++valid_encodes;
   return 0;
}

static int
check_ldshdr_reject_case(uint64_t shader_va, uint32_t fetch_span,
                         size_t capacity, uint32_t salt)
{
   uint8_t direct_buf[32];
   uint8_t ref_buf[32];
   uint8_t generic_buf[32];
   uint8_t before[32];

   CHECK(capacity <= sizeof(direct_buf));
   fill_canary(before, sizeof(before), salt);
   memcpy(direct_buf, before, sizeof(before));
   memcpy(ref_buf, before, sizeof(before));
   memcpy(generic_buf, before, sizeof(before));

   CHECK(!agx_ioto_g17p_compiler_ldshdr_pack(shader_va, fetch_span, direct_buf,
                                             capacity));
   CHECK(!ref_g17p_compiler_ldshdr_pack(shader_va, fetch_span, ref_buf,
                                        capacity));

   struct agx_ioto_state state = {0};
   state.f[0] = shader_va;
   state.f[1] = fetch_span;
   state.f[23] = 1;
   CHECK(!agx_ioto_state_pack_ldshdr(AGX_CHIP_G17P, &state, generic_buf,
                                     capacity));

   CHECK(memcmp(direct_buf, before, sizeof(before)) == 0);
   CHECK(memcmp(ref_buf, before, sizeof(before)) == 0);
   CHECK(memcmp(generic_buf, before, sizeof(before)) == 0);

   ++buffer_preserving_rejections;
   return 0;
}

static int
check_spectpr_valid_case(uint32_t spill_bytes, uint32_t d_value,
                         size_t capacity, size_t offset, uint32_t salt)
{
   uint8_t direct_buf[48];
   uint8_t ref_buf[48];
   uint8_t generic_buf[48];
   uint8_t canary[48];

   CHECK(offset + capacity <= sizeof(direct_buf));
   fill_canary(canary, sizeof(canary), salt);
   memcpy(direct_buf, canary, sizeof(canary));
   memcpy(ref_buf, canary, sizeof(canary));
   memcpy(generic_buf, canary, sizeof(canary));

   CHECK(agx_ioto_g17p_compiler_spectpr_pack(spill_bytes, d_value,
                                             direct_buf + offset, capacity));
   CHECK(ref_g17p_compiler_spectpr_pack(spill_bytes, d_value, ref_buf + offset,
                                        capacity));

   struct agx_ioto_state state;
   CHECK(agx_ioto_g17p_compiler_spectpr_build(spill_bytes, d_value, &state));
   CHECK(agx_ioto_state_pack_spectpr(AGX_CHIP_G17P, &state,
                                     generic_buf + offset, capacity));

   CHECK(memcmp(direct_buf, ref_buf, sizeof(direct_buf)) == 0);
   CHECK(memcmp(direct_buf, generic_buf, sizeof(direct_buf)) == 0);
   CHECK(memcmp(direct_buf, canary, offset) == 0);
   CHECK(memcmp(direct_buf + offset + AGX_IOTO_SPECTPR_SIZE,
                canary + offset + AGX_IOTO_SPECTPR_SIZE,
                sizeof(direct_buf) - offset - AGX_IOTO_SPECTPR_SIZE) == 0);

   struct agx_ioto_record decoded;
   size_t consumed = 0;
   CHECK(agx_ioto_decode_one(AGX_CHIP_G17P, direct_buf + offset,
                             AGX_IOTO_SPECTPR_SIZE, &decoded, &consumed));
   CHECK(decoded.kind == AGX_IOTO_KIND_SPECTPR_EP0 &&
         consumed == AGX_IOTO_SPECTPR_SIZE);

   ++valid_encodes;
   return 0;
}

static int
check_spectpr_reject_case(uint32_t spill_bytes, uint32_t d_value,
                          size_t capacity, uint32_t salt)
{
   uint8_t direct_buf[32];
   uint8_t ref_buf[32];
   uint8_t generic_buf[32];
   uint8_t before[32];

   CHECK(capacity <= sizeof(direct_buf));
   fill_canary(before, sizeof(before), salt);
   memcpy(direct_buf, before, sizeof(before));
   memcpy(ref_buf, before, sizeof(before));
   memcpy(generic_buf, before, sizeof(before));

   CHECK(!agx_ioto_g17p_compiler_spectpr_pack(spill_bytes, d_value, direct_buf,
                                              capacity));
   CHECK(!ref_g17p_compiler_spectpr_pack(spill_bytes, d_value, ref_buf,
                                         capacity));

   struct agx_ioto_state state;
   struct agx_ioto_state state_before;
   memset(&state, 0x5c, sizeof(state));
   memcpy(&state_before, &state, sizeof(state));
   if (agx_ioto_g17p_compiler_spectpr_build(spill_bytes, d_value, &state)) {
      /* Inputs were valid, so capacity must be insufficient. */
      CHECK(capacity < AGX_IOTO_SPECTPR_SIZE);
      CHECK(!agx_ioto_state_pack_spectpr(AGX_CHIP_G17P, &state, generic_buf,
                                         capacity));
   } else {
      CHECK(memcmp(&state, &state_before, sizeof(state)) == 0);
   }

   CHECK(memcmp(direct_buf, before, sizeof(before)) == 0);
   CHECK(memcmp(ref_buf, before, sizeof(before)) == 0);
   CHECK(memcmp(generic_buf, before, sizeof(before)) == 0);

   ++buffer_preserving_rejections;
   return 0;
}

static int
test_g17p_compiler_ldshdr_differential(void)
{
   uint64_t valid_vas[65];
   size_t nr_valid_vas = 0;

   /* Single-bit sweep across every encoded VA bit [6..47] (42 cases). */
   for (unsigned bit = 6; bit <= 47; ++bit)
      valid_vas[nr_valid_vas++] = UINT64_C(1) << bit;

   /* Boundary, word-split, alternating, and realistic G17P shader VAs. */
   static const uint64_t extra_valid_vas[] = {
      UINT64_C(0x0),
      UINT64_C(0x0000000000c0),
      UINT64_C(0x000000007fc0),
      UINT64_C(0x000000008040),
      UINT64_C(0x00007fff8000),
      UINT64_C(0x00007fffffc0),
      UINT64_C(0x000080000040),
      UINT64_C(0x7fff80000000),
      UINT64_C(0x7fffffffffc0),
      UINT64_C(0x800000000040),
      UINT64_C(0xffffffffff80),
      UINT64_C(0xffffffffffc0),
      UINT64_C(0xaaaaaaaaaac0),
      UINT64_C(0x555555555540),
      UINT64_C(0x123456789ac0),
      UINT64_C(0xfedcba987640),
      UINT64_C(0x010000000000),
      UINT64_C(0x010000130640),
      UINT64_C(0x0100000006c0),
      UINT64_C(0xdeadbeef0040),
      UINT64_C(0x00cafe000080),
      UINT64_C(0x3c3c3c3c3c00),
      UINT64_C(0xc3c3c3c3c3c0),
   };
   for (size_t i = 0; i < ARRAY_SIZE(extra_valid_vas); ++i)
      valid_vas[nr_valid_vas++] = extra_valid_vas[i];
   CHECK(nr_valid_vas == ARRAY_SIZE(valid_vas));

   static const size_t valid_caps[] = {
      AGX_IOTO_LDSHDR_SIZE,
      AGX_IOTO_LDSHDR_SIZE + 1,
      16,
      32,
   };
   static const size_t offsets[] = {0, 1, 3, 7};

   uint32_t salt = 1;
   for (size_t i = 0; i < nr_valid_vas; ++i) {
      for (uint32_t span = 0; span <= 3; ++span) {
         for (size_t c = 0; c < ARRAY_SIZE(valid_caps); ++c) {
            CHECK(check_ldshdr_valid_case(valid_vas[i], span, valid_caps[c],
                                          offsets[(i + span + c) & 3],
                                          salt++) == 0);
         }
      }
   }

   static const uint64_t invalid_vas[] = {
      UINT64_C(0x1),
      UINT64_C(0x2),
      UINT64_C(0x3),
      UINT64_C(0x4),
      UINT64_C(0x7),
      UINT64_C(0x8),
      UINT64_C(0x10),
      UINT64_C(0x1f),
      UINT64_C(0x20),
      UINT64_C(0x3f),
      UINT64_C(0x10000000001),
      UINT64_C(0x1000000003f),
      UINT64_C(0xffffffffffc1),
      UINT64_C(0xffffffffffff),
      UINT64_C(1) << 48,
      (UINT64_C(1) << 48) | UINT64_C(0x40),
      (UINT64_C(1) << 48) | UINT64_C(0x1),
      UINT64_C(1) << 49,
      UINT64_C(1) << 52,
      UINT64_C(1) << 63,
      (UINT64_C(1) << 63) | UINT64_C(0x40),
      UINT64_C(0x0001ffffffffffc0),
      UINT64_MAX & ~UINT64_C(0x3f),
      UINT64_MAX,
   };
   static const uint32_t invalid_spans[] = {
      4, 5, 6, 7, 8, 16, 255, UINT32_MAX,
   };
   static const uint32_t mixed_spans[] = {
      0, 1, 2, 3, 4, 5, 16, UINT32_MAX,
   };
   static const size_t short_caps[] = {0, 1, 2, 4, 8, 9};
   static const size_t mixed_caps[] = {0, 1, 9, AGX_IOTO_LDSHDR_SIZE, 32};

   /* Invalid VAs across mixed fetch spans and capacities. */
   for (size_t i = 0; i < ARRAY_SIZE(invalid_vas); ++i) {
      for (size_t s = 0; s < ARRAY_SIZE(mixed_spans); ++s) {
         for (size_t c = 0; c < ARRAY_SIZE(mixed_caps); ++c) {
            CHECK(check_ldshdr_reject_case(invalid_vas[i], mixed_spans[s],
                                           mixed_caps[c], salt++) == 0);
         }
      }
   }

   /* Valid VAs with invalid fetch spans. */
   for (size_t i = 0; i < 20; ++i) {
      for (size_t s = 0; s < ARRAY_SIZE(invalid_spans); ++s) {
         for (size_t c = 0; c < ARRAY_SIZE(mixed_caps); ++c) {
            CHECK(check_ldshdr_reject_case(valid_vas[i * 3], invalid_spans[s],
                                           mixed_caps[c], salt++) == 0);
         }
      }
   }

   /* Valid VAs and valid fetch spans with insufficient capacity. */
   for (size_t i = 0; i < 20; ++i) {
      for (uint32_t span = 0; span <= 3; ++span) {
         for (size_t c = 0; c < ARRAY_SIZE(short_caps); ++c) {
            CHECK(check_ldshdr_reject_case(valid_vas[i * 3], span,
                                           short_caps[c], salt++) == 0);
         }
      }
   }

   /* NULL output rejections across valid and invalid inputs/capacities. */
   static const size_t null_caps[] = {0, 9, AGX_IOTO_LDSHDR_SIZE, 32};
   static const uint32_t null_spans[] = {0, 1, 2, 3, 4, UINT32_MAX};
   for (size_t i = 0; i < 16; ++i) {
      const uint64_t va = (i < 10) ? valid_vas[i * 6] : invalid_vas[i - 10];
      for (size_t s = 0; s < ARRAY_SIZE(null_spans); ++s) {
         for (size_t c = 0; c < ARRAY_SIZE(null_caps); ++c) {
            CHECK(!agx_ioto_g17p_compiler_ldshdr_pack(va, null_spans[s], NULL,
                                                      null_caps[c]));
            CHECK(!ref_g17p_compiler_ldshdr_pack(va, null_spans[s], NULL,
                                                 null_caps[c]));
            ++null_rejections;
         }
      }
   }

   return 0;
}

static int
test_g17p_compiler_spectpr_differential(void)
{
   static const uint32_t valid_spills[] = {
      0,      1,      2,      3,      4,      5,      6,      7,
      8,      12,     16,     63,     64,     65,     252,    253,
      255,    256,    257,    1024,   4092,   4096,   4097,   16380,
      16384,  65532,  65536,  131072, 262136, 262137, 262139, 262140,
   };
   static const uint32_t valid_ds[] = {
      0,          1,          2,          3,          4,          5,
      6,          7,          8,          9,          12,         15,
      16,         24,         32,         64,         128,        256,
      1024,       4092,       4096,       8184,       8185,       8188,
      8189,       8192,       16376,      16377,      16380,      16384,
      32768,      65536,      0x7ffffffc, 0xfffffffc,
   };
   static const size_t valid_caps[] = {
      AGX_IOTO_SPECTPR_SIZE,
      AGX_IOTO_SPECTPR_SIZE + 1,
      16,
      32,
   };
   static const size_t offsets[] = {0, 1, 3, 7};

   uint32_t salt = 10000;
   for (size_t i = 0; i < ARRAY_SIZE(valid_spills); ++i) {
      for (size_t j = 0; j < ARRAY_SIZE(valid_ds); ++j) {
         CHECK(check_spectpr_valid_case(valid_spills[i], valid_ds[j],
                                        valid_caps[(i + j) & 1],
                                        offsets[(i + j) & 3], salt++) == 0);
         CHECK(check_spectpr_valid_case(valid_spills[i], valid_ds[j],
                                        valid_caps[2 + ((i + j) & 1)],
                                        offsets[(i + j + 2) & 3],
                                        salt++) == 0);
      }
   }

   static const uint32_t invalid_spills[] = {
      262141,     262142,     262143,     262144,     0x40004,
      0x80000,    0x100000,   0x80000000, 0xfffffffc, UINT32_MAX,
   };
   static const uint32_t invalid_ds[] = {
      UINT32_MAX - 2,
      UINT32_MAX - 1,
      UINT32_MAX,
   };
   static const uint32_t mixed_ds[] = {
      0,    1,          4,          8,          1024,       8188,
      8192, 0x7ffffffc, 0xfffffffc, 0xfffffffd, 0xfffffffe, UINT32_MAX,
   };
   static const size_t short_caps[] = {0, 1, 2, 4, 8, 9};
   static const size_t mixed_caps[] = {0, 1, 9, AGX_IOTO_SPECTPR_SIZE, 32};

   /* Invalid spill_bytes across mixed d_values and capacities. */
   for (size_t i = 0; i < ARRAY_SIZE(invalid_spills); ++i) {
      for (size_t j = 0; j < ARRAY_SIZE(mixed_ds); ++j) {
         for (size_t c = 0; c < ARRAY_SIZE(mixed_caps); ++c) {
            CHECK(check_spectpr_reject_case(invalid_spills[i], mixed_ds[j],
                                            mixed_caps[c], salt++) == 0);
         }
      }
   }

   /* Valid spill_bytes with invalid d_values. */
   for (size_t i = 0; i < ARRAY_SIZE(valid_spills); ++i) {
      for (size_t j = 0; j < ARRAY_SIZE(invalid_ds); ++j) {
         for (size_t c = 0; c < ARRAY_SIZE(mixed_caps); ++c) {
            CHECK(check_spectpr_reject_case(valid_spills[i], invalid_ds[j],
                                            mixed_caps[c], salt++) == 0);
         }
      }
   }

   /* Valid spill_bytes and valid d_values with insufficient capacity. */
   for (size_t i = 0; i < ARRAY_SIZE(valid_spills); ++i) {
      for (size_t j = 0; j < 20; ++j) {
         for (size_t c = 0; c < ARRAY_SIZE(short_caps); ++c) {
            CHECK(check_spectpr_reject_case(valid_spills[i], valid_ds[j],
                                            short_caps[c], salt++) == 0);
         }
      }
   }

   /* NULL output rejections across valid and invalid inputs/capacities. */
   static const size_t null_caps[] = {0, 9, AGX_IOTO_SPECTPR_SIZE, 32};
   static const uint32_t null_spills[] = {
      0, 4, 256, 4096, 65536, 262139, 262140, 262141, 262144, 0x100000,
      0x80000000, UINT32_MAX,
   };
   static const uint32_t null_ds[] = {
      0, 4, 8188, 16380, 0xfffffffc, 0xfffffffd, 0xfffffffe, UINT32_MAX,
   };
   for (size_t i = 0; i < ARRAY_SIZE(null_spills); ++i) {
      for (size_t j = 0; j < ARRAY_SIZE(null_ds); ++j) {
         CHECK(!agx_ioto_g17p_compiler_spectpr_build(null_spills[i], null_ds[j],
                                                     NULL));
         ++null_rejections;
         for (size_t c = 0; c < ARRAY_SIZE(null_caps); ++c) {
            CHECK(!agx_ioto_g17p_compiler_spectpr_pack(
               null_spills[i], null_ds[j], NULL, null_caps[c]));
            CHECK(!ref_g17p_compiler_spectpr_pack(null_spills[i], null_ds[j],
                                                  NULL, null_caps[c]));
            ++null_rejections;
         }
      }
   }

   return 0;
}

/* The fragment SPECLM imageblock record. 32x16 tiles of 24-byte samples at 4x
 * are 48 KiB; our J773g (G16G) trace of a deferred 4x pass programs exactly
 * this record on its fragment, background and end-of-tile programs. Larger
 * tiles stay refused, and the output is left untouched. */
static int
test_g17p_fragment_speclm(void)
{
   static const uint8_t trace_32x16_24b_4x[AGX_IOTO_SPECLM_SIZE] = {
      0x77, 0x00, 0xaa, 0x06, 0x43, 0x00, 0x00, 0x00,
   };
   uint8_t out[AGX_IOTO_SPECLM_SIZE];

   const struct agx_ioto_g17p_imageblock tile48k = {
      .sample_size_B = 24, .nr_samples = 4, .tile_size_px = 32 * 16,
   };
   CHECK(tile48k.sample_size_B * tile48k.nr_samples * tile48k.tile_size_px ==
         AGX_NATIVE_TILE_BUFFER_48K_BYTES);
   CHECK(agx_ioto_g17p_fragment_speclm_build(&tile48k, out, sizeof(out)));
   CHECK(memcmp(out, trace_32x16_24b_4x, sizeof(out)) == 0);
   ++valid_encodes;

   const struct agx_ioto_g17p_imageblock too_large[] = {
      /* 64 KiB */
      {.sample_size_B = 32, .nr_samples = 4, .tile_size_px = 32 * 16},
      /* 96 KiB */
      {.sample_size_B = 24, .nr_samples = 4, .tile_size_px = 32 * 32},
      /* 56 KiB */
      {.sample_size_B = 56, .nr_samples = 1, .tile_size_px = 32 * 32},
   };
   for (size_t i = 0; i < ARRAY_SIZE(too_large); ++i) {
      memset(out, 0xa5, sizeof(out));
      CHECK(!agx_ioto_g17p_fragment_speclm_build(&too_large[i], out,
                                                 sizeof(out)));
      for (size_t b = 0; b < sizeof(out); ++b)
         CHECK(out[b] == 0xa5);
      ++buffer_preserving_rejections;
   }

   return 0;
}

int
main(void)
{
   CHECK(test_g17p_compiler_ldshdr_differential() == 0);
   CHECK(test_g17p_compiler_spectpr_differential() == 0);
   CHECK(test_g17p_fragment_speclm() == 0);
   printf("agx_ioto_state_core: %llu valid_encodes, "
          "%llu buffer_preserving_rejections, %llu null_rejections, "
          "0 mismatches\n",
          (unsigned long long)valid_encodes,
          (unsigned long long)buffer_preserving_rejections,
          (unsigned long long)null_rejections);
   return 0;
}

#include "agx3_rt_gen1_default.h"
#include "agx3_binder.h"
#include "agx3_pack.h"
#include <string.h>
#define KV(f, v) {(f), (v)}
static const uint8_t default_body_prefix_end[4] = {0x0e, 0x00, 0x00, 0x00};
bool
agx3_rt_gen1_default_body_generate(uint8_t out[AGX3_RT_GEN1_DEFAULT_BODY_MAX_SIZE], size_t *length)
{
   if (!out || !length) return false;
   uint8_t staged[AGX3_RT_GEN1_DEFAULT_BODY_MAX_SIZE] = {0};
   size_t used = 0;
   memcpy(staged, default_body_prefix_end, sizeof(default_body_prefix_end));
   for (unsigned i = 2; i < 32; ++i) staged[2 * i] = 0x06;
   used = 64;
#define OP(name, ep, ...) do { \
   const struct agx3_field_kv fields[] = {__VA_ARGS__}; uint8_t n; \
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, name, ep, fields, \
      sizeof(fields) / sizeof(fields[0]), NULL, 0, staged + used, &n)) return false; \
   used += n; \
} while (0)
#define OP0(name, ep) do { uint8_t n; \
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, name, ep, NULL, 0, NULL, 0, staged + used, &n)) return false; \
   used += n; \
} while (0)
   OP(AGX3_MN_rld, 0, KV(9, 2), KV(10, 1), KV(12, 1), KV(21, 57), KV(23, 7), KV(24, 1), KV(38, 1));
   OP(AGX3_MN_rld, 0, KV(2, 2), KV(9, 2), KV(10, 1), KV(12, 2), KV(21, 7), KV(23, 7), KV(24, 1), KV(38, 1));
   OP(AGX3_MN_imadd, 0, KV(9, 2), KV(10, 2), KV(23, 2), KV(24, 1), KV(27, 1), KV(32, 4), KV(38, 3), KV(39, 1), KV(53, 1), KV(54, 1), KV(60, 1));
   OP(AGX3_MN_clr, 0, KV(2, 3), KV(9, 2), KV(10, 1));
   OP(AGX3_MN_ldscs, 0, KV(2, 3), KV(9, 2), KV(12, 6), KV(16, 2), KV(23, 2), KV(24, 2), KV(38, 2), KV(39, 2), KV(54, 1), KV(61, 1), KV(68, 1), KV(70, 1), KV(71, 1), KV(80, 1));
   OP(AGX3_MN_icmp, 1, KV(2, 3), KV(9, 2), KV(10, 1), KV(14, 1), KV(24, 1), KV(25, 1), KV(39, 1), KV(42, 4));
   OP(AGX3_MN_cndst, 0, KV(12, 1));
   OP(AGX3_MN_bn, 0, KV(0, 206));
   OP(AGX3_MN_ldscs, 0, KV(2, 5), KV(9, 2), KV(10, 1), KV(12, 6), KV(23, 2), KV(24, 2), KV(32, 8), KV(38, 3), KV(39, 2), KV(54, 1), KV(68, 1), KV(71, 198));
   OP(AGX3_MN_rld, 0, KV(2, 4), KV(9, 2), KV(10, 1), KV(12, 1), KV(21, 58), KV(23, 7), KV(24, 1), KV(38, 1));
   OP(AGX3_MN_icmp, 1, KV(2, 5), KV(9, 2), KV(10, 1), KV(13, 1), KV(14, 1), KV(24, 1), KV(25, 1), KV(39, 1), KV(42, 4));
   OP(AGX3_MN_bit_and, 1, KV(2, 4), KV(9, 2), KV(10, 1), KV(11, 1), KV(16, 4), KV(23, 2), KV(24, 1), KV(27, 1), KV(31, 6), KV(37, 3), KV(38, 1), KV(44, 1));
   OP(AGX3_MN_ldimm, 1, KV(2, 5), KV(9, 2), KV(10, 1), KV(14, 8192));
   OP(AGX3_MN_cndst, 0, KV(0, 1), KV(12, 1));
   OP(AGX3_MN_bn, 0, KV(0, 102));
   OP(AGX3_MN_ldscs, 0, KV(2, 6), KV(9, 2), KV(10, 2), KV(12, 6), KV(23, 1), KV(24, 1), KV(38, 1), KV(39, 1), KV(54, 1), KV(68, 1), KV(71, 24588));
   OP(AGX3_MN_bit_and, 1, KV(2, 10), KV(9, 2), KV(10, 2), KV(16, 4), KV(23, 2), KV(24, 1), KV(31, 5), KV(37, 3), KV(38, 1));
   OP(AGX3_MN_imadd, 0, KV(2, 10), KV(9, 2), KV(10, 2), KV(16, 7), KV(23, 2), KV(24, 1), KV(27, 1), KV(31, 10), KV(38, 2), KV(39, 2), KV(42, 1), KV(53, 3), KV(54, 2), KV(65, 1));
   OP(AGX3_MN_bit_and, 1, KV(2, 14), KV(9, 2), KV(10, 2), KV(16, 4), KV(23, 2), KV(24, 1), KV(29, 31), KV(37, 1), KV(38, 2));
   OP(AGX3_MN_ldscs, 0, KV(2, 6), KV(9, 2), KV(10, 2), KV(12, 6), KV(16, 14), KV(23, 2), KV(24, 2), KV(31, 10), KV(38, 2), KV(39, 2), KV(54, 1), KV(68, 1), KV(70, 3), KV(71, 260));
   OP(AGX3_MN_bit_or, 1, KV(2, 6), KV(9, 2), KV(10, 2), KV(16, 6), KV(23, 2), KV(24, 2), KV(27, 1), KV(31, 10), KV(37, 3), KV(38, 2), KV(49, 1));
   OP(AGX3_MN_ldimm, 1, KV(2, 5), KV(9, 2), KV(10, 1), KV(11, 1), KV(14, 24576));
   OP(AGX3_MN_stscs, 0, KV(2, 6), KV(9, 2), KV(10, 2), KV(13, 1), KV(16, 14), KV(23, 2), KV(24, 2), KV(27, 1), KV(31, 10), KV(38, 2), KV(39, 2), KV(42, 1), KV(54, 1), KV(68, 1), KV(70, 3), KV(71, 260));
   OP(AGX3_MN_cndend, 0, KV(10, 1), KV(11, 2), KV(12, 2));
   OP(AGX3_MN_iadd, 0, KV(2, 4), KV(9, 2), KV(10, 1), KV(16, 5), KV(23, 2), KV(24, 1), KV(27, 1), KV(31, 4), KV(38, 2), KV(39, 1), KV(42, 1));
   OP(AGX3_MN_isub, 0, KV(2, 2), KV(9, 2), KV(10, 1), KV(16, 2), KV(23, 2), KV(24, 1), KV(27, 1), KV(31, 3), KV(38, 2), KV(39, 1), KV(42, 1));
   OP(AGX3_MN_clr, 0, KV(2, 5), KV(9, 2), KV(10, 1));
   OP(AGX3_MN_stscs, 0, KV(2, 4), KV(9, 2), KV(10, 2), KV(13, 1), KV(16, 2), KV(23, 2), KV(24, 1), KV(27, 1), KV(38, 2), KV(39, 2), KV(42, 1), KV(54, 1), KV(68, 1), KV(70, 2));
   OP(AGX3_MN_cndend, 0, KV(10, 1), KV(11, 2), KV(12, 2));
   OP(AGX3_MN_rld, 0, KV(2, 4), KV(9, 2), KV(10, 1), KV(12, 1), KV(21, 57), KV(23, 7), KV(24, 1), KV(38, 1));
   OP(AGX3_MN_ldimm, 1, KV(9, 2), KV(10, 2), KV(11, 1), KV(14, 192));
   OP(AGX3_MN_ldimm, 1, KV(2, 2), KV(9, 2), KV(10, 2), KV(11, 1), KV(14, 512));
   OP(AGX3_MN_imadd, 0, KV(9, 2), KV(10, 2), KV(16, 4), KV(23, 2), KV(24, 1), KV(27, 1), KV(31, 2), KV(38, 2), KV(39, 2), KV(42, 1), KV(53, 2), KV(54, 2), KV(57, 1), KV(60, 1));
   OP(AGX3_MN_ldscs, 0, KV(10, 2), KV(23, 1), KV(24, 1), KV(28, 1), KV(38, 2), KV(39, 2), KV(42, 1), KV(54, 1), KV(68, 3), KV(80, 1));
   OP(AGX3_MN_barrier, 0, KV(25, 1), KV(30, 1), KV(39, 1));
   OP0(AGX3_MN_end, 0);
#undef OP
#undef OP0
   if (used > sizeof(staged)) return false;
   memcpy(out, staged, used);
   *length = used;
   return true;
}
void
agx3_rt_gen1_default_constants(uint8_t out[AGX3_RT_GEN1_DEFAULT_CONSTANTS_SIZE])
{
   const uint32_t words[4] = {0x0fe00200, 0x00000fff, 0, 0x40000000};
   memcpy(out, words, sizeof(words));
}
bool
agx3_rt_gen1_runtime_code_map(uint64_t rows[AGX3_RT_GEN1_RUNTIME_MAP_ROWS][3])
{
   const uint64_t kinds[3] = {0, 0x10000, 0x20000};
   uint64_t offset = 0;
   for (unsigned s = 0; s < AGX3_RT_GEN1_RUNTIME_SLOTS; ++s)
      for (unsigned k = 0; k < 3; ++k) {
         const bool large = k == 0 && s != 0 && s <= 0x80 && (s % 16 == 15 || s % 16 == 0);
         const uint64_t size = large ? UINT64_C(0x10000) : AGX3_RT_GEN1_RUNTIME_MAP_BYTES;
         rows[s * 3 + k][0] = offset;
         rows[s * 3 + k][1] = size;
         rows[s * 3 + k][2] = AGX3_RT_GEN1_RUNTIME_BASE + s * AGX3_RT_GEN1_RUNTIME_SLOT_STRIDE + kinds[k];
         offset += size;
      }
   return offset == AGX3_RT_GEN1_RUNTIME_IMAGE_BYTES;
}
struct agx3_rt_gen1_ift_entry { uint8_t block, position, slot; uint32_t control; };
static const struct agx3_rt_gen1_ift_entry ift_table[] = {
   {1, 0, 1, 0x1d0},
   {1, 1, 3, 0x1d0},
   {1, 2, 5, 0x1d0},
   {1, 3, 7, 0x1d0},
   {1, 4, 9, 0x1d0},
   {1, 5, 11, 0x1d0},
   {2, 0, 17, 0x1d0},
   {2, 1, 19, 0x1d0},
   {2, 2, 21, 0x1d0},
   {2, 3, 23, 0x1d0},
   {2, 4, 25, 0x1d0},
   {2, 5, 27, 0x1d0},
   {3, 0, 33, 0x1d0},
   {3, 1, 35, 0x1d0},
   {3, 2, 37, 0x1d0},
   {3, 3, 39, 0x1d0},
   {3, 4, 41, 0x1d0},
   {3, 5, 43, 0x1d0},
   {4, 0, 49, 0x1d0},
   {4, 1, 51, 0x1d0},
   {4, 2, 53, 0x1d0},
   {4, 3, 55, 0x1d0},
   {4, 4, 57, 0x1d0},
   {4, 5, 59, 0x1d0},
   {5, 0, 65, 0x1d0},
   {5, 1, 67, 0x1d0},
   {5, 2, 69, 0x1d0},
   {5, 3, 71, 0x1d0},
   {5, 4, 73, 0x1d0},
   {5, 5, 75, 0x1d0},
   {6, 0, 81, 0x1d0},
   {6, 1, 83, 0x1d0},
   {6, 2, 85, 0x1d0},
   {6, 3, 87, 0x1d0},
   {6, 4, 89, 0x1d0},
   {6, 5, 91, 0x1d0},
   {7, 0, 97, 0x1d0},
   {7, 1, 99, 0x1d0},
   {7, 2, 101, 0x1d0},
   {7, 3, 103, 0x1d0},
   {7, 4, 105, 0x1d0},
   {7, 5, 107, 0x1d0},
   {8, 0, 113, 0x1d0},
   {8, 1, 115, 0x1d0},
   {8, 2, 117, 0x1d0},
   {8, 3, 119, 0x1d0},
   {8, 4, 121, 0x1d0},
   {8, 5, 123, 0x1d0},
   {9, 0, 15, 0x1d0},
   {9, 1, 16, 0x1d0},
   {10, 0, 31, 0x1d0},
   {10, 1, 32, 0x1d0},
   {11, 0, 47, 0x1d0},
   {11, 1, 48, 0x1d0},
   {12, 0, 63, 0x1d0},
   {12, 1, 64, 0x1d0},
   {13, 0, 79, 0x1d0},
   {13, 1, 80, 0x1d0},
   {14, 0, 95, 0x1d0},
   {14, 1, 96, 0x1d0},
   {15, 0, 111, 0x1d0},
   {15, 1, 112, 0x1d0},
   {16, 0, 127, 0x1d0},
   {16, 1, 128, 0x1d0},
   {57, 0, 209, 0x150},
   {57, 1, 209, 0x150},
   {65, 0, 13, 0x1d0},
   {65, 1, 14, 0x1d0},
   {66, 0, 29, 0x1d0},
   {66, 1, 30, 0x1d0},
   {67, 0, 45, 0x1d0},
   {67, 1, 46, 0x1d0},
   {68, 0, 61, 0x1d0},
   {68, 1, 62, 0x1d0},
   {69, 0, 77, 0x1d0},
   {69, 1, 78, 0x1d0},
   {70, 0, 93, 0x1d0},
   {70, 1, 94, 0x1d0},
   {71, 0, 109, 0x1d0},
   {71, 1, 110, 0x1d0},
   {72, 0, 125, 0x1d0},
   {72, 1, 126, 0x1d0},
   {73, 0, 129, 0x180},
   {73, 1, 130, 0x100},
   {74, 0, 139, 0x180},
   {74, 1, 140, 0x100},
   {75, 0, 149, 0x180},
   {75, 1, 150, 0x100},
   {76, 0, 159, 0x180},
   {76, 1, 160, 0x100},
   {77, 0, 169, 0x180},
   {77, 1, 170, 0x100},
   {78, 0, 179, 0x180},
   {78, 1, 180, 0x100},
   {79, 0, 189, 0x180},
   {79, 1, 190, 0x100},
   {80, 0, 199, 0x180},
   {80, 1, 200, 0x100},
   {81, 0, 131, 0x180},
   {81, 1, 132, 0x100},
   {81, 2, 133, 0x180},
   {81, 3, 134, 0x100},
   {82, 0, 141, 0x180},
   {82, 1, 142, 0x100},
   {82, 2, 143, 0x180},
   {82, 3, 144, 0x100},
   {83, 0, 151, 0x180},
   {83, 1, 152, 0x100},
   {83, 2, 153, 0x180},
   {83, 3, 154, 0x100},
   {84, 0, 161, 0x180},
   {84, 1, 162, 0x100},
   {84, 2, 163, 0x180},
   {84, 3, 164, 0x100},
   {85, 0, 171, 0x180},
   {85, 1, 172, 0x100},
   {85, 2, 173, 0x180},
   {85, 3, 174, 0x100},
   {86, 0, 181, 0x180},
   {86, 1, 182, 0x100},
   {86, 2, 183, 0x180},
   {86, 3, 184, 0x100},
   {87, 0, 191, 0x180},
   {87, 1, 192, 0x100},
   {87, 2, 193, 0x180},
   {87, 3, 194, 0x100},
   {88, 0, 201, 0x180},
   {88, 1, 202, 0x100},
   {88, 2, 203, 0x180},
   {88, 3, 204, 0x100},
   {89, 0, 135, 0x180},
   {89, 1, 136, 0x180},
   {89, 2, 137, 0x180},
   {89, 3, 138, 0x180},
   {90, 0, 145, 0x180},
   {90, 1, 146, 0x180},
   {90, 2, 147, 0x180},
   {90, 3, 148, 0x180},
   {91, 0, 155, 0x180},
   {91, 1, 156, 0x180},
   {91, 2, 157, 0x180},
   {91, 3, 158, 0x180},
   {92, 0, 165, 0x180},
   {92, 1, 166, 0x180},
   {92, 2, 167, 0x180},
   {92, 3, 168, 0x180},
   {93, 0, 175, 0x180},
   {93, 1, 176, 0x180},
   {93, 2, 177, 0x180},
   {93, 3, 178, 0x180},
   {94, 0, 185, 0x180},
   {94, 1, 186, 0x180},
   {94, 2, 187, 0x180},
   {94, 3, 188, 0x180},
   {95, 0, 195, 0x180},
   {95, 1, 196, 0x180},
   {95, 2, 197, 0x180},
   {95, 3, 198, 0x180},
   {96, 0, 205, 0x180},
   {96, 1, 206, 0x180},
   {96, 2, 207, 0x180},
   {96, 3, 208, 0x180},
   {97, 0, 2, 0x180},
   {97, 1, 4, 0x180},
   {97, 2, 6, 0x180},
   {97, 3, 8, 0x180},
   {97, 4, 10, 0x180},
   {97, 5, 12, 0x180},
   {98, 0, 18, 0x180},
   {98, 1, 20, 0x180},
   {98, 2, 22, 0x180},
   {98, 3, 24, 0x180},
   {98, 4, 26, 0x180},
   {98, 5, 28, 0x180},
   {99, 0, 34, 0x180},
   {99, 1, 36, 0x180},
   {99, 2, 38, 0x180},
   {99, 3, 40, 0x180},
   {99, 4, 42, 0x180},
   {99, 5, 44, 0x180},
   {100, 0, 50, 0x180},
   {100, 1, 52, 0x180},
   {100, 2, 54, 0x180},
   {100, 3, 56, 0x180},
   {100, 4, 58, 0x180},
   {100, 5, 60, 0x180},
   {101, 0, 66, 0x180},
   {101, 1, 68, 0x180},
   {101, 2, 70, 0x180},
   {101, 3, 72, 0x180},
   {101, 4, 74, 0x180},
   {101, 5, 76, 0x180},
   {102, 0, 82, 0x180},
   {102, 1, 84, 0x180},
   {102, 2, 86, 0x180},
   {102, 3, 88, 0x180},
   {102, 4, 90, 0x180},
   {102, 5, 92, 0x180},
   {103, 0, 98, 0x180},
   {103, 1, 100, 0x180},
   {103, 2, 102, 0x180},
   {103, 3, 104, 0x180},
   {103, 4, 106, 0x180},
   {103, 5, 108, 0x180},
   {104, 0, 114, 0x180},
   {104, 1, 116, 0x180},
   {104, 2, 118, 0x180},
   {104, 3, 120, 0x180},
   {104, 4, 122, 0x180},
   {104, 5, 124, 0x180},
   {105, 0, 0, 0x160},
   {105, 1, 0, 0x160},
   {105, 2, 0, 0x160},
   {105, 3, 0, 0x160},
   {105, 4, 0, 0x160},
   {105, 5, 0, 0x160},
   {105, 6, 0, 0x160},
   {105, 7, 0, 0x160},
};
void
agx3_rt_gen1_runtime_ift(uint8_t out[AGX3_RT_GEN1_IFT_BYTES])
{
   memset(out, 0, AGX3_RT_GEN1_IFT_BYTES);
   for (size_t i = 0; i < sizeof(ift_table) / sizeof(ift_table[0]); ++i) {
      const struct agx3_rt_gen1_ift_entry *e = &ift_table[i];
      const uint64_t entry = AGX3_RT_GEN1_RUNTIME_BASE + e->slot * AGX3_RT_GEN1_RUNTIME_SLOT_STRIDE + 0x20000;
      const uint64_t control = (UINT64_C(0x80000000) | e->control) << 32;
      memcpy(out + e->block * 128 + e->position * 16, &entry, 8);
      memcpy(out + e->block * 128 + e->position * 16 + 8, &control, 8);
   }
}
bool
agx3_rt_gen1_stub_body_generate(unsigned slot, uint8_t *out, size_t capacity, size_t *length)
{
   uint8_t staged[256] = {0};
   size_t used = 64;
   uint8_t n;
   if (!out || !length || slot >= AGX3_RT_GEN1_RUNTIME_SLOTS) return false;
   memcpy(staged, default_body_prefix_end, sizeof(default_body_prefix_end));
   for (unsigned i = 2; i < 32; ++i) staged[2 * i] = 0x06;
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, AGX3_MN_end, 0, NULL, 0, NULL, 0, staged + used, &n)) return false;
   used += n;
   if (used > capacity) return false;
   memcpy(out, staged, used);
   *length = used;
   return true;
}

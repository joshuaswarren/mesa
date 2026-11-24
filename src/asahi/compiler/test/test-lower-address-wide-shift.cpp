/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <limits.h>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
#include "agx_nir.h"
}

/* agx_nir_lower_address on shader addresses built from an index shifted
 * within or beyond the access's scale and from base constants, then
 * evaluated: the lowered access (base, 32-bit index, scale, byte offset,
 * extension) must name the address the original 64-bit expression does, for
 * indices whose shift wraps at 2^32 and changes sign at 2^31 and for any
 * 64-bit constant.
 */

namespace {

enum shape {
   ADD_TO_LEA, /* p + u2u64(x << a), which agx_nir_fuse_lea makes a LEA */
   ADD_ZERO,   /* p + (u2u64(x << a) << 0): the direct-add arm */
   ADD_MOV,    /* p + mov(u2u64(x << a)): the direct-add arm */
   ULEA,       /* ulea(p, x << a, s) */
   ILEA,       /* ilea(p, x << a, s) */
   CONST_BASE, /* (p + k) + u2u64(x << a) */
   CONST_LATE, /* u2u64(x << a) + (k + p) */
   CONST_OFF,  /* (p + k) + u2u64(16 + amul(x, 16)) */
   CONST_NEG,  /* (p + k) + u2u64(amul(x, 16) - 16) */
};

/* Whether agx_nir_fuse_lea leaves the address an add (the direct-add arm of
 * agx_nir_lower_address) rather than making it a LEA. */
static bool
is_direct_add(enum shape sh)
{
   return sh == ADD_ZERO || sh == ADD_MOV;
}

enum kind { LOAD, LOAD_CONSTANT, STORE };

struct access {
   unsigned bit_size, comps;
   enum kind kind;
};

/* (byte_indices, byte_offsets) as each target passes them: M5 and G18 run
 * the rescaling, the others must not. */
struct options {
   bool byte_indices, byte_offsets;
};

static const struct options all_options[] = {
   {false, true}, /* M5 */
   {true, true},  /* G18 */
   {true, false}, /* G17P */
   {false, false},
};

static uint64_t
reference(enum shape sh, uint64_t p, uint32_t x, unsigned a, unsigned s,
          int64_t k)
{
   const uint32_t o = x << (a & 31);
   switch (sh) {
   case ADD_TO_LEA:
   case ADD_ZERO:
   case ADD_MOV:
      return p + (uint64_t)o;
   case ULEA:
      return p + ((uint64_t)o << s);
   case ILEA:
      return p + ((uint64_t)(int64_t)(int32_t)o << s);
   case CONST_BASE:
   case CONST_LATE:
      return p + (uint64_t)k + (uint64_t)o;
   case CONST_OFF:
      return p + (uint64_t)k + (uint64_t)(uint32_t)(16u + x * 16u);
   case CONST_NEG:
      return p + (uint64_t)k + (uint64_t)(uint32_t)(x * 16u - 16u);
   }
   return 0;
}

static nir_def *
address(nir_builder *b, enum shape sh, nir_def *p, nir_def *x, unsigned a,
        unsigned s, int64_t k)
{
   nir_def *o = nir_ishl(b, x, nir_imm_int(b, a));
   switch (sh) {
   case ADD_TO_LEA:
      return nir_iadd(b, p, nir_u2u64(b, o));
   case ADD_ZERO:
      return nir_iadd(b, p, nir_ishl(b, nir_u2u64(b, o), nir_imm_int(b, 0)));
   case ADD_MOV:
      return nir_iadd(b, p, nir_mov(b, nir_u2u64(b, o)));
   case ULEA:
      return nir_ulea_agx(b, p, o, nir_imm_int(b, s));
   case ILEA:
      return nir_ilea_agx(b, p, o, nir_imm_int(b, s));
   case CONST_BASE:
      return nir_iadd(b, nir_iadd(b, p, nir_imm_int64(b, k)), nir_u2u64(b, o));
   case CONST_LATE:
      return nir_iadd(b, nir_u2u64(b, o), nir_iadd(b, nir_imm_int64(b, k), p));
   case CONST_OFF:
      return nir_iadd(
         b, nir_iadd(b, p, nir_imm_int64(b, k)),
         nir_u2u64(b, nir_iadd_imm(b, nir_amul_imm(b, x, 16), 16)));
   case CONST_NEG:
      return nir_iadd(
         b, nir_iadd(b, p, nir_imm_int64(b, k)),
         nir_u2u64(b, nir_iadd_imm(b, nir_amul_imm(b, x, 16), -16)));
   }
   return NULL;
}

/* Builds one access at the address of `sh`; *xp is x. */
static nir_builder
build(enum shape sh, struct access acc, unsigned a, unsigned s, int64_t k,
      nir_def **pp, nir_def **xp)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "address");
   nir_def *p = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *x = nir_load_preamble(&b, 1, 32, .base = 4);
   nir_def *addr = address(&b, sh, p, x, a, s, k);
   const unsigned align = acc.bit_size / 8;
   if (acc.kind == STORE) {
      nir_store_global(&b, nir_imm_zero(&b, acc.comps, acc.bit_size), addr,
                       .align_mul = align);
   } else {
      nir_def *v = acc.kind == LOAD_CONSTANT
                      ? nir_load_global_constant(&b, acc.comps, acc.bit_size,
                                                 addr, .align_mul = align)
                      : nir_load_global(&b, acc.comps, acc.bit_size, addr,
                                        .align_mul = align);
      /* Keep the load alive: store it through a separate pointer. */
      nir_store_global(&b, v, nir_load_preamble(&b, 1, 64, .base = 8),
                       .align_mul = align);
   }
   *pp = p;
   *xp = x;
   return b;
}

/* The lowered access of `acc` in the shader, or NULL. */
static nir_intrinsic_instr *
lowered_access(nir_shader *shader, struct access acc, unsigned *base_src)
{
   nir_foreach_block(block, nir_shader_get_entrypoint(shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if ((intr->intrinsic == nir_intrinsic_load_agx && acc.kind == LOAD) ||
             (intr->intrinsic == nir_intrinsic_load_constant_agx &&
              acc.kind == LOAD_CONSTANT)) {
            *base_src = 0;
            return intr;
         }
         if (intr->intrinsic == nir_intrinsic_store_agx && acc.kind == STORE) {
            *base_src = 1;
            return intr;
         }
      }
   }
   return NULL;
}

static bool
has_lea(nir_shader *shader)
{
   nir_foreach_block(block, nir_shader_get_entrypoint(shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_alu &&
             (nir_instr_as_alu(instr)->op == nir_op_ulea_agx ||
              nir_instr_as_alu(instr)->op == nir_op_ilea_agx))
            return true;
      }
   }
   return false;
}

/* Builds, lowers and evaluates one access; returns the address it names. */
static uint64_t
lowered(enum shape sh, struct access acc, struct options o, uint64_t p_value,
        uint32_t x_value, unsigned a, unsigned s, int64_t k, bool *lowered_ok)
{
   nir_def *p, *x;
   nir_builder b = build(sh, acc, a, s, k, &p, &x);

   /* The entry path: the direct-add shapes must survive LEA fusion. */
   if (is_direct_add(sh) || sh == ADD_TO_LEA) {
      NIR_PASS(_, b.shader, agx_nir_fuse_lea);
      EXPECT_EQ(has_lea(b.shader), !is_direct_add(sh)) << "shape " << sh;
   }

   agx_nir_lower_address(b.shader, o.byte_indices, o.byte_offsets, false);

   b.cursor = nir_before_impl(nir_shader_get_entrypoint(b.shader));
   nir_def_rewrite_uses(p, nir_imm_int64(&b, p_value));
   nir_def_rewrite_uses(x, nir_imm_int(&b, x_value));
   nir_opt_constant_folding(b.shader);

   uint64_t result = 0;
   *lowered_ok = false;
   unsigned base_src;
   nir_intrinsic_instr *intr = lowered_access(b.shader, acc, &base_src);
   if (intr && nir_src_is_const(intr->src[base_src]) &&
       nir_src_is_const(intr->src[base_src + 1])) {
      const uint64_t base = nir_src_as_uint(intr->src[base_src]);
      const uint32_t index = nir_src_as_uint(intr->src[base_src + 1]);
      const unsigned mem_base = nir_intrinsic_base(intr);
      const int16_t byte_offset =
         (int16_t)(mem_base >> AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT);
      const enum pipe_format format = nir_intrinsic_format(intr);
      const unsigned format_shift =
         util_logbase2(util_format_get_blocksize(format));
      const uint64_t ext = nir_intrinsic_sign_extend(intr)
                              ? (uint64_t)(int64_t)(int32_t)index
                              : (uint64_t)index;
      const unsigned scale = (mem_base & AGX_NIR_MEMORY_INDEX_BYTES)
                                ? 0
                                : format_shift + (mem_base & 7);
      result = base + (ext << scale) + (uint64_t)(int64_t)byte_offset;
      *lowered_ok = true;
   }
   ralloc_free(b.shader);
   return result;
}

static const uint32_t xs[] = {
   0u,          1u,          3u,          0x00ffffffu, 0x01000000u, 0x01ffffffu,
   0x02000000u, 0x0fffffffu, 0x7fffffffu, 0x80000000u, 0xfffffffeu, 0xffffffffu,
};

static const struct access accesses[] = {
   {32, 1, LOAD},  {32, 4, LOAD},          {16, 1, LOAD},
   {8, 1, LOAD},   {32, 2, LOAD_CONSTANT}, {16, 1, LOAD_CONSTANT},
   {32, 1, STORE}, {16, 2, STORE},         {8, 1, STORE},
};

static void
check(enum shape sh, const unsigned *as, unsigned na, const unsigned *ss,
      unsigned ns, const int64_t *ks, unsigned nk)
{
   const uint64_t p = 0x0000123400000000ull;
   for (struct options o : all_options) {
      for (struct access acc : accesses) {
         for (unsigned ia = 0; ia < na; ++ia) {
            for (unsigned is = 0; is < ns; ++is) {
               for (unsigned ik = 0; ik < nk; ++ik) {
                  for (uint32_t x : xs) {
                     /* amul is an array stride: valid indices do not wrap
                      * (agx_nir_lower_address relies on that), so only
                      * those are compared. */
                     if ((sh == CONST_OFF && x >= 0x07ffffffu) ||
                         (sh == CONST_NEG && (x == 0 || x >= 0x07ffffffu)))
                        continue;
                     bool ok;
                     const uint64_t got =
                        lowered(sh, acc, o, p, x, as[ia], ss[is], ks[ik], &ok);
                     ASSERT_TRUE(ok) << "shape " << sh;
                     EXPECT_EQ(got, reference(sh, p, x, as[ia], ss[is], ks[ik]))
                        << std::hex << "shape " << sh << " options "
                        << o.byte_indices << o.byte_offsets << " bits "
                        << acc.bit_size << "x" << acc.comps << " kind "
                        << acc.kind << " x 0x" << x << " a " << std::dec
                        << as[ia] << " s " << ss[is] << " k " << ks[ik];
                  }
               }
            }
         }
      }
   }
}

/* Around every access's threshold (2 to 5 with the outer scales below),
 * wide ones, and 32-36 and 39, which are 0-4 and 7 to NIR. */
static const unsigned wide_shifts[] = {1,  2,  3,  4,  5,  7,  12, 25,
                                       31, 32, 33, 34, 35, 36, 39};
static const int64_t no_k[] = {0};

TEST(LowerAddressWideShift, AddIndexWrapsAtTwoToTheThirtyTwo)
{
   const unsigned s[] = {0};
   check(ADD_TO_LEA, wide_shifts, ARRAY_SIZE(wide_shifts), s, 1, no_k, 1);
   check(ADD_ZERO, wide_shifts, ARRAY_SIZE(wide_shifts), s, 1, no_k, 1);
   check(ADD_MOV, wide_shifts, ARRAY_SIZE(wide_shifts), s, 1, no_k, 1);
}

TEST(LowerAddressWideShift, UnsignedLeaKeepsItsWrap)
{
   const unsigned s[] = {0, 1, 2};
   check(ULEA, wide_shifts, ARRAY_SIZE(wide_shifts), s, ARRAY_SIZE(s), no_k, 1);
}

TEST(LowerAddressWideShift, SignedLeaKeepsItsSign)
{
   const unsigned s[] = {0, 1, 2};
   check(ILEA, wide_shifts, ARRAY_SIZE(wide_shifts), s, ARRAY_SIZE(s), no_k, 1);
}

static const int64_t ks[] = {
   16,         -16,       32767,     32768,          -32768,         -32769,
   0x7ffffff0, INT64_MAX, INT64_MIN, INT64_MAX - 15, INT64_MIN + 15,
};

TEST(LowerAddressWideShift, BaseConstantsOfAnySize)
{
   const unsigned s[] = {0};
   check(CONST_BASE, wide_shifts, ARRAY_SIZE(wide_shifts), s, 1, ks,
         ARRAY_SIZE(ks));
   check(CONST_LATE, wide_shifts, ARRAY_SIZE(wide_shifts), s, 1, ks,
         ARRAY_SIZE(ks));
}

TEST(LowerAddressWideShift, BaseConstantBesideAnIndexOffset)
{
   const unsigned a[] = {0}, s[] = {0};
   const int64_t k2[] = {
      16,     -16,       32751,     32752,          -32784,
      -32785, INT64_MAX, INT64_MIN, INT64_MAX - 15, INT64_MIN + 15};
   check(CONST_OFF, a, 1, s, 1, k2, ARRAY_SIZE(k2));
   /* A negative displacement beside it: k - 16 fits for -32752..32783. */
   const int64_t k3[] = {
      16,     -16,       32783,     32784,          -32752,
      -32753, INT64_MAX, INT64_MIN, INT64_MAX - 15, INT64_MIN + 15};
   check(CONST_NEG, a, 1, s, 1, k3, ARRAY_SIZE(k3));
}

/* The amount the lowered access's index shifts o = x << a right by, or -1
 * when the index is not such a shift. x is unbounded, so o may wrap:
 * - Past the access's largest scale (s < max_shift < s + (a & 31)), the index
 *   is o >> (max_shift - s) at the full scale. Through a LEA that needs s
 *   below the element size's scale, so 8-bit accesses rescale only through
 *   the direct-add arm.
 * - Within it (a <= max_shift - s), it is o >> a at scale s + a.
 * At a = max_shift - s the two are the same index.
 */
static int
index_shift(enum shape sh, unsigned bits, unsigned a, unsigned s)
{
   nir_def *p, *x;
   struct access acc = {bits, 1, LOAD};
   nir_builder b = build(sh, acc, a, s, 0, &p, &x);
   agx_nir_lower_address(b.shader, false, true, false);
   unsigned base_src;
   nir_intrinsic_instr *intr = lowered_access(b.shader, acc, &base_src);
   int amount = -1;
   if (intr) {
      nir_scalar index =
         nir_scalar_chase_movs(nir_get_scalar(intr->src[base_src + 1].ssa, 0));
      if (nir_scalar_is_alu(index) &&
          (nir_scalar_alu_op(index) == nir_op_ushr ||
           nir_scalar_alu_op(index) == nir_op_ishr) &&
          nir_scalar_is_const(nir_scalar_chase_alu_src(index, 1)))
         amount = nir_scalar_as_uint(nir_scalar_chase_alu_src(index, 1));
   }
   ralloc_free(b.shader);
   return amount;
}

TEST(LowerAddressWideShift, RescalesExactlyPastTheThreshold)
{
   /* 32-bit: max 4, through a LEA for s 0 and 1 */
   EXPECT_EQ(index_shift(ULEA, 32, 3, 0), 3);
   EXPECT_EQ(index_shift(ULEA, 32, 4, 0), 4);
   EXPECT_EQ(index_shift(ULEA, 32, 5, 0), 4);
   EXPECT_EQ(index_shift(ULEA, 32, 12, 0), 4);
   EXPECT_EQ(index_shift(ULEA, 32, 3, 1), 3);
   EXPECT_EQ(index_shift(ULEA, 32, 4, 1), 3);
   EXPECT_EQ(index_shift(ILEA, 32, 4, 1), 3);
   EXPECT_EQ(index_shift(ULEA, 32, 7, 2), -1); /* s is the element scale */
   EXPECT_EQ(index_shift(ADD_ZERO, 32, 4, 0), 4);
   EXPECT_EQ(index_shift(ADD_ZERO, 32, 5, 0), 4);
   /* 16-bit: max 3 */
   EXPECT_EQ(index_shift(ULEA, 16, 3, 0), 3);
   EXPECT_EQ(index_shift(ULEA, 16, 4, 0), 3);
   EXPECT_EQ(index_shift(ADD_MOV, 16, 4, 0), 3);
   /* 8-bit: max 2, the direct-add arm only */
   EXPECT_EQ(index_shift(ADD_ZERO, 8, 2, 0), 2);
   EXPECT_EQ(index_shift(ADD_ZERO, 8, 3, 0), 2);
   EXPECT_EQ(index_shift(ADD_MOV, 8, 7, 0), 2);
   EXPECT_EQ(index_shift(ULEA, 8, 7, 0), -1);
   /* Shift counts are mod 32: 32 is 0, 36 is 4, 37 is 5. */
   EXPECT_EQ(index_shift(ULEA, 32, 32, 0), -1);
   EXPECT_EQ(index_shift(ULEA, 32, 37, 0), 4);
}

/* Where x << a cannot wrap, x itself is the index within the scale, as
 * before; past the scale the index is still o rescaled. */
TEST(LowerAddressWideShift, BoundedIndexWithinTheScaleIsTakenAsIs)
{
   for (unsigned a : {2u, 4u, 5u, 7u}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "bounded");
      nir_def *p = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *x =
         nir_iand_imm(&b, nir_load_preamble(&b, 1, 32, .base = 4), 0xffff);
      nir_def *addr = nir_iadd(&b, p, nir_u2u64(&b, nir_ishl_imm(&b, x, a)));
      nir_store_global(&b, nir_imm_int(&b, 0), addr, .align_mul = 4);
      agx_nir_lower_address(b.shader, false, true, false);
      unsigned base_src;
      nir_intrinsic_instr *st =
         lowered_access(b.shader, {32, 1, STORE}, &base_src);
      ASSERT_NE(st, nullptr);
      nir_scalar index =
         nir_scalar_chase_movs(nir_get_scalar(st->src[base_src + 1].ssa, 0));
      if (a <= 4) {
         EXPECT_EQ(index.def, x) << "a " << a;
         EXPECT_EQ(nir_intrinsic_base(st) & 7, a - 2) << "a " << a;
      } else {
         ASSERT_TRUE(nir_scalar_is_alu(index)) << "a " << a;
         EXPECT_EQ(nir_scalar_alu_op(index), nir_op_ushr) << "a " << a;
         EXPECT_EQ(nir_scalar_as_uint(nir_scalar_chase_alu_src(index, 1)), 4u)
            << "a " << a;
         EXPECT_EQ(nir_intrinsic_base(st) & 7, 2u) << "a " << a;
      }
      ralloc_free(b.shader);
   }
}

/* Atomics keep their element scale: no rescaling, and the address is
 * still the original one. */
TEST(LowerAddressWideShift, AtomicsAreNotRescaled)
{
   const uint64_t p_value = 0x0000123400000000ull;
   for (unsigned a : wide_shifts) {
      for (uint32_t x_value : xs) {
         /* An atomic's scale is its element's: a shift up to 2 is taken
          * as o >> a at scale a, and nothing past it is rescaled. */
         const unsigned e = a & 31;
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "atomic");
         nir_def *p = nir_load_preamble(&b, 1, 64, .base = 0);
         nir_def *x = nir_load_preamble(&b, 1, 32, .base = 4);
         nir_def *addr = address(&b, ADD_TO_LEA, p, x, a, 0, 0);
         nir_def *r = nir_global_atomic(&b, 32, addr, nir_imm_int(&b, 1),
                                        .atomic_op = nir_atomic_op_iadd);
         nir_store_global(&b, r, nir_load_preamble(&b, 1, 64, .base = 8),
                          .align_mul = 4);
         agx_nir_lower_address(b.shader, false, true, false);

         nir_intrinsic_instr *atomic = NULL;
         nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_alu) {
                  nir_alu_instr *alu = nir_instr_as_alu(instr);
                  if (alu->op == nir_op_ushr || alu->op == nir_op_ishr) {
                     EXPECT_TRUE(e >= 1 && e <= 2) << "a " << a;
                     EXPECT_EQ(nir_src_as_uint(alu->src[1].src), e)
                        << "a " << a;
                  }
               } else if (instr->type == nir_instr_type_intrinsic &&
                          nir_instr_as_intrinsic(instr)->intrinsic ==
                             nir_intrinsic_global_atomic_agx) {
                  atomic = nir_instr_as_intrinsic(instr);
               }
            }
         }
         ASSERT_NE(atomic, nullptr);
         b.cursor = nir_before_impl(nir_shader_get_entrypoint(b.shader));
         nir_def_rewrite_uses(p, nir_imm_int64(&b, p_value));
         nir_def_rewrite_uses(x, nir_imm_int(&b, x_value));
         nir_opt_constant_folding(b.shader);
         ASSERT_TRUE(nir_src_is_const(atomic->src[0]) &&
                     nir_src_is_const(atomic->src[1]));
         /* An atomic's index is in elements. */
         const uint32_t index = nir_src_as_uint(atomic->src[1]);
         const uint64_t ext = nir_intrinsic_sign_extend(atomic)
                                 ? (uint64_t)(int64_t)(int32_t)index
                                 : (uint64_t)index;
         EXPECT_EQ(nir_src_as_uint(atomic->src[0]) + (ext << 2),
                   reference(ADD_TO_LEA, p_value, x_value, a, 0, 0))
            << std::hex << "x 0x" << x_value << std::dec << " a " << a;
         ralloc_free(b.shader);
      }
   }
}

/* The rewrite still takes the wide shift as an index on M5. */
TEST(LowerAddressWideShift, WideShiftBecomesAnIndex)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "index");
   nir_def *p = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *x = nir_load_preamble(&b, 1, 32, .base = 4);
   nir_def *addr = nir_iadd(&b, nir_iadd_imm(&b, p, 32),
                            nir_u2u64(&b, nir_ishl_imm(&b, x, 7)));
   nir_store_global(&b, nir_imm_zero(&b, 4, 32), addr, .align_mul = 16);
   agx_nir_lower_address(b.shader, false, true, false);
   bool found = false;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(instr)->intrinsic !=
                nir_intrinsic_store_agx)
            continue;
         nir_intrinsic_instr *st = nir_instr_as_intrinsic(instr);
         /* base p, an index, scale 2 + 2, byte offset 32 */
         EXPECT_EQ(st->src[1].ssa, p);
         EXPECT_EQ(nir_intrinsic_base(st) & 7, 2u);
         EXPECT_EQ((int16_t)(nir_intrinsic_base(st) >>
                             AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT),
                   32);
         found = true;
      }
   }
   EXPECT_TRUE(found);
   ralloc_free(b.shader);
}

} // namespace

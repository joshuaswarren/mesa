/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
#include "agx_nir.h"
}

/* agx_nir_lower_address takes an index x << a (or x * 2^a) within the
 * access's scale as x at scale a only where that is the same address: the
 * 32-bit product must not wrap (unsigned for a zero-extended index, signed
 * for a sign-extended one). The lowered access must name the original
 * address for every x. Where the product provably does not wrap, x is the
 * index; elsewhere the product shifted back by a is.
 */

namespace {

enum shape {
   ADD,  /* p + u2u64(x << a), a LEA by then */
   MUL,  /* p + u2u64(x * 2^a) */
   ULEA, /* ulea(p, x << a, s) */
   ILEA, /* ilea(p, x << a, s) */
   IMUL, /* ilea(p, x * 2^a, s) */
};

/* No-wrap flags on the product */
enum wrap_flags { NONE = 0, NUW = 1, NSW = 2 };

enum kind { LOAD, STORE, ATOMIC };

struct access {
   unsigned bit_size, comps;
   enum kind kind;
};

static uint64_t
reference(enum shape sh, uint64_t p, uint32_t x, unsigned a, unsigned s)
{
   const uint32_t o = x << a;
   switch (sh) {
   case ADD:
   case MUL:
      return p + (uint64_t)o;
   case ULEA:
      return p + ((uint64_t)o << s);
   case ILEA:
   case IMUL:
      return p + ((uint64_t)(int64_t)(int32_t)o << s);
   }
   return 0;
}

/* x is masked with `mask` first (all ones: unbounded), and the product
 * carries the no-wrap flags `flags`. */
static nir_def *
address(nir_builder *b, enum shape sh, nir_def *p, nir_def *x, unsigned a,
        unsigned s, uint32_t mask, unsigned flags)
{
   if (mask != UINT32_MAX)
      x = nir_iand_imm(b, x, mask);
   nir_def *o = (sh == MUL || sh == IMUL) ? nir_imul_imm(b, x, 1u << a)
                                          : nir_ishl(b, x, nir_imm_int(b, a));
   nir_def_as_alu(o)->no_unsigned_wrap = flags & NUW;
   nir_def_as_alu(o)->no_signed_wrap = flags & NSW;
   switch (sh) {
   case ADD:
   case MUL:
      return nir_iadd(b, p, nir_u2u64(b, o));
   case ULEA:
      return nir_ulea_agx(b, p, o, nir_imm_int(b, s));
   case ILEA:
   case IMUL:
      return nir_ilea_agx(b, p, o, nir_imm_int(b, s));
   }
   return NULL;
}

struct result {
   bool ok;
   uint64_t address;
   bool folded; /* the index is x (or x's mask), not the product */
   unsigned scale;
};

static struct result
lower(enum shape sh, struct access acc, bool byte_indices, bool byte_offsets,
      uint64_t p_value, uint32_t x_value, unsigned a, unsigned s, uint32_t mask,
      unsigned flags)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "scale");
   nir_def *p = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *x = nir_load_preamble(&b, 1, 32, .base = 4);
   nir_def *addr = address(&b, sh, p, x, a, s, mask, flags);
   const unsigned align = acc.bit_size / 8;
   if (acc.kind == STORE) {
      nir_store_global(&b, nir_imm_zero(&b, acc.comps, acc.bit_size), addr,
                       .align_mul = align);
   } else {
      nir_def *v = acc.kind == ATOMIC
                      ? nir_global_atomic(&b, acc.bit_size, addr,
                                          nir_imm_intN_t(&b, 1, acc.bit_size),
                                          .atomic_op = nir_atomic_op_iadd)
                      : nir_load_global(&b, acc.comps, acc.bit_size, addr,
                                        .align_mul = align);
      nir_store_global(&b, v, nir_load_preamble(&b, 1, 64, .base = 8),
                       .align_mul = align);
   }
   agx_nir_lower_address(b.shader, byte_indices, byte_offsets);

   nir_intrinsic_instr *intr = NULL;
   unsigned base_src = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
         if ((i->intrinsic == nir_intrinsic_load_agx && acc.kind == LOAD) ||
             (i->intrinsic == nir_intrinsic_global_atomic_agx &&
              acc.kind == ATOMIC)) {
            intr = i;
            base_src = 0;
         } else if (i->intrinsic == nir_intrinsic_store_agx &&
                    acc.kind == STORE) {
            intr = i;
            base_src = 1;
         }
      }
   }
   struct result r = {false, 0, false, 0};
   if (!intr) {
      ralloc_free(b.shader);
      return r;
   }
   /* Folded: the index is x itself or its mask, not a shift or product. */
   nir_scalar index =
      nir_scalar_chase_movs(nir_get_scalar(intr->src[base_src + 1].ssa, 0));
   r.folded = index.def == x || (nir_scalar_is_alu(index) &&
                                 nir_scalar_alu_op(index) == nir_op_iand);

   b.cursor = nir_before_impl(nir_shader_get_entrypoint(b.shader));
   nir_def_rewrite_uses(p, nir_imm_int64(&b, p_value));
   nir_def_rewrite_uses(x, nir_imm_int(&b, x_value));
   nir_opt_constant_folding(b.shader);
   if (!nir_src_is_const(intr->src[base_src]) ||
       !nir_src_is_const(intr->src[base_src + 1])) {
      ralloc_free(b.shader);
      return r;
   }
   const uint64_t base = nir_src_as_uint(intr->src[base_src]);
   const uint32_t idx = nir_src_as_uint(intr->src[base_src + 1]);
   const uint64_t ext = nir_intrinsic_sign_extend(intr)
                           ? (uint64_t)(int64_t)(int32_t)idx
                           : (uint64_t)idx;
   const unsigned format_shift = util_logbase2(acc.bit_size / 8);
   unsigned scale = format_shift; /* atomics: elements */
   if (acc.kind != ATOMIC) {
      const unsigned mem_base = nir_intrinsic_base(intr);
      const enum pipe_format format = nir_intrinsic_format(intr);
      const unsigned fs = util_logbase2(util_format_get_blocksize(format));
      scale = (mem_base & AGX_NIR_MEMORY_INDEX_BYTES) ? 0 : fs + (mem_base & 7);
      r.address =
         base + (ext << scale) +
         (uint64_t)(int64_t)(int16_t)(mem_base >>
                                      AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT);
   } else {
      r.address = base + (ext << scale);
   }
   r.scale = scale;
   r.ok = true;
   ralloc_free(b.shader);
   return r;
}

static const uint32_t xs[] = {
   0u,          1u,          0x00ffffffu, 0x1fffffffu, 0x20000000u, 0x3fffffffu,
   0x40000000u, 0x7fffffffu, 0x80000000u, 0xc0000000u, 0xfffffffeu, 0xffffffffu,
};

static const struct access accesses[] = {
   {32, 1, LOAD},  {32, 4, LOAD},  {16, 1, LOAD}, {8, 1, LOAD},
   {32, 1, STORE}, {16, 2, STORE}, {8, 1, STORE}, {32, 1, ATOMIC},
};

static const bool option_sets[][2] = {
   {false, true}, {true, true}, {true, false}, {false, false}};

TEST(LowerAddressExactScale, EveryIndexKeepsItsWrap)
{
   const uint64_t p = 0x0000123400000000ull;
   for (enum shape sh : {ADD, MUL, ULEA, ILEA, IMUL}) {
      for (struct access acc : accesses) {
         for (const bool *o : option_sets) {
            for (unsigned a = 1; a <= 4; ++a) {
               for (unsigned s : {0u, 1u, 2u}) {
                  if ((sh == ADD || sh == MUL) && s)
                     continue;
                  for (uint32_t x : xs) {
                     struct result r = lower(sh, acc, o[0], o[1], p, x, a, s,
                                             UINT32_MAX, NONE);
                     ASSERT_TRUE(r.ok) << "shape " << sh;
                     EXPECT_EQ(r.address, reference(sh, p, x, a, s))
                        << std::hex << "shape " << sh << " bits "
                        << acc.bit_size << " kind " << acc.kind << " x 0x" << x
                        << std::dec << " a " << a << " s " << s;
                  }
               }
            }
         }
      }
   }
}

/* Where the product cannot wrap, the index is still x at scale a (a 32-bit
 * access takes scales 2 to 4); elsewhere it is the product shifted back by
 * a, at the same scale. */
TEST(LowerAddressExactScale, ProvenIndicesStayFolded)
{
   const uint64_t p = 0x0000123400000000ull;
   const struct access acc = {32, 1, LOAD};
   for (enum shape sh : {ADD, MUL, ULEA, ILEA, IMUL}) {
      for (unsigned a = 2; a <= 4; ++a) {
         /* x below 2^(31 - a): bounded by its mask */
         const uint32_t mask = INT32_MAX >> a;
         for (uint32_t x : xs) {
            struct result r =
               lower(sh, acc, false, true, p, x, a, 0, mask, NONE);
            ASSERT_TRUE(r.ok);
            EXPECT_TRUE(r.folded) << "shape " << sh << " a " << a;
            EXPECT_EQ(r.scale, a);
            EXPECT_EQ(r.address, reference(sh, p, x & mask, a, 0));
         }
         /* marked with the no-wrap flag its extension needs */
         const bool sext = sh == ILEA || sh == IMUL;
         struct result r = lower(sh, acc, false, true, p, 3, a, 0, UINT32_MAX,
                                 sext ? NSW : NUW);
         EXPECT_TRUE(r.folded) << "shape " << sh << " a " << a << " flag";
         EXPECT_EQ(r.address, reference(sh, p, 3, a, 0));
         /* unbounded: the product shifted back, still at scale a */
         r = lower(sh, acc, false, true, p, 3, a, 0, UINT32_MAX, NONE);
         EXPECT_FALSE(r.folded) << "shape " << sh << " a " << a;
         EXPECT_EQ(r.scale, a) << "shape " << sh << " a " << a;
      }
   }
}

/* Each extension needs its own contract: no unsigned wrap for a
 * zero-extended index, no signed wrap for a sign-extended one, and the
 * range limits differ the same way. */
TEST(LowerAddressExactScale, WrapFlagsAreTheirOwnContracts)
{
   const uint64_t p = 0x0000123400000000ull;
   const struct access acc = {32, 1, LOAD};
   struct result r;

   /* nuw alone does not cover a sign-extended index: 0x20000000 << 2 is
    * negative as a signed word. */
   for (enum shape sh : {ILEA, IMUL}) {
      r = lower(sh, acc, false, true, p, 0x20000000u, 2, 0, UINT32_MAX, NUW);
      EXPECT_FALSE(r.folded) << "shape " << sh;
      EXPECT_EQ(r.address, p - 0x80000000ull) << "shape " << sh;
   }

   /* nsw alone does not cover a zero-extended index: 0xffffffff << 2 wraps
    * as an unsigned word ... */
   for (enum shape sh : {ADD, MUL, ULEA}) {
      r = lower(sh, acc, false, true, p, 0xffffffffu, 2, 0, UINT32_MAX, NSW);
      EXPECT_FALSE(r.folded) << "shape " << sh;
      EXPECT_EQ(r.address, p + 0xfffffffcull) << "shape " << sh;
   }
   /* ... but does for a sign-extended one: -1 << 2 is -4. */
   for (enum shape sh : {ILEA, IMUL}) {
      r = lower(sh, acc, false, true, p, 0xffffffffu, 2, 0, UINT32_MAX, NSW);
      EXPECT_TRUE(r.folded) << "shape " << sh;
      EXPECT_EQ(r.address, p - 4) << "shape " << sh;
   }

   /* x bounded below 2^30 proves x << 2 unsigned-exact, not signed-exact. */
   for (enum shape sh : {ADD, MUL, ULEA}) {
      r = lower(sh, acc, false, true, p, 0x20000000u, 2, 0, 0x3fffffffu, NONE);
      EXPECT_TRUE(r.folded) << "shape " << sh;
      EXPECT_EQ(r.address, p + 0x80000000ull) << "shape " << sh;
   }
   for (enum shape sh : {ILEA, IMUL}) {
      r = lower(sh, acc, false, true, p, 0x20000000u, 2, 0, 0x3fffffffu, NONE);
      EXPECT_FALSE(r.folded) << "shape " << sh;
      EXPECT_EQ(r.address, p - 0x80000000ull) << "shape " << sh;
   }
}

/* The range analysis caches by SSA index, which every implementation
 * numbers from zero. A bounded x in one implementation must not prove the
 * x at the same index in another. */
TEST(LowerAddressExactScale, RangesStayWithinTheirImplementation)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "two");
   /* main: %0 = p, %1 = x, a subgroup invocation (below 32) */
   nir_def *p0 = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *x0 = nir_load_subgroup_invocation(&b);
   nir_def *v0 = nir_load_global(
      &b, 1, 32, nir_iadd(&b, p0, nir_u2u64(&b, nir_ishl_imm(&b, x0, 2))),
      .align_mul = 4);
   nir_store_global(&b, v0, nir_load_preamble(&b, 1, 64, .base = 8),
                    .align_mul = 4);

   /* a second implementation: %0 = p, %1 = x, unbounded */
   nir_function *f = nir_function_create(b.shader, "second");
   nir_function_impl *impl = nir_function_impl_create(f);
   nir_builder c = nir_builder_at(nir_after_impl(impl));
   nir_def *p1 = nir_load_preamble(&c, 1, 64, .base = 0);
   nir_def *x1 = nir_load_preamble(&c, 1, 32, .base = 4);
   ASSERT_EQ(x1->index, x0->index);
   nir_def *v1 = nir_load_global(
      &c, 1, 32, nir_iadd(&c, p1, nir_u2u64(&c, nir_ishl_imm(&c, x1, 2))),
      .align_mul = 4);
   nir_store_global(&c, v1, nir_load_preamble(&c, 1, 64, .base = 8),
                    .align_mul = 4);

   agx_nir_lower_address(b.shader, false, true);

   /* The second implementation's access at x = 0x40000000, where x << 2
    * wraps to 0. */
   const uint64_t p_value = 0x0000123400000000ull;
   c.cursor = nir_before_impl(impl);
   nir_def_rewrite_uses(p1, nir_imm_int64(&c, p_value));
   nir_def_rewrite_uses(x1, nir_imm_int(&c, 0x40000000));
   nir_opt_constant_folding(b.shader);
   bool found = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(instr)->intrinsic != nir_intrinsic_load_agx)
            continue;
         nir_intrinsic_instr *ld = nir_instr_as_intrinsic(instr);
         ASSERT_TRUE(nir_src_is_const(ld->src[0]) &&
                     nir_src_is_const(ld->src[1]));
         const unsigned mem_base = nir_intrinsic_base(ld);
         const unsigned scale =
            (mem_base & AGX_NIR_MEMORY_INDEX_BYTES) ? 0 : 2 + (mem_base & 7);
         const uint64_t address =
            nir_src_as_uint(ld->src[0]) +
            ((uint64_t)(uint32_t)nir_src_as_uint(ld->src[1]) << scale) +
            (uint64_t)(int64_t)(int16_t)(mem_base >>
                                         AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT);
         EXPECT_EQ(address, p_value);
         found = true;
      }
   }
   EXPECT_TRUE(found);
   ralloc_free(b.shader);
}

} // namespace

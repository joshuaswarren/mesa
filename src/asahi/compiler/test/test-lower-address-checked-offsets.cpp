/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <vector>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
#include "agx_nir.h"
}

/* A robust buffer access is bounds-checked in software on its 32-bit offset,
 * built by nir_lower_explicit_io as amul(index, stride) plus member offsets.
 * The check sees that offset modulo 2^32, so an index such as 0x40000001 into
 * a uint[] passes it at offset 4. With checked offsets, agx_nir_lower_address
 * must address every access at base + that same 32-bit offset, for every
 * index: never at the unwrapped index * stride, which lies 4 GiB away.
 * Without checked offsets an access is valid, and only valid offsets (below
 * 2^31) must keep their address.
 */

namespace {

enum shape {
   ARRAY,          /* amul(x, d)            a[x] */
   MEMBER,         /* amul(x, d) + c        s[x].m, or a header before a[] */
   SHIFTED,        /* amul(x + k, d)        a[x + k] */
   SHIFTED_MEMBER, /* amul(x + k, d) + c    s[x + k].m */
};

enum kind { LOAD, STORE, ATOMIC };

struct access {
   unsigned bit_size, comps;
   enum kind kind;
};

struct form {
   enum shape shape;
   uint32_t d;
   int32_t k, c;
};

/* The 32-bit offset as the shader computes it, and the check tests it. */
static uint32_t
offset32(const struct form &f, uint32_t x)
{
   const bool shifted = f.shape == SHIFTED || f.shape == SHIFTED_MEMBER;
   const bool member = f.shape == MEMBER || f.shape == SHIFTED_MEMBER;
   const uint32_t i = shifted ? x + (uint32_t)f.k : x;
   return i * f.d + (member ? (uint32_t)f.c : 0);
}

/* The same offset without wrapping, as a valid access has it. */
static int64_t
offset_unwrapped(const struct form &f, uint32_t x)
{
   const bool shifted = f.shape == SHIFTED || f.shape == SHIFTED_MEMBER;
   const bool member = f.shape == MEMBER || f.shape == SHIFTED_MEMBER;
   const int64_t i = shifted ? (int64_t)x + f.k : (int64_t)x;
   return i * f.d + (member ? f.c : 0);
}

static nir_def *
build_offset(nir_builder *b, const struct form &f, nir_def *x)
{
   const bool shifted = f.shape == SHIFTED || f.shape == SHIFTED_MEMBER;
   const bool member = f.shape == MEMBER || f.shape == SHIFTED_MEMBER;
   nir_def *i = shifted ? nir_iadd_imm(b, x, f.k) : x;
   nir_def *o = nir_amul_imm(b, i, f.d);
   return member ? nir_iadd_imm(b, o, f.c) : o;
}

struct result {
   bool ok;
   uint64_t address;
};

/* The address of the access, from its lowered form */
static uint64_t
lowered_address(nir_intrinsic_instr *intr, unsigned base_src,
                const struct access &acc)
{
   const uint64_t base = nir_src_as_uint(intr->src[base_src]);
   const uint32_t idx = nir_src_as_uint(intr->src[base_src + 1]);
   const uint64_t ext = nir_intrinsic_sign_extend(intr)
                           ? (uint64_t)(int64_t)(int32_t)idx
                           : (uint64_t)idx;
   if (acc.kind == ATOMIC)
      return base + (ext << util_logbase2(acc.bit_size / 8));

   const unsigned mem_base = nir_intrinsic_base(intr);
   const enum pipe_format format = nir_intrinsic_format(intr);
   const unsigned fs = util_logbase2(util_format_get_blocksize(format));
   const unsigned scale =
      (mem_base & AGX_NIR_MEMORY_INDEX_BYTES) ? 0 : fs + (mem_base & 7);
   return base + (ext << scale) +
          (uint64_t)(int64_t)(int16_t)(mem_base >>
                                       AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT);
}

/* Lower base + u2u64(offset), with offset built by build() from x, and give
 * the address of the access for p_value and x_value. */
template <typename F>
static struct result
lower(F build, struct access acc, bool byte_indices, bool byte_offsets,
      bool checked, uint64_t p_value, uint32_t x_value)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "checked");
   nir_def *p = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *x = nir_load_preamble(&b, 1, 32, .base = 4);
   nir_def *addr = nir_iadd(&b, p, nir_u2u64(&b, build(&b, x)));
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
   agx_nir_lower_address(b.shader, byte_indices, byte_offsets, checked);

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

   struct result r = {false, 0};
   if (intr) {
      b.cursor = nir_before_impl(nir_shader_get_entrypoint(b.shader));
      nir_def_rewrite_uses(p, nir_imm_int64(&b, p_value));
      nir_def_rewrite_uses(x, nir_imm_int(&b, x_value));
      nir_opt_constant_folding(b.shader);
      if (nir_src_is_const(intr->src[base_src]) &&
          nir_src_is_const(intr->src[base_src + 1])) {
         r.address = lowered_address(intr, base_src, acc);
         r.ok = true;
      }
   }
   ralloc_free(b.shader);
   return r;
}

static struct result
lower_form(const struct form &f, struct access acc, bool byte_indices,
           bool byte_offsets, bool checked, uint64_t p, uint32_t x)
{
   return lower([&](nir_builder *b, nir_def *xd) { return build_offset(b, f, xd); },
                acc, byte_indices, byte_offsets, checked, p, x);
}

static const uint32_t xs[] = {
   0u,          1u,          2u,          0x0fffffffu, 0x10000000u,
   0x1fffffffu, 0x3fffffffu, 0x40000000u, 0x40000001u, 0x55555555u,
   0x55555556u,
   0x7fffffffu, 0x80000000u, 0xaaaaaaabu, 0xefffffffu, 0xf0000000u,
   0xfffffffeu, 0xffffffffu,
};

static const struct access accesses[] = {
   {32, 1, LOAD},  {32, 4, LOAD},  {16, 1, LOAD},  {8, 1, LOAD},
   {32, 1, STORE}, {16, 2, STORE}, {8, 1, STORE},  {32, 1, ATOMIC},
};

/* {byte_indices, byte_offsets}: AGX2, AGX3B, M5 */
static const bool option_sets[][2] = {
   {false, false}, {true, false}, {false, true}, {true, true}};

static std::vector<struct form>
forms(const struct access &acc)
{
   const uint32_t size = acc.bit_size / 8;
   std::vector<struct form> out;
   for (uint32_t d : {1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u, 32u, 48u, 64u}) {
      if (d % size)
         continue;
      out.push_back({ARRAY, d, 0, 0});
      for (int32_t c : {(int32_t)size, 4, 8, 12, 16, 0x7ff0})
         if (c % size == 0)
            out.push_back({MEMBER, d, 0, c});
      for (int32_t k : {-1, 1, 3}) {
         out.push_back({SHIFTED, d, k, 0});
         out.push_back({SHIFTED_MEMBER, d, k, (int32_t)size});
         out.push_back({SHIFTED_MEMBER, d, k, 16});
      }
   }
   return out;
}

/* Checked: every index, wrapping or not, reaches base + the checked offset.
 * This is what keeps a robust access within the buffer its check tested. */
TEST(LowerAddressCheckedOffsets, EveryAccessGoesWhereItWasChecked)
{
   const uint64_t p = 0x0000123400000000ull;
   for (struct access acc : accesses) {
      for (const struct form &f : forms(acc)) {
         for (const bool *o : option_sets) {
            for (uint32_t x : xs) {
               struct result r = lower_form(f, acc, o[0], o[1], true, p, x);
               ASSERT_TRUE(r.ok);
               EXPECT_EQ(r.address, p + offset32(f, x))
                  << std::hex << "shape " << f.shape << " d " << f.d << " k "
                  << std::dec << f.k << " c " << f.c << " bits "
                  << acc.bit_size << " kind " << acc.kind << " options "
                  << o[0] << o[1] << std::hex << " x 0x" << x;
            }
         }
      }
   }
}

/* The finding's examples, at the M5's options: each passes the check at a
 * small offset, and must be addressed there. */
TEST(LowerAddressCheckedOffsets, WrappingIndicesStayInTheBuffer)
{
   const uint64_t p = 0x0000123400000000ull;
   const struct access store = {32, 1, STORE};
   const struct {
      struct form f;
      uint32_t x, checked;
   } cases[] = {
      /* uint data[]; data[0x40000001] */
      {{ARRAY, 4, 0, 0}, 0x40000001u, 4},
      /* uvec4 data[]; data[0x10000000].y */
      {{MEMBER, 16, 0, 4}, 0x10000000u, 4},
      /* struct { uint n; uvec4 a[]; }; a[0x0fffffff].x */
      {{MEMBER, 16, 0, 16}, 0x0fffffffu, 0},
      /* uint data[]; data[i + 1], i = 0x3fffffff */
      {{SHIFTED, 4, 1, 0}, 0x3fffffffu, 0},
      /* uvec4 data[]; data[i + 1].y, i = 0x0fffffff */
      {{SHIFTED_MEMBER, 16, 1, 4}, 0x0fffffffu, 4},
      /* the same at a negative 64-bit index: i = 0xefffffff */
      {{SHIFTED_MEMBER, 16, 1, 4}, 0xefffffffu, 4},
   };
   for (const auto &t : cases) {
      ASSERT_EQ(offset32(t.f, t.x), t.checked);
      for (const bool *o : option_sets) {
         struct result r = lower_form(t.f, store, o[0], o[1], true, p, t.x);
         ASSERT_TRUE(r.ok);
         EXPECT_EQ(r.address, p + t.checked)
            << "shape " << t.f.shape << std::hex << " x 0x" << t.x;
      }
   }
}

/* Unchecked: a valid access (a nonnegative array index, an offset below
 * 2^31) keeps its address, and that includes the last bytes of a 2 GiB
 * buffer. */
TEST(LowerAddressCheckedOffsets, ValidOffsetsKeepTheirAddressUnchecked)
{
   const uint64_t p = 0x0000123400000000ull;
   for (struct access acc : accesses) {
      for (const struct form &f : forms(acc)) {
         /* x for the last valid element, and the elements around it */
         std::vector<uint32_t> vals(std::begin(xs), std::end(xs));
         const int64_t member = (f.shape == MEMBER || f.shape == SHIFTED_MEMBER)
                                   ? f.c
                                   : 0;
         const int64_t k = (f.shape == SHIFTED || f.shape == SHIFTED_MEMBER)
                              ? f.k
                              : 0;
         const int64_t last = (INT32_MAX - (acc.bit_size / 8) * acc.comps -
                               member) / f.d - k;
         for (int64_t dx = -2; dx <= 0; ++dx)
            vals.push_back((uint32_t)(last + dx));
         for (const bool *o : option_sets) {
            for (uint32_t x : vals) {
               /* A valid access has a nonnegative array index. */
               const int64_t v = offset_unwrapped(f, x);
               if ((int64_t)x + k < 0 || v < 0 ||
                   v + (acc.bit_size / 8) * acc.comps > INT32_MAX)
                  continue;
               struct result r = lower_form(f, acc, o[0], o[1], false, p, x);
               ASSERT_TRUE(r.ok);
               EXPECT_EQ(r.address, p + (uint64_t)v)
                  << std::hex << "shape " << f.shape << " d " << f.d << " k "
                  << std::dec << f.k << " c " << f.c << " bits "
                  << acc.bit_size << " kind " << acc.kind << " options "
                  << o[0] << o[1] << std::hex << " x 0x" << x;
            }
         }
      }
   }
}

/* Checked offsets ignore the shader's no-wrap flags: 0x40000001 << 2 marked
 * nuw still wraps to 4, and the access goes there. */
TEST(LowerAddressCheckedOffsets, NoWrapFlagsDoNotMoveACheckedAccess)
{
   const uint64_t p = 0x0000123400000000ull;
   const struct access load = {32, 1, LOAD};
   for (const bool *o : option_sets) {
      for (bool shl : {true, false}) {
         auto build = [&](nir_builder *b, nir_def *x) {
            nir_def *v = shl ? nir_ishl_imm(b, x, 2)
                             : nir_imul(b, x, nir_imm_int(b, 4));
            nir_def_as_alu(v)->no_unsigned_wrap = true;
            nir_def_as_alu(v)->no_signed_wrap = true;
            return v;
         };
         struct result r = lower(build, load, o[0], o[1], true, p, 0x40000001u);
         ASSERT_TRUE(r.ok);
         EXPECT_EQ(r.address, p + 4) << "shl " << shl;
      }
   }
}

/* Where the index's range proves the product cannot wrap, the checked access
 * still takes the index itself at the stride's scale: no shift back. */
TEST(LowerAddressCheckedOffsets, ProvenIndicesStayFolded)
{
   for (const bool *o : option_sets) {
      for (uint32_t d : {4u, 8u, 16u}) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "proven");
         nir_def *p = nir_load_preamble(&b, 1, 64, .base = 0);
         nir_def *x =
            nir_iand_imm(&b, nir_load_preamble(&b, 1, 32, .base = 4),
                         UINT32_MAX / d);
         nir_def *v = nir_load_global(
            &b, 1, 32, nir_iadd(&b, p, nir_u2u64(&b, nir_amul_imm(&b, x, d))),
            .align_mul = 4);
         nir_store_global(&b, v, nir_load_preamble(&b, 1, 64, .base = 8),
                          .align_mul = 4);
         agx_nir_lower_address(b.shader, o[0], o[1], true);

         bool found = false;
         nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic ||
                   nir_instr_as_intrinsic(instr)->intrinsic !=
                      nir_intrinsic_load_agx)
                  continue;
               nir_intrinsic_instr *ld = nir_instr_as_intrinsic(instr);
               EXPECT_EQ(ld->src[1].ssa, x) << "d " << d;
               EXPECT_EQ(2 + (nir_intrinsic_base(ld) & 7), util_logbase2(d));
               EXPECT_FALSE(nir_intrinsic_sign_extend(ld));
               found = true;
            }
         }
         EXPECT_TRUE(found);
         ralloc_free(b.shader);
      }
   }
}

/* Where the loop guard proves `i + c <= UINT32_MAX`, zero-extended AGX3B
 * loads fold `c` into the signed 16-bit byte offset without sign-extending
 * `i`; offsets that could wrap (or negative signed IV starts) stay unfolded. */
TEST(LowerAddressCheckedOffsets, ZeroExtendedByteOffsetGuardProof)
{
   auto run_u32_loop = [](uint32_t c, bool checked) -> int16_t {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "zext-u32-guard");
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *n = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_def *limit = nir_bcsel(&b, nir_uge(&b, n, nir_imm_int(&b, 224)),
                                 nir_iadd_imm(&b, n, -224), nir_imm_int(&b, 0));

      nir_variable *iv_var =
         nir_local_variable_create(b.impl, glsl_uint_type(), "iv");
      nir_variable *sum_var =
         nir_local_variable_create(b.impl, glsl_uint_type(), "sum");
      nir_store_var(&b, iv_var, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, sum_var, nir_imm_int(&b, 0), 1);

      nir_loop *loop = nir_push_loop(&b);
      nir_def *iv = nir_load_var(&b, iv_var);
      nir_break_if(&b, nir_uge(&b, iv, limit));
      nir_def *idx = nir_iadd_imm(&b, iv, c);
      nir_def *addr =
         nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, idx), 3));
      nir_def *v = nir_load_global(&b, 2, 32, addr, .align_mul = 8);
      nir_store_var(&b, sum_var,
                    nir_iadd(&b, nir_load_var(&b, sum_var), nir_channel(&b, v, 0)),
                    1);
      nir_store_var(&b, iv_var, nir_iadd_imm(&b, iv, 256), 1);
      nir_pop_loop(&b, loop);

      nir_store_global(&b, nir_load_var(&b, sum_var), out, .align_mul = 4);
      nir_lower_vars_to_ssa(b.shader);
      agx_nir_lower_address(b.shader, true, true, checked);

      int16_t folded = 0;
      bool found = false;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic !=
                   nir_intrinsic_load_agx)
               continue;
            nir_intrinsic_instr *ld = nir_instr_as_intrinsic(instr);
            EXPECT_FALSE(nir_intrinsic_sign_extend(ld));
            folded = (int16_t)(nir_intrinsic_base(ld) >>
                               AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT);
            found = true;
         }
      }
      EXPECT_TRUE(found);
      ralloc_free(b.shader);
      return folded;
   };

   /* max(iv) <= UINT32_MAX - 225, so +224 (+1792B) cannot wrap 32-bit unsigned. */
   EXPECT_EQ(run_u32_loop(224u, false), 1792);
   /* Exact boundary: +226 can wrap when n == UINT32_MAX and iv == UINT32_MAX - 225. */
   EXPECT_EQ(run_u32_loop(226u, false), 0);
   /* Checked offsets keep constant byte-offset folding disabled. */
   EXPECT_EQ(run_u32_loop(224u, true), 0);

   auto run_s32_loop = [](int32_t init_val, int32_t step_val, uint32_t c) -> int16_t {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "zext-s32-guard");
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *out = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *n = nir_load_preamble(&b, 1, 32, .base = 8);
      nir_def *limit =
         nir_bcsel(&b, nir_ige(&b, n, nir_imm_int(&b, INT32_MIN + 16)),
                   nir_iadd_imm(&b, n, -16), nir_imm_int(&b, INT32_MIN));

      nir_variable *iv_var =
         nir_local_variable_create(b.impl, glsl_int_type(), "iv");
      nir_variable *sum_var =
         nir_local_variable_create(b.impl, glsl_uint_type(), "sum");
      nir_store_var(&b, iv_var, nir_imm_int(&b, init_val), 1);
      nir_store_var(&b, sum_var, nir_imm_int(&b, 0), 1);

      nir_loop *loop = nir_push_loop(&b);
      nir_def *iv = nir_load_var(&b, iv_var);
      nir_break_if(&b, nir_ige(&b, iv, limit));
      nir_def *idx = nir_iadd_imm(&b, iv, c);
      nir_def *addr =
         nir_iadd(&b, base, nir_ishl_imm(&b, nir_u2u64(&b, idx), 2));
      nir_def *v = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
      nir_store_var(&b, sum_var, nir_iadd(&b, nir_load_var(&b, sum_var), v), 1);
      nir_store_var(&b, iv_var, nir_iadd_imm(&b, iv, step_val), 1);
      nir_pop_loop(&b, loop);

      nir_store_global(&b, nir_load_var(&b, sum_var), out, .align_mul = 4);
      nir_lower_vars_to_ssa(b.shader);
      agx_nir_lower_address(b.shader, true, true, false);

      int16_t folded = 0;
      nir_foreach_block(block, b.impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic !=
                   nir_intrinsic_load_agx)
               continue;
            nir_intrinsic_instr *ld = nir_instr_as_intrinsic(instr);
            EXPECT_FALSE(nir_intrinsic_sign_extend(ld));
            folded = (int16_t)(nir_intrinsic_base(ld) >>
                               AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT);
         }
      }
      ralloc_free(b.shader);
      return folded;
   };

   /* Positive: 0 <= iv <= INT32_MAX - 17, step 4 cannot wrap negative, +16 folds. */
   EXPECT_EQ(run_s32_loop(0, 4, 16u), 64);
   /* Negative control: negative init can wrap u32 on +16. */
   EXPECT_EQ(run_s32_loop(-4, 4, 16u), 0);
   /* Negative control: step 32 can wrap INT32_MAX to negative on next iteration. */
   EXPECT_EQ(run_s32_loop(0, 32, 16u), 0);
}

} // namespace

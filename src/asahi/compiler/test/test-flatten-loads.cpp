/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <functional>
#include <map>
#include <vector>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "asahi/lib/agx_abi.h"
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

static unsigned
count_ifs(nir_shader *s)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_cf_node *next = nir_cf_node_next(&block->cf_node);
         n += next && next->type == nir_cf_node_if;
      }
   }
   return n;
}

static nir_intrinsic_instr *
find_load(nir_shader *s, nir_intrinsic_op op)
{
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               return nir_instr_as_intrinsic(instr);
         }
      }
   }
   return NULL;
}

/* Preamble slots the tests read their inputs from. */
enum {
   SLOT_BASE = 0,
   SLOT_I = 2,
   SLOT_COND = 4,
   SLOT_J = 6,
};

static const uint64_t BASE = 0x1234560000ull;

/* The address the load `op` of `s` reads, constant folded with each preamble
 * slot replaced by its value in `values`. Destroys the shader's code. */
static uint64_t
evaluate_address(nir_shader *s, nir_intrinsic_op op,
                 const std::map<unsigned, uint64_t> &values)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(s);
   nir_builder b = nir_builder_create(impl);

   /* Keep the load alive when its result is not selected. */
   nir_intrinsic_instr *load = find_load(s, op);
   EXPECT_NE(load, nullptr);
   if (!load)
      return 0;
   b.cursor = nir_after_instr(&load->instr);
   nir_store_global(&b, &load->def, nir_imm_int64(&b, 8),
                    .align_mul = load->def.bit_size / 8u);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_preamble)
            continue;
         const uint64_t v = values.at(nir_intrinsic_base(intr));
         b.cursor = nir_before_instr(instr);
         nir_def_rewrite_uses(&intr->def,
                              nir_imm_intN_t(&b, v, intr->def.bit_size));
         nir_instr_remove(instr);
      }
   }
   bool progress;
   do {
      progress = nir_opt_constant_folding(s);
      progress |= nir_opt_copy_prop(s);
      progress |= nir_opt_algebraic(s);
      progress |= nir_opt_dce(s);
   } while (progress);
   nir_src *addr = &load->src[0];
   EXPECT_TRUE(nir_src_is_const(*addr));
   return nir_src_as_uint(*addr);
}

/* The address of `s`'s load `op` with the arm `taken` or not, on a clone. */
static uint64_t
address_of(nir_shader *s, nir_intrinsic_op op, bool cond, uint64_t i = 5)
{
   nir_shader *clone = nir_shader_clone(NULL, s);
   const uint64_t addr = evaluate_address(
      clone, op,
      {{SLOT_BASE, BASE}, {SLOT_I, i}, {SLOT_J, 3}, {SLOT_COND, cond}});
   ralloc_free(clone);
   return addr;
}

enum arm { ARM_THEN, ARM_ELSE };

struct load_desc {
   unsigned components = 1, bit_size = 32;
   unsigned align_mul = 4, align_offset = 0;
   gl_access_qualifier access = ACCESS_NON_WRITEABLE;
};

using address_fn = std::function<nir_def *(nir_builder *b, nir_def *base,
                                           nir_def *i, nir_def *j)>;
using extra_fn = std::function<void(nir_builder *b, nir_def *base)>;

/* if (cond) x = ...; else x = ...;, one arm load_global(address(base, i, j))
 * and the other a constant, the condition the preamble's value 4 (or a lane
 * value). `extra` adds to the arm `extra_arm`, by default the load's. */
static nir_shader *
make_guarded_load(enum arm arm, address_fn address, const load_desc &d = {},
                  bool uniform = true, extra_fn extra = nullptr,
                  int extra_arm = -1)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "guarded load");
   nir_def *base = agx_test_load_preamble(&b, 1, 64, SLOT_BASE);
   nir_def *i = agx_test_load_preamble(&b, 1, 32, SLOT_I);
   nir_def *j = agx_test_load_preamble(&b, 1, 32, SLOT_J);
   nir_def *cond = uniform
      ? nir_ine_imm(&b, agx_test_load_preamble(&b, 1, 32, SLOT_COND), 0)
      : nir_ine_imm(&b, nir_load_subgroup_invocation(&b), 0);
   nir_def *addr = address(&b, base, i, j);
   nir_def *other = nir_imm_zero(&b, d.components, d.bit_size);

   nir_def *arm_defs[2];
   nir_push_if(&b, cond);
   for (unsigned a = 0; a < 2; ++a) {
      if (a == ARM_ELSE)
         nir_push_else(&b, NULL);
      if (a == arm) {
         arm_defs[a] = nir_load_global(&b, d.components, d.bit_size, addr,
                                       .access = d.access,
                                       .align_mul = d.align_mul,
                                       .align_offset = d.align_offset);
      } else {
         arm_defs[a] = other;
      }
      if (extra && (extra_arm < 0 ? a == arm : a == (unsigned)extra_arm))
         extra(&b, base);
   }
   nir_pop_if(&b, NULL);
   nir_def *phi = nir_if_phi(&b, arm_defs[ARM_THEN], arm_defs[ARM_ELSE]);
   nir_store_global(&b, phi, nir_imm_int64(&b, 0), .align_mul = 4);
   nir_validate_shader(b.shader, "before flattening");
   return b.shader;
}

static nir_def *
index_times_4(nir_builder *b, nir_def *base, nir_def *i, nir_def *)
{
   return nir_iadd(b, base, nir_imul_imm(b, nir_u2u64(b, i), 4));
}

class FlattenLoads : public ::testing::TestWithParam<std::tuple<arm, bool>> {
 protected:
   FlattenLoads() { glsl_type_singleton_init_or_ref(); }
   ~FlattenLoads() { glsl_type_singleton_decref(); }
};

/* A uniform if becomes selects; the load keeps its address when its arm is
 * taken and reads the zero page when it is not. A divergent if stays. */
TEST_P(FlattenLoads, RedirectsTheUntakenArm)
{
   const enum arm arm = std::get<0>(GetParam());
   const bool uniform = std::get<1>(GetParam());

   nir_shader *s = make_guarded_load(arm, index_times_4, {}, uniform);
   if (!uniform) {
      EXPECT_FALSE(agx_nir_opt_flatten_loads(s, 64, 4));
      EXPECT_EQ(count_ifs(s), 1u);
      ralloc_free(s);
      return;
   }
   ASSERT_TRUE(agx_nir_opt_flatten_loads(s, 64, 4));
   nir_validate_shader(s, "after flattening");
   EXPECT_EQ(count_ifs(s), 0u);
   const bool then_taken = arm == ARM_THEN;
   EXPECT_EQ(address_of(s, nir_intrinsic_load_global, then_taken),
             BASE + 5 * 4);
   EXPECT_EQ(address_of(s, nir_intrinsic_load_global, !then_taken),
             AGX_ZERO_PAGE_ADDRESS);
   ralloc_free(s);
}

INSTANTIATE_TEST_SUITE_P(Arms, FlattenLoads,
                         ::testing::Combine(::testing::Values(ARM_THEN, ARM_ELSE),
                                            ::testing::Bool()));

/* One global address form: the address, the load, the index value, and the
 * untaken arm's expected offset into the zero page and whether the address
 * keeps its base + offset shape for it. */
struct address_case {
   const char *name;
   address_fn address;
   load_desc load;
   uint64_t i;
   uint64_t zero_page_offset;
   bool keeps_shape;
};

static nir_def *
imm64(nir_builder *b, int64_t v)
{
   return nir_imm_int64(b, v);
}

static const address_case address_cases[] = {
   {"Index", index_times_4, {}, 5, 0, true},
   /* Constants under multiplies: the untaken offset of a shape-keeping
    * redirect would be far outside the zero page. */
   {"ConstantUnderAmul",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       return nir_iadd(b, base,
                       nir_amul(b, nir_iadd(b, nir_u2u64(b, i), imm64(b, 1)),
                                imm64(b, 131072)));
    },
    {}, 5, 0, false},
   {"NegativeMultiplier",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       return nir_iadd(b, base,
                       nir_imul(b, nir_iadd(b, nir_i2i64(b, i), imm64(b, 1)),
                                imm64(b, -4096)));
    },
    {}, 5, 0, false},
   {"ConstantUnderShift",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       return nir_iadd(b, base,
                       nir_ishl_imm(b, nir_iadd(b, nir_u2u64(b, i), imm64(b, 1)),
                                    20));
    },
    {}, 5, 0, false},
   {"NegativeConstant",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       return nir_iadd(b, base,
                       nir_iadd(b, nir_imul_imm(b, nir_u2u64(b, i), 4),
                                imm64(b, -4)));
    },
    {}, 5, 0, false},
   /* Two offset sums, each with a constant: 8 + 32 stay. */
   {"NestedSums",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *j) {
       nir_def *t1 = nir_iadd(b, nir_imul_imm(b, nir_u2u64(b, i), 4), imm64(b, 8));
       nir_def *t2 = nir_iadd(b, nir_amul(b, nir_u2u64(b, j), imm64(b, 16)),
                              imm64(b, 32));
       return nir_iadd(b, nir_iadd(b, base, t1), t2);
    },
    {.components = 2, .bit_size = 32, .align_mul = 8}, 5, 40, true},
   /* Constants each below half the page whose sum is past it. */
   {"RetainedConstantsPastThePage",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       nir_def *t = nir_iadd(b, nir_imul_imm(b, nir_u2u64(b, i), 4),
                             imm64(b, 24000));
       return nir_iadd(b, nir_iadd(b, nir_iadd(b, base, imm64(b, 24000)), t),
                       imm64(b, 24000));
    },
    {}, 5, 0, false},
   /* (i + 2^63) * 2 wraps: the untaken offset is 0 modulo 2^64. */
   {"Overflow",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       return nir_iadd(b, base,
                       nir_imul(b, nir_iadd(b, nir_u2u64(b, i),
                                            imm64(b, INT64_MIN)),
                                imm64(b, 2)));
    },
    {}, 5, 0, true},
   /* A negative signed index: taken reads below the base. */
   {"SignedIndex",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       return nir_iadd(b, nir_iadd(b, base, nir_imul_imm(b, nir_i2i64(b, i), 4)),
                       imm64(b, 4));
    },
    {}, (uint64_t)-2, 4, true},
   /* Vector and 64-bit loads ending exactly at the end of the page, and just
    * past it. */
   {"Vec4AtPageEnd",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       return nir_iadd(b, nir_iadd(b, base, nir_imul_imm(b, nir_u2u64(b, i), 16)),
                       imm64(b, AGX_ZERO_PAGE_SIZE - 16));
    },
    {.components = 4, .bit_size = 32, .align_mul = 16},
    5, AGX_ZERO_PAGE_SIZE - 16, true},
   {"Vec4PastPageEnd",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       return nir_iadd(b, nir_iadd(b, base, nir_imul_imm(b, nir_u2u64(b, i), 16)),
                       imm64(b, AGX_ZERO_PAGE_SIZE - 8));
    },
    {.components = 4, .bit_size = 32, .align_mul = 8}, 5, 0, false},
   {"U64x2AtPageEnd",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       return nir_iadd(b, nir_iadd(b, base, nir_imul_imm(b, nir_u2u64(b, i), 16)),
                       imm64(b, AGX_ZERO_PAGE_SIZE - 16));
    },
    {.components = 2, .bit_size = 64, .align_mul = 16},
    5, AGX_ZERO_PAGE_SIZE - 16, true},
   {"U64PastPageEnd",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       return nir_iadd(b, nir_iadd(b, base, nir_imul_imm(b, nir_u2u64(b, i), 8)),
                       imm64(b, AGX_ZERO_PAGE_SIZE - 4));
    },
    {.components = 1, .bit_size = 64, .align_mul = 4}, 5, 0, false},
   /* A residual the declared alignment does not allow, and an alignment
    * offset kept by both forms. */
   {"MisalignedResidual",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       return nir_iadd(b, nir_iadd(b, base, nir_imul_imm(b, nir_u2u64(b, i), 16)),
                       imm64(b, 8));
    },
    {.components = 4, .bit_size = 32, .align_mul = 16}, 5, 0, false},
   {"AlignOffset",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       return nir_iadd(b, nir_iadd(b, base, nir_imul_imm(b, nir_u2u64(b, i), 16)),
                       imm64(b, 4));
    },
    {.components = 1, .bit_size = 32, .align_mul = 16, .align_offset = 4},
    5, 4, true},
   {"AlignOffsetWholeAddress",
    [](nir_builder *b, nir_def *base, nir_def *i, nir_def *) {
       nir_def *t = nir_amul(b, nir_iadd(b, nir_u2u64(b, i), imm64(b, 1)),
                             imm64(b, 131072));
       return nir_iadd(b, nir_iadd(b, base, t), imm64(b, 4));
    },
    {.components = 1, .bit_size = 32, .align_mul = 16, .align_offset = 4},
    5, 4, false},
};

class FlattenLoadAddress
    : public ::testing::TestWithParam<std::tuple<address_case, arm>> {
 protected:
   FlattenLoadAddress() { glsl_type_singleton_init_or_ref(); }
   ~FlattenLoadAddress() { glsl_type_singleton_decref(); }
};

/* With its arm taken the load reads the address it read before; with it not
 * taken, a whole load inside the zero page at the load's alignment. */
TEST_P(FlattenLoadAddress, StaysInsideTheZeroPage)
{
   const address_case &c = std::get<0>(GetParam());
   const enum arm arm = std::get<1>(GetParam());
   const bool then_taken = arm == ARM_THEN;

   nir_shader *s = make_guarded_load(arm, c.address, c.load);
   const uint64_t before = address_of(s, nir_intrinsic_load_global, then_taken, c.i);

   ASSERT_TRUE(agx_nir_opt_flatten_loads(s, 64, 4));
   nir_validate_shader(s, "after flattening");
   EXPECT_EQ(count_ifs(s), 0u);

   nir_intrinsic_instr *load = find_load(s, nir_intrinsic_load_global);
   ASSERT_NE(load, nullptr);
   EXPECT_EQ(nir_def_as_alu(load->src[0].ssa)->op,
             c.keeps_shape ? nir_op_iadd : nir_op_bcsel);

   EXPECT_EQ(address_of(s, nir_intrinsic_load_global, then_taken, c.i), before);

   const uint64_t untaken =
      address_of(s, nir_intrinsic_load_global, !then_taken, c.i);
   EXPECT_EQ(untaken, AGX_ZERO_PAGE_ADDRESS + c.zero_page_offset);
   const unsigned bytes = c.load.components * c.load.bit_size / 8;
   EXPECT_GE(untaken, AGX_ZERO_PAGE_ADDRESS);
   EXPECT_LE(untaken + bytes, AGX_ZERO_PAGE_ADDRESS + AGX_ZERO_PAGE_SIZE);
   EXPECT_EQ(untaken % c.load.align_mul, c.load.align_offset);
   ralloc_free(s);
}

INSTANTIATE_TEST_SUITE_P(
   Forms, FlattenLoadAddress,
   ::testing::Combine(::testing::ValuesIn(address_cases),
                      ::testing::Values(ARM_THEN, ARM_ELSE)),
   [](const ::testing::TestParamInfo<FlattenLoadAddress::ParamType> &info) {
      return std::string(std::get<0>(info.param).name) +
             (std::get<1>(info.param) == ARM_THEN ? "Then" : "Else");
   });

class FlattenLoadsKeeps : public ::testing::Test {
 protected:
   FlattenLoadsKeeps() { glsl_type_singleton_init_or_ref(); }
   ~FlattenLoadsKeeps() { glsl_type_singleton_decref(); }

   /* Whether the if of `s` is flattened, freeing `s`. */
   static bool flattens(nir_shader *s, unsigned max_loads = 4)
   {
      const bool progress = agx_nir_opt_flatten_loads(s, 64, max_loads);
      EXPECT_EQ(count_ifs(s), progress ? 0u : 1u);
      ralloc_free(s);
      return progress;
   }
};

/* An arm with a store, or a volatile or atomic load, stays an if. */
TEST_F(FlattenLoadsKeeps, StoresVolatileAndAtomicLoads)
{
   EXPECT_FALSE(flattens(make_guarded_load(
      ARM_THEN, index_times_4, {}, true, [](nir_builder *b, nir_def *) {
         nir_store_global(b, nir_imm_int(b, 1), nir_imm_int64(b, 64),
                          .align_mul = 4);
      })));
   EXPECT_FALSE(flattens(make_guarded_load(ARM_THEN, index_times_4,
                                           {.access = ACCESS_VOLATILE})));
   EXPECT_FALSE(flattens(make_guarded_load(ARM_THEN, index_times_4,
                                           {.access = ACCESS_ATOMIC})));
}

/* A load the pass redirects next to a memory read that may be reordered but
 * not speculated, whose address an untaken arm may not have made valid,
 * stays an if; the same read marked speculatable does not. */
TEST_F(FlattenLoadsKeeps, ReadsThatCannotBeSpeculated)
{
   const gl_access_qualifier reorder =
      (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
   const gl_access_qualifier speculate =
      (gl_access_qualifier)(reorder | ACCESS_CAN_SPECULATE);

   for (gl_access_qualifier access : {reorder, speculate}) {
      const bool expected = access == speculate;
      EXPECT_EQ(flattens(make_guarded_load(
                   ARM_THEN, index_times_4, {}, true,
                   [=](nir_builder *b, nir_def *base) {
                      nir_load_global_constant_offset(
                         b, 1, 32, base, nir_imm_int(b, 0x100000),
                         .access = access, .align_mul = 4);
                   })),
                expected);
      EXPECT_EQ(flattens(make_guarded_load(
                   ARM_THEN, index_times_4, {}, true,
                   [=](nir_builder *b, nir_def *base) {
                      nir_load_global_constant_bounded(
                         b, 1, 32, base, nir_imm_int(b, 0x100000),
                         nir_imm_int(b, 16), .access = access, .align_mul = 4);
                   })),
                expected);
      EXPECT_EQ(flattens(make_guarded_load(
                   ARM_THEN, index_times_4, {}, true,
                   [=](nir_builder *b, nir_def *) {
                      nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0),
                                    nir_imm_int(b, 0x100000), .access = access,
                                    .align_mul = 4);
                   })),
                expected);
   }
}

/* The budget counts every memory read of both arms, speculatable or not. */
TEST_F(FlattenLoadsKeeps, TooManyLoads)
{
   EXPECT_FALSE(flattens(make_guarded_load(ARM_THEN, index_times_4), 0));

   const gl_access_qualifier speculate = (gl_access_qualifier)(
      ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
   auto speculatable_read = [=](nir_builder *b, nir_def *base) {
      nir_load_global_constant(b, 1, 32, base, .access = speculate,
                               .align_mul = 4);
   };
   EXPECT_TRUE(flattens(make_guarded_load(ARM_THEN, index_times_4, {}, true,
                                          speculatable_read),
                        2));
   EXPECT_FALSE(flattens(make_guarded_load(ARM_THEN, index_times_4, {}, true,
                                           speculatable_read),
                         1));

   /* The same with the extra read in the other arm, and with a second load
    * to redirect there. */
   auto redirected_read = [](nir_builder *b, nir_def *base) {
      nir_load_global(b, 1, 32, nir_iadd_imm(b, base, 64), .align_mul = 4);
   };
   for (const extra_fn &extra : {extra_fn(speculatable_read),
                                 extra_fn(redirected_read)}) {
      for (enum arm arm : {ARM_THEN, ARM_ELSE}) {
         const enum arm other = arm == ARM_THEN ? ARM_ELSE : ARM_THEN;
         EXPECT_TRUE(flattens(make_guarded_load(arm, index_times_4, {}, true,
                                                extra, other),
                              2));
         EXPECT_FALSE(flattens(make_guarded_load(arm, index_times_4, {}, true,
                                                 extra, other),
                               1));
      }
   }
}

/* Threadgroup loads as the AGX3 pipeline hands them to the pass: a byte
 * offset converted to an element index first, either as the frontend's
 * preprocessing does it (agx_nir_lower_shared_bitsize with element indices,
 * which also narrows the index to 16 bits) or as the backstop in
 * agx_compile_shader_nir() does (agx_nir_lower_shared_element_index). The
 * load then reads from byte BASE + index * scalar bytes. (The frontend splits
 * 64-bit threadgroup accesses first; one left whole converts the same way.) */
enum shared_lowering { LOWER_PREPROCESS, LOWER_BACKSTOP, LOWER_NONE };
static const char *const shared_lowering_names[] = {"Preprocess", "Backstop",
                                                    "Bytes"};

struct shared_case {
   const char *name;
   unsigned shared_size, base, components, bit_size, align_mul, align_offset;
   /* The byte the untaken arm reads from, or ~0 when the if stays. */
   unsigned untaken;
};

static const shared_case shared_cases[] = {
   /* A 32-bit load at byte 4 of 16: index 1. The untaken index is 1, not 4
    * (which would read bytes 16 to 20). */
   {"AlignOffset", 16, 0, 1, 32, 16, 4, 4},
   {"AlignOffsetAtTheEnd", 8, 0, 1, 32, 16, 4, 4},
   {"AlignOffsetPastTheEnd", 7, 0, 1, 32, 16, 4, ~0u},
   /* BASE already at the declared alignment, and BASE plus two elements. */
   {"Base", 12, 8, 1, 32, 16, 8, 8},
   {"BasePastTheEnd", 11, 8, 1, 32, 16, 8, ~0u},
   {"BasePlusElements", 16, 4, 1, 32, 16, 12, 12},
   {"BasePlusElementsPastTheEnd", 15, 4, 1, 32, 16, 12, ~0u},
   /* Vectors and narrow and wide scalars, each ending at the end of the
    * allocation and one byte past it. */
   {"Vec4", 16, 0, 4, 32, 16, 0, 0},
   {"Vec4PastTheEnd", 15, 0, 4, 32, 16, 0, ~0u},
   {"Vec4Base", 32, 16, 4, 32, 16, 0, 16},
   {"U16", 8, 2, 1, 16, 8, 6, 6},
   {"U16PastTheEnd", 7, 2, 1, 16, 8, 6, ~0u},
   {"U16x2", 6, 0, 2, 16, 4, 2, 2},
   {"U16x2PastTheEnd", 5, 0, 2, 16, 4, 2, ~0u},
   {"U8", 4, 0, 1, 8, 4, 3, 3},
   {"U8PastTheEnd", 3, 0, 1, 8, 4, 3, ~0u},
   {"U8x4", 8, 4, 4, 8, 4, 0, 4},
   {"U8x4PastTheEnd", 7, 4, 4, 8, 4, 0, ~0u},
   {"U64", 16, 0, 1, 64, 16, 8, 8},
   {"U64PastTheEnd", 15, 0, 1, 64, 16, 8, ~0u},
   /* No allocation. */
   {"NoAllocation", 0, 0, 1, 32, 4, 0, ~0u},
};

/* if (cond) x = load_shared(i * align_mul + c); else x = 0; (or the other
 * way around), with c the byte offset that gives the whole address BASE + c
 * the declared alignment, lowered as `lowering` says. */
static nir_shader *
make_guarded_shared_load(const shared_case &c, enum arm arm,
                         enum shared_lowering lowering)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "guarded shared load");
   b.shader->info.shared_size = c.shared_size;
   nir_def *i = agx_test_load_preamble(&b, 1, 32, SLOT_I);
   nir_def *cond =
      nir_ine_imm(&b, agx_test_load_preamble(&b, 1, 32, SLOT_COND), 0);
   const unsigned gap = (c.align_offset - c.base) & (c.align_mul - 1);
   nir_def *offset = nir_iadd_imm(&b, nir_imul_imm(&b, i, c.align_mul), gap);
   nir_def *zero = nir_imm_zero(&b, c.components, c.bit_size);

   nir_def *arm_defs[2];
   nir_push_if(&b, cond);
   for (unsigned a = 0; a < 2; ++a) {
      if (a == ARM_ELSE)
         nir_push_else(&b, NULL);
      arm_defs[a] = a == arm
         ? nir_load_shared(&b, c.components, c.bit_size, offset,
                           .base = (int)c.base, .align_mul = c.align_mul,
                           .align_offset = c.align_offset)
         : zero;
   }
   nir_pop_if(&b, NULL);
   nir_def *phi = nir_if_phi(&b, arm_defs[ARM_THEN], arm_defs[ARM_ELSE]);
   nir_store_global(&b, phi, nir_imm_int64(&b, 0), .align_mul = 4);

   if (lowering == LOWER_PREPROCESS) {
      agx_nir_lower_shared_bitsize(b.shader, true);
      b.shader->info.shared_element_index_agx = true;
   } else if (lowering == LOWER_BACKSTOP) {
      agx_nir_lower_shared_element_index(b.shader);
   }
   nir_validate_shader(b.shader, "before flattening");
   return b.shader;
}

/* The byte the shared load of `s` reads from with its arm taken or not. */
static uint64_t
shared_byte_of(nir_shader *s, bool taken)
{
   nir_intrinsic_instr *load = find_load(s, nir_intrinsic_load_shared);
   const uint64_t index = address_of(s, nir_intrinsic_load_shared, taken);
   return nir_intrinsic_base(load) + index * (load->def.bit_size / 8);
}

class FlattenSharedLoads
    : public ::testing::TestWithParam<
         std::tuple<shared_case, shared_lowering, arm>> {
 protected:
   FlattenSharedLoads() { glsl_type_singleton_init_or_ref(); }
   ~FlattenSharedLoads() { glsl_type_singleton_decref(); }
};

/* With its arm taken the load reads the byte it read before; with it not
 * taken, a whole load inside the threadgroup allocation at the load's
 * alignment. A load whose source still counts bytes, or that no index fits,
 * stays an if. */
TEST_P(FlattenSharedLoads, StaysInsideTheAllocation)
{
   const shared_case &c = std::get<0>(GetParam());
   const shared_lowering lowering = std::get<1>(GetParam());
   const enum arm arm = std::get<2>(GetParam());
   const bool then_taken = arm == ARM_THEN;

   nir_shader *s = make_guarded_shared_load(c, arm, lowering);
   const unsigned gap = (c.align_offset - c.base) & (c.align_mul - 1);
   const uint64_t taken_byte = c.base + 5 * c.align_mul + gap;
   const bool flattens = c.untaken != ~0u && lowering != LOWER_NONE;

   ASSERT_EQ(agx_nir_opt_flatten_loads(s, 64, 4), flattens);
   nir_validate_shader(s, "after flattening");
   EXPECT_EQ(count_ifs(s), flattens ? 0u : 1u);
   if (flattens) {
      EXPECT_EQ(shared_byte_of(s, then_taken), taken_byte);

      const uint64_t untaken = shared_byte_of(s, !then_taken);
      const unsigned bytes = c.components * c.bit_size / 8;
      EXPECT_EQ(untaken, c.untaken);
      EXPECT_LE(untaken + bytes, c.shared_size);
      EXPECT_EQ(untaken % c.align_mul, c.align_offset);
   }
   ralloc_free(s);
}

INSTANTIATE_TEST_SUITE_P(
   Forms, FlattenSharedLoads,
   ::testing::Combine(::testing::ValuesIn(shared_cases),
                      ::testing::Values(LOWER_PREPROCESS, LOWER_BACKSTOP,
                                        LOWER_NONE),
                      ::testing::Values(ARM_THEN, ARM_ELSE)),
   [](const ::testing::TestParamInfo<FlattenSharedLoads::ParamType> &info) {
      return std::string(std::get<0>(info.param).name) +
             shared_lowering_names[std::get<1>(info.param)] +
             (std::get<2>(info.param) == ARM_THEN ? "Then" : "Else");
   });

/* The pass is M5's: G18G compiles as AGX3C with emit_agx3d and keeps the if.
 * A guarded load compiles to a different program with and without
 * dont_flatten on M5, to the same one on G18G. */
static std::vector<uint8_t>
compile_guarded_load(bool emit_agx3d, bool dont_flatten)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "guarded load");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *output = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *by_value = nir_load_preamble(&b, 1, 32, .base = 8);
   nir_if *nif = nir_push_if(&b, nir_ine_imm(&b, by_value, 0));
   if (dont_flatten)
      nif->control = nir_selection_control_dont_flatten;
   nir_def *value = nir_iadd_imm(&b, by_value, 1);
   nir_push_else(&b, nif);
   nir_def *loaded = nir_load_global(&b, 1, 32, input, .align_mul = 4);
   nir_pop_if(&b, nif);
   nir_store_global(&b, nir_if_phi(&b, value, loaded), output, .align_mul = 4);
   agx_preprocess_nir(b.shader);

   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = emit_agx3d ? AGX_CHIP_G17G : AGX_CHIP_G17C;
   key.dev.emit_agx3d = emit_agx3d;
   key.reserved_preamble = 10;
   key.no_preamble = true;
   agx_shader_part part = {};
   std::vector<uint8_t> binary;
   EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   if (part.binary) {
      binary.assign((const uint8_t *)part.binary,
                    (const uint8_t *)part.binary + part.info.binary_size);
   }
   free(part.binary);
   ralloc_free(b.shader);
   return binary;
}

TEST_F(FlattenLoadsKeeps, G18GKeepsTheIf)
{
   const std::vector<uint8_t> m5 = compile_guarded_load(false, false);
   EXPECT_FALSE(m5.empty());
   EXPECT_NE(m5, compile_guarded_load(false, true));

   const std::vector<uint8_t> g18g = compile_guarded_load(true, false);
   EXPECT_FALSE(g18g.empty());
   EXPECT_EQ(g18g, compile_guarded_load(true, true));
}

static nir_shader *
make_short_circuit_uniform_load(bool arm_in_then, bool divergent_cond,
                                bool divergent_addr,
                                enum gl_access_qualifier access,
                                bool dont_flatten, bool non_bool_phi,
                                bool non_const_empty, bool both_arms_non_empty,
                                bool subgroup_in_arm = false)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "short circuit uniform load");
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = SLOT_BASE);
   nir_def *ucond = nir_ine_imm(&b, nir_load_preamble(&b, 1, 32, .base = SLOT_COND), 0);
   nir_def *cond = divergent_cond ? nir_ilt_imm(&b, gid, 64) : ucond;
   nir_def *addr = divergent_addr
                      ? nir_iadd(&b, base, nir_imul_imm(&b, nir_u2u64(&b, gid), 4))
                      : base;

   nir_if *nif = nir_push_if(&b, cond);
   if (dont_flatten)
      nif->control = nir_selection_control_dont_flatten;

   nir_def *then_val = NULL;
   nir_def *else_val = NULL;
   auto build_arm = [&]() -> nir_def * {
      nir_def *lim = nir_load_global(&b, 1, 32, addr, .access = access, .align_mul = 4);
      if (non_bool_phi)
         return lim;
      nir_def *v = nir_ilt(&b, gid, lim);
      return subgroup_in_arm ? nir_vote_any(&b, 1, v) : v;
   };
   auto build_empty = [&]() -> nir_def * {
      if (non_bool_phi)
         return non_const_empty ? gid : nir_imm_int(&b, 0);
      return non_const_empty ? nir_ilt_imm(&b, gid, 32) : nir_imm_false(&b);
   };

   if (arm_in_then) {
      then_val = build_arm();
      nir_push_else(&b, nif);
      if (both_arms_non_empty)
         else_val = nir_ilt_imm(&b, nir_iadd_imm(&b, gid, 1), 16);
      else
         else_val = build_empty();
   } else {
      if (both_arms_non_empty)
         then_val = nir_ilt_imm(&b, nir_iadd_imm(&b, gid, 1), 16);
      else
         then_val = build_empty();
      nir_push_else(&b, nif);
      else_val = build_arm();
   }
   nir_pop_if(&b, nif);

   nir_def *res = nir_if_phi(&b, then_val, else_val);
   nir_def *out_val = non_bool_phi ? res : nir_b2i32(&b, res);
   nir_store_global(&b, out_val, nir_imm_int64(&b, 8), .align_mul = 4);
   nir_validate_shader(b.shader, "before short-circuit flattening");
   return b.shader;
}

TEST_F(FlattenLoadsKeeps, ShortCircuitUniformLoadFlattensAndRedirects)
{
   nir_shader *s = make_short_circuit_uniform_load(
      true, true, false, ACCESS_NON_WRITEABLE, false, false, false, false);
   ASSERT_TRUE(agx_nir_opt_flatten_short_circuit_uniform_loads(s));
   nir_validate_shader(s, "after short-circuit flattening");
   EXPECT_EQ(count_ifs(s), 0u);

   nir_intrinsic_instr *load = find_load(s, nir_intrinsic_load_global);
   ASSERT_NE(load, nullptr);
   /* Redirected via bcsel against AGX_ZERO_PAGE_ADDRESS when not speculatable. */
   nir_alu_instr *sel = nir_def_as_alu_or_null(load->src[0].ssa);
   ASSERT_NE(sel, nullptr);
   EXPECT_EQ(sel->op, nir_op_bcsel);
   EXPECT_EQ(nir_src_as_uint(sel->src[2].src), AGX_ZERO_PAGE_ADDRESS);

   /* The redirect tests whether any lane takes the arm, so the address of the
    * uniform load stays uniform. */
   nir_alu_instr *any = nir_def_as_alu_or_null(sel->src[0].src.ssa);
   ASSERT_NE(any, nullptr);
   EXPECT_EQ(any->op, nir_op_ine);
   nir_instr *ballot = nir_def_instr(any->src[0].src.ssa);
   ASSERT_EQ(ballot->type, nir_instr_type_intrinsic);
   EXPECT_EQ(nir_instr_as_intrinsic(ballot)->intrinsic, nir_intrinsic_ballot);
   nir_divergence_analysis(s);
   EXPECT_FALSE(load->src[0].ssa->divergent);
   ralloc_free(s);
}

/* With the loads in the else arm, the arm is taken when the condition is
 * false, and the phi becomes bcsel(cond, then value, else value). */
TEST_F(FlattenLoadsKeeps, ShortCircuitElseArmSelectsInOrder)
{
   nir_shader *s = make_short_circuit_uniform_load(
      false, true, false, ACCESS_NON_WRITEABLE, false, false, false, false);
   ASSERT_TRUE(agx_nir_opt_flatten_short_circuit_uniform_loads(s));
   nir_validate_shader(s, "after short-circuit flattening");
   EXPECT_EQ(count_ifs(s), 0u);

   nir_intrinsic_instr *load = find_load(s, nir_intrinsic_load_global);
   ASSERT_NE(load, nullptr);
   nir_alu_instr *addr_sel = nir_def_as_alu_or_null(load->src[0].ssa);
   ASSERT_NE(addr_sel, nullptr);
   ASSERT_EQ(addr_sel->op, nir_op_bcsel);
   nir_alu_instr *any = nir_def_as_alu_or_null(addr_sel->src[0].src.ssa);
   ASSERT_NE(any, nullptr);
   ASSERT_EQ(any->op, nir_op_ine);
   nir_instr *ballot_instr = nir_def_instr(any->src[0].src.ssa);
   ASSERT_EQ(ballot_instr->type, nir_instr_type_intrinsic);
   nir_intrinsic_instr *ballot = nir_instr_as_intrinsic(ballot_instr);
   ASSERT_EQ(ballot->intrinsic, nir_intrinsic_ballot);
   nir_alu_instr *taken = nir_def_as_alu_or_null(ballot->src[0].ssa);
   ASSERT_NE(taken, nullptr);
   EXPECT_EQ(taken->op, nir_op_inot);
   nir_def *cond = taken->src[0].src.ssa;

   /* The stored value is b2i32(bcsel(cond, false, gid < load)). */
   nir_alu_instr *result = NULL;
   nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_alu &&
             nir_instr_as_alu(instr)->op == nir_op_b2i32)
            result = nir_def_as_alu_or_null(
               nir_instr_as_alu(instr)->src[0].src.ssa);
      }
   }
   ASSERT_NE(result, nullptr);
   ASSERT_EQ(result->op, nir_op_bcsel);
   EXPECT_EQ(result->src[0].src.ssa, cond);
   ASSERT_TRUE(nir_src_is_const(result->src[1].src));
   EXPECT_FALSE(nir_src_as_bool(result->src[1].src));
   nir_alu_instr *cmp = nir_def_as_alu_or_null(result->src[2].src.ssa);
   ASSERT_NE(cmp, nullptr);
   EXPECT_EQ(cmp->op, nir_op_ilt);
   EXPECT_EQ(cmp->src[1].src.ssa, &load->def);
   ralloc_free(s);
}

TEST_F(FlattenLoadsKeeps, ShortCircuitSpeculatableUniformLoadKeepsAddress)
{
   nir_shader *s = make_short_circuit_uniform_load(
      false, true, false,
      (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_SPECULATE),
      false, false, false, false);
   ASSERT_TRUE(agx_nir_opt_flatten_short_circuit_uniform_loads(s));
   nir_validate_shader(s, "after short-circuit flattening");
   EXPECT_EQ(count_ifs(s), 0u);

   nir_intrinsic_instr *load = find_load(s, nir_intrinsic_load_global);
   ASSERT_NE(load, nullptr);
   /* Already speculatable: address remains the original uniform preamble load. */
   nir_instr *addr_instr = nir_def_instr(load->src[0].ssa);
   EXPECT_EQ(addr_instr->type, nir_instr_type_intrinsic);
   EXPECT_EQ(nir_instr_as_intrinsic(addr_instr)->intrinsic,
             nir_intrinsic_load_preamble);
   ralloc_free(s);
}

TEST_F(FlattenLoadsKeeps, ShortCircuitUniformLoadRejectsUnsafeShapes)
{
   struct {
      const char *name;
      bool divergent_cond;
      bool divergent_addr;
      enum gl_access_qualifier access;
      bool dont_flatten;
      bool non_bool_phi;
      bool non_const_empty;
      bool both_arms_non_empty;
      bool subgroup_in_arm;
   } cases[] = {
      {"uniform_cond", false, false, ACCESS_NON_WRITEABLE, false, false, false, false},
      {"divergent_addr", true, true, ACCESS_NON_WRITEABLE, false, false, false, false},
      {"writable_load", true, false, (enum gl_access_qualifier)0, false, false, false, false},
      {"coherent_load", true, false, (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_COHERENT), false, false, false, false},
      {"volatile_load", true, false, (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_VOLATILE), false, false, false, false},
      {"dont_flatten", true, false, ACCESS_NON_WRITEABLE, true, false, false, false},
      {"non_bool_phi", true, false, ACCESS_NON_WRITEABLE, false, true, false, false},
      {"non_const_empty", true, false, ACCESS_NON_WRITEABLE, false, false, true, false},
      {"both_arms_non_empty", true, false, ACCESS_NON_WRITEABLE, false, false, false, true},
      {"subgroup_op_in_arm", true, false, ACCESS_NON_WRITEABLE, false, false, false, false, true},
   };

   for (const auto &c : cases) {
      nir_shader *s = make_short_circuit_uniform_load(
         true, c.divergent_cond, c.divergent_addr, c.access, c.dont_flatten,
         c.non_bool_phi, c.non_const_empty, c.both_arms_non_empty,
         c.subgroup_in_arm);
      EXPECT_FALSE(agx_nir_opt_flatten_short_circuit_uniform_loads(s)) << c.name;
      EXPECT_EQ(count_ifs(s), 1u) << c.name;
      ralloc_free(s);
   }
}

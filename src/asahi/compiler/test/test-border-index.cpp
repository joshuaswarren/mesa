#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "asahi/vulkan/hk_nir_border_index.h"
}

TEST(BorderIndex, MatchesOnlyRelatedImmutableDescriptorFields)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "border descriptor provenance");
   nir_def *base = nir_load_global_invocation_id(&b, 32);
   nir_def *ptr = nir_u2u64(&b, nir_channel(&b, base, 0));
   nir_def *other = nir_u2u64(&b, nir_channel(&b, base, 1));
   auto load = [&](nir_def *p, unsigned offset, unsigned bits, bool commute,
                   enum gl_access_qualifier access) {
      nir_def *constant = nir_imm_int64(&b, offset);
      nir_def *addr = commute ? nir_iadd(&b, constant, p)
                             : nir_iadd(&b, p, constant);
      return nir_load_global_constant(&b, 1, bits, addr, .access = access,
                                      .align_mul = 2);
   };
   auto original = load(ptr, 690, 16, false, ACCESS_CAN_SPECULATE);
   auto unbiased = load(ptr, 682, 16, false, ACCESS_CAN_SPECULATE);
   EXPECT_TRUE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(original, 0), nir_get_scalar(unbiased, 0), 8));
   auto commuted = load(ptr, 682, 16, true, ACCESS_CAN_SPECULATE);
   EXPECT_TRUE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(original, 0), nir_get_scalar(commuted, 0), 8));
   auto wrong_base = load(other, 682, 16, false, ACCESS_CAN_SPECULATE);
   auto wrong_field = load(ptr, 680, 16, false, ACCESS_CAN_SPECULATE);
   auto wrong_width = load(ptr, 682, 32, false, ACCESS_CAN_SPECULATE);
   auto volatile_load = load(ptr, 682, 16, false, ACCESS_VOLATILE);
   for (nir_def *invalid : {wrong_base, wrong_field, wrong_width, volatile_load}) {
      EXPECT_FALSE(hk_nir_border_indices_same_descriptor(
         nir_get_scalar(original, 0), nir_get_scalar(invalid, 0), 8));
   }
   nir_def *variable = nir_load_global_constant(
      &b, 1, 16, nir_iadd(&b, ptr, other), .align_mul = 2);
   EXPECT_FALSE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(original, 0), nir_get_scalar(variable, 0), 8));

   /* Dynamically-indexed sampler arrays compute set_ptr + u2u64(dyn + offs). */
   nir_def *idx = nir_channel(&b, base, 2);
   nir_def *dyn = nir_imul_imm(&b, idx, 72);
   nir_def *other_dyn = nir_imul_imm(&b, nir_channel(&b, base, 0), 72);
   auto dyn_load = [&](nir_def *p, nir_def *d, unsigned offset, bool commute_outer,
                       bool commute_inner) {
      nir_def *off32 = offset
                          ? (commute_inner ? nir_iadd(&b, nir_imm_int(&b, offset), d)
                                           : nir_iadd(&b, d, nir_imm_int(&b, offset)))
                          : d;
      nir_def *off64 = nir_u2u64(&b, off32);
      nir_def *addr = commute_outer ? nir_iadd(&b, off64, p) : nir_iadd(&b, p, off64);
      return nir_load_global_constant(&b, 1, 16, addr, .access = ACCESS_CAN_SPECULATE,
                                      .align_mul = 2);
   };
   nir_def *dyn_orig = dyn_load(ptr, dyn, 42, false, false);
   nir_def *dyn_unb = dyn_load(ptr, dyn, 34, false, false);
   EXPECT_TRUE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(dyn_orig, 0), nir_get_scalar(dyn_unb, 0), 8));
   EXPECT_TRUE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(dyn_load(ptr, dyn, 42, true, true), 0),
      nir_get_scalar(dyn_unb, 0), 8));
   EXPECT_TRUE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(dyn_load(ptr, dyn, 8, false, false), 0),
      nir_get_scalar(dyn_load(ptr, dyn, 0, false, false), 0), 8));

   /* Power-of-two descriptor strides can fold iadd into ior. */
   nir_def *pow2_dyn = nir_ishl_imm(&b, idx, 6);
   nir_def *ior_orig = nir_load_global_constant(
      &b, 1, 16, nir_iadd(&b, ptr, nir_u2u64(&b, nir_ior_imm(&b, pow2_dyn, 42))),
      .access = ACCESS_CAN_SPECULATE, .align_mul = 2);
   nir_def *ior_unb = nir_load_global_constant(
      &b, 1, 16, nir_iadd(&b, ptr, nir_u2u64(&b, nir_ior_imm(&b, pow2_dyn, 34))),
      .access = ACCESS_CAN_SPECULATE, .align_mul = 2);
   EXPECT_TRUE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(ior_orig, 0), nir_get_scalar(ior_unb, 0), 8));

   /* Reject mismatched dynamic index, base, offset delta, or non-disjoint ior. */
   EXPECT_FALSE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(dyn_orig, 0),
      nir_get_scalar(dyn_load(ptr, other_dyn, 34, false, false), 0), 8));
   EXPECT_FALSE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(dyn_orig, 0),
      nir_get_scalar(dyn_load(other, dyn, 34, false, false), 0), 8));
   EXPECT_FALSE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(dyn_orig, 0),
      nir_get_scalar(dyn_load(ptr, dyn, 32, false, false), 0), 8));
   EXPECT_FALSE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(dyn_unb, 0), nir_get_scalar(dyn_orig, 0), 8));
   nir_def *bad_ior = nir_load_global_constant(
      &b, 1, 16, nir_iadd(&b, ptr, nir_u2u64(&b, nir_ior_imm(&b, dyn, 42))),
      .access = ACCESS_CAN_SPECULATE, .align_mul = 2);
   EXPECT_FALSE(hk_nir_border_indices_same_descriptor(
      nir_get_scalar(bad_ior, 0),
      nir_get_scalar(dyn_load(ptr, dyn, 34, false, false), 0), 8));

   nir_validate_shader(b.shader, "border descriptor provenance");
   ralloc_free(b.shader);
}

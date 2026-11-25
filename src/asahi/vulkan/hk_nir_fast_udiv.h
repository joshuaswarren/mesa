/* SPDX-License-Identifier: MIT */
#ifndef HK_NIR_FAST_UDIV_H
#define HK_NIR_FAST_UDIV_H

#include "compiler/nir/nir_builder.h"

struct hk_nir_fast_udiv {
   nir_def *den;
   nir_def *m_prime;
   nir_def *sh1;
   nir_def *sh2;
};

/* Exact Granlund-Montgomery / Hacker's Delight (section 10-9) unsigned 32-bit
 * division preparation for any uniform divisor d >= 1 (all 2^32 - 1 nonzero
 * uint32_t divisors: d == 1, powers of 2, odd, even, and d >= 2^31):
 *
 *   l       = (d <= 1) ? 0 : (ufind_msb(d - 1) + 1)         in [0, 32]
 *   rem_hi  = (1ULL << l) - (uint64_t)d                     in [0, 2^31 - 1]
 *   m_prime = (uint32_t)(((rem_hi << 32) / (uint64_t)d) + 1ULL)
 *   sh1     = (l > 0) ? 1 : 0
 *   sh2     = (l > 0) ? (l - 1) : 0
 *
 * Computed in pure 32-bit branchless ALU with zero 64-bit integer division and
 * zero loops: two conservative FP32 reciprocal steps (scaled by c = 1 - 2^-18
 * so neither step can overestimate or overflow uint32_t) with exact 64-bit
 * residuals reconstructed from (umul_high, imul), followed by two branchless
 * remainder corrections.
 *
 * Error bound. Let Q = floor((rem_hi << 32) / d) < 2^32, u = 2^-24 (u2f32,
 * fmul and fadd round to nearest even; f2u32 truncates) and let the hardware
 * reciprocal return frcp(x) = (1 / x)(1 + e). This frcp is not exact, so
 * agx_nir_lower_fdiv() does not refine it: e is the raw hardware error.
 *
 *  - Each step's estimate is its exact quotient times c (1 + e) and at most
 *    four rounding factors (1 + u) over one (1 - u) (step 2: u2f32 of r0_lo,
 *    the fadd, u2f32 of d, the fmul by c and the final fmul). Neither step
 *    overestimates, so q0 <= q1 <= Q and the residual r1 is never negative,
 *    when c (1 + e)(1 + u)^4 / (1 - u) <= 1, i.e. e <= 2^-18.117: an frcp that
 *    overestimates by at most 29 ulp.
 *  - With eta_k = 1 - c (1 - |e|)(1 - u)^k / (1 + u), step 1 leaves
 *    Q - q0 < eta_3 2^32 + 1, so r0 < 2^56 and r0_hi converts exactly, and
 *    step 2 leaves Q - q1 < eta_4 (eta_3 2^32 + 2) + 1. That is at most 1
 *    (one correction) for an underestimate of up to 93 ulp and at most 2
 *    (both corrections) up to 146 ulp, so r < d on exit and m_prime = Q + 1.
 *
 * Measured on G17P over every float input in [1, 2^33), frcp is within
 * 0.72 ulp in both directions.
 */
static inline struct hk_nir_fast_udiv
hk_nir_fast_udiv_prepare(nir_builder *b, nir_def *den)
{
   nir_def *safe_den = nir_umax(b, den, nir_imm_int(b, 1));
   nir_def *is_gt1 = nir_ugt_imm(b, safe_den, 1);
   nir_def *l = nir_bcsel(
      b, is_gt1,
      nir_iadd_imm(b, nir_ufind_msb(b, nir_iadd_imm(b, safe_den, -1)), 1),
      nir_imm_int(b, 0));

   /* rem_hi = (1ULL << l) - safe_den fits strictly in [0, 2^31 - 1]. */
   nir_def *rem_hi = nir_bcsel(
      b, nir_ieq_imm(b, l, 32), nir_ineg(b, safe_den),
      nir_isub(b, nir_ishl(b, nir_imm_int(b, 1), l), safe_den));

   /* Conservative reciprocal rcp_lo = frcp(safe_den) * (1 - 2^-18). */
   nir_def *rcp_lo = nir_fmul_imm(
      b, nir_frcp(b, nir_u2f32(b, safe_den)), 0x1.ffff8p-1f);
   nir_def *two32 = nir_imm_float(b, 4294967296.0f);

   /* Step 1: initial underestimate q0 <= floor((rem_hi << 32) / safe_den). */
   nir_def *q0 = nir_f2u32(
      b, nir_fmul(b, nir_fmul(b, nir_u2f32(b, rem_hi), two32), rcp_lo));

   /* Exact 64-bit residual r0 = (rem_hi << 32) - q0 * safe_den in (r0_hi, r0_lo). */
   nir_def *p0_lo = nir_imul(b, q0, safe_den);
   nir_def *p0_hi = nir_umul_high(b, q0, safe_den);
   nir_def *r0_lo = nir_ineg(b, p0_lo);
   nir_def *b0 = nir_b2i32(b, nir_ine_imm(b, p0_lo, 0));
   nir_def *r0_hi = nir_isub(b, nir_isub(b, rem_hi, p0_hi), b0);

   /* Step 2: refine residual r0 -> dq1 <= floor(r0 / safe_den). */
   nir_def *f_r0 = nir_fadd(
      b, nir_fmul(b, nir_u2f32(b, r0_hi), two32), nir_u2f32(b, r0_lo));
   nir_def *dq1 = nir_f2u32(b, nir_fmul(b, f_r0, rcp_lo));
   nir_def *q1 = nir_iadd(b, q0, dq1);

   /* Exact 64-bit residual r1 = (rem_hi << 32) - q1 * safe_den < 3 * safe_den. */
   nir_def *p1_lo = nir_imul(b, q1, safe_den);
   nir_def *p1_hi = nir_umul_high(b, q1, safe_den);
   nir_def *r1_lo = nir_ineg(b, p1_lo);
   nir_def *b1 = nir_b2i32(b, nir_ine_imm(b, p1_lo, 0));
   nir_def *r1_hi = nir_isub(b, nir_isub(b, rem_hi, p1_hi), b1);

   /* Step 3: two branchless corrections ensure r < safe_den for all d >= 1. */
   nir_def *q = q1;
   nir_def *r_hi = r1_hi;
   nir_def *r_lo = r1_lo;
   for (unsigned step = 0; step < 2; ++step) {
      nir_def *ge = nir_ior(b, nir_ine_imm(b, r_hi, 0),
                            nir_uge(b, r_lo, safe_den));
      nir_def *sub_borrow = nir_b2i32(b, nir_ult(b, r_lo, safe_den));
      q = nir_iadd(b, q, nir_b2i32(b, ge));
      r_lo = nir_bcsel(b, ge, nir_isub(b, r_lo, safe_den), r_lo);
      r_hi = nir_bcsel(b, ge, nir_isub(b, r_hi, sub_borrow), r_hi);
   }

   nir_def *m_prime = nir_iadd_imm(b, q, 1);
   nir_def *l_gt0 = nir_ugt_imm(b, l, 0);
   nir_def *sh1 = nir_bcsel(b, l_gt0, nir_imm_int(b, 1), nir_imm_int(b, 0));
   nir_def *sh2 = nir_bcsel(b, l_gt0, nir_iadd_imm(b, l, -1), nir_imm_int(b, 0));
   return (struct hk_nir_fast_udiv){
      .den = safe_den,
      .m_prime = m_prime,
      .sh1 = sh1,
      .sh2 = sh2,
   };
}

static inline void
hk_nir_fast_udivmod(nir_builder *b, const struct hk_nir_fast_udiv *fd,
                    nir_def *num, nir_def **q_out, nir_def **r_out)
{
   nir_def *t1 = nir_umul_high(b, fd->m_prime, num);
   nir_def *q = nir_ushr(
      b, nir_iadd(b, t1, nir_ushr(b, nir_isub(b, num, t1), fd->sh1)), fd->sh2);
   nir_def *r = nir_isub(b, num, nir_imul(b, q, fd->den));
   *q_out = q;
   if (r_out)
      *r_out = r;
}

#endif

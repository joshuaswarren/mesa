/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"

static nir_def *
repair_tininess(nir_builder *b, nir_def *a, nir_def *c, nir_def *z,
                nir_def *native)
{
   /* G17P and M5 detect FMA tininess before rounding. Repair only a flushed
    * result whose exact magnitude lies in [2^-126 - 2^-150, 2^-126). RTE rounds that
    * interval (including its even tie) to the smallest normal.
    *
    * Work in units of 2^-150. The exact product has 48 significant bits.
    * Keep its integer part in 96 bits and retain a sticky fractional bit;
    * a normal addend is an integer in these units. This also handles a tiny
    * opposite-sign product subtracted from a normal addend, without losing
    * its low bits in floating point. FP32 subnormal inputs are flushed.
    */
   nir_def *ea = nir_iand_imm(b, nir_ushr_imm(b, a, 23), 255);
   nir_def *ec = nir_iand_imm(b, nir_ushr_imm(b, c, 23), 255);
   nir_def *ez = nir_iand_imm(b, nir_ushr_imm(b, z, 23), 255);
   nir_def *sum = nir_iadd(b, ea, ec);
   nir_def *ma = nir_ior_imm(b, nir_iand_imm(b, a, 0x7fffff), 0x800000);
   nir_def *mc = nir_ior_imm(b, nir_iand_imm(b, c, 0x7fffff), 0x800000);
   nir_def *mz =
      nir_bcsel(b, nir_ieq_imm(b, ez, 0), nir_imm_int(b, 0),
                nir_ior_imm(b, nir_iand_imm(b, z, 0x7fffff), 0x800000));
   nir_def *product = nir_imul(b, nir_u2u64(b, ma), nir_u2u64(b, mc));
   nir_def *left =
      nir_umin_imm(b, nir_usub_sat(b, sum, nir_imm_int(b, 150)), 24);
   nir_def *right =
      nir_umin_imm(b, nir_usub_sat(b, nir_imm_int(b, 150), sum), 63);
   nir_def *fractional =
      nir_ine(b, product, nir_ishl(b, nir_ushr(b, product, right), right));
   nir_def *pl = nir_ishl(b, nir_ushr(b, product, right), left);
   /* NIR masks shift counts: explicitly select zero for a shift by 64. */
   nir_def *ph = nir_bcsel(
      b, nir_ieq_imm(b, left, 0), nir_imm_int(b, 0),
      nir_u2u32(b,
                nir_ushr(b, product, nir_isub(b, nir_imm_int(b, 64), left))));
   nir_def *zs = nir_umin_imm(b, ez, 48);
   nir_def *zl = nir_ishl(b, nir_u2u64(b, mz), zs);
   nir_def *zh =
      nir_bcsel(b, nir_ieq_imm(b, zs, 0), nir_imm_int(b, 0),
                nir_u2u32(b, nir_ushr(b, nir_u2u64(b, mz),
                                      nir_isub(b, nir_imm_int(b, 64), zs))));
   nir_def *p_ge_z =
      nir_ior(b, nir_ult(b, zh, ph),
              nir_iand(b, nir_ieq(b, ph, zh), nir_uge(b, pl, zl)));
   nir_def *large_l = nir_bcsel(b, p_ge_z, pl, zl);
   nir_def *small_l = nir_bcsel(b, p_ge_z, zl, pl);
   nir_def *large_h = nir_bcsel(b, p_ge_z, ph, zh);
   nir_def *small_h = nir_bcsel(b, p_ge_z, zh, ph);
   nir_def *dl = nir_isub(b, large_l, small_l);
   nir_def *dh = nir_isub(b, nir_isub(b, large_h, small_h),
                          nir_b2i32(b, nir_ult(b, large_l, small_l)));
   /* Z - (floor(P) + fraction): borrow one unit when the fraction is nonzero. */
   nir_def *borrow = nir_iand(b, nir_inot(b, p_ge_z), fractional);
   dh =
      nir_isub(b, dh, nir_b2i32(b, nir_iand(b, borrow, nir_ieq_imm(b, dl, 0))));
   dl = nir_isub(b, dl, nir_b2i64(b, borrow));
   nir_def *sl = nir_iadd(b, pl, zl);
   nir_def *sh =
      nir_iadd(b, nir_iadd(b, ph, zh), nir_b2i32(b, nir_ult(b, sl, pl)));
   nir_def *opposite = nir_ine_imm(
      b, nir_iand_imm(b, nir_ixor(b, nir_ixor(b, a, c), z), 0x80000000u), 0);
   nir_def *lo = nir_bcsel(b, opposite, dl, sl);
   nir_def *hi = nir_bcsel(b, opposite, dh, sh);
   nir_def *sign = nir_bcsel(b, nir_iand(b, opposite, nir_inot(b, p_ge_z)), z,
                             nir_ixor(b, a, c));
   nir_def *fix =
      nir_iand(b, nir_ieq_imm(b, hi, 0), nir_ieq_imm(b, lo, 0xffffff));
   /* For larger products/addends, cancellation is quantized in units larger
    * than the interval above. Such a result cannot lie in that interval.
    * Clamp all shifts above even for these rejected inputs.
    */
   fix = nir_iand(
      b, fix, nir_iand(b, nir_ult_imm(b, sum, 175), nir_ult_imm(b, ez, 49)));
   fix = nir_iand(b, fix,
                  nir_iand(b, nir_ine_imm(b, ea, 0), nir_ine_imm(b, ec, 0)));
   fix =
      nir_iand(b, fix, nir_ieq_imm(b, nir_iand_imm(b, native, 0x7fffffff), 0));
   nir_def *normal =
      nir_ior_imm(b, nir_iand_imm(b, sign, 0x80000000u), 0x800000);
   return nir_bcsel(b, fix, normal, native);
}

nir_def *
agx_nir_fma_rte_fixup(nir_builder *b, nir_def *a, nir_def *c, nir_def *z,
                      nir_def *native)
{
   /* The repair only changes a flushed zero. Keep its multiword integer
    * arithmetic off the normal-result path, particularly for shaders which
    * explicitly request RTE for every FP32 operation. A select would execute
    * that arithmetic for every FMA even when the native result is normal.
    */
   nir_def *is_zero =
      nir_ieq_imm(b, nir_iand_imm(b, native, 0x7fffffff), 0);
   nir_if *repair = nir_push_if(b, is_zero);
   nir_def *fixed = repair_tininess(b, a, c, z, native);
   nir_push_else(b, repair);
   nir_pop_if(b, repair);
   return nir_if_phi(b, fixed, native);
}

static bool
filter(const nir_instr *instr, const void *data)
{
   return instr->type == nir_instr_type_alu &&
          nir_instr_as_alu(instr)->op == nir_op_ffma &&
          nir_instr_as_alu(instr)->def.bit_size == 32;
}

static nir_def *
lower(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *c = nir_ssa_for_alu_src(b, alu, 1);
   nir_def *z = nir_ssa_for_alu_src(b, alu, 2);

   /* The shader is not scalarized yet, and the repair branches on each
    * component's native result. */
   nir_def *fixed[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < alu->def.num_components; ++i) {
      fixed[i] = agx_nir_fma_rte_fixup(b, nir_channel(b, a, i),
                                       nir_channel(b, c, i),
                                       nir_channel(b, z, i),
                                       nir_channel(b, &alu->def, i));
   }
   return alu->def.num_components == 1
             ? fixed[0]
             : nir_vec(b, fixed, alu->def.num_components);
}

static bool
fma_rte_applies(const nir_shader *shader)
{
   unsigned mode = shader->info.float_controls_execution_mode;
   return (mode & FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32) &&
          !(mode & FLOAT_CONTROLS_DENORM_PRESERVE_FP32);
}

/* G17P and the AGX3C parts detect FP32 FMA tininess before rounding. */
static bool
fma_tininess_before_rounding(enum agx_chip chip)
{
   return chip == AGX_CHIP_G17P ||
          agx_chip_isa_encoding(chip) == AGX_ISA_ENCODING_AGX3C;
}

/*
 * Only a strict ffma promises one correctly rounded result: OpFmaKHR
 * (VK_KHR_shader_fma) and OpenCL fma() reach NIR as ffma. GLSL.std.450 Fma
 * reaches it as ffma_weak, which may be evaluated unfused, and NIR fuses
 * contractable multiply-adds into ffma by itself; neither carries the contract
 * this repair keeps. nir_opt_algebraic rewrites ffma_weak into ffma, after
 * which the two cannot be told apart, so the driver runs this once, where a
 * shader enters it and before any algebraic optimization
 * (hk_preprocess_nir_internal(), agx_shader_initialize()). The repair guards
 * the native ffma it keeps, so a second run would repair it again.
 */
bool
agx_nir_lower_fma_rte(nir_shader *shader, enum agx_chip chip)
{
   if (!fma_tininess_before_rounding(chip) || !fma_rte_applies(shader))
      return false;
   return nir_shader_lower_instructions(shader, filter, lower, NULL);
}

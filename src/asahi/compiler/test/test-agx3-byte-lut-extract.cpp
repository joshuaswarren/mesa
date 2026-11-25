/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/*
 * The byte read out of a packed loop-invariant byte LUT
 * (agx_nir_opt_pack_byte_lut_select) on every AGX3 compute target.
 *
 * The shader is the inner loop of a FAST-style corner test: 16 ring bytes
 * loaded once, then a loop over 24 ring positions that reads byte (i & 15),
 * subtracts it from the centre pixel and tracks a run of hits. The run resets
 * the position by 8, so the loop has no trip count and is neither unrolled
 * nor peeled; the LUT pass packs the ring into four words and reads one byte
 * of the selected word per iteration.
 *
 * After the pass the byte must be one bitfield extract (no shift and mask,
 * no modulo-32 count), and the centre-minus-byte subtract must stay one
 * isub: the pass's cleanup must not re-run the main nir_opt_algebraic, which
 * would split every isub of the shader back into an ineg plus an iadd after
 * nir_opt_algebraic_late had recombined them.
 */

#include <cstdio>
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_constant_expressions.h"

extern "C" {
#include "agx_compile.h"
#include "agx_nir.h"
}

namespace {

struct target {
   agx_isa_encoding encoding;
   agx_chip chip;
   const char *name;
};

const target targets[] = {
   {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G15G, "G15G"},
   {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G16G, "G16G"},
   {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, "G17P"},
   {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, "G17C"},
};

nir_def *
load_byte(nir_builder *b, nir_def *base, nir_def *offset)
{
   nir_def *addr = nir_iadd(b, base, nir_u2u64(b, offset));
   nir_def *v = nir_load_global(b, 1, 8, addr);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(v);
   nir_intrinsic_set_align(intr, 1, 0);
   nir_intrinsic_set_access(intr, ACCESS_NON_WRITEABLE);
   return v;
}

/* Balanced select tree over ring[lo..hi] indexed by idx. */
nir_def *
ring_tree(nir_builder *b, nir_def *idx, nir_def *const ring[16], unsigned lo,
          unsigned hi)
{
   if (lo == hi)
      return nir_u2u32(b, ring[lo]);
   const unsigned mid = (lo + hi + 1) / 2;
   return nir_bcsel(b, nir_ult_imm(b, idx, mid),
                    ring_tree(b, idx, ring, lo, mid - 1),
                    ring_tree(b, idx, ring, mid, hi));
}

nir_shader *
build_arc_loop(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "arc");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *src = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *dst = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *thr = agx_test_load_preamble(&b, 1, 32, 8);
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);

   nir_def *ring[16];
   for (unsigned k = 0; k < 16; ++k)
      ring[k] = load_byte(&b, src, nir_iadd_imm(&b, gid, 7 * k));
   nir_def *centre = nir_u2u32(&b, load_byte(&b, src, nir_iadd_imm(&b, gid, 200)));

   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *run = nir_local_variable_create(b.impl, glsl_uint_type(), "run");
   nir_variable *best = nir_local_variable_create(b.impl, glsl_int_type(), "best");
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, run, nir_imm_int(&b, 9), 1);
   nir_store_var(&b, best, nir_imm_int(&b, 0), 1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_uge_imm(&b, i, 24));
      nir_def *v = ring_tree(&b, nir_iand_imm(&b, i, 15), ring, 0, 15);
      nir_def *d = nir_isub(&b, centre, v);
      nir_store_var(&b, best, nir_imax(&b, nir_load_var(&b, best), d), 1);

      nir_def *hit = nir_ilt(&b, thr, d);
      nir_def *r = nir_bcsel(&b, hit, nir_iadd_imm(&b, nir_load_var(&b, run), -1),
                             nir_imm_int(&b, 9));
      nir_def *done = nir_ieq_imm(&b, r, 0);
      nir_store_var(&b, run, nir_bcsel(&b, done, nir_imm_int(&b, 9), r), 1);
      nir_def *next = nir_bcsel(&b, done, nir_iadd_imm(&b, i, -8), i);
      nir_store_var(&b, iv, nir_iadd_imm(&b, next, 1), 1);
   }
   nir_pop_loop(&b, loop);

   nir_def *out = nir_iadd(&b, dst, nir_u2u64(&b, nir_imul_imm(&b, gid, 4)));
   nir_intrinsic_instr *st = nir_store_global(&b, nir_load_var(&b, best), out);
   nir_intrinsic_set_align(st, 4, 0);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   return b.shader;
}

bool
in_loop(nir_block *block)
{
   for (nir_cf_node *p = block->cf_node.parent; p; p = p->parent) {
      if (p->type == nir_cf_node_loop)
         return true;
   }
   return false;
}

unsigned
count_alu_in_loops(nir_shader *s, nir_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, s) {
      if (impl->function->is_preamble)
         continue;
      nir_foreach_block(block, impl) {
         if (!in_loop(block))
            continue;
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               n++;
         }
      }
   }
   return n;
}

struct result {
   bool ok;
   unsigned instrs;
   unsigned ubfe, ushr, ineg, isub;
};

result
compile_arc_loop(const target &t)
{
   nir_shader *nir = build_arc_loop();
   agx_preprocess_nir(nir);

   agx_shader_key key = {};
   key.dev.encoding = t.encoding;
   key.dev.chip = t.chip;
   key.no_preamble = true;
   key.reserved_preamble = 16;
   agx_shader_part part = {};

   result r = {};
   r.ok = agx_compile_shader_nir(nir, &key, &part);
   r.instrs = part.info.stats.instrs;
   /* agx_compile_shader_nir optimizes the shader it is given in place. */
   r.ubfe = count_alu_in_loops(nir, nir_op_ubitfield_extract);
   r.ushr = count_alu_in_loops(nir, nir_op_ushr);
   r.ineg = count_alu_in_loops(nir, nir_op_ineg);
   r.isub = count_alu_in_loops(nir, nir_op_isub);
   free(part.binary);
   ralloc_free(nir);
   return r;
}

/* Value of s when the loop position is i_val and the ring holds bytes[]. */
uint32_t
eval(nir_scalar s, const uint8_t bytes[16], nir_def *const ring[16],
     nir_def *i_def, uint32_t i_val)
{
   if (s.def == i_def)
      return i_val;
   for (unsigned k = 0; k < 16; ++k) {
      if (s.def == ring[k])
         return bytes[k];
   }
   if (nir_scalar_is_const(s))
      return (uint32_t)nir_scalar_as_uint(s);
   EXPECT_TRUE(nir_scalar_is_alu(s));
   nir_alu_instr *alu = nir_def_as_alu(s.def);
   const nir_op_info *info = &nir_op_infos[alu->op];
   nir_const_value src[4][NIR_MAX_VEC_COMPONENTS] = {};
   nir_const_value *srcs[4];
   for (unsigned i = 0; i < info->num_inputs; ++i) {
      nir_scalar in = nir_scalar_chase_alu_src(s, i);
      src[i][0] = nir_const_value_for_uint(eval(in, bytes, ring, i_def, i_val),
                                           in.def->bit_size);
      srcs[i] = src[i];
   }
   nir_const_value dest[NIR_MAX_VEC_COMPONENTS] = {};
   uint16_t poison = 0;
   nir_eval_const_opcode(alu->op, dest, &poison, 1, alu->def.bit_size, srcs, 0);
   return (uint32_t)nir_const_value_as_uint(dest[0], alu->def.bit_size);
}

/* A ring read whose index is not a mask of the position: idx = umin(i, 15)
 * (positions past 15 clamp to the last byte). Its byte lane is idx & 3, not a
 * mask of i: evaluate every position. */
void
check_clamped_index(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "clamp");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *src = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *ring[16];
   for (unsigned k = 0; k < 16; ++k)
      ring[k] = load_byte(&b, src, nir_imm_int(&b, 3 * k));

   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_def *idx_def = NULL, *sum_def = NULL;
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_uge_imm(&b, i, 24));
      idx_def = nir_umin(&b, i, nir_imm_int(&b, 15));
      sum_def = nir_iadd(&b, nir_load_var(&b, acc),
                         ring_tree(&b, idx_def, ring, 0, 15));
      nir_store_var(&b, acc, sum_def, 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   nir_intrinsic_instr *st =
      nir_store_global(&b, nir_load_var(&b, acc),
                       nir_iadd_imm(&b, src, 4096));
   nir_intrinsic_set_align(st, 4, 0);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);
   /* The loop position i: its phi, past the copy vars_to_ssa leaves. */
   nir_def *i_def =
      nir_scalar_chase_movs(nir_get_scalar(nir_def_as_alu(idx_def)->src[0].src.ssa, 0)).def;

   ASSERT_TRUE(agx_nir_opt_pack_byte_lut_select(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "clamp");
   EXPECT_EQ(count_alu_in_loops(b.shader, nir_op_ubitfield_extract), 1u);

   const uint8_t bytes[16] = {
      0x9e, 0x01, 0xff, 0x40, 0x7f, 0x80, 0x23, 0xc5,
      0x00, 0x5a, 0xa5, 0x11, 0xee, 0x37, 0x6b, 0xd2,
   };
   nir_scalar val = nir_scalar_resolved(nir_def_as_alu(sum_def)->src[1].src.ssa, 0);
   for (unsigned k = 0; k < 24; ++k)
      EXPECT_EQ(eval(val, bytes, ring, i_def, k), bytes[k < 15 ? k : 15])
         << "position " << k;
   ralloc_free(b.shader);
}

} // namespace

TEST(AGX3ByteLutExtract, ClampedIndexReadsEveryByte)
{
   check_clamped_index();
}

TEST(AGX3ByteLutExtract, OneExtractAndOneSubtractPerByte)
{
   for (const target &t : targets) {
      const result r = compile_arc_loop(t);
      printf("%s: %u instrs; in loop: %u ubitfield_extract, %u ushr, "
             "%u ineg, %u isub\n",
             t.name, r.instrs, r.ubfe, r.ushr, r.ineg, r.isub);
      ASSERT_TRUE(r.ok) << t.name;

      /* One packed LUT read per iteration, as one bitfield extract. */
      EXPECT_EQ(r.ubfe, 1u) << t.name;
      EXPECT_EQ(r.ushr, 0u) << t.name;

      /* centre - byte stays a subtract. */
      EXPECT_EQ(r.ineg, 0u) << t.name;
      EXPECT_GE(r.isub, 1u) << t.name;
   }
}

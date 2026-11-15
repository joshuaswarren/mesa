/*
 * Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}
#include "compiler/nir/nir_constant_expressions.h"

/* The hardware values a shader sees for one invocation. */
struct hw_thread {
   unsigned gid[3], wg[3], lid[3], index;
};

static uint64_t
eval(nir_scalar s, const hw_thread &t)
{
   s = nir_scalar_chase_movs(s);
   if (nir_scalar_is_const(s))
      return nir_scalar_as_uint(s);

   nir_instr *instr = nir_def_instr(s.def);
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      switch (intr->intrinsic) {
      case nir_intrinsic_load_global_invocation_id:
         return t.gid[s.comp];
      case nir_intrinsic_load_workgroup_id:
         return t.wg[s.comp];
      case nir_intrinsic_load_local_invocation_id:
         return t.lid[s.comp];
      case nir_intrinsic_load_local_invocation_index:
         return t.index;
      default:
         ADD_FAILURE() << "unexpected intrinsic "
                       << nir_intrinsic_infos[intr->intrinsic].name;
         return 0;
      }
   }

   if (!nir_scalar_is_alu(s)) {
      ADD_FAILURE() << "unexpected instruction";
      return 0;
   }

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   const nir_op_info *info = &nir_op_infos[alu->op];
   nir_const_value src[NIR_ALU_MAX_INPUTS] = {};
   nir_const_value *ptr[NIR_ALU_MAX_INPUTS] = {};
   for (unsigned i = 0; i < info->num_inputs; ++i) {
      nir_scalar c = nir_scalar_chase_alu_src(s, i);
      src[i] = nir_const_value_for_uint(eval(c, t), c.def->bit_size);
      ptr[i] = &src[i];
   }
   nir_const_value r = {};
   nir_eval_const_opcode(alu->op, &r, nullptr, 1, s.def->bit_size, ptr, 0);
   return nir_const_value_as_uint(r, s.def->bit_size);
}

class WorkgroupSplit : public testing::Test {
 protected:
   WorkgroupSplit()
   {
      b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                         &agx_nir_options, "split");
   }

   ~WorkgroupSplit() { ralloc_free(b.shader); }

   void size(unsigned x, unsigned y, unsigned z)
   {
      b.shader->info.workgroup_size[0] = x;
      b.shader->info.workgroup_size[1] = y;
      b.shader->info.workgroup_size[2] = z;
   }

   /* A use of the value that the pass rewrites in place (nothing runs DCE). */
   nir_def *keep(nir_def *def) { return nir_mov(&b, def); }

   nir_builder b;
};

TEST_F(WorkgroupSplit, BuiltinsMatchForBothLaunchShapes)
{
   struct { unsigned api[3], hw[3]; } cases[] = {
      {{8, 8, 8}, {8, 8, 4}},
      {{32, 32, 1}, {32, 8, 1}},
      {{64, 8, 1}, {64, 4, 1}},
      {{4, 8, 24}, {4, 8, 6}},
   };

   for (auto &c : cases) {
      ralloc_free(b.shader);
      b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                         &agx_nir_options, "split");
      size(c.api[0], c.api[1], c.api[2]);
      nir_def *wg = keep(nir_load_workgroup_id(&b));
      nir_def *lid = keep(nir_load_local_invocation_id(&b));
      nir_def *index = keep(nir_load_local_invocation_index(&b));
      nir_def *wsz = keep(nir_load_workgroup_size(&b));

      uint16_t hw[3];
      ASSERT_TRUE(agx_workgroup_split_size(b.shader, hw));
      for (unsigned i = 0; i < 3; ++i)
         ASSERT_EQ(hw[i], c.hw[i]);
      ASSERT_TRUE(agx_nir_lower_workgroup_split(b.shader, hw));
      nir_validate_shader(b.shader, "after workgroup split");

      const unsigned groups[3] = {3, 2, 3};
      for (const unsigned *shape : {c.hw, c.api}) {
         for (unsigned z = 0; z < groups[2] * c.api[2]; ++z) {
            for (unsigned y = 0; y < groups[1] * c.api[1]; ++y) {
               for (unsigned x = 0; x < groups[0] * c.api[0]; ++x) {
                  hw_thread t;
                  const unsigned g[3] = {x, y, z};
                  for (unsigned i = 0; i < 3; ++i) {
                     t.gid[i] = g[i];
                     t.wg[i] = g[i] / shape[i];
                     t.lid[i] = g[i] % shape[i];
                  }
                  t.index = t.lid[0] + shape[0] * (t.lid[1] + shape[1] * t.lid[2]);

                  unsigned api_lid[3];
                  for (unsigned i = 0; i < 3; ++i) {
                     api_lid[i] = g[i] % c.api[i];
                     ASSERT_EQ(eval(nir_get_scalar(wg, i), t), g[i] / c.api[i]);
                     ASSERT_EQ(eval(nir_get_scalar(lid, i), t), api_lid[i]);
                     ASSERT_EQ(eval(nir_get_scalar(wsz, i), t), c.api[i]);
                  }
                  ASSERT_EQ(eval(nir_get_scalar(index, 0), t),
                            api_lid[0] + c.api[0] * (api_lid[1] + c.api[1] * api_lid[2]));
               }
            }
         }
      }
   }
}

TEST_F(WorkgroupSplit, HardwareGroupsKeepWholeSubgroups)
{
   /* Splitting the outermost axis keeps each hardware group a contiguous run
    * of API local indices, a multiple of the 32-wide subgroup. */
   size(8, 8, 8);
   uint16_t hw[3];
   ASSERT_TRUE(agx_workgroup_split_size(b.shader, hw));
   EXPECT_EQ(hw[0], 8);
   EXPECT_EQ(hw[1], 8);
   EXPECT_EQ(hw[2] * 8 * 8 % 32, 0);
}

TEST_F(WorkgroupSplit, Refusals)
{
   uint16_t hw[3];

   size(16, 16, 1);
   EXPECT_FALSE(agx_workgroup_split_size(b.shader, hw)) << "already small";

   size(5, 5, 32);
   EXPECT_FALSE(agx_workgroup_split_size(b.shader, hw)) << "no 32-multiple split";

   size(1024, 1, 1);
   EXPECT_FALSE(agx_workgroup_split_size(b.shader, hw)) << "x-only shape";

   size(8, 8, 8);
   b.shader->info.shared_size = 4;
   EXPECT_FALSE(agx_workgroup_split_size(b.shader, hw)) << "shared memory";
   b.shader->info.shared_size = 0;

   b.shader->info.workgroup_size_variable = true;
   EXPECT_FALSE(agx_workgroup_split_size(b.shader, hw)) << "variable size";
   b.shader->info.workgroup_size_variable = false;

   ASSERT_TRUE(agx_workgroup_split_size(b.shader, hw));
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP,
               .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
   EXPECT_FALSE(agx_workgroup_split_size(b.shader, hw)) << "workgroup barrier";
}

TEST_F(WorkgroupSplit, DeviceScopeBarrierAndSubgroupIdRefusal)
{
   uint16_t hw[3];
   size(8, 8, 8);
   nir_barrier(&b, .execution_scope = SCOPE_SUBGROUP,
               .memory_scope = SCOPE_DEVICE,
               .memory_semantics = NIR_MEMORY_ACQ_REL,
               .memory_modes = nir_var_mem_global);
   EXPECT_TRUE(agx_workgroup_split_size(b.shader, hw));

   keep(nir_load_subgroup_id(&b));
   EXPECT_FALSE(agx_workgroup_split_size(b.shader, hw));
}

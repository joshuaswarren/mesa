/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "asahi/lib/agx_chip_caps.h"
#include "hk_nir_root_size.h"
#include "nir_builder.h"
#undef NDEBUG
#include <assert.h>
#include <stdio.h>

static nir_builder
builder(void)
{
   return nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL, "root-size");
}

static void
load(nir_builder *b, nir_def *addr, unsigned components, unsigned bits)
{
   nir_load_global_constant(b, components, bits, addr, .align_mul = bits / 8);
}

int main(void)
{
   const unsigned full = 8192;
   nir_builder b = builder();
   assert(hk_nir_root_size(b.shader, full) == 0);
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 832), 4, 32);
   assert(hk_nir_root_size(b.shader, full) == 848);
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 40), 1, 64);
   assert(hk_nir_root_size(b.shader, full) == 848);
   ralloc_free(b.shader);

   b = builder();
   nir_def *addr = nir_iadd_imm(&b, nir_load_root_agx(&b), 32);
   load(&b, nir_iadd_imm(&b, addr, 64), 2, 32);
   assert(hk_nir_root_size(b.shader, full) == 104);
   ralloc_free(b.shader);

   b = builder();
   addr = nir_iadd(&b, nir_load_root_agx(&b), nir_u2u64(&b, nir_load_local_invocation_index(&b)));
   load(&b, addr, 1, 32);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);

   b = builder();
   nir_store_global(&b, nir_imm_int(&b, 0), nir_load_root_agx(&b), .align_mul = 4, .write_mask = 1);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);

   for (int offset = -1; offset <= (int)full; offset += full + 1) {
      b = builder();
      load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), offset), 4, 32);
      assert(hk_nir_root_size(b.shader, full) == full);
      ralloc_free(b.shader);
   }
   b = builder();
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), full - 4), 4, 32);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);

   b = builder();
   addr = nir_load_root_agx(&b);
   for (unsigned i = 0; i < 10; ++i)
      addr = nir_iadd_imm(&b, addr, 8);
   load(&b, addr, 1, 32);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);
   b = builder();
   nir_def *reflected = nir_load_global_constant(&b, 1, 64,
      nir_load_root_agx(&b), .align_mul = 8);
   load(&b, nir_iadd_imm(&b, reflected, 4096), 1, 32);
   assert(hk_nir_root_size(b.shader, full) == full);
   ralloc_free(b.shader);

   /* Test hk_shader_root_descriptor_size(): compute trims to the highest
    * read / set-base prefix, whereas graphics VS/FS are floored at push_offset
    * so driver-synthesized prolog/epilog loads in the draw arm stay covered. */
   const unsigned sets_off = 32;
   const unsigned push_off = 512;
   const mesa_shader_stage gfx_stages[] = {
      MESA_SHADER_VERTEX,
      MESA_SHADER_FRAGMENT,
   };

   /* Compute with 1 set and no root loads stays below push_off. */
   b = builder();
   assert(hk_shader_root_descriptor_size(b.shader, MESA_SHADER_COMPUTE, 1,
                                         sets_off, push_off, full) == 48);
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 60), 1, 32);
   assert(hk_shader_root_descriptor_size(b.shader, MESA_SHADER_COMPUTE, 1,
                                         sets_off, push_off, full) == 64);
   ralloc_free(b.shader);

   for (unsigned i = 0; i < ARRAY_SIZE(gfx_stages); ++i) {
      /* Empty shader / low root load is floored at push_offset (aligned to 16). */
      b = nir_builder_init_simple_shader(gfx_stages[i], NULL, "gfx-root-floor");
      assert(hk_shader_root_descriptor_size(b.shader, gfx_stages[i], 2,
                                            sets_off, push_off, full) ==
             push_off);
      load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 128), 4, 32);
      assert(hk_shader_root_descriptor_size(b.shader, gfx_stages[i], 2,
                                            sets_off, push_off, full) ==
             push_off);
      /* Constant push-constant read past push_off extends and aligns to 16B. */
      load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), push_off + 20), 1, 32);
      assert(hk_shader_root_descriptor_size(b.shader, gfx_stages[i], 2,
                                            sets_off, push_off, full) ==
             push_off + 32);
      ralloc_free(b.shader);

      /* Dynamic indexing stays full-sized. */
      b = nir_builder_init_simple_shader(gfx_stages[i], NULL, "gfx-root-dyn");
      addr = nir_iadd(&b, nir_load_root_agx(&b),
                      nir_u2u64(&b, nir_load_vertex_id(&b)));
      load(&b, addr, 1, 32);
      assert(hk_shader_root_descriptor_size(b.shader, gfx_stages[i], 2,
                                            sets_off, push_off, full) == full);
      ralloc_free(b.shader);
   }

   /* Test hk_gfx_root_size_from_stages() (VS/FS max, VS-only, and fallbacks). */
   assert(hk_gfx_root_size_from_stages(512, true, 576, full) == 576);
   assert(hk_gfx_root_size_from_stages(640, true, 512, full) == 640);
   assert(hk_gfx_root_size_from_stages(512, false, 0, full) == 512);
   assert(hk_gfx_root_size_from_stages(0, true, 512, full) == full);
   assert(hk_gfx_root_size_from_stages(512, true, 0, full) == full);
   assert(hk_gfx_root_size_from_stages(full + 64, false, 0, full) == full);

   /* Test hk_root_upload_needs_growth() re-upload trigger. */
   assert(hk_root_upload_needs_growth(0, 512));
   assert(hk_root_upload_needs_growth(512, 576));
   assert(!hk_root_upload_needs_growth(576, 576));
   assert(!hk_root_upload_needs_growth(640, 512));

   /* Range-read proof for cs.group_count_addr check. */
   b = builder();
   assert(!hk_nir_root_reads_range(b.shader, 80, 88, full));
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 72), 1, 64); /* [72, 80) */
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 88), 3, 32); /* [88, 100) */
   assert(!hk_nir_root_reads_range(b.shader, 80, 88, full));
   /* Chained constant offset missing [80, 88). */
   addr = nir_iadd_imm(&b, nir_load_root_agx(&b), 64);
   load(&b, nir_iadd_imm(&b, addr, 32), 4, 32);
   assert(!hk_nir_root_reads_range(b.shader, 80, 88, full));
   /* Direct hit on [80, 88). */
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 80), 1, 64);
   assert(hk_nir_root_reads_range(b.shader, 80, 88, full));
   ralloc_free(b.shader);

   /* Vectorized load spanning into [80, 88) from before it (76..92). */
   b = builder();
   load(&b, nir_iadd_imm(&b, nir_load_root_agx(&b), 76), 4, 32);
   assert(hk_nir_root_reads_range(b.shader, 80, 88, full));
   ralloc_free(b.shader);

   /* Swizzled vector load_const offset (component 1 = 80, component 0 = 40):
    * root_extent and root_may_read_range must read the selected swizzle lane,
    * not component 0. */
   b = builder();
   nir_const_value vec_off[2] = {
      nir_const_value_for_uint(40, 64),
      nir_const_value_for_uint(80, 64),
   };
   nir_def *vec_const = nir_build_imm(&b, 2, 64, vec_off);
   nir_alu_instr *swiz_add = nir_alu_instr_create(b.shader, nir_op_iadd);
   nir_def_init(&swiz_add->instr, &swiz_add->def, 1, 64);
   swiz_add->src[0].src = nir_src_for_ssa(nir_load_root_agx(&b));
   swiz_add->src[1].src = nir_src_for_ssa(vec_const);
   swiz_add->src[1].swizzle[0] = 1;
   nir_builder_instr_insert(&b, &swiz_add->instr);
   load(&b, &swiz_add->def, 1, 64);
   assert(hk_nir_root_size(b.shader, full) == 88);
   assert(hk_nir_root_reads_range(b.shader, 80, 88, full));
   assert(!hk_nir_root_reads_range(b.shader, 40, 48, full));
   ralloc_free(b.shader);

   /* Out-of-range swizzle on vector load_const must fail closed. */
   b = builder();
   vec_const = nir_build_imm(&b, 2, 64, vec_off);
   swiz_add = nir_alu_instr_create(b.shader, nir_op_iadd);
   nir_def_init(&swiz_add->instr, &swiz_add->def, 1, 64);
   swiz_add->src[0].src = nir_src_for_ssa(nir_load_root_agx(&b));
   swiz_add->src[1].src = nir_src_for_ssa(vec_const);
   swiz_add->src[1].swizzle[0] = 2;
   nir_builder_instr_insert(&b, &swiz_add->instr);
   load(&b, &swiz_add->def, 1, 64);
   assert(hk_nir_root_size(b.shader, full) == full);
   assert(hk_nir_root_reads_range(b.shader, 80, 88, full));
   ralloc_free(b.shader);

   /* Sub-byte (1-bit) load rounds up to 1 byte instead of 0. */
   b = builder();
   nir_load_global_constant(&b, 1, 1,
                            nir_iadd_imm(&b, nir_load_root_agx(&b), 80),
                            .align_mul = 1);
   assert(hk_nir_root_size(b.shader, full) == 81);
   assert(hk_nir_root_reads_range(b.shader, 80, 88, full));
   ralloc_free(b.shader);

   /* Dynamic offset must fail closed (returns true). */
   b = builder();
   addr = nir_iadd(&b, nir_load_root_agx(&b), nir_u2u64(&b, nir_load_local_invocation_index(&b)));
   load(&b, addr, 1, 32);
   assert(hk_nir_root_reads_range(b.shader, 80, 88, full));
   ralloc_free(b.shader);

   /* Escaping store use and reflected root pointer must fail closed. */
   b = builder();
   nir_store_global(&b, nir_imm_int(&b, 0), nir_load_root_agx(&b), .align_mul = 4, .write_mask = 1);
   assert(hk_nir_root_reads_range(b.shader, 80, 88, full));
   ralloc_free(b.shader);

   b = builder();
   reflected = nir_load_global_constant(&b, 1, 64,
      nir_load_root_agx(&b), .align_mul = 8);
   load(&b, nir_iadd_imm(&b, reflected, 4096), 1, 32);
   assert(hk_nir_root_reads_range(b.shader, 80, 88, full));
   ralloc_free(b.shader);

   /* Workgroup coalescing: compile-time NIR proof checks. */
   {
      /* Pure global_invocation_id shader with wg <= 256 (multiple of 32). */
      const uint16_t good_wgs[][3] = {
         {32, 1, 1}, {64, 1, 1}, {32, 2, 1}, {96, 1, 1}, {128, 1, 1}, {64, 2, 1},
         {256, 1, 1}, {16, 16, 1},
      };
      for (unsigned i = 0; i < ARRAY_SIZE(good_wgs); ++i) {
         b = builder();
         b.shader->info.internal = false;
         b.shader->info.workgroup_size[0] = good_wgs[i][0];
         b.shader->info.workgroup_size[1] = good_wgs[i][1];
         b.shader->info.workgroup_size[2] = good_wgs[i][2];
         nir_def *gid = nir_load_global_invocation_id(&b, 32);
         BITSET_SET(b.shader->info.system_values_read, SYSTEM_VALUE_GLOBAL_INVOCATION_ID);
         nir_store_global(&b, nir_channel(&b, gid, 0), nir_imm_int64(&b, 0x1000),
                          .align_mul = 4, .write_mask = 1);
         assert(hk_nir_can_coalesce_workgroup(b.shader));
         ralloc_free(b.shader);
      }

      /* Accept <= 1024 multiples of 32 (512/1024 qualify for small-grid split
       * while coalescing itself still caps at <= 256), and reject non-multiples
       * of 32 or > 1024 / 0. */
      const uint16_t split_wgs[][3] = {
         {320, 1, 1}, {512, 1, 1}, {1024, 1, 1},
      };
      for (unsigned i = 0; i < ARRAY_SIZE(split_wgs); ++i) {
         b = builder();
         b.shader->info.internal = false;
         b.shader->info.workgroup_size[0] = split_wgs[i][0];
         b.shader->info.workgroup_size[1] = split_wgs[i][1];
         b.shader->info.workgroup_size[2] = split_wgs[i][2];
         assert(hk_nir_can_coalesce_workgroup(b.shader));
         ralloc_free(b.shader);
      }
      const uint16_t bad_wgs[][3] = {
         {16, 1, 1}, {48, 1, 1}, {80, 1, 1}, {1056, 1, 1}, {0, 1, 1},
      };
      for (unsigned i = 0; i < ARRAY_SIZE(bad_wgs); ++i) {
         b = builder();
         b.shader->info.internal = false;
         b.shader->info.workgroup_size[0] = bad_wgs[i][0];
         b.shader->info.workgroup_size[1] = bad_wgs[i][1];
         b.shader->info.workgroup_size[2] = bad_wgs[i][2];
         assert(!hk_nir_can_coalesce_workgroup(b.shader));
         ralloc_free(b.shader);
      }

      /* Reject shared_size, variable wg, internal, ray_queries,
       * derivative_group, has_cooperative_matrix. */
      for (unsigned mode = 0; mode < 6; ++mode) {
         b = builder();
         b.shader->info.internal = false;
         b.shader->info.workgroup_size[0] = 32;
         b.shader->info.workgroup_size[1] = 1;
         b.shader->info.workgroup_size[2] = 1;
         if (mode == 0) b.shader->info.shared_size = 16;
         else if (mode == 1) b.shader->info.workgroup_size_variable = true;
         else if (mode == 2) b.shader->info.internal = true;
         else if (mode == 3) b.shader->info.ray_queries = true;
         else if (mode == 4) b.shader->info.derivative_group = DERIVATIVE_GROUP_LINEAR;
         else b.shader->info.cs.has_cooperative_matrix = true;
         assert(!hk_nir_can_coalesce_workgroup(b.shader));
         ralloc_free(b.shader);
      }

      /* Accept load_num_workgroups (API group_count_addr on direct dispatches)
       * and load_workgroup_size (constant-folded API local size). */
      {
         b = builder();
         b.shader->info.internal = false;
         b.shader->info.workgroup_size[0] = 64;
         b.shader->info.workgroup_size[1] = 1;
         b.shader->info.workgroup_size[2] = 1;
         nir_load_num_workgroups(&b);
         nir_load_workgroup_size(&b);
         assert(hk_nir_can_coalesce_workgroup(b.shader));
         ralloc_free(b.shader);
      }

      /* Reject barrier, local_id, wg_id, ballot, inverse_ballot,
       * shuffle_up_intel, read_invocation, elect, shared, scs_thread_index_agx,
       * enter_query_agx. */
      for (unsigned mode = 0; mode < 11; ++mode) {
         b = builder();
         b.shader->info.internal = false;
         b.shader->info.workgroup_size[0] = 32;
         b.shader->info.workgroup_size[1] = 1;
         b.shader->info.workgroup_size[2] = 1;
         if (mode == 0)
            nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP);
         else if (mode == 1)
            nir_load_local_invocation_id(&b);
         else if (mode == 2)
            nir_load_workgroup_id(&b);
         else if (mode == 3)
            nir_ballot(&b, 1, 32, nir_imm_true(&b));
         else if (mode == 4)
            nir_inverse_ballot(&b, nir_imm_int(&b, 1));
         else if (mode == 5)
            nir_shuffle_up_intel(&b, nir_imm_int(&b, 1), nir_imm_int(&b, 0),
                                 nir_imm_int(&b, 1));
         else if (mode == 6)
            nir_read_invocation(&b, nir_imm_int(&b, 1), nir_imm_int(&b, 0));
         else if (mode == 7)
            nir_elect(&b, 1);
         else if (mode == 8)
            nir_load_shared(&b, 1, 32, nir_imm_int(&b, 0), .align_mul = 4);
         else if (mode == 9)
            nir_load_scs_thread_index_agx(&b);
         else
            nir_enter_query_agx(&b, .base = 0);
         assert(!hk_nir_can_coalesce_workgroup(b.shader));
         ralloc_free(b.shader);
      }
   }

   /* Workgroup coalescing: dispatch-time exact divisibility & occupancy checks. */
   {
      uint16_t out[3];
      /* 1D 118M threads (3,695,120 * 32) -> (256, 1, 1). */
      const uint16_t wg32[3] = {32, 1, 1};
      const uint32_t g118m[3] = {3695120u * 32u, 1, 1};
      assert(hk_coalesce_workgroup_size(wg32, g118m, 1024, out) &&
             out[0] == 256 && out[1] == 1 && out[2] == 1);

      /* 2D 1920x1080 -> (128, 2, 1) = 256 threads, exact divisor of both axes. */
      const uint32_t g1080p[3] = {1920, 1080, 1};
      assert(hk_coalesce_workgroup_size(wg32, g1080p, 1024, out) &&
             out[0] == 128 && out[1] == 2 && out[2] == 1);

      /* 2D 6016x4000 -> (128, 2, 1). */
      const uint32_t g24m[3] = {6016, 4000, 1};
      assert(hk_coalesce_workgroup_size(wg32, g24m, 1024, out) &&
             out[0] == 128 && out[1] == 2 && out[2] == 1);

      /* 2D 96x1000 (X odd multiple of 32, Y divisible by 8) -> (32, 8, 1). */
      const uint32_t g96k[3] = {96, 1000, 1};
      assert(hk_coalesce_workgroup_size(wg32, g96k, 1024, out) &&
             out[0] == 32 && out[1] == 8 && out[2] == 1);

      /* wg=(64, 1, 1) -> (256, 1, 1). */
      const uint16_t wg64[3] = {64, 1, 1};
      const uint32_t g23m[3] = {739024u * 32u, 1, 1};
      assert(hk_coalesce_workgroup_size(wg64, g23m, 1024, out) &&
             out[0] == 256 && out[1] == 1 && out[2] == 1);

      /* Small dispatch (< 256 workgroups at target size) stays uncoalesced. */
      const uint32_t gsmall[3] = {32 * 100, 1, 1};
      assert(!hk_coalesce_workgroup_size(wg32, gsmall, 1024, out) &&
             out[0] == 32 && out[1] == 1 && out[2] == 1);

      /* Large prime group counts coalesce via bounded partial-edge steps
       * (<= 6.25% inactive edge lanes), while small prime grids whose edge
       * padding would exceed 6.25% stay uncoalesced. */
      const uint32_t gprime[3] = {32 * 997, 991, 1};
      assert(hk_coalesce_workgroup_size(wg32, gprime, 1024, out) &&
             (uint64_t)out[0] * out[1] * out[2] == 256u && out[2] == 1);
      const uint32_t gprime_small[3] = {32 * 13, 13, 1};
      assert(!hk_coalesce_workgroup_size(wg32, gprime_small, 1024, out) &&
             out[0] == 32 && out[1] == 1 && out[2] == 1);

      /* Real odd-tiled 2D grids (1512x984 at 8x8 -> 16x16, 760x496 at 8x8 ->
       * 16x16 after Pass 1 takes Y to 16 and Pass 2 takes X to 16). */
      const uint16_t wg8x8[3] = {8, 8, 1};
      const uint32_t g_odd_2d[3] = {1512, 984, 1};
      assert(hk_coalesce_workgroup_size(wg8x8, g_odd_2d, 1024, out) &&
             out[0] == 16 && out[1] == 16 && out[2] == 1);
      const uint32_t g_half_odd[3] = {760, 496, 1};
      assert(hk_coalesce_workgroup_size(wg8x8, g_half_odd, 1024, out) &&
             out[0] == 16 && out[1] == 16 && out[2] == 1);
      /* Asymmetric odd grid: X has only 13 groups (doubling wastes 1/13 = 7.7%
       * > 6.25%), while Y has 301 groups (doubling to 8 wastes 3/301 = 1.0% <=
       * 6.25% with 13 * 38 = 494 >= 256 WGs) -> coalesces along Y only. */
      const uint32_t g_asym_odd[3] = {32 * 13, 301, 1};
      assert(hk_coalesce_workgroup_size(wg32, g_asym_odd, 1024, out) &&
             out[0] == 32 && out[1] == 8 && out[2] == 1);

      /* max_wg_threads cap (128) respected. */
      assert(hk_coalesce_workgroup_size(wg32, g118m, 128, out) &&
             out[0] == 128 && out[1] == 1 && out[2] == 1);

      /* Non-divisible partial grid fails closed. */
      const uint32_t gpartial[3] = {32 * 1000 + 1, 1, 1};
      assert(!hk_coalesce_workgroup_size(wg32, gpartial, 1024, out));

      /* wg=(128, 1, 1) coalesces to (256, 1, 1) on 1D and (128, 2, 1) on 2D. */
      const uint16_t wg128[3] = {128, 1, 1};
      const uint32_t g23m_512[3] = {46200u * 512u, 1, 1};
      assert(hk_coalesce_workgroup_size(wg128, g23m_512, 1024, out) &&
             out[0] == 256 && out[1] == 1 && out[2] == 1);
      assert(hk_coalesce_workgroup_size(wg128, g24m, 1024, out) &&
             out[0] == 128 && out[1] == 2 && out[2] == 1);

      /* Small wg=128 grid (< 256 WGs at 256) stays uncoalesced. */
      const uint32_t g128_small[3] = {36864, 1, 1};
      assert(!hk_coalesce_workgroup_size(wg128, g128_small, 1024, out) &&
             out[0] == 128 && out[1] == 1 && out[2] == 1);

      /* Short-shader 512 target: wg=256 and wg=128 step up to 512 when
       * >= 1024 WGs remain, and cap at 256 when between 256 and 1023 WGs. */
      const uint32_t axis_z64[3] = {1024u, 1024u, 64u};
      const uint32_t axis_z1024[3] = {1024u, 1024u, 1024u};
      const uint16_t wg256[3] = {256, 1, 1};
      assert(!hk_coalesce_workgroup_size(wg256, g23m_512, 1024, out) &&
             out[0] == 256 && out[1] == 1 && out[2] == 1);
      assert(hk_coalesce_workgroup_size_ex(wg256, g23m_512, axis_z64, 1024,
                                           512, 256, out) &&
             out[0] == 512 && out[1] == 1 && out[2] == 1);
      assert(hk_coalesce_workgroup_size_ex(wg128, g24m, axis_z64, 1024, 512,
                                           256, out) &&
             out[0] == 128 && out[1] == 4 && out[2] == 1);
      /* 512 * 512 = 262,144 threads: 1024 WGs at 256, only 512 WGs at 512 ->
       * stops at 256 instead of stepping to 512. */
      const uint32_t g_mid[3] = {512u * 512u, 1, 1};
      assert(hk_coalesce_workgroup_size_ex(wg32, g_mid, axis_z64, 1024, 512,
                                           256, out) &&
             out[0] == 256 && out[1] == 1 && out[2] == 1);
      assert(!hk_coalesce_workgroup_size_ex(wg256, g_mid, axis_z64, 1024, 512,
                                            256, out) &&
             out[0] == 256 && out[1] == 1 && out[2] == 1);

      /* Hard ceiling at 512: requesting 1024 never exceeds 512. */
      assert(hk_coalesce_workgroup_size_ex(wg256, g23m_512, axis_z64, 1024,
                                           1024, 256, out) &&
             out[0] == 512 && out[1] == 1 && out[2] == 1);
      /* Register-file max_wg_threads=256 (or 384/448 from preamble GPR cap)
       * blocks stepping to 512. */
      assert(!hk_coalesce_workgroup_size_ex(wg256, g23m_512, axis_z64, 256,
                                            512, 256, out) &&
             out[0] == 256 && out[1] == 1 && out[2] == 1);
      assert(!hk_coalesce_workgroup_size_ex(wg256, g23m_512, axis_z64, 448,
                                            512, 256, out) &&
             out[0] == 256 && out[1] == 1 && out[2] == 1);

      /* 3D overflow fails closed. */
      const uint32_t g_ovf[3] = {0x80000000u, 0x80000000u, 16u};
      assert(!hk_coalesce_workgroup_size_ex(wg32, g_ovf, axis_z64, 1024, 512,
                                            256, out));

      /* Non-zero dispatch base_group must be rejected. */
      const uint32_t base_zero[3] = {0, 0, 0};
      const uint32_t base_x[3] = {1, 0, 0};
      const uint32_t base_y[3] = {0, 2, 0};
      const uint32_t base_z[3] = {0, 0, 3};
      assert(hk_nir_dispatch_zero_base(base_zero));
      assert(!hk_nir_dispatch_zero_base(base_x));
      assert(!hk_nir_dispatch_zero_base(base_y));
      assert(!hk_nir_dispatch_zero_base(base_z));

      /* Per-axis limits from max_wg_size[]: Z caps at 64 under {1024, 1024, 64}
       * and can grow further when the caller provides a higher Z limit. */
      const uint16_t wg_z32[3] = {1, 1, 32};
      const uint32_t g_zonly[3] = {1, 1, 32u * 4096u};
      assert(hk_coalesce_workgroup_size(wg_z32, g_zonly, 1024, out) &&
             out[0] == 1 && out[1] == 1 && out[2] == 64);
      assert(hk_coalesce_workgroup_size_ex(wg_z32, g_zonly, axis_z1024, 1024,
                                           256, 256, out) &&
             out[0] == 1 && out[1] == 1 && out[2] == 256);

      /* An API workgroup whose Z already exceeds max_wg_size[2] never grows Z
       * further, though X/Y may still coalesce. */
      const uint16_t wg_z128[3] = {1, 1, 128};
      const uint32_t g_z128_only[3] = {1, 1, 128u * 4096u};
      assert(!hk_coalesce_workgroup_size(wg_z128, g_z128_only, 1024, out) &&
             out[0] == 1 && out[1] == 1 && out[2] == 128);
      const uint32_t g_z128_xz[3] = {2, 1, 128u * 4096u};
      assert(hk_coalesce_workgroup_size(wg_z128, g_z128_xz, 1024, out) &&
             out[0] == 2 && out[1] == 1 && out[2] == 128);

      /* Zero or NULL per-axis limits fail closed. */
      const uint32_t axis_bad[3] = {1024u, 1024u, 0u};
      assert(!hk_coalesce_workgroup_size_ex(wg32, g118m, NULL, 1024, 256, 256,
                                            out));
      assert(!hk_coalesce_workgroup_size_ex(wg32, g118m, axis_bad, 1024, 256,
                                            256, out));

      /* Small-grid workgroup splitting for wide multi-core GPUs (e.g. 40 cores,
       * min_workgroups = 80). */
      const uint32_t g4096[3] = {4096, 1, 1};
      const uint32_t g4032[3] = {4032, 1, 1};
      const uint16_t wg512[3] = {512, 1, 1};
      const uint16_t wg1024[3] = {1024, 1, 1};
      assert(hk_split_workgroup_size_ex(wg128, g4096, 80, out) &&
             out[0] == 32 && out[1] == 1 && out[2] == 1);
      assert(hk_split_workgroup_size_ex(wg256, g4096, 80, out) &&
             out[0] == 32 && out[1] == 1 && out[2] == 1);
      assert(hk_split_workgroup_size_ex(wg512, g4096, 80, out) &&
             out[0] == 32 && out[1] == 1 && out[2] == 1);
      assert(hk_split_workgroup_size_ex(wg1024, g4096, 80, out) &&
             out[0] == 32 && out[1] == 1 && out[2] == 1);
      assert(hk_split_workgroup_size_ex(wg64, g4032, 80, out) &&
             out[0] == 32 && out[1] == 1 && out[2] == 1);
      const uint32_t g6400[3] = {6400, 1, 1};
      assert(!hk_split_workgroup_size_ex(wg64, g6400, 80, out) &&
             out[0] == 64 && out[1] == 1 && out[2] == 1);
      assert(!hk_split_workgroup_size_ex(wg32, g4096, 200, out) &&
             out[0] == 32 && out[1] == 1 && out[2] == 1);
      const uint16_t wg32x4[3] = {32, 4, 1};
      const uint32_t g2d_small[3] = {320, 20, 1}; /* 50 WGs < 80 -> 100 WGs */
      assert(hk_split_workgroup_size_ex(wg32x4, g2d_small, 80, out) &&
             out[0] == 32 && out[1] == 2 && out[2] == 1);
      const uint16_t wg32x3[3] = {32, 3, 1};
      const uint32_t g2d_odd[3] = {320, 15, 1};
      assert(!hk_split_workgroup_size_ex(wg32x3, g2d_odd, 80, out) &&
             out[0] == 32 && out[1] == 3 && out[2] == 1);

      /* Capability table: enabled on G17P and M5 (G17G, G17S, G17C); other
       * parts stay off until per-generation same-machine A/B + CTS rows
       * qualify them. */
      for (int chip = AGX_CHIP_G13G; chip <= AGX_CHIP_UNKNOWN; ++chip) {
         const enum agx_chip c = (enum agx_chip)chip;
         const bool expected = c == AGX_CHIP_G17P || c == AGX_CHIP_G17G ||
                               c == AGX_CHIP_G17S || c == AGX_CHIP_G17C;
         assert(agx_chip_has_cap(c, AGX_CAP_COALESCE_PURE_GLOBAL_WORKGROUPS) ==
                expected);
      }
   }

   puts("root footprint and workgroup coalescing controls passed");
   return 0;
}

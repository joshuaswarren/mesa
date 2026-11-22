/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Every chip gets one native AS format end to end: the format the AS entry
 * points dispatch on (hk_agx3_as_format_for_chip()), the encoding scratch
 * and image layout its builds size and address, and the node encoder and
 * refit programs its pipelines are created from. GEN1 (G17P and the other
 * non-M5 chips) and M5 (G17G/G17S) must not share any of them.
 */
#undef NDEBUG
#include "hk_acceleration_structure.h"
#include "hk_agx3_as_layout.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* x of the LocalSize execution mode of a SPIR-V compute module. */
static unsigned
spirv_local_size_x(const uint32_t *code, size_t size_B)
{
   assert(size_B >= 20 && size_B % 4 == 0 && code[0] == 0x07230203);
   unsigned x = 0, modes = 0;
   for (size_t i = 5; i < size_B / 4;) {
      const unsigned words = code[i] >> 16, op = code[i] & 0xffff;
      assert(words && i + words <= size_B / 4);
      if (op == 16 /* OpExecutionMode */ && words >= 6 &&
          code[i + 2] == 17 /* LocalSize */) {
         x = code[i + 3];
         assert(code[i + 4] == 1 && code[i + 5] == 1);
         modes++;
      }
      i += words;
   }
   assert(modes == 1);
   return x;
}

static void
check_layout(enum hk_agx3_as_format format, uint32_t leaves,
             const struct hk_agx3_as_layout *l)
{
   const uint32_t align = format == HK_AGX3_AS_FORMAT_M5 ? 1024 : 16384;
   const uint32_t tail = format == HK_AGX3_AS_FORMAT_M5 ? 32 : 48;
   assert(hk_agx3_as_image_align(format) == align);
   assert(hk_agx3_as_plan_tail(format) == tail);
   if (!leaves) {
      assert(l->slot_bound == 3 && l->encode_bytes == 0);
      assert(l->as_bytes == 192 + (align == 16384 ? 0 : align - 1));
      return;
   }
   const uint64_t nodes = (uint64_t)leaves + (leaves > 1 ? leaves : 2) - 1;
   const uint64_t slots = (uint64_t)leaves * 4 + 1;
   assert(l->node_bound == nodes && l->slot_bound == slots);
   assert(l->jobs == nodes * 16 && l->envelope == l->jobs + nodes * 64);
   assert(l->state == l->envelope + 48);
   /* GEN1's plan writes the indirect encode dispatch after its state. */
   assert(l->node_status == l->state + tail);
   assert(l->envelope_status == l->node_status + nodes * 4);
   const uint64_t refit_levels = (l->envelope_status + 4 + 7) & ~7u;
   assert(l->refit_levels == refit_levels);
   assert(l->encode_bytes == refit_levels +
          (format == HK_AGX3_AS_FORMAT_GEN1 ? 16 * 12 : 0));
   assert(l->as_bytes == slots * HK_AGX3_AS_SLOT_B + 184 +
                            (align == 16384 ? 0 : align - 1));
}

int
main(void)
{
   unsigned m5 = 0, gen1 = 0;
   for (enum agx_chip chip = 0; chip < AGX_CHIP_UNKNOWN; ++chip) {
      const enum hk_agx3_as_format format = hk_agx3_as_format_for_chip(chip);
      const bool is_m5 = chip == AGX_CHIP_G17G || chip == AGX_CHIP_G17S ||
                         chip == AGX_CHIP_G17C || chip == AGX_CHIP_G17D;
      assert(format == (is_m5 ? HK_AGX3_AS_FORMAT_M5 : HK_AGX3_AS_FORMAT_GEN1));
      m5 += is_m5;
      gen1 += !is_m5;

      /* The scratch and image layout the chip's builds use. */
      static const uint32_t leaves[] = {0, 1, 2, 3, 7, 64, 1000, 1u << 20};
      for (unsigned i = 0; i < sizeof(leaves) / sizeof(leaves[0]); ++i) {
         struct hk_agx3_as_layout l, other;
         assert(hk_agx3_as_layout_init_format(format, leaves[i], &l));
         assert(is_m5 ? hk_agx3_as_layout_init(leaves[i], &other)
                      : hk_gen1_as_layout_init(leaves[i], &other));
         assert(!memcmp(&l, &other, sizeof(l)));
         check_layout(format, leaves[i], &l);
      }

      /* The node encoder: one node per 8 lanes on M5, four 8-lane jobs per
       * workgroup on GEN1. */
      size_t size_B;
      const uint32_t *encoder = hk_agx3_node_encoder_spirv(chip, &size_B);
      assert(encoder && spirv_local_size_x(encoder, size_B) ==
                           (is_m5 ? HK_AGX3_M5_NODE_GROUP_LANES
                                  : HK_AGX3_GEN1_NODE_GROUP_LANES));
      size_t own_B;
      const uint32_t *own = is_m5 ? hk_m5_agx3_node_encoder_spirv(&own_B)
                                  : hk_gen1_agx3_node_encoder_spirv(false, &own_B);
      assert(own == encoder && own_B == size_B);

      /* The refit of the same records. */
      const uint32_t *refit = hk_agx3_refit_spirv(chip, &size_B);
      own = is_m5 ? hk_m5_agx3_refit_spirv(&own_B) : hk_gen1_agx3_refit_spirv(&own_B);
      assert(refit && own == refit && own_B == size_B);
   }
   assert(m5 == 4 && gen1 == AGX_CHIP_UNKNOWN - 4);

   /* The two formats share no program, and the GEN1 triangle pair variant
    * keeps its workgroup shape. */
   size_t gen1_B, m5_B, pairs_B, gen1_refit_B, m5_refit_B;
   const uint32_t *gen1_enc = hk_gen1_agx3_node_encoder_spirv(false, &gen1_B);
   const uint32_t *m5_enc = hk_m5_agx3_node_encoder_spirv(&m5_B);
   const uint32_t *pairs = hk_gen1_agx3_node_encoder_spirv(true, &pairs_B);
   assert(gen1_enc != m5_enc && pairs != gen1_enc);
   assert(gen1_B != m5_B || memcmp(gen1_enc, m5_enc, gen1_B));
   assert(spirv_local_size_x(pairs, pairs_B) == HK_AGX3_GEN1_NODE_GROUP_LANES);
   const uint32_t *gen1_refit = hk_gen1_agx3_refit_spirv(&gen1_refit_B);
   const uint32_t *m5_refit = hk_m5_agx3_refit_spirv(&m5_refit_B);
   assert(gen1_refit != m5_refit &&
          (gen1_refit_B != m5_refit_B ||
           memcmp(gen1_refit, m5_refit, gen1_refit_B)));

   size_t refit_top_B;
   const uint32_t *refit_top = hk_gen1_agx3_refit_top_spirv(&refit_top_B);
   assert(refit_top && refit_top != gen1_refit && refit_top != m5_refit);
   assert(spirv_local_size_x(refit_top, refit_top_B) == HK_AGX3_REFIT_TOP_LANES);

   size_t refit_direct_B;
   const uint32_t *refit_direct =
      hk_gen1_agx3_refit_encode_direct_spirv(&refit_direct_B);
   assert(refit_direct && refit_direct != gen1_enc && refit_direct != m5_enc);
   assert(spirv_local_size_x(refit_direct, refit_direct_B) ==
          HK_AGX3_GEN1_NODE_GROUP_LANES);

   /* GEN1 refit scheduling policy: small/medium updates fuse every inner level
    * and envelope into one direct workgroup; larger updates fuse the top three
    * levels plus envelope and keep aligned prepare grids. */
   assert(hk_agx3_refit_fused_all(0));
   assert(hk_agx3_refit_fused_all(1));
   assert(hk_agx3_refit_fused_all(256));
   assert(hk_agx3_refit_fused_all(2048));
   assert(hk_agx3_refit_fused_all(HK_AGX3_REFIT_FUSED_MAX_LEAVES));
   assert(!hk_agx3_refit_fused_all(HK_AGX3_REFIT_FUSED_MAX_LEAVES + 1));
   assert(!hk_agx3_refit_fused_all(1u << 20));
   assert(hk_agx3_refit_top_levels(2048) == 16u);
   assert(hk_agx3_refit_top_levels(HK_AGX3_REFIT_FUSED_MAX_LEAVES) == 16u);
   assert(hk_agx3_refit_top_levels(HK_AGX3_REFIT_FUSED_MAX_LEAVES + 1) ==
          HK_AGX3_REFIT_TOP_LEVELS);
   assert(hk_agx3_refit_prepare_threads(2048, 4095, 0, false, false) ==
          HK_AGX3_REFIT_PREPARE_GROUP_SIZE);
   assert(hk_agx3_refit_prepare_threads(272, 543, 0, false, true) == 320u);
   assert(hk_agx3_refit_prepare_threads(65536, 131071, 0, false, false) ==
          HK_AGX3_REFIT_PREPARE_MAX_THREADS);
   assert(hk_agx3_refit_prepare_threads(1u << 20, (2u << 20) - 1, 1ull << 26,
                                        true, false) ==
          HK_AGX3_REFIT_COPY_MAX_THREADS);
   static const uint32_t sample_leaves[] = {0, 1, 17, 256, 2048, 8192, 65536};
   for (unsigned i = 0; i < sizeof(sample_leaves) / sizeof(sample_leaves[0]); ++i) {
      const uint32_t n = sample_leaves[i];
      for (unsigned copy = 0; copy < 2; ++copy) {
         for (unsigned inst = 0; inst < 2; ++inst) {
            const uint32_t t = hk_agx3_refit_prepare_threads(
               n, n ? 2 * n - 1 : 0, (uint64_t)n * 64, copy, inst);
            assert(t >= HK_AGX3_REFIT_PREPARE_GROUP_SIZE);
            assert(t % HK_AGX3_REFIT_PREPARE_GROUP_SIZE == 0);
         }
      }
   }

   /* Direct leaf-job refit dispatch sizing and single-geometry triangle fast
    * path predicates. */
   assert(hk_agx3_refit_direct_leaf_invocations(0) == 32u);
   assert(hk_agx3_refit_direct_leaf_invocations(1) == 32u);
   assert(hk_agx3_refit_direct_leaf_invocations(4) == 32u);
   assert(hk_agx3_refit_direct_leaf_invocations(5) == 64u);
   assert(hk_agx3_refit_direct_leaf_invocations(256) == 2048u);
   assert(hk_agx3_refit_direct_leaf_invocations(272) == 2176u);
   assert(hk_agx3_refit_direct_leaf_invocations(2048) == 16384u);
   assert(hk_agx3_refit_direct_leaf_invocations(8192) == 65536u);
   assert(hk_agx3_refit_uses_direct_triangles(true, true, 1, 2048, 2048));
   assert(hk_agx3_refit_uses_direct_triangles(true, true, 1, 8192, 8192));
   assert(!hk_agx3_refit_uses_direct_triangles(false, true, 1, 2048, 2048));
   assert(!hk_agx3_refit_uses_direct_triangles(true, false, 1, 2048, 2048));
   assert(!hk_agx3_refit_uses_direct_triangles(true, true, 2, 2048, 2048));
   assert(!hk_agx3_refit_uses_direct_triangles(true, true, 1, 1024, 2048));
   assert(!hk_agx3_refit_uses_direct_triangles(true, true, 1, 0, 0));
   assert(!hk_agx3_refit_uses_direct_triangles(true, true, 1, 8193, 8193));

   /* The layouts differ exactly where the formats do. */
   struct hk_agx3_as_layout g, m;
   assert(hk_gen1_as_layout_init(64, &g) && hk_agx3_as_layout_init(64, &m));
   assert(g.state == m.state && g.node_status == m.node_status + 16);
   assert(m.as_bytes == g.as_bytes + 1023);

   printf("hk_agx3_as_format: %u M5 and %u GEN1 chips, each with its own "
          "layout, encoder and refit\n", m5, gen1);
   return 0;
}

/* Copyright 2026 Aurora Silicon */
/* SPDX-License-Identifier: MIT */

/* Direct unit tests for G17P native compute constant-program (preamble) wiring
 * in fused submit-pack dispatches (hk_fused_compute_shader_eligible and
 * hk_emit_fused_compute_dispatch). No DRM device or GPU is used. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "asahi/compiler/agx3_cprog_input.h"
#include "asahi/lib/agx3_cdm_direct_launch_semantic_core.h"
#include "hk_agx3_ioto_program_core.h"
#include "hk_cmd_buffer.h"
#include "hk_compute_pair.h"
#include "hk_cs_root_size.h"
#include "hk_device.h"
#include "hk_physical_device.h"
#include "hk_shader.h"

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                  \
         abort();                                                              \
      }                                                                        \
   } while (0)

static void
init_base_fixture(struct hk_device *dev, struct hk_physical_device *pdev,
                  struct hk_shader *s, struct hk_linked_shader *linked,
                  struct agx_bo *shader_bo, struct agx_va *shader_va,
                  struct agx_bo *linked_bo, struct agx_va *linked_va,
                  struct hk_compute_pair_dispatch *p, uint8_t *code,
                  size_t code_size)
{
   memset(dev, 0, sizeof(*dev));
   memset(pdev, 0, sizeof(*pdev));
   memset(s, 0, sizeof(*s));
   memset(linked, 0, sizeof(*linked));
   memset(shader_bo, 0, sizeof(*shader_bo));
   memset(shader_va, 0, sizeof(*shader_va));
   memset(linked_bo, 0, sizeof(*linked_bo));
   memset(linked_va, 0, sizeof(*linked_va));
   memset(p, 0, sizeof(*p));
   memset(code, 0, code_size);

   /* Valid inline constant-program instruction sequence at offset 32. */
   static const uint8_t cprog_bytes[6] = {0x0c, 0x10, 0x0e, 0, 0, 0};
   memcpy(code + 32, cprog_bytes, sizeof(cprog_bytes));

   dev->vk.physical = &pdev->vk;
   pdev->vk.properties.maxComputeWorkGroupInvocations = 1024;
   dev->dev.chip = AGX_CHIP_G17P;
   dev->dev.shader_base = UINT64_C(0x1000000000);

   shader_va->addr = dev->dev.shader_base + 0x100000;
   shader_bo->va = shader_va;

   linked_va->addr = dev->dev.shader_base + 0x200000;
   linked_bo->va = linked_va;

   linked->b.bo = linked_bo;
   linked->b.entry_offset = 0x40;
   linked->b.size = 256;
   linked->b.nr_gprs = 24;

   s->only_linked = linked;
   s->bo = shader_bo;
   s->bo_offset = 0x80;
   s->code_ptr = code;
   s->code_size = code_size;
   s->info.stage = MESA_SHADER_COMPUTE;
   s->info.agx3_uses_root = true;
   s->info.agx3_root_uniform = 0;
   s->info.set_count = 1;

   s->b.info.stage = MESA_SHADER_COMPUTE;
   s->b.info.workgroup_size[0] = 128;
   s->b.info.workgroup_size[1] = 1;
   s->b.info.workgroup_size[2] = 1;
   s->b.info.binary_size = (uint32_t)code_size;
   s->b.info.agx3_staging_known = true;
   s->b.info.agx3_staging_halfs = 12;
   s->b.info.local_size = 4096;

   p->exec_local[0] = 128;
   p->exec_local[1] = 1;
   p->exec_local[2] = 1;
   p->exec_threads[0] = 256;
   p->exec_threads[1] = 16;
   p->exec_threads[2] = 1;
   p->root_va = UINT64_C(0x5a89abcde0);
}

static void
test_g17p_fused_cprog_eligibility(void)
{
   struct hk_device dev;
   struct hk_physical_device pdev;
   struct hk_shader s;
   struct hk_linked_shader linked;
   struct agx_bo shader_bo, linked_bo;
   struct agx_va shader_va, linked_va;
   struct hk_compute_pair_dispatch p;
   uint8_t code[512];

   init_base_fixture(&dev, &pdev, &s, &linked, &shader_bo, &shader_va,
                     &linked_bo, &linked_va, &p, code, sizeof(code));

   /* Non-preamble shader is eligible even without s->bo. */
   s.bo = NULL;
   CHECK(hk_fused_compute_shader_eligible(&dev, &s, &p));

   /* Preamble shader with valid G17P metadata requires s->bo && s->bo->va. */
   uint64_t halves[AGX3_CPROG_INPUT_WORDS] = {0};
   halves[0] = 0x3full; /* u0..u5 (within root range u0..u7) */
   CHECK(agx3_cprog_inputs_init(halves, &s.b.info.agx3_preamble_inputs));

   s.b.info.has_preamble = true;
   s.b.info.preamble_offset = 32;
   s.b.info.preamble_size = 6;
   s.b.info.nr_preamble_gprs = 40;
   s.b.info.agx3_preamble_staging_known = true;
   s.b.info.agx3_preamble_staging_halfs = 20;
   s.b.info.agx3_preamble_uniform_inputs = true;
   s.b.info.agx3_preamble_tps_inputs = true;
   s.b.info.preamble_scratch_size = s.b.info.agx3_preamble_inputs.storage_bytes;

   CHECK(!hk_fused_compute_shader_eligible(&dev, &s, &p));

   shader_bo.va = NULL;
   s.bo = &shader_bo;
   CHECK(!hk_fused_compute_shader_eligible(&dev, &s, &p));

   shader_bo.va = &shader_va;
   CHECK(hk_fused_compute_shader_eligible(&dev, &s, &p));
}

static void
test_g17p_fused_cprog_emit_matches_program(void)
{
   struct hk_device dev;
   struct hk_physical_device pdev;
   struct hk_shader s;
   struct hk_linked_shader linked;
   struct agx_bo shader_bo, linked_bo;
   struct agx_va shader_va, linked_va;
   struct hk_compute_pair_dispatch p;
   uint8_t code[512];

   init_base_fixture(&dev, &pdev, &s, &linked, &shader_bo, &shader_va,
                     &linked_bo, &linked_va, &p, code, sizeof(code));

   uint64_t halves[AGX3_CPROG_INPUT_WORDS] = {0};
   halves[0] = 0x3full; /* 6 halfs = 12 bytes -> aligned 16B cprog_frame */
   CHECK(agx3_cprog_inputs_init(halves, &s.b.info.agx3_preamble_inputs));
   CHECK(s.b.info.agx3_preamble_inputs.storage_bytes > 0);

   s.b.info.has_preamble = true;
   s.b.info.preamble_offset = 32;
   s.b.info.preamble_size = 6;
   s.b.info.nr_preamble_gprs = 40; /* > linked->b.nr_gprs (24) */
   s.b.info.agx3_preamble_staging_known = true;
   s.b.info.agx3_preamble_staging_halfs = 20; /* > agx3_staging_halfs (12) */
   s.b.info.agx3_preamble_uniform_inputs = true;
   s.b.info.agx3_preamble_tps_inputs = true;
   s.b.info.preamble_scratch_size = s.b.info.agx3_preamble_inputs.storage_bytes;

   uint8_t cs_buf[HK_CS_SPACE_PADDING + 1024] = {0};
   struct hk_cs cs = {
      .current = cs_buf,
      .end = cs_buf + sizeof(cs_buf),
   };

   uint8_t ioto_buf[4096] = {0};
   const uint64_t ioto_va = dev.dev.shader_base + 0x300000;
   size_t ioto_used = 0;

   CHECK(hk_emit_fused_compute_dispatch(&dev, &cs, &s, &p, ioto_buf, ioto_va,
                                        sizeof(ioto_buf), &ioto_used));
   CHECK(ioto_used > 0 && ioto_used <= sizeof(ioto_buf));
   CHECK(cs.current > (void *)cs_buf);
   CHECK(cs.weak_barrier_end == cs.current);

   /* Build the exact expected G17P compute program directly and verify
    * byte-for-byte equality with what hk_emit_fused_compute_dispatch packed. */
   struct agx3_ioto_uniform_range range = {0};
   CHECK(hk_agx3_g17p_root_range(s.info.agx3_root_uniform, s.info.set_count,
                                 p.root_va, hk_shader_root_size(&s), &range));

   struct hk_agx3_g17p_compute_program_desc expected_desc = {0};
   CHECK(agx3_cdm_direct_template_sources_compute(
      false, false, false, &expected_desc.launch_sources));
   CHECK(hk_agx3_g17p_constant_program_from_binary(
      s.code_ptr, s.code_size, s.b.info.preamble_offset, s.b.info.preamble_size,
      hk_shader_code_va(&s), s.b.info.binary_size,
      &expected_desc.constant_program));
   expected_desc.constant_program.force_external = true;
   expected_desc.constant_program.absolute_branch = true;
   expected_desc.constant_program.request_prelude = true;
   expected_desc.cprog_inputs = &s.b.info.agx3_preamble_inputs;
   const unsigned cprog_frame =
      ALIGN_POT(MAX2((unsigned)s.b.info.agx3_preamble_inputs.storage_bytes,
                     s.b.info.preamble_scratch_size),
                16);
   expected_desc.launch_sources.ti_spill_bytes =
      MAX2(expected_desc.launch_sources.ti_spill_bytes, cprog_frame);
   expected_desc.launch_sources.ti_largest_ipr_bytes =
      MAX2(expected_desc.launch_sources.ti_largest_ipr_bytes,
           (uint32_t)s.b.info.agx3_preamble_staging_halfs * 2u);
   expected_desc.program_va = ioto_va;
   expected_desc.shader_va = linked_va.addr + linked.b.entry_offset;
   expected_desc.uniform_ranges = &range;
   expected_desc.uniform_range_count = 1;
   expected_desc.shader_binary_size = linked.b.size;
   expected_desc.nr_gprs = 40;
   expected_desc.compiler_d_valid = true;
   expected_desc.compiler_d = 40;
   expected_desc.threadgroup_memory_size_B = 4096;
   expected_desc.ldshdr_f1 = AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3;

   struct hk_agx3_g17p_compute_program *expected = calloc(1, sizeof(*expected));
   CHECK(expected);
   CHECK(hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &expected_desc,
                                               expected));
   CHECK(ioto_used == expected->length);
   CHECK(memcmp(ioto_buf, expected->bytes, ioto_used) == 0);

   /* Confirm that omitting the constant program produces different bytes, so
    * the check above proves preamble wiring was actually applied. */
   struct hk_agx3_g17p_compute_program_desc no_cprog_desc = expected_desc;
   memset(&no_cprog_desc.constant_program, 0,
          sizeof(no_cprog_desc.constant_program));
   no_cprog_desc.cprog_inputs = NULL;
   no_cprog_desc.nr_gprs = linked.b.nr_gprs;
   no_cprog_desc.compiler_d = (uint32_t)s.b.info.agx3_staging_halfs * 2u;
   CHECK(agx3_cdm_direct_template_sources_compute(
      false, false, false, &no_cprog_desc.launch_sources));

   struct hk_agx3_g17p_compute_program *no_cprog = calloc(1, sizeof(*no_cprog));
   CHECK(no_cprog);
   CHECK(hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &no_cprog_desc,
                                               no_cprog));
   CHECK(ioto_used != no_cprog->length ||
         memcmp(ioto_buf, no_cprog->bytes, ioto_used) != 0);

   /* Invalid constant-program extent (odd size) must fail closed on emit. */
   s.b.info.preamble_size = 5;
   cs.current = cs_buf;
   size_t bad_used = 0xdead;
   CHECK(!hk_emit_fused_compute_dispatch(&dev, &cs, &s, &p, ioto_buf, ioto_va,
                                         sizeof(ioto_buf), &bad_used));
   CHECK(bad_used == 0xdead);

   /* Non-preamble dispatch still emits cleanly and matches no_cprog. */
   s.b.info.has_preamble = false;
   s.b.info.preamble_size = 0;
   s.b.info.preamble_scratch_size = 0;
   s.b.info.nr_preamble_gprs = 0;
   s.b.info.agx3_preamble_staging_halfs = 0;
   s.b.info.agx3_preamble_uniform_inputs = false;
   s.b.info.agx3_preamble_tps_inputs = false;
   memset(&s.b.info.agx3_preamble_inputs, 0,
          sizeof(s.b.info.agx3_preamble_inputs));

   cs.current = cs_buf;
   ioto_used = 0;
   CHECK(hk_emit_fused_compute_dispatch(&dev, &cs, &s, &p, ioto_buf, ioto_va,
                                        sizeof(ioto_buf), &ioto_used));
   CHECK(ioto_used == no_cprog->length);
   CHECK(memcmp(ioto_buf, no_cprog->bytes, ioto_used) == 0);

   /* Second emit with a shifted root_va and ioto_va exercises the cached
    * hk_g17p_loader patch-plan path and must remain byte-exact. */
   p.root_va += 0x400;
   const uint64_t ioto_va2 = dev.dev.shader_base + 0x301000;
   CHECK(hk_agx3_g17p_root_range(s.info.agx3_root_uniform, s.info.set_count,
                                 p.root_va, hk_shader_root_size(&s), &range));
   no_cprog_desc.program_va = ioto_va2;
   no_cprog_desc.uniform_ranges = &range;
   CHECK(hk_agx3_g17p_compute_program_generate(AGX_CHIP_G17P, &no_cprog_desc,
                                               no_cprog));
   cs.current = cs_buf;
   ioto_used = 0;
   CHECK(hk_emit_fused_compute_dispatch(&dev, &cs, &s, &p, ioto_buf, ioto_va2,
                                        sizeof(ioto_buf), &ioto_used));
   CHECK(ioto_used == no_cprog->length);
   CHECK(memcmp(ioto_buf, no_cprog->bytes, ioto_used) == 0);

   hk_g17p_linked_compute_cache_finish(&linked);
   free(no_cprog);
   free(expected);
}

int
main(void)
{
   test_g17p_fused_cprog_eligibility();
   test_g17p_fused_cprog_emit_matches_program();
   printf("PASS hk_fused_cprog (%u checks)\n", checks);
   return 0;
}

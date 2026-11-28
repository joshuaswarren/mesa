/* SPDX-License-Identifier: MIT */
/* M5 program composition over the shared uniform-load scheduler. */
#include "agx3_graphics_program.h"
#include "agx3_ioto_load_packer.h"
#include "asahi/lib/agx_ioto_state_core.h"
#include "asahi/lib/agx_ioto_preshader_core.h"
#include "util/macros.h"
#include <string.h>

bool
agx3_g17g_graphics_program_bound(
   enum agx_chip chip, const struct agx3_ioto_uniform_range *ranges,
   size_t range_count, size_t *size)
{
   size_t body, chunks;
   if (!size || !agx_chip_g17_m5(chip) ||
       range_count > AGX3_G17P_GRAPHICS_MAX_RANGES ||
       !agx3_ioto_uniform_program_body_bound(chip, ranges, range_count,
                                             &body, &chunks) ||
       chunks > AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS ||
       body > AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE -
                 AGX_IOTO_G17G_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE)
      return false;

   *size = body + AGX_IOTO_G17G_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE;
   return true;
}

static bool
g17g_graphics_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   enum agx_ioto_g17g_program_stage stage,
   struct agx3_g17p_graphics_program *program,
   struct agx3_g17p_graphics_program_template *template)
{
   size_t bound;
   const bool mesh = stage == AGX_IOTO_G17G_PROGRAM_MESH ||
                     stage == AGX_IOTO_G17G_PROGRAM_OBJECT;
   if (!agx_chip_g17_m5(chip) ||
       !desc || !program || (mesh && desc->fragment) ||
       !desc->shader_binary_size ||
       desc->preloaded_root ||
       (desc->cprog_inputs && desc->cprog_inputs->count &&
        !desc->constant_program.code_size) ||
       (desc->constant_program.code_size &&
        stage != AGX_IOTO_G17G_PROGRAM_COMPUTE &&
        stage != AGX_IOTO_G17G_PROGRAM_VERTEX &&
        stage != AGX_IOTO_G17G_PROGRAM_FRAGMENT &&
        stage != AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT) ||
       !desc->nr_gprs || desc->nr_gprs > 256 ||
       desc->scratch_size > UINT16_MAX * 4u ||
       !desc->compiler_d_valid ||

       desc->compiler_d >
          ((desc->fragment || stage == AGX_IOTO_G17G_PROGRAM_COMPUTE ||
            stage == AGX_IOTO_G17G_PROGRAM_VERTEX) ? 512 : 256) ||
       (desc->writes_sample_mask && !desc->fragment) ||
       (desc->threadgroup_memory_size_B &&
        ((stage != AGX_IOTO_G17G_PROGRAM_COMPUTE && !mesh) ||
         desc->threadgroup_memory_size_B < (mesh ? 64 : 256) ||
         desc->threadgroup_memory_size_B > 32768 ||
         (desc->threadgroup_memory_size_B & 63))) ||
       (desc->colour_classes & ~7u) ||
       (!desc->fragment && desc->colour_classes) ||

       (desc->fragment &&
        ((desc->imageblock.sample_size_B > 248 ||
          (desc->imageblock.sample_size_B & 7)) ||
         (desc->imageblock.nr_samples != 1 &&
          desc->imageblock.nr_samples != 2 &&
          desc->imageblock.nr_samples != 4 &&
          desc->imageblock.nr_samples != 8) ||
         (desc->imageblock.tile_size_px != 32 * 32 &&
          desc->imageblock.tile_size_px != 32 * 16 &&
          desc->imageblock.tile_size_px != 16 * 16) ||
         desc->imageblock.nr_samples * desc->imageblock.tile_size_px > 2048 ||
         desc->imageblock.sample_size_B * desc->imageblock.nr_samples *
            desc->imageblock.tile_size_px > 65536)) ||
       !agx3_g17g_graphics_program_bound(chip, desc->uniform_ranges,
                                          desc->uniform_range_count, &bound))
      return false;


   static const uint8_t speclm[2][AGX_IOTO_SPECLM_SIZE] = {
      {0x07,0x00,0x00,0x0c,0x00,0x00,0x00,0x00},
      {0x17,0x00,0x00,0x2c,0x00,0x00,0x18,0x08},
   };
   struct agx_ioto_state ldshdr = {0}, spectpr = {0};
   ldshdr.f[0] = desc->shader_va;

   const uint64_t span =
      (((uint64_t)desc->shader_binary_size + 63) & ~UINT64_C(63)) >> 7;
   ldshdr.f[1] = MIN2(span, 3);
   ldshdr.f[5] = 3;
   ldshdr.f[6] = 107;
   ldshdr.f[23] = 1;
   spectpr.f[0] = 3;
   spectpr.f[1] = 166;
   spectpr.f[20] = DIV_ROUND_UP(desc->scratch_size, 4);
   spectpr.f[21] = spectpr.f[20];
   spectpr.f[22] = (((desc->compiler_d + 3) >> 2) + 1) & ~1u;

   struct agx_ioto_state_tail_output tail = {
      .length = AGX_IOTO_STATE_TAIL_TYPE0_SIZE,
   };
   /* Original LdShdrTransformFragment skips encodeImageBlockData when the
    * allocation is zero and emits the no-imageblock form, including for
    * fragment tile programs. Nonzero strides are rounded to eight bytes;
    * they need not be powers of two (native stride24 emits byte7=0x18). */
   memcpy(tail.bytes,
          speclm[desc->fragment && desc->imageblock.sample_size_B != 0],
          AGX_IOTO_SPECLM_SIZE);
   if (desc->threadgroup_memory_size_B) {

      struct agx_ioto_state shared = {0};
      shared.f[0] = 3;
      shared.f[1] = 165;
      shared.f[18] = desc->threadgroup_memory_size_B;
      shared.f[24] = 1;
      if (!agx_ioto_state_pack_speclm(chip, &shared, tail.bytes,
                                       AGX_IOTO_SPECLM_SIZE))
         return false;
   }
   if (desc->fragment && desc->imageblock.sample_size_B) {

      tail.bytes[7] = (uint8_t)desc->imageblock.sample_size_B;
      tail.bytes[3] = desc->imageblock.tile_size_px == 16 * 16 ? 0xac :
                      desc->imageblock.tile_size_px == 32 * 16 ? 0x1c : 0x2c;
      tail.bytes[6] = desc->imageblock.tile_size_px == 16 * 16 ? 0x10 : 0x18;
   }
   if (desc->fragment) {
      /* Sample count and tile geometry occupy independent SpecLM fields. */
      const unsigned samples = desc->imageblock.nr_samples;
      tail.bytes[6] |= (samples == 8 ? 3u : samples == 4 ? 2u :
                           samples == 2 ? 1u : 0u) << 6;
   }
   if (!agx_ioto_state_pack_ldshdr(
          chip, &ldshdr, tail.bytes + AGX_IOTO_SPECLM_SIZE,
          AGX_IOTO_LDSHDR_SIZE) ||
       !agx_ioto_state_pack_spectpr(
          chip, &spectpr, tail.bytes + AGX_IOTO_SPECLM_SIZE +
                            AGX_IOTO_LDSHDR_SIZE, AGX_IOTO_SPECTPR_SIZE))
      return false;

   uint8_t body[AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE];
   struct agx_ioto_inflight_load loads[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
   struct agx_ioto_inflight_load sort[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
   struct agx_ioto_load_cache_entry cache[AGX3_G17P_GRAPHICS_MAX_LOAD_CHUNKS];
   size_t relocation_count = 0;
   const struct agx3_ioto_uniform_program_desc load_desc = {
      .program_va = desc->program_va,
      .ranges = desc->uniform_ranges,
      .range_count = desc->uniform_range_count,
      .patchable = desc->patchable,
      .cprog_inputs = desc->cprog_inputs,
   };
   const struct agx3_ioto_uniform_program_workspace workspace = {
      .body = body, .body_capacity = sizeof(body),
      .loads = loads, .load_capacity = ARRAY_SIZE(loads),
      .sort = sort, .sort_capacity = ARRAY_SIZE(sort),
      .cache = cache, .cache_capacity = ARRAY_SIZE(cache),
      .relocations = template ? template->relocations : NULL,
      .relocation_capacity = template ? ARRAY_SIZE(template->relocations) : 0,
      .relocation_count = &relocation_count,
   };
   size_t input_bound, input_chunks;
   if (!agx3_ioto_uniform_program_desc_body_bound(
          chip, &load_desc, &input_bound, &input_chunks) ||
       input_chunks > ARRAY_SIZE(loads) || input_bound > sizeof(body))
      return false;
   /* The existing range bound excludes import stores and CProg suffix.
    * The public program buffer is the transactional final bound here. */
   if (desc->constant_program.code_size)
      bound = AGX3_G17P_GRAPHICS_PROGRAM_MAX_SIZE;
   size_t body_size, copy_offset, written;
   struct agx3_g17p_graphics_program staged = {0};
   if (!agx3_ioto_uniform_program_body_generate(
          chip, &load_desc, workspace, &body_size, &copy_offset) ||
       !agx_ioto_m5_resource_program_compose_classes(
          chip, desc->usc_clusters, desc->program_va, body, body_size, &tail,
          stage, MAX2(desc->colour_classes, 1),
          staged.bytes, sizeof(staged.bytes), &written) ||
       written > bound || written > UINT16_MAX)
      return false;
   /* The retained native compute loader executes the profile callback before
    * publishing CProg input TPS. The callback context effect is
    * unknown; preserve that observed order in this probe. Put the prefix
    * before all loads:
    * placing it merely before COPY can clobber pending GPR0/2 carriers.
    * Graphics stages keep their composed order: the constant program only
    * replaces the final END, after PIXREL, as HAL300
    * native state-load encoder's finish step appends it (PIXREL, then
    * the inline code or the external branch). */
   const bool compute_cprog = desc->constant_program.code_size &&
                              stage == AGX_IOTO_G17G_PROGRAM_COMPUTE;
   if (compute_cprog) {
      const size_t state_size = AGX_IOTO_STATE_TAIL_TYPE0_SIZE;
      const size_t helper_size = AGX_IOTO_G17S_MINIMAL_PROFILE_LOADER_SIZE;
      const size_t prefix_size = state_size + helper_size;
      const size_t end_size = AGX_IOTO_PROGRAM_TERMINATE_SIZE;
      uint8_t helper[AGX_IOTO_G17S_MINIMAL_PROFILE_LOADER_SIZE];
      if (written != body_size + prefix_size + end_size)
         return false;
      memcpy(helper, staged.bytes + body_size + state_size, helper_size);
      memcpy(staged.bytes, tail.bytes, state_size);
      memcpy(staged.bytes + state_size, helper, helper_size);
      memcpy(staged.bytes + prefix_size, body, body_size);
   }
   const size_t constant_suffix_offset = written - AGX_IOTO_PROGRAM_TERMINATE_SIZE;
   if (desc->constant_program.code_size) {
      static const uint8_t absent[] = {0x0e, 0, 0, 0};
      if (written < sizeof(absent) ||
          memcmp(staged.bytes + written - sizeof(absent), absent, sizeof(absent)))
         return false;
      const size_t suffix_offset = written - sizeof(absent);
      struct agx_ioto_preshader_output suffix;
      if (!agx_ioto_constant_program_emit(chip, &desc->constant_program,
             desc->program_va + suffix_offset, &suffix) ||
          suffix.length > sizeof(staged.bytes) - suffix_offset)
         return false;
      memcpy(staged.bytes + suffix_offset, suffix.bytes, suffix.length);
      written = suffix_offset + suffix.length;
   }
   if (template) {
      /* A large uniform pool can flush an earlier allocation round.
       * Initialize shader state before its copies too, then relocate every
       * later address materialization past the inserted state prefix. */
      size_t first_copy;
      if (!agx_ioto_program_first_copy_offset(chip, body, body_size, &first_copy))
         return false;
      const size_t prefix = AGX_IOTO_STATE_TAIL_TYPE0_SIZE +
         (stage == AGX_IOTO_G17G_PROGRAM_COMPUTE ? 0 :
          AGX_IOTO_G17S_MINIMAL_PROFILE_LOADER_SIZE);
      for (size_t i = 0; i < relocation_count; ++i) {
         struct agx_ioto_load_relocation *r = &template->relocations[i];
         if (compute_cprog) {
            r->offset += AGX_IOTO_STATE_TAIL_TYPE0_SIZE +
                         AGX_IOTO_G17S_MINIMAL_PROFILE_LOADER_SIZE;
         } else if (r->offset >= first_copy) {
            r->offset += prefix;
         } else if (r->offset + r->low_length + r->high_length > first_copy) {
            return false;
         }
      }
      template->relocation_count = relocation_count;
      template->constant_suffix_offset = constant_suffix_offset;
   }
   staged.length = written;
   *program = staged;
   return true;
}

bool
agx3_g17g_graphics_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program *program)
{
   /* G18G keeps its own builder (agx3_graphics_program.c). */
   if (chip == AGX_CHIP_G18G)
      return agx3_g18g_graphics_program_generate(desc, program);

   const enum agx_ioto_g17g_program_stage stage =
      !desc                      ? AGX_IOTO_G17G_PROGRAM_VERTEX
      : !desc->fragment          ? AGX_IOTO_G17G_PROGRAM_VERTEX
      : desc->writes_sample_mask ? AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT
                                 : AGX_IOTO_G17G_PROGRAM_FRAGMENT;
   return desc && g17g_graphics_program_generate(chip, desc, stage, program, NULL);
}

bool
agx3_g17g_mesh_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   bool object, struct agx3_g17p_graphics_program *program)
{
   /* The native mesh/object tails use the ordinary TA order: shader state,
    * profile helper, then uniform COPYs. TGmem64/192 are captured leaves;
    * full dispatch ownership and resource budgets belong to the caller. */
   return desc && g17g_graphics_program_generate(chip, desc,
      object ? AGX_IOTO_G17G_PROGRAM_OBJECT : AGX_IOTO_G17G_PROGRAM_MESH,
      program, NULL);
}

bool
agx3_g17g_compute_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program *program)
{
   /* G18G keeps its own builder; this entry point has no TPS frame. */
   if (chip == AGX_CHIP_G18G)
      return desc && !desc->scratch_size &&
             agx3_g18g_compute_program_generate(desc, program);
   return desc && !desc->fragment && g17g_graphics_program_generate(
      chip, desc, AGX_IOTO_G17G_PROGRAM_COMPUTE, program, NULL);
}

bool
agx3_g17g_tile_program_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   bool store, struct agx3_g17p_graphics_program *program)
{
   /* G18G keeps its own builder (agx3_graphics_program.c). */
   if (chip == AGX_CHIP_G18G)
      return agx3_g18g_tile_program_generate(desc, store, program);
   return desc && desc->fragment && g17g_graphics_program_generate(
      chip, desc, store ? AGX_IOTO_G17G_PROGRAM_EOT
                       : AGX_IOTO_G17G_PROGRAM_BACKGROUND, program, NULL);
}

bool
agx3_g18g_general_tile_program_generate(
   const struct agx3_g17p_graphics_program_desc *desc, bool store,
   struct agx3_g17p_graphics_program *program)
{
   if (!desc)
      return false;
   struct agx3_g17p_graphics_program_desc g18_desc = *desc;
   g18_desc.usc_clusters = 1;
   return agx3_g17g_tile_program_generate(AGX_CHIP_G17G, &g18_desc, store,
                                          program);
}

bool
agx3_g17g_graphics_program_template_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program_template *program)
{
   if (!desc || !program)
      return false;
   const enum agx_ioto_g17g_program_stage stage =
      !desc->fragment ? AGX_IOTO_G17G_PROGRAM_VERTEX
      : desc->writes_sample_mask ? AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT
                                 : AGX_IOTO_G17G_PROGRAM_FRAGMENT;
   return g17g_graphics_program_generate(chip, desc, stage,
                                         &program->program, program);
}


bool
agx3_g17g_compute_program_template_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   struct agx3_g17p_graphics_program_template *program)
{
   return desc && program && !desc->fragment &&
          g17g_graphics_program_generate(chip, desc,
             AGX_IOTO_G17G_PROGRAM_COMPUTE, &program->program, program);
}

bool
agx3_g17g_tile_program_template_generate(
   enum agx_chip chip, const struct agx3_g17p_graphics_program_desc *desc,
   bool store, struct agx3_g17p_graphics_program_template *program)
{
   return desc && program && desc->fragment &&
          g17g_graphics_program_generate(chip, desc,
             store ? AGX_IOTO_G17G_PROGRAM_EOT : AGX_IOTO_G17G_PROGRAM_BACKGROUND,
             &program->program, program);
}

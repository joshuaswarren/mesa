/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_bg_eot.h"
#include "agx_g17p_wide_mrt_eot.h"
#include "util/simple_mtx.h"
#include "util/u_debug.h"
#include "agx_compile.h"
#include "agx3_code_container_core.h"
#include "agx3_tile_core.h"
#include "agx_device.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "agx_tilebuffer.h"
#include "agx_usc.h"
#include "libagx_shaders.h"
#include "nir.h"
#include "nir_builder.h"
#include "pool.h"

#include <inttypes.h>
#include <stdlib.h>

/* msaa163: a G18G two- or four-sample tile. Its
 * background stores through the SR39 register mask and its end-of-tile program
 * resolves through the ordinary 2D store (MSAA notes P4/P5). */
static bool
agx_g18_msaa4_tib(const struct agx_device *dev,
                  const struct agx_tilebuffer_layout *tib)
{
   return dev->chip == AGX_CHIP_G18G &&
          (tib->nr_samples == 2 || tib->nr_samples == 4);
}

static bool
agx_g17p_rt0_eot_text_supported(const struct agx_shader_part *bin)
{
   /* Admit only complete native builtin texts: the common 68-byte RT0
    * program, an exact 84-byte layered or signed-integer RT0 program, the
    * admitted 120-byte RGBA8 pair, or the
    * 238-byte different-partition RGBA16F+R8 WBOIT pair. None has a synthetic
    * compute preamble; the CODE container supplies the stop constant. */
   const bool rt0 = bin && bin->info.binary_size == 68 &&
                    bin->info.nr_gprs == 8 &&
                    bin->info.agx3_staging_halfs == 8;
   const bool native_84_rt0 = bin && bin->info.binary_size == 84 &&
                              bin->info.nr_gprs == 8 &&
                              bin->info.agx3_staging_halfs == 8;
   const bool rgba8_mrt = bin && bin->info.binary_size == 120 &&
                          bin->info.nr_gprs == 8 &&
                          bin->info.agx3_staging_halfs == 16;
   const bool wboit_mrt = bin && bin->info.binary_size == 238 &&
                          bin->info.nr_gprs == 8 &&
                          bin->info.agx3_staging_halfs == 16;
   /* The multi-target program of two to eight targets, eight staging halves
    * and one 44-byte tail each. */
   const bool raw32_mrt = bin && bin->info.agx3_staging_halfs >= 16 &&
                         bin->info.agx3_staging_halfs <=
                            8 * AGX_MAX_RENDER_TARGETS &&
                         bin->info.agx3_staging_halfs % 8 == 0 &&
                         bin->info.binary_size ==
                            32 + 44 * (bin->info.agx3_staging_halfs / 8) &&
                         bin->info.nr_gprs == 8 &&
                         bin->info.agx3_staging_known;
   /* The 2D-ARRAY multi-target program of two to eight targets. It carries the
    * layer wire in addition to the XY tile origin, so it has a longer common
    * lead and a 54-byte tail each, and Apple's metadata asks for five GPR
    * dwords rather than four. See agx3_g17p_array_eot.h. */
   const bool array_mrt = bin && bin->info.agx3_staging_halfs >= 16 &&
                          bin->info.agx3_staging_halfs <=
                             8 * AGX_MAX_RENDER_TARGETS &&
                          bin->info.agx3_staging_halfs % 8 == 0 &&
                          bin->info.binary_size ==
                             42 + 54 * (bin->info.agx3_staging_halfs / 8) &&
                          bin->info.nr_gprs == 10 &&
                          bin->info.agx3_staging_known;
   return bin && bin->binary &&
          (rt0 || native_84_rt0 || rgba8_mrt || wboit_mrt || raw32_mrt ||
           array_mrt) &&
          bin->info.main_offset == 0 &&
          bin->info.main_size == bin->info.binary_size &&
          !bin->info.scratch_size && bin->info.agx3_staging_known;
}

static bool
agx_g17p_upload_bg_eot(struct agx_bg_eot_cache *cache,
                       const struct agx_shader_part *bin,
                       struct agx_bg_eot_shader *res)
{
   /* Match the standalone allocation lead already used for G17P main
    * shaders. Native packed CODE blocks do not each own such a lead. */
   const bool m5 = agx_chip_g17_m5(cache->dev->chip);
   const size_t lead_size =
      cache->dev->chip == AGX_CHIP_G18G ? AGX3_G18G_CONTAINER_LEAD_SIZE
      : m5 ? (size_t)cache->dev->params.num_clusters_total * 0x300u + 0x40u
           : 0x340;
   /* LDSHDR's existing auxiliary fetch-span metadata is based on compiler
    * text size. Reserve at least that span after the main entry too, so the
    * EOT text128/main64 case cannot prefetch outside this CODE block. */
   const size_t main_span = MAX2(bin->info.main_size, bin->info.binary_size);
   const size_t block_size = agx3_code_block_size(main_span);
   const unsigned main_offset = bin->info.main_offset;
   if (!bin->binary || !block_size || block_size > SIZE_MAX - lead_size ||
       bin->info.has_preamble ||
       bin->info.rodata.size_16 ||
       (main_offset != 0 &&
        main_offset != AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE) ||
       main_offset > bin->info.binary_size ||
       bin->info.main_size > bin->info.binary_size - main_offset) {
      mesa_loge("asahi: G17P %s program geometry is unproven: binary=%u "
                "main=%u@%u preamble=%u rodata=%u",
                bin->info.stage == MESA_SHADER_COMPUTE ? "end-of-tile"
                                                       : "background",
                bin->info.binary_size, bin->info.main_size, main_offset,
                bin->info.has_preamble, bin->info.rodata.size_16);
      return false;
   }

   const size_t size = lead_size + block_size;
   uint8_t *image = malloc(size);
   if (!image)
      return false;

   if (cache->dev->chip == AGX_CHIP_G18G) {
      if (!agx3_build_g18g_container_lead(image, lead_size)) {
         free(image);
         return false;
      }
   } else if (m5) {
      if (!agx3_build_m5_container_lead(image, lead_size,
                                       cache->dev->params.num_clusters_total)) {
         free(image);
         return false;
      }
   } else {
      agx3_build_container_lead(image, cache->dev->chip, cache->dev->params.num_cores_per_cluster);
   }
   const uint8_t *text = bin->binary;
   bool built = agx3_build_code_block(
      image + lead_size, block_size, text + main_offset, bin->info.main_size,
      main_span, main_offset ? text : NULL, true);
   /* G18G: the compiled text is agx3d already; only the synthesized stop
    * program is agx3c. */
   if (built && !main_offset && cache->dev->chip == AGX_CHIP_G18G)
      agx3_build_container_stop_program_agx3d(
         image + lead_size + AGX3_CODE_BLOCK_HEADER_SIZE);
   const uint64_t allocation = built
      ? agx_pool_upload_aligned_with_bo(&cache->pool, image, size, 128,
                                         &res->bo)
      : 0;
   if (!allocation) {
      free(image);
      return false;
   }

   /* Keep compiler-text metadata intact: native EOT has binary_size=68,
    * main_offset=0 and main_size=68. Its LDSHDR fetch-span calculation uses
    * the main padded to128, not this allocation's size. Rebase ptr so
    * the existing ptr + main_offset expression names CODE block +0x80.
    * Clear/reload have main_offset=0 and get a fragment constant program.
    *
    * The valid 194053 first EOT Ioto names a different, larger shader whose
    * code page was not captured. This repairs the common loader contract;
    * it is not an assertion of byte parity with that uncaptured EOT block.
    */
   res->ptr = allocation + lead_size + AGX3_CODE_BLOCK_ENTRY_OFFSET - main_offset;
   free(image);
   return true;
}

static bool
lower_tex_handle_to_u0(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_texture_handle_agx)
      return false;

   const unsigned descriptor_size = *(const unsigned *)data;
   assert(descriptor_size == 24 || descriptor_size == 32);

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def_rewrite_uses(
      &intr->def, nir_bindless_image_agx(
                     b, nir_imul_imm(b, intr->src[0].ssa, descriptor_size),
                     .desc_set = 0));

   return true;
}

static unsigned
agx_bg_eot_texture_descriptor_size(const struct agx_device *dev)
{
   switch (agx_chip_descriptor_generation(dev->chip)) {
   case AGX_DESCRIPTOR_GENERATION_7:
   case AGX_DESCRIPTOR_GENERATION_8:
      return 24;
   case AGX_DESCRIPTOR_GENERATION_9:
   case AGX_DESCRIPTOR_GENERATION_10:
      return 32;
   case AGX_DESCRIPTOR_GENERATION_UNKNOWN:
   default:
      return 0;
   }
}

static struct agx_bg_eot_shader *
agx_compile_bg_eot_shader(struct agx_bg_eot_cache *cache, nir_shader *shader,
                          struct agx_shader_key *key,
                          struct agx_tilebuffer_layout *tib)
{
   agx_nir_lower_texture(shader);
   agx_preprocess_nir(shader);
   if (tib) {
      unsigned descriptor_size =
         agx_bg_eot_texture_descriptor_size(cache->dev);

      if (!descriptor_size) {
         mesa_loge("asahi: background shader has no proven native descriptor geometry");
         ralloc_free(shader);
         return NULL;
      }

      unsigned bindless_base = 0;
      NIR_PASS(_, shader, agx_nir_lower_tilebuffer_with_conversions, tib, NULL,
               &bindless_base, NULL, NULL, false,
               cache->dev->chip == AGX_CHIP_G18G,
               cache->dev->chip == AGX_CHIP_G18G,
               agx_chip_native_or_m5(cache->dev->chip),
               cache->dev->chip == AGX_CHIP_G18G);
      NIR_PASS(_, shader, agx_nir_lower_monolithic_msaa, tib->nr_samples);
      NIR_PASS(_, shader, agx_nir_lower_multisampled_image_store);
      NIR_PASS(_, shader, agx_nir_lower_texture);
      NIR_PASS(_, shader, nir_shader_intrinsics_pass, lower_tex_handle_to_u0,
               nir_metadata_control_flow, &descriptor_size);
      if (agx_chip_native_agx3(cache->dev->chip) ||
          (agx_chip_g17_m5(cache->dev->chip) && tib->nr_samples > 1)) {
         /* Monolithic MSAA introduces a small sample loop after the common
          * preprocessing pass has already run. Fold that loop here so each
          * local-pixel access has the static one-hot sample mask required by
          * the native G17P tile selector. */
         bool progress;
         do {
            progress = false;
            NIR_PASS(progress, shader, nir_lower_vars_to_ssa);
            NIR_PASS(progress, shader, nir_opt_copy_prop);
            NIR_PASS(progress, shader, nir_opt_constant_folding);
            NIR_PASS(progress, shader, nir_opt_loop_unroll);
            NIR_PASS(progress, shader, nir_opt_dead_cf);
            NIR_PASS(progress, shader, nir_opt_dce);
         } while (progress);

         NIR_PASS(_, shader, agx_nir_lower_g17p_tile_samples,
                  tib->nr_samples);
      } else if (agx_g18_msaa4_tib(cache->dev, tib)) {
         /* msaa163: the same fold, then G18's four-sample form. Apple's G18
          * 4x background stores its clear through rld SR39 into the lstcol
          * register form (f51=2, f55=1), exactly what this produces; the
          * backend refuses any tile store left with an immediate mask. */
         bool progress;
         do {
            progress = false;
            NIR_PASS(progress, shader, nir_lower_vars_to_ssa);
            NIR_PASS(progress, shader, nir_opt_copy_prop);
            NIR_PASS(progress, shader, nir_opt_constant_folding);
            NIR_PASS(progress, shader, nir_opt_loop_unroll);
            NIR_PASS(progress, shader, nir_opt_dead_cf);
            NIR_PASS(progress, shader, nir_opt_dce);
         } while (progress);

         NIR_PASS(_, shader, agx_nir_lower_g18_tile_samples,
                  tib->nr_samples);
      }
   }

   struct agx_bg_eot_shader *res = rzalloc(cache->ht, struct agx_bg_eot_shader);
   struct agx_shader_part bin;
   if (!agx_compile_shader_nir(shader, key, &bin)) {
      mesa_loge("asahi: the %s shader did not compile for this chip",
                shader->info.stage == MESA_SHADER_COMPUTE ? "end-of-tile"
                                                          : "background");
      ralloc_free(shader);
      ralloc_free(res);
      return NULL;
   }

   if (agx_chip_native_agx3(cache->dev->chip) && tib == NULL &&
       !agx_g17p_rt0_eot_text_supported(&bin)) {
      free(bin.binary);
      ralloc_free(shader);
      ralloc_free(res);
      return NULL;
   }

   res->info = bin.info;
   if (agx_chip_agx3_path(cache->dev->chip)) {
      if (!agx_g17p_upload_bg_eot(cache, &bin, res)) {
         free(bin.binary);
         ralloc_free(shader);
         ralloc_free(res);
         return NULL;
      }
   } else {
      res->ptr = agx_pool_upload_aligned_with_bo(
         &cache->pool, bin.binary, bin.info.binary_size, 128, &res->bo);
   }
   free(bin.binary);
   ralloc_free(shader);

   return res;
}

static nir_def *
build_background_op(nir_builder *b, enum agx_bg_eot_op op, unsigned rt,
                    unsigned nr, bool source_msaa, bool layered,
                    bool wide_coords, unsigned clear_base)
{
   if (op == AGX_BG_LOAD) {
      nir_def *coord = nir_load_pixel_coord(b);

      /* MSAA/array lowerings assume 32-bit coordinates, could be lifted but
       * that's going to need more work.
       */
      if (layered || source_msaa || wide_coords)
         coord = nir_u2u32(b, coord);

      if (layered) {
         coord = nir_vec3(b, nir_channel(b, coord, 0), nir_channel(b, coord, 1),
                          nir_load_layer_id(b));
      }

      b->shader->info.fs.uses_sample_shading |= source_msaa;

      nir_def *tex = nir_txf(
         b, coord, .ms_index = source_msaa ? nir_load_sample_id(b) : NULL,
         .texture_index = rt * 2,
         .dim = source_msaa ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D,
         .is_array = layered,

         /* The type doesn't matter as long as it matches the store */
         .dest_type = nir_type_uint32,

         /* Layer is necessarily already in-bounds so we do not want the
          * compiler to clamp it, which would require reading the descriptor
          */
         .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP);

      return nir_trim_vector(b, tex, nr);
   } else {
      assert(op == AGX_BG_CLEAR);

      return nir_load_preamble(b, nr, 32, clear_base + (rt * 8));
   }
}

static struct agx_bg_eot_shader *
agx_build_background_shader(struct agx_bg_eot_cache *cache,
                            struct agx_bg_eot_key *key)
{
   if (agx_chip_g17_m5(cache->dev->chip) && !key->tib.sample_size_B &&
       key->tib.nr_samples <= 1 && !key->tib.layered &&
       !key->reserved_preamble && !agx_tilebuffer_spills(&key->tib)) {
      bool empty = true;
      for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt)
         empty &= key->op[rt] == AGX_BG_EOT_NONE;
      if (empty) {
         /* Fresh G17G's no-op compute object is exactly END, with common
          * field0=1 GPR dword and absent field38/42 (zero staging bytes).
          * BG and EOT loaders use the same empty executable; their release
          * policy remains in the native stage-specific Ioto program. */
         uint8_t text[] = {0x0e, 0, 0, 0};
         const struct agx_shader_part bin = {
            .binary = text,
            .info = {
               .stage = MESA_SHADER_COMPUTE,
               .binary_size = sizeof(text),
               .main_size = sizeof(text),
               .nr_gprs = 2,
               .agx3_staging_known = true,
               .agx3_staging_halfs = 0,
            },
         };
         struct agx_bg_eot_shader *res =
            rzalloc(cache->ht, struct agx_bg_eot_shader);
         if (!res || !agx_g17p_upload_bg_eot(cache, &bin, res)) {
            ralloc_free(res);
            return NULL;
         }
         res->info = bin.info;
         return res;
      }
   }

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "agx_background");
   b.shader->info.fs.untyped_color_outputs = true;

   struct agx_shader_key compiler_key = {
      .dev = agx_gather_device_key(cache->dev),
      /* M5 multisample backgrounds retain the colour tile transaction.
       *
       * A background owns its pixels first, so it normally skips the pixel
       * protocol. Not G18's four-sample one: Apple's 4x background waits
       * before its register-mask stores, and this background without the
       * pixwait/pixrel pair stalled the 3D stage (run166c: the same 4x render
       * with an empty background retires). */
      .fs.ignore_tib_dependencies =
         (!agx_chip_g17_m5(cache->dev->chip) || key->tib.nr_samples == 1) &&
         !agx_g18_msaa4_tib(cache->dev, &key->tib),
      .fs.background = true,
      /* Native multisample backgrounds store through the coverage register,
       * and so do G18G four-sample ones (msaa163). */
      .fs.nr_samples =
         agx_chip_native_or_m5(cache->dev->chip) ||
               agx_g18_msaa4_tib(cache->dev, &key->tib)
            ? key->tib.nr_samples : 0,
      .reserved_preamble = key->reserved_preamble,
      .no_preamble = agx_chip_agx3_path(cache->dev->chip),
   };

   /* The multi-target background protocol is keyed, not inferred: without
    * this the packer sees a bare run of tile stores, refuses the shader and
    * the draw is dropped. Vulkan says so through mixed_mrt; GL reaches the
    * same shapes through an ordinary draw-buffer mask, and
    * KHR-GL30.draw_buffers clears eight RGBA8 targets at once. */
   unsigned background_targets = 0;
   bool has_load = false;
   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt)
      background_targets += key->op[rt] != AGX_BG_EOT_NONE;
   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt)
      has_load |= key->op[rt] == AGX_BG_LOAD;
   compiler_key.g17p_mrt_bg = key->mixed_mrt || background_targets > 1;
   /* Run 1139: a G18 single-sample mixed-format tile's background stores each
    * attachment once with its lane form (agx3_g17p_multi_background_protocol). */
   compiler_key.g18_native_mixed =
      cache->dev->chip == AGX_CHIP_G18G &&
      agx_g18g_native_mixed_mask(&key->tib) != 0;
   /* Native464's resident three-target MS background owns a combined
    * class252 transaction. The packer validates every sample/partition and
    * emits its single acquire, without generic per-store class12 releases.
    * Preserve the established single-target MS dependency path above. */
   if (cache->dev->chip == AGX_CHIP_G18G && background_targets == 3 &&
       util_bitcount(agx_g18g_ms_mrt_mask(&key->tib)) == 3)
      compiler_key.fs.ignore_tib_dependencies = true;
   /* A reload binds texture/sampler pointers in u0..7. A mixed G18
    * background must place clear values after both pointers. */
   unsigned clear_base =
      cache->dev->chip == AGX_CHIP_G18G && has_load ? 8 : 4;
   if (agx_chip_g17_m5(cache->dev->chip) &&
       (key->mixed_mrt || background_targets > 1)) {
      for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
         if (key->op[rt] == AGX_BG_LOAD)
            clear_base = 8;
      }
   }

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
      if (key->op[rt] == AGX_BG_EOT_NONE)
         continue;

      unsigned nr = util_format_get_nr_components(key->tib.logical_format[rt]);
      bool msaa = key->tib.nr_samples > 1;
      bool source_msaa = msaa && !key->single_sample_reload;
      bool layered = key->tib.layered;
      assert(nr > 0);

      if (key->g17p_packed_clear && key->op[rt] == AGX_BG_CLEAR &&
          agx_chip_native_agx3(cache->dev->chip) && rt == 0 &&
          agx_g17p_packed_clear_ok(&key->tib)) {
         /* One raw tile word: the driver packed the clear colour exactly as
          * the R8G8B8A8_UNORM FLTNORM pair would, so store its 32 bits. */
         nir_def *packed = nir_load_preamble(&b, 1, 32, 4 + (rt * 8));
         nir_store_local_pixel_agx(&b, packed,
                                   nir_imm_intN_t(&b, 0xFF, 16),
                                   nir_undef(&b, 2, 16),
                                   .base = agx_tilebuffer_offset_B(&key->tib, rt),
                                   .write_mask = 0x1,
                                   .format = PIPE_FORMAT_R32_UINT, .target = rt);
         continue;
      }

      nir_def *color = build_background_op(
         &b, key->op[rt], rt, nr, source_msaa, layered,
         agx_chip_agx3_path(cache->dev->chip), clear_base);
      if (key->preserve_resolve_outside) {
         assert(rt == 0 && key->op[rt] == AGX_BG_LOAD);
         nir_def *xy = nir_u2u32(&b, nir_load_pixel_coord(&b));
         nir_def *rect = nir_load_preamble(&b, 4, 32, 8);
         nir_def *inside = nir_iand(&b,
            nir_ball(&b, nir_uge(&b, xy, nir_channels(&b, rect, 0x3))),
            nir_ball(&b, nir_ult(&b, xy, nir_channels(&b, rect, 0xc))));
         nir_def *saved = build_background_op(
            &b, AGX_BG_LOAD, 1, nr, false, layered, true, clear_base);
         color = nir_bcsel(&b, inside, color, saved);
      }

      nir_store_output(
         &b, color,
         nir_imm_int(&b, 0), .write_mask = BITFIELD_MASK(nr),
         .src_type = nir_type_uint32,
         .io_semantics.location = FRAG_RESULT_DATA0 + rt,
         .io_semantics.num_slots = 1);

      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + rt);
   }

   return agx_compile_bg_eot_shader(cache, b.shader, &compiler_key, &key->tib);
}

static struct agx_bg_eot_shader *
agx_build_end_of_tile_shader(struct agx_bg_eot_cache *cache,
                             struct agx_bg_eot_key *key)
{
   const bool g18g = cache->dev->chip == AGX_CHIP_G18G;
   const unsigned g18_mrt = cache->dev->chip == AGX_CHIP_G18G
                              ? agx_g18g_word_mrt_mask(&key->tib) |
                                   agx_g18g_raw32_mrt_mask(&key->tib) |
                                   agx_g18g_ms_mrt_mask(&key->tib) |
                                   agx_g18g_native_mixed_mask(&key->tib)
                                 : 0;
   /* msaa163: G18G's four-sample tile is resolved at end of tile into the
    * single-sample PBE target, the shape Apple's G18 4x render uses: its
    * end-of-tile code is the same program at 1x and 4x (byte-identical over
    * 0x1400 bytes) and its resolve descriptor is the plain 1x one (MSAA
    * notes s2.5). Persistent attachments and partial stores instead use the
    * native multisample IMGWBLK dimension, preserving individual samples. */
   const bool g18_msaa = agx_g18_msaa4_tib(cache->dev, &key->tib);
   const bool g18_resolve =
      g18_msaa && !key->multisample_store;
   if (g18g) {
      /* The compiler substitutes Apple's 78-byte native text for the G17G RT0
       * store, and it has that text for a U8NORM imageblock only, so the gate
       * is the block store's physical format rather than a list of logical
       * ones. Within U8NORM the text does not vary with the logical format:
       * the component count and the RGBA/BGRA byte order are properties of
       * the PBE descriptor that the Ioto program binds, not of the shader.
       * Apple's own builtin factory emits byte-identical U8NORM programs for
       * R8_UNORM and RGBA8 (see the selector notes above agx3_rt0_eot_kind),
       * and diffing Apple's M5 render state between an RGBA8 and an R32F
       * target moved no PPP record, only the program word and the PBE. sRGB
       * passes because the loop below rewrites it to U8NORM. R11G11B10 keeps
       * its RG11B10F physical here (the raw-word rewrite is G17P only) and
       * is refused, as is every format whose text has not been witnessed:
       * the compiler would fail those shaders anyway, and refusing up front
       * keeps the fallback explicit.
       */
      if (key->op[0] != AGX_EOT_STORE ||
          (key->tib.nr_samples != 1 && !g18_msaa) ||
          (key->tib.layered && cache->dev->chip != AGX_CHIP_G18G) ||
          key->tib.spilled[0] ||
          agx_tilebuffer_offset_B(&key->tib, 0) != 0 ||
          (!(cache->dev->chip == AGX_CHIP_G18G &&
             key->tib.logical_format[0] == PIPE_FORMAT_R11G11B10_FLOAT) &&
           !agx_g17g_rt0_eot_physical_format_supported(
              agx_tilebuffer_physical_format(&key->tib, 0))))
         return NULL;

      for (unsigned rt = 1; rt < ARRAY_SIZE(key->op); ++rt) {
         if (key->op[rt] != ((g18_mrt & BITFIELD_BIT(rt)) ? AGX_EOT_STORE
                                                       : AGX_BG_EOT_NONE))
            return NULL;
      }
   }

   unsigned m5_stores = 0;
   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt)
      m5_stores += key->op[rt] == AGX_EOT_STORE;
   const bool m5_layered_pair = agx_chip_g17_m5(cache->dev->chip) &&
      key->tib.layered && key->m5_store_and_resolve;
   if (agx_chip_g17_m5(cache->dev->chip) && m5_stores &&
       (m5_stores > 1 || key->op[0] != AGX_EOT_STORE || m5_layered_pair)) {
      if (agx_tilebuffer_spills(&key->tib))
         return NULL;
      uint8_t selectors[8];
      unsigned active = 0;
      for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
         if (key->op[rt] != AGX_EOT_STORE)
            continue;
         const unsigned selector =
            agx_mrt_eot_selector(cache->dev->chip, key->tib.logical_format[rt]);
         if (selector != 0x15 && selector != 0x35 && selector != 0x55 &&
             selector != 0x75 && selector != 0x95)
            return NULL;
         selectors[active++] = selector;
      }
      uint8_t text[AGX3_M5_MRT_EOT_MAX_LENGTH];
      size_t size;
      if (m5_layered_pair) {
         if (m5_stores != 1 || !key->multisample_store)
            return NULL;
         selectors[1] = selectors[0];
         m5_stores = 2;
      }
      const unsigned sample_mask = m5_layered_pair ? 1 :
         key->multisample_store ? (1u << m5_stores) - 1 : 0;
      if (!agx3_m5_mrt_eot_generate_sample_mask(m5_stores, selectors,
             sample_mask, key->tib.layered, text, sizeof(text), &size))
         return NULL;
      const struct agx_shader_part bin = {
         .binary = text,
         .info = {
            .stage = MESA_SHADER_COMPUTE,
            .binary_size = size,
            .main_size = size,
            /* Native ordinary register extent fits the qualified RT0
             * allocation; each per-target wire needs eight staging halves. */
            .nr_gprs = agx3_m5_mrt_eot_nr_gprs(m5_stores),
            .agx3_staging_known = true,
            .agx3_staging_halfs = agx3_m5_mrt_eot_staging_halfs(m5_stores),
         },
      };
      struct agx_bg_eot_shader *res = rzalloc(cache->ht, struct agx_bg_eot_shader);
      if (!res || !agx_g17p_upload_bg_eot(cache, &bin, res)) {
         ralloc_free(res);
         return NULL;
      }
      res->info = bin.info;
      return res;
   }

   const unsigned rgba8_mask = agx_g17p_rgba8_mrt_mask(&key->tib);
   const unsigned active_count = util_bitcount(rgba8_mask);
   if (agx_chip_native_agx3(cache->dev->chip) && rgba8_mask) {
      /* EOT partitions are physical offset/sample-stride quotients, not
       * fragment synchronization classes. All these compact offsets stay
       * in partition zero; native PBEs and offset arguments use active order. */
      for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
         if (key->op[rt] != ((rgba8_mask & BITFIELD_BIT(rt))
                              ? AGX_EOT_STORE : AGX_BG_EOT_NONE))
            return NULL;
      }
      struct agx_g17p_wide_mrt_eot native;
      if (!agx_g17p_wide_mrt_eot_get(active_count, &native))
         return NULL;
      struct agx_shader_part bin = {
         .binary = (void *)native.code,
         .info = {
            .stage = MESA_SHADER_COMPUTE,
            .binary_size = native.size,
            .main_size = native.size,
            .nr_gprs = 8,
            .agx3_staging_known = true,
            .agx3_staging_halfs = native.staging_halfs,
         },
      };
      struct agx_bg_eot_shader *res = rzalloc(cache->ht, struct agx_bg_eot_shader);
      if (!res || !agx_g17p_upload_bg_eot(cache, &bin, res)) {
         ralloc_free(res);
         return NULL;
      }
      res->info = bin.info;
      return res;
   }

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "agx_eot");

   /* A multisample tile can be written out two ways. The default is to
    * resolve it through an ordinary single-sample PBE target, which is what
    * every G17P path except VK_ATTACHMENT_STORE_OP_STORE wants and the only
    * shape the earlier native EOT substitution covered. Storing the surface
    * unresolved needs the multisample IMGWBLK dimension, and on G17P that is
    * a separate Apple builtin rather than a field the backend may set on the
    * 2D one -- so it is only selected when the caller asked for it and the
    * target really is multisample.
    */
   const bool g17p = agx_chip_native_agx3(cache->dev->chip);
   const bool native_eot = g17p || agx_chip_g17_m5(cache->dev->chip);
   if (key->store_and_resolve) {
      /* This bounded path has two memory outputs from one physical tile. */
      assert(g17p && key->multisample_store && key->tib.nr_samples > 1 &&
             !key->tib.layered && !key->tib.spilled[0] &&
             key->op[0] == AGX_EOT_STORE);
      for (unsigned rt = 1; rt < ARRAY_SIZE(key->op); ++rt)
         assert(key->op[rt] == AGX_BG_EOT_NONE);
   }
   const bool store_multisample =
      key->tib.nr_samples > 1 && !g18_resolve &&
      (!native_eot || key->multisample_store);

   /* AGX2 carries the sample in the store's coordinate vector, so its
    * multisample block store is a three-component GLSL_SAMPLER_DIM_MS one.
    * Apple's native G17P/G17G builtins do not: both the 2D and multisample
    * end-of-tile programs read exactly the same two grid coordinates and
    * differ only in the IMGWBLK dimension class the packer deposits. Keep
    * the two-coordinate 2D shape on these targets and let the compiler key select
    * which of the two Apple programs is meant, the same way every other
    * G17P end-of-tile selector is chosen. */
   enum glsl_sampler_dim dim = (store_multisample && !native_eot && !g18g)
                                  ? GLSL_SAMPLER_DIM_MS
                                  : GLSL_SAMPLER_DIM_2D;

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
      if (key->op[rt] == AGX_BG_EOT_NONE)
         continue;

      /* The end-of-tile shader is unsuitable to handle spilled render targets.
       * Skip them. If blits are needed with spilled render targets, other parts
       * of the driver need to implement them.
       */
      if (key->tib.spilled[rt])
         continue;

      assert(key->op[rt] == AGX_EOT_STORE);
      unsigned offset_B = agx_tilebuffer_offset_B(&key->tib, rt);

      nir_def *layer = nir_undef(&b, 1, 16);
      if (key->tib.layered)
         layer = nir_u2u16(&b, nir_load_layer_id(&b));

      bool explicit = false;
      nir_def *coords = layer;
      if (agx_chip_native_or_m5(cache->dev->chip) || g18g) {
         /* AGX3's EOT imgwblk has no implicit pixel-coordinate form.  Apple's
          * G17P compiler reads the hardware grid X/Y SRs explicitly.  Array
          * stores append the hardware layer ID; dropping it here made every
          * layered EOT store address layer zero even though the image
          * descriptor and tilebuffer were correctly marked as arrays. */
         nir_def *xy = nir_trim_vector(
            &b, nir_load_global_invocation_id(&b, 32), 2);
         coords = key->tib.layered
                     ? nir_vec3(&b, nir_channel(&b, xy, 0),
                                nir_channel(&b, xy, 1),
                                nir_u2u32(&b, nir_load_layer_id(&b)))
                     : xy;
         explicit = true;
      }

      enum pipe_format physical =
         agx_tilebuffer_physical_format(&key->tib, rt);
      if (key->tib.logical_format[rt] == PIPE_FORMAT_R11G11B10_FLOAT &&
          agx_rg11b10_tile_is_packed_float(cache->dev->chip)) {
         /* M5 reads the packed word as the packed-float tile format and
          * converts it to the R11G11B10 PBE (agx_rt0_pbe_format). */
         physical = (enum pipe_format)AIL_ISA_FORMAT_RG11B10F;
      } else if ((agx_chip_native_or_m5(cache->dev->chip) &&
                  (key->tib.logical_format[rt] == PIPE_FORMAT_R11G11B10_FLOAT ||
                   key->tib.logical_format[rt] == PIPE_FORMAT_R9G9B9E5_FLOAT)) ||
                 (g18g &&
                  key->tib.logical_format[rt] == PIPE_FORMAT_R11G11B10_FLOAT)) {
         /* G17P keeps packed-float tiles as one raw 32-bit word. The matching
          * PBE descriptor also uses R32_UINT to preserve those exact bits.
          * G17G stores RG11B10Float and RGB9E5Float tiles the same way: one
          * 32-bit lane (port/fbwit lstcol/lldcol W32 mask1). G18G packs
          * R11G11B10 explicitly into one raw word as well; texture views
          * retain the original decoding format. */
         physical = PIPE_FORMAT_R32_UINT;
      } else if ((agx_chip_native_or_m5(cache->dev->chip) || g18g) &&
          util_format_is_srgb(key->tib.logical_format[rt])) {
         /* G17P applies the sRGB transfer in the fragment's own packs
          * (fltnorm/normflt sRGB modes), and G18G lowers it explicitly before
          * ST_TILE; both store the resulting bytes in a
          * U8NORM imageblock.  EOT is a raw transfer
          * from that imageblock to the format-aware PBE descriptor; asking
          * IMGWBLK to consume SRGBA8 would perform a second conversion and has
          * no native builtin.  Match the physical format selected by the
          * fragment lowering, just as background clear/reload already do. */
         physical = (enum pipe_format)AIL_ISA_FORMAT_U8NORM;
      }

      nir_image_store_block_agx(
         &b, nir_imm_intN_t(&b, rt, 16), nir_imm_int(&b, offset_B), coords,
         .format = physical,
         .image_dim = dim, .image_array = key->tib.layered,
         .explicit_coord = explicit);

      if (key->store_and_resolve) {
         /* Resolving is non-destructive: first preserve the source samples,
          * then resolve the same imageblock through a different PBE. The
          * compiler key supplies each output's native dimension class. */
         nir_image_store_block_agx(
            &b, nir_imm_intN_t(&b, 1, 16), nir_imm_int(&b, offset_B), coords,
            .format = physical, .image_dim = GLSL_SAMPLER_DIM_2D,
            .image_array = false, .explicit_coord = true);
      }
   }

   /* Run 1139: the compacted key of a G18 single-sample mixed-format tile. */
   const bool g18_native_mixed = cache->dev->chip == AGX_CHIP_G18G &&
                                 agx_g18g_native_mixed_mask(&key->tib) != 0;
   struct agx_shader_key compiler_key = {
      .dev = agx_gather_device_key(cache->dev),
      .reserved_preamble = key->reserved_preamble,
      .no_preamble = agx_chip_native_or_m5(cache->dev->chip) || g18g,
      .m5_rt0_eot = agx_chip_g17_m5(cache->dev->chip),
      .m5_rt0_eot_store_and_resolve = key->m5_store_and_resolve,
      .g17g_rt0_eot = g18g,
      .g18_native_mixed = g18_native_mixed,
   };

   if (native_eot || g18g) {
      /* The store shape is a property of the logical format, not of the GPU:
       * it says how many imageblock words the target occupies and which raw
       * store the tilebuffer lowering leaves behind. Both generations pick
       * their native program from it. */
      agx_g17p_configure_rt0_eot_key(key->tib.logical_format[0], &compiler_key);
      compiler_key.g17p_rt0_eot_multisample = store_multisample;
   }

   if (agx_chip_native_agx3(cache->dev->chip) || g18g) {
      /* A mixed layout storing several targets uses the multi-target
       * builtin, selecting each target's IMGWBLK tail by its format. A
       * single stored target keeps its format's own program. */
      /* Counted whatever named the layout. A GL draw-buffer mask reaches the
       * same several-target store as a Vulkan mixed layout; the selectors
       * come from each target's own logical format either way. */
      unsigned stores = 0;
      for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt)
         stores += key->op[rt] == AGX_EOT_STORE;

      if (key->store_and_resolve) {
         assert(stores == 1);
         compiler_key.g17p_mrt_eot_count = 2;
         compiler_key.g17p_mrt_eot_resolve_mask = BITFIELD_BIT(1);
         compiler_key.g17p_mrt_eot_selectors[0] =
            compiler_key.g17p_mrt_eot_selectors[1] =
               agx_g17p_mrt_eot_selector(key->tib.logical_format[0]);
      } else if (stores > 1 &&
                 stores <= ARRAY_SIZE(compiler_key.g17p_mrt_eot_selectors)) {
         /* Native1130 (run 1139): Apple's mixed-format end of tile stores
          * each target with IMGWBLK f29 by its tile class (0 8-bit integer,
          * 1 16-bit, 2 32-bit, 3 attachment-encoded: 8-bit normalized,
          * RGB10A2 and R11G11B10F) in all 39 workloads. Each class has its
          * IMGWBLK selector; only R11G11B10F differs from the G17P selector
          * table, whose W32 raw-word choice (0x35) is f29 = 2. */
         static const uint8_t class_selector[] = {0x55, 0x15, 0x35, 0x75};
         for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
            if (key->op[rt] == AGX_EOT_STORE) {
               const enum pipe_format format = key->tib.logical_format[rt];
               const int class =
                  g18_native_mixed ? agx_g18g_native_mixed_eot_class(format)
                                   : -1;
               compiler_key.g17p_mrt_eot_selectors
                  [compiler_key.g17p_mrt_eot_count++] =
                  class >= 0 ? class_selector[class]
                             : agx_g17p_mrt_eot_selector(format);
            }
         }
      }
   }

   compiler_key.g17p_r11g11b10_mrt_eot =
      agx_chip_native_agx3(cache->dev->chip) &&
      agx_g17p_classify_mrt(&key->tib) == AGX_G17P_MRT_R11G11B10;

   return agx_compile_bg_eot_shader(cache, b.shader, &compiler_key, NULL);
}

static struct agx_bg_eot_shader *
agx_get_bg_eot_shader_locked(struct agx_bg_eot_cache *cache,
                             struct agx_bg_eot_key *key)
{
   struct hash_entry *ent = _mesa_hash_table_search(cache->ht, key);
   if (ent)
      return ent->data;

   struct agx_bg_eot_shader *ret = NULL;
   bool has_store = false;

   for (unsigned rt = 0; rt < ARRAY_SIZE(key->op); ++rt) {
      if (key->op[rt] == AGX_EOT_STORE) {
         has_store = true;
         ret = agx_build_end_of_tile_shader(cache, key);
         break;
      }
   }

   if (!ret && !has_store)
      ret = agx_build_background_shader(cache, key);

   if (!ret)
      return NULL;

   ret->key = *key;
   _mesa_hash_table_insert(cache->ht, &ret->key, ret);
   return ret;
}

/* Honeykrisp shares one cache between every command buffer of a device, and
 * command buffers record on any thread: the lookup, the build (which
 * allocates from cache->ht and uploads to cache->pool) and the insert all run
 * under the lock that agx_get_precompiled() takes for the same pool.
 */
struct agx_bg_eot_shader *
agx_get_bg_eot_shader(struct agx_bg_eot_cache *cache,
                      struct agx_bg_eot_key *key)
{
   simple_mtx_lock(&cache->lock);
   struct agx_bg_eot_shader *ret = agx_get_bg_eot_shader_locked(cache, key);
   simple_mtx_unlock(&cache->lock);
   return ret;
}

DERIVE_HASH_TABLE(agx_bg_eot_key)

void
agx_bg_eot_init(struct agx_bg_eot_cache *cache, struct agx_device *dev)
{
   agx_pool_init(&cache->pool, dev, "Internal programs",
                 AGX_BO_EXEC | AGX_BO_LOW_VA, true);
   simple_mtx_init(&cache->lock, mtx_plain);
   cache->ht = agx_bg_eot_key_table_create(NULL);
   cache->dev = dev;
}

void
agx_bg_eot_cleanup(struct agx_bg_eot_cache *cache)
{
   agx_pool_cleanup(&cache->pool);
   _mesa_hash_table_destroy(cache->ht, NULL);
   simple_mtx_destroy(&cache->lock);
   cache->ht = NULL;
   cache->dev = NULL;
}

static struct agx_precompiled_shader *
agx_get_precompiled_locked(struct agx_bg_eot_cache *cache, unsigned program)
{
   simple_mtx_assert_locked(&cache->lock);

   /* It is possible that, while waiting for the lock, another thread uploaded
    * the shader. Check for that so we don't double-upload.
    */
   if (cache->precomp[program])
      return cache->precomp[program];

   if (!cache->dev->libagx_programs)
      return NULL;

   /* Otherwise, we need to upload. */
   struct agx_precompiled_shader *p =
      ralloc(cache->ht, struct agx_precompiled_shader);

   const uint32_t *bin = cache->dev->libagx_programs[program];
   if (!bin) {
      mesa_loge("asahi: libagx program %u is not built for this chip", program);
      ralloc_free(p);
      return NULL;
   }
   const struct agx_precompiled_kernel_info *info = (void *)bin;
   const void *binary = (const uint8_t *)bin + sizeof(*info);

   p->info = *info;
   p->rodata_offset = 0;

   assert(info->main_offset == 0 || program != LIBAGX_HELPER);

   p->b.workgroup =
      agx_workgroup(info->workgroup_size[0], info->workgroup_size[1],
                    info->workgroup_size[2]);

   /* G18G's kernels (agx3d, compiled as G17G) sit in the same CODE container
    * as G17P's: the M5 compute profile takes the entry exactly as the API
    * compute path builds it (agx_g17p_append_compute). */
   const bool agx3_image = agx_chip_agx3_path(cache->dev->chip);

   if (agx3_image) {
      const bool m5 = agx_chip_g17_m5(cache->dev->chip);
      const size_t lead_size = m5
         ? (size_t)cache->dev->params.num_clusters_total * 0x300u + 0x40u
         : 0x340;
      const size_t main_span = MAX2(info->main_size, info->binary_size);
      const size_t block_size = agx3_code_block_size(main_span);
      /* Promoted constants live in the binary and are published as a uniform
       * range beside the kernel's arguments, the same way the legacy USC path
       * maps them with agx_usc_immediates(). They are carried in this image
       * so one allocation still holds the whole program. */
      const size_t rodata_size = (size_t)info->rodata.size_16 * 2;
      const size_t rodata_pad = rodata_size ? ALIGN_POT(rodata_size, 16) : 0;
      if (!block_size || block_size > SIZE_MAX - lead_size - rodata_pad ||
          info->main_offset > info->binary_size ||
          info->main_size > info->binary_size - info->main_offset ||
          (rodata_size &&
           (info->rodata.offset > info->binary_size ||
            rodata_size > info->binary_size - info->rodata.offset)) ||
          info->preamble_offset || info->preamble_size ||
          info->nr_preamble_gprs) {
         mesa_loge("asahi: libagx program %u has no G17P launch shape: "
                   "main %u@%u of %u, preamble %u@%u/%u, rodata %u",
                   program, info->main_size, info->main_offset,
                   info->binary_size, info->preamble_size, info->preamble_offset,
                   info->nr_preamble_gprs, info->rodata.size_16);
         ralloc_free(p);
         return NULL;
      }

      const size_t size = lead_size + block_size + rodata_pad;
      uint8_t *image = calloc(1, size);
      if (!image) {
         ralloc_free(p);
         return NULL;
      }
      const bool lead_ok = m5
         ? agx3_build_m5_container_lead(image, lead_size,
                                        cache->dev->params.num_clusters_total)
         : cache->dev->chip == AGX_CHIP_G18G
         /* G18G kernels keep the six-slot G17P lead they ran with (j873). */
         ? (agx3_build_container_lead(image, AGX_CHIP_G17P, 0), true)
         : (agx3_build_container_lead(image, cache->dev->chip, cache->dev->params.num_cores_per_cluster), true);
      const bool built = lead_ok && agx3_build_code_block(
         image + lead_size, block_size,
         (const uint8_t *)binary + info->main_offset, info->main_size,
         main_span, NULL, m5);
      if (rodata_size) {
         memcpy(image + lead_size + block_size,
                (const uint8_t *)binary + info->rodata.offset, rodata_size);
         p->rodata_offset = lead_size + block_size;
      }
      p->ptr = built
         ? agx_pool_upload_aligned_with_bo(&cache->pool, image, size, 128,
                                           &p->bo)
         : 0;
      p->entry_offset = lead_size + AGX3_CODE_BLOCK_ENTRY_OFFSET;
      free(image);
   } else {
      p->ptr = agx_pool_upload_aligned_with_bo(
         &cache->pool, binary, info->binary_size, 128, &p->bo);
      p->entry_offset = info->main_offset;
   }
   if (!p->ptr) {
      ralloc_free(p);
      return NULL;
   }

   if (agx3_image) {
      /* HAL200 consumes the CODE entry through a dispatch-local Ioto program;
       * none of the legacy USC launch words below are part of that ABI. */
      p_atomic_set(&cache->precomp[program], p);
      return p;
   }

   /* Bake launch */
   agx_pack(&p->b.launch, CDM_LAUNCH_WORD_0, cfg) {
      cfg.texture_state_register_count = 8 /* same encoding as 0 */;
      cfg.sampler_state_register_count = 1;
      cfg.uniform_register_count = info->push_count;
      cfg.preshader_register_count = info->nr_preamble_gprs;
   }

   /* Bake USC */
   struct agx_usc_builder b =
      agx_usc_builder(p->b.usc.data, sizeof(p->b.usc.data));

   agx_usc_immediates(&b, &info->rodata, p->ptr);

   if (info->uses_txf)
      agx_usc_push_packed(&b, SAMPLER, cache->dev->txf_sampler);

   agx_usc_shared(&b, info->local_size, info->imageblock_stride, 0);

   agx_usc_pack(&b, SHADER, cfg) {
      cfg.code = agx_usc_addr(cache->dev, p->ptr + info->main_offset);
      cfg.unk_2 = 3;
   }

   agx_usc_pack(&b, REGISTERS, cfg) {
      cfg.register_count = info->nr_gprs;
      cfg.spill_size = 0;
   }

   if (info->nr_preamble_gprs) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code = agx_usc_addr(cache->dev, p->ptr + info->preamble_offset);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   p->b.usc.size = b.head - p->b.usc.data;

   /* We must only write to the cache once we are done compiling, since other
    * threads may be reading the cache concurrently. Do this last.
    */
   p_atomic_set(&cache->precomp[program], p);
   return p;
}

struct agx_precompiled_shader *
agx_get_precompiled(struct agx_bg_eot_cache *cache, unsigned program)
{
   /* Shaders are immutable once written, so if we atomically read a non-NULL
    * shader, then we have a valid cached shader and are done.
    */
   struct agx_precompiled_shader *ret = p_atomic_read(cache->precomp + program);

   if (ret != NULL)
      return ret;

   /* Otherwise, take the lock and upload. */
   simple_mtx_lock(&cache->lock);
   ret = agx_get_precompiled_locked(cache, program);
   simple_mtx_unlock(&cache->lock);

   return ret;
}

uint64_t
agx_helper_program(struct agx_bg_eot_cache *cache)
{
   struct agx_precompiled_shader *pc =
      agx_get_precompiled(cache, LIBAGX_HELPER);

   return pc ? (pc->ptr | 1) : 0;
}

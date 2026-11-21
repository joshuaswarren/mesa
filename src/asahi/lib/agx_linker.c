/*
 * Copyright 2024 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "agx_linker.h"
#include "agx_bytes.h"
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include "util/macros.h"
#include "util/ralloc.h"
#include "agx_abi.h"
#include "agx3_code_container_core.h"
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx3_fltnorm_core.h"
#include "agx3_pixel_core.h"
#include "agx3_vary_core.h"
#include "agx_device.h"
#include "agx_pack.h"
#include "agx_scratch.h"

/*
 * When sample shading is used with a non-monolithic fragment shader, we
 * fast-link a program with the following structure:
 *
 *    Fragment prolog;
 *
 *    for (u16 sample_bit = 1; sample_bit < (1 << # of samples); ++sample_bit) {
 *       API fragment shader;
 *       Fragment epilog;
 *    }
 *
 * This means the prolog runs per-pixel but the fragment shader and epilog run
 * per-sample. To do this, we need to generate the loop on the fly. The
 * following binary sequences form the relevant loop.
 */

static_assert(AGX_ABI_FIN_SAMPLE_MASK == 2, "r1l known");

void
agx_fs_epilog_apply_device_policy(enum agx_chip chip,
                                  struct agx_fs_epilog_key *key)
{
   if (!key)
      return;

   const bool g17p = agx_chip_native_agx3(chip);
   /* G18 coverage also dirties color during depth-only or fully masked draws.
    * Preserve the previous tile value while executing the other side effects.
    */
   key->preserve_zero_mask = g17p || agx_chip_hal300_path(chip);
   key->sample_loop_mrt = chip == AGX_CHIP_G18G && key->nr_samples > 1;
   /* G18G: both AGX3 tile interfaces move packed bytes. The RGB transfer
    * function belongs around blending; alpha remains linear. R11G11B10 is
    * explicitly packed into one raw tile word, with a matching R32_UINT store
    * PBE. */
   key->lower_srgb = chip == AGX_CHIP_G18G;
   key->lower_r11g11b10 = chip == AGX_CHIP_G18G;
   key->native_epilog = chip == AGX_CHIP_G18G;
   if (chip == AGX_CHIP_G18G &&
       (key->nr_samples == 1 || key->nr_samples == 2 || key->nr_samples == 4)) {
      /* Match the render-pass layout. Native G18's 48-byte MS samples
       * retain 32x32 at 2x and 32x16 at 4x: 96 KiB and 2048 raster samples.
       * Single-sample MRT keeps its previous byte and raster limits. */
      key->max_bytes_per_tile = key->nr_samples == 1 ? 32768 : 98304;
      key->max_samples_per_tile = key->nr_samples == 1 ? 1024 : 2048;
   }
   key->m5_coverage = agx_chip_g17_m5(chip);
   key->g17p_tile_formats = g17p || agx_chip_g17_m5(chip);
}

bool
agx_fs_epilog_may_discard(const struct agx_fs_epilog_link_info *link,
                          bool alpha_to_coverage,
                          bool nontrivial_force_early)
{
   return !nontrivial_force_early &&
          (alpha_to_coverage || (link && (link->write_z || link->write_s)));
}

/* clang-format off */
static const uint8_t sample_loop_header[] = {
   /* mov_imm r0l, 0x0, 0b0 */
   0x62, 0x00, 0x00, 0x00,

   /* mov_imm r1l, 0x0, 0b0 */
   0x62, 0x04, 0x01, 0x00,
};

#define STOP                                                                   \
   /* stop */                                                                  \
   0x88, 0x00,                                                                 \
                                                                               \
   /* trap */                                                                  \
   0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00,                             \
   0x08, 0x00, 0x08, 0x00, 0x08, 0x00, 0x08, 0x00,

static const uint8_t stop[] = {STOP};

static const uint8_t sample_loop_footer[] = {
   /* iadd r1l, 0, r1l, lsl 1 */
   0x0e, 0x04, 0x00, 0x20, 0x84, 0x00, 0x00, 0x00,

   /* while_icmp r0l, ult, r1h, 0, 1 */
   0x52, 0x2c, 0x42, 0x00, 0x00, 0x00,

   /* jmp_exec_any */
   0x00, 0xc0, 0x00, 0x00, 0x00, 0x00,

   /* pop_exec r0l, 1 */
   0x52, 0x0e, 0x00, 0x00, 0x00, 0x00,

   STOP
};

/* Offset in sample_loop_footer to the jmp_exec_any's target */
#define SAMPLE_LOOP_FOOTER_JMP_PATCH_OFFS (16)

/* Offset of the jmp_exec_any, for calculating the PC offsets */
#define SAMPLE_LOOP_FOOTER_JMP_OFFS (14)

/* Offset in sample_loop_footer to the while_icmp's sample count immediate. Bit
 * position in the byte given by the shift.
 */
#define SAMPLE_LOOP_FOOTER_COUNT_PATCH_OFFS (11)
#define SAMPLE_LOOP_FOOTER_COUNT_SHIFT (4)
/* clang-format on */

/*
 * Apple's AGX3 code container.
 *
 * On G17P the LDSHDR record names an entry point INSIDE a container, not the
 * base of an executable allocation, and the shader loader reads container
 * metadata at fixed negative offsets from that entry.  The m1n1 add3 witness
 * (proxyclient/experiments/g17p_add3_code.py) lays it out as:
 *
 *   +0x000  dword 0x340                 size of the leading block
 *   +0x040  0x0006 filler .. +0x340
 *   +0x100  helper record table
 *   +0x200  helper record table (identical)
 *   +0x340  dword block_size            == entry - 0x80, the CODE block size
 *   +0x380  constant program (0x40)     == entry - 0x40
 *   +0x3C0  main program                <- LDSHDR points here
 *
 * block_size = 0x40 (its own header) + 0x40 (constant program) +
 * align64(code).  For the witness's 56-byte main program that is
 * 0x40 + 0x40 + 0x40 = 0xC0, which is the dword the capture actually
 * contains, and the container ends at 0x340 + 0xC0 = 0x400 -- exactly the
 * 1 KiB closure the witness proved on hardware.  So this rule reproduces the
 * witness byte for byte rather than merely resembling it.
 *
 * Honeykrisp's compute path guarantees no preamble, so the constant program
 * is the witness's minimal terminating one.
 */
#define AGX3_CONTAINER_LEAD        0x340u
#define AGX3_CONTAINER_ENTRY       0x3C0u
#define AGX3_CONTAINER_BLOCK_HDR   0x40u
#define AGX3_CONTAINER_CONST_PROG  0x40u

/* "0f 00 54 <opcode> 00 x6 06 00 06 00 06 00" per record, then two terminals.
 * The opcodes count down by 0x10 from 0x10 * slots, so G17P's six slots are
 * 0x60..0x10 and the ten-slot table of G15 and G16 parts is 0xa0..0x10. */
static const uint8_t agx3_helper_terminal[16] = {
   0xf7, 0x03, 0xaa, 0x00, 0x8f, 0x02, 0x54, 0x01,
   0x06, 0x00, 0x06, 0x00, 0x06, 0x00, 0x06, 0x00,
};

static void
agx3_fill_helper_table(uint8_t *image, size_t base, unsigned slots)
{
   const uint8_t *terminal = agx3_helper_terminal;

   assert(slots <= 14 && "both tables fit below the block-size dword");
   for (unsigned i = 0; i < slots; ++i) {
      uint8_t *record = image + base + i * 0x10;

      memset(record, 0, 0x10);
      record[0] = 0x0f;
      record[2] = 0x54;
      record[3] = (slots - i) * 0x10;
      for (unsigned h = 10; h < 16; h += 2) {
         record[h] = 0x06;
         record[h + 1] = 0x00;
      }
   }
   memcpy(image + base + slots * 0x10, terminal, 0x10);
   memcpy(image + base + (slots + 1) * 0x10, terminal, 0x10);
}

/* Store the helper tables' terminal record (a return) in every 16-byte slot
 * of [from, to), so a helper call that lands on any of them returns rather
 * than running whatever bytes the image holds there. */
static void
agx3_fill_helper_terminals(uint8_t *image, size_t from, size_t to)
{
   for (size_t offset = from; offset + 0x10 <= to; offset += 0x10)
      memcpy(image + offset, agx3_helper_terminal, 0x10);
}

/* Total container bytes for `code_size` bytes of executable. */
static size_t
agx3_container_lead_size(const struct agx_device *dev)
{
   /* G18G's lead holds four agx3d helper tables. */
   if (dev->chip == AGX_CHIP_G18G)
      return AGX3_G18G_CONTAINER_LEAD_SIZE;

   return agx_chip_g17_m5(dev->chip)
      ? (size_t)dev->params.num_clusters_total * 0x300u + 0x40u
      : AGX3_CONTAINER_LEAD;
}

static size_t
agx3_container_size(const struct agx_device *dev, size_t code_size)
{
   return agx3_container_lead_size(dev) + AGX3_CONTAINER_BLOCK_HDR +
          AGX3_CONTAINER_CONST_PROG + ALIGN_POT(code_size, 64);
}

/* The container's leading block: dword 0x340, 0x0006 filler, two helper
 * tables. Identical in both captured Apple code images, so it is a fixed
 * prologue rather than anything workload-specific. */
void
agx3_build_container_lead(uint8_t *image, enum agx_chip chip,
                          unsigned cores_per_cluster)
{
   const unsigned slots = agx_chip_usc_helper_table_slots(chip, cores_per_cluster);

   memset(image, 0, AGX3_CONTAINER_LEAD);
   agx_store_le32(image, AGX3_CONTAINER_LEAD);
   for (size_t offset = 0x40; offset < AGX3_CONTAINER_LEAD; offset += 2) {
      image[offset] = 0x06;
      image[offset + 1] = 0x00;
   }
   agx3_fill_helper_table(image, 0x100, slots);
   agx3_fill_helper_table(image, 0x200, slots);
}

/* The USC window's base page: the container lead, then zeroes, so the zero
 * dword at +0x340 ends the page's container chain.  Where
 * agx_chip_usc_base_page_fill_returns() holds, every slot after the two
 * ten-slot tables (from +0x2c0) and after that dword (from +0x350) to the
 * end of the page returns, so a helper call that misses the tables stays in
 * the page.  Only the base page gets this; per-shader containers keep the
 * plain lead. */
void
agx3_build_usc_base_page(uint8_t *image, size_t size, enum agx_chip chip,
                         unsigned cores_per_cluster, unsigned num_clusters)
{
   assert(size >= AGX3_CONTAINER_LEAD + 0x10);

   memset(image, 0, size);
   agx3_build_container_lead(image, chip, cores_per_cluster);

   if (agx_chip_usc_base_page_fill_returns_for(chip, num_clusters)) {
      const unsigned slots =
         agx_chip_usc_helper_table_slots(chip, cores_per_cluster);
      agx3_fill_helper_terminals(image, 0x200 + (slots + 2) * 0x10,
                                 AGX3_CONTAINER_LEAD);
      agx3_fill_helper_terminals(image, AGX3_CONTAINER_LEAD + 0x10, size);
   }
}

static bool
agx3_build_container(const struct agx_device *dev, uint8_t *image,
                     size_t code_size, bool fragment, bool sampled_fragment)
{
   const bool m5 = agx_chip_g17_m5(dev->chip);
   const size_t lead_size = agx3_container_lead_size(dev);
   memset(image, 0, agx3_container_size(dev, code_size));
   if (dev->chip == AGX_CHIP_G18G) {
      if (!agx3_build_g18g_container_lead(image, lead_size))
         return false;
   } else if (m5) {
      if (!agx3_build_m5_container_lead(image, lead_size,
                                       dev->params.num_clusters_total))
         return false;
   } else {
      agx3_build_container_lead(image, dev->chip, dev->params.num_cores_per_cluster);
   }
   agx_store_le32(image + lead_size,
                 AGX3_CONTAINER_BLOCK_HDR + AGX3_CONTAINER_CONST_PROG +
                    ALIGN_POT(code_size, 64));
   uint8_t *constant = image + lead_size + AGX3_CONTAINER_BLOCK_HDR;
   if (dev->chip == AGX_CHIP_G18G) {
      agx3_build_container_stop_program_agx3d(constant);
      return true;
   }
   agx3_build_container_constant_program(constant, fragment || m5,
                                         sampled_fragment && !m5);
   return true;
}

static bool
agx3_uses_code_container(const struct agx_device *dev)
{
   return agx_chip_agx3_path(dev->chip);
}

/* Give a relocatable vertex program's varyings the indices of the bound
 * fragment shader's layout (agx_unlinked_uvs_layout::relocatable): re-encode
 * each listed UVSW in the linked copy of the main program. */
static bool
agx3_relocate_main_uvs(const struct agx_device *dev,
                       const struct agx_shader_part *main, uint8_t *code,
                       size_t size, const uint8_t *uvs_map)
{
   const struct agx_shader_info *info = &main->info;
   const size_t bytes = (size_t)info->agx3_uvs_reloc_count * AGX3_UVS_RELOC_SIZE;
   if (!uvs_map || !agx_chip_native_agx3(dev->chip) || !main->binary ||
       info->agx3_uvs_relocs_offset > info->binary_size ||
       info->binary_size - info->agx3_uvs_relocs_offset < bytes) {
      fprintf(stderr, "agx_fast_link: vertex program needs its varying "
                      "layout to link (%u relocations)\n",
              info->agx3_uvs_reloc_count);
      return false;
   }

   if (!agx3_vary_relocate_uvsw(AGX_ISA_ENCODING_AGX3B, code, size,
                                (const uint8_t *)main->binary +
                                   info->agx3_uvs_relocs_offset,
                                info->agx3_uvs_reloc_count, uvs_map)) {
      fprintf(stderr, "agx_fast_link: varying writes do not match their "
                      "relocation list\n");
      return false;
   }

   return true;
}

bool
agx_fast_link(struct agx_linked_shader *linked, struct agx_device *dev,
              bool fragment, struct agx_shader_part *main,
              struct agx_shader_part *prolog, struct agx_shader_part *epilog,
              unsigned nr_samples_shaded, const uint8_t *uvs_map)
{
   /* These literal loop fragments are AGX2 machine code. G17P exposes only
    * one sample and its Vulkan path folds sample shading before split-FS
    * lowering; fail closed if a caller ever bypasses that canonicalization. */
   if (agx_chip_agx3_path(dev->chip) &&
       nr_samples_shaded) {
      fprintf(stderr,
              "agx_fast_link: external sample loop is not AGX3 machine code\n");
      return false;
   }

   uint8_t fused_colour[AGX3_COLOUR_LINK_SIZE];
   const bool colour_fused = agx_chip_native_agx3(dev->chip) && fragment &&
      !prolog && !nr_samples_shaded && main && epilog &&
      agx3_fuse_colour_parts(&main->info.agx3_colour_link,
                              &epilog->info.agx3_colour_link, fused_colour);
   /* A native fragment enters its pixel transaction once across all linked
    * parts.  A sample-mask prolog already opens it, so retaining the main
    * shader's second entry resets the coverage state immediately before its
    * forced-early wait.  Keep the first gate and remove every later one.  A
    * gate at offset zero is outside all of that part's branches; otherwise
    * require the compiler's proof that no relative branch spans the bytes. */
   struct agx_shader_part *parts[] = {prolog, main, epilog};
   bool drop_entry[ARRAY_SIZE(parts)] = {false};
   /* A later entry that a relative branch spans (the empty-EXEC skip over a
    * testing coverage update can) is overwritten with NOPs instead: the
    * part keeps its size, so every branch offset stays valid. */
   bool nop_entry[ARRAY_SIZE(parts)] = {false};
   bool entry_seen = false;
   int kept_entry = -1;
   if (agx_chip_agx3_path(dev->chip) &&
       fragment && !nr_samples_shaded && !colour_fused) {
      for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i) {
         struct agx_shader_part *part = parts[i];
         if (!part || !part->info.agx3_colour_link.entry_emitted)
            continue;

         const uint32_t off = part->info.agx3_colour_link.entry_offset;
         if (entry_seen) {
            if (off != 0 && !part->info.agx3_colour_link.entry_unspanned) {
               /* The NOP fill below is agx3b/agx3c bytes. G18G (agx3d)
                * refuses a spanned later entry, as j873 did. */
               if (dev->chip == AGX_CHIP_G18G)
                  return false;
               nop_entry[i] = true;
            } else {
               drop_entry[i] = true;
            }
         } else {
            entry_seen = true;
            kept_entry = i;
         }
      }

      for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i) {
         if (!drop_entry[i] && !nop_entry[i])
            continue;
         struct agx_shader_part *part = parts[i];
         const uint32_t off = part->info.agx3_colour_link.entry_offset;
         uint8_t gate[AGX3_PIXEL_MAX_LENGTH], gate_size;
         if (dev->chip == AGX_CHIP_G18G) {
            /* Native G18 pixwait512 (native330). The compiler relocated
             * entry_offset with the instruction-boundary transcode map. */
            static const uint8_t g18_gate[] = {0x07,0x30,0x00,0x05,0x00,0x00};
            memcpy(gate, g18_gate, sizeof(g18_gate));
            gate_size = sizeof(g18_gate);
         }
         if ((off & 1) || off > part->info.main_size ||
             part->info.main_size - off < AGX3_PIXEL_MAX_LENGTH ||
             (dev->chip != AGX_CHIP_G18G &&
              !agx3_pixel_colour_entry(agx_chip_g17_m5(dev->chip)
                                          ? AGX_ISA_ENCODING_AGX3C
                                          : AGX_ISA_ENCODING_AGX3B,
                                       gate, &gate_size)) ||
             gate_size != AGX3_PIXEL_MAX_LENGTH ||
             memcmp((const uint8_t *)part->binary + part->info.main_offset +
                       off,
                    gate, gate_size))
         {
            fprintf(stderr, "agx_fast_link: part %u entry mismatch at %u of %u bytes\n",
                    i, off, part->info.main_size);
            return false;
         }
      }
   }
   size_t size = 0;
   unsigned nr_gprs = 0, scratch_size = 0, agx3_spectpr_scratch_size = 0;
   unsigned staging_halfs = 0;
   unsigned colour_classes = 0;
   bool staging_known = true;
   bool reads_tib = false, writes_sample_mask = false,
        disable_tri_merging = false, tag_write_disable = true;
   bool sampled_fragment = false;

   if (nr_samples_shaded) {
      size += sizeof(sample_loop_header);

      if (nr_samples_shaded > 1)
         size += sizeof(sample_loop_footer);
      else
         size += sizeof(stop);
   }

   for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i) {
      struct agx_shader_part *part = parts[i];
      if (!part)
         continue;

      size += part->info.main_size;

      nr_gprs = MAX2(nr_gprs, part->info.nr_gprs);
      staging_halfs = MAX2(staging_halfs, part->info.agx3_staging_halfs);
      staging_known &= part->info.agx3_staging_known;
      scratch_size = MAX2(scratch_size, part->info.scratch_size);
      agx3_spectpr_scratch_size =
         MAX2(agx3_spectpr_scratch_size,
              part->info.agx3_spectpr_scratch_size);
      reads_tib |= part->info.reads_tib;
      writes_sample_mask |= part->info.writes_sample_mask;
      disable_tri_merging |= part->info.disable_tri_merging;
      linked->uses_base_param |= part->info.uses_base_param;
      linked->uses_txf |= part->info.uses_txf;
      sampled_fragment |= part->info.g17p_sampled_fragment;
      tag_write_disable &= part->info.tag_write_disable;
      if (part->info.agx3_colour_link.colour_classes &
          ~BITFIELD_MASK(AGX3_G17P_MAX_COLOUR_PARTITIONS))
         return false;
      colour_classes |= part->info.agx3_colour_link.colour_classes;
   }

   if (colour_fused)
      size = sizeof(fused_colour);
   else
      for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i)
         size -= drop_entry[i] ? AGX3_PIXEL_MAX_LENGTH : 0;

   assert(size > 0 && "must stop");

   /* The staging accounting is per instruction, so a linked software vertex
    * shader (a compute-stage main behind the vertex prolog) is described the
    * same way as a hardware vertex or fragment program. */
   if (agx_chip_agx3_path(dev->chip) &&
       (main->info.stage == MESA_SHADER_VERTEX ||
        main->info.stage == MESA_SHADER_FRAGMENT ||
        main->info.stage == MESA_SHADER_COMPUTE ||
        main->info.stage == MESA_SHADER_TASK ||
        main->info.stage == MESA_SHADER_MESH) &&
       staging_known && !linked->g17p_compiler_d_valid) {
      linked->g17p_compiler_d_valid = true;
      linked->g17p_compiler_d = staging_halfs * 2;
   }

   const bool container = agx3_uses_code_container(dev);
   const size_t alloc_size = container ? agx3_container_size(dev, size) : size;

   uint32_t code_offset = 0;
   linked->bo = dev->code_alloc
                   ? dev->code_alloc(dev, alloc_size, &code_offset)
                   : agx_bo_create(dev, alloc_size, 0,
                                   AGX_BO_EXEC | AGX_BO_LOW_VA,
                                   "Linked executable");
   linked->size = size;
   linked->entry_offset =
      code_offset + (container ? agx3_container_lead_size(dev) +
                                  AGX3_CONTAINER_BLOCK_HDR + AGX3_CONTAINER_CONST_PROG : 0);
   if (!linked->bo) {
      fprintf(stderr,
              "agx_fast_link: executable low-VA allocation of %zu bytes "
              "failed (shader base 0x%" PRIx64 ")\n",
              size, dev->shader_base);
      return false;
   }

   uint8_t *container_map = agx_bo_map(linked->bo);
   if (container_map)
      container_map += code_offset;
   uint8_t *linked_map =
      container_map ? container_map + (linked->entry_offset - code_offset)
                    : NULL;
   if (!linked_map) {
      fprintf(stderr,
              "agx_fast_link: executable BO mapping failed (size %zu, "
              "VA 0x%" PRIx64 ", handle %u)\n",
              size, linked->bo->va ? linked->bo->va->addr : 0,
              (unsigned)linked->bo->handle);
      agx_bo_unreference(dev, linked->bo);
      linked->bo = NULL;
      return false;
   }

   if (container && !agx3_build_container(dev, container_map, size, fragment,
                                          fragment && sampled_fragment))
      goto invalid_start_marker;

   size_t offset = 0;
   if (colour_fused) {
      memcpy(linked_map, fused_colour, sizeof(fused_colour));
      offset = sizeof(fused_colour);
   }

   /* Where each part starts in the linked program. */
   size_t part_offset[ARRAY_SIZE(parts)] = {0};

   /* FS prolog happens per-pixel, outside the sample loop */
   if (prolog) {
      part_offset[0] = offset;
      size_t sz = prolog->info.main_size;
      memcpy(linked_map + offset, prolog->binary, sz);
      if (agx_chip_native_agx3(dev->chip) && fragment &&
          !agx3_fltnorm_relocate_start(linked_map + offset, sz,
                                       prolog->info.agx3_fltnorm_start, offset))
         goto invalid_start_marker;
      offset += sz;
   }

   if (nr_samples_shaded) {
      memcpy(linked_map + offset, sample_loop_header,
             sizeof(sample_loop_header));
      offset += sizeof(sample_loop_header);
   }

   size_t sample_loop_begin = offset;

   /* Main shader and epilog happen in the sample loop, so start from i=1 */
   for (unsigned i = 1; !colour_fused && i < ARRAY_SIZE(parts); ++i) {
      struct agx_shader_part *part = parts[i];
      if (!part)
         continue;

      part_offset[i] = offset;
      size_t sz = part->info.main_size;
      const uint8_t *part_code = (const uint8_t *)part->binary + part->info.main_offset;
      if (drop_entry[i]) {
         const unsigned gate = part->info.agx3_colour_link.entry_offset;
         memcpy(linked_map + offset, part_code, gate);
         memcpy(linked_map + offset + gate, part_code + gate + AGX3_PIXEL_MAX_LENGTH,
                sz - gate - AGX3_PIXEL_MAX_LENGTH);
         sz -= AGX3_PIXEL_MAX_LENGTH;
      } else {
         memcpy(linked_map + offset, part_code, sz);
         if (part == main && part->info.agx3_uvs_reloc_count &&
             !agx3_relocate_main_uvs(dev, part, linked_map + offset, sz,
                                     uvs_map)) {
            agx_bo_unreference(dev, linked->bo);
            linked->bo = NULL;
            return false;
         }
         if (nop_entry[i]) {
            uint8_t *gate = linked_map + offset +
                            part->info.agx3_colour_link.entry_offset;
            for (unsigned h = 0; h < AGX3_PIXEL_MAX_LENGTH; h += 2) {
               gate[h] = 0x06;
               gate[h + 1] = 0x00;
            }
         }
      }
      /* agx3b/agx3c bytes; G18G (agx3d) keeps the epilog's own wait. */
      if (i == 2 && main && dev->chip != AGX_CHIP_G18G &&
          main->info.agx3_colour_link.rt_query_write_acquired) {
         const struct agx3_colour_link_info *link = &part->info.agx3_colour_link;
         if (link->rt_query_wait_count != 1 || link->rt_query_wait_unsupported)
            goto invalid_start_marker;
         uint32_t at = link->rt_query_wait_offset;
         if (drop_entry[i] && at > link->entry_offset) at -= AGX3_PIXEL_MAX_LENGTH;
         /* The main shader already holds the colour write class (pixwait 4
          * ahead of its first traversal), so the epilog's read+write wait
          * becomes a read wait. Native M5 colour fragments with a ray query
          * show the same pair: pixwait 4 before intersect_ray, pixwait 8
          * before lstcol. Encode both in the device's ISA. */
         const enum agx_isa_encoding enc = agx_chip_g17_m5(dev->chip)
                                              ? AGX_ISA_ENCODING_AGX3C
                                              : AGX_ISA_ENCODING_AGX3B;
         uint8_t before[AGX3_PIXEL_MAX_LENGTH], after[AGX3_PIXEL_MAX_LENGTH];
         uint8_t before_len, after_len;
         if (at > sz || sz - at < AGX3_PIXEL_MAX_LENGTH ||
             !agx3_pixel_wait(enc, 12, link->rt_query_wait_mask,
                             before, &before_len) ||
             !agx3_pixel_wait(enc, 8, link->rt_query_wait_mask,
                             after, &after_len) || before_len != after_len ||
             before_len != AGX3_PIXEL_MAX_LENGTH ||
             memcmp(linked_map + offset + at, before, before_len))
            goto invalid_start_marker;
         memcpy(linked_map + offset + at, after, after_len);
      }
      if (agx_chip_native_agx3(dev->chip) && fragment &&
          !agx3_fltnorm_relocate_start(linked_map + offset, sz,
                                       part->info.agx3_fltnorm_start, offset))
         goto invalid_start_marker;

      offset += sz;
   }

   if (nr_samples_shaded > 1) {
      assert(sample_loop_footer[SAMPLE_LOOP_FOOTER_COUNT_PATCH_OFFS] == 0);

      /* Make a stack copy of the footer so we can efficiently patch it */
      uint8_t footer[sizeof(sample_loop_footer)];
      memcpy(footer, sample_loop_footer, sizeof(footer));

      /* Patch in sample end */
      uint8_t end = (1u << nr_samples_shaded) - 1;
      footer[SAMPLE_LOOP_FOOTER_COUNT_PATCH_OFFS] =
         end << SAMPLE_LOOP_FOOTER_COUNT_SHIFT;

      /* Patch in the branch target */
      int32_t loop_size = offset - sample_loop_begin;
      int32_t branch_offs = -(SAMPLE_LOOP_FOOTER_JMP_OFFS + loop_size);
      int32_t *target = (int32_t *)(footer + SAMPLE_LOOP_FOOTER_JMP_PATCH_OFFS);
      *target = branch_offs;

      /* Copy in the patched footer */
      memcpy(linked_map + offset, footer, sizeof(footer));
      offset += sizeof(footer);
   } else if (nr_samples_shaded) {
      /* Just end after the first sample, no need to loop for a single sample */
      memcpy(linked_map + offset, stop, sizeof(stop));
      offset += sizeof(stop);
   }

   assert(offset == size);

   /* Native M5 fragments writing depth or stencil alone at several samples
    * (native G17G compiler zs_depth_{2,4,8}x, zs_stencil_{2,4}x, and
    * zs_depth_discard_4x after its kill) open with the f34 entry gate and
    * never release the depth/coverage class explicitly. Where a part ends
    * such a covered-sample loop, take that form for the program: mark its
    * entry gate and drop that release. If either instruction is not the plain
    * one recorded, keep both. */
   if (agx_chip_g17_m5(dev->chip) && fragment && kept_entry >= 0 &&
       !colour_fused) {
      const struct agx_shader_part *kept = parts[kept_entry];
      for (unsigned i = 0; i < ARRAY_SIZE(parts); ++i) {
         const struct agx_shader_part *part = parts[i];
         if (!part || !part->info.agx3_colour_link.zs_loop_release ||
             (int)i < kept_entry)
            continue;
         size_t release_at = part->info.agx3_colour_link.zs_loop_release_offset;
         if (drop_entry[i] &&
             release_at > part->info.agx3_colour_link.entry_offset)
            release_at -= AGX3_PIXEL_MAX_LENGTH;
         release_at += part_offset[i];
         const size_t gate_at =
            part_offset[kept_entry] + kept->info.agx3_colour_link.entry_offset;
         uint8_t gate[AGX3_PIXEL_MAX_LENGTH], unreleased[AGX3_PIXEL_MAX_LENGTH];
         uint8_t release[AGX3_PIXEL_MAX_LENGTH];
         uint8_t gate_len, unreleased_len, release_len;
         if (release_at > gate_at && release_at + AGX3_PIXEL_MAX_LENGTH <= size &&
             agx3_pixel_colour_entry(AGX_ISA_ENCODING_AGX3C, gate, &gate_len) &&
             agx3_pixel_colour_entry_zs_unreleased(AGX_ISA_ENCODING_AGX3C,
                                                   unreleased, &unreleased_len) &&
             agx3_pixel_release(AGX_ISA_ENCODING_AGX3C, 1, false, 0, release,
                                &release_len) &&
             gate_len == AGX3_PIXEL_MAX_LENGTH &&
             unreleased_len == AGX3_PIXEL_MAX_LENGTH &&
             release_len == AGX3_PIXEL_MAX_LENGTH &&
             !memcmp(linked_map + gate_at, gate, gate_len) &&
             !memcmp(linked_map + release_at, release, release_len)) {
            memcpy(linked_map + gate_at, unreleased, unreleased_len);
            for (unsigned h = 0; h < AGX3_PIXEL_MAX_LENGTH; h += 2) {
               linked_map[release_at + h] = 0x06;
               linked_map[release_at + h + 1] = 0x00;
            }
         }
         break;
      }
   }

   linked->scratch_size = scratch_size;
   linked->agx3_spectpr_scratch_size = agx3_spectpr_scratch_size;
   linked->nr_gprs = nr_gprs;
   linked->g17p_colour_classes = colour_classes;

   agx_bo_assert_range(linked->bo, code_offset, alloc_size);

   agx_pack(&linked->shader, USC_SHADER, cfg) {
      cfg.code = agx_usc_addr(dev, linked->bo->va->addr + linked->entry_offset);
      cfg.unk_2 = fragment ? 2 : 3;

      if (fragment)
         cfg.loads_varyings = linked->cf.nr_bindings > 0;
   }

   agx_pack(&linked->regs, USC_REGISTERS, cfg) {
      cfg.register_count = nr_gprs;
      cfg.unk_1 = fragment;
      cfg.spill_size = scratch_size ? agx_scratch_get_bucket(scratch_size) : 0;
      cfg.unk_4 = 1;
   }

   if (fragment) {
      agx_pack(&linked->fragment_props, USC_FRAGMENT_PROPERTIES, cfg) {
         cfg.early_z_testing = !writes_sample_mask;
         cfg.unk_2 = true;
         cfg.unk_3 = 0xf;
         cfg.unk_4 = 0x2;
         cfg.unk_5 = 0x0;
      }

      /* If we skip the fragment shader, we can optimize the pass type */
      if (tag_write_disable) {
         reads_tib = false;
         writes_sample_mask = false;
         linked->no_op = true;
      }

      agx_pack(&linked->fragment_control, FRAGMENT_CONTROL, cfg) {
         cfg.tag_write_disable = tag_write_disable;
         cfg.disable_tri_merging = disable_tri_merging;

         if (reads_tib && writes_sample_mask)
            cfg.pass_type = AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH;
         else if (reads_tib)
            cfg.pass_type = AGX_PASS_TYPE_TRANSLUCENT;
         else if (writes_sample_mask)
            cfg.pass_type = AGX_PASS_TYPE_PUNCH_THROUGH;
         else
            cfg.pass_type = AGX_PASS_TYPE_OPAQUE;
      }

      /* Merge the CF binding lists from the prolog to handle cull distance */
      memcpy(&linked->cf, &main->info.varyings.fs,
             sizeof(struct agx_varyings_fs));

      struct agx_varyings_fs *prolog_vary =
         prolog ? &prolog->info.varyings.fs : NULL;

      if (prolog_vary && prolog_vary->nr_bindings) {
         assert(!prolog_vary->reads_z);
         linked->cf.nr_cf = MAX2(linked->cf.nr_cf, prolog_vary->nr_cf);

         assert(linked->cf.nr_bindings + prolog_vary->nr_bindings <=
                   ARRAY_SIZE(linked->cf.bindings) &&
                "bounded by # of coeff registers");

         memcpy(linked->cf.bindings + linked->cf.nr_bindings,
                prolog_vary->bindings,
                sizeof(struct agx_cf_binding) * prolog_vary->nr_bindings);

         linked->cf.nr_bindings += prolog_vary->nr_bindings;
      }

      agx_pack(&linked->osel, OUTPUT_SELECT, cfg) {
         cfg.varyings = linked->cf.nr_bindings > 0;
         cfg.frag_coord_z = linked->cf.reads_z;
      }
   }

   /* The linked program is what runs; its parts are dumped at compile. */
   if ((agx_get_compiler_debug() & AGX_DBG_SHADERS) &&
       (dev->chip == AGX_CHIP_G17P || agx_chip_g17_m5(dev->chip) ||
        dev->chip == AGX_CHIP_G18G)) {
      fprintf(stderr, "AGX3 linked: fragment=%u size=%zu hex=", fragment, size);
      for (size_t i = 0; i < size; ++i)
         fprintf(stderr, "%02x", linked_map[i]);
      fprintf(stderr, "\n");
   }

   return true;

invalid_start_marker:
   fprintf(stderr, "agx_fast_link: invalid G17P part-start FLTNORM marker\n");
   agx_bo_unreference(dev, linked->bo);
   linked->bo = NULL;
   return false;
}

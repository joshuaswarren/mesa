/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_ioto_program_core.h"
#include "agx_bytes.h"

#include "agx_ioto_core.h"
#include "agx_ioto_load_core.h"


#include <string.h>
#include <stdlib.h>

static const uint8_t
   g17p_default_profile_prefix[AGX_IOTO_G17P_DEFAULT_PROFILE_PREFIX_SIZE] = {
      0x1c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14, 0x81, 0x11, 0x06,
      0x00, 0x00, 0x00, 0x00, 0x0c, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x9f, 0x11, 0x54, 0x00, 0x02, 0x00, 0x08, 0xa8, 0x10, 0x05, 0x1c, 0x80,
      0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0f, 0x12, 0x54, 0x00, 0x4c, 0x00,
};

static const uint8_t
   g17p_default_profile_suffix[AGX_IOTO_G17P_DEFAULT_PROFILE_SUFFIX_SIZE] = {
      0x03, 0x00, 0x07, 0x00, 0x02, 0x00, 0x00, 0x00, 0x60, 0x00,
};

enum {
   G17P_FRAGMENT_PIXREL_SIZE = 6,
   FRAGMENT_COLOUR_PARTITIONS = 3,
};

/* Fragment __GPU_METADATA_2 root field 1 is a release mask for pixel classes
 * which the shader does not consume.  Fresh applegpu_g17p one-target programs
 * carry 0x2f3, while both the two-RGBA8 and mixed RGBA16F+R8 native programs
 * carry 0x2c3: active partition 1's combined class 0x30 is no longer released
 * by the loader.  Partition 2 analogously owns class 0xc0.  A multi-target
 * fragment writing only some targets releases the classes of the others,
 * class 0x0c included: native fragments writing each subset of two to four
 * targets carry 0x2cf with only partition 1, 0x23f with only partition 2,
 * 0x233 for partitions 0 and 2, and 0x20f for partitions 1 and 2.  Coverage
 * shaders additionally keep class 0 acquired, clearing bit 0 of each form. */
static bool
fragment_pixrel_descriptor(unsigned colour_classes, bool writes_sample_mask,
                           uint16_t *out)
{
   if (!out || colour_classes == 0 ||
       colour_classes >= (1u << FRAGMENT_COLOUR_PARTITIONS))
      return false;

   uint16_t descriptor = UINT16_C(0x2ff);
   for (unsigned c = 0; c < FRAGMENT_COLOUR_PARTITIONS; ++c) {
      if (colour_classes & (1u << c))
         descriptor &= ~(UINT16_C(12) << (2 * c));
   }
   if (writes_sample_mask)
      descriptor &= ~UINT16_C(1);

   *out = descriptor;
   return true;
}

static bool
g17p_fragment_pixrel_compose(unsigned colour_classes,
                             bool writes_sample_mask,
                             uint8_t out[G17P_FRAGMENT_PIXREL_SIZE])
{
   uint16_t descriptor;
   if (!out || !fragment_pixrel_descriptor(colour_classes, writes_sample_mask,
                                          &descriptor))
      return false;

   const uint8_t staged[G17P_FRAGMENT_PIXREL_SIZE] = {
      0x07, 0x02, 0x54, descriptor, descriptor >> 8, 0x00,
   };
   memcpy(out, staged, sizeof(staged));
   return true;
}

/* Background programs do not inherit the ordinary fragment release mask.
 * Valid 20260904-194053 doorbell 1 BG and partial-BG Ioto both emit this
 * record at +0x1de. On G17P a background program's PIXREL value comes
 * from root field 1 of its __GPU_METADATA_2 section.
 *
 * The BG value is 0x200, versus ordinary FS 0x2f3. This form is for the
 * bounded internal clear/reload shaders compiled with ignore_tib_dependencies,
 * not arbitrary shaders that own pixel waits.
 */
static const uint8_t g17p_background_pixrel[] = {
   0x07, 0x02, 0x54, 0x00, 0x02, 0x00,
};

_Static_assert(AGX_IOTO_STATE_TAIL_TYPE0_SIZE +
                     sizeof(g17p_default_profile_prefix) +
                     sizeof(g17p_default_profile_suffix) +
                     AGX_IOTO_PROGRAM_TERMINATE_SIZE ==
                  AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE,
               "G17P no-resource Ioto size must remain 90 bytes");

_Static_assert(AGX_IOTO_STATE_TAIL_TYPE0_SIZE +
                     sizeof(g17p_default_profile_prefix) +
                     G17P_FRAGMENT_PIXREL_SIZE +
                     AGX_IOTO_PROGRAM_TERMINATE_SIZE ==
                  AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE,
               "G17P fragment no-resource Ioto size must remain 86 bytes");

static uint16_t
g17p_load_le16(const uint8_t *bytes)
{
   return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
}

static bool
g17p_ldshdr_has_shader_va(const uint8_t *ldshdr)
{
   const uint16_t word1 = agx_load_le16(ldshdr + 2);
   const uint16_t word2 = agx_load_le16(ldshdr + 4);
   const uint16_t word3 = agx_load_le16(ldshdr + 6);
   const uint16_t word4 = agx_load_le16(ldshdr + 8);
   return (word1 & UINT16_C(0xff80)) != 0 || word2 != 0 || word3 != 0 ||
          (word4 & UINT16_C(1)) != 0;
}

static bool
g17p_type0_tail_shape_valid(
   const struct agx_ioto_state_tail_output *packed_tail)
{
   if (!packed_tail || packed_tail->length != AGX_IOTO_STATE_TAIL_TYPE0_SIZE)
      return false;

   return g17p_ldshdr_has_shader_va(packed_tail->bytes +
                                    AGX_IOTO_SPECLM_SIZE);
}

static bool
g17p_type0_tail_valid(const struct agx_ioto_state_tail_output *packed_tail)
{
   if (!g17p_type0_tail_shape_valid(packed_tail))
      return false;

   static const enum agx_ioto_kind kinds[] = {
      AGX_IOTO_KIND_SPECLM_EP0,
      AGX_IOTO_KIND_LDSHDR_EP0,
      AGX_IOTO_KIND_SPECTPR_EP0,
   };
   static const size_t sizes[] = {
      AGX_IOTO_SPECLM_SIZE,
      AGX_IOTO_LDSHDR_SIZE,
      AGX_IOTO_SPECTPR_SIZE,
   };
   size_t offset = 0;
   for (size_t i = 0; i < sizeof(kinds) / sizeof(kinds[0]); ++i) {
      struct agx_ioto_record record;
      size_t consumed = 0;
      if (!agx_ioto_validate_kind(
             AGX_CHIP_G17P, kinds[i], packed_tail->bytes + offset,
             packed_tail->length - offset, &record, &consumed) ||
          consumed != sizes[i])
         return false;
      offset += consumed;
   }

   return offset == packed_tail->length;
}

static const uint8_t g17s_minimal_compute_resource_body[] = {
   /* First-use gather base at half4/6, then two-dword gather load to half20. */
   0x2c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
   0x3c, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
   /* Runtime-patched buffer(1) pointer at half24/26. */
   0xcc, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
   0xdc, 0x80, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00,
   /* Constructor-owned absolute UniformData base at half28/30. */
   0xec, 0xa0, 0x02, 0x00, 0x00, 0x00, 0x0c, 0x00,
   0xfc, 0x80, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00,
   /* gather(2), buffer(1) root uniform(1), absolute(3), base-sorted. */
   0xaf, 0x04, 0x03, 0x00, 0xa8, 0x0c, 0x00, 0x40,
   0xcf, 0x18, 0x03, 0x00, 0x88, 0x04, 0x00, 0xc0,
   0x01, 0x00, 0x00, 0x00, 0x00, 0x80,
   0xef, 0x1c, 0x03, 0x00, 0xc8, 0x0c, 0x00, 0x40,
};

static const uint8_t g17s_minimal_compute_copies[] = {
   /* The pointer loaded by loadImmediateUserBuffer's kind-3 Register zero is
    * copied into CProg dwords 2..3. */
   0x2b, 0x00, 0x01, 0x04,
   0x3b, 0x00, 0x01, 0x04,
   /* Gather half20/22 -> CProg half0/2 (first COPY waits on slot 1). */
   0x0b, 0x14, 0x09, 0x44,
   0x1b, 0x16, 0x09, 0x04,
   /* Root uniform half24 -> CProg half8. */
   0x4b, 0x18, 0x09, 0x04,
   /* Absolute result half28/30/32 -> CProg half10/12/14. */
   0x5b, 0x1c, 0x09, 0x04,
   0x6b, 0x1e, 0x09, 0x04,
   0x7b, 0x20, 0x09, 0x04,
};

static bool
agx_ioto_program_target_gpu0(enum agx_chip chip)
{
   return agx_chip_ioto_generation(chip) == AGX_IOTO_GENERATION_GPU0;
}

/* GPU0 EslBuilder appends the AGX3b LDDEV instruction directly rather than
 * through an AGXIotoInstruction_LDDEV object.  Its low eleven opcode bits and
 * fourteen-byte boundary are exact, but it deliberately remains outside the
 * public Ioto record-kind namespace. */
static bool
agx_ioto_program_gpu0_inline_lddev(enum agx_chip chip, const uint8_t *bytes,
                                   size_t available)
{
   if (!agx_ioto_program_target_gpu0(chip) ||
       available < AGX_IOTO_GPU0_INLINE_LDDEV_SIZE)
      return false;

   const uint16_t word = (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
   return (word & UINT16_C(0x07ff)) == UINT16_C(0x0067);
}

static bool
agx_ioto_straight_line_body_kind(enum agx_ioto_kind kind)
{
   switch (kind) {
   case AGX_IOTO_KIND_NOP_EP0:
   case AGX_IOTO_KIND_LDIMM_EP0:
   case AGX_IOTO_KIND_LDIMM_EP1:
   case AGX_IOTO_KIND_LDIMM_EP2:
   case AGX_IOTO_KIND_LDIMM_EP3:
   case AGX_IOTO_KIND_COPY_EP0:
   case AGX_IOTO_KIND_COPY_EP1:
   case AGX_IOTO_KIND_RLD_EP0:
   case AGX_IOTO_KIND_RLD_EP1:
   case AGX_IOTO_KIND_IADD_EP0:
   case AGX_IOTO_KIND_LDDEV_EP0:
   case AGX_IOTO_KIND_LDDEV_EP1:
   case AGX_IOTO_KIND_LDDEV_EP2:
   case AGX_IOTO_KIND_LDDEV_EP3:
   case AGX_IOTO_KIND_LDDEV_EP4:
   case AGX_IOTO_KIND_LDDEV_EP5:
      return true;
   default:
      return false;
   }
}

static bool
agx_ioto_straight_line_body_validate(enum agx_chip chip, const void *body,
                                     size_t body_size)
{
   if (body_size == 0)
      return true;
   if (!body)
      return false;

   const uint8_t *bytes = body;
   size_t offset = 0;
   while (offset < body_size) {
      if (agx_ioto_program_gpu0_inline_lddev(chip, bytes + offset,
                                             body_size - offset)) {
         offset += AGX_IOTO_GPU0_INLINE_LDDEV_SIZE;
         continue;
      }

      struct agx_ioto_record record;
      size_t consumed;
      if (!agx_ioto_decode_one(chip, bytes + offset, body_size - offset,
                               &record, &consumed) ||
          consumed == 0 || !agx_ioto_straight_line_body_kind(record.kind))
         return false;

      offset += consumed;
   }

   return offset == body_size;
}

static bool
agx_ioto_g17p_copy_body_validate(const void *body, size_t body_size)
{
   if (body_size == 0)
      return true;
   if (!body)
      return false;

   const uint8_t *bytes = body;
   size_t offset = 0;
   while (offset < body_size) {
      struct agx_ioto_record record;
      size_t consumed = 0;
      if (!agx_ioto_decode_one(AGX_CHIP_G17P, bytes + offset,
                               body_size - offset, &record, &consumed) ||
          consumed == 0 || (record.kind != AGX_IOTO_KIND_COPY_EP0 &&
                            record.kind != AGX_IOTO_KIND_COPY_EP1))
         return false;
      offset += consumed;
   }

   return offset == body_size;
}

static bool
agx_ioto_g17p_pre_copy_body_validate(const void *body, size_t body_size)
{
   /* The straight-line validator already checks every record and the exact
    * end boundary. A second decoder traversal adds no further constraints. */
   return agx_ioto_straight_line_body_validate(AGX_CHIP_G17P, body, body_size);
}

bool
agx_ioto_program_compose_no_pixrel(
   enum agx_chip chip, uint64_t program_va, const void *body, size_t body_size,
   const struct agx_ioto_state_tail *tail,
   const struct agx_ioto_profile_control *profile,
   const struct agx_ioto_preshader_desc *preshader, void *output,
   size_t capacity, size_t *written)
{
   if (!output || !written || !tail || !profile || !preshader ||
       !agx_ioto_target_supported(chip) || profile->chip != chip ||
       program_va > UINT64_C(0xffffffffffff) ||
       (program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       body_size > capacity ||
       !agx_ioto_straight_line_body_validate(chip, body, body_size))
      return false;

   struct agx_ioto_state_tail_output packed_tail;
   if (!agx_ioto_state_tail_pack(chip, tail, &packed_tail))
      return false;

   uint8_t packed_profile[AGX_IOTO_PROFILE_CONTROL_MAX_SIZE];
   if (!agx_ioto_profile_control_pack(profile, packed_profile))
      return false;

   struct agx_ioto_preshader_output packed_preshader;
   if (!agx_ioto_preshader_emit(chip, preshader, &packed_preshader))
      return false;

   const size_t profile_size = agx_ioto_profile_control_size(chip);
   if (profile_size == 0 || body_size > SIZE_MAX - packed_tail.length ||
       body_size + packed_tail.length > SIZE_MAX - profile_size ||
       body_size + packed_tail.length + profile_size >
          SIZE_MAX - packed_preshader.length)
      return false;

   const size_t total =
      body_size + packed_tail.length + profile_size + packed_preshader.length;
   if (capacity < total)
      return false;

   uint8_t *dst = output;
   if (body_size)
      memmove(dst, body, body_size);
   memcpy(dst + body_size, packed_profile, profile_size);
   memcpy(dst + body_size + profile_size, packed_tail.bytes,
          packed_tail.length);
   memcpy(dst + body_size + profile_size + packed_tail.length,
          packed_preshader.bytes, packed_preshader.length);
   *written = total;
   return true;
}

bool
agx_ioto_straight_line_program_compose(
   enum agx_chip chip, uint64_t program_va, const void *body, size_t body_size,
   const struct agx_ioto_state_tail *tail,
   const struct agx_ioto_profile_control *profile, void *output,
   size_t capacity, size_t *written)
{
   const struct agx_ioto_preshader_desc absent = {
      .external_va = AGX_IOTO_PRESHADER_NO_VA,
   };

   return agx_ioto_program_compose_no_pixrel(chip, program_va, body, body_size,
                                             tail, profile, &absent, output,
                                             capacity, written);
}

bool
agx_ioto_g17p_no_resource_program_compose_trusted(
   uint64_t program_va, const struct agx_ioto_state_tail_output *packed_tail,
   void *output, size_t capacity, size_t *written)
{
   if (!output || !written || !g17p_type0_tail_shape_valid(packed_tail) ||
       program_va == 0 || program_va > UINT64_C(0xffffffffffff) ||
       (program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       capacity < AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE)
      return false;

   uint8_t staged[AGX_IOTO_G17P_NO_RESOURCE_PROGRAM_SIZE];
   size_t offset = 0;
   memcpy(staged + offset, packed_tail->bytes, packed_tail->length);
   offset += packed_tail->length;
   memcpy(staged + offset, g17p_default_profile_prefix,
          sizeof(g17p_default_profile_prefix));
   offset += sizeof(g17p_default_profile_prefix);
   memcpy(staged + offset, g17p_default_profile_suffix,
          sizeof(g17p_default_profile_suffix));
   offset += sizeof(g17p_default_profile_suffix);
   memset(staged + offset, 0, AGX_IOTO_PROGRAM_TERMINATE_SIZE);
   staged[offset] = UINT8_C(0x0e);
   offset += AGX_IOTO_PROGRAM_TERMINATE_SIZE;

   if (offset != sizeof(staged))
      return false;

   memcpy(output, staged, sizeof(staged));
   *written = sizeof(staged);
   return true;
}

static bool
type0_tail_records_valid(enum agx_chip chip,
                         const struct agx_ioto_state_tail_output *packed_tail)
{
   if (!packed_tail || packed_tail->length != AGX_IOTO_STATE_TAIL_TYPE0_SIZE)
      return false;

   static const enum agx_ioto_kind kinds[] = {
      AGX_IOTO_KIND_SPECLM_EP0,
      AGX_IOTO_KIND_LDSHDR_EP0,
      AGX_IOTO_KIND_SPECTPR_EP0,
   };
   static const size_t sizes[] = {
      AGX_IOTO_SPECLM_SIZE,
      AGX_IOTO_LDSHDR_SIZE,
      AGX_IOTO_SPECTPR_SIZE,
   };
   size_t offset = 0;
   for (size_t i = 0; i < sizeof(kinds) / sizeof(kinds[0]); ++i) {
      struct agx_ioto_record record;
      size_t consumed = 0;
      if (!agx_ioto_validate_kind(
             chip, kinds[i], packed_tail->bytes + offset,
             packed_tail->length - offset, &record, &consumed) ||
          consumed != sizes[i])
         return false;
      offset += consumed;
   }

   return offset == packed_tail->length;
}

bool
agx_ioto_g17g_no_resource_program_compose(
   uint64_t program_va, const struct agx_ioto_state_tail_output *packed_tail,
   bool fragment, void *output, size_t capacity, size_t *written)
{
   const size_t size = fragment
      ? AGX_IOTO_G17G_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE
      : AGX_IOTO_G17G_VERTEX_NO_RESOURCE_PROGRAM_SIZE;
   if (!output || !written ||
       !type0_tail_records_valid(AGX_CHIP_G17G, packed_tail) ||
       program_va == 0 || program_va > UINT64_C(0x1000000000000) - size ||
       (program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 || capacity < size)
      return false;

   const uint8_t *ldshdr = packed_tail->bytes + AGX_IOTO_SPECLM_SIZE;
   if (!(g17p_load_le16(ldshdr) & UINT16_C(0xffc0)) &&
       !g17p_load_le16(ldshdr + 6) && !g17p_load_le16(ldshdr + 8))
      return false;

   uint8_t staged[AGX_IOTO_G17G_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE] = {0};
   memcpy(staged, packed_tail->bytes, packed_tail->length);
   size_t offset = packed_tail->length;
   /* The M5 vertex witness matches this independently recovered HAL300
    * loader exactly. The M5 fragment witness changes its third LDIMM's
    * immediate from8 to4; this is not the later fourth LDIMM. */
   if (!agx_ioto_g17s_minimal_profile_loader_pack(staged + offset,
                                                sizeof(staged) - offset))
      return false;
   if (fragment)
      staged[offset + 20] = 4;
   offset += AGX_IOTO_G17S_MINIMAL_PROFILE_LOADER_SIZE;
   if (fragment) {
      static const uint8_t pixrel[] = {0x9f, 0x17, 0x00, 0x04, 0x00, 0x00};
      memcpy(staged + offset, pixrel, sizeof(pixrel));
      offset += sizeof(pixrel);
   }
   staged[offset] = 0x0e;
   offset += AGX_IOTO_PROGRAM_TERMINATE_SIZE;
   if (offset != size)
      return false;

   memcpy(output, staged, size);
   *written = size;
   return true;
}

bool
agx_ioto_g17g_resource_program_compose_classes(
   uint64_t program_va, const void *body, size_t body_size,
   const struct agx_ioto_state_tail_output *packed_tail,
   enum agx_ioto_g17g_program_stage stage, unsigned colour_classes,
   void *output, size_t capacity, size_t *written)
{
   if (!output || !written || !colour_classes ||
       colour_classes >= (1u << FRAGMENT_COLOUR_PARTITIONS) ||
       (stage != AGX_IOTO_G17G_PROGRAM_VERTEX &&
        stage != AGX_IOTO_G17G_PROGRAM_FRAGMENT &&
        stage != AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT &&
        stage != AGX_IOTO_G17G_PROGRAM_EOT &&
        stage != AGX_IOTO_G17G_PROGRAM_BACKGROUND &&
        stage != AGX_IOTO_G17G_PROGRAM_COMPUTE))
      return false;

   size_t copy_offset;
   if (!agx_ioto_program_copy_suffix_offset(AGX_CHIP_G17G, body, body_size,
                                            &copy_offset))
      return false;

   uint8_t base[AGX_IOTO_G17G_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE];
   size_t base_size;
   if (!agx_ioto_g17g_no_resource_program_compose(
          program_va, packed_tail, stage != AGX_IOTO_G17G_PROGRAM_VERTEX,
          base, sizeof(base), &base_size))
      return false;

   const size_t prefix_size = AGX_IOTO_STATE_TAIL_TYPE0_SIZE +
                             AGX_IOTO_G17S_MINIMAL_PROFILE_LOADER_SIZE;
   const bool pixel_release = stage == AGX_IOTO_G17G_PROGRAM_FRAGMENT ||
                              stage == AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT ||
                              stage == AGX_IOTO_G17G_PROGRAM_BACKGROUND;
   if (stage == AGX_IOTO_G17G_PROGRAM_FRAGMENT ||
       stage == AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT) {
      /* PIXREL has identical bytes in AGX3c/d. Native G18 submitted
       * fragments use f14=0x203 for all three colour classes and 0x233
       * for classes 0 and 2 (capture308). These extend the one/two-target
       * 0x2f3/0x2c3 witnesses using the same class ownership as G17P.
       * M5 coverage fragments additionally retain bit 0, giving 0x2f2
       * instead of 0x2f3, independently of the colour-class fields. */
      uint16_t descriptor;
      if (!fragment_pixrel_descriptor(
             colour_classes,
             stage == AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT, &descriptor))
         return false;
      const uint16_t low = UINT16_C(7) | (descriptor << 3);
      const uint8_t pixrel[] = {low, low >> 8, 0, 4, 0, 0};
      memcpy(base + prefix_size, pixrel, sizeof(pixrel));
   }
   if (stage == AGX_IOTO_G17G_PROGRAM_BACKGROUND) {
      /* Native M5 main/partial background Ioto at +0x19e releases with
       * 071000040000, unlike ordinary fragment 9f1700040000. The native
       * preamble follows that record. Our no-preamble internal shaders
       * terminate there, using the CODE block's stop constant. */
      static const uint8_t background_pixrel[] = {7, 0x10, 0, 4, 0, 0};
      memcpy(base + prefix_size, background_pixrel, sizeof(background_pixrel));
   }
   const size_t suffix_size = pixel_release
                                ? base_size - prefix_size
                                : AGX_IOTO_PROGRAM_TERMINATE_SIZE;
   const size_t fixed_size = prefix_size + suffix_size;
   if (body_size > SIZE_MAX - fixed_size)
      return false;
   const size_t size = body_size + fixed_size;
   if (capacity < size || size > UINT64_C(0x1000000000000) ||
       program_va > UINT64_C(0x1000000000000) - size)
      return false;

   uint8_t *staged = malloc(size);
   if (!staged)
      return false;
   if (copy_offset)
      memcpy(staged, body, copy_offset);
   memcpy(staged + copy_offset, base, prefix_size);
   if (body_size != copy_offset)
      memcpy(staged + copy_offset + prefix_size,
             (const uint8_t *)body + copy_offset, body_size - copy_offset);
   memcpy(staged + body_size + prefix_size,
          base + base_size - suffix_size, suffix_size);
   memcpy(output, staged, size);
   free(staged);
   *written = size;
   return true;
}

bool
agx_ioto_g17p_fragment_no_resource_program_compose_trusted(
   uint64_t program_va, const struct agx_ioto_state_tail_output *packed_tail,
   unsigned colour_classes, void *output, size_t capacity, size_t *written)
{
   uint8_t pixrel[G17P_FRAGMENT_PIXREL_SIZE];
   if (!g17p_fragment_pixrel_compose(colour_classes, false, pixrel) ||
       !output || !written || !g17p_type0_tail_shape_valid(packed_tail) ||
       program_va == 0 || program_va > UINT64_C(0xffffffffffff) ||
       (program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       capacity < AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE)
      return false;

   uint8_t staged[AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE];
   size_t offset = 0;
   memcpy(staged + offset, packed_tail->bytes, packed_tail->length);
   offset += packed_tail->length;
   memcpy(staged + offset, g17p_default_profile_prefix,
          sizeof(g17p_default_profile_prefix));
   offset += sizeof(g17p_default_profile_prefix);
   memcpy(staged + offset, pixrel, sizeof(pixrel));
   offset += sizeof(pixrel);
   memset(staged + offset, 0, AGX_IOTO_PROGRAM_TERMINATE_SIZE);
   staged[offset] = UINT8_C(0x0e);
   offset += AGX_IOTO_PROGRAM_TERMINATE_SIZE;

   if (offset != sizeof(staged))
      return false;

   memcpy(output, staged, sizeof(staged));
   *written = sizeof(staged);
   return true;
}

static bool
g17p_resource_program_compose(
   uint64_t program_va, const void *pre_copy_body, size_t pre_copy_body_size,
   const void *copies, size_t copies_size,
   const struct agx_ioto_state_tail_output *packed_tail, void *output,
   size_t capacity, size_t *written, const uint8_t *terminal_protocol,
   size_t terminal_protocol_size, bool validate_body,
   const struct agx_ioto_preshader_desc *preshader)
{
   const struct agx_ioto_preshader_desc absent = {
      .external_va = AGX_IOTO_PRESHADER_NO_VA,
   };
   struct agx_ioto_preshader_output suffix;
   if (!agx_ioto_preshader_emit(AGX_CHIP_G17P,
                               preshader ? preshader : &absent, &suffix))
      return false;

   const bool tail_valid = validate_body
                              ? g17p_type0_tail_valid(packed_tail)
                              : g17p_type0_tail_shape_valid(packed_tail);
   if (!output || !written || !tail_valid ||
       program_va == 0 || program_va > UINT64_C(0xffffffffffff) ||
       (program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       (validate_body &&
        (!agx_ioto_g17p_pre_copy_body_validate(pre_copy_body,
                                               pre_copy_body_size) ||
         !agx_ioto_g17p_copy_body_validate(copies, copies_size))) ||
       (pre_copy_body_size && !pre_copy_body) ||
       (copies_size && !copies) ||
       pre_copy_body_size > SIZE_MAX - packed_tail->length ||
       pre_copy_body_size + packed_tail->length >
          SIZE_MAX - sizeof(g17p_default_profile_prefix) ||
       pre_copy_body_size + packed_tail->length +
             sizeof(g17p_default_profile_prefix) >
          SIZE_MAX - copies_size ||
       pre_copy_body_size + packed_tail->length +
             sizeof(g17p_default_profile_prefix) + copies_size >
          SIZE_MAX - terminal_protocol_size ||
       pre_copy_body_size + packed_tail->length +
             sizeof(g17p_default_profile_prefix) + copies_size +
             terminal_protocol_size >
           SIZE_MAX - suffix.length)
      return false;

   const size_t total = pre_copy_body_size + packed_tail->length +
                        sizeof(g17p_default_profile_prefix) + copies_size +
                        terminal_protocol_size +
                         suffix.length;
   if (capacity < total)
      return false;

   uint8_t *dst = output;
   size_t offset = 0;
#define APPEND_SPAN(span, span_size)                                           \
   do {                                                                        \
      if ((span_size) != 0)                                                    \
         memmove(dst + offset, (span), (span_size));                           \
      offset += (span_size);                                                   \
   } while (0)
   APPEND_SPAN(pre_copy_body, pre_copy_body_size);
   APPEND_SPAN(packed_tail->bytes, packed_tail->length);
   APPEND_SPAN(g17p_default_profile_prefix,
               sizeof(g17p_default_profile_prefix));
   APPEND_SPAN(copies, copies_size);
   APPEND_SPAN(terminal_protocol, terminal_protocol_size);
   APPEND_SPAN(suffix.bytes, suffix.length);
#undef APPEND_SPAN
   if (offset != total)
      return false;

   *written = total;
   return true;
}

bool
agx_ioto_g17p_program_compose_with_preshader(
   uint64_t program_va, const void *pre_copy_body, size_t pre_copy_body_size,
   const void *copies, size_t copies_size,
   const struct agx_ioto_state_tail_output *packed_tail,
   enum agx_ioto_g17p_program_stage stage, unsigned colour_classes,
   const struct agx_ioto_preshader_desc *preshader, void *output,
   size_t capacity, size_t *written)
{
   uint8_t pixrel[G17P_FRAGMENT_PIXREL_SIZE];
   const uint8_t *protocol = g17p_default_profile_suffix;
   size_t protocol_size = sizeof(g17p_default_profile_suffix);
   switch (stage) {
   case AGX_IOTO_G17P_PROGRAM_VERTEX_COMPUTE:
      if (colour_classes)
         return false;
      break;
   case AGX_IOTO_G17P_PROGRAM_FRAGMENT:
   case AGX_IOTO_G17P_PROGRAM_COVERAGE_FRAGMENT:
      if (!g17p_fragment_pixrel_compose(
             colour_classes,
             stage == AGX_IOTO_G17P_PROGRAM_COVERAGE_FRAGMENT, pixrel))
         return false;
      protocol = pixrel;
      protocol_size = sizeof(pixrel);
      break;
   default:
      return false;
   }

   return g17p_resource_program_compose(
      program_va, pre_copy_body, pre_copy_body_size, copies, copies_size,
      packed_tail, output, capacity, written, protocol, protocol_size, true,
      preshader);
}

bool
agx_ioto_g17p_resource_program_compose_trusted(
   uint64_t program_va, const void *pre_copy_body, size_t pre_copy_body_size,
   const void *copies, size_t copies_size,
   const struct agx_ioto_state_tail_output *packed_tail, void *output,
   size_t capacity, size_t *written)
{
   return g17p_resource_program_compose(
      program_va, pre_copy_body, pre_copy_body_size, copies, copies_size,
      packed_tail, output, capacity, written, g17p_default_profile_suffix,
      sizeof(g17p_default_profile_suffix), false, NULL);
}

bool
agx_ioto_g17p_fragment_resource_program_compose_trusted(
   uint64_t program_va, const void *pre_copy_body, size_t pre_copy_body_size,
   const void *copies, size_t copies_size,
   const struct agx_ioto_state_tail_output *packed_tail,
   unsigned colour_classes, void *output, size_t capacity, size_t *written)
{
   uint8_t pixrel[G17P_FRAGMENT_PIXREL_SIZE];
   if (!g17p_fragment_pixrel_compose(colour_classes, false, pixrel))
      return false;

   return g17p_resource_program_compose(
      program_va, pre_copy_body, pre_copy_body_size, copies, copies_size,
      packed_tail, output, capacity, written, pixrel, sizeof(pixrel), false, NULL);
}

bool
agx_ioto_g17p_coverage_fragment_program_compose_trusted(
   uint64_t program_va, const void *pre_copy_body, size_t pre_copy_body_size,
   const void *copies, size_t copies_size,
   const struct agx_ioto_state_tail_output *packed_tail,
   unsigned colour_classes, void *output, size_t capacity, size_t *written)
{
   uint8_t pixrel[G17P_FRAGMENT_PIXREL_SIZE];
   if (!g17p_fragment_pixrel_compose(colour_classes, true, pixrel))
      return false;

   return g17p_resource_program_compose(
      program_va, pre_copy_body, pre_copy_body_size, copies, copies_size,
      packed_tail, output, capacity, written, pixrel, sizeof(pixrel), false, NULL);
}

static bool
g17p_external_setup_launcher_compose(
   uint64_t program_va, const void *root_prefix, size_t root_prefix_size,
   const struct agx_ioto_state_tail_output *packed_tail, uint64_t setup_va,
   void *output, size_t capacity, size_t *written, bool trusted)
{
   if (!output || !written || !root_prefix || root_prefix_size == 0 ||
       !g17p_type0_tail_shape_valid(packed_tail) || program_va == 0 ||
       program_va > UINT64_C(0xffffffffffff) ||
       (program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       setup_va == 0 || setup_va >= (UINT64_C(1) << 48) ||
       (setup_va & 1) != 0 ||
       (!trusted &&
        !agx_ioto_straight_line_body_validate(AGX_CHIP_G17P, root_prefix,
                                              root_prefix_size)))
      return false;

   const size_t branch_offset = root_prefix_size + packed_tail->length +
                                sizeof(g17p_default_profile_prefix) +
                                AGX_IOTO_PRESHADER_PRELUDE_SIZE;
   if (branch_offset > UINT64_MAX - program_va)
      return false;
   const uint64_t branch_va = program_va + branch_offset;
   const int64_t displacement = (int64_t)setup_va - (int64_t)branch_va;
   if (displacement < -(INT64_C(1) << 47) ||
       displacement >= (INT64_C(1) << 47) || (displacement & 1) != 0)
      return false;
   const uint64_t encoded_displacement =
      (uint64_t)displacement & ((UINT64_C(1) << 48) - 1);

   struct agx_ioto_preshader_output branch;
   if (trusted) {
      branch = (struct agx_ioto_preshader_output){
         .mode = AGX_IOTO_PRESHADER_MODE_EXTERNAL,
         .length = AGX_IOTO_PRESHADER_PRELUDE_SIZE +
                   AGX_IOTO_PRESHADER_BRANCH_SIZE,
         .branch_offset = AGX_IOTO_PRESHADER_PRELUDE_SIZE,
         .bytes = {0x0c, 0x10},
      };
      if (!agx_ioto_preshader_pack_branch(
             AGX_CHIP_G17P, encoded_displacement, false,
             branch.bytes + branch.branch_offset))
         return false;
   } else {
      const struct agx_ioto_preshader_desc external = {
         .external_va = encoded_displacement,
         .force_external = true,
         .request_prelude = true,
      };
      if (!agx_ioto_preshader_emit(AGX_CHIP_G17P, &external, &branch) ||
          branch.mode != AGX_IOTO_PRESHADER_MODE_EXTERNAL ||
          branch.branch_offset != AGX_IOTO_PRESHADER_PRELUDE_SIZE)
         return false;
   }

   if (root_prefix_size > SIZE_MAX - packed_tail->length ||
       root_prefix_size + packed_tail->length >
          SIZE_MAX - sizeof(g17p_default_profile_prefix) ||
       root_prefix_size + packed_tail->length +
             sizeof(g17p_default_profile_prefix) >
          SIZE_MAX - branch.length)
      return false;

   const size_t total = root_prefix_size + packed_tail->length +
                        sizeof(g17p_default_profile_prefix) + branch.length;
   if (capacity < total)
      return false;

   uint8_t staged[AGX_IOTO_G17P_EXTERNAL_LAUNCHER_MAX_SIZE];
   if (total > sizeof(staged))
      return false;

   size_t offset = 0;
   memcpy(staged + offset, root_prefix, root_prefix_size);
   offset += root_prefix_size;
   memcpy(staged + offset, packed_tail->bytes, packed_tail->length);
   offset += packed_tail->length;
   memcpy(staged + offset, g17p_default_profile_prefix,
          sizeof(g17p_default_profile_prefix));
   offset += sizeof(g17p_default_profile_prefix);
   memcpy(staged + offset, branch.bytes, branch.length);
   offset += branch.length;
   if (offset != total)
      return false;

   memcpy(output, staged, total);
   *written = total;
   return true;
}

bool
agx_ioto_g17p_external_setup_launcher_compose_trusted(
   uint64_t program_va, const void *root_prefix, size_t root_prefix_size,
   const struct agx_ioto_state_tail_output *packed_tail, uint64_t setup_va,
   void *output, size_t capacity, size_t *written)
{
   return g17p_external_setup_launcher_compose(
      program_va, root_prefix, root_prefix_size, packed_tail, setup_va,
      output, capacity, written, true);
}

bool
agx_ioto_g17p_external_setup_suffix_compose(
   const void *body, size_t body_size, bool fragment, bool coverage,
   unsigned colour_classes, void *output, size_t capacity, size_t *written)
{
   if (!output || !written || (body_size && !body) ||
       !agx_ioto_straight_line_body_validate(AGX_CHIP_G17P, body, body_size))
      return false;

   uint8_t terminal[G17P_FRAGMENT_PIXREL_SIZE];
   const uint8_t *terminal_bytes = g17p_default_profile_suffix;
   size_t terminal_size = sizeof(g17p_default_profile_suffix);
   if (fragment) {
      if (!g17p_fragment_pixrel_compose(colour_classes, coverage, terminal))
         return false;
      terminal_bytes = terminal;
      terminal_size = sizeof(terminal);
   }

   if (body_size > SIZE_MAX - terminal_size ||
       body_size + terminal_size > SIZE_MAX - AGX_IOTO_PROGRAM_TERMINATE_SIZE)
      return false;
   const size_t total = body_size + terminal_size +
                        AGX_IOTO_PROGRAM_TERMINATE_SIZE;
   if (capacity < total)
      return false;

   memmove(output, body, body_size);
   memcpy((uint8_t *)output + body_size, terminal_bytes, terminal_size);
   memset((uint8_t *)output + body_size + terminal_size, 0,
          AGX_IOTO_PROGRAM_TERMINATE_SIZE);
   ((uint8_t *)output)[body_size + terminal_size] = UINT8_C(0x0e);
   *written = total;
   return true;
}

bool
agx_ioto_g17p_background_resource_program_compose_trusted(
   uint64_t program_va, const void *pre_copy_body, size_t pre_copy_body_size,
   const void *copies, size_t copies_size,
   const struct agx_ioto_state_tail_output *packed_tail, void *output,
   size_t capacity, size_t *written)
{
   return g17p_resource_program_compose(
      program_va, pre_copy_body, pre_copy_body_size, copies, copies_size,
      packed_tail, output, capacity, written, g17p_background_pixrel,
      sizeof(g17p_background_pixrel), false, NULL);
}

static bool
agx_ioto_g17s_ldimm_ep2_pack(uint16_t dest_half, uint32_t immediate,
                            uint8_t output[8])
{
   if (!output || (dest_half & 1) != 0 || dest_half >= 256)
      return false;

   /* Exact HAL300 LDIMM ep2 equations.  This deliberately does not select
    * the two-byte short form: emitUscStateLoad patches the pre-existing
    * sixteen-byte address pair in place. */
   const uint16_t w0 = UINT16_C(0x800c) |
                       (uint16_t)(((dest_half >> 1) & 0xf) << 4) |
                       (uint16_t)((immediate & 0x7f) << 8);
   const uint16_t w1 = UINT16_C(0x0002) |
                       (uint16_t)(((dest_half >> 5) & 3) << 6) |
                       (uint16_t)(((dest_half >> 7) & 1) << 3) |
                       (uint16_t)(((immediate >> 25) & 0x7f) << 9);
   const uint16_t w2 = (uint16_t)(((immediate >> 7) & 0xf) << 1) |
                       (uint16_t)(((immediate >> 11) & 3) << 10);
   const uint16_t w3 = (uint16_t)((immediate >> 13) & 0xf) |
                       (uint16_t)(((immediate >> 17) & 0xff) << 4);

   output[0] = (uint8_t)w0;
   output[1] = (uint8_t)(w0 >> 8);
   output[2] = (uint8_t)w1;
   output[3] = (uint8_t)(w1 >> 8);
   output[4] = (uint8_t)w2;
   output[5] = (uint8_t)(w2 >> 8);
   output[6] = (uint8_t)w3;
   output[7] = (uint8_t)(w3 >> 8);
   return true;
}

static bool
agx_ioto_g17s_minimal_compute_program_compose_internal(
   uint64_t program_va, uint64_t shader_va,
   const struct agx_ioto_g17s_minimal_compute_resources *resources,
   void *output, size_t capacity, size_t *written)
{
   if (!output || !written ||
       program_va > UINT64_C(0xffffffffffff) ||
       (program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 ||
       capacity < AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE ||
       (resources &&
        (resources->gather_buffer_va == 0 ||
         resources->user_compute_argument_gpu_va_1 == 0 ||
         resources->user_compute_argument_gpu_va_1 >
            UINT64_MAX -
               AGX_IOTO_G17S_MINIMAL_COMPUTE_USER_BUFFER_ADDEND ||
         resources->absolute_uniform_data_va == 0)))
      return false;

   struct agx_ioto_state_tail tail;
   if (!agx_ioto_g17s_minimal_compute_tail_build(shader_va, &tail))
      return false;

   struct agx_ioto_state_tail_output packed_tail;
   if (!agx_ioto_state_tail_pack(AGX_CHIP_G17S, &tail, &packed_tail) ||
       packed_tail.length != AGX_IOTO_STATE_TAIL_TYPE0_SIZE)
      return false;

   uint8_t staged[AGX_IOTO_G17S_MINIMAL_COMPUTE_PROGRAM_SIZE];
   size_t offset = 0;
   memcpy(staged + offset, g17s_minimal_compute_resource_body,
          sizeof(g17s_minimal_compute_resource_body));
   const uint64_t user_buffer_va =
      resources ? resources->user_compute_argument_gpu_va_1 +
                     AGX_IOTO_G17S_MINIMAL_COMPUTE_USER_BUFFER_ADDEND
                : 0;
   if (resources &&
       (!agx_ioto_g17s_ldimm_ep2_pack(
           4, (uint32_t)resources->gather_buffer_va, staged + offset) ||
        !agx_ioto_g17s_ldimm_ep2_pack(
           6, (uint32_t)(resources->gather_buffer_va >> 32),
           staged + offset + 8) ||
        !agx_ioto_g17s_ldimm_ep2_pack(
           24, (uint32_t)user_buffer_va, staged + offset + 16) ||
        !agx_ioto_g17s_ldimm_ep2_pack(
           26, (uint32_t)(user_buffer_va >> 32),
           staged + offset + 24) ||
        !agx_ioto_g17s_ldimm_ep2_pack(
           28, (uint32_t)resources->absolute_uniform_data_va,
           staged + offset + 32) ||
        !agx_ioto_g17s_ldimm_ep2_pack(
           30, (uint32_t)(resources->absolute_uniform_data_va >> 32),
           staged + offset + 40)))
      return false;
   offset += sizeof(g17s_minimal_compute_resource_body);
   memcpy(staged + offset, packed_tail.bytes, packed_tail.length);
   offset += packed_tail.length;
   if (!agx_ioto_g17s_minimal_profile_loader_pack(
          staged + offset, sizeof(staged) - offset))
      return false;
   offset += AGX_IOTO_G17S_MINIMAL_PROFILE_LOADER_SIZE;
   memcpy(staged + offset, g17s_minimal_compute_copies,
          sizeof(g17s_minimal_compute_copies));
   offset += sizeof(g17s_minimal_compute_copies);
   static const uint8_t terminal[AGX_IOTO_PROGRAM_TERMINATE_SIZE] = {
      0x0e, 0x00, 0x00, 0x00,
   };
   memcpy(staged + offset, terminal, sizeof(terminal));
   offset += sizeof(terminal);

   if (offset != sizeof(staged))
      return false;

   memcpy(output, staged, sizeof(staged));
   *written = sizeof(staged);
   return true;
}

bool
agx_ioto_g17s_minimal_compute_program_compose(
   uint64_t program_va, uint64_t shader_va, void *output, size_t capacity,
   size_t *written)
{
   return agx_ioto_g17s_minimal_compute_program_compose_internal(
      program_va, shader_va, NULL, output, capacity, written);
}

bool
agx_ioto_g17s_minimal_compute_program_compose_resources(
   uint64_t program_va, uint64_t shader_va,
   const struct agx_ioto_g17s_minimal_compute_resources *resources,
   void *output, size_t capacity, size_t *written)
{
   if (!resources)
      return false;

   return agx_ioto_g17s_minimal_compute_program_compose_internal(
      program_va, shader_va, resources, output, capacity, written);
}


bool
agx_ioto_program_copy_suffix_offset(enum agx_chip chip, const void *body,
                                      size_t body_size, size_t *offset)
{
   if (!offset || (body_size && !body))
      return false;

   const uint8_t *bytes = body;
   size_t cursor = 0, suffix = 0;
   while (cursor < body_size) {
      if (agx_ioto_program_gpu0_inline_lddev(chip, bytes + cursor,
                                             body_size - cursor)) {
         cursor += AGX_IOTO_GPU0_INLINE_LDDEV_SIZE;
         suffix = cursor;
         continue;
      }
      struct agx_ioto_record record;
      size_t consumed = 0;
      if (!agx_ioto_decode_one(chip, bytes + cursor, body_size - cursor,
                                 &record, &consumed) || consumed == 0 ||
          !agx_ioto_straight_line_body_kind(record.kind))
         return false;
      cursor += consumed;
      if (record.kind != AGX_IOTO_KIND_COPY_EP0 &&
          record.kind != AGX_IOTO_KIND_COPY_EP1)
         suffix = cursor;
   }
   *offset = suffix;
   return true;
}

/* Loader allocation can finish an early round before more LDIMM/LDDEVs.
 * State setup must precede those early uniform writes as well as the final
 * contiguous COPY suffix. A one-round body keeps its original ordering. */
bool
agx_ioto_program_first_copy_offset(enum agx_chip chip, const void *body,
                                    size_t body_size, size_t *offset)
{
   if (!offset || (body_size && !body))
      return false;
   const uint8_t *bytes = body;
   size_t cursor = 0;
   *offset = body_size;
   while (cursor < body_size) {
      struct agx_ioto_record record;
      size_t consumed = 0;
      if (!agx_ioto_decode_one(chip, bytes + cursor, body_size - cursor,
                               &record, &consumed) || !consumed ||
          !agx_ioto_straight_line_body_kind(record.kind))
         return false;
      if (*offset == body_size &&
          (record.kind == AGX_IOTO_KIND_COPY_EP0 ||
           record.kind == AGX_IOTO_KIND_COPY_EP1))
         *offset = cursor;
      cursor += consumed;
   }
   return true;
}

static bool
m5_no_resource_program_compose(
   enum agx_chip chip, uint64_t program_va, const struct agx_ioto_state_tail_output *packed_tail,
   bool fragment, void *output, size_t capacity, size_t *written)
{
   const size_t size = fragment
      ? AGX_IOTO_G17G_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE
      : AGX_IOTO_G17G_VERTEX_NO_RESOURCE_PROGRAM_SIZE;
   if (!output || !written ||
       !type0_tail_records_valid(AGX_CHIP_G17G, packed_tail) ||
       program_va == 0 || program_va > UINT64_C(0x1000000000000) - size ||
       (program_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) != 0 || capacity < size)
      return false;

   const uint8_t *ldshdr = packed_tail->bytes + AGX_IOTO_SPECLM_SIZE;
   if (!(agx_load_le16(ldshdr) & UINT16_C(0xffc0)) &&
       !agx_load_le16(ldshdr + 6) && !agx_load_le16(ldshdr + 8))
      return false;

   uint8_t staged[AGX_IOTO_G17G_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE] = {0};
   memcpy(staged, packed_tail->bytes, packed_tail->length);
   size_t offset = packed_tail->length;

   if (!agx_ioto_g17s_minimal_profile_loader_pack(staged + offset,
                                                sizeof(staged) - offset))
      return false;

   /* G17C takes the G17S value; scaling it with the cluster count (x4)
    * drops trailing draws of multi-draw passes. */
   staged[offset + 20] =
      (fragment ? 4 : 8) *
      ((chip == AGX_CHIP_G17S || chip == AGX_CHIP_G17C) ? 2 : 1);
   offset += AGX_IOTO_G17S_MINIMAL_PROFILE_LOADER_SIZE;
   if (fragment) {
      static const uint8_t pixrel[] = {0x9f, 0x17, 0x00, 0x04, 0x00, 0x00};
      memcpy(staged + offset, pixrel, sizeof(pixrel));
      offset += sizeof(pixrel);
   }
   staged[offset] = 0x0e;
   offset += AGX_IOTO_PROGRAM_TERMINATE_SIZE;
   if (offset != size)
      return false;

   memcpy(output, staged, size);
   *written = size;
   return true;
}

bool
agx_ioto_m5_resource_program_compose_classes(
   enum agx_chip chip, uint64_t program_va, const void *body, size_t body_size,
   const struct agx_ioto_state_tail_output *packed_tail,
   enum agx_ioto_g17g_program_stage stage, unsigned colour_classes,
   void *output, size_t capacity, size_t *written)
{
   if (!agx_chip_g17_m5(chip) || !output || !written ||
       (colour_classes == 0 || (colour_classes & ~7u)) ||
       (stage != AGX_IOTO_G17G_PROGRAM_VERTEX &&
        stage != AGX_IOTO_G17G_PROGRAM_FRAGMENT &&
        stage != AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT &&
        stage != AGX_IOTO_G17G_PROGRAM_EOT &&
        stage != AGX_IOTO_G17G_PROGRAM_BACKGROUND &&
        stage != AGX_IOTO_G17G_PROGRAM_COMPUTE &&
        stage != AGX_IOTO_G17G_PROGRAM_MESH &&
        stage != AGX_IOTO_G17G_PROGRAM_OBJECT))
      return false;

   size_t copy_offset;
   if (!agx_ioto_program_first_copy_offset(AGX_CHIP_G17G, body, body_size,
                                            &copy_offset))
      return false;

   uint8_t base[AGX_IOTO_G17G_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE];
   size_t base_size;
   if (!m5_no_resource_program_compose(
          chip, program_va, packed_tail, stage != AGX_IOTO_G17G_PROGRAM_VERTEX,
          base, sizeof(base), &base_size))
      return false;

   const size_t prefix_size = AGX_IOTO_STATE_TAIL_TYPE0_SIZE +
                             AGX_IOTO_G17S_MINIMAL_PROFILE_LOADER_SIZE;
   const bool pixel_release = stage == AGX_IOTO_G17G_PROGRAM_FRAGMENT ||
                              stage == AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT ||
                              stage == AGX_IOTO_G17G_PROGRAM_BACKGROUND;
   if (stage == AGX_IOTO_G17G_PROGRAM_FRAGMENT ||
       stage == AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT) {
      /* HAL300 PIXREL for 0x2ff with the shader's colour classes
       * removed. Sparse targets retain their own class pair. */
      static const uint8_t release[8][6] = {
         {0},
         {0x9f, 0x17, 0, 4, 0, 0},
         {0x7f, 0x16, 0, 4, 0, 0},
         {0x1f, 0x16, 0, 4, 0, 0},
         {0xff, 0x11, 0, 4, 0, 0},
         {0x9f, 0x11, 0, 4, 0, 0},
         {0x7f, 0x10, 0, 4, 0, 0},
         {0x1f, 0x10, 0, 4, 0, 0},
      };
      memcpy(base + prefix_size, release[colour_classes], sizeof(release[0]));
      /* Depth/coverage consumes class 0 as well. HAL300 clears descriptor
       * bit 0, encoded as bit 3 of the first PIXREL byte. */
      if (stage == AGX_IOTO_G17G_PROGRAM_COVERAGE_FRAGMENT)
         base[prefix_size] &= ~8u;
   }
   if (stage == AGX_IOTO_G17G_PROGRAM_BACKGROUND) {

      static const uint8_t background_pixrel[] = {7, 0x10, 0, 4, 0, 0};
      memcpy(base + prefix_size, background_pixrel, sizeof(background_pixrel));
   }
   const size_t suffix_size = pixel_release
                                ? base_size - prefix_size
                                : AGX_IOTO_PROGRAM_TERMINATE_SIZE;
   const size_t fixed_size = prefix_size + suffix_size;
   if (body_size > SIZE_MAX - fixed_size)
      return false;
   const size_t size = body_size + fixed_size;
   if (capacity < size || size > UINT64_C(0x1000000000000) ||
       program_va > UINT64_C(0x1000000000000) - size)
      return false;

   uint8_t *staged = malloc(size);
   if (!staged)
      return false;
   /* The compute profile helper must run after the uniform uploads. On M5,
    * calling it before COPY leaves the probe's uniforms zero, even when
    * those copies follow in the Ioto stream. Keep shader-state setup
    * before the first copy, but defer the helper until all uploads finish.
    */
   const size_t before_copy = stage == AGX_IOTO_G17G_PROGRAM_COMPUTE
                                ? AGX_IOTO_STATE_TAIL_TYPE0_SIZE : prefix_size;
   if (copy_offset)
      memcpy(staged, body, copy_offset);
   memcpy(staged + copy_offset, base, before_copy);
   if (body_size != copy_offset)
      memcpy(staged + copy_offset + before_copy,
             (const uint8_t *)body + copy_offset, body_size - copy_offset);
   if (before_copy != prefix_size)
      memcpy(staged + body_size + before_copy, base + before_copy,
             prefix_size - before_copy);
   memcpy(staged + body_size + prefix_size,
          base + base_size - suffix_size, suffix_size);
   memcpy(output, staged, size);
   free(staged);
   *written = size;
   return true;
}

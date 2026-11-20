/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_vary_core.h"

#include <assert.h>
#include <string.h>

#include "agx3_binder.h"

static_assert(sizeof(struct agx3_uvs_reloc) == AGX3_UVS_RELOC_SIZE, "packed");

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

static bool
is_b(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3B;
}

static bool
is_d(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3D;
}

/* At most one scoreboard slot: agx3d's re-encoded forms carry a single wait
 * slot code rather than agx3c's mask. */
static bool
single_wait(uint16_t wait_mask)
{
   return (wait_mask & (wait_mask - 1)) == 0;
}

static bool
agx3_vary_interp_mode(enum agx_isa_encoding enc, uint16_t dest,
                      uint16_t coef_index, bool perspective,
                      enum agx3_vary_location location, bool first,
                      bool projective_rhw, bool explicit_sample,
                      uint16_t coords, bool coords_kill,
                      uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   /* Native G17G/G18G pipelines use the same explicit coordinate token for
    * centroid and sample interpolation, including projective RHW. There is
    * no reserved implicit centroid register: its token is allocated normally.
    * Native657's 1x/2x/4x G17G/G18G center profiles also carry the private
    * RHW marker, including first-input metadata on the first load; AGX3D
    * (G18G) admits neither first-input metadata nor center RHW here. */
   if ((enc != AGX_ISA_ENCODING_AGX3B &&
        !agx_encoding_is_g17_class(enc)) ||
       (is_d(enc) && (first || (projective_rhw && !explicit_sample))) ||
       (projective_rhw && ((!explicit_sample && location != AGX3_VARY_CENTER) ||
                           !perspective || coef_index > 0x7f)) ||
       (explicit_sample && (first || (coords & 1) != 0 || coords >= 256)) ||
       (dest & 1) != 0 || dest >= 256 ||
       (location != AGX3_VARY_CENTER && location != AGX3_VARY_CENTROID))
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 2), KV(10, 2), KV(25, 6),
                                          KV(26, 2), KV(40, 2)};
   const struct agx3_field_kv base_c[] = {KV(9, 2),  KV(10, 2), KV(25, 6),
                                          KV(26, 2), KV(40, 2), KV(44, 3),
                                          KV(45, 58)};
   struct agx3_field_kv ov[9];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(22, coef_index);
   /* Native G17G token interpolation (lane-mesh/cwit-interp
    * ias_k1, ias_k1_np, ias_dyn, sp@1) sets f11 for perspective and
    * no-perspective inputs alike. AGX3D (G18G) keeps f11 = perspective. */
   ov[n++] = (struct agx3_field_kv)KV(
      11, (perspective || (explicit_sample && !is_b(enc) && !is_d(enc)))
             ? 1 : 0);
   if (explicit_sample) {
      /* Apple Metal 32023.917, applegpu_g17p. The source SHA-256
       * 2f0835ef... produces perspective, no-perspective, and mixed
       * center/sample programs:
       * f39/f40 select a 32-bit coordinate register, f32 names the token,
       * and f41/f43 are its live/final-use alternatives. Native G17G
       * (lane-mesh/cwit-interp) encodes the same leaf on its common AGX3C
       * f32interp base (f44/f45). */
      ov[n++] = (struct agx3_field_kv)KV(39, 2);
      ov[n++] = (struct agx3_field_kv)KV(32, coords);
      ov[n++] = (struct agx3_field_kv)KV(coords_kill ? 43 : 41, 1);
   } else if (location == AGX3_VARY_CENTER) {
      ov[n++] = (struct agx3_field_kv)KV(37, 66);
      ov[n++] = (struct agx3_field_kv)KV(39, 7);
      ov[n++] = (struct agx3_field_kv)KV(41, 0);
      ov[n++] = (struct agx3_field_kv)KV(32, 0);
   } else {
      /* Neither G17P nor G17G has an implicit centroid leaf: both read SR39
       * into an explicit ITRCOORDS f38=1 token (lane-mesh/cwit-interp iac,
       * cwit-interp2 cp/cnp at 1 and 4 samples), consumed through the token
       * leaf above. The former AGX3C form read its coordinates from a fixed
       * r0/r6 register instead. */
      return false;
   }
   if (projective_rhw)
      ov[n++] = (struct agx3_field_kv)KV(54, 1);
   if (first)
      ov[n++] = (struct agx3_field_kv)KV(63, 1);
   return agx3_bind_compose(enc, AGX3_MN_f32interp, agx3_bind_ep(enc, 0),
                            is_b(enc) ? base_b : base_c, is_b(enc) ? 5 : 7,
                            ov, n, out, len);
}

/* ITRCOORDS modes: f38=2 derives a sample's coordinates from a sample index,
 * f38=1 derives the centroid of the covered samples from a coverage word. */
static bool
agx3_vary_coords(enum agx_isa_encoding enc, uint16_t dest, uint8_t mode,
                 uint16_t source, bool source_kill, uint16_t wait_mask,
                 uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   /* G17P has six scoreboard slots, f30..f35. The ten-byte G17G form adds
    * f36 and f37 for slots 6 and 7, at the same bits (22 and 3) as every
    * other C-class wait field. */
   const uint16_t slots = is_b(enc) ? 0x3f : 0xff;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !agx_encoding_is_g17_class(enc)) ||
       (dest & 1) != 0 || dest >= 256 || source >= 256 ||
       (wait_mask & ~slots) != 0)
      return false;

   /* ITRCOORDS is an eight-byte G17P instruction: register/32-bit
    * destination, register/16-bit source, and the mode in f38. G17G
    * (lane-mesh/cwit-interp, cwit-interp2) uses the same fields in a
    * ten-byte form that adds f28/f29. Every native G17G instance marks its
    * source final (f27); the unmarked form keeps G17P's field meaning.
    * G18G adds the same opcode constants f28=3/f29=0x66. */
   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, 2), KV(11, 1), KV(23, 2), KV(24, 1), KV(38, mode),
      KV(28, 3), KV(29, 102),
   };
   struct agx3_field_kv ov[11];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(16, source);
   if (source_kill)
      ov[n++] = (struct agx3_field_kv)KV(27, 1);
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);

   return agx3_bind_compose(enc, AGX3_MN_itrcoords, agx3_bind_ep(enc, 0), base,
                            is_b(enc) ? 6 : 8, ov, n, out, len);
}

bool
agx3_vary_sample_coords(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t sample_index, bool sample_index_kill,
                        uint16_t wait_mask, uint8_t out[AGX3_VARY_MAX_LENGTH],
                        uint8_t *len)
{
   /* Fresh Apple Metal 32023.917 `interpolate_at_sample` and
    * `[[sample_{,no_}perspective]]` witnesses. */
   return agx3_vary_coords(enc, dest, 2, sample_index, sample_index_kill,
                           wait_mask, out, len);
}

bool
agx3_vary_centroid_coords(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t coverage, bool coverage_kill,
                          uint16_t wait_mask,
                          uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   /* Fresh applegpu_g17p `[[centroid_{,no_}perspective]]` witnesses at
    * raster_sample_count 1 and 4 read SR39 and emit ITRCOORDS f38=1 on it,
    * then consume the token exactly as explicit sample interpolation does. */
   return agx3_vary_coords(enc, dest, 1, coverage, coverage_kill, wait_mask,
                           out, len);
}

bool
agx3_vary_interp_sample(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t coef_index, bool perspective, uint16_t coords,
                        bool coords_kill, uint8_t out[AGX3_VARY_MAX_LENGTH],
                        uint8_t *len)
{
   return agx3_vary_interp_mode(enc, dest, coef_index, perspective,
                                AGX3_VARY_CENTER, false, false, true, coords,
                                coords_kill, out, len);
}

bool
agx3_vary_projective_rhw_sample(enum agx_isa_encoding enc, uint16_t dest,
                                uint16_t coef_index, uint16_t coords,
                                bool coords_kill,
                                uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   return agx3_vary_interp_mode(enc, dest, coef_index, true, AGX3_VARY_CENTER,
                                false, true, true, coords, coords_kill, out,
                                len);
}

bool
agx3_vary_interp_at_start(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t coef_index, bool perspective,
                         enum agx3_vary_location location, bool first,
                         uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   return agx3_vary_interp_mode(enc, dest, coef_index, perspective, location,
                                first, false, false, 0, false, out, len);
}

bool
agx3_vary_projective_rhw(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t coef_index, bool first,
                         uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   return agx3_vary_interp_mode(enc, dest, coef_index, true, AGX3_VARY_CENTER,
                                first, true, false, 0, false, out, len);
}

bool
agx3_vary_interp_at(enum agx_isa_encoding enc, uint16_t dest,
                    uint16_t coef_index, bool perspective,
                    enum agx3_vary_location location,
                    uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   return agx3_vary_interp_at_start(enc, dest, coef_index, perspective,
                                    location, false, out, len);
}

bool
agx3_vary_ldcf_at_start(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t coef_index, uint8_t slot, bool first,
                        uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   if ((enc != AGX_ISA_ENCODING_AGX3B &&
        !agx_encoding_is_g17_class(enc)) ||
       (first && !is_b(enc)) ||
       (dest & 1) != 0 || dest >= 256 || slot < 1 ||
       slot > (is_b(enc) ? 6 : 8))
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 2), KV(10, 2), KV(23, 1),
                                          KV(24, 1), KV(38, 7)};
   const struct agx3_field_kv base_c[] = {KV(9, 2),  KV(10, 2), KV(23, 1),
                                          KV(24, 1), KV(38, 7), KV(29, 103)};
   /* G18G (agx3d) flat loads keep every agx3c operand field, but f38 is 5
    * instead of 7 and f49 = 2 is new, in each of the 27 applegpu_g18g LDCFR
    * words witnessed: float, float2, float3, float4, half4, int and uint
    * [[flat]] inputs (scratch flat probe, the m5 zs/depth fragments and the
    * glxgears flat-colour fragment), against f38 = 7 and no f49 in every
    * applegpu_g17g build of the same inputs. */
   const struct agx3_field_kv base_d[] = {KV(9, 2),  KV(10, 2), KV(23, 1),
                                          KV(24, 1), KV(38, 5), KV(29, 103),
                                          KV(49, 2)};
   const struct agx3_field_kv ov[] = {KV(2, dest), KV(15, coef_index),
                                      KV(12, slot), KV(48, first)};
   return agx3_bind_compose(enc, AGX3_MN_ldcfr, agx3_bind_ep(enc, 0),
                            is_b(enc) ? base_b : is_d(enc) ? base_d : base_c,
                            is_b(enc) ? 5 : is_d(enc) ? 7 : 6, ov,
                            first ? 4 : 3, out, len);
}

bool
agx3_vary_ldcf(enum agx_isa_encoding enc, uint16_t dest, uint16_t coef_index,
               uint8_t slot, uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   return agx3_vary_ldcf_at_start(enc, dest, coef_index, slot, false, out, len);
}

bool
agx3_vary_uvsw(enum agx_isa_encoding enc, uint16_t value, uint16_t uvs_index,
               uint8_t slot, bool value_kill, uint16_t wait_mask,
               uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   /* G18G (agx3d) writes vertex outputs with the agx3c uvsw field vector:
    * the 181 aligned applegpu_g17g/applegpu_g18g uvsw words of the m5
    * reference corpus differ only in output register, UVS index, slot, kill
    * and wait choices, and every G18G vertex program here ends in them. Its
    * wait field is one slot code. */
   const unsigned nwait = is_b(enc) ? 6 : 8;
   if ((enc != AGX_ISA_ENCODING_AGX3B &&
        !agx_encoding_is_g17_class(enc)) ||
       (is_d(enc) && !single_wait(wait_mask)) ||
       (value & 1) != 0 || value >= 256 || slot < 1 || slot > nwait ||
       (wait_mask & ~((1u << nwait) - 1)) != 0)
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 5), KV(23, 1), KV(24, 1)};
   const struct agx3_field_kv base_c[] = {KV(9, 5),  KV(10, 2), KV(23, 1),
                                          KV(24, 1), KV(28, 2), KV(29, 196)};
   struct agx3_field_kv ov[16];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(5, value);
   ov[n++] = (struct agx3_field_kv)KV(15, uvs_index);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(13, value_kill ? 1 : 0);
   /* Wait bits f30.. (6 on agx3b, 8 on agx3c). */
   for (unsigned i = 0; i < nwait; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);

   return agx3_bind_compose(enc, AGX3_MN_uvsw, agx3_bind_ep(enc, 0),
                            is_b(enc) ? base_b : base_c, is_b(enc) ? 3 : 6,
                            ov, n, out, len);
}

bool
agx3_vary_uvsw_register_index(enum agx_isa_encoding enc, uint16_t value,
                              uint16_t index, uint8_t slot, bool value_kill,
                              uint16_t wait_mask,
                              uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   /* uvsw's index operand has three kinds in Apple's AGX3 tables:
    * f23 = 1 immediate (f15), 2 register (f16) and 3 uniform (f17), each a
    * half wide. Metal never computes a vertex output's index, so only the
    * immediate kind is witnessed in compiled programs; the register kind is
    * the table's own form for the same field. */
   const unsigned nwait = is_b(enc) ? 6 : 8;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !agx_encoding_is_g17_class(enc)) ||
       (is_d(enc) && !single_wait(wait_mask)) ||
       (value & 1) != 0 || value >= 256 || index >= 256 ||
       slot < 1 || slot > nwait ||
       (wait_mask & ~((1u << nwait) - 1)) != 0)
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 5), KV(23, 2), KV(24, 1)};
   const struct agx3_field_kv base_c[] = {KV(9, 5), KV(10, 2), KV(23, 2),
                                        KV(24, 1), KV(28, 2), KV(29, 196)};
   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(5, value);
   ov[n++] = (struct agx3_field_kv)KV(16, index);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(13, value_kill ? 1 : 0);
   for (unsigned i = 0; i < nwait; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);

   return agx3_bind_compose(enc, AGX3_MN_uvsw, agx3_bind_ep(enc, 0),
                            is_b(enc) ? base_b : base_c, is_b(enc) ? 3 : 6,
                            ov, n, out, len);
}

bool
agx3_vary_uvsw_uniform_index(enum agx_isa_encoding enc, uint16_t value,
                             uint16_t uniform, uint8_t slot, bool value_kill,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   /* The uniform kind of uvsw's index operand (f23 = 3, half index in f17;
    * see agx3_vary_uvsw_register_index()). */
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned nslots = m5 ? 8 : 6;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || (value & 1) != 0 || value >= 256 ||
       uniform >= 512 || slot < 1 || slot > nslots || (wait_mask & ~((1u << nslots) - 1)) != 0)
      return false;
   const struct agx3_field_kv base[] = {KV(9, 5), KV(23, 3), KV(24, 1)};
   const struct agx3_field_kv base_c[] = {
      KV(9,5), KV(10,2), KV(23,3), KV(24,1), KV(28,2), KV(29,196),
   };
   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(5, value);
   ov[n++] = (struct agx3_field_kv)KV(17, uniform);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   ov[n++] = (struct agx3_field_kv)KV(13, value_kill ? 1 : 0);
   for (unsigned i = 0; i < nslots; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);

   return agx3_bind_compose(enc, AGX3_MN_uvsw, 0, m5 ? base_c : base, m5 ? 6 : 3, ov, n, out, len);
}

bool
agx3_vary_interp(enum agx_isa_encoding enc, uint16_t dest,
                 uint16_t coef_index, bool perspective,
                 uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   return agx3_vary_interp_at(enc, dest, coef_index, perspective,
                              AGX3_VARY_CENTER, out, len);
}

static bool
agx3_vary_relocated_uvsw(enum agx_isa_encoding enc,
                         const struct agx3_uvs_reloc *r, uint16_t index,
                         uint8_t out[AGX3_VARY_MAX_LENGTH], uint8_t *len)
{
   return agx3_vary_uvsw(enc, r->value, index, r->slot, true, r->wait, out,
                         len);
}

bool
agx3_vary_relocate_uvsw(enum agx_isa_encoding enc, uint8_t *code, size_t size,
                        const uint8_t *relocs, unsigned count,
                        const uint8_t map[AGX3_UVS_RELOC_MAP_SIZE])
{
   if (count && (!code || !relocs))
      return false;

   /* Check every write, and that its new form encodes at the same length,
    * before changing any: an encoding's length follows from its form, not
    * from the index, so a failure here means the list does not describe
    * this code. */
   for (unsigned pass = 0; pass < 2; ++pass) {
      for (unsigned i = 0; i < count; ++i) {
         struct agx3_uvs_reloc r;
         memcpy(&r, relocs + i * AGX3_UVS_RELOC_SIZE, sizeof(r));

         uint8_t old[AGX3_VARY_MAX_LENGTH], moved[AGX3_VARY_MAX_LENGTH];
         uint8_t old_len = 0, moved_len = 0;
         const uint16_t index = map ? map[r.index] : r.index;
         if (!agx3_vary_relocated_uvsw(enc, &r, r.index, old, &old_len) ||
             !agx3_vary_relocated_uvsw(enc, &r, index, moved, &moved_len) ||
             old_len != moved_len || r.offset > size ||
             size - r.offset < old_len)
            return false;

         if (pass == 0) {
            if (memcmp(code + r.offset, old, old_len))
               return false;
         } else {
            memcpy(code + r.offset, moved, moved_len);
         }
      }
   }

   return true;
}

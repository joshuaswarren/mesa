#include "agx3_rt_gen1_entry.h"
#include "agx3_binder.h"
#include "agx3_pack.h"
#include <string.h>
#define KV(f, v) {(f), (v)}

bool
agx3_rt_gen1_entry_generate(const struct agx3_rt_gen1_entry_desc *d,
                            uint8_t out[AGX3_RT_GEN1_ENTRY_MAX_SIZE],
                            size_t *length)
{
   if (!d || !out || !length || !d->uniforms || !d->constants_va ||
       (d->constants_va & 3) || (d->constants_va >> 48) ||
       !d->shader_va || (d->shader_va & 63) || (d->shader_va >> 42) ||
       !d->branch_va || (d->branch_va & 3) || (d->branch_va >> 48) ||
       d->fetch_span < 1 || d->fetch_span > 3 ||
       d->uniform_count < 2 || d->uniform_count > AGX3_RT_GEN1_ENTRY_MAX_UNIFORMS ||
       (d->context48 ? (d->constant_dwords != 28 && d->constant_dwords != 32)
                     : (d->constant_dwords < 2 || d->constant_dwords > 34 ||
                        d->constant_dwords % 4 != 2)))
      return false;
   for (unsigned i = 0; i < d->uniform_count; ++i) {
      if ((d->uniforms[i].uniform_half & 1) ||
          (d->uniforms[i].register_half & 1) ||
          d->uniforms[i].uniform_half > 70 ||
          d->uniforms[i].register_half > 94)
         return false;
   }
   uint8_t staged[AGX3_RT_GEN1_ENTRY_MAX_SIZE] = {0};
   size_t used = 0;
   const uint16_t base = d->context48 ? 28 : 24;
   const uint16_t dest[] = {8, base, base + 2};
   const uint32_t value[] = {0x6000, d->constants_va, d->constants_va >> 32};
   for (unsigned i = 0; i < 3; ++i) {
      struct agx3_encoded_ldimm imm;
      if (!agx3_pack_ldimm_u32(AGX_ISA_ENCODING_AGX3B, dest[i], value[i], &imm))
         return false;
      memcpy(staged + used, imm.bytes, imm.length);
      used += imm.length;
   }
   uint8_t n;
   for (unsigned i = 0; i <= d->context48; ++i) {
      const struct agx3_field_kv fields[] = {
         KV(2, i ? 20 : base - 4), KV(9, 2), KV(10, 2), KV(12, 2),
         KV(23, 1), KV(24, 1), KV(31, 8), KV(38, 2), KV(39, 2),
         KV(54, 1), KV(68, 3), KV(70, 2), KV(71, i ? 48 : 80),
      };
      if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, AGX3_MN_ldscs, 0,
                            fields, sizeof(fields) / sizeof(fields[0]),
                            NULL, 0, staged + used, &n))
         return false;
      used += n;
   }
   for (unsigned c = (d->constant_dwords + 3) / 4; c-- > 0;) {
      unsigned words = d->constant_dwords - 4 * c;
      if (words > 4) words = 4;
      const struct agx3_field_kv fields[] = {
         KV(2, base + 8 * c), KV(9, 2), KV(10, 2), KV(12, 2),
         KV(23, 1), KV(24, 1), KV(31, base), KV(38, 2), KV(39, 3),
         KV(42, c == 0), KV(54, 1), KV(73, (1u << words) - 1),
         KV(75, 2), KV(76, 16 * c),
      };
      if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, AGX3_MN_lddev, 0,
                            fields, sizeof(fields) / sizeof(fields[0]),
                            NULL, 0, staged + used, &n))
         return false;
      used += n;
   }
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, AGX3_MN_specLM, 0,
                         NULL, 0, NULL, 0, staged + used, &n))
      return false;
   used += n;
   const struct agx3_field_kv shader[] = {
      KV(0, d->shader_va), KV(1, d->fetch_span), KV(23, 1),
   };
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, AGX3_MN_ldshdr, 0,
                         shader, sizeof(shader) / sizeof(shader[0]),
                         NULL, 0, staged + used, &n))
      return false;
   used += n;
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, AGX3_MN_specTPR, 0,
                         NULL, 0, NULL, 0, staged + used, &n))
      return false;
   used += n;
   for (unsigned i = 0; i < d->uniform_count; ++i) {
      const struct agx3_field_kv fields[] = {
         KV(3, d->uniforms[i].uniform_half), KV(9, 3), KV(10, 2),
         KV(16, d->uniforms[i].register_half), KV(23, 2), KV(24, 2),
         KV(27, 1), KV(31, i == 0),
      };
      if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, AGX3_MN_copy, 0,
                            fields, sizeof(fields) / sizeof(fields[0]),
                            NULL, 0, staged + used, &n))
         return false;
      used += n;
   }
   const struct agx3_field_kv branch[] = {KV(0, d->branch_va), KV(11, 1)};
   if (!agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, AGX3_MN_b, 0,
                         branch, sizeof(branch) / sizeof(branch[0]),
                         NULL, 0, staged + used, &n))
      return false;
   used += n;
   memcpy(out, staged, used);
   *length = used;
   return true;
}

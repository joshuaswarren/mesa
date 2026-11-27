#include "agx3_rt_gen1_traversal.h"
#include "agx3_binder.h"
#include "agx3_stage_core.h"
#include "agx3_rt_core.h"
#include <string.h>
#define KV(f, v) {(f), (v)}
bool
agx3_rt_gen1_encode_intersect(enum agx_isa_encoding enc,
   const struct agx3_rt_gen1_intersect_desc *d, uint8_t out[16], uint8_t *length)
{
   if (!out || !length) return false;
   memset(out, 0, 16); *length = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B || !d ||
       (d->bvh & 1) || (d->request & 1) ||
       d->bvh > (d->bvh_uniform ? 508 : 188) ||
       d->request > (d->request_uniform ? 508 : 188) ||
       d->context_uniform > 511)
      return false;
   const struct agx3_field_kv f[] = {
      KV(9,5), KV(12,6), KV(13,1), KV(30,1), KV(49,32), KV(55,3), KV(56,3),
      KV(d->bvh_uniform ? 63 : 62, d->bvh), KV(69,d->bvh_uniform ? 3 : 2),
      KV(70,2), KV(d->request_uniform ? 78 : 77,d->request),
      KV(84,d->request_uniform ? 3 : 2), KV(85,2),
      KV(88,!d->request_uniform), KV(92,d->context_uniform), KV(98,3), KV(99,1),
   };
   return agx3_bind_compose(enc,AGX3_MN_intersect_ray,0,f,sizeof(f)/sizeof(f[0]),
                            NULL,0,out,length);
}
bool
agx3_rt_gen1_encode_continue(enum agx_isa_encoding enc,
   uint8_t out[16], uint8_t *length)
{
   if (!out || !length) return false;
   memset(out,0,16); *length=0;
   if (enc != AGX_ISA_ENCODING_AGX3B) return false;
   const struct agx3_field_kv f[] = {KV(5,4),KV(9,5),KV(12,6),KV(13,1),KV(16,1)};
   return agx3_bind_compose(enc,AGX3_MN_continue_ray,0,f,sizeof(f)/sizeof(f[0]),
                            NULL,0,out,length);
}

bool
agx3_rt_gen1_encode_launch_word(unsigned word, uint16_t source,
   uint8_t out[16], uint8_t *length)
{
   return word<2 && agx3_stage_copy_liveness(AGX_ISA_ENCODING_AGX3B,
      2*word,source,true,false,1,0,out,length);
}

bool
agx3_rt_gen1_encode_resume(enum agx_isa_encoding enc, uint16_t resident_source,
                          uint8_t out[32], uint8_t *length)
{
   if (!out || !length) return false;
   memset(out, 0, 32); *length = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B) return false;
   uint8_t copy[16], fence[16], resume[16], nc, nf, nr;
   /* Fresh G17P force-opaque query +0x4e0 zero-extends SR59 into staging4;
    * +0x4f8 publishes SCS, +0x4fe consumes that staging operand. The NIR
    * source already has the extension, so a 32-bit copy is equivalent. */
   if (!agx3_stage_copy_liveness(enc, 4, resident_source, false, false, 1, 0,
                                copy, &nc) ||
       !agx3_rt_traversal_barrier(enc, AGX3_RT_BARRIER_IN_LOOP, fence, &nf) ||
       !agx3_rt_gen1_encode_continue(enc, resume, &nr))
      return false;
   memcpy(out, copy, nc);
   memcpy(out + nc, fence, nf);
   memcpy(out + nc + nf, resume, nr);
   *length = nc + nf + nr;
   return true;
}

bool
agx3_rt_gen1_encode_refill(enum agx_isa_encoding enc, uint16_t resident_source,
                          uint8_t out[32], uint8_t *length)
{
   if (!out || !length) return false;
   memset(out, 0, 32); *length = 0;
   if (enc != AGX_ISA_ENCODING_AGX3B) return false;
   uint8_t copy[16], fence[16], resume[16], nc, nf, nr;
   const struct agx3_field_kv f[] = {KV(9,5),KV(12,6),KV(13,1),KV(16,1)};
   /* Fresh guarded candidate_ids +0x9f8..0xa24 uses staging0 and the
    * entry-scope SCS publication after consuming the candidate queue. */
   if (!agx3_stage_copy_liveness(enc, 0, resident_source, false, false, 1, 0,
                                copy, &nc) ||
       !agx3_rt_traversal_barrier(enc, AGX3_RT_BARRIER_ENTRY, fence, &nf) ||
       !agx3_bind_compose(enc,AGX3_MN_continue_ray,0,f,sizeof(f)/sizeof(f[0]),
                         NULL,0,resume,&nr))
      return false;
   memcpy(out, copy, nc); memcpy(out + nc, fence, nf);
   memcpy(out + nc + nf, resume, nr); *length = nc + nf + nr;
   return true;
}

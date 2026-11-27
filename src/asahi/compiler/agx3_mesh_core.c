/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT */
#include "agx3_mesh_core.h"
#include "agx3_binder.h"
#define KV(f,v) {(f),(v)}
static bool agx3_gen1_mesh_allocate_uvb(uint16_t primitive_count, uint16_t vertex_count, uint8_t out[8], uint8_t *length)
{
   if (!out || !length || primitive_count >= 256 || vertex_count >= 256)
      return false;
   const struct agx3_field_kv fields[] = {
      KV(2,primitive_count), KV(9,2), KV(10,1), KV(16,vertex_count), KV(23,2), KV(24,1),
   };
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3B,AGX3_MN_allocate_uvb,0,fields,
                           sizeof(fields)/sizeof(fields[0]),NULL,0,out,length);
}
static bool agx3_gen1_mesh_uvst(uint16_t descriptor, uint8_t kind, bool last_use, uint8_t out[8], uint8_t *length)
{
   if (!out || !length || (descriptor & 1) || descriptor > 252 || kind > 2)
      return false;
   const struct agx3_field_kv fields[] = {
      KV(2,descriptor), KV(9,2), KV(10,1), KV(13,last_use), KV(24,1), KV(25,kind), KV(34,1),
   };
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3B,AGX3_MN_uvst,0,fields,
                           sizeof(fields)/sizeof(fields[0]),NULL,0,out,length);
}
static bool agx3_gen1_mesh_object_childcount(uint8_t out[8], uint8_t *length)
{
   if (!out || !length)
      return false;
   const struct agx3_field_kv fields[] = {
      KV(9,5), KV(12,1), KV(13,1), KV(24,1), KV(30,1),
   };
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3B,AGX3_MN_uvsw,0,fields,
                           sizeof(fields)/sizeof(fields[0]),NULL,0,out,length);
}

bool
agx3_mesh_allocate_uvb(enum agx_isa_encoding enc, uint16_t operand0,
                       uint16_t operand1, bool first_last_use, uint8_t out[8], uint8_t *length)
{
   if (enc == AGX_ISA_ENCODING_AGX3B)
      return agx3_gen1_mesh_allocate_uvb(operand0, operand1, out, length);
   if (!out || !length || enc != AGX_ISA_ENCODING_AGX3C ||
       operand0 >= 256 || operand1 >= 256)
      return false;
   

   const struct agx3_field_kv fields[] = {
      KV(2,operand0), KV(9,2), KV(10,1), KV(13,first_last_use), KV(16,operand1),
      KV(23,2), KV(24,1), KV(28,2), KV(29,5),
   };
   return agx3_bind_compose(enc,AGX3_MN_allocate_uvb,0,fields,
                            sizeof(fields)/sizeof(fields[0]),NULL,0,out,length);
}
bool
agx3_mesh_uvst(enum agx_isa_encoding enc, uint16_t descriptor,
              bool primitive, bool last_use, uint8_t out[8], uint8_t *length)
{
   if (enc == AGX_ISA_ENCODING_AGX3B)
      return agx3_gen1_mesh_uvst(descriptor, primitive ? 1 : 0, last_use, out, length);
   if (!out || !length || enc != AGX_ISA_ENCODING_AGX3C ||
       (descriptor & 1) || descriptor > 252)
      return false;
   

   const struct agx3_field_kv fields[] = {
      KV(2,descriptor), KV(9,2), KV(10,1), KV(13,last_use), KV(14,2),
      KV(15,195), KV(24,1), KV(25,primitive ? 1 : 0), KV(34,1),
   };
   return agx3_bind_compose(enc,AGX3_MN_uvst,0,fields,
                            sizeof(fields)/sizeof(fields[0]),NULL,0,out,length);
}

bool
agx3_mesh_passthrough_object(uint16_t grid_uniform, uint8_t out[64],
                             uint8_t *length)
{
   if (!out || !length || (grid_uniform & 3) || grid_uniform > 508)
      return false;
   uint8_t temp[64], size;unsigned n=0;
#define EMIT(name,ep,...) do { \
   const struct agx3_field_kv fields[]={__VA_ARGS__}; \
   if(!agx3_bind_compose(AGX_ISA_ENCODING_AGX3C,name,ep,fields, \
                         sizeof(fields)/sizeof(fields[0]),NULL,0,temp+n,&size)) \
      return false; \
   n+=size; \
} while(0)
   EMIT(AGX3_MN_rld,1,KV(9,2),KV(10,1),KV(11,1),KV(12,1),KV(21,13),
        KV(23,7),KV(24,1),KV(28,4),KV(29,144),KV(38,1));
   EMIT(AGX3_MN_clr,0,KV(2,1),KV(9,2),KV(10,1),KV(14,3),KV(15,25));
   EMIT(AGX3_MN_lddev,2,KV(9,2),KV(10,2),KV(12,8),KV(23,2),KV(24,2),
        KV(32,grid_uniform),KV(38,3),KV(39,3),KV(45,15),KV(53,1),
        KV(54,1),KV(59,104),KV(60,1),KV(73,1),KV(75,2),KV(82,2));
   EMIT(AGX3_MN_copy,1,KV(9,5),KV(10,2),KV(12,1),KV(23,2),KV(24,2),
        KV(27,1),KV(28,1),KV(29,35),KV(37,1));
   EMIT(AGX3_MN_uvsw,0,KV(9,5),KV(10,2),KV(12,1),KV(13,1),KV(24,1),
        KV(28,2),KV(29,196),KV(30,1));
   EMIT(AGX3_MN_end,0,KV(0,3),KV(1,39));
#undef EMIT
   for(unsigned i=0;i<n;i++)out[i]=temp[i];
   *length=n;return true;
}

bool
agx3_mesh_object_childcount(enum agx_isa_encoding enc, uint8_t out[8],
                             uint8_t *length)
{
   if (enc == AGX_ISA_ENCODING_AGX3B)
      return agx3_gen1_mesh_object_childcount(out, length);
   if (enc != AGX_ISA_ENCODING_AGX3C || !out || !length)
      return false;
   


   const struct agx3_field_kv fields[] = {
      KV(9,5), KV(10,2), KV(12,1), KV(13,1), KV(24,1),
      KV(28,2), KV(29,196), KV(30,1),
   };
   return agx3_bind_compose(enc,AGX3_MN_uvsw,0,fields,
      sizeof(fields)/sizeof(fields[0]),NULL,0,out,length);
}

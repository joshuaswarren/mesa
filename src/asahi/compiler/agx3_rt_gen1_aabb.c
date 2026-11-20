#include "agx3_rt_gen1_aabb.h"
#include "agx3_binder.h"
#include "agx3_alu_core.h"
#include "agx3_isel_core.h"
#include "agx3_cf_core.h"
#include "agx3_pack.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define CAPACITY 16384u
#include "agx3_rt_gen1_aabb_ops.h"
#define AABB_COUNT (sizeof(agx3_rt_gen1_aabb_ops)/sizeof(agx3_rt_gen1_aabb_ops[0]))
struct stream {uint8_t bytes[CAPACITY];size_t size;bool ok;};
static void append(struct stream *s,const uint8_t *bytes,unsigned n,const char *name,int original){(void)name;(void)original;if(!n||n>16||s->size+n>CAPACITY){s->ok=false;return;}memcpy(s->bytes+s->size,bytes,n);s->size+=n;}
#undef assert
#define assert(x) do { if (!(x)) { s->ok = false; return; } } while (0)
static void fselect(struct stream *s,unsigned dest,bool narrow,unsigned a,unsigned b,bool immediate,unsigned condition,unsigned t,bool ti,unsigned f,bool fi,unsigned wait,bool discard_source,const char *name){
 struct agx3_fsel_desc d={.dest=dest,.dest_16=narrow,.dest_cache=true,.cmp0=a,.cmp1=b,.cmp1_is_immediate=immediate,.sel0_kind=ti?AGX3_FSEL_SEL_IMMEDIATE:AGX3_FSEL_SEL_REGISTER,.sel1_kind=fi?AGX3_FSEL_SEL_IMMEDIATE:AGX3_FSEL_SEL_REGISTER,.sel0=t,.sel1=f,.sel_16=narrow,.condition=condition,.wait_mask=wait,.cmp0_hint=discard_source?AGX3_ALU_HINT_DISCARD:AGX3_ALU_HINT_NONE};
 uint8_t bytes[AGX3_ALU_MAX_LENGTH],n;assert(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B,&d,bytes,&n));append(s,bytes,n,name,-1);
}
static void state(struct stream *s,unsigned flag,unsigned lo,unsigned hi,unsigned origin,unsigned direction,unsigned wait,const char *axis){
 char name[80];snprintf(name,sizeof(name),"%s-origin-ge-min",axis);fselect(s,flag,true,origin,lo,false,5,1,true,2,true,wait,false,name);
 snprintf(name,sizeof(name),"%s-origin-le-max",axis);fselect(s,flag,true,origin,hi,false,6,flag,false,2,true,0,false,name);
 snprintf(name,sizeof(name),"%s-direction-zero",axis);fselect(s,flag,true,direction,0,true,0,flag,false,0,true,0,flag==9,name);
}
static void immediate(struct stream *s,unsigned dest,uint32_t value,const char *name){struct agx3_encoded_ldimm i;assert(agx3_pack_ldimm_u32(AGX_ISA_ENCODING_AGX3B,dest,value,&i));append(s,i.bytes,i.length,name,-1);}
static void guard(struct stream *s,unsigned flag,unsigned dest,bool upper,const char *name){
 for(unsigned state=1;state<=2;state++){
  unsigned source=(upper==(state==1))?48:50;
  struct agx3_isel_desc d={.dest=dest,.cmp0=flag,.cmp1=state,.cond=AGX3_ISEL_ICOND_EQ,.cmp1_immediate=true,.cmp_16=true,.explicit_hints=true,.dest_cache=true,.true_kind=AGX3_ISEL_REGISTER,.false_kind=AGX3_ISEL_REGISTER,.sel_true=source,.sel_false=dest};
  uint8_t bytes[AGX3_ISEL_MAX_LENGTH],n;assert(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B,&d,bytes,&n));append(s,bytes,n,name,-1);
 }
}
static void body(struct stream *s,bool corrected){
 size_t positions[AABB_COUNT];
 for(unsigned j=0;j<AABB_COUNT;j++){
  const struct agx3_rt_gen1_native_op *op=&agx3_rt_gen1_aabb_ops[j];positions[j]=s->size;
  
  uint8_t bytes[16],n;
  if(op->target!=UINT32_MAX){assert(agx3_cf_branch(AGX_ISA_ENCODING_AGX3B,true,bytes,&n));}
  else{
   struct agx3_field_kv fields[48];memcpy(fields,op->fields,sizeof(fields));
   if(corrected&&op->offset==380){assert(agx3_alu_f32rcp(AGX_ISA_ENCODING_AGX3B,48,12,AGX3_ALU_HINT_NONE,true,63,bytes,&n));}
   else {
    if(corrected&&op->offset==652){bool found=false;for(unsigned k=0;k<op->nfields;k++)if(fields[k].field==110){assert(fields[k].value==1);fields[k].value=6;found=true;}assert(found);}
    assert(agx3_bind_compose(AGX_ISA_ENCODING_AGX3B,op->name,op->ep,fields,op->nfields,NULL,0,bytes,&n));
   }
  }
  assert(n==op->size);append(s,bytes,n,agx3_mnemonic_name(op->name),op->offset);
  if(corrected){
   switch(op->offset){
   case 380:state(s,9,0,6,10,12,0,"x");state(s,57,2,36,14,16,0,"y");state(s,59,4,38,18,20,0,"z");break;
   case 488:immediate(s,48,UINT32_C(0x7f800000),"positive-infinity");immediate(s,50,UINT32_C(0xff800000),"negative-infinity");guard(s,9,4,false,"x-near");guard(s,9,16,true,"x-far");break;
   case 518:guard(s,57,6,true,"y-far");break;
   case 540:guard(s,57,2,false,"y-near");break;
   case 534:guard(s,59,6,true,"z-far");break;
   case 564:guard(s,59,0,false,"z-near");break;
   default:break;
   }
  }
 }
 for(unsigned j=0;j<AABB_COUNT;j++)if(agx3_rt_gen1_aabb_ops[j].target!=UINT32_MAX){size_t at=positions[j],target=positions[agx3_rt_gen1_aabb_ops[j].target];assert(agx3_cf_patch_branch(AGX_ISA_ENCODING_AGX3B,true,s->bytes+at,(int64_t)target-(int64_t)at));}
}
#undef assert
bool
agx3_rt_gen1_aabb_body_generate(uint8_t *out, size_t capacity, size_t *length)
{
   static const uint8_t prefix[4] = {0x0e, 0, 0, 0};
   struct stream s = {.ok = true};
   body(&s, true);
   if (!s.ok || 64 + s.size > capacity) return false;
   memset(out, 0, 64);
   memcpy(out, prefix, sizeof(prefix));
   for (unsigned i = 2; i < 32; ++i) out[2 * i] = 0x06;
   memcpy(out + 64, s.bytes, s.size);
   *length = 64 + s.size;
   return true;
}
static const uint8_t aabb_constants[56] = {0xff,0xff,0xff,0xff,0x00,0x80,0x00,0x10,0x00,0x02,0x00,0x08,0x00,0x00,0xe0,0x01,0x00,0x00,0xc0,0x7f,0xe0,0x0f,0xff,0x0f,0x00,0x18,0x00,0x20,0x00,0x28,0x00,0xc0,0x00,0x00,0x00,0x00,0x00,0x00,0x20,0x00,0x00,0x00,0x40,0x00,0x00,0x00,0x00,0x40,0x00,0x00,0x00,0x80,0xff,0xff,0x1f,0xfe};
void
agx3_rt_gen1_aabb_constants(const uint8_t **data, size_t *length)
{
   *data = aabb_constants;
   *length = sizeof(aabb_constants);
}

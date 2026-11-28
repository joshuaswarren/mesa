/* Encode and compile the AGX3C SCS memory forms. */
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_nir.h"
#include "agx3_scs_core.h"
#include "agx3_sr_core.h"
#include "agx3_pixel_core.h"
#include "agx3_graphics_program.h"
#include "asahi/lib/agx3_compute_program_core.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/glsl_types.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void write_file(const char *name,const void *data,size_t size)
{
 FILE *f=fopen(name,"wb");assert(f);assert(fwrite(data,1,size,f)==size);assert(!fclose(f));
}

/* mode0 direct, mode1 uniform indexed, mode2 register base, mode3 status. */
static void compile(unsigned mode,unsigned count,unsigned width,unsigned flags,
                    const char *dir)
{
 nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,&agx_nir_options,"M5 query SCS forms");
 b.shader->info.workgroup_size[0]=64;
 b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
 nir_def *id=nir_channel(&b,nir_load_global_invocation_id(&b,32),0);
 nir_def *address=nir_iadd_imm(&b,nir_u2u64(&b,nir_ishl_imm(&b,id,2)),0x1000001cd00ull);
 nir_def *input=nir_load_global(&b,1,32,address,.align_mul=4);
 nir_def *expected=nir_iadd_imm(&b,nir_imul_imm(&b,input,3),5);
 nir_def *base=mode==1?nir_load_preamble(&b,1,64,.base=122):nir_ishl_imm(&b,id,4);
 enum pipe_format format=width==8?PIPE_FORMAT_R8_UINT:width==16?PIPE_FORMAT_R16_UINT:PIPE_FORMAT_R32_UINT;
 const unsigned result_bits=(width==32||(flags&2))?32:16;
 nir_def *components[4];
 for(unsigned c=0;c<count;c++) {
  nir_def *v=nir_iadd_imm(&b,expected,17*c);
  components[c]=width==32?v:nir_u2u16(&b,v);
 }
 nir_def *data=nir_vec(&b,components,count);
 if(mode==0)nir_push_if(&b,nir_ieq_imm(&b,id,0));
 nir_def *values;
 if(mode==0) {
  nir_store_scs_agx(&b,data,.base=0x6000);
  values=nir_load_scs_agx(&b,count,result_bits,.base=0x6000);
 } else if(mode==1) {
  nir_def *index=nir_ishl_imm(&b,id,4);
  nir_store_scs_uniform_agx(&b,data,base,index,.base=0x6000,.format=format,.component=0);
  values=nir_load_scs_uniform_agx(&b,count,result_bits,base,index,.base=0x6000,.format=format,.component=0,.flags=flags);
 } else {
  nir_store_scs_base_agx(&b,data,base,.base=0x6000,.format=format,.component=4);
  values=nir_load_scs_base_agx(&b,count,result_bits,base,.base=0x6000,.format=format,.component=4,.flags=flags);
 }
 if(getenv("M5_QUERY_RAW") && mode==1 && (flags&2)) {
  nir_def *raw_address=nir_iadd_imm(&b,nir_u2u64(&b,nir_ishl_imm(&b,id,2)),0x1000001ed00ull);
  nir_store_global(&b,nir_u2u32(&b,values),raw_address,.align_mul=4,.write_mask=1);
 }
 nir_def *valid=nir_imm_true(&b);
 for(unsigned c=0;c<count;c++) {
  nir_def *v=nir_u2u32(&b,nir_channel(&b,values,c));
  nir_def *e=nir_u2u32(&b,components[c]);
  if(width<32) {
   unsigned mask=(1u<<width)-1;
   e=nir_iand_imm(&b,e,mask);
   if(!(flags&2))v=nir_iand_imm(&b,v,mask);
  }
  valid=nir_iand(&b,valid,nir_ieq(&b,v,e));
 }
 nir_def *result=nir_bcsel(&b,valid,expected,nir_imm_int(&b,0xbad0bad0));
 if(mode==0) {
  nir_push_else(&b,NULL);
  nir_pop_if(&b,NULL);
  result=nir_if_phi(&b,result,expected);
 }
 address=nir_iadd_imm(&b,nir_u2u64(&b,nir_ishl_imm(&b,id,2)),0x1000001dd00ull);
 nir_store_global(&b,result,address,.align_mul=4,.write_mask=1);
 agx_preprocess_nir(b.shader);
 struct agx_shader_key key={.dev={.chip=AGX_CHIP_G17S,.encoding=AGX_ISA_ENCODING_AGX3C},.no_preamble=true};
 struct agx_shader_part part={0};
 bool ok=agx_compile_shader_nir(b.shader,&key,&part);
 printf("mode=%u count=%u width=%u flags=%u %s %s\n",mode,count,width,flags,ok?"PASS":"FAIL",ok?"":agx_last_compile_refusal());
 assert(ok&&part.info.uses_ray_traversal&&!part.info.ray_traversal_depth_nonzero);
 unsigned stores=0,loads=0;
 for(size_t off=part.info.main_offset;off<part.info.main_offset+part.info.main_size;off++)
  for(unsigned n=2;n<=16&&off+n<=part.info.main_offset+part.info.main_size;n+=2) {
   struct agx3_scs_desc d;bool load=agx3_scs_decode_load(AGX_ISA_ENCODING_AGX3C,part.binary+off,n,&d);
   bool store=!load&&agx3_scs_decode_store(AGX_ISA_ENCODING_AGX3C,part.binary+off,n,&d);
   if(!load&&!store)continue;
   assert(d.address_kind==(mode==0?AGX3_SCS_ADDRESS_DIRECT:mode==1?AGX3_SCS_ADDRESS_UNIFORM_INDEXED:AGX3_SCS_ADDRESS_REGISTER_BASE));
   assert(d.component_mask==((1u<<count)-1)&&d.byte_offset==0x6000);
   assert(d.width==(width==8?AGX3_MEM_W8:width==16?AGX3_MEM_W16:AGX3_MEM_W32));
   if(load)assert(d.zero_extend==!!(flags&2)&&d.c_load_marker==!!(flags&1)&&d.c_status_load==!!(flags&4));
   stores+=store;loads+=load;
  }
 assert(stores==1&&loads==1);
 if(dir) {
  char name[512];snprintf(name,sizeof(name),"%s/form-%u-%u-%u-%u.bin",dir,mode,count,width,flags);
  write_file(name,part.binary+part.info.main_offset,part.info.main_size);
  struct agx3_ioto_uniform_range range={.start_halfs=122,.size_halfs=4,.address=0x10000090000ull};
  struct agx3_g17p_graphics_program_desc desc={.program_va=0x10000000000ull,.shader_va=0x10000003000ull,
   .usc_clusters=2, /* the M5 Pro's lead */
   .uniform_ranges=mode==1?&range:NULL,.uniform_range_count=mode==1,
   .shader_binary_size=part.info.main_size,.nr_gprs=part.info.nr_gprs,
   .scratch_size=MAX2(part.info.scratch_size,part.info.agx3_spectpr_scratch_size),
   .compiler_d_valid=part.info.agx3_staging_known,.compiler_d=(uint32_t)part.info.agx3_staging_halfs*2};
  struct agx3_g17p_graphics_program program;assert(agx3_g17g_compute_program_generate(AGX_CHIP_G17S,&desc,&program));
  snprintf(name,sizeof(name),"%s/form-%u-%u-%u-%u.program",dir,mode,count,width,flags);write_file(name,program.bytes,program.length);
  struct agx3_g17p_native_compute_program_desc ld={.program_va=desc.program_va,.shader_va=desc.shader_va};
  assert(agx3_cdm_direct_template_sources_compute(false,true,false,&ld.launch_sources));
  struct agx3_compute_program launch;assert(agx3_m5_native_compute_program_bind(AGX_CHIP_G17S,&ld,&launch));
  uint8_t cdm[48]={0};struct agx3_compute_dispatch_sink sink={.data=cdm,.capacity=44};
  const struct agx3_compute_direct_grid grid={.source=AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,.size={1024,1,1},.local_size={64,1,1},.dimensions=1};
  assert(agx3_m5_native_compute_dispatch_append(&launch,&grid,0,&sink));assert(sink.length==44);cdm[47]=0x40;
  snprintf(name,sizeof(name),"%s/form-%u-%u-%u-%u.cdm",dir,mode,count,width,flags);write_file(name,cdm,48);
  printf("  bytes=%u gprs=%u loader=%u\n",part.info.main_size,part.info.nr_gprs,program.length);
 }
 free(part.binary);ralloc_free(b.shader);
}
static void refused(unsigned format,unsigned flags,unsigned selector,int offset)
{
 nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,&agx_nir_options,"invalid SCS form");
 nir_def *v=nir_load_scs_base_agx(&b,1,16,nir_imm_int(&b,0),.base=offset,
                                .format=format,.flags=flags,.component=selector);
 nir_store_global(&b,v,nir_imm_int64(&b,0x10000000000ull),.align_mul=4,.write_mask=1);
 agx_preprocess_nir(b.shader);
 struct agx_shader_key key={.dev={.chip=AGX_CHIP_G17S,.encoding=AGX_ISA_ENCODING_AGX3C},.no_preamble=true};
 struct agx_shader_part part={0};
 assert(!agx_compile_shader_nir(b.shader,&key,&part)&&!part.binary);
 ralloc_free(b.shader);
}

static void
instance_completion_byte(void)
{
 nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,&agx_nir_options,"instance completion byte");
 nir_def *base=nir_load_global(&b,1,32,nir_imm_int64(&b,0x10000),.align_mul=4);
 nir_def *lane=nir_load_global(&b,1,32,nir_imm_int64(&b,0x10004),.align_mul=4);
 nir_def *v=nir_load_scs_register_agx(&b,1,16,base,lane,
    .base=263,.format=PIPE_FORMAT_R8_UINT,.component=3,.flags=9);
 nir_store_global(&b,nir_u2u32(&b,v),nir_imm_int64(&b,0x20000),.align_mul=4,.write_mask=1);
 agx_preprocess_nir(b.shader);
 struct agx_shader_key key={.dev={.chip=AGX_CHIP_G17S,.encoding=AGX_ISA_ENCODING_AGX3C},.no_preamble=true};
 struct agx_shader_part part={0};assert(agx_compile_shader_nir(b.shader,&key,&part));
 unsigned found=0;
 for(size_t off=part.info.main_offset;off<part.info.main_offset+part.info.main_size;off++)
  for(unsigned n=2;n<=16&&off+n<=part.info.main_offset+part.info.main_size;n+=2){
   struct agx3_scs_desc d;if(!agx3_scs_decode_load(AGX_ISA_ENCODING_AGX3C,part.binary+off,n,&d))continue;
   assert(d.address_kind==AGX3_SCS_ADDRESS_REGISTER_INDEXED&&d.width==AGX3_MEM_W8&&
          d.byte_offset==263&&d.address_selector==3&&d.component_mask==1&&d.c_load_marker&&(d.wait_mask&0x80));
   found++;
  }
 assert(found==1);free(part.binary);ralloc_free(b.shader);
 puts("M5_INSTANCE_COMPLETION_BYTE_PASS");
}

/* The SR59/W16 special-register read (rld h28). */
static void
scs_thread_identity(void)
{
 struct agx3_sr_desc d={.dest=28,.sr=59,.width=AGX3_SR_W16,.slot=3};
 const uint8_t native[]={0xe4,0xea,0x10,0x46};uint8_t out[16],len=0;
 assert(agx3_sr_encode(AGX_ISA_ENCODING_AGX3C,&d,out,&len));
 assert(len==sizeof(native)&&!memcmp(out,native,len));
 assert(agx3_sr_encode(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
 puts("M5_NATIVE_SCS_THREAD_ID_BYTES_PASS");
}

int main(int argc,char **argv)
{
 setbuf(stdout,NULL);glsl_type_singleton_init_or_ref();const char *out=argc>1?argv[1]:NULL;
 scs_thread_identity();
 instance_completion_byte();
 for(unsigned mode=0;mode<3;mode++)for(unsigned n=1;n<=4;n++)compile(mode,n,32,0,out);
 compile(0,1,16,0,out);
 for(unsigned mode=1;mode<=2;mode++)for(unsigned width=8;width<=16;width+=8)compile(mode,1,width,0,out);
 for(unsigned width=8;width<=16;width+=8)compile(1,1,width,2,out);
 /* The status conversion is unknown: exact encoding only, never run using
  * fabricated query state. */
 compile(3,1,8,5,NULL);
 refused(PIPE_FORMAT_R8_UINT,4,4,0);  /* status requires marker */
 refused(PIPE_FORMAT_R16_UINT,5,4,0); /* status is byte-only */
 refused(PIPE_FORMAT_R16_UINT,0,5,0); /* invalid selector */
 refused(PIPE_FORMAT_R16_UINT,0,4,1); /* unaligned access */
 refused(PIPE_FORMAT_R32_FLOAT,0,4,0); /* unsupported memory format */
 printf("Invalid SCS forms rejected transactionally\n");
 glsl_type_singleton_decref();return 0;
}

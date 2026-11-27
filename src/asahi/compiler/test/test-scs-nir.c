/* Explicit NIR SCS compiler qualification, independent of API ray queries. */
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx3_graphics_program.h"
#include "agx3_scs_core.h"
#include "agx3_rt_m5_core.h"
#include "asahi/lib/agx3_compute_program_core.h"
#include "agx_nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/glsl_types.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool compile(unsigned mode, unsigned variant, unsigned width,
                    int offset, enum agx_isa_encoding encoding,
                    const char *out)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                 &agx_nir_options, "M5 NIR SCS");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   nir_def *id = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *in_addr = nir_iadd_imm(&b, nir_u2u64(&b, nir_ishl_imm(&b,id,2)), 0x1000001cd00ull);
   nir_def *input = nir_load_global(&b, 1, 32, in_addr, .align_mul=4);
   nir_def *value = nir_iadd_imm(&b, nir_imul_imm(&b,input,3),5);
   nir_def *result;
   if (mode == 0) {
      nir_def *v = nir_u2uN(&b, value, width);
      nir_store_scs_agx(&b, v, .base=offset);
      result = nir_u2u32(&b, nir_load_scs_agx(&b,1,width,.base=offset));
   } else if (mode == 1 || mode == 3) {
      nir_def *v[4];
      for(unsigned i=0;i<variant;i++) v[i]=nir_iadd_imm(&b,value,i*17);
      nir_def *base = nir_ishl_imm(&b,id,4);
      /* Keep both address operands live in registers. */
      nir_def *index = nir_iand_imm(&b,input,0);
      nir_store_scs_indexed_agx(&b,nir_vec(&b,v,variant),base,index,
                                .base=offset,.component=4);
      nir_def *loaded = nir_load_scs_indexed_agx(&b,variant, 32,base,index,
                                               .base=offset,.component=4);
      nir_def *valid=nir_imm_true(&b);
      for(unsigned i=0;i<variant;i++)
         valid=nir_iand(&b,valid,nir_ieq(&b,nir_channel(&b,loaded,i),v[i]));
      if (mode == 3) {
         for(unsigned i=0;i<variant;i++) v[i]=nir_iadd_imm(&b,v[i],33);
         nir_store_scs_indexed_agx(&b,nir_vec(&b,v,variant),base,index,
                                   .base=offset,.component=4);
         nir_def *again=nir_load_scs_indexed_agx(&b,variant, 32,base,index,
                                               .base=offset,.component=4);
         for(unsigned i=0;i<variant;i++)
            valid=nir_iand(&b,valid,nir_ieq(&b,nir_channel(&b,again,i),v[i]));
      }
      result=nir_bcsel(&b,valid,value,nir_imm_int(&b,0xbad0bad0));
   } else {
      nir_enter_query_agx(&b);
      nir_intersect_ray_agx(&b,nir_imm_int64(&b,0x100),nir_imm_int64(&b,0x200),
                            nir_imm_int(&b,0x10000000),nir_imm_int(&b,0),id);
      nir_continue_ray_agx(&b, id);
      result=value;
   }
   nir_def *out_addr = nir_iadd_imm(&b,nir_u2u64(&b,nir_ishl_imm(&b,id,2)),0x1000001dd00ull);
   nir_store_global(&b,result,out_addr,.align_mul=4,.write_mask=1);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key={.dev={.chip=encoding == AGX_ISA_ENCODING_AGX3B ? AGX_CHIP_G17P : AGX_CHIP_G17S,.encoding=encoding},
                              .no_preamble=true};
   struct agx_shader_part part={0};
   bool ok=agx_compile_shader_nir(b.shader,&key,&part);
   printf("mode=%u vec=%u width=%u offset=%d enc=%u %s %s\n",mode,variant,width,offset,encoding,ok?"PASS":"REFUSED",ok?"":agx_last_compile_refusal());
   if(ok) {
      assert(part.info.uses_ray_traversal);
      assert(!part.info.ray_traversal_depth_nonzero);
      unsigned loads=0,stores=0,entries=0;
      /* The query entry (incStk, allocation index 4) writes its query pointer
       * to a register-allocated f2 destination, as native does (witnesses
       * 7f04400000000000 = f2 46, 3f04000000000000 = f2 6); it no longer
       * writes a fixed r0 (0f04000000000000). Accept any allocated half. */
      for(size_t i=part.info.main_offset;i<part.info.main_offset+part.info.main_size;i++) {
         for(unsigned dest=0;dest<=252 && i+8<=part.info.main_offset+part.info.main_size;dest+=2) {
            uint8_t entry[16],entry_len=0;
            if(agx3_rt_m5_encode_enter_query(AGX_ISA_ENCODING_AGX3C,dest,4,entry,&entry_len) &&
               entry_len==8 && !memcmp(part.binary+i,entry,8)) { entries++; break; }
         }
         for(unsigned n=2;n<=AGX3_SCS_MAX_LENGTH && i+n<=part.info.main_offset+part.info.main_size;n+=2) {
            struct agx3_scs_desc d;
            bool load=agx3_scs_decode_load(encoding,part.binary+i,n,&d);
            bool store=!load && agx3_scs_decode_store(encoding,part.binary+i,n,&d);
            if(!load && !store)continue;
            assert(d.byte_offset==offset);
            if(mode==1 || mode==3) {
               assert(d.address_kind==AGX3_SCS_ADDRESS_REGISTER_INDEXED);
               assert(d.component_mask==((1u<<variant)-1));
               assert(d.address_selector==4);
            }
            loads+=load;stores+=store;
         }
      }
      assert(loads==(mode==2?0:mode==3?2:1));
      assert(stores==loads);
      assert(entries==(mode==2 ? 1 : 0));
      if(out){FILE *f=fopen(out,"wb");assert(f);assert(fwrite(part.binary+part.info.main_offset,1,part.info.main_size,f)==part.info.main_size);assert(!fclose(f));printf("bytes=%u gprs=%u\n",part.info.main_size,part.info.nr_gprs);
         struct agx3_g17p_graphics_program_desc desc={
            .program_va=0x10000000000ull, .shader_va=0x10000003000ull,
            .usc_clusters=2, /* the M5 Pro's lead */
            .shader_binary_size=part.info.main_size,.nr_gprs=part.info.nr_gprs,
            .scratch_size=MAX2(part.info.scratch_size, part.info.agx3_spectpr_scratch_size),
            .compiler_d_valid=part.info.agx3_staging_known,
            .compiler_d=(uint32_t)part.info.agx3_staging_halfs*2,
         };
         struct agx3_g17p_graphics_program program;
         assert(agx3_g17g_compute_program_generate(AGX_CHIP_G17S,&desc,&program));
         char file[512];snprintf(file,sizeof(file),"%s.program",out);
         f=fopen(file,"wb");assert(f);assert(fwrite(program.bytes,1,program.length,f)==program.length);assert(!fclose(f));
         struct agx3_g17p_native_compute_program_desc launch_desc={
            .program_va=desc.program_va,.shader_va=desc.shader_va,
         };
         assert(agx3_cdm_direct_template_sources_compute(false,true,false,&launch_desc.launch_sources));
         struct agx3_compute_program launch;
         assert(agx3_m5_native_compute_program_bind(AGX_CHIP_G17S,&launch_desc,&launch));
         uint8_t cdm[48]={0};struct agx3_compute_dispatch_sink sink={.data=cdm,.capacity=44};
         const struct agx3_compute_direct_grid grid={.source=AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,.size={1024,1,1},.local_size={64,1,1},.dimensions=1};
         assert(agx3_m5_native_compute_dispatch_append(&launch,&grid,0,&sink));
         assert(sink.length==44);cdm[47]=0x40;
         snprintf(file,sizeof(file),"%s.cdm",out);f=fopen(file,"wb");assert(f);assert(fwrite(cdm,1,48,f)==48);assert(!fclose(f));
}
   } else assert(!part.binary);
   free(part.binary);ralloc_free(b.shader);return ok;
}
int main(int argc,char **argv) {
   setbuf(stdout,NULL);
   glsl_type_singleton_init_or_ref();
   for(unsigned m=0;m<4;m++)for(unsigned n=1;n<=4;n*=2) {
      if(m==2)continue;
      if(!m&&n!=1)continue;
      char path[256];snprintf(path,sizeof(path),"%s/scs-%u-%u.bin",argc>1?argv[1]:".",m,n);
      assert(compile(m,n,32,0x6000,AGX_ISA_ENCODING_AGX3C,argc>1?path:NULL));
   }
   assert(compile(0,1,16,-128,AGX_ISA_ENCODING_AGX3C,NULL));
   assert(compile(2,1,32,0,AGX_ISA_ENCODING_AGX3C,NULL));
   assert(compile(1,3,32,0,AGX_ISA_ENCODING_AGX3C,NULL));
   assert(!compile(1,4,32,32768,AGX_ISA_ENCODING_AGX3C,NULL));
   assert(compile(1,4,32,0,AGX_ISA_ENCODING_AGX3B,NULL));
   assert(!compile(1,4,32,0,AGX_ISA_ENCODING_AGX2,NULL));
   glsl_type_singleton_decref();return 0;
}

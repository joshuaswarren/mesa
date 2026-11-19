#include "agx_test.h"
#include <gtest/gtest.h>
extern "C" {
#include "agx3_static_spill_core.h"
#include "agx3_alu_core.h"
}
class CProgCertificate : public testing::Test {
protected:
   void *memctx=ralloc_context(NULL);
   agx_builder *b=agx_test_builder(memctx);
   agx_shader_key key={};
   agx_shader_info info={};
   CProgCertificate() {
      key.dev.encoding=AGX_ISA_ENCODING_AGX3B;
      b->shader->key=&key;b->shader->out=&info;
      b->shader->nir=rzalloc(memctx,nir_shader);
      b->shader->nir->info.stage=MESA_SHADER_VERTEX;
      b->shader->max_reg=7;
      key.g17p_native_uniforms=true;key.agx3_native_cprog=true;
      b->shader->stage=info.stage=MESA_SHADER_VERTEX;
      b->shader->is_preamble=true;b->shader->agx3_cprog_inputs_lowered=true;
      info.has_preamble=true;info.agx3_preamble_uniform_inputs=true;
      info.agx3_preamble_tps_inputs=true;
      uint64_t masks[8]={UINT64_C(0xf0000)};
      EXPECT_TRUE(agx3_cprog_inputs_init(masks,&info.agx3_preamble_inputs));
      b->shader->scratch_size_B=32;
   }
   ~CProgCertificate(){ralloc_free(memctx);}
   agx_instr *import() { return agx_load_cprog_agx3_to(b,agx_register(0,AGX_SIZE_32),16); }
   struct Ops { agx_instr *store,*load; };
   Ops spill(unsigned frame=36,unsigned off=32,bool half=false) {
      b->shader->spill_base_B=32;b->shader->scratch_size_B=frame;
      info.preamble_scratch_size=align(frame,4);
      auto size=half?AGX_SIZE_16:AGX_SIZE_32;
      auto format=half?AGX_FORMAT_I16:AGX_FORMAT_I32;
      return {agx_stack_store(b,agx_register(0,size),agx_immediate(off),format,1),
              agx_stack_load_to(b,agx_register(4,size),agx_immediate(off),format,1)};
   }
   bool certify(bool provenance=true) {return agx3_cprog_certify_static_spill(b->shader,provenance);}
};
TEST_F(CProgCertificate, ImportsOnlyBeforeAllocator) {
   import();EXPECT_TRUE(agx3_cprog_spill_provenance(b->shader));
   spill();EXPECT_FALSE(agx3_cprog_spill_provenance(b->shader));
   EXPECT_TRUE(certify());EXPECT_FALSE(certify(false));
}
TEST_F(CProgCertificate, RejectsExistingLocalReservation) {
   import();b->shader->scratch_size_B=48;EXPECT_FALSE(agx3_cprog_spill_provenance(b->shader));
   spill();b->shader->spill_base_B=48;EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, RejectsAnyScratchFlag) {
   import();b->shader->any_scratch=true;EXPECT_FALSE(agx3_cprog_spill_provenance(b->shader));
   spill();EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, ExactFinalFrame) {
   import();spill();EXPECT_TRUE(certify());
   info.preamble_scratch_size=40;EXPECT_FALSE(certify());
   info.preamble_scratch_size=32;EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, StoreCannotOverwriteImports) {
   import();auto ops=spill();ops.store->src[1]=agx_immediate(28);EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, ReloadCannotReadImportsAsSpill) {
   import();auto ops=spill();ops.load->src[0]=agx_immediate(16);EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, MisalignedAndDynamicOffsetsRejected) {
   import();auto ops=spill();ops.store->src[1]=agx_immediate(33);EXPECT_FALSE(certify());
   ops.store->src[1]=agx_register(8,AGX_SIZE_32);EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, EndOf4096ByteFrame) {
   import();auto ops=spill(4096,4092);EXPECT_FALSE(certify());
   ops.load->src[0]=agx_immediate(4096);EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, LastHalfOf4096ByteFrame) {
   import();auto ops=spill(4096,4094,true);EXPECT_FALSE(certify());
   ops.store->src[1]=agx_immediate(4096);EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, ExceedingFrameLimitRejected) {
   import();spill(4100,4096);EXPECT_FALSE(certify());
   b->shader->scratch_size_B=UINT32_MAX;EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, HalfEndingAtUnalignedRawFrame) {
   import();spill(34,32,true);EXPECT_TRUE(certify());
}
TEST_F(CProgCertificate, VectorAndFormatRejectedBeforeScalarLowering) {
   import();auto ops=spill();ops.store->mask=3;EXPECT_FALSE(certify());
   ops.store->mask=1;ops.load->format=AGX_FORMAT_I8;EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, ImportRangeAndContiguity) {
   auto *i=import();spill();i->imm=19;EXPECT_FALSE(certify());
   i->imm=17;EXPECT_FALSE(certify());i->imm=16;EXPECT_TRUE(certify());
}
TEST_F(CProgCertificate, ImportsAndOutputsMustStayDisjoint) {
   import();spill();auto *w=agx_uniform_store(b,agx_register(4,AGX_SIZE_32),agx_immediate(16),1);
   EXPECT_FALSE(certify());w->src[1]=agx_immediate(64);EXPECT_TRUE(certify());
}
TEST_F(CProgCertificate, MissingImportsStoreOrLoadRejected) {
   auto ops=spill();EXPECT_FALSE(certify());import();EXPECT_TRUE(certify());
   list_del(&ops.store->link);EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, NoImportAndMetadataMismatchRejected) {
   import();spill();info.agx3_preamble_inputs.count=0;EXPECT_FALSE(certify());
   info.agx3_preamble_inputs.count=4;info.agx3_preamble_inputs.storage_bytes=48;EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, OtherStageAndEncodingRejected) {
   import();spill();b->shader->stage=MESA_SHADER_COMPUTE;EXPECT_FALSE(certify());
   b->shader->stage=MESA_SHADER_VERTEX;key.dev.encoding=AGX_ISA_ENCODING_AGX2;EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, HiddenTPSAndOtherStorageRejected) {
   import();spill();const agx_opcode ops[]={AGX_OPCODE_ATOMIC,AGX_OPCODE_LOCAL_ATOMIC,
      AGX_OPCODE_LOCAL_LOAD,AGX_OPCODE_LOCAL_STORE,AGX_OPCODE_STACK_MAP,AGX_OPCODE_STACK_UNMAP,
      AGX_OPCODE_STACK_ADJUST,AGX_OPCODE_SCS_LOAD_AGX3,AGX_OPCODE_SCS_STORE_AGX3,
      AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3,AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3,
      AGX_OPCODE_SCS_LOAD_REGISTER_INDEXED_AGX3,AGX_OPCODE_SCS_STORE_REGISTER_INDEXED_AGX3};
   for(auto op:ops){auto *i=agx_alloc_instr(b,op,0,0);agx_builder_insert(&b->cursor,i);
      EXPECT_FALSE(certify())<<op;list_del(&i->link);b->cursor=agx_after_block(agx_start_block(b->shader));}
}
TEST_F(CProgCertificate, RuntimeCertificateVersionAndMetadata) {
   info.agx3_preamble_static_spill_version=AGX3_CPROG_STATIC_SPILL_VERSION;
   info.preamble_scratch_size=36;EXPECT_TRUE(agx3_cprog_static_spill_info_valid(&info));
   info.agx3_preamble_static_spill_version=0;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.agx3_preamble_static_spill_version=2;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.agx3_preamble_static_spill_version=1;info.stage=MESA_SHADER_FRAGMENT;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.stage=MESA_SHADER_VERTEX;info.preamble_scratch_size=32;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.preamble_scratch_size=188;EXPECT_TRUE(agx3_cprog_static_spill_info_valid(&info));
   info.preamble_scratch_size=4096;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.preamble_scratch_size=4100;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.preamble_scratch_size=37;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.preamble_scratch_size=36;info.agx3_preamble_inputs.halves[0]=0;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
}

TEST_F(CProgCertificate, PackBoundaryDependencyChain) {
   import();ASSERT_TRUE(agx3_cprog_spill_provenance(b->shader));
   spill(4096,4092);
   agx_stack_store(b,agx_register(4,AGX_SIZE_32),agx_immediate(4092),AGX_FORMAT_I32,1);
   agx_stack_load_to(b,agx_register(6,AGX_SIZE_32),agx_immediate(4092),AGX_FORMAT_I32,1);
   agx_uniform_store(b,agx_register(6,AGX_SIZE_32),agx_immediate(64),1);agx_stop(b);
   ASSERT_FALSE(certify());
   struct util_dynarray binary=UTIL_DYNARRAY_INIT;
   ASSERT_TRUE(agx3_pack_binary(b->shader,&binary));
   ASSERT_FALSE(certify());
   /* Decode actual COPY/RLD fields, including dependencies at the final
    * word of the 4096-byte frame. Each producer must be retired before reuse. */
   const uint8_t *code=(const uint8_t *)binary.data;
   ASSERT_GE(binary.size,44u);
   agx3_static_spill_load_desc loads[3]={};
   agx3_static_spill_store_desc stores[2]={};
   ASSERT_TRUE(agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3B,code,8,&loads[0]));
   ASSERT_TRUE(agx3_static_spill_decode_store(AGX_ISA_ENCODING_AGX3B,code+8,10,&stores[0]));
   ASSERT_TRUE(agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3B,code+18,8,&loads[1]));
   ASSERT_TRUE(agx3_static_spill_decode_store(AGX_ISA_ENCODING_AGX3B,code+26,10,&stores[1]));
   ASSERT_TRUE(agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3B,code+36,8,&loads[2]));
   EXPECT_EQ(loads[0].tps_half,8u);EXPECT_NE(loads[0].wait_mask&32u,0u);
   for(unsigned i=0;i<2;i++) {
      EXPECT_EQ(stores[i].tps_half,2046u);EXPECT_EQ(loads[i+1].tps_half,2046u);
      EXPECT_NE(stores[i].wait_mask&(1u<<(loads[i].slot-1)),0u);
      EXPECT_NE(loads[i+1].wait_mask&(1u<<(stores[i].slot-1)),0u);
   }
   EXPECT_TRUE(loads[1].tps_live);EXPECT_FALSE(loads[2].tps_live);
   /* The final RLD is retired by the existing identity carrier before its
    * uniform export. Match the complete carrier with that exact wait mask;
    * this is an encoder-backed byte check, not a generic ALU decoder. */
   uint8_t carrier[AGX3_ALU_MAX_LENGTH],length=0;
   ASSERT_TRUE(agx3_alu_band_identity_wait(AGX_ISA_ENCODING_AGX3B,6,
      1u<<(loads[2].slot-1),carrier,&length));
   ASSERT_GE(binary.size,44u+length);
   EXPECT_EQ(memcmp(code+44,carrier,length),0);
   const char *output=getenv("CPROG_CERT_BOUNDARY_OUTPUT");
   if(output) {
      FILE *f=fopen(output,"wb");ASSERT_NE(f,nullptr);
      ASSERT_EQ(fwrite(binary.data,1,binary.size,f),binary.size);fclose(f);
   }
   util_dynarray_fini(&binary);
}

/* Two 16-bit imports into the halves of one register, high half first: the
 * second RLD must wait for the first (G17P loses a half when both are in
 * flight). Halves of different registers stay independent. */
static void
pack_half_imports(agx_builder *b, unsigned first_dest, unsigned second_dest,
                  agx3_static_spill_load_desc loads[2])
{
   agx_load_cprog_agx3_to(b,agx_register(first_dest,AGX_SIZE_16),17);
   agx_load_cprog_agx3_to(b,agx_register(second_dest,AGX_SIZE_16),16);
   agx_uniform_store(b,agx_register(0,AGX_SIZE_32),agx_immediate(64),1);
   agx_uniform_store(b,agx_register(2,AGX_SIZE_32),agx_immediate(66),1);
   agx_stop(b);
   struct util_dynarray binary=UTIL_DYNARRAY_INIT;
   ASSERT_TRUE(agx3_pack_binary(b->shader,&binary));
   /* An RLD is 8 bytes, or 4 in its compact form (no wait). */
   size_t offset=0;
   for(unsigned i=0;i<2;i++) {
      const uint8_t *code=(const uint8_t *)binary.data+offset;
      ASSERT_GE(binary.size,offset+8);
      const size_t length=
         agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3B,code,8,&loads[i])?8:4;
      ASSERT_TRUE(length==8||
         agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3B,code,4,&loads[i]));
      offset+=length;
   }
   util_dynarray_fini(&binary);
}

TEST_F(CProgCertificate, HalfImportsOfOneRegisterAreOrdered) {
   agx3_static_spill_load_desc loads[2]={};
   pack_half_imports(b,1,0,loads);
   EXPECT_EQ(loads[0].dest,1u);EXPECT_EQ(loads[0].tps_half,9u);
   EXPECT_EQ(loads[1].dest,0u);EXPECT_EQ(loads[1].tps_half,8u);
   EXPECT_NE(loads[1].wait_mask&(1u<<(loads[0].slot-1)),0u);
}

TEST_F(CProgCertificate, HalfImportsOfTwoRegistersStayIndependent) {
   agx3_static_spill_load_desc loads[2]={};
   pack_half_imports(b,1,4,loads);
   EXPECT_EQ(loads[0].dest,1u);EXPECT_EQ(loads[1].dest,4u);
   EXPECT_EQ(loads[1].wait_mask&(1u<<(loads[0].slot-1)),0u);
}

TEST_F(CProgCertificate, Qualified188ByteBoundary) {
   import();auto ops=spill(188,184);EXPECT_TRUE(certify());
   ops.load->src[0]=agx_immediate(188);EXPECT_FALSE(certify());
}
TEST_F(CProgCertificate, Qualified188ByteHalfBoundary) {
   import();auto ops=spill(188,186,true);EXPECT_TRUE(certify());
   ops.store->src[1]=agx_immediate(188);EXPECT_FALSE(certify());
   b->shader->scratch_size_B=AGX3_CPROG_STATIC_SPILL_MAX_B+2;
   info.preamble_scratch_size=align(b->shader->scratch_size_B,4);EXPECT_FALSE(certify());
}

TEST_F(CProgCertificate, Extension360WordBoundary) {
   import();auto ops=spill(360,356);EXPECT_TRUE(certify());
   ops.load->src[0]=agx_immediate(360);EXPECT_FALSE(certify());
   info.agx3_preamble_static_spill_version=1;
   EXPECT_TRUE(agx3_cprog_static_spill_info_valid(&info));
   info.preamble_scratch_size=364;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
}
TEST_F(CProgCertificate, Extension360HalfBoundary) {
   import();auto ops=spill(360,358,true);EXPECT_TRUE(certify());
   ops.store->src[1]=agx_immediate(360);EXPECT_FALSE(certify());
}

class FragmentCProgCertificate : public CProgCertificate {
protected:
   FragmentCProgCertificate() {
      b->shader->stage=info.stage=b->shader->nir->info.stage=MESA_SHADER_FRAGMENT;
   }
};
TEST_F(FragmentCProgCertificate, SeparateStageVersionContract) {
   info.preamble_scratch_size=192;
   info.agx3_preamble_static_spill_version=AGX3_CPROG_FS_STATIC_SPILL_VERSION;
   EXPECT_TRUE(agx3_cprog_static_spill_info_valid(&info));
   info.agx3_preamble_static_spill_version=AGX3_CPROG_STATIC_SPILL_VERSION;
   EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.agx3_preamble_static_spill_version=0;
   EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.agx3_preamble_static_spill_version=3;
   EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.agx3_preamble_static_spill_version=AGX3_CPROG_FS_STATIC_SPILL_VERSION;
   info.stage=MESA_SHADER_VERTEX;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.stage=MESA_SHADER_COMPUTE;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.stage=MESA_SHADER_GEOMETRY;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.stage=MESA_SHADER_FRAGMENT;info.preamble_scratch_size=196;
   EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.preamble_scratch_size=191;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.preamble_scratch_size=32;EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.preamble_scratch_size=64;EXPECT_TRUE(agx3_cprog_static_spill_info_valid(&info));
   info.preamble_scratch_size=104;EXPECT_TRUE(agx3_cprog_static_spill_info_valid(&info));
}
TEST_F(FragmentCProgCertificate, ProvenanceAnd192WordBoundary) {
   import();ASSERT_TRUE(agx3_cprog_spill_provenance(b->shader));
   auto ops=spill(192,188);EXPECT_TRUE(certify());EXPECT_FALSE(certify(false));
   EXPECT_FALSE(agx3_cprog_spill_provenance(b->shader));
   ops.load->src[0]=agx_immediate(192);EXPECT_FALSE(certify());
   ops.load->src[0]=agx_immediate(188);b->shader->scratch_size_B=196;
   info.preamble_scratch_size=196;EXPECT_FALSE(certify());
}
TEST_F(FragmentCProgCertificate, Last192HalfAndRoundedFrame) {
   import();auto ops=spill(192,190,true);EXPECT_TRUE(certify());
   ops.store->src[1]=agx_immediate(192);EXPECT_FALSE(certify());
   ops.store->src[1]=agx_immediate(190);b->shader->scratch_size_B=194;
   info.preamble_scratch_size=196;EXPECT_FALSE(certify());
}
TEST_F(FragmentCProgCertificate, ImportsCannotBeSpillOrExport) {
   import();auto ops=spill(64,60);EXPECT_TRUE(certify());
   ops.store->src[1]=agx_immediate(28);EXPECT_FALSE(certify());
   ops.store->src[1]=agx_immediate(60);ops.load->src[0]=agx_immediate(16);EXPECT_FALSE(certify());
   ops.load->src[0]=agx_immediate(60);
   agx_uniform_store(b,agx_register(4,AGX_SIZE_32),agx_immediate(16),1);
   EXPECT_FALSE(certify());
}
TEST_F(FragmentCProgCertificate, DynamicOffsetsAndOtherStorageRejected) {
   import();auto ops=spill(104,100);EXPECT_TRUE(certify());
   ops.store->src[1]=agx_register(8,AGX_SIZE_32);EXPECT_FALSE(certify());
   ops.store->src[1]=agx_immediate(100);
   const agx_opcode forbidden[]={AGX_OPCODE_ATOMIC,AGX_OPCODE_LOCAL_ATOMIC,
      AGX_OPCODE_LOCAL_LOAD,AGX_OPCODE_LOCAL_STORE,AGX_OPCODE_STACK_MAP,
      AGX_OPCODE_STACK_UNMAP,AGX_OPCODE_STACK_ADJUST,AGX_OPCODE_SCS_LOAD_AGX3,
      AGX_OPCODE_SCS_STORE_AGX3,AGX_OPCODE_SCS_LOAD_UNIFORM_INDEXED_AGX3,
      AGX_OPCODE_SCS_STORE_UNIFORM_INDEXED_AGX3,
      AGX_OPCODE_SCS_LOAD_REGISTER_INDEXED_AGX3,AGX_OPCODE_SCS_STORE_REGISTER_INDEXED_AGX3};
   for(auto op:forbidden) {
      auto *i=agx_alloc_instr(b,op,0,0);agx_builder_insert(&b->cursor,i);
      EXPECT_FALSE(certify())<<op;list_del(&i->link);
      b->cursor=agx_after_block(agx_start_block(b->shader));
   }
}
TEST_F(FragmentCProgCertificate, ExistingScratchAndStageMismatchRejected) {
   import();b->shader->nir->info.stage=MESA_SHADER_VERTEX;
   EXPECT_FALSE(agx3_cprog_spill_provenance(b->shader));
   b->shader->nir->info.stage=MESA_SHADER_FRAGMENT;b->shader->scratch_size_B=48;
   EXPECT_FALSE(agx3_cprog_spill_provenance(b->shader));
   b->shader->scratch_size_B=32;b->shader->any_scratch=true;
   EXPECT_FALSE(agx3_cprog_spill_provenance(b->shader));
   b->shader->any_scratch=false;spill(64,60);b->shader->nir->info.stage=MESA_SHADER_VERTEX;
   EXPECT_FALSE(certify());
}
TEST_F(FragmentCProgCertificate, Pack192ByteBoundary) {
   import();ASSERT_TRUE(agx3_cprog_spill_provenance(b->shader));
   spill(192,188);agx_uniform_store(b,agx_register(4,AGX_SIZE_32),agx_immediate(64),1);
   agx_stop(b);ASSERT_TRUE(certify());
   struct util_dynarray binary=UTIL_DYNARRAY_INIT;
   ASSERT_TRUE(agx3_pack_binary(b->shader,&binary));ASSERT_TRUE(certify());
   const uint8_t *code=(const uint8_t *)binary.data;
   agx3_static_spill_load_desc imports={},reload={};agx3_static_spill_store_desc store={};
   ASSERT_TRUE(agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3B,code,8,&imports));
   ASSERT_TRUE(agx3_static_spill_decode_store(AGX_ISA_ENCODING_AGX3B,code+8,10,&store));
   ASSERT_TRUE(agx3_static_spill_decode_load(AGX_ISA_ENCODING_AGX3B,code+18,8,&reload));
   EXPECT_EQ(imports.tps_half,8u);EXPECT_NE(imports.wait_mask&32u,0u);
   EXPECT_EQ(store.tps_half,94u);EXPECT_EQ(reload.tps_half,94u);
   EXPECT_NE(store.wait_mask&(1u<<(imports.slot-1)),0u);
   EXPECT_NE(reload.wait_mask&(1u<<(store.slot-1)),0u);
   uint8_t carrier[AGX3_ALU_MAX_LENGTH],length=0;
   ASSERT_TRUE(agx3_alu_band_identity_wait(AGX_ISA_ENCODING_AGX3B,4,
      1u<<(reload.slot-1),carrier,&length));
   ASSERT_GE(binary.size,26u+length);EXPECT_EQ(memcmp(code+26,carrier,length),0);
   util_dynarray_fini(&binary);
}

TEST_F(FragmentCProgCertificate, OutputStagePublishedAfterCertification) {
   /* agx_compile_shader_nir fills info.stage only after function compilation.
    * Match that real phase: context/NIR identify FS while output is zeroed. */
   info.stage=MESA_SHADER_VERTEX;
   import();EXPECT_TRUE(agx3_cprog_spill_provenance(b->shader));
   spill(192,188);EXPECT_TRUE(certify());
   info.agx3_preamble_static_spill_version=AGX3_CPROG_FS_STATIC_SPILL_VERSION;
   EXPECT_FALSE(agx3_cprog_static_spill_info_valid(&info));
   info.stage=MESA_SHADER_FRAGMENT;
   EXPECT_TRUE(agx3_cprog_static_spill_info_valid(&info));
}

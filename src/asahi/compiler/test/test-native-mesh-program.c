#undef NDEBUG
#include "agx3_graphics_program.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
int main(void)
{
   const struct {
      bool object; unsigned shared, binary_size, scratch, compiler_d;
      uint64_t shader; uint8_t tail[28];
   } cases[] = {
      {true,0,32,0,1,UINT64_C(0x1000000280),
       {7,0,0,12,0,0,0,0,135,2,64,14,0,0,0,0,16,0,
        39,0,0,13,0,0,0,0,0,0}},
      {false,192,660,16,0,UINT64_C(0x10000002a40),
       {15,24,0,12,0,0,0,0,71,42,88,14,0,0,0,0,0,1,
        7,0,0,13,0,0,4,0,4,0}},
      {true,64,334,16,1,UINT64_C(0x10000002e40),
       {15,8,0,12,0,0,0,0,71,46,88,14,0,0,0,0,0,1,
        39,0,0,13,0,0,4,0,4,0}},
   };
   for (unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);i++) {
      const __typeof__(cases[0]) c=cases[i];
      struct agx3_g17p_graphics_program_desc d = {0};
      d.program_va = UINT64_C(0x10000128000);d.shader_va=c.shader;
      d.shader_binary_size=c.binary_size;d.scratch_size=c.scratch;
      d.nr_gprs=32;d.compiler_d_valid=true;d.compiler_d=c.compiler_d;
      d.threadgroup_memory_size_B=c.shared;
      d.usc_clusters=2; /* the M5 Pro's lead */
      struct agx3_g17p_graphics_program p = {0};
      const enum agx_chip m5_chips[] = {
         AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G17C, AGX_CHIP_G17D,
      };
      for (unsigned ci = 0; ci < sizeof(m5_chips)/sizeof(m5_chips[0]); ci++) {
         assert(agx3_g17g_mesh_program_generate(m5_chips[ci],&d,c.object,&p));
         assert(p.length>=28);assert(memcmp(p.bytes,c.tail,28)==0);
      }
      assert(!agx3_g17g_mesh_program_generate(AGX_CHIP_G17P,&d,c.object,&p));
      d.fragment=true;
      for (unsigned ci = 0; ci < sizeof(m5_chips)/sizeof(m5_chips[0]); ci++)
         assert(!agx3_g17g_mesh_program_generate(m5_chips[ci],&d,c.object,&p));
   }
   puts("Native mesh/object IOTO tails: 3/3 exact captured matches");
   return 0;
}

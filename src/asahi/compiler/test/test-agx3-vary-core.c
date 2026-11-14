/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The AGX3 varying encoders must reproduce real Apple f32interp/ldcfr/uvsw
 * instructions (clean center-location, marker-free forms) from the varyops
 * corpus.
 */
#include "agx3_vary_core.h"
#include "agx3_vary_core_vectors.c"
#include <stdio.h>
#include <string.h>
int main(void){
   unsigned pass=0,fail=0;
   for(size_t i=0;i<vary_vec_count;i++){
      const struct vary_vec *v=&vary_vecs[i];
      enum agx_isa_encoding e=v->enc==0?AGX_ISA_ENCODING_AGX3B:AGX_ISA_ENCODING_AGX3C;
      uint8_t out[AGX3_VARY_MAX_LENGTH]; uint8_t len=0; bool ok;
      if(!strcmp(v->op,"interp")) ok=agx3_vary_interp(e,v->a,v->b,v->c!=0,out,&len);
      else if(!strcmp(v->op,"ldcf")) ok=agx3_vary_ldcf(e,v->a,v->b,(uint8_t)v->c,out,&len);
      else ok=agx3_vary_uvsw(e,v->a,v->b,(uint8_t)v->c,v->d!=0,(uint16_t)v->wait,out,&len);
      if(!ok||len!=v->nbytes||memcmp(out,v->bytes,v->nbytes)){ if(fail<10) fprintf(stderr,"FAIL %s enc=%d i=%zu\n",v->op,v->enc,i); fail++; }
      else pass++;
   }
   {
      uint8_t out[AGX3_VARY_MAX_LENGTH]; uint8_t len=0;
      if(agx3_vary_interp(AGX_ISA_ENCODING_AGX3B,1,0,false,out,&len)) fail++; else pass++;
      if(agx3_vary_ldcf(AGX_ISA_ENCODING_AGX3C,0,0,9,out,&len)) fail++; else pass++;
      if(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B,0,0,0,false,0,out,&len)) fail++; else pass++;
      if(agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B,0,0,1,false,0x40,out,&len)) fail++; else pass++;
   }
   {
      /* An unlinked vertex shader's index register: the same write of staging
       * half 4 on slot 1 with f23 = 2 and the index in f16 rather than an
       * immediate in f15. */
      static const uint8_t immediate[] = {0x57,0x16,0x54,0x04,0x00,0x41,0x48,0x00};
      static const uint8_t reg[] = {0x57,0x16,0x54,0x04,0x00,0x01,0x58,0x00};
      static const uint8_t reg_c[] = {0x35,0x04,0x13,0x10,0x20,0x00,0xa0,0x12};
      uint8_t out[AGX3_VARY_MAX_LENGTH]; uint8_t len=0;
      if(!agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B,4,8,1,true,1,out,&len) ||
         len!=sizeof(immediate) || memcmp(out,immediate,sizeof(immediate))) fail++;
      else pass++;
      if(!agx3_vary_uvsw_register_index(AGX_ISA_ENCODING_AGX3B,4,8,1,true,1,out,&len) ||
         len!=sizeof(reg) || memcmp(out,reg,sizeof(reg))) fail++;
      else pass++;
      if(!agx3_vary_uvsw_register_index(AGX_ISA_ENCODING_AGX3C,4,8,1,true,1,out,&len) ||
         len!=sizeof(reg_c) || memcmp(out,reg_c,sizeof(reg_c))) fail++;
      else pass++;
      if(agx3_vary_uvsw_register_index(AGX_ISA_ENCODING_AGX3B,4,8,7,true,1,out,&len)) fail++; else pass++;
   }
   {
      const uint8_t native[] = {0x2f,0x0d,0x54,0,3,0,4,2,0x10,0};
      uint8_t out[AGX3_VARY_MAX_LENGTH]; uint8_t len=0;
      if (!agx3_vary_projective_rhw(AGX_ISA_ENCODING_AGX3B,0,0,true,out,&len) ||
          len != sizeof(native) || memcmp(out,native,sizeof(native))) fail++;
      else pass++;
      if (agx3_vary_projective_rhw(AGX_ISA_ENCODING_AGX3B,0,128,false,out,&len)) fail++;
      else pass++;
      if (agx3_vary_projective_rhw(AGX_ISA_ENCODING_AGX3B,1,0,false,out,&len)) fail++;
      else pass++;
   }
   {
      /* Fresh Apple Metal 32023.917 applegpu_g17p witnesses from
       * explicit_sample.metal (source SHA-256 2f0835ef...). These isolate
       * ITRCOORDS, perspective/projective f32interp, no-perspective
       * f32interp, and the coordinate token's live/final-use alternatives. */
      static const uint8_t coords_terminal[] = {0xaf, 0x04, 0x54, 0x0a,
                                                0x03, 0x06, 0x0a, 0x03};
      static const uint8_t coords_unmarked[] = {0xaf, 0x04, 0x54, 0x00,
                                                0x03, 0x10, 0x0a, 0x02};
      static const uint8_t projective_live[] = {0x2f, 0x05, 0x54, 0x0c, 0x03,
                                                0x00, 0x05, 0x02, 0x09, 0x00};
      static const uint8_t perspective_live[] = {0x2f, 0x05, 0x54, 0x0c, 0x03,
                                                 0x06, 0x05, 0x02, 0x08, 0x00};
      static const uint8_t linear_terminal[] = {0x2f, 0x05, 0x54, 0x10, 0x02,
                                                0x08, 0x08, 0x02, 0x20, 0x00};
      uint8_t out[AGX3_VARY_MAX_LENGTH];
      uint8_t len = 0;

      if (!agx3_vary_sample_coords(AGX_ISA_ENCODING_AGX3B, 10, 3, true, 0, out,
                                   &len) ||
          len != sizeof(coords_terminal) ||
          memcmp(out, coords_terminal, sizeof(coords_terminal)))
         fail++;
      else
         pass++;
      if (!agx3_vary_sample_coords(AGX_ISA_ENCODING_AGX3B, 0, 8, false, 0, out,
                                   &len) ||
          len != sizeof(coords_unmarked) ||
          memcmp(out, coords_unmarked, sizeof(coords_unmarked)))
         fail++;
      else
         pass++;
      if (!agx3_vary_projective_rhw_sample(AGX_ISA_ENCODING_AGX3B, 12, 0, 10,
                                           false, out, &len) ||
          len != sizeof(projective_live) ||
          memcmp(out, projective_live, sizeof(projective_live)))
         fail++;
      else
         pass++;
      if (!agx3_vary_interp_sample(AGX_ISA_ENCODING_AGX3B, 12, 3, true, 10,
                                   false, out, &len) ||
          len != sizeof(perspective_live) ||
          memcmp(out, perspective_live, sizeof(perspective_live)))
         fail++;
      else
         pass++;
      if (!agx3_vary_interp_sample(AGX_ISA_ENCODING_AGX3B, 16, 4, false, 16,
                                   true, out, &len) ||
          len != sizeof(linear_terminal) ||
          memcmp(out, linear_terminal, sizeof(linear_terminal)))
         fail++;
      else
         pass++;

      if (agx3_vary_sample_coords(AGX_ISA_ENCODING_AGX3B, 1, 0, true, 0, out,
                                  &len))
         fail++;
      else
         pass++;
      if (agx3_vary_sample_coords(AGX_ISA_ENCODING_AGX3B, 0, 256, true, 0, out,
                                  &len))
         fail++;
      else
         pass++;
      if (agx3_vary_sample_coords(AGX_ISA_ENCODING_AGX3B, 0, 0, true, 0x40, out,
                                  &len))
         fail++;
      else
         pass++;
      if (agx3_vary_interp_sample(AGX_ISA_ENCODING_AGX3B, 0, 0, false, 3, false,
                                  out, &len))
         fail++;
      else
         pass++;
      if (agx3_vary_projective_rhw_sample(AGX_ISA_ENCODING_AGX3B, 0, 128, 0,
                                          false, out, &len))
         fail++;
      else
         pass++;
   }
   {
      /* Fresh Apple flat float3 fragment, 2026-09-05: only its first
       * coefficient load carries f48. Later loads preserve the old codec. */
      const uint8_t first[] = {0x1f,0x0b,0x54,0x07,0x00,0x05};
      const uint8_t next[] = {0x1f,0x03,0x54,0x83,0x00,0x05};
      uint8_t out[AGX3_VARY_MAX_LENGTH]; uint8_t len=0;
      if(!agx3_vary_ldcf_at_start(AGX_ISA_ENCODING_AGX3B,14,0,6,true,out,&len) ||
         len!=sizeof(first) || memcmp(out,first,sizeof(first))) fail++; else pass++;
      if(!agx3_vary_ldcf_at_start(AGX_ISA_ENCODING_AGX3B,6,1,6,false,out,&len) ||
         len!=sizeof(next) || memcmp(out,next,sizeof(next))) fail++; else pass++;
      if(agx3_vary_ldcf_at_start(AGX_ISA_ENCODING_AGX3C,14,0,6,true,out,&len)) fail++; else pass++;
   }
   {
      /* Fresh applegpu_g17p [[centroid_{,no_}perspective]] fragments at
       * raster_sample_count 1 and 4: ITRCOORDS f38=1 on the terminal SR39
       * coverage word, waiting on its read in slot 1. G17P has no implicit
       * centroid coordinate form. */
      static const uint8_t centroid_perspective[] =
         {0xaf,0x14,0x54,0x06,0x03,0x00,0x0a,0x01};
      static const uint8_t centroid_linear[] =
         {0xaf,0x14,0x54,0x04,0x03,0x00,0x0a,0x01};
      uint8_t out[AGX3_VARY_MAX_LENGTH]; uint8_t len=0;
      if(!agx3_vary_centroid_coords(AGX_ISA_ENCODING_AGX3B,6,0,true,1,out,&len) ||
         len!=sizeof(centroid_perspective) ||
         memcmp(out,centroid_perspective,sizeof(centroid_perspective))) fail++; else pass++;
      if(!agx3_vary_centroid_coords(AGX_ISA_ENCODING_AGX3B,4,0,true,1,out,&len) ||
         len!=sizeof(centroid_linear) ||
         memcmp(out,centroid_linear,sizeof(centroid_linear))) fail++; else pass++;
      if(agx3_vary_interp_at(AGX_ISA_ENCODING_AGX3B,0,1,true,
                             AGX3_VARY_CENTROID,out,&len)) fail++; else pass++;
   }
   {
      /* A vertex program compiled without its fragment shader writes its
       * varyings at placeholder indices; the link re-encodes each listed
       * write with the bound layout's index and keeps every other operand. */
      struct agx3_uvs_reloc r[3] = {
         {.index = 4, .value = 8, .slot = 1, .wait = 1},
         {.index = 5, .value = 10, .slot = 2, .wait = 2},
         {.index = 6, .value = 12, .slot = 3, .wait = 4},
      };
      uint8_t code[64], want[64], relocs[3 * AGX3_UVS_RELOC_SIZE];
      uint8_t map[AGX3_UVS_RELOC_MAP_SIZE];
      for (unsigned i = 0; i < AGX3_UVS_RELOC_MAP_SIZE; ++i) map[i] = i;
      map[4] = 6; map[5] = 4; map[6] = 5;
      memset(code, 0x06, sizeof(code)); memset(want, 0x06, sizeof(want));
      unsigned at = 2; bool ok = true;
      for (unsigned i = 0; i < 3; ++i) {
         uint8_t len = 0, len2 = 0;
         r[i].offset = at;
         ok &= agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, r[i].value, r[i].index,
                              r[i].slot, true, r[i].wait, code + at, &len);
         ok &= agx3_vary_uvsw(AGX_ISA_ENCODING_AGX3B, r[i].value,
                              map[r[i].index], r[i].slot, true, r[i].wait,
                              want + at, &len2);
         ok &= len == len2;
         memcpy(relocs + i * AGX3_UVS_RELOC_SIZE, &r[i], sizeof(r[i]));
         at += len + 2;
      }
      uint8_t orig[64]; memcpy(orig, code, sizeof(code));
      /* Checking alone (no map) leaves the bytes as they are. */
      ok &= agx3_vary_relocate_uvsw(AGX_ISA_ENCODING_AGX3B, code, at, relocs,
                                    3, NULL) && !memcmp(code, orig, sizeof(code));
      /* A list that does not describe the code changes nothing. */
      code[r[2].offset + 1] ^= 1;
      ok &= !agx3_vary_relocate_uvsw(AGX_ISA_ENCODING_AGX3B, code, at, relocs,
                                     3, map);
      code[r[2].offset + 1] ^= 1;
      ok &= !memcmp(code, orig, sizeof(code));
      ok &= !agx3_vary_relocate_uvsw(AGX_ISA_ENCODING_AGX3B, code,
                                     r[2].offset + 2, relocs, 3, map) &&
            !memcmp(code, orig, sizeof(code));
      ok &= agx3_vary_relocate_uvsw(AGX_ISA_ENCODING_AGX3B, code, at, relocs,
                                    3, map) && !memcmp(code, want, sizeof(code));
      if (ok) pass++; else fail++;
   }
   if(fail){ fprintf(stderr,"agx3_vary_core: %u of %zu failed\n",fail,vary_vec_count); return 1; }
   printf("agx3_vary_core: all %u varying ops reproduced\n",pass); return 0;
}

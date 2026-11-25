/* SPDX-License-Identifier: MIT */

#include "agx_compile.h"
#include "agx3_static_spill_core.h"
#include "agx_test.h"
#include "agx3_copy_core.h"
#include "agx3_tile_core.h"
#include "agx3_transcode.h"
#include "agx3_alu_core.h"
#include "agx3_atomic_core.h"
#include "agx3_tex_core.h"
#include "agx3_iadd_core.h"
#include "agx3_uniform_integer.h"
#include "agx3_mem_core.h"
#include "agx3_scs_core.h"
#include "agx3_rt_m5_core.h"
#include "agx3_local_tps_core.h"
#include "agx3_isel_core.h"
#include "agx3_cf_core.h"
#include "agx3_subgroup_core.h"
#include "agx3_falu_mod_core.h"
#include "agx3_stage_core.h"
#include "agx3_pixel_core.h"
#include "agx3_fltnorm_core.h"
#include "agx3_pack.h"
#include "asahi/lib/agx_bg_eot_key.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* AGX3C SCS encoder constraints. */
static bool
test_native_ray_scs(void)
{
   uint8_t bytes[16], length;
   struct agx3_scs_desc desc = {
      .data = 250, .width = AGX3_MEM_W32, .component_mask = 15, .slot = 8,
   };
   /* A vector may not wrap the register file, even if its first word fits. */
   if (agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &desc, bytes, &length))
      return false;
   desc.data = 248;
   if (!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &desc, bytes, &length))
      return false;
   desc.index_kill = true;
   if (agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &desc, bytes, &length))
      return false;
   desc.index_kill = false;
   desc.c_store_marker = true;
   if (agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &desc, bytes, &length))
      return false;
   desc = (struct agx3_scs_desc) {
      .data = 2, .width = AGX3_MEM_W8, .component_mask = 1, .slot = 8,
      .address_kind = AGX3_SCS_ADDRESS_REGISTER_BASE, .base_register = 2,
      .has_address_selector = true, .address_selector = 4,
      .byte_offset = 7, .c_load_marker = true, .c_status_load = true,
   };
   if (!agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &desc, bytes, &length) ||
       agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3B, &desc, bytes, &length) ||
       agx3_scs_encode_store(AGX_ISA_ENCODING_AGX3C, &desc, bytes, &length))
      return false;
   desc.c_load_marker = false;
   if (agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &desc, bytes, &length))
      return false;
   desc.c_load_marker = true;
   desc.width = AGX3_MEM_W32;
   if (agx3_scs_encode_load(AGX_ISA_ENCODING_AGX3C, &desc, bytes, &length))
      return false;

   struct agx3_scs_epilogue_desc epilogue = {
      .index = 2, .base_register = 2, .component_mask = 15,
      .address_selector = 4,
   };
   if (!agx3_scs_encode_epilogue(AGX_ISA_ENCODING_AGX3C, &epilogue,
                                 bytes, &length) ||
       agx3_scs_encode_epilogue(AGX_ISA_ENCODING_AGX3B, &epilogue,
                                 bytes, &length))
      return false;
   for (unsigned invalid = 0; invalid < 5; ++invalid) {
      struct agx3_scs_epilogue_desc bad = epilogue;
      switch (invalid) {
      case 0: bad.index = 256; break;
      case 1: bad.base_register = 256; break;
      case 2: bad.byte_offset = 1; break;
      case 3: bad.component_mask = 0; break;
      case 4: bad.address_selector = 3; break;
      }
      memset(bytes, 0xff, sizeof(bytes));
      length = 0xff;
      if (agx3_scs_encode_epilogue(AGX_ISA_ENCODING_AGX3C, &bad,
                                    bytes, &length) || length)
         return false;
      for (unsigned j = 0; j < sizeof(bytes); ++j)
         if (bytes[j])
            return false;
   }
   return true;
}

static bool
test_native_rt_m5(void)
{
   const struct {
      struct agx3_rt_m5_intersect_desc desc;
      uint8_t bytes[14];
   } cases[] = {
      {{.address_register = 0, .bvh_base = 96, .bvh_uniform = true},
       {0x27,0x04,0x29,0x01,0x22,0xc1,0xae,0x02,0x01,0x98,0x00,0x00,0x00,0xa0}},
      {{.address_register = 0, .bvh_base = 46, .bvh_kill = true},
       {0x27,0x04,0x29,0x01,0xa2,0xc3,0xae,0x02,0x80,0x8a,0x00,0x00,0x00,0xa0}},
      {{.address_register = 34, .bvh_base = 96, .bvh_uniform = true},
       {0x27,0x04,0x29,0x01,0x22,0xc1,0xae,0x02,0x01,0x98,0x21,0x00,0x00,0xa0}},
   };
   uint8_t bytes[16], length;
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      if (!agx3_rt_m5_encode_intersect(AGX_ISA_ENCODING_AGX3C,
                                      &cases[i].desc, bytes, &length) ||
          length != sizeof(cases[i].bytes) ||
          memcmp(bytes, cases[i].bytes, length))
         return false;
   }
   const uint8_t continuation[] = {0x27,0x04,0x09,0x00,0x32,0xc0,0xaf,0x02};
   if (!agx3_rt_m5_encode_continue(AGX_ISA_ENCODING_AGX3C, bytes, &length) ||
       length != sizeof(continuation) || memcmp(bytes, continuation, length))
      return false;
   struct agx3_rt_m5_intersect_desc bad = cases[0].desc;
   if (agx3_rt_m5_encode_intersect(AGX_ISA_ENCODING_AGX3B, &bad, bytes, &length))
      return false;
   bad.address_register = 253;
   if (agx3_rt_m5_encode_intersect(AGX_ISA_ENCODING_AGX3C, &bad, bytes, &length))
      return false;
   bad = cases[0].desc;
   bad.bvh_kill = true;
   if (agx3_rt_m5_encode_intersect(AGX_ISA_ENCODING_AGX3C, &bad, bytes, &length))
      return false;
   bad = cases[1].desc;
   bad.bvh_base = 256;
   return !agx3_rt_m5_encode_intersect(AGX_ISA_ENCODING_AGX3C, &bad, bytes, &length);
}

int main(void)
{
   if (!test_native_ray_scs() || !test_native_rt_m5())
      return 140;
   /* Preserve register-base SCS and both native BVH operand classes through
    * the actual IR packer. The SCS load overlaps the ray address operand,
    * exercising input retirement before the field-free ray instruction. */
   {
      void *memctx = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(memctx);
      struct agx_shader_key key = {0};
      struct agx_shader_info info = {0};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      b->shader->key = &key;
      b->shader->out = &info;
      b->shader->nir = rzalloc(memctx, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
      b->shader->stage = info.stage = MESA_SHADER_COMPUTE;
      b->shader->max_reg = 49;
      agx_index value = agx_register(4, AGX_SIZE_32);
      value.channels_m1 = 1;
      agx_scs_store_register_indexed_agx3(b, value,
         agx_register(32, AGX_SIZE_32), agx_register(28, AGX_SIZE_32),
         0, AGX_FORMAT_I32, 3, 3);
      agx_scs_load_register_indexed_agx3_to(b, agx_register(2, AGX_SIZE_32),
         agx_register(32, AGX_SIZE_32), agx_register(28, AGX_SIZE_32),
         512, AGX_FORMAT_I32, 1, 3, true, false);
      agx_intersect_ray_m5(b, agx_register(0, AGX_SIZE_64),
                          agx_uniform(96, AGX_SIZE_64), agx_register(10, AGX_SIZE_32),
                          agx_register(12, AGX_SIZE_32), agx_register(14, AGX_SIZE_32));
      agx_index bvh = agx_register(46, AGX_SIZE_64);
      bvh.kill = true;
      agx_intersect_ray_m5(b, agx_register(0, AGX_SIZE_64), bvh,
                          agx_register(10, AGX_SIZE_32), agx_register(12, AGX_SIZE_32),
                          agx_register(14, AGX_SIZE_32));
      agx_continue_ray_m5(b, agx_register(14, AGX_SIZE_32));
      agx_stop(b);
      struct util_dynarray binary = UTIL_DYNARRAY_INIT;
      if (!agx3_pack_binary(b->shader, &binary))
         return 141;
      unsigned stores = 0, loads = 0, rays = 0, continuations = 0;
      uint8_t staged[3 * AGX3_STAGE_MAX_LENGTH + 6];
      unsigned staged_len = 0;
      for (unsigned w = 0; w < 3; ++w) {
         uint8_t len;
         if (!agx3_stage_copy_liveness(AGX_ISA_ENCODING_AGX3C, 2 * w,
                10 + 2 * w, false, false, 1, 0, staged + staged_len, &len))
            return 143;
         staged_len += len;
      }
      memcpy(staged + staged_len, "\x17\x11\x00\x06\x00\x00", 6);
      staged_len += 6;
      const uint8_t native_rays[2][14] = {
         {0x27,0x04,0x29,0x01,0x22,0xc1,0xae,0x02,0x01,0x98,0x00,0x00,0x00,0xa0},
         {0x27,0x04,0x29,0x01,0xa2,0xc3,0xae,0x02,0x80,0x8a,0x00,0x00,0x00,0xa0},
      };
      const uint8_t native_continue[] = {0x27,0x04,0x09,0x00,0x32,0xc0,0xaf,0x02};
      for (unsigned off = 0; off < binary.size; off += 2) {
         struct agx3_scs_desc d;
         const uint8_t *bytes = (const uint8_t *)binary.data + off;
         for (unsigned r = 0; r < 2; ++r) {
            if (off + sizeof(native_rays[r]) <= binary.size &&
                !memcmp(bytes, native_rays[r], sizeof(native_rays[r]))) {
               if (off < staged_len ||
                   memcmp(bytes - staged_len, staged, staged_len))
                  return 144;
               rays++;
            }
         }
         if (off + sizeof(native_continue) <= binary.size &&
             !memcmp(bytes, native_continue, sizeof(native_continue))) {
            uint8_t prefix[AGX3_STAGE_MAX_LENGTH + 6], size;
            if (!agx3_stage_copy_liveness(AGX_ISA_ENCODING_AGX3C, 8, 14,
                   false, false, 1, 0, prefix, &size))
               return 145;
            memcpy(prefix + size, "\x17\x09\x00\x06\x00\x00", 6);
            if (off < size + 6 || memcmp(bytes - size - 6, prefix, size + 6))
               return 146;
            continuations++;
         }
         if (off + 14 <= binary.size &&
             agx3_scs_decode_store(AGX_ISA_ENCODING_AGX3C, bytes, 14, &d) &&
             d.data == 4 && d.address_kind == AGX3_SCS_ADDRESS_REGISTER_INDEXED &&
             d.base_register == 32 && d.index == 28 && d.component_mask == 3 &&
             d.address_selector == 3 && d.byte_offset == 0)
            stores++;
         if (off + 16 <= binary.size &&
             agx3_scs_decode_load(AGX_ISA_ENCODING_AGX3C, bytes, 16, &d) &&
             d.data == 2 && d.address_kind == AGX3_SCS_ADDRESS_REGISTER_INDEXED &&
             d.base_register == 32 && d.index == 28 && d.component_mask == 1 &&
             d.address_selector == 3 && d.byte_offset == 512 &&
             d.c_load_marker && d.slot != 0)
            loads++;
      }
      util_dynarray_fini(&binary);
      ralloc_free(memctx);
      if (stores != 1 || loads != 1 || rays != 2 || continuations != 1)
         return 142;
      puts("M5_NATIVE_RAY_SCS_IR_PASS");
   }
   /* M5 STTPS has an implicit slot-1 completion. The following static
    * reload must consume it, as in Metal552 and the array CTS reproducer. */
   {
      void *memctx = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(memctx);
      struct agx_shader_key key = {0};
      struct agx_shader_info info = {0};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      b->shader->key = &key;
      b->shader->out = &info;
      b->shader->nir = rzalloc(memctx, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
      b->shader->stage = info.stage = MESA_SHADER_COMPUTE;
      b->shader->max_reg = 9;
      b->shader->scratch_size_B = 64;
      agx_stack_store(b, agx_register(0, AGX_SIZE_32),
                      agx_register(4, AGX_SIZE_32), AGX_FORMAT_I32, 1);
      agx_stack_load_to(b, agx_register(8, AGX_SIZE_32),
                        agx_immediate(32), AGX_FORMAT_I32, 1);
      agx_stop(b);
      struct util_dynarray binary = UTIL_DYNARRAY_INIT;
      if (!agx3_pack_binary(b->shader, &binary))
         return 130;
      unsigned found = 0;
      for (unsigned off = 0; off < binary.size; off += 2) {
         for (unsigned len = 4; len <= 8; len += 4) {
            if (off + len > binary.size)
               continue;
            struct agx3_static_spill_load_desc load;
            if (agx3_static_spill_decode_load(
                   AGX_ISA_ENCODING_AGX3C,
                   (const uint8_t *)binary.data + off, len, &load) &&
                load.tps_half == 16 && load.dest == 8) {
               if (!(load.wait_mask & 1))
                  return 131;
               found++;
            }
         }
      }
      if (found != 1)
         return 132;
      util_dynarray_fini(&binary);
      ralloc_free(memctx);
   }
   /* Metal532: int/short variable ASR, then all four mixed-width signs. */
   {
      const uint8_t shifts[][12] = {
         {0x27,0x04,0x00,0x1a,0x26,0x00,0xa3,0x02,0x08,0x01,0x60,0x00},
         {0x37,0x04,0x00,0x38,0x26,0x00,0xa3,0x02,0x80,0x80,0x60,0x00},
      };
      const enum agx3_alu_hint hints[] = {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
      for (unsigned h=0; h<2; h++) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len;
         if (!agx3_alu_asr_reg_m5(h ? 2 : 0, h ? 4 : 0, h ? 2 : 4,
                h, h, hints, false, 1, bytes, &len) ||
             len != 12 || memcmp(bytes, shifts[h], 12))
            return 121;
      }
      for (unsigned signs=0; signs<4; signs++) {
         struct agx3_imad_desc d = {
            .dest_64=true, .src_a=0, .src_a_kill=true,
            .src_a_sign_extend=!!(signs&1),
            .multiplier_is_register=true, .multiplier_reg=2,
            .multiplier_16=true, .multiplier_kill=true,
            .multiplier_sign_extend=!!(signs&2),
            .addend_is_immediate=true, .wait_mask=128,
         };
         uint8_t expected[] = {0x2f,0,4,2,0x21,0,0xa9,2,0xa8,0,0x20,0x30};
         expected[5] |= signs&2 ? 8 : 0;
         expected[11] |= signs&1 ? 0x80 : 0;
         uint8_t bytes[AGX3_IADD_MAX_LENGTH], len;
         if (!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C,&d,bytes,&len) ||
             len != sizeof(expected) || memcmp(bytes,expected,len))
            return 122;
      }
      puts("M5_DYNAMIC_ASR_AND_MIXED_WIDE_IMAD_METAL_BYTES_PASS");
   }

   /* Byte-exact Metal ushort2 -> uint3 AND/OR/XOR, including odd half 5. */
   {
      const enum agx3_alu_op ops[] = {AGX3_ALU_BAND, AGX3_ALU_BOR, AGX3_ALU_BXOR};
      const uint8_t expected[3][10] = {
         {0x2b,0x00,0x07,0x00,0x30,0x70,0xa2,0x02,0x42,0x01},
         {0x33,0x00,0x07,0x00,0x32,0x78,0xa2,0x02,0x42,0x01},
         {0x23,0x00,0x06,0x00,0x32,0x78,0xa4,0x0a,0x44,0x01},
      };
      const uint16_t src[] = {4, 5};
      for (unsigned i = 0; i < 3; ++i) {
         const enum agx3_alu_hint hints[] = {
            i == 2 ? AGX3_ALU_HINT_DISCARD : AGX3_ALU_HINT_CACHE,
            i == 2 ? AGX3_ALU_HINT_DISCARD : AGX3_ALU_HINT_CACHE,
         };
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], len = 0;
         if (!agx3_alu_bit_half_to_word_m5(ops[i], i * 2, src, hints,
                 false, i == 0 ? 128 : 0, bytes, &len) ||
             len != 10 || memcmp(bytes, expected[i], len))
            return 120;
      }
      puts("M5_HALF_TO_WORD_BITWISE_METAL_BYTES_PASS");
   }

   /* Fresh G17G saturated scale, ABS, signed-immediate and narrow witnesses. */
   const uint8_t saturated_scale[][14] = {
      {0x29,0,5,0x30,0x22,0x80,0xac,2,0x0a,0,0x20,2,0,0x10},
      {0x29,0,5,0x30,0x22,0x80,0xac,2,0x0a,0,0x20,0x0a,0,0x10},
      {0x29,0,5,0x30,0x22,0x80,0xad,2,0x0a,0,0x20,2,0,0x10},
      {0x39,0,5,0x38,0x22,0x80,0xac,2,0x8a,0,0x20,2,0,0x10},
   };
   for (unsigned i = 0; i < 4; ++i) {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
      const bool narrow = i == 3;
      if (!agx3_alu_fmul_minifloat_narrow(AGX_ISA_ENCODING_AGX3C,
             narrow ? 2 : 0, narrow ? 2 : 0, i == 2 ? 132 : 4,
             false, i == 1, narrow, true, AGX3_ALU_HINT_DISCARD, 128,
             bytes, &length) || length != 14 ||
          memcmp(bytes, saturated_scale[i], length)) {
         puts("M5_SATURATED_SCALE_NATIVE_MISMATCH");
         return 1;
      }
   }
   /* Apple G17G vertex-native394 normalized16 and packed fetch witnesses. */
   for (unsigned snorm = 0; snorm < 2; ++snorm) {
      uint8_t bytes[AGX3_FLTNORM_MAX_LENGTH], length;
      uint8_t expected[] = {0x2f,0,0,0x1a,0x2d,0,0xa8,2,0x98,0};
      expected[8] = snorm ? 0x58 : 0x98;
      const struct agx3_normflt_desc d = {
         .dest_hint = true, .two_lanes = true, .is_16bit = true,
         .snorm = snorm, .wait_mask = 128,
      };
      if (!agx3_normflt_encode(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length) ||
          length != sizeof(expected) || memcmp(bytes, expected, length)) {
         puts("M5_NORMALIZED16_VERTEX_NATIVE_MISMATCH");
         return 1;
      }
   }
   for (unsigned layout = 0; layout <= 2; layout += 2) {
      uint8_t bytes[16], length;
      uint8_t expected[] = {0x2f,0,0,2,0x2e,0x80,0xac,2,0x48,0};
      if (layout) { expected[8] = 8; expected[9] = 2; }
      const struct agx3_rgb10a2_unpack_desc d = {
         .dest_hint = true, .wait_mask = 128, .discard = true,
         .layout = layout,
      };
      if (!agx3_rgb10a2_unpack_encode(AGX_ISA_ENCODING_AGX3C,
                                                 &d, bytes, &length) ||
          length != sizeof(expected) || memcmp(bytes, expected, length)) {
         puts("M5_PACKED_VERTEX_NATIVE_MISMATCH");
         return 1;
      }
   }
   /* Apple G17G arithmetic captures from metal-browser379/380. */
   {
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], size;
      const struct agx3_falu_mods mods = {.saturate = true};
      const struct agx3_falu_runtime runtime = {
         .discard = {true, true, false}, .wait_mask = 128,
      };
      const uint8_t add[] = {0x39,0,4,0x18,0x22,0x30,0xa0,2,4,0x18,0x1a,0};
      const uint8_t mul[] = {0x39,0,5,0x38,0x22,0x80,0xa0,2,0x9c,0,0x20,2,0,0x10};
      if (!agx3_falu_add_narrow_m5(2,2,4,&mods,&runtime,bytes,&size) ||
          size != sizeof(add) || memcmp(bytes,add,size) ||
          !agx3_falu_mul_narrow_m5(2,2,4,&mods,&runtime,bytes,&size) ||
          size != sizeof(mul) || memcmp(bytes,mul,size)) {
         puts("M5_SATURATED_NARROW_ARITHMETIC_NATIVE_MISMATCH");
         return 1;
      }
      const uint16_t src[] = {5,4,2};
      const enum agx3_alu_hint hints[] = {
         AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,
      };
      const uint8_t hfma[] = {0x38,1,6,0x0a,0x23,0x20,0xa4,2,5,1,0x80,0x22};
      if (!agx3_alu_hfma_saturate_m5(2,src,hints,true,false,false,128,bytes,&size) ||
          size != sizeof(hfma) || memcmp(bytes,hfma,size)) {
         puts("M5_SATURATED_HFMA_NATIVE_MISMATCH");
         return 1;
      }
      const uint16_t cmp_src[] = {0,2};
      const bool half[] = {false,false}, absolute[] = {true,false};
      const struct agx3_cf_source_hints cmp_hints[] = {
         {.discard=true}, {.discard=true},
      };
      const uint8_t cmp[] = {0x2a,4,0x2b,0,0x23,0x82,0xa0,0x12,0x80,0x20};
      if (!agx3_cf_fcmp_width_m5(AGX3_CF_FCOND_EQ,false,0,cmp_src,half,
                               absolute,cmp_hints,1,bytes,&size) ||
          size != sizeof(cmp) || memcmp(bytes,cmp,size)) {
         puts("M5_ABSOLUTE_FCMP_NATIVE_MISMATCH");
         return 1;
      }
   }
   /* Native G17G R8/RG8 framebuffer-fetch captures (metal-browser381). */
   for (unsigned pair = 0; pair < 2; ++pair) {
      uint8_t expected[] = {
         0x27,0x01,0x05,0x08,0x01,0x87,0xbc,0x02,0,0,0x40,0x18,0x20,0,
      };
      if (pair)
         expected[11] = 0x98;
      uint8_t bytes[AGX3_TILE_MAX_LENGTH], size;
      if (!agx3_tile_load(AGX_ISA_ENCODING_AGX3C, 0, AGX3_TILE_W8, 8,
                         pair ? 3 : 1, 0, AGX3_TILE_SAMPLE_1X, 0,
                         true, true, bytes, &size) ||
          size != sizeof(expected) || memcmp(bytes, expected, size)) {
         puts("M5_SCALAR_PAIR_TILE_LOAD_NATIVE_MISMATCH");
         return 1;
      }
   }
   {
      const uint8_t expected[] = {
         0x2f,0x00,0x04,0x1a,0x25,0x00,0xa3,0x02,0x20,0x00,0xa0,0x00,
      };
      const struct agx3_iadd_desc desc = {
         .src_16 = true, .sign_extend = true, .shift = 4,
         .src_kill = true, .wait_mask = 128,
      };
      uint8_t bytes[AGX3_IADD_MAX_LENGTH], size;
      if (!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3C, &desc, bytes, &size) ||
          size != sizeof(expected) || memcmp(bytes, expected, size)) {
         puts("M5_SIGNED_HALF_SHIFT_NATIVE_MISMATCH");
         return 1;
      }
   }
   {
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], size;
      const uint8_t expected[] = {0x29,0x00,0x05,0x90,0x22,0x80,0xac,
         0x02,0x0b,0x00,0x20,0x08,0x00,0x10};
      if (!agx3_alu_fmul_minifloat_narrow(AGX_ISA_ENCODING_AGX3C, 0, 0,
             0x50, false, true, false, false, AGX3_ALU_HINT_DISCARD, 128,
             bytes, &size) || size != sizeof(expected) ||
          memcmp(bytes, expected, size)) {
         puts("M5_ABS_MUL_NATIVE_MISMATCH");
         return 1;
      }
   }
   {
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], size;
      const struct agx3_falu_mods mods = {0};
      const struct agx3_falu_runtime runtime = {
         .discard = {true, true, true}, .wait_mask = 128,
      };
      /* Native device-half result from three device-float operands. */
      const uint8_t expected[] = {0x39,0x00,0x07,0x08,0x20,0x30,0xa3,
         0x02,0xb8,0x00,0xa0,0x00,0x04,0x08,0x00,0x80};
      if (!agx3_falu_fma_narrow_m5(2, 2, 0, 4, &mods, &runtime, bytes, &size) ||
          size != sizeof(expected) || memcmp(bytes, expected, size)) {
         puts("M5_NARROW_FMA_NATIVE_MISMATCH");
         return 1;
      }
   }
   /* Fresh G17G RGBA32Float fragments: retain Apple's native width pairs,
    * fused arithmetic and modifiers, checked byte-for-byte. */
   {
      uint8_t b[AGX3_ALU_MAX_LENGTH], n;
      const uint16_t same[] = {0, 0}, pair[] = {0, 2}, add_pair[] = {0, 4};
      const enum agx3_alu_hint dead[] = {AGX3_ALU_HINT_DISCARD,
                                         AGX3_ALU_HINT_DISCARD};
      const bool abs[] = {true, false}, neg[] = {false, true};
      const uint8_t rcp[] = {0x27, 0x00, 0x04, 0x12, 0x2a,
                             0x20, 0xa1, 0x02, 0x28, 0x00};
      const uint8_t rsqrt[] = {0x27, 0x00, 0x04, 0x12, 0x2a,
                               0x20, 0xa3, 0x02, 0x28, 0x00};
      const uint8_t sqrt[] = {0x27, 0x82, 0x04, 0x18, 0x2a,
                              0x20, 0xa2, 0x02, 0x08, 0x00};
      const uint8_t fma[] = {0x00, 0x00, 0x3e, 0x00, 0x21, 0xb0, 0x00, 0x00};
      const uint8_t add[] = {0x20, 0x00, 0x24, 0x02, 0x2a,
                             0x20, 0xa0, 0x02, 0x06, 0x28};
      const uint8_t mixed_add[] = {0x21, 0x00, 0x24, 0xca, 0x2a, 0x20,
                                   0xac, 0x02, 0x01, 0x08, 0x40, 0x40};
      const uint8_t mul[] = {0x01, 0x01, 0x39, 0x03};
#define CHECK_NATIVE(call, expected)                                           \
   do {                                                                        \
      if (!(call) || n != sizeof(expected) || memcmp(b, expected, n)) {        \
         printf("M5_HALF_NATIVE_MISMATCH %s len=%u\n", #expected, n);          \
         for (unsigned i = 0; i < n; i++)                                      \
            printf("%02x", b[i]);                                              \
         puts("");                                                             \
         return 1;                                                             \
      }                                                                        \
   } while (0)
      CHECK_NATIVE(agx3_alu_rcp(AGX_ISA_ENCODING_AGX3C, 0, 0, true, true, false,
                                false, AGX3_ALU_HINT_DISCARD, true, 0, b, &n),
                   rcp);
      CHECK_NATIVE(agx3_alu_hrsqrt(AGX_ISA_ENCODING_AGX3C, 0, 0,
                                   AGX3_ALU_HINT_DISCARD, true, 0, b, &n),
                   rsqrt);
      CHECK_NATIVE(agx3_alu_sqrt_step_half(AGX_ISA_ENCODING_AGX3C, 0, 2,
                                           AGX3_ALU_HINT_CACHE, true, 0, b, &n),
                   sqrt);
      CHECK_NATIVE(
         agx3_alu_hfma_addend_minifloat(AGX_ISA_ENCODING_AGX3C, 0, same, 48,
                                        dead, true, 0, b, &n),
         fma);
      CHECK_NATIVE(agx3_alu_hadd_mod_m5(0, add_pair, dead, abs, neg, true,
                                        false, 0, b, &n),
                   add);
      /* The G17G fsat lowering adds the minifloat negative zero and needs
       * saturation: field 14 selects the wider ep2 form (0x81 in byte 4). */
      const uint8_t sat_add[] = {0x10, 0x80, 0x1c, 0x13,
                                 0x81, 0x00, 0x00, 0x02};
      CHECK_NATIVE(agx3_alu_hadd_minifloat(AGX_ISA_ENCODING_AGX3C, 2, 19,
                                           0x80, false, true,
                                           AGX3_ALU_HINT_DISCARD, false, 0,
                                           b, &n),
                   sat_add);
      CHECK_NATIVE(
         agx3_alu_half_add_imm_to_f32_m5(0, 0, 48, true, true,
                                         AGX3_ALU_HINT_DISCARD, true, 0, b, &n),
         mixed_add);
      CHECK_NATIVE(agx3_alu_fmul_narrow(AGX_ISA_ENCODING_AGX3C, 0, pair, dead,
                                        true, 0, b, &n),
                   mul);
      /* The buffer inputs force real half operands. Apple evaluates these
       * in the constant program; the operand widths are not inferred from
       * the half render-target format. */
      const uint8_t half_abs[] = {0x20, 0x04, 0x04, 0x02, 0x2a,
                                  0x20, 0xa0, 0x02, 0x84, 0x26};
      const uint8_t half_neg[] = {0x00, 0x00, 0x3c, 0x80,
                                  0x01, 0x22, 0x02, 0x00};
      const uint8_t half_negabs[] = {0x20, 0x04, 0x04, 0x02, 0x2a,
                                     0x20, 0xa0, 0x02, 0x84, 0xa6};
      const uint8_t mixed_narrow[] = {0x01, 0x01, 0x3d, 0x04, 0x00, 0x20};
      const uint8_t float_uint[] = {0x27, 0x04, 0x00, 0x1a, 0x2a,
                                    0x00, 0xae, 0x02, 0xb0, 0x03};
      CHECK_NATIVE(agx3_alu_half_add_negzero_m5(
                      0, 0, true, false, false, AGX3_ALU_HINT_DISCARD, true, 1, b, &n),
                   half_abs);
      CHECK_NATIVE(agx3_alu_half_add_negzero_m5(
                      0, 0, false, true, false, AGX3_ALU_HINT_DISCARD, true, 1, b, &n),
                   half_neg);
      CHECK_NATIVE(agx3_alu_half_add_negzero_m5(
                      0, 0, true, true, false, AGX3_ALU_HINT_DISCARD, true, 1, b, &n),
                   half_negabs);
      CHECK_NATIVE(agx3_alu_fmul_mixed_half(
                      AGX_ISA_ENCODING_AGX3C, 0, 0, 4, AGX3_ALU_HINT_DISCARD,
                      AGX3_ALU_HINT_DISCARD, true, 1, b, &n),
                   mixed_narrow);
      CHECK_NATIVE(
         agx3_alu_float_to_int32(AGX_ISA_ENCODING_AGX3C, 0, 0, false,
                                 AGX3_ALU_HINT_DISCARD, true, 1, b, &n),
         float_uint);
      const uint16_t half_pair[] = {1, 0};
      const bool both_abs[] = {true, true};
      const uint8_t abs_product[] = {
         0x20,0x05,0x05,0x02,0x2a,0x20,0xa4,0x02,0x24,0x80};
      const uint8_t signed_half[] = {
         0x27,0x00,0x04,0x1a,0x29,0x00,0xa3,0x02,0x20,0x00,0x4b,0x00};
      CHECK_NATIVE(agx3_alu_hmul_abs(AGX_ISA_ENCODING_AGX3C,0,half_pair,
                   dead,both_abs,false,true,1,b,&n), abs_product);
      const struct agx3_isub_imm_desc sign_extend = {
         .dest = 0, .src = 0, .src_16 = true, .sign_extend = true,
         .dest_cache = true, .src_kill = true,
      };
      CHECK_NATIVE(agx3_iadd_immediate_encode(AGX_ISA_ENCODING_AGX3C,
                   &sign_extend,b,&n), signed_half);
#undef CHECK_NATIVE
      puts("M5_HALF_MATH_NATIVE_PASS");
   }

   {
      const struct agx3_falu_mods m = {.src0_16 = true};
      const struct agx3_falu_runtime runtime = {
         .dest_cache = true, .discard = {true, true},
      };
      const uint8_t expected_mul[] = {0x09,0x0f,0x3d,0x00,0x10,0x00};
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], length;
      if (!agx3_falu_mul_mod_runtime(AGX_ISA_ENCODING_AGX3C, 0, 1, 14,
                                    &m, &runtime, bytes, &length) ||
          length != sizeof(expected_mul) ||
          memcmp(bytes, expected_mul, length)) return 1;
      puts("M5_MIXED_MUL_NATIVE_PASS");
   }
   {
      const uint8_t native[] = {0x37,0x02,0x04,0x38,0x2a,0x20,0xa2,0x12,0x10,0x00};
      uint8_t bytes[AGX3_ALU_MAX_LENGTH], length;
      if (!agx3_alu_sqrt_step_source_modifiers(AGX_ISA_ENCODING_AGX3C,
              10, 6, AGX3_ALU_HINT_NONE, true, 0, false, false, bytes, &length) ||
          length != sizeof(native) || memcmp(bytes, native, length)) return 1;
      puts("M5_SQRT_STEP_NATIVE_PASS");
   }
   /* Native level-zero samples isolate each mask bit, then compose XZ,
    * XW, XZW and XYZW without changing the operand ABI. */
   {
      static const unsigned masks[] = {1, 2, 4, 8, 5, 9, 13, 15};
      static const unsigned char mask_bytes[][4] = {
         {0x00,0x21,0x01,0x00}, {0x20,0x21,0x01,0x00},
         {0x40,0x21,0x01,0x00}, {0x60,0x21,0x01,0x00},
         {0x00,0x21,0x03,0x00}, {0x00,0x29,0x01,0x00},
         {0x00,0x29,0x03,0x00}, {0x60,0x29,0x01,0x00},
      };
      for (unsigned i = 0; i < 8; ++i) {
         const struct agx3_tex_sample_desc d = {
            .width = AGX3_TEX_W32, .mask = masks[i],
            .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
            .coord_kill = true, .final = true,
         };
         unsigned char bytes[AGX3_TEX_MAX_LENGTH]; uint8_t length;
         unsigned char native[] = {
            0x0f,0x04,0x16,0x00,0x41,0x80,0x00,0x21,0x01,0x00,0x01,
            0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00,
         };
         memcpy(native + 6, mask_bytes[i], 4);
         if (!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &d, bytes,
                                    &length) || length != sizeof(native) ||
             memcmp(bytes, native, length)) return 1;
         /* metal625 sparse_sample/gather/read all carry f142 in byte15[7]
          * of the 22-byte form. The driver's operand profile stays intact. */
         struct agx3_tex_sample_desc sparse = d;
         sparse.sparse = true;
         native[15] |= 0x80;
         if (!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &sparse, bytes,
                                     &length) || length != sizeof(native) ||
             memcmp(bytes, native, length)) return 1;
      }
      puts("M5_SAMPLE_COMPONENT_MASK_NATIVE_PASS");
   }
   {
      const unsigned char native[] = {0x27, 0, 0, 0x1a, 0x2e, 0, 0xa9, 2, 8, 0};
      unsigned char bytes[AGX3_ALU_MAX_LENGTH];
      uint8_t length;
      if (!agx3_alu_bitrev_m5(0, 0, false, AGX3_ALU_HINT_DISCARD, true, 0,
                              bytes, &length) ||
          length != sizeof(native) || memcmp(bytes, native, length))
         return 1;
      puts("M5_BIT_REVERSE_NATIVE_PASS");
   }
   /* metal-fence189: a memory-only device fence differs from a rendezvous. */
   const unsigned char native_fence[] = {0x0f,0x2b,0x00,0x06,0x00,0x00};
   struct agx3_encoded_scoped_barrier fence;
   if (!agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3C,
                                AGX3_BARRIER_SCOPE_DEVICE_FENCE, &fence) ||
       fence.length != sizeof(native_fence) ||
       memcmp(fence.bytes, native_fence, sizeof(native_fence))) return 1;
   puts("M5_DEVICE_FENCE_PASS");
   memset(&fence, 0xa5, sizeof(fence));
   if (!agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3C,
                                AGX3_BARRIER_SCOPE_SUBGROUP_SHARED, &fence) ||
       fence.length != 0 ||
       !agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3B,
                                AGX3_BARRIER_SCOPE_SUBGROUP_SHARED, &fence) ||
       fence.length != 0)
      return 1;
   puts("M5_SUBGROUP_SHARED_NO_RENDEZVOUS_PASS");


   static const unsigned char expected[] = {
      0x1c,0x9c,0x10,0x06,0x24,0xa8,0x14,0x06,
      0x0c,0x9d,0x10,0x26,0x24,0x81,0x01,0x40,
      0x28,0x01,0xa0,0x0a,0x27,0x06,0x04,0x0a,
      0x21,0x00,0xa1,0x02,0x68,0x01,0x20,0x00,
      0x01,0x02,0x37,0x08,0x04,0x0a,0x21,0x00,
      0xa1,0x02,0x28,0x01,0x20,0x00,0x01,0x02,
      0x27,0x00,0x20,0x2a,0x21,0x00,0xab,0x0a,
      0x00,0x00,0x4a,0x30,0x18,0x00,0x25,0x04,
      0x15,0x00,0x24,0x00,0xa0,0x22,0x00,0xe0,
      0x00,0x55,0x0e,0x00,0x00,0x00,
   };
   const struct {unsigned d,s,w; bool discard; unsigned char bytes[4];} copies[] = {
      {16,1,1,false,{0x83,0x01,0x21,0x20}},
      {0,1,0,true,{0x03,0x01,0x29,0x00}},
      {1,2,0,true,{0x03,0x02,0x29,0x01}},
   };
   for (unsigned i=0;i<3;i++) {
      unsigned char out[AGX3_COPY_MAX_LENGTH]; uint8_t len=0;
      if (!agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3C,copies[i].d,
             copies[i].s,true,false,copies[i].discard,copies[i].w,out,&len) ||
          len!=4 || memcmp(out,copies[i].bytes,4)) return 2;
   }
   unsigned char out[MAX2(AGX3_STAGE_MAX_LENGTH, AGX3_ALU_MAX_LENGTH)];
   uint8_t len=0;
   const unsigned char reciprocal[] = {
      0x37,0x06,0x04,0x18,0x2a,0x20,0xa1,0x02,0x30,0x00,
   };
   if (!agx3_alu_rcp(AGX_ISA_ENCODING_AGX3C,2,2,false,false,false,false,
                     AGX3_ALU_HINT_DISCARD,true,1,out,&len) ||
       len!=sizeof(reciprocal) || memcmp(out,reciprocal,len)) return 1;
   const unsigned char uniform_half[] = {0x13,0x08,0x01,0x08};
   if (!agx3_copy_encode_uniform_to_reg16(AGX_ISA_ENCODING_AGX3C,2,8,
                                          false,out,&len) ||
       len!=sizeof(uniform_half) || memcmp(out,uniform_half,len)) return 1;
   const unsigned char fragment_entry[] = {0x07,0x30,0x00,0x05,0x00,0x00};
   if (!agx3_pixel_colour_entry(AGX_ISA_ENCODING_AGX3C,out,&len) ||
       len!=sizeof(fragment_entry) || memcmp(out,fragment_entry,len)) return 1;
   /* Native zs_depth_4x / zs_stencil_4x entry gate (f34), AGX3c only. */
   const unsigned char zs_unreleased_entry[] = {0x07,0x70,0x00,0x05,0x00,0x00};
   if (!agx3_pixel_colour_entry_zs_unreleased(AGX_ISA_ENCODING_AGX3C,out,&len) ||
       len!=sizeof(zs_unreleased_entry) ||
       memcmp(out,zs_unreleased_entry,len)) return 1;
   if (agx3_pixel_colour_entry_zs_unreleased(AGX_ISA_ENCODING_AGX3B,out,&len))
      return 1;
   const unsigned char zs_release[] = {0x0f,0x00,0x00,0x04,0x00,0x00};
   if (!agx3_pixel_release(AGX_ISA_ENCODING_AGX3C,1,false,0,out,&len) ||
       len!=sizeof(zs_release) || memcmp(out,zs_release,len)) return 1;
   const unsigned char clear[]={0x13,0,0,0};
   if (!agx3_clr(AGX_ISA_ENCODING_AGX3C,2,true,false,out,&len) ||
       len!=4 || memcmp(out,clear,4)) return 3;
   puts("M5_COPY_CLEAR_PASS");
   const struct agx3_imad_desc mad = {
      .dest=2, .src_a=8, .multiplier=32, .src_c=2,
      .src_a_kill=true, .src_c_kill=true, .wait_mask=9,
   };
   const unsigned char mad_bytes[] = {
      0x37,0x24,0x04,0x5a,0x21,0x00,0xa1,0x02,0x28,0x08,0xca,0x00,0x02,0x00,
   };
   unsigned char mad_out[AGX3_IADD_MAX_LENGTH]; len=0;
   if (!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C,&mad,mad_out,&len) ||
       len!=sizeof(mad_bytes) || memcmp(mad_out,mad_bytes,sizeof(mad_bytes))) return 4;
   puts("M5_IMAD_PASS");
   const struct agx3_iadd_desc add = {
      .dest=24, .src_a_is_register=true, .src_a_reg=2, .src_a_cache=true,
      .src=16, .src_kill=true, .wait_mask=16,
   };
   const unsigned char add_bytes[] = {
      0xa7,0xc2,0x04,0x1a,0x21,0x00,0xa3,0x12,0x08,0x84,0x21,0x00,
   };
   len=0;
   if (!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3C,&add,mad_out,&len) ||
       len!=sizeof(add_bytes) || memcmp(mad_out,add_bytes,sizeof(add_bytes))) return 5;
   struct agx3_mem_desc load = {
      .data=16, .width=AGX3_MEM_W32, .mask=7, .base_is_uniform=true,
      .base=8, .idx_kind=AGX3_MEM_IDX_IMM, .byte_off=776, .slot=5,
   };
   const unsigned char load_bytes[] = {
      0x8f,0x08,0x03,0x00,0x40,0x0c,0x10,0xa1,0x01,0,0,0,0,3,
   };
   unsigned char memory[AGX3_MEM_MAX_LENGTH];len=0;
   if (!agx3_mem_encode_load(AGX_ISA_ENCODING_AGX3C,&load,memory,&len) ||
       len!=sizeof(load_bytes) || memcmp(memory,load_bytes,sizeof(load_bytes))) return 6;
   const struct agx3_isel_desc sel = {
      .dest=1,.dest_16=true,.dest_cache=true,.cmp0=2,.cmp1=12,
      .cond=AGX3_ISEL_ICOND_ULT,.true_kind=AGX3_ISEL_IMMEDIATE,
      .false_kind=AGX3_ISEL_IMMEDIATE,.sel_true=1,.sel_false=0,
      .discard={true,true,false,false},.wait_mask=64,
   };
   const unsigned char sel_bytes[] = {
      0x22,0x02,0x45,0x0b,0x2f,0x00,0xac,0x02,0x00,0x83,0x64,0x98,0x00,0x28,
   };
   unsigned char select[AGX3_ISEL_MAX_LENGTH];len=0;
   if (!agx3_isel_encode(AGX_ISA_ENCODING_AGX3C,&sel,select,&len) ||
       len!=sizeof(sel_bytes) || memcmp(select,sel_bytes,sizeof(sel_bytes))) return 7;
   puts("M5_MEMORY_SELECT_LIVENESS_PASS");
   const unsigned char triple_bytes[] = {
      0x27,0x04,0x04,0x1a,0x29,0x00,0xa3,0x02,0x20,0x00,0x24,0x00,
   };
   const struct agx3_iadd_desc triple = {
      .dest=0, .dest_cache=true, .src_a_is_register=true,
      .src_a_reg=0, .src_a_16=true, .src_a_kill=true,
      .src=0, .src_16=true, .src_kill=true, .shift=1, .wait_mask=1,
   };
   len=0;
   if (!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3C,&triple,mad_out,&len) ||
       len!=sizeof(triple_bytes) || memcmp(mad_out,triple_bytes,len)) return 8;
   const unsigned char add_scaled_bytes[] = {
      0x27,0x08,0x04,0x3a,0x21,0x00,0xa3,0x02,0x20,0x80,0x24,0x00,
   };
   const struct agx3_iadd_desc add_scaled = {
      .dest=0, .src_a_is_register=true, .src_a_reg=4,
      .src_a_16=true, .src_a_kill=true, .src=0, .src_kill=true,
      .shift=1, .wait_mask=2,
   };
   len=0;
   if (!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3C,&add_scaled,mad_out,&len) ||
       len!=sizeof(add_scaled_bytes) || memcmp(mad_out,add_scaled_bytes,len)) return 9;
   puts("M5_NARROW_SCALED_ADD_PASS");
   const uint16_t fma_src[] = {2, 3, 4};
   const enum agx3_alu_hint fma_hint[] = {
      AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD,
   };
   const unsigned char fma_expected[] = {
      0x38,0x02,0x06,0x02,0x23,0x20,0xa4,0x02,0xc6,0x00,0x80,0x00,
   };
   unsigned char native[AGX3_TEX_MAX_LENGTH];
   if (!agx3_alu_hfma(AGX_ISA_ENCODING_AGX3C, 2, fma_src, fma_hint,
                       false, false, false, 128, native, &len) ||
       len != sizeof(fma_expected) || memcmp(native, fma_expected, len)) return 10;
   const struct agx3_iadd_desc sat_sub = {
      .dest=0, .src_a_is_register=true, .src_a_reg=0, .src_a_kill=true,
      .src=4, .src_kill=true, .subtract=true, .saturate=true, .wait_mask=128,
   };
   const unsigned char sat_expected[] = {
      0x2f,0x00,0x04,0x1a,0x25,0x00,0xa2,0x02,0x18,0x81,0x20,0x00,
   };
   if (!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3C, &sat_sub, native, &len) ||
       len != sizeof(sat_expected) || memcmp(native, sat_expected, len)) return 11;
   const struct agx3_local_atomic_desc atom = {
      .op=AGX3_ATOMIC_ADD, .value=0, .value_kill=true,
      .address_kind=AGX3_ATOMIC_INDEX_IMMEDIATE,
   };
   const unsigned char atom_expected[] = {
      0x0f,0x00,0x03,0x00,0x02,0x05,0x04,0xa0,0x90,0x00,0x40,0x02,
   };
   if (!agx3_atomic_encode_local(AGX_ISA_ENCODING_AGX3C, &atom, native, &len) ||
       len != sizeof(atom_expected) || memcmp(native, atom_expected, len)) return 12;
   puts("M5_FMA_SAT_ATOMIC_NATIVE_BYTES_PASS");
   const unsigned char sub16_expected[] = {
      0x3f,0x02,0x04,0x18,0x25,0x00,0xa2,0x02,0x40,0x00,0x0b,0x00,
   };
   if (!agx3_alu_isub_imm16_m5(2, 2, 1, AGX3_ALU_HINT_DISCARD, false,
                              128, native, &len) ||
       len != sizeof(sub16_expected) || memcmp(native, sub16_expected, len)) return 13;
   const unsigned char and_expected[] = {
      0x2b,0x00,0x07,0x80,0x30,0x30,0xa4,0x02,0x04,0x01,
   };
   if (!agx3_alu_and_mixed_m5(0, 0, 4, AGX3_ALU_HINT_DISCARD,
                             AGX3_ALU_HINT_DISCARD, false, 128, native, &len) ||
       len != sizeof(and_expected) || memcmp(native, and_expected, len)) return 14;
   const unsigned char scale_expected[] = {
      0x29,0x00,0x05,0x10,0x22,0x80,0xac,0x02,0x0b,0x00,0x20,0x00,0x00,0x10,
   };
   if (!agx3_alu_fmul_minifloat_narrow(AGX_ISA_ENCODING_AGX3C, 0, 0, 64,
          false, false, false, false, AGX3_ALU_HINT_DISCARD, 128, native, &len) ||
       len != sizeof(scale_expected) || memcmp(native, scale_expected, len)) return 15;
   puts("M5_MIXED_AND_HALF_SUB_SCALE_NATIVE_BYTES_PASS");
   const unsigned char add64_expected[] = {
      0x3f,0x00,0x04,0x18,0x25,0x00,0xa3,0x02,0xa1,0x00,0x21,0x00,
   };
   if (!agx3_alu_iadd16_gpu1(2, false, true, 64, AGX3_ALU_HINT_NONE,
                            2, AGX3_ALU_HINT_DISCARD, 0, 128, native, &len) ||
       len != sizeof(add64_expected) || memcmp(native, add64_expected, len)) return 16;
   const uint16_t andc_src[] = {0,4};
   const enum agx3_alu_hint andc_hints[] = {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
   const unsigned char andc_expected[] = {
      0x2b,0x00,0x06,0x80,0x32,0x30,0xa4,0x02,0x04,0x81,
   };
   if (!agx3_alu_bit_andc(AGX_ISA_ENCODING_AGX3C, 0, andc_src, andc_hints,
                         false, 128, native, &len) ||
       len != sizeof(andc_expected) || memcmp(native, andc_expected, len)) return 17;
   puts("M5_ADD64_ANDC_NATIVE_BYTES_PASS");
   struct agx3_imad_desc mixed_mad = {
      .dest=0, .src_a=0, .src_a_kill=true,
      .multiplier_is_register=true, .multiplier_reg=8,
      .multiplier_16=true, .multiplier_kill=true,
      .src_c=4, .src_c_kill=true, .wait_mask=128,
   };
   const unsigned char mixed_mad_expected[] = {
      0x2f,0x00,0x04,0x1a,0x21,0x00,0xa1,0x02,0x28,0x02,0xe0,0x00,0x04,0x00,
   };
   if (!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C, &mixed_mad, native, &len) ||
       len != sizeof(mixed_mad_expected) ||
       memcmp(native, mixed_mad_expected, len)) return 18;
   mixed_mad.multiplier_sign_extend=true;
   unsigned char signed_mad_expected[sizeof(mixed_mad_expected)];
   memcpy(signed_mad_expected, mixed_mad_expected, sizeof(mixed_mad_expected));
   signed_mad_expected[11] = 0x80;
   if (!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C, &mixed_mad, native, &len) ||
       len != sizeof(signed_mad_expected) ||
       memcmp(native, signed_mad_expected, len)) return 19;
   const struct agx3_imad_desc wide_mad = {
      .dest=0, .dest_64=true, .src_a=0, .src_a_kill=true,
      .multiplier=96, .addend_is_immediate=true, .wait_mask=128,
   };
   const unsigned char wide_mad_expected[] = {
      0x2f,0x00,0x04,0x02,0x21,0x80,0xa9,0x02,0x28,0x18,0x02,0x30,
   };
   if (!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C, &wide_mad, native, &len) ||
       len != sizeof(wide_mad_expected) ||
       memcmp(native, wide_mad_expected, len)) return 20;
   mixed_mad.src_a=4;
   mixed_mad.src_a_16=true;
   mixed_mad.multiplier_reg=5;
   mixed_mad.multiplier_sign_extend=false;
   mixed_mad.src_c=0;
   const unsigned char half_pair_expected[] = {
      0x2f,0x00,0x04,0x3a,0x21,0x00,0xa1,0x02,0x60,0x01,0xe0,0x00,0x00,0x00,
   };
   if (!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C, &mixed_mad, native, &len) ||
       len != sizeof(half_pair_expected) ||
       memcmp(native, half_pair_expected, len)) return 21;
   puts("M5_MIXED_IMAD_WIDE_IMMEDIATE_NATIVE_BYTES_PASS");
   const unsigned char asr_expected[] = {
      0x3f,0x02,0x00,0x18,0x26,0x00,0xa3,0x02,0x30,0x82,0x40,0x00,
   };
   if (!agx3_alu_asr_imm_m5(2,2,true,true,8,AGX3_ALU_HINT_DISCARD,
          false,128,native,&len) || len != sizeof(asr_expected) ||
       memcmp(native,asr_expected,len)) { fprintf(stderr,"ASR len=%u ",len); for(unsigned i=0;i<len;i++) fprintf(stderr,"%02x",native[i]); fputc(10,stderr); return 30; }
   const struct agx3_imad_desc half_addend = {
      .dest=0,.src_a=1,.src_a_16=true,.src_a_kill=true,
      .multiplier=129,.src_c=0,.src_c_16=true,.src_c_kill=true,.wait_mask=128,
   };
   const unsigned char half_addend_expected[] = {
      0x2f,0x01,0x04,0x1a,0x21,0x00,0xa1,0x02,0x60,0x20,0x8a,0x00,0x00,0x00,
   };
   if (!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C,&half_addend,native,&len) ||
       len != sizeof(half_addend_expected) ||
       memcmp(native,half_addend_expected,len)) return 31;
   const struct agx3_falu_mods abs_add = {.src0_abs=true,.src1_abs=true};
   const struct agx3_falu_runtime abs_runtime = {
      .discard={true,true,false},.wait_mask=128,
   };
   const unsigned char abs_expected[] = {
      0x29,0x00,0x04,0x1a,0x22,0x20,0xa0,0x02,0x04,0x88,0x12,0x40,
   };
   if (!agx3_falu_add_mod_runtime(AGX_ISA_ENCODING_AGX3C,0,0,4,&abs_add,
          &abs_runtime,native,&len) || len != sizeof(abs_expected) ||
       memcmp(native,abs_expected,len)) return 32;
   const struct agx3_isel_desc mixed_select = {
      .dest=2,.dest_16=true,.cmp0=2,.cmp1=4,.cmp_mixed_width=true,
      .cond=AGX3_ISEL_ICOND_UGT,.true_kind=AGX3_ISEL_IMMEDIATE,
      .false_kind=AGX3_ISEL_IMMEDIATE,.sel_true=1,.sel_false=0,
      .discard={true,true,false,false},.wait_mask=128,
   };
   const unsigned char mixed_select_expected[] = {
      0x32,0x02,0x05,0x0a,0x27,0x00,0xac,0x02,0x00,0x01,0x64,0x1c,0x00,0x38,
   };
   if (!agx3_isel_encode(AGX_ISA_ENCODING_AGX3C,&mixed_select,native,&len) ||
       len != sizeof(mixed_select_expected) ||
       memcmp(native,mixed_select_expected,len)) return 33;
   puts("M5_ASR_HALF_ADDEND_ABS_MIXED_SELECT_NATIVE_BYTES_PASS");
   const uint16_t srl_src[] = {4,0};
   const bool srl_half[] = {true,false};
   const enum agx3_alu_hint srl_hints[] = {AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD};
   const unsigned char srl_expected[] = {
      0x27,0x04,0x00,0x3a,0x26,0x00,0xa1,0x02,0x00,0x80,0x20,0x34,0x00,0x00,
   };
   if (!agx3_alu_srl_reg(AGX_ISA_ENCODING_AGX3C,0,false,false,srl_src,
          srl_half,srl_hints,1,native,&len) || len != sizeof(srl_expected) ||
       memcmp(native,srl_expected,len)) return 34;
   const uint16_t narrow_bit_src[] = {3,2};
   const enum agx3_alu_hint narrow_bit_hints[] = {AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD};
   const enum agx3_alu_op narrow_bit_op[] = {AGX3_ALU_BAND,AGX3_ALU_BOR,AGX3_ALU_BXOR};
   const unsigned char narrow_bit_expected[][10] = {
      {0x3b,0x03,0x07,0x00,0x30,0x20,0xa4,0x02,0x84,0x00},
      {0x3b,0x03,0x07,0x00,0x32,0x28,0xa4,0x02,0x84,0x00},
      {0x3b,0x03,0x06,0x00,0x32,0x28,0xa4,0x02,0x84,0x00},
   };
   for (unsigned i=0;i<3;i++) {
      if (!agx3_alu_bit_reg16(AGX_ISA_ENCODING_AGX3C,narrow_bit_op[i],2,
             narrow_bit_src,narrow_bit_hints,128,native,&len) || len != 10 ||
          memcmp(native,narrow_bit_expected[i],len)) return 35+i;
   }
   puts("M5_NARROW_BITOPS_SRL_NATIVE_BYTES_PASS");
   const unsigned char half_int_expected[][10] = {
      {0x2f,0x00,0x00,0x1a,0x22,0x00,0xae,0x02,0xa8,0x03},
      {0x2f,0x00,0x00,0x1a,0x22,0x00,0xae,0x02,0x68,0x01},
   };
   for (unsigned sign=0;sign<2;sign++) {
      if (!agx3_alu_f16_to_int32(AGX_ISA_ENCODING_AGX3C,0,0,sign,
             AGX3_ALU_HINT_DISCARD,false,128,native,&len) || len != 10 ||
          memcmp(native,half_int_expected[sign],len)) return 38+sign;
   }
   const struct agx3_imad_desc wide_half_mad = {
      .dest=0,.dest_64=true,.src_a=0,.src_a_kill=true,.src_a_sign_extend=true,
      .multiplier_is_register=true,.multiplier_reg=2,.multiplier_16=true,
      .multiplier_kill=true,.multiplier_sign_extend=true,
      .addend_is_immediate=true,.wait_mask=128,
   };
   const unsigned char wide_half_expected[] = {
      0x2f,0x00,0x04,0x02,0x21,0x08,0xa9,0x02,0xa8,0x00,0x20,0xb0,
   };
   if (!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C,&wide_half_mad,native,&len) ||
       len != sizeof(wide_half_expected) ||
       memcmp(native,wide_half_expected,len)) return 40;
   puts("M5_HALF_CONVERSION_WIDE_MULTIPLY_NATIVE_BYTES_PASS");
   const struct agx3_fsel32_gpu1_desc fsel_half_reg = {
      .dest=2,.dest_cache=true,.cmp0=2,.cmp1=4,
      .cmp0_hint=AGX3_ALU_HINT_DISCARD,.cmp1_hint=AGX3_ALU_HINT_DISCARD,
      .sel={6,0},.sel_immediate={false,true},
      .sel_hint={AGX3_ALU_HINT_CACHE,AGX3_ALU_HINT_NONE},
      .condition=0,.wait_mask=128,
   };
   const unsigned char fsel_half_reg_expected[] = {
      0x32,0x02,0x05,0x0a,0x2b,0x00,0xa0,0x02,0x16,0x01,0x21,0x18,0x03,0x00,0x80,0x00,
   };
   if (!agx3_alu_fcmp_select16_gpu1(&fsel_half_reg,native,&len) ||
       len != sizeof(fsel_half_reg_expected) ||
       memcmp(native,fsel_half_reg_expected,len)) return 41;
   puts("M5_FLOAT_REGISTER_SELECT_NATIVE_BYTES_PASS");
   const struct agx3_iadd_desc mixed_sub = {
      .dest=0,.src_a_is_register=true,.src_a_reg=4,.src_a_16=true,
      .src_a_kill=true,.src=0,.src_kill=true,.subtract=true,.wait_mask=128,
   };
   const unsigned char mixed_sub_expected[] = {
      0x2f,0x00,0x04,0x3a,0x25,0x00,0xa2,0x02,0x00,0x80,0x21,0x00,
   };
   if (!agx3_iadd_encode(AGX_ISA_ENCODING_AGX3C,&mixed_sub,native,&len) ||
       len != sizeof(mixed_sub_expected) ||
       memcmp(native,mixed_sub_expected,len)) return 42;
   puts("M5_MIXED_SUBTRACT_NATIVE_BYTES_PASS");
   const unsigned char active_mask_expected[] = {
      0x27,0x01,0x20,0x1a,0x25,0x00,0xae,0x02,0x30,0x00,0x08,0x40,
   };
   if (!agx3_subgroup_encode_active_mask(0,false,0,native,&len) || len != 12 ||
       memcmp(native,active_mask_expected,len)) return 43;
   const struct agx3_subgroup_ballot_desc ballot = {
      .dest=0,.src=0,.type=AGX3_SUBGROUP_BALLOT_INTEGER,
      .compare_immediate=true,.condition=4,.invert=true,.wait_mask=128,
   };
   const unsigned char ballot_expected[] = {
      0x2f,0x00,0x00,0x1a,0x25,0x00,0xae,0x02,0x38,0x00,0x08,0x04,
   };
   if (!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C,&ballot,native,&len) ||
       len != 12 || memcmp(native,ballot_expected,len)) return 44;
   const struct agx3_subgroup_shuffle_desc quad = {
      .quad=true,.op=AGX3_SUBGROUP_SHUFFLE_BROADCAST,
      .lane_kind=AGX3_SUBGROUP_LANE_IMMEDIATE,
      .dest=0,.src=0,.lane=0,.wait_mask=128,
   };
   const unsigned char quad_expected[] = {
      0x2f,0x00,0x21,0x1a,0x20,0x00,0xa0,0x02,0x38,0x00,
   };
   if (!agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3C,&quad,native,&len) ||
       len != 10 || memcmp(native,quad_expected,len)) return 45;
   const unsigned char int8_float_expected[] = {
      0x2f,0x00,0x00,0x02,0x22,0x80,0xaf,0x02,0x70,0x20,
   };
   if (!agx3_alu_narrow_int_to_float(AGX_ISA_ENCODING_AGX3C,0,false,0,true,
          true,AGX3_ALU_HINT_DISCARD,false,128,native,&len) || len != 10 ||
       memcmp(native,int8_float_expected,len)) return 46;
   puts("M5_BALLOT_QUAD_INT8_FLOAT_NATIVE_BYTES_PASS");
   const unsigned char mixed_xor_expected[] = {
      0x2b,0x00,0x06,0x80,0x32,0x38,0xa4,0x02,0x04,0x01,
   };
   if (!agx3_alu_bit_mixed_m5(AGX3_ALU_BXOR,0,0,4,AGX3_ALU_HINT_DISCARD,
          AGX3_ALU_HINT_DISCARD,false,128,native,&len) || len != 10 ||
       memcmp(native,mixed_xor_expected,len)) return 47;
   const struct agx3_subgroup_shuffle_desc half_bcast = {
      .op=AGX3_SUBGROUP_SHUFFLE_BROADCAST,.lane_kind=AGX3_SUBGROUP_LANE_IMMEDIATE,
      .dest=2,.src=2,.half=true,.wait_mask=128,
   };
   const unsigned char half_bcast_expected[] = {
      0x3f,0x02,0x21,0x18,0x20,0x00,0xa8,0x02,0x30,0x00,
   };
   if (!agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3C,&half_bcast,native,&len) ||
       len != 10 || memcmp(native,half_bcast_expected,len)) return 48;
   const struct agx3_subgroup_shuffle_desc reg_bcast = {
      .op=AGX3_SUBGROUP_SHUFFLE_BROADCAST,.lane_kind=AGX3_SUBGROUP_LANE_REGISTER,
      .dest=4,.src=0,.lane=4,.dest_hint=true,.src_cache=true,
      .lane_discard=true,.wait_mask=1,
   };
   const unsigned char reg_bcast_expected[] = {
      0x27,0x84,0x01,0x1a,0x28,0x00,0xa8,0x0a,0x08,0x81,
   };
   if (!agx3_subgroup_encode_shuffle(AGX_ISA_ENCODING_AGX3C,&reg_bcast,native,&len) ||
       len != 10 || memcmp(native,reg_bcast_expected,len)) return 49;
   puts("M5_MIXED_XOR_HALF_AND_REGISTER_BROADCAST_NATIVE_BYTES_PASS");
   const unsigned char ft1_expected[] = {
      0x2f,0x80,0x00,0x1a,0x2a,0x00,0xab,0x0a,0x08,0x00,
   };
   if (!agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3C,4,0,AGX3_ALU_HINT_CACHE,
          true,128,native,&len) || len != 10 || memcmp(native,ft1_expected,len)) return 50;
   const unsigned char pop_expected[] = {
      0x2f,0x00,0x00,0x80,0x32,0x30,0xaa,0x02,0x04,0x00,
   };
   if (!agx3_alu_popcount(AGX_ISA_ENCODING_AGX3C,0,0,false,
          AGX3_ALU_HINT_DISCARD,false,128,native,&len) || len != 10 ||
       memcmp(native,pop_expected,len)) return 51;
   /* Fresh G17G fma_loop: fma(r1, 0.75, 1.0), no dependency. The source
    * discard is independent of the absent slot-8 wait. */
   const struct agx3_falu_runtime fma2_no_wait = {.discard={true}};
   const unsigned char fma2_short[] = {0x19,0x03,0x0e,0xa9,0x21,0xb0,0x02,0x02};
   if (!agx3_falu_fma_two_imm(AGX_ISA_ENCODING_AGX3C,2,2,40,48,
          &fma2_no_wait,native,&len) || len != sizeof(fma2_short) ||
       memcmp(native,fma2_short,len)) return 102;

   /* Native G17G Minecraft terrain fragments: fma.sat(a, b, 0.5)
    * keeps its minifloat addend (f51 = 32, f14 = 1) beside two discarded
    * GPR multipliers and a cached result. */
   {
      const struct agx3_falu_mods sat = {.saturate = true};
      const struct agx3_falu_runtime sat_runtime = {
         .discard = {true, true}, .dest_cache = true,
      };
      const struct { uint16_t dest, src0, src1; unsigned char bytes[10]; } sat_cases[] = {
         {88, 96, 88, {0xc9,0x59,0xbe,0x61,0x22,0xa0,0x02,0x00,0x00,0x02}},
         {90, 92, 90, {0xd9,0x5b,0xbe,0x5d,0x22,0xa0,0x02,0x00,0x00,0x02}},
         {18, 22, 18, {0x99,0x13,0x3e,0x17,0x22,0xa0,0x02,0x00,0x00,0x02}},
      };
      for (unsigned i = 0; i < 3; ++i) {
         if (!agx3_falu_fma_addend_imm_width_mod(AGX_ISA_ENCODING_AGX3C,
                sat_cases[i].dest, sat_cases[i].src0, sat_cases[i].src1, 32,
                &sat, &sat_runtime, native, &len) ||
             len != 10 || memcmp(native, sat_cases[i].bytes, len)) {
            fprintf(stderr, "fma.sat addend witness %u: len %u\n", i, len);
            for (unsigned b = 0; b < len; ++b) fprintf(stderr, "%02x", native[b]);
            fprintf(stderr, "\n");
            return 140;
         }
      }
   }

   /* With the native G17G compiler, a fragment sampling an argument-buffer
    * texture with an argument-buffer sampler at an explicit register LOD:
    * both handles are aligned 64-bit uniforms (f97/f103 = u20/3,
    * f126/f132 = u16/3), otherwise the driver-ABI register sample. */
   {
      const struct agx3_tex_sample_desc uniform_handles = {
         .dest = 16, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
         .lod = AGX3_TEX_LOD_REGISTER, .slot = 2, .wait_mask = 2,
         .coord_kill = true, .sample_group_nonterminal = true,
         .texture_kind = AGX3_TEX_OPERAND_UNIFORM, .texture_value = 20,
         .sampler_kind = AGX3_TEX_OPERAND_UNIFORM, .sampler_value = 16,
         .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
      };
      const unsigned char expected[] = {0x8f,0x07,0x12,0x00,0x8d,0x20,0x60,0x48,0x89,0x18,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00};
      if (!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &uniform_handles,
                                  native, &len) ||
          len != sizeof(expected) || memcmp(native, expected, len)) {
         fprintf(stderr, "uniform-handle smp: len %u ", len);
         for (unsigned b = 0; b < len; ++b) fprintf(stderr, "%02x", native[b]);
         fprintf(stderr, "\n");
         return 141;
      }
   }

   /* Native G17G Minecraft fragments: a select whose first
    * comparand is a 32-bit uniform (f19, f25 = 3) beside a register,
    * choosing a minifloat or a register (fself32). Every distinct one. */
   {
      static const struct {
         uint16_t dest; bool dest_cache; uint16_t uniform, cmp1;
         enum agx3_alu_hint cmp1_hint; uint16_t imm, sel1;
         enum agx3_alu_hint sel1_hint; unsigned condition;
         unsigned char bytes[16]; unsigned length;
      } sel_cases[] = {
      {8, true, 10, 6, AGX3_ALU_HINT_CACHE, 0, 8, AGX3_ALU_HINT_DISCARD, 6, {0x42,0x0b,0x23,0x87,0x25,0x80,0x83,0x00}, 8},
      {20, true, 42, 16, AGX3_ALU_HINT_CACHE, 48, 20, AGX3_ALU_HINT_DISCARD, 6, {0xa2,0x2b,0x23,0x91,0x25,0xb0,0x83,0x00}, 8},
      {20, false, 40, 16, AGX3_ALU_HINT_DISCARD, 0, 20, AGX3_ALU_HINT_DISCARD, 5, {0xa2,0x29,0x13,0x11,0x25,0x80,0x82,0x00}, 8},
      {22, true, 46, 18, AGX3_ALU_HINT_CACHE, 48, 22, AGX3_ALU_HINT_DISCARD, 6, {0xb2,0x2f,0x23,0x93,0x25,0xb0,0x83,0x00}, 8},
      {18, true, 44, 18, AGX3_ALU_HINT_DISCARD, 0, 22, AGX3_ALU_HINT_DISCARD, 5, {0x92,0x2d,0x33,0x13,0x26,0x80,0x82,0x00,0x80,0x16}, 10},
      {12, true, 18, 10, AGX3_ALU_HINT_CACHE, 48, 8, AGX3_ALU_HINT_DISCARD, 6, {0x62,0x13,0x23,0x8b,0x26,0xb0,0x83,0x00,0x80,0x08}, 10},
      {30, false, 16, 10, AGX3_ALU_HINT_DISCARD, 0, 12, AGX3_ALU_HINT_DISCARD, 5, {0xf2,0x11,0x13,0x0b,0x26,0x80,0x82,0x00,0x80,0x0c}, 10},
      {16, true, 22, 8, AGX3_ALU_HINT_CACHE, 48, 10, AGX3_ALU_HINT_DISCARD, 6, {0x82,0x17,0x23,0x89,0x26,0xb0,0x83,0x00,0x80,0x0a}, 10},
      {8, false, 20, 8, AGX3_ALU_HINT_DISCARD, 0, 16, AGX3_ALU_HINT_DISCARD, 5, {0x42,0x15,0x13,0x09,0x26,0x80,0x82,0x00,0x80,0x10}, 10},
      {20, true, 42, 18, AGX3_ALU_HINT_CACHE, 48, 14, AGX3_ALU_HINT_DISCARD, 6, {0xa2,0x2b,0x23,0x93,0x26,0xb0,0x83,0x00,0x80,0x0e}, 10},
      {20, false, 40, 18, AGX3_ALU_HINT_DISCARD, 0, 20, AGX3_ALU_HINT_DISCARD, 5, {0xa2,0x29,0x13,0x13,0x25,0x80,0x82,0x00}, 8},
      {22, false, 46, 12, AGX3_ALU_HINT_NONE, 48, 16, AGX3_ALU_HINT_DISCARD, 6, {0xb2,0x2f,0x03,0x0d,0x26,0xb0,0x83,0x00,0x80,0x10}, 10},
      {12, true, 44, 12, AGX3_ALU_HINT_DISCARD, 0, 22, AGX3_ALU_HINT_DISCARD, 5, {0x62,0x2d,0x33,0x0d,0x26,0x80,0x82,0x00,0x80,0x16}, 10},
      {6, true, 20, 2, AGX3_ALU_HINT_CACHE, 48, 6, AGX3_ALU_HINT_DISCARD, 6, {0x32,0x15,0x23,0x83,0x25,0xb0,0x83,0x00}, 8},
      {4, true, 20, 0, AGX3_ALU_HINT_CACHE, 48, 4, AGX3_ALU_HINT_DISCARD, 6, {0x22,0x15,0x23,0x81,0x25,0xb0,0x83,0x00}, 8},
      {0, true, 20, 0, AGX3_ALU_HINT_DISCARD, 0, 4, AGX3_ALU_HINT_DISCARD, 5, {0x02,0x15,0x33,0x01,0x26,0x80,0x82,0x00,0x80,0x04}, 10},
      };
      for (unsigned i = 0; i < sizeof(sel_cases) / sizeof(sel_cases[0]); ++i) {
         const struct agx3_fsel32_gpu1_desc d = {
            .dest = sel_cases[i].dest, .dest_cache = sel_cases[i].dest_cache,
            .cmp0 = sel_cases[i].uniform, .cmp0_uniform = true,
            .cmp1 = sel_cases[i].cmp1, .cmp1_hint = sel_cases[i].cmp1_hint,
            .sel_immediate = {true, false},
            .sel = {sel_cases[i].imm, sel_cases[i].sel1},
            .sel_hint = {AGX3_ALU_HINT_NONE, sel_cases[i].sel1_hint},
            .condition = sel_cases[i].condition,
         };
         if (!agx3_alu_fcmp_select32_gpu1(&d, native, &len) ||
             len != sel_cases[i].length ||
             memcmp(native, sel_cases[i].bytes, len)) {
            fprintf(stderr, "uniform select witness %u: len %u ", i, len);
            for (unsigned b = 0; b < len; ++b) fprintf(stderr, "%02x", native[b]);
            fprintf(stderr, "\n");
            return 142;
         }
      }
   }

   const struct agx3_falu_runtime fma2_runtime = {.discard={true},.wait_mask=128};
   const unsigned char fma2_expected[] = {
      0x29,0x00,0x07,0x0a,0x20,0x20,0xa3,0x02,0x30,0x8d,0x59,0x00,0x14,0x06,0x00,0x80,
   };
   if (!agx3_falu_fma_two_imm(AGX_ISA_ENCODING_AGX3C,0,0,52,84,
          &fma2_runtime,native,&len) || len != sizeof(fma2_expected) ||
       memcmp(native,fma2_expected,len)) return 52;
   const struct agx3_imad_desc mul_half = {
      .dest=2,.dest_16=true,.src_a=3,.src_a_16=true,.src_a_kill=true,
      .multiplier_is_register=true,.multiplier_reg=2,.multiplier_16=true,
      .multiplier_kill=true,.no_addend=true,.wait_mask=128,
   };
   const unsigned char mul_half_expected[] = {
      0x3f,0x03,0x04,0x18,0x21,0x00,0xa1,0x02,0xa0,0x00,0x20,0x00,0x01,0x02,
   };
   if (!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C,&mul_half,native,&len) ||
       len != sizeof(mul_half_expected) || memcmp(native,mul_half_expected,len)) return 53;
   const enum agx3_alu_hint extract_hint[] = {AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD};
   const unsigned char extract_expected[] = {
      0x27,0x04,0x00,0x1a,0x26,0x00,0xa1,0x02,0x08,0x01,0x20,0x34,0x00,0x10,
   };
   if (!agx3_alu_bfeil_m5(0,0,false,4,true,true,1,extract_hint,1,native,&len) ||
       len != sizeof(extract_expected) || memcmp(native,extract_expected,len)) return 54;
   const struct agx3_subgroup_reduce_desc red_half = {
      .op=AGX3_SUBGROUP_REDUCE_IADD,.dest=2,.src=2,.half=true,.wait_mask=128,
   };
   const unsigned char red_half_expected[] = {
      0x3f,0x02,0x04,0x08,0x27,0x80,0xa3,0x02,0x00,0x02,
   };
   if (!agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C,&red_half,native,&len) ||
       len != sizeof(red_half_expected) || memcmp(native,red_half_expected,len)) return 55;
   puts("M5_FT1_POPCOUNT_FMA2_NARROW_MUL_EXTRACT_REDUCE_NATIVE_BYTES_PASS");
   const unsigned char ft1_alias_expected[] = {
      0x27,0x00,0x00,0x1a,0x2a,0x00,0xab,0x02,0x28,0x00,
   };
   if (!agx3_alu_ft1_u32(AGX_ISA_ENCODING_AGX3C,0,0,AGX3_ALU_HINT_DISCARD,
          true,0,native,&len) || len != 10 || memcmp(native,ft1_alias_expected,len)) return 56;
   const unsigned char convert_short_expected[][10] = {
      {0x3f,0x02,0x00,0x18,0x22,0x00,0xae,0x02,0x30,0x03},
      {0x3f,0x02,0x00,0x18,0x22,0x00,0xae,0x02,0x28,0x03},
   };
   for (unsigned half=0;half<2;half++) {
      if (!agx3_alu_float_to_int16(AGX_ISA_ENCODING_AGX3C,2,2,half,true,
             AGX3_ALU_HINT_DISCARD,false,128,native,&len) || len != 10 ||
          memcmp(native,convert_short_expected[half],len)) return 57+half;
   }
   puts("M5_INPLACE_FT1_FLOAT_TO_SHORT_NATIVE_BYTES_PASS");
   const struct agx3_subgroup_ballot_desc ballot_pair = {
      .src_half=true,.compare_register=true,.compare_value=1,.compare_half=true,
      .compare_discard=true,.condition=4,.invert=true,.wait_mask=128,
   };
   const unsigned char ballot_pair_expected[] = {
      0x2f,0x00,0x00,0x1a,0x25,0x00,0xae,0x02,0x40,0x00,0x28,0x04,
   };
   if (!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C,&ballot_pair,native,&len) ||
       len != 12 || memcmp(native,ballot_pair_expected,len)) return 59;
   const enum agx3_subgroup_reduce_op quad_ops[] = {
      AGX3_SUBGROUP_REDUCE_IADD,AGX3_SUBGROUP_REDUCE_SMIN,AGX3_SUBGROUP_REDUCE_SMAX,
   };
   const unsigned char quad_int_expected[][10] = {
      {0x3f,0x02,0x00,0x08,0x27,0x80,0xa3,0x02,0x00,0x02},
      {0x3f,0x02,0x00,0x08,0x27,0x80,0xa4,0x02,0x40,0x02},
      {0x3f,0x02,0x00,0x08,0x27,0x80,0xa5,0x02,0x40,0x02},
   };
   for (unsigned op=0;op<3;op++) {
      const struct agx3_subgroup_reduce_desc d = {
         .quad=true,.op=quad_ops[op],.dest=2,.src=2,.half=true,.wait_mask=128,
      };
      if (!agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C,&d,native,&len) ||
          len != 10 || memcmp(native,quad_int_expected[op],len)) return 60+op;
   }
   puts("M5_REGISTER_BALLOT_QUAD_INTEGER_NATIVE_BYTES_PASS");
   const unsigned char short_srl_expected[] = {
      0x3f,0x02,0x00,0x18,0x26,0x00,0xa1,0x02,0x30,0x82,0x00,0x34,0x00,0x00,
   };
   if (!agx3_alu_srl_imm16(AGX_ISA_ENCODING_AGX3C,2,2,8,
          AGX3_ALU_HINT_DISCARD,false,128,native,&len) ||
       len != 14 || memcmp(native,short_srl_expected,len)) return 63;
   const uint16_t andc_mixed_src[] = {4,0};
   const bool andc_half[] = {true,false};
   const enum agx3_alu_hint andc_mixed_hints[] = {
      AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,
   };
   const unsigned char andc_half_expected[] = {
      0x2b,0x00,0x06,0x00,0x32,0x70,0xa4,0x02,0x04,0x80,
   };
   if (!agx3_alu_bit_andc_width_m5(0,andc_mixed_src,andc_half,andc_mixed_hints,
          false,128,native,&len) || len != 10 ||
       memcmp(native,andc_half_expected,len)) return 64;
   const enum agx3_subgroup_reduce_op quad_more_ops[] = {
      AGX3_SUBGROUP_REDUCE_UMIN,AGX3_SUBGROUP_REDUCE_UMAX,
      AGX3_SUBGROUP_REDUCE_FADD,AGX3_SUBGROUP_REDUCE_FMIN,AGX3_SUBGROUP_REDUCE_FMAX,
   };
   const unsigned char quad_more_expected[][10] = {
      {0x3f,0x02,0x00,0x08,0x27,0x80,0xa6,0x02,0x00,0x02},
      {0x3f,0x02,0x00,0x08,0x27,0x80,0xa7,0x02,0x00,0x02},
      {0x3f,0x02,0x00,0x0a,0x27,0x00,0xa8,0x02,0x00,0x02},
      {0x3f,0x02,0x00,0x08,0x27,0x80,0xab,0x02,0x00,0x02},
      {0x3f,0x02,0x00,0x08,0x27,0x80,0xaf,0x02,0x00,0x02},
   };
   for (unsigned op=0;op<5;op++) {
      const struct agx3_subgroup_reduce_desc d = {
         .quad=true,.op=quad_more_ops[op],.dest=2,.src=2,.half=true,.wait_mask=128,
      };
      if (!agx3_subgroup_encode_reduce(AGX_ISA_ENCODING_AGX3C,&d,native,&len) ||
          len != 10 || memcmp(native,quad_more_expected[op],len)) return 65+op;
   }
   puts("M5_NARROW_SHIFT_ANDC_QUAD_FLOAT_NATIVE_BYTES_PASS");
   const unsigned char ushort_float_expected[] = {
      0x2f,0x00,0x00,0x02,0x22,0x80,0xaf,0x02,0x20,0x00,
   };
   if (!agx3_alu_narrow_int_to_float(AGX_ISA_ENCODING_AGX3C,0,false,0,false,
          false,AGX3_ALU_HINT_DISCARD,false,128,native,&len) ||
       len != 10 || memcmp(native,ushort_float_expected,len)) return 70;
   puts("M5_USHORT_FLOAT_NATIVE_BYTES_PASS");
   const unsigned char float_ballot_expected[][12] = {
      {0x2f,0x00,0x00,0x1a,0x25,0x00,0xaf,0x02,0x10,0x01,0x29,0x00},
      {0x2f,0x00,0x00,0x1a,0x25,0x00,0xaf,0x02,0x48,0x80,0x28,0x00},
   };
   for (unsigned half=0;half<2;half++) {
      const struct agx3_subgroup_ballot_desc d = {
         .type=AGX3_SUBGROUP_BALLOT_FLOAT,.src_half=half,
         .compare_register=true,.compare_value=half ? 1 : 4,.compare_half=half,
         .compare_discard=true,.condition=0,.invert=true,.wait_mask=128,
      };
      if (!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C,&d,native,&len) ||
          len != 12 || memcmp(native,float_ballot_expected[half],len)) return 71+half;
   }
   puts("M5_FLOAT_REGISTER_BALLOT_NATIVE_BYTES_PASS");
   const unsigned char selected_float_expected[][16] = {
      {0x32,0x02,0x03,0x0a,0x23,0x00,0xa0,0x02,0x94,0x02,0x21,0x08,0x00,0x06,0x80,0x00},
      {0x32,0x00,0x03,0x4a,0x23,0x00,0xa0,0x02,0x4c,0x82,0x20,0x08,0x00,0x06,0x80,0x00},
   };
   for (unsigned half=0;half<2;half++) {
      const struct agx3_fsel32_gpu1_desc d = {
         .dest=2,.cmp0=half ? 8 : 2,.cmp1=half ? 9 : 10,
         .cmp0_16=half,.cmp1_16=half,
         .cmp0_hint=AGX3_ALU_HINT_DISCARD,.cmp1_hint=AGX3_ALU_HINT_DISCARD,
         .sel={0,48},.sel_immediate={false,true},.wait_mask=128,
      };
      if (!agx3_alu_fcmp_select32_gpu1(&d,native,&len) || len != 16 ||
          memcmp(native,selected_float_expected[half],len)) {
         printf("FSEL_MISMATCH half=%u len=%u ",half,len);
         for (unsigned byte=0;byte<len;byte++) printf("%02x",native[byte]);
         puts("");
         return 73+half;
      }
   }
   puts("M5_FLOAT_SELECTED_CONSTANT_NATIVE_BYTES_PASS");
   const uint16_t half_bin_src[] = {2,3};
   const enum agx3_alu_hint half_bin_hints[] = {
      AGX3_ALU_HINT_DISCARD,AGX3_ALU_HINT_DISCARD,
   };
   const bool half_sub_neg[] = {false,true};
   const unsigned char half_bin_expected[][10] = {
      {0x38,0x02,0x04,0x02,0x22,0xa0,0xa0,0x02,0x05,0x08},
      {0x38,0x02,0x05,0x02,0x22,0x20,0xa4,0x02,0xc4,0x00},
      {0x38,0x02,0x24,0x02,0x22,0xa0,0xa0,0x02,0x05,0x08},
   };
   for (unsigned op=0;op<3;op++) {
      bool ok = op == 0 ? agx3_alu_hadd(AGX_ISA_ENCODING_AGX3C,2,
                      half_bin_src,half_bin_hints,false,128,native,&len)
                : op == 1 ? agx3_alu_hmul(AGX_ISA_ENCODING_AGX3C,2,
                      half_bin_src,half_bin_hints,false,128,native,&len)
                : agx3_alu_hadd_signed(AGX_ISA_ENCODING_AGX3C,2,
                      half_bin_src,half_bin_hints,half_sub_neg,false,128,native,&len);
      if (!ok || len != 10 || memcmp(native,half_bin_expected[op],len)) return 75+op;
   }
   puts("M5_HALF_ARITHMETIC_NATIVE_BYTES_PASS");
   const struct agx3_imad_desc all_half_mad = {
      .src_a=1,.src_a_16=true,.src_a_kill=true,
      .multiplier_is_register=true,.multiplier_reg=4,.multiplier_16=true,
      .multiplier_kill=true,.src_c_16=true,.src_c_kill=true,.wait_mask=128,
   };
   const unsigned char all_half_mad_expected[] = {
      0x2f,0x01,0x04,0x1a,0x21,0x00,0xa1,0x02,0x20,0x01,0xa0,0x00,0x00,0x00,
   };
   if (!agx3_imad_encode(AGX_ISA_ENCODING_AGX3C,&all_half_mad,native,&len) ||
       len != 14 || memcmp(native,all_half_mad_expected,len)) return 78;
   const struct agx3_fsel32_gpu1_desc select_two = {
      .cmp0_16=true,.cmp1=1,.cmp1_16=true,
      .cmp0_hint=AGX3_ALU_HINT_DISCARD,.cmp1_hint=AGX3_ALU_HINT_DISCARD,
      .sel={2,0},.sel_immediate={true,true},.sel_integer=true,.wait_mask=128,
   };
   const unsigned char select_two_expected[] = {
      0x22,0x00,0x07,0x0a,0x23,0x00,0xa0,0x02,0x4c,0x80,0x60,0x08,0x02,0x40,
   };
   if (!agx3_alu_fcmp_select32_gpu1(&select_two,native,&len) ||
       len != 14 || memcmp(native,select_two_expected,len)) return 79;
   puts("M5_HALF_MULTIPLY_ADD_INTEGER_SELECT_NATIVE_BYTES_PASS");
   const uint16_t shift_src[] = {4, 0};
   const bool shift_half[] = {true, false};
   const enum agx3_alu_hint shift_hint[] = {AGX3_ALU_HINT_DISCARD, AGX3_ALU_HINT_DISCARD};
   const unsigned char shift_expected[] = {
      0x27,0x04,0x00,0x3a,0x26,0x00,0xa0,0x02,0x00,0x80,0x20,0x34,0x00,0x00,
   };
   if (!agx3_alu_sll_reg(AGX_ISA_ENCODING_AGX3C,0,false,false,shift_src,
          shift_half,shift_hint,1,native,&len) || len != sizeof(shift_expected) ||
       memcmp(native,shift_expected,len)) return 22;
   const unsigned char or_expected[] = {
      0x2b,0x00,0x07,0x80,0x32,0x38,0xa4,0x02,0x04,0x01,
   };
   if (!agx3_alu_bit_mixed_m5(AGX3_ALU_BOR,0,0,4,AGX3_ALU_HINT_DISCARD,
          AGX3_ALU_HINT_DISCARD,false,128,native,&len) ||
       len != sizeof(or_expected) || memcmp(native,or_expected,len)) return 23;
   const unsigned char cmp_expected[] = {
      0x2a,0x02,0x3b,0x04,0x23,0x80,0xa5,0x02,0x88,0x00,
   };
   if (!agx3_cf_icmp_imm16_runtime(AGX_ISA_ENCODING_AGX3C,AGX3_CF_ICOND_ULT,
          0,2,2,(struct agx3_cf_source_hints){false,true},128,native,&len) ||
       len != sizeof(cmp_expected) || memcmp(native,cmp_expected,len)) return 24;
   const unsigned char fcmp_expected[] = {
      0x2a,0x02,0x2b,0x00,0x23,0x00,0xa3,0x12,0x00,0x81,
   };
   if (!agx3_cf_fcmp_zero_imm(AGX_ISA_ENCODING_AGX3C,AGX3_CF_FCOND_EQ,
          false,0,2,false,0,(struct agx3_cf_source_hints){false,true},128,
          native,&len) || len != sizeof(fcmp_expected) ||
       memcmp(native,fcmp_expected,len)) return 25;
   const struct agx3_fsel32_gpu1_desc fbool = {
      .dest=2, .cmp0=2, .cmp0_hint=AGX3_ALU_HINT_DISCARD,
      .cmp1_immediate=true, .sel_immediate={true,true}, .sel={1,0},
      .sel_integer=true, .wait_mask=128,
   };
   const unsigned char fbool_expected[] = {
      0x32,0x02,0x05,0x0a,0x23,0x00,0xa0,0x02,0x14,0x80,0x4e,0x18,0x01,0x00,0x80,0x00,
   };
   if (!agx3_alu_fcmp_select16_gpu1(&fbool,native,&len) ||
       len != sizeof(fbool_expected) || memcmp(native,fbool_expected,len)) return 26;
   const struct agx3_subgroup_ballot_desc quad_vote = {
      .type=AGX3_SUBGROUP_BALLOT_INTEGER, .quad=true,
      .dest=2, .dest_half=true, .dest_cache=true, .src=2,
      .src_live=false, .condition=4, .invert=true, .wait_mask=128,
   };
   const unsigned char quad_vote_expected[] = {
      0x3f,0x02,0x00,0x18,0x2d,0x00,0xa6,0x02,0x38,0x00,0x08,0x04,
   };
   if (!agx3_subgroup_encode_ballot(AGX_ISA_ENCODING_AGX3C,&quad_vote,native,&len) ||
       len != sizeof(quad_vote_expected) || memcmp(native,quad_vote_expected,len)) return 76;
   const uint16_t hcompare_src[] = {2, 3};
   const struct agx3_cf_source_hints hcompare_hints[] = {{false,true},{false,true}};
   const unsigned char hcompare_expected[] = {0x2a,0x02,0x2b,0x00,0x23,0x02,0xa0,0x0a,0xc0,0x81};
   if (!agx3_cf_fcmp_reg16(AGX_ISA_ENCODING_AGX3C,AGX3_CF_FCOND_EQ,
          false,0,hcompare_src,hcompare_hints,128,native,&len) ||
       len != sizeof(hcompare_expected) || memcmp(native,hcompare_expected,len)) return 75;
   const unsigned char hfma_expected[] = {0x38,0x02,0x06,0x02,0x23,0x20,0xa0,0x02,0x96,0x02,0x78,0x00};
   if (!agx3_alu_hfma_immediates_m5(2,2,10,20,AGX3_ALU_HINT_DISCARD,
          false,128,native,&len) || len != sizeof(hfma_expected) ||
       memcmp(native,hfma_expected,len)) return 78;
   for (unsigned sign = 0; sign < 2; sign++) {
      unsigned char expected_half[] = {0x3f,0x02,0x00,0x00,0x22,0x80,0xaf,0x02,0x28,0x00};
      if (sign) expected_half[8] = 0x68;
      if (!agx3_alu_i32_to_f16(AGX_ISA_ENCODING_AGX3C, 2, 2, sign,
                               AGX3_ALU_HINT_DISCARD, false, 128, native, &len) ||
          len != sizeof(expected_half) || memcmp(native, expected_half, len)) return 80;
   }
   const struct agx3_tps_desc byte_tps = {
      .data=10, .width=AGX3_MEM_W8, .mask=7, .byte_offset=4,
      .index_immediate=true, .frame_region=6, .value_kill=true,
      .wait_mask=1,
   };
   const unsigned char byte_tps_expected[] = {0x57,0x00,0x0b,0x00,0xc1,0x2c,0x92,0xa0,0x00,0x04};
   if (!agx3_tps_encode_store(AGX_ISA_ENCODING_AGX3C,&byte_tps,native,&len) ||
       len != sizeof(byte_tps_expected) || memcmp(native,byte_tps_expected,len)) return 79;
   puts("M5_SHIFT_COMPARE_OR_NATIVE_BYTES_PASS");
   /* A vertex ID converted to float may still index a later attribute load.
    * The old compact M5 packer always discarded that live integer source. */
   const unsigned char int_to_float_hints[3][10] = {
      {0x37,0x00,0x00,0x02,0x2a,0x80,0xaf,0x02,0x48,0x00},
      {0x37,0x00,0x00,0x02,0x2a,0x80,0xaf,0x02,0x68,0x00},
      {0x37,0x80,0x00,0x02,0x2a,0x80,0xaf,0x02,0x48,0x00},
   };
   for (unsigned hint = 0; hint < 3; ++hint) {
      if (!agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3C, 2, 0, true,
                                hint, true, 0, native, &len) ||
          len != 10 || memcmp(native, int_to_float_hints[hint], 10)) return 80;
   }
   if (!agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3C, 254, 252, false,
                             AGX3_ALU_HINT_CACHE, true, 128, native, &len) ||
       agx3_alu_int_to_float(AGX_ISA_ENCODING_AGX3C, 256, 0, false,
                            AGX3_ALU_HINT_NONE, false, 0, native, &len)) return 81;
   puts("M5_INT_TO_FLOAT_SOURCE_LIFETIME_PASS");
   const struct agx3_falu_runtime saturate_runtime = {
      .dest_cache = true, .discard = {true}, .wait_mask = 1,
   };
   const unsigned char saturate_expected[] = {
      0x09, 0x01, 0x3c, 0x81, 0x01, 0x22, 0x00, 0x02,
   };
   if (!agx3_falu_add_imm_m5(0, 0, 0x80, false, false, true,
                             &saturate_runtime, native, &len) ||
       len != sizeof(saturate_expected) ||
       memcmp(native, saturate_expected, len)) return 50;
   puts("M5_SATURATING_ADD_NATIVE_BYTES_PASS");
   glsl_type_singleton_init_or_ref();
   bool ok = true;
   /* A dynamic ray-query array exposed the old G17P limit on an M5 add
    * allocated to half232. Check both chip limits and transactional refusal. */
   const unsigned integer_regs[] = {190, 192, 232, 250, 252};
   for (unsigned chip = 0; chip < 2; ++chip) {
      const enum agx_isa_encoding enc = chip ? AGX_ISA_ENCODING_AGX3C :
                                             AGX_ISA_ENCODING_AGX3B;
      for (unsigned i = 0; i < ARRAY_SIZE(integer_regs); ++i) {
         struct agx3_uniform_integer_desc d = {
            .op = AGX3_UI_ADD, .dest = integer_regs[i],
            .src = {{.kind = AGX3_UI_UNIFORM, .value = 134},
                    {.kind = AGX3_UI_REGISTER, .value = 16}},
         };
         uint8_t bytes[16], size = 0;
         memset(bytes, 0xa5, sizeof(bytes));
         const bool expected = integer_regs[i] < (chip ? 252 : 192);
         const bool encoded = agx3_uniform_integer_encode(enc, &d, bytes, &size);
         ok &= encoded == expected;
         if (encoded) {
            ok &= size > 0 && size <= sizeof(bytes);
         } else {
            for (unsigned j = 0; j < sizeof(bytes); ++j)
               ok &= bytes[j] == 0xa5;
         }
      }
   }
   puts(ok ? "M5_UNIFORM_INTEGER_REGISTER_LIMIT_PASS" :
             "M5_UNIFORM_INTEGER_REGISTER_LIMIT_FAIL");
   /* Native G17G 4x sparse MRT: metal577 RT0+RT2 and RT1+RT2. */
   const uint32_t sparse_classes[] = {204, 240};
   const uint8_t sparse_wait[][6] = {
      {0x67, 0x46, 0x00, 0x05, 0x00, 0x00},
      {0x87, 0x47, 0x00, 0x05, 0x00, 0x00},
   };
   const uint8_t sparse_release[][6] = {
      {0x67, 0x16, 0x00, 0x04, 0x00, 0x00},
      {0x87, 0x17, 0x00, 0x04, 0x00, 0x00},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(sparse_classes); i++) {
      uint8_t bytes[AGX3_PIXEL_MAX_LENGTH], length = 0;
      ok &= agx3_pixel_wait(AGX_ISA_ENCODING_AGX3C, sparse_classes[i], 0,
                           bytes, &length) && length == 6 &&
            memcmp(bytes, sparse_wait[i], 6) == 0;
      ok &= agx3_pixel_release(AGX_ISA_ENCODING_AGX3C, sparse_classes[i],
                              true, 0, bytes, &length) && length == 6 &&
            memcmp(bytes, sparse_release[i], 6) == 0;
   }

   /* The custom MRT generator's one-target limit must reproduce the
    * independently compiled, hardware-qualified RT0 path, including layers
    * and MS arrays. All target counts retain bounded register allocation. */
   for (unsigned layered = 0; layered < 2; ++layered) {
      for (unsigned ms = 0; ms < 2; ++ms) {
         const uint8_t selectors[8] = {
            0x75, 0x15, 0x35, 0x55, 0x95, 0x75, 0x15, 0x35,
         };
         for (unsigned targets = 1; targets <= 8; ++targets) {
            uint8_t generated[AGX3_M5_MRT_EOT_MAX_LENGTH];
            size_t length = 0;
            if (!agx3_m5_mrt_eot_generate_layered(targets, selectors, ms,
                   layered, generated, sizeof(generated), &length) ||
                agx3_m5_mrt_eot_nr_gprs(targets) != 8 ||
                agx3_m5_mrt_eot_staging_halfs(targets) != targets * 8)
               return 133;
            /* G18G re-encodes every M5 MRT EOT shape directly to agx3d. */
            {
               struct util_dynarray code;
               util_dynarray_init(&code, NULL);
               memcpy(util_dynarray_grow_bytes(&code, 1, length), generated,
                      length);
               const char *why = NULL;
               unsigned at = 0;
               const bool ok = agx3_transcode_agx3d(&code, 0, &why, &at) &&
                               code.size > 0 &&
                               code.size <= AGX3_M5_MRT_EOT_MAX_LENGTH * 2;
               util_dynarray_fini(&code);
               if (!ok) {
                  fprintf(stderr,
                          "G18G MRT EOT transcode failed (targets=%u ms=%u layered=%u): %s at +0x%x\n",
                          targets, ms, layered, why ? why : "size", at);
                  return 134;
               }
            }
            if (targets != 1)
               continue;
            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_COMPUTE, &agx_nir_options, "M5 array MRT limit");
            nir_def *xy = nir_trim_vector(
               &b, nir_load_global_invocation_id(&b, 32), 2);
            nir_def *coords = layered
               ? nir_vec3(&b, nir_channel(&b, xy, 0),
                          nir_channel(&b, xy, 1),
                          nir_u2u32(&b, nir_load_layer_id(&b)))
               : xy;
            nir_image_store_block_agx(&b, nir_imm_intN_t(&b, 0, 16),
               nir_imm_int(&b, 0), coords, .format=PIPE_FORMAT_R8_UNORM,
               .image_dim=GLSL_SAMPLER_DIM_2D,
               .image_array=layered, .explicit_coord=true);
            agx_preprocess_nir(b.shader);
            struct agx_shader_key key = {
               .dev.chip=AGX_CHIP_G17S, .dev.encoding=AGX_ISA_ENCODING_AGX3C,
               .no_preamble=true, .m5_rt0_eot=true,
               .g17p_rt0_eot_multisample=ms,
            };
            agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R8_UNORM, &key);
            struct agx_shader_part part = {0};
            bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
            bool match = compiled && part.info.binary_size == length &&
               memcmp(part.binary, generated, length) == 0;
            printf("M5_MRT_RT0_ARRAY_%u_MS_%u_%s size=%zu\n",
                   layered, ms, match ? "PASS" : "FAIL", length);
            ok &= match;
            free(part.binary);
            ralloc_free(b.shader);
         }
      }
   }

   for (unsigned ms = 0; ms < 2; ++ms) {
      const enum pipe_format formats[] = {
         PIPE_FORMAT_R10G10B10A2_UNORM, PIPE_FORMAT_B10G10R10A2_UNORM,
      };
      for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "M5 layered RGB10 EOT");
         nir_def *xy = nir_trim_vector(
            &b, nir_load_global_invocation_id(&b, 32), 2);
         nir_def *coords = nir_vec3(&b, nir_channel(&b, xy, 0),
            nir_channel(&b, xy, 1), nir_u2u32(&b, nir_load_layer_id(&b)));
         nir_image_store_block_agx(&b, nir_imm_intN_t(&b, 0, 16),
            nir_imm_int(&b, 0), coords, .format=PIPE_FORMAT_R10G10B10A2_UNORM,
            .image_dim=GLSL_SAMPLER_DIM_2D, .image_array=true,
            .explicit_coord=true);
         agx_preprocess_nir(b.shader);
         struct agx_shader_key key = {
            .dev.chip=AGX_CHIP_G17S, .dev.encoding=AGX_ISA_ENCODING_AGX3C,
            .no_preamble=true, .m5_rt0_eot=true,
            .g17p_rt0_eot_multisample=ms,
         };
         agx_g17p_configure_rt0_eot_key(formats[f], &key);
         struct agx_shader_part part = {0};
         uint8_t generated[AGX3_M5_MRT_EOT_MAX_LENGTH];
         size_t length = 0;
         const uint8_t selector = 0x75;
         bool generated_ok = agx3_m5_mrt_eot_generate_layered(
            1, &selector, ms, true, generated, sizeof(generated), &length);
         bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
         bool match = generated_ok && compiled &&
            part.info.binary_size == length &&
            memcmp(part.binary, generated, length) == 0;
         printf("M5_LAYERED_RGB10_%u_MS_%u_%s\n", f, ms,
                match ? "PASS" : "FAIL");
         ok &= match;
         free(part.binary);
         ralloc_free(b.shader);
      }
   }

   /* A narrow zero-extended low word vanishes from this funnel shift.
    * These local-ID address calculations formerly failed after narrowing. */
   for (unsigned shift = 16; shift <= 31; shift++) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "M5 narrow funnel shift");
      nir_def *index = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *wide = nir_ishl_imm(&b, nir_i2i64(&b, index), 32 - shift);
      nir_store_global(&b, nir_unpack_64_2x32_split_y(&b, wide),
                       nir_imm_int64(&b, 0x100000000ull), .align_mul = 4);
      agx_preprocess_nir_m5(b.shader);
      struct agx_shader_key key = {
         .dev.chip = AGX_CHIP_G17S, .dev.encoding = AGX_ISA_ENCODING_AGX3C,
         .no_preamble = true,
      };
      struct agx_shader_part part = {0};
      const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
      printf("M5_NARROW_FUNNEL_SHIFT_%u_%s\n", shift,
             compiled ? "PASS" : "FAIL");
      ok &= compiled && part.info.binary_size > 0;
      free(part.binary);
      ralloc_free(b.shader);
   }
   /* The same shared-memory operations need a rendezvous only when the
    * execution scope is the entire workgroup. In particular, a subgroup
    * fence may occur under a subgroup-uniform branch. */
   for (unsigned workgroup = 0; workgroup < 2; ++workgroup) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "M5 shared barrier scope");
      b.shader->info.workgroup_size[0] = 256;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
      b.shader->info.shared_size = 1024;
      nir_def *id = nir_load_local_invocation_index(&b);
      nir_def *offset = nir_ishl_imm(&b, id, 2);
      nir_store_shared(&b, id, offset, .align_mul = 4);
      nir_barrier(&b,
         .execution_scope = workgroup ? SCOPE_WORKGROUP : SCOPE_NONE,
         .memory_scope = workgroup ? SCOPE_WORKGROUP : SCOPE_SUBGROUP,
         .memory_semantics = NIR_MEMORY_ACQUIRE | NIR_MEMORY_RELEASE,
         .memory_modes = nir_var_mem_shared);
      nir_def *other = nir_ishl_imm(&b, nir_ixor(&b, id, nir_imm_int(&b, 1)), 2);
      nir_def *value = nir_load_shared(&b, 1, 32, other, .align_mul = 4);
      nir_store_global(&b, value,
         nir_iadd(&b, nir_imm_int64(&b, 0x100000000ull), nir_u2u64(&b, offset)),
         .align_mul = 4);
      agx_preprocess_nir_m5(b.shader);
      struct agx_shader_key key = {
         .dev.chip = AGX_CHIP_G17S, .dev.encoding = AGX_ISA_ENCODING_AGX3C,
         .no_preamble = true,
      };
      struct agx_shader_part part = {0};
      bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
      unsigned barriers = 0;
      const uint8_t rendezvous[] = {0x27, 0x51, 0, 6, 0, 0};
      for (unsigned i = part.info.main_offset;
           compiled && i + sizeof(rendezvous) <=
              part.info.main_offset + part.info.main_size; i += 2)
         barriers += !memcmp(part.binary + i, rendezvous, sizeof(rendezvous));
      bool correct = compiled && barriers == workgroup;
      printf("M5_SHARED_BARRIER_SCOPE_%u_%s\n", workgroup,
             correct ? "PASS" : "FAIL");
      ok &= correct;
      free(part.binary);
      ralloc_free(b.shader);
   }
   const enum pipe_format eot_formats[] = {PIPE_FORMAT_R8_UNORM,
                                           PIPE_FORMAT_R8_UINT,
                                           PIPE_FORMAT_R8_SINT};
   for (unsigned i = 0; i < 3; ++i) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                    &agx_nir_options, "M5 EOT");
      nir_image_store_block_agx(&b, nir_imm_intN_t(&b,0,16), nir_imm_int(&b,0),
         nir_trim_vector(&b, nir_load_global_invocation_id(&b,32), 2),
         /* IMGWBLK names the raw tile class. Signedness belongs to the
          * PBE descriptor, so both integer8 logical formats use I8. */
         .format=i ? PIPE_FORMAT_R8_UINT : PIPE_FORMAT_R8_UNORM,
         .image_dim=GLSL_SAMPLER_DIM_2D,
         .image_array=false, .explicit_coord=true);
      agx_preprocess_nir(b.shader);
      struct agx_shader_key key = {
         .dev.chip=AGX_CHIP_G17S, .dev.encoding=AGX_ISA_ENCODING_AGX3C,
         .no_preamble=true, .m5_rt0_eot=true,
      };
      agx_g17p_configure_rt0_eot_key(eot_formats[i], &key);
      unsigned char expected_format[sizeof(expected)];
      memcpy(expected_format, expected, sizeof(expected));
      if (i)
         expected_format[71] = 0xa0;
      struct agx_shader_part part={0};
      bool compiled=agx_compile_shader_nir(b.shader,&key,&part);
      bool format_ok=compiled && part.info.binary_size==sizeof(expected) &&
         part.info.main_offset==0 && part.info.main_size==sizeof(expected) &&
         part.info.nr_gprs==8 && part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs==8 &&
         !memcmp(part.binary,expected_format,sizeof(expected));
      printf("M5_EOT_FORMAT_%u_%s compiled=%u size=%u gprs=%u staging=%u\n",
         i,format_ok ? "PASS":"FAIL",compiled,part.info.binary_size,
         part.info.nr_gprs,part.info.agx3_staging_halfs);
      ok &= format_ok;
      free(part.binary);
      ralloc_free(b.shader);
   }
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "M5 robust UBO end offset");
      nir_def *offset = nir_load_global(&b, 1, 32,
         nir_imm_int64(&b, 0x100000000ull), .align_mul = 4);
      nir_def *end = nir_uadd_sat(&b, offset, nir_imm_int(&b, 4));
      nir_store_global(&b, end, nir_imm_int64(&b, 0x100000010ull),
                       .align_mul = 4);
      agx_preprocess_nir_m5(b.shader);
      struct agx_shader_key key = {
         .dev.chip = AGX_CHIP_G17S, .dev.encoding = AGX_ISA_ENCODING_AGX3C,
         .no_preamble = true,
      };
      struct agx_shader_part part = {0};
      const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
      printf("M5_ROBUST_UBO_UADD_SAT_IMMEDIATE_%s compiled=%u size=%u\n",
             compiled ? "PASS" : "FAIL", compiled, part.info.binary_size);
      ok &= compiled && part.info.binary_size > 0;
      free(part.binary);
      ralloc_free(b.shader);
   }
   /* Qt's curve coverage shader clamps a fused float product plus a
    * minifloat constant. Keep the all-f32 form on the native FMA path. */
   for (unsigned i = 0; i < 3; ++i) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "M5 saturated FMA immediate");
      nir_def *a = nir_load_global(&b, 1, 32,
         nir_imm_int64(&b, 0x100000000ull), .align_mul = 4);
      nir_def *c = nir_load_global(&b, 1, 32,
         nir_imm_int64(&b, 0x100000004ull), .align_mul = 4);
      nir_def *value = nir_fsat(&b, nir_ffma(&b, a, c,
                                           nir_imm_float(&b, i * 0.5f)));
      nir_store_global(&b, value, nir_imm_int64(&b, 0x100000010ull),
                       .align_mul = 4);
      agx_preprocess_nir_m5(b.shader);
      struct agx_shader_key key = {
         .dev.chip = AGX_CHIP_G17S, .dev.encoding = AGX_ISA_ENCODING_AGX3C,
         .no_preamble = true,
      };
      struct agx_shader_part part = {0};
      const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
      printf("M5_SATURATED_FMA_IMMEDIATE_%u_%s\n", i,
             compiled ? "PASS" : "FAIL");
      ok &= compiled && part.info.binary_size > 0;
      free(part.binary);
      ralloc_free(b.shader);
   }
   glsl_type_singleton_decref();
   return !ok;
}

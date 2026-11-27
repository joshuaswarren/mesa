/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Verify AGX2 (Gen7/Gen8) texel buffer lowering in agx_nir_lower_texture() and
 * libagx_texture_load_rgb32():
 *   - RGB32 loads take the raw view coordinate and add buffer_offset_sw (the
 *     sub-16-byte tail in 4-byte components, 0..3) inside
 *     libagx_texture_load_rgb32(), so 4/8/12-byte-aligned views and Gallium's
 *     16-byte-aligned (buffer_offset_sw = 0) views both compute the exact
 *     element address.
 *   - Out-of-bounds RGB32 coordinates (coord >= elements, negative/UINT32_MAX,
 *     elements == 0) bypass the raw global load and route to the 2D hardware
 *     fetch with OOB coordinate (-1).
 *   - Non-RGB32 texel buffers still add buffer_offset_sw (tail_offset_el) to
 *     the view coordinate before 2D remapping by AGX_TEXTURE_BUFFER_WIDTH.
 */
#include <array>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "agx_pack.h"
#include "asahi/lib/agx_abi.h"
}

static nir_tex_instr *
build_buffer_fetch(nir_builder *b, nir_def *handle, nir_def *coord,
                   nir_alu_type dest_type)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 3);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_BUF;
   tex->dest_type = dest_type;
   tex->coord_components = 1;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_int(b, 0));
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return tex;
}

static void
lower_agx2_buffer_shader(nir_shader *shader)
{
   nir_validate_shader(shader, "before AGX2 texture lowering");
   EXPECT_TRUE(agx_nir_lower_texture_for_generation(shader, 7));
   bool progress;
   do {
      progress = nir_opt_constant_folding(shader);
      progress |= nir_opt_algebraic(shader);
      progress |= nir_opt_dead_cf(shader);
      progress |= nir_opt_dce(shader);
   } while (progress);
   nir_validate_shader(shader, "after AGX2 texture lowering");
}

struct Agx2BufferEvaluator {
   static constexpr uint64_t DESC_ADDR = UINT64_C(0x20000);
   static constexpr uint64_t DATA_BASE = UINT64_C(0x40000000);

   using Value = std::array<uint64_t, NIR_MAX_VEC_COMPONENTS>;
   std::unordered_map<nir_def *, Value> values;
   Value output = {};
   nir_block *previous = nullptr;

   struct agx_texture_packed desc = {};
   uint32_t coord_input = 0;

   unsigned raw_reads = 0;
   uint64_t last_raw_addr = 0;
   unsigned tex_reads = 0;
   uint32_t last_tex_x = 0;
   uint32_t last_tex_y = 0;

   static uint32_t data_word(uint64_t word_index)
   {
      return uint32_t(word_index * UINT64_C(0x9e3779b1) + UINT64_C(0x13579bdf));
   }

   uint64_t scalar(nir_def *def, unsigned c = 0)
   {
      auto it = values.find(def);
      EXPECT_NE(it, values.end());
      return it == values.end() ? 0 : it->second[c];
   }

   uint64_t read_desc(uint64_t addr, unsigned bit_size)
   {
      EXPECT_GE(addr, DESC_ADDR);
      uint64_t off = addr - DESC_ADDR;
      unsigned bytes = bit_size / 8;
      EXPECT_LE(off + bytes, sizeof(desc));
      uint64_t v = 0;
      if (off + bytes <= sizeof(desc))
         memcpy(&v, reinterpret_cast<const uint8_t *>(&desc) + off, bytes);
      return v;
   }

   void eval_instr(nir_instr *instr)
   {
      Value result = {};
      nir_def *def = nullptr;

      if (instr->type == nir_instr_type_load_const) {
         nir_load_const_instr *c = nir_instr_as_load_const(instr);
         def = &c->def;
         for (unsigned i = 0; i < def->num_components; ++i)
            result[i] = nir_const_value_as_uint(c->value[i], def->bit_size);
      } else if (instr->type == nir_instr_type_alu) {
         nir_alu_instr *alu = nir_instr_as_alu(instr);
         def = &alu->def;
         for (unsigned c = 0; c < def->num_components; ++c) {
            auto src = [&](unsigned i) {
               return scalar(alu->src[i].src.ssa, alu->src[i].swizzle[c]);
            };
            switch (alu->op) {
            case nir_op_mov: result[c] = src(0); break;
            case nir_op_vec2:
            case nir_op_vec3:
            case nir_op_vec4:
               result[c] = scalar(alu->src[c].src.ssa, alu->src[c].swizzle[0]);
               break;
            case nir_op_iadd: result[c] = src(0) + src(1); break;
            case nir_op_isub: result[c] = src(0) - src(1); break;
            case nir_op_imul: result[c] = src(0) * src(1); break;
            case nir_op_udiv: result[c] = src(0) / src(1); break;
            case nir_op_umod: result[c] = src(0) % src(1); break;
            case nir_op_ishl: result[c] = src(0) << (src(1) & 63); break;
            case nir_op_ushr: result[c] = src(0) >> (src(1) & 63); break;
            case nir_op_iand: result[c] = src(0) & src(1); break;
            case nir_op_ior: result[c] = src(0) | src(1); break;
            case nir_op_ixor: result[c] = src(0) ^ src(1); break;
            case nir_op_inot: result[c] = ~src(0); break;
            case nir_op_ieq: result[c] = src(0) == src(1); break;
            case nir_op_ine: result[c] = src(0) != src(1); break;
            case nir_op_uge: result[c] = src(0) >= src(1); break;
            case nir_op_ult: result[c] = src(0) < src(1); break;
            case nir_op_bcsel: result[c] = src(0) ? src(1) : src(2); break;
            case nir_op_u2u8:
            case nir_op_u2u16:
            case nir_op_u2u32:
            case nir_op_u2u64:
               result[c] = src(0);
               break;
            case nir_op_ubitfield_extract: {
               unsigned off = unsigned(src(1)) & 63;
               unsigned bits = unsigned(src(2)) & 63;
               uint64_t mask = bits >= 64 ? ~UINT64_C(0) : ((UINT64_C(1) << bits) - 1);
               result[c] = (src(0) >> off) & mask;
               break;
            }
            case nir_op_pack_64_2x32_split:
               result[c] = (src(0) & 0xffffffffu) | ((src(1) & 0xffffffffu) << 32);
               break;
            case nir_op_pack_32_2x16_split:
               result[c] = (src(0) & 0xffffu) | ((src(1) & 0xffffu) << 16);
               break;
            case nir_op_unpack_64_2x32_split_x:
               result[c] = src(0) & 0xffffffffu;
               break;
            case nir_op_unpack_64_2x32_split_y:
               result[c] = (src(0) >> 32) & 0xffffffffu;
               break;
            default:
               ADD_FAILURE() << "Unhandled ALU op " << nir_op_infos[alu->op].name;
            }
         }
      } else if (instr->type == nir_instr_type_phi) {
         nir_phi_instr *phi = nir_instr_as_phi(instr);
         def = &phi->def;
         bool found = false;
         nir_foreach_phi_src(s, phi) {
            if (s->pred == previous) {
               result = values.at(s->src.ssa);
               found = true;
            }
         }
         EXPECT_TRUE(found);
      } else if (instr->type == nir_instr_type_tex) {
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         EXPECT_EQ(tex->op, nir_texop_txf);
         EXPECT_EQ(tex->sampler_dim, GLSL_SAMPLER_DIM_2D);
         int xy = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
         ASSERT_GE(xy, 0);
         nir_def *coords = tex->src[xy].src.ssa;
         last_tex_x = uint32_t(scalar(coords, 0));
         last_tex_y = uint32_t(scalar(coords, 1));
         ++tex_reads;
         def = &tex->def;
         result = {last_tex_x, last_tex_y, 0xdeadbeefu, 0xcafebabeu};
      } else if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         def = nir_intrinsic_infos[intr->intrinsic].has_dest ? &intr->def : nullptr;
         switch (intr->intrinsic) {
         case nir_intrinsic_load_global_invocation_id:
            result[0] = coord_input;
            break;
         case nir_intrinsic_load_from_texture_handle_agx:
            result[0] = DESC_ADDR;
            break;
         case nir_intrinsic_load_global_constant:
         case nir_intrinsic_load_global: {
            uint64_t addr = scalar(intr->src[0].ssa);
            if (addr >= DESC_ADDR && addr < DESC_ADDR + sizeof(desc)) {
               unsigned step = intr->def.bit_size / 8;
               for (unsigned c = 0; c < intr->def.num_components; ++c)
                  result[c] = read_desc(addr + c * step, intr->def.bit_size);
            } else {
               EXPECT_EQ(intr->def.bit_size, 32u);
               EXPECT_EQ(intr->def.num_components, 3u);
               EXPECT_EQ(nir_intrinsic_align_mul(intr), 4u);
               EXPECT_GE(addr, DATA_BASE);
               EXPECT_EQ((addr - DATA_BASE) % 4, 0u);
               last_raw_addr = addr;
               uint64_t word = (addr - DATA_BASE) / 4;
               for (unsigned c = 0; c < 3; ++c)
                  result[c] = data_word(word + c);
               ++raw_reads;
            }
            break;
         }
         case nir_intrinsic_store_global:
            output = values.at(intr->src[0].ssa);
            break;
         default:
            ADD_FAILURE() << "Unhandled intrinsic "
                          << nir_intrinsic_infos[intr->intrinsic].name;
         }
      } else {
         ADD_FAILURE() << "Unhandled instruction type " << instr->type;
      }

      if (def) {
         if (def->bit_size < 64) {
            uint64_t mask = (UINT64_C(1) << def->bit_size) - 1;
            for (unsigned c = 0; c < def->num_components; ++c)
               result[c] &= mask;
         }
         values[def] = result;
      }
   }

   void eval_list(struct exec_list *nodes)
   {
      foreach_list_typed(nir_cf_node, node, node, nodes) {
         if (node->type == nir_cf_node_block) {
            nir_block *block = nir_cf_node_as_block(node);
            nir_foreach_instr(instr, block)
               eval_instr(instr);
            previous = block;
         } else if (node->type == nir_cf_node_if) {
            nir_if *nif = nir_cf_node_as_if(node);
            eval_list(scalar(nif->condition.ssa) ? &nif->then_list
                                                 : &nif->else_list);
         } else {
            ADD_FAILURE() << "Unexpected control flow";
         }
      }
   }
};

static void
pack_agx2_buffer_desc(struct agx_texture_packed *out,
                      enum agx_channels channels, uint64_t address,
                      uint32_t elements, uint32_t buffer_offset_sw,
                      uint32_t blocksize_B)
{
   struct AGX_TEXTURE tex = {AGX_TEXTURE_header};
   tex.dimension = AGX_TEXTURE_DIMENSION_2D;
   tex.layout = AGX_LAYOUT_LINEAR;
   tex.channels = channels;
   tex.type = AGX_TEXTURE_TYPE_UINT;
   tex.swizzle_r = AGX_CHANNEL_R;
   tex.swizzle_g = AGX_CHANNEL_G;
   tex.swizzle_b = AGX_CHANNEL_B;
   tex.swizzle_a = AGX_CHANNEL_1;
   tex.width = AGX_TEXTURE_BUFFER_WIDTH;
   uint32_t tail_el =
      channels == AGX_CHANNELS_R32G32B32_EMULATED ? 0 : buffer_offset_sw;
   tex.height = MAX2(DIV_ROUND_UP(elements + tail_el, AGX_TEXTURE_BUFFER_WIDTH), 1);
   tex.address = address;
   tex.buffer_size_sw = elements;
   tex.buffer_offset_sw = buffer_offset_sw;
   tex.depth = 1;
   tex.stride = AGX_TEXTURE_BUFFER_WIDTH * blocksize_B - 16;
   AGX_TEXTURE_pack(out->opaque, &tex);
}

TEST(Agx2TexelBuffer, Rgb32WordTailOffsetAndBounds)
{
   for (nir_alu_type dest_type : {nir_type_uint32, nir_type_float32}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "agx2_rgb32_texel_buffer");
      nir_def *coord =
         nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_tex_instr *tex =
         build_buffer_fetch(&b, nir_imm_int(&b, 0), coord, dest_type);
      nir_store_global(&b, &tex->def, nir_imm_int64(&b, 0x80000));
      lower_agx2_buffer_shader(b.shader);

      const uint32_t expected_alpha =
         dest_type == nir_type_float32 ? 0x3f800000u : 1u;

      /* Test all four 4-byte-aligned sub-16-byte offsets (0, 4, 8, 12). */
      for (uint32_t tail_B : {0u, 4u, 8u, 12u}) {
         const uint32_t elements = 8;
         for (uint32_t x : {0u, 1u, 7u, 8u, 100u, UINT32_MAX}) {
            SCOPED_TRACE(::testing::Message()
                         << "type=" << dest_type << " tail_B=" << tail_B
                         << " x=" << x);
            Agx2BufferEvaluator eval;
            pack_agx2_buffer_desc(&eval.desc,
                                  AGX_CHANNELS_R32G32B32_EMULATED,
                                  Agx2BufferEvaluator::DATA_BASE, elements,
                                  tail_B / 4, 12);
            eval.coord_input = x;
            eval.eval_list(&nir_shader_get_entrypoint(b.shader)->body);

            if (x < elements) {
               const uint64_t expected_addr =
                  Agx2BufferEvaluator::DATA_BASE + tail_B + uint64_t(x) * 12;
               const uint64_t word = (tail_B / 4) + uint64_t(x) * 3;
               EXPECT_EQ(eval.raw_reads, 1u);
               EXPECT_EQ(eval.tex_reads, 0u);
               EXPECT_EQ(eval.last_raw_addr, expected_addr);
               EXPECT_EQ(eval.output[0], Agx2BufferEvaluator::data_word(word));
               EXPECT_EQ(eval.output[1], Agx2BufferEvaluator::data_word(word + 1));
               EXPECT_EQ(eval.output[2], Agx2BufferEvaluator::data_word(word + 2));
               EXPECT_EQ(eval.output[3], expected_alpha);
            } else {
               EXPECT_EQ(eval.raw_reads, 0u);
               EXPECT_EQ(eval.tex_reads, 1u);
               EXPECT_EQ(eval.last_tex_x, UINT32_MAX % AGX_TEXTURE_BUFFER_WIDTH);
               EXPECT_EQ(eval.last_tex_y, UINT32_MAX / AGX_TEXTURE_BUFFER_WIDTH);
            }
         }

         /* Zero-element RGB32 view must treat x = 0 as OOB and never load. */
         Agx2BufferEvaluator eval;
         pack_agx2_buffer_desc(&eval.desc, AGX_CHANNELS_R32G32B32_EMULATED,
                               Agx2BufferEvaluator::DATA_BASE, 0, tail_B / 4, 12);
         eval.coord_input = 0;
         eval.eval_list(&nir_shader_get_entrypoint(b.shader)->body);
         EXPECT_EQ(eval.raw_reads, 0u);
         EXPECT_EQ(eval.tex_reads, 1u);
         EXPECT_EQ(eval.last_tex_x, UINT32_MAX % AGX_TEXTURE_BUFFER_WIDTH);
         EXPECT_EQ(eval.last_tex_y, UINT32_MAX / AGX_TEXTURE_BUFFER_WIDTH);
      }

      ralloc_free(b.shader);
   }
}

TEST(Agx2TexelBuffer, NonRgb32StillAppliesTailOffsetAndRowWrap)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "agx2_r32_texel_buffer");
   nir_def *coord = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_tex_instr *tex =
      build_buffer_fetch(&b, nir_imm_int(&b, 0), coord, nir_type_uint32);
   nir_store_global(&b, &tex->def, nir_imm_int64(&b, 0x80000));
   lower_agx2_buffer_shader(b.shader);

   const uint32_t elements = AGX_TEXTURE_BUFFER_WIDTH;
   const uint32_t tail_el = 1;

   /* Valid last texel x = 16383 with tail_el = 1 wraps to (0, 1). */
   for (uint32_t x : {0u, 16382u, 16383u, 16384u, UINT32_MAX}) {
      Agx2BufferEvaluator eval;
      pack_agx2_buffer_desc(&eval.desc, AGX_CHANNELS_R32,
                            Agx2BufferEvaluator::DATA_BASE, elements, tail_el,
                            4);
      eval.coord_input = x;
      eval.eval_list(&nir_shader_get_entrypoint(b.shader)->body);

      EXPECT_EQ(eval.raw_reads, 0u);
      EXPECT_EQ(eval.tex_reads, 1u);
      if (x < elements) {
         uint32_t eff = x + tail_el;
         EXPECT_EQ(eval.last_tex_x, eff % AGX_TEXTURE_BUFFER_WIDTH);
         EXPECT_EQ(eval.last_tex_y, eff / AGX_TEXTURE_BUFFER_WIDTH);
      } else {
         EXPECT_EQ(eval.last_tex_x, UINT32_MAX % AGX_TEXTURE_BUFFER_WIDTH);
         EXPECT_EQ(eval.last_tex_y, UINT32_MAX / AGX_TEXTURE_BUFFER_WIDTH);
      }
   }

   /* Null descriptor (AGX_ZERO_PAGE_ADDRESS) reports size 0 -> OOB. */
   {
      Agx2BufferEvaluator eval;
      struct AGX_TEXTURE null_tex = {AGX_TEXTURE_header};
      null_tex.layout = AGX_LAYOUT_TWIDDLED;
      null_tex.channels = AGX_CHANNELS_R8;
      null_tex.type = AGX_TEXTURE_TYPE_UNORM;
      null_tex.address = AGX_ZERO_PAGE_ADDRESS;
      AGX_TEXTURE_pack(eval.desc.opaque, &null_tex);
      eval.coord_input = 0;
      eval.eval_list(&nir_shader_get_entrypoint(b.shader)->body);
      EXPECT_EQ(eval.raw_reads, 0u);
      EXPECT_EQ(eval.tex_reads, 1u);
      EXPECT_EQ(eval.last_tex_x, UINT32_MAX % AGX_TEXTURE_BUFFER_WIDTH);
      EXPECT_EQ(eval.last_tex_y, UINT32_MAX / AGX_TEXTURE_BUFFER_WIDTH);
   }

   ralloc_free(b.shader);
}

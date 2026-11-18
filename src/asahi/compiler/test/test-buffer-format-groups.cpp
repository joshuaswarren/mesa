#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "asahi/lib/agx_abi.h"
}

static nir_tex_instr *
buffer_fetch(nir_builder *b, nir_def *handle, nir_def *coord,
             nir_def *lod, nir_alu_type type = nir_type_uint32,
             bool sparse = false)
{
   nir_tex_instr *tex = nir_tex_instr_create(b->shader, 3);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_BUF;
   tex->dest_type = type;
   tex->coord_components = 1;
   tex->is_sparse = sparse;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
   nir_def_init(&tex->instr, &tex->def, sparse ? 5 : 4, 32);
   nir_builder_instr_insert(b, &tex->instr);
   return tex;
}

static bool
is_metadata(nir_def *def, unsigned offset)
{
   nir_intrinsic_instr *load = nir_src_as_intrinsic(nir_src_for_ssa(def));
   if (!load || load->intrinsic != nir_intrinsic_load_global_constant)
      return false;
   nir_scalar addr = nir_scalar_chase_movs(nir_get_scalar(load->src[0].ssa, 0));
   if (!nir_scalar_is_alu(addr) || nir_scalar_alu_op(addr) != nir_op_iadd)
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      nir_scalar value = nir_scalar_chase_alu_src(addr, i);
      if (nir_scalar_is_const(value) && nir_scalar_as_uint(value) == offset)
         return true;
   }
   return false;
}

static bool
is_format_choice(nir_if *nif)
{
   nir_alu_instr *cond = nir_src_as_alu(nif->condition);
   if (!cond || cond->op != nir_op_ine)
      return false;
   return is_metadata(cond->src[0].src.ssa, AGX_G17P_BUFFER_TEXTURE_RGB32_OFFSET) ||
          is_metadata(cond->src[1].src.ssa, AGX_G17P_BUFFER_TEXTURE_RGB32_OFFSET);
}

static std::vector<nir_if *>
format_choices(nir_shader *shader)
{
   std::vector<nir_if *> result;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_if *nif = nir_block_get_following_if(block);
         if (nif && is_format_choice(nif))
            result.push_back(nif);
      }
   }
   return result;
}

static unsigned
count_path(struct exec_list *list, bool raw)
{
   unsigned count = 0;
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);
         count += count_path(&nif->then_list, raw);
         count += count_path(&nif->else_list, raw);
      } else if (node->type == nir_cf_node_block) {
         nir_foreach_instr(instr, nir_cf_node_as_block(node)) {
            if (raw && instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               count += intr->intrinsic == nir_intrinsic_load_global_constant &&
                        intr->def.num_components == 3;
            } else if (!raw && instr->type == nir_instr_type_tex) {
               count += nir_instr_as_tex(instr)->op == nir_texop_txf;
            }
         }
      }
   }
   return count;
}

static void
lower_buffer_shader(nir_shader *shader, bool sparse_feature = false)
{
   nir_validate_shader(shader, "before buffer format groups");
   EXPECT_TRUE(agx_nir_lower_texture_with_buffer_metadata(
      shader, 9, false, 0, sparse_feature));
   bool progress;
   do {
      progress = nir_opt_constant_folding(shader);
      progress |= nir_opt_algebraic(shader);
      progress |= nir_opt_dead_cf(shader);
      progress |= nir_opt_dce(shader);
   } while (progress);
   nir_validate_shader(shader, "after buffer format groups");
}

struct BufferGraphEvaluator {
   using Value = std::array<uint64_t, NIR_MAX_VEC_COMPONENTS>;
   std::unordered_map<nir_def *, Value> values;
   std::vector<Value> outputs;
   nir_block *previous = nullptr;
   uint32_t index, elements, tail;
   bool rgb32;
   unsigned raw_reads = 0, texture_reads = 0;

   static uint32_t raw_word(uint64_t word)
   {
      return uint32_t(word * UINT64_C(0x9e3779b1) + UINT64_C(0x12345f67));
   }

   static uint32_t texture_word(uint32_t coord, unsigned component)
   {
      return coord == UINT32_MAX ? 0u :
         uint32_t(coord * 0x85ebca6bu + component * 0xc2b2ae35u + 0x456789abu);
   }

   uint64_t scalar(nir_def *def, unsigned component = 0)
   {
      auto it = values.find(def);
      EXPECT_NE(it, values.end());
      return it == values.end() ? 0 : it->second[component];
   }

   void instruction(nir_instr *instr)
   {
      Value result = {};
      nir_def *def = nullptr;
      if (instr->type == nir_instr_type_load_const) {
         nir_load_const_instr *constant = nir_instr_as_load_const(instr);
         def = &constant->def;
         for (unsigned c = 0; c < def->num_components; ++c)
            result[c] = nir_const_value_as_uint(constant->value[c], def->bit_size);
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
            case nir_op_vec4: result[c] = scalar(alu->src[c].src.ssa,
                                                    alu->src[c].swizzle[0]); break;
            case nir_op_iadd: result[c] = src(0) + src(1); break;
            case nir_op_imul: result[c] = src(0) * src(1); break;
            case nir_op_umax: result[c] = std::max(src(0), src(1)); break;
            case nir_op_ult: result[c] = src(0) < src(1); break;
            case nir_op_ishl: result[c] = src(0) << (src(1) & 63); break;
            case nir_op_ushr: result[c] = src(0) >> (src(1) & 63); break;
            case nir_op_udiv: result[c] = src(0) / src(1); break;
            case nir_op_umod: result[c] = src(0) % src(1); break;
            case nir_op_u2u16:
            case nir_op_u2u32:
            case nir_op_u2u64: result[c] = src(0); break;
            case nir_op_uge: result[c] = src(0) >= src(1); break;
            case nir_op_ieq: result[c] = src(0) == src(1); break;
            case nir_op_ine: result[c] = src(0) != src(1); break;
            case nir_op_iand: result[c] = src(0) & src(1); break;
            case nir_op_inot: result[c] = ~src(0); break;
            case nir_op_bcsel: result[c] = src(0) ? src(1) : src(2); break;
            default: ADD_FAILURE() << "Unhandled ALU " << nir_op_infos[alu->op].name;
            }
         }
      } else if (instr->type == nir_instr_type_phi) {
         nir_phi_instr *phi = nir_instr_as_phi(instr);
         def = &phi->def;
         bool found = false;
         nir_foreach_phi_src(src, phi) {
            if (src->pred == previous) {
               result = values.at(src->src.ssa);
               found = true;
            }
         }
         EXPECT_TRUE(found);
      } else if (instr->type == nir_instr_type_tex) {
         nir_tex_instr *tex = nir_instr_as_tex(instr);
         EXPECT_EQ(tex->op, nir_texop_txf);
         int xy = nir_tex_instr_src_index(tex, nir_tex_src_backend1);
         EXPECT_GE(xy, 0);
         if (xy < 0)
            return;
         nir_def *coords = tex->src[xy].src.ssa;
         uint32_t coord = uint32_t(scalar(coords, 0) +
                                  scalar(coords, 1) * AGX_TEXTURE_BUFFER_WIDTH);
         def = &tex->def;
         for (unsigned c = 0; c < def->num_components; ++c)
            result[c] = texture_word(coord, c);
         ++texture_reads;
      } else if (instr->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         def = nir_intrinsic_infos[intr->intrinsic].has_dest ? &intr->def : nullptr;
         switch (intr->intrinsic) {
         case nir_intrinsic_load_global_invocation_id:
            result[0] = index;
            break;
         case nir_intrinsic_load_from_texture_handle_agx:
            result[0] = 0x10000;
            break;
         case nir_intrinsic_load_global_constant: {
            uint64_t addr = scalar(intr->src[0].ssa);
            if (intr->def.num_components == 3) {
               EXPECT_GE(addr, UINT64_C(0x100000));
               uint64_t word = (addr - 0x100000) / 4;
               for (unsigned c = 0; c < 3; ++c)
                  result[c] = raw_word(word + c);
               ++raw_reads;
            } else {
               switch (addr - 0x10000) {
               case AGX_G17P_BUFFER_TEXTURE_SIZE_OFFSET: result[0] = elements; break;
               case AGX_G17P_BUFFER_TEXTURE_TAIL_OFFSET: result[0] = tail; break;
               case AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET: result[0] = 0x100000; break;
               case AGX_G17P_BUFFER_TEXTURE_RGB32_OFFSET: result[0] = rgb32; break;
               default: ADD_FAILURE() << "Unexpected metadata " << addr;
               }
            }
            break;
         }
         case nir_intrinsic_store_global:
            outputs.push_back(values.at(intr->src[0].ssa));
            break;
         default: ADD_FAILURE() << "Unhandled intrinsic " <<
                     nir_intrinsic_infos[intr->intrinsic].name;
         }
      } else {
         ADD_FAILURE() << "Unhandled instruction " << instr->type;
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

   void list(struct exec_list *nodes)
   {
      foreach_list_typed(nir_cf_node, node, node, nodes) {
         if (node->type == nir_cf_node_block) {
            nir_block *block = nir_cf_node_as_block(node);
            nir_foreach_instr(instr, block)
               instruction(instr);
            previous = block;
         } else if (node->type == nir_cf_node_if) {
            nir_if *nif = nir_cf_node_as_if(node);
            list(scalar(nif->condition.ssa) ? &nif->then_list : &nif->else_list);
         } else {
            ADD_FAILURE() << "Unexpected control flow";
         }
      }
   }
};

TEST(BufferFormatGroups, GroupsIndependentReadsAndPreservesBothReadPaths)
{
   for (unsigned count : {4u, 16u, 32u, 33u}) {
      for (nir_alu_type type : {nir_type_uint32, nir_type_int32, nir_type_float32}) {
         SCOPED_TRACE(::testing::Message() << count << " type " << type);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "independent buffer format paths");
         nir_def *input = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
         nir_def *handle = nir_imm_int(&b, 7);
         nir_def *lod = nir_imm_int(&b, 0);
         std::vector<nir_def *> reads;
         for (unsigned i = 0; i < count; ++i)
            reads.push_back(&buffer_fetch(&b, handle, nir_iadd_imm(&b, input, i),
                                         lod, type)->def);
         for (unsigned i = 0; i < count; ++i)
            nir_store_global(&b, reads[i], nir_imm_int64(&b, 0x200000 + 16 * i));
         lower_buffer_shader(b.shader);
         auto choices = format_choices(b.shader);
         ASSERT_EQ(choices.size(), 1u);
         EXPECT_EQ(count_path(&choices[0]->then_list, true), 2 * std::min(count, 32u));
         EXPECT_EQ(count_path(&choices[0]->then_list, false), std::min(count, 32u));
         EXPECT_EQ(count_path(&choices[0]->else_list, true), 0u);
         EXPECT_EQ(count_path(&choices[0]->else_list, false), std::min(count, 32u));
         for (bool rgb : {false, true}) {
            for (uint32_t elements : {0u, 1u, 16u, 0xfffffff0u}) {
               for (uint32_t tail : {0u, 1u, 17u, 0xfffffffcu}) {
                  for (uint32_t index : {0u, 1u, 15u, 16u, 0xfffffff0u, UINT32_MAX}) {
                     SCOPED_TRACE(::testing::Message() << rgb << "," << elements <<
                                   "," << tail << "," << index);
                     BufferGraphEvaluator eval;
                     eval.index = index; eval.elements = elements;
                     eval.tail = tail; eval.rgb32 = rgb;
                     eval.list(&nir_shader_get_entrypoint(b.shader)->body);
                     ASSERT_EQ(eval.outputs.size(), count);
                     unsigned expected_raw = 0;
                     for (unsigned i = 0; i < count; ++i) {
                        uint32_t coord = index + i;
                        bool raw = rgb && coord < elements;
                        expected_raw += raw;
                        /* Only the texture adds the tail; RGB32 reads the
                         * view's exact address (agx_abi.h). */
                        uint32_t adjusted = coord + tail;
                        uint64_t word = uint32_t(coord * 3u);
                        for (unsigned c = 0; c < 4; ++c) {
                           uint32_t expected = raw ?
                              (c < 3 ? BufferGraphEvaluator::raw_word(word + c) :
                               type == nir_type_float32 ? 0x3f800000u : 1u) :
                              BufferGraphEvaluator::texture_word(
                                 coord >= elements ? UINT32_MAX : adjusted, c);
                           EXPECT_EQ(eval.outputs[i][c], expected);
                        }
                     }
                     EXPECT_EQ(eval.raw_reads, expected_raw);
                     EXPECT_EQ(eval.texture_reads, count - expected_raw);
                  }
               }
            }
         }
         ralloc_free(b.shader);
      }
   }
}

enum class BufferGroupControl {
   Three, Dependent, Store, Atomic, Barrier, DifferentResource,
   DifferentType, SparseOpcode, SparseFeature
};

TEST(BufferFormatGroups, RecognizesRepeatedPureHandlesAndEqualLodConstants)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "repeated buffer handles");
   nir_def *input = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   std::vector<nir_def *> reads;
   for (unsigned i = 0; i < 16; ++i) {
      nir_def *handle = nir_bindless_image_agx(&b, nir_imm_int(&b, 0),
                                               .desc_set = 2);
      nir_def *lod = nir_imm_intN_t(&b, 0, 16);
      reads.push_back(&buffer_fetch(&b, handle, nir_iadd_imm(&b, input, i),
                                   lod)->def);
   }
   for (unsigned i = 0; i < reads.size(); ++i)
      nir_store_global(&b, reads[i], nir_imm_int64(&b, 0x200000 + 16 * i));
   lower_buffer_shader(b.shader);
   auto choices = format_choices(b.shader);
   ASSERT_EQ(choices.size(), 1u);
   EXPECT_EQ(count_path(&choices[0]->then_list, true), 32u);
   EXPECT_EQ(count_path(&choices[0]->then_list, false), 16u);
   EXPECT_EQ(count_path(&choices[0]->else_list, true), 0u);
   EXPECT_EQ(count_path(&choices[0]->else_list, false), 16u);
   ralloc_free(b.shader);
}

TEST(BufferFormatGroups, LeavesIneligibleReadSequencesSeparate)
{
   for (BufferGroupControl control : {
           BufferGroupControl::Three, BufferGroupControl::Dependent,
           BufferGroupControl::Store, BufferGroupControl::Atomic,
           BufferGroupControl::Barrier, BufferGroupControl::DifferentResource,
           BufferGroupControl::DifferentType, BufferGroupControl::SparseOpcode,
           BufferGroupControl::SparseFeature}) {
      SCOPED_TRACE(unsigned(control));
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "buffer format grouping exclusions");
      nir_def *input = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *handle = nir_imm_int(&b, 7);
      nir_def *other = nir_imm_int(&b, 8);
      nir_def *lod = nir_imm_int(&b, 0);
      nir_def *previous = input;
      std::vector<nir_def *> reads;
      unsigned count = control == BufferGroupControl::Three ? 3 : 4;
      for (unsigned i = 0; i < count; ++i) {
         nir_def *coord = nir_iadd_imm(&b,
            control == BufferGroupControl::Dependent ? previous : input, i);
         nir_alu_type type = control == BufferGroupControl::DifferentType && i == 2
                               ? nir_type_float32 : nir_type_uint32;
         nir_def *resource = control == BufferGroupControl::DifferentResource && (i & 1)
                               ? other : handle;
         nir_tex_instr *tex = buffer_fetch(&b, resource, coord, lod, type,
                                     control == BufferGroupControl::SparseOpcode);
         reads.push_back(&tex->def);
         if (control == BufferGroupControl::Dependent)
            previous = nir_channel(&b, &tex->def, 0);
         if (i == 1) {
            if (control == BufferGroupControl::Store)
               nir_store_global(&b, input, nir_imm_int64(&b, 0x300000));
            if (control == BufferGroupControl::Atomic) {
               nir_def *value = nir_global_atomic(&b, 32,
                  nir_imm_int64(&b, 0x300000), input);
               nir_intrinsic_set_atomic_op(nir_def_as_intrinsic(value), nir_atomic_op_iadd);
            }
            if (control == BufferGroupControl::Barrier) {
               nir_intrinsic_instr *barrier = nir_barrier(&b);
               nir_intrinsic_set_execution_scope(barrier, SCOPE_WORKGROUP);
               nir_intrinsic_set_memory_scope(barrier, SCOPE_DEVICE);
               nir_intrinsic_set_memory_semantics(barrier, NIR_MEMORY_ACQ_REL);
               nir_intrinsic_set_memory_modes(barrier, nir_var_mem_global);
            }
         }
      }
      for (unsigned i = 0; i < count; ++i)
         nir_store_global(&b, nir_trim_vector(&b, reads[i], 4),
                          nir_imm_int64(&b, 0x200000 + 16 * i));
      lower_buffer_shader(b.shader, control == BufferGroupControl::SparseFeature);
      EXPECT_TRUE(format_choices(b.shader).empty());
      ralloc_free(b.shader);
   }
}

TEST(BufferFormatGroups, DoesNotCSEUnrelatedCodeWithoutAGroup)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "nongroupable CSE preservation");
   nir_def *input = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *constant = nir_imm_int(&b, 17);
   nir_def *a = nir_iadd(&b, input, constant);
   nir_def *other = nir_iadd(&b, input, constant);
   nir_intrinsic_instr *store_a = nir_store_global(
      &b, a, nir_imm_int64(&b, 0x400000));
   nir_intrinsic_instr *store_b = nir_store_global(
      &b, other, nir_imm_int64(&b, 0x400010));
   std::vector<nir_def *> reads;
   for (unsigned i = 0; i < 4; ++i)
      reads.push_back(&buffer_fetch(&b, nir_imm_int(&b, 7 + (i & 1)),
         nir_iadd_imm(&b, input, i), nir_imm_int(&b, 0))->def);
   for (unsigned i = 0; i < reads.size(); ++i)
      nir_store_global(&b, reads[i], nir_imm_int64(&b, 0x500000 + 16 * i));
   lower_buffer_shader(b.shader);
   EXPECT_TRUE(format_choices(b.shader).empty());
   EXPECT_EQ(store_a->src[0].ssa, a);
   EXPECT_EQ(store_b->src[0].ssa, other);
   EXPECT_NE(store_a->src[0].ssa, store_b->src[0].ssa);
   ralloc_free(b.shader);
}

TEST(BufferFormatGroups, PureHandlesRequireTheSameHeapAndOffset)
{
   for (bool change_heap : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "pure handle identity controls");
      nir_def *input = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      std::vector<nir_def *> reads;
      for (unsigned i = 0; i < 4; ++i) {
         nir_def *handle = nir_bindless_image_agx(&b,
            nir_imm_int(&b, change_heap ? 0 : 16 * (i & 1)),
            .desc_set = 2 + (change_heap ? (i & 1) : 0));
         reads.push_back(&buffer_fetch(&b, handle, nir_iadd_imm(&b, input, i),
                                      nir_imm_int(&b, 0))->def);
      }
      for (unsigned i = 0; i < reads.size(); ++i)
         nir_store_global(&b, reads[i], nir_imm_int64(&b, 0x200000 + 16 * i));
      lower_buffer_shader(b.shader);
      EXPECT_TRUE(format_choices(b.shader).empty());
      ralloc_free(b.shader);
   }
}

TEST(BufferFormatGroups, WalksOriginalSequencesAcrossControlFlow)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &agx_nir_options, "groups across control flow");
   nir_def *input = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *handle = nir_imm_int(&b, 7);
   nir_def *lod = nir_imm_int(&b, 0);
   nir_if *nif = nir_push_if(&b, nir_ieq_imm(&b, nir_iand_imm(&b, input, 1), 0));
   for (unsigned path = 0; path < 2; ++path) {
      if (path)
         nir_push_else(&b, nif);
      for (unsigned g = 0; g < (path ? 5u : 3u); ++g) {
         unsigned count = (g & 1) ? 8 : 4;
         std::vector<nir_def *> reads;
         for (unsigned i = 0; i < count; ++i)
            reads.push_back(&buffer_fetch(&b, handle, nir_iadd_imm(&b, input, 16*g+i), lod)->def);
         for (unsigned i = 0; i < count; ++i)
            nir_store_global(&b, reads[i], nir_imm_int64(&b, 0x200000+256*g+16*i));
      }
   }
   nir_pop_if(&b, nif);
   lower_buffer_shader(b.shader);
   ASSERT_EQ(format_choices(b.shader).size(), 8u);
   for (bool rgb : {false, true}) {
      for (uint32_t elements : {0u, 1u, 37u, UINT32_MAX}) {
         for (uint32_t tail : {0u, 1u, 0xfffffffcu}) {
            for (uint32_t index : {0u, 1u, 17u, UINT32_MAX}) {
               BufferGraphEvaluator eval;
               eval.index = index; eval.elements = elements;
               eval.tail = tail; eval.rgb32 = rgb;
               eval.list(&nir_shader_get_entrypoint(b.shader)->body);
               unsigned output = 0;
               for (unsigned g = 0; g < ((index & 1) ? 5u : 3u); ++g) {
                  for (unsigned i = 0; i < ((g & 1) ? 8u : 4u); ++i) {
                     uint32_t coord = index+16*g+i;
                     bool raw = rgb && coord < elements;
                     uint32_t adjusted = coord+tail;
                     uint64_t word = uint32_t(coord*3u);
                     ASSERT_LT(output, eval.outputs.size());
                     for (unsigned c = 0; c < 4; ++c) {
                        uint32_t expected = raw ? (c < 3 ? BufferGraphEvaluator::raw_word(word+c) : 1u) : BufferGraphEvaluator::texture_word(coord >= elements ? UINT32_MAX : adjusted, c);
                        EXPECT_EQ(eval.outputs[output][c], expected);
                     }
                     ++output;
                  }
               }
               EXPECT_EQ(eval.outputs.size(), output);
            }
         }
      }
   }
   ralloc_free(b.shader);
}

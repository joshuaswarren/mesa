/*
 * Copyright © 2019 Red Hat, Inc
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gtest/gtest.h>

#include "nir.h"
#include "nir_builder.h"
#include "nir_serialize.h"

namespace {

class nir_serialize_test : public ::testing::TestWithParam<int> {
protected:
   nir_serialize_test();
   ~nir_serialize_test();

   void serialize();
   nir_alu_instr *get_last_alu(nir_shader *);
   void ASSERT_SWIZZLE_EQ(nir_alu_instr *, nir_alu_instr *, unsigned count, unsigned src);

   nir_builder *b, _b;
   nir_shader *dup;
   const nir_shader_compiler_options options;
};

nir_serialize_test::nir_serialize_test()
:  dup(NULL), options()
{
   glsl_type_singleton_init_or_ref();

   _b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "serialize test");
   b = &_b;
}

nir_serialize_test::~nir_serialize_test()
{
   if (HasFailure()) {
      printf("\nShader from the failed test\n\n");
      printf("original Shader:\n");
      nir_print_shader(b->shader, stdout);
      printf("serialized Shader:\n");
      nir_print_shader(dup, stdout);
   }

   ralloc_free(b->shader);

   glsl_type_singleton_decref();
}

void
nir_serialize_test::serialize() {
   struct blob blob;
   struct blob_reader reader;

   blob_init(&blob);

   nir_serialize(&blob, b->shader, false);
   blob_reader_init(&reader, blob.data, blob.size);
   nir_shader *cloned = nir_deserialize(b->shader, &options, &reader);
   blob_finish(&blob);

   dup = cloned;

   nir_validate_shader(b->shader, "original");
   nir_validate_shader(b->shader, "cloned");
}

nir_alu_instr *
nir_serialize_test::get_last_alu(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   return nir_instr_as_alu(nir_block_last_instr(nir_impl_last_block(impl)));
}

void
nir_serialize_test::ASSERT_SWIZZLE_EQ(nir_alu_instr *a, nir_alu_instr *b, unsigned c, unsigned s)
{
   ASSERT_EQ(memcmp(a->src[s].swizzle, b->src[s].swizzle, c), 0);
}

class nir_serialize_all_test : public nir_serialize_test {};
class nir_serialize_all_but_one_test : public nir_serialize_test {};

} // namespace

#if NIR_MAX_VEC_COMPONENTS == 16
#define COMPONENTS 2, 3, 4, 8, 16
#else
#define COMPONENTS 2, 3, 4
#endif


INSTANTIATE_TEST_SUITE_P(
   nir_serialize_all_test,
   nir_serialize_all_test,
   ::testing::Values(1, COMPONENTS)
);

INSTANTIATE_TEST_SUITE_P(
   nir_serialize_all_but_one_test,
   nir_serialize_all_but_one_test,
   ::testing::Values(COMPONENTS)
);

TEST_P(nir_serialize_all_test, alu_single_value_src_swizzle)
{
   nir_def *zero = nir_imm_zero(b, GetParam(), 32);
   nir_def *fmax = nir_fmax(b, zero, zero);

   nir_alu_instr *fmax_alu = nir_def_as_alu(fmax);

   memset(fmax_alu->src[0].swizzle, GetParam() - 1, NIR_MAX_VEC_COMPONENTS);
   memset(fmax_alu->src[1].swizzle, GetParam() - 1, NIR_MAX_VEC_COMPONENTS);

   serialize();

   nir_alu_instr *fmax_alu_dup = get_last_alu(dup);

   ASSERT_SWIZZLE_EQ(fmax_alu, fmax_alu_dup, GetParam(), 0);
   ASSERT_SWIZZLE_EQ(fmax_alu, fmax_alu_dup, GetParam(), 1);
}

TEST_P(nir_serialize_all_test, alu_vec)
{
   nir_def *undef = nir_undef(b, GetParam(), 32);
   nir_def *undefs[] = {
      undef, undef, undef, undef,
      undef, undef, undef, undef,
      undef, undef, undef, undef,
      undef, undef, undef, undef,
   };

   nir_def *vec = nir_vec(b, undefs, GetParam());
   nir_alu_instr *vec_alu = nir_def_as_alu(vec);
   for (int i = 0; i < GetParam(); i++)
      vec_alu->src[i].swizzle[0] = (GetParam() - 1) - i;

   serialize();

   nir_alu_instr *vec_alu_dup = get_last_alu(dup);

   ASSERT_SWIZZLE_EQ(vec_alu, vec_alu_dup, 1, 0);
}

TEST_P(nir_serialize_all_test, alu_two_components_full_swizzle)
{
   nir_def *undef = nir_undef(b, 2, 32);
   nir_def *fma = nir_ffma_weak(b, undef, undef, undef);
   nir_alu_instr *fma_alu = nir_def_as_alu(fma);

   fma->num_components = GetParam();

   memset(fma_alu->src[0].swizzle, 1, GetParam());
   memset(fma_alu->src[1].swizzle, 1, GetParam());
   memset(fma_alu->src[2].swizzle, 1, GetParam());

   serialize();

   nir_alu_instr *fma_alu_dup = get_last_alu(dup);

   ASSERT_SWIZZLE_EQ(fma_alu, fma_alu_dup, GetParam(), 0);
   ASSERT_SWIZZLE_EQ(fma_alu, fma_alu_dup, GetParam(), 1);
   ASSERT_SWIZZLE_EQ(fma_alu, fma_alu_dup, GetParam(), 2);
}

TEST_P(nir_serialize_all_but_one_test, single_channel)
{
   nir_def *zero = nir_undef(b, GetParam(), 32);
   nir_def *vec = nir_channel(b, zero, GetParam() - 1);
   nir_alu_instr *vec_alu = nir_def_as_alu(vec);

   serialize();

   nir_alu_instr *vec_alu_dup = get_last_alu(dup);

   ASSERT_SWIZZLE_EQ(vec_alu, vec_alu_dup, 1, 0);
}

namespace {

/* nir_deserialize() must reject a malformed blob instead of reading or
 * writing outside its object table. Each test starts from a valid blob and
 * changes one field.
 */
class nir_deserialize_malformed_test : public ::testing::Test {
protected:
   nir_deserialize_malformed_test()
   {
      glsl_type_singleton_init_or_ref();
   }

   ~nir_deserialize_malformed_test()
   {
      glsl_type_singleton_decref();
   }

   /* A compute shader whose main calls function `callee`, one of two. */
   void serialize_call(unsigned callee, struct blob *out)
   {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &options, "call");
      nir_function *fns[] = {
         nir_function_create(b.shader, "f0"),
         nir_function_create(b.shader, "f1"),
      };
      nir_build_call(&b, fns[callee], 0, NULL);
      nir_iadd_imm(&b, nir_imm_int(&b, 1), 2);

      blob_init(out);
      nir_serialize(out, b.shader, false);
      ralloc_free(b.shader);
   }

   /* Whether data deserializes, reading all of it. */
   bool deserializes(const void *data, size_t size)
   {
      struct blob_reader reader;
      blob_reader_init(&reader, data, size);
      nir_shader *nir = nir_deserialize(NULL, &options, &reader);
      const bool ok = nir && !reader.overrun && reader.current == reader.end;
      ralloc_free(nir);
      return ok;
   }

   bool deserializes_function(const void *data, size_t size)
   {
      void *mem = ralloc_context(NULL);
      struct blob_reader reader;
      blob_reader_init(&reader, data, size);
      nir_function *fxn = nir_deserialize_function(mem, &options, &reader);
      const bool ok = fxn && !reader.overrun && reader.current == reader.end;
      ralloc_free(mem);
      return ok;
   }

   static uint32_t read_u32(const struct blob *blob, size_t offset)
   {
      uint32_t v;
      memcpy(&v, blob->data + offset, sizeof(v));
      return v;
   }

   static void write_u32(struct blob *blob, size_t offset, uint32_t v)
   {
      memcpy(blob->data + offset, &v, sizeof(v));
   }

   const nir_shader_compiler_options options = {};
};

} // namespace

TEST_F(nir_deserialize_malformed_test, object_count)
{
   struct blob blob;
   serialize_call(0, &blob);
   ASSERT_FALSE(blob.out_of_memory);
   ASSERT_TRUE(deserializes(blob.data, blob.size));

   /* The blob starts with the number of objects it holds. */
   const uint32_t count = read_u32(&blob, 0);
   ASSERT_GT(count, 1u);

   const struct {
      uint32_t count;
      bool ok;
   } cases[] = {
      { 0, false },
      { count - 1, false },
      { count, true },
      { count + 1, true },
      { (1u << 20) + 1, false },
      { UINT32_MAX, false },
   };
   for (const auto &c : cases) {
      write_u32(&blob, 0, c.count);
      EXPECT_EQ(deserializes(blob.data, blob.size), c.ok)
         << "object count " << c.count << ", needs " << count;
   }

   blob_finish(&blob);
}

TEST_F(nir_deserialize_malformed_test, object_reference)
{
   /* The two blobs differ only in the callee's object index. */
   struct blob a, b;
   serialize_call(0, &a);
   serialize_call(1, &b);
   ASSERT_FALSE(a.out_of_memory || b.out_of_memory);
   ASSERT_EQ(a.size, b.size);

   size_t ref = a.size;
   for (size_t i = 0; i + 4 <= a.size; i += 4) {
      if (read_u32(&a, i) != read_u32(&b, i)) {
         ASSERT_EQ(ref, a.size) << "blobs differ in more than one word";
         ref = i;
      }
   }
   ASSERT_LT(ref, a.size);

   const uint32_t count = read_u32(&a, 0);
   const uint32_t other = read_u32(&b, ref);

   const struct {
      uint32_t idx;
      bool ok;
   } cases[] = {
      { other, true },
      /* Read after the call: not an object yet. */
      { count - 1, false },
      { count, false },
      { UINT32_MAX, false },
   };
   for (const auto &c : cases) {
      write_u32(&a, ref, c.idx);
      EXPECT_EQ(deserializes(a.data, a.size), c.ok)
         << "callee index " << c.idx << " of " << count << " objects";
   }

   blob_finish(&a);
   blob_finish(&b);
}

TEST_F(nir_deserialize_malformed_test, truncated)
{
   struct blob blob;
   serialize_call(0, &blob);
   ASSERT_FALSE(blob.out_of_memory);
   ASSERT_TRUE(deserializes(blob.data, blob.size));

   for (size_t size = 0; size < blob.size; ++size)
      EXPECT_FALSE(deserializes(blob.data, size)) << "first " << size << " bytes";

   blob_finish(&blob);
}

TEST_F(nir_deserialize_malformed_test, function)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &options, "function");
   nir_iadd_imm(&b, nir_imm_int(&b, 1), 2);

   struct blob blob;
   blob_init(&blob);
   nir_serialize_function(&blob, b.impl->function);
   ralloc_free(b.shader);
   ASSERT_FALSE(blob.out_of_memory);
   ASSERT_TRUE(deserializes_function(blob.data, blob.size));

   const uint32_t count = read_u32(&blob, 0);
   ASSERT_GT(count, 1u);

   for (size_t size = 0; size < blob.size; ++size)
      EXPECT_FALSE(deserializes_function(blob.data, size)) << "first " << size << " bytes";

   const struct {
      uint32_t count;
      bool ok;
   } cases[] = {
      { 0, false },
      { count - 1, false },
      { count, true },
      { (1u << 20) + 1, false },
   };
   for (const auto &c : cases) {
      write_u32(&blob, 0, c.count);
      EXPECT_EQ(deserializes_function(blob.data, blob.size), c.ok)
         << "object count " << c.count << ", needs " << count;
   }

   blob_finish(&blob);
}

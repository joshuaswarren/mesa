/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A shader object keeps some NIR serialized in blobs of its own (sample
 * variant and twin sources, monolithic and local-read programs, the vertex
 * preamble) and deserializes it on first use. Those blobs come from shader
 * binaries and caches, so a blob that is cut short, carries extra bytes,
 * holds a shader of another stage, or has a broken object table must be
 * refused, not built from.
 */
#undef NDEBUG
#include "hk_shader.h"

#include "asahi/compiler/agx_compile.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "util/blob.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
serialize(mesa_shader_stage stage, bool entrypoint, struct blob *blob)
{
   nir_builder b =
      nir_builder_init_simple_shader(stage, &agx_nir_options, "nested");
   nir_iadd_imm(&b, nir_imm_int(&b, 1), 2);
   if (!entrypoint)
      b.impl->function->is_entrypoint = false;

   blob_init(blob);
   nir_serialize(blob, b.shader, false);
   assert(!blob->out_of_memory);
   ralloc_free(b.shader);
}

static bool
accepts(const void *data, size_t size, mesa_shader_stage stage)
{
   void *mem = ralloc_context(NULL);
   nir_shader *nir = hk_deserialize_nested_nir(mem, data, size, stage);
   const bool ok = nir != NULL;
   if (nir)
      assert(nir->info.stage == stage && nir_shader_get_entrypoint(nir));
   ralloc_free(mem);
   return ok;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();

   struct blob blob;
   serialize(MESA_SHADER_FRAGMENT, true, &blob);
   assert(accepts(blob.data, blob.size, MESA_SHADER_FRAGMENT));

   /* A shader of another stage than the one it is built as. */
   assert(!accepts(blob.data, blob.size, MESA_SHADER_COMPUTE));
   assert(!accepts(blob.data, blob.size, MESA_SHADER_VERTEX));

   /* Cut short anywhere. */
   for (size_t size = 0; size < blob.size; ++size)
      assert(!accepts(blob.data, size, MESA_SHADER_FRAGMENT));

   /* Extra bytes after the shader. */
   struct blob longer;
   blob_init(&longer);
   blob_write_bytes(&longer, blob.data, blob.size);
   blob_write_uint32(&longer, 0);
   assert(!longer.out_of_memory);
   assert(!accepts(longer.data, longer.size, MESA_SHADER_FRAGMENT));
   blob_finish(&longer);

   /* An object table too small for the objects the blob holds. The blob
    * starts with its size. */
   uint32_t count;
   memcpy(&count, blob.data, sizeof(count));
   assert(count > 1);
   const uint32_t bad_counts[] = { 0, count - 1, UINT32_MAX };
   for (unsigned i = 0; i < ARRAY_SIZE(bad_counts); ++i) {
      memcpy(blob.data, &bad_counts[i], sizeof(count));
      assert(!accepts(blob.data, blob.size, MESA_SHADER_FRAGMENT));
   }
   memcpy(blob.data, &count, sizeof(count));
   assert(accepts(blob.data, blob.size, MESA_SHADER_FRAGMENT));
   blob_finish(&blob);

   /* A shader with no entrypoint. */
   serialize(MESA_SHADER_FRAGMENT, false, &blob);
   assert(!accepts(blob.data, blob.size, MESA_SHADER_FRAGMENT));
   blob_finish(&blob);

   glsl_type_singleton_decref();
   printf("hk_nested_nir: ok\n");
   return 0;
}

/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Submit-time packing lets a looping compute alternate share a packet only up
 * to the bound its loops were proven to when it was compiled
 * (hk_shader::pack_loop_bound), and counts their instructions by the work
 * those loops do. A shader that comes from the disk shader cache, a pipeline
 * cache or a shader binary is not compiled again, so it must bring both with
 * it, or its loops pack as unbounded. Round-trips shaders through
 * hk_shader_serialize() and hk_deserialize_shader().
 */
#undef NDEBUG
#include "hk_device.h"
#include "hk_physical_device.h"
#include "hk_shader.h"

#include "util/blob.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The fake device has no instance to report a refused binary to. */
__typeof__(__vk_errorf) __wrap___vk_errorf;
VkResult
__wrap___vk_errorf(const void *object, VkResult result, const char *file,
                   int line, const char *format, ...)
{
   return result;
}

/* What follows the bound: code_size and data_size, and no code or data. */
#define TAIL_SIZE (2 * sizeof(uint32_t))
/* At least valid, count, work_valid, work_count and static_nir_ops, then
 * kind, lhs, rhs and imm per op. */
#define BOUND_MIN_SIZE(count) (8 + 7 * (size_t)(count))

static const uint64_t params[HK_PACK_LOOP_PARAM_COUNT] = {
   [HK_PACK_LOOP_PARAM_IM2COL_KH] = 3,
   [HK_PACK_LOOP_PARAM_DW_KH] = 5,
   [HK_PACK_LOOP_PARAM_DW_KW] = 7,
   [HK_PACK_LOOP_PARAM_GEMM_K_ITERS] = 11,
   [HK_PACK_LOOP_PARAM_PAD_H] = 13,
   [HK_PACK_LOOP_PARAM_PAD_W] = 17,
   [HK_PACK_LOOP_PARAM_PAD_ZLP_ITERS] = 19,
};

/* Every op kind, and a constant that needs all 32 bits of imm. */
static void
fill_mixed_bound(struct hk_pack_loop_bound *lb)
{
   static const struct hk_pack_loop_op ops[] = {
      {HK_PACK_LOOP_OP_PARAM, 0, 0, HK_PACK_LOOP_PARAM_DW_KH},
      {HK_PACK_LOOP_OP_PARAM, 0, 0, HK_PACK_LOOP_PARAM_DW_KW},
      {HK_PACK_LOOP_OP_MUL, 0, 1, 0},
      {HK_PACK_LOOP_OP_CONST, 0, 0, 0x89abcdefu},
      {HK_PACK_LOOP_OP_ADD, 2, 3, 0},
      {HK_PACK_LOOP_OP_PARAM, 0, 0, HK_PACK_LOOP_PARAM_PAD_ZLP_ITERS},
      {HK_PACK_LOOP_OP_MAX, 4, 5, 0},
   };
   lb->valid = true;
   lb->count = ARRAY_SIZE(ops);
   memcpy(lb->ops, ops, sizeof(ops));

   static const struct hk_pack_loop_op work_ops[] = {
      {HK_PACK_LOOP_OP_PARAM, 0, 0, HK_PACK_LOOP_PARAM_GEMM_K_ITERS},
      {HK_PACK_LOOP_OP_CONST, 0, 0, 0x07654321u},
      {HK_PACK_LOOP_OP_MUL, 0, 1, 0},
      {HK_PACK_LOOP_OP_CONST, 0, 0, 9},
      {HK_PACK_LOOP_OP_ADD, 2, 3, 0},
   };
   lb->work_valid = true;
   lb->work_count = ARRAY_SIZE(work_ops);
   lb->static_nir_ops = 0x00012345u;
   memcpy(lb->work_ops, work_ops, sizeof(work_ops));
}

/* As many ops as a bound holds: 1, then 63 increments by it, and for the
 * work 2, then 127 increments by it. The unused imm of each ADD still has to
 * come back. */
static void
fill_full_bound(struct hk_pack_loop_bound *lb)
{
   lb->valid = true;
   lb->count = HK_PACK_LOOP_MAX_OPS;
   lb->ops[0].kind = HK_PACK_LOOP_OP_CONST;
   lb->ops[0].imm = 1;
   for (unsigned i = 1; i < HK_PACK_LOOP_MAX_OPS; ++i) {
      lb->ops[i].kind = HK_PACK_LOOP_OP_ADD;
      lb->ops[i].lhs = i - 1;
      lb->ops[i].rhs = 0;
      lb->ops[i].imm = 0xa5000000u | i;
   }

   lb->work_valid = true;
   lb->work_count = HK_PACK_LOOP_WORK_MAX_OPS;
   lb->static_nir_ops = 0xfedcba98u;
   lb->work_ops[0].kind = HK_PACK_LOOP_OP_CONST;
   lb->work_ops[0].imm = 2;
   for (unsigned i = 1; i < HK_PACK_LOOP_WORK_MAX_OPS; ++i) {
      lb->work_ops[i].kind = HK_PACK_LOOP_OP_ADD;
      lb->work_ops[i].lhs = i - 1;
      lb->work_ops[i].rhs = 0;
      lb->work_ops[i].imm = 0x5a000000u | i;
   }
}

static void
serialize(struct hk_device *dev, const struct hk_shader *shader,
          struct blob *blob)
{
   blob_init(blob);
   hk_shader_serialize(&dev->vk, shader, blob);
   assert(!blob->out_of_memory && blob->size > TAIL_SIZE);
}

/* Reads `size` bytes of `data` as a variant that was never compiled, which
 * needs no upload. The bound of `out` starts as garbage, so a success shows
 * that every field of the bound was written. */
static VkResult
deserialize(struct hk_device *dev, const void *data, size_t size,
            struct hk_shader *out)
{
   memset(out, 0, sizeof(*out));
   memset(&out->pack_loop_bound, 0xa5, sizeof(out->pack_loop_bound));
   struct blob_reader reader;
   blob_reader_init(&reader, data, size);
   const VkResult result = hk_deserialize_shader(dev, &reader, out, true,
                                                 false);
   if (result == VK_SUCCESS)
      assert(reader.current == reader.end && !reader.overrun);
   return result;
}

/* Serializes `in` and reads it back into `out`; returns the blob's size. */
static size_t
round_trip(struct hk_device *dev, const struct hk_shader *in,
           struct hk_shader *out)
{
   struct blob blob;
   serialize(dev, in, &blob);
   assert(deserialize(dev, blob.data, blob.size, out) == VK_SUCCESS);
   assert(out->info.stage == in->info.stage);
   const size_t size = blob.size;
   blob_finish(&blob);
   return size;
}

static void
check_bound_equal(const struct hk_pack_loop_bound *a,
                  const struct hk_pack_loop_bound *b)
{
   assert(a->valid == b->valid);
   assert(a->count == b->count);
   /* Every op, so the ops beyond count come back zeroed as they were. */
   for (unsigned i = 0; i < HK_PACK_LOOP_MAX_OPS; ++i) {
      assert(a->ops[i].kind == b->ops[i].kind);
      assert(a->ops[i].lhs == b->ops[i].lhs);
      assert(a->ops[i].rhs == b->ops[i].rhs);
      assert(a->ops[i].imm == b->ops[i].imm);
   }

   assert(a->work_valid == b->work_valid);
   assert(a->work_count == b->work_count);
   assert(a->static_nir_ops == b->static_nir_ops);
   for (unsigned i = 0; i < HK_PACK_LOOP_WORK_MAX_OPS; ++i) {
      assert(a->work_ops[i].kind == b->work_ops[i].kind);
      assert(a->work_ops[i].lhs == b->work_ops[i].lhs);
      assert(a->work_ops[i].rhs == b->work_ops[i].rhs);
      assert(a->work_ops[i].imm == b->work_ops[i].imm);
   }

   uint64_t bound_a = 0, bound_b = 0;
   bool ok_a = hk_pack_eval_loop_bound(a, params, &bound_a);
   bool ok_b = hk_pack_eval_loop_bound(b, params, &bound_b);
   assert(ok_a == ok_b && bound_a == bound_b);
   ok_a = hk_pack_eval_work_bound(a, params, &bound_a);
   ok_b = hk_pack_eval_work_bound(b, params, &bound_b);
   assert(ok_a == ok_b && bound_a == bound_b);
   ok_a = hk_pack_eval_effective_instrs(a, 40, params, &bound_a);
   ok_b = hk_pack_eval_effective_instrs(b, 40, params, &bound_b);
   assert(ok_a == ok_b && bound_a == bound_b);
}

/* A blob of `in` in which byte `offset` is `value` is refused. */
static void
check_refused(struct hk_device *dev, const struct hk_shader *in,
              size_t offset, uint8_t value, struct hk_shader *out)
{
   struct blob blob;
   serialize(dev, in, &blob);
   assert(offset < blob.size);
   blob.data[offset] = value;
   assert(deserialize(dev, blob.data, blob.size, out) ==
          VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   blob_finish(&blob);
}

static void
check_chip(enum agx_chip chip)
{
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_shader *in = calloc(1, sizeof(*in));
   struct hk_shader *out = calloc(1, sizeof(*out));
   assert(pdev && dev && in && out);
   pdev->dev.chip = chip;
   vk_object_base_init(NULL, &pdev->vk.base, VK_OBJECT_TYPE_PHYSICAL_DEVICE);
   dev->dev.chip = chip;
   dev->vk.physical = &pdev->vk;
   dev->vk.alloc = *vk_default_allocator();
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);

   /* The chips that compile compute alternates, whose loop bounds count. */
   const bool carried = agx_chip_native_or_m5(chip);

   /* A fragment shader carries no bound, on any chip. */
   in->info.stage = MESA_SHADER_FRAGMENT;
   const size_t fragment_size = round_trip(dev, in, out);

   struct hk_pack_loop_bound none;
   memset(&none, 0, sizeof(none));
   uint64_t bound = 0;

   /* A proven bound comes back op for op, and evaluates the same. AGX2 never
    * compiles compute alternates and writes none. */
   in->info.stage = MESA_SHADER_COMPUTE;
   fill_mixed_bound(&in->pack_loop_bound);
   const unsigned mixed_count =
      in->pack_loop_bound.count + in->pack_loop_bound.work_count;
   assert(hk_pack_eval_loop_bound(&in->pack_loop_bound, params, &bound) &&
          bound == 0x89abcdefull + 5 * 7);
   assert(hk_pack_eval_work_bound(&in->pack_loop_bound, params, &bound) &&
          bound == 11ull * 0x07654321u + 9);
   const size_t mixed_size = round_trip(dev, in, out);
   check_bound_equal(carried ? &in->pack_loop_bound : &none,
                     &out->pack_loop_bound);
   if (carried) {
      /* An unproven work bound, then an unproven loop bound, keeps the ops
       * it was built from. */
      in->pack_loop_bound.work_valid = false;
      round_trip(dev, in, out);
      check_bound_equal(&in->pack_loop_bound, &out->pack_loop_bound);
      in->pack_loop_bound.valid = false;
      round_trip(dev, in, out);
      check_bound_equal(&in->pack_loop_bound, &out->pack_loop_bound);
   }

   /* Without a bound, a compute shader loads without one. Its blob is the
    * fragment shader's with the bound before code_size, so the bound starts
    * where the fragment shader's code_size does. */
   memset(&in->pack_loop_bound, 0, sizeof(in->pack_loop_bound));
   const size_t empty_size = round_trip(dev, in, out);
   check_bound_equal(&none, &out->pack_loop_bound);
   const size_t at = fragment_size - TAIL_SIZE;
   if (carried) {
      assert(empty_size >= fragment_size + BOUND_MIN_SIZE(0));
      assert(mixed_size >= empty_size + 7 * mixed_count);
   } else {
      /* The AGX2 payload is unchanged. */
      assert(empty_size == fragment_size && mixed_size == fragment_size);
   }

   memset(&in->pack_loop_bound, 0, sizeof(in->pack_loop_bound));
   fill_full_bound(&in->pack_loop_bound);
   assert(hk_pack_eval_loop_bound(&in->pack_loop_bound, params, &bound) &&
          bound == HK_PACK_LOOP_MAX_OPS);
   assert(hk_pack_eval_work_bound(&in->pack_loop_bound, params, &bound) &&
          bound == 2 * HK_PACK_LOOP_WORK_MAX_OPS);
   const size_t full_size = round_trip(dev, in, out);
   check_bound_equal(carried ? &in->pack_loop_bound : &none,
                     &out->pack_loop_bound);
   if (carried) {
      assert(full_size >= empty_size + 7 * (HK_PACK_LOOP_MAX_OPS +
                                            HK_PACK_LOOP_WORK_MAX_OPS));

      /* The bound starts where expected, and malformed copies of it are
       * refused: valid neither 0 nor 1, more ops than a bound holds. */
      struct blob blob;
      serialize(dev, in, &blob);
      assert(blob.data[at] == 1 && blob.data[at + 1] == HK_PACK_LOOP_MAX_OPS);
      check_refused(dev, in, at, 2, out);
      check_refused(dev, in, at + 1, HK_PACK_LOOP_MAX_OPS + 1, out);
      check_refused(dev, in, at + 1, UINT8_MAX, out);

      /* Cut anywhere in the bound or after it. */
      for (size_t cut = at; cut < blob.size; ++cut)
         assert(deserialize(dev, blob.data, cut, out) ==
                VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
      blob_finish(&blob);

      /* With no loop ops, the work header follows valid and count. */
      in->pack_loop_bound.valid = false;
      in->pack_loop_bound.count = 0;
      memset(in->pack_loop_bound.ops, 0, sizeof(in->pack_loop_bound.ops));
      round_trip(dev, in, out);
      check_bound_equal(&in->pack_loop_bound, &out->pack_loop_bound);
      serialize(dev, in, &blob);
      uint32_t static_nir_ops;
      memcpy(&static_nir_ops, blob.data + at + 4, sizeof(static_nir_ops));
      assert(blob.data[at] == 0 && blob.data[at + 1] == 0 &&
             blob.data[at + 2] == 1 &&
             blob.data[at + 3] == HK_PACK_LOOP_WORK_MAX_OPS &&
             static_nir_ops == 0xfedcba98u);
      blob_finish(&blob);
      check_refused(dev, in, at + 2, 2, out);
      check_refused(dev, in, at + 3, HK_PACK_LOOP_WORK_MAX_OPS + 1, out);
      check_refused(dev, in, at + 3, UINT8_MAX, out);
   } else {
      assert(full_size == empty_size);
   }

   /* A fragment shader never writes one, whatever it holds. */
   in->info.stage = MESA_SHADER_FRAGMENT;
   assert(round_trip(dev, in, out) == fragment_size);
   check_bound_equal(&none, &out->pack_loop_bound);

   free(out);
   free(in);
   free(dev);
   free(pdev);
}

int
main(void)
{
   /* G17P (A18 Pro), M5, the native M3 and M4 parts, and AGX2. */
   const enum agx_chip chips[] = {
      AGX_CHIP_G17P, AGX_CHIP_G17G, AGX_CHIP_G17S, AGX_CHIP_G16G,
      AGX_CHIP_G15G, AGX_CHIP_G13G, AGX_CHIP_G14G,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(chips); ++i)
      check_chip(chips[i]);

   printf("PASS: packed-loop bounds round-trip through shader binaries on "
          "%zu chips\n",
          ARRAY_SIZE(chips));
   return 0;
}

/*
 * Copyright 2021 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include <gtest/gtest.h>

#define CASE(copies, expected)                                                 \
   do {                                                                        \
      agx_builder *A = agx_test_builder(mem_ctx);                              \
      agx_builder *B = agx_test_builder(mem_ctx);                              \
                                                                               \
      agx_emit_parallel_copies(A, copies, ARRAY_SIZE(copies));                 \
                                                                               \
      {                                                                        \
         agx_builder *b = B;                                                   \
         expected;                                                             \
      }                                                                        \
                                                                               \
      ASSERT_SHADER_EQUAL(A->shader, B->shader);                               \
   } while (0)

class LowerParallelCopy : public testing::Test {
 protected:
   LowerParallelCopy()
   {
      mem_ctx = ralloc_context(NULL);
   }

   ~LowerParallelCopy()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;
};

TEST_F(LowerParallelCopy, UnrelatedCopies)
{
   struct agx_copy test_1[] = {
      {.dest = 0, .src = agx_register(2, AGX_SIZE_32)},
      {.dest = 4, .src = agx_register(6, AGX_SIZE_32)},
   };

   CASE(test_1, {
      agx_mov_to(b, agx_register(0, AGX_SIZE_32), agx_register(2, AGX_SIZE_32));
      agx_mov_to(b, agx_register(4, AGX_SIZE_32), agx_register(6, AGX_SIZE_32));
   });

   struct agx_copy test_2[] = {
      {.dest = 0, .src = agx_register(1, AGX_SIZE_16)},
      {.dest = 4, .src = agx_register(5, AGX_SIZE_16)},
   };

   CASE(test_2, {
      agx_mov_to(b, agx_register(0, AGX_SIZE_16), agx_register(1, AGX_SIZE_16));
      agx_mov_to(b, agx_register(4, AGX_SIZE_16), agx_register(5, AGX_SIZE_16));
   });
}

TEST_F(LowerParallelCopy, RelatedSource)
{
   struct agx_copy test_1[] = {
      {.dest = 0, .src = agx_register(2, AGX_SIZE_32)},
      {.dest = 4, .src = agx_register(2, AGX_SIZE_32)},
   };

   CASE(test_1, {
      agx_mov_to(b, agx_register(0, AGX_SIZE_32), agx_register(2, AGX_SIZE_32));
      agx_mov_to(b, agx_register(4, AGX_SIZE_32), agx_register(2, AGX_SIZE_32));
   });

   struct agx_copy test_2[] = {
      {.dest = 0, .src = agx_register(1, AGX_SIZE_16)},
      {.dest = 4, .src = agx_register(1, AGX_SIZE_16)},
   };

   CASE(test_2, {
      agx_mov_to(b, agx_register(0, AGX_SIZE_16), agx_register(1, AGX_SIZE_16));
      agx_mov_to(b, agx_register(4, AGX_SIZE_16), agx_register(1, AGX_SIZE_16));
   });
}

TEST_F(LowerParallelCopy, DependentCopies)
{
   struct agx_copy test_1[] = {
      {.dest = 0, .src = agx_register(2, AGX_SIZE_32)},
      {.dest = 4, .src = agx_register(0, AGX_SIZE_32)},
   };

   CASE(test_1, {
      agx_mov_to(b, agx_register(4, AGX_SIZE_32), agx_register(0, AGX_SIZE_32));
      agx_mov_to(b, agx_register(0, AGX_SIZE_32), agx_register(2, AGX_SIZE_32));
   });

   struct agx_copy test_2[] = {
      {.dest = 0, .src = agx_register(1, AGX_SIZE_16)},
      {.dest = 4, .src = agx_register(0, AGX_SIZE_16)},
   };

   CASE(test_2, {
      agx_mov_to(b, agx_register(4, AGX_SIZE_16), agx_register(0, AGX_SIZE_16));
      agx_mov_to(b, agx_register(0, AGX_SIZE_16), agx_register(1, AGX_SIZE_16));
   });
}

TEST_F(LowerParallelCopy, ManyDependentCopies)
{
   struct agx_copy test_1[] = {
      {.dest = 0, .src = agx_register(2, AGX_SIZE_32)},
      {.dest = 4, .src = agx_register(0, AGX_SIZE_32)},
      {.dest = 8, .src = agx_register(6, AGX_SIZE_32)},
      {.dest = 6, .src = agx_register(4, AGX_SIZE_32)},
   };

   CASE(test_1, {
      agx_mov_to(b, agx_register(8, AGX_SIZE_32), agx_register(6, AGX_SIZE_32));
      agx_mov_to(b, agx_register(6, AGX_SIZE_32), agx_register(4, AGX_SIZE_32));
      agx_mov_to(b, agx_register(4, AGX_SIZE_32), agx_register(0, AGX_SIZE_32));
      agx_mov_to(b, agx_register(0, AGX_SIZE_32), agx_register(2, AGX_SIZE_32));
   });

   struct agx_copy test_2[] = {
      {.dest = 0, .src = agx_register(1, AGX_SIZE_16)},
      {.dest = 2, .src = agx_register(0, AGX_SIZE_16)},
      {.dest = 4, .src = agx_register(3, AGX_SIZE_16)},
      {.dest = 3, .src = agx_register(2, AGX_SIZE_16)},
   };

   CASE(test_2, {
      agx_mov_to(b, agx_register(4, AGX_SIZE_16), agx_register(3, AGX_SIZE_16));
      agx_mov_to(b, agx_register(3, AGX_SIZE_16), agx_register(2, AGX_SIZE_16));
      agx_mov_to(b, agx_register(2, AGX_SIZE_16), agx_register(0, AGX_SIZE_16));
      agx_mov_to(b, agx_register(0, AGX_SIZE_16), agx_register(1, AGX_SIZE_16));
   });
}

TEST_F(LowerParallelCopy, Swap)
{
   struct agx_copy test_1[] = {
      {.dest = 0, .src = agx_register(2, AGX_SIZE_32)},
      {.dest = 2, .src = agx_register(0, AGX_SIZE_32)},
   };

   CASE(test_1, {
      agx_swap(b, agx_register(0, AGX_SIZE_32), agx_register(2, AGX_SIZE_32));
   });

   struct agx_copy test_2[] = {
      {.dest = 0, .src = agx_register(1, AGX_SIZE_16)},
      {.dest = 1, .src = agx_register(0, AGX_SIZE_16)},
   };

   CASE(test_2, {
      agx_swap(b, agx_register(0, AGX_SIZE_16), agx_register(1, AGX_SIZE_16));
   });
}

TEST_F(LowerParallelCopy, Cycle3)
{
   struct agx_copy test[] = {
      {.dest = 0, .src = agx_register(1, AGX_SIZE_16)},
      {.dest = 1, .src = agx_register(2, AGX_SIZE_16)},
      {.dest = 2, .src = agx_register(0, AGX_SIZE_16)},
   };

   CASE(test, {
      agx_swap(b, agx_register(0, AGX_SIZE_16), agx_register(1, AGX_SIZE_16));
      agx_swap(b, agx_register(1, AGX_SIZE_16), agx_register(2, AGX_SIZE_16));
   });
}

TEST_F(LowerParallelCopy, Immediate64)
{
   agx_index imm = agx_immediate(10);
   imm.size = AGX_SIZE_64;

   struct agx_copy test_1[] = {
      {.dest = 4, .src = imm},
   };

   CASE(test_1, {
      agx_mov_imm_to(b, agx_register(4, AGX_SIZE_32), 10);
      agx_mov_imm_to(b, agx_register(6, AGX_SIZE_32), 0);
   });
}

/* Test case from Hack et al */
TEST_F(LowerParallelCopy, TwoSwaps)
{
   struct agx_copy test[] = {
      {.dest = 4, .src = agx_register(2, AGX_SIZE_32)},
      {.dest = 6, .src = agx_register(4, AGX_SIZE_32)},
      {.dest = 2, .src = agx_register(6, AGX_SIZE_32)},
      {.dest = 8, .src = agx_register(8, AGX_SIZE_32)},
   };

   CASE(test, {
      agx_swap(b, agx_register(4, AGX_SIZE_32), agx_register(2, AGX_SIZE_32));
      agx_swap(b, agx_register(6, AGX_SIZE_32), agx_register(2, AGX_SIZE_32));
   });
}

TEST_F(LowerParallelCopy, VectorizeAlignedHalfRegs)
{
   struct agx_copy test[] = {
      {.dest = 0, .src = agx_register(10, AGX_SIZE_16)},
      {.dest = 1, .src = agx_register(11, AGX_SIZE_16)},
      {.dest = 2, .src = agx_uniform(8, AGX_SIZE_16)},
      {.dest = 3, .src = agx_uniform(9, AGX_SIZE_16)},
   };

   CASE(test, {
      agx_mov_to(b, agx_register(0, AGX_SIZE_32),
                 agx_register(10, AGX_SIZE_32));
      agx_mov_to(b, agx_register(2, AGX_SIZE_32), agx_uniform(8, AGX_SIZE_32));
   });
}

TEST_F(LowerParallelCopy, StackCopies)
{
   struct agx_copy test[] = {
      {.dest = 21, .dest_mem = true, .src = agx_register(20, AGX_SIZE_16)},
      {.dest = 22, .dest_mem = true, .src = agx_register(22, AGX_SIZE_32)},
      {.dest = 0, .src = agx_memory_register(10, AGX_SIZE_16)},
      {.dest = 1, .src = agx_memory_register(11, AGX_SIZE_16)},
      {.dest = 0, .dest_mem = true, .src = agx_memory_register(12, AGX_SIZE_16)},
      {.dest = 1, .dest_mem = true, .src = agx_memory_register(13, AGX_SIZE_16)},
      {.dest = 2,
       .dest_mem = true,
       .src = agx_memory_register(804, AGX_SIZE_32)},
      {.dest = 804,
       .dest_mem = true,
       .src = agx_memory_register(2, AGX_SIZE_32)},
      {.dest = 807,
       .dest_mem = true,
       .src = agx_memory_register(808, AGX_SIZE_16)},
      {.dest = 808,
       .dest_mem = true,
       .src = agx_memory_register(807, AGX_SIZE_16)},
   };

   CASE(test, {
      /* Vectorized fill */
      agx_mov_to(b, agx_register(0, AGX_SIZE_32),
                 agx_memory_register(10, AGX_SIZE_32));

      /* Regular spills */
      agx_mov_to(b, agx_memory_register(21, AGX_SIZE_16),
                 agx_register(20, AGX_SIZE_16));
      agx_mov_to(b, agx_memory_register(22, AGX_SIZE_32),
                 agx_register(22, AGX_SIZE_32));

      /* Vectorized stack->stack copy */
      agx_mov_to(b, agx_register(2, AGX_SIZE_32),
                 agx_memory_register(12, AGX_SIZE_32));

      agx_mov_to(b, agx_memory_register(0, AGX_SIZE_32),
                 agx_register(2, AGX_SIZE_32));

      /* Stack swap: 32-bit */
      agx_index temp1 = agx_register(4, AGX_SIZE_32);
      agx_index temp2 = agx_register(6, AGX_SIZE_32);
      agx_index spilled_gpr_vec2 = agx_register(0, AGX_SIZE_32);
      spilled_gpr_vec2.channels_m1++;

      agx_mov_to(b, temp1, agx_memory_register(2, AGX_SIZE_32));
      agx_mov_to(b, temp2, agx_memory_register(804, AGX_SIZE_32));
      agx_mov_to(b, agx_memory_register(804, AGX_SIZE_32), temp1);
      agx_mov_to(b, agx_memory_register(2, AGX_SIZE_32), temp2);

      /* Stack swap: 16-bit */
      spilled_gpr_vec2.size = AGX_SIZE_16;
      temp1.size = AGX_SIZE_16;
      temp2.size = AGX_SIZE_16;

      agx_mov_to(b, temp1, agx_memory_register(807, AGX_SIZE_16));
      agx_mov_to(b, temp2, agx_memory_register(808, AGX_SIZE_16));
      agx_mov_to(b, agx_memory_register(808, AGX_SIZE_16), temp1);
      agx_mov_to(b, agx_memory_register(807, AGX_SIZE_16), temp2);
   });
}

TEST_F(LowerParallelCopy, G17PStackCopiesAvoidDivergentShuffleScratch)
{
   struct agx_copy copies[] = {
      {.dest = 0,
       .dest_mem = true,
       .src = agx_memory_register(12, AGX_SIZE_32)},
   };

   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   actual->shader->any_quad_divergent_shuffle = true;

   agx_emit_parallel_copies(actual, copies, ARRAY_SIZE(copies));
   agx_mov_to(expected, agx_register(4, AGX_SIZE_32),
              agx_memory_register(12, AGX_SIZE_32));
   agx_mov_to(expected, agx_memory_register(0, AGX_SIZE_32),
              agx_register(4, AGX_SIZE_32));

   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

#if 0
TEST_F(LowerParallelCopy, LooksLikeASwap) {
   struct agx_copy test[] = {
        { .dest = 0, .src = agx_register(2, AGX_SIZE_32) },
        { .dest = 2, .src = agx_register(0, AGX_SIZE_32) },
        { .dest = 4, .src = agx_register(2, AGX_SIZE_32) },
   };

   CASE(test, {
         agx_mov_to(b, agx_register(4, AGX_SIZE_32), agx_register(2, AGX_SIZE_32));
         agx_mov_to(b, agx_register(2, AGX_SIZE_32), agx_register(0, AGX_SIZE_32));
         agx_mov_to(b, agx_register(0, AGX_SIZE_32), agx_register(4, AGX_SIZE_32));
   });
}
#endif

TEST_F(LowerParallelCopy, EmptyCopies)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);

   agx_emit_parallel_copies(actual, NULL, 0);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

TEST_F(LowerParallelCopy, MemoryOnlyAtFileEnd)
{
   struct agx_copy copies[] = {
      {.dest = 2047, .dest_mem = true,
       .src = agx_memory_register(2045, AGX_SIZE_16)},
   };

   CASE(copies, {
      agx_mov_to(b, agx_register(2, AGX_SIZE_16),
                 agx_memory_register(2045, AGX_SIZE_16));
      agx_mov_to(b, agx_memory_register(2047, AGX_SIZE_16),
                 agx_register(2, AGX_SIZE_16));
   });
}

TEST_F(LowerParallelCopy, MemoryOnly64AtFileEnd)
{
   struct agx_copy copies[] = {
      {.dest = 2044, .dest_mem = true,
       .src = agx_memory_register(2040, AGX_SIZE_64)},
   };

   CASE(copies, {
      agx_mov_to(b, agx_register(2, AGX_SIZE_32),
                 agx_memory_register(2040, AGX_SIZE_32));
      agx_mov_to(b, agx_memory_register(2044, AGX_SIZE_32),
                 agx_register(2, AGX_SIZE_32));
      agx_mov_to(b, agx_register(2, AGX_SIZE_32),
                 agx_memory_register(2042, AGX_SIZE_32));
      agx_mov_to(b, agx_memory_register(2046, AGX_SIZE_32),
                 agx_register(2, AGX_SIZE_32));
   });
}

TEST_F(LowerParallelCopy, IndependentClassesWithSameIndices)
{
   struct agx_copy copies[] = {
      {.dest = 8, .dest_mem = true,
       .src = agx_memory_register(10, AGX_SIZE_32)},
      {.dest = 8, .src = agx_register(10, AGX_SIZE_32)},
   };

   CASE(copies, {
      agx_mov_to(b, agx_register(8, AGX_SIZE_32),
                 agx_register(10, AGX_SIZE_32));
      agx_mov_to(b, agx_register(2, AGX_SIZE_32),
                 agx_memory_register(10, AGX_SIZE_32));
      agx_mov_to(b, agx_memory_register(8, AGX_SIZE_32),
                 agx_register(2, AGX_SIZE_32));
   });
}

TEST_F(LowerParallelCopy, HighSlotMemoryAt8191)
{
   struct agx_copy copies[] = {
      {.dest = AGX_NUM_MODELED_REGS - 1,
       .dest_mem = true,
       .src = agx_memory_register(AGX_NUM_MODELED_REGS - 3, AGX_SIZE_16)},
   };

   CASE(copies, {
      agx_mov_to(b, agx_register(2, AGX_SIZE_16),
                 agx_memory_register(AGX_NUM_MODELED_REGS - 3, AGX_SIZE_16));
      agx_mov_to(b, agx_memory_register(AGX_NUM_MODELED_REGS - 1, AGX_SIZE_16),
                 agx_register(2, AGX_SIZE_16));
   });
}

TEST_F(LowerParallelCopy, HighSlotMemory64AndSwap)
{
   struct agx_copy copies[] = {
      {.dest = AGX_NUM_MODELED_REGS - 4,
       .dest_mem = true,
       .src = agx_memory_register(AGX_NUM_MODELED_REGS - 8, AGX_SIZE_64)},
      {.dest = 4096,
       .dest_mem = true,
       .src = agx_memory_register(6000, AGX_SIZE_32)},
      {.dest = 6000,
       .dest_mem = true,
       .src = agx_memory_register(4096, AGX_SIZE_32)},
   };

   CASE(copies, {
      agx_mov_to(b, agx_register(2, AGX_SIZE_32),
                 agx_memory_register(AGX_NUM_MODELED_REGS - 8, AGX_SIZE_32));
      agx_mov_to(b, agx_memory_register(AGX_NUM_MODELED_REGS - 4, AGX_SIZE_32),
                 agx_register(2, AGX_SIZE_32));
      agx_mov_to(b, agx_register(2, AGX_SIZE_32),
                 agx_memory_register(AGX_NUM_MODELED_REGS - 6, AGX_SIZE_32));
      agx_mov_to(b, agx_memory_register(AGX_NUM_MODELED_REGS - 2, AGX_SIZE_32),
                 agx_register(2, AGX_SIZE_32));

      agx_index temp1 = agx_register(4, AGX_SIZE_32);
      agx_index temp2 = agx_register(6, AGX_SIZE_32);
      agx_mov_to(b, temp1, agx_memory_register(4096, AGX_SIZE_32));
      agx_mov_to(b, temp2, agx_memory_register(6000, AGX_SIZE_32));
      agx_mov_to(b, agx_memory_register(6000, AGX_SIZE_32), temp1);
      agx_mov_to(b, agx_memory_register(4096, AGX_SIZE_32), temp2);
   });
}

/* Memory is written from a register only, so an immediate or uniform copied to
 * memory (a spilled phi's constant source) goes through the scratch register
 * of memory-memory copies. A 64-bit uniform is copied a word at a time.
 */
TEST_F(LowerParallelCopy, ImmediateAndUniformToMemory)
{
   agx_index imm = agx_immediate(7);
   imm.size = AGX_SIZE_32;

   struct agx_copy test[] = {
      {.dest = 4, .dest_mem = true, .src = imm},
      {.dest = 8, .dest_mem = true, .src = agx_uniform(6, AGX_SIZE_32)},
      {.dest = 12, .dest_mem = true, .src = agx_uniform(20, AGX_SIZE_64)},
      {.dest = 17, .dest_mem = true, .src = agx_uniform(9, AGX_SIZE_16)},
   };

   CASE(test, {
      agx_index scratch = agx_register(2, AGX_SIZE_32);

      agx_mov_imm_to(b, scratch, 7);
      agx_mov_to(b, agx_memory_register(4, AGX_SIZE_32), scratch);
      agx_mov_to(b, scratch, agx_uniform(6, AGX_SIZE_32));
      agx_mov_to(b, agx_memory_register(8, AGX_SIZE_32), scratch);
      agx_mov_to(b, scratch, agx_uniform(20, AGX_SIZE_32));
      agx_mov_to(b, agx_memory_register(12, AGX_SIZE_32), scratch);
      agx_mov_to(b, scratch, agx_uniform(22, AGX_SIZE_32));
      agx_mov_to(b, agx_memory_register(14, AGX_SIZE_32), scratch);
      agx_mov_to(b, agx_register(2, AGX_SIZE_16), agx_uniform(9, AGX_SIZE_16));
      agx_mov_to(b, agx_memory_register(17, AGX_SIZE_16),
                 agx_register(2, AGX_SIZE_16));
   });
}
